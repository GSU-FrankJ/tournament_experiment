"""Multi-stage PPO trainer for the dynamic tournament (Claim-B framing).

TEL-PPO generates candidate state-dependent effort functions; the independent
DP verifier (``utils/dp_verifier``) certifies them. This module is the
candidate GENERATOR only.

Phase 04 is built incrementally. This first commit lands the piece that must
be correct before any GPU time is spent: a **trajectory-aware GAE** and a
rollout buffer that stores each player's episode as a separate trajectory.

Why not reuse the one-stage agent's GAE: ``PPOTwoPlayersBandit._compute_gae``
treats the storage as one flat sequence and chains ``next_value`` across
consecutive indices, resetting only on ``done``. That is correct for
single-step bandit (every transition is terminal) and for a buffer of
CONTIGUOUS trajectories, but it silently MISBOOTSTRAPS if the multi-stage
rollout interleaves the two players' transitions (p0_s1, p1_s1, p0_s2, ...),
because p0's stage-1 bootstrap would read p1's stage-1 value. The
trajectory-aware GAE here is ordering-independent: it computes advantages per
trajectory with a terminal bootstrap of zero (finite horizon), so the caller
cannot introduce this bug by storing in the wrong order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch


def compute_gae_single(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GAE for ONE finite-horizon trajectory (terminal bootstrap = 0).

    The final transition is terminal, so there is no next-state value to
    bootstrap from; the recursion starts with ``next_value = 0``. With
    ``gamma = gae_lambda = 1`` (the plan's main spec) the advantage reduces
    to the Monte Carlo advantage ``A_t = (sum_{s>=t} r_s) - V_t`` and the
    return to the realized finite-horizon payoff ``sum_{s>=t} r_s``.

    Args:
        rewards: 1-D tensor of stage rewards ``[r_0, ..., r_{L-1}]``.
        values: 1-D tensor of critic values ``[V_0, ..., V_{L-1}]``.
        gamma: Discount (1.0 for the finite-horizon economic payoff).
        gae_lambda: GAE lambda.

    Returns:
        ``(advantages, returns)``, each 1-D of length ``L``.

    Raises:
        ValueError: If ``rewards`` and ``values`` differ in length or are
            not 1-D.
    """
    if rewards.dim() != 1 or values.dim() != 1:
        raise ValueError("rewards and values must be 1-D")
    if rewards.numel() != values.numel():
        raise ValueError(
            f"length mismatch: rewards {rewards.numel()} vs values {values.numel()}"
        )
    length = rewards.numel()
    advantages = torch.zeros(length, dtype=values.dtype, device=values.device)
    lastgae = torch.zeros((), dtype=values.dtype, device=values.device)
    next_value = torch.zeros((), dtype=values.dtype, device=values.device)
    for t in reversed(range(length)):
        delta = rewards[t] + gamma * next_value - values[t]
        lastgae = delta + gamma * gae_lambda * lastgae
        advantages[t] = lastgae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def compute_gae_trajectories(
    rewards_list: List[torch.Tensor],
    values_list: List[torch.Tensor],
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GAE over a list of independent trajectories, concatenated in input order.

    Each trajectory is treated as a finite-horizon episode with a zero
    terminal bootstrap. The result is ordering-independent PER TRAJECTORY:
    permuting ``rewards_list``/``values_list`` permutes the output blocks but
    never mixes bootstrap values across trajectories.

    Args:
        rewards_list: List of 1-D reward tensors, one per trajectory.
        values_list: List of 1-D value tensors, one per trajectory.
        gamma: Discount.
        gae_lambda: GAE lambda.

    Returns:
        ``(advantages, returns)`` concatenated in the same order as the input
        trajectories.

    Raises:
        ValueError: If the two lists differ in length.
    """
    if len(rewards_list) != len(values_list):
        raise ValueError(
            f"trajectory count mismatch: {len(rewards_list)} vs {len(values_list)}"
        )
    adv_blocks: List[torch.Tensor] = []
    ret_blocks: List[torch.Tensor] = []
    for r, v in zip(rewards_list, values_list):
        a, ret = compute_gae_single(r, v, gamma, gae_lambda)
        adv_blocks.append(a)
        ret_blocks.append(ret)
    if not adv_blocks:
        empty = torch.zeros(0)
        return empty, empty.clone()
    return torch.cat(adv_blocks), torch.cat(ret_blocks)


@dataclass
class _Trajectory:
    """A single player's episode: parallel per-stage lists."""

    states: List[torch.Tensor] = field(default_factory=list)
    actions_norm: List[torch.Tensor] = field(default_factory=list)
    logps: List[torch.Tensor] = field(default_factory=list)
    rewards: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)


class MultiStageRolloutBuffer:
    """Stores per-player trajectories and computes trajectory-aware GAE.

    Usage per episode-player:
        buf.start_trajectory()
        for each stage: buf.add(state, action_norm, logp, reward, value)
        buf.end_trajectory()

    ``compute(gamma, lam)`` returns the flattened tensors ready for a PPO
    update, with advantages/returns computed correctly regardless of the
    order in which trajectories were added.
    """

    def __init__(self) -> None:
        self._trajectories: List[_Trajectory] = []
        self._current: _Trajectory | None = None

    def start_trajectory(self) -> None:
        """Begin a new trajectory.

        Raises:
            RuntimeError: If a trajectory is already open.
        """
        if self._current is not None:
            raise RuntimeError("start_trajectory called with an open trajectory")
        self._current = _Trajectory()

    def add(
        self,
        state: torch.Tensor,
        action_norm: torch.Tensor,
        logp: torch.Tensor,
        reward: float | torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Append one stage transition to the open trajectory.

        Args:
            state: Observation tensor.
            action_norm: Normalized action in [0, 1].
            logp: Log-prob of the action.
            reward: Sampled stage reward.
            value: Critic value estimate.

        Raises:
            RuntimeError: If no trajectory is open.
        """
        if self._current is None:
            raise RuntimeError("add called with no open trajectory")
        self._current.states.append(state.squeeze(0).detach().cpu())
        self._current.actions_norm.append(action_norm.squeeze().detach().cpu())
        self._current.logps.append(logp.squeeze().detach().cpu())
        self._current.rewards.append(torch.as_tensor(reward, dtype=torch.float32))
        self._current.values.append(value.squeeze().detach().cpu())

    def end_trajectory(self) -> None:
        """Close the open trajectory and store it.

        Raises:
            RuntimeError: If no trajectory is open.
        """
        if self._current is None:
            raise RuntimeError("end_trajectory called with no open trajectory")
        if self._current.rewards:  # skip empty trajectories
            self._trajectories.append(self._current)
        self._current = None

    def __len__(self) -> int:
        return len(self._trajectories)

    @property
    def num_transitions(self) -> int:
        """Total stored transitions across all trajectories."""
        return sum(len(t.rewards) for t in self._trajectories)

    def reset(self) -> None:
        """Drop all stored trajectories."""
        self._trajectories = []
        self._current = None

    def compute(
        self, gamma: float, gae_lambda: float
    ) -> dict[str, torch.Tensor]:
        """Flatten stored trajectories and compute trajectory-aware GAE.

        Args:
            gamma: Discount.
            gae_lambda: GAE lambda.

        Returns:
            Dict with keys ``states``, ``actions_norm``, ``logp``,
            ``rewards``, ``values``, ``advantages``, ``returns`` — all
            concatenated across trajectories in insertion order.

        Raises:
            RuntimeError: If a trajectory is still open.
        """
        if self._current is not None:
            raise RuntimeError("compute called with an open trajectory")
        rewards_list = [torch.stack(t.rewards) for t in self._trajectories]
        values_list = [torch.stack(t.values) for t in self._trajectories]
        advantages, returns = compute_gae_trajectories(
            rewards_list, values_list, gamma, gae_lambda
        )
        states = torch.stack([s for t in self._trajectories for s in t.states])
        actions = torch.stack([a for t in self._trajectories for a in t.actions_norm])
        logps = torch.stack([lp for t in self._trajectories for lp in t.logps])
        return {
            "states": states,
            "actions_norm": actions,
            "logp": logps,
            "rewards": torch.cat(rewards_list),
            "values": torch.cat(values_list),
            "advantages": advantages,
            "returns": returns,
        }
