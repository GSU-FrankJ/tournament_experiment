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

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def beta_mode_normalized(alpha, beta) -> np.ndarray:
    """Normalized Beta mode with a mean fallback for non-interior shapes.

    Returns ``(alpha-1)/(alpha+beta-2)`` elementwise where ``alpha > 1`` AND
    ``beta > 1`` (the interior unimodal case), else the Beta mean
    ``alpha/(alpha+beta)`` (the mode sits at a boundary and is not a useful
    point estimate). Clamped to ``(0, 1)``. The MEAN remains the primary
    reported extraction (repo invariant); this supports the mean-vs-mode
    diagnostic only.

    Args:
        alpha: Beta alpha parameter(s), scalar or array.
        beta: Beta beta parameter(s), scalar or array.

    Returns:
        Normalized point estimate(s) in ``(0, 1)``, broadcast to the input shape.
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    mean = alpha / (alpha + beta)
    interior = (alpha > 1.0) & (beta > 1.0)
    denom = np.where(interior, alpha + beta - 2.0, 1.0)
    mode = np.where(interior, (alpha - 1.0) / denom, mean)
    return np.clip(mode, 1e-6, 1.0 - 1e-6)


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

    def add_np(
        self,
        state: "np.ndarray",
        action_norm: float,
        logp: float,
        reward: float,
        value: float,
    ) -> None:
        """Append one transition from numpy/scalar values (vectorized rollout).

        Args:
            state: Observation row, shape ``(state_dim,)``.
            action_norm: Normalized action scalar in [0, 1].
            logp: Log-prob scalar.
            reward: Sampled stage reward scalar.
            value: Critic value scalar.

        Raises:
            RuntimeError: If no trajectory is open.
        """
        if self._current is None:
            raise RuntimeError("add_np called with no open trajectory")
        self._current.states.append(torch.as_tensor(state, dtype=torch.float32))
        self._current.actions_norm.append(torch.as_tensor(action_norm, dtype=torch.float32))
        self._current.logps.append(torch.as_tensor(logp, dtype=torch.float32))
        self._current.rewards.append(torch.as_tensor(reward, dtype=torch.float32))
        self._current.values.append(torch.as_tensor(value, dtype=torch.float32))

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


# ---------------------------------------------------------------------------
# Actor-critic (Beta policy, mean/concentration parametrization)
# ---------------------------------------------------------------------------

class MultiStageActorCritic(nn.Module):
    """Shared-trunk Beta actor + value critic over the 2-D state [t/T, d_norm].

    Uses the mean/concentration parametrization so the reported effort
    (Beta MEAN, the repo invariant) is read directly off ``mean``:

        mean = sigmoid(mean_head(h)) in (0, 1)
        conc = softplus(conc_head(h)) * conc_scale + conc_min   (opt. clamped)
        alpha = mean * conc,  beta = (1 - mean) * conc

    No theory-align concentration ramp or opponent-lag head: those were
    one-stage stabilizers and are not part of the multi-stage plan.
    """

    def __init__(
        self,
        state_dim: int = 2,
        hidden: int = 64,
        conc_min: float = 1.0,
        conc_scale: float = 1.0,
        conc_max: Optional[float] = None,
    ):
        """Initialize the network.

        Args:
            state_dim: Observation dimension (2 for [t/T, d/(q sqrt t)]).
            hidden: Hidden width of the shared trunk.
            conc_min: Floor on the Beta concentration alpha+beta.
            conc_scale: Multiplier on the softplus concentration.
            conc_max: Optional cap on the concentration.
        """
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.conc_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)
        self.conc_min = float(conc_min)
        self.conc_scale = float(conc_scale)
        self.conc_max = None if conc_max is None else float(conc_max)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Beta ``(alpha, beta)`` and the state value.

        Args:
            x: State batch, shape ``(B, state_dim)``.

        Returns:
            ``(alpha, beta, value)`` with ``alpha``/``beta`` shape ``(B, 1)``
            and ``value`` shape ``(B,)``.
        """
        h = self.shared(x)
        mean = torch.sigmoid(self.mean_head(h))
        conc = F.softplus(self.conc_head(h)) * max(self.conc_scale, 1e-8) + self.conc_min
        if self.conc_max is not None:
            conc = torch.clamp(conc, max=self.conc_max)
        alpha = mean * conc
        beta = (1.0 - mean) * conc
        value = self.value_head(h).squeeze(-1)
        return alpha, beta, value

    def dist(self, x: torch.Tensor) -> Tuple[torch.distributions.Beta, torch.Tensor]:
        """Return the Beta policy distribution and the state value.

        Args:
            x: State batch, shape ``(B, state_dim)``.

        Returns:
            ``(Beta(alpha, beta), value)``.
        """
        alpha, beta, value = self.forward(x)
        return torch.distributions.Beta(alpha, beta), value


@dataclass
class MultiStagePPOConfig:
    """Hyperparameters for the multi-stage PPO trainer.

    Defaults follow the plan's training spec. Note gamma = gae_lambda = 1.0
    (finite-horizon economic payoff, plan section 3.4) -- deliberately NOT
    the one-stage agent's 0.99/0.95, which must never leak in.
    """

    state_dim: int = 2
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 10
    minibatch_size: int = 256
    conc_min: float = 1.0
    conc_scale: float = 1.0
    conc_max: Optional[float] = None
    seed: int = 42


class MultiStagePPO:
    """Symmetric self-play PPO candidate generator for the multi-stage game.

    Both players share this single policy; player j simply observes -d. The
    trainer is the Claim-B candidate GENERATOR: the DP verifier certifies its
    output. Reported effort is the Beta MEAN (repo invariant).
    """

    def __init__(
        self,
        effort_bounds: Tuple[float, float],
        cfg: MultiStagePPOConfig = MultiStagePPOConfig(),
        device: Optional[str] = None,
    ):
        """Construct the trainer.

        Args:
            effort_bounds: ``(low, high)`` effort range; normalized Beta
                actions in [0, 1] map to ``low + a * (high - low)``.
            cfg: Hyperparameters.
            device: Torch device string; auto-selects cuda when available.
        """
        self.cfg = cfg
        self.low, self.high = float(effort_bounds[0]), float(effort_bounds[1])
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.net = MultiStageActorCritic(
            state_dim=cfg.state_dim,
            hidden=cfg.hidden,
            conc_min=cfg.conc_min,
            conc_scale=cfg.conc_scale,
            conc_max=cfg.conc_max,
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.buffer = MultiStageRolloutBuffer()

    def _state_tensor(self, state) -> torch.Tensor:
        """Coerce a state to a ``(1, state_dim)`` float tensor on the device."""
        t = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
        t = t.to(self.device, dtype=torch.float32)
        return t.view(1, -1)

    def act(self, state) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        """Sample an action for one state.

        Args:
            state: Observation (length ``state_dim`` tensor or sequence).

        Returns:
            ``(action_norm[1], effort, logp[], value[])`` — the normalized
            action tensor, the scaled effort (float), the log-prob, and the
            critic value (both 0-d tensors).
        """
        s = self._state_tensor(state)
        with torch.no_grad():
            dist, value = self.net.dist(s)
            a = dist.sample()                       # (1, 1)
            logp = dist.log_prob(a).squeeze()       # ()
        a_norm = a.squeeze(0)                        # (1,)
        effort = self.low + float(a_norm.item()) * (self.high - self.low)
        return a_norm, effort, logp, value.squeeze(0)

    @torch.no_grad()
    def act_batch(
        self, states: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample actions for a batch of states (vectorized rollout).

        Args:
            states: State batch, shape ``(N, state_dim)``.

        Returns:
            ``(action_norm[N], effort[N], logp[N], value[N])`` as numpy arrays.
        """
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        dist, value = self.net.dist(s)             # value already shape (N,)
        a = dist.sample()                          # (N, 1)
        logp = dist.log_prob(a).squeeze(-1)        # (N,)
        a_norm = a.squeeze(-1).cpu().numpy()       # (N,)
        effort = self.low + a_norm * (self.high - self.low)
        return a_norm, effort, logp.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def mean_effort(self, state) -> float:
        """Deterministic reported effort = Beta mean mapped to effort units.

        Args:
            state: Observation.

        Returns:
            Effort (float) in ``[low, high]``.
        """
        s = self._state_tensor(state)
        dist, _ = self.net.dist(s)
        a_mean = float(dist.mean.clamp(1e-6, 1 - 1e-6).item())
        return self.low + a_mean * (self.high - self.low)

    def _stage_states(self, t: int, d: np.ndarray, T: int, q: float) -> torch.Tensor:
        """Normalized states [t/T, d/(q sqrt t)] for a stage-t gap array."""
        d = np.asarray(d, dtype=np.float32)
        t_norm = np.full_like(d, t / T)
        d_norm = d / (q * np.sqrt(max(t, 1)))
        return torch.tensor(np.stack([t_norm, d_norm], axis=-1), dtype=torch.float32,
                            device=self.device)

    @torch.no_grad()
    def beta_params(self, t: int, d: np.ndarray, T: int, q: float) -> Tuple[np.ndarray, np.ndarray]:
        """Raw Beta ``(alpha, beta)`` of the policy at stage t over gaps d.

        Feeds the mean-vs-mode diagnostic and the alpha/beta checkpoint dump
        (so the extraction question can be revisited post-hoc without a re-run).

        Args:
            t: Stage (1-indexed).
            d: Score gaps (array).
            T: Horizon.
            q: Noise half-width.

        Returns:
            ``(alpha, beta)`` arrays, same shape as ``d``.
        """
        states = self._stage_states(t, d, T, q)
        alpha, beta, _ = self.net.forward(states)
        return alpha.squeeze(-1).cpu().numpy(), beta.squeeze(-1).cpu().numpy()

    @torch.no_grad()
    def effort_function(
        self, t: int, d: np.ndarray, T: int, q: float, extraction: str = "mean"
    ) -> np.ndarray:
        """Vectorized learned effort function e_hat_t(d) for the verifier.

        Builds the normalized state [t/T, d/(q sqrt t)] for each gap and returns
        the extracted effort. ``extraction="mean"`` (default, the repo invariant
        and the object passed to ``utils.dp_verifier.verify``) uses the Beta
        mean; ``extraction="mode"`` uses the mean-fallback Beta mode for the
        diagnostic only.

        Args:
            t: Stage (1-indexed).
            d: Score gaps (array).
            T: Horizon.
            q: Noise half-width (for the state normalization).
            extraction: ``"mean"`` (default) or ``"mode"``.

        Returns:
            Effort array, same shape as ``d``.
        """
        states = self._stage_states(t, d, T, q)
        if extraction == "mode":
            alpha, beta, _ = self.net.forward(states)
            a_pt = beta_mode_normalized(alpha.squeeze(-1).cpu().numpy(),
                                        beta.squeeze(-1).cpu().numpy())
        elif extraction == "mean":
            dist, _ = self.net.dist(states)
            a_pt = dist.mean.squeeze(-1).clamp(1e-6, 1 - 1e-6).cpu().numpy()
        else:
            raise ValueError(f"unknown extraction {extraction!r} (use 'mean' or 'mode')")
        return self.low + a_pt * (self.high - self.low)

    def save(self, path: str) -> None:
        """Persist policy/critic weights + config + effort bounds to a .pt file.

        Args:
            path: Destination checkpoint path (parent dirs created).
        """
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {"state_dict": self.net.state_dict(), "cfg": asdict(self.cfg),
             "effort_bounds": [self.low, self.high]},
            path,
        )

    def load(self, path: str) -> Dict:
        """Load weights saved by :meth:`save` onto this agent's device.

        Args:
            path: Checkpoint path written by :meth:`save`.

        Returns:
            The raw checkpoint dict (state_dict / cfg / effort_bounds).
        """
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        return ckpt

    def evaluate_actions(
        self, states: torch.Tensor, actions_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probs, entropy, and values for a batch (PPO inner loop).

        Args:
            states: ``(N, state_dim)`` states.
            actions_norm: ``(N, 1)`` normalized actions in [0, 1].

        Returns:
            ``(logp[N], entropy[], values[N])``.
        """
        dist, values = self.net.dist(states)
        a_safe = actions_norm.clamp(1e-6, 1 - 1e-6)
        logp = dist.log_prob(a_safe).squeeze(-1)
        entropy = dist.entropy().mean()
        return logp, entropy, values.view(-1)

    def update(self) -> Dict[str, float]:
        """Run a PPO update from the rollout buffer, then clear it.

        Advantages/returns come from the trajectory-aware GAE
        (``buffer.compute``), so storage order is irrelevant. Standard
        clipped-surrogate + value-MSE + entropy objective with advantage
        normalization and minibatch epochs.

        Returns:
            Diagnostics dict (losses, entropy, approx_kl, clip_frac,
            grad_norm, transitions).

        Raises:
            RuntimeError: If the buffer is empty.
        """
        if len(self.buffer) == 0:
            raise RuntimeError("update called with an empty buffer")
        data = self.buffer.compute(self.cfg.gamma, self.cfg.gae_lambda)
        states = data["states"].to(self.device)
        actions_norm = data["actions_norm"].view(-1, 1).to(self.device)
        old_logp = data["logp"].to(self.device)
        returns = data["returns"].to(self.device)
        advantages = data["advantages"].to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        n = states.size(0)
        clip_eps = float(self.cfg.clip_eps)
        idx = np.arange(n)
        pl, vl, ent, kl, cf, gn = [], [], [], [], [], []

        for _ in range(self.cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = idx[start:start + self.cfg.minibatch_size]
                mb_states = states[mb]
                mb_actions = actions_norm[mb]
                mb_adv = advantages[mb]
                mb_ret = returns[mb]
                mb_old_logp = old_logp[mb]

                logp, entropy, values = self.evaluate_actions(mb_states, mb_actions)
                log_ratio = logp - mb_old_logp
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_ret)
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.opt.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                pl.append(float(policy_loss.item()))
                vl.append(float(value_loss.item()))
                ent.append(float(entropy.item()))
                kl.append(float((mb_old_logp - logp).mean().item()))
                cf.append(float((ratio.gt(1 + clip_eps) | ratio.lt(1 - clip_eps)).float().mean().item()))
                gn.append(float(grad_norm))

        self.buffer.reset()
        return {
            "policy_loss": float(np.mean(pl)),
            "value_loss": float(np.mean(vl)),
            "entropy": float(np.mean(ent)),
            "approx_kl": float(np.mean(kl)),
            "clip_frac": float(np.mean(cf)),
            "grad_norm": float(np.mean(gn)),
            "transitions": int(n),
        }
