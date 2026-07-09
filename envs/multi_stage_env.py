"""Multi-stage dynamic Lazear-Rosen tournament environment (terminal reward only).

Implements the game in ``docs/Experiments Plan_Multi-stage.md`` section 1,
with the canonical parameters/conventions of
``config/multi_stage_two_players.py`` (repo convention c(e) = k e^2).

This is a from-scratch rewrite; ``envs/two_stage_env.py`` implements a
DIFFERENT game (per-stage prize flow, logit win model, expected-value
rewards, no gap state) and is kept only for historical reference.

Model (two players, horizon T, N=2):
    - Stage output:  y_{i,t} = e_{i,t} + eps_{i,t},  eps ~ U(-q, q) i.i.d.
    - Markov state:  (t, d_t), where d_t = S_{i,t-1} - S_{j,t-1} is player i's
      cumulative score gap; player j observes -d_t.
    - Transition:    d_{t+1} = d_t + (e_i - e_j) + (eps_i - eps_j).
    - Terminal:      only after stage T. Winner (final gap > 0) gets w_h,
      loser w_l, tie broken uniformly (measure zero under continuous noise).
    - Payoff:        U_i = E[ R_i(d_{T+1}) - sum_t k e_{i,t}^2 ].

REPO INVARIANT (sampled training rewards only): ``step`` returns SAMPLED
one-step outcomes. Intermediate stages return reward = -k e^2 (cost only);
the terminal stage realizes the winner from the ACCUMULATED sampled gap and
pays the sampled prize. No closed-form win probability or expected payoff
ever enters a reward returned by ``step``. Closed-form helpers live in
``utils/theory_multistage`` and are evaluation-only.

Exploring starts (owner decision 2026-07-09): episodes may reset from a
random (t0, d0) so off-path states receive gradient signal, supporting the
full approximate-MPE claim. ``(t, d)`` is a sufficient Markov statistic, so
starting mid-tournament from a given gap is exactly consistent with the
equilibrium object; d0 stands in for the unsimulated stages 1..t0-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class StepResult:
    """Structured return of :meth:`MultiStageEnv.step`.

    Attributes:
        obs: Per-player normalized observations for the NEXT state
            (both zero vectors once the episode is done).
        rewards: Per-player sampled rewards for this stage.
        costs: Per-player effort costs k e^2 for this stage.
        done: Whether the tournament has finished (stage T completed).
        info: Diagnostics (raw state, efforts, noises, realized gap, winner).
    """

    obs: Tuple[torch.Tensor, torch.Tensor]
    rewards: torch.Tensor
    costs: torch.Tensor
    done: bool
    info: Dict[str, Any]


class MultiStageEnv:
    """Simultaneous-move two-player multi-stage tournament (one stage per step)."""

    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """Construct the environment from a canonical config dict.

        Args:
            config: Parameters; must include ``w_h``, ``w_l``, ``k``, ``q``,
                ``T``, ``effort_range``. Exploring-starts knobs
                (``exploring_starts``, ``es_on_path_fraction``,
                ``es_d_range_factor``, ``es_stage_distribution``) are read
                with sensible defaults if absent.
            seed: RNG seed (overrides ``config['seed']`` when given).
        """
        self.w_h = float(config["w_h"])
        self.w_l = float(config["w_l"])
        self.k = float(config["k"])
        self.q = float(config["q"])
        self.T = int(config["T"])
        if self.T < 1:
            raise ValueError(f"T must be >= 1, got {self.T}")
        lo, hi = config["effort_range"]
        self.effort_low = float(lo)
        self.effort_high = float(hi)
        self.num_players = int(config.get("num_players", 2))
        if self.num_players != 2:
            raise ValueError("MultiStageEnv currently supports exactly 2 players")

        # Exploring-starts settings
        self.exploring_starts = bool(config.get("exploring_starts", False))
        self.es_on_path_fraction = float(config.get("es_on_path_fraction", 0.5))
        self.es_d_range_factor = float(config.get("es_d_range_factor", 1.0))
        self.es_stage_distribution = str(config.get("es_stage_distribution", "uniform"))

        seed_val = int(config.get("seed", 42)) if seed is None else int(seed)
        self.seed = seed_val
        self.rng = np.random.default_rng(seed_val)

        # Episode state
        self.current_stage: int = 1
        self.gap: float = 0.0  # player-0 perspective: d = S0 - S1
        self._done: bool = True

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _obs_vector(self, t: int, d: float) -> torch.Tensor:
        """Normalized observation [t/T, d/(q*sqrt(t))] for a player at gap d.

        Args:
            t: Current stage (1-indexed).
            d: This player's cumulative score gap.

        Returns:
            Length-2 float tensor.
        """
        d_norm = d / (self.q * np.sqrt(max(t, 1)))
        return torch.tensor([t / self.T, d_norm], dtype=torch.float32)

    def observations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Current per-player observations (player 0 at +d, player 1 at -d).

        Returns:
            Tuple of length-2 tensors ``(obs_p0, obs_p1)``.
        """
        return (
            self._obs_vector(self.current_stage, self.gap),
            self._obs_vector(self.current_stage, -self.gap),
        )

    # ------------------------------------------------------------------
    # Reset / exploring starts
    # ------------------------------------------------------------------

    def reset(
        self, t0: int = 1, d0: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset to a given Markov state (t0, d0).

        Args:
            t0: Starting stage in ``{1, ..., T}``.
            d0: Player-0 cumulative score gap entering stage ``t0``.

        Returns:
            Per-player observations for the start state.

        Raises:
            ValueError: If ``t0`` is out of range.
        """
        if not (1 <= t0 <= self.T):
            raise ValueError(f"t0 must be in [1, {self.T}], got {t0}")
        self.current_stage = int(t0)
        self.gap = float(d0)
        self._done = False
        return self.observations()

    def sample_exploring_start(self) -> Tuple[int, float]:
        """Draw a training start (t0, d0) per the exploring-starts config.

        With probability ``es_on_path_fraction`` returns the on-path root
        (1, 0.0). Otherwise draws a stage t0 (uniform by default) and a gap
        d0 ~ Uniform(-R, R) with R = es_d_range_factor * 2q * sqrt(t0-1)
        (the O(sqrt) scale of the accumulated shock gap), so t0=1 always
        yields d0=0.

        Returns:
            ``(t0, d0)``.
        """
        if not self.exploring_starts or self.rng.random() < self.es_on_path_fraction:
            return 1, 0.0
        if self.es_stage_distribution == "uniform":
            t0 = int(self.rng.integers(1, self.T + 1))
        else:
            raise ValueError(f"Unknown es_stage_distribution: {self.es_stage_distribution}")
        r = self.es_d_range_factor * 2.0 * self.q * np.sqrt(max(t0 - 1, 0))
        d0 = float(self.rng.uniform(-r, r)) if r > 0 else 0.0
        return t0, d0

    def reset_exploring(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset to a freshly sampled exploring start.

        Returns:
            Per-player observations for the sampled start state.
        """
        t0, d0 = self.sample_exploring_start()
        return self.reset(t0, d0)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _clip(self, e: float) -> float:
        return float(np.clip(e, self.effort_low, self.effort_high))

    def draw_shock_diff(self, size: int = 1) -> np.ndarray:
        """Draw xi = eps_i - eps_j samples via two Uniform(-q, q) draws each.

        Sampling the two uniforms (rather than the triangular directly)
        keeps the env's noise identical to the per-player output model.

        Args:
            size: Number of samples.

        Returns:
            Array of shock differences, each in ``[-2q, 2q]``.
        """
        eps_i = self.rng.uniform(-self.q, self.q, size=size)
        eps_j = self.rng.uniform(-self.q, self.q, size=size)
        return eps_i - eps_j

    def step(
        self,
        efforts: Tuple[Any, Any],
        noise: Optional[Tuple[float, float]] = None,
    ) -> StepResult:
        """Advance one stage with both players' efforts and sampled outcomes.

        Args:
            efforts: ``(e0, e1)`` as floats or 0-d/1-elem tensors. Clipped to
                the effort bounds.
            noise: Optional pre-drawn ``(eps0, eps1)`` for common-random-number
                evaluation; drawn from U(-q, q) when ``None``.

        Returns:
            A :class:`StepResult`.

        Raises:
            RuntimeError: If called after the episode is already done.
        """
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")

        e0 = self._clip(efforts[0].item() if hasattr(efforts[0], "item") else efforts[0])
        e1 = self._clip(efforts[1].item() if hasattr(efforts[1], "item") else efforts[1])

        if noise is None:
            eps0 = float(self.rng.uniform(-self.q, self.q))
            eps1 = float(self.rng.uniform(-self.q, self.q))
        else:
            eps0, eps1 = float(noise[0]), float(noise[1])

        d_prev = self.gap
        d_next = d_prev + (e0 - e1) + (eps0 - eps1)

        cost0 = self.k * e0 * e0
        cost1 = self.k * e1 * e1
        costs = torch.tensor([cost0, cost1], dtype=torch.float32)

        is_terminal = self.current_stage >= self.T
        winner: Optional[int] = None
        if is_terminal:
            # Winner from the REALIZED accumulated final gap (sampled outcome).
            if d_next > 0.0:
                winner = 0
            elif d_next < 0.0:
                winner = 1
            else:
                winner = int(self.rng.integers(0, 2))
            prize0 = self.w_h if winner == 0 else self.w_l
            prize1 = self.w_h if winner == 1 else self.w_l
            rewards = torch.tensor([prize0 - cost0, prize1 - cost1], dtype=torch.float32)
        else:
            rewards = torch.tensor([-cost0, -cost1], dtype=torch.float32)

        info = {
            "stage": self.current_stage,
            "gap_in": d_prev,
            "gap_out": d_next,
            "efforts": (e0, e1),
            "noises": (eps0, eps1),
            "terminal": is_terminal,
            "winner": winner,
        }

        # Advance state
        self.gap = d_next
        self.current_stage += 1
        self._done = is_terminal

        if is_terminal:
            zero = torch.zeros(2, dtype=torch.float32)
            obs = (zero.clone(), zero.clone())
        else:
            obs = self.observations()

        return StepResult(obs=obs, rewards=rewards, costs=costs, done=is_terminal, info=info)

    # ------------------------------------------------------------------
    # Evaluation-only helpers (NOT training rewards)
    # ------------------------------------------------------------------

    def rollout_policy(
        self,
        policy_fn,
        n_episodes: int,
        start: Tuple[int, float] = (1, 0.0),
    ) -> Dict[str, np.ndarray]:
        """Simulate episodes under a shared symmetric policy (evaluation only).

        Both players follow ``policy_fn(t, d) -> effort`` (player 1 queried at
        gap -d). Used by the env self-check to reproduce U_eq and win rates;
        never used to produce training rewards.

        Args:
            policy_fn: Callable mapping ``(stage, own_gap)`` to effort.
            n_episodes: Number of episodes to simulate.
            start: Fixed ``(t0, d0)`` start for every episode.

        Returns:
            Dict of per-episode arrays: ``payoff0``, ``payoff1``,
            ``final_gap``, ``winner``, and stagewise ``effort0``/``effort1``
            (shape ``[n_episodes, T - t0 + 1]``).
        """
        t0, d0 = start
        payoff0 = np.zeros(n_episodes)
        payoff1 = np.zeros(n_episodes)
        final_gap = np.zeros(n_episodes)
        winners = np.zeros(n_episodes, dtype=int)
        n_stages = self.T - t0 + 1
        eff0 = np.zeros((n_episodes, n_stages))
        eff1 = np.zeros((n_episodes, n_stages))

        for ep in range(n_episodes):
            self.reset(t0, d0)
            g0 = 0.0
            g1 = 0.0
            for s in range(n_stages):
                t = self.current_stage
                d = self.gap
                e0 = policy_fn(t, d)
                e1 = policy_fn(t, -d)
                eff0[ep, s] = e0
                eff1[ep, s] = e1
                res = self.step((e0, e1))
                g0 += res.rewards[0].item()
                g1 += res.rewards[1].item()
            payoff0[ep] = g0
            payoff1[ep] = g1
            final_gap[ep] = self.gap
            winners[ep] = res.info["winner"]

        return {
            "payoff0": payoff0,
            "payoff1": payoff1,
            "final_gap": final_gap,
            "winner": winners,
            "effort0": eff0,
            "effort1": eff1,
        }
