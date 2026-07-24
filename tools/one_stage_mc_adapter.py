#!/usr/bin/env python3
"""Degenerate point-policy adapter for the SHIPPED legacy MC exploitability estimator.

`run/run_two_players.py:eval_exploitability` consumes a *policy object* (it samples
the Beta for both players) and cannot evaluate a fixed deterministic effort. This
adapter exposes exactly the four attributes that estimator touches, so the shipped
code path runs UNMODIFIED on a deterministic profile:

  agent.net.parameters()                      -> device lookup      (:201, :290)
  agent.state_from_params(q, k, w_h, w_l)     -> state tensor       (:212)
  agent.dist(state) -> (dist, value)          -> policy             (:214)
  dist.sample((M,), generator=...)            -> normalized actions (:216)

Shape contract is matched to the real agent (verified empirically): the real
`ActorCritic` yields a Beta with batch_shape (1,1), so `dist.sample((M,))` has
shape (M,1,1) and `_sample_policy_efforts` returns (M,1). The stub reproduces
this exactly, so the downstream broadcasting in `_payoff_player1` is identical to
production.

Nothing in agents/ envs/ run/ utils/ is modified.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class _PointDist:
    """Degenerate 'distribution' whose samples are a constant normalized action."""

    def __init__(self, a_norm: float, device: torch.device):
        self._a = float(a_norm)
        self._device = device

    def sample(self, shape: Tuple[int, ...], generator: Optional[torch.Generator] = None
               ) -> torch.Tensor:
        """Return a constant tensor shaped like Beta(batch=(1,1)).sample(shape)."""
        m = int(shape[0])
        return torch.full((m, 1, 1), self._a, dtype=torch.float32, device=self._device)


class BetaPolicyAgent:
    """Minimal stand-in for PPOTwoPlayersBandit at a FIXED Beta(alpha, beta) policy.

    Unlike :class:`PointPolicyAgent` (degenerate point profile), this adapter
    reconstructs the *stochastic* terminal policy from a run's final
    ``(alpha_mean, beta_mean)`` snapshot, so ``eval_exploitability`` runs the
    IDENTICAL code path it takes on a live agent: ``dist.sample((M,),
    generator=...)`` raises TypeError on a real ``torch.distributions.Beta``
    and falls back to the derived-seed path — exactly as in training. Batch
    shape (1,1) matches the real agent's shape contract (samples (M,1,1)).

    Added 2026-07-23 (follow-up session, Task 2): post-hoc exploitability of
    the ``no_exploit`` ablation arm's terminal profiles. Additive change only.
    """

    def __init__(self, alpha: float, beta: float,
                 effort_bounds: Tuple[float, float] = (0.0, 100.0),
                 device: str = "cpu"):
        """Build a frozen Beta policy with the given shape parameters.

        Args:
            alpha: Beta alpha of the terminal policy.
            beta: Beta beta of the terminal policy.
            effort_bounds: (low, high) effort bounds used by the estimator.
            device: Torch device string.
        """
        self.low, self.high = float(effort_bounds[0]), float(effort_bounds[1])
        self.device = torch.device(device)
        # Only used by the estimator for a device lookup.
        self.net = nn.Linear(1, 1).to(self.device)
        self._alpha = torch.full((1, 1), float(alpha), dtype=torch.float32,
                                 device=self.device)
        self._beta = torch.full((1, 1), float(beta), dtype=torch.float32,
                                device=self.device)

    def state_from_params(self, *, q: float, k: float, w_h: float, w_l: float) -> torch.Tensor:
        """Mirror the real agent's 3-feature normalized state (value is irrelevant here)."""
        s = torch.tensor([float(q) / 60.0, float(k) / 1e-3, (float(w_h) - float(w_l)) / 10.0],
                         dtype=torch.float32, device=self.device)
        return s.unsqueeze(0)

    def dist(self, state: torch.Tensor, *, net=None):
        """Return the frozen Beta distribution and a dummy value."""
        return torch.distributions.Beta(self._alpha, self._beta), None


class PointPolicyAgent:
    """Minimal stand-in for PPOTwoPlayersBandit at a fixed deterministic effort."""

    def __init__(self, effort: float, effort_bounds: Tuple[float, float] = (0.0, 100.0),
                 device: str = "cpu"):
        """Build a point policy that always plays ``effort``.

        Args:
            effort: The deterministic effort to play.
            effort_bounds: (low, high) effort bounds used by the estimator.
            device: Torch device string.
        """
        self.low, self.high = float(effort_bounds[0]), float(effort_bounds[1])
        self.device = torch.device(device)
        # Only used by the estimator for a device lookup.
        self.net = nn.Linear(1, 1).to(self.device)
        a = (float(effort) - self.low) / (self.high - self.low)
        self._a = min(max(a, 1e-9), 1.0 - 1e-9)
        self.effort = float(effort)

    def state_from_params(self, *, q: float, k: float, w_h: float, w_l: float) -> torch.Tensor:
        """Mirror the real agent's 3-feature normalized state (value is irrelevant here)."""
        s = torch.tensor([float(q) / 60.0, float(k) / 1e-3, (float(w_h) - float(w_l)) / 10.0],
                         dtype=torch.float32, device=self.device)
        return s.unsqueeze(0)

    def dist(self, state: torch.Tensor, *, net=None):
        """Return the degenerate point 'distribution' and a dummy value."""
        return _PointDist(self._a, self.device), None
