"""
Different Cost Environment for Two Players
=========================================

Implements a one-stage two-player tournament with asymmetric cost parameters
k1 != k2 (k1 < k2 in the target experiments).

Model (one-step stochastic game):
- Output (stage one): y_i = e_i + ε_i, with ε_i ~ U(-q, q) drawn fresh each episode
- Rank-order payoff: the player with the higher realized output receives w_H,
  the other w_L (exact ties broken uniformly at random)
- Cost: c(e_i) = k_i e_i^2
- Reward: r_i = payoff_i - k_i e_i^2 — a REALIZED, SAMPLED outcome, never an
  expectation

Training agents observe ONLY these sampled outcomes (mirrors TwoPlayersEnv).
The closed-form ``expected_utility`` helper below is EVALUATION/BASELINE-ONLY
(numerical gradient reference, offline diagnostics) and must never enter the
training reward path.
"""

from __future__ import annotations

from typing import Tuple, List, Dict, Any
import numpy as np
import torch

from utils.prob import p_from_efforts


class DifferentCostEnv:
    """Two players, different cost parameters (k1, k2); sampled rewards."""

    def __init__(self, *, w_h: float, w_l: float, k1: float, k2: float, q: float,
                 effort_bounds: Tuple[float, float] = (0.0, 200.0), seed: int = 42):
        self.w_h = float(w_h)
        self.w_l = float(w_l)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.q = float(q)
        self.low = float(effort_bounds[0])
        self.high = float(effort_bounds[1])
        self.seed = int(seed)
        # Single RNG that advances across steps; construct the env once per
        # run so noise is not re-seeded between episodes.
        self.rng = np.random.default_rng(self.seed)

    def draw_noise_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw Uniform(-q, q) noise and tie-break decisions for CRN-friendly batches.

        Mirrors TwoPlayersEnv.draw_noise_batch: one shared batch is reused for
        all perturbed evaluations of an MC-FD central difference (common random
        numbers).
        """
        eps1 = self.rng.uniform(-self.q, self.q, size=int(batch_size))
        eps2 = self.rng.uniform(-self.q, self.q, size=int(batch_size))
        tie_breaks = self.rng.integers(0, 2, size=int(batch_size))
        return eps1, eps2, tie_breaks

    # ---- closed-form helper (EVALUATION / BASELINE ONLY) ----
    def expected_utility(self, *, e_self: float, e_opp: float, k_self: float) -> float:
        """Closed-form E[u] — used by the numerical gradient reference and
        offline evaluation only. Must never be used as a training reward."""
        p = float(p_from_efforts(e_self, e_opp, self.q))
        return self.w_l + p * (self.w_h - self.w_l) - k_self * (e_self ** 2)

    # ---- gym-like API ----
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor([0.0]), torch.tensor([0.0]))

    def step(self, efforts: Tuple[torch.Tensor, torch.Tensor]):
        e1 = float(efforts[0].item())
        e2 = float(efforts[1].item())
        e1 = max(self.low, min(self.high, e1))
        e2 = max(self.low, min(self.high, e2))

        # Sampled tournament outcome: y_i = e_i + eps_i, winner takes w_H.
        eps1 = float(self.rng.uniform(-self.q, self.q))
        eps2 = float(self.rng.uniform(-self.q, self.q))
        y1 = e1 + eps1
        y2 = e2 + eps2
        if y1 > y2:
            winner = 0
        elif y2 > y1:
            winner = 1
        else:
            winner = int(self.rng.integers(0, 2))

        payoffs = [self.w_l, self.w_l]
        payoffs[winner] = self.w_h
        u1 = payoffs[0] - self.k1 * e1 * e1
        u2 = payoffs[1] - self.k2 * e2 * e2
        rewards = torch.tensor([u1, u2], dtype=torch.float32)
        costs = torch.tensor([self.k1 * e1 * e1, self.k2 * e2 * e2], dtype=torch.float32)
        obs = (torch.tensor([0.0]), torch.tensor([0.0]))
        info = {
            "efforts": (e1, e2),
            "noises": (eps1, eps2),
            "outputs": (y1, y2),
            "winner": winner,
        }
        done = True
        return obs, rewards, costs, done, info

