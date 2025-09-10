"""
Different Cost Environment for Two Players
=========================================

Implements a one-stage two-player tournament with asymmetric cost parameters
k1 != k2 (k1 < k2 in the target experiments). Noise is Uniform(-q, q), and the
win probability uses the exact closed-form triangular CDF from utils.prob.

Model:
- Output (stage one): y_i = e_i + ε_i, with ε_i ~ U(-q, q)
- Cost: c(e_i) = k_i e_i^2
- Expected utility: E[u_i] = w_L + p_i(win) (w_H - w_L) - k_i e_i^2

This file is a cleaned and corrected version migrated from backup, with the
linear-approximate probability replaced by the exact function. It exposes a
small gym-like API compatible with our PPO runner.
"""

from __future__ import annotations

from typing import Tuple, List, Dict, Any
import torch

from utils.prob import p_from_efforts


class DifferentCostEnv:
    """Two players, different cost parameters (k1, k2)."""

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

    # ---- utilities ----
    def expected_utility(self, *, e_self: float, e_opp: float, k_self: float) -> float:
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

        u1 = self.expected_utility(e_self=e1, e_opp=e2, k_self=self.k1)
        u2 = self.expected_utility(e_self=e2, e_opp=e1, k_self=self.k2)
        rewards = torch.tensor([u1, u2], dtype=torch.float32)
        costs = torch.tensor([self.k1 * e1 * e1, self.k2 * e2 * e2], dtype=torch.float32)
        obs = (torch.tensor([0.0]), torch.tensor([0.0]))
        info = {"efforts": (e1, e2)}
        done = True
        return obs, rewards, costs, done, info

