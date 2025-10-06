"""
Symmetric two-player single-stage tournament environment.

Implements expected utilities with uniform noise and cost k e^2. Uses the
closed-form probability p(d, q) from utils.prob.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import torch

from utils.prob import p_from_efforts


class TwoPlayersEnv:
    """One-stage symmetric 2-player environment."""

    def __init__(self, w_h: float, w_l: float, k: float, q: float, effort_bounds: Tuple[float, float] = (0.0, 200.0), seed: int = 42):
        self.w_h = float(w_h)
        self.w_l = float(w_l)
        self.k = float(k)
        self.q = float(q)
        self.effort_low = float(effort_bounds[0])
        self.effort_high = float(effort_bounds[1])
        self.rng = np.random.default_rng(seed)

    def expected_utility(self, e_i: float, e_j: float) -> float:
        """E[u_i] = w_L + p(e_i,e_j)(w_H - w_L) - k e_i^2."""
        p = float(p_from_efforts(e_i, e_j, self.q))
        return self.w_l + p * (self.w_h - self.w_l) - self.k * (e_i ** 2)

    def step(self, efforts: Tuple[torch.Tensor, torch.Tensor]):
        e1 = float(efforts[0].item())
        e2 = float(efforts[1].item())
        u1 = self.expected_utility(e1, e2)
        u2 = self.expected_utility(e2, e1)
        costs = torch.tensor([self.k * e1 * e1, self.k * e2 * e2], dtype=torch.float32)
        rewards = torch.tensor([u1, u2], dtype=torch.float32)
        obs = (torch.tensor([0.0]), torch.tensor([0.0]))
        return obs, rewards, costs, True, {"efforts": (e1, e2)}





















