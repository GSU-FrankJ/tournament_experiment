"""
Symmetric three-player one-stage tournament environment.

For plotting and PPO comparisons; uses simple Monte Carlo to estimate win
probabilities without assuming symmetry inside the step.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import torch


class ThreePlayersEnv:
    def __init__(self, w_h: float, w_l: float, k: float, q: float, effort_bounds: Tuple[float, float] = (0.0, 200.0), seed: int = 42, mc_samples: int = 20000):
        self.w_h = float(w_h)
        self.w_l = float(w_l)
        self.k = float(k)
        self.q = float(q)
        self.effort_low = float(effort_bounds[0])
        self.effort_high = float(effort_bounds[1])
        self.mc_samples = int(mc_samples)
        self.rng = np.random.default_rng(seed)

    def _win_prob(self, e_i: float, e_j: float, e_k: float) -> float:
        eps_i = self.rng.uniform(-self.q, self.q, self.mc_samples)
        eps_j = self.rng.uniform(-self.q, self.q, self.mc_samples)
        eps_k = self.rng.uniform(-self.q, self.q, self.mc_samples)
        xi = e_i + eps_i
        xj = e_j + eps_j
        xk = e_k + eps_k
        return float(np.mean((xi > xj) & (xi > xk)))

    def expected_utility(self, e_i: float, others: Tuple[float, float]) -> float:
        p = self._win_prob(e_i, others[0], others[1])
        return self.w_l + p * (self.w_h - self.w_l) - self.k * (e_i ** 2)

    def step(self, efforts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        e1 = float(efforts[0].item())
        e2 = float(efforts[1].item())
        e3 = float(efforts[2].item())
        u1 = self.expected_utility(e1, (e2, e3))
        u2 = self.expected_utility(e2, (e1, e3))
        u3 = self.expected_utility(e3, (e1, e2))
        costs = torch.tensor([self.k * e1 * e1, self.k * e2 * e2, self.k * e3 * e3], dtype=torch.float32)
        rewards = torch.tensor([u1, u2, u3], dtype=torch.float32)
        obs = (torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
        return obs, rewards, costs, True, {"efforts": (e1, e2, e3)}


