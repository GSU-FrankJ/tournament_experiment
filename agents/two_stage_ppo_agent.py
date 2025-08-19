"""
Two-stage PPO agent honoring training-order semantics:

- PPO method: Stage 1 first. We optimize Stage-1 objective that includes the
  continuation term E[k e_2^2] in the return. For simplicity and reproducibility,
  we train a single scalar policy that outputs symmetric efforts per stage based
  on a dummy state, and we collect rolling averages for plotting and CSV outputs.
"""

from __future__ import annotations

from typing import Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ScalarPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoStagePPOAgent:
    def __init__(self, effort_bounds_stage1: Tuple[float, float] = (0.0, 100.0), effort_bounds_stage2: Tuple[float, float] = (0.0, 200.0)):
        self.b1 = effort_bounds_stage1
        self.b2 = effort_bounds_stage2
        self.pi1 = ScalarPolicy()
        self.pi2 = ScalarPolicy()
        self.opt = optim.Adam(list(self.pi1.parameters()) + list(self.pi2.parameters()), lr=1e-3)
        self.entropy_coef = 1e-3

        self.stage1_history: List[float] = []
        self.stage2_history: List[float] = []

    def _scale(self, y: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        return lo + y * (hi - lo)

    def act_stage1(self, t: int) -> torch.Tensor:
        with torch.no_grad():
            y = self.pi1(torch.tensor([[t / 10000.0]], dtype=torch.float32)).squeeze()
            e = self._scale(y, self.b1)
        return e

    def act_stage2(self, t: int) -> torch.Tensor:
        with torch.no_grad():
            y = self.pi2(torch.tensor([[t / 10000.0]], dtype=torch.float32)).squeeze()
            e = self._scale(y, self.b2)
        return e

    def update(self, loss: torch.Tensor) -> None:
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.pi1.parameters()) + list(self.pi2.parameters()), 0.5)
        self.opt.step()


