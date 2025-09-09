"""
Different Ability Environment for Two Players Tournament Game
============================================================

This environment implements a two-player contest with different ability
parameters (l1 > l2) under a single-stage setting with uniform noise.

Key points:
- Output (stage one): y_i = e_i + l_i + ε_i, ε_i ~ U(-q, q)
- Cost: c(e_i) = k_i e_i^2 with k1 = k2 in this scenario
- Expected utility: w_L + p_i(win) (w_H - w_L) - k_i e_i^2
- Win probability uses the exact triangular CDF for ε1 - ε2 ∈ [-2q, 2q]

This file is migrated from backup with no behavior changes, so agents and
run scripts can import envs.different_ability_env.DifferentAbilityEnv.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Tuple, List, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)


class DifferentAbilityEnv:
    """Two players with different ability parameters (l1 > l2)."""

    def __init__(self, config: Dict[str, Any]):
        # Ability parameters
        self.l1 = config.get("l1", 10)
        self.l2 = config.get("l2", 5)

        # Cost parameters (equal in this scenario)
        self.k = float(config.get("k", 0.0004))
        self.k1 = float(config.get("k1", self.k))
        self.k2 = float(config.get("k2", self.k))

        # Noise and rewards
        self.q = float(config["q"])
        self.w_h = float(config["w_h"])  
        self.w_l = float(config["w_l"])  
        self.w_diff = self.w_h - self.w_l

        # Bounds and misc
        self.effort_range = tuple(config["effort_range"])  # (low, high)
        self.seed = int(config.get("seed", 42))
        self.num_players = 2

        # Theoretical values (optional; used for analysis)
        self.theoretical_efforts = list(config.get("theoretical_efforts", []))
        self.theoretical_costs = list(config.get("theoretical_costs", []))
        self.theoretical_effort1 = float(config.get("theoretical_effort1", 0.0))
        self.theoretical_effort2 = float(config.get("theoretical_effort2", 0.0))

        self._validate_config()

        logger.info(
            f"DifferentAbilityEnv: l1={self.l1}, l2={self.l2}, k1={self.k1}, k2={self.k2}, q={self.q}, "
            f"w_h={self.w_h}, w_l={self.w_l}, bounds={self.effort_range}"
        )

    def _validate_config(self):
        if self.l1 <= 0 or self.l2 <= 0:
            raise ValueError("Ability parameters must be positive")
        if self.k1 <= 0 or self.k2 <= 0:
            raise ValueError("Cost parameters must be positive")
        if self.q <= 0:
            raise ValueError("Noise parameter q must be positive")
        if self.w_h <= self.w_l:
            raise ValueError("High reward must be greater than low reward")
        if len(self.effort_range) != 2 or self.effort_range[0] >= self.effort_range[1]:
            raise ValueError("Effort range must be (min, max) with min < max")

    # ---- probability / utility ----
    def probability_win_player1(self, e1: float, e2: float) -> float:
        """P(e1 + l1 + ε1 > e2 + l2 + ε2) with ε1, ε2 ~ U(-q, q)."""
        score1 = e1 + self.l1
        score2 = e2 + self.l2
        d = score2 - score1  # threshold for ε1 - ε2

        if d <= -2 * self.q:
            return 1.0
        if d >= 2 * self.q:
            return 0.0
        if d < 0:
            return 1.0 - ((d + 2 * self.q) ** 2) / (8 * self.q ** 2)
        return ((2 * self.q - d) ** 2) / (8 * self.q ** 2)

    def compute_utility(self, player_id: int, effort: float, other_effort: float) -> Tuple[float, float]:
        if player_id == 0:
            p_win = self.probability_win_player1(effort, other_effort)
            cost = self.k1 * effort * effort
        else:
            p_win = 1.0 - self.probability_win_player1(other_effort, effort)
            cost = self.k2 * effort * effort
        expected_reward = self.w_l + p_win * self.w_diff
        return expected_reward - cost, cost

    # ---- gym-like API ----
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([0.0]), torch.tensor([0.0])

    def step(self, actions: List[torch.Tensor]):
        if len(actions) != 2:
            raise ValueError("Expected 2 actions")
        e1 = float(actions[0].item())
        e2 = float(actions[1].item())
        low, high = self.effort_range
        e1 = max(low, min(high, e1))
        e2 = max(low, min(high, e2))

        u1, c1 = self.compute_utility(0, e1, e2)
        u2, c2 = self.compute_utility(1, e2, e1)
        p1 = self.probability_win_player1(e1, e2)

        obs = (torch.tensor([0.0]), torch.tensor([0.0]))
        rewards = torch.tensor([u1, u2], dtype=torch.float32)
        costs = torch.tensor([c1, c2], dtype=torch.float32)
        done = True
        info = {
            "efforts": [e1, e2],
            "effective_efforts": [e1 + self.l1, e2 + self.l2],
            "win_probabilities": [p1, 1.0 - p1],
            "costs": [c1, c2],
            "utilities": [u1, u2],
            "ability_parameters": [self.l1, self.l2],
            "cost_parameters": [self.k1, self.k2],
        }
        return obs, rewards, costs, done, info

    # ---- helpers for solvers ----
    def analyze_equilibrium(self, efforts: List[float]) -> Dict[str, Any]:
        e1, e2 = efforts
        u1, c1 = self.compute_utility(0, e1, e2)
        u2, c2 = self.compute_utility(1, e2, e1)
        p1 = self.probability_win_player1(e1, e2)

        if self.theoretical_efforts and len(self.theoretical_efforts) == 2:
            gaps = [abs(e1 - self.theoretical_efforts[0]), abs(e2 - self.theoretical_efforts[1])]
        elif self.theoretical_effort1 > 0 and self.theoretical_effort2 > 0:
            gaps = [abs(e1 - self.theoretical_effort1), abs(e2 - self.theoretical_effort2)]
        else:
            gaps = []

        quality = None
        if gaps:
            mg = max(gaps)
            if mg < 0.5:
                quality = "Excellent"
            elif mg < 1.0:
                quality = "Good"
            elif mg < 5.0:
                quality = "Fair"
            else:
                quality = "Poor"

        return {
            "efforts": efforts,
            "effective_efforts": [e1 + self.l1, e2 + self.l2],
            "theoretical_efforts": self.theoretical_efforts or [self.theoretical_effort1, self.theoretical_effort2],
            "gaps": gaps,
            "max_gap": max(gaps) if gaps else 0.0,
            "utilities": [u1, u2],
            "costs": [c1, c2],
            "win_probabilities": [p1, 1.0 - p1],
            "ability_parameters": [self.l1, self.l2],
            "cost_parameters": [self.k1, self.k2],
            "convergence_quality": quality or "Unknown",
        }

    def compute_gradients(self, efforts: List[float], eps: float = 1e-4) -> List[float]:
        e1, e2 = efforts
        u1, _ = self.compute_utility(0, e1, e2)
        u1p, _ = self.compute_utility(0, e1 + eps, e2)
        g1 = (u1p - u1) / eps
        u2, _ = self.compute_utility(1, e2, e1)
        u2p, _ = self.compute_utility(1, e2 + eps, e1)
        g2 = (u2p - u2) / eps
        return [g1, g2]

    def get_theoretical_efforts(self) -> List[float]:
        return self.theoretical_efforts[:] if self.theoretical_efforts else [self.theoretical_effort1, self.theoretical_effort2]

    def get_ability_parameters(self) -> List[float]:
        return [self.l1, self.l2]

    def get_cost_parameters(self) -> List[float]:
        return [self.k1, self.k2]
