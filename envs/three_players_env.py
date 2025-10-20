#!/usr/bin/env python3
"""
Three-player one-stage tournament environment (self-play).

Implements the model:
- Output of stage one: y_i = e_i + eps_i, eps_i ~ U(-q, q)
- Cost: c(e_i) = k e_i^2
- Expected utility: Eu_i = w_L + p_i (w_H - w_L) - k e_i^2

This env performs true self-play for three identical competitors: one winner,
two losers in each episode. Win probabilities are computed via a shared
Monte-Carlo draw across the three players for numerical stability. The env is
single-step (bandit) and returns dummy observations.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, List
import numpy as np
import torch


class ThreePlayersEnv:
    """Self-play environment for three identical competitors."""

    def __init__(self, *, w_h: float, w_l: float, k: float, q: float,
                 effort_bounds: Tuple[float, float] = (0.0, 200.0),
                 seed: int = 42, mc_samples: int = 3000,
                 allow_near_symmetric_shortcut: bool = True,
                 track_shortcut_stats: bool = False):
        self.w_h = float(w_h)
        self.w_l = float(w_l)
        self.k = float(k)
        self.q = float(q)
        self.effort_range = (float(effort_bounds[0]), float(effort_bounds[1]))
        self.seed = int(seed)
        self.num_players = 3
        self.mc_samples = int(mc_samples)
        self.allow_near_symmetric_shortcut = bool(allow_near_symmetric_shortcut)
        self.track_shortcut_stats = bool(track_shortcut_stats)

        # Episode counter for deterministic RNG per-episode
        self._episode = 0
        self.shortcut_hits = 0
        self.full_path_calls = 0

    # --- probability helpers ---
    def _win_probs(self, e1: float, e2: float, e3: float) -> Tuple[float, float, float]:
        """Monte Carlo estimate of win probabilities for three players.

        Uses one shared noise matrix (shape [mc_samples, 3]) to ensure the
        probability mass is coherent across players in the same episode.
        """
        # Near-symmetric shortcut to avoid excessive Monte Carlo when efforts
        # are very close compared to the noise scale.
        if (self.allow_near_symmetric_shortcut and
                max(abs(e1 - e2), abs(e1 - e3), abs(e2 - e3)) <= 0.01 * self.q):
            if self.track_shortcut_stats:
                self.shortcut_hits += 1
            return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

        if self.track_shortcut_stats:
            self.full_path_calls += 1

        rng = np.random.default_rng(self.seed + self._episode)
        eps = rng.uniform(-self.q, self.q, size=(self.mc_samples, 3))
        scores = np.array([[e1, e2, e3]]) + eps
        winners = np.argmax(scores, axis=1)
        p1 = float(np.mean(winners == 0))
        p2 = float(np.mean(winners == 1))
        p3 = float(np.mean(winners == 2))
        return p1, p2, p3

    # --- gym-like API ---
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._episode += 1
        zero = torch.tensor([0.0], dtype=torch.float32)
        return zero.clone(), zero.clone(), zero.clone()

    def step(self, actions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        if len(actions) != 3:
            raise ValueError(f"ThreePlayersEnv expected 3 actions, got {len(actions)}")

        # clamp actions to effort bounds
        lo, hi = self.effort_range
        efforts: List[float] = []
        for a in actions:
            e = float(a.item()) if isinstance(a, torch.Tensor) else float(a)
            e = max(lo, min(hi, e))
            efforts.append(e)

        p1, p2, p3 = self._win_probs(efforts[0], efforts[1], efforts[2])
        win_probs = [p1, p2, p3]

        # utilities and costs
        utilities: List[float] = []
        costs: List[float] = []
        for i in range(3):
            reward = self.w_l + win_probs[i] * (self.w_h - self.w_l)
            cost = self.k * efforts[i] * efforts[i]
            utilities.append(reward - cost)
            costs.append(cost)

        obs = (torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
        rewards = torch.tensor(utilities, dtype=torch.float32)
        costs_t = torch.tensor(costs, dtype=torch.float32)
        done = True
        info: Dict[str, Any] = {
            "efforts": tuple(efforts),
            "win_probabilities": tuple(win_probs),
        }
        return obs, rewards, costs_t, done, info

    def get_shortcut_stats(self) -> Dict[str, int]:
        """Return usage statistics for the near-symmetric shortcut."""
        return {
            "shortcut_hits": int(self.shortcut_hits),
            "full_path_calls": int(self.full_path_calls),
        }
