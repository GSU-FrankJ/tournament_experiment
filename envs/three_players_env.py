#!/usr/bin/env python3
"""
Three-player one-stage tournament environment (self-play).

Model (one-step stochastic game):
- Output: y_i = e_i + eps_i, eps_i ~ U(-q, q), drawn fresh each episode
- Rank-order payoff: the player with the highest realized output receives w_H;
  the other two receive w_L (exact ties broken uniformly at random)
- Cost: c(e_i) = k e_i^2
- Reward: r_i = payoff_i - k e_i^2 — a REALIZED, SAMPLED outcome, never an
  expectation

Training agents observe ONLY these sampled outcomes. Closed-form win
probabilities / expected utilities must never enter the training reward path;
the only closed-form helper kept on this class is ``expected_utility_gradient``,
which exists for the Appendix-A numerical baseline and offline evaluation.

The env is single-step (bandit) and returns dummy observations.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, List
import numpy as np
import torch

from utils.prob import win_prob_three_players_grad


class ThreePlayersEnv:
    """Self-play environment for three identical competitors (sampled rewards)."""

    def __init__(self, *, w_h: float, w_l: float, k: float, q: float,
                 effort_bounds: Tuple[float, float] = (0.0, 200.0),
                 seed: int = 42):
        self.w_h = float(w_h)
        self.w_l = float(w_l)
        self.k = float(k)
        self.q = float(q)
        self.effort_range = (float(effort_bounds[0]), float(effort_bounds[1]))
        self.seed = int(seed)
        self.num_players = 3
        # Single RNG that advances across steps; the env must be constructed
        # once per run so noise is not re-seeded between episodes.
        self._rng = np.random.default_rng(self.seed)
        self._episode = 0

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

        costs_list: List[float] = [self.k * e * e for e in efforts]

        # Sampled tournament outcome: y_i = e_i + eps_i, winner takes w_H.
        eps = self._rng.uniform(-self.q, self.q, size=3)
        outputs = [efforts[i] + float(eps[i]) for i in range(3)]
        max_y = max(outputs)
        top = [i for i in range(3) if outputs[i] == max_y]
        winner = top[0] if len(top) == 1 else int(self._rng.choice(top))

        payoffs = [self.w_l] * 3
        payoffs[winner] = self.w_h
        utilities = [payoffs[i] - costs_list[i] for i in range(3)]

        obs = (torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
        rewards = torch.tensor(utilities, dtype=torch.float32)
        costs_t = torch.tensor(costs_list, dtype=torch.float32)
        done = True
        info: Dict[str, Any] = {
            "efforts": tuple(efforts),
            "noises": tuple(float(x) for x in eps),
            "outputs": tuple(outputs),
            "winner": winner,
        }
        return obs, rewards, costs_t, done, info

    # --- closed-form helpers (EVALUATION / BASELINE ONLY) ---
    def expected_utility_gradient(self, e1: float, e2: float, e3: float) -> Tuple[float, float, float]:
        """Analytical gradient of each player's expected utility w.r.t. their OWN effort.

        EVALUATION / BASELINE ONLY: used by the numerical gradient reference
        (Appendix A) and offline diagnostics. Must never be used to compute
        training rewards.

        Returns:
            (dU_1/de_1, dU_2/de_2, dU_3/de_3)
        """
        reward_gap = self.w_h - self.w_l

        # Player 1's gradient: dU_1/de_1 = (w_h - w_l) * dp_1/de_1 - 2k*e_1
        dp1_de1, _, _ = win_prob_three_players_grad(e1, e2, e3, self.q)
        grad1 = reward_gap * dp1_de1 - 2.0 * self.k * e1

        # Player 2's gradient: dU_2/de_2 = (w_h - w_l) * dp_2/de_2 - 2k*e_2
        dp2_de2, _, _ = win_prob_three_players_grad(e2, e1, e3, self.q)
        grad2 = reward_gap * dp2_de2 - 2.0 * self.k * e2

        # Player 3's gradient: dU_3/de_3 = (w_h - w_l) * dp_3/de_3 - 2k*e_3
        dp3_de3, _, _ = win_prob_three_players_grad(e3, e1, e2, self.q)
        grad3 = reward_gap * dp3_de3 - 2.0 * self.k * e3

        return grad1, grad2, grad3
