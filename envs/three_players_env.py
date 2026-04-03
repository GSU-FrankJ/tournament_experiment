#!/usr/bin/env python3
"""
Three-player one-stage tournament environment (self-play).

Implements the model:
- Output of stage one: y_i = e_i + eps_i, eps_i ~ U(-q, q)
- Cost: c(e_i) = k e_i^2
- Expected utility: Eu_i = w_L + p_i (w_H - w_L) - k e_i^2

This env performs true self-play for three identical competitors: one winner,
two losers in each episode. Win probabilities are, by default, evaluated via the
closed-form integrals in ``utils.prob`` with an optional Monte-Carlo fallback
for diagnostics. The env is single-step (bandit) and returns dummy observations.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, List
import numpy as np
import torch

from utils.prob import p_from_diff, win_prob_three_players, win_prob_three_players_grad


class ThreePlayersEnv:
    """Self-play environment for three identical competitors."""

    def __init__(self, *, w_h: float, w_l: float, k: float, q: float,
                 effort_bounds: Tuple[float, float] = (0.0, 200.0),
                 seed: int = 42, mc_samples: int = 3000,
                 allow_near_symmetric_shortcut: bool = True,
                 track_shortcut_stats: bool = False,
                 use_analytic_probabilities: bool = True,
                 use_binary_rewards: bool = False,
                 reward_mode: str = "expected",
                 noise_scale: float = 0.0):
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
        self.use_analytic = bool(use_analytic_probabilities)
        self.use_binary_rewards = bool(use_binary_rewards)
        self.reward_mode = str(reward_mode)
        self.noise_scale = float(noise_scale)
        self._rng = np.random.default_rng(self.seed)

        # Episode counter for deterministic RNG per-episode
        self._episode = 0
        self.shortcut_hits = 0
        self.full_path_calls = 0
        self.analytic_calls = 0

    # --- probability helpers ---
    def _win_probs(self, e1: float, e2: float, e3: float) -> Tuple[float, float, float]:
        """Monte Carlo estimate of win probabilities for three players.

        Uses one shared noise matrix (shape [mc_samples, 3]) to ensure the
        probability mass is coherent across players in the same episode.
        """
        if self.use_analytic:
            p1 = win_prob_three_players(e1, e2, e3, self.q)
            p2 = win_prob_three_players(e2, e1, e3, self.q)
            p3 = win_prob_three_players(e3, e1, e2, self.q)
            total = p1 + p2 + p3
            if total > 0.0:
                p1 /= total
                p2 /= total
                p3 /= total
            else:
                p1 = p2 = p3 = 1.0 / 3.0
            p1 = float(np.clip(p1, 0.0, 1.0))
            p2 = float(np.clip(p2, 0.0, 1.0))
            p3 = float(np.clip(p3, 0.0, 1.0))
            if self.track_shortcut_stats:
                self.full_path_calls += 1
            self.analytic_calls += 1
            return p1, p2, p3

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

        costs_list: List[float] = [self.k * e * e for e in efforts]

        if self.reward_mode == "hybrid":
            # Expected pairwise utility + tunable stochastic noise.
            # noise_scale=0 → pure expected, noise_scale=1 → full pairwise binary.
            # Preserves equilibrium for any noise_scale (noise is zero-mean).
            w_gap = self.w_h - self.w_l
            e1, e2, e3 = efforts
            ns = self.noise_scale

            # Analytical pairwise win probs
            p12 = float(p_from_diff(e1 - e2, self.q))
            p13 = float(p_from_diff(e1 - e3, self.q))
            p23 = float(p_from_diff(e2 - e3, self.q))

            # Stochastic pairwise duels (for noise term)
            eps_12 = self._rng.uniform(-self.q, self.q, size=2)
            win_1v2 = float((e1 + eps_12[0]) > (e2 + eps_12[1]))
            eps_13 = self._rng.uniform(-self.q, self.q, size=2)
            win_1v3 = float((e1 + eps_13[0]) > (e3 + eps_13[1]))
            eps_23 = self._rng.uniform(-self.q, self.q, size=2)
            win_2v3 = float((e2 + eps_23[0]) > (e3 + eps_23[1]))

            # Zero-mean noise per duel: (indicator - analytical_prob)
            # Player i's effective win rate = p_ij + ns*(1{i>j} - p_ij)
            eff_12 = p12 + ns * (win_1v2 - p12)
            eff_13 = p13 + ns * (win_1v3 - p13)
            eff_21 = (1.0 - p12) + ns * ((1.0 - win_1v2) - (1.0 - p12))
            eff_23 = p23 + ns * (win_2v3 - p23)
            eff_31 = (1.0 - p13) + ns * ((1.0 - win_1v3) - (1.0 - p13))
            eff_32 = (1.0 - p23) + ns * ((1.0 - win_2v3) - (1.0 - p23))

            u1 = self.w_l + 0.5 * (eff_12 + eff_13) * w_gap - costs_list[0]
            u2 = self.w_l + 0.5 * (eff_21 + eff_23) * w_gap - costs_list[1]
            u3 = self.w_l + 0.5 * (eff_31 + eff_32) * w_gap - costs_list[2]

            utilities = [u1, u2, u3]
            win_probs = [0.5 * (p12 + p13), 0.5 * (1.0 - p12 + p23),
                         0.5 * (1.0 - p13 + 1.0 - p23)]

        elif self.reward_mode == "pairwise_binary":
            # Stochastic pairwise binary duels: each pair (i,j) is an independent
            # 2-player contest. Player i's reward = avg of binary payoffs from
            # duels (i,j) and (i,k) minus cost. Preserves equilibrium e*=(w_H-w_L)/(4qk).
            w_gap = self.w_h - self.w_l
            e1, e2, e3 = efforts
            # Duel 1-vs-2: independent noise draw
            eps_12 = self._rng.uniform(-self.q, self.q, size=2)
            win_1v2 = (e1 + eps_12[0]) > (e2 + eps_12[1])
            # Duel 1-vs-3: independent noise draw
            eps_13 = self._rng.uniform(-self.q, self.q, size=2)
            win_1v3 = (e1 + eps_13[0]) > (e3 + eps_13[1])
            # Duel 2-vs-3: independent noise draw
            eps_23 = self._rng.uniform(-self.q, self.q, size=2)
            win_2v3 = (e2 + eps_23[0]) > (e3 + eps_23[1])

            # Each player's reward: average of binary payoffs from their two duels
            pay_1 = 0.5 * ((self.w_h if win_1v2 else self.w_l)
                           + (self.w_h if win_1v3 else self.w_l))
            pay_2 = 0.5 * ((self.w_h if not win_1v2 else self.w_l)
                           + (self.w_h if win_2v3 else self.w_l))
            pay_3 = 0.5 * ((self.w_h if not win_1v3 else self.w_l)
                           + (self.w_h if not win_2v3 else self.w_l))

            utilities = [pay_1 - costs_list[0],
                         pay_2 - costs_list[1],
                         pay_3 - costs_list[2]]
            # Win probs for info (pairwise average, analytical)
            win_probs = [
                0.5 * (float(p_from_diff(e1 - e2, self.q))
                       + float(p_from_diff(e1 - e3, self.q))),
                0.5 * (float(p_from_diff(e2 - e1, self.q))
                       + float(p_from_diff(e2 - e3, self.q))),
                0.5 * (float(p_from_diff(e3 - e1, self.q))
                       + float(p_from_diff(e3 - e2, self.q))),
            ]
        elif self.use_binary_rewards:
            # Stochastic binary rewards: sample noise, pick winner, w_H/w_L payoffs
            # Matches 2-player env reward structure for consistent gradient signals
            eps = self._rng.uniform(-self.q, self.q, size=3)
            outputs = [efforts[i] + eps[i] for i in range(3)]
            winner = int(np.argmax(outputs))
            payoffs = [self.w_l] * 3
            payoffs[winner] = self.w_h
            utilities = [payoffs[i] - costs_list[i] for i in range(3)]
            win_probs = [0.0, 0.0, 0.0]
            win_probs[winner] = 1.0
        else:
            p1, p2, p3 = self._win_probs(efforts[0], efforts[1], efforts[2])
            win_probs = [p1, p2, p3]
            utilities = [self.w_l + win_probs[i] * (self.w_h - self.w_l) - costs_list[i]
                         for i in range(3)]

        obs = (torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
        rewards = torch.tensor(utilities, dtype=torch.float32)
        costs_t = torch.tensor(costs_list, dtype=torch.float32)
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
            "analytic_calls": int(self.analytic_calls),
            "mode": "analytic" if self.use_analytic else "mc",
        }

    @staticmethod
    def _dp_from_diff_dd(d: float, q: float) -> float:
        """Derivative of p_from_diff(d, q) with respect to d.

        For |d| < 2q: dp/dd = 1/(2q) - |d|/(4q^2).
        Otherwise 0.
        """
        if abs(d) >= 2.0 * q:
            return 0.0
        return 1.0 / (2.0 * q) - abs(d) / (4.0 * q * q)

    def expected_utility_gradient(self, e1: float, e2: float, e3: float) -> Tuple[float, float, float]:
        """Analytical gradient of each player's expected utility with respect to their OWN effort.

        Returns:
            (∂U_1/∂e_1, ∂U_2/∂e_2, ∂U_3/∂e_3) - the gradient each player uses to update their effort

        Dispatches based on reward_mode:
        - "expected": standard 3-player win prob gradient
        - "pairwise_binary": gradient of expected pairwise reward
        """
        reward_gap = self.w_h - self.w_l

        if self.reward_mode in ("pairwise_binary", "hybrid"):
            # E[R_i] = w_L + 0.5*(p_ij + p_ik)*w_gap - k*e_i^2
            # ∂E[R_i]/∂e_i = 0.5*w_gap*(dp_ij/dd + dp_ik/dd) - 2k*e_i
            dp = self._dp_from_diff_dd
            grad1 = 0.5 * reward_gap * (dp(e1 - e2, self.q) + dp(e1 - e3, self.q)) - 2.0 * self.k * e1
            grad2 = 0.5 * reward_gap * (dp(e2 - e1, self.q) + dp(e2 - e3, self.q)) - 2.0 * self.k * e2
            grad3 = 0.5 * reward_gap * (dp(e3 - e1, self.q) + dp(e3 - e2, self.q)) - 2.0 * self.k * e3
            return grad1, grad2, grad3

        if not self.use_analytic:
            raise RuntimeError("Analytic gradients require use_analytic_probabilities=True")

        # Player 1's gradient: ∂U_1/∂e_1 = (w_h - w_l) * ∂p_1/∂e_1 - 2k*e_1
        # Call with e1 as focal player (first arg)
        dp1_de1, _, _ = win_prob_three_players_grad(e1, e2, e3, self.q)
        grad1 = reward_gap * dp1_de1 - 2.0 * self.k * e1

        # Player 2's gradient: ∂U_2/∂e_2 = (w_h - w_l) * ∂p_2/∂e_2 - 2k*e_2
        # Call with e2 as focal player (first arg)
        dp2_de2, _, _ = win_prob_three_players_grad(e2, e1, e3, self.q)
        grad2 = reward_gap * dp2_de2 - 2.0 * self.k * e2

        # Player 3's gradient: ∂U_3/∂e_3 = (w_h - w_l) * ∂p_3/∂e_3 - 2k*e_3
        # Call with e3 as focal player (first arg)
        dp3_de3, _, _ = win_prob_three_players_grad(e3, e1, e2, self.q)
        grad3 = reward_gap * dp3_de3 - 2.0 * self.k * e3

        return grad1, grad2, grad3
