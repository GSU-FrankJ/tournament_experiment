#!/usr/bin/env python3
"""
Rollout Statistics Accumulator for PPO training diagnostics.

Tracks per-update statistics for:
- Sampled efforts (from learner policy only)
- State vectors (learner transitions only)
- Rewards (learner transitions only)

Uses numerically stable Welford's online algorithm for mean/variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
import torch
import numpy as np


@dataclass
class WelfordAccumulator:
    """
    Welford's online algorithm for numerically stable mean and variance.
    
    Reference: Welford, B.P. (1962). "Note on a Method for Calculating 
    Corrected Sums of Squares and Products". Technometrics. 4(3): 419–420.
    """
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0  # sum of squared differences from the current mean

    def update(self, value: float) -> None:
        """Add a single scalar value to the accumulator."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def update_batch(self, values: np.ndarray) -> None:
        """Add a batch of values (flattened)."""
        for v in values.flat:
            self.update(float(v))

    def get_mean(self) -> float:
        """Return current mean (0.0 if no samples)."""
        return self.mean if self.count > 0 else 0.0

    def get_std(self, unbiased: bool = False) -> float:
        """
        Return current standard deviation.
        
        Args:
            unbiased: If True, use Bessel's correction (divide by n-1).
                      If False, divide by n (population std).
        
        Returns:
            Standard deviation, or 0.0 if insufficient samples.
        """
        if self.count < 2:
            return 0.0
        if unbiased:
            variance = self.M2 / (self.count - 1)
        else:
            variance = self.M2 / self.count
        return math.sqrt(max(0.0, variance))

    def reset(self) -> None:
        """Reset all statistics."""
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0


@dataclass
class RolloutStatsAccumulator:
    """
    Accumulates rollout statistics for PPO diagnostics.
    
    Tracks sampled efforts, states, and rewards from learner-generated
    transitions only.
    
    Attributes:
        effort_stats: Welford accumulator for sampled efforts
        state_stats: Welford accumulator for state vector elements (global)
        reward_stats: Welford accumulator for rewards
    """
    effort_stats: WelfordAccumulator = field(default_factory=WelfordAccumulator)
    state_stats: WelfordAccumulator = field(default_factory=WelfordAccumulator)
    reward_stats: WelfordAccumulator = field(default_factory=WelfordAccumulator)

    def update_effort(self, effort: float, *, player: str | None = None) -> None:
        """
        Record a sampled effort value from learner policy.
        
        Args:
            effort: The effort value (in original effort_range units, not normalized)
            player: Optional player identifier ("p1" or "p2") for symmetry diagnostics.
        """
        self.effort_stats.update(float(effort))
        # Per-player effort stats (for symmetry diagnostics)
        if player == "p1":
            if not hasattr(self, "_effort_p1_stats"):
                self._effort_p1_stats = WelfordAccumulator()
            self._effort_p1_stats.update(float(effort))
        elif player == "p2":
            if not hasattr(self, "_effort_p2_stats"):
                self._effort_p2_stats = WelfordAccumulator()
            self._effort_p2_stats.update(float(effort))

    def update_state(self, state: torch.Tensor | np.ndarray) -> None:
        """
        Record a state vector from learner transition.
        
        Computes global statistics across all state dimensions.
        
        Args:
            state: State tensor/array, any shape (will be flattened)
        """
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        self.state_stats.update_batch(np.asarray(state).flatten())

    def update_reward(self, reward: float) -> None:
        """
        Record a reward from learner transition.
        
        Args:
            reward: The reward value
        """
        self.reward_stats.update(float(reward))

    def get_sample_avg_effort(self) -> float:
        """Return mean of sampled efforts (0.0 if no samples)."""
        return self.effort_stats.get_mean()

    def get_effort_count(self) -> int:
        """Return number of effort samples recorded."""
        return self.effort_stats.count

    def get_state_mean(self) -> float:
        """Return global mean of state elements."""
        return self.state_stats.get_mean()

    def get_state_std(self) -> float:
        """Return global std of state elements (population std)."""
        return self.state_stats.get_std(unbiased=False)

    def get_reward_mean(self) -> float:
        """Return mean reward."""
        return self.reward_stats.get_mean()

    def get_reward_std(self) -> float:
        """Return reward std (population std)."""
        return self.reward_stats.get_std(unbiased=False)

    def get_summary(self) -> dict:
        """
        Return all statistics as a dictionary.
        
        Returns:
            Dictionary with keys:
                - sample_avg_effort: mean sampled effort
                - effort_sample_count: number of effort samples
                - state_mean: global mean of state elements
                - state_std: global std of state elements
                - reward_mean: mean reward
                - reward_std: reward std
                - sample_avg_effort_p1: mean effort for player1 (if tracked)
                - sample_avg_effort_p2: mean effort for player2 (if tracked)
        """
        effort_p1_mean = None
        effort_p2_mean = None
        effort_p1_count = None
        effort_p2_count = None
        if hasattr(self, "_effort_p1_stats"):
            effort_p1_mean = self._effort_p1_stats.get_mean()
            effort_p1_count = self._effort_p1_stats.count
        if hasattr(self, "_effort_p2_stats"):
            effort_p2_mean = self._effort_p2_stats.get_mean()
            effort_p2_count = self._effort_p2_stats.count
        return {
            "sample_avg_effort": self.get_sample_avg_effort(),
            "effort_sample_count": self.get_effort_count(),
            "state_mean": self.get_state_mean(),
            "state_std": self.get_state_std(),
            "reward_mean": self.get_reward_mean(),
            "reward_std": self.get_reward_std(),
            "sample_avg_effort_p1": effort_p1_mean,
            "sample_avg_effort_p2": effort_p2_mean,
            "effort_sample_count_p1": effort_p1_count,
            "effort_sample_count_p2": effort_p2_count,
        }

    def reset(self) -> None:
        """Reset all accumulators for a new update period."""
        self.effort_stats.reset()
        self.state_stats.reset()
        self.reward_stats.reset()
        if hasattr(self, "_effort_p1_stats"):
            self._effort_p1_stats.reset()
        if hasattr(self, "_effort_p2_stats"):
            self._effort_p2_stats.reset()


@dataclass
class PPOUpdateStats:
    """
    Statistics from a single PPO update, returned by agent.update().
    
    Contains advantage statistics (raw and normalized if applicable),
    along with other training metrics.
    """
    adv_mean: float = 0.0          # Mean of raw advantages (before normalization)
    adv_std: float = 0.0           # Std of raw advantages (before normalization)
    adv_norm_std: float = 0.0      # Std of normalized advantages (should be ~1.0)
    approx_kl: float = 0.0         # Approximate KL divergence
    batch_entropy: float = 0.0     # Mean batch entropy
    value_mean: float = 0.0        # Mean of value estimates
    value_std: float = 0.0         # Std of value estimates
    opponent_history_size: int = 0
    opponent_last_sync: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/CSV."""
        return {
            "adv_mean": self.adv_mean,
            "adv_std": self.adv_std,
            "adv_norm_std": self.adv_norm_std,
            "approx_kl": self.approx_kl,
            "batch_entropy": self.batch_entropy,
            "value_mean": self.value_mean,
            "value_std": self.value_std,
            "opponent_history_size": self.opponent_history_size,
            "opponent_last_sync": self.opponent_last_sync,
        }


def compute_policy_mean_effort(
    alpha_mean: float,
    beta_mean: float,
    effort_low: float,
    effort_high: float,
) -> float:
    """
    Compute policy mean effort from Beta distribution parameters.
    
    The Beta distribution mean is alpha / (alpha + beta), which gives
    a normalized action in [0, 1]. This is then scaled to the effort range.
    
    Args:
        alpha_mean: Mean of alpha concentration parameter
        beta_mean: Mean of beta concentration parameter
        effort_low: Lower bound of effort range
        effort_high: Upper bound of effort range
    
    Returns:
        Policy mean effort = effort_low + beta_mean_norm * (effort_high - effort_low)
        where beta_mean_norm = alpha_mean / (alpha_mean + beta_mean)
    
    Reference: 
        ppo_two_players_clean.py:45-48 (dist method returning Beta distribution)
        run_two_players.py:569-572 (effort computation from dist.mean)
    """
    if alpha_mean + beta_mean <= 0:
        return effort_low  # Fallback for edge case
    
    beta_mean_norm = alpha_mean / (alpha_mean + beta_mean)
    return effort_low + beta_mean_norm * (effort_high - effort_low)


def verify_policy_mean(
    reported_policy: float,
    alpha_mean: float,
    beta_mean: float,
    effort_low: float,
    effort_high: float,
    tolerance: float = 0.1,
) -> tuple[bool, float]:
    """
    Verify that reported policy equals Beta mean mapped to effort range.
    
    Args:
        reported_policy: The policy value as reported in logs
        alpha_mean: Mean alpha parameter
        beta_mean: Mean beta parameter
        effort_low: Lower bound of effort range
        effort_high: Upper bound of effort range
        tolerance: Acceptable absolute error
    
    Returns:
        Tuple of (is_valid, error) where:
            - is_valid: True if |reported - computed| < tolerance
            - error: The absolute difference
    """
    computed = compute_policy_mean_effort(alpha_mean, beta_mean, effort_low, effort_high)
    error = abs(reported_policy - computed)
    return (error < tolerance, error)

