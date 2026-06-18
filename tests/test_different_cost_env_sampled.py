"""Equivalence-in-expectation test for the sampled DifferentCostEnv rewards.

Audit fix, target 1 safeguard: the rewritten env returns SAMPLED rank-order
rewards (y_i = e_i + eps_i, eps ~ U(-q, q); winner gets w_H, loser w_L; minus
k_i e_i^2). Averaged over many draws, these must match the closed-form expected
utility the OLD env returned as its training reward:

    EU_i = w_L + p(e_i, e_j, q) * (w_H - w_L) - k_i e_i^2

which is exactly the (kept, eval-only) ``DifferentCostEnv.expected_utility``.

Run directly (python tests/test_different_cost_env_sampled.py) or via pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from envs.different_cost_env import DifferentCostEnv

# Paper parameters (het-cost scenario): k1=0.0004, k2=0.00055, w=(8, 5.5), q=35
PARAMS = {"w_h": 8.0, "w_l": 5.5, "k1": 0.0004, "k2": 0.00055, "q": 35.0}
EFFORT_BOUNDS = (0.0, 100.0)
N_SAMPLES = 120_000
SEED = 123


def _sample_mean_rewards(efforts: tuple, n: int = N_SAMPLES, seed: int = SEED) -> np.ndarray:
    env = DifferentCostEnv(**PARAMS, effort_bounds=EFFORT_BOUNDS, seed=seed)
    actions = (torch.tensor([float(efforts[0])]), torch.tensor([float(efforts[1])]))
    out = np.empty((n, 2), dtype=np.float64)
    for t in range(n):
        _, rewards, _, _, _ = env.step(actions)
        out[t] = rewards.numpy()
    return out


def _check_profile(efforts: tuple) -> None:
    env_ref = DifferentCostEnv(**PARAMS, effort_bounds=EFFORT_BOUNDS, seed=SEED)
    e1, e2 = float(efforts[0]), float(efforts[1])
    # Old training reward == closed-form expected utility (kept as eval-only helper)
    ref = [
        env_ref.expected_utility(e_self=e1, e_opp=e2, k_self=PARAMS["k1"]),
        env_ref.expected_utility(e_self=e2, e_opp=e1, k_self=PARAMS["k2"]),
    ]
    samples = _sample_mean_rewards(efforts)
    mean = samples.mean(axis=0)
    se = samples.std(axis=0, ddof=1) / math.sqrt(samples.shape[0])
    for i in range(2):
        tol = max(6.0 * se[i], 0.02)
        diff = abs(mean[i] - ref[i])
        print(
            f"  efforts={efforts} player{i + 1}: sampled_mean={mean[i]:.4f} "
            f"closed_form={ref[i]:.4f} |diff|={diff:.4f} tol={tol:.4f}"
        )
        assert diff < tol, (
            f"dc sampled reward mean deviates from old closed-form EU: "
            f"player {i + 1}, efforts={efforts}, diff={diff:.4f} >= tol={tol:.4f}"
        )


def test_at_analytical_equilibrium():
    # e1* = 38.03, e2* = 27.66 at q=35 (analytical het-cost equilibrium)
    _check_profile((38.03, 27.66))


def test_symmetric_profile():
    _check_profile((30.0, 30.0))


def test_spread_profile():
    _check_profile((60.0, 15.0))


if __name__ == "__main__":
    print("[dc] equivalence-in-expectation test (sampled vs old closed-form)")
    test_at_analytical_equilibrium()
    test_symmetric_profile()
    test_spread_profile()
    print("[dc] PASS")
