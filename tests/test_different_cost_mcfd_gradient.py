"""Equivalence-in-expectation test for the het-cost sampled MC-FD gradient.

Audit follow-up safeguard: the ported MC-FD baseline gradient
(_compute_gradients_different_cost: sampled payoffs + CRN, central
differences) must, averaged over many independent CRN batches, match the
central finite difference of the closed-form expected utility at the same
delta — the (kept, eval-only) ``DifferentCostEnv.expected_utility``, which is
exactly what the OLD closed-form FD differentiated. Valid because
E[sampled payoff] = closed-form EU (tests/test_different_cost_env_sampled.py)
and expectation is linear.

Run directly (python tests/test_different_cost_mcfd_gradient.py) or via pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.different_cost_env import DifferentCostEnv
from run.run_different_cost import _compute_gradients_different_cost

PARAMS = {"w_h": 8.0, "w_l": 5.5, "k1": 0.0004, "k2": 0.00055, "q": 35.0}
EFFORT_BOUNDS = (0.0, 100.0)
DELTA = 0.5
NUM_SAMPLES = 8192
N_REPS = 50
SEED = 123


def _closed_form_fd(env: DifferentCostEnv, e1: float, e2: float) -> list:
    """Central difference of closed-form EU_i w.r.t. e_i at the same DELTA."""
    g1 = (
        env.expected_utility(e_self=e1 + DELTA, e_opp=e2, k_self=env.k1)
        - env.expected_utility(e_self=e1 - DELTA, e_opp=e2, k_self=env.k1)
    ) / (2.0 * DELTA)
    g2 = (
        env.expected_utility(e_self=e2 + DELTA, e_opp=e1, k_self=env.k2)
        - env.expected_utility(e_self=e2 - DELTA, e_opp=e1, k_self=env.k2)
    ) / (2.0 * DELTA)
    return [g1, g2]


def _check_profile(efforts: tuple) -> None:
    env = DifferentCostEnv(**PARAMS, effort_bounds=EFFORT_BOUNDS, seed=SEED)
    e1, e2 = float(efforts[0]), float(efforts[1])
    ref = _closed_form_fd(env, e1, e2)
    # Independent CRN batches: the env RNG advances across calls
    reps = np.array([
        _compute_gradients_different_cost(env, e1, e2, DELTA, NUM_SAMPLES)
        for _ in range(N_REPS)
    ])
    mean = reps.mean(axis=0)
    se = reps.std(axis=0, ddof=1) / math.sqrt(N_REPS)
    for i in range(2):
        tol = max(6.0 * se[i], 2e-3)
        diff = abs(mean[i] - ref[i])
        print(
            f"  efforts={efforts} g{i + 1}: sampled_mean={mean[i]:+.5f} "
            f"closed_form_fd={ref[i]:+.5f} |diff|={diff:.5f} tol={tol:.5f}"
        )
        assert diff < tol, (
            f"dc MC-FD gradient deviates from closed-form FD: player {i + 1}, "
            f"efforts={efforts}, diff={diff:.5f} >= tol={tol:.5f}"
        )


def test_at_analytical_equilibrium():
    # (e1*, e2*) = (38.03, 27.66) at q=35; both gradients should be ~0 there
    _check_profile((38.03, 27.66))


def test_symmetric_profile():
    _check_profile((30.0, 30.0))


if __name__ == "__main__":
    print("[dc MC-FD] equivalence-in-expectation test (sampled FD vs closed-form FD)")
    test_at_analytical_equilibrium()
    test_symmetric_profile()
    print("[dc MC-FD] PASS")
