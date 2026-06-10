"""Equivalence-in-expectation test for the het-ability sampled MC-FD gradient.

Audit follow-up safeguard: the ported MC-FD baseline gradient
(_compute_gradients_different_ability: sampled payoffs + CRN, central
differences over ability-shifted outputs y_i = e_i + l_i + eps_i) must,
averaged over many independent CRN batches, match the central finite
difference of the closed-form expected utility at the same delta — the
(kept, eval-only) ``DifferentAbilityEnv.compute_utility``, which is exactly
what the OLD closed-form FD differentiated. Valid because
E[sampled payoff] = closed-form EU (tests/test_different_ability_env_sampled.py)
and expectation is linear.

Run directly (python tests/test_different_ability_mcfd_gradient.py) or via pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.different_ability_env import DifferentAbilityEnv
from run.run_different_ability import _compute_gradients_different_ability

CONFIG = {
    "l1": 10.0,
    "l2": 5.0,
    "k": 0.0005,
    "k1": 0.0005,
    "k2": 0.0005,
    "q": 35.0,
    "w_h": 6.5,
    "w_l": 3.0,
    "effort_range": (0.0, 100.0),
    "seed": 123,
}
DELTA = 0.5
NUM_SAMPLES = 8192
N_REPS = 50


def _closed_form_fd(env: DifferentAbilityEnv, e1: float, e2: float) -> list:
    """Central difference of closed-form EU_i w.r.t. e_i at the same DELTA."""
    u1p, _ = env.compute_utility(0, e1 + DELTA, e2)
    u1m, _ = env.compute_utility(0, e1 - DELTA, e2)
    u2p, _ = env.compute_utility(1, e2 + DELTA, e1)
    u2m, _ = env.compute_utility(1, e2 - DELTA, e1)
    return [(u1p - u1m) / (2.0 * DELTA), (u2p - u2m) / (2.0 * DELTA)]


def _check_profile(efforts: tuple) -> None:
    env = DifferentAbilityEnv(dict(CONFIG))
    e1, e2 = float(efforts[0]), float(efforts[1])
    ref = _closed_form_fd(env, e1, e2)
    # Independent CRN batches: the env RNG advances across calls
    reps = np.array([
        _compute_gradients_different_ability(env, e1, e2, DELTA, NUM_SAMPLES)
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
            f"da MC-FD gradient deviates from closed-form FD: player {i + 1}, "
            f"efforts={efforts}, diff={diff:.5f} >= tol={tol:.5f}"
        )


def test_at_equilibrium():
    # e* = 46.43 at q=35 (symmetric); gradients should be ~0 there
    _check_profile((46.43, 46.43))


def test_asymmetric_profile():
    _check_profile((40.0, 50.0))


if __name__ == "__main__":
    print("[da MC-FD] equivalence-in-expectation test (sampled FD vs closed-form FD)")
    test_at_equilibrium()
    test_asymmetric_profile()
    print("[da MC-FD] PASS")
