"""Equivalence-in-expectation test for the 3-player sampled MC-FD gradient.

Audit follow-up safeguard: the ported MC-FD baseline gradient
(_stochastic_fd_gradients_3p: sampled payoffs + CRN, central differences)
must, averaged over many independent CRN batches, match the central
finite difference of the closed-form expected utility at the same delta:

    E[ (u_i(e_i+d) - u_i(e_i-d)) / 2d ]  =  (EU_i(e_i+d) - EU_i(e_i-d)) / 2d

because E[sampled payoff] = closed-form EU exactly (established by
tests/test_three_players_env_sampled.py) and expectation is linear.
EU_i(e) = w_L + p_i*(w_H-w_L) - k e_i^2 with p_i = win_prob_three_players
(probabilities sum to 1 exactly, so the old env's normalization is a no-op).

Run directly (python tests/test_three_players_mcfd_gradient.py) or via pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.three_players_env import ThreePlayersEnv
from run.run_three_players import _stochastic_fd_gradients_3p
from utils.prob import win_prob_three_players

PARAMS = {"w_h": 6.5, "w_l": 3.0, "k": 0.001, "q": 35.0}
EFFORT_BOUNDS = (0.0, 100.0)
DELTA = 0.5
NUM_SAMPLES = 8192
N_REPS = 50
SEED = 123


def _eu(e: list, i: int) -> float:
    """Closed-form expected utility of player i (old env's reward formula)."""
    others = [e[j] for j in range(3) if j != i]
    p_i = win_prob_three_players(e[i], others[0], others[1], PARAMS["q"])
    return PARAMS["w_l"] + p_i * (PARAMS["w_h"] - PARAMS["w_l"]) - PARAMS["k"] * e[i] ** 2


def _closed_form_fd(e: list) -> list:
    """Central difference of closed-form EU_i w.r.t. e_i at the same DELTA."""
    grads = []
    for i in range(3):
        ep = list(e)
        em = list(e)
        ep[i] += DELTA
        em[i] -= DELTA
        grads.append((_eu(ep, i) - _eu(em, i)) / (2.0 * DELTA))
    return grads


def _check_profile(efforts: tuple) -> None:
    env = ThreePlayersEnv(**PARAMS, effort_bounds=EFFORT_BOUNDS, seed=SEED)
    e = [float(x) for x in efforts]
    ref = _closed_form_fd(e)
    # Independent CRN batches: the env RNG advances across calls
    reps = np.array([
        _stochastic_fd_gradients_3p(env, e[0], e[1], e[2], delta=DELTA, num_samples=NUM_SAMPLES)
        for _ in range(N_REPS)
    ])
    mean = reps.mean(axis=0)
    se = reps.std(axis=0, ddof=1) / math.sqrt(N_REPS)
    for i in range(3):
        tol = max(6.0 * se[i], 2e-3)
        diff = abs(mean[i] - ref[i])
        print(
            f"  efforts={efforts} g{i + 1}: sampled_mean={mean[i]:+.5f} "
            f"closed_form_fd={ref[i]:+.5f} |diff|={diff:.5f} tol={tol:.5f}"
        )
        assert diff < tol, (
            f"3P MC-FD gradient deviates from closed-form FD: player {i + 1}, "
            f"efforts={efforts}, diff={diff:.5f} >= tol={tol:.5f}"
        )


def test_at_equilibrium():
    # e* = 25.0 at q=35; gradients should be ~0 there
    _check_profile((25.0, 25.0, 25.0))


def test_asymmetric_profile():
    _check_profile((20.0, 25.0, 30.0))


if __name__ == "__main__":
    print("[3P MC-FD] equivalence-in-expectation test (sampled FD vs closed-form FD)")
    test_at_equilibrium()
    test_asymmetric_profile()
    print("[3P MC-FD] PASS")
