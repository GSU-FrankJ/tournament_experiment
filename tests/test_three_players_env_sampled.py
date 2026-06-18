"""Equivalence-in-expectation test for the sampled ThreePlayersEnv rewards.

Audit fix, target 1 safeguard: the rewritten env returns SAMPLED rank-order
rewards (y_i = e_i + eps_i, eps ~ U(-q, q); winner gets w_H, others w_L; minus
k e_i^2). Averaged over many draws, these must match the closed-form expected
utility the OLD env returned in its default "expected" reward mode:

    EU_i = w_L + p_i * (w_H - w_L) - k e_i^2,
    p_i  = win_prob_three_players(e_i, e_j, e_k, q), normalized over players
           (the pre-fix env normalized p1+p2+p3 to 1; see git history of
           envs/three_players_env.py, lines 66-76 / 210-214).

Run directly (python tests/test_three_players_env_sampled.py) or via pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from envs.three_players_env import ThreePlayersEnv
from utils.prob import win_prob_three_players

# Paper parameters (3P scenario): k=0.001, w=(6.5, 3.0), q=35 -> e* = 25.0
PARAMS = {"w_h": 6.5, "w_l": 3.0, "k": 0.001, "q": 35.0}
EFFORT_BOUNDS = (0.0, 100.0)
N_SAMPLES = 120_000
SEED = 123


def closed_form_expected_utilities(efforts: list, p: dict) -> list:
    """Replicate the OLD env's closed-form 'expected' reward exactly."""
    e1, e2, e3 = efforts
    probs = [
        win_prob_three_players(e1, e2, e3, p["q"]),
        win_prob_three_players(e2, e1, e3, p["q"]),
        win_prob_three_players(e3, e1, e2, p["q"]),
    ]
    total = sum(probs)
    if total > 0.0:
        probs = [x / total for x in probs]
    else:
        probs = [1.0 / 3.0] * 3
    gap = p["w_h"] - p["w_l"]
    return [p["w_l"] + probs[i] * gap - p["k"] * efforts[i] ** 2 for i in range(3)]


def _sample_mean_rewards(efforts: tuple, n: int = N_SAMPLES, seed: int = SEED) -> np.ndarray:
    env = ThreePlayersEnv(**PARAMS, effort_bounds=EFFORT_BOUNDS, seed=seed)
    actions = tuple(torch.tensor([float(e)]) for e in efforts)
    out = np.empty((n, 3), dtype=np.float64)
    for t in range(n):
        _, rewards, _, _, _ = env.step(actions)
        out[t] = rewards.numpy()
    return out


def _check_profile(efforts: tuple) -> None:
    samples = _sample_mean_rewards(efforts)
    ref = closed_form_expected_utilities([float(e) for e in efforts], PARAMS)
    mean = samples.mean(axis=0)
    se = samples.std(axis=0, ddof=1) / math.sqrt(samples.shape[0])
    for i in range(3):
        tol = max(6.0 * se[i], 0.02)
        diff = abs(mean[i] - ref[i])
        print(
            f"  efforts={efforts} player{i + 1}: sampled_mean={mean[i]:.4f} "
            f"closed_form={ref[i]:.4f} |diff|={diff:.4f} tol={tol:.4f}"
        )
        assert diff < tol, (
            f"3P sampled reward mean deviates from old closed-form EU: "
            f"player {i + 1}, efforts={efforts}, diff={diff:.4f} >= tol={tol:.4f}"
        )


def test_symmetric_at_equilibrium():
    _check_profile((25.0, 25.0, 25.0))


def test_asymmetric_profile():
    _check_profile((20.0, 25.0, 30.0))


def test_spread_profile():
    _check_profile((5.0, 50.0, 95.0))


if __name__ == "__main__":
    print("[3P] equivalence-in-expectation test (sampled vs old closed-form)")
    test_symmetric_at_equilibrium()
    test_asymmetric_profile()
    test_spread_profile()
    print("[3P] PASS")
