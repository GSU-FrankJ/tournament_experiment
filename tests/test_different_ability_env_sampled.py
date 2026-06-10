"""Equivalence-in-expectation test for the sampled DifferentAbilityEnv rewards.

Audit fix, target 1 safeguard: the rewritten env returns SAMPLED rank-order
rewards (y_i = e_i + l_i + eps_i, eps ~ U(-q, q); winner gets w_H, loser w_L;
minus k_i e_i^2). Averaged over many draws, these must match the closed-form
expected utility the OLD env returned as its training reward:

    EU_i = w_L + p_i(win) * (w_H - w_L) - k_i e_i^2

which is exactly the (kept, eval-only) ``DifferentAbilityEnv.compute_utility``.

Run directly (python tests/test_different_ability_env_sampled.py) or via pytest.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from envs.different_ability_env import DifferentAbilityEnv

# Paper parameters (het-ability scenario): l=(10,5), k=0.0005, w=(6.5,3), q=35
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
N_SAMPLES = 120_000


def _sample_mean_rewards(efforts: tuple, n: int = N_SAMPLES) -> np.ndarray:
    env = DifferentAbilityEnv(dict(CONFIG))
    actions = [torch.tensor([float(efforts[0])]), torch.tensor([float(efforts[1])])]
    out = np.empty((n, 2), dtype=np.float64)
    for t in range(n):
        _, rewards, _, _, _ = env.step(actions)
        out[t] = rewards.numpy()
    return out


def _check_profile(efforts: tuple) -> None:
    env_ref = DifferentAbilityEnv(dict(CONFIG))
    e1, e2 = float(efforts[0]), float(efforts[1])
    # Old training reward == closed-form expected utility (kept as eval-only helper)
    u1_ref, _ = env_ref.compute_utility(0, e1, e2)
    u2_ref, _ = env_ref.compute_utility(1, e2, e1)
    ref = [u1_ref, u2_ref]
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
            f"da sampled reward mean deviates from old closed-form EU: "
            f"player {i + 1}, efforts={efforts}, diff={diff:.4f} >= tol={tol:.4f}"
        )


def test_symmetric_at_equilibrium():
    # e* = 46.43 at q=35 (analytical het-ability symmetric equilibrium)
    _check_profile((46.43, 46.43))


def test_asymmetric_profile():
    _check_profile((40.0, 50.0))


def test_low_effort_profile():
    _check_profile((20.0, 30.0))


if __name__ == "__main__":
    print("[da] equivalence-in-expectation test (sampled vs old closed-form)")
    test_symmetric_at_equilibrium()
    test_asymmetric_profile()
    test_low_effort_profile()
    print("[da] PASS")
