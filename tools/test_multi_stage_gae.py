"""Unit tests for the trajectory-aware multi-stage GAE.

Self-checking script (repo has no pytest suite; follows the tools/verify_*.py
convention). Verifies:

  1. Hand-computed 2-step advantages for (gamma, lam) = (1, 0.5).
  2. gamma=lam=1 reduces to Monte Carlo: A_t = sum_{s>=t} r_s - V_t.
  3. Ordering independence: permuting trajectories permutes output blocks
     but never changes a trajectory's own advantages.
  4. The interleaving BUG: the flat one-stage-style GAE gives WRONG results
     on interleaved p0/p1 storage, while the trajectory-aware GAE (fed the
     same data as separate trajectories) gives the correct per-player values.

Run:
    python tools/test_multi_stage_gae.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.ppo_multi_stage import (  # noqa: E402
    compute_gae_single,
    compute_gae_trajectories,
)


def _flat_gae_with_dones(rewards, values, dones, gamma, lam):
    """Reference implementation of the one-stage agent's flat GAE.

    Mirrors PPOTwoPlayersBandit._compute_gae so we can demonstrate the
    interleaving misbootstrap without importing torch-heavy agent code.
    """
    n = rewards.numel()
    adv = torch.zeros(n)
    lastgae = torch.zeros(())
    next_value = torch.zeros(())
    for t in reversed(range(n)):
        mask = (~dones[t]).float()
        delta = rewards[t] + gamma * next_value * mask - values[t]
        lastgae = delta + gamma * lam * mask * lastgae
        adv[t] = lastgae
        next_value = values[t]
    return adv, adv + values


def check(name, cond, detail=""):
    """Print a pass/fail line; return 1 on failure."""
    print(f"[{'OK ' if cond else 'BAD'}] {name}" + (f"  {detail}" if detail and not cond else ""))
    return 0 if cond else 1


def main() -> int:
    """Run all GAE unit checks; return 0 on success."""
    failures = 0
    tol = 1e-6

    # --- 1. hand-computed 2-step, gamma=1, lam=0.5 ---
    r = torch.tensor([2.0, 5.0])
    v = torch.tensor([3.0, 4.0])
    adv, ret = compute_gae_single(r, v, gamma=1.0, gae_lambda=0.5)
    # adv1 = r1 - v1 = 1; adv0 = (r0 + v1 - v0) + 0.5*(r1 - v1) = 3 + 0.5 = 3.5
    exp_adv = torch.tensor([3.5, 1.0])
    exp_ret = exp_adv + v  # [6.5, 5.0]
    failures += check("hand-computed (gamma=1, lam=0.5) advantages",
                      torch.allclose(adv, exp_adv, atol=tol), f"got {adv.tolist()}")
    failures += check("hand-computed returns",
                      torch.allclose(ret, exp_ret, atol=tol), f"got {ret.tolist()}")

    # --- 2. gamma=lam=1 == Monte Carlo return ---
    r = torch.tensor([-1.0, -2.0, 7.0])  # e.g. -cost, -cost, prize-cost
    v = torch.tensor([1.5, 0.3, 6.2])
    adv, ret = compute_gae_single(r, v, gamma=1.0, gae_lambda=1.0)
    mc_ret = torch.tensor([r[0] + r[1] + r[2], r[1] + r[2], r[2]])  # [4, 5, 7]
    failures += check("gamma=lam=1 returns == Monte Carlo",
                      torch.allclose(ret, mc_ret, atol=tol), f"got {ret.tolist()}")
    failures += check("gamma=lam=1 advantages == MC return - value",
                      torch.allclose(adv, mc_ret - v, atol=tol), f"got {adv.tolist()}")

    # --- 3. ordering independence across trajectories ---
    tA_r, tA_v = torch.tensor([1.0, 4.0]), torch.tensor([0.5, 3.0])
    tB_r, tB_v = torch.tensor([2.0, -1.0, 3.0]), torch.tensor([1.0, 0.0, 2.5])
    advAB, _ = compute_gae_trajectories([tA_r, tB_r], [tA_v, tB_v], 1.0, 0.95)
    advBA, _ = compute_gae_trajectories([tB_r, tA_r], [tB_v, tA_v], 1.0, 0.95)
    a_alone, _ = compute_gae_single(tA_r, tA_v, 1.0, 0.95)
    b_alone, _ = compute_gae_single(tB_r, tB_v, 1.0, 0.95)
    order_ok = (
        torch.allclose(advAB[:2], a_alone, atol=tol)
        and torch.allclose(advAB[2:], b_alone, atol=tol)
        and torch.allclose(advBA[:3], b_alone, atol=tol)
        and torch.allclose(advBA[3:], a_alone, atol=tol)
    )
    failures += check("ordering independence (per-trajectory advantages stable)", order_ok)

    # --- 4. interleaving bug demonstration ---
    # Two players, T=2. Storage order INTERLEAVED: p0_s1, p1_s1, p0_s2, p1_s2.
    # rewards: intermediate = -cost, terminal = prize - cost.
    p0 = {"r": torch.tensor([-1.0, 6.0]), "v": torch.tensor([2.0, 5.5])}
    p1 = {"r": torch.tensor([-1.5, 2.0]), "v": torch.tensor([1.8, 1.9])}
    interleaved_r = torch.tensor([p0["r"][0], p1["r"][0], p0["r"][1], p1["r"][1]])
    interleaved_v = torch.tensor([p0["v"][0], p1["v"][0], p0["v"][1], p1["v"][1]])
    interleaved_done = torch.tensor([False, False, True, True])
    flat_adv, _ = _flat_gae_with_dones(interleaved_r, interleaved_v,
                                        interleaved_done, 1.0, 1.0)
    # Correct per-player advantages via trajectory-aware GAE:
    adv_p0, _ = compute_gae_single(p0["r"], p0["v"], 1.0, 1.0)
    adv_p1, _ = compute_gae_single(p1["r"], p1["v"], 1.0, 1.0)
    # Flat GAE's p0-stage-1 advantage sits at index 0; correct value is adv_p0[0].
    # It misbootstraps from p1_s2's value (index 3), so it should DIFFER.
    flat_p0_s1 = flat_adv[0]
    bug_present = not torch.isclose(flat_p0_s1, adv_p0[0], atol=1e-4)
    failures += check("flat GAE misbootstraps on interleaved storage (bug reproduced)",
                      bug_present,
                      f"flat={flat_p0_s1.item():.4f} correct={adv_p0[0].item():.4f}")
    # And the trajectory-aware GAE on the SAME data (as 2 trajectories) is correct:
    traj_adv, _ = compute_gae_trajectories([p0["r"], p1["r"]], [p0["v"], p1["v"]],
                                           1.0, 1.0)
    fix_ok = (
        torch.allclose(traj_adv[:2], adv_p0, atol=tol)
        and torch.allclose(traj_adv[2:], adv_p1, atol=tol)
    )
    failures += check("trajectory-aware GAE gives correct per-player advantages", fix_ok)

    print("\nPASS" if failures == 0 else f"\nFAIL ({failures} checks)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
