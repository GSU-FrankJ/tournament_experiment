"""Calibrate the DP verifier on the closed form and run the falsification suite.

Plan section 4.5 / Experiments 3-4: before the verifier is trusted on T>=3
(no closed-form benchmark), confirm on T=2 that

  1. EXP(e*_CF) ~ 0                         -> establishes the error floor,
  2. EXP(bad policy) >> EXP(e*_CF)          -> confirms discriminatory power,
  3. delta_sum_reachable >= EXP for every policy -> the certificate really is
     an upper bound on exploitability,
  4. EXP is stable under score-gap grid refinement (Richardson consistent).

Run:
    python tools/calibrate_verifier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.multi_stage_two_players import config  # noqa: E402
from utils.dp_verifier import verify, verify_grid_refinement  # noqa: E402
from utils.theory_multistage import (  # noqa: E402
    g1_two_stage,
    g2_two_stage,
)

W_H, W_L, K, EBAR = 6.0, 2.0, 1.0 / 3500.0, 100.0
DW = W_H - W_L


def make_policies(q: float):
    """Build the closed-form policy and the falsification policies for T=2.

    Args:
        q: Noise half-width.

    Returns:
        Dict name -> vectorized policy(t, d_array) -> effort_array.
    """
    g1 = g1_two_stage(q, W_H, W_L, K)
    e_one_stage = DW / (4.0 * q * K)  # static tournament effort (denominator 4)

    def closed_form(t, d):
        d = np.asarray(d, dtype=float)
        if t >= 2:
            return g2_two_stage(d, q, W_H, W_L, K, EBAR)
        return np.full_like(d, g1)

    def const_low(t, d):
        return np.full_like(np.asarray(d, dtype=float), 5.0)

    def const_high(t, d):
        return np.full_like(np.asarray(d, dtype=float), EBAR)

    def one_stage_repeated(t, d):
        return np.full_like(np.asarray(d, dtype=float), e_one_stage)

    def no_gap_stage2(t, d):
        # correct stage 1, but stage 2 ignores the gap (uses the d=0 peak)
        d = np.asarray(d, dtype=float)
        if t >= 2:
            return np.full_like(d, float(g2_two_stage(np.asarray(0.0), q, W_H, W_L, K, EBAR)))
        return np.full_like(d, g1)

    def random_mean(t, d):
        # deterministic mean of a Uniform(0, e_bar) random policy
        return np.full_like(np.asarray(d, dtype=float), EBAR / 2.0)

    return {
        "closed_form_CF": closed_form,
        "bad:const_low(5)": const_low,
        "bad:const_high(100)": const_high,
        "bad:one_stage_repeated": one_stage_repeated,
        "bad:no_gap_stage2": no_gap_stage2,
        "bad:random_mean(50)": random_mean,
    }


def main() -> int:
    """Run calibration + falsification at q=50; return 0 on success."""
    q = 50.0
    T = 2
    policies = make_policies(q)
    eps_over_dw = config["verifier"]["epsilon_over_dw"]

    print(f"DP verifier calibration  (T={T}, q={q}, DW={DW}, eps/DW threshold={eps_over_dw})")
    print(f"  EXP        = root exploitability V_1^BR(0) - V_1^e(0)  (plan's certification quantity)")
    print(f"  dReach     = BR-reachable-support Δ-sum (primary certificate, upper-bounds EXP)")
    print(f"  dFull      = full-grid worst-case Δ-sum (robustness; over-states on-path-only policies)")
    print(f"  onpathΔ    = on-path Δ-sum\n")
    print(f"{'policy':<24} {'EXP':>9} {'EXP/DW':>8} {'dReach':>8} {'dReach/DW':>10} "
          f"{'dFull':>8} {'onpathΔ':>9} {'cert?':>6}")

    results = {}
    for name, pol in policies.items():
        r = verify(pol, w_h=W_H, w_l=W_L, k=K, q=q, T=T, e_bar=EBAR,
                   epsilon_over_dw=eps_over_dw)
        results[name] = r
        print(f"{name:<24} {r.exp:>9.4f} {r.exp_over_dw:>8.4f} {r.delta_sum_reachable:>8.4f} "
              f"{r.delta_sum_reachable / DW:>10.4f} {r.delta_sum_full:>8.4f} "
              f"{r.delta_onpath_sum:>9.4f} {str(r.certified):>6}")

    cf = results["closed_form_CF"]
    floor = cf.exp  # error floor is the closed-form root exploitability
    failures = 0

    # Check 1: closed-form error floor near zero (EXP and reachable certificate)
    if not (cf.exp_over_dw < 0.01 and cf.delta_sum_reachable / DW < eps_over_dw):
        failures += 1
        print(f"  CHECK FAILED: closed-form floor too large "
              f"(EXP/DW={cf.exp_over_dw:.4f}, dReach/DW={cf.delta_sum_reachable / DW:.4f})")

    # Check 2: every bad policy's EXP well above the floor (discriminatory power)
    for name, r in results.items():
        if name.startswith("bad:"):
            if not (r.exp > 10.0 * max(floor, 1e-4)):
                failures += 1
                print(f"  CHECK FAILED: {name} EXP not >> floor "
                      f"(EXP={r.exp:.4f} vs floor {floor:.4f})")

    # Check 3: reachable Δ-sum upper-bounds EXP for every policy
    for name, r in results.items():
        if r.delta_sum_reachable + 1e-4 < r.exp:
            failures += 1
            print(f"  CHECK FAILED: {name} delta_sum_reachable {r.delta_sum_reachable:.4f} "
                  f"< EXP {r.exp:.4f} (bound violated)")

    # Check 4: only the closed form certifies; bad policies do not
    if not cf.certified:
        failures += 1
        print("  CHECK FAILED: closed form not certified")
    for name, r in results.items():
        if name.startswith("bad:") and r.certified:
            failures += 1
            print(f"  CHECK FAILED: {name} wrongly certified")

    # Check 5: grid refinement / Richardson on the closed form
    ref = verify_grid_refinement(policies["closed_form_CF"], w_h=W_H, w_l=W_L,
                                 k=K, q=q, T=T, e_bar=EBAR)
    print("\ngrid refinement (closed form):")
    for m, e, ub in zip(ref["d_grid_sizes"], ref["exp"], ref["delta_sum_reachable"]):
        print(f"  M={m:>4}: EXP={e:.5f}  dReach={ub:.5f}")
    print(f"  Richardson EXP -> {ref['exp_richardson']:.5f}")
    if not (abs(ref["exp_richardson"]) < 0.02 and max(abs(e) for e in ref["exp"]) < 0.05):
        failures += 1
        print("  CHECK FAILED: EXP not stable/near-zero under refinement")

    print("\nPASS" if failures == 0 else f"\nFAIL ({failures} checks)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
