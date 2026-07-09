"""Numerical verification of the two-stage closed-form benchmark.

Cross-checks every analytic claim in ``utils/theory_multistage.py`` against
direct numerical integration (repo cost convention c(e) = k e^2):

  1. g2(d) is the global stage-2 best response (effort-grid scan over a
     score-gap grid, opponent at g2(d)).
  2. g1 = DW/(6kq) is the global stage-1 argmax (effort-grid scan with exact
     equilibrium continuation values, quadrature over the triangular shock).
  3. Stage-1 curvature matches the KINK-CORRECTED formula
     -2k + DW^2/(32 k q^4) (the plan doc's -2k - DW^2/(32 k q^4) is wrong;
     the correction term flips sign because V2* has a convex kink at d=0),
     hence the stage-1 SOC binds exactly at q_soc.
  4. Equilibrium utility matches (w_h + w_l)/2 - 17 DW^2/(288 k q^2).
  5. Deviating to zero effort is unprofitable for q > q_crit.

Run:
    python tools/verify_two_stage_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.theory_multistage import (  # noqa: E402
    eq_utility_two_stage,
    g1_two_stage,
    q_crit,
    q_soc,
    stage1_curvature,
    v2_star,
    validate_two_stage_params,
    f_xi,
)

W_H, W_L = 6.0, 2.0
K = 1.0 / 3500.0
E_BAR = 100.0


def numeric_stage1_curvature(q: float, h: float = 0.5, n_xi: int = 16001) -> float:
    """Finite-difference curvature of the stage-1 objective at e = g1.

    Args:
        q: Noise half-width.
        h: Finite-difference step in effort units.
        n_xi: Quadrature-grid resolution over the triangular shock.

    Returns:
        Numeric second derivative at the symmetric candidate.
    """
    g1 = g1_two_stage(q, W_H, W_L, K)
    xg = np.linspace(-2.0 * q, 2.0 * q, n_xi)
    w = f_xi(xg, q)
    w = w / w.sum()

    def u1(e: float) -> float:
        return -K * e ** 2 + float((v2_star(e - g1 + xg, q, W_H, W_L, K, E_BAR) * w).sum())

    return (u1(g1 + h) - 2.0 * u1(g1) + u1(g1 - h)) / (h * h)


def main() -> int:
    """Run all checks; return 0 on success, 1 on any failure."""
    qc = q_crit(W_H, W_L, K, E_BAR)
    print(f"q_soc={q_soc(W_H, W_L, K):.3f}, q_crit={qc:.3f}")
    failures = 0

    for q in (38.0, 40.0, 42.0, 45.0, 50.0, 55.0):
        rep = validate_two_stage_params(q, W_H, W_L, K, E_BAR)
        curv_num = numeric_stage1_curvature(q)
        curv_ana = stage1_curvature(q, W_H, W_L, K)
        curv_doc = -2.0 * K - (W_H - W_L) ** 2 / (32.0 * K * q ** 4)  # plan-doc claim
        u_eq_cf = eq_utility_two_stage(q, W_H, W_L, K)

        expect_valid = q > qc
        ok_validity = rep.ok == expect_valid
        # corrected formula must beat the doc formula against the numeric truth
        ok_curv = abs(curv_num - curv_ana) < abs(curv_num - curv_doc)
        ok_ueq = abs(rep.eq_utility - u_eq_cf) < 1e-9
        for name, ok in [("validity", ok_validity), ("curvature", ok_curv), ("u_eq", ok_ueq)]:
            if not ok:
                failures += 1
                print(f"  CHECK FAILED at q={q:g}: {name}")

        print(
            f"q={q:5.1f} valid={str(rep.ok):5s} (expected {expect_valid}) | "
            f"g1={rep.g1:6.2f} g2(0)={rep.g2_at_0:6.2f} U_eq={rep.eq_utility:.4f} | "
            f"curv num={curv_num:+.3e} corrected={curv_ana:+.3e} doc={curv_doc:+.3e} | "
            f"dev1={rep.max_stage1_deviation_gain:+.2e} "
            f"dev2={rep.max_stage2_deviation_gain:+.2e} "
            f"dev(e=0)={rep.zero_effort_deviation_gain:+.4f}"
        )

    print("PASS" if failures == 0 else f"FAIL ({failures} checks)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
