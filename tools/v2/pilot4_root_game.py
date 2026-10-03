#!/usr/bin/env python3
"""Pilot 4 section 1a: numeric check of the root stage game's BR slope and own curvature.

Q_1(0, e | opponent e_opp) is the verifier's own stage-1 Q^mean (``utils.v2_metrics.stage1_Q``):
-k e^2 + sum_x w_x interp(e - e_opp + x; D_2 grid, V_2^mean), with V_2^mean from ``verify`` on the
closed-form candidate (e1*, e2*) for both players. V_2^mean does not depend on stage 1.

BR(e_opp): grid argmax of Q_1(., e_opp) on the verifier's effort grid E, then a least-squares
quadratic a e^2 + b e + c fitted to Q_1 at the effort-grid nodes within +-W of the grid argmax;
BR = -b / (2a). W in FIT_HALF_WIDTHS (in effort units; the node count is 2W/step + 1).
Slope: central differences (BR(e1*+h) - BR(e1*-h)) / 2h for h in H_STEPS.
Own curvature: 2a of the same fit at e_opp = e1*, reported as 2a / (2k); E[V_2'']/(2k) = 1 + 2a/(2k).

Tiers: 'final' (requested) and 'fine' (state 0.5, effort 0.25, GL 64 per half; robustness only).
Evaluation only (closed form enters nothing that trains).

Usage: python tools/v2/pilot4_root_game.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from common import analytic_policy, spec_for  # noqa: E402
from utils.dp_br_verifier import FINAL_CONFIG, VerifierConfig, verify  # noqa: E402
from utils.v2_metrics import stage1_Q  # noqa: E402

OUT = ROOT / "results" / "v2_pilots" / "pilot4" / "analysis" / "root_game"
FINE = VerifierConfig("fine", state_step=0.5, effort_step=0.25, gl_half=64)
FIT_HALF_WIDTHS = (1.0, 2.0, 4.0)
H_STEPS = (0.25, 0.5, 1.0, 2.0, 4.0)
REFERENCE = {50: {"slope": -0.961, "ev2_over_2k": 0.490}, 60: {"slope": -0.309, "ev2_over_2k": 0.236}}


def quad_fit(Q, E: np.ndarray, e_opp: float, W: float):
    """(BR, 2a, grid argmax, n points) from the local quadratic fit around the grid maximum."""
    vals = Q(E, e_opp)
    j = int(np.argmax(vals))
    m = np.abs(E - E[j]) <= W + 1e-12
    x, y = E[m], vals[m]
    a, b, _ = np.polyfit(x - E[j], y, 2)       # centred for conditioning
    return float(E[j] - b / (2 * a)), float(2 * a), float(E[j]), int(m.sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, curv = [], []
    for q in (50, 60):
        spec = spec_for(q)
        eq = analytic_policy(spec)
        g1 = float(eq(1, np.zeros(1))[0])
        for cfg in (FINAL_CONFIG, FINE):
            res = verify(eq, w_h=spec.w_h, w_l=spec.w_l, k=spec.k, q=spec.q, T=2, e_min=spec.e_min,
                         e_max=spec.e_max, cfg=cfg)
            Q = stage1_Q(res, spec.k)
            E = res.e_grid
            for W in FIT_HALF_WIDTHS:
                br0, c2, jg, n = quad_fit(Q, E, g1, W)
                r = c2 / (2 * spec.k)
                curv.append({"q": q, "tier": cfg.name, "fit_half_width": W, "fit_points": n, "e1_star": g1,
                             "BR_at_e1star": br0, "BR_minus_e1star": br0 - g1, "grid_argmax": jg,
                             "own_curv_2a": c2, "own_curv_over_2k": r, "ev2pp_over_2k": 1.0 + r,
                             "slope_implied_by_curv": -(1.0 + r) / (-r),
                             "ref_ev2pp_over_2k": REFERENCE[q]["ev2_over_2k"],
                             "ref_own_curv_over_2k": -(1 - REFERENCE[q]["ev2_over_2k"])})
                for h in H_STEPS:
                    bp = quad_fit(Q, E, g1 + h, W)[0]
                    bm = quad_fit(Q, E, g1 - h, W)[0]
                    rows.append({"q": q, "tier": cfg.name, "fit_half_width": W, "h": h, "BR_plus": bp, "BR_minus": bm,
                                 "slope": (bp - bm) / (2 * h), "ref_slope": REFERENCE[q]["slope"]})
    sl = pd.DataFrame(rows)
    cv = pd.DataFrame(curv)
    sl.to_csv(OUT / "br_slope.csv", index=False)
    cv.to_csv(OUT / "own_curvature.csv", index=False)
    pd.set_option("display.width", 200)
    print(cv.to_string())
    print(sl.pivot_table(index=["q", "tier", "fit_half_width"], columns="h", values="slope").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
