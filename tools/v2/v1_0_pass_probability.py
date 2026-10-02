#!/usr/bin/env python3
"""Recompute the PI-side estimate of the v1.0 confirmation pass probability (both q >= 18/20).

Input: the v1.0 rehearsal (results/v2_T2_locked/rehearsal_analysis/gates_per_run.csv).
  (a) rate model: true per-run pass probability = the rehearsal pass rate per q (9/10, 8/10);
  (b) normal model: signed stage-1 relative error ~ N(mean, SD) fitted to the rehearsal per q; a run
      passes iff |error| <= 0.10 (G-A and the Gmax criterion treated as always passing, as in 20/20
      rehearsal runs); reported with SD ddof=1 and ddof=0 and with the PI's rounded inputs.
P(pass) = prod_q P(Binomial(20, p_q) >= 18). Also: SD and 90th percentile of |error| per q.

Usage: python tools/v2/v1_0_pass_probability.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "v2_T2_locked" / "rehearsal_analysis" / "gates_per_run.csv"
OUT = ROOT / "results" / "v2_T2_locked" / "v1_1" / "v1_0_pass_probability.json"
PI = {"rate": {50: 0.9, 60: 0.8}, "mean": {50: -0.014, 60: -0.002}, "sd": {50: 0.052, 60: 0.070},
      "p_rate": 0.14, "p_normal": 0.35, "p90_abs": {50: 0.089, 60: 0.114}}


def p_both(p: dict) -> float:
    return float(np.prod([binom.sf(17, 20, p[q]) for q in (50, 60)]))


def main() -> int:
    g = pd.read_csv(SRC)
    out = {"source": str(SRC.relative_to(ROOT)), "per_q": {}, "pi_reference": PI}
    rate, pn1, pn0, pnpi = {}, {}, {}, {}
    for q in (50, 60):
        x = g[g.q == q].B_stage1_rel_err_signed.astype(float).to_numpy()
        rp = float(g[g.q == q].run_pass.mean())
        m, s1, s0 = float(x.mean()), float(x.std(ddof=1)), float(x.std(ddof=0))
        rate[q] = rp
        pn1[q] = float(norm.cdf(0.10, m, s1) - norm.cdf(-0.10, m, s1))
        pn0[q] = float(norm.cdf(0.10, m, s0) - norm.cdf(-0.10, m, s0))
        pnpi[q] = float(norm.cdf(0.10, PI["mean"][q], PI["sd"][q]) - norm.cdf(-0.10, PI["mean"][q], PI["sd"][q]))
        out["per_q"][str(q)] = {"n": int(x.size), "rehearsal_pass_rate": rp, "mean_signed": m, "sd_ddof1": s1, "sd_ddof0": s0,
                                "p90_abs_err_linear_interp": float(np.quantile(np.abs(x), 0.9)),
                                "per_run_pass_prob_normal_ddof1": pn1[q], "per_run_pass_prob_normal_ddof0": pn0[q],
                                "per_run_pass_prob_normal_PI_inputs": pnpi[q],
                                "P_ge18_of_20_rate": float(binom.sf(17, 20, rp)),
                                "P_ge18_of_20_normal_ddof1": float(binom.sf(17, 20, pn1[q]))}
    out["P_both_rate_model"] = p_both(rate)
    out["P_both_normal_ddof1"] = p_both(pn1)
    out["P_both_normal_ddof0"] = p_both(pn0)
    out["P_both_normal_PI_inputs"] = p_both(pnpi)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
