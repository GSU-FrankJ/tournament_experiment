#!/usr/bin/env python3
"""Recompute the normal-model pass probability of gate G-S (v2.0, decision D4).

Sibling of ``tools/v2/v1_0_pass_probability.py``. Input: the signed stage-1 relative errors of the
R1 arm ``B_expcont`` on the development seeds (``results/v2_refine/analysis/stage1_per_run.csv``),
which the v2.0 rehearsal must reproduce bit-identically (check R2). Per q the model is
``signed error ~ N(mean, SD)`` fitted to the 10 runs (SD with ddof = 1, ddof = 0 also reported); a
run passes G-S iff ``|error| <= 0.05``; G-A, G-F and G-N are treated as passing (as in the v1.1
round). The probability that at least 18 of 20 fresh runs pass is a binomial tail,
``P(Binomial(20, p_q) >= 18)``; the probability that both q reach the rule is the product.

Usage: python tools/v2/v2_0_pass_probability.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "v2_refine" / "analysis" / "stage1_per_run.csv"
OUT = ROOT / "results" / "v2_T2_locked" / "v2_0" / "v2_0_pass_probability.json"
ARM = "B_expcont"
THRESHOLD = 0.05
N_FRESH, K_NEEDED = 20, 18
PI_REFERENCE = {"P_ge18_of_20": {50: 0.996, 60: 0.9995}, "within_0.05": {50: 10, 60: 10},
                "max_abs_err": {50: 0.0407, 60: 0.0314}, "mean_signed": {50: -0.0118, 60: 0.0009},
                "sd_ddof1": {50: 0.0178, 60: 0.0187}}


def p_pass(mean: float, sd: float, thr: float = THRESHOLD) -> float:
    """P(|N(mean, sd)| <= thr)."""
    return float(norm.cdf(thr, mean, sd) - norm.cdf(-thr, mean, sd))


def p_rule(p: float, n: int = N_FRESH, k: int = K_NEEDED) -> float:
    """P(Binomial(n, p) >= k)."""
    return float(binom.sf(k - 1, n, p))


def main() -> int:
    d = pd.read_csv(SRC)
    d = d[d.arm == ARM]
    out: Dict = {"source": str(SRC.relative_to(ROOT)), "arm": ARM, "threshold": THRESHOLD,
                 "rule": f">= {K_NEEDED} of {N_FRESH} fresh runs pass G-S", "per_q": {},
                 "pi_reference": PI_REFERENCE}
    p1, p0 = {}, {}
    for q in (50, 60):
        x = d[d.q == q].stage1_rel_err_signed.astype(float).to_numpy()
        m, s1, s0 = float(x.mean()), float(x.std(ddof=1)), float(x.std(ddof=0))
        p1[q], p0[q] = p_pass(m, s1), p_pass(m, s0)
        out["per_q"][str(q)] = {
            "n": int(x.size), "n_within_threshold": int((np.abs(x) <= THRESHOLD).sum()),
            "max_abs_err": float(np.abs(x).max()), "mean_signed": m, "sd_ddof1": s1, "sd_ddof0": s0,
            "per_run_pass_prob_normal_ddof1": p1[q], "per_run_pass_prob_normal_ddof0": p0[q],
            "P_ge18_of_20_normal_ddof1": p_rule(p1[q]), "P_ge18_of_20_normal_ddof0": p_rule(p0[q])}
    out["P_both_normal_ddof1"] = float(np.prod([p_rule(p1[q]) for q in (50, 60)]))
    out["P_both_normal_ddof0"] = float(np.prod([p_rule(p0[q]) for q in (50, 60)]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
