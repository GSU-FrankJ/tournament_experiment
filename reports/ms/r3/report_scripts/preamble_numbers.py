#!/usr/bin/env python3
"""Reproduction of the numbers of items 1-3 of the MS-R3 prompt preamble (section 2.4), from the MS-R2 analysis table.

The numbers come from the arm means of ``results/ms_r2/analysis/per_run.csv`` of ``origin/ms-r2`` (columns ``e2_at_0``,
``gap``, ``smoothing``, ``g2_at_0``). Each table prints the quoted value next to the reproduced one; a quoted value
that does not equal the reproduced one at the precision it was quoted with is marked ``differs``.

Usage (repository root):
    python reports/ms/r3/report_scripts/preamble_numbers.py --block tie_effort|sampler|budget|slope|models
        [--per-run results/ms_r2/analysis/per_run.csv]
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ARMS = ["NL_bb_s1", "NL_bb_s4", "NL_bb_s16", "NL_st_s1", "NL_st_s4", "NL_st_s16"]
QS = [50, 60]
PER_RUN = "results/ms_r2/analysis/per_run.csv"


def _md(rows: List[Dict[str, object]]) -> str:
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def _load(path: str) -> pd.DataFrame:
    p = pd.read_csv(path, float_precision="round_trip", low_memory=False)
    return p[p["status"] == "done"].copy()


def _arm_means(p: pd.DataFrame) -> pd.DataFrame:
    return p[p["arm"].isin(ARMS)].groupby(["arm", "q"])[["e2_at_0", "gap", "smoothing", "g2_at_0"]].mean()


def _flag(quoted: float, got: float, dp: int) -> str:
    """``ok`` when ``round(got, dp)`` equals the quoted value, else ``differs``."""
    return "ok" if abs(round(got, dp) - quoted) < 0.5 * 10 ** (-dp) * 0.01 else "differs"


def block_tie_effort(p: pd.DataFrame) -> str:
    """Seed means of e_hat_2(0) at s = 1 / 4 / 16 (the preamble table)."""
    m = _arm_means(p)
    quoted = {("bb", 50): "66.04 / 66.44 / 66.10", ("st", 50): "66.97 / 66.87 / 67.19",
              ("bb", 60): "56.00 / 55.33 / 55.58", ("st", 60): "55.64 / 55.99 / 55.91"}
    rows = []
    for q in QS:
        r: Dict[str, object] = {"q": q}
        for st, lab in (("bb", "bin-balanced"), ("st", "stratified")):
            got = " / ".join(f"{m.loc[(f'NL_{st}_s{s}', q), 'e2_at_0']:.2f}" for s in (1, 4, 16))
            r[f"{lab}: reproduced"] = got
            r[f"{lab}: quoted"] = quoted[(st, q)]
            r[f"{lab}: match"] = "ok" if got == quoted[(st, q)] else "differs"
        rows.append(r)
    return _md(rows)


def block_sampler(p: pd.DataFrame) -> str:
    """Item 1: the stratified sampler's shift of the seed-mean tie effort (st - bb) at s = 1, 4, 16."""
    m = _arm_means(p)
    quoted = {50: "+0.4 to +1.1", 60: "-0.4 to +0.7"}
    rows = []
    for q in QS:
        d = [m.loc[(f"NL_st_s{s}", q), "e2_at_0"] - m.loc[(f"NL_bb_s{s}", q), "e2_at_0"] for s in (1, 4, 16)]
        lo, hi = min(d), max(d)
        rows.append({"q": q, "st - bb at s = 1 / 4 / 16": " / ".join(f"{x:+.3f}" for x in d),
                     "reproduced range (1 dp)": f"{lo:+.1f} to {hi:+.1f}", "quoted": quoted[q],
                     "match": "ok" if f"{lo:+.1f} to {hi:+.1f}" == quoted[q] else "differs"})
    return _md(rows)


def block_budget(p: pd.DataFrame) -> str:
    """Item 1: MS_base2400 against parents_A (1600 -> 2400 updates), seed-mean e_hat_2(0)."""
    quoted = {50: 0.88, 60: 0.50}
    rows = []
    for q in QS:
        a = p[(p["arm"] == "MS_base2400") & (p["q"] == q)]["e2_at_0"].mean()
        b = p[(p["arm"] == "parents_A") & (p["q"] == q)]["e2_at_0"].mean()
        rows.append({"q": q, "MS_base2400": f"{a:.3f}", "parents_A": f"{b:.3f}", "difference": f"{a - b:+.3f}",
                     "quoted": f"{quoted[q]:+.2f}", "match": "ok" if f"{a - b:+.2f}" == f"{quoted[q]:+.2f}" else "differs"})
    return _md(rows)


def block_slope(p: pd.DataFrame) -> str:
    """Item 2: gap / tent slope (slope = e2*(0) / 2q), averaged over the six MS-R2 arms (60 runs per q)."""
    quoted = {50: 4.85, 60: 5.33}
    rows = []
    for q in QS:
        g = p[p["arm"].isin(ARMS) & (p["q"] == q)]
        w = g["gap"] / (g["g2_at_0"] / (2.0 * q))
        per_arm = ", ".join(f"{a} {w[g['arm'] == a].mean():.3f}" for a in ARMS)
        rows.append({"q": q, "mean over the 60 runs": f"{w.mean():.3f}", "arm means": per_arm,
                     "quoted": f"{quoted[q]:.2f}", "match": "ok" if abs(w.mean() - quoted[q]) < 0.0051 else "differs"})
    return _md(rows)


def block_models(p: pd.DataFrame) -> str:
    """Item 3: additive and quadrature predictions of the gap at s = 16 against the observed one (arm means)."""
    m = _arm_means(p)
    quoted = {"additive": ["2.57", "1.36", "1.66", "1.71"], "observed": ["3.90", "2.75", "2.81", "2.42"],
              "quadrature": ["3.51", "1.95", "2.44", "2.36"]}
    cells = [("bb", 50), ("bb", 60), ("st", 50), ("st", 60)]
    add, obs, quad, flags = [], [], [], []
    for st, q in cells:
        a, b = m.loc[(f"NL_{st}_s1", q)], m.loc[(f"NL_{st}_s16", q)]
        add.append(a["gap"] - (a["smoothing"] - b["smoothing"]))
        f2 = a["gap"] ** 2 - a["smoothing"] ** 2
        flags.append("F^2 < 0" if f2 < 0 else "")
        quad.append(float(np.sqrt(max(f2, 0.0) + b["smoothing"] ** 2)))
        obs.append(b["gap"])
    rows = []
    for name, vals in (("additive", add), ("quadrature", quad), ("observed", obs)):
        got = [f"{v:.2f}" for v in vals]
        rows.append({"gap(s = 16)": name, "bin-balanced q50, q60; stratified q50, q60 (reproduced)": " / ".join(got),
                     "quoted": " / ".join(quoted[name]), "match": "ok" if got == quoted[name] else "differs"})
    below = all(qv < ov for qv, ov in zip(quad, obs))
    closer = all(abs(qv - ov) < abs(av - ov) for qv, ov, av in zip(quad, obs, add))
    note = (f"\nQuadrature prediction below the observed gap in every cell: {below}; closer than the additive one in every "
            f"cell: {closer}; F^2 < 0 flags: {[f or 'none' for f in flags]}.")
    return _md(rows) + note


BLOCKS: Dict[str, Callable[[pd.DataFrame], str]] = {
    "tie_effort": block_tie_effort, "sampler": block_sampler, "budget": block_budget, "slope": block_slope,
    "models": block_models}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--block", required=True, choices=sorted(BLOCKS))
    ap.add_argument("--per-run", default=PER_RUN)
    a = ap.parse_args(argv)
    print(BLOCKS[a.block](_load(a.per_run)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
