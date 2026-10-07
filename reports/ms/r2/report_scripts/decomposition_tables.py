#!/usr/bin/env python3
"""Markdown tables of ``reports/ms/r2/01_decomposition.md``, read from ``results/ms_r2/decomposition/`` (written by
``tools/ms/r2_decomposition.py``); no number is typed by hand.

Usage (repository root):
    python reports/ms/r2/report_scripts/decomposition_tables.py --block premise|arms|paired|facts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

D = Path("results/ms_r2/decomposition")
ARMS = ["MS_base", "MS_base2400", "MS_rule", "MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5"]


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(out)


def block_premise() -> str:
    """The six rows of the preamble table: preamble value, reproduced value (2 decimals), match."""
    c = json.loads((D / "decomposition_checks.json").read_text())
    rows = []
    for t in c["table"]:
        rows.append({"q": t["q"], "arm": f"`{t['arm']}`", "preamble (sigma, gap, smoothing, remainder)": ", ".join(f"{x:.2f}" for x in t["preamble"]),
                     "reproduced (2 dp)": ", ".join(f"{x:.2f}" for x in t["rounded_2dp"]),
                     "reproduced (4 dp)": ", ".join(f"{x:.4f}" for x in t["reproduced"]), "match": "yes" if t["match_2dp"] else "NO"})
    return _md(pd.DataFrame(rows).sort_values(["q"], kind="stable"))


def block_arms() -> str:
    """Means over the 10 seeds per arm and q: sigma_2(0), gap, smoothing part, remainder, remainder SD and median."""
    t = pd.read_csv(D / "decomposition_table.csv")
    rows = []
    for q in (50, 60):
        for arm in ARMS:
            r = t[(t.arm == arm) & (t.q == q)].iloc[0]
            rows.append({"q": q, "arm": f"`{arm}`", "sigma_2(0)": f"{r.sigma:.2f}", "gap": f"{r.gap:.2f}", "smoothing part": f"{r.smoothing:.2f}",
                         "remainder": f"{r.remainder:.2f}", "remainder median": f"{r.remainder_median:.2f}", "remainder seed SD": f"{r.remainder_sd:.2f}"})
    return _md(pd.DataFrame(rows))


def block_paired() -> str:
    """Paired differences (budget: MS_base2400 - MS_base; sampler: arm - MS_base2400) with percentile bootstrap intervals."""
    p = pd.read_csv(D / "decomposition_paired.csv")
    rows = []
    for r in p.itertuples():
        rows.append({"comparison": r.comparison, "arm - baseline": f"`{r.arm}` - `{r.baseline}`", "q": r.q, "quantity": r.quantity,
                     "mean [95 % CI]": f"{r.mean:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", "interval contains 0": "yes" if r.contains_0 else "no"})
    return _md(pd.DataFrame(rows))


def block_facts() -> str:
    """The checks of section 2.4 and the statements of the preamble, as a list of facts with their values."""
    c = json.loads((D / "decomposition_checks.json").read_text())
    f = [f"- runs: {c['n_runs']} (role `ms_arm`, status done); ratio smoothing part / [e2*(0) sigma_2(0) / (sqrt(pi) q)]: min {c['ratio_min']:.5f}, "
         f"max {c['ratio_max']:.5f}; runs outside [0.99, 1.01]: {len(c['ratio_outside_limits'])}",
         f"- table of the preamble reproduced to its second decimal: {'yes' if c['table_reproduced'] else 'NO'}",
         f"- learned peak below e2*(0): {c['learned_peak_below_closed_form']} of {c['n_runs']} runs; median remainder of `MS_s35a5` at q = 50: "
         f"{c['median_remainder_MS_s35a5_q50']:+.3f}",
         f"- seed SD of the remainder over the 14 (arm, q) cells: {c['remainder_seed_sd_range'][0]:.2f} to {c['remainder_seed_sd_range'][1]:.2f}",
         f"- the four sampler arms against `MS_base2400`: largest |change of the smoothing part| {c['sampler_smoothing_max_abs']:.3f}; change of the remainder "
         f"{c['sampler_remainder_range_q50'][0]:+.2f} to {c['sampler_remainder_range_q50'][1]:+.2f} at q = 50 (every interval contains 0: "
         f"{c['sampler_remainder_all_ci_contain_0_q50']}), {c['sampler_remainder_range_q60'][0]:+.2f} to {c['sampler_remainder_range_q60'][1]:+.2f} at q = 60 "
         f"({c['sampler_remainder_all_ci_contain_0_q60']})"]
    for q in (50, 60):
        s, sm, rm = c[f"budget_sigma_effort_at_0_t2_q{q}"], c[f"budget_smoothing_q{q}"], c[f"budget_remainder_q{q}"]
        f.append(f"- budget (`MS_base2400` - `MS_base`), q = {q}: sigma_2(0) {s[0]:+.3f} [{s[1]:+.3f}, {s[2]:+.3f}]; smoothing part {sm[0]:+.3f} "
                 f"[{sm[1]:+.3f}, {sm[2]:+.3f}]; remainder {rm[0]:+.3f} [{rm[1]:+.3f}, {rm[2]:+.3f}]")
    if "skewness_range" in c:
        f.append(f"- learned Beta noise at d = 0 ({c['recomputed_runs']} runs, alpha and beta from `freeze_stage2_final.npz`): skewness {c['skewness_range'][0]:+.3f} to "
                 f"{c['skewness_range'][1]:+.3f}, excess kurtosis {c['excess_kurtosis_range'][0]:+.3f} to {c['excess_kurtosis_range'][1]:+.3f} (a Gaussian has 0 and 0); "
                 f"e_sigma(0) recomputed from alpha, beta differs from `smoothed_e_pred_0` by at most {c['recomputed_e_sigma_max_abs_diff']:.2e}; the node standard "
                 f"deviation differs from `sigma_effort_at_0_t2` by at most {c['recomputed_sigma_max_abs_diff']:.4f} effort units")
    return "\n".join(f)


BLOCKS = {"premise": block_premise, "arms": block_arms, "paired": block_paired, "facts": block_facts}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--block", required=True, choices=sorted(BLOCKS))
    a = p.parse_args(argv)
    print(BLOCKS[a.block]())
    return 0


if __name__ == "__main__":
    sys.exit(main())
