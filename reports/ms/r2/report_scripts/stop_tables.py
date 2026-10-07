#!/usr/bin/env python3
"""Markdown tables of the stop-candidate reports (``01b_stop_candidates.md`` and its pilot repeat), read from the CSVs
of ``tools/ms/r2_stop_candidates.py`` (``<dir>/tables/*.csv`` and the per-run CSVs); no number is typed by hand.

Usage (repository root):
    python reports/ms/r2/report_scripts/stop_tables.py --dir results/ms_r2/stop_calibration --block spearman|freeze|jcheck|diag|fire|facts
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

LABEL = {"C1": "C1 = R0", "C2": "C2 = |c2|", "C2s": "C2s = c2 (signed)", "C3": "C3 = R_defl", "R": "R", "Delta": "Delta"}


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)

    def esc(x: object) -> str:
        return str(x).replace("|", "\\|")
    out = ["| " + " | ".join(esc(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(esc(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(out)


def _t(d: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(d / "tables" / f"{name}.csv", float_precision="round_trip")


def block_spearman(d: Path) -> str:
    """Pooled Spearman correlation of the candidates with |peak| and RMSE_pos (u >= 400), per q."""
    a = _t(d, "a_spearman")
    a = a[a.group == "all"]
    rows = []
    for q in (50, 60):
        for cand in ("C1", "C2", "C2 signed", "C3", "R", "Delta"):
            r = {"q": q, "candidate": cand}
            for sub, lab in (("all", "all exports"), ("constLR", "constant-LR exports")):
                for tgt in ("|peak|", "RMSE_pos"):
                    x = a[(a.q == q) & (a.candidate == cand) & (a.subset == sub) & (a.target == tgt)]
                    r[f"{lab}: {tgt}"] = f"{x.spearman_pooled.iloc[0]:.3f}" if len(x) else "n/a"
            rows.append(r)
    return _md(pd.DataFrame(rows))


def block_freeze(d: Path) -> str:
    """Distribution at the freeze: min / median / max per q and candidate."""
    b = _t(d, "b_freeze")
    b = b[b.group == "all"] if "group" in b.columns else b
    rows = [{"q": r.q, "quantity": r.candidate, "n runs": int(r.n), "min": f"{r.min:.4f}", "median": f"{r.median:.4f}", "max": f"{r.max:.4f}"}
            for r in b.itertuples()]
    return _md(pd.DataFrame(rows))


def block_jcheck(d: Path) -> str:
    """The J check against the linearisation."""
    c = _t(d, "c_jcheck")
    return _md(pd.DataFrame([{"q": r.q, "side": r.side, "n exports": int(r.n_exports), "mean J": f"{r.mean_J:.4f}", "linearised J": f"{r.J_lin:.4f}",
                              "mean abs deviation": f"{r.mean_abs_dev:.2e}", "max abs deviation": f"{r.max_abs_dev:.2e}"} for r in c.itertuples()]))


def block_diag(d: Path) -> str:
    """Development-tier best-response grid error and the C3 exclusions."""
    x = _t(d, "x_diagnostics")
    return _md(pd.DataFrame([{"q": r.q, "n exports": int(r.n_exports), "|s - s_exact| median": f"{r.abs_s_minus_s_exact_median:.1e}",
                              "|s - s_exact| max": f"{r.abs_s_minus_s_exact_max:.3f}", "|R0 - R0_exact| max": f"{r.abs_R0_minus_R0_exact_max:.4f}",
                              "non-tail nodes": int(r.n_defl_nodes_median), "excluded nodes (total)": int(r.n_defl_excluded_total)} for r in x.itertuples()]))


def block_fire(d: Path) -> str:
    """Fire of 'candidate <= theta at M = 3 consecutive exports', pooled over arms, per q."""
    f = _t(d, "fire")
    f = f[f.group == "all"]
    rows = []
    for r in f.itertuples():
        cell = lambda m, lo, hi: "-" if not np.isfinite(m) else f"{m:.4f} [{lo:.4f}, {hi:.4f}]"  # noqa: E731
        rows.append({"candidate": LABEL.get(r.candidate, r.candidate), "theta": f"{r.theta:g}", "q": r.q, "fired": f"{int(r.n_fired)}/{int(r.n_runs)}",
                     "fire update median [min, max]": "-" if not np.isfinite(r.fire_u_median) else f"{r.fire_u_median:.0f} [{r.fire_u_min:.0f}, {r.fire_u_max:.0f}]",
                     "|peak| at fire": cell(r.abs_peak_at_fire_median, r.abs_peak_at_fire_min, r.abs_peak_at_fire_max),
                     "|peak| at the end (same runs)": cell(r.abs_peak_at_end_median, r.abs_peak_at_end_min, r.abs_peak_at_end_max),
                     "|peak| <= 0.05 at fire": f"{int(r.n_peak_le_0p05_at_fire)}/{int(r.n_fired)}" if "n_peak_le_0p05_at_fire" in f.columns else "n/a"})
    return _md(pd.DataFrame(rows))


def _all_rows(d: Path) -> pd.DataFrame:
    fs = sorted(glob.glob(str(d / "*" / "*" / "q*" / "seed*.csv")))
    return pd.concat([pd.read_csv(x) for x in fs], ignore_index=True)


def block_facts(d: Path) -> str:
    """Facts quoted in the report: R0 against |peak|, the C2 floor at the freeze, runs and exports counted."""
    df = _all_rows(d)
    df = df[df.valid.astype(str) == "True"]
    late = df[df["update"] >= 400]
    lines: List[str] = [f"- exports replayed: {len(df)} valid terminal-stage exports of {df.groupby(['source', 'arm', 'q', 'seed']).ngroups} runs"]
    for q in (50, 60):
        x = late[(late.q == q) & (late.stage2_peak_rel_err_abs > 0)]
        ratio = x.R0 / x.stage2_peak_rel_err_abs
        lines.append(f"- q = {q}: R0 / |peak| over the exports with u >= 400: median {ratio.median():.3f}, 5th-95th percentile "
                     f"[{ratio.quantile(0.05):.3f}, {ratio.quantile(0.95):.3f}] (the linearised value is 2k / (2k + a))")
    last = df.sort_values("update").groupby(["source", "arm", "q", "seed"]).tail(1)
    for q in (50, 60):
        x = last[last.q == q]
        lines.append(f"- q = {q}, at the freeze ({len(x)} runs): |c2| below its noise floor sigma_2(0) / (sqrt(pi) q) in {int((x.c2.abs() < x.floor).sum())} runs; "
                     f"c2 < 0 (the learned tie effort above e_sigma(0)) in {int((x.c2 < 0).sum())} runs")
    return "\n".join(lines)


BLOCKS = {"spearman": block_spearman, "freeze": block_freeze, "jcheck": block_jcheck, "diag": block_diag, "fire": block_fire, "facts": block_facts}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True)
    p.add_argument("--block", required=True, choices=sorted(BLOCKS))
    a = p.parse_args(argv)
    print(BLOCKS[a.block](Path(a.dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
