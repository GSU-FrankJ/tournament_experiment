#!/usr/bin/env python3
"""Markdown tables of ``01_supervised_screen.md`` and ``01b_rl_actor_diagnostics.md``, read from the CSV / JSON files of
``tools/ms/r3_supervised_screen.py`` (``summary_by_cell.csv``, ``summary_median.csv``, ``summary_extended.csv``,
``premise_check.json``) and ``tools/ms/r3_actor_diagnostics.py``; no number is typed by hand except the values the PI
quoted in the prompt (block ``sandbox``, marked as quoted and as not repository evidence).

Usage (repository root):
    python reports/ms/r3/report_scripts/screen_tables.py --block grid|metrics|premise|extended|sandbox|seeds|diag_arm|diag_relation|diag_regime|diag_ranges|diag_check|diag_dist|diag_dist_facts|grid_rmse|grid_tail|grid_weff|grid_maxw
        [--dir results/ms_r3/supervised_screen] [--diag-dir results/ms_r3/rl_actor_diagnostics]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ACTORS = ["t1", "relu", "t10"]
STARTS = ["bb", "st"]
STARTS_LAB = {"bb": "bin-balanced", "st": "stratified"}
QS = [50, 60]
DIR = "results/ms_r3/supervised_screen"
DIAG = "results/ms_r3/rl_actor_diagnostics"
#: values quoted in the prompt (PI-side numpy sandbox, 3 seeds per cell, not repository evidence): median tip deficit
#: (effort units) at q = 50 / 60, and RMSE_pos for bin-balanced starts
SANDBOX_DEFICIT = {("t1", "bb"): (1.90, 1.49), ("t1", "st"): (0.76, 0.76), ("relu", "bb"): (0.49, 0.04),
                   ("relu", "st"): (0.08, 0.03), ("t10", "bb"): (0.54, 0.45), ("t10", "st"): (0.44, 0.39)}
SANDBOX_RMSE_BB = {"t1": (0.54, 0.39), "relu": (0.08, 0.02), "t10": (0.07, 0.07)}
SANDBOX_MAXW = {("t1", "bb"): "1.7 / 1.7", ("t1", "st"): "1.4-1.5", ("relu", "bb"): "0.76-0.92", ("t10", "bb"): "not quoted"}


def _md(rows: List[Dict[str, object]]) -> str:
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def _med(d: Path, name: str = "summary_median.csv") -> pd.DataFrame:
    return pd.read_csv(d / name, float_precision="round_trip")


def _cell(r: pd.Series, m: str, p: int = 3) -> str:
    return f"{r[f'{m}_median']:.{p}f} [{r[f'{m}_min']:.{p}f}, {r[f'{m}_max']:.{p}f}]"


def _grid(d: Path, metric: str, p: int) -> str:
    """One metric of ``summary_median.csv``, median [min, max] over the ten seeds, at the four checkpoints."""
    m = _med(d)
    rows = []
    for a in ACTORS:
        for s in STARTS:
            for q in QS:
                g = m[(m.actor == a) & (m.starts == s) & (m.q == q)]
                r: Dict[str, object] = {"actor": a, "starts": STARTS_LAB[s], "q": q}
                for st in (16000, 32000, 48000, 56000):
                    x = g[g.steps == st]
                    r[f"{st // 1000}k"] = _cell(x.iloc[0], metric, p) if len(x) else "n/a"
                r["n seeds"] = int(g[g.steps == 56000].n.iloc[0]) if len(g[g.steps == 56000]) else 0
                rows.append(r)
    return _md(rows)


def block_grid(d: Path) -> str:
    """Tip deficit e2*(0) - e_hat(0) (effort units), median [min, max] over the ten seeds, at the four checkpoints."""
    return _grid(d, "tip_deficit", 3)


def block_grid_rmse(d: Path) -> str:
    """RMSE_pos (effort units) at the four checkpoints."""
    return _grid(d, "rmse_pos", 3)


def block_grid_tail(d: Path) -> str:
    """Tail mean (effort units) at the four checkpoints."""
    return _grid(d, "tail_mean", 3)


def block_grid_weff(d: Path) -> str:
    """w_eff = tip deficit / (e2*(0) / 2q), units of d, at the four checkpoints."""
    return _grid(d, "w_eff", 2)


def block_grid_maxw(d: Path) -> str:
    """Max abs first-layer weight on the d input (units of d / B; ten times the stored weight for t10) at the four checkpoints."""
    return _grid(d, "max_abs_w_d", 2)


def block_metrics(d: Path) -> str:
    """The other reported metrics at 56,000 steps: RMSE_pos (effort units), tail mean, w_eff (units of d), max |w| (d / B)."""
    m = _med(d)
    rows = []
    for a in ACTORS:
        for s in STARTS:
            for q in QS:
                x = m[(m.actor == a) & (m.starts == s) & (m.q == q) & (m.steps == 56000)]
                if x.empty:
                    continue
                x = x.iloc[0]
                rows.append({"actor": a, "starts": STARTS_LAB[s], "q": q, "RMSE_pos": _cell(x, "rmse_pos"),
                             "tail mean": _cell(x, "tail_mean"), "w_eff": _cell(x, "w_eff", 2),
                             "max abs w on d (d / B)": _cell(x, "max_abs_w_d", 2),
                             "tip deficit / e2*(0)": _cell(x, "tip_deficit_over_g2_0", 4)})
    return _md(rows)


def block_premise(d: Path) -> str:
    """The premise check of prompt section 2.5 (bin-balanced starts, 56,000 steps), from ``premise_check.json``."""
    p = json.loads((d / "premise_check.json").read_text())
    rows = []
    for q, v in p["condition_i"]["by_q"].items():
        rows.append({"condition": "(i) median deficit of t1 >= 1.0", "variant": "t1", "q": q,
                     "median deficit": f"{v['median_t1']:.4f}", "limit": f">= {v['threshold']:g}", "result": "pass" if v["pass"] else "FAIL"})
    for var, c in p["condition_ii"].items():
        for q, v in c["by_q"].items():
            rows.append({"condition": "(ii) median deficit <= 0.5 x that of t1", "variant": var, "q": q,
                         "median deficit": f"{v['median_variant']:.4f}", "limit": f"<= {v['limit']:.4f} (ratio {v['ratio']:.3f})",
                         "result": "pass" if v["pass"] else "FAIL"})
    tail = (f"\nOutcome: **{p['outcome']}** ({p['outcome_reason']}). Seeds per group: "
            f"{sorted({n for a in p['n_seeds'].values() for n in a.values()})}; complete: {p['complete']}; initial weights "
            f"identical across the three actors for every (q, seed): {p['init_identical_across_actors']}.")
    return _md(rows) + tail


def block_extended(d: Path) -> str:
    """The extended cell (t1, bin-balanced, 224,000 steps): median [min, max] of the tip deficit and RMSE_pos by checkpoint."""
    e = _med(d, "summary_extended.csv")
    rows = []
    for q in QS:
        g = e[(e.actor == "t1") & (e.starts == "bb") & (e.q == q)].sort_values("steps")
        r: Dict[str, object] = {"q": q}
        for _, x in g.iterrows():
            r[f"deficit @ {int(x.steps) // 1000}k"] = _cell(x, "tip_deficit")
        rows.append(r)
    rows2 = []
    for q in QS:
        g = e[(e.actor == "t1") & (e.starts == "bb") & (e.q == q)].sort_values("steps")
        r2: Dict[str, object] = {"q": q}
        for _, x in g.iterrows():
            r2[f"RMSE_pos @ {int(x.steps) // 1000}k"] = _cell(x, "rmse_pos")
        rows2.append(r2)
    return _md(rows) + "\n\n" + _md(rows2)


def block_sandbox(d: Path) -> str:
    """The repository medians at 56,000 steps next to the values the PI quoted from the sandbox (3 seeds; not evidence)."""
    m = _med(d)
    rows = []
    for a in ACTORS:
        for s in STARTS:
            r: Dict[str, object] = {"actor": a, "starts": STARTS_LAB[s]}
            for i, q in enumerate(QS):
                x = m[(m.actor == a) & (m.starts == s) & (m.q == q) & (m.steps == 56000)]
                r[f"q = {q}: repository median (10 seeds)"] = f"{x.iloc[0]['tip_deficit_median']:.2f}" if len(x) else "n/a"
                r[f"q = {q}: PI sandbox (3 seeds, quoted)"] = f"{SANDBOX_DEFICIT[(a, s)][i]:.2f}"
            rows.append(r)
    rows2 = []
    for a in ACTORS:
        r2: Dict[str, object] = {"actor": a, "starts": "bin-balanced"}
        for i, q in enumerate(QS):
            x = m[(m.actor == a) & (m.starts == "bb") & (m.q == q) & (m.steps == 56000)]
            r2[f"q = {q}: repository RMSE_pos median"] = f"{x.iloc[0]['rmse_pos_median']:.2f}" if len(x) else "n/a"
            r2[f"q = {q}: PI sandbox (quoted)"] = f"{SANDBOX_RMSE_BB[a][i]:.2f}"
        rows2.append(r2)
    rows3 = []
    for (a, s), quoted in SANDBOX_MAXW.items():
        r3: Dict[str, object] = {"actor": a, "starts": STARTS_LAB[s], "PI sandbox: largest first-layer d-weight (d / B), quoted": quoted}
        for q in QS:
            x = m[(m.actor == a) & (m.starts == s) & (m.q == q) & (m.steps == 56000)]
            r3[f"q = {q}: repository median max abs w"] = f"{x.iloc[0]['max_abs_w_d_median']:.2f}" if len(x) else "n/a"
        rows3.append(r3)
    return _md(rows) + "\n\n" + _md(rows2) + "\n\n" + _md(rows3)


def block_seeds(d: Path) -> str:
    """The ten per-seed tip deficits at 56,000 steps, bin-balanced starts (the spread behind the medians)."""
    c = pd.read_csv(d / "summary_by_cell.csv", float_precision="round_trip")
    c = c[(c.extended == 0) & (c.steps == 56000) & (c.starts == "bb")]
    rows = []
    for a in ACTORS:
        for q in QS:
            g = c[(c.actor == a) & (c.q == q)].sort_values("seed")
            rows.append({"actor": a, "q": q, **{str(int(s)): f"{v:.2f}" for s, v in zip(g.seed, g.tip_deficit)},
                         "seeds with deficit > 4": int((g.tip_deficit > 4.0).sum())})
    return _md(rows)


def _diag(dd: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(dd / name, float_precision="round_trip")


def block_diag_arm(dd: Path) -> str:
    """RL actors: median over the seeds of max |w| on the d input (units of d / B) and of w_eff (units of d) at four exports."""
    a = _diag(dd, "per_arm.csv")
    rows = []
    for (src, arm, q), g in a.groupby(["source", "arm", "q"], sort=False):
        r: Dict[str, object] = {"source": src, "arm": arm, "q": int(q)}
        for w in ("u400", "u1200", "u2000", "final"):
            x = g[g.which == w]
            r[f"{w}: max abs w / w_eff"] = (f"{x.iloc[0].median_max_abs_w_d:.3f} / {x.iloc[0].median_w_eff:.2f}"
                                           f" (u{int(x.iloc[0].update_used_median)})") if len(x) else "n/a"
        rows.append(r)
    return _md(rows)


def block_diag_relation(dd: Path) -> str:
    """Spearman correlation of max |w| with w_eff over the terminal-stage exports (pooled over seeds / median within a run)."""
    a = _diag(dd, "relation.csv")
    return _md([{"source": r.source, "arm": r.arm, "q": int(r.q), "runs": int(r.n_runs), "pooled rho": f"{r.pooled_spearman:.3f}",
                 "within-run median rho": f"{r.within_run_median_spearman:.3f}",
                 "final export, across seeds: rho": f"{r.final_cross_seed_spearman:.3f}"} for r in a.itertuples()])


def block_diag_ranges(dd: Path) -> str:
    """Ranges over the source / arm / q rows of the seed medians at u400 and at the final export, and of the correlations."""
    a = _diag(dd, "per_arm.csv")
    rel = _diag(dd, "relation.csv")
    reg = _diag(dd, "regime.csv")
    rows = []
    for w in ("u400", "final"):
        x = a[a.which == w]
        rows.append({"quantity": f"median max abs w on d (d / B), {w}", "min": f"{x.median_max_abs_w_d.min():.3f}",
                     "max": f"{x.median_max_abs_w_d.max():.3f}", "rows": len(x)})
        rows.append({"quantity": f"median w_eff (units of d), {w}", "min": f"{x.median_w_eff.min():.2f}",
                     "max": f"{x.median_w_eff.max():.2f}", "rows": len(x)})
    fin = a[a.which == "final"]
    for q in QS:
        x = fin[fin.q == q]
        width = (100.0 + 2.0 * q) / x.median_max_abs_w_d          # B / max|w|: the d-range over which the sharpest tanh unit bends
        rows.append({"quantity": f"B / median max abs w, final, q = {q} (B = {100 + 2 * q}; units of d)",
                     "min": f"{width.min():.0f}", "max": f"{width.max():.0f}", "rows": len(x)})
    u4, fi = a[a.which == "u400"].set_index(["source", "arm", "q"]), a[a.which == "final"].set_index(["source", "arm", "q"])
    growth = (fi.median_max_abs_w_d / u4.median_max_abs_w_d - 1.0).dropna()
    rows.append({"quantity": "growth of the median max abs w from u400 to final (fraction)", "min": f"{growth.min():.3f}",
                 "max": f"{growth.max():.3f}", "rows": len(growth)})
    rows.append({"quantity": "across the ten final exports: Spearman(max abs w, w_eff)",
                 "min": f"{rel.final_cross_seed_spearman.min():.3f}", "max": f"{rel.final_cross_seed_spearman.max():.3f}", "rows": len(rel)})
    rows.append({"quantity": "within-run median Spearman(max abs w, w_eff)", "min": f"{rel.within_run_median_spearman.min():.3f}",
                 "max": f"{rel.within_run_median_spearman.max():.3f}", "rows": len(rel)})
    rows.append({"quantity": "pooled Spearman(max abs w, w_eff)", "min": f"{rel.pooled_spearman.min():.3f}",
                 "max": f"{rel.pooled_spearman.max():.3f}", "rows": len(rel)})
    rows.append({"quantity": "share of final exports with max abs w <= the screen reference", "min": f"{reg.frac_final_at_or_below.min():.2f}",
                 "max": f"{reg.frac_final_at_or_below.max():.2f}", "rows": len(reg)})
    return _md(rows)


def block_diag_check(dd: Path) -> str:
    """The reload of the diagnostics against the verifier: final terminal-stage export vs ``e2_at_0`` of the analysis tables."""
    e = _diag(dd, "per_export.csv")
    e = e[e.stage == 2]
    last = e.sort_values("update").groupby(["source", "arm", "q", "seed"]).tail(1)
    rows = []
    for src, path in (("ms_r2_pilot", "results/ms_r2/analysis/per_run.csv"), ("ms_r1_pilot", "results/ms_r1/analysis/per_run.csv"),
                      ("ms_r1_base", "results/ms_r1/analysis/per_run.csv")):
        pr = pd.read_csv(path, float_precision="round_trip", low_memory=False)
        pr = pr[pr["status"] == "done"][["arm", "q", "seed", "e2_at_0", "g2_at_0"]]
        m = last[last.source == src].merge(pr, on=["arm", "q", "seed"], how="inner", suffixes=("", "_verifier"))
        if m.empty:
            continue
        de = (m.e_hat_2_0 - m.e2_at_0).abs()
        wv = (m.g2_at_0 - m.e2_at_0) / (m.g2_at_0 / (2.0 * m.q))
        rows.append({"source": src, "runs compared": len(m), "max abs diff of e_hat_2(0) (effort units)": f"{de.max():.2e}",
                     "max abs diff of w_eff (units of d)": f"{(m.w_eff - wv).abs().max():.2e}", "analysis table": path})
    return _md(rows)


def block_diag_dist(dd: Path) -> str:
    """The distribution of |first-layer d-weight| over the 64 units (units of d / B): per source / arm / q the median over the
    seeds of the quantiles 25 / 50 / 90 / 100 % and of the number of units above 1, 2, 3 and 5, at update 400 and at the final
    terminal-stage export."""
    e = _diag(dd, "per_export.csv")
    e = e[e.stage == 2]
    rows = []
    for (src, arm, q), g in e.groupby(["source", "arm", "q"], sort=False):
        r: Dict[str, object] = {"source": src, "arm": arm, "q": int(q)}
        fin = g.sort_values("update").groupby("seed").tail(1)
        for lab, x in (("u400", g[g["update"] == 400]), ("final", fin)):
            if x.empty:
                r[lab] = "n/a"
                continue
            r[lab] = (f"|w| q25/q50/q90/max {x.absw_q25.median():.2f}/{x.absw_q50.median():.2f}/{x.absw_q90.median():.2f}/"
                      f"{x.absw_q100.median():.2f}; units > 1/2/3/5: {x.n_absw_gt1.median():.0f}/{x.n_absw_gt2.median():.0f}/"
                      f"{x.n_absw_gt3.median():.0f}/{x.n_absw_gt5.median():.0f}")
        rows.append(r)
    return _md(rows)


def block_diag_dist_facts(dd: Path) -> str:
    """Facts about the |first-layer d-weight| distribution at the final terminal-stage export of every run."""
    e = _diag(dd, "per_export.csv")
    fin = e[e.stage == 2].sort_values("update").groupby(["source", "arm", "q", "seed"]).tail(1)
    lines = [f"- final terminal-stage exports: {len(fin)} runs"]
    for c, lab in (("n_absw_gt1", "units with |w| > 1"), ("n_absw_gt2", "units with |w| > 2"), ("n_absw_gt3", "units with |w| > 3"),
                   ("n_absw_gt5", "units with |w| > 5")):
        lines.append(f"- {lab}: min {int(fin[c].min())}, median {fin[c].median():.0f}, max {int(fin[c].max())} (of 64 units)")
    for c, lab in (("absw_q50", "median |w| over the units"), ("absw_q90", "90 % quantile of |w|"), ("absw_q100", "max |w|")):
        lines.append(f"- {lab}: min {fin[c].min():.3f}, median over runs {fin[c].median():.3f}, max {fin[c].max():.3f}")
    return "\n".join(lines)


def block_diag_regime(dd: Path) -> str:
    """Share of RL exports with max |w| at or below the screen's t1 reference (bin-balanced starts, 56,000 steps, median)."""
    a = _diag(dd, "regime.csv")
    return _md([{"source": r.source, "arm": r.arm, "q": int(r.q), "reference max abs w": f"{r.reference_max_abs_w_d:.3f}",
                 "all exports": f"{r.frac_all_at_or_below:.3f}", "terminal-stage exports": f"{r.frac_terminal_at_or_below:.3f}",
                 "final exports (of %d)" % r.n_final: f"{r.frac_final_at_or_below:.2f}"} for r in a.itertuples()])


BLOCKS: Dict[str, Callable[[Path], str]] = {"grid": block_grid, "metrics": block_metrics, "premise": block_premise,
                                            "extended": block_extended, "sandbox": block_sandbox, "seeds": block_seeds,
                                            "diag_arm": block_diag_arm, "diag_relation": block_diag_relation,
                                            "diag_regime": block_diag_regime,
                                            "diag_ranges": block_diag_ranges,
                                            "diag_check": block_diag_check, "diag_dist": block_diag_dist, "diag_dist_facts": block_diag_dist_facts,
                                            "grid_rmse": block_grid_rmse, "grid_tail": block_grid_tail,
                                            "grid_weff": block_grid_weff, "grid_maxw": block_grid_maxw}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--block", required=True)
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--diag-dir", default=DIAG)
    a = ap.parse_args(argv)
    if a.block not in BLOCKS:
        raise SystemExit(f"unknown block {a.block!r}; known: {sorted(BLOCKS)}")
    print(BLOCKS[a.block](Path(a.diag_dir if a.block.startswith("diag_") else a.dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
