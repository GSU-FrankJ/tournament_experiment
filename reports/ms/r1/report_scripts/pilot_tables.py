#!/usr/bin/env python3
"""Markdown tables of the MS-R1 pilot reports, read from the analysis CSVs (no number is typed by hand).

Usage (from the repository root):
    python reports/ms/r1/report_scripts/pilot_tables.py --block primary
    python reports/ms/r1/report_scripts/pilot_tables.py --block arm:MS_s25a0
    python reports/ms/r1/report_scripts/pilot_tables.py --list

``--analysis`` is the analysis directory (default ``results/ms_r1/analysis``, written by ``tools/ms/r1_analysis.py``).
Everything except the ``primary`` block (the pre-registered criterion) is descriptive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ARMS = ["MS_base", "MS_base2400", "MS_rule", "MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5"]
RULE_ARMS = ["MS_rule", "MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5"]
COMPARATORS = ["parents_A", "rehearsal_v2_0"]
QS = [50, 60]
A: Path = Path("results/ms_r1/analysis")


def _md(df: pd.DataFrame) -> str:
    """GitHub markdown table of a frame (strings as given)."""
    cols = list(df.columns)

    def esc(x: object) -> str:
        return str(x).replace("|", "\\|")      # a pipe inside a cell (|peak|) must not split the cell
    out = ["| " + " | ".join(esc(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(esc(r[c]) for c in cols) + " |")
    return "\n".join(out)


def _ci(m: float, lo: float, hi: float, p: int = 5) -> str:
    """``+m [lo, hi]`` with a star when the interval excludes 0."""
    if any(not np.isfinite(x) for x in (m, lo, hi)):
        return "nan"
    star = "" if lo <= 0.0 <= hi else " *"
    return f"{m:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]{star}"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(A / name, float_precision="round_trip")


def block_primary() -> str:
    """Pre-registered criterion: parts (a) and (b) of every arm against parents_A."""
    c = _read("criterion.csv")
    rows = []
    for _, r in c.iterrows():
        rows.append({"arm": r["arm"],
                     "(a) q=50 mean [95% CI]": _ci(r["mean_q50"], r["ci_mean_lo_q50"], r["ci_mean_hi_q50"]),
                     "met q=50": "yes" if r["a_q50"] else "no",
                     "(a) q=60 mean [95% CI]": _ci(r["mean_q60"], r["ci_mean_lo_q60"], r["ci_mean_hi_q60"]),
                     "met q=60": "yes" if r["a_q60"] else "no",
                     "pairs": f"{int(r['n_pairs_q50'])}+{int(r['n_pairs_q60'])}",
                     "(b)": r["b_status"] + (f" ({r['b_violations']})" if isinstance(r["b_violations"], str)
                                             and r["b_violations"] else ""),
                     "overall": r["overall"]})
    return _md(pd.DataFrame(rows))


def block_secondary() -> str:
    """Secondary criterion: rule arms minus MS_base2400 (descriptive)."""
    c = _read("criterion_vs_MS_base2400.csv")
    rows = []
    for _, r in c.iterrows():
        rows.append({"arm": r["arm"],
                     "(a) q=50 mean [95% CI]": _ci(r["mean_q50"], r["ci_mean_lo_q50"], r["ci_mean_hi_q50"]),
                     "met q=50": "yes" if r["a_q50"] else "no",
                     "(a) q=60 mean [95% CI]": _ci(r["mean_q60"], r["ci_mean_lo_q60"], r["ci_mean_hi_q60"]),
                     "met q=60": "yes" if r["a_q60"] else "no",
                     "(b)": r["b_status"] + (f" ({r['b_violations']})" if isinstance(r["b_violations"], str)
                                             and r["b_violations"] else ""),
                     "overall": r["overall"]})
    return _md(pd.DataFrame(rows))


def block_overview() -> str:
    """Per arm and q: |peak| <= 0.05 counts, signed peak error with its interval, tail, eta_2, gates, shares."""
    per = _read("per_run.csv")
    dec = _read("decision_inputs.csv")
    rows = []
    for arm in ARMS + COMPARATORS:
        for q in QS:
            g = per[(per.arm == arm) & (per.q == q) & (per.status == "done")]
            d = dec[(dec.arm == arm) & (dec.q.astype(str) == str(q))].iloc[0]
            sm = pd.to_numeric(g["smoothed_share_peak_gap_d0"], errors="coerce").dropna()
            rows.append({
                "arm": arm, "q": q, "n": len(g), "|peak|<=0.05": int(d["n_abs_peak_le_0.05"]),
                "mean |peak|": f"{d['abs_peak_mean']:.4f}",
                "signed peak mean [95% CI]": _ci(d["signed_peak_mean"], d["signed_peak_ci_lo"], d["signed_peak_ci_hi"]),
                "signed<0": int(d["n_signed_peak_negative"]),
                "RMSE_pos": f"{d['rmse_mean']:.4f}", "tail mean": f"{d['tail_mean_mean']:.4f}",
                "tail max (mean)": f"{d['tail_max_mean']:.4f}", "eta2/DW": f"{d['eta2_mean']:.5f}",
                "G-A+G-N(eta)": f"{int(d['n_gate_pass'])}/{len(g)}",
                "smoothed share, median": f"{sm.median():.3f}" if len(sm) else "n/a"})
    return _md(pd.DataFrame(rows))


PAIR_METRICS = [("stage2_peak_rel_err_abs", "|peak error|"), ("stage2_peak_rel_err_signed", "signed peak error"),
                ("stage2_rmse_pos_over_g2_0", "RMSE_pos/e2*(0)"), ("stage2_tail_mean_over_g2_0", "tail mean/e2*(0)"),
                ("stage2_tail_max_over_g2_0", "tail max/e2*(0)"), ("eta_T_over_dw", "eta_2/DW (final)"),
                ("stage1_rel_err_abs", "stage-1 |error|")]


def _paired(comp: str, arms: Sequence[str], metrics: Sequence = PAIR_METRICS) -> str:
    f = {"vs parents_A": "paired_vs_parents_A.csv", "vs MS_base2400": "paired_vs_MS_base2400.csv",
         "vs MS_rule": "paired_vs_MS_rule.csv", "vs rehearsal_v2_0": "paired_vs_rehearsal_v2_0.csv"}[comp]
    p = _read(f)
    p = p[p.comparison == comp]
    rows = []
    for arm in arms:
        for m, lab in metrics:
            cells = {"arm": arm, "metric": lab}
            ok = False
            for q in QS:
                r = p[(p.arm == arm) & (p.q == q) & (p.metric == m)]
                if len(r):
                    r = r.iloc[0]
                    cells[f"q={q}: mean [95% CI] (arm - baseline)"] = _ci(r["mean"], r["ci_mean_lo"], r["ci_mean_hi"])
                    cells[f"q={q}: seeds lower"] = f"{int(r['n_neg'])}/{int(r['n_pairs'])}"
                    ok = True
                else:
                    cells[f"q={q}: mean [95% CI] (arm - baseline)"] = "n/a"
                    cells[f"q={q}: seeds lower"] = "n/a"
            if ok:
                rows.append(cells)
    return _md(pd.DataFrame(rows))


def block_paired_parents() -> str:
    """All arms minus parents_A on the terminal-stage metrics."""
    return _paired("vs parents_A", ARMS, PAIR_METRICS[:6])


def block_paired_b2400() -> str:
    """Rule arms minus MS_base2400 on the terminal-stage metrics."""
    return _paired("vs MS_base2400", RULE_ARMS, PAIR_METRICS[:6])


def block_paired_rule() -> str:
    """Sampler arms minus MS_rule on the terminal-stage metrics."""
    return _paired("vs MS_rule", RULE_ARMS[1:], PAIR_METRICS[:4])


def block_rule() -> str:
    """Rule record of the terminal stage and stage 1 per arm and q."""
    r = _read("rule.csv")
    rows = []
    for _, x in r.iterrows():
        leg = x["arm"] in ("MS_base", "MS_base2400")
        rows.append({
            "arm": x["arm"], "q": int(x["q"]),
            "t2 stop fired": int(x["t2_n_fire"]), "t2 cap (budget_forced)": int(x["t2_n_budget_forced"]),
            "t2 blocks (global+polish)": "-" if leg else
            f"{int(x['t2_n_global_blocks_total'])}+{int(x['t2_n_polish_blocks_total'])}",
            "runs with polish": "-" if leg else int(x["t2_n_runs_with_polish"]),
            "localized / broad": "-" if leg else f"{int(x['t2_n_localized_total'])} / {int(x['t2_n_broad_total'])}",
            "|S_t| mean (max)": "-" if leg else f"{x['t2_nS_at_classification_mean']:.2f} ({int(x['t2_nS_at_classification_max'])})",
            "would-fire (legacy)": int(x["t2_n_would_fire"]) if leg else "-",
            "t1 stop fired": int(x["t1_n_fire"]), "t1 cap": int(x["t1_n_budget_forced"]),
            "t1 would-fire (legacy)": int(x["t1_n_would_fire"]) if leg else "-",
            "t1 fire local (median)": "-" if leg or not np.isfinite(x["t1_fire_local_median"]) else f"{x['t1_fire_local_median']:.0f}"})
    return _md(pd.DataFrame(rows))


def block_classif() -> str:
    """A3 (c): |S_t| composition at every classification, per arm and q."""
    r = _read("rule.csv")
    rows = []
    for _, x in r.iterrows():
        if x["arm"] in ("MS_base", "MS_base2400"):
            continue
        n = max(1.0, x["t2_n_classifications_total"])
        rows.append({"arm": x["arm"], "q": int(x["q"]), "classifications": int(x["t2_n_classifications_total"]),
                     "localized": int(x["t2_n_localized_total"]), "broad": int(x["t2_n_broad_total"]),
                     "sum |S| near-tie bins": int(x["t2_S_near_total"]), "sum |S| middle bins": int(x["t2_S_mid_total"]),
                     "sum |S| tail bins": int(x["t2_S_tail_total"]),
                     "classifications with near-tie bins in S": f"{int(x['t2_n_classifications_with_near_in_S'])}/{int(n)}",
                     "runs with near-tie bins in S": f"{int(x['t2_n_runs_with_near_in_S'])}/{int(x['n_runs'])}"})
    return _md(pd.DataFrame(rows))


def block_budget() -> str:
    """Budget to the freeze per arm and q (medians over the runs)."""
    b = _read("budget.csv")
    rows = []
    for arm in ARMS + COMPARATORS:
        for q in QS:
            g = b[(b.arm == arm) & (b.q == q)].set_index("metric")["median"]
            rows.append({"arm": arm, "q": q,
                         "terminal-stage updates": f"{g.get('t2_updates', np.nan):.0f}",
                         "stage-1 updates": "-" if not np.isfinite(g.get("t1_updates", np.nan)) else f"{g['t1_updates']:.0f}",
                         "total updates": f"{g.get('total_updates', np.nan):.0f}",
                         "total episodes": f"{g.get('total_episodes', np.nan):.0f}",
                         "optimiser steps": "-" if not np.isfinite(g.get("total_minibatch_steps", np.nan)) else f"{g['total_minibatch_steps']:.0f}",
                         "wall s (median)": f"{g.get('total_wall_sec', np.nan):.0f}"})
    return _md(pd.DataFrame(rows))


def block_stage1() -> str:
    """Stage-1 results per arm and q."""
    s = _read("stage1.csv")
    rows = []
    for _, x in s.iterrows():
        rows.append({"arm": x["arm"], "q": int(x["q"]),
                     "signed error mean [95% CI]": _ci(x["stage1_rel_err_signed_mean"], x["stage1_rel_err_signed_ci_lo"],
                                                       x["stage1_rel_err_signed_ci_hi"]),
                     "|error| mean": f"{x['stage1_rel_err_abs_mean']:.4f}", "|error| median": f"{x['stage1_rel_err_abs_median']:.4f}",
                     "R_1 final (median)": "n/a" if not np.isfinite(x["t1_R_final_median"]) else f"{x['t1_R_final_median']:.4f}",
                     "G-S": f"{int(x['n_G_S_pass'])}/{int(x['n_runs'])}", "G-F": f"{int(x['n_G_F_pass'])}/{int(x['n_runs'])}",
                     "G-N(Gmax)": f"{int(x['n_G_N_gmax_pass'])}/{int(x['n_runs'])}",
                     "v2.0 combination": f"{int(x['n_v20_combination_pass'])}/{int(x['n_runs'])}",
                     "learning (mean)": f"{x['learning_rel_mean']:+.4f}", "inherited (mean)": f"{x['inherited_rel_mean']:+.4f}"})
    return _md(pd.DataFrame(rows))


def block_stage1_vs_rehearsal() -> str:
    """Stage-1 |error| minus rehearsal_v2_0."""
    return _paired("vs rehearsal_v2_0", ARMS, [("stage1_rel_err_abs", "stage-1 |error|"),
                                               ("stage1_rel_err_signed", "stage-1 signed error")])


def block_stage1_r1() -> str:
    """A3 (d): firing streaks containing an exact R_1 = 0 check, and R_1 on the final tier at the stage-1 freeze."""
    s = _read("stage1_R1.csv")
    rows = []
    for _, x in s.iterrows():
        if x["arm"] in COMPARATORS:
            continue
        leg = x["arm"] in ("MS_base", "MS_base2400")
        nst = int(x["n_streaks_would_fire"] if leg else x["n_streaks_fire"])
        rows.append({"arm": x["arm"], "q": int(x["q"]), "streak kind": "would-fire" if leg else "fire",
                     "streaks": nst, "streaks with an R_1 = 0 check": int(x["n_streaks_with_R0"]),
                     "checks with R_1 = 0 (all)": int(x["n_checks_R0_total"]),
                     "runs with such a check": int(x["n_runs_with_R0_check"]),
                     "R_1 final tier: median": f"{x['t1_R_final_median']:.4f}", "min": f"{x['t1_R_final_min']:.4f}",
                     "max": f"{x['t1_R_final_max']:.4f}"})
    return _md(pd.DataFrame(rows))


def block_r0() -> str:
    """A3 (a): R0 = r_2(0)/s_2 at the terminal-stage freeze and |peak|/R0."""
    r = _read("r0.csv")
    rows = []
    for _, x in r.iterrows():
        rows.append({"arm": x["arm"], "q": int(x["q"]), "R0 final (median)": f"{x['t2_R0_final_median']:.4f}",
                     "R0 final [min, max]": f"[{x['t2_R0_final_min']:.4f}, {x['t2_R0_final_max']:.4f}]",
                     "R0 dev (median)": f"{x['t2_R0_dev_median']:.4f}",
                     "|peak|/R0 final (median)": f"{x['t2_peak_over_R0_final_median']:.3f}",
                     "|peak|/R0 final [min, max]": f"[{x['t2_peak_over_R0_final_min']:.3f}, {x['t2_peak_over_R0_final_max']:.3f}]",
                     "|peak|/R0 dev (median)": f"{x['t2_peak_over_R0_dev_median']:.3f}",
                     "calibration median": f"{x['calib_median_peak_over_R0']:.2f}",
                     "linearised (2k+a)/(2k)": f"{x['linearised_factor']:.3f}"})
    return _md(pd.DataFrame(rows))


def block_strata(stratum: str = "mid", tier: str = "final") -> str:
    """A3 (b): closed-form error and max r_2/s_2 per stratum and side at the terminal-stage freeze."""
    s = _read("strata_summary.csv")
    s = s[(s.tier == tier) & (s.stratum == stratum)]
    rows = []
    for arm in ARMS + COMPARATORS:
        for q in QS:
            cells = {"arm": arm, "q": q}
            for side in ("d<0", "d>0"):
                x = s[(s.arm == arm) & (s.q == q) & (s.side == side)]
                if len(x):
                    x = x.iloc[0]
                    cells[f"{side}: mean signed err"] = f"{x['err_mean_mean']:+.3f}"
                    cells[f"{side}: RMSE"] = f"{x['err_rmse_mean']:.3f}"
                    cells[f"{side}: max |err| (max over runs)"] = f"{x['err_max_abs_max']:.2f}"
                    cells[f"{side}: max r/s (mean, max)"] = f"{x['max_r_over_s_mean']:.3f}, {x['max_r_over_s_max']:.3f}"
            rows.append(cells)
    return _md(pd.DataFrame(rows))


def block_strata_middle() -> str:
    return block_strata("mid")


def block_strata_near() -> str:
    return block_strata("near")


def block_strata_tail() -> str:
    return block_strata("tail")


def block_shares() -> str:
    """Measured start shares per stratum over the training blocks (mean over runs)."""
    per = _read("per_run.csv")
    rows = []
    for arm in ARMS:
        for q in QS:
            g = per[(per.arm == arm) & (per.q == q) & (per.status == "done")]
            rows.append({"arm": arm, "q": q, "design lambda_P": "-" if not np.isfinite(g["design_lambda_P"].iloc[0]) else f"{g['design_lambda_P'].iloc[0]:.4f}",
                         "design lambda_M": "-" if not np.isfinite(g["design_lambda_M"].iloc[0]) else f"{g['design_lambda_M'].iloc[0]:.4f}",
                         "measured tail": f"{g['t2_share_tail_train'].mean():.4f}", "measured near-tie": f"{g['t2_share_near_train'].mean():.4f}",
                         "measured middle": f"{g['t2_share_mid_train'].mean():.4f}",
                         "raw-draw clamp share, learner": f"{g['t2_clamp_L_frac'].mean():.4f}" if "t2_clamp_L_frac" in g else "n/a"})
    return _md(pd.DataFrame(rows))


def block_arm(arm: str) -> str:
    """One arm: paired tables against the comparators, rule record, budget, stage 1."""
    out: List[str] = []
    out.append("Paired against `parents_A` (arm - baseline; `*` = the 95 % interval excludes 0):\n")
    out.append(_paired("vs parents_A", [arm], PAIR_METRICS))
    if arm in RULE_ARMS:
        out.append("\nPaired against `MS_base2400` (rule, polishing and sampler at matched budget; descriptive):\n")
        out.append(_paired("vs MS_base2400", [arm], PAIR_METRICS))
    if arm in RULE_ARMS[1:]:
        out.append("\nPaired against `MS_rule` (the sampler's effect net of the rule; descriptive):\n")
        out.append(_paired("vs MS_rule", [arm], PAIR_METRICS[:4]))
    return "\n".join(out)


def block_side_by_side() -> str:
    """Arms side by side, both q pooled (20 runs per arm): criterion flags, |peak| counts, signed peak, gates."""
    per = _read("per_run.csv")
    dec = _read("decision_inputs.csv")
    crit = _read("criterion.csv").set_index("arm")
    rows = []
    for arm in ARMS + COMPARATORS:
        d = dec[(dec.arm == arm) & (dec.q.astype(str) == "both")].iloc[0]
        g = per[(per.arm == arm) & (per.status == "done")]
        sm = pd.to_numeric(g["smoothed_share_peak_gap_d0"], errors="coerce").dropna()
        if arm in crit.index:
            c = crit.loc[arm]
            a50, a60, b = ("yes" if c["a_q50"] else "no"), ("yes" if c["a_q60"] else "no"), c["b_status"]
        else:
            a50 = a60 = b = "-"
        rows.append({"arm": arm, "(a) q=50": a50, "(a) q=60": a60, "(b)": b,
                     "abs(peak)<=0.05 (of 20)": int(d["n_abs_peak_le_0.05"]), "mean abs(peak)": f"{d['abs_peak_mean']:.4f}",
                     "signed peak mean [95% CI]": _ci(d["signed_peak_mean"], d["signed_peak_ci_lo"],
                                                      d["signed_peak_ci_hi"], 4),
                     "mean RMSE_pos": f"{d['rmse_mean']:.4f}", "mean tail mean": f"{d['tail_mean_mean']:.4f}",
                     "G-A+G-N(eta) (of 20)": int(d["n_gate_pass"]), "terminal updates": f"{d['t2_updates_mean']:.0f}",
                     "wall s (mean; parents_A: terminal stage only)": f"{d['total_wall_sec_mean']:.0f}",
                     "smoothed share, median": f"{sm.median():.2f}" if len(sm) else "n/a"})
    return _md(pd.DataFrame(rows))


def block_budget_share() -> str:
    """Mean |peak| difference against parents_A of the sampler arms and of the budget control, and their ratio."""
    crit = _read("criterion.csv").set_index("arm")
    rows = []
    for arm in ["MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5"]:
        r = {"arm": arm}
        for q in QS:
            arm_d, b_d = crit.loc[arm, f"mean_q{q}"], crit.loc["MS_base2400", f"mean_q{q}"]
            r[f"q={q}: arm - parents_A"] = f"{arm_d:+.4f}"
            r[f"q={q}: MS_base2400 - parents_A"] = f"{b_d:+.4f}"
            r[f"q={q}: control / arm"] = f"{b_d / arm_d:.2f}"
        rows.append(r)
    return _md(pd.DataFrame(rows))


def block_seeds() -> str:
    """Rough planning arithmetic: seeds for 80 % power at the observed paired differences (rule arms - MS_base2400).

    Two-sided alpha = 0.05, power 0.80, normal approximation, ``n = ((1.96 + 0.8416) SD / |mean|)^2`` with the sample
    SD (ddof = 1) of the ten paired differences of |peak error|; per q, not for the both-q criterion.
    """
    sl = _read("paired_seed_level.csv")
    x = sl[(sl.comparison == "vs MS_base2400") & (sl.metric == "stage2_peak_rel_err_abs")]
    g = x.groupby(["arm", "q"])["diff"].agg(["mean", "std"])
    rows = []
    for arm in RULE_ARMS:
        r = {"arm": arm}
        for q in QS:
            m, sd = g.loc[(arm, q), "mean"], g.loc[(arm, q), "std"]
            n = ((1.96 + 0.8416) * sd / abs(m)) ** 2 if abs(m) > 0 else np.inf
            r[f"q={q}: mean diff"] = f"{m:+.4f}"
            r[f"q={q}: SD of the 10 paired differences"] = f"{sd:.4f}"
            r[f"q={q}: seeds for 80 % power (normal approx.)"] = ">1000" if n > 1000 else f"{n:.0f}"
        rows.append(r)
    return _md(pd.DataFrame(rows))


BLOCKS: Dict[str, Callable[[], str]] = {
    "primary": block_primary, "secondary": block_secondary, "overview": block_overview,
    "paired_parents": block_paired_parents, "paired_b2400": block_paired_b2400, "paired_rule": block_paired_rule,
    "rule": block_rule, "classif": block_classif, "budget": block_budget, "stage1": block_stage1,
    "stage1_vs_rehearsal": block_stage1_vs_rehearsal, "stage1_r1": block_stage1_r1, "r0": block_r0,
    "strata_middle": block_strata_middle, "strata_near": block_strata_near, "strata_tail": block_strata_tail,
    "shares": block_shares, "side_by_side": block_side_by_side, "budget_share": block_budget_share,
    "seeds": block_seeds}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI."""
    global A
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--block", help="a block name, or arm:<arm>")
    p.add_argument("--analysis", default=str(A))
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    A = Path(a.analysis)
    if a.list or not a.block:
        print("\n".join(list(BLOCKS) + ["arm:<%s>" % "|".join(ARMS)]))
        return 0
    if a.block.startswith("arm:"):
        print(block_arm(a.block.split(":", 1)[1]))
    else:
        print(BLOCKS[a.block]())
    return 0


if __name__ == "__main__":
    sys.exit(main())
