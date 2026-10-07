#!/usr/bin/env python3
"""Markdown tables of the MS-R2 pilot reports, read from the analysis CSVs of ``tools/ms/r2_analysis.py`` and the launch
files (no number is typed by hand). Everything except the ``primary`` block is descriptive.

Usage (repository root):
    python reports/ms/r2/report_scripts/pilot_tables.py --list
    python reports/ms/r2/report_scripts/pilot_tables.py --block primary
    python reports/ms/r2/report_scripts/pilot_tables.py --block checks --pilot results/ms_r2/pilot
``--analysis`` (default ``results/ms_r2/analysis``) and ``--pilot`` (default ``results/ms_r2/pilot``) name the inputs.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ARMS = ["NL_bb_s1", "NL_bb_s4", "NL_bb_s16", "NL_st_s1", "NL_st_s4", "NL_st_s16"]
REFS = ["MS_base2400", "MS_s35a5", "parents_A", "rehearsal_v2_0"]
QS = [50, 60]
A: Path = Path("results/ms_r2/analysis")
P: Path = Path("results/ms_r2/pilot")


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)

    def esc(x: object) -> str:
        return str(x).replace("|", "\\|")
    out = ["| " + " | ".join(esc(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(esc(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(out)


def _ci(m: float, lo: float, hi: float, p: int = 4) -> str:
    """``+m [lo, hi]`` with a star when the interval excludes 0."""
    if any(not np.isfinite(x) for x in (m, lo, hi)):
        return "nan"
    return f"{m:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]" + ("" if lo <= 0.0 <= hi else " *")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(A / name, float_precision="round_trip")


def _per_run() -> pd.DataFrame:
    return _read("per_run.csv")


def block_checks() -> str:
    """Post-launch checks of ``launch_checks.json``."""
    c = json.loads((P / "launch_checks.json").read_text())
    b, s = c["base"], c["summary"]
    rows = [("status exit 0 / manifest at the launch commit, clean tree / files complete / global-RNG assertions / tail-share coverage",
             " / ".join(f"{b['n_ok_per_check'][k]}/{b['n_runs']}" for k in ("status", "manifest", "files", "global_rng", "tail_share_coverage"))),
            ("start-share tests (flagged at |z| > 3; expected by chance)", f"{b['start_share_tests']['n_tests']} tests, {b['start_share_tests']['n_flagged']} flagged "
                                                                         f"({b['start_share_tests']['expected_flagged_under_null']:.1f} expected)"),
            ("applied scale equals the D2 schedule at every update, schedule and exported scales recorded", f"{s['scale_ok']}/{s['scale_n']}"),
            ("C-NL (s = 4, 16 against s = 1 through update 2001, u02025 differs)", f"{s['C_NL_pass']}/{s['C_NL_n']}"),
            ("C-MS3 (`NL_bb_s1` against MS-R1 `MS_base2400`)", f"{s['C_MS3_pass']}/{s['C_MS3_n']}"),
            ("C-MS4 (`NL_st_s1` against MS-R1 `MS_s35a5`, through 2001 or the last update before its first polishing block)", f"{s['C_MS4_pass']}/{s['C_MS4_n']}"),
            ("all checks pass", str(c["all_ok"]))]
    return _md(pd.DataFrame(rows, columns=["check", "result"]))


def block_launch() -> str:
    """Facts of the launch record."""
    f = sorted(glob.glob(str(P / "launch_2*.json")))
    r = json.loads(Path(f[0]).read_text())
    w = sorted(x["wall_sec"] for x in r["runs"])
    lines = [f"- launch record `{f[0]}`: wave `{r['wave']}`, {r['n_planned']} planned, {len(r['runs'])} finished, state `{r['state']}`, workers {r['workers']}, "
             f"started {r['started']}",
             f"- nonzero exits: {sum(1 for x in r['runs'] if x.get('returncode') != 0)}; wall per run min {w[0]:.0f} s, median {w[len(w) // 2]:.0f} s, max {w[-1]:.0f} s",
             f"- HEAD `{r['head'][:8]}`, code commit argument `{r['code_commit']}`; `git diff --stat <code commit> HEAD -- run utils envs agents protocols` is "
             f"{'empty' if not r['diff_stat_run_code_to_head'] else 'NOT empty'}; `git status --porcelain` {'empty' if not r['status_porcelain'] else r['status_porcelain']}",
             f"- parameter file SHA-256 `{r['params_sha256']}`; nproc {r['nproc']}, load average at start {', '.join(f'{x:.1f}' for x in r['loadavg_at_start'])}, "
             f"at the end {', '.join(f'{x:.1f}' for x in r['loadavg_at_end'])}, free disk {r['disk_free_bytes'] / 1e9:.0f} GB"]
    return "\n".join(lines)


def block_primary() -> str:
    """The pre-registered criterion: parts (a) and (b) per (sampler, s) against the same sampler's s = 1 arm."""
    c = _read("criterion.csv")
    rows = []
    for r in c.itertuples():
        b = r.b_status + (f" ({r.b_violations})" if isinstance(r.b_violations, str) and r.b_violations else "")
        rows.append({"arm": f"`{r.arm}`", "baseline": f"`{r.baseline}`",
                     "(a) q=50 mean [95% CI]": _ci(r.mean_q50, r.ci_mean_lo_q50, r.ci_mean_hi_q50, 5), "met q=50": "yes" if r.a_q50 else "no",
                     "(a) q=60 mean [95% CI]": _ci(r.mean_q60, r.ci_mean_lo_q60, r.ci_mean_hi_q60, 5), "met q=60": "yes" if r.a_q60 else "no",
                     "pairs": f"{int(r.n_pairs_q50)}+{int(r.n_pairs_q60)}", "(b)": b, "overall": r.overall})
    return _md(pd.DataFrame(rows))


def block_overview() -> str:
    """Per arm and q: |peak|, signed peak with interval, RMSE_pos, tail, eta_2, gates, R0, R."""
    per = _per_run()
    per = per[per.status == "done"]
    rows = []
    for arm in ARMS + REFS:
        for q in QS:
            g = per[(per.arm == arm) & (per.q == q)]
            if g.empty:
                continue
            sg = g["stage2_peak_rel_err_signed"].to_numpy(dtype=float)
            rng = np.random.default_rng(20261007)
            bs = sg[rng.integers(0, len(sg), size=(10000, len(sg)))].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ab = g["stage2_peak_rel_err_abs"]
            rows.append({"arm": arm, "q": q, "n": len(g), "abs(peak)<=0.05": int((ab <= 0.05).sum()), "mean abs(peak)": f"{ab.mean():.4f}",
                         "signed peak mean [95% CI]": _ci(sg.mean(), lo, hi), "signed<0": int((sg < 0).sum()),
                         "RMSE_pos": f"{g['stage2_rmse_pos_over_g2_0'].mean():.4f}", "tail mean": f"{g['stage2_tail_mean_over_g2_0'].mean():.4f}",
                         "eta2/DW": f"{g['eta_T_over_dw'].mean():.5f}", "G-A+G-N(eta)": f"{int(g['gate_pass'].astype(str).eq('True').sum())}/{len(g)}",
                         "R0 (median)": f"{g['t2_R0_final'].median():.4f}", "R (median)": f"{g['t2_R_final'].median():.4f}"})
    return _md(pd.DataFrame(rows))


def block_decomp() -> str:
    """The freeze decomposition per arm and q (means over the seeds, effort units) and the floor in percent of e*(0)."""
    per = _per_run()
    per = per[per.status == "done"]
    rows = []
    for arm in ARMS + REFS:
        for q in QS:
            g = per[(per.arm == arm) & (per.q == q)]
            if g.empty or not np.isfinite(g["smoothing"]).any():
                continue
            rows.append({"arm": arm, "q": q, "n": len(g), "sigma_2(0)": f"{g['sigma_2_0'].mean():.3f}", "gap": f"{g['gap'].mean():.3f}",
                         "smoothing part": f"{g['smoothing'].mean():.3f}", "remainder": f"{g['remainder'].mean():.3f}",
                         "remainder median": f"{g['remainder'].median():.3f}", "remainder seed SD": f"{g['remainder'].std(ddof=1):.3f}",
                         "gap % of e*(0)": f"{100 * g['gap_rel'].mean():.2f}", "smoothing % of e*(0)": f"{100 * g['smoothing_rel'].mean():.2f}",
                         "remainder % of e*(0)": f"{100 * g['remainder_rel'].mean():.2f}"})
    return _md(pd.DataFrame(rows))


def block_predictions() -> str:
    """D5 predictions against the outcome."""
    p = _read("predictions.csv")
    rows = []
    for r in p.itertuples():
        rows.append({"arm": r.arm, "q": r.q, "sigma_2(0) mean": f"{r.sigma_2_0_mean:.3f}",
                     "sigma(s) / [sigma(1)/sqrt(s)]: mean [min, max]": "-" if not np.isfinite(r.sigma_ratio_mean) else
                     f"{r.sigma_ratio_mean:.3f} [{r.sigma_ratio_min:.3f}, {r.sigma_ratio_max:.3f}]",
                     "smoothing / formula: mean [min, max]": f"{r.smoothing_over_formula_mean:.5f} [{r.smoothing_over_formula_min:.5f}, {r.smoothing_over_formula_max:.5f}]",
                     "runs outside 0.5 %": int(r.n_outside_0p5pct), "floor % of e*(0): mean": f"{r.floor_pct_mean:.2f}",
                     "planning floor %": f"{r.planning_floor_pct:.2f}"})
    return _md(pd.DataFrame(rows))


SECONDARY = [("smoothing", "smoothing part"), ("remainder", "remainder"), ("gap", "gap"), ("sigma_2_0", "sigma_2(0)"),
             ("stage2_peak_rel_err_abs", "abs(peak error)"), ("stage2_peak_rel_err_signed", "signed peak error"),
             ("stage2_rmse_pos_over_g2_0", "RMSE_pos/e2*(0)"), ("stage2_tail_mean_over_g2_0", "tail mean/e2*(0)"),
             ("eta_T_over_dw", "eta_2/DW"), ("t2_R0_final", "R0"), ("t2_R_final", "R")]


def _paired(df: pd.DataFrame, arms: Sequence[str], metrics: Sequence = SECONDARY) -> str:
    rows = []
    for arm in arms:
        for m, lab in metrics:
            cells = {"arm": f"`{arm}`", "baseline": "", "metric": lab}
            ok = False
            for q in QS:
                r = df[(df.arm == arm) & (df.q == q) & (df.metric == m)]
                if len(r):
                    r = r.iloc[0]
                    cells["baseline"] = f"`{r.baseline}`"
                    cells[f"q={q}: mean [95% CI] (arm - baseline)"] = _ci(r["mean"], r.ci_mean_lo, r.ci_mean_hi, 4 if "peak" in m or "eta" in m or "rmse" in m or "tail" in m else 3)
                    cells[f"q={q}: seeds lower"] = f"{int(r.n_neg)}/{int(r.n_pairs)}"
                    ok = True
                else:
                    cells[f"q={q}: mean [95% CI] (arm - baseline)"] = "n/a"
                    cells[f"q={q}: seeds lower"] = "n/a"
            if ok:
                rows.append(cells)
    return _md(pd.DataFrame(rows))


def block_secondary() -> str:
    """Paired changes against the same sampler's s = 1 arm."""
    return _paired(_read("paired_secondary.csv"), [a for a in ARMS if not a.endswith("_s1")])


def block_transmission() -> str:
    """Transmission ratio: (change of the gap) / (change of the smoothing part) per arm and q."""
    t = _read("transmission.csv")
    return _md(pd.DataFrame([{"arm": f"`{r.arm}`", "q": r.q, "n pairs": int(r.n_pairs), "mean change of the gap": f"{r.mean_d_gap:+.3f}",
                              "mean change of the smoothing part": f"{r.mean_d_smoothing:+.3f}",
                              "ratio [95% CI]": f"{r.ratio:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", "per-seed ratios": r.per_seed_ratios} for r in t.itertuples()]))


def block_interaction() -> str:
    """(NL_st_s - NL_st_s1) - (NL_bb_s - NL_bb_s1) per q."""
    t = _read("interaction.csv")
    rows = [{"s": int(r.s), "q": r.q, "metric": r.metric, "n": int(r.n_pairs), "mean [95% CI]": _ci(r.mean, r.ci_mean_lo, r.ci_mean_hi, 3 if r.metric != "stage2_peak_rel_err_abs" else 4),
             "seeds < 0": int(r.n_neg)} for r in t.itertuples()]
    return _md(pd.DataFrame(rows))


def block_vs_parents() -> str:
    """Every NL arm against parents_A with MS-R1's criterion."""
    c = _read("criterion_vs_parents_A.csv")
    rows = []
    for r in c.itertuples():
        b = r.b_status + (f" ({r.b_violations})" if isinstance(r.b_violations, str) and r.b_violations else "")
        rows.append({"arm": f"`{r.arm}`", "(a) q=50 mean [95% CI]": _ci(r.mean_q50, r.ci_mean_lo_q50, r.ci_mean_hi_q50, 5), "met q=50": "yes" if r.a_q50 else "no",
                     "(a) q=60 mean [95% CI]": _ci(r.mean_q60, r.ci_mean_lo_q60, r.ci_mean_hi_q60, 5), "met q=60": "yes" if r.a_q60 else "no", "(b)": b, "overall": r.overall})
    return _md(pd.DataFrame(rows))


def block_refs() -> str:
    """The s = 1 arms against their MS-R1 references."""
    return _paired(_read("paired_vs_ms_r1_refs.csv"), ["NL_bb_s1", "NL_st_s1"])


def block_segments() -> str:
    """Segment means of the PPO diagnostics and segment-end values of the tie quantities, per arm and q."""
    s = _read("segments.csv")
    rows = [{"arm": r.arm, "q": r.q, "segment": r.segment, "local": f"{int(r.first_local)}-{int(r.last_local)}", "KL": f"{r.kl_final_epoch_mean:.4f}",
             "clip frac": f"{r.clip_frac_mean:.3f}", "adv SD": f"{r.adv_raw_std_mean:.3f}", "value loss": f"{r.value_loss_mean:.3f}",
             "scale [min, max]": f"{r.conc_scale_min:.3g}, {r.conc_scale_max:.3g}", "end check": "-" if not np.isfinite(r.end_check_local) else f"{r.end_check_local:.0f}",
             "e_hat(0) at end": f"{r.e_hat_2_0_end:.2f}", "sigma(0) at end": f"{r.sigma_0_end:.3f}", "smoothing at end": f"{r.smoothing_end:.3f}",
             "remainder at end": f"{r.remainder_end:.3f}"} for r in s.itertuples()]
    return _md(pd.DataFrame(rows))


def block_trajectory() -> str:
    """Mean over seeds of e_hat_2(0), sigma_2(0), the smoothing part and the remainder at the checks 1800, 2000, 2200, 2400, 2600, 2800."""
    t = _read("trajectory_checks.csv")
    rows = []
    for arm in ARMS:
        for q in QS:
            row = {"arm": arm, "q": q}
            for u in (1800, 2000, 2200, 2400, 2600, 2800):
                g = t[(t.arm == arm) & (t.q == q) & (t["update"] == u)]
                row[f"u{u}: e_hat, sigma, smoothing, remainder"] = "n/a" if g.empty else (
                    f"{g.e2_at_0.mean():.2f}, {g.sigma_0.mean():.2f}, {g.smoothing.mean():.2f}, {g.remainder.mean():.2f}")
            rows.append(row)
    return _md(pd.DataFrame(rows))


def block_strata(stratum: str = "mid") -> str:
    """A3 (b) strata at the terminal freeze (final tier)."""
    s = _read("strata_summary.csv")
    s = s[(s.tier == "final") & (s.stratum == stratum)]
    rows = []
    for arm in ARMS + REFS:
        for q in QS:
            cells = {"arm": arm, "q": q}
            for side in ("d<0", "d>0"):
                x = s[(s.arm == arm) & (s.q == q) & (s.side == side)]
                if len(x):
                    x = x.iloc[0]
                    cells[f"{side}: mean signed err"] = f"{x['err_mean_mean']:+.3f}"
                    cells[f"{side}: RMSE"] = f"{x['err_rmse_mean']:.3f}"
                    cells[f"{side}: max r/s (mean, max)"] = f"{x['max_r_over_s_mean']:.3f}, {x['max_r_over_s_max']:.3f}"
            if len(cells) > 2:
                rows.append(cells)
    return _md(pd.DataFrame(rows))


def block_strata_near() -> str:
    return block_strata("near")


def block_strata_tail() -> str:
    return block_strata("tail")


def block_stage1() -> str:
    """Stage 1 per arm and q."""
    s = _read("stage1.csv")
    rows = []
    for x in s.itertuples():
        rows.append({"arm": x.arm, "q": x.q, "n": int(x.n_runs), "signed error mean [95% CI]": _ci(x.stage1_rel_err_signed_mean, x.stage1_rel_err_signed_ci_lo, x.stage1_rel_err_signed_ci_hi),
                     "abs error mean": f"{x.stage1_rel_err_abs_mean:.4f}", "R_1 final (median)": "n/a" if not np.isfinite(x.t1_R_final_median) else f"{x.t1_R_final_median:.4f}",
                     "G-S": f"{int(x.n_G_S_pass)}/{int(x.n_runs)}", "G-F": f"{int(x.n_G_F_pass)}/{int(x.n_runs)}", "G-N(Gmax)": f"{int(x.n_G_N_gmax_pass)}/{int(x.n_runs)}",
                     "v2.0 combination": f"{int(x.n_v20_combination_pass)}/{int(x.n_runs)}"})
    return _md(pd.DataFrame(rows))


def block_gates() -> str:
    """Gate counts per arm and q."""
    g = _read("gates.csv")
    return _md(pd.DataFrame([{"arm": r.arm, "q": r.q, "runs done": int(r.n_done), "G-A (eta)": r.n_G_A_eta, "G-A (RMSE)": r.n_G_A_rmse, "G-A (tail)": r.n_G_A_tail,
                              "G-A": r.n_G_A, "G-N(eta)": r.n_G_N_eta, "G-A and G-N(eta)": r.n_G_A_and_G_N_eta, "v2.0 combination": r.n_v20_combination} for r in g.itertuples()]))


def block_shares() -> str:
    """Measured start shares per stratum (training updates) and the design shares."""
    per = _per_run()
    per = per[(per.status == "done") & per.arm.isin(ARMS)]
    rows = []
    for arm in ARMS:
        for q in QS:
            g = per[(per.arm == arm) & (per.q == q)]
            rows.append({"arm": arm, "q": q, "design lambda_P": f"{g['design_lambda_P'].iloc[0]:.4f}" if np.isfinite(g["design_lambda_P"].iloc[0]) else "-",
                         "design lambda_M": f"{g['design_lambda_M'].iloc[0]:.4f}" if np.isfinite(g["design_lambda_M"].iloc[0]) else "-",
                         "measured tail": f"{g['t2_share_tail_train'].mean():.4f}", "measured near-tie": f"{g['t2_share_near_train'].mean():.4f}",
                         "measured middle": f"{g['t2_share_mid_train'].mean():.4f}", "learner clamp share (mean)": f"{g['t2_clamp_L_frac'].mean():.4f}"})
    return _md(pd.DataFrame(rows))


def block_budget() -> str:
    """Budget to the freeze (medians)."""
    b = _read("budget.csv")
    rows = []
    for arm in ARMS:
        for q in QS:
            g = b[(b.arm == arm) & (b.q == q)].set_index("metric")["median"]
            rows.append({"arm": arm, "q": q, "terminal-stage updates": f"{g.get('t2_updates', np.nan):.0f}", "stage-1 updates": f"{g.get('t1_updates', np.nan):.0f}",
                         "total episodes": f"{g.get('total_episodes', np.nan):.0f}", "wall s (median)": f"{g.get('total_wall_sec', np.nan):.0f}"})
    return _md(pd.DataFrame(rows))


def block_sampler() -> str:
    """What the stratified sampler adds at each s: paired (q, seed) differences NL_st_s - NL_bb_s (descriptive; not a D5 table).

    Computed here from ``per_run.csv`` (no CSV of the analysis tool holds it): the mean of the per-seed difference with the 95 %
    percentile bootstrap interval, 10,000 resamples, a fresh ``default_rng(20261007)`` per (q, statistic) in table order.
    """
    per = _per_run()
    per = per[per.status == "done"]
    stats = [("smoothing", "smoothing part", 3), ("remainder", "remainder", 3), ("gap", "gap", 3), ("stage2_peak_rel_err_abs", "abs(peak error)", 4),
             ("stage2_rmse_pos_over_g2_0", "RMSE_pos/e2*(0)", 4), ("stage2_tail_mean_over_g2_0", "tail mean/e2*(0)", 4)]
    rows = []
    for sv in (1, 4, 16):
        for col, lab, p in stats:
            cells = {"s": sv, "metric": lab}
            for q in QS:
                a = per[(per.arm == f"NL_st_s{sv}") & (per.q == q)].set_index("seed")[col]
                b = per[(per.arm == f"NL_bb_s{sv}") & (per.q == q)].set_index("seed")[col]
                d = (a - b.reindex(a.index)).dropna().to_numpy(dtype=float)
                rng = np.random.default_rng(20261007)
                bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                cells[f"q={q}: mean [95% CI] (st - bb)"] = _ci(d.mean(), lo, hi, p)
                cells[f"q={q}: seeds lower"] = f"{int((d < 0).sum())}/{len(d)}"
            rows.append(cells)
    return _md(pd.DataFrame(rows))


def block_window() -> str:
    """Supplement (post hoc, not pre-registered): the paired changes of the decomposition averaged over the last eight checks.

    The window is the checks at updates 2625-2800 of ``trajectory_checks.csv`` (the LR falls from 1.5e-4 at update 2625 to 3e-5 at 2800), averaged per run; then the
    arm - (same sampler, s = 1) difference per (q, seed), mean with the 95 % percentile bootstrap interval (10,000 resamples, a fresh
    ``default_rng(20261007)`` per (q, statistic) in table order). It exists because the freeze is one checkpoint per run.
    """
    t = _read("trajectory_checks.csv")
    t = t[(t.status == "done") & t.arm.isin(ARMS) & (t["update"] >= 2625)]
    w = t.groupby(["arm", "q", "seed"])[["smoothing", "remainder", "gap", "stage2_peak_rel_err_abs"]].mean().reset_index()
    stats = [("smoothing", "smoothing part", 3), ("remainder", "remainder", 3), ("gap", "gap", 3), ("stage2_peak_rel_err_abs", "abs(peak error)", 4)]
    rows = []
    for arm in [a for a in ARMS if not a.endswith("_s1")]:
        base = arm.rsplit("_s", 1)[0] + "_s1"
        for col, lab, p in stats:
            cells = {"arm": f"`{arm}`", "baseline": f"`{base}`", "metric (mean of the 8 checks)": lab}
            for q in QS:
                a = w[(w.arm == arm) & (w.q == q)].set_index("seed")[col]
                b = w[(w.arm == base) & (w.q == q)].set_index("seed")[col]
                d = (a - b.reindex(a.index)).dropna().to_numpy(dtype=float)
                rng = np.random.default_rng(20261007)
                bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                cells[f"q={q}: mean [95% CI] (arm - baseline)"] = _ci(d.mean(), lo, hi, p)
                cells[f"q={q}: seeds lower"] = f"{int((d < 0).sum())}/{len(d)}"
            rows.append(cells)
    return _md(pd.DataFrame(rows))


def block_side_by_side() -> str:
    """The six arms and the references in one table (means over the ten seeds; `rehearsal_v2_0` stands for the `parents_A` candidate)."""
    per = _per_run()
    per = per[per.status == "done"]
    rows = []
    for arm in ARMS + ["MS_base2400", "MS_s35a5", "rehearsal_v2_0"]:
        for q in QS:
            g = per[(per.arm == arm) & (per.q == q)]
            if g.empty:
                continue
            rows.append({"arm": arm, "q": q, "mean abs(peak)": f"{g['stage2_peak_rel_err_abs'].mean():.4f}", "abs(peak)<=0.05": f"{int((g['stage2_peak_rel_err_abs'] <= 0.05).sum())}/{len(g)}",
                         "sigma_2(0)": f"{g['sigma_2_0'].mean():.3f}", "smoothing part": f"{g['smoothing'].mean():.3f}", "remainder": f"{g['remainder'].mean():.3f}",
                         "gap": f"{g['gap'].mean():.3f}", "RMSE_pos/e2*(0)": f"{g['stage2_rmse_pos_over_g2_0'].mean():.4f}",
                         "tail mean/e2*(0)": f"{g['stage2_tail_mean_over_g2_0'].mean():.4f}", "eta2/DW": f"{g['eta_T_over_dw'].mean():.5f}",
                         "R0 (median)": f"{g['t2_R0_final'].median():.4f}", "R (median)": f"{g['t2_R_final'].median():.4f}"})
    return _md(pd.DataFrame(rows))


def block_directions() -> str:
    """Over the eight (arm, q) cells of the secondary table: how many changes against s = 1 are negative, and how many intervals exclude 0."""
    t = _read("paired_secondary.csv")
    t = t[t.arm.isin([a for a in ARMS if not a.endswith("_s1")])]
    rows = []
    for m, lab in SECONDARY:
        x = t[t.metric == m]
        if x.empty:
            continue
        rows.append({"metric": lab, "cells": len(x), "mean change < 0": int((x["mean"] < 0).sum()), "interval below 0": int((x.ci_mean_hi < 0).sum()),
                     "interval above 0": int((x.ci_mean_lo > 0).sum()), "range of the mean changes": f"{x['mean'].min():+.4f} to {x['mean'].max():+.4f}"})
    return _md(pd.DataFrame(rows))


def block_tie_effort() -> str:
    """Mean tie effort at the freeze: the closed form e*(0), the smoothed target e_sigma(0) and the learned e_hat_2(0) (effort units)."""
    per = _per_run()
    per = per[per.status == "done"]
    rows = []
    for arm in ARMS:
        for q in QS:
            g = per[(per.arm == arm) & (per.q == q)]
            rows.append({"arm": arm, "q": q, "e*(0)": f"{g['g2_at_0'].mean():.2f}", "e_sigma(0)": f"{g['smoothed_e_pred_0'].mean():.2f}",
                         "e_hat_2(0)": f"{g['e2_at_0'].mean():.2f}", "e_hat_2(0) seed SD": f"{g['e2_at_0'].std(ddof=1):.2f}"})
    return _md(pd.DataFrame(rows))


BLOCKS: Dict[str, Callable[[], str]] = {
    "checks": block_checks, "launch": block_launch, "primary": block_primary, "overview": block_overview, "decomp": block_decomp,
    "predictions": block_predictions, "secondary": block_secondary, "transmission": block_transmission, "interaction": block_interaction,
    "vs_parents": block_vs_parents, "refs": block_refs, "segments": block_segments, "trajectory": block_trajectory,
    "strata_mid": block_strata, "strata_near": block_strata_near, "strata_tail": block_strata_tail, "stage1": block_stage1,
    "gates": block_gates, "shares": block_shares, "budget": block_budget, "sampler": block_sampler, "window": block_window,
    "side_by_side": block_side_by_side, "directions": block_directions, "tie_effort": block_tie_effort}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI."""
    global A, P
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--block")
    p.add_argument("--analysis", default=str(A))
    p.add_argument("--pilot", default=str(P))
    p.add_argument("--list", action="store_true")
    p.add_argument("--arm", default=None, help="keep only the table rows whose first cell is this arm")
    a = p.parse_args(argv)
    A, P = Path(a.analysis), Path(a.pilot)
    if a.list or not a.block:
        print("\n".join(BLOCKS))
        return 0
    text = BLOCKS[a.block]()
    if a.arm:
        lines = text.split("\n")
        keep = lines[:2] + [ln for ln in lines[2:] if ln.split("|")[1].strip().strip("`") == a.arm]
        text = "\n".join(keep)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
