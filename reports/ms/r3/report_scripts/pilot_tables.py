#!/usr/bin/env python3
"""Markdown tables of the MS-R3 pilot reports, read from the analysis CSVs of ``tools/ms/r3_analysis.py`` and the launch files
(no number is typed by hand). Everything except the ``primary`` block is descriptive.

Usage (repository root):
    python reports/ms/r3/report_scripts/pilot_tables.py --list
    python reports/ms/r3/report_scripts/pilot_tables.py --block primary
    python reports/ms/r3/report_scripts/pilot_tables.py --block overview --arm relu_bb_s1
``--analysis`` (default ``results/ms_r3/analysis``) and ``--pilot`` (default ``results/ms_r3/pilot``) name the inputs; ``--arm`` keeps
the table rows whose first cell is that arm (or actor, for the blocks whose first cell is the actor).
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

ACTORS = ["t1", "relu", "t10"]
STARTS = ["bb", "st"]
SCALES = [1, 16]
ARMS = [f"{a}_{s}_s{v}" for a in ACTORS for s in STARTS for v in SCALES]
REFS = ["parents_A", "rehearsal_v2_0"]
QS = [50, 60]
A: Path = Path("results/ms_r3/analysis")
P: Path = Path("results/ms_r3/pilot")
#: metrics of the paired tables in report order: (column of ``paired_secondary.csv``, label, decimals)
METRICS = [("stage2_peak_rel_err_abs", "abs(peak error)", 4), ("stage2_peak_rel_err_signed", "signed peak error", 4),
           ("gap", "gap", 3), ("smoothing", "smoothing part", 3), ("remainder", "remainder", 3), ("sigma_2_0", "sigma_2(0)", 3),
           ("stage2_rmse_pos_over_g2_0", "RMSE_pos/e2*(0)", 4), ("stage2_tail_mean_over_g2_0", "tail mean/e2*(0)", 4),
           ("eta_T_over_dw", "eta_2/DW", 5), ("t2_R0_final", "R0", 4), ("t2_R_final", "R", 4), ("w_eff", "w_eff (units of d)", 3),
           ("peak_locfree_rel_err", "location-free peak error", 4), ("sym_err_max_rel", "symmetry error/e2*(0)", 4),
           ("w1d_abs_max", "max abs first-layer d-weight (d/B)", 3)]


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
    return pd.read_csv(A / name, float_precision="round_trip", low_memory=False)


def _per_run() -> pd.DataFrame:
    p = _read("per_run.csv")
    return p[p["status"] == "done"].copy()


def _f(x: float, p: int) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.{p}f}"


# ------------------------------------------------------------------------------------------------ checks and launch
def block_checks() -> str:
    """Post-launch checks of ``launch_checks.json``."""
    c = json.loads((P / "launch_checks.json").read_text())
    b, s, e = c["base"], c["summary"], c["expected"]
    rows = [("status exit 0 / manifest at the launch commit, clean tree / files complete / global-RNG assertions / tail-share coverage",
             " / ".join(f"{b['n_ok_per_check'][k]}/{b['n_runs']}" for k in ("status", "manifest", "files", "global_rng", "tail_share_coverage"))),
            ("start-share tests (flagged at \\|z\\| > 3; expected by chance)",
             f"{b['start_share_tests']['n_tests']} tests, {b['start_share_tests']['n_flagged']} flagged "
             f"({b['start_share_tests']['expected_flagged_under_null']:.1f} expected)"),
            ("applied scale equals the D3 schedule at every update; 1.0 throughout stage 1", f"{s['scale_ok']}/{s['scale_n']} (expected {e['scale_n']})"),
            ("C-INIT (one `init_state_sha256` across all arms of every (q, seed))", f"{s['C_INIT_pass']}/{s['C_INIT_n']} (expected {e['C_INIT_n']})"),
            ("C-NL (s = 16 against s = 1 within each (actor, starts, q, seed) through update 2001, u02025 differs)",
             f"{s['C_NL_pass']}/{s['C_NL_n']} (expected {e['C_NL_n']})"),
            ("C-MS5 (each `t1` arm against the MS-R2 `NL_*` arm, whole run, bit for bit)", f"{s['C_MS5_pass']}/{s['C_MS5_n']} (expected {e['C_MS5_n']})"),
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
             f"{'empty' if not r['diff_stat_run_code_to_head'] else 'NOT empty'}; `git status --porcelain` lists {len(r['status_porcelain'])} entries",
             f"- parameter file SHA-256 `{r['params_sha256']}`; nproc {r['nproc']}, load average at start {', '.join(f'{x:.1f}' for x in r['loadavg_at_start'])}, "
             f"at the end {', '.join(f'{x:.1f}' for x in r['loadavg_at_end'])}, free disk {r['disk_free_bytes'] / 1e9:.0f} GB"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the criterion
def block_primary() -> str:
    """The pre-registered criterion: parts (a) and (b) per variant, starts and s against the ``t1`` arm with the same starts and s."""
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
    rows = []
    for arm in ARMS + REFS:
        for q in QS:
            g = per[(per.arm == arm) & (per.q == q)]
            if g.empty:
                continue
            sg = g["stage2_peak_rel_err_signed"].to_numpy(dtype=float)
            rng = np.random.default_rng(20261008)
            bs = sg[rng.integers(0, len(sg), size=(10000, len(sg)))].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ab = g["stage2_peak_rel_err_abs"]
            rows.append({"arm": arm, "q": q, "n": len(g), "abs(peak)<=0.05": int((ab <= 0.05).sum()), "mean abs(peak)": f"{ab.mean():.4f}",
                         "signed peak mean [95% CI]": _ci(sg.mean(), lo, hi), "signed<0": int((sg < 0).sum()),
                         "RMSE_pos": f"{g['stage2_rmse_pos_over_g2_0'].mean():.4f}", "tail mean": f"{g['stage2_tail_mean_over_g2_0'].mean():.4f}",
                         "tail max": f"{g['stage2_tail_max_over_g2_0'].mean():.4f}", "eta2/DW": f"{g['eta_T_over_dw'].mean():.5f}",
                         "G-A+G-N(eta)": f"{int(g['gate_pass'].astype(str).eq('True').sum())}/{len(g)}",
                         "R0 (median)": f"{g['t2_R0_final'].median():.4f}", "R (median)": f"{g['t2_R_final'].median():.4f}"})
    return _md(pd.DataFrame(rows))


def block_decomp() -> str:
    """The freeze decomposition per arm and q (means over the seeds, effort units) and the floor in percent of e*(0)."""
    per = _per_run()
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


def block_resolution() -> str:
    """The tie and resolution metrics at the freeze (means over the seeds): w_eff, R0/|peak|, location-free peak and argmax, symmetry."""
    per = _per_run()
    rows = []
    for arm in ARMS:
        for q in QS:
            g = per[(per.arm == arm) & (per.q == q)]
            if g.empty:
                continue
            rows.append({"arm": arm, "q": q, "w_eff (units of d)": f"{g['w_eff'].mean():.2f}", "w_eff median": f"{g['w_eff'].median():.2f}",
                         "R0/|peak| final tier (mean)": f"{g['t2_R0_over_peak_final'].mean():.3f}",
                         "linearised 2k/(2k+a)": f"{g['linearised_R0_over_peak'].iloc[0]:.3f}",
                         "location-free peak error": f"{g['peak_locfree_rel_err'].mean():+.4f}",
                         "argmax d (median, min, max)": f"{g['peak_locfree_argmax_d'].median():+.1f}, {g['peak_locfree_argmax_d'].min():+.1f}, {g['peak_locfree_argmax_d'].max():+.1f}",
                         "symmetry error/e2*(0) (mean)": f"{g['sym_err_max_rel'].mean():.4f}",
                         "max abs d-weight, last export (d/B, median)": f"{g['w1d_abs_max'].median():.3f}",
                         "bend width B/max abs w (units of d, median)": f"{g['w1d_bend_d_min'].median():.0f}"})
    return _md(pd.DataFrame(rows))


# ------------------------------------------------------------------------------------------------ paired tables
def _paired(df: pd.DataFrame, comps: Sequence, metrics: Sequence = METRICS, arm_col: str = "arm") -> str:
    """Rows per (arm, baseline, metric) with q = 50 and q = 60 columns."""
    rows = []
    for arm, base in comps:
        for m, lab, p in metrics:
            cells: Dict[str, object] = {"arm": f"`{arm}`", "baseline": f"`{base}`", "metric": lab}
            ok = False
            for q in QS:
                r = df[(df[arm_col] == arm) & (df.baseline == base) & (df.q == q) & (df.metric == m)]
                if len(r):
                    r = r.iloc[0]
                    cells[f"q={q}: mean [95% CI] (arm - baseline)"] = _ci(r["mean"], r.ci_mean_lo, r.ci_mean_hi, p)
                    cells[f"q={q}: seeds lower"] = f"{int(r.n_neg)}/{int(r.n_pairs)}"
                    ok = True
                else:
                    cells[f"q={q}: mean [95% CI] (arm - baseline)"] = "n/a"
                    cells[f"q={q}: seeds lower"] = "n/a"
            if ok:
                rows.append(cells)
    return _md(pd.DataFrame(rows))


def _comps(kind: str) -> List:
    if kind == "variant":
        return [(f"{a}_{s}_s{v}", f"t1_{s}_s{v}") for a in ACTORS[1:] for s in STARTS for v in SCALES]
    if kind == "landing":
        return [(f"{a}_{s}_s16", f"{a}_{s}_s1") for a in ACTORS for s in STARTS]
    return [(f"{a}_st_s{v}", f"{a}_bb_s{v}") for a in ACTORS for v in SCALES]


def block_secondary() -> str:
    """Paired changes v - t1 (same starts and s) of the secondary metrics."""
    return _paired(_read("paired_secondary.csv"), _comps("variant"))


def block_landing() -> str:
    """The noise-landing effect s = 16 - s = 1 within each (actor, starts)."""
    nl = _read("noise_landing.csv")
    nl = nl[nl.metric != "transmission_ratio"]
    return _paired(nl, _comps("landing"), [m for m in METRICS if m[0] in (
        "stage2_peak_rel_err_abs", "gap", "smoothing", "remainder", "sigma_2_0", "stage2_rmse_pos_over_g2_0", "t2_R0_final", "w_eff")])


def block_transmission() -> str:
    """Transmission ratio (change of the gap) / (change of the smoothing part) per (actor, starts) and q."""
    t = _read("transmission.csv")
    rows = []
    for a in ACTORS:
        for s in STARTS:
            for q in QS:
                x = t[(t.actor == a) & (t.starts == s) & (t.q == q)]
                if x.empty:
                    continue
                r = x.iloc[0]
                rows.append({"actor": a, "starts": s, "q": q, "n pairs": int(r.n_pairs), "mean change of the gap": f"{r.mean_d_gap:+.3f}",
                             "mean change of the smoothing part": f"{r.mean_d_smoothing:+.3f}",
                             "ratio [95% CI]": f"{r.ratio:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]" + ("" if r.ci_lo <= 0.0 <= r.ci_hi else " *"),
                             "per-seed ratios": r.per_seed_ratios})
    return _md(pd.DataFrame(rows))


def block_interaction() -> str:
    """(v_s16 - v_s1) - (t1_s16 - t1_s1) per variant, starts and q."""
    t = _read("interaction.csv")
    rows = []
    for a in ACTORS[1:]:
        for s in STARTS:
            for m, lab, p in (("stage2_peak_rel_err_abs", "abs(peak error)", 4), ("gap", "gap", 3), ("remainder", "remainder", 3), ("smoothing", "smoothing part", 3)):
                cells: Dict[str, object] = {"variant": a, "starts": s, "metric": lab}
                for q in QS:
                    r = t[(t.actor == a) & (t.starts == s) & (t.q == q) & (t.metric == m)]
                    if len(r):
                        r = r.iloc[0]
                        cells[f"q={q}: mean [95% CI]"] = _ci(r["mean"], r.ci_mean_lo, r.ci_mean_hi, p)
                        cells[f"q={q}: seeds < 0"] = f"{int(r.n_neg)}/{int(r.n_pairs)}"
                rows.append(cells)
    return _md(pd.DataFrame(rows))


def block_starts() -> str:
    """The starts effect (stratified - bin-balanced) within each actor and s."""
    return _paired(_read("starts_effect.csv"), _comps("starts"), [m for m in METRICS if m[0] in (
        "stage2_peak_rel_err_abs", "gap", "smoothing", "remainder", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0", "w_eff")])


def block_quadrature() -> str:
    """The quadrature check: the observed gap(s16) next to the quadrature and the additive predictions (arm means)."""
    t = _read("quadrature_check.csv")
    rows = [{"actor": r.actor, "starts": r.starts, "q": r.q, "gap(s1)": f"{r.gap_s1:.3f}", "smoothing(s1)": f"{r.smoothing_s1:.3f}",
             "smoothing(s16)": f"{r.smoothing_s16:.3f}", "F^2": f"{r.F2:+.3f}" + (" (negative: F = 0)" if r.F2_negative else ""),
             "quadrature gap(s16)": f"{r.quadrature_pred:.3f}", "additive gap(s16)": f"{r.additive_pred:.3f}", "observed gap(s16)": f"{r.gap_s16:.3f}",
             "error quad / add": f"{r.abs_err_quadrature:.3f} / {r.abs_err_additive:.3f}", "closer": r.closer} for r in t.itertuples()]
    return _md(pd.DataFrame(rows))


def block_predictions() -> str:
    """The evidence for the predictions P1-P4 (descriptive; no verdict)."""
    t = _read("predictions.csv")
    rows = []
    for r in t.itertuples():
        rows.append({"actor": r.actor, "starts": r.starts, "q": r.q,
                     "P1 abs(peak) s=1, arm / t1": f"{_f(r.p1_abs_peak_s1, 4)} / {_f(r.p1_abs_peak_t1_s1, 4)} ({int(r.p1_n_lower) if np.isfinite(r.p1_n_lower) else 'n/a'} of {int(r.p1_n_pairs) if np.isfinite(r.p1_n_pairs) else 'n/a'} seeds lower)",
                     "P2 remainder / smoothing at s=1": _f(r.p2_remainder_over_smoothing, 3),
                     "P2 share of runs with abs(remainder) < smoothing": _f(r.p2_share_abs_remainder_lt_smoothing, 2),
                     "P3 abs(peak) at s=16 / floor": f"{_f(r.p3_abs_peak_s16, 4)} / {_f(r.p3_floor_rel_smoothing, 4)} = {_f(r.p3_abs_peak_over_floor, 2)}",
                     "P3 transmission": f"{_f(r.p3_transmission_ratio, 3)} [{_f(r.p3_transmission_ci_lo, 3)}, {_f(r.p3_transmission_ci_hi, 3)}]",
                     "P4 RMSE_pos s=1, arm / t1": f"{_f(r.p4_rmse_s1, 4)} / {_f(r.p4_rmse_t1_s1, 4)}",
                     "P4 RMSE_pos s=16, arm / t1": f"{_f(r.p4_rmse_s16, 4)} / {_f(r.p4_rmse_t1_s16, 4)}"})
    return _md(pd.DataFrame(rows))


def block_vs_parents() -> str:
    """Every arm against ``parents_A`` with MS-R1's criterion."""
    c = _read("criterion_vs_parents_A.csv")
    rows = []
    for r in c.itertuples():
        b = r.b_status + (f" ({r.b_violations})" if isinstance(r.b_violations, str) and r.b_violations else "")
        rows.append({"arm": f"`{r.arm}`", "(a) q=50 mean [95% CI]": _ci(r.mean_q50, r.ci_mean_lo_q50, r.ci_mean_hi_q50, 5), "met q=50": "yes" if r.a_q50 else "no",
                     "(a) q=60 mean [95% CI]": _ci(r.mean_q60, r.ci_mean_lo_q60, r.ci_mean_hi_q60, 5), "met q=60": "yes" if r.a_q60 else "no", "(b)": b, "overall": r.overall})
    return _md(pd.DataFrame(rows))


# ------------------------------------------------------------------------------------------------ along the run
def block_segments() -> str:
    """Segment means of the PPO diagnostics and segment-end values of the tie quantities, per arm and q."""
    s = _read("segments.csv")
    rows = [{"arm": r.arm, "q": r.q, "segment": r.segment, "local": f"{int(r.first_local)}-{int(r.last_local)}", "KL": f"{r.kl_final_epoch_mean:.4f}",
             "clip frac": f"{r.clip_frac_mean:.3f}", "adv SD": f"{r.adv_raw_std_mean:.3f}", "value loss": f"{r.value_loss_mean:.3f}",
             "scale [min, max]": f"{r.conc_scale_min:.3g}, {r.conc_scale_max:.3g}", "end check": "-" if not np.isfinite(r.end_check_local) else f"{r.end_check_local:.0f}",
             "e_hat(0) at end": f"{r.e_hat_2_0_end:.2f}", "sigma(0) at end": f"{r.sigma_0_end:.3f}", "smoothing at end": f"{r.smoothing_end:.3f}",
             "remainder at end": f"{r.remainder_end:.3f}"} for r in s.itertuples() if r.arm in ARMS]
    return _md(pd.DataFrame(rows))


def block_trajectory() -> str:
    """Seed means over the checks 1800, 2000, 2200, 2400, 2600, 2800: e_hat_2(0), smoothing part, remainder, w_eff, R0."""
    t = _read("trajectory_by_arm.csv")
    rows = []
    for arm in ARMS:
        for q in QS:
            row: Dict[str, object] = {"arm": arm, "q": q}
            for u in (1800, 2000, 2200, 2400, 2600, 2800):
                g = t[(t.arm == arm) & (t.q == q) & (t["update"] == u)]
                if g.empty:
                    row[f"u{u}"] = "n/a"
                    continue
                g = g.iloc[0]
                row[f"u{u}: e_hat, smoothing, remainder, w_eff, R0"] = (
                    f"{g.e2_at_0_mean:.2f}, {g.smoothing_mean:.2f}, {g.remainder_mean:.2f}, {g.w_eff_mean:.2f}, {g.R0_mean:.4f}")
            rows.append(row)
    return _md(pd.DataFrame(rows))


def block_first_layer() -> str:
    """Median over the seeds of max abs first-layer d-weight (units of d / B) at selected exports."""
    t = _read("first_layer_summary.csv")
    rows = []
    for arm in ARMS:
        for q in QS:
            row: Dict[str, object] = {"arm": arm, "q": q}
            for u in (400, 1200, 2000, 2800):
                g = t[(t.arm == arm) & (t.q == q) & (t["update"] == u)]
                row[f"u{u}"] = "n/a" if g.empty else (f"{g.iloc[0].w_abs_max_median:.3f} "
                                                      f"[{g.iloc[0].w_abs_max_min:.3f}, {g.iloc[0].w_abs_max_max:.3f}]")
            rows.append(row)
    return _md(pd.DataFrame(rows))


# ------------------------------------------------------------------------------------------------ strata, stage 1, gates
def block_strata(stratum: str = "mid") -> str:
    """The strata at the terminal freeze (final tier): per side the mean signed error, the RMSE and max r/s."""
    s = _read("strata_summary.csv")
    s = s[(s.tier == "final") & (s.stratum == stratum)]
    rows = []
    for arm in ARMS + REFS:
        for q in QS:
            cells: Dict[str, object] = {"arm": arm, "q": q}
            for side in ("d<0", "d>0"):
                x = s[(s.arm == arm) & (s.q == q) & (s.side == side)]
                if len(x):
                    x = x.iloc[0]
                    cells[f"{side}: mean signed err"] = f"{x['err_mean_mean']:+.3f}"
                    cells[f"{side}: RMSE"] = f"{x['err_rmse_mean']:.3f}"
                    cells[f"{side}: max abs err (mean, max)"] = f"{x['err_max_abs_mean']:.3f}, {x['err_max_abs_max']:.3f}"
                    cells[f"{side}: max r/s (mean, max)"] = f"{x['max_r_over_s_mean']:.3f}, {x['max_r_over_s_max']:.3f}"
            if len(cells) > 2:
                rows.append(cells)
    return _md(pd.DataFrame(rows))


def block_strata_near() -> str:
    return block_strata("near")


def block_strata_tail() -> str:
    return block_strata("tail")


def block_stage1() -> str:
    """Stage 1 per arm and q: G-S, S1, G-F, G-N(Gmax), the stage-1 error, R_1."""
    s = _read("stage1.csv")
    g = _read("gates.csv").set_index(["arm", "q"])
    rows = []
    for x in s.itertuples():
        gg = g.loc[(x.arm, x.q)] if (x.arm, x.q) in g.index else None
        rows.append({"arm": x.arm, "q": x.q, "n": int(x.n_runs),
                     "signed error mean [95% CI]": _ci(x.stage1_rel_err_signed_mean, x.stage1_rel_err_signed_ci_lo, x.stage1_rel_err_signed_ci_hi),
                     "abs error mean": f"{x.stage1_rel_err_abs_mean:.4f}",
                     "R_1 final (median)": "n/a" if not np.isfinite(x.t1_R_final_median) else f"{x.t1_R_final_median:.4f}",
                     "G-S": "n/a" if gg is None else f"{int(gg.n_G_S)}/{int(gg.n_done)}", "S1": "n/a" if gg is None else f"{int(gg.n_S1)}/{int(gg.n_done)}",
                     "G-F": "n/a" if gg is None else f"{int(gg.n_G_F)}/{int(gg.n_done)}",
                     "G-N(Gmax)": "n/a" if gg is None else f"{int(gg.n_G_N_gmax)}/{int(gg.n_done)}"})
    return _md(pd.DataFrame(rows))


def block_stage1_paired() -> str:
    """Stage-1 paired differences: v - t1 (same starts and s) and every arm - rehearsal_v2_0 (descriptive)."""
    a = _paired(_read("stage1_vs_t1.csv"), _comps("variant"), [("stage1_rel_err_abs", "stage-1 abs error", 4), ("stage1_rel_err_signed", "stage-1 signed error", 4),
                                                                ("t1_R_final", "R_1 final", 4)])
    r = _read("paired_vs_rehearsal_v2_0.csv")
    return a + "\n\n" + _paired(r, [(arm, "rehearsal_v2_0") for arm in ARMS],
                                [("stage1_rel_err_abs", "stage-1 abs error", 4), ("stage1_rel_err_signed", "stage-1 signed error", 4), ("t1_R_final", "R_1 final", 4)])


def block_gates() -> str:
    """Gate counts per arm and q."""
    g = _read("gates.csv")
    return _md(pd.DataFrame([{"arm": r.arm, "q": r.q, "runs done": int(r.n_done), "G-A (eta)": r.n_G_A_eta, "G-A (RMSE)": r.n_G_A_rmse, "G-A (tail)": r.n_G_A_tail,
                              "G-A": r.n_G_A, "G-N(eta)": r.n_G_N_eta, "G-A and G-N(eta)": r.n_G_A_and_G_N_eta, "G-S": r.n_G_S, "S1": r.n_S1,
                              "G-F": r.n_G_F, "G-N(Gmax)": r.n_G_N_gmax, "v2.0 combination": r.n_v20_combination} for r in g.itertuples()]))


def block_shares() -> str:
    """Measured start shares per stratum (training updates) and the design shares; the learner clamp share."""
    per = _per_run()
    per = per[per.arm.isin(ARMS)]
    rows = []
    for arm in ARMS:
        for q in QS:
            g = per[(per.arm == arm) & (per.q == q)]
            rows.append({"arm": arm, "q": q, "design lambda_P": _f(g["design_lambda_P"].iloc[0], 4) if np.isfinite(g["design_lambda_P"].iloc[0]) else "-",
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


def block_side_by_side() -> str:
    """The twelve arms and the references in one table (means over the ten seeds; effort units for the decomposition)."""
    per = _per_run()
    rows = []
    for arm in ARMS + ["rehearsal_v2_0"]:
        for q in QS:
            g = per[(per.arm == arm) & (per.q == q)]
            if g.empty:
                continue
            rows.append({"arm": arm, "q": q, "mean abs(peak)": f"{g['stage2_peak_rel_err_abs'].mean():.4f}",
                         "abs(peak)<=0.05": f"{int((g['stage2_peak_rel_err_abs'] <= 0.05).sum())}/{len(g)}", "sigma_2(0)": f"{g['sigma_2_0'].mean():.3f}",
                         "smoothing part": f"{g['smoothing'].mean():.3f}", "remainder": f"{g['remainder'].mean():.3f}", "gap": f"{g['gap'].mean():.3f}",
                         "w_eff": f"{g['w_eff'].mean():.2f}", "RMSE_pos/e2*(0)": f"{g['stage2_rmse_pos_over_g2_0'].mean():.4f}",
                         "tail mean/e2*(0)": f"{g['stage2_tail_mean_over_g2_0'].mean():.4f}", "eta2/DW": f"{g['eta_T_over_dw'].mean():.5f}",
                         "R0 (median)": f"{g['t2_R0_final'].median():.4f}", "R (median)": f"{g['t2_R_final'].median():.4f}"})
    return _md(pd.DataFrame(rows))


def block_tie_effort() -> str:
    """Mean tie effort at the freeze: the closed form e*(0), the smoothed target e_sigma(0) and the learned e_hat_2(0) (effort units)."""
    per = _per_run()
    rows = []
    for arm in ARMS:
        for q in QS:
            g = per[(per.arm == arm) & (per.q == q)]
            rows.append({"arm": arm, "q": q, "e*(0)": f"{g['g2_at_0'].mean():.2f}", "e_sigma(0)": f"{g['smoothed_e_pred_0'].mean():.2f}",
                         "e_hat_2(0)": f"{g['e2_at_0'].mean():.2f}", "e_hat_2(0) seed SD": f"{g['e2_at_0'].std(ddof=1):.2f}"})
    return _md(pd.DataFrame(rows))


def block_r0() -> str:
    """R0 as the terminal-stage tie-accuracy metric: Spearman with |peak| (freeze and all checks)."""
    t = _read("r0_spearman.csv")
    return _md(pd.DataFrame([{"scope": r.scope, "actors": r.actors, "q": r.q, "n": int(r.n), "Spearman(R0, abs(peak))": f"{r.spearman_R0_vs_abs_peak:.3f}"}
                             for r in t.itertuples()]))


BLOCKS: Dict[str, Callable[[], str]] = {
    "checks": block_checks, "launch": block_launch, "primary": block_primary, "overview": block_overview, "decomp": block_decomp,
    "resolution": block_resolution, "secondary": block_secondary, "landing": block_landing, "transmission": block_transmission,
    "interaction": block_interaction, "starts": block_starts, "quadrature": block_quadrature, "predictions": block_predictions,
    "vs_parents": block_vs_parents, "segments": block_segments, "trajectory": block_trajectory, "first_layer": block_first_layer,
    "strata_mid": block_strata, "strata_near": block_strata_near, "strata_tail": block_strata_tail, "stage1": block_stage1,
    "stage1_paired": block_stage1_paired, "gates": block_gates, "shares": block_shares, "budget": block_budget,
    "side_by_side": block_side_by_side, "tie_effort": block_tie_effort, "r0": block_r0}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI."""
    global A, P
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--block")
    p.add_argument("--analysis", default=str(A))
    p.add_argument("--pilot", default=str(P))
    p.add_argument("--list", action="store_true")
    p.add_argument("--arm", default=None, help="keep only the table rows whose first cell is this arm (or actor)")
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
