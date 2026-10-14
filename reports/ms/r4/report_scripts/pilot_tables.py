#!/usr/bin/env python3
"""Tables of the MS-R4 pilot reports (``04_e1_actor.md``, ``04_e2_feedback.md``, ``05_decision_inputs.md``).

Every table is printed from the CSV / JSON files of ``results/ms_r4/`` (the launch record, the launch checks, the offline
curves and the E1 / E2 analysis tables); nothing is computed here that the analysis tools did not write, except counts,
medians and per-run reshapes of the per-run tables, and the post hoc supplements that are labelled as such in the reports.

Usage:
  python reports/ms/r4/report_scripts/pilot_tables.py --block <name> [key=value ...]
  python reports/ms/r4/report_scripts/pilot_tables.py --fill draft.md --out final.md   # replaces {{name key=value}}
  python reports/ms/r4/report_scripts/pilot_tables.py --list
The repository root is taken from the location of this file; ``MS_R4_ROOT`` overrides it.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

ROOT = Path(os.environ.get("MS_R4_ROOT") or Path(__file__).resolve().parents[4])
R = ROOT / "results" / "ms_r4"
A = R / "analysis"
E1 = A / "e1"
E2 = A / "e2"

E1_ARMS = ["t1_st_s16", "relu_st_s16", "lrsm_st_s16"]
E2_ARMS = ["fb_a0", "fb_r", "fb_rg", "fb_rx"]
QS = ["50", "60"]
GATES = [("G_A", "G-A"), ("G_F", "G-F"), ("G_N", "G-N"), ("G_S", "G-S")]


# --------------------------------------------------------------------------- helpers
def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dicts."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(x: Any) -> float:
    """Float of a CSV cell; NaN for an empty or non-numeric cell."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def is_true(x: Any) -> bool:
    """Boolean of a CSV cell written by pandas / csv."""
    return str(x).strip().lower() in ("true", "1", "1.0")


def _nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fx(x: float, nd: int = 4) -> str:
    """Fixed-point format; ``nan`` for NaN."""
    return "nan" if _nan(x) else f"{x:.{nd}f}"


def sx(x: float, nd: int = 4) -> str:
    """Signed fixed-point format."""
    return "nan" if _nan(x) else f"{x:+.{nd}f}"


def gx(x: float, nd: int = 3) -> str:
    """General format with ``nd`` significant digits (scientific for small values)."""
    return "nan" if _nan(x) else f"{x:.{nd}g}"


def sg(x: float, nd: int = 3) -> str:
    """Signed general format."""
    return "nan" if _nan(x) else f"{x:+.{nd}g}"


def ci(lo: float, hi: float, nd: int = 4, signed: bool = True) -> str:
    """Interval text ``[lo, hi]`` in fixed point."""
    f = sx if signed else fx
    return f"[{f(lo, nd)}, {f(hi, nd)}]"


def cig(lo: float, hi: float, nd: int = 3) -> str:
    """Interval text with general (significant-digit) format."""
    return f"[{sg(lo, nd)}, {sg(hi, nd)}]"


def md(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """GitHub-flavoured markdown table."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def rel(p: Path) -> str:
    """Repository-relative path text."""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def per_run(exp: str) -> List[Dict[str, str]]:
    """The per-run table of an experiment (``e1`` or ``e2``)."""
    return read_csv((E1 if exp == "e1" else E2) / "per_run.csv")


def failed_gates(r: Dict[str, str]) -> List[str]:
    """Names of the gates a run fails (empty when all four pass); ``exit`` for a nonzero exit."""
    out = [name for key, name in GATES if not is_true(r["gate_" + key])]
    if is_true(r.get("nonzero_exit", "False")):
        out.append("exit")
    return out


def arm_label(arm: str) -> str:
    """Short arm name used in the tables."""
    return arm.replace("_st_s16", "")


# --------------------------------------------------------------------------- launch and checks
def latest_launch() -> Path:
    """The launch record of wave r4."""
    return sorted((R / "pilot").glob("launch_2*.json"))[-1]


def b_launch() -> str:
    """Key facts of the launch record."""
    f = latest_launch()
    d = json.load(open(f))
    runs = d["runs"]
    nz = sum(1 for r in runs if r.get("returncode") != 0)
    porc = d["status_porcelain"]
    outside = [p for p in porc if not p[3:].startswith("results/ms_r4/")]
    la0, la1 = d["loadavg_at_start"], d.get("loadavg_at_end") or []
    rows = [
        ("launch record", f"`{rel(f)}`"),
        ("wave, workers, state", f"`{d['wave']}`, {d['workers']} workers (single-threaded), `{d['state']}`"),
        ("started (stamp)", d["started"]),
        ("`nproc`", d["nproc"]),
        ("load average at start (1, 5, 15 min)", ", ".join(f"{x:.2f}" for x in la0)),
        ("load average at end (1, 5, 15 min)", ", ".join(f"{x:.2f}" for x in la1)),
        ("free disk at start", f"{d['disk_free_bytes'] / 1e9:.0f} GB of {d['disk_total_bytes'] / 1e9:.0f} GB"),
        ("HEAD at the launch", f"`{d['head']}`"),
        ("code commit (`--code-commit`)", f"`{d['code_commit_resolved']}`"),
        ("`git diff --stat <code commit> HEAD`",
         "empty" if not str(d["diff_stat_code_commit_to_head"]).strip() else str(d["diff_stat_code_commit_to_head"])),
        ("`git diff --stat` on the run code paths",
         "empty" if not str(d["diff_stat_run_code_to_head"]).strip() else str(d["diff_stat_run_code_to_head"])),
        ("`git status --porcelain`", f"{len(porc)} entries, {len(outside)} of them outside `results/ms_r4/` "
         "(the entries are the untracked C-MS6 and C-R7 results of the same worktree)"),
        ("parameter file SHA-256", f"`{d['params_sha256']}`"),
        ("runs planned / finished / nonzero exit", f"{d['n_planned']} / {len(runs)} / {nz}"),
    ]
    return md(["item", "value"], rows)


def b_launch_wall() -> str:
    """Wall time of every arm from the launch record."""
    d = json.load(open(latest_launch()))
    g: Dict[Any, List[float]] = collections.defaultdict(list)
    for r in d["runs"]:
        parts = r["out"].rstrip("/").split("/")
        g[(parts[-1], parts[-3][1:])].append(r["wall_sec"])
    rows = []
    for arm in E1_ARMS + E2_ARMS:
        for q in QS:
            v = g[(arm, q)]
            rows.append((f"`{arm}`", q, len(v), f"{statistics.median(v):.0f}", f"{min(v):.0f}", f"{max(v):.0f}"))
    return md(["arm", "q", "runs", "median wall (s)", "min (s)", "max (s)"], rows)


def b_checks() -> str:
    """The post-launch checks of section 3.2 and the offline checks of section 3.3."""
    lc = json.load(open(R / "pilot" / "launch_checks.json"))
    n_ok = lc["base"]["n_ok_per_check"]
    s = lc["summary"]
    ss = lc["base"]["start_share_tests"]
    off = read_csv(R / "curves" / "c_off.csv")
    info = json.load(open(R / "curves" / "curves_info.json"))
    e1b = json.load(open(A / "e1_blind.stdout"))
    e2b = json.load(open(A / "e2_blind.stdout"))
    n = lc["base"]["n_runs"]
    rows = [
        ("runs done with exit code 0 (`status.json`)", f"{n_ok['status']} / {n}", "pass"),
        ("manifest at the launch commit, clean tree", f"{n_ok['manifest']} / {n}", "pass"),
        ("files complete", f"{n_ok['files']} / {n}", "pass"),
        ("global-RNG assertions", f"{n_ok['global_rng']} / {n}", "pass"),
        ("tail share equals lambda_T at every update (coverage)", f"{n_ok['tail_share_coverage']} / {n}", "pass"),
        ("tail mass equals lambda_T to 1e-12 at every update (deterministic)",
         f"{s['strata_masses_ok']} / {s['strata_masses_n']}", "pass"),
        ("near-tie and middle masses equal the design to 1e-12 in the within-stratum arms `fb_a0`, `fb_r`, `fb_rg`",
         f"{sum(1 for x in lc['strata_masses'] if x['arm'] in ('fb_a0', 'fb_r', 'fb_rg'))} / "
         f"{sum(1 for x in lc['strata_masses'] if x['arm'] in ('fb_a0', 'fb_r', 'fb_rg'))}"
         if lc["within_arms_masses_exact"] is True else "FAILED", "pass"),
        ("applied concentration scale equals the D2 schedule", f"{s['scale_ok']} / {s['scale_n']}", "pass"),
        ("C-INIT: initial weights identical across `t1`, `relu`, `lrsm`", f"{s['C_INIT_pass']} / {s['C_INIT_n']}",
         "pass"),
        ("C-FB25: the four E2 arms bit-identical through local update 25", f"{s['C_FB25_pass']} / {s['C_FB25_n']}",
         "pass"),
        ("C-X2: `fb_rx` equals `t1_st_s16` through local update 1201", f"{s['C_X2_pass']} / {s['C_X2_n']}", "pass"),
        ("sampling weights consistent with `p_digest_next`",
         f"{s['sampling_weights_ok']} / {s['sampling_weights_n']}", "pass"),
        ("C-OFF: development tier offline equals the online checks, runs",
         f"{sum(is_true(r['ok']) for r in off)} / {len(off)}", "pass"),
        ("C-OFF: comparisons / mismatches / max abs difference",
         f"{sum(int(r['n_compared']) for r in off):,} / {sum(int(r['n_mismatch']) for r in off)} / "
         f"{max(num(r['max_abs_diff']) for r in off)}", "pass"),
        ("offline curve points written", f"{info['n_points']:,} ({info['n_ok']} / {info['n_runs']} runs)", "done"),
        ("blind recomputation of `criterion_e1.csv` (1e-12)", f"{e1b['n_differences']} differences", "agree"),
        ("blind recomputation of `criterion_e2.csv` and the decision (1e-12)", f"{e2b['n_differences']} differences",
         "agree"),
        ("statistical start-share flags (|z| > 3), reported, not a failure",
         f"{ss['n_flagged']} of {ss['n_tests']} tests (chance expectation {ss['expected_flagged_under_null']:.2f})",
         "reported"),
    ]
    return md(["check", "result", "status"], rows)


def b_start_flags() -> str:
    """The flagged start-share tests."""
    lc = json.load(open(R / "pilot" / "launch_checks.json"))
    rows = []
    for r in lc["base"]["runs"]:
        for fl in (r.get("start_shares") or {}).get("flagged", []):
            rows.append((f"`q{r['q']}/seed{r['seed']}/{r['arm']}`", fl["stage"], fl["stratum"], fl["n_updates"],
                         f"{fl['observed']:.0f}", f"{fl['expected']:.1f}", f"{fl['z']:+.2f}"))
    return md(["run", "stage", "stratum", "updates", "observed starts", "expected", "z"], rows)


def b_config_arms() -> str:
    """Per arm: the configuration keys that define it, read from ``run_config.json`` of q = 50, seed 10511."""
    roles = {"t1_st_s16": "E1 control", "relu_st_s16": "E1 control", "lrsm_st_s16": "E1 candidate",
             "fb_a0": "E2 arm (1)", "fb_r": "E2 arm (2)", "fb_rg": "E2 arm (3)", "fb_rx": "E2 descriptive"}
    rows = []
    for arm in E1_ARMS + E2_ARMS:
        p = R / "pilot" / "q50" / "seed10511" / arm
        c = json.load(open(p / "run_config.json"))
        m = json.load(open(p / "manifest.json"))
        sw = c["start_weights"]
        cs = c.get("conc_scale_schedule")
        landing = (f"s {cs['scale_first']:.0f} -> {cs['scale_last']:.0f} over local {cs['local_first']}-"
                   f"{cs['local_last']}" if cs else "none (s = 1)")
        focus = c.get("focus_scope") or ("cross-stratum (existing)" if sw["alpha_global"] > 0 else "none (alpha 0)")
        if c.get("focus_map"):
            fmap = c["focus_map"]
        else:
            fmap = "residual" if (c.get("focus_scope") or sw["alpha_global"] > 0) else "-"
        rows.append((f"`{arm}`", roles[arm], c.get("actor_variant", "t1"), sw["alpha_global"], focus, fmap,
                     c["pipeline"]["budgets"]["2"], landing, c["pipeline"]["budgets"]["1"], f"`{m['commit'][:8]}`"))
    return md(["arm", "role", "actor", "alpha", "focus scope", "focus map", "stage-2 updates", "landing",
               "stage-1 updates", "code commit"], rows)


def b_config_diffs() -> str:
    """Keys in which an arm's config differs from its comparator (q = 50; q = 60 is the same)."""
    rows = []
    for r in read_csv(ROOT / "reports" / "ms" / "r4" / "arm_config_differences.csv"):
        if r["q"] != "50" or "MS-R" in r["arm"]:
            continue
        rows.append((f"`{r['arm']}`", f"`{r['comparator']}`", r["purpose"], r["keys_that_differ"].replace(";", ", ")))
    return md(["arm", "against", "purpose", "config keys that differ (besides arm / run / output labels)"], rows)


# --------------------------------------------------------------------------- E1
def b_e1_per_seed(q: str = "50") -> str:
    """|peak error| of every run by arm; a failing run shows its failed gates."""
    rows_by: Dict[Any, Dict[str, str]] = {(r["arm"], r["q"], r["seed"]): r for r in per_run("e1")}
    seeds = sorted({r["seed"] for r in per_run("e1")})
    rows = []
    for s in seeds:
        cells = []
        for arm in E1_ARMS:
            r = rows_by[(arm, q, s)]
            fg = failed_gates(r)
            cells.append(fx(num(r["peak_abs"])) + (f" **fails {', '.join(fg)}**" if fg else ""))
        rows.append((s, *cells))
    means = ["mean"] + [fx(statistics.mean(num(rows_by[(a, q, s)]["peak_abs"]) for s in seeds)) for a in E1_ARMS]
    rows.append(tuple(f"**{c}**" for c in means))
    return md(["seed", *[f"`{a}`" for a in E1_ARMS]], rows)


def b_e1_criterion() -> str:
    """The three parts of the E1 target at both q."""
    rows = []
    allrows = read_csv(E1 / "criterion_e1.csv")
    for r in allrows:
        if r["q"] == "both":
            continue
        fails = [f"G-A {r['n_G_A_fail']}", f"G-F {r['n_G_F_fail']}", f"G-N {r['n_G_N_fail']}",
                 f"G-S {r['n_G_S_fail']}"]
        rows.append((r["q"], f"{fx(num(r['mean_abs_peak_lrsm']))} (<= {r['a_threshold']})", r["a_pass"],
                     f"{sx(num(r['mean_diff_vs_t1']))} {ci(num(r['ci_lo']), num(r['ci_hi']))}", r["b_pass"],
                     f"{r['n_gate_fail']} of {r['n_runs_done']} ({', '.join(fails)}; nonzero exits "
                     f"{r['n_nonzero_exit']})", r["c_pass"], r["met_q"]))
    last = [r for r in allrows if r["q"] == "both"][0]
    t = md(["q", "(E1-a) mean abs peak error, `lrsm_st_s16`", "a", "(E1-b) paired difference vs `t1_st_s16`, 95 % CI",
            "b", "(E1-c) runs failing a gate", "c", "met at q"], rows)
    return t + f"\n\nVerdict (`criterion_e1.csv`, row `both`): **{last['status']}**"


def b_e1_paired_peak() -> str:
    """Paired statistics of |peak error|: the criterion comparison and the two descriptive ones."""
    rows = []
    for r in read_csv(E1 / "paired_e1.csv"):
        if r["metric"] != "peak_abs":
            continue
        rows.append((r["comparison"].replace("_st_s16", ""), r["q"], fx(num(r["mean_a"])), fx(num(r["mean_b"])),
                     sx(num(r["mean_diff"])), ci(num(r["ci_lo"]), num(r["ci_hi"])), sx(num(r["median_diff"])),
                     f"{r['n_neg']} / {r['n_pos']}"))
    return md(["comparison (a - b)", "q", "mean a", "mean b", "mean diff", "95 % CI", "median diff",
               "pairs a < b / a > b"], rows)


def b_e1_failure_rates() -> str:
    """Runs failing a gate or exiting nonzero, per arm and q, exact Clopper-Pearson 95 % intervals."""
    rows = []
    for r in read_csv(E1 / "failure_rates.csv"):
        rows.append((f"`{r['arm']}`", r["q"], r["n_runs"], r["n_fail"], fx(num(r["rate"]), 3),
                     ci(num(r["cp95_lo"]), num(r["cp95_hi"]), 3, signed=False)))
    return md(["arm", "q", "runs", "failing", "rate", "Clopper-Pearson 95 %"], rows)


def b_e1_failed_runs() -> str:
    """Every E1 run that fails a gate, with its end values."""
    pr = {(r["arm"], r["q"], r["seed"]): r for r in per_run("e1")}
    rows = []
    for r in read_csv(E1 / "failures.csv"):
        x = pr[(r["arm"], r["q"], r["seed"])]
        rows.append((f"`{r['arm']}`", r["q"], r["seed"], ", ".join(failed_gates(x)), fx(num(r["peak_abs"])),
                     f"{num(r['e2_at_0']):.2f} / {num(r['g2_at_0']):.2f}", gx(num(r["eta_T_over_dw"])),
                     gx(num(r["stage2_rmse_pos_over_g2_0"])),
                     "yes" if is_true(r["tie_at_lower_bound"]) or is_true(r["tie_at_upper_bound"]) else "no"))
    return md(["arm", "q", "seed", "gates failed", "abs peak error", "e_hat_2(0) / e2*(0)", "eta_2/DW",
               "RMSE_pos/e2*(0)", "tie mean at the bound of the mean map"], rows)


E1_DECOMP = [("peak_signed", "signed peak error (relative)"), ("gap", "peak gap (effort units)"), ("smoothing", "smoothing part"), ("remainder", "remainder"),
             ("sigma_2_0", "sigma_2(0)"), ("stage2_rmse_pos_over_g2_0", "RMSE_pos / e2*(0)"),
             ("stage2_tail_mean_over_g2_0", "tail mean / e2*(0)"), ("eta_T_over_dw", "eta_2 / DW"),
             ("t2_R0_final", "R0"), ("w_eff", "w_eff"), ("stage1_rel_err_abs", "stage-1 abs relative error"),
             ("Gmax_full_over_dw", "Gmax_full / DW")]


def b_e1_decomp(a: str = "lrsm", b: str = "t1") -> str:
    """Paired changes of the decomposition and the secondary quantities (descriptive), arm ``a`` - arm ``b``."""
    comparison = f"{a}_st_s16 - {b}_st_s16"
    rows = []
    allrows = read_csv(E1 / "paired_e1.csv")
    for q in QS:
        for key, label in E1_DECOMP:
            for r in allrows:
                if r["comparison"] == comparison and r["q"] == q and r["metric"] == key:
                    rows.append((q, label, gx(num(r["mean_a"])), gx(num(r["mean_b"])), sg(num(r["mean_diff"])),
                                 cig(num(r["ci_lo"]), num(r["ci_hi"])), f"{r['n_neg']} / {r['n_pos']}"))
    return md(["q", "quantity", f"mean `{a}`", f"mean `{b}`", "mean diff", "95 % CI", "pairs a < b / a > b"], rows)


def b_e1_overview() -> str:
    """Per arm and q: the end-of-run numbers of the frozen terminal stage (means and medians over the 20 runs)."""
    pr = per_run("e1")
    rows = []
    for q in QS:
        for arm in E1_ARMS:
            rs = [r for r in pr if r["arm"] == arm and r["q"] == q]

            def col(k: str) -> List[float]:
                return [num(r[k]) for r in rs]
            fails = sum(1 for r in rs if failed_gates(r))
            rows.append((q, f"`{arm}`", len(rs), fx(statistics.mean(col("peak_abs"))),
                         fx(statistics.median(col("peak_abs"))), fx(statistics.mean(col("gap")), 2),
                         fx(statistics.median(col("gap")), 2), gx(statistics.mean(col("stage2_rmse_pos_over_g2_0"))),
                         gx(statistics.median(col("eta_T_over_dw"))), fails))
    return md(["q", "arm", "runs", "mean abs peak error", "median", "mean gap (effort units)", "median gap",
               "mean RMSE_pos / e2*(0)", "median eta_2/DW", "runs failing a gate"], rows)


def b_e1_pass_fail_split() -> str:
    """Post hoc supplement: mean abs peak error of the runs that pass all four gates and of those that fail one."""
    pr = per_run("e1")
    rows = []
    for q in QS:
        for arm in E1_ARMS:
            rs = [r for r in pr if r["arm"] == arm and r["q"] == q]
            ok = [num(r["peak_abs"]) for r in rs if not failed_gates(r)]
            bad = [num(r["peak_abs"]) for r in rs if failed_gates(r)]
            rows.append((q, f"`{arm}`", len(ok), fx(statistics.mean(ok)), len(bad),
                         fx(statistics.mean(bad)) if bad else "-", fx(max(bad)) if bad else "-"))
    return md(["q", "arm", "runs passing all four gates", "their mean abs peak error", "runs failing a gate",
               "their mean abs peak error", "max"], rows)


def b_e1_locfree() -> str:
    """Location-free peak error, its argmax and the symmetry error at the freeze (final tier), per arm and q."""
    pr = per_run("e1")
    rows = []
    for q in QS:
        for arm in E1_ARMS:
            rs = [r for r in pr if r["arm"] == arm and r["q"] == q]

            def col(k: str) -> List[float]:
                return [num(r[k]) for r in rs]
            rows.append((q, f"`{arm}`", sx(statistics.mean(col("peak_locfree_rel_err"))),
                         sx(statistics.median(col("peak_locfree_rel_err"))),
                         f"{statistics.median(col('peak_locfree_argmax_d')):+.2f}",
                         fx(statistics.mean(col("sym_err_max_rel"))), fx(statistics.median(col("sym_err_max_rel")))))
    return md(["q", "arm", "mean location-free peak error (signed, relative)", "median",
               "median argmax d of the location-free peak", "mean symmetry error (relative)", "median"], rows)


def b_e1_strata() -> str:
    """Per-stratum errors of the learned effort at the freeze (final tier), relative to e2*(d), mean over runs."""
    rows = []
    for r in read_csv(E1 / "strata_summary.csv"):
        if r["tier"] != "final":
            continue
        rows.append((r["q"], f"`{r['arm']}`", r["stratum"], r["side"], sx(num(r["err_mean_rel_mean"]), 5),
                     fx(num(r["err_rmse_rel_mean"]), 5), fx(num(r["err_max_abs_rel_mean"]), 5)))
    rows.sort(key=lambda x: (x[0], ["near", "mid", "tail"].index(x[2]), x[3], E1_ARMS.index(x[1].strip("`"))))
    return md(["q", "arm", "stratum", "side", "mean signed error / e2*", "RMSE / e2*", "mean max abs error / e2*"],
              rows)


def b_e1_failure_traj() -> str:
    """Trajectories of the failed runs: e_hat_2(0), RMSE_pos / e2*(0), R0 and the symmetry error at selected updates."""
    by: Dict[Any, Dict[int, Dict[str, str]]] = collections.defaultdict(dict)
    for r in read_csv(E1 / "failure_trajectories.csv"):
        by[(r["arm"], r["q"], r["seed"])][int(r["local"])] = r
    ups = [100, 400, 1200, 1600, 2000, 2200, 2800]
    rows = []
    for (arm, q, seed), v in sorted(by.items()):
        rows.append((f"`{arm}`", q, seed, "e_hat_2(0)", *[fx(num(v[u]["e2_at_0"]), 1) for u in ups]))
        rows.append(("", "", "", "RMSE_pos / e2*(0)", *[fx(num(v[u]["stage2_rmse_pos_over_g2_0"]), 3) for u in ups]))
        rows.append(("", "", "", "R0", *[fx(num(v[u]["R0"]), 3) for u in ups]))
        rows.append(("", "", "", "symmetry error (max, effort units)", *[fx(num(v[u]["sym_err_max"]), 1) for u in ups]))
    return md(["arm", "q", "seed", "quantity", *[f"u{u}" for u in ups]], rows)


def b_e1_units() -> str:
    """Hidden units of the failed relu / lrsm runs by the share of development nodes with a positive pre-activation."""
    g: Dict[Any, Dict[str, List[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in read_csv(E1 / "failure_units.csv"):
        g[(r["arm"], r["q"], r["seed"])][r["layer"]].append(num(r["share_positive"]))
    rows = []
    for (arm, q, seed), layers in sorted(g.items()):
        cells = []
        for ly in ("layer1", "layer2"):
            v = layers[ly]
            cells.append(f"{sum(1 for x in v if x == 0)} / {sum(1 for x in v if x == 1)} / "
                         f"{sum(1 for x in v if 0 < x < 1)}")
        rows.append((f"`{arm}`", q, seed, *cells))
    return md(["arm", "q", "seed", "layer 1: never / always / mixed positive", "layer 2: never / always / mixed"],
              rows)


def _curve_table(path: Path, arms: Sequence[str], updates: Sequence[int], col: str,
                 fmt: Callable[[float], str]) -> str:
    idx = {(r["arm"], r["q"], int(r["update"])): r for r in read_csv(path)}
    rows = []
    for q in QS:
        for arm in arms:
            rows.append((q, f"`{arm}`", *[fmt(num(idx[(arm, q, u)][col])) for u in updates]))
    return md(["q", "arm", *[f"u{u}" for u in updates]], rows)


def b_e1_curves(metric: str = "eta") -> str:
    """Seed medians of the final-tier eta_2/DW, |peak error| or e_hat_2(0) at selected stage-2 updates."""
    ups = [25, 50, 100, 200, 400, 800, 1600, 2000, 2400, 2800]
    col = {"eta": "eta_median", "peak": "peak_abs_median", "e2": "e2_at_0_median"}[metric]
    return _curve_table(E1 / "learning_curves_e1.csv", E1_ARMS, ups, col,
                        (lambda x: fx(x, 2)) if metric == "e2" else (lambda x: gx(x, 3)))


def b_cost(exp: str = "e1") -> str:
    """Cost per arm and q: median wall times and the development verifier's share of the stage-2 wall time."""
    f = E1 / "cost_e1.csv" if exp == "e1" else E2 / "cost_e2.csv"
    idx = {(r["arm"], r["q"], r["quantity"]): r for r in read_csv(f)}
    arms = E1_ARMS if exp == "e1" else E2_ARMS
    rows = []
    for q in QS:
        for arm in arms:
            def get(k: str) -> float:
                return num(idx[(arm, q, k)]["median"])
            rows.append((q, f"`{arm}`", f"{get('wall_total_sec'):.0f}", f"{get('wall_stage1_sec'):.0f}",
                         f"{get('wall_stage2_sec'):.0f}", f"{get('stage2_train_rollout_sec'):.1f}",
                         f"{get('stage2_train_update_sec'):.0f}", f"{get('stage2_dev_verifier_sec'):.2f}", f"{get('stage1_dev_verifier_sec'):.2f}",
                         f"{100 * get('dev_verifier_share_stage2'):.2f} %",
                         f"{100 * num(idx[(arm, q, 'dev_verifier_share_stage2')]['max']):.2f} %"))
    return md(["q", "arm", "total wall (s)", "stage 1 (s)", "stage 2 (s)", "stage-2 rollouts (s)",
               "stage-2 updates (s)", "stage-2 dev. verifier (s)", "stage-1 dev. verifier (s)",
               "verifier share of stage 2 (median)", "(max over runs)"], rows)


def b_e1_predictions() -> str:
    """Predictions E1-P1 and E1-P2 against the outcome."""
    rows = []
    for r in read_csv(E1 / "predictions_e1.csv"):
        if r["prediction"] == "E1-P1":
            rows.append(("E1-P1", r["q"], "`lrsm_st_s16` has a lower mean abs peak error than `t1_st_s16`",
                         f"prior `relu` - `t1` on 10501-10510: {r['prior_relu_minus_t1']}",
                         f"`lrsm` - `t1` {sx(num(r['observed_lrsm_minus_t1']))} {r['observed_lrsm_ci']}; "
                         f"`relu` - `t1` {sx(num(r['observed_relu_minus_t1']))}", r["outcome"]))
        elif r["q"] == "50":
            rows.append((r["prediction"], r["q"], "`relu` fails at least one gate at q = 50 and `lrsm` fails fewer",
                         "10501-10510: 5 of 40 `relu` runs at q = 50 failed G-A (two cases)",
                         f"`relu` {int(num(r['observed_relu_failures']))}, `lrsm` "
                         f"{int(num(r['observed_lrsm_failures']))} failing runs of 20", r["outcome"]))
    return md(["prediction", "q", "statement", "prior", "observed", "outcome"], rows)


# --------------------------------------------------------------------------- E2
def b_e2_area_seed(q: str = "50") -> str:
    """Area A (x 1e-3) of every E2 run; a failing run shows its failed gates."""
    pr = {(r["arm"], r["q"], r["seed"]): r for r in per_run("e2")}
    seeds = sorted({r["seed"] for r in per_run("e2")})
    rows = []
    for s in seeds:
        cells = []
        for arm in E2_ARMS:
            r = pr[(arm, q, s)]
            fg = failed_gates(r)
            cells.append(fx(1e3 * num(r["A"]), 3) + (f" **fails {', '.join(fg)}**" if fg else ""))
        rows.append((s, *cells))
    means = ["mean"] + [fx(1e3 * statistics.mean(num(pr[(a, q, s)]["A"]) for s in seeds), 3) for a in E2_ARMS]
    rows.append(tuple(f"**{c}**" for c in means))
    return md(["seed", *[f"`{a}`" for a in E2_ARMS]], rows)


def b_e2_overview() -> str:
    """Per arm and q: the end points of the E2 runs (means and medians over the 20 runs)."""
    pr = per_run("e2")
    rows = []
    for q in QS:
        for arm in E2_ARMS:
            rs = [r for r in pr if r["arm"] == arm and r["q"] == q]

            def col(k: str) -> List[float]:
                return [num(r[k]) for r in rs]
            rows.append((q, f"`{arm}`", fx(1e3 * statistics.mean(col("A")), 3), fx(1e3 * statistics.median(col("A")), 3),
                         gx(statistics.median(col("eta_u1600"))), fx(statistics.mean(col("peak_abs"))),
                         fx(statistics.median(col("gap")), 2), gx(statistics.mean(col("rmse_pos"))),
                         gx(statistics.mean(col("tail_mean"))), gx(statistics.mean(col("s1_abs"))),
                         sum(1 for r in rs if is_true(r["gates_all_pass"]))))
    return md(["q", "arm", "mean A x 1e3", "median A x 1e3", "median eta_2/DW at u1600", "mean abs peak error",
               "median gap", "mean RMSE_pos / e2*(0)", "mean tail mean / e2*(0)", "mean stage-1 abs rel. error",
               "runs passing all four gates"], rows)


def b_e2_early_share() -> str:
    """Share of the area A that lies in local updates 25-200 (A and A_late of the per-run table)."""
    pr = per_run("e2")
    rows = []
    for q in QS:
        for arm in E2_ARMS:
            sh = [1.0 - 1400.0 * num(r["A_late"]) / (1575.0 * num(r["A"])) for r in pr if r["arm"] == arm
                  and r["q"] == q]
            rows.append((q, f"`{arm}`", fx(statistics.mean(sh), 3), fx(min(sh), 3), fx(max(sh), 3)))
    return md(["q", "arm", "mean share of the area in updates 25-200", "min over runs", "max"], rows)


def b_a2_rule_i() -> str:
    """Post hoc illustration of rule (i) of the pre-registered confirmation rule A2 (at least 18 of 20 runs pass)."""
    from scipy.stats import beta  # type: ignore

    pr = per_run("e1")
    rows = []
    for arm in E1_ARMS:
        for q in QS:
            rs = [r for r in pr if r["arm"] == arm and r["q"] == q]
            n, k = len(rs), sum(1 for r in rs if not failed_gates(r))
            lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
            hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
            rows.append((f"`{arm}`", q, f"{k} / {n}", ci(lo, hi, 3, signed=False)))
    t1 = md(["arm", "q", "runs passing all four gates (development seeds)", "Clopper-Pearson 95 %"], rows)
    rows2 = []
    for p in (0.80, 0.85, 0.90, 0.95, 0.99):
        pr18 = sum(math.comb(20, j) * p ** j * (1 - p) ** (20 - j) for j in range(18, 21))
        rows2.append((f"{p:.2f}", f"{pr18:.3f}"))
    t2 = md(["assumed true per-run pass rate at one q", "probability of at least 18 of 20 passing runs at that q"], rows2)
    return t1 + "\n\n" + t2


def b_a2_verbatim() -> str:
    """The confirmation rule A2 of the PI's reply at gate G1, copied from ``pi_record/24_g1_reply.md``."""
    text = (ROOT / "reports" / "ms" / "r4" / "pi_record" / "24_g1_reply.md").read_text()
    a2 = text[text.index("A2. The confirmation rule"):text.index("Then\nContinue")].rstrip()
    return "\n".join(("> " + ln) if ln else ">" for ln in a2.splitlines())


def b_tiers(exp: str = "e1") -> str:
    """Final and development tier at the freeze: eta_2/DW, Gmax_full/DW and their G-N differences (means over runs)."""
    pr = per_run(exp)
    arms = E1_ARMS if exp == "e1" else E2_ARMS
    rows = []
    for q in QS:
        for arm in arms:
            rs = [r for r in pr if r["arm"] == arm and r["q"] == q]

            def col(k: str) -> List[float]:
                return [num(r[k]) for r in rs]
            rows.append((q, f"`{arm}`", gx(statistics.mean(col("eta_T_over_dw"))), gx(statistics.mean(col("eta_dev"))),
                         gx(statistics.mean(col("eta_dev_minus_final_abs"))), gx(max(col("eta_dev_minus_final_abs"))),
                         gx(statistics.mean(col("gmax_dev_minus_final_abs"))), gx(max(col("gmax_dev_minus_final_abs"))),
                         sum(1 for r in rs if not is_true(r["gate_G_N"]))))
    return md(["q", "arm", "mean eta_2/DW, final tier", "development tier", "mean |dev - final| of eta_2/DW", "max",
               "mean |dev - final| of Gmax_full/DW", "max", "runs failing G-N"], rows)


def b_e1_shares() -> str:
    """Realised start shares over all updates of the E1 arms (prompt 3.2) against the design."""
    pr = per_run("e1")
    rows = []
    for q in QS:
        for arm in E1_ARMS:
            rs = [r for r in pr if r["arm"] == arm and r["q"] == q]
            rows.append((q, f"`{arm}`", f"{num(rs[0]['design_lambda_P']):.3f} / {num(rs[0]['design_lambda_M']):.3f}",
                         f"{statistics.mean(num(r['t2_share_near_all']) for r in rs):.4f} / "
                         f"{statistics.mean(num(r['t2_share_mid_all']) for r in rs):.4f}"))
    return md(["q", "arm", "design near-tie / middle", "realised near-tie / middle (mean of 20 runs, all updates)"], rows)


def b_e2_peak_seed(q: str = "50") -> str:
    """Abs peak error of every E2 run at the stage-2 freeze (final tier)."""
    pr = {(r["arm"], r["q"], r["seed"]): r for r in per_run("e2")}
    seeds = sorted({r["seed"] for r in per_run("e2")})
    rows = [(s, *[fx(num(pr[(a, q, s)]["peak_abs"])) for a in E2_ARMS]) for s in seeds]
    rows.append(tuple(f"**{c}**" for c in ["mean", *[fx(statistics.mean(num(pr[(a, q, s)]["peak_abs"]) for s in seeds))
                                                    for a in E2_ARMS]]))
    return md(["seed", *[f"`{a}`" for a in E2_ARMS]], rows)


def b_e1_rmse_rise() -> str:
    """Post hoc: RMSE_pos / e2*(0) at update 2000 against update 1200, passing and failing E1 runs (curves_all.csv)."""
    cur: Dict[Any, Dict[int, float]] = collections.defaultdict(dict)
    for r in read_csv(R / "curves" / "curves_all.csv"):
        if r["arm"] in E1_ARMS and int(r["update"]) in (1200, 2000):
            cur[(r["arm"], r["q"], r["seed"])][int(r["update"])] = num(r["final_stage2_rmse_pos_over_g2_0"])
    pr = {(r["arm"], r["q"], r["seed"]): r for r in per_run("e1")}
    groups = {False: [], True: []}
    for k, v in cur.items():
        groups[bool(failed_gates(pr[k]))].append((k, v[2000] / v[1200]))
    thr = min(x for _, x in groups[True])
    rows = []
    for label, failing in (("runs passing all four gates", False), ("runs failing a gate", True)):
        ratios = [x for _, x in groups[failing]]
        rows.append((label, len(ratios), fx(statistics.median(ratios), 2), fx(min(ratios), 2), fx(max(ratios), 2),
                     sum(1 for x in ratios if x >= thr), sum(1 for x in ratios if x > 1.0)))
    reach = ", ".join(f"`{k[0]}` q = {k[1]} seed {k[2]}: {x:.2f}" for k, x in groups[False] if x >= thr)
    return (md(["group (all three E1 arms, both q)", "runs", "median ratio RMSE_pos(u2000) / RMSE_pos(u1200)", "min",
                "max", f"runs with ratio >= {thr:.3f} (the smallest ratio among the failing runs)", "runs with ratio > 1"],
               rows) + f"\n\nPassing runs that reach the threshold: {reach}.")


def b_recovery_tier_check() -> str:
    """The five recovery metrics on the final and the development tier over every curve point (curves_all.csv)."""
    keys = ["stage2_peak_rel_err_abs", "e2_at_0", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0",
            "stage2_sym_err_max"]
    n, mx = 0, 0.0
    for r in read_csv(R / "curves" / "curves_all.csv"):
        n += 1
        for k in keys:
            mx = max(mx, abs(num(r["final_" + k]) - num(r["dev_" + k])))
    return (f"The five recovery metrics (abs peak error, e_hat_2(0), RMSE_pos, tail mean and symmetry error) are "
            f"evaluated on the closed-form recovery grid with state spacing 0.5 (`utils/v2_metrics.py`, "
            f"`recovery_metrics`), not on the verifier grid, so their development-tier and final-tier values are the "
            f"same function of the policy: the largest |final - development| over all {n:,} curve points of "
            f"`curves_all.csv` is {mx:g}. Only the verifier-based quantities (eta_2/DW, Gmax_full/DW, Delta, s, R, C) "
            f"differ between the tiers (block `tiers`).")


def b_e2_criterion() -> str:
    """The three parts of the E2 target per arm and q (95 % and 97.5 % intervals)."""
    rows = []
    allrows = read_csv(E2 / "criterion_e2.csv")
    for r in allrows:
        if r["q"] == "both" or r["arm"] == "DECISION":
            continue
        viol = f"{int(num(r['n_gate_violations']))}" + (f" (seed {r['violating_seeds']})" if r["violating_seeds"] else "")
        rows.append((f"`{r['arm']}`", r["role"], r["q"],
                     f"{num(r['mean_A_x']) * 1e3:.3f} / {num(r['mean_A_ref']) * 1e3:.3f}", sx(num(r["rho"])),
                     ci(num(r["ci95_lo"]), num(r["ci95_hi"])), ci(num(r["ci975_lo"]), num(r["ci975_hi"])), r["a_pass"],
                     viol, r["b_pass"], f"{sg(num(r['mean_gmax_diff']), 3)}, upper {sg(num(r['gmax_ci_hi']), 3)}",
                     r["c_pass"], r["met_q"]))
    dec = [r for r in allrows if r["arm"] == "DECISION"][0]
    t = md(["arm", "role", "q", "mean A x 1e3 (arm / `fb_a0`)", "rho", "95 % CI", "97.5 % CI",
            "(a) rho >= 0.20, lower > 0", "(b) gate violations", "b", "(c) Gmax_full/DW paired diff", "c", "met at q"],
           rows)
    return t + f"\n\nDecision (`criterion_e2.csv`, row `DECISION`): **{dec['status']}**"


def b_e2_curves(q: str = "50") -> str:
    """Seed medians of the final-tier eta_2/DW at selected checkpoints."""
    ups = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]
    idx = {(r["arm"], r["q"], int(r["update"])): r for r in read_csv(E2 / "learning_curves_e2.csv")}
    rows = [(f"`{arm}`", *[gx(num(idx[(arm, q, u)]["eta_median"]), 3) for u in ups]) for arm in E2_ARMS]
    return md(["arm", *[f"u{u}" for u in ups]], rows)


def b_e2_gate_counts() -> str:
    """Runs passing all four gates per arm and q, with the failed gates."""
    from scipy.stats import beta  # type: ignore

    rows = []
    pr = per_run("e2")
    for q in QS:
        for arm in E2_ARMS:
            rs = [r for r in pr if r["arm"] == arm and r["q"] == q]
            n = len(rs)
            k = sum(is_true(r["gates_all_pass"]) for r in rs)
            lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
            hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
            cnt = {name: sum(1 for r in rs if not is_true(r["gate_" + key])) for key, name in GATES}
            rows.append((q, f"`{arm}`", f"{k} / {n}", ci(lo, hi, 3, signed=False),
                         ", ".join(f"{a} {b}" for a, b in cnt.items())))
    return md(["q", "arm", "runs passing all four gates", "Clopper-Pearson 95 %", "gate failures"], rows)


E2_SEC = [("A_late", "area over local 200-1600 (normalised by 1400)", True),
          ("A_dev_online", "area of the development-tier online values", True),
          ("log10_mean", "mean log10(eta_2/DW) over the 64 points", False),
          ("first_le_level", "first export with eta_2/DW <= 0.005", False), ("eta_u1600", "eta_2/DW at u1600", False),
          ("peak_abs", "abs peak error at the freeze", False), ("gap", "peak gap at the freeze", False),
          ("rmse_pos", "RMSE_pos / e2*(0)", False), ("tail_mean", "tail mean / e2*(0)", False),
          ("s1_abs", "stage-1 abs relative error", False)]


def b_e2_secondary(arm: str = "fb_r") -> str:
    """Secondary statistics of one arm against `fb_a0` (descriptive)."""
    idx = {(r["arm"], r["q"], r["statistic"]): r for r in read_csv(E2 / "secondary_e2.csv")}
    rows = []
    for stat, label, has_rho in E2_SEC:
        for q in QS:
            r = idx[(arm, q, stat)]
            extra = f"{sx(num(r['rho']), 3)} {cig(num(r['rho_ci_lo']), num(r['rho_ci_hi']))}" if has_rho else ""
            rows.append((label if q == "50" else "", q, gx(num(r["mean_a"])), gx(num(r["mean_b"])),
                         sg(num(r["mean_diff"])), cig(num(r["ci_lo"]), num(r["ci_hi"])), f"{r['n_neg']} / {r['n_pos']}",
                         extra))
    return md(["statistic", "q", f"mean `{arm}`", "mean `fb_a0`", "mean diff", "95 % CI", "pairs < / >",
               "reduction rho of the mean (95 % CI)"], rows)


def b_e2_secondary_count() -> str:
    """How many of the descriptive secondary intervals exclude zero (multiplicity note)."""
    n = k = 0
    per: Dict[str, List[int]] = collections.defaultdict(lambda: [0, 0])
    for r in read_csv(E2 / "secondary_e2.csv"):
        if r["statistic"] in ("share_near", "share_mid", "dev_verifier_share_stage2", "wall_total_sec", "gmax_final"):
            continue
        lo, hi = num(r["ci_lo"]), num(r["ci_hi"])
        if math.isnan(lo) or math.isnan(hi):
            continue
        n += 1
        per[r["arm"]][0] += 1
        if lo > 0 or hi < 0:
            k += 1
            per[r["arm"]][1] += 1
    arms = "; ".join(f"`{a}` {v[1]} of {v[0]}" for a, v in per.items())
    return (f"{k} of {n} descriptive secondary intervals (95 %, `secondary_e2.csv`, without the strata shares, the cost "
            f"rows and `gmax_final`) exclude zero: {arms}. If no arm differed from `fb_a0`, about {0.05 * n:.1f} of "
            f"{n} would exclude zero by chance at 95 % (the statistics are correlated, so this is only a rough scale).")


def b_e2_weights() -> str:
    """Sampling weights in force after every check: deviation from the alpha-0 weights and the strata masses."""
    g: Dict[Any, List[Dict[str, str]]] = collections.defaultdict(list)
    for r in read_csv(E2 / "sampling_weights.csv"):
        g[(r["arm"], r["q"])].append(r)
    rows = []
    for q in QS:
        for arm in E2_ARMS:
            v = g[(arm, q)]

            def col(k: str) -> List[float]:
                return [num(r[k]) for r in v if not math.isnan(num(r[k]))]
            tv, rn, rm = col("tv_from_alpha0"), col("max_ratio_near"), col("max_ratio_mid")
            rows.append((q, f"`{arm}`", len(v), fx(statistics.median(tv), 3), fx(max(tv), 3),
                         f"{statistics.median(rn):.2f} / {max(rn):.2f}", f"{statistics.median(rm):.2f} / {max(rm):.2f}",
                         f"{min(col('mass_tail')):.4f}-{max(col('mass_tail')):.4f}",
                         f"{min(col('mass_near')):.3f}-{max(col('mass_near')):.3f}",
                         f"{min(col('mass_mid')):.3f}-{max(col('mass_mid')):.3f}"))
    return md(["q", "arm", "checks (20 runs x 64)", "TV from the alpha-0 weights: median", "max",
               "largest bin probability / stratum mean, near-tie: median / max", "middle: median / max", "tail mass",
               "near-tie mass",
               "middle mass"], rows)


def b_e2_strata() -> str:
    """Realised start shares per arm and q against the design."""
    pr = per_run("e2")
    rows = []
    for q in QS:
        for arm in E2_ARMS:
            rs = [r for r in pr if r["arm"] == arm and r["q"] == q]
            dn = num(rs[0]["design_lambda_P"])
            dm = num(rs[0]["design_lambda_M"])
            rows.append((q, f"`{arm}`", f"{dn:.3f} / {dm:.3f}",
                         f"{statistics.mean(num(r['share_near']) for r in rs):.4f} / "
                         f"{statistics.mean(num(r['share_mid']) for r in rs):.4f}"))
    return md(["q", "arm", "design near-tie / middle", "realised near-tie / middle (mean of 20 runs, all updates)"],
              rows)


def b_gmax_equality() -> str:
    """Gmax_full/DW against eta_2/DW and the stage-1 deviation gain, in every run of both experiments."""
    rows = []
    for exp in ("e1", "e2"):
        pr = per_run(exp)
        neq = sum(1 for r in pr if num(r["Gmax_full_over_dw"]) != num(r["eta_T_over_dw"]))
        d1 = [num(r["t1_Delta_final"]) for r in pr]
        ratio = max(num(r["t1_Delta_final"]) / num(r["eta_T_over_dw"]) for r in pr)
        rows.append((exp.upper(), len(pr), neq, gx(statistics.median(d1), 3), gx(max(d1), 3),
                     gx(min(num(r["eta_T_over_dw"]) for r in pr), 3), fx(ratio, 2)))
    return md(["experiment", "runs", "runs with Gmax_full/DW != eta_2/DW", "stage-1 one-step deviation gain Delta_1/DW: "
               "median", "max", "smallest eta_2/DW", "largest Delta_1 / eta_2 of a run"], rows)


def b_e2_stage1_gain() -> str:
    """Post hoc supplement: the stage-1 deviation gain Delta_1/DW (final tier) per arm and q, paired with `fb_a0`."""
    pr = {(r["arm"], r["q"], r["seed"]): r for r in per_run("e2")}
    seeds = sorted({r["seed"] for r in per_run("e2")})
    rows = []
    for q in QS:
        base = [num(pr[("fb_a0", q, s)]["t1_Delta_final"]) for s in seeds]
        for arm in E2_ARMS:
            v = [num(pr[(arm, q, s)]["t1_Delta_final"]) for s in seeds]
            d = [a - b for a, b in zip(v, base)]
            rows.append((q, f"`{arm}`", gx(statistics.mean(v), 3), gx(max(v), 3),
                         "-" if arm == "fb_a0" else sg(statistics.mean(d), 3),
                         "-" if arm == "fb_a0" else f"{sum(1 for x in d if x < 0)} / {sum(1 for x in d if x > 0)}"))
    return md(["q", "arm", "mean Delta_1/DW", "max over runs", "mean paired diff vs `fb_a0`", "pairs < / >"], rows)


def b_e2_predictions() -> str:
    """Predictions E2-P1 to E2-P3 against the outcome."""
    rows = []
    for r in read_csv(E2 / "predictions_e2.csv"):
        if r["prediction"] == "E2-P1":
            rows.append(("E2-P1", "`fb_rx` has a larger area than `fb_a0` at q = 60",
                         f"rho(`fb_rx`, q = 60) = {sx(num(r['observed_rho_fb_rx_q60']))}", r["outcome"]))
        elif r["prediction"] == "E2-P2":
            rows.append(("E2-P2", f"`{r['arm']}` falls short of 20 % at q = {int(num(r['q']))}; gain larger in the late window",
                         f"rho full {sx(num(r['rho_full']))}, rho local 200-1600 {sx(num(r['rho_late']))}",
                         "short of 20 %" if is_true(r["short_of_20pct"]) else "reaches 20 %"))
        else:
            rows.append(("E2-P3", f"development checks under 1 % of the stage-2 wall time (`{r['arm']}`)",
                         f"maximum share over runs {100 * num(r['max_dev_verifier_share_stage2']):.2f} %",
                         r["outcome"]))
    return md(["prediction", "statement", "observed", "outcome"], rows)


BLOCKS: Dict[str, Callable[..., str]] = {
    "launch": b_launch, "launch_wall": b_launch_wall, "checks": b_checks, "start_flags": b_start_flags,
    "config_arms": b_config_arms, "config_diffs": b_config_diffs,
    "e1_per_seed": b_e1_per_seed, "e1_criterion": b_e1_criterion, "e1_paired_peak": b_e1_paired_peak,
    "e1_failure_rates": b_e1_failure_rates, "e1_failed_runs": b_e1_failed_runs, "e1_decomp": b_e1_decomp,
    "e1_units": b_e1_units, "e1_curves": b_e1_curves, "e1_overview": b_e1_overview, "e1_locfree": b_e1_locfree,
    "e1_strata": b_e1_strata, "e1_pass_fail_split": b_e1_pass_fail_split, "e1_failure_traj": b_e1_failure_traj, "e2_overview": b_e2_overview, "tiers": b_tiers, "recovery_tier_check": b_recovery_tier_check, "e1_shares": b_e1_shares, "e2_peak_seed": b_e2_peak_seed, "e1_rmse_rise": b_e1_rmse_rise, "a2_rule_i": b_a2_rule_i, "a2_verbatim": b_a2_verbatim, "e2_early_share": b_e2_early_share, "cost": b_cost, "e1_predictions": b_e1_predictions,
    "e2_area_seed": b_e2_area_seed, "e2_criterion": b_e2_criterion, "e2_curves": b_e2_curves,
    "e2_gate_counts": b_e2_gate_counts, "e2_secondary": b_e2_secondary, "e2_secondary_count": b_e2_secondary_count,
    "e2_weights": b_e2_weights, "e2_strata": b_e2_strata, "gmax_equality": b_gmax_equality,
    "e2_stage1_gain": b_e2_stage1_gain, "e2_predictions": b_e2_predictions,
}


def run_block(name: str, kv: Dict[str, str]) -> str:
    """Print one block."""
    if name not in BLOCKS:
        raise SystemExit(f"unknown block {name!r}; known: {', '.join(sorted(BLOCKS))}")
    return BLOCKS[name](**kv)


def parse_kv(items: Sequence[str]) -> Dict[str, str]:
    """``key=value`` words into a dict."""
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def fill(text: str) -> str:
    """Replace every ``{{name key=value ...}}`` of a draft by the printed block."""
    def sub(m: "re.Match[str]") -> str:
        words = m.group(1).split()
        return run_block(words[0], parse_kv(words[1:]))
    return re.sub(r"\{\{([^{}]+)\}\}", sub, text)


def main() -> None:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block")
    ap.add_argument("--fill")
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("kv", nargs="*")
    a = ap.parse_args()
    if a.list:
        print("\n".join(sorted(BLOCKS)))
    elif a.fill:
        res = fill(Path(a.fill).read_text())
        if a.out:
            Path(a.out).write_text(res)
        else:
            sys.stdout.write(res)
    elif a.block:
        print(run_block(a.block, parse_kv(a.kv)))
    else:
        ap.error("--block, --fill or --list")


if __name__ == "__main__":
    main()
