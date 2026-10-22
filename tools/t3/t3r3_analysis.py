#!/usr/bin/env python3
"""Analysis of the T3-R3 development wave (pre-registered in ``reports/t3/r3/04_preregistration.md``).

Inputs: the evaluation tables of ``t3r3_eval.py`` (development wave: 120 runs, diagnostic: 2 runs), the run directories, the
infrastructure ledger, the post-launch checks. The pass rule is that of ``t3c_analysis.run_row`` (exit 0, global-RNG assertion ok, a
valid evaluation on both tiers, G-A, G-F and G-N with inclusive ``<=`` at full float64 precision; a run without a row, or that the
ledger lists as stopped without a verdict, is a failed run). Outputs (``--out-dir``):

  per_run.csv            one row per run: verdict, gate values, S3, the trigger values, first-attempt quantities, stretches, cost
  counts.csv             pass counts per arm and q with exact intervals, worst ``Gmax_full``, medians of the reported metrics
  headline.json          the counts, the target outcome per arm, the selection, the first-attempt invariant, the fire rates
  paired.csv             each candidate arm minus ``bw``, per seed: paired differences of ``Gmax_full`` and ``eta_3`` with paired
                         bootstrap intervals (10,000 resamples, seed 20261021), and the pass / fail cross-table (``crosstab.csv``)
  first_attempt.csv      the first-attempt outcome per arm (a fired run counts as a failure of its first attempt)
  causal_b.csv, b_pairs.csv   the causal reading of B (first attempts): S3, the exact McNemar test on the paired regime indicator,
                         the tail at the mean clamp and the minimum alpha at the first attempt's ``u01600``, the development
                         ``Delta`` at local 1600, the paired differences of ``eta_3`` and ``peak_T_signed`` for the seeds where
                         the trigger fired in neither arm, the stage-2 stretches, the fire rates
  fired.csv              every trigger that fired, with the outcome
  diagnostic.csv         the (q = 60, seed 60518) runs
  cost.csv               updates, policy rows, minibatch steps, wall time, verifier seconds per arm and the ratio to ``bw``

The closed form enters only through the evaluation table's own columns (``rmse_pos_T_over_eT0`` ...) and ``peak_T_signed``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import r4_common as R4  # noqa: E402
import t3_eval as te  # noqa: E402
import t3_metrics as tm  # noqa: E402
import t3c_analysis as ca  # noqa: E402
import t3r3_diag as dg  # noqa: E402
import t3r3_launch as lt  # noqa: E402
from run import ms_retry  # noqa: E402

N_BOOT, BOOT_SEED = 10000, 20261021
MIN_TOTAL, MIN_PER_Q = 38, 18
ARMS = lt.ARMS
CANDIDATES = lt.CANDIDATES
QS = lt.QS
REGIME = dg.REGIME_SHARE
REPORT_METRICS = ("eta_T_over_dw", "rmse_pos_T_over_eT0", "tail_mean_T_over_eT0", "peak_T_signed", "Gmax_full_over_dw", "EXP_root_over_dw")
NAN = float("nan")
# the fields that ``run_record`` leaves out for a run that did not finish
RECORD_DEFAULTS: Dict[str, Any] = {"fired": False, "would_fire": False, "trig_Delta": NAN, "trig_s": NAN, "trig_C": NAN, "trig_R": NAN,
                                   "trig_R_tail": NAN, "trig_reasons": "", "S3_frozen": NAN, "S3_first": NAN, "first_out_mu_at_clamp": NAN,
                                   "first_alpha_min": NAN, "first_mu_min": NAN, "stretch_nodes": 0, "has_stretch": False,
                                   "stretch_longest_d": "", "wall_sec": NAN, "dev_verifier_sec": NAN, "total_updates": None,
                                   "total_transitions": None, "total_minibatch_steps": None, "total_dev_checks": None,
                                   "a2_final_Delta": NAN, "a2_final_s": NAN, "a2_final_C": NAN}


# ---------------------------------------------------------------------------------- pass counts and the target
def cp(k: int, n: int) -> Tuple[float, float]:
    """Exact two-sided 95 % interval of ``k / n``."""
    return ca.clopper_pearson(k, n) if n else (NAN, NAN)


def arm_counts(rows: Sequence[Mapping[str, Any]], arm: str) -> Dict[str, Any]:
    """Pass counts of one arm per q and in total over its 40 development runs, with exact intervals."""
    out: Dict[str, Any] = {"arm": arm}
    for q in QS:
        sel = [r for r in rows if r["arm"] == arm and r["q"] == q and r["run_kind"] == "dev"]
        k = sum(1 for r in sel if r["passed"])
        lo, hi = cp(k, len(sel))
        out.update({f"n_q{q}": len(sel), f"pass_q{q}": k, f"ci_lo_q{q}": lo, f"ci_hi_q{q}": hi})
    k, n = out["pass_q50"] + out["pass_q60"], out["n_q50"] + out["n_q60"]
    lo, hi = cp(k, n)
    out.update({"n_total": n, "pass_total": k, "ci_lo_total": lo, "ci_hi_total": hi})
    return out


def target_outcome(cnt: Mapping[str, Any]) -> Dict[str, Any]:
    """The development target: at least 38 of the 40 runs pass and at least 18 of 20 at each q (all 40 runs must be planned)."""
    full = bool(cnt["n_q50"] == cnt["n_q60"] == 20)
    ok_t, ok50, ok60 = cnt["pass_total"] >= MIN_TOTAL, cnt["pass_q50"] >= MIN_PER_Q, cnt["pass_q60"] >= MIN_PER_Q
    return {"arm": cnt["arm"], "total_condition": bool(ok_t), "q50_condition": bool(ok50), "q60_condition": bool(ok60),
            "reaches_target": bool(full and ok_t and ok50 and ok60), "pass_total": int(cnt["pass_total"]),
            "pass_q50": int(cnt["pass_q50"]), "pass_q60": int(cnt["pass_q60"])}


def selection(targets: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """The first candidate arm in the order ``bwA``, ``bwAB`` that reaches the target is proposed for the lock (the control is
    not eligible); if none does, no scheme is proposed. If both reach it, ``bwA`` is proposed (the PI may lock ``bwAB`` instead only
    with a stated reason from the items of section 2 of the G1 reply)."""
    for arm in CANDIDATES:
        if targets[arm]["reaches_target"]:
            both = all(targets[a]["reaches_target"] for a in CANDIDATES)
            return {"proposed": arm, "both_reach_target": bool(both),
                    "reason": f"{arm} is the first candidate arm in the order {list(CANDIDATES)} that reaches the target"
                              + ("; both arms reach it, so bwA is proposed" if both else "")}
    return {"proposed": None, "both_reach_target": False, "reason": "no candidate arm reaches the target: no scheme is proposed"}


# ---------------------------------------------------------------------------------- per-run records
def read_json(path: Path) -> Dict[str, Any]:
    """A JSON file as a dict."""
    return json.loads(Path(path).read_text())


def first_check(d: Path) -> Dict[str, Any]:
    """The check that the trigger reads: the last row of the first attempt's stage-3 check file (``attempt1/`` if the run fired)."""
    p = d / "attempt1" / "ms_checks_stage3.csv" if (d / "retry.json").is_file() else d / "ms_checks_stage3.csv"
    last = dg.read_csv(p)[-1]
    return {"stage": 3, "local": int(float(last["local"])), "Delta": dg.fnum(last["Delta"]), "s": dg.fnum(last["s"]), "C": dg.fnum(last["C"]),
            "R": dg.fnum(last["R"]), "R_tail": dg.fnum(last["R_tail"])}


def stage2_stretch(d: Path, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """The stage-2 stretch of the frozen stage 2 (the export ``u03200``): in-band final-tier nodes with mean effort below 1."""
    spec = te.spec_of(cfg)
    ppo = cfg["record"]["ppo"]
    arrays = {k: v for k, v in np.load(d / "weights" / "u03200.npz").items()}
    grid = dg.stage_grid(spec, 2)
    e = dg.network_on_grid(arrays, spec, grid, 2, hidden=int(ppo["hidden"]), c_min=float(ppo["c_min"]), mu_clamp=float(ppo["mu_clamp"]),
                           with_critic=False)["e_hat"]
    s = dg.entry_stretch(grid, e, spec.q)
    return {"stretch_nodes": int(s["n_nodes"]), "has_stretch": bool(s["n_nodes"] > 0), "stretch_longest_d": s["longest_d"]}


def stage3_export_summary(d: Path, cfg: Mapping[str, Any], u: int) -> Dict[str, float]:
    """``export_summary`` of the stage-3 network of export ``u`` (a path under ``d``) on the final-tier grid of D_3."""
    spec = te.spec_of(cfg)
    ppo = cfg["record"]["ppo"]
    arrays = {k: v for k, v in np.load(d / f"u{u:05d}.npz").items()}
    grid = dg.stage_grid(spec, spec.T)
    net = dg.network_on_grid(arrays, spec, grid, spec.T, hidden=int(ppo["hidden"]), c_min=float(ppo["c_min"]), mu_clamp=float(ppo["mu_clamp"]))
    return dg.export_summary(grid, net, spec.q)


def run_record(args: Any) -> Dict[str, Any]:
    """Everything the analysis reads from one run directory (no evaluation row involved)."""
    job, = args
    d = Path(job["run_dir"])
    out: Dict[str, Any] = {"run": job["run"], "arm": job["arm"], "q": int(job["q"]), "seed": int(job["seed"]), "run_kind": job["run_kind"],
                           "run_dir": str(d)}
    st = te.status_of(d)
    out["status"], out["exit_code"] = st.get("state", "missing"), st.get("exit_code")
    if st.get("state") != "done" or st.get("exit_code") != 0:
        return out
    cfg = te.read_json(d / "run_config.json")
    fired = (d / "retry.json").is_file()
    out["fired"] = fired
    chk = first_check(d)
    out["trig_Delta"], out["trig_s"], out["trig_C"], out["trig_R"], out["trig_R_tail"] = chk["Delta"], chk["s"], chk["C"], chk["R"], chk["R_tail"]
    b3 = int(cfg["pipeline"]["budgets"]["3"])
    fires, reasons = ms_retry.trigger(chk, lt.RETRY["delta_gt"], b3)
    out["would_fire"], out["trig_reasons"] = bool(fires), ";".join(reasons)
    rows_f = dg.read_csv(d / "ms_updates.csv")
    out["S3_frozen"] = dg.s3_of_rows(rows_f)[0]
    rows_1 = dg.read_csv(d / "attempt1" / "ms_updates_stage3.csv") if fired else rows_f
    out["S3_first"] = dg.s3_of_rows(rows_1)[0]
    first_dir = d / "attempt1" / "weights" if fired else d / "weights"
    sm = stage3_export_summary(first_dir, cfg, 1600) if (first_dir / "u01600.npz").is_file() else {}
    out["first_out_mu_at_clamp"] = sm.get("out_mu_at_clamp", NAN)
    out["first_alpha_min"] = sm.get("alpha_min", NAN)
    out["first_mu_min"] = sm.get("mu_min", NAN)
    out.update(stage2_stretch(d, cfg))
    summ = te.read_json(d / "ms_run_summary.json")
    out["wall_sec"] = float(st.get("total_wall_sec", NAN))
    out["dev_verifier_sec"] = float(summ["costs"]["seconds"].get("dev_verifier", 0.0))
    if fired:
        rj = te.read_json(d / "retry.json")
        t = rj["totals"]
        out.update(total_updates=t["updates"], total_transitions=t["transitions"], total_minibatch_steps=t["minibatch_steps"],
                   total_dev_checks=t["dev_checks"], dev_verifier_sec=out["dev_verifier_sec"] + float(rj["attempt1"]["counts"]["dev_verifier_sec"]),
                   a2_final_Delta=dg.fnum(rj["attempt2"]["final_check"]["Delta"]), a2_final_s=dg.fnum(rj["attempt2"]["final_check"]["s"]),
                   a2_final_C=dg.fnum(rj["attempt2"]["final_check"]["C"]))
    else:
        out.update(total_updates=3800, total_transitions=3800 * 512, total_minibatch_steps=76000, total_dev_checks=152,
                   a2_final_Delta=NAN, a2_final_s=NAN, a2_final_C=NAN)
    return out


def merge_eval(rec: Mapping[str, Any], erow: Optional[Mapping[str, Any]], gates: Mapping[str, Any], stopped: Sequence[str]) -> Dict[str, Any]:
    """The run record joined with its evaluation row: verdict, gate values and the reported metrics."""
    v = ca.run_row(erow, gates)
    if rec["run"] in stopped:
        v.update(passed=False, fail_reason="infrastructure failure twice (stopped without a verdict)")
    out = {**RECORD_DEFAULTS, **rec}
    out.update({k: v[k] for k in ("exit_ok", "rng_ok", "eval_ok", "ga_eta", "ga_rmse", "ga_tail", "gate_G_A", "gate_G_F", "gate_G_N",
                                  "passed", "fail_reason")})
    out["failing_gates"] = ";".join(g for g in ("gate_G_A", "gate_G_F", "gate_G_N") if v["exit_ok"] and v["eval_ok"] and not v[g])
    for c in REPORT_METRICS + ("dev_eta_T_over_dw", "dev_Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "eT_at_0", "eT_star_at_0"):
        out[c] = ca.fnum(erow.get(c)) if erow is not None else NAN
    return out


# ---------------------------------------------------------------------------------- tables
def median_or_nan(vals: Sequence[float]) -> float:
    """Median of the finite values (NaN if none)."""
    a = np.array([v for v in vals if v == v], dtype=float)
    return float(np.median(a)) if a.size else NAN


def counts_table(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per arm and q: pass counts with exact intervals, the worst ``Gmax_full`` and its run, the medians of the reported metrics."""
    out: List[Dict[str, Any]] = []
    for arm in ARMS:
        for q in (50, 60, "both"):
            sel = [r for r in rows if r["arm"] == arm and r["run_kind"] == "dev" and (q == "both" or r["q"] == q)]
            k = sum(1 for r in sel if r["passed"])
            lo, hi = cp(k, len(sel))
            fin = [r for r in sel if r["Gmax_full_over_dw"] == r["Gmax_full_over_dw"]]
            worst = max(fin, key=lambda r: r["Gmax_full_over_dw"]) if fin else None
            row = {"arm": arm, "q": q, "runs": len(sel), "pass": k, "ci_lo": lo, "ci_hi": hi,
                   "worst_Gmax_full": worst["Gmax_full_over_dw"] if worst else NAN, "worst_Gmax_full_run": worst["run"] if worst else ""}
            for m in REPORT_METRICS:
                row[f"median_{m}"] = median_or_nan([r[m] for r in sel])
            out.append(row)
    return out


def paired_vs(rows: Sequence[Mapping[str, Any]], a: str, b: str, stat: str, q: Any) -> Dict[str, Any]:
    """Paired differences ``a - b`` of ``stat`` over the development seeds of q (``"both"``: pooled), complete pairs only, with the
    paired bootstrap interval (10,000 resamples of the pairs, seed 20261021)."""
    xa, xb = [], []
    for qq in QS:
        if q != "both" and qq != q:
            continue
        for s in lt.SEEDS:
            ra = next(r for r in rows if r["arm"] == a and r["q"] == qq and r["seed"] == s and r["run_kind"] == "dev")
            rb = next(r for r in rows if r["arm"] == b and r["q"] == qq and r["seed"] == s and r["run_kind"] == "dev")
            xa.append(ra[stat])
            xb.append(rb[stat])
    st = R4.paired_stats(xa, xb, N_BOOT, BOOT_SEED)
    return {"a": a, "b": b, "stat": stat, "q": q, **{k: st[k] for k in ("n_pairs", "mean_a", "mean_b", "mean_diff", "median_diff", "ci_lo", "ci_hi", "n_neg", "n_pos")}}


def crosstab(rows: Sequence[Mapping[str, Any]], arm: str, ref: str = "bw") -> Dict[str, Any]:
    """The pass / fail cross-table of an arm against the reference over the 40 development seeds."""
    c = {"both_pass": 0, "ref_only": 0, "arm_only": 0, "both_fail": 0}
    for q in QS:
        for s in lt.SEEDS:
            ra = next(r for r in rows if r["arm"] == arm and r["q"] == q and r["seed"] == s and r["run_kind"] == "dev")
            rb = next(r for r in rows if r["arm"] == ref and r["q"] == q and r["seed"] == s and r["run_kind"] == "dev")
            c["both_pass" if ra["passed"] and rb["passed"] else "ref_only" if rb["passed"] else "arm_only" if ra["passed"] else "both_fail"] += 1
    return {"arm": arm, "ref": ref, **c}


def first_attempt_pass(r: Mapping[str, Any]) -> bool:
    """Pass of the first attempt with a fired run counted as a failure of it. For a run that did not fire the first attempt is the
    run, so its pass is the run's pass; a fired run's first attempt is a failure by the convention of the G1 reply."""
    return bool(r["passed"] and not r.get("would_fire", False)) if r["arm"] == "bw" else bool(r["passed"] and not r.get("fired", False))


def first_attempt_table(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The first-attempt outcome per arm, and the invariant that ``bwA`` equals ``bw`` under the same convention (a ``bw`` run is
    counted as fired when the trigger rule applied to its own stage-3 check fires). A per-seed mismatch is a stop."""
    out: Dict[str, Any] = {"arms": {}, "mismatch_bwA_vs_bw": []}
    for arm in ARMS:
        sel = [r for r in rows if r["arm"] == arm and r["run_kind"] == "dev"]
        out["arms"][arm] = {"first_attempt_pass": sum(1 for r in sel if first_attempt_pass(r)), "pass_final": sum(1 for r in sel if r["passed"]),
                            "fired": sum(1 for r in sel if (r.get("fired") if arm != "bw" else r.get("would_fire")))}
    for q in QS:
        for s in lt.SEEDS:
            a = next(r for r in rows if r["arm"] == "bwA" and r["q"] == q and r["seed"] == s and r["run_kind"] == "dev")
            b = next(r for r in rows if r["arm"] == "bw" and r["q"] == q and r["seed"] == s and r["run_kind"] == "dev")
            if first_attempt_pass(a) != first_attempt_pass(b) or bool(a.get("fired")) != bool(b.get("would_fire")):
                out["mismatch_bwA_vs_bw"].append({"q": q, "seed": s, "bwA": first_attempt_pass(a), "bw": first_attempt_pass(b),
                                                  "bwA_fired": bool(a.get("fired")), "bw_would_fire": bool(b.get("would_fire"))})
    out["stop"] = bool(out["mismatch_bwA_vs_bw"])
    raw_bw = out["arms"]["bw"]["pass_final"]
    out["bw_raw_pass_count"] = raw_bw
    out["bw_fired_runs_that_pass_in_bw"] = sum(1 for r in rows if r["arm"] == "bw" and r["run_kind"] == "dev" and r["passed"] and r.get("would_fire"))
    return out


def mcnemar_exact(x: Sequence[bool], y: Sequence[bool]) -> Dict[str, Any]:
    """Exact McNemar test (two-sided binomial test on the discordant pairs) for paired indicators ``x`` and ``y``."""
    n10 = sum(1 for a, b in zip(x, y) if a and not b)
    n01 = sum(1 for a, b in zip(x, y) if b and not a)
    n = n10 + n01
    p = 1.0 if n == 0 else float(binomtest(min(n10, n01), n, 0.5, alternative="two-sided").pvalue)
    return {"n_x_only": n10, "n_y_only": n01, "n_discordant": n, "p_exact": p}


def causal_b(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The causal reading of B on first attempts (see the module docstring): per-arm summaries and per-pair tests."""
    per_arm: List[Dict[str, Any]] = []
    for arm in ARMS:
        for q in (50, 60, "both"):
            sel = [r for r in rows if r["arm"] == arm and r["run_kind"] == "dev" and (q == "both" or r["q"] == q)]
            reg = [r for r in sel if r["S3_first"] > REGIME]
            per_arm.append({"arm": arm, "q": q, "runs": len(sel), "S3_first_gt_0.2": len(reg), "median_S3_first": median_or_nan([r["S3_first"] for r in sel]),
                            "median_first_out_mu_at_clamp": median_or_nan([r["first_out_mu_at_clamp"] for r in sel]),
                            "runs_with_tail_at_mean_clamp": sum(1 for r in sel if r["first_out_mu_at_clamp"] > 0),
                            "median_first_alpha_min": median_or_nan([r["first_alpha_min"] for r in sel]),
                            "median_first_trigger_Delta_local1600": median_or_nan([r["trig_Delta"] for r in sel]),
                            "fired": sum(1 for r in sel if (r.get("fired") if arm != "bw" else r.get("would_fire"))),
                            "stretch_runs_final": sum(1 for r in sel if r.get("has_stretch")),
                            "median_stretch_nodes": median_or_nan([r["stretch_nodes"] for r in sel])})
    pairs: List[Dict[str, Any]] = []
    for a, b in (("bwAB", "bw"), ("bwAB", "bwA"), ("bwA", "bw")):
        for q in (50, 60, "both"):
            ra = [next(r for r in rows if r["arm"] == a and r["q"] == qq and r["seed"] == s and r["run_kind"] == "dev")
                  for qq in QS if q in (qq, "both") for s in lt.SEEDS]
            rb = [next(r for r in rows if r["arm"] == b and r["q"] == qq and r["seed"] == s and r["run_kind"] == "dev")
                  for qq in QS if q in (qq, "both") for s in lt.SEEDS]
            mc = mcnemar_exact([x["S3_first"] > REGIME for x in ra], [x["S3_first"] > REGIME for x in rb])
            row = {"arm_a": a, "arm_b": b, "q": q, "regime_a": sum(1 for x in ra if x["S3_first"] > REGIME),
                   "regime_b": sum(1 for x in rb if x["S3_first"] > REGIME), **mc}
            # the seeds where the trigger fired in neither arm: paired differences of eta_3 and peak_T_signed
            keep = [i for i, (x, y) in enumerate(zip(ra, rb)) if not (x.get("fired") if a != "bw" else x.get("would_fire"))
                    and not (y.get("fired") if b != "bw" else y.get("would_fire"))]
            row["n_neither_fired"] = len(keep)
            for stat in ("eta_T_over_dw", "peak_T_signed"):
                st = R4.paired_stats([ra[i][stat] for i in keep], [rb[i][stat] for i in keep], N_BOOT, BOOT_SEED)
                row.update({f"{stat}_mean_diff": st["mean_diff"], f"{stat}_ci_lo": st["ci_lo"], f"{stat}_ci_hi": st["ci_hi"], f"{stat}_n_pairs": st["n_pairs"]})
            pairs.append(row)
    return per_arm, pairs


def cost_table(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per arm: updates, policy rows, minibatch steps, development checks, wall time and verifier seconds, and the ratio to ``bw``
    (of the means over the 40 development runs)."""
    out = []
    ref: Dict[str, float] = {}
    for arm in ARMS:
        sel = [r for r in rows if r["arm"] == arm and r["run_kind"] == "dev" and r.get("total_updates") is not None]
        n = len(sel)
        row = {"arm": arm, "runs": n, "mean_updates": np.mean([r["total_updates"] for r in sel]) if n else NAN,
               "mean_policy_rows": np.mean([r["total_transitions"] for r in sel]) if n else NAN,
               "mean_minibatch_steps": np.mean([r["total_minibatch_steps"] for r in sel]) if n else NAN,
               "mean_dev_checks": np.mean([r["total_dev_checks"] for r in sel]) if n else NAN,
               "total_wall_sec": float(np.sum([r["wall_sec"] for r in sel])) if n else NAN,
               "mean_wall_sec": float(np.mean([r["wall_sec"] for r in sel])) if n else NAN,
               "median_wall_sec": median_or_nan([r["wall_sec"] for r in sel]),
               "mean_dev_verifier_sec": float(np.mean([r["dev_verifier_sec"] for r in sel])) if n else NAN}
        if arm == "bw":
            ref = {k: row[k] for k in row if k.startswith("mean_") or k == "total_wall_sec"}
        for k in list(ref):
            row[f"ratio_{k}"] = row[k] / ref[k] if ref.get(k) else NAN
        out.append(row)
    return out


def fired_table(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every trigger that fired (candidate arms), with the trigger values and the outcome of the run."""
    return [{"run": r["run"], "arm": r["arm"], "run_kind": r["run_kind"], "q": r["q"], "seed": r["seed"], "trigger_Delta": r["trig_Delta"],
             "trigger_s": r["trig_s"], "trigger_C": r["trig_C"], "reasons": r["trig_reasons"], "attempt2_final_Delta": r["a2_final_Delta"],
             "passed": r["passed"], "fail_reason": r["fail_reason"], "eta_3": r["eta_T_over_dw"], "Gmax_full": r["Gmax_full_over_dw"]}
            for r in rows if r["arm"] in CANDIDATES and r.get("fired")]


def diagnostic_rows(rows: Sequence[Mapping[str, Any]], checks: Optional[Mapping[str, Any]], dg2_runs: Optional[Path]) -> List[Dict[str, Any]]:
    """The two (q = 60, seed 60518) runs: attempt 1 against T3-R2's original (from the post-launch checks), whether A fired, and for
    attempt 2 the formation measure F(u) at every stage-3 export, the formation export, the gap-sensitivity and the final gates."""
    thr = NAN
    if dg2_runs and Path(dg2_runs).is_file():
        thr = float(next(r for r in dg.read_csv(Path(dg2_runs)) if int(float(r["q"])) == 60)["formation_threshold"])
    out = []
    for r in [x for x in rows if x["run_kind"] == "diagnostic"]:
        d = Path(r["run_dir"])
        row = {"arm": r["arm"], "run": r["run"], "fired": r.get("fired"), "trigger_Delta": r.get("trig_Delta"), "trigger_s": r.get("trig_s"),
               "trigger_C": r.get("trig_C"), "passed": r["passed"], "fail_reason": r["fail_reason"], "eta_3": r["eta_T_over_dw"],
               "rmse_pos": r["rmse_pos_T_over_eT0"], "Gmax_full": r["Gmax_full_over_dw"], "S3_first": r.get("S3_first"), "S3_frozen": r.get("S3_frozen")}
        if checks:
            c = next((x for x in checks["runs"] if x["run"] == r["run"]), None)
            row["attempt1_equals_T3R2_original"] = bool(c and c.get("pair_ok")) if c else None
        if (d / "weights").is_dir() and r.get("status") == "done":
            cfg = te.read_json(d / "run_config.json")
            ups = list(range(25, 1601, 25))
            Fs, sens = [], []
            for u in ups:
                sm = stage3_export_summary(d / "weights", cfg, u)
                Fs.append(sm["F"])
                sens.append(sm["sens_in_mean"])
            Fa, ua = np.array(Fs), np.array(ups)
            row.update(F_u1600=float(Fa[-1]), formation_threshold=thr, formation_export=dg.formation_export(Fa, ua, thr) if thr == thr else None,
                       sens_u1600=float(sens[-1]), sens_min_u100_300=float(min(s for u, s in zip(ups, sens) if 100 <= u <= 300)))
        out.append(row)
    return out


# ---------------------------------------------------------------------------------- driver
def analyse(root: Path, dev_eval: Path, diag_eval: Optional[Path], out_dir: Path, checks_path: Optional[Path], ledger_path: Path,
            workers: int = 8, dg2_runs: Optional[Path] = None) -> Dict[str, Any]:
    """Run the whole analysis; write the tables and ``headline.json``; return the headline."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    import t3r3_checks as ck
    out_dir.mkdir(parents=True, exist_ok=True)
    gates = tm.load_criteria()[0]["gates"]
    ledger = ca.read_ledger(ledger_path)
    stopped = ca.stopped_runs(ledger)
    evals = {r["run"]: r for r in pd.read_csv(dev_eval).to_dict("records")}
    if diag_eval and Path(diag_eval).is_file():
        evals.update({r["run"]: r for r in pd.read_csv(diag_eval).to_dict("records")})
    jobs = ck.all_jobs(root)
    if workers <= 1:
        recs = [run_record((j,)) for j in jobs]
    else:
        import torch  # noqa: F401
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as pool:
            recs = list(pool.map(run_record, [(j,) for j in jobs]))
    rows = [merge_eval(rec, evals.get(rec["run"]), gates, stopped) for rec in recs]
    pd.DataFrame(rows).to_csv(out_dir / "per_run.csv", index=False)
    cnt = {arm: arm_counts(rows, arm) for arm in ARMS}
    targets = {arm: target_outcome(cnt[arm]) for arm in ARMS}
    sel = selection(targets)
    ct = counts_table(rows)
    pd.DataFrame(ct).to_csv(out_dir / "counts.csv", index=False)
    paired = [paired_vs(rows, arm, "bw", stat, q) for arm in CANDIDATES for stat in ("Gmax_full_over_dw", "eta_T_over_dw") for q in (50, 60, "both")]
    paired += [paired_vs(rows, "bwAB", "bwA", stat, q) for stat in ("Gmax_full_over_dw", "eta_T_over_dw") for q in (50, 60, "both")]
    pd.DataFrame(paired).to_csv(out_dir / "paired.csv", index=False)
    pd.DataFrame([crosstab(rows, a) for a in CANDIDATES] + [crosstab(rows, "bwAB", "bwA")]).to_csv(out_dir / "crosstab.csv", index=False)
    fa = first_attempt_table(rows)
    pd.DataFrame([{"arm": a, **v} for a, v in fa["arms"].items()]).to_csv(out_dir / "first_attempt.csv", index=False)
    per_arm, pairs = causal_b(rows)
    pd.DataFrame(per_arm).to_csv(out_dir / "causal_b.csv", index=False)
    pd.DataFrame(pairs).to_csv(out_dir / "b_pairs.csv", index=False)
    pd.DataFrame(fired_table(rows)).to_csv(out_dir / "fired.csv", index=False)
    checks = json.loads(Path(checks_path).read_text()) if checks_path and Path(checks_path).is_file() else None
    pd.DataFrame(diagnostic_rows(rows, checks, dg2_runs)).to_csv(out_dir / "diagnostic.csv", index=False)
    pd.DataFrame(cost_table(rows)).to_csv(out_dir / "cost.csv", index=False)
    head = {"counts": cnt, "targets": targets, "selection": sel, "first_attempt": fa, "stopped_without_verdict": stopped,
            "infra_events": ledger.get("events", []), "n_rows": len(rows), "n_dev_runs": sum(1 for r in rows if r["run_kind"] == "dev"),
            "paired_bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED}}
    (out_dir / "headline.json").write_text(json.dumps(head, indent=1, default=str))
    return head


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring). Exit code 3 if the first-attempt invariant of ``bwA`` against ``bw`` fails (a stop)."""
    ap = argparse.ArgumentParser(description="T3-R3 development-wave analysis", allow_abbrev=False)
    ap.add_argument("--root", default=str(lt.DEFAULT_ROOT))
    ap.add_argument("--dev-eval", required=True)
    ap.add_argument("--diag-eval", default=None)
    ap.add_argument("--checks", default=None, help="launch_checks.json (for the diagnostic C-PAIR result)")
    ap.add_argument("--ledger", required=True, help="infra_events.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dg2-runs", default=str(ROOT / "results" / "t3_r3" / "diag" / "dg2_runs.csv"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    head = analyse(Path(a.root), Path(a.dev_eval), Path(a.diag_eval) if a.diag_eval else None, Path(a.out_dir),
                   Path(a.checks) if a.checks else None, Path(a.ledger), a.workers, Path(a.dg2_runs))
    print(json.dumps({k: head[k] for k in ("counts", "targets", "selection")}, indent=1, default=str))
    if head["first_attempt"]["stop"]:
        print("STOP: the first-attempt outcome of bwA differs from bw's:", head["first_attempt"]["mismatch_bwA_vs_bw"], file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
