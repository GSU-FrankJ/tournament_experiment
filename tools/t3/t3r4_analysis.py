#!/usr/bin/env python3
"""Analysis of the T3-R4 confirmation (prompt 3.3): the verdict under the rule and the reported items.

Input: the evaluation table of ``t3r4_eval.py``, the run directories, the infrastructure ledger, optionally ``launch_checks.json``.
The pass rule is that of ``t3c_analysis.run_row`` (exit 0, global-RNG assertion ok, an evaluation without error that is valid on both
tiers, G-A, G-F and G-N, every ``<=`` inclusive in float64). **The verdict:** the confirmation passes iff at most 6 of the 80 runs fail
and at most 4 of the 40 runs fail at each q; a planned run without a row, an evaluation problem that a re-run did not cure, an exit
other than 0 and a global-RNG violation are failed runs; a run the ledger lists as stopped without a verdict (infrastructure failure
twice) makes it a stop without a verdict.

Reported, deciding nothing (``--out-dir``): ``headline.json`` (counts with exact intervals, the verdict with its three conditions, the
trigger, the first-attempt outcome, the comparisons), ``per_run.csv``, ``counts.csv``, ``fired.csv``, ``failures.csv`` (gates and values,
the stage of the failing quantity, S3, the stage-2 stretch, F(u01600), the last-16-export range of the failing stage-3 quantity, the
suggested class), ``last16.csv``, ``cost.csv``, ``comparisons.csv``.

The class of a failure is suggested by rule and confirmed by hand in the report: (K) S3 of the frozen attempt above 0.2; (F) the
formation measure F(u01600) of the frozen attempt not above the formation threshold of its q (half the median final F of the passing
runs of T3-R3's ``dg2_runs.csv``); (V) the failing stage-3 value is the largest of the last 16 exports of the phase; (U) otherwise.
(E), an implementation error, is a judgement and is never suggested.

Usage:
    python tools/t3/t3r4_analysis.py --root $PWD/results/t3_r4/confirmation --eval results/t3_r4/confirmation/eval/per_run.csv \
        --ledger results/t3_r4/confirmation/infra_events.json --out-dir results/t3_r4/confirmation/analysis --workers 16
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import t3_eval as te  # noqa: E402
import t3c_analysis as ca  # noqa: E402
import t3r2_audit as au  # noqa: E402
import t3r3_analysis as r3a  # noqa: E402
import t3r3_diag as dg  # noqa: E402
import t3r3_last16 as l16  # noqa: E402
import t3r4_checks as ck  # noqa: E402
import t3r4_launch as lt  # noqa: E402

QS = lt.QS
SEEDS = lt.SEEDS
MAX_FAIL_TOTAL, MAX_FAIL_PER_Q = 6, 4
REGIME = dg.REGIME_SHARE
NAN = float("nan")
CRITERIA = ROOT / "reports" / "t3" / "r1" / "prereg_criteria.json"
DG2_RUNS = ROOT / "results" / "t3_r3" / "diag" / "dg2_runs.csv"
T3R3_PER_RUN = ROOT / "results" / "t3_r3" / "dev" / "analysis" / "per_run.csv"
T3R2_PER_RUN = ROOT / "results" / "t3_r2" / "confirmation" / "analysis" / "per_run_table.csv"
T3R1_PER_RUN = ROOT / "results" / "t3_r1" / "eval" / "per_run.csv"
FAIL_QUANTITIES = (("ga_eta", "eta_3_over_dw", "eta_T_over_dw", 0.005), ("ga_rmse", "rmse_pos_T_over_eT0", "rmse_pos_T_over_eT0", 0.05),
                   ("ga_tail", "tail_mean_T_over_eT0", "tail_mean_T_over_eT0", 0.02))


# ---------------------------------------------------------------------------------- the rule
def rule_verdict(fail_q50: int, fail_q60: int, stopped: Sequence[str]) -> Dict[str, Any]:
    """The verdict under the confirmation rule.

    Args:
        fail_q50: Failed runs among the 40 of q = 50.
        fail_q60: Failed runs among the 40 of q = 60.
        stopped: Runs the ledger lists as stopped without a verdict.

    Returns:
        ``{"verdict": "pass" | "fail" | "no_verdict", "reason", "total_condition", "q50_condition", "q60_condition", ...}``.
    """
    t_ok = bool(fail_q50 + fail_q60 <= MAX_FAIL_TOTAL)
    a_ok, b_ok = bool(fail_q50 <= MAX_FAIL_PER_Q), bool(fail_q60 <= MAX_FAIL_PER_Q)
    if stopped:
        v, why = "no_verdict", f"the confirmation stopped without a verdict: infrastructure failure twice in {list(stopped)}"
    elif t_ok and a_ok and b_ok:
        v, why = "pass", "at most 6 of the 80 runs fail and at most 4 of the 40 at each q"
    else:
        miss = [n for n, ok in (("total failures <= 6", t_ok), ("q=50 failures <= 4", a_ok), ("q=60 failures <= 4", b_ok)) if not ok]
        v, why = "fail", "not met: " + ", ".join(miss)
    return {"verdict": v, "reason": why, "total_condition": t_ok, "q50_condition": a_ok, "q60_condition": b_ok,
            "fail_total": int(fail_q50 + fail_q60), "fail_q50": int(fail_q50), "fail_q60": int(fail_q60),
            "rule": {"max_failures_total": MAX_FAIL_TOTAL, "max_failures_per_q": MAX_FAIL_PER_Q}, "stopped_without_verdict": list(stopped)}


def counts_of(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pass counts per q and in total over the planned runs, with exact intervals."""
    out: Dict[str, Any] = {}
    for q in QS:
        sel = [r for r in rows if int(r["q"]) == q]
        k = sum(1 for r in sel if r["passed"])
        lo, hi = r3a.cp(k, len(sel))
        out.update({f"n_q{q}": len(sel), f"pass_q{q}": k, f"fail_q{q}": len(sel) - k, f"ci_lo_q{q}": lo, f"ci_hi_q{q}": hi})
    k, n = out["pass_q50"] + out["pass_q60"], out["n_q50"] + out["n_q60"]
    lo, hi = r3a.cp(k, n)
    out.update({"n_total": n, "pass_total": k, "fail_total": n - k, "ci_lo_total": lo, "ci_hi_total": hi})
    return out


# ---------------------------------------------------------------------------------- failures
def failing_quantities(r: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The failing quantities of a run with the stage they live at: G-A (``eta_3``, ``rmse_pos``, ``tail_mean``) is stage 3; G-F is the stage
    ``Gmax_full_t`` at which the maximal deviation gain occurs; G-N is the development-final difference of ``eta_3`` (stage 3) or of
    ``Gmax_full`` (stage ``Gmax_full_t``)."""
    out: List[Dict[str, Any]] = []
    if not (r["exit_ok"] and r["eval_ok"]):
        return [{"quantity": r["fail_reason"], "value": NAN, "limit": NAN, "stage": ""}]
    gmax_stage = int(r["Gmax_full_t"]) if r["Gmax_full_t"] == r["Gmax_full_t"] else ""
    if not r["ga_eta"]:
        out.append({"quantity": "eta_3", "value": r["eta_T_over_dw"], "limit": 0.005, "stage": 3})
    if not r["ga_rmse"]:
        out.append({"quantity": "rmse_pos_T_over_eT0", "value": r["rmse_pos_T_over_eT0"], "limit": 0.05, "stage": 3})
    if not r["ga_tail"]:
        out.append({"quantity": "tail_mean_T_over_eT0", "value": r["tail_mean_T_over_eT0"], "limit": 0.02, "stage": 3})
    if not r["gate_G_F"]:
        out.append({"quantity": "Gmax_full", "value": r["Gmax_full_over_dw"], "limit": 0.01, "stage": gmax_stage})
    if not r["gate_G_N"]:
        out.append({"quantity": "|dev - final| of eta_3 or Gmax_full", "value": abs(r["dev_eta_T_over_dw"] - r["eta_T_over_dw"]),
                    "limit": 0.001, "stage": f"3 / {gmax_stage}"})
    return out


def last16_items(rows: Sequence[Mapping[str, Any]], failing: Sequence[Mapping[str, Any]]) -> List[Tuple[Dict[str, Any], int]]:
    """The ``(job, update)`` items of ``t3r2_audit.a8_one`` for the failing runs: the exports ``u01225`` ... ``u01600`` of the frozen attempt."""
    jobs = [{"run_dir": r["run_dir"], "q": int(r["q"]), "seed": int(r["seed"]), "run": r["run"]} for r in failing if r["exit_ok"] and r["eval_ok"]]
    return [(j, u) for j in jobs for u in au.LAST16]


def last16_tables(rows: Sequence[Mapping[str, Any]], failing: Sequence[Mapping[str, Any]], workers: int
                  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(last16, summary)``: the three G-A quantities at the last 16 stage-3 exports of every failing run (``t3r3_last16``'s evaluator;
    ``u01600`` must reproduce the evaluated value, else ``ValueError``), and per failing run and quantity the min / median / max, the number
    of exports above the limit, the rank of the final value from the top, and whether that quantity is one of the failing ones."""
    items = last16_items(rows, failing)
    if not items:
        return [], []
    if workers <= 1:
        res = [l16._one(it) for it in items]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(items)), mp_context=mp.get_context("spawn")) as pool:
            res = list(pool.map(l16._one, items))
    table = [{"run": j["run"], "q": j["q"], "seed": j["seed"], "update": u, "eta_3_over_dw": r["eta_3_over_dw"],
              "rmse_pos_T_over_eT0": r["rmse_pos_T_over_eT0"], "tail_mean_T_over_eT0": r["tail_mean_T_over_eT0"], "valid": r["valid"]}
             for (j, u), r in zip(items, res)]
    by_run = {r["run"]: r for r in failing}
    summary: List[Dict[str, Any]] = []
    for name, r in by_run.items():
        mine = [x for x in table if x["run"] == name]
        if not mine:
            continue
        final = next(x for x in mine if x["update"] == 1600)
        for flag, col, pcol, limit in FAIL_QUANTITIES:
            vals = [x[col] for x in mine]
            if abs(final[col] - float(r[pcol])) > 1e-9:
                raise ValueError(f"{name}: u01600 {col} {final[col]} differs from the evaluated {r[pcol]}")
            summary.append({"run": name, "quantity": col, "failing": bool(not r[flag]), "limit": limit, "final_value": float(r[pcol]),
                            "last16_min": min(vals), "last16_median": float(np.median(vals)), "last16_max": max(vals),
                            "n_above_limit": sum(1 for v in vals if v > limit), "final_rank_from_top": l16.rank_from_top(vals, final[col])})
    return table, summary


def formation_thresholds(dg2_runs: Path) -> Dict[int, float]:
    """The formation threshold per q of T3-R3's ``dg2_runs.csv`` (half the median final F of the passing runs of the q)."""
    out: Dict[int, float] = {}
    for r in dg.read_csv(Path(dg2_runs)):
        out[int(float(r["q"]))] = float(r["formation_threshold"])
    return out


def f_u1600(r: Mapping[str, Any]) -> float:
    """The formation measure F(u01600) of the frozen attempt (``export_summary``'s ``F`` on the final-tier grid of D_3)."""
    cfg = te.read_json(Path(r["run_dir"]) / "run_config.json")
    return float(r3a.stage3_export_summary(Path(r["run_dir"]) / "weights", cfg, 1600)["F"])


def suggested_class(r: Mapping[str, Any], f_u: float, thr: float, l16_rows: Sequence[Mapping[str, Any]]) -> str:
    """The class suggested by rule (see the module docstring)."""
    if r["S3_frozen"] == r["S3_frozen"] and r["S3_frozen"] > REGIME:
        return "K"
    if f_u == f_u and thr == thr and f_u <= thr:
        return "F"
    if any(x["failing"] and x["final_rank_from_top"] == 1 for x in l16_rows if x["run"] == r["run"]):
        return "V"
    return "U"


def failure_rows(failing: Sequence[Mapping[str, Any]], l16_summary: Sequence[Mapping[str, Any]], thr: Mapping[int, float]) -> List[Dict[str, Any]]:
    """One row per failing run: gates and values, the stage of the failing quantity, S3, the stretch, F(u01600), the suggested class."""
    out: List[Dict[str, Any]] = []
    for r in failing:
        fq = failing_quantities(r)
        ok_run = bool(r["exit_ok"] and r["eval_ok"])
        fu = f_u1600(r) if ok_run else NAN
        out.append({"run": r["run"], "q": r["q"], "seed": r["seed"], "failing_gates": r["failing_gates"] or r["fail_reason"],
                    "failing_quantities": "; ".join(f"{x['quantity']} = {x['value']:.6g} (limit {x['limit']}) at stage {x['stage']}" for x in fq),
                    "eta_3": r["eta_T_over_dw"], "rmse_pos": r["rmse_pos_T_over_eT0"], "tail_mean": r["tail_mean_T_over_eT0"],
                    "Gmax_full": r["Gmax_full_over_dw"], "Gmax_full_stage": r["Gmax_full_t"], "fired": bool(r["fired"]),
                    "S3_frozen": r["S3_frozen"], "S3_first": r["S3_first"], "stretch_nodes": r["stretch_nodes"], "has_stretch": bool(r["has_stretch"]),
                    "F_u1600": fu, "formation_threshold": thr.get(int(r["q"]), NAN),
                    "class_suggested": suggested_class(r, fu, thr.get(int(r["q"]), NAN), l16_summary) if ok_run else "E?"})
    return out


# ---------------------------------------------------------------------------------- reported items
def fired_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every trigger that fired, with the trigger values and the outcome of the run."""
    return [{"run": r["run"], "q": r["q"], "seed": r["seed"], "trigger_Delta": r["trig_Delta"], "s": r["trig_s"], "C": r["trig_C"],
             "reasons": r["trig_reasons"], "S3_first": r["S3_first"], "S3_frozen": r["S3_frozen"], "attempt2_final_Delta": r["a2_final_Delta"],
             "passed": bool(r["passed"]), "fail_reason": r["fail_reason"], "eta_3": r["eta_T_over_dw"], "Gmax_full": r["Gmax_full_over_dw"]}
            for r in rows if r.get("fired")]


def values_table(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per q and both: the worst ``Gmax_full`` and its run, and the medians of the reported metrics."""
    out: List[Dict[str, Any]] = []
    for q in (50, 60, "both"):
        sel = [r for r in rows if q == "both" or r["q"] == q]
        fin = [r for r in sel if r["Gmax_full_over_dw"] == r["Gmax_full_over_dw"]]
        worst = max(fin, key=lambda r: r["Gmax_full_over_dw"]) if fin else None
        row = {"q": q, "runs": len(sel), "pass": sum(1 for r in sel if r["passed"]),
               "worst_Gmax_full": worst["Gmax_full_over_dw"] if worst else NAN, "worst_Gmax_full_run": worst["run"] if worst else ""}
        for m in r3a.REPORT_METRICS:
            row[f"median_{m}"] = r3a.median_or_nan([r[m] for r in sel])
        out.append(row)
    return out


def cost_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Cost per q and in total: updates, policy rows, minibatch steps, development checks, wall time, and the share of fired runs; the
    ratios are those of the mean updates and policy rows to the 3,800 updates / 1,945,600 rows of a run that does not fire."""
    out: List[Dict[str, Any]] = []
    for q in (50, 60, "both"):
        sel = [r for r in rows if (q == "both" or r["q"] == q) and r.get("total_updates") is not None]
        n = len(sel)
        fired = [r for r in sel if r.get("fired")]
        quiet = [r for r in sel if not r.get("fired")]
        row = {"q": q, "runs": n, "fired": len(fired), "fired_share": len(fired) / n if n else NAN,
               "mean_updates": float(np.mean([r["total_updates"] for r in sel])) if n else NAN,
               "mean_policy_rows": float(np.mean([r["total_transitions"] for r in sel])) if n else NAN,
               "mean_minibatch_steps": float(np.mean([r["total_minibatch_steps"] for r in sel])) if n else NAN,
               "mean_dev_checks": float(np.mean([r["total_dev_checks"] for r in sel])) if n else NAN,
               "total_wall_sec": float(np.sum([r["wall_sec"] for r in sel])) if n else NAN,
               "mean_wall_sec": float(np.mean([r["wall_sec"] for r in sel])) if n else NAN,
               "median_wall_sec": r3a.median_or_nan([r["wall_sec"] for r in sel]),
               "mean_wall_sec_fired": float(np.mean([r["wall_sec"] for r in fired])) if fired else NAN,
               "mean_wall_sec_not_fired": float(np.mean([r["wall_sec"] for r in quiet])) if quiet else NAN,
               "mean_dev_verifier_sec": float(np.mean([r["dev_verifier_sec"] for r in sel])) if n else NAN}
        row["ratio_updates_to_not_fired"] = row["mean_updates"] / 3800.0 if n else NAN
        row["ratio_policy_rows_to_not_fired"] = row["mean_policy_rows"] / (3800 * 512) if n else NAN
        out.append(row)
    return out


def _pass_rows_from_table(path: Path, arm: Optional[str], kind: Optional[str], gates: Mapping[str, Any], recompute: bool) -> List[Dict[str, Any]]:
    """``[{q, seed, passed}]`` of a tracked per-run table. ``recompute``: apply the pass rule to the evaluation rows (T3-R1's table has no
    verdict column); otherwise read the ``passed`` column."""
    df = pd.read_csv(path)
    if arm is not None and "arm" in df:
        df = df[df["arm"] == arm]
    if kind is not None and "run_kind" in df:
        df = df[df["run_kind"] == kind]
    out = []
    for rec in df.to_dict("records"):
        passed = bool(ca.run_row(rec, gates)["passed"]) if recompute else ca.truthy(rec["passed"])
        out.append({"q": int(rec["q"]), "seed": int(rec["seed"]), "passed": passed})
    return out


def comparison_rows(rows: Sequence[Mapping[str, Any]], gates: Mapping[str, Any], t3r3: Path = T3R3_PER_RUN, t3r2: Path = T3R2_PER_RUN,
                    t3r1: Path = T3R1_PER_RUN) -> List[Dict[str, Any]]:
    """The descriptive comparisons of 3.3 (they decide nothing): this confirmation pooled with ``bwA``'s 40 development runs of T3-R3;
    the three blocks of T3-1.0's ``bw`` (T3-R1's development, T3-R2's confirmation, T3-R3's development control) and their pool; the
    pass rate of T3-1.0 implied by the first-attempt outcome (a fired run counted as a failure of its first attempt)."""
    mine = [{"q": r["q"], "seed": r["seed"], "passed": bool(r["passed"])} for r in rows]
    first = [{"q": r["q"], "seed": r["seed"], "passed": bool(r["passed"] and not r.get("fired"))} for r in rows]
    bwa3 = _pass_rows_from_table(t3r3, "bwA", "dev", gates, recompute=False)
    bw1 = _pass_rows_from_table(t3r1, "bw", None, gates, recompute=True)
    bw2 = _pass_rows_from_table(t3r2, None, None, gates, recompute=False)
    bw3 = _pass_rows_from_table(t3r3, "bw", "dev", gates, recompute=False)
    sets = [("t3r4_confirmation_bwA", mine), ("t3r3_development_bwA", bwa3), ("pooled_bwA_120", mine + bwa3),
            ("t3r1_development_bw", bw1), ("t3r2_confirmation_bw", bw2), ("t3r3_development_bw", bw3), ("pooled_bw_120", bw1 + bw2 + bw3),
            ("t3r4_first_attempt_bwA_equivalent", first)]
    out: List[Dict[str, Any]] = []
    for label, tab in sets:
        for q in (None,) + QS:
            sel = [t for t in tab if q is None or t["q"] == q]
            k, n = sum(1 for t in sel if t["passed"]), len(sel)
            lo, hi = r3a.cp(k, n)
            out.append({"set": label, "q": "all" if q is None else q, "n": n, "passes": k, "rate": k / n if n else NAN, "ci_lo": lo, "ci_hi": hi})
    return out


# ---------------------------------------------------------------------------------- the analysis
def analyse(root: Path, eval_csv: Path, ledger_path: Path, out_dir: Path, workers: int, checks_path: Optional[Path] = None,
            dg2_runs: Path = DG2_RUNS, criteria: Path = CRITERIA) -> Dict[str, Any]:
    """Run the analysis and write the outputs; returns ``headline``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gates = json.loads(Path(criteria).read_text())["gates"]
    ledger = ca.read_ledger(ledger_path)
    stopped = ca.stopped_runs(ledger)
    ev = {r["run"]: r for r in pd.read_csv(eval_csv).to_dict("records")}
    jobs = ck.all_jobs(root)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs))), mp_context=mp.get_context("spawn")) as pool:
        recs = list(pool.map(r3a.run_record, [(j,) for j in jobs]))
    rows = [r3a.merge_eval(rec, ev.get(rec["run"]), gates, stopped) for rec in recs]
    pd.DataFrame(rows).to_csv(out_dir / "per_run.csv", index=False)
    cnt = counts_of(rows)
    verdict = rule_verdict(cnt["fail_q50"], cnt["fail_q60"], stopped)
    failing = [r for r in rows if not r["passed"]]
    l16_table, l16_summary = last16_tables(rows, failing, workers)
    pd.DataFrame(l16_table).to_csv(out_dir / "last16.csv", index=False)
    pd.DataFrame(l16_summary).to_csv(out_dir / "last16_summary.csv", index=False)
    fails = failure_rows(failing, l16_summary, formation_thresholds(dg2_runs))
    pd.DataFrame(fails).to_csv(out_dir / "failures.csv", index=False)
    fired = fired_rows(rows)
    pd.DataFrame(fired).to_csv(out_dir / "fired.csv", index=False)
    pd.DataFrame(values_table(rows)).to_csv(out_dir / "counts.csv", index=False)
    cost = cost_rows(rows)
    pd.DataFrame(cost).to_csv(out_dir / "cost.csv", index=False)
    comp = comparison_rows(rows, gates)
    pd.DataFrame(comp).to_csv(out_dir / "comparisons.csv", index=False)
    n_fired = sum(1 for r in rows if r.get("fired"))
    first_pass = sum(1 for r in rows if r["passed"] and not r.get("fired"))
    lo, hi = r3a.cp(n_fired, len(rows))
    head = {"counts": cnt, "verdict": verdict,
            "trigger": {"fired": n_fired, "n": len(rows), "rate": n_fired / len(rows), "ci_lo": lo, "ci_hi": hi, "runs": [r["run"] for r in rows if r.get("fired")],
                        "first_attempt_pass": first_pass, "first_attempt_rate": first_pass / len(rows),
                        "fired_runs_that_pass": sum(1 for r in rows if r.get("fired") and r["passed"])},
            "failures": [{k: f[k] for k in ("run", "failing_gates", "class_suggested")} for f in fails],
            "stopped_without_verdict": stopped, "infra_events": ledger.get("events", []), "n_rows": len(rows)}
    if checks_path and Path(checks_path).is_file():
        chk = json.loads(Path(checks_path).read_text())
        head["checks"] = {"checks_ok": ck.checks_ok(chk), "n_runs": chk["n_runs"], "fired": chk.get("fired")}
    (out_dir / "headline.json").write_text(json.dumps(head, indent=1, default=str))
    return head


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring). Exit code 0 on any verdict (the verdict is in ``headline.json``); 3 if the checks file is
    given and not ok."""
    ap = argparse.ArgumentParser(description="T3-R4 confirmation analysis", allow_abbrev=False)
    ap.add_argument("--root", default=str(lt.DEFAULT_ROOT), help="results root, an absolute path")
    ap.add_argument("--eval", required=True, help="per_run.csv of t3r4_eval.py")
    ap.add_argument("--checks", default=None, help="launch_checks.json")
    ap.add_argument("--ledger", required=True, help="infra_events.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dg2-runs", default=str(DG2_RUNS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    root = lt.require_absolute(ap, "--root", a.root)
    head = analyse(root, Path(a.eval), Path(a.ledger), Path(a.out_dir), a.workers, Path(a.checks) if a.checks else None, Path(a.dg2_runs))
    print(json.dumps({k: head[k] for k in ("counts", "verdict", "trigger", "failures")}, indent=1, default=str))
    if "checks" in head and not head["checks"]["checks_ok"]:
        print("STOP: the post-launch checks are not ok", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
