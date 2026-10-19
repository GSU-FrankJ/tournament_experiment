#!/usr/bin/env python3
"""Verdict and reported items of the T3-R2 confirmation (D4), from the evaluation table ``t3c_eval.py`` writes.

Decides (pure functions, :func:`run_row`, :func:`counts`, :func:`verdict`):

  * a run PASSES iff it exited with code 0 (status ``done``), its process-global RNG assertion is ok, it has a valid evaluation
    (no error, ``valid`` and ``dev_valid`` true) and it passes G-A and G-F and G-N (T3-R1's thresholds, inclusive ``<=``,
    full float64 precision; recomputed here from the metric columns);
  * the confirmation PASSES iff at least 36 of the 40 runs pass, at least 17 of 20 at q = 50 pass and at least 17 of 20 at
    q = 60 pass; every planned run is counted (a run without a row, or with a failed status, is a failed run);
  * if ``infra_events.json`` lists a run in ``stopped_without_verdict`` (its repeat also failed for an infrastructure reason),
    there is NO verdict and that run is counted neither way.

Reports (deciding nothing): pass counts per q with exact Clopper-Pearson 95 % intervals, the per-failure diagnosis
(``--diagnose``: failing gate and part, the maximum's stage and gap, the learned and the best-response effort, and the A7
analysis of ``t3r2_audit``), the near-zero stage-2 stretches of all 40 runs, the pass rate pooled with T3-R1's 40 ``bw`` runs
(descriptive) and the gate values set against T3-R1's.

Usage:
    python tools/t3/t3c_analysis.py --per-run results/t3_r2/confirmation/eval/per_run.csv \
        --ledger results/t3_r2/confirmation/infra_events.json --out-dir results/t3_r2/confirmation/analysis [--diagnose]
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

import t3_metrics as tm  # noqa: E402

QS = (50, 60)
SEEDS = tuple(range(60501, 60521))
MIN_TOTAL, MIN_PER_Q = 36, 17
T3R1_PER_RUN = ROOT / "results" / "t3_r1" / "eval" / "per_run.csv"
NAN = float("nan")


def truthy(v: Any) -> bool:
    """A CSV / Python truth value (``True``, ``"True"``, ``1``, ``1.0``)."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "1.0")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return bool(v)


def fnum(v: Any) -> float:
    """A cell as a float (NaN if empty or not a number)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def le(x: float, lim: float) -> bool:
    """``x <= lim`` (inclusive, float64); False for NaN."""
    return bool(x == x and x <= lim)


def run_row(r: Optional[Mapping[str, Any]], gates: Mapping[str, Any]) -> Dict[str, Any]:
    """Pass / fail of one planned run from its evaluation row (``None``: no row).

    Returns:
        ``exit_ok``, ``rng_ok``, ``eval_ok`` (no error, both tiers valid), the gate parts and verdicts, ``passed`` and
        ``fail_reason`` (empty for a pass; the first failing condition otherwise).
    """
    ga, gf, gn = gates["G-A"], gates["G-F"], gates["G-N"]
    out: Dict[str, Any] = {"exit_ok": False, "rng_ok": False, "eval_ok": False, "ga_eta": False, "ga_rmse": False,
                           "ga_tail": False, "gate_G_A": False, "gate_G_F": False, "gn_eta": False, "gn_gmax": False,
                           "gate_G_N": False, "passed": False, "fail_reason": "no row"}
    if r is None:
        return out
    out["exit_ok"] = bool(str(r.get("status")) == "done" and fnum(r.get("exit_code")) == 0.0)
    out["rng_ok"] = truthy(r.get("global_rng_ok"))
    err_raw = r.get("eval_error")
    err = "" if err_raw is None or (isinstance(err_raw, float) and math.isnan(err_raw)) else str(err_raw)     # an empty CSV cell is NaN
    out["eval_ok"] = bool(not err and truthy(r.get("valid")) and truthy(r.get("dev_valid")))
    if out["eval_ok"]:
        eta, rm, tl = fnum(r["eta_T_over_dw"]), fnum(r["rmse_pos_T_over_eT0"]), fnum(r["tail_mean_T_over_eT0"])
        gm, deta, dgm = fnum(r["Gmax_full_over_dw"]), fnum(r["dev_eta_T_over_dw"]), fnum(r["dev_Gmax_full_over_dw"])
        out["ga_eta"] = le(eta, ga["eta_T_over_dw"])
        out["ga_rmse"] = le(rm, ga["rmse_pos_T_over_eT0"])
        out["ga_tail"] = le(tl, ga["tail_mean_T_over_eT0"])
        out["gate_G_A"] = out["ga_eta"] and out["ga_rmse"] and out["ga_tail"]
        out["gate_G_F"] = le(gm, gf["Gmax_full_over_dw"])
        out["gn_eta"] = le(abs(deta - eta), gn["eta_T_over_dw_dev_minus_final_abs"])
        out["gn_gmax"] = le(abs(dgm - gm), gn["Gmax_full_over_dw_dev_minus_final_abs"])
        out["gate_G_N"] = out["gn_eta"] and out["gn_gmax"]
    if not out["exit_ok"]:
        reason = f"exit: status {r.get('status')!r}, exit_code {r.get('exit_code')!r}"
    elif not out["rng_ok"]:
        reason = "global-RNG violation"
    elif not out["eval_ok"]:
        reason = f"evaluation: error {err!r}, valid {r.get('valid')!r}, dev_valid {r.get('dev_valid')!r}"
    else:
        failed = [g for g in ("gate_G_A", "gate_G_F", "gate_G_N") if not out[g]]
        reason = "; ".join(failed)
    out["passed"] = bool(out["exit_ok"] and out["rng_ok"] and out["eval_ok"] and out["gate_G_A"] and out["gate_G_F"]
                         and out["gate_G_N"])
    out["fail_reason"] = "" if out["passed"] else reason
    return out


def clopper_pearson(k: int, n: int) -> Tuple[float, float]:
    """Exact two-sided 95 % interval of ``k / n``."""
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def counts(table: Sequence[Mapping[str, Any]], excluded: Sequence[str] = ()) -> Dict[str, Any]:
    """Pass counts per q and in total over the planned runs (runs in ``excluded`` are counted neither way)."""
    out: Dict[str, Any] = {}
    for q in QS:
        sel = [t for t in table if int(t["q"]) == q and t["run"] not in excluded]
        k = sum(1 for t in sel if t["passed"])
        lo, hi = clopper_pearson(k, len(sel)) if sel else (NAN, NAN)
        out[f"n_planned_q{q}"], out[f"n_counted_q{q}"], out[f"pass_q{q}"] = len([t for t in table if int(t["q"]) == q]), len(sel), k
        out[f"ci_lo_q{q}"], out[f"ci_hi_q{q}"] = lo, hi
    k_all = out["pass_q50"] + out["pass_q60"]
    n_all = out["n_counted_q50"] + out["n_counted_q60"]
    lo, hi = clopper_pearson(k_all, n_all) if n_all else (NAN, NAN)
    out.update(pass_total=k_all, n_counted_total=n_all, n_planned_total=out["n_planned_q50"] + out["n_planned_q60"],
               ci_lo_total=lo, ci_hi_total=hi)
    return out


def verdict(cnt: Mapping[str, Any], stopped: Sequence[str]) -> Dict[str, Any]:
    """The verdict under the confirmation rule.

    Args:
        cnt: :func:`counts`.
        stopped: Runs the ledger lists in ``stopped_without_verdict``.

    Returns:
        ``{"verdict": "pass" | "fail" | "no_verdict", ...}`` with the three conditions of the rule.
    """
    c_total = bool(cnt["pass_total"] >= MIN_TOTAL)
    c50, c60 = bool(cnt["pass_q50"] >= MIN_PER_Q), bool(cnt["pass_q60"] >= MIN_PER_Q)
    if stopped:
        v, why = "no_verdict", f"the confirmation stopped without a verdict: infrastructure failure twice in {list(stopped)}"
    elif c_total and c50 and c60:
        v, why = "pass", "at least 36 of 40 pass and at least 17 of 20 pass at each q"
    else:
        miss = [n for n, ok in (("total >= 36", c_total), ("q=50 >= 17", c50), ("q=60 >= 17", c60)) if not ok]
        v, why = "fail", "not met: " + ", ".join(miss)
    return {"verdict": v, "reason": why, "total_condition": c_total, "q50_condition": c50, "q60_condition": c60,
            "pass_total": int(cnt["pass_total"]), "pass_q50": int(cnt["pass_q50"]), "pass_q60": int(cnt["pass_q60"]),
            "rule": {"min_total": MIN_TOTAL, "min_per_q": MIN_PER_Q}, "stopped_without_verdict": list(stopped)}


def stopped_runs(ledger: Mapping[str, Any]) -> List[str]:
    """Runs whose repeat also failed for an infrastructure reason: those the ledger lists under ``stopped_without_verdict`` and
    those with a recorded event of attempt 2 or later (an operator may record the events by hand)."""
    names = list(ledger.get("stopped_without_verdict", []))
    for e in ledger.get("events", []):
        if int(e.get("attempt", 1)) >= 2 and e["run"] not in names:
            names.append(e["run"])
    return names


def read_ledger(path: Path) -> Dict[str, Any]:
    """``infra_events.json`` (refuses a missing file: the ledger must exist, empty if there was no event)."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path}: the infrastructure ledger must exist (an empty ledger if there was no event)")
    return json.loads(Path(path).read_text())


def planned_table(per_run: pd.DataFrame, gates: Mapping[str, Any], seeds: Sequence[int] = SEEDS,
                  prefix: str = "t3r2", arm: str = "bw") -> List[Dict[str, Any]]:
    """One entry per planned run (q-major, seed order) with its row's verdict and the gate values."""
    rows = {(int(r["q"]), int(r["seed"])): r for r in per_run.to_dict("records") if r.get("arm", arm) == arm}
    out = []
    for q in QS:
        for s in seeds:
            r = rows.get((q, s))
            e = run_row(r, gates)
            e.update(q=q, seed=s, run=f"{prefix}_q{q}_s{s}_{arm}")
            for c in ("eta_T_over_dw", "rmse_pos_T_over_eT0", "tail_mean_T_over_eT0", "Gmax_full_over_dw", "dev_eta_T_over_dw",
                      "dev_Gmax_full_over_dw", "EXP_root_over_dw", "Gmax_full_t", "Gmax_full_d"):
                e[c] = fnum(r.get(c)) if r is not None else NAN
            e["gn_eta_diff"] = abs(e["dev_eta_T_over_dw"] - e["eta_T_over_dw"])
            e["gn_gmax_diff"] = abs(e["dev_Gmax_full_over_dw"] - e["Gmax_full_over_dw"])
            out.append(e)
    return out


def stretches_of(arrays: Mapping[str, np.ndarray], q: float) -> Dict[str, Any]:
    """In-band stage-2 final-tier nodes with learned mean effort below 1 (the stretches of A7)."""
    d, e = np.asarray(arrays["fin_t2_d_grid"]), np.asarray(arrays["fin_t2_e_hat"])
    m = (e < 1.0) & (np.abs(d) < 2.0 * q - 1e-9)
    idx = np.where(m)[0]
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if idx.size else []
    return {"nodes": int(m.sum()), "nodes_in_band": int((np.abs(d) < 2.0 * q - 1e-9).sum()),
            "stretches": "; ".join(f"[{d[g[0]]:+.0f}, {d[g[-1]]:+.0f}]" for g in groups)}


GATE_VALUE_COLS = ("eta_T_over_dw", "rmse_pos_T_over_eT0", "tail_mean_T_over_eT0", "Gmax_full_over_dw", "gn_eta_diff", "gn_gmax_diff")


def gate_values_table(new: Sequence[Mapping[str, Any]], old: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Median, maximum and the number of runs above the limit of every gate value, per q, new against T3-R1."""
    lim = {"eta_T_over_dw": gates["G-A"]["eta_T_over_dw"], "rmse_pos_T_over_eT0": gates["G-A"]["rmse_pos_T_over_eT0"],
           "tail_mean_T_over_eT0": gates["G-A"]["tail_mean_T_over_eT0"], "Gmax_full_over_dw": gates["G-F"]["Gmax_full_over_dw"],
           "gn_eta_diff": gates["G-N"]["eta_T_over_dw_dev_minus_final_abs"],
           "gn_gmax_diff": gates["G-N"]["Gmax_full_over_dw_dev_minus_final_abs"]}
    rows = []
    for label, tab in (("t3r2_confirmation", new), ("t3r1_bw", old)):
        for q in QS:
            sel = [t for t in tab if int(t["q"]) == q]
            for c in GATE_VALUE_COLS:
                v = np.array([t[c] for t in sel], dtype=float)
                v = v[np.isfinite(v)]
                rows.append({"set": label, "q": q, "quantity": c, "limit": lim[c], "n": int(v.size),
                             "median": float(np.median(v)) if v.size else NAN, "max": float(v.max()) if v.size else NAN,
                             "n_above_limit": int((v > lim[c]).sum())})
    return rows


def pooled_table(new: Sequence[Mapping[str, Any]], old: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Pass counts of the confirmation, T3-R1's 40 ``bw`` runs and both pooled, with exact intervals (descriptive)."""
    rows = []
    for label, tabs in (("t3r2_confirmation", [new]), ("t3r1_bw_development", [old]), ("pooled_80_descriptive", [new, old])):
        for q in (None,) + QS:
            sel = [t for tab in tabs for t in tab if q is None or int(t["q"]) == q]
            k, n = sum(1 for t in sel if t["passed"]), len(sel)
            lo, hi = clopper_pearson(k, n)
            rows.append({"set": label, "q": "all" if q is None else q, "n": n, "passes": k, "rate": k / n, "ci_lo": lo, "ci_hi": hi})
    return rows


def diagnose(root: Path, table: Sequence[Mapping[str, Any]], out_dir: Path, workers: int) -> Dict[str, Any]:
    """Per-failure diagnosis (T3-R1 section 4.1 form) and the A7 analysis of the failing runs; the stretches of all 40 runs.

    Reads the run directories under ``root`` (weights, ``ms_updates.csv``) and re-evaluates the final tier (pure).
    """
    import t3_eval as te
    import t3r2_audit as au
    from utils.dp_br_verifier import FINAL_CONFIG

    rows, stretch_rows, a7_items = [], [], []
    for t in table:
        d = Path(root) / f"q{t['q']}" / f"seed{t['seed']}" / "bw"
        if not (d / "state_end_stage1.pt").is_file():
            continue
        cfg = te.read_json(d / "run_config.json")
        cand = te.load_bw(d, cfg)
        dw = float(cand.spec.dw)
        ev = te.evaluate_tier(cand, FINAL_CONFIG)
        stretch_rows.append({"q": t["q"], "seed": t["seed"], "passed": t["passed"],
                             **stretches_of({f"fin_t2_{k}": ev.arrays[f"v_t2_{k}"] for k in ("d_grid", "e_hat")}, t["q"])})
        if t["passed"]:
            continue
        parts = [(n, f) for n, f in (("G-A eta_3", "ga_eta"), ("G-A rmse_pos", "ga_rmse"), ("G-A tail_mean", "ga_tail"),
                                     ("G-F Gmax_full", "gate_G_F"), ("G-N", "gate_G_N")) if not t[f]]
        rows.append({"q": t["q"], "seed": t["seed"], "failing": "all failing parts: " + (", ".join(n for n, _ in parts) or t["fail_reason"]),
                     "eta_3": t["eta_T_over_dw"], "rmse_pos": t["rmse_pos_T_over_eT0"], "tail_mean": t["tail_mean_T_over_eT0"],
                     "Gmax_full": t["Gmax_full_over_dw"], "gn_eta_diff": t["gn_eta_diff"], "gn_gmax_diff": t["gn_gmax_diff"]})
        for name, flag, stage in (("G-A eta_3", "ga_eta", 3), ("G-F Gmax_full", "gate_G_F", None)):
            if t[flag]:
                continue
            if stage is None:
                stage, dd = int(t["Gmax_full_t"]), float(t["Gmax_full_d"])
            else:
                dd = float(ev.arrays[f"v_t{stage}_d_grid"][int(np.argmax(ev.arrays[f"v_t{stage}_delta"]))])
            g = ev.arrays[f"v_t{stage}_d_grid"]
            i = int(np.argmin(np.abs(g - dd)))
            rows.append({"q": t["q"], "seed": t["seed"], "failing": name, "stage": stage, "d": dd,
                         "Delta_over_dw": float(ev.arrays[f"v_t{stage}_delta"][i]) / dw,
                         "G_over_dw": float(ev.arrays[f"v_t{stage}_G"][i]) / dw,
                         "learned_effort": float(ev.arrays[f"v_t{stage}_e_hat"][i]),
                         "best_response_effort": float(ev.arrays[f"v_t{stage}_a_br"][i]),
                         "one_step_argmax_effort": float(ev.arrays[f"v_t{stage}_a_dev"][i])})
        if all((d / "weights" / f"u{u:05d}.npz").is_file() for u in (1625, 3200)):     # the A7 exports of a full-length run
            a7_items.append(({"arm": "bw", "q": t["q"], "seed": t["seed"], "T": 3, "run": t["run"], "run_dir": str(d),
                              "kind": "bw", "arrays_dir": None}, None, "failure"))
    a7 = au.parallel(au.a7_run, a7_items, min(workers, max(1, len(a7_items)))) if a7_items else []
    out_dir.mkdir(parents=True, exist_ok=True)
    if a7:
        au.write_json(out_dir / "a7_failures.json", [{k: v for k, v in r.items() if k != "analysis"} | {
            "summary": r["analysis"]["summary"], "formation": r["analysis"]["formation"],
            "objective": r["analysis"]["objective"], "nodes": r["analysis"]["nodes"]} for r in a7])
    pd.DataFrame(rows).to_csv(out_dir / "failures.csv", index=False)
    pd.DataFrame(stretch_rows).to_csv(out_dir / "stretches.csv", index=False)
    return {"failures": rows, "stretches": stretch_rows, "n_a7": len(a7)}


def analyse(per_run_path: Path, ledger_path: Path, out_dir: Path, t3r1_per_run: Path = T3R1_PER_RUN,
            root: Optional[Path] = None, do_diagnose: bool = False, workers: int = 4) -> Dict[str, Any]:
    """The whole analysis; writes the CSV / JSON files into ``out_dir`` and returns the verdict record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    crit, crit_sha = tm.load_criteria()
    gates = crit["gates"]
    ledger = read_ledger(ledger_path)
    per_run = pd.read_csv(per_run_path, float_precision="round_trip")
    table = planned_table(per_run, gates)
    stopped = stopped_runs(ledger)
    cnt = counts(table, excluded=stopped)
    vd = verdict(cnt, stopped)
    old = pd.read_csv(t3r1_per_run, float_precision="round_trip")
    old_table = planned_table(old[old["arm"] == "bw"], gates, seeds=tuple(range(50501, 50521)), prefix="t3r1")
    pd.DataFrame(table).to_csv(out_dir / "per_run_table.csv", index=False)
    pd.DataFrame([{k: v for k, v in cnt.items()}]).to_csv(out_dir / "pass_counts.csv", index=False)
    record = {"verdict": vd, "counts": cnt, "criteria_sha256": crit_sha, "n_infra_events": len(ledger.get("events", [])),
              "infra_events": ledger.get("events", []), "per_run": str(per_run_path)}
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(record, f, indent=1)
    pd.DataFrame(gate_values_table(table, old_table, gates)).to_csv(out_dir / "gate_values_vs_t3r1.csv", index=False)
    pd.DataFrame(pooled_table(table, old_table)).to_csv(out_dir / "pooled_descriptive.csv", index=False)
    if do_diagnose and root is not None:
        record["diagnosis"] = {k: len(v) if isinstance(v, list) else v for k, v in diagnose(root, table, out_dir, workers).items()}
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R2 confirmation analysis", allow_abbrev=False)
    ap.add_argument("--per-run", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--t3r1-per-run", default=str(T3R1_PER_RUN))
    ap.add_argument("--root", default=None, help="the confirmation root (for --diagnose)")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    rec = analyse(Path(a.per_run), Path(a.ledger), Path(a.out_dir), Path(a.t3r1_per_run),
                  Path(a.root) if a.root else None, a.diagnose, a.workers)
    print(json.dumps(rec["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
