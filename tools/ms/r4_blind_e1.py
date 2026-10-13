#!/usr/bin/env python3
"""Independent recomputation of the E1 criterion table of MS-R4 (prompt section 3.3, "Blind recomputation").

Recomputes ``criterion_e1.csv`` from ``per_run.csv`` and ``reports/ms/r4/prereg_criteria.json`` with its own code: it imports
neither ``r4_e1_analysis`` nor ``r4_common`` nor any other analysis module (only numpy / pandas / the standard library). With
``--check-gates-json`` the gate verdicts and the |peak error| of every ``done`` run are also re-read from the run's own
``gates.json`` (the ``run_dir`` column) and compared with the per-run columns.

With ``--compare <criterion_e1.csv>`` the recomputed table is compared with the analysis table: numbers to 1e-12 (absolute
and relative), booleans and the verdict exactly; the exit code is 0 only if they agree.

Usage:
    python tools/ms/r4_blind_e1.py --per-run results/ms_r4/analysis/e1/per_run.csv \
        --criteria reports/ms/r4/prereg_criteria.json --out results/ms_r4/analysis/e1/criterion_e1_blind.csv \
        --compare results/ms_r4/analysis/e1/criterion_e1.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

TOL = 1e-12
GATES = ("G_A", "G_F", "G_N", "G_S")


def _truthy(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() == "true"
    try:
        return bool(v) and not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _boot_ci(x: np.ndarray, n_boot: int, seed: int, lo_pct: float, hi_pct: float) -> List[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = np.take(x, idx).sum(axis=1) / x.size
    return [float(np.percentile(means, lo_pct)), float(np.percentile(means, hi_pct))]


def recompute(per_run: pd.DataFrame, crit: Dict[str, Any]) -> pd.DataFrame:
    """The E1 criterion rows (per q and the verdict) from the per-run table."""
    arms = crit["e1"]["arms"]
    n_exp = int(crit["seeds"]["n_per_q"])
    n_boot, seed = int(crit["bootstrap"]["n_resamples"]), int(crit["bootstrap"]["seed_e1"])
    lo_p, hi_p = crit["bootstrap"]["pct95"]
    rows: List[Dict[str, Any]] = []
    for q in crit["qs"]:
        sub = per_run[per_run["q"] == q]
        cand = sub[sub["arm"] == arms["candidate"]].drop_duplicates("seed").set_index("seed")
        ctrl = sub[sub["arm"] == arms["control"]].drop_duplicates("seed").set_index("seed")
        done = cand[cand["status"].astype(str) == "done"]
        peaks = done["peak_abs"].astype(float).to_numpy()
        mean_a = float(peaks.mean()) if peaks.size else float("nan")
        ctrl_done = ctrl[ctrl["status"].astype(str) == "done"]
        common = sorted(set(done.index) & set(ctrl_done.index))
        diff = np.array([float(done.loc[s, "peak_abs"]) - float(ctrl_done.loc[s, "peak_abs"]) for s in common])
        a_pass = bool(peaks.size == n_exp and np.isfinite(peaks).all() and mean_a <= float(crit["e1"]["peak_abs_mean_max"]))
        if diff.size:
            lo, hi = _boot_ci(diff, n_boot, seed, lo_p, hi_p)
        else:
            lo = hi = float("nan")
        b_pass = bool(diff.size == n_exp and np.isfinite(diff).all() and hi < float(crit["e1"]["peak_diff_ci_upper_below"]))
        missing = max(0, n_exp - len(cand))
        fails = {g: missing + sum(1 for s in cand.index if not _truthy(cand.loc[s, "gate_" + g])) for g in GATES}
        nonzero = missing + sum(1 for s in cand.index if _truthy(cand.loc[s, "nonzero_exit"]))
        any_fail = missing + sum(1 for s in cand.index
                                 if _truthy(cand.loc[s, "nonzero_exit"]) or not all(_truthy(cand.loc[s, "gate_" + g]) for g in GATES)
                                 or not _truthy(cand.loc[s, "gates_all_pass"]))
        c_pass = bool(len(cand) == n_exp and any_fail == 0 and nonzero == 0)
        rows.append({"q": q, "n_expected": n_exp, "n_runs_done": int(len(done)), "mean_abs_peak_lrsm": mean_a,
                     "a_threshold": float(crit["e1"]["peak_abs_mean_max"]), "a_pass": a_pass, "n_pairs": int(diff.size),
                     "mean_diff_vs_t1": float(diff.sum() / diff.size) if diff.size else float("nan"), "ci_lo": lo, "ci_hi": hi,
                     "b_pass": b_pass, "n_gate_fail": any_fail, **{f"n_{g}_fail": fails[g] for g in GATES},
                     "n_nonzero_exit": nonzero, "c_pass": c_pass, "met_q": bool(a_pass and b_pass and c_pass)})
    out = pd.DataFrame(rows)
    out["met_overall"] = bool(len(out) == len(crit["qs"]) and out["met_q"].all())
    return out


def cross_check_gates_json(per_run: pd.DataFrame, arm: str) -> List[str]:
    """Re-read ``gates.json`` of every ``done`` run of ``arm`` and compare the gate verdicts and |peak error| with the table."""
    problems: List[str] = []
    for r in per_run[(per_run["arm"] == arm) & (per_run["status"].astype(str) == "done")].itertuples(index=False):
        try:
            g = json.load(open(Path(r.run_dir) / "gates.json"))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"q{r.q} s{r.seed}: gates.json unreadable ({exc})")
            continue
        for name in GATES:
            want = bool(g[name.replace("_", "-")]["pass"])
            if want != _truthy(getattr(r, "gate_" + name)):
                problems.append(f"q{r.q} s{r.seed}: {name} {want} in gates.json, {getattr(r, 'gate_' + name)} in the table")
        pk = float(g["reported"]["end_of_stage2"]["final"]["stage2_peak_rel_err_abs"])
        if abs(pk - float(r.peak_abs)) > TOL:
            problems.append(f"q{r.q} s{r.seed}: |peak| {pk!r} in gates.json, {r.peak_abs!r} in the table")
    return problems


def compare(blind: pd.DataFrame, analysis_csv: Path) -> List[str]:
    """Differences between the recomputed table and the analysis table (empty list = agreement to 1e-12)."""
    ref = pd.read_csv(analysis_csv)
    ref = ref[ref["q"].astype(str) != "both"].copy()
    ref["q"] = ref["q"].astype(int)
    diffs: List[str] = []
    if len(ref) != len(blind):
        return [f"{len(ref)} analysis rows against {len(blind)} recomputed rows"]
    for _, b in blind.iterrows():
        a = ref[ref["q"] == b["q"]]
        if a.empty:
            diffs.append(f"q={b['q']}: missing in the analysis table")
            continue
        a = a.iloc[0]
        for col in blind.columns:
            if col in ("met_overall",) or col not in a.index:
                continue
            x, y = b[col], a[col]
            if isinstance(x, (bool, np.bool_)) or str(y) in ("True", "False"):
                if _truthy(x) != _truthy(y):
                    diffs.append(f"q={b['q']} {col}: {x} (blind) against {y} (analysis)")
                continue
            fx, fy = float(x), float(y)
            if math.isnan(fx) and math.isnan(fy):
                continue
            if not (abs(fx - fy) <= TOL or abs(fx - fy) <= TOL * max(abs(fx), abs(fy))):
                diffs.append(f"q={b['q']} {col}: {fx!r} (blind) against {fy!r} (analysis)")
    full = pd.read_csv(analysis_csv)
    both = full[full["q"].astype(str) == "both"]
    if len(both) and _truthy(both["met_q"].iloc[0]) != bool(blind["met_overall"].iloc[0]):
        diffs.append(f"verdict: {bool(blind['met_overall'].iloc[0])} (blind) against {both['met_q'].iloc[0]} (analysis)")
    return diffs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("--per-run", required=True)
    p.add_argument("--criteria", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--compare", default=None)
    p.add_argument("--check-gates-json", action="store_true")
    a = p.parse_args(argv)
    crit = json.load(open(a.criteria))
    per_run = pd.read_csv(a.per_run)
    blind = recompute(per_run, crit)
    blind.to_csv(a.out, index=False)
    problems: List[str] = []
    if a.check_gates_json:
        problems += cross_check_gates_json(per_run, crit["e1"]["arms"]["candidate"])
    diffs = compare(blind, Path(a.compare)) if a.compare else []
    report = {"tool": "tools/ms/r4_blind_e1.py", "out": a.out, "compare": a.compare, "n_differences": len(diffs),
              "differences": diffs[:20], "gates_json_problems": problems[:20], "met_overall": bool(blind["met_overall"].iloc[0])}
    print(json.dumps(report, indent=1))
    return 0 if not diffs and not problems else 3


if __name__ == "__main__":
    raise SystemExit(main())
