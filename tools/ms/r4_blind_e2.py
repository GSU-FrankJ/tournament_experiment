#!/usr/bin/env python3
"""Independent recomputation of the E2 criterion table of MS-R4 (prompt section 3.3, "Blind recomputation").

Recomputes ``criterion_e2.csv`` from ``per_run.csv``, the offline curve file ``curves_all.csv`` and
``reports/ms/r4/prereg_criteria.json`` with its own code: it imports neither ``r4_e2_analysis`` nor ``r4_common`` nor any other
analysis module (only numpy / pandas / the standard library). The area of every run is recomputed from the 64 curve rows
(trapezoid of the final-tier eta_2/DW over the updates, divided by 1575), not read from the per-run table.

With ``--compare <criterion_e2.csv>`` the recomputed rows are compared with the analysis table: numbers to 1e-12 (absolute and
relative), booleans exactly, and the decision; the exit code is 0 only if they agree.

Usage:
    python tools/ms/r4_blind_e2.py --per-run results/ms_r4/analysis/e2/per_run.csv --curves results/ms_r4/curves/curves_all.csv \
        --criteria reports/ms/r4/prereg_criteria.json --out results/ms_r4/analysis/e2/criterion_e2_blind.csv \
        --compare results/ms_r4/analysis/e2/criterion_e2.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TOL = 1e-12
COL = "final_eta_T_over_dw"


def _truthy(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() == "true"
    try:
        return bool(v) and not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def area_of(curves: pd.DataFrame, arm: str, q: int, seed: int, crit: Dict[str, Any]) -> float:
    """``A`` of one run from its curve rows; NaN if the 64 checkpoints are not all present and finite."""
    a = crit["e2"]["area"]
    g = curves[(curves["arm"] == arm) & (curves["q"] == q) & (curves["seed"] == seed)].sort_values("update")
    want = np.arange(int(a["first_update"]), int(a["last_update"]) + 1, int(a["step"]))
    g = g[g["update"].isin(want)]                      # the fixed checkpoints; further exports of the curve file are ignored
    if len(g) != want.size or not np.array_equal(g["update"].to_numpy(dtype=int), want):
        return float("nan")
    y = g[COL].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        return float("nan")
    x = want.astype(float)
    return float(((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0).sum() / float(a["divisor"]))


def _ci_ratio(x: np.ndarray, r: np.ndarray, n_boot: int, seed: int, pcts: Sequence[Sequence[float]]) -> Tuple[float, List[List[float]]]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    rho_b = 1.0 - (np.take(x, idx).sum(axis=1) / x.size) / (np.take(r, idx).sum(axis=1) / x.size)
    return float(1.0 - x.mean() / r.mean()), [[float(np.percentile(rho_b, lo)), float(np.percentile(rho_b, hi))] for lo, hi in pcts]


def _ci_mean(d: np.ndarray, n_boot: int, seed: int, lo: float, hi: float) -> List[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    m = np.take(d, idx).sum(axis=1) / d.size
    return [float(np.percentile(m, lo)), float(np.percentile(m, hi))]


def recompute(per_run: pd.DataFrame, curves: pd.DataFrame, crit: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The E2 criterion rows (per arm and q) and the decision."""
    e2 = crit["e2"]
    ref = e2["arms"]["reference"]
    n_exp = int(crit["seeds"]["n_per_q"])
    n_boot, seed = int(crit["bootstrap"]["n_resamples"]), int(crit["bootstrap"]["seed_e2"])
    p95, p975 = crit["bootstrap"]["pct95"], crit["bootstrap"]["pct975"]
    rows: List[Dict[str, Any]] = []
    for role, xs in (("decision", e2["arms"]["x"]), ("descriptive", e2["arms"]["descriptive"])):
        for x in xs:
            met_q = []
            for q in crit["qs"]:
                sub = per_run[per_run["q"] == q]
                px = sub[sub["arm"] == x].drop_duplicates("seed").set_index("seed")
                pr = sub[sub["arm"] == ref].drop_duplicates("seed").set_index("seed")
                seeds = sorted(set(px.index) & set(pr.index))
                ax_ = np.array([area_of(curves, x, q, s, crit) if str(px.loc[s, "status"]) == "done" else np.nan for s in seeds])
                ar_ = np.array([area_of(curves, ref, q, s, crit) if str(pr.loc[s, "status"]) == "done" else np.nan for s in seeds])
                ok = np.isfinite(ax_) & np.isfinite(ar_)
                ax_, ar_ = ax_[ok], ar_[ok]
                n_pairs = int(ok.sum())
                if n_pairs:
                    rho, (c95, c975) = _ci_ratio(ax_, ar_, n_boot, seed, [p95, p975])
                else:
                    rho, c95, c975 = float("nan"), [float("nan")] * 2, [float("nan")] * 2
                a_pass = bool(n_pairs == n_exp and rho >= float(e2["reduction_min"]) and c95[0] > float(e2["reduction_ci_lower_above"]))
                viol = []
                for s in sorted(pr.index):
                    ref_pass = _truthy(pr.loc[s, "gates_all_pass"])
                    x_pass = bool(s in px.index and _truthy(px.loc[s, "gates_all_pass"]))
                    if ref_pass and not x_pass:
                        viol.append(int(s))
                gs = [s for s in seeds if np.isfinite(float(px.loc[s, "gmax_final"])) and np.isfinite(float(pr.loc[s, "gmax_final"]))]
                gd = np.array([float(px.loc[s, "gmax_final"]) - float(pr.loc[s, "gmax_final"]) for s in gs])
                glo, ghi = _ci_mean(gd, n_boot, seed, p95[0], p95[1]) if gd.size else (float("nan"), float("nan"))
                c_pass = bool(gd.size == n_exp and ghi <= float(e2["gmax_noninferiority_margin"]))
                met = bool(a_pass and not viol and c_pass)
                met_q.append(met)
                rows.append({"arm": x, "role": role, "q": q, "n_pairs": n_pairs,
                             "mean_A_x": float(ax_.mean()) if n_pairs else float("nan"),
                             "mean_A_ref": float(ar_.mean()) if n_pairs else float("nan"), "rho": rho,
                             "ci95_lo": c95[0], "ci95_hi": c95[1], "ci975_lo": c975[0], "ci975_hi": c975[1], "a_pass": a_pass,
                             "n_gate_violations": len(viol), "b_pass": not viol,
                             "mean_gmax_diff": float(gd.sum() / gd.size) if gd.size else float("nan"),
                             "gmax_ci_lo": glo, "gmax_ci_hi": ghi, "c_pass": c_pass, "met_q": met})
            for r in rows[-len(crit["qs"]):]:
                r["met_both"] = bool(len(met_q) == len(crit["qs"]) and all(met_q))
    out = pd.DataFrame(rows)
    rest = None
    for arm in e2["decision_order"]:
        sel = out[(out["arm"] == arm) & (out["role"] == "decision")]
        if len(sel) and bool(sel["met_both"].iloc[0]):
            rest = arm
            break
    return out, {"statement_rests_on": rest}


def cross_check_gates_json(per_run: pd.DataFrame) -> List[str]:
    """Re-read ``gates.json`` of every ``done`` run and compare the quantities E2-b and E2-c use (all four gate verdicts with the
    process-global RNG assertion, and Gmax_full/DW of the final tier at the end of stage 1) with the per-run table."""
    problems: List[str] = []
    for r in per_run[per_run["status"].astype(str) == "done"].itertuples(index=False):
        try:
            g = json.load(open(Path(r.run_dir) / "gates.json"))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{r.arm} q{r.q} s{r.seed}: gates.json unreadable ({exc})")
            continue
        passed = bool(all(g[n]["pass"] for n in ("G-A", "G-F", "G-N", "G-S")) and (g.get("global_rng") or {}).get("status") == "ok")
        if passed != _truthy(r.gates_all_pass):
            problems.append(f"{r.arm} q{r.q} s{r.seed}: all gates pass = {passed} in gates.json, {r.gates_all_pass} in the table")
        gm = float(g["metric_values"]["gmax_final"])
        if abs(gm - float(r.gmax_final)) > TOL:
            problems.append(f"{r.arm} q{r.q} s{r.seed}: Gmax_full/DW {gm!r} in gates.json, {r.gmax_final!r} in the table")
    return problems


def compare(blind: pd.DataFrame, decision: Dict[str, Any], analysis_csv: str, analysis_info: Optional[str]) -> List[str]:
    """Differences between the recomputed rows and the analysis table (empty list = agreement to 1e-12)."""
    ref = pd.read_csv(analysis_csv)
    diffs: List[str] = []
    for _, b in blind.iterrows():
        a = ref[(ref["arm"] == b["arm"]) & (ref["q"].astype(str) == str(b["q"])) & (ref["role"] == b["role"])]
        if a.empty:
            diffs.append(f"{b['arm']} q={b['q']}: missing in the analysis table")
            continue
        a = a.iloc[0]
        for col in blind.columns:
            if col in ("met_both", "arm", "role", "q") or col not in a.index:
                continue
            x, y = b[col], a[col]
            if isinstance(x, (bool, np.bool_)) or str(y) in ("True", "False"):
                if _truthy(x) != _truthy(y):
                    diffs.append(f"{b['arm']} q={b['q']} {col}: {x} (blind) against {y} (analysis)")
                continue
            fx, fy = float(x), float(y)
            if math.isnan(fx) and math.isnan(fy):
                continue
            if not (abs(fx - fy) <= TOL or abs(fx - fy) <= TOL * max(abs(fx), abs(fy))):
                diffs.append(f"{b['arm']} q={b['q']} {col}: {fx!r} (blind) against {fy!r} (analysis)")
        both = ref[(ref["arm"] == b["arm"]) & (ref["q"].astype(str) == "both") & (ref["role"] == b["role"])]
        if len(both) and _truthy(both["met_q"].iloc[0]) != bool(b["met_both"]):
            diffs.append(f"{b['arm']}: verdict {b['met_both']} (blind) against {both['met_q'].iloc[0]} (analysis)")
    if analysis_info:
        got = json.load(open(analysis_info))["decision"]["statement_rests_on"]
        if got != decision["statement_rests_on"]:
            diffs.append(f"decision: {decision['statement_rests_on']!r} (blind) against {got!r} (analysis)")
    return diffs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("--per-run", required=True)
    p.add_argument("--curves", required=True)
    p.add_argument("--criteria", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--compare", default=None)
    p.add_argument("--analysis-info", default=None, help="analysis_info.json (its decision is compared too)")
    p.add_argument("--check-gates-json", action="store_true", help="re-read gates.json of every done run (run_dir column)")
    a = p.parse_args(argv)
    crit = json.load(open(a.criteria))
    per_run = pd.read_csv(a.per_run)
    blind, dec = recompute(per_run, pd.read_csv(a.curves), crit)
    blind.to_csv(a.out, index=False)
    diffs = compare(blind, dec, a.compare, a.analysis_info) if a.compare else []
    problems = cross_check_gates_json(per_run) if a.check_gates_json else []
    print(json.dumps({"tool": "tools/ms/r4_blind_e2.py", "out": a.out, "compare": a.compare, "n_differences": len(diffs),
                      "differences": diffs[:20], "gates_json_problems": problems[:20], "decision": dec}, indent=1))
    return 0 if not diffs and not problems else 3


if __name__ == "__main__":
    raise SystemExit(main())
