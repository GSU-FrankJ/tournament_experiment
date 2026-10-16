#!/usr/bin/env python3
"""Independent recomputation of the pre-registered comparison of round T3-R1 (prompt section 3.3, "Blind recomputation").

Reads only ``per_run.csv`` (``tools/t3/t3_eval.py``) and ``reports/t3/r1/prereg_criteria.json`` and recomputes the four
hypothesis rows (H1, H2, H3, H1-T2 per q: complete pairs, means, 95 % and 98.75 % percentile intervals of the mean paired
difference, median with its interval, sign count) and the gate counts with exact Clopper-Pearson intervals. The gate
verdicts are recomputed from the metric columns and the thresholds of the criteria file. Nothing is imported from
``t3_analysis.py``, ``t3_metrics.py`` or ``tools/ms/r4_common.py``; the statistics are written out here from D7.

``--analysis-dir`` additionally compares the result with ``criterion.csv`` and ``gate_counts.csv`` of the analysis tool and
reports the largest absolute difference (they must agree to 1e-12).

Usage:
    python tools/t3/t3_blind.py --per-run results/t3_r1/eval/per_run.csv --out-dir results/t3_r1/analysis/blind \
        [--analysis-dir results/t3_r1/analysis]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
CRITERIA = ROOT / "reports" / "t3" / "r1" / "prereg_criteria.json"
SEEDS = {3: list(range(50501, 50521)), 2: list(range(30501, 30521))}      # D1: wave t3 and wave t2j
HORIZON = {"bw": 3, "jt": 3, "jt_es": 3, "v20": 2, "jt2": 2}
QS = (50, 60)
TOL = 1e-12


def num(x: Any) -> float:
    """A CSV cell as a float (empty / missing: NaN)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_table(path: Path) -> Dict[Tuple[str, int, int], Dict[str, str]]:
    """``{(arm, q, seed): row}`` of the per-run table."""
    out: Dict[Tuple[str, int, int], Dict[str, str]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["arm"], int(float(r["q"])), int(float(r["seed"])))] = r
    return out


def exit_ok(r: Optional[Dict[str, str]]) -> bool:
    """Status ``done`` and exit code 0."""
    return bool(r is not None and r.get("status") == "done" and num(r.get("exit_code")) == 0.0)


def values(tab: Dict[Tuple[str, int, int], Dict[str, str]], arm: str, q: int, col: str) -> np.ndarray:
    """The column for the planned seeds of (arm, q) in seed order; NaN for a missing or failed run."""
    out = []
    for s in SEEDS[HORIZON[arm]]:
        r = tab.get((arm, q, s))
        out.append(num(r.get(col)) if exit_ok(r) else float("nan"))
    return np.array(out)


def pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Differences a - b of the complete pairs (both finite)."""
    ok = np.isfinite(a) & np.isfinite(b)
    return (a - b)[ok]


def boot_means(d: np.ndarray, S: int, n_boot: int, stat: Any = np.mean) -> np.ndarray:
    """The statistic of ``n_boot`` resamples with indices ``default_rng(S).integers(0, n, size=(n_boot, n))``."""
    idx = np.random.default_rng(S).integers(0, d.size, size=(n_boot, d.size))
    return stat(d[idx], axis=1)


def criterion_rows(tab: Dict[Tuple[str, int, int], Dict[str, str]], crit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The hypothesis rows of D7 on the primary statistic, recomputed."""
    st = crit["statistics"]
    n_boot = int(st["n_boot"])
    stat_col = crit["primary"]["statistic"]
    rows = []
    for h, spec in crit["hypotheses"].items():
        S = int(st["bootstrap_seed"][str(spec["horizon"])])
        for q in QS:
            a = values(tab, spec["a"], q, stat_col)
            b = values(tab, spec["b"], q, stat_col)
            ok = np.isfinite(a) & np.isfinite(b)
            d = (a - b)[ok]
            r: Dict[str, Any] = {"hypothesis": h, "q": q, "n_pairs": int(d.size), "dropped_pairs": int(len(a) - d.size)}
            if d.size:
                m = boot_means(d, S, n_boot)
                lo, hi = np.percentile(m, st["percentiles"]["ci95"])
                lo2, hi2 = np.percentile(m, st["percentiles"]["ci98_75"])
                md = boot_means(d, S, n_boot, np.median)
                mlo, mhi = np.percentile(md, st["percentiles"]["ci95"])
                r.update({"mean_a": float(a[ok].mean()), "mean_b": float(b[ok].mean()), "mean_diff": float(d.mean()),
                          "ci95_lo": float(lo), "ci95_hi": float(hi), "ci98_75_lo": float(lo2), "ci98_75_hi": float(hi2),
                          "median_diff": float(np.median(d)), "median_ci95_lo": float(mlo), "median_ci95_hi": float(mhi),
                          "n_neg": int((d < 0).sum())})
            else:
                r.update({k: float("nan") for k in ("mean_a", "mean_b", "mean_diff", "ci95_lo", "ci95_hi", "ci98_75_lo",
                                                    "ci98_75_hi", "median_diff", "median_ci95_lo", "median_ci95_hi")})
                r["n_neg"] = 0
            r["ci_hi_below_0"] = bool(r["ci95_hi"] < 0.0)
            rows.append(r)
    for h in {r["hypothesis"] for r in rows}:
        sel = [r for r in rows if r["hypothesis"] == h]
        for r in sel:
            r["hypothesis_supported"] = all(x["ci_hi_below_0"] for x in sel)
    return rows


def clopper_pearson(k: int, n: int) -> Tuple[float, float]:
    """Exact two-sided 95 % interval of k / n (``scipy.stats.binomtest(...).proportion_ci(method='exact')``)."""
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def gate_verdicts(r: Dict[str, str], T: int, crit: Dict[str, Any]) -> Dict[str, bool]:
    """G-A, G-F, G-N (and G-S at T = 2) of one run from its metric columns and the thresholds (inclusive <=)."""
    g = crit["gates"]
    ga = (num(r["eta_T_over_dw"]) <= g["G-A"]["eta_T_over_dw"] and num(r["rmse_pos_T_over_eT0"]) <= g["G-A"]["rmse_pos_T_over_eT0"]
          and num(r["tail_mean_T_over_eT0"]) <= g["G-A"]["tail_mean_T_over_eT0"])
    gf = num(r["Gmax_full_over_dw"]) <= g["G-F"]["Gmax_full_over_dw"]
    gn = (abs(num(r["dev_eta_T_over_dw"]) - num(r["eta_T_over_dw"])) <= g["G-N"]["eta_T_over_dw_dev_minus_final_abs"]
          and abs(num(r["dev_Gmax_full_over_dw"]) - num(r["Gmax_full_over_dw"])) <= g["G-N"]["Gmax_full_over_dw_dev_minus_final_abs"])
    out = {"G-A": bool(ga), "G-F": bool(gf), "G-N": bool(gn)}
    if T == g["G-S"]["applies_at_horizon"]:
        out["G-S"] = bool(num(r["stage1_rel_err_abs"]) <= g["G-S"]["stage1_rel_err_abs"])
    return out


def gate_count_rows(tab: Dict[Tuple[str, int, int], Dict[str, str]], crit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per (arm, q): the gate passes among the planned runs, with Clopper-Pearson intervals."""
    rows = []
    for arm in ("bw", "jt", "jt_es", "v20", "jt2"):
        T = HORIZON[arm]
        for q in QS:
            runs = [tab.get((arm, q, s)) for s in SEEDS[T]]
            if all(r is None for r in runs):
                continue
            n = len(SEEDS[T])
            r_out: Dict[str, Any] = {"arm": arm, "q": q, "n_planned": n, "n_exit_ok": sum(1 for r in runs if exit_ok(r))}
            names = ["G-A", "G-F", "G-N"] + (["G-S"] if T == 2 else [])
            ver = [(r, gate_verdicts(r, T, crit)) for r in runs if exit_ok(r)]
            for g in names:
                k = sum(1 for _, v in ver if v[g])
                lo, hi = clopper_pearson(k, n)
                r_out.update({f"{g}_pass": k, f"{g}_lo": lo, f"{g}_hi": hi})
            k = sum(1 for r, v in ver if all(v.values()) and r.get("global_rng_ok") in ("True", "true", "1", "1.0"))
            lo, hi = clopper_pearson(k, n)
            r_out.update({"run_pass": k, "run_pass_lo": lo, "run_pass_hi": hi})
            rows.append(r_out)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Dict rows to a CSV (the union of the keys, in first-seen order)."""
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def compare(blind: Sequence[Dict[str, Any]], analysis_csv: Path, keys: Sequence[str]) -> Tuple[float, int, List[str]]:
    """Largest absolute difference over the numeric columns the two tables share (rows matched on ``keys``); the number of
    compared numbers; the columns of ``blind`` that the analysis table lacks."""
    ana: Dict[Tuple, Dict[str, str]] = {}
    with open(analysis_csv, newline="") as f:
        for r in csv.DictReader(f):
            ana[tuple(str(int(float(r[k]))) if k == "q" else r[k] for k in keys)] = r
    worst, n_cmp, missing = 0.0, 0, []
    for b in blind:
        a = ana.get(tuple(str(b[k]) for k in keys))
        if a is None:
            missing.append("row " + ",".join(str(b[k]) for k in keys))
            continue
        for k, v in b.items():
            if k in keys or isinstance(v, str):
                continue
            if k not in a:
                if k not in missing:
                    missing.append(k)
                continue
            x, y = float(v), num(a[k]) if a[k] not in ("True", "False") else float(a[k] == "True")
            if math.isnan(x) and math.isnan(y):
                continue
            n_cmp += 1
            worst = max(worst, abs(x - y)) if not (math.isnan(x) or math.isnan(y)) else float("inf")
    return worst, n_cmp, missing


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R1 blind recomputation", allow_abbrev=False)
    ap.add_argument("--per-run", required=True)
    ap.add_argument("--criteria", default=str(CRITERIA))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--analysis-dir", default=None)
    a = ap.parse_args(argv)
    crit = json.load(open(a.criteria))
    tab = read_table(Path(a.per_run))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    crows = criterion_rows(tab, crit)
    grows = gate_count_rows(tab, crit)
    write_csv(out / "blind_criterion.csv", crows)
    write_csv(out / "blind_gate_counts.csv", grows)
    print(f"{len(crows)} criterion rows, {len(grows)} gate-count rows -> {out}")
    if a.analysis_dir:
        ok = True
        for rows, name, keys in ((crows, "criterion.csv", ("hypothesis", "q")), (grows, "gate_counts.csv", ("arm", "q"))):
            worst, n_cmp, missing = compare(rows, Path(a.analysis_dir) / name, keys)
            print(f"{name}: {n_cmp} numbers compared, largest absolute difference {worst:.3g}, "
                  f"not in the analysis table: {missing}")
            ok = ok and worst <= TOL and not [m for m in missing if m.startswith("row ")]
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
