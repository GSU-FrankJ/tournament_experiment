#!/usr/bin/env python3
"""Independent recomputation of the T3-R2 confirmation counts and verdict (prompt section 3.3, "Blind recomputation").

Reads only the evaluation table ``per_run.csv`` (``tools/t3/t3c_eval.py``), the infrastructure ledger ``infra_events.json`` and
the thresholds of ``reports/t3/r1/prereg_criteria.json``; nothing is imported from ``t3c_analysis.py``, ``t3_metrics.py`` or
any other tool of the round. The gate verdicts are recomputed from the metric columns; the Clopper-Pearson intervals come from
the beta-quantile formula (the analysis tool uses ``binomtest``), so the two paths share no code.

``--analysis-dir`` compares the counts, the intervals and the verdict with ``verdict.json`` of the analysis tool (they must
agree: counts and verdict exactly, intervals to 1e-9).

Usage:
    python tools/t3/t3c_blind.py --per-run results/t3_r2/confirmation/eval/per_run.csv \
        --ledger results/t3_r2/confirmation/infra_events.json --out-dir results/t3_r2/confirmation/analysis/blind \
        [--analysis-dir results/t3_r2/confirmation/analysis]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[2]
CRITERIA = ROOT / "reports" / "t3" / "r1" / "prereg_criteria.json"
QS = (50, 60)
SEEDS = list(range(60501, 60521))
TOL_CI = 1e-9


def num(x: Any) -> float:
    """A CSV cell as a float (empty / missing: NaN)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def is_true(x: Any) -> bool:
    """The text ``True`` / ``true`` / ``1`` / ``1.0`` (as a CSV writes a Python bool or a float flag)."""
    return str(x).strip().lower() in ("true", "1", "1.0")


def read_rows(path: Path) -> Dict[Tuple[int, int], Dict[str, str]]:
    """``{(q, seed): row}`` of the ``bw`` rows of the evaluation table."""
    out: Dict[Tuple[int, int], Dict[str, str]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("arm", "bw") == "bw":
                out[(int(float(r["q"])), int(float(r["seed"])))] = r
    return out


def cp_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided interval of k / n from the beta quantiles."""
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def run_passes(r: Optional[Dict[str, str]], g: Dict[str, Any]) -> bool:
    """Whether a planned run passes: exit 0, RNG ok, a valid evaluation, and G-A, G-F, G-N (inclusive ``<=``)."""
    if r is None:
        return False
    if r.get("status") != "done" or num(r.get("exit_code")) != 0.0 or not is_true(r.get("global_rng_ok")):
        return False
    if (r.get("eval_error") or "") != "" or not is_true(r.get("valid")) or not is_true(r.get("dev_valid")):
        return False
    eta, rm, tl = num(r["eta_T_over_dw"]), num(r["rmse_pos_T_over_eT0"]), num(r["tail_mean_T_over_eT0"])
    gm = num(r["Gmax_full_over_dw"])
    d_eta = abs(num(r["dev_eta_T_over_dw"]) - eta)
    d_gm = abs(num(r["dev_Gmax_full_over_dw"]) - gm)
    checks = [eta <= g["G-A"]["eta_T_over_dw"], rm <= g["G-A"]["rmse_pos_T_over_eT0"], tl <= g["G-A"]["tail_mean_T_over_eT0"],
              gm <= g["G-F"]["Gmax_full_over_dw"], d_eta <= g["G-N"]["eta_T_over_dw_dev_minus_final_abs"],
              d_gm <= g["G-N"]["Gmax_full_over_dw_dev_minus_final_abs"]]
    return all(bool(c) for c in checks)       # a NaN compares False


def recompute(per_run: Path, ledger: Path, criteria: Path = CRITERIA) -> Dict[str, Any]:
    """Counts, intervals and the verdict.

    Raises:
        FileNotFoundError: The ledger does not exist (it must, empty if there was no infrastructure event).
    """
    g = json.loads(Path(criteria).read_text())["gates"]
    if not Path(ledger).is_file():
        raise FileNotFoundError(f"{ledger}: the infrastructure ledger must exist")
    led = json.loads(Path(ledger).read_text())
    stopped = list(led.get("stopped_without_verdict", []))
    for e in led.get("events", []):                      # an event of attempt 2 or later: the repeat also failed
        if int(e.get("attempt", 1)) >= 2 and e["run"] not in stopped:
            stopped.append(e["run"])
    rows = read_rows(per_run)
    res: Dict[str, Any] = {}
    total = counted = 0
    for q in QS:
        runs = [f"t3r2_q{q}_s{s}_bw" for s in SEEDS]
        keep = [(s, name) for s, name in zip(SEEDS, runs) if name not in stopped]
        k = sum(1 for s, _ in keep if run_passes(rows.get((q, s)), g))
        lo, hi = cp_interval(k, len(keep)) if keep else (float("nan"), float("nan"))
        res[f"pass_q{q}"], res[f"n_counted_q{q}"], res[f"ci_lo_q{q}"], res[f"ci_hi_q{q}"] = k, len(keep), lo, hi
        total, counted = total + k, counted + len(keep)
    res["pass_total"], res["n_counted_total"] = total, counted
    res["ci_lo_total"], res["ci_hi_total"] = cp_interval(total, counted) if counted else (float("nan"), float("nan"))
    ok = res["pass_total"] >= 36 and res["pass_q50"] >= 17 and res["pass_q60"] >= 17
    res["verdict"] = "no_verdict" if stopped else ("pass" if ok else "fail")
    res["stopped_without_verdict"] = stopped
    return res


def compare(blind: Dict[str, Any], analysis_json: Path) -> Tuple[bool, List[str]]:
    """Whether the blind result agrees with ``verdict.json`` of the analysis tool; the differences."""
    a = json.loads(Path(analysis_json).read_text())
    diffs = []
    if a["verdict"]["verdict"] != blind["verdict"]:
        diffs.append(f"verdict {a['verdict']['verdict']} != {blind['verdict']}")
    for k in ("pass_q50", "pass_q60", "pass_total", "n_counted_q50", "n_counted_q60", "n_counted_total"):
        if int(a["counts"][k]) != int(blind[k]):
            diffs.append(f"{k}: {a['counts'][k]} != {blind[k]}")
    for k in ("ci_lo_q50", "ci_hi_q50", "ci_lo_q60", "ci_hi_q60", "ci_lo_total", "ci_hi_total"):
        x, y = a["counts"][k], blind[k]
        if not (math.isnan(x) and math.isnan(y)) and abs(x - y) > TOL_CI:
            diffs.append(f"{k}: {x} vs {y}")
    return not diffs, diffs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R2 blind recomputation", allow_abbrev=False)
    ap.add_argument("--per-run", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--criteria", default=str(CRITERIA))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--analysis-dir", default=None)
    a = ap.parse_args(argv)
    res = recompute(Path(a.per_run), Path(a.ledger), Path(a.criteria))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "blind_verdict.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    if a.analysis_dir:
        ok, diffs = compare(res, Path(a.analysis_dir) / "verdict.json")
        print("agrees with the analysis tool:", ok, diffs)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
