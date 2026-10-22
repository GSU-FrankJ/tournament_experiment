#!/usr/bin/env python3
"""Blind recomputation of the T3-R4 confirmation counts and verdict (prompt 3.3).

Independent of ``t3r4_analysis.py`` (it imports neither it nor ``t3c_analysis`` nor ``t3r3_analysis``): it reads the evaluation table
(``per_run.csv`` of ``t3r4_eval.py``), the infrastructure ledger and the gate thresholds of ``reports/t3/r1/prereg_criteria.json``, and
recomputes the number of failed runs per q and in total, the exact intervals of the pass counts and the verdict under the confirmation rule
(at most 6 of the 80 runs fail and at most 4 of the 40 at each q). A run passes iff it exited with code 0, its global-RNG assertion is ok,
its evaluation has no error and is valid on both tiers, and eta_3 <= 0.005, rmse_pos_T_over_eT0 <= 0.05, tail_mean_T_over_eT0 <= 0.02,
Gmax_full <= 0.01, and the development-final differences of eta_3 and of Gmax_full are <= 0.001 (every <= inclusive, float64). A planned
run without a row is a failed run; a run the ledger lists as stopped without a verdict (a second infrastructure failure) makes the
confirmation a stop without a verdict.

Usage:
    python tools/t3/t3r4_blind.py --per-run results/t3_r4/confirmation/eval/per_run.csv \
        --ledger results/t3_r4/confirmation/infra_events.json \
        [--headline results/t3_r4/confirmation/analysis/headline.json] [--out results/t3_r4/confirmation/analysis/blind.json]
Exit code 1 if a headline is given and disagrees.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[2]
CRITERIA = ROOT / "reports" / "t3" / "r1" / "prereg_criteria.json"
QS = (50, 60)
SEEDS = tuple(range(80501, 80541))
MAX_FAIL_TOTAL, MAX_FAIL_PER_Q = 6, 4


def num(x: Any) -> float:
    """A CSV cell as float (NaN if empty / not a number)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def is_true(x: Any) -> bool:
    """A CSV truth value."""
    return str(x).strip().lower() in ("true", "1", "1.0")


def cp_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided interval from the beta quantiles."""
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def passes(r: Optional[Mapping[str, str]], g: Mapping[str, Any]) -> bool:
    """Whether a planned run passes (see the module docstring); ``None``: no row."""
    if r is None:
        return False
    if r.get("status") != "done" or num(r.get("exit_code")) != 0.0 or not is_true(r.get("global_rng_ok")):
        return False
    if (r.get("eval_error") or "").strip() not in ("", "nan") or not is_true(r.get("valid")) or not is_true(r.get("dev_valid")):
        return False
    eta, rm, tl, gm = (num(r[k]) for k in ("eta_T_over_dw", "rmse_pos_T_over_eT0", "tail_mean_T_over_eT0", "Gmax_full_over_dw"))
    d_eta = abs(num(r["dev_eta_T_over_dw"]) - eta)
    d_gm = abs(num(r["dev_Gmax_full_over_dw"]) - gm)
    checks = [eta <= g["G-A"]["eta_T_over_dw"], rm <= g["G-A"]["rmse_pos_T_over_eT0"], tl <= g["G-A"]["tail_mean_T_over_eT0"],
              gm <= g["G-F"]["Gmax_full_over_dw"], d_eta <= g["G-N"]["eta_T_over_dw_dev_minus_final_abs"],
              d_gm <= g["G-N"]["Gmax_full_over_dw_dev_minus_final_abs"]]
    return all(bool(c) for c in checks)       # a NaN compares False


def run_name(q: int, seed: int) -> str:
    """``t3r4_q{q}_s{seed}_bwA``."""
    return f"t3r4_q{q}_s{seed}_bwA"


def rule_verdict(fail_q50: int, fail_q60: int, stopped: Sequence[str]) -> Tuple[str, bool, bool, bool]:
    """``(verdict, total ok, q50 ok, q60 ok)`` under the rule: at most 6 failures in all and at most 4 at each q; a stop without a verdict
    if the ledger lists a run."""
    t_ok = bool(fail_q50 + fail_q60 <= MAX_FAIL_TOTAL)
    a_ok, b_ok = bool(fail_q50 <= MAX_FAIL_PER_Q), bool(fail_q60 <= MAX_FAIL_PER_Q)
    if stopped:
        return "no_verdict", t_ok, a_ok, b_ok
    return ("pass" if (t_ok and a_ok and b_ok) else "fail"), t_ok, a_ok, b_ok


def recompute(per_run: Path, ledger: Path, criteria: Path = CRITERIA) -> Dict[str, Any]:
    """Failures per q, intervals and the verdict.

    Raises:
        FileNotFoundError: The ledger does not exist (it must, empty if there was no infrastructure event).
    """
    g = json.loads(Path(criteria).read_text())["gates"]
    if not Path(ledger).is_file():
        raise FileNotFoundError(f"{ledger}: the infrastructure ledger must exist")
    led = json.loads(Path(ledger).read_text())
    stopped = list(led.get("stopped_without_verdict", []))
    for e in led.get("events", []):
        if int(e.get("attempt", 1)) >= 2 and e["run"] not in stopped:
            stopped.append(e["run"])
    with open(per_run, newline="") as f:
        rows = {r["run"]: r for r in csv.DictReader(f)}
    out: Dict[str, Any] = {}
    fails: Dict[int, int] = {}
    tot = n_tot = 0
    for q in QS:
        k = sum(1 for s in SEEDS if run_name(q, s) not in stopped and passes(rows.get(run_name(q, s)), g))
        n = len(SEEDS)
        lo, hi = cp_interval(k, n)
        out[f"pass_q{q}"], out[f"n_q{q}"], out[f"ci_lo_q{q}"], out[f"ci_hi_q{q}"] = k, n, lo, hi
        fails[q] = n - k
        out[f"fail_q{q}"] = n - k
        tot, n_tot = tot + k, n_tot + n
    out["pass_total"], out["n_total"], out["fail_total"] = tot, n_tot, n_tot - tot
    out["ci_lo_total"], out["ci_hi_total"] = cp_interval(tot, n_tot)
    out["verdict"], out["total_condition"], out["q50_condition"], out["q60_condition"] = rule_verdict(fails[50], fails[60], stopped)
    out["stopped_without_verdict"] = stopped
    return out


def compare(blind: Mapping[str, Any], headline: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """Whether the blind result agrees with ``headline.json`` of the analysis tool; the differences."""
    diffs: List[str] = []
    c, v = headline["counts"], headline["verdict"]
    for k in ("pass_q50", "pass_q60", "pass_total", "n_q50", "n_q60", "n_total"):
        if int(c[k]) != int(blind[k]):
            diffs.append(f"{k}: {c[k]} != {blind[k]}")
    for k in ("ci_lo_q50", "ci_hi_q50", "ci_lo_q60", "ci_hi_q60", "ci_lo_total", "ci_hi_total"):
        if not math.isclose(float(c[k]), float(blind[k]), rel_tol=0, abs_tol=1e-9):
            diffs.append(f"{k}: {c[k]} != {blind[k]}")
    if v["verdict"] != blind["verdict"]:
        diffs.append(f"verdict: {v['verdict']} != {blind['verdict']}")
    for k in ("total_condition", "q50_condition", "q60_condition"):
        if bool(v[k]) != bool(blind[k]):
            diffs.append(f"{k}: {v[k]} != {blind[k]}")
    return not diffs, diffs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R4 blind recomputation", allow_abbrev=False)
    ap.add_argument("--per-run", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--headline", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = recompute(Path(a.per_run), Path(a.ledger))
    ok = True
    if a.headline:
        ok, diffs = compare(res, json.loads(Path(a.headline).read_text()))
        res["agrees_with_headline"], res["differences"] = ok, diffs
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
