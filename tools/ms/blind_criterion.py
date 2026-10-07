#!/usr/bin/env python3
"""Blind recomputation of ``criterion.csv`` from ``per_run.csv`` (MS-R1, prompt sections 3.2 and 3.3).

An independent script: plain numpy / pandas only, it imports nothing from ``tools/ms/r1_analysis.py`` (no
constant, no helper, no thresholds). It re-derives, for every arm of ``criterion.csv`` and every q,

  (a) the number of pairs, the mean and the 95% percentile bootstrap interval of the paired difference
      ``arm - parents_A`` of ``stage2_peak_rel_err_abs`` (the seeds where both runs are done and the value is
      finite), with 10,000 resamples and ONE fresh ``numpy.random.default_rng(20261006)`` per (q, statistic),
      the resampled indices ``rng.integers(0, n, size=(10000, n))``, percentiles 2.5 / 97.5 of the resampled
      mean, and the flag "upper bound < 0 at BOTH q";
  (b) the runs that pass G-A with the eta part of G-N under ``parents_A`` (G-A on the final tier: eta_2/DW,
      RMSE_pos/e2*(0) and tail mean/e2*(0) against the thresholds of the protocol file; G-N eta part:
      |eta(development) - eta(final)| against its threshold; recomputed from the component columns of
      ``per_run.csv`` and compared with its ``gate_pass`` column) and which of them fail under the arm
      (violation: arm run done and failing, or failed; pending: arm run missing / running / incomplete),

and compares every number with ``criterion.csv`` (agreement to 1e-12 for floats, equality for integers, flags
and lists). If ``decision_inputs.csv`` is present the counts of runs with |peak error| <= 0.05 are compared as
well. The report is written to ``blind_recomputation.txt`` next to ``criterion.csv`` (``--out``). Exit code 0 if
everything agrees, 1 on any disagreement (a number, a flag, a missing or an extra arm), 2 if an input cannot
be read.

Usage:
    python tools/ms/blind_criterion.py --analysis-dir results/ms_r1/analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SEED = 20261006
N_RESAMPLES = 10000
TOL = 1e-12
METRIC = "stage2_peak_rel_err_abs"
BASELINE = "parents_A"
COMPARATOR_ROLE = "comparator"
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[2] / "protocols" / "v2_T2_locked_v2_0.json"


def truthy(v: Any) -> bool:
    """Bool of a value read from a CSV cell (NaN / empty are False)."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() == "true"
    try:
        return bool(v) and not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def thresholds(protocol: Path) -> Dict[str, float]:
    """The three G-A thresholds and the G-N eta threshold from the protocol file (own reader)."""
    p = json.loads(Path(protocol).read_text())
    ga = {c["metric"]: float(c["threshold"]) for c in p["gates"]["G-A"]["all_must_hold"]}
    gn = {c["metric"]: float(c["threshold"]) for c in p["gates"]["G-N"]["all_must_hold"]}
    return {"eta": ga["eta_T_over_dw"], "rmse": ga["stage2_rmse_pos_over_g2_0"],
            "tail": ga["stage2_tail_mean_over_g2_0"], "n_eta": gn["eta_T_over_dw_dev_minus_final_abs"]}


def passes(row: pd.Series, th: Dict[str, float]) -> bool:
    """G-A (final tier) and the eta part of G-N from the component columns of one per_run row."""
    eta, rmse, tail = (float(row["eta_T_over_dw"]), float(row["stage2_rmse_pos_over_g2_0"]),
                       float(row["stage2_tail_mean_over_g2_0"]))
    gap = abs(float(row["eta_dev"]) - eta)
    return bool(eta <= th["eta"] and rmse <= th["rmse"] and tail <= th["tail"] and gap <= th["n_eta"])


def boot_mean_ci(x: np.ndarray) -> Tuple[float, float]:
    """95% percentile bootstrap interval of the mean with one fresh generator (the pre-registered scheme)."""
    n = x.size
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def recompute_arm(per: pd.DataFrame, arm: str, qs: Sequence[int], th: Dict[str, float]) -> Dict[str, Any]:
    """Every number of one criterion row, from ``per_run.csv`` alone."""
    out: Dict[str, Any] = {}
    a_flags: List[bool] = []
    a_full: List[bool] = []
    viol: List[str] = []
    pend: List[str] = []
    n_base = 0
    gate_col_mismatch: List[str] = []
    for q in qs:
        base_rows = per[(per["arm"] == BASELINE) & (per["q"] == q)].set_index("seed")
        arm_rows = per[(per["arm"] == arm) & (per["q"] == q)].set_index("seed")
        seeds = sorted(set(base_rows.index) | set(arm_rows.index))
        diffs = []
        for s in seeds:
            if s in base_rows.index and s in arm_rows.index:
                b, a = base_rows.loc[s], arm_rows.loc[s]
                if a["status"] == "done" and b["status"] == "done":
                    x, y = float(a[METRIC]), float(b[METRIC])
                    if np.isfinite(x) and np.isfinite(y):
                        diffs.append(x - y)
        d = np.asarray(diffs, dtype=float)
        n = int(d.size)
        out["n_pairs_q%d" % q] = n
        if n:
            lo, hi = boot_mean_ci(d)
            out["mean_q%d" % q] = float(d.mean())
            out["ci_mean_lo_q%d" % q] = lo
            out["ci_mean_hi_q%d" % q] = hi
            flag = bool(hi < 0.0)
        else:
            out["mean_q%d" % q] = out["ci_mean_lo_q%d" % q] = out["ci_mean_hi_q%d" % q] = float("nan")
            flag = False
        out["a_q%d" % q] = flag
        a_flags.append(flag)
        a_full.append(n == len(seeds))
        for s in seeds:
            b = base_rows.loc[s] if s in base_rows.index else None
            base_pass = bool(b is not None and b["status"] == "done" and passes(b, th))
            if b is not None and b["status"] == "done" and passes(b, th) != truthy(b["gate_pass"]):
                gate_col_mismatch.append("%s q%d/%d" % (BASELINE, q, s))
            if not base_pass:
                continue
            n_base += 1
            a = arm_rows.loc[s] if s in arm_rows.index else None
            st = "missing" if a is None else str(a["status"])
            if st == "done":
                if not passes(a, th):
                    viol.append("q%d/%d" % (q, s))
                if passes(a, th) != truthy(a["gate_pass"]):
                    gate_col_mismatch.append("%s q%d/%d" % (arm, q, s))
            elif st == "failed":
                viol.append("q%d/%d" % (q, s))
            else:                                   # missing / running / incomplete
                pend.append("q%d/%d" % (q, s))
    out["a_met"] = bool(all(a_flags))
    out["a_complete"] = bool(all(a_full))
    out["n_base_pass"] = n_base
    out["b_violations"] = " ".join(viol)
    out["b_pending"] = " ".join(pend)
    out["b_n_violations"], out["b_n_pending"] = len(viol), len(pend)
    out["b_status"] = "violated" if viol else ("incomplete" if pend else "holds")
    if out["b_status"] == "violated":
        out["overall"] = "not met"
    elif out["a_complete"] and out["b_status"] == "holds":
        out["overall"] = "met" if out["a_met"] else "not met"
    else:
        out["overall"] = "incomplete"
    out["_gate_col_mismatch"] = gate_col_mismatch
    return out


def compare(name: str, mine: Any, theirs: Any, lines: List[str]) -> bool:
    """Compare one value (floats to TOL, tokens of lists as sets, everything else by equality)."""
    if isinstance(mine, float) or (isinstance(theirs, float) and not isinstance(mine, (bool, str))):
        a, b = float(mine), float(theirs)
        if math.isnan(a) and math.isnan(b):
            ok, diff = True, 0.0
        else:
            diff = abs(a - b)
            ok = bool(diff <= TOL)
        lines.append("  %-18s tool=% .17g blind=% .17g |diff|=%.3g %s" % (name, b, a, diff,
                                                                          "ok" if ok else "DISAGREE"))
        return ok
    if name in ("b_violations", "b_pending"):
        a_set, b_set = set(str(mine).split()), set(("" if pd.isna(theirs) else str(theirs)).split())
        ok = a_set == b_set
        lines.append("  %-18s tool=%r blind=%r %s" % (name, " ".join(sorted(b_set)), " ".join(sorted(a_set)),
                                                      "ok" if ok else "DISAGREE"))
        return ok
    if isinstance(mine, (bool, np.bool_)):
        ok = bool(mine) == truthy(theirs)
    elif isinstance(mine, (int, np.integer)):
        ok = int(mine) == int(theirs)
    else:
        ok = str(mine) == str(theirs)
    lines.append("  %-18s tool=%r blind=%r %s" % (name, theirs, mine, "ok" if ok else "DISAGREE"))
    return ok


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(per_run: Path, criterion: Path, out: Path, protocol: Path, decision: Optional[Path]) -> int:
    """Recompute, compare, write the report; returns the exit code (0 agree, 1 disagree, 2 unreadable)."""
    lines: List[str] = []
    try:
        per = pd.read_csv(per_run, float_precision="round_trip")
        crit = pd.read_csv(criterion, float_precision="round_trip")
        th = thresholds(protocol)
    except Exception as exc:  # noqa: BLE001
        msg = "cannot read the inputs: %s: %s" % (type(exc).__name__, exc)
        print(msg, file=sys.stderr)
        out.write_text(msg + "\n")
        return 2
    qs = sorted(int(m.group(1)) for c in crit.columns for m in [re.fullmatch(r"n_pairs_q(\d+)", c)] if m)
    lines.append("blind recomputation of criterion.csv from per_run.csv (independent of tools/ms/r1_analysis.py)")
    lines.append("per_run.csv  sha256 %s" % sha256(per_run))
    lines.append("criterion.csv sha256 %s" % sha256(criterion))
    lines.append("protocol %s sha256 %s; thresholds %s" % (protocol, sha256(protocol), th))
    info_path = Path(per_run).parent / "analysis_info.json"
    if info_path.exists():                                   # provenance only: the roots of the analysed run
        try:
            info = json.loads(info_path.read_text())
            lines.append("roots (analysis_info.json, roots_id %s): %s" % (info.get("roots_id"), info.get("roots")))
        except Exception as exc:  # noqa: BLE001
            lines.append("analysis_info.json unreadable: %s" % exc)
    lines.append("bootstrap: %d resamples, numpy.random.default_rng(%d), one fresh generator per (q, statistic); "
                 "tolerance %.0e" % (N_RESAMPLES, SEED, TOL))
    lines.append("qs %s; rows by arm: %s" % (qs, per.groupby("arm").size().to_dict()))
    lines.append("status values: %s" % per["status"].value_counts().to_dict())
    all_ok = True
    per_arms = [a for a in per.loc[per["role"] != COMPARATOR_ROLE, "arm"].unique()]
    crit_arms = list(crit["arm"])
    for a in per_arms:
        if a not in crit_arms:
            lines.append("DISAGREE: arm %s is in per_run.csv but not in criterion.csv" % a)
            all_ok = False
    for a in crit_arms:
        if a not in per_arms:
            lines.append("DISAGREE: arm %s is in criterion.csv but not in per_run.csv" % a)
            all_ok = False
    n_checked = 0
    for _, row in crit.iterrows():
        arm = row["arm"]
        if arm not in per_arms:
            continue
        mine = recompute_arm(per, arm, qs, th)
        lines.append("%s " % arm + " | ".join(
            "a%d %+.5f [%+.5f,%+.5f] n=%d %s" % (q, mine["mean_q%d" % q], mine["ci_mean_lo_q%d" % q],
                                                  mine["ci_mean_hi_q%d" % q], mine["n_pairs_q%d" % q],
                                                  "met" if mine["a_q%d" % q] else "not") for q in qs)
                     + " | b_viol=[%s] b_pend=[%s] | overall %s" % (mine["b_violations"], mine["b_pending"],
                                                                    mine["overall"]))
        for k, v in mine.items():
            if k.startswith("_"):
                continue
            if k not in row.index:
                lines.append("  %-18s missing in criterion.csv DISAGREE" % k)
                all_ok = False
                continue
            n_checked += 1
            all_ok &= compare(k, v, row[k], lines)
        if mine["_gate_col_mismatch"]:
            lines.append("  per_run.csv gate_pass differs from the recomputed G-A and G-N eta verdict at: %s "
                         "DISAGREE" % ", ".join(mine["_gate_col_mismatch"]))
            all_ok = False
    if decision is not None and Path(decision).exists():
        dec = pd.read_csv(decision, float_precision="round_trip")
        lines.append("decision_inputs.csv: runs with |peak error| <= 0.05")
        for _, r in dec.iterrows():
            q = r["q"]
            sel = (per["arm"] == r["arm"]) & (per["status"] == "done")
            if str(q) != "both":
                sel &= per["q"] == int(q)
            v = pd.to_numeric(per.loc[sel, METRIC], errors="coerce").dropna()
            mine_n = int((v <= 0.05).sum())
            n_checked += 1
            ok = mine_n == int(r["n_abs_peak_le_0.05"])
            all_ok &= ok
            lines.append("  %-15s q=%-5s tool=%d blind=%d %s" % (r["arm"], q, int(r["n_abs_peak_le_0.05"]), mine_n,
                                                                "ok" if ok else "DISAGREE"))
    lines.append("")
    lines.append(("ALL %d numbers agree with criterion.csv to %.0e (floats) / exactly (counts, flags, lists)"
                  % (n_checked, TOL)) if all_ok else "DISAGREEMENT: criterion.csv does not match the blind "
                 "recomputation")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-1:]))
    return 0 if all_ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="blind recomputation of criterion.csv")
    p.add_argument("--analysis-dir", default="results/ms_r1/analysis")
    p.add_argument("--per-run", default=None)
    p.add_argument("--criterion", default=None)
    p.add_argument("--decision-inputs", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    a = p.parse_args(argv)
    d = Path(a.analysis_dir)
    per = Path(a.per_run) if a.per_run else d / "per_run.csv"
    crit = Path(a.criterion) if a.criterion else d / "criterion.csv"
    dec = Path(a.decision_inputs) if a.decision_inputs else d / "decision_inputs.csv"
    out = Path(a.out) if a.out else d / "blind_recomputation.txt"
    return run(per, crit, out, Path(a.protocol), dec)


if __name__ == "__main__":
    sys.exit(main())
