#!/usr/bin/env python3
"""Blind recomputation of ``criterion.csv`` and the ``interaction.csv`` summary from ``per_run.csv`` (MS-R2, D5, 3.3).

An independent script: plain numpy / pandas only, it imports nothing from ``tools/ms/`` (no constant, no helper, no
thresholds). From ``per_run.csv`` alone it re-derives

  * for every comparison arm ``NL_{bb,st}_s{4,16}`` of ``criterion.csv`` (baseline: the SAME sampler's ``NL_*_s1``
    arm, derived here from the arm name) and every q: (a) the number of pairs, the mean and the 95% percentile
    bootstrap interval of the paired difference ``arm - baseline`` of ``stage2_peak_rel_err_abs`` (the seeds where both
    runs are ``done`` and the value is finite), 10,000 resamples, ONE fresh ``numpy.random.default_rng(20261007)`` per
    (q, statistic), resampled indices ``rng.integers(0, n, size=(10000, n))``, percentiles 2.5 / 97.5 of the
    resampled mean, and the flag "upper bound < 0 at BOTH q"; (b) the runs that pass G-A with the eta part of G-N under
    the baseline (G-A on the final tier: eta_2/DW, RMSE_pos/e2*(0), tail mean/e2*(0) against the thresholds of the
    protocol file; G-N eta part: |eta(development) - eta(final)| against its threshold; recomputed from the component
    columns of ``per_run.csv`` and compared with its ``gate_pass`` column) and which of them fail under the arm
    (violation: arm run done and failing, or failed; pending: arm run missing / running / incomplete);
  * for ``interaction.csv`` (summary rows per s in {4, 16}, q and metric in {``stage2_peak_rel_err_abs``,
    ``remainder``, ``smoothing``}): the paired interaction ``(NL_st_s - NL_st_s1) - (NL_bb_s - NL_bb_s1)`` per seed
    (the seeds where all four runs are ``done`` and the value finite), its count, mean, median, sign counts and the
    percentile bootstrap intervals of the mean and the median (same scheme),

and compares every number with the files (agreement to 1e-12 for floats, equality for integers, flags and lists).
The report is written to ``blind_recomputation.txt`` next to ``criterion.csv`` (``--out``). Exit code 0 if everything
agrees, 1 on any disagreement (a number, a flag, a missing or an extra row), 2 if an input cannot be read.

Usage:
    python tools/ms/r2_blind_criterion.py --analysis-dir results/ms_r2/analysis
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

SEED = 20261007
N_RESAMPLES = 10000
TOL = 1e-12
METRIC = "stage2_peak_rel_err_abs"
INTERACTION_METRICS = (METRIC, "remainder", "smoothing")
INTERACTION_SCALES = (4, 16)
ARM_RE = re.compile(r"NL_(bb|st)_s(\d+)")
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


def boot_ci(x: np.ndarray, stat: str = "mean") -> Tuple[float, float]:
    """95% percentile bootstrap interval of the mean (or median) with one fresh generator (the pre-registered scheme)."""
    n = x.size
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    r = x[idx]
    s = r.mean(axis=1) if stat == "mean" else np.median(r, axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(lo), float(hi)


def parse_arm(arm: str) -> Optional[Tuple[str, int]]:
    """``(sampler, s)`` of an ``NL_{bb,st}_s{s}`` arm, else None."""
    m = ARM_RE.fullmatch(str(arm))
    return (m.group(1), int(m.group(2))) if m else None


def baseline_of(arm: str) -> str:
    """The same sampler's s = 1 arm."""
    k = parse_arm(arm)
    assert k is not None
    return "NL_%s_s1" % k[0]


def recompute_arm(per: pd.DataFrame, arm: str, qs: Sequence[int], th: Dict[str, float]) -> Dict[str, Any]:
    """Every number of one criterion row, from ``per_run.csv`` alone (baseline: the same sampler's s = 1 arm)."""
    baseline = baseline_of(arm)
    out: Dict[str, Any] = {}
    a_flags: List[bool] = []
    a_full: List[bool] = []
    viol: List[str] = []
    pend: List[str] = []
    n_base = 0
    gate_col_mismatch: List[str] = []
    for q in qs:
        base_rows = per[(per["arm"] == baseline) & (per["q"] == q)].set_index("seed")
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
            lo, hi = boot_ci(d)
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
                gate_col_mismatch.append("%s q%d/%d" % (baseline, q, s))
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


def interaction_summary(per: pd.DataFrame, s: int, q: int, metric: str) -> Dict[str, Any]:
    """One row of the interaction summary: (st_s - st_s1) - (bb_s - bb_s1) over the seeds with all four runs done and
    finite on ``metric``."""
    vals = {}
    for k in ("bb", "st"):
        for tag, sc in (("s", s), ("s1", 1)):
            g = per[(per["arm"] == "NL_%s_s%d" % (k, sc)) & (per["q"] == q) & (per["status"] == "done")]
            v = pd.to_numeric(g.set_index("seed")[metric], errors="coerce")
            vals["%s_%s" % (k, tag)] = v[np.isfinite(v)]
    seeds = sorted(set.intersection(*[set(v.index) for v in vals.values()]))
    d = np.array([(vals["st_s"].loc[x] - vals["st_s1"].loc[x]) - (vals["bb_s"].loc[x] - vals["bb_s1"].loc[x])
                  for x in seeds], dtype=float)
    out: Dict[str, Any] = {"n_pairs": int(d.size), "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
                           "n_zero": int((d == 0).sum())}
    if d.size:
        out["mean"], out["median"] = float(d.mean()), float(np.median(d))
        out["ci_mean_lo"], out["ci_mean_hi"] = boot_ci(d, "mean")
        out["ci_median_lo"], out["ci_median_hi"] = boot_ci(d, "median")
    else:
        for k in ("mean", "median", "ci_mean_lo", "ci_mean_hi", "ci_median_lo", "ci_median_hi"):
            out[k] = float("nan")
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
        try:
            ok = int(mine) == int(theirs)
        except (TypeError, ValueError):
            ok = False
    else:
        ok = str(mine) == str(theirs)
    lines.append("  %-18s tool=%r blind=%r %s" % (name, theirs, mine, "ok" if ok else "DISAGREE"))
    return ok


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_criterion(per: pd.DataFrame, crit: pd.DataFrame, qs: Sequence[int], th: Dict[str, float],
                    lines: List[str]) -> Tuple[bool, int]:
    """Recompute every arm row of ``criterion.csv`` and compare it. Returns ``(all_ok, n_checked)``."""
    all_ok, n_checked = True, 0
    per_arms = [a for a in per.loc[per["role"] != COMPARATOR_ROLE, "arm"].unique()
                if parse_arm(a) is not None and parse_arm(a)[1] != 1]
    crit_arms = list(crit["arm"])
    for a in per_arms:
        if a not in crit_arms:
            lines.append("DISAGREE: arm %s is in per_run.csv but not in criterion.csv" % a)
            all_ok = False
    for a in crit_arms:
        if a not in per_arms:
            lines.append("DISAGREE: arm %s is in criterion.csv but not a comparison arm of per_run.csv" % a)
            all_ok = False
    if len(set(crit_arms)) != len(crit_arms):
        lines.append("DISAGREE: criterion.csv has a repeated arm")
        all_ok = False
    for _, row in crit.iterrows():
        arm = row["arm"]
        if arm not in per_arms:
            continue
        mine = recompute_arm(per, arm, qs, th)
        k = parse_arm(arm)
        mine.update(baseline=baseline_of(arm), sampler=k[0], s=k[1], primary_metric=METRIC, boot_seed=SEED,
                    n_boot=N_RESAMPLES)
        lines.append("%s (vs %s) " % (arm, mine["baseline"]) + " | ".join(
            "a%d %+.5f [%+.5f,%+.5f] n=%d %s" % (q, mine["mean_q%d" % q], mine["ci_mean_lo_q%d" % q],
                                                  mine["ci_mean_hi_q%d" % q], mine["n_pairs_q%d" % q],
                                                  "met" if mine["a_q%d" % q] else "not") for q in qs)
                     + " | b_viol=[%s] b_pend=[%s] | overall %s" % (mine["b_violations"], mine["b_pending"],
                                                                    mine["overall"]))
        for key, v in mine.items():
            if key.startswith("_"):
                continue
            if key not in row.index:
                lines.append("  %-18s missing in criterion.csv DISAGREE" % key)
                all_ok = False
                continue
            n_checked += 1
            all_ok &= compare(key, v, row[key], lines)
        if mine["_gate_col_mismatch"]:
            lines.append("  per_run.csv gate_pass differs from the recomputed G-A and G-N eta verdict at: %s "
                         "DISAGREE" % ", ".join(mine["_gate_col_mismatch"]))
            all_ok = False
    return all_ok, n_checked


def check_interaction(per: pd.DataFrame, inter: pd.DataFrame, qs: Sequence[int], lines: List[str]
                      ) -> Tuple[bool, int]:
    """Recompute every row of the ``interaction.csv`` summary and compare it. Returns ``(all_ok, n_checked)``."""
    all_ok, n_checked = True, 0
    expected = [(s, q, m) for s in INTERACTION_SCALES for q in qs for m in INTERACTION_METRICS]
    have = [(int(r["s"]), int(r["q"]), str(r["metric"])) for _, r in inter.iterrows()]
    for key in expected:
        if key not in have:
            lines.append("DISAGREE: interaction.csv has no row for s=%d q=%d %s" % key)
            all_ok = False
    for key in have:
        if key not in expected:
            lines.append("DISAGREE: interaction.csv has an unexpected row for s=%d q=%d %s" % key)
            all_ok = False
    if len(set(have)) != len(have):
        lines.append("DISAGREE: interaction.csv has a repeated row")
        all_ok = False
    for _, row in inter.iterrows():
        key = (int(row["s"]), int(row["q"]), str(row["metric"]))
        if key not in expected:
            continue
        mine = interaction_summary(per, *key)
        lines.append("interaction s=%d q=%d %-24s n=%d mean %+.6f [%+.6f, %+.6f]" % (
            key + (mine["n_pairs"], mine["mean"], mine["ci_mean_lo"], mine["ci_mean_hi"])))
        for k, v in mine.items():
            if k not in row.index:
                lines.append("  %-18s missing in interaction.csv DISAGREE" % k)
                all_ok = False
                continue
            n_checked += 1
            all_ok &= compare(k, v, row[k], lines)
    return all_ok, n_checked


def run(per_run: Path, criterion: Path, interaction: Path, out: Path, protocol: Path) -> int:
    """Recompute, compare, write the report; returns the exit code (0 agree, 1 disagree, 2 unreadable)."""
    lines: List[str] = []
    try:
        per = pd.read_csv(per_run, float_precision="round_trip")
        crit = pd.read_csv(criterion, float_precision="round_trip")
        inter = pd.read_csv(interaction, float_precision="round_trip")
        th = thresholds(protocol)
        qs = sorted(int(m.group(1)) for c in crit.columns for m in [re.fullmatch(r"n_pairs_q(\d+)", c)] if m)
    except Exception as exc:  # noqa: BLE001
        msg = "cannot read the inputs: %s: %s" % (type(exc).__name__, exc)
        print(msg, file=sys.stderr)
        try:
            out.write_text(msg + "\n")
        except OSError:
            pass
        return 2
    lines.append("blind recomputation of criterion.csv and the interaction.csv summary from per_run.csv "
                 "(independent of tools/ms/r2_analysis.py and tools/ms/r1_analysis.py)")
    lines.append("per_run.csv     sha256 %s" % sha256(per_run))
    lines.append("criterion.csv   sha256 %s" % sha256(criterion))
    lines.append("interaction.csv sha256 %s" % sha256(interaction))
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
    ok1, n1 = check_criterion(per, crit, qs, th, lines)
    ok2, n2 = check_interaction(per, inter, qs, lines)
    all_ok, n_checked = ok1 and ok2, n1 + n2
    lines.append("")
    lines.append(("ALL %d numbers agree with criterion.csv and interaction.csv to %.0e (floats) / exactly (counts, "
                  "flags, lists)" % (n_checked, TOL)) if all_ok else
                 "DISAGREEMENT: criterion.csv / interaction.csv do not match the blind recomputation")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-1:]))
    return 0 if all_ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="blind recomputation of the MS-R2 criterion and interaction tables")
    p.add_argument("--analysis-dir", default="results/ms_r2/analysis")
    p.add_argument("--per-run", default=None)
    p.add_argument("--criterion", default=None)
    p.add_argument("--interaction", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    a = p.parse_args(argv)
    d = Path(a.analysis_dir)
    per = Path(a.per_run) if a.per_run else d / "per_run.csv"
    crit = Path(a.criterion) if a.criterion else d / "criterion.csv"
    inter = Path(a.interaction) if a.interaction else d / "interaction.csv"
    out = Path(a.out) if a.out else crit.parent / "blind_recomputation.txt"
    return run(per, crit, inter, out, Path(a.protocol))


if __name__ == "__main__":
    sys.exit(main())
