#!/usr/bin/env python3
"""Blind recomputation of ``criterion.csv``, ``transmission.csv`` and the ``interaction.csv`` summary from
``per_run.csv`` (MS-R3, D5, D6, 3.3).

An independent script: plain numpy / pandas only, it imports nothing from ``tools/ms/`` (no constant, no helper, no
thresholds) and nothing from the analysis tool. From ``per_run.csv`` alone it re-derives

  * for every arm ``{relu,t10}_{bb,st}_s{s}`` of ``criterion.csv`` (baseline: the ``t1`` arm with the SAME starts and
    s, derived here from the arm name) and every q: (a) the number of pairs, the mean and the 95% percentile bootstrap
    interval of the paired difference ``arm - baseline`` of ``stage2_peak_rel_err_abs`` (the seeds where both runs are
    ``done`` and the value is finite), 10,000 resamples, ONE fresh ``numpy.random.default_rng(20261008)`` per
    (q, statistic), resampled indices ``rng.integers(0, n, size=(10000, n))``, percentiles 2.5 / 97.5 of the
    resampled mean, and the flag "upper bound < 0 at BOTH q"; (b) the runs that pass G-A with the eta part of G-N
    under the baseline (G-A on the final tier: eta_2/DW, RMSE_pos/e2*(0), tail mean/e2*(0) against the thresholds of
    the protocol file; G-N eta part: |eta(development) - eta(final)| against its threshold; recomputed from the
    component columns of ``per_run.csv`` and compared with its ``gate_pass`` column) and which of them fail under the
    arm (violation: arm run done and failing, or failed; pending: arm run missing / running / incomplete);
  * for ``transmission.csv`` (one row per ``{actor}_{starts}_s16`` arm and q, baseline: the same actor's and
    starts' s = 1 arm): over the seeds where both runs are ``done`` and the gap and the smoothing part are finite in
    both, the changes ``s16 - s1`` of the gap and of the smoothing part, their means, the ratio of the means (NaN when
    the mean change of the smoothing part is not above 1e-9 in absolute value), its percentile bootstrap interval (the
    ratio is recomputed on every resample; resamples whose mean smoothing change is not above 1e-9 in absolute value
    are dropped and counted), the median of the per-seed ratios and the per-seed ratio string;
  * for ``interaction.csv`` (summary rows per actor in {relu, t10}, starts, q and metric in
    {``stage2_peak_rel_err_abs``, ``gap``, ``remainder``, ``smoothing``}): the paired interaction
    ``(v_s16 - v_s1) - (t1_s16 - t1_s1)`` per seed (the seeds where all four runs are ``done`` and the value finite),
    its count, mean, median, sign counts and the percentile bootstrap intervals of the mean and the median (same
    scheme),

and compares every number with the files (agreement to 1e-12 for floats, equality for integers, flags and lists). The
report is written to ``blind_recomputation.txt`` next to ``criterion.csv`` (``--out``). Exit code 0 if everything
agrees, 1 on any disagreement (a number, a flag, a missing or an extra row), 2 if an input cannot be read.

Usage:
    python tools/ms/r3_blind_criterion.py --analysis-dir results/ms_r3/analysis
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

SEED = 20261008
N_RESAMPLES = 10000
TOL = 1e-12
DENOM_TOL = 1e-9
METRIC = "stage2_peak_rel_err_abs"
CONTROL = "t1"
S_LOW, S_HIGH = 1, 16
INTERACTION_METRICS = (METRIC, "gap", "remainder", "smoothing")
ARM_RE = re.compile(r"(t1|relu|t10)_(bb|st)_s(\d+)")
COMPARATOR_ROLE = "comparator"
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[2] / "protocols" / "v2_T2_locked_v2_0.json"
STRING_KEYS = ("baseline", "actor", "starts", "primary_metric", "b_status", "overall", "per_seed_ratios", "metric")
LIST_KEYS = ("b_violations", "b_pending")


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


def norm_str(v: Any) -> str:
    """A CSV string cell as a string (an empty cell is read back as NaN)."""
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


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
    """95% percentile bootstrap interval of the mean (or median) with one fresh generator (the pre-registered
    scheme)."""
    n = x.size
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    r = x[idx]
    s = r.mean(axis=1) if stat == "mean" else np.median(r, axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(lo), float(hi)


def parse_arm(arm: str) -> Optional[Tuple[str, str, int]]:
    """``(actor, starts, s)`` of an ``{actor}_{bb|st}_s{s}`` arm, else None."""
    m = ARM_RE.fullmatch(str(arm))
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def arm_name(actor: str, starts: str, s: int) -> str:
    """``{actor}_{starts}_s{s}``."""
    return "%s_%s_s%d" % (actor, starts, s)


def done_values(per: pd.DataFrame, arm: str, q: int, metric: str) -> pd.Series:
    """Finite values of ``metric`` of the ``done`` runs of (arm, q), indexed by seed."""
    g = per[(per["arm"] == arm) & (per["q"] == q) & (per["status"] == "done")]
    v = pd.to_numeric(g.set_index("seed")[metric], errors="coerce")
    return v[np.isfinite(v)]


def recompute_arm(per: pd.DataFrame, arm: str, baseline: str, qs: Sequence[int],
                  th: Dict[str, float]) -> Dict[str, Any]:
    """Every number of one criterion row, from ``per_run.csv`` alone."""
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


def recompute_transmission(per: pd.DataFrame, arm: str, baseline: str, q: int) -> Dict[str, Any]:
    """One transmission row: the paired changes ``arm - baseline`` of the gap and of the smoothing part, their ratio
    of means and its bootstrap interval, over the seeds where both runs are ``done`` and both quantities are
    finite."""
    gap_a, gap_b = done_values(per, arm, q, "gap"), done_values(per, baseline, q, "gap")
    sm_a, sm_b = done_values(per, arm, q, "smoothing"), done_values(per, baseline, q, "smoothing")
    seeds = sorted(set(gap_a.index) & set(gap_b.index) & set(sm_a.index) & set(sm_b.index))
    dg = np.array([gap_a.loc[s] - gap_b.loc[s] for s in seeds], dtype=float)
    ds = np.array([sm_a.loc[s] - sm_b.loc[s] for s in seeds], dtype=float)
    nan = float("nan")
    out: Dict[str, Any] = {"n_pairs": int(dg.size), "mean_d_gap": float(dg.mean()) if dg.size else nan,
                           "mean_d_smoothing": float(ds.mean()) if ds.size else nan, "ratio": nan, "ci_lo": nan,
                           "ci_hi": nan, "n_boot_valid": 0}
    per_seed = [float(g / s_) if abs(s_) > DENOM_TOL else nan for g, s_ in zip(dg, ds)]
    fin = [x for x in per_seed if np.isfinite(x)]
    out["median_per_seed_ratio"] = float(np.median(fin)) if fin else nan
    out["per_seed_ratios"] = ";".join("%d:%.6g" % (sd, x) for sd, x in zip(seeds, per_seed))
    if dg.size == 0:
        return out
    if abs(out["mean_d_smoothing"]) > DENOM_TOL:
        out["ratio"] = out["mean_d_gap"] / out["mean_d_smoothing"]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, dg.size, size=(N_RESAMPLES, dg.size))
    mg, ms = dg[idx].mean(axis=1), ds[idx].mean(axis=1)
    ok = np.abs(ms) > DENOM_TOL
    out["n_boot_valid"] = int(ok.sum())
    if ok.any():
        r = mg[ok] / ms[ok]
        lo, hi = np.percentile(r, [2.5, 97.5])
        out["ci_lo"], out["ci_hi"] = float(lo), float(hi)
    return out


def interaction_summary(per: pd.DataFrame, actor: str, starts: str, q: int, metric: str) -> Dict[str, Any]:
    """One row of the interaction summary: (v_s16 - v_s1) - (t1_s16 - t1_s1) over the seeds with all four runs done
    and finite on ``metric``."""
    v = {"v_s16": done_values(per, arm_name(actor, starts, S_HIGH), q, metric),
         "v_s1": done_values(per, arm_name(actor, starts, S_LOW), q, metric),
         "t1_s16": done_values(per, arm_name(CONTROL, starts, S_HIGH), q, metric),
         "t1_s1": done_values(per, arm_name(CONTROL, starts, S_LOW), q, metric)}
    seeds = sorted(set.intersection(*[set(x.index) for x in v.values()]))
    d = np.array([(v["v_s16"].loc[x] - v["v_s1"].loc[x]) - (v["t1_s16"].loc[x] - v["t1_s1"].loc[x]) for x in seeds],
                 dtype=float)
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
    """Compare one value (floats to TOL, tokens of lists as sets, strings and everything else by equality)."""
    if name in LIST_KEYS:
        a_set, b_set = set(norm_str(mine).split()), set(norm_str(theirs).split())
        ok = a_set == b_set
        lines.append("  %-18s tool=%r blind=%r %s" % (name, " ".join(sorted(b_set)), " ".join(sorted(a_set)),
                                                      "ok" if ok else "DISAGREE"))
        return ok
    if name in STRING_KEYS:
        ok = norm_str(mine) == norm_str(theirs)
        lines.append("  %-18s tool=%r blind=%r %s" % (name, norm_str(theirs), norm_str(mine),
                                                      "ok" if ok else "DISAGREE"))
        return ok
    if isinstance(mine, (bool, np.bool_)):
        ok = bool(mine) == truthy(theirs)
    elif isinstance(mine, float):
        try:
            a, b = float(mine), float(theirs)
        except (TypeError, ValueError):
            a, b = float("nan"), float("inf")
        diff = 0.0 if (math.isnan(a) and math.isnan(b)) else abs(a - b)
        ok = bool(diff <= TOL) if not (math.isnan(a) != math.isnan(b)) else False
        lines.append("  %-18s tool=% .17g blind=% .17g |diff|=%.3g %s" % (name, b, a, diff,
                                                                            "ok" if ok else "DISAGREE"))
        return ok
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


def design_of(per: pd.DataFrame) -> Tuple[List[int], List[Tuple[str, str, int]]]:
    """The q values and the ``(actor, starts, s)`` of the arms of ``per_run.csv`` that are not comparator rows."""
    ms = per[per["role"] != COMPARATOR_ROLE]
    qs = sorted(int(q) for q in ms["q"].unique())
    arms = sorted({parse_arm(a) for a in ms["arm"].unique() if parse_arm(a) is not None})
    return qs, arms


def _key_check(have: List[Tuple], expected: List[Tuple], what: str, lines: List[str]) -> bool:
    """Missing, extra and repeated rows of a table against the expected keys."""
    ok = True
    for key in expected:
        if key not in have:
            lines.append("DISAGREE: %s has no row for %s" % (what, key))
            ok = False
    for key in have:
        if key not in expected:
            lines.append("DISAGREE: %s has an unexpected row for %s" % (what, key))
            ok = False
    if len(set(have)) != len(have):
        lines.append("DISAGREE: %s has a repeated row" % what)
        ok = False
    return ok


def check_criterion(per: pd.DataFrame, crit: pd.DataFrame, qs: Sequence[int], arms: Sequence[Tuple[str, str, int]],
                    th: Dict[str, float], lines: List[str]) -> Tuple[bool, int]:
    """Recompute every arm row of ``criterion.csv`` and compare it. Returns ``(all_ok, n_checked)``."""
    expected = [arm_name(a, k, s) for a, k, s in arms if a != CONTROL]
    have = [str(x) for x in crit["arm"]] if "arm" in crit.columns else []
    all_ok = _key_check(have, expected, "criterion.csv", lines)
    n_checked = 0
    for _, row in crit.iterrows():
        arm = str(row["arm"])
        k = parse_arm(arm)
        if arm not in expected or k is None:
            continue
        baseline = arm_name(CONTROL, k[1], k[2])
        mine = recompute_arm(per, arm, baseline, qs, th)
        mine.update(baseline=baseline, actor=k[0], starts=k[1], s=k[2], primary_metric=METRIC, boot_seed=SEED,
                    n_boot=N_RESAMPLES)
        lines.append("%s (vs %s) " % (arm, baseline) + " | ".join(
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


def check_transmission(per: pd.DataFrame, trans: pd.DataFrame, qs: Sequence[int],
                       arms: Sequence[Tuple[str, str, int]], lines: List[str]) -> Tuple[bool, int]:
    """Recompute every row of ``transmission.csv`` and compare it. Returns ``(all_ok, n_checked)``."""
    expected = [(arm_name(a, k, s), q) for a, k, s in arms if s == S_HIGH for q in qs]
    have = [(str(r["arm"]), int(r["q"])) for _, r in trans.iterrows()] if len(trans) else []
    all_ok = _key_check(have, expected, "transmission.csv", lines)
    n_checked = 0
    for _, row in trans.iterrows():
        arm, q = str(row["arm"]), int(row["q"])
        k = parse_arm(arm)
        if (arm, q) not in expected or k is None:
            continue
        baseline = arm_name(k[0], k[1], S_LOW)
        mine = recompute_transmission(per, arm, baseline, q)
        mine.update(baseline=baseline, actor=k[0], starts=k[1], s=k[2])
        lines.append("transmission %s q=%d n=%d d gap %+.6f d smoothing %+.6f ratio %+.6f [%+.6f, %+.6f] kept %d" % (
            arm, q, mine["n_pairs"], mine["mean_d_gap"], mine["mean_d_smoothing"], mine["ratio"], mine["ci_lo"],
            mine["ci_hi"], mine["n_boot_valid"]))
        for key, v in mine.items():
            if key not in row.index:
                lines.append("  %-18s missing in transmission.csv DISAGREE" % key)
                all_ok = False
                continue
            n_checked += 1
            all_ok &= compare(key, v, row[key], lines)
    return all_ok, n_checked


def check_interaction(per: pd.DataFrame, inter: pd.DataFrame, qs: Sequence[int], arms: Sequence[Tuple[str, str, int]],
                      lines: List[str]) -> Tuple[bool, int]:
    """Recompute every row of the ``interaction.csv`` summary and compare it. Returns ``(all_ok, n_checked)``."""
    actors = sorted({a for a, _, _ in arms if a != CONTROL})
    starts = sorted({k for _, k, _ in arms})
    expected = [(a, k, q, m) for a in actors for k in starts for q in qs for m in INTERACTION_METRICS]
    have = [(str(r["actor"]), str(r["starts"]), int(r["q"]), str(r["metric"])) for _, r in inter.iterrows()] \
        if len(inter) else []
    all_ok = _key_check(have, expected, "interaction.csv", lines)
    n_checked = 0
    for _, row in inter.iterrows():
        key = (str(row["actor"]), str(row["starts"]), int(row["q"]), str(row["metric"]))
        if key not in expected:
            continue
        mine = interaction_summary(per, *key)
        lines.append("interaction %s %s q=%d %-24s n=%d mean %+.6f [%+.6f, %+.6f]" % (
            key + (mine["n_pairs"], mine["mean"], mine["ci_mean_lo"], mine["ci_mean_hi"])))
        for k, v in mine.items():
            if k not in row.index:
                lines.append("  %-18s missing in interaction.csv DISAGREE" % k)
                all_ok = False
                continue
            n_checked += 1
            all_ok &= compare(k, v, row[k], lines)
    return all_ok, n_checked


def run(per_run: Path, criterion: Path, transmission: Path, interaction: Path, out: Path, protocol: Path) -> int:
    """Recompute, compare, write the report; returns the exit code (0 agree, 1 disagree, 2 unreadable)."""
    lines: List[str] = []
    try:
        per = pd.read_csv(per_run, float_precision="round_trip")
        crit = pd.read_csv(criterion, float_precision="round_trip")
        trans = pd.read_csv(transmission, float_precision="round_trip")
        inter = pd.read_csv(interaction, float_precision="round_trip")
        th = thresholds(protocol)
        qs, arms = design_of(per)
    except Exception as exc:  # noqa: BLE001
        msg = "cannot read the inputs: %s: %s" % (type(exc).__name__, exc)
        print(msg, file=sys.stderr)
        try:
            out.write_text(msg + "\n")
        except OSError:
            pass
        return 2
    lines.append("blind recomputation of criterion.csv, transmission.csv and the interaction.csv summary from "
                 "per_run.csv (independent of tools/ms/r3_analysis.py, r2_analysis.py and r1_analysis.py)")
    lines.append("per_run.csv      sha256 %s" % sha256(per_run))
    lines.append("criterion.csv    sha256 %s" % sha256(criterion))
    lines.append("transmission.csv sha256 %s" % sha256(transmission))
    lines.append("interaction.csv  sha256 %s" % sha256(interaction))
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
    lines.append("qs %s; arms %s; rows by arm: %s" % (qs, ["%s_%s_s%d" % a for a in arms],
                                                      per.groupby("arm").size().to_dict()))
    lines.append("status values: %s" % per["status"].value_counts().to_dict())
    ok1, n1 = check_criterion(per, crit, qs, arms, th, lines)
    ok2, n2 = check_transmission(per, trans, qs, arms, lines)
    ok3, n3 = check_interaction(per, inter, qs, arms, lines)
    all_ok, n_checked = ok1 and ok2 and ok3, n1 + n2 + n3
    lines.append("")
    lines.append(("ALL %d numbers agree with criterion.csv, transmission.csv and interaction.csv to %.0e (floats) / "
                  "exactly (counts, flags, lists)" % (n_checked, TOL)) if all_ok else
                 "DISAGREEMENT: criterion.csv / transmission.csv / interaction.csv do not match the blind "
                 "recomputation")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-1:]))
    return 0 if all_ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="blind recomputation of the MS-R3 criterion, transmission and "
                                            "interaction tables")
    p.add_argument("--analysis-dir", default="results/ms_r3/analysis")
    p.add_argument("--per-run", default=None)
    p.add_argument("--criterion", default=None)
    p.add_argument("--transmission", default=None)
    p.add_argument("--interaction", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    a = p.parse_args(argv)
    d = Path(a.analysis_dir)
    per = Path(a.per_run) if a.per_run else d / "per_run.csv"
    crit = Path(a.criterion) if a.criterion else d / "criterion.csv"
    trans = Path(a.transmission) if a.transmission else d / "transmission.csv"
    inter = Path(a.interaction) if a.interaction else d / "interaction.csv"
    out = Path(a.out) if a.out else crit.parent / "blind_recomputation.txt"
    return run(per, crit, trans, inter, out, Path(a.protocol))


if __name__ == "__main__":
    sys.exit(main())
