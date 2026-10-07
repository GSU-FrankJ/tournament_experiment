#!/usr/bin/env python3
"""MS-R2 pre-registered analysis of the noise-landing pilot (prompt D4, D5 and section 3.3).

Inputs (all read only; every root is an explicit argument and is recorded in ``analysis_info.json``,
``summary.txt``, the figure footers, a ``roots_id`` column of every CSV and the ``run_dir`` of every row of
``per_run.csv``):

  * ``--pilot-root``        ``q*/seed*/<NL arm>`` of MS-R2 (``NL_{bb,st}_s{1,4,16}``, runs of
                            ``run/run_ms_stagewise.py`` with the noise-report columns)
  * ``--ms-r1-pilot-root``  ``q*/seed*/<arm>`` of MS-R1 (``MS_base2400``, ``MS_s35a5``: the references of the s = 1
                            arms)
  * ``--parents-root``      ``q*/seed*`` of v2.0 ``parents_A``;  ``--rehearsal-root`` ``q*/seed*`` of ``rehearsal_v2_0``
  * ``--calibration-root``  optional, ``results/ms_r1/calibration`` (D3 columns of the two v2.0 comparators)

Rows: the six NL arms (role ``ms_arm``) and the reference rows ``MS_base2400``, ``MS_s35a5``, ``parents_A``,
``rehearsal_v2_0`` (role ``comparator``). A missing, running, failed (``status.json`` not ``done`` / exit code 0, which
includes a global-RNG violation) or unreadable run is a row with its ``status``, never a silent skip; it never enters a
pair (as in ``r1_analysis``; the per-run extraction is the one of ``r1_analysis``).

Decomposition (D4; reporting only, the closed form never enters training): per run, from the final-tier columns of
``per_run.csv`` (the definitions of ``utils.ms_noise.decompose_gap`` and ``tools/ms/r2_decomposition.py``)

    gap = g2_at_0 - e2_at_0,   smoothing = g2_at_0 - smoothed_e_pred_0,   remainder = smoothed_e_pred_0 - e2_at_0,
    gap = smoothing + remainder;  *_rel = divided by g2_at_0;  smoothing_over_formula = smoothing /
    (g2_at_0 * sigma / (sqrt(pi) * q)),  sigma = sigma_effort_at_0_t2 (also ``sigma_2_0``).

For the NL arms the decomposition is cross-checked against ``rule_log.json`` stages["2"]["freeze"]["noise"] (1e-9; a
disagreement is a flag in the row's ``flags``, never an exception).

Pre-registered criterion (D5; ``criterion.csv``; descriptive, not a gate, no selection follows; the decision is the PI's).
One row per sampler in {bb, st} x s in {4, 16}: |peak error| of the frozen terminal-stage candidate (final tier,
``stage2_peak_rel_err_abs``) paired by (q, seed) against the SAME sampler's s = 1 arm. (a) The 95% percentile bootstrap
interval of the mean paired difference lies below 0 (``ci_mean_hi < 0``, strict) at BOTH q. (b) No run that passes G-A
with its G-N eta part under the s = 1 arm fails it under s (the pass definition, thresholds from the protocol file,
and the violated / pending / holds logic of ``r1_analysis.criterion_row``). Bootstrap: 10,000 resamples, ONE FRESH
``numpy.random.default_rng(20261007)`` per (q, statistic), the resampled indices are ``rng.integers(0, n,
size=(10000, n))``, the interval is the 2.5 / 97.5 percentiles of the resampled statistic. Everything that is not this
criterion is descriptive; an interval that contains 0 with 10 seeds does not show that a mechanism has no effect.

Transmission ratio (descriptive): ``(change of the gap) / (change of the smoothing part)`` with the changes taken as
arm - s=1 of the same sampler, computed as the ratio of the MEAN paired changes; 1 = the whole smoothing change reaches
the gap, 0 = the remainder offsets it (gap = smoothing + remainder, so the ratio is 1 + change of the remainder /
change of the smoothing part). This is the PI's "-(change of the gap) / (change of the smoothing part)" read as
reduction over reduction (both changes carry the same sign when the smoothing reduction reaches the gap). Its interval
resamples the seeds and recomputes the ratio on every resample; a resample whose denominator mean is below
:data:`DENOM_TOL` in absolute value is dropped from the percentiles (the count that was kept is ``n_boot_valid``).

Outputs under ``--out`` (every CSV has ``roots_id`` as its first column): ``per_run.csv``, ``criterion.csv``,
``paired_secondary.csv``, ``paired_seed_level.csv``, ``transmission.csv``, ``transmission_seed_level.csv``,
``interaction.csv`` (summary), ``interaction_seed_level.csv``, ``paired_vs_parents_A.csv``,
``criterion_vs_parents_A.csv``, ``paired_vs_ms_r1_refs.csv``, ``paired_vs_rehearsal_v2_0.csv``, ``predictions.csv``,
``trajectory_checks.csv``, ``segments.csv``, ``freeze_decomposition.csv``, ``strata.csv``, ``strata_summary.csv``,
``r0.csv``, ``stage1.csv``, ``stage1_R1_runs.csv``, ``stage1_R1.csv``, ``gates.csv``, ``arm_summary.csv``,
``budget.csv``, ``completeness.csv``, ``analysis_info.json``, ``summary.txt``, ``figures/*.png``. No timestamps are
written: the tool is deterministic given the same inputs.

Exit code: 0 when every planned run is done; 3 when a run is missing / failed / incomplete (reported in the tables
and on stderr); 2 on an argument error.

Usage:
    python tools/ms/r2_analysis.py --pilot-root results/ms_r2/pilot \
        --ms-r1-pilot-root <MS-R1 pilot root> --parents-root <v2-t2-refine>/results/v2_refine/parents_A \
        --rehearsal-root <v2-t2-refine>/results/v2_T2_locked/rehearsal_v2_0 --out results/ms_r2/analysis
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import r1_analysis as R1  # noqa: E402
from utils.ms_noise import decompose_gap  # noqa: E402

# --------------------------------------------------------------------------------------------- constants
N_BOOT = 10000
BOOT_SEED = 20261007
QS: Tuple[int, ...] = (50, 60)
SEEDS: Tuple[int, ...] = tuple(range(10501, 10511))
SAMPLERS: Tuple[str, ...] = ("bb", "st")
SCALES: Tuple[int, ...] = (1, 4, 16)
NL_ARMS: Tuple[str, ...] = tuple("NL_%s_s%d" % (k, s) for k in SAMPLERS for s in SCALES)
REF_MS_ARMS: Tuple[str, ...] = ("MS_base2400", "MS_s35a5")
PARENTS, REHEARSAL = R1.PARENTS, R1.REHEARSAL
REF_ARMS: Tuple[str, ...] = REF_MS_ARMS + (PARENTS, REHEARSAL)
#: the comparison arms of the primary criterion: (arm, same sampler's s = 1 arm), in table order
S1_COMPARISONS: List[Tuple[str, str]] = [("NL_%s_s%d" % (k, s), "NL_%s_s1" % k) for k in SAMPLERS for s in (4, 16)]
MS_R1_REF_COMPARISONS: List[Tuple[str, str]] = [("NL_bb_s1", "MS_base2400"), ("NL_st_s1", "MS_s35a5")]
ARM_RE = re.compile(r"NL_(bb|st)_s(\d+)")
PRIMARY, SIGNED = R1.PRIMARY, R1.SIGNED
LABEL_S1 = "vs s=1 (same sampler)"
LABEL_PARENTS = "vs parents_A"
LABEL_REFS = "vs MS-R1 reference"
LABEL_REHEARSAL = "vs rehearsal_v2_0"
CRITERION_NOTE = ("pre-registered criterion (descriptive, not a gate); no selection rule and no protocol change "
                  "follow from it")
PARENTS_NOTE = ("SECONDARY, descriptive table (every NL arm against parents_A with MS-R1's criterion): not the "
                "pre-registered criterion of MS-R2, not a gate; no selection rule and no protocol change follow "
                "from it")
NO_EFFECT_SENTENCE = R1.NO_EFFECT_SENTENCE
DENOM_TOL = 1e-9                 # |mean change of the smoothing part| (effort units) below which a ratio is undefined
NOISE_TOL = 1e-9                 # absolute agreement of gap / smoothing / remainder / e_sigma / g2 / e_hat with the record
#: relative agreement of sigma and of the smoothing / formula ratio with the record: the per-run sigma is the verifier's
#: (``sigma_effort_at_0_t2``, float32 path), the record's ``sigma_0`` the analytic float64 Beta SD; they differ by ~1e-7
SIGMA_RTOL = 1e-5
PLANNING_SIGMA = 2.5             # D5: sigma_2(0) at s = 1, the planning value of the floor
PLANNING_FLOOR_NOTE = "planning_floor_pct = 100 * (2.5 / sqrt(s)) / (sqrt(pi) * q) (D5 planning value sigma_2(0) = 2.5 at s = 1)"
SCALE_COLORS = {1: "tab:blue", 4: "tab:orange", 16: "tab:red"}
SAMPLER_STYLE = {"bb": "-", "st": "--"}

R2_COLS: List[str] = ["sampler", "s", "gap", "smoothing", "remainder", "gap_rel", "smoothing_rel", "remainder_rel",
                      "sigma_2_0", "smoothing_over_formula", "conc_scale_final"]
COLUMNS: List[str] = list(R1.COLUMNS) + R2_COLS
STR_COLS = set(R1.STR_COLS) | {"sampler"}
BOOL_COLS = set(R1.BOOL_COLS)

#: (column, lower is better / None): the secondary metrics of D5, |peak error| first
SECONDARY_METRICS: List[Tuple[str, Optional[bool]]] = [
    (PRIMARY, True), (SIGNED, None), ("smoothing", True), ("remainder", None), ("gap", True),
    ("smoothing_rel", True), ("remainder_rel", None), ("gap_rel", True), ("sigma_2_0", True),
    ("stage2_rmse_pos_over_g2_0", True), ("stage2_tail_mean_over_g2_0", True), ("stage2_tail_max_over_g2_0", True),
    ("eta_T_over_dw", True), ("t2_R0_final", True), ("t2_R_final", True)]
#: the metrics of the comparisons with the references (secondary metrics, then the rest of MS-R1's terminal list)
REF_METRICS: List[Tuple[str, Optional[bool]]] = SECONDARY_METRICS + [
    ("e2_at_0", None), ("eta_dev", True), ("eta_dev_minus_final_abs", True), ("t2_R0_dev", True),
    ("t2_Rtail_final", True), ("t2_Delta_final", True), ("t2_updates", True), ("t2_episodes", True),
    ("t2_minibatch_steps", True), ("t2_wall_sec", True)]
INTERACTION_METRICS: Tuple[str, ...] = (PRIMARY, "remainder", "smoothing")
TRAJ_PEAK_COLS = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_rmse_pos_over_g2_0",
                  "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "eta_T_over_dw")
TRAJ_COLS = (["arm", "q", "seed", "status", "update", "local", "lr", "conc_scale", "e2_at_0", "sigma_0", "e_sigma_0",
              "g2_0", "smoothing", "remainder", "gap", "R0", "R", "Delta", "C", "valid"] + list(TRAJ_PEAK_COLS))
UPDATE_DIAG_COLS = ("kl_final_epoch", "clip_frac", "adv_raw_std", "value_loss", "mean_effort")
#: (record key, per-run column, tolerance kind): "abs" -> NOISE_TOL, "rel" -> SIGMA_RTOL
NOISE_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("gap", "gap", "abs"), ("smoothing", "smoothing", "abs"), ("remainder", "remainder", "abs"),
    ("e_sigma_0", "smoothed_e_pred_0", "abs"), ("g2_0", "g2_at_0", "abs"), ("e_hat_0", "e2_at_0", "abs"),
    ("sigma_0", "sigma_effort_at_0_t2", "rel"), ("smoothing_over_gaussian_formula", "smoothing_over_formula", "rel"))


@dataclass(frozen=True)
class Windows:
    """The terminal-stage segment boundaries (local updates, D2) and the first update of the trajectory tables."""

    ramp_first: int = 2001
    ramp_last: int = 2200
    hold_last: int = 2400
    decay_last: int = 2800
    traj_from: int = 1800

    def segments(self) -> List[Tuple[str, int, int]]:
        """``(name, first, last)`` of training (1..ramp_first - 1), ramp (ramp_first..ramp_last: the update at
        ``ramp_first`` still runs at scale 1), hold and decay (D2's table)."""
        return [("training", 1, self.ramp_first - 1), ("ramp", self.ramp_first, self.ramp_last),
                ("hold", self.ramp_last + 1, self.hold_last), ("decay", self.hold_last + 1, self.decay_last)]

    def valid(self) -> bool:
        """True if 1 <= ramp_first < ramp_last <= hold_last <= decay_last."""
        return 1 <= self.ramp_first < self.ramp_last <= self.hold_last <= self.decay_last


# --------------------------------------------------------------------------------------------- small helpers
_f, _fin, _truthy = R1._f, R1._fin, R1._truthy


def parse_arm(arm: str) -> Tuple[str, float]:
    """``(sampler, s)`` of an ``NL_{bb,st}_s{s}`` arm; ``("", nan)`` for any other arm."""
    m = ARM_RE.fullmatch(str(arm))
    return (m.group(1), float(m.group(2))) if m else ("", float("nan"))


def _flag(row: Dict[str, Any], msg: str) -> None:
    """Append ``msg`` to the row's ``flags`` (joined with ``"; "`` as in ``r1_analysis.finish_row``)."""
    row["flags"] = (str(row["flags"]) + "; " + msg) if str(row.get("flags") or "") else msg


# --------------------------------------------------------------------------------------------- statistics
def paired_summary(diff: Sequence[float], lower_better: Optional[bool]) -> Dict[str, Any]:
    """The statistics of ``r1_analysis.paired_summary`` with the MS-R2 bootstrap seed (fresh generator per call).

    Args:
        diff: Paired differences ``arm - baseline`` (one per seed).
        lower_better: True if a decrease is an improvement; None for a metric without direction.
    """
    d = np.asarray(diff, dtype=float)
    lo_m, hi_m = R1.boot_ci(d, "mean", N_BOOT, BOOT_SEED)
    lo_d, hi_d = R1.boot_ci(d, "median", N_BOOT, BOOT_SEED)
    return {"n_pairs": int(d.size), "mean": float(d.mean()) if d.size else float("nan"),
            "median": float(np.median(d)) if d.size else float("nan"),
            "n_better": int((d < 0).sum()) if lower_better else None,
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_zero": int((d == 0).sum()),
            "ci_mean_lo": lo_m, "ci_mean_hi": hi_m, "ci_median_lo": lo_d, "ci_median_hi": hi_d,
            "ci_mean_contains_0": bool(d.size > 0 and lo_m <= 0.0 <= hi_m),
            "ci_mean_below_0": bool(d.size > 0 and hi_m < 0.0)}


def summary_stats(x: Sequence[float]) -> Dict[str, Any]:
    """Descriptive statistics of one sample with the bootstrap interval of its mean (MS-R2 seed)."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    lo, hi = R1.boot_ci(a, "mean", N_BOOT, BOOT_SEED)
    return {"n": int(a.size), "mean": float(a.mean()) if a.size else float("nan"),
            "median": float(np.median(a)) if a.size else float("nan"),
            "sd": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
            "min": float(a.min()) if a.size else float("nan"), "max": float(a.max()) if a.size else float("nan"),
            "ci_mean_lo": lo, "ci_mean_hi": hi,
            "ci_mean_contains_0": bool(a.size > 0 and lo <= 0.0 <= hi)}


@contextmanager
def r2_bootstrap() -> Iterator[None]:
    """Rebind ``r1_analysis.summary_stats`` to the MS-R2 seed while the descriptive MS-R1 tables are built.

    ``r1_analysis.arm_summary_table`` / ``stage1_table`` call the module's ``summary_stats``, whose bootstrap seed is
    MS-R1's 20261006; inside this context they use 20261007 like every other interval of this round.
    """
    old = R1.summary_stats
    R1.summary_stats = summary_stats
    try:
        yield
    finally:
        R1.summary_stats = old


def paired_tables(df: pd.DataFrame, comparisons: Sequence[Tuple[str, str]],
                  metrics: Sequence[Tuple[str, Optional[bool]]], qs: Sequence[int], seeds: Sequence[int], label: str,
                  primary_note: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Paired-difference table (arm - baseline) and its seed-level long form, bootstrap seed 20261007.

    Args:
        df: The per-run table.
        comparisons: ``(arm, baseline)`` in table order.
        metrics: ``(column, direction)`` list; a metric without any finite pair is omitted except the first one.
        qs: q values.
        seeds: Development seeds (the pairing keys).
        label: Name of the comparison.
        primary_note: If given, the ``status`` of the |peak error| rows (the criterion statistic); every other row
            is ``descriptive``.

    Returns:
        ``(summary, seed_level)``.
    """
    rows, long_rows = [], []
    for arm, base in comparisons:
        for q in qs:
            for i, (m, direction) in enumerate(metrics):
                sv = R1.paired_seed_values(df, arm, base, q, m, seeds)
                if sv.empty and i > 0:
                    continue
                note = primary_note if (primary_note and m == PRIMARY) else "descriptive"
                rows.append({"comparison": label, "arm": arm, "baseline": base, "q": q, "metric": m,
                             "direction": "lower is better" if direction else "none", "n_expected": len(seeds),
                             "status": note, **paired_summary(sv["diff"].to_numpy(), direction)})
                for r in sv.itertuples(index=False):
                    long_rows.append({"comparison": label, "arm": arm, "baseline": base, "q": q, "metric": m,
                                      "seed": r.seed, "arm_value": r.arm_value, "base_value": r.base_value,
                                      "diff": r.diff, "status": note})
    cols = ["comparison", "arm", "baseline", "q", "metric", "seed", "arm_value", "base_value", "diff", "status"]
    return pd.DataFrame(rows), pd.DataFrame(long_rows, columns=cols)


def criterion_table(df: pd.DataFrame, paired: pd.DataFrame, comparisons: Sequence[Tuple[str, str]],
                    qs: Sequence[int], seeds: Sequence[int], label: str, note: str) -> pd.DataFrame:
    """``criterion.csv``: parts (a) and (b) per (arm, baseline) of ``comparisons`` (table order, q in the given order).

    The numbers of one row are ``r1_analysis.criterion_row`` of the |peak error| rows of ``paired`` and the gate pairs
    of (arm, baseline); only the bootstrap seed (the paired summary above) and the per-row baseline differ from
    ``r1_analysis.criterion_table``.
    """
    rows = []
    for arm, base in comparisons:
        prim = paired[(paired["arm"] == arm) & (paired["baseline"] == base) & (paired["metric"] == PRIMARY)
                      & (paired["comparison"] == label)]
        gp = R1.gate_pairs(df, arm, base, qs, seeds)
        sampler, s = parse_arm(arm)
        rows.append({"arm": arm, "baseline": base, "sampler": sampler, "s": int(s) if np.isfinite(s) else "",
                     "comparison": label, "primary_metric": PRIMARY,
                     **R1.criterion_row(prim, gp, qs, len(seeds)), "boot_seed": BOOT_SEED, "n_boot": N_BOOT,
                     "note": note})
    return pd.DataFrame(rows)


def transmission_ratio(d_gap: Sequence[float], d_smooth: Sequence[float], n_boot: int = N_BOOT,
                       seed: int = BOOT_SEED) -> Dict[str, Any]:
    """Ratio of the mean paired changes ``mean(d_gap) / mean(d_smooth)`` with its percentile bootstrap interval.

    A fresh ``default_rng(seed)`` is created per call; the seeds are resampled with ``rng.integers(0, n, size=(n_boot,
    n))`` and the ratio of the two resampled means is recomputed on every resample. NaN-safe: a mean denominator below
    :data:`DENOM_TOL` in absolute value gives a NaN ratio (point estimate) or drops the resample (interval).

    Args:
        d_gap: Paired changes of the gap (arm - s=1), one per seed.
        d_smooth: Paired changes of the smoothing part, the same seeds in the same order.
        n_boot: Number of resamples.
        seed: Generator seed.

    Returns:
        ``n_pairs``, ``mean_d_gap``, ``mean_d_smoothing``, ``ratio``, ``ci_lo``, ``ci_hi``, ``n_boot_valid``.
    """
    g, s = np.asarray(d_gap, dtype=float), np.asarray(d_smooth, dtype=float)
    nan = float("nan")
    out: Dict[str, Any] = {"n_pairs": int(g.size), "mean_d_gap": float(g.mean()) if g.size else nan,
                           "mean_d_smoothing": float(s.mean()) if s.size else nan, "ratio": nan, "ci_lo": nan,
                           "ci_hi": nan, "n_boot_valid": 0}
    if g.size == 0:
        return out
    if abs(out["mean_d_smoothing"]) > DENOM_TOL:
        out["ratio"] = out["mean_d_gap"] / out["mean_d_smoothing"]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, g.size, size=(n_boot, g.size))
    mg, ms = g[idx].mean(axis=1), s[idx].mean(axis=1)
    ok = np.abs(ms) > DENOM_TOL
    out["n_boot_valid"] = int(ok.sum())
    if ok.any():
        r = mg[ok] / ms[ok]
        out["ci_lo"], out["ci_hi"] = float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))
    return out


def transmission_tables(df: pd.DataFrame, comparisons: Sequence[Tuple[str, str]], qs: Sequence[int],
                        seeds: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``transmission.csv`` (per arm and q) and ``transmission_seed_level.csv`` (the per-seed ratios).

    The pairs are the seeds where both runs are complete and the gap and the smoothing part are finite in both.
    """
    rows, long_rows = [], []
    note = ("ratio = mean(d gap) / mean(d smoothing), d = arm - s=1 of the same sampler; 1 = the whole smoothing change "
            "reaches the gap, 0 = the remainder offsets it; interval = percentile bootstrap of the ratio of "
            "resampled means")
    for arm, base in comparisons:
        for q in qs:
            sg = R1.paired_seed_values(df, arm, base, q, "gap", seeds).set_index("seed")["diff"]
            ss = R1.paired_seed_values(df, arm, base, q, "smoothing", seeds).set_index("seed")["diff"]
            common = [s for s in seeds if s in sg.index and s in ss.index]
            dg = np.array([sg.loc[s] for s in common], dtype=float)
            ds = np.array([ss.loc[s] for s in common], dtype=float)
            per_seed = [float(g / s_) if abs(s_) > DENOM_TOL else float("nan") for g, s_ in zip(dg, ds)]
            tr = transmission_ratio(dg, ds)
            fin = [x for x in per_seed if np.isfinite(x)]
            sampler, s = parse_arm(arm)
            rows.append({"arm": arm, "baseline": base, "sampler": sampler, "s": int(s), "q": q, **tr,
                         "median_per_seed_ratio": float(np.median(fin)) if fin else float("nan"),
                         "per_seed_ratios": ";".join("%d:%.6g" % (sd, x) for sd, x in zip(common, per_seed)),
                         "status": "descriptive", "note": note})
            for sd, g_, s_, x in zip(common, dg, ds, per_seed):
                long_rows.append({"arm": arm, "baseline": base, "q": q, "seed": sd, "d_gap": float(g_),
                                  "d_smoothing": float(s_), "ratio": x, "status": "descriptive"})
    cols = ["arm", "baseline", "q", "seed", "d_gap", "d_smoothing", "ratio", "status"]
    return pd.DataFrame(rows), pd.DataFrame(long_rows, columns=cols)


def _value_series(df: pd.DataFrame, arm: str, q: int, metric: str) -> pd.Series:
    """Finite values of ``metric`` of the complete runs of (arm, q), indexed by seed."""
    g = df[(df["arm"] == arm) & (df["q"] == q) & df["complete"].astype(bool)].set_index("seed")[metric]
    g = pd.to_numeric(g, errors="coerce")
    return g[np.isfinite(g)]


def interaction_tables(df: pd.DataFrame, qs: Sequence[int], seeds: Sequence[int],
                       scales: Sequence[int] = (4, 16), metrics: Sequence[str] = INTERACTION_METRICS
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``interaction.csv`` (summary) and ``interaction_seed_level.csv``: (st_s - st_s1) - (bb_s - bb_s1) per (q, seed).

    A seed enters only when all four runs (both samplers at s and at s = 1) are complete and finite on the metric.
    """
    rows, long_rows = [], []
    for s in scales:
        for q in qs:
            for m in metrics:
                v = {"%s_%s" % (k, tag): _value_series(df, "NL_%s_s%d" % (k, sc), q, m)
                     for k in SAMPLERS for tag, sc in (("s", s), ("s1", 1))}
                diffs = []
                for sd in seeds:
                    if not all(sd in x.index for x in v.values()):
                        continue
                    st_ch = float(v["st_s"].loc[sd] - v["st_s1"].loc[sd])
                    bb_ch = float(v["bb_s"].loc[sd] - v["bb_s1"].loc[sd])
                    diffs.append(st_ch - bb_ch)
                    long_rows.append({"s": s, "q": q, "metric": m, "seed": sd, "st_s": float(v["st_s"].loc[sd]),
                                      "st_s1": float(v["st_s1"].loc[sd]), "bb_s": float(v["bb_s"].loc[sd]),
                                      "bb_s1": float(v["bb_s1"].loc[sd]), "st_change": st_ch, "bb_change": bb_ch,
                                      "interaction": st_ch - bb_ch, "status": "descriptive"})
                rows.append({"s": s, "q": q, "metric": m, "n_expected": len(seeds), "status": "descriptive",
                             **paired_summary(np.asarray(diffs, dtype=float), None)})
    cols = ["s", "q", "metric", "seed", "st_s", "st_s1", "bb_s", "bb_s1", "st_change", "bb_change", "interaction",
            "status"]
    return pd.DataFrame(rows), pd.DataFrame(long_rows, columns=cols)


# --------------------------------------------------------------------------------------------- extraction
def noise_record(d: Path) -> Dict[str, Any]:
    """The decomposition record of the terminal-stage freeze (``rule_log.json`` stages["2"]["freeze"]["noise"]);
    ``{}`` when the file or the record is absent."""
    rl = R1._jload(Path(d) / "rule_log.json", [], False)
    return R1._get(rl, "stages", "2", "freeze", "noise") or {}


def add_r2_columns(row: Dict[str, Any], is_nl: bool) -> None:
    """Add the MS-R2 columns to a per-run row of ``r1_analysis`` (in place).

    The decomposition comes from the row's own final-tier columns (``utils.ms_noise.decompose_gap``, the definition
    of ``tools/ms/r2_decomposition.py``). For an NL run it is compared with the ``noise`` record of ``rule_log.json``
    (:data:`NOISE_TOL` absolute for gap, smoothing, remainder, e_sigma, e2*(0), e_hat; :data:`SIGMA_RTOL` relative for
    sigma and the smoothing / formula ratio); a disagreement or a missing record of a done run is appended to ``flags``,
    never raised.
    """
    sampler, s = parse_arm(row["arm"])
    row["sampler"], row["s"] = sampler, s
    g2, e2, es = _f(row["g2_at_0"]), _f(row["e2_at_0"]), _f(row["smoothed_e_pred_0"])
    sig, q = _f(row["sigma_effort_at_0_t2"]), _f(row["q"])
    dec = decompose_gap(g2, es, e2, sig, q)
    nan = float("nan")
    for k in ("gap", "smoothing", "remainder"):
        row[k] = dec[k]
        row[k + "_rel"] = dec[k] / g2 if np.isfinite(g2) and g2 != 0.0 else nan
    row["sigma_2_0"] = sig
    row["smoothing_over_formula"] = dec["ratio"]
    row["conc_scale_final"] = nan
    if not is_nl or row["status"] not in ("done", "incomplete"):
        return
    rec = noise_record(Path(row["run_dir"]))
    if not rec:
        if row["status"] == "done":
            _flag(row, "rule_log.json stage 2 freeze has no noise record (noise_report)")
        return
    row["conc_scale_final"] = _f(rec.get("conc_scale"))
    bad = []
    for key, col, kind in NOISE_PAIRS:
        a, b = _f(rec.get(key)), _f(row[col])
        tol = NOISE_TOL if kind == "abs" else SIGMA_RTOL * max(abs(a), abs(b), 1e-300)
        if not (np.isfinite(a) and np.isfinite(b)) or abs(a - b) > tol:
            bad.append("%s (record %.12g, per-run %.12g)" % (key, a, b))
    if bad:
        _flag(row, "decomposition disagrees with the rule_log noise record: " + ", ".join(bad))


def arm_dir(arm: str, q: int, seed: int, roots: Mapping[str, str]) -> Tuple[Path, str]:
    """Run directory of (arm, q, seed) and the name of the root it lives under."""
    if arm == PARENTS:
        return Path(roots["parents"]) / ("q%d" % q) / ("seed%d" % seed), "parents"
    if arm == REHEARSAL:
        return Path(roots["rehearsal"]) / ("q%d" % q) / ("seed%d" % seed), "rehearsal"
    key = "ms_r1_pilot" if arm in REF_MS_ARMS else "pilot"
    return Path(roots[key]) / ("q%d" % q) / ("seed%d" % seed) / arm, key


def extract_all(roots: Mapping[str, str], qs: Sequence[int], seeds: Sequence[int], th: R1.Thresholds,
                arms: Sequence[str] = NL_ARMS) -> pd.DataFrame:
    """The per-run table: the NL arms (role ``ms_arm``), then the four reference rows (role ``comparator``).

    The per-run extraction is ``r1_analysis``'s (``extract_ms_run`` for the NL arms and the two MS-R1 references,
    ``extract_parents_run``, ``extract_rehearsal_run``; the D3 columns of the two v2.0 comparators come from
    ``--calibration-root`` when given); the MS-R2 columns are added by :func:`add_r2_columns`.
    """
    rows: List[Dict[str, Any]] = []
    for arm in list(arms) + list(REF_ARMS):
        for q in qs:
            for sd in seeds:
                d, src = arm_dir(arm, q, sd, roots)
                if arm == PARENTS:
                    r = R1.extract_parents_run(d, q, sd, th, src)
                elif arm == REHEARSAL:
                    r = R1.extract_rehearsal_run(d, q, sd, th, src)
                else:
                    r, _, _ = R1.extract_ms_run(d, arm, q, sd, th, src)
                if arm in REF_ARMS:
                    r["role"] = "comparator"
                if arm in (PARENTS, REHEARSAL) and roots.get("calibration"):
                    cal = R1.calibration_d3(Path(roots["calibration"]), q, sd)
                    if arm == PARENTS:
                        cal = {k: v for k, v in cal.items() if k.startswith("t2_")}
                    r.update(cal)
                add_r2_columns(r, arm in NL_ARMS)
                rows.append(r)
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in COLUMNS:
        if c not in STR_COLS and c not in BOOL_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    """A CSV, or None when it is missing or unreadable."""
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def trajectory_table(df: pd.DataFrame, win: Windows, arms: Sequence[str] = NL_ARMS) -> pd.DataFrame:
    """``trajectory_checks.csv``: every check row of ``ms_checks_stage2.csv`` with local update >= ``win.traj_from`` of
    every NL run (the run's ``status`` is a column; a run without a check file has no rows)."""
    parts: List[pd.DataFrame] = []
    for r in df[df["arm"].isin(list(arms))].itertuples():
        ck = _read_csv(Path(r.run_dir) / "ms_checks_stage2.csv")
        if ck is None or ck.empty:
            continue
        ck = ck[pd.to_numeric(ck["local"], errors="coerce") >= win.traj_from].copy()
        ck["status"] = r.status
        parts.append(ck.reindex(columns=TRAJ_COLS))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=TRAJ_COLS)


def segment_table(df: pd.DataFrame, win: Windows, arms: Sequence[str] = NL_ARMS,
                  qs: Sequence[int] = QS) -> pd.DataFrame:
    """``segments.csv``: per (arm, q, segment) over the complete runs.

    Segments (local updates of the terminal stage): training 1..ramp_first - 1, ramp ramp_first..ramp_last, hold, decay. Columns: the means over
    runs (of each run's mean over the segment's updates) of the PPO diagnostics of ``ms_updates.csv`` (``kl_final_epoch``,
    ``clip_frac``, ``adv_raw_std``, ``value_loss``, ``mean_effort``), the applied concentration scale (mean, min, max
    over all updates of the segment), and the segment-end values of e_hat_2(0), sigma_2(0), the smoothing part and the
    remainder: the check row with the largest local update inside the segment (``end_check_local``; checks are every K
    updates, so the last check of a segment need not be its last update), mean over runs.
    """
    nan = float("nan")
    cols = (["arm", "q", "segment", "first_local", "last_local", "n_runs", "n_updates_mean"]
            + ["%s_mean" % c for c in UPDATE_DIAG_COLS] + ["conc_scale_mean", "conc_scale_min", "conc_scale_max",
                                                           "end_check_local", "e_hat_2_0_end", "sigma_0_end",
                                                           "smoothing_end", "remainder_end", "status"])
    rows: List[Dict[str, Any]] = []
    for arm in arms:
        for q in qs:
            runs = df[(df["arm"] == arm) & (df["q"] == q) & df["complete"].astype(bool)].sort_values("seed")
            data = []
            for r in runs.itertuples():
                data.append((_read_csv(Path(r.run_dir) / "ms_updates.csv"),
                             _read_csv(Path(r.run_dir) / "ms_checks_stage2.csv")))
            for name, first, last in win.segments():
                acc: Dict[str, List[float]] = {c: [] for c in UPDATE_DIAG_COLS}
                scale_vals: List[np.ndarray] = []
                n_upd: List[float] = []
                ends: Dict[str, List[float]] = {c: [] for c in ("e2_at_0", "sigma_0", "smoothing", "remainder",
                                                                 "local")}
                for upd, chk in data:
                    if upd is not None and not upd.empty:
                        u = upd[(upd["stage"] == 2) & (upd["local"] >= first) & (upd["local"] <= last)]
                        n_upd.append(float(len(u)))
                        for c in UPDATE_DIAG_COLS:
                            acc[c].append(float(pd.to_numeric(u[c], errors="coerce").mean()) if len(u) else nan)
                        if "conc_scale" in u.columns:
                            scale_vals.append(pd.to_numeric(u["conc_scale"], errors="coerce").to_numpy())
                    if chk is not None and not chk.empty:
                        loc = pd.to_numeric(chk["local"], errors="coerce")
                        c_in = chk[(loc >= first) & (loc <= last)]
                        if len(c_in):
                            row_end = c_in.iloc[int(np.argmax(pd.to_numeric(c_in["local"], errors="coerce")
                                                              .to_numpy()))]
                            for c in ends:
                                ends[c].append(_f(row_end.get(c)))
                sv = np.concatenate(scale_vals) if scale_vals else np.array([])
                sv = sv[np.isfinite(sv)]
                row = {"arm": arm, "q": q, "segment": name, "first_local": first, "last_local": last,
                       "n_runs": len(runs), "n_updates_mean": float(np.nanmean(n_upd)) if n_upd else nan,
                       "conc_scale_mean": float(sv.mean()) if sv.size else nan,
                       "conc_scale_min": float(sv.min()) if sv.size else nan,
                       "conc_scale_max": float(sv.max()) if sv.size else nan,
                       "end_check_local": float(np.nanmedian(ends["local"])) if ends["local"] else nan,
                       "e_hat_2_0_end": float(np.nanmean(ends["e2_at_0"])) if ends["e2_at_0"] else nan,
                       "sigma_0_end": float(np.nanmean(ends["sigma_0"])) if ends["sigma_0"] else nan,
                       "smoothing_end": float(np.nanmean(ends["smoothing"])) if ends["smoothing"] else nan,
                       "remainder_end": float(np.nanmean(ends["remainder"])) if ends["remainder"] else nan,
                       "status": "descriptive"}
                for c in UPDATE_DIAG_COLS:
                    row["%s_mean" % c] = float(np.nanmean(acc[c])) if acc[c] and np.isfinite(acc[c]).any() else nan
                rows.append(row)
    return pd.DataFrame(rows, columns=cols)


FREEZE_QUANTITIES: Tuple[str, ...] = ("gap", "smoothing", "remainder", "sigma_2_0", "gap_rel", "smoothing_rel",
                                      "remainder_rel", "smoothing_over_formula")


def freeze_decomposition_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """``freeze_decomposition.csv``: per (arm, q) and quantity, over the complete runs, n / mean / median / SD / min /
    max and the percentile bootstrap interval of the mean (effort units for gap, smoothing, remainder, sigma; the
    ``*_rel`` rows are relative to e2*(0)). Rows without a finite value are omitted."""
    rows = []
    for arm in arms:
        for q in qs:
            g = df[(df["arm"] == arm) & (df["q"] == q) & df["complete"].astype(bool)]
            for qty in FREEZE_QUANTITIES:
                v = pd.to_numeric(g[qty], errors="coerce").dropna().to_numpy()
                if v.size == 0:
                    continue
                rows.append({"arm": arm, "q": q, "quantity": qty,
                             "unit": "relative to e2*(0)" if qty.endswith("_rel") else (
                                 "ratio" if qty == "smoothing_over_formula" else "effort units"),
                             **summary_stats(v), "status": "descriptive"})
    return pd.DataFrame(rows)


def predictions_table(df: pd.DataFrame, qs: Sequence[int], arms: Sequence[str] = NL_ARMS) -> pd.DataFrame:
    """``predictions.csv`` (D5 predictions, descriptive) per NL arm and q over the complete runs.

    sigma_2(0) at the freeze; the per-run ratio ``sigma(s) / (sigma(s = 1 of the same sampler, same seed) / sqrt(s))``;
    ``smoothing_over_formula`` (expected within 0.5 % of 1) and the floor in percent of e2*(0), ``100 * smoothing /
    g2_at_0``; ``planning_floor_pct`` is the planning value of D5 for comparison.
    """
    nan = float("nan")
    rows = []
    for arm in arms:
        sampler, s = parse_arm(arm)
        for q in qs:
            g = df[(df["arm"] == arm) & (df["q"] == q) & df["complete"].astype(bool)].set_index("seed")
            base = df[(df["arm"] == "NL_%s_s1" % sampler) & (df["q"] == q) & df["complete"].astype(bool)
                      ].set_index("seed")["sigma_2_0"]
            ratios = []
            for sd in g.index:
                if sd in base.index and np.isfinite(g.loc[sd, "sigma_2_0"]) and np.isfinite(base.loc[sd]):
                    ratios.append(float(g.loc[sd, "sigma_2_0"] / (base.loc[sd] / math.sqrt(s))))
            sig = pd.to_numeric(g["sigma_2_0"], errors="coerce").dropna()
            sof = pd.to_numeric(g["smoothing_over_formula"], errors="coerce").dropna()
            floor = 100.0 * pd.to_numeric(g["smoothing"], errors="coerce") / pd.to_numeric(g["g2_at_0"],
                                                                                           errors="coerce")
            floor = floor.dropna()
            ra = np.asarray(ratios, dtype=float)

            def stat(x: Any, fn: str) -> float:
                return float(getattr(np, fn)(x)) if len(x) else nan
            rows.append({
                "arm": arm, "sampler": sampler, "s": int(s), "q": q, "n": int(len(g)),
                "sigma_2_0_mean": stat(sig, "mean"), "sigma_2_0_median": stat(sig, "median"),
                "sigma_2_0_min": stat(sig, "min"), "sigma_2_0_max": stat(sig, "max"),
                "sigma_ratio_n": int(ra.size), "sigma_ratio_mean": stat(ra, "mean"),
                "sigma_ratio_median": stat(ra, "median"), "sigma_ratio_min": stat(ra, "min"),
                "sigma_ratio_max": stat(ra, "max"),
                "smoothing_over_formula_n": int(sof.size), "smoothing_over_formula_mean": stat(sof, "mean"),
                "smoothing_over_formula_min": stat(sof, "min"), "smoothing_over_formula_max": stat(sof, "max"),
                "n_outside_0p5pct": int((np.abs(sof - 1.0) > 0.005).sum()),
                "floor_pct_mean": stat(floor, "mean"), "floor_pct_min": stat(floor, "min"),
                "floor_pct_max": stat(floor, "max"),
                "planning_sigma_2_0": PLANNING_SIGMA / math.sqrt(s),
                "planning_floor_pct": 100.0 * (PLANNING_SIGMA / math.sqrt(s)) / (math.sqrt(math.pi) * q),
                "status": "descriptive", "note": PLANNING_FLOOR_NOTE})
    return pd.DataFrame(rows)


def _count_defined(s: pd.Series) -> Any:
    """Number of True values; ``None`` (missing) when no value of the series is defined (a quantity the arm does not
    have)."""
    if not any(isinstance(v, (bool, np.bool_)) or (isinstance(v, float) and not math.isnan(v)) for v in s):
        return None
    return int(sum(1 for v in s if _truthy(v)))


GATE_COLS: Tuple[Tuple[str, str], ...] = (
    ("G_A_eta_pass", "n_G_A_eta"), ("G_A_rmse_pass", "n_G_A_rmse"), ("G_A_tail_pass", "n_G_A_tail"),
    ("G_A_pass", "n_G_A"), ("G_N_eta_pass", "n_G_N_eta"), ("gate_pass", "n_G_A_and_G_N_eta"),
    ("G_F_pass", "n_G_F"), ("G_N_gmax_pass", "n_G_N_gmax"), ("G_S_pass", "n_G_S"), ("S1_pass", "n_S1"),
    ("v20_combination_pass", "n_v20_combination"))


def gates_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """``gates.csv``: per (arm, q) the numbers of complete runs that pass each gate (D7; descriptive): G-A (and its
    three parts) and G-N(eta) on the final tier at the terminal freeze; G-F, G-N(Gmax), G-S and S1 at the end of stage
    1; the v2.0 combination. NaN where the row's runs do not have the quantity (``parents_A``: stage 1 untrained)."""
    rows = []
    for arm in list(arms) + [PARENTS, REHEARSAL]:
        for q in qs:
            g = df[(df["arm"] == arm) & (df["q"] == q)]
            done = g[g["complete"].astype(bool)]
            r: Dict[str, Any] = {"arm": arm, "q": q, "n_runs": int(len(g)), "n_done": int(len(done))}
            for col, name in GATE_COLS:
                r[name] = _count_defined(done[col])
            r["status"] = "descriptive"
            rows.append(r)
    out = pd.DataFrame(rows)
    for _, name in GATE_COLS:
        out[name] = out[name].astype("Int64")
    return out


# --------------------------------------------------------------------------------------------- figures
def _footer(fig: Any, roots: Mapping[str, str]) -> None:
    """Provenance footer: the roots and the bootstrap setting."""
    txt = ("roots: MS-R2 pilot=%s | MS-R1 pilot=%s | parents_A=%s | rehearsal_v2_0=%s | bootstrap %d resamples, "
           "default_rng(%d)" % (roots["pilot"], roots["ms_r1_pilot"], roots["parents"], roots["rehearsal"], N_BOOT,
                               BOOT_SEED))
    fig.text(0.005, 0.003, txt, fontsize=5, ha="left", va="bottom", wrap=True)


def _paired_axis(ax: Any, seed_df: pd.DataFrame, summ: pd.DataFrame, comparison: str, metric: str,
                 arms: Sequence[str], q: int, labels: Mapping[str, str]) -> None:
    """One panel: per arm the seed dots, the mean and its 95% percentile bootstrap interval of a paired difference."""
    for i, arm in enumerate(arms):
        sv = seed_df[(seed_df["comparison"] == comparison) & (seed_df["arm"] == arm) & (seed_df["q"] == q)
                     & (seed_df["metric"] == metric)]
        if len(sv):
            jit = np.linspace(-0.18, 0.18, len(sv))
            ax.scatter(sv["diff"], np.full(len(sv), float(i)) + jit, s=12, alpha=0.55, color="tab:blue")
        rw = summ[(summ["comparison"] == comparison) & (summ["arm"] == arm) & (summ["q"] == q)
                  & (summ["metric"] == metric)]
        if len(rw) and np.isfinite(rw["mean"].iloc[0]) and np.isfinite(rw["ci_mean_lo"].iloc[0]):
            m, lo, hi = float(rw["mean"].iloc[0]), float(rw["ci_mean_lo"].iloc[0]), float(rw["ci_mean_hi"].iloc[0])
            ax.errorbar([m], [i], xerr=[[m - lo], [hi - m]], fmt="D", color="k", capsize=3, ms=5, lw=1.4)
    ax.axvline(0.0, color="tab:red", lw=0.8, ls="--")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([labels.get(a, a) for a in arms], fontsize=7)
    ax.invert_yaxis()
    ax.grid(alpha=0.25)


def fig_paired(path: Path, seed_df: pd.DataFrame, summ: pd.DataFrame, comparison: str, metrics: Sequence[Tuple[str, str]],
               arms: Sequence[str], qs: Sequence[int], title: str, roots: Mapping[str, str],
               labels: Optional[Mapping[str, str]] = None) -> None:
    """Paired differences per arm (seed dots, mean, 95% bootstrap interval): one row per metric, one column per q."""
    plt = R1._plt()
    labels = labels or {}
    fig, axes = plt.subplots(len(metrics), len(qs), figsize=(5.6 * len(qs), (0.45 * len(arms) + 1.6) * len(metrics)
                                                             + 0.8), squeeze=False)
    for i, (metric, mlabel) in enumerate(metrics):
        for j, q in enumerate(qs):
            ax = axes[i][j]
            _paired_axis(ax, seed_df, summ, comparison, metric, arms, q, labels)
            ax.set_title("q = %d: %s" % (q, mlabel), fontsize=8)
            if i == len(metrics) - 1:
                ax.set_xlabel("paired difference (arm - baseline)", fontsize=7)
    fig.suptitle(textwrap.fill(title, 120), fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    _footer(fig, roots)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_trajectories(path: Path, traj: pd.DataFrame, win: Windows, qs: Sequence[int], roots: Mapping[str, str]) -> int:
    """e_hat_2(0), sigma_2(0), the smoothing part and the remainder against the update (``traj_from`` ..): mean over the
    complete runs with a min-max band, one line per arm (colour = s, dashes = stratified), ramp / hold / decay shaded.

    Returns:
        Number of (arm, q) lines drawn.
    """
    plt = R1._plt()
    quantities = [("e2_at_0", "e_hat_2(0) (effort)"), ("sigma_0", "sigma_2(0) (effort)"),
                  ("smoothing", "smoothing part (effort)"), ("remainder", "remainder (effort)")]
    fig, axes = plt.subplots(len(quantities), len(qs), figsize=(6.0 * len(qs), 2.5 * len(quantities)), sharex=True,
                             squeeze=False)
    n_lines = 0
    for i, (col, lab) in enumerate(quantities):
        for j, q in enumerate(qs):
            ax = axes[i][j]
            for (name_, a_, b_), shade in zip(win.segments()[1:], ("0.86", "0.95", "0.78")):
                ax.axvspan(a_ - 1, b_, color=shade, lw=0)
                if i == 0:
                    ax.text(0.5 * (a_ + b_), 1.01, name_, transform=ax.get_xaxis_transform(), fontsize=6,
                            ha="center", va="bottom")
            for arm in NL_ARMS:
                sampler, s = parse_arm(arm)
                g = traj[(traj["arm"] == arm) & (traj["q"] == q)]
                if g.empty:
                    continue
                agg = g.groupby("local")[col].agg(["mean", "min", "max"]).dropna()
                if agg.empty:
                    continue
                ax.plot(agg.index, agg["mean"], color=SCALE_COLORS[int(s)], ls=SAMPLER_STYLE[sampler], lw=1.1,
                        label=arm)
                ax.fill_between(agg.index, agg["min"], agg["max"], color=SCALE_COLORS[int(s)], alpha=0.10, lw=0)
                n_lines += 1
            if col == "e2_at_0" and "g2_0" in traj.columns:
                gg = pd.to_numeric(traj[traj["q"] == q]["g2_0"], errors="coerce").dropna()
                if len(gg):
                    ax.axhline(float(gg.iloc[0]), color="k", lw=0.6, ls=":")
            ax.set_title("q = %d" % q, fontsize=8)
            ax.set_ylabel(lab, fontsize=7)
            ax.grid(alpha=0.2)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, ncol=2, loc="best")
            if i == len(quantities) - 1:
                ax.set_xlabel("terminal-stage local update", fontsize=7)
    fig.suptitle(textwrap.fill("Descriptive: the tie decomposition along the landing (mean over seeds, band = min-max); "
                               "shaded = ramp / hold / decay; colour = s (blue 1, orange 4, red 16), dashed = "
                               "stratified starts; dotted = e2*(0)", 130), fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _footer(fig, roots)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return n_lines


def fig_change_scatter(path: Path, seed_level: pd.DataFrame, qs: Sequence[int], roots: Mapping[str, str]) -> int:
    """Per-run change of the remainder against the change of the smoothing part (arm - s=1 of the same sampler, per
    (q, seed)), marked by arm, with the line on which the two cancel (change of the remainder = - change of the
    smoothing part: no change of the gap).

    Returns:
        Number of points drawn.
    """
    plt = R1._plt()
    fig, axes = plt.subplots(1, len(qs), figsize=(5.6 * len(qs), 5.2), squeeze=False)
    markers = {"NL_bb_s4": "o", "NL_bb_s16": "s", "NL_st_s4": "^", "NL_st_s16": "D"}
    n_pts = 0
    for j, q in enumerate(qs):
        ax = axes[0][j]
        sm = seed_level[(seed_level["comparison"] == LABEL_S1) & (seed_level["q"] == q)
                        & (seed_level["metric"] == "smoothing")].set_index(["arm", "seed"])["diff"]
        rm = seed_level[(seed_level["comparison"] == LABEL_S1) & (seed_level["q"] == q)
                        & (seed_level["metric"] == "remainder")].set_index(["arm", "seed"])["diff"]
        both = sm.index.intersection(rm.index)
        for arm in markers:
            idx = [k for k in both if k[0] == arm]
            if not idx:
                continue
            sampler, s = parse_arm(arm)
            ax.scatter([sm.loc[k] for k in idx], [rm.loc[k] for k in idx], marker=markers[arm], s=26, alpha=0.75,
                       color=SCALE_COLORS[int(s)], edgecolor="k" if sampler == "st" else "none", label=arm)
            n_pts += len(idx)
        px = np.array([sm.loc[k] for k in both], dtype=float)
        py = np.array([rm.loc[k] for k in both], dtype=float)
        if px.size:
            lo, hi = min(px.min(), -py.max(), 0.0), max(px.max(), -py.min(), 0.0)
            pad = 0.08 * (hi - lo if hi > lo else 1.0)
            xs = np.array([lo - pad, hi + pad])
        else:
            xs = np.array([-1.0, 1.0])
        ax.plot(xs, -xs, color="tab:red", lw=0.9, ls="--", label="change of the gap = 0")
        ax.axhline(0.0, color="0.6", lw=0.5)
        ax.axvline(0.0, color="0.6", lw=0.5)
        ax.set_title("q = %d" % q, fontsize=9)
        ax.set_xlabel("change of the smoothing part (arm - s=1), effort units", fontsize=8)
        ax.set_ylabel("change of the remainder (arm - s=1), effort units", fontsize=8)
        ax.grid(alpha=0.2)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Descriptive: per-run change of the remainder against the change of the smoothing part; on the dashed "
                 "line the two cancel (no change of the gap)", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    _footer(fig, roots)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return n_pts


# --------------------------------------------------------------------------------------------- summary text
def _g(x: Any) -> str:
    """Compact signed number."""
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else "%+.5f" % float(x)


def summary_text(roots: Mapping[str, str], crit: pd.DataFrame, trans: pd.DataFrame, inter: pd.DataFrame,
                 comp: pd.DataFrame, qs: Sequence[int], seeds: Sequence[int],
                 command: Optional[Sequence[str]] = None) -> str:
    """The CLI summary (also ``summary.txt``): provenance, the criterion, the transmission and interaction headline
    numbers, the run status.

    Args:
        roots: The resolved run roots.
        crit: ``criterion.csv``.
        trans: ``transmission.csv``.
        inter: ``interaction.csv`` (the summary table).
        comp: ``completeness.csv``.
        qs: q values.
        seeds: Development seeds.
        command: Command line to record.
    """
    L: List[str] = ["MS-R2 analysis (prompt D4, D5, section 3.3)"]
    if command:
        L.append("command: " + " ".join(str(c) for c in command))
    L.append("independent check: python tools/ms/r2_blind_criterion.py --analysis-dir <the --out directory>")
    L.append("roots: MS-R2 pilot=%s | MS-R1 pilot=%s | parents_A=%s | rehearsal_v2_0=%s" % (
        roots["pilot"], roots["ms_r1_pilot"], roots["parents"], roots["rehearsal"]))
    L.append("bootstrap: %d resamples, numpy.random.default_rng(%d), one fresh generator per (q, statistic)" %
             (N_BOOT, BOOT_SEED))
    L.append("")
    L.append("== PRE-REGISTERED CRITERION: %s ==" % CRITERION_NOTE)
    L.append("primary metric %s of the frozen terminal-stage candidate; (a) CI of the mean paired difference "
             "s - (same sampler, s = 1) below 0 at BOTH q; (b) no run passing G-A with its G-N eta part under s = 1 "
             "fails it under s" % PRIMARY)
    lines, any_ci0 = R1._criterion_lines(crit, qs, seeds)
    L.extend(lines)
    if any_ci0:
        L.append("NOTE: " + NO_EFFECT_SENTENCE)
    L.append("")
    L.append("== TRANSMISSION (descriptive): mean(d gap) / mean(d smoothing), d = arm - s=1 of the same sampler; "
             "1 = the smoothing change reaches the gap, 0 = the remainder offsets it ==")
    for r in trans.itertuples():
        L.append("%-10s q=%d n=%d  d gap %s  d smoothing %s  ratio %s [%s, %s]  (%d resamples kept)" % (
            r.arm, r.q, r.n_pairs, _g(r.mean_d_gap), _g(r.mean_d_smoothing), _g(r.ratio), _g(r.ci_lo), _g(r.ci_hi),
            r.n_boot_valid))
    L.append("")
    L.append("== INTERACTION (descriptive): (NL_st_s - NL_st_s1) - (NL_bb_s - NL_bb_s1) per (q, seed) ==")
    for r in inter.itertuples():
        L.append("s=%-2d q=%d %-24s n=%d  mean %s [%s, %s]  median %s" % (
            r.s, r.q, r.metric, r.n_pairs, _g(r.mean), _g(r.ci_mean_lo), _g(r.ci_mean_hi), _g(r.median)))
    L.append("")
    L.append("== RUN STATUS ==")
    bad = comp[(comp["n_done"] != comp["n_planned"])]
    if bad.empty:
        L.append("every planned run of every arm and reference is done")
    for r in bad.itertuples():
        L.append("%s q=%d: planned %d, done %d, failed %d, running %d, incomplete %d, missing %d -> %s" % (
            r.arm, r.q, r.n_planned, r.n_done, r.n_failed, r.n_running, r.n_incomplete, r.n_missing,
            r.not_done_runs))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------- driver
def not_stored_by_design() -> Dict[str, List[str]]:
    """Quantities that a reference row does not have (NaN in ``per_run.csv``), by reference."""
    return {
        PARENTS: ["smoothing, remainder, smoothing_over_formula and smoothed_e_pred_0 (final_v2.json carries no "
                  "smoothed_game; NaN; gap is defined)", "conc_scale_final (no noise record; NaN)",
                  "D3 columns t2_R_*, t2_Rtail_*, t2_C_*, t2_s_* (no one-step best-response effort stored; NaN unless "
                  "--calibration-root)", "every stage-1 column (stage 1 untrained)",
                  "rule record, start shares, clamp counts (NaN)"],
        REHEARSAL: ["conc_scale_final (no noise record; NaN)", "D3 columns t*_R_*, t*_C_*, t*_s_* (NaN unless "
                    "--calibration-root)", "rule record, start shares, clamp counts (NaN)",
                    "t*_minibatch_steps (not stored in v2_run_summary.json; NaN)",
                    "t*_episodes = updates x episodes_per_update (derived)"],
        "MS_base2400": ["conc_scale_final (MS-R1 runs carry no noise record; NaN)", "the per-check decomposition "
                        "(trajectory_checks.csv, segments.csv cover the six NL arms only)"],
        "MS_s35a5": ["conc_scale_final (MS-R1 runs carry no noise record; NaN)", "the per-check decomposition "
                     "(trajectory_checks.csv, segments.csv cover the six NL arms only)"]}


def run_analysis(roots: Mapping[str, str], out_dir: Path, qs: Sequence[int] = QS, seeds: Sequence[int] = SEEDS,
                 protocol: Path = R1.DEFAULT_PROTOCOL, win: Windows = Windows(), figures: bool = True,
                 argv: Optional[Sequence[str]] = None) -> Tuple[Dict[str, pd.DataFrame], bool]:
    """Extract, tabulate and write every output.

    Args:
        roots: ``{pilot, ms_r1_pilot, parents, rehearsal[, calibration]}`` run roots.
        out_dir: Output directory.
        qs: q values.
        seeds: Development seeds.
        protocol: ``protocols/v2_T2_locked_v2_0.json`` (the gate thresholds).
        win: Segment boundaries and the first update of the trajectory table.
        figures: Whether to draw the figures.
        argv: Command line recorded in ``analysis_info.json``.

    Returns:
        ``(tables, all_done)`` with ``all_done`` True when every planned run of every arm and reference is done.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    th, proto_sha = R1.load_thresholds(protocol)
    proto = json.loads(Path(protocol).read_text())
    rid = R1.roots_id(roots)
    df = extract_all(roots, qs, seeds, th)
    arms_r1 = list(NL_ARMS) + list(REF_MS_ARMS)           # the arms the MS-R1 tables loop over (+ the two comparators)
    cmp_all_parents = [(a, PARENTS) for a in NL_ARMS]
    cmp_rehearsal = [(a, REHEARSAL) for a in NL_ARMS]
    # ---- the pre-registered criterion and the secondary tables (all with the MS-R2 bootstrap seed)
    p_s1, sl_s1 = paired_tables(df, S1_COMPARISONS, SECONDARY_METRICS, qs, seeds, LABEL_S1)
    crit = criterion_table(df, p_s1, S1_COMPARISONS, qs, seeds, LABEL_S1, CRITERION_NOTE)
    p_par, sl_par = paired_tables(df, cmp_all_parents, REF_METRICS, qs, seeds, LABEL_PARENTS)
    crit_par = criterion_table(df, p_par, cmp_all_parents, qs, seeds, LABEL_PARENTS, PARENTS_NOTE)
    p_ref, sl_ref = paired_tables(df, MS_R1_REF_COMPARISONS, REF_METRICS, qs, seeds, LABEL_REFS)
    p_reh, sl_reh = paired_tables(df, cmp_rehearsal, R1.S1_METRICS, qs, seeds, LABEL_REHEARSAL)
    seed_level = pd.concat([sl_s1, sl_par, sl_ref, sl_reh], ignore_index=True)
    trans, trans_seed = transmission_tables(df, S1_COMPARISONS, qs, seeds)
    inter, inter_seed = interaction_tables(df, qs, seeds)
    comp = R1.completeness_table(df, arms_r1, qs, seeds)
    # ---- decomposition along the run
    traj = trajectory_table(df, win)
    traj_ok = traj[traj["status"] == "done"] if len(traj) else traj
    strata = R1.strata_table(df)
    with r2_bootstrap():
        arm_summary = R1.arm_summary_table(df, arms_r1, qs)
        stage1 = R1.stage1_table(df, list(NL_ARMS) + list(REF_MS_ARMS), qs, th)
    tables: Dict[str, pd.DataFrame] = {
        "per_run": df, "criterion": crit, "paired_secondary": p_s1, "paired_seed_level": seed_level,
        "transmission": trans, "transmission_seed_level": trans_seed, "interaction": inter,
        "interaction_seed_level": inter_seed, "paired_vs_parents_A": p_par, "criterion_vs_parents_A": crit_par,
        "paired_vs_ms_r1_refs": p_ref, "paired_vs_rehearsal_v2_0": p_reh,
        "predictions": predictions_table(df, qs), "trajectory_checks": traj,
        "segments": segment_table(df, win, NL_ARMS, qs),
        "freeze_decomposition": freeze_decomposition_table(df, list(NL_ARMS) + list(REF_ARMS), qs),
        "strata": strata, "strata_summary": R1.strata_summary_table(strata),
        "r0": R1.r0_table(df, arms_r1, qs, proto), "stage1": stage1,
        "stage1_R1_runs": R1.stage1_R1_runs_table(df, arms_r1), "stage1_R1": R1.stage1_R1_table(df, arms_r1, qs),
        "gates": gates_table(df, arms_r1, qs), "arm_summary": arm_summary,
        "budget": R1.budget_table(df, arms_r1, qs), "completeness": comp}
    for name, t in tables.items():
        R1.write_csv(t, out_dir / ("%s.csv" % name), rid)
    # ---- figures (a failing figure is recorded, not raised)
    fig_status: Dict[str, str] = {}
    if figures:
        fdir = out_dir / "figures"
        fdir.mkdir(exist_ok=True)
        lab = {a: "%s (vs %s)" % (a, b) for a, b in S1_COMPARISONS}
        jobs = [
            ("paired_abs_peak_vs_s1.png", lambda p: fig_paired(
                p, sl_s1, p_s1, LABEL_S1, [(PRIMARY, "|peak error|, criterion statistic")],
                [a for a, _ in S1_COMPARISONS], qs,
                "Primary criterion part (a) (pre-registered statistic; the figure is descriptive): paired difference of "
                "|peak error|, arm - s=1 of the same sampler; dots = seeds, diamond = mean, bar = 95% percentile "
                "bootstrap interval", roots, lab)),
            ("paired_secondary_vs_s1.png", lambda p: fig_paired(
                p, sl_s1, p_s1, LABEL_S1, [("smoothing", "smoothing part"), ("remainder", "remainder"),
                                           ("gap", "gap"), ("sigma_2_0", "sigma_2(0)"),
                                           (SIGNED, "signed peak error")],
                [a for a, _ in S1_COMPARISONS], qs,
                "Descriptive: paired differences of the decomposition (effort units) and of the signed peak error, "
                "arm - s=1 of the same sampler", roots, lab)),
            ("paired_abs_peak_vs_parents_A.png", lambda p: fig_paired(
                p, sl_par, p_par, LABEL_PARENTS, [(PRIMARY, "|peak error|")], list(NL_ARMS), qs,
                "Descriptive: paired difference of |peak error|, arm - parents_A", roots)),
            ("trajectory_decomposition.png", lambda p: fig_trajectories(p, traj_ok, win, qs, roots)),
            ("scatter_remainder_vs_smoothing_change.png", lambda p: fig_change_scatter(p, sl_s1, qs, roots))]
        for fname, fn in jobs:
            try:
                fn(fdir / fname)
                fig_status[fname] = "ok"
            except Exception as exc:  # noqa: BLE001
                fig_status[fname] = "FAILED: %s: %s" % (type(exc).__name__, exc)
                print("[warn] figure %s failed: %s: %s" % (fname, type(exc).__name__, exc), file=sys.stderr)
    all_done = bool((comp["n_done"] == comp["n_planned"]).all())
    try:
        import matplotlib
        mpl_version = matplotlib.__version__
    except Exception:  # noqa: BLE001
        mpl_version = None
    info = {
        "tool": "tools/ms/r2_analysis.py", "roots": dict(roots), "roots_id": rid, "protocol": str(protocol),
        "protocol_sha256": proto_sha, "thresholds": th.__dict__, "boot_seed": BOOT_SEED, "n_boot": N_BOOT,
        "bootstrap_scheme": "fresh numpy.random.default_rng(seed) per (q, statistic); idx = rng.integers(0, n, "
                            "size=(n_boot, n)); 2.5 / 97.5 percentiles of the resampled mean (or median); the "
                            "transmission ratio is recomputed on every resample (resamples with |mean d smoothing| <= "
                            "%g are dropped)" % DENOM_TOL,
        "qs": list(qs), "seeds": list(seeds), "arms": list(NL_ARMS), "references": list(REF_ARMS),
        "windows": {"ramp_first": win.ramp_first, "ramp_last": win.ramp_last, "hold_last": win.hold_last,
                    "decay_last": win.decay_last, "traj_from": win.traj_from,
                    "segments": [list(x) for x in win.segments()]},
        "criterion": {
            "primary_metric": PRIMARY, "comparisons": [list(c) for c in S1_COMPARISONS],
            "a": "ci_mean_hi < 0 of the mean paired difference s - (same sampler, s = 1) at BOTH q",
            "b": "no run passing G-A (final tier) with the eta part of G-N (|eta_dev - eta_final| <= %g) under s = 1 "
                 "fails it under s" % th.n_eta, "note": CRITERION_NOTE,
            "parts_per_arm": {r.arm: {"a_met": bool(r.a_met), "b_status": r.b_status, "overall": r.overall}
                              for r in crit.itertuples()}},
        "decomposition": "gap = g2_at_0 - e2_at_0; smoothing = g2_at_0 - smoothed_e_pred_0; remainder = "
                         "smoothed_e_pred_0 - e2_at_0 (final tier, utils.ms_noise.decompose_gap); *_rel = / g2_at_0; "
                         "smoothing_over_formula = smoothing / (g2_at_0 * sigma_effort_at_0_t2 / (sqrt(pi) * q))",
        "noise_record_check": "NL runs: the decomposition against rule_log.json stages[2].freeze.noise (absolute %g for "
                              "gap, smoothing, remainder, e_sigma, e2*(0), e_hat; relative %g for sigma and the "
                              "smoothing / formula ratio, whose per-run sigma is the verifier's float32 value); a "
                              "disagreement is a flag in the row's flags column" % (NOISE_TOL, SIGMA_RTOL),
        "transmission": "ratio = mean(d gap) / mean(d smoothing), d = arm - s=1 of the same sampler (the PI's "
                        "-(change of the gap) / (change of the smoothing part) read as reduction over reduction: 1 = "
                        "the whole smoothing change reaches the gap, 0 = the remainder offsets it)",
        "status_counts": {a: {s: int(((df["arm"] == a) & (df["status"] == s)).sum()) for s in R1.STATUSES}
                          for a in list(NL_ARMS) + list(REF_ARMS)},
        "all_done": all_done, "figures": fig_status, "command": list(argv) if argv is not None else None,
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "matplotlib": mpl_version, "not_stored_by_design": not_stored_by_design()}
    with open(out_dir / "analysis_info.json", "w") as f:
        json.dump(info, f, indent=1, sort_keys=True)
    (out_dir / "summary.txt").write_text(summary_text(roots, crit, trans, inter, comp, qs, seeds, argv))
    return tables, all_done


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point (see the module docstring)."""
    p = argparse.ArgumentParser(description="MS-R2 analysis (D4, D5, section 3.3)")
    p.add_argument("--pilot-root", default=str(ROOT / "results" / "ms_r2" / "pilot"))
    p.add_argument("--ms-r1-pilot-root", required=True, help="MS-R1 pilot root (MS_base2400, MS_s35a5)")
    p.add_argument("--parents-root", required=True)
    p.add_argument("--rehearsal-root", required=True)
    p.add_argument("--calibration-root", default=None,
                   help="results/ms_r1/calibration: fills the D3 quantities of the two v2.0 comparators")
    p.add_argument("--out", default=str(ROOT / "results" / "ms_r2" / "analysis"))
    p.add_argument("--qs", type=int, nargs="+", default=list(QS))
    p.add_argument("--seeds", nargs="+", default=["%d-%d" % (SEEDS[0], SEEDS[-1])])
    p.add_argument("--protocol", default=str(R1.DEFAULT_PROTOCOL))
    p.add_argument("--ramp-first", type=int, default=Windows.ramp_first)
    p.add_argument("--ramp-last", type=int, default=Windows.ramp_last)
    p.add_argument("--hold-last", type=int, default=Windows.hold_last)
    p.add_argument("--decay-last", type=int, default=Windows.decay_last)
    p.add_argument("--traj-from", type=int, default=Windows.traj_from)
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    win = Windows(a.ramp_first, a.ramp_last, a.hold_last, a.decay_last, a.traj_from)
    if not win.valid():
        print("[error] the segment boundaries need 1 <= ramp-first < ramp-last <= hold-last <= decay-last",
              file=sys.stderr)
        return 2
    roots = {"pilot": str(Path(a.pilot_root).resolve()), "ms_r1_pilot": str(Path(a.ms_r1_pilot_root).resolve()),
             "parents": str(Path(a.parents_root).resolve()), "rehearsal": str(Path(a.rehearsal_root).resolve())}
    if a.calibration_root:
        roots["calibration"] = str(Path(a.calibration_root).resolve())
    for k in ("pilot", "ms_r1_pilot", "parents", "rehearsal"):
        if not Path(roots[k]).is_dir():
            print("[error] %s root %s is not a directory (a missing reference root is a stop-and-report)" %
                  (k, roots[k]), file=sys.stderr)
            return 2
    tables, all_done = run_analysis(roots, Path(a.out), tuple(a.qs), R1._parse_seeds(a.seeds), Path(a.protocol), win,
                                    not a.no_figures, list(argv) if argv is not None else sys.argv)
    sys.stdout.write((Path(a.out) / "summary.txt").read_text())
    if not all_done:
        print("[incomplete] at least one planned run is missing / failed / running / incomplete "
              "(see completeness.csv)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
