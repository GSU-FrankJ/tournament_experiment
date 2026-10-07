#!/usr/bin/env python3
"""MS-R3 pre-registered analysis of the actor-resolution pilot (prompt D5, D6 and section 3.3).

Arms (D4): ``{actor}_{bb|st}_s{1|16}`` with actor in ``t1`` (the locked tanh actor, the control), ``relu`` and ``t10``
(``--actors`` names the actors that were run; ``t1`` is required; a planned run that is missing is a row with status
``missing`` and exit code 3, never a silent skip). Runs live at ``<pilot-root>/q<q>/seed<seed>/<arm>/``.

Inputs (all read only; every root is an explicit argument and is recorded in ``analysis_info.json``, ``summary.txt``,
the figure footers, a ``roots_id`` column of every CSV and the ``run_dir`` of every row of ``per_run.csv``):

  * ``--pilot-root``        the MS-R3 runs
  * ``--ms-r2-pilot-root``  ``q*/seed*/NL_{bb,st}_s{1,16}`` of MS-R2: reference rows (role ``comparator``) and the
                            metric-level context of check C-MS5 (each ``t1`` arm is a bit-identical re-run of its
                            ``NL_*`` arm)
  * ``--parents-root``      ``q*/seed*`` of v2.0 ``parents_A``;  ``--rehearsal-root``  ``q*/seed*`` of
                            ``rehearsal_v2_0``
  * ``--calibration-root``  optional, ``results/ms_r1/calibration`` (D3 columns of the two v2.0 comparators)

Per-run table (``per_run.csv``): everything of ``r1_analysis`` / ``r2_analysis`` (the decomposition ``gap = g2_at_0 -
e2_at_0 = smoothing + remainder`` with ``smoothing = g2_at_0 - smoothed_e_pred_0``, cross-checked against the
``rule_log.json`` noise record) plus the D5 additions, all evaluation / reporting only (the closed form never enters
training):

  * ``t2_R0_final`` / ``t2_R0_dev`` (R0 = r_2(0) / s_2 at the freeze, both tiers), ``t2_R0_over_peak_final`` /
    ``_dev`` = R0 / |peak| and ``linearised_R0_over_peak`` = 2k / (2k + a), a = DW / (4 q^2), DW = w_H - w_L, k from the
    protocol record;
  * ``peak_locfree_rel_err`` / ``peak_locfree_argmax_d``: the location-free peak. On the final-tier recovery grid
    (``recovery_d_grid`` / ``recovery_e2`` of ``freeze_stage2_final.npz``, step 0.5) j = argmax over the WHOLE grid of
    ``recovery_e2``; ``peak_locfree_rel_err = (recovery_e2[j] - g2_at_0) / g2_at_0`` (signed, negative = below the
    closed-form tie value, the sign of ``stage2_peak_rel_err_signed``) and
    ``peak_locfree_argmax_d = recovery_d_grid[j]`` (where the learned kink sits). This is the definition of
    ``run/run_ms_stagewise.py`` (``stage2_peak_locfree_*`` in ``gates.json``);
  * ``sym_err_max`` = max over the recovery nodes with |d| < 2q of |e_hat_2(d) - e_hat_2(-d)| (effort units; the grid is
    symmetric, e_hat_2(-d) is the reversed array), ``sym_err_max_rel`` = / g2_at_0, ``sym_err_argmax_abs_d`` = |d| of
    the maximum (the ``stage2_sym_err_max`` recorded in ``gates.json`` is the maximum over the whole grid);
  * ``tent_slope`` = g2_at_0 / (2q) and the effective rounding width ``w_eff = gap / tent_slope`` (units of d);
  * first-layer weights on the d input of the LAST terminal-stage weight export (update ``--decay-last``): column 1 of
    ``actor.l1.weight`` (column 0 is the stage feature); units of d / B, B = (e_max - e_min) + 2q, so the stored weight
    for ``t1`` / ``relu`` and TEN times the stored weight for ``t10`` (the variant is read from the export's
    ``actor_variant`` entry, absent = ``t1``): ``w1d_abs_max``, ``w1d_abs_q25/q50/q75/q90`` (quantiles of |w| over the
    hidden units), ``w1d_n_gt1`` (units with |w| > 1) and ``w1d_bend_d_min = B / max |w|`` (the smallest tanh bend scale
    in units of d). ``actor_variant_export`` is the variant read; a disagreement with the arm name or with
    ``run_config.json`` is a flag.

Primary criterion (D6; ``criterion.csv``; descriptive, not a gate, no selection follows; the decision is the PI's). One
row per v in {relu, t10} x starts in {bb, st} x s in {1, 16} (eight rows when both variants ran): |peak error| of the
frozen terminal-stage candidate (final tier, ``stage2_peak_rel_err_abs``) paired by (q, seed) against the ``t1`` arm
with the same starts and s. (a) The 95% percentile bootstrap interval of the mean paired difference lies below 0
(``ci_mean_hi < 0``, strict) at BOTH q. (b) No run that passes G-A with its G-N eta part under ``t1`` fails it under v
(the pass definition, thresholds from the protocol file, and the violated / pending / holds logic of
``r1_analysis.criterion_row``). Bootstrap: 10,000 resamples, ONE FRESH ``numpy.random.default_rng(20261008)`` per
(q, statistic), the resampled indices are ``rng.integers(0, n, size=(10000, n))``, the interval is the 2.5 / 97.5
percentiles of the resampled statistic. Everything that is not this criterion is descriptive; an interval that contains
0 with 10 seeds does not show that a mechanism has no effect.

Secondary tables (descriptive; paired by (q, seed), arm - baseline): ``paired_secondary.csv`` (v against ``t1``, same
starts and s), ``noise_landing.csv`` (s = 16 against s = 1 within each (actor, starts), with the transmission ratio),
``starts_effect.csv`` (st against bb within each (actor, s)), ``paired_vs_parents_A.csv`` /
``criterion_vs_parents_A.csv`` (every arm against ``parents_A`` with MS-R1's criterion),
``paired_vs_rehearsal_v2_0.csv`` (stage 1), ``stage1_vs_t1.csv``, ``paired_vs_ms_r2_nl.csv`` (each ``t1`` arm against
its MS-R2 ``NL_*`` re-run: the metric-level context of C-MS5); ``paired_seed_level.csv`` holds the per-seed values.

Transmission ratio (``transmission.csv``, also the ``metric == "transmission_ratio"`` rows of ``noise_landing.csv``;
descriptive): ``mean(d gap) / mean(d smoothing)`` with d = s16 - s1 within one (actor, starts), per q; 1 = the whole
smoothing change reaches the gap, 0 = the remainder offsets it (gap = smoothing + remainder, so the ratio is 1 + change
of the remainder / change of the smoothing part). Its interval resamples the seeds and recomputes the ratio on every
resample; a resample whose denominator mean is below :data:`DENOM_TOL` in absolute value is dropped from the percentiles
(``n_boot_valid`` counts the kept ones). Same sign convention as MS-R2's ``transmission.csv``. Interaction
(``interaction.csv``): (v_s16 - v_s1) - (t1_s16 - t1_s1) per (v, starts, q, seed) on |peak|, the gap, the remainder and
the smoothing part. Quadrature check (``quadrature_check.csv``): per (actor, starts, q), from ARM MEANS,
F^2 = gap(s1)^2 - smoothing(s1)^2 (negative: F = 0, flagged), quadrature prediction of gap(s16) =
sqrt(F^2 + smoothing(s16)^2), additive prediction gap(s1) - (smoothing(s1) - smoothing(s16)).

Outputs under ``--out`` (which must be empty or absent: an existing non-empty directory is refused; every CSV has
``roots_id`` as its first column; no timestamps, the tool is deterministic given the same inputs): ``per_run.csv``,
``criterion.csv``, ``paired_secondary.csv``, ``paired_seed_level.csv``, ``noise_landing.csv``, ``transmission.csv``,
``transmission_seed_level.csv``, ``interaction.csv``, ``interaction_seed_level.csv``, ``starts_effect.csv``,
``paired_vs_parents_A.csv``, ``criterion_vs_parents_A.csv``, ``paired_vs_rehearsal_v2_0.csv``,
``paired_vs_ms_r2_nl.csv``, ``stage1.csv``, ``stage1_vs_t1.csv``, ``stage1_R1_runs.csv``, ``stage1_R1.csv``,
``gates.csv``, ``budget.csv``, ``segments.csv``, ``trajectory_checks.csv``, ``trajectory_by_arm.csv``,
``quadrature_check.csv``, ``predictions.csv``, ``freeze_decomposition.csv``, ``strata.csv``, ``strata_summary.csv``,
``r0.csv``, ``r0_spearman.csv``, ``first_layer_weights.csv``, ``first_layer_summary.csv``,
``first_layer_units.csv``, ``tie_profile.csv``, ``arm_summary.csv``, ``completeness.csv``, ``analysis_info.json``,
``summary.txt``, ``figures/*.png``.

Exit code: 0 when every planned run is done; 3 when a run is missing / failed / incomplete (reported in the tables and
on stderr); 2 on an argument error or an output directory that is not empty.

Usage:
    python tools/ms/r3_analysis.py --pilot-root results/ms_r3/pilot \
        --ms-r2-pilot-root <MS-R2 worktree>/results/ms_r2/pilot \
        --parents-root <v2-t2-refine>/results/v2_refine/parents_A \
        --rehearsal-root <v2-t2-refine>/results/v2_T2_locked/rehearsal_v2_0 --actors t1 relu t10 \
        --out results/ms_r3/analysis
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import r1_analysis as R1  # noqa: E402
import r2_analysis as R2  # noqa: E402

# --------------------------------------------------------------------------------------------- constants
N_BOOT = 10000
BOOT_SEED = 20261008
QS: Tuple[int, ...] = (50, 60)
SEEDS: Tuple[int, ...] = tuple(range(10501, 10511))
ACTORS: Tuple[str, ...] = ("t1", "relu", "t10")
CONTROL = "t1"
STARTS: Tuple[str, ...] = ("bb", "st")
SCALES: Tuple[int, ...] = (1, 16)
S_LOW, S_HIGH = 1, 16
ARM_RE = re.compile(r"(t1|relu|t10)_(bb|st)_s(\d+)")
MS_R2_ARMS: Tuple[str, ...] = ("NL_bb_s1", "NL_bb_s16", "NL_st_s1", "NL_st_s16")
PARENTS, REHEARSAL = R1.PARENTS, R1.REHEARSAL
REF_ARMS: Tuple[str, ...] = MS_R2_ARMS + (PARENTS, REHEARSAL)
PRIMARY, SIGNED = R1.PRIMARY, R1.SIGNED
# D2: the d component of the t10 actor's input is multiplied by this, so a stored weight acts like ten times its value
T10_D_SCALE = 10.0
LABEL_T1 = "vs t1 (same starts and s)"
LABEL_STAGE1_T1 = "stage 1: vs t1 (same starts and s)"
LABEL_NL = "s=16 vs s=1 (same actor and starts)"
LABEL_STARTS = "st vs bb (same actor and s)"
LABEL_PARENTS = "vs parents_A"
LABEL_REHEARSAL = "vs rehearsal_v2_0"
LABEL_MS_R2 = "vs MS-R2 NL re-run (C-MS5 context)"
CRITERION_NOTE = ("pre-registered criterion (descriptive, not a gate); no selection rule and no protocol change "
                  "follow from it")
PARENTS_NOTE = ("SECONDARY, descriptive table (every arm against parents_A with MS-R1's criterion): not the "
                "pre-registered criterion of MS-R3, not a gate; no selection rule and no protocol change follow "
                "from it")
NO_EFFECT_SENTENCE = R1.NO_EFFECT_SENTENCE
DENOM_TOL = 1e-9                 # |mean change of the smoothing part| (effort units) below which a ratio is undefined
TRANSMISSION_NOTE = ("ratio = mean(d gap) / mean(d smoothing), d = s16 - s1 within one (actor, starts); 1 = the whole "
                     "smoothing change reaches the gap, 0 = the remainder offsets it; interval = percentile bootstrap "
                     "of the ratio of resampled means (resamples with |mean d smoothing| <= 1e-9 dropped)")
ACTOR_COLORS = {"t1": "tab:blue", "relu": "tab:red", "t10": "tab:green"}
START_NAME = {"bb": "bin-balanced", "st": "stratified"}
S_STYLE = {1: "-", 16: "--"}
START_MARKER = {"bb": "o", "st": "^"}
FIG_WIDTH_IN, FIG_DPI = 10.0, 100  # every figure is 10 in x 100 dpi = 1000 px wide whatever the number of q columns

#: (column, lower is better / None): the metrics of the paired tables (D6 secondary), |peak error| first. The first
# block : is the D6 list, the second block (resolution / location metrics of D5) is added.
SECONDARY_METRICS: List[Tuple[str, Optional[bool]]] = [
    (PRIMARY, True), (SIGNED, None), ("gap", True), ("smoothing", True), ("remainder", None), ("sigma_2_0", True),
    ("gap_rel", True), ("smoothing_rel", True), ("remainder_rel", None),
    ("stage2_rmse_pos_over_g2_0", True), ("stage2_tail_mean_over_g2_0", True), ("stage2_tail_max_over_g2_0", True),
    ("eta_T_over_dw", True), ("eta_dev", True), ("t2_R0_final", True), ("t2_R_final", True), ("w_eff", True),
    ("peak_locfree_rel_err", None), ("sym_err_max_rel", True), ("w1d_abs_max", None)]
#: the metrics of the comparisons with the references
REF_METRICS: List[Tuple[str, Optional[bool]]] = SECONDARY_METRICS + [
    ("e2_at_0", None), ("eta_dev_minus_final_abs", True), ("t2_R0_dev", True), ("t2_Rtail_final", True),
    ("t2_Delta_final", True), ("t2_updates", True), ("t2_episodes", True), ("t2_minibatch_steps", True),
    ("t2_wall_sec", True)]
INTERACTION_METRICS: Tuple[str, ...] = (PRIMARY, "gap", "remainder", "smoothing")
#: the metrics of the C-MS5 context table (t1 arm - its MS-R2 re-run): every metric except the wall-clock ones, which
#: differ between two runs of the same computation
CONTEXT_METRICS: List[Tuple[str, Optional[bool]]] = [
    (m, d) for m, d in REF_METRICS + list(R1.S1_METRICS) if "wall" not in m]

R3_COLS: List[str] = [
    "actor", "starts", "s", "gap", "smoothing", "remainder", "gap_rel", "smoothing_rel", "remainder_rel", "sigma_2_0",
    "smoothing_over_formula", "conc_scale_final", "t2_R0_over_peak_final", "t2_R0_over_peak_dev",
    "linearised_R0_over_peak", "peak_locfree_rel_err", "peak_locfree_argmax_d", "sym_err_max", "sym_err_max_rel",
    "sym_err_argmax_abs_d", "tent_slope", "w_eff", "actor_variant_export", "w1d_export_update", "w1d_abs_max",
    "w1d_abs_q25", "w1d_abs_q50", "w1d_abs_q75", "w1d_abs_q90", "w1d_n_gt1", "w1d_bend_d_min"]
COLUMNS: List[str] = list(R1.COLUMNS) + R3_COLS
STR_COLS = set(R1.STR_COLS) | {"actor", "starts", "actor_variant_export"}
BOOL_COLS = set(R1.BOOL_COLS)
TIE_COLS = ("peak_locfree_rel_err", "peak_locfree_argmax_d", "sym_err_max", "sym_err_max_rel", "sym_err_argmax_abs_d",
            "tent_slope", "w_eff")
FL_COLS = ["arm", "actor", "starts", "s", "q", "seed", "status", "update", "local", "variant", "d_scale", "B",
           "w_abs_max", "w_abs_q25", "w_abs_q50", "w_abs_q75", "w_abs_q90", "n_w_abs_gt1", "bend_d_min", "error"]
FU_COLS = ["arm", "actor", "starts", "s", "q", "seed", "status", "update", "local", "variant", "unit", "w_d", "w_d_abs",
           "w_stage", "bias", "center_d"]
TRAJ_COLS = (["arm", "actor", "starts", "s", "q", "seed", "status", "update", "local", "lr", "conc_scale", "e2_at_0",
              "sigma_0", "e_sigma_0", "g2_0", "smoothing", "remainder", "gap", "w_eff", "R0", "R", "Delta", "C",
              "valid"]
             + list(R2.TRAJ_PEAK_COLS))
TRAJ_STAT_COLS: Tuple[str, ...] = ("e2_at_0", "smoothing", "remainder", "sigma_0", "w_eff", "R0", "gap")
TIE_PROFILE_HALF_WIDTH = 30.0    # the near-tie window of the profile table / figure: |d| <= 30


# --------------------------------------------------------------------------------------------- small helpers
_f = R1._f


def arm_name(actor: str, starts: str, s: int) -> str:
    """``{actor}_{starts}_s{s}``."""
    return "%s_%s_s%d" % (actor, starts, int(s))


def parse_arm(arm: str) -> Tuple[str, str, float]:
    """``(actor, starts, s)`` of an ``{actor}_{bb|st}_s{s}`` arm; ``("", "", nan)`` for any other arm."""
    m = ARM_RE.fullmatch(str(arm))
    return (m.group(1), m.group(2), float(m.group(3))) if m else ("", "", float("nan"))


def check_actors(actors: Sequence[str]) -> Tuple[str, ...]:
    """The actors in canonical order (:data:`ACTORS`).

    Raises:
        ValueError: On an unknown or repeated actor, or if the control ``t1`` is not among them (the paired tables
        need it).
    """
    got = list(actors)
    bad = [a for a in got if a not in ACTORS]
    if bad:
        raise ValueError("unknown actors %s; actors are %s" % (bad, list(ACTORS)))
    if len(set(got)) != len(got):
        raise ValueError("repeated actors in %s" % got)
    if CONTROL not in got:
        raise ValueError("the control actor %r must be among --actors (every paired table needs it)" % CONTROL)
    return tuple(a for a in ACTORS if a in got)


def r3_arms(actors: Sequence[str], starts: Sequence[str] = STARTS, scales: Sequence[int] = SCALES) -> List[str]:
    """The planned arms, actor-major (``t1_bb_s1, t1_bb_s16, t1_st_s1, ..., relu_bb_s1, ...``)."""
    return [arm_name(a, k, s) for a in actors for k in starts for s in scales]


def primary_comparisons(actors: Sequence[str], starts: Sequence[str] = STARTS,
                        scales: Sequence[int] = SCALES) -> List[Tuple[str, str]]:
    """``(v arm, t1 arm)`` of the primary criterion in table order: v outermost, then starts, then s."""
    return [(arm_name(v, k, s), arm_name(CONTROL, k, s))
            for v in actors if v != CONTROL for k in starts for s in scales]


def noise_landing_comparisons(actors: Sequence[str], starts: Sequence[str] = STARTS) -> List[Tuple[str, str]]:
    """``(s=16 arm, s=1 arm)`` within each (actor, starts)."""
    return [(arm_name(a, k, S_HIGH), arm_name(a, k, S_LOW)) for a in actors for k in starts]


def starts_comparisons(actors: Sequence[str], scales: Sequence[int] = SCALES) -> List[Tuple[str, str]]:
    """``(st arm, bb arm)`` within each (actor, s)."""
    return [(arm_name(a, "st", s), arm_name(a, "bb", s)) for a in actors for s in scales]


def ms_r2_comparisons(actors: Sequence[str], starts: Sequence[str] = STARTS,
                      scales: Sequence[int] = SCALES) -> List[Tuple[str, str]]:
    """``(t1 arm, MS-R2 NL arm)``: the bit-identical re-runs of check C-MS5 (only for the starts / scales that
    exist)."""
    return [(arm_name(CONTROL, k, s), "NL_%s_s%d" % (k, s)) for k in starts for s in scales
            if "NL_%s_s%d" % (k, s) in MS_R2_ARMS and CONTROL in actors]


def _tags(arm: str) -> Dict[str, Any]:
    """The ``actor`` / ``starts`` / ``s`` columns of an arm name (empty for a reference arm)."""
    a, k, s = parse_arm(arm)
    return {"actor": a, "starts": k, "s": int(s) if np.isfinite(s) else ""}


def _flag(row: Dict[str, Any], msg: str) -> None:
    """Append ``msg`` to the row's ``flags`` (joined with ``"; "`` as in ``r1_analysis.finish_row``)."""
    row["flags"] = (str(row["flags"]) + "; " + msg) if str(row.get("flags") or "") else msg


# --------------------------------------------------------------------------------------------- statistics
def paired_summary(diff: Sequence[float], lower_better: Optional[bool]) -> Dict[str, Any]:
    """Mean / median, sign counts and the two bootstrap intervals of paired differences (fresh generator per call).

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
    """Descriptive statistics of one sample with the bootstrap interval of its mean (MS-R3 seed)."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    lo, hi = R1.boot_ci(a, "mean", N_BOOT, BOOT_SEED)
    return {"n": int(a.size), "mean": float(a.mean()) if a.size else float("nan"),
            "median": float(np.median(a)) if a.size else float("nan"),
            "sd": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
            "min": float(a.min()) if a.size else float("nan"), "max": float(a.max()) if a.size else float("nan"),
            "ci_mean_lo": lo, "ci_mean_hi": hi, "ci_mean_contains_0": bool(a.size > 0 and lo <= 0.0 <= hi)}


@contextmanager
def r3_bootstrap() -> Iterator[None]:
    """Rebind ``r1_analysis.summary_stats`` to the MS-R3 seed while the descriptive MS-R1 tables are built.

    ``r1_analysis.stage1_table`` calls the module's ``summary_stats`` (bootstrap seed 20261006); inside this context
    it uses 20261008 like every other interval of this round.
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
    """Paired-difference table (arm - baseline) and its seed-level long form, bootstrap seed 20261008.

    Args:
        df: The per-run table.
        comparisons: ``(arm, baseline)`` in table order.
        metrics: ``(column, direction)`` list; a metric without any finite pair is omitted except the first one.
        qs: q values.
        seeds: Development seeds (the pairing keys).
        label: Name of the comparison.
        primary_note: If given, the ``status`` of the |peak error| rows (the criterion statistic); every other row is
            ``descriptive``.

    Returns:
        ``(summary, seed_level)``.
    """
    rows, long_rows = [], []
    for arm, base in comparisons:
        tags = _tags(arm)
        for q in qs:
            for i, (m, direction) in enumerate(metrics):
                sv = R1.paired_seed_values(df, arm, base, q, m, seeds)
                if sv.empty and i > 0:
                    continue
                note = primary_note if (primary_note and m == PRIMARY) else "descriptive"
                rows.append({"comparison": label, "arm": arm, "baseline": base, **tags, "q": q, "metric": m,
                             "direction": "lower is better" if direction else "none", "n_expected": len(seeds),
                             "status": note, **paired_summary(sv["diff"].to_numpy(), direction)})
                for r in sv.itertuples(index=False):
                    long_rows.append({"comparison": label, "arm": arm, "baseline": base, **tags, "q": q, "metric": m,
                                      "seed": r.seed, "arm_value": r.arm_value, "base_value": r.base_value,
                                      "diff": r.diff, "status": note})
    cols = ["comparison", "arm", "baseline", "actor", "starts", "s", "q", "metric", "seed", "arm_value", "base_value",
            "diff", "status"]
    return pd.DataFrame(rows), pd.DataFrame(long_rows, columns=cols)


def criterion_table(df: pd.DataFrame, paired: pd.DataFrame, comparisons: Sequence[Tuple[str, str]],
                    qs: Sequence[int], seeds: Sequence[int], label: str, note: str) -> pd.DataFrame:
    """``criterion.csv``: parts (a) and (b) per (arm, baseline) of ``comparisons`` (table order, q in the given
    order).

    The numbers of one row are ``r1_analysis.criterion_row`` of the |peak error| rows of ``paired`` and the gate pairs
    of (arm, baseline); only the bootstrap seed (the paired summary above) differs from
    ``r1_analysis.criterion_table``.
    """
    rows = []
    for arm, base in comparisons:
        prim = paired[(paired["arm"] == arm) & (paired["baseline"] == base) & (paired["metric"] == PRIMARY)
                      & (paired["comparison"] == label)]
        gp = R1.gate_pairs(df, arm, base, qs, seeds)
        rows.append({"arm": arm, "baseline": base, **_tags(arm), "comparison": label, "primary_metric": PRIMARY,
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
        d_gap: Paired changes of the gap (s16 - s1), one per seed.
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
            rows.append({"arm": arm, "baseline": base, **_tags(arm), "q": q, **tr,
                         "median_per_seed_ratio": float(np.median(fin)) if fin else float("nan"),
                         "per_seed_ratios": ";".join("%d:%.6g" % (sd, x) for sd, x in zip(common, per_seed)),
                         "status": "descriptive", "note": TRANSMISSION_NOTE})
            for sd, g_, s_, x in zip(common, dg, ds, per_seed):
                long_rows.append({"arm": arm, "baseline": base, "q": q, "seed": sd, "d_gap": float(g_),
                                  "d_smoothing": float(s_), "ratio": x, "status": "descriptive"})
    cols = ["arm", "baseline", "q", "seed", "d_gap", "d_smoothing", "ratio", "status"]
    return pd.DataFrame(rows), pd.DataFrame(long_rows, columns=cols)


def noise_landing_table(paired: pd.DataFrame, trans: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    """``noise_landing.csv``: the paired changes s16 - s1 of every metric per (actor, starts, q) (the rows of
    ``paired``) and, per (actor, starts, q), one row with ``metric == "transmission_ratio"`` that carries the
    transmission table of the same cell: ``mean`` = the ratio of the mean changes of the gap and of the smoothing
    part, ``ci_mean_lo`` / ``ci_mean_hi`` its percentile bootstrap interval, ``median`` the median per-seed ratio,
    ``mean_d_gap`` / ``mean_d_smoothing`` / ``n_boot_valid`` / ``per_seed_ratios`` and the sign convention in ``note``
    (the same numbers as ``transmission.csv``)."""
    rows = []
    for r in trans.itertuples():
        rows.append({"comparison": LABEL_NL, "arm": r.arm, "baseline": r.baseline, "actor": r.actor, "starts": r.starts,
                     "s": r.s, "q": r.q, "metric": "transmission_ratio", "direction": "none", "n_expected": len(seeds),
                     "status": "descriptive", "n_pairs": r.n_pairs, "mean": r.ratio, "median": r.median_per_seed_ratio,
                     "ci_mean_lo": r.ci_lo, "ci_mean_hi": r.ci_hi, "mean_d_gap": r.mean_d_gap,
                     "mean_d_smoothing": r.mean_d_smoothing, "n_boot_valid": r.n_boot_valid,
                     "per_seed_ratios": r.per_seed_ratios, "note": r.note})
    return pd.concat([paired, pd.DataFrame(rows)], ignore_index=True) if rows else paired


def _value_series(df: pd.DataFrame, arm: str, q: int, metric: str) -> pd.Series:
    """Finite values of ``metric`` of the complete runs of (arm, q), indexed by seed."""
    g = df[(df["arm"] == arm) & (df["q"] == q) & df["complete"].astype(bool)].set_index("seed")[metric]
    g = pd.to_numeric(g, errors="coerce")
    return g[np.isfinite(g)]


def interaction_tables(df: pd.DataFrame, actors: Sequence[str], starts: Sequence[str], qs: Sequence[int],
                       seeds: Sequence[int], metrics: Sequence[str] = INTERACTION_METRICS
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``interaction.csv`` (summary) and ``interaction_seed_level.csv``: (v_s16 - v_s1) - (t1_s16 - t1_s1) per
    (v, starts, q, seed).

    A seed enters only when all four runs (v and t1, each at s = 16 and s = 1) are complete and finite on the metric.
    """
    rows, long_rows = [], []
    for v in [a for a in actors if a != CONTROL]:
        for k in starts:
            for q in qs:
                for m in metrics:
                    ser = {"v_s16": _value_series(df, arm_name(v, k, S_HIGH), q, m),
                           "v_s1": _value_series(df, arm_name(v, k, S_LOW), q, m),
                           "t1_s16": _value_series(df, arm_name(CONTROL, k, S_HIGH), q, m),
                           "t1_s1": _value_series(df, arm_name(CONTROL, k, S_LOW), q, m)}
                    diffs = []
                    for sd in seeds:
                        if not all(sd in x.index for x in ser.values()):
                            continue
                        v_ch = float(ser["v_s16"].loc[sd] - ser["v_s1"].loc[sd])
                        t_ch = float(ser["t1_s16"].loc[sd] - ser["t1_s1"].loc[sd])
                        diffs.append(v_ch - t_ch)
                        long_rows.append({"actor": v, "starts": k, "q": q, "metric": m, "seed": sd,
                                          "v_s16": float(ser["v_s16"].loc[sd]), "v_s1": float(ser["v_s1"].loc[sd]),
                                          "t1_s16": float(ser["t1_s16"].loc[sd]), "t1_s1": float(ser["t1_s1"].loc[sd]),
                                          "v_change": v_ch, "t1_change": t_ch, "interaction": v_ch - t_ch,
                                          "status": "descriptive"})
                    rows.append({"actor": v, "starts": k, "q": q, "metric": m, "n_expected": len(seeds),
                                 "status": "descriptive", **paired_summary(np.asarray(diffs, dtype=float), None)})
    cols = ["actor", "starts", "q", "metric", "seed", "v_s16", "v_s1", "t1_s16", "t1_s1", "v_change", "t1_change",
            "interaction", "status"]
    return pd.DataFrame(rows), pd.DataFrame(long_rows, columns=cols)


def _arm_mean(df: pd.DataFrame, arm: str, q: int, col: str) -> Tuple[float, int]:
    """``(mean, n)`` of the finite values of ``col`` over the complete runs of (arm, q)."""
    v = pd.to_numeric(df[(df["arm"] == arm) & (df["q"] == q) & df["complete"].astype(bool)][col], errors="coerce")
    v = v.dropna()
    return (float(v.mean()) if len(v) else float("nan"), int(len(v)))


def quadrature_check(g1: float, sm1: float, g16: float, sm16: float) -> Dict[str, Any]:
    """The quadrature check of D6 from four arm means.

    ``F^2 = g1^2 - sm1^2`` (negative: ``F = 0`` and ``F2_negative``); quadrature prediction of the s = 16 gap
    ``sqrt(F^2 + sm16^2)``; additive prediction ``g1 - (sm1 - sm16)``; the absolute errors against the observed
    ``g16`` and which prediction is closer (``quadrature`` / ``additive`` / ``tie``).
    """
    nan = float("nan")
    out: Dict[str, Any] = {"F2": nan, "F2_negative": False, "F": nan, "quadrature_pred": nan, "additive_pred": nan,
                           "abs_err_quadrature": nan, "abs_err_additive": nan, "closer": ""}
    if not all(np.isfinite(x) for x in (g1, sm1, g16, sm16)):
        return out
    f2 = g1 * g1 - sm1 * sm1
    f2c = max(f2, 0.0)
    out["F2"], out["F2_negative"], out["F"] = f2, bool(f2 < 0.0), math.sqrt(f2c)
    out["quadrature_pred"] = math.sqrt(f2c + sm16 * sm16)
    out["additive_pred"] = g1 - (sm1 - sm16)
    out["abs_err_quadrature"] = abs(out["quadrature_pred"] - g16)
    out["abs_err_additive"] = abs(out["additive_pred"] - g16)
    eq, ea = out["abs_err_quadrature"], out["abs_err_additive"]
    out["closer"] = "quadrature" if eq < ea else ("additive" if ea < eq else "tie")
    return out


def quadrature_table(df: pd.DataFrame, actors: Sequence[str], starts: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """``quadrature_check.csv``: per (actor, starts, q), from the arm means of the complete runs (descriptive)."""
    rows = []
    for a in actors:
        for k in starts:
            for q in qs:
                lo, hi = arm_name(a, k, S_LOW), arm_name(a, k, S_HIGH)
                (g1, n_g1), (sm1, _) = _arm_mean(df, lo, q, "gap"), _arm_mean(df, lo, q, "smoothing")
                (g16, n_g16), (sm16, _) = _arm_mean(df, hi, q, "gap"), _arm_mean(df, hi, q, "smoothing")
                rows.append({"actor": a, "starts": k, "q": q, "n_s1": n_g1, "n_s16": n_g16, "gap_s1": g1,
                             "smoothing_s1": sm1, "gap_s16": g16, "smoothing_s16": sm16,
                             **quadrature_check(g1, sm1, g16, sm16), "status": "descriptive",
                             "note": "from arm means; F2 = gap(s1)^2 - smoothing(s1)^2 (negative: F = 0, flagged); "
                                     "quadrature = sqrt(F2 + smoothing(s16)^2); additive = gap(s1) - (smoothing(s1) - "
                                     "smoothing(s16)); closer = the prediction with the smaller absolute error against "
                                     "the observed gap(s16)"})
    return pd.DataFrame(rows)


def _frac_lt(x: np.ndarray, y: np.ndarray) -> float:
    """Share of positions with ``x < y`` (NaN for empty input)."""
    return float((x < y).mean()) if x.size else float("nan")


def predictions_table(df: pd.DataFrame, actors: Sequence[str], starts: Sequence[str], qs: Sequence[int],
                      seeds: Sequence[int], trans: pd.DataFrame) -> pd.DataFrame:
    """``predictions.csv``: the evidence of D6 P1-P4 per (actor, starts, q); descriptive, no verdict column.

    * P1 (v only): |peak| of the arm against ``t1`` at s = 1: pairs, runs lower / higher, mean change.
    * P2 (s = 1): mean remainder, smoothing part, e_hat_2(0) and e_sigma(0), the ratio of the means remainder /
    smoothing,
      the median of the per-run ratios and the share of runs with |remainder| < smoothing part.
    * P3 (s = 16): mean |peak|, the smoothing floor in units of e2*(0) (``floor_rel_sigma`` = mean of sigma_2(0) /
      (sqrt(pi) q), ``floor_rel_smoothing`` = mean of the measured smoothing part / e2*(0)), |peak| / floor, and the
      transmission ratio of the (actor, starts, q) cell.
    * P4 (v only, both s): RMSE_pos / e2*(0) of the arm and of ``t1``, runs lower than ``t1`` and the mean change.
    """
    nan = float("nan")
    rows = []
    for a in actors:
        for k in starts:
            for q in qs:
                r: Dict[str, Any] = {"actor": a, "starts": k, "q": q, "status": "descriptive"}
                lo, hi = arm_name(a, k, S_LOW), arm_name(a, k, S_HIGH)
                if a != CONTROL:
                    sv = R1.paired_seed_values(df, lo, arm_name(CONTROL, k, S_LOW), q, PRIMARY, seeds)
                    d = sv["diff"].to_numpy(dtype=float)
                    r.update(p1_n_pairs=int(d.size), p1_n_lower=int((d < 0).sum()), p1_n_higher=int((d > 0).sum()),
                             p1_mean_change=float(d.mean()) if d.size else nan,
                             p1_abs_peak_s1=_arm_mean(df, lo, q, PRIMARY)[0],
                             p1_abs_peak_t1_s1=_arm_mean(df, arm_name(CONTROL, k, S_LOW), q, PRIMARY)[0])
                g = df[(df["arm"] == lo) & (df["q"] == q) & df["complete"].astype(bool)]
                rem = pd.to_numeric(g["remainder"], errors="coerce").to_numpy(dtype=float)
                sm = pd.to_numeric(g["smoothing"], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(rem) & np.isfinite(sm) & (sm > 0.0)
                mr, ms_ = (float(rem[ok].mean()), float(sm[ok].mean())) if ok.any() else (nan, nan)
                r.update(p2_n=int(ok.sum()), p2_mean_remainder=mr, p2_mean_smoothing=ms_,
                         p2_mean_e_hat_2_0=_arm_mean(df, lo, q, "e2_at_0")[0],
                         p2_mean_e_sigma_0=_arm_mean(df, lo, q, "smoothed_e_pred_0")[0],
                         p2_remainder_over_smoothing=mr / ms_ if ok.any() and ms_ != 0.0 else nan,
                         p2_median_run_ratio=float(np.median(rem[ok] / sm[ok])) if ok.any() else nan,
                         p2_share_abs_remainder_lt_smoothing=_frac_lt(np.abs(rem[ok]), sm[ok]))
                gh = df[(df["arm"] == hi) & (df["q"] == q) & df["complete"].astype(bool)]
                peak = pd.to_numeric(gh[PRIMARY], errors="coerce").to_numpy(dtype=float)
                sig = pd.to_numeric(gh["sigma_2_0"], errors="coerce").to_numpy(dtype=float)
                sml = pd.to_numeric(gh["smoothing_rel"], errors="coerce").to_numpy(dtype=float)
                floor_sig = float(np.nanmean(sig / (math.sqrt(math.pi) * q))) if np.isfinite(sig).any() else nan
                floor_sm = float(np.nanmean(sml)) if np.isfinite(sml).any() else nan
                ap = float(np.nanmean(peak)) if np.isfinite(peak).any() else nan
                tr = trans[(trans["arm"] == hi) & (trans["q"] == q)] if len(trans) else trans
                r.update(p3_n=int(np.isfinite(peak).sum()), p3_abs_peak_s16=ap, p3_floor_rel_sigma=floor_sig,
                         p3_floor_rel_smoothing=floor_sm,
                         p3_abs_peak_over_floor=ap / floor_sig if np.isfinite(ap) and floor_sig > 0 else nan,
                         p3_transmission_ratio=float(tr["ratio"].iloc[0]) if len(tr) else nan,
                         p3_transmission_ci_lo=float(tr["ci_lo"].iloc[0]) if len(tr) else nan,
                         p3_transmission_ci_hi=float(tr["ci_hi"].iloc[0]) if len(tr) else nan)
                for s_, arm_ in ((S_LOW, lo), (S_HIGH, hi)):
                    r["p4_rmse_s%d" % s_] = _arm_mean(df, arm_, q, "stage2_rmse_pos_over_g2_0")[0]
                    r["p4_rmse_t1_s%d" % s_] = _arm_mean(df, arm_name(CONTROL, k, s_), q,
                                                         "stage2_rmse_pos_over_g2_0")[0]
                    if a != CONTROL:
                        sv = R1.paired_seed_values(df, arm_, arm_name(CONTROL, k, s_), q, "stage2_rmse_pos_over_g2_0",
                                                   seeds)
                        d = sv["diff"].to_numpy(dtype=float)
                        r["p4_n_pairs_s%d" % s_] = int(d.size)
                        r["p4_n_lower_s%d" % s_] = int((d < 0).sum())
                        r["p4_mean_change_s%d" % s_] = float(d.mean()) if d.size else nan
                rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- tie and weight metrics
def tie_metrics(rd: Sequence[float], e2: Sequence[float], g20: float, gap: float, q: float) -> Dict[str, float]:
    """Location-free peak, symmetry error and effective rounding width from a recovery grid (evaluation only).

    Args:
        rd: Recovery grid ``recovery_d_grid`` (symmetric, uniform).
        e2: Learned e_hat_2 on that grid (``recovery_e2``).
        g20: The closed-form tie value e2*(0) (``g2_at_0``).
        gap: ``g2_at_0 - e2_at_0`` of the run.
        q: Noise half-width.

    Returns:
        ``peak_locfree_rel_err`` = (max e_hat_2 - g20) / g20 at ``peak_locfree_argmax_d`` (the argmax over the whole
        grid, the definition of ``run/run_ms_stagewise.py``); ``sym_err_max`` = max over the nodes with |d| < 2q of
        |e_hat_2(d) - e_hat_2(-d)| with ``sym_err_max_rel`` (/ g20) and ``sym_err_argmax_abs_d``; ``tent_slope`` = g20
        / (2q) and ``w_eff`` = gap / tent_slope. A quantity that cannot be computed is NaN.
    """
    nan = float("nan")
    out = {k: nan for k in TIE_COLS}
    d = np.asarray(rd, dtype=float)
    e = np.asarray(e2, dtype=float)
    if d.size == 0 or d.shape != e.shape or not (np.isfinite(g20) and g20 != 0.0 and q > 0):
        return out
    j = int(np.argmax(e))
    out["peak_locfree_rel_err"] = float((e[j] - g20) / g20)
    out["peak_locfree_argmax_d"] = float(d[j])
    if np.allclose(d, -d[::-1], rtol=0.0, atol=1e-9):
        sym = np.abs(e - e[::-1])
        pos = np.abs(d) < 2.0 * q
        if pos.any():
            js = int(np.argmax(np.where(pos, sym, -np.inf)))
            out["sym_err_max"] = float(sym[js])
            out["sym_err_max_rel"] = float(sym[js]) / g20
            out["sym_err_argmax_abs_d"] = float(abs(d[js]))
    slope = g20 / (2.0 * q)
    out["tent_slope"] = float(slope)
    if np.isfinite(gap):
        out["w_eff"] = float(gap / slope)
    return out


def first_layer_stats(w_first: np.ndarray, variant: str, B: float) -> Dict[str, Any]:
    """Statistics of the first-layer weights on the d input, in units of d / B.

    Args:
        w_first: ``actor.l1.weight[:, 1]`` of an export (the weights on the d component of the input; column 0 is the
        stage
            feature). The stored values are used for ``t1`` and ``relu``; TEN times the stored values for ``t10`` (its
            d component is multiplied by ten inside the actor, so a stored weight acts like ten times its value on d /
            B).
        variant: ``t1``, ``relu`` or ``t10``.
        B: ``(e_max - e_min) + 2q``.

    Returns:
        ``d_scale``, ``w_abs_max``, ``w_abs_q25/q50/q75/q90`` (quantiles of |w| over the hidden units),
        ``n_w_abs_gt1`` and ``bend_d_min = B / w_abs_max`` (units of d).

    Raises:
        ValueError: On an unknown variant (never the tanh d / B reading of an export that names another variant).
    """
    if variant not in ACTORS:
        raise ValueError("unknown actor variant %r; known: %s" % (variant, list(ACTORS)))
    scale = T10_D_SCALE if variant == "t10" else 1.0
    wa = np.abs(np.asarray(w_first, dtype=float)) * scale
    mx = float(wa.max())
    q25, q50, q75, q90 = (float(x) for x in np.percentile(wa, [25, 50, 75, 90]))
    return {"d_scale": scale, "w_abs_max": mx, "w_abs_q25": q25, "w_abs_q50": q50, "w_abs_q75": q75, "w_abs_q90": q90,
            "n_w_abs_gt1": int((wa > 1.0).sum()), "bend_d_min": float(B / mx) if mx > 0.0 else float("nan")}


def export_variant(z: Mapping[str, Any]) -> str:
    """The actor variant of an opened export: its ``actor_variant`` entry, absent = ``t1``."""
    if "actor_variant" in z.keys():
        return str(np.asarray(z["actor_variant"]).item())
    return "t1"


def read_export_stats(path: Path, B: float) -> Dict[str, Any]:
    """:func:`first_layer_stats` of one weight export (the variant is read from the export)."""
    with np.load(path) as z:
        variant = export_variant(z)
        w = np.asarray(z["actor.l1.weight"])[:, 1]
    return {"variant": variant, **first_layer_stats(w, variant, B)}


def terminal_exports(wdir: Path, entry: int, last_local: int) -> List[Tuple[int, int, Path]]:
    """``(update, local, path)`` of the exports ``u*.npz`` of the terminal stage (``0 < local <= last_local``)."""
    out: List[Tuple[int, int, Path]] = []
    if not Path(wdir).is_dir():
        return out
    for p in Path(wdir).glob("u*.npz"):
        m = re.fullmatch(r"u(\d+)\.npz", p.name)
        if m and 0 < int(m.group(1)) - entry <= last_local:
            out.append((int(m.group(1)), int(m.group(1)) - entry, p))
    return sorted(out)


def _terminal_entry(run_dir: Path) -> int:
    """Global update at which the terminal stage starts (0 in every pilot: stage 2 runs first)."""
    summ = R1._jload(Path(run_dir) / "ms_run_summary.json", [], False)
    v = R1._get(summ, "phase_timing", "stage2", "global_entry")
    return int(v) if v is not None and np.isfinite(_f(v)) else 0


def game_B(proto: Mapping[str, Any], q: int) -> float:
    """``B = (e_max - e_min) + 2q`` from the protocol record of ``q`` (NaN if the record is absent)."""
    g = R1._get(proto, "records", str(int(q)), "game")
    return (_f(g["e_max"]) - _f(g["e_min"]) + 2.0 * _f(g["q"])) if g else float("nan")


def first_layer_table(df: pd.DataFrame, win: R2.Windows, proto: Mapping[str, Any], arms: Sequence[str]) -> pd.DataFrame:
    """``first_layer_weights.csv``: first-layer d-weight statistics at every terminal-stage weight export of every run
    of ``arms`` (every 25 updates; the exports of local update >= 1800 and those of every 100th update are the rows
    the prompt names, the others are kept so that the figure of all exports can be drawn from this file).

    A run without a weights directory has no rows; an unreadable export is a row with NaN statistics and the reason in
    ``error``.
    """
    rows: List[Dict[str, Any]] = []
    for r in df[df["arm"].isin(list(arms))].itertuples():
        if r.status == "missing":
            continue
        B = game_B(proto, int(r.q))
        tags = _tags(r.arm)
        for upd, loc, path in terminal_exports(Path(r.run_dir) / "weights", _terminal_entry(Path(r.run_dir)),
                                               win.decay_last):
            row: Dict[str, Any] = {"arm": r.arm, **{k: tags[k] for k in ("actor", "starts", "s")}, "q": r.q,
                                   "seed": r.seed, "status": r.status, "update": upd, "local": loc, "B": B, "error": ""}
            try:
                row.update(read_export_stats(path, B))
            except Exception as exc:  # noqa: BLE001
                row["error"] = "%s: %s" % (type(exc).__name__, exc)
            rows.append(row)
    return pd.DataFrame(rows, columns=FL_COLS)


def unit_rows(w: np.ndarray, bias: np.ndarray, variant: str, B: float, tau: float = 1.0) -> Dict[str, np.ndarray]:
    """Per-hidden-unit first-layer quantities of one export (the distribution behind :func:`first_layer_stats`).

    Args:
        w: ``actor.l1.weight`` (hidden, 2): column 0 the stage feature, column 1 the d component.
        bias: ``actor.l1.bias`` (hidden,).
        variant: ``t1``, ``relu`` or ``t10`` (``t10``: the d column acts ten times as strongly).
        B: ``(e_max - e_min) + 2q``.
        tau: The stage feature at the terminal stage (1 at T = 2).

    Returns:
        ``w_d`` (signed, units of d / B, ten times the stored weight for ``t10``), ``w_d_abs``, ``w_stage`` and
        ``bias`` (stored), and ``center_d`` = -B (w_stage tau + bias) / w_d, the d at which the unit's pre-activation
        is zero (where a ReLU unit has its kink and a tanh unit its centre; NaN for w_d = 0).

    Raises:
        ValueError: On an unknown variant.
    """
    if variant not in ACTORS:
        raise ValueError("unknown actor variant %r; known: %s" % (variant, list(ACTORS)))
    scale = T10_D_SCALE if variant == "t10" else 1.0
    wd = np.asarray(w, dtype=float)[:, 1] * scale
    ws, b = np.asarray(w, dtype=float)[:, 0], np.asarray(bias, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        center = np.where(wd != 0.0, -B * (ws * tau + b) / wd, np.nan)
    return {"w_d": wd, "w_d_abs": np.abs(wd), "w_stage": ws, "bias": b, "center_d": center}


def first_layer_units_table(df: pd.DataFrame, win: R2.Windows, proto: Mapping[str, Any],
                            arms: Sequence[str]) -> pd.DataFrame:
    """``first_layer_units.csv``: one row per hidden unit of the LAST terminal-stage export of every run of ``arms``
    (:func:`unit_rows`: the distribution of the first-layer d-weights and where the units bend). An unreadable export
    has no rows (``first_layer_weights.csv`` carries its error)."""
    rows: List[Dict[str, Any]] = []
    for r in df[df["arm"].isin(list(arms))].itertuples():
        if r.status == "missing":
            continue
        ex = terminal_exports(Path(r.run_dir) / "weights", _terminal_entry(Path(r.run_dir)), win.decay_last)
        if not ex:
            continue
        upd, loc, path = ex[-1]
        try:
            with np.load(path) as z:
                variant = export_variant(z)
                u = unit_rows(np.asarray(z["actor.l1.weight"]), np.asarray(z["actor.l1.bias"]), variant,
                              game_B(proto, int(r.q)))
        except Exception:  # noqa: BLE001
            continue
        tags = _tags(r.arm)
        for j in range(len(u["w_d"])):
            rows.append({"arm": r.arm, **{k: tags[k] for k in ("actor", "starts", "s")}, "q": r.q, "seed": r.seed,
                         "status": r.status, "update": upd, "local": loc, "variant": variant, "unit": j,
                         **{k: float(u[k][j]) for k in ("w_d", "w_d_abs", "w_stage", "bias", "center_d")}})
    return pd.DataFrame(rows, columns=FU_COLS)


def attach_last_export(df: pd.DataFrame, fl: pd.DataFrame, win: R2.Windows) -> pd.DataFrame:
    """Fill the ``w1d_*`` / ``actor_variant_export`` columns of ``df`` from the last terminal-stage export of each run
    (local update ``win.decay_last``; the largest export below it if that one is absent). Flags (appended to the row's
    ``flags``, never raised): the last export is not the one of local update ``win.decay_last``; it cannot be read; its
    ``actor_variant`` (absent = ``t1``) or the one of the run's ``run_config.json`` disagrees with the arm name."""
    df = df.copy()
    if len(fl):
        last = fl.sort_values("local").groupby(["arm", "q", "seed"], sort=False).tail(1)
        key = {(r.arm, r.q, r.seed): r for r in last.itertuples()}
        for i, r in df.iterrows():
            e = key.get((r["arm"], r["q"], r["seed"]))
            if e is None:
                continue
            msgs: List[str] = []
            df.at[i, "w1d_export_update"] = e.update
            if r["status"] == "done" and int(e.local) != int(win.decay_last):
                msgs.append("last terminal-stage weight export is local %d, not %d"
                            % (int(e.local), int(win.decay_last)))
            if e.error:
                msgs.append("weight export: " + e.error)
            else:
                df.at[i, "actor_variant_export"] = e.variant
                for src, dst in (("w_abs_max", "w1d_abs_max"), ("w_abs_q25", "w1d_abs_q25"),
                                 ("w_abs_q50", "w1d_abs_q50"), ("w_abs_q75", "w1d_abs_q75"),
                                 ("w_abs_q90", "w1d_abs_q90"), ("n_w_abs_gt1", "w1d_n_gt1"),
                                 ("bend_d_min", "w1d_bend_d_min")):
                    df.at[i, dst] = getattr(e, src)
                cfg = R1._jload(Path(r["run_dir"]) / "run_config.json", [], False)
                cfg_var = str(cfg.get("actor_variant", "t1")) if cfg else ""
                want = r["actor"]
                if e.variant != want or (cfg_var and cfg_var != want):
                    msgs.append("actor variant: arm name says %s, last export says %s, run_config.json says %s" % (
                        want, e.variant, cfg_var or "unreadable"))
            if msgs:
                df.at[i, "flags"] = "; ".join(([str(r["flags"])] if str(r["flags"]) else []) + msgs)
    return df


# --------------------------------------------------------------------------------------------- extraction
def add_r3_columns(row: Dict[str, Any], proto: Mapping[str, Any]) -> None:
    """Add the MS-R3 columns that need only the row and its freeze arrays (in place; call after
    ``r2_analysis.add_r2_columns``): the arm tags, R0 / |peak| next to the linearised value, the location-free peak,
    the symmetry error and the effective rounding width. A freeze file that cannot be read of a ``done`` run is a
    flag, never an exception."""
    arm = str(row["arm"])
    actor, starts, s = parse_arm(arm)
    row["actor"], row["starts"], row["s"] = actor, starts, s
    row["actor_variant_export"] = ""
    nan = float("nan")
    q = _f(row["q"])
    peak = _f(row[PRIMARY])
    for tier in ("final", "dev"):
        r0 = _f(row["t2_R0_" + tier])
        row["t2_R0_over_peak_" + tier] = r0 / peak if np.isfinite(r0) and np.isfinite(peak) and peak > 0.0 else nan
    game = R1._get(proto, "records", str(int(q)), "game")
    row["linearised_R0_over_peak"] = 1.0 / R1.linearised_factor(game) if game else nan
    if row["status"] in ("done", "incomplete"):
        an: List[str] = []
        z = R1._npz_arrays(R1.freeze_files(Path(row["run_dir"]), arm)[0], ("recovery_d_grid", "recovery_e2"), an)
        if z:
            row.update(tie_metrics(z["recovery_d_grid"], z["recovery_e2"], _f(row["g2_at_0"]), _f(row["gap"]), q))
        if row["status"] == "done":
            for msg in an:
                _flag(row, msg)


def arm_dir(arm: str, q: int, seed: int, roots: Mapping[str, str]) -> Tuple[Path, str]:
    """Run directory of (arm, q, seed) and the name of the root it lives under."""
    if arm == PARENTS:
        return Path(roots["parents"]) / ("q%d" % q) / ("seed%d" % seed), "parents"
    if arm == REHEARSAL:
        return Path(roots["rehearsal"]) / ("q%d" % q) / ("seed%d" % seed), "rehearsal"
    key = "ms_r2_pilot" if arm in MS_R2_ARMS else "pilot"
    return Path(roots[key]) / ("q%d" % q) / ("seed%d" % seed) / arm, key


def extract_all(roots: Mapping[str, str], qs: Sequence[int], seeds: Sequence[int], th: R1.Thresholds,
                proto: Mapping[str, Any], actors: Sequence[str] = ACTORS, starts: Sequence[str] = STARTS,
                scales: Sequence[int] = SCALES) -> pd.DataFrame:
    """The per-run table: the MS-R3 arms (role ``ms_arm``), then the reference rows (role ``comparator``).

    The per-run extraction is ``r1_analysis``'s (``extract_ms_run`` for the MS-R3 arms and the MS-R2 ``NL_*`` re-runs,
    ``extract_parents_run``, ``extract_rehearsal_run``; the D3 columns of the two v2.0 comparators come from
    ``--calibration-root`` when given); the decomposition columns are ``r2_analysis.add_r2_columns``, the tie columns
    :func:`add_r3_columns`. The first-layer weight columns are filled afterwards (:func:`attach_last_export`).
    """
    rows: List[Dict[str, Any]] = []
    for arm in r3_arms(actors, starts, scales) + list(REF_ARMS):
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
                R2.add_r2_columns(r, arm not in (PARENTS, REHEARSAL))
                add_r3_columns(r, proto)
                rows.append(r)
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in COLUMNS:
        if c not in STR_COLS and c not in BOOL_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def trajectory_table(df: pd.DataFrame, win: R2.Windows, arms: Sequence[str]) -> pd.DataFrame:
    """``trajectory_checks.csv``: every check row of ``ms_checks_stage2.csv`` with local update >= ``win.traj_from``
    of every run of ``arms`` (the run's ``status`` is a column; a run without a check file has no rows), plus the
    effective rounding width ``w_eff = gap / (g2_0 / (2q))`` at every check (units of d). R0, R, Delta and C are the
    check columns."""
    parts: List[pd.DataFrame] = []
    for r in df[df["arm"].isin(list(arms))].itertuples():
        ck = R2._read_csv(Path(r.run_dir) / "ms_checks_stage2.csv")
        if ck is None or ck.empty:
            continue
        ck = ck[pd.to_numeric(ck["local"], errors="coerce") >= win.traj_from].copy()
        ck["status"] = r.status
        ck["arm"], ck["q"], ck["seed"] = r.arm, r.q, r.seed         # the run directory names the run
        for k, v in _tags(r.arm).items():
            ck[k] = v
        if "gap" in ck.columns and "g2_0" in ck.columns:
            ck["w_eff"] = pd.to_numeric(ck["gap"], errors="coerce") / (
                pd.to_numeric(ck["g2_0"], errors="coerce") / (2.0 * float(r.q)))
        else:
            ck["w_eff"] = float("nan")
        parts.append(ck.reindex(columns=TRAJ_COLS))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=TRAJ_COLS)


def trajectory_by_arm_table(traj: pd.DataFrame, arms: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """``trajectory_by_arm.csv``: per (arm, q, update) over the ``done`` runs the mean / SD / min / max of e_hat_2(0),
    the smoothing part, the remainder, sigma_2(0), w_eff, R0 and the gap (arms in the given order, then q, then
    update)."""
    cols = ["arm", "actor", "starts", "s", "q", "update", "local", "n_runs"] + [
        "%s_%s" % (c, f) for c in TRAJ_STAT_COLS for f in ("mean", "sd", "min", "max")]
    if traj.empty:
        return pd.DataFrame(columns=cols)
    t = traj[(traj["status"] == "done") & traj["arm"].isin(list(arms))]
    rows = []
    for arm in arms:
        for q in qs:
            g_arm = t[(t["arm"] == arm) & (t["q"] == q)]
            for (upd, loc), g in g_arm.groupby(["update", "local"], sort=True):
                r: Dict[str, Any] = {"arm": arm, **_tags(arm), "q": q, "update": upd, "local": loc, "n_runs": len(g)}
                for c in TRAJ_STAT_COLS:
                    v = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy(dtype=float)
                    r[c + "_mean"] = float(v.mean()) if v.size else float("nan")
                    r[c + "_sd"] = float(v.std(ddof=1)) if v.size > 1 else float("nan")
                    r[c + "_min"] = float(v.min()) if v.size else float("nan")
                    r[c + "_max"] = float(v.max()) if v.size else float("nan")
                rows.append(r)
    return pd.DataFrame(rows, columns=cols)


def first_layer_summary_table(fl: pd.DataFrame, arms: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """``first_layer_summary.csv``: per (arm, q, export) over the ``done`` runs the median / quartiles / min / max of
    the per-run max |w| (units of d / B), the median of the per-run median |w| and of the bend scale ``B / max
    |w|``."""
    cols = ["arm", "actor", "starts", "s", "q", "update", "local", "n_runs", "w_abs_max_median", "w_abs_max_q25",
            "w_abs_max_q75", "w_abs_max_min", "w_abs_max_max", "w_abs_q50_median", "bend_d_min_median"]
    if fl.empty:
        return pd.DataFrame(columns=cols)
    t = fl[(fl["status"] == "done") & (fl["error"].astype(str) == "") & fl["arm"].isin(list(arms))]
    rows = []
    for arm in arms:
        for q in qs:
            for (upd, loc), g in t[(t["arm"] == arm) & (t["q"] == q)].groupby(["update", "local"], sort=True):
                v = pd.to_numeric(g["w_abs_max"], errors="coerce").dropna().to_numpy(dtype=float)
                rows.append({"arm": arm, **_tags(arm), "q": q, "update": upd, "local": loc, "n_runs": len(g),
                             "w_abs_max_median": float(np.median(v)) if v.size else float("nan"),
                             "w_abs_max_q25": float(np.percentile(v, 25)) if v.size else float("nan"),
                             "w_abs_max_q75": float(np.percentile(v, 75)) if v.size else float("nan"),
                             "w_abs_max_min": float(v.min()) if v.size else float("nan"),
                             "w_abs_max_max": float(v.max()) if v.size else float("nan"),
                             "w_abs_q50_median": float(pd.to_numeric(g["w_abs_q50"], errors="coerce").median()),
                             "bend_d_min_median": float(pd.to_numeric(g["bend_d_min"], errors="coerce").median())})
    return pd.DataFrame(rows, columns=cols)


def tie_profile_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int],
                      half_width: float = TIE_PROFILE_HALF_WIDTH) -> pd.DataFrame:
    """``tie_profile.csv``: per (arm, q) and recovery node with |d| <= ``half_width`` the mean / SD / min / max over
    the ``done`` runs of the learned e_hat_2(d) at the final-tier freeze and the closed-form e2*(d) (evaluation
    only)."""
    cols = ["arm", "actor", "starts", "s", "q", "d", "n_runs", "e_hat_mean", "e_hat_sd", "e_hat_min", "e_hat_max", "g2"]
    rows = []
    for arm in arms:
        for q in qs:
            g = df[(df["arm"] == arm) & (df["q"] == q) & df["complete"].astype(bool)]
            grids, e_all, g2 = [], [], None
            for r in g.itertuples():
                z = R1._npz_arrays(R1.freeze_files(Path(r.run_dir), arm)[0],
                                   ("recovery_d_grid", "recovery_e2", "recovery_g2"), [])
                if not z:
                    continue
                grids.append(z["recovery_d_grid"])
                e_all.append(z["recovery_e2"])
                g2 = z["recovery_g2"] if g2 is None else g2
            if not grids or any(x.shape != grids[0].shape or not np.array_equal(x, grids[0]) for x in grids):
                continue
            d = grids[0]
            m = np.abs(d) <= half_width + 1e-9
            e = np.vstack(e_all)[:, m]
            for j, dj in enumerate(d[m]):
                col = e[:, j]
                rows.append({"arm": arm, **_tags(arm), "q": q, "d": float(dj), "n_runs": int(col.size),
                             "e_hat_mean": float(col.mean()),
                             "e_hat_sd": float(col.std(ddof=1)) if col.size > 1 else float("nan"),
                             "e_hat_min": float(col.min()), "e_hat_max": float(col.max()), "g2": float(g2[m][j])})
    return pd.DataFrame(rows, columns=cols)


def freeze_decomposition_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """``freeze_decomposition.csv``: per (arm, q) and quantity, over the complete runs, n / mean / median / SD / min /
    max and the percentile bootstrap interval of the mean (effort units for gap, smoothing, remainder, sigma; the
    ``*_rel`` rows are relative to e2*(0)). Rows without a finite value are omitted."""
    rows = []
    for arm in arms:
        for q in qs:
            g = df[(df["arm"] == arm) & (df["q"] == q) & df["complete"].astype(bool)]
            for qty in R2.FREEZE_QUANTITIES:
                v = pd.to_numeric(g[qty], errors="coerce").dropna().to_numpy()
                if v.size == 0:
                    continue
                rows.append({"arm": arm, **_tags(arm), "q": q, "quantity": qty,
                             "unit": "relative to e2*(0)" if qty.endswith("_rel") else (
                                 "ratio" if qty == "smoothing_over_formula" else "effort units"),
                             **summary_stats(v), "status": "descriptive"})
    return pd.DataFrame(rows)


def arm_summary_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """``arm_summary.csv``: descriptive statistics (with the bootstrap interval of the mean) of the secondary metrics,
    the stage-1 metrics and the tie / weight columns, per (arm, q), complete runs only."""
    metrics = [m for m, _ in REF_METRICS] + [m for m, _ in R1.S1_METRICS if m not in dict(REF_METRICS)] + [
        "peak_locfree_argmax_d", "t2_R0_over_peak_final", "w1d_abs_q50", "w1d_bend_d_min"]
    rows = []
    for arm in list(arms) + [PARENTS, REHEARSAL]:
        for q in qs:
            g = R1._sel(df, arm, q)
            for m in metrics:
                if m not in df.columns:
                    continue
                v = pd.to_numeric(g[m], errors="coerce").dropna()
                if v.empty and m not in (PRIMARY, SIGNED):
                    continue
                rows.append({"arm": arm, "q": q, "metric": m, "status": "descriptive", **summary_stats(v.to_numpy())})
    return pd.DataFrame(rows)


def r0_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int], proto: Mapping[str, Any]) -> pd.DataFrame:
    """``r0.csv``: R0 = r_2(0)/s_2 at the terminal-stage freeze (both tiers), |peak| / R0 and R0 / |peak| per arm and
    q (every arm and both comparators), with the linearised value 2k / (2k + a) of the q, the v2.0 calibration median
    of
    |peak| / R0 and the linearised factor (2k + a) / (2k) as reference columns."""
    rows = []
    for arm in list(arms) + [PARENTS, REHEARSAL]:
        for q in qs:
            g = R1._sel(df, arm, q)
            r: Dict[str, Any] = {"arm": arm, "q": q, "n": len(g), "status": "descriptive"}
            for m in R1.R0_COLS + ["t2_R0_over_peak_final", "t2_R0_over_peak_dev"]:
                R1._desc(g[m], m, r)
            game = R1._get(proto, "records", str(q), "game")
            r["calib_median_peak_over_R0"] = R1.CALIB_MEDIAN_PEAK_OVER_R0.get(int(q), float("nan"))
            r["linearised_factor"] = R1.linearised_factor(game) if game else float("nan")
            r["linearised_R0_over_peak"] = 1.0 / r["linearised_factor"] if game else float("nan")
            rows.append(r)
    return pd.DataFrame(rows)


def r0_spearman_table(df: pd.DataFrame, traj: pd.DataFrame, actors: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """``r0_spearman.csv``: Spearman correlation of R0 with |peak error| (descriptive) per q, over the ``done`` runs
    of all MS-R3 arms and of each actor: at the freeze (``t2_R0_final`` against ``stage2_peak_rel_err_abs``) and over
    all check rows of ``trajectory_checks.csv`` (``R0`` against ``stage2_peak_rel_err_abs``)."""
    rows = []
    ok = df[df["complete"].astype(bool) & df["role"].eq("ms_arm")]
    tr = traj[traj["status"] == "done"] if len(traj) else traj
    for q in qs:
        for scope, src, x, y in (("freeze", ok, "t2_R0_final", PRIMARY), ("checks", tr, "R0", PRIMARY)):
            for who in ["all"] + list(actors):
                g = src[(src["q"] == q) & ((src["actor"] == who) if who != "all" else True)] if len(src) else src
                a = pd.to_numeric(g[x], errors="coerce") if len(g) else pd.Series(dtype=float)
                b = pd.to_numeric(g[y], errors="coerce") if len(g) else pd.Series(dtype=float)
                m = np.isfinite(a) & np.isfinite(b)
                rows.append({"scope": scope, "actors": who, "q": q, "n": int(m.sum()),
                             "spearman_R0_vs_abs_peak": float(a[m].corr(b[m], method="spearman")) if m.sum() > 2
                             else float("nan"), "status": "descriptive"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- figures
def _footer(fig: Any, roots: Mapping[str, str]) -> None:
    """Provenance footer: the roots and the bootstrap setting."""
    txt = ("roots: MS-R3 pilot=%s | MS-R2 pilot=%s | parents_A=%s | rehearsal_v2_0=%s | bootstrap %d resamples, "
           "default_rng(%d)" % (roots["pilot"], roots["ms_r2_pilot"], roots["parents"], roots["rehearsal"], N_BOOT,
                               BOOT_SEED))
    fig.text(0.005, 0.003, txt, fontsize=5, ha="left", va="bottom", wrap=True)


def _finish(fig: Any, plt: Any, path: Path, roots: Mapping[str, str], top: float) -> None:
    """Lay out the figure above a 0.4 in footer strip (the provenance text may wrap to three lines), write the footer
    and the PNG (``FIG_WIDTH_IN`` x ``FIG_DPI`` = 1000 px wide) and close the figure."""
    fig.tight_layout(rect=(0, 0.4 / fig.get_figheight(), 1, top))
    _footer(fig, roots)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def _paired_axis(ax: Any, seed_df: pd.DataFrame, summ: pd.DataFrame, comparison: str, metric: str,
                 arms: Sequence[str], q: int, labels: Mapping[str, str]) -> None:
    """One panel: per arm the seed dots, the mean and its 95% percentile bootstrap interval of a paired difference."""
    for i, arm in enumerate(arms):
        sv = seed_df[(seed_df["comparison"] == comparison) & (seed_df["arm"] == arm) & (seed_df["q"] == q)
                     & (seed_df["metric"] == metric)]
        if len(sv):
            jit = np.linspace(-0.18, 0.18, len(sv))
            ax.scatter(sv["diff"], np.full(len(sv), float(i)) + jit, s=12, alpha=0.55,
                       color=ACTOR_COLORS.get(parse_arm(arm)[0], "tab:blue"))
        rw = summ[(summ["comparison"] == comparison) & (summ["arm"] == arm) & (summ["q"] == q)
                  & (summ["metric"] == metric)]
        if len(rw) and np.isfinite(rw["mean"].iloc[0]) and np.isfinite(rw["ci_mean_lo"].iloc[0]):
            m, lo, hi = float(rw["mean"].iloc[0]), float(rw["ci_mean_lo"].iloc[0]), float(rw["ci_mean_hi"].iloc[0])
            ax.errorbar([m], [i], xerr=[[max(m - lo, 0.0)], [max(hi - m, 0.0)]], fmt="D", color="k", capsize=3, ms=5,
                        lw=1.4)
    ax.axvline(0.0, color="tab:red", lw=0.8, ls="--")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([labels.get(a, a) for a in arms], fontsize=7)
    ax.invert_yaxis()
    ax.grid(alpha=0.25)


def fig_paired(path: Path, seed_df: pd.DataFrame, summ: pd.DataFrame, comparison: str,
               metrics: Sequence[Tuple[str, str]], arms: Sequence[str], qs: Sequence[int], title: str,
               roots: Mapping[str, str], labels: Optional[Mapping[str, str]] = None) -> None:
    """Paired differences per arm (seed dots, mean, 95% bootstrap interval): one row per metric, one column per q."""
    plt = R1._plt()
    labels = labels or {}
    fig, axes = plt.subplots(len(metrics), len(qs), figsize=(FIG_WIDTH_IN, (0.42 * len(arms) + 1.5) * len(metrics)
                                                             + 0.9), squeeze=False)
    for i, (metric, mlabel) in enumerate(metrics):
        for j, q in enumerate(qs):
            ax = axes[i][j]
            _paired_axis(ax, seed_df, summ, comparison, metric, arms, q, labels)
            ax.set_title("q = %d: %s" % (q, mlabel), fontsize=8)
            if i == len(metrics) - 1:
                ax.set_xlabel("paired difference (arm - baseline)", fontsize=7)
    fig.suptitle(textwrap.fill(title, 110), fontsize=8)
    _finish(fig, plt, path, roots, 0.94)


def _shade_segments(ax: Any, win: R2.Windows, label: bool) -> None:
    """Ramp / hold / decay shaded (the D3 table), names above the first panel."""
    for (name_, a_, b_), shade in zip(win.segments()[1:], ("0.92", "0.97", "0.88")):
        ax.axvspan(a_ - 1, b_, color=shade, lw=0)
        if label:
            ax.text(0.5 * (a_ + b_), 0.985, name_, transform=ax.get_xaxis_transform(), fontsize=6, ha="center",
                    va="top", color="0.35")


def fig_trajectories(path: Path, tba: pd.DataFrame, win: R2.Windows, actors: Sequence[str], starts_: str,
                     scales: Sequence[int], qs: Sequence[int], roots: Mapping[str, str]) -> int:
    """e_hat_2(0), the smoothing part, the remainder and w_eff against the update (``traj_from`` ..) for one sampler:
    seed mean with a min-max band, one line per (actor, s) (colour = actor, dashes = s = 16), ramp / hold / decay
    shaded.

    Returns:
        Number of lines drawn.
    """
    plt = R1._plt()
    quantities = [("e2_at_0", "e_hat_2(0) (effort)"), ("smoothing", "smoothing part (effort)"),
                  ("remainder", "remainder (effort)"), ("w_eff", "w_eff = gap / tent slope (units of d)")]
    fig, axes = plt.subplots(len(quantities), len(qs), figsize=(FIG_WIDTH_IN, 2.4 * len(quantities) + 0.9),
                             sharex=True, squeeze=False)
    n_lines = 0
    for i, (col, lab) in enumerate(quantities):
        for j, q in enumerate(qs):
            ax = axes[i][j]
            _shade_segments(ax, win, i == 0)
            for a in actors:
                for s in scales:
                    g = tba[(tba["arm"] == arm_name(a, starts_, s)) & (tba["q"] == q)]
                    if g.empty:
                        continue
                    x = g["local"].to_numpy(dtype=float)
                    ax.plot(x, g[col + "_mean"], color=ACTOR_COLORS[a], ls=S_STYLE.get(s, ":"), lw=1.1,
                            label="%s s=%d" % (a, s))
                    ax.fill_between(x, g[col + "_min"], g[col + "_max"], color=ACTOR_COLORS[a], alpha=0.08, lw=0)
                    n_lines += 1
            if col == "e2_at_0":
                ax.axhline({50: 70.0, 60: 58.3333333333}.get(int(q), float("nan")), color="k", lw=0.6, ls=":")
            ax.set_title("q = %d" % q, fontsize=8)
            ax.set_ylabel(lab, fontsize=7)
            ax.grid(alpha=0.2)
            if i == 0 and j == 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6, ncol=2, loc="best")
            if i == len(quantities) - 1:
                ax.set_xlabel("terminal-stage local update", fontsize=7)
    fig.suptitle(textwrap.fill(
        "Descriptive, %s starts: the tie decomposition and the rounding width along the landing (mean over seeds, "
        "band = min-max; colour = actor, dashed = s = 16; shaded = ramp / hold / decay; dotted = e2*(0))"
        % START_NAME.get(starts_, starts_), 110), fontsize=7)
    _finish(fig, plt, path, roots, 0.95)
    return n_lines


def fig_tie_profile(path: Path, tp: pd.DataFrame, actors: Sequence[str], starts: Sequence[str], scales: Sequence[int],
                    qs: Sequence[int], roots: Mapping[str, str]) -> int:
    """e_hat_2(d) against the closed form e2*(d) near the tie (|d| <= 30) at the final-tier freeze: seed mean with a
    min-max band per actor; one panel per (starts, s, q).

    Returns:
        Number of curves drawn.
    """
    plt = R1._plt()
    combos = [(k, s) for k in starts for s in scales]
    fig, axes = plt.subplots(len(combos), len(qs), figsize=(FIG_WIDTH_IN, 3.0 * len(combos) + 0.9), squeeze=False)
    n = 0
    for i, (k, s) in enumerate(combos):
        for j, q in enumerate(qs):
            ax = axes[i][j]
            closed = None
            for a in actors:
                g = tp[(tp["arm"] == arm_name(a, k, s)) & (tp["q"] == q)]
                if g.empty:
                    continue
                ax.plot(g["d"], g["e_hat_mean"], color=ACTOR_COLORS[a], lw=1.4, label=a)
                ax.fill_between(g["d"], g["e_hat_min"], g["e_hat_max"], color=ACTOR_COLORS[a], alpha=0.12, lw=0)
                closed = g
                n += 1
            if closed is not None:
                ax.plot(closed["d"], closed["g2"], color="k", ls="--", lw=1.0, label="closed form e2*(d)")
            ax.set_title("%s starts, s = %d, q = %d" % (START_NAME.get(k, k), s, q), fontsize=8)
            ax.set_xlabel("d", fontsize=7)
            ax.set_ylabel("effort", fontsize=7)
            ax.grid(alpha=0.25)
            if i == 0 and j == 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6, loc="lower center")
    fig.suptitle(textwrap.fill(
        "Descriptive: learned e_hat_2(d) at the final-tier terminal freeze against the closed form near the tie "
        "(seed mean, band = min-max over seeds)", 110), fontsize=8)
    _finish(fig, plt, path, roots, 0.96)
    return n


def fig_first_layer(path: Path, fls: pd.DataFrame, actors: Sequence[str], starts: Sequence[str],
                    scales: Sequence[int], qs: Sequence[int], roots: Mapping[str, str]) -> int:
    """Max |w| of the first-layer d-weights (units of d / B; ten times the stored weight for ``t10``) over training,
    one line per (actor, s): seed median with an inter-quartile band, at every weight export of the terminal stage.

    Returns:
        Number of lines drawn.
    """
    plt = R1._plt()
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(len(starts), len(qs), figsize=(FIG_WIDTH_IN, 3.3 * len(starts) + 0.9), sharex=True,
                             squeeze=False)
    n = 0
    for i, k in enumerate(starts):
        for j, q in enumerate(qs):
            ax = axes[i][j]
            for a in actors:
                for s in scales:
                    g = fls[(fls["arm"] == arm_name(a, k, s)) & (fls["q"] == q)]
                    if g.empty:
                        continue
                    x = g["local"].to_numpy(dtype=float)
                    ax.plot(x, g["w_abs_max_median"], color=ACTOR_COLORS[a], ls=S_STYLE.get(s, ":"), lw=1.2,
                            label="%s s=%d" % (a, s))
                    ax.fill_between(x, g["w_abs_max_q25"], g["w_abs_max_q75"], color=ACTOR_COLORS[a], alpha=0.10, lw=0)
                    n += 1
            ax.axhline(1.0, color="0.5", lw=0.6, ls=":")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: "%g" % v if v in (0.2, 0.3, 0.5, 2, 3, 5, 20, 30)
                                                       else ""))
            ax.set_title("%s starts, q = %d" % (START_NAME.get(k, k), q), fontsize=8)
            ax.set_ylabel("max |w| (units of d / B)", fontsize=7)
            ax.grid(alpha=0.25, which="both")
            if i == len(starts) - 1:
                ax.set_xlabel("terminal-stage local update (every weight export)", fontsize=7)
            if i == 0 and j == 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle(textwrap.fill(
        "Descriptive: largest first-layer weight on the d input (d / B units; t10: ten times the stored weight), seed "
        "median and inter-quartile band; the s = 1 and s = 16 runs of an actor are identical up to local update 2001",
        110), fontsize=8)
    _finish(fig, plt, path, roots, 0.95)
    return n


def fig_change_scatter(path: Path, seed_level: pd.DataFrame, actors: Sequence[str], starts: Sequence[str],
                       qs: Sequence[int], roots: Mapping[str, str]) -> int:
    """Per-run change of the remainder against the change of the smoothing part (s16 - s1 within one (actor, starts),
    per (q, seed)), coloured by actor, marked by starts, with the line on which the two cancel (no change of the gap).

    Returns:
        Number of points drawn.
    """
    plt = R1._plt()
    fig, axes = plt.subplots(1, len(qs), figsize=(FIG_WIDTH_IN, 5.2), squeeze=False)
    n_pts = 0
    for j, q in enumerate(qs):
        ax = axes[0][j]
        sub = seed_level[(seed_level["comparison"] == LABEL_NL) & (seed_level["q"] == q)]
        sm = sub[sub["metric"] == "smoothing"].set_index(["arm", "seed"])["diff"]
        rm = sub[sub["metric"] == "remainder"].set_index(["arm", "seed"])["diff"]
        both = sm.index.intersection(rm.index)
        for a in actors:
            for k in starts:
                arm = arm_name(a, k, S_HIGH)
                idx = [kk for kk in both if kk[0] == arm]
                if not idx:
                    continue
                ax.scatter([sm.loc[kk] for kk in idx], [rm.loc[kk] for kk in idx],
                           marker=START_MARKER.get(k, "o"), s=26, alpha=0.75, color=ACTOR_COLORS[a],
                           edgecolor="k" if k == "st" else "none", label="%s %s" % (a, k))
                n_pts += len(idx)
        px = np.array([sm.loc[kk] for kk in both], dtype=float)
        py = np.array([rm.loc[kk] for kk in both], dtype=float)
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
        ax.set_xlabel("change of the smoothing part (s16 - s1), effort units", fontsize=8)
        ax.set_ylabel("change of the remainder (s16 - s1), effort units", fontsize=8)
        ax.grid(alpha=0.2)
        if j == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)
    fig.suptitle(textwrap.fill(
        "Descriptive: per-run change of the remainder against the change of the smoothing part (s16 - s1 within one "
        "actor and sampler; colour = actor, circle = bin-balanced, triangle = stratified); on the dashed line the two "
        "cancel (no change of the gap)", 110), fontsize=8)
    _finish(fig, plt, path, roots, 0.93)
    return n_pts


# --------------------------------------------------------------------------------------------- summary text
def _g(x: Any) -> str:
    """Compact signed number."""
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else "%+.5f" % float(x)


def summary_text(roots: Mapping[str, str], actors: Sequence[str], crit: pd.DataFrame, trans: pd.DataFrame,
                 inter: pd.DataFrame, quad: pd.DataFrame, comp: pd.DataFrame, qs: Sequence[int], seeds: Sequence[int],
                 command: Optional[Sequence[str]] = None) -> str:
    """The CLI summary (also ``summary.txt``): provenance, the criterion, the transmission, interaction and quadrature
    headline numbers, the run status.

    Args:
        roots: The resolved run roots.
        actors: The actors that were analysed.
        crit: ``criterion.csv``.
        trans: ``transmission.csv``.
        inter: ``interaction.csv`` (the summary table).
        quad: ``quadrature_check.csv``.
        comp: ``completeness.csv``.
        qs: q values.
        seeds: Development seeds.
        command: Command line to record.
    """
    L: List[str] = ["MS-R3 analysis (prompt D5, D6, section 3.3)"]
    if command:
        L.append("command: " + " ".join(str(c) for c in command))
    L.append("independent check: python tools/ms/r3_blind_criterion.py --analysis-dir <the --out directory>")
    L.append("roots: MS-R3 pilot=%s | MS-R2 pilot=%s | parents_A=%s | rehearsal_v2_0=%s" % (
        roots["pilot"], roots["ms_r2_pilot"], roots["parents"], roots["rehearsal"]))
    L.append("actors: %s" % " ".join(actors))
    L.append("bootstrap: %d resamples, numpy.random.default_rng(%d), one fresh generator per (q, statistic)" %
             (N_BOOT, BOOT_SEED))
    L.append("")
    L.append("== PRE-REGISTERED CRITERION: %s ==" % CRITERION_NOTE)
    L.append("primary metric %s of the frozen terminal-stage candidate; (a) CI of the mean paired difference "
             "v - t1 (same starts and s) below 0 at BOTH q; (b) no run passing G-A with its G-N eta part under t1 "
             "fails it under v" % PRIMARY)
    lines, any_ci0 = R1._criterion_lines(crit, qs, seeds)
    L.extend(lines)
    if any_ci0:
        L.append("NOTE: " + NO_EFFECT_SENTENCE)
    L.append("")
    L.append("== TRANSMISSION (descriptive): mean(d gap) / mean(d smoothing), d = s16 - s1 within one (actor, starts); "
             "1 = the smoothing change reaches the gap, 0 = the remainder offsets it ==")
    for r in trans.itertuples():
        L.append("%-12s q=%d n=%d  d gap %s  d smoothing %s  ratio %s [%s, %s]  (%d resamples kept)" % (
            r.arm, r.q, r.n_pairs, _g(r.mean_d_gap), _g(r.mean_d_smoothing), _g(r.ratio), _g(r.ci_lo), _g(r.ci_hi),
            r.n_boot_valid))
    L.append("")
    L.append("== INTERACTION (descriptive): (v_s16 - v_s1) - (t1_s16 - t1_s1) per (starts, q, seed) ==")
    for r in inter.itertuples():
        L.append("%-5s %s q=%d %-24s n=%d  mean %s [%s, %s]  median %s" % (
            r.actor, r.starts, r.q, r.metric, r.n_pairs, _g(r.mean), _g(r.ci_mean_lo), _g(r.ci_mean_hi), _g(r.median)))
    L.append("")
    L.append("== QUADRATURE CHECK (descriptive, from arm means): observed gap(s16) against the quadrature and the "
             "additive prediction ==")
    for r in quad.itertuples():
        L.append("%-5s %s q=%d  gap s1 %s -> s16 observed %s | quadrature %s (err %s) | additive %s (err %s) | "
                 "closer: %s%s" % (
                     r.actor, r.starts, r.q, _g(r.gap_s1), _g(r.gap_s16), _g(r.quadrature_pred),
                     _g(r.abs_err_quadrature), _g(r.additive_pred), _g(r.abs_err_additive), r.closer or "n/a",
                     "  [F2 < 0: F = 0]" if r.F2_negative else ""))
    L.append("")
    L.append("== RUN STATUS ==")
    bad = comp[(comp["n_done"] != comp["n_planned"])]
    if bad.empty:
        L.append("every planned run of every arm and reference is done")
    for r in bad.itertuples():
        L.append("%s q=%d: planned %d, done %d, failed %d, running %d, incomplete %d, missing %d -> %s" % (
            r.arm, r.q, r.n_planned, r.n_done, r.n_failed, r.n_running, r.n_incomplete, r.n_missing, r.not_done_runs))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------- driver
def not_stored_by_design() -> Dict[str, List[str]]:
    """Quantities that a reference row does not have (NaN in ``per_run.csv``), by reference."""
    return {
        PARENTS: ["smoothing, remainder, smoothing_over_formula and smoothed_e_pred_0 (final_v2.json carries no "
                  "smoothed_game; NaN; gap is defined)", "conc_scale_final (no noise record; NaN)",
                  "D3 columns t2_R_*, t2_Rtail_*, t2_C_*, t2_s_* (no one-step best-response effort stored; NaN unless "
                  "--calibration-root)", "every stage-1 column (stage 1 untrained)",
                  "rule record, start shares, clamp counts (NaN)",
                  "first-layer weight columns (not read for references)"],
        REHEARSAL: ["conc_scale_final (no noise record; NaN)", "D3 columns t*_R_*, t*_C_*, t*_s_* (NaN unless "
                    "--calibration-root)", "rule record, start shares, clamp counts (NaN)",
                    "t*_minibatch_steps (not stored in v2_run_summary.json; NaN)",
                    "t*_episodes = updates x episodes_per_update (derived)",
                    "first-layer weight columns (not read for references)"],
        "NL_*": ["first-layer weight columns and the actor-variant cross-check (not read for the MS-R2 re-run "
                 "references; their exports are those of the t1 arms, check C-MS5)"]}


def run_analysis(roots: Mapping[str, str], out_dir: Path, qs: Sequence[int] = QS, seeds: Sequence[int] = SEEDS,
                 actors: Sequence[str] = ACTORS, protocol: Path = R1.DEFAULT_PROTOCOL, win: R2.Windows = R2.Windows(),
                 figures: bool = True, argv: Optional[Sequence[str]] = None, starts: Sequence[str] = STARTS,
                 scales: Sequence[int] = SCALES) -> Tuple[Dict[str, pd.DataFrame], bool]:
    """Extract, tabulate and write every output.

    Args:
        roots: ``{pilot, ms_r2_pilot, parents, rehearsal[, calibration]}`` run roots.
        out_dir: Output directory; must be absent or empty.
        qs: q values.
        seeds: Development seeds.
        actors: The actors that were run (``t1`` required).
        protocol: ``protocols/v2_T2_locked_v2_0.json`` (the gate thresholds and the game records).
        win: Segment boundaries, the first update of the trajectory tables and the last terminal-stage update.
        figures: Whether to draw the figures.
        argv: Command line recorded in ``analysis_info.json``.
        starts: Samplers of the arms (the pre-registered design has both; fewer only for reduced test waves).
        scales: Noise-landing scales of the arms (``1`` and ``16``).

    Returns:
        ``(tables, all_done)`` with ``all_done`` True when every planned run of every arm and reference is done.

    Raises:
        ValueError: On an invalid actor set.
        FileExistsError: If ``out_dir`` exists and is not empty (an analysis never overwrites).
    """
    actors = check_actors(actors)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("refusing to overwrite: %s exists and is not empty" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    th, proto_sha = R1.load_thresholds(protocol)
    proto = json.loads(Path(protocol).read_text())
    rid = R1.roots_id(roots)
    arms = r3_arms(actors, starts, scales)
    arms_all = arms + list(MS_R2_ARMS)
    df = extract_all(roots, qs, seeds, th, proto, actors, starts, scales)
    fl = first_layer_table(df, win, proto, arms)
    df = attach_last_export(df, fl, win)
    # ---- the pre-registered criterion and the secondary tables (all with the MS-R3 bootstrap seed)
    prim_cmp = primary_comparisons(actors, starts, scales)
    nl_cmp = noise_landing_comparisons(actors, starts)
    st_cmp = starts_comparisons(actors, scales) if len(starts) == 2 else []
    par_cmp = [(a, PARENTS) for a in arms]
    reh_cmp = [(a, REHEARSAL) for a in arms]
    ref_cmp = ms_r2_comparisons(actors, starts, scales)
    p_t1, sl_t1 = paired_tables(df, prim_cmp, SECONDARY_METRICS, qs, seeds, LABEL_T1, CRITERION_NOTE)
    crit = criterion_table(df, p_t1, prim_cmp, qs, seeds, LABEL_T1, CRITERION_NOTE)
    p_nl, sl_nl = paired_tables(df, nl_cmp, SECONDARY_METRICS, qs, seeds, LABEL_NL)
    p_st, sl_st = paired_tables(df, st_cmp, SECONDARY_METRICS, qs, seeds, LABEL_STARTS)
    p_par, sl_par = paired_tables(df, par_cmp, REF_METRICS, qs, seeds, LABEL_PARENTS)
    crit_par = criterion_table(df, p_par, par_cmp, qs, seeds, LABEL_PARENTS, PARENTS_NOTE)
    p_reh, sl_reh = paired_tables(df, reh_cmp, R1.S1_METRICS, qs, seeds, LABEL_REHEARSAL)
    p_ref, sl_ref = paired_tables(df, ref_cmp, CONTEXT_METRICS, qs, seeds, LABEL_MS_R2)
    p_s1t1, sl_s1t1 = paired_tables(df, prim_cmp, R1.S1_METRICS, qs, seeds, LABEL_STAGE1_T1)
    seed_level = pd.concat([sl_t1, sl_nl, sl_st, sl_par, sl_reh, sl_ref, sl_s1t1], ignore_index=True)
    trans, trans_seed = transmission_tables(df, nl_cmp, qs, seeds)
    inter, inter_seed = interaction_tables(df, actors, starts, qs, seeds)
    comp = R1.completeness_table(df, arms_all, qs, seeds)
    all_done = bool((comp["n_done"] == comp["n_planned"]).all())
    # ---- the decomposition along the run
    traj = trajectory_table(df, win, arms)
    tba = trajectory_by_arm_table(traj, arms, qs)
    fls = first_layer_summary_table(fl, arms, qs)
    fu = first_layer_units_table(df, win, proto, arms)
    tp = tie_profile_table(df, arms, qs)
    strata = R1.strata_table(df)
    with r3_bootstrap():
        stage1 = R1.stage1_table(df, arms_all, qs, th)
    tables: Dict[str, pd.DataFrame] = {
        "per_run": df, "criterion": crit, "paired_secondary": p_t1, "paired_seed_level": seed_level,
        "noise_landing": noise_landing_table(p_nl, trans, seeds), "transmission": trans,
        "transmission_seed_level": trans_seed, "interaction": inter, "interaction_seed_level": inter_seed,
        "starts_effect": p_st, "paired_vs_parents_A": p_par, "criterion_vs_parents_A": crit_par,
        "paired_vs_rehearsal_v2_0": p_reh, "paired_vs_ms_r2_nl": p_ref, "stage1": stage1, "stage1_vs_t1": p_s1t1,
        "stage1_R1_runs": R1.stage1_R1_runs_table(df, arms_all), "stage1_R1": R1.stage1_R1_table(df, arms_all, qs),
        "gates": R2.gates_table(df, arms_all, qs), "budget": R1.budget_table(df, arms_all, qs),
        "segments": R2.segment_table(df, win, arms, qs), "trajectory_checks": traj, "trajectory_by_arm": tba,
        "quadrature_check": quadrature_table(df, actors, starts, qs),
        "predictions": predictions_table(df, actors, starts, qs, seeds, trans),
        "freeze_decomposition": freeze_decomposition_table(df, arms + list(REF_ARMS), qs),
        "strata": strata, "strata_summary": R1.strata_summary_table(strata), "r0": r0_table(df, arms_all, qs, proto),
        "r0_spearman": r0_spearman_table(df, traj, actors, qs), "first_layer_weights": fl,
        "first_layer_summary": fls, "first_layer_units": fu, "tie_profile": tp,
        "arm_summary": arm_summary_table(df, arms_all, qs), "completeness": comp}
    for name, t in tables.items():
        R1.write_csv(t, out_dir / ("%s.csv" % name), rid)
    # ---- figures (a failing figure is recorded, not raised)
    fig_status: Dict[str, str] = {}
    if figures:
        fdir = out_dir / "figures"
        fdir.mkdir(exist_ok=True)
        lab = {a: "%s (vs %s)" % (a, b) for a, b in prim_cmp}
        lab_nl = {a: "%s (vs %s)" % (a, b) for a, b in nl_cmp}
        jobs: List[Tuple[str, Any]] = [
            ("paired_abs_peak_vs_t1.png", lambda p: fig_paired(
                p, sl_t1, p_t1, LABEL_T1, [(PRIMARY, "|peak error|, criterion statistic")],
                [a for a, _ in prim_cmp], qs,
                "Primary criterion part (a) (pre-registered statistic; the figure is descriptive): paired difference "
                "of |peak error|, v - t1 (same starts and s); dots = seeds, diamond = mean, bar = 95% percentile "
                "bootstrap interval", roots, lab)),
            ("paired_secondary_vs_t1.png", lambda p: fig_paired(
                p, sl_t1, p_t1, LABEL_T1, [("smoothing", "smoothing part"), ("remainder", "remainder"),
                                           ("gap", "gap"), ("w_eff", "w_eff (units of d)")],
                [a for a, _ in prim_cmp], qs,
                "Descriptive: paired differences of the decomposition (effort units) and of the effective rounding "
                "width, v - t1 (same starts and s)", roots, lab)),
            ("paired_noise_landing.png", lambda p: fig_paired(
                p, sl_nl, p_nl, LABEL_NL, [("smoothing", "smoothing part"), ("remainder", "remainder"),
                                           ("gap", "gap")],
                [a for a, _ in nl_cmp], qs,
                "Descriptive: the noise landing within each actor and sampler, paired difference s = 16 minus s = 1",
                roots, lab_nl)),
            ("paired_abs_peak_vs_parents_A.png", lambda p: fig_paired(
                p, sl_par, p_par, LABEL_PARENTS, [(PRIMARY, "|peak error|")], arms, qs,
                "Descriptive: paired difference of |peak error|, arm - parents_A", roots)),
            ("tie_profile_near_tie.png", lambda p: fig_tie_profile(p, tp, actors, starts, scales, qs, roots)),
            ("first_layer_d_weights.png", lambda p: fig_first_layer(p, fls, actors, starts, scales, qs, roots)),
            ("scatter_remainder_vs_smoothing_change.png", lambda p: fig_change_scatter(
                p, sl_nl, actors, starts, qs, roots))]
        for k in starts:
            jobs.append(("trajectory_decomposition_%s.png" % k, lambda p, k=k: fig_trajectories(
                p, tba, win, actors, k, scales, qs, roots)))
        for fname, fn in jobs:
            try:
                fn(fdir / fname)
                fig_status[fname] = "ok"
            except Exception as exc:  # noqa: BLE001
                fig_status[fname] = "FAILED: %s: %s" % (type(exc).__name__, exc)
                print("[warn] figure %s failed: %s: %s" % (fname, type(exc).__name__, exc), file=sys.stderr)
    try:
        import matplotlib
        mpl_version = matplotlib.__version__
    except Exception:  # noqa: BLE001
        mpl_version = None
    info = {
        "tool": "tools/ms/r3_analysis.py", "roots": dict(roots), "roots_id": rid, "protocol": str(protocol),
        "protocol_sha256": proto_sha, "thresholds": th.__dict__, "boot_seed": BOOT_SEED, "n_boot": N_BOOT,
        "bootstrap_scheme": "fresh numpy.random.default_rng(seed) per (q, statistic); idx = rng.integers(0, n, "
                            "size=(n_boot, n)); 2.5 / 97.5 percentiles of the resampled mean (or median); the "
                            "transmission ratio is recomputed on every resample (resamples with |mean d smoothing| <= "
                            "%g are dropped)" % DENOM_TOL,
        "qs": list(qs), "seeds": list(seeds), "actors": list(actors), "starts": list(starts), "scales": list(scales),
        "arms": arms, "references": list(REF_ARMS),
        "windows": {"ramp_first": win.ramp_first, "ramp_last": win.ramp_last, "hold_last": win.hold_last,
                    "decay_last": win.decay_last, "traj_from": win.traj_from,
                    "segments": [list(x) for x in win.segments()]},
        "criterion": {
            "primary_metric": PRIMARY, "comparisons": [list(c) for c in prim_cmp],
            "a": "ci_mean_hi < 0 of the mean paired difference v - t1 (same starts and s) at BOTH q",
            "b": "no run passing G-A (final tier) with the eta part of G-N (|eta_dev - eta_final| <= %g) under t1 "
                 "fails it under v" % th.n_eta, "note": CRITERION_NOTE,
            "parts_per_arm": {r.arm: {"a_met": bool(r.a_met), "b_status": r.b_status, "overall": r.overall}
                              for r in crit.itertuples()}},
        "decomposition": "gap = g2_at_0 - e2_at_0; smoothing = g2_at_0 - smoothed_e_pred_0; remainder = "
                         "smoothed_e_pred_0 - e2_at_0 (final tier, utils.ms_noise.decompose_gap); *_rel = / g2_at_0; "
                         "smoothing_over_formula = smoothing / (g2_at_0 * sigma_effort_at_0_t2 / (sqrt(pi) * q)); "
                         "cross-checked against the rule_log.json noise record (a disagreement is a flag)",
        "tie_metrics": {
            "peak_locfree": "argmax over the whole final-tier recovery grid of recovery_e2 (run/run_ms_stagewise.py "
                            "stage2_peak_locfree_*): peak_locfree_rel_err = (max e_hat_2 - g2_at_0) / g2_at_0, "
                            "peak_locfree_argmax_d = d of the maximum",
            "sym_err": "max over recovery nodes with |d| < 2q of |e_hat_2(d) - e_hat_2(-d)| (effort units; _rel: "
                       "/ g2_at_0)",
            "w_eff": "gap / (g2_at_0 / (2q)) in units of d (tent slope g2_at_0 / 2q)",
            "R0_over_peak": "t2_R0_{final,dev} / stage2_peak_rel_err_abs; linearised value 2k / (2k + a), "
                            "a = DW / (4 q^2)"},
        "first_layer_weights": {
            "column": "actor.l1.weight[:, 1] (column 0 is the stage feature)",
            "units": "d / B with B = (e_max - e_min) + 2q: the stored weight for t1 and relu, TEN times the stored "
                     "weight for t10 (the variant is read from the export's actor_variant entry, absent = t1)",
            "last_export": "local update %d" % win.decay_last,
            "rows": "first_layer_weights.csv holds every terminal-stage export (every 25 updates), a superset of the "
                    "requested rows (local >= %d, and every 100th update before)" % win.traj_from},
        "transmission": TRANSMISSION_NOTE,
        "quadrature_check": "per (actor, starts, q) from arm means: F2 = gap(s1)^2 - smoothing(s1)^2 (negative: "
                            "F = 0 and F2_negative); quadrature = sqrt(F2 + smoothing(s16)^2); additive = gap(s1) - "
                            "(smoothing(s1) - smoothing(s16)); closer = smaller absolute error against gap(s16)",
        "status_counts": {a: {s: int(((df["arm"] == a) & (df["status"] == s)).sum()) for s in R1.STATUSES}
                          for a in arms + list(REF_ARMS)},
        "all_done": all_done, "figures": fig_status, "command": list(argv) if argv is not None else None,
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "matplotlib": mpl_version, "not_stored_by_design": not_stored_by_design()}
    with open(out_dir / "analysis_info.json", "w") as f:
        json.dump(info, f, indent=1, sort_keys=True)
    (out_dir / "summary.txt").write_text(summary_text(roots, actors, crit, trans, inter, tables["quadrature_check"],
                                                      comp, qs, seeds, argv))
    return tables, all_done


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point (see the module docstring)."""
    p = argparse.ArgumentParser(description="MS-R3 analysis (D5, D6, section 3.3)")
    p.add_argument("--pilot-root", default=str(ROOT / "results" / "ms_r3" / "pilot"))
    p.add_argument("--ms-r2-pilot-root", required=True, help="MS-R2 pilot root (NL_* arms: C-MS5 context)")
    p.add_argument("--parents-root", required=True)
    p.add_argument("--rehearsal-root", required=True)
    p.add_argument("--calibration-root", default=None,
                   help="results/ms_r1/calibration: fills the D3 quantities of the two v2.0 comparators")
    p.add_argument("--actors", nargs="+", default=list(ACTORS),
                   help="the actors that were run, space or comma separated (t1 is required): "
                        "t1 relu t10 / t1,relu,t10")
    p.add_argument("--out", default=str(ROOT / "results" / "ms_r3" / "analysis"))
    p.add_argument("--qs", type=int, nargs="+", default=list(QS))
    p.add_argument("--seeds", nargs="+", default=["%d-%d" % (SEEDS[0], SEEDS[-1])])
    p.add_argument("--protocol", default=str(R1.DEFAULT_PROTOCOL))
    p.add_argument("--ramp-first", type=int, default=R2.Windows.ramp_first)
    p.add_argument("--ramp-last", type=int, default=R2.Windows.ramp_last)
    p.add_argument("--hold-last", type=int, default=R2.Windows.hold_last)
    p.add_argument("--decay-last", type=int, default=R2.Windows.decay_last)
    p.add_argument("--traj-from", type=int, default=R2.Windows.traj_from)
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    win = R2.Windows(a.ramp_first, a.ramp_last, a.hold_last, a.decay_last, a.traj_from)
    if not win.valid():
        print("[error] the segment boundaries need 1 <= ramp-first < ramp-last <= hold-last <= decay-last",
              file=sys.stderr)
        return 2
    try:
        actors = check_actors([x.strip() for tok in a.actors for x in tok.split(",")])
    except ValueError as exc:
        print("[error] %s" % exc, file=sys.stderr)
        return 2
    roots = {"pilot": str(Path(a.pilot_root).resolve()), "ms_r2_pilot": str(Path(a.ms_r2_pilot_root).resolve()),
             "parents": str(Path(a.parents_root).resolve()), "rehearsal": str(Path(a.rehearsal_root).resolve())}
    if a.calibration_root:
        roots["calibration"] = str(Path(a.calibration_root).resolve())
    for k in ("pilot", "ms_r2_pilot", "parents", "rehearsal"):
        if not Path(roots[k]).is_dir():
            print("[error] %s root %s is not a directory (a missing reference root is a stop-and-report)" %
                  (k, roots[k]), file=sys.stderr)
            return 2
    try:
        tables, all_done = run_analysis(roots, Path(a.out), tuple(a.qs), R1._parse_seeds(a.seeds), actors,
                                        Path(a.protocol), win, not a.no_figures,
                                        list(argv) if argv is not None else sys.argv)
    except FileExistsError as exc:
        print("[error] %s" % exc, file=sys.stderr)
        return 2
    sys.stdout.write((Path(a.out) / "summary.txt").read_text())
    if not all_done:
        print("[incomplete] at least one planned run is missing / failed / running / incomplete "
              "(see completeness.csv)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
