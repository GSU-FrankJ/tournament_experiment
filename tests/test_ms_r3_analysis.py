"""MS-R3 analysis tool tests (``tools/ms/r3_analysis.py`` and the independent ``tools/ms/r3_blind_criterion.py``).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r3_analysis.py -p no:cacheprovider -q

Sections: (1) arms, comparisons and actor sets; (2) synthetic per-run frames: the primary criterion parts (a) and (b) by hand
(a violation, a missing run, a baseline that does not pass), the fresh generator per (q, statistic) with seed 20261008, the
paired tables; (3) transmission, interaction and the quadrature check on constructed data (F^2 < 0 included, the preamble
numbers of MS-R2 reproduced from its per_run.csv); (4) the tie and weight metrics on constructed arrays and exports (a ``t10``
export reports ten times the stored weight); (5) the blind recomputation: agreement, tampered criterion / transmission /
interaction / per_run, unreadable inputs, independence from the analysis tool; (5b) ten jittered pairs per cell (the design size,
non-degenerate intervals, structural arm means): every table of ``run_analysis`` against a manual recomputation with a fresh
``default_rng(20261008)`` per interval, a spy that sees only that seed, the same run with every seed shifted (every interval column
of every table changes), transmission / interaction / criterion values by hand, P1-P4 by independent formulas; (6) one real tiny
wave (six reduced runs of one (q, seed): ``t1`` / ``relu`` / ``t10`` x s = 1, 16): extraction, the tie metrics against the
``gates.json`` record (the |d| < 2q window of the symmetry error pinned), the weight statistics against the exports (stage-1 exports
included as ``stage == 1`` rows, excluded from the summary and from the last-export columns), the reload of the last export at
d = 0 and d != 0, the trajectory / profile / summary tables, the strata (middle and tail included, tail max), the CLI end to end
with a genuine ``parents_A`` and ``rehearsal_v2_0`` run (the vs-parents / vs-rehearsal rows equal arm minus the right reference,
by hand), missing planned runs, a reduced actor set, no overwrite, argument errors.
"""

from __future__ import annotations

import ast
import json
import math
import os
import shutil
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import r1_analysis as R1  # noqa: E402
import r2_analysis as R2  # noqa: E402
import r3_analysis as R3  # noqa: E402
import r3_blind_criterion as BL  # noqa: E402

PROTO_PATH = ROOT / "protocols" / "v2_T2_locked_v2_0.json"
TH, _ = R1.load_thresholds(PROTO_PATH)
PROTO = json.loads(PROTO_PATH.read_text())
SEEDS4 = (10501, 10502, 10503, 10504)
QS = (50, 60)
BOOT = 20261008


# ====================================================================== 1. arms, comparisons, actor sets
def test_constants_are_the_preregistered_scheme():
    assert R3.BOOT_SEED == 20261008 and R3.N_BOOT == 10000 and BL.SEED == 20261008 and BL.N_RESAMPLES == 10000
    assert R3.ACTORS == ("t1", "relu", "t10") and R3.CONTROL == "t1" and R3.STARTS == ("bb", "st") and R3.SCALES == (1, 16)
    assert R3.parse_arm("relu_st_s16") == ("relu", "st", 16.0) and R3.parse_arm("NL_bb_s1")[0] == ""
    assert R3.parse_arm("t10_bb_s1") == ("t10", "bb", 1.0) and math.isnan(R3.parse_arm("MS_base2400")[2])
    assert R3.r3_arms(R3.ACTORS)[:4] == ["t1_bb_s1", "t1_bb_s16", "t1_st_s1", "t1_st_s16"] and len(R3.r3_arms(R3.ACTORS)) == 12
    assert R3.T10_D_SCALE == 10.0


def test_primary_comparisons_are_the_eight_rows_in_table_order_and_depend_on_the_actor_set():
    assert R3.primary_comparisons(R3.ACTORS) == [
        ("relu_bb_s1", "t1_bb_s1"), ("relu_bb_s16", "t1_bb_s16"), ("relu_st_s1", "t1_st_s1"), ("relu_st_s16", "t1_st_s16"),
        ("t10_bb_s1", "t1_bb_s1"), ("t10_bb_s16", "t1_bb_s16"), ("t10_st_s1", "t1_st_s1"), ("t10_st_s16", "t1_st_s16")]
    assert [a for a, _ in R3.primary_comparisons(("t1", "t10"))] == ["t10_bb_s1", "t10_bb_s16", "t10_st_s1", "t10_st_s16"]
    assert R3.primary_comparisons(("t1",)) == []
    assert R3.noise_landing_comparisons(("t1", "relu")) == [("t1_bb_s16", "t1_bb_s1"), ("t1_st_s16", "t1_st_s1"),
                                                            ("relu_bb_s16", "relu_bb_s1"), ("relu_st_s16", "relu_st_s1")]
    assert R3.starts_comparisons(("t1", "t10")) == [("t1_st_s1", "t1_bb_s1"), ("t1_st_s16", "t1_bb_s16"),
                                                    ("t10_st_s1", "t10_bb_s1"), ("t10_st_s16", "t10_bb_s16")]
    assert R3.ms_r2_comparisons(("t1", "relu")) == [("t1_bb_s1", "NL_bb_s1"), ("t1_bb_s16", "NL_bb_s16"),
                                                    ("t1_st_s1", "NL_st_s1"), ("t1_st_s16", "NL_st_s16")]


def test_actor_sets_are_validated_and_put_in_canonical_order():
    assert R3.check_actors(("t10", "t1")) == ("t1", "t10") and R3.check_actors(R3.ACTORS) == R3.ACTORS
    for bad in (("relu", "t10"), ("t1", "gelu"), ("t1", "t1", "relu"), ()):
        with pytest.raises(ValueError):
            R3.check_actors(bad)


# ====================================================================== 2. synthetic per-run frames
def indep_ci(d: Sequence[float], stat: str = "mean", seed: int = BOOT) -> Tuple[float, float]:
    """The pre-registered interval written out independently: fresh generator, indices (10000, n), 2.5 / 97.5."""
    a = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(10000, a.size))
    r = a[idx]
    s = r.mean(axis=1) if stat == "mean" else np.median(r, axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def blank(arm: str, q: int, seed: int, **kw: Any) -> Dict[str, Any]:
    """A done per-run row of every column (NaN / empty) with a passing G-A / G-N eta and the given overrides."""
    row: Dict[str, Any] = {c: ("" if c in R3.STR_COLS else float("nan")) for c in R3.COLUMNS}
    actor, starts, s = R3.parse_arm(arm)
    row.update(arm=arm, q=q, seed=seed, role="ms_arm" if actor else "comparator", status="done", status_info="",
               complete=True, run_dir="", flags="", eta_T_over_dw=0.003, eta_dev=0.0031,
               stage2_rmse_pos_over_g2_0=0.02, stage2_tail_mean_over_g2_0=0.005, g2_at_0=70.0 if q == 50 else 58.0,
               actor=actor, starts=starts, s=s)
    row.update(kw)
    return row


def settle(row: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute the gate columns of a done row from its component columns (thresholds of the protocol file)."""
    if row["status"] != "done":
        row["complete"] = False
        return row
    eta, rmse, tail = row["eta_T_over_dw"], row["stage2_rmse_pos_over_g2_0"], row["stage2_tail_mean_over_g2_0"]
    row["eta_dev_minus_final_abs"] = abs(row["eta_dev"] - eta)
    row["G_A_pass"] = bool(eta <= TH.eta and rmse <= TH.rmse and tail <= TH.tail)
    row["G_N_eta_pass"] = bool(row["eta_dev_minus_final_abs"] <= TH.n_eta)
    row["gate_pass"] = bool(row["G_A_pass"] and row["G_N_eta_pass"])
    row["stage2_peak_rel_err_signed"] = -row["stage2_peak_rel_err_abs"]
    return row


def t1_peak(starts: str, s: int, q: int, i: int) -> float:
    """The |peak| of a ``t1`` run (deterministic): seed index i, a little higher at q = 60 and under s = 16."""
    return 0.10 + 0.01 * i + (0.02 if q == 60 else 0.0) + (0.005 if s == 16 else 0.0) - (0.01 if starts == "st" else 0.0)


#: paired differences v - t1 of |peak| per (arm, q), four seeds each (see make_world)
DELTA: Dict[Tuple[str, int], List[float]] = {
    ("relu_bb_s1", 50): [-0.03, -0.04, -0.02, -0.05], ("relu_bb_s1", 60): [-0.02, -0.03, -0.01, -0.02],
    ("relu_bb_s16", 50): [-0.05, -0.05, -0.04, -0.05], ("relu_bb_s16", 60): [-0.02, 0.03, -0.01, 0.05],
    ("relu_st_s1", 50): [-0.02, -0.01, -0.03, -0.02], ("relu_st_s1", 60): [-0.01, -0.02, -0.01, -0.03],
    ("relu_st_s16", 50): [-0.03, -0.03, -0.02, -0.03], ("relu_st_s16", 60): [-0.02, -0.03, -0.01, -0.04],
    ("t10_bb_s1", 50): [0.01, 0.02, 0.01, 0.03], ("t10_bb_s1", 60): [0.02, 0.01, 0.03, 0.02],
    ("t10_bb_s16", 50): [-0.02, -0.03, -0.02, -0.03], ("t10_bb_s16", 60): [-0.03, -0.02, -0.03, -0.02],
    ("t10_st_s1", 50): [-0.02, -0.02, -0.03, -0.02], ("t10_st_s1", 60): [-0.02, -0.03, -0.02, -0.02],
    ("t10_st_s16", 50): [-0.04, -0.03, -0.04, -0.03], ("t10_st_s16", 60): [-0.03, -0.04, -0.03, -0.04]}
ACTOR_REM = {"t1": 0.0, "relu": -0.5, "t10": -0.3}


def make_world() -> pd.DataFrame:
    """Twelve arms x q x four seeds (hand-designed):

    * ``t1_bb_s1`` at q=50 seed 10502 does not pass G-A (eta 0.01): it is not a baseline pass, and a variant failing there
      is no violation;
    * ``relu_bb_s1``: every paired difference negative at both q, b holds -> met;
    * ``relu_bb_s16``: q=60 differences [-0.02, +0.03, -0.01, +0.05], mean +0.0125 -> part (a) not met;
    * ``relu_st_s1``: part (a) met, but its run at (q=50, 10503) fails the tail limit (0.03 > 0.02) -> violated, not met;
    * ``relu_st_s16``: the run at (q=60, 10504) is missing -> 3 pairs at q=60, b pending -> incomplete;
    * ``t10_bb_s1``: every difference positive -> (a) not met;
    * ``t10_bb_s16`` / ``t10_st_s1`` / ``t10_st_s16``: (a) met, b holds.
    """
    rows = []
    for actor in R3.ACTORS:
        for k in R3.STARTS:
            for s in R3.SCALES:
                arm = R3.arm_name(actor, k, s)
                for q in QS:
                    for i, sd in enumerate(SEEDS4):
                        peak = t1_peak(k, s, q, i) + (0.0 if actor == "t1" else DELTA[(arm, q)][i])
                        kw: Dict[str, Any] = {}
                        if (k, s, q, sd) == ("bb", 1, 50, 10502):
                            kw = {"eta_T_over_dw": 0.01, "eta_dev": 0.0101}
                        if (arm, q, sd) == ("relu_st_s1", 50, 10503):
                            kw["stage2_tail_mean_over_g2_0"] = 0.03
                        sm = 2.0 / math.sqrt(s) + 0.01 * i + (0.05 if q == 60 else 0.0)
                        rem = 1.5 + 0.1 * i + ACTOR_REM[actor] + (0.3 * math.log2(s) if k == "bb" else 0.1 * math.log2(s))
                        r = blank(arm, q, sd, stage2_peak_rel_err_abs=peak, smoothing=sm, remainder=rem, gap=sm + rem,
                                  stage1_rel_err_signed=0.01 * (i + 1), stage1_rel_err_abs=0.01 * (i + 1), **kw)
                        if (arm, q, sd) == ("relu_st_s16", 60, 10504):
                            r = blank(arm, q, sd, status="missing", complete=False)
                        rows.append(settle(r))
    return pd.DataFrame(rows, columns=R3.COLUMNS)


@pytest.fixture(scope="module")
def world() -> pd.DataFrame:
    return make_world()


def _crit(df: pd.DataFrame, comparisons: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
    comps = comparisons or R3.primary_comparisons(R3.ACTORS)
    p, _ = R3.paired_tables(df, comps, [(R3.PRIMARY, True)], QS, SEEDS4, R3.LABEL_T1)
    return R3.criterion_table(df, p, comps, QS, SEEDS4, R3.LABEL_T1, R3.CRITERION_NOTE)


def test_criterion_parts_a_and_b_by_hand(world):
    crit = _crit(world).set_index("arm")
    assert list(crit.index) == [a for a, _ in R3.primary_comparisons(R3.ACTORS)] and len(crit) == 8
    assert (crit["boot_seed"] == 20261008).all() and (crit["n_boot"] == 10000).all()
    assert (crit["baseline"] == [b for _, b in R3.primary_comparisons(R3.ACTORS)]).all()
    assert list(crit["actor"]) == ["relu"] * 4 + ["t10"] * 4 and list(crit["starts"]) == ["bb", "bb", "st", "st"] * 2
    assert list(crit["s"]) == [1, 16, 1, 16] * 2
    # ---- relu_bb_s1: diffs q50 [-.03 -.04 -.02 -.05], q60 [-.02 -.03 -.01 -.02]
    r = crit.loc["relu_bb_s1"]
    assert r["n_pairs_q50"] == 4 and r["n_pairs_q60"] == 4
    assert r["mean_q50"] == pytest.approx(-0.035, abs=1e-12) and r["mean_q60"] == pytest.approx(-0.02, abs=1e-12)
    lo, hi = indep_ci([-0.03, -0.04, -0.02, -0.05])
    assert r["ci_mean_lo_q50"] == pytest.approx(lo, abs=1e-12) and r["ci_mean_hi_q50"] == pytest.approx(hi, abs=1e-12)
    assert hi < 0 and bool(r["a_q50"]) and bool(r["a_q60"]) and bool(r["a_met"]) and bool(r["a_complete"])
    assert r["b_status"] == "holds" and r["b_violations"] == "" and r["overall"] == "met"
    assert r["n_base_pass"] == 7                       # t1_bb_s1 at q=50 seed 10502 does not pass G-A
    # ---- relu_bb_s16: q60 diffs [-.02 +.03 -.01 +.05]: mean +0.0125, the interval reaches above 0 -> (a) not met
    r = crit.loc["relu_bb_s16"]
    assert r["mean_q60"] == pytest.approx(0.0125, abs=1e-12)
    lo, hi = indep_ci([-0.02, 0.03, -0.01, 0.05])
    assert hi > 0 and r["ci_mean_hi_q60"] == pytest.approx(hi, abs=1e-12)
    assert bool(r["a_q50"]) and not bool(r["a_q60"]) and not bool(r["a_met"]) and r["overall"] == "not met"
    assert r["n_base_pass"] == 8
    # ---- relu_st_s1: (a) met at both q, but the run at q=50 seed 10503 passes under t1 and fails the tail limit under relu
    r = crit.loc["relu_st_s1"]
    assert bool(r["a_met"]) and r["b_violations"] == "q50/10503" and r["b_n_violations"] == 1
    assert r["b_status"] == "violated" and r["overall"] == "not met" and r["n_base_pass"] == 8
    # ---- relu_st_s16: the q=60 run of seed 10504 is missing: 3 pairs, (a) not complete, (b) pending -> incomplete
    r = crit.loc["relu_st_s16"]
    assert r["n_pairs_q60"] == 3 and r["n_pairs_q50"] == 4 and not bool(r["a_complete"])
    assert r["b_pending"] == "q60/10504" and r["b_n_pending"] == 1 and r["b_violations"] == ""
    assert r["b_status"] == "incomplete" and r["overall"] == "incomplete"
    lo, hi = indep_ci([-0.02, -0.03, -0.01])           # the three seeds that have a pair
    assert r["mean_q60"] == pytest.approx(-0.02, abs=1e-12) and r["ci_mean_hi_q60"] == pytest.approx(hi, abs=1e-12)
    # ---- t10_bb_s1: every difference positive -> not met; t10_bb_s16 / t10_st_*: met
    assert not bool(crit.loc["t10_bb_s1", "a_met"]) and crit.loc["t10_bb_s1", "overall"] == "not met"
    assert crit.loc["t10_bb_s1", "ci_mean_lo_q50"] > 0
    for arm in ("t10_bb_s16", "t10_st_s1", "t10_st_s16"):
        assert crit.loc[arm, "overall"] == "met" and bool(crit.loc[arm, "a_met"]), arm


def test_the_criterion_is_strict_below_zero(world):
    """ci_mean_hi == 0 does not meet (a): all differences 0 -> interval [0, 0]."""
    df = world.copy()
    m = (df["arm"] == "relu_bb_s1")
    base = df[df["arm"] == "t1_bb_s1"].set_index(["q", "seed"])[R3.PRIMARY]
    df.loc[m, R3.PRIMARY] = [base.loc[(q, s)] for q, s in zip(df.loc[m, "q"], df.loc[m, "seed"])]
    r = _crit(df).set_index("arm").loc["relu_bb_s1"]
    assert r["ci_mean_hi_q50"] == 0.0 and not bool(r["a_q50"]) and r["overall"] == "not met"


def test_a_failed_arm_run_counts_as_violation_and_a_running_one_as_pending(world):
    df = world.copy()
    i = df.index[(df["arm"] == "relu_bb_s1") & (df["q"] == 60) & (df["seed"] == 10501)][0]
    df.loc[i, ["status", "complete"]] = ["failed", False]
    j = df.index[(df["arm"] == "relu_bb_s1") & (df["q"] == 60) & (df["seed"] == 10502)][0]
    df.loc[j, ["status", "complete"]] = ["running", False]
    r = _crit(df).set_index("arm").loc["relu_bb_s1"]
    assert r["b_violations"] == "q60/10501" and r["b_pending"] == "q60/10502" and r["n_pairs_q60"] == 2
    assert r["b_status"] == "violated" and r["overall"] == "not met"


def test_a_missing_baseline_run_never_enters_a_pair(world):
    df = world.copy()
    i = df.index[(df["arm"] == "t1_st_s16") & (df["q"] == 50) & (df["seed"] == 10501)][0]
    df.loc[i, ["status", "complete"]] = ["failed", False]
    r = _crit(df).set_index("arm")
    assert r.loc["relu_st_s16", "n_pairs_q50"] == 3 and r.loc["t10_st_s16", "n_pairs_q50"] == 3
    assert r.loc["relu_st_s1", "n_pairs_q50"] == 4                    # the other (starts, s) pairs are untouched
    assert r.loc["relu_bb_s16", "n_pairs_q50"] == 4


def test_fresh_generator_per_q_and_statistic_makes_the_numbers_independent_of_table_order(world):
    """Every interval uses a fresh default_rng(20261008): the numbers do not depend on the order of the rows, of the
    comparisons, of the q values or on how many tables were built before."""
    ref = _crit(world).set_index("arm")
    rng = np.random.default_rng(3)
    shuffled = world.iloc[rng.permutation(len(world))].reset_index(drop=True)
    again = _crit(shuffled, list(reversed(R3.primary_comparisons(R3.ACTORS)))).set_index("arm")
    for arm in ref.index:
        for c in ("mean_q50", "ci_mean_lo_q50", "ci_mean_hi_q50", "mean_q60", "ci_mean_lo_q60", "ci_mean_hi_q60"):
            assert ref.loc[arm, c] == again.loc[arm, c], (arm, c)
    d = [-0.03, -0.04, -0.02, -0.05]
    a = R3.paired_summary(d, True)
    R3.paired_summary([1.0, 2.0, 3.0, 4.0], True)                    # a call in between does not move the next one
    b = R3.paired_summary(d, True)
    assert a["ci_mean_lo"] == b["ci_mean_lo"] and a["ci_mean_hi"] == b["ci_mean_hi"]
    assert (a["ci_mean_lo"], a["ci_mean_hi"]) == indep_ci(d)
    assert (a["ci_median_lo"], a["ci_median_hi"]) == indep_ci(d, "median")
    assert R3.paired_summary([1.0, 2.0, 3.0], None)["ci_mean_lo"] == indep_ci([1.0, 2.0, 3.0])[0]
    assert R3.summary_stats(d)["ci_mean_lo"] == indep_ci(d)[0]
    # the MS-R2 seed (20261007) gives other numbers on continuous data: the MS-R3 tool does not use it
    x = np.random.default_rng(11).normal(size=9)
    assert R3.paired_summary(x, None)["ci_mean_lo"] != R2.paired_summary(x, None)["ci_mean_lo"]
    assert R3.paired_summary(x, None)["ci_mean_lo"] == indep_ci(x)[0]


def test_stage1_table_uses_the_ms_r3_seed_inside_the_context_and_restores_the_ms_r1_function(world):
    before = R1.summary_stats
    with R3.r3_bootstrap():
        assert R1.summary_stats is R3.summary_stats
        s1 = R1.stage1_table(world, ["t1_bb_s1", "relu_bb_s1"], QS, TH).set_index(["arm", "q"])
    assert R1.summary_stats is before
    vals = world[(world["arm"] == "relu_bb_s1") & (world["q"] == 50)]["stage1_rel_err_signed"].to_numpy(dtype=float)
    lo, hi = indep_ci(vals)
    assert s1.loc[("relu_bb_s1", 50), "stage1_rel_err_signed_ci_lo"] == lo
    assert s1.loc[("relu_bb_s1", 50), "stage1_rel_err_signed_ci_hi"] == hi


def test_paired_tables_report_status_tags_and_sign_counts(world):
    comps = R3.primary_comparisons(R3.ACTORS)
    p, long = R3.paired_tables(world, comps, R3.SECONDARY_METRICS, QS, SEEDS4, R3.LABEL_T1)
    assert set(p["status"]) == {"descriptive"} and set(long["status"]) == {"descriptive"}
    r = p[(p["arm"] == "relu_bb_s16") & (p["q"] == 60) & (p["metric"] == R3.PRIMARY)].iloc[0]
    assert (r["n_pairs"], r["n_pos"], r["n_neg"], r["n_zero"], r["n_better"]) == (4, 2, 2, 0, 2)
    assert r["mean"] == pytest.approx(0.0125, abs=1e-12) and r["median"] == pytest.approx(0.01, abs=1e-12)
    assert (r["actor"], r["starts"], r["s"], r["baseline"]) == ("relu", "bb", 16, "t1_bb_s16")
    # UNEQUAL counts, so that a swapped sign convention is seen: all four differences negative / all four positive
    cell = lambda arm, q: p[(p["arm"] == arm) & (p["q"] == q) & (p["metric"] == R3.PRIMARY)].iloc[0]       # noqa: E731
    r = cell("relu_bb_s1", 50)                                          # diffs [-.03, -.04, -.02, -.05]
    assert (r["n_pairs"], r["n_pos"], r["n_neg"], r["n_zero"], r["n_better"]) == (4, 0, 4, 0, 4)
    assert r["mean"] == pytest.approx(-0.035, abs=1e-12) and r["median"] == pytest.approx(-0.035, abs=1e-12)
    r = cell("t10_bb_s1", 50)                                           # diffs [+.01, +.02, +.01, +.03]
    assert (r["n_pairs"], r["n_pos"], r["n_neg"], r["n_zero"], r["n_better"]) == (4, 4, 0, 0, 0)
    assert r["mean"] == pytest.approx(0.0175, abs=1e-12)
    # an exact tie is neither better nor worse: one difference made exactly 0 -> (n_pos, n_neg, n_zero, n_better) = (0, 3, 1, 3)
    tie = world.copy()
    m_arm, m_base = (tie["arm"] == "relu_bb_s1") & (tie["q"] == 50) & (tie["seed"] == 10501), (
        (tie["arm"] == "t1_bb_s1") & (tie["q"] == 50) & (tie["seed"] == 10501))
    tie.loc[m_arm, R3.PRIMARY] = float(tie.loc[m_base, R3.PRIMARY].iloc[0])
    pt, _ = R3.paired_tables(tie, comps, [(R3.PRIMARY, True)], QS, SEEDS4, R3.LABEL_T1)
    r = pt[(pt["arm"] == "relu_bb_s1") & (pt["q"] == 50)].iloc[0]
    assert (r["n_pairs"], r["n_pos"], r["n_neg"], r["n_zero"], r["n_better"]) == (4, 0, 3, 1, 3)
    pn, _ = R3.paired_tables(world, comps, [(R3.PRIMARY, True), ("gap", True)], QS, SEEDS4, R3.LABEL_T1, "NOTE")
    assert set(pn[pn["metric"] == R3.PRIMARY]["status"]) == {"NOTE"}
    assert set(pn[pn["metric"] == "gap"]["status"]) == {"descriptive"}
    # the gap change of relu_bb_s1 against t1_bb_s1 is the remainder offset -0.5 for every pair
    g = p[(p["arm"] == "relu_bb_s1") & (p["q"] == 50) & (p["metric"] == "gap")].iloc[0]
    assert g["mean"] == pytest.approx(-0.5, abs=1e-12) and g["ci_mean_hi"] < 0 and g["n_pairs"] == 4
    assert len(long[(long["arm"] == "relu_bb_s1") & (long["q"] == 50) & (long["metric"] == "gap")]) == 4
    # a metric that no pair has (w_eff is NaN in the world) is omitted, the first metric never is
    assert "w_eff" not in set(p["metric"]) and R3.PRIMARY in set(p["metric"])


def test_the_descriptive_tables_use_the_ms_r3_bootstrap_seed(world):
    gap = world[(world["arm"] == "relu_bb_s1") & (world["q"] == 50)]["gap"].to_numpy(dtype=float)
    fd = R3.freeze_decomposition_table(world, ["relu_bb_s1"], QS)
    r = fd[(fd["q"] == 50) & (fd["quantity"] == "gap")].iloc[0]
    assert (r["ci_mean_lo"], r["ci_mean_hi"]) == indep_ci(gap) and r["n"] == 4 and r["unit"] == "effort units"
    arm = R3.arm_summary_table(world, ["relu_bb_s1"], QS)
    pk = world[(world["arm"] == "relu_bb_s1") & (world["q"] == 50)][R3.PRIMARY].to_numpy(dtype=float)
    r = arm[(arm["arm"] == "relu_bb_s1") & (arm["q"] == 50) & (arm["metric"] == R3.PRIMARY)].iloc[0]
    assert (r["ci_mean_lo"], r["ci_mean_hi"]) == indep_ci(pk) and r["mean"] == pytest.approx(pk.mean())


def test_noise_landing_table_appends_the_transmission_rows_to_the_paired_changes(world):
    nl_cmp = R3.noise_landing_comparisons(R3.ACTORS)
    p, _ = R3.paired_tables(world, nl_cmp, R3.SECONDARY_METRICS, QS, SEEDS4, R3.LABEL_NL)
    trans, _ = R3.transmission_tables(world, nl_cmp, QS, SEEDS4)
    nl = R3.noise_landing_table(p, trans, SEEDS4)
    assert len(nl) == len(p) + len(trans) == len(p) + 12
    # the paired block is the paired table, unchanged and first (compared with the INPUT, not with itself) ...
    pd.testing.assert_frame_equal(nl.iloc[: len(p)][list(p.columns)].reset_index(drop=True), p.reset_index(drop=True),
                                  check_dtype=False)
    assert (nl.iloc[: len(p)]["metric"] != "transmission_ratio").all() and (nl.iloc[len(p):]["metric"] == "transmission_ratio").all()
    # ... and the 12 appended rows carry the transmission ratios worked out by hand: the smoothing part falls 2.0 -> 0.5 in
    # every run, the remainder rises by 0.3 log2(16) = 1.2 (bin-balanced) or 0.1 log2(16) = 0.4 (stratified), whatever the actor
    tx = nl[nl["metric"] == "transmission_ratio"].reset_index(drop=True)
    assert list(tx["arm"]) == [a for a, _ in nl_cmp for _ in QS] and list(tx["q"]) == list(QS) * 6
    for r_ in tx.itertuples():
        assert r_.mean == pytest.approx(0.2 if "_bb_" in r_.arm else 1.1 / 1.5, abs=1e-12), r_.arm
    for (_, a), (_, b) in zip(tx.iterrows(), trans.iterrows()):
        assert a["arm"] == b["arm"] and a["q"] == b["q"] and a["n_pairs"] == b["n_pairs"] and a["mean"] == b["ratio"]
        assert (a["ci_mean_lo"], a["ci_mean_hi"]) == (b["ci_lo"], b["ci_hi"]) and a["per_seed_ratios"] == b["per_seed_ratios"]
        assert a["median"] == b["median_per_seed_ratio"] and "s16 - s1" in a["note"] and a["comparison"] == R3.LABEL_NL
    assert R3.noise_landing_table(p, trans.iloc[0:0], SEEDS4).equals(p)                       # no transmission rows: unchanged


def test_trajectory_by_arm_aggregates_the_done_runs_per_update():
    rows = []
    for seed, status, scale in ((10501, "done", 1.0), (10502, "done", 2.0), (10503, "done", 4.0), (10504, "failed", 100.0)):
        for local in (1800, 1825):
            rows.append({c: float("nan") for c in R3.TRAJ_COLS} | {
                "arm": "relu_bb_s1", "q": 50, "seed": seed, "status": status, "update": local, "local": local,
                "e2_at_0": 60.0 * scale + local / 100.0, "smoothing": scale, "remainder": 2.0 * scale, "sigma_0": 1.0,
                "w_eff": scale, "R0": 0.01 * scale, "gap": 3.0 * scale})
    t = R3.trajectory_by_arm_table(pd.DataFrame(rows, columns=R3.TRAJ_COLS), ["relu_bb_s1", "t1_bb_s1"], (50,))
    assert len(t) == 2 and list(t["local"]) == [1800, 1825] and (t["n_runs"] == 3).all()            # the failed run is not in
    a = t.iloc[0]
    assert a["smoothing_mean"] == pytest.approx(7.0 / 3.0) and a["smoothing_min"] == 1.0 and a["smoothing_max"] == 4.0
    assert a["smoothing_sd"] == pytest.approx(np.std([1.0, 2.0, 4.0], ddof=1)) and a["gap_mean"] == pytest.approx(7.0)
    assert (a["actor"], a["starts"], a["s"]) == ("relu", "bb", 1) and a["e2_at_0_max"] == pytest.approx(240.0 + 18.0)
    assert R3.trajectory_by_arm_table(pd.DataFrame(columns=R3.TRAJ_COLS), ["relu_bb_s1"], (50,)).empty


def test_starts_effect_and_the_ms_r2_context_comparisons(world):
    comps = R3.starts_comparisons(R3.ACTORS)
    p, _ = R3.paired_tables(world, comps, R3.SECONDARY_METRICS, QS, SEEDS4, R3.LABEL_STARTS)
    r = p[(p["arm"] == "t10_st_s16") & (p["q"] == 50) & (p["metric"] == R3.PRIMARY)].iloc[0]
    d = [t1_peak("st", 16, 50, i) + DELTA[("t10_st_s16", 50)][i] - (t1_peak("bb", 16, 50, i) + DELTA[("t10_bb_s16", 50)][i])
         for i in range(4)]
    assert r["baseline"] == "t10_bb_s16" and r["mean"] == pytest.approx(np.mean(d), abs=1e-12) and r["n_pairs"] == 4
    assert (r["ci_mean_lo"], r["ci_mean_hi"]) == indep_ci(d)


# ====================================================================== 3. transmission, interaction, quadrature
def tx_frame(rows: Dict[Tuple[str, int], Tuple[float, float]], q: int = 50) -> pd.DataFrame:
    """Per-run frame with the given (smoothing, remainder) per (arm, seed); gap = smoothing + remainder."""
    return pd.DataFrame([blank(arm, q, sd, smoothing=sm, remainder=rem, gap=sm + rem)
                         for (arm, sd), (sm, rem) in rows.items()], columns=R3.COLUMNS)


def test_transmission_ratio_on_constructed_data():
    seeds = (10501, 10502)
    rows: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for sd in seeds:
        rows[("t1_bb_s1", sd)] = (4.0, 6.0)          # gap 10
        rows[("t1_bb_s16", sd)] = (3.0, 6.0)         # smoothing -1, remainder 0 -> gap -1: ratio 1
        rows[("relu_bb_s1", sd)] = (4.0, 6.0)
        rows[("relu_bb_s16", sd)] = (3.0, 7.0)       # smoothing -1, remainder +1 -> gap 0: ratio 0 (the remainder offsets)
        rows[("t10_bb_s1", sd)] = (4.0, 6.0)
        rows[("t10_bb_s16", sd)] = (4.0, 5.0)        # smoothing 0 -> the denominator is 0: NaN, no exception
    t, long = R3.transmission_tables(tx_frame(rows), R3.noise_landing_comparisons(R3.ACTORS, ("bb",)), (50,), seeds)
    t = t.set_index("arm")
    assert t.loc["t1_bb_s16", "ratio"] == pytest.approx(1.0, abs=1e-12)
    assert t.loc["t1_bb_s16", "ci_lo"] == pytest.approx(1.0, abs=1e-12)
    assert t.loc["t1_bb_s16", "ci_hi"] == pytest.approx(1.0, abs=1e-12)
    assert t.loc["t1_bb_s16", "mean_d_gap"] == pytest.approx(-1.0)
    assert t.loc["t1_bb_s16", "mean_d_smoothing"] == pytest.approx(-1.0)
    assert t.loc["relu_bb_s16", "ratio"] == pytest.approx(0.0, abs=1e-12)
    assert t.loc["relu_bb_s16", "ci_hi"] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(t.loc["t10_bb_s16", "ratio"]) and math.isnan(t.loc["t10_bb_s16", "ci_lo"])
    assert t.loc["t10_bb_s16", "n_boot_valid"] == 0
    assert t.loc["t1_bb_s16", "per_seed_ratios"] == "10501:1;10502:1"
    assert "10501:nan" in t.loc["t10_bb_s16", "per_seed_ratios"]
    assert (t["baseline"] == ["t1_bb_s1", "relu_bb_s1", "t10_bb_s1"]).all() and list(t["actor"]) == ["t1", "relu", "t10"]
    assert set(long["arm"]) == {"t1_bb_s16", "relu_bb_s16", "t10_bb_s16"} and len(long) == 6
    assert set(t["status"]) == {"descriptive"} and "s16 - s1" in t["note"].iloc[0]


def test_transmission_interval_resamples_the_seeds_and_recomputes_the_ratio():
    """d gap [-1, -2], d smoothing [-1, -1]: ratio of means 1.5; the resampled means give ratios 1, 1.5, 2 with probabilities
    1/4, 1/2, 1/4, so the 2.5 / 97.5 percentiles are 1 and 2 (computed by hand)."""
    r = R3.transmission_ratio([-1.0, -2.0], [-1.0, -1.0])
    assert r["ratio"] == pytest.approx(1.5) and r["ci_lo"] == pytest.approx(1.0) and r["ci_hi"] == pytest.approx(2.0)
    assert r["n_boot_valid"] == 10000
    g, s = np.array([-1.0, -2.0]), np.array([-1.0, -1.0])
    idx = np.random.default_rng(BOOT).integers(0, 2, size=(10000, 2))
    rat = g[idx].mean(axis=1) / s[idx].mean(axis=1)
    assert (r["ci_lo"], r["ci_hi"]) == (float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5)))
    # a denominator that straddles 0: the resamples with a ~0 mean denominator are dropped, never a division error
    r = R3.transmission_ratio([-1.0, -1.0, 2.0, 0.0], [1.0, -1.0, 1.0, -1.0])
    assert r["n_boot_valid"] < 10000 and np.isfinite(r["ci_lo"])
    assert math.isnan(R3.transmission_ratio([1.0, 1.0], [1.0, -1.0])["ratio"])        # mean denominator 0
    assert math.isnan(R3.transmission_ratio([], [])["ratio"])


def test_transmission_uses_a_fresh_generator_per_call_and_per_table_row():
    g = np.array([-1.1, -2.3, -0.4, -3.7, -1.9, -0.8])
    sm = np.array([-1.0, -1.2, -0.7, -1.5, -0.9, -1.1])
    a, b = R3.transmission_ratio(g, sm), R3.transmission_ratio(g, sm)
    assert (a["ci_lo"], a["ci_hi"]) == (b["ci_lo"], b["ci_hi"])
    idx = np.random.default_rng(BOOT).integers(0, 6, size=(10000, 6))
    rat = g[idx].mean(axis=1) / sm[idx].mean(axis=1)
    assert (a["ci_lo"], a["ci_hi"]) == (float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5)))
    assert a["ratio"] == pytest.approx(g.mean() / sm.mean()) and a["ci_lo"] < a["ratio"] < a["ci_hi"]
    seeds = tuple(range(10501, 10507))
    rows: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for i, sd in enumerate(seeds):
        for actor in ("t1", "relu"):
            rows[("%s_bb_s1" % actor, sd)] = (4.0, 6.0)
            rows[("%s_bb_s16" % actor, sd)] = (4.0 + sm[i], 6.0 + g[i] - sm[i])
    df = tx_frame(rows)
    cmp_ = R3.noise_landing_comparisons(("t1", "relu"), ("bb",))
    t1 = R3.transmission_tables(df, cmp_, (50,), seeds)[0].set_index("arm")
    t2 = R3.transmission_tables(df, list(reversed(cmp_)), (50,), seeds)[0].set_index("arm")
    for arm in t1.index:
        assert (t1.loc[arm, "ci_lo"], t1.loc[arm, "ci_hi"], t1.loc[arm, "ratio"]) == (
            t2.loc[arm, "ci_lo"], t2.loc[arm, "ci_hi"], t2.loc[arm, "ratio"])
    assert t1.loc["t1_bb_s16", "ci_lo"] == pytest.approx(a["ci_lo"], abs=1e-12)


def test_interaction_on_constructed_data():
    """(v_s16 - v_s1) - (t1_s16 - t1_s1) per (v, starts, q, seed) on |peak|, the gap, the remainder and the smoothing part."""
    seeds = (10501, 10502, 10503, 10504)
    adj = {10501: 0.0, 10502: 0.01, 10503: 0.02, 10504: 0.03}
    rows = []
    for sd in seeds:
        for arm, peak, sm, rem in (("t1_bb_s1", 0.10, 4.0, 6.0), ("t1_bb_s16", 0.07, 3.0, 6.5), ("relu_bb_s1", 0.08, 4.0, 5.0),
                                   ("relu_bb_s16", 0.04 + adj[sd], 2.5, 5.0 + adj[sd] * 10)):
            rows.append(settle(blank(arm, 50, sd, stage2_peak_rel_err_abs=peak, smoothing=sm, remainder=rem, gap=sm + rem)))
    df = pd.DataFrame(rows, columns=R3.COLUMNS)
    # seed 10504 of t1_bb_s1 is missing: that seed leaves the interaction (3 pairs)
    i = df.index[(df["arm"] == "t1_bb_s1") & (df["seed"] == 10504)][0]
    df.loc[i, ["status", "complete"]] = ["missing", False]
    summ, long = R3.interaction_tables(df, ("t1", "relu"), ("bb",), (50,), seeds)
    assert set(summ["actor"]) == {"relu"} and set(summ["metric"]) == set(R3.INTERACTION_METRICS) and len(summ) == 4
    pk = summ[summ["metric"] == R3.PRIMARY].iloc[0]
    # per seed: relu change = (0.04 + adj) - 0.08 = -0.04 + adj; t1 change = 0.07 - 0.10 = -0.03; interaction = -0.01 + adj
    d = [-0.01, 0.0, 0.01]
    assert pk["n_pairs"] == 3 and pk["mean"] == pytest.approx(0.0, abs=1e-12) and pk["median"] == pytest.approx(0.0, abs=1e-12)
    lo, hi = indep_ci(d)
    assert pk["ci_mean_lo"] == pytest.approx(lo, abs=1e-12) and pk["ci_mean_hi"] == pytest.approx(hi, abs=1e-12)
    assert pk["n_pos"] + pk["n_neg"] + pk["n_zero"] == 3 and pk["n_neg"] >= 1 and pk["n_pos"] >= 1
    rem = summ[summ["metric"] == "remainder"].iloc[0]
    # relu change = 10 * adj; t1 change = +0.5 -> interaction = 10 * adj - 0.5 over seeds 10501-3 = [-0.5, -0.4, -0.3]
    assert rem["mean"] == pytest.approx(-0.4, abs=1e-12) and rem["n_pairs"] == 3
    sm = summ[summ["metric"] == "smoothing"].iloc[0]
    assert sm["mean"] == pytest.approx((-1.5) - (-1.0), abs=1e-12)                 # relu -1.5, t1 -1.0 -> -0.5
    gp = summ[summ["metric"] == "gap"].iloc[0]
    assert gp["mean"] == pytest.approx(sm["mean"] + rem["mean"], abs=1e-12)        # gap = smoothing + remainder
    r = long[(long["metric"] == R3.PRIMARY) & (long["seed"] == 10503)].iloc[0]
    assert (r["v_s16"], r["v_s1"], r["t1_s16"], r["t1_s1"]) == pytest.approx((0.06, 0.08, 0.07, 0.10), abs=1e-12)
    assert r["interaction"] == pytest.approx(0.01, abs=1e-12) and r["v_change"] == pytest.approx(-0.02)
    assert len(long[long["metric"] == R3.PRIMARY]) == 3 and set(summ["status"]) == {"descriptive"}


def test_quadrature_check_by_hand_and_the_negative_F2_flag():
    # F^2 = 3^2 - 2^2 = 5; quadrature sqrt(5 + 1^2) = 2.449...; additive 3 - (2 - 1) = 2; observed 2.6 -> quadrature closer
    r = R3.quadrature_check(3.0, 2.0, 2.6, 1.0)
    assert r["F2"] == pytest.approx(5.0) and not r["F2_negative"] and r["F"] == pytest.approx(math.sqrt(5.0))
    assert r["quadrature_pred"] == pytest.approx(math.sqrt(6.0)) and r["additive_pred"] == pytest.approx(2.0)
    assert r["abs_err_quadrature"] == pytest.approx(abs(math.sqrt(6.0) - 2.6)) and r["abs_err_additive"] == pytest.approx(0.6)
    assert r["closer"] == "quadrature"
    # F^2 < 0: gap(s1) 1.5 < smoothing(s1) 2.0 -> F = 0 and flagged; the quadrature prediction is smoothing(s16)
    r = R3.quadrature_check(1.5, 2.0, 0.9, 0.5)
    assert r["F2"] == pytest.approx(1.5 ** 2 - 4.0) and r["F2_negative"] and r["F"] == 0.0
    assert r["quadrature_pred"] == pytest.approx(0.5) and r["additive_pred"] == pytest.approx(1.5 - 1.5 + 0.0)
    assert r["abs_err_quadrature"] == pytest.approx(0.4) and r["abs_err_additive"] == pytest.approx(0.9)
    assert r["closer"] == "quadrature"
    # additive closer; an exact tie; undefined input
    assert R3.quadrature_check(3.0, 2.0, 2.0, 1.0)["closer"] == "additive"
    assert R3.quadrature_check(2.0, 2.0, 1.0, 1.0)["closer"] == "tie"          # F = 0: both predictions are the same
    r = R3.quadrature_check(float("nan"), 1.0, 1.0, 1.0)
    assert r["closer"] == "" and math.isnan(r["quadrature_pred"])


def test_quadrature_table_uses_arm_means(world):
    t = R3.quadrature_table(world, ("t1", "relu"), ("bb",), QS).set_index(["actor", "starts", "q"])
    g = world[(world["arm"] == "relu_bb_s1") & (world["q"] == 50)]
    g16 = world[(world["arm"] == "relu_bb_s16") & (world["q"] == 50)]
    r = t.loc[("relu", "bb", 50)]
    assert r["gap_s1"] == pytest.approx(g["gap"].mean()) and r["smoothing_s16"] == pytest.approx(g16["smoothing"].mean())
    assert r["quadrature_pred"] == pytest.approx(math.sqrt(max(g["gap"].mean() ** 2 - g["smoothing"].mean() ** 2, 0.0)
                                                           + g16["smoothing"].mean() ** 2))
    assert r["additive_pred"] == pytest.approx(g["gap"].mean() - (g["smoothing"].mean() - g16["smoothing"].mean()))
    assert r["n_s1"] == 4 and r["n_s16"] == 4 and set(t["status"]) == {"descriptive"}
    # the missing run (relu_st_s16, q=60, seed 10504) is not in the arm mean
    assert R3.quadrature_table(world, ("relu",), ("st",), (60,)).iloc[0]["n_s16"] == 3


R2_PER_RUN = ROOT / "results" / "ms_r2" / "analysis" / "per_run.csv"


@pytest.mark.skipif(not R2_PER_RUN.exists(), reason="results/ms_r2/analysis/per_run.csv is not in this checkout")
def test_the_quadrature_check_reproduces_the_numbers_of_the_prompt_preamble_from_the_ms_r2_per_run_table():
    """Preamble item 3 of the MS-R3 prompt: additive 2.57 / 1.36 / 1.66 / 1.71, quadrature 3.51 / 1.95 / 2.44 / 2.36,
    observed 3.90 / 2.75 / 2.81 / 2.42 (bin-balanced q50, q60; stratified q50, q60), from the arm means of MS-R2."""
    per = pd.read_csv(R2_PER_RUN)
    per = per[per["status"] == "done"]
    got = []
    for k in ("bb", "st"):
        for q in (50, 60):
            m = lambda s, c: per[(per["arm"] == "NL_%s_s%d" % (k, s)) & (per["q"] == q)][c].mean()  # noqa: E731
            r = R3.quadrature_check(m(1, "gap"), m(1, "smoothing"), m(16, "gap"), m(16, "smoothing"))
            got.append((r["additive_pred"], r["quadrature_pred"], m(16, "gap")))
            assert r["closer"] == "quadrature"
    assert [round(x[0], 2) for x in got] == [2.57, 1.36, 1.66, 1.71]
    assert [round(x[1], 2) for x in got] == [3.51, 1.95, 2.44, 2.36]
    assert [round(x[2], 2) for x in got] == [3.90, 2.75, 2.81, 2.42]


def test_predictions_table_on_the_world(world):
    """The 4-seed world with the values worked out by hand from its construction (``t1_peak``, ``DELTA``, the smoothing
    and remainder formulas of ``make_world``)."""
    trans, _ = R3.transmission_tables(world, R3.noise_landing_comparisons(R3.ACTORS), QS, SEEDS4)
    t = R3.predictions_table(world, R3.ACTORS, R3.STARTS, QS, SEEDS4, trans).set_index(["actor", "starts", "q"])
    assert len(t) == 12 and set(t["status"]) == {"descriptive"} and not any("verdict" in c for c in t.columns)
    r = t.loc[("relu", "bb", 50)]
    # P1: |peak| of relu_bb_s1 = t1_bb_s1 + DELTA = 0.115 - 0.035 against 0.115 (t1_peak = 0.10 + 0.01 i: mean 0.115)
    assert r["p1_n_pairs"] == 4 and r["p1_n_lower"] == 4 and r["p1_n_higher"] == 0
    assert r["p1_mean_change"] == pytest.approx(-0.035, abs=1e-12)
    assert r["p1_abs_peak_s1"] == pytest.approx(0.08, abs=1e-12) and r["p1_abs_peak_t1_s1"] == pytest.approx(0.115, abs=1e-12)
    assert t.loc[("t10", "bb", 50), "p1_n_lower"] == 0 and t.loc[("t10", "bb", 50), "p1_n_higher"] == 4
    assert t.loc[("t10", "bb", 50), "p1_mean_change"] == pytest.approx(0.0175, abs=1e-12)
    assert math.isnan(t.loc[("t1", "bb", 50), "p1_n_pairs"]) and math.isnan(t.loc[("t1", "bb", 50), "p4_n_pairs_s1"])
    # P2 at s = 1: remainder 1.5 + 0.1 i - 0.5 = 1.0 ... 1.3 (mean 1.15), smoothing 2.0 + 0.01 i (mean 2.015)
    assert r["p2_n"] == 4 and r["p2_mean_remainder"] == pytest.approx(1.15) and r["p2_mean_smoothing"] == pytest.approx(2.015)
    assert r["p2_remainder_over_smoothing"] == pytest.approx(1.15 / 2.015, abs=1e-12)
    assert r["p2_median_run_ratio"] == pytest.approx(np.median([1.0 / 2.0, 1.1 / 2.01, 1.2 / 2.02, 1.3 / 2.03]), abs=1e-12)
    assert r["p2_share_abs_remainder_lt_smoothing"] == 1.0                    # |remainder| < smoothing part in every run
    # P3 at s = 16: |peak| = 0.105 + 0.01 i + DELTA (mean 0.0725); sigma_2_0 is NaN in the world: the ratio is NaN, not an error
    assert r["p3_n"] == 4 and r["p3_abs_peak_s16"] == pytest.approx(0.0725, abs=1e-12) and math.isnan(r["p3_abs_peak_over_floor"])
    # the transmission ratio by hand: the smoothing part falls 2.0 -> 0.5 (-1.5), the remainder rises by 0.3 log2(16) = 1.2,
    # so the gap falls by 0.3: ratio 0.2 in every seed (relu_st: remainder + 0.4, gap - 1.1, ratio 0.7333)
    assert r["p3_transmission_ratio"] == pytest.approx(0.2, abs=1e-12)
    assert r["p3_transmission_ci_lo"] == pytest.approx(0.2, abs=1e-12) and r["p3_transmission_ci_hi"] == pytest.approx(0.2, abs=1e-12)
    assert t.loc[("t10", "st", 60), "p3_transmission_ratio"] == pytest.approx(1.1 / 1.5, abs=1e-12)
    # P4: RMSE_pos is the same constant 0.02 for every arm in this world: EQUAL values are not "lower" (strict), change 0
    assert r["p4_n_pairs_s1"] == 4 and r["p4_n_lower_s1"] == 0 and r["p4_mean_change_s1"] == 0.0 and r["p4_mean_change_s16"] == 0.0
    assert r["p4_rmse_s1"] == pytest.approx(0.02) and r["p4_rmse_t1_s16"] == pytest.approx(0.02)


def five_run_frame() -> pd.DataFrame:
    """relu / t1, bin-balanced, q = 50, five seeds, every number chosen so that the predictions can be worked out on paper
    with unequal counts everywhere (a swapped sign or inequality changes a count)."""
    seeds = SEEDS4 + (10505,)
    cols = {
        "t1_bb_s1": dict(peak=[0.10, 0.12, 0.08, 0.11, 0.09], sm=[1.0] * 5, rem=[0.5] * 5, rmse=[0.02] * 5),
        "relu_bb_s1": dict(peak=[0.05, 0.13, 0.04, 0.06, 0.07], sm=[1.0, 1.0, 0.5, 0.5, 2.0], rem=[0.2, -1.5, 0.1, 0.6, 0.4],
                           rmse=[0.018, 0.022, 0.015, 0.02, 0.016]),
        "t1_bb_s16": dict(peak=[0.2] * 5, sm=[0.25] * 5, rem=[0.6] * 5, rmse=[0.019] * 5),
        "relu_bb_s16": dict(peak=[0.04, 0.06, 0.05, 0.07, 0.03], sm=[0.3, 0.2, 0.25, 0.3, 0.2],
                            rem=[0.3, -0.9, 0.2, 0.5, 0.45], rmse=[0.016, 0.02, 0.014, 0.018, 0.017])}
    rows = []
    for arm, c in cols.items():
        for i, sd in enumerate(seeds):
            g2 = 70.0
            kw: Dict[str, Any] = dict(stage2_peak_rel_err_abs=c["peak"][i], smoothing=c["sm"][i], remainder=c["rem"][i],
                                      gap=c["sm"][i] + c["rem"][i], stage2_rmse_pos_over_g2_0=c["rmse"][i],
                                      smoothed_e_pred_0=g2 - c["sm"][i], e2_at_0=g2 - c["sm"][i] - c["rem"][i])
            if arm == "relu_bb_s16":
                kw.update(sigma_2_0=[0.6, 0.7, 0.65, 0.75, 0.5][i], smoothing_rel=[0.006, 0.008, 0.007, 0.009, 0.005][i])
            rows.append(settle(blank(arm, 50, sd, **kw)))
    return pd.DataFrame(rows, columns=R3.COLUMNS)


def test_predictions_p1_to_p4_by_hand_on_five_runs():
    df = five_run_frame()
    seeds = SEEDS4 + (10505,)
    trans, _ = R3.transmission_tables(df, R3.noise_landing_comparisons(("t1", "relu"), ("bb",)), (50,), seeds)
    t = R3.predictions_table(df, ("t1", "relu"), ("bb",), (50,), seeds, trans).set_index("actor")
    r = t.loc["relu"]
    # P1: relu - t1 at s = 1: [-.05, +.01, -.04, -.05, -.02]: four lower, one higher, mean -0.03; arm means 0.07 and 0.10
    assert (r["p1_n_pairs"], r["p1_n_lower"], r["p1_n_higher"]) == (5, 4, 1)
    assert r["p1_mean_change"] == pytest.approx(-0.03, abs=1e-12)
    assert r["p1_abs_peak_s1"] == pytest.approx(0.07, abs=1e-12) and r["p1_abs_peak_t1_s1"] == pytest.approx(0.10, abs=1e-12)
    # P2 (s = 1): remainder [.2, -1.5, .1, .6, .4] (mean -0.04), smoothing [1, 1, .5, .5, 2] (mean 1.0): ratio of means -0.04;
    # per-run ratios [.2, -1.5, .2, 1.2, .2] (median 0.2); |remainder| < smoothing in runs 1, 3, 5 only (-1.5: no, 0.6 > 0.5: no)
    assert r["p2_n"] == 5 and r["p2_mean_remainder"] == pytest.approx(-0.04, abs=1e-12)
    assert r["p2_mean_smoothing"] == pytest.approx(1.0, abs=1e-12)
    assert r["p2_remainder_over_smoothing"] == pytest.approx(-0.04, abs=1e-12)
    assert r["p2_median_run_ratio"] == pytest.approx(0.2, abs=1e-12)
    assert r["p2_share_abs_remainder_lt_smoothing"] == pytest.approx(0.6, abs=1e-12)        # a signed comparison would give 0.8
    assert r["p2_mean_e_sigma_0"] == pytest.approx(69.0, abs=1e-12) and r["p2_mean_e_hat_2_0"] == pytest.approx(69.04, abs=1e-12)
    # P3 (s = 16): |peak| mean 0.05, sigma mean 0.64 -> floor sigma / (sqrt(pi) q), smoothing_rel mean 0.007
    floor = 0.64 / (math.sqrt(math.pi) * 50.0)
    assert r["p3_n"] == 5 and r["p3_abs_peak_s16"] == pytest.approx(0.05, abs=1e-12)
    assert r["p3_floor_rel_sigma"] == pytest.approx(floor, abs=1e-15) and r["p3_floor_rel_smoothing"] == pytest.approx(0.007, abs=1e-15)
    assert r["p3_abs_peak_over_floor"] == pytest.approx(0.05 / floor, abs=1e-9)
    # transmission by hand: d smoothing = mean(.3,.2,.25,.3,.2) - 1.0 = -0.75; gap 0.36 - 0.96 = -0.6: ratio 0.8
    assert r["p3_transmission_ratio"] == pytest.approx(0.8, abs=1e-12)
    dg = np.array([0.6, -0.7, 0.45, 0.8, 0.65]) - np.array([1.2, -0.5, 0.6, 1.1, 2.4])
    ds = np.array([0.3, 0.2, 0.25, 0.3, 0.2]) - np.array([1.0, 1.0, 0.5, 0.5, 2.0])
    idx = np.random.default_rng(BOOT).integers(0, 5, size=(10000, 5))
    ratios = dg[idx].mean(axis=1) / ds[idx].mean(axis=1)
    assert r["p3_transmission_ci_lo"] == float(np.percentile(ratios, 2.5))
    assert r["p3_transmission_ci_hi"] == float(np.percentile(ratios, 97.5))
    assert r["p3_transmission_ci_lo"] < 0.8 < r["p3_transmission_ci_hi"]
    # P4: RMSE_pos; s = 1: relu - t1 = [-.002, +.002, -.005, 0, -.004]: three lower (an exact tie is not lower), mean -0.0018
    assert (r["p4_n_pairs_s1"], r["p4_n_lower_s1"]) == (5, 3) and r["p4_mean_change_s1"] == pytest.approx(-0.0018, abs=1e-12)
    assert r["p4_rmse_s1"] == pytest.approx(0.0182, abs=1e-12) and r["p4_rmse_t1_s1"] == pytest.approx(0.02, abs=1e-12)
    # s = 16: [-.003, +.001, -.005, -.001, -.002]: four lower, mean -0.002
    assert (r["p4_n_pairs_s16"], r["p4_n_lower_s16"]) == (5, 4) and r["p4_mean_change_s16"] == pytest.approx(-0.002, abs=1e-12)
    assert r["p4_rmse_s16"] == pytest.approx(0.017, abs=1e-12) and r["p4_rmse_t1_s16"] == pytest.approx(0.019, abs=1e-12)
    # the control row has the P2 / P3 / P4 evidence of its own arms and no paired columns
    c = t.loc["t1"]
    assert c["p2_remainder_over_smoothing"] == pytest.approx(0.5) and c["p2_share_abs_remainder_lt_smoothing"] == 1.0
    assert math.isnan(c["p1_n_pairs"]) and math.isnan(c["p4_n_lower_s1"]) and c["p4_rmse_s1"] == pytest.approx(0.02)


def test_predictions_p2_leaves_out_runs_without_a_positive_smoothing_part():
    rows = [settle(blank("relu_bb_s1", 50, sd, stage2_peak_rel_err_abs=0.05, smoothing=sm, remainder=rem, gap=sm + rem))
            for sd, sm, rem in ((10501, 1.0, 0.5), (10502, 0.0, 0.3), (10503, 2.0, 1.0))]
    df = pd.DataFrame(rows, columns=R3.COLUMNS)
    r = R3.predictions_table(df, ("relu",), ("bb",), (50,), SEEDS4[:3], pd.DataFrame()).iloc[0]
    assert r["p2_n"] == 2 and r["p2_mean_remainder"] == pytest.approx(0.75) and r["p2_mean_smoothing"] == pytest.approx(1.5)
    assert r["p2_remainder_over_smoothing"] == pytest.approx(0.5) and r["p2_median_run_ratio"] == pytest.approx(0.5)
    assert r["p2_share_abs_remainder_lt_smoothing"] == 1.0


def test_predictions_p3_floor_from_the_noise_columns():
    rows = []
    for sd, sig, sm_rel, peak in ((10501, 0.65, 0.0070, 0.05), (10502, 0.70, 0.0075, 0.06)):
        rows.append(settle(blank("relu_bb_s16", 50, sd, stage2_peak_rel_err_abs=peak, sigma_2_0=sig, smoothing_rel=sm_rel)))
    df = pd.DataFrame(rows, columns=R3.COLUMNS)
    no_trans = pd.DataFrame(columns=["arm", "q", "ratio", "ci_lo", "ci_hi"])
    t = R3.predictions_table(df, ("relu",), ("bb",), (50,), SEEDS4[:2], no_trans)
    r = t.iloc[0]
    floor = np.mean([0.65, 0.70]) / (math.sqrt(math.pi) * 50)
    assert r["p3_floor_rel_sigma"] == pytest.approx(floor) and r["p3_floor_rel_smoothing"] == pytest.approx(0.00725)
    assert r["p3_abs_peak_over_floor"] == pytest.approx(0.055 / floor) and math.isnan(r["p3_transmission_ratio"])


# ====================================================================== 4. tie metrics and weight statistics
def tent(d: np.ndarray, centre: float = 0.0, top: float = 70.0, slope: float = 0.7, half: float = 100.0) -> np.ndarray:
    """A tent of height ``top`` at ``centre`` falling with ``slope`` to 0 at distance ``half`` from 0."""
    return np.clip(top - slope * np.abs(d - centre), 0.0, None) * (np.abs(d) < half)


def test_tie_metrics_on_a_shifted_tent():
    d = np.arange(-200.0, 200.5, 0.5)
    e = tent(d, centre=3.0)
    m = R3.tie_metrics(d, e, 70.0, 70.0 - float(e[np.argmin(np.abs(d))]), 50.0)
    assert m["peak_locfree_rel_err"] == pytest.approx(0.0, abs=1e-12) and m["peak_locfree_argmax_d"] == 3.0
    assert m["sym_err_max"] == pytest.approx(4.2, abs=1e-9) and m["sym_err_max_rel"] == pytest.approx(4.2 / 70.0, abs=1e-12)
    assert m["sym_err_argmax_abs_d"] >= 3.0
    assert m["tent_slope"] == pytest.approx(0.7) and m["w_eff"] == pytest.approx(2.1 / 0.7)         # gap 2.1 -> 3 units of d
    # a centred tent with a rounded tip: the kink is at 0, no asymmetry, the location-free peak equals the tie value
    e0 = 70.0 - 0.7 * np.sqrt(np.abs(d) ** 2 + 25.0) + 3.5
    m = R3.tie_metrics(d, np.clip(e0, 0, None), 70.0, 70.0 - e0[np.argmin(np.abs(d))], 50.0)
    assert m["peak_locfree_argmax_d"] == 0.0 and m["sym_err_max"] == pytest.approx(0.0, abs=1e-12)
    assert m["peak_locfree_rel_err"] == pytest.approx((e0[np.argmin(np.abs(d))] - 70.0) / 70.0)


def test_tie_metrics_symmetry_is_restricted_to_the_support_and_needs_a_symmetric_grid():
    d = np.arange(-200.0, 200.5, 0.5)
    e = np.zeros_like(d)
    e[np.argmin(np.abs(d - 150.0))] = 5.0                       # an asymmetry in the tail (|d| >= 2q): not counted
    e[np.argmin(np.abs(d - 100.0))] = 4.0                       # at |d| = 2q exactly: not counted either (the window is strict)
    e[np.argmin(np.abs(d - 10.0))] = 1.0                        # asymmetries inside |d| < 2q: the larger one lies between q and 2q
    e[np.argmin(np.abs(d - 75.0))] = 2.0
    m = R3.tie_metrics(d, e, 70.0, 1.0, 50.0)
    assert m["sym_err_max"] == 2.0 and m["sym_err_argmax_abs_d"] == 75.0
    assert m["peak_locfree_argmax_d"] == 150.0                  # the location-free peak is the argmax over the whole grid
    m = R3.tie_metrics(d[1:], e[1:], 70.0, 1.0, 50.0)           # an asymmetric grid: the symmetry error is undefined
    assert math.isnan(m["sym_err_max"]) and np.isfinite(m["peak_locfree_rel_err"])
    m = R3.tie_metrics(d, e, float("nan"), 1.0, 50.0)
    assert all(math.isnan(v) for v in m.values())
    assert math.isnan(R3.tie_metrics(d, e, 70.0, float("nan"), 50.0)["w_eff"])


def test_first_layer_stats_units_and_quantiles():
    w = np.array([0.5, -2.0, 1.5, 0.12])
    s = R3.first_layer_stats(w, "t1", 200.0)
    assert s["d_scale"] == 1.0 and s["w_abs_max"] == 2.0 and s["n_w_abs_gt1"] == 2 and s["bend_d_min"] == pytest.approx(100.0)
    assert s["w_abs_q50"] == pytest.approx(np.median(np.abs(w)))
    assert s["w_abs_q25"] == pytest.approx(np.percentile(np.abs(w), 25))
    assert R3.first_layer_stats(w, "relu", 200.0) == s                         # a relu export is read like t1 (stored weights)
    t = R3.first_layer_stats(w, "t10", 200.0)                                  # t10: ten times the stored weight
    assert t["d_scale"] == 10.0 and t["w_abs_max"] == 20.0 and t["n_w_abs_gt1"] == 4 and t["bend_d_min"] == pytest.approx(10.0)
    assert t["w_abs_q90"] == pytest.approx(10.0 * s["w_abs_q90"])
    with pytest.raises(ValueError):
        R3.first_layer_stats(w, "gelu", 200.0)                                 # never the tanh reading of another variant


def test_unit_rows_give_the_distribution_and_the_centre_of_every_hidden_unit():
    w = np.array([[0.5, 0.2], [-0.3, -0.4], [0.1, 0.0]])
    b = np.array([0.05, 0.1, 0.2])
    u = R3.unit_rows(w, b, "t1", 200.0)
    assert u["w_d"].tolist() == pytest.approx([0.2, -0.4, 0.0]) and u["w_d_abs"].tolist() == pytest.approx([0.2, 0.4, 0.0])
    assert u["w_stage"].tolist() == [0.5, -0.3, 0.1] and u["bias"].tolist() == [0.05, 0.1, 0.2]
    # centre d: the pre-activation w_stage * tau + w_d * d / B + bias vanishes there (tau = 1); NaN where w_d = 0
    assert u["center_d"][0] == pytest.approx(-200.0 * (0.5 + 0.05) / 0.2) and u["center_d"][1] == pytest.approx(-100.0)
    assert math.isnan(u["center_d"][2])
    assert R3.unit_rows(w, b, "relu", 200.0)["center_d"][1] == pytest.approx(-100.0)
    t = R3.unit_rows(w, b, "t10", 200.0)                                  # t10: ten times the stored d weight, ten times closer
    assert t["w_d"].tolist() == pytest.approx([2.0, -4.0, 0.0]) and t["center_d"][0] == pytest.approx(-55.0)
    assert t["center_d"][1] == pytest.approx(-10.0) and t["w_stage"].tolist() == [0.5, -0.3, 0.1]
    assert R3.unit_rows(w, b, "t1", 200.0, tau=0.0)["center_d"][0] == pytest.approx(-200.0 * 0.05 / 0.2)
    with pytest.raises(ValueError):
        R3.unit_rows(w, b, "gelu", 200.0)


def write_export(path: Path, w_d: np.ndarray, variant: Optional[str]) -> None:
    """A minimal weight export: ``actor.l1.weight`` of shape (hidden, 2) (column 0 the stage feature, column 1 the d input)."""
    w = np.zeros((len(w_d), 2), dtype=np.float32)
    w[:, 0] = 7.0
    w[:, 1] = w_d
    arrs: Dict[str, Any] = {"actor.l1.weight": w, "actor.l1.bias": np.zeros(len(w_d), dtype=np.float32)}
    if variant is not None:
        arrs["actor_variant"] = np.asarray(variant)
    np.savez(path, **arrs)


def test_a_t10_export_reports_ten_times_the_stored_weight_and_an_absent_variant_is_t1(tmp_path):
    w = np.array([0.4, -1.2, 0.8, 0.05], dtype=np.float32)
    write_export(tmp_path / "t1.npz", w, None)
    write_export(tmp_path / "relu.npz", w, "relu")
    write_export(tmp_path / "t10.npz", w, "t10")
    a, b, c = (R3.read_export_stats(tmp_path / n, 200.0) for n in ("t1.npz", "relu.npz", "t10.npz"))
    assert (a["variant"], b["variant"], c["variant"]) == ("t1", "relu", "t10")
    assert a["w_abs_max"] == pytest.approx(1.2) and b["w_abs_max"] == pytest.approx(1.2)    # stage-feature column not read
    assert c["w_abs_max"] == pytest.approx(12.0) and c["d_scale"] == 10.0 and c["bend_d_min"] == pytest.approx(200.0 / 12.0)
    write_export(tmp_path / "bad.npz", w, "gelu")
    with pytest.raises(ValueError):
        R3.read_export_stats(tmp_path / "bad.npz", 200.0)
    assert R3.T10_D_SCALE == 10.0


def test_the_scale_constant_is_the_one_of_the_actor():
    from agents.ppo_curriculum import ACTOR_VARIANTS, D_FEATURE_SCALE_T10
    assert R3.T10_D_SCALE == D_FEATURE_SCALE_T10 and tuple(R3.ACTORS) == tuple(ACTOR_VARIANTS)


def test_terminal_export_selection_and_the_terminal_entry(tmp_path):
    wd = tmp_path / "weights"
    wd.mkdir()
    for u in (25, 50, 75, 100, 125, 150):
        write_export(wd / ("u%05d.npz" % u), np.array([1.0]), None)
    (wd / "notes.txt").write_text("x")
    ex = R3.terminal_exports(wd, 0, 100)
    assert [(u, loc) for u, loc, _ in ex] == [(25, 25), (50, 50), (75, 75), (100, 100)]
    ex = R3.terminal_exports(wd, 50, 75)                         # a terminal stage that starts at global update 50
    assert [(u, loc) for u, loc, _ in ex] == [(75, 25), (100, 50), (125, 75)]
    assert R3.terminal_exports(tmp_path / "nope", 0, 100) == []
    (tmp_path / "ms_run_summary.json").write_text(json.dumps({"phase_timing": {"stage2": {"global_entry": 50}}}))
    assert R3._terminal_entry(tmp_path) == 50 and R3._terminal_entry(tmp_path / "nope") == 0


def test_first_layer_table_and_the_attachment_to_the_per_run_rows(tmp_path):
    rows = []
    for actor, variant, scale in (("t1", None, 1.0), ("relu", "relu", 1.0), ("t10", "t10", 10.0), ("relu", "t10", 10.0)):
        d = tmp_path / ("q50_%s_%s" % (actor, variant))
        (d / "weights").mkdir(parents=True)
        for u, mx in ((25, 0.5), (50, 1.0), (75, 1.5)):                    # the terminal stage (decay_last = 75)
            write_export(d / "weights" / ("u%05d.npz" % u), np.array([mx, -0.15, 0.2]), variant)
        for u in (100, 125, 150, 175, 200):          # stage 1 follows: larger weights, and local updates up to 125 > 75
            write_export(d / "weights" / ("u%05d.npz" % u), np.array([9.0, -0.15, 0.2]), variant)
        (d / "run_config.json").write_text(json.dumps({"actor_variant": variant} if variant else {}))
        arm = "relu_bb_s1" if (actor, variant) == ("relu", "t10") else R3.arm_name(actor, "bb", 1)
        rows.append(settle(blank(arm, 50, 10501 + len(rows), run_dir=str(d), stage2_peak_rel_err_abs=0.05)))
    df = pd.DataFrame(rows, columns=R3.COLUMNS)
    df.loc[3, "seed"] = 10510                                       # a second relu_bb_s1 row, whose export says t10
    win = R2.Windows(ramp_first=21, ramp_last=40, hold_last=60, decay_last=75, traj_from=50)
    fl = R3.first_layer_table(df, win, PROTO, R3.r3_arms(R3.ACTORS))
    assert len(fl) == 4 * (3 + 5) and list(fl.columns) == R3.FL_COLS and (fl["error"] == "").all()
    # EVERY export is a row: the three terminal-stage ones (stage 2) and the five stage-1 ones (stage 1; local counts the
    # updates of stage 1)
    t2, st1 = fl[fl["stage"] == 2], fl[fl["stage"] == 1]
    assert len(t2) == 12 and sorted(t2["local"].unique()) == [25, 50, 75] and (t2["update"] == t2["local"]).all()
    assert len(st1) == 20 and sorted(st1["local"].unique()) == [25, 50, 75, 100, 125] and (st1["update"] == st1["local"] + 75).all()
    assert (st1["w_abs_max"] == 9.0 * st1["d_scale"]).all() and (t2["w_abs_max"] <= 1.5 * t2["d_scale"]).all()
    one = fl[(fl["arm"] == "t10_bb_s1") & (fl["local"] == 75) & (fl["stage"] == 2)].iloc[0]
    assert one["variant"] == "t10" and one["w_abs_max"] == pytest.approx(15.0) and one["B"] == pytest.approx(200.0)
    assert one["bend_d_min"] == pytest.approx(200.0 / 15.0) and one["n_w_abs_gt1"] == 3
    # the summary is the terminal stage only: no stage-1 export, whose weights (9, or 90 for t10) are all larger
    sm = R3.first_layer_summary_table(fl, R3.r3_arms(R3.ACTORS), (50,))
    assert sorted(sm["update"].unique()) == [25, 50, 75] and sm["w_abs_max_max"].max() == pytest.approx(15.0)
    assert len(sm) == 9 and sorted(sm[sm["arm"] == "relu_bb_s1"]["n_runs"].unique()) == [2]
    # the last-export columns of per_run come from the LAST TERMINAL-STAGE export (update 75), not from the last export of the
    # run (stage 1, update 200, whose local update 125 is larger than 75)
    out = R3.attach_last_export(df, fl, win)
    t1 = out[out["arm"] == "t1_bb_s1"].iloc[0]
    assert t1["actor_variant_export"] == "t1" and t1["w1d_abs_max"] == pytest.approx(1.5) and t1["w1d_export_update"] == 75
    assert t1["w1d_bend_d_min"] == pytest.approx(200.0 / 1.5) and t1["w1d_n_gt1"] == 1
    assert t1["flags"] == "" and out[out["arm"] == "t10_bb_s1"].iloc[0]["w1d_abs_max"] == pytest.approx(15.0)
    bad = out[(out["arm"] == "relu_bb_s1")]
    assert (bad["flags"].str.contains("actor variant: arm name says relu, last export says t10")).sum() == 1
    assert bad[bad["actor_variant_export"] == "relu"]["flags"].iloc[0] == ""
    # a last export that is not the one of the end of the terminal stage is flagged (a wrong --decay-last would show here)
    out3 = R3.attach_last_export(df, fl, R2.Windows(ramp_first=21, ramp_last=40, hold_last=60, decay_last=100, traj_from=50))
    assert out3["flags"].str.contains("last terminal-stage weight export is local 75, not 100").sum() == 4
    # an unreadable export is a row with the reason, never an exception; the run keeps NaN statistics and a flag
    (tmp_path / "q50_t1_None" / "weights" / "u00075.npz").write_bytes(b"not an npz")
    fl2 = R3.first_layer_table(df, win, PROTO, R3.r3_arms(R3.ACTORS))
    assert fl2[(fl2["arm"] == "t1_bb_s1") & (fl2["local"] == 75) & (fl2["stage"] == 2)]["error"].iloc[0] != ""
    assert (fl2[(fl2["arm"] == "t1_bb_s1") & (fl2["stage"] == 1)]["error"] == "").all()
    out2 = R3.attach_last_export(df, fl2, win)
    r = out2[out2["arm"] == "t1_bb_s1"].iloc[0]
    assert "weight export" in r["flags"] and math.isnan(r["w1d_abs_max"])


def write_freeze(d: Path, e: np.ndarray, rd: Optional[np.ndarray] = None) -> None:
    """A minimal final-tier freeze file with the three recovery arrays."""
    rd = np.arange(-200.0, 200.5, 0.5) if rd is None else rd
    np.savez(d / "freeze_stage2_final.npz", recovery_d_grid=rd, recovery_e2=e, recovery_g2=np.zeros_like(rd))


def test_add_r3_columns_on_a_constructed_run_dir(tmp_path):
    d = np.arange(-200.0, 200.5, 0.5)
    write_freeze(tmp_path, tent(d, centre=-2.0))
    row = blank("relu_bb_s1", 50, 10501, run_dir=str(tmp_path), gap=2.8, e2_at_0=67.2, t2_R0_final=0.03, t2_R0_dev=0.032,
                stage2_peak_rel_err_abs=0.04)
    R3.add_r3_columns(row, PROTO)
    assert (row["actor"], row["starts"], row["s"]) == ("relu", "bb", 1.0) and row["actor_variant_export"] == ""
    assert row["t2_R0_over_peak_final"] == pytest.approx(0.75) and row["t2_R0_over_peak_dev"] == pytest.approx(0.8)
    assert row["linearised_R0_over_peak"] == pytest.approx(0.588235, abs=1e-6)       # 2k / (2k + a) at q = 50 (the R1 factor 1.7)
    assert row["linearised_R0_over_peak"] == pytest.approx(1.0 / R1.linearised_factor(PROTO["records"]["50"]["game"]))
    assert row["peak_locfree_argmax_d"] == -2.0 and row["peak_locfree_rel_err"] == pytest.approx(0.0, abs=1e-12)
    assert row["w_eff"] == pytest.approx(2.8 / 0.7) and row["sym_err_max"] == pytest.approx(0.7 * 4.0, abs=1e-9)
    # a done run without a freeze file: a flag, never an exception; a q-60 row uses its own linearised value
    row = blank("relu_bb_s1", 60, 10501, run_dir=str(tmp_path / "nope"), gap=2.0)
    R3.add_r3_columns(row, PROTO)
    assert "freeze_stage2_final.npz missing" in row["flags"] and math.isnan(row["w_eff"])
    assert row["linearised_R0_over_peak"] == pytest.approx(1.0 / R1.linearised_factor(PROTO["records"]["60"]["game"]))
    # a row that is not done does not read anything and flags nothing
    row = blank("relu_bb_s1", 50, 10501, run_dir=str(tmp_path / "nope"), status="running", complete=False)
    R3.add_r3_columns(row, PROTO)
    assert row["flags"] == "" and math.isnan(row["peak_locfree_rel_err"])


# ====================================================================== 5. the blind recomputation
def write_outputs(df: pd.DataFrame, d: Path, seeds: Sequence[int] = SEEDS4) -> None:
    """per_run.csv, criterion.csv, transmission.csv and the interaction.csv summary of a frame (the tool's own writers)."""
    d.mkdir(parents=True, exist_ok=True)
    comps = R3.primary_comparisons(R3.ACTORS)
    p, _ = R3.paired_tables(df, comps, R3.SECONDARY_METRICS, QS, seeds, R3.LABEL_T1, R3.CRITERION_NOTE)
    crit = R3.criterion_table(df, p, comps, QS, seeds, R3.LABEL_T1, R3.CRITERION_NOTE)
    trans, _ = R3.transmission_tables(df, R3.noise_landing_comparisons(R3.ACTORS), QS, seeds)
    inter, _ = R3.interaction_tables(df, R3.ACTORS, R3.STARTS, QS, seeds)
    for name, t in (("per_run", df), ("criterion", crit), ("transmission", trans), ("interaction", inter)):
        R1.write_csv(t, d / ("%s.csv" % name), "abc123def456")


@pytest.fixture(scope="module")
def synth_dir(world, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("synth")
    write_outputs(world, d)
    return d


def test_blind_recomputation_agrees_on_the_synthetic_output(synth_dir):
    assert BL.main(["--analysis-dir", str(synth_dir)]) == 0
    txt = (synth_dir / "blind_recomputation.txt").read_text()
    assert "ALL " in txt and "DISAGREE" not in txt and "numpy.random.default_rng(20261008)" in txt
    assert "relu_st_s16 (vs t1_st_s16)" in txt and "q50/10503" in txt and "transmission relu_bb_s16 q=50" in txt
    assert "interaction t10 st q=60" in txt
    assert txt.startswith("blind recomputation of criterion.csv, transmission.csv and the interaction.csv summary")


def _tamper(synth_dir: Path, tmp_path: Path, name: str, fn: Any) -> int:
    d = tmp_path / "t"
    shutil.copytree(synth_dir, d)
    t = pd.read_csv(d / name, float_precision="round_trip")
    fn(t)
    t.to_csv(d / name, index=False)
    return BL.main(["--analysis-dir", str(d)])


@pytest.mark.parametrize("tamper", ["ci_hi", "mean", "flag", "violation", "pending", "pairs", "drop_arm", "extra_arm",
                                    "overall", "baseline", "boot_seed", "n_base_pass", "actor"])
def test_blind_recomputation_fails_when_criterion_is_tampered_with(synth_dir, tmp_path, tamper):
    def fn(crit: pd.DataFrame) -> None:
        i = int(crit.index[crit["arm"] == "relu_bb_s16"][0])
        if tamper == "ci_hi":
            crit.loc[i, "ci_mean_hi_q50"] += 1e-9
        elif tamper == "mean":
            crit.loc[i, "mean_q60"] *= 1.000001
        elif tamper == "flag":
            crit.loc[i, "a_q60"] = True
        elif tamper == "violation":
            crit.loc[i, "b_violations"] = "q50/10501"
        elif tamper == "pending":
            crit.loc[i, "b_pending"] = "q50/10501"
        elif tamper == "pairs":
            crit.loc[i, "n_pairs_q50"] = 3
        elif tamper == "drop_arm":
            crit.drop(index=i, inplace=True)
        elif tamper == "extra_arm":
            crit.loc[len(crit)] = {**crit.iloc[i].to_dict(), "arm": "relu_bb_s64"}
        elif tamper == "overall":
            crit.loc[i, "overall"] = "met"
        elif tamper == "baseline":
            crit.loc[i, "baseline"] = "t1_st_s16"
        elif tamper == "boot_seed":
            crit.loc[i, "boot_seed"] = 20261007
        elif tamper == "n_base_pass":
            crit.loc[i, "n_base_pass"] += 1
        elif tamper == "actor":
            crit.loc[i, "actor"] = "t10"
    assert _tamper(synth_dir, tmp_path, "criterion.csv", fn) == 1


@pytest.mark.parametrize("tamper", ["ratio", "ci_lo", "n_boot_valid", "per_seed", "mean_d_gap", "n_pairs", "median_ratio",
                                    "drop_row", "extra_row", "baseline"])
def test_blind_recomputation_fails_when_the_transmission_table_is_tampered_with(synth_dir, tmp_path, tamper):
    def fn(t: pd.DataFrame) -> None:
        i = int(t.index[(t["arm"] == "t10_st_s16") & (t["q"] == 60)][0])
        if tamper == "ratio":
            t.loc[i, "ratio"] += 1e-9
        elif tamper == "ci_lo":
            t.loc[i, "ci_lo"] -= 1e-9
        elif tamper == "n_boot_valid":
            t.loc[i, "n_boot_valid"] -= 1
        elif tamper == "per_seed":
            t.loc[i, "per_seed_ratios"] = str(t.loc[i, "per_seed_ratios"]).replace("10501:", "10501:1")
        elif tamper == "mean_d_gap":
            t.loc[i, "mean_d_gap"] *= 1.0000001
        elif tamper == "n_pairs":
            t.loc[i, "n_pairs"] += 1
        elif tamper == "median_ratio":
            t.loc[i, "median_per_seed_ratio"] += 1e-9
        elif tamper == "drop_row":
            t.drop(index=i, inplace=True)
        elif tamper == "extra_row":
            t.loc[len(t)] = {**t.iloc[i].to_dict(), "q": 70}
        elif tamper == "baseline":
            t.loc[i, "baseline"] = "t10_bb_s1"
    assert _tamper(synth_dir, tmp_path, "transmission.csv", fn) == 1


@pytest.mark.parametrize("tamper", ["mean", "ci_lo", "n_pairs", "median", "drop_row", "extra_row", "sign_count"])
def test_blind_recomputation_fails_when_the_interaction_table_is_tampered_with(synth_dir, tmp_path, tamper):
    def fn(t: pd.DataFrame) -> None:
        i = int(t.index[(t["actor"] == "relu") & (t["starts"] == "st") & (t["q"] == 60) & (t["metric"] == "remainder")][0])
        if tamper == "mean":
            t.loc[i, "mean"] += 1e-9
        elif tamper == "ci_lo":
            t.loc[i, "ci_mean_lo"] -= 1e-9
        elif tamper == "n_pairs":
            t.loc[i, "n_pairs"] += 1
        elif tamper == "median":
            t.loc[i, "ci_median_hi"] += 1e-9
        elif tamper == "drop_row":
            t.drop(index=i, inplace=True)
        elif tamper == "extra_row":
            t.loc[len(t)] = {**t.iloc[i].to_dict(), "metric": "e2_at_0"}
        elif tamper == "sign_count":
            t.loc[i, "n_pos"] += 1
    assert _tamper(synth_dir, tmp_path, "interaction.csv", fn) == 1


@pytest.mark.parametrize("what", ["peak", "gate_column", "status", "remainder", "smoothing", "gap", "eta"])
def test_blind_recomputation_fails_when_a_per_run_value_is_changed(synth_dir, tmp_path, what):
    def fn(per: pd.DataFrame) -> None:
        j = int(per.index[(per["arm"] == "t10_st_s16") & (per["q"] == 60) & (per["seed"] == 10501)][0])
        if what == "peak":
            per.loc[j, "stage2_peak_rel_err_abs"] += 1e-6
        elif what == "gate_column":
            per.loc[j, "gate_pass"] = False
        elif what == "status":
            per.loc[j, "status"] = "failed"
        elif what == "remainder":
            per.loc[j, "remainder"] += 1e-6
        elif what == "smoothing":
            per.loc[j, "smoothing"] += 1e-6
        elif what == "gap":
            per.loc[j, "gap"] += 1e-6
        elif what == "eta":
            per.loc[j, "eta_T_over_dw"] = 0.0049
            per.loc[j, "eta_dev"] = 0.0062
    assert _tamper(synth_dir, tmp_path, "per_run.csv", fn) == 1
    assert "DISAGREE" in (tmp_path / "t" / "blind_recomputation.txt").read_text()


def test_blind_agrees_on_the_boundary_case_where_the_interval_ends_exactly_at_zero(world, tmp_path):
    """All paired differences 0: ci_mean_hi == 0, (a) is NOT met (strict), and the blind script says the same."""
    df = world.copy()
    m = df["arm"] == "relu_bb_s1"
    base = df[df["arm"] == "t1_bb_s1"].set_index(["q", "seed"])[R3.PRIMARY]
    df.loc[m, R3.PRIMARY] = [base.loc[(q, s)] for q, s in zip(df.loc[m, "q"], df.loc[m, "seed"])]
    write_outputs(df, tmp_path)
    crit = pd.read_csv(tmp_path / "criterion.csv").set_index("arm")
    assert crit.loc["relu_bb_s1", "ci_mean_hi_q50"] == 0.0 and not crit.loc["relu_bb_s1", "a_q50"]
    assert BL.main(["--analysis-dir", str(tmp_path)]) == 0
    crit.loc["relu_bb_s1", "a_q50"] = True                                   # the non-strict reading is a disagreement
    crit.reset_index().to_csv(tmp_path / "criterion.csv", index=False)
    assert BL.main(["--analysis-dir", str(tmp_path)]) == 1


def test_blind_agrees_that_failed_or_running_runs_with_finite_values_never_enter_a_pair(world, tmp_path):
    df = world.copy()
    i = df.index[(df["arm"] == "relu_bb_s1") & (df["q"] == 60) & (df["seed"] == 10501)][0]
    j = df.index[(df["arm"] == "t10_st_s16") & (df["q"] == 50) & (df["seed"] == 10502)][0]
    k = df.index[(df["arm"] == "t1_bb_s16") & (df["q"] == 50) & (df["seed"] == 10503)][0]
    df.loc[i, ["status", "complete"]] = ["failed", False]
    df.loc[j, ["status", "complete"]] = ["running", False]
    df.loc[k, ["status", "complete"]] = ["incomplete", False]
    assert np.isfinite(df.loc[i, R3.PRIMARY]) and np.isfinite(df.loc[j, R3.PRIMARY]) and np.isfinite(df.loc[k, "gap"])
    write_outputs(df, tmp_path)
    crit = pd.read_csv(tmp_path / "criterion.csv").set_index("arm")
    assert crit.loc["relu_bb_s1", "n_pairs_q60"] == 3 and crit.loc["t10_st_s16", "n_pairs_q50"] == 3
    assert crit.loc["relu_bb_s16", "n_pairs_q50"] == 3 and crit.loc["t10_bb_s16", "n_pairs_q50"] == 3     # t1 baseline lost
    inter = pd.read_csv(tmp_path / "interaction.csv")
    r = inter[(inter["actor"] == "relu") & (inter["starts"] == "bb") & (inter["q"] == 50)
              & (inter["metric"] == R3.PRIMARY)].iloc[0]
    assert r["n_pairs"] == 3
    tr = pd.read_csv(tmp_path / "transmission.csv")
    assert tr[(tr["arm"] == "t1_bb_s16") & (tr["q"] == 50)]["n_pairs"].iloc[0] == 3
    assert BL.main(["--analysis-dir", str(tmp_path)]) == 0


def test_blind_recomputation_unreadable_inputs_exit_2(synth_dir, tmp_path):
    for name in ("interaction.csv", "transmission.csv", "per_run.csv", "criterion.csv"):
        d = tmp_path / ("u_" + name)
        shutil.copytree(synth_dir, d)
        (d / name).unlink()
        assert BL.main(["--analysis-dir", str(d)]) == 2, name
    assert BL.main(["--analysis-dir", str(synth_dir), "--protocol", str(tmp_path / "nope.json")]) == 2


def test_blind_script_does_not_import_the_analysis_tool():
    """Plain numpy / pandas / stdlib only: no import of r1_analysis, r2_analysis, r3_analysis or anything from tools/ms."""
    src = (ROOT / "tools" / "ms" / "r3_blind_criterion.py").read_text()
    tree = ast.parse(src)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "argparse", "hashlib", "json", "math", "re", "sys", "pathlib", "typing", "numpy",
                    "pandas"}, mods
    for name in ("r1_analysis", "r2_analysis", "r3_analysis", "ms_configs", "launch_checks", "blind_criterion",
                 "import tools", "from tools", "sys.path"):
        assert ("import " + name) not in src and ("from " + name) not in src
    assert "sys.path" not in src
    assert "20261008" in src and "10000" in src and "20261007" not in src


def test_blind_script_hard_codes_the_pre_registered_scheme_and_reads_the_protocol_thresholds(tmp_path):
    assert BL.SEED == 20261008 and BL.N_RESAMPLES == 10000 and BL.TOL == 1e-12 and BL.DENOM_TOL == 1e-9
    assert BL.INTERACTION_METRICS == ("stage2_peak_rel_err_abs", "gap", "remainder", "smoothing")
    th = BL.thresholds(PROTO_PATH)
    assert th == {"eta": TH.eta, "rmse": TH.rmse, "tail": TH.tail, "n_eta": TH.n_eta}
    alt = json.loads(PROTO_PATH.read_text())
    for c in alt["gates"]["G-A"]["all_must_hold"]:
        if c["metric"] == "stage2_tail_mean_over_g2_0":
            c["threshold"] = 0.001
    (tmp_path / "p.json").write_text(json.dumps(alt))
    assert BL.thresholds(tmp_path / "p.json")["tail"] == 0.001
    assert R3.DENOM_TOL == BL.DENOM_TOL and R3.S_LOW == BL.S_LOW and R3.S_HIGH == BL.S_HIGH


def test_blind_agrees_on_a_reduced_actor_set(world, tmp_path):
    """Only t1 and t10 ran: the arms are derived from per_run.csv, relu rows do not exist in any table."""
    df = world[world["actor"].isin(["t1", "t10"])].reset_index(drop=True)
    comps = R3.primary_comparisons(("t1", "t10"))
    p, _ = R3.paired_tables(df, comps, R3.SECONDARY_METRICS, QS, SEEDS4, R3.LABEL_T1, R3.CRITERION_NOTE)
    crit = R3.criterion_table(df, p, comps, QS, SEEDS4, R3.LABEL_T1, R3.CRITERION_NOTE)
    trans, _ = R3.transmission_tables(df, R3.noise_landing_comparisons(("t1", "t10")), QS, SEEDS4)
    inter, _ = R3.interaction_tables(df, ("t1", "t10"), R3.STARTS, QS, SEEDS4)
    for name, t in (("per_run", df), ("criterion", crit), ("transmission", trans), ("interaction", inter)):
        R1.write_csv(t, tmp_path / ("%s.csv" % name), "abc")
    assert len(crit) == 4 and set(inter["actor"]) == {"t10"} and len(trans) == 8
    assert BL.main(["--analysis-dir", str(tmp_path)]) == 0
    trans.iloc[:-1].pipe(lambda t: R1.write_csv(t, tmp_path / "transmission.csv", "abc"))     # a dropped row is a disagreement
    assert BL.main(["--analysis-dir", str(tmp_path)]) == 1


# ====================================================================== 5b. ten jittered pairs per cell: the seed at EVERY call site
# The 4-seed ``world`` has hand-designed outcomes, but its bootstrap intervals are over at most four distinct values (a seed
# changed at ONE call site of the tool gives the same numbers there) and its transmission and interaction rows are constant
# over the seeds. ``make_world10`` has the design size (10 seeds per cell) and seed-to-seed variation in every quantity,
# while the arm means are structural (the jitter of an arm is a zero-sum permutation), so that means, ratios and interaction
# values can be written down by hand and the intervals still differ for every bootstrap seed.
SEEDS10 = tuple(range(10501, 10511))
J10 = np.array([-0.47, -0.31, -0.22, -0.09, -0.02, 0.05, 0.14, 0.21, 0.33, 0.38])      # zero sum, no regular spacing
assert abs(J10.sum()) < 1e-12
#: share of the smoothing reduction (s = 16 against s = 1) that the remainder takes back: transmission ratio = 1 - RHO
RHO = {("t1", "bb"): 0.2, ("t1", "st"): 0.4, ("relu", "bb"): 0.1, ("relu", "st"): 0.3, ("t10", "bb"): 0.5, ("t10", "st"): 0.7}
SM1 = {"t1": 2.0, "relu": 2.1, "t10": 2.3}                               # smoothing part at s = 1, q = 50 (effort units)
RMSE0 = {"t1": 0.020, "relu": 0.0165, "t10": 0.0185}
ACTOR_IDX = {"t1": 0, "relu": 1, "t10": 2}
G2 = {50: 70.0, 60: 58.0}                                                # g2_at_0 of ``blank``
W10_ROOTS = {"pilot": "/w10/pilot", "ms_r2_pilot": "/w10/ms_r2", "parents": "/w10/parents", "rehearsal": "/w10/rehearsal"}


def jit(key: str, q: int, quantity: str, scale: float) -> np.ndarray:
    """Ten values, a zero-sum permutation of ``amp * scale * J10``; the order and the amplitude ``amp`` (1.0 ... 1.4) are fixed
    by crc32 of (key, q, quantity), so every arm and quantity has its own: a paired difference varies from seed to seed
    (no two arms share values), every arm mean is its structural value."""
    order = sorted(range(10), key=lambda i: zlib.crc32(("%s|%d|%s|%d" % (key, q, quantity, i)).encode()))
    amp = 1.0 + 0.011 * (zlib.crc32(("%s|%d|%s|amp" % (key, q, quantity)).encode()) % 37)       # its own amplitude as well
    return amp * scale * J10[np.array(order)]


def w10_structural_peak(arm: str, q: int) -> float:
    """Mean |peak error| of an MS-R3 arm of the 10-seed world: the ``t1`` value of its cell + the mean of its ``DELTA``."""
    actor, k, s = R3.parse_arm(arm)
    base = 0.10 + (0.02 if q == 60 else 0.0) + (0.005 if s == 16 else 0.0) - (0.01 if k == "st" else 0.0)
    return base + (0.0 if actor == "t1" else float(np.mean(DELTA[(arm, q)])))


def w10_values(arm: str, q: int, like: Optional[str] = None) -> Dict[str, np.ndarray]:
    """The ten per-seed values of every populated column of ``arm`` at ``q``: structure of the MS-R3 arm ``like`` (default
    ``arm``) plus the zero-sum jitter keyed by ``arm``."""
    ref = like or arm
    actor, k, s = R3.parse_arm(ref)
    q60 = q == 60
    peak = w10_structural_peak(ref, q) + jit(arm, q, "peak", 0.004)
    sm1 = SM1[actor] + (0.05 if q60 else 0.0)
    rem1 = 1.5 + ACTOR_REM[actor] + (0.1 if k == "st" else 0.0) + (0.05 if q60 else 0.0)
    sm_mean, rem_mean = (sm1, rem1) if s == 1 else (sm1 / 4.0, rem1 + RHO[(actor, k)] * 0.75 * sm1)
    sm = sm_mean + jit(arm, q, "sm", 0.06)
    rem = rem_mean + jit(arm, q, "rem", 0.08)
    gap, g2 = sm + rem, G2[q]
    eta = 0.003 + jit(arm, q, "eta", 0.0003)
    s1 = 0.02 + 0.002 * ACTOR_IDX[actor] + jit(arm, q, "s1", 0.003)
    return {"stage2_peak_rel_err_abs": peak, "smoothing": sm, "remainder": rem, "gap": gap, "e2_at_0": g2 - gap,
            "smoothed_e_pred_0": g2 - sm, "smoothing_rel": sm / g2, "gap_rel": gap / g2, "remainder_rel": rem / g2,
            "w_eff": gap / (g2 / (2.0 * q)), "sigma_2_0": 0.65 + 0.02 * ACTOR_IDX[actor] + jit(arm, q, "sig", 0.03),
            "stage2_rmse_pos_over_g2_0": RMSE0[actor] + jit(arm, q, "rmse", 0.002),
            "stage2_tail_mean_over_g2_0": 0.005 + jit(arm, q, "tail", 0.0005), "eta_T_over_dw": eta, "eta_dev": eta + 0.0001,
            "stage1_rel_err_abs": s1, "stage1_rel_err_signed": s1}


def make_world10() -> pd.DataFrame:
    """Eighteen arms (twelve MS-R3 arms, ``parents_A``, ``rehearsal_v2_0`` and the four ``NL_*`` re-runs) x two q x ten seeds.

    * every per-seed quantity is a structural arm mean plus a zero-sum jitter (see :func:`jit`);
    * |peak| of ``v`` = the ``t1`` value of the cell + the mean of ``DELTA`` of the 4-seed world (so the criterion has its
      pattern: met for ``relu_bb_s1``, ``relu_st_s16``, ``t10_bb_s16``, ``t10_st_s1``, ``t10_st_s16``; not met for
      ``relu_bb_s16`` (q = 60), ``t10_bb_s1``; ``relu_st_s1`` is violated at (q = 50, seed 10503));
    * smoothing(s = 16) = smoothing(s = 1) / 4 and the remainder takes back ``RHO[(actor, starts)]`` of the reduction: the
      transmission ratio of a cell is exactly ``1 - RHO``; the smoothing part at s = 1 differs between the actors, so every
      interaction value is nonzero;
    * ``t1_bb_s1`` at (q = 50, seed 10502) fails G-A (eta 0.01): it is not a baseline pass;
    * ``parents_A`` / ``rehearsal_v2_0`` have |peak| and the gap, ``rehearsal_v2_0`` the stage-1 error; ``NL_*`` are the
      ``t1`` arms with their own jitter.
    """
    rows = []
    for arm in R3.r3_arms(R3.ACTORS) + list(R3.REF_ARMS):
        for q in QS:
            if arm in ("parents_A", "rehearsal_v2_0"):
                vals = w10_values(arm, q, like="t1_bb_s1")
                top = (0.08 if arm == "parents_A" else 0.07) + (0.02 if q == 60 else 0.0)
                vals["stage2_peak_rel_err_abs"] = top + jit(arm, q, "peak", 0.004)
                for c in ("smoothing", "remainder", "smoothed_e_pred_0", "smoothing_rel", "remainder_rel", "sigma_2_0"):
                    del vals[c]                                       # a reference has no noise record
                gap = 5.0 + jit(arm, q, "gap", 0.1)
                vals.update(gap=gap, e2_at_0=G2[q] - gap, gap_rel=gap / G2[q], w_eff=gap / (G2[q] / (2.0 * q)))
                if arm == "parents_A":
                    for c in ("stage1_rel_err_abs", "stage1_rel_err_signed"):
                        del vals[c]
            elif arm.startswith("NL_"):
                vals = w10_values(arm, q, like="t1_" + arm[3:])
            else:
                vals = w10_values(arm, q)
            for i, sd in enumerate(SEEDS10):
                kw = {c: float(v[i]) for c, v in vals.items()}
                if (arm, q, i) == ("t1_bb_s1", 50, 1):
                    kw.update(eta_T_over_dw=0.01, eta_dev=0.0101)
                if (arm, q, i) == ("relu_st_s1", 50, 2):
                    kw["stage2_tail_mean_over_g2_0"] = 0.03
                rows.append(settle(blank(arm, q, sd, run_dir="/w10/%s/q%d/seed%d" % (arm, q, sd), **kw)))
    return pd.DataFrame(rows, columns=R3.COLUMNS)


class Oracle:
    """The paired / interaction / summary values written out from the per-run frame (no tool function): the complete runs
    indexed by (arm, q, seed)."""

    def __init__(self, df: pd.DataFrame, seeds: Sequence[int] = SEEDS10):
        self.seeds = tuple(seeds)
        self.all = {(r["arm"], int(r["q"]), int(r["seed"])): r for r in df.to_dict("records")}
        self.by = {k: r for k, r in self.all.items() if bool(r["complete"])}

    def value(self, arm: str, q: int, seed: int, metric: str) -> float:
        r = self.by.get((arm, q, seed))
        x = float(r[metric]) if r is not None else float("nan")
        return x

    def diffs(self, arm: str, base: str, q: int, metric: str) -> np.ndarray:
        """arm - baseline over the seeds where both runs are complete and both values are finite."""
        out = []
        for sd in self.seeds:
            x, y = self.value(arm, q, sd, metric), self.value(base, q, sd, metric)
            if np.isfinite(x) and np.isfinite(y):
                out.append(x - y)
        return np.array(out, dtype=float)

    def values(self, arm: str, q: int, metric: str) -> np.ndarray:
        """Finite values of the complete runs of (arm, q) in seed order."""
        v = [self.value(arm, q, sd, metric) for sd in self.seeds]
        return np.array([x for x in v if np.isfinite(x)], dtype=float)

    def transmission(self, arm: str, base: str, q: int) -> Tuple[np.ndarray, np.ndarray]:
        """(d gap, d smoothing) over the seeds where both are finite in the pair."""
        dg, ds = [], []
        for sd in self.seeds:
            g = self.value(arm, q, sd, "gap") - self.value(base, q, sd, "gap")
            m = self.value(arm, q, sd, "smoothing") - self.value(base, q, sd, "smoothing")
            if np.isfinite(g) and np.isfinite(m):
                dg.append(g)
                ds.append(m)
        return np.array(dg), np.array(ds)

    def interaction(self, v: str, k: str, q: int, metric: str) -> np.ndarray:
        """(v_s16 - v_s1) - (t1_s16 - t1_s1) over the seeds where the four values are finite."""
        out = []
        for sd in self.seeds:
            a, b = (self.value(R3.arm_name(x, k, s), q, sd, metric) for x, s in ((v, 16), (v, 1)))
            c, d = (self.value(R3.arm_name("t1", k, s), q, sd, metric) for s in (16, 1))
            if all(np.isfinite([a, b, c, d])):
                out.append((a - b) - (c - d))
        return np.array(out, dtype=float)


def paired_expect(d: np.ndarray, direction: Optional[bool]) -> Dict[str, Any]:
    """Every number of a ``paired_summary`` row from the differences ``d`` (fresh generator per interval)."""
    nan = float("nan")
    lo, hi = indep_ci(d) if d.size else (nan, nan)
    mlo, mhi = indep_ci(d, "median") if d.size else (nan, nan)
    return {"n_pairs": d.size, "mean": float(d.mean()) if d.size else nan, "median": float(np.median(d)) if d.size else nan,
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_zero": int((d == 0).sum()),
            "ci_mean_lo": lo, "ci_mean_hi": hi, "ci_median_lo": mlo, "ci_median_hi": mhi}


def same(a: Any, b: Any) -> bool:
    """Equal, NaN == NaN."""
    return bool(a == b) or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def check_paired_table(tab: pd.DataFrame, orc: Oracle, comps: Sequence[Tuple[str, str]],
                       metrics: Sequence[Tuple[str, Optional[bool]]], label: str) -> int:
    """The rows of a paired table (arm - baseline) are exactly the rows of the omission rule (a metric without any finite pair
    only as the first metric), in table order, and every number equals the manual recomputation. Returns the row count."""
    want = []
    for arm, base in comps:
        for q in QS:
            for j, (m, direction) in enumerate(metrics):
                d = orc.diffs(arm, base, q, m)
                if d.size or j == 0:
                    want.append((arm, base, q, m, d, direction))
    got = list(tab[["arm", "baseline", "q", "metric"]].itertuples(index=False, name=None))
    assert got == [w[:4] for w in want]
    assert set(tab["comparison"]) == {label} and set(tab["n_expected"]) == {len(SEEDS10)}
    for (arm, base, q, m, d, direction), r in zip(want, tab.to_dict("records")):
        e = paired_expect(d, direction)
        for c, v in e.items():
            assert same(r[c], v) or (isinstance(v, float) and abs(r[c] - v) <= 1e-12), (arm, base, q, m, c, r[c], v)
        assert r["direction"] == ("lower is better" if direction else "none")
        assert bool(r["ci_mean_below_0"]) == bool(d.size and e["ci_mean_hi"] < 0.0)
        assert bool(r["ci_mean_contains_0"]) == bool(d.size and e["ci_mean_lo"] <= 0.0 <= e["ci_mean_hi"])
    return len(want)


def criterion_expect(orc: Oracle, arm: str, base: str) -> Dict[str, Any]:
    """Both parts of the criterion of one (arm, baseline) written out from the frame."""
    out: Dict[str, Any] = {}
    flags, full = [], []
    for q in QS:
        d = orc.diffs(arm, base, q, R3.PRIMARY)
        e = paired_expect(d, True)
        out.update({"n_pairs_q%d" % q: d.size, "mean_q%d" % q: e["mean"], "ci_mean_lo_q%d" % q: e["ci_mean_lo"],
                    "ci_mean_hi_q%d" % q: e["ci_mean_hi"], "a_q%d" % q: bool(d.size and e["ci_mean_hi"] < 0.0)})
        flags.append(out["a_q%d" % q])
        full.append(d.size == len(SEEDS10))
    viol, pend, n_base = [], [], 0
    for q in QS:
        for sd in SEEDS10:
            b = orc.all.get((base, q, sd))
            if not (b is not None and bool(b["complete"]) and bool(b["gate_pass"])):
                continue
            n_base += 1
            a = orc.all.get((arm, q, sd))
            status = str(a["status"]) if a is not None else "missing"
            if (status == "done" and not bool(a["gate_pass"])) or status == "failed":
                viol.append("q%d/%d" % (q, sd))
            elif status in ("missing", "incomplete", "running"):
                pend.append("q%d/%d" % (q, sd))
    b_status = "violated" if viol else ("incomplete" if pend else "holds")
    out.update(a_met=all(flags), a_complete=all(full), n_base_pass=n_base, b_violations=" ".join(viol),
               b_pending=" ".join(pend), b_n_violations=len(viol), b_n_pending=len(pend), b_status=b_status)
    out["overall"] = "not met" if b_status == "violated" else (
        ("met" if out["a_met"] else "not met") if out["a_complete"] and b_status == "holds" else "incomplete")
    return out


def check_criterion_table(tab: pd.DataFrame, orc: Oracle, comps: Sequence[Tuple[str, str]]) -> None:
    assert list(zip(tab["arm"], tab["baseline"])) == list(comps)
    assert (tab["boot_seed"] == BOOT).all() and (tab["n_boot"] == 10000).all()
    for (arm, base), r in zip(comps, tab.to_dict("records")):
        for c, v in criterion_expect(orc, arm, base).items():
            assert same(r[c], v) or (isinstance(v, float) and abs(r[c] - v) <= 1e-12), (arm, base, c, r[c], v)


def check_transmission_table(tab: pd.DataFrame, orc: Oracle, comps: Sequence[Tuple[str, str]]) -> None:
    """transmission.csv against the manual recomputation: a fresh generator per row, the ratio recomputed on every resample."""
    assert list(zip(tab["arm"], tab["baseline"], tab["q"])) == [(a, b, q) for a, b in comps for q in QS]
    for r in tab.to_dict("records"):
        dg, ds = orc.transmission(r["arm"], r["baseline"], r["q"])
        n = dg.size
        idx = np.random.default_rng(BOOT).integers(0, n, size=(10000, n))
        mg, ms = dg[idx].mean(axis=1), ds[idx].mean(axis=1)
        ok = np.abs(ms) > 1e-9
        rat = mg[ok] / ms[ok]
        per_seed = dg / ds
        assert r["n_pairs"] == n and r["n_boot_valid"] == int(ok.sum())
        assert r["ratio"] == pytest.approx(dg.mean() / ds.mean(), abs=1e-12)
        assert r["mean_d_gap"] == pytest.approx(dg.mean(), abs=1e-12) and r["mean_d_smoothing"] == pytest.approx(ds.mean(), abs=1e-12)
        assert r["ci_lo"] == float(np.percentile(rat, 2.5)) and r["ci_hi"] == float(np.percentile(rat, 97.5))
        assert r["median_per_seed_ratio"] == pytest.approx(float(np.median(per_seed)), abs=1e-12)
        assert r["per_seed_ratios"] == ";".join("%d:%.6g" % (sd, x) for sd, x in zip(SEEDS10, per_seed))


def check_interaction_table(tab: pd.DataFrame, orc: Oracle) -> None:
    keys = [(v, k, q, m) for v in ("relu", "t10") for k in R3.STARTS for q in QS for m in R3.INTERACTION_METRICS]
    assert list(zip(tab["actor"], tab["starts"], tab["q"], tab["metric"])) == keys
    for (v, k, q, m), r in zip(keys, tab.to_dict("records")):
        for c, x in paired_expect(orc.interaction(v, k, q, m), None).items():
            assert same(r[c], x) or (isinstance(x, float) and abs(r[c] - x) <= 1e-12), (v, k, q, m, c, r[c], x)


def check_summary_rows(tab: pd.DataFrame, orc: Oracle, value_col: str) -> int:
    """Rows (arm, q, ``value_col``) of a ``summary_stats`` table against the manual recomputation of n / mean / median / sd /
    min / max and the interval of the mean (fresh generator, seed 20261008). Returns the number of rows."""
    for r in tab.to_dict("records"):
        x = orc.values(r["arm"], int(r["q"]), r[value_col])
        assert x.size > 1, r                                            # the 10-seed world has data in every listed row
        lo, hi = indep_ci(x)
        assert r["n"] == x.size and abs(r["mean"] - x.mean()) <= 1e-12 and abs(r["median"] - np.median(x)) <= 1e-12, r
        assert r["min"] == x.min() and r["max"] == x.max() and abs(r["sd"] - x.std(ddof=1)) <= 1e-12, r
        assert r["ci_mean_lo"] == lo and r["ci_mean_hi"] == hi, (r["arm"], r["q"], r[value_col])
    return len(tab)


def run_w10(df: pd.DataFrame, out: Path, offset: int = 0) -> Tuple[Dict[str, pd.DataFrame], List[Any]]:
    """``R3.run_analysis`` (the whole table-building of the CLI, figures off) on a per-run frame given in place of the disk
    extraction, with a spy around ``numpy.random.default_rng`` that records every seed (``offset`` shifts them: the
    'bootstrap seed changed everywhere' variant)."""
    seen: List[Any] = []
    orig = np.random.default_rng

    def spy(seed: Any = None, *a: Any, **kw: Any) -> Any:
        seen.append(seed)
        return orig(seed + offset if (offset and seed is not None) else seed, *a, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(np.random, "default_rng", spy)
    mp.setattr(R3, "extract_all", lambda *a, **kw: df)
    try:
        tables, _ = R3.run_analysis(W10_ROOTS, out, QS, SEEDS10, R3.ACTORS, PROTO_PATH, R2.Windows(), False)
    finally:
        mp.undo()
    return tables, seen


@pytest.fixture(scope="module")
def w10() -> pd.DataFrame:
    return make_world10()


@pytest.fixture(scope="module")
def w10_orc(w10) -> Oracle:
    return Oracle(w10)


@pytest.fixture(scope="module")
def w10_run(w10, tmp_path_factory) -> Tuple[Dict[str, pd.DataFrame], List[Any]]:
    """Every table of the tool for the 10-seed world, with the bootstrap seed of every interval recorded."""
    return run_w10(w10, tmp_path_factory.mktemp("w10_real") / "out")


@pytest.fixture(scope="module")
def w10_shifted(w10, tmp_path_factory) -> Tuple[Dict[str, pd.DataFrame], List[Any]]:
    """The same run with every generator seed shifted by one (the 'bootstrap seed changed everywhere' mutant)."""
    return run_w10(w10, tmp_path_factory.mktemp("w10_shifted") / "out", offset=1)


def test_the_ten_seed_world_is_structured_and_every_interval_input_varies(w10, w10_orc):
    """The fixture can show a seed change: in every cell the paired differences, the changes s16 - s1, the per-seed
    transmission ratios and the interaction values take ten different values, while the arm means are the structural ones."""
    assert len(w10) == 18 * 2 * 10 and w10["complete"].all() and set(w10["status"]) == {"done"}
    for arm, base in R3.primary_comparisons(R3.ACTORS):
        for q in QS:
            d = w10_orc.diffs(arm, base, q, R3.PRIMARY)
            assert d.size == 10 and np.unique(np.round(d, 12)).size == 10, (arm, q)
            assert d.mean() == pytest.approx(np.mean(DELTA[(arm, q)]), abs=1e-12)           # structural: the DELTA mean
    for a, b in R3.noise_landing_comparisons(R3.ACTORS):
        for q in QS:
            dg, ds = w10_orc.transmission(a, b, q)
            assert min(np.unique(np.round(x, 12)).size for x in (dg, ds, dg / ds)) == 10, (a, q)
    for v in ("relu", "t10"):
        for k in R3.STARTS:
            for q in QS:
                for m in R3.INTERACTION_METRICS:
                    assert np.unique(np.round(w10_orc.interaction(v, k, q, m), 12)).size == 10, (v, k, q, m)
    # the arm means are structural: zero-sum jitter
    for arm in R3.r3_arms(R3.ACTORS):
        for q in QS:
            sm1 = SM1[R3.parse_arm(arm)[0]] + (0.05 if q == 60 else 0.0)
            want = sm1 if arm.endswith("_s1") else sm1 / 4.0
            assert w10_orc.values(arm, q, "smoothing").mean() == pytest.approx(want, abs=1e-12), (arm, q)


def test_criterion_of_the_ten_seed_world_equals_the_manual_recomputation_and_the_designed_pattern(w10_run, w10_orc):
    tables, _ = w10_run
    crit = tables["criterion"]
    comps = R3.primary_comparisons(R3.ACTORS)
    check_criterion_table(crit, w10_orc, comps)
    c = crit.set_index("arm")
    # by hand: the designed pattern (the mean of a cell is the DELTA mean, the interval decides (a))
    assert list(crit["overall"]) == ["met", "not met", "not met", "met", "not met", "met", "met", "met"]
    assert list(crit["b_status"]) == ["holds", "holds", "violated", "holds", "holds", "holds", "holds", "holds"]
    assert c.loc["relu_st_s1", "b_violations"] == "q50/10503" and c.loc["relu_bb_s1", "n_base_pass"] == 19
    assert c.loc["t10_bb_s1", "n_base_pass"] == 19 and c.loc["relu_bb_s16", "n_base_pass"] == 20
    for arm in c.index:
        for q in QS:
            assert c.loc[arm, "mean_q%d" % q] == pytest.approx(np.mean(DELTA[(arm, q)]), abs=1e-12), (arm, q)
            lo, hi = c.loc[arm, "ci_mean_lo_q%d" % q], c.loc[arm, "ci_mean_hi_q%d" % q]
            assert lo < c.loc[arm, "mean_q%d" % q] < hi                                       # a real interval
    assert not c.loc["relu_bb_s16", "a_q60"] and c.loc["relu_bb_s16", "ci_mean_lo_q60"] > 0.0      # mean +0.0125: above 0
    assert c.loc["t10_bb_s1", "ci_mean_lo_q50"] > 0.0 and c.loc["relu_bb_s1", "ci_mean_hi_q50"] < 0.0
    # the criterion table against parents_A (MS-R1's criterion, every arm): same manual recomputation, and by hand
    arms = R3.r3_arms(R3.ACTORS)
    cp_tab = tables["criterion_vs_parents_A"]
    check_criterion_table(cp_tab, w10_orc, [(a, "parents_A") for a in arms])
    cp = cp_tab.set_index("arm")
    assert list(cp.index) == arms and (cp["comparison"] == R3.LABEL_PARENTS).all() and (cp["note"] == R3.PARENTS_NOTE).all()
    for arm in arms:              # parents_A: |peak| 0.08 at q = 50 and 0.10 at q = 60, zero-sum jitter
        assert cp.loc[arm, "mean_q50"] == pytest.approx(w10_structural_peak(arm, 50) - 0.08, abs=1e-12), arm
        assert cp.loc[arm, "mean_q60"] == pytest.approx(w10_structural_peak(arm, 60) - 0.10, abs=1e-12), arm
    assert cp.loc["t1_bb_s1", "n_base_pass"] == 20 and cp.loc["relu_st_s1", "b_violations"] == "q50/10503"


def test_transmission_of_the_ten_seed_world_has_real_intervals_and_the_designed_ratios(w10_run, w10_orc):
    tables, _ = w10_run
    tr = tables["transmission"]
    check_transmission_table(tr, w10_orc, R3.noise_landing_comparisons(R3.ACTORS))
    assert len(tr) == 12 and (tr["n_pairs"] == 10).all() and (tr["n_boot_valid"] == 10000).all()
    for r in tr.itertuples():
        actor, k, _ = R3.parse_arm(r.arm)
        sm1 = SM1[actor] + (0.05 if r.q == 60 else 0.0)
        # by hand: the smoothing part falls by 3/4 of its s = 1 value, the remainder takes back RHO of that
        assert r.ratio == pytest.approx(1.0 - RHO[(actor, k)], abs=1e-12), (r.arm, r.q)
        assert r.mean_d_smoothing == pytest.approx(-0.75 * sm1, abs=1e-12)
        assert r.mean_d_gap == pytest.approx(-(1.0 - RHO[(actor, k)]) * 0.75 * sm1, abs=1e-12)
        assert r.ci_lo < r.ratio < r.ci_hi and r.ci_hi - r.ci_lo > 0.01, (r.arm, r.q)            # not degenerate
        per_seed = [float(x.split(":")[1]) for x in r.per_seed_ratios.split(";")]
        assert len(set(per_seed)) == 10 and min(per_seed) < r.ratio < max(per_seed)
    # the transmission rows of noise_landing.csv are these rows
    nl = tables["noise_landing"]
    tx = nl[nl["metric"] == "transmission_ratio"].reset_index(drop=True)
    assert len(tx) == 12 and list(zip(tx["arm"], tx["q"])) == list(zip(tr["arm"], tr["q"]))
    assert (tx["mean"] == tr["ratio"]).all() and (tx["ci_mean_lo"] == tr["ci_lo"]).all() and (tx["ci_mean_hi"] == tr["ci_hi"]).all()


def test_interaction_of_the_ten_seed_world_has_real_values_by_hand(w10_run, w10_orc):
    tables, _ = w10_run
    inter = tables["interaction"]
    check_interaction_table(inter, w10_orc)
    assert len(inter) == 32
    i = inter.set_index(["actor", "starts", "q", "metric"])
    for v in ("relu", "t10"):
        for k in R3.STARTS:
            for q in QS:
                sm1 = {a: SM1[a] + (0.05 if q == 60 else 0.0) for a in ("t1", v)}
                # smoothing: (v_s16 - v_s1) - (t1_s16 - t1_s1) = -0.75 (sm1_v - sm1_t1)
                want_sm = -0.75 * (sm1[v] - sm1["t1"])
                # remainder: the part of the smoothing reduction that each actor's remainder takes back
                want_rem = 0.75 * (RHO[(v, k)] * sm1[v] - RHO[("t1", k)] * sm1["t1"])
                want_pk = np.mean(DELTA[(R3.arm_name(v, k, 16), q)]) - np.mean(DELTA[(R3.arm_name(v, k, 1), q)])
                got = {m: i.loc[(v, k, q, m)] for m in R3.INTERACTION_METRICS}
                assert got["smoothing"]["mean"] == pytest.approx(want_sm, abs=1e-12), (v, k, q)
                assert got["remainder"]["mean"] == pytest.approx(want_rem, abs=1e-12), (v, k, q)
                assert got["gap"]["mean"] == pytest.approx(want_sm + want_rem, abs=1e-12), (v, k, q)
                assert got[R3.PRIMARY]["mean"] == pytest.approx(want_pk, abs=1e-12), (v, k, q)
                for m, r in got.items():
                    assert r["n_pairs"] == 10 and abs(r["mean"]) > 0.005, (v, k, q, m)           # not float-noise zeros
                    assert r["ci_mean_lo"] < r["mean"] < r["ci_mean_hi"] and r["ci_mean_hi"] - r["ci_mean_lo"] > 1e-4, (v, k, q, m)
                    assert r["ci_median_lo"] < r["ci_median_hi"]


PAIRED_TABLES = {
    # table name: (comparisons, metrics, label) as run_analysis builds them for the 10-seed world
    "paired_secondary": (R3.primary_comparisons(R3.ACTORS), R3.SECONDARY_METRICS, R3.LABEL_T1),
    "starts_effect": (R3.starts_comparisons(R3.ACTORS), R3.SECONDARY_METRICS, R3.LABEL_STARTS),
    "paired_vs_parents_A": ([(a, "parents_A") for a in R3.r3_arms(R3.ACTORS)], R3.REF_METRICS, R3.LABEL_PARENTS),
    "paired_vs_rehearsal_v2_0": ([(a, "rehearsal_v2_0") for a in R3.r3_arms(R3.ACTORS)], R1.S1_METRICS, R3.LABEL_REHEARSAL),
    "paired_vs_ms_r2_nl": (R3.ms_r2_comparisons(R3.ACTORS), R3.CONTEXT_METRICS, R3.LABEL_MS_R2),
    "stage1_vs_t1": (R3.primary_comparisons(R3.ACTORS), R1.S1_METRICS, R3.LABEL_STAGE1_T1)}


@pytest.mark.parametrize("name", sorted(PAIRED_TABLES))
def test_every_paired_table_of_the_ten_seed_world_equals_the_manual_recomputation(w10_run, w10_orc, name):
    """Mean, median, sign counts and BOTH intervals (mean and median) of every row, fresh generator with seed 20261008 per
    interval, in table order."""
    comps, metrics, label = PAIRED_TABLES[name]
    n = check_paired_table(w10_run[0][name], w10_orc, comps, metrics, label)
    assert n > 30


def test_noise_landing_table_is_the_paired_changes_followed_by_the_transmission_rows(w10_run, w10_orc):
    tables, _ = w10_run
    nl, tr = tables["noise_landing"], tables["transmission"]
    paired = nl[nl["metric"] != "transmission_ratio"].reset_index(drop=True)
    n = check_paired_table(paired, w10_orc, R3.noise_landing_comparisons(R3.ACTORS), R3.SECONDARY_METRICS, R3.LABEL_NL)
    assert len(nl) == n + len(tr) and list(nl["metric"].iloc[:n]) == list(paired["metric"])        # paired block first, then 12 rows


def test_the_summary_tables_of_the_ten_seed_world_equal_the_manual_recomputation(w10_run, w10_orc):
    """arm_summary.csv, freeze_decomposition.csv (summary_stats of the MS-R3 seed) and stage1.csv (R1.stage1_table inside
    ``r3_bootstrap``): n, mean, median, sd, min, max and the interval of the mean, row by row."""
    tables, _ = w10_run
    arm_s, fd, s1 = tables["arm_summary"], tables["freeze_decomposition"], tables["stage1"]
    assert check_summary_rows(arm_s, w10_orc, "metric") > 200
    assert check_summary_rows(fd, w10_orc, "quantity") > 100
    n = 0
    for r in s1.to_dict("records"):
        for col in ("stage1_rel_err_signed", "stage1_rel_err_abs", "t1_R_final", "t1_R_dev", "t1_Delta_final", "learning_rel",
                    "inherited_rel", "Gmax_full_over_dw"):
            x = w10_orc.values(r["arm"], int(r["q"]), col)
            lo, hi = indep_ci(x) if x.size else (float("nan"), float("nan"))
            assert same(r[col + "_ci_lo"], lo) and same(r[col + "_ci_hi"], hi), (r["arm"], r["q"], col)
            assert abs(r[col + "_mean"] - (x.mean() if x.size else float("nan"))) <= 1e-12 or (x.size == 0 and math.isnan(r[col + "_mean"]))
            n += int(x.size > 0)
    assert n >= 20                                                       # the stage-1 columns of the arms and of rehearsal_v2_0


def test_every_bootstrap_draw_of_the_whole_analysis_uses_the_ms_r3_seed(w10_run):
    """A spy around ``numpy.random.default_rng`` saw every generator the tool created while it built all tables of the
    10-seed world: always 20261008 (never the MS-R1 20261006 or the MS-R2 20261007 of the modules it reuses, never an
    unseeded generator). The spy sees only generators made through ``default_rng``; a call site that builds its generator
    another way is caught by the manual recomputations above (same numbers needed) and by the sensitivity test below (the
    shifted run would leave that table unchanged)."""
    _, seen = w10_run
    assert len(seen) > 2000 and set(seen) == {20261008}


# the tables of the tool that carry a bootstrap interval, with the interval columns
INTERVAL_COLUMNS = {
    "criterion": ["ci_mean_lo_q50", "ci_mean_hi_q50", "ci_mean_lo_q60", "ci_mean_hi_q60"],
    "paired_secondary": ["ci_mean_lo", "ci_mean_hi", "ci_median_lo", "ci_median_hi"],
    "noise_landing": ["ci_mean_lo", "ci_mean_hi", "ci_median_lo", "ci_median_hi"],
    "transmission": ["ci_lo", "ci_hi"],
    "interaction": ["ci_mean_lo", "ci_mean_hi", "ci_median_lo", "ci_median_hi"],
    "starts_effect": ["ci_mean_lo", "ci_mean_hi", "ci_median_lo", "ci_median_hi"],
    "paired_vs_parents_A": ["ci_mean_lo", "ci_mean_hi", "ci_median_lo", "ci_median_hi"],
    "criterion_vs_parents_A": ["ci_mean_lo_q50", "ci_mean_hi_q50", "ci_mean_lo_q60", "ci_mean_hi_q60"],
    "paired_vs_rehearsal_v2_0": ["ci_mean_lo", "ci_mean_hi", "ci_median_lo", "ci_median_hi"],
    "paired_vs_ms_r2_nl": ["ci_mean_lo", "ci_mean_hi", "ci_median_lo", "ci_median_hi"],
    "stage1": ["stage1_rel_err_abs_ci_lo", "stage1_rel_err_abs_ci_hi", "stage1_rel_err_signed_ci_lo",
               "stage1_rel_err_signed_ci_hi"],
    "stage1_vs_t1": ["ci_mean_lo", "ci_mean_hi", "ci_median_lo", "ci_median_hi"],
    "freeze_decomposition": ["ci_mean_lo", "ci_mean_hi"],
    "arm_summary": ["ci_mean_lo", "ci_mean_hi"],
    "predictions": ["p3_transmission_ci_lo", "p3_transmission_ci_hi"]}


@pytest.mark.parametrize("name", sorted(INTERVAL_COLUMNS))
def test_every_bootstrap_table_changes_when_the_seed_changes(w10_run, w10_shifted, name):
    """Sensitivity: with every generator seed shifted by one, EVERY interval column of EVERY table of the 10-seed world
    changes, so a wrong seed at any one call site cannot go unseen there. The interval of a mean changes in nearly every row
    (a few rows keep it by chance: the resampled means of ten values are a discrete set); the interval of a median is
    coarser still (the median of a resample of ten values has few atoms), so for it a handful of rows is the criterion."""
    real, shifted = w10_run[0][name], w10_shifted[0][name]
    assert real.shape == shifted.shape
    for c in INTERVAL_COLUMNS[name]:
        a, b = pd.to_numeric(real[c], errors="coerce").to_numpy(), pd.to_numeric(shifted[c], errors="coerce").to_numpy()
        fin = np.isfinite(a)
        assert fin.sum() >= 8, (name, c)
        changed = a[fin] != b[fin]
        if "median" in c:
            assert changed.sum() >= 3, (name, c, int(changed.sum()))
        else:
            assert changed.mean() > 0.6, (name, c, changed.mean())


def predictions_expect(orc: Oracle, actor: str, k: str, q: int) -> Dict[str, Any]:
    """The evidence of P1-P4 for one (actor, starts, q) cell, from the frame by independent formulas."""
    lo, hi, t_lo = R3.arm_name(actor, k, 1), R3.arm_name(actor, k, 16), R3.arm_name("t1", k, 1)
    out: Dict[str, Any] = {}
    if actor != "t1":
        d = orc.diffs(lo, t_lo, q, R3.PRIMARY)
        out.update(p1_n_pairs=d.size, p1_n_lower=int((d < 0).sum()), p1_n_higher=int((d > 0).sum()), p1_mean_change=d.mean(),
                   p1_abs_peak_s1=orc.values(lo, q, R3.PRIMARY).mean(), p1_abs_peak_t1_s1=orc.values(t_lo, q, R3.PRIMARY).mean())
    rem = np.array([orc.value(lo, q, sd, "remainder") for sd in SEEDS10])
    sm = np.array([orc.value(lo, q, sd, "smoothing") for sd in SEEDS10])
    ok = np.isfinite(rem) & np.isfinite(sm) & (sm > 0.0)
    out.update(p2_n=int(ok.sum()), p2_mean_remainder=rem[ok].mean(), p2_mean_smoothing=sm[ok].mean(),
               p2_remainder_over_smoothing=rem[ok].mean() / sm[ok].mean(), p2_median_run_ratio=float(np.median(rem[ok] / sm[ok])),
               p2_share_abs_remainder_lt_smoothing=float(np.mean(np.abs(rem[ok]) < sm[ok])),
               p2_mean_e_hat_2_0=orc.values(lo, q, "e2_at_0").mean(), p2_mean_e_sigma_0=orc.values(lo, q, "smoothed_e_pred_0").mean())
    peak, sig = orc.values(hi, q, R3.PRIMARY), orc.values(hi, q, "sigma_2_0")
    floor = float(np.mean(sig / (math.sqrt(math.pi) * q)))
    dg, ds = orc.transmission(hi, lo, q)
    idx = np.random.default_rng(BOOT).integers(0, dg.size, size=(10000, dg.size))
    rat = dg[idx].mean(axis=1) / ds[idx].mean(axis=1)
    out.update(p3_n=peak.size, p3_abs_peak_s16=peak.mean(), p3_floor_rel_sigma=floor,
               p3_floor_rel_smoothing=orc.values(hi, q, "smoothing_rel").mean(), p3_abs_peak_over_floor=peak.mean() / floor,
               p3_transmission_ratio=dg.mean() / ds.mean(), p3_transmission_ci_lo=float(np.percentile(rat, 2.5)),
               p3_transmission_ci_hi=float(np.percentile(rat, 97.5)))
    for s_, arm_ in ((1, lo), (16, hi)):
        base = R3.arm_name("t1", k, s_)
        out["p4_rmse_s%d" % s_] = orc.values(arm_, q, "stage2_rmse_pos_over_g2_0").mean()
        out["p4_rmse_t1_s%d" % s_] = orc.values(base, q, "stage2_rmse_pos_over_g2_0").mean()
        if actor != "t1":
            d = orc.diffs(arm_, base, q, "stage2_rmse_pos_over_g2_0")
            out.update({"p4_n_pairs_s%d" % s_: d.size, "p4_n_lower_s%d" % s_: int((d < 0).sum()),
                        "p4_mean_change_s%d" % s_: d.mean()})
    return out


def test_predictions_of_the_ten_seed_world_equal_the_independent_formulas_in_every_cell(w10_run, w10_orc):
    tables, _ = w10_run
    pred = tables["predictions"].set_index(["actor", "starts", "q"])
    assert len(pred) == 12
    for a in R3.ACTORS:
        for k in R3.STARTS:
            for q in QS:
                row = pred.loc[(a, k, q)]
                for c, v in predictions_expect(w10_orc, a, k, q).items():
                    assert row[c] == pytest.approx(v, abs=1e-12, rel=1e-12), (a, k, q, c, row[c], v)
                if a == "t1":
                    assert math.isnan(row["p1_n_pairs"]) and math.isnan(row["p4_n_pairs_s1"])
                else:                            # the designed pattern: the sign counts of P1 by hand (DELTA signs, ten pairs)
                    d = np.mean(DELTA[(R3.arm_name(a, k, 1), q)])
                    assert (row["p1_n_lower"], row["p1_n_higher"]) == ((10, 0) if d < 0 else (0, 10)), (a, k, q)
    # P2 / P3 by hand for one cell: relu, bb, q = 50: remainder 1.0 + 0.0 (s = 1), smoothing 2.1
    r = pred.loc[("relu", "bb", 50)]
    assert r["p2_mean_smoothing"] == pytest.approx(2.1, abs=1e-12) and r["p2_mean_remainder"] == pytest.approx(1.0, abs=1e-12)
    assert r["p2_remainder_over_smoothing"] == pytest.approx(1.0 / 2.1, abs=1e-12)
    assert r["p3_transmission_ratio"] == pytest.approx(0.9, abs=1e-12) and r["p3_floor_rel_smoothing"] == pytest.approx(0.525 / 70.0)
    assert r["p3_transmission_ci_lo"] < 0.9 < r["p3_transmission_ci_hi"]


def test_the_blind_script_agrees_on_the_ten_seed_world(w10, tmp_path):
    """The independent recomputation (criterion, transmission, interaction) agrees with the tool on ten jittered pairs per
    cell, where the intervals depend on the seed of every call site."""
    write_outputs(w10, tmp_path, SEEDS10)
    assert BL.main(["--analysis-dir", str(tmp_path)]) == 0
    assert "DISAGREE" not in (tmp_path / "blind_recomputation.txt").read_text()


# ====================================================================== 6. a real tiny wave
FAKE_GIT = {"commit": "abc1234", "short": "abc1234", "dirty": False}
Q, SEED = 50, 10501
RAMP = (21, 40)
WIN = R2.Windows(ramp_first=21, ramp_last=40, hold_last=48, decay_last=60, traj_from=20)
TINY: List[Tuple[str, str, int]] = [(a, "bb", s) for a in R3.ACTORS for s in (1, 16)]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Six reduced runs of one (q, seed): t1 / relu / t10 x bin-balanced x s = 1, 16 (ramp 21-40, constant LR to 48, decay to
    60, stage 1: 20 updates, a weight export every 5 updates), built from the NL arms of MS-R2 with ``actor_variant`` set."""
    from run import run_ms_stagewise as rms
    from test_ms_r2_runner import reduced_nl
    root = tmp_path_factory.mktemp("tinywave3")
    pilot = root / "pilot"
    mp = pytest.MonkeyPatch()
    mp.setattr(rms, "git_state", lambda: dict(FAKE_GIT))
    try:
        for actor, k, s in TINY:
            arm = R3.arm_name(actor, k, s)
            d = pilot / ("q%d" % Q) / ("seed%d" % SEED) / arm
            d.mkdir(parents=True)
            cfg = reduced_nl("NL_%s_s%d" % (k, s), str(d), q=Q, seed=SEED, ramp=RAMP)
            cfg["arm"], cfg["run"] = arm, "msr3_q%d_s%d_%s" % (Q, SEED, arm)
            if actor != "t1":
                cfg["actor_variant"] = actor
            assert rms.run_pipeline(cfg, str(d), "pytest", band_step=2.0) == 0
    finally:
        mp.undo()
    refs = {k: root / k for k in ("ms_r2_pilot", "parents", "rehearsal")}
    for p in refs.values():
        p.mkdir()
    return {"pilot": str(pilot), **{k: str(v) for k, v in refs.items()}}, root


#: Scalars of the genuine reference runs written by :func:`write_reference_runs` (the hand values of the vs-parents /
#: vs-rehearsal tests). The ``parents_A`` run and the stage-2 half of ``rehearsal_v2_0`` have the same shape, different numbers.
PARENTS_FINAL = {"stage2_peak_rel_err_signed": -0.08, "stage2_peak_rel_err_abs": 0.08, "stage2_rmse_pos_over_g2_0": 0.02,
                 "stage2_tail_mean_over_g2_0": 0.005, "stage2_tail_max_over_g2_0": 0.01, "e2_at_0": 64.4, "g2_at_0": 70.0,
                 "sigma_effort_at_0_t2": 0.5, "eta_T_over_dw": 0.003}
REHEARSAL_FINAL = {**PARENTS_FINAL, "stage2_peak_rel_err_signed": 0.03, "stage2_peak_rel_err_abs": 0.03, "e2_at_0": 67.9,
                   "sigma_effort_at_0_t2": 0.6}
REHEARSAL_STAGE1 = {"stage1_rel_err_signed": 0.045, "stage1_rel_err_abs": 0.045, "e1_at_0": 44.55, "g1": 46.6666666667,
                    "sigma_effort_at_0_t1": 0.7, "Gmax_full_over_dw": 0.004}


def _write_reference_arrays(d: Path, names: Sequence[str], scale: float) -> None:
    """The freeze arrays of a reference run (recovery grid and verifier nodes of the terminal stage) in both tiers."""
    rd = np.arange(-200.0, 200.5, 0.5)
    g2 = 70.0 * np.clip(1.0 - np.abs(rd) / 100.0, 0.0, None)
    vd = np.arange(-200.0, 200.5, 4.0)
    tent = 70.0 * np.clip(1.0 - np.abs(vd) / 100.0, 0.0, None)
    for n in names:
        np.savez(d / n, recovery_d_grid=rd, recovery_e2=scale * g2, recovery_g2=g2, v_t2_d_grid=vd,
                 v_t2_e_hat=scale * tent, v_t2_a_dev=tent * 0.97 + 0.3)


def write_reference_runs(root: Path) -> Dict[str, str]:
    """A ``parents_A`` run and a ``rehearsal_v2_0`` run for (q = 50, seed 10501) in the layouts ``r1_analysis`` reads
    (``<root>/<ref>/q50/seed10501/``): ``status.json``, ``final_v2.json`` + ``final_*.npz`` + ``v2_run_summary.json`` for
    ``parents_A``; ``gates.json`` (``end_of_A`` / ``end_of_B``), ``induced_band.json``, ``v2_run_summary.json``,
    ``run_config.json`` + ``gateA_*.npz`` for ``rehearsal_v2_0``. Returns the two root paths."""
    status = {"state": "done", "exit_code": 0, "git": {"commit": "feedbee", "dirty": False}, "final_global_update": 1600,
              "total_wall_sec": 321.0}
    dev = {"eta_T_over_dw": 0.0031}
    par = root / "parents" / ("q%d" % Q) / ("seed%d" % SEED)
    reh = root / "rehearsal" / ("q%d" % Q) / ("seed%d" % SEED)
    par.mkdir(parents=True)
    reh.mkdir(parents=True)
    (par / "status.json").write_text(json.dumps(status))
    (par / "final_v2.json").write_text(json.dumps({"final": PARENTS_FINAL, "development": dev}))
    (par / "v2_run_summary.json").write_text(json.dumps({"phase_timing": {"A": {"updates": 1600, "wall_sec": 300.0,
                                                                                "process_cpu_sec": 280.0}},
                                                         "costs": {"total_episodes": 819200, "minibatch_steps": 64000}}))
    _write_reference_arrays(par, ("final_final.npz", "final_development.npz"), 0.93)
    (reh / "status.json").write_text(json.dumps(status))
    (reh / "gates.json").write_text(json.dumps({
        "outcome": "pass", "global_rng": {"status": "ok"}, "run_pass": True,
        "reported": {"end_of_A": {"final": REHEARSAL_FINAL, "development": dev},
                     "end_of_B": {"final": REHEARSAL_STAGE1, "development": {"Gmax_full_over_dw": 0.0041}}}}))
    (reh / "induced_band.json").write_text(json.dumps({}))
    (reh / "v2_run_summary.json").write_text(json.dumps({"phase_timing": {"A": {"updates": 1600, "wall_sec": 300.0},
                                                                          "B": {"updates": 1800, "wall_sec": 400.0}}}))
    (reh / "run_config.json").write_text(json.dumps({"record": {"protocol": {"episodes_per_update": 512}}}))
    _write_reference_arrays(reh, ("gateA_final.npz", "gateA_development.npz"), 0.97)
    return {"parents": str(root / "parents"), "rehearsal": str(root / "rehearsal")}


@pytest.fixture(scope="module")
def tiny_refs(tiny, tmp_path_factory):
    """The tiny wave's roots with a genuine ``parents_A`` run and a genuine ``rehearsal_v2_0`` run (MS-R2 references stay
    absent)."""
    roots, _ = tiny
    return dict(roots, **write_reference_runs(tmp_path_factory.mktemp("refs")))


@pytest.fixture(scope="module")
def tiny_df(tiny):
    roots, _ = tiny
    df = R3.extract_all(roots, (Q,), (SEED,), TH, PROTO, R3.ACTORS, ("bb",), (1, 16))
    fl = R3.first_layer_table(df, WIN, PROTO, R3.r3_arms(R3.ACTORS, ("bb",)))
    return R3.attach_last_export(df, fl, WIN)


def test_extraction_of_the_tiny_wave(tiny_df):
    df = tiny_df
    arms = R3.r3_arms(R3.ACTORS, ("bb",))
    ms = df[df["arm"].isin(arms)]
    assert list(ms["arm"]) == arms and (ms["status"] == "done").all() and ms["complete"].all()
    assert (ms["role"] == "ms_arm").all() and (ms["flags"] == "").all()
    ref = df[~df["arm"].isin(arms)]
    assert list(ref["arm"]) == list(R3.REF_ARMS) and (ref["status"] == "missing").all() and (ref["role"] == "comparator").all()
    assert not ref["complete"].astype(bool).any()
    assert list(df.columns) == R3.COLUMNS and df.shape[0] == 12
    for r in ms.itertuples():                                         # the decomposition columns are the formulas
        assert r.gap == pytest.approx(r.g2_at_0 - r.e2_at_0, abs=1e-12)
        assert r.smoothing == pytest.approx(r.g2_at_0 - r.smoothed_e_pred_0, abs=1e-12)
        assert r.remainder == pytest.approx(r.smoothed_e_pred_0 - r.e2_at_0, abs=1e-12)
        assert r.gap == pytest.approx(r.smoothing + r.remainder, abs=1e-12) and r.sigma_2_0 == r.sigma_effort_at_0_t2
        assert 0.9 < r.smoothing_over_formula < 1.1
        assert r.conc_scale_final == pytest.approx(R3.parse_arm(r.arm)[2])           # the scale at the freeze is s
        assert (r.actor, r.starts, r.s) == (R3.parse_arm(r.arm)[0], "bb", R3.parse_arm(r.arm)[2])
        assert r.actor_variant_export == r.actor                                       # read from the export, absent = t1


def test_the_decomposition_equals_the_rule_log_noise_record(tiny_df):
    for r in tiny_df[tiny_df["arm"].isin(R3.r3_arms(R3.ACTORS, ("bb",)))].itertuples():
        rec = R2.noise_record(Path(r.run_dir))
        assert rec["gap"] == pytest.approx(r.gap, abs=1e-9) and rec["smoothing"] == pytest.approx(r.smoothing, abs=1e-9)
        assert rec["remainder"] == pytest.approx(r.remainder, abs=1e-9) and rec["conc_scale"] == r.conc_scale_final
        assert rec["sigma_0"] == pytest.approx(r.sigma_2_0, rel=1e-5) and rec["e_sigma_0"] == r.smoothed_e_pred_0


def test_the_variants_changed_the_run_but_not_the_shared_prefix(tiny_df):
    """C-NL in the tiny wave (an s = 16 run equals its s = 1 run through the ramp start) and the variants differ."""
    e = tiny_df.set_index("arm")["e2_at_0"]
    assert len({round(float(e["t1_bb_s1"]), 9), round(float(e["relu_bb_s1"]), 9), round(float(e["t10_bb_s1"]), 9)}) == 3
    for a in ("t1", "relu", "t10"):
        u1 = pd.read_csv(Path(tiny_df.set_index("arm").loc["%s_bb_s1" % a, "run_dir"]) / "ms_updates.csv")
        u16 = pd.read_csv(Path(tiny_df.set_index("arm").loc["%s_bb_s16" % a, "run_dir"]) / "ms_updates.csv")
        m = (u1["stage"] == 2) & (u1["local"] <= 21)
        assert np.array_equal(u1[m]["mean_effort"].to_numpy(), u16[m]["mean_effort"].to_numpy()), a


def test_the_tie_columns_equal_the_gates_json_record(tiny_df):
    """The location-free peak and its argmax and the symmetry error are recomputed from the freeze arrays and equal the
    values the runner recorded (``stage2_peak_locfree_*`` over the whole grid; ``stage2_sym_err_max`` over the whole grid
    equals the |d| < 2q one when its argmax is inside the support, which the check asserts)."""
    for r in tiny_df[tiny_df["role"] == "ms_arm"].itertuples():
        g = json.loads((Path(r.run_dir) / "gates.json").read_text())["reported"]["end_of_stage2"]["final"]
        assert r.peak_locfree_rel_err == g["stage2_peak_locfree_rel_err"]
        assert r.peak_locfree_argmax_d == g["stage2_peak_locfree_argmax_d"]
        z = np.load(Path(r.run_dir) / "freeze_stage2_final.npz")
        d, e = z["recovery_d_grid"], z["recovery_e2"]
        sym = np.abs(e - e[::-1])
        full = float(sym.max())
        assert full == pytest.approx(g["stage2_sym_err_max"], abs=1e-12)           # the runner's value: the WHOLE grid
        # the column is the maximum over the nodes with |d| < 2q ONLY (strict), written out here from the freeze arrays
        inside = np.abs(d) < 2 * Q
        want = float(sym[inside].max())
        assert r.sym_err_max == pytest.approx(want, abs=1e-12) and want <= full
        assert r.sym_err_argmax_abs_d == abs(float(d[int(np.argmax(np.where(inside, sym, -np.inf)))]))
        assert r.sym_err_max_rel == pytest.approx(want / r.g2_at_0, abs=1e-12)
        # the window pinned on these very arrays: an asymmetry planted at |d| = 2q (excluded: the window is strict) or in the tail
        # leaves the column unchanged, one at |d| = 2q - 0.5 (the last node inside) is the new maximum
        for tail_d in (100.0, 100.5, 150.0, -200.0):
            e_tail = e.copy()
            e_tail[int(np.argmin(np.abs(d - tail_d)))] += 7.0 + full
            m = R3.tie_metrics(d, e_tail, r.g2_at_0, r.gap, Q)
            assert m["sym_err_max"] == pytest.approx(want, abs=1e-12), (r.arm, tail_d)
        e_in = e.copy()
        e_in[int(np.argmin(np.abs(d - 99.5)))] += 7.0 + full
        m = R3.tie_metrics(d, e_in, r.g2_at_0, r.gap, Q)
        assert m["sym_err_max"] > 7.0 and m["sym_err_argmax_abs_d"] == 99.5
        assert r.tent_slope == pytest.approx(r.g2_at_0 / (2 * Q)) and r.w_eff == pytest.approx(r.gap / r.tent_slope)
        assert r.t2_R0_over_peak_final == pytest.approx(r.t2_R0_final / r.stage2_peak_rel_err_abs)
        assert r.t2_peak_over_R0_final * r.t2_R0_over_peak_final == pytest.approx(1.0)
        k = PROTO["records"]["50"]["game"]["k"]
        assert r.linearised_R0_over_peak == pytest.approx(2 * k / (2 * k + 4.0 / (4 * Q * Q)), abs=1e-12)
        assert r.peak_locfree_rel_err >= (r.e2_at_0 - r.g2_at_0) / r.g2_at_0 - 1e-12      # max >= the value at d = 0


def test_the_weight_columns_equal_the_exports(tiny_df):
    """w1d_* of the last terminal-stage export (update 60) against an independent read: the stored d-column, times ten for t10."""
    for r in tiny_df[tiny_df["role"] == "ms_arm"].itertuples():
        z = np.load(Path(r.run_dir) / "weights" / "u00060.npz")
        stored = np.abs(z["actor.l1.weight"][:, 1].astype(float))
        want_variant = r.actor
        assert (str(z["actor_variant"]) if "actor_variant" in z.files else "t1") == want_variant
        assert ("actor_variant" in z.files) == (r.actor != "t1")                       # a t1 export has no variant entry
        scale = 10.0 if r.actor == "t10" else 1.0
        assert r.w1d_abs_max == pytest.approx(scale * stored.max(), rel=1e-12) and r.w1d_export_update == 60
        assert r.w1d_abs_q50 == pytest.approx(scale * np.median(stored), rel=1e-12)
        assert r.w1d_n_gt1 == int((scale * stored > 1.0).sum())
        assert r.w1d_bend_d_min == pytest.approx(200.0 / (scale * stored.max()))


def test_first_layer_units_of_the_tiny_wave_are_the_exported_units(tiny_df):
    arms = R3.r3_arms(R3.ACTORS, ("bb",))
    fu = R3.first_layer_units_table(tiny_df, WIN, PROTO, arms)
    assert list(fu.columns) == R3.FU_COLS and len(fu) == 6 * 64 and (fu["local"] == 60).all()
    for r in tiny_df[tiny_df["role"] == "ms_arm"].itertuples():
        g = fu[fu["arm"] == r.arm]
        assert len(g) == 64 and set(g["variant"]) == {r.actor} and g["w_d_abs"].max() == pytest.approx(r.w1d_abs_max, rel=1e-12)
        z = np.load(Path(r.run_dir) / "weights" / "u00060.npz")
        scale = 10.0 if r.actor == "t10" else 1.0
        W, b = z["actor.l1.weight"].astype(float), z["actor.l1.bias"].astype(float)
        assert g["w_d"].to_numpy() == pytest.approx(scale * W[:, 1], rel=1e-12)
        ok = np.isfinite(g["center_d"].to_numpy())
        assert ok.sum() > 50
        c = g["center_d"].to_numpy()[ok]
        # the first-layer pre-activation of the actor's own input at (tau = 1, d = centre) is zero
        pre = W[ok, 0] * 1.0 + W[ok, 1] * (scale * c / 200.0) + b[ok]
        assert np.abs(pre).max() < 1e-9


def test_the_last_export_is_the_freeze_candidate(tiny_df):
    """The numpy forward of the last terminal-stage export gives e_hat_2(d) of the freeze at d = 0 AND at d != 0 (variant-aware
    reload). At d = 0 the x10 of the ``t10`` actor multiplies 0: a reload that ignores the variant is right there, and only
    the nodes off d = 0 can tell it."""
    from agents.ppo_curriculum import mean_effort_numpy
    d_nodes = np.array([-95.0, -40.0, -7.5, 0.0, 12.0, 40.0, 95.0])
    for r in tiny_df[tiny_df["role"] == "ms_arm"].itertuples():
        z = {k: v for k, v in np.load(Path(r.run_dir) / "weights" / "u00060.npz").items()}
        obs = np.array([[1.0, 0.0]], dtype=np.float32)                     # stage feature tau = 1, d = 0
        m, _, _ = mean_effort_numpy(z, obs)
        assert float(m[0]) == pytest.approx(r.e2_at_0, abs=1e-3), r.arm
        fz = np.load(Path(r.run_dir) / "freeze_stage2_final.npz")
        grid, e2 = fz["recovery_d_grid"], fz["recovery_e2"]
        j = np.array([int(np.argmin(np.abs(grid - x))) for x in d_nodes])
        assert np.allclose(grid[j], d_nodes)                               # the nodes are grid nodes: no interpolation
        obs = np.stack([np.ones_like(d_nodes), d_nodes / 200.0], 1).astype(np.float32)        # (tau = 1, d / B), B = 200 at q = 50
        m, _, _ = mean_effort_numpy(z, obs)
        assert np.abs(m - e2[j]).max() < 1e-3, (r.arm, np.abs(m - e2[j]).max())
        if r.actor != "t1":                       # the variant name is what makes it right: without it the tanh reading is off
            m_tanh, _, _ = mean_effort_numpy({k: v for k, v in z.items() if k != "actor_variant"}, obs)
            off = np.abs(m_tanh - e2[j])
            assert off[d_nodes != 0.0].max() > 1e-2, r.arm
            if r.actor == "t10":
                assert off[d_nodes == 0.0].max() < 1e-3                    # at d = 0 the two readings agree: the old test was blind


def test_first_layer_table_of_the_tiny_wave(tiny, tiny_df):
    fl = R3.first_layer_table(tiny_df, WIN, PROTO, R3.r3_arms(R3.ACTORS, ("bb",)))
    assert len(fl) == 6 * (12 + 4) and (fl["error"] == "").all()            # every export: 12 of the terminal stage + 4 of stage 1
    t2, t1 = fl[fl["stage"] == 2], fl[fl["stage"] == 1]
    assert len(t2) == 6 * 12 and len(t1) == 6 * 4 and set(fl["stage"]) == {1, 2}
    assert sorted(t2["local"].unique()) == list(range(5, 61, 5)) and (t2["update"] == t2["local"]).all()   # exports at 5, ..., 60
    assert sorted(t1["local"].unique()) == [5, 10, 15, 20] and (t1["update"] == t1["local"] + 60).all()     # stage 1 follows
    s2 = R3.first_layer_summary_table(fl, R3.r3_arms(R3.ACTORS, ("bb",)), (Q,))
    assert len(s2) == 6 * 12 and s2["update"].max() == 60                    # the summary is the terminal stage only
    assert set(fl["variant"]) == {"t1", "relu", "t10"} and (fl.groupby("arm")["variant"].nunique() == 1).all()
    assert (fl["B"] == 200.0).all()
    s = R3.first_layer_summary_table(fl, R3.r3_arms(R3.ACTORS, ("bb",)), (Q,))
    assert len(s) == 6 * 12 and (s["n_runs"] == 1).all()
    a = s[(s["arm"] == "t10_bb_s16") & (s["local"] == 60)].iloc[0]
    assert a["w_abs_max_median"] == pytest.approx(fl[(fl["arm"] == "t10_bb_s16") & (fl["local"] == 60)]["w_abs_max"].iloc[0])


def test_trajectory_table_with_reduced_windows(tiny_df):
    arms = R3.r3_arms(R3.ACTORS, ("bb",))
    t = R3.trajectory_table(tiny_df, WIN, arms)
    assert len(t) == 6 * 5                                      # K = 10: checks at local 20, 30, 40, 50, 60 per run
    assert t["local"].min() == 20 and t["local"].max() == 60 and set(t["status"]) == {"done"}
    assert list(t.columns) == R3.TRAJ_COLS
    for arm in arms:
        s = R3.parse_arm(arm)[2]
        g = t[t["arm"] == arm].set_index("local")
        assert g.loc[20, "conc_scale"] == 1.0 and g.loc[60, "conc_scale"] == pytest.approx(s)
        assert g.loc[30, "conc_scale"] == pytest.approx(1.0 + (s - 1.0) * (30 - 21) / (40 - 21))      # the D3 schedule
        row = g.loc[60]
        assert row["gap"] == pytest.approx(row["smoothing"] + row["remainder"], abs=1e-9)
        assert row["w_eff"] == pytest.approx(row["gap"] / (row["g2_0"] / (2 * Q)), abs=1e-12)           # w_eff at every check
        assert g["w_eff"].notna().all() and g["R0"].notna().all() and {"R", "Delta", "C"} <= set(g.columns)
        assert (g["actor"] == R3.parse_arm(arm)[0]).all() and (g["arm"] == arm).all()
    r = tiny_df[tiny_df["arm"] == "relu_bb_s16"].iloc[0]
    assert t[(t["arm"] == "relu_bb_s16") & (t["local"] == 60)]["e2_at_0"].iloc[0] == pytest.approx(r["e2_at_0"], abs=1e-9)
    assert len(R3.trajectory_table(tiny_df, R2.Windows(21, 40, 48, 60, 45), arms)) == 6 * 2
    tba = R3.trajectory_by_arm_table(t, arms, (Q,))
    assert len(tba) == 6 * 5 and (tba["n_runs"] == 1).all()
    a = tba[(tba["arm"] == "t10_bb_s1") & (tba["local"] == 40)].iloc[0]
    one = t[(t["arm"] == "t10_bb_s1") & (t["local"] == 40)].iloc[0]
    assert a["e2_at_0_mean"] == one["e2_at_0"] and a["w_eff_mean"] == one["w_eff"] and math.isnan(a["gap_sd"])
    assert a["R0_min"] == a["R0_max"] == one["R0"]


def test_segment_and_gates_tables_reuse_the_ms_r2_definitions_on_the_new_arms(tiny_df):
    arms = R3.r3_arms(R3.ACTORS, ("bb",))
    sg = R2.segment_table(tiny_df, WIN, arms, (Q,))
    assert len(sg) == 6 * 4 and list(sg["segment"].iloc[:4]) == ["training", "ramp", "hold", "decay"]
    a = sg[sg["arm"] == "relu_bb_s16"].set_index("segment")
    assert a.loc["training", ["conc_scale_min", "conc_scale_max"]].tolist() == [1.0, 1.0]
    assert a.loc["hold", ["conc_scale_min", "conc_scale_max"]].tolist() == [16.0, 16.0]
    g = R2.gates_table(tiny_df, arms + list(R3.MS_R2_ARMS), (Q,)).set_index(["arm", "q"])
    assert g.loc[("t10_bb_s1", Q), "n_done"] == 1 and g.loc[("NL_bb_s1", Q), "n_done"] == 0
    assert g.loc[("t10_bb_s1", Q), "n_G_A_and_G_N_eta"] in (0, 1) and pd.isna(g.loc[("t10_bb_s1", Q), "n_G_F"]) is False


def test_tie_profile_and_the_strata_tables_cover_every_stratum_and_the_tail_max(tiny_df):
    arms = R3.r3_arms(R3.ACTORS, ("bb",))
    tp = R3.tie_profile_table(tiny_df, arms, (Q,))
    assert len(tp) == 6 * 121 and (tp["n_runs"] == 1).all() and tp["d"].min() == -30.0 and tp["d"].max() == 30.0
    a = tp[(tp["arm"] == "relu_bb_s1") & (tp["d"] == 0.0)].iloc[0]
    r = tiny_df[tiny_df["arm"] == "relu_bb_s1"].iloc[0]
    assert a["e_hat_mean"] == pytest.approx(r["e2_at_0"], abs=1e-9) and a["g2"] == pytest.approx(r["g2_at_0"])
    assert a["e_hat_min"] == a["e_hat_max"] == a["e_hat_mean"]
    strata = R1.strata_table(tiny_df)
    ms = strata[strata["arm"].isin(arms)]
    assert set(ms["stratum"]) == {"near", "mid", "tail"} and set(ms["side"]) == {"d<0", "d>0"}
    assert set(ms["tier"]) == {"final", "development"}
    assert len(ms) == 6 * 2 * 6 and ms[ms["stratum"].isin(["mid", "tail"])]["err_rmse"].notna().all()
    # tail max: the larger of the two tail sides equals stage2_tail_max_over_g2_0 * e2*(0) of the run (the error is e_hat - 0)
    for run in tiny_df[tiny_df["arm"].isin(arms)].itertuples():
        t = ms[(ms["arm"] == run.arm) & (ms["tier"] == "final") & (ms["stratum"] == "tail")]
        assert t["err_max_abs"].max() == pytest.approx(run.stage2_tail_max_over_g2_0 * run.g2_at_0, abs=1e-9)
    summ = R1.strata_summary_table(strata)
    assert {"mid", "tail", "near"} <= set(summ["stratum"]) and "err_max_abs_rel_max" in summ.columns


def test_r0_tables_report_r0_over_peak_next_to_the_linearised_value(tiny_df):
    arms = R3.r3_arms(R3.ACTORS, ("bb",))
    r0 = R3.r0_table(tiny_df, arms, (Q,), PROTO).set_index(["arm", "q"])
    row = r0.loc[("t1_bb_s1", Q)]
    one = tiny_df[tiny_df["arm"] == "t1_bb_s1"].iloc[0]
    assert row["t2_R0_final_mean"] == one["t2_R0_final"] and row["t2_R0_over_peak_final_median"] == one["t2_R0_over_peak_final"]
    assert row["linearised_R0_over_peak"] == pytest.approx(1.0 / row["linearised_factor"])
    assert row["calib_median_peak_over_R0"] == 1.65
    traj = R3.trajectory_table(tiny_df, WIN, arms)
    sp = R3.r0_spearman_table(tiny_df, traj, R3.ACTORS, (Q,))
    assert set(sp["scope"]) == {"freeze", "checks"} and set(sp["actors"]) == {"all"} | set(R3.ACTORS)
    ck = sp[(sp["scope"] == "checks") & (sp["actors"] == "all")].iloc[0]
    assert ck["n"] == 6 * 5 and -1.0 <= ck["spearman_R0_vs_abs_peak"] <= 1.0
    fz = sp[(sp["scope"] == "freeze") & (sp["actors"] == "all")].iloc[0]
    assert fz["n"] == 6 and -1.0 <= fz["spearman_R0_vs_abs_peak"] <= 1.0
    assert sp[(sp["scope"] == "freeze") & (sp["actors"] == "t1")].iloc[0]["n"] == 2      # too few runs for a rank correlation
    assert math.isnan(sp[(sp["scope"] == "freeze") & (sp["actors"] == "t1")].iloc[0]["spearman_R0_vs_abs_peak"])


def test_the_ms_r2_context_table_is_zero_for_a_bit_identical_rerun(tiny, tmp_path):
    """The MS-R2 reference rows are read from ``--ms-r2-pilot-root`` under their NL names; a copy of a t1 run stands in for its
    bit-identical MS-R2 re-run and every paired difference is exactly 0 (the metric-level context of check C-MS5)."""
    roots, _ = tiny
    ref = tmp_path / "ms_r2"
    for arm, nl in (("t1_bb_s1", "NL_bb_s1"), ("t1_bb_s16", "NL_bb_s16")):
        shutil.copytree(Path(roots["pilot"]) / ("q%d" % Q) / ("seed%d" % SEED) / arm,
                        ref / ("q%d" % Q) / ("seed%d" % SEED) / nl)
    df = R3.extract_all(dict(roots, ms_r2_pilot=str(ref)), (Q,), (SEED,), TH, PROTO, ("t1",), ("bb",), (1, 16))
    nl_rows = df[df["arm"].isin(["NL_bb_s1", "NL_bb_s16"])]
    assert (nl_rows["status"] == "done").all() and (nl_rows["role"] == "comparator").all()
    assert set(nl_rows["source_root"]) == {"ms_r2_pilot"} and (nl_rows["actor"] == "").all()
    comps = R3.ms_r2_comparisons(("t1",), ("bb",), (1, 16))
    assert comps == [("t1_bb_s1", "NL_bb_s1"), ("t1_bb_s16", "NL_bb_s16")]
    p, _ = R3.paired_tables(df, comps, R3.CONTEXT_METRICS, (Q,), (SEED,), R3.LABEL_MS_R2)
    got = p[p["n_pairs"] == 1]
    assert len(got) > 40 and set(got["arm"]) == {"t1_bb_s1", "t1_bb_s16"}
    assert not any("wall" in m for m in p["metric"])
    assert {R3.PRIMARY, "gap", "t2_R0_final", "stage1_rel_err_abs"} <= set(p["metric"])
    for r in got.itertuples():
        assert r.mean == 0.0 and r.ci_mean_lo == 0.0 and r.ci_mean_hi == 0.0 and r.n_zero == 1, (r.arm, r.metric)


def _args(roots: Dict[str, str], out: Path, *extra: str) -> List[str]:
    return ["--pilot-root", roots["pilot"], "--ms-r2-pilot-root", roots["ms_r2_pilot"],
            "--parents-root", roots["parents"], "--rehearsal-root", roots["rehearsal"], "--out", str(out),
            "--qs", str(Q), "--seeds", str(SEED), "--ramp-first", "21", "--ramp-last", "40", "--hold-last", "48",
            "--decay-last", "60", "--traj-from", "20", *extra]


NAMES = ["per_run", "criterion", "paired_secondary", "paired_seed_level", "noise_landing", "transmission",
         "transmission_seed_level", "interaction", "interaction_seed_level", "starts_effect", "paired_vs_parents_A",
         "criterion_vs_parents_A", "paired_vs_rehearsal_v2_0", "paired_vs_ms_r2_nl", "stage1", "stage1_vs_t1",
         "stage1_R1_runs", "stage1_R1", "gates", "budget", "segments", "trajectory_checks", "trajectory_by_arm",
         "quadrature_check", "predictions", "freeze_decomposition", "strata", "strata_summary", "r0", "r0_spearman",
         "first_layer_weights", "first_layer_summary", "first_layer_units", "tie_profile", "arm_summary", "completeness"]


def _cli(roots: Dict[str, str], out: Path, *extra: str) -> Tuple[int, str]:
    """``R3.main`` with its stderr captured: ``(exit code, stderr text)``."""
    import contextlib
    import io
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        code = R3.main(_args(roots, out, *extra))
    return code, err.getvalue()


@pytest.fixture(scope="module")
def cli_full(tiny_refs, tmp_path_factory):
    """The CLI on the tiny wave with figures: the six bin-balanced arms ran, the six stratified ones did not; ``parents_A``
    and ``rehearsal_v2_0`` are genuine runs (:func:`write_reference_runs`), the MS-R2 references are absent."""
    roots = tiny_refs
    out = tmp_path_factory.mktemp("cli_full") / "analysis"
    code, err = _cli(roots, out)
    return out, code, err


@pytest.fixture(scope="module")
def cli_nofig(tiny_refs, tmp_path_factory):
    roots = tiny_refs
    out = tmp_path_factory.mktemp("cli_nofig") / "analysis"
    code, _ = _cli(roots, out, "--no-figures")
    return out, code


def test_cli_end_to_end_on_the_tiny_wave_the_missing_planned_runs_are_an_error(tiny_refs, cli_full):
    roots = tiny_refs
    out, code, err = cli_full
    assert code == 3                                           # missing planned runs and the four MS-R2 rows: reported
    assert "[incomplete]" in err
    summ = (out / "summary.txt").read_text()
    for head in ("PRE-REGISTERED CRITERION", "TRANSMISSION", "INTERACTION", "QUADRATURE CHECK", "RUN STATUS"):
        assert head in summ
    assert "relu_st_s16 q=50: planned 1, done 0" in summ and "t1_bb_s1" in summ
    assert "NL_bb_s1 q=50: planned 1, done 0" in summ and "NL_st_s16 q=50: planned 1, done 0" in summ
    assert "parents_A q=50" not in summ and "rehearsal_v2_0 q=50" not in summ          # both genuine references are done
    assert "actors: t1 relu t10" in summ
    for n in NAMES:
        t = pd.read_csv(out / ("%s.csv" % n))
        assert t.columns[0] == "roots_id" and t["roots_id"].nunique() <= 1, n
    info = json.loads((out / "analysis_info.json").read_text())
    assert info["boot_seed"] == 20261008 and info["n_boot"] == 10000 and info["all_done"] is False
    assert info["actors"] == ["t1", "relu", "t10"] and len(info["arms"]) == 12 and info["windows"]["ramp_last"] == 40
    assert info["protocol_sha256"] and info["roots"]["pilot"] == roots["pilot"]
    assert info["roots"]["ms_r2_pilot"] == roots["ms_r2_pilot"] and "NL_*" in info["not_stored_by_design"]
    assert info["roots"]["parents"] == roots["parents"] and info["roots"]["rehearsal"] == roots["rehearsal"]
    assert set(info["roots"]) == {"pilot", "ms_r2_pilot", "parents", "rehearsal"}
    assert set(info["figures"].values()) == {"ok"} and len(info["figures"]) == 9
    assert "s16 - s1" in info["transmission"] and "superset" in info["first_layer_weights"]["rows"]
    import matplotlib.image as mi
    for f in info["figures"]:
        assert (out / "figures" / f).stat().st_size > 3000
        assert mi.imread(str(out / "figures" / f)).shape[1] == 1000                    # legible at about 1000 px width
    comp = pd.read_csv(out / "completeness.csv").set_index(["arm", "q"])
    assert comp.loc[("relu_bb_s16", Q), "n_done"] == 1 and comp.loc[("relu_st_s16", Q), "n_missing"] == 1
    assert comp.loc[("parents_A", Q), "n_done"] == 1 and comp.loc[("rehearsal_v2_0", Q), "n_done"] == 1
    assert comp.loc[("NL_st_s16", Q), "n_missing"] == 1 and comp.loc[("NL_bb_s1", Q), "n_missing"] == 1
    crit = pd.read_csv(out / "criterion.csv")
    assert list(crit["arm"]) == [a for a, _ in R3.primary_comparisons(R3.ACTORS)]
    c = crit.set_index("arm")
    bb, st = ["relu_bb_s1", "relu_bb_s16", "t10_bb_s1", "t10_bb_s16"], ["relu_st_s1", "relu_st_s16", "t10_st_s1", "t10_st_s16"]
    assert (c.loc[bb, "n_pairs_q50"] == 1).all() and (c.loc[st, "n_pairs_q50"] == 0).all()
    assert (c.loc[st, "overall"] == "incomplete").all() and c.loc[bb, "overall"].isin(["met", "not met"]).all()
    per = pd.read_csv(out / "per_run.csv")
    assert len(per) == 12 + 4 * 1 + 2 and (per[per["role"] == "comparator"].set_index("arm").loc[list(R3.MS_R2_ARMS), "status"]
                                           == "missing").all()
    assert (per[per["arm"].isin(["parents_A", "rehearsal_v2_0"])]["status"] == "done").all()
    assert (per[per["arm"].isin(["relu_st_s1", "t10_st_s16"])]["status"] == "missing").all()
    tr = pd.read_csv(out / "transmission.csv")
    assert (tr[tr["starts"] == "bb"]["n_pairs"] == 1).all() and (tr[tr["starts"] == "st"]["n_pairs"] == 0).all()
    nl = pd.read_csv(out / "noise_landing.csv")                                  # the paired changes and the transmission rows
    txr = nl[nl["metric"] == "transmission_ratio"].set_index(["arm", "q"])
    assert len(txr) == 6 and set(nl["metric"]) >= {"gap", "smoothing", "remainder", R3.PRIMARY, "w_eff", "transmission_ratio"}
    same = lambda a, b: (math.isnan(a) and math.isnan(b)) or a == b       # noqa: E731
    for (arm, qq), r in tr.set_index(["arm", "q"]).iterrows():
        row = txr.loc[(arm, qq)]
        assert same(row["mean"], r["ratio"]) and same(row["ci_mean_lo"], r["ci_lo"]) and same(row["ci_mean_hi"], r["ci_hi"])
        assert row["n_boot_valid"] == r["n_boot_valid"] and row["n_pairs"] == r["n_pairs"] and "s16 - s1" in row["note"]
        assert row["per_seed_ratios"] == r["per_seed_ratios"] or (pd.isna(row["per_seed_ratios"])
                                                                  and pd.isna(r["per_seed_ratios"]))
    assert set(nl[nl["metric"] == "gap"]["comparison"]) == {R3.LABEL_NL}
    q = pd.read_csv(out / "quadrature_check.csv")
    assert len(q) == 6 and q[q["starts"] == "st"]["closer"].isna().all()
    fl = pd.read_csv(out / "first_layer_weights.csv")
    assert len(fl) == 6 * (12 + 4) and set(fl["variant"]) == {"t1", "relu", "t10"} and set(fl["stage"]) == {1, 2}
    st1 = fl[fl["stage"] == 1]
    assert len(st1) == 6 * 4 and (st1["update"] == st1["local"] + 60).all()             # stage-1 exports follow update 60
    assert (per[per["arm"].isin(R3.r3_arms(R3.ACTORS, ("bb",)))]["w1d_export_update"] == 60).all()   # terminal stage's last
    assert len(pd.read_csv(out / "first_layer_units.csv")) == 6 * 64
    assert len(pd.read_csv(out / "first_layer_summary.csv")) == 6 * 12


def test_the_blind_script_agrees_on_the_real_partially_complete_output(cli_full):
    out, _, _ = cli_full
    assert BL.main(["--analysis-dir", str(out)]) == 0
    assert "ALL " in (out / "blind_recomputation.txt").read_text()


def test_cli_vs_parents_and_vs_rehearsal_rows_are_the_arm_minus_the_right_reference_run(tiny_refs, cli_full):
    """With a genuine ``parents_A`` run (|peak| 0.08, gap 5.6) and a genuine ``rehearsal_v2_0`` run (|peak| 0.03, gap 2.1,
    stage-1 error 0.045, e1(0) 44.55) on disk, every row of ``paired_vs_parents_A`` / ``criterion_vs_parents_A`` /
    ``paired_vs_rehearsal_v2_0`` is the arm's own value (read here from the arm's ``gates.json``) minus the value of the
    RIGHT reference. A swap of the two roots, of the two baselines or of the labels cannot give these numbers."""
    roots = tiny_refs
    out, _, _ = cli_full
    per = pd.read_csv(out / "per_run.csv", float_precision="round_trip").set_index("arm")
    for ref, (peak, gap, root_key) in (("parents_A", (0.08, 5.6, "parents")), ("rehearsal_v2_0", (0.03, 2.1, "rehearsal"))):
        row = per.loc[ref]
        assert row["status"] == "done" and row["source_root"] == root_key and row["role"] == "comparator"
        assert row["stage2_peak_rel_err_abs"] == pytest.approx(peak, abs=1e-12) and row["gap"] == pytest.approx(gap, abs=1e-12)
        assert Path(row["run_dir"]) == Path(roots[root_key]) / ("q%d" % Q) / ("seed%d" % SEED) and pd.isna(row["flags"])
    assert per.loc["rehearsal_v2_0", "stage1_rel_err_abs"] == pytest.approx(0.045) and pd.isna(per.loc["parents_A", "stage1_rel_err_abs"])
    vp = pd.read_csv(out / "paired_vs_parents_A.csv", float_precision="round_trip")
    vr = pd.read_csv(out / "paired_vs_rehearsal_v2_0.csv", float_precision="round_trip")
    cp = pd.read_csv(out / "criterion_vs_parents_A.csv", float_precision="round_trip").set_index("arm")
    arms = R3.r3_arms(R3.ACTORS, ("bb",))
    cell = lambda t, arm, m: t[(t["arm"] == arm) & (t["q"] == Q) & (t["metric"] == m)]      # noqa: E731
    for arm in arms:
        g = json.loads((Path(roots["pilot"]) / ("q%d" % Q) / ("seed%d" % SEED) / arm / "gates.json").read_text())["reported"]
        s2, s2dev, s1 = g["end_of_stage2"]["final"], g["end_of_stage2"]["development"], g["end_of_stage1"]["final"]
        for metric, got, ref_value in ((R3.PRIMARY, s2["stage2_peak_rel_err_abs"], 0.08),
                                       ("gap", s2["g2_at_0"] - s2["e2_at_0"], 5.6), ("e2_at_0", s2["e2_at_0"], 64.4),
                                       ("stage2_rmse_pos_over_g2_0", s2["stage2_rmse_pos_over_g2_0"], 0.02)):
            r = cell(vp, arm, metric)
            assert len(r) == 1, (arm, metric)
            r = r.iloc[0]
            assert r["baseline"] == "parents_A" and r["comparison"] == R3.LABEL_PARENTS and r["n_pairs"] == 1
            assert r["mean"] == pytest.approx(got - ref_value, abs=1e-12) and r["median"] == r["mean"], (arm, metric)
            assert r["ci_mean_lo"] == r["mean"] == r["ci_mean_hi"]                       # one pair: the interval is the value
        for metric, got, ref_value in (("stage1_rel_err_abs", s1["stage1_rel_err_abs"], 0.045),
                                       ("stage1_rel_err_signed", s1["stage1_rel_err_signed"], 0.045),
                                       ("e1_at_0", s1["e1_at_0"], 44.55)):
            r = cell(vr, arm, metric)
            assert len(r) == 1, (arm, metric)
            r = r.iloc[0]
            assert r["baseline"] == "rehearsal_v2_0" and r["comparison"] == R3.LABEL_REHEARSAL and r["n_pairs"] == 1
            assert r["mean"] == pytest.approx(got - ref_value, abs=1e-12), (arm, metric)
        # the criterion against parents_A: (a) from the one q = 50 pair, (b) from the baseline gate and the arm's own gate
        c = cp.loc[arm]
        eta, rmse, tail = s2["eta_T_over_dw"], s2["stage2_rmse_pos_over_g2_0"], s2["stage2_tail_mean_over_g2_0"]
        arm_pass = eta <= TH.eta and rmse <= TH.rmse and tail <= TH.tail and abs(s2dev["eta_T_over_dw"] - eta) <= TH.n_eta
        assert c["baseline"] == "parents_A" and c["n_pairs_q50"] == 1 and "n_pairs_q60" not in c.index      # --qs 50
        assert c["mean_q50"] == pytest.approx(s2["stage2_peak_rel_err_abs"] - 0.08, abs=1e-12)
        assert bool(c["a_q50"]) == (s2["stage2_peak_rel_err_abs"] < 0.08)
        assert c["n_base_pass"] == 1                                                      # the parents_A run passes the gate
        assert (c["b_violations"] if isinstance(c["b_violations"], str) else "") == ("" if arm_pass else "q50/10501")
        assert c["overall"] == ("met" if arm_pass and s2["stage2_peak_rel_err_abs"] < 0.08 else "not met")
    # each reference has only its own metrics: no stage-1 metric against parents_A, no stage-2 |peak| against rehearsal_v2_0
    assert R3.PRIMARY in set(vp["metric"]) and "stage1_rel_err_abs" not in set(vp["metric"])
    assert "stage1_rel_err_abs" in set(vr["metric"]) and R3.PRIMARY not in set(vr["metric"])
    assert set(vp["baseline"]) == {"parents_A"} and set(vr["baseline"]) == {"rehearsal_v2_0"}


def test_cli_is_deterministic_and_refuses_to_overwrite(tiny, cli_full, cli_nofig):
    roots, _ = tiny
    out1, _, _ = cli_full
    out2, code2 = cli_nofig
    assert code2 == 3 and not (out2 / "figures").exists()
    for n in NAMES:
        assert (out1 / ("%s.csv" % n)).read_text() == (out2 / ("%s.csv" % n)).read_text(), n
    before = (out2 / "per_run.csv").read_bytes()
    code, err = _cli(roots, out2, "--no-figures")
    assert code == 2 and "refusing to overwrite" in err                                # an existing analysis is not overwritten
    assert (out2 / "per_run.csv").read_bytes() == before
    with pytest.raises(FileExistsError):
        R3.run_analysis(roots, out2, (Q,), (SEED,), R3.ACTORS, PROTO_PATH, WIN, False)


def test_a_reduced_actor_set_derives_the_arms_from_the_actors_argument(tiny, tmp_path):
    roots, _ = tiny
    out = tmp_path / "r"
    out.mkdir()                                                                        # an existing empty directory is fine
    tables, all_done = R3.run_analysis(roots, out, (Q,), (SEED,), ("t1", "relu"), PROTO_PATH, WIN, False, None,
                                       starts=("bb",), scales=(1, 16))
    assert not all_done                                          # the references are missing; the four arms are all done
    assert list(tables["criterion"]["arm"]) == ["relu_bb_s1", "relu_bb_s16"] and (tables["criterion"]["n_pairs_q50"] == 1).all()
    assert set(tables["per_run"]["arm"]) == {"t1_bb_s1", "t1_bb_s16", "relu_bb_s1", "relu_bb_s16"} | set(R3.REF_ARMS)
    assert not tables["per_run"]["arm"].str.startswith("t10").any()                           # the t10 runs on disk are ignored
    assert set(tables["interaction"]["actor"]) == {"relu"} and len(tables["transmission"]) == 2
    comp = tables["completeness"]
    assert (comp[comp["arm"].isin(["t1_bb_s1", "t1_bb_s16", "relu_bb_s1", "relu_bb_s16"])]["n_done"] == 1).all()
    assert len(tables["starts_effect"]) == 0                    # one sampler only: no starts comparison
    info = json.loads((out / "analysis_info.json").read_text())
    assert info["actors"] == ["t1", "relu"] and info["arms"] == ["t1_bb_s1", "t1_bb_s16", "relu_bb_s1", "relu_bb_s16"]
    assert BL.main(["--analysis-dir", str(out)]) == 0


def test_cli_accepts_the_actors_as_a_comma_list_like_the_launcher(tiny, tmp_path):
    roots, _ = tiny
    code, _ = _cli(roots, tmp_path / "c", "--actors", "t1,t10", "--no-figures")
    assert code == 3
    info = json.loads((tmp_path / "c" / "analysis_info.json").read_text())
    assert info["actors"] == ["t1", "t10"] and len(info["arms"]) == 8 and all(a.startswith(("t1_", "t10_")) for a in info["arms"])
    crit = pd.read_csv(tmp_path / "c" / "criterion.csv")
    assert set(crit["actor"]) == {"t10"} and len(crit) == 4


def test_a_planned_run_that_is_not_on_disk_is_a_missing_row_and_exit_3(tiny, tmp_path):
    roots, root = tiny
    pilot = tmp_path / "pilot_no_t10"
    for d in (root / "pilot").glob("q*/seed*/*"):
        if not d.name.startswith("t10"):
            (pilot / d.parent.relative_to(root / "pilot")).mkdir(parents=True, exist_ok=True)
            (pilot / d.relative_to(root / "pilot")).symlink_to(d)
    code, err = _cli(dict(roots, pilot=str(pilot)), tmp_path / "o", "--no-figures")
    assert code == 3 and "[incomplete]" in err
    per = pd.read_csv(tmp_path / "o" / "per_run.csv")
    t10 = per[per["arm"].str.startswith("t10")]
    assert (t10["status"] == "missing").all() and len(t10) == 4
    crit = pd.read_csv(tmp_path / "o" / "criterion.csv")
    assert (crit[crit["actor"] == "t10"]["overall"] == "incomplete").all()
    assert (crit[crit["actor"] == "t10"]["n_pairs_q50"] == 0).all()


def test_cli_argument_errors_exit_2(tiny, tmp_path):
    roots, _ = tiny
    out = tmp_path / "o"
    assert _cli(roots, out, "--ramp-first", "50", "--ramp-last", "40")[0] == 2                # unordered windows
    assert _cli(roots, out, "--actors", "relu", "t10")[0] == 2                                # the control is missing
    assert _cli(roots, out, "--actors", "t1", "gelu")[0] == 2                                 # an unknown actor
    assert _cli(roots, out, "--actors", "t1,,relu")[0] == 2                                   # an empty name in a comma list
    assert _cli(dict(roots, parents=str(tmp_path / "does_not_exist")), out)[0] == 2           # a missing reference root
    assert _cli(dict(roots, ms_r2_pilot=str(tmp_path / "does_not_exist")), out)[0] == 2
    assert _cli(dict(roots, pilot=str(tmp_path / "does_not_exist")), out)[0] == 2
    assert not out.exists()                                                                   # nothing was written
    base = _args(roots, out)
    with pytest.raises(SystemExit) as e:
        R3.main(base[:-2] + ["--no-such-flag"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        R3.main(["--pilot-root", roots["pilot"]])                                            # required roots are not given
    assert e.value.code == 2
