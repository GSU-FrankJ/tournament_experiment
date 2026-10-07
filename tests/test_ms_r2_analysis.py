"""MS-R2 analysis tool tests (``tools/ms/r2_analysis.py`` and the independent ``tools/ms/r2_blind_criterion.py``).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r2_analysis.py -p no:cacheprovider -q

Sections: (1) synthetic per-run frames: the criterion parts (a) and (b) by hand (a violation, a missing run, a baseline that
does not pass), the fresh generator per (q, statistic), the paired tables; (2) the transmission ratio and the
interaction on constructed data; (3) the decomposition columns against the formulas and against a noise record; (4) the
blind recomputation: agreement, tampered criterion / interaction / per_run, unreadable inputs, independence from the
analysis tool; (5) one real tiny wave (six reduced NL runs of one (q, seed)): extraction, the NL decomposition
cross-check, the trajectory and segment tables with reduced windows, the CLI end to end.
"""

from __future__ import annotations

import ast
import json
import math
import os
import shutil
import sys
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
import r2_blind_criterion as BL  # noqa: E402

PROTO_PATH = ROOT / "protocols" / "v2_T2_locked_v2_0.json"
TH, _ = R1.load_thresholds(PROTO_PATH)
SEEDS4 = (10501, 10502, 10503, 10504)
QS = (50, 60)
BOOT = 20261007


# ====================================================================== synthetic per-run frames
def indep_ci(d: Sequence[float], stat: str = "mean") -> Tuple[float, float]:
    """The pre-registered interval written out independently: fresh generator, indices (10000, n), 2.5 / 97.5."""
    a = np.asarray(d, dtype=float)
    rng = np.random.default_rng(BOOT)
    idx = rng.integers(0, a.size, size=(10000, a.size))
    r = a[idx]
    s = r.mean(axis=1) if stat == "mean" else np.median(r, axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def blank(arm: str, q: int, seed: int, **kw: Any) -> Dict[str, Any]:
    """A done per-run row of every column (NaN / empty) with a passing G-A / G-N eta and the given overrides."""
    row: Dict[str, Any] = {c: ("" if c in R2.STR_COLS else float("nan")) for c in R2.COLUMNS}
    row.update(arm=arm, q=q, seed=seed, role="ms_arm" if arm.startswith("NL_") else "comparator", status="done",
               status_info="", complete=True, run_dir="", flags="", eta_T_over_dw=0.003, eta_dev=0.0031,
               stage2_rmse_pos_over_g2_0=0.02, stage2_tail_mean_over_g2_0=0.005, g2_at_0=70.0 if q == 50 else 58.0)
    row.update(kw)
    sampler, s = R2.parse_arm(arm)
    row["sampler"], row["s"] = sampler, s
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


BASE_PEAK = {("bb", 50): [0.10, 0.12, 0.09, 0.11], ("bb", 60): [0.08, 0.09, 0.07, 0.10],
             ("st", 50): [0.07, 0.08, 0.06, 0.09], ("st", 60): [0.06, 0.07, 0.05, 0.08]}
DELTA = {("NL_bb_s4", 50): [-0.03, -0.04, -0.02, -0.05], ("NL_bb_s4", 60): [-0.02, -0.03, -0.01, -0.02],
         ("NL_bb_s16", 50): [-0.05, -0.05, -0.04, -0.05], ("NL_bb_s16", 60): [-0.02, 0.03, -0.01, 0.05],
         ("NL_st_s4", 50): [-0.02, -0.01, -0.03, -0.02], ("NL_st_s4", 60): [-0.01, -0.02, -0.01, -0.03],
         ("NL_st_s16", 50): [-0.03, -0.03, -0.02, -0.03], ("NL_st_s16", 60): [-0.02, -0.03, -0.01, -0.04]}


def make_world() -> pd.DataFrame:
    """Six NL arms x q x four seeds (hand-designed):

    * NL_bb_s1 at q=50 seed 10502 does not pass G-A (eta 0.01): it is not a baseline pass, and NL_bb_s4 failing there is
      no violation;
    * NL_bb_s4: every paired difference negative at both q, b holds -> met;
    * NL_bb_s16: q=60 differences [-0.02, +0.03, -0.01, +0.05], mean +0.0125 -> part (a) not met;
    * NL_st_s4: part (a) met, but its run at (q=50, 10503) fails the tail limit (0.03 > 0.02) -> violated, not met;
    * NL_st_s16: the run at (q=60, 10504) is missing -> 3 pairs at q=60, b pending -> incomplete.
    """
    rows = []
    for k in ("bb", "st"):
        for q in QS:
            for i, sd in enumerate(SEEDS4):
                base = BASE_PEAK[(k, q)][i]
                kw1: Dict[str, Any] = {}
                if (k, q, sd) == ("bb", 50, 10502):
                    kw1 = {"eta_T_over_dw": 0.01, "eta_dev": 0.0101}
                rows.append(settle(blank("NL_%s_s1" % k, q, sd, stage2_peak_rel_err_abs=base, **kw1)))
                for s in (4, 16):
                    arm = "NL_%s_s%d" % (k, s)
                    kw: Dict[str, Any] = dict(kw1)
                    if (arm, q, sd) == ("NL_st_s4", 50, 10503):
                        kw["stage2_tail_mean_over_g2_0"] = 0.03
                    r = blank(arm, q, sd, stage2_peak_rel_err_abs=base + DELTA[(arm, q)][i], **kw)
                    if (arm, q, sd) == ("NL_st_s16", 60, 10504):
                        r = blank(arm, q, sd, status="missing", complete=False)
                    rows.append(settle(r))
    df = pd.DataFrame(rows, columns=R2.COLUMNS)
    # decomposition columns (deterministic): smoothing falls with s, the remainder differs by sampler
    for i, r in df.iterrows():
        if r["status"] != "done":
            continue
        sd_i = SEEDS4.index(int(r["seed"]))
        s = float(r["s"])
        sm = 2.0 / math.sqrt(s) + 0.01 * sd_i + (0.05 if r["q"] == 60 else 0.0)
        rem = 1.5 + 0.1 * sd_i + (0.3 * math.log2(s) if r["sampler"] == "bb" else 0.1 * math.log2(s))
        df.loc[i, ["smoothing", "remainder"]] = (sm, rem)
        df.loc[i, "gap"] = sm + rem
    return df


@pytest.fixture(scope="module")
def world() -> pd.DataFrame:
    return make_world()


def _crit(df: pd.DataFrame, comparisons: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
    comps = comparisons or R2.S1_COMPARISONS
    p, _ = R2.paired_tables(df, comps, [(R2.PRIMARY, True)], QS, SEEDS4, R2.LABEL_S1)
    return R2.criterion_table(df, p, comps, QS, SEEDS4, R2.LABEL_S1, R2.CRITERION_NOTE)


def test_constants_are_the_preregistered_scheme():
    assert R2.BOOT_SEED == 20261007 and R2.N_BOOT == 10000
    assert R2.S1_COMPARISONS == [("NL_bb_s4", "NL_bb_s1"), ("NL_bb_s16", "NL_bb_s1"), ("NL_st_s4", "NL_st_s1"),
                                 ("NL_st_s16", "NL_st_s1")]
    assert R2.NL_ARMS == ("NL_bb_s1", "NL_bb_s4", "NL_bb_s16", "NL_st_s1", "NL_st_s4", "NL_st_s16")
    assert R2.parse_arm("NL_st_s16") == ("st", 16.0) and R2.parse_arm("MS_base2400")[0] == ""


def test_criterion_parts_a_and_b_by_hand(world):
    crit = _crit(world).set_index("arm")
    assert list(crit.index) == ["NL_bb_s4", "NL_bb_s16", "NL_st_s4", "NL_st_s16"]
    assert (crit["baseline"] == ["NL_bb_s1", "NL_bb_s1", "NL_st_s1", "NL_st_s1"]).all()
    assert (crit["boot_seed"] == 20261007).all() and (crit["n_boot"] == 10000).all()
    # ---- NL_bb_s4: diffs q50 [-.03 -.04 -.02 -.05], q60 [-.02 -.03 -.01 -.02]
    r = crit.loc["NL_bb_s4"]
    assert r["n_pairs_q50"] == 4 and r["n_pairs_q60"] == 4
    assert r["mean_q50"] == pytest.approx(-0.035, abs=1e-12) and r["mean_q60"] == pytest.approx(-0.02, abs=1e-12)
    lo, hi = indep_ci([-0.03, -0.04, -0.02, -0.05])
    assert r["ci_mean_lo_q50"] == pytest.approx(lo, abs=1e-12) and r["ci_mean_hi_q50"] == pytest.approx(hi, abs=1e-12)
    assert hi < 0 and bool(r["a_q50"]) and bool(r["a_q60"]) and bool(r["a_met"]) and bool(r["a_complete"])
    assert r["b_status"] == "holds" and r["b_violations"] == "" and r["overall"] == "met"
    assert r["n_base_pass"] == 7                       # NL_bb_s1 at q=50 seed 10502 does not pass G-A
    # ---- NL_bb_s16: q60 diffs [-.02 +.03 -.01 +.05]: mean +0.0125, the interval reaches above 0 -> (a) not met
    r = crit.loc["NL_bb_s16"]
    assert r["mean_q60"] == pytest.approx(0.0125, abs=1e-12)
    lo, hi = indep_ci([-0.02, 0.03, -0.01, 0.05])
    assert hi > 0 and r["ci_mean_hi_q60"] == pytest.approx(hi, abs=1e-12)
    assert bool(r["a_q50"]) and not bool(r["a_q60"]) and not bool(r["a_met"]) and r["overall"] == "not met"
    # ---- NL_st_s4: (a) met at both q, but the run at q=50 seed 10503 passes under s=1 and fails the tail limit under s=4
    r = crit.loc["NL_st_s4"]
    assert bool(r["a_met"]) and r["b_violations"] == "q50/10503" and r["b_n_violations"] == 1
    assert r["b_status"] == "violated" and r["overall"] == "not met" and r["n_base_pass"] == 8
    # ---- NL_st_s16: the q=60 run of seed 10504 is missing: 3 pairs, (a) not complete, (b) pending -> incomplete
    r = crit.loc["NL_st_s16"]
    assert r["n_pairs_q60"] == 3 and r["n_pairs_q50"] == 4 and not bool(r["a_complete"])
    assert r["b_pending"] == "q60/10504" and r["b_n_pending"] == 1 and r["b_violations"] == ""
    assert r["b_status"] == "incomplete" and r["overall"] == "incomplete"
    lo, hi = indep_ci([-0.02, -0.03, -0.01])           # the three seeds that have a pair
    assert r["mean_q60"] == pytest.approx(-0.02, abs=1e-12) and r["ci_mean_hi_q60"] == pytest.approx(hi, abs=1e-12)


def test_the_criterion_is_strict_below_zero(world):
    """ci_mean_hi == 0 does not meet (a): all differences 0 -> interval [0, 0]."""
    df = world.copy()
    m = (df["arm"] == "NL_bb_s4")
    base = df[df["arm"] == "NL_bb_s1"].set_index(["q", "seed"])[R2.PRIMARY]
    df.loc[m, R2.PRIMARY] = [base.loc[(q, s)] for q, s in zip(df.loc[m, "q"], df.loc[m, "seed"])]
    r = _crit(df).set_index("arm").loc["NL_bb_s4"]
    assert r["ci_mean_hi_q50"] == 0.0 and not bool(r["a_q50"]) and r["overall"] == "not met"


def test_a_failed_arm_run_counts_as_violation_and_a_running_one_as_pending(world):
    df = world.copy()
    i = df.index[(df["arm"] == "NL_bb_s4") & (df["q"] == 60) & (df["seed"] == 10501)][0]
    df.loc[i, ["status", "complete"]] = ["failed", False]
    j = df.index[(df["arm"] == "NL_bb_s4") & (df["q"] == 60) & (df["seed"] == 10502)][0]
    df.loc[j, ["status", "complete"]] = ["running", False]
    r = _crit(df).set_index("arm").loc["NL_bb_s4"]
    assert r["b_violations"] == "q60/10501" and r["b_pending"] == "q60/10502" and r["n_pairs_q60"] == 2
    assert r["b_status"] == "violated" and r["overall"] == "not met"


def test_a_missing_baseline_run_never_enters_a_pair(world):
    df = world.copy()
    i = df.index[(df["arm"] == "NL_st_s1") & (df["q"] == 50) & (df["seed"] == 10501)][0]
    df.loc[i, ["status", "complete"]] = ["failed", False]
    r = _crit(df).set_index("arm")
    assert r.loc["NL_st_s4", "n_pairs_q50"] == 3 and r.loc["NL_st_s16", "n_pairs_q50"] == 3
    assert r.loc["NL_bb_s4", "n_pairs_q50"] == 4                   # the other sampler is untouched


def test_fresh_generator_per_q_and_statistic_makes_the_numbers_independent_of_table_order(world):
    """Every interval uses a fresh default_rng(20261007): the numbers do not depend on the order of the rows, of the
    comparisons, of the q values or on how many tables were built before."""
    ref = _crit(world).set_index("arm")
    rng = np.random.default_rng(3)
    shuffled = world.iloc[rng.permutation(len(world))].reset_index(drop=True)
    again = _crit(shuffled, list(reversed(R2.S1_COMPARISONS))).set_index("arm")
    for arm in ref.index:
        for c in ("mean_q50", "ci_mean_lo_q50", "ci_mean_hi_q50", "mean_q60", "ci_mean_lo_q60", "ci_mean_hi_q60"):
            assert ref.loc[arm, c] == again.loc[arm, c], (arm, c)
    # the same differences give the same interval wherever they sit (two arms, two q)
    d = [-0.03, -0.04, -0.02, -0.05]
    a = R2.paired_summary(d, True)
    R2.paired_summary([1.0, 2.0, 3.0, 4.0], True)                    # a call in between does not move the next one
    b = R2.paired_summary(d, True)
    assert a["ci_mean_lo"] == b["ci_mean_lo"] and a["ci_mean_hi"] == b["ci_mean_hi"]
    assert (a["ci_mean_lo"], a["ci_mean_hi"]) == indep_ci(d)
    assert (a["ci_median_lo"], a["ci_median_hi"]) == indep_ci(d, "median")
    # the same seed 20261007 with another n changes the draws, as it must (idx = integers(0, n, size=(10000, n)))
    assert R2.paired_summary([1.0, 2.0, 3.0], None)["ci_mean_lo"] == indep_ci([1.0, 2.0, 3.0])[0]


def test_paired_tables_report_status_and_sign_counts(world):
    p, long = R2.paired_tables(world, R2.S1_COMPARISONS, R2.SECONDARY_METRICS, QS, SEEDS4, R2.LABEL_S1)
    assert set(p["status"]) == {"descriptive"} and set(long["status"]) == {"descriptive"}
    r = p[(p["arm"] == "NL_bb_s16") & (p["q"] == 60) & (p["metric"] == R2.PRIMARY)].iloc[0]
    assert (r["n_pairs"], r["n_pos"], r["n_neg"], r["n_zero"], r["n_better"]) == (4, 2, 2, 0, 2)
    assert r["mean"] == pytest.approx(0.0125, abs=1e-12) and r["median"] == pytest.approx(0.01, abs=1e-12)
    pn = R2.paired_tables(world, R2.S1_COMPARISONS, [(R2.PRIMARY, True)], QS, SEEDS4, R2.LABEL_S1, "NOTE")[0]
    assert set(pn["status"]) == {"NOTE"}
    # the smoothing differences: s=4 minus s=1 of the bb arm = 2/2 - 2/1 = -1 for every pair
    r = p[(p["arm"] == "NL_bb_s4") & (p["q"] == 50) & (p["metric"] == "smoothing")].iloc[0]
    assert r["mean"] == pytest.approx(-1.0, abs=1e-12) and r["ci_mean_hi"] < 0
    assert len(long[(long["arm"] == "NL_bb_s4") & (long["q"] == 50) & (long["metric"] == "smoothing")]) == 4


# ====================================================================== transmission and interaction
def tx_frame(rows: Dict[Tuple[str, int], Tuple[float, float]], q: int = 50) -> pd.DataFrame:
    """Per-run frame with the given (smoothing, remainder) per (arm, seed); gap = smoothing + remainder."""
    out = []
    for (arm, sd), (sm, rem) in rows.items():
        out.append(blank(arm, q, sd, smoothing=sm, remainder=rem, gap=sm + rem))
    return pd.DataFrame(out, columns=R2.COLUMNS)


def test_transmission_ratio_on_constructed_data():
    seeds = (10501, 10502)
    rows: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for sd in seeds:
        rows[("NL_bb_s1", sd)] = (4.0, 6.0)          # gap 10
        rows[("NL_bb_s4", sd)] = (3.0, 6.0)          # smoothing -1, remainder 0 -> gap -1: ratio 1
        rows[("NL_st_s1", sd)] = (4.0, 6.0)
        rows[("NL_st_s4", sd)] = (3.0, 7.0)          # smoothing -1, remainder +1 -> gap 0: ratio 0 (remainder offsets)
        rows[("NL_bb_s16", sd)] = (3.0, 6.0)
        rows[("NL_st_s16", sd)] = (4.0, 5.0)         # smoothing 0 -> the denominator is 0: NaN, no exception
    t, long = R2.transmission_tables(tx_frame(rows), R2.S1_COMPARISONS, (50,), seeds)
    t = t.set_index("arm")
    assert t.loc["NL_bb_s4", "ratio"] == pytest.approx(1.0, abs=1e-12)
    assert t.loc["NL_bb_s4", "ci_lo"] == pytest.approx(1.0, abs=1e-12) and t.loc["NL_bb_s4", "ci_hi"] == pytest.approx(1.0, abs=1e-12)
    assert t.loc["NL_bb_s4", "mean_d_gap"] == pytest.approx(-1.0) and t.loc["NL_bb_s4", "mean_d_smoothing"] == pytest.approx(-1.0)
    assert t.loc["NL_st_s4", "ratio"] == pytest.approx(0.0, abs=1e-12) and t.loc["NL_st_s4", "ci_hi"] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(t.loc["NL_st_s16", "ratio"]) and math.isnan(t.loc["NL_st_s16", "ci_lo"])
    assert t.loc["NL_st_s16", "n_boot_valid"] == 0
    assert t.loc["NL_bb_s4", "per_seed_ratios"] == "10501:1;10502:1"
    assert "10501:nan" in t.loc["NL_st_s16", "per_seed_ratios"]
    assert set(long["arm"]) == {"NL_bb_s4", "NL_bb_s16", "NL_st_s4", "NL_st_s16"} and len(long) == 8


def test_transmission_interval_resamples_the_seeds_and_recomputes_the_ratio():
    """d gap [-1, -2], d smoothing [-1, -1]: ratio of means 1.5; the resampled means give ratios 1, 1.5, 2 with
    probabilities 1/4, 1/2, 1/4, so the 2.5 / 97.5 percentiles are 1 and 2 (computed by hand)."""
    r = R2.transmission_ratio([-1.0, -2.0], [-1.0, -1.0])
    assert r["ratio"] == pytest.approx(1.5) and r["ci_lo"] == pytest.approx(1.0) and r["ci_hi"] == pytest.approx(2.0)
    assert r["n_boot_valid"] == 10000
    # an independent resampling with the same fresh generator reproduces the numbers exactly
    g, s = np.array([-1.0, -2.0]), np.array([-1.0, -1.0])
    idx = np.random.default_rng(BOOT).integers(0, 2, size=(10000, 2))
    rat = g[idx].mean(axis=1) / s[idx].mean(axis=1)
    assert (r["ci_lo"], r["ci_hi"]) == (float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5)))
    # a denominator that straddles 0: the resamples with a ~0 mean denominator are dropped, never a division error
    r = R2.transmission_ratio([-1.0, -1.0, 2.0, 0.0], [1.0, -1.0, 1.0, -1.0])
    assert r["n_boot_valid"] < 10000 and np.isfinite(r["ci_lo"])
    assert math.isnan(R2.transmission_ratio([1.0, 1.0], [1.0, -1.0])["ratio"])        # mean denominator 0
    assert math.isnan(R2.transmission_ratio([], [])["ratio"])


def test_transmission_uses_a_fresh_generator_per_call_and_per_table_row():
    """Continuous data (the percentiles depend on the draws): two calls give the same interval, and it is the interval of
    a fresh default_rng(20261007); a table built in a different arm order gives the same numbers."""
    g = np.array([-1.1, -2.3, -0.4, -3.7, -1.9, -0.8])
    sm = np.array([-1.0, -1.2, -0.7, -1.5, -0.9, -1.1])
    a, b = R2.transmission_ratio(g, sm), R2.transmission_ratio(g, sm)
    assert (a["ci_lo"], a["ci_hi"]) == (b["ci_lo"], b["ci_hi"])
    idx = np.random.default_rng(BOOT).integers(0, 6, size=(10000, 6))
    rat = g[idx].mean(axis=1) / sm[idx].mean(axis=1)
    assert (a["ci_lo"], a["ci_hi"]) == (float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5)))
    assert a["ratio"] == pytest.approx(g.mean() / sm.mean()) and a["ci_lo"] < a["ratio"] < a["ci_hi"]
    seeds = tuple(range(10501, 10507))
    rows: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for i, sd in enumerate(seeds):
        for k in ("bb", "st"):
            rows[("NL_%s_s1" % k, sd)] = (4.0, 6.0)
            for s_ in (4, 16):
                rows[("NL_%s_s%d" % (k, s_), sd)] = (4.0 + sm[i], 6.0 + g[i] - sm[i])
    df = tx_frame(rows)
    t1 = R2.transmission_tables(df, R2.S1_COMPARISONS, (50,), seeds)[0].set_index("arm")
    t2 = R2.transmission_tables(df, list(reversed(R2.S1_COMPARISONS)), (50,), seeds)[0].set_index("arm")
    for arm in t1.index:
        assert (t1.loc[arm, "ci_lo"], t1.loc[arm, "ci_hi"], t1.loc[arm, "ratio"]) == (
            t2.loc[arm, "ci_lo"], t2.loc[arm, "ci_hi"], t2.loc[arm, "ratio"])
    assert t1.loc["NL_bb_s4", "ci_lo"] == pytest.approx(a["ci_lo"], abs=1e-12)


def test_interaction_on_constructed_data():
    """(st_s - st_s1) - (bb_s - bb_s1) per (q, seed) on |peak|, the remainder and the smoothing part."""
    seeds = (10501, 10502, 10503, 10504)
    rows = []
    adj = {10501: 0.0, 10502: 0.01, 10503: 0.02, 10504: 0.03}
    for sd in seeds:
        for arm, peak in (("NL_bb_s1", 0.10), ("NL_bb_s4", 0.07), ("NL_st_s1", 0.08), ("NL_st_s4", 0.04 + adj[sd]),
                          ("NL_bb_s16", 0.05), ("NL_st_s16", 0.05)):
            sm = {"NL_bb_s1": 4.0, "NL_bb_s4": 3.0, "NL_st_s1": 4.0, "NL_st_s4": 2.5}.get(arm, 3.0)
            rem = {"NL_bb_s1": 6.0, "NL_bb_s4": 6.5, "NL_st_s1": 5.0, "NL_st_s4": 5.0 + adj[sd] * 10}.get(arm, 5.0)
            rows.append(settle(blank(arm, 50, sd, stage2_peak_rel_err_abs=peak, smoothing=sm, remainder=rem,
                                     gap=sm + rem)))
    df = pd.DataFrame(rows, columns=R2.COLUMNS)
    # seed 10504 of NL_bb_s4 is missing: that seed leaves the s = 4 interaction (3 pairs) but not the s = 16 one
    i = df.index[(df["arm"] == "NL_bb_s4") & (df["seed"] == 10504)][0]
    df.loc[i, ["status", "complete"]] = ["missing", False]
    summ, long = R2.interaction_tables(df, (50,), seeds)
    s4 = summ[(summ["s"] == 4) & (summ["metric"] == R2.PRIMARY)].iloc[0]
    # per seed: st change = (0.04 + adj) - 0.08 = -0.04 + adj; bb change = 0.07 - 0.10 = -0.03; interaction = -0.01 + adj
    d = [-0.01, 0.0, 0.01]
    assert s4["n_pairs"] == 3 and s4["mean"] == pytest.approx(0.0, abs=1e-12) and s4["median"] == pytest.approx(0.0, abs=1e-12)
    lo, hi = indep_ci(d)
    assert s4["ci_mean_lo"] == pytest.approx(lo, abs=1e-12) and s4["ci_mean_hi"] == pytest.approx(hi, abs=1e-12)
    assert s4["n_pos"] + s4["n_neg"] + s4["n_zero"] == 3 and s4["n_neg"] >= 1 and s4["n_pos"] >= 1
    rem4 = summ[(summ["s"] == 4) & (summ["metric"] == "remainder")].iloc[0]
    # st change = 10 * adj; bb change = +0.5 -> interaction = 10 * adj - 0.5 over seeds 10501-3 = [-0.5, -0.4, -0.3]
    assert rem4["mean"] == pytest.approx(-0.4, abs=1e-12) and rem4["n_pairs"] == 3
    sm4 = summ[(summ["s"] == 4) & (summ["metric"] == "smoothing")].iloc[0]
    assert sm4["mean"] == pytest.approx((-1.5) - (-1.0), abs=1e-12)                # st -1.5, bb -1.0 -> -0.5
    s16 = summ[(summ["s"] == 16) & (summ["metric"] == R2.PRIMARY)].iloc[0]
    assert s16["n_pairs"] == 4 and s16["mean"] == pytest.approx((0.05 - 0.08) - (0.05 - 0.10), abs=1e-12)
    # seed-level rows: the four components and the interaction
    r = long[(long["s"] == 4) & (long["metric"] == R2.PRIMARY) & (long["seed"] == 10503)].iloc[0]
    assert (r["st_s"], r["st_s1"], r["bb_s"], r["bb_s1"]) == pytest.approx((0.06, 0.08, 0.07, 0.10), abs=1e-12)
    assert r["interaction"] == pytest.approx(0.01, abs=1e-12) and r["st_change"] == pytest.approx(-0.02)
    assert len(long[(long["s"] == 4) & (long["metric"] == R2.PRIMARY)]) == 3 and set(summ["status"]) == {"descriptive"}


# ====================================================================== the decomposition columns
def test_decomposition_columns_against_the_formulas():
    row = blank("NL_bb_s4", 50, 10501, g2_at_0=70.0, e2_at_0=66.0, smoothed_e_pred_0=68.5, sigma_effort_at_0_t2=3.0)
    R2.add_r2_columns(row, False)
    assert row["gap"] == pytest.approx(4.0) and row["smoothing"] == pytest.approx(1.5)
    assert row["remainder"] == pytest.approx(2.5)
    assert row["gap"] == pytest.approx(row["smoothing"] + row["remainder"])
    assert row["gap_rel"] == pytest.approx(4.0 / 70.0) and row["smoothing_rel"] == pytest.approx(1.5 / 70.0)
    assert row["remainder_rel"] == pytest.approx(2.5 / 70.0) and row["sigma_2_0"] == 3.0
    assert row["smoothing_over_formula"] == pytest.approx(1.5 / (70.0 * 3.0 / (math.sqrt(math.pi) * 50.0)))
    assert (row["sampler"], row["s"]) == ("bb", 4.0) and math.isnan(row["conc_scale_final"])
    # a comparator without smoothed_game (parents_A): the gap is defined, the split is NaN
    p = blank("parents_A", 60, 10501, g2_at_0=58.3333, e2_at_0=55.0, sigma_effort_at_0_t2=2.9)
    R2.add_r2_columns(p, False)
    assert p["gap"] == pytest.approx(3.3333) and math.isnan(p["smoothing"]) and math.isnan(p["remainder"])
    assert math.isnan(p["smoothing_over_formula"]) and p["sampler"] == "" and math.isnan(p["s"])


def _write_noise(d: Path, **over: float) -> None:
    rec = {"e_hat_0": 66.0, "sigma_0": 3.0, "e_sigma_0": 68.5, "g2_0": 70.0, "smoothing": 1.5, "remainder": 2.5,
           "gap": 4.0, "smoothing_over_gaussian_formula": 1.5 / (70.0 * 3.0 / (math.sqrt(math.pi) * 50.0)),
           "conc_scale": 4.0}
    rec.update(over)
    (d / "rule_log.json").write_text(json.dumps({"stages": {"2": {"freeze": {"noise": rec}}}}))


def _nl_row(d: Path) -> Dict[str, Any]:
    return blank("NL_bb_s4", 50, 10501, g2_at_0=70.0, e2_at_0=66.0, smoothed_e_pred_0=68.5, sigma_effort_at_0_t2=3.0,
                 run_dir=str(d))


def test_the_noise_record_cross_check_flags_a_disagreement_and_never_raises(tmp_path):
    _write_noise(tmp_path)
    row = _nl_row(tmp_path)
    R2.add_r2_columns(row, True)
    assert row["flags"] == "" and row["conc_scale_final"] == 4.0
    # the float32 sigma of the verifier: 1e-7 relative is not a disagreement
    _write_noise(tmp_path, sigma_0=3.0 * (1 + 1e-7))
    row = _nl_row(tmp_path)
    R2.add_r2_columns(row, True)
    assert row["flags"] == ""
    # a 1e-7 absolute difference of the gap is
    _write_noise(tmp_path, gap=4.0 + 1e-7)
    row = _nl_row(tmp_path)
    R2.add_r2_columns(row, True)
    assert "decomposition disagrees" in row["flags"] and "gap" in row["flags"]
    _write_noise(tmp_path, sigma_0=3.001)                                   # 3e-4 relative on sigma
    row = _nl_row(tmp_path)
    R2.add_r2_columns(row, True)
    assert "sigma_0" in row["flags"]
    (tmp_path / "rule_log.json").write_text(json.dumps({"stages": {"2": {"freeze": {}}}}))   # no record
    row = _nl_row(tmp_path)
    R2.add_r2_columns(row, True)
    assert "no noise record" in row["flags"] and math.isnan(row["conc_scale_final"])
    (tmp_path / "rule_log.json").unlink()                                   # unreadable file: still no exception
    row = _nl_row(tmp_path)
    R2.add_r2_columns(row, True)
    assert "no noise record" in row["flags"]
    row = _nl_row(tmp_path)
    row["status"] = "running"
    R2.add_r2_columns(row, True)
    assert row["flags"] == ""                                               # a running run has no record yet: no flag


def test_predictions_table_on_constructed_data(world):
    df = world.copy()
    m = df["complete"].astype(bool)
    # sigma(s=1) = 4; sigma(s) = 4 / sqrt(s) * (1 + 0.01 * seed index) on the bb arms
    for i, r in df[m].iterrows():
        k = SEEDS4.index(int(r["seed"]))
        df.loc[i, "sigma_2_0"] = 4.0 if r["s"] == 1 else 4.0 / math.sqrt(r["s"]) * (1 + 0.01 * k)
        df.loc[i, "smoothing_over_formula"] = 1.0 + (0.001 if k < 2 else 0.01)
    t = R2.predictions_table(df, QS).set_index(["arm", "q"])
    r = t.loc[("NL_bb_s4", 50)]
    assert r["sigma_ratio_n"] == 4 and r["sigma_ratio_min"] == pytest.approx(1.0) and r["sigma_ratio_max"] == pytest.approx(1.03)
    assert r["sigma_ratio_mean"] == pytest.approx(1.015) and r["n_outside_0p5pct"] == 2
    assert t.loc[("NL_bb_s1", 50), "sigma_ratio_mean"] == 1.0
    assert r["planning_floor_pct"] == pytest.approx(100 * 1.25 / (math.sqrt(math.pi) * 50))
    assert r["floor_pct_mean"] == pytest.approx(100 * pd.to_numeric(df[(df["arm"] == "NL_bb_s4") & (df["q"] == 50)]["smoothing"]).div(70.0).mean())
    assert t.loc[("NL_st_s16", 60), "n"] == 3                                # the missing run is not in the table


def test_gates_table_counts_and_nan_for_a_quantity_an_arm_does_not_have(world):
    g = R2.gates_table(world, list(R2.NL_ARMS), QS).set_index(["arm", "q"])
    assert g.loc[("NL_bb_s1", 50), "n_G_A"] == 3 and g.loc[("NL_bb_s1", 50), "n_done"] == 4
    assert g.loc[("NL_st_s4", 50), "n_G_A_and_G_N_eta"] == 3
    assert pd.isna(g.loc[("NL_bb_s1", 50), "n_G_F"]) and pd.isna(g.loc[("parents_A", 50), "n_S1"])


# ====================================================================== the blind recomputation
def write_outputs(df: pd.DataFrame, d: Path) -> None:
    """per_run.csv, criterion.csv and the interaction.csv summary of a frame (the analysis tool's own writers)."""
    d.mkdir(parents=True, exist_ok=True)
    comps = R2.S1_COMPARISONS
    p, _ = R2.paired_tables(df, comps, R2.SECONDARY_METRICS, QS, SEEDS4, R2.LABEL_S1)
    crit = R2.criterion_table(df, p, comps, QS, SEEDS4, R2.LABEL_S1, R2.CRITERION_NOTE)
    inter, _ = R2.interaction_tables(df, QS, SEEDS4)
    for name, t in (("per_run", df), ("criterion", crit), ("interaction", inter)):
        R1.write_csv(t, d / ("%s.csv" % name), "abc123def456")


@pytest.fixture(scope="module")
def synth_dir(world, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("synth")
    write_outputs(world, d)
    return d


def test_blind_recomputation_agrees_on_the_synthetic_output(synth_dir):
    assert BL.main(["--analysis-dir", str(synth_dir)]) == 0
    txt = (synth_dir / "blind_recomputation.txt").read_text()
    assert "ALL " in txt and "DISAGREE" not in txt and "numpy.random.default_rng(20261007)" in txt
    assert "interaction s=4 q=50" in txt and "NL_st_s16 (vs NL_st_s1)" in txt and "q50/10503" in txt
    assert txt.startswith("blind recomputation of criterion.csv and the interaction.csv summary")


def _tamper(synth_dir: Path, tmp_path: Path, name: str, fn: Any) -> int:
    d = tmp_path / "t"
    shutil.copytree(synth_dir, d)
    t = pd.read_csv(d / name, float_precision="round_trip")
    fn(t)
    t.to_csv(d / name, index=False)
    return BL.main(["--analysis-dir", str(d)])


@pytest.mark.parametrize("tamper", ["ci_hi", "mean", "flag", "violation", "pending", "pairs", "drop_arm", "extra_arm",
                                    "overall", "baseline", "boot_seed", "n_base_pass"])
def test_blind_recomputation_fails_when_criterion_is_tampered_with(synth_dir, tmp_path, tamper):
    def fn(crit: pd.DataFrame) -> None:
        i = int(crit.index[crit["arm"] == "NL_bb_s16"][0])
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
            crit.loc[len(crit)] = {**crit.iloc[i].to_dict(), "arm": "NL_bb_s64"}
        elif tamper == "overall":
            crit.loc[i, "overall"] = "met"
        elif tamper == "baseline":
            crit.loc[i, "baseline"] = "NL_st_s1"
        elif tamper == "boot_seed":
            crit.loc[i, "boot_seed"] = 20261006
        elif tamper == "n_base_pass":
            crit.loc[i, "n_base_pass"] += 1
    assert _tamper(synth_dir, tmp_path, "criterion.csv", fn) == 1


@pytest.mark.parametrize("tamper", ["mean", "ci_lo", "n_pairs", "median", "drop_row", "extra_row", "sign_count"])
def test_blind_recomputation_fails_when_the_interaction_table_is_tampered_with(synth_dir, tmp_path, tamper):
    def fn(t: pd.DataFrame) -> None:
        i = int(t.index[(t["s"] == 16) & (t["q"] == 60) & (t["metric"] == "remainder")][0])
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
            t.loc[len(t)] = {**t.iloc[i].to_dict(), "metric": "gap"}
        elif tamper == "sign_count":
            t.loc[i, "n_pos"] += 1
    assert _tamper(synth_dir, tmp_path, "interaction.csv", fn) == 1


@pytest.mark.parametrize("what", ["peak", "gate_column", "status", "remainder", "smoothing", "eta"])
def test_blind_recomputation_fails_when_a_per_run_value_is_changed(synth_dir, tmp_path, what):
    def fn(per: pd.DataFrame) -> None:
        j = int(per.index[(per["arm"] == "NL_st_s4") & (per["q"] == 50) & (per["seed"] == 10501)][0])
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
        elif what == "eta":
            per.loc[j, "eta_T_over_dw"] = 0.0049
            per.loc[j, "eta_dev"] = 0.0062
    assert _tamper(synth_dir, tmp_path, "per_run.csv", fn) == 1
    assert "DISAGREE" in (tmp_path / "t" / "blind_recomputation.txt").read_text()


def test_blind_agrees_on_the_boundary_case_where_the_interval_ends_exactly_at_zero(world, tmp_path):
    """All paired differences 0: ci_mean_hi == 0, (a) is NOT met (strict), and the blind script says the same."""
    df = world.copy()
    m = df["arm"] == "NL_bb_s4"
    base = df[df["arm"] == "NL_bb_s1"].set_index(["q", "seed"])[R2.PRIMARY]
    df.loc[m, R2.PRIMARY] = [base.loc[(q, s)] for q, s in zip(df.loc[m, "q"], df.loc[m, "seed"])]
    write_outputs(df, tmp_path)
    crit = pd.read_csv(tmp_path / "criterion.csv").set_index("arm")
    assert crit.loc["NL_bb_s4", "ci_mean_hi_q50"] == 0.0 and not crit.loc["NL_bb_s4", "a_q50"]
    assert BL.main(["--analysis-dir", str(tmp_path)]) == 0
    crit.loc["NL_bb_s4", "a_q50"] = True                                   # the non-strict reading is a disagreement
    crit.reset_index().to_csv(tmp_path / "criterion.csv", index=False)
    assert BL.main(["--analysis-dir", str(tmp_path)]) == 1


def test_blind_agrees_that_failed_or_running_runs_with_finite_values_never_enter_a_pair(world, tmp_path):
    """Pairs are made of done runs only: a failed / running run that still carries finite metrics is excluded by the
    analysis (criterion and interaction) and by the blind script, which agree."""
    df = world.copy()
    i = df.index[(df["arm"] == "NL_bb_s4") & (df["q"] == 60) & (df["seed"] == 10501)][0]
    j = df.index[(df["arm"] == "NL_st_s16") & (df["q"] == 50) & (df["seed"] == 10502)][0]
    df.loc[i, ["status", "complete"]] = ["failed", False]
    df.loc[j, ["status", "complete"]] = ["running", False]
    assert np.isfinite(df.loc[i, R2.PRIMARY]) and np.isfinite(df.loc[j, R2.PRIMARY])
    write_outputs(df, tmp_path)
    crit = pd.read_csv(tmp_path / "criterion.csv").set_index("arm")
    assert crit.loc["NL_bb_s4", "n_pairs_q60"] == 3 and crit.loc["NL_st_s16", "n_pairs_q50"] == 3
    inter = pd.read_csv(tmp_path / "interaction.csv")
    r = inter[(inter["s"] == 16) & (inter["q"] == 50) & (inter["metric"] == R2.PRIMARY)].iloc[0]
    assert r["n_pairs"] == 3
    assert BL.main(["--analysis-dir", str(tmp_path)]) == 0


def test_blind_recomputation_unreadable_inputs_exit_2(synth_dir, tmp_path):
    d = tmp_path / "u"
    shutil.copytree(synth_dir, d)
    (d / "interaction.csv").unlink()
    assert BL.main(["--analysis-dir", str(d)]) == 2
    d2 = tmp_path / "u2"
    shutil.copytree(synth_dir, d2)
    (d2 / "per_run.csv").unlink()
    assert BL.main(["--analysis-dir", str(d2)]) == 2
    assert BL.main(["--analysis-dir", str(synth_dir), "--protocol", str(tmp_path / "nope.json")]) == 2


def test_blind_script_does_not_import_the_analysis_tool():
    """Plain numpy / pandas / stdlib only: no import of r1_analysis, r2_analysis or anything from tools/ms."""
    src = (ROOT / "tools" / "ms" / "r2_blind_criterion.py").read_text()
    tree = ast.parse(src)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "argparse", "hashlib", "json", "math", "re", "sys", "pathlib", "typing", "numpy",
                    "pandas"}, mods
    for name in ("r1_analysis", "r2_analysis", "ms_configs", "launch_checks", "blind_criterion"):
        assert "import " + name not in src and "from " + name not in src
    assert "20261007" in src and "10000" in src


def test_blind_script_hard_codes_the_pre_registered_scheme_and_reads_the_protocol_thresholds(tmp_path):
    assert BL.SEED == 20261007 and BL.N_RESAMPLES == 10000 and BL.TOL == 1e-12
    th = BL.thresholds(PROTO_PATH)
    assert th == {"eta": TH.eta, "rmse": TH.rmse, "tail": TH.tail, "n_eta": TH.n_eta}
    alt = json.loads(PROTO_PATH.read_text())
    for c in alt["gates"]["G-A"]["all_must_hold"]:
        if c["metric"] == "stage2_tail_mean_over_g2_0":
            c["threshold"] = 0.001
    (tmp_path / "p.json").write_text(json.dumps(alt))
    assert BL.thresholds(tmp_path / "p.json")["tail"] == 0.001


# ====================================================================== a real tiny wave
FAKE_GIT = {"commit": "abc1234", "short": "abc1234", "dirty": False}
Q, SEED = 50, 10501
RAMP = (21, 40)
WIN = R2.Windows(ramp_first=21, ramp_last=40, hold_last=48, decay_last=60, traj_from=20)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The six reduced NL runs of one (q, seed) (ramp 21-40, constant LR to 48, decay to 60, stage 1: 20 updates)."""
    from run import run_ms_stagewise as rms
    from test_ms_r2_runner import reduced_nl
    root = tmp_path_factory.mktemp("tinywave")
    pilot = root / "pilot"
    mp = pytest.MonkeyPatch()
    mp.setattr(rms, "git_state", lambda: dict(FAKE_GIT))
    try:
        for arm in R2.NL_ARMS:
            d = pilot / ("q%d" % Q) / ("seed%d" % SEED) / arm
            d.mkdir(parents=True)
            assert rms.run_pipeline(reduced_nl(arm, str(d), q=Q, seed=SEED, ramp=RAMP), str(d), "pytest",
                                    band_step=2.0) == 0
    finally:
        mp.undo()
    refs = {k: root / k for k in ("ms_r1_pilot", "parents", "rehearsal")}
    for p in refs.values():
        p.mkdir()
    roots = {"pilot": str(pilot), **{k: str(v) for k, v in refs.items()}}
    return roots, root


@pytest.fixture(scope="module")
def tiny_df(tiny):
    roots, _ = tiny
    return R2.extract_all(roots, (Q,), (SEED,), TH)


def test_extraction_of_the_tiny_wave(tiny_df):
    df = tiny_df
    nl = df[df["arm"].isin(R2.NL_ARMS)]
    assert list(nl["arm"]) == list(R2.NL_ARMS) and (nl["status"] == "done").all() and nl["complete"].all()
    assert (nl["role"] == "ms_arm").all() and (nl["flags"] == "").all()
    ref = df[~df["arm"].isin(R2.NL_ARMS)]
    assert list(ref["arm"]) == list(R2.REF_ARMS) and (ref["status"] == "missing").all() and (ref["role"] == "comparator").all()
    assert not ref["complete"].astype(bool).any()
    assert list(df.columns) == R2.COLUMNS and df.shape[0] == 10
    # the decomposition columns are the formulas of the per-run columns
    for r in nl.itertuples():
        assert r.gap == pytest.approx(r.g2_at_0 - r.e2_at_0, abs=1e-12)
        assert r.smoothing == pytest.approx(r.g2_at_0 - r.smoothed_e_pred_0, abs=1e-12)
        assert r.remainder == pytest.approx(r.smoothed_e_pred_0 - r.e2_at_0, abs=1e-12)
        assert r.gap == pytest.approx(r.smoothing + r.remainder, abs=1e-12)
        assert r.sigma_2_0 == r.sigma_effort_at_0_t2 and r.gap_rel == pytest.approx(r.gap / r.g2_at_0)
        f = r.g2_at_0 * r.sigma_effort_at_0_t2 / (math.sqrt(math.pi) * r.q)
        assert r.smoothing_over_formula == pytest.approx(r.smoothing / f, abs=1e-12)
        assert 0.9 < r.smoothing_over_formula < 1.1
        assert r.conc_scale_final == pytest.approx(R2.parse_arm(r.arm)[1])          # the scale at the freeze is s
        assert r.s == R2.parse_arm(r.arm)[1] and r.sampler == R2.parse_arm(r.arm)[0]


def test_the_decomposition_equals_the_rule_log_noise_record(tiny_df):
    for r in tiny_df[tiny_df["arm"].isin(R2.NL_ARMS)].itertuples():
        rec = R2.noise_record(Path(r.run_dir))
        assert rec["gap"] == pytest.approx(r.gap, abs=1e-9) and rec["smoothing"] == pytest.approx(r.smoothing, abs=1e-9)
        assert rec["remainder"] == pytest.approx(r.remainder, abs=1e-9)
        assert rec["sigma_0"] == pytest.approx(r.sigma_2_0, rel=1e-5) and rec["e_sigma_0"] == r.smoothed_e_pred_0
        assert rec["conc_scale"] == r.conc_scale_final


def test_trajectory_table_with_reduced_windows(tiny_df):
    t = R2.trajectory_table(tiny_df, WIN)
    assert len(t) == 6 * 5                                      # K = 10: checks at local 20, 30, 40, 50, 60 per run
    assert t["local"].min() == 20 and t["local"].max() == 60 and set(t["status"]) == {"done"}
    assert list(t.columns) == R2.TRAJ_COLS
    for arm in R2.NL_ARMS:
        s = R2.parse_arm(arm)[1]
        g = t[t["arm"] == arm].set_index("local")
        assert g.loc[20, "conc_scale"] == 1.0 and g.loc[60, "conc_scale"] == pytest.approx(s)
        assert g.loc[30, "conc_scale"] == pytest.approx(1.0 + (s - 1.0) * (30 - 21) / (40 - 21))      # the D2 schedule
        assert g.loc[50, "conc_scale"] == pytest.approx(s)
        row = g.loc[60]
        assert row["gap"] == pytest.approx(row["smoothing"] + row["remainder"], abs=1e-9)
        assert row["smoothing"] == pytest.approx(row["g2_0"] - row["e_sigma_0"], abs=1e-9)
        assert row["remainder"] == pytest.approx(row["e_sigma_0"] - row["e2_at_0"], abs=1e-9)
    r = tiny_df[tiny_df["arm"] == "NL_st_s4"].iloc[0]
    assert t[(t["arm"] == "NL_st_s4") & (t["local"] == 60)]["e2_at_0"].iloc[0] == pytest.approx(r["e2_at_0"], abs=1e-9)
    assert len(R2.trajectory_table(tiny_df, R2.Windows(21, 40, 48, 60, 45))) == 6 * 2


def test_segment_table_with_reduced_windows(tiny_df):
    sg = R2.segment_table(tiny_df, WIN, R2.NL_ARMS, (Q,))
    assert len(sg) == 6 * 4 and list(sg["segment"].iloc[:4]) == ["training", "ramp", "hold", "decay"]
    assert list(sg[["first_local", "last_local"]].iloc[:4].itertuples(index=False, name=None)) == [
        (1, 20), (21, 40), (41, 48), (49, 60)]
    a = sg[sg["arm"] == "NL_bb_s4"].set_index("segment")
    assert a.loc["training", ["conc_scale_min", "conc_scale_max"]].tolist() == [1.0, 1.0] and a.loc["training", "n_updates_mean"] == 20
    assert a.loc["ramp", "conc_scale_min"] == 1.0 and a.loc["ramp", "conc_scale_max"] == pytest.approx(4.0)   # update 21 is in the ramp, at scale 1
    assert a.loc["hold", ["conc_scale_min", "conc_scale_max", "conc_scale_mean"]].tolist() == [4.0, 4.0, 4.0]
    assert a.loc["decay", "n_updates_mean"] == 12 and a.loc["decay", "conc_scale_mean"] == 4.0
    # segment ends: the last check inside the segment (checks at 10, 20, ..., 60)
    assert a.loc["training", "end_check_local"] == 20 and a.loc["ramp", "end_check_local"] == 40
    assert math.isnan(a.loc["hold", "end_check_local"]) and math.isnan(a.loc["hold", "smoothing_end"])   # no check in 41..48
    assert a.loc["decay", "end_check_local"] == 60
    r = tiny_df[tiny_df["arm"] == "NL_bb_s4"].iloc[0]
    assert a.loc["decay", "smoothing_end"] == pytest.approx(r["smoothing"], abs=1e-9)
    assert a.loc["decay", "remainder_end"] == pytest.approx(r["remainder"], abs=1e-9)
    assert a.loc["decay", "e_hat_2_0_end"] == pytest.approx(r["e2_at_0"], abs=1e-9)
    assert a.loc["decay", "sigma_0_end"] == pytest.approx(r["sigma_2_0"], rel=1e-5)
    # the PPO diagnostics are the means of the per-update series over the segment
    u = pd.read_csv(Path(r["run_dir"]) / "ms_updates.csv")
    assert a.loc["ramp", "kl_final_epoch_mean"] == pytest.approx(u[(u["local"] >= 21) & (u["local"] <= 40)]["kl_final_epoch"].mean())
    assert a.loc["hold", "mean_effort_mean"] == pytest.approx(u[(u["local"] >= 41) & (u["local"] <= 48)]["mean_effort"].mean())
    s1 = sg[(sg["arm"] == "NL_bb_s1") & (sg["segment"] == "decay")].iloc[0]
    assert (s1["conc_scale_min"], s1["conc_scale_max"]) == (1.0, 1.0)             # s = 1: the scale is 1 throughout


def test_freeze_decomposition_and_predictions_on_the_tiny_wave(tiny_df):
    fd = R2.freeze_decomposition_table(tiny_df, list(R2.NL_ARMS), (Q,))
    g = fd[(fd["arm"] == "NL_bb_s16") & (fd["quantity"] == "gap")].iloc[0]
    r = tiny_df[tiny_df["arm"] == "NL_bb_s16"].iloc[0]
    assert g["n"] == 1 and g["mean"] == pytest.approx(r["gap"]) and g["unit"] == "effort units"
    assert set(fd["quantity"]) == set(R2.FREEZE_QUANTITIES)
    pr = R2.predictions_table(tiny_df, (Q,))
    assert len(pr) == 6 and (pr["sigma_ratio_n"] == 1).all()
    assert (pr[pr["s"] == 1]["sigma_ratio_mean"] == 1.0).all()
    assert pr[pr["arm"] == "NL_st_s16"]["sigma_ratio_mean"].iloc[0] == pytest.approx(
        tiny_df[tiny_df["arm"] == "NL_st_s16"]["sigma_2_0"].iloc[0] / (tiny_df[tiny_df["arm"] == "NL_st_s1"]["sigma_2_0"].iloc[0] / 4.0))


def test_cli_end_to_end_on_the_tiny_wave(tiny, tmp_path, capsys):
    roots, _ = tiny
    out = tmp_path / "analysis"
    args = ["--pilot-root", roots["pilot"], "--ms-r1-pilot-root", roots["ms_r1_pilot"], "--parents-root",
            roots["parents"], "--rehearsal-root", roots["rehearsal"], "--out", str(out), "--qs", str(Q), "--seeds",
            str(SEED), "--ramp-first", "21", "--ramp-last", "40", "--hold-last", "48", "--decay-last", "60",
            "--traj-from", "20"]
    code = R2.main(args)
    assert code == 3                                           # the four reference rows are missing: reported, not silent
    summ = (out / "summary.txt").read_text()
    assert "PRE-REGISTERED CRITERION" in summ and "TRANSMISSION" in summ and "INTERACTION" in summ
    assert "parents_A q=50: planned 1, done 0" in summ and "MS_base2400 q=50: planned 1, done 0" in summ
    names = ["per_run", "criterion", "paired_secondary", "paired_seed_level", "transmission", "transmission_seed_level",
             "interaction", "interaction_seed_level", "paired_vs_parents_A", "criterion_vs_parents_A",
             "paired_vs_ms_r1_refs", "paired_vs_rehearsal_v2_0", "predictions", "trajectory_checks", "segments",
             "freeze_decomposition", "strata", "strata_summary", "r0", "stage1", "stage1_R1_runs", "stage1_R1", "gates",
             "arm_summary", "budget", "completeness"]
    for n in names:
        t = pd.read_csv(out / ("%s.csv" % n))
        assert t.columns[0] == "roots_id" and t["roots_id"].nunique() <= 1, n
    info = json.loads((out / "analysis_info.json").read_text())
    assert info["boot_seed"] == 20261007 and info["n_boot"] == 10000 and info["all_done"] is False
    assert info["windows"]["ramp_last"] == 40 and set(info["not_stored_by_design"]) == set(R2.REF_ARMS)
    assert info["protocol_sha256"] and info["roots"]["pilot"] == roots["pilot"]
    assert set(info["figures"].values()) == {"ok"} and len(info["figures"]) == 5
    for f in info["figures"]:
        assert (out / "figures" / f).stat().st_size > 5000
    comp = pd.read_csv(out / "completeness.csv").set_index(["arm", "q"])
    assert comp.loc[("NL_st_s16", Q), "n_done"] == 1 and comp.loc[("parents_A", Q), "n_missing"] == 1
    crit = pd.read_csv(out / "criterion.csv")
    assert list(crit["arm"]) == ["NL_bb_s4", "NL_bb_s16", "NL_st_s4", "NL_st_s16"] and (crit["n_pairs_q50"] == 1).all()
    # the references are rows with a status, never dropped, and never in a pair
    per = pd.read_csv(out / "per_run.csv")
    assert (per[per["role"] == "comparator"]["status"] == "missing").all() and len(per) == 10
    pr = pd.read_csv(out / "paired_vs_parents_A.csv")
    assert (pr["n_pairs"] == 0).all()
    # determinism: a second run gives byte-identical CSVs
    out2 = tmp_path / "analysis2"
    args2 = list(args)
    args2[args2.index("--out") + 1] = str(out2)
    assert R2.main(args2 + ["--no-figures"]) == 3
    for n in names:
        assert (out / ("%s.csv" % n)).read_text() == (out2 / ("%s.csv" % n)).read_text(), n
    # the blind script agrees on the real output
    assert BL.main(["--analysis-dir", str(out)]) == 0
    capsys.readouterr()


def test_cli_argument_errors_exit_2(tiny, tmp_path):
    roots, _ = tiny
    base = ["--pilot-root", roots["pilot"], "--ms-r1-pilot-root", roots["ms_r1_pilot"], "--parents-root",
            roots["parents"], "--rehearsal-root", roots["rehearsal"], "--out", str(tmp_path / "o")]
    assert R2.main(base + ["--ramp-first", "50", "--ramp-last", "40"]) == 2          # unordered windows
    bad = list(base)
    bad[bad.index("--parents-root") + 1] = str(tmp_path / "does_not_exist")
    assert R2.main(bad) == 2                                                           # a missing reference root
    with pytest.raises(SystemExit) as e:
        R2.main(base[:-2] + ["--no-such-flag"])
    assert e.value.code == 2
