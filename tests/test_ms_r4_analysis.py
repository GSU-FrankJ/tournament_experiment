"""MS-R4 analysis tools (r4_common, r4_e1_analysis, r4_e2_analysis) and the independent blind recomputation (r4_blind_e1,
r4_blind_e2): every part of both criteria on synthetic rows with known answers (a failed run and a missing pair included), the
statistics against hand calculations, the blind tools against the analysis tables, and an end-to-end run on genuine reduced runs.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r4_analysis.py -p no:cacheprovider -q
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ms_r4_helpers as H  # noqa: E402,I001
from ms_r4_helpers import ROOT, run_dir  # noqa: E402

import r4_blind_e1 as B1  # noqa: E402
import r4_blind_e2 as B2  # noqa: E402
import r4_common as C  # noqa: E402
import r4_configs as rc  # noqa: E402
import r4_curves as CU  # noqa: E402
import r4_e1_analysis as E1  # noqa: E402
import r4_e2_analysis as E2  # noqa: E402

CRIT = json.load(open(ROOT / "reports" / "ms" / "r4" / "prereg_criteria.json"))
SEEDS = list(range(10511, 10531))


# ---------------------------------------------------------------------------------------------- 1. statistics
def test_the_preregistered_parameters_are_those_of_d6():
    assert CRIT["e1"]["peak_abs_mean_max"] == 0.04 and CRIT["e2"]["reduction_min"] == 0.20
    assert CRIT["e2"]["gmax_noninferiority_margin"] == 0.001
    assert (CRIT["bootstrap"]["n_resamples"], CRIT["bootstrap"]["seed_e1"], CRIT["bootstrap"]["seed_e2"]) == (10000, 20261013, 20261014)
    a = CRIT["e2"]["area"]
    assert (a["first_update"], a["last_update"], a["step"], a["n_points"], a["divisor"]) == (25, 1600, 25, 64, 1575)
    assert len(range(a["first_update"], a["last_update"] + 1, a["step"])) == 64 and a["last_update"] - a["first_update"] == 1575
    assert CRIT["e2"]["late_window"] == {"first_update": 200, "last_update": 1600, "divisor": 1400}
    assert CRIT["seeds"] == {"pilot": [10511, 10530], "reproduction": [10501, 10510], "n_per_q": 20}
    assert CRIT["e2"]["decision_order"] == ["fb_r", "fb_rg"] and CRIT["e2"]["eta_level"] == 0.005
    c, sha = C.load_criteria(ROOT / "reports" / "ms" / "r4" / "prereg_criteria.json")
    assert c == CRIT and len(sha) == 64


def test_bootstrap_conventions_by_hand():
    x = np.arange(20, dtype=float) / 7.0
    rng = np.random.default_rng(20261013)
    idx = rng.integers(0, 20, size=(10000, 20))
    want = (float(np.percentile(x[idx].mean(axis=1), 2.5)), float(np.percentile(x[idx].mean(axis=1), 97.5)))
    assert C.boot_mean_ci(x, 10000, 20261013) == want
    assert np.array_equal(C.boot_indices(20, 10000, 20261013), idx)
    a, b = np.linspace(0.5, 1.5, 20), np.linspace(1.0, 2.0, 20)
    red = C.boot_reduction(a, b, 10000, 20261014, ((2.5, 97.5), (1.25, 98.75)))
    idx2 = np.random.default_rng(20261014).integers(0, 20, size=(10000, 20))
    rb = 1.0 - a[idx2].mean(axis=1) / b[idx2].mean(axis=1)
    assert red["rho"] == 1.0 - a.mean() / b.mean()
    assert red["intervals"][0] == (float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5)))
    assert red["intervals"][1] == (float(np.percentile(rb, 1.25)), float(np.percentile(rb, 98.75)))
    assert red["intervals"][1][0] <= red["intervals"][0][0] and red["intervals"][1][1] >= red["intervals"][0][1]
    assert math.isnan(C.boot_mean_ci([], 100, 1)[0]) and math.isnan(C.boot_reduction([], [], 100, 1)["rho"])


def test_clopper_pearson_and_the_power_numbers():
    assert C.clopper_pearson(0, 40)[0] == 0.0 and abs(C.clopper_pearson(0, 40)[1] - 0.0881) < 5e-5
    lo, hi = C.clopper_pearson(1, 20)
    assert abs(lo - 0.00126509) < 1e-7 and abs(hi - 0.24873277) < 1e-7
    assert C.clopper_pearson(20, 20)[1] == 1.0 and C.clopper_pearson(5, 5)[0] > 0.4
    assert abs(0.9 ** 20 - 0.1216) < 1e-4
    assert math.isnan(C.clopper_pearson(0, 0)[0])


def test_trapezoid_and_paired_stats():
    x = np.arange(25, 1601, 25, dtype=float)
    assert abs(C.trapezoid(x, np.full(64, 0.3)) - 0.3 * 1575) < 1e-12
    assert abs(C.trapezoid(x, x) - (1600 ** 2 - 25 ** 2) / 2) < 1e-9
    assert math.isnan(C.trapezoid(x, np.r_[np.nan, np.ones(63)])) and math.isnan(C.trapezoid([1.0], [1.0]))
    st = C.paired_stats(np.arange(10.0), np.arange(10.0) + 1.0, 1000, 5)
    assert st["n_pairs"] == 10 and st["mean_diff"] == -1.0 and st["ci_lo"] == st["ci_hi"] == -1.0 and st["n_neg"] == 10
    st = C.paired_stats([1.0, np.nan, 3.0], [0.0, 1.0, 1.0], 100, 1)
    assert st["n_pairs"] == 2 and st["mean_diff"] == 1.5                       # a non-finite pair is dropped


# ---------------------------------------------------------------------------------------------- 2. E1 criterion
def _e1_rows(lrsm: Dict[int, List[float]], t1: Dict[int, List[float]], relu: Optional[Dict[int, List[float]]] = None) -> pd.DataFrame:
    rows = []
    for arm, d in (("lrsm_st_s16", lrsm), ("t1_st_s16", t1), ("relu_st_s16", relu or t1)):
        for q in (50, 60):
            for s, v in zip(SEEDS, d[q]):
                rows.append(dict(arm=arm, q=q, seed=s, status="done" if np.isfinite(v) else "failed", exit_code=0.0 if np.isfinite(v) else 1.0,
                                 peak_abs=v, nonzero_exit=not np.isfinite(v), gate_G_A=bool(np.isfinite(v)), gate_G_F=bool(np.isfinite(v)),
                                 gate_G_N=bool(np.isfinite(v)), gate_G_S=bool(np.isfinite(v)), gates_all_pass=bool(np.isfinite(v)),
                                 run_dir=f"/nonexistent/{arm}/{q}/{s}"))
    return pd.DataFrame(rows)


def _ramp(a: float, b: float) -> List[float]:
    return list(np.linspace(a, b, 20))


GOOD_L = {50: _ramp(0.010, 0.030), 60: _ramp(0.012, 0.035)}                    # means 0.020 / 0.0235, all <= 0.04
GOOD_T = {50: [v + 0.010 + 0.001 * (i % 3) for i, v in enumerate(GOOD_L[50])], 60: [v + 0.012 + 0.001 * (i % 3) for i, v in enumerate(GOOD_L[60])]}


def _verdict(tab: pd.DataFrame) -> bool:
    return bool(tab[tab["q"].astype(str) == "both"]["met_q"].iloc[0])


def test_e1_all_three_parts_hold():
    tab = E1.criterion_e1(_e1_rows(GOOD_L, GOOD_T), CRIT)
    for q in (50, 60):
        r = tab[tab["q"].astype(str) == str(q)].iloc[0]
        assert r["a_pass"] and r["b_pass"] and r["c_pass"] and r["met_q"] and r["status"] == "ok"
        assert r["n_runs_done"] == 20 and r["n_pairs"] == 20 and r["n_gate_fail"] == 0 and r["n_nonzero_exit"] == 0
        want = np.mean(np.array(GOOD_L[q]) - np.array(GOOD_T[q]))
        assert abs(r["mean_diff_vs_t1"] - want) < 1e-15 and abs(r["mean_abs_peak_lrsm"] - np.mean(GOOD_L[q])) < 1e-15
        assert r["ci_hi"] < 0 and r["ci_lo"] <= r["mean_diff_vs_t1"] <= r["ci_hi"]
    assert _verdict(tab) and "MET" in tab["status"].iloc[-1] and "NOT" not in tab["status"].iloc[-1]


def test_e1_part_a_is_the_mean_not_the_maximum_and_the_bound_is_inclusive():
    crit = copy.deepcopy(CRIT)
    crit["e1"]["peak_abs_mean_max"] = 0.5                                        # binary-exact values: the mean is exactly 0.5
    t = {50: [2.0] * 20, 60: [2.0] * 20}
    l = {50: [0.25] * 10 + [0.75] * 10, 60: [0.5] * 20}                          # q = 50: mean 0.5 exactly, max 0.75; q = 60: 0.5
    tab = E1.criterion_e1(_e1_rows(l, t), crit)
    assert tab["a_pass"].iloc[0] and tab["a_pass"].iloc[1] and _verdict(tab) and tab["mean_abs_peak_lrsm"].iloc[0] == 0.5
    l[60] = [0.5] * 19 + [0.5 + 2.0 ** -20 * 20]                                 # q = 60: the mean exceeds 0.5 by 2^-20
    tab = E1.criterion_e1(_e1_rows(l, t), crit)
    assert tab["a_pass"].iloc[0] and not tab["a_pass"].iloc[1] and not _verdict(tab)
    assert not tab["met_q"].iloc[1] and tab["b_pass"].iloc[1] and tab["c_pass"].iloc[1]


def test_e1_part_b_needs_the_interval_below_zero():
    # a mean improvement of 0.001 with a spread of +-0.02: the interval contains 0
    t = {q: [0.05 + (0.02 if i % 2 else -0.02) for i in range(20)] for q in (50, 60)}
    l = {q: [0.04 + (-0.02 if i % 2 else 0.02) for i in range(20)] for q in (50, 60)}
    tab = E1.criterion_e1(_e1_rows(l, t), CRIT)
    r = tab.iloc[0]
    assert r["mean_diff_vs_t1"] < 0 and r["ci_hi"] > 0 and not r["b_pass"] and r["a_pass"] and not _verdict(tab)
    # identical diffs of -1e-9: the interval is a point below 0
    t2 = {q: [0.03] * 20 for q in (50, 60)}
    l2 = {q: [0.03 - 1e-9] * 20 for q in (50, 60)}
    assert E1.criterion_e1(_e1_rows(l2, t2), CRIT)["b_pass"].iloc[0]
    # a zero difference is not below zero
    assert not E1.criterion_e1(_e1_rows(t2, t2), CRIT)["b_pass"].iloc[0]


def test_e1_part_c_a_gate_failure_and_a_nonzero_exit():
    df = _e1_rows(GOOD_L, GOOD_T)
    i = df[(df.arm == "lrsm_st_s16") & (df.q == 50) & (df.seed == 10515)].index[0]
    df.loc[i, ["gate_G_F", "gates_all_pass"]] = False
    tab = E1.criterion_e1(df, CRIT)
    r = tab.iloc[0]
    assert not r["c_pass"] and r["n_gate_fail"] == 1 and r["n_G_F_fail"] == 1 and r["n_G_A_fail"] == 0 and r["a_pass"] and r["b_pass"]
    assert not r["met_q"] and tab.iloc[1]["met_q"] and not _verdict(tab)
    # a run that exits nonzero: no metrics, status failed -> counted in c, and parts (a) / (b) are 'incomplete'
    l = {q: list(v) for q, v in GOOD_L.items()}
    l[60][3] = float("nan")
    tab = E1.criterion_e1(_e1_rows(l, GOOD_T), CRIT)
    r = tab.iloc[1]
    assert r["n_nonzero_exit"] == 1 and r["n_gate_fail"] == 1 and not r["c_pass"] and r["status"] == "incomplete"
    assert r["n_runs_done"] == 19 and not r["a_pass"] and not r["b_pass"] and not _verdict(tab)


def test_e1_a_missing_run_or_pair_is_a_failure_never_a_skip():
    df = _e1_rows(GOOD_L, GOOD_T)
    gone = df[(df.arm == "lrsm_st_s16") & (df.q == 50) & (df.seed == 10520)].index
    tab = E1.criterion_e1(df.drop(gone), CRIT)
    r = tab.iloc[0]
    assert r["n_gate_fail"] == 1 and r["n_nonzero_exit"] == 1 and not r["c_pass"] and not r["a_pass"] and r["status"] == "incomplete"
    gone_c = df[(df.arm == "t1_st_s16") & (df.q == 60) & (df.seed == 10511)].index
    r = E1.criterion_e1(df.drop(gone_c), CRIT).iloc[1]
    assert r["n_pairs"] == 19 and not r["b_pass"] and r["a_pass"] and r["c_pass"] and r["status"] == "incomplete"


def test_e1_descriptive_tables_by_hand():
    df = _e1_rows(GOOD_L, GOOD_T)
    df["gap"] = df["peak_abs"] * 100
    paired, longs = E1.paired_table(df, CRIT, [("lrsm_st_s16", "t1_st_s16"), ("relu_st_s16", "t1_st_s16")])
    r = paired[(paired["comparison"] == "lrsm_st_s16 - t1_st_s16") & (paired["q"] == 50) & (paired["metric"] == "peak_abs")].iloc[0]
    assert r["n_pairs"] == 20 and abs(r["mean_diff"] - np.mean(np.array(GOOD_L[50]) - np.array(GOOD_T[50]))) < 1e-15 and r["ci_hi_below_0"]
    g = paired[(paired["comparison"] == "lrsm_st_s16 - t1_st_s16") & (paired["q"] == 50) & (paired["metric"] == "gap")].iloc[0]
    assert abs(g["mean_diff"] - 100 * r["mean_diff"]) < 1e-12
    z = paired[(paired["comparison"] == "relu_st_s16 - t1_st_s16") & (paired["metric"] == "peak_abs")]
    assert (z["mean_diff"] == 0).all() and len(longs[longs["metric"] == "peak_abs"]) == 2 * 2 * 20
    rates = E1.failure_rate_table(df, CRIT, ["lrsm_st_s16", "relu_st_s16"])
    assert list(rates["n_fail"]) == [0, 0, 0, 0, 0, 0] and rates.iloc[2]["n_runs"] == 40 and abs(rates.iloc[2]["cp95_hi"] - 0.0881) < 5e-5
    df.loc[df[(df.arm == "relu_st_s16") & (df.q == 50)].index[:5], "gates_all_pass"] = False
    rates = E1.failure_rate_table(df, CRIT, ["relu_st_s16"])
    assert list(rates["n_fail"]) == [5, 0, 5] and abs(rates.iloc[0]["rate"] - 0.25) < 1e-15
    assert rates.iloc[0]["cp95_lo"] == C.clopper_pearson(5, 20)[0] and rates.iloc[0]["cp95_hi"] == C.clopper_pearson(5, 20)[1]


def test_e1_unit_activity_counts_positive_preactivations():
    from envs.curriculum_env import GameSpec
    spec = GameSpec(w_h=6.5, w_l=3.0, k=0.001, q=50.0, T=2, e_min=0.0, e_max=100.0)
    rng = np.random.default_rng(0)
    arrays = {"actor.l1.weight": rng.standard_normal((64, 2)).astype(np.float32), "actor.l1.bias": rng.standard_normal(64).astype(np.float32),
              "actor.l2.weight": rng.standard_normal((64, 64)).astype(np.float32), "actor.l2.bias": rng.standard_normal(64).astype(np.float32),
              "actor_variant": np.asarray("lrsm")}
    act = E1.unit_activity(arrays, spec)
    assert act["layer1"].shape == act["layer2"].shape == (64,) and ((0 <= act["layer1"]) & (act["layer1"] <= 1)).all()
    from run.run_final_dp_br import dense_grid
    x = spec.encode_obs(2, dense_grid(spec.domain_half(2), 4.0))
    z1 = x @ arrays["actor.l1.weight"].T + arrays["actor.l1.bias"]
    assert np.array_equal(act["layer1"], (z1 > 0).mean(axis=0))
    h1 = np.where(z1 > 0, z1, 0.01 * z1)
    z2 = h1 @ arrays["actor.l2.weight"].T + arrays["actor.l2.bias"]
    assert np.allclose(act["layer2"], (z2 > 0).mean(axis=0))
    with pytest.raises(ValueError):
        E1.unit_activity({**arrays, "actor_variant": np.asarray("t10")}, spec)


# ---------------------------------------------------------------------------------------------- 3. E2 criterion
def _curve(level: float, early: float = 0.0) -> np.ndarray:
    """A 64-point curve: ``level`` everywhere (area exactly ``level``), plus ``early`` on the first 8 points."""
    y = np.full(64, level)
    y[:8] += early
    return y


def _e2_frame(levels: Dict[str, Dict[int, List[float]]], gmax: Optional[Dict[str, float]] = None,
              broken: Optional[set] = None, drop_curve: Optional[tuple] = None, fail: Optional[set] = None):
    """Per-run table and curve file of constant curves: A of a run is its level."""
    rows, crows = [], []
    for arm, per_q in levels.items():
        for q in (50, 60):
            for s, lvl in zip(SEEDS, per_q[q]):
                ok = not (fail and (arm, q, s) in fail)
                rows.append(dict(arm=arm, q=q, seed=s, status="done" if ok else "failed", exit_code=0.0 if ok else 1.0, run_dir="/nonexistent",
                                 gates_all_pass=bool(ok), nonzero_exit=not ok, gmax_final=(gmax or {}).get(arm, 0.0004) + 1e-5 * (s - 10511),
                                 peak_abs=0.02, t2_share_near_all=0.35, t2_share_mid_all=0.15))
                if drop_curve == (arm, q, s):
                    continue
                for u, y in zip(range(25, 1601, 25), _curve(lvl)):
                    crows.append({"arm": arm, "q": q, "seed": s, "update": u, "local": u, "final_eta_T_over_dw": y})
    return pd.DataFrame(rows), pd.DataFrame(crows)


def _levels(ref=0.10, x_frac=0.7, noise=0.002) -> Dict[int, List[float]]:
    return {50: [ref + noise * ((i % 5) - 2) for i in range(20)], 60: [ref * 1.5 + noise * ((i % 5) - 2) for i in range(20)]}


def _lv(ref_levels, frac):
    return {q: [v * frac for v in vs] for q, vs in ref_levels.items()}


def _e2(levels, **kw):
    df, curves = _e2_frame(levels, **kw)
    df = E2.add_curve_columns(df, curves, CRIT)
    return df, curves, E2.criterion_e2(df, CRIT)


def _row(tab: pd.DataFrame, arm: str, q) -> pd.Series:
    return tab[(tab["arm"] == arm) & (tab["q"].astype(str) == str(q))].iloc[0]


def test_e2_the_area_is_the_trapezoid_over_1575_updates():
    y = np.linspace(0.2, 0.001, 64)
    df = pd.DataFrame([dict(arm="fb_a0", q=50, seed=10511, status="done", run_dir="/x")])
    curves = pd.DataFrame({"arm": "fb_a0", "q": 50, "seed": 10511, "update": np.arange(25, 1601, 25), "final_eta_T_over_dw": y})
    out = E2.add_curve_columns(df, curves, CRIT).iloc[0]
    u = np.arange(25, 1601, 25, dtype=float)
    want = float(((u[1:] - u[:-1]) * (y[1:] + y[:-1]) / 2).sum() / 1575)
    assert abs(out["A"] - want) < 1e-15 and bool(out["curve_complete"])
    m = u >= 200
    assert abs(out["A_late"] - float(((u[m][1:] - u[m][:-1]) * (y[m][1:] + y[m][:-1]) / 2).sum() / 1400)) < 1e-15
    assert abs(out["log10_mean"] - np.log10(y).mean()) < 1e-15 and out["eta_u1600"] == y[-1]
    assert out["first_le_level"] == u[np.flatnonzero(y <= 0.005)[0]]
    y2 = np.full(64, 0.1)
    c2 = curves.assign(final_eta_T_over_dw=y2)
    o2 = E2.add_curve_columns(df, c2, CRIT).iloc[0]
    assert abs(o2["A"] - 0.1) < 1e-15 and abs(o2["A_late"] - 0.1) < 1e-15 and math.isnan(o2["first_le_level"])
    short = curves.iloc[:-1]
    assert math.isnan(E2.add_curve_columns(df, short, CRIT).iloc[0]["A"])                      # 63 points: not a complete curve
    nan = curves.copy()
    nan.loc[5, "final_eta_T_over_dw"] = np.nan
    assert math.isnan(E2.add_curve_columns(df, nan, CRIT).iloc[0]["A"])


def test_e2_a_target_met_at_both_q():
    ref = _levels()
    df, curves, tab = _e2({"fb_a0": ref, "fb_r": _lv(ref, 0.7), "fb_rg": _lv(ref, 0.95), "fb_rx": _lv(ref, 1.1)})
    for q in (50, 60):
        r = _row(tab, "fb_r", q)
        assert abs(r["rho"] - 0.3) < 1e-12 and r["a_pass"] and r["b_pass"] and r["c_pass"] and r["met_q"] and r["status"] == "ok"
        assert r["ci95_lo"] > 0.2 and r["ci975_lo"] <= r["ci95_lo"] and r["ci975_hi"] >= r["ci95_hi"] and r["n_pairs"] == 20
        assert abs(r["mean_A_ref"] - np.mean(ref[q])) < 1e-15
        assert abs(r["mean_gmax_diff"]) < 1e-15
    assert _row(tab, "fb_r", "both")["met_q"] and not _row(tab, "fb_rg", "both")["met_q"]
    rx = _row(E2.criterion_e2(df, CRIT, ["fb_rx"], "descriptive"), "fb_rx", 60)
    assert rx["role"] == "descriptive" and rx["rho"] < 0 and "fb_rx" not in set(tab["arm"])
    dec = E2.decide(tab, CRIT)
    assert dec["statement_rests_on"] == "fb_r" and "fb_rg" in dec["descriptive"] and "lowers the area" in dec["text"]


def test_e2_the_decision_order_fb_r_first_then_fb_rg_else_nothing():
    ref = _levels()
    both = {"fb_a0": ref, "fb_r": _lv(ref, 0.6), "fb_rg": _lv(ref, 0.5)}
    _, _, tab = _e2(both)
    assert _row(tab, "fb_r", "both")["met_q"] and _row(tab, "fb_rg", "both")["met_q"]
    assert E2.decide(tab, CRIT)["statement_rests_on"] == "fb_r"                                  # fb_rg is descriptive if fb_r meets it
    _, _, tab = _e2({"fb_a0": ref, "fb_r": _lv(ref, 0.9), "fb_rg": _lv(ref, 0.5)})
    d = E2.decide(tab, CRIT)
    assert d["statement_rests_on"] == "fb_rg" and "fb_r" in d["descriptive"]
    _, _, tab = _e2({"fb_a0": ref, "fb_r": _lv(ref, 0.9), "fb_rg": _lv(ref, 0.85)})
    d = E2.decide(tab, CRIT)
    assert d["statement_rests_on"] is None and "verifier only" in d["text"] and "no training improvement is claimed" in d["text"]


def test_e2_part_a_needs_20_percent_at_both_q_and_a_positive_lower_bound():
    ref = _levels()
    lv = {50: [v * 0.7 for v in ref[50]], 60: [v * 0.85 for v in ref[60]]}                      # q = 60: only 15 %
    _, _, tab = _e2({"fb_a0": ref, "fb_r": lv, "fb_rg": _lv(ref, 1.0)})
    assert _row(tab, "fb_r", 50)["met_q"] and not _row(tab, "fb_r", 60)["a_pass"] and not _row(tab, "fb_r", "both")["met_q"]
    assert abs(_row(tab, "fb_r", 60)["rho"] - 0.15) < 1e-12
    # a reduction of 20 % up to rounding: rho is 0.19999999999999996 or 0.2 in float64 (1 - 0.8 is one ulp below 0.2), so a true tie
    # is decided by the last bit; the pre-registered comparison is exact (>=) and a value that close to the bound would be flagged
    # by the blind recomputation, not decided by it (review finding R4-08)
    _, _, tab = _e2({"fb_a0": ref, "fb_r": _lv(ref, 0.8), "fb_rg": _lv(ref, 1.0)})
    assert abs(_row(tab, "fb_r", 50)["rho"] - 0.2) < 1e-12 and _row(tab, "fb_r", 50)["a_pass"] == bool(_row(tab, "fb_r", 50)["rho"] >= 0.2)
    # a 25 % mean reduction with a seed spread so wide that the interval reaches 0
    rng = np.random.default_rng(3)
    noisy = {q: list(np.array(ref[q]) * np.where(np.arange(20) % 2 == 0, 0.05, 1.45)) for q in (50, 60)}
    _, _, tab = _e2({"fb_a0": ref, "fb_r": noisy, "fb_rg": _lv(ref, 1.0)})
    r = _row(tab, "fb_r", 50)
    assert r["rho"] > 0.2 and r["ci95_lo"] < 0 and not r["a_pass"]


def test_e2_part_b_one_new_gate_failure_fails_the_target():
    ref = _levels()
    seed = 10517
    _, _, tab = _e2({"fb_a0": ref, "fb_r": _lv(ref, 0.6), "fb_rg": _lv(ref, 0.6)}, fail={("fb_r", 60, seed)})
    r = _row(tab, "fb_r", 60)
    assert r["n_gate_violations"] == 1 and r["violating_seeds"] == str(seed) and not r["b_pass"] and not r["met_q"]
    assert _row(tab, "fb_r", 50)["b_pass"] and _row(tab, "fb_rg", "both")["met_q"]
    # a seed that fails under fb_a0 too is not a new failure
    _, _, tab = _e2({"fb_a0": ref, "fb_r": _lv(ref, 0.6), "fb_rg": _lv(ref, 0.6)}, fail={("fb_r", 60, seed), ("fb_a0", 60, seed)})
    assert bool(_row(tab, "fb_r", 60)["b_pass"]) and _row(tab, "fb_r", 60)["n_gate_violations"] == 0
    # but with the reference run failed there is no pair: the area pair is dropped (19 pairs -> incomplete -> part a false)
    assert _row(tab, "fb_r", 60)["n_pairs"] == 19 and not _row(tab, "fb_r", 60)["a_pass"] and _row(tab, "fb_r", 60)["status"] == "incomplete"


def test_e2_part_c_the_stage_1_gain_must_not_get_worse_by_more_than_the_margin():
    ref = _levels()
    base = {"fb_a0": ref, "fb_r": _lv(ref, 0.6), "fb_rg": _lv(ref, 0.6)}
    _, _, tab = _e2(base, gmax={"fb_r": 0.0004 + 0.0009, "fb_rg": 0.0004 + 0.0011})
    r1, r2 = _row(tab, "fb_r", 50), _row(tab, "fb_rg", 50)
    assert abs(r1["mean_gmax_diff"] - 0.0009) < 1e-12 and r1["gmax_ci_hi"] <= 0.001 + 1e-12 and r1["c_pass"] and r1["met_q"]
    assert abs(r2["mean_gmax_diff"] - 0.0011) < 1e-12 and r2["gmax_ci_hi"] > 0.001 and not r2["c_pass"] and not r2["met_q"]
    _, _, tab = _e2(base, gmax={"fb_r": 0.0004 - 0.01})                                          # better is fine
    assert _row(tab, "fb_r", 50)["c_pass"]


def test_e2_a_missing_curve_leaves_a_pair_missing_not_a_skipped_check():
    ref = _levels()
    _, _, tab = _e2({"fb_a0": ref, "fb_r": _lv(ref, 0.6), "fb_rg": _lv(ref, 0.6)}, drop_curve=("fb_r", 50, 10520))
    r = _row(tab, "fb_r", 50)
    assert r["n_pairs"] == 19 and not r["a_pass"] and r["status"] == "incomplete" and not r["met_q"] and not _row(tab, "fb_r", "both")["met_q"]
    assert _row(tab, "fb_rg", "both")["met_q"]


def test_e2_secondary_and_sampling_tables_by_hand():
    ref = _levels()
    df, curves, _ = _e2({"fb_a0": ref, "fb_r": _lv(ref, 0.5), "fb_rg": _lv(ref, 0.9), "fb_rx": _lv(ref, 1.2)})
    sec = E2.secondary_table(df, CRIT, ["fb_r", "fb_rx"])
    a = sec[(sec["arm"] == "fb_r") & (sec["q"] == 50) & (sec["statistic"] == "A_late")].iloc[0]
    assert abs(a["rho"] - 0.5) < 1e-12 and a["n_pairs"] == 20
    p = sec[(sec["arm"] == "fb_rx") & (sec["q"] == 60) & (sec["statistic"] == "peak_abs")].iloc[0]
    assert p["mean_diff"] == 0.0
    cs = E2.curve_summary(curves, ["fb_a0"])
    assert len(cs) == 2 * 64 and abs(cs[(cs["q"] == 50) & (cs["update"] == 25)]["eta_median"].iloc[0] - np.median(ref[50])) < 1e-15
    preds = E2.predictions_table(pd.concat([E2.criterion_e2(df, CRIT), E2.criterion_e2(df, CRIT, ["fb_rx"], "descriptive")]), sec,
                                 df.assign(dev_verifier_share_stage2=0.004), CRIT)
    p1 = preds[preds["prediction"] == "E2-P1"].iloc[0]
    assert p1["observed_rho_fb_rx_q60"] < 0 and "larger" in p1["outcome"]
    assert (preds[preds["prediction"] == "E2-P3"]["max_dev_verifier_share_stage2"] < 0.01).all()


# ---------------------------------------------------------------------------------------------- 4. the blind tools
def test_blind_e1_agrees_with_the_analysis_and_catches_a_difference(tmp_path):
    for l, t in ((GOOD_L, GOOD_T), ({50: [0.0] * 10 + [0.08] * 10, 60: [0.05] * 20}, {50: [0.05] * 20, 60: [0.06] * 20})):
        df = _e1_rows(l, t)
        i = df[(df.arm == "lrsm_st_s16") & (df.q == 60) & (df.seed == 10513)].index[0]
        df.loc[i, ["gate_G_A", "gates_all_pass"]] = False
        tab = E1.criterion_e1(df, CRIT)
        p_run, p_crit = tmp_path / "per_run.csv", tmp_path / "criterion_e1.csv"
        df.to_csv(p_run, index=False)
        tab.to_csv(p_crit, index=False)
        blind = B1.recompute(pd.read_csv(p_run), CRIT)
        assert B1.compare(blind, p_crit) == []
        for col in ("mean_abs_peak_lrsm", "mean_diff_vs_t1", "ci_lo", "ci_hi"):
            assert np.all(np.abs(blind[col].to_numpy(float) - tab[tab["q"].astype(str) != "both"][col].to_numpy(float)) < 1e-12), col
        bad = pd.read_csv(p_crit)
        bad.loc[0, "ci_hi"] += 1e-9
        bad.to_csv(p_crit, index=False)
        assert any("ci_hi" in d for d in B1.compare(blind, p_crit))
        bad = pd.read_csv(p_crit)
        bad.loc[0, "ci_hi"] -= 1e-9
        bad.loc[1, "c_pass"] = not bool(bad.loc[1, "c_pass"])
        bad.to_csv(p_crit, index=False)
        assert any("c_pass" in d for d in B1.compare(blind, p_crit))


def test_blind_e1_cli_and_the_gates_json_cross_check(tmp_path):
    df = _e1_rows(GOOD_L, GOOD_T)
    df.to_csv(tmp_path / "per_run.csv", index=False)
    E1.criterion_e1(df, CRIT).to_csv(tmp_path / "criterion_e1.csv", index=False)
    crit_p = tmp_path / "crit.json"
    json.dump(CRIT, open(crit_p, "w"))
    args = ["--per-run", str(tmp_path / "per_run.csv"), "--criteria", str(crit_p), "--out", str(tmp_path / "blind.csv"),
            "--compare", str(tmp_path / "criterion_e1.csv")]
    assert B1.main(args) == 0 and (tmp_path / "blind.csv").exists()
    # --check-gates-json: a run directory without gates.json is a problem
    assert B1.main(args + ["--check-gates-json"]) == 3
    # a run directory whose gates.json agrees with the table
    d = tmp_path / "run"
    d.mkdir()
    json.dump({"G-A": {"pass": True}, "G-F": {"pass": True}, "G-N": {"pass": True}, "G-S": {"pass": True},
               "reported": {"end_of_stage2": {"final": {"stage2_peak_rel_err_abs": 0.0123}}}}, open(d / "gates.json", "w"))
    one = pd.DataFrame([dict(arm="lrsm_st_s16", q=50, seed=10511, status="done", run_dir=str(d), peak_abs=0.0123, gate_G_A=True,
                             gate_G_F=True, gate_G_N=True, gate_G_S=True)])
    assert B1.cross_check_gates_json(one, "lrsm_st_s16") == []
    assert B1.cross_check_gates_json(one.assign(gate_G_F=False), "lrsm_st_s16")
    assert B1.cross_check_gates_json(one.assign(peak_abs=0.0124), "lrsm_st_s16")


def test_blind_e2_agrees_with_the_analysis_and_catches_a_difference(tmp_path):
    ref = _levels()
    cases = [
        ({"fb_a0": ref, "fb_r": _lv(ref, 0.7), "fb_rg": _lv(ref, 0.95), "fb_rx": _lv(ref, 1.1)}, {}),
        ({"fb_a0": ref, "fb_r": _lv(ref, 0.9), "fb_rg": _lv(ref, 0.5), "fb_rx": _lv(ref, 0.79)}, {"fail": {("fb_rg", 50, 10512)}}),
        ({"fb_a0": ref, "fb_r": _lv(ref, 0.9), "fb_rg": _lv(ref, 0.85), "fb_rx": _lv(ref, 1.0)}, {"drop_curve": ("fb_rx", 60, 10530)}),
    ]
    for levels, kw in cases:
        df, curves, _ = _e2(levels, **kw)
        tab = pd.concat([E2.criterion_e2(df, CRIT), E2.criterion_e2(df, CRIT, ["fb_rx"], "descriptive")], ignore_index=True)
        dec = E2.decide(tab, CRIT)
        p_run, p_cur, p_crit, p_info = tmp_path / "per_run.csv", tmp_path / "curves_all.csv", tmp_path / "criterion_e2.csv", tmp_path / "info.json"
        df.to_csv(p_run, index=False)
        curves.to_csv(p_cur, index=False)
        tab.to_csv(p_crit, index=False)
        json.dump({"decision": dec}, open(p_info, "w"))
        blind, bdec = B2.recompute(pd.read_csv(p_run), pd.read_csv(p_cur), CRIT)
        assert B2.compare(blind, bdec, str(p_crit), str(p_info)) == []
        assert bdec["statement_rests_on"] == dec["statement_rests_on"]
        for col in ("rho", "ci95_lo", "ci95_hi", "ci975_lo", "ci975_hi", "mean_A_x", "mean_A_ref", "mean_gmax_diff", "gmax_ci_hi"):
            a = blind.sort_values(["role", "arm", "q"])[col].to_numpy(float)
            b = tab[tab["q"].astype(str) != "both"].sort_values(["role", "arm", "q"])[col].to_numpy(float)
            assert np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True), col
        bad = pd.read_csv(p_crit)
        bad.loc[bad.index[0], "rho"] += 1e-9
        bad.to_csv(p_crit, index=False)
        assert any("rho" in d for d in B2.compare(blind, bdec, str(p_crit), str(p_info)))
        json.dump({"decision": {"statement_rests_on": "fb_rg" if dec["statement_rests_on"] != "fb_rg" else None}}, open(p_info, "w"))
        bad.loc[bad.index[0], "rho"] -= 1e-9
        bad.to_csv(p_crit, index=False)
        assert any("decision" in d for d in B2.compare(blind, bdec, str(p_crit), str(p_info)))


def test_the_blind_tools_do_not_import_the_analysis_tools():
    import ast
    for name in ("r4_blind_e1.py", "r4_blind_e2.py"):
        tree = ast.parse((ROOT / "tools" / "ms" / name).read_text())
        mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)} | \
               {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not (mods & {"r4_common", "r4_e1_analysis", "r4_e2_analysis", "r1_analysis", "r2_analysis", "r3_analysis", "r4_configs"}), (name, mods)


# ---------------------------------------------------------------------------------------------- 5. end to end on genuine runs
@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    wave = H.get_wave(tmp_path_factory)
    base = tmp_path_factory.mktemp("e2e")
    curves = base / "curves"
    CU.run_all(wave, curves, list(rc.ARMS), [50], [10511], workers=4, c_off=False)
    # the pre-registered parameters at the reduced scale of the test runs: one seed, one q, the 12 exports of the 60-update arms
    crit = copy.deepcopy(CRIT)
    crit["qs"] = [50]
    crit["seeds"].update(pilot=[10511, 10511], n_per_q=1)
    crit["e2"]["area"] = {"curve": "reduced", "first_update": 10, "last_update": 60, "step": 10, "n_points": 6, "rule": "trapezoid", "divisor": 50}
    crit["e2"]["late_window"] = {"first_update": 20, "last_update": 60, "divisor": 40}
    crit["bootstrap"]["n_resamples"] = 500
    crit_p = base / "crit.json"
    json.dump(crit, open(crit_p, "w"))
    return wave, curves, crit_p, base


def test_end_to_end_e1_analysis_on_genuine_runs(e2e, tmp_path):
    wave, curves, crit_p, _ = e2e
    out = tmp_path / "e1"
    info = E1.run_analysis(wave, curves, crit_p, out)
    assert info["n_planned"] == 3 and info["n_not_done"] == 0
    df = pd.read_csv(out / "per_run.csv")
    assert set(df["arm"]) == set(rc.E1_ARMS) and (df["status"] == "done").all() and df["gates_all_pass"].dtype == bool
    assert np.isfinite(df["peak_abs"]).all() and np.isfinite(df["wall_total_sec"]).all() and (df["dev_verifier_share_stage2"] < 0.05).all()
    assert (df["actor_variant_cfg"] == ["t1", "relu", "lrsm"]).all()
    for f in ("criterion_e1.csv", "paired_e1.csv", "failure_rates.csv", "failures.csv", "cost_e1.csv", "strata.csv", "strata_summary.csv",
              "learning_curves_e1.csv",
              "tie_profile_e1.csv", "predictions_e1.csv", "analysis_info.json", "summary.txt"):
        assert (out / f).exists(), f
    for fig in ("peak_abs_by_arm.png", "paired_differences.png", "tie_profile.png", "learning_curves.png"):
        assert (out / "figures" / fig).stat().st_size > 1000
    crit_tab = pd.read_csv(out / "criterion_e1.csv")
    assert list(crit_tab["q"].astype(str)) == ["50", "both"]
    blind = B1.recompute(df, json.load(open(crit_p)))
    assert B1.compare(blind, out / "criterion_e1.csv") == []
    assert B1.cross_check_gates_json(df, "lrsm_st_s16") == []
    with pytest.raises(SystemExit):                                                         # an existing output directory is refused
        E1.main(["--pilot-root", str(wave), "--curves-root", str(curves), "--criteria", str(crit_p), "--out", str(out)])


def test_end_to_end_e2_analysis_on_genuine_runs(e2e, tmp_path):
    wave, curves, crit_p, _ = e2e
    out = tmp_path / "e2"
    info = E2.run_analysis(wave, curves, crit_p, out)
    assert info["n_planned"] == 4 and info["n_not_done"] == 0
    df = pd.read_csv(out / "per_run.csv")
    assert bool(df["curve_complete"].all()) and np.isfinite(df["A"]).all() and np.isfinite(df["A_dev_online"]).all()
    tab = pd.read_csv(out / "criterion_e2.csv")
    assert set(tab["arm"]) == {"fb_r", "fb_rg", "fb_rx", "DECISION"} and (tab[tab["q"].astype(str) == "50"]["n_pairs"].dropna() == 1).all()
    last = tab.iloc[-1]
    assert last["arm"] == "DECISION" and last["role"] == "decision_rule" and last["status"] == info["decision"]["text"]
    sw = pd.read_csv(out / "sampling_weights.csv")
    assert set(sw["arm"]) == {"fb_a0", "fb_r", "fb_rg", "fb_rx"} and len(sw) == 4 * 6
    for arm in ("fb_a0", "fb_r", "fb_rg"):
        s = sw[(sw["arm"] == arm) & sw["mass_tail"].notna()]
        assert np.allclose(s[["mass_tail", "mass_near", "mass_mid"]].to_numpy(), [0.5, 0.35, 0.15], atol=1e-12)
    assert sw[sw["arm"] == "fb_rx"]["mass_near"].min() < 0.35 - 1e-3
    assert sw[(sw["arm"] == "fb_a0")]["tv_from_alpha0"].max() < 1e-12 and sw[(sw["arm"] == "fb_r")]["tv_from_alpha0"].max() > 1e-3
    assert int(sw["p_digest_next"].isna().sum()) == 4                                           # the last check of each run has no next update
    bins = pd.read_csv(out / "sampling_weights_bins_seed_mean.csv")                              # review finding R4-09
    assert set(bins["kind"]) == {"rho_bar", "gain_bar", "probs_next"} and set(bins["arm"]) == {"fb_a0", "fb_r", "fb_rg", "fb_rx"}
    pn = bins[(bins["kind"] == "probs_next") & (bins["arm"] == "fb_r") & bins["seed_mean"].notna()]
    assert np.allclose(pn.groupby("update")["seed_mean"].sum(), 1.0, atol=1e-12) and set(pn["stratum"]) == {"tail", "near", "mid"}
    assert np.allclose(pn.groupby(["update", "stratum"])["seed_mean"].sum().unstack()[["tail", "near", "mid"]].to_numpy(),
                       [0.5, 0.35, 0.15], atol=1e-12)
    assert int(df["n_final_invalid"].sum()) == 0 and df["n_final_invalid"].notna().all()                # review finding R4-11
    sh = pd.read_csv(out / "strata_shares_by_update.csv")
    assert len(sh) == 4 * 60 and np.allclose(sh[["share_tail", "share_near", "share_mid"]].sum(axis=1), 1.0)
    for fig in ("eta_curves_log.png", "eta_curves_linear.png", "area_paired.png", "rho_bar_heatmaps.png", "gain_bar_heatmaps.png", "strata_shares.png"):
        assert (out / "figures" / fig).stat().st_size > 1000, fig
    blind, bdec = B2.recompute(df, pd.read_csv(curves / "curves_all.csv"), json.load(open(crit_p)))
    assert B2.compare(blind, bdec, str(out / "criterion_e2.csv"), str(out / "analysis_info.json")) == []
    # the blind tool re-derives the areas from the curve file
    a_blind = B2.area_of(pd.read_csv(curves / "curves_all.csv"), "fb_r", 50, 10511, json.load(open(crit_p)))
    assert abs(a_blind - float(df[(df["arm"] == "fb_r")]["A"].iloc[0])) < 1e-15


def test_e1_failure_diagnostics_leave_a_trace_for_missing_files(tmp_path):
    """A failed run whose files are missing is a row with a note, never a silent skip (review finding R4-12)."""
    fail = pd.DataFrame([dict(arm="relu_st_s16", q=50, seed=10511, run_dir=str(tmp_path / "nope")),
                         dict(arm="lrsm_st_s16", q=60, seed=10512, run_dir=str(tmp_path / "also_nope")),
                         dict(arm="t1_st_s16", q=60, seed=10513, run_dir=str(tmp_path / "t1_nope"))])
    traj = E1.failure_trajectories(fail, None)
    assert list(traj["seed"]) == [10511, 10512, 10513] and (traj["note"] == "ms_checks_stage2.csv missing").all()
    assert list(traj.columns) == ["arm", "q", "seed", "local", "e2_at_0", "R0", "stage2_rmse_pos_over_g2_0", "stage2_peak_rel_err_abs",
                                  "sym_err_max", "note"]
    units = E1.failure_units_table(fail, tmp_path)
    assert list(units["arm"]) == ["relu_st_s16", "lrsm_st_s16"] and units["note"].str.len().gt(0).all()      # t1 has no units to report
    # a check table without curve rows: the symmetry error is missing and says so
    d = tmp_path / "run"
    d.mkdir()
    pd.DataFrame({"local": [25, 50], "e2_at_0": [1.0, 2.0], "R0": [0.1, 0.2], "stage2_rmse_pos_over_g2_0": [0.3, 0.4],
                  "stage2_peak_rel_err_abs": [0.5, 0.6]}).to_csv(d / "ms_checks_stage2.csv", index=False)
    f2 = pd.DataFrame([dict(arm="relu_st_s16", q=50, seed=10511, run_dir=str(d))])
    cv = pd.DataFrame({"arm": ["relu_st_s16"] * 2, "q": [50, 50], "seed": [99, 99], "local": [25, 50], "final_stage2_sym_err_max": [7.0, 8.0]})
    t2 = E1.failure_trajectories(f2, cv)
    assert list(t2["e2_at_0"]) == [1.0, 2.0] and t2["sym_err_max"].isna().all() and "no offline curve rows" in t2["note"].iloc[0]
    cv["seed"] = 10511
    assert list(E1.failure_trajectories(f2, cv)["sym_err_max"]) == [7.0, 8.0]


def test_the_unit_activity_default_grid_is_the_development_tier():
    from envs.curriculum_env import GameSpec
    from utils.dp_br_verifier import DEV_CONFIG
    spec = GameSpec(w_h=6.5, w_l=3.0, k=0.001, q=50.0, T=2, e_min=0.0, e_max=100.0)
    rng = np.random.default_rng(1)
    arrays = {"actor.l1.weight": rng.standard_normal((64, 2)).astype(np.float32), "actor.l1.bias": rng.standard_normal(64).astype(np.float32),
              "actor.l2.weight": rng.standard_normal((64, 64)).astype(np.float32), "actor.l2.bias": rng.standard_normal(64).astype(np.float32),
              "actor_variant": np.asarray("relu")}
    a, b = E1.unit_activity(arrays, spec), E1.unit_activity(arrays, spec, DEV_CONFIG.state_step)
    assert np.array_equal(a["layer1"], b["layer1"]) and np.array_equal(a["layer2"], b["layer2"])
    assert not np.array_equal(a["layer1"], E1.unit_activity(arrays, spec, 40.0)["layer1"])


def test_blind_e2_gates_json_cross_check(tmp_path):
    def run(name: str, passes: bool, gmax: float) -> Path:
        d = tmp_path / name
        d.mkdir()
        json.dump({"G-A": {"pass": True}, "G-F": {"pass": passes}, "G-N": {"pass": True}, "G-S": {"pass": True},
                   "global_rng": {"status": "ok"}, "metric_values": {"gmax_final": gmax}}, open(d / "gates.json", "w"))
        return d
    d1, d2 = run("a", True, 0.0004), run("b", False, 0.0006)
    table = pd.DataFrame([dict(arm="fb_r", q=50, seed=10511, status="done", run_dir=str(d1), gates_all_pass=True, gmax_final=0.0004),
                          dict(arm="fb_a0", q=50, seed=10511, status="done", run_dir=str(d2), gates_all_pass=False, gmax_final=0.0006),
                          dict(arm="fb_rg", q=50, seed=10511, status="failed", run_dir=str(tmp_path / "gone"), gates_all_pass=False, gmax_final=np.nan)])
    assert B2.cross_check_gates_json(table) == []                                       # a failed run has no gates.json to read
    assert len(B2.cross_check_gates_json(table.assign(gates_all_pass=[False, False, False]))) == 1
    assert len(B2.cross_check_gates_json(table.assign(gmax_final=[0.0004, 0.0007, np.nan]))) == 1


def test_the_preamble_tool_reproduces_the_prompts_numbers():
    import r4_preamble as P
    r1, r3 = ROOT / "results" / "ms_r1", ROOT / "results" / "ms_r3"
    if not (r1 / "pilot").exists():
        pytest.skip("results/ms_r1 is not in this checkout")
    red = P.reductions(P.areas_table(r1))
    assert len(red) == 8 and bool(red["matches_prompt_to_3dp"].all())
    sh = P.shares_table(r1, r3)
    assert len(sh) == 6 and bool(sh["matches_prompt_to_3dp"].all())
    pw = P.power_numbers()
    assert abs(pw["cp95_upper_0_of_40"] - 0.0881) < 5e-5 and abs(pw["pass_all_20_at_rate_0.1"] - 0.1216) < 5e-5
    cost = P.cost_numbers(r1)
    assert cost["n_checks"] == 11520 and cost["n_updates"] == 288000 and abs(cost["verifier_sec_median"] - 0.0165) < 1e-4
