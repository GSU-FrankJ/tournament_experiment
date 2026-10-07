"""Unit tests of the descriptive additions A3 of ``tools/ms/r1_analysis.py`` (fast, no pipeline run).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_analysis_a3.py -p no:cacheprovider -q

Each helper is checked on a few small synthetic files with known answers: R0 = r_2(0)/s_2 against the repo's own
definition in ``utils/ms_residual.py`` and the linearised factor of the protocol, the stratum / side split of the
closed-form error and the first-order residual (the node d = 0 belongs to neither side), the |S_t| composition by
the stratum labels of ``envs.curriculum_env.StartSampler`` (as ``tools/ms/launch_checks.py`` builds them), and the
stage-1 firing streak with and without a check at R_1 = 0 exactly. The CLI outputs of these helpers on the
synthetic world are tested in ``test_ms_analysis.py``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import r1_analysis as R  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from utils.ms_residual import stage_diag  # noqa: E402

PROTO = json.loads((ROOT / "protocols" / "v2_T2_locked_v2_0.json").read_text())


def _spec(q: float) -> GameSpec:
    g = PROTO["records"][str(int(q))]["game"]
    return GameSpec(**{k: g[k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")})


def _verifier_result(d: np.ndarray, e_hat: np.ndarray, a_dev: np.ndarray) -> Any:
    """The part of a ``VerifierResult`` that ``stage_diag`` reads (terminal stage = 2 of T = 2)."""
    return SimpleNamespace(valid=True, dw=4.0, full_delta_max={2: 0.01},
                           stages={2: SimpleNamespace(d_grid=d, e_hat=e_hat, a_dev=a_dev)})


# ====================================================================== R0 = r_2(0)/s_2
def test_R0_is_the_node_zero_value_of_the_D3_residual_map_of_utils_ms_residual():
    """``node0_residual`` is ``(r_2(0), s_2)`` of ``stage_diag``: ``s = a_dev(0)``, ``r = |e_hat - a_dev|``, so
    R0 = r/s equals the ratio at d = 0 of the map that defines R_t = max over the non-tail region."""
    q = 50
    d = np.arange(-200.0, 200.0 + 1e-9, 2.0)
    a_dev = np.where(np.abs(d) < 2 * q, 66.0 * (1.0 - np.abs(d) / (2 * q)), 0.0)
    for r0, other in ((0.08, 0.01), (0.002, 0.03)):        # residual maximal at d = 0, and maximal elsewhere
        delta = np.where(np.abs(d) < 2 * q, other * 66.0, 0.0) * np.where(d < 0, -1.0, 1.0)
        delta[d == 0.0] = r0 * 66.0
        e_hat = a_dev + delta
        diag = stage_diag(_verifier_result(d, e_hat, a_dev), _spec(q), 2, {"valid": True, "max_std_norm": 0.01})
        r, s = R.node0_residual(d, e_hat, a_dev)
        assert s == diag.s == a_dev[d == 0.0][0]
        ratio0 = r / s
        assert ratio0 == pytest.approx(r0, abs=1e-12)
        if r0 > other:
            assert diag.R == ratio0 and diag.argmax_d == 0.0           # R_t is exactly R0 when the maximum is at 0
        else:
            assert ratio0 < diag.R and diag.R == pytest.approx(other, abs=1e-12)


def test_r0_cols_from_freeze_files_both_tiers_and_nan_when_R0_is_zero_or_missing(tmp_path):
    d = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    a_dev = np.array([10.0, 20.0, 40.0, 20.0, 10.0])
    e_hat = a_dev + np.array([0.5, 0.5, -2.0, 0.5, 0.5])
    fin, dev = tmp_path / "fin.npz", tmp_path / "dev.npz"
    np.savez(fin, v_t2_d_grid=d, v_t2_e_hat=e_hat, v_t2_a_dev=a_dev)
    np.savez(dev, v_t2_d_grid=d, v_t2_e_hat=a_dev + np.array([0.0, 0.0, 4.0, 0.0, 0.0]), v_t2_a_dev=a_dev)
    an: List[str] = []
    out = R.r0_cols((fin, dev), 0.10, an)
    assert out["t2_R0_final"] == pytest.approx(2.0 / 40.0) and out["t2_R0_dev"] == pytest.approx(0.1)
    assert out["t2_peak_over_R0_final"] == pytest.approx(0.10 / 0.05)
    assert out["t2_peak_over_R0_dev"] == pytest.approx(1.0)
    assert an == [] and list(out) == R.R0_COLS
    zero = tmp_path / "zero.npz"                                          # R0 == 0: the ratio is NaN, R0 stays 0
    np.savez(zero, v_t2_d_grid=d, v_t2_e_hat=a_dev, v_t2_a_dev=a_dev)
    z = R.r0_cols((zero, dev), 0.10, an)
    assert z["t2_R0_final"] == 0.0 and np.isnan(z["t2_peak_over_R0_final"]) and z["t2_peak_over_R0_dev"] == 1.0
    miss = R.r0_cols((tmp_path / "nope.npz", dev), 0.10, an)
    assert np.isnan(miss["t2_R0_final"]) and np.isnan(miss["t2_peak_over_R0_final"]) and miss["t2_R0_dev"] == 0.1
    assert any("nope.npz missing" in a for a in an)
    nan_peak = R.r0_cols((fin, dev), float("nan"), [])
    assert nan_peak["t2_R0_final"] == pytest.approx(0.05) and np.isnan(nan_peak["t2_peak_over_R0_final"])
    shifted = tmp_path / "shift.npz"                                      # a grid without a zero node: undefined
    np.savez(shifted, v_t2_d_grid=d + 1.0, v_t2_e_hat=e_hat, v_t2_a_dev=a_dev)
    assert np.isnan(R.r0_cols((shifted, dev), 0.1, [])["t2_R0_final"])


def test_linearised_factor_from_the_protocol_game():
    f50 = R.linearised_factor(PROTO["records"]["50"]["game"])
    f60 = R.linearised_factor(PROTO["records"]["60"]["game"])
    assert round(f50, 3) == 1.700 and round(f60, 3) == 1.486
    g = PROTO["records"]["50"]["game"]
    dw, k = g["w_h"] - g["w_l"], g["k"]
    assert f50 == pytest.approx(1.0 + dw / (4 * 50 ** 2) / (2 * k), abs=1e-12)
    assert R.CALIB_MEDIAN_PEAK_OVER_R0 == {50: 1.65, 60: 1.45}


def test_freeze_files_of_the_three_kinds_of_run(tmp_path):
    assert R.freeze_files(tmp_path, "MS_rule") == (tmp_path / "freeze_stage2_final.npz",
                                                   tmp_path / "freeze_stage2_development.npz")
    assert R.freeze_files(tmp_path, "MS_base2400")[0].name == "freeze_stage2_final.npz"
    assert R.freeze_files(tmp_path, "parents_A") == (tmp_path / "final_final.npz", tmp_path / "final_development.npz")
    assert R.freeze_files(tmp_path, "rehearsal_v2_0") == (tmp_path / "gateA_final.npz",
                                                          tmp_path / "gateA_development.npz")


# ====================================================================== strata
def _toy_arrays(q: float) -> Dict[str, np.ndarray]:
    """Recovery grid -200..200 step 0.5 with e2 - g2 = d / 100 (so the error at a node is its own gap / 100) and a
    verifier grid step 10 (nodes at |d| = 20 and 2q exactly) with e_hat - a_dev = d / 10 against s = a_dev(0) = 50."""
    rd = np.arange(-200.0, 200.0 + 1e-9, 0.5)
    g2 = np.full(rd.size, 70.0)
    vd = np.arange(-200.0, 200.0 + 1e-9, 10.0)
    a_dev = np.full(vd.size, 50.0)
    return {"recovery_d_grid": rd, "recovery_g2": g2, "recovery_e2": g2 + rd / 100.0,
            "v_t2_d_grid": vd, "v_t2_a_dev": a_dev, "v_t2_e_hat": a_dev + vd / 10.0}


def test_strata_split_boundaries_and_the_node_d0_in_neither_side():
    q = 50.0
    rows = {(r["stratum"], r["side"]): r for r in R.strata_rows(_toy_arrays(q), q)}
    assert len(rows) == 6 and sum(r["n_nodes"] for r in rows.values()) == 800           # d = 0 is in no cell
    # near |d| < 20: d = -19.5 ... -0.5 (39 nodes); mid 20 <= |d| < 100: -99.5 ... -20 (160); tail |d| >= 100: 201
    assert [rows[(s, "d<0")]["n_nodes"] for s in R.STRATA] == [39, 160, 201]
    assert [rows[(s, "d>0")]["n_nodes"] for s in R.STRATA] == [39, 160, 201]
    # verifier nodes of step 10: near -10, 10 (|d| = 20 is middle); mid -90 ... -20 (8); tail -200 ... -100 (11)
    assert [rows[(s, "d<0")]["n_verifier_nodes"] for s in R.STRATA] == [1, 8, 11]
    assert [rows[(s, "d>0")]["n_verifier_nodes"] for s in R.STRATA] == [1, 8, 11]
    near_neg = rows[("near", "d<0")]                                                 # d in [-19.5, -0.5]
    d = np.arange(-19.5, -0.5 + 1e-9, 0.5)
    assert near_neg["err_mean"] == pytest.approx((d / 100).mean(), abs=1e-12)
    assert near_neg["err_rmse"] == pytest.approx(np.sqrt(np.mean((d / 100) ** 2)), abs=1e-12)
    assert near_neg["err_max_abs"] == pytest.approx(0.195, abs=1e-12)
    assert near_neg["err_mean_rel"] == pytest.approx(near_neg["err_mean"] / 70.0, abs=1e-12)
    assert near_neg["err_max_abs_rel"] == pytest.approx(0.195 / 70.0, abs=1e-12)
    mid_pos = rows[("mid", "d>0")]                                                   # d in [20, 99.5]
    assert mid_pos["err_max_abs"] == pytest.approx(0.995, abs=1e-12) and mid_pos["err_mean"] > 0
    tail_neg = rows[("tail", "d<0")]                                                 # d in [-200, -100]
    assert tail_neg["err_mean"] == pytest.approx(-1.5, abs=1e-12) and tail_neg["err_max_abs"] == pytest.approx(2.0)
    # max r/s over the verifier nodes: |d/10| / 50, the node at the cell boundary included on its own side
    assert rows[("near", "d<0")]["max_r_over_s"] == pytest.approx(1.0 / 50.0)          # d = -10
    assert rows[("mid", "d<0")]["max_r_over_s"] == pytest.approx(9.0 / 50.0)           # d = -90 (-20 is also mid)
    assert rows[("tail", "d>0")]["max_r_over_s"] == pytest.approx(20.0 / 50.0)         # d = 200
    assert rows[("tail", "d<0")]["max_r_over_s"] == pytest.approx(20.0 / 50.0)
    # another q moves the tail boundary: q = 60 puts |d| in [100, 120) into the middle stratum
    rows60 = {(r["stratum"], r["side"]): r for r in R.strata_rows(_toy_arrays(60.0), 60.0)}
    assert [rows60[(s, "d<0")]["n_nodes"] for s in R.STRATA] == [39, 200, 161]
    assert rows60[("tail", "d>0")]["n_verifier_nodes"] == 9                           # d = 120 ... 200


def test_strata_rows_without_a_zero_node_or_with_zero_s_have_nan_relative_columns():
    z = _toy_arrays(50.0)
    z["recovery_d_grid"] = z["recovery_d_grid"] + 0.25                                # no node at d = 0
    rows = R.strata_rows(z, 50.0)
    assert all(np.isnan(r["err_mean_rel"]) and np.isfinite(r["err_mean"]) for r in rows)
    z2 = _toy_arrays(50.0)
    z2["v_t2_a_dev"] = np.zeros_like(z2["v_t2_a_dev"])                                # s = 0: r / s undefined
    assert all(np.isnan(r["max_r_over_s"]) for r in R.strata_rows(z2, 50.0))


def test_stratum_codes_follow_the_run_q():
    d = np.array([-100.0, -99.9, -20.0, -19.9, 0.0, 19.9, 20.0, 99.9, 100.0, 120.0])
    assert list(R.stratum_codes(d, 50.0)) == ["tail", "mid", "mid", "near", "near", "near", "mid", "mid", "tail",
                                             "tail"]
    assert list(R.stratum_codes(d, 60.0))[-3:] == ["mid", "mid", "tail"]


# ====================================================================== |S_t| composition
def _cfg(q: int, hw: Any = 20.0) -> Dict[str, Any]:
    g = PROTO["records"][str(q)]["game"]
    sw: Dict[str, Any] = {"scheme": "stratified_priority"}
    if hw is not None:
        sw["near_tie_half_width"] = hw
    return {"record": {"game": g, "protocol": {"es_bin_width": 10}}, "start_weights": sw}


def test_stage2_labels_are_the_start_sampler_labels_and_S_composition_counts_them():
    for q in (50, 60):
        cfg = _cfg(q)
        an: List[str] = []
        lab = R.stage2_labels(cfg, an)
        sampler = StartSampler(_spec(q), 10.0)                      # the call tools/ms/launch_checks.py makes
        assert an == [] and (lab == sampler.stratum_labels(2, 20.0)).all()
        n_bins = 40 if q == 50 else 44
        assert lab.size == n_bins and int((lab == 0).sum()) == 20 and int((lab == 1).sum()) == 4
        near = np.flatnonzero(lab == 1)
        assert list(near) == ([18, 19, 20, 21] if q == 50 else [20, 21, 22, 23])
    lab = R.stage2_labels(_cfg(50), [])
    assert R.s_composition([10, 11, 12], lab, []) == (0.0, 3.0, 0.0)
    assert R.s_composition(list(range(10, 29)), lab, []) == (4.0, 15.0, 0.0)           # 19 bins: 4 near, 15 mid
    assert R.s_composition([0, 5, 19, 25, 39], lab, []) == (1.0, 1.0, 3.0)             # defensive: tail bins counted
    assert R.s_composition([], lab, []) == (0.0, 0.0, 0.0)
    an2: List[str] = []
    out = R.s_composition([3, 40], lab, an2)                                            # an index outside the bins
    assert all(np.isnan(x) for x in out) and "outside" in an2[0]


def test_stage2_labels_use_the_runs_half_width_default_20_and_report_unusable_configs():
    assert (R.stage2_labels(_cfg(50, None), []) == R.stage2_labels(_cfg(50, 20.0), [])).all()   # bin_balanced: default
    wide = R.stage2_labels(_cfg(50, 30.0), [])
    assert int((wide == 1).sum()) == 6                                                  # bins intersecting (-30, 30)
    an: List[str] = []
    assert R.stage2_labels({"record": {}}, an) is None and "unavailable" in an[0]


# ====================================================================== stage-1 streaks
def _checks(path: Path, r: List[float], local_step: int = 25) -> Path:
    pd.DataFrame({"stage": 1, "local": [local_step * (i + 1) for i in range(len(r))], "R": r}).to_csv(path, index=False)
    return path


def test_streak_with_and_without_an_exact_zero_check(tmp_path):
    f = _checks(tmp_path / "c.csv", [0.2, 0.1, 0.02, 0.0, 0.01, 0.015, 0.0, 0.02])      # locals 25 ... 200
    # fire at local 150 (6th check): the streak is checks 4, 5, 6 = (0.0, 0.01, 0.015): contains the zero
    a = R.streak_cols(f, {"fire_local": 150, "would_fire_local": None}, 3, [])
    assert a == {"t1_n_checks_R0": 2.0, "t1_streak_kind": "fire", "t1_streak_has_R0": True}
    # fire at local 125 (5th): checks 3, 4, 5 = (0.02, 0.0, 0.01): zero in the middle
    assert R.streak_cols(f, {"fire_local": 125}, 3, [])["t1_streak_has_R0"] is True
    # fire at local 100 (4th): checks 2, 3, 4: (0.1, 0.02, 0.0): the zero at the firing check
    assert R.streak_cols(f, {"fire_local": 100}, 3, [])["t1_streak_has_R0"] is True
    # fire at local 75 (3rd): checks 1, 2, 3 = (0.2, 0.1, 0.02): no zero, although later checks have one
    b = R.streak_cols(f, {"fire_local": 75}, 3, [])
    assert b["t1_streak_has_R0"] is False and b["t1_n_checks_R0"] == 2.0
    # the streak is exactly M checks: the zero at the 4th check is not in the streak ending at the 6th when M = 2
    assert R.streak_cols(f, {"fire_local": 150}, 2, [])["t1_streak_has_R0"] is False
    # the zero at the 7th check is outside the streak ending at the 6th
    g = _checks(tmp_path / "g.csv", [0.2, 0.1, 0.02, 0.03, 0.01, 0.015, 0.0])
    assert R.streak_cols(g, {"fire_local": 150}, 3, []) == {"t1_n_checks_R0": 1.0, "t1_streak_kind": "fire",
                                                            "t1_streak_has_R0": False}
    # a legacy arm: would_fire_local, labelled "would_fire"; a fire wins over a would-fire when both are recorded
    c = R.streak_cols(f, {"fire_local": None, "would_fire_local": 150}, 3, [])
    assert c["t1_streak_kind"] == "would_fire" and c["t1_streak_has_R0"] is True
    assert R.streak_cols(f, {"fire_local": 75, "would_fire_local": 150}, 3, [])["t1_streak_kind"] == "fire"


def test_streak_cols_no_streak_missing_rows_and_exactness(tmp_path):
    f = _checks(tmp_path / "c.csv", [0.2, 1e-300, 0.02, -0.0, 0.01])                    # tiny is not zero, -0.0 is
    none = R.streak_cols(f, {"fire_local": None, "would_fire_local": None}, 3, [])
    assert none["t1_n_checks_R0"] == 1.0 and none["t1_streak_kind"] == "" and np.isnan(none["t1_streak_has_R0"])
    an: List[str] = []
    off = R.streak_cols(f, {"fire_local": 90}, 3, an)                                  # not a row of the table
    assert off["t1_streak_kind"] == "" and np.isnan(off["t1_streak_has_R0"]) and "not a row" in an[0]
    first = R.streak_cols(f, {"fire_local": 25}, 3, [])                                # fewer than M rows before it
    assert first["t1_streak_kind"] == "fire" and first["t1_streak_has_R0"] is False
    an2: List[str] = []
    gone = R.streak_cols(tmp_path / "missing.csv", {"fire_local": 25}, 3, an2)
    assert np.isnan(gone["t1_n_checks_R0"]) and gone["t1_streak_kind"] == "" and "missing" in an2[0]
    nan_r = _checks(tmp_path / "n.csv", [float("nan"), 0.0, float("nan")])
    assert R.streak_cols(nan_r, {}, 3, [])["t1_n_checks_R0"] == 1.0                    # NaN is not an exact zero


def test_calibration_replay_counts_exact_zero_stage1_exports(tmp_path):
    cal = tmp_path / "cal" / "rehearsal_v2_0" / "q50"
    cal.mkdir(parents=True)
    rows = [dict(update=u, stage=st, tier=t, Delta=0.0, s=1.0, R=r, R_tail=0.0, C=0.0)
            for u, st, t, r in ((1625, 1, "dev", 0.0), (1650, 1, "dev", 0.01), (1675, 1, "dev", 0.0),
                                (1675, 1, "final", 0.0), (1600, 2, "dev", 0.0), (2200, 1, "dev", 0.03))]
    pd.DataFrame(rows).to_csv(cal / "seed10501.csv", index=False)
    out = R.calibration_d3(tmp_path / "cal", 50, 10501)
    assert out["t1_n_checks_R0"] == 2.0                      # development tier, stage 1 only
