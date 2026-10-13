"""MS-R4 post-launch checks (tools/ms/r4_launch_checks.py) on a reduced genuine wave and on tampered copies.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r4_launch_checks.py -p no:cacheprovider -q

The fixture runs the real runner (reduced budgets of ``ms_r4_helpers.reduced_r4``: E2 arms 60 + 20 updates, E1 arms 100 + 20
with the ramp over 61-80, checks every 10, exports every 5) on all seven arms of one (q, seed), four runs at a time. The checks
run on that real output; the tests then break copies of it one way at a time.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ms_r4_helpers as H  # noqa: E402,I001
from ms_r4_helpers import run_dir  # noqa: E402

import launch_checks as LC1  # noqa: E402
import r3_launch_checks as R3C  # noqa: E402
import r4_configs as rc  # noqa: E402
import r4_launch_checks as RC  # noqa: E402

Q, SEED = 50, 10511


@pytest.fixture(scope="module")
def wave(tmp_path_factory) -> Path:
    return H.get_wave(tmp_path_factory)


def _check(root: Path, **kw: Any) -> Dict[str, Any]:
    mp = pytest.MonkeyPatch()
    mp.setattr(LC1, "WEIGHTS_EVERY", 5)
    try:
        args = dict(ramp=H.E1_RAMP, fb25_upto=10, x2_upto=41, export_every=5)
        args.update(kw)
        return RC.check_pilot(root, [Q], [SEED], "abc1234", **args)
    finally:
        mp.undo()


def _copy(wave: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "w"
    shutil.copytree(wave, dst)
    return dst


def _edit_npz(path: Path, fn: Callable[[Dict[str, np.ndarray]], None]) -> None:
    z = dict(np.load(path))
    fn(z)
    np.savez(path, **z)


def _edit_json(path: Path, fn: Callable[[Any], None]) -> None:
    d = json.load(open(path))
    fn(d)
    json.dump(d, open(path, "w"))


def _edit_csv(path: Path, fn: Callable[[list], None]) -> None:
    rows = list(csv.DictReader(open(path)))
    cols = list(rows[0])
    fn(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _ulp(key: str) -> Callable[[Dict[str, np.ndarray]], None]:
    def fn(z: Dict[str, np.ndarray]) -> None:
        a = z[key].copy()
        i = int(np.flatnonzero(np.isfinite(a.ravel()))[0])
        a.flat[i] = np.nextafter(a.flat[i], np.array(np.inf, dtype=a.dtype))
        z[key] = a
    return fn


def _labels(root: Path, arm: str) -> np.ndarray:
    """Stratum code per bin (0 tail, 1 near-tie, 2 middle) of the terminal stage of a run."""
    from envs.curriculum_env import GameSpec, StartSampler
    cfg = json.load(open(run_dir(root, arm) / "run_config.json"))
    g = cfg["record"]["game"]
    sp = StartSampler(GameSpec(**{k: g[k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")}), 10.0)
    return sp.stratum_labels(2, float(cfg["start_weights"]["near_tie_half_width"]))


def _fails(res: Dict[str, Any], check: str, arm: str) -> bool:
    return any(not (x.get("pass") or x.get("ok")) for x in res[check] if x.get("arm") == arm)


# ---------------------------------------------------------------------------------------------- the genuine wave
def test_the_genuine_reduced_wave_passes_every_check(wave):
    res = _check(wave)
    assert res["all_ok"] is True, {k: v for k, v in res["summary"].items()}
    assert res["summary"] == {"scale_ok": 7, "scale_n": 7, "strata_masses_ok": 7, "strata_masses_n": 7, "sampling_weights_ok": 4,
                              "sampling_weights_n": 4, "C_INIT_pass": 1, "C_INIT_n": 1, "C_FB25_pass": 3, "C_FB25_n": 3,
                              "C_X2_pass": 1, "C_X2_n": 1}
    assert res["expected"] == {"scale_n": 7, "strata_masses_n": 7, "sampling_weights_n": 4, "C_INIT_n": 1, "C_FB25_n": 3, "C_X2_n": 1}
    assert res["within_arms_masses_exact"] is True
    for x in res["C_FB25"] + res["C_X2"]:
        assert x["ALL"] is True and x["first_export_after_differs"] is True and x["first_difference"] is None
    assert res["C_INIT"][0]["n_with_digest"] == 3 and res["C_INIT"][0]["all_arms"] is True
    assert res["base"]["all_ok"] is True and res["base"]["n_runs"] == 7


def test_the_strata_shares_are_reported_for_every_arm(wave):
    res = _check(wave)
    rep = res["strata_shares_reported"]
    assert set(rep) == set(rc.ARMS)
    for arm in ("fb_a0", "fb_r", "fb_rg"):                                    # the design shares, up to sampling noise
        m = next(x for x in res["strata_masses"] if x["arm"] == arm)
        assert max(m["max_dev"]) <= 1e-12 and np.allclose(m["mean_probability_share"], m["lambda"], atol=1e-12)
    m = next(x for x in res["strata_masses"] if x["arm"] == "fb_rx")
    assert m["ok_tail"] and not m["ok_within"]


def test_the_cli_writes_the_json_and_returns_zero(wave, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(LC1, "WEIGHTS_EVERY", 5)
    out = tmp_path / "launch_checks.json"
    monkeypatch.setattr(RC, "RAMP", H.E1_RAMP)
    # the CLI has the real-pilot defaults for the prefix lengths; only the tool-level check_pilot takes the reduced ones
    rc_code = RC.main(["--mode", "pilot", "--root", str(wave), "--code-commit", "abc1234", "--out", str(out), "--qs", str(Q),
                       "--seeds", str(SEED), "--arms", "fb_a0", "fb_r"])
    data = json.load(open(out))
    assert data["mode"] == "pilot" and data["arms"] == ["fb_a0", "fb_r"] and "C_FB25" in data
    assert rc_code in (0, 2)


# ---------------------------------------------------------------------------------------------- C-FB25
def test_c_fb25_fails_on_a_changed_export_before_the_first_check(wave, tmp_path):
    w = _copy(wave, tmp_path)
    _edit_npz(run_dir(w, "fb_r") / "weights" / "u00010.npz", _ulp("actor.l1.weight"))
    res = _check(w)
    assert res["all_ok"] is False and _fails(res, "C_FB25", "fb_r")
    assert not _fails(res, "C_FB25", "fb_rg") and not _fails(res, "C_FB25", "fb_rx")
    assert "weight_exports" in next(x for x in res["C_FB25"] if x["arm"] == "fb_r")["first_difference"]


def test_c_fb25_fails_on_a_changed_per_update_value(wave, tmp_path):
    w = _copy(wave, tmp_path)
    _edit_csv(run_dir(w, "fb_rg") / "ms_updates.csv", lambda rows: rows[3].__setitem__("mean_effort", "12.5"))
    res = _check(w)
    assert _fails(res, "C_FB25", "fb_rg") and not _fails(res, "C_FB25", "fb_r")


def test_c_fb25_excepts_the_alpha_column_and_the_next_digest(wave, tmp_path):
    """The two excepted columns may differ; nothing else may."""
    w = _copy(wave, tmp_path)
    _edit_csv(run_dir(w, "fb_r") / "ms_updates.csv", lambda rows: rows[2].__setitem__("alpha", "0.123"))
    _edit_csv(run_dir(w, "fb_r") / "ms_checks_stage2.csv", lambda rows: rows[0].__setitem__("p_digest_next", "deadbeefdeadbeef"))
    assert not _fails(_check(w), "C_FB25", "fb_r")
    _edit_csv(run_dir(w, "fb_r") / "ms_checks_stage2.csv", lambda rows: rows[0].__setitem__("p_digest", "deadbeefdeadbeef"))
    assert _fails(_check(w), "C_FB25", "fb_r")                                   # the digest of the probabilities used is not excepted


def test_c_fb25_needs_the_next_export_to_differ(wave, tmp_path):
    w = _copy(wave, tmp_path)
    shutil.copy(run_dir(w, "fb_a0") / "weights" / "u00015.npz", run_dir(w, "fb_rg") / "weights" / "u00015.npz")
    res = _check(w)
    x = next(x for x in res["C_FB25"] if x["arm"] == "fb_rg")
    assert x["ALL"] is True and x["first_export_after_differs"] is False and x["pass"] is False and res["all_ok"] is False


# ---------------------------------------------------------------------------------------------- C-X2
def test_c_x2_fails_on_a_changed_export_or_learning_rate(wave, tmp_path):
    w = _copy(wave, tmp_path)
    _edit_npz(run_dir(w, "fb_rx") / "weights" / "u00035.npz", _ulp("critic.out.bias"))
    res = _check(w)
    assert res["summary"]["C_X2_pass"] == 0 and res["all_ok"] is False
    w2 = _copy(wave, tmp_path / "b")
    _edit_csv(run_dir(w2, "t1_st_s16") / "ms_updates.csv", lambda rows: [r.__setitem__("lr", "0.000299") for r in rows
                                                                          if r["stage"] == "2" and r["local"] == "30"])
    assert _check(w2)["summary"]["C_X2_pass"] == 0


def test_c_x2_requires_the_exports_after_the_prefix_to_differ(wave, tmp_path):
    w = _copy(wave, tmp_path)
    shutil.copy(run_dir(w, "t1_st_s16") / "weights" / "u00045.npz", run_dir(w, "fb_rx") / "weights" / "u00045.npz")
    res = _check(w, x2_upto=41)
    x = res["C_X2"][0]
    assert x["ALL"] is True and x["first_export_after"] == 45 and x["first_export_after_differs"] is False and not x["pass"]


# ---------------------------------------------------------------------------------------------- strata masses, sampling weights
def test_the_strata_mass_check_catches_a_shifted_near_tie_mass_in_a_within_arm_only(wave, tmp_path):
    w = _copy(wave, tmp_path)

    def shift(z: Dict[str, np.ndarray]) -> None:
        pt = z["probs_table"].copy()
        from envs.curriculum_env import GameSpec, StartSampler
        cfg = json.load(open(run_dir(w, "fb_r") / "run_config.json"))
        g = cfg["record"]["game"]
        sp = StartSampler(GameSpec(**{k: g[k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")}), 10.0)
        lab = sp.stratum_labels(2, 20.0)
        i, j = int(np.flatnonzero(lab == 1)[0]), int(np.flatnonzero(lab == 2)[0])
        pt[-1, i] -= 0.01
        pt[-1, j] += 0.01
        z["probs_table"] = pt
    _edit_npz(run_dir(w, "fb_r") / "ms_binmaps_stage2.npz", shift)
    res = _check(w)
    assert _fails(res, "strata_masses", "fb_r") and res["all_ok"] is False
    w2 = _copy(wave, tmp_path / "b")
    _edit_npz(run_dir(w2, "fb_rx") / "ms_binmaps_stage2.npz", shift)            # fb_rx: only the tail mass is deterministic
    assert not _fails(_check(w2), "strata_masses", "fb_rx")


def test_the_tail_mass_check_catches_a_shifted_tail_mass_in_every_arm(wave, tmp_path):
    w = _copy(wave, tmp_path)

    def shift(z: Dict[str, np.ndarray]) -> None:
        pt = z["probs_table"].copy()
        lab = _labels(w, "t1_st_s16")
        pt[0, int(np.flatnonzero(lab == 0)[0])] += 1e-6                          # a tail bin gains ...
        pt[0, int(np.flatnonzero(lab == 1)[0])] -= 1e-6                          # ... what a near-tie bin loses
        z["probs_table"] = pt
    _edit_npz(run_dir(w, "t1_st_s16") / "ms_binmaps_stage2.npz", shift)
    assert _fails(_check(w), "strata_masses", "t1_st_s16")


def test_the_sampling_weights_check_compares_probs_next_with_the_digest(wave, tmp_path):
    w = _copy(wave, tmp_path)
    _edit_csv(run_dir(w, "fb_r") / "ms_checks_stage2.csv", lambda rows: rows[2].__setitem__("p_digest_next", "0" * 16))
    res = _check(w)
    assert _fails(res, "sampling_weights", "fb_r") and not _fails(res, "sampling_weights", "fb_rg")
    w2 = _copy(wave, tmp_path / "b")
    _edit_csv(run_dir(w2, "fb_rg") / "ms_checks_stage2.csv", lambda rows: rows[-1].__setitem__("p_digest_next", "0" * 16))
    assert _fails(_check(w2), "sampling_weights", "fb_rg")                         # the last check must have an empty digest
    w3 = _copy(wave, tmp_path / "c")

    def nudge(z: Dict[str, np.ndarray]) -> None:
        p = z["probs_next"].copy()
        p[1, 0] = np.nextafter(p[1, 0], 1.0)
        z["probs_next"] = p
    _edit_npz(run_dir(w3, "fb_r") / "ms_binmaps_stage2.npz", nudge)
    assert _fails(_check(w3), "sampling_weights", "fb_r")


def test_the_sampling_weights_of_a_cross_stratum_arm_are_reconstructed_from_the_table(wave):
    res = _check(wave)
    by_arm = {x["arm"]: x for x in res["sampling_weights"]}
    assert by_arm["fb_rx"]["source"] == "probs_table" and by_arm["fb_a0"]["source"] == "probs_table"
    assert by_arm["fb_r"]["source"] == "probs_next" and by_arm["fb_rg"]["source"] == "probs_next"
    assert all(x["n_checked"] == 5 and x["n_bad"] == 0 for x in by_arm.values())        # 6 checks, the last has no next update


# ---------------------------------------------------------------------------------------------- C-INIT, scale, missing runs
def test_c_init_fails_on_a_different_initial_state(wave, tmp_path):
    w = _copy(wave, tmp_path)
    _edit_json(run_dir(w, "lrsm_st_s16") / "manifest.json", lambda m: m.__setitem__("init_state_sha256", "0" * 64))
    res = _check(w)
    assert res["summary"]["C_INIT_pass"] == 0 and res["all_ok"] is False and res["C_INIT"][0]["n_distinct"] == 2


def test_the_scale_check_catches_a_wrong_applied_scale(wave, tmp_path):
    w = _copy(wave, tmp_path)
    _edit_csv(run_dir(w, "lrsm_st_s16") / "ms_updates.csv",
              lambda rows: [r.__setitem__("conc_scale", "3.0") for r in rows if r["stage"] == "2" and r["local"] == "90"])
    res = _check(w)
    assert _fails(res, "scale", "lrsm_st_s16") and not _fails(res, "scale", "t1_st_s16")
    w2 = _copy(wave, tmp_path / "b")
    _edit_csv(run_dir(w2, "fb_a0") / "ms_updates.csv", lambda rows: rows[5].__setitem__("conc_scale", "2.0"))
    assert _fails(_check(w2), "scale", "fb_a0")                                      # the E2 arms are unscaled throughout


def test_a_missing_run_is_a_failure_never_a_skip(wave, tmp_path):
    w = _copy(wave, tmp_path)
    shutil.rmtree(run_dir(w, "fb_rg"))
    res = _check(w)
    assert res["all_ok"] is False and res["base"]["all_ok"] is False
    assert res["summary"]["scale_ok"] == 6 and res["expected"]["scale_n"] == 7
    assert any(not x["ok"] and x["arm"] == "fb_rg" for x in res["scale"])


# ---------------------------------------------------------------------------------------------- C-MS6
def test_c_ms6_passes_for_identical_controls_and_fails_on_one_ulp(wave, tmp_path):
    ref = _copy(wave, tmp_path / "ref")
    mp = pytest.MonkeyPatch()
    mp.setattr(LC1, "WEIGHTS_EVERY", 5)
    try:
        res = RC.check_cms6(wave, ref, [Q], [SEED], "abc1234")
        assert res["all_ok"] is True and res["summary"] == {"C_MS6_pass": 2, "C_MS6_n": 2} and res["expected"] == {"C_MS6_n": 2}
        for r in res["C_MS6"]:
            assert r["ALL"] is True and r["compared"]["exports"] == 24 and r["compared"]["gate_values"] > 20
        _edit_npz(run_dir(ref, "relu_st_s16") / "weights" / "u00050.npz", _ulp("actor.out.weight"))
        bad = RC.check_cms6(wave, ref, [Q], [SEED], "abc1234")
        assert bad["all_ok"] is False and bad["summary"]["C_MS6_pass"] == 1
        assert next(r for r in bad["C_MS6"] if r["arm"] == "relu_st_s16")["first_difference"].startswith("weight_exports")
        shutil.rmtree(run_dir(ref, "t1_st_s16"))
        assert RC.check_cms6(wave, ref, [Q], [SEED], "abc1234")["summary"]["C_MS6_pass"] == 0                  # a missing reference fails
    finally:
        mp.undo()


def test_the_cms6_arms_are_the_two_controls():
    assert rc.CMS6_ARMS == ("t1_st_s16", "relu_st_s16") and rc.SEEDS_CMS6 == tuple(range(10501, 10511))
    assert R3C.GATES_IGNORED and "arm" in R3C.GATES_IGNORED
