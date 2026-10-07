"""MS-R2 post-launch checks (tools/ms/r2_launch_checks.py) on a reduced genuine wave and on tampered copies."""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import launch_checks as LC1  # noqa: E402
import ms_configs as mc  # noqa: E402
import r2_launch_checks as RC  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from test_ms_r2_runner import EPU, PARAMS, PROTO, reduced_nl  # noqa: E402
from test_ms_runner import make_cfg, small_params  # noqa: E402

FAKE_GIT = {"commit": "abc1234", "short": "abc1234", "dirty": False}
RAMP = (21, 40)
Q, SEED = 50, 10501


@pytest.fixture(scope="module")
def wave(tmp_path_factory):
    """The six reduced NL runs of one (q, seed) and the two reduced MS-R1-style reference runs."""
    root = tmp_path_factory.mktemp("r2wave")
    new, ref = root / "r2" / "pilot", root / "r1" / "pilot"
    mp = pytest.MonkeyPatch()
    mp.setattr(rms, "git_state", lambda: dict(FAKE_GIT))
    mp.setattr(LC1, "WEIGHTS_EVERY", 5)
    try:
        for arm in mc.R2_ARMS:
            d = new / f"q{Q}" / f"seed{SEED}" / arm
            d.mkdir(parents=True)
            assert rms.run_pipeline(reduced_nl(arm, str(d), q=Q, seed=SEED), str(d), "pytest", band_step=2.0) == 0
        d = ref / f"q{Q}" / f"seed{SEED}" / "MS_base2400"                   # MS_base2400-like: its LR window starts at 21
        d.mkdir(parents=True)
        prm = copy.deepcopy(PARAMS)
        prm["K"] = 10
        cfg = mc.build_config(PROTO, Q, SEED, "MS_base2400", str(d), prm)
        cfg["record"]["protocol"]["episodes_per_update"] = EPU
        cfg["record"]["protocol"]["weights_every"] = 5
        cfg["pipeline"]["budgets"] = {"2": 48, "1": 20}
        cfg["pipeline"]["lr_windows"] = {"2": [{"first": 21, "last": 48, "start": 3e-4, "end": 3e-5}],
                                         "1": [{"first": 1, "last": 20, "start": 3e-4, "end": 3e-5}]}
        assert rms.run_pipeline(cfg, str(d), "pytest", band_step=2.0) == 0
        d = ref / f"q{Q}" / f"seed{SEED}" / "MS_s35a5"                      # rule mode, stop and polishing impossible
        d.mkdir(parents=True)
        prm = small_params(2, K=10, block=20, cap=20, land=40, rho=0.0)   # cap 20: the landing starts at update 21
        prm["localized_fraction"] = 1e-9
        cfg = make_cfg(str(d), "MS_s35a5", params=prm, epu=EPU)
        cfg["record"]["protocol"]["weights_every"] = 5
        assert rms.run_pipeline(cfg, str(d), "pytest", band_step=2.0) == 0
    finally:
        mp.undo()
    return new, ref


def _check(new: Path, ref: Path, **kw):
    mp = pytest.MonkeyPatch()
    mp.setattr(LC1, "WEIGHTS_EVERY", 5)
    try:
        return RC.check_root(new, ref, [Q], [SEED], "abc1234", ramp=RAMP, export_every=5, **kw)
    finally:
        mp.undo()


def _copy(wave, tmp_path):
    new, ref = wave
    n2, r2 = tmp_path / "n", tmp_path / "r"
    shutil.copytree(new, n2)
    shutil.copytree(ref, r2)
    return n2, r2


def test_the_schedule_function_is_the_d2_table():
    assert [RC.schedule_value(j, 4.0) for j in (1, 2000, 2001)] == [1.0, 1.0, 1.0]
    assert RC.schedule_value(2002, 4.0) == pytest.approx(1.0 + 3.0 / 199.0)
    assert RC.schedule_value(2100, 16.0) == pytest.approx(1.0 + 15.0 * 99.0 / 199.0)
    assert [RC.schedule_value(j, 16.0) for j in (2200, 2201, 2800)] == [16.0, 16.0, 16.0]
    assert all(RC.schedule_value(j, 1.0) == 1.0 for j in (1, 2100, 2800))
    assert RC.arm_scale("NL_bb_s16") == 16.0 and RC.arm_scale("NL_st_s1") == 1.0
    with pytest.raises(ValueError):
        RC.arm_scale("MS_rule")


def test_a_genuine_wave_passes_every_check(wave):
    new, ref = wave
    r = _check(new, ref)
    s = r["summary"]
    assert r["all_ok"], json.dumps({k: v for k, v in r.items() if k in ("summary",)}, indent=1) + json.dumps(
        [x for x in r["C_NL"] + r["C_MS3"] + r["C_MS4"] if not x.get("pass")][:2], indent=1)[:1500]
    assert (s["scale_ok"], s["scale_n"], s["C_NL_pass"], s["C_NL_n"], s["C_MS3_pass"], s["C_MS4_pass"]) == (6, 6, 4, 4, 1, 1)
    assert r["C_MS4"][0]["identical_through_update"] == RAMP[0] and r["C_MS4"][0]["first_polish_local"] is None
    assert all(x["first_export_after_differs"] for x in r["C_NL"] + r["C_MS3"])


def test_a_tampered_applied_scale_or_exported_scale_fails(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    d = new / f"q{Q}" / f"seed{SEED}" / "NL_bb_s4"
    rows = list(csv.DictReader(open(d / "ms_updates.csv")))
    rows[45]["conc_scale"] = "3.9"
    with open(d / "ms_updates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    r = _check(new, ref)
    assert not r["all_ok"] and r["summary"]["scale_ok"] == 5
    new2, ref2 = _copy(wave, tmp_path / "b")
    f = new2 / f"q{Q}" / f"seed{SEED}" / "NL_st_s16" / "weights" / "u00050.npz"
    z = dict(np.load(f))
    z["conc_scale"] = np.asarray(8.0)
    np.savez(f, **z)
    r = _check(new2, ref2)
    s16 = [x for x in r["scale"] if x["arm"] == "NL_st_s16"][0]
    assert not s16["ok"] and s16["n_export_mismatches"] == 1


def test_a_run_with_a_set_scale_where_none_is_expected_fails(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    d = new / f"q{Q}" / f"seed{SEED}" / "NL_bb_s1"
    cfg = json.load(open(d / "run_config.json"))
    cfg["conc_scale_schedule"] = {"stage": 2, "local_first": 21, "local_last": 40, "scale_first": 1.0, "scale_last": 2.0}
    json.dump(cfg, open(d / "run_config.json", "w"))
    assert not _check(new, ref)["all_ok"]


def test_c_nl_finds_a_changed_export_series_value_or_stream_position(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    d = new / f"q{Q}" / f"seed{SEED}" / "NL_bb_s4"
    f = d / "weights" / "u00010.npz"
    z = dict(np.load(f))
    k = [k for k in z if k.startswith("actor.")][0]
    z[k] = z[k].copy()
    z[k].flat[0] += np.float32(1e-7)
    np.savez(f, **z)
    r = _check(new, ref)
    bad = [x for x in r["C_NL"] if x["sampler"] == "bb" and x["s"] == 4][0]
    assert not r["all_ok"] and not bad["pass"] and bad["weight_exports"] is False
    assert r["summary"]["C_NL_pass"] == 3
    new2, ref2 = _copy(wave, tmp_path / "b")
    d2 = new2 / f"q{Q}" / f"seed{SEED}" / "NL_st_s16"
    th = json.load(open(d2 / "train_history.json"))
    th["history"][10]["policy_loss"] += 1e-9
    json.dump(th, open(d2 / "train_history.json", "w"))
    r = _check(new2, ref2)
    assert [x for x in r["C_NL"] if x["sampler"] == "st" and x["s"] == 16][0]["train_history"] is False
    new3, ref3 = _copy(wave, tmp_path / "c")
    d3 = new3 / f"q{Q}" / f"seed{SEED}" / "NL_st_s4"
    rows = list(csv.DictReader(open(d3 / "ms_updates.csv")))
    rows[5]["rngpos_start"] = "0:0:0"
    with open(d3 / "ms_updates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    r = _check(new3, ref3)
    assert [x for x in r["C_NL"] if x["sampler"] == "st" and x["s"] == 4][0]["updates_csv_and_stream_positions"] is False


def test_c_nl_requires_the_first_export_after_the_ramp_start_to_differ(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    shutil.copy(new / f"q{Q}" / f"seed{SEED}" / "NL_bb_s1" / "weights" / "u00025.npz",
                new / f"q{Q}" / f"seed{SEED}" / "NL_bb_s16" / "weights" / "u00025.npz")
    r = _check(new, ref)
    x = [x for x in r["C_NL"] if x["sampler"] == "bb" and x["s"] == 16][0]
    assert x["ALL"] and not x["first_export_after_differs"] and not x["pass"] and not r["all_ok"]


def test_c_ms3_and_c_ms4_find_a_wrong_reference(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    d = ref / f"q{Q}" / f"seed{SEED}" / "MS_base2400"
    f = d / "weights" / "u00015.npz"
    z = dict(np.load(f))
    k = [k for k in z if k.startswith("actor.")][0]
    z[k] = z[k].copy()
    z[k].flat[0] += np.float32(1e-7)
    np.savez(f, **z)
    r = _check(new, ref)
    assert not r["all_ok"] and r["summary"]["C_MS3_pass"] == 0 and r["summary"]["C_NL_pass"] == 4
    new2, ref2 = _copy(wave, tmp_path / "b")
    d2 = ref2 / f"q{Q}" / f"seed{SEED}" / "MS_s35a5"
    rows = list(csv.DictReader(open(d2 / "ms_updates.csv")))
    rows[14]["rngpos_start"] = "0:0:0"                                  # update 15, before the bound 21
    with open(d2 / "ms_updates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert _check(new2, ref2)["summary"]["C_MS4_pass"] == 0
    # a polishing block that starts at local update 17 moves the bound to 16 and the same tamper at 15 still fails,
    # while a tamper at update 17 no longer matters
    rl = json.load(open(d2 / "rule_log.json"))
    rl["stages"]["2"]["blocks"].append({"block_id": 9, "type": "polish", "first_local": 17, "last_local": 20})
    json.dump(rl, open(d2 / "rule_log.json", "w"))
    r = _check(new2, ref2)
    assert r["C_MS4"][0]["identical_through_update"] == 16 and r["summary"]["C_MS4_pass"] == 0
    rows[14]["rngpos_start"] = [x for x in csv.DictReader(open(ref / f"q{Q}" / f"seed{SEED}" / "MS_s35a5" / "ms_updates.csv"))][14]["rngpos_start"]
    rows[17]["rngpos_start"] = "0:0:0"
    with open(d2 / "ms_updates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    r = _check(new2, ref2)
    assert r["summary"]["C_MS4_pass"] == 1 and r["C_MS4"][0]["identical_through_update"] == 16


def _rewrite_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_c_ms4_ignores_p_digest_next_only_where_the_reference_polishes(wave, tmp_path):
    """MS-R1's check row at the block end before a polishing block carries the POLISHING sampler's digest in
    ``p_digest_next`` while ``NL_st_s1`` (global at every update) carries the global one: found by the review. With a
    polishing block following at update 21 the bound is 20 (a check row) and that one column is not compared; where the
    reference does not polish it is, and ``p_digest`` (the probabilities of every update) is compared in both cases."""
    new, ref = _copy(wave, tmp_path)
    d = ref / f"q{Q}" / f"seed{SEED}" / "MS_s35a5"
    rows = list(csv.DictReader(open(d / "ms_checks_stage2.csv")))
    i = [k for k, r in enumerate(rows) if r["update"] == "20"][0]
    assert rows[i]["p_digest_next"] != ""
    rows[i]["p_digest_next"] = "582a3653636d0dc2"                           # the digest of a (hypothetical) polishing setting
    _rewrite_csv(d / "ms_checks_stage2.csv", rows)
    assert _check(new, ref)["summary"]["C_MS4_pass"] == 0                  # no polishing block: the column is compared
    rl = json.load(open(d / "rule_log.json"))
    rl["stages"]["2"]["blocks"].append({"block_id": 2, "type": "polish", "first_local": 21, "last_local": 40})
    json.dump(rl, open(d / "rule_log.json", "w"))
    r = _check(new, ref)
    assert r["C_MS4"][0]["identical_through_update"] == 20 and r["summary"]["C_MS4_pass"] == 1
    rows[i]["p_digest"] = "0123456789abcdef"                               # a different probability vector in force at 20
    _rewrite_csv(d / "ms_checks_stage2.csv", rows)
    assert _check(new, ref)["summary"]["C_MS4_pass"] == 0


def test_a_missing_run_is_a_failure_and_the_cli_exit_code(wave, tmp_path):
    new, ref = _copy(wave, tmp_path)
    shutil.rmtree(new / f"q{Q}" / f"seed{SEED}" / "NL_bb_s16")
    r = _check(new, ref)
    assert not r["all_ok"] and r["summary"]["C_NL_pass"] == 3
    mp = pytest.MonkeyPatch()
    mp.setattr(LC1, "WEIGHTS_EVERY", 5)
    try:
        out = tmp_path / "lc.json"
        argv = ["--root", str(wave[0]), "--ms-r1-pilot-root", str(wave[1]), "--code-commit", "abc1234", "--qs", str(Q),
                "--seeds", str(SEED), "--ramp-first", "21", "--ramp-last", "40", "--export-every", "5", "--out", str(out)]
        assert RC.main(argv) == 0 and json.load(open(out))["all_ok"] is True
        argv[argv.index("--root") + 1] = str(new)
        assert RC.main(argv) == 2
    finally:
        mp.undo()
