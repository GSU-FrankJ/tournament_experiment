"""MS-R1 launcher (tools/ms/launch_ms_r1.py) and config builder (tools/ms/ms_configs.py)."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import launch_ms_r1 as L  # noqa: E402
import ms_configs as mc  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402


def _dry(tmp_path, *argv):
    return L.main(["--dry-run", "--root", str(tmp_path), *argv])


def test_wave_plans_and_job_counts(tmp_path):
    for wave, n in (("pilot", 100), ("base", 20), ("v20_repro", 20)):
        jobs = L.build_jobs(wave, mc.DEFAULT_QS, mc.DEFAULT_SEEDS, None, tmp_path, mc.DEFAULT_PARAMS)
        assert len(jobs) == n, wave
    pilot = L.build_jobs("pilot", mc.DEFAULT_QS, mc.DEFAULT_SEEDS, None, tmp_path, mc.DEFAULT_PARAMS)
    assert {j.cfg["arm"] for j in pilot} == set(mc.PILOT_ARMS) and "MS_base" not in {j.cfg["arm"] for j in pilot}
    assert all(j.out_dir.endswith(f"q{j.cfg['q']}/seed{j.cfg['seed']}/{j.cfg['arm']}") for j in pilot)
    base = L.build_jobs("base", mc.DEFAULT_QS, mc.DEFAULT_SEEDS, None, tmp_path, mc.DEFAULT_PARAMS)
    assert {j.cfg["arm"] for j in base} == {"MS_base"} and "/base/" in base[0].out_dir
    rep = L.build_jobs("v20_repro", mc.DEFAULT_QS, mc.DEFAULT_SEEDS, None, tmp_path, mc.DEFAULT_PARAMS)
    assert all(L.is_locked_entry(j.cfg) for j in rep) and "v20_reproduction" in rep[0].out_dir


def test_commands_use_the_right_entry_points(tmp_path):
    pilot = L.build_jobs("pilot", [50], [10501], ["MS_rule"], tmp_path, mc.DEFAULT_PARAMS)[0]
    cmd = L.job_command(*pilot)
    assert cmd[3].endswith("run/run_ms_stagewise.py") and "--config" in cmd and "run_config.json" in cmd[5]
    rep = L.build_jobs("v20_repro", [50], [10501], None, tmp_path, mc.DEFAULT_PARAMS)[0]
    cmd = L.job_command(*rep)
    assert cmd[3].endswith("run/run_v2_T2_locked.py") and cmd[-4:] == ["--seed", "10501", "--out-dir", rep.out_dir]
    assert "--config" not in cmd


def test_every_config_validates_and_carries_every_key(tmp_path):
    jobs = L.build_jobs("pilot", mc.DEFAULT_QS, mc.DEFAULT_SEEDS, None, tmp_path, mc.DEFAULT_PARAMS)
    rows = L.validate_jobs(jobs)
    assert len(rows) == 100 and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]][:2]
    for j in jobs:
        assert set(j.cfg) >= set(rms.REQUIRED) and j.cfg["clamp_likelihood"] == "density"
        assert j.cfg["rule"]["enabled"] is True and j.cfg["pipeline"]["budgets"] is None
        assert j.cfg["start_weights"]["scheme"] == "stratified_priority" and "derived" in j.cfg


def test_the_five_pilot_arms_differ_from_ms_rule_only_in_start_weights(tmp_path):
    jobs = {(j.cfg["arm"], j.cfg["q"], j.cfg["seed"]): j.cfg
            for j in L.build_jobs("pilot", mc.DEFAULT_QS, [10501], None, tmp_path, mc.DEFAULT_PARAMS)}
    for q in mc.DEFAULT_QS:
        base = jobs[("MS_rule", q, 10501)]
        for arm in mc.PILOT_ARMS[1:]:
            c = jobs[(arm, q, 10501)]
            same = {k for k in base if k not in ("arm", "run", "start_weights", "derived", "record")}
            assert all(base[k] == c[k] for k in same), arm
            sw0, sw1 = base["start_weights"], c["start_weights"]
            assert {k for k in sw0 if sw0[k] != sw1[k]} <= {"lambda_P", "alpha_global"}


def test_base_arm_is_the_legacy_configuration(tmp_path):
    c = L.build_jobs("base", [50], [10501], None, tmp_path, mc.DEFAULT_PARAMS)[0].cfg
    assert c["rule"]["enabled"] is False and c["start_weights"] == {"scheme": "bin_balanced"}
    assert c["pipeline"]["budgets"] == {"2": 1600, "1": 600}
    assert c["pipeline"]["lr_windows"] == {"2": [{"first": 1201, "last": 1600, "start": 3e-4, "end": 3e-5}],
                                           "1": [{"first": 1, "last": 600, "start": 3e-4, "end": 3e-5}]}
    proto = mc.load_protocol()
    lr = {w["phase"]: w for w in proto["pipeline"]["lr_decay"]}
    assert (lr["A"]["local_first"], lr["A"]["local_last"], lr["B"]["local_first"], lr["B"]["local_last"]) == (1201, 1600, 1, 600)
    assert c["record"] == {**proto["records"]["50"], "seed": 10501, "run": c["run"], "output_dir": c["record"]["output_dir"]}


def test_the_parameter_file_is_used_and_recorded(tmp_path):
    prm = copy.deepcopy(mc.DEFAULT_PARAMS)
    prm["K"], prm["stages"]["2"]["rho"], prm["alpha_polish"] = 50, 0.05, 0.7
    f = tmp_path / "params.json"
    f.write_text(json.dumps(prm))
    root = tmp_path / "r"
    assert L.main(["--wave", "pilot", "--arms", "MS_s25a5", "--qs", "50", "--seeds", "10501", "--params", str(f),
                   "--dry-run", "--root", str(root), "--code-commit", "HEAD"]) == 0
    cfg = json.load(open(root / "pilot" / "q50" / "seed10501" / "MS_s25a5" / "run_config.json"))
    assert cfg["rule"]["K"] == 50 and cfg["rule"]["stages"]["2"]["rho"] == 0.05 and cfg["start_weights"]["alpha_polish"] == 0.7
    rec = json.load(open(next((root / "pilot").glob("dryrun_*.json"))))
    assert rec["params_sha256"] == mc.sha256_file(f) and rec["params"] == prm and rec["dry_run"] is True
    for k in ("nproc", "loadavg_at_start", "disk_free_bytes", "head", "status_porcelain", "code_commit",
              "diff_stat_code_commit_to_head", "validation"):
        assert k in rec
    assert rec["validation"]["n_invalid"] == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"K": 25}))
    with pytest.raises(SystemExit):
        L.main(["--wave", "pilot", "--params", str(bad), "--dry-run", "--root", str(tmp_path / "x")])


def test_refusals(tmp_path):
    with pytest.raises(SystemExit):
        _dry(tmp_path, "--wave", "pilot", "--seeds", "40501")              # fresh seeds stay reserved
    with pytest.raises(SystemExit):
        _dry(tmp_path, "--wave", "pilot", "--workers", "41")
    with pytest.raises(SystemExit):
        _dry(tmp_path, "--wave", "pilot", "--arms", "MS_base")             # not an arm of the pilot wave
    with pytest.raises(SystemExit):
        _dry(tmp_path, "--wave", "v20_repro", "--arms", "MS_rule")
    assert _dry(tmp_path / "ok", "--wave", "base", "--qs", "50", "--seeds", "10501") == 0
    with pytest.raises(SystemExit):                                         # the run directory exists now
        (tmp_path / "ok" / "base" / "q50" / "seed10501" / "MS_base" / "status.json").write_text("{}")
        _dry(tmp_path / "ok", "--wave", "base", "--qs", "50", "--seeds", "10501")


def test_the_defaults_of_the_arm_table_match_d6():
    assert set(mc.ARMS) == {"MS_base", "MS_rule", "MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5"}
    assert [(mc.ARMS[a]["lambda_P"], mc.ARMS[a]["alpha_global"]) for a in mc.PILOT_ARMS] == [
        (None, 0.0), (0.25, 0.0), (0.25, 0.5), (0.35, 0.0), (0.35, 0.5)]
    d = mc.DEFAULT_PARAMS
    assert (d["K"], d["M"], d["conc_limit"], d["localized_fraction"], d["alpha_polish"], d["ema_beta"]) == (
        25, 3, 0.04, 0.25, 0.5, 0.5)
    assert d["stages"]["2"] == {"eps": 0.005, "rho": 0.03, "tau": 0.02, "n_block": 400, "u_cap": 2000, "n_land": 400}
    assert d["stages"]["1"] == {"eps": 0.005, "rho": 0.03, "tau": 0.02, "n_block": 200, "u_cap": 600, "n_land": 400}
