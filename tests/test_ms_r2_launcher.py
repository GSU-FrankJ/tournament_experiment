"""MS-R2 wave of the launcher (tools/ms/launch_ms_r1.py, wave ``r2``) and the NL arms of the config builder."""

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

PARAMS = json.load(open(ROOT / "reports" / "ms" / "r1" / "prereg_parameters.json"))
PROTO = mc.load_protocol()


def _jobs(tmp_path, arms=None, qs=mc.DEFAULT_QS, seeds=mc.DEFAULT_SEEDS):
    return L.build_jobs("r2", qs, seeds, arms, tmp_path, PARAMS)


def test_the_wave_plans_120_runs_into_the_pilot_directory(tmp_path):
    jobs = _jobs(tmp_path)
    assert len(jobs) == 120 and {j.cfg["arm"] for j in jobs} == set(mc.R2_ARMS)
    assert mc.R2_ARMS == ("NL_bb_s1", "NL_bb_s4", "NL_bb_s16", "NL_st_s1", "NL_st_s4", "NL_st_s16")
    assert all(j.out_dir.endswith(f"/pilot/q{j.cfg['q']}/seed{j.cfg['seed']}/{j.cfg['arm']}") for j in jobs)
    cmd = L.job_command(*jobs[0])
    assert cmd[3].endswith("run/run_ms_stagewise.py") and "--config" in cmd
    assert L.WAVE_ARMS["r2"] == mc.R2_ARMS and L.WAVE_DIR["r2"] == "pilot"


def test_the_ms_r1_waves_are_unchanged(tmp_path):
    for wave, n in (("pilot", 120), ("base", 20), ("v20_repro", 20)):
        assert len(L.build_jobs(wave, mc.DEFAULT_QS, mc.DEFAULT_SEEDS, None, tmp_path, PARAMS)) == n
    assert set(mc.ARMS) == {"MS_base", "MS_base2400", "MS_rule", "MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5"}


def test_every_config_validates_and_carries_the_r2_keys_as_designed(tmp_path):
    jobs = _jobs(tmp_path)
    rows = L.validate_jobs(jobs)
    assert len(rows) == 120 and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]][:2]
    for j in jobs:
        c, arm = j.cfg, j.cfg["arm"]
        s = float(arm.split("_s")[-1])
        assert c["pilot"] == "ms_r2" and c["run"].startswith("msr2_q") and c["noise_report"] is True
        assert c["clamp_likelihood"] == "density" and c["rule"]["enabled"] is False
        assert c["pipeline"]["budgets"] == {"2": 2800, "1": 600}
        assert c["pipeline"]["lr_windows"] == {"2": [{"first": 2401, "last": 2800, "start": 3e-4, "end": 3e-5}],
                                               "1": [{"first": 1, "last": 600, "start": 3e-4, "end": 3e-5}]}
        assert c["rule"]["stages"]["2"] == {"eps": 0.005, "rho": 0.05, "tau": 0.02}      # the would-fire thresholds of the file
        if s == 1.0:
            assert "conc_scale_schedule" not in c
        else:
            assert c["conc_scale_schedule"] == {"stage": 2, "local_first": 2001, "local_last": 2200, "scale_first": 1.0,
                                                "scale_last": s}
        if "_st_" in arm:
            assert c["fixed_global_sampler"] is True and c["start_weights"]["scheme"] == "stratified_priority"
        else:
            assert "fixed_global_sampler" not in c and c["start_weights"] == {"scheme": "bin_balanced"}


def test_the_six_configs_differ_only_in_start_weights_and_the_scale_keys(tmp_path):
    jobs = {(j.cfg["arm"], j.cfg["q"], j.cfg["seed"]): j.cfg for j in _jobs(tmp_path, seeds=[10501])}
    for q in mc.DEFAULT_QS:
        ref = jobs[("NL_bb_s1", q, 10501)]
        for arm in mc.R2_ARMS[1:]:
            c = jobs[(arm, q, 10501)]
            diff = {k for k in set(ref) | set(c) if ref.get(k) != c.get(k)}
            allowed = {"arm", "run", "record", "start_weights", "derived", "conc_scale_schedule", "fixed_global_sampler"}
            assert diff <= allowed, (arm, diff - allowed)
            assert {k for k in ref["record"] if ref["record"][k] != c["record"][k]} <= {"run", "output_dir"}
        # the three samplers share their scale key, the two samplers their schedule
        assert jobs[("NL_st_s4", q, 10501)]["conc_scale_schedule"] == jobs[("NL_bb_s4", q, 10501)]["conc_scale_schedule"]
        assert jobs[("NL_st_s16", q, 10501)]["conc_scale_schedule"] == jobs[("NL_bb_s16", q, 10501)]["conc_scale_schedule"]


def test_the_stratified_arms_use_the_ms_s35a5_sampler_settings(tmp_path):
    for q in mc.DEFAULT_QS:
        ms = mc.build_config(PROTO, q, 10501, "MS_s35a5", str(tmp_path), PARAMS)
        for s in (1, 4, 16):
            nl = mc.build_config(PROTO, q, 10501, f"NL_st_s{s}", str(tmp_path), PARAMS)
            assert nl["start_weights"] == ms["start_weights"]
            assert nl["derived"] == ms["derived"]
            assert nl["rule"]["K"] == ms["rule"]["K"] and nl["rule"]["M"] == ms["rule"]["M"]
    ms = mc.build_config(PROTO, 50, 10501, "MS_s35a5", str(tmp_path), PARAMS)["start_weights"]
    assert (ms["lambda_P"], ms["alpha_global"], ms["ema_beta"], ms["near_tie_half_width"]) == (0.35, 0.5, 0.5, 20.0)


def test_the_dry_run_writes_the_record_and_validates_all_configs(tmp_path):
    f = ROOT / "reports" / "ms" / "r1" / "prereg_parameters.json"
    assert L.main(["--wave", "r2", "--params", str(f), "--workers", "40", "--code-commit", "HEAD", "--dry-run",
                   "--root", str(tmp_path)]) == 0
    rec = json.load(open(next((tmp_path / "pilot").glob("dryrun_*.json"))))
    assert rec["wave"] == "r2" and rec["n_planned"] == 120 and rec["validation"]["n_invalid"] == 0
    assert rec["params_sha256"] == mc.sha256_file(f) and rec["diff_stat_run_code_to_head"] == ""
    assert (tmp_path / "pilot" / "q60" / "seed10510" / "NL_st_s16" / "run_config.json").is_file()
    with pytest.raises(SystemExit):
        L.main(["--wave", "r2", "--arms", "MS_rule", "--dry-run", "--root", str(tmp_path / "x")])


def test_the_default_root_of_the_r2_wave_is_results_ms_r2():
    assert L.default_root("r2") == (ROOT / "results" / "ms_r2").resolve()
    for wave in ("pilot", "base", "v20_repro"):
        assert L.default_root(wave) == (ROOT / "results" / "ms_r1").resolve()
