"""MS-R3 wave of the launcher (tools/ms/launch_ms_r1.py, wave ``r3``) and the twelve arms of the config builder
(tools/ms/ms_configs.py).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r3_launcher.py -p no:cacheprovider -q

Sections: (1) the arm tables and the wave plan; (2) every config validates and carries the MS-R3 keys as designed;
(3) a ``t1`` arm is the MS-R2 ``NL_*`` arm (labels and ``init_digest`` apart), the three actors differ in the variant key
only; (4) nothing of MS-R1 / MS-R2 changed (tables and ``build_config`` outputs against the base commit of the round);
(5) the ``--actors`` filter, the dry-run record and the refusals.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import launch_ms_r1 as L  # noqa: E402
import ms_configs as mc  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402

PARAMS_FILE = ROOT / "reports" / "ms" / "r1" / "prereg_parameters.json"
PARAMS = json.load(open(PARAMS_FILE))
PROTO = mc.load_protocol()
BASE_COMMIT = "e8eb9a08"                        # the MS-R2 close, the base of the round (ms-r3 is created from it)
EXPECTED_ARMS = ("t1_bb_s1", "t1_bb_s16", "t1_st_s1", "t1_st_s16", "relu_bb_s1", "relu_bb_s16", "relu_st_s1", "relu_st_s16",
                 "t10_bb_s1", "t10_bb_s16", "t10_st_s1", "t10_st_s16")
LABELS = {"arm", "run", "pilot"}                # label-like top-level keys (the round tag, the run name, the arm)


def _jobs(tmp_path, arms=None, qs=mc.DEFAULT_QS, seeds=mc.DEFAULT_SEEDS, actors=None):
    return L.build_jobs("r3", qs, seeds, arms, tmp_path, PARAMS, actors)


def _cfg(arm: str, q: int = 50, seed: int = 10501, out: str = "/unused", params=PARAMS, mod=mc) -> Dict[str, Any]:
    return mod.build_config(PROTO, q, seed, arm, out, params)


def _strip(cfg: Dict[str, Any], *drop: str) -> Dict[str, Any]:
    """A config without the label-like keys (and ``drop``), and without the run name inside the embedded record."""
    c = {k: v for k, v in cfg.items() if k not in LABELS and k not in drop}
    c["record"] = {k: v for k, v in cfg["record"].items() if k != "run"}
    return c


@pytest.fixture(scope="module")
def base_mod() -> types.ModuleType:
    """``tools/ms/ms_configs.py`` as committed at the base of the round, loaded without touching the working tree."""
    try:
        src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:tools/ms/ms_configs.py"], text=True,
                                      stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pytest.skip(f"commit {BASE_COMMIT} is not available in this clone")
    mod = types.ModuleType("ms_configs_at_base_commit")
    mod.__file__ = str(ROOT / "tools" / "ms" / "ms_configs.py")          # the module resolves ROOT from its own path
    exec(compile(src, f"ms_configs.py@{BASE_COMMIT}", "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------------------------------------- 1. tables and plan
def test_the_twelve_arms_are_kept_apart_from_the_earlier_tables():
    assert mc.R3_ARMS == EXPECTED_ARMS and mc.R3_ACTORS == ("t1", "relu", "t10") and mc.R3_SCALES == (1, 16)
    assert set(mc.R3_ARM_TABLE) == set(mc.R3_ARMS) and len(mc.R3_ARMS) == 12
    assert not set(mc.R3_ARMS) & (set(mc.ARMS) | set(mc.R2_ARM_TABLE))
    assert mc.R2_ARMS == ("NL_bb_s1", "NL_bb_s4", "NL_bb_s16", "NL_st_s1", "NL_st_s4", "NL_st_s16")
    assert mc.R3_MS_R2_REFERENCE == {"t1_bb_s1": "NL_bb_s1", "t1_bb_s16": "NL_bb_s16", "t1_st_s1": "NL_st_s1",
                                     "t1_st_s16": "NL_st_s16"}
    for arm in mc.R3_ARMS:
        actor, kind, s = mc.r3_arm_parts(arm)
        assert arm == f"{actor}_{kind}_s{s}" and mc.r2_equivalent(arm) == f"NL_{kind}_s{s}"
        assert mc.R3_ARM_TABLE[arm]["round"] == "ms_r3" and mc.R3_ARM_TABLE[arm]["rule_enabled"] is False
    with pytest.raises(KeyError):
        mc.r3_arm_parts("NL_bb_s1")
    with pytest.raises(KeyError):
        mc.build_config(PROTO, 50, 10501, "t1_bb_s4", "/unused", PARAMS)


def test_the_wave_plans_240_runs_into_the_pilot_directory(tmp_path):
    jobs = _jobs(tmp_path)
    assert len(jobs) == 240 and {j.cfg["arm"] for j in jobs} == set(mc.R3_ARMS)
    assert all(sum(1 for j in jobs if j.cfg["arm"] == a) == 20 for a in mc.R3_ARMS)
    assert {(j.cfg["q"], j.cfg["seed"]) for j in jobs} == {(q, s) for q in (50, 60) for s in range(10501, 10511)}
    assert all(j.out_dir == str(tmp_path / "pilot" / f"q{j.cfg['q']}" / f"seed{j.cfg['seed']}" / j.cfg["arm"]) for j in jobs)
    assert [(j.cfg["q"], j.cfg["seed"], j.cfg["arm"]) for j in jobs[:13]] == [
        (50, 10501, a) for a in EXPECTED_ARMS] + [(50, 10502, "t1_bb_s1")]
    cmd = L.job_command(*jobs[0])
    assert cmd[3].endswith("run/run_ms_stagewise.py") and cmd[4] == "--config" and cmd[5].endswith("t1_bb_s1/run_config.json")
    assert L.WAVE_ARMS["r3"] == mc.R3_ARMS and L.WAVE_DIR["r3"] == "pilot" and L.WAVES[-1] == "r3"


def test_the_earlier_waves_are_unchanged(tmp_path):
    for wave, n in (("pilot", 120), ("base", 20), ("v20_repro", 20), ("r2", 120)):
        assert len(L.build_jobs(wave, mc.DEFAULT_QS, mc.DEFAULT_SEEDS, None, tmp_path, PARAMS)) == n, wave
    assert L.WAVES[:4] == ("base", "v20_repro", "pilot", "r2")
    assert L.WAVE_ARMS["r2"] == mc.R2_ARMS and L.WAVE_ARMS["pilot"] == mc.PILOT_ARMS + ("MS_base2400",)
    assert L.WAVE_DIR == {"base": "base", "v20_repro": "v20_reproduction", "pilot": "pilot", "r2": "pilot", "r3": "pilot"}
    for wave in ("pilot", "base", "v20_repro"):
        assert L.default_root(wave) == (ROOT / "results" / "ms_r1").resolve()
    assert L.default_root("r2") == (ROOT / "results" / "ms_r2").resolve()


def test_the_default_root_of_the_r3_wave_is_results_ms_r3():
    assert L.default_root("r3") == (ROOT / "results" / "ms_r3").resolve()
    assert L.MAX_WORKERS == 40


# ---------------------------------------------------------------------------------------------- 2. configs
def test_every_config_validates_and_carries_the_r3_keys_as_designed(tmp_path):
    jobs = _jobs(tmp_path)
    rows = L.validate_jobs(jobs)
    assert len(rows) == 240 and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]][:2]
    for j in jobs:
        c, arm = j.cfg, j.cfg["arm"]
        actor, kind, s = mc.r3_arm_parts(arm)
        assert c["pilot"] == "ms_r3" and c["run"] == f"msr3_q{c['q']}_s{c['seed']}_{arm}" and c["noise_report"] is True
        assert c["init_digest"] is True                                          # all twelve arms
        if actor == "t1":
            assert "actor_variant" not in c                                      # the control writes no variant key
        else:
            assert c["actor_variant"] == actor
        assert c["clamp_likelihood"] == "density" and c["rule"]["enabled"] is False
        assert c["pipeline"]["budgets"] == {"2": 2800, "1": 600}
        assert c["pipeline"]["lr_windows"] == {"2": [{"first": 2401, "last": 2800, "start": 3e-4, "end": 3e-5}],
                                               "1": [{"first": 1, "last": 600, "start": 3e-4, "end": 3e-5}]}
        assert c["rule"]["stages"]["2"] == {"eps": 0.005, "rho": 0.05, "tau": 0.02}       # the would-fire thresholds
        if s == 1:
            assert "conc_scale_schedule" not in c                                # an s = 1 arm never sets a scale
        else:
            assert c["conc_scale_schedule"] == {"stage": 2, "local_first": 2001, "local_last": 2200, "scale_first": 1.0,
                                                "scale_last": 16.0}
        if kind == "st":
            sw = c["start_weights"]
            assert c["fixed_global_sampler"] is True and sw["scheme"] == "stratified_priority"
            assert (sw["lambda_P"], sw["alpha_global"], sw["ema_beta"], sw["near_tie_half_width"]) == (0.35, 0.5, 0.5, 20.0)
        else:
            assert "fixed_global_sampler" not in c and c["start_weights"] == {"scheme": "bin_balanced"}
        assert set(c) <= set(rms.REQUIRED) | set(rms.OPTIONAL)


def test_the_stratified_arms_use_the_nl_st_sampler_settings():
    for q in mc.DEFAULT_QS:
        nl = _cfg("NL_st_s1", q)
        for arm in mc.R3_ARMS:
            if "_st_" in arm:
                c = _cfg(arm, q)
                assert c["start_weights"] == nl["start_weights"] and c["derived"] == nl["derived"]
                assert c["rule"] == nl["rule"] and c["fixed_global_sampler"] is True


# ---------------------------------------------------------------------------------------------- 3. t1 = NL, actors
def test_a_t1_arm_is_the_ms_r2_arm_except_labels_and_the_init_digest(base_mod):
    for q in mc.DEFAULT_QS:
        for seed in (10501, 10510):
            for arm, nl_arm in mc.R3_MS_R2_REFERENCE.items():
                c = _cfg(arm, q, seed, "/out")
                nl = _cfg(nl_arm, q, seed, "/out", mod=base_mod)                 # the NL arm as the base commit defines it
                assert {k for k in set(c) | set(nl) if c.get(k) != nl.get(k)} == LABELS | {"init_digest", "record"}
                assert {k for k in c["record"] if c["record"][k] != nl["record"][k]} == {"run"}
                assert c["init_digest"] is True and "init_digest" not in nl
                assert _strip(c, "init_digest") == _strip(nl)                    # everything else is identical
                assert (c["arm"], nl["arm"]) == (arm, nl_arm) and (c["pilot"], nl["pilot"]) == ("ms_r3", "ms_r2")


def test_the_three_actors_differ_only_in_the_variant_key():
    for q in mc.DEFAULT_QS:
        for kind in ("bb", "st"):
            for s in mc.R3_SCALES:
                t1, relu, t10 = (_cfg(f"{a}_{kind}_s{s}", q) for a in mc.R3_ACTORS)
                for x, y in ((t1, relu), (t1, t10), (relu, t10)):
                    diff = {k for k in set(x) | set(y) if x.get(k) != y.get(k)}
                    assert diff == {"arm", "run", "record", "actor_variant"}, diff
                    assert {k for k in x["record"] if x["record"][k] != y["record"][k]} == {"run"}
                assert _strip(relu, "actor_variant") == _strip(t1) == _strip(t10, "actor_variant")
                assert (relu["actor_variant"], t10["actor_variant"]) == ("relu", "t10") and "actor_variant" not in t1


# ---------------------------------------------------------------------------------------------- 4. nothing else changed
def test_the_earlier_tables_and_every_earlier_build_are_unchanged(base_mod, tmp_path):
    for name in ("ARMS", "R2_ARM_TABLE", "R2_ARMS", "R2_SCALES", "PILOT_ARMS", "CONTROL_2400_ARM", "DEFAULT_PARAMS", "LEGACY_PIPELINE",
                 "LEGACY_PIPELINE_2400", "LEGACY_PIPELINE_NL", "CONC_RAMP_FIRST", "CONC_RAMP_LAST", "PILOT", "PILOT_R2",
                 "DEFAULT_QS", "DEFAULT_SEEDS", "PROTOCOL_REL", "REPORT_NEAR_TIE_HALF_WIDTH"):
        assert getattr(mc, name) == getattr(base_mod, name), name
    old_arms = list(base_mod.ARMS) + list(base_mod.R2_ARM_TABLE)
    assert len(old_arms) == 13
    for arm in old_arms:
        for q in mc.DEFAULT_QS:
            for params in (PARAMS, None):
                new = json.dumps(mc.build_config(PROTO, q, 10501, arm, str(tmp_path), params), sort_keys=True)
                old = json.dumps(base_mod.build_config(PROTO, q, 10501, arm, str(tmp_path), params), sort_keys=True)
                assert new == old, (arm, q)
    one = dict(eps=0.005, rho=0.03, tau=0.02, n_block=20, u_cap=40, n_land=10)          # a T = 3 build (the tests of the runner)
    prm = copy.deepcopy(mc.DEFAULT_PARAMS)
    prm["K"], prm["stages"] = 10, {str(t): dict(one) for t in (1, 2, 3)}
    assert json.dumps(mc.build_config(PROTO, 50, 10501, "MS_rule", "/o", prm, T=3), sort_keys=True) == json.dumps(
        base_mod.build_config(PROTO, 50, 10501, "MS_rule", "/o", prm, T=3), sort_keys=True)
    for bad in (lambda m: m.build_config(PROTO, 50, 10501, "nope", "/o", PARAMS), lambda m: m.build_config(PROTO, 55, 10501, "MS_rule", "/o")):
        for m in (mc, base_mod):
            with pytest.raises(KeyError):
                bad(m)


def test_the_run_names_of_the_earlier_rounds_keep_their_prefix():
    assert _cfg("MS_rule")["run"] == "msr1_q50_s10501_MS_rule" and _cfg("MS_rule")["pilot"] == "ms_r1"
    assert _cfg("NL_st_s4")["run"] == "msr2_q50_s10501_NL_st_s4" and _cfg("NL_st_s4")["pilot"] == "ms_r2"
    assert _cfg("relu_st_s16")["run"] == "msr3_q50_s10501_relu_st_s16" and _cfg("relu_st_s16")["pilot"] == "ms_r3"


# ---------------------------------------------------------------------------------------------- 5. launcher
def test_the_dry_run_writes_the_record_and_validates_all_configs(tmp_path):
    assert L.main(["--wave", "r3", "--params", str(PARAMS_FILE), "--workers", "40", "--code-commit", "HEAD", "--dry-run",
                   "--root", str(tmp_path)]) == 0
    rec = json.load(open(next((tmp_path / "pilot").glob("dryrun_*.json"))))
    assert rec["wave"] == "r3" and rec["dry_run"] is True and rec["n_planned"] == 240 and rec["validation"]["n_invalid"] == 0
    assert sorted(rec["validation"]["per_arm"]) == sorted(mc.R3_ARMS)
    assert all(v == {"n": 20, "valid": 20, "errors": []} for v in rec["validation"]["per_arm"].values())
    assert rec["actors"] == ["t1", "relu", "t10"] and rec["workers"] == 40
    assert rec["params_sha256"] == mc.sha256_file(PARAMS_FILE) and rec["diff_stat_run_code_to_head"] == ""
    for k in ("nproc", "loadavg_at_start", "disk_free_bytes", "head", "status_porcelain", "code_commit",
              "diff_stat_code_commit_to_head", "planned", "params"):
        assert k in rec
    cfg = json.load(open(tmp_path / "pilot" / "q60" / "seed10510" / "t10_st_s16" / "run_config.json"))
    rms.validate_config(cfg)
    assert cfg["actor_variant"] == "t10" and cfg["init_digest"] is True and cfg["pilot"] == "ms_r3"
    t1 = json.load(open(tmp_path / "pilot" / "q50" / "seed10501" / "t1_bb_s1" / "run_config.json"))
    assert "actor_variant" not in t1 and t1["init_digest"] is True
    with pytest.raises(SystemExit):                                          # an arm of another wave
        L.main(["--wave", "r3", "--arms", "NL_bb_s1", "--dry-run", "--root", str(tmp_path / "x")])
    with pytest.raises(SystemExit):                                          # the run directories exist now
        (tmp_path / "pilot" / "q50" / "seed10501" / "t1_bb_s1" / "run.log").write_text("")
        L.main(["--wave", "r3", "--dry-run", "--root", str(tmp_path)])


def test_the_actors_filter_plans_only_the_arms_of_the_listed_actors(tmp_path):
    assert len(_jobs(tmp_path, actors=("t1", "relu", "t10"))) == 240 and len(_jobs(tmp_path, actors=None)) == 240
    for actors, n_arms in ((("t1", "relu"), 8), (("relu", "t10"), 8), (("t1", "t10"), 8), (("t1",), 4), (("relu",), 4), (("t10",), 4)):
        jobs = _jobs(tmp_path, actors=actors)
        assert len(jobs) == n_arms * 20 and {j.cfg["arm"] for j in jobs} == set(mc.r3_arms_of(actors))
        assert {j.cfg["arm"].split("_")[0] for j in jobs} == set(actors)
    assert [j.cfg["arm"] for j in _jobs(tmp_path, actors=("t10", "t1"), qs=[50], seeds=[10501])] == [
        "t1_bb_s1", "t1_bb_s16", "t1_st_s1", "t1_st_s16", "t10_bb_s1", "t10_bb_s16", "t10_st_s1", "t10_st_s16"]
    only = _jobs(tmp_path, arms=["relu_bb_s1"], actors=("relu",))
    assert len(only) == 20 and {j.cfg["arm"] for j in only} == {"relu_bb_s1"}
    with pytest.raises(ValueError):                                          # an arm outside the listed actors
        _jobs(tmp_path, arms=["relu_bb_s1"], actors=("t1",))
    with pytest.raises(ValueError):
        _jobs(tmp_path, actors=("t1", "bogus"))
    with pytest.raises(ValueError):                                          # defined for wave r3 only
        L.build_jobs("r2", [50], [10501], None, tmp_path, PARAMS, ("t1",))
    # the configs of an arm do not depend on the filter
    full = {j.cfg["arm"]: j.cfg for j in _jobs(tmp_path, qs=[50], seeds=[10501])}
    assert all(j.cfg == full[j.cfg["arm"]] for j in _jobs(tmp_path, qs=[50], seeds=[10501], actors=("relu", "t10")))


def test_the_actors_filter_on_the_command_line(tmp_path):
    f = ["--params", str(PARAMS_FILE), "--dry-run", "--code-commit", "HEAD", "--wave", "r3"]
    assert L.main([*f, "--actors", "relu,t10", "--root", str(tmp_path / "a")]) == 0
    rec = json.load(open(next((tmp_path / "a" / "pilot").glob("dryrun_*.json"))))
    assert rec["n_planned"] == 160 and rec["actors"] == ["relu", "t10"] and rec["validation"]["n_invalid"] == 0
    assert not (tmp_path / "a" / "pilot" / "q50" / "seed10501" / "t1_bb_s1").exists()
    assert (tmp_path / "a" / "pilot" / "q50" / "seed10501" / "relu_bb_s1" / "run_config.json").is_file()
    assert L.main([*f, "--actors", "t10,t1", "--root", str(tmp_path / "b")]) == 0               # the order does not matter
    rec = json.load(open(next((tmp_path / "b" / "pilot").glob("dryrun_*.json"))))
    assert rec["n_planned"] == 160 and rec["actors"] == ["t1", "t10"]
    assert L.main([*f, "--actors", "t1,relu,t10", "--root", str(tmp_path / "c")]) == 0
    assert json.load(open(next((tmp_path / "c" / "pilot").glob("dryrun_*.json"))))["n_planned"] == 240
    for i, actors in enumerate(("t1,bogus", "", "t1,,relu", "T1", "t1;relu")):                # unknown or empty names
        with pytest.raises(SystemExit):
            L.main([*f, "--actors", actors, "--root", str(tmp_path / f"bad{i}")])
    for wave in ("pilot", "r2", "base", "v20_repro"):                                           # r3 only
        with pytest.raises(SystemExit):
            L.main(["--wave", wave, "--actors", "t1", "--dry-run", "--root", str(tmp_path / f"w{wave}")])
    with pytest.raises(SystemExit):                                                             # an arm outside the filter
        L.main([*f, "--actors", "t1", "--arms", "relu_bb_s1", "--root", str(tmp_path / "d")])
    assert not (tmp_path / "bad0").exists() and not (tmp_path / "wpilot").exists()              # nothing was written


def test_the_other_refusals_apply_to_the_r3_wave(tmp_path):
    with pytest.raises(SystemExit):
        L.main(["--wave", "r3", "--dry-run", "--seeds", "40501", "--root", str(tmp_path / "s")])    # fresh seeds stay reserved
    with pytest.raises(SystemExit):
        L.main(["--wave", "r3", "--dry-run", "--workers", "41", "--root", str(tmp_path / "w")])
    with pytest.raises(SystemExit):
        L.main(["--wave", "r3", "--dry-run", "--params", str(tmp_path / "missing.json"), "--root", str(tmp_path / "p")])
