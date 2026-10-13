"""MS-R4 configs and launcher (tools/ms/r4_configs.py, tools/ms/r4_launch.py): the arm tables, the seed guards, the wave sizes,
the differences between the configs of an experiment, the premise of C-MS6 (the controls' configs equal MS-R3's apart from the
round tag, the run name and the output directory), and the unchanged default behaviour of the existing config builder and
launcher.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r4_launcher.py -p no:cacheprovider -q
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ms_r4_helpers as H  # noqa: E402,I001
from ms_r4_helpers import PARAMS, PROTO, ROOT  # noqa: E402

import launch_ms_r1 as L1  # noqa: E402
import ms_configs as mc  # noqa: E402
import r4_configs as rc  # noqa: E402
import r4_launch as RL  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402

PARAMS_FILE = ROOT / "reports" / "ms" / "r1" / "prereg_parameters.json"


def _cfg(arm: str, q: int = 50, seed: int = 10511, out: str = "/x") -> Dict[str, Any]:
    return rc.build_config_r4(PROTO, q, seed, arm, f"{out}/{arm}", PARAMS)


# ---------------------------------------------------------------------------------------------- 1. the arm tables
def test_the_seven_arms_and_their_tables():
    assert rc.ARMS == ("t1_st_s16", "relu_st_s16", "lrsm_st_s16", "fb_a0", "fb_r", "fb_rg", "fb_rx")
    assert set(rc.R4_E1_ARM_TABLE) == set(rc.E1_ARMS) and set(rc.R4_E2_ARM_TABLE) == set(rc.E2_ARMS)
    assert rc.SEEDS == tuple(range(10511, 10531)) and rc.SEEDS_CMS6 == tuple(range(10501, 10511)) and rc.QS == (50, 60)
    assert all(rc.experiment_of(a) == "E1" for a in rc.E1_ARMS) and all(rc.experiment_of(a) == "E2" for a in rc.E2_ARMS)
    for arm in rc.ARMS:
        cfg = _cfg(arm)
        rms.validate_config(cfg)
        assert cfg["pilot"] == "ms_r4" and cfg["run"] == f"msr4_q50_s10511_{arm}" and cfg["arm"] == arm
        assert cfg["init_digest"] is True and cfg["noise_report"] is True and cfg["fixed_global_sampler"] is True
        assert cfg["clamp_likelihood"] == "density" and cfg["rule"]["enabled"] is False and cfg["rule"]["K"] == 25
        sw = cfg["start_weights"]
        assert (sw["scheme"], sw["lambda_P"], sw["alpha_polish"], sw["ema_beta"], sw["near_tie_half_width"]) == \
               ("stratified_priority", 0.35, 0.5, 0.5, 20.0)
        assert cfg["derived"]["2"]["lambda_T"] == (0.5 if cfg["q"] == 50 else 20 / 44)


def test_the_e1_settings_are_those_of_ms_r3_st_s16():
    for arm in rc.E1_ARMS:
        cfg = _cfg(arm)
        assert cfg["pipeline"]["budgets"] == {"2": 2800, "1": 600}
        assert cfg["pipeline"]["lr_windows"] == {"2": [{"first": 2401, "last": 2800, "start": 3e-4, "end": 3e-5}],
                                                 "1": [{"first": 1, "last": 600, "start": 3e-4, "end": 3e-5}]}
        assert cfg["conc_scale_schedule"] == {"stage": 2, "local_first": 2001, "local_last": 2200, "scale_first": 1.0,
                                              "scale_last": 16.0}
        assert cfg["start_weights"]["alpha_global"] == 0.5 and "focus_scope" not in cfg and "focus_map" not in cfg
    assert "actor_variant" not in _cfg("t1_st_s16")
    assert _cfg("relu_st_s16")["actor_variant"] == "relu" and _cfg("lrsm_st_s16")["actor_variant"] == "lrsm"


def test_the_e2_settings():
    for arm in rc.E2_ARMS:
        cfg = _cfg(arm)
        assert cfg["pipeline"]["budgets"] == {"2": 1600, "1": 600}
        assert cfg["pipeline"]["lr_windows"] == mc.LEGACY_PIPELINE["lr_windows"]
        assert "conc_scale_schedule" not in cfg and "actor_variant" not in cfg
    for q in (50, 60):
        assert _cfg("fb_a0", q)["start_weights"]["alpha_global"] == 0.0
        assert _cfg("fb_r", q)["focus_scope"] == "within_strata" and "focus_map" not in _cfg("fb_r", q)
        assert (_cfg("fb_rg", q)["focus_scope"], _cfg("fb_rg", q)["focus_map"]) == ("within_strata", "residual_gain")
        assert "focus_scope" not in _cfg("fb_rx", q) and _cfg("fb_rx", q)["start_weights"]["alpha_global"] == 0.5


def test_configs_within_an_experiment_differ_only_in_the_intended_keys():
    lab = {"arm", "run", "record"}
    for q in (50, 60):
        c = {a: _cfg(a, q) for a in rc.ARMS}
        assert rc.top_level_diff(c["t1_st_s16"], c["relu_st_s16"]) == sorted(lab | {"actor_variant"})
        assert rc.top_level_diff(c["relu_st_s16"], c["lrsm_st_s16"]) == sorted(lab | {"actor_variant"})
        assert rc.top_level_diff(c["t1_st_s16"], c["lrsm_st_s16"]) == sorted(lab | {"actor_variant"})
        assert rc.config_diff(c["relu_st_s16"], c["lrsm_st_s16"]) == ["actor_variant", "arm", "record.output_dir", "record.run", "run"]
        assert rc.config_diff(c["fb_a0"], c["fb_rx"]) == ["arm", "record.output_dir", "record.run", "run", "start_weights.alpha_global"]
        assert rc.config_diff(c["fb_rx"], c["fb_r"]) == ["arm", "focus_scope", "record.output_dir", "record.run", "run"]
        assert rc.config_diff(c["fb_r"], c["fb_rg"]) == ["arm", "focus_map", "record.output_dir", "record.run", "run"]
        # fb_rx and t1_st_s16 differ in the budget / landing keys only (check C-X2)
        d = rc.config_diff(c["fb_rx"], c["t1_st_s16"])
        assert set(d) <= {"arm", "run", "record.output_dir", "record.run", "pipeline.budgets.2", "conc_scale_schedule.stage",
                          "conc_scale_schedule.local_first", "conc_scale_schedule.local_last", "conc_scale_schedule.scale_first",
                          "conc_scale_schedule.scale_last", "pipeline.lr_windows.2[0].first", "pipeline.lr_windows.2[0].last"} | \
               {k for k in d if k.startswith(("conc_scale_schedule", "pipeline.lr_windows.2"))}
        assert "start_weights.alpha_global" not in d and "actor_variant" not in d


def test_the_controls_equal_ms_r3s_configs_apart_from_the_round_tag_run_name_and_output_directory():
    """The premise of C-MS6: same seeds as MS-R3, the R4 control config and the MS-R3 config of the same name."""
    for arm in rc.CMS6_ARMS:
        for q in (50, 60):
            a = mc.build_config(PROTO, q, 10503, arm, "/out/x", PARAMS)                              # the MS-R3 lookup by name
            b = rc.build_config_r4(PROTO, q, 10503, arm, "/out/x", PARAMS)
            assert a["pilot"] == "ms_r3" and b["pilot"] == "ms_r4"
            assert a["run"] == f"msr3_q{q}_s10503_{arm}" and b["run"] == f"msr4_q{q}_s10503_{arm}"
            a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
            for c in (a2, b2):
                c["pilot"] = c["run"] = c["record"]["run"] = None
            assert a2 == b2, rc.config_diff(a2, b2)


def test_build_config_without_an_arm_table_is_unchanged_from_the_base_commit(tmp_path):
    try:
        src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{H.BASE_COMMIT}:tools/ms/ms_configs.py"],
                                      stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        pytest.skip("the base commit is not available in this clone")
    mod = types.ModuleType("ms_configs_base")
    mod.__file__ = str(ROOT / "tools" / "ms" / "ms_configs.py")      # the module locates the repository root from __file__
    sys.modules["ms_configs_base"] = mod
    exec(compile(src, "ms_configs_base.py", "exec"), mod.__dict__)
    arms = list(mc.ARMS) + list(mc.R2_ARM_TABLE) + list(mc.R3_ARM_TABLE)
    for arm in arms:
        for q in (50, 60):
            new = mc.build_config(PROTO, q, 10501, arm, "/o", PARAMS)
            old = mod.build_config(PROTO, q, 10501, arm, "/o", PARAMS)
            assert new == old, arm
    assert dict(mod.RUN_PREFIX) == {k: v for k, v in mc.RUN_PREFIX.items() if k != "ms_r4"}
    with pytest.raises(KeyError):
        mc.build_config(PROTO, 50, 10501, "fb_r", "/o", PARAMS)                                     # not resolvable without the table


# ---------------------------------------------------------------------------------------------- 2. the waves
def test_wave_sizes_and_layout(tmp_path):
    jobs = RL.build_jobs("r4", rc.QS, rc.SEEDS, None, tmp_path, PARAMS)
    assert len(jobs) == 280
    assert {j.cfg["arm"] for j in jobs} == set(rc.ARMS) and {j.cfg["q"] for j in jobs} == {50, 60}
    assert sorted({j.cfg["seed"] for j in jobs}) == list(rc.SEEDS)
    for arm in rc.ARMS:
        assert sum(1 for j in jobs if j.cfg["arm"] == arm) == 40
    j0 = jobs[0]
    assert j0.out_dir == str(tmp_path / "pilot" / "q50" / "seed10511" / "t1_st_s16") and j0.cfg["record"]["output_dir"] == j0.out_dir
    cj = RL.build_jobs("r4_cms6", rc.QS, rc.SEEDS_CMS6, None, tmp_path, PARAMS)
    assert len(cj) == 40 and {j.cfg["arm"] for j in cj} == set(rc.CMS6_ARMS)
    assert cj[0].out_dir == str(tmp_path / "cms6" / "q50" / "seed10501" / "t1_st_s16")
    for j in jobs + cj:
        assert j.cfg["seed"] not in rc.FORBIDDEN_SEEDS
    assert len({j.cfg["run"] for j in jobs + cj}) == 320


@pytest.mark.parametrize("wave,bad", [("r4", 30501), ("r4", 30520), ("r4", 40501), ("r4", 40520), ("r4_cms6", 30501),
                                      ("r4_cms6", 40510), ("r4", 10501), ("r4", 10510), ("r4", 10531), ("r4", 10510),
                                      ("r4_cms6", 10511), ("r4_cms6", 10530), ("r4_cms6", 10500)])
def test_the_seed_guards(tmp_path, wave, bad):
    with pytest.raises(ValueError):
        RL.build_jobs(wave, rc.QS, [bad], None, tmp_path, PARAMS)
    with pytest.raises(SystemExit):
        RL.main(["--wave", wave, "--seeds", str(bad), "--params", str(PARAMS_FILE), "--plan-csv", str(tmp_path / "p.csv")])
    assert not (tmp_path / "p.csv").exists()


def test_each_wave_accepts_exactly_its_own_block(tmp_path):
    RL.check_seeds("r4", list(range(10511, 10531)))
    RL.check_seeds("r4_cms6", list(range(10501, 10511)))
    for s in list(range(30501, 30521)) + list(range(40501, 40521)):
        for wave in RL.WAVES:
            with pytest.raises(ValueError):
                RL.check_seeds(wave, [s])


def test_arm_and_q_guards(tmp_path):
    with pytest.raises(ValueError):
        RL.build_jobs("r4_cms6", rc.QS, rc.SEEDS_CMS6, ["fb_r"], tmp_path, PARAMS)           # fb_r is not a cms6 arm
    with pytest.raises(ValueError):
        RL.build_jobs("r4", [55], rc.SEEDS, None, tmp_path, PARAMS)
    with pytest.raises(ValueError):
        RL.build_jobs("r4", rc.QS, rc.SEEDS, ["NL_st_s16"], tmp_path, PARAMS)


def test_the_plan_csv_hashes_every_run_config_and_creates_nothing(tmp_path):
    p = tmp_path / "plan.csv"
    assert RL.main(["--wave", "r4", "--params", str(PARAMS_FILE), "--plan-csv", str(p), "--root", str(tmp_path / "res")]) == 0
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 280 and len({r["run_config_sha256"] for r in rows}) == 280 and not (tmp_path / "res").exists()
    one = next(r for r in rows if r["arm"] == "fb_rg" and r["q"] == "60" and r["seed"] == "10530")
    cfg = rc.build_config_r4(PROTO, 60, 10530, "fb_rg", one["out_dir"], PARAMS)
    import hashlib
    assert hashlib.sha256(json.dumps(cfg, indent=1).encode()).hexdigest() == one["run_config_sha256"] == rc.config_sha256(cfg)
    with pytest.raises(FileExistsError):                                                      # never replaces a plan
        RL.write_plan_csv(p, [])


def test_a_real_launch_needs_the_preregistered_parameter_file(tmp_path):
    with pytest.raises(ValueError, match="passed explicitly"):
        RL.load_params(None, True)
    other = tmp_path / "other.json"
    other.write_text(PARAMS_FILE.read_text().replace("0.5", "0.6", 1))
    with pytest.raises(ValueError, match="SHA-256"):
        RL.load_params(str(other), True)
    assert RL.load_params(str(PARAMS_FILE), True) == PARAMS and RL.PARAMS_SHA256 == mc.sha256_file(PARAMS_FILE)
    assert RL.load_params(str(other), False)                                                   # tests and dry runs may use another file


def test_workers_are_capped_at_40(tmp_path):
    with pytest.raises(SystemExit):
        RL.main(["--wave", "r4", "--params", str(PARAMS_FILE), "--workers", "41", "--root", str(tmp_path)])
    assert RL.MAX_WORKERS == 40


def test_a_dry_run_writes_validated_configs_and_a_record(tmp_path, capsys):
    root = tmp_path / "res"
    code = RL.main(["--wave", "r4", "--params", str(PARAMS_FILE), "--dry-run", "--root", str(root), "--qs", "50", "--seeds", "10511",
                    "--arms", "fb_rg", "lrsm_st_s16", "--code-commit", "HEAD"])
    assert code == 0
    recs = list((root / "pilot").glob("dryrun_*.json"))
    assert len(recs) == 1
    rec = json.load(open(recs[0]))
    assert rec["dry_run"] is True and rec["n_planned"] == 2 and rec["validation"]["n_invalid"] == 0 and rec["tool"] == "tools/ms/r4_launch.py"
    for arm in ("fb_rg", "lrsm_st_s16"):
        assert (root / "pilot" / "q50" / "seed10511" / arm / "run_config.json").exists()
    # a second dry run into the same place is refused (the out dirs hold configs, not runs: no status.json -> allowed); a launch
    # record is never replaced
    with pytest.raises(FileExistsError):
        L1.write_record(recs[0], {}, create=True)


# ---------------------------------------------------------------------------------------------- 3. the existing launcher
def test_the_existing_launcher_is_unchanged():
    assert L1.WAVES == ("base", "v20_repro", "pilot", "r2", "r3")
    assert L1.WAVE_DIR == {"base": "base", "v20_repro": "v20_reproduction", "pilot": "pilot", "r2": "pilot", "r3": "pilot"}
    assert L1.MAX_WORKERS == 40
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", H.BASE_COMMIT, "--", "tools/ms/launch_ms_r1.py", "run/run_v2_T2_locked.py",
                           "run/run_v2_stagewise.py", "run/v2_rollout.py", "utils/v2_continuation.py", "utils/v2_metrics.py",
                           "utils/dp_br_verifier.py", "envs", "tools/v2", "protocols"], capture_output=True, text=True)
    if diff.returncode == 0:
        assert diff.stdout.strip() == "", diff.stdout          # the frozen files are byte-identical to the base commit
