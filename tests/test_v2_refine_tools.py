"""Tests of the R1 wave launcher (tools/v2/launch_refine.py) and the C-R1 comparison tool
(tools/v2/cr1_compare.py).

Launcher: job counts of the four waves (20 / 20 / 160 / 220), arm tables, every arm differs from
its base arm in exactly the pre-registered config keys, explicit new keys, worker cap, refusal to
overwrite, launch record, validation of the configs. Comparison tool: synthetic pairs of tiny run
directories (identical -> ALL true; one perturbed value -> the first differing field is named with
both values), window selection of the stage modes. Nothing here runs a training.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

import cr1_compare as C  # noqa: E402
import launch_refine as L  # noqa: E402

QS, SEEDS = (50, 60), tuple(range(10501, 10511))
EXPECTED_JOBS = {"v11_repro": 20, "parents_A": 20, "stage1": 160, "stage2": 220}
STAGE1_ARMS = ["B_base", "B_polish1", "B_polish2", "B_batch", "B_batch_mb256", "B_kl005", "B_kl010",
               "B_expcont"]
STAGE2_ARMS = ["A_base", "A_polish1", "A_polish2", "A_batch", "A_batch_mb256", "A_kl005", "A_kl010",
               "A_anneal2", "A_anneal4", "A_ctrl200", "A_detmean"]
BATCH = {"budget_overrides.episodes_per_update", "ppo_overrides.minibatch"}
BATCH256 = {"budget_overrides.episodes_per_update"}
# PROMPT.md 4.2 / 4.3 written out independently of the module constant EXPECTED_DIFFS
PREREG: Dict[str, Dict[str, Tuple[str, set]]] = {
    "stage1": {
        "B_polish1": ("B_base", {"lr_decay"}), "B_polish2": ("B_base", {"lr_decay"}),
        "B_batch": ("B_base", BATCH), "B_batch_mb256": ("B_base", BATCH256),
        "B_kl005": ("B_base", {"ppo_overrides.target_kl"}),
        "B_kl010": ("B_base", {"ppo_overrides.target_kl"}),
        "B_expcont": ("B_base", {"continuation_value_mode"}),
    },
    "stage2": {
        "A_polish1": ("A_base", {"lr_decay"}), "A_polish2": ("A_base", {"lr_decay"}),
        "A_batch": ("A_base", BATCH), "A_batch_mb256": ("A_base", BATCH256),
        "A_kl005": ("A_base", {"ppo_overrides.target_kl"}),
        "A_kl010": ("A_base", {"ppo_overrides.target_kl"}),
        "A_anneal2": ("A_base", {"conc_anneal"}), "A_anneal4": ("A_base", {"conc_anneal"}),
        "A_ctrl200": ("A_base", {"parent_checkpoint", "parent_sha256", "lr_decay",
                                 "budget_overrides.phase_caps.A"}),
        "A_detmean": ("A_ctrl200", {"mode", "lr_decay", "budget_overrides.phase_caps.A",
                                    "budget_overrides.phase_caps.P",
                                    "budget_overrides.verifier_timeout"}),
    },
}


# =========================================================================== launcher
@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    """{wave: jobs} for the full default grid (parents of stage 2 do not exist yet)."""
    root = tmp_path_factory.mktemp("refine_root")
    return root, {w: L.build_configs(w, QS, SEEDS, None, root, require_parents=False)
                  for w in L.WAVES}


@pytest.mark.parametrize("wave", L.WAVES)
def test_job_counts(planned, wave):
    _, jobs = planned
    assert len(jobs[wave]) == EXPECTED_JOBS[wave]
    assert len({j.out_dir for j in jobs[wave]}) == EXPECTED_JOBS[wave]


def test_arm_tables_are_the_pilot_tables():
    assert list(L.STAGE1_ARMS) == STAGE1_ARMS
    assert list(L.STAGE2_ARMS) == STAGE2_ARMS
    assert list(L.PARENTS_A_ARMS) == ["A_parent"]
    assert L.METHOD_ARMS["5 deterministic mean"]["stage2"] == ("A_ctrl200", "A_detmean")


def test_expected_diffs_constant_matches_the_preregistration():
    for stage, table in PREREG.items():
        got = {a: (b, set(k)) for a, (b, k) in L.EXPECTED_DIFFS[stage].items()}
        assert got == table


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_every_arm_differs_in_exactly_the_preregistered_keys(planned, stage):
    _, jobs = planned
    cfgs = {(j.cfg["q"], j.cfg["seed"], j.cfg["arm"]): j.cfg for j in jobs[stage]}
    for (q, seed, arm), cfg in cfgs.items():
        if arm in L.BASE_ARM.values():
            continue
        base, keys = PREREG[stage][arm]
        assert set(L.arm_diff(cfg, cfgs[(q, seed, base)])) == keys, (q, seed, arm)


def test_stage1_arm_values(planned):
    _, jobs = planned
    c = {j.cfg["arm"]: j.cfg for j in jobs["stage1"] if (j.cfg["q"], j.cfg["seed"]) == (50, 10501)}
    for cfg in c.values():
        assert cfg["mode"] == "phase_B" and cfg["fixed_budget"] is True
        assert cfg["flags"] == {"reward_mode": "expected", "stage2_update_mode": "frozen",
                                "adv_norm_scope": "stage1_rows", "continuation_action_mode": "mean"}
        assert cfg["parent_checkpoint"] == str(L.REHEARSAL / "q50" / "seed10501" / "state_end_A.pt")
        assert cfg["parent_checkpoint"].startswith("/")
        assert len(cfg["parent_sha256"]) == 64
        assert cfg["lr_decay"] is not None and cfg["conc_anneal"] is None
    base = c["B_base"]
    assert base["lr_decay"] == [{"phase": "B", "start_lr": 3e-4, "end_lr": 3e-5,
                                 "local_first": 1, "local_last": 600}]
    assert base["budget_overrides"] == {"episodes_per_update": 512}
    assert base["ppo_overrides"] == {"minibatch": 256, "target_kl": None}
    assert base["continuation_value_mode"] == "sampled"
    assert c["B_polish1"]["lr_decay"] == [
        {"phase": "B", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 1, "local_last": 400},
        {"phase": "B", "start_lr": 3e-5, "end_lr": 3e-6, "local_first": 401, "local_last": 600}]
    assert c["B_polish2"]["lr_decay"] == [{"phase": "B", "start_lr": 3e-4, "end_lr": 3e-6,
                                           "local_first": 1, "local_last": 600}]
    assert c["B_batch"]["budget_overrides"]["episodes_per_update"] == 2048
    assert c["B_batch"]["ppo_overrides"]["minibatch"] == 1024
    assert c["B_batch_mb256"]["ppo_overrides"]["minibatch"] == 256
    assert c["B_kl005"]["ppo_overrides"]["target_kl"] == 0.005
    assert c["B_kl010"]["ppo_overrides"]["target_kl"] == 0.01
    assert c["B_expcont"]["continuation_value_mode"] == "expected"


def test_stage2_arm_values(planned):
    root, jobs = planned
    c = {j.cfg["arm"]: j.cfg for j in jobs["stage2"] if (j.cfg["q"], j.cfg["seed"]) == (60, 10510)}
    pa = {"reward_mode": "expected", "stage2_update_mode": "joint", "adv_norm_scope": "all_rows",
          "continuation_action_mode": "stochastic"}
    u1200 = str(Path(root).resolve() / "parents_A" / "q60" / "seed10510" / "state_u01200.pt")
    for arm, cfg in c.items():
        assert cfg["flags"] == pa and cfg["fixed_budget"] is True and "conc_anneal" in cfg
        end_a = str(L.REHEARSAL / "q60" / "seed10510" / "state_end_A.pt")
        assert cfg["parent_checkpoint"] == (end_a if arm in ("A_ctrl200", "A_detmean") else u1200)
        assert cfg["mode"] == ("phase_P" if arm == "A_detmean" else "phase_A_continue")
    assert c["A_base"]["budget_overrides"]["phase_caps"] == {"A": 400, "B": 600, "C": 1000}
    assert c["A_base"]["lr_decay"] == [{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-5,
                                        "local_first": 1, "local_last": 400}]
    assert c["A_base"]["conc_anneal"] is None
    assert c["A_polish1"]["lr_decay"] == [
        {"phase": "A", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 1, "local_last": 250},
        {"phase": "A", "start_lr": 3e-5, "end_lr": 3e-6, "local_first": 251, "local_last": 400}]
    assert c["A_polish2"]["lr_decay"] == [{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-6,
                                           "local_first": 1, "local_last": 400}]
    assert c["A_batch"]["budget_overrides"]["episodes_per_update"] == 2048
    assert c["A_batch"]["ppo_overrides"] == {"minibatch": 1024, "target_kl": None}
    assert c["A_batch_mb256"]["ppo_overrides"] == {"minibatch": 256, "target_kl": None}
    assert c["A_kl005"]["ppo_overrides"]["target_kl"] == 0.005
    assert c["A_kl010"]["ppo_overrides"]["target_kl"] == 0.01
    for arm, s in (("A_anneal2", 2.0), ("A_anneal4", 4.0)):
        assert c[arm]["conc_anneal"] == {"phase": "A", "local_first": 1, "local_last": 400,
                                         "scale_first": 1.0, "scale_last": s}
    assert c["A_ctrl200"]["budget_overrides"]["phase_caps"] == {"A": 200, "B": 600, "C": 1000}
    assert c["A_ctrl200"]["lr_decay"] == [{"phase": "A", "start_lr": 3e-5, "end_lr": 3e-5,
                                           "local_first": 1, "local_last": 200}]
    d = c["A_detmean"]
    assert d["budget_overrides"]["phase_caps"] == {"A": 1600, "B": 600, "C": 1000, "P": 200}
    assert d["budget_overrides"]["verifier_timeout"] == {"A": 100, "B": 25, "C": 25, "P": 50}
    assert d["lr_decay"] == [{"phase": "P", "start_lr": 3e-5, "end_lr": 3e-5, "local_first": 1,
                              "local_last": 200}]
    assert d["parent_checkpoint"] == c["A_ctrl200"]["parent_checkpoint"]
    assert d["parent_sha256"] == c["A_ctrl200"]["parent_sha256"]


def test_parents_a_values(planned):
    root, jobs = planned
    cfg, out = next(j for j in jobs["parents_A"] if (j.cfg["q"], j.cfg["seed"]) == (50, 10503))
    assert out == str(Path(root).resolve() / "parents_A" / "q50" / "seed10503")
    assert cfg["mode"] == "phase_A" and cfg["parent_checkpoint"] is None
    assert cfg["parent_sha256"] is None and cfg["full_state_at"] == [1200]
    assert cfg["lr_decay"] == [{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-5,
                                "local_first": 1201, "local_last": 1600}]
    assert cfg["flags"]["stage2_update_mode"] == "joint" and cfg["run"] == \
        "v2refine_parents_A_q50_s10503_A_parent"


def test_every_config_carries_the_protocol_record_and_explicit_new_keys(planned):
    _, jobs = planned
    proto = json.load(open(L.PROTOCOL_PATH))
    for wave in ("parents_A", "stage1", "stage2"):
        for cfg, out in jobs[wave]:
            assert cfg["schema"] == "v2_run_config/1" and cfg["base_commit"] == proto["base_commit"]
            assert cfg["pilot"] == f"v2_refine_{wave}" and cfg["fixed_budget"] is True
            assert cfg["run"] == f"v2refine_{wave}_q{cfg['q']}_s{cfg['seed']}_{cfg['arm']}"
            assert cfg["threads_per_process"] == proto["threads_per_process"]
            assert cfg["arm_definition"]
            for k in ("ppo_overrides", "continuation_value_mode"):
                assert k in cfg
            assert "episodes_per_update" in cfg["budget_overrides"]
            assert set(cfg["ppo_overrides"]) == {"minibatch", "target_kl"}
            rec = copy.deepcopy(cfg["record"])
            ref = copy.deepcopy(proto["records"][str(cfg["q"])])
            assert rec.pop("output_dir") == out and rec.pop("run") == cfg["run"]
            assert rec.pop("seed") == cfg["seed"]
            for k in ("output_dir", "run", "seed"):
                ref.pop(k)
            assert rec == ref
            assert L.lr_window_problems(cfg) == []
            json.dumps(cfg, allow_nan=False)


def test_v11_repro_runs_the_unchanged_entry_point(tmp_path):
    jobs = L.build_configs("v11_repro", [50], [10501], None, tmp_path)
    (cfg, out), = jobs
    assert L.is_locked_entry(cfg)
    assert out == str((tmp_path / "v11_reproduction" / "q50" / "seed10501").resolve())
    assert L.job_command(cfg, out) == [L.PY, "-B", "-u", str(ROOT / "run" / "run_v2_T2_locked.py"),
                                       "--q", "50", "--seed", "10501", "--out-dir", out]
    assert L.main(["--wave", "v11_repro", "--qs", "50", "--seeds", "10501", "--root", str(tmp_path),
                   "--dry-run"]) == 0
    assert not list(tmp_path.glob("v11_reproduction/q50/seed10501/run_config.json"))


def test_unknown_arm_and_q_refused(tmp_path):
    with pytest.raises(ValueError):
        L.build_configs("stage1", [50], [10501], ["A_base"], tmp_path)
    with pytest.raises(ValueError):
        L.build_configs("stage1", [70], [10501], None, tmp_path)
    with pytest.raises(FileNotFoundError):
        L.build_configs("stage2", [50], [10501], ["A_base"], tmp_path, require_parents=True)


@pytest.mark.parametrize("wave", ["parents_A", "stage1", "stage2"])
def test_dry_run_writes_configs_and_record(tmp_path, wave):
    assert L.main(["--wave", wave, "--root", str(tmp_path), "--dry-run", "--no-validate",
                   "--code-commit", "HEAD"]) == 0
    cfgs = list((tmp_path / wave).glob("q*/seed*/**/run_config.json"))
    assert len(cfgs) == EXPECTED_JOBS[wave]
    (rec_path,) = (tmp_path / wave).glob("dryrun_*.json")
    assert not list((tmp_path / wave).glob("launch_*.json"))
    rec = json.load(open(rec_path))
    assert rec["n_planned"] == EXPECTED_JOBS[wave] and rec["dry_run"] is True
    assert rec["nproc"] and len(rec["loadavg_at_start"]) == 3 and rec["disk_free_bytes"] > 0
    assert rec["head"] and rec["code_commit"] == "HEAD"
    assert rec["diff_stat_code_commit_to_head"] == ""
    assert isinstance(rec["status_porcelain"], list)
    one = json.load(open(cfgs[0]))
    assert Path(one["record"]["output_dir"]).resolve() == cfgs[0].parent.resolve()


def test_worker_cap(tmp_path):
    base = ["--wave", "stage1", "--root", str(tmp_path), "--dry-run", "--no-validate", "--qs", "50",
            "--seeds", "10501"]
    for bad in ("41", "0", "100"):
        with pytest.raises(SystemExit) as e:
            L.main(base + ["--workers", bad])
        assert e.value.code == 2
    assert not list(tmp_path.rglob("*.json"))
    assert L.main(base + ["--workers", "40"]) == 0


def test_refuses_to_overwrite_a_previous_run(tmp_path):
    out = tmp_path / "stage1" / "q50" / "seed10501" / "B_base"
    out.mkdir(parents=True)
    (out / "status.json").write_text("{}")
    args = ["--wave", "stage1", "--root", str(tmp_path), "--no-validate", "--qs", "50", "--seeds",
            "10501", "--arms", "B_base"]
    for extra in ([], ["--dry-run"]):
        with pytest.raises(SystemExit) as e:
            L.main(args + extra)
        assert e.value.code == 2
    assert not list((tmp_path / "stage1").glob("*.json")) and not (out / "run_config.json").exists()
    (out / "status.json").unlink()
    (out / "run.log").write_text("old log")
    with pytest.raises(SystemExit):
        L.main(args)
    assert (out / "run.log").read_text() == "old log"


def test_run_job_never_overwrites_a_log(tmp_path):
    cmd = [sys.executable, "-c", "print('hello')"]
    res = L.run_job(cmd, str(tmp_path), L.child_env())
    assert res["returncode"] == 0 and res["wall_sec"] >= 0
    assert (tmp_path / "run.log").read_text().strip() == "hello"
    with pytest.raises(FileExistsError):
        L.run_job(cmd, str(tmp_path), L.child_env())
    assert (tmp_path / "run.log").read_text().strip() == "hello"


def test_launch_record_and_bounded_pool(tmp_path, monkeypatch):
    def fake_command(cfg: Dict[str, Any], out_dir: str) -> List[str]:
        code = ("import os, sys; print(os.environ['OMP_NUM_THREADS'], "
                "os.environ['MKL_NUM_THREADS'], os.environ['OPENBLAS_NUM_THREADS']); "
                f"sys.exit(3 if {cfg['seed']} == 10502 else 0)")
        return [sys.executable, "-c", code]
    monkeypatch.setattr(L, "job_command", fake_command)
    rc = L.main(["--wave", "v11_repro", "--qs", "50", "--seeds", "10501", "10502", "10503",
                 "--workers", "2", "--root", str(tmp_path), "--code-commit", "HEAD"])
    assert rc == 1
    (rec_path,) = (tmp_path / "v11_reproduction").glob("launch_*.json")
    rec = json.load(open(rec_path))
    assert rec["state"] == "failed" and rec["n_planned"] == 3 and len(rec["runs"]) == 3
    assert sorted(r["returncode"] for r in rec["runs"]) == [0, 0, 3]
    assert rec["workers"] == 2 and rec["dry_run"] is False and "loadavg_at_end" in rec
    for r in rec["runs"]:
        assert (Path(r["out"]) / "run.log").read_text().split() == ["1", "1", "1"]
        assert r["wall_sec"] >= 0
        assert not (Path(r["out"]) / "run_config.json").exists()


def test_configs_validate_or_fail_only_on_new_keys(tmp_path):
    """Every config passes validate_config as written, or after dropping the new keys."""
    qs, seeds = [50], [10501]
    jobs = []
    for wave in ("parents_A", "stage1", "stage2"):
        jobs += L.build_configs(wave, qs, seeds, None, tmp_path, require_parents=False)
    rows = L.validate_jobs(jobs)
    assert len(rows) == len(jobs) == 1 + 8 + 11
    for r in rows:
        assert r["ok"] or r["legacy_ok"], r
        assert r["window_problems"] == []
    summary = L.summarize_validation(rows)
    assert set(summary) == {"A_parent"} | set(STAGE1_ARMS) | set(STAGE2_ARMS)


# =========================================================================== comparison tool
UPD_A = (1, 2, 25, 1200, 1201, 1220, 1225, 1300, 1600)
UPD_B = (1601, 1602, 1625, 2200)
LOCKED_FLAGS = ("expected", "frozen", "stage1_rows", "mean")
PHASEA_FLAGS = ("expected", "joint", "all_rows", "stochastic")


def _phase(u: int) -> str:
    return "A" if u <= 1600 else "B"


def _start(kind: str, u: int) -> int:
    return {"locked": 0 if u <= 1600 else 1600, "phase_A": 0, "phase_B": 1600,
            "phase_A_continue": 1200}[kind]


def _updates(kind: str) -> Tuple[int, ...]:
    return {"locked": UPD_A + UPD_B, "phase_A": UPD_A, "phase_B": UPD_B,
            "phase_A_continue": tuple(u for u in UPD_A if u > 1200)}[kind]


def _arr(key: int, n: int = 5) -> np.ndarray:
    return np.random.default_rng(key).normal(size=n).astype(np.float32)


def _rng_state(key: int) -> Dict[str, Any]:
    return {"bit_generator": "PCG64", "state": {"state": 10 ** 20 + key, "inc": 7 * key + 1},
            "has_uint32": 0, "uinteger": 0}


def _tensor(key: int, n: int = 4) -> torch.Tensor:
    return torch.randn(n, generator=torch.Generator().manual_seed(key))


def _state(key: int, refreshes: int, frozen: bool, gseed: int) -> Dict[str, Any]:
    def net(k: int) -> Dict[str, torch.Tensor]:
        return {"w": _tensor(k), "b": _tensor(k + 1, 2)}

    def opt(k: int) -> Dict[str, Any]:
        return {"state": {0: {"step": torch.tensor(float(key)), "m": _tensor(k)}},
                "param_groups": [{"lr": 3e-5}]}
    names = ("env", "learn", "opp", "start")
    return {"format": "v2_full_state/1",
            "agent": {"actor": net(key), "critic": net(key + 10), "opponent": net(key + 20),
                      "frozen": net(key + 30) if frozen else None,
                      "opt_actor": opt(key + 2), "opt_critic": opt(key + 3),
                      "rng_minibatch": _rng_state(key + 4), "snapshot_refreshes": refreshes,
                      "cfg": {}},
            "rng": {n: _rng_state(key + 5 + i) for i, n in enumerate(names)},
            "torch_generator_state": torch.randint(0, 255, (8,), dtype=torch.uint8,
                                                   generator=torch.Generator().manual_seed(key)),
            "torch_global_rng_state": _tensor(gseed, 3), "numpy_global_rng_state": gseed,
            "python_random_state": gseed,
            "counters": {"global_u": 1600 if key < 100 else 2200, "total_episodes": key,
                         "total_transitions": key, "snapshot_refreshes": refreshes,
                         "next_snapshot_refresh_update": 1620}}


def _scal(tier: str, ph: str) -> Dict[str, Any]:
    t = 1.0 if tier == "final" else 1.1
    d = {"eta_T_over_dw": 1e-3 * t, "stage2_rmse_pos_over_g2_0": 2e-3 * t,
         "stage2_tail_mean_over_g2_0": 3e-3 * t, "valid": True}
    d.update({"Gmax_full_over_dw": 0.8 * t, "stage1_rel_err_abs": 0.5} if ph == "A" else
             {"Gmax_full_over_dw": 4e-3 * t, "stage1_rel_err_abs": 0.02 * t})
    return d


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    np.savez(path, **arrays)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1))


def _hist(kind: str, u: int) -> Dict[str, Any]:
    return {"update": u, "phase": _phase(u), "local": u - _start(kind, u),
            "actor_lr": 3e-4 - 1e-7 * u, "kl_final_epoch": 1e-3 * (u % 7 + 1),
            "kl_epochs": [0.1 * (u % 5), 0.2 * (u % 3)], "snapshot_refreshed": u % 20 == 0}


def _stab(kind: str, u: int) -> Dict[str, Any]:
    first = kind == "phase_A_continue" and u == 1220
    return {"update": u, "phase": _phase(u), "local": u - _start(kind, u),
            "drift": None if first else 0.01 * (u % 3 + 1), "kl": 1e-3 * (u % 4 + 1),
            "stable": False if first else True, "consecutive": 0 if first else u % 5,
            "max_std_norm": 0.01 * (u % 3), "e_hat": {"2": [u * 1e-3, u * 2e-3]}}


def _ver_updates(kind: str) -> Tuple[int, ...]:
    return {"locked": (1300, 1600, 1625, 2200), "phase_A": (1300, 1600), "phase_B": (1625, 2200),
            "phase_A_continue": (1300, 1400, 1600)}[kind]


def _ver(kind: str, u: int, gseed: int) -> Dict[str, Any]:
    cont = kind == "phase_A_continue"
    return {"update": u, "phase": _phase(u), "local": u - _start(kind, u),
            "reason": "warmup_forced" if cont and u == 1300 else "timeout",
            "actor_lr": 3e-4 - 1e-7 * u, "valid": True, "criterion_value_over_dw": 1e-5 * u,
            "consecutive_eligible": 1 if cont else u % 4, "time_sec": 0.5 * u + gseed,
            "summary": {"x": u * 1.0},
            "visitation_cumulative_phase": {"1": [u + (5000 if cont else 0)]}}


def _snapshots(kind: str) -> List[Dict[str, Any]]:
    def every(us) -> List[Dict[str, Any]]:
        return [{"update": u, "reason": "every_20"} for u in us if u % 20 == 0]
    head = [{"update": 0, "reason": "init"}, {"update": 0, "reason": "phase_A_entry"}]
    if kind == "phase_A":
        return head + every(UPD_A)
    if kind in ("locked", "phase_B"):    # a phase-B run restores the parent's log
        return head + every(UPD_A) + [{"update": 1600, "reason": "phase_B_entry"}] + every(UPD_B)
    pre = head + [{"update": 1200, "reason": "every_20"}]
    later = [u for u in UPD_A if u > 1200]
    return pre + [{"update": 1200, "reason": "phase_A_entry"}] + every(later)


def build(d: Path, kind: str, gseed: int = 0, extra_cols: Tuple[str, ...] = ()) -> Path:
    """A tiny run directory of the given kind (locked | phase_A | phase_B | phase_A_continue)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "weights").mkdir(exist_ok=True)
    ups = _updates(kind)
    locked = kind == "locked"
    # ---- states
    if kind in ("locked", "phase_A", "phase_A_continue"):
        refreshes = 82 + (2 if kind == "phase_A_continue" else 0)
        torch.save(_state(1, refreshes, False, gseed), d / "state_end_A.pt")
    if kind in ("locked", "phase_B"):
        torch.save(_state(101, 120, True, gseed), d / "state_end_B.pt")
    # ---- weights
    for u in ups:
        if u % 25 == 0:
            _write_npz(d / "weights" / f"u{u:05d}.npz", w=_arr(u), b=_arr(u + 1, 2))
    ends_b = kind in ("locked", "phase_B")
    _write_npz(d / "checkpoint_weights.npz", w=_arr(9999 if ends_b else 9998))
    # ---- train_history.json
    cont = kind == "phase_A_continue"
    cur_a = {"phase": "A", "entry_update": 1200 if cont else 0, "exit_update": 1600,
             "local_updates": 400 if cont else 1600, "cap": 1600}
    cur_b = {"phase": "B", "entry_update": 1600, "exit_update": 2200, "local_updates": 600,
             "cap": 600}
    cur = {"locked": [cur_a, cur_b], "phase_A": [cur_a], "phase_A_continue": [cur_a],
           "phase_B": [cur_b]}
    hist = {"run": "x", "group": "g", "history": [_hist(kind, u) for u in ups],
            "stability": [_stab(kind, u) for u in ups if u % 20 == 0],
            "verifier_calls": [_ver(kind, u, gseed) for u in _ver_updates(kind)],
            "curriculum": cur[kind], "snapshots": _snapshots(kind),
            "weight_checkpoints": [{"update": u, "phase": _phase(u), "local": u - _start(kind, u),
                                    "file": f"weights/u{u:05d}.npz"} for u in ups if u % 25 == 0],
            "stopping_record": None}
    _write_json(d / "train_history.json", hist)
    # ---- CSVs
    upd = pd.DataFrame([{"update": u, "phase": _phase(u), "local": u - _start(kind, u),
                         "kl_final_epoch": 1e-3 * (u % 7 + 1),
                         "adv_norm_scope": "stage1_rows" if u > 1600 else "all_rows",
                         "update_wall_sec": 0.01 * u + gseed, "rngpos_env": f"{u:x}:0:0"}
                        for u in ups])
    for c in extra_cols:
        upd[c] = 1.5
    upd.to_csv(d / "v2_updates.csv", index=False)
    fl = LOCKED_FLAGS if locked else PHASEA_FLAGS
    if kind == "phase_B":
        fl = LOCKED_FLAGS
    for ph in ("A", "B"):
        rows = [{"run": "locked_run" if locked else "refine_run", "pilot": "p", "arm": "a",
                 "mode": kind, "reward_mode": fl[0], "stage2_update_mode": fl[1],
                 "adv_norm_scope": fl[2], "continuation_action_mode": fl[3], "update": u,
                 "phase": ph, "local": u - _start(kind, u), "reason": _ver(kind, u, 0)["reason"],
                 "consecutive_eligible": _ver(kind, u, 0)["consecutive_eligible"],
                 "Gmax_full_over_dw": 1e-4 * u, "verifier_sec": 0.1 + gseed,
                 "elapsed_phase_wall_sec": 5.0 + gseed}
                for u in _ver_updates(kind) if _phase(u) == ph]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        for c in extra_cols:
            df[c] = 2.5
        name = f"v2_checkpoints_{ph}.csv" if locked else "v2_checkpoints.csv"
        df.to_csv(d / name, index=False)
    # ---- evaluation outputs
    end = "B" if kind in ("locked", "phase_B") else "A"
    for tier in ("final", "development"):
        g = 1 if tier == "final" else 2
        if locked:
            _write_npz(d / f"gateA_{tier}.npz", v_t1_G=_arr(500 + g, 3))
        _write_npz(d / f"final_{tier}.npz", v_t1_G=_arr(500 + g + (10 if end == "B" else 0), 3))
    if locked:
        _write_npz(d / "band_sweep.npz", e=_arr(7, 6))
        _write_json(d / "induced_band.json", {"e_tilde": 56.1, "band": [55.0, 57.0]})
        _write_json(d / "drift_test.json", {"pass": True, "max": 0.0})
        sa = {t: _scal(t, "A") for t in ("final", "development")}
        sb = {t: _scal(t, "B") for t in ("final", "development")}
        sa["final"]["stage2_peak_locfree_rel_err"] = -0.01
        sa["development"]["stage2_peak_locfree_rel_err"] = -0.01
        fin_a, dev_a, fin_b, dev_b = sa["final"], sa["development"], sb["final"], sb["development"]
        mv = {"eta_final": fin_a["eta_T_over_dw"], "eta_dev": dev_a["eta_T_over_dw"],
              "rmse": fin_a["stage2_rmse_pos_over_g2_0"],
              "tail": fin_a["stage2_tail_mean_over_g2_0"],
              "gmax_final": fin_b["Gmax_full_over_dw"], "gmax_dev": dev_b["Gmax_full_over_dw"],
              "s1": fin_b["stage1_rel_err_abs"]}
        g_a = ("eta_T_over_dw", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0")
        dv = {"G-A": {k: dev_a[k] for k in g_a},
              "G-F": {"Gmax_full_over_dw": dev_b["Gmax_full_over_dw"]},
              "S1": {"stage1_rel_err_abs": dev_b["stage1_rel_err_abs"]}}
        _write_json(d / "gates.json", {
            "metric_values": mv, "dev_tier_values": dv,
            "reported": {"end_of_A": sa, "end_of_B": sb},
            "G-A": {"pass": True}, "G-F": {"pass": True}, "G-N": {"pass": True},
            "outcome": "pass", "commit": f"c{gseed}"})
    else:
        _write_json(d / "final_v2.json", {"stage1_status": "x", "final": _scal("final", end),
                                          "development": _scal("development", end)})
    return d


def roots(tmp: Path, kind: str, arm: Optional[str] = None, **kw: Any) -> Tuple[Path, Path]:
    """(ref_root, new_root) with one run each at q50 / seed10501."""
    build(tmp / "ref" / "q50" / "seed10501", "locked")
    new = tmp / "new" / "q50" / "seed10501"
    build(new / arm if arm else new, kind, **kw)
    return tmp / "ref", tmp / "new"


def run(mode: str, tmp: Path, kind: str, arm: Optional[str] = None, **kw: Any) -> Dict[str, Any]:
    ref, new = roots(tmp, kind, arm, **kw)
    out = C.compare_roots(mode, ref, new, [50], [10501], arm)
    return out


def _edit_json(path: Path, fn) -> None:
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj, indent=1))


def _ulp(x: float) -> float:
    return float(np.nextafter(np.float32(x), np.float32(np.inf)))


def test_first_diff_unit():
    t = torch.tensor([1.0, 2.0, 3.0])
    nan = float("nan")
    assert C.first_diff({"a": [t, 1], "b": nan}, {"a": [t.clone(), 1.0], "b": nan}) is None
    d = C.first_diff({"a": [t, 1]}, {"a": [torch.tensor([1.0, 2.5, 3.0]), 1]})
    assert d.path == "a[0][1]" and d.ref == "2.0" and d.new == "2.5"
    assert C.first_diff({"a": 1}, {"b": 1}).path == "{keys}"
    assert C.first_diff([1, 2], [1, 2, 3]).path == ".len"
    assert C.first_diff(t, t.double()).path.endswith(".dtype")
    assert C.first_diff(np.zeros(3), np.zeros(4)).path.endswith(".shape")
    assert C.first_diff({"k": None}, {"k": 0.0}) is not None
    assert C.first_diff(True, 1) is not None


def test_cr1_identical_ignoring_wall_clock_global_rng_and_new_columns(tmp_path):
    new_cols = ("d1_L_s2_n", "n_epochs_run", "conc_scale")
    out = run("cr1", tmp_path, "locked", gseed=7, extra_cols=new_cols)
    (r,) = out["runs"]
    assert r["ALL"] is True and r["first_difference"] is None and r["differences"] == {}
    assert out["ALL"] and out["n_identical"] == out["n"] == 1 and out["failing_fields"] == []
    assert all(r["fields"].values()) and len(r["fields"]) > 40
    cols = r["info"]["csv_columns_only_in_one"]["v2_updates.csv"]
    assert cols["only_in_new"] == list(new_cols) and cols["only_in_ref"] == []
    assert cols["unexpected_new"] == []
    assert r["info"]["state_end_A.pt:snapshot_refreshes"]["equal"] is True


def test_cr1_unexpected_new_column_is_reported_not_compared(tmp_path):
    (r,) = run("cr1", tmp_path, "locked", extra_cols=("mystery",))["runs"]
    assert r["ALL"] is True
    only = r["info"]["csv_columns_only_in_one"]
    assert only["v2_checkpoints_A.csv"]["unexpected_new"] == ["mystery"]


def test_cr1_perturbed_weight_array_names_the_field_and_values(tmp_path):
    ref, new = roots(tmp_path, "locked")
    p = new / "q50" / "seed10501" / "weights" / "u01200.npz"
    z = dict(np.load(p))
    old = float(z["w"][2])
    z["w"][2] = _ulp(old)
    np.savez(p, **z)
    out = C.compare_roots("cr1", ref, new, [50], [10501])
    (r,) = out["runs"]
    assert not r["ALL"] and not out["ALL"] and out["n_identical"] == 0
    assert [k for k, v in r["fields"].items() if not v] == ["weight_exports"]
    fd = r["first_difference"]
    assert fd["field"] == "weight_exports" and fd["path"] == "u01200.npz.w[2]"
    assert float(fd["ref"]) == old and float(fd["new"]) == _ulp(old) and "1 of 5" in fd["note"]
    assert out["first_differences"][0]["seed"] == 10501


def test_cr1_state_difference_is_reported_before_later_fields(tmp_path):
    ref, new = roots(tmp_path, "locked")
    p = new / "q50" / "seed10501" / "state_end_B.pt"
    s = torch.load(p, weights_only=False)
    s["agent"]["frozen"]["w"][1] += 1e-6
    torch.save(s, p)
    q = new / "q50" / "seed10501" / "weights" / "u00025.npz"
    z = dict(np.load(q))
    z["w"][0] += 1.0
    np.savez(q, **z)
    (r,) = C.compare_roots("cr1", ref, new, [50], [10501])["runs"]
    assert r["first_difference"]["field"] == "state_end_B.pt:frozen"
    assert r["first_difference"]["path"] == "frozen.w[1]"
    assert set(r["differences"]) == {"state_end_B.pt:frozen", "weight_exports"}


def test_cr1_csv_history_and_gate_differences(tmp_path):
    ref, new = roots(tmp_path, "locked")
    d = new / "q50" / "seed10501"
    csv = pd.read_csv(d / "v2_updates.csv", dtype=str, keep_default_na=False)
    csv.loc[csv["update"] == "1220", "kl_final_epoch"] = "0.123"
    csv.to_csv(d / "v2_updates.csv", index=False)
    _edit_json(d / "train_history.json", lambda o: o["history"][5]["kl_epochs"].__setitem__(1, 9.0))
    _edit_json(d / "gates.json", lambda o: o["metric_values"].__setitem__("gmax_final", 0.5))
    (r,) = C.compare_roots("cr1", ref, new, [50], [10501])["runs"]
    bad = [k for k, v in r["fields"].items() if not v]
    assert bad == ["train_history:history", "v2_updates.csv", "gates:metric_values"]
    csv_path = r["differences"]["v2_updates.csv"]["path"]
    assert csv_path == "v2_updates.csv[update=1220].kl_final_epoch"
    assert r["differences"]["v2_updates.csv"]["new"] == "'0.123'"
    assert r["differences"]["train_history:history"]["path"] == "history.1220.kl_epochs[1]"
    assert r["differences"]["train_history:history"]["new"] == "9.0"
    assert r["differences"]["gates:metric_values"]["path"] == "gates.metric_values.gmax_final"


def test_cr1_missing_run_dir_and_missing_file(tmp_path):
    ref, new = roots(tmp_path, "locked")
    out = C.compare_roots("cr1", ref, new, [50, 60], [10501])
    assert out["n"] == 2 and out["n_identical"] == 1 and not out["ALL"]
    assert out["runs"][1]["first_difference"]["field"] == "run_dir_present"
    (new / "q50" / "seed10501" / "gates.json").unlink()
    (r,) = C.compare_roots("cr1", ref, new, [50], [10501])["runs"]
    fd = r["first_difference"]
    assert fd["field"] == "gates:metric_values" and fd["path"] == "<error>"
    assert "gates.json" in fd["ref"]


def test_cli_summary_file_and_exit_code(tmp_path, capsys):
    ref, new = roots(tmp_path, "locked")
    out = tmp_path / "res" / "checks.json"
    args = ["--mode", "cr1", "--ref", str(ref), "--new", str(new), "--qs", "50", "--seeds", "10501",
            "--out", str(out)]
    assert C.main(args) == 0
    assert json.load(open(out))["n_identical"] == 1
    p = new / "q50" / "seed10501" / "induced_band.json"
    _edit_json(p, lambda o: o.__setitem__("e_tilde", 56.2))
    assert C.main(args) == 2
    txt = capsys.readouterr().out
    assert "first differing field induced_band.json" in txt
    assert "ref=56.1" in txt and "new=56.2" in txt
    assert json.load(open(out))["failing_fields"] == ["induced_band.json"]


def test_parentsA_compares_only_the_phase_a_window(tmp_path):
    ref, new = roots(tmp_path, "phase_A")
    out = C.compare_roots("parentsA", ref, new, [50], [10501])
    (r,) = out["runs"]
    assert r["ALL"], r["differences"]
    assert r["info"]["n_compared_history"] == len(UPD_A)
    assert r["info"]["n_weight_exports_compared"] == 5
    # a change in the reference's Phase-B part is outside the window
    _edit_json(ref / "q50" / "seed10501" / "train_history.json",
               lambda o: o["history"][len(UPD_A)].__setitem__("kl_final_epoch", 5.0))
    assert C.compare_roots("parentsA", ref, new, [50], [10501])["ALL"]
    # a change at update 1600 is inside it
    _edit_json(ref / "q50" / "seed10501" / "train_history.json",
               lambda o: o["history"][len(UPD_A) - 1].__setitem__("kl_final_epoch", 5.0))
    (r,) = C.compare_roots("parentsA", ref, new, [50], [10501])["runs"]
    assert r["first_difference"]["field"] == "train_history:history"
    assert r["first_difference"]["path"] == "history.1600.kl_final_epoch"


def test_parentsA_end_of_a_evaluation_and_state(tmp_path):
    ref, new = roots(tmp_path, "phase_A")
    _edit_json(new / "q50" / "seed10501" / "final_v2.json",
               lambda o: o["final"].__setitem__("eta_T_over_dw", 9e-3))
    p = new / "q50" / "seed10501" / "state_end_A.pt"
    s = torch.load(p, weights_only=False)
    s["agent"]["opt_actor"]["state"][0]["m"][0] += 1.0
    torch.save(s, p)
    (r,) = C.compare_roots("parentsA", ref, new, [50], [10501])["runs"]
    assert r["first_difference"]["field"] == "state_end_A.pt:opt_actor"
    assert r["first_difference"]["path"] == "opt_actor.state.0.m[0]"
    assert r["differences"]["end_of_A.final_scalars"]["path"] == "end_of_A.final.eta_T_over_dw"
    assert r["info"]["end_of_A.final_keys_not_in_new"] == ["stage2_peak_locfree_rel_err"]


def test_stage1_base_window_and_end_of_b_values(tmp_path):
    ref, new = roots(tmp_path, "phase_B", arm="B_base")
    (r,) = C.compare_roots("stage1_base", ref, new, [50], [10501], "B_base")["runs"]
    assert r["ALL"], r["differences"]
    assert r["info"]["n_compared_history"] == len(UPD_B)
    d = new / "q50" / "seed10501" / "B_base"
    _edit_json(d / "final_v2.json", lambda o: o["final"].__setitem__("Gmax_full_over_dw", 0.01))
    _edit_json(d / "final_v2.json",
               lambda o: o["development"].__setitem__("stage1_rel_err_abs", 0.3))
    (r,) = C.compare_roots("stage1_base", ref, new, [50], [10501], "B_base")["runs"]
    bad = [k for k, v in r["fields"].items() if not v]
    assert bad == ["end_of_B.final_scalars", "end_of_B.development_scalars", "metric_values",
                   "dev_tier_values"]
    assert r["differences"]["metric_values"]["path"] == "metric_values.gmax_final"
    assert r["differences"]["dev_tier_values"]["path"] == "dev_tier_values.S1.stage1_rel_err_abs"


def test_stage2_base_ignores_phase_local_cadence_but_not_state(tmp_path):
    ref, new = roots(tmp_path, "phase_A_continue", arm="A_base")
    out = C.compare_roots("stage2_base", ref, new, [50], [10501], "A_base")
    (r,) = out["runs"]
    assert r["ALL"], r["differences"]            # local, cadence, curriculum differ by construction
    assert r["info"]["state_end_A.pt:snapshot_refreshes"] == {"ref": 82, "new": 84, "equal": False}
    assert r["info"]["n_compared_verifier_calls"] == 2       # 1300 and 1600 are common; 1400 is not
    assert "train_history:curriculum" in r["info"]
    d = new / "q50" / "seed10501" / "A_base"
    _edit_json(d / "train_history.json",
               lambda o: o["history"][1].__setitem__("kl_epochs", [9.0, 9.0]))
    (r,) = C.compare_roots("stage2_base", ref, new, [50], [10501], "A_base")["runs"]
    assert r["first_difference"]["field"] == "train_history:history"
    assert r["first_difference"]["path"] == "history.1220.kl_epochs[0]"


# =========================================================================== end to end
# The launcher's own configs, run by the launcher's own command (run_job / run_pool), with the
# budgets cut to a few updates. Before this section no test reached execute() with a launcher
# config: validate_config and Run() both accept configs that execute() then refuses (stage-2
# arms from a mid-phase parents_A state, run/run_v2_stagewise.py "parent phases_done").
SNAPSHOT_EVERY = 20           # locked protocol: the u1200 state sits on a snapshot refresh
TINY_CAP, TINY_MID = 40, 20   # tiny analogue of the parents_A run: cap 1600, full state at 1200
N_BASE, N_ARM = 20, 10        # updates of A_base (= 40 - 20, exact analogue) and of every other arm
QUIET = {"warmup": 100, "stability_every": 100}     # no in-phase verifier calls (speed only)
PPO_EPOCHS = 10
E2E_ARMS = STAGE1_ARMS + STAGE2_ARMS


def _phase_of(mode: str) -> str:
    return {"phase_B": "B", "phase_P": "P"}.get(mode, "A")


def _scale_windows(wins: List[Dict[str, Any]], old_cap: int, new_cap: int) -> List[Dict[str, Any]]:
    """The lr_decay windows with their local boundaries scaled from ``old_cap`` to ``new_cap``."""
    out: List[Dict[str, Any]] = []
    prev_last = 0
    for w in sorted(wins, key=lambda x: x["local_first"]):
        last = new_cap if w["local_last"] == old_cap else max(
            prev_last + 2, round(w["local_last"] * new_cap / old_cap))
        out.append({**w, "local_first": prev_last + 1, "local_last": last})
        prev_last = last
    return out


def _tiny(cfg: Dict[str, Any], n: int, parent: Path) -> Dict[str, Any]:
    """The launcher's config with ``n`` updates of its phase and a tiny parent; nothing else."""
    c = copy.deepcopy(cfg)
    phase = _phase_of(c["mode"])
    bo = c["budget_overrides"]
    caps = dict(c["record"]["protocol"]["phase_caps"])
    caps.update(bo.get("phase_caps", {}))
    old = caps[phase]
    caps[phase] = n
    bo["phase_caps"] = caps
    bo.update(QUIET)
    vt = bo.get("verifier_timeout")
    bo["verifier_timeout"] = {k: 100 for k in vt} if isinstance(vt, dict) else 100
    c["lr_decay"] = _scale_windows(c["lr_decay"], old, n)
    if c.get("conc_anneal"):
        c["conc_anneal"]["local_last"] = n
    c["parent_checkpoint"] = str(parent)
    c["parent_sha256"] = L.sha256_file(str(parent))
    assert L.lr_window_problems(c) == []
    return c


def _launch(jobs: List[Any], record_path: Path, workers: int = 6) -> Dict[str, Dict[str, Any]]:
    """Write every run_config.json and run the jobs through the launcher's pool."""
    for cfg, out in jobs:
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / "run_config.json", "w") as f:
            json.dump(cfg, f, indent=1)
    record: Dict[str, Any] = {"runs": []}
    L.run_pool(jobs, workers, record, record_path)
    return {r["out"]: r for r in record["runs"]}


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """Tiny parents_A run, then all 19 stage-1 / stage-2 arms from it (each arm launched once)."""
    root = tmp_path_factory.mktemp("e2e")
    pj = L.build_parents_a("A_parent", 50, 10501, root)
    cfg = copy.deepcopy(pj.cfg)
    proto = cfg["record"]["protocol"]
    assert int(proto["snapshot_every"]) == SNAPSHOT_EVERY
    cfg["budget_overrides"]["phase_caps"] = {**proto["phase_caps"], "A": TINY_CAP}
    cfg["budget_overrides"].update(QUIET, verifier_timeout=100)
    cfg["lr_decay"] = [dict(cfg["lr_decay"][0], local_first=TINY_MID + 1, local_last=TINY_CAP)]
    cfg["full_state_at"] = [TINY_MID]
    res = _launch([L.Job(cfg, pj.out_dir)], root / "parents.json", workers=1)
    parent_dir = Path(pj.out_dir)
    assert res[pj.out_dir]["returncode"] == 0, (parent_dir / "run.log").read_text()[-1500:]
    mid, end = parent_dir / "state_u00020.pt", parent_dir / "state_end_A.pt"
    jobs, by_arm = [], {}
    for arm in E2E_ARMS:
        stage1 = arm in STAGE1_ARMS
        job = (L.build_stage1(arm, 50, 10501, root, False) if stage1 else
               L.build_stage2(arm, 50, 10501, root, False))
        mid_phase = (not stage1) and L.STAGE2_ARMS[arm].get("kind", "continue") == "continue"
        n = N_BASE if arm == "A_base" else N_ARM
        c = _tiny(job.cfg, n, mid if mid_phase else end)
        by_arm[arm] = L.Job(c, job.out_dir)
        jobs.append(by_arm[arm])
    results = _launch(jobs, root / "arms.json")
    return {"root": root, "parent": parent_dir, "mid": mid, "end": end, "jobs": by_arm,
            "results": results}


def _load_state(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def test_parents_a_mid_phase_state_is_what_stage2_continues_from(e2e):
    """state_u<k> of a phase_A run: phase A in progress (phases_done empty), on a snapshot."""
    s = _load_state(e2e["mid"])
    assert s["phase_done"] == "A" and s["counters"]["global_u"] == TINY_MID
    assert s["phases_done"] == []           # appended only when run_phase('A') ends
    assert TINY_MID % SNAPSHOT_EVERY == 0   # else the continuation's entry refresh changes the run
    end = _load_state(e2e["end"])
    assert end["phases_done"] == ["A"] and end["counters"]["global_u"] == TINY_CAP


@pytest.mark.parametrize("arm", E2E_ARMS)
def test_launcher_config_runs_and_the_run_records_its_settings(e2e, arm):
    cfg, out = e2e["jobs"][arm]
    d = Path(out)
    log = (d / "run.log").read_text()[-1200:] if (d / "run.log").exists() else "no run.log"
    assert e2e["results"][out]["returncode"] == 0, f"{arm}: {log}"
    status = json.load(open(d / "status.json"))
    assert status["state"] == "done" and status["exit_code"] == 0
    man = json.load(open(d / "manifest.json"))
    assert man["ppo_overrides"] == cfg["ppo_overrides"]
    assert man["target_kl"] == cfg["ppo_overrides"]["target_kl"]
    assert man["continuation_value_mode"] == cfg["continuation_value_mode"]
    assert man["conc_anneal"] == cfg.get("conc_anneal")
    assert man["parent_sha256"] == cfg["parent_sha256"] and man["mode"] == cfg["mode"]
    assert (man["budget_overrides"]["episodes_per_update"]
            == cfg["budget_overrides"]["episodes_per_update"])
    # the settings act on the run: LR per local update, rows per update, scale, epochs
    hist = json.load(open(d / "train_history.json"))["history"]
    phase = _phase_of(cfg["mode"])
    n = cfg["budget_overrides"]["phase_caps"][phase]
    assert [h["local"] for h in hist] == list(range(1, n + 1))
    for h in hist:
        w = next(w for w in cfg["lr_decay"] if w["local_first"] <= h["local"] <= w["local_last"])
        want = w["start_lr"] + (w["end_lr"] - w["start_lr"]) * (h["local"] - w["local_first"]) \
            / (w["local_last"] - w["local_first"])
        assert abs(h["actor_lr"] - want) <= 1e-15, (arm, h["local"], h["actor_lr"], want)
        assert h["n_episodes"] == cfg["budget_overrides"]["episodes_per_update"]
    upd = pd.read_csv(d / "v2_updates.csv")
    if phase != "P":
        assert upd["n_epochs_run"].between(1, PPO_EPOCHS).all()
    if cfg.get("conc_anneal"):
        ca = cfg["conc_anneal"]
        assert upd["conc_scale"].iloc[0] == pytest.approx(ca["scale_first"])
        assert upd["conc_scale"].iloc[-1] == pytest.approx(ca["scale_last"])
        assert upd["conc_scale"].is_monotonic_increasing
    if phase == "P":
        chk = json.load(open(d / "phaseP_checks.json"))
        assert all(chk.values()), chk
    assert (d / f"state_end_{phase}.pt").exists()


def test_a_base_ends_in_the_parents_end_of_phase_a_state(e2e):
    """A_base from the mid-phase state (launcher configs) == the uninterrupted parents_A run."""
    cfg, out = e2e["jobs"]["A_base"]
    assert e2e["results"][out]["returncode"] == 0
    ref, new = _load_state(e2e["end"]), _load_state(Path(out) / "state_end_A.pt")
    for k in C.AGENT_KEYS:
        assert C.first_diff(ref["agent"].get(k), new["agent"].get(k), k) is None, k
    assert C.first_diff(ref["rng"], new["rng"], "rng") is None
    assert C.first_diff(ref["torch_generator_state"], new["torch_generator_state"]) is None
    for k in ("global_u", "total_episodes", "total_transitions", "next_snapshot_refresh_update"):
        assert ref["counters"][k] == new["counters"][k], k
    # the one expected difference: the continued phase refreshes the snapshot at its entry
    assert ref["agent"]["snapshot_refreshes"] + 1 == new["agent"]["snapshot_refreshes"]
    assert new["phases_done"] == ["A"]
