"""Tests of ``tools/t3/t3r3_launch.py``: the arms, the seed guards, the config checks against the lock, the plan, the infrastructure
ledger."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

import launch_ms_r1 as L1  # noqa: E402
import t3_configs as tc  # noqa: E402
import t3c_launch as cl  # noqa: E402
import t3r3_launch as lt  # noqa: E402

RUN_FIELDS = {"record.output_dir", "record.run", "run"}


@pytest.fixture(scope="module")
def setup():
    params = tc.load_bw_params()
    lock = cl.check_lock(ROOT / "protocols", use_record=True)
    jobs = lt.build_jobs(Path("/tmp/t3r3_launch_test_root"), params)
    return params, lock, jobs


def test_the_wave_has_122_runs_with_the_planned_composition(setup):
    _, _, jobs = setup
    assert len(jobs) == 122
    by_arm = {a: [j for j in jobs if lt.arm_of_dir(j.out_dir) == a] for a in lt.ARMS}
    assert {a: len(v) for a, v in by_arm.items()} == {"bw": 40, "bwA": 41, "bwAB": 41}
    for a, v in by_arm.items():
        dev = [(j.cfg["q"], j.cfg["seed"]) for j in v if "/dev/" in j.out_dir]
        assert sorted(dev) == [(q, s) for q in (50, 60) for s in range(70501, 70521)]
    diag = [(lt.arm_of_dir(j.out_dir), j.cfg["q"], j.cfg["seed"]) for j in jobs if "/diagnostic/" in j.out_dir]
    assert sorted(diag) == [("bwA", 60, 60518), ("bwAB", 60, 60518)]
    assert [lt.arm_of_dir(j.out_dir) for j in jobs[:2]] == ["bwA", "bwAB"]       # the diagnostic runs start first
    names = [j.cfg["run"] for j in jobs]
    assert len(set(names)) == 122 and "t3r3_q50_s70501_bw" in names and "t3r3_q60_s60518_bwAB" in names


def test_every_planned_config_validates_and_matches_the_lock(setup):
    _, lock, jobs = setup
    rows = lt.validate_jobs(jobs, lock)
    assert all(r["ok"] for r in rows), [r for r in rows if not r["ok"]][:2]


def test_a_candidate_config_differs_from_the_control_only_in_the_scheme_keys_and_run_fields(setup):
    _, _, jobs = setup
    ctl = {(j.cfg["q"], j.cfg["seed"]): j.cfg for j in jobs if lt.arm_of_dir(j.out_dir) == "bw"}
    for j in jobs:
        arm = lt.arm_of_dir(j.out_dir)
        if arm == "bw" or "/diagnostic/" in j.out_dir:
            continue
        diff = set(tc.config_diff(ctl[(j.cfg["q"], j.cfg["seed"])], j.cfg))
        own = {"stage3_retry.delta_gt", "stage3_retry.max_retries"} | ({"terminal_clamp_likelihood"} if arm == "bwAB" else set())
        assert diff == RUN_FIELDS | own, (arm, diff ^ (RUN_FIELDS | own))
    for j in jobs:
        if lt.arm_of_dir(j.out_dir) == "bw":
            assert "stage3_retry" not in j.cfg and "terminal_clamp_likelihood" not in j.cfg and j.cfg["pilot"] == "t3_r3"


@pytest.mark.parametrize("q,seed", [(50, 60501), (60, 60520), (50, 50501), (60, 70521), (50, 70500), (60, 80501), (50, 40501),
                                    (60, 10501), (50, 30501), (50, 60518), (70, 70501)])
def test_other_seeds_are_refused(setup, q, seed):
    params, _, _ = setup
    with pytest.raises(ValueError):
        lt.build_arm_config(tc.load_protocol(), params, "bw", q, seed, "/tmp/x")


def test_the_diagnostic_run_exists_only_at_q60_seed_60518(setup):
    params, _, _ = setup
    cfg = lt.build_arm_config(tc.load_protocol(), params, "bwA", 60, 60518, "/tmp/x")
    assert cfg["seed"] == 60518 and lt.kind_of(60, 60518) == "diagnostic"


def test_config_problems_reports_departures(setup):
    params, lock, jobs = setup
    j = next(j for j in jobs if lt.arm_of_dir(j.out_dir) == "bwA" and j.cfg["q"] == 50 and j.cfg["seed"] == 70501)
    assert lt.config_problems(j.cfg, lock, "bwA") == []
    assert lt.config_problems(j.cfg, lock, "bw")                                  # the keys of bwA are not those of bw
    assert lt.config_problems(j.cfg, lock, "bwAB")
    for mut in (lambda c: c["pipeline"]["budgets"].__setitem__("3", 1599), lambda c: c.__setitem__("pilot", "t3_r2"),
                lambda c: c["stage3_retry"].__setitem__("delta_gt", 0.006), lambda c: c.__setitem__("terminal_clamp_likelihood", "censored"),
                lambda c: c["record"]["ppo"].__setitem__("lr", 1e-3), lambda c: c.__setitem__("seed", 70502)):
        c = copy.deepcopy(j.cfg)
        mut(c)
        assert lt.config_problems(c, lock, "bwA"), mut


def test_the_plan_digests_attempt1_is_the_bw_digest_and_attempt2_is_the_derived_one(setup):
    _, _, jobs = setup
    pick = [j for j in jobs if j.cfg["q"] == 50 and j.cfg["seed"] == 70501]
    assert {lt.arm_of_dir(j.out_dir) for j in pick} == set(lt.ARMS)
    rows = {lt.arm_of_dir(r["out_dir"]): r for r in lt.plan_rows(pick, Path("/tmp/t3r3_launch_test_root"))}
    d1 = {a: rows[a]["init_state_sha256"] for a in rows}
    assert len(set(d1.values())) == 1 and len(d1["bw"]) == 64                       # the scheme does not change the initial network
    assert rows["bw"]["attempt2_init_state_sha256"] == ""
    assert rows["bwA"]["attempt2_init_state_sha256"] == rows["bwAB"]["attempt2_init_state_sha256"] != d1["bw"]
    assert len({rows[a]["run_config_sha256"] for a in rows}) == 3


def test_the_tree_guard_ignores_results_only(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    assert lt.tree_problems(tmp_path) == []
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "a.txt").write_text("x")
    assert lt.tree_problems(tmp_path) == []
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "b.py").write_text("x")
    assert lt.tree_problems(tmp_path)


def test_infrastructure_failures_are_repeated_once_and_recorded(tmp_path):
    cfgs = []
    jobs = []
    for i, arm in enumerate(("bwA", "bw")):
        out = tmp_path / "dev" / "q50" / f"seed7050{i + 1}" / arm
        out.mkdir(parents=True)
        cfg = {"q": 50, "seed": 70501 + i, "run": f"r{i}"}
        jobs.append(L1.Job(cfg, str(out)))
    results = [{"out": jobs[0].out_dir, "returncode": -9}, {"out": jobs[1].out_dir, "returncode": 1}]
    ledger = {"events": [], "stopped_without_verdict": []}
    seen = []

    def runner(job):
        seen.append(job.out_dir)
        return {"out": job.out_dir, "returncode": 0}
    reps = lt.handle_infra_failures(jobs, results, tmp_path, runner, ledger)
    assert seen == [jobs[0].out_dir] and len(reps) == 1 and reps[0]["returncode"] == 0       # only the signal is repeated
    assert [e["reason"] for e in ledger["events"]] == ["negative_returncode"] and ledger["stopped_without_verdict"] == []
    assert (tmp_path / "crashed" / "dev" / "q50" / "seed70501" / "bwA_attempt1").is_dir()
    # a second infrastructure failure stops the run without a verdict
    ledger2 = {"events": [], "stopped_without_verdict": []}
    jobs[0].cfg["seed"] = 70501

    def runner2(job):
        return {"out": job.out_dir, "returncode": -9}
    Path(jobs[0].out_dir).mkdir(parents=True, exist_ok=True)
    lt.handle_infra_failures(jobs[:1], [{"out": jobs[0].out_dir, "returncode": -9}], tmp_path, runner2, ledger2)
    assert ledger2["stopped_without_verdict"] == ["r0"] and [e["attempt"] for e in ledger2["events"]] == [1, 2]
