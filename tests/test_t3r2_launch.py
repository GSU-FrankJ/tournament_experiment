"""T3-R2: tests of the confirmation launcher ``tools/t3/t3c_launch.py`` (seed and arm guards, 40 jobs, the identity of the
configs with T3-R1's ``bw`` configs outside the D4 fields, the lock checks, the plan with the init digests, the infrastructure
ledger, the git guards) and of the evaluation tool ``tools/t3/t3c_eval.py`` on a reduced run.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_t3r2_launch.py -p no:cacheprovider -q
"""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "t3", ROOT / "tools" / "ms", ROOT / "tests"):
    sys.path.insert(0, str(_p))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import t3_helpers as H  # noqa: E402
import t3_configs as tc  # noqa: E402
import t3_eval as te  # noqa: E402
import t3c_eval as ce  # noqa: E402
import t3c_launch as cl  # noqa: E402
import t3r2_make_lock as ml  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402

PROTO, PARAMS = H.PROTO, H.BW_PARAMS


@pytest.fixture(scope="module")
def lock_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("lock")
    assert ml.main(["--out", str(d)]) == 0
    return d


@pytest.fixture(scope="module")
def lock(lock_dir):
    return json.loads((lock_dir / "v2_T3_locked_v1_0.json").read_text())


def record_for(lock_dir: Path) -> dict:
    return {"protocol": cl.LOCK_REL, "protocol_version": "T3-1.0", "lock_commit": "x" * 40,
            "protocol_sha256": cl.sha256_file(lock_dir / "v2_T3_locked_v1_0.json"),
            "protocol_md_sha256": cl.sha256_file(lock_dir / "v2_T3_locked_v1_0.md"), "date": "d",
            "rule": "nothing in protocols/ changes after this lock without a new version number and a stated reason"}


# ====================================================================== jobs and guards
def test_forty_jobs_with_unique_names_directories_and_the_confirmation_fields(tmp_path):
    jobs = cl.build_jobs(cl.QS, cl.SEEDS, tmp_path, PARAMS)
    assert len(jobs) == 40
    assert len({j.cfg["run"] for j in jobs}) == 40 and len({j.out_dir for j in jobs}) == 40
    for j in jobs:
        c = j.cfg
        assert c["arm"] == "bw" and c["pilot"] == "t3_r2" and c["run"] == f"t3r2_q{c['q']}_s{c['seed']}_bw"
        assert c["record"]["run"] == c["run"] and c["record"]["seed"] == c["seed"] and c["record"]["output_dir"] == j.out_dir
        assert j.out_dir == str(tmp_path / f"q{c['q']}" / f"seed{c['seed']}" / "bw")
        assert 60501 <= c["seed"] <= 60520 and c["init_digest"] is True
    assert {(j.cfg["q"], j.cfg["seed"]) for j in jobs} == {(q, s) for q in (50, 60) for s in range(60501, 60521)}


@pytest.mark.parametrize("qs,seeds", [((50,), tuple(range(60501, 60521))), ((50, 60), tuple(range(60501, 60520))),
                                      ((50, 60), tuple(range(60502, 60522))), ((50, 60), tuple(range(50501, 50521))),
                                      ((50, 60), tuple(range(40501, 40521))), ((50, 60), tuple(range(30501, 30521))),
                                      ((50, 55), tuple(range(60501, 60521))), ((50, 60), (60501, 60501) + tuple(range(60503, 60521)))])
def test_the_seed_guard_accepts_only_the_whole_block_at_both_q(tmp_path, qs, seeds):
    with pytest.raises(ValueError):
        cl.build_jobs(qs, seeds, tmp_path, PARAMS)


def test_the_arm_guard_and_the_run_checks_refuse_a_foreign_config(lock):
    good = cl.build_confirmation_config(PROTO, 50, 60501, "/x", PARAMS)
    assert cl.config_problems(good, lock) == []
    for field, bad in (("arm", "jt"),):
        c = copy.deepcopy(good)
        c[field] = bad
        assert cl.config_problems(c, lock)
    c = copy.deepcopy(good)
    c["seed"] = c["record"]["seed"] = 50501
    c["run"] = c["record"]["run"] = "t3r2_q50_s50501_bw"
    assert any("outside the confirmation block" in p for p in cl.config_problems(c, lock))
    c = copy.deepcopy(good)
    c["pilot"] = "t3_r1"
    assert any("pilot" in p for p in cl.config_problems(c, lock))


def test_each_config_differs_from_t3r1s_bw_config_of_the_same_q_only_in_the_d4_fields(tmp_path):
    allowed = {"seed", "run", "record.run", "record.seed", "record.output_dir", "pilot"}
    assert set(cl.ALLOWED_DIFF) == allowed
    for q in (50, 60):
        ref = tc.build_bw_config(PROTO, q, 50501, "/ref", PARAMS)
        recorded = json.loads((ROOT / "results" / "t3_r1" / "runs" / f"q{q}" / "seed50501" / "bw" / "run_config.json").read_text())
        for s in (60501, 60510, 60520):
            cfg = cl.build_confirmation_config(PROTO, q, s, str(tmp_path / f"q{q}"), PARAMS)
            assert set(tc.config_diff(cfg, ref)) == allowed
            assert set(tc.config_diff(cfg, recorded)) <= allowed
            same_seed = tc.build_bw_config(PROTO, q, s, str(tmp_path / f"q{q}"), PARAMS)
            assert set(tc.config_diff(cfg, same_seed)) == {"run", "record.run", "pilot"}


def test_a_config_that_differs_from_the_lock_outside_the_d4_fields_is_refused(lock):
    good = cl.build_confirmation_config(PROTO, 60, 60507, "/x", PARAMS)
    assert cl.config_problems(good, lock) == []
    edits = [("pipeline.budgets.2", lambda c: c["pipeline"]["budgets"].__setitem__("2", 1599)),
             ("lr window", lambda c: c["pipeline"]["lr_windows"]["3"][0].__setitem__("first", 1200)),
             ("episodes", lambda c: c["record"]["protocol"].__setitem__("episodes_per_update", 256)),
             ("ppo lr", lambda c: c["record"]["ppo"].__setitem__("lr", 1e-3)),
             ("game k", lambda c: c["record"]["game"].__setitem__("k", 0.001)),
             ("rule enabled", lambda c: c["rule"].__setitem__("enabled", True)),
             ("start scheme", lambda c: c.__setitem__("start_weights", {"scheme": "stratified_priority"})),
             ("clamp", lambda c: c.__setitem__("clamp_likelihood", "censored")),
             ("init_digest", lambda c: c.__setitem__("init_digest", False)),
             ("protocol_ref", lambda c: c["protocol_ref"].__setitem__("sha256", "0" * 64)),
             ("an extra key", lambda c: c.__setitem__("actor_variant", "relu"))]
    for name, edit in edits:
        c = copy.deepcopy(good)
        edit(c)
        assert cl.config_problems(c, lock), name
    # the D4 fields themselves are free
    c = copy.deepcopy(good)
    c["record"]["output_dir"] = "/somewhere/else"
    assert cl.config_problems(c, lock) == []


# ====================================================================== the lock
def test_the_lock_check_needs_the_record_and_matching_files_and_tools(tmp_path, lock_dir):
    lockfile = tmp_path / "LOCK"
    lockfile.write_text(json.dumps({"protocol": "protocols/v2_T2_locked_v2_0.json"}, indent=1) + "\n")
    with pytest.raises(ValueError, match="no record"):
        cl.check_lock(lock_dir, True, lockfile)                                   # a missing LOCK record
    lockfile.write_text(lockfile.read_text() + json.dumps(record_for(lock_dir), indent=1) + "\n")
    assert cl.t3_record(lockfile)["protocol_version"] == "T3-1.0"
    assert cl.check_lock(lock_dir, True, lockfile)["schema"] == "t3_locked_protocol/1"
    # a modified lock file (JSON or MD) no longer matches the record
    bad = tmp_path / "bad"
    shutil.copytree(lock_dir, bad)
    (bad / "v2_T3_locked_v1_0.md").write_text((bad / "v2_T3_locked_v1_0.md").read_text() + "x")
    with pytest.raises(ValueError, match="SHA-256 of the LOCK record"):
        cl.check_lock(bad, True, lockfile)
    shutil.copytree(lock_dir, tmp_path / "bad2")
    js = json.loads((tmp_path / "bad2" / "v2_T3_locked_v1_0.json").read_text())
    js["confirmation"]["rule"]["min_passes_per_q"] = 16
    (tmp_path / "bad2" / "v2_T3_locked_v1_0.json").write_text(json.dumps(js, indent=1) + "\n")
    with pytest.raises(ValueError, match="SHA-256 of the LOCK record"):
        cl.check_lock(tmp_path / "bad2", True, lockfile)
    # the drafts need no record
    assert cl.check_lock(lock_dir, False, tmp_path / "does_not_exist")["protocol_file"] == tc.PROTOCOL_REL


def test_a_modified_listed_tool_or_input_file_is_refused(tmp_path, lock_dir, lock):
    repo = tmp_path / "repo"
    files = list(lock["tool_sha256"]) + [lock["protocol_file"], lock["params_file"], lock["criteria_file"]]
    for f in files:
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, repo / f)
    assert cl.check_lock(lock_dir, False, repo=repo)
    for victim in ("tools/t3/t3_eval.py", "tools/ms/ms_configs.py", "tools/t3/t3c_launch.py", lock["params_file"],
                   lock["criteria_file"], lock["protocol_file"]):
        (repo / victim).write_text((repo / victim).read_text() + "\n# changed\n")
        with pytest.raises(ValueError):
            cl.check_lock(lock_dir, False, repo=repo)
        shutil.copy(ROOT / victim, repo / victim)
    (repo / "tools/t3/t3_metrics.py").unlink()
    with pytest.raises(ValueError, match="differ"):
        cl.check_lock(lock_dir, False, repo=repo)


def test_the_cli_refuses_without_a_lock_record_and_with_a_modified_params_file(tmp_path, monkeypatch, lock_dir):
    monkeypatch.setattr(cl, "LOCK_FILE", tmp_path / "LOCK_without_record")
    (tmp_path / "LOCK_without_record").write_text("")
    with pytest.raises(SystemExit):
        cl.main(["--lock-dir", str(lock_dir), "--root", str(tmp_path / "r"), "--dry-run-not-a-flag"])       # argparse error
    with pytest.raises(SystemExit):
        cl.main(["--lock-dir", str(lock_dir), "--root", str(tmp_path / "r")])                            # no record: refused
    assert not (tmp_path / "r").exists()
    bad = tmp_path / "params.json"
    d = json.load(open(tc.PARAMS_PATH))
    d["K"] = 26
    json.dump(d, open(bad, "w"), indent=1)
    with pytest.raises(SystemExit):
        cl.main(["--params", str(bad), "--lock-dir", str(lock_dir), "--plan-csv", str(tmp_path / "p.csv")])
    for seeds in (["50501"], ["40501"], ["60501"]):
        with pytest.raises(SystemExit):
            cl.main(["--seeds", *seeds, "--lock-dir", str(lock_dir), "--plan-csv", str(tmp_path / "p2.csv")])


# ====================================================================== the plan with the init digests
def test_plan_csv_emits_the_run_name_directory_config_hash_and_init_digest(tmp_path, lock_dir):
    out = tmp_path / "plan.csv"
    assert cl.main(["--plan-csv", str(out), "--lock-dir", str(lock_dir), "--root", str(tmp_path / "conf")]) == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 40 and list(rows[0]) == cl.PLAN_FIELDS
    assert len({r["run"] for r in rows}) == 40 and len({r["init_state_sha256"] for r in rows}) == 40
    assert all(len(r["init_state_sha256"]) == 64 and len(r["run_config_sha256"]) == 64 for r in rows)
    assert rows[0]["run"] == "t3r2_q50_s60501_bw" and rows[-1]["run"] == "t3r2_q60_s60520_bw"
    assert rows[0]["out_dir"] == str(tmp_path / "conf" / "q50" / "seed60501" / "bw")
    # the digest is the one the run itself records at construction
    r = rows[7]
    cfg = cl.build_confirmation_config(PROTO, int(r["q"]), int(r["seed"]), r["out_dir"], PARAMS)
    assert rms.MSRun(cfg, str(tmp_path / "scratch")).init_state_sha256 == r["init_state_sha256"]
    # the same seed gives the same digest at the same q, whatever the root; the relative hash does not depend on the worktree
    a = cl.plan_with_digests(cl.build_jobs(cl.QS, cl.SEEDS, tmp_path / "a" / "confirmation", PARAMS)[:2], tmp_path / "a" / "confirmation")
    b = cl.plan_with_digests(cl.build_jobs(cl.QS, cl.SEEDS, tmp_path / "b" / "confirmation", PARAMS)[:2], tmp_path / "b" / "confirmation")
    assert [x["init_state_sha256"] for x in a] == [x["init_state_sha256"] for x in b]
    assert [x["run_config_sha256_relpath"] for x in a] == [x["run_config_sha256_relpath"] for x in b]
    assert [x["run_config_sha256"] for x in a] != [x["run_config_sha256"] for x in b]
    with pytest.raises(FileExistsError):
        cl.write_plan(out, rows)


def test_the_command_of_a_job_is_the_reused_runner_command(tmp_path):
    job = cl.build_jobs(cl.QS, cl.SEEDS, tmp_path, PARAMS)[0]
    cmd = cl.job_command(job.cfg, job.out_dir)
    assert cmd[-4:] == ["--config", os.path.join(job.out_dir, "run_config.json"), "--out-dir", job.out_dir]
    assert cmd[3].endswith("run/run_ms_stagewise.py")
    import t3_launch as tl
    assert cl.job_command is tl.job_command and cl.run_pool is tl.run_pool and cl.plan_rows is tl.plan_rows


# ====================================================================== infrastructure failures
def _job(tmp_path, q, s):
    return cl.build_jobs(cl.QS, cl.SEEDS, tmp_path / "conf", PARAMS)[(0 if q == 50 else 20) + (s - 60501)]


def _write_status(out_dir, **kw):
    os.makedirs(out_dir, exist_ok=True)
    (Path(out_dir) / "status.json").write_text(json.dumps(kw))


def test_classification_of_a_finished_attempt(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    assert cl.classify_attempt({"returncode": 0}, d) is None
    assert cl.classify_attempt({"returncode": 1}, d) is None                         # a pipeline exception: a failed run
    assert cl.classify_attempt({"returncode": 5}, d) is None                         # a global-RNG violation: a failed run
    assert cl.classify_attempt({"returncode": -9}, d)[0] == "negative_returncode"
    assert cl.classify_attempt({"returncode": -11}, d)[0] == "negative_returncode"
    _write_status(d, state="failed", traceback="Traceback ...\nOSError: [Errno 28] No space left on device\n")
    assert cl.classify_attempt({"returncode": 1}, d)[0] == "enospc"
    _write_status(d, state="failed", traceback="ValueError: x")
    (d / "run.log").write_text("fine\nMemoryError\n")
    assert cl.classify_attempt({"returncode": 1}, d)[0] == "out_of_memory"
    assert cl.classify_attempt({"returncode": None, "error": "OSError: [Errno 28] No space left on device"}, tmp_path / "none")[0] == "enospc"


def test_an_infrastructure_failure_is_moved_aside_and_repeated_once_a_second_one_stops_the_confirmation(tmp_path):
    repo = tmp_path / "repo"
    jobs = [_job(tmp_path, 50, 60501), _job(tmp_path, 50, 60502), _job(tmp_path, 60, 60501), _job(tmp_path, 60, 60502)]
    for j in jobs:
        os.makedirs(j.out_dir)
        (Path(j.out_dir) / "run_config.json").write_text("{}")
        (Path(j.out_dir) / "run.log").write_text("log of the first attempt\n")
    _write_status(jobs[3].out_dir, state="failed", traceback="OSError: [Errno 28] No space left on device")
    results = [{"out": jobs[0].out_dir, "returncode": -9}, {"out": jobs[1].out_dir, "returncode": 1},
               {"out": jobs[2].out_dir, "returncode": -9}, {"out": jobs[3].out_dir, "returncode": 1}]
    calls = []

    def runner(job):
        calls.append(job.cfg["run"])
        os.makedirs(job.out_dir, exist_ok=True)
        if job.cfg["run"] == "t3r2_q60_s60501_bw":
            return {"out": job.out_dir, "returncode": -9}          # the repeat is killed again
        return {"out": job.out_dir, "returncode": 0}
    ledger = cl.new_ledger()
    reps = cl.handle_infra_failures(jobs, results, tmp_path / "conf", runner, ledger, repo=repo)
    assert calls == ["t3r2_q50_s60501_bw", "t3r2_q60_s60501_bw", "t3r2_q60_s60502_bw"]       # job 1 (exit 1) is never repeated
    assert [e["reason"] for e in ledger["events"]] == ["negative_returncode", "negative_returncode", "negative_returncode", "enospc"]
    assert [e["attempt"] for e in ledger["events"]] == [1, 1, 2, 1]
    assert ledger["stopped_without_verdict"] == ["t3r2_q60_s60501_bw"]
    assert len(reps) == 3
    assert (repo / "results" / "t3_r2" / "crashed" / "q50" / "seed60501" / "bw_attempt1" / "run.log").is_file()
    assert (repo / "results" / "t3_r2" / "crashed" / "q60" / "seed60501" / "bw_attempt2").is_dir()
    assert (Path(jobs[1].out_dir) / "run.log").is_file()                                       # the ordinary failure stays where it is
    assert (Path(jobs[0].out_dir) / "run_config.json").is_file()                                # the repeat's directory was recreated


# ====================================================================== the git guards
def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def test_tree_problems_allow_only_new_t3c_and_t3r2_tool_files(tmp_path):
    repo = tmp_path / "g"
    for rel in ("agents/a.py", "envs/e.py", "utils/u.py", "run/r.py", "tools/ms/m.py", "tools/v2/v.py", "tools/t3/t3_eval.py",
                "protocols/LOCK", "reports/x.md"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    assert cl.tree_problems(repo, base) == []
    for rel in ("tools/t3/t3c_new.py", "tools/t3/t3r2_new.py"):
        (repo / rel).write_text("y\n")
    (repo / "protocols/LOCK").write_text("x\nrecord\n")                    # protocols/ and reports/ are not code paths
    (repo / "reports/y.md").write_text("y\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "ok")
    assert cl.tree_problems(repo, base) == []
    for rel, how in (("tools/t3/t3_eval.py", "modify"), ("tools/t3/other.py", "add"), ("run/r.py", "modify"), ("utils/new.py", "add")):
        (repo / rel).write_text("changed\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", f"{how} {rel}")
        assert any(rel in p for p in cl.tree_problems(repo, base)), rel
    (repo / "reports/dirty.md").write_text("z\n")
    assert any("not clean" in p for p in cl.tree_problems(repo, base))
    _git(repo, "rm", "-q", "-f", "agents/a.py")
    _git(repo, "commit", "-q", "-m", "delete")
    assert any("agents/a.py" in p for p in cl.tree_problems(repo, base))        # a deleted code file


# ====================================================================== t3c_eval on a reduced run
def test_t3c_eval_reproduces_a_reduced_runs_freeze_arrays_bit_for_bit(tmp_path):
    d, cfg = H.reduced_bw_run()
    root = tmp_path / "conf"
    dest = cl.run_dir(root, 50, 60501)
    shutil.copytree(d, dest)
    jobs = ce.plan_jobs(root, tmp_path / "arrays")
    assert len(jobs) == 40 and jobs[0]["run"] == "t3r2_q50_s60501_bw" and jobs[0]["kind"] == "bw" and jobs[0]["T"] == 3
    row = te.evaluate_run(jobs[0])
    assert row["status"] == "done" and row["eval_error"] == "" and row["valid"] is True and row["dev_valid"] is True
    assert te.replay_mismatches(dest, "bw") == []
    with np.load(dest / "freeze_stage1_final.npz") as z:
        gmax = max(float(np.max(z[f"v_t{t}_G"])) for t in (1, 2, 3)) / 4.0
    assert row["Gmax_full_over_dw"] == gmax
    # a run that does not exist is a row without metrics
    miss = te.evaluate_run(jobs[1])
    assert miss["status"] == "missing" and "Gmax_full_over_dw" not in miss
    assert not ce.needs_rerun(miss) and not ce.needs_rerun(row) and ce.usable(row)
    assert ce.needs_rerun({**row, "valid": False}) and ce.needs_rerun({**row, "eval_error": "ValueError: x"})
    assert not ce.usable({**row, "dev_valid": False})
