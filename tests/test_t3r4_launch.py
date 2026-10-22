"""Tests of ``tools/t3/t3r4_launch.py``: the seeds, the arm, the configs against the lock, the lock and git guards, the plan, the
infrastructure ledger."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"          # the runner refuses a run whose thread environment differs from the config

import launch_ms_r1 as L1  # noqa: E402
import t3_configs as tc  # noqa: E402
import t3c_launch as cl  # noqa: E402
import t3r3_launch as r3  # noqa: E402
import t3r4_launch as lt  # noqa: E402
import t3r4_make_lock as ml  # noqa: E402

PARAMS = tc.load_bw_params()
PROTO = tc.load_protocol()


@pytest.fixture(scope="module")
def lock_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("lock")
    assert ml.main(["--out", str(d)]) == 0
    return d


@pytest.fixture(scope="module")
def lock(lock_dir):
    return json.loads((lock_dir / "v2_T3_locked_v1_1.json").read_text())


@pytest.fixture(scope="module")
def jobs():
    return lt.build_jobs(lt.QS, lt.SEEDS, Path("/tmp/t3r4_launch_test_root"), PARAMS)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True, text=True).stdout


# ---------------------------------------------------------------------------------- the wave
def test_the_wave_has_80_runs_with_the_planned_names_and_directories(jobs):
    assert len(jobs) == 80
    names = [j.cfg["run"] for j in jobs]
    assert len(set(names)) == 80 and names[0] == "t3r4_q50_s80501_bwA" and names[-1] == "t3r4_q60_s80540_bwA"
    for q in lt.QS:
        sel = [j for j in jobs if j.cfg["q"] == q]
        assert [j.cfg["seed"] for j in sel] == list(range(80501, 80541))
    for j in jobs:
        q, s = j.cfg["q"], j.cfg["seed"]
        assert j.out_dir == f"/tmp/t3r4_launch_test_root/q{q}/seed{s}/bwA" and j.cfg["pilot"] == "t3_r4" and j.cfg["stage3_retry"] == lt.RETRY
        assert j.cfg["arm"] == "bw" and "terminal_clamp_likelihood" not in j.cfg


def test_every_planned_config_validates_and_matches_the_lock(jobs, lock):
    rows = lt.validate_jobs(jobs, lock)
    assert all(r["ok"] for r in rows), [r for r in rows if not r["ok"]][:2]


def test_each_config_differs_from_the_reference_and_from_t3r3s_bwa_config_only_in_the_six_run_fields(jobs):
    for j in jobs:
        q = j.cfg["q"]
        ref = lt.reference_config(PROTO, PARAMS, q)
        assert set(tc.config_diff(ref, j.cfg)) == set(lt.ALLOWED_DIFF), (q, j.cfg["seed"])
    for q in lt.QS:                                                         # against T3-R3's bwA config of the same q
        old = r3.build_arm_config(PROTO, PARAMS, "bwA", q, 70501, "/tmp/x")
        new = lt.build_confirmation_config(PROTO, PARAMS, q, 80501, "/tmp/y")
        assert set(tc.config_diff(old, new)) == set(lt.ALLOWED_DIFF)


def test_the_reference_hash_equals_the_normalised_config_hash(jobs, lock):
    for j in jobs[:3] + jobs[-3:]:
        assert tc.config_sha256(cl.normalise(j.cfg)) == lock["reference_config_sha256"][str(j.cfg["q"])] == lt.reference_config_sha256(
            PROTO, PARAMS, j.cfg["q"])


# ---------------------------------------------------------------------------------- seeds, arm, config departures
@pytest.mark.parametrize("q,seed", [(50, 70501), (60, 70520), (50, 60501), (60, 60518), (50, 80500), (60, 80541), (50, 50500), (70, 80501),
                                    (50, 40501), (60, 10501), (50, 30501)])
def test_other_seeds_and_q_are_refused(q, seed):
    with pytest.raises(ValueError):
        lt.build_confirmation_config(PROTO, PARAMS, q, seed, "/tmp/x")


def test_a_partial_block_or_another_q_is_refused():
    lt.check_seeds([50, 60], list(range(80501, 80541)))
    for qs, seeds in (([50], list(lt.SEEDS)), ([50, 60, 70], list(lt.SEEDS)), ([50, 60], list(range(80501, 80540))),
                      ([50, 60], list(range(80500, 80540))), ([50, 60], list(lt.SEEDS) + [80541])):
        with pytest.raises(ValueError):
            lt.check_seeds(qs, seeds)


def test_config_problems_reports_departures(jobs, lock):
    cfg = jobs[0].cfg
    assert lt.config_problems(cfg, lock) == []
    muts = {
        "another arm field": lambda c: c.__setitem__("arm", "jt"),
        "no scheme key": lambda c: c.pop("stage3_retry"),
        "the censored key (bwAB)": lambda c: c.__setitem__("terminal_clamp_likelihood", "censored"),
        "delta_gt": lambda c: c["stage3_retry"].__setitem__("delta_gt", 0.006),
        "max_retries": lambda c: c["stage3_retry"].__setitem__("max_retries", 2),
        "pilot": lambda c: c.__setitem__("pilot", "t3_r3"),
        "run name": lambda c: c.__setitem__("run", "t3r4_q50_s80502_bwA"),
        "seed": lambda c: c.__setitem__("seed", 80502),
        "seed outside the block": lambda c: c.__setitem__("seed", 70501),
        "a stage-3 budget": lambda c: c["pipeline"]["budgets"].__setitem__("3", 1599),
        "the learning rate": lambda c: c["record"]["ppo"].__setitem__("lr", 1e-3)}
    for name, mut in muts.items():
        c = copy.deepcopy(cfg)
        mut(c)
        assert lt.config_problems(c, lock), name


# ---------------------------------------------------------------------------------- the lock
def _lock_file(tmp_path, lock_dir, extra_after=False, sha_js=None, sha_md=None, with_record=True):
    js, md = lock_dir / "v2_T3_locked_v1_1.json", lock_dir / "v2_T3_locked_v1_1.md"
    recs = [{"protocol": "protocols/v2_T3_locked_v1_0.json", "protocol_version": "T3-1.0"}]
    if with_record:
        recs.append({"protocol": lt.LOCK_REL, "protocol_version": "T3-1.1", "protocol_sha256": sha_js or cl.sha256_file(js),
                     "protocol_md_sha256": sha_md or cl.sha256_file(md)})
    if extra_after:
        recs.append({"protocol": "protocols/other.json"})
    path = tmp_path / "LOCK"
    path.write_text("".join(json.dumps(r, indent=1) + "\n" for r in recs))
    return path


def test_the_drafts_pass_without_a_record_and_the_lock_with_a_matching_last_record(tmp_path, lock_dir):
    assert lt.check_lock(lock_dir, use_record=False)["protocol_version"] == "T3-1.1"
    assert lt.check_lock(lock_dir, use_record=True, lock_file=_lock_file(tmp_path, lock_dir))["protocol_version"] == "T3-1.1"


def test_a_missing_record_a_non_final_record_a_hash_mismatch_and_a_missing_file_are_refused(tmp_path, lock_dir):
    with pytest.raises(ValueError, match="not locked"):
        lt.check_lock(lock_dir, use_record=True, lock_file=_lock_file(tmp_path, lock_dir, with_record=False))
    with pytest.raises(ValueError, match="not the last record"):
        lt.check_lock(lock_dir, use_record=True, lock_file=_lock_file(tmp_path, lock_dir, extra_after=True))
    with pytest.raises(ValueError, match="SHA-256 of the LOCK record"):
        lt.check_lock(lock_dir, use_record=True, lock_file=_lock_file(tmp_path, lock_dir, sha_js="0" * 64))
    with pytest.raises(ValueError, match="SHA-256 of the LOCK record"):
        lt.check_lock(lock_dir, use_record=True, lock_file=_lock_file(tmp_path, lock_dir, sha_md="0" * 64))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="missing"):
        lt.check_lock(empty, use_record=False)


def test_two_t31_records_are_refused_even_when_the_first_one_is_right(tmp_path, lock_dir):
    js, md = lock_dir / "v2_T3_locked_v1_1.json", lock_dir / "v2_T3_locked_v1_1.md"
    good = {"protocol": lt.LOCK_REL, "protocol_version": "T3-1.1", "protocol_sha256": cl.sha256_file(js), "protocol_md_sha256": cl.sha256_file(md)}
    wrong = dict(good, protocol_sha256="0" * 64)
    for recs in ([good, wrong], [wrong, good]):
        path = tmp_path / "LOCK2"
        path.write_text("".join(json.dumps(r, indent=1) + "\n" for r in recs))
        with pytest.raises(ValueError, match="exactly one"):
            lt.check_lock(lock_dir, use_record=True, lock_file=path)


def test_a_listed_file_that_differs_from_the_lock_is_refused(tmp_path, lock_dir):
    d = tmp_path / "mut"
    d.mkdir()
    js = json.loads((lock_dir / "v2_T3_locked_v1_1.json").read_text())
    for victim in ("run/ms_retry.py", "run/run_ms_stagewise.py", "tools/t3/t3r4_launch.py", "tools/t3/t3_eval.py"):
        assert victim in js["tool_sha256"], victim
        mut = copy.deepcopy(js)
        mut["tool_sha256"][victim] = "0" * 64
        (d / "v2_T3_locked_v1_1.json").write_text(json.dumps(mut, indent=1) + "\n")
        (d / "v2_T3_locked_v1_1.md").write_text("x")
        rec = _lock_file(tmp_path, d)
        with pytest.raises(ValueError, match="differ from the lock"):
            lt.check_lock(d, use_record=True, lock_file=rec)
    mut = copy.deepcopy(js)
    mut["protocol_version"] = "T3-1.0"
    (d / "v2_T3_locked_v1_1.json").write_text(json.dumps(mut, indent=1) + "\n")
    with pytest.raises(ValueError, match="protocol_version"):
        lt.check_lock(d, use_record=False)


# ---------------------------------------------------------------------------------- git guards (temporary repositories)
def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    for rel in ("protocols/LOCK", "run/a.py", "tools/t3/b.py", "tools/ms/c.py", "tools/v2/e.py", "agents/d.py", "envs/f.py", "utils/g.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD").strip()


def test_head_must_be_the_lock_record_commit(tmp_path):
    repo, _ = _repo(tmp_path)
    assert lt.head_problems(repo) == []                                            # the base commit last changed protocols/LOCK
    (repo / "tools" / "t3" / "b.py").write_text("y\n")
    _git(repo, "commit", "-q", "-am", "later")
    assert lt.head_problems(repo) and "not the LOCK-record commit" in lt.head_problems(repo)[0]
    (repo / "protocols" / "LOCK").write_text("record\n")
    _git(repo, "commit", "-q", "-am", "record")
    assert lt.head_problems(repo) == []


def test_a_dirty_tree_outside_results_is_refused_and_one_inside_is_not(tmp_path):
    repo, _ = _repo(tmp_path)
    assert lt.tree_problems(repo) == []
    (repo / "results").mkdir()
    (repo / "results" / "a.txt").write_text("x")
    assert lt.tree_problems(repo) == []
    (repo / "reports").mkdir()
    (repo / "reports" / "b.md").write_text("x")                                    # an untracked report file, as in T3-R3's D-1
    assert lt.tree_problems(repo)
    (repo / "reports" / "b.md").unlink()
    (repo / "run" / "a.py").write_text("changed\n")                                # a tracked change
    assert lt.tree_problems(repo)


def test_a_code_diff_since_the_base_is_refused_but_tools_t3_and_tests_are_free(tmp_path):
    repo, base = _repo(tmp_path)
    assert lt.code_diff_problems(repo, base) == []
    (repo / "tools" / "t3" / "b.py").write_text("y\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "t.py").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "tools/t3 and tests")
    assert lt.code_diff_problems(repo, base) == []
    for rel in ("run/a.py", "tools/ms/c.py", "tools/v2/e.py", "agents/d.py", "envs/f.py", "utils/g.py"):         # every frozen path is watched
        (repo / rel).write_text("z\n")
        _git(repo, "commit", "-q", "-am", rel)
        assert any(rel in p for p in lt.code_diff_problems(repo, base)), rel
    assert len(lt.launch_problems(repo, base)) >= 6


def test_the_frozen_paths_and_the_base_commit_are_the_ones_of_the_prompt():
    assert set(lt.CODE_DIFF_PATHS) == {"agents", "envs", "utils", "run", "tools/ms", "tools/v2"}
    assert lt.BASE_COMMIT.startswith("b565f0e8") and len(lt.BASE_COMMIT) == 40
    assert lt.ALLOWED_DIFF == ("seed", "run", "record.run", "record.seed", "record.output_dir", "pilot")


def test_the_frozen_code_is_unchanged_since_the_base_commit():
    """The real repository: ``git diff b565f0e8 HEAD -- agents envs utils run tools/ms tools/v2`` is empty."""
    assert lt.code_diff_problems(ROOT) == []


# ---------------------------------------------------------------------------------- CLI guards, plan, ledger
def test_a_relative_root_is_refused(tmp_path):
    for extra in (["--plan-csv", str(tmp_path / "p.csv")], ["--dry-run"], []):
        with pytest.raises(SystemExit) as exc:
            lt.main(["--root", "results/t3_r4/confirmation", *extra])
        assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_a_dry_run_writes_80_configs_and_an_empty_ledger_and_trains_nothing(tmp_path, lock_dir):
    root = tmp_path / "confirmation"
    assert lt.main(["--root", str(root), "--dry-run", "--lock-dir", str(lock_dir)]) == 0
    cfgs = sorted(root.glob("q*/seed*/bwA/run_config.json"))
    assert len(cfgs) == 80 and not list(root.glob("q*/seed*/bwA/status.json"))
    led = json.loads((root / "infra_events.json").read_text())
    assert led == {"schema": "t3r4_infra_events/1", "events": [], "stopped_without_verdict": []}
    assert json.loads(cfgs[0].read_text())["stage3_retry"] == lt.RETRY


def test_the_plan_has_80_rows_and_the_digests_of_attempt_1_and_2(jobs):
    rows = lt.plan_rows(jobs[:1], Path("/tmp/t3r4_launch_test_root"))
    r = rows[0]
    assert r["run"] == "t3r4_q50_s80501_bwA" and len(r["init_state_sha256"]) == 64 and len(r["attempt2_init_state_sha256"]) == 64
    assert r["init_state_sha256"] != r["attempt2_init_state_sha256"] and len(r["run_config_sha256_relpath"]) == 64
    assert set(r) == set(lt.PLAN_FIELDS)


def test_infrastructure_failures_are_repeated_once_and_recorded(tmp_path):
    jobs, results = [], []
    for i, rc in enumerate((-9, 1)):
        out = tmp_path / "q50" / f"seed8050{i + 1}" / "bwA"
        out.mkdir(parents=True)
        jobs.append(L1.Job({"q": 50, "seed": 80501 + i, "run": f"r{i}"}, str(out)))
        results.append({"out": str(out), "returncode": rc})
    ledger = lt.new_ledger()
    seen = []

    def runner(job):
        seen.append(job.out_dir)
        return {"out": job.out_dir, "returncode": 0}
    reps = lt.handle_infra_failures(jobs, results, tmp_path, runner, ledger)
    assert seen == [jobs[0].out_dir] and len(reps) == 1 and reps[0]["returncode"] == 0       # only the signal is repeated
    assert [e["reason"] for e in ledger["events"]] == ["negative_returncode"] and ledger["stopped_without_verdict"] == []
    assert (tmp_path / "crashed" / "q50" / "seed80501" / "bwA_attempt1").is_dir()
    ledger2 = lt.new_ledger()                                                                 # a second infrastructure failure stops the run
    lt.handle_infra_failures(jobs[:1], [{"out": jobs[0].out_dir, "returncode": -9}], tmp_path,
                             lambda j: {"out": j.out_dir, "returncode": -9}, ledger2)
    assert ledger2["stopped_without_verdict"] == ["r0"] and [e["attempt"] for e in ledger2["events"]] == [1, 2]


# ---------------------------------------------------------------------------------- the real-launch wiring (the pool is stubbed: nothing trains)
def _stub_pool(jobs, workers, record, record_path):
    record["runs"].extend({"out": j.out_dir, "returncode": 0} for j in jobs)


def test_a_real_launch_needs_the_t31_record_and_every_guard_and_then_runs_the_pool(tmp_path, lock_dir, monkeypatch, capsys):
    monkeypatch.setattr(lt, "run_pool", _stub_pool)
    root = tmp_path / "confirmation"
    asked = []
    # (a) the LOCK file holds no T3-1.1 record: refused before anything is written
    monkeypatch.setattr(lt, "LOCK_FILE", _lock_file(tmp_path, lock_dir, with_record=False))
    monkeypatch.setattr(lt, "launch_problems", lambda *a, **k: asked.append("x") or [])
    with pytest.raises(SystemExit) as exc:
        lt.main(["--root", str(root), "--lock-dir", str(lock_dir), "--workers", "1"])
    assert exc.value.code == 2 and not root.exists() and asked == []
    # (b) a record, but a guard fails: refused, nothing written
    monkeypatch.setattr(lt, "LOCK_FILE", _lock_file(tmp_path, lock_dir))
    monkeypatch.setattr(lt, "launch_problems", lambda *a, **k: asked.append("b") or ["HEAD is not the LOCK-record commit"])
    with pytest.raises(SystemExit) as exc:
        lt.main(["--root", str(root), "--lock-dir", str(lock_dir), "--workers", "1"])
    assert exc.value.code == 2 and asked == ["b"] and not root.exists() and "LOCK-record commit" in capsys.readouterr().err
    # (c) a record and clean guards: the configs, the ledger and the launch record are written and the pool runs
    monkeypatch.setattr(lt, "launch_problems", lambda *a, **k: asked.append("c") or [])
    assert lt.main(["--root", str(root), "--lock-dir", str(lock_dir), "--workers", "1", "--code-commit", "abc"]) == 0
    assert asked == ["b", "c"] and len(list(root.glob("q*/seed*/bwA/run_config.json"))) == 80
    rec = json.loads(next(root.glob("launch_*.json")).read_text())
    assert rec["state"] == "done" and rec["n_planned"] == 80 and len(rec["runs"]) == 80 and rec["tool"] == "tools/t3/t3r4_launch.py"
    assert json.loads((root / "infra_events.json").read_text())["events"] == []


def test_a_run_that_exited_0_is_never_an_infrastructure_failure_whatever_its_log_says(tmp_path):
    out = tmp_path / "q50" / "seed80501" / "bwA"
    out.mkdir(parents=True)
    (out / "run.log").write_text("MemoryError: Cannot allocate memory\n")
    job = L1.Job({"q": 50, "seed": 80501, "run": "r0"}, str(out))
    ledger = lt.new_ledger()
    reps = lt.handle_infra_failures([job], [{"out": str(out), "returncode": 0}], tmp_path, lambda j: pytest.fail("a repeat"), ledger)
    assert reps == [] and ledger["events"] == [] and out.is_dir()
    seen = []
    lt.handle_infra_failures([job], [{"out": str(out), "returncode": 1}], tmp_path,
                             lambda j: seen.append(1) or {"out": j.out_dir, "returncode": 0}, ledger)
    assert seen == [1] and [e["reason"] for e in ledger["events"]] == ["out_of_memory"]            # the same log with a nonzero exit is one
    # a repeat that exits 0 is not classified either (its log may say anything), so it cannot become a second infrastructure failure
    out2 = tmp_path / "q60" / "seed80501" / "bwA"
    out2.mkdir(parents=True)
    job2 = L1.Job({"q": 60, "seed": 80501, "run": "r2"}, str(out2))
    ledger2 = lt.new_ledger()

    def repeat_with_a_scary_log(j):
        (Path(j.out_dir) / "run.log").write_text("MemoryError: Cannot allocate memory\n")
        return {"out": j.out_dir, "returncode": 0}
    lt.handle_infra_failures([job2], [{"out": str(out2), "returncode": -9}], tmp_path / "other", repeat_with_a_scary_log, ledger2)
    assert ledger2["stopped_without_verdict"] == [] and [e["attempt"] for e in ledger2["events"]] == [1] and out2.is_dir()
    # while the same repeat with a nonzero exit is the second failure
    out3 = tmp_path / "q60" / "seed80502" / "bwA"
    out3.mkdir(parents=True)
    job3 = L1.Job({"q": 60, "seed": 80502, "run": "r3"}, str(out3))
    ledger3 = lt.new_ledger()

    def repeat_that_fails(j):
        (Path(j.out_dir) / "run.log").write_text("MemoryError: Cannot allocate memory\n")
        return {"out": j.out_dir, "returncode": 1}
    lt.handle_infra_failures([job3], [{"out": str(out3), "returncode": -9}], tmp_path / "other", repeat_that_fails, ledger3)
    assert ledger3["stopped_without_verdict"] == ["r3"]


def test_the_lock_checks_the_protocol_params_and_criteria_files_by_hash(tmp_path, lock_dir):
    js = json.loads((lock_dir / "v2_T3_locked_v1_1.json").read_text())
    d = tmp_path / "mut"
    d.mkdir()
    (d / "v2_T3_locked_v1_1.md").write_text("x")
    for key in ("protocol_sha256", "params_sha256", "criteria_sha256"):
        mut = copy.deepcopy(js)
        mut[key] = "0" * 64
        (d / "v2_T3_locked_v1_1.json").write_text(json.dumps(mut, indent=1) + "\n")
        with pytest.raises(ValueError, match="differs from the lock"):
            lt.check_lock(d, use_record=False)


def test_the_ledger_is_saved_after_every_event_before_the_repeat_starts(tmp_path):
    out = tmp_path / "q50" / "seed80501" / "bwA"
    out.mkdir(parents=True)
    job = L1.Job({"q": 50, "seed": 80501, "run": "r0"}, str(out))
    ledger = lt.new_ledger()

    def runner(j):
        saved = json.loads((tmp_path / "infra_events.json").read_text())                  # the repeat starts: the first event is on disk
        assert [e["attempt"] for e in saved["events"]] == [1]
        raise RuntimeError("the launcher dies during the repeat")
    with pytest.raises(RuntimeError):
        lt.handle_infra_failures([job], [{"out": str(out), "returncode": -9}], tmp_path, runner, ledger)
    assert [e["attempt"] for e in json.loads((tmp_path / "infra_events.json").read_text())["events"]] == [1]
    ledger2 = lt.new_ledger()
    out2 = tmp_path / "q50" / "seed80502" / "bwA"
    out2.mkdir(parents=True)
    job2 = L1.Job({"q": 50, "seed": 80502, "run": "r2"}, str(out2))
    lt.handle_infra_failures([job2], [{"out": str(out2), "returncode": -9}], tmp_path, lambda j: {"out": j.out_dir, "returncode": -9}, ledger2)
    on_disk = json.loads((tmp_path / "infra_events.json").read_text())
    assert on_disk["stopped_without_verdict"] == ["r2"] and [e["attempt"] for e in on_disk["events"]] == [1, 2]
