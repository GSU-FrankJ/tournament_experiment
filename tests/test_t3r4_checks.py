"""Tests of ``tools/t3/t3r4_checks.py``: the per-run checks on reduced fired / not-fired runs (budget 40 / 40 / 20 updates), with negative
controls, and the aggregate rule."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

import t3_helpers as th  # noqa: E402
import t3r4_checks as ck  # noqa: E402
import t3r4_launch as lt  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from t3_launch import relpath_sha256  # noqa: E402

LOCK = {"reference_config_sha256": {}}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two reduced runs in the launcher's layout ``<root>/q50/seed50500/bwA``: the trigger fires (``delta_gt`` 1e-9) / does not (1e9)."""
    out = {}
    for name, delta in (("fired", 1e-9), ("nofire", 1e9)):
        root = tmp_path_factory.mktemp(name) / "results" / "t3_r4" / "confirmation"
        d = root / "q50" / "seed50500" / "bwA"
        d.mkdir(parents=True)
        cfg = {**th.bw_cfg(str(d)), "stage3_retry": {"delta_gt": delta, "max_retries": 1}}
        cfg["run"] = cfg["record"]["run"] = f"t3r4test_{name}"
        assert rms.run_pipeline(cfg, str(d), "pytest") == 0, name
        out[name], out[name + "_root"] = str(d), str(root)
    return out


def _job(d, cfg):
    return {"wave": "t3r4", "arm": "bwA", "q": 50, "seed": 50500, "run": cfg["run"], "run_dir": str(d), "kind": "bw", "run_kind": "confirmation"}


def _plan(d, root, fired):
    cfg = json.load(open(Path(d) / "run_config.json"))
    man = json.load(open(Path(d) / "manifest.json"))
    return cfg, {"run_config_sha256_relpath": relpath_sha256(cfg, Path(root)), "init_state_sha256": man["init_state_sha256"],
                 "attempt2_init_state_sha256": json.load(open(Path(d) / "retry.json"))["attempt2"]["init_state_sha256"] if fired else ""}


def test_the_jobs_are_the_80_runs_in_plan_order(tmp_path):
    jobs = ck.all_jobs(tmp_path)
    assert len(jobs) == 80 and jobs[0]["run"] == "t3r4_q50_s80501_bwA" and jobs[39]["run"] == "t3r4_q50_s80540_bwA"
    assert jobs[40]["run"] == "t3r4_q60_s80501_bwA" and jobs[-1]["run_dir"] == str(tmp_path / "q60" / "seed80540" / "bwA")
    assert {j["arm"] for j in jobs} == {"bwA"} and len({j["run"] for j in jobs}) == 80


@pytest.mark.parametrize("name,fired", [("fired", True), ("nofire", False)])
def test_check_run_on_a_reduced_run(runs, name, fired):
    """The deterministic parts (the production budgets are not met by a reduced run, so ``budget_ok`` is not asserted)."""
    d, root = Path(runs[name]), Path(runs[name + "_root"])
    cfg, plan = _plan(d, root, fired)
    job = _job(d, cfg)
    man = json.load(open(d / "manifest.json"))
    res = ck.check_run((job, plan, man["commit"], LOCK, str(root)))
    assert "error" not in res and res["exit_ok"] and res["fired"] is fired and res["files_complete"], res
    assert res["manifest_commit_ok"] and res["global_rng_ok"] and res["config_equals_plan"] and res["init_equals_plan"], res
    assert res["trigger_consistent"] and res["would_fire"] is fired and res["lr_ok"] and res["replay_ok"], res
    assert res["config_production_ok"] is False                             # seed 50500 is outside the confirmation block
    if fired:
        assert res["total_updates"] == 140 and res["retry_budget_ok"] is False        # 140 = 40 + 40 + 40 + 20, not the production 5,400
    else:
        assert res["total_updates"] == 3800 and res["retry_budget_ok"] is True
    # negative controls: another config hash, init digest, commit
    assert ck.check_run((job, dict(plan, run_config_sha256_relpath="0" * 64), None, LOCK, str(root)))["config_equals_plan"] is False
    assert ck.check_run((job, dict(plan, init_state_sha256="0" * 64), None, LOCK, str(root)))["init_equals_plan"] is False
    assert ck.check_run((job, plan, "deadbeef", LOCK, str(root)))["manifest_commit_ok"] is False
    if fired:
        assert ck.check_run((job, dict(plan, attempt2_init_state_sha256="0" * 64), None, LOCK, str(root)))["init_equals_plan"] is False
        assert ck.check_run((job, dict(plan, attempt2_init_state_sha256=""), None, LOCK, str(root)))["init_equals_plan"] is False


def test_a_fired_run_without_attempt1_or_a_quiet_run_with_it_fails_the_file_check(runs, tmp_path):
    fired_copy = tmp_path / "fired" / "q50" / "seed50500" / "bwA"
    shutil.copytree(runs["fired"], fired_copy)
    cfg, plan = _plan(fired_copy, tmp_path / "fired", True)
    shutil.rmtree(fired_copy / "attempt1")
    res = ck.check_run((_job(fired_copy, cfg), plan, None, LOCK, str(tmp_path / "fired")))
    assert res["files_complete"] is False and "attempt1" in res["files_missing"]
    quiet_copy = tmp_path / "quiet" / "q50" / "seed50500" / "bwA"
    shutil.copytree(runs["nofire"], quiet_copy)
    (quiet_copy / "attempt1").mkdir()
    cfg, plan = _plan(quiet_copy, tmp_path / "quiet", False)
    res = ck.check_run((_job(quiet_copy, cfg), plan, None, LOCK, str(tmp_path / "quiet")))
    assert res["files_complete"] is False and res["files_missing"] == "attempt1"


def test_the_trigger_check_catches_a_retry_json_that_disagrees_with_the_check_file(runs, tmp_path):
    base = tmp_path / "fired" / "q50" / "seed50500" / "bwA"
    shutil.copytree(runs["fired"], base)
    cfg, plan = _plan(base, tmp_path / "fired", True)
    rj = json.load(open(base / "retry.json"))
    rj["attempt1"]["reasons"] = ["not_finite"]
    json.dump(rj, open(base / "retry.json", "w"))
    assert ck.check_run((_job(base, cfg), plan, None, LOCK, str(tmp_path / "fired")))["trigger_consistent"] is False
    # a fired run whose retry.json is missing is a run whose trigger "did not fire" with a stray attempt1/: caught by the file check
    (base / "retry.json").unlink()
    res = ck.check_run((_job(base, cfg), plan, None, LOCK, str(tmp_path / "fired")))
    assert res["fired"] is False and res["files_complete"] is False and res["trigger_consistent"] is False


def test_an_exception_inside_the_checks_of_one_run_is_recorded_and_does_not_escape(runs):
    d = Path(runs["fired"])
    cfg = json.load(open(d / "run_config.json"))
    res = ck.check_run((_job(d, cfg), "not a plan row", None, LOCK, str(runs["fired_root"])))      # a malformed plan row raises inside the checks
    assert "error" in res and res["arm"] == "bwA"


def _agg(**over):
    base = {"verdicts": {"exit_ok": {"pass": 80, "fail": 0, "not_evaluated": 0}}, "infra_events_present": True, "n_runs": 80}
    base.update(over)
    return base


def test_checks_ok_needs_every_verdict_the_ledger_and_80_runs():
    assert ck.checks_ok(_agg())
    assert not ck.checks_ok(_agg(verdicts={"exit_ok": {"pass": 79, "fail": 1, "not_evaluated": 0}}))
    assert not ck.checks_ok(_agg(verdicts={"exit_ok": {"pass": 79, "fail": 0, "not_evaluated": 1}}))
    assert not ck.checks_ok(_agg(infra_events_present=False))
    assert not ck.checks_ok(_agg(n_runs=79)) and not ck.checks_ok(_agg(n_runs=122))
