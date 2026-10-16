"""T3-R1: tests of ``tools/t3/t3_configs.py`` and ``tools/t3/t3_launch.py`` (waves, seed guards, the arm configs against
each other, the plan with its hashes, the production values of D5) and of the pre-launch / post-launch checks of
``tools/t3/t3_launch_checks.py``.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_t3_launcher.py -p no:cacheprovider -q
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import shutil

import pytest

import t3_helpers as H
import t3_configs as tc  # noqa: E402
import t3_launch as tl  # noqa: E402
import t3_launch_checks as lc  # noqa: E402

PROTO, PARAMS = H.PROTO, H.BW_PARAMS


def allowed(diffs, prefixes):
    return [p for p in diffs if not any(p == a or p.startswith(a + ".") or p.startswith(a + "[") for a in prefixes)]


# ====================================================================== waves, seeds, plan
@pytest.mark.parametrize("wave,n", [("smoke", 8), ("t3", 120), ("t2j", 40)])
def test_job_counts(tmp_path, wave, n):
    jobs = tl.build_jobs(wave, tc.QS, list(tc.WAVE_SEEDS[wave]), None, tmp_path, PARAMS)
    assert len(jobs) == n
    names = [j.cfg["run"] for j in jobs]
    assert len(set(names)) == n and len({j.out_dir for j in jobs}) == n
    assert all(j.cfg["pilot"] == "t3_r1" if j.cfg["arm"] == "bw" else j.cfg["round"] == "t3_r1" for j in jobs)


def test_seed_guards(tmp_path):
    for wave, bad in (("t3", [40501]), ("t3", [10501]), ("t3", [50500]), ("t3", [30501]), ("t2j", [50501]),
                      ("t2j", [30500]), ("t2j", [30521]), ("smoke", [50501]), ("smoke", [40520]), ("smoke", [10530])):
        with pytest.raises(ValueError):
            tl.build_jobs(wave, tc.QS, bad, None, tmp_path, PARAMS)
    assert tc.WAVE_SEEDS["t3"] == tuple(range(50501, 50521)) and tc.WAVE_SEEDS["t2j"] == tuple(range(30501, 30521))
    assert tc.WAVE_SEEDS["smoke"] == (50500,)
    allseeds = {s for w in tc.WAVES for s in tc.WAVE_SEEDS[w]}
    assert not allseeds & set(tc.FORBIDDEN_SEEDS) and set(range(40501, 40521)) <= set(tc.FORBIDDEN_SEEDS)
    assert set(range(10501, 10531)) <= set(tc.FORBIDDEN_SEEDS)
    with pytest.raises(ValueError):
        tl.build_jobs("t3", [55], [50501], None, tmp_path, PARAMS)
    with pytest.raises(ValueError):
        tl.build_jobs("t2j", tc.QS, [30501], ["bw"], tmp_path, PARAMS)


def test_cli_refuses_forbidden_seeds_and_a_modified_params_file(tmp_path, capsys):
    for wave, seed in (("t3", "40501"), ("t2j", "10501"), ("smoke", "50501")):
        with pytest.raises(SystemExit):
            tl.main(["--wave", wave, "--seeds", seed, "--plan-csv", str(tmp_path / "p.csv")])
    bad = tmp_path / "params.json"
    d = json.load(open(tc.PARAMS_PATH))
    d["K"] = 26
    json.dump(d, open(bad, "w"), indent=1)
    with pytest.raises(SystemExit):
        tl.main(["--wave", "smoke", "--params", str(bad), "--plan-csv", str(tmp_path / "p.csv")])
    assert tc.sha256_file(tc.PARAMS_PATH) == tc.PARAMS_SHA256


def test_run_directories(tmp_path):
    assert tc.run_dir(tmp_path, "t3", 50, 50501, "bw") == tmp_path / "runs" / "q50" / "seed50501" / "bw"
    assert tc.run_dir(tmp_path, "t2j", 60, 30520, "jt2") == tmp_path / "runs_t2" / "q60" / "seed30520" / "jt2"
    assert tc.run_dir(tmp_path, "smoke", 60, 50500, "jt_es") == tmp_path / "smoke" / "q60" / "jt_es"
    assert tc.run_name(50, 50501, "jt_es") == "t3r1_q50_s50501_jt_es"


def test_plan_has_168_rows_with_unique_names_and_a_path_independent_hash(tmp_path):
    rows = []
    for wave in tc.WAVES:
        for root in (tmp_path / "a" / "results" / "t3_r1", tmp_path / "b" / "results" / "t3_r1"):
            jobs = tl.build_jobs(wave, tc.QS, list(tc.WAVE_SEEDS[wave]), None, root, PARAMS)
            r = tl.plan_rows(wave, jobs, root)
            rows.append((wave, root.parent.parent.name, r))
    for wave in tc.WAVES:
        a = [x for x in rows if x[0] == wave and x[1] == "a"][0][2]
        b = [x for x in rows if x[0] == wave and x[1] == "b"][0][2]
        assert [r["run_config_sha256_relpath"] for r in a] == [r["run_config_sha256_relpath"] for r in b]
        assert [r["run_config_sha256"] for r in a] != [r["run_config_sha256"] for r in b]      # the absolute path is in the text
    allr = [r for x in rows if x[1] == "a" for r in x[2]]
    assert len(allr) == 168 and len({r["run"] for r in allr}) == 168
    p = tmp_path / "plan.csv"
    tl.write_plan_csv(p, allr)
    with pytest.raises(FileExistsError):
        tl.write_plan_csv(p, allr)
    assert [r for r in csv.DictReader(open(p))][0]["wave"] == "smoke"


def test_dry_run_writes_valid_configs_and_launches_nothing(tmp_path, capsys):
    root = tmp_path / "results" / "t3_r1"
    assert tl.main(["--wave", "smoke", "--dry-run", "--root", str(root), "--workers", "2"]) == 0
    rec = json.load(open(next((root / "smoke").glob("dryrun_*.json"))))
    assert rec["n_planned"] == 8 and rec["validation"]["n_invalid"] == 0 and rec["state"] == "dry_run"
    assert sorted(rec["validation"]["per_arm"]) == ["bw", "jt", "jt2", "jt_es"]
    assert rec["params_sha256"] == tc.PARAMS_SHA256 and "diff_stat_code_commit_to_head" not in rec
    cfgs = sorted((root / "smoke").glob("q*/*/run_config.json"))
    assert len(cfgs) == 8 and not list((root / "smoke").glob("q*/*/status.json"))


def test_several_waves_refuse_explicit_arms_and_the_launch_commit_is_required(tmp_path):
    for extra in (["--arms", "jt"], ["--seeds", "50501"]):
        with pytest.raises(SystemExit):
            tl.main(["--wave", "t3", "t2j", *extra, "--plan-csv", str(tmp_path / "x.csv")])
    with pytest.raises(SystemExit):                    # post-launch needs the launch commit and the planned digests
        lc.main(["post-launch", "--out", str(tmp_path / "o.json"), "--plan-digests", str(tmp_path / "p.csv")])
    with pytest.raises(SystemExit):
        lc.main(["post-launch", "--out", str(tmp_path / "o.json"), "--commit", "x" * 40])


def test_the_deterministic_checks_decide_the_exit_status():
    ok = {"verdicts": {"exit_ok": {"pass": 4, "fail": 0, "not_evaluated": 0}, "lr_ok": {"pass": 4, "fail": 0, "not_evaluated": 0}},
          "c_init_within_q_seed": {"groups": 2, "equal": 2}, "c_init_equals_plan": {"runs": 4, "equal": 4},
          "start_share_flags": {"cells": 100, "flags": 5, "expected_under_null": 0.27}}
    assert lc.post_launch_ok(ok) is True                     # the z flags are descriptive
    import copy as _c
    for path, val in ((("verdicts", "lr_ok", "fail"), 1), (("verdicts", "exit_ok", "not_evaluated"), 1),
                      (("c_init_within_q_seed", "equal"), 1), (("c_init_equals_plan", "equal"), 3)):
        bad = _c.deepcopy(ok)
        node = bad
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = val
        assert lc.post_launch_ok(bad) is False, path
    no_plan = _c.deepcopy(ok)
    del no_plan["c_init_equals_plan"]
    assert lc.post_launch_ok(no_plan) is True


def test_two_waves_are_one_launch_with_one_record_and_one_plan(tmp_path):
    root = tmp_path / "results" / "t3_r1"
    plan = tmp_path / "plan.csv"
    assert tl.main(["--wave", "t3", "t2j", "--root", str(root), "--plan-csv", str(plan)]) == 0
    rows = list(csv.DictReader(open(plan)))
    assert len(rows) == 160 and [r["wave"] for r in rows].count("t3") == 120 and [r["wave"] for r in rows].count("t2j") == 40
    assert tl.main(["--wave", "t3", "t2j", "--dry-run", "--root", str(root), "--workers", "2"]) == 0
    rec = json.load(open(next(root.glob("dryrun_t3_t2j_*.json"))))
    assert rec["n_planned"] == 160 and rec["wave"] == "t3+t2j" and rec["validation"]["n_invalid"] == 0
    assert len(list((root / "runs").glob("q*/seed*/*/run_config.json"))) == 120
    assert len(list((root / "runs_t2").glob("q*/seed*/jt2/run_config.json"))) == 40
    with pytest.raises(SystemExit):
        tl.main(["--wave", "t3", "t2j", "--seeds", "50501", "--plan-csv", str(tmp_path / "x.csv")])


def test_job_command_and_pool(tmp_path, monkeypatch):
    out = str(tmp_path / "o")
    bw = tc.build_config(PROTO, 50, 50501, "bw", out, PARAMS)
    jt = tc.build_config(PROTO, 50, 50501, "jt", out, PARAMS)
    cb, cj = tl.job_command(bw, out), tl.job_command(jt, out)
    assert cb[-4:] == ["--config", os.path.join(out, "run_config.json"), "--out-dir", out] and cb[-5].endswith("run_ms_stagewise.py")
    assert cj[-5].endswith("run_joint.py") and cj[:3] == [tl.PY, "-B", "-u"]
    # the pool runs every job and records the return codes
    jobs = [tl.L1.Job(copy.deepcopy(jt), str(tmp_path / f"j{i}")) for i in range(3)]
    for j in jobs:
        os.makedirs(j.out_dir)
    monkeypatch.setattr(tl, "job_command", lambda cfg, out_dir: [tl.PY, "-c", "print('ok')"])
    rec = {"runs": []}
    tl.run_pool(jobs, 2, rec, tmp_path / "rec.json")
    assert rec["state"] == "done" and sorted(r["returncode"] for r in rec["runs"]) == [0, 0, 0]
    assert all(open(os.path.join(j.out_dir, "run.log")).read() == "ok\n" for j in jobs)
    monkeypatch.setattr(tl, "job_command", lambda cfg, out_dir: [tl.PY, "-c", "raise SystemExit(3)"])
    jobs2 = [tl.L1.Job(copy.deepcopy(jt), str(tmp_path / "k0"))]
    os.makedirs(jobs2[0].out_dir)
    rec2 = {"runs": []}
    tl.run_pool(jobs2, 1, rec2, tmp_path / "rec2.json")
    assert rec2["state"] == "failed" and rec2["runs"][0]["returncode"] == 3


def test_a_busy_output_directory_is_refused(tmp_path):
    root = tmp_path / "results" / "t3_r1"
    assert tl.main(["--wave", "smoke", "--dry-run", "--root", str(root)]) == 0
    jobs = tl.build_jobs("smoke", tc.QS, [50500], None, root, PARAMS)
    d = jobs[0].out_dir
    open(os.path.join(d, "status.json"), "w").write("{}")
    with pytest.raises(SystemExit):
        tl.main(["--wave", "smoke", "--root", str(root), "--dry-run"])


# ====================================================================== configs against each other
@pytest.mark.parametrize("q,seed", [(50, 50501), (60, 50520)])
def test_jt_and_jt_es_differ_only_in_the_arm_labels_and_the_budget_keys(q, seed):
    a = tc.build_config(PROTO, q, seed, "jt", "/x/jt", PARAMS)
    b = tc.build_config(PROTO, q, seed, "jt_es", "/x/jt_es", PARAMS)
    diff = tc.config_diff(a, b)
    assert not allowed(diff, ["arm", "run", "record.run", "record.output_dir", "joint.updates", "joint.starts",
                              "joint.lr_windows"]), diff
    assert {"arm", "run", "record.run", "record.output_dir", "joint.updates"} <= set(diff)
    assert a["joint"]["T"] == b["joint"]["T"] == 3 and a["joint"]["check_every"] == b["joint"]["check_every"] == 25


@pytest.mark.parametrize("arm", ["jt", "jt_es"])
@pytest.mark.parametrize("q,seed", [(50, 50501), (60, 50507)])
def test_bw_and_a_joint_arm_share_the_record_and_the_provenance(arm, q, seed):
    bw = tc.build_config(PROTO, q, seed, "bw", "/x/bw", PARAMS)
    jt = tc.build_config(PROTO, q, seed, arm, f"/x/{arm}", PARAMS)
    ra, rb = copy.deepcopy(bw["record"]), copy.deepcopy(jt["record"])
    for r in (ra, rb):
        r.pop("run")
        r.pop("output_dir")
    assert ra == rb
    for k in ("base_commit", "protocol_ref", "record_sha256", "threads_per_process", "clamp_likelihood", "init_digest",
              "q", "seed"):
        assert bw[k] == jt[k], k
    assert bw["base_commit"] == "657f54a" and jt["round"] == "t3_r1" and bw["pilot"] == "t3_r1"


@pytest.mark.parametrize("arm", ["bw", "jt", "jt_es", "jt2"])
def test_within_an_arm_configs_differ_only_in_q_seed_run_output_and_derived_fields(arm):
    base = tc.build_config(PROTO, 50, 50501, arm, "/x/a", PARAMS)
    other_seed = tc.build_config(PROTO, 50, 50502, arm, "/x/b", PARAMS)
    assert not allowed(tc.config_diff(base, other_seed), ["seed", "run", "record.seed", "record.run", "record.output_dir"])
    other_q = tc.build_config(PROTO, 60, 50501, arm, "/x/c", PARAMS)
    rec_diff = {"record." + p for p in tc.config_diff(PROTO["records"]["50"], PROTO["records"]["60"])}
    ok = ["q", "run", "record.run", "record.output_dir", "record_sha256", "record.q", "record.game.q"] + sorted(rec_diff)
    d = tc.config_diff(base, other_q)
    assert not allowed(d, ok), allowed(d, ok)
    assert "record.game.k" not in d and "record.protocol.episodes_per_update" not in d     # the economics other than q are shared


def test_the_bw_config_embeds_the_unmodified_record_hash():
    cfg = tc.build_config(PROTO, 50, 50501, "bw", "/x", PARAMS)
    assert cfg["record_sha256"] == tc.mc.sha256_json(PROTO["records"]["50"])
    assert cfg["protocol_ref"] == {"path": "protocols/v2_T2_locked_v2_0.json",
                                   "sha256": tc.sha256_file(tc.PROTOCOL_PATH)}


# ====================================================================== D5 arithmetic and production values
def test_the_budget_table_of_d5():
    b = {a: tc.expected_budget(a) for a in ("bw", "jt", "jt_es", "v20", "jt2")}
    assert b["bw"]["rows_by_stage"] == {1: 307200, 2: 819200, 3: 819200} and b["bw"]["rows_total"] == 1945600
    assert b["bw"]["minibatch_steps"] == 76000 and b["bw"]["updates"] == 3800
    assert b["jt"]["rows_by_stage"] == {1: 648704, 2: 648704, 3: 648704} and b["jt"]["rows_total"] == 1946112
    assert b["jt"]["minibatch_steps"] == 76020 and b["jt"]["updates"] == 1267
    assert b["jt_es"]["rows_per_update"] == {1: 256, 2: 341, 3: 512}
    assert b["jt_es"]["rows_by_stage"] == {1: 449024, 2: 598114, 3: 898048} and b["jt_es"]["rows_total"] == 1945186
    assert b["jt_es"]["minibatch_steps"] == 87700 and b["jt_es"]["updates"] == 1754
    assert b["v20"]["rows_by_stage"] == {1: 307200, 2: 819200} and b["v20"]["minibatch_steps"] == 56000
    assert b["jt2"]["rows_by_stage"] == {1: 563200, 2: 563200} and b["jt2"]["rows_total"] == 1126400
    assert b["jt2"]["minibatch_steps"] == 44000 and b["jt2"]["updates"] == 1100
    # equal policy rows to within one update
    for a in ("jt", "jt_es"):
        assert abs(b[a]["rows_total"] - b["bw"]["rows_total"]) < sum(b[a]["rows_per_update"].values())
    assert b["jt2"]["rows_total"] == b["v20"]["rows_total"]
    for arm, w in (("jt", (951, 1267)), ("jt_es", (1316, 1754)), ("jt2", (826, 1100))):
        assert tc.decay_window(tc.JOINT_BUDGETS[arm]["updates"]) == w == tc.JOINT_BUDGETS[arm]["window"]


def test_production_problems_detect_departures():
    cfgs = {a: tc.build_config(PROTO, 50, 50501, a, "/x", PARAMS) for a in tc.ARMS}
    assert all(tc.production_problems(c) == [] for c in cfgs.values())
    c = copy.deepcopy(cfgs["jt"])
    c["record"]["protocol"]["episodes_per_update"] = 48
    assert tc.production_problems(c)
    c = copy.deepcopy(cfgs["jt_es"])
    c["joint"]["starts"]["root"], c["joint"]["starts"]["stage2"] = 257, 84
    assert tc.production_problems(c)
    c = copy.deepcopy(cfgs["jt2"])
    c["joint"]["lr_windows"][0]["first"] = 800
    assert tc.production_problems(c)
    c = copy.deepcopy(cfgs["jt"])
    c["joint"]["check_every"] = 20
    assert tc.production_problems(c)
    c = copy.deepcopy(cfgs["bw"])
    c["pipeline"]["budgets"]["3"] = 1599
    assert tc.production_problems(c)
    c = copy.deepcopy(cfgs["bw"])
    c["rule"]["enabled"] = True
    assert tc.production_problems(c)
    c = copy.deepcopy(cfgs["bw"])
    c["run"] = "msr1_q50_s50501_bw"
    assert tc.production_problems(c)


def test_the_parameter_file_has_the_values_of_d3():
    prm = tc.load_bw_params()
    src = json.load(open(tc.ROOT / tc.MS_PARAMS_REL))
    assert tc.sha256_file(tc.ROOT / tc.MS_PARAMS_REL) == tc.MS_PARAMS_SHA256
    for k in ("K", "M", "conc_limit", "localized_fraction", "alpha_polish", "ema_beta", "near_tie_half_width"):
        assert prm[k] == src[k], k
    assert prm["stages"] == tc.BW_STAGE_PARAMS and prm["legacy_pipeline"] == {"budgets": tc.BW_BUDGETS, "lr_windows": tc.BW_WINDOWS}
    assert {k: src["stages"]["2"][k] for k in ("eps", "rho", "tau")} == prm["stages"]["3"]
    assert {k: src["stages"]["1"][k] for k in ("eps", "rho", "tau")} == prm["stages"]["2"] == prm["stages"]["1"]


# ====================================================================== checks (C-INIT plan, post launch)
def test_c_init_plan_on_a_few_seeds(tmp_path, monkeypatch):
    monkeypatch.setitem(tc.WAVE_SEEDS, "t3", (50501,))
    monkeypatch.setitem(tc.WAVE_SEEDS, "t2j", (30501,))
    monkeypatch.setitem(tc.WAVE_SEEDS, "smoke", (50500,))
    out = tmp_path / "c_init_plan.csv"
    res = lc.c_init_plan(["smoke", "t3", "t2j"], out, json.load(open(tc.ROOT / tc.MS_PARAMS_REL)))
    assert res["t3_groups"] == 4 and res["t3_equal"] == 4 and res["t2_groups"] == 4 and res["t2_equal"] == 4
    rows = list(csv.DictReader(open(out)))
    assert {r["arm"] for r in rows} == {"bw", "jt", "jt_es", "jt2", "ms_t2_reference"}
    by = {(r["q"], r["seed"], r["arm"]): r["init_state_sha256"] for r in rows}
    assert by[("50", "50501", "bw")] == by[("50", "50501", "jt")] == by[("50", "50501", "jt_es")]
    assert by[("50", "50501", "bw")] != by[("60", "50501", "bw")] and by[("50", "50501", "bw")] != by[("50", "50500", "bw")]
    assert by[("60", "30501", "jt2")] == by[("60", "30501", "ms_t2_reference")]


def _reduced_job(name):
    d, cfg = H.reduced_runs()[name]
    arm = cfg["arm"]
    return {"wave": "t3", "arm": arm, "q": cfg["q"], "seed": cfg["seed"], "run": cfg["run"], "run_dir": d}


def _reduced_expected(name):
    d, cfg = H.reduced_runs()[name]
    T, U = cfg["joint"]["T"], cfg["joint"]["updates"]
    per = {t: v for t, v in lc.rj.policy_rows_per_update(cfg["joint"]["starts"], T).items()}
    n = sum(per.values())
    return {"updates": U, "rows_per_update": per, "rows_by_stage": {t: per[t] * U for t in per},
            "rows_total": n * U, "minibatch_steps": U * 10 * math.ceil(n / 256)}


def test_post_launch_flags_the_reduced_runs_that_are_not_production_runs():
    """A reduced run fails exactly the checks that pin production values (plan equality, 25-update schedules)."""
    res = lc.check_run(_reduced_job("t3_es"), _reduced_expected("t3_es"), commit=None)
    assert res["exit_ok"] and res["files_complete"] and res["global_rng_ok"] and res["replay_ok"]
    assert res["budget_ok"] and res["lr_ok"] and res["refresh_ok"]
    assert not res["config_equals_plan"] and not res["config_production_ok"]      # 48 episodes, U = 30
    assert not res["exports_ok"] and not res["checks_ok"]                         # exports and checks every 10 updates
    man = json.load(open(os.path.join(_reduced_job("t3_es")["run_dir"], "manifest.json")))
    assert res["clean_tree"] is man["clean_tree"]                       # the flag of the manifest, whatever the tree was
    assert lc.check_run(_reduced_job("t3_es"), _reduced_expected("t3_es"), commit=man["commit"])["manifest_commit_ok"] is True
    assert lc.check_run(_reduced_job("t3_es"), _reduced_expected("t3_es"), commit="0" * 40)["manifest_commit_ok"] is False


def test_post_launch_detects_a_wrong_budget_and_a_missing_file(tmp_path):
    d, cfg = H.reduced_runs()["t3_root"]
    copy_dir = tmp_path / "copy"
    shutil.copytree(d, copy_dir)
    job = dict(_reduced_job("t3_root"), run_dir=str(copy_dir))
    exp = _reduced_expected("t3_root")
    assert lc.check_run(job, exp, None)["budget_ok"]
    bad = dict(exp, minibatch_steps=exp["minibatch_steps"] + 1)
    assert not lc.check_run(job, bad, None)["budget_ok"]
    os.remove(copy_dir / "visitation.npz")
    r = lc.check_run(job, exp, None)
    assert r["files_complete"] is False and "visitation.npz" in r["files_missing"] and "budget_ok" not in r
    json.dump({"state": "failed", "exit_code": 1}, open(copy_dir / "status.json", "w"))
    assert lc.check_run(job, exp, None)["exit_ok"] is False


def test_post_launch_replay_detects_a_changed_array(tmp_path):
    import numpy as np
    d, cfg = H.reduced_runs()["t2_root"]
    copy_dir = tmp_path / "copy"
    shutil.copytree(d, copy_dir)
    assert lc.check_run(dict(_reduced_job("t2_root"), run_dir=str(copy_dir)), _reduced_expected("t2_root"), None)["replay_ok"]
    z = dict(np.load(copy_dir / "final_eval_final.npz"))
    k = "v_t2_delta"
    z[k] = z[k].copy()
    z[k][3] += 1e-15
    np.savez(copy_dir / "final_eval_final.npz", **z)
    assert not lc.check_run(dict(_reduced_job("t2_root"), run_dir=str(copy_dir)), _reduced_expected("t2_root"), None)["replay_ok"]


def test_z_flags_and_lr_expected():
    import numpy as np
    r = lc.z_flags(np.array([100, 100, 100, 100]), np.full(4, 0.25))
    assert r == {"cells": 4, "flags": 0, "max_abs_z": 0.0}
    r = lc.z_flags(np.array([160, 40, 100, 100]), np.full(4, 0.25))
    assert r["flags"] == 2 and r["cells"] == 4
    assert lc.z_flags(np.array([5, 5]), np.array([0.0, 1.0]))["cells"] == 0
    sched = dict(H.PROTO["records"]["50"]["lr_schedule"])
    w = [{"first": 951, "last": 1267, "start": 3e-4, "end": 3e-5}]
    assert lc.lr_expected(sched, w, 950) == 3e-4 and lc.lr_expected(sched, w, 951) == 3e-4
    assert lc.lr_expected(sched, w, 1267) == pytest.approx(3e-5, rel=1e-12)


def test_compare_to_gates_is_exact():
    row = {"eta_T_over_dw": 0.1, "dev_eta_T_over_dw": 0.2, "rmse_pos_T_over_eT0": 0.3, "tail_mean_T_over_eT0": 0.4,
           "Gmax_full_over_dw": 0.5, "dev_Gmax_full_over_dw": 0.6, "stage1_rel_err_abs": 0.7,
           "gate_G_A": True, "gate_G_F": False, "gate_G_N": True, "gate_G_S": True}
    gates = {"metric_values": {"eta_final": 0.1, "eta_dev": 0.2, "rmse": 0.3, "tail": 0.4, "gmax_final": 0.5,
                               "gmax_dev": 0.6, "s1": 0.7},
             "G-A": {"pass": True}, "G-F": {"pass": False}, "G-N": {"pass": True}, "G-S": {"pass": True}}
    assert all(lc.compare_to_gates(row, gates).values())
    gates["metric_values"]["tail"] = 0.4 + 1e-17 * 40       # a different float
    assert not lc.compare_to_gates(row, gates)["tail"]
    gates["G-F"]["pass"] = True
    assert not lc.compare_to_gates(row, gates)["G-F"]
    assert not lc.compare_to_gates({}, gates)["eta_final"]
