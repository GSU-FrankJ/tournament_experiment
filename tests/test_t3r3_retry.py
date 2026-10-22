"""Tests of the P2 scheme of round T3-R3: candidate A (``stage3_retry``, ``run/ms_retry.py``) and candidate B
(``terminal_clamp_likelihood``) in ``run/run_ms_stagewise.py``, on reduced-budget configs in temporary directories."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

import t3_helpers as th  # noqa: E402
import t3r3_pair as tp  # noqa: E402
from agents.ppo_curriculum import PPOConfig  # noqa: E402
from agents.ppo_curriculum_v2 import CurriculumPPOv2  # noqa: E402
from run import ms_retry  # noqa: E402
from run import run_ms_stagewise as rms  # noqa: E402
from utils.beta_tail import log_betainc_small_x  # noqa: E402


def _check(delta=0.001, s=60.0, C=0.03, **extra):
    return {"stage": 3, "local": 1600, "update": 1600, "valid": True, "Delta": delta, "s": s, "C": C, "R": 0.2, "R_tail": 0.01, **extra}


# ---------------------------------------------------------------------------------- the config keys
def test_the_keys_are_declared_and_validated():
    cfg = th.bw_cfg("x", budgets=(3, 3, 3))
    rms.validate_config(cfg)
    for add in ({"stage3_retry": {"delta_gt": 0.005, "max_retries": 1}}, {"terminal_clamp_likelihood": "censored"},
                {"stage3_retry": {"delta_gt": 0.005, "max_retries": 1}, "terminal_clamp_likelihood": "censored"}):
        rms.validate_config({**copy.deepcopy(cfg), **add})
    bad = [{"stage3_retry": {"delta_gt": 0.005}}, {"stage3_retry": {"delta_gt": 0.005, "max_retries": 2}},
           {"stage3_retry": {"delta_gt": 0.0, "max_retries": 1}}, {"stage3_retry": {"delta_gt": float("nan"), "max_retries": 1}},
           {"stage3_retry": {"delta_gt": 0.005, "max_retries": True}}, {"stage3_retry": "yes"},
           {"terminal_clamp_likelihood": "density"}, {"terminal_clamp_likelihood": "mass"}]
    for add in bad:
        with pytest.raises(rms.ConfigError):
            rms.validate_config({**copy.deepcopy(cfg), **add})
    assert ms_retry.validate({"delta_gt": 0.005, "max_retries": 1}, 2) is not None        # T = 3 only
    with pytest.raises(rms.ConfigError):
        rms.validate_config({**copy.deepcopy(cfg), "unknown_key": 1})


def test_absent_keys_leave_the_run_in_the_legacy_state():
    cfg = th.bw_cfg("x", budgets=(3, 3, 3))
    run = rms.MSRun(copy.deepcopy(cfg), os.path.join("x", "_not_written"))
    assert run.retry_cfg is None and run.terminal_clamp == "density" and run.last_check is None


# ---------------------------------------------------------------------------------- the trigger
@pytest.mark.parametrize("check,fires,reason", [
    (_check(0.005), False, None), (_check(0.0050000001), True, "delta_gt"), (_check(0.0049999), False, None),
    (_check(float("nan")), True, "non_finite"), (_check(float("inf")), True, "delta_gt"), (_check(0.001, s=0.0), True, "s_le_0"),
    (_check(0.001, s=-3.0), True, "s_le_0"), (_check(0.001, s=float("nan")), True, "non_finite"),
    (_check(0.001, C=float("nan")), True, "non_finite"), (_check(0.001, C=float("inf")), True, "non_finite"),
    (_check(0.001, valid=False), False, None), (_check(0.001, R=99.0, R_tail=9.0), False, None)])
def test_the_trigger_fires_exactly_on_the_specified_condition(check, fires, reason):
    got, reasons = ms_retry.trigger(check, 0.005, 1600)
    assert got is fires and (reason in reasons if reason else reasons == [])


def test_the_trigger_needs_the_stage3_check_of_the_last_local_update():
    for bad in (None, {**_check(), "stage": 2}, {**_check(), "local": 1575}):
        with pytest.raises(RuntimeError):
            ms_retry.trigger(bad, 0.005, 1600)


def test_the_trigger_reads_no_closed_form_quantity():
    src = inspect.getsource(ms_retry.trigger)
    body = src.split('"""')[2]                                      # the code after the docstring
    for banned in ("rmse", "tail_mean", "peak", "Gmax", "theory", "g2_two_stage", "e_star", "eT_", '"valid"', '"R"', '"R_tail"'):
        assert banned not in body, banned
    assert all(k in body for k in ('"Delta"', '"s"', '"C"'))


# ---------------------------------------------------------------------------------- re-initialisation and derived streams
def _agent(seed: int):
    gen = torch.Generator().manual_seed(seed)
    return CurriculumPPOv2(PPOConfig(), gen, np.random.default_rng(seed))


def test_in_place_reinit_reproduces_the_constructor_exactly():
    a = _agent(123)
    b = _agent(999)
    assert not torch.equal(a.actor.l1.weight, b.actor.l1.weight)
    b.actor.out.bias.data += 1.0
    ms_retry.reinit_networks(b, torch.Generator().manual_seed(123))
    for net_a, net_b in ((a.actor, b.actor), (a.critic, b.critic)):
        sa, sb = net_a.state_dict(), net_b.state_dict()
        assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_the_derived_seed_and_streams():
    ns = {"init": 7, "env_noise": 11, "learner_action": 12, "opponent_action": 13, "starts_roles": 14, "minibatch": 15}
    gen, rngs = ms_retry.derived_streams(60518, 60.0, ns)
    want = np.random.SeedSequence([60518, 60, 7, 2]).generate_state(1)[0]
    assert torch.equal(gen.get_state(), torch.Generator().manual_seed(int(want)).get_state())
    for key, name in ms_retry.STREAM_NAMES.items():
        ref = np.random.default_rng(np.random.SeedSequence([60518, 60, ns[name], 2]))
        first = np.random.default_rng(np.random.SeedSequence([60518, 60, ns[name]]))
        x = rngs[key].random(4)
        assert np.array_equal(x, ref.random(4)) and not np.array_equal(x, first.random(4))


# ---------------------------------------------------------------------------------- the restart on a reduced run
@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The reduced legacy ``bw`` run (session cache), a reduced run whose trigger fires, one whose trigger does not, and one with the
    censored key whose phase has no clipped draw."""
    from run import run_ms_stagewise as rms_
    base_dir, base_cfg = th.reduced_bw_run()
    out = {"bw": base_dir}
    for name, add in (("fired", {"stage3_retry": {"delta_gt": 1e-9, "max_retries": 1}}),
                      ("nofire", {"stage3_retry": {"delta_gt": 1e9, "max_retries": 1}}),
                      ("censored", {"terminal_clamp_likelihood": "censored"})):
        d = str(tmp_path_factory.mktemp(name))
        cfg = {**th.bw_cfg(d), **add}
        assert rms_.run_pipeline(cfg, d, "pytest") == 0, name
        out[name] = d
    return out


def test_a_fired_run_keeps_attempt1_equal_to_bw_and_freezes_attempt2(runs):
    d, bw = Path(runs["fired"]), Path(runs["bw"])
    assert (d / "retry.json").is_file() and (d / "attempt1").is_dir()
    pair = tp.compare_pair(bw, d)
    assert pair["fired"] and pair["ok"], pair                           # attempt 1 equals the bw stage 3, bit for bit
    rj = json.load(open(d / "retry.json"))
    assert rj["fired"] and rj["frozen_attempt"] == 2 and rj["attempt1"]["reasons"] == ["delta_gt"]
    # the frozen path has the bw layout: 40 / 40 / 20 updates, the stage-3 exports and checks are attempt 2's
    import csv
    rows = list(csv.DictReader(open(d / "ms_updates.csv")))
    assert [sum(1 for r in rows if r["stage"] == s) for s in ("3", "2", "1")] == [40, 40, 20] and len(rows) == 100
    assert int(rows[0]["update"]) == 1 and int(rows[39]["update"]) == 40 and int(rows[40]["update"]) == 41
    assert len(list((d / "weights").glob("u*.npz"))) == len(list((Path(runs["bw"]) / "weights").glob("u*.npz")))
    a1 = np.load(d / "attempt1" / "weights" / "u00025.npz")
    a2 = np.load(d / "weights" / "u00025.npz")
    assert any(not np.array_equal(a1[k], a2[k]) for k in a1.files)       # a different initialisation and different streams
    assert rj["totals"]["updates"] == 140 and rj["totals"]["transitions"] == 140 * 48 and rj["frozen_path_counts"]["updates"] == 100
    assert rj["attempt1"]["counts"]["updates"] == 40
    assert tp.file_set(d) == tp.file_set(bw)                                  # the same files as the bw run (plus attempt1/, retry.json)
    gates = json.load(open(d / "gates.json"))
    assert gates["global_rng"]["status"] == "ok" and gates["budget_totals"]["total_updates"] == 100


def test_the_attempt2_init_digest_is_the_planned_one(runs):
    d = Path(runs["fired"])
    cfg = json.load(open(d / "run_config.json"))
    run = rms.MSRun(cfg, str(d / "_not_written"))
    planned = ms_retry.attempt2_init_digest(run)
    rj = json.load(open(d / "retry.json"))
    man = json.load(open(d / "manifest.json"))
    assert rj["attempt2"]["init_state_sha256"] == planned
    assert man["init_state_sha256"] == run.init_state_sha256 != planned          # attempt 1 keeps the bw digest


def test_a_run_whose_trigger_does_not_fire_is_the_bw_run(runs):
    d, bw = Path(runs["nofire"]), Path(runs["bw"])
    assert not (d / "retry.json").exists() and not (d / "attempt1").exists()
    pair = tp.compare_pair(bw, d)
    assert pair["fired"] is False and pair["ok"], pair                     # same files, same bits (the wall-clock columns aside)


def test_the_censored_key_without_a_clipped_draw_is_the_bw_run(runs):
    pair = tp.compare_pair(Path(runs["bw"]), Path(runs["censored"]))
    assert pair["ok"], pair


def test_the_restart_draws_nothing_from_the_process_global_generators_and_overwrites_nothing(tmp_path):
    cfg = {**th.bw_cfg(str(tmp_path), budgets=(3, 3, 3)), "stage3_retry": {"delta_gt": 1e-9, "max_retries": 1}}
    out = str(tmp_path / "run")
    os.makedirs(out)
    rms.seed_globals(cfg["seed"])
    run = rms.MSRun(cfg, out)
    rec = run.run_stage(3)
    before = rms.digests(rms.global_states())
    opt_ids = (id(run.agent.opt_actor), id(run.agent.opt_critic))
    rec1 = ms_retry.restart_stage3(run, rec, ["delta_gt"])
    assert rms.digests(rms.global_states()) == before                          # no global RNG moved
    assert (id(run.agent.opt_actor), id(run.agent.opt_critic)) == opt_ids == (run.opt_ids["actor"], run.opt_ids["critic"])
    assert len(run.agent.opt_actor.state) == 0 and len(run.agent.opt_critic.state) == 0
    assert (run.global_u, run.total_episodes, run.total_transitions, run.history, run.rows) == (0, 0, 0, [], [])
    assert run.snapshot_log == [{"update": 0, "reason": "init"}] and run.agent.snapshot_refreshes == 1 and run.last_check is None
    assert rec1["counts"]["updates"] == 3 and rec1["trigger_check"]["local"] == 3
    adir = Path(out) / "attempt1"
    assert (adir / "ms_updates_stage3.csv").is_file() and (adir / "train_history_stage3.json").is_file()
    assert (adir / "state_attempt1_end.pt").is_file() and (adir / "ms_checks_stage3.csv").is_file() and (adir / "ms_binmaps_stage3.npz").is_file()
    assert not (Path(out) / "ms_checks_stage3.csv").exists()
    digest = hashlib.sha256((adir / "ms_checks_stage3.csv").read_bytes()).hexdigest()
    with pytest.raises(RuntimeError):
        ms_retry.restart_stage3(run, rec, ["delta_gt"])
    assert hashlib.sha256((adir / "ms_checks_stage3.csv").read_bytes()).hexdigest() == digest
    # attempt 2 starts from freshly initialised networks and runs
    rec2 = run.run_stage(3)
    assert run.global_u == 3 and rec2["total_updates"] == 3 and run.last_check["local"] == 3


# ---------------------------------------------------------------------------------- candidate B in the stage-3 phase
def _stage3_with_clipped_draws(tmp_path, monkeypatch, censored: bool):
    cfg = th.bw_cfg(str(tmp_path), budgets=(3, 3, 3))
    if censored:
        cfg["terminal_clamp_likelihood"] = "censored"
    out = str(tmp_path / ("c" if censored else "d"))
    os.makedirs(out)
    rms.seed_globals(cfg["seed"])
    run = rms.MSRun(cfg, out)
    batches, calls = [], []
    orig_collect = rms.collect_batch_v2

    def collect(*a, **k):
        b = orig_collect(*a, **k)
        batches.append((b, k.get("clamp_likelihood")))
        return b
    monkeypatch.setattr(rms, "collect_batch_v2", collect)
    orig_update = run.agent.update

    def update(*a, **k):
        calls.append(k)
        return orig_update(*a, **k)
    run.agent.update = update

    def hook(t):                                         # a tail policy: Beta mean 1.2e-4, alpha about 0.012, most draws clipped
        with torch.no_grad():
            run.agent.actor.out.bias[0] = -9.0
    run.phase_start_hook = hook
    run.run_stage(3)
    return run, batches, calls


def test_censored_rows_carry_the_beta_tail_mass_and_unflagged_rows_the_density(tmp_path, monkeypatch):
    run, batches, calls = _stage3_with_clipped_draws(tmp_path, monkeypatch, censored=True)
    assert len(batches) == 3 and len(calls) == 3
    c = run.ppo_cfg.action_clamp
    n_flag = n_free = 0
    for (b, lik), kw in zip(batches, calls):
        assert lik == "censored" and np.array_equal(kw["clamp_side"], b["clamp_side"])
        side = b["clamp_side"]
        a = torch.as_tensor(b["d1_buf"]["alpha"])
        be = torch.as_tensor(b["d1_buf"]["beta"])
        act = torch.as_tensor(b["actions"])
        dens = torch.distributions.Beta(a, be).log_prob(act).numpy()
        lo = side < 0
        assert not (side > 0).any()
        want = dens.copy()
        want[lo] = log_betainc_small_x(c, a[torch.as_tensor(lo)], be[torch.as_tensor(lo)]).to(torch.float32).numpy()
        assert np.array_equal(b["logp"], want)                              # flagged: the censored log-mass; unflagged: the density
        assert np.all(b["logp"][lo] <= 0.0) and not np.array_equal(b["logp"][lo], dens[lo])
        n_flag += int(lo.sum())
        n_free += int((~lo).sum())
    assert n_flag > 100 and n_free > 10


def test_the_density_path_is_used_without_the_key(tmp_path, monkeypatch):
    run, batches, calls = _stage3_with_clipped_draws(tmp_path, monkeypatch, censored=False)
    for (b, lik), kw in zip(batches, calls):
        assert lik == "density" and kw == {}
        a, be = torch.as_tensor(b["d1_buf"]["alpha"]), torch.as_tensor(b["d1_buf"]["beta"])
        assert np.array_equal(b["logp"], torch.distributions.Beta(a, be).log_prob(torch.as_tensor(b["actions"])).numpy())
        assert (b["clamp_side"] < 0).any()


# ---------------------------------------------------------------------------------- the post-launch checks on reduced runs
def test_check_run_on_reduced_runs(runs):
    """The deterministic parts of ``t3r3_checks.check_run`` on the reduced fired / not-fired runs (the budgets are those of the
    production runs and are not met by reduced runs, so ``budget_ok`` is not asserted)."""
    import t3r3_checks as ck
    import t3r3_launch as lt
    from t3_launch import relpath_sha256
    lock = {"reference_config_sha256": {}}
    for name, arm, fired in (("fired", "bwA", True), ("nofire", "bwA", False)):
        d = Path(runs[name])
        cfg = json.load(open(d / "run_config.json"))
        man = json.load(open(d / "manifest.json"))
        job = {"wave": "t3r3_dev", "arm": arm, "q": 50, "seed": 50500, "run": cfg["run"], "run_dir": str(d), "kind": "bw", "run_kind": "dev"}
        plan = {"run_config_sha256_relpath": relpath_sha256(cfg, Path(d).parents[2]), "init_state_sha256": man["init_state_sha256"],
                "attempt2_init_state_sha256": json.load(open(d / "retry.json"))["attempt2"]["init_state_sha256"] if fired else ""}
        # the reference for the pair is the reduced bw run: give the checker its directory through a fake root
        import shutil
        root = Path(runs["bw"]).parent / ("root_" + name)
        ref = root / "dev" / "q50" / "seed50500" / "bw"
        if not ref.exists():
            ref.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(runs["bw"], ref)
        res = ck.check_run((job, plan, None, lock, str(root), "/nonexistent"))
        assert res["exit_ok"] and res["fired"] is fired and res["files_complete"], res
        assert res["trigger_consistent"] and res["global_rng_ok"] and res["config_equals_plan"] and res["init_equals_plan"], res
        assert res["replay_ok"] and res["pair_ok"] and res["pair_binding"], res
        assert res["would_fire"] is fired and res["lr_ok"]
        if fired:
            assert res["total_updates"] == 140
    agg = ck.checks_ok({"verdicts": {"exit_ok": {"fail": 0, "not_evaluated": 0}}, "infra_events_present": True, "n_runs": 122,
                        "cross_arm_init": {"ok": True}})
    assert agg and not ck.checks_ok({"verdicts": {"exit_ok": {"fail": 1, "not_evaluated": 0}}, "infra_events_present": True, "n_runs": 122,
                                     "cross_arm_init": {"ok": True}})
    assert not ck.checks_ok({"verdicts": {"exit_ok": {"fail": 0, "not_evaluated": 1}}, "infra_events_present": True, "n_runs": 122,
                             "cross_arm_init": {"ok": True}})
