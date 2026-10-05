"""Tests that pin behaviours of the R2b code which the first 53 tests of ``test_v2_r2b.py`` left open.

They come from the independent mutation-testing pass of the P1 review: each one kills the mutants named in its
header comment (M## = the mutation id of the review; the mutants are one-line edits of the R2b code, e.g. the runner
handing the wrong likelihood to the rollout, or a phase loop drawing bin-balanced starts although ``start_weights``
asks for peak-focused ones). All tests pass on the committed code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import betainc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_v2_r2b as T  # noqa: E402
from test_v2_r2b import (_ClampingRng, _agent, _agent_with_adam, _buffer, _fake_launched_waves,  # noqa: E402
                         _p_config, _run_p, p_parent, spec_for, tree_equal)
from agents.ppo_curriculum import CurriculumPPO  # noqa: E402
from agents.ppo_curriculum_v2 import CurriculumPPOv2, masked_actor_loss  # noqa: E402
from agents.ppo_pathwise import effort_mean, foc_residual, pathwise_update  # noqa: E402
from envs.curriculum_env import StartSampler  # noqa: E402
from rng_alignment import base_config  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from run.run_v2_stagewise import Run, execute, validate_config  # noqa: E402
from run.v2_rollout import collect_batch_v2  # noqa: E402
from utils.beta_tail import log_betainc_small_x  # noqa: E402

PEAK = {"scheme": "peak_focused", "peak_half_width": 20, "peak_share": 0.5}


# ---------------------------------------------------------------- M03: series accuracy over the documented domain
@pytest.mark.parametrize("x, b", [(1e-6, 5.0e4), (1e-4, 500.0), (1e-3, 50.0), (1e-2, 5.0)])
def test_series_matches_scipy_up_to_the_domain_edge(x, b):
    """x * b = 0.05 at the edge: the 1e-6 claim of the docstring, not only at x * b <= 1e-3."""
    a = np.array([1e-4, 1e-2, 0.5, 3.0, 10.0])
    bb = np.full_like(a, b)
    ref = betainc(a, bb, x)
    assert np.all(ref > 0)
    got = np.exp(log_betainc_small_x(x, torch.tensor(a), torch.tensor(bb)).numpy())
    assert np.max(np.abs(got / ref - 1.0)) <= 1e-6


# ---------------------------------------------------------------- M15: documented stream order of peak_focused
def test_peak_focused_replays_the_bin_draw_then_the_position_draw():
    sp = StartSampler(spec_for(50), 10.0)
    n, s = 257, 0.25
    got = sp.peak_focused(2, n, np.random.default_rng(21), 20.0, s)
    r = np.random.default_rng(21)
    ub, up = r.random(n), r.random(n)
    edges = sp.bin_edges(2)
    b = np.searchsorted(np.cumsum(sp.peak_bin_probs(2, 20.0, s)), ub, side="right")
    assert np.array_equal(got, edges[b] + up * (edges[b + 1] - edges[b]))


# ---------------------------------------------------------------- M51: "intersects (-h, h)" off the bin grid
def test_peak_set_is_the_set_of_bins_that_intersect_the_window_off_the_grid():
    sp = StartSampler(spec_for(50), 10.0)
    for h, want in ((15.0, [-20.0, -10.0, 0.0, 10.0]),
                    (25.0, [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0])):
        m = sp.peak_set(2, h)
        assert list(sp.bin_edges(2)[:-1][m]) == want, h


# ---------------------------------------------------------------- M49: share guard of the sampler itself
def test_peak_bin_probs_refuses_a_share_outside_the_open_interval():
    sp = StartSampler(spec_for(50), 10.0)
    for s in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="share"):
            sp.peak_bin_probs(2, 20.0, s)


# ---------------------------------------------------------------- M18b: size of every minibatch
def test_pathwise_update_minibatch_sizes(monkeypatch):
    import agents.ppo_pathwise as pw
    spec, a = _agent_with_adam()
    sizes = []

    def rec(agent, spec_, rows, stage, max_grad_norm=0.5):
        sizes.append(len(rows))
        return {"loss": 0.0, "grad_norm_pre_clip": 0.0}

    monkeypatch.setattr(pw, "pathwise_step", rec)
    d = np.random.default_rng(4).uniform(-200, 200, 512)
    pathwise_update(a, spec, d, 2, epochs=2, minibatch=256)
    assert sizes == [256, 256, 256, 256]
    sizes.clear()
    pathwise_update(a, spec, d[:500], 2, epochs=1, minibatch=256)
    assert sizes == [256, 244]


# ---------------------------------------------------------------- M26/M27/M54: reduction semantics
def test_pathwise_update_reduces_loss_and_norm_over_the_steps(monkeypatch):
    import agents.ppo_pathwise as pw
    spec, a = _agent_with_adam()
    ls, gs = iter([1.0, 2.0, 6.0, 7.0]), iter([0.1, 0.9, 0.2, 0.4])
    monkeypatch.setattr(pw, "pathwise_step", lambda *args, **kw: {"loss": next(ls), "grad_norm_pre_clip": next(gs)})
    out = pathwise_update(a, spec, np.random.default_rng(4).uniform(-200, 200, 512), 2, epochs=2, minibatch=256)
    assert out["n_steps"] == 4 and out["loss"] == pytest.approx(4.0)       # mean (last would be 7, first 1)
    assert out["grad_norm_pre_clip"] == pytest.approx(0.4) and out["grad_norm_pre_clip_max"] == 0.9


# ---------------------------------------------------------------- M20b/M28: foc and e0 are those of the FINAL weights
def test_pathwise_update_foc_and_e0_are_those_of_the_final_weights():
    spec, a = _agent_with_adam()
    d = np.random.default_rng(4).uniform(-200, 200, 512)
    out = pathwise_update(a, spec, d, 2, epochs=3, minibatch=128)
    with torch.no_grad():
        e = effort_mean(a.actor, torch.as_tensor(spec.encode_obs(2, d)), spec)
        e_opp = effort_mean(a.opponent, torch.as_tensor(spec.encode_obs(2, -d)), spec)
        foc = foc_residual(spec, torch.as_tensor(d, dtype=torch.float64), e, e_opp).abs()
        e0 = effort_mean(a.actor, torch.as_tensor(spec.encode_obs(2, np.zeros(1))), spec)[0]
    assert out["foc_abs_mean"] == pytest.approx(float(foc.mean()), rel=1e-12)
    assert out["foc_abs_max"] == pytest.approx(float(foc.max()), rel=1e-12)
    assert out["e0"] == pytest.approx(float(e0), rel=1e-12)


# ---------------------------------------------------------------- M48: guard message (the test of the suite passes without it)
def test_pathwise_update_budget_guard_has_its_own_message():
    spec, a = _agent_with_adam()
    for e, m in ((0, 4), (2, 0)):
        with pytest.raises(ValueError, match="must be positive"):
            pathwise_update(a, spec, np.zeros(8), 2, epochs=e, minibatch=m)


# ---------------------------------------------------------------- M21: mixed epochs / minibatch settings
@pytest.mark.parametrize("epochs, minibatch, steps", [(3, None, 3), (1, 256, 2)])
def test_phase_p_mixed_budget_settings_are_not_the_r1_single_step(p_parent, tmp_path, epochs, minibatch, steps):
    run = _run_p(_p_config(epochs, minibatch, cap=2), p_parent, str(tmp_path / "p"))
    assert all(h["n_steps"] == steps and h["n_minibatch_steps"] == steps for h in run.history)


# ---------------------------------------------------------------- M29: gradient of a flagged row reaches the actor
def test_flagged_row_gradient_reaches_the_actor():
    spec, agent = spec_for(50), _agent()
    st = spec.encode_obs(2, np.array([-170.0, 20.0]))
    a, b = agent.beta_params(st)
    c = agent.cfg.action_clamp
    side = np.array([-1, 0], dtype=np.int8)
    act = np.array([c, 0.4], dtype=np.float32)
    olp = agent.log_prob(a, b, act, clamp_side=side)
    t_st, t_ac, t_olp = (torch.as_tensor(x) for x in (st, act, olp))
    # row 0 (flagged) has advantage 1, row 1 advantage 0: every gradient comes from the censored log-mass
    loss, _, _ = masked_actor_loss(agent.actor, t_st, t_ac, t_olp, torch.tensor([1.0, 0.0]), torch.arange(2), 0.2,
                                   torch.as_tensor(side), c)
    loss.backward()
    g = torch.cat([p.grad.flatten() for p in agent.actor.parameters() if p.grad is not None])
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0


# ---------------------------------------------------------------- M32: KL of flagged rows under a policy mask
def test_masked_update_with_flagged_rows_has_zero_kl_at_lr0():
    spec, agent = spec_for(50), _agent(lr=0.0)
    st, act, _, ret, adv = _buffer(agent, spec)
    side = np.zeros(512, dtype=np.int8)
    side[[200, 300]] = -1
    side[400] = 1
    act = act.copy()
    c = agent.cfg.action_clamp
    act[[200, 300]] = c
    act[400] = 1.0 - c
    alpha, beta = agent.beta_params(st)
    olp = agent.log_prob(alpha, beta, act, clamp_side=side)
    pm = np.ones(512, dtype=bool)
    pm[:150] = False                                   # the stage-2-like rows are masked out of the policy loss
    d = agent.update(st, act, olp, ret, adv, policy_mask=pm, clamp_side=side)
    assert d["n_policy_rows"] == 362 and d["n_censored_rows"] == 3
    assert d["kl_epochs"] == [0.0] * len(d["kl_epochs"]) and d["clip_frac"] == 0.0


# ---------------------------------------------------------------- M33/M34: documented modes accept the keys
def _cfg_for_mode(mode):
    cfg = base_config()
    cfg["mode"] = mode
    if mode in ("phase_B", "phase_A_continue", "phase_P"):
        cfg["parent_checkpoint"], cfg["parent_sha256"] = "x", "y"
    if mode == "phase_B":
        cfg["flags"].update(stage2_update_mode="frozen", adv_norm_scope="stage1_rows", continuation_action_mode="mean")
    if mode == "phase_P":
        cfg["lr_decay"] = [{"phase": "P", "start_lr": 3e-5, "end_lr": 3e-5, "local_first": 1, "local_last": 6}]
    return cfg


@pytest.mark.parametrize("mode", ["phase_A", "phase_A_continue", "phase_B"])
def test_censored_is_accepted_in_every_documented_mode(mode):
    cfg = _cfg_for_mode(mode)
    cfg["clamp_likelihood"] = "censored"
    validate_config(cfg)


@pytest.mark.parametrize("mode", ["phase_A", "phase_A_continue", "phase_P"])
def test_peak_focused_is_accepted_in_every_documented_mode(mode):
    cfg = _cfg_for_mode(mode)
    cfg["start_weights"] = dict(PEAK)
    validate_config(cfg)


# ---------------------------------------------------------------- M52: rollout validation
def test_rollout_refuses_an_unknown_clamp_likelihood(tmp_path):
    r = T.make_run("phase_A", {"reward_mode": "expected"}, str(tmp_path / "x"))
    n = 4
    with pytest.raises(ValueError, match="clamp_likelihood"):
        collect_batch_v2(r.spec, r.agent, np.full(n, r.spec.T), np.zeros(n), np.zeros(n, dtype=int), r.rngs["env"],
                         r.rngs["learn"], r.rngs["opp"], 1.0, 1.0, 10.0, reward_mode="expected",
                         clamp_likelihood="bogus")


# ---------------------------------------------------------------- M31: density default with clamped rows == original update
def test_density_default_with_clamped_rows_is_exactly_the_original_update(monkeypatch, tmp_path):
    """Clamped learner rows are common in default runs (R1 parents_A: 12044 of 32000 updates have one). In density mode
    the runner must keep dispatching to the original update; forcing the original update gives the same weights."""
    def run_one(tag, force_base):
        cfg = base_config()
        cfg["flags"]["reward_mode"] = "expected"
        d = str(tmp_path / tag)
        os.makedirs(d)
        run = Run(cfg, d)
        run.rngs["learn"] = _ClampingRng(run.rngs["learn"], 2, 1)
        with monkeypatch.context() as m:
            if force_base:
                m.setattr(CurriculumPPOv2, "update",
                          lambda self, s, a, o, r_, adv, policy_mask=None, norm_mask=None, clamp_side=None:
                          CurriculumPPO.update(self, s, a, o, r_, adv))
            assert execute(run, cfg, d, "pytest") == 0
        return run

    a, b = run_one("a", False), run_one("b", True)
    assert all(r["d1_L_s2_lo"] >= 2 and r["d1_L_s2_hi"] >= 1 for r in a.v2_history)       # the clamps did occur
    assert tree_equal(a.agent.actor.state_dict(), b.agent.actor.state_dict())


# ---------------------------------------------------------------- M35/M36: both phase loops draw through Run.draw_starts
def test_phase_loops_draw_their_starts_through_run_draw_starts(monkeypatch, p_parent, tmp_path):
    calls = []
    orig = Run.draw_starts

    def rec(self, stage, n, local=1):
        calls.append((stage, n))
        return orig(self, stage, n, local)

    monkeypatch.setattr(Run, "draw_starts", rec)
    cfg = base_config()
    cfg["start_weights"] = dict(PEAK)
    cfg["flags"]["reward_mode"] = "expected"
    d = str(tmp_path / "a")
    os.makedirs(d)
    run = Run(cfg, d)
    assert execute(run, cfg, d, "pytest") == 0
    assert len(calls) == len(run.history) > 0
    calls.clear()
    cfg_p = _p_config(cap=3)
    cfg_p["start_weights"] = dict(PEAK)
    run_p = _run_p(cfg_p, p_parent, str(tmp_path / "p"))
    assert len(calls) == len(run_p.history) == 3 and all(c[0] == run_p.spec.T for c in calls)


# ---------------------------------------------------------------- M37: launch checks flag post-code-commit changes
def test_launch_checks_flag_files_changed_since_the_code_commit(tmp_path, monkeypatch):
    K, root, r1, parents = _fake_launched_waves(tmp_path)
    monkeypatch.setattr(K, "sha256_file", lambda p: "abc")
    monkeypatch.setattr(K, "commit_touches_only_allowed", lambda code, c: ["agents/ppo_pathwise.py"])
    r = K.check_run("r2b_waveA", "A_peak25", 50, 10501, root, r1, parents, "HEAD", {})
    assert not r["ok"] and any("outside results" in s for s in r["problems"])


# ---------------------------------------------------------------- L1/L2/L3: remaining launch-check branches
def test_launch_checks_missing_dirty_flag_parent_path_and_parent_file_hash(tmp_path, monkeypatch):
    import json
    K, root, r1, parents = _fake_launched_waves(tmp_path)
    monkeypatch.setattr(K, "sha256_file", lambda p: "abc")

    def run(wave, arm):
        return K.check_run(wave, arm, 50, 10501, root, r1, parents, "HEAD", {})

    def edit(wave, arm, fn):
        p = root / ("waveA" if wave == "r2b_waveA" else "waveP") / "q50" / "seed10501" / arm / "manifest.json"
        m = json.load(open(p))
        fn(m)
        json.dump(m, open(p, "w"))

    edit("r2b_waveA", "A_peak25", lambda m: m["git"].pop("dirty"))                     # L1: dirty absent is not clean
    assert any("dirty" in s for s in run("r2b_waveA", "A_peak25")["problems"])
    edit("r2b_waveP", "P20_lr3e-5", lambda m: m.update(parent_checkpoint="/elsewhere/state_end_A.pt"))   # L2
    assert any("parent_checkpoint" in s for s in run("r2b_waveP", "P20_lr3e-5")["problems"])
    monkeypatch.setattr(K, "sha256_file", lambda p: "something else")                                  # L3
    assert any("parent file" in s for s in run("r2b_waveP", "P20_lr3e-4")["problems"])


# ---------------------------------------------------------------- M55/M56: the runner hands the configured likelihood to the rollout
@pytest.mark.parametrize("mode", ["density", "censored"])
def test_run_stores_the_configured_log_prob_for_clamped_rows(monkeypatch, tmp_path, mode):
    import run.run_v2_stagewise as RS
    seen = []
    orig = RS.collect_batch_v2

    def wrap(*args, **kw):
        b = orig(*args, **kw)
        a, bb = b["d1_buf"]["alpha"], b["d1_buf"]["beta"]
        agent = args[1]
        want = agent.log_prob(a, bb, b["actions"], clamp_side=b["clamp_side"] if mode == "censored" else None)
        seen.append((int((b["clamp_side"] != 0).sum()), bool(np.array_equal(b["logp"], want.astype(np.float32)))))
        return b

    monkeypatch.setattr(RS, "collect_batch_v2", wrap)
    cfg = base_config()
    cfg["flags"]["reward_mode"] = "expected"
    cfg["clamp_likelihood"] = mode
    d = str(tmp_path / mode)
    os.makedirs(d)
    run = Run(cfg, d)
    run.rngs["learn"] = _ClampingRng(run.rngs["learn"], 2, 1)
    assert execute(run, cfg, d, "pytest") == 0
    assert seen and all(n == 3 and ok for n, ok in seen), seen


# ---------------------------------------------------------------- M57: censored through the MASKED update branch (phase B)
def test_censored_phase_b_run_end_to_end(p_parent, tmp_path):
    cfg = base_config()
    cfg["mode"] = "phase_B"
    cfg["flags"].update(reward_mode="expected", stage2_update_mode="frozen", adv_norm_scope="stage1_rows",
                        continuation_action_mode="mean")
    cfg["clamp_likelihood"] = "censored"
    cfg["parent_checkpoint"], cfg["parent_sha256"] = p_parent, "in-process"
    cfg["budget_overrides"]["phase_caps"]["B"] = 3
    d = str(tmp_path / "b")
    os.makedirs(d)
    run = Run(cfg, d)
    run.rngs["learn"] = _ClampingRng(run.rngs["learn"], 2, 1)
    assert execute(run, cfg, d, "pytest") == 0
    rows = [r for r in run.v2_history if r.get("phase") == "B"]
    assert rows and all(r["n_clamped_rows_learner"] == 3 and r["n_censored_rows"] == 3 for r in rows)


# ---------------------------------------------------------------- added after the conformance audit
@pytest.mark.parametrize("q", [50, 60])
def test_bin_balanced_equals_an_inline_copy_of_the_original_balanced(q):
    """The original ``balanced``: a bin by ``rng.integers``, then the position by ``rng.random`` (no other draw). The
    sampler is compared with this inline copy (not with itself), values and final stream state."""
    sp = StartSampler(spec_for(q), 10.0)
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    got = sp.balanced(2, 257, r1)
    e = sp.bin_edges(2)
    b = r2.integers(0, e.size - 1, size=257)
    u = r2.random(257)
    assert np.array_equal(got, e[b] + u * (e[b + 1] - e[b])) and r1.bit_generator.state == r2.bit_generator.state


def test_masked_update_with_all_rows_and_no_flags_equals_the_original_update():
    """A_censored's single-change property: a buffer without flagged rows takes the original update, and the masked
    branch with every row in the policy loss and no flags gives the same weights, optimiser state and stream."""
    spec = spec_for(50)
    a1, a2 = _agent(), _agent()
    buf = _buffer(a1, spec)
    d1 = a1.update(*buf)
    d2 = a2.update(*buf, policy_mask=np.ones(512, dtype=bool))
    assert tree_equal(T._full(a1), T._full(a2))
    assert d1["policy_loss"] == d2["policy_loss"] and d1["kl_epochs"] == d2["kl_epochs"]
