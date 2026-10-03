"""R1 refinement tests: new flags, schedules, target-KL, annealing, D1 instrumentation, defaults.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_v2_refine.py -q
Module-level tests of the continuation table, the pathwise step, the launcher and the D2 tool live in
tests/test_v2_refine_{continuation,pathwise,tools,d2}.py.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

from agents.ppo_curriculum import BetaActor, mean_effort_numpy  # noqa: E402
from compare_runs import _walk  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from run.run_v2_stagewise import Run, execute, validate_config  # noqa: E402
from run.v2_rollout import collect_batch_v2  # noqa: E402
from rng_alignment import B_ARMS, base_config, make_run  # noqa: E402

C7_CONFIG = ROOT / "results/v2_pilots/phase2_regression/run_config.json"
C7_BEFORE = ROOT / "results/v2_pilots/phase1_regression/before/FINAL_A400_B25_C25/tel_q50_s10501"
C7_OLD_V2 = ROOT / "results/v2_pilots/phase2_regression/v2_full_431474d"
WALL = {"time_sec", "update_wall_sec", "elapsed_phase_wall_sec", "verifier_sec"}
B2_MEAN = {"stage2_update_mode": "frozen", "adv_norm_scope": "stage1_rows", "continuation_action_mode": "mean"}


def _read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def _eq(a, b, path=""):
    """Recursive exact equality (numpy / torch aware); returns the differing paths."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in WALL:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k} missing")
            else:
                out += _eq(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path} len"]
        for i, (x, y) in enumerate(zip(a, b)):
            out += _eq(x, y, f"{path}[{i}]")
    elif isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)):
            out.append(path)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True):
            out.append(path)
    elif isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        pass
    elif a != b:
        out.append(path)
    return out


@pytest.fixture(scope="module")
def parent(tmp_path_factory) -> str:
    """Throwaway end-of-phase-A full state (q=50, seed 10501, 6 expected-reward A updates)."""
    d = str(tmp_path_factory.mktemp("rparent"))
    cfg = base_config()
    cfg["flags"]["reward_mode"] = "expected"
    run = Run(cfg, d)
    assert execute(run, cfg, d, "pytest") == 0
    return os.path.join(d, "state_end_A.pt")


def _cont_cfg(parent_path, **kw):
    """phase_A_continue config (expected reward, cap 6) with optional R1 keys."""
    cfg = base_config()
    cfg.update(mode="phase_A_continue", parent_checkpoint=parent_path, parent_sha256="in-process")
    cfg["flags"]["reward_mode"] = "expected"
    cfg["budget_overrides"]["phase_caps"]["A"] = 6
    cfg.update(kw)
    return cfg


# ====================================================================== defaults are bit-identical (C7)
def test_defaults_bit_identical_to_c7(tmp_path):
    """Every new key absent: the regression pipeline equals the existing runner's output."""
    cfg = json.load(open(C7_CONFIG))
    d = str(tmp_path / "c7")
    os.makedirs(d, exist_ok=True)
    run = Run(cfg, d)
    assert execute(run, cfg, d, "pytest") == 0
    for f in ("train_history.json", "final_eval.json"):
        diffs, skipped = [], set()
        _walk(json.load(open(C7_BEFORE / f)), json.load(open(os.path.join(d, f))), f, diffs, skipped)
        assert diffs == [], diffs[:3]
    for f in ("arrays.npz", "checkpoint_weights.npz", "phase_A_exit_arrays.npz", "phase_B_exit_arrays.npz"):
        za, zb = np.load(C7_BEFORE / f), np.load(os.path.join(d, f))
        assert set(za.files) == set(zb.files)
        assert all(np.array_equal(za[k], zb[k]) for k in za.files), f
    if (C7_BEFORE / "checkpoint.pt").exists():     # *.pt is gitignored: present in the canonical worktree only
        ca = torch.load(C7_BEFORE / "checkpoint.pt", weights_only=False)
        cb = torch.load(os.path.join(d, "checkpoint.pt"), weights_only=False)
        for k in ("actor", "critic", "opponent", "opt_actor", "opt_critic"):
            assert _eq(ca[k], cb[k]) == [], k
    # v2_updates.csv: the old columns unchanged, only new columns appended
    old, new = _read_csv(C7_OLD_V2 / "v2_updates.csv"), _read_csv(os.path.join(d, "v2_updates.csv"))
    old_cols, new_cols = list(old[0]), list(new[0])
    assert new_cols[:len(old_cols)] == old_cols
    extra = new_cols[len(old_cols):]
    assert extra[:2] == ["n_epochs_run", "conc_scale"] and all(c.startswith("d1_") for c in extra[2:])
    assert len(old) == len(new)
    for ro, rn in zip(old, new):
        assert all(ro[c] == rn[c] for c in old_cols if c not in WALL)
    assert all(float(r["conc_scale"]) == 1.0 and int(r["n_epochs_run"]) == 10 for r in new)
    # no new key leaks into the logged history at default; the default full state has no conc_scale entry
    assert "n_epochs_run" not in run.history[0]
    assert "conc_scale" not in run.agent.full_state()


def test_default_weight_export_has_no_conc_scale(parent, tmp_path):
    r = make_run("phase_B", B_ARMS["B2_frozen_s1norm"], str(tmp_path / "b"), parent)
    r.agent.export_weights_npz(str(tmp_path / "w.npz"))
    assert "conc_scale" not in np.load(tmp_path / "w.npz").files


# ====================================================================== piecewise learning-rate windows
def _lr_run(mode, tmp, windows, cap):
    cfg = base_config()
    cfg.update(mode=mode)
    if mode in ("phase_B", "phase_A_continue"):
        cfg.update(parent_checkpoint="x", parent_sha256="in-process")
    if mode == "phase_B":
        cfg["flags"].update(B2_MEAN)
    cfg["budget_overrides"]["phase_caps"].update({"A": cap, "B": cap})
    cfg["lr_decay"] = windows
    os.makedirs(tmp, exist_ok=True)
    return Run(cfg, str(tmp))


def _lin(s, e, j, j0, j1):
    return s + (e - s) * (j - j0) / (j1 - j0)


def test_piecewise_lr_phase_B_P1_P2(tmp_path):
    p1 = [{"phase": "B", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 1, "local_last": 400},
          {"phase": "B", "start_lr": 3e-5, "end_lr": 3e-6, "local_first": 401, "local_last": 600}]
    r = _lr_run("phase_B", tmp_path / "p1", p1, 600)
    for j in range(1, 601):
        want = _lin(3e-4, 3e-5, j, 1, 400) if j <= 400 else _lin(3e-5, 3e-6, j, 401, 600)
        assert r.lr_for("B", j) == want, j
    assert r.lr_for("B", 1) == 3e-4 and r.lr_for("B", 400) == pytest.approx(3e-5, rel=1e-12)
    assert r.lr_for("B", 401) == 3e-5 and r.lr_for("B", 600) == pytest.approx(3e-6, rel=1e-12)
    p2 = [{"phase": "B", "start_lr": 3e-4, "end_lr": 3e-6, "local_first": 1, "local_last": 600}]
    r2 = _lr_run("phase_B", tmp_path / "p2", p2, 600)
    assert all(r2.lr_for("B", j) == _lin(3e-4, 3e-6, j, 1, 600) for j in range(1, 601))
    p0 = [{"phase": "B", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 1, "local_last": 600}]
    r0 = _lr_run("phase_B", tmp_path / "p0", p0, 600)           # the locked single-window form
    assert all(r0.lr_for("B", j) == _lin(3e-4, 3e-5, j, 1, 600) for j in range(1, 601))


def test_piecewise_lr_phase_A_continuation_P1_P2(tmp_path):
    p1 = [{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 1, "local_last": 250},
          {"phase": "A", "start_lr": 3e-5, "end_lr": 3e-6, "local_first": 251, "local_last": 400}]
    r = _lr_run("phase_A_continue", tmp_path / "p1", p1, 400)
    for j in range(1, 401):
        want = _lin(3e-4, 3e-5, j, 1, 250) if j <= 250 else _lin(3e-5, 3e-6, j, 251, 400)
        assert r.lr_for("A", j) == want, j
    p2 = [{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-6, "local_first": 1, "local_last": 400}]
    r2 = _lr_run("phase_A_continue", tmp_path / "p2", p2, 400)
    assert all(r2.lr_for("A", j) == _lin(3e-4, 3e-6, j, 1, 400) for j in range(1, 401))
    base = [{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 1, "local_last": 400}]
    rb = _lr_run("phase_A_continue", tmp_path / "b", base, 400)
    assert [rb.lr_for("A", j) for j in (1, 200, 400)] == [_lin(3e-4, 3e-5, j, 1, 400) for j in (1, 200, 400)]


def test_constant_window_only_for_continued_phase(tmp_path):
    const = [{"phase": "A", "start_lr": 3e-5, "end_lr": 3e-5, "local_first": 1, "local_last": 200}]
    r = _lr_run("phase_A_continue", tmp_path / "ok", const, 200)
    assert all(r.lr_for("A", j) == 3e-5 for j in range(1, 201))
    with pytest.raises(ConfigError, match="lr_decay.start_lr must equal"):
        _lr_run("phase_A", tmp_path / "bad_A", const, 200)
    const_b = [{"phase": "B", "start_lr": 3e-5, "end_lr": 3e-5, "local_first": 1, "local_last": 200}]
    with pytest.raises(ConfigError, match="lr_decay.start_lr must equal"):
        _lr_run("phase_B", tmp_path / "bad_B", const_b, 200)
    late = [{"phase": "A", "start_lr": 3e-5, "end_lr": 3e-6, "local_first": 101, "local_last": 200}]
    with pytest.raises(ConfigError, match="lr_decay.start_lr must equal"):   # record schedule applies before it
        _lr_run("phase_A_continue", tmp_path / "late", late, 200)


@pytest.mark.parametrize("wins, msg", [
    ([{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 1, "local_last": 100},
      {"phase": "A", "start_lr": 3e-5, "end_lr": 3e-6, "local_first": 102, "local_last": 200}], "contiguous"),
    ([{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 1, "local_last": 100},
      {"phase": "A", "start_lr": 3e-5, "end_lr": 3e-6, "local_first": 100, "local_last": 200}], "contiguous"),
    ([{"phase": "A", "start_lr": 3e-4, "end_lr": 3e-5, "local_first": 1, "local_last": 100},
      {"phase": "A", "start_lr": 3e-5, "end_lr": 3e-6, "local_first": 101, "local_last": 150}], "phase cap"),
])
def test_piecewise_lr_refusals(tmp_path, wins, msg):
    with pytest.raises(ConfigError, match=msg):
        _lr_run("phase_A_continue", tmp_path / "x", wins, 200)


# ====================================================================== batch / minibatch
@pytest.mark.parametrize("mb, steps_a", [(1024, 20), (256, 80)])
def test_batch_steps_phase_A(tmp_path, mb, steps_a):
    cfg = base_config()
    cfg["budget_overrides"].update(episodes_per_update=2048, phase_caps={"A": 1, "B": 1, "C": 1})
    cfg["ppo_overrides"] = {"minibatch": mb, "target_kl": None}
    d = str(tmp_path / "a")
    os.makedirs(d, exist_ok=True)
    run = Run(cfg, d)
    assert run.ppo_cfg.minibatch == mb and int(run.P["episodes_per_update"]) == 2048
    assert execute(run, cfg, d, "pytest") == 0
    assert run.history[0]["n_transitions"] == 2048
    assert run.history[0]["n_minibatch_steps"] == steps_a


@pytest.mark.parametrize("mb, steps_b", [(1024, 40), (256, 160)])
def test_batch_steps_phase_B(parent, tmp_path, mb, steps_b):
    cfg = base_config()
    cfg.update(mode="phase_B", parent_checkpoint=parent, parent_sha256="in-process")
    cfg["flags"].update(B2_MEAN)
    cfg["flags"]["reward_mode"] = "expected"
    cfg["budget_overrides"].update(episodes_per_update=2048)
    cfg["budget_overrides"]["phase_caps"]["B"] = 1
    cfg["ppo_overrides"] = {"minibatch": mb, "target_kl": None}
    d = str(tmp_path / "b")
    os.makedirs(d, exist_ok=True)
    run = Run(cfg, d)
    assert execute(run, cfg, d, "pytest") == 0
    assert run.history[0]["n_transitions"] == 4096
    assert run.history[0]["n_minibatch_steps"] == steps_b


def test_phase_c_episode_check_only_when_phase_c_runs(tmp_path):
    cfg = base_config()
    cfg["budget_overrides"]["episodes_per_update"] = 2048
    Run(cfg, str(tmp_path / "ok"))                       # mode phase_A: phase C is not run -> accepted
    cfg2 = base_config()
    cfg2.update(mode="full", fixed_budget=False)
    cfg2["flags"] = {"reward_mode": "sampled", "stage2_update_mode": "joint", "adv_norm_scope": "all_rows",
                     "continuation_action_mode": "stochastic"}
    cfg2["budget_overrides"]["episodes_per_update"] = 2048
    with pytest.raises(ConfigError, match="phase_c_root"):
        Run(cfg2, str(tmp_path / "bad"))


# ====================================================================== target-KL
def _two_agents(tmp_path, parent_path=None, masked=False):
    """Two identical runs (same seeds, same restored state) and one shared batch."""
    if masked:
        def mk(n):
            return make_run("phase_B", B_ARMS["B2_frozen_s1norm"], str(tmp_path / n), parent_path)
    else:
        def mk(n):
            return make_run("phase_A", {"reward_mode": "expected"}, str(tmp_path / n))
    r1, r2 = mk("r1"), mk("r2")
    spec, P = r1.spec, r1.P
    n = int(P["episodes_per_update"])
    if masked:
        t0, d0 = np.ones(n, dtype=int), np.zeros(n)
    else:
        t0, d0 = np.full(n, spec.T), r1.sampler.balanced(spec.T, n, r1.rngs["start"])
    roles = r1.rngs["start"].integers(0, 2, size=n)
    frozen = r1.agent.frozen if masked else None
    b = collect_batch_v2(spec, r1.agent, t0, d0, roles, r1.rngs["env"], r1.rngs["learn"], r1.rngs["opp"],
                         1.0, 1.0, P["es_bin_width"], reward_mode="sampled", frozen=frozen,
                         continuation_action_mode="mean" if masked else "stochastic")
    return r1, r2, b


@pytest.mark.parametrize("masked", [False, True])
def test_target_kl_stops_after_epoch_and_keeps_stream_position(parent, tmp_path, masked):
    r1, r2, b = _two_agents(tmp_path, parent, masked)
    kw = dict(policy_mask=(b["stage"] == 1), norm_mask=(b["stage"] == 1)) if masked else {}
    args = (b["states"], b["actions"], b["logp"], b["returns"], b["advantages"])
    d_full = r1.agent.update(*args, **kw)                 # baseline: all 10 epochs
    assert "n_epochs_run" not in d_full and len(d_full["kl_epochs"]) == 10
    steps_per_epoch = d_full["n_minibatch_steps"] // 10
    # (a) epoch-1 KL above the target -> exactly one epoch
    r2.agent.target_kl = 0.0
    d_one = r2.agent.update(*args, **kw)
    assert d_one["n_epochs_run"] == 1 and len(d_one["kl_epochs"]) == 1
    assert d_one["n_minibatch_steps"] == steps_per_epoch
    assert d_one["kl_epochs"][0] == d_full["kl_epochs"][0]            # same first epoch, bit for bit
    assert r2.agent.rng_mb.bit_generator.state == r1.agent.rng_mb.bit_generator.state   # rule A6
    # (b) targets crossed at different epochs: stops after the first epoch whose KL exceeds the target
    kl = d_full["kl_epochs"]
    for n_try, quant in enumerate((0.3, 0.6, 0.9)):
        target = float(np.quantile(kl, quant))
        want = next((i for i, x in enumerate(kl) if x > target), 9) + 1
        r3, _, _ = _two_agents(tmp_path / f"again{n_try}", parent, masked)
        r3.agent.target_kl = target
        d_mid = r3.agent.update(*args, **kw)
        assert d_mid["n_epochs_run"] == want and d_mid["kl_epochs"] == kl[:want], (quant, want)
        assert r3.agent.rng_mb.bit_generator.state == r1.agent.rng_mb.bit_generator.state
    # (c) a target never crossed -> identical to the baseline update (weights too)
    r4, _, _ = _two_agents(tmp_path / "third", parent, masked)
    r4.agent.target_kl = 1e9
    d_none = r4.agent.update(*args, **kw)
    assert d_none["n_epochs_run"] == 10 and d_none["kl_epochs"] == kl
    for k, v in r1.agent.actor.state_dict().items():
        assert torch.equal(v, r4.agent.actor.state_dict()[k])


def test_target_kl_logged_per_update_in_csv(parent, tmp_path):
    cfg = _cont_cfg(parent)
    cfg["ppo_overrides"] = {"minibatch": None, "target_kl": 1e-12}
    d = str(tmp_path / "t")
    os.makedirs(d, exist_ok=True)
    run = Run(cfg, d)
    assert run.agent.target_kl == 1e-12
    assert execute(run, cfg, d, "pytest") == 0
    rows = _read_csv(os.path.join(d, "v2_updates.csv"))
    assert [int(r["n_epochs_run"]) for r in rows] == [1] * 6
    assert all(h["n_epochs_run"] == 1 for h in run.history)
    man = json.load(open(os.path.join(d, "manifest.json")))
    assert man["target_kl"] == 1e-12 and man["ppo_overrides"] == {"minibatch": None, "target_kl": 1e-12}


# ====================================================================== concentration annealing
def test_conc_scale_identity_and_scaling():
    gen = torch.Generator().manual_seed(7)
    a = BetaActor(64, 100.0, 1e-6, gen)
    with torch.no_grad():
        a.out.weight.normal_(0, 0.5, generator=gen)
        a.out.bias.normal_(0, 0.3, generator=gen)
    x = torch.rand(500, 2, generator=gen)
    al0, be0 = a(x)
    a.conc_scale = 1.0
    al1, be1 = a(x)
    assert torch.equal(al0, al1) and torch.equal(be0, be1)
    for s in (0.5, 1.37, 2.0, 4.0):
        a.conc_scale = s
        als, bes = a(x)
        np.testing.assert_allclose((als + bes).detach().numpy(), s * (al0 + be0).detach().numpy(), rtol=2e-6)
        np.testing.assert_allclose((als / (als + bes)).detach().numpy(), (al0 / (al0 + be0)).detach().numpy(),
                                   rtol=2e-6)
    # the framework-free reload reproduces the scaled (alpha, beta) of the exported weights
    w = {f"actor.{k}": v.detach().numpy() for k, v in a.state_dict().items()}
    for s in (1.0, 1.37, 2.0, 4.0):
        a.conc_scale = s
        als, bes = a(x)
        _, an, bn = mean_effort_numpy(w, x.numpy(), conc_scale=s)
        np.testing.assert_allclose(an, als.detach().numpy(), rtol=1e-6)
        np.testing.assert_allclose(bn, bes.detach().numpy(), rtol=1e-6)


def test_annealing_run_scale_schedule_state_manifest_export(parent, tmp_path):
    cfg = _cont_cfg(parent)
    cfg["conc_anneal"] = {"phase": "A", "local_first": 1, "local_last": 6, "scale_first": 1.0, "scale_last": 2.0}
    d = str(tmp_path / "an")
    os.makedirs(d, exist_ok=True)
    run = Run(cfg, d)
    assert [run.conc_scale_for(j) for j in (1, 2, 6, 7)] == [1.0, 1.0 + 1.0 / 5, 2.0, 2.0]
    assert execute(run, cfg, d, "pytest") == 0
    rows = _read_csv(os.path.join(d, "v2_updates.csv"))
    assert [float(r["conc_scale"]) for r in rows] == [run.conc_scale_for(j) for j in range(1, 7)]
    assert run.agent.actor.conc_scale == 2.0 and run.agent.opponent.conc_scale == 2.0   # end of phase
    end = torch.load(os.path.join(d, "state_end_A.pt"), weights_only=False)
    assert end["agent"]["conc_scale"]["actor"] == 2.0 and end["agent"]["conc_scale"]["opponent"] == 2.0
    man = json.load(open(os.path.join(d, "manifest.json")))
    assert man["conc_anneal"]["scale_last"] == 2.0 and man["conc_scale_final"]["actor"] == 2.0
    assert float(np.load(os.path.join(d, "checkpoint_weights.npz"))["conc_scale"]) == 2.0
    r2 = make_run("phase_A", {"reward_mode": "expected"}, str(tmp_path / "r2"))
    r2.restore(os.path.join(d, "state_end_A.pt"))
    assert r2.agent.actor.conc_scale == 2.0 and r2.agent.opponent.conc_scale == 2.0
    r3 = make_run("phase_A", {"reward_mode": "expected"}, str(tmp_path / "r3"))
    r3.restore(parent)
    assert r3.agent.actor.conc_scale == 1.0 and r3.agent.opponent.conc_scale == 1.0
    # mean unchanged, concentration doubled: beta_fn of the annealed actor reports the scaled Beta
    a0 = copy.deepcopy(run.agent.actor)
    a0.conc_scale = 1.0
    x = torch.as_tensor(run.spec.encode_obs(2, np.linspace(-100, 100, 41)))
    als, bes = run.agent.actor(x)
    al0, be0 = a0(x)
    np.testing.assert_allclose((als / (als + bes)).detach().numpy(), (al0 / (al0 + be0)).detach().numpy(),
                               rtol=2e-6)
    assert float((als + bes).min()) > 1.99 * float((al0 + be0).min())


def test_annealing_rollout_unscaled_identity_logprob_and_opponent(tmp_path):
    """At scale 1.0 the rollout is bit-identical; at scale s the stored log-prob is the scaled Beta's;
    the opponent carries the scale after a refresh."""
    r_base = make_run("phase_A", {"reward_mode": "expected"}, str(tmp_path / "b"))
    r_s = make_run("phase_A", {"reward_mode": "expected"}, str(tmp_path / "s"))

    def batch(r):
        n = int(r.P["episodes_per_update"])
        d0 = r.sampler.balanced(r.spec.T, n, r.rngs["start"])
        roles = r.rngs["start"].integers(0, 2, size=n)
        return collect_batch_v2(r.spec, r.agent, np.full(n, r.spec.T), d0, roles, r.rngs["env"],
                                r.rngs["learn"], r.rngs["opp"], 1.0, 1.0, r.P["es_bin_width"],
                                reward_mode="expected")
    r_s.agent.actor.conc_scale = 1.0
    b0, b1 = batch(r_base), batch(r_s)
    for k in ("states", "actions", "logp", "returns", "advantages"):
        assert np.array_equal(b0[k], b1[k])
    r_s.agent.actor.conc_scale = 3.0
    r_s.agent.opponent.conc_scale = 3.0
    b3 = batch(r_s)
    al, be = r_s.agent.beta_params(b3["states"])
    lp = torch.distributions.Beta(torch.as_tensor(al), torch.as_tensor(be)).log_prob(torch.as_tensor(b3["actions"]))
    assert np.array_equal(lp.numpy(), b3["logp"])
    r_s.agent.actor.conc_scale = 2.5
    r_s.agent.refresh_snapshot()
    assert r_s.agent.opponent.conc_scale == 2.5


def test_opponent_scale_set_before_every_rollout_in_a_run(parent, tmp_path):
    """The runner sets the scale on the lagged opponent before each rollout (not only at refreshes)."""
    cfg = _cont_cfg(parent)
    cfg["conc_anneal"] = {"phase": "A", "local_first": 1, "local_last": 6, "scale_first": 1.0, "scale_last": 4.0}
    d = str(tmp_path / "op")
    os.makedirs(d, exist_ok=True)
    run = Run(cfg, d)
    seen = []
    import run.run_v2_stagewise as _rs
    orig = _rs.collect_batch_v2

    def spy(*a, **k):
        seen.append((run.agent.actor.conc_scale, run.agent.opponent.conc_scale))
        return orig(*a, **k)
    _rs.collect_batch_v2 = spy
    try:
        assert execute(run, cfg, d, "pytest") == 0
    finally:
        _rs.collect_batch_v2 = orig
    assert len(seen) == 6 and all(a == o for a, o in seen)
    assert [a for a, _ in seen] == [run.conc_scale_for(j) for j in range(1, 7)]


def test_annealing_refused_outside_phase_A_continue():
    cfg = base_config()
    cfg["conc_anneal"] = {"phase": "A", "local_first": 1, "local_last": 6, "scale_first": 1.0, "scale_last": 2.0}
    with pytest.raises(ConfigError, match="conc_anneal is defined"):
        validate_config(cfg)


# ====================================================================== config refusals for the new keys
@pytest.mark.parametrize("mutate, msg", [
    (lambda c: c.update(ppo_overrides={"epochs": 5}), "ppo_overrides keys"),
    (lambda c: c.update(ppo_overrides={"minibatch": 0}), "positive int"),
    (lambda c: c.update(ppo_overrides={"target_kl": -1.0}), "positive number"),
    (lambda c: c.update(continuation_value_mode="bogus"), "continuation_value_mode"),
    (lambda c: c.update(continuation_value_mode="expected"), "requires stage2_update_mode=frozen"),
    (lambda c: c.update(mode="phase_P"), "needs parent"),
])
def test_new_key_refusals(mutate, msg):
    c = base_config()
    mutate(c)
    with pytest.raises(ConfigError, match=msg):
        validate_config(c)


def test_expected_continuation_refused_without_frozen_mean():
    c = base_config()
    c.update(mode="phase_B", parent_checkpoint="x", parent_sha256="y", continuation_value_mode="expected")
    c["flags"].update(stage2_update_mode="frozen", adv_norm_scope="stage1_rows",
                      continuation_action_mode="stochastic")
    with pytest.raises(ConfigError, match="continuation_action_mode=mean"):
        validate_config(c)
    c["flags"]["continuation_action_mode"] = "mean"
    validate_config(c)                                   # frozen + mean: accepted


# ====================================================================== D1 instrumentation
class _StubAgent:
    """Minimal agent whose actor Beta parameters are constants (for the clamp-count test)."""

    class _Cfg:
        action_clamp = 1e-6

    cfg = _Cfg()

    def __init__(self, alpha: float, beta: float):
        self.a, self.b = alpha, beta
        self.opponent = self

    def beta_params(self, obs, net=None):
        n = obs.shape[0]
        return np.full(n, self.a, dtype=np.float32), np.full(n, self.b, dtype=np.float32)

    def sample_actions(self, alpha, beta, rng):
        c = self.cfg.action_clamp
        return np.clip(rng.beta(alpha.astype(float), beta.astype(float)), c, 1.0 - c).astype(np.float32)

    def log_prob(self, alpha, beta, actions):
        dist = torch.distributions.Beta(torch.as_tensor(alpha), torch.as_tensor(beta))
        return dist.log_prob(torch.as_tensor(actions)).numpy()

    def value(self, obs):
        return np.zeros(obs.shape[0], dtype=np.float32)


def test_d1_counts_match_beta_cdf_and_leave_streams_untouched():
    from scipy.stats import beta as beta_dist
    from envs.curriculum_env import GameSpec
    spec = GameSpec(w_h=6.0, w_l=2.0, k=2.0 / 7000.0, q=50.0, T=2)
    n = 100_000
    agent = _StubAgent(0.3, 0.3)
    rngs = [np.random.default_rng(s) for s in (1, 2, 3)]
    rngs_ref = [np.random.default_rng(s) for s in (1, 2, 3)]
    d0 = np.random.default_rng(5).uniform(-spec.domain_half(2), spec.domain_half(2), n)
    b = collect_batch_v2(spec, agent, np.full(n, 2), d0, np.zeros(n, dtype=int), rngs[0], rngs[1], rngs[2],
                         1.0, 1.0, 10.0)
    c = agent.cfg.action_clamp
    p_lo, p_hi = float(beta_dist.cdf(c, 0.3, 0.3)), float(beta_dist.sf(1.0 - c, 0.3, 0.3))
    d1 = b["d1"]
    for who in ("L", "O"):
        assert d1[f"d1_{who}_s2_n"] == n and d1[f"d1_{who}_s1_n"] == 0
        for key, p in (("lo", p_lo), ("hi", p_hi)):
            se = (n * p * (1 - p)) ** 0.5
            assert abs(d1[f"d1_{who}_s2_{key}"] - n * p) <= 3 * se, (who, key)
    inside = np.abs(d0) < 2 * spec.q
    assert d1["d1_L_s2_in_n"] == int(inside.sum()) and d1["d1_L_s2_out_n"] == int((~inside).sum())
    assert d1["d1_L_s2_in_lo"] + d1["d1_L_s2_out_lo"] == d1["d1_L_s2_lo"]
    assert d1["d1_L_s2_in_hi"] + d1["d1_L_s2_out_hi"] == d1["d1_L_s2_hi"]
    # the instrumentation consumed exactly the draws of the original sampler: replay on fresh streams
    al, be = agent.beta_params(spec.encode_obs(2, d0))
    ref_l = agent.sample_actions(al, be, rngs_ref[1])
    agent.sample_actions(al, be, rngs_ref[2])
    assert np.array_equal(b["actions"], ref_l)
    assert np.array_equal(np.clip(b["d1_buf"]["raw"], c, 1 - c).astype(np.float32), ref_l)
    assert rngs[1].bit_generator.state == rngs_ref[1].bit_generator.state
    assert rngs[2].bit_generator.state == rngs_ref[2].bit_generator.state
    assert b["d1_buf"]["alpha"].shape == (n,) and float(b["d1_buf"]["alpha"][0]) == float(np.float32(0.3))


def test_d1_columns_and_buffers_in_a_run(parent, tmp_path):
    cfg = _cont_cfg(parent)
    cfg["budget_overrides"]["phase_caps"]["A"] = 100
    cfg["budget_overrides"]["warmup"] = 1000
    d = str(tmp_path / "d1")
    os.makedirs(d, exist_ok=True)
    run = Run(cfg, d)
    assert execute(run, cfg, d, "pytest") == 0
    rows = _read_csv(os.path.join(d, "v2_updates.csv"))
    r0, n = rows[0], int(run.P["episodes_per_update"])
    assert int(r0["d1_L_s2_n"]) == n and int(r0["d1_L_s1_n"]) == 0
    assert int(r0["d1_L_s2_in_n"]) + int(r0["d1_L_s2_out_n"]) == n
    assert int(r0["d1_pol_n_rows"]) == n and float(r0["d1_pol_alpha_min"]) > 0
    files = sorted(os.listdir(os.path.join(d, "d1_buffers")))
    zs = [np.load(os.path.join(d, "d1_buffers", f)) for f in files]
    assert [int(z["local"]) for z in zs] == [25, 50, 100]               # 25, midpoint, cap
    assert files == [f"u{6 + loc:05d}.npz" for loc in (25, 50, 100)]    # named by the global update
    assert set(zs[0].files) >= {"states", "raw", "actions", "alpha", "beta", "stage", "old_logp", "adv_raw",
                                "policy_mask", "adv_norm_mean", "adv_norm_std", "local", "global_update"}
    assert zs[0]["states"].shape == (n, 2) and zs[0]["policy_mask"].all()
    assert np.array_equal(np.clip(zs[0]["raw"], 1e-6, 1 - 1e-6).astype(np.float32), zs[0]["actions"])


def test_d1_policy_rows_are_stage1_rows_in_phase_B(parent, tmp_path):
    cfg = base_config()
    cfg.update(mode="phase_B", parent_checkpoint=parent, parent_sha256="in-process")
    cfg["flags"].update(B2_MEAN)
    cfg["flags"]["reward_mode"] = "expected"
    cfg["budget_overrides"]["phase_caps"]["B"] = 2
    d = str(tmp_path / "b")
    os.makedirs(d, exist_ok=True)
    run = Run(cfg, d)
    assert execute(run, cfg, d, "pytest") == 0
    r0 = _read_csv(os.path.join(d, "v2_updates.csv"))[0]
    n = int(run.P["episodes_per_update"])
    assert int(r0["d1_L_s1_n"]) == n and int(r0["d1_L_s2_n"]) == n and int(r0["d1_pol_n_rows"]) == n
    z = np.load(os.path.join(d, "d1_buffers", sorted(os.listdir(os.path.join(d, "d1_buffers")))[0]))
    assert int(z["policy_mask"].sum()) == n and (z["stage"][z["policy_mask"]] == 1).all()


# ====================================================================== expected continuation (method 6)
def _phase_b_batch(parent_path, tmp_path, table=None):
    """One phase-B rollout (frozen stage 2, mean continuation, expected reward) with an optional table."""
    r = make_run("phase_B", B_ARMS["B2_frozen_s1norm"], str(tmp_path), parent_path)
    spec, P = r.spec, r.P
    n = int(P["episodes_per_update"])
    roles = r.rngs["start"].integers(0, 2, size=n)
    b = collect_batch_v2(spec, r.agent, np.ones(n, dtype=int), np.zeros(n), roles, r.rngs["env"],
                         r.rngs["learn"], r.rngs["opp"], 1.0, 1.0, P["es_bin_width"], reward_mode="expected",
                         frozen=r.agent.frozen, continuation_action_mode="mean", cont_table=table)
    return r, b


def test_expected_continuation_rollout_changes_only_stage1_returns(parent, tmp_path):
    from utils.v2_continuation import build_continuation_table
    r0, b0 = _phase_b_batch(parent, tmp_path / "s", None)
    tab = build_continuation_table(r0.agent.frozen, r0.spec, stage=2, step=0.25)
    r1, b1 = _phase_b_batch(parent, tmp_path / "e", tab)
    s1, s2 = b0["stage"] == 1, b0["stage"] == 2
    for k in ("states", "actions", "logp", "stage"):                       # every draw and stored value
        assert np.array_equal(b0[k], b1[k]), k
    assert np.array_equal(b0["returns"][s2], b1["returns"][s2])            # (iv) stage-2 rows unchanged
    assert np.array_equal(b0["advantages"][s2], b1["advantages"][s2])
    assert not np.array_equal(b0["returns"][s1], b1["returns"][s1])
    for k, g0 in r0.rngs.items():                                          # (iii) rule A6: streams equal
        assert g0.bit_generator.state == r1.rngs[k].bit_generator.state, k
    # stage-1 advantage = return - V(s1); return - (-k e1^2) lies inside the table's value range
    e1 = r0.spec.effort_from_action(b1["actions"][s1])
    v1 = r1.agent.value(b1["states"][s1]).astype(float)
    assert np.allclose(b1["returns"][s1] - b1["advantages"][s1], v1, atol=1e-5)
    lo, hi = float(tab.values.min()), float(tab.values.max())
    cost = -r0.spec.k * e1 ** 2
    assert (b1["returns"][s1] >= cost + lo - 1e-4).all() and (b1["returns"][s1] <= cost + hi + 1e-4).all()


def test_expected_continuation_refused_by_the_rollout_without_frozen_mean(parent, tmp_path):
    from utils.v2_continuation import build_continuation_table
    r = make_run("phase_B", B_ARMS["B2_frozen_s1norm"], str(tmp_path / "x"), parent)
    tab = build_continuation_table(r.agent.frozen, r.spec, stage=2, step=0.5)
    n = 8
    with pytest.raises(ValueError, match="expected-continuation table requires"):
        collect_batch_v2(r.spec, r.agent, np.ones(n, dtype=int), np.zeros(n), np.zeros(n, dtype=int),
                         r.rngs["env"], r.rngs["learn"], r.rngs["opp"], 1.0, 1.0, 10.0, frozen=None,
                         cont_table=tab)


def test_expected_continuation_run_matches_baseline_streams(parent, tmp_path):
    """Phase B with continuation_value_mode=expected: same stream positions every update (A6)."""
    outs = []
    for mode in ("sampled", "expected"):
        cfg = base_config()
        cfg.update(mode="phase_B", parent_checkpoint=parent, parent_sha256="in-process",
                   continuation_value_mode=mode)
        cfg["flags"].update(B2_MEAN)
        cfg["flags"]["reward_mode"] = "expected"
        cfg["budget_overrides"]["phase_caps"]["B"] = 3
        d = str(tmp_path / mode)
        os.makedirs(d)
        run = Run(cfg, d)
        assert execute(run, cfg, d, "pytest") == 0
        outs.append((d, run))
    rows = [_read_csv(os.path.join(d, "v2_updates.csv")) for d, _ in outs]
    pos = [c for c in rows[0][0] if c.startswith("rngpos_")]
    assert len(pos) == 5
    for r0, r1 in zip(*rows):
        assert all(r0[c] == r1[c] for c in pos)
    assert float(rows[1][0]["adv_s1_std"]) != float(rows[0][0]["adv_s1_std"])
    man = json.load(open(os.path.join(outs[1][0], "manifest.json")))
    assert man["continuation_value_mode"] == "expected"
    assert outs[1][1].cont_table is not None and outs[0][1].cont_table is None
    assert "continuation_table" in json.load(open(os.path.join(outs[1][0], "v2_run_summary.json")))["costs_v2"]["seconds"]


# ====================================================================== phase P (method 5)
def _p_cfg(parent_path, cap=6):
    cfg = base_config()
    cfg.update(mode="phase_P", parent_checkpoint=parent_path, parent_sha256="in-process")
    cfg["flags"]["reward_mode"] = "expected"
    cfg["budget_overrides"]["phase_caps"].update({"A": cap, "P": cap})
    cfg["budget_overrides"]["verifier_timeout"] = 3
    cfg["lr_decay"] = [{"phase": "P", "start_lr": 3e-5, "end_lr": 3e-5, "local_first": 1, "local_last": cap}]
    return cfg


def test_phase_P_untouched_head_critic_and_start_stream(parent, tmp_path):
    dP, dA = str(tmp_path / "P"), str(tmp_path / "A")
    os.makedirs(dP)
    os.makedirs(dA)
    cfgP = _p_cfg(parent)
    run = Run(cfgP, dP)
    assert execute(run, cfgP, dP, "pytest") == 0
    chk = json.load(open(os.path.join(dP, "phaseP_checks.json")))
    assert chk == {"concentration_head_bit_identical": True, "critic_bit_identical": True,
                   "critic_adam_state_bit_identical": True}
    par = torch.load(parent, weights_only=False)
    end = torch.load(os.path.join(dP, "state_end_P.pt"), weights_only=False)
    pa, ea = par["agent"]["actor"], end["agent"]["actor"]
    assert torch.equal(pa["out.weight"][1], ea["out.weight"][1]) and torch.equal(pa["out.bias"][1], ea["out.bias"][1])
    assert not torch.equal(pa["out.weight"][0], ea["out.weight"][0])           # the mean head did move
    assert not torch.equal(pa["l1.weight"], ea["l1.weight"])
    assert all(torch.equal(v, end["agent"]["critic"][k]) for k, v in par["agent"]["critic"].items())
    assert end["phases_done"] == ["A", "P"] and end["counters"]["global_u"] == par["counters"]["global_u"] + 6
    rows = _read_csv(os.path.join(dP, "v2_updates.csv"))
    assert [r["phase"] for r in rows] == ["P"] * 6
    assert {"loss", "grad_norm_pre_clip", "foc_abs_mean", "foc_abs_max", "e0", "actor_lr"} <= set(rows[0])
    assert all(float(r["actor_lr"]) == 3e-5 for r in rows)
    ck = _read_csv(os.path.join(dP, "v2_checkpoints_P.csv"))
    assert [(r["local"], r["reason"]) for r in ck] == [("3", "timeout"), ("6", "timeout")]
    fv = json.load(open(os.path.join(dP, "final_v2.json")))
    assert "development" in fv and "final" in fv
    # the start stream sits where an equally long phase-A continuation leaves it; the other streams do not move
    cfgA = _cont_cfg(parent)
    runA = Run(cfgA, dA)
    assert execute(runA, cfgA, dA, "pytest") == 0
    endA = torch.load(os.path.join(dA, "state_end_A.pt"), weights_only=False)
    assert end["rng"]["start"] == endA["rng"]["start"]
    for k in ("env", "learn", "opp"):
        assert end["rng"][k] == par["rng"][k]                                  # no shock / action draws
    assert end["agent"]["rng_minibatch"] == par["agent"]["rng_minibatch"]
    rA = _read_csv(os.path.join(dA, "v2_updates.csv"))
    assert [r["rngpos_start"] for r in rows] == [r["rngpos_start"] for r in rA]


def test_phase_P_requires_phase_window_and_cap(parent, tmp_path):
    cfg = _p_cfg(parent)
    cfg.pop("lr_decay")
    with pytest.raises(ConfigError, match="needs an lr_decay window for phase P"):
        Run(cfg, str(tmp_path / "a"))
    cfg = _p_cfg(parent)
    cfg["budget_overrides"]["phase_caps"].pop("P")
    with pytest.raises(ConfigError, match="no entry for phase P"):
        Run(cfg, str(tmp_path / "b"))


# ====================================================================== review follow-ups
def test_d1_buffer_stores_the_generating_actor(parent, tmp_path):
    """Each buffer carries the pre-update actor (and scale) that generated it: its log-probs equal old_logp."""
    cfg = _cont_cfg(parent)
    cfg["budget_overrides"]["phase_caps"]["A"] = 25
    cfg["budget_overrides"]["warmup"] = 1000
    cfg["conc_anneal"] = {"phase": "A", "local_first": 1, "local_last": 25, "scale_first": 1.0, "scale_last": 2.0}
    d = str(tmp_path / "pre")
    os.makedirs(d)
    run = Run(cfg, d)
    assert execute(run, cfg, d, "pytest") == 0
    z = np.load(os.path.join(d, "d1_buffers", "u00031.npz"))               # local 25 = the cap
    assert float(z["conc_scale_pre"]) == run.conc_scale_for(25) == 2.0
    actor = BetaActor(64, 100.0, 1e-6, torch.Generator().manual_seed(0))
    actor.load_state_dict({k[len("actor_pre."):]: torch.as_tensor(z[k]) for k in z.files if k.startswith("actor_pre.")})
    actor.conc_scale = float(z["conc_scale_pre"])
    al, be = actor(torch.as_tensor(z["states"]))
    assert np.array_equal(al.detach().numpy(), z["alpha"]) and np.array_equal(be.detach().numpy(), z["beta"])
    lp = torch.distributions.Beta(al, be).log_prob(torch.as_tensor(z["actions"]))
    assert np.array_equal(lp.detach().numpy(), z["old_logp"])
    # the post-update export of the same update is a different actor
    w = np.load(os.path.join(d, "weights", "u00031.npz")) if os.path.exists(os.path.join(d, "weights", "u00031.npz")) else None
    if w is not None:
        assert not np.array_equal(w["actor.out.weight"], z["actor_pre.out.weight"])


def test_draw_equals_sample_actions_on_shared_streams():
    from run.v2_rollout import _draw
    gen = torch.Generator().manual_seed(3)
    agent = make_run("phase_A", {"reward_mode": "expected"}, "/tmp/_unused_refine_draw").agent
    a = np.random.default_rng(1).uniform(0.2, 90.0, 2000).astype(np.float32)
    b = np.random.default_rng(2).uniform(0.2, 90.0, 2000).astype(np.float32)
    r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
    raw, act = _draw(agent, a, b, r1)
    assert np.array_equal(act, agent.sample_actions(a, b, r2))
    assert r1.bit_generator.state == r2.bit_generator.state
    del gen


def test_mean_effort_numpy_reads_the_exported_scale(parent, tmp_path):
    cfg = _cont_cfg(parent)
    cfg["conc_anneal"] = {"phase": "A", "local_first": 1, "local_last": 6, "scale_first": 1.0, "scale_last": 2.5}
    d = str(tmp_path / "ex")
    os.makedirs(d)
    run = Run(cfg, d)
    assert execute(run, cfg, d, "pytest") == 0
    w = dict(np.load(os.path.join(d, "checkpoint_weights.npz")))
    obs = run.spec.encode_obs(2, np.linspace(-100, 100, 41))
    _, an, bn = mean_effort_numpy(w, obs)                                     # picks conc_scale up from the export
    al, be = run.agent.beta_params(obs)
    np.testing.assert_allclose(an, al, rtol=1e-6)
    np.testing.assert_allclose(bn, be, rtol=1e-6)
    _, an1, _ = mean_effort_numpy({k: v for k, v in w.items() if k != "conc_scale"}, obs)
    assert not np.allclose(an1, al, rtol=1e-3)                                # unscaled would disagree


def test_manifest_records_the_scale_of_the_restored_parent(parent, tmp_path):
    cfg = _cont_cfg(parent)
    cfg["conc_anneal"] = {"phase": "A", "local_first": 1, "local_last": 6, "scale_first": 1.0, "scale_last": 2.0}
    d1 = str(tmp_path / "a")
    os.makedirs(d1)
    run = Run(cfg, d1)
    assert execute(run, cfg, d1, "pytest") == 0
    cfg2 = _cont_cfg(os.path.join(d1, "state_end_A.pt"))                       # annealed parent, no anneal key
    d2 = str(tmp_path / "b")
    os.makedirs(d2)
    run2 = Run(cfg2, d2)
    assert execute(run2, cfg2, d2, "pytest") == 0
    man = json.load(open(os.path.join(d2, "manifest.json")))
    assert man["conc_scale_initial"]["actor"] == 2.0 and man["conc_scale_final"]["actor"] == 2.0
    man1 = json.load(open(os.path.join(d1, "manifest.json")))
    assert man1["conc_scale_initial"]["actor"] == 1.0 and man1["conc_scale_final"]["actor"] == 2.0


def test_phase_P_timeout_entry_checked_at_construction(parent, tmp_path):
    cfg = _p_cfg(parent)
    cfg["budget_overrides"]["verifier_timeout"] = {"A": 5, "B": 5, "C": 5}
    with pytest.raises(ConfigError, match="verifier_timeout has no entry for phase P"):
        Run(cfg, str(tmp_path / "a"))


def test_record_phase_c_consistency_still_checked_for_every_mode(tmp_path):
    cfg = base_config()
    cfg["record"]["protocol"]["phase_c_root"] = 100        # inconsistent record: 100 + 256 != 512
    with pytest.raises(ConfigError, match="phase_c_root"):
        Run(cfg, str(tmp_path / "a"))
