"""Phase-2 v2 infrastructure tests (C1-C7 and the 3b list).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_v2_infra.py -q
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

from agents.ppo_curriculum import CurriculumPPO  # noqa: E402
from agents.ppo_curriculum_v2 import masked_actor_loss  # noqa: E402
from envs.curriculum_env import stage_reward, step_gap  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from run.run_v2_stagewise import Run, execute, main as v2_main, validate_config  # noqa: E402
from run.v2_rollout import collect_batch_v2, expected_terminal_reward  # noqa: E402
from rng_alignment import B_ARMS, alignment_table, base_config, make_run, one_episode_batch, states  # noqa: E402

N_B = 20   # phase-B updates in the branch tests


def _eq(a, b, path="", skip=("time_sec", "update_wall_sec", "elapsed_phase_wall_sec", "verifier_sec")):
    """Recursive exact equality (numpy-aware); returns the list of differing paths."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in skip:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k} missing")
            else:
                out += _eq(a[k], b[k], f"{path}/{k}", skip)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path} len"]
        for i, (x, y) in enumerate(zip(a, b)):
            out += _eq(x, y, f"{path}[{i}]", skip)
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
    """Throwaway end-of-phase-A full-state checkpoint (q=50, seed 10501, 6 A updates)."""
    d = str(tmp_path_factory.mktemp("parent"))
    cfg = base_config()
    run = Run(cfg, d)
    assert execute(run, cfg, d, "pytest") == 0
    return os.path.join(d, "state_end_A.pt")


# ---------------------------------------------------------------- config refusals
@pytest.mark.parametrize("mutate, msg", [
    (lambda c: c.pop("parent_sha256"), "missing"),
    (lambda c: c["flags"].update(adv_norm_scope="stage1_rows", stage2_update_mode="joint") or c.update(mode="phase_B", parent_checkpoint="x", parent_sha256="y"), "joint requires"),
    (lambda c: c["flags"].update(stage2_update_mode="frozen"), "only defined in mode phase_B"),
    (lambda c: c["flags"].update(continuation_action_mode="mean"), "requires stage2_update_mode=frozen"),
    (lambda c: c.update(mode="phase_B"), "needs parent"),
    (lambda c: c.update(mode="full"), "regression mode"),
    (lambda c: c["flags"].pop("reward_mode"), "flags must have"),
])
def test_config_refusals(mutate, msg):
    c = base_config()
    mutate(c)
    with pytest.raises(ConfigError, match=msg):
        validate_config(c)


def test_parent_hash_mismatch_refused(parent, tmp_path, monkeypatch):
    c = base_config()
    c.update(mode="phase_B", parent_checkpoint=parent, parent_sha256="0" * 64)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(c))
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(p), "--out-dir", str(tmp_path / "o")])
    with pytest.raises(ConfigError, match="sha256"):
        v2_main()


# ---------------------------------------------------------------- C1: full state / branching
def test_continuous_equals_branched(parent, tmp_path):
    """A then B in one process == A, save, restore, B (joint, fixed budget)."""
    r1 = make_run("phase_A", {}, str(tmp_path / "cont"), caps={"B": N_B})
    r1.run_phase("A")
    p1 = str(tmp_path / "s.pt")
    torch.save(r1.full_state("A"), p1)
    r1.run_phase("B")
    r2 = make_run("phase_B", B_ARMS["A_joint"], str(tmp_path / "br"), p1, caps={"B": N_B})
    r2.run_phase("B")
    nA = len([h for h in r1.history if h["phase"] == "A"])
    assert _eq(r1.history[nA:], r2.history) == []
    assert _eq([v for v in r1.verifier_log if v["phase"] == "B"], r2.verifier_log) == []
    for k, v in r1.agent.actor.state_dict().items():
        assert torch.equal(v, r2.agent.actor.state_dict()[k])
    assert _eq({k: g.bit_generator.state for k, g in r1.rngs.items()},
               {k: g.bit_generator.state for k, g in r2.rngs.items()}) == []


@pytest.fixture(scope="module")
def branches(parent, tmp_path_factory):
    """Two identical-flag branches per arm (A, B1, B2), N_B phase-B updates, via execute()."""
    out = {}
    for arm in ("A_joint", "B1_frozen_allnorm", "B2_frozen_s1norm"):
        dirs = []
        for rep in (0, 1):
            d = str(tmp_path_factory.mktemp(f"{arm}_{rep}"))
            cfg = base_config()
            cfg.update(mode="phase_B", parent_checkpoint=parent, parent_sha256="in-process", arm=arm)
            cfg["flags"].update(B_ARMS[arm])
            cfg["budget_overrides"]["phase_caps"]["B"] = N_B
            run = Run(cfg, d)
            assert execute(run, cfg, d, "pytest") == 0
            dirs.append(d)
        out[arm] = dirs
    return out


def _read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("arm", ["A_joint", "B1_frozen_allnorm", "B2_frozen_s1norm"])
def test_branch_reproducibility(branches, arm):
    d0, d1 = branches[arm]
    h0, h1 = (json.load(open(os.path.join(d, "train_history.json"))) for d in (d0, d1))
    assert len(h0["history"]) == N_B
    assert _eq(h0["history"], h1["history"]) == []
    assert _eq(h0["verifier_calls"], h1["verifier_calls"]) == []
    u0, u1 = (_read_csv(os.path.join(d, "v2_updates.csv")) for d in (d0, d1))
    assert _eq(u0, u1) == []
    c0, c1 = (_read_csv(os.path.join(d, "v2_checkpoints.csv")) for d in (d0, d1))
    assert len(c0) > 0 and _eq(c0, c1) == []
    w0, w1 = (np.load(os.path.join(d, "checkpoint_weights.npz")) for d in (d0, d1))
    assert all(np.array_equal(w0[k], w1[k]) for k in w0.files)


# ---------------------------------------------------------------- 3b.2 / C5 snapshot immutability
@pytest.mark.parametrize("arm", ["B1_frozen_allnorm", "B2_frozen_s1norm"])
def test_snapshot_immutable(branches, parent, arm):
    d = branches[arm][0]
    t = json.load(open(os.path.join(d, "drift_test.json")))
    assert t["pass"] and t["snapshot_params_bit_identical_to_parent_actor"]
    assert all(v == 0.0 for v in t["max_abs_diff_vs_freeze_time"].values())
    z = np.load(os.path.join(d, "drift_test_arrays.npz"))
    for k in ("mean", "alpha", "beta"):
        assert np.array_equal(z[f"parent_{k}"], z[f"end_{k}"])
    rows = _read_csv(os.path.join(d, "v2_checkpoints.csv"))
    assert all(float(r["stage2_drift_cand_maxabs"]) == 0.0 for r in rows)
    st = torch.load(os.path.join(d, "state_end_B.pt"), weights_only=False)
    par = torch.load(parent, weights_only=False)
    for k, v in st["agent"]["frozen"].items():
        assert torch.equal(v, par["agent"]["actor"][k])


def test_joint_logs_live_drift(branches):
    t = json.load(open(os.path.join(branches["A_joint"][0], "drift_test.json")))
    assert t["mode"] == "joint" and t["max_abs_diff_vs_freeze_time"]["mean"] > 0.0
    rows = _read_csv(os.path.join(branches["A_joint"][0], "v2_checkpoints.csv"))
    assert "stage2_drift_live_on_max" in rows[0] and "stage2_drift_live_off_max" in rows[0]


# ---------------------------------------------------------------- batches for unit tests
def _batch(parent, tmp_path, flags):
    r = make_run("phase_B", flags, str(tmp_path / "b"), parent)
    spec, P = r.spec, r.P
    n = int(P["episodes_per_update"])
    t0, d0 = np.ones(n, dtype=int), np.zeros(n)
    roles = r.rngs["start"].integers(0, 2, size=n)
    frozen = r.agent.frozen if flags["stage2_update_mode"] == "frozen" else None
    b = collect_batch_v2(spec, r.agent, t0, d0, roles, r.rngs["env"], r.rngs["learn"], r.rngs["opp"],
                         1.0, 1.0, P["es_bin_width"], frozen=frozen,
                         continuation_action_mode=flags["continuation_action_mode"])
    return r, b


# ---------------------------------------------------------------- 3b.1 gradient isolation
@pytest.mark.parametrize("arm", ["B1_frozen_allnorm", "B2_frozen_s1norm"])
def test_gradient_isolation(parent, tmp_path, arm):
    r, b = _batch(parent, tmp_path, B_ARMS[arm])
    st, ac, olp = (torch.as_tensor(b[k]) for k in ("states", "actions", "logp"))
    s1 = torch.as_tensor(b["stage"] == 1)
    adv = torch.as_tensor(b["advantages"]).clone().requires_grad_(True)
    rows = torch.nonzero(s1).squeeze(-1)
    loss, _, _ = masked_actor_loss(r.agent.actor, st, ac, olp, adv, rows, 0.2)
    (g_adv,) = torch.autograd.grad(loss, adv)
    assert torch.all(g_adv[~s1] == 0.0)
    assert torch.any(g_adv[s1] != 0.0)

    def param_grad(a):
        r.agent.actor.zero_grad(set_to_none=True)
        l_, _, _ = masked_actor_loss(r.agent.actor, st, ac, olp, a, rows, 0.2)
        l_.backward()
        return [p.grad.clone() for p in r.agent.actor.parameters()]
    base = torch.as_tensor(b["advantages"]).clone()
    g0 = param_grad(base)
    pert2 = base.clone()
    pert2[~s1] += 7.0
    g2 = param_grad(pert2)
    assert all(torch.equal(x, y) for x, y in zip(g0, g2))
    pert1 = base.clone()
    pert1[s1] += 7.0
    g1 = param_grad(pert1)
    assert any(not torch.equal(x, y) for x, y in zip(g0, g1))


def test_b2_update_ignores_stage2_advantages_end_to_end(parent, tmp_path):
    """B2: perturbing raw stage-2 advantages leaves the whole update bit-identical."""
    r, b = _batch(parent, tmp_path, B_ARMS["B2_frozen_s1norm"])
    s1 = b["stage"] == 1
    a1, a2 = copy.deepcopy(r.agent), copy.deepcopy(r.agent)
    adv2 = b["advantages"].copy()
    adv2[~s1] += 3.0
    a1.update(b["states"], b["actions"], b["logp"], b["returns"], b["advantages"], policy_mask=s1, norm_mask=s1)
    a2.update(b["states"], b["actions"], b["logp"], b["returns"], adv2, policy_mask=s1, norm_mask=s1)
    for k, v in a1.actor.state_dict().items():
        assert torch.equal(v, a2.actor.state_dict()[k])


# ---------------------------------------------------------------- 3b.3 normalization scope
@pytest.mark.parametrize("arm", ["B1_frozen_allnorm", "B2_frozen_s1norm"])
def test_norm_scope(parent, tmp_path, arm):
    r, b = _batch(parent, tmp_path, B_ARMS[arm])
    s1 = b["stage"] == 1
    adv = torch.as_tensor(b["advantages"])
    rows = adv if arm.startswith("B1") else adv[torch.as_tensor(s1)]
    diag = r.agent.update(b["states"], b["actions"], b["logp"], b["returns"], b["advantages"],
                          policy_mask=s1, norm_mask=None if arm.startswith("B1") else s1)
    assert diag["adv_raw_mean"] == float(rows.mean().item())
    assert diag["adv_raw_std"] == float(rows.std(unbiased=False).item())
    assert diag["n_norm_rows"] == (s1.size if arm.startswith("B1") else int(s1.sum()))
    assert diag["n_policy_rows"] == int(s1.sum())


# ---------------------------------------------------------------- 3b.4 joint unchanged
def test_joint_update_bit_identical_to_original(parent, tmp_path):
    r, b = _batch(parent, tmp_path, B_ARMS["A_joint"])
    a_new = copy.deepcopy(r.agent)
    a_old = copy.deepcopy(r.agent)
    d_new = a_new.update(b["states"], b["actions"], b["logp"], b["returns"], b["advantages"])
    d_old = CurriculumPPO.update(a_old, b["states"], b["actions"], b["logp"], b["returns"], b["advantages"])
    assert _eq(d_new, d_old) == []
    for k, v in a_new.actor.state_dict().items():
        assert torch.equal(v, a_old.actor.state_dict()[k])
    for k, v in a_new.critic.state_dict().items():
        assert torch.equal(v, a_old.critic.state_dict()[k])


# ---------------------------------------------------------------- C2 expected reward
@pytest.mark.parametrize("q", [50.0, 60.0])
def test_expected_reward_matches_sampled_mean(q):
    """Mean of many sampled terminal rewards == r_bar within MC error, both players."""
    from common import spec_for
    spec = spec_for(q)
    rng = np.random.default_rng(4242)
    n = 200_000
    for d in (-150.0, -60.0, 0.0, 35.0, 120.0):
        for ei, ej in ((0.0, 0.0), (40.0, 70.0), (90.0, 10.0)):
            eps_i, eps_j = rng.uniform(-q, q, n), rng.uniform(-q, q, n)
            for own_d, e_own, e_opp, e_a, e_b in ((d, ei, ej, eps_i, eps_j), (-d, ej, ei, eps_j, eps_i)):
                dn = step_gap(spec, np.full(n, own_d), np.full(n, e_own), np.full(n, e_opp), e_a, e_b)
                r = stage_reward(spec, 2, np.full(n, e_own), dn)
                rbar = float(expected_terminal_reward(spec, np.array([own_d]), np.array([e_own]), np.array([e_opp]))[0])
                if np.ptp(r) == 0:
                    assert abs(r.mean() - rbar) <= 1e-12
                else:
                    assert abs(r.mean() - rbar) <= 5 * r.std(ddof=1) / np.sqrt(n)


def test_expected_mode_preserves_rng_after_one_batch(parent, tmp_path):
    ra = make_run("phase_B", dict(B_ARMS["A_joint"], reward_mode="sampled"), str(tmp_path / "a"), parent)
    rb = make_run("phase_B", dict(B_ARMS["A_joint"], reward_mode="expected"), str(tmp_path / "b"), parent)
    one_episode_batch(ra, "B")
    one_episode_batch(rb, "B")
    assert _eq(states(ra), states(rb)) == []


# ---------------------------------------------------------------- 3b.5 / C4 RNG alignment
def test_numpy_beta_consumption_depends_on_parameters():
    """Source of any learn/opp misalignment: Generator.beta uses rejection sampling."""
    def st(a, b):
        g = np.random.default_rng(1)
        g.beta(a, b)
        return g.bit_generator.state["state"]["state"]
    base = st(np.full(512, 50.0), np.full(512, 50.0))
    rng = np.random.default_rng(0)
    assert any(st(rng.uniform(1, 200, 512), rng.uniform(1, 200, 512)) != base for _ in range(20))


@pytest.fixture(scope="module")
def alignment(parent, tmp_path_factory):
    return alignment_table(parent, N_B, str(tmp_path_factory.mktemp("align")))


def test_rng_alignment_after_one_batch(alignment):
    for name, res in alignment.items():
        assert all(res["after_one_batch"].values()), (name, res["after_one_batch"])


def test_rng_alignment_after_updates_fixed_count_streams(alignment):
    """env / start / minibatch streams have parameter-independent consumption: always aligned."""
    for name, res in alignment.items():
        r = res[f"after_{N_B}_updates"]
        assert r["env"] and r["start"] and r["minibatch"], (name, r)


# ---------------------------------------------------------------- 1c RNG position log
def test_rng_positions_logged_and_divergence_utility(branches):
    from rng_divergence import first_divergence
    d0, d1 = branches["A_joint"]
    rows = _read_csv(os.path.join(d0, "v2_updates.csv"))
    assert all(r[f"rngpos_{s}"] for r in rows for s in ("env", "learn", "opp", "start", "minibatch"))
    rec = first_divergence(d0, d1)
    assert rec["n_updates_compared"] == N_B
    assert all(rec[s] == "never" for s in ("env", "learn", "opp", "start", "minibatch"))
    # positions advance every update (they are positions, not constants)
    assert len({r["rngpos_env"] for r in rows}) == N_B
