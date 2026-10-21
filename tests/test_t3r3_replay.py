"""Tests of ``tools/t3/t3r3_replay.py`` (T3-R3 P1): the per-region quantities of DG5 on a hand-made batch, the passivity of the
instrumentation (reduced-budget ``bw`` config, temporary directories) and the bit-identity of a replay with its original."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))

import t3_helpers as th  # noqa: E402
import t3r3_replay as rp  # noqa: E402
from agents.ppo_curriculum import BetaActor  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402

SPEC = GameSpec(w_h=6.0, w_l=2.0, k=1.0 / 3500.0, q=50.0, T=3)


def _batch():
    """Six hand-made samples at stage 3 (q = 50, 2q = 100): gaps, actions (one at each clamp), raw advantages."""
    d = np.array([-150.0, -30.0, 0.0, 40.0, 120.0, 300.0])
    states = SPEC.encode_obs(3, d)
    actions = np.array([1e-6, 0.4, 0.6, 0.3, 1.0 - 1e-6, 0.05], dtype=np.float32)
    advs = np.array([0.5, -1.0, 2.0, 0.25, -0.75, 1.5], dtype=np.float32)
    return d, states, actions, advs


def _actor():
    gen = torch.Generator().manual_seed(11)
    net = BetaActor(64, 100.0, 1e-6, gen)
    with torch.no_grad():
        for p in net.parameters():
            p += 0.2 * torch.randn(p.shape, generator=gen)
    net.eval()
    return net


def _loop_grads(net, states, actions):
    """Per-sample gradients by one backward pass per sample (the training function ``Beta.log_prob``)."""
    st, ac = torch.as_tensor(states), torch.as_tensor(actions)
    out = []
    for i in range(st.shape[0]):
        al, be = net(st[i:i + 1])
        lp = torch.distributions.Beta(al, be).log_prob(ac[i:i + 1])[0]
        g = torch.autograd.grad(lp, list(net.parameters()))
        out.append(torch.cat([x.reshape(-1) for x in g]).numpy().astype(np.float64))
    return np.stack(out)


# ---------------------------------------------------------------------------------- hand computation of the per-region quantities
def test_per_sample_gradients_equal_the_backward_loop():
    net = _actor()
    d, states, actions, advs = _batch()
    J = rp.per_sample_grads(net, torch.as_tensor(states), torch.as_tensor(actions)).numpy()
    Jl = _loop_grads(net, states, actions)
    assert J.shape == Jl.shape and np.allclose(J, Jl, rtol=1e-4, atol=1e-5 * np.abs(Jl).max())


def test_update_diag_matches_a_hand_computation_and_shares_sum_to_one():
    net = _actor()
    before = [p.detach().clone() for p in net.parameters()]
    d, states, actions, advs = _batch()
    out = rp.update_diag(net, SPEC, states, actions, advs)
    sc = out["scalars"]
    # hand computation
    N = 6
    adv = (advs.astype(np.float64) - advs.astype(np.float64).mean()) / (advs.astype(np.float64).std() + 1e-8)
    J = _loop_grads(net, states, actions)
    inb = np.abs(d) < 100.0
    parts = {"in": inb, "out": ~inb}
    G = {k: (adv[m][:, None] * J[m]).sum(axis=0) / N for k, m in parts.items()}
    g = G["in"] + G["out"]
    tol = dict(rel=2e-3)
    assert sc["n_in"] == 3 and sc["n_out"] == 3 and sc["N"] == 6
    for k in ("in", "out"):
        assert sc[f"gnorm_{k}"] == pytest.approx(np.linalg.norm(G[k]), **tol)
        assert sc[f"share_{k}"] == pytest.approx(G[k] @ g / (g @ g), rel=2e-3, abs=1e-5)
        assert sc[f"adv_raw_mean_{k}"] == pytest.approx(advs[parts[k]].astype(np.float64).mean(), rel=1e-6)
        assert sc[f"adv_norm_mean_{k}"] == pytest.approx(adv[parts[k]].mean(), rel=1e-4, abs=1e-6)
    assert sc["share_in"] + sc["share_out"] == pytest.approx(1.0, abs=1e-9)
    assert sc["gnorm_total"] == pytest.approx(np.linalg.norm(g), **tol)
    # the clamp partition: one draw at each clamp, four free
    lab = np.array([0, 2, 2, 2, 1, 2])
    assert (sc["n_clamp_lo"], sc["n_clamp_hi"], sc["n_clamp_free"]) == (1, 1, 4)
    assert sc["share_clamp_lo"] + sc["share_clamp_hi"] + sc["share_clamp_free"] == pytest.approx(1.0, abs=1e-9)
    Gc = (adv[lab == 0][:, None] * J[lab == 0]).sum(axis=0) / N
    assert sc["share_clamp_lo"] == pytest.approx(Gc @ g / (g @ g), rel=2e-3, abs=1e-5)
    assert sc["sample_norm_mean_clamp_lo"] == pytest.approx(np.linalg.norm(J[0]), **tol)
    assert (sc["n_clamp_lo_out"], sc["n_clamp_lo_in"], sc["n_clamp_hi_out"], sc["n_clamp_hi_in"]) == (1, 0, 1, 0)
    # the bins: shares sum to one over the partition and the counts are the samples
    b = out["bins"]
    assert b["n"].sum() == N and np.nansum(b["share"]) == pytest.approx(1.0, abs=1e-9)
    nb = b["n"].size
    assert nb == 40 and b["n"][int(np.searchsorted(np.linspace(-400, 400, 41), 300.0, side="right") - 1)] == 1
    # the actor and its gradients are untouched
    for p, q0 in zip(net.parameters(), before):
        assert torch.equal(p, q0) and p.grad is None


def test_sensitivity_and_saturation_over_the_in_band_samples():
    net = _actor()
    d, states, actions, advs = _batch()
    sc = rp.update_diag(net, SPEC, states, actions, advs)["scalars"]
    x = torch.as_tensor(states[np.abs(d) < 100.0]).clone().requires_grad_(True)
    al, be = net(x)
    mu = al / (al + be)
    s = torch.autograd.grad(mu.sum(), x)[0][:, 1].abs().detach().numpy()
    assert sc["sens_in_mean"] == pytest.approx(float(s.mean()), rel=1e-4)
    assert 0.0 <= sc["sat1_in"] <= 1.0 and 0.0 <= sc["sat2_in"] <= 1.0


# ---------------------------------------------------------------------------------- passivity and bit identity
@pytest.fixture(scope="module")
def reduced_cfg(tmp_path_factory):
    base = str(tmp_path_factory.mktemp("t3r3_replay"))
    cfg = th.bw_cfg(base, budgets=(50, 40, 20), epu=48)
    return base, cfg


def _read_rows(path):
    import csv
    with open(path, newline="") as f:
        return [{k: v for k, v in r.items() if k not in ("update_wall_sec", "verifier_sec")} for r in csv.DictReader(f)]


def test_instrumentation_is_passive_and_bit_identical(reduced_cfg):
    base, cfg = reduced_cfg
    a, b = Path(base) / "plain", Path(base) / "instrumented"
    ra = rp.replay_stage3(json.loads(json.dumps(cfg)), str(a), instrument=False)
    rb = rp.replay_stage3(json.loads(json.dumps(cfg)), str(b), instrument=True)
    assert ra["violations"] == [] and rb["violations"] == []            # the instrumentation consumed no global RNG
    cmp = rp.compare_with_original(a, b)
    assert cmp["all_identical"], cmp
    # the stage-3 state (actor, critic, optimizer, streams) is identical, tensor by tensor
    sa = torch.load(a / "state_end_stage3.pt", weights_only=False, map_location="cpu")
    sb = torch.load(b / "state_end_stage3.pt", weights_only=False, map_location="cpu")
    keys = ("actor", "critic", "opponent", "opt_actor", "opt_critic")           # networks and both Adam states
    assert th.tree_equal({k: sa["agent"][k] for k in keys}, {k: sb["agent"][k] for k in keys})
    assert sa["rng"] == sb["rng"] and sa["agent"]["rng_minibatch"] == sb["agent"]["rng_minibatch"]
    # the records: one row per update; the rollout's clamp counters agree with the stored actions
    rows = _read_rows(b / "replay_updates.csv")
    assert len(rows) == 50
    cc = rp.counts_match([{k: float(v) for k, v in r.items()} for r in rows])
    assert cc["samples_differ_in_lo"] == cc["samples_differ_out_lo"] == cc["samples_differ_in_hi"] == cc["samples_differ_out_hi"] == 0
    assert cc["updates_differ_n_in"] == 0
    for r in rows:
        assert float(r["share_in"]) + float(r["share_out"]) == pytest.approx(1.0, abs=1e-6)
        assert int(float(r["n_in"])) + int(float(r["n_out"])) == 48
    z = np.load(b / "replay_bins.npz")
    assert {"adv_raw_std", "adv_norm_std"} <= set(z.files)
    assert z["share"].shape == (50, 40) and np.allclose(np.nansum(z["share"], axis=1), 1.0, atol=1e-6)


def test_replay_reproduces_the_original_stage3_bit_for_bit():
    """The replay driver on the reduced legacy ``bw`` run's own config equals that run's stage-3 files (exports, rows, checks,
    freeze arrays)."""
    orig, cfg = th.reduced_bw_run()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rec = rp.replay_one((orig, str(Path(td) / "rep"), True))
        assert rec["all_identical"], rec
        assert rec["clamp_count_crosscheck"]["samples_differ_out_lo"] == 0
        assert rec["global_rng_violations"] == []
