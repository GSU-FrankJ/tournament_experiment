"""MS-R3 actor-variant tests: the defaults are bit-identical to commit e8eb9a08, the three variants (t1 / relu / t10)
follow D2, the initial weights are identical across variants (C-INIT), the loaders never read a relu / t10 export as
the tanh d / B actor, and the opponent, the refresh, the frozen snapshot and the stage-1 table carry the variant.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r3_actor.py -p no:cacheprovider -q
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import ms_configs as mc  # noqa: E402
import replay_dev_rule as rdr  # noqa: E402
from agents.ppo_curriculum import (  # noqa: E402
    ACTOR_VARIANTS, BetaActor, CurriculumPPO, PPOConfig, mean_effort_numpy)
from run import run_ms_stagewise as rms  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from test_ms_r2_runner import (  # noqa: E402
    PARAMS, PROTO, _same_run_outputs, read_csv, reduced_nl, run_cfg, run_in_tree, same_weights, weights)
from test_ms_runner import _t3_params, make_cfg, small_params  # noqa: E402
from utils import ms_continuation as mcont  # noqa: E402
from utils.v2_continuation import shock_quadrature, terminal_value  # noqa: E402

BASE_COMMIT = "e8eb9a08"      # the MS-R2 head: the code the defaults must reproduce bit for bit
VARIANTS = ("relu", "t10")


def same_networks(a: str, b: str, u: int) -> bool:
    """The network arrays (``actor.*``, ``critic.*``) of the two exports at update ``u`` are identical. The labels of an export
    (``actor_variant``, ``conc_scale``) are not weights: a relu export always has an ``actor_variant`` entry that a t1 export
    lacks, and an s = 16 export a ``conc_scale`` entry that an s = 1 export lacks, so ``same_weights`` would be False for
    identical networks (review finding M1/M2)."""
    x, y = weights(a, u), weights(b, u)
    kx = sorted(k for k in x if k.startswith(("actor.", "critic.")))
    ky = sorted(k for k in y if k.startswith(("actor.", "critic.")))
    return kx == ky and all(np.array_equal(x[k], y[k]) for k in kx)


def with_variant(cfg: Dict[str, Any], variant: str) -> Dict[str, Any]:
    """The config with the MS-R3 keys (``actor_variant`` for relu / t10, ``init_digest`` for every actor)."""
    cfg = copy.deepcopy(cfg)
    if variant != "t1":
        cfg["actor_variant"] = variant
    cfg["init_digest"] = True
    return cfg


# ---------------------------------------------------------------------------------------------- 1. defaults == e8eb9a08
@pytest.fixture(scope="module")
def base_tree(tmp_path_factory) -> str:
    """The run code of commit e8eb9a08 (``git archive`` of run utils envs agents protocols) in a temp directory."""
    try:
        tar = subprocess.check_output(["git", "-C", str(ROOT), "archive", BASE_COMMIT, "--", "run", "utils", "envs",
                                       "agents", "protocols"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pytest.skip(f"commit {BASE_COMMIT} is not available in this clone")
    d = tmp_path_factory.mktemp("tree_e8eb9a08")
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(d)
    return str(d)


@pytest.mark.parametrize("arm", ["NL_st_s16", "NL_bb_s4"])
def test_without_the_new_keys_an_nl_run_equals_e8eb9a08(base_tree, tmp_path, arm):
    """An MS-R2 NL arm (reduced budgets; fixed-budget stratified / bin-balanced, scale ramp, noise report): the current
    tree and the e8eb9a08 tree end in the same state, same exports, same series, no new column."""
    cfg = reduced_nl(arm, str(tmp_path / "x"))
    for k in ("actor_variant", "init_digest"):
        assert k not in cfg
    a, b = str(tmp_path / "now"), str(tmp_path / "old")
    run_cfg(copy.deepcopy(cfg), a)
    run_in_tree(base_tree, copy.deepcopy(cfg), b)
    _same_run_outputs(a, b)
    for f in sorted(os.listdir(os.path.join(a, "weights"))):
        assert "actor_variant" not in weights(a, int(f[1:6]))


def test_without_the_new_keys_a_rule_run_equals_e8eb9a08(base_tree, tmp_path):
    """The MS_s35a5 rule configuration (reduced budgets, a polishing-capable rule)."""
    prm = small_params(2, K=10, block=20, cap=40, land=20, rho=0.05)
    cfg = make_cfg(str(tmp_path / "x"), "MS_s35a5", params=prm, epu=64)
    cfg["record"]["protocol"]["weights_every"] = 5
    a, b = str(tmp_path / "now"), str(tmp_path / "old")
    run_cfg(copy.deepcopy(cfg), a)
    run_in_tree(base_tree, copy.deepcopy(cfg), b)
    _same_run_outputs(a, b)


def test_init_digest_changes_only_the_manifest_of_a_t1_run(tmp_path):
    """``init_digest: true`` on a t1 run: every output except the manifest / run_config is unchanged; no variant field."""
    cfg = reduced_nl("NL_bb_s1", str(tmp_path / "x"))
    a, b = str(tmp_path / "plain"), str(tmp_path / "digest")
    run_cfg(copy.deepcopy(cfg), a)
    run_cfg(with_variant(cfg, "t1"), b)
    _same_run_outputs(a, b)
    ma, mb = json.load(open(os.path.join(a, "manifest.json"))), json.load(open(os.path.join(b, "manifest.json")))
    assert "init_state_sha256" not in ma and "actor_variant" not in ma and "actor_variant" not in mb
    ms = rms.MSRun(copy.deepcopy(cfg), str(tmp_path / "again"))
    assert mb["init_state_sha256"] == rms.MSRun._init_digest(ms.agent)


# ---------------------------------------------------------------------------------------------- 2. the forward of D2
def _random_actor(variant: str, seed: int = 5) -> BetaActor:
    net = BetaActor(64, 100.0, 1e-6, torch.Generator().manual_seed(seed))
    g = torch.Generator().manual_seed(seed + 1)
    net.out.weight.data = (torch.randn(net.out.weight.shape, generator=g) * 0.3)
    net.out.bias.data = (torch.randn(net.out.bias.shape, generator=g) * 0.1)
    net.l1.bias.data = (torch.randn(net.l1.bias.shape, generator=g) * 0.1)
    net.variant = variant
    return net


def _manual(net: BetaActor, x: np.ndarray, variant: str) -> np.ndarray:
    """D2 written out in float64 numpy: (alpha, beta) of the variant."""
    x = np.asarray(x, dtype=np.float64).copy()
    if variant == "t10":
        x[:, 1] *= 10.0
    act = (lambda a: np.maximum(a, 0.0)) if variant == "relu" else np.tanh
    w = {k: v.detach().double().numpy() for k, v in net.state_dict().items()}
    h = act(x @ w["l1.weight"].T + w["l1.bias"])
    h = act(h @ w["l2.weight"].T + w["l2.bias"])
    z = h @ w["out.weight"].T + w["out.bias"]
    mu = np.clip(1.0 / (1.0 + np.exp(-z[:, 0])), 1e-6, 1 - 1e-6)
    c = 100.0 + np.logaddexp(0.0, z[:, 1])
    return np.stack([mu * c, (1.0 - mu) * c], 1)


@pytest.mark.parametrize("variant", ACTOR_VARIANTS)
def test_the_forward_of_each_variant_matches_d2(variant):
    d = np.linspace(-120.0, 120.0, 41)
    x = np.stack([np.ones_like(d), d / 120.0], 1).astype(np.float32)
    net = _random_actor(variant)
    a, b = net(torch.as_tensor(x))
    got = np.stack([a.detach().numpy(), b.detach().numpy()], 1)
    np.testing.assert_allclose(got, _manual(net, x, variant), rtol=2e-5)


def test_t10_scales_the_d_component_only_and_everywhere_the_actor_is_evaluated():
    """t10(x) == t1(x * (1, 10)) exactly; the stage feature (column 0) is not scaled; the same through the policy
    functions that the verifier and the continuation table use."""
    d = np.linspace(-100.0, 100.0, 21)
    x = np.stack([np.ones_like(d), d / 110.0], 1).astype(np.float32)
    t10, t1 = _random_actor("t10"), _random_actor("t1")
    x10 = x.copy()
    x10[:, 1] *= np.float32(10.0)
    a10, b10 = t10(torch.as_tensor(x))
    a1, b1 = t1(torch.as_tensor(x10))
    assert torch.equal(a10, a1) and torch.equal(b10, b1)
    xs = x.copy()
    xs[:, 0] *= np.float32(10.0)
    a_s, _ = t1(torch.as_tensor(xs))
    assert not torch.equal(a10, a_s)                       # scaling the stage feature would be a different function
    from envs.curriculum_env import GameSpec
    from run.run_final_dp_br import make_policy_fns
    from utils.v2_continuation import frozen_mean_effort
    cfg = mc.build_config(PROTO, 50, 10501, "NL_bb_s1", "unused", PARAMS)
    spec = GameSpec(**{k: cfg["record"]["game"][k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")})

    class Shim:                                            # the (agent, net) interface of make_policy_fns
        device = torch.device("cpu")

        def beta_params(self, obs, net=None):
            with torch.no_grad():
                a, b = net(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
            return a.numpy(), b.numpy()
    dd = np.linspace(-90.0, 90.0, 19)
    xb = spec.encode_obs(2, dd)
    xb[:, 1] *= np.float32(10.0)                           # what t10 does inside its forward
    with torch.no_grad():
        a, b = t1(torch.as_tensor(xb))
    want = spec.e_min + spec.e_range * (a / (a + b)).numpy().astype(float)
    with torch.no_grad():
        np.testing.assert_allclose(frozen_mean_effort(t10, spec, 2, dd), want, rtol=1e-6)     # continuation table path
    np.testing.assert_allclose(make_policy_fns(Shim(), spec, net=t10)[0](2, dd), want, rtol=1e-6)   # verifier path


def test_set_actor_variant_refuses_an_unknown_variant_and_forward_refuses_a_bad_attribute():
    ag = CurriculumPPO(PPOConfig(), torch.Generator().manual_seed(1), np.random.default_rng(0))
    with pytest.raises(ValueError):
        ag.set_actor_variant("gelu")
    ag.actor.variant = "gelu"
    with pytest.raises(ValueError):
        ag.actor(torch.zeros(2, 2))


# ---------------------------------------------------------------------------------------------- 3. C-INIT
def test_initial_weights_are_identical_across_variants_for_a_q_seed(tmp_path):
    """Actor and critic initial weights (and the runner's recorded digest) do not depend on the variant."""
    states: Dict[str, Tuple[Dict[str, torch.Tensor], str]] = {}
    base = reduced_nl("NL_bb_s1", str(tmp_path / "x"))
    for v in ("t1",) + VARIANTS:
        ms = rms.MSRun(with_variant(base, v), str(tmp_path / v))
        sd = {f"{n}.{k}": t.clone() for n, net in (("actor", ms.agent.actor), ("critic", ms.agent.critic))
              for k, t in net.state_dict().items()}
        states[v] = (sd, ms.init_state_sha256)
        assert ms.actor_variant == v and ms.agent.actor.variant == v and ms.agent.opponent.variant == v
    for v in VARIANTS:
        assert states[v][1] == states["t1"][1]
        assert all(torch.equal(states[v][0][k], states["t1"][0][k]) for k in states["t1"][0])
    # a different seed gives a different digest (the digest is not a constant)
    other = rms.MSRun(with_variant(reduced_nl("NL_bb_s1", str(tmp_path / "y"), seed=10502), "relu"), str(tmp_path / "o"))
    assert other.init_state_sha256 != states["t1"][1]


def test_the_init_digest_covers_the_actor_and_the_critic_and_consumes_no_rng(tmp_path):
    """A change of any actor or critic initial weight changes the digest (review finding M3: an actor-only digest would
    leave C-INIT blind to the critic), and computing it draws no random number."""
    ms = rms.MSRun(with_variant(reduced_nl("NL_bb_s1", str(tmp_path / "x")), "t1"), str(tmp_path / "x"))
    base = rms.MSRun._init_digest(ms.agent)
    assert base == ms.init_state_sha256
    state_before = (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), copy.deepcopy(ms.torch_gen.get_state()))
    rms.MSRun._init_digest(ms.agent)
    assert torch.equal(state_before[0], torch.get_rng_state()) and np.array_equal(state_before[1], np.random.get_state()[1])
    assert torch.equal(state_before[2], ms.torch_gen.get_state())
    for net, name in ((ms.agent.critic, "out.bias"), (ms.agent.critic, "out.weight"), (ms.agent.critic, "l1.weight"),
                      (ms.agent.actor, "l1.bias"), (ms.agent.actor, "l2.weight"), (ms.agent.actor, "out.weight")):
        p = dict(net.named_parameters())[name]
        old = p.detach().clone()
        with torch.no_grad():
            p.view(-1)[0] += 1e-3
        assert rms.MSRun._init_digest(ms.agent) != base, name
        with torch.no_grad():
            p.copy_(old)
        assert rms.MSRun._init_digest(ms.agent) == base, name


@pytest.mark.parametrize("bad", [{"actor_variant": "t1"}, {"actor_variant": "gelu"}, {"actor_variant": 3},
                                 {"init_digest": False}, {"init_digest": "yes"}])
def test_configuration_refusals(tmp_path, bad):
    cfg = reduced_nl("NL_bb_s1", str(tmp_path / "x"))
    cfg.update(bad)
    with pytest.raises(ConfigError):
        rms.MSRun(cfg, str(tmp_path / "x"))


# ---------------------------------------------------------------------------------------------- 4. runs with a variant
@pytest.fixture(scope="module")
def variant_runs(tmp_path_factory) -> Dict[str, str]:
    """Reduced NL_st_s4 runs (ramp inside the budget) for t1, relu and t10 with init_digest."""
    root = tmp_path_factory.mktemp("variant_runs")
    out: Dict[str, str] = {}
    for v in ("t1",) + VARIANTS:
        d = str(root / v)
        run_cfg(with_variant(reduced_nl("NL_st_s4", d), v), d)
        out[v] = d
    return out


def test_exports_carry_the_variant_and_t1_exports_do_not(variant_runs):
    for v, d in variant_runs.items():
        for f in sorted(os.listdir(os.path.join(d, "weights"))):
            w = weights(d, int(f[1:6]))
            if v == "t1":
                assert "actor_variant" not in w
            else:
                assert str(np.asarray(w["actor_variant"])) == v, (v, f)
        man = json.load(open(os.path.join(d, "manifest.json")))
        assert man.get("actor_variant", "t1") == v
        assert ("actor_variant" in man) == (v != "t1")
    digests = {json.load(open(os.path.join(d, "manifest.json")))["init_state_sha256"] for d in variant_runs.values()}
    assert len(digests) == 1                                           # C-INIT on the recorded digests


def test_the_variants_train_to_different_functions_from_the_same_start(variant_runs):
    """Same seed, same init, same RNG streams: the runs differ (the variant is not a no-op) from the first export."""
    t1 = variant_runs["t1"]
    for v in VARIANTS:
        assert not same_networks(t1, variant_runs[v], 5)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_numpy_reload_reproduces_the_torch_forward_and_never_the_tanh_forward(variant_runs, variant):
    d = variant_runs[variant]
    state = torch.load(os.path.join(d, "state_end_stage2.pt"), weights_only=False)
    assert state["agent"]["actor_variant"] == variant
    net = BetaActor(64, 100.0, 1e-6, torch.Generator().manual_seed(0))
    net.load_state_dict(state["agent"]["actor"])
    net.variant = variant
    dd = np.linspace(-110.0, 110.0, 45)
    x = np.stack([np.ones_like(dd), dd / 110.0], 1).astype(np.float32)
    with torch.no_grad():
        a, b = net(torch.as_tensor(x))
    w = weights(d, 50)                                                  # an export of the (scaled) terminal stage
    # the export of u50 is not the end state: compare the reload with a torch actor built from THAT export
    arrays = {k: v for k, v in w.items()}
    built = rdr.build_actor(arrays)
    assert built.variant == variant
    with torch.no_grad():
        ta, tb = built(torch.as_tensor(x))
    e_np, a_np, b_np = mean_effort_numpy(arrays, x)
    np.testing.assert_allclose(a_np, ta.numpy(), rtol=5e-5)
    np.testing.assert_allclose(b_np, tb.numpy(), rtol=5e-5)
    e_t = 100.0 * (ta / (ta + tb)).numpy().astype(float)
    np.testing.assert_allclose(e_np, e_t, atol=1e-3)
    # an export stripped of its variant name is the tanh actor: the reload would be wrong (the reason the entry exists)
    stripped = {k: v for k, v in arrays.items() if k != "actor_variant"}
    e_tanh, _, _ = mean_effort_numpy(stripped, x)
    assert np.abs(e_tanh - e_np).max() > 1e-2
    with pytest.raises(ValueError):
        mean_effort_numpy({**arrays, "actor_variant": np.asarray("gelu")}, x)
    with pytest.raises(ValueError):
        rdr.build_actor({**arrays, "actor_variant": np.asarray("gelu")})
    # the explicit argument overrides the entry (a documented use for the screen / diagnostics)
    e_arg, _, _ = mean_effort_numpy(stripped, x, variant=variant)
    np.testing.assert_allclose(e_arg, e_np, atol=0.0)


def test_the_residual_asymmetry_example_refuses_a_non_t1_export():
    src = (ROOT / "tools" / "ms" / "residual_asymmetry_example.py").read_text()
    assert "actor_variant" in src and "t1 exports only" in src


def test_opponent_refresh_and_frozen_snapshot_carry_the_variant(tmp_path):
    for variant in VARIANTS:
        out = str(tmp_path / variant)
        cfg = with_variant(reduced_nl("NL_bb_s4", out), variant)
        os.makedirs(out)
        ms = rms.MSRun(cfg, out)
        seen: List[Tuple[str, str]] = []
        orig_update, orig_refresh = ms.agent.update, ms.agent.refresh_snapshot

        def spy_update(*a: Any, **k: Any) -> Any:
            seen.append((ms.agent.actor.variant, ms.agent.opponent.variant))
            return orig_update(*a, **k)

        refreshed: List[Tuple[str, str]] = []

        def spy_refresh() -> None:
            orig_refresh()
            refreshed.append((ms.agent.actor.variant, ms.agent.opponent.variant))
        ms.agent.update, ms.agent.refresh_snapshot = spy_update, spy_refresh
        rec = ms.run_stage(2)
        assert seen and all(a == o == variant for a, o in seen)
        assert refreshed and all(a == o == variant for a, o in refreshed)
        ms.freeze_stage(2, rec, lambda point: None)
        assert ms.frozen_nets[2].variant == variant
        fs = ms.full_state("stage2")
        assert fs["actor_variant"] == variant and fs["agent"]["actor_variant"] == variant
        seen.clear()
        refreshed.clear()
        rms.MSRun.run_stage(ms, 1)                          # stage 1 trains the same (shared) actor
        assert all(a == o == variant for a, o in seen) and all(a == o == variant for a, o in refreshed)
        assert ms.frozen_nets[2].variant == variant


def test_the_agent_full_state_roundtrip_restores_the_variant():
    from agents.ppo_curriculum_v2 import CurriculumPPOv2
    a = CurriculumPPOv2(PPOConfig(), torch.Generator().manual_seed(2), np.random.default_rng(0))
    a.set_actor_variant("t10")
    st = a.full_state()
    assert st["actor_variant"] == "t10"
    b = CurriculumPPOv2(PPOConfig(), torch.Generator().manual_seed(9), np.random.default_rng(1))
    b.freeze_stage2_snapshot()
    st["frozen"] = {k: v.clone() for k, v in a.actor.state_dict().items()}
    b.load_full_state(st)
    assert b.actor.variant == b.opponent.variant == b.frozen.variant == "t10"
    plain = CurriculumPPOv2(PPOConfig(), torch.Generator().manual_seed(2), np.random.default_rng(0)).full_state()
    assert "actor_variant" not in plain                     # a t1 state is the state of every earlier round


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_stage_1_table_of_a_variants_frozen_snapshot_equals_direct_evaluation(variant_runs, variant, tmp_path):
    """The table built by the runner from the frozen terminal-stage actor equals an independent evaluation through the
    numpy reload of an export of the same weights (Beta means, composite Gauss-Legendre in the shock)."""
    d = variant_runs[variant]
    z = np.load(os.path.join(d, "continuation_table_stage1.npz"))
    y, vals = z["y_grid"], z["values"]
    state = torch.load(os.path.join(d, "state_end_stage2.pt"), weights_only=False)
    arrays = {f"actor.{k}": v.numpy() for k, v in state["agent"]["actor"].items()}
    arrays["actor_variant"] = np.asarray(variant)
    arrays["conc_scale"] = np.asarray(float(state["agent"]["conc_scale"]["actor"]), dtype=np.float64) \
        if "conc_scale" in state["agent"] else np.asarray(1.0)
    cfg = reduced_nl("NL_st_s4", str(tmp_path / "x"))
    ms = rms.MSRun(with_variant(cfg, variant), str(tmp_path / "x"))
    spec = ms.spec
    nodes, wts, _ = shock_quadrature(spec.q, 1.0, 6)
    for yv in (-40.0, -3.0, 0.0, 7.5, 33.0):
        dd = yv + nodes
        e_own = mean_effort_numpy(arrays, spec.encode_obs(2, dd), e_min=spec.e_min, e_max=spec.e_max)[0]
        e_opp = mean_effort_numpy(arrays, spec.encode_obs(2, -dd), e_min=spec.e_min, e_max=spec.e_max)[0]
        direct = float(terminal_value(spec, dd, e_own, e_opp) @ wts)
        table = float(np.interp(yv, y, vals))
        assert abs(direct - table) <= 1e-3 * spec.dw, (variant, yv, direct, table)
    # and the table differs from the t1 run's (the variant is in the function, not only in the label)
    t1 = np.load(os.path.join(variant_runs["t1"], "continuation_table_stage1.npz"))["values"]
    assert np.abs(t1 - vals).max() > 1e-6


# ---------------------------------------------------------------------------------------------- 5. C-NL and T=3
@pytest.mark.parametrize("variant", VARIANTS)
def test_c_nl_holds_for_each_variant(tmp_path, variant):
    """s = 4 equals s = 1 through the first ramp update and differs afterwards, for relu and t10."""
    d1, d4 = str(tmp_path / "s1"), str(tmp_path / "s4")
    run_cfg(with_variant(reduced_nl("NL_bb_s1", d1), variant), d1)
    run_cfg(with_variant(reduced_nl("NL_bb_s4", d4), variant), d4)
    for u in (5, 10, 15, 20):
        assert same_weights(d1, d4, u) and same_networks(d1, d4, u), u
    assert not same_networks(d1, d4, 25)                      # the networks have moved apart (not merely the labels)
    a, b = read_csv(os.path.join(d1, "ms_updates.csv")), read_csv(os.path.join(d4, "ms_updates.csv"))
    skip = {"update_wall_sec", "verifier_sec", "diag_sec", "run", "arm", "conc_scale"}
    cols = (set(a[0]) & set(b[0])) - skip
    assert all(x[c] == y[c] for x, y in zip(a[:21], b[:21]) for c in cols)


@pytest.mark.parametrize("variant", VARIANTS)
def test_t3_smoke_with_a_variant(tmp_path, variant):
    """The reduced T = 3 pipeline (phases 3, 2, 1; nested tables) runs with a relu / t10 actor."""
    d = str(tmp_path / "t3")
    os.makedirs(d)
    cfg = make_cfg(d, T=3, params=_t3_params(False), epu=48)
    cfg["actor_variant"] = variant
    assert rms.run_pipeline(cfg, d, "pytest") == 0
    for t in (3, 2, 1):
        assert os.path.exists(os.path.join(d, f"state_end_stage{t}.pt"))
        assert torch.load(os.path.join(d, f"state_end_stage{t}.pt"), weights_only=False)["actor_variant"] == variant
    assert os.path.exists(os.path.join(d, "continuation_table_stage2.npz"))
    assert os.path.exists(os.path.join(d, "continuation_table_stage1.npz"))
    g = json.load(open(os.path.join(d, "gates.json")))
    assert g["global_rng"]["status"] == "ok"
    w = [f for f in sorted(os.listdir(os.path.join(d, "weights")))]
    assert w and all(str(np.asarray(np.load(os.path.join(d, "weights", f))["actor_variant"])) == variant for f in w)
