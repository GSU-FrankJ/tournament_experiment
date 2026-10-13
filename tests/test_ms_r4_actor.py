"""MS-R4 actor tests: the ``lrsm`` variant (LeakyReLU 0.01 hidden units, mean m + (1 - 2 m) sigmoid(z0), no clamp) follows D3,
its initial weights are those of ``t1`` and ``relu`` (C-INIT), every loader used in this round builds its forward or refuses
its export, the opponent / refresh / frozen snapshot / stage-1 table carry it, and the defaults (``t1`` / ``relu`` / ``t10``,
every earlier arm) are bit-identical to the base commit ``be2881ff``.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_r4_actor.py -p no:cacheprovider -q
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ms_r4_helpers as H  # noqa: E402,I001
from ms_r4_helpers import ROOT, reduced_r4  # noqa: E402

import ms_configs as mc  # noqa: E402
import replay_dev_rule as rdr  # noqa: E402
import r3_analysis as R3  # noqa: E402
import r3_actor_diagnostics as R3D  # noqa: E402
from agents import ppo_curriculum as pc  # noqa: E402
from agents.ppo_curriculum import (  # noqa: E402
    ACTOR_VARIANTS, ACTOR_VARIANTS_R4, ALL_ACTOR_VARIANTS, BetaActor, CurriculumPPO, PPOConfig, mean_effort_numpy)
from run import run_ms_stagewise as rms  # noqa: E402
from run.run_final_dp_br_round3_dense import ConfigError  # noqa: E402
from test_ms_r2_runner import _same_run_outputs, reduced_nl, run_cfg, run_in_tree, weights  # noqa: E402
from test_ms_runner import _t3_params, make_cfg  # noqa: E402
from utils import ms_continuation as mcont  # noqa: E402
from utils.v2_continuation import shock_quadrature, terminal_value  # noqa: E402


def _random_actor(variant: str, seed: int = 5, out_scale: float = 0.3) -> BetaActor:
    net = BetaActor(64, 100.0, 1e-6, torch.Generator().manual_seed(seed))
    g = torch.Generator().manual_seed(seed + 1)
    net.out.weight.data = torch.randn(net.out.weight.shape, generator=g) * out_scale
    net.out.bias.data = torch.randn(net.out.bias.shape, generator=g) * 0.1
    net.l1.bias.data = torch.randn(net.l1.bias.shape, generator=g) * 0.1
    net.variant = variant
    return net


def _x(n: int = 41) -> torch.Tensor:
    d = np.linspace(-120.0, 120.0, n)
    return torch.as_tensor(np.stack([np.ones_like(d), d / 120.0], 1).astype(np.float32))


def _manual_lrsm(net: BetaActor, x: np.ndarray) -> np.ndarray:
    """D3 written out in float64 numpy: (alpha, beta) of the lrsm actor."""
    w = {k: v.detach().double().numpy() for k, v in net.state_dict().items()}
    act = lambda a: np.where(a > 0, a, 0.01 * a)  # noqa: E731
    h = act(np.asarray(x, dtype=np.float64) @ w["l1.weight"].T + w["l1.bias"])
    h = act(h @ w["l2.weight"].T + w["l2.bias"])
    z = h @ w["out.weight"].T + w["out.bias"]
    m = 1e-6
    mu = m + (1.0 - 2.0 * m) / (1.0 + np.exp(-z[:, 0]))
    c = 100.0 + np.logaddexp(0.0, z[:, 1])
    return np.stack([mu * c, (1.0 - mu) * c], 1)


# ---------------------------------------------------------------------------------------------- 1. registration
def test_the_variant_tuples_are_registered_without_touching_the_r3_tuple():
    assert ACTOR_VARIANTS == ("t1", "relu", "t10")                       # MS-R3's tests iterate it
    assert ACTOR_VARIANTS_R4 == ("lrsm",)
    assert ALL_ACTOR_VARIANTS == ACTOR_VARIANTS + ACTOR_VARIANTS_R4
    assert tuple(R3.ACTORS) == ACTOR_VARIANTS                            # tools/ms/r3_analysis.R3.ACTORS is compared with it
    assert pc.LRSM_NEG_SLOPE == 0.01


def test_set_actor_variant_accepts_lrsm_and_refuses_others():
    ag = CurriculumPPO(PPOConfig(), torch.Generator().manual_seed(1), np.random.default_rng(0))
    ag.set_actor_variant("lrsm")
    assert ag.actor.variant == ag.opponent.variant == "lrsm"
    with pytest.raises(ValueError):
        ag.set_actor_variant("gelu")
    ag.actor.variant = "gelu"
    with pytest.raises(ValueError):
        ag.actor(torch.zeros(2, 2))


# ---------------------------------------------------------------------------------------------- 2. the forward of D3
def test_the_lrsm_forward_matches_d3():
    x = _x()
    net = _random_actor("lrsm")
    a, b = net(x)
    got = np.stack([a.detach().numpy(), b.detach().numpy()], 1)
    np.testing.assert_allclose(got, _manual_lrsm(net, x.numpy()), rtol=3e-5)


def test_lrsm_hidden_units_are_leaky_with_slope_001_in_both_layers():
    """A negative pre-activation is scaled by 0.01 (not zeroed, not left alone), in the first and in the second layer."""
    net = _random_actor("lrsm")
    x = _x()
    h = net._hidden_variant(x)
    z1 = net.l1(x)
    h1 = torch.nn.functional.leaky_relu(z1, 0.01)
    assert (z1 < 0).any() and torch.equal(h1, torch.where(z1 > 0, z1, z1 * 0.01))
    z2 = net.l2(h1)
    assert (z2 < 0).any() and torch.equal(h, torch.nn.functional.leaky_relu(z2, 0.01))
    relu = _random_actor("relu")
    assert not torch.equal(relu._hidden_variant(x), h)


@pytest.mark.parametrize("z0", [-20.0, 15.0])
def test_the_mean_has_a_gradient_where_the_clamp_gives_none(z0):
    """z0 = -20 and +15: under the current clamp d mean / d z0 is exactly 0; the lrsm mean is smooth there."""
    grads = {}
    for v in ("t1", "relu", "lrsm"):
        net = BetaActor(64, 100.0, 1e-6, torch.Generator().manual_seed(2))
        net.variant = v
        net.out.weight.data.zero_()
        net.out.bias.data = torch.tensor([z0, 0.0])
        a, b = net(_x(5))
        (a / (a + b)).sum().backward()
        grads[v] = float(net.out.bias.grad[0])
    assert grads["t1"] == 0.0 and grads["relu"] == 0.0
    assert grads["lrsm"] > 0.0


def test_the_lrsm_mean_range_and_the_float32_saturation_points():
    """The range is that of the current clamp; the derivative is positive for about -88 < z0 < 17 and exactly 0 beyond."""
    net = BetaActor(64, 100.0, 1e-6, torch.Generator().manual_seed(2))
    net.variant = "lrsm"
    net.out.weight.data.zero_()
    pos = {}
    for z0 in (-80.0, -30.0, 0.0, 16.0, 17.0, 18.0, 25.0, -100.0):
        net.out.bias.data = torch.tensor([z0, 0.0])
        net.out.bias.grad = None
        a, b = net(_x(3))
        (a / (a + b)).sum().backward()
        mu = float((a / (a + b))[0])
        assert 1e-6 - 1e-9 <= mu <= 1.0 - 1e-6 + 2e-7, (z0, mu)
        pos[z0] = float(net.out.bias.grad[0]) > 0.0
    assert pos[-80.0] and pos[-30.0] and pos[0.0] and pos[16.0]
    assert not pos[18.0] and not pos[25.0] and not pos[-100.0]


# ---------------------------------------------------------------------------------------------- 3. numpy reload and loaders
def _export_arrays(net: BetaActor, scale: float = 1.0) -> Dict[str, np.ndarray]:
    ag = CurriculumPPO(PPOConfig(), torch.Generator().manual_seed(3), np.random.default_rng(0))
    ag.actor.load_state_dict(net.state_dict())
    ag.actor.conc_scale = scale
    ag.set_actor_variant(net.variant)
    p = Path(os.environ.get("TMPDIR", "/tmp")) / f"ms_r4_export_{os.getpid()}_{id(net)}.npz"
    ag.export_weights_npz(str(p))
    try:
        with np.load(p) as z:
            return {k: z[k] for k in z.files}
    finally:
        p.unlink()


@pytest.mark.parametrize("out_scale", [0.3, 6.0])
def test_the_numpy_reload_reproduces_the_torch_forward_to_float32_rounding(out_scale):
    """Including a saturating mean (out_scale 6 pushes z0 past the clamp range)."""
    net = _random_actor("lrsm", out_scale=out_scale)
    arrays = _export_arrays(net, scale=4.0)
    assert str(np.asarray(arrays["actor_variant"])) == "lrsm"
    x = _x(61)
    net.conc_scale = 4.0
    with torch.no_grad():
        ta, tb = net(x)
    e, a_np, b_np = mean_effort_numpy(arrays, x.numpy())
    np.testing.assert_allclose(a_np, ta.numpy(), rtol=5e-5)
    np.testing.assert_allclose(b_np, tb.numpy(), rtol=5e-5)
    np.testing.assert_allclose(e, 100.0 * (ta / (ta + tb)).numpy().astype(float), atol=2e-3)


def test_the_numpy_reload_never_reads_an_lrsm_export_as_the_tanh_or_relu_actor():
    net = _random_actor("lrsm")
    arrays = _export_arrays(net)
    x = _x(31).numpy()
    e_l, _, _ = mean_effort_numpy(arrays, x)
    stripped = {k: v for k, v in arrays.items() if k != "actor_variant"}
    assert np.abs(mean_effort_numpy(stripped, x)[0] - e_l).max() > 1e-2                 # the tanh reading is a different function
    assert np.abs(mean_effort_numpy(stripped, x, variant="relu")[0] - e_l).max() > 1e-4  # so is the relu one
    np.testing.assert_allclose(mean_effort_numpy(stripped, x, variant="lrsm")[0], e_l, atol=0.0)
    with pytest.raises(ValueError):
        mean_effort_numpy({**arrays, "actor_variant": np.asarray("gelu")}, x)


def test_the_loaders_of_this_round_build_the_lrsm_forward_or_refuse():
    net = _random_actor("lrsm", out_scale=2.0)
    arrays = _export_arrays(net, scale=2.0)
    built = rdr.build_actor(arrays)                       # tools/ms/replay_dev_rule.build_actor: the loader behind r4_curves
    assert built.variant == "lrsm" and built.conc_scale == 2.0
    x = _x(31)
    net.conc_scale = 2.0
    with torch.no_grad():
        a0, b0 = net(x)
        a1, b1 = built(x)
    assert torch.equal(a0, a1) and torch.equal(b0, b1)    # bit for bit: the same float32 operations
    with pytest.raises(ValueError):
        rdr.build_actor({**arrays, "actor_variant": np.asarray("gelu")})
    # loaders that are not used on MS-R4 exports refuse an lrsm export (never a silent tanh / relu reading)
    with pytest.raises(ValueError):
        R3D.export_variant(arrays)


def test_the_residual_asymmetry_example_refuses_an_lrsm_export(tmp_path, monkeypatch):
    """The MS-R1 example rebuilds the tanh d / B actor only: an lrsm export is refused before any verifier call."""
    from test_ms_r3_actor import _write_stored_export
    _write_stored_export(tmp_path, "lrsm")
    import residual_asymmetry_example as rae
    calls: List[Any] = []
    monkeypatch.setattr(rae, "verify", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(sys, "argv", ["residual_asymmetry_example.py", "--root-v2", str(tmp_path), "--q", "50", "--seed", "10501"])
    with pytest.raises(ValueError, match="t1 exports only") as e:
        rae.main()
    assert "'lrsm'" in str(e.value) and not calls


def test_no_new_tool_imports_a_tools_v2_module():
    """The loaders of tools/v2 are unchanged and are not used on MS-R4 exports: no tools/ms/r4_*.py (nor utils/ms_focus.py)
    imports a module of tools/v2 (the comparison helpers reach the tools only through tools/ms/launch_checks, which is
    unchanged)."""
    import ast
    v2 = {p.stem for p in (ROOT / "tools" / "v2").glob("*.py")}
    assert {"refine_analysis", "diag_seed30510"} <= v2
    for f in sorted((ROOT / "tools" / "ms").glob("r4_*.py")) + [ROOT / "utils" / "ms_focus.py"]:
        tree = ast.parse(f.read_text())
        mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)} | \
               {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not (mods & v2), (f.name, sorted(mods & v2))


def test_the_curve_tool_refuses_an_export_of_an_unknown_variant():
    import r4_curves
    arrays = _export_arrays(_random_actor("lrsm"))
    cfg = reduced_r4("lrsm_st_s16", "/unused")
    spec, ppo = r4_curves._spec_ppo(cfg)
    with pytest.raises(ValueError):
        r4_curves.evaluate_export({**arrays, "actor_variant": np.asarray("gelu")}, spec, ppo, 0.5)


# ---------------------------------------------------------------------------------------------- 4. C-INIT, configuration
def test_initial_weights_are_identical_across_t1_relu_and_lrsm(tmp_path):
    states: Dict[str, Tuple[Dict[str, torch.Tensor], str]] = {}
    for arm in ("t1_st_s16", "relu_st_s16", "lrsm_st_s16"):
        out = str(tmp_path / arm)
        ms = rms.MSRun(reduced_r4(arm, out), out)
        sd = {f"{n}.{k}": t.clone() for n, net in (("actor", ms.agent.actor), ("critic", ms.agent.critic))
              for k, t in net.state_dict().items()}
        states[arm] = (sd, ms.init_state_sha256)
        want = arm.split("_")[0]
        assert ms.actor_variant == want and ms.agent.actor.variant == want and ms.agent.opponent.variant == want
    ref = states["t1_st_s16"]
    for arm in ("relu_st_s16", "lrsm_st_s16"):
        assert states[arm][1] == ref[1] and all(torch.equal(states[arm][0][k], ref[0][k]) for k in ref[0])
    out = str(tmp_path / "other")
    other = rms.MSRun(reduced_r4("lrsm_st_s16", out, seed=10512), out)
    assert other.init_state_sha256 != ref[1]


@pytest.mark.parametrize("bad", [{"actor_variant": "t1"}, {"actor_variant": "gelu"}, {"actor_variant": 3}])
def test_actor_variant_refusals_and_acceptance(tmp_path, bad):
    cfg = reduced_r4("t1_st_s16", str(tmp_path / "x"))
    cfg.update(bad)
    with pytest.raises(ConfigError):
        rms.validate_config(cfg)
    ok = reduced_r4("lrsm_st_s16", str(tmp_path / "y"))
    assert ok["actor_variant"] == "lrsm"
    rms.validate_config(ok)


# ---------------------------------------------------------------------------------------------- 5. runs with the variants
@pytest.fixture(scope="module")
def variant_runs(tmp_path_factory) -> Dict[str, str]:
    root = tmp_path_factory.mktemp("r4_variant_runs")
    out: Dict[str, str] = {}
    for arm in ("t1_st_s16", "relu_st_s16", "lrsm_st_s16"):
        d = str(root / arm)
        run_cfg(reduced_r4(arm, d), d)
        out[arm] = d
    return out


def test_exports_manifest_config_and_state_carry_the_variant(variant_runs):
    for arm, d in variant_runs.items():
        v = arm.split("_")[0]
        for f in sorted(os.listdir(os.path.join(d, "weights"))):
            w = weights(d, int(f[1:6]))
            if v == "t1":
                assert "actor_variant" not in w
            else:
                assert str(np.asarray(w["actor_variant"])) == v, (arm, f)
        man = json.load(open(os.path.join(d, "manifest.json")))
        cfg = json.load(open(os.path.join(d, "run_config.json")))
        assert man.get("actor_variant", "t1") == v == cfg.get("actor_variant", "t1")
        assert ("actor_variant" in man) == ("actor_variant" in cfg) == (v != "t1")
        st = torch.load(os.path.join(d, "state_end_stage2.pt"), weights_only=False)
        assert st.get("actor_variant", "t1") == v and st["agent"].get("actor_variant", "t1") == v
    digests = {json.load(open(os.path.join(d, "manifest.json")))["init_state_sha256"] for d in variant_runs.values()}
    assert len(digests) == 1                                           # C-INIT on the recorded digests


def test_the_lrsm_run_trains_to_a_different_function_from_the_same_start(variant_runs):
    a, b, c = (weights(variant_runs[k], 5) for k in ("t1_st_s16", "relu_st_s16", "lrsm_st_s16"))
    for x, y in ((a, c), (b, c), (a, b)):
        assert any(not np.array_equal(x[k], y[k]) for k in x if k.startswith("actor."))


def test_the_stage_1_table_of_the_lrsm_snapshot_equals_direct_evaluation(variant_runs, tmp_path):
    d = variant_runs["lrsm_st_s16"]
    z = np.load(os.path.join(d, "continuation_table_stage1.npz"))
    y, vals = z["y_grid"], z["values"]
    state = torch.load(os.path.join(d, "state_end_stage2.pt"), weights_only=False)
    arrays = {f"actor.{k}": v.numpy() for k, v in state["agent"]["actor"].items()}
    arrays["actor_variant"] = np.asarray("lrsm")
    arrays["conc_scale"] = np.asarray(float(state["agent"]["conc_scale"]["actor"]), dtype=np.float64) \
        if "conc_scale" in state["agent"] else np.asarray(1.0)
    ms = rms.MSRun(reduced_r4("lrsm_st_s16", str(tmp_path / "x")), str(tmp_path / "x"))
    spec = ms.spec
    nodes, wts, _ = shock_quadrature(spec.q, 1.0, 6)
    for yv in (-40.0, -3.0, 0.0, 7.5, 33.0):
        dd = yv + nodes
        e_own = mean_effort_numpy(arrays, spec.encode_obs(2, dd), e_min=spec.e_min, e_max=spec.e_max)[0]
        e_opp = mean_effort_numpy(arrays, spec.encode_obs(2, -dd), e_min=spec.e_min, e_max=spec.e_max)[0]
        direct = float(terminal_value(spec, dd, e_own, e_opp) @ wts)
        assert abs(direct - float(np.interp(yv, y, vals))) <= 1e-3 * spec.dw, (yv, direct)
    t1 = np.load(os.path.join(variant_runs["t1_st_s16"], "continuation_table_stage1.npz"))["values"]
    assert np.abs(t1 - vals).max() > 1e-6                                # the variant is in the function, not only in the label


def test_opponent_refresh_and_frozen_snapshot_carry_lrsm(tmp_path):
    out = str(tmp_path / "x")
    os.makedirs(out)
    ms = rms.MSRun(reduced_r4("lrsm_st_s16", out), out)
    seen: List[Tuple[str, str]] = []
    refreshed: List[Tuple[str, str]] = []
    orig_update, orig_refresh = ms.agent.update, ms.agent.refresh_snapshot

    def spy_update(*a: Any, **k: Any) -> Any:
        seen.append((ms.agent.actor.variant, ms.agent.opponent.variant))
        return orig_update(*a, **k)

    def spy_refresh() -> None:
        orig_refresh()
        refreshed.append((ms.agent.actor.variant, ms.agent.opponent.variant))
    ms.agent.update, ms.agent.refresh_snapshot = spy_update, spy_refresh
    rec = ms.run_stage(2)
    assert seen and all(a == o == "lrsm" for a, o in seen) and refreshed and all(a == o == "lrsm" for a, o in refreshed)
    ms.freeze_stage(2, rec, lambda point: None)
    assert ms.frozen_nets[2].variant == "lrsm"
    fs = ms.full_state("stage2")
    assert fs["actor_variant"] == "lrsm" and fs["agent"]["actor_variant"] == "lrsm"
    seen.clear()
    refreshed.clear()
    rms.MSRun.run_stage(ms, 1)
    assert all(a == o == "lrsm" for a, o in seen) and all(a == o == "lrsm" for a, o in refreshed)
    assert ms.frozen_nets[2].variant == "lrsm"


def test_the_agent_full_state_roundtrip_restores_lrsm():
    from agents.ppo_curriculum_v2 import CurriculumPPOv2
    a = CurriculumPPOv2(PPOConfig(), torch.Generator().manual_seed(2), np.random.default_rng(0))
    a.set_actor_variant("lrsm")
    st = a.full_state()
    assert st["actor_variant"] == "lrsm"
    b = CurriculumPPOv2(PPOConfig(), torch.Generator().manual_seed(9), np.random.default_rng(1))
    b.freeze_stage2_snapshot()
    st["frozen"] = {k: v.clone() for k, v in a.actor.state_dict().items()}
    b.load_full_state(st)
    assert b.actor.variant == b.opponent.variant == b.frozen.variant == "lrsm"


def test_t3_smoke_with_lrsm(tmp_path):
    d = str(tmp_path / "t3")
    os.makedirs(d)
    cfg = make_cfg(d, T=3, params=_t3_params(False), epu=48)
    cfg["actor_variant"] = "lrsm"
    assert rms.run_pipeline(cfg, d, "pytest") == 0
    for t in (3, 2, 1):
        assert torch.load(os.path.join(d, f"state_end_stage{t}.pt"), weights_only=False)["actor_variant"] == "lrsm"
    assert os.path.exists(os.path.join(d, "continuation_table_stage1.npz"))
    assert json.load(open(os.path.join(d, "gates.json")))["global_rng"]["status"] == "ok"


# ---------------------------------------------------------------------------------------------- 6. defaults == be2881ff
@pytest.fixture(scope="module")
def base_tree(tmp_path_factory) -> str:
    try:
        return H.base_tree(tmp_path_factory.mktemp("tree_be2881ff"))
    except subprocess.CalledProcessError:
        pytest.skip(f"commit {H.BASE_COMMIT} is not available in this clone")


@pytest.fixture(scope="module")
def base_module(base_tree):
    """The ``agents/ppo_curriculum.py`` of the base commit loaded as a module of its own."""
    src = open(os.path.join(base_tree, "agents", "ppo_curriculum.py")).read()
    mod = types.ModuleType("ppo_curriculum_base")
    sys.modules["ppo_curriculum_base"] = mod              # the dataclass machinery looks the module up by name
    exec(compile(src, "ppo_curriculum_base.py", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("variant", ["t1", "relu", "t10"])
@pytest.mark.parametrize("out_scale", [0.3, 8.0])
def test_the_forwards_of_t1_relu_and_t10_are_unchanged_bit_for_bit(base_module, variant, out_scale):
    """Same weights, same inputs (including a saturating mean where the clamp is active): bitwise equal (alpha, beta) in the base
    commit's module and in the current one, and the clamp is still in the path."""
    old = base_module.BetaActor(64, 100.0, 1e-6, torch.Generator().manual_seed(5))
    new = _random_actor(variant, out_scale=out_scale)
    old.load_state_dict(new.state_dict())
    old.variant = variant
    for scale in (1.0, 4.0):
        old.conc_scale = new.conc_scale = scale
        x = _x(81)
        a0, b0 = old(x)
        a1, b1 = new(x)
        assert torch.equal(a0, a1) and torch.equal(b0, b1)
    new.out.bias.data = torch.tensor([30.0, 0.0])
    new.out.weight.data.zero_()
    a, b = new(_x(5))
    assert float((a / (a + b))[0]) <= 1.0 - 1e-6 + 1e-9        # the upper clamp bound is intact for these variants


@pytest.mark.parametrize("mk", ["nl_st_s16", "relu_st_s16", "fb_rx", "t1_st_s16"])
def test_without_the_new_variant_and_keys_a_run_equals_the_base_commit(base_tree, tmp_path, mk):
    """An MS-R2 NL arm, the R4 control arms (t1 and relu) and the R4 ``fb_rx`` arm (no new keys) end in the same state, with the
    same exports and series, in the current tree and in the tree of be2881ff."""
    cfg = reduced_nl("NL_st_s16", str(tmp_path / "x")) if mk == "nl_st_s16" else reduced_r4(mk, str(tmp_path / "x"))
    for k in ("focus_scope", "focus_map"):
        assert k not in cfg
    a, b = str(tmp_path / "now"), str(tmp_path / "old")
    run_cfg(copy.deepcopy(cfg), a)
    run_in_tree(base_tree, copy.deepcopy(cfg), b)
    _same_run_outputs(a, b)
    for f in sorted(os.listdir(os.path.join(a, "weights"))):
        assert ("actor_variant" in weights(a, int(f[1:6]))) == (mk == "relu_st_s16")
