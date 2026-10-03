"""Tests of the pathwise terminal fine-tuning step (method 5, ``agents/ppo_pathwise.py``).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest \
     tests/test_v2_refine_pathwise.py -p no:cacheprovider -q

The closed-form equilibrium (``g2_two_stage``) is used here as an evaluation reference only.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

from agents.ppo_curriculum import BetaActor, PPOConfig  # noqa: E402
from agents.ppo_curriculum_v2 import CurriculumPPOv2  # noqa: E402
from agents.ppo_pathwise import (  # noqa: E402
    effort_mean, expected_payoff, foc_residual, pathwise_loss, pathwise_step, torch_F_xi,
    torch_f_xi)
from envs.curriculum_env import GameSpec  # noqa: E402
from utils.theory_multistage import F_xi, f_xi, g2_two_stage  # noqa: E402

RECORDS = json.loads((ROOT / "protocols" / "v2_T2_locked_v1_1.json").read_text())["records"]
QS = (35, 50, 60)
STAGE = 2


def spec_for(q: int) -> GameSpec:
    """The locked-protocol game of this q (q=50 and q=60 records; q=35 reuses the q=50 prizes)."""
    rec = RECORDS["50" if str(q) not in RECORDS else str(q)]["game"]
    return GameSpec(**{**rec, "q": float(q)})


def make_agent(seed: int = 0) -> CurriculumPPOv2:
    """Fresh agent with the PPOConfig defaults, a seeded torch generator and numpy rng_mb."""
    gen = torch.Generator().manual_seed(seed)
    return CurriculumPPOv2(PPOConfig(), gen, np.random.default_rng(seed + 1))


def set_lr(agent: CurriculumPPOv2, lr: float) -> None:
    """Set the actor LR the way the runner does (param group)."""
    for g in agent.opt_actor.param_groups:
        g["lr"] = lr


def rows(spec: GameSpec, n: int = 512, seed: int = 7) -> np.ndarray:
    """Learner gaps drawn uniformly over the final-stage domain."""
    half = spec.domain_half(STAGE)
    return np.random.default_rng(seed).uniform(-half, half, n)


def preload_adam(agent: CurriculumPPOv2, spec: GameSpec, n_updates: int = 3) -> None:
    """A few standard PPO updates on a synthetic stage-2 buffer (nonzero Adam moments)."""
    rng = np.random.default_rng(3)
    for _ in range(n_updates):
        d = rows(spec, 512, int(rng.integers(1_000_000)))
        st = spec.encode_obs(STAGE, d)
        a, b = agent.beta_params(st)
        act = agent.sample_actions(a, b, rng)
        olp = agent.log_prob(a, b, act)
        ret = rng.normal(size=512).astype(np.float32)
        adv = rng.normal(size=512).astype(np.float32)
        agent.update(st, act, olp, ret, adv)
    agent.refresh_snapshot()


def snap(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Cloned state dict."""
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def same(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> bool:
    """Bit-exact equality of two state dicts."""
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def opt_state_clone(opt: torch.optim.Optimizer) -> list:
    """Deep copy of an optimizer's per-parameter state in parameter order."""
    params = [p for g in opt.param_groups for p in g["params"]]
    return [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in opt.state[p].items()}
            for p in params]


def opt_state_same(a: list, b: list) -> bool:
    """Bit-exact equality of two :func:`opt_state_clone` outputs."""
    if len(a) != len(b):
        return False
    for sa, sb in zip(a, b):
        if sa.keys() != sb.keys():
            return False
        for k in sa:
            if torch.is_tensor(sa[k]):
                if not torch.equal(sa[k], sb[k]):
                    return False
            elif sa[k] != sb[k]:
                return False
    return True


# ------------------------------------------------------------------ torch F_xi / f_xi
def dense_grid(q: float) -> np.ndarray:
    """Dense grid over [-4q, 4q] that contains the knots and their float neighbours."""
    knots = np.array([-4 * q, -2 * q, -q, 0.0, q, 2 * q, 4 * q])
    near = np.concatenate([np.nextafter(knots, np.inf), np.nextafter(knots, -np.inf)])
    return np.unique(np.concatenate([np.linspace(-4 * q, 4 * q, 400_001), knots, near]))


@pytest.mark.parametrize("q", [35, 50, 60])
def test_torch_F_xi_matches_numpy_on_dense_grid(q: int) -> None:
    """torch_F_xi equals utils.theory_multistage.F_xi to 1e-12; float64 is preserved."""
    x = dense_grid(float(q))
    out = torch_F_xi(torch.as_tensor(x, dtype=torch.float64), float(q))
    assert out.dtype == torch.float64
    ref = F_xi(x, float(q))
    assert float(np.max(np.abs(out.numpy() - ref))) <= 1e-12
    for knot, val in ((-2.0 * q, 0.0), (0.0, 0.5), (2.0 * q, 1.0)):
        assert float(torch_F_xi(torch.tensor([knot], dtype=torch.float64), float(q))[0]) == val
    assert out.shape == torch.Size(x.shape)


@pytest.mark.parametrize("q", [35, 50, 60])
def test_torch_F_xi_gradient_is_triangular_density(q: int) -> None:
    """Autograd gradient = triangular density (analytic and central finite difference)."""
    x_np = dense_grid(float(q))[::997]
    x = torch.as_tensor(x_np, dtype=torch.float64).requires_grad_(True)
    (g,) = torch.autograd.grad(torch_F_xi(x, float(q)).sum(), x)
    assert float(np.max(np.abs(g.numpy() - f_xi(x_np, float(q))))) <= 1e-14
    assert float(np.max(np.abs(torch_f_xi(x.detach(), float(q)).numpy()
                               - f_xi(x_np, float(q))))) <= 1e-15
    h = 1e-5
    xt = torch.as_tensor(x_np, dtype=torch.float64)
    fd = (torch_F_xi(xt + h, float(q)) - torch_F_xi(xt - h, float(q))) / (2.0 * h)
    assert float(np.max(np.abs(fd.numpy() - f_xi(x_np, float(q))))) <= 1e-8


def test_torch_F_xi_float32_input_is_computed_in_float64() -> None:
    """A float32 input is promoted; the result is float64."""
    out = torch_F_xi(torch.linspace(-150.0, 150.0, 11, dtype=torch.float32), 50.0)
    assert out.dtype == torch.float64


# ------------------------------------------------------------------ dR/de
@pytest.mark.parametrize("q", [35, 50, 60])
def test_dR_de_autograd_vs_finite_difference_vs_analytic(q: int) -> None:
    """Autograd dR/de = central FD (h=1e-5) = DW f_xi(y) - 2 k e on random states and efforts."""
    spec = spec_for(q)
    rng = np.random.default_rng(100 + q)
    n = 4000
    y = rng.uniform(-2.5 * q, 2.5 * q, n)          # y = d + e - e_opp covers the support
    e = rng.uniform(1.0, 99.0, n)
    d = rng.uniform(-3.0 * q, 3.0 * q, n)
    e_opp = d + e - y
    d_t, e_opp_t = (torch.as_tensor(v, dtype=torch.float64) for v in (d, e_opp))
    e_t = torch.as_tensor(e, dtype=torch.float64).requires_grad_(True)
    r = expected_payoff(spec, d_t, e_t, e_opp_t)
    (auto,) = torch.autograd.grad(r.sum(), e_t)

    h = 1e-5
    ep, em = (torch.as_tensor(e + s * h, dtype=torch.float64) for s in (1, -1))
    fd = (expected_payoff(spec, d_t, ep, e_opp_t) - expected_payoff(spec, d_t, em, e_opp_t)) / (
        2.0 * h)
    analytic = foc_residual(spec, d_t, e_t.detach(), e_opp_t)
    ref = spec.dw * f_xi(y, float(q)) - 2.0 * spec.k * e          # independent numpy expression
    scale = float(np.max(np.abs(ref)))
    assert float(np.max(np.abs(auto.numpy() - ref))) <= 1e-12 * max(scale, 1.0)
    assert float(np.max(np.abs(analytic.numpy() - ref))) <= 1e-12 * max(scale, 1.0)
    assert float(np.max(np.abs(fd.numpy() - ref))) <= 1e-8
    # the payoff itself against an independent numpy expression
    r_np = spec.w_l + spec.dw * F_xi(y, float(q)) - spec.k * e ** 2
    assert float(np.max(np.abs(r.detach().numpy() - r_np))) <= 1e-12


@pytest.mark.parametrize("q", [35, 50, 60])
def test_foc_vanishes_at_analytic_equilibrium_policy(q: int) -> None:
    """At e*(d) = g2(d) (even in d) the residual is 0 on the whole domain (evaluation reference)."""
    spec = spec_for(q)
    d = np.concatenate([np.linspace(-spec.domain_half(STAGE), spec.domain_half(STAGE), 4001),
                        np.linspace(-2.2 * q, 2.2 * q, 4001)])
    e_star = g2_two_stage(d, q, spec.w_h, spec.w_l, spec.k)
    e_opp = g2_two_stage(-d, q, spec.w_h, spec.w_l, spec.k)
    d_t, e_t, eo_t = (torch.as_tensor(v, dtype=torch.float64) for v in (d, e_star, e_opp))
    assert float(foc_residual(spec, d_t, e_t, eo_t).abs().max()) <= 1e-12
    e_leaf = e_t.clone().requires_grad_(True)
    (auto,) = torch.autograd.grad(expected_payoff(spec, d_t, e_leaf, eo_t).sum(), e_leaf)
    assert float(auto.abs().max()) <= 1e-12
    assert float(foc_residual(spec, d_t, e_t + 1.0, eo_t).abs().max()) > 1e-4   # not trivially 0


# ------------------------------------------------------------------ objective on the network
def test_effort_mean_is_bit_identical_to_the_numpy_mean_path() -> None:
    """Float64 mean from the float32 (alpha, beta) equals the rollout / evaluation expression."""
    spec = spec_for(50)
    agent = make_agent(1)
    preload_adam(agent, spec)
    obs = spec.encode_obs(STAGE, rows(spec, 300, 11))
    a, b = agent.beta_params(obs)
    m = a.astype(float) / (a.astype(float) + b.astype(float))
    ref = spec.e_min + spec.e_range * m
    with torch.no_grad():
        out = effort_mean(agent.actor, torch.as_tensor(obs), spec)
    assert out.dtype == torch.float64
    assert np.array_equal(out.numpy(), ref)


def test_loss_gradient_equals_foc_weighted_jacobian() -> None:
    """grad of the loss = -(1/N) sum_i dR/de_i * de_i/dtheta with the analytic dR/de."""
    spec = spec_for(50)
    agent = make_agent(2)
    preload_adam(agent, spec)
    d = rows(spec, 256, 21)
    params = list(agent.actor.parameters())
    loss, d_t, e, e_opp = pathwise_loss(agent, spec, d, STAGE)
    g_loss = torch.autograd.grad(loss, params, retain_graph=True)
    w = -foc_residual(spec, d_t, e.detach(), e_opp) / d.size
    g_chain = torch.autograd.grad(e, params, grad_outputs=w)
    assert float(loss.item()) == pytest.approx(
        -float(expected_payoff(spec, d_t, e.detach(), e_opp).mean()), rel=1e-14)
    for ga, gb, p in zip(g_loss, g_chain, params):
        scale = max(float(gb.abs().max()), 1e-12)
        assert float((ga - gb).abs().max()) <= 1e-5 * scale, tuple(p.shape)
    assert any(float(g.abs().max()) > 0.0 for g in g_loss)


def test_opponent_is_evaluated_at_minus_d_without_grad() -> None:
    """e_opp carries no graph and equals the opponent's mean at the observation of -d."""
    spec = spec_for(50)
    agent = make_agent(3)
    preload_adam(agent, spec)
    # make the lagged opponent differ from the live actor
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.add_(0.05)
    d = rows(spec, 64, 5)
    _, _, e, e_opp = pathwise_loss(agent, spec, d, STAGE)
    assert not e_opp.requires_grad and e.requires_grad
    a, b = agent.beta_params(spec.encode_obs(STAGE, -d), agent.opponent)
    ref = spec.e_min + spec.e_range * (a.astype(float) / (a.astype(float) + b.astype(float)))
    assert np.array_equal(e_opp.numpy(), ref)


# ------------------------------------------------------------------ pathwise_step
def test_25_steps_fresh_agent_contract() -> None:
    """Head, critic and opponent bit-identical; mean parameters move; loss and FOC decrease."""
    spec = spec_for(50)
    agent = make_agent(0)
    set_lr(agent, 1e-3)
    d = rows(spec)
    w1, b1 = agent.actor.out.weight[1].detach().clone(), agent.actor.out.bias[1].detach().clone()
    actor0, critic0, opp0 = snap(agent.actor), snap(agent.critic), snap(agent.opponent)
    opt_c0 = opt_state_clone(agent.opt_critic)
    outs = [pathwise_step(agent, spec, d, STAGE) for _ in range(25)]

    assert torch.equal(agent.actor.out.weight[1], w1) and torch.equal(agent.actor.out.bias[1], b1)
    assert same(snap(agent.critic), critic0)
    assert opt_state_same(opt_state_clone(agent.opt_critic), opt_c0)
    assert same(snap(agent.opponent), opp0)
    now = snap(agent.actor)
    assert not torch.equal(now["out.weight"][0], actor0["out.weight"][0])
    assert not torch.equal(now["out.bias"][0], actor0["out.bias"][0])
    assert not torch.equal(now["l2.weight"], actor0["l2.weight"])     # hidden layers move too
    assert not torch.equal(now["l1.weight"], actor0["l1.weight"])
    assert outs[-1]["loss"] < outs[0]["loss"]
    assert outs[-1]["foc_abs_mean"] < outs[0]["foc_abs_mean"]
    assert agent.opt_actor.param_groups[0]["lr"] == 1e-3                 # LR untouched
    assert set(outs[0]) == {"loss", "grad_norm_pre_clip", "foc_abs_mean", "foc_abs_max", "e0"}
    assert all(np.isfinite(v) for o in outs for v in o.values())
    assert outs[0]["foc_abs_max"] >= outs[0]["foc_abs_mean"] > 0.0
    assert outs[0]["e0"] == pytest.approx(50.0, abs=1e-9)                # zero-head init: mean 1/2


def test_head_stays_bit_identical_with_preloaded_adam_and_restore_is_needed() -> None:
    """With nonzero Adam moments on row 1 the head still does not move; zero-grad alone would."""
    spec = spec_for(50)
    agent = make_agent(4)
    preload_adam(agent, spec)
    exp_avg = agent.opt_actor.state[agent.actor.out.weight]["exp_avg"]
    assert float(exp_avg[1].abs().max()) > 0.0                           # precondition
    assert float(agent.opt_actor.state[agent.actor.out.bias]["exp_avg"][1].abs()) > 0.0
    set_lr(agent, 3e-5)
    d = rows(spec)

    # negative control: zero the row-1 gradient only (no restore) -> the head moves
    control = copy.deepcopy(agent)
    w_before = control.actor.out.weight[1].detach().clone()
    control.opt_actor.zero_grad(set_to_none=True)
    loss, *_ = pathwise_loss(control, spec, d, STAGE)
    loss.backward()
    control.actor.out.weight.grad[1].zero_()
    control.actor.out.bias.grad[1].zero_()
    control.opt_actor.step()
    assert not torch.equal(control.actor.out.weight[1], w_before)

    w1, b1 = agent.actor.out.weight[1].detach().clone(), agent.actor.out.bias[1].detach().clone()
    critic0, opp0 = snap(agent.critic), snap(agent.opponent)
    opt_c0 = opt_state_clone(agent.opt_critic)
    actor0 = snap(agent.actor)
    for _ in range(25):
        pathwise_step(agent, spec, d, STAGE)
    assert torch.equal(agent.actor.out.weight[1], w1) and torch.equal(agent.actor.out.bias[1], b1)
    assert same(snap(agent.critic), critic0)
    assert opt_state_same(opt_state_clone(agent.opt_critic), opt_c0)
    assert same(snap(agent.opponent), opp0)
    assert not torch.equal(snap(agent.actor)["out.weight"][0], actor0["out.weight"][0])


def test_step_consumes_no_random_numbers() -> None:
    """No numpy draw (any held generator, the agent's rng_mb, global) and no torch RNG draw."""
    spec = spec_for(50)
    gen = torch.Generator().manual_seed(9)
    agent = CurriculumPPOv2(PPOConfig(), gen, np.random.default_rng(10))
    set_lr(agent, 3e-4)
    held = [np.random.default_rng(s) for s in range(4)] + [agent.rng_mb]
    np_before = [copy.deepcopy(g.bit_generator.state) for g in held]
    np_global = np.random.get_state()
    t_global, t_gen = torch.get_rng_state(), gen.get_state()
    for _ in range(3):
        pathwise_step(agent, spec, rows(spec), STAGE)
    assert [g.bit_generator.state for g in held] == np_before
    after = np.random.get_state()
    assert after[0] == np_global[0] and np.array_equal(after[1], np_global[1])
    assert after[2:] == np_global[2:]
    assert torch.equal(torch.get_rng_state(), t_global) and torch.equal(gen.get_state(), t_gen)


def test_returned_numbers_match_independent_computation() -> None:
    """Pre-clip norm, FOC mean/max (autograd) and e0 equal independent pre-step values."""
    spec = spec_for(60)
    agent = make_agent(5)
    preload_adam(agent, spec)
    set_lr(agent, 3e-5)
    d = rows(spec, 400, 31)

    # independent: autograd dR/de on the same rows, grad norm with row 1 zeroed, e0 via numpy
    ref_agent = copy.deepcopy(agent)
    ref_agent.opt_actor.zero_grad(set_to_none=True)
    loss, d_t, e, e_opp = pathwise_loss(ref_agent, spec, d, STAGE)
    e_leaf = e.detach().clone().requires_grad_(True)
    (auto,) = torch.autograd.grad(expected_payoff(spec, d_t, e_leaf, e_opp).sum(), e_leaf)
    loss.backward()
    ref_agent.actor.out.weight.grad[1].zero_()
    ref_agent.actor.out.bias.grad[1].zero_()
    norm = float(torch.sqrt(sum((p.grad.double() ** 2).sum()
                                for p in ref_agent.actor.parameters())))
    a0, b0 = agent.beta_params(spec.encode_obs(STAGE, np.zeros(1)))
    e0 = spec.e_min + spec.e_range * (a0.astype(float) / (a0.astype(float) + b0.astype(float)))

    out = pathwise_step(agent, spec, d, STAGE, max_grad_norm=1e-3)
    assert out["loss"] == float(loss.item())
    assert out["foc_abs_mean"] == pytest.approx(float(auto.abs().mean()), rel=1e-12)
    assert out["foc_abs_max"] == pytest.approx(float(auto.abs().max()), rel=1e-12)
    assert out["e0"] == float(e0[0])
    assert out["grad_norm_pre_clip"] == pytest.approx(norm, rel=1e-5)
    assert out["grad_norm_pre_clip"] > 1e-3                              # clip really engaged
    post = float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in agent.actor.parameters())))
    assert post == pytest.approx(1e-3, rel=1e-4)
    assert float(agent.actor.out.weight.grad[1].abs().max()) == 0.0


def test_stage_is_a_parameter_not_hard_coded() -> None:
    """A T=3 spec with stage=3 runs and uses encode_obs(3, .) for learner and opponent."""
    spec = GameSpec(w_h=6.0, w_l=2.0, k=2.0 / 7000.0, q=50.0, T=3)
    agent = make_agent(6)
    set_lr(agent, 3e-4)
    d = np.random.default_rng(1).uniform(-spec.domain_half(3), spec.domain_half(3), 128)
    out = pathwise_step(agent, spec, d, 3)
    assert all(np.isfinite(v) for v in out.values())
    assert out["e0"] == pytest.approx(50.0, abs=1e-9)
    a, b = agent.beta_params(spec.encode_obs(3, -d), agent.opponent)
    _, _, _, e_opp = pathwise_loss(agent, spec, d, 3)
    ref = spec.e_min + spec.e_range * (a.astype(float) / (a.astype(float) + b.astype(float)))
    assert np.array_equal(e_opp.numpy(), ref)


# ------------------------------------------------------------------ optional: fitted network
def _fit_to_equilibrium(spec: GameSpec, steps: int = 2500) -> BetaActor:
    """Regress the actor mean onto the analytic e*(d)/100 (test-only construction)."""
    gen = torch.Generator().manual_seed(0)
    actor = BetaActor(64, 100.0, 1e-6, gen)
    half = spec.domain_half(STAGE)
    d = np.linspace(-half, half, 1601)
    obs = torch.as_tensor(spec.encode_obs(STAGE, d))
    target = torch.as_tensor(g2_two_stage(d, spec.q, spec.w_h, spec.w_l, spec.k) / spec.e_range,
                             dtype=torch.float32)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        a, b = actor(obs)
        loss = torch.mean((a / (a + b) - target) ** 2)
        loss.backward()
        opt.step()
    return actor


def test_foc_residual_small_for_network_fitted_to_equilibrium() -> None:
    """A network regressed onto e* has a far smaller residual than the flat initial policy."""
    spec = spec_for(50)
    fitted = make_agent(0)
    fitted.actor.load_state_dict(_fit_to_equilibrium(spec).state_dict())
    fitted.refresh_snapshot()
    flat = make_agent(0)
    d = rows(spec, 1024, 77)
    set_lr(fitted, 0.0)
    set_lr(flat, 0.0)
    out_fit = pathwise_step(fitted, spec, d, STAGE)
    out_flat = pathwise_step(flat, spec, d, STAGE)
    assert out_fit["foc_abs_mean"] < 0.1 * out_flat["foc_abs_mean"]
    assert out_fit["loss"] < out_flat["loss"]
