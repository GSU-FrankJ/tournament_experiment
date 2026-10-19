"""T3-R2 (audit A4): stage-2 and stage-3 training semantics at T = 3, with hand-computed examples.

The returns are stored in float32; they are compared with a relative tolerance of 1e-5.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_t3r2_semantics.py -p no:cacheprovider -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "t3", ROOT / "tools" / "ms"):
    sys.path.insert(0, str(_p))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

from agents.ppo_curriculum import PPOConfig  # noqa: E402
from agents.ppo_curriculum_v2 import CurriculumPPOv2  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402
from run.ms_rollout import collect_stage_batch  # noqa: E402
from run.v2_rollout import collect_batch_v2  # noqa: E402
from utils.theory_multistage import F_xi  # noqa: E402
from utils.v2_continuation import ContinuationTable  # noqa: E402

Q = 50.0
SPEC = GameSpec(6.0, 2.0, 1.0 / 3500.0, Q, 3)
B = SPEC.B                      # 200
RTOL = 1e-5


def make_agent(seed: int = 7) -> CurriculumPPOv2:
    """An agent whose actor depends on (tau, d): the output head is made non-zero."""
    gen = torch.Generator().manual_seed(seed)
    agent = CurriculumPPOv2(PPOConfig(), gen, np.random.default_rng(seed))
    g2 = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        agent.actor.out.weight.copy_(0.8 * torch.randn(agent.actor.out.weight.shape, generator=g2))
        agent.actor.out.bias.copy_(torch.tensor([0.3, -0.5]))
    agent.refresh_snapshot()                       # the lagged copy = this actor
    return agent


def make_table() -> ContinuationTable:
    """A smooth table on the stage-2 y-grid [-(B + 100), B + 100] (so linear interpolation is checkable by np.interp)."""
    y = np.linspace(-(B + 100.0), B + 100.0, 12001)
    return ContinuationTable(y_grid=y, values=3.0 + 0.004 * y + 1.5e-5 * y ** 2 - 2e-8 * y ** 3)


def draws(agent, rng_l, rng_o, obs_L, obs_O):
    """The learner and opponent efforts exactly as ``collect_stage_batch`` / ``collect_batch_v2`` draw them."""
    a_L, b_L = agent.beta_params(obs_L, None)
    a_O, b_O = agent.beta_params(obs_O, agent.opponent)
    c = agent.cfg.action_clamp
    act_L = np.clip(rng_l.beta(a_L.astype(float), b_L.astype(float)), c, 1.0 - c).astype(np.float32)
    act_O = np.clip(rng_o.beta(a_O.astype(float), b_O.astype(float)), c, 1.0 - c).astype(np.float32)
    return SPEC.effort_from_action(act_L), SPEC.effort_from_action(act_O), act_L


def test_the_observation_at_stage_2_of_a_three_stage_game_is_tau_half_and_d_over_B():
    d = np.array([0.0, 100.0, -50.0, 200.0])
    obs = SPEC.encode_obs(2, d)
    assert obs.dtype == np.float32 and obs.shape == (4, 2)
    assert np.array_equal(obs[:, 0], np.full(4, np.float32(0.5)))
    assert np.array_equal(obs[:, 1], np.array([0.0, 0.5, -0.25, 1.0], dtype=np.float32))
    assert B == 200.0
    # the other stages of the same game: tau = 1 and d / (2B) at the terminal stage, (0, 0) at the root
    assert np.array_equal(SPEC.encode_obs(3, np.array([400.0, -100.0])), np.array([[1.0, 1.0], [1.0, -0.25]], dtype=np.float32))
    assert np.array_equal(SPEC.encode_obs(1, np.array([0.0])), np.zeros((1, 2), dtype=np.float32))


def test_a_learner_in_role_1_sees_minus_d_and_the_opponent_sees_the_negative_of_that():
    agent = make_agent()
    n = 8
    d0 = np.array([-120.0, -40.0, -5.0, 0.0, 7.0, 33.0, 90.0, 180.0])
    roles = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    table = make_table()
    out = collect_stage_batch(SPEC, agent, 2, d0, roles, np.random.default_rng(1), np.random.default_rng(2), table)
    want = np.where(roles == 0, d0, -d0) / B
    assert np.array_equal(out["states"][:, 1], want.astype(np.float32))
    assert np.array_equal(out["states"][:, 0], np.full(n, np.float32(0.5)))
    assert (out["stage"] == 2).all() and out["n_transitions"] == n


def test_the_stage_2_return_is_minus_k_e_squared_plus_the_table_at_d_plus_eL_minus_eO():
    agent = make_agent()
    table = make_table()
    rng = np.random.default_rng(5)
    n = 64
    d0 = rng.uniform(-B, B, n)
    roles = rng.integers(0, 2, n)
    out = collect_stage_batch(SPEC, agent, 2, d0, roles, np.random.default_rng(11), np.random.default_rng(12), table)
    d_L = np.where(roles == 0, d0, -d0)
    e_L, e_O, act_L = draws(agent, np.random.default_rng(11), np.random.default_rng(12), SPEC.encode_obs(2, d_L),
                            SPEC.encode_obs(2, -d_L))
    y = d_L + e_L - e_O
    assert np.abs(y).max() <= B + 100.0
    rew = -SPEC.k * e_L ** 2 + np.interp(y, table.y_grid, table.values)
    assert np.allclose(out["returns"], rew, rtol=RTOL, atol=1e-6)
    val = agent.value(SPEC.encode_obs(2, d_L)).astype(float)
    assert np.allclose(out["advantages"], rew - val, rtol=RTOL, atol=1e-5)
    assert np.array_equal(out["actions"], act_L)
    assert out["returns"].dtype == np.float32
    # one hand-computed number: a single episode, the table linear (value = 1 + 0.01 y), effort 100 x 0.5 for both players
    lin = ContinuationTable(y_grid=np.linspace(-300.0, 300.0, 6001), values=1.0 + 0.01 * np.linspace(-300.0, 300.0, 6001))
    ag = make_agent(3)
    with torch.no_grad():
        ag.actor.out.weight.zero_()
        ag.actor.out.bias.zero_()                      # mu = 0.5, c = 100 + ln 2: the mean action is exactly 0.5
    ag.refresh_snapshot()
    o = collect_stage_batch(SPEC, ag, 2, np.array([20.0]), np.array([0]), np.random.default_rng(1), np.random.default_rng(2), lin)
    aL, bL = ag.beta_params(SPEC.encode_obs(2, np.array([20.0])))
    aO, bO = ag.beta_params(SPEC.encode_obs(2, np.array([-20.0])), ag.opponent)
    a_L = np.random.default_rng(1).beta(aL.astype(float), bL.astype(float))
    a_O = np.random.default_rng(2).beta(aO.astype(float), bO.astype(float))
    eL, eO = 100.0 * float(np.float32(a_L[0])), 100.0 * float(np.float32(a_O[0]))
    assert o["returns"][0] == pytest.approx(-SPEC.k * eL ** 2 + 1.0 + 0.01 * (20.0 + eL - eO), rel=RTOL)


def test_the_table_lookup_refuses_an_out_of_grid_y():
    table = make_table()
    lo, hi = float(table.y_grid[0]), float(table.y_grid[-1])
    table.lookup(np.array([lo, 0.0, hi]))                             # the end points are inside
    for bad in (lo - 1.0, hi + 1.0, np.nan):
        with pytest.raises(ValueError):
            table.lookup(np.array([0.0, bad]))
    # through the rollout: a table that is too narrow for the reachable y is refused, never clipped silently
    agent = make_agent()
    narrow = ContinuationTable(y_grid=np.linspace(-1.0, 1.0, 101), values=np.zeros(101))
    with pytest.raises(ValueError):
        collect_stage_batch(SPEC, agent, 2, np.array([150.0, -150.0, 10.0, 0.0]), np.array([0, 1, 0, 1]),
                            np.random.default_rng(1), np.random.default_rng(2), narrow)


def test_the_opponents_stage_2_action_comes_from_the_lagged_copy_not_the_live_actor():
    agent = make_agent()
    table = make_table()
    with torch.no_grad():                               # the live actor moves; the lagged copy does not
        agent.actor.out.bias.add_(torch.tensor([1.2, 0.0]))
    d0 = np.linspace(-150.0, 150.0, 32)
    roles = np.tile([0, 1], 16)
    out = collect_stage_batch(SPEC, agent, 2, d0, roles, np.random.default_rng(21), np.random.default_rng(22), table)
    d_L = np.where(roles == 0, d0, -d0)
    obs_L, obs_O = SPEC.encode_obs(2, d_L), SPEC.encode_obs(2, -d_L)
    e_L, e_O_lagged, _ = draws(agent, np.random.default_rng(21), np.random.default_rng(22), obs_L, obs_O)
    rew_lagged = -SPEC.k * e_L ** 2 + np.interp(d_L + e_L - e_O_lagged, table.y_grid, table.values)
    assert np.allclose(out["returns"], rew_lagged, rtol=RTOL, atol=1e-6)
    # had the opponent used the live actor, the return would differ clearly
    a_O, b_O = agent.beta_params(obs_O, None)
    c = agent.cfg.action_clamp
    e_O_live = SPEC.effort_from_action(np.clip(np.random.default_rng(22).beta(a_O.astype(float), b_O.astype(float)), c, 1 - c))
    rew_live = -SPEC.k * e_L ** 2 + np.interp(d_L + e_L - e_O_live, table.y_grid, table.values)
    assert np.abs(rew_live - rew_lagged).max() > 1e-3
    assert not np.allclose(out["returns"], rew_live, rtol=RTOL, atol=1e-6)
    # the learner's own draw and log-probability use the live actor
    a_L, b_L = agent.beta_params(obs_L, None)
    assert np.allclose(out["logp"], agent.log_prob(a_L, b_L, out["actions"]), rtol=1e-6, atol=1e-6)


def test_the_stage_3_phase_reward_is_the_expected_terminal_reward():
    agent = make_agent()
    rng = np.random.default_rng(9)
    n = 48
    d0 = rng.uniform(-2 * B, 2 * B, n)
    roles = rng.integers(0, 2, n)
    out = collect_batch_v2(SPEC, agent, np.full(n, 3), d0, roles, np.random.default_rng(31), np.random.default_rng(32),
                           np.random.default_rng(33), 1.0, 1.0, 10.0, reward_mode="expected", frozen=None,
                           continuation_action_mode="stochastic", cont_table=None, clamp_likelihood="density")
    d_L = np.where(roles == 0, d0, -d0)
    obs_L, obs_O = SPEC.encode_obs(3, d_L), SPEC.encode_obs(3, -d_L)
    assert np.array_equal(out["states"][:, 0], np.full(n, np.float32(1.0)))
    assert np.array_equal(out["states"][:, 1], (d_L / (2 * B)).astype(np.float32))
    e_L, e_O, act_L = draws(agent, np.random.default_rng(32), np.random.default_rng(33), obs_L, obs_O)
    rew = SPEC.w_l + SPEC.dw * F_xi(d_L + e_L - e_O, Q) - SPEC.k * e_L ** 2
    assert np.allclose(out["returns"], rew, rtol=RTOL, atol=1e-6)
    assert (out["stage"] == 3).all()
    # a hand-computed value: d = 0, equal efforts 50: w_l + DW * F(0) - k 50^2 = 2 + 4 * 0.5 - 2500/3500
    assert SPEC.w_l + SPEC.dw * float(F_xi(0.0, Q)[0]) - SPEC.k * 50.0 ** 2 == pytest.approx(4.0 - 2500.0 / 3500.0, rel=1e-12)
