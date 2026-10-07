"""Tests of the nested expected-continuation tables (spec 2.2 "Continuation (D2)"; D8 T=3 items).

Run (about 4 minutes on one thread; the T=3 self-convergence alone is about 2.5 minutes):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest \
        tests/test_ms_continuation.py -q -s

Module under test: ``utils/ms_continuation.py`` (``build_table``, ``build_nested_tables``,
``g_values``, ``y_grid_for_stage``). Reference implementations used here: ``utils.v2_continuation``
(T = 2 equality), ``utils.dp_br_verifier.verify`` at a refined tier (T = 3 agreement), the
Irwin-Hall closed form for flat (mean 0.5) actors, and an independent finer quadrature.

Frozen actors. ``BetaActor(64, 100.0, 1e-6, gen)`` as constructed has an all-zero output layer,
i.e. a CONSTANT policy (mean 0.5, effort 50); it is used for the closed-form tests. The other
tests use the same actor with random output-head weights (``random_actor``), so that the mean
effort varies widely with d, as the existing R1 continuation tests do.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ.setdefault(f"{_k}_NUM_THREADS", "1")

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.ppo_curriculum import BetaActor, CurriculumPPO, PPOConfig  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402
from run.run_final_dp_br import make_policy_fns  # noqa: E402
from utils import ms_continuation as mc  # noqa: E402
from utils import v2_continuation as vc  # noqa: E402
from utils.dp_br_verifier import VerifierConfig, verify  # noqa: E402
from utils.ms_continuation import (build_nested_tables, build_table, g_values,  # noqa: E402
                                   y_grid_for_stage)
from utils.v2_continuation import ContinuationTable, build_continuation_table  # noqa: E402

RECORDS = json.loads((ROOT / "protocols" / "v2_T2_locked_v2_0.json").read_text())["records"]
EQ_TOL_OVER_DW = 1e-12            # T = 2 equality with utils.v2_continuation
SELF_CONV_OVER_DW = 1e-8          # D8: nested self-convergence of V~_2
VERIFIER_OVER_DW = 1e-6           # D8: agreement with the refined verifier
REFINED = VerifierConfig("refined", state_step=0.25, effort_step=1.0, gl_half=64)
E1_HAT = 40.0                     # fixed stage-1 mean of the verifier candidates (the opponent)


def spec_for(q: int, T: int = 2) -> GameSpec:
    """The v2.0 game of this q with horizon T."""
    return GameSpec(**{**RECORDS[str(q)]["game"], "q": float(q), "T": int(T)})


def random_actor(seed: int) -> BetaActor:
    """BetaActor with random non-trivial output-head weights (mean effort varies widely with d)."""
    cfg = PPOConfig()
    gen = torch.Generator().manual_seed(seed)
    actor = BetaActor(cfg.hidden, cfg.c_min, cfg.mu_clamp, gen)
    with torch.no_grad():
        actor.out.weight.copy_(torch.randn(2, cfg.hidden, generator=gen) * 0.8)
        actor.out.bias.copy_(torch.randn(2, generator=gen) * 0.5)
    actor.eval()
    return actor


def flat_actor(seed: int = 0) -> BetaActor:
    """BetaActor(64, 100.0, 1e-6, gen) as in the brief: zero output layer = constant e = 50."""
    actor = BetaActor(64, 100.0, 1e-6, torch.Generator().manual_seed(seed))
    actor.eval()
    return actor


def irwin_hall_cdf(x: np.ndarray, n: int) -> np.ndarray:
    """CDF of the sum of n iid U(0, 1)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for j in range(n + 1):
        out += np.where(x > j, (-1) ** j * math.comb(n, j) * np.maximum(x - j, 0.0) ** n, 0.0)
    return np.where(x >= n, 1.0, out / math.factorial(n))


_CHAINS: Dict[Tuple[str, float, int, float], Dict[int, ContinuationTable]] = {}


def t3_actors() -> Dict[int, BetaActor]:
    """Random frozen actors of stages 2 and 3 of the T = 3 game."""
    return {2: random_actor(11), 3: random_actor(12)}


def t3_chain(panel_width: float = vc.DEFAULT_PANEL_WIDTH,
             nodes: int = vc.DEFAULT_NODES_PER_PANEL, step: float = vc.DEFAULT_STEP
             ) -> Dict[int, ContinuationTable]:
    """Nested tables {2: V~_3, 1: V~_2} of the random T = 3 suffix under the given rule (cached)."""
    key = ("t3", float(panel_width), int(nodes), float(step))
    if key not in _CHAINS:
        _CHAINS[key] = build_nested_tables(t3_actors(), spec_for(50, T=3), 1, step, panel_width,
                                           nodes)
    return _CHAINS[key]


# ----------------------------------------------------------------------------------------------
# T = 2: the new table of stage 1 is the v2.0 table
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("q,kind", [(50, "random3"), (60, "random5"), (50, "flat")])
def test_t2_table_equals_utils_v2_continuation_bitwise(q, kind):
    spec = spec_for(q)
    actor = flat_actor() if kind == "flat" else random_actor(int(kind[-1]))
    old = build_continuation_table(actor, spec, stage=2, step=0.05)
    new = build_table(actor, spec, 1)
    assert np.array_equal(old.y_grid, new.y_grid)                 # bitwise
    assert np.array_equal(old.values, new.values)                 # bitwise
    diff = float(np.abs(old.values - new.values).max()) / spec.dw
    print(f"\n[{kind}, q={q}] max|new - v2_continuation| = {diff:.3e} DW "
          f"(limit {EQ_TOL_OVER_DW:g})")
    assert diff <= EQ_TOL_OVER_DW
    for key in ("q", "w_h", "w_l", "k", "dw", "e_min", "e_max", "step_requested", "step_actual",
                "n_y", "panel_width", "nodes_per_panel", "n_panels", "n_nodes", "weight_sum",
                "conc_scale"):
        assert old.meta[key] == new.meta[key], key
    assert new.meta["stage"] == 2 and new.meta["phase_stage"] == 1 and new.meta["nested"] is False
    assert new.meta["schema"] == "ms_continuation_table/1" and new.meta["T"] == 2
    assert new.meta["n_panels"] == 2 * math.ceil(2 * q) and new.meta["n_nodes"] == 24 * q
    assert new.meta["weight_sum"] == pytest.approx(1.0, abs=1e-12)
    assert new.meta["build_seconds"] > 0.0 and new.meta["y_half"] == 100.0


def test_t2_flat_actor_table_matches_the_irwin_hall_closed_form():
    """A constant policy (e = 50 for both players) has V(y) = w_l - k 2500 + DW P(S4 <= y), S4 the
    sum of four iid U(-q, q): an exact reference independent of any quadrature."""
    spec = spec_for(50)
    tab = build_table(flat_actor(), spec, 1)
    y = tab.y_grid
    want = spec.w_l - spec.k * 50.0 ** 2 + spec.dw * irwin_hall_cdf(y / (2 * spec.q) + 2.0, 4)
    err = float(np.abs(tab.values - want).max()) / spec.dw
    print(f"\n[flat T=2] max|table - Irwin-Hall| = {err:.3e} DW")
    assert err <= 1e-9                                           # measured 2.1e-10


# ----------------------------------------------------------------------------------------------
# y-grid coverage
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("q,T,t", [(50, 2, 1), (60, 2, 1), (50, 3, 1), (50, 3, 2), (60, 3, 2)])
def test_y_grid_covers_every_reachable_shift_and_all_nodes_land_in_the_next_domain(q, T, t):
    spec = spec_for(q, T)
    y = y_grid_for_stage(spec, t)
    half = spec.domain_half(t) + spec.e_range
    assert y[0] == -half and y[-1] == half and y[(y.size - 1) // 2] == 0.0
    assert np.allclose(y, -y[::-1], rtol=0, atol=1e-12) and np.diff(y).max() <= 0.05 + 1e-12
    nodes, w, _ = vc.shock_quadrature(spec.q, vc.DEFAULT_PANEL_WIDTH, vc.DEFAULT_NODES_PER_PANEL)
    assert np.abs(y).max() + np.abs(nodes).max() < spec.domain_half(t + 1)    # inside D_{t+1}
    # every reachable shift d_t + e_t - e_t^opp lies on the grid:
    # |d_t| <= domain_half(t) and |e - e'| <= e_range
    assert spec.domain_half(t) + spec.e_range <= half
    if T == 2:
        assert np.array_equal(y, vc.y_grid_for(spec))                          # = v2.0's grid
    assert mc.y_grid_for_stage(spec, t, step=0.5).size < y.size


def test_a_landing_node_outside_the_next_domain_raises(monkeypatch):
    """Force the nodes out of D_{t+1} by shrinking domain_half of the next stage."""
    spec = spec_for(50, T=2)
    actor = random_actor(3)
    build_table(actor, spec, 1, step=5.0)                                      # fine unpatched
    real = GameSpec.domain_half
    monkeypatch.setattr(GameSpec, "domain_half",
                        lambda self, t: real(self, t) * (0.99 if t >= 2 else 1.0))
    with pytest.raises(ValueError, match="outside D_2"):
        build_table(actor, spec, 1, step=5.0)
    monkeypatch.setattr(GameSpec, "domain_half", real)
    # T = 3: shrink D_3 (table of stage 2 fails) and, separately, D_2 (table of stage 1 fails)
    spec3 = spec_for(50, T=3)
    actors = t3_actors()
    for bad_stage in (3, 2):
        monkeypatch.setattr(GameSpec, "domain_half",
                            lambda self, t, b=bad_stage: real(self, t) * (0.99 if t == b else 1.0))
        with pytest.raises(ValueError, match=f"outside D_{bad_stage}"):
            build_nested_tables(actors, spec3, 1, step=5.0)
    monkeypatch.setattr(GameSpec, "domain_half", real)
    build_nested_tables(actors, spec3, 1, step=5.0)                            # fine again


def test_next_table_that_does_not_cover_the_reachable_shifts_raises():
    spec3 = spec_for(50, T=3)
    actors = t3_actors()
    small = ContinuationTable(y_grid=np.linspace(-100.0, 100.0, 41), values=np.zeros(41))
    with pytest.raises(ValueError, match="outside the table grid"):
        build_table(actors[2], spec3, 1, small, step=5.0)


# ----------------------------------------------------------------------------------------------
# purity: no RNG movement, no mutation
# ----------------------------------------------------------------------------------------------

def _rng_snapshot() -> Dict[str, object]:
    return {"np_legacy": np.random.get_state()[1].copy(), "np_pos": np.random.get_state()[2],
            "torch": torch.get_rng_state().clone(), "py": random.getstate()}


def _same_snapshot(a: Dict[str, object], b: Dict[str, object]) -> bool:
    return (np.array_equal(a["np_legacy"], b["np_legacy"]) and a["np_pos"] == b["np_pos"]
            and bool(torch.equal(a["torch"], b["torch"])) and a["py"] == b["py"])


def test_a_build_moves_no_rng_stream_and_mutates_nothing(monkeypatch):
    np.random.seed(123)
    torch.manual_seed(456)
    random.seed(789)
    gen = np.random.default_rng(2026)                      # a stream the build must not touch
    gen_state = copy_state(gen)
    spec = spec_for(50, T=3)
    actors = t3_actors()
    # (nn.Linear init draws from the torch global RNG: build first)
    actor2 = random_actor(3)
    np.random.seed(123)
    torch.manual_seed(456)
    random.seed(789)
    before = _rng_snapshot()
    params_before = {t: {k: v.clone() for k, v in a.state_dict().items()}
                     for t, a in actors.items()}

    def forbidden(*a, **k):
        raise AssertionError("a table build must not create a random generator")
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    tabs = build_nested_tables(actors, spec, 1, step=2.0, panel_width=2.0, nodes_per_panel=4)
    t2 = build_table(actor2, spec_for(50), 1, step=2.0, panel_width=2.0, nodes_per_panel=4)
    monkeypatch.undo()
    assert tabs[1].values.size > 0 and t2.values.size > 0
    assert _same_snapshot(before, _rng_snapshot())          # numpy legacy, torch global, python
    assert copy_state(gen) == gen_state                     # a numpy default_rng stream
    for t, a in actors.items():
        assert not a.training
        for k, v in a.state_dict().items():
            assert torch.equal(v, params_before[t][k]), (t, k)
        assert all(p.grad is None for p in a.parameters())


def copy_state(gen: np.random.Generator) -> Dict[str, object]:
    """Deep copy of a generator's bit-generator state (for equality checks)."""
    return json.loads(json.dumps(gen.bit_generator.state, default=int))


def test_chunk_size_does_not_change_the_table(monkeypatch):
    spec = spec_for(50, T=3)
    actors = t3_actors()
    ref = build_nested_tables(actors, spec, 1, step=2.0, panel_width=2.0, nodes_per_panel=4)
    monkeypatch.setattr(vc, "_MAX_POINTS_PER_CHUNK", 5000)
    small = build_nested_tables(actors, spec, 1, step=2.0, panel_width=2.0, nodes_per_panel=4)
    for t in (1, 2):
        assert np.abs(ref[t].values - small[t].values).max() / spec.dw <= 1e-13


# ----------------------------------------------------------------------------------------------
# T = 3: structure, closed form, independent quadrature
# ----------------------------------------------------------------------------------------------

def test_nested_tables_structure_and_meta():
    spec = spec_for(50, T=3)
    tabs = t3_chain(step=0.25)             # coarse step: structure only
    assert sorted(tabs) == [1, 2]
    m2, m1 = tabs[2].meta, tabs[1].meta
    assert (m2["stage"], m2["phase_stage"], m2["nested"], m2["T"]) == (3, 2, False, 3)
    assert (m1["stage"], m1["phase_stage"], m1["nested"], m1["T"]) == (2, 1, True, 3)
    assert tabs[2].y_grid[0] == -300.0 and tabs[2].y_grid[-1] == 300.0
    assert tabs[1].y_grid[0] == -100.0 and tabs[1].y_grid[-1] == 100.0
    assert np.array_equal(tabs[2].y_grid, y_grid_for_stage(spec, 2, 0.25))
    assert np.array_equal(tabs[1].y_grid, y_grid_for_stage(spec, 1, 0.25))
    for tab in tabs.values():
        assert tab.meta["weight_sum"] == pytest.approx(1.0, abs=1e-12)
        assert tab.meta["n_panels"] == 200 and tab.meta["nodes_per_panel"] == 6
    # build_nested_tables = the sequential build_table calls; t_min = 2 builds only the last table
    actors = t3_actors()
    only2 = build_nested_tables(actors, spec, 2, step=0.25)
    assert sorted(only2) == [2] and np.array_equal(only2[2].values, tabs[2].values)
    seq1 = build_table(actors[2], spec, 1, tabs[2], step=0.25)
    assert np.array_equal(seq1.values, tabs[1].values)


def test_g_values_terminal_and_nested_forms():
    spec = spec_for(50, T=3)
    actors = t3_actors()
    tabs = t3_chain(step=0.25)
    d3 = np.linspace(-400.0, 400.0, 161)
    d2 = np.linspace(-200.0, 200.0, 161)
    assert np.array_equal(g_values(actors[3], spec, 3, d3, None),
                          vc.g_frozen(actors[3], spec, 3, d3))
    e_own = vc.frozen_mean_effort(actors[2], spec, 2, d2)
    e_opp = vc.frozen_mean_effort(actors[2], spec, 2, -d2)
    want = -spec.k * e_own ** 2 + tabs[2].lookup(d2 + e_own - e_opp)
    assert np.array_equal(g_values(actors[2], spec, 2, d2, tabs[2]), want)
    with pytest.raises(ValueError, match="needs the continuation table"):
        g_values(actors[2], spec, 2, d2, None)


@pytest.mark.parametrize("q", [50, 60])
def test_flat_chain_matches_the_irwin_hall_closed_forms_at_t3(q):
    """Constant policies (e = 50 at every stage): V~_3(y) = w_l - k 2500 + DW P(S4 <= y) and
    V~_2(y) = w_l - 2 k 2500 + DW P(S6 <= y), S_n a sum of n iid U(-q, q). Step 0.25 keeps it fast
    (the table lookups then carry linear-interpolation error, hence the looser bound)."""
    spec = spec_for(q, T=3)
    actors = {2: flat_actor(), 3: flat_actor(1)}
    tabs = build_nested_tables(actors, spec, 1, step=0.25)
    c = spec.k * 50.0 ** 2
    for t, n, costs in ((2, 4, 1), (1, 6, 2)):
        y = tabs[t].y_grid
        want = spec.w_l - costs * c + spec.dw * irwin_hall_cdf(y / (2 * q) + n / 2.0, n)
        err = float(np.abs(tabs[t].values - want).max()) / spec.dw
        print(f"\n[flat T=3, q={q}] table of phase stage {t} vs Irwin-Hall n={n}: {err:.3e} DW")
        assert err <= 1e-6                                       # measured 3.1e-7 (interpolation)


def test_nested_table_agrees_with_an_independent_finer_quadrature_t3():
    """V~_2(y) at table nodes y, recomputed with an independent g_2 evaluation (direct torch call)
    and an independent, finer composite Gauss-Legendre rule (panel 0.25, 16 nodes): <= 1e-8 DW."""
    spec = spec_for(50, T=3)
    actors = t3_actors()
    tabs = t3_chain(step=0.25)
    q = spec.q

    @torch.no_grad()
    def mean_effort(actor: BetaActor, t: int, d: np.ndarray) -> np.ndarray:
        a, b = actor(torch.as_tensor(spec.encode_obs(t, d)))
        a, b = a.numpy().astype(np.float64), b.numpy().astype(np.float64)
        return spec.e_min + spec.e_range * (a / (a + b))

    m = int(2 * q / 0.25)                                       # panels per half
    x, w = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(0.0, 2 * q, m + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    pos = (0.5 * (lo + hi) + 0.5 * (hi - lo) * x[None, :]).ravel()
    wpos = (0.5 * (hi - lo) * w[None, :]).ravel()
    z = np.concatenate([-pos[::-1], pos])
    wz = np.concatenate([wpos[::-1], wpos]) * (2 * q - np.abs(np.concatenate([-pos[::-1], pos]))) \
        / (4 * q * q)
    assert wz.sum() == pytest.approx(1.0, abs=1e-12)
    worst = 0.0
    for y in (-100.0, -35.0, 0.0, 12.5, 65.0, 100.0):
        d = y + z
        e_own, e_opp = mean_effort(actors[2], 2, d), mean_effort(actors[2], 2, -d)
        g2 = -spec.k * e_own ** 2 + np.interp(d + e_own - e_opp, tabs[2].y_grid, tabs[2].values)
        indep = float(g2 @ wz)
        j = int(np.argmin(np.abs(tabs[1].y_grid - y)))
        assert tabs[1].y_grid[j] == y
        worst = max(worst, abs(indep - tabs[1].values[j]) / spec.dw)
    print(f"\n[T=3 independent quadrature] max |V~_2 table - independent| = {worst:.3e} DW")
    assert worst <= 1e-8                                         # measured 2.1e-9


# slow: ~150 s on one thread (default chain 30 s + chain with panel width 0.5 and 12 nodes 120 s)
def test_t3_nested_self_convergence():
    """Halving the panel width and doubling the nodes per panel for the WHOLE chain changes the
    stage-1 table V~_2 by <= 1e-8 DW (the stage-2 table V~_3 is reported too)."""
    spec = spec_for(50, T=3)
    base = t3_chain()
    fine = t3_chain(panel_width=vc.DEFAULT_PANEL_WIDTH / 2.0, nodes=2 * vc.DEFAULT_NODES_PER_PANEL)
    d_v2 = float(np.abs(base[1].values - fine[1].values).max()) / spec.dw
    d_v3 = float(np.abs(base[2].values - fine[2].values).max()) / spec.dw
    print(f"\n[T=3 self-convergence, q=50] max|dV~_2| = {d_v2:.3e} DW, max|dV~_3| = {d_v3:.3e} DW "
          f"(limit {SELF_CONV_OVER_DW:g})")
    assert fine[1].meta["n_panels"] == 400 and fine[1].meta["nodes_per_panel"] == 12
    assert d_v2 <= SELF_CONV_OVER_DW, f"measured {d_v2:.6e} DW"
    assert d_v3 <= SELF_CONV_OVER_DW, f"measured {d_v3:.6e} DW"


def _verifier_candidate(spec: GameSpec):
    """Composite candidate: stage 1 constant, stages 2 and 3 the random frozen actors' Beta means
    (``make_policy_fns``-style wrappers around the production mean path)."""
    agent = CurriculumPPO(PPOConfig(), torch.Generator().manual_seed(0), np.random.default_rng(0))
    actors = t3_actors()
    fns = {t: make_policy_fns(agent, spec, actors[t])[0] for t in (2, 3)}

    def policy(t: int, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        return np.full(d.shape, E1_HAT) if t == 1 else fns[t](t, d)
    return policy


_VERIFIER: Dict[str, object] = {}


def refined_verifier_result():
    """verify() of the T = 3 composite candidate at the refined tier (cached)."""
    if "res" not in _VERIFIER:
        spec = spec_for(50, T=3)
        _VERIFIER["res"] = verify(_verifier_candidate(spec), w_h=spec.w_h, w_l=spec.w_l, k=spec.k,
                                  q=spec.q, T=3, e_min=spec.e_min, e_max=spec.e_max, cfg=REFINED)
    return _VERIFIER["res"]


def test_t3_table_of_stage_1_agrees_with_the_refined_verifier():
    """(i) E_x[interp(y + x, G_2, v_mean_2)] (verifier GL rule over the shock, its stage-2 values)
    against the stage-1 table V~_2(y), and the verifier's own stage-1 Q: <= 1e-6 DW."""
    spec = spec_for(50, T=3)
    res = refined_verifier_result()
    tabs = t3_chain()
    s1, s2 = res.stages[1], res.stages[2]
    y = np.linspace(-100.0, 100.0, 801)
    acc = np.zeros_like(y)
    for x, w in zip(res.gl_nodes, res.gl_weights):
        acc += w * np.interp(y + x, s2.d_grid, s2.v_mean)
    d_direct = float(np.abs(acc - tabs[1].lookup(y)).max()) / spec.dw
    E = res.e_grid
    q1 = s1.q_mean_grid[0] + spec.k * E ** 2
    d_q1 = float(np.abs(q1 - tabs[1].lookup(E - E1_HAT)).max()) / spec.dw
    print(f"\n[T=3 vs refined verifier, (i)] direct E_x interp: {d_direct:.3e} DW; "
          f"stage-1 Q grid: {d_q1:.3e} DW (limit {VERIFIER_OVER_DW:g})")
    assert res.config.state_step == 0.25 and res.config.gl_half == 64
    assert d_direct <= VERIFIER_OVER_DW and d_q1 <= VERIFIER_OVER_DW


def test_t3_table_of_stage_2_agrees_with_the_refined_verifier_q_mean_grid():
    """(ii) the verifier's stage-2 Q^mean grid against -k e^2 + V~_3.lookup(d - e_hat_2(-d) + e)."""
    spec = spec_for(50, T=3)
    res = refined_verifier_result()
    tabs = t3_chain()
    s2 = res.stages[2]
    E = res.e_grid
    land = (s2.d_grid - s2.e_opp)[:, None] + E[None, :]
    want = -spec.k * E[None, :] ** 2 + tabs[2].lookup(land)
    diff = np.abs(s2.q_mean_grid - want)
    err = float(diff.max()) / spec.dw
    j = np.unravel_index(int(np.argmax(diff)), diff.shape)
    print(f"\n[T=3 vs refined verifier, (ii)] max|Q^mean_2 - (-k e^2 + V~_3)| = {err:.3e} DW "
          f"at d = {s2.d_grid[j[0]]:g}, e = {E[j[1]]:g} (limit {VERIFIER_OVER_DW:g})")
    assert s2.q_mean_grid.shape == (1601, 101)
    assert err <= VERIFIER_OVER_DW


# ----------------------------------------------------------------------------------------------
# refusals
# ----------------------------------------------------------------------------------------------

def test_build_table_refuses_a_missing_next_table_and_a_stage_out_of_range():
    spec3, spec2 = spec_for(50, T=3), spec_for(50, T=2)
    a = random_actor(3)
    with pytest.raises(ValueError, match="table of stage 2 is required"):
        # s = 2 < T = 3, no next table
        build_table(a, spec3, 1)
    for t in (0, -1, 3, 4):
        with pytest.raises(ValueError, match=r"t must lie in \[1, 2\]"):
            build_table(a, spec3, t, None, step=5.0)
    for t in (0, 2, 3):
        with pytest.raises(ValueError, match=r"t must lie in \[1, 1\]"):
            build_table(a, spec2, t, None, step=5.0)
    build_table(a, spec2, 1, None, step=5.0)                            # T = 2 needs no next table
    tab3 = build_table(a, spec3, 2, None, step=5.0)                     # the terminal table: none
    build_table(a, spec3, 1, tab3, step=5.0)                            # nested: given
