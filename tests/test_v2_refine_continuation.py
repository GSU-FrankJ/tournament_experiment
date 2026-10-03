"""Tests of the expected-continuation table (R1 method 6; PROMPT.md D4 and section 2.3).

Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python -m pytest tests/test_v2_refine_continuation.py -p no:cacheprovider -q

Write the check record ``results/v2_refine/continuation_check.json`` (several minutes, run it in
tmux):
    .venv/bin/python tests/test_v2_refine_continuation.py --write

Real frozen actors are the stage-2 actors of the v1.1 rehearsal parents (seed 10501) in the
canonical results worktree (read-only); a test that needs one is skipped only if the file is
absent. Every test also runs on a synthetic actor with random non-trivial output-head weights.
The module under test must not import the verifier; this file may.

Status of PROMPT.md 2.3 (ii) (table vs the verifier's stage-1 Q, final tier, limit 1e-6 DW):
NOT MET on the production final tier. ``test_ii_final_tier_within_spec_tolerance`` keeps the
literal assertion and is marked ``xfail(strict=True)``: it is a record of an open failure that
needs a PI decision, not a pass. The measured numbers are in
``results/v2_refine/continuation_check.json`` (keys ``spec_requirement``, ``table_vs_verifier``,
``diagnosis``). Nothing here was loosened.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from run.v2_rollout import expected_terminal_reward  # noqa: E402
from utils import v2_continuation as vc  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG, FINAL_CONFIG, VerifierConfig, verify  # noqa: E402
from utils.theory_multistage import F_xi  # noqa: E402
from utils.v2_continuation import ContinuationTable, build_continuation_table  # noqa: E402

CAN = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2")
PARENT_SEED = 10501
W_H, W_L, K = 6.0, 2.0, 1.0 / 3500.0     # as-run values (reports/v2/phase0_audit.md section 2)
E1_HAT = {50: 40.0, 60: 45.0}            # fixed stage-1 mean: it only fixes the opponent
QS = (50, 60)
FINAL_TOL_OVER_DW = 1e-6                 # PROMPT.md section 2.3 (ii), final tier
MC_DRAWS = 10 ** 6
MC_SEED = 91011
Y_MC = (-80.0, -35.0, 0.0, 12.37, 65.0)
OUT_JSON = ROOT / "results" / "v2_refine" / "continuation_check.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def spec_for(q: float) -> GameSpec:
    """As-run T=2 game for one q."""
    return GameSpec(w_h=W_H, w_l=W_L, k=K, q=float(q), T=2, e_min=0.0, e_max=100.0)


def parent_path(q: int, seed: int = PARENT_SEED) -> Path:
    """Parent end-of-A full state of the v1.1 rehearsal (canonical worktree, read-only)."""
    return (CAN / "results" / "v2_T2_locked" / "rehearsal_v1_1" / f"q{q}" / f"seed{seed}"
            / "state_end_A.pt")


def new_agent() -> CurriculumPPO:
    """Agent with the run's PPOConfig (hidden 64, c_min 100, mu_clamp 1e-6)."""
    return CurriculumPPO(PPOConfig(), torch.Generator().manual_seed(0), np.random.default_rng(0))


def parent_agent(q: int) -> Optional[CurriculumPPO]:
    """Agent whose actor holds the parent's stage-2 weights; None if the file is absent."""
    path = parent_path(q)
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    agent = new_agent()
    agent.actor.load_state_dict(state["agent"]["actor"])
    agent.actor.eval()
    return agent


def synthetic_actor(seed: int = 7) -> BetaActor:
    """BetaActor with random non-trivial output-head weights (mean varies widely with d)."""
    cfg = PPOConfig()
    gen = torch.Generator().manual_seed(seed)
    actor = BetaActor(cfg.hidden, cfg.c_min, cfg.mu_clamp, gen)
    with torch.no_grad():
        actor.out.weight.copy_(torch.randn(2, cfg.hidden, generator=gen) * 0.8)
        actor.out.bias.copy_(torch.randn(2, generator=gen) * 0.5)
    actor.eval()
    return actor


def get_actor(kind: str, q: int) -> BetaActor:
    """Frozen actor of ``kind`` ('parent' or 'synthetic'); skip a parent test if file absent."""
    if kind == "synthetic":
        return synthetic_actor()
    agent = parent_agent(q)
    if agent is None:
        pytest.skip(f"parent state absent: {parent_path(q)}")
    return agent.actor


_TABLES: Dict[Tuple[str, int], ContinuationTable] = {}


def get_table(kind: str, q: int) -> Tuple[BetaActor, GameSpec, ContinuationTable]:
    """Default-rule table of the actor (built once per (kind, q) and cached)."""
    actor, spec = get_actor(kind, q), spec_for(q)
    if (kind, q) not in _TABLES:
        _TABLES[(kind, q)] = build_continuation_table(actor, spec, stage=2)
    return actor, spec, _TABLES[(kind, q)]


@torch.no_grad()
def g2_direct(actor: BetaActor, spec: GameSpec, d: np.ndarray) -> np.ndarray:
    """Independent implementation of g2(d) (frozen mean, both players), not via the module."""
    def mean_effort(dd: np.ndarray) -> np.ndarray:
        a, b = actor(torch.as_tensor(spec.encode_obs(2, dd)))
        a = a.numpy().astype(np.float64)
        b = b.numpy().astype(np.float64)
        return spec.e_min + spec.e_range * (a / (a + b))

    e_own, e_opp = mean_effort(d), mean_effort(-d)
    return spec.w_l + spec.dw * F_xi(d + e_own - e_opp, spec.q) - spec.k * e_own ** 2


def irwin_hall4_cdf(x: np.ndarray) -> np.ndarray:
    """CDF of the sum of four iid U(0, 1)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for j in range(5):
        out += np.where(x > j, (-1) ** j * math.comb(4, j) * (x - j) ** 4, 0.0)
    return np.where(x >= 4.0, 1.0, out / 24.0)


class EvenActor:
    """Callable actor with an exactly even mean in d (depends on the input only through dn^2)."""

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dn2 = x[:, 1] * x[:, 1]
        mean = torch.clamp(0.05 + 0.5 / (1.0 + 40.0 * dn2), 1e-6, 1.0 - 1e-6)
        conc = torch.full_like(mean, 120.0)
        return mean * conc, (1.0 - mean) * conc


def exact_interp_integral(grid: np.ndarray, vals: np.ndarray, q: float, y: np.ndarray
                          ) -> np.ndarray:
    """Exact ``int f(z) interp(y + z; grid, vals) dz`` for piecewise-linear interpolation.

    The integrand is quadratic between consecutive breakpoints (grid knots shifted by -y and
    the density kinks 0, +-2q), so a 2-node Gauss-Legendre rule per piece is exact.
    """
    xg, wg = np.polynomial.legendre.leggauss(2)
    out = np.empty(y.size)
    for i, yy in enumerate(y):
        br = np.concatenate([grid - yy, [-2.0 * q, 0.0, 2.0 * q]])
        br = np.unique(br[(br >= -2.0 * q) & (br <= 2.0 * q)])
        lo, hi = br[:-1], br[1:]
        half = 0.5 * (hi - lo)
        zn = 0.5 * (lo + hi)[:, None] + half[:, None] * xg[None, :]
        dens = (2.0 * q - np.abs(zn)) / (4.0 * q * q)
        out[i] = np.sum(half[:, None] * wg[None, :] * dens * np.interp(yy + zn, grid, vals))
    return out


def verifier_policy(agent: CurriculumPPO, spec: GameSpec, e1_hat: float):
    """Verifier-facing candidate: stage 1 = constant ``e1_hat``, stage 2 = production mean_fn."""
    mean_fn, _ = make_policy_fns(agent, spec)

    def policy(t: int, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        return np.full(d.shape, float(e1_hat)) if t == 1 else mean_fn(t, d)
    return policy


def compare_with_verifier(agent: CurriculumPPO, spec: GameSpec, e1_hat: float,
                          cfg: VerifierConfig, table: Optional[ContinuationTable] = None
                          ) -> Dict[str, Any]:
    """V(e - e1_hat) of the table against the verifier's stage-1 Q^ehat(0, e) + k e^2.

    ``Q^ehat(0, e)`` is ``res.stages[1].q_mean_grid[0]`` (stage-1 root state, effort grid ``e``,
    opponent at its stage-1 mean ``e1_hat``): ``-k e^2 + sum_x w_x interp(e - e1_hat + x;
    D_2 grid, V_2^ehat)``. Besides the comparison, the discrepancy is decomposed into the
    verifier's interpolation of stage-2 values on its grid (A - B) and its Gauss-Legendre rule
    (B - C, A - D).
    """
    res = verify(verifier_policy(agent, spec, e1_hat), w_h=spec.w_h, w_l=spec.w_l, k=spec.k,
                 q=spec.q, T=2, cfg=cfg)
    if table is None:
        table = build_continuation_table(agent.actor, spec, stage=2)
    E = res.e_grid
    y = E - e1_hat
    verifier_q = res.stages[1].q_mean_grid[0] + spec.k * E ** 2          # C
    tab = table.lookup(y)                                                # A (through the table)
    s2 = res.stages[2]
    interp_exact = exact_interp_integral(s2.d_grid, s2.v_mean, spec.q, y)   # B
    gl_exact_g = np.array([np.sum(res.gl_weights * vc.g_frozen(agent.actor, spec, 2,
                                                               yy + res.gl_nodes)) for yy in y])
    diff = tab - verifier_q
    j = int(np.argmax(np.abs(diff)))
    dw = spec.dw
    return {
        "tier": cfg.name, "state_step": cfg.state_step, "effort_step": cfg.effort_step,
        "gl_half": cfg.gl_half, "valid": bool(res.valid), "n_effort_nodes": int(E.size),
        "e1_hat": float(e1_hat),
        "max_abs_diff": float(np.abs(diff).max()),
        "max_abs_diff_over_dw": float(np.abs(diff).max() / dw),
        "argmax_effort": float(E[j]),
        "mean_signed_diff_over_dw": float(diff.mean() / dw),
        "stage2_values_vs_g_max_over_dw": float(
            np.abs(vc.g_frozen(agent.actor, spec, 2, s2.d_grid) - s2.v_mean).max() / dw),
        "decomposition_over_dw": {
            "A_minus_C_total": float(np.abs(tab - verifier_q).max() / dw),
            "A_minus_B_interpolation_of_stage2_values": float(
                np.abs(tab - interp_exact).max() / dw),
            "B_minus_C_verifier_gl_on_interpolant": float(
                np.abs(interp_exact - verifier_q).max() / dw),
            "A_minus_D_verifier_gl_nodes_on_exact_g": float(np.abs(tab - gl_exact_g).max() / dw),
        },
    }


# ---------------------------------------------------------------------------
# (i) Monte-Carlo agreement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["parent", "synthetic"])
@pytest.mark.parametrize("q", QS)
def test_i_table_agrees_with_monte_carlo(kind: str, q: int) -> None:
    """Table value vs 1e6-draw Monte Carlo of E[g2(y + z)] within 3 standard errors."""
    actor, spec, table = get_table(kind, q)
    rng = np.random.default_rng(MC_SEED)
    z = rng.uniform(-q, q, size=MC_DRAWS) - rng.uniform(-q, q, size=MC_DRAWS)
    for y in Y_MC:
        g = g2_direct(actor, spec, y + z)             # evaluated directly, not via the table
        mc, se = float(g.mean()), float(g.std(ddof=1) / math.sqrt(MC_DRAWS))
        val = float(table.lookup(np.array([y]))[0])
        assert abs(val - mc) <= 3.0 * se, (kind, q, y, val, mc, se)


# ---------------------------------------------------------------------------
# (ii) agreement with the verifier's stage-1 Q
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="PROMPT.md 2.3 (ii) NOT MET on the production final tier: verifier-side errors, "
           "mainly its linear interpolation of stage-2 values on the D_2 grid (step 2.0, about "
           "2e-5 DW) and, at q=50 also, its Gauss-Legendre rule (32 nodes per half interval, "
           "about 1.3e-6 DW). Open for a PI decision; numbers in "
           "results/v2_refine/continuation_check.json. The tolerance is NOT loosened; remove "
           "this marker when the criterion is decided.")
@pytest.mark.parametrize("q", QS)
def test_ii_final_tier_within_spec_tolerance(q: int) -> None:
    """SPEC REQUIREMENT: max |V(e - e1_hat) - (Q^ehat(0, e) + k e^2)| <= 1e-6 DW, final tier.

    Expected to FAIL (hence ``xfail(strict=True)``); the measured values are recorded in
    ``results/v2_refine/continuation_check.json`` (``spec_requirement``). The causes sit on the
    verifier side, visible in ``decomposition_over_dw``: its linear interpolation of stage-2
    values on the D_2 grid (``A_minus_B``, the dominant term) and its Gauss-Legendre rule on
    the kinked integrand (``A_minus_D``: the verifier's own nodes applied to the exact g, no
    interpolation; above 1e-6 DW at q=50 only). The table's own quadrature error is about
    1e-8 DW (``convergence``). The requirement is NOT loosened here (PROMPT.md: stop and
    report); see also the refined-verifier-grid test below.
    """
    agent = parent_agent(q)
    if agent is None:
        pytest.skip(f"parent state absent: {parent_path(q)}")
    spec = spec_for(q)
    out = compare_with_verifier(agent, spec, E1_HAT[q], FINAL_CONFIG)
    print(f"\nq={q} final tier: max|V - Q| = {out['max_abs_diff']:.4e} "
          f"= {out['max_abs_diff_over_dw']:.3e} DW at e={out['argmax_effort']}")
    assert out["valid"]
    assert out["max_abs_diff_over_dw"] <= FINAL_TOL_OVER_DW, out["decomposition_over_dw"]


@pytest.mark.parametrize("q", QS)
def test_ii_dev_tier_reported_and_scales_with_state_step(q: int) -> None:
    """Dev tier: report the max difference; it exceeds the final tier's by about (4/2)^2.

    A linear-interpolation error scales with the square of the state-grid step (dev 4.0,
    final 2.0), so the dev/final ratio of the discrepancy is about 4 if interpolation is the
    cause (measured 4.1 at q=50, 4.3 at q=60). No numerical tolerance is specified for the dev
    tier.
    """
    agent = parent_agent(q)
    if agent is None:
        pytest.skip(f"parent state absent: {parent_path(q)}")
    spec = spec_for(q)
    dev = compare_with_verifier(agent, spec, E1_HAT[q], DEV_CONFIG)
    fin = compare_with_verifier(agent, spec, E1_HAT[q], FINAL_CONFIG)
    print(f"\nq={q} dev tier: max|V - Q| = {dev['max_abs_diff']:.4e} = "
          f"{dev['max_abs_diff_over_dw']:.3e} DW; dev/final = "
          f"{dev['max_abs_diff'] / fin['max_abs_diff']:.2f}")
    assert dev["valid"] and np.isfinite(dev["max_abs_diff"])
    assert 2.0 <= dev["max_abs_diff"] / fin["max_abs_diff"] <= 8.0


@pytest.mark.parametrize("cfg", [DEV_CONFIG, FINAL_CONFIG], ids=lambda c: c.name)
@pytest.mark.parametrize("q", QS)
def test_ii_diag_stage2_values_identical(q: int, cfg: VerifierConfig) -> None:
    """The verifier's stage-2 values on its grid are exactly g2 of the table (float rounding)."""
    agent = parent_agent(q)
    if agent is None:
        pytest.skip(f"parent state absent: {parent_path(q)}")
    spec = spec_for(q)
    res = verify(verifier_policy(agent, spec, E1_HAT[q]), w_h=W_H, w_l=W_L, k=K, q=float(q), T=2,
                 cfg=cfg)
    s2 = res.stages[2]
    g = vc.g_frozen(agent.actor, spec, 2, s2.d_grid)
    assert np.abs(g - s2.v_mean).max() <= 1e-12 * spec.dw


@pytest.mark.parametrize("q", QS)
def test_ii_diag_identity_holds_on_refined_verifier_grid(q: int) -> None:
    """With the verifier's state grid refined (step 0.25, 64 GL nodes) the identity is <= 1e-6 DW.

    Causal check that the final-tier discrepancy comes from the verifier's interpolation of
    stage-2 values on its D_2 grid and not from the table's quadrature (which does not depend
    on the verifier grid): the discrepancy falls with the state step (measured 5.0e-7 DW at
    q=50 and 4.1e-7 DW at q=60).
    """
    agent = parent_agent(q)
    if agent is None:
        pytest.skip(f"parent state absent: {parent_path(q)}")
    spec = spec_for(q)
    fine = VerifierConfig("final_refined_state_0.25", state_step=0.25, effort_step=0.5, gl_half=64)
    out = compare_with_verifier(agent, spec, E1_HAT[q], fine)
    print(f"\nq={q} refined verifier grid: {out['max_abs_diff_over_dw']:.3e} DW")
    assert out["max_abs_diff_over_dw"] <= FINAL_TOL_OVER_DW


# ---------------------------------------------------------------------------
# (iii) lookup, symmetry, determinism, float path
# ---------------------------------------------------------------------------

def test_iii_default_grid_and_dtypes() -> None:
    """Default grid: step <= 0.05, covers [-e_range, e_range], symmetric, contains 0, float64."""
    spec = spec_for(50)
    grid = vc.y_grid_for(spec)
    assert grid.dtype == np.float64
    assert np.diff(grid).max() <= 0.05 + 1e-12
    assert grid[0] == -spec.e_range and grid[-1] == spec.e_range
    assert np.allclose(grid, -grid[::-1], atol=1e-12, rtol=0.0)
    assert 0.0 in grid
    _, _, table = get_table("synthetic", 50)
    assert table.values.dtype == np.float64 and table.y_grid.dtype == np.float64
    assert table.values.shape == table.y_grid.shape
    assert table.meta["step_actual"] <= 0.05 + 1e-12 and table.meta["n_y"] == grid.size
    assert abs(table.meta["weight_sum"] - 1.0) <= 1e-13
    assert table.meta["stage"] == 2 and table.meta["panel_width"] <= 1.0


def test_iii_lookup_exact_at_nodes_and_linear_between() -> None:
    """Lookup is exact at nodes and reproduces a linear function between them."""
    _, _, table = get_table("synthetic", 50)
    assert np.array_equal(table.lookup(table.y_grid), table.values)
    lin = ContinuationTable(table.y_grid.copy(), 3.0 + 0.7 * table.y_grid)
    ys = np.random.default_rng(1).uniform(-100.0, 100.0, size=1000)
    assert np.allclose(lin.lookup(ys), 3.0 + 0.7 * ys, rtol=0.0, atol=1e-12)
    mid = 0.5 * (table.y_grid[10] + table.y_grid[11])
    expect = 0.5 * (table.values[10] + table.values[11])
    assert abs(float(table.lookup(np.array([mid]))[0]) - expect) <= 1e-13
    assert table.lookup(np.zeros((3, 2))).shape == (3, 2)


def test_iii_lookup_refuses_outside_grid() -> None:
    """Outside the grid beyond float noise lookup raises; float noise at the edge is accepted."""
    _, _, table = get_table("synthetic", 50)
    with pytest.raises(ValueError):
        table.lookup(np.array([100.001]))
    with pytest.raises(ValueError):
        table.lookup(np.array([0.0, -100.5]))
    with pytest.raises(ValueError):
        table.lookup(np.array([np.nan]))
    edge = table.lookup(np.array([100.0 + 1e-12, -100.0 - 1e-12]))
    assert edge[0] == table.values[-1] and edge[1] == table.values[0]


def test_iii_constant_policy_matches_closed_form_and_is_symmetric() -> None:
    """Constant mean policy: V(y) = w_l - k c^2 + DW P(S4 <= y) exactly (sum of four U(-q, q)).

    A fresh BetaActor (zero output head) has mean 1/2 for every d, so c = 50. The odd part
    V(y) - V(0 centre) is antisymmetric: V(y) + V(-y) = 2 (w_l - k c^2) + DW.
    """
    for q in QS:
        spec = spec_for(q)
        agent = new_agent()
        table = build_continuation_table(agent.actor, spec, stage=2)
        c = 50.0
        exact = (spec.w_l - spec.k * c ** 2
                 + spec.dw * irwin_hall4_cdf(table.y_grid / (2.0 * q) + 2.0))
        err = np.abs(table.values - exact).max() / spec.dw
        sym = np.abs(table.values + table.values[::-1]
                     - (2.0 * (spec.w_l - spec.k * c ** 2) + spec.dw)).max() / spec.dw
        print(f"\nq={q} constant policy: |V - exact|/DW = {err:.3e}, symmetry = {sym:.3e}")
        assert err <= 1e-8
        assert sym <= 1e-8


def test_iii_even_policy_symmetry_identity() -> None:
    """Even mean policy: V(y) + V(-y) = 2 w_l + DW - 2 k E_z[e_hat(y + z)^2] (independent check)."""
    spec = spec_for(50)
    actor = EvenActor()
    table = build_continuation_table(actor, spec, stage=2)
    zs = np.linspace(-2 * spec.q, 2 * spec.q, 40001)
    dens = (2 * spec.q - np.abs(zs)) / (4 * spec.q ** 2)
    wts = dens * (zs[1] - zs[0])
    wts[[0, -1]] *= 0.5
    ys = table.y_grid[::200]
    e2 = np.array([np.sum(wts * vc.frozen_mean_effort(actor, spec, 2, y + zs) ** 2) for y in ys])
    lhs = table.lookup(ys) + table.lookup(-ys)
    rhs = 2 * spec.w_l + spec.dw - 2 * spec.k * e2
    assert np.abs(lhs - rhs).max() / spec.dw <= 1e-7


@pytest.mark.parametrize("kind", ["parent", "synthetic"])
def test_iii_build_is_deterministic(kind: str) -> None:
    """Two builds with identical inputs are bitwise equal (coarser y step for speed)."""
    actor, spec = get_actor(kind, 50), spec_for(50)
    t1 = build_continuation_table(actor, spec, stage=2, step=0.25)
    t2 = build_continuation_table(actor, spec, stage=2, step=0.25)
    assert np.array_equal(t1.y_grid, t2.y_grid)
    assert np.array_equal(t1.values, t2.values)


def test_iii_float_path_matches_the_rollout() -> None:
    """e_hat is bitwise the rollout's ``mean`` path; g equals ``expected_terminal_reward``."""
    agent = parent_agent(50)
    if agent is None:
        pytest.skip(f"parent state absent: {parent_path(50)}")
    spec = spec_for(50)
    d = np.random.default_rng(3).uniform(-200.0, 200.0, size=5000)
    a, b = agent.beta_params(spec.encode_obs(2, d), agent.actor)          # as in collect_batch_v2
    m = a.astype(float) / (a.astype(float) + b.astype(float))
    assert np.array_equal(vc.frozen_mean_effort(agent.actor, spec, 2, d),
                          spec.e_min + spec.e_range * m)
    a_o, b_o = agent.beta_params(spec.encode_obs(2, -d), agent.actor)
    m_o = a_o.astype(float) / (a_o.astype(float) + b_o.astype(float))
    ref = expected_terminal_reward(spec, d, spec.e_min + spec.e_range * m,
                                   spec.e_min + spec.e_range * m_o)
    assert np.allclose(vc.g_frozen(agent.actor, spec, 2, d), ref, rtol=0.0, atol=1e-12)


def test_iii_stage_must_be_the_final_stage() -> None:
    """The integrated stage is the final one of the game; anything else is refused."""
    spec = spec_for(50)
    for bad in (1, 3):
        with pytest.raises(ValueError):
            build_continuation_table(synthetic_actor(), spec, stage=bad)


def test_iii_quadrature_weights_and_panels() -> None:
    """Weights sum to 1, nodes are interior, z = 0 and z = +-2q are panel edges, width <= 1."""
    for q in QS:
        nodes, weights, n_panels = vc.shock_quadrature(q, 1.0, 6)
        assert abs(weights.sum() - 1.0) <= 1e-13
        assert np.all(np.abs(nodes) < 2 * q) and np.all(weights > 0)
        assert n_panels == 2 * math.ceil(2 * q / 1.0) and nodes.size == 6 * n_panels
        assert np.allclose(nodes, -nodes[::-1], atol=1e-12)
        assert abs(float(np.sum(weights * nodes ** 2)) - 2 * q ** 2 / 3) <= 1e-10   # Var z = 2q^2/3


# ---------------------------------------------------------------------------
# Check record
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def convergence_record(actor: Any, spec: GameSpec) -> List[Dict[str, Any]]:
    """Max |change| of the default-grid table when halving the panel width and doubling nodes."""
    grid = vc.y_grid_for(spec)
    cache: Dict[Tuple[float, int], np.ndarray] = {}

    def table_at(width: float, n: int) -> np.ndarray:
        if (width, n) not in cache:
            cache[(width, n)] = vc.expected_continuation(actor, spec, 2, grid, width, n)
        return cache[(width, n)]

    rows = []
    for (w1, n1) in ((2.0, 3), (1.0, 6), (0.5, 12)):
        w2, n2 = w1 / 2.0, 2 * n1
        rows.append({
            "from_panel_width": w1, "from_nodes_per_panel": n1,
            "to_panel_width": w2, "to_nodes_per_panel": n2,
            "n_nodes_from": int(vc.shock_quadrature(spec.q, w1, n1)[0].size),
            "n_nodes_to": int(vc.shock_quadrature(spec.q, w2, n2)[0].size),
            "max_abs_change_over_dw": float(np.abs(table_at(w1, n1) - table_at(w2, n2)).max()
                                            / spec.dw),
        })
    return rows


def batch_noise_record(actor: Any, spec: GameSpec, table: ContinuationTable
                       ) -> Dict[str, float]:
    """Float32 batch-composition noise of V, in units of DW (probe of a HYPOTHESIS).

    Hypothesis (from an earlier report, untested then): the change of the table under rule
    refinement (``convergence``) is float32 noise from different batch compositions of the actor
    forward pass, not quadrature error. Probe: recompute V at every 20th y-node of the default
    grid with the default rule but different batch compositions (all probe nodes in one call,
    then one y per call) and compare with the table values at the same nodes. The verdict is
    computed in ``write_check_record`` (``diagnosis.float32_batch_noise_hypothesis``).
    """
    idx = np.arange(0, table.y_grid.size, 20)
    y = table.y_grid[idx]
    ref = table.values[idx]
    sub = vc.expected_continuation(actor, spec, 2, y)
    single = np.array([vc.expected_continuation(actor, spec, 2, y[i:i + 1])[0]
                       for i in range(y.size)])
    return {"n_probe_nodes": int(y.size),
            "all_probe_nodes_one_call_vs_table_over_dw": float(np.abs(sub - ref).max() / spec.dw),
            "one_y_per_call_vs_table_over_dw": float(np.abs(single - ref).max() / spec.dw)}


def write_check_record(path: Path = OUT_JSON) -> Dict[str, Any]:
    """Compute and write the check record (convergence, timing, verifier comparison)."""
    rec: Dict[str, Any] = {
        "schema": "v2_refine_continuation_check/1",
        "generated_by": "tests/test_v2_refine_continuation.py --write",
        "game": {"w_h": W_H, "w_l": W_L, "k": K, "dw": W_H - W_L, "T": 2},
        "integration_rule": {
            "rule": "composite Gauss-Legendre in z (density-weighted), panels aligned at z=0 and "
                    "z=+-2q, frozen actor evaluated directly at d=y+z (no interpolation of g)",
            "panel_width": vc.DEFAULT_PANEL_WIDTH, "nodes_per_panel": vc.DEFAULT_NODES_PER_PANEL,
            "y_step": vc.DEFAULT_STEP,
        },
        "parents": {}, "timing": {}, "convergence": {}, "table_vs_verifier": {},
        "batch_noise_probe": {},
        "diagnosis": {"verifier_grid_sweep_max_abs_diff_over_dw": {}},
    }
    t_total = time.time()
    for q in QS:
        spec = spec_for(q)
        agent = parent_agent(q)
        assert agent is not None, f"parent state absent: {parent_path(q)}"
        rec["parents"][f"q{q}"] = {"path": str(parent_path(q)),
                                   "sha256": sha256_of(parent_path(q)), "seed": PARENT_SEED,
                                   "e1_hat_fixed_opponent": E1_HAT[q]}
        # timing: one default table build per actor
        t0 = time.perf_counter()
        table = build_continuation_table(agent.actor, spec, stage=2)
        rec["timing"][f"q{q}_parent"] = {"build_seconds_wall": time.perf_counter() - t0,
                                         **{k: table.meta[k] for k in
                                            ("n_y", "n_panels", "nodes_per_panel", "n_nodes")},
                                         "threads": 1}
        rec["convergence"][f"q{q}_parent"] = convergence_record(agent.actor, spec)
        rec["convergence"][f"q{q}_synthetic"] = convergence_record(synthetic_actor(), spec)
        rec["batch_noise_probe"][f"q{q}_parent"] = batch_noise_record(agent.actor, spec, table)
        for cfg in (DEV_CONFIG, FINAL_CONFIG):
            out = compare_with_verifier(agent, spec, E1_HAT[q], cfg, table)
            if cfg.name == FINAL_CONFIG.name:
                out["spec_tolerance_over_dw"] = FINAL_TOL_OVER_DW
                out["meets_spec_tolerance"] = bool(out["max_abs_diff_over_dw"] <= FINAL_TOL_OVER_DW)
            rec["table_vs_verifier"].setdefault(cfg.name, {})[f"q{q}"] = out
        sweep: Dict[str, Dict[str, float]] = {}
        for ss in (4.0, 2.0, 1.0, 0.5, 0.25):
            sweep[f"state_step_{ss}"] = {}
            for gh in (16, 32, 64, 128):
                cfg = VerifierConfig("sweep", state_step=ss, effort_step=0.5, gl_half=gh)
                sweep[f"state_step_{ss}"][f"gl_half_{gh}"] = compare_with_verifier(
                    agent, spec, E1_HAT[q], cfg, table)["max_abs_diff_over_dw"]
        rec["diagnosis"]["verifier_grid_sweep_max_abs_diff_over_dw"][f"q{q}"] = sweep
    fin = rec["table_vs_verifier"]["final"]
    sweep_all = rec["diagnosis"]["verifier_grid_sweep_max_abs_diff_over_dw"]
    met = bool(all(fin[f"q{q}"]["meets_spec_tolerance"] for q in QS))
    rec["spec_requirement"] = {
        "statement": "PROMPT.md 2.3 (ii): on the final tier max |V(e-e1_hat) - (Q + k e^2)| "
                     "<= 1e-6 DW",
        "met": met,
        "status": "MET" if met else "NOT_MET (open: needs a PI decision; nothing was loosened)",
        "final_tier_max_abs_diff_over_dw": {f"q{q}": fin[f"q{q}"]["max_abs_diff_over_dw"]
                                            for q in QS},
        "dev_tier_max_abs_diff_over_dw": {
            f"q{q}": rec["table_vs_verifier"]["development"][f"q{q}"]["max_abs_diff_over_dw"]
            for q in QS},
        "final_tier_terms_over_dw": {f"q{q}": {
            "total": fin[f"q{q}"]["max_abs_diff_over_dw"],
            "verifier_stage2_interpolation_A_minus_B": fin[f"q{q}"]["decomposition_over_dw"][
                "A_minus_B_interpolation_of_stage2_values"],
            "verifier_gl_nodes_on_exact_g_A_minus_D": fin[f"q{q}"]["decomposition_over_dw"][
                "A_minus_D_verifier_gl_nodes_on_exact_g"],
            "interpolation_alone_exceeds_spec": bool(fin[f"q{q}"]["decomposition_over_dw"][
                "A_minus_B_interpolation_of_stage2_values"] > FINAL_TOL_OVER_DW),
            "gl_rule_alone_exceeds_spec": bool(fin[f"q{q}"]["decomposition_over_dw"][
                "A_minus_D_verifier_gl_nodes_on_exact_g"] > FINAL_TOL_OVER_DW),
        } for q in QS},
        "options_for_pi": {
            "a_literal_production_final_tier": "fails (see final_tier_max_abs_diff_over_dw)",
            "b_refined_verifier_grid_state_step_0.25_gl_half_64_over_dw": {
                f"q{q}": sweep_all[f"q{q}"]["state_step_0.25"]["gl_half_64"] for q in QS},
            "c_restated_tolerance": "any value above final_tier_max_abs_diff_over_dw",
        },
    }
    rec["diagnosis"]["conclusion"] = (
        "Facts only; the numbers are the evidence. Table side: the change under rule refinement "
        "is in `convergence` (units DW). Verifier side: `table_vs_verifier.*.decomposition_"
        "over_dw` splits the discrepancy into A_minus_B (the verifier's linear interpolation of "
        "stage-2 values on its D_2 grid), A_minus_D (the verifier's Gauss-Legendre nodes applied "
        "to the exact g, no interpolation) and B_minus_C (its rule on the interpolant). The "
        "verifier's stage-2 values equal g2 at its nodes to float rounding "
        "(`stage2_values_vs_g_max_over_dw`). The sweep shows the total as a function of the "
        "verifier's state_step and gl_half. Which terms exceed the spec tolerance is in "
        "`spec_requirement.final_tier_terms_over_dw`.")
    noise = max(v for p in rec["batch_noise_probe"].values() for kk, v in p.items()
                if kk != "n_probe_nodes")
    last = {kk: v[-1]["max_abs_change_over_dw"] for kk, v in rec["convergence"].items()}
    decays = {kk: all(v[i + 1]["max_abs_change_over_dw"] < v[i]["max_abs_change_over_dw"]
                      for i in range(len(v) - 1)) for kk, v in rec["convergence"].items()}
    rejected = bool(noise < 0.01 * min(last.values()) and all(decays.values()))
    rec["diagnosis"]["float32_batch_noise_hypothesis"] = {
        "claim_tested": "the change of the table under rule refinement (`convergence`) is float32 "
                        "noise from the batch composition of the actor forward pass",
        "probe": "`batch_noise_probe`: same rule, V recomputed with other batch compositions",
        "max_batch_noise_over_dw": noise,
        "smallest_last_rung_change_over_dw": min(last.values()),
        "change_decreases_at_every_refinement": decays,
        "verdict": ("REJECTED: probe noise is below 1% of the smallest last-rung change and the "
                    "change shrinks at every refinement, so the changes are quadrature "
                    "differences, not batch noise" if rejected else
                    "NOT REJECTED by this probe"),
    }
    rec["wall_seconds_total"] = time.time() - t_total
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    return rec


if __name__ == "__main__":
    if "--write" in sys.argv:
        out = write_check_record()
        print(json.dumps(out["spec_requirement"], indent=2))
    else:
        raise SystemExit(pytest.main([__file__, "-p", "no:cacheprovider", "-q"]))
