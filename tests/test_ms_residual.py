"""Tests of the MS-R1 residual metrics (spec 2.2 "Residual metrics (D3)"): ``utils/ms_residual.py``.

Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_ms_residual.py -q

All verifier calls are single-threaded T=2 calls at DEV_CONFIG (about 5 ms each). The v2.0 game:
w_h = 6, w_l = 2, k = 1/3500, e in [0, 100], q in {50, 60}; closed form e2*(d) = DW f_xi(d) / (2k)
with peak 70 (q = 50) / 58.33 (q = 60). The closed form is used here ONLY to build candidate
policies whose one-step best response is known; ``stage_diag`` never sees it.

Measured grid noise of the dev-tier best response (q = 50, DEV_CONFIG, state step 4, effort step 1):
  * candidate = closed form: R = 2.9e-14 (q = 60: 1.8e-14), max r = 2.1e-12 effort units;
  * damped best-response iteration reaches the fixed point R = 0.0 in ~57 iterations;
  * candidate = closed form + c with the opponent at the closed form: r = c +- 2e-12;
  * the verifier refines a grid best response only by an INTERIOR parabola vertex, so a true best
    response in (0, 1) (below the first grid node) is returned as 0: error up to one effort step
    (0.467 effort units at d = -96 for the symmetric shift c = 1, see ``test_symmetric_shift``).

Why the "shifted candidate" tests of the brief are done with the opponent fixed at the closed form:
in the symmetric verifier the opponent plays the SAME shifted policy, so the best response moves by
x = -a c / (2k - a) (trailing, d <= 0) or +a c / (2k + a) (leading, d > 0), a = DW / (4 q^2), and
r(d) = c 2k / (2k -+ a) rather than c. ``verify(opponent_policy=...)`` (documented, used for MC-BR)
fixes the opponent and gives r = c exactly; the symmetric case is checked against an independent
exact best-response solver and the closed-form factors.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ.setdefault(f"{_k}_NUM_THREADS", "1")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import utils.theory_multistage as theory  # noqa: E402
import utils.v2_metrics as v2m  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler, gap_bin_index  # noqa: E402
from utils import ms_residual  # noqa: E402
from utils.dp_br_verifier import (DEV_CONFIG, concentration_stats, stage_grid,  # noqa: E402
                                  verify)
from utils.ms_residual import StageDiag, ema_update, invalid_diag, stage_diag  # noqa: E402
from utils.ms_rule import StageController, StageRule  # noqa: E402
from utils.theory_multistage import F_xi, g2_two_stage  # noqa: E402

RECORDS = json.loads((ROOT / "protocols" / "v2_T2_locked_v2_0.json").read_text())["records"]
E1_FIXED = 40.0              # stage-1 policy of the T=2 test candidates (irrelevant for stage 2)
CONC_OK = {"valid": True, "max_std_norm": 0.01}
R_TOL = 1e-9                 # tolerance for "R = 0" claims (measured <= 3e-14)
R_EFFORT_TOL = 1e-9          # tolerance on r = c (measured 2e-12 effort units)


def spec_for(q: int, T: int = 2) -> GameSpec:
    """The v2.0 game of this q."""
    return GameSpec(**{**RECORDS[str(q)]["game"], "q": float(q), "T": int(T)})


def closed_form(spec: GameSpec, d: np.ndarray) -> np.ndarray:
    """Closed-form stage-2 effort (test fixture only)."""
    return g2_two_stage(np.asarray(d, dtype=float), spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max)


def policy_from(stage2: Callable[[np.ndarray], np.ndarray]) -> Callable:
    """Verifier policy: stage 1 constant, stage 2 the given function of d."""
    def pol(t: int, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        return np.full(d.shape, E1_FIXED) if t == 1 else stage2(d)
    return pol


def shifted(spec: GameSpec, c: float, region: str = "interior") -> Callable:
    """Closed form plus c on |d| < 2q (``interior``), on |d| >= 2q (``tail``) or everywhere."""
    def f(d: np.ndarray) -> np.ndarray:
        g = closed_form(spec, d)
        inside = np.abs(d) < 2.0 * spec.q
        sel = {"interior": inside, "tail": ~inside, "all": np.ones_like(inside)}[region]
        return g + c * sel
    return policy_from(f)


def run_verify(spec: GameSpec, pol: Callable, opponent: Optional[Callable] = None):
    """DEV-tier verifier call of a T=2 candidate (optionally against a fixed opponent)."""
    return verify(pol, w_h=spec.w_h, w_l=spec.w_l, k=spec.k, q=spec.q, T=spec.T, e_min=spec.e_min,
                  e_max=spec.e_max, cfg=DEV_CONFIG, opponent_policy=opponent)


def exact_br(spec: GameSpec, d: float, e_opp: float) -> float:
    """Exact terminal-stage best response argmax_e -k e^2 + DW F_xi(d + e - e_opp) on [0, e_max].

    Q is concave (q > q_soc) and piecewise quadratic with kinks of the second derivative at
    d + e - e_opp in {-2q, 0, 2q}; the global maximiser is among the box ends, the three kink
    efforts and the stationary points of the two linear-density pieces.
    """
    q, k, dw = spec.q, spec.k, spec.dw
    a = dw / (4.0 * q * q)
    cands = [0.0, spec.e_max] + [yk - d + e_opp for yk in (-2.0 * q, 0.0, 2.0 * q)]
    cands.append(a * (2.0 * q + d - e_opp) / (2.0 * k - a))
    cands.append(a * (2.0 * q - d + e_opp) / (2.0 * k + a))
    c = np.clip(np.array(cands), 0.0, spec.e_max)
    val = -k * c ** 2 + dw * F_xi(d + c - e_opp, q)
    return float(c[int(np.argmax(val))])


def diag_of(res, spec: GameSpec, stage: int = 2) -> StageDiag:
    """StageDiag of a verifier result with a fixed healthy concentration."""
    return stage_diag(res, spec, stage, CONC_OK)


def same_bits(a: StageDiag, b: StageDiag) -> bool:
    """Bit-identical comparison of two StageDiag (NaN equal to NaN)."""
    for name in a.__dataclass_fields__:
        x, y = getattr(a, name), getattr(b, name)
        if np.asarray(x, dtype=float).tobytes() != np.asarray(y, dtype=float).tobytes():
            return False
    return True


# ----------------------------------------------------------------------------------------------
# (a) candidate equal to the verifier's own best response gives R ~ 0
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("q,peak", [(50, 70.0), (60, 4.0 * 3500.0 / (4.0 * 60.0))])
def test_closed_form_candidate_has_zero_residual(q, peak):
    """The closed form is a fixed point of the stage-2 best response: R <= 1e-9 (measured 3e-14)."""
    spec = spec_for(q)
    res = run_verify(spec, policy_from(lambda d: closed_form(spec, d)))
    sd = diag_of(res, spec)
    s2 = res.stages[2]
    assert res.valid and sd.valid and sd.tail_term
    assert sd.R <= R_TOL and abs(sd.R_tail) <= 1e-12
    assert np.abs(s2.e_hat - s2.a_dev).max() <= 1e-9                     # raw r in effort units
    assert sd.s == pytest.approx(peak, abs=1e-9)
    assert sd.delta_over_dw <= 1e-12
    assert sd.n_nodes_nontail + sd.n_nodes_tail == s2.d_grid.size


def test_best_response_fixed_point_has_zero_residual():
    """Damped iteration P <- (P + a_dev) / 2 from a shifted start converges to R = 0 at the nodes
    (the fixed point P = a_dev of the verifier's own best response)."""
    spec = spec_for(50)
    G = stage_grid(2, spec.B, DEV_CONFIG.state_step)
    P = closed_form(spec, G) + 3.0 * (np.abs(G) < 2 * spec.q)
    first_R = None
    last = None
    for it in range(400):
        pol = policy_from(lambda d, P=P: np.interp(d, G, P))
        res = run_verify(spec, pol)
        last = diag_of(res, spec)
        first_R = last.R if first_R is None else first_R
        if last.R <= 1e-12:
            break
        P = 0.5 * P + 0.5 * res.stages[2].a_dev
    assert first_R > 0.15                      # the start is far from the fixed point (R = 0.159)
    assert last.R <= 1e-10, (it, last.R)
    assert it < 200
    # and the fixed point is (to the grid noise, measured 2.4e-6 effort units) the closed form
    assert np.abs(P - closed_form(spec, G)).max() <= 1e-4


# ----------------------------------------------------------------------------------------------
# (b) shifted candidates
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("q,c", [(50, 0.25), (50, 1.0), (50, 3.0), (60, 1.0)])
def test_shift_by_c_against_the_closed_form_opponent_gives_r_equal_c(q, c):
    """Candidate = closed form + c on |d| < 2q, opponent at the closed form: its best response is
    the closed form, so r_t(d) = c on every non-tail node (+-1e-9 effort units; measured 2e-12)
    and R * s = c, argmax inside the interior; the tail (candidate unchanged, 0) has R_tail = 0.

    Exception (verifier grid noise, q = 60 only): at d = 0 the exact best response 58.3333 sits at
    the y = 0 kink of the objective BETWEEN the effort nodes 58 and 59, the 3-point parabola is not
    exact across a kink, and the verifier returns 58.0867 (error 0.2467 < the effort step 1). At
    q = 50 the peak 70 is an effort node and the error is 0. That node is checked separately, and
    ``s`` and ``R * s`` carry the same error there.
    """
    spec = spec_for(q)
    pol = shifted(spec, c)
    opp = policy_from(lambda d: closed_form(spec, d))
    res = run_verify(spec, pol, opponent=opp)
    s2 = res.stages[2]
    sd = diag_of(res, spec)
    nontail = np.abs(s2.d_grid) < 2.0 * q
    r = np.abs(s2.e_hat - s2.a_dev)
    peak = float(closed_form(spec, np.zeros(1))[0])
    kink_straddle = nontail & (s2.d_grid == 0.0) & (abs(peak - round(peak)) > 1e-9)
    strict = nontail & ~kink_straddle
    assert np.abs(r[strict] - c).max() <= R_EFFORT_TOL
    if kink_straddle.any():
        err0 = abs(float(s2.a_dev[kink_straddle][0]) - peak)
        assert 0.1 < err0 < DEV_CONFIG.effort_step                    # measured 0.2467
        assert sd.s == pytest.approx(peak, abs=DEV_CONFIG.effort_step)
        assert sd.R * sd.s == pytest.approx(c, abs=DEV_CONFIG.effort_step)
    else:
        assert sd.s == pytest.approx(peak, abs=1e-9)                  # a_dev(0)
        assert sd.R * sd.s == pytest.approx(c, abs=R_EFFORT_TOL)
    assert abs(sd.argmax_d) < 2.0 * q
    assert abs(sd.R_tail) <= 1e-12
    # per-bin map: every non-tail bin carries c / s (its nodes all have r = c, bar the d = 0 node)
    nt_bins = ~StartSampler(spec, 10.0).tail_mask(2)
    expect = np.where(nt_bins, c / sd.s, np.nan)
    if kink_straddle.any():
        assert np.allclose(sd.rho_bins[nt_bins], c / sd.s, atol=DEV_CONFIG.effort_step / sd.s)
    else:
        assert np.allclose(sd.rho_bins[nt_bins], expect[nt_bins], rtol=1e-9)
    assert np.isnan(sd.rho_bins[~nt_bins]).all()


def test_tail_shift_gives_r_tail_equal_c_over_s():
    """Candidate shifted by c on the tail only: the non-tail residual is 0 and R_tail = c / s."""
    spec = spec_for(50)
    c = 0.7
    res = run_verify(spec, shifted(spec, c, "tail"),
                     opponent=policy_from(lambda d: closed_form(spec, d)))
    sd = diag_of(res, spec)
    assert sd.R <= R_TOL
    assert sd.R_tail == pytest.approx(c / 70.0, rel=1e-12)          # mean{e_hat : |d| >= 2q} / s
    assert sd.tail_term


@pytest.mark.parametrize("c", [0.25, 3.0])
def test_symmetric_shift_matches_an_independent_exact_best_response(c):
    """Both players play the shifted policy (the pipeline's situation). The verifier's a_dev is the
    exact best response to 1e-9 here, and R equals max r_exact / s_exact; with the analytic factors
    r(d <= 0) = c 2k / (2k - a), r(d > 0) = c 2k / (2k + a) in the linear regime."""
    spec = spec_for(50)
    pol = shifted(spec, c)
    res = run_verify(spec, pol)
    s2 = res.stages[2]
    sd = diag_of(res, spec)
    G = s2.d_grid
    p_opp = pol(2, -G)
    br = np.array([exact_br(spec, float(d), float(o)) for d, o in zip(G, p_opp)])
    assert np.abs(s2.a_dev - br).max() <= 1e-9
    nontail = np.abs(G) < 2.0 * spec.q
    r_exact = np.abs(pol(2, G) - br)
    j0 = int(np.argmin(np.abs(G)))
    assert sd.s == pytest.approx(br[j0], abs=1e-9)
    assert sd.R == pytest.approx(r_exact[nontail].max() / br[j0], abs=1e-9)
    a, k2 = spec.dw / (4.0 * spec.q ** 2), 2.0 * spec.k
    lin = np.abs(G) <= 88.0                       # best response stays interior (no clipping at 0)
    expect = np.where(G <= 0.0, c * k2 / (k2 - a), c * k2 / (k2 + a))
    assert np.abs(r_exact[lin] - expect[lin]).max() <= 1e-9
    assert sd.R * sd.s == pytest.approx(c * k2 / (k2 - a), abs=1e-9)    # attained at d <= 0


def test_symmetric_shift_c1_documents_the_one_effort_step_boundary_noise():
    """c = 1: the exact best response at d = -96 is 0.467 (in (0, 1)); the verifier returns 0 (no
    interior vertex), so R is overstated by <= effort_step / s. Everywhere else it is exact."""
    spec = spec_for(50)
    pol = shifted(spec, 1.0)
    res = run_verify(spec, pol)
    s2 = res.stages[2]
    sd = diag_of(res, spec)
    G = s2.d_grid
    br = np.array([exact_br(spec, float(d), float(o)) for d, o in zip(G, pol(2, -G))])
    err = np.abs(s2.a_dev - br)
    assert err.max() < DEV_CONFIG.effort_step
    assert err.max() == pytest.approx(0.4666666667, abs=1e-6) and G[int(np.argmax(err))] == -96.0
    assert np.sort(err)[-2] <= 1e-9                                  # exactly one node is off
    nontail = np.abs(G) < 2.0 * spec.q
    r_exact = np.abs(pol(2, G) - br)
    j0 = int(np.argmin(np.abs(G)))
    r_expect = r_exact[nontail].max() / br[j0]
    assert r_expect == pytest.approx(0.04926, abs=1e-5)
    assert sd.R == pytest.approx(0.05616, abs=1e-5) and sd.argmax_d == -96.0
    assert abs(sd.R - r_expect) <= DEV_CONFIG.effort_step / sd.s


def test_dev_tier_best_response_noise_over_a_sweep_of_symmetric_shifts():
    """Grid noise of the dev-tier first-order residual for the candidates of the tests above: for
    c in {0, 0.25, ..., 4} the verifier's a_dev differs from the exact best response by < 1 effort
    step, and R differs from the exact R by <= effort_step / s. Measured over c in {0, 0.1, ..., 4}:
    max |a_dev - exact| = 0.467 (q = 50) and 0.483 (q = 60) effort units, max |R - R_exact| =
    0.0075 (q = 50) and 0.0088 (q = 60), i.e. about a quarter of rho = 0.03. The error occurs where
    the exact best response lies in (0, 1) (no interior vertex below the first effort node) or at a
    kink of the objective between two effort nodes."""
    for q in (50, 60):
        spec = spec_for(q)
        for c in np.arange(0.0, 4.01, 0.25):
            pol = shifted(spec, float(c))
            res = run_verify(spec, pol)
            s2, sd = res.stages[2], diag_of(res, spec)
            G = s2.d_grid
            br = np.array([exact_br(spec, float(d), float(o)) for d, o in zip(G, pol(2, -G))])
            assert np.abs(s2.a_dev - br).max() < DEV_CONFIG.effort_step, (q, c)
            nt = np.abs(G) < 2.0 * q
            j0 = int(np.argmin(np.abs(G)))
            r_exact = np.abs(pol(2, G) - br)[nt].max() / br[j0]
            assert abs(sd.R - r_exact) <= 2.0 * DEV_CONFIG.effort_step / br[j0], (q, c)
            assert np.nanmax(sd.rho_bins) == pytest.approx(sd.R, rel=1e-12, abs=1e-15)


# ----------------------------------------------------------------------------------------------
# (c) node-to-bin aggregation (hand-made inputs)
# ----------------------------------------------------------------------------------------------

def fake_result(spec: GameSpec, stage: int, d: np.ndarray, e_hat: np.ndarray, a_dev: np.ndarray,
                delta_max: float = 0.004, valid: bool = True) -> SimpleNamespace:
    """The few fields of a VerifierResult that ``stage_diag`` reads."""
    st = SimpleNamespace(d_grid=d, e_hat=e_hat, a_dev=a_dev)
    return SimpleNamespace(stages={stage: st}, valid=valid, dw=spec.dw,
                           full_delta_max={stage: delta_max})


def test_node_to_bin_aggregation_by_hand_t2_q50():
    spec = spec_for(50)
    # -200, -196, ..., 200 (101 nodes)
    d = stage_grid(2, spec.B, 4.0)
    assert d.size == 101 and d[50] == 0.0
    e_hat = np.where(np.abs(d) < 100.0, 30.0, 0.4)                  # tail nodes: e_hat = 0.4
    r = np.zeros(d.size)
    ratio_at = {-96.0: 0.01, -92.0: 0.03, 8.0: 0.05, 12.0: 0.02}    # residual / s with s = 30
    for dd, rt in ratio_at.items():
        r[np.flatnonzero(d == dd)[0]] = rt * 30.0
    r[np.flatnonzero(d == -100.0)[0]] = 12.0       # tail NODE inside the non-tail bin [-100, -90)
    r[np.flatnonzero(d == 100.0)[0]] = 9.0         # tail node, tail bin
    r[np.flatnonzero(d == -200.0)[0]] = 6.0
    a_dev = e_hat + r * np.where(np.arange(d.size) % 2 == 0, 1.0, -1.0)   # sign is irrelevant
    a_dev[50] = 30.0                                                # s = a_dev(0) = 30
    res = fake_result(spec, 2, d, e_hat, a_dev, delta_max=0.0123)
    sd = stage_diag(res, spec, 2, {"valid": True, "max_std_norm": 0.0123})
    assert sd.valid and sd.stage == 2 and sd.tail_term
    assert sd.s == 30.0 and sd.C == 0.0123
    assert sd.delta_over_dw == pytest.approx(0.0123 / 4.0)
    assert sd.R == pytest.approx(0.05) and sd.argmax_d == 8.0        # tail nodes do not enter R
    assert sd.n_nodes_nontail == 49 and sd.n_nodes_tail == 52
    assert sd.R_tail == pytest.approx(0.4 / 30.0)                    # mean{e_hat : |d| >= 100} / s
    rho = sd.rho_bins
    assert rho.shape == (40,)
    # independent bin assignment with integer arithmetic: nodes are multiples of 4, edges of 10
    bins = (d.astype(int) + 200) // 10
    assert np.array_equal(np.minimum(bins, 39), gap_bin_index(spec, 2, d)[0])     # last bin closed
    tail_bins = np.r_[np.arange(0, 10), np.arange(30, 40)]
    assert np.isnan(rho[tail_bins]).all()
    nt = np.setdiff1d(np.arange(40), tail_bins)
    assert not np.isnan(rho[nt]).any()
    ratio = r / 30.0
    for b in nt:
        nodes = (bins == b) & (np.abs(d) < 100.0)
        assert rho[b] == pytest.approx(ratio[nodes].max()), b
    # bin [-100, -90): nodes -96, -92 (node -100 void)
    assert rho[10] == pytest.approx(0.03)
    assert rho[20] == pytest.approx(0.05) and rho[21] == pytest.approx(0.02)
    assert (rho[nt] == 0.0).sum() == nt.size - 3


def test_node_to_bin_aggregation_uses_gap_bin_index():
    """The per-bin map puts every node in the bin of ``gap_bin_index`` (edge nodes: right bin)."""
    spec = spec_for(60)
    d = stage_grid(2, spec.B, 4.0)                  # half = 220: nodes -220 .. 220
    rng = np.random.default_rng(5)
    e_hat = rng.uniform(5, 60, d.size)
    a_dev = e_hat + rng.normal(0.0, 1.0, d.size)
    a_dev[np.flatnonzero(d == 0.0)[0]] = 40.0
    sd = stage_diag(fake_result(spec, 2, d, e_hat, a_dev), spec, 2, CONC_OK)
    idx, nb = gap_bin_index(spec, 2, d)
    assert nb == 44 and sd.rho_bins.shape == (44,)
    nontail = np.abs(d) < 120.0
    tail_bins = StartSampler(spec, 10.0).tail_mask(2)
    ratio = np.abs(e_hat - a_dev) / 40.0
    for b in range(nb):
        sel = (idx == b) & nontail & ~tail_bins[b]
        if tail_bins[b] or not sel.any():
            assert np.isnan(sd.rho_bins[b])
        else:
            assert sd.rho_bins[b] == ratio[sel].max()
    assert sd.R == ratio[nontail].max() and sd.argmax_d == d[nontail][np.argmax(ratio[nontail])]


def test_t3_stage2_has_no_tail_term_and_a_full_bin_map():
    """T = 3, stage 2: thr = 4q = domain edge, no tail bin: tail term void, R_tail NaN, no NaN bin;
    the two edge nodes (|d| = 200 >= thr) do not enter R or rho."""
    spec = spec_for(50, T=3)
    d = stage_grid(2, spec.B, 4.0)
    e_hat = np.full(d.size, 30.0)
    a_dev = e_hat + 0.3
    a_dev[d == 0.0] = 30.0
    a_dev[d == -200.0] = 99.0                                       # huge residual at the edge node
    a_dev[d == 200.0] = 0.0
    sd = stage_diag(fake_result(spec, 2, d, e_hat, a_dev), spec, 2, CONC_OK)
    assert not sd.tail_term and np.isnan(sd.R_tail)
    assert sd.rho_bins.shape == (40,) and not np.isnan(sd.rho_bins).any()
    assert sd.R == pytest.approx(0.3 / 30.0) and sd.n_nodes_tail == 2
    assert sd.rho_bins[0] == pytest.approx(0.3 / 30.0)              # edge node -200 excluded
    assert sd.valid


def test_t3_stage3_bin_map_has_60_nan_tail_bins():
    spec = spec_for(50, T=3)
    d = stage_grid(3, spec.B, 4.0)                                  # half 400
    e_hat = np.where(np.abs(d) < 100.0, 20.0, 0.0)
    a_dev = e_hat + 0.2
    a_dev[d == 0.0] = 20.0
    sd = stage_diag(fake_result(spec, 3, d, e_hat, a_dev), spec, 3, CONC_OK)
    assert sd.tail_term and sd.rho_bins.shape == (80,)
    assert int(np.isnan(sd.rho_bins).sum()) == 60
    assert sd.R == pytest.approx(0.2 / 20.0) and sd.R_tail == 0.0


def test_stage_diag_on_a_real_t3_verifier_result():
    """Structural invariants on a T = 3 DEV-tier verifier result of an arbitrary smooth candidate:
    bin-map shapes and NaN patterns per stage, tail-term flags, ``nanmax(rho_bins) == R``."""
    spec = spec_for(50, T=3)

    def pol(t: int, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        if t == 1:
            return np.full(d.shape, 45.0)
        if t == 2:
            return 50.0 + 30.0 * np.cos(d / 53.0)
        return np.clip(35.0 + 20.0 * np.sin(d / 37.0), 0.0, 100.0)
    res = run_verify(spec, pol)
    d1, d2, d3 = (stage_diag(res, spec, t, CONC_OK) for t in (1, 2, 3))
    assert d1.valid and d2.valid and d3.valid
    assert d1.rho_bins.shape == (0,) and not d1.tail_term
    assert d2.rho_bins.shape == (40,) and not np.isnan(d2.rho_bins).any() and not d2.tail_term
    assert d3.rho_bins.shape == (80,) and int(np.isnan(d3.rho_bins).sum()) == 60 and d3.tail_term
    assert np.isnan(d3.rho_bins[:30]).all() and np.isnan(d3.rho_bins[50:]).all()
    assert not np.isnan(d3.rho_bins[30:50]).any()                  # the 20 bins of |d| < 100
    assert np.nanmax(d2.rho_bins) == d2.R and np.nanmax(d3.rho_bins) == d3.R
    for t, dg in ((2, d2), (3, d3)):
        st = res.stages[t]
        nt = np.abs(st.d_grid) < 2.0 * 50.0 * (3 - t + 1) - 1e-9
        j0 = int(np.argmin(np.abs(st.d_grid)))
        assert dg.s == st.a_dev[j0]
        assert dg.R == (np.abs(st.e_hat - st.a_dev)[nt] / st.a_dev[j0]).max()
        assert dg.delta_over_dw == res.full_delta_max[t] / res.dw
    assert d3.R_tail == pytest.approx(res.stages[3].e_hat[np.abs(res.stages[3].d_grid) >= 100.0]
                                      .mean() / d3.s, rel=1e-12)
    assert d1.R == abs(res.stages[1].e_hat[0] - res.stages[1].a_dev[0]) / res.stages[1].a_dev[0]


# ----------------------------------------------------------------------------------------------
# (d) EMA
# ----------------------------------------------------------------------------------------------

def test_ema_update_init_weighting_and_nan_handling():
    new1 = np.array([0.4, np.nan, 0.2, 0.0])
    ema = ema_update(None, new1, 0.5)
    assert np.array_equal(ema, new1, equal_nan=True) and ema is not new1      # initialised, a copy
    new1[0] = 99.0
    assert ema[0] == 0.4
    new2 = np.array([0.2, 0.6, np.nan, 0.4])
    ema2 = ema_update(ema, new2, 0.5)
    assert ema2[0] == pytest.approx(0.5 * 0.4 + 0.5 * 0.2)           # beta weighting
    assert ema2[1] == 0.6                                            # NaN in the past -> take new
    # NaN in the new map -> keep past
    assert ema2[2] == 0.2
    assert ema2[3] == pytest.approx(0.2)
    ema3 = ema_update(np.array([np.nan, 1.0]), np.array([np.nan, 3.0]), 0.5)
    assert np.isnan(ema3[0]) and ema3[1] == 2.0                      # both NaN stays NaN
    for beta in (0.0, 0.25, 0.9):
        out = ema_update(np.array([1.0, 2.0]), np.array([3.0, 4.0]), beta)
        assert np.allclose(out, beta * np.array([1.0, 2.0]) + (1 - beta) * np.array([3.0, 4.0]))
    # the input of an update is not modified
    prev = np.array([1.0, 2.0])
    ema_update(prev, np.array([3.0, 4.0]), 0.5)
    assert np.array_equal(prev, [1.0, 2.0])
    # sequence: EMA of a constant map stays that map exactly
    e = None
    for _ in range(5):
        e = ema_update(e, np.array([0.1, 0.2]), 0.5)
    assert np.array_equal(e, [0.1, 0.2])


# ----------------------------------------------------------------------------------------------
# (e) stage 1
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("q", [50, 60])
def test_stage1_diag_is_the_relative_distance_to_the_one_step_best_response(q):
    spec = spec_for(q)
    res = run_verify(spec, policy_from(lambda d: closed_form(spec, d)))
    s1 = res.stages[1]
    assert s1.d_grid.shape == (1,) and s1.e_hat[0] == E1_FIXED
    sd = stage_diag(res, spec, 1, CONC_OK)
    want = abs(s1.e_hat[0] - s1.a_dev[0]) / s1.a_dev[0]
    assert sd.valid and sd.stage == 1
    assert sd.R == want and sd.s == s1.a_dev[0] and sd.R > 0.0
    assert sd.delta_over_dw == res.full_delta_max[1] / res.dw
    assert not sd.tail_term and np.isnan(sd.R_tail)
    assert sd.rho_bins.shape == (0,)
    assert sd.n_nodes_nontail == 1 and sd.n_nodes_tail == 0 and sd.argmax_d == 0.0


def test_stage1_hand_made():
    spec = spec_for(50)
    res = fake_result(spec, 1, np.zeros(1), np.array([40.0]), np.array([46.0]), delta_max=0.002)
    sd = stage_diag(res, spec, 1, CONC_OK)
    assert sd.R == pytest.approx(6.0 / 46.0) and sd.s == 46.0
    assert sd.delta_over_dw == pytest.approx(0.0005) and not sd.tail_term
    assert sd.rho_bins.size == 0 and np.isnan(sd.R_tail)
    # T = 3: stage 1 is still the single root node
    spec3 = spec_for(50, T=3)
    sd3 = stage_diag(fake_result(spec3, 1, np.zeros(1), np.array([40.0]), np.array([40.0])),
                     spec3, 1, CONC_OK)
    assert sd3.R == 0.0 and sd3.valid and not sd3.tail_term


# ----------------------------------------------------------------------------------------------
# (f) invalid inputs
# ----------------------------------------------------------------------------------------------

def healthy(spec: GameSpec) -> SimpleNamespace:
    d = stage_grid(2, spec.B, 4.0)
    e_hat = np.where(np.abs(d) < 100.0, 30.0, 0.0)
    return fake_result(spec, 2, d, e_hat, e_hat + 0.1)


@pytest.mark.parametrize("conc", [None, {"valid": False, "max_std_norm": 0.01},
                                  {"valid": True, "max_std_norm": float("nan")},
                                  {"valid": True, "max_std_norm": float("inf")},
                                  {"valid": False, "n_points": 0}])
def test_non_finite_or_missing_concentration_is_invalid(conc):
    spec = spec_for(50)
    res = healthy(spec)
    res.stages[2].a_dev[res.stages[2].d_grid == 0.0] = 30.0
    sd = stage_diag(res, spec, 2, conc)
    assert not sd.valid
    assert sd.R == pytest.approx(0.1 / 30.0)       # the residual itself is still computed


@pytest.mark.parametrize("s", [0.0, -3.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_scale_is_invalid(s):
    spec = spec_for(50)
    res = healthy(spec)
    res.stages[2].a_dev[res.stages[2].d_grid == 0.0] = s
    sd = stage_diag(res, spec, 2, CONC_OK)
    if np.isfinite(s) and s > 0:
        assert sd.valid
        return
    assert not sd.valid
    assert np.isnan(sd.R) and np.isnan(sd.R_tail)
    assert sd.rho_bins.shape == (40,) and np.isnan(sd.rho_bins).all()
    assert sd.delta_over_dw == res.full_delta_max[2] / res.dw       # still reported


def test_invalid_verifier_result_is_invalid_and_missing_zero_node_raises():
    spec = spec_for(50)
    res = healthy(spec)
    res.stages[2].a_dev[res.stages[2].d_grid == 0.0] = 30.0
    res.valid = False
    assert not stage_diag(res, spec, 2, CONC_OK).valid
    d = np.arange(-199.0, 201.0, 4.0)                  # no node at 0
    bad = fake_result(spec, 2, d, np.full(d.size, 30.0), np.full(d.size, 30.0))
    with pytest.raises(ValueError, match="zero node"):
        stage_diag(bad, spec, 2, CONC_OK)


def test_invalid_diag_placeholder_is_never_eligible():
    from utils.ms_rule import is_eligible
    sd = invalid_diag(2, 40)
    assert not sd.valid and sd.stage == 2
    assert sd.rho_bins.shape == (40,) and np.isnan(sd.rho_bins).all()
    assert invalid_diag(1).rho_bins.shape == (0,)
    rule = StageRule(stage=2, n_nontail_bins=20)
    assert not is_eligible(rule, sd)
    sc = StageDiag(stage=2, valid=True, delta_over_dw=0.0, s=1.0, R=0.0, R_tail=0.0, C=0.0,
                   tail_term=True).scalars()
    assert set(sc) == {"valid", "Delta", "s", "R", "R_tail", "C", "tail_term", "R_argmax_d"}


# ----------------------------------------------------------------------------------------------
# (g) the closed form does not enter: stubbed-closed-form invariance
# ----------------------------------------------------------------------------------------------

def _garbage_recovery(policy, spec, step):
    return ({"g1": -1.0, "g2_at_0": -2.0, "e1_at_0": -3.0, "stage1_rel_err_signed": 123456.0,
             "stage2_peak_rel_err_signed": 654321.0, "stage2_peak_rel_err_abs": 654321.0},
            {"recovery_d_grid": np.zeros(3), "recovery_e2": np.zeros(3),
             "recovery_g2": np.zeros(3)})


def _install_stubs(monkeypatch) -> None:
    """Replace the closed form in the defining module AND in the importing module of evaluate()."""
    monkeypatch.setattr(v2m, "recovery_metrics", _garbage_recovery)
    for mod in (theory, v2m):
        monkeypatch.setattr(mod, "g1_two_stage", lambda *a, **k: -999.0)
        monkeypatch.setattr(mod, "g2_two_stage",
                            lambda d, *a, **k: np.full(np.shape(d), 12345.0))


def _bump_policy(spec: GameSpec, c: float) -> Callable:
    """Closed form + c on |d - 30| <= 15 (a localized perturbation)."""
    return policy_from(lambda d: closed_form(spec, d) + c * (np.abs(d - 30.0) <= 15.0))


def _evaluate_diag(spec: GameSpec, pol: Callable, stage: int = 2) -> Tuple[StageDiag, v2m.V2Eval]:
    def beta_fn(t: int, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = np.asarray(pol(t, d), dtype=float) / spec.e_range
        return m * 400.0, (1.0 - m) * 400.0
    ev = v2m.evaluate(pol, spec, DEV_CONFIG, beta_fn=beta_fn)
    conc = concentration_stats(beta_fn, {stage: ev.res.stages[stage].d_grid})
    return stage_diag(ev.res, spec, stage, conc), ev


@pytest.mark.parametrize("c", [0.0, 1.0, 3.0])
def test_stage_diag_is_bit_identical_when_the_closed_form_is_stubbed(monkeypatch, c):
    spec = spec_for(50)
    pol = _bump_policy(spec, c)
    ref, ev_ref = _evaluate_diag(spec, pol)
    _install_stubs(monkeypatch)
    got, ev_got = _evaluate_diag(spec, pol)                          # evaluate() still runs
    assert ev_got.scalars["stage1_rel_err_signed"] == 123456.0        # the stub is really active
    assert ev_got.scalars["stage2_peak_rel_err_signed"] == 654321.0
    assert ev_ref.scalars["stage2_peak_rel_err_signed"] != 654321.0
    assert same_bits(ref, got)
    assert np.array_equal(ev_ref.res.stages[2].a_dev, ev_got.res.stages[2].a_dev)
    # and the diag of the second tier of the verifier path (pure function of the policy)
    assert ref.valid and got.valid


def _drive(spec: GameSpec, schedule, rule: StageRule, n_updates: int = 80):
    """Run a StageController over a scripted candidate sequence (shift c per check index)."""
    ctrl = StageController(rule, lambda s, e, a, b, j: s + (e - s) * (j - a) / (b - a))
    trace, probs, checks = [], [], 0
    sp = StartSampler(spec, 10.0)
    for j in range(1, n_updates + 1):
        if ctrl.finished:
            break
        st = ctrl.sampler_setting()
        probs.append(sp.stratified_bin_probs(2, 0.25, 20.0, alpha=st.alpha, focus=st.focus))
        trace.append((j, ctrl.lr(j), ctrl.block_label(), st.block_type, st.alpha))

        def diag_fn() -> StageDiag:
            nonlocal checks
            c = schedule[min(checks, len(schedule) - 1)]
            checks += 1
            return _evaluate_diag(spec, _bump_policy(spec, c))[0]

        ctrl.after_update(j, diag_fn)
    return ctrl, trace, probs


@pytest.mark.parametrize("name,schedule,rule_kw", [
    ("stop_in_first_block", [3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     dict(K=5, M=3, n_block=40, u_cap=80, n_land=6)),
    ("localized_then_polish", [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     dict(K=5, M=3, n_block=10, u_cap=60, n_land=6)),
    ("broad_then_cap", [3.0] * 20, dict(K=5, M=3, n_block=10, u_cap=30, n_land=4)),
])
def test_rule_and_sampler_decisions_are_identical_when_the_closed_form_is_stubbed(
        monkeypatch, name, schedule, rule_kw):
    spec = spec_for(50)
    rule = StageRule(stage=2, n_nontail_bins=20, **rule_kw)
    ctrl_a, trace_a, probs_a = _drive(spec, schedule, rule)
    _install_stubs(monkeypatch)
    ctrl_b, trace_b, probs_b = _drive(spec, schedule, rule)
    rec_a = json.dumps(ctrl_a.record(), sort_keys=True)
    assert rec_a == json.dumps(ctrl_b.record(), sort_keys=True)
    assert trace_a == trace_b
    assert len(probs_a) == len(probs_b)
    assert all(np.array_equal(x, y) for x, y in zip(probs_a, probs_b))
    rec = ctrl_a.record()
    if name == "stop_in_first_block":
        assert rec["fire_local"] == 25 and not rec["budget_forced"]
    if name == "localized_then_polish":
        assert rec["blocks"][0]["classification"] == "localized"
        assert rec["blocks"][0]["n_S"] == 4 and rec["blocks"][1]["type"] == "polish"
        assert rec["blocks"][0]["S"] == [15, 16, 17, 18]
    if name == "broad_then_cap":
        assert rec["budget_forced"] and rec["blocks"][0]["classification"] == "broad"
        assert rec["blocks"][0]["n_S"] == 8


# ----------------------------------------------------------------------------------------------
# (g') the new modules do not import the closed form
# ----------------------------------------------------------------------------------------------

CLOSED_FORM_NAMES = {"theory_multistage", "theory", "v2_metrics", "g1_two_stage", "g2_two_stage",
                     "v2_star", "recovery_metrics", "eq_utility_two_stage", "stage1_curvature"}


@pytest.mark.parametrize("path", ["utils/ms_residual.py", "utils/ms_rule.py",
                                  "utils/ms_continuation.py",
                                  "run/ms_rollout.py", "envs/curriculum_env.py"])
def test_new_modules_import_no_closed_form_equilibrium(path):
    tree = ast.parse((ROOT / path).read_text())
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            seen |= {a.name.split(".")[-1] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            seen |= {(node.module or "").split(".")[-1]} | {a.name for a in node.names}
    assert not (seen & CLOSED_FORM_NAMES), seen & CLOSED_FORM_NAMES
    assert hasattr(ms_residual, "stage_diag") and not hasattr(ms_residual, "g2_two_stage")
