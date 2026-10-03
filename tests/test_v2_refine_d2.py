"""Tests for tools/v2/verifier_sensitivity.py (R1 diagnostic D2, verifier sensitivity).

Covers: the delta = 0 candidate is the exact analytic policy and sits at the verifier's numerical
floor; the uniform-kernel table is exactly even, maps a constant to itself, agrees with an
independent numerical convolution and induces the peak error -h / (4 q); family (d) only touches
|d| >= 2q; the output clip; the detection-limit and fit helpers on synthetic data (not reached,
asymmetric, non-monotone, floor); three closed-form cross-checks of the whole chain (candidate ->
verifier -> metric -> fit); the process pool; and the smoke run with its report on dev-only and
final-only tiers.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest \
     tests/test_v2_refine_d2.py -p no:cacheprovider -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import verifier_sensitivity as vs  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402
from utils.theory_multistage import g1_two_stage, g2_two_stage, stage1_curvature  # noqa: E402

PROTO = json.loads((ROOT / "protocols" / "v2_T2_locked_v1_1.json").read_text())
QS = (50, 60)
GIT = {"commit": "test", "short": "test", "dirty": None}


def spec_for(q: int) -> GameSpec:
    """Game of the locked protocol record for q."""
    return GameSpec(**PROTO["records"][str(q)]["game"])


def g2_of(spec: GameSpec, d: np.ndarray) -> np.ndarray:
    """Closed-form stage-2 effort for a spec."""
    return g2_two_stage(d, spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max)


def eval_row(cand: "vs.Candidate", q: int, tier: str) -> Dict[str, object]:
    """One verifier evaluation exactly as the tool does it."""
    rec = PROTO["records"][str(q)]
    row = vs.evaluate_task({"cand": cand, "game": rec["game"], "tier": tier,
                            "recovery_step": rec["protocol"]["recovery_step"],
                            "commit": "test", "dirty": None})
    assert row["status"] == "ok", row["status"]
    return row


# ---------------------------------------------------------------------------------------------
# delta = 0: exact policy and numerical floor
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("cand", [vs.Candidate("a"), vs.Candidate("b")], ids=["a0", "b0"])
def test_delta0_is_bitwise_the_analytic_policy(q: int, cand: "vs.Candidate") -> None:
    """(1 + 0) scaling returns the closed form bit for bit on the whole stage-2 domain."""
    s = spec_for(q)
    pol = vs.PerturbedPolicy(s, cand)
    n = int(round(s.domain_half(2) / 0.01))
    d = np.arange(-n, n + 1) * 0.01
    assert np.array_equal(pol(2, d), g2_of(s, d))
    assert pol(1, np.zeros(1))[0] == g1_two_stage(s.q, s.w_h, s.w_l, s.k)
    assert not pol.clip_report()["clip_binds"]


@pytest.mark.parametrize("q", QS)
def test_delta0_sits_at_the_verifier_floor(q: int) -> None:
    """Exact candidate: zero recovery error; deviation gain at floor on both tiers.

    The final tier at q = 60 has a measured floor of 3.5e-07 (Gmax_full/DW at t = 1, d = 0,
    stage-1 interpolation); every gate threshold is >= 1e-3, so 1e-6 is a meaningful bound.
    """
    for tier, gbound in (("development", 1e-9), ("final", 1e-6 if q == 60 else 1e-9)):
        r = eval_row(vs.Candidate("a"), q, tier)
        assert bool(r["valid"])
        assert r["Gmax_full_over_dw"] < gbound
        assert r["eta_T_over_dw"] < 1e-9
        assert r["stage1_rel_err_signed"] == 0.0
        assert r["stage2_peak_rel_err_signed"] == 0.0
        assert r["stage2_rmse_pos_over_g2_0"] == 0.0
        assert r["stage2_tail_mean_over_g2_0"] == 0.0


# ---------------------------------------------------------------------------------------------
# Kernel (family c)
# ---------------------------------------------------------------------------------------------
def test_box_average_constant_and_linear() -> None:
    """A constant maps to itself; a linear function maps to itself (symmetric window)."""
    step, n_h = 0.01, 500
    x = np.arange(-2000, 2001) * step
    out_c = vs.box_average(np.full(x.shape, 3.7), n_h, step)
    assert out_c.shape == (x.size - 2 * n_h,)
    assert np.max(np.abs(out_c - 3.7)) < 1e-12
    out_l = vs.box_average(2.0 * x + 1.0, n_h, step)
    assert np.max(np.abs(out_l - (2.0 * x[n_h:-n_h] + 1.0))) < 1e-10
    assert np.array_equal(vs.box_average(x, 0, step), x)
    with pytest.raises(ValueError):
        vs.box_average(x[:10], 20, step)


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("h", vs.KERNEL_H)
def test_kernel_table_even_covers_domain_and_unit_mass(q: int, h: float) -> None:
    """Table is exactly even, spans [-B, B], is non-negative and vanishes beyond 2q + h."""
    s = spec_for(q)
    x, v = vs.kernel_table(s, h)
    assert np.array_equal(v, v[::-1])
    assert np.array_equal(x, -x[::-1])
    assert x[0] == -s.domain_half(2) and x[-1] == s.domain_half(2)
    assert np.all(v >= 0.0)
    far = np.abs(x) >= 2 * q + h + 1e-9
    assert np.all(v[far] == 0.0)
    mid = (np.abs(x) > 2 * q) & (np.abs(x) < 2 * q + h - 1e-9)
    assert np.all(v[mid] > 0.0)           # the kernel leaks effort into the zero-effort tail
    g_mass = float(np.sum(0.5 * (g2_of(s, x)[1:] + g2_of(s, x)[:-1])) * 0.01)
    k_mass = float(np.sum(0.5 * (v[1:] + v[:-1])) * 0.01)
    assert abs(k_mass - g_mass) / g_mass < 1e-9   # convolution with a unit-mass kernel


def test_kernel_rejects_off_grid_half_width() -> None:
    """h must be a positive multiple of the 0.01 grid."""
    with pytest.raises(ValueError):
        vs.kernel_table(spec_for(50), 12.005)
    with pytest.raises(ValueError):
        vs.kernel_table(spec_for(50), 0.0)


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("h", vs.KERNEL_H)
def test_kernel_induced_peak_error_is_minus_h_over_4q(q: int, h: float) -> None:
    """For h <= 2q the smoothed peak is g2*(0) (1 - h / (4 q)): induced signed error -h / (4 q)."""
    s = spec_for(q)
    assert h <= 2 * q
    pol = vs.PerturbedPolicy(s, vs.Candidate("c", h=h))
    g20 = float(g2_of(s, np.zeros(1))[0])
    peak_err = (float(pol(2, np.zeros(1))[0]) - g20) / g20
    assert abs(peak_err - (-h / (4.0 * q))) < 1e-9


def _window_mean(s: GameSpec, x: float, h: float, n: int = 400001) -> float:
    """Independent check: trapezoid mean of g2* over [x - h, x + h] on a fine grid."""
    t = np.linspace(x - h, x + h, n)
    y = g2_of(s, t)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)) / (2.0 * h))


@pytest.mark.parametrize("x", [0.0, 5.0, 30.0, 94.0, 100.0, 105.0, 111.0, 112.0, 150.0, 33.333])
def test_kernel_matches_independent_convolution(x: float) -> None:
    """Table + linear interpolation agrees with a fine direct integral (q = 50, h = 12)."""
    s = spec_for(50)
    pol = vs.PerturbedPolicy(s, vs.Candidate("c", h=12.0))
    got = float(pol(2, np.array([x]))[0])
    assert abs(got - _window_mean(s, x, 12.0)) < 1e-6


@pytest.mark.parametrize("q", QS)
def test_recovery_metrics_use_the_locked_step_and_match_a_direct_computation(q: int) -> None:
    """RMSE_pos, tail mean and signed errors of the (e) candidate on the 0.5 recovery grid.

    Recomputed here from the policy and the closed form without ``utils.v2_metrics``; the step
    is the protocol's ``recovery_step`` (0.5). Both quantities differ on a 2.0 grid.
    """
    s = spec_for(q)
    assert PROTO["records"][str(q)]["protocol"]["recovery_step"] == 0.5
    cand = vs.Candidate("e", delta1=0.1, h=12.0)
    pol = vs.PerturbedPolicy(s, cand)
    r = eval_row(cand, q, "development")

    def direct(step: float) -> Dict[str, float]:
        n = int(round(s.domain_half(2) / step))
        d = np.arange(-n, n + 1) * step
        e2, g2 = pol(2, d), g2_of(s, d)
        pos = np.abs(d) < 2 * q
        g20 = float(g2_of(s, np.zeros(1))[0])
        return {"rmse": float(np.sqrt(np.mean((e2[pos] - g2[pos]) ** 2))) / g20,
                "tail": float(np.mean(e2[~pos])) / g20}

    d05, d20 = direct(0.5), direct(2.0)
    assert r["stage2_rmse_pos_over_g2_0"] == pytest.approx(d05["rmse"], rel=1e-12)
    assert r["stage2_tail_mean_over_g2_0"] == pytest.approx(d05["tail"], rel=1e-12)
    assert abs(d05["rmse"] - d20["rmse"]) > 1e-6 * d05["rmse"]          # the step matters
    g1 = g1_two_stage(s.q, s.w_h, s.w_l, s.k)
    assert r["stage1_rel_err_signed"] == pytest.approx(0.1, rel=1e-12)
    assert r["e1_at_0"] == pytest.approx(1.1 * g1, rel=1e-12)
    assert r["stage2_peak_rel_err_signed"] == pytest.approx(-12.0 / (4.0 * q), abs=1e-9)


def test_kernel_policy_query_outside_domain_is_an_error() -> None:
    """The interpolation domain check refuses a query beyond [-B, B]."""
    s = spec_for(50)
    pol = vs.PerturbedPolicy(s, vs.Candidate("c", h=12.0))
    with pytest.raises(ValueError):
        pol(2, np.array([s.domain_half(2) + 1.0]))


# ---------------------------------------------------------------------------------------------
# Families (a), (b), (d), (e): construction and clip
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", QS)
def test_family_d_touches_only_the_tail(q: int) -> None:
    """e2 + tau exactly where |d| >= 2q and unchanged inside; stage 1 untouched."""
    s = spec_for(q)
    tau = 0.5
    pol = vs.PerturbedPolicy(s, vs.Candidate("d", tau=tau))
    d = np.arange(-int(s.domain_half(2) * 2), int(s.domain_half(2) * 2) + 1) * 0.5
    d = d[np.abs(d) <= s.domain_half(2)]
    inner = np.abs(d) < 2 * q
    assert np.array_equal(pol(2, d)[inner], g2_of(s, d)[inner])
    assert np.all(pol(2, d)[~inner] == tau)
    assert pol(1, np.zeros(1))[0] == g1_two_stage(s.q, s.w_h, s.w_l, s.k)


def test_families_a_and_b_scale_the_right_stage() -> None:
    """(a) scales stage 1 only; (b) scales stage 2 only; (e) = (a) with the kernel stage 2."""
    s = spec_for(50)
    g1 = g1_two_stage(s.q, s.w_h, s.w_l, s.k)
    d = np.linspace(-60.0, 60.0, 25)
    pa, pb = vs.PerturbedPolicy(s, vs.Candidate("a", delta1=0.1)), vs.PerturbedPolicy(
        s, vs.Candidate("b", delta2=-0.05))
    assert pa(1, np.zeros(1))[0] == pytest.approx(1.1 * g1, rel=1e-12)
    assert np.array_equal(pa(2, d), g2_of(s, d))
    assert pb(1, np.zeros(1))[0] == g1
    assert np.allclose(pb(2, d), 0.95 * g2_of(s, d), rtol=1e-12, atol=0.0)
    pe = vs.PerturbedPolicy(s, vs.Candidate("e", delta1=0.05, h=12.0))
    pc = vs.PerturbedPolicy(s, vs.Candidate("c", h=12.0))
    assert pe(1, np.zeros(1))[0] == pytest.approx(1.05 * g1, rel=1e-12)
    assert np.array_equal(pe(2, d), pc(2, d))


def test_policy_contract_shape_dtype_range_and_stage() -> None:
    """Output matches the input shape, is float64, finite, inside [e_min, e_max]; stage 3 fails."""
    s = spec_for(60)
    pol = vs.PerturbedPolicy(s, vs.Candidate("e", delta1=0.15, h=12.0))
    for t, d in ((1, np.zeros(1)), (2, np.linspace(-220.0, 220.0, 441))):
        out = pol(t, d)
        assert out.shape == d.shape and out.dtype == np.float64 and np.all(np.isfinite(out))
        assert out.min() >= s.e_min and out.max() <= s.e_max
    with pytest.raises(ValueError):
        pol(3, np.zeros(1))
    with pytest.raises(ValueError):
        vs.PerturbedPolicy(GameSpec(6, 2, 1 / 3500, 50, 3), vs.Candidate("a"))


def test_clip_never_binds_on_the_preregistered_grids() -> None:
    """Largest effort is 1.15 * e2*(0) < e_max for every candidate of every family and q."""
    for q in QS:
        s = spec_for(q)
        for fam in vs.FAMILIES:
            for c in vs.family_candidates(fam, kernel_h=12.0):
                rep = vs.PerturbedPolicy(s, c).clip_report()
                assert rep["clip_binds"] is False, (q, c.cand_id, rep)
                assert rep["clip_raw_min"] >= s.e_min
                assert rep["clip_raw_max"] <= 1.15 * float(g2_of(s, np.zeros(1))[0]) + 1e-9


def test_clip_binds_and_is_applied_for_a_synthetic_large_delta() -> None:
    """A +100 % amplitude error exceeds e_max: reported and clipped; verifier stays valid."""
    s = spec_for(50)
    big = vs.PerturbedPolicy(s, vs.Candidate("b", delta2=1.0))
    rep = big.clip_report()
    assert rep["clip_binds"] is True and rep["clip_raw_max"] == pytest.approx(140.0)
    assert float(big(2, np.zeros(1))[0]) == s.e_max
    neg = vs.PerturbedPolicy(s, vs.Candidate("a", delta1=-1.5))
    assert neg.clip_report()["clip_binds"] is True and neg.clip_report()["clip_raw_min"] < 0.0
    assert float(neg(1, np.zeros(1))[0]) == s.e_min
    r = eval_row(vs.Candidate("b", delta2=1.0), 50, "development")
    assert r["clip_binds"] is True and bool(r["valid"])


def test_grids_match_the_spec() -> None:
    """Grids of PROMPT.md section 3.2 and the candidate ids."""
    assert vs.DELTA_GRID == (-0.15, -0.1, -0.05, -0.02, -0.01, -0.005, 0.0,
                             0.005, 0.01, 0.02, 0.05, 0.1, 0.15)
    assert vs.KERNEL_H == (2.0, 5.0, 10.0, 12.0, 20.0)
    assert vs.TAU_GRID == (0.25, 0.5, 1.0, 2.0) and vs.E_DELTAS == (0.05, 0.10, 0.15)
    assert vs.KERNEL_TARGET == -0.06
    assert [len(vs.family_candidates(f, 12.0)) for f in vs.FAMILIES] == [13, 13, 5, 4, 3]
    assert vs.family_candidates("a")[6].cand_id == "a_d+0"
    assert vs.family_candidates("c")[3].cand_id == "c_h12"
    assert vs.family_candidates("e", 12.0)[0].cand_id == "e_d+0.05_h12"
    with pytest.raises(ValueError):
        vs.family_candidates("e")


# ---------------------------------------------------------------------------------------------
# Closed-form cross-checks of the whole chain (candidate -> verifier -> metric)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("tier", ["development", "final"])
def test_family_d_gain_is_k_tau2_over_dw(q: int, tier: str) -> None:
    """A tail offset tau wastes cost k tau^2 and nothing else: Gmax = eta_2 = k tau^2 / DW."""
    s = spec_for(q)
    for tau in vs.TAU_GRID:
        r = eval_row(vs.Candidate("d", tau=tau), q, tier)
        want = s.k * tau * tau / s.dw
        assert r["Gmax_full_over_dw"] == pytest.approx(want, rel=1e-9)
        assert r["eta_T_over_dw"] == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("tier", ["development", "final"])
def test_family_b_positive_side_is_exactly_quadratic(q: int, tier: str) -> None:
    """For delta2 > 0 the best deviation at d = 0 gains k g^2 delta^2 / (DW (1 - g / (2q))).

    The optimal deviation is y* = delta g / (1 - g / (2q)) effort units below the candidate. The
    verifier finds it exactly (1e-9 relative; measured <= 3e-12) on the final tier and, on the
    development tier, whenever y* >= 2 effort steps. When y* is below that the effort grid has
    too few nodes around the optimum (the kink of F at y = 0 can fall inside the vertex stencil:
    measured shortfall 1.0 % at q = 60, delta = 0.005 and 2.6e-06 at delta = 0.01), and the
    grid best response can only under-find the gain: the value is bounded from above by the
    closed form.
    """
    s = spec_for(q)
    g = float(g2_of(s, np.zeros(1))[0])
    a_theory = s.k * g * g / (s.dw * (1.0 - g / (2.0 * q)))
    step = vs.TIERS[tier].effort_step
    for delta in (0.005, 0.01, 0.02, 0.05, 0.15):
        r = eval_row(vs.Candidate("b", delta2=delta), q, tier)
        want = a_theory * delta ** 2
        y_star = delta * g / (1.0 - g / (2.0 * q))
        for key in ("Gmax_full_over_dw", "eta_T_over_dw"):
            if tier == "final" or y_star >= 2.0 * step:
                assert r[key] == pytest.approx(want, rel=1e-9), (delta, key)
            else:
                assert r[key] <= want * (1.0 + 1e-9), (delta, key)
        if tier == "final" or y_star >= 2.0 * step:
            assert r["Gmax_full_t"] == 2 and r["Gmax_full_d"] == 0.0   # peak of the hump


@pytest.mark.parametrize("q", QS)
def test_family_a_small_delta_matches_the_stage1_curvature_law(q: int) -> None:
    """Stage-1 gain ~ U'(e_c)^2 / (2 |U''|) with U'(e_c) = -2 k g1 delta (opponent shifted too).

    So Gmax/DW ~ a delta^2, a = 2 (k g1)^2 / (|U''| DW), U'' = stage1_curvature. The verifier
    grids (effort step 0.5 / 1) and the cubic term bound the agreement: 8 % at |delta| = 0.05.
    """
    s = spec_for(q)
    g1 = g1_two_stage(s.q, s.w_h, s.w_l, s.k)
    a_theory = 2.0 * (s.k * g1) ** 2 / (abs(stage1_curvature(s.q, s.w_h, s.w_l, s.k)) * s.dw)
    for delta in (0.05, -0.05):
        r = eval_row(vs.Candidate("a", delta1=delta), q, "final")
        assert abs(r["Gmax_full_over_dw"] / (delta ** 2) / a_theory - 1.0) < 0.08
        assert r["eta_T_over_dw"] < 1e-9              # stage 2 exact


# ---------------------------------------------------------------------------------------------
# Helpers on synthetic data
# ---------------------------------------------------------------------------------------------
def test_fit_through_origin_exact_and_normal_equations() -> None:
    """Exact quadratic recovered; arbitrary data match numpy least squares; x = 0 is dropped."""
    x = np.array([-0.1, -0.05, 0.0, 0.02, 0.1])
    f = vs.fit_through_origin(x, 3.0 * x ** 2)
    assert f["a"] == pytest.approx(3.0, rel=1e-12) and f["n"] == 4
    assert f["max_abs_resid"] < 1e-15 and f["r2_uncentered"] == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    xs = rng.uniform(0.01, 0.2, 9)
    ys = 2.0 * xs ** 2 + rng.normal(0.0, 1e-4, 9)
    f = vs.fit_through_origin(xs, ys)
    a_ls = np.linalg.lstsq((xs ** 2)[:, None], ys, rcond=None)[0][0]
    assert f["a"] == pytest.approx(a_ls, rel=1e-12)
    res = np.array([float(v) for v in f["residuals"].split(";")])
    assert np.allclose(res, ys - f["a"] * xs ** 2, atol=1e-6)
    assert f["x_at_max_abs_resid"] == xs[int(np.argmax(np.abs(ys - f["a"] * xs ** 2)))]
    with pytest.raises(ValueError):
        vs.fit_through_origin([0.0, 0.0], [1.0, 2.0])


def test_detection_limit_ascending_strict_and_not_reached() -> None:
    """First exceedance in ascending |p|; strict '>'; explicit 'not reached' record."""
    p = [-0.1, -0.05, 0.0, 0.05, 0.1]
    v = [0.02, 0.005, 0.0, 0.003, 0.012]
    anyd = vs.detection_limit(p, v, 0.01, "any")
    assert anyd["reached"] and anyd["limit"] == 0.1 and anyd["p_at_limit"] == -0.1
    assert anyd["bracket_lo"] == "0.05" and anyd["value_at_limit"] == 0.02
    assert vs.detection_limit(p, v, 0.01, "neg")["p_at_limit"] == -0.1
    assert vs.detection_limit(p, v, 0.01, "pos")["p_at_limit"] == 0.1
    nr = vs.detection_limit(p, v, 0.05, "any")
    assert nr["reached"] is False and nr["limit"] is None
    assert nr["limit_str"] == vs.NOT_REACHED and nr["max_value_on_grid"] == 0.02
    assert nr["max_abs_pert_on_grid"] == 0.1
    # value exactly at the threshold is not an exceedance
    assert vs.detection_limit([0.0, 0.1], [0.0, 0.01], 0.01, "any")["reached"] is False


def test_detection_limit_asymmetric_and_nonmonotone() -> None:
    """Only one side exceeds: the other side says 'not reached'; non-monotone takes the first."""
    p = [-0.1, -0.05, 0.0, 0.05, 0.1]
    v = [0.001, 0.0005, 0.0, 0.02, 0.03]
    assert vs.detection_limit(p, v, 0.01, "neg")["reached"] is False
    pos = vs.detection_limit(p, v, 0.01, "pos")
    assert pos["limit"] == 0.05 and pos["bracket_lo"] == "0"
    assert vs.detection_limit(p, v, 0.01, "any")["p_at_limit"] == 0.05
    non = vs.detection_limit([0.0, 0.01, 0.02, 0.05], [0.0, 0.02, 0.001, 0.03], 0.01, "any")
    assert non["limit"] == 0.01 and non["bracket_lo"] == "0"


def _synthetic_paired(fam: str, a_neg: float, a_pos: float, eta: float = 0.0) -> pd.DataFrame:
    """Paired-table stand-in for a signed family with y = a_side x^2 on both tiers."""
    rows: List[Dict[str, object]] = []
    for dlt in vs.DELTA_GRID:
        y = (a_neg if dlt < 0 else a_pos) * dlt * dlt
        rows.append({"family": fam, "cand_id": f"{fam}_{dlt}", "q": 50, "perturbation": dlt,
                     "perturbation_name": "delta", "x": dlt, "x_name": "delta",
                     "Gmax_full_over_dw_dev": y, "Gmax_full_over_dw_final": y * 1.1,
                     "eta_T_over_dw_dev": eta, "eta_T_over_dw_final": eta,
                     "absdiff_Gmax": abs(y * 0.1), "absdiff_eta": 0.0})
    return pd.DataFrame(rows)


def test_build_fits_sides_floor_and_eta_flag() -> None:
    """Side fits recover each side's coefficient; floor and eta-independent fits are flagged."""
    th = vs.thresholds(PROTO)
    paired = pd.concat([_synthetic_paired("b", 2.0, 4.0, eta=1e-16),
                        _synthetic_paired("a", 2.0, 4.0, eta=1e-16)])
    fits = vs.build_fits(paired, th)
    fb = fits[(fits["family"] == "b") & (fits["tier"] == "development")
              & (fits["metric"] == "Gmax_full_over_dw")].set_index("side")
    assert fb.loc["neg", "a"] == pytest.approx(2.0) and fb.loc["pos", "a"] == pytest.approx(4.0)
    assert 2.0 < fb.loc["pooled", "a"] < 4.0
    assert fb.loc["pos", "x_at_threshold_fit"] == pytest.approx(np.sqrt(0.01 / 4.0))
    fe = fits[(fits["family"] == "b") & (fits["metric"] == "eta_T_over_dw")]
    assert fe["x_at_threshold_fit"].isna().all()
    assert fe["fit_note"].str.contains("numerical floor").all()
    fa = fits[(fits["family"] == "a") & (fits["metric"] == "eta_T_over_dw")]
    assert fa["fit_note"].str.contains("does not depend on x").all()
    assert not fits[(fits["family"] == "b") & (fits["metric"] == "Gmax_full_over_dw")][
        "fit_note"].str.len().any()


def test_build_detection_sides_tiers_and_gn() -> None:
    """G-F per tier, G-N on |dev - final|, signed families split into neg / pos / any."""
    th = vs.thresholds(PROTO)
    assert th == {"G-F": 0.01, "G-A": 0.005, "G-N_Gmax": 0.001, "G-N_eta": 0.001}
    # y = 8 x^2 (pos) and 0.5 x^2 (neg): pos exceeds 0.01 first at 0.05 (0.02; at 0.02 it is
    # 0.0032), neg only at 0.15 (0.01125; at 0.1 it is 0.005)
    det = vs.build_detection(_synthetic_paired("b", 0.5, 8.0), th)
    sel = det[(det["criterion"] == "G-F") & (det["tier"] == "development")].set_index("side")
    assert sel.loc["pos", "limit"] == 0.05 and sel.loc["pos", "bracket_lo"] == "0.02"
    assert sel.loc["neg", "limit"] == 0.15 and sel.loc["neg", "bracket_lo"] == "0.1"
    assert sel.loc["any", "limit"] == 0.05 and sel.loc["any", "p_at_limit"] == 0.05
    assert sel.loc["pos", "x_at_limit"] == 0.05
    fin = det[(det["criterion"] == "G-F") & (det["tier"] == "final")].set_index("side")
    assert fin.loc["pos", "limit"] == 0.05            # final = 1.1 x dev: 0.00352 at 0.02, 0.022
    gn = det[(det["criterion"] == "G-N_Gmax") & (det["tier"] == "dev-final")].set_index("side")
    assert set(gn.index) == {"neg", "pos", "any"}
    assert gn.loc["pos", "limit"] == 0.05 and gn.loc["neg", "limit"] == 0.15   # 0.1 * y > 1e-3
    ga = det[(det["criterion"] == "G-A") & (det["tier"] == "development")]
    assert not ga["reached"].any() and (ga["limit_str"] == vs.NOT_REACHED).all()


def test_select_kernel_closest_to_target_and_ties() -> None:
    """The kernel with the peak error nearest to -0.06; ties go to the smaller h."""
    for q in QS:
        rows = [{"kernel_h": h, "stage2_peak_rel_err_signed": -h / (4.0 * q)}
                for h in vs.KERNEL_H]
        ch = vs.select_kernel(rows)
        assert ch["h"] == 12.0 and ch["target"] == -0.06
    tie = vs.select_kernel([{"kernel_h": 7.0, "stage2_peak_rel_err_signed": -0.75},
                            {"kernel_h": 5.0, "stage2_peak_rel_err_signed": -0.25}],
                           target=-0.5)                      # distances 0.25 and 0.25 exactly
    assert tie["h"] == 5.0
    with pytest.raises(ValueError):
        vs.select_kernel([])


# ---------------------------------------------------------------------------------------------
# Pool, smoke, report
# ---------------------------------------------------------------------------------------------
def test_process_pool_equals_inline() -> None:
    """Two workers give the same rows (order and values) as inline execution."""
    kw = dict(qs=[50], tiers=["development"], families=["a", "c", "e"], proto=PROTO, git=GIT,
              only=["a_d+0.05", "c_h12", "c_h5", "e_d+0.05_h12"])
    r1, c1 = vs.run_evaluations(workers=1, **kw)
    r2, c2 = vs.run_evaluations(workers=2, **kw)
    drop = ["eval_wall_sec", "task_wall_sec"]
    d1, d2 = pd.DataFrame(r1).drop(columns=drop), pd.DataFrame(r2).drop(columns=drop)
    pd.testing.assert_frame_equal(d1, d2)
    assert len(r1) == 4 and c1 == c2


def test_family_e_kernel_choice_without_family_c_in_the_run() -> None:
    """Without (c) rows the peak error is computed directly and h = 12 is still chosen."""
    rows, choice = vs.run_evaluations([50, 60], ["development"], ["e"], 1, PROTO, GIT,
                                      only=["e_d+0.05_h12"])
    assert [r["kernel_h"] for r in rows] == [12.0, 12.0]
    for q in ("50", "60"):
        assert choice[q]["h"] == 12.0 and "directly" in choice[q]["source"]
    assert choice["50"]["peak_err"] == pytest.approx(-0.06, abs=1e-9)
    assert choice["60"]["peak_err"] == pytest.approx(-0.05, abs=1e-9)


def test_smoke_run_files_report_and_missing_tier(tmp_path: Path) -> None:
    """--smoke writes every file, the delta=0 floor block is absent, report renders (dev only)."""
    out = tmp_path / "smoke"
    assert vs.main(["--out", str(out), "--smoke"]) == 0
    for name in ("evaluations.csv", "paired.csv", "fits.csv", "detection_limits.csv",
                 "meta.json") + (("family_e_confirmation.csv",)
                                 if vs.CONFIRMATION_DIR.exists() else ()):
        assert (out / name).exists(), name
    figs = sorted(p.name for p in (out / "figures").iterdir())
    assert "d2_family_a.png" in figs and "d2_family_e.pdf" in figs
    ev = pd.read_csv(out / "evaluations.csv")
    assert list(ev["cand_id"]) == ["a_d+0.05", "c_h12", "e_d+0.05_h12"]
    assert set(ev["tier"]) == {"development"} and (ev["status"] == "ok").all()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_errors"] == 0 and meta["kernel_choice"]["50"]["h"] == 12.0
    assert meta["tiers_equal_protocol_record"] == {"development": True, "final": True}
    assert meta["protocol_sha256"].startswith("21d85983")
    assert meta["timing_by_tier"]["development"]["n"] == 3
    assert meta["numerical_floors_delta0"] == {}        # control not part of the smoke
    paired = pd.read_csv(out / "paired.csv")
    assert paired["Gmax_full_over_dw_final"].isna().all()
    rep = tmp_path / "report.md"
    assert vs.main(["--out", str(out), "--report", "--report-path", str(rep)]) == 0
    text = rep.read_text()
    assert "## 3. Unperturbed control" in text and "not part of this run" in text
    assert "n/a" in text and "Quadratic fit through the origin" in text


def test_report_without_confirmation_files(tmp_path: Path) -> None:
    """No confirmation directory: no (e) comparison file, report says so and still renders."""
    out = tmp_path / "noconf"
    assert vs.main(["--out", str(out), "--smoke", "--confirmation-dir",
                    str(tmp_path / "does_not_exist")]) == 0
    assert not (out / "family_e_confirmation.csv").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["confirmation_per_run"] is None
    rep = tmp_path / "report.md"
    assert vs.main(["--out", str(out), "--report", "--report-path", str(rep)]) == 0
    assert "No comparison table in this run" in rep.read_text()


def test_final_only_run_with_control_and_report(tmp_path: Path) -> None:
    """Final-only tier: control floors recorded, dev columns blank, report renders."""
    out = tmp_path / "final_only"
    assert vs.main(["--out", str(out), "--qs", "60", "--tiers", "final",
                    "--families", "a", "d"]) == 0
    ev = pd.read_csv(out / "evaluations.csv")
    assert set(ev["tier"]) == {"final"} and len(ev) == 13 + 4
    meta = json.loads((out / "meta.json").read_text())
    fl = meta["numerical_floors_delta0"]["q60_final"]
    assert 0.0 < fl["Gmax_full_over_dw"] < 1e-6      # measured 3.51e-07 (stage-1 interpolation)
    assert fl["eta_T_over_dw"] < 1e-9 and fl["stage1_rel_err_signed"] == 0.0
    paired = pd.read_csv(out / "paired.csv")
    assert paired["Gmax_full_over_dw_dev"].isna().all()
    det = pd.read_csv(out / "detection_limits.csv")
    assert not (det["tier"] == "dev-final").any()          # G-N needs both tiers
    rep = tmp_path / "report.md"
    assert vs.main(["--out", str(out), "--report", "--report-path", str(rep)]) == 0
    assert "## 4. Family (a)" in rep.read_text()
