"""Phase-1 v2 verifier tests: invariants, candidate pmf, on/off split, recovery, purity.

Run: /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_v2_verifier.py -q

Tolerance for every invariant is ROUND = 1e-12 (units of DeltaW): floating-point rounding only.
The largest residual observed over 900 synthetic and 240 trained-checkpoint evaluations is
5.6e-16 (results/v2_pilots/phase1/invariants_probe.csv).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from common import TIERS, analytic_policy, spec_for, zero_policy  # noqa: E402
from envs.curriculum_env import step_gap  # noqa: E402
from run.run_final_dp_br import recovery_metrics as old_recovery  # noqa: E402
from utils.dp_br_verifier import stage_grid  # noqa: E402
from utils.theory_multistage import F_xi, g2_two_stage  # noqa: E402
from utils.v2_metrics import (  # noqa: E402
    append_csv,
    evaluate,
    onoff_split,
    recovery_metrics,
    save_npz,
    symmetric_grid,
)

ROUND = 1e-12
QS = (50.0, 60.0)


def _perturbed(spec, e1: float, amp: float, fr: float, ph: float):
    eq = analytic_policy(spec)

    def pol(t, d):
        d = np.asarray(d, float)
        if t == 1:
            return np.full(d.shape, e1)
        return np.clip(eq(2, d) + amp * np.sin(fr * d + ph), 0.0, 100.0)
    return pol


def _candidates(spec):
    return {
        "analytic": analytic_policy(spec),
        "zero": zero_policy,
        "const30": lambda t, d: np.full(np.asarray(d).shape, 30.0),
        "eq2_e1_low": _perturbed(spec, 10.0, 0.0, 0.01, 0.0),
        "wiggle": _perturbed(spec, 55.0, 8.0, 0.03, 1.0),
        "wiggle_fast": _perturbed(spec, 40.0, 5.0, 0.2, 2.0),
    }


@pytest.mark.parametrize("q", QS)
def test_zero_is_exact_grid_node(q):
    spec = spec_for(q)
    for cfg in TIERS:
        g = stage_grid(2, spec.B, cfg.state_step)
        assert g[g.size // 2] == 0.0
    assert symmetric_grid(spec.domain_half(2), 0.5)[int(np.ceil(spec.domain_half(2) / 0.5))] == 0.0


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("tier", [c.name for c in TIERS])
def test_invariants(q, tier):
    spec = spec_for(q)
    cfg = {c.name: c for c in TIERS}[tier]
    for name, pol in _candidates(spec).items():
        s = evaluate(pol, spec, cfg).scalars
        assert s["valid"], (name, s["invalid_reasons"])
        assert s["inv_GT_eq_DeltaT_absdiff_over_dw"] <= ROUND, name
        assert s["inv_EXP_eq_G1_absdiff_over_dw"] <= ROUND, name
        assert s["inv_Delta_le_G_pointwise_excess_over_dw"] <= ROUND, name
        assert s["inv_Deltamax_le_Gmax_excess_over_dw"] <= ROUND, name
        assert s["inv_Gmax_le_dFull_excess_over_dw"] <= ROUND, name
        assert s["inv_EXP_le_dReach_excess_over_dw"] <= ROUND, name
        assert s["inv_dReach_le_dFull_excess_over_dw"] <= ROUND, name


@pytest.mark.parametrize("q", QS)
def test_analytic_floor_and_zero_policy_discriminates(q):
    spec = spec_for(q)
    eq = evaluate(analytic_policy(spec), spec, TIERS[0]).scalars
    z = evaluate(zero_policy, spec, TIERS[0]).scalars
    assert eq["Gmax_full_over_dw"] < 1e-5
    assert z["Gmax_full_over_dw"] > 0.1


@pytest.mark.parametrize("q", QS)
def test_candidate_pmf_mass_and_moments(q):
    spec = spec_for(q)
    ev = evaluate(zero_policy, spec, TIERS[0])
    p = ev.pmf_cand[2]
    g = ev.res.stages[2].d_grid
    h = float(g[1] - g[0])
    assert abs(p.sum() - 1.0) <= 1e-12
    mean = float(np.sum(p * g))
    var = float(np.sum(p * g * g)) - mean ** 2
    assert abs(mean) <= 1e-9
    # triangular xi on [-2q, 2q] has variance 2q^2/3; the linear mass split adds at most h^2/4
    assert -1e-9 <= var - 2 * q * q / 3 <= h * h / 4 + 1e-9
    # support: drift is 0, so on-path nodes lie within one grid step of (-2q, 2q)
    on = g[p > 0]
    assert on.min() > -2 * q - h and on.max() < 2 * q + h


def test_onoff_split():
    g = np.linspace(-4, 4, 9)
    pmf = np.array([0, 0, .1, .2, .4, .2, .1, 0, 0])
    v = np.array([9., 1, 2, 3, 4, 5, 6, 7, 0])
    s = onoff_split(v, pmf, g)
    assert s["n_on"] == 5 and s["n_off"] == 4
    assert s["on_max"] == 6 and s["on_argmax_d"] == 2.0
    assert s["off_max"] == 9 and s["off_argmax_d"] == -4.0
    assert s["on_mean_unweighted"] == pytest.approx(4.0)
    assert s["on_mean_pmf_weighted"] == pytest.approx(.2 + .6 + 1.6 + 1.0 + .6)
    assert s["off_mean_unweighted"] == pytest.approx((9 + 1 + 7 + 0) / 4)


@pytest.mark.parametrize("q", QS)
def test_recovery_analytic_and_zero(q):
    spec = spec_for(q)
    s, _ = recovery_metrics(analytic_policy(spec), spec, 0.5)
    for k in ("stage1_rel_err_signed", "stage2_peak_rel_err_signed", "stage2_rmse_pos",
              "stage2_tail_mean", "stage2_tail_max", "stage2_sym_err_max"):
        assert s[k] == 0.0, k
    z, arr = recovery_metrics(zero_policy, spec, 0.5)
    assert z["stage1_rel_err_signed"] == -1.0 and z["stage2_peak_rel_err_signed"] == -1.0
    pos = np.abs(arr["recovery_d_grid"]) < 2 * q
    assert z["stage2_rmse_pos"] == pytest.approx(np.sqrt(np.mean(arr["recovery_g2"][pos] ** 2)))


@pytest.mark.parametrize("q", QS)
def test_recovery_agrees_with_existing_helper(q):
    spec = spec_for(q)
    pol = _perturbed(spec, 41.0, 6.0, 0.04, 0.7)
    new, _ = recovery_metrics(pol, spec, 0.5)
    old = old_recovery(pol, spec, 0.5)
    assert new["stage2_rmse_pos"] == pytest.approx(old["stage2_positive_region"]["role1"]["rmse"], abs=1e-12)
    assert new["stage2_tail_mean"] == pytest.approx(old["tail_region"]["role1_mean"], abs=1e-12)
    assert new["stage2_tail_max"] == pytest.approx(old["tail_region"]["role1_max"], abs=1e-12)
    assert new["stage2_sym_err_max"] == pytest.approx(old["symmetry"]["max_abs_e2_d_minus_e2_negd"], abs=1e-12)
    assert new["stage1_rel_err_signed"] * old["g1"] == pytest.approx(old["stage1"]["signed_error"], abs=1e-12)


def test_evaluate_consumes_no_rng():
    spec = spec_for(50.0)
    gens = [np.random.default_rng(s) for s in range(5)]
    before = [g.bit_generator.state for g in gens]
    np_state = np.random.get_state()
    t_state = torch.get_rng_state().clone()
    evaluate(_perturbed(spec, 41.0, 6.0, 0.04, 0.7), spec, TIERS[0])
    assert [g.bit_generator.state for g in gens] == before
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1]) and after[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), t_state)


@pytest.mark.parametrize("q", QS)
def test_terminal_win_probability_matches_cdf(q):
    spec = spec_for(q)
    rng = np.random.default_rng(777)
    n = 100_000
    for d, ei, ej in ((0.0, 50.0, 50.0), (30.0, 20.0, 60.0), (-80.0, 90.0, 10.0), (150.0, 0.0, 0.0)):
        eps_i, eps_j = rng.uniform(-q, q, n), rng.uniform(-q, q, n)
        dn = step_gap(spec, np.full(n, d), np.full(n, ei), np.full(n, ej), eps_i, eps_j)
        p = float(((spec.terminal_reward(dn) - spec.w_l) / spec.dw).mean())
        F = float(F_xi(np.array([d + ei - ej]), q)[0])
        se = np.sqrt(F * (1 - F) / n)
        assert (abs(p - F) <= 5 * se) if se > 0 else p == F


def test_storage_roundtrip(tmp_path):
    spec = spec_for(50.0)
    ev = evaluate(analytic_policy(spec), spec, TIERS[0])
    save_npz(ev, str(tmp_path / "a.npz"))
    z = np.load(tmp_path / "a.npz")
    assert np.array_equal(z["v_t2_G"], ev.G[2])
    assert np.array_equal(z["recovery_g2"], g2_two_stage(z["recovery_d_grid"], 50.0, 6, 2, 1 / 3500, 100))
    append_csv(str(tmp_path / "m.csv"), {"a": 1, "b": 2})
    append_csv(str(tmp_path / "m.csv"), {"a": 3, "b": 4})
    with pytest.raises(ValueError):
        append_csv(str(tmp_path / "m.csv"), {"a": 1, "c": 2})
