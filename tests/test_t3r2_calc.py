"""T3-R2: tests of the independent best-response calculator ``tools/t3/t3r2_br_calc.py`` (audit A5).

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_t3r2_calc.py -p no:cacheprovider -q
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "t3", ROOT / "tools" / "ms"):
    sys.path.insert(0, str(_p))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import t3r2_br_calc as bc  # noqa: E402
from utils.theory_multistage import F_xi, f_xi  # noqa: E402

W_H, W_L, K = 6.0, 2.0, 1.0 / 3500.0
COARSE = bc.CalcConfig(table_step=0.1, effort_step=0.05, panel_width=1.0, nodes_per_panel=4)


def game(q: float, T: int) -> bc.Game:
    return bc.Game(W_H, W_L, K, float(q), T)


def smooth_policy(q: float):
    """Smooth bounded stage policies (not equilibria): a hump at stage T, a ramp at the earlier stages."""
    def pol(t: int, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        if t == 3:
            return 5.0 + 55.0 * np.exp(-(d / (1.3 * q)) ** 2) + 5.0 * np.tanh(d / 80.0)
        if t == 2:
            return 12.0 + 30.0 * np.exp(-(d / (0.9 * q)) ** 2) - 6.0 * np.tanh(d / 50.0)
        return np.full(d.shape, 33.0)
    return pol


# ---------------------------------------------------------------------------------- independence
FORBIDDEN = {"verify", "gl_shock_rule", "stage_grid", "effort_grid", "evaluate"}


def test_the_calculator_uses_no_name_of_the_verifier_or_metrics_numerics():
    tree = ast.parse((ROOT / "tools" / "t3" / "t3r2_br_calc.py").read_text())
    seen = set()
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            seen.add(node.name)
        elif isinstance(node, ast.arg):
            seen.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            seen.add(node.arg)
        elif isinstance(node, ast.alias):
            seen.add(node.name.split(".")[-1])
            seen.add(node.asname or "")
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
    assert not (FORBIDDEN & seen), FORBIDDEN & seen
    assert not {m for m in modules if "dp_br_verifier" in m or "v2_metrics" in m}


# ---------------------------------------------------------------------------------- the shock distribution
@pytest.mark.parametrize("q", [20.0, 50.0])
def test_f_xi_and_F_xi_match_numerical_integration_of_the_triangular_density(q):
    assert abs(integrate.quad(lambda z: float(f_xi(z, q)), -2 * q, 2 * q, points=[0.0], epsabs=1e-13)[0] - 1.0) < 1e-12
    for x in np.linspace(-2.5 * q, 2.5 * q, 23):
        lo = -2 * q
        want = integrate.quad(lambda z: float(f_xi(z, q)), lo, min(x, 2 * q), points=[0.0] if lo < 0 < min(x, 2 * q) else None,
                              epsabs=1e-13)[0] if x > lo else 0.0
        assert abs(float(F_xi(x, q)[0]) - want) < 1e-11
    # the density is the triangle (2q - |x|)/(4 q^2) by definition
    assert abs(float(f_xi(0.0, q)) - 1.0 / (2 * q)) < 1e-15 and float(f_xi(2 * q, q)) == 0.0 and float(f_xi(3 * q, q)) == 0.0


def test_the_shock_rule_integrates_polynomials_and_has_unit_mass():
    nodes, w = bc.shock_rule(50.0, 1.0, 4)
    assert abs(w.sum() - 1.0) < 1e-13
    assert abs((w * nodes).sum()) < 1e-12
    assert abs((w * nodes ** 2).sum() - 2 * 50.0 ** 2 / 3.0) < 1e-9      # Var(xi) = 2 q^2 / 3
    assert nodes.min() > -100.0 and nodes.max() < 100.0
    assert bc.sym_grid(100.0, 0.05)[len(bc.sym_grid(100.0, 0.05)) // 2] == 0.0


# ---------------------------------------------------------------------------------- the T = 2 closed form
@pytest.mark.parametrize("q", [50.0, 60.0])
def test_the_closed_form_policy_has_no_gain_at_t2(q):
    g = game(q, 2)
    assert q > 41.83
    calc = bc.BRCalculator(g, bc.closed_form_policy(g))
    r = calc.root()
    assert r["EXP_root"] / g.dw <= 1e-6 and r["Delta_1"] / g.dw <= 1e-6
    d = np.linspace(-g.B, g.B, 50)
    sv = calc.stage_values(2, d)
    assert (sv["Delta"] / g.dw).max() <= 1e-6 and (sv["G"] / g.dw).max() <= 1e-6
    assert (sv["Delta"] >= -1e-12).all() and (sv["G"] >= sv["Delta"] - 1e-12).all()


def test_a_perturbed_closed_form_has_a_positive_gain_that_the_calculator_finds():
    g = game(50.0, 2)
    base = bc.closed_form_policy(g)

    def bad(t, d):
        out = np.asarray(base(t, d), dtype=float).copy()
        return out * 0.5 if t == 2 else out
    calc = bc.BRCalculator(g, bad)
    sv = calc.stage_values(2, [0.0])
    # at d = 0 the opponent plays 35 (half the peak 70); the first-order condition of the last stage,
    # DW f(e - 35) = 2 k e, i.e. e = 0.7 (100 - |e - 35|), gives the best response e = 94.5 / 1.7
    assert abs(float(sv["a_dev"][0]) - 94.5 / 1.7) < 1e-6
    def q_last(e):
        return W_L + g.dw * float(F_xi(e - 35.0, 50.0)[0]) - K * e ** 2
    assert abs(float(sv["Delta"][0]) - (q_last(94.5 / 1.7) - q_last(35.0))) < 1e-9


# ---------------------------------------------------------------------------------- T = 3 against direct quadrature
def test_the_t3_continuation_matches_direct_nested_quadrature():
    q = 30.0
    g = game(q, 3)
    pol = smooth_policy(q)
    calc = bc.BRCalculator(g, pol, COARSE)

    def v3(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        e, eo = pol(3, x), pol(3, -x)
        return W_L + g.dw * F_xi(x + e - eo, q) - K * e ** 2

    for d, e in ((-25.0, 10.0), (0.0, 40.0), (40.0, 3.0)):
        y = d + e - float(pol(2, np.array([-d]))[0])
        direct = 0.0
        for lo, hi in ((-2 * q, 0.0), (0.0, 2 * q)):
            val, _ = integrate.quad(lambda z: float(v3(y + z)[0]) * float(f_xi(z, q)), lo, hi, limit=800, epsabs=1e-12,
                                    epsrel=1e-12)
            direct += val
        want = -K * e ** 2 + direct
        got = float(calc.q_curve(2, d, np.array([e]), "mean")[0])
        assert abs(got - want) < 2e-7, (d, e, got, want)


def test_the_best_response_chain_is_at_least_the_mean_chain_and_the_root_gain_is_g1():
    q = 30.0
    g = game(q, 3)
    calc = bc.BRCalculator(g, smooth_policy(q), COARSE)
    for t in (3, 2):
        sv = calc.stage_values(t, np.linspace(-g.half(t), g.half(t), 9))
        assert (sv["G"] >= sv["Delta"] - 1e-9).all() and (sv["Delta"] >= -1e-12).all()
        assert (sv["a_br"] >= 0).all() and (sv["a_br"] <= 100).all()
    r = calc.root()
    assert r["V1_br"] >= r["V1_mean"] - 1e-12 and r["EXP_root"] >= 0.0 and r["Delta_1"] <= r["EXP_root"] + 1e-12
    # the terminal stage has one chain: G_T = Delta_T
    sv3 = calc.stage_values(3, [-30.0, 5.0])
    assert np.array_equal(sv3["G"], sv3["Delta"])


def test_the_vectorised_refinement_equals_scipy_bounded_brent():
    q = 30.0
    g = game(q, 3)
    calc = bc.BRCalculator(g, smooth_policy(q), COARSE)
    d = np.array([-70.0, -25.0, 0.0, 12.0, 55.0])
    for t in (3, 2):
        a = calc.stage_values(t, d, brent=False)
        b = calc.stage_values(t, d, brent=True)
        assert np.max(np.abs(a["Delta"] - b["Delta"])) < 1e-9 and np.max(np.abs(a["G"] - b["G"])) < 1e-9
        assert np.max(np.abs(a["a_dev"] - b["a_dev"])) < 2e-3 and np.max(np.abs(a["a_br"] - b["a_br"])) < 2e-3


def test_the_maximiser_handles_both_bounds():
    q = 30.0
    g = game(q, 2)
    calc = bc.BRCalculator(g, bc.closed_form_policy(g), COARSE)
    # far behind, the best response is zero effort (the bound e = 0); far ahead it is also zero
    for d in (-g.B, g.B):
        sv = calc.stage_values(2, [d])
        assert float(sv["a_dev"][0]) == 0.0
    # a hand-made concave objective with its maximum at the upper bound
    v, a = calc.maximize(lambda y: 0.2 * y, np.array([0.0]))
    assert 0.2 > 2 * K * 100.0 and float(a[0]) == pytest.approx(100.0)
    assert v[0] == pytest.approx(-K * a[0] ** 2 + 0.2 * a[0])


def test_the_convergence_helper_reports_small_changes_for_the_closed_form():
    g = game(50.0, 2)
    res = bc.convergence(g, bc.closed_form_policy(g), [(1, 0.0), (2, 0.0), (2, -60.0), (2, 30.0)])
    assert res["ok_values"], res
    # the root objective is nearly flat (curvature ~3e-4), so only the values converge there; the efforts of the
    # last stage, whose continuation is closed form, converge to the 1e-3 of the prompt
    assert all(max(r["e_hat_change"], r["a_dev_change"], r["a_br_change"]) <= 1e-3 for r in res["rows"] if r["t"] == 2), res


def test_the_calculator_agrees_with_the_verifier_on_a_synthetic_t3_policy():
    from envs.curriculum_env import GameSpec
    from utils.dp_br_verifier import FINAL_CONFIG
    from utils.v2_metrics import evaluate

    q = 40.0
    g = game(q, 3)
    pol = smooth_policy(q)
    spec = GameSpec(W_H, W_L, K, q, 3)
    ev = evaluate(pol, spec, FINAL_CONFIG)
    calc = bc.BRCalculator(g, pol)
    worst = 0.0
    for t in (1, 2, 3):
        grid, vm = ev.arrays[f"v_t{t}_d_grid"], ev.arrays[f"v_t{t}_v_mean"]
        idx = np.unique(np.round(np.linspace(0, grid.size - 1, 20)).astype(int)) if t > 1 else np.array([0])
        worst = max(worst, float(np.max(np.abs(calc.mean_value(t, grid[idx]) - vm[idx]))) / g.dw)
    assert worst < 1e-4, worst
    # the stage-3 gain at the verifier's own maximiser and the root gain agree to the G-N tolerance
    j = int(np.argmax(ev.arrays["v_t3_delta"]))
    sv = calc.stage_values(3, [float(ev.arrays["v_t3_d_grid"][j])])
    assert abs(float(sv["Delta"][0]) - float(ev.arrays["v_t3_delta"][j])) / g.dw < 1e-3
    assert abs(calc.root()["EXP_root"] - ev.res.exp_root) / g.dw < 1e-3


def brute_chain(g: bc.Game, pol, d_states, h: float = 0.1, de: float = 0.05):
    """An independent brute-force T = 3 chain (exact cell masses of the shock, linear interpolation, plain effort grid): returns
    Delta_2, G_2 at ``d_states`` and EXP_root. Shares nothing with the calculator but ``F_xi``."""
    q, k, dw, wl, B = g.q, g.k, g.dw, g.w_l, g.B
    grid = lambda half: np.arange(-int(round(half / h)), int(round(half / h)) + 1) * h          # noqa: E731
    z = grid(2 * q)
    mass = F_xi(z + h / 2, q) - F_xi(z - h / 2, q)
    E = np.arange(0.0, 100.0 + de / 2, de)
    x3 = grid(2 * B)
    e3, o3 = pol(3, x3), pol(3, -x3)
    vm3 = wl + dw * F_xi(x3 + e3 - o3, q) - k * e3 ** 2
    vb3 = np.empty_like(vm3)
    for i in range(0, x3.size, 500):
        sl = slice(i, i + 500)
        vb3[sl] = (wl + dw * F_xi((x3[sl, None] + E[None, :] - o3[sl, None]).ravel(), q).reshape(-1, E.size) - k * E ** 2).max(axis=1)
    vb3 = np.maximum(vb3, vm3)
    y2 = grid(B + 100.0)
    W3m = np.array([np.interp(y + z, x3, vm3) @ mass for y in y2])
    W3b = np.array([np.interp(y + z, x3, vb3) @ mass for y in y2])

    def stage2(d):
        d = np.atleast_1d(np.asarray(d, dtype=float))
        e, o = pol(2, d), pol(2, -d)
        vm = -k * e ** 2 + np.interp(d + e - o, y2, W3m)
        qm = -k * E[None, :] ** 2 + np.interp((d - o)[:, None] + E[None, :], y2, W3m)
        qb = -k * E[None, :] ** 2 + np.interp((d - o)[:, None] + E[None, :], y2, W3b)
        return vm, np.maximum(qm.max(axis=1), vm), np.maximum(qb.max(axis=1), vm)
    out = []
    for d in d_states:
        vm, qm, qb = stage2([d])
        out.append(((qm - vm)[0] / dw, (qb - vm)[0] / dw))
    x2 = grid(B)
    vm2, _, vb2 = (np.concatenate(a) for a in zip(*[tuple(map(np.atleast_1d, stage2(np.array([xx])))) for xx in x2]))
    y1 = grid(100.0)
    W2m = np.array([np.interp(y + z, x2, vm2) @ mass for y in y1])
    W2b = np.array([np.interp(y + z, x2, vb2) @ mass for y in y1])
    e1 = float(pol(1, np.zeros(1))[0])
    v1m = -k * e1 ** 2 + np.interp(0.0, y1, W2m)
    v1b = max((-k * E ** 2 + np.interp(E - e1, y1, W2b)).max(), v1m)
    return out, (v1b - v1m) / dw


def test_the_t3_chain_agrees_with_an_independent_brute_force_chain():
    q = 20.0
    g = game(q, 3)
    pol = smooth_policy(q)
    calc = bc.BRCalculator(g, pol, COARSE)
    ds = [-60.0, -15.0, 0.0, 22.0]
    bf, exp_bf = brute_chain(g, pol, ds)
    sv = calc.stage_values(2, ds)
    for i, _ in enumerate(ds):
        assert abs(float(sv["Delta"][i]) / g.dw - bf[i][0]) < 1e-5, (ds[i], sv["Delta"][i] / g.dw, bf[i])
        assert abs(float(sv["G"][i]) / g.dw - bf[i][1]) < 1e-5, (ds[i], sv["G"][i] / g.dw, bf[i])
    assert abs(calc.root()["EXP_root"] / g.dw - exp_bf) < 1e-5, (calc.root()["EXP_root"] / g.dw, exp_bf)
    assert max(b[1] for b in bf) > 1e-4                         # the comparison is not between zeros
