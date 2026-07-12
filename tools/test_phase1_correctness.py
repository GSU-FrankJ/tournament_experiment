"""Phase-1 correctness tests for the two-stage TEL-PPO pass.

Self-checking script (repo has no pytest suite; follows the tools/test_*.py
convention). Written BEFORE the fixes (test-first). Covers:

  T1  onpath_expected_stage2_effort: E[e_2(d_2)] = int e_2(delta) f_xi(delta) d(delta)
      by deterministic Gauss-Legendre quadrature -> reproduces the closed-form
      recovery target g1 = DW/(6qk) = 46.6667 on the CF stage-2 curve, and equals
      a dense-trapezoid reference.
  T2  certify_refined: deterministic nested-grid dReach with a discretization UCB
      dReach_UCB = dReach_fine + |dReach_fine - dReach_coarse|. UCB >= fine;
      identical on repeat (no RNG); CF policy certifies; const-high policy fails.
  T8  beta_mode_normalized: Beta mode (a-1)/(a+b-2) where a>1,b>1, else mean
      fallback; monotone mapping to effort.
  T7  grid independence: the checkpoint-SELECTION grid is disjoint from the two
      CERTIFICATION grids (selection cannot peek at the certifier, not even via the
      discretization UCB's coarse term).
  T9  SOC: at valid q=50 the global argmax of Q_2(d,.) coincides with the interior
      FOC g2(d) (validate stage-2 deviation gain ~ 0); at q=35 the SOC screen fails;
      the parameter threshold is q_soc = sqrt(DW/(8k)).

Run:
    python tools/test_phase1_correctness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.ppo_multi_stage import beta_mode_normalized  # noqa: E402
from utils.dp_verifier import certify_refined  # noqa: E402
from utils.multi_stage_metrics import onpath_expected_stage2_effort  # noqa: E402
from utils.theory_multistage import (  # noqa: E402
    f_xi,
    g1_two_stage,
    g2_two_stage,
    q_soc,
    validate_two_stage_params,
)

WH, WL, K, Q, EBAR = 6.0, 2.0, 1.0 / 3500.0, 50.0, 100.0
DW = WH - WL


def _check(name: str, cond: bool, detail: str = "") -> int:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return 0 if cond else 1


def cf_policy(t, d):
    """Closed-form symmetric benchmark policy e*_t(d)."""
    d = np.asarray(d, float)
    if t >= 2:
        return g2_two_stage(d, Q, WH, WL, K, EBAR)
    return np.full_like(d, g1_two_stage(Q, WH, WL, K))


def test_T1() -> int:
    print("T1  onpath_expected_stage2_effort (E[e2(d2)] recovery quadrature)")
    fails = 0
    g1 = g1_two_stage(Q, WH, WL, K)  # 46.6667 = target E[g2]

    e2_cf = lambda t, d: g2_two_stage(np.asarray(d, float), Q, WH, WL, K, EBAR)
    val = onpath_expected_stage2_effort(e2_cf, q=Q)
    fails += _check("E[g2(d2)] == g1 (46.6667)", abs(val - g1) < 1e-3,
                    f"got {val:.4f}, target {g1:.4f}")

    # dense-trapezoid reference on the same curve
    dd = np.linspace(-2 * Q, 2 * Q, 200001)
    ref = float(np.trapezoid(g2_two_stage(dd, Q, WH, WL, K, EBAR) * f_xi(dd, Q), dd))
    fails += _check("matches dense-trapezoid reference", abs(val - ref) < 1e-2,
                    f"quad {val:.4f} vs trap {ref:.4f}")

    # a constant policy c integrates to c (since f_xi is a normalized density)
    c = 30.0
    valc = onpath_expected_stage2_effort(lambda t, d: np.full_like(np.asarray(d, float), c), q=Q)
    fails += _check("constant policy c -> c", abs(valc - c) < 1e-3, f"got {valc:.4f}")
    return fails


def test_T2() -> int:
    print("T2  certify_refined (deterministic dReach_UCB gate)")
    fails = 0
    r = certify_refined(cf_policy, w_h=WH, w_l=WL, k=K, q=Q, T=2, e_bar=EBAR,
                        epsilon_over_dw=0.03)
    fails += _check("UCB >= fine (conservative bound)",
                    r["dreach_ucb"] >= r["dreach_fine"] - 1e-12,
                    f"ucb {r['dreach_ucb']:.5f} fine {r['dreach_fine']:.5f}")
    fails += _check("UCB = fine + |fine - coarse|",
                    abs(r["dreach_ucb"] - (r["dreach_fine"] + abs(r["dreach_fine"] - r["dreach_coarse"]))) < 1e-12)
    fails += _check("CF policy certifies (UCB/DW << 0.03)",
                    r["certified"] and r["dreach_ucb_over_dw"] < 0.03,
                    f"UCB/DW={r['dreach_ucb_over_dw']:.5f}")

    # determinism: identical on a repeat call (no Monte Carlo)
    r2 = certify_refined(cf_policy, w_h=WH, w_l=WL, k=K, q=Q, T=2, e_bar=EBAR,
                         epsilon_over_dw=0.03)
    fails += _check("deterministic (identical on repeat)",
                    r["dreach_ucb"] == r2["dreach_ucb"] and r["exp_fine"] == r2["exp_fine"])

    # falsification: a wildly-off constant-high policy must NOT certify
    bad = lambda t, d: np.full_like(np.asarray(d, float), EBAR)
    rb = certify_refined(bad, w_h=WH, w_l=WL, k=K, q=Q, T=2, e_bar=EBAR, epsilon_over_dw=0.03)
    fails += _check("const-high policy fails certification",
                    (not rb["certified"]) and rb["dreach_ucb_over_dw"] > 0.03,
                    f"UCB/DW={rb['dreach_ucb_over_dw']:.4f}")
    return fails


def test_T8() -> int:
    print("T8  beta_mode_normalized (mean/mode diagnostic extraction)")
    fails = 0
    # symmetric a=b=3 -> mode = mean = 0.5
    fails += _check("mode(3,3) == 0.5", abs(float(beta_mode_normalized(3.0, 3.0)) - 0.5) < 1e-9)
    # a=2,b=5 -> (2-1)/(2+5-2) = 1/5 = 0.2
    fails += _check("mode(2,5) == 0.2", abs(float(beta_mode_normalized(2.0, 5.0)) - 0.2) < 1e-9)
    # a<=1 -> fall back to mean a/(a+b) = 0.5/5.5
    fb = float(beta_mode_normalized(0.5, 5.0))
    fails += _check("a<=1 falls back to mean", abs(fb - 0.5 / 5.5) < 1e-9, f"got {fb:.5f}")
    # vectorized + guard mix
    a = np.array([3.0, 2.0, 0.5, 4.0]); b = np.array([3.0, 5.0, 5.0, 1.0])
    out = beta_mode_normalized(a, b)
    exp = np.array([0.5, 0.2, 0.5 / 5.5, 4.0 / 5.0])  # last: b<=1 -> mean 4/5
    fails += _check("vectorized guard/mix", np.allclose(out, exp, atol=1e-9),
                    f"{np.round(out, 4)}")
    return fails


def test_T7() -> int:
    print("T7  grid independence (selection grid disjoint from certification grids)")
    fails = 0
    from config.multi_stage_two_players import config as base_config
    from run.run_multi_stage import (
        CERT_COARSE_GRID_INDEX,
        CERT_FINE_GRID_INDEX,
        SELECTION_GRID_INDEX,
    )
    dgs = base_config["verifier"]["d_grid_sizes"]
    sel = dgs[SELECTION_GRID_INDEX]
    c_coarse = dgs[CERT_COARSE_GRID_INDEX]
    c_fine = dgs[CERT_FINE_GRID_INDEX]
    fails += _check("selection grid NOT a certification grid",
                    sel not in (c_coarse, c_fine),
                    f"sel={sel}, cert=({c_coarse},{c_fine})")
    fails += _check("certification coarse/fine distinct", c_coarse != c_fine,
                    f"({c_coarse},{c_fine})")
    fails += _check("expected grid roles (51 select / 101,201 certify)",
                    sel == 51 and c_coarse == 101 and c_fine == 201,
                    f"sel={sel}, cert=({c_coarse},{c_fine})")
    return fails


def test_T9() -> int:
    print("T9  SOC: global argmax of Q_2 == interior FOC at valid q; threshold")
    fails = 0
    # q_soc closed form
    qs = q_soc(WH, WL, K)
    fails += _check("q_soc == sqrt(DW/8k)", abs(qs - np.sqrt(DW / (8 * K))) < 1e-9,
                    f"q_soc={qs:.3f}")
    # valid q=50: stage-2 global best-response scan gain ~ 0 (interior FOC is global max)
    rep = validate_two_stage_params(q=50.0, w_h=WH, w_l=WL, k=K, e_bar=EBAR)
    fails += _check("q=50 valid (ok) and stage-2 dev gain ~ 0",
                    rep.ok and rep.max_stage2_deviation_gain < 1e-3,
                    f"ok={rep.ok} dev2={rep.max_stage2_deviation_gain:.2e}")
    # invalid q=35 < q_soc: SOC / validity fails
    rep_bad = validate_two_stage_params(q=35.0, w_h=WH, w_l=WL, k=K, e_bar=EBAR)
    fails += _check("q=35 invalid (SOC screen fails)", (not rep_bad.ok) and 35.0 <= qs,
                    f"ok={rep_bad.ok}, q_soc={qs:.2f}")
    return fails


def main() -> int:
    print("=" * 72)
    print("Phase-1 correctness tests (two-stage TEL-PPO)")
    print("=" * 72)
    fails = 0
    for t in (test_T1, test_T2, test_T7, test_T8, test_T9):
        fails += t()
        print()
    print("=" * 72)
    print(f"{'ALL PASS' if fails == 0 else str(fails) + ' CHECK(S) FAILED'}")
    print("=" * 72)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
