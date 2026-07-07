#!/usr/bin/env python3
"""Phase-0 layered decomposition: L0 (raw mean) -> L1 (+mode) -> L2 (+MC-BR polish).

ZERO GPU. Runs the Component-3 (mode) + Component-4 (polish) + post-polish
exploitability/FOC acceptance from `utils.mc_br_polish` on the EXISTING r5_sampled
3P / different-cost / different-ability runs, and reports the raw/mode/polished
table per cell with per-layer signed Delta and the threshold acceptance verdict.

Sampled-payoff path only (utils.mc_br_polish is verified against _payoff_player).
`utils.theory` provides the closed-form e* benchmark for error reporting only.

Thresholds (owner-proposed): tau_g=0.001, tau_E=0.005, tau_e=0.1.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mc_br_polish import (  # noqa: E402
    beta_mean, beta_mode, beta_std_effort, check_acceptance, mc_br_polish,
)
from utils.theory import (  # noqa: E402  (e* benchmark only)
    e_star_two_players_asymmetric_cost, e_star_two_players_different_ability,
)

TAU_G, TAU_E, TAU_E_CONV = 0.001, 0.005, 0.1
BOUNDS = (0.0, 100.0)


def _polish_cell(name, q, l, k, w_h, w_l, e_star, labels, seed_profiles, seed_modes):
    """seed_profiles: list of (n,) L0 mean efforts; seed_modes: list of (n,) modes or None."""
    n = e_star.shape[0]
    L0, L1, L2, accs, drifts, ses = [], [], [], [], [], []
    for si, e_mean in enumerate(seed_profiles):
        e_mode = seed_modes[si] if seed_modes is not None else e_mean
        # L2: polish starting from the L1 deterministic profile (mode if available, else mean)
        res = mc_br_polish(e_mode, l, k, w_h, w_l, q, BOUNDS, eta=0.3, M=200_000,
                           min_rounds=60, max_rounds=220, n_avg=80, tau_e=TAU_E_CONV,
                           seed=2000 + si)
        acc = check_acceptance(res.e_polished, l, k, w_h, w_l, q, BOUNDS, res.drift,
                               tau_E=TAU_E, tau_g=TAU_G, tau_e=TAU_E_CONV)
        se = float(np.max(np.std(res.trajectory[-80:], axis=0) / np.sqrt(80)))
        L0.append(e_mean); L1.append(e_mode); L2.append(res.e_polished)
        accs.append(acc); drifts.append(res.drift); ses.append(se)
    L0, L1, L2 = np.array(L0), np.array(L1), np.array(L2)
    print(f"\n### {name}  (e*={np.array2string(e_star, precision=2)}, {len(L0)} seeds)")
    have_mode = seed_modes is not None
    for p in range(len(labels)):
        m0, m1, m2 = L0[:, p].mean(), L1[:, p].mean(), L2[:, p].mean()
        s2 = L2[:, p].std()
        e = e_star[p]
        mode_cell = f"{m1:6.2f}" if have_mode else "  N/A "
        d01 = f"{m1 - m0:+.2f}" if have_mode else "  —  "
        print(f"  {labels[p]:10s} | L0 mean={m0:6.2f} ({m0-e:+5.2f}) | "
              f"L1 mode={mode_cell} ({d01}) | L2 pol={m2:6.2f}±{s2:.2f} ({m2-e:+5.2f}) | "
              f"Δ(→pol)={m2-m1:+.2f}")
    macc = np.mean([a.accepted for a in accs])
    pe = np.mean([a.pass_exp for a in accs]); pf = np.mean([a.pass_foc for a in accs])
    pd = np.mean([a.pass_drift for a in accs])
    exp_m = np.mean([a.exp_polished for a in accs])
    foc_m = np.mean([np.max(np.abs(a.foc)) for a in accs])
    print(f"  mean|err|: L0={np.mean(np.abs(L0.mean(0)-e_star)):.3f} -> "
          f"L2={np.mean(np.abs(L2.mean(0)-e_star)):.3f}   |  "
          f"accept {macc*100:.0f}% (EXP {pe*100:.0f}%/FOC {pf*100:.0f}%/conv {pd*100:.0f}%)")
    print(f"  scales: EXP_pol={exp_m:.4f}(τ_E={TAU_E})  max|FOC|={foc_m:.5f}(τ_g={TAU_G})  "
          f"drift={np.mean(drifts):.3f} / SE={np.mean(ses):.3f}(τ_e={TAU_E_CONV})")


def main():
    print("=" * 100)
    print("PHASE 0 DECOMPOSITION — L0 raw-mean -> L1 +mode -> L2 +MC-BR-polish (ZERO GPU)")
    print(f"thresholds: τ_g={TAU_G} (interior |FOC|), τ_E={TAU_E} (post-polish exploit), "
          f"τ_e={TAU_E_CONV} (convergence)")
    print("=" * 100)

    # --- 3P (symmetric): alpha,beta ARE stored -> mode computable ---
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(
            f"results/three_players/convergence/ppo_3p_q{q}_*r5_sampled_convergence.json"))
        means, modes = [], []
        for f in fs:
            d = json.load(open(f))
            a, b = d["alpha_mean"][-1], d["beta_mean"][-1]
            means.append(np.full(3, beta_mean(a, b, *BOUNDS)))
            modes.append(np.full(3, beta_mode(a, b, *BOUNDS)))
        estar = (6.5 - 3.0) / (4 * 0.001 * q)
        _polish_cell(f"3P q{int(q)}", q, np.zeros(3), np.full(3, 0.001), 6.5, 3.0,
                     np.full(3, estar), ["P(sym)"], means, modes)

    # --- different cost (asym): alpha,beta NOT stored -> L1 mode = N/A ---
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(
            f"results/different_cost/convergence/different_cost_ppo_q{q}_*r5_sampled_convergence.json"))
        means = [np.array([json.load(open(f))["history"]["agent1_effort"][-1],
                           json.load(open(f))["history"]["agent2_effort"][-1]]) for f in fs]
        e1s, e2s = e_star_two_players_asymmetric_cost(q, 8.0, 5.5, 0.0004, 0.00055)
        _polish_cell(f"dc q{int(q)}", q, np.zeros(2), np.array([0.0004, 0.00055]), 8.0, 5.5,
                     np.array([e1s, e2s]), ["P1(low k)", "P2(high k)"], means, None)

    # --- different ability (sym effort): alpha,beta NOT stored -> L1 mode = N/A ---
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(
            f"results/different_ability/convergence/different_ability_ppo_q{q}_*r5_sampled_std_convergence.json"))
        means = [np.array([json.load(open(f))["history"]["effort"][-1]] * 2) for f in fs]
        es = max(0.0, min(100.0, e_star_two_players_different_ability(q, 6.5, 3.0, 0.0005, 10, 5)))
        _polish_cell(f"da q{int(q)}", q, np.array([10.0, 5.0]), np.full(2, 0.0005), 6.5, 3.0,
                     np.array([es, es]), ["P1(l=10)", "P2(l=5)"], means, None)

    print("\n(L1 mode is 3P-only: dc/da JSONs store effort but not α,β. Per owner ruling,")
    print(" mode is a transparency diagnostic; polish is the undershoot fix.)")


if __name__ == "__main__":
    main()
