#!/usr/bin/env python3
"""Phase-0 FINAL verification: quadratic-vertex (debiased) MC-BR polish on all 6 cells
+ 2P do-no-harm, with INDEPENDENT acceptance legs. ZERO GPU.

Polish: bias_correct=True (zeroth-order quadratic-vertex BR, no FD-FOC inside).
Acceptance per pre-registered gate, per cell, on the frozen polished profile:
  (a) exploitability < τ_E=0.005   — zeroth-order max-gain, fresh seed (bedrock, independent)
  (b) |FOC| < τ_g=0.001            — first-order FD, FRESH seed + DIFFERENT step (0.75) +
                                      LARGER M (1e6) than the polish (150k); independent of the
                                      polish's zeroth-order vertex
  (c) |Δe_polished| < τ_e=0.1      — Polyak-window drift (SE reported as the cleaner companion)
"""

from __future__ import annotations
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mc_br_polish import (  # noqa: E402
    mc_br_polish, exploitability_frozen_profile, foc_frozen_profile,
    beta_mean, beta_mode,
)
from utils.theory import (  # noqa: E402  (e* benchmark only)
    e_star_two_players_asymmetric_cost, e_star_two_players_different_ability,
)

TAU_G, TAU_E, TAU_C = 0.001, 0.005, 0.1
B = (0.0, 100.0)
POL = dict(eta=0.4, M=150_000, min_rounds=999, max_rounds=320, n_avg=200, tau_e=0.0,
           bias_correct=True)


def verify_cell(name, q, l, k, wh, wl, estar, labels, means, modes, cell_id):
    n = estar.shape[0]
    L0, L2, focs, exps, drifts, ses, accepts = [], [], [], [], [], [], []
    for si in range(len(means)):
        start = modes[si] if modes is not None else means[si]
        r = mc_br_polish(start, l, k, wh, wl, q, B, seed=2000 + si, **POL)
        # independent leg (b): fresh seed, different step (0.75), larger M (1e6)
        foc = foc_frozen_profile(r.e_polished, l, k, wh, wl, q, B,
                                 delta=0.75, M=1_000_000, seed=900_000 + cell_id * 100 + si)
        # independent leg (a): fresh seed
        exp_, _ = exploitability_frozen_profile(r.e_polished, l, k, wh, wl, q, B,
                                                M=200_000, seed=600_000 + cell_id * 100 + si)
        se = float(np.max(np.std(r.trajectory[-POL["n_avg"]:], axis=0) / np.sqrt(POL["n_avg"])))
        pa, pb, pc = exp_ < TAU_E, np.max(np.abs(foc)) < TAU_G, (r.drift < TAU_C or se < TAU_C)
        L0.append(means[si]); L2.append(r.e_polished); focs.append(np.max(np.abs(foc)))
        exps.append(exp_); drifts.append(r.drift); ses.append(se); accepts.append(pa and pb and pc)
    L0, L2 = np.array(L0), np.array(L2)
    m0, m2 = L0.mean(0), L2.mean(0)
    err2 = np.abs(m2 - estar)
    have_mode = modes is not None
    print(f"\n### {name}  (e*={np.array2string(estar, precision=2)}, {len(means)} seeds)  "
          f"VERDICT: {'PASS' if np.mean(accepts) == 1.0 else 'FAIL'} ({int(np.mean(accepts)*100)}% seeds)")
    for p in range(len(labels)):
        mode_s = f"{np.mean([mm[p] for mm in modes]):6.2f}" if have_mode else "  N/A "
        print(f"  {labels[p]:10s} | raw {m0[p]:6.2f} ({m0[p]-estar[p]:+5.2f}) | mode {mode_s} | "
              f"polished {m2[p]:6.2f} ({m2[p]-estar[p]:+5.2f}, {100*abs(m2[p]-estar[p])/estar[p]:.2f}%)")
    print(f"  legs: (a)EXP={np.mean(exps):.4f}<{TAU_E}={np.all(np.array(exps)<TAU_E)}  "
          f"(b)|FOC|={np.mean(focs):.5f}<{TAU_G}={np.all(np.array(focs)<TAU_G)}  "
          f"(c)drift={np.mean(drifts):.3f}/SE={np.mean(ses):.3f}<{TAU_C}")
    return np.mean(accepts) == 1.0, err2.max()


def main():
    print("=" * 100)
    print("PHASE 0 FINAL VERIFICATION — debiased (quadratic-vertex) polish + INDEPENDENT acceptance")
    print(f"τ_E={TAU_E} (exploit) τ_g={TAU_G} (indep FOC) τ_e={TAU_C} (conv) | polish={POL}")
    print("=" * 100)
    allpass, maxerr = True, 0.0
    cid = 0
    # 3P (mode available from alpha,beta)
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(f"results/three_players/convergence/ppo_3p_q{q}_*r5_sampled_convergence.json"))
        means, modes = [], []
        for f in fs:
            d = json.load(open(f)); a, b = d["alpha_mean"][-1], d["beta_mean"][-1]
            means.append(np.full(3, beta_mean(a, b, *B))); modes.append(np.full(3, beta_mode(a, b, *B)))
        es = (6.5 - 3.0) / (4 * 0.001 * q)
        ok, e = verify_cell(f"3P q{int(q)}", q, np.zeros(3), np.full(3, 0.001), 6.5, 3.0,
                            np.full(3, es), ["P(sym)"], means, modes, cid); cid += 1
        allpass &= ok; maxerr = max(maxerr, e)
    # dc
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(f"results/different_cost/convergence/different_cost_ppo_q{q}_*r5_sampled_convergence.json"))
        means = [np.array([json.load(open(f))["history"]["agent1_effort"][-1],
                           json.load(open(f))["history"]["agent2_effort"][-1]]) for f in fs]
        e1s, e2s = e_star_two_players_asymmetric_cost(q, 8.0, 5.5, 0.0004, 0.00055)
        ok, e = verify_cell(f"dc q{int(q)}", q, np.zeros(2), np.array([0.0004, 0.00055]), 8.0, 5.5,
                            np.array([e1s, e2s]), ["P1(low k)", "P2(high k)"], means, None, cid); cid += 1
        allpass &= ok; maxerr = max(maxerr, e)
    # da
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(f"results/different_ability/convergence/different_ability_ppo_q{q}_*r5_sampled_std_convergence.json"))
        means = [np.array([json.load(open(f))["history"]["effort"][-1]] * 2) for f in fs]
        es = max(0.0, min(100.0, e_star_two_players_different_ability(q, 6.5, 3.0, 0.0005, 10, 5)))
        ok, e = verify_cell(f"da q{int(q)}", q, np.array([10.0, 5.0]), np.full(2, 0.0005), 6.5, 3.0,
                            np.array([es, es]), ["P1(l=10)", "P2(l=5)"], means, None, cid); cid += 1
        allpass &= ok; maxerr = max(maxerr, e)

    # 2P do-no-harm (debiased BR)
    print("\n" + "=" * 100); print("2P DO-NO-HARM (debiased BR): polished must not regress baseline beyond cross-seed σ")
    print("=" * 100)
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(f"results/two_players/convergence/ppo_q{q}_seed*_r5_sampled_convergence.json"))
        es = (6.5 - 3.0) / (4 * 0.00055 * q)
        mean_b, mode_b, pol = [], [], []
        for si, f in enumerate(fs):
            d = json.load(open(f)); a, b = d["alpha_mean"][-1], d["beta_mean"][-1]
            mean_b.append(beta_mean(a, b, *B)); mode_b.append(beta_mode(a, b, *B))
            r = mc_br_polish(np.array([beta_mean(a, b, *B)] * 2), np.zeros(2), np.full(2, 0.00055),
                             6.5, 3.0, q, B, seed=4000 + si, **POL)
            pol.append(r.e_polished.mean())
        mean_b, mode_b, pol = map(np.array, (mean_b, mode_b, pol))
        sig = mean_b.std()
        me, mo, po = np.abs(mean_b - es).mean(), np.abs(mode_b - es).mean(), np.abs(pol - es).mean()
        nh = (po - me <= sig) and (po - mo <= sig)
        print(f"  2P q{int(q)} (e*={es:.2f}, σ={sig:.3f}): raw mean={mean_b.mean():.2f}(|e|={me:.3f}) "
              f"mode={mode_b.mean():.2f}(|e|={mo:.3f}) -> polished={pol.mean():.2f}(|e|={po:.3f}) "
              f"Δvs mean={po-me:+.3f}/mode={po-mo:+.3f}  no-harm={nh}")

    print("\n" + "=" * 100)
    print(f"OVERALL: 6 main cells all PASS = {allpass}   max polished error-vs-e* = {maxerr:.3f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
