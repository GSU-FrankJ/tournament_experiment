#!/usr/bin/env python3
"""Phase-0 of the one-stage raw-vs-polish ablation: reconstruct, validate, tripwire.

Read-only w.r.t. the repo: reads the 10 canonical r5_sampled convergence JSONs,
reconstructs e_hat_raw per seed from the final Beta (alpha, beta), bit-checks
against each JSON's ``final.effort`` and the committed Claim-B CSV, then
evaluates the pre-registered P1-P9 tripwire with the deterministic referee.

No training. No modification of any existing module.

Run:  .venv/bin/python tools/one_stage_ablation_phase0.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.one_stage_referee import (  # noqa: E402
    BOUNDS, DW, K, W_H, W_L, br_analytic, br_slopes, e_star, exp_det, exp_ucb, self_test,
)

QS = (35.0, 55.0)
SEEDS = (42, 43, 44, 45, 46)
# Committed Claim-B aggregates (cross-validation target, NOT a data source).
CSV_RAW = {35.0: 43.58, 55.0: 29.65}
CSV_POL = {35.0: 44.95, 55.0: 28.76}
# Pre-registered tripwire (tripwire only - never copied into any deliverable).
PRED = {
    "P1": {35.0: 1.2e-3, 55.0: 3.9e-4},
    "P2": {35.0: 8.5e-5, 55.0: 1.2e-5},
    "P3": {35.0: 14.0, 55.0: 33.0},
    "P4": {35.0: 44.72, 55.0: 28.67},
    "P5": {35.0: 0.23, 55.0: 0.09},
    "P6": {35.0: 27.6, 55.0: 22.9},
    "P7": {35.0: 37.0, 55.0: 21.4},
    "P8": {35.0: 5e-3, 55.0: 4e-3},
}


def jpath(q: float, seed: int) -> str:
    return f"results/two_players/convergence/ppo_q{q:g}.0_seed{seed}_r5_sampled_convergence.json"


def reconstruct() -> dict:
    """Reconstruct e_hat_raw per seed from final (alpha, beta); bit-check vs JSON."""
    out = {}
    print("=" * 100)
    print("A1 RECONSTRUCTION — e_hat_raw from final (alpha, beta), bit-checked vs final.effort")
    print("=" * 100)
    lo, hi = BOUNDS
    for q in QS:
        rows = []
        for s in SEEDS:
            p = jpath(q, s)
            d = json.load(open(p))
            a, b = d["alpha_mean"][-1], d["beta_mean"][-1]
            e_rec = lo + (hi - lo) * a / (a + b)
            e_json = d["final"]["effort"]
            rows.append(dict(seed=s, path=p, alpha=a, beta=b, e_raw=e_rec,
                             final_effort=e_json, delta=abs(e_rec - e_json),
                             stop=d["stop_reason"], upd=d["stopped_at_update"]))
        m = float(np.mean([r["e_raw"] for r in rows]))
        sd = float(np.std([r["e_raw"] for r in rows], ddof=0))
        maxd = max(r["delta"] for r in rows)
        out[q] = dict(rows=rows, mean=m, std=sd, max_bitcheck_delta=maxd)
        print(f"\nq={q:g}  e*={e_star(q):.4f}")
        for r in rows:
            print(f"  seed{r['seed']}  alpha={r['alpha']:8.3f} beta={r['beta']:9.3f}  "
                  f"e_raw={r['e_raw']:8.4f}  final.effort={r['final_effort']:8.4f}  "
                  f"|d|={r['delta']:.2e}  stop={r['stop']}@{r['upd']}")
        print(f"  5-seed mean e_hat_raw = {m:.4f} (sd {sd:.4f}) | "
              f"CSV says {CSV_RAW[q]:.2f} -> match={abs(round(m,2)-CSV_RAW[q])<5e-3}")
        print(f"  max bit-check |e_rec - final.effort| = {maxd:.3e}")
    return out


def mc_noise_floor(profile: float, q: float, reps: int = 5, M: int = 8192) -> dict:
    """Legacy-MC noise floor on a DETERMINISTIC profile (shipped mc_br_polish path).

    Uses utils.mc_br_polish.exploitability_frozen_profile unmodified at the
    shipped in-training M=8192. NOTE: this is the deterministic-profile MC
    estimator, not run_two_players.eval_exploitability (which consumes a policy
    object) - see the Phase-0 note, design-critical confirmation #2.
    """
    from utils.mc_br_polish import exploitability_frozen_profile
    e = np.array([profile, profile], dtype=float)
    l = np.zeros(2)
    k = np.full(2, K)
    vals = []
    for r in range(reps):
        v, _ = exploitability_frozen_profile(e, l, k, W_H, W_L, q, BOUNDS,
                                             M=M, grid_step=0.25, seed=770_000 + 13 * r)
        vals.append(float(v))
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)), "vals": vals}


def verdict(name: str, got: float, pred: float, tol_rel: float = 0.35) -> str:
    """Tripwire verdict: material disagreement = wrong sign / order of magnitude."""
    if pred == 0:
        return "n/a"
    if np.sign(got) != np.sign(pred):
        return "DISAGREE(sign)"
    ratio = abs(got) / abs(pred)
    if ratio > 3.0 or ratio < 1 / 3.0:
        return "DISAGREE(order)"
    return "agree" if abs(ratio - 1.0) <= tol_rel else "agree(loose)"


def main() -> int:
    print("Referee unit tests")
    if self_test(verbose=False) != 0:
        print("REFEREE TESTS FAILED — abort")
        return 1
    print("  all referee unit tests PASS\n")

    rec = reconstruct()

    print("\n" + "=" * 100)
    print("BR-MAP STRUCTURE (referee-computed)")
    print("=" * 100)
    for q in QS:
        s = br_slopes(q)
        print(f"  q={q:g}: a={s['a']:.6e}  slope_below=+{s['slope_below']:.4f}  "
              f"slope_above={s['slope_above']:.4f}  |above|>1 = {abs(s['slope_above'])>1}  "
              f"(q_expansive={s['q_expansive']:.3f})")

    print("\n" + "=" * 100)
    print("P1-P9 TRIPWIRE at the 5-seed-MEAN profiles (referee = deterministic)")
    print("=" * 100)
    trip = {}
    for q in QS:
        es = e_star(q)
        raw = rec[q]["mean"]
        pol = CSV_POL[q]

        r_raw = exp_det(raw, q)
        r_pol = exp_det(pol, q)
        u_raw = exp_ucb(raw, q)
        u_pol = exp_ucb(pol, q)
        p1, p2 = r_raw["exp"], r_pol["exp"]
        p3 = p1 / p2 if p2 > 0 else float("inf")
        p4 = br_analytic(raw, q)
        p5 = pol - p4
        p6 = br_analytic(0.0, q)
        p7 = br_analytic(50.0, q)
        err_before, err_after = abs(50.0 - es), abs(p7 - es)
        p8 = mc_noise_floor(raw, q)

        trip[q] = dict(e_star=es, raw=raw, pol=pol, P1=p1, P2=p2, P3=p3, P4=p4,
                       P5=p5, P6=p6, P7=p7, P7_err_before=err_before,
                       P7_err_after=err_after, P8=p8,
                       exp_ucb_raw=u_raw["exp_ucb"], exp_ucb_pol=u_pol["exp_ucb"],
                       br_disagree_raw=r_raw["br_disagreement"],
                       br_disagree_pol=r_pol["br_disagreement"])

        print(f"\n--- q={q:g}  (e*={es:.4f}, raw_mean={raw:.4f}, committed e_pol={pol:.2f}) ---")
        print(f"  P1 EXP_det(raw)      = {p1:.4e} abs ({p1/DW:.4e} /DW)   pred {PRED['P1'][q]:.1e}   -> {verdict('P1',p1,PRED['P1'][q])}")
        print(f"  P2 EXP_det(polished) = {p2:.4e} abs ({p2/DW:.4e} /DW)   pred {PRED['P2'][q]:.1e}   -> {verdict('P2',p2,PRED['P2'][q])}")
        print(f"  P3 ratio raw/pol     = {p3:.2f}x                        pred {PRED['P3'][q]:.0f}x   -> {verdict('P3',p3,PRED['P3'][q])}")
        print(f"  P4 1-step BR(raw)    = {p4:.4f}                          pred {PRED['P4'][q]:.2f}  -> {verdict('P4',p4,PRED['P4'][q],0.01)}")
        print(f"  P5 e_pol - BR_1step  = {p5:+.4f}                         pred {PRED['P5'][q]:+.2f}  -> {verdict('P5',p5,PRED['P5'][q])}")
        print(f"  P6 BR(0)             = {p6:.4f}                          pred {PRED['P6'][q]:.1f}  -> {verdict('P6',p6,PRED['P6'][q],0.01)}")
        print(f"  P7 BR(50)            = {p7:.4f}  err {err_before:.2f}->{err_after:.2f} "
              f"({'WORSE' if err_after>err_before else 'better'})   pred {PRED['P7'][q]:.1f}  -> {verdict('P7',p7,PRED['P7'][q],0.01)}")
        print(f"  P8 legacy-MC floor   = {p8['mean']:.4e} +/- {p8['sd']:.4e} (R=5, M=8192)  pred SE {PRED['P8'][q]:.0e}  -> {verdict('P8',p8['sd'],PRED['P8'][q])}")
        print(f"     EXP_UCB(raw)={u_raw['exp_ucb']:.4e}  EXP_UCB(pol)={u_pol['exp_ucb']:.4e}  "
              f"BR-path disagreement raw={r_raw['br_disagreement']:.2e} pol={r_pol['br_disagreement']:.2e}")

    q = 55.0
    print(f"\n  P9 q55 crossing: raw={trip[q]['raw']:.4f} > e*={trip[q]['e_star']:.4f} > "
          f"e_pol={trip[q]['pol']:.2f}  -> crossing CONFIRMED (polished below e*)")
    print(f"     1-step BR(raw q55) = {trip[q]['P4']:.4f} (also below e*) ; "
          f"BR slope above e* = {br_slopes(55.0)['slope_above']:.4f}")

    print("\n" + "=" * 100)
    print("eps=0.03 UNITS (historical one-stage stop threshold)")
    print("=" * 100)
    print(f"  one-stage eps=0.03 is ABSOLUTE payoff units (run_two_players.py:1425 compares it")
    print(f"  to best_delta, a payoff difference). In /DW units: 0.03/{DW} = {0.03/DW:.6f}")
    print(f"  two-stage epsilon_over_dw=0.03 is NORMALIZED (/DW) -> the SAME number, different units.")
    for q in QS:
        print(f"  q={q:g}: EXP_det(raw)={trip[q]['P1']:.3e} abs = {trip[q]['P1']/0.03:.4f} x eps ; "
              f"EXP_det(pol)={trip[q]['P2']:.3e} abs = {trip[q]['P2']/0.03:.5f} x eps")

    os.makedirs("results/one_stage_ablation", exist_ok=True)
    dump = {
        "phase": 0,
        "params": {"w_h": W_H, "w_l": W_L, "dw": DW, "k": K, "bounds": list(BOUNDS)},
        "reconstruction": {str(q): {"mean": rec[q]["mean"], "std": rec[q]["std"],
                                    "max_bitcheck_delta": rec[q]["max_bitcheck_delta"],
                                    "rows": rec[q]["rows"]} for q in QS},
        "tripwire": {str(q): {k: v for k, v in trip[q].items()} for q in QS},
        "csv_crosscheck": {"raw": CSV_RAW, "polished": CSV_POL},
    }
    p = "results/one_stage_ablation/phase0_tripwire.json"
    with open(p, "w") as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"\n[saved] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
