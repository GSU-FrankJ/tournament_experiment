#!/usr/bin/env python3
"""q=45 post-polish exploitability for the 2P Set-1 cell — mirrors phase 1 exactly.

Fills the MISSING q=45 cell of the one-stage ablation exploitability record.
Replicates ``tools/one_stage_ablation_phase1.py`` for q=45, with one structural
difference stated explicitly: phase 1's headline point ``e_fp`` averaged ALL 7
polish landings in a cell (A2 per-seed n=5 + phase-0 probe controls n=2); no
probe controls exist at q=45, so here the evaluated cross-seed point is the mean
of the 5 per-seed landings from ``polish_per_seed_all.json`` (the same object
T-A/T-C quote as the polished estimate, 35.09).

Evaluated objects (all Set-1 params k=0.00055, w=6.5/3.0):
  1. per-seed polished landings (5): referee EXP_det + EXP_UCB at each
  2. cross-seed mean landing (35.0902...): EXP_det + EXP_UCB   <- headline cell
  3. cross-seed mean RAW effort: EXP_det + EXP_UCB              (raw-arm parity)
  4. legacy MC (shipped eval_exploitability via point adapter, M=8192, CRN,
     grid 5.0/1.0/0.25, R=5 reps, seeds 100+7r) at e*, raw mean, mean landing

Writes (sibling artifact — committed ablation_results.json is NOT touched):
  results/one_stage_ablation/ablation_results_q45.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.mc_br_polish as _mp  # noqa: E402,F401  (import-order guard, as in phase 1)

from tools.one_stage_mc_adapter import PointPolicyAgent  # noqa: E402
from tools.one_stage_referee import (  # noqa: E402
    BOUNDS, DW, K, W_H, W_L, e_star, exp_det, exp_ucb,
)
from run.run_two_players import eval_exploitability  # noqa: E402

Q = 45.0
SEEDS = (42, 43, 44, 45, 46)
R_REPS = 5
MC_M = 8192
GRID_CFG = {"stage_a_step": 5.0, "stage_b_radius": 15.0, "stage_b_step": 1.0,
            "stage_c_radius": 3.0, "stage_c_step": 0.25}
OUT = "results/one_stage_ablation/ablation_results_q45.json"


def legacy_mc(effort: float, q: float, reps: int = R_REPS) -> dict:
    """SHIPPED eval_exploitability on a deterministic effort via the point adapter."""
    vals = []
    for r in range(reps):
        ag = PointPolicyAgent(effort, BOUNDS)
        out = eval_exploitability(ag, q=q, effort_bounds=BOUNDS, M=MC_M,
                                  grid_cfg=GRID_CFG, seed=100 + 7 * r,
                                  w_h=W_H, w_l=W_L, k=K)
        vals.append(float(out["exploitability"]))
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
            "mean_over_dw": float(np.mean(vals)) / DW,
            "sd_over_dw": float(np.std(vals, ddof=1)) / DW, "vals": vals}


def main() -> int:
    es = e_star(Q)

    # raw per-seed efforts, reconstructed from final Beta (alpha, beta) as in phase 1
    lo, hi = BOUNDS
    raw = []
    for s in SEEDS:
        p = f"results/two_players/convergence/ppo_q{Q:g}.0_seed{s}_r5_sampled_convergence.json"
        d = json.load(open(p))
        a, b = d["alpha_mean"][-1], d["beta_mean"][-1]
        raw.append({"seed": s, "alpha": a, "beta": b,
                    "e": lo + (hi - lo) * a / (a + b),
                    "json_final_effort": d["final"]["effort"],
                    "json_exploit_max": d["final_exploit_max"]})
    raw_mean = float(np.mean([r["e"] for r in raw]))

    # per-seed polished landings from the committed per-seed polish artifact
    pps = json.load(open("results/one_stage_ablation/polish_per_seed_all.json"))
    pol = [r for r in pps["rows"] if r["experiment"] == "two_players" and r["q"] == Q]
    assert len(pol) == 5, f"expected 5 q45 polish rows, found {len(pol)}"
    landings = [float(r["single_value"]) for r in pol]
    land_mean = float(np.mean(landings))
    land_sd = float(np.std(landings, ddof=1))

    # referee EXP per seed (polished landings)
    pol_eval = []
    for r in pol:
        x = float(r["single_value"])
        d = exp_det(x, Q)
        pol_eval.append({
            "seed": r["seed"], "landing": x,
            "exp_det": d["exp"], "exp_det_over_dw": d["exp"] / DW,
            "exp_ucb_over_dw": exp_ucb(x, Q)["exp_ucb"] / DW,
        })

    res = {
        "params": {"w_h": W_H, "w_l": W_L, "dw": DW, "k": K, "bounds": list(BOUNDS)},
        "note": ("q=45 mirror of tools/one_stage_ablation_phase1.py. Headline point = "
                 "mean of the 5 per-seed polish landings (polish_per_seed_all.json); "
                 "phase 1's e_fp additionally averaged 2 probe-control landings, which "
                 "do not exist at q=45. Committed ablation_results.json untouched."),
        "cell": {
            "q": Q,
            "e_star": es,
            "raw": [dict(r, exp_det=exp_det(r["e"], Q)["exp"]) for r in raw],
            "raw_mean": raw_mean,
            "polish_per_seed": pol_eval,
            "polish_mean": land_mean,
            "polish_sd": land_sd,
            "exp_det_at_raw_mean_over_dw": exp_det(raw_mean, Q)["exp"] / DW,
            "exp_ucb_at_raw_mean_over_dw": exp_ucb(raw_mean, Q)["exp_ucb"] / DW,
            "exp_det_at_polish_mean_over_dw": exp_det(land_mean, Q)["exp"] / DW,
            "exp_ucb_at_polish_mean_over_dw": exp_ucb(land_mean, Q)["exp_ucb"] / DW,
            "legacy_mc": {
                "e_star": legacy_mc(es, Q),
                "raw_mean": legacy_mc(raw_mean, Q),
                "polish_mean": legacy_mc(land_mean, Q),
            },
        },
    }

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, default=float)

    c = res["cell"]
    print(f"q={Q}  e*={es:.4f}")
    print(f"  raw_mean={raw_mean:.4f}   polish_mean={land_mean:.4f} (sd {land_sd:.4f})")
    print(f"  EXP_det@raw_mean    = {c['exp_det_at_raw_mean_over_dw']*DW:.3e} abs "
          f"({c['exp_det_at_raw_mean_over_dw']:.3e}/DW)")
    print(f"  EXP_det@polish_mean = {c['exp_det_at_polish_mean_over_dw']*DW:.3e} abs "
          f"({c['exp_det_at_polish_mean_over_dw']:.3e}/DW)")
    per_seed = ", ".join(f"{p['seed']}:{p['exp_det']:.2e}" for p in pol_eval)
    print(f"  EXP_det per-seed landings (abs): {per_seed}")
    lm = c["legacy_mc"]
    print(f"  legacy MC (abs): e*={lm['e_star']['mean']:.2e}±{lm['e_star']['sd']:.1e}  "
          f"raw={lm['raw_mean']['mean']:.2e}±{lm['raw_mean']['sd']:.1e}  "
          f"pol={lm['polish_mean']['mean']:.2e}±{lm['polish_mean']['sd']:.1e}")
    print(f"[saved] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
