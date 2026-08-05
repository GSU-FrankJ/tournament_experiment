"""Deterministic extracted-policy re-certification of the T>=3 multi-stage runs.

Applies the corrected certification semantics (deterministic nested-grid dReach
with the discretization UCB, ``utils.dp_verifier.certify_refined``) to the
existing T=3/4/5 gated runs WITHOUT retraining.

IMPORTANT — what this is and is not:
  * It rebuilds the learned effort function e_hat_t(d) from the saved
    ``effort_curves`` (Beta-MEAN extraction, linear interpolation with flat
    tails) and certifies THAT deterministic policy. This is an
    **extracted-policy certification**, NOT a full stochastic-policy
    re-evaluation.
  * The T=3/4/5 runs saved no ``.pt`` checkpoint and no Beta (alpha, beta), and
    ``effort_curves`` cover only ``|d| <= 4q``; states beyond that use flat
    extrapolation. This is faithful for the reachable-support certificate (the
    reachable mass is concentrated near d=0), but a fully faithful re-evaluation
    of the trained network would require a retrain.

Writes ``results/multi_stage/recertification_T345.json`` and prints a table.

Run:
    OMP_NUM_THREADS=4 python tools/recertify_multistage.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dp_verifier import certify_refined  # noqa: E402

WH, WL, K, Q, EBAR = 6.0, 2.0, 1.0 / 3500.0, 50.0, 100.0
DW = WH - WL
CONV = os.path.join("results", "multi_stage", "convergence")
EPS = 0.03


def make_extracted_policy(run: dict, T: int):
    """Interpolant e_hat_t(d) from a run's saved effort_curves (flat tails)."""
    ec = run["effort_curves"]
    d = np.asarray(ec["d_grid"], dtype=float)
    curves = {t: np.asarray(ec["stages"][str(t)]["learned"], dtype=float)
              for t in range(1, T + 1)}

    def policy(t: int, dd) -> np.ndarray:
        dd = np.asarray(dd, dtype=float)
        return np.interp(dd, d, curves[t])  # np.interp clamps -> flat tails

    return policy


def recertify_T(T: int) -> List[Dict]:
    """Re-certify every seed of horizon ``T`` from its saved effort curves."""
    rows: List[Dict] = []
    files = sorted(glob.glob(os.path.join(CONV, f"ms_T{T}_q50_seed*_gateT{T}_convergence.json")))
    for f in files:
        run = json.load(open(f))
        seed = int(f.split("seed")[1].split("_")[0])
        if "effort_curves" not in run:
            rows.append({"T": T, "seed": seed, "status": "MISSING effort_curves"})
            continue
        pol = make_extracted_policy(run, T)
        cert = certify_refined(pol, w_h=WH, w_l=WL, k=K, q=Q, T=T, e_bar=EBAR,
                               epsilon_over_dw=EPS)
        old = run.get("final_eval", {}).get("delta_sum_reachable", float("nan")) / DW
        rows.append({
            "T": T, "seed": seed,
            "old_dreach_over_dw": float(old),
            "dreach_coarse_over_dw": cert["dreach_coarse_over_dw"],
            "dreach_fine_over_dw": cert["dreach_fine_over_dw"],
            "dreach_ucb_over_dw": cert["dreach_ucb_over_dw"],
            "certified_ucb": bool(cert["certified"]),
        })
    return rows


def main() -> int:
    print("Deterministic extracted-policy re-certification (T=3,4,5) — dReach_UCB gate")
    print("(interpolated MEAN-effort policy from saved effort_curves; NOT a network re-eval)\n")
    all_rows: List[Dict] = []
    summary: Dict[str, Dict] = {}
    header = f"{'T':>2} {'seed':>4} {'old dR/DW':>10} {'coarse':>8} {'fine':>8} {'UCB/DW':>8} {'cert':>5}"
    for T in (3, 4, 5):
        rows = recertify_T(T)
        all_rows.extend(rows)
        print(f"--- T={T} ---")
        print(header)
        ucbs, ncert = [], 0
        for r in rows:
            if r.get("status"):
                print(f"{r['T']:>2} {r['seed']:>4}   {r['status']}")
                continue
            print(f"{r['T']:>2} {r['seed']:>4} {r['old_dreach_over_dw']:>10.4f} "
                  f"{r['dreach_coarse_over_dw']:>8.4f} {r['dreach_fine_over_dw']:>8.4f} "
                  f"{r['dreach_ucb_over_dw']:>8.4f} {str(r['certified_ucb']):>5}")
            ucbs.append(r["dreach_ucb_over_dw"])
            ncert += int(r["certified_ucb"])
        if ucbs:
            summary[f"T{T}"] = {"n": len(ucbs), "n_cert": ncert,
                                "ucb_mean": float(np.mean(ucbs)),
                                "ucb_max": float(np.max(ucbs))}
            print(f"    -> certify {ncert}/{len(ucbs)}  UCB/DW mean {np.mean(ucbs):.4f} "
                  f"max {np.max(ucbs):.4f}\n")
    out = {"note": "extracted-policy (interpolated mean-effort) certification; "
                   "not a stochastic-policy re-eval; effort_curves cover |d|<=4q",
           "epsilon_over_dw": EPS, "rows": all_rows, "summary": summary}
    out_path = os.path.join("results", "multi_stage", "recertification_T345.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
