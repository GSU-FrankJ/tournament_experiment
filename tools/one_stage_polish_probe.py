#!/usr/bin/env python3
"""Phase-0 probe: does the canonical MC-BR polish behave as a SINGLE BR step?

Runs the canonical polish (exact POL config from tools/phase0_verify.py:30) from
several deterministic starts and compares each landing against:
  - the referee's exact 1-step BR of that start (what the ablation design assumed),
  - the closed-form e*,
  - the landing from the raw-mean start (init-independence check).

If the landings from different starts coincide (and differ from the 1-step BR),
the polish is a fixed-point solver, not a one-step BR -> the P4/P5/P6/P7
predictions and the raw->polished error decomposition must be re-specified.

Read-only w.r.t. the repo; uses utils.mc_br_polish exactly as shipped.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.one_stage_referee import BOUNDS, K, W_H, W_L, br_analytic, e_star  # noqa: E402
from utils.mc_br_polish import mc_br_polish  # noqa: E402

# EXACT canonical polish config (tools/phase0_verify.py:30) and 2P call site (:116-117).
POL = dict(eta=0.4, M=150_000, min_rounds=999, max_rounds=320, n_avg=200,
           tau_e=0.0, bias_correct=True)
RAW_MEAN = {35.0: 43.5798, 55.0: 29.6543}   # recomputed this session (phase0 tripwire)


def main() -> int:
    out = {"pol_config": POL, "rows": []}
    print("=" * 104)
    print("PHASE-0 PROBE — canonical MC-BR polish: single BR step or fixed-point solver?")
    print(f"  POL = {POL}")
    print("=" * 104)
    print(f"{'q':>5} {'start':>9} {'landing':>9} {'1step_BR':>9} {'e*':>8} "
          f"{'|land-BR1|':>10} {'|land-e*|':>9} {'rounds':>6} {'drift':>7} {'sec':>6}")
    print("-" * 104)
    for q in (35.0, 55.0):
        es = e_star(q)
        for start in (RAW_MEAN[q], 0.0, 50.0):
            t0 = time.time()
            r = mc_br_polish(np.array([start, start]), np.zeros(2), np.full(2, K),
                             W_H, W_L, q, BOUNDS, seed=4000, **POL)
            land = float(r.e_polished.mean())
            br1 = br_analytic(start, q)
            dt = time.time() - t0
            row = dict(q=q, start=start, landing=land, br_1step=br1, e_star=es,
                       dist_to_br1=abs(land - br1), dist_to_estar=abs(land - es),
                       rounds=int(r.rounds), drift=float(r.drift), sec=dt)
            out["rows"].append(row)
            print(f"{q:>5.0f} {start:>9.4f} {land:>9.4f} {br1:>9.4f} {es:>8.4f} "
                  f"{abs(land-br1):>10.4f} {abs(land-es):>9.4f} {r.rounds:>6} "
                  f"{r.drift:>7.4f} {dt:>6.1f}")
    print("-" * 104)
    for q in (35.0, 55.0):
        lands = [r["landing"] for r in out["rows"] if r["q"] == q]
        spread = max(lands) - min(lands)
        print(f"  q={q:.0f}: landings from starts {{raw, 0, 50}} = "
              f"{[round(x,3) for x in lands]}  spread={spread:.4f}  "
              f"-> {'INIT-INDEPENDENT (fixed-point solver)' if spread < 0.6 else 'init-dependent'}")
    os.makedirs("results/one_stage_ablation", exist_ok=True)
    p = "results/one_stage_ablation/phase0_polish_probe.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[saved] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
