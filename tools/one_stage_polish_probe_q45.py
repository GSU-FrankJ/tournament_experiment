#!/usr/bin/env python3
"""q=45 init-independence probe for the canonical MC-BR polish.

Mirror of ``tools/one_stage_polish_probe.py`` (identical POL config, seed=4000,
starts {raw cross-seed mean, 0, 50}) for the 2P Set-1 q=45 cell, prompted by the
T-A observation that the polished err_of_mean (0.263) exceeds the raw
err_of_mean (0.108) there. If the landings from all three starts coincide near
35.09, the polished value is the solver's sampled fixed point e_fp (with its
systematic offset e_fp − e* ≈ −0.26) and not an artifact of the raw starts.

Writes: results/one_stage_ablation/q45_polish_probe.json
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

POL = dict(eta=0.4, M=150_000, min_rounds=999, max_rounds=320, n_avg=200,
           tau_e=0.0, bias_correct=True)
Q = 45.0
RAW_MEAN = 35.4616  # cross-seed mean of the 5 q45 r5_sampled final efforts
OUT = "results/one_stage_ablation/q45_polish_probe.json"


def main() -> int:
    es = e_star(Q)
    out = {"pol_config": POL, "q": Q, "raw_mean_start": RAW_MEAN, "rows": []}
    print(f"q=45 polish probe (POL as phase0_verify, seed=4000)  e*={es:.4f}")
    print(f"{'start':>9} {'landing':>9} {'1step_BR':>9} {'|land-e*|':>9} {'rounds':>6} {'sec':>6}")
    for start in (RAW_MEAN, 0.0, 50.0):
        t0 = time.time()
        r = mc_br_polish(np.array([start, start]), np.zeros(2), np.full(2, K),
                         W_H, W_L, Q, BOUNDS, seed=4000, **POL)
        land = float(r.e_polished.mean())
        dt = time.time() - t0
        out["rows"].append(dict(q=Q, start=start, landing=land,
                                br_1step=br_analytic(start, Q), e_star=es,
                                dist_to_estar=abs(land - es),
                                rounds=int(r.rounds), drift=float(r.drift), sec=dt))
        print(f"{start:>9.4f} {land:>9.4f} {br_analytic(start, Q):>9.4f} "
              f"{abs(land-es):>9.4f} {r.rounds:>6} {dt:>6.0f}", flush=True)
    lands = [r["landing"] for r in out["rows"]]
    out["landing_spread"] = float(max(lands) - min(lands))
    print(f"landing spread across starts: {out['landing_spread']:.4f}")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
