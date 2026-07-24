#!/usr/bin/env python3
"""Arm A2: per-seed canonical MC-BR polish for the one-stage ablation (Phase 1).

Per-seed polished values were never persisted (tools/phase0_verify.py:112-125 prints
only the cell mean), so they are regenerated here with the CANONICAL polish used
exactly as shipped:

    start = that seed's raw mean effort (reconstructed from the final Beta)
    seed  = 4000 + si   (si = 0..4 over sorted seeds 42..46)   [phase0_verify.py:117]
    POL   = eta=0.4, M=150k, min_rounds=999, max_rounds=320, n_avg=200,
            tau_e=0.0, bias_correct=True                        [phase0_verify.py:30]

utils/mc_br_polish.py is NOT modified. ~111 s per run, 10 runs, CPU only.

Run (tmux):  .venv/bin/python tools/one_stage_ablation_a2.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.one_stage_referee import BOUNDS, K, W_H, W_L, e_star  # noqa: E402
from utils.mc_br_polish import mc_br_polish  # noqa: E402

POL = dict(eta=0.4, M=150_000, min_rounds=999, max_rounds=320, n_avg=200,
           tau_e=0.0, bias_correct=True)
QS = (35.0, 55.0)
SEEDS = (42, 43, 44, 45, 46)


def raw_mean_for(q: float, seed: int) -> float:
    """Reconstruct that seed's raw effort from the final Beta (alpha, beta)."""
    p = f"results/two_players/convergence/ppo_q{q:g}.0_seed{seed}_r5_sampled_convergence.json"
    d = json.load(open(p))
    a, b = d["alpha_mean"][-1], d["beta_mean"][-1]
    lo, hi = BOUNDS
    return lo + (hi - lo) * a / (a + b)


def main() -> int:
    out = {"pol_config": POL, "seed_policy": "seed = 4000 + si over sorted seeds 42..46",
           "rows": []}
    print("=" * 96, flush=True)
    print("A2 — per-seed canonical MC-BR polish (start = that seed's raw mean)", flush=True)
    print(f"  POL = {POL}", flush=True)
    print("=" * 96, flush=True)
    print(f"{'q':>4} {'seed':>5} {'si':>3} {'polish_seed':>11} {'start_raw':>10} "
          f"{'landing':>9} {'e*':>8} {'land-e*':>8} {'rounds':>6} {'drift':>7} {'sec':>6}",
          flush=True)
    print("-" * 96, flush=True)
    for q in QS:
        es = e_star(q)
        for si, seed in enumerate(SEEDS):
            start = raw_mean_for(q, seed)
            t0 = time.time()
            r = mc_br_polish(np.array([start, start]), np.zeros(2), np.full(2, K),
                             W_H, W_L, q, BOUNDS, seed=4000 + si, **POL)
            land = float(r.e_polished.mean())
            dt = time.time() - t0
            row = dict(q=q, seed=seed, si=si, polish_seed=4000 + si, start_raw=start,
                       landing=land, e_star=es, signed_err=land - es,
                       rounds=int(r.rounds), drift=float(r.drift), sec=dt,
                       e_polished_per_player=[float(x) for x in r.e_polished])
            out["rows"].append(row)
            print(f"{q:>4.0f} {seed:>5} {si:>3} {4000+si:>11} {start:>10.4f} "
                  f"{land:>9.4f} {es:>8.4f} {land-es:>+8.4f} {r.rounds:>6} "
                  f"{r.drift:>7.4f} {dt:>6.1f}", flush=True)
    print("-" * 96, flush=True)
    for q in QS:
        L = np.array([r["landing"] for r in out["rows"] if r["q"] == q])
        print(f"  q={q:.0f}: A2 landings mean={L.mean():.4f} sd={L.std(ddof=1):.4f} "
              f"spread={L.max()-L.min():.4f}", flush=True)
    os.makedirs("results/one_stage_ablation", exist_ok=True)
    p = "results/one_stage_ablation/a2_polish_per_seed.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[saved] {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
