#!/usr/bin/env python3
"""Persist per-seed canonical MC-BR polished efforts for the 2P SET-2 cells.

Set-2 = the weight-variant two-player runs (w_H=8, w_L=4, k=0.0006), tag
``wh8_wl4_r5_sampled``, q in {35,45,55} x seeds 42..46. Runs the IDENTICAL
canonical polish as ``tools/one_stage_polish_per_seed.py`` (same ``POL`` config
from ``tools/phase0_verify.py:30``, same start policy = final Beta mean, same
per-cell seed policy 4000+si) and writes a SIBLING artifact — the committed
``polish_per_seed_all.json`` is never touched (append-only discipline).

Output: results/one_stage_ablation/polish_per_seed_set2.json

Run in tmux (~30-35 min CPU):
    tmux new-session -d -s polish_set2 "cd /home/fjiang4/tournament_experiment && \
        .venv/bin/python tools/one_stage_polish_per_seed_set2.py \
        > results/one_stage_ablation/polish_per_seed_set2.log 2>&1"
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mc_br_polish import beta_mean  # noqa: E402
from tools.one_stage_polish_per_seed import POL, B, _polish_row, _seed_of  # noqa: E402

OUT = os.path.join("results", "one_stage_ablation", "polish_per_seed_set2.json")

K2, WH2, WL2 = 0.0006, 8.0, 4.0


def cells_two_players_set2() -> List[dict]:
    """2P Set-2 (k=0.0006, w=8/4), q in {35,45,55}. start=Beta mean, seed=4000+si."""
    rows = []
    for q in (35.0, 45.0, 55.0):
        fs = sorted(glob.glob(
            f"results/two_players/convergence/ppo_q{q}_seed*_wh8_wl4_r5_sampled_convergence.json"))
        assert len(fs) == 5, f"expected 5 Set-2 files for q={q}, found {len(fs)}"
        es = (WH2 - WL2) / (4 * K2 * q)
        labels = {"l": np.zeros(2), "k": np.full(2, K2), "wh": WH2, "wl": WL2}
        for si, f in enumerate(fs):
            d = json.load(open(f))
            a, b = d["alpha_mean"][-1], d["beta_mean"][-1]
            mean = beta_mean(a, b, *B)
            rows.append(_polish_row("two_players_set2", q, _seed_of(f), si, 4000 + si,
                                    np.array([es, es]), np.array([mean, mean]), labels))
    return rows


def main() -> int:
    print("=" * 96, flush=True)
    print("PER-SEED canonical MC-BR polish — 2P SET-2 (wh8_wl4) cells", flush=True)
    print(f"  POL = {POL}  |  k={K2}, w_H={WH2}, w_L={WL2}", flush=True)
    print("=" * 96, flush=True)

    t0 = time.time()
    rows = cells_two_players_set2()

    out = {
        "pol_config": POL,
        "bounds": list(B),
        "seed_policy": {"two_players_set2": "4000+si"},
        "start_policy": {"two_players_set2": "Beta mean"},
        "params": {"k": K2, "w_h": WH2, "w_l": WL2},
        "source": "tools/one_stage_polish_per_seed_set2.py (identical POL as Set-1 canonical polish)",
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n" + "=" * 96, flush=True)
    print("CROSS-SEED MEANS (Set-2)", flush=True)
    print("=" * 96, flush=True)
    for q in (35.0, 45.0, 55.0):
        vals = np.array([r["single_value"] for r in rows if r["q"] == q])
        es = (WH2 - WL2) / (4 * K2 * q)
        print(f"  q{int(q)}: polished mean={vals.mean():.4f} sd={vals.std(ddof=1):.4f}  "
              f"e*={es:.4f}  err_of_mean={abs(vals.mean() - es):.4f}", flush=True)

    print(f"\n[wrote] {OUT}  ({len(rows)} rows, {time.time()-t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
