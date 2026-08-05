"""MC-BR-polishing-ONLY baseline: run the canonical polish from uninformed starts.

Answers the reviewer question the dual-endpoint Table 3 raises: if MC-BR
polishing pulls every arm to e*, what does TEL-PPO contribute? This tool
measures what the polish does WITHOUT any training, so the paper can locate
TEL-PPO's contribution honestly (pre-registered narrative:
docs/ablation_narrative_preregistered.md).

Two arms per Table-4 cell (2P Set1 q35/45/55, 3P q35/55, dc q35/55, da q35/55):

  mid50   start = midpoint 50 for every player (PPO's own init mean),
          5 repeats over the SAME polish seeds as the TEL-PPO polish rows
          (2P: 4000+si, others: 2000+si) -> common random numbers, so the
          difference vs the TEL-PPO-start rows is start-driven only.
  ladder  starts 10/30/70/90 (same value for all players), one repeat each,
          seeds 4000/2000+si -> start-robustness of the solver.

Cost: the canonical POL runs a FIXED 320 rounds (min_rounds > max_rounds,
tau_e=0) — polish cost is start-independent BY DESIGN. Do not use this tool
to claim compute savings from a better start.

Output: results/one_stage_ablation/mc_br_only.json (never touches the
TEL-PPO polish files). Row schema matches polish_per_seed_r7.json; the
"seed" field holds the repeat index (there is no training seed here).

Usage:
    python tools/mc_br_only_baseline.py [--list]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.one_stage_polish_per_seed import B, POL, _polish_row  # noqa: E402
from utils.theory import (  # noqa: E402
    e_star_two_players_asymmetric_cost,
    e_star_two_players_different_ability,
)

OUT = os.path.join("results", "one_stage_ablation", "mc_br_only.json")

LADDER = (10.0, 30.0, 70.0, 90.0)


def cells() -> List[dict]:
    """Return per-cell specs: experiment, q, n_players, e*, labels, seed base."""
    out = []
    for q in (35.0, 45.0, 55.0):
        es = (6.5 - 3.0) / (4 * 0.00055 * q)
        out.append(dict(experiment="two_players", q=q, n=2, seed_base=4000,
                        e_star=np.array([es, es]),
                        labels={"l": np.zeros(2), "k": np.full(2, 0.00055),
                                "wh": 6.5, "wl": 3.0}))
    for q in (35.0, 55.0):
        es = (6.5 - 3.0) / (4 * 0.001 * q)
        out.append(dict(experiment="three_players", q=q, n=3, seed_base=2000,
                        e_star=np.full(3, es),
                        labels={"l": np.zeros(3), "k": np.full(3, 0.001),
                                "wh": 6.5, "wl": 3.0}))
    for q in (35.0, 55.0):
        e1s, e2s = e_star_two_players_asymmetric_cost(q, 8.0, 5.5, 0.0004, 0.00055)
        out.append(dict(experiment="different_cost", q=q, n=2, seed_base=2000,
                        e_star=np.array([e1s, e2s]),
                        labels={"l": np.zeros(2), "k": np.array([0.0004, 0.00055]),
                                "wh": 8.0, "wl": 5.5}))
    for q in (35.0, 55.0):
        es = max(0.0, min(100.0, e_star_two_players_different_ability(
            q, 6.5, 3.0, 0.0005, 10, 5)))
        out.append(dict(experiment="different_ability", q=q, n=2, seed_base=2000,
                        e_star=np.array([es, es]),
                        labels={"l": np.array([10.0, 5.0]), "k": np.full(2, 0.0005),
                                "wh": 6.5, "wl": 3.0}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    specs = cells()
    plan = []
    for c in specs:
        for si in range(5):
            plan.append((c, "mc_br_only_mid50", si, np.full(c["n"], 50.0),
                         c["seed_base"] + si))
        for si, s0 in enumerate(LADDER):
            plan.append((c, "mc_br_only_ladder", si, np.full(c["n"], s0),
                         c["seed_base"] + si))

    if args.list:
        for c, arm, si, start, ps in plan:
            print(f"  {c['experiment']:18s} q{int(c['q']):<3} {arm:18s} si{si} "
                  f"start={start[0]:.0f} ps={ps}")
        print(f"total {len(plan)} rows (~{len(plan) * 127 / 3600:.1f} h)")
        return 0

    t0 = time.time()
    rows: List[Dict] = []
    for c, arm, si, start, ps in plan:
        row = _polish_row(c["experiment"], c["q"], si, si, ps,
                          c["e_star"], start, c["labels"])
        row["arm"] = arm
        row["start_value"] = float(start[0])
        rows.append(row)

    out = {
        "pol_config": POL,
        "bounds": list(B),
        "design": "polish-only from uninformed starts; no training anywhere",
        "arms": {"mc_br_only_mid50": "start 50 (PPO init mean), 5 polish seeds (CRN "
                                     "with TEL-PPO polish rows)",
                 "mc_br_only_ladder": "starts 10/30/70/90, seeds base+si"},
        "note": "POL runs a fixed 320 rounds (no early stop): polish cost is "
                "start-independent by design; 'seed' field = repeat index.",
        "source": "tools/mc_br_only_baseline.py",
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[wrote] {OUT}  ({len(rows)} rows, {time.time()-t0:.0f}s)", flush=True)
    print("MC_BR_ONLY_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
