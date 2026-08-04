"""Per-seed canonical MC-BR polish for the r7/r8 (4-dim state, unified config) generation.

Same POL config, seed policy, and row schema as tools/one_stage_polish_per_seed.py
(the r5 tool), whose helpers are imported directly. Differences:

- Reads the r7/r8 tags:  2P Set1/Set2 + both fig7 ablation arms from r7,
  3P/dc/da from r8_unified (the no-floor, v2-everywhere wave).
- Writes results/one_stage_ablation/polish_per_seed_r7.json — a NEW file; the
  r5 output (polish_per_seed_all.json) is never touched.
- The two fig7 ablation arms are polished with the SAME polish seed (4000+si)
  as the full arm in the same (q, si) cell, so common random numbers cancel
  most estimator noise in arm-to-arm differences (dual-endpoint Table 3).
- da starts from the per-player raw endpoints [effort1, effort2] (the 4-dim
  runs report both); the r5 tool used a symmetric start, which no longer
  reflects the raw profile.

Stages (so 2P/ablation polish can run while the r8 GPU wave is still going):
  --stage A    2P Set1, Set2, fig7_no_stability, fig7_no_exploit   (60 rows)
  --stage B    3P, dc, da from r8_unified                          (30 rows)
  --stage all  both
  --list       print resolved cells and exit (no polishing)

Usage:
    python tools/one_stage_polish_per_seed_r7.py --stage A
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.one_stage_polish_per_seed import (  # noqa: E402
    B, POL, _polish_row, _seed_of, beta_mean, beta_mode,
)
from utils.theory import (  # noqa: E402
    e_star_two_players_asymmetric_cost,
    e_star_two_players_different_ability,
)

OUT = os.path.join("results", "one_stage_ablation", "polish_per_seed_r7.json")


def cells_two_players_arm(arm: str, variant: str = "") -> List[dict]:
    """2P cells for one arm/variant. seed=4000+si (CRN across arms per (q,si))."""
    rows = []
    vtag = f"{variant}_" if variant else ""
    if variant == "wh8_wl4":
        k, wh, wl = 0.0006, 8.0, 4.0
    else:
        k, wh, wl = 0.00055, 6.5, 3.0
    for q in (35.0, 45.0, 55.0):
        pat = f"results/two_players/convergence/ppo_q{q}_seed*_{vtag}{arm}_convergence.json"
        fs = [f for f in sorted(glob.glob(pat))
              if (("wh8_wl4" in f) == (variant == "wh8_wl4"))]
        es = (wh - wl) / (4 * k * q)
        labels = {"l": np.zeros(2), "k": np.full(2, k), "wh": wh, "wl": wl}
        exp = "two_players" + ("_set2" if variant == "wh8_wl4" else "")
        for si, f in enumerate(fs):
            d = json.load(open(f))
            a, b = d["alpha_mean"][-1], d["beta_mean"][-1]
            mean = beta_mean(a, b, *B)
            row = _polish_row(exp, q, _seed_of(f), si, 4000 + si,
                              np.array([es, es]), np.array([mean, mean]), labels)
            row["arm"] = arm
            rows.append(row)
    return rows


def cells_three_players_r8() -> List[dict]:
    """3P r8_unified. seed=2000+si, start=Beta mode (r5 convention)."""
    rows = []
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(
            f"results/three_players/convergence/ppo_3p_q{q}_*r8_unified_convergence.json"))
        es = (6.5 - 3.0) / (4 * 0.001 * q)
        labels = {"l": np.zeros(3), "k": np.full(3, 0.001), "wh": 6.5, "wl": 3.0}
        for si, f in enumerate(fs):
            d = json.load(open(f))
            a, b = d["alpha_mean"][-1], d["beta_mean"][-1]
            mode = beta_mode(a, b, *B)
            row = _polish_row("three_players", q, _seed_of(f), si, 2000 + si,
                              np.full(3, es), np.full(3, mode), labels)
            row["arm"] = "r8_unified"
            rows.append(row)
    return rows


def cells_different_cost_r8() -> List[dict]:
    """dc r8_unified. seed=2000+si, start=raw a1/a2 (r5 convention)."""
    rows = []
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(
            f"results/different_cost/convergence/different_cost_ppo_q{q}_*r8_unified_convergence.json"))
        e1s, e2s = e_star_two_players_asymmetric_cost(q, 8.0, 5.5, 0.0004, 0.00055)
        labels = {"l": np.zeros(2), "k": np.array([0.0004, 0.00055]), "wh": 8.0, "wl": 5.5}
        for si, f in enumerate(fs):
            d = json.load(open(f))
            start = np.array([d["history"]["agent1_effort"][-1],
                              d["history"]["agent2_effort"][-1]])
            row = _polish_row("different_cost", q, _seed_of(f), si, 2000 + si,
                              np.array([e1s, e2s]), start, labels)
            row["arm"] = "r8_unified"
            rows.append(row)
    return rows


def cells_different_ability_r8() -> List[dict]:
    """da r8_unified (v2 head). seed=2000+si, start=[effort1, effort2] (per-player raw)."""
    rows = []
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(
            "results/different_ability/convergence/"
            f"different_ability_ppo_q{q}_*r8_unified_convergence.json"))
        es = max(0.0, min(100.0, e_star_two_players_different_ability(q, 6.5, 3.0, 0.0005, 10, 5)))
        labels = {"l": np.array([10.0, 5.0]), "k": np.full(2, 0.0005), "wh": 6.5, "wl": 3.0}
        for si, f in enumerate(fs):
            d = json.load(open(f))
            fin = d["final"]
            start = np.array([fin.get("effort1", fin["effort"]),
                              fin.get("effort2", fin["effort"])])
            row = _polish_row("different_ability", q, _seed_of(f), si, 2000 + si,
                              np.array([es, es]), start, labels)
            row["arm"] = "r8_unified"
            rows.append(row)
    return rows


def stage_a() -> List[dict]:
    rows: List[dict] = []
    rows += cells_two_players_arm("r7_state4")
    rows += cells_two_players_arm("r7_state4", variant="wh8_wl4")
    rows += cells_two_players_arm("r7_fig7_no_stability")
    rows += cells_two_players_arm("r7_fig7_no_exploit")
    return rows


def stage_b() -> List[dict]:
    rows: List[dict] = []
    rows += cells_three_players_r8()
    rows += cells_different_cost_r8()
    rows += cells_different_ability_r8()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "all"], default="all")
    ap.add_argument("--list", action="store_true", help="print resolved cells, no polish")
    args = ap.parse_args()

    if args.list:
        import unittest.mock as mock
        with mock.patch(f"{__name__}._polish_row",
                        side_effect=lambda exp, q, seed, si, ps, es, st, lb: {
                            "experiment": exp, "q": q, "seed": seed, "si": si,
                            "polish_seed": ps,
                            "start_raw_per_player": list(np.atleast_1d(st).astype(float)),
                        }):
            rows = ([] if args.stage == "B" else stage_a()) + \
                   ([] if args.stage == "A" else stage_b())
        for r in rows:
            print(f"  {r['experiment']:20s} q{int(r['q']):<3} seed{r['seed']} "
                  f"ps={r['polish_seed']} start={np.round(r['start_raw_per_player'], 2)}")
        print(f"total {len(rows)} rows")
        return 0

    t0 = time.time()
    rows: List[dict] = []
    if args.stage in ("A", "all"):
        rows += stage_a()
    if args.stage in ("B", "all"):
        rows += stage_b()

    existing: List[dict] = []
    if os.path.exists(OUT):
        existing = json.load(open(OUT))["rows"]
        keys = {(r["experiment"], r.get("arm"), r["q"], r["seed"]) for r in rows}
        existing = [r for r in existing
                    if (r["experiment"], r.get("arm"), r["q"], r["seed"]) not in keys]

    out = {
        "pol_config": POL,
        "bounds": list(B),
        "generation": "r7/r8 (4-dim state, unified config, verifier M=16384)",
        "seed_policy": {"two_players*": "4000+si shared across the three Table-3 arms (CRN)",
                        "three_players": "2000+si", "different_cost": "2000+si",
                        "different_ability": "2000+si"},
        "start_policy": {"two_players*": "Beta mean", "three_players": "Beta mode",
                         "different_cost": "raw a1/a2",
                         "different_ability": "raw [effort1, effort2] (per-player)"},
        "source": "tools/one_stage_polish_per_seed_r7.py",
        "rows": existing + rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[wrote] {OUT}  ({len(existing + rows)} rows total, "
          f"+{len(rows)} this stage, {time.time()-t0:.0f}s)", flush=True)
    print("POLISH_STAGE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
