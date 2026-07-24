#!/usr/bin/env python3
"""Append error-convention columns to results/one_stage_claimb_summary.csv (C1).

The committed CSV's ``|ē−e*|`` column mixes conventions: raw rows carry the
MAE across seeds while polished rows carry the deviation of the cross-seed
mean. Per locked convention C1 (2026-07-24 session) the paper tables use
err_of_mean = |mean_over_seeds − e*|; this tool ADDS four explicitly named
columns so both conventions are on record, WITHOUT touching any existing cell:

  raw_err_of_mean, raw_mae_across_seeds,
  polished_err_of_mean, polished_mae_across_seeds

Per-row granularity mirrors the CSV (heterogeneous-cost rows are the
agent-combined scalar (a1+a2)/2 vs the combined e*). The 3P polished columns
use convention C2 (per-seed player-mean landings from polish_per_seed_all.json,
cross-seed mean 24.75/15.84); the historical ``Polished`` column (player-1 mean
24.68/15.82 from the phase0-verify log) is left untouched.

Sources: raw = the r5_sampled convergence JSONs; polished = per-seed landings in
results/one_stage_ablation/polish_per_seed_all.json. Documented in
results/README.md (appended section).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.theory import (  # noqa: E402
    e_star_two_players,
    e_star_three_players,
    e_star_two_players_asymmetric_cost,
    e_star_two_players_different_ability,
)

CSV_PATH = "results/one_stage_claimb_summary.csv"
NEW_COLS = ["raw_err_of_mean", "raw_mae_across_seeds",
            "polished_err_of_mean", "polished_mae_across_seeds"]


def _raw_per_seed(scenario: str, q: float):
    """Per-seed raw scalar (agent-combined where applicable) + matching e*."""
    if scenario == "Two-Player":
        fs = [f for f in sorted(glob.glob(
            f"results/two_players/convergence/ppo_q{q}_seed*_r5_sampled_convergence.json"))
            if "wh8_wl4" not in f]
        vals = [json.load(open(f))["final"]["effort"] for f in fs]
        return vals, e_star_two_players(q, 6.5, 3.0, 0.00055)
    if scenario == "Three-Player":
        fs = sorted(glob.glob(
            f"results/three_players/convergence/ppo_3p_q{q}_*r5_sampled_convergence.json"))
        vals = [json.load(open(f))["final"]["effort"] for f in fs]
        return vals, e_star_three_players(q, 6.5, 3.0, 0.001)
    if scenario == "Het. Cost":
        fs = sorted(glob.glob(
            f"results/different_cost/convergence/different_cost_ppo_q{q}_*r5_sampled_convergence.json"))
        vals = []
        for f in fs:
            d = json.load(open(f))
            vals.append(0.5 * (d["final"]["effort1"] + d["final"]["effort2"]))
        e1, e2 = e_star_two_players_asymmetric_cost(q, 8.0, 5.5, 0.0004, 0.00055)
        return vals, 0.5 * (e1 + e2)
    if scenario == "Het. Ability":
        fs = sorted(glob.glob(
            "results/different_ability/convergence/"
            f"different_ability_ppo_q{q}_*r5_sampled_std_convergence.json"))
        vals = [json.load(open(f))["final"]["effort"] for f in fs]
        return vals, float(e_star_two_players_different_ability(q, 6.5, 3.0, 0.0005, 10, 5))
    raise ValueError(scenario)


_PPS = json.load(open("results/one_stage_ablation/polish_per_seed_all.json"))["rows"]
_EXP_KEY = {"Two-Player": "two_players", "Three-Player": "three_players",
            "Het. Cost": "different_cost", "Het. Ability": "different_ability"}


def _pol_per_seed(scenario: str, q: float):
    rows = [r for r in _PPS
            if r["experiment"] == _EXP_KEY[scenario] and r["q"] == q]
    assert len(rows) == 5, (scenario, q, len(rows))
    return [float(r["single_value"]) for r in rows]


def main() -> int:
    with open(CSV_PATH, newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    header = rows[0]
    assert not any(c in header for c in NEW_COLS), "columns already added"

    out_rows = [header + NEW_COLS]
    added = []
    for row in rows[1:]:
        scenario, q_s, method = row[0], row[1], row[2]
        q = float(q_s)
        if method == "Theory":
            new = ["0.00", "0.00", "-", "-"]
        else:
            raw_vals, es = _raw_per_seed(scenario, q)
            pol_vals = _pol_per_seed(scenario, q)
            raw_vals, pol_vals = np.array(raw_vals), np.array(pol_vals)
            new = [f"{abs(raw_vals.mean() - es):.2f}",
                   f"{np.abs(raw_vals - es).mean():.2f}",
                   f"{abs(pol_vals.mean() - es):.2f}",
                   f"{np.abs(pol_vals - es).mean():.2f}"]
            added.append((scenario, q_s, new))
        out_rows.append(row + new)

    with open(CSV_PATH, "w", newline="") as fh:
        csv.writer(fh, lineterminator="\r\n").writerows(out_rows)

    print(f"[extended] {CSV_PATH}: +{len(NEW_COLS)} columns "
          f"({', '.join(NEW_COLS)}); existing cells untouched")
    for scenario, q_s, new in added:
        print(f"  {scenario:13s} q{q_s}: err_of_mean raw={new[0]} pol={new[2]} | "
              f"MAE raw={new[1]} pol={new[3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
