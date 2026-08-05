#!/usr/bin/env python3
"""Persist per-seed canonical MC-BR polished efforts for ALL one-stage cells.

``tools/phase0_verify.py`` computes ``r.e_polished`` per seed but only prints the
cross-seed MEAN (``phase0_verify.py:47-51``); the per-seed polished values are
discarded. This driver runs the SAME canonical polish
(``utils.mc_br_polish.mc_br_polish`` with the shipped ``POL`` config) per seed and
PERSISTS the per-player polished efforts, so the equilibrium-recovery dot plot can
show per-seed *polished* (Claim-B) dots instead of raw PPO landings.

Reading, params, start point, and per-cell seed policy MATCH ``phase0_verify.py``
exactly, so the cross-seed means reproduce ``results/phase0_verify_20260701_1941.log``
(the Claim-B verification). Coverage:

    2P {q35, q45, q55}   start=Beta mean, seed=4000+si   (q45 has no log ref)
    3P {q35, q55}        start=Beta mode, seed=2000+si
    dc {q35, q55}        start=raw a1/a2, seed=2000+si
    da {q35, q55}        start=raw effort, seed=2000+si

Nothing in ``utils/`` is modified. CPU only, ~2 min/seed. Run in tmux:

    tmux new-session -d -s polish "cd /home/fjiang4/tournament_experiment && \
        .venv/bin/python tools/one_stage_polish_per_seed.py"
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mc_br_polish import beta_mean, beta_mode, mc_br_polish  # noqa: E402
from utils.theory import (  # noqa: E402
    e_star_two_players_asymmetric_cost,
    e_star_two_players_different_ability,
)

# Canonical polish config — identical to tools/phase0_verify.py:29-30.
POL = dict(eta=0.4, M=150_000, min_rounds=999, max_rounds=320, n_avg=200, tau_e=0.0,
           bias_correct=True)
B = (0.0, 100.0)
OUT = os.path.join("results", "one_stage_ablation", "polish_per_seed_all.json")

_SEED_RE = re.compile(r"seed(\d+)")


def _seed_of(path: str) -> int:
    m = _SEED_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def _polish_row(experiment, q, seed, si, polish_seed, e_star_vec, start_vec, labels):
    """Run one canonical polish and return a serialisable per-seed row."""
    l, k, wh, wl = labels["l"], labels["k"], labels["wh"], labels["wl"]
    t0 = time.time()
    r = mc_br_polish(start_vec, l, k, wh, wl, float(q), B, seed=polish_seed, **POL)
    sec = time.time() - t0
    e_pol = r.e_polished
    row = {
        "experiment": experiment,
        "q": float(q),
        "seed": int(seed),
        "si": int(si),
        "polish_seed": int(polish_seed),
        "e_star_per_player": [float(x) for x in np.atleast_1d(e_star_vec)],
        "start_raw_per_player": [float(x) for x in np.atleast_1d(start_vec)],
        "e_polished_per_player": [float(x) for x in np.atleast_1d(e_pol)],
        # Plot-facing scalars: single marker uses the player-mean; dc uses agent1/2.
        "single_value": float(np.mean(e_pol)),
        "agent1_effort": float(e_pol[0]),
        "agent2_effort": float(e_pol[-1]),
        "drift": float(r.drift),
        "rounds": int(r.rounds),
        "sec": round(sec, 2),
    }
    print(f"  {experiment:18s} q{int(q):<3} seed{seed} (si{si}) "
          f"start={np.array2string(np.atleast_1d(start_vec), precision=2)} "
          f"-> polished={np.array2string(np.atleast_1d(e_pol), precision=2)} "
          f"e*={np.array2string(np.atleast_1d(e_star_vec), precision=2)} "
          f"[{sec:.0f}s]", flush=True)
    return row


def cells_two_players():
    """2P Set-1 (k=0.00055, w=6.5/3.0), q in {35,45,55}. start=Beta mean, seed=4000+si."""
    rows = []
    for q in (35.0, 45.0, 55.0):
        fs = [f for f in sorted(glob.glob(
            f"results/two_players/convergence/ppo_q{q}_seed*_r5_sampled_convergence.json"))
            if "wh8_wl4" not in f]
        es = (6.5 - 3.0) / (4 * 0.00055 * q)
        labels = {"l": np.zeros(2), "k": np.full(2, 0.00055), "wh": 6.5, "wl": 3.0}
        for si, f in enumerate(fs):
            d = json.load(open(f))
            a, b = d["alpha_mean"][-1], d["beta_mean"][-1]
            mean = beta_mean(a, b, *B)
            rows.append(_polish_row("two_players", q, _seed_of(f), si, 4000 + si,
                                    np.array([es, es]), np.array([mean, mean]), labels))
    return rows


def cells_three_players():
    """3P (k=0.001, w=6.5/3.0), q in {35,55}. start=Beta mode, seed=2000+si."""
    rows = []
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(
            f"results/three_players/convergence/ppo_3p_q{q}_*r5_sampled_convergence.json"))
        es = (6.5 - 3.0) / (4 * 0.001 * q)
        labels = {"l": np.zeros(3), "k": np.full(3, 0.001), "wh": 6.5, "wl": 3.0}
        for si, f in enumerate(fs):
            d = json.load(open(f))
            a, b = d["alpha_mean"][-1], d["beta_mean"][-1]
            mode = beta_mode(a, b, *B)
            rows.append(_polish_row("three_players", q, _seed_of(f), si, 2000 + si,
                                    np.full(3, es), np.full(3, mode), labels))
    return rows


def cells_different_cost():
    """dc (k1=0.0004,k2=0.00055, w=8/5.5), q in {35,55}. start=raw a1/a2, seed=2000+si."""
    rows = []
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(
            f"results/different_cost/convergence/different_cost_ppo_q{q}_*r5_sampled_convergence.json"))
        e1s, e2s = e_star_two_players_asymmetric_cost(q, 8.0, 5.5, 0.0004, 0.00055)
        labels = {"l": np.zeros(2), "k": np.array([0.0004, 0.00055]), "wh": 8.0, "wl": 5.5}
        for si, f in enumerate(fs):
            d = json.load(open(f))
            start = np.array([d["history"]["agent1_effort"][-1],
                              d["history"]["agent2_effort"][-1]])
            rows.append(_polish_row("different_cost", q, _seed_of(f), si, 2000 + si,
                                    np.array([e1s, e2s]), start, labels))
    return rows


def cells_different_ability():
    """da (k=0.0005, l1=10,l2=5, w=6.5/3.0), q in {35,55}. start=raw effort, seed=2000+si."""
    rows = []
    for q in (35.0, 55.0):
        fs = sorted(glob.glob(
            "results/different_ability/convergence/"
            f"different_ability_ppo_q{q}_*r5_sampled_std_convergence.json"))
        es = max(0.0, min(100.0, e_star_two_players_different_ability(q, 6.5, 3.0, 0.0005, 10, 5)))
        labels = {"l": np.array([10.0, 5.0]), "k": np.full(2, 0.0005), "wh": 6.5, "wl": 3.0}
        for si, f in enumerate(fs):
            d = json.load(open(f))
            start = np.full(2, d["history"]["effort"][-1])
            rows.append(_polish_row("different_ability", q, _seed_of(f), si, 2000 + si,
                                    np.array([es, es]), start, labels))
    return rows


def main() -> int:
    print("=" * 96, flush=True)
    print("PER-SEED canonical MC-BR polish (persisted) — all one-stage cells", flush=True)
    print(f"  POL = {POL}", flush=True)
    print("=" * 96, flush=True)

    t0 = time.time()
    rows: List[dict] = []
    rows += cells_two_players()
    rows += cells_three_players()
    rows += cells_different_cost()
    rows += cells_different_ability()

    out = {
        "pol_config": POL,
        "bounds": list(B),
        "seed_policy": {"two_players": "4000+si", "three_players": "2000+si",
                        "different_cost": "2000+si", "different_ability": "2000+si"},
        "start_policy": {"two_players": "Beta mean", "three_players": "Beta mode",
                         "different_cost": "raw a1/a2", "different_ability": "raw effort"},
        "source": "tools/one_stage_polish_per_seed.py (reproduces phase0_verify per-seed polish)",
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)

    # Cross-seed-mean cross-check (should reproduce the Claim-B verify log).
    print("\n" + "=" * 96, flush=True)
    print("CROSS-SEED MEAN CROSS-CHECK (vs results/phase0_verify_20260701_1941.log)", flush=True)
    print("=" * 96, flush=True)
    by_cell: dict = {}
    for r in rows:
        by_cell.setdefault((r["experiment"], r["q"]), []).append(r["e_polished_per_player"])
    for (exp, q), vals in sorted(by_cell.items()):
        arr = np.array(vals)  # (n_seed, n_player)
        m = arr.mean(0)
        es = np.array(rows[[i for i, r in enumerate(rows)
                            if r["experiment"] == exp and r["q"] == q][0]]["e_star_per_player"])
        print(f"  {exp:18s} q{int(q):<3} polished(mean)="
              f"{np.array2string(m, precision=2)}  e*={np.array2string(es, precision=2)}  "
              f"|err|={np.array2string(np.abs(m - es), precision=2)}", flush=True)

    print(f"\n[wrote] {OUT}  ({len(rows)} rows, {time.time()-t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
