#!/usr/bin/env python3
"""
Three-player baseline_largeb: steps_per_update=16384 (4x default).

Runs q=25, q=40, q=55 with seed=42, saving as ablation 'baseline_largeb'.
Same total episodes (20.48M) but fewer, larger updates (1250 vs 5000).

Usage:
    python run/run_three_players_largeb.py
    python run/run_three_players_largeb.py --q 55          # single q
    python run/run_three_players_largeb.py --seed 123      # different seed
    python run/run_three_players_largeb.py --episodes 40960000  # more episodes
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.one_stage_three_players import config as base_config
from run.run_three_players import run_ppo


def main():
    parser = argparse.ArgumentParser(description="Three-player largeb experiment")
    parser.add_argument("--q", type=float, default=None, help="Single q value (default: all of 25,40,55)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20_480_000)
    parser.add_argument("--steps-per-update", type=int, default=16384,
                        help="Rollout batch size per update (default: 16384, 4x normal)")
    args = parser.parse_args()

    cfg = dict(base_config)
    cfg["seed"] = args.seed
    cfg["steps_per_update"] = args.steps_per_update

    q_values = [args.q] if args.q is not None else [25.0, 40.0, 55.0]

    print(f"[largeb] steps_per_update={args.steps_per_update}, "
          f"episodes={args.episodes}, seed={args.seed}, q={q_values}", flush=True)

    run_ppo(
        cfg,
        episodes=args.episodes,
        train_qs=q_values,
        ablation_name="baseline_largeb",
    )


if __name__ == "__main__":
    main()
