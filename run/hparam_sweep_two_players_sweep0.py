#!/usr/bin/env python3
"""
Sweep 0 hyperparameter search focused on aggressive PPO settings for q=55.

Targets configs that keep the lag/opponent setup fixed while widening clips,
pushing the KL controller higher, and preventing entropy collapse so that
approx_kl lands in a healthier band and abs_err drops below prior runs.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.one_stage_two_players import config as base_config
from run.run_two_players import run_ppo
from utils.logger import save_standardized_result


def make_experiments() -> List[Dict]:
    exps: List[Dict] = []

    # Sweep0-A: moderately aggressive baseline
    exps.append(
        {
            "id": "sweep0_A_mild",
            "clip_range_start": 0.30,
            "clip_range_end": 0.22,
            "entropy_coef_start": 0.02,
            "entropy_coef_hold": 0.02,
            "entropy_coef_end": 0.006,
            "target_kl": 0.020,
        }
    )

    # Sweep0-B: higher KL + wide clip, still moderate entropy
    exps.append(
        {
            "id": "sweep0_B_kl_high",
            "clip_range_start": 0.32,
            "clip_range_end": 0.24,
            "entropy_coef_start": 0.02,
            "entropy_coef_hold": 0.02,
            "entropy_coef_end": 0.006,
            "target_kl": 0.025,
        }
    )

    # Sweep0-C: more exploration + aggressive KL
    exps.append(
        {
            "id": "sweep0_C_explore_more",
            "clip_range_start": 0.30,
            "clip_range_end": 0.22,
            "entropy_coef_start": 0.03,
            "entropy_coef_hold": 0.03,
            "entropy_coef_end": 0.008,
            "target_kl": 0.020,
        }
    )

    # Sweep0-D: very aggressive (stress test)
    exps.append(
        {
            "id": "sweep0_D_very_aggressive",
            "clip_range_start": 0.35,
            "clip_range_end": 0.26,
            "entropy_coef_start": 0.03,
            "entropy_coef_hold": 0.03,
            "entropy_coef_end": 0.010,
            "target_kl": 0.030,
        }
    )

    # Sweep0-E: wide clip + safer KL (stability check)
    exps.append(
        {
            "id": "sweep0_E_wide_clip_safe",
            "clip_range_start": 0.30,
            "clip_range_end": 0.22,
            "entropy_coef_start": 0.025,
            "entropy_coef_hold": 0.025,
            "entropy_coef_end": 0.007,
            "target_kl": 0.018,
        }
    )

    # Sweep0-F: control variant vs previous runs
    exps.append(
        {
            "id": "sweep0_F_control",
            "clip_range_start": 0.25,
            "clip_range_end": 0.18,
            "entropy_coef_start": 0.02,
            "entropy_coef_hold": 0.02,
            "entropy_coef_end": 0.005,
            "target_kl": 0.015,
        }
    )

    return exps


def run_sweep() -> None:
    out_csv = os.path.join("results", "one_stage_two_players_sweep0_q55.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    experiments = make_experiments()

    for exp in experiments:
        exp_id = exp["id"]
        print("=" * 80)
        print(f"[SWEEP0] Running {exp_id}")

        cfg = deepcopy(base_config)

        # Fixed PPO and opponent settings for q=55.
        cfg["q_list"] = [55.0]
        cfg["seed"] = 42
        cfg["steps_per_update"] = 4096
        cfg["update_epochs"] = 6
        cfg["minibatch_size"] = 1024
        cfg["episodes"] = 3_000_000
        cfg["opponent_mode"] = "periodic"
        cfg["opponent_sync_interval"] = 2
        cfg["lag_warmup_updates"] = 10
        cfg["lag_fade_updates"] = 10
        cfg["opponent_history_sample_p"] = 0.0
        cfg["opponent_history_sample_p_end"] = 0.0

        # Apply sweep-specific overrides.
        for key, value in exp.items():
            if key == "id":
                continue
            cfg[key] = value

        rows = run_ppo(
            cfg,
            episodes=cfg["episodes"],
            train_qs=[55.0],
            eval_qs=[55.0],
            eval_symmetric=True,
            eval_vs_opponent=False,
            eval_vs_history=False,
        )

        if not rows:
            print(f"[SWEEP0] {exp_id} produced no rows, skipping.")
            continue

        row = rows[0]
        row["sweep_id"] = exp_id
        row.setdefault("abs_err", row.get("stage2_gap_unweighted", float("nan")))
        row["clip_range_start"] = cfg.get("clip_range_start")
        row["clip_range_end"] = cfg.get("clip_range_end")
        row["entropy_coef_start"] = cfg.get("entropy_coef_start")
        row["entropy_coef_end"] = cfg.get("entropy_coef_end")
        row["target_kl"] = cfg.get("target_kl")

        save_standardized_result(row, out_csv)

        print(
            f"[SWEEP0] {exp_id} done: abs_err={row['abs_err']:.4f}, "
            f"target_kl={row['target_kl']}, "
            f"clip_end={row['clip_range_end']}, "
            f"entropy_end={row['entropy_coef_end']}"
        )


if __name__ == "__main__":
    run_sweep()

