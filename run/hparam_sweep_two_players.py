#!/usr/bin/env python3
"""
Hyperparameter sweep for One-Stage Two-Player PPO focused on q=55 (seed=42).

The sweep explores a small grid over late-phase PPO hyperparameters:
    - clip_range_end (tight vs relaxed tail updates)
    - entropy_coef_end (residual exploration)
    - target_kl (KL controller aggressiveness)

All other settings stay aligned with config.one_stage_two_players /
run_two_players.py. Each configuration trains only on q=55 and appends a row to
results/one_stage_two_players_sweep_q55.csv summarizing the outcome.
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


def make_experiments() -> List[Dict[str, float]]:
    """Define the sweep grid."""
    experiments: List[Dict[str, float]] = []

    experiments.append(
        {
            "id": "A_baseline_soft",
            "clip_range_start": 0.25,
            "clip_range_end": 0.16,
            "entropy_coef_start": 0.02,
            "entropy_coef_hold": 0.02,
            "entropy_coef_end": 0.004,
            "target_kl": 0.010,
        }
    )

    experiments.append(
        {
            "id": "B_more_explore",
            "clip_range_start": 0.25,
            "clip_range_end": 0.18,
            "entropy_coef_start": 0.03,
            "entropy_coef_hold": 0.03,
            "entropy_coef_end": 0.006,
            "target_kl": 0.015,
        }
    )

    experiments.append(
        {
            "id": "C_conservative",
            "clip_range_start": 0.22,
            "clip_range_end": 0.14,
            "entropy_coef_start": 0.02,
            "entropy_coef_hold": 0.02,
            "entropy_coef_end": 0.002,
            "target_kl": 0.010,
        }
    )

    experiments.append(
        {
            "id": "D_clip_wide",
            "clip_range_start": 0.25,
            "clip_range_end": 0.18,
            "entropy_coef_start": 0.02,
            "entropy_coef_hold": 0.02,
            "entropy_coef_end": 0.005,
            "target_kl": 0.010,
        }
    )

    return experiments


def run_sweep() -> None:
    out_csv = os.path.join("results", "one_stage_two_players_sweep_q55.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    q_target = 55.0
    seed = 42
    experiments = make_experiments()

    for exp in experiments:
        exp_id = exp["id"]
        print("=" * 80)
        print(f"[SWEEP] Running config id={exp_id}")

        cfg = deepcopy(base_config)
        cfg["q_list"] = [q_target]
        cfg["seed"] = seed
        cfg["steps_per_update"] = int(cfg.get("steps_per_update", 4096))
        cfg["update_epochs"] = int(cfg.get("update_epochs", 6))
        cfg["minibatch_size"] = int(cfg.get("minibatch_size", 1024))
        cfg["episodes"] = int(cfg.get("episodes", 3_000_000))
        cfg["opponent_mode"] = cfg.get("opponent_mode", "periodic")
        cfg["opponent_sync_interval"] = int(cfg.get("opponent_sync_interval", 2))
        cfg["lag_warmup_updates"] = int(cfg.get("lag_warmup_updates", 10))
        cfg["lag_fade_updates"] = int(cfg.get("lag_fade_updates", 10))
        cfg["opponent_history_sample_p"] = float(cfg.get("opponent_history_sample_p", 0.0))
        cfg["opponent_history_sample_p_end"] = float(cfg.get("opponent_history_sample_p_end", 0.0))

        for key, value in exp.items():
            if key == "id":
                continue
            cfg[key] = value

        rows = run_ppo(
            cfg,
            episodes=cfg["episodes"],
            train_qs=[q_target],
            eval_qs=[q_target],
            eval_symmetric=True,
            eval_vs_opponent=False,
            eval_vs_history=False,
        )

        if not rows:
            print(f"[SWEEP] id={exp_id} produced no rows, skipping.")
            continue

        row = rows[0]
        row["sweep_id"] = exp_id
        row["seed"] = seed
        row["q"] = q_target
        row.setdefault("abs_err", row.get("stage2_gap_unweighted", float("nan")))
        row["target_kl"] = cfg.get("target_kl")
        row["clip_range_start"] = cfg.get("clip_range_start")
        row["clip_range_end"] = cfg.get("clip_range_end")
        row["entropy_coef_start"] = cfg.get("entropy_coef_start")
        row["entropy_coef_end"] = cfg.get("entropy_coef_end")

        save_standardized_result(row, out_csv)

        print(
            f"[SWEEP] id={exp_id} done. "
            f"abs_err={row['abs_err']:.4f}, "
            f"clip_end={cfg['clip_range_end']}, "
            f"entropy_end={cfg['entropy_coef_end']}, "
            f"target_kl={cfg['target_kl']}"
        )


if __name__ == "__main__":
    run_sweep()
