#!/usr/bin/env python3
"""
Aggressive PPO hyperparameter sweep for one-stage two-player q=55 (seed=42).

Targets runs where the prior sweep under-shot the equilibrium due to an
overly conservative KL controller / clip tail. The grid below focuses on
higher KL ceilings, wider late clips, and slightly elevated entropy floors.
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

    exps.append(
        {
            "id": "A_mild",
            "clip_range_start": 0.25,
            "clip_range_end": 0.18,
            "entropy_coef_start": 0.02,
            "entropy_coef_hold": 0.02,
            "entropy_coef_end": 0.005,
            "target_kl": 0.015,
        }
    )

    exps.append(
        {
            "id": "B_kl_high",
            "clip_range_start": 0.25,
            "clip_range_end": 0.20,
            "entropy_coef_start": 0.02,
            "entropy_coef_hold": 0.02,
            "entropy_coef_end": 0.005,
            "target_kl": 0.020,
        }
    )

    exps.append(
        {
            "id": "C_explore_more",
            "clip_range_start": 0.25,
            "clip_range_end": 0.20,
            "entropy_coef_start": 0.03,
            "entropy_coef_hold": 0.03,
            "entropy_coef_end": 0.006,
            "target_kl": 0.015,
        }
    )

    exps.append(
        {
            "id": "D_very_aggressive",
            "clip_range_start": 0.30,
            "clip_range_end": 0.22,
            "entropy_coef_start": 0.03,
            "entropy_coef_hold": 0.03,
            "entropy_coef_end": 0.008,
            "target_kl": 0.025,
        }
    )

    exps.append(
        {
            "id": "E_wide_clip_safe_kl",
            "clip_range_start": 0.28,
            "clip_range_end": 0.20,
            "entropy_coef_start": 0.02,
            "entropy_coef_hold": 0.02,
            "entropy_coef_end": 0.005,
            "target_kl": 0.015,
        }
    )

    exps.append(
        {
            "id": "F_baseline_plus",
            "clip_range_start": 0.25,
            "clip_range_end": 0.16,
            "entropy_coef_start": 0.02,
            "entropy_coef_hold": 0.02,
            "entropy_coef_end": 0.004,
            "target_kl": 0.012,
        }
    )

    return exps


def run_sweep() -> None:
    out_csv = os.path.join("results", "one_stage_two_players_sweep_q55_aggressive.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    q_target = 55.0
    seed = 42
    target_episodes = 3_000_000
    experiments = make_experiments()

    for exp in experiments:
        exp_id = exp["id"]
        print("=" * 80)
        print(f"[SWEEP] Running aggressive config id={exp_id}")

        cfg = deepcopy(base_config)
        cfg["q_list"] = [q_target]
        cfg["seed"] = seed
        cfg["steps_per_update"] = 4096
        cfg["update_epochs"] = 6
        cfg["minibatch_size"] = 1024
        cfg["episodes"] = target_episodes
        cfg["opponent_mode"] = "periodic"
        cfg["opponent_sync_interval"] = 2
        cfg["lag_warmup_updates"] = 10
        cfg["lag_fade_updates"] = 10
        cfg["opponent_history_sample_p"] = 0.0
        cfg["opponent_history_sample_p_end"] = 0.0

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
        row["clip_range_start"] = cfg["clip_range_start"]
        row["clip_range_end"] = cfg["clip_range_end"]
        row["entropy_coef_start"] = cfg["entropy_coef_start"]
        row["entropy_coef_hold"] = cfg["entropy_coef_hold"]
        row["entropy_coef_end"] = cfg["entropy_coef_end"]
        row["target_kl"] = cfg["target_kl"]

        save_standardized_result(row, out_csv)

        print(
            "[SWEEP] id={id} done. abs_err={err:.4f}, "
            "clip_end={clip_end}, entropy_end={entropy_end}, target_kl={target}".format(
                id=exp_id,
                err=float(row.get("abs_err", float("nan"))),
                clip_end=cfg["clip_range_end"],
                entropy_end=cfg["entropy_coef_end"],
                target=cfg["target_kl"],
            )
        )


if __name__ == "__main__":
    run_sweep()
