#!/usr/bin/env python3
import os
import sys
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.one_stage_two_players import config as base_config
from run.run_two_players import run_ppo
from utils.logger import save_standardized_result


def make_experiments():
    exps = []

    # Sweep1-A: high KL, wide clip, modest entropy
    exps.append({
        "id": "sweep1_A_kl0.05",
        "clip_range_start": 0.35,
        "clip_range_end": 0.25,
        "entropy_coef_start": 0.02,
        "entropy_coef_hold": 0.02,
        "entropy_coef_end": 0.008,
        "target_kl": 0.05,
    })

    # Sweep1-B: even higher KL and clip, same entropy
    exps.append({
        "id": "sweep1_B_kl0.06",
        "clip_range_start": 0.40,
        "clip_range_end": 0.28,
        "entropy_coef_start": 0.02,
        "entropy_coef_hold": 0.02,
        "entropy_coef_end": 0.008,
        "target_kl": 0.06,
    })

    # Sweep1-C: higher exploration + KL 0.06
    exps.append({
        "id": "sweep1_C_explore_kl0.06",
        "clip_range_start": 0.40,
        "clip_range_end": 0.28,
        "entropy_coef_start": 0.03,
        "entropy_coef_hold": 0.03,
        "entropy_coef_end": 0.010,
        "target_kl": 0.06,
    })

    # Sweep1-D: KL 0.07, very wide clip, more exploration
    exps.append({
        "id": "sweep1_D_kl0.07",
        "clip_range_start": 0.45,
        "clip_range_end": 0.30,
        "entropy_coef_start": 0.03,
        "entropy_coef_hold": 0.03,
        "entropy_coef_end": 0.012,
        "target_kl": 0.07,
    })

    # Sweep1-E: KL 0.08, max stress test
    exps.append({
        "id": "sweep1_E_kl0.08",
        "clip_range_start": 0.50,
        "clip_range_end": 0.35,
        "entropy_coef_start": 0.03,
        "entropy_coef_hold": 0.03,
        "entropy_coef_end": 0.015,
        "target_kl": 0.08,
    })

    # Sweep1-F: aggressive but slightly safer baseline
    exps.append({
        "id": "sweep1_F_safe_aggressive",
        "clip_range_start": 0.35,
        "clip_range_end": 0.25,
        "entropy_coef_start": 0.025,
        "entropy_coef_hold": 0.025,
        "entropy_coef_end": 0.009,
        "target_kl": 0.05,
    })

    return exps


def run_sweep():
    out_csv = os.path.join("results", "one_stage_two_players_sweep1_q55.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    experiments = make_experiments()
    q_target = 55.0
    seed = 42

    for exp in experiments:
        exp_id = exp["id"]
        print("=" * 80)
        print(f"[SWEEP1] Running {exp_id}")

        cfg = deepcopy(base_config)

        # Fix q, seed, PPO structural defaults
        cfg["q_list"] = [q_target]
        cfg["seed"] = seed
        cfg["steps_per_update"] = int(cfg.get("steps_per_update", 4096))
        cfg["update_epochs"] = int(cfg.get("update_epochs", 6))
        cfg["minibatch_size"] = int(cfg.get("minibatch_size", 1024))
        cfg["episodes"] = int(cfg.get("episodes", 3_000_000))

        # Opponent / lag settings
        cfg["opponent_mode"] = "periodic"
        cfg["opponent_sync_interval"] = 2
        cfg["lag_warmup_updates"] = 10
        cfg["lag_fade_updates"] = 10
        cfg["opponent_history_sample_p"] = 0.0
        cfg["opponent_history_sample_p_end"] = 0.0

        # KL controller knobs for aggressive search
        cfg["clip_floor"] = 0.10
        cfg["clip_ceiling"] = 0.60
        cfg["min_lr"] = 5e-5
        cfg["max_lr"] = 8e-4
        cfg["kl_low"] = 0.5 * exp["target_kl"]
        cfg["kl_high"] = 3.0 * exp["target_kl"]
        cfg["kl_clip_factor_up"] = 1.6
        cfg["kl_clip_factor_down"] = 0.7
        cfg["kl_lr_factor_up"] = 1.6
        cfg["kl_lr_factor_down"] = 0.7
        cfg["warm_decay_ratio"] = 0.7
        cfg["force_kl_gate"] = True

        # Apply sweep-specific hyperparameters
        for k, v in exp.items():
            if k != "id":
                cfg[k] = v

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
            print(f"[SWEEP1] {exp_id} produced no rows, skipping.")
            continue

        row = rows[0]
        row["sweep_id"] = exp_id
        # Ensure we record key knobs
        row.setdefault("abs_err", row.get("stage2_gap_unweighted", float("nan")))
        row["clip_range_start"] = cfg.get("clip_range_start")
        row["clip_range_end"] = cfg.get("clip_range_end")
        row["entropy_coef_start"] = cfg.get("entropy_coef_start")
        row["entropy_coef_end"] = cfg.get("entropy_coef_end")
        row["target_kl"] = cfg.get("target_kl")
        row["kl_low"] = cfg.get("kl_low")
        row["kl_high"] = cfg.get("kl_high")

        save_standardized_result(row, out_csv)

        print(
            f"[SWEEP1] {exp_id} done: "
            f"abs_err={row['abs_err']:.4f}, "
            f"target_kl={row['target_kl']}, "
            f"clip_end={row['clip_range_end']}, "
            f"entropy_end={row['entropy_coef_end']}"
        )


if __name__ == "__main__":
    run_sweep()
