config = {
    # Prizes (weights in CSV): stage1_weight = w_L, stage2_weight = w_H
    "stage1_weight": 3.0,
    "stage2_weight": 6.5,

    # Core parameters
    "k": 0.0004,
    "k1": 0.0004,
    "k2": 0.0004,
    "q": 40.0,
    "q_list": [25.0, 40.0, 55.0],
    "w_h": 6.5,
    "w_l": 3.0,
    "effort_range": [0, 200],
    "effort_bounds_stage1": [0, 100],
    "effort_bounds_stage2": [0, 200],
    "seed": 42,

    # Opponent (lag) settings
    "opponent_mode": "periodic",
    "opponent_sync_interval": 2,
    "opponent_ema_tau": 0.20,
    "opponent_snapshot_keep": 10,
    "opponent_history_sample_p": 0.0,

    # PPO rollout & update (bandit-friendly defaults)
    "steps_per_update": 4096,
    "minibatch_size": 1024,
    "update_epochs": 6,
    "episodes": 2048000,      # ≈500 updates × 4096 steps/update
    "max_updates": 500,
    "eval_every_updates": 20,
    "early_stop_abs_err": 0.8,
    "early_stop_patience": 6,
    "entropy_coef_start": 0.02,
    "entropy_coef_hold": 0.02,
    "entropy_coef_end": 0.008, # 略高于默认水平，尾段仍保留少许探索
    "lr_start": 3e-4,
    "lr_end": 2e-4,
    "clip_range_start": 0.25,
    "clip_range_end": 0.18,   # 由 0.16 再放宽到 0.18，后期允许更大步幅
    "target_kl": 0.015,       # 略放松 KL 约束，容许更大胆更新
    "lag_warmup_updates": 10,
    "lag_fade_updates": 10,
    "opponent_history_sample_p_end": 0.0,

    # Plotting and evaluation
    "enable_overlay": True,
    "convergence_rel_err_threshold": 0.10,
    "information_revelation": "none",
}

# Dynamically compute theoretical symmetric equilibrium effort (legacy fields)
# NOTE: For overlays, use utils.theory.e_star with denominator 6 per spec.
# Two-player single-stage uses denominator 4
config["effort"] = (config["w_h"] - config["w_l"]) / (4 * config["k"] * config["q"])  # 2p-1stage
config["cost"] = config["k"] * config["effort"] ** 2
config["eu"] = round(((config["w_h"] + config["w_l"]) / 2 - config["k"] * config["effort"] ** 2), 2)
