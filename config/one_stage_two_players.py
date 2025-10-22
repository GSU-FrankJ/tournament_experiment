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
    "opponent_mode": "ema",
    "opponent_sync_interval": 10,
    "opponent_ema_tau": 0.05,
    "opponent_snapshot_keep": 10,
    "opponent_history_sample_p": 0.5,

    # PPO rollout & update (bandit-friendly defaults)
    "steps_per_update": 8192,
    "minibatch_size": 1024,
    "update_epochs": 6,
    "eval_every_updates": 20,
    "early_stop_abs_err": 1.0,
    "early_stop_patience": 5,
    "entropy_coef_start": 0.02,
    "entropy_coef_hold": 0.01,
    "entropy_coef_end": 0.002,
    "lr_start": 3e-4,
    "lr_end": 1e-4,
    "clip_range_start": 0.2,
    "clip_range_end": 0.1,
    "lag_warmup_updates": None,
    "lag_fade_updates": None,

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
