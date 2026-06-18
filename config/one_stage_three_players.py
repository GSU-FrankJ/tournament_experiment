# One-Stage Three-Player PPO Configuration
# Pure self-play only (no opponent modes) - all three players share the same policy.
# Theoretical symmetric equilibrium: e* = (w_h - w_l) / (4 * k * q)

config = {
    # Core game parameters (identical competitors)
    "k": 0.001,
    "q": 35.0,
    "q_list": [35.0, 55.0],
    "w_h": 6.5,
    "w_l": 3.0,
    "effort_range": [0, 100],
    "seed": 42,
    "num_players": 3,

    # Gradient method settings (one-stage 3-player, uniform noise)
    "gradient_lr": 5.0,
    "gradient_steps": 5000,
    "gradient_delta": 0.01,
    "gradient_lr_decay": 1.0,
    "gradient_tol": 1e-6,
    "gradient_num_samples": 64,
    "gradient_init_perturb": 1.0,

    # PPO rollout & update (bandit-friendly defaults, self-play only)
    "steps_per_update": 4096,
    "minibatch_size": 1024,
    "update_epochs": 6,
    "episodes": 6_144_000,    # 1500 updates at 4096 steps/update (3x budget for q=25 convergence)
    "max_updates": 1500,
    "eval_every_updates": 20,

    # Learning rate schedule
    "lr_start": 3e-4,
    "lr_end": 2e-4,

    # Entropy schedule (wider exploration that decays)
    "entropy_coef_start": 0.03,
    "entropy_coef_hold": 0.03,
    "entropy_coef_end": 0.005,

    # Clip range schedule
    "clip_range_start": 0.50,
    "clip_range_end": 0.35,

    # Target KL for adaptive updates
    "target_kl": 0.08,
    "kl_clip_factor_up": 1.2,    # Softer than default 1.5 to reduce late-training oscillation
    "kl_clip_factor_down": 0.8,  # Softer than default 0.7
    "kl_lr_factor_up": 1.2,
    "kl_lr_factor_down": 0.8,

    # Convergence / early-stop (ON by default for PPO with relaxed profile)
    "convergence": {
        "enabled": True,
        "cheap_gate_profile": "relaxed",
        "eval_every_updates": 20,
        "cheap_gate_profiles": {
            "default": {
                "window_size": 20,
                "mean_kl_thresh": 0.0045,
                "std_kl_thresh": 0.0035,
                "drift_effort_thresh": 2.0,
                "patience_drift": 2,
            },
            "conservative": {
                "window_size": 20,
                "mean_kl_thresh": 0.0038,
                "std_kl_thresh": 0.0030,
                "drift_effort_thresh": 1.5,
                "patience_drift": 3,
            },
            "aggressive": {
                "window_size": 20,
                "mean_kl_thresh": 0.0060,
                "std_kl_thresh": 0.0075,
                "drift_effort_thresh": 5.5,
                "patience_drift": 1,
            },
            "relaxed": {
                "window_size": 20,
                "mean_kl_thresh": 0.015,      # Relaxed to 1.5% for theory-align-v2 KL fluctuation
                "std_kl_thresh": 0.012,
                "drift_effort_thresh": 8.0,   # More lenient drift tolerance
                "patience_drift": 1,
            },
        },
        "cheap_gate": {
            "window_size": 20,
            "mean_kl_thresh": 0.0045,
            "std_kl_thresh": 0.0035,
            "drift_effort_thresh": 2.0,
            "patience_drift": 2,
        },
        "exploit": {
            "exploit_eps": 0.03,  # unified across all scenarios (audit decision)
            "patience_exploit": 5,
            "M": 16384,
            "grid": {
                "stage_a_step": 5.0,
                "stage_b_radius": 15.0,
                "stage_b_step": 1.0,
                "stage_c_radius": 3.0,
                "stage_c_step": 0.25,
            },
        },
        "symmetry": {
            "symmetry_gap_thresh": 0.5,
            "symmetry_fail_patience": 3,
        },
    },

    # Opponent lag (disabled by default: sync_interval=0 means pure self-play)
    "opponent_mode": "periodic",
    "opponent_sync_interval": 0,
    "opponent_ema_tau": 0.20,
    "opponent_snapshot_keep": 10,
    "opponent_history_sample_p": 0.3,
    "lag_warmup_updates": 0,
    "lag_fade_updates": None,

    # KL early-stop (defaults keep behaviour OFF)
    "kl_early_stop": True,
    "kl_stop_patience": 1,
    "kl_stop_threshold": None,
    "ratio_stop_threshold": None,

    # Plotting and evaluation
    "enable_overlay": True,
    "convergence_rel_err_threshold": 0.10,
}

# Theoretical symmetric equilibrium effort per user spec:
# e* = (w_H - w_L) / (4 * q * k) -- same formula as two-player
config["effort"] = (config["w_h"] - config["w_l"]) / (4 * config["k"] * config["q"])  # 3p-1stage
config["cost"] = config["k"] * config["effort"] ** 2
# Expected utility at equilibrium: w_L + (1/3)(w_H - w_L) - cost (each player wins with prob 1/3)
config["eu"] = round((config["w_l"] + (1.0/3.0) * (config["w_h"] - config["w_l"])) - config["cost"], 4)

