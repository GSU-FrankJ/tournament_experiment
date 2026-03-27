# One-Stage Two-Player PPO Baseline (2025-XX)
# Derived from Sweep1_E_kl0.08 (q=55, seed=42) as the current best-so-far:
# - abs_err(q=55) ≈ 0.93 with stable KL/entropy behaviour.
# 后续 sweep 请以此配置为基准做局部搜索，不要退回更保守的旧默认值。

config = {
    # Core parameters
    "k": 0.0004,
    "k1": 0.0004,
    "k2": 0.0004,
    "q": 40.0,
    "q_list": [25.0, 40.0, 55.0],
    "w_h": 6.5,
    "w_l": 3.0,
    "stage1_weight": 3.0,   # Map to w_L for standardized CSV
    "stage2_weight": 6.5,   # Map to w_H for standardized CSV
    "effort_range": [0, 200],
    "effort_bounds_stage1": [0, 100],
    "effort_bounds_stage2": [0, 200],
    "seed": 42,

    # Gradient (one-stage 2p, uniform noise)
    "gradient_lr": 0.08,
    "gradient_steps": 1500,
    "gradient_delta": 0.5,
    "gradient_tol": 1e-4,
    "gradient_num_samples": 64,
    "gradient_init_perturb": 1.0,

    # Opponent (lag) settings – VESTIGIAL: the agent code maintains a lagged
    # opponent network, but selfplay rollout mode never calls act_opponent(),
    # so these values have no effect on training. Kept at 0/disabled to avoid
    # misleading readers into thinking lag is active.
    "opponent_mode": "periodic",
    "opponent_sync_interval": 0,
    "opponent_ema_tau": 0.0,
    "opponent_snapshot_keep": 0,
    "opponent_history_sample_p": 0.0,

    # PPO rollout & update (bandit-friendly defaults)
    "steps_per_update": 4096,
    "minibatch_size": 1024,
    "update_epochs": 6,
    "episodes": 6_144_000,    # 1500 updates at 4096 steps/update (3x budget for q=25 convergence)
    "max_updates": 1500,
    "eval_every_updates": 20,
    # Sweep1_E-inspired entropy schedule: wider exploration that decays to 0.015 (更激进但稳定)
    "entropy_coef_start": 0.03,
    "entropy_coef_hold": 0.03,
    "entropy_coef_end": 0.005,
    "lr_start": 3e-4,
    "lr_end": 2e-4,
    "clip_range_start": 0.50, # Wide clips from Sweep1_E best run
    "clip_range_end": 0.35,   # Tail clip remains wide to avoid collapsing updates
    "target_kl": 0.08,        # Assumes kl_low = 0.5 * target, kl_high = 3 * target inside run loop
    "kl_clip_factor_up": 1.2,    # Softer than default 1.5 to reduce late-training oscillation
    "kl_clip_factor_down": 0.8,  # Softer than default 0.7
    "kl_lr_factor_up": 1.2,
    "kl_lr_factor_down": 0.8,
    "lag_warmup_updates": 10,
    "lag_fade_updates": 10,
    "opponent_history_sample_p_end": 0.0,

    # Convergence / early-stop (ON by default for PPO with relaxed profile)
    "convergence": {
        "enabled": True,
        "cheap_gate_profile": "relaxed",
        "eval_every_updates": 10,     # matches actual training runs (10, not 20)
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
                "mean_kl_thresh": 0.015,      # 放宽到1.5% - 适合theory-align-v2的KL波动
                "std_kl_thresh": 0.012,       # 放宽相应
                "drift_effort_thresh": 8.0,   # 更宽松的drift容忍
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
            "exploit_eps": 0.03,       # matches actual training runs (0.03, not 0.05)
            "patience_exploit": 5,
            "M": 8192,                 # matches actual training runs (8192, not 16384)
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
    # KL early-stop (defaults keep behaviour OFF)
    "kl_early_stop": True,
    "kl_stop_patience": 1,
    "kl_stop_threshold": None,
    "ratio_stop_threshold": None,

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
