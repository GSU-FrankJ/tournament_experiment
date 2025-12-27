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

    # Opponent (lag) settings – periodic sync every 2 updates, short warmup/fade,
    # no opponent-history sampling (与 Sweep1_E 一致，保持频繁同步、短期滞后)
    "opponent_mode": "periodic",
    "opponent_sync_interval": 2,
    "opponent_ema_tau": 0.20,
    "opponent_snapshot_keep": 10,
    "opponent_history_sample_p": 0.0,

    # PPO rollout & update (bandit-friendly defaults)
    "steps_per_update": 4096,
    "minibatch_size": 1024,
    "update_epochs": 6,
    "episodes": 2_048_000,    # Default sweep budget: exactly 500 updates at 4096 steps/update (可用 CLI 覆盖)
    "max_updates": 500,
    "eval_every_updates": 20,
    "early_stop_abs_err": 0.8,
    "early_stop_patience": 6,
    # Sweep1_E-inspired entropy schedule: wider exploration that decays to 0.015 (更激进但稳定)
    "entropy_coef_start": 0.03,
    "entropy_coef_hold": 0.03,
    "entropy_coef_end": 0.015,
    "lr_start": 3e-4,
    "lr_end": 2e-4,
    "clip_range_start": 0.50, # Wide clips from Sweep1_E best run
    "clip_range_end": 0.35,   # Tail clip remains wide to avoid collapsing updates
    "target_kl": 0.08,        # Assumes kl_low = 0.5 * target, kl_high = 3 * target inside run loop
    "lag_warmup_updates": 10,
    "lag_fade_updates": 10,
    "opponent_history_sample_p_end": 0.0,

    # Convergence / early-stop (OFF by default to preserve baseline behavior)
    "convergence": {
        "enabled": False,
        "eval_every_updates": 20,
        "cheap_gate": {
            "window_size": 20,
            "mean_kl_thresh": 0.003,
            "std_kl_thresh": 0.001,
            "drift_effort_thresh": 0.5,
            "patience_drift": 3,
        },
        "exploit": {
            "exploit_eps": 0.05,
            "patience_exploit": 5,
            "M": 8192,
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
