# One-Stage Two-Player Different Cost Configuration (k1 < k2, l1 = l2)
# 
# Implements experiment type "III.2.b Two Players with Different Cost Functions"
# where cost functions are C_i(e) = k_i * e^2 with k1 < k2.
#
# Theoretical equilibrium:
#   e1* = 2 k2 q (w_H - w_L) / (8 k1 k2 q^2 - (k1 - k2)(w_H - w_L))
#   e2* = 2 k1 q (w_H - w_L) / (8 k1 k2 q^2 - (k1 - k2)(w_H - w_L))
# 
# Player 1 (lower cost k1) exerts more effort in equilibrium.

config = {
    # === Asymmetric cost parameters ===
    "k1": 0.0004,       # Player 1 cost parameter (lower)
    "k2": 0.00055,      # Player 2 cost parameter (higher)
    
    # === Game parameters ===
    "w_h": 8.0,          # High prize (winner)
    "w_l": 5.5,          # Low prize (loser)
    "q": 35.0,           # Default noise parameter
    "q_list": [35.0, 55.0],  # Q values to sweep
    "effort_range": [0, 100],
    "effort_bounds_stage2": [0, 100],
    "seed": 42,
    "num_players": 2,
    
    # === Gradient descent parameters ===
    "gradient_lr": 5.0,
    "gradient_steps": 5000,
    "gradient_delta": 0.01,
    "gradient_lr_decay": 1.0,
    "gradient_tol": 1e-6,
    "gradient_num_samples": 64,
    "gradient_init_perturb": 1.0,
    
    # === PPO hyperparameters (same as symmetric baseline) ===
    "steps_per_update": 4096,
    "minibatch_size": 1024,
    "update_epochs": 6,
    "episodes": 6_144_000,    # 1500 updates at 4096 steps/update (3x budget for q=25 convergence)
    "max_updates": 1500,
    "eval_every_updates": 20,
    
    # Entropy schedule: wider exploration that decays
    "entropy_coef_start": 0.03,
    "entropy_coef_hold": 0.03,
    "entropy_coef_end": 0.005,
    "entropy_hold_fraction": 2.0 / 3.0,
    
    # Learning rate schedule
    "lr_start": 3e-4,
    "lr_end": 2e-4,
    
    # Clip range schedule
    "clip_range_start": 0.50,
    "clip_range_end": 0.35,
    
    # KL target and early stop
    "target_kl": 0.08,
    "kl_clip_factor_up": 1.2,    # Softer than default 1.5 to reduce late-training oscillation
    "kl_clip_factor_down": 0.8,  # Softer than default 0.7
    "kl_lr_factor_up": 1.2,
    "kl_lr_factor_down": 0.8,
    "kl_early_stop": True,
    "kl_stop_patience": 1,
    "kl_stop_threshold": None,
    "ratio_stop_threshold": None,
    
    # === Convergence / early-stop settings ===
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
                "mean_kl_thresh": 0.015,
                "std_kl_thresh": 0.012,
                "drift_effort_thresh": 8.0,
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
            "exploit_eps": 0.03,  # unified across all scenarios (audit decision; Round-4 runs already used 0.03 via CLI)
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
    },
    
    # Convergence threshold for determining "converged"
    "convergence_rel_err_threshold": 0.10,
    
    # Information revelation (none for this experiment)
    "information_revelation": "none",
}


def get_theoretical_efforts(q: float, k1: float = None, k2: float = None, 
                            w_h: float = None, w_l: float = None) -> tuple:
    """
    Compute per-player equilibrium efforts for asymmetric cost parameters.
    
    Args:
        q: Noise parameter
        k1: Player 1 cost (default from config)
        k2: Player 2 cost (default from config)
        w_h: High prize (default from config)
        w_l: Low prize (default from config)
        
    Returns:
        (e1_star, e2_star): Equilibrium efforts for each player
    """
    from utils.theory import e_star_two_players_asymmetric_cost
    
    # Use config defaults if not specified
    k1 = k1 if k1 is not None else config["k1"]
    k2 = k2 if k2 is not None else config["k2"]
    w_h = w_h if w_h is not None else config["w_h"]
    w_l = w_l if w_l is not None else config["w_l"]
    
    return e_star_two_players_asymmetric_cost(q, w_h, w_l, k1, k2)


def get_theoretical_utilities(q: float, k1: float = None, k2: float = None,
                              w_h: float = None, w_l: float = None) -> tuple:
    """
    Compute per-player expected utilities at equilibrium.
    
    Args:
        q: Noise parameter
        k1: Player 1 cost (default from config)
        k2: Player 2 cost (default from config)
        w_h: High prize (default from config)
        w_l: Low prize (default from config)
        
    Returns:
        (eu1, eu2): Expected utilities at equilibrium for each player
    """
    from utils.theory import eu_two_players_asymmetric_cost
    
    # Use config defaults if not specified
    k1 = k1 if k1 is not None else config["k1"]
    k2 = k2 if k2 is not None else config["k2"]
    w_h = w_h if w_h is not None else config["w_h"]
    w_l = w_l if w_l is not None else config["w_l"]
    
    return eu_two_players_asymmetric_cost(q, w_h, w_l, k1, k2)


# === Pre-compute theoretical values for default parameters ===
# These are stored in config for quick reference

k1, k2 = config["k1"], config["k2"]
w_h, w_l = config["w_h"], config["w_l"]

# Compute for default q
q_default = config["q"]
e1_default, e2_default = get_theoretical_efforts(q_default)
config["theoretical_effort1"] = e1_default
config["theoretical_effort2"] = e2_default
config["theoretical_efforts"] = [e1_default, e2_default]

# Compute for all q values in q_list
config["theoretical_by_q"] = {}
for q_val in config["q_list"]:
    e1, e2 = get_theoretical_efforts(q_val)
    eu1, eu2 = get_theoretical_utilities(q_val)
    config["theoretical_by_q"][q_val] = {
        "effort1": e1,
        "effort2": e2,
        "eu1": eu1,
        "eu2": eu2,
    }


# === Print summary when module is imported ===
if __name__ == "__main__":
    print("=" * 60)
    print("Different Cost Configuration Summary")
    print("=" * 60)
    print(f"Cost parameters: k1={k1}, k2={k2}")
    print(f"Prizes: w_h={w_h}, w_l={w_l}")
    print(f"Effort range: {config['effort_range']}")
    print()
    print("Theoretical Equilibrium Efforts:")
    print("-" * 40)
    for q_val in config["q_list"]:
        data = config["theoretical_by_q"][q_val]
        print(f"  q={q_val:5.1f}: e1*={data['effort1']:7.3f}, e2*={data['effort2']:7.3f}")
    print()
    print("(Player 1 with lower cost exerts more effort)")
