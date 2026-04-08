# One-Stage Two-Player Different Ability Configuration (l1 > l2, k1 = k2)
# 
# Implements experiment type "III.2.c Players with Different Abilities"
# where output is y_i = e_i + l_i + ε_i with l1 > l2 (additive ability model).
#
# Theoretical equilibrium (symmetric):
#   e* = ((2q - (l1 - l2)) * (w_H - w_L)) / (8kq²)
#   Both players exert the same effort; player 1 wins more often due to ability.
# 
# Default parameters: l1=10, l2=5, k=0.0005, w_h=6.5, w_l=3.0

from __future__ import annotations

from typing import Dict, Tuple

# === Default ability parameters ===
DEFAULT_L1 = 10.0  # Player 1 ability (higher)
DEFAULT_L2 = 5.0   # Player 2 ability (lower)

config = {
    # === Ability parameters ===
    "l1": DEFAULT_L1,   # Player 1 ability parameter (higher)
    "l2": DEFAULT_L2,   # Player 2 ability parameter (lower)
    
    # === Cost parameters (equal for both players) ===
    "k": 0.0005,         # Cost coefficient
    "k1": 0.0005,        # Player 1 cost (= k)
    "k2": 0.0005,        # Player 2 cost (= k)
    
    # === Game parameters ===
    "w_h": 6.5,         # High prize (winner)
    "w_l": 3.0,         # Low prize (loser)
    "q": 35.0,           # Default noise parameter
    "q_list": [35.0, 55.0],  # Q values to sweep
    "effort_range": [0, 100],
    "effort_bounds_stage2": [0, 100],
    "seed": 42,
    "num_players": 2,
    
    # === Gradient descent parameters ===
    # Note: Higher learning rate (5.0) needed for fast convergence
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
            "exploit_eps": 0.05,
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


def calculate_theoretical_effort_different_ability(
    q: float,
    k: float,
    l1: float,
    l2: float,
    w_h: float,
    w_l: float,
) -> float:
    """
    Calculate symmetric equilibrium effort for additive ability model.
    
    Model: y_i = e_i + l_i + ε_i, where ε_i ~ U(-q, q)
    
    The symmetric equilibrium effort (both players) is:
        e* = ((2q - (l1 - l2)) * (w_H - w_L)) / (8kq²)
    
    Args:
        q: Noise parameter
        k: Cost coefficient (same for both players)
        l1: Player 1 ability (higher)
        l2: Player 2 ability (lower)
        w_h: High prize (winner)
        w_l: Low prize (loser)
        
    Returns:
        e*: Symmetric equilibrium effort
    """
    w_diff = float(w_h - w_l)
    delta_l = float(l1 - l2)
    
    # Closed-form solution (bound at 0 to avoid negative effort)
    e_star = ((2.0 * q - delta_l) * w_diff) / (8.0 * k * (q ** 2))
    return max(0.0, e_star)


def calculate_win_probability_different_ability(
    e1: float,
    e2: float,
    l1: float,
    l2: float,
    q: float,
) -> float:
    """
    Calculate P(player 1 wins) for additive ability model.
    
    Player 1 wins if: e1 + l1 + ε1 > e2 + l2 + ε2
    Let d = (e2 + l2) - (e1 + l1), then ε1 - ε2 ~ Tri(-2q, 2q)
    
    Args:
        e1: Player 1 effort
        e2: Player 2 effort  
        l1: Player 1 ability
        l2: Player 2 ability
        q: Noise parameter
        
    Returns:
        P(player 1 wins)
    """
    d = (e2 + l2) - (e1 + l1)
    
    if d <= -2 * q:
        return 1.0
    elif d >= 2 * q:
        return 0.0
    elif d < 0:
        return 1.0 - ((d + 2 * q) ** 2) / (8 * q ** 2)
    else:
        return ((2 * q - d) ** 2) / (8 * q ** 2)


def get_theoretical_efforts(
    q: float,
    l1: float = None,
    l2: float = None,
    k: float = None,
    w_h: float = None,
    w_l: float = None,
) -> Tuple[float, float]:
    """
    Compute equilibrium efforts for different ability scenario.
    
    Note: Both players exert the same effort at equilibrium.
    
    Args:
        q: Noise parameter
        l1: Player 1 ability (default from config)
        l2: Player 2 ability (default from config)
        k: Cost coefficient (default from config)
        w_h: High prize (default from config)
        w_l: Low prize (default from config)
        
    Returns:
        (e1_star, e2_star): Equilibrium efforts (equal in this model)
    """
    # Use config defaults if not specified
    l1 = l1 if l1 is not None else config["l1"]
    l2 = l2 if l2 is not None else config["l2"]
    k = k if k is not None else config["k"]
    w_h = w_h if w_h is not None else config["w_h"]
    w_l = w_l if w_l is not None else config["w_l"]
    
    e_star = calculate_theoretical_effort_different_ability(q, k, l1, l2, w_h, w_l)
    return e_star, e_star  # Symmetric equilibrium


def get_theoretical_utilities(
    q: float,
    l1: float = None,
    l2: float = None,
    k: float = None,
    w_h: float = None,
    w_l: float = None,
) -> Tuple[float, float]:
    """
    Compute per-player expected utilities at equilibrium.
    
    Args:
        q: Noise parameter
        l1: Player 1 ability (default from config)
        l2: Player 2 ability (default from config)
        k: Cost coefficient (default from config)
        w_h: High prize (default from config)
        w_l: Low prize (default from config)
        
    Returns:
        (eu1, eu2): Expected utilities at equilibrium
    """
    # Use config defaults if not specified
    l1 = l1 if l1 is not None else config["l1"]
    l2 = l2 if l2 is not None else config["l2"]
    k = k if k is not None else config["k"]
    w_h = w_h if w_h is not None else config["w_h"]
    w_l = w_l if w_l is not None else config["w_l"]
    
    e_star = calculate_theoretical_effort_different_ability(q, k, l1, l2, w_h, w_l)
    cost = k * (e_star ** 2)
    
    # Win probability at equilibrium
    p1_win = calculate_win_probability_different_ability(e_star, e_star, l1, l2, q)
    p2_win = 1.0 - p1_win
    
    w_diff = w_h - w_l
    eu1 = w_l + p1_win * w_diff - cost
    eu2 = w_l + p2_win * w_diff - cost
    
    return eu1, eu2


def get_win_probability_at_equilibrium(
    q: float,
    l1: float = None,
    l2: float = None,
) -> float:
    """
    Compute player 1's win probability at equilibrium.
    
    Args:
        q: Noise parameter
        l1: Player 1 ability (default from config)
        l2: Player 2 ability (default from config)
        
    Returns:
        P(player 1 wins) at equilibrium
    """
    l1 = l1 if l1 is not None else config["l1"]
    l2 = l2 if l2 is not None else config["l2"]
    k = config["k"]
    w_h = config["w_h"]
    w_l = config["w_l"]
    
    e_star = calculate_theoretical_effort_different_ability(q, k, l1, l2, w_h, w_l)
    return calculate_win_probability_different_ability(e_star, e_star, l1, l2, q)


# === Pre-compute theoretical values for default parameters ===

l1, l2 = config["l1"], config["l2"]
k = config["k"]
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
    p1_win = get_win_probability_at_equilibrium(q_val)
    config["theoretical_by_q"][q_val] = {
        "effort1": e1,
        "effort2": e2,
        "eu1": eu1,
        "eu2": eu2,
        "p1_win": p1_win,
    }


# === Print summary when module is run directly ===
if __name__ == "__main__":
    print("=" * 60)
    print("Different Ability Configuration Summary")
    print("=" * 60)
    print(f"Ability parameters: l1={l1}, l2={l2} (Δl={l1-l2})")
    print(f"Cost parameter: k={k}")
    print(f"Prizes: w_h={w_h}, w_l={w_l}")
    print(f"Effort range: {config['effort_range']}")
    print()
    print("Theoretical Equilibrium (symmetric effort, asymmetric win probability):")
    print("-" * 60)
    for q_val in config["q_list"]:
        data = config["theoretical_by_q"][q_val]
        print(
            f"  q={q_val:5.1f}: e*={data['effort1']:7.3f}, "
            f"P(p1 wins)={data['p1_win']:.4f}, "
            f"EU1={data['eu1']:.4f}, EU2={data['eu2']:.4f}"
        )
    print()
    print("(Both players exert same effort; player 1 wins more due to ability advantage)")
