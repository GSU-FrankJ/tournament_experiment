# Configuration for two-stage game environment
# Stage 1: Initial effort decisions
# Stage 2: Second-round decisions based on Stage 1 outcomes

config = {
    # Stage weights: equal contribution of both stages to final payoff
    # If you prefer absolute prize scaling, change both to the same positive value.
    "stage1_weight": 0.5,
    "stage2_weight": 0.5,

    # Basic game parameters
    "k": 0.0004,        # Default cost (symmetric)
    "k1": 0.0004,       # Stage 1 cost parameter
    "k2": 0.0004,       # Stage 2 cost parameter
    "q": 25.0,          # Noise parameter
    "q_list": [25.0, 40.0, 55.0],
    "w_h": 6.5,         # High prize
    "w_l": 3.0,         # Low prize
    "effort_range": [0, 200],
    "effort_bounds_stage1": [0, 100],
    "effort_bounds_stage2": [0, 200],
    "seed": 42,
    "num_players": 2,

    # Information structure
    "information_revelation": "partial",  # "none", "partial", "full"
    "stage1_noise_factor": 1.0,
    "stage2_noise_factor": 1.0,
    "reveal_opponent_effort": True,
    "reveal_stage1_outcome": True,
    "reveal_noise_realization": False,
    # Monte Carlo samples for total-output probability in Stage 2
    "mc_total_samples": 16384,

    # Plotting and evaluation
    "enable_overlay": True,
    "convergence_rel_err_threshold": 0.10,
}

# Compute theoretical equilibrium efforts for both stages
# This is a simplified calculation - actual equilibrium requires backward induction
stage1_effort_base = (config["w_h"] - config["w_l"]) / (6 * config["k1"] * config["q"])  # spec
stage2_effort_base = (config["w_h"] - config["w_l"]) / (6 * config["k2"] * config["q"])  # spec

# Baseline theoretical efforts per stage (unweighted)
config["stage1_effort"] = stage1_effort_base
config["stage2_effort"] = stage2_effort_base

# Total expected costs and utilities
config["stage1_cost"] = config["k1"] * config["stage1_effort"] ** 2
config["stage2_cost"] = config["k2"] * config["stage2_effort"] ** 2
config["total_cost"] = config["stage1_cost"] + config["stage2_cost"]

# Expected utility (simplified - actual calculation requires full backward induction)
config["expected_utility"] = (config["w_h"] + config["w_l"]) / 2 - config["total_cost"]

if __name__ == "__main__":
    print("Two-Stage Game Configuration:")
    print(f"Stage 1 theoretical effort: {config['stage1_effort']:.2f}")
    print(f"Stage 2 theoretical effort: {config['stage2_effort']:.2f}")
    print(f"Total expected cost: {config['total_cost']:.4f}")
    print(f"Expected utility: {config['expected_utility']:.2f}")
    print(f"Information revelation: {config['information_revelation']}")
    print(f"Stage weights: {config['stage1_weight']:.1f} / {config['stage2_weight']:.1f}") 
