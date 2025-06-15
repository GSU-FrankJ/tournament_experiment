# Configuration for two-stage game environment
# Stage 1: Initial effort decisions
# Stage 2: Second-round decisions based on Stage 1 outcomes

config = {
    # Basic game parameters
    "k": 0.0004,        # Cost parameter
    "q": 25.0,          # Noise parameter
    "w_h": 6.5,         # High reward
    "w_l": 3.0,         # Low reward
    "effort_range": [0, 200],
    "seed": 42,
    "num_players": 2,
    
    # Two-stage specific parameters
    "stage1_weight": 0.6,    # Weight of Stage 1 in final outcome
    "stage2_weight": 0.4,    # Weight of Stage 2 in final outcome
    "information_revelation": "partial",  # "none", "partial", "full"
    "stage1_noise_factor": 1.0,  # Noise multiplier for Stage 1
    "stage2_noise_factor": 0.8,  # Noise multiplier for Stage 2 (often lower)
    
    # Information revelation settings
    "reveal_opponent_effort": False,     # Whether to reveal opponent's Stage 1 effort
    "reveal_stage1_outcome": True,       # Whether to reveal Stage 1 winner
    "reveal_noise_realization": False,   # Whether to reveal actual noise values
    
    # Stage-specific cost parameters (can be different)
    "k1": 0.0004,       # Stage 1 cost parameter
    "k2": 0.0005,       # Stage 2 cost parameter (slightly higher)
}

# Compute theoretical equilibrium efforts for both stages
# This is a simplified calculation - actual equilibrium requires backward induction
stage1_effort_base = (config["w_h"] - config["w_l"]) / (4 * config["k1"] * config["q"])
stage2_effort_base = (config["w_h"] - config["w_l"]) / (4 * config["k2"] * config["q"])

# Adjust for stage weights and inter-stage effects
config["stage1_effort"] = stage1_effort_base * config["stage1_weight"]
config["stage2_effort"] = stage2_effort_base * config["stage2_weight"]

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