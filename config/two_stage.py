config = {
    "k": 0.0004,
    "q": 25.0,
    "w_h": 6.5,
    "w_l": 3.0,
    "num_players": 2,
    "num_stages": 2,
    "information_revelation": True,
    "stage1_weight": 0.5,  # Weight of stage 1 in total payoff
    "stage2_weight": 0.5,  # Weight of stage 2 in total payoff
    "effort_range": [0, 100],
    "seed": 42
}

# Two-stage game calculations
# Stage 1: Players choose effort with limited information
# Stage 2: Players observe stage 1 results and choose effort again
# For now, use symmetric equilibrium approximation for each stage
w_diff = config["w_h"] - config["w_l"]
k, q = config["k"], config["q"]

# Simplified approach: each stage treated as independent game with adjusted payoffs
config["stage1_effort"] = w_diff / (4 * k * q * (1 / config["stage1_weight"]))
config["stage2_effort"] = w_diff / (4 * k * q * (1 / config["stage2_weight"]))

config["stage1_cost"] = k * config["stage1_effort"] ** 2
config["stage2_cost"] = k * config["stage2_effort"] ** 2
config["total_cost"] = config["stage1_cost"] + config["stage2_cost"]

# Total expected utility across both stages
config["eu"] = round(((config["w_h"] + config["w_l"]) / 2 - config["total_cost"]), 2)