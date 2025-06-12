config = {
    "k": 0.0004,  # Same cost for both players
    "q": 25.0,
    "w_h1": 7.0,  # Higher ability for player 1
    "w_l1": 3.5,
    "w_h2": 6.0,  # Lower ability for player 2  
    "w_l2": 2.5,
    "num_players": 2,
    "effort_range": [0, 100],
    "seed": 42
}

# For asymmetric ability parameters with same cost:
# Player i maximizes: (w_hi + w_li)/2 * P_i - k * e_i^2
# Where P_i depends on relative efforts
# Simplified Nash equilibrium for different abilities:
k, q = config["k"], config["q"]
w_diff_1 = config["w_h1"] - config["w_l1"]
w_diff_2 = config["w_h2"] - config["w_l2"]

# Symmetric approximation adjusted for ability differences
config["effort_1"] = w_diff_1 / (3 * k * q)
config["effort_2"] = w_diff_2 / (3 * k * q)

config["cost_1"] = k * config["effort_1"] ** 2
config["cost_2"] = k * config["effort_2"] ** 2
config["eu_1"] = round(((config["w_h1"] + config["w_l1"]) / 2 - config["cost_1"]), 2)
config["eu_2"] = round(((config["w_h2"] + config["w_l2"]) / 2 - config["cost_2"]), 2)