config = {
    "k1": 0.0002,  # Lower cost for player 1
    "k2": 0.0006,  # Higher cost for player 2
    "q": 25.0,
    "w_h": 6.5,
    "w_l": 3.0,
    "num_players": 2,
    "effort_range": [0, 100],
    "seed": 42
}

# For asymmetric cost parameters, equilibrium efforts are:
# Player 1: e1 = (w_h - w_l) * (2*k2 + k1*q) / (3*k1*k2*q + k1^2*q^2)
# Player 2: e2 = (w_h - w_l) * (2*k1 + k2*q) / (3*k1*k2*q + k2^2*q^2)
# Simplified approximation for small q:
k1, k2, q = config["k1"], config["k2"], config["q"]
w_diff = config["w_h"] - config["w_l"]

denominator = 3 * k1 * k2 * q
config["effort_1"] = w_diff * (2 * k2 + k1 * q) / (denominator + (k1 * q) ** 2)
config["effort_2"] = w_diff * (2 * k1 + k2 * q) / (denominator + (k2 * q) ** 2)

config["cost_1"] = k1 * config["effort_1"] ** 2
config["cost_2"] = k2 * config["effort_2"] ** 2
config["eu_1"] = round(((config["w_h"] + config["w_l"]) / 2 - config["cost_1"]), 2)
config["eu_2"] = round(((config["w_h"] + config["w_l"]) / 2 - config["cost_2"]), 2)
