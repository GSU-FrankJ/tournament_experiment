config = {
    "k": 0.0004,
    "q": 25.0,
    "w_h": 6.5,
    "w_l": 3.0,
    "num_players": 3,
    "effort_range": [0, 100],
    "seed": 42
}

# Dynamically compute theoretical symmetric equilibrium effort for 3 players
# For N identical players, effort = (w_h - w_l) / ((N+1) * k * q)
config["effort"] = (config["w_h"] - config["w_l"]) / ((config["num_players"] + 1) * config["k"] * config["q"])
config["cost"] = config["k"] * config["effort"] ** 2
config["eu"] = round(((config["w_h"] + config["w_l"]) / 2 - config["k"] * config["effort"] ** 2), 2)
