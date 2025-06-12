config = {
    "k": 0.0004,
    "q": 40.0,
    "w_h": 6.5,
    "w_l": 3.0,
    "effort_range": [0, 200],
    "seed": 42
}

# Dynamically compute theoretical symmetric equilibrium effort
config["effort"] = (config["w_h"] - config["w_l"]) / (4 * config["k"] * config["q"])
config["cost"] = config["k"] * config["effort"] ** 2
config["eu"] = round(((config["w_h"] + config["w_l"]) / 2 - config["k"] * config["effort"] ** 2), 2)