config = {
    # Prizes (weights in CSV): map one-stage to stage-2 fields
    "stage1_weight": 3.0,
    "stage2_weight": 6.5,

    # Core parameters (identical competitors)
    "k": 0.0004,
    "q": 40.0,
    "q_list": [25.0, 40.0, 55.0],
    "w_h": 6.5,
    "w_l": 3.0,
    "effort_range": [0, 200],
    "effort_bounds_stage2": [0, 200],
    "seed": 42,
    "num_players": 3,
}

# Theoretical symmetric equilibrium effort per user spec:
# e* = (w_H - w_L) / (4 * q * k)
config["effort"] = (config["w_h"] - config["w_l"]) / (4 * config["k"] * config["q"])  # 3p-1stage per requirement
config["cost"] = config["k"] * config["effort"] ** 2
config["eu"] = round((config["w_l"] + (1.0/3.0) * (config["w_h"] - config["w_l"])) - config["cost"], 4)

