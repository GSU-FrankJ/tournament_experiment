# Configuration for asymmetric cost parameters experiment
# Player 1 has lower cost (k1 < k2), equal ability parameters (l1 = l2)
# Based on user requirements: k1=0.0004, k2=0.00055

config = {
    # Prizes mapped to CSV fields: stage1_weight = w_L, stage2_weight = w_H
    "stage1_weight": 5.5,
    "stage2_weight": 8.0,

    # Cost asymmetry
    "k1": 0.0004,  # Player 1 cost parameter (lower)
    "k2": 0.00055, # Player 2 cost parameter (higher)
    "k": 0.0004,   # default (for overlays)

    # Noise and bounds
    "q": 25.0,     # Noise parameter (will be varied in experiment)
    "q_list": [25.0, 40.0, 55.0],
    "w_h": 8.0,    # High reward
    "w_l": 5.5,    # Low reward
    "effort_range": [0, 200],
    "effort_bounds_stage1": [0, 100],
    "effort_bounds_stage2": [0, 200],
    "seed": 42,
    "num_players": 2,
    "information_revelation": "none",
}

def calculate_theoretical_efforts(q_value, k1, k2, w_h, w_l):
    """
    Calculate theoretical asymmetric equilibrium efforts for given parameters
    
    For asymmetric costs, the Nash equilibrium efforts are:
    e1* = (w_h - w_l) * k2 / (4 * q * (k1 + k2) * k1)
    e2* = (w_h - w_l) * k1 / (4 * q * (k1 + k2) * k2)
    """
    w_diff = w_h - w_l
    denominator = 4 * q_value * (k1 + k2)
    
    effort1 = w_diff * k2 / (denominator * k1)
    effort2 = w_diff * k1 / (denominator * k2)
    
    cost1 = k1 * effort1 ** 2
    cost2 = k2 * effort2 ** 2
    
    return effort1, effort2, cost1, cost2

# Compute theoretical values for default q
k1, k2 = config["k1"], config["k2"]
w_h, w_l = config["w_h"], config["w_l"]
q = config["q"]

effort1, effort2, cost1, cost2 = calculate_theoretical_efforts(q, k1, k2, w_h, w_l)

config["effort1"] = effort1
config["effort2"] = effort2
config["cost1"] = cost1
config["cost2"] = cost2
config["theoretical_efforts"] = [effort1, effort2]
config["theoretical_costs"] = [cost1, cost2]

# Test configurations for multiple q values (matching user's table)
test_configs = []
for q_val in [25.0, 40.0, 55.0]:
    e1, e2, c1, c2 = calculate_theoretical_efforts(q_val, k1, k2, w_h, w_l)
    test_config = {
        "q": q_val,
        "k1": k1,
        "k2": k2,
        "w_h": w_h,
        "w_l": w_l,
        "effort_range": [0, 200],
        "theoretical_effort1": e1,
        "theoretical_effort2": e2,
        "theoretical_cost1": c1,
        "theoretical_cost2": c2,
        "seed": 42
    }
    test_configs.append(test_config)

config["test_configs"] = test_configs

print(f"Asymmetric Cost Configuration:")
print(f"k1 = {k1}, k2 = {k2}")
print(f"w_h = {w_h}, w_l = {w_l}")
print(f"Default q = {q}")
print(f"Theoretical efforts: e1* = {effort1:.2f}, e2* = {effort2:.2f}")
print(f"Theoretical costs: c1* = {cost1:.2f}, c2* = {cost2:.2f}")
print(f"Test configurations prepared for q values: {[tc['q'] for tc in test_configs]}") 