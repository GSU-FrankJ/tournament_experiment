# Configuration for asymmetric ability parameters experiment
# Player 1 has higher ability (l1 > l2), equal cost parameters (k1 = k2)

config = {
    # Prizes mapped to CSV fields: stage1_weight = w_L, stage2_weight = w_H
    "stage1_weight": 3.0,
    "stage2_weight": 6.5,

    "l1": 1.2,      # Player 1 ability parameter (higher)
    "l2": 0.8,      # Player 2 ability parameter (lower)
    "k": 0.0004,    # Equal cost parameter for both players
    "k1": 0.0004,
    "k2": 0.0004,
    "q": 25.0,      # Noise parameter
    "q_list": [25.0, 40.0, 55.0],
    "w_h": 6.5,     # High reward
    "w_l": 3.0,     # Low reward
    "effort_range": [0, 200],
    "effort_bounds_stage1": [0, 100],
    "effort_bounds_stage2": [0, 200],
    "seed": 42,
    "num_players": 2,
    "information_revelation": "none",
}

# Compute theoretical asymmetric equilibrium efforts for different abilities
# For asymmetric abilities with equal costs, the Nash equilibrium efforts are:
# The player with higher ability (l1 > l2) will generally exert more effort
# because their effort is more effective in winning

# Theoretical analysis for asymmetric abilities:
# In a contest with abilities l_i, the effective effort is l_i * e_i
# Win probability depends on l_i * e_i vs l_j * e_j
# 
# For uniform noise model with abilities:
# P(player i wins) = P(l_i * e_i + ε_i > l_j * e_j + ε_j)
# 
# First-order condition: (w_h - w_l) * dP/de_i = 2 * k * e_i
# Where dP/de_i depends on the ability parameter l_i
#
# For the uniform noise case with abilities:
# dP/de_i ≈ l_i / (2 * q) (marginal effect of effort scaled by ability)
#
# Therefore: (w_h - w_l) * l_i / (2 * q) = 2 * k * e_i
# Solving: e_i* = (w_h - w_l) * l_i / (4 * k * q)

l1, l2 = config["l1"], config["l2"]
k = config["k"]
w_diff = config["w_h"] - config["w_l"]
q = config["q"]

# Theoretical optimal efforts (scaled by ability)
config["effort1"] = w_diff * l1 / (4 * k * q)
config["effort2"] = w_diff * l2 / (4 * k * q)

# Theoretical costs (same cost parameter k for both players)
config["cost1"] = k * config["effort1"]**2
config["cost2"] = k * config["effort2"]**2

# Store for easy access
config["theoretical_efforts"] = [config["effort1"], config["effort2"]]
config["theoretical_costs"] = [config["cost1"], config["cost2"]]

# Expected utilities at equilibrium
# EU_i = w_l + P_i(win) * (w_h - w_l) - k * e_i^2
# At symmetric equilibrium with abilities, P_1(win) ≠ P_2(win)
# Higher ability player should have higher win probability

# Approximate win probabilities (will be computed exactly in environment)
# For now, use simplified approximation
total_effective_effort = l1 * config["effort1"] + l2 * config["effort2"]
if total_effective_effort > 0:
    p1_win_approx = (l1 * config["effort1"]) / total_effective_effort
    p2_win_approx = (l2 * config["effort2"]) / total_effective_effort
else:
    p1_win_approx = p2_win_approx = 0.5

config["eu1"] = config["w_l"] + p1_win_approx * w_diff - config["cost1"]
config["eu2"] = config["w_l"] + p2_win_approx * w_diff - config["cost2"]

if __name__ == "__main__":
    print("Asymmetric Ability Configuration:")
    print(f"l1 = {config['l1']}, l2 = {config['l2']}")
    print(f"Theoretical efforts: e1* = {config['effort1']:.2f}, e2* = {config['effort2']:.2f}")
    print(f"Theoretical costs: c1* = {config['cost1']:.2f}, c2* = {config['cost2']:.2f}")
    print(f"Expected utilities: EU1* = {config['eu1']:.2f}, EU2* = {config['eu2']:.2f}")
    print(f"Win probabilities (approx): P1 = {p1_win_approx:.3f}, P2 = {p2_win_approx:.3f}") 