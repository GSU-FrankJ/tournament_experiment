# Configuration for asymmetric cost parameters experiment
# Player 1 has lower cost (k1 < k2), equal ability parameters (l1 = l2)

config = {
    "k1": 0.0003,  # Player 1 cost parameter (lower)
    "k2": 0.0005,  # Player 2 cost parameter (higher)
    "q": 25.0,     # Noise parameter
    "w_h": 6.5,    # High reward
    "w_l": 3.0,    # Low reward
    "effort_range": [0, 200],
    "seed": 42,
    "num_players": 2
}

# Compute theoretical asymmetric equilibrium efforts
# For asymmetric costs, the Nash equilibrium efforts are:
# e1* = (w_h - w_l) * k2 / (4 * q * (k1 + k2) * k1)
# e2* = (w_h - w_l) * k1 / (4 * q * (k1 + k2) * k2)

k1, k2 = config["k1"], config["k2"]
w_diff = config["w_h"] - config["w_l"]
q = config["q"]

config["effort1"] = w_diff * k2 / (4 * q * (k1 + k2) * k1)
config["effort2"] = w_diff * k1 / (4 * q * (k1 + k2) * k2)

config["cost1"] = k1 * config["effort1"] ** 2
config["cost2"] = k2 * config["effort2"] ** 2

# Expected utilities at equilibrium
# Need to compute win probabilities at equilibrium
e1_star = config["effort1"]
e2_star = config["effort2"]

# For uniform noise, P(e1 + ε1 > e2 + ε2) with ε ~ U(-q, q)
# This is a function of (e1 - e2), we'll compute it in the environment
# For now, store the theoretical efforts
config["theoretical_efforts"] = [e1_star, e2_star]
config["theoretical_costs"] = [config["cost1"], config["cost2"]]

print(f"Asymmetric Cost Configuration:")
print(f"k1 = {k1}, k2 = {k2}")
print(f"Theoretical efforts: e1* = {e1_star:.2f}, e2* = {e2_star:.2f}")
print(f"Theoretical costs: c1* = {config['cost1']:.2f}, c2* = {config['cost2']:.2f}") 