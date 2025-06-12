import numpy as np

def calculate_three_player_equilibrium(k, q, w_h, w_l):
    """
    Calculate theoretical symmetric equilibrium for three-player tournament.
    Based on experiment plan: one winner, two losers structure.
    
    For symmetric equilibrium with uniform noise U(-q, q):
    - Each player chooses the same effort e*
    - Win probability ≈ 1/3 (symmetric case)
    - First-order condition: (w_h - w_l) * dP/de = 2*k*e
    
    For three identical players, the marginal effect of effort on win probability
    can be approximated using the density of the uniform distribution.
    """
    
    # For symmetric 3-player case with uniform noise
    # Theoretical win probability = 1/3
    p_win_symmetric = 1.0 / 3.0
    
    # The marginal effect of effort on win probability depends on the noise distribution
    # For uniform U(-q, q), the effective "marginal density" is related to 1/(2q)
    # In multi-player contests, this becomes more complex
    
    # Using contest theory approximation for uniform distribution:
    # For n players: dP/de ≈ (n-1) / (n * 2q) in symmetric equilibrium
    n = 3
    dp_de_approx = (n - 1) / (n * 2 * q)
    
    # First-order condition: (w_h - w_l) * dP/de = 2*k*e
    # Solving: e* = (w_h - w_l) * dP/de / (2*k)
    e_star = (w_h - w_l) * dp_de_approx / (2 * k)
    
    # Calculate equilibrium values
    cost_star = k * e_star ** 2
    eu_star = w_l + p_win_symmetric * (w_h - w_l) - cost_star
    
    return e_star, cost_star, eu_star, p_win_symmetric

def verify_equilibrium_numerically(e, k, q, w_h, w_l):
    """
    Verify the equilibrium by checking if marginal utility is close to zero.
    """
    def prob_win_at_effort(my_effort, others_effort, q, num_samples=30000):
        """Calculate win probability when I choose my_effort and others choose others_effort"""
        np.random.seed(42)
        
        eps_me = np.random.uniform(-q, q, num_samples)
        eps_2 = np.random.uniform(-q, q, num_samples)
        eps_3 = np.random.uniform(-q, q, num_samples)
        
        my_total = my_effort + eps_me
        other2_total = others_effort + eps_2
        other3_total = others_effort + eps_3
        
        wins = (my_total > other2_total) & (my_total > other3_total)
        return np.mean(wins)
    
    # Check marginal utility at proposed equilibrium
    eps = 0.1
    prob_plus = prob_win_at_effort(e + eps, e, q)
    prob_minus = prob_win_at_effort(e - eps, e, q)
    
    marginal_prob = (prob_plus - prob_minus) / (2 * eps)
    marginal_utility = (w_h - w_l) * marginal_prob - 2 * k * e
    
    prob_current = prob_win_at_effort(e, e, q)
    
    return marginal_utility, prob_current

# Configuration for three identical players based on theoretical formulas
# Formula from user:
# effort = (w_h - w_l) / (4 * k * q)
# cost = k * effort^2
# EU = (w_h + w_l + w_l) / 3 - k * effort^2

config = {
    "k": 0.0004,
    "q": 25.0,
    "w_h": 6.5,
    "w_l": 3.0,
    "num_players": 3,
    "effort_range": [0, 100],
    "seed": 42
}

# Calculate theoretical values using the correct formulas for 3 players
effort_theory = (config["w_h"] - config["w_l"]) / (4 * config["k"] * config["q"])
cost_theory = config["k"] * effort_theory ** 2
eu_theory = ((config["w_h"] + config["w_l"] + config["w_l"]) / 3) - cost_theory

# Add calculated values to config
config["effort"] = round(effort_theory, 2)
config["cost"] = round(cost_theory, 2)
config["eu"] = round(eu_theory, 2)

print(f"Three-player theoretical values (corrected formulas):")
print(f"  Effort: {config['effort']}")
print(f"  Cost: {config['cost']}")
print(f"  Expected Utility: {config['eu']}")
print(f"  Theoretical win probability: 1/3 = 0.333")
