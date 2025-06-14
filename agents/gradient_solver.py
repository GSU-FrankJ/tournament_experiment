import math
import os

def gradient_descent_solver(env, lr=0.1, steps=100000, eps=1e-3):
    """
    Gradient descent with central difference to solve for symmetric effort.
    Works for both 2-player and 3-player games.
    Args:
        env: environment implementing utility(e1, *other_efforts)
        lr: learning rate
        steps: number of iterations
        eps: small epsilon for finite-difference gradient
    Returns:
        e_final: converged effort value
        final_u: utility at symmetric equilibrium
        final_cost: cost at e_final
    """
    # Initialize effort at midpoint of range if env provides range, else 1.0
    if hasattr(env, "effort_range"):
        low, high = env.effort_range
        e = (low + high) / 2.0
    else:
        e = 1.0

    # Determine number of players
    num_players = getattr(env, 'num_players', 2)
    
    log_path = f"/Users/fengjiang/Documents/GSU/tournament_experiment/results/logs/gradient_log_{num_players}p.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f_log:
        f_log.write("Step,Effort,Gradient,Utility\n")

    for step in range(steps):
        # For symmetric equilibrium, all players choose the same effort
        # Compute gradient by perturbing one player's effort while others stay at e
        if num_players == 2:
            # For 2 players: compute gradient of utility(e+eps, e) vs utility(e-eps, e)
            u_plus, _ = env.utility(e + eps, e)
            u_minus, _ = env.utility(e - eps, e)
        elif num_players == 3:
            # For 3 players: compute gradient of utility(e+eps, e, e) vs utility(e-eps, e, e)
            u_plus, _ = env.utility(e + eps, e, e)
            u_minus, _ = env.utility(e - eps, e, e)
        else:
            # General case: all other players at effort e
            other_efforts = [e] * (num_players - 1)
            u_plus, _ = env.utility(e + eps, *other_efforts)
            u_minus, _ = env.utility(e - eps, *other_efforts)
        
        # Finite difference gradient
        grad = (u_plus - u_minus) / (2 * eps)
        
        # Update e and clamp to valid range
        e += lr * grad
        if hasattr(env, "effort_range"):
            low, high = env.effort_range
        else:
            low, high = 0.0, 100.0
        e = min(max(e, low), high)

        # Log current state
        with open(log_path, "a") as f_log:
            if num_players == 2:
                current_u, _ = env.utility(e, e)
            elif num_players == 3:
                current_u, _ = env.utility(e, e, e)
            else:
                other_efforts = [e] * (num_players - 1)
                current_u, _ = env.utility(e, *other_efforts)
            f_log.write(f"{step},{e:.6f},{grad:.6f},{current_u:.6f}\n")

    # Compute final utility and cost at symmetric equilibrium
    if num_players == 2:
        final_u, final_cost = env.utility(e, e)
    elif num_players == 3:
        final_u, final_cost = env.utility(e, e, e)
    else:
        other_efforts = [e] * (num_players - 1)
        final_u, final_cost = env.utility(e, *other_efforts)
    
    return e, final_u, final_cost

class GradientSolver:
    """Wrapper class for gradient descent solver"""
    
    def __init__(self, env, config):
        self.env = env
        self.learning_rate = config.get('learning_rate', 0.01)
        self.max_iterations = config.get('max_iterations', 10000)
    
    def solve(self):
        """Solve using gradient descent"""
        effort, utility, cost = gradient_descent_solver(
            self.env, 
            lr=self.learning_rate, 
            steps=self.max_iterations
        )
        return effort