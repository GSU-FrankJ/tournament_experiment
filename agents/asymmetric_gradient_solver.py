import math
import os
import numpy as np

def asymmetric_gradient_descent_solver(env, lr=0.1, steps=100000, eps=1e-3):
    """
    Gradient descent solver for asymmetric cost parameters.
    Each player optimizes their effort independently given others' efforts.
    
    Args:
        env: AsymmetricCostEnv implementing utility(player_id, effort, *other_efforts)
        lr: learning rate
        steps: number of iterations
        eps: small epsilon for finite-difference gradient
    Returns:
        efforts_final: list of converged effort values for each player
        utilities_final: list of utilities at equilibrium
        costs_final: list of costs at equilibrium
    """
    num_players = env.num_players
    
    # Initialize efforts at midpoint of range
    if hasattr(env, "effort_range"):
        low, high = env.effort_range
        efforts = [(low + high) / 2.0] * num_players
    else:
        efforts = [1.0] * num_players

    log_path = f"/Users/fengjiang/Documents/GSU/tournament_experiment/results/logs/asymmetric_gradient_log.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f_log:
        header = "Step," + ",".join([f"Effort{i+1}" for i in range(num_players)]) + "," + \
                ",".join([f"Gradient{i+1}" for i in range(num_players)]) + "," + \
                ",".join([f"Utility{i+1}" for i in range(num_players)]) + "\n"
        f_log.write(header)

    for step in range(steps):
        # Store old efforts for convergence check
        old_efforts = efforts.copy()
        
        # Update each player's effort sequentially (Gauss-Seidel style)
        for player_id in range(num_players):
            # Get other players' current efforts
            other_efforts = [efforts[j] for j in range(num_players) if j != player_id]
            
            # Compute gradient for this player
            u_plus, _ = env.utility(player_id, efforts[player_id] + eps, *other_efforts)
            u_minus, _ = env.utility(player_id, efforts[player_id] - eps, *other_efforts)
            grad = (u_plus - u_minus) / (2 * eps)
            
            # Update this player's effort
            efforts[player_id] += lr * grad
            
            # Clamp to valid range
            if hasattr(env, "effort_range"):
                low, high = env.effort_range
            else:
                low, high = 0.0, 100.0
            efforts[player_id] = min(max(efforts[player_id], low), high)

        # Log current state every 1000 steps
        if step % 1000 == 0:
            with open(log_path, "a") as f_log:
                # Compute current utilities and gradients for logging
                utilities = []
                gradients = []
                for player_id in range(num_players):
                    other_efforts = [efforts[j] for j in range(num_players) if j != player_id]
                    u, _ = env.utility(player_id, efforts[player_id], *other_efforts)
                    utilities.append(u)
                    
                    # Compute gradient for logging
                    u_plus, _ = env.utility(player_id, efforts[player_id] + eps, *other_efforts)
                    u_minus, _ = env.utility(player_id, efforts[player_id] - eps, *other_efforts)
                    grad = (u_plus - u_minus) / (2 * eps)
                    gradients.append(grad)
                
                log_line = f"{step}," + ",".join([f"{e:.6f}" for e in efforts]) + "," + \
                          ",".join([f"{g:.6f}" for g in gradients]) + "," + \
                          ",".join([f"{u:.6f}" for u in utilities]) + "\n"
                f_log.write(log_line)
        
        # Check for convergence
        if step > 1000:
            max_change = max(abs(efforts[i] - old_efforts[i]) for i in range(num_players))
            if max_change < 1e-6:
                print(f"Asymmetric gradient descent converged at step {step}")
                break

    # Compute final utilities and costs
    utilities_final = []
    costs_final = []
    for player_id in range(num_players):
        other_efforts = [efforts[j] for j in range(num_players) if j != player_id]
        u, cost = env.utility(player_id, efforts[player_id], *other_efforts)
        utilities_final.append(u)
        costs_final.append(cost)
    
    return efforts, utilities_final, costs_final

class AsymmetricGradientSolver:
    """Wrapper class for asymmetric gradient descent solver"""
    
    def __init__(self, env, config):
        self.env = env
        self.learning_rate = config.get('learning_rate', 0.01)
        self.max_iterations = config.get('max_iterations', 10000)
    
    def solve(self):
        """Solve using asymmetric gradient descent"""
        efforts, utilities, costs = asymmetric_gradient_descent_solver(
            self.env, 
            lr=self.learning_rate, 
            steps=self.max_iterations
        )
        return efforts, utilities, costs 