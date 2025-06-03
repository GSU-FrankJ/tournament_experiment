import math
import os

def gradient_descent_solver(env, lr=0.1, steps=100000, eps=1e-3):
    """
    Gradient descent with central difference to solve for symmetric effort.
    Args:
        env: environment implementing utility(e1, e2)
        lr: learning rate
        steps: number of iterations
        eps: small epsilon for finite-difference gradient
    Returns:
        e_final: converged effort value
        final_u: utility at (e_final, e_final)
        final_cost: cost at e_final
    """
    # Initialize effort at midpoint of range if env provides range, else 1.0
    if hasattr(env, "effort_range"):
        low, high = env.effort_range
        e = (low + high) / 2.0
    else:
        e = 1.0

    log_path = "/Users/fengjiang/Documents/GSU/tournament_experiment/results/logs/gradient_log.txt"
    with open(log_path, "w") as f_log:
        f_log.write("Step,Effort,Gradient,Utility\n")

    for _ in range(steps):
        # Compute utility at e + eps and e - eps (keeping opponent's effort = e)
        u_plus, _ = env.utility(e + eps, e)
        u_minus, _ = env.utility(e - eps, e)
        # Finite difference gradient
        grad = (u_plus - u_minus) / (2 * eps)
        # Update e and clamp to valid range [low, high] if available, else [0, 100]
        e += lr * grad
        if hasattr(env, "effort_range"):
            low, high = env.effort_range
        else:
            low, high = 0.0, 100.0
        e = min(max(e, low), high)

        with open(log_path, "a") as f_log:
            current_u, _ = env.utility(e, e)
            f_log.write(f"{_},{e:.6f},{grad:.6f},{current_u:.6f}\n")

    final_u, final_cost = env.utility(e, e)
    return e, final_u, final_cost