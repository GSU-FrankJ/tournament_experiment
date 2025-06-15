import math
import os
import numpy as np

def asymmetric_ability_gradient_descent_solver(env, lr=0.1, steps=100000, eps=1e-3):
    """
    Gradient descent solver for asymmetric ability parameters.
    Each player optimizes their effort independently given others' efforts.
    
    Args:
        env: AsymmetricAbilityEnv implementing utility(player_id, effort, *other_efforts)
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
        efforts = [(low + high) / 2.0 for _ in range(num_players)]
    else:
        efforts = [50.0 for _ in range(num_players)]  # Default initialization
    
    print(f"Asymmetric ability gradient descent starting with efforts: {[round(e, 2) for e in efforts]}")
    print(f"Ability parameters: {env.get_ability_parameters()}")
    
    # Track convergence
    prev_efforts = efforts.copy()
    convergence_threshold = 1e-4
    patience = 1000
    no_improvement_count = 0
    
    for step in range(steps):
        # Update each player's effort using gradient descent
        for i in range(num_players):
            # Get current utility
            other_efforts = [efforts[j] for j in range(num_players) if j != i]
            current_utility, _ = env.utility(i, efforts[i], *other_efforts)
            
            # Compute gradient using finite differences
            # Forward difference: u(e + eps) - u(e)
            utility_plus, _ = env.utility(i, efforts[i] + eps, *other_efforts)
            gradient = (utility_plus - current_utility) / eps
            
            # Gradient ascent step (maximize utility)
            new_effort = efforts[i] + lr * gradient
            
            # Clamp to valid range
            if hasattr(env, "effort_range"):
                low, high = env.effort_range
                new_effort = max(low, min(high, new_effort))
            else:
                new_effort = max(0, new_effort)  # Effort must be non-negative
            
            efforts[i] = new_effort
        
        # Check convergence every 100 steps
        if step % 100 == 0:
            # Compute change in efforts
            effort_changes = [abs(efforts[i] - prev_efforts[i]) for i in range(num_players)]
            max_change = max(effort_changes)
            
            if step % 1000 == 0:
                print(f"Step {step}: efforts = {[round(e, 2) for e in efforts]}, max_change = {max_change:.6f}")
            
            # Check for convergence
            if max_change < convergence_threshold:
                no_improvement_count += 1
                if no_improvement_count >= patience // 100:
                    print(f"Asymmetric ability gradient descent converged at step {step}")
                    break
            else:
                no_improvement_count = 0
            
            prev_efforts = efforts.copy()
    
    # Compute final utilities and costs
    utilities_final = []
    costs_final = []
    
    for i in range(num_players):
        other_efforts = [efforts[j] for j in range(num_players) if j != i]
        utility, cost = env.utility(i, efforts[i], *other_efforts)
        utilities_final.append(utility)
        costs_final.append(cost)
    
    print(f"Final efforts: {[round(e, 2) for e in efforts]}")
    print(f"Final utilities: {[round(u, 3) for u in utilities_final]}")
    print(f"Final costs: {[round(c, 3) for c in costs_final]}")
    
    # Analyze equilibrium
    if hasattr(env, 'analyze_equilibrium'):
        analysis = env.analyze_equilibrium(efforts)
        print(f"Win probabilities: {[round(p, 3) for p in analysis['win_probabilities']]}")
        if analysis['gaps']:
            print(f"Gaps from theoretical: {[round(g, 3) for g in analysis['gaps']]}")
    
    return efforts, utilities_final, costs_final

def verify_equilibrium_conditions(env, efforts, eps=1e-3):
    """
    Verify that the given efforts satisfy the first-order conditions for Nash equilibrium.
    
    Args:
        env: AsymmetricAbilityEnv
        efforts: list of effort values
        eps: epsilon for numerical gradient computation
    
    Returns:
        dict with equilibrium analysis
    """
    num_players = len(efforts)
    gradients = []
    
    print("\n=== Equilibrium Verification ===")
    
    for i in range(num_players):
        other_efforts = [efforts[j] for j in range(num_players) if j != i]
        
        # Compute numerical gradient
        utility_current, _ = env.utility(i, efforts[i], *other_efforts)
        utility_plus, _ = env.utility(i, efforts[i] + eps, *other_efforts)
        gradient = (utility_plus - utility_current) / eps
        
        gradients.append(gradient)
        
        print(f"Player {i+1}: effort = {efforts[i]:.2f}, gradient = {gradient:.6f}")
    
    # Check if all gradients are close to zero (equilibrium condition)
    max_gradient = max(abs(g) for g in gradients)
    is_equilibrium = max_gradient < 0.01  # Tolerance for equilibrium
    
    print(f"Max gradient magnitude: {max_gradient:.6f}")
    print(f"Is equilibrium: {is_equilibrium}")
    
    return {
        "gradients": gradients,
        "max_gradient": max_gradient,
        "is_equilibrium": is_equilibrium,
        "efforts": efforts
    } 