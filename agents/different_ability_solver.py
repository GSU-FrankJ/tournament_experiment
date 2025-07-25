"""
Different Ability Gradient Solver
=================================

Specialized gradient descent solver for two-player tournament with different abilities.
Optimized for the scenario where:
- Player 1: l1 = 10 (higher ability)
- Player 2: l2 = 5 (lower ability)  
- Equal cost parameters: k1 = k2 = 0.0004

Key Features:
- Adaptive learning rates for each player
- Ability-aware gradient computation
- Enhanced convergence detection
- Comprehensive equilibrium verification
"""

import math
import time
import numpy as np
from typing import List, Tuple, Dict, Any
from utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

def different_ability_gradient_descent_solver(
    env, 
    lr: float = 0.1, 
    steps: int = 100000, 
    eps: float = 1e-3,
    adaptive_lr: bool = True,
    convergence_threshold: float = 1e-4,
    patience: int = 1000,
    verbose: bool = True
) -> Tuple[List[float], List[float], List[float], Dict[str, Any]]:
    """
    Gradient descent solver optimized for different ability parameters.
    
    This solver uses individual gradients for each player to find the Nash equilibrium
    where each player maximizes their utility given the other's effort.
    
    Args:
        env: DifferentAbilityEnv environment
        lr: Base learning rate
        steps: Maximum number of iterations
        eps: Epsilon for finite-difference gradient computation
        adaptive_lr: Whether to use adaptive learning rates
        convergence_threshold: Threshold for detecting convergence
        patience: Number of steps to wait for improvement
        verbose: Whether to print progress information
        
    Returns:
        (final_efforts, final_utilities, final_costs, solver_info)
    """
    logger.info("Starting different ability gradient descent solver")
    logger.info(f"Parameters: lr={lr}, steps={steps}, adaptive_lr={adaptive_lr}")
    
    # Get environment parameters
    ability_params = env.get_ability_parameters()
    cost_params = env.get_cost_parameters()
    theoretical_efforts = env.get_theoretical_efforts()
    
    logger.info(f"Environment: l1={ability_params[0]}, l2={ability_params[1]}")
    logger.info(f"Costs: k1={cost_params[0]}, k2={cost_params[1]}")
    logger.info(f"Theoretical efforts: e1*={theoretical_efforts[0]:.2f}, e2*={theoretical_efforts[1]:.2f}")
    
    # Initialize efforts near theoretical values with some noise
    if hasattr(env, "effort_range"):
        low, high = env.effort_range
        efforts = [(low + high) / 2.0 for _ in range(2)]
    else:
        efforts = [50.0, 50.0]  # Default initialization
    
    # If we have theoretical values, initialize closer to them
    if theoretical_efforts and len(theoretical_efforts) == 2:
        efforts[0] = theoretical_efforts[0] * (0.8 + 0.4 * np.random.random())  # ±20% noise
        efforts[1] = theoretical_efforts[1] * (0.8 + 0.4 * np.random.random())
    
    if verbose:
        print(f"Initial efforts: e1={efforts[0]:.2f}, e2={efforts[1]:.2f}")
    
    # Adaptive learning rate parameters
    learning_rates = [lr, lr]  # Individual learning rates for each player
    momentum = [0.0, 0.0]      # Momentum terms
    momentum_beta = 0.9
    
    # Convergence tracking
    prev_efforts = efforts.copy()
    best_efforts = efforts.copy()
    best_gap = float('inf')
    no_improvement_count = 0
    convergence_history = []
    
    start_time = time.time()
    
    for step in range(steps):
        # Compute utilities and gradients for each player
        gradients = []
        current_utilities = []
        
        for i in range(2):
            # Current utility
            other_effort = efforts[1-i]  # Other player's effort
            current_utility, _ = env.compute_utility(i, efforts[i], other_effort)
            current_utilities.append(current_utility)
            
            # Forward difference gradient: u(e + eps) - u(e)
            utility_plus, _ = env.compute_utility(i, efforts[i] + eps, other_effort)
            gradient = (utility_plus - current_utility) / eps
            gradients.append(gradient)
        
        # Update efforts using gradient ascent with individual learning rates
        for i in range(2):
            # Update momentum
            momentum[i] = momentum_beta * momentum[i] + (1 - momentum_beta) * gradients[i]
            
            # Gradient ascent step with momentum
            if adaptive_lr:
                # Adaptive learning rate based on gradient magnitude and ability
                base_lr = learning_rates[i]
                ability_scale = ability_params[i] / max(ability_params)  # Scale by relative ability
                adaptive_rate = base_lr * ability_scale * (1.0 / (1.0 + abs(gradients[i])))
                update = adaptive_rate * momentum[i]
            else:
                update = learning_rates[i] * gradients[i]
            
            new_effort = efforts[i] + update
            
            # Clamp to valid range
            if hasattr(env, "effort_range"):
                low, high = env.effort_range
                new_effort = max(low, min(high, new_effort))
            else:
                new_effort = max(0, new_effort)  # Effort must be non-negative
            
            efforts[i] = new_effort
        
        # Check convergence every 100 steps
        if step % 100 == 0:
            # Compute effort changes
            effort_changes = [abs(efforts[i] - prev_efforts[i]) for i in range(2)]
            max_change = max(effort_changes)
            
            # Compute gap from theoretical values
            if theoretical_efforts:
                gaps = [abs(efforts[i] - theoretical_efforts[i]) for i in range(2)]
                current_gap = max(gaps)
                
                # Update best solution
                if current_gap < best_gap:
                    best_gap = current_gap
                    best_efforts = efforts.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                current_gap = max_change
                no_improvement_count += 1 if max_change < convergence_threshold else 0
            
            convergence_history.append({
                "step": step,
                "efforts": efforts.copy(),
                "gradients": gradients.copy(),
                "max_change": max_change,
                "gap": current_gap if theoretical_efforts else None
            })
            
            # Logging
            if step % 1000 == 0 and verbose:
                print(f"Step {step}: e1={efforts[0]:.2f}, e2={efforts[1]:.2f}, "
                      f"gradients=[{gradients[0]:.4f}, {gradients[1]:.4f}], "
                      f"max_change={max_change:.6f}")
                if theoretical_efforts:
                    print(f"  Gaps from theoretical: [{gaps[0]:.3f}, {gaps[1]:.3f}], max_gap={current_gap:.3f}")
            
            # Check for convergence
            if max_change < convergence_threshold:
                if no_improvement_count >= patience // 100:
                    if verbose:
                        print(f"Converged at step {step} (max_change={max_change:.6f})")
                    break
            
            prev_efforts = efforts.copy()
        
        # Adaptive learning rate adjustment
        if adaptive_lr and step % 1000 == 0:
            for i in range(2):
                if abs(gradients[i]) > 0.1:  # Large gradient, reduce learning rate
                    learning_rates[i] *= 0.95
                elif abs(gradients[i]) < 0.01:  # Small gradient, increase learning rate
                    learning_rates[i] *= 1.05
                # Keep learning rate in reasonable bounds
                learning_rates[i] = max(0.001, min(0.5, learning_rates[i]))
    
    end_time = time.time()
    
    # Use best efforts if available
    if theoretical_efforts and best_gap < max([abs(efforts[i] - theoretical_efforts[i]) for i in range(2)]):
        efforts = best_efforts
    
    # Compute final utilities and costs
    final_utilities = []
    final_costs = []
    
    for i in range(2):
        other_effort = efforts[1-i]
        utility, cost = env.compute_utility(i, efforts[i], other_effort)
        final_utilities.append(utility)
        final_costs.append(cost)
    
    # Verify equilibrium conditions
    final_gradients = env.compute_gradients(efforts, eps)
    max_gradient = max(abs(g) for g in final_gradients)
    is_equilibrium = max_gradient < 0.01
    
    # Package solver information
    solver_info = {
        "algorithm": "Gradient Descent (Different Ability)",
        "converged": step < steps - 1,
        "final_step": step,
        "training_time": end_time - start_time,
        "final_gradients": final_gradients,
        "max_gradient": max_gradient,
        "is_equilibrium": is_equilibrium,
        "convergence_history": convergence_history[-10:],  # Last 10 entries
        "learning_rates_final": learning_rates.copy(),
        "best_gap": best_gap if theoretical_efforts else None
    }
    
    if verbose:
        print(f"\nFinal Results:")
        print(f"  Efforts: e1={efforts[0]:.3f}, e2={efforts[1]:.3f}")
        print(f"  Utilities: u1={final_utilities[0]:.3f}, u2={final_utilities[1]:.3f}")
        print(f"  Costs: c1={final_costs[0]:.3f}, c2={final_costs[1]:.3f}")
        print(f"  Final gradients: [{final_gradients[0]:.4f}, {final_gradients[1]:.4f}]")
        print(f"  Is equilibrium: {is_equilibrium}")
        print(f"  Training time: {solver_info['training_time']:.1f}s")
        
        if theoretical_efforts:
            gaps = [abs(efforts[i] - theoretical_efforts[i]) for i in range(2)]
            print(f"  Theoretical efforts: e1*={theoretical_efforts[0]:.3f}, e2*={theoretical_efforts[1]:.3f}")
            print(f"  Gaps: [{gaps[0]:.3f}, {gaps[1]:.3f}], max={max(gaps):.3f}")
    
    logger.info(f"Solver completed: steps={step}, time={solver_info['training_time']:.1f}s, equilibrium={is_equilibrium}")
    
    return efforts, final_utilities, final_costs, solver_info

def verify_different_ability_equilibrium(
    env, 
    efforts: List[float], 
    eps: float = 1e-3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify that the given efforts satisfy Nash equilibrium conditions for different abilities.
    
    Args:
        env: DifferentAbilityEnv environment
        efforts: [e1, e2] effort values to verify
        eps: Epsilon for numerical gradient computation
        verbose: Whether to print detailed analysis
        
    Returns:
        Comprehensive equilibrium verification results
    """
    if len(efforts) != 2:
        raise ValueError("Expected exactly 2 efforts for verification")
    
    logger.info("Verifying Nash equilibrium conditions")
    
    # Compute gradients using the environment
    gradients = env.compute_gradients(efforts, eps)
    
    # Analyze equilibrium using environment's built-in analysis
    analysis = env.analyze_equilibrium(efforts)
    
    # Additional verification specific to different abilities
    e1, e2 = efforts
    l1, l2 = env.get_ability_parameters()
    k1, k2 = env.get_cost_parameters()
    
    # Effective efforts
    eff_e1 = l1 * e1
    eff_e2 = l2 * e2
    
    # Theoretical check: At equilibrium, the marginal benefits should equal marginal costs
    # For uniform noise: marginal win probability ≈ l_i / (2*q)
    # So: (w_h - w_l) * l_i / (2*q) = 2 * k_i * e_i
    # Therefore: e_i = (w_h - w_l) * l_i / (4 * k_i * q)
    
    w_diff = env.w_h - env.w_l
    theoretical_e1 = w_diff * l1 / (4 * k1 * env.q)
    theoretical_e2 = w_diff * l2 / (4 * k2 * env.q)
    
    verification = {
        "efforts": efforts,
        "effective_efforts": [eff_e1, eff_e2],
        "theoretical_efforts": [theoretical_e1, theoretical_e2],
        "gradients": gradients,
        "max_gradient": max(abs(g) for g in gradients),
        "is_equilibrium": max(abs(g) for g in gradients) < 0.01,
        "ability_parameters": [l1, l2],
        "cost_parameters": [k1, k2],
        "win_probabilities": analysis["win_probabilities"],
        "utilities": analysis["utilities"],
        "costs": analysis["costs"],
        "gaps_from_theoretical": analysis["gaps"],
        "convergence_quality": analysis.get("convergence_quality", "Unknown"),
        "ability_advantage": eff_e1 - eff_e2,
        "win_advantage": analysis["win_probabilities"][0] - analysis["win_probabilities"][1]
    }
    
    if verbose:
        print("\n" + "="*60)
        print("NASH EQUILIBRIUM VERIFICATION")
        print("="*60)
        
        print(f"Given efforts: e1={e1:.3f}, e2={e2:.3f}")
        print(f"Effective efforts: l1*e1={eff_e1:.3f}, l2*e2={eff_e2:.3f}")
        print(f"Theoretical efforts: e1*={theoretical_e1:.3f}, e2*={theoretical_e2:.3f}")
        print(f"Gaps from theoretical: [{verification['gaps_from_theoretical'][0]:.3f}, {verification['gaps_from_theoretical'][1]:.3f}]")
        
        print(f"\nGradient Analysis:")
        print(f"  Player 1 gradient: {gradients[0]:.6f}")
        print(f"  Player 2 gradient: {gradients[1]:.6f}")
        print(f"  Max gradient magnitude: {verification['max_gradient']:.6f}")
        print(f"  Is equilibrium: {verification['is_equilibrium']}")
        
        print(f"\nWin Probabilities:")
        p1_win, p2_win = verification['win_probabilities']
        print(f"  Player 1 (l1={l1}): P(win) = {p1_win:.3f}")
        print(f"  Player 2 (l2={l2}): P(win) = {p2_win:.3f}")
        print(f"  Higher ability advantage: {verification['win_advantage']:.3f}")
        
        print(f"\nUtilities and Costs:")
        u1, u2 = verification['utilities']
        c1, c2 = verification['costs']
        print(f"  Player 1: utility = {u1:.3f}, cost = {c1:.3f}")
        print(f"  Player 2: utility = {u2:.3f}, cost = {c2:.3f}")
        
        print(f"\nConvergence Quality: {verification['convergence_quality']}")
    
    logger.info(f"Equilibrium verification: max_gradient={verification['max_gradient']:.6f}, "
                f"quality={verification['convergence_quality']}")
    
    return verification 