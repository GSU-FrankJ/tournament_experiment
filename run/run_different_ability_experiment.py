#!/usr/bin/env python3
"""
Different Ability Two Players Experiment
=========================================

This experiment implements the "different ability" scenario with:
- Player 1: Higher ability (l1 = 10)
- Player 2: Lower ability (l2 = 5)  
- Equal cost parameters (k1 = k2 = 0.0004)

Tests three algorithms (Gradient, PPO) across all required conditions:
- q values: [25.0, 40.0, 55.0]
- effort ranges: [(0, 100), (0, 200)]

Performance targets:
- Excellent: Gap < 0.5
- Good: Gap < 1.0  
- Minimum requirement: All conditions achieve "Good" quality
"""

import os
import sys
import time
import json
import csv
import torch
import numpy as np
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration and components
from config.different_ability_two_players import test_configs, DIFFERENT_ABILITY_CONFIG
from envs.different_ability_env import DifferentAbilityEnv
from agents.different_ability_solver import different_ability_gradient_descent_solver, verify_different_ability_equilibrium
from agents.two_players_ppo_agent import UltraOptimizedPPOAgent as PPOAgent
from utils.logger import get_logger, save_standardized_result, create_experiment_result

# Initialize logger
logger = get_logger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def analyze_convergence_quality(gaps: List[float]) -> Dict[str, Any]:
    """
    Analyze convergence quality based on gaps from theoretical values.
    
    Quality levels:
    - Excellent: Gap < 0.5
    - Good: Gap < 1.0
    - Fair: Gap < 5.0  
    - Poor: Gap >= 5.0
    """
    if not gaps:
        return {"quality": "Unknown", "max_gap": 0, "avg_gap": 0}
        
    max_gap = max(gaps)
    avg_gap = sum(gaps) / len(gaps)
    
    if max_gap < 0.5:
        quality = "Excellent"
    elif max_gap < 1.0:
        quality = "Good"
    elif max_gap < 5.0:
        quality = "Fair"
    else:
        quality = "Poor"
    
    return {
        "quality": quality,
        "max_gap": max_gap,
        "avg_gap": avg_gap,
        "individual_gaps": gaps
    }

def run_gradient_algorithm(env: DifferentAbilityEnv, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run gradient descent algorithm for the given configuration."""
    
    print(f"\n{'='*60}")
    print("GRADIENT DESCENT ALGORITHM")
    print(f"Configuration: q={config['q']}, effort_range={config['effort_range']}")
    print(f"{'='*60}")
    
    logger.info(f"Running gradient descent: q={config['q']}, range={config['effort_range']}")
    
    start_time = time.time()
    
    # Run gradient descent solver
    efforts, utilities, costs, solver_info = different_ability_gradient_descent_solver(
        env=env,
        lr=0.1,           # Learning rate
        steps=50000,      # Max iterations
        eps=1e-3,         # Gradient epsilon
        adaptive_lr=True, # Use adaptive learning rates
        verbose=True
    )
    
    end_time = time.time()
    
    # Get theoretical values for comparison
    theoretical_efforts = env.get_theoretical_efforts()
    
    # Compute gaps
    gaps = [abs(efforts[i] - theoretical_efforts[i]) for i in range(len(efforts))]
    
    # Analyze convergence quality
    convergence_analysis = analyze_convergence_quality(gaps)
    
    # Verify equilibrium
    equilibrium_verification = verify_different_ability_equilibrium(env, efforts, verbose=False)
    
    # Package results
    result = {
        "algorithm": "Gradient",
        "q": config["q"],
        "effort_range": config["effort_range"],
        "efforts": efforts,
        "theoretical_efforts": theoretical_efforts,
        "gaps": gaps,
        "utilities": utilities,
        "costs": costs,
        "training_time": end_time - start_time,
        "convergence": convergence_analysis,
        "equilibrium_verification": equilibrium_verification,
        "solver_info": solver_info,
        "ability_parameters": env.get_ability_parameters(),
        "cost_parameters": env.get_cost_parameters()
    }
    
    # Print summary
    print(f"Training time: {result['training_time']:.1f}s")
    print(f"Final efforts: e1={efforts[0]:.2f}, e2={efforts[1]:.2f}")
    print(f"Theoretical efforts: e1*={theoretical_efforts[0]:.2f}, e2*={theoretical_efforts[1]:.2f}")
    print(f"Gaps: [{gaps[0]:.3f}, {gaps[1]:.3f}] (max: {max(gaps):.3f})")
    print(f"Convergence quality: {convergence_analysis['quality']}")
    print(f"Nash equilibrium: {equilibrium_verification['is_equilibrium']}")
    
    logger.info(f"Gradient algorithm completed: quality={convergence_analysis['quality']}, max_gap={max(gaps):.3f}")
    
    return result

def run_ppo_algorithm(env: DifferentAbilityEnv, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run PPO algorithm for the given configuration."""
    
    print(f"\n{'='*60}")
    print("PPO ALGORITHM")
    print(f"Configuration: q={config['q']}, effort_range={config['effort_range']}")
    print(f"{'='*60}")
    
    logger.info(f"Running PPO: q={config['q']}, range={config['effort_range']}")
    
    # PPO configuration optimized for different ability scenario
    ppo_config = {
        "learning_rate": 0.0001,
        "hidden_dim": 256,
        "num_layers": 4,
        "activation": "tanh",
        "clip_epsilon": 0.2,
        "value_coef": 0.75,
        "entropy_coef": 0.005,
        "max_grad_norm": 0.3,
        "batch_size": 64,
        "update_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "weight_decay": 1e-5,
        "dropout_rate": 0.05,
        "lr_schedule": "cosine_annealing",
        "separate_networks": True,
        "reward_normalization": True,
        "episodes": 15000  # Training episodes
    }
    
    start_time = time.time()
    
    # Create PPO agents for each player
    theoretical_efforts = env.get_theoretical_efforts()
    agents = []
    
    for i in range(env.num_players):
        # Create agent with theoretical effort as target
        agent = PPOAgent(
            effort_range=config["effort_range"],
            theoretical_effort=theoretical_efforts[i]
        )
        agents.append(agent)
    
    # Training loop
    episodes = ppo_config["episodes"]
    
    for episode in range(episodes):
        # Reset environment  
        states = env.reset()
        
        # Get actions from all agents
        actions = []
        for i, agent in enumerate(agents):
            action = agent.select_action(states[i])
            actions.append(action)
        
        # Environment step
        next_states, rewards, costs, done, info = env.step(actions)
        
        # Store rewards and update agents
        for i, agent in enumerate(agents):
            agent.store_reward(rewards[i])
            
            # Update every batch_size episodes
            if (episode + 1) % ppo_config["batch_size"] == 0:
                agent.update_policy()
        
        # Progress logging
        if episode % 2000 == 0:
            efforts = [action.item() for action in actions]
            print(f"Episode {episode}: e1={efforts[0]:.2f}, e2={efforts[1]:.2f}, "
                  f"rewards=[{rewards[0].item():.3f}, {rewards[1].item():.3f}]")
    
    end_time = time.time()
    
    # Final evaluation
    states = env.reset()
    actions = []
    for i, agent in enumerate(agents):
        # Get final action
        action = agent.select_action(states[i])
        actions.append(action)
    
    efforts = [action.item() for action in actions]
    next_states, rewards, costs, done, info = env.step(actions)
    utilities = [reward.item() for reward in rewards]
    costs_list = [cost.item() for cost in costs]
    
    # Compute gaps from theoretical values
    gaps = [abs(efforts[i] - theoretical_efforts[i]) for i in range(len(efforts))]
    
    # Analyze convergence quality
    convergence_analysis = analyze_convergence_quality(gaps)
    
    # Package results
    result = {
        "algorithm": "PPO",
        "q": config["q"],
        "effort_range": config["effort_range"],
        "efforts": efforts,
        "theoretical_efforts": theoretical_efforts,
        "gaps": gaps,
        "utilities": utilities,
        "costs": costs_list,
        "training_time": end_time - start_time,
        "convergence": convergence_analysis,
        "config": ppo_config,
        "ability_parameters": env.get_ability_parameters(),
        "cost_parameters": env.get_cost_parameters()
    }
    
    # Print summary
    print(f"Training time: {result['training_time']:.1f}s")
    print(f"Final efforts: e1={efforts[0]:.2f}, e2={efforts[1]:.2f}")
    print(f"Theoretical efforts: e1*={theoretical_efforts[0]:.2f}, e2*={theoretical_efforts[1]:.2f}")
    print(f"Gaps: [{gaps[0]:.3f}, {gaps[1]:.3f}] (max: {max(gaps):.3f})")
    print(f"Convergence quality: {convergence_analysis['quality']}")
    
    logger.info(f"PPO algorithm completed: quality={convergence_analysis['quality']}, max_gap={max(gaps):.3f}")
    
    return result

def save_algorithm_result_to_csv(result: Dict[str, Any]):
    """Save algorithm result in standardized CSV format."""
    
    # Calculate weighted effort (average of both players)
    efforts = result["efforts"]
    theoretical_efforts = result["theoretical_efforts"]
    
    final_effort = sum(efforts) / len(efforts)
    theoretical_effort = sum(theoretical_efforts) / len(theoretical_efforts)
    
    # Create standardized result
    standard_result = create_experiment_result(
        algorithm=result["algorithm"],
        final_effort=final_effort,
        theoretical_effort=theoretical_effort,
        convergence_quality=result["convergence"]["quality"],
        episodes=f"{result.get('config', {}).get('episodes', 'gradient')}_episodes" if result["algorithm"] == "PPO" else "gradient_steps",
        k1=result["cost_parameters"][0],
        information_revelation="none"
    )
    
    # Add additional fields specific to different ability
    standard_result.update({
        "l1": result["ability_parameters"][0],
        "l2": result["ability_parameters"][1], 
        "q": result["q"],
        "effort_range_min": result["effort_range"][0],
        "effort_range_max": result["effort_range"][1],
        "effort1": result["efforts"][0],
        "effort2": result["efforts"][1],
        "theoretical_effort1": result["theoretical_efforts"][0],
        "theoretical_effort2": result["theoretical_efforts"][1],
        "gap1": result["gaps"][0],
        "gap2": result["gaps"][1],
        "max_gap": result["convergence"]["max_gap"],
        "training_time": result["training_time"]
    })
    
    # Save to CSV
    save_standardized_result(standard_result, "results/tables/different_ability.csv")

def run_comprehensive_experiment():
    """Run the comprehensive different ability experiment across all test conditions."""
    
    print("="*80)
    print("DIFFERENT ABILITY TWO PLAYERS EXPERIMENT")
    print("="*80)
    print(f"Player abilities: l1={DIFFERENT_ABILITY_CONFIG['l1']}, l2={DIFFERENT_ABILITY_CONFIG['l2']}")
    print(f"Cost parameters: k1=k2={DIFFERENT_ABILITY_CONFIG['k']}")
    print(f"Test conditions: {len(test_configs)} configurations")
    print(f"Q values: [25.0, 40.0, 55.0]")
    print(f"Effort ranges: [(0, 100), (0, 200)]")
    print()
    
    logger.info("Starting comprehensive different ability experiment")
    
    all_results = []
    
    # Test each configuration
    for i, config in enumerate(test_configs):
        print(f"\n{'='*80}")
        print(f"TEST CONFIGURATION {i+1}/{len(test_configs)}")
        print(f"q = {config['q']}, effort_range = {config['effort_range']}")
        print(f"Theoretical: e1* = {config['theoretical_effort1']:.2f}, e2* = {config['theoretical_effort2']:.2f}")
        print(f"{'='*80}")
        
        logger.info(f"Testing config {i+1}: q={config['q']}, range={config['effort_range']}")
        
        # Create environment for this configuration
        env = DifferentAbilityEnv(config)
        
        config_results = []
        
        try:
            # Run Gradient algorithm
            gradient_result = run_gradient_algorithm(env, config)
            config_results.append(gradient_result)
            save_algorithm_result_to_csv(gradient_result)
            
            # Run PPO algorithm  
            ppo_result = run_ppo_algorithm(env, config)
            config_results.append(ppo_result)
            save_algorithm_result_to_csv(ppo_result)
            
            # Validate performance standards
            for result in config_results:
                quality = result["convergence"]["quality"]
                max_gap = result["convergence"]["max_gap"]
                
                print(f"\n🔍 Performance Validation for {result['algorithm']}:")
                print(f"  Quality: {quality}")
                print(f"  Max gap: {max_gap:.3f}")
                
                # Check minimum performance standard
                if quality not in ["Excellent", "Good"]:
                    print(f"  ❌ WARNING: Performance below standard (requires Good+)")
                    logger.warning(f"Poor performance: {result['algorithm']} - {quality}")
                else:
                    print(f"  ✅ Performance meets standard")
                    logger.info(f"Good performance: {result['algorithm']} - {quality}")
            
        except Exception as e:
            print(f"❌ Error in configuration {i+1}: {e}")
            logger.error(f"Configuration {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        all_results.extend(config_results)
    
    # Generate comprehensive summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    # Group results by algorithm
    algorithm_results = {}
    for result in all_results:
        algo = result["algorithm"]
        if algo not in algorithm_results:
            algorithm_results[algo] = []
        algorithm_results[algo].append(result)
    
    # Analyze performance by algorithm
    for algo, results in algorithm_results.items():
        print(f"\n{algo} Algorithm Summary:")
        print(f"  Total tests: {len(results)}")
        
        qualities = [r["convergence"]["quality"] for r in results]
        quality_counts = {q: qualities.count(q) for q in set(qualities)}
        
        print(f"  Quality distribution: {quality_counts}")
        
        max_gaps = [r["convergence"]["max_gap"] for r in results]
        avg_max_gap = sum(max_gaps) / len(max_gaps)
        
        print(f"  Average max gap: {avg_max_gap:.3f}")
        
        # Check if all tests meet standard
        passing_tests = sum(1 for q in qualities if q in ["Excellent", "Good"])
        pass_rate = passing_tests / len(results) * 100
        
        print(f"  Pass rate (Good+): {passing_tests}/{len(results)} ({pass_rate:.1f}%)")
        
        if pass_rate == 100:
            print(f"  ✅ All tests meet performance standard")
        else:
            print(f"  ❌ {len(results) - passing_tests} tests below standard")
    
    # Save comprehensive results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    results_serializable = convert_numpy_types(all_results)
    
    results_file = os.path.join(results_dir, "different_ability_experiment_results.json")
    with open(results_file, "w") as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate performance matrix
    print(f"\n{'='*80}")
    print("PERFORMANCE MATRIX")
    print(f"{'='*80}")
    
    print(f"{'Algorithm':<12} {'q=25 (0,100)':<12} {'q=25 (0,200)':<12} {'q=40 (0,100)':<12} {'q=40 (0,200)':<12} {'q=55 (0,100)':<12} {'q=55 (0,200)':<12}")
    print("-" * 80)
    
    for algo in ["Gradient", "PPO"]:
        row = f"{algo:<12} "
        algo_results = algorithm_results.get(algo, [])
        
        for q in [25.0, 40.0, 55.0]:
            for effort_range in [(0, 100), (0, 200)]:
                # Find result for this configuration
                matching_result = None
                for r in algo_results:
                    if r["q"] == q and r["effort_range"] == effort_range:
                        matching_result = r
                        break
                
                if matching_result:
                    quality = matching_result["convergence"]["quality"]
                    gap = matching_result["convergence"]["max_gap"]
                    cell = f"{quality[0]}{gap:.1f}"  # First letter + gap
                else:
                    cell = "N/A"
                
                row += f"{cell:<12} "
        
        print(row)
    
    logger.info("Comprehensive experiment completed")
    
    return all_results

def main():
    """Main experiment entry point."""
    try:
        results = run_comprehensive_experiment()
        print(f"\n✅ Experiment completed successfully with {len(results)} algorithm runs")
        return 0
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 