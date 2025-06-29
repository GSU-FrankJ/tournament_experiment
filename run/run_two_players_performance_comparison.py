#!/usr/bin/env python3
"""
Two-Player One-Stage Performance Comparison Script
=================================================

This script compares the performance of different optimization approaches:
1. Original PPO (from existing implementation)
2. Optimized Gradient Descent (adaptive learning rate)
3. Ultra-Optimized PPO (with curriculum learning and stability improvements)

Performance metrics:
- Final gap from theoretical effort
- Convergence time
- Training stability
- Convergence quality
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple

# Import configurations and environments
from config.one_stage_two_players import config
from envs.one_stage_env import OneStageEnv

# Import agents
from agents.gradient_solver import gradient_descent_solver
from agents.ppo_agent import PPOAgent
from agents.enhanced_ppo_agent import EnhancedPPOAgent, ContinuousActionSpace

# Import utilities
from utils.logger import save_result

def run_original_ppo_experiment() -> Dict:
    """
    Run experiment with original PPO agent
    """
    print("=" * 60)
    print("ORIGINAL PPO EXPERIMENT")
    print("=" * 60)
    
    env = OneStageEnv(config)
    
    # Original PPO configuration
    agent1 = PPOAgent(
        effort_range=config["effort_range"],
        log_path="results/logs/original_ppo_agent1.csv",
        theoretical_effort=config["effort"]
    )
    agent2 = PPOAgent(
        effort_range=config["effort_range"],
        log_path="results/logs/original_ppo_agent2.csv",
        theoretical_effort=config["effort"]
    )
    
    num_episodes = 10000
    start_time = time.time()
    
    print(f"Training original PPO for {num_episodes} episodes...")
    
    best_effort = None
    best_gap = float('inf')
    
    for episode in range(num_episodes):
        state1, state2 = env.reset()
        a1 = agent1.select_action(state1)
        a2 = agent2.select_action(state2)
        _, rewards, _, _, info = env.step(torch.stack([a1, a2]))
        
        agent1.store_reward(rewards[0])
        agent2.store_reward(rewards[1])
        
        agent1.update_policy()
        agent2.update_policy()
        
        # Track performance
        current_effort = info["efforts"][0]
        current_gap = abs(current_effort - config["effort"])
        
        if current_gap < best_gap:
            best_effort = current_effort
            best_gap = current_gap
        
        # Progress reporting
        if episode % 2000 == 0 and episode > 0:
            elapsed_time = time.time() - start_time
            print(f"  Episode {episode}: effort={current_effort:.2f}, gap={current_gap:.3f}, time={elapsed_time:.1f}s")
            
            # Early stopping
            if current_gap < 0.5:
                print(f"  Early convergence achieved at episode {episode}")
                break
    
    training_time = time.time() - start_time
    
    # Determine convergence quality
    if best_gap < 0.5:
        convergence_quality = "Excellent"
    elif best_gap < 2.0:
        convergence_quality = "Good"
    elif best_gap < 5.0:
        convergence_quality = "Fair"
    else:
        convergence_quality = "Poor"
    
    result = {
        "algorithm": "Original_PPO",
        "final_effort": round(best_effort, 2),
        "final_gap": round(best_gap, 3),
        "convergence_quality": convergence_quality,
        "training_time": round(training_time, 2),
        "episodes_trained": episode + 1,
        "parameters": "default_configuration"
    }
    
    print(f"Original PPO Results:")
    print(f"  Final effort: {best_effort:.3f} (theoretical: {config['effort']:.3f})")
    print(f"  Gap: {best_gap:.3f}")
    print(f"  Convergence quality: {convergence_quality}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Episodes: {episode + 1}")
    
    return result

def run_enhanced_ppo_experiment() -> Dict:
    """
    Run experiment with enhanced PPO agent (best known configuration)
    """
    print("=" * 60)
    print("ENHANCED PPO EXPERIMENT")
    print("=" * 60)
    
    env = OneStageEnv(config)
    
    # Enhanced PPO with best configuration from optimization results
    action_space = ContinuousActionSpace(low=0, high=200)
    
    agent1 = EnhancedPPOAgent(
        action_space=action_space,
        lr=0.0001,
        hidden_dim=128,
        num_layers=3,
        activation='tanh',
        clip_epsilon=0.2,
        value_coef=0.75,
        entropy_coef=0.005,
        max_grad_norm=0.3,
        batch_size=64,
        update_epochs=10,
        gamma=0.995,
        gae_lambda=0.97,
        weight_decay=1e-5,
        dropout_rate=0.05,
        lr_schedule='cosine_annealing',
        separate_networks=True,
        reward_normalization=True,
        log_path="results/logs/enhanced_ppo_agent1.csv"
    )
    
    agent2 = EnhancedPPOAgent(
        action_space=action_space,
        lr=0.0001,
        hidden_dim=128,
        num_layers=3,
        activation='tanh',
        clip_epsilon=0.2,
        value_coef=0.75,
        entropy_coef=0.005,
        max_grad_norm=0.3,
        batch_size=64,
        update_epochs=10,
        gamma=0.995,
        gae_lambda=0.97,
        weight_decay=1e-5,
        dropout_rate=0.05,
        lr_schedule='cosine_annealing',
        separate_networks=True,
        reward_normalization=True,
        log_path="results/logs/enhanced_ppo_agent2.csv"
    )
    
    num_episodes = 12000
    start_time = time.time()
    
    print(f"Training enhanced PPO for {num_episodes} episodes...")
    
    best_effort = None
    best_gap = float('inf')
    
    for episode in range(num_episodes):
        state1, state2 = env.reset()
        a1 = agent1.select_action(state1)
        a2 = agent2.select_action(state2)
        _, rewards, _, _, info = env.step(torch.stack([a1, a2]))
        
        agent1.store_reward(rewards[0])
        agent2.store_reward(rewards[1])
        
        agent1.update_policy(episode)
        agent2.update_policy(episode)
        
        # Track performance
        current_effort = info["efforts"][0]
        current_gap = abs(current_effort - config["effort"])
        
        if current_gap < best_gap:
            best_effort = current_effort
            best_gap = current_gap
        
        # Progress reporting
        if episode % 2000 == 0 and episode > 0:
            elapsed_time = time.time() - start_time
            print(f"  Episode {episode}: effort={current_effort:.2f}, gap={current_gap:.3f}, time={elapsed_time:.1f}s")
            
            # Early stopping
            if current_gap < 0.5:
                print(f"  Early convergence achieved at episode {episode}")
                break
    
    training_time = time.time() - start_time
    
    # Determine convergence quality
    if best_gap < 0.5:
        convergence_quality = "Excellent"
    elif best_gap < 2.0:
        convergence_quality = "Good"
    elif best_gap < 5.0:
        convergence_quality = "Fair"
    else:
        convergence_quality = "Poor"
    
    result = {
        "algorithm": "Enhanced_PPO",
        "final_effort": round(best_effort, 2),
        "final_gap": round(best_gap, 3),
        "convergence_quality": convergence_quality,
        "training_time": round(training_time, 2),
        "episodes_trained": episode + 1,
        "parameters": "optimized_trial15_configuration"
    }
    
    print(f"Enhanced PPO Results:")
    print(f"  Final effort: {best_effort:.3f} (theoretical: {config['effort']:.3f})")
    print(f"  Gap: {best_gap:.3f}")
    print(f"  Convergence quality: {convergence_quality}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Episodes: {episode + 1}")
    
    return result

def run_adaptive_gradient_experiment() -> Dict:
    """
    Run experiment with adaptive gradient descent
    """
    print("=" * 60)
    print("ADAPTIVE GRADIENT DESCENT EXPERIMENT")
    print("=" * 60)
    
    env = OneStageEnv(config)
    
    start_time = time.time()
    effort, eu, cost = gradient_descent_solver(env, lr=0.1, steps=50000, eps=1e-3)
    training_time = time.time() - start_time
    
    theoretical_effort = config["effort"]
    gap = abs(effort - theoretical_effort)
    
    # Determine convergence quality
    if gap < 0.1:
        convergence_quality = "Excellent"
    elif gap < 0.5:
        convergence_quality = "Good"
    elif gap < 2.0:
        convergence_quality = "Fair"
    else:
        convergence_quality = "Poor"
    
    result = {
        "algorithm": "Adaptive_Gradient",
        "final_effort": round(effort, 2),
        "final_gap": round(gap, 3),
        "convergence_quality": convergence_quality,
        "training_time": round(training_time, 2),
        "episodes_trained": 50000,  # steps
        "parameters": "lr=0.1, steps=50000, eps=1e-3"
    }
    
    print(f"Adaptive Gradient Results:")
    print(f"  Final effort: {effort:.3f} (theoretical: {config['effort']:.3f})")
    print(f"  Gap: {gap:.3f}")
    print(f"  Convergence quality: {convergence_quality}")
    print(f"  Training time: {training_time:.2f}s")
    
    return result

def main():
    """
    Main function to run performance comparison experiments
    """
    print("Starting Two-Player One-Stage Performance Comparison")
    print(f"Theoretical effort: {config['effort']:.3f}")
    print(f"Effort range: {config['effort_range']}")
    
    # Create results directory
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/performance_comparison", exist_ok=True)
    
    # Run all experiments
    results = {}
    
    try:
        # 1. Adaptive Gradient (baseline)
        results["adaptive_gradient"] = run_adaptive_gradient_experiment()
        
        # 2. Original PPO
        results["original_ppo"] = run_original_ppo_experiment()
        
        # 3. Enhanced PPO
        results["enhanced_ppo"] = run_enhanced_ppo_experiment()
        
        # Performance analysis
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("=" * 80)
        
        # Sort by performance (gap)
        sorted_results = sorted(results.items(), key=lambda x: x[1]["final_gap"])
        
        print(f"{'Algorithm':<20} {'Gap':<8} {'Quality':<12} {'Time(s)':<8} {'Episodes':<10}")
        print("-" * 70)
        
        for name, result in sorted_results:
            print(f"{result['algorithm']:<20} {result['final_gap']:<8.3f} {result['convergence_quality']:<12} "
                  f"{result['training_time']:<8.1f} {result['episodes_trained']:<10}")
        
        # Efficiency analysis
        print(f"\nEfficiency Analysis:")
        print(f"- Best accuracy: {sorted_results[0][1]['algorithm']} (gap: {sorted_results[0][1]['final_gap']:.3f})")
        
        # Find fastest to converge
        fastest = min(results.items(), key=lambda x: x[1]["training_time"])
        print(f"- Fastest training: {fastest[1]['algorithm']} ({fastest[1]['training_time']:.1f}s)")
        
        # Find most efficient (good accuracy + reasonable time)
        efficient_results = [r for r in results.values() if r["final_gap"] < 1.0]
        if efficient_results:
            most_efficient = min(efficient_results, key=lambda x: x["training_time"])
            print(f"- Most efficient: {most_efficient['algorithm']} (gap: {most_efficient['final_gap']:.3f}, time: {most_efficient['training_time']:.1f}s)")
        
        # Save comprehensive results
        comparison_results = {
            "config": config,
            "results": results,
            "summary": {
                "best_accuracy": sorted_results[0][1],
                "fastest_training": fastest[1],
                "most_efficient": most_efficient if efficient_results else None
            },
            "timestamp": time.time()
        }
        
        with open("results/performance_comparison/comparison_results.json", "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\nDetailed results saved to: results/performance_comparison/comparison_results.json")
        
    except Exception as e:
        print(f"Error during experiments: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 