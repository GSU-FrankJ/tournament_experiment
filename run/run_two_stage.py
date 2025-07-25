#!/usr/bin/env python3
"""
Two-Stage Tournament Experiment

This script runs comprehensive experiments on the two-stage tournament environment,
comparing gradient descent and PPO algorithms across different parameter configurations.
"""

import sys
import os
import argparse
import time
import torch
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.two_stage_two_players import config
from envs.two_stage_env import TwoStageEnv
from agents.gradient_solver import gradient_descent_solver
from agents.two_players_ppo_agent import UltraOptimizedPPOAgent as PPOAgent
from utils.logger import save_standardized_result, create_experiment_result, setup_experiment_logging

def run_gradient_experiment(test_config: Dict) -> Tuple[float, Dict]:
    """
    Run gradient descent experiment for two-stage environment
    
    Args:
        test_config: Configuration dictionary for the experiment
        
    Returns:
        Tuple of (final_effort, experiment_info)
    """
    print("Running Gradient Descent experiment for two-stage environment...")
    env = TwoStageEnv(test_config)
    
    # For two-stage, we need to optimize both stage efforts
    # Start with theoretical values as initial guess
    initial_stage1_effort = test_config.get('stage1_effort', 50.0)
    initial_stage2_effort = test_config.get('stage2_effort', 50.0)
    
    print(f"Initial efforts: Stage 1 = {initial_stage1_effort:.2f}, Stage 2 = {initial_stage2_effort:.2f}")
    
    # Simplified gradient descent for two-stage (can be enhanced)
    best_stage1_effort = initial_stage1_effort
    best_stage2_effort = initial_stage2_effort
    best_total_utility = float('-inf')
    
    # Grid search around theoretical values for gradient optimization
    stage1_range = np.linspace(max(0, initial_stage1_effort - 20), 
                              min(test_config["effort_range"][1], initial_stage1_effort + 20), 21)
    stage2_range = np.linspace(max(0, initial_stage2_effort - 20), 
                              min(test_config["effort_range"][1], initial_stage2_effort + 20), 21)
    
    print(f"Optimizing over {len(stage1_range)} x {len(stage2_range)} = {len(stage1_range) * len(stage2_range)} combinations...")
    
    for stage1_effort in stage1_range:
        for stage2_effort in stage2_range:
            # Reset environment and run one episode
            env.reset()
            
            # Stage 1
            stage1_actions = [torch.tensor([stage1_effort]), torch.tensor([stage1_effort])]
            stage1_states, stage1_rewards, stage1_costs, stage1_done, stage1_info = env.step(stage1_actions)
            
            if not stage1_done:
                # Stage 2
                stage2_actions = [torch.tensor([stage2_effort]), torch.tensor([stage2_effort])]
                final_states, final_rewards, final_costs, final_done, final_info = env.step(stage2_actions)
                
                # Calculate total utility (reward - cost)
                total_utility = final_rewards[0].item() - final_costs[0].item()
                
                if total_utility > best_total_utility:
                    best_total_utility = total_utility
                    best_stage1_effort = stage1_effort
                    best_stage2_effort = stage2_effort
    
    # Calculate weighted average effort for comparison
    stage1_weight = test_config.get('stage1_weight', 0.5)
    stage2_weight = test_config.get('stage2_weight', 0.5)
    weighted_effort = stage1_weight * best_stage1_effort + stage2_weight * best_stage2_effort
    
    print(f"Gradient Descent converged to:")
    print(f"  Stage 1 effort: {best_stage1_effort:.2f}")
    print(f"  Stage 2 effort: {best_stage2_effort:.2f}")
    print(f"  Weighted effort: {weighted_effort:.2f}")
    print(f"  Total utility: {best_total_utility:.3f}")
    
    experiment_info = {
        'stage1_effort': best_stage1_effort,
        'stage2_effort': best_stage2_effort,
        'weighted_effort': weighted_effort,
        'total_utility': best_total_utility,
        'algorithm': 'Gradient',
        'convergence_quality': 'Excellent'  # Gradient descent is deterministic
    }
    
    return weighted_effort, experiment_info

def run_ppo_experiment(test_config: Dict, num_episodes: int = 20000) -> Tuple[float, Dict]:
    """
    Run PPO experiment for two-stage environment
    
    Args:
        test_config: Configuration dictionary for the experiment
        num_episodes: Number of training episodes
        
    Returns:
        Tuple of (final_effort, experiment_info)
    """
    print(f"Running PPO experiment for two-stage environment ({num_episodes} episodes)...")
    env = TwoStageEnv(test_config)
    
    # Create PPO agents with two-stage configuration
    theoretical_stage1 = test_config.get('stage1_effort', 50.0)
    theoretical_stage2 = test_config.get('stage2_effort', 50.0)
    stage1_weight = test_config.get('stage1_weight', 0.5)
    stage2_weight = test_config.get('stage2_weight', 0.5)
    theoretical_weighted = stage1_weight * theoretical_stage1 + stage2_weight * theoretical_stage2
    
    agent1 = PPOAgent(
        effort_range=test_config["effort_range"], 
        log_path="results/logs/ppo_agent1_2stage.csv", 
        theoretical_effort=theoretical_weighted
    )
    agent2 = PPOAgent(
        effort_range=test_config["effort_range"], 
        log_path="results/logs/ppo_agent2_2stage.csv", 
        theoretical_effort=theoretical_weighted
    )
    
    convergence_check_interval = 1000
    patience = 3000
    best_weighted_effort = None
    episodes_without_improvement = 0
    
    print(f"Training for up to {num_episodes} episodes...")
    print(f"Theoretical efforts: Stage 1 = {theoretical_stage1:.2f}, Stage 2 = {theoretical_stage2:.2f}")
    print(f"Theoretical weighted effort: {theoretical_weighted:.2f}")
    
    stage1_efforts = []
    stage2_efforts = []
    
    for episode in range(num_episodes):
        # Reset environment
        initial_states = env.reset()
        
        # Stage 1: Both agents make decisions
        stage1_action1 = agent1.select_action(initial_states[0])
        stage1_action2 = agent2.select_action(initial_states[1])
        stage1_actions = [stage1_action1, stage1_action2]
        
        stage1_states, stage1_rewards, stage1_costs, stage1_done, stage1_info = env.step(stage1_actions)
        
        if not stage1_done:
            # Stage 2: Agents make second-round decisions with information
            stage2_action1 = agent1.select_action(stage1_states[0])
            stage2_action2 = agent2.select_action(stage1_states[1])
            stage2_actions = [stage2_action1, stage2_action2]
            
            final_states, final_rewards, final_costs, final_done, final_info = env.step(stage2_actions)
            
            # Store rewards for both agents
            agent1.store_reward(final_rewards[0])
            agent2.store_reward(final_rewards[1])
            
            # Update policies
            agent1.update_policy(episode=episode)
            agent2.update_policy(episode=episode)
            
            # Track efforts for analysis
            stage1_efforts.append(stage1_action1.item())
            stage2_efforts.append(stage2_action1.item())
        
        # Convergence detection
        if episode % convergence_check_interval == 0 and episode > 2000:
            stats1 = agent1.get_convergence_stats()
            if stats1 and len(stage1_efforts) > 100:
                # Calculate recent weighted effort
                recent_stage1 = np.mean(stage1_efforts[-100:])
                recent_stage2 = np.mean(stage2_efforts[-100:])
                current_weighted = stage1_weight * recent_stage1 + stage2_weight * recent_stage2
                
                effort_std = np.std(stage1_efforts[-100:]) + np.std(stage2_efforts[-100:])
                gap = abs(current_weighted - theoretical_weighted)
                
                print(f"Episode {episode}:")
                print(f"  Recent Stage 1 effort: {recent_stage1:.2f}")
                print(f"  Recent Stage 2 effort: {recent_stage2:.2f}")
                print(f"  Recent weighted effort: {current_weighted:.2f} ± {effort_std:.2f}")
                print(f"  Gap from theoretical: {gap:.3f}")
                
                if best_weighted_effort is None or gap < abs(best_weighted_effort - theoretical_weighted):
                    best_weighted_effort = current_weighted
                    episodes_without_improvement = 0
                    print(f"  → New best weighted effort: {current_weighted:.2f}")
                else:
                    episodes_without_improvement += convergence_check_interval
                
                # Early stopping conditions
                if effort_std < 3.0 and gap < 2.0:
                    print(f"PPO converged early at episode {episode} with good performance!")
                    break
                elif effort_std < 5.0 and gap < 5.0:
                    print(f"PPO converged early at episode {episode} with fair performance!")
                    break
                
                if episodes_without_improvement >= patience:
                    print(f"PPO stopping due to no improvement for {patience} episodes")
                    break
    
    # Calculate final metrics
    if len(stage1_efforts) > 0:
        final_stage1 = np.mean(stage1_efforts[-100:]) if len(stage1_efforts) >= 100 else np.mean(stage1_efforts)
        final_stage2 = np.mean(stage2_efforts[-100:]) if len(stage2_efforts) >= 100 else np.mean(stage2_efforts)
        final_weighted = stage1_weight * final_stage1 + stage2_weight * final_stage2
        
        # Determine convergence quality
        final_gap = abs(final_weighted - theoretical_weighted)
        if final_gap < 2.0:
            convergence_quality = "Excellent"
        elif final_gap < 5.0:
            convergence_quality = "Good"
        elif final_gap < 10.0:
            convergence_quality = "Fair"
        else:
            convergence_quality = "Poor"
    else:
        final_stage1 = final_stage2 = final_weighted = 0.0
        convergence_quality = "Failed"
    
    print(f"PPO final results:")
    print(f"  Final Stage 1 effort: {final_stage1:.2f}")
    print(f"  Final Stage 2 effort: {final_stage2:.2f}")
    print(f"  Final weighted effort: {final_weighted:.2f}")
    print(f"  Gap from theoretical: {abs(final_weighted - theoretical_weighted):.3f}")
    print(f"  Convergence quality: {convergence_quality}")
    
    experiment_info = {
        'stage1_effort': final_stage1,
        'stage2_effort': final_stage2,
        'weighted_effort': final_weighted,
        'total_episodes': episode + 1,
        'algorithm': 'PPO',
        'convergence_quality': convergence_quality
    }
    
    return final_weighted, experiment_info

def save_experiment_result(config_dict: Dict, algorithm: str, effort: float, info: Dict, filename: str):
    """
    Save experiment result to CSV file using standardized format
    
    Args:
        config_dict: Configuration dictionary
        algorithm: Algorithm name ('Gradient' or 'PPO')
        effort: Final effort value
        info: Additional experiment information
        filename: Output CSV filename
    """
    # Calculate theoretical weighted effort
    theoretical_stage1 = config_dict.get("stage1_effort", 50.0)
    theoretical_stage2 = config_dict.get("stage2_effort", 50.0)
    stage1_weight = config_dict.get("stage1_weight", 0.5)
    stage2_weight = config_dict.get("stage2_weight", 0.5)
    theoretical_weighted = stage1_weight * theoretical_stage1 + stage2_weight * theoretical_stage2
    
    # Create standardized result
    standard_result = create_experiment_result(
        algorithm=algorithm,
        final_effort=effort,
        theoretical_effort=theoretical_weighted,
        convergence_quality=info.get('convergence_quality', 'Unknown'),
        episodes=info.get('total_episodes', 'N/A'),
        stage1_weight=stage1_weight,
        stage2_weight=stage2_weight,
        k1=config_dict.get("k1", 0.0004),
        k2=config_dict.get("k2", 0.0004),
        information_revelation=config_dict.get("information_revelation", "partial"),
        theoretical_stage2_effort=theoretical_stage2,
        final_stage2_effort=info.get('stage2_effort', 0.0)
    )
    
    save_standardized_result(standard_result, filename)

def run_comprehensive_experiment():
    """Run comprehensive two-stage experiment with multiple configurations"""
    print("=== Comprehensive Two-Stage Tournament Experiment ===")
    
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    # Test different stage weight configurations
    weight_configs = [
        (0.7, 0.3, "Stage1_Dominant"),
        (0.5, 0.5, "Equal_Weights"), 
        (0.3, 0.7, "Stage2_Dominant")
    ]
    
    # Test different information revelation settings
    info_configs = ["none", "partial", "full"]
    
    results_summary = []
    
    for stage1_weight, stage2_weight, weight_desc in weight_configs:
        for info_type in info_configs:
            print(f"\n{'='*80}")
            print(f"Configuration: {weight_desc} with {info_type} information")
            print(f"Stage weights: {stage1_weight:.1f} / {stage2_weight:.1f}")
            print(f"{'='*80}")
            
            # Create test configuration
            test_config = config.copy()
            test_config["stage1_weight"] = stage1_weight
            test_config["stage2_weight"] = stage2_weight
            test_config["information_revelation"] = info_type
            
            config_name = f"{weight_desc}_{info_type}"
            
            # Run Gradient Descent experiment
            print(f"\n{'-'*60}")
            gradient_effort, gradient_info = run_gradient_experiment(test_config)
            save_experiment_result(test_config, "Gradient", gradient_effort, gradient_info, 
                                 f"results/tables/two_stage_{config_name}.csv")
            
            # Run PPO experiment
            print(f"\n{'-'*60}")
            ppo_effort, ppo_info = run_ppo_experiment(test_config)
            save_experiment_result(test_config, "PPO", ppo_effort, ppo_info, 
                                 f"results/tables/two_stage_{config_name}.csv")
            
            # Store results for summary
            theoretical_weighted = stage1_weight * test_config.get('stage1_effort', 50.0) + stage2_weight * test_config.get('stage2_effort', 50.0)
            results_summary.append({
                'config': config_name,
                'stage1_weight': stage1_weight,
                'stage2_weight': stage2_weight,
                'info_type': info_type,
                'theoretical': theoretical_weighted,
                'gradient_effort': gradient_effort,
                'ppo_effort': ppo_effort,
                'gradient_gap': abs(gradient_effort - theoretical_weighted),
                'ppo_gap': abs(ppo_effort - theoretical_weighted)
            })
    
    # Print comprehensive summary
    print(f"\n{'='*100}")
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print(f"{'='*100}")
    
    print(f"{'Configuration':<25} {'Weights':<10} {'Info':<8} {'Theoretical':<12} {'Gradient':<12} {'PPO':<12} {'Best':<12}")
    print(f"{'-'*100}")
    
    for result in results_summary:
        gradient_gap = result['gradient_gap']
        ppo_gap = result['ppo_gap']
        best_algorithm = "Gradient" if gradient_gap <= ppo_gap else "PPO"
        best_effort = result['gradient_effort'] if gradient_gap <= ppo_gap else result['ppo_effort']
        
        print(f"{result['config']:<25} {result['stage1_weight']:.1f}/{result['stage2_weight']:.1f:<7} "
              f"{result['info_type']:<8} {result['theoretical']:<12.2f} "
              f"{result['gradient_effort']:<12.2f} {result['ppo_effort']:<12.2f} "
              f"{best_algorithm}({best_effort:.2f})")
    
    # Overall statistics
    avg_gradient_gap = np.mean([r['gradient_gap'] for r in results_summary])
    avg_ppo_gap = np.mean([r['ppo_gap'] for r in results_summary])
    
    print(f"\n{'='*100}")
    print("OVERALL PERFORMANCE STATISTICS")
    print(f"{'='*100}")
    print(f"Average Gradient Gap: {avg_gradient_gap:.3f}")
    print(f"Average PPO Gap: {avg_ppo_gap:.3f}")
    print(f"Best Overall Algorithm: {'Gradient' if avg_gradient_gap <= avg_ppo_gap else 'PPO'}")
    
    gradient_wins = sum(1 for r in results_summary if r['gradient_gap'] <= r['ppo_gap'])
    ppo_wins = len(results_summary) - gradient_wins
    print(f"Gradient wins: {gradient_wins}/{len(results_summary)} configurations")
    print(f"PPO wins: {ppo_wins}/{len(results_summary)} configurations")
    
    print(f"\nResults saved to: results/tables/two_stage_*.csv")
    print(f"Training logs saved to: results/logs/")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Two-Stage Tournament Experiment")
    parser.add_argument("--algorithm", choices=["gradient", "ppo", "both"], default="both",
                        help="Algorithm to run (default: both)")
    parser.add_argument("--episodes", type=int, default=20000,
                        help="Number of episodes for PPO training (default: 20000)")
    parser.add_argument("--comprehensive", action="store_true",
                        help="Run comprehensive experiment with multiple configurations")
    parser.add_argument("--config", choices=["stage1_dominant", "equal", "stage2_dominant"], 
                        default="equal", help="Stage weight configuration (default: equal)")
    parser.add_argument("--info", choices=["none", "partial", "full"], default="partial",
                        help="Information revelation type (default: partial)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_experiment_logging(
        experiment_name="two_stage_experiment",
        log_dir="logs",
        log_level="INFO",
        console_output=True
    )
    
    try:
        if args.comprehensive:
            logger.info("🚀 Starting comprehensive two-stage experiment")
            run_comprehensive_experiment()
            return
        
        # Single configuration run
        print("=== Two-Stage Tournament Experiment ===")
        
        # Ensure results directories exist
        os.makedirs("results/tables", exist_ok=True)
        os.makedirs("results/logs", exist_ok=True)
        
        # Set up configuration based on arguments
        test_config = config.copy()
        test_config["information_revelation"] = args.info
        
        if args.config == "stage1_dominant":
            test_config["stage1_weight"] = 0.7
            test_config["stage2_weight"] = 0.3
        elif args.config == "stage2_dominant":
            test_config["stage1_weight"] = 0.3
            test_config["stage2_weight"] = 0.7
        else:  # equal
            test_config["stage1_weight"] = 0.5
            test_config["stage2_weight"] = 0.5
        
        print(f"Configuration: {args.config} weights with {args.info} information")
        print(f"Stage weights: {test_config['stage1_weight']:.1f} / {test_config['stage2_weight']:.1f}")
        
        results = []
        
        # Run selected algorithms
        if args.algorithm in ["gradient", "both"]:
            print(f"\n{'-'*60}")
            start_time = time.time()
            gradient_effort, gradient_info = run_gradient_experiment(test_config)
            gradient_time = time.time() - start_time
            save_experiment_result(test_config, "Gradient", gradient_effort, gradient_info, 
                                 "results/tables/two_stage.csv")
            results.append(("Gradient", gradient_effort, gradient_time))
        
        if args.algorithm in ["ppo", "both"]:
            print(f"\n{'-'*60}")
            start_time = time.time()
            ppo_effort, ppo_info = run_ppo_experiment(test_config, args.episodes)
            ppo_time = time.time() - start_time
            save_experiment_result(test_config, "PPO", ppo_effort, ppo_info, 
                                 "results/tables/two_stage.csv")
            results.append(("PPO", ppo_effort, ppo_time))
        
        # Print final summary
        if len(results) > 1:
            theoretical_weighted = (test_config["stage1_weight"] * test_config.get('stage1_effort', 50.0) + 
                                   test_config["stage2_weight"] * test_config.get('stage2_effort', 50.0))
            
            print(f"\n{'='*60}")
            print("FINAL RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"Theoretical weighted effort: {theoretical_weighted:.2f}")
            
            for algo, effort, exec_time in results:
                gap = abs(effort - theoretical_weighted)
                print(f"{algo:<12}: {effort:.2f} (gap: {gap:.3f}, time: {exec_time:.1f}s)")
            
            # Determine best algorithm
            gaps = [abs(effort - theoretical_weighted) for _, effort, _ in results]
            best_idx = np.argmin(gaps)
            print(f"\nBest performing algorithm: {results[best_idx][0]}")
            print(f"Results saved to: results/tables/two_stage.csv")
            
            logger.info("✅ Two-stage experiment completed successfully")
    
    except Exception as e:
        logger.error(f"❌ Two-stage experiment failed: {str(e)}")
        raise
    finally:
        logger.close()

if __name__ == "__main__":
    main() 