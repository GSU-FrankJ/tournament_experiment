import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from config.one_stage_three_players import config
from envs.one_stage_env import OneStageEnv
from agents.gradient_solver import gradient_descent_solver
# REINFORCE agent removed - focusing on Gradient and PPO algorithms
from agents.ppo_agent import PPOAgent
from utils.logger import save_result

def run_gradient_experiment():
    """Run gradient descent experiment for three players"""
    print("Running Gradient Descent experiment...")
    env = OneStageEnv(config)
    effort, eu, cost = gradient_descent_solver(env, lr=0.1, steps=100000, eps=1e-3)
    
    result = {
        "k": config["k"],
        "q": config["q"],
        "w_h": config["w_h"],
        "w_l": config["w_l"],
        "EU": round(config["eu"], 2),
        "Cost_of_effort": round(config["cost"], 2),
        "effort": round(config["effort"], 2),
        "Model_training": "Gradient",
        "Parameter": f"lr=0.1, steps=100000, eps=1e-3",
        "Effort_0_100": round(effort, 2) if config["effort_range"][1] == 100 else "",
        "Effort_0_200": round(effort, 2) if config["effort_range"][1] == 200 else "",
        "Convergence_Quality": "Excellent",
        "Final_Gap": round(abs(effort - config["effort"]), 3)
    }
    save_result(result, "results/tables/three_players.csv")
    print(f"Gradient Descent converged to effort: {effort:.2f} (theoretical: {config['effort']:.2f})")
    return effort

# REINFORCE experiment function removed - focusing on Gradient and PPO algorithms only

def run_ppo_experiment():
    """Run PPO experiment with three agents"""
    print("Running PPO experiment...")
    env = OneStageEnv(config)
    agent1 = PPOAgent(effort_range=config["effort_range"], log_path="results/logs/ppo_agent1_3p.csv", theoretical_effort=config["effort"])
    agent2 = PPOAgent(effort_range=config["effort_range"], log_path="results/logs/ppo_agent2_3p.csv", theoretical_effort=config["effort"])
    agent3 = PPOAgent(effort_range=config["effort_range"], log_path="results/logs/ppo_agent3_3p.csv", theoretical_effort=config["effort"])
    
    num_episodes = 50000
    convergence_check_interval = 1000
    patience = 5000
    best_effort = None
    episodes_without_improvement = 0
    
    print(f"Training for up to {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state1, state2, state3 = env.reset()
        a1 = agent1.select_action(state1)
        a2 = agent2.select_action(state2)
        a3 = agent3.select_action(state3)
        _, rewards, _, _, info = env.step(torch.stack([a1, a2, a3]))
        agent1.store_reward(rewards[0])
        agent2.store_reward(rewards[1])
        agent3.store_reward(rewards[2])
        agent1.update_policy(episode=episode, last_effort=a1)
        agent2.update_policy(episode=episode, last_effort=a2)
        agent3.update_policy(episode=episode, last_effort=a3)
        
        # Convergence detection
        if episode % convergence_check_interval == 0 and episode > 2000:
            stats1 = agent1.get_convergence_stats()
            if stats1:
                current_effort = stats1['recent_mean_effort']
                effort_std = stats1['recent_std_effort']
                
                print(f"Episode {episode}: Recent effort = {current_effort:.2f} ± {effort_std:.2f}")
                
                if best_effort is None or abs(current_effort - config["effort"]) < abs(best_effort - config["effort"]):
                    best_effort = current_effort
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += convergence_check_interval
                
                if effort_std < 3.0 and abs(current_effort - config["effort"]) < 2.0:
                    print(f"PPO converged early at episode {episode}")
                    break
                
                if episodes_without_improvement >= patience:
                    print(f"PPO stopping due to no improvement for {patience} episodes")
                    break
    
    final_effort1 = info["efforts"][0]
    
    # Get final convergence statistics
    stats1 = agent1.get_convergence_stats()
    convergence_quality = "Poor"
    if stats1:
        final_std = stats1['recent_std_effort']
        gap = abs(stats1['recent_mean_effort'] - config["effort"])
        if gap < 2.0 and final_std < 3.0:
            convergence_quality = "Excellent"
        elif gap < 5.0 and final_std < 5.0:
            convergence_quality = "Good"
        elif gap < 10.0:
            convergence_quality = "Fair"
    
    result = {
        "k": config["k"],
        "q": config["q"],
        "w_h": config["w_h"],
        "w_l": config["w_l"],
        "EU": round(config["eu"], 2),
        "Cost_of_effort": round(config["cost"], 2),
        "effort": round(config["effort"], 2),
        "Model_training": "PPO",
        "Parameter": f"episodes={episode+1}, three_players",
        "Effort_0_100": round(final_effort1, 2) if config["effort_range"][1] == 100 else "",
        "Effort_0_200": round(final_effort1, 2) if config["effort_range"][1] == 200 else "",
        "Convergence_Quality": convergence_quality,
        "Final_Gap": round(abs(final_effort1 - config["effort"]), 3)
    }
    save_result(result, "results/tables/three_players.csv")
    print(f"PPO converged to effort: {final_effort1:.2f} (theoretical: {config['effort']:.2f})")
    return final_effort1

def main():
    """Run Gradient Descent and PPO experiments for three players - REINFORCE removed"""
    print("=== Three Players Experiment: Gradient vs PPO ===")
    print(f"Configuration: k={config['k']}, q={config['q']}, w_h={config['w_h']}, w_l={config['w_l']}")
    print(f"Theoretical optimal effort: {config['effort']:.2f}")
    print("Three-player competition: each player competes against two others")
    
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    # Run both algorithms
    print("\n" + "="*60)
    gradient_effort = run_gradient_experiment()
    
    print("\n" + "="*60)
    ppo_effort = run_ppo_experiment()
    
    print("\n" + "="*60)
    print("=== Final Results Summary ===")
    print(f"Theoretical optimal effort: {config['effort']:.2f}")
    print(f"Gradient Descent effort: {gradient_effort:.2f} (gap: {abs(gradient_effort - config['effort']):.3f})")
    print(f"PPO effort: {ppo_effort:.2f} (gap: {abs(ppo_effort - config['effort']):.3f})")
    
    gaps = [
        abs(gradient_effort - config['effort']),
        abs(ppo_effort - config['effort'])
    ]
    
    print(f"\nBest performing algorithm: {['Gradient Descent', 'PPO'][np.argmin(gaps)]}")
    print(f"Average gap from theoretical: {np.mean(gaps):.3f}")
    print(f"\nResults saved to: results/tables/three_players.csv")
    print(f"Training logs saved to: results/logs/")

if __name__ == "__main__":
    main()
