import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from config.asymmetric_cost_two_players import config
from envs.asymmetric_cost_env import AsymmetricCostEnv
from agents.asymmetric_gradient_solver import asymmetric_gradient_descent_solver
# REINFORCE agent removed - focusing on Gradient and PPO algorithms
from agents.ppo_agent import PPOAgent
from utils.logger import save_result

def run_asymmetric_gradient_experiment():
    """Run gradient descent experiment with asymmetric costs"""
    print("Running Asymmetric Gradient Descent experiment...")
    env = AsymmetricCostEnv(config)
    
    print(f"Cost parameters: k1={config['k1']}, k2={config['k2']}")
    print(f"Theoretical efforts: e1*={config['effort1']:.2f}, e2*={config['effort2']:.2f}")
    
    efforts, utilities, costs = asymmetric_gradient_descent_solver(env, lr=0.1, steps=100000, eps=1e-3)
    
    # Calculate gaps from theoretical values
    gaps = [abs(efforts[i] - config['theoretical_efforts'][i]) for i in range(2)]
    
    result = {
        "k1": config["k1"],
        "k2": config["k2"],
        "q": config["q"],
        "w_h": config["w_h"],
        "w_l": config["w_l"],
        "Theoretical_e1": round(config["effort1"], 2),
        "Theoretical_e2": round(config["effort2"], 2),
        "Model_training": "Asymmetric_Gradient",
        "Parameter": f"lr=0.1, steps=100000, eps=1e-3",
        "Effort1": round(efforts[0], 2),
        "Effort2": round(efforts[1], 2),
        "Gap1": round(gaps[0], 3),
        "Gap2": round(gaps[1], 3),
        "Utility1": round(utilities[0], 3),
        "Utility2": round(utilities[1], 3),
        "Cost1": round(costs[0], 3),
        "Cost2": round(costs[1], 3),
        "Convergence_Quality": "Excellent" if max(gaps) < 1.0 else "Good" if max(gaps) < 3.0 else "Fair"
    }
    save_result(result, "results/tables/asymmetric_cost.csv")
    
    print(f"Gradient Descent converged to efforts: e1={efforts[0]:.2f}, e2={efforts[1]:.2f}")
    print(f"Theoretical efforts: e1*={config['effort1']:.2f}, e2*={config['effort2']:.2f}")
    print(f"Gaps: {gaps[0]:.3f}, {gaps[1]:.3f}")
    
    return efforts

# REINFORCE experiment function removed - focusing on Gradient and PPO algorithms only

def run_asymmetric_ppo_experiment():
    """Run PPO experiment with asymmetric costs"""
    print("Running Asymmetric PPO experiment...")
    env = AsymmetricCostEnv(config)
    
    # Create agents with different theoretical efforts
    agent1 = PPOAgent(
        lr=0.0001,
        effort_range=config["effort_range"], 
        log_path="results/logs/asymmetric_ppo_agent1.csv", 
        theoretical_effort=config["effort1"]  # Player 1's theoretical effort
    )
    agent2 = PPOAgent(
        lr=0.0001,
        effort_range=config["effort_range"], 
        log_path="results/logs/asymmetric_ppo_agent2.csv", 
        theoretical_effort=config["effort2"]  # Player 2's theoretical effort
    )
    
    num_episodes = 20000
    convergence_check_interval = 1000
    patience = 4000
    best_gaps = [None, None]
    episodes_without_improvement = 0
    
    print(f"Training for up to {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        states = env.reset()
        
        # Each agent selects action
        a1 = agent1.select_action(states[0])
        a2 = agent2.select_action(states[1])
        
        # Environment step
        _, rewards, _, _, info = env.step(torch.stack([a1, a2]))
        
        # Store rewards
        agent1.store_reward(rewards[0])
        agent2.store_reward(rewards[1])
        
        # Update policies
        agent1.update_policy(gamma=0.99, episode=episode, last_effort=a1)
        agent2.update_policy(gamma=0.99, episode=episode, last_effort=a2)
        
        # Check convergence
        if episode % convergence_check_interval == 0 and episode > 3000:
            stats1 = agent1.get_convergence_stats()
            stats2 = agent2.get_convergence_stats()
            
            if stats1 and stats2:
                current_efforts = [stats1['recent_mean_effort'], stats2['recent_mean_effort']]
                effort_stds = [stats1['recent_std_effort'], stats2['recent_std_effort']]
                gaps = [abs(current_efforts[i] - config['theoretical_efforts'][i]) for i in range(2)]
                
                print(f"Episode {episode}: Efforts = [{current_efforts[0]:.2f}±{effort_stds[0]:.2f}, {current_efforts[1]:.2f}±{effort_stds[1]:.2f}], Gaps = [{gaps[0]:.2f}, {gaps[1]:.2f}]")
                
                # Check for improvement
                improved = False
                for i in range(2):
                    if best_gaps[i] is None or gaps[i] < best_gaps[i]:
                        best_gaps[i] = gaps[i]
                        improved = True
                
                if improved:
                    episodes_without_improvement = 0
                    print(f"  → New best gaps: [{best_gaps[0]:.2f}, {best_gaps[1]:.2f}]")
                else:
                    episodes_without_improvement += convergence_check_interval
                
                # Early stopping conditions
                if max(effort_stds) < 3.0 and max(gaps) < 3.0:
                    print(f"PPO converged early at episode {episode} with good performance!")
                    break
                
                if episodes_without_improvement >= patience:
                    print(f"PPO stopping due to no improvement for {patience} episodes")
                    break
    
    # Get final results
    final_efforts = [info["efforts"][0], info["efforts"][1]]
    final_gaps = [abs(final_efforts[i] - config['theoretical_efforts'][i]) for i in range(2)]
    
    # Determine convergence quality
    convergence_quality = "Poor"
    if max(final_gaps) < 3.0:
        convergence_quality = "Excellent"
    elif max(final_gaps) < 6.0:
        convergence_quality = "Good"
    elif max(final_gaps) < 10.0:
        convergence_quality = "Fair"
    
    result = {
        "k1": config["k1"],
        "k2": config["k2"],
        "q": config["q"],
        "w_h": config["w_h"],
        "w_l": config["w_l"],
        "Theoretical_e1": round(config["effort1"], 2),
        "Theoretical_e2": round(config["effort2"], 2),
        "Model_training": "Asymmetric_PPO",
        "Parameter": f"episodes={episode+1}, lr=0.0001, gamma=0.99",
        "Effort1": round(final_efforts[0], 2),
        "Effort2": round(final_efforts[1], 2),
        "Gap1": round(final_gaps[0], 3),
        "Gap2": round(final_gaps[1], 3),
        "Convergence_Quality": convergence_quality
    }
    save_result(result, "results/tables/asymmetric_cost.csv")
    
    print(f"PPO converged to efforts: e1={final_efforts[0]:.2f}, e2={final_efforts[1]:.2f}")
    print(f"Gaps: {final_gaps[0]:.3f}, {final_gaps[1]:.3f}")
    
    return final_efforts

def main():
    """Run asymmetric cost experiments: Gradient vs PPO - REINFORCE removed"""
    print("=== Asymmetric Cost Parameters Experiment: Gradient vs PPO ===")
    print(f"Configuration: k1={config['k1']}, k2={config['k2']}, q={config['q']}")
    print(f"Theoretical optimal efforts: e1*={config['effort1']:.2f}, e2*={config['effort2']:.2f}")
    print("Goal: Compare Gradient and PPO algorithms with different cost parameters (k1 < k2)")
    
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    # Run both algorithms
    print("\n" + "="*60)
    gradient_efforts = run_asymmetric_gradient_experiment()
    
    print("\n" + "="*60)
    ppo_efforts = run_asymmetric_ppo_experiment()
    
    print("\n" + "="*60)
    print("=== Final Results Summary ===")
    print(f"Theoretical optimal efforts: e1*={config['effort1']:.2f}, e2*={config['effort2']:.2f}")
    print(f"Gradient Descent efforts: e1={gradient_efforts[0]:.2f}, e2={gradient_efforts[1]:.2f}")
    print(f"PPO efforts: e1={ppo_efforts[0]:.2f}, e2={ppo_efforts[1]:.2f}")
    
    # Calculate and display gaps
    gradient_gaps = [abs(gradient_efforts[i] - config['theoretical_efforts'][i]) for i in range(2)]
    ppo_gaps = [abs(ppo_efforts[i] - config['theoretical_efforts'][i]) for i in range(2)]
    
    print(f"\nGaps from theoretical:")
    print(f"Gradient Descent: [{gradient_gaps[0]:.3f}, {gradient_gaps[1]:.3f}] (avg: {np.mean(gradient_gaps):.3f})")
    print(f"PPO: [{ppo_gaps[0]:.3f}, {ppo_gaps[1]:.3f}] (avg: {np.mean(ppo_gaps):.3f})")
    
    avg_gaps = [np.mean(gradient_gaps), np.mean(ppo_gaps)]
    best_algorithm = ["Gradient Descent", "PPO"][np.argmin(avg_gaps)]
    print(f"\nBest performing algorithm: {best_algorithm}")
    print(f"\nResults saved to: results/tables/asymmetric_cost.csv")
    print(f"Training logs saved to: results/logs/")

if __name__ == "__main__":
    main() 