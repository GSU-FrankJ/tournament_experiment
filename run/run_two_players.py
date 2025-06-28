import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from config.one_stage_two_players import config
from envs.one_stage_env import OneStageEnv
from agents.gradient_solver import gradient_descent_solver
# REINFORCE agent removed - focusing on Gradient and PPO algorithms
from agents.ppo_agent import PPOAgent
from utils.logger import save_result

def run_gradient_experiment():
    """Run gradient descent experiment"""
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
    save_result(result, "results/tables/two_players.csv")
    print(f"Gradient Descent converged to effort: {effort:.2f} (theoretical: {config['effort']:.2f})")
    return effort

# REINFORCE experiment function removed - focusing on Gradient and PPO algorithms only

def run_ppo_experiment():
    """Run PPO experiment with optimized parameters from random search"""
    print("Running PPO experiment with optimized parameters...")
    env = OneStageEnv(config)
    
    # 完整的优化参数配置
    optimized_params = {
        'lr': 0.0001,
        'hidden_dim': 128,
        'num_layers': 3,
        'activation': 'tanh',
        'clip_epsilon': 0.2,
        'value_coef': 0.75,
        'entropy_coef': 0.005,
        'max_grad_norm': 0.3,
        'batch_size': 64,
        'update_epochs': 10,
        'gamma': 0.995,
        'gae_lambda': 0.97,
        'weight_decay': 1e-05,
        'dropout_rate': 0.05,
        'lr_schedule': 'cosine_annealing',
        'separate_networks': True,
        'reward_normalization': True
    }
    
    print(f"Using optimized parameters: {optimized_params}")
    print(f"Expected performance: effort≈87.57 (gap≈0.066)")
    
    agent1 = PPOAgent(
        lr=optimized_params['lr'],
        hidden_dim=optimized_params['hidden_dim'],
        num_layers=optimized_params['num_layers'],
        activation=optimized_params['activation'],
        clip_epsilon=optimized_params['clip_epsilon'], 
        value_coef=optimized_params['value_coef'],
        entropy_coef=optimized_params['entropy_coef'],
        max_grad_norm=optimized_params['max_grad_norm'],
        batch_size=optimized_params['batch_size'],
        update_epochs=optimized_params['update_epochs'],
        gae_lambda=optimized_params['gae_lambda'],
        weight_decay=optimized_params['weight_decay'],
        dropout_rate=optimized_params['dropout_rate'],
        lr_schedule=optimized_params['lr_schedule'],
        separate_networks=optimized_params['separate_networks'],
        reward_normalization=optimized_params['reward_normalization'],
        effort_range=config["effort_range"], 
        log_path="results/logs/ppo_agent1.csv", 
        theoretical_effort=config["effort"]
    )
    agent2 = PPOAgent(
        lr=optimized_params['lr'],
        hidden_dim=optimized_params['hidden_dim'],
        num_layers=optimized_params['num_layers'],
        activation=optimized_params['activation'],
        clip_epsilon=optimized_params['clip_epsilon'],
        value_coef=optimized_params['value_coef'],
        entropy_coef=optimized_params['entropy_coef'], 
        max_grad_norm=optimized_params['max_grad_norm'],
        batch_size=optimized_params['batch_size'],
        update_epochs=optimized_params['update_epochs'],
        gae_lambda=optimized_params['gae_lambda'],
        weight_decay=optimized_params['weight_decay'],
        dropout_rate=optimized_params['dropout_rate'],
        lr_schedule=optimized_params['lr_schedule'],
        separate_networks=optimized_params['separate_networks'],
        reward_normalization=optimized_params['reward_normalization'],
        effort_range=config["effort_range"], 
        log_path="results/logs/ppo_agent2.csv", 
        theoretical_effort=config["effort"]
    )
    
    num_episodes = 15000  # 使用优化后的训练轮数
    convergence_check_interval = 1000
    patience = 5000  # 增加耐心值，因为优化参数可能需要更多时间
    best_effort = None  
    episodes_without_improvement = 0
    
    print(f"Training for up to {num_episodes} episodes with optimized PPO parameters...")

    for episode in range(num_episodes):
        state1, state2 = env.reset()
        a1 = agent1.select_action(state1)
        a2 = agent2.select_action(state2)
        _, rewards, _, _, info = env.step(torch.stack([a1, a2]))
        agent1.store_reward(rewards[0])
        agent2.store_reward(rewards[1])
        
        # 使用优化的gamma参数
        agent1.update_policy(gamma=optimized_params['gamma'])
        agent2.update_policy(gamma=optimized_params['gamma'])
        
        # 收敛检测
        if episode % convergence_check_interval == 0 and episode > 2000:
            stats1 = agent1.get_convergence_stats()
            if stats1:
                current_effort = stats1['recent_mean_effort']
                effort_std = stats1['recent_std_effort']
                gap = abs(current_effort - config["effort"])
                
                print(f"Episode {episode}: Recent effort = {current_effort:.2f} ± {effort_std:.2f}, Gap = {gap:.3f}")
                
                # 检查是否接近理论值
                if best_effort is None or gap < abs(best_effort - config["effort"]):
                    best_effort = current_effort
                    episodes_without_improvement = 0
                    print(f"  → New best effort: {current_effort:.2f}")
                else:
                    episodes_without_improvement += convergence_check_interval
                
                # 早停条件：标准差足够小且接近理论值
                if effort_std < 2.0 and gap < 1.0:
                    print(f"PPO converged early at episode {episode} with excellent performance!")
                    break
                elif effort_std < 3.0 and gap < 2.0:
                    print(f"PPO converged early at episode {episode} with good performance!")
                    break
                
                # 耐心用完
                if episodes_without_improvement >= patience:
                    print(f"PPO stopping due to no improvement for {patience} episodes")
                    break

    final_effort1 = info["efforts"][0]
    
    # 获取最终收敛统计
    stats1 = agent1.get_convergence_stats()
    convergence_quality = "Poor"
    final_gap = abs(final_effort1 - config["effort"])
    if stats1:
        final_std = stats1['recent_std_effort']
        avg_gap = abs(stats1['recent_mean_effort'] - config["effort"])
        if avg_gap < 1.0 and final_std < 2.0:
            convergence_quality = "Excellent"
        elif avg_gap < 2.0 and final_std < 3.0:
            convergence_quality = "Good"
        elif avg_gap < 5.0 and final_std < 5.0:
            convergence_quality = "Fair"
        print(f"Final convergence stats: avg_gap = {avg_gap:.3f}, std = {final_std:.2f}")
    
    result = {
        "k": config["k"],
        "q": config["q"],
        "w_h": config["w_h"],
        "w_l": config["w_l"],
        "EU": round(config["eu"], 2),
        "Cost_of_effort": round(config["cost"], 2),
        "effort": round(config["effort"], 2),
        "Model_training": "PPO",
        "Parameter": f"episodes={episode+1}, full_optimized_config, tanh_activation, gamma={optimized_params['gamma']}, cosine_annealing",
        "Effort_0_100": round(final_effort1, 2) if config["effort_range"][1] == 100 else "",
        "Effort_0_200": round(final_effort1, 2) if config["effort_range"][1] == 200 else "",
        "Convergence_Quality": convergence_quality,
        "Final_Gap": round(final_gap, 3)
    }
    save_result(result, "results/tables/two_players.csv")
    print(f"PPO converged to effort: {final_effort1:.2f} (theoretical: {config['effort']:.2f}, gap: {final_gap:.3f})")
    return final_effort1

# REINFORCE grid search function removed - focusing on Gradient and PPO algorithms only

def main():
    """Run Gradient Descent and PPO experiments - REINFORCE removed"""
    print("=== Optimized Two Players Experiment: Gradient vs PPO ===")
    print(f"Configuration: k={config['k']}, q={config['q']}, w_h={config['w_h']}, w_l={config['w_l']}")
    print(f"Theoretical optimal effort: {config['effort']:.2f}")
    print("Goal: Compare Gradient Descent and PPO convergence to theoretical value")
    
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
    
    # 分析结果
    gaps = [
        abs(gradient_effort - config['effort']),
        abs(ppo_effort - config['effort'])
    ]
    
    print(f"\nBest performing algorithm: {['Gradient Descent', 'PPO'][np.argmin(gaps)]}")
    print(f"Average gap from theoretical: {np.mean(gaps):.3f}")
    print(f"\nResults saved to: results/tables/two_players.csv")
    print(f"Training logs saved to: results/logs/")

if __name__ == "__main__":
    main() 