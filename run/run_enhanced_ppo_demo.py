#!/usr/bin/env python3
"""
Enhanced PPO Agent Demonstration

This script demonstrates the enhanced PPO agent with both discrete and continuous action spaces,
showcasing the improved modularization and advanced features.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.enhanced_ppo_agent import (
    EnhancedPPOAgent, ContinuousActionSpace, DiscreteActionSpace
)

def continuous_action_demo():
    """Demonstrate PPO agent with continuous action space"""
    print("=" * 60)
    print("ENHANCED PPO AGENT - CONTINUOUS ACTION SPACE DEMO")
    print("=" * 60)
    
    # Create continuous action space (effort range 0-100)
    action_space = ContinuousActionSpace(low=0.0, high=100.0, shape=(1,))
    
    # Create enhanced PPO agent
    agent = EnhancedPPOAgent(
        action_space=action_space,
        lr=1e-3,
        hidden_dim=128,
        num_layers=3,
        activation='tanh',
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        gae_lambda=0.95,
        separate_networks=True,
        reward_normalization=True,
        log_path="logs/enhanced_ppo_continuous_demo.csv"
    )
    
    print(f"Agent Configuration:")
    print(f"  Action Space: {action_space.action_type} [{action_space.low}, {action_space.high}]")
    print(f"  Policy Network: {type(agent.policy_network).__name__}")
    print(f"  Value Network: {type(agent.value_network).__name__}")
    print(f"  Separate Networks: {agent.separate_networks}")
    print(f"  Reward Normalization: {agent.reward_normalization}")
    print()
    
    # Training parameters
    num_episodes = 100
    target_effort = 87.5  # Theoretical optimum
    
    # Storage for results
    efforts = []
    rewards = []
    losses = []
    
    print("Training Progress:")
    print("Episode | Effort | Reward | Policy Loss | Value Loss | Entropy")
    print("-" * 65)
    
    for episode in range(num_episodes):
        # Simple state (dummy state for this demo)
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Agent selects action
        action = agent.select_action(state)
        effort = action.item()
        
        # Simple reward function: closer to target is better
        distance = abs(effort - target_effort)
        reward = max(0, 1.0 - distance / 100.0)  # Reward in [0, 1]
        
        # Store reward and update policy
        agent.store_reward(torch.tensor(reward))
        loss_dict = agent.update_policy(episode=episode)
        
        # Store results
        efforts.append(effort)
        rewards.append(reward)
        if loss_dict:
            losses.append(loss_dict)
        
        # Print progress every 10 episodes
        if episode % 10 == 0:
            policy_loss = loss_dict.get('policy_loss', 0) if loss_dict else 0
            value_loss = loss_dict.get('value_loss', 0) if loss_dict else 0
            entropy = loss_dict.get('entropy', 0) if loss_dict else 0
            
            print(f"{episode:7d} | {effort:6.2f} | {reward:6.3f} | {policy_loss:11.4f} | {value_loss:10.4f} | {entropy:7.4f}")
    
    # Final statistics
    final_efforts = efforts[-10:]
    final_rewards = rewards[-10:]
    
    print()
    print("Final Results (last 10 episodes):")
    print(f"  Mean Effort: {np.mean(final_efforts):.2f} ± {np.std(final_efforts):.2f}")
    print(f"  Target Effort: {target_effort:.2f}")
    print(f"  Mean Distance: {np.mean([abs(e - target_effort) for e in final_efforts]):.2f}")
    print(f"  Mean Reward: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
    
    # Get convergence statistics
    stats = agent.get_convergence_stats()
    if stats:
        print(f"  Convergence Stats:")
        print(f"    Recent Mean Effort: {stats['recent_mean_effort']:.2f}")
        print(f"    Recent Std Effort: {stats['recent_std_effort']:.2f}")
        print(f"    Recent Mean Reward: {stats['recent_mean_reward']:.3f}")
    
    return efforts, rewards, losses

def discrete_action_demo():
    """Demonstrate PPO agent with discrete action space"""
    print("\n" + "=" * 60)
    print("ENHANCED PPO AGENT - DISCRETE ACTION SPACE DEMO")
    print("=" * 60)
    
    # Create discrete action space (10 discrete effort levels)
    action_space = DiscreteActionSpace(n=10)
    
    # Create enhanced PPO agent
    agent = EnhancedPPOAgent(
        action_space=action_space,
        lr=1e-3,
        hidden_dim=64,
        num_layers=2,
        activation='relu',
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.02,  # Higher entropy for exploration
        gae_lambda=0.95,
        separate_networks=False,  # Use shared networks
        reward_normalization=False,
        log_path="logs/enhanced_ppo_discrete_demo.csv"
    )
    
    print(f"Agent Configuration:")
    print(f"  Action Space: {action_space.action_type} (n={action_space.n})")
    print(f"  Policy Network: {type(agent.policy_network).__name__}")
    print(f"  Value Network: {type(agent.value_network).__name__}")
    print(f"  Separate Networks: {agent.separate_networks}")
    print(f"  Reward Normalization: {agent.reward_normalization}")
    print()
    
    # Training parameters
    num_episodes = 100
    # Map discrete actions to effort levels (0-9 -> 10-100 effort)
    effort_mapping = {i: 10 + i * 10 for i in range(10)}
    target_action = 8  # Corresponds to effort level 90 (close to 87.5)
    
    # Storage for results
    actions = []
    efforts = []
    rewards = []
    losses = []
    
    print("Training Progress:")
    print("Episode | Action | Effort | Reward | Policy Loss | Value Loss | Entropy")
    print("-" * 75)
    
    for episode in range(num_episodes):
        # Simple state (dummy state for this demo)
        state = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Agent selects action
        action = agent.select_action(state)
        action_idx = action.item()
        effort = effort_mapping[action_idx]
        
        # Reward function: prefer actions closer to target
        distance = abs(action_idx - target_action)
        reward = max(0, 1.0 - distance / 10.0)  # Reward in [0, 1]
        
        # Store reward and update policy
        agent.store_reward(torch.tensor(reward))
        loss_dict = agent.update_policy(episode=episode)
        
        # Store results
        actions.append(action_idx)
        efforts.append(effort)
        rewards.append(reward)
        if loss_dict:
            losses.append(loss_dict)
        
        # Print progress every 10 episodes
        if episode % 10 == 0:
            policy_loss = loss_dict.get('policy_loss', 0) if loss_dict else 0
            value_loss = loss_dict.get('value_loss', 0) if loss_dict else 0
            entropy = loss_dict.get('entropy', 0) if loss_dict else 0
            
            print(f"{episode:7d} | {action_idx:6d} | {effort:6.1f} | {reward:6.3f} | {policy_loss:11.4f} | {value_loss:10.4f} | {entropy:7.4f}")
    
    # Final statistics
    final_actions = actions[-10:]
    final_rewards = rewards[-10:]
    
    print()
    print("Final Results (last 10 episodes):")
    print(f"  Most Common Action: {max(set(final_actions), key=final_actions.count)}")
    print(f"  Target Action: {target_action}")
    print(f"  Action Distribution: {[final_actions.count(i) for i in range(10)]}")
    print(f"  Mean Reward: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
    
    return actions, efforts, rewards, losses

def compare_action_spaces():
    """Compare performance between continuous and discrete action spaces"""
    print("\n" + "=" * 60)
    print("ACTION SPACE COMPARISON")
    print("=" * 60)
    
    # Run both demos
    print("Running continuous action space demo...")
    cont_efforts, cont_rewards, cont_losses = continuous_action_demo()
    
    print("\nRunning discrete action space demo...")
    disc_actions, disc_efforts, disc_rewards, disc_losses = discrete_action_demo()
    
    # Compare final performance
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Continuous space results
    cont_final_rewards = cont_rewards[-10:]
    cont_final_efforts = cont_efforts[-10:]
    target_effort = 87.5
    
    print("Continuous Action Space:")
    print(f"  Final Mean Reward: {np.mean(cont_final_rewards):.3f}")
    print(f"  Final Mean Effort: {np.mean(cont_final_efforts):.2f}")
    print(f"  Distance from Target: {np.mean([abs(e - target_effort) for e in cont_final_efforts]):.2f}")
    
    # Discrete space results
    disc_final_rewards = disc_rewards[-10:]
    disc_final_efforts = disc_efforts[-10:]
    
    print("\nDiscrete Action Space:")
    print(f"  Final Mean Reward: {np.mean(disc_final_rewards):.3f}")
    print(f"  Final Mean Effort: {np.mean(disc_final_efforts):.2f}")
    print(f"  Distance from Target: {np.mean([abs(e - target_effort) for e in disc_final_efforts]):.2f}")
    
    # Learning curves comparison
    if len(cont_losses) > 0 and len(disc_losses) > 0:
        print("\nLearning Curves:")
        print(f"  Continuous - Final Policy Loss: {cont_losses[-1]['policy_loss']:.4f}")
        print(f"  Discrete - Final Policy Loss: {disc_losses[-1]['policy_loss']:.4f}")
        print(f"  Continuous - Final Entropy: {cont_losses[-1]['entropy']:.4f}")
        print(f"  Discrete - Final Entropy: {disc_losses[-1]['entropy']:.4f}")

def test_advanced_features():
    """Test advanced features of the enhanced PPO agent"""
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("=" * 60)
    
    # Test different configurations
    configs = [
        {
            'name': 'Separate Networks + Reward Norm',
            'separate_networks': True,
            'reward_normalization': True,
            'lr_schedule': 'cosine_annealing'
        },
        {
            'name': 'Shared Networks + No Reward Norm',
            'separate_networks': False,
            'reward_normalization': False,
            'lr_schedule': 'step'
        },
        {
            'name': 'High Entropy + Layer Norm',
            'separate_networks': True,
            'reward_normalization': True,
            'entropy_coef': 0.05,
            'lr_schedule': 'constant'
        }
    ]
    
    action_space = ContinuousActionSpace(low=0.0, high=100.0)
    target_effort = 87.5
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 40)
        
        # Create agent with specific configuration
        agent = EnhancedPPOAgent(
            action_space=action_space,
            lr=1e-3,
            hidden_dim=64,
            separate_networks=config.get('separate_networks', True),
            reward_normalization=config.get('reward_normalization', True),
            entropy_coef=config.get('entropy_coef', 0.01),
            lr_schedule=config.get('lr_schedule', 'constant')
        )
        
        # Quick training
        rewards = []
        for episode in range(20):
            state = torch.tensor([[0.0]], dtype=torch.float32)
            action = agent.select_action(state)
            effort = action.item()
            
            distance = abs(effort - target_effort)
            reward = max(0, 1.0 - distance / 100.0)
            
            agent.store_reward(torch.tensor(reward))
            agent.update_policy(episode=episode)
            rewards.append(reward)
        
        # Report results
        final_reward = np.mean(rewards[-5:])
        print(f"  Final Mean Reward: {final_reward:.3f}")
        
        # Check convergence stats
        stats = agent.get_convergence_stats()
        if stats:
            print(f"  Recent Mean Effort: {stats['recent_mean_effort']:.2f}")
            print(f"  Distance from Target: {abs(stats['recent_mean_effort'] - target_effort):.2f}")

def main():
    """Main demonstration function"""
    print("Enhanced PPO Agent Demonstration")
    print("This demo showcases the enhanced PPO agent with both discrete and continuous action spaces")
    print("and demonstrates advanced features like modular networks, GAE, and improved loss computation.")
    print()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run demonstrations
    try:
        # Basic demos
        compare_action_spaces()
        
        # Advanced features
        test_advanced_features()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("Key Features Demonstrated:")
        print("✓ Continuous and discrete action space support")
        print("✓ Modular policy and value networks")
        print("✓ Generalized Advantage Estimation (GAE)")
        print("✓ PPO clipped objective and value loss")
        print("✓ Separate vs shared network architectures")
        print("✓ Reward normalization")
        print("✓ Learning rate scheduling")
        print("✓ Gradient clipping")
        print("✓ Comprehensive logging")
        print("✓ Convergence monitoring")
        print()
        print("Log files saved to 'logs/' directory")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 