#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.enhanced_ppo_agent import EnhancedPPOAgent, ContinuousActionSpace

def debug_ppo():
    """Debug PPO agent step by step"""
    print("Creating action space...")
    action_space = ContinuousActionSpace(low=0.0, high=100.0)
    
    print("Creating agent...")
    agent = EnhancedPPOAgent(
        action_space=action_space,
        gae_lambda=0.95  # Use exact same config as test
    )
    
    print("Creating state...")
    state = torch.tensor([[0.0]], dtype=torch.float32)
    
    def check_network_health(step_name):
        """Check if network weights are healthy"""
        first_layer = agent.policy_network.layers[0]
        weight_nan = torch.isnan(first_layer.weight).any()
        bias_nan = torch.isnan(first_layer.bias).any()
        print(f"{step_name}: Weight NaN: {weight_nan}, Bias NaN: {bias_nan}")
        return weight_nan or bias_nan
    
    check_network_health("Initial")
    
    print("Testing single action selection...")
    action = agent.select_action(state)
    print(f"Action: {action}, shape: {action.shape}")
    check_network_health("After first action")
    
    print("Storing reward...")
    agent.store_reward(torch.tensor(1.0))
    check_network_health("After first reward")
    
    print("Testing multiple steps...")
    for i in range(10):
        print(f"Step {i+1}...")
        action = agent.select_action(state)
        print(f"  Action: {action.item():.4f}")
        if check_network_health(f"  After action {i+1}"):
            print(f"  NaN detected at step {i+1} action selection!")
            break
            
        agent.store_reward(torch.tensor(float(i + 2)))  # i+2 because we already stored 1.0
        if check_network_health(f"  After reward {i+1}"):
            print(f"  NaN detected at step {i+1} reward storage!")
            break
    
    print("Final network check before update...")
    if not check_network_health("Before update"):
        print("Network is healthy before update, testing policy update...")
        
        # Debug the actual values being used in the update
        print("Debugging policy update values...")
        states = torch.stack(agent.states)
        actions = torch.stack(agent.actions)
        old_log_probs = torch.stack(agent.log_probs)
        rewards = torch.tensor(agent.rewards, dtype=torch.float32)
        
        print(f"States shape: {states.shape}, unique values: {torch.unique(states)}")
        print(f"Actions shape: {actions.shape}, min: {actions.min():.4f}, max: {actions.max():.4f}")
        print(f"Old log_probs shape: {old_log_probs.shape}, min: {old_log_probs.min():.4f}, max: {old_log_probs.max():.4f}")
        print(f"Rewards shape: {rewards.shape}, min: {rewards.min():.4f}, max: {rewards.max():.4f}")
        
        # Check individual stored items
        print(f"First few stored actions shapes: {[a.shape for a in agent.actions[:3]]}")
        print(f"First few stored log_probs shapes: {[lp.shape for lp in agent.log_probs[:3]]}")
        print(f"First few stored states shapes: {[s.shape for s in agent.states[:3]]}")
        
        # Test forward pass
        with torch.no_grad():
            mean, std = agent.policy_network.forward(states)
            print(f"Mean shape: {mean.shape}, min: {mean.min():.4f}, max: {mean.max():.4f}")
            print(f"Std shape: {std.shape}, min: {std.min():.4f}, max: {std.max():.4f}")
            
            dist = agent.policy_network.get_action_distribution(states)
            log_probs = dist.log_prob(actions)
            print(f"Log probs shape: {log_probs.shape}, min: {log_probs.min():.4f}, max: {log_probs.max():.4f}")
            print(f"Log probs has inf: {torch.isinf(log_probs).any()}")
            print(f"Log probs has nan: {torch.isnan(log_probs).any()}")
            
            # Test with individual action
            if len(agent.actions) > 0:
                single_state = agent.states[0]
                single_action = agent.actions[0]
                print(f"Single state shape: {single_state.shape}")
                print(f"Single action shape: {single_action.shape}")
                single_dist = agent.policy_network.get_action_distribution(single_state)
                single_log_prob = single_dist.log_prob(single_action)
                print(f"Single log prob shape: {single_log_prob.shape}, value: {single_log_prob.item():.4f}")
        
        try:
            losses = agent.update_policy(episode=1)
            print(f"Update successful: {losses}")
        except Exception as e:
            print(f"Update failed: {e}")
    else:
        print("Network already corrupted before update!")

if __name__ == "__main__":
    debug_ppo() 