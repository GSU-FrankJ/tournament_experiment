#!/usr/bin/env python3
"""
Two-Stage Game Environment Demonstration

This script demonstrates the key features of the two-stage tournament environment:
1. Sequential decision making across two stages
2. Information revelation between stages
3. Weighted combination of stage outcomes
4. Inter-stage dependency modeling
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.two_stage_two_players import config
from envs.two_stage_env import TwoStageEnv

def demonstrate_basic_flow():
    """Demonstrate basic two-stage game flow"""
    print("=" * 60)
    print("BASIC TWO-STAGE GAME FLOW DEMONSTRATION")
    print("=" * 60)
    
    env = TwoStageEnv(config)
    
    print(f"Configuration:")
    print(f"  Stage 1 weight: {config['stage1_weight']}")
    print(f"  Stage 2 weight: {config['stage2_weight']}")
    print(f"  Stage 1 cost parameter (k1): {config['k1']}")
    print(f"  Stage 2 cost parameter (k2): {config['k2']}")
    print(f"  Information revelation: {config['information_revelation']}")
    print(f"  Theoretical efforts: Stage 1 = {config['stage1_effort']:.2f}, Stage 2 = {config['stage2_effort']:.2f}")
    print()
    
    # Reset environment
    initial_states = env.reset()
    print(f"Initial states: {[s.item() for s in initial_states]}")
    print(f"Current stage: {env.get_current_stage()}")
    print()
    
    # Stage 1: Players make initial effort decisions
    print("STAGE 1: Initial Effort Decisions")
    print("-" * 40)
    stage1_efforts = [55.0, 65.0]  # Player 1: 55, Player 2: 65
    stage1_actions = [torch.tensor([e]) for e in stage1_efforts]
    
    print(f"Player efforts: {stage1_efforts}")
    
    stage1_states, stage1_rewards, stage1_costs, stage1_done, stage1_info = env.step(stage1_actions)
    
    print(f"Stage 1 rewards: {[r.item() for r in stage1_rewards]}")
    print(f"Stage 1 costs: {[c.item() for c in stage1_costs]}")
    print(f"Stage 1 winner: Player {stage1_info['stage1_winner'] + 1}")
    print(f"Game done after Stage 1: {stage1_done}")
    print(f"Current stage: {env.get_current_stage()}")
    print()
    
    # Information revelation
    print("INFORMATION REVELATION BETWEEN STAGES")
    print("-" * 40)
    for player_id in range(2):
        info = env.get_information_state(player_id)
        print(f"Player {player_id + 1} information:")
        for key, value in info.items():
            if key != "player_id":
                print(f"  {key}: {value}")
        print()
    
    # Stage 2: Players make second-round decisions
    print("STAGE 2: Second-Round Decisions")
    print("-" * 40)
    stage2_efforts = [45.0, 50.0]  # Player 1: 45, Player 2: 50
    stage2_actions = [torch.tensor([e]) for e in stage2_efforts]
    
    print(f"Player efforts: {stage2_efforts}")
    
    final_states, final_rewards, final_costs, final_done, final_info = env.step(stage2_actions)
    
    stage2_rewards = [final_info[f'p{i+1}_stage2_utility'] for i in range(2)]
    stage2_costs = [final_info[f'p{i+1}_stage2_cost'] for i in range(2)]
    print(f"Stage 2 rewards: {stage2_rewards}")
    print(f"Stage 2 costs: {stage2_costs}")
    print(f"Stage 2 winner: Player {final_info['stage2_winner'] + 1}")
    print()
    
    # Final outcomes
    print("FINAL OUTCOMES (Weighted Combination)")
    print("-" * 40)
    final_reward_values = [r.item() for r in final_rewards]
    final_cost_values = [c.item() for c in final_costs]
    print(f"Final total rewards: {final_reward_values}")
    print(f"Final total costs: {final_cost_values}")
    print(f"Game done: {final_done}")
    print()
    
    # Detailed breakdown
    print("DETAILED BREAKDOWN")
    print("-" * 40)
    for i in range(2):
        player_num = i + 1
        stage1_utility = final_info[f'p{player_num}_stage1_utility']
        stage2_utility = final_info[f'p{player_num}_stage2_utility']
        stage1_cost = final_info[f'p{player_num}_stage1_cost']
        stage2_cost = final_info[f'p{player_num}_stage2_cost']
        total_cost = final_info[f'p{player_num}_total_cost']
        
        print(f"Player {player_num}:")
        print(f"  Stage 1 effort: {stage1_efforts[i]:.1f}, utility: {stage1_utility:.3f}, cost: {stage1_cost:.3f}")
        print(f"  Stage 2 effort: {stage2_efforts[i]:.1f}, utility: {stage2_utility:.3f}, cost: {stage2_cost:.3f}")
        print(f"  Total cost: {total_cost:.3f}")
        print(f"  Final weighted utility: {final_rewards[i].item():.3f}")
        print()

def demonstrate_information_scenarios():
    """Demonstrate different information revelation scenarios"""
    print("=" * 60)
    print("INFORMATION REVELATION SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        ("none", "No Information"),
        ("partial", "Partial Information"),
        ("full", "Full Information")
    ]
    
    for info_type, description in scenarios:
        print(f"\n{description.upper()} ({info_type})")
        print("-" * 40)
        
        # Create environment with specific information revelation
        test_config = config.copy()
        test_config["information_revelation"] = info_type
        env = TwoStageEnv(test_config)
        
        # Execute Stage 1
        env.reset()
        stage1_actions = [torch.tensor([60.0]), torch.tensor([70.0])]
        env.step(stage1_actions)
        
        # Show information available to each player
        for player_id in range(2):
            info = env.get_information_state(player_id)
            print(f"Player {player_id + 1} sees:")
            if len(info) <= 2:  # Only basic info (stage, player_id)
                print("  No additional information")
            else:
                for key, value in info.items():
                    if key not in ["stage", "player_id"]:
                        print(f"  {key}: {value}")
            print()

def demonstrate_stage_weights():
    """Demonstrate the effect of different stage weights"""
    print("=" * 60)
    print("STAGE WEIGHTS EFFECT DEMONSTRATION")
    print("=" * 60)
    
    weight_scenarios = [
        (0.8, 0.2, "Stage 1 Dominant"),
        (0.5, 0.5, "Equal Weights"),
        (0.2, 0.8, "Stage 2 Dominant")
    ]
    
    # Fixed efforts for comparison
    stage1_efforts = [50.0, 60.0]
    stage2_efforts = [70.0, 55.0]  # Player 1 does better in Stage 2
    
    for w1, w2, description in weight_scenarios:
        print(f"\n{description.upper()} (Stage 1: {w1}, Stage 2: {w2})")
        print("-" * 40)
        
        # Create environment with specific weights
        test_config = config.copy()
        test_config["stage1_weight"] = w1
        test_config["stage2_weight"] = w2
        env = TwoStageEnv(test_config)
        
        # Execute both stages
        env.reset()
        env.step([torch.tensor([e]) for e in stage1_efforts])
        final_states, final_rewards, final_costs, final_done, final_info = env.step([torch.tensor([e]) for e in stage2_efforts])
        
        print(f"Stage 1 efforts: {stage1_efforts}")
        print(f"Stage 2 efforts: {stage2_efforts}")
        print(f"Stage 1 winner: Player {final_info['stage1_winner'] + 1}")
        print(f"Stage 2 winner: Player {final_info['stage2_winner'] + 1}")
        final_reward_vals = [r.item() for r in final_rewards]
        print(f"Final weighted rewards: {final_reward_vals}")
        
        # Determine overall winner
        winner = 1 if final_rewards[0] > final_rewards[1] else 2
        print(f"Overall winner: Player {winner}")

def main():
    """Run all demonstrations"""
    print("TWO-STAGE TOURNAMENT ENVIRONMENT DEMONSTRATION")
    print("=" * 60)
    print("This demonstration shows the key features of the two-stage environment:")
    print("1. Sequential decision making across two stages")
    print("2. Information revelation between stages")
    print("3. Weighted combination of stage outcomes")
    print("4. Inter-stage dependency modeling")
    print()
    
    demonstrate_basic_flow()
    print("\n" + "=" * 60 + "\n")
    
    demonstrate_information_scenarios()
    print("\n" + "=" * 60 + "\n")
    
    demonstrate_stage_weights()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("The two-stage environment successfully implements:")
    print("✓ Sequential decision making")
    print("✓ Inter-stage information flow")
    print("✓ Configurable information revelation")
    print("✓ Weighted stage outcomes")
    print("✓ Stage-specific cost parameters")
    print("✓ Comprehensive state management")

if __name__ == "__main__":
    main() 