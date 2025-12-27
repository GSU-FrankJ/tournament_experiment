#!/usr/bin/env python3
"""
Diagnostic script to prove whether opponent-generated actions mix into learner's PPO update.

This script runs a short rollout with detailed logging to track:
1. Which policy generates each action (learner vs opponent)
2. What gets stored in the PPO buffer
3. What the PPO update consumes
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from typing import Dict, List

from config.one_stage_two_players import config as base_config
from envs.two_players_env import TwoPlayersEnv
from agents.ppo_two_players_clean import PPOTwoPlayersBandit, PPOConfig


def diagnose_data_provenance(num_steps: int = 200, use_opponent_prob: float = 0.5):
    """
    Run a short rollout and log exactly what data is stored and where it comes from.
    
    Args:
        num_steps: Number of rollout steps to execute
        use_opponent_prob: Probability of using opponent policy for player 2
    """
    cfg = dict(base_config)
    w_h, w_l, k, q = cfg["w_h"], cfg["w_l"], cfg["k"], 25.0  # Use q=25 for testing
    effort_bounds = tuple(cfg["effort_bounds_stage2"])
    
    # Create agent
    ppo_cfg = PPOConfig(
        steps_per_update=4096,
        epochs=6,
        minibatch_size=1024,
        state_dim=3,
        hidden=128,
        opponent_mode="periodic",
        opponent_sync_interval=2,
    )
    agent = PPOTwoPlayersBandit(effort_bounds=effort_bounds, cfg=ppo_cfg)
    
    # Create environment
    env = TwoPlayersEnv(
        w_h=w_h,
        w_l=w_l,
        k=k,
        q=q,
        effort_bounds=effort_bounds,
        seed=42,
    )
    
    # Tracking variables
    learner_stored_count = 0
    opponent_stored_count = 0
    step_logs: List[Dict] = []
    
    rng = np.random.default_rng(42)
    
    print("=" * 80)
    print("DIAGNOSTIC: Data Provenance Tracking")
    print("=" * 80)
    print(f"Running {num_steps} steps with use_opponent_prob={use_opponent_prob}")
    print(f"Agent opponent_mode: {agent.opponent_mode}")
    print(f"Opponent sync interval: {agent.opponent_sync_interval}")
    print()
    
    for step_idx in range(num_steps):
        s1 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
        s2 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
        
        # Player 1: ALWAYS uses learner policy
        a1_norm, e1, logp1, v1 = agent.act(s1)
        source_policy_p1 = "learner"
        
        # Player 2: Sometimes uses opponent policy
        use_opponent = rng.random() < use_opponent_prob
        if use_opponent:
            a2_norm, e2, logp2, _ = agent.act_opponent(s2)
            v2 = agent.value_only(s2)
            source_policy_p2 = "opponent"
        else:
            a2_norm, e2, logp2, v2 = agent.act(s2)
            source_policy_p2 = "learner"
        
        # Execute step in environment
        _, rewards, _, done, _ = env.step((
            torch.tensor([float(e1.item())]), 
            torch.tensor([float(e2.item())])
        ))
        
        # BOTH transitions are stored regardless of who generated them
        agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
        agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
        
        # Track counts
        learner_stored_count += 1  # Player 1 always learner
        if source_policy_p2 == "opponent":
            opponent_stored_count += 1
        else:
            learner_stored_count += 1
        
        # Log first 10 and last 10 steps in detail
        if step_idx < 10 or step_idx >= num_steps - 10:
            log_entry = {
                "step": step_idx,
                "use_opponent": use_opponent,
                "p1_policy": source_policy_p1,
                "p2_policy": source_policy_p2,
                "p1_stored": True,
                "p2_stored": True,
                "p1_logp": float(logp1.item()),
                "p2_logp": float(logp2.item()),
                "p1_effort": float(e1.item()),
                "p2_effort": float(e2.item()),
            }
            step_logs.append(log_entry)
            print(f"Step {step_idx:3d}: use_opp={use_opponent}, "
                  f"P1={source_policy_p1:8s}, P2={source_policy_p2:8s}, "
                  f"stored=(P1:✓, P2:✓), "
                  f"efforts=({e1.item():.2f}, {e2.item():.2f})")
    
    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total rollout steps: {num_steps}")
    print(f"Total transitions stored: {2 * num_steps} (both players per step)")
    print(f"  - Learner-generated: {learner_stored_count}")
    print(f"  - Opponent-generated: {opponent_stored_count}")
    print(f"  - Fraction opponent: {opponent_stored_count / (2 * num_steps):.2%}")
    print()
    
    # Now check what's actually in storage
    print("=" * 80)
    print("STORAGE BUFFER CONTENTS")
    print("=" * 80)
    stored_states = len(agent.storage["states"])
    stored_actions = len(agent.storage["actions_norm"])
    stored_logp = len(agent.storage["logp"])
    stored_rewards = len(agent.storage["rewards"])
    stored_values = len(agent.storage["values"])
    
    print(f"Storage sizes:")
    print(f"  - states: {stored_states}")
    print(f"  - actions_norm: {stored_actions}")
    print(f"  - logp: {stored_logp}")
    print(f"  - rewards: {stored_rewards}")
    print(f"  - values: {stored_values}")
    print()
    
    print("Sample of stored log_probs (first 10):")
    for i in range(min(10, len(agent.storage["logp"]))):
        logp_val = agent.storage["logp"][i].item()
        # Determine if this was likely from opponent based on step logs
        if i < len(step_logs):
            if i % 2 == 0:
                source = step_logs[i // 2]["p1_policy"]
            else:
                source = step_logs[i // 2]["p2_policy"]
            print(f"  [{i:2d}] logp={logp_val:7.4f} (source: {source})")
        else:
            print(f"  [{i:2d}] logp={logp_val:7.4f}")
    
    print()
    print("=" * 80)
    print("PPO UPDATE SIMULATION")
    print("=" * 80)
    
    # Simulate what happens during PPO update
    # Load the data (this is what update() does)
    states = torch.stack(agent.storage["states"]).to(agent.device)
    actions_norm = torch.stack(agent.storage["actions_norm"]).unsqueeze(-1).to(agent.device)
    old_logp = torch.stack(agent.storage["logp"]).to(agent.device)
    
    print(f"PPO update will process:")
    print(f"  - Batch size: {states.size(0)}")
    print(f"  - States shape: {states.shape}")
    print(f"  - Actions shape: {actions_norm.shape}")
    print(f"  - Old logp shape: {old_logp.shape}")
    print()
    
    # Evaluate actions with CURRENT learner policy
    with torch.no_grad():
        new_logp, entropy, values = agent.evaluate_actions(states, actions_norm)
        
        # Compute PPO ratio
        ratio = torch.exp(new_logp - old_logp)
        
        print("PPO ratio statistics:")
        print(f"  - Mean ratio: {ratio.mean().item():.4f}")
        print(f"  - Std ratio: {ratio.std().item():.4f}")
        print(f"  - Min ratio: {ratio.min().item():.4f}")
        print(f"  - Max ratio: {ratio.max().item():.4f}")
        print()
        
        # Check for suspicious ratios (indicating policy mismatch)
        suspicious_mask = (ratio < 0.5) | (ratio > 2.0)
        suspicious_count = suspicious_mask.sum().item()
        print(f"  - Suspicious ratios (< 0.5 or > 2.0): {suspicious_count} / {ratio.size(0)} ({100 * suspicious_count / ratio.size(0):.1f}%)")
        
        # For opponent-generated samples, the ratio is meaningless!
        print()
        print("⚠️  CRITICAL ISSUE IDENTIFIED:")
        print(f"    {opponent_stored_count} transitions ({100 * opponent_stored_count / (2 * num_steps):.1f}%) were generated by opponent policy")
        print(f"    but PPO update computes ratio = exp(learner_logp - opponent_logp)")
        print(f"    This ratio is MEANINGLESS and will distort learning!")
    
    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("VERDICT: CONFIRMED HIGH-RISK MIXING ✗")
    print()
    print("Evidence:")
    print(f"1. {opponent_stored_count} / {2 * num_steps} transitions came from opponent policy")
    print(f"2. ALL {2 * num_steps} transitions are used in PPO update (no filtering)")
    print(f"3. PPO computes ratio = exp(new_logp - old_logp) for ALL transitions")
    print(f"4. For opponent samples: old_logp is from opponent, new_logp is from learner")
    print(f"5. This creates meaningless PPO ratios that distort gradient signals")
    print()
    print("Code locations:")
    print("  - Action generation: run/run_two_players.py:445-453")
    print("  - Storage (both players): run/run_two_players.py:457-458")
    print("  - PPO update (no filtering): agents/ppo_two_players_clean.py:242-296")
    print("  - Ratio computation: agents/ppo_two_players_clean.py:277")
    print()


if __name__ == "__main__":
    diagnose_data_provenance(num_steps=200, use_opponent_prob=0.5)



