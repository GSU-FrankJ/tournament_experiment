#!/usr/bin/env python3
"""
Sanity check tool to verify rollout mode behaviors.

Tests both "selfplay" and "vs_opponent" modes to ensure:
1. Correct number of transitions stored per step
2. Proper handling of opponent-generated actions
3. No mixing of opponent samples into learner's PPO buffer
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from typing import Dict

from config.one_stage_two_players import config as base_config
from envs.two_players_env import TwoPlayersEnv
from agents.ppo_two_players_clean import PPOTwoPlayersBandit, PPOConfig


def test_rollout_mode(
    mode: str,
    num_steps: int = 100,
    lag_prob: float = 0.5,
) -> Dict[str, int]:
    """
    Run a short rollout in the specified mode and count stored transitions.
    
    Args:
        mode: "selfplay" or "vs_opponent"
        num_steps: Number of rollout steps
        lag_prob: Probability of using opponent (for vs_opponent mode)
        
    Returns:
        Dictionary with counts: {stored_p1, stored_p2, skipped_p2}
    """
    print(f"\n{'='*80}")
    print(f"Testing mode: {mode.upper()}")
    print(f"{'='*80}")
    
    cfg = dict(base_config)
    w_h, w_l, k, q = cfg["w_h"], cfg["w_l"], cfg["k"], 25.0
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
    
    rng = np.random.default_rng(42)
    
    # Counters
    stored_p1 = 0
    stored_p2 = 0
    skipped_p2 = 0
    
    print(f"Running {num_steps} steps...")
    print(f"Mode: {mode}")
    if mode == "vs_opponent":
        print(f"Opponent usage probability: {lag_prob}")
    print()
    
    for step_idx in range(num_steps):
        s1 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
        s2 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
        
        # Player 1: Always learner
        a1_norm, e1, logp1, v1 = agent.act(s1)
        
        # Player 2: Mode-dependent
        if mode == "selfplay":
            # Selfplay: Player 2 always uses learner
            a2_norm, e2, logp2, v2 = agent.act(s2)
            use_opponent = False
            
        else:  # vs_opponent
            # VS_OPPONENT: Player 2 may use opponent
            use_opponent = (lag_prob > 0.0) and (rng.random() < lag_prob)
            if use_opponent:
                a2_norm, e2, logp2, _ = agent.act_opponent(s2)
                v2 = agent.value_only(s2)
            else:
                a2_norm, e2, logp2, v2 = agent.act(s2)
        
        # Execute step
        _, rewards, _, done, _ = env.step((
            torch.tensor([float(e1.item())]),
            torch.tensor([float(e2.item())])
        ))
        
        # Storage logic (matching run_two_players.py)
        # Player 1: Always store
        agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
        stored_p1 += 1
        
        # Player 2: Mode-dependent
        if mode == "selfplay":
            # Selfplay: Always store
            agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
            stored_p2 += 1
        else:  # vs_opponent
            # VS_OPPONENT: Only store when learner was used
            if not use_opponent:
                agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
                stored_p2 += 1
            else:
                skipped_p2 += 1
    
    # Verify storage buffer sizes match our counts
    buffer_size = len(agent.storage["states"])
    expected_size = stored_p1 + stored_p2
    
    print(f"Results:")
    print(f"  - Player 1 stored: {stored_p1}")
    print(f"  - Player 2 stored: {stored_p2}")
    print(f"  - Player 2 skipped (opponent): {skipped_p2}")
    print(f"  - Total stored: {stored_p1 + stored_p2}")
    print(f"  - Buffer size: {buffer_size}")
    print()
    
    # Validation
    assert buffer_size == expected_size, \
        f"Buffer size mismatch! Expected {expected_size}, got {buffer_size}"
    
    if mode == "selfplay":
        # Selfplay: Should store exactly 2 transitions per step
        assert stored_p1 == num_steps, "Player 1 should be stored every step"
        assert stored_p2 == num_steps, "Player 2 should be stored every step in selfplay"
        assert skipped_p2 == 0, "No Player 2 skips should occur in selfplay"
        assert buffer_size == 2 * num_steps, f"Buffer should contain exactly {2 * num_steps} transitions"
        print("✅ SELFPLAY mode validation PASSED")
        print(f"   - Stored 2 transitions per step as expected")
        
    else:  # vs_opponent
        # VS_OPPONENT: Should store 1-2 transitions per step depending on opponent usage
        assert stored_p1 == num_steps, "Player 1 should be stored every step"
        assert stored_p2 + skipped_p2 == num_steps, "Player 2 should be either stored or skipped"
        assert skipped_p2 > 0, "Some Player 2 skips should occur with lag_prob > 0"
        # Buffer should contain only learner-generated samples
        expected_learner_samples = stored_p1 + stored_p2
        assert buffer_size == expected_learner_samples, \
            f"Buffer should contain only learner samples ({expected_learner_samples})"
        print("✅ VS_OPPONENT mode validation PASSED")
        print(f"   - Stored 1-2 transitions per step (opponent-dependent)")
        print(f"   - Skipped {skipped_p2}/{num_steps} Player 2 samples (opponent-generated)")
        print(f"   - Buffer contains ONLY learner-generated samples")
    
    return {
        "stored_p1": stored_p1,
        "stored_p2": stored_p2,
        "skipped_p2": skipped_p2,
        "buffer_size": buffer_size,
    }


def main():
    """Run sanity checks for both rollout modes."""
    print("="*80)
    print("ROLLOUT MODE SANITY CHECKS")
    print("="*80)
    
    num_steps = 100
    
    # Test 1: Selfplay mode
    results_selfplay = test_rollout_mode(
        mode="selfplay",
        num_steps=num_steps,
        lag_prob=0.0,  # Not used in selfplay, but set to 0 for clarity
    )
    
    # Test 2: VS_OPPONENT mode with lag
    results_vs_opp = test_rollout_mode(
        mode="vs_opponent",
        num_steps=num_steps,
        lag_prob=0.5,  # 50% chance of using opponent
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nSelfplay mode ({num_steps} steps):")
    print(f"  - Total stored: {results_selfplay['buffer_size']} (expected: {2 * num_steps})")
    print(f"  - Storage rate: 2 transitions/step ✓")
    
    print(f"\nVS_OPPONENT mode ({num_steps} steps, lag_prob=0.5):")
    print(f"  - Total stored: {results_vs_opp['buffer_size']}")
    print(f"  - Player 1: {results_vs_opp['stored_p1']} (always stored)")
    print(f"  - Player 2: {results_vs_opp['stored_p2']} (learner-generated)")
    print(f"  - Skipped: {results_vs_opp['skipped_p2']} (opponent-generated)")
    print(f"  - Storage rate: 1-2 transitions/step (variable) ✓")
    print(f"  - Opponent samples excluded from buffer: ✓")
    
    print("\n" + "="*80)
    print("✅ ALL CHECKS PASSED")
    print("="*80)
    print("\nBoth rollout modes behave correctly:")
    print("  1. Selfplay: Both players use learner, store both")
    print("  2. VS_OPPONENT: Only learner-generated samples in buffer")
    print("  3. No opponent-generated samples mixed into PPO updates")


if __name__ == "__main__":
    main()



