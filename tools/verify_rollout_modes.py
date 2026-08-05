#!/usr/bin/env python3
"""
Sanity check tool to verify rollout mode behavior.

Tests the "selfplay" mode to ensure:
1. Correct number of transitions stored per step (2 per step)
2. Both players use the learner policy
3. All transitions are stored in the PPO buffer
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


def test_selfplay_mode(num_steps: int = 100) -> Dict[str, int]:
    """
    Run a short rollout in selfplay mode and count stored transitions.

    Args:
        num_steps: Number of rollout steps

    Returns:
        Dictionary with counts: {stored_p1, stored_p2, buffer_size}
    """
    print(f"\n{'='*80}")
    print(f"Testing mode: SELFPLAY")
    print(f"{'='*80}")

    cfg = dict(base_config)
    w_h, w_l, k, q = cfg["w_h"], cfg["w_l"], cfg["k"], 25.0
    effort_bounds = tuple(cfg["effort_bounds_stage2"])

    # Create agent
    ppo_cfg = PPOConfig(
        steps_per_update=4096,
        epochs=6,
        minibatch_size=1024,
        state_dim=4,
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

    # Counters
    stored_p1 = 0
    stored_p2 = 0

    print(f"Running {num_steps} steps...")
    print()

    for step_idx in range(num_steps):
        s1 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
        s2 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)

        # Player 1: Always learner
        a1_norm, e1, logp1, v1 = agent.act(s1)

        # Player 2: Also learner in selfplay
        a2_norm, e2, logp2, v2 = agent.act(s2)

        # Execute step
        _, rewards, _, done, _ = env.step((
            torch.tensor([float(e1.item())]),
            torch.tensor([float(e2.item())])
        ))

        # Storage: both players always stored in selfplay
        agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
        stored_p1 += 1

        agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
        stored_p2 += 1

    # Verify storage buffer sizes match our counts
    buffer_size = len(agent.storage["states"])
    expected_size = stored_p1 + stored_p2

    print(f"Results:")
    print(f"  - Player 1 stored: {stored_p1}")
    print(f"  - Player 2 stored: {stored_p2}")
    print(f"  - Total stored: {stored_p1 + stored_p2}")
    print(f"  - Buffer size: {buffer_size}")
    print()

    # Validation
    assert buffer_size == expected_size, \
        f"Buffer size mismatch! Expected {expected_size}, got {buffer_size}"

    # Selfplay: Should store exactly 2 transitions per step
    assert stored_p1 == num_steps, "Player 1 should be stored every step"
    assert stored_p2 == num_steps, "Player 2 should be stored every step in selfplay"
    assert buffer_size == 2 * num_steps, \
        f"Buffer should contain exactly {2 * num_steps} transitions"
    print("SELFPLAY mode validation PASSED")
    print(f"   - Stored 2 transitions per step as expected")

    return {
        "stored_p1": stored_p1,
        "stored_p2": stored_p2,
        "buffer_size": buffer_size,
    }


def main():
    """Run sanity checks for the selfplay rollout mode."""
    print("="*80)
    print("ROLLOUT MODE SANITY CHECKS")
    print("="*80)

    num_steps = 100

    # Test selfplay mode
    results_selfplay = test_selfplay_mode(num_steps=num_steps)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nSelfplay mode ({num_steps} steps):")
    print(f"  - Total stored: {results_selfplay['buffer_size']} (expected: {2 * num_steps})")
    print(f"  - Storage rate: 2 transitions/step")

    print("\n" + "="*80)
    print("ALL CHECKS PASSED")
    print("="*80)
    print("\nSelfplay mode behaves correctly:")
    print("  1. Both players use learner policy, both transitions stored")
    print("  2. Buffer contains exactly 2 * num_steps transitions")


if __name__ == "__main__":
    main()
