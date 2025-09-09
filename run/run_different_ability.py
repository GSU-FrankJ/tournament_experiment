#!/usr/bin/env python3
"""
Different-Ability Two-Player Experiment (True PPO + Gradient)
=============================================================

This script runs the two-player different-ability scenario (l1 > l2, k1=k2)
with two options:
- gradient: uses agents.different_ability_solver.different_ability_gradient_descent_solver
- ppo:      uses a true PPO implementation (Beta policy + GAE + clipping)
            from agents.ppo_two_players_clean.PPOTwoPlayersBandit, extended
            with ability features in the state for self-play.

Results are saved to results/tables/different_ability_two_players.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

# Ensure project root is on sys.path when executing as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.different_ability_two_players import (
    DIFFERENT_ABILITY_CONFIG,
    calculate_theoretical_efforts_different_ability,
)
from envs.different_ability_env import DifferentAbilityEnv
from agents.different_ability_solver import different_ability_gradient_descent_solver
from agents.ppo_two_players_clean import PPOTwoPlayersBandit, PPOConfig


def norm01(x: float, denom: float, clip: float = 1e-9) -> float:
    d = max(clip, float(denom))
    return float(x) / d


def run_gradient(cfg: Dict) -> Dict:
    env = DifferentAbilityEnv(cfg)
    e1_th, e2_th, _, _, EU1_th, EU2_th = calculate_theoretical_efforts_different_ability(
        cfg["q"], cfg["k1"], cfg["k2"], cfg["l1"], cfg["l2"], cfg["w_h"], cfg["w_l"],
    )
    efforts, utils, costs, info = different_ability_gradient_descent_solver(
        env, lr=0.1, steps=100000, eps=1e-3, adaptive_lr=True, convergence_threshold=1e-4, patience=1000, verbose=False
    )
    gap = max(abs(efforts[0] - e1_th), abs(efforts[1] - e2_th))
    quality = (
        "Excellent" if gap < 0.5 else
        "Good" if gap < 1.0 else
        "Fair" if gap < 5.0 else
        "Poor"
    )
    return {
        "method": "gradient",
        "q": cfg["q"],
        "l1": cfg["l1"],
        "l2": cfg["l2"],
        "k1": cfg["k1"],
        "k2": cfg["k2"],
        "w_h": cfg["w_h"],
        "w_l": cfg["w_l"],
        "theoretical_e1": e1_th,
        "theoretical_e2": e2_th,
        "final_e1": efforts[0],
        "final_e2": efforts[1],
        "episodes": info.get("final_step", 0),
        "max_gap": gap,
        "quality": quality,
    }


def run_ppo(cfg: Dict, episodes: int, updates: Optional[int] = None) -> Dict:
    # Build environment and theory
    env = DifferentAbilityEnv(cfg)
    e1_th, e2_th, _, _, _, _ = calculate_theoretical_efforts_different_ability(
        cfg["q"], cfg["k1"], cfg["k2"], cfg["l1"], cfg["l2"], cfg["w_h"], cfg["w_l"],
    )

    # Configure PPO with state including ability features
    ppo_cfg = PPOConfig(
        steps_per_update=16384,
        epochs=20,
        minibatch_size=1024,
        state_dim=5,  # [q_norm, k_norm, wgap_norm, l_self_norm, l_other_norm]
        hidden=64,
        opponent_sync_interval=1,
        opponent_ema_tau=0.0,
        entropy_coef=0.02,
        lr=3e-4,
        clip_eps=0.25,
    )
    agent = PPOTwoPlayersBandit(effort_bounds=tuple(cfg["effort_range"]), cfg=ppo_cfg)

    def state_for(l_self: float, l_other: float) -> torch.Tensor:
        q_norm = norm01(cfg["q"], 60.0)
        k_norm = norm01(cfg["k1"], 1e-3)
        wgap_norm = norm01(cfg["w_h"] - cfg["w_l"], 10.0)
        denom = max(cfg["l1"], cfg["l2"], 1.0)
        l_self_n = norm01(l_self, denom)
        l_other_n = norm01(l_other, denom)
        s = torch.tensor([q_norm, k_norm, wgap_norm, l_self_n, l_other_n], dtype=torch.float32)
        return s.unsqueeze(0)

    # Schedules similar to run_two_players.py
    total_steps_target = int(episodes)
    steps_done = 0
    update_idx = 0
    low, high = tuple(cfg["effort_range"])

    # If explicit updates provided, override total steps using PPO steps_per_update
    if updates is not None:
        total_steps_target = int(updates) * ppo_cfg.steps_per_update

    # Late-phase setup
    total_updates = (total_steps_target + ppo_cfg.steps_per_update - 1) // ppo_cfg.steps_per_update
    late_updates = min(100, total_updates)
    start_late = max(0, total_updates - late_updates)
    # Entropy scheduling
    start_entropy, end_entropy, decay_updates = 0.02, 0.002, 50
    entropy_zero_updates = min(30, total_updates)
    start_entropy_zero = max(0, total_updates - entropy_zero_updates)
    # LR boost late
    lr_base = ppo_cfg.lr
    lr_boost_value = 4e-4
    lr_boost_updates = min(50, total_updates)
    start_lr_late = max(0, total_updates - lr_boost_updates)

    rng = np.random.default_rng(cfg.get("seed", 42))

    while steps_done < total_steps_target:
        progress = min(1.0, float(update_idx) / float(decay_updates))
        agent.cfg.entropy_coef = start_entropy + (end_entropy - start_entropy) * progress
        if update_idx >= start_entropy_zero:
            agent.cfg.entropy_coef = 0.0
        if update_idx >= start_late:
            prog_late = 1.0 if late_updates <= 1 else float(update_idx - start_late) / float(late_updates - 1)
            agent.cfg.clip_eps = 0.35 - 0.10 * max(0.0, min(1.0, prog_late))
        else:
            agent.cfg.clip_eps = 0.25
        new_lr = lr_boost_value if update_idx >= start_lr_late else lr_base
        for g in agent.opt.param_groups:
            g["lr"] = new_lr

        steps_this = min(ppo_cfg.steps_per_update, total_steps_target - steps_done)
        for _ in range(steps_this):
            # Single env instance per step; different ability is static here
            s1 = state_for(cfg["l1"], cfg["l2"])
            s2 = state_for(cfg["l2"], cfg["l1"])

            a1_norm, e1, logp1, v1 = agent.act(s1)
            a2_norm, e2, logp2, v2 = agent.act_opponent(s2)

            # Step environment
            _, rewards, _, done, _ = env.step([
                torch.tensor([float(e1.item())]),
                torch.tensor([float(e2.item())]),
            ])

            # Store samples (late-phase store both players)
            if update_idx >= start_late:
                agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
                agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
            else:
                agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))

        agent.update()
        # Quick eval after update
        with torch.no_grad():
            se1 = state_for(cfg["l1"], cfg["l2"])
            se2 = state_for(cfg["l2"], cfg["l1"])
            dist1, _ = agent.net.dist(se1)
            dist2, _ = agent.net.dist(se2)
            a1 = dist1.mean.squeeze().clamp(0.0, 1.0)
            a2 = dist2.mean.squeeze().clamp(0.0, 1.0)
            e1_eval = float(low + a1.item() * (high - low))
            e2_eval = float(low + a2.item() * (high - low))
            gap = max(abs(e1_eval - e1_th), abs(e2_eval - e2_th))
            print(f"[Update {update_idx+1}] q={cfg['q']}: e1*={e1_th:.2f}, e2*={e2_th:.2f}, policy=({e1_eval:.2f}, {e2_eval:.2f}), gap={gap:.2f}, entropy={agent.cfg.entropy_coef:.3f}")

        update_idx += 1
        steps_done += steps_this

    # Final evaluation
    with torch.no_grad():
        se1 = state_for(cfg["l1"], cfg["l2"])
        se2 = state_for(cfg["l2"], cfg["l1"])
        d1, _ = agent.net.dist(se1)
        d2, _ = agent.net.dist(se2)
        a1 = d1.mean.squeeze().clamp(0.0, 1.0)
        a2 = d2.mean.squeeze().clamp(0.0, 1.0)
        e1_final = float(low + a1.item() * (high - low))
        e2_final = float(low + a2.item() * (high - low))

    gap = max(abs(e1_final - e1_th), abs(e2_final - e2_th))
    quality = (
        "Excellent" if gap < 0.5 else
        "Good" if gap < 1.0 else
        "Fair" if gap < 5.0 else
        "Poor"
    )
    return {
        "method": "ppo",
        "q": cfg["q"],
        "l1": cfg["l1"],
        "l2": cfg["l2"],
        "k1": cfg["k1"],
        "k2": cfg["k2"],
        "w_h": cfg["w_h"],
        "w_l": cfg["w_l"],
        "theoretical_e1": e1_th,
        "theoretical_e2": e2_th,
        "final_e1": e1_final,
        "final_e2": e2_final,
        "episodes": int(episodes),
        "max_gap": gap,
        "quality": quality,
    }


def save_results(rows: List[Dict]):
    import pandas as pd
    os.makedirs("results/tables", exist_ok=True)
    path = "results/tables/different_ability_two_players.csv"
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        # append while keeping columns superset
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True, sort=False)
    df.to_csv(path, index=False)
    print(f"Saved results to {path}")


def main():
    parser = argparse.ArgumentParser(description="Different-Ability Two-Player Experiment")
    parser.add_argument("--method", choices=["gradient", "ppo"], default="gradient")
    parser.add_argument("--q", type=float, help="Override q (else run default from config)")
    parser.add_argument("--episodes", type=int, default=100000, help="Total steps for PPO (ignored if --updates is set)")
    parser.add_argument("--updates", type=int, help="Number of PPO updates (episodes = updates * steps_per_update)")
    args = parser.parse_args()

    base = dict(DIFFERENT_ABILITY_CONFIG)
    # Use a default effort range that matches other runs
    base["effort_range"] = [0, 200]

    if args.q is not None:
        base["q"] = float(args.q)

    rows: List[Dict] = []
    if args.method == "gradient":
        rows.append(run_gradient(base))
    else:
        rows.append(run_ppo(base, episodes=args.episodes, updates=args.updates))
    save_results(rows)


if __name__ == "__main__":
    main()
