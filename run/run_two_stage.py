#!/usr/bin/env python3
"""
Two-Stage Tournament Experiment (gradient + true PPO)

This script wires together:
- Gradient/static solver honoring stage order
- A proper PPO agent (clipped objective, GAE, value/entropy terms)

It saves standardized CSV rows and a learning curve figure.
"""

import sys
import os
import argparse
from typing import Dict, List
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.two_stage_two_players import config as base_config
from utils.theory import e_star, clip_stage1, clip_stage2
from utils.eval import build_csv_row
from utils.plot import plot_effort_curve
from utils.logger import save_standardized_result
from agents.two_stage_gradient_agent import TwoStageGradientAgent
from agents.two_stage_ppo_agent import TwoStagePPOAgent
from envs.two_stage_env import TwoStageEnv


def run_gradient(cfg: Dict) -> Dict:
    w_h, w_l = cfg["w_h"], cfg["w_l"]
    k1, k2 = cfg["k1"], cfg["k2"]
    q = cfg["q"]

    solver = TwoStageGradientAgent(w_h, w_l, k1, k2, q)
    e1, e2 = solver.solve()

    row = build_csv_row(
        stage1_weight=cfg["stage1_weight"],
        stage2_weight=cfg["stage2_weight"],
        k1=k1,
        k2=k2,
        information_revelation=cfg.get("information_revelation", "partial"),
        theoretical_stage1_effort=clip_stage1(e_star(q, w_h, w_l, k1), tuple(cfg["effort_bounds_stage1"])),
        theoretical_stage2_effort=clip_stage2(e_star(q, w_h, w_l, k2), tuple(cfg["effort_bounds_stage2"])),
        model_training="gradient",
        final_stage1_effort=e1,
        final_stage2_effort=e2,
        episodes=0,
    )
    return row


def run_ppo(cfg: Dict, episodes: int = 5000, *, prob: str = "uniform", episodes_per_update: int = 1) -> Dict:
    """True PPO with Beta policy, GAE(λ), and logistic win probability self-play.

    Training loop:
    - Create env with prob_model='logit'
    - Both players share the same PPO agent
    - For each episode: Stage 1 act/store; Stage 2 act/store; then update PPO
    """
    w_h, w_l = cfg["w_h"], cfg["w_l"]
    k1, k2 = cfg["k1"], cfg["k2"]
    q = cfg["q"]

    # Set env probability model based on arg (align with theory when 'uniform')
    env_cfg = dict(cfg)
    env_cfg["prob_model"] = str(prob)
    env = TwoStageEnv(env_cfg)

    # Compute theoretical efforts for reporting
    e2_star_val = clip_stage2(e_star(q, w_h, w_l, k2), tuple(cfg["effort_bounds_stage2"]))
    e1_star_val = clip_stage1(e_star(q, w_h, w_l, k1), tuple(cfg["effort_bounds_stage1"]))

    # Use more conservative defaults for stability
    agent = TwoStagePPOAgent(
        effort_bounds_stage1=tuple(cfg["effort_bounds_stage1"]),
        effort_bounds_stage2=tuple(cfg["effort_bounds_stage2"]),
        q_value=q,
        w_h=w_h,
        w_l=w_l,
        k1=k1,
        k2=k2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.002,
        value_coef=0.5,
        lr=1e-4,
        hidden=64,
    )

    history: List[float] = []
    if episodes_per_update <= 0:
        episodes_per_update = 1
    for upd in range(episodes):
        # collect multiple episodes per PPO update
        avg_e1_batch = []
        for _ in range(episodes_per_update):
            _ = env.reset()
            # Stage 1
            e1_p1, logp1, v1, a1_p1, s1_p1 = agent.act(stage=1, opp_signal=agent.opp_avg(1), bounds=tuple(cfg["effort_bounds_stage1"]))
            e1_p2, logp2, v2, a1_p2, s1_p2 = agent.act(stage=1, opp_signal=agent.opp_avg(1), bounds=tuple(cfg["effort_bounds_stage1"]))
            next_states, rewards_s1, costs_s1, done, info = env.step_stage1([
                torch.tensor(e1_p1, dtype=torch.float32),
                torch.tensor(e1_p2, dtype=torch.float32)
            ])

            r1_p1 = float(rewards_s1[0].item())
            r1_p2 = float(rewards_s1[1].item())
            agent.store(s1_p1, a1_p1, e1_p1, logp1, v1, r1_p1, False)
            agent.store(s1_p2, a1_p2, e1_p2, logp2, v2, r1_p2, False)
            agent.update_opponent_avg(stage=1, opponent_effort=e1_p2)
            agent.update_opponent_avg(stage=1, opponent_effort=e1_p1)

            # Stage 2
            s2_obs_p1 = next_states[0]
            s2_obs_p2 = next_states[1]
            e2_p1, logp3, v3, a2_p1, s2_p1 = agent.act_with_env_obs(s2_obs_p1, tuple(cfg["effort_bounds_stage1"]), tuple(cfg["effort_bounds_stage2"]))
            e2_p2, logp4, v4, a2_p2, s2_p2 = agent.act_with_env_obs(s2_obs_p2, tuple(cfg["effort_bounds_stage1"]), tuple(cfg["effort_bounds_stage2"]))
            final_states, rewards_s2, total_costs, done, info2 = env.step_stage2([
                torch.tensor(e2_p1, dtype=torch.float32),
                torch.tensor(e2_p2, dtype=torch.float32)
            ])
            r2_p1 = float(rewards_s2[0].item())
            r2_p2 = float(rewards_s2[1].item())
            agent.store(s2_p1, a2_p1, e2_p1, logp3, v3, r2_p1, True)
            agent.store(s2_p2, a2_p2, e2_p2, logp4, v4, r2_p2, True)

            avg_e1_batch.append((e1_p1 + e1_p2) / 2.0)

        # One PPO update per batch
        agent.update()
        history.append(float(np.mean(avg_e1_batch)))

    final_e1 = float(np.mean(history[-100:])) if len(history) >= 100 else float(np.mean(history))
    # Estimate final stage2 effort by sampling deterministically using a neutral Stage-2 obs
    neutral_obs = torch.tensor([2.0, 0.0, float(final_e1), 0.0, 0.0], dtype=torch.float32)
    e2_p1_det, _, _, _, _ = agent.act_with_env_obs(neutral_obs, tuple(cfg["effort_bounds_stage1"]), tuple(cfg["effort_bounds_stage2"]), deterministic=True)
    final_e2 = float(e2_p1_det)

    row = build_csv_row(
        stage1_weight=cfg["stage1_weight"],
        stage2_weight=cfg["stage2_weight"],
        k1=k1,
        k2=k2,
        information_revelation=cfg.get("information_revelation", "partial"),
        theoretical_stage1_effort=e1_star_val,
        theoretical_stage2_effort=e2_star_val,
        model_training="ppo",
        final_stage1_effort=final_e1,
        final_stage2_effort=final_e2,
        episodes=episodes,
    )

    # Plot effort curve with overlays
    plot_effort_curve(
        efforts=history,
        qs=cfg["q_list"],
        e_star_fn=e_star,
        w_h=w_h,
        w_l=w_l,
        k=k1,
        title="Two-Stage (True PPO) learned effort vs episodes",
        output_png=os.path.join("results", "two_stage_two_players.png"),
        effort_bounds=tuple(cfg["effort_bounds_stage1"]),
    )

    return row


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Tournament Experiment")
    parser.add_argument("--method", choices=["gradient", "ppo"], default="gradient")
    parser.add_argument("--q", type=float, help="Override q (otherwise run all in config q_list)")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of PPO updates")
    parser.add_argument("--prob", choices=["uniform", "logit"], default="uniform", help="Win-probability model in env")
    parser.add_argument("--episodes-per-update", type=int, default=1, help="Episodes collected per PPO update")
    args = parser.parse_args()

    cfg = dict(base_config)
    csv_path = os.path.join("results", "two_stage_two_players.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    q_values = [args.q] if args.q is not None else list(cfg["q_list"])

    for q in q_values:
        cfg["q"] = float(q)

        if args.method == "gradient":
            row = run_gradient(cfg)
        else:
            row = run_ppo(cfg, episodes=args.episodes, prob=args.prob, episodes_per_update=args.episodes_per_update)

        save_standardized_result(row, csv_path)

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
