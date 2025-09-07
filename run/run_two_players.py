#!/usr/bin/env python3
"""
One-Stage Two-Player Experiment (spec-compliant)

Writes standardized CSV and figure overlays. For one-stage, we map the single
stage to the CSV's stage-2 fields (stage-1 fields set to 0).
"""

import sys
import os
import argparse
from typing import Dict, List, Optional
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.one_stage_two_players import config as base_config
from utils.theory import e_star_two_players, clip_stage2
from utils.eval import build_csv_row
from utils.plot import plot_effort_curve
from utils.logger import save_standardized_result
from envs.two_players_env import TwoPlayersEnv
from agents.ppo_two_players_clean import PPOTwoPlayersBandit, PPOConfig


def run_gradient(cfg: Dict) -> Dict:
    w_h, w_l, k, q = cfg["w_h"], cfg["w_l"], cfg["k"], cfg["q"]
    e2 = clip_stage2(e_star_two_players(q, w_h, w_l, k), tuple(cfg["effort_bounds_stage2"]))
    row = build_csv_row(
        stage1_weight=cfg["stage1_weight"],
        stage2_weight=cfg["stage2_weight"],
        k1=cfg["k1"],
        k2=cfg["k2"],
        information_revelation=cfg.get("information_revelation", "none"),
        theoretical_stage1_effort=0.0,
        theoretical_stage2_effort=e2,
        model_training="gradient",
        final_stage1_effort=0.0,
        final_stage2_effort=e2,
        episodes=0,
    )
    return row


def run_ppo(cfg: Dict, episodes: int = 5000, train_qs: Optional[List[float]] = None, eval_qs: Optional[List[float]] = None) -> List[Dict]:
    """Train PPO via self-play with conditioning on (q, k, w_gap).

    - Trains over ``train_qs`` (defaults to cfg["q_list" ]).
    - Returns a list of CSV rows, one per q in ``eval_qs`` (defaults to train_qs).
    """
    w_h, w_l, k = cfg["w_h"], cfg["w_l"], cfg["k"]
    effort_bounds = tuple(cfg["effort_bounds_stage2"])  # (0, 200)
    train_qs = list(train_qs if train_qs is not None else cfg["q_list"])
    eval_qs = list(eval_qs if eval_qs is not None else train_qs)

    # PPO agent with 3-dim state: [q, k, w_gap]
    ppo_cfg = PPOConfig(steps_per_update=2048, epochs=15, minibatch_size=128, state_dim=3, hidden=64)
    agent = PPOTwoPlayersBandit(effort_bounds=effort_bounds, cfg=ppo_cfg)

    history: List[float] = []
    total_steps_target = int(episodes)
    steps_done = 0
    rng = np.random.default_rng(cfg.get("seed", 42))

    while steps_done < total_steps_target:
        steps_this = min(ppo_cfg.steps_per_update, total_steps_target - steps_done)
        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env = TwoPlayersEnv(w_h=w_h, w_l=w_l, k=k, q=q, effort_bounds=effort_bounds, seed=cfg.get("seed", 42))

            s1 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            a1_norm, e1, logp1, v1 = agent.act(s1)
            s2 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            a2_norm, e2, logp2, v2 = agent.act(s2)

            _, rewards, _, done, _ = env.step((torch.tensor([float(e1.item())]), torch.tensor([float(e2.item())])))

            agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
            history.append(float((e1.item() + e2.item()) / 2.0))
        agent.update()
        steps_done += steps_this

    # Build rows for each evaluation q
    rows: List[Dict] = []
    for q in eval_qs:
        e2_star_val = clip_stage2(e_star_two_players(q, w_h, w_l, k), effort_bounds)

        # Evaluate by using mean action of current policy
        s_eval = agent.state_from_params(q=float(q), k=k, w_h=w_h, w_l=w_l)
        with torch.no_grad():
            dist, _ = agent.net.dist(s_eval)
            a_mean = (dist.concentration1 / (dist.concentration1 + dist.concentration0)).squeeze().cpu().item()
        final_e2 = float(effort_bounds[0] + a_mean * (effort_bounds[1] - effort_bounds[0]))

        row = build_csv_row(
            stage1_weight=cfg["stage1_weight"],
            stage2_weight=cfg["stage2_weight"],
            k1=cfg["k1"],
            k2=cfg["k2"],
            information_revelation=cfg.get("information_revelation", "none"),
            theoretical_stage1_effort=0.0,
            theoretical_stage2_effort=e2_star_val,
            model_training="ppo",
            final_stage1_effort=0.0,
            final_stage2_effort=final_e2,
            episodes=episodes,
        )
        rows.append(row)

    # Plot overlays for each q (training history)
    plot_effort_curve(
        efforts=history,
        qs=eval_qs,
        e_star_fn=e_star_two_players,
        w_h=w_h,
        w_l=w_l,
        k=k,
        title="One-Stage Two-Player learned effort vs episodes",
        output_png=os.path.join("results", "one_stage_two_players.png"),
        effort_bounds=effort_bounds,
    )

    return rows


def main():
    parser = argparse.ArgumentParser(description="One-Stage Two-Player Experiment (spec)")
    parser.add_argument("--method", choices=["gradient", "ppo"], default="gradient")
    parser.add_argument("--q", type=float, help="Override q (otherwise run all in config q_list)")
    parser.add_argument("--episodes", type=int, default=5000, help="Episodes for PPO")
    args = parser.parse_args()

    cfg = dict(base_config)
    csv_path = os.path.join("results", "one_stage_two_players.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if args.method == "gradient":
        q_values = [args.q] if args.q is not None else list(cfg["q_list"])
        for q in q_values:
            cfg["q"] = float(q)
            row = run_gradient(cfg)
            save_standardized_result(row, csv_path)
    else:
        # Train once; evaluate for all q (or the specified q)
        train_qs = [args.q] if args.q is not None else list(cfg["q_list"])
        eval_qs = train_qs
        rows = run_ppo(cfg, episodes=args.episodes, train_qs=train_qs, eval_qs=eval_qs)
        for row in rows:
            save_standardized_result(row, csv_path)

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()

 
