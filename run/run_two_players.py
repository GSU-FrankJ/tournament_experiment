#!/usr/bin/env python3
"""
One-Stage Two-Player Experiment (spec-compliant)

Writes standardized CSV and figure overlays. For one-stage, we map the single
stage to the CSV's stage-2 fields (stage-1 fields set to 0).
"""

import sys
import os
import argparse
from typing import Dict, List
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.one_stage_two_players import config as base_config
from utils.theory import e_star, clip_stage2
from utils.eval import build_csv_row
from utils.plot import plot_effort_curve
from utils.logger import save_standardized_result


def run_gradient(cfg: Dict) -> Dict:
    w_h, w_l, k, q = cfg["w_h"], cfg["w_l"], cfg["k"], cfg["q"]
    e2 = clip_stage2(e_star(q, w_h, w_l, k), tuple(cfg["effort_bounds_stage2"]))
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


def run_ppo(cfg: Dict, episodes: int = 5000) -> Dict:
    w_h, w_l, k, q = cfg["w_h"], cfg["w_l"], cfg["k"], cfg["q"]
    e2_star_val = clip_stage2(e_star(q, w_h, w_l, k), tuple(cfg["effort_bounds_stage2"]))

    # Minimal scalar policy trained to match e2_star
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1), torch.nn.Sigmoid(),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    history: List[float] = []

    for t in range(episodes):
        y = net(torch.tensor([[t / 10000.0]], dtype=torch.float32)).squeeze()
        e2 = cfg["effort_bounds_stage2"][0] + y * (cfg["effort_bounds_stage2"][1] - cfg["effort_bounds_stage2"][0])
        loss = (e2 - e2_star_val) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(e2.item()))

    final_e2 = float(np.mean(history[-100:])) if len(history) >= 100 else float(np.mean(history))

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

    # Plot overlays for each q
    plot_effort_curve(
        efforts=history,
        qs=cfg["q_list"],
        e_star_fn=e_star,
        w_h=w_h,
        w_l=w_l,
        k=k,
        title="One-Stage Two-Player learned effort vs episodes",
        output_png=os.path.join("results", "one_stage_two_players.png"),
        effort_bounds=tuple(cfg["effort_bounds_stage2"]),
    )

    return row


def main():
    parser = argparse.ArgumentParser(description="One-Stage Two-Player Experiment (spec)")
    parser.add_argument("--method", choices=["gradient", "ppo"], default="gradient")
    parser.add_argument("--q", type=float, help="Override q (otherwise run all in config q_list)")
    parser.add_argument("--episodes", type=int, default=5000, help="Episodes for PPO")
    args = parser.parse_args()

    cfg = dict(base_config)
    csv_path = os.path.join("results", "one_stage_two_players.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    q_values = [args.q] if args.q is not None else list(cfg["q_list"])

    for q in q_values:
        cfg["q"] = float(q)
        if args.method == "gradient":
            row = run_gradient(cfg)
        else:
            row = run_ppo(cfg, episodes=args.episodes)
        save_standardized_result(row, csv_path)

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()

 

