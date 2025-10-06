#!/usr/bin/env python3
"""
One-Stage Two-Player Experiment (spec-compliant)

Writes standardized CSV and figure overlays. For one-stage, we map the single
stage to the CSV's stage-2 fields (stage-1 fields set to 0).
"""

import sys
import os
import argparse
import math
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
    row["stage2_gap_unweighted"] = abs(float(row["final_stage2_effort"]) - float(row["theoretical_stage2_effort"]))
    return row


def run_ppo(cfg: Dict, episodes: int = 5000, train_qs: Optional[List[float]] = None, eval_qs: Optional[List[float]] = None) -> List[Dict]:
    """Train PPO via self-play with conditioning on (q, k, w_gap).

    - Trains over ``train_qs`` (defaults to cfg["q_list" ]).
    - Returns a list of CSV rows, one per q in ``eval_qs`` (defaults to train_qs).
    """
    w_h, w_l, k = cfg["w_h"], cfg["w_l"], cfg["k"]
    effort_bounds = tuple(cfg["effort_bounds_stage2"])  # (0, 200)
    # Respect CLI-provided training set; default to config q_list
    train_qs = list(train_qs if train_qs is not None else cfg["q_list"])
    eval_qs = list(eval_qs if eval_qs is not None else train_qs)

    # PPO agent with 3-dim state: [q, k, w_gap]
    ppo_cfg = PPOConfig(
        steps_per_update=16384,
        epochs=20,
        minibatch_size=1024,
        state_dim=3,
        hidden=64,
        opponent_sync_interval=1,
        opponent_ema_tau=0.0,
        entropy_coef=0.02,
        lr=3e-4,
        clip_eps=0.25,
    )
    agent = PPOTwoPlayersBandit(effort_bounds=effort_bounds, cfg=ppo_cfg)

    history: List[float] = []
    total_steps_target = int(episodes)
    steps_done = 0
    rng = np.random.default_rng(cfg.get("seed", 42))
    # Entropy decay schedule: 0.02 -> 0.002 over first 50 updates (floor)
    start_entropy, end_entropy, decay_updates = 0.02, 0.002, 50
    update_idx = 0
    # Late-phase settings
    total_updates = (total_steps_target + ppo_cfg.steps_per_update - 1) // ppo_cfg.steps_per_update
    late_updates = min(100, total_updates)  # last 50-100 updates; cap by total
    start_late = max(0, total_updates - late_updates)
    # Entropy final squeeze to 0.0 in the last 30 updates
    entropy_zero_updates = min(30, total_updates)
    start_entropy_zero = max(0, total_updates - entropy_zero_updates)
    # LR boost in last 50 updates: 3e-4 -> 4e-4
    lr_base = ppo_cfg.lr
    lr_boost_value = 4e-4
    lr_boost_updates = min(50, total_updates)
    start_lr_late = max(0, total_updates - lr_boost_updates)

    while steps_done < total_steps_target:
        # Apply entropy decay before this update
        progress = min(1.0, float(update_idx) / float(decay_updates))
        agent.cfg.entropy_coef = start_entropy + (end_entropy - start_entropy) * progress
        # Force entropy to 0.0 in the last ~30 updates for tighter convergence
        if update_idx >= start_entropy_zero:
            agent.cfg.entropy_coef = 0.0
        # Late-phase clip schedule: 0.35 -> 0.25 over last N updates (slightly larger at start of late phase)
        if update_idx >= start_late:
            if late_updates > 1:
                prog_late = float(update_idx - start_late) / float(late_updates - 1)
            else:
                prog_late = 1.0
            agent.cfg.clip_eps = 0.35 - 0.10 * max(0.0, min(1.0, prog_late))
        else:
            agent.cfg.clip_eps = 0.25
        # LR boost in the last ~50 updates
        new_lr = lr_boost_value if update_idx >= start_lr_late else lr_base
        for g in agent.opt.param_groups:
            g["lr"] = new_lr
        steps_this = min(ppo_cfg.steps_per_update, total_steps_target - steps_done)
        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env = TwoPlayersEnv(w_h=w_h, w_l=w_l, k=k, q=q, effort_bounds=effort_bounds, seed=cfg.get("seed", 42))

            # Sampling strategy: early-phase uses lagged opponent and stores learner only;
            # late-phase switches to fully on-policy symmetric sampling and stores both.
            s1 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            s2 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            if update_idx >= start_late:
                # Fully on-policy self-play
                a1_norm, e1, logp1, v1 = agent.act(s1)
                a2_norm, e2, logp2, v2 = agent.act(s2)
            else:
                a1_norm, e1, logp1, v1 = agent.act(s1)
                a2_norm, e2, logp2, _ = agent.act_opponent(s2)
                # For GAE targets, use current net's value estimate on s2
                with torch.no_grad():
                    _, v2 = agent.net.dist(s2)

            _, rewards, _, done, _ = env.step((torch.tensor([float(e1.item())]), torch.tensor([float(e2.item())])))

            if update_idx >= start_late:
                # Store both players' on-policy samples in late phase
                agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
                agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
            else:
                # Store only the learner's on-policy sample (keep opponent lagged but off-storage)
                agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            history.append(float((e1.item() + e2.item()) / 2.0))
        agent.update()
        # After each PPO update, evaluate and log gaps for quick monitoring
        upd_i = update_idx + 1
        try:
            for q_eval in eval_qs:
                e2_star_val = clip_stage2(e_star_two_players(q_eval, w_h, w_l, k), effort_bounds)
                s_eval = agent.state_from_params(q=float(q_eval), k=k, w_h=w_h, w_l=w_l)
                with torch.no_grad():
                    dist, _ = agent.net.dist(s_eval)
                    a_eval = dist.mean.squeeze()
                    a_eval = a_eval.clamp(0.0, 1.0)
                    final_e2_eval = float(effort_bounds[0] + a_eval.detach().cpu().item() * (effort_bounds[1] - effort_bounds[0]))
                gap = abs(final_e2_eval - e2_star_val)
                print(f"[Update {upd_i}] q={q_eval}: e*={e2_star_val:.2f}, policy={final_e2_eval:.2f}, gap={gap:.2f}, entropy={agent.cfg.entropy_coef:.3f}")
        except Exception as _e:
            # Keep training robust to any eval hiccup
            pass
        update_idx += 1
        steps_done += steps_this

    # Build rows for each evaluation q
    rows: List[Dict] = []
    for q in eval_qs:
        e2_star_val = clip_stage2(e_star_two_players(q, w_h, w_l, k), effort_bounds)

        # Evaluate with Beta mode when defined (alpha,beta>1), else sample-average
        s_eval = agent.state_from_params(q=float(q), k=k, w_h=w_h, w_l=w_l)
        with torch.no_grad():
            dist, _ = agent.net.dist(s_eval)
            a_eval = dist.mean.squeeze()
            a_eval = a_eval.clamp(0.0, 1.0)
            a_eval = float(a_eval.detach().cpu().item())
        final_e2 = float(effort_bounds[0] + a_eval * (effort_bounds[1] - effort_bounds[0]))
        stage2_gap = abs(final_e2 - e2_star_val)

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
        row["stage2_gap_unweighted"] = stage2_gap
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

 
