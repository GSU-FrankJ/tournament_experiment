#!/usr/bin/env python3
"""
Asymmetric-Cost Two-Player (True PPO)
====================================

Runs gradient/theory and a proper PPO self-play for the setting:
  - y_i = e_i + ε_i,  ε_i ~ U(-q, q)
  - c(e_i) = k_i e_i^2 with k1 < k2, and l1 = l2
  - E[u_i] = w_L + p_i(win)(w_H - w_L) - k_i e_i^2

This script uses the exact triangular CDF for win probabilities (via
utils.prob) and a clean PPO implementation (agents.ppo_two_players_clean).
Output rows are saved to results/tables/asymmetric_cost_two_players.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.asymmetric_cost_two_players import config as base_config
from envs.different_cost_env import DifferentCostEnv
from agents.ppo_two_players_clean import PPOTwoPlayersBandit, PPOConfig
from utils.theory import e_star_two_players_asymmetric_cost
from utils.logger import save_result


def _state_tensor(agent: PPOTwoPlayersBandit, q: float, k_self: float, k_opp: float, w_h: float, w_l: float) -> torch.Tensor:
    # Normalize features to roughly [0,1]
    q_norm = float(q) / 60.0
    ks_norm = float(k_self) / 1e-3
    ko_norm = float(k_opp) / 1e-3
    wgap_norm = (float(w_h) - float(w_l)) / 10.0
    s = torch.tensor([q_norm, ks_norm, ko_norm, wgap_norm], dtype=torch.float32, device=agent.device)
    return s.unsqueeze(0)


def run_gradient(cfg: Dict) -> Dict:
    k1, k2, q, w_h, w_l = cfg["k1"], cfg["k2"], cfg["q"], cfg["w_h"], cfg["w_l"]
    e1, e2 = e_star_two_players_asymmetric_cost(q, w_h, w_l, k1, k2)
    row = {
        "model": "gradient",
        "q": q,
        "k1": k1,
        "k2": k2,
        "w_h": w_h,
        "w_l": w_l,
        "theoretical_e1": e1,
        "theoretical_e2": e2,
        "final_e1": e1,
        "final_e2": e2,
        "gap1": 0.0,
        "gap2": 0.0,
        "avg_gap": 0.0,
        "episodes": 0,
    }
    return row


def run_ppo(
    cfg: Dict,
    episodes: int = 5000,
    train_qs: Optional[List[float]] = None,
    eval_qs: Optional[List[float]] = None,
    steps_per_update_override: Optional[int] = None,
    log_every: int = 10,
) -> List[Dict]:
    w_h, w_l = cfg["w_h"], cfg["w_l"]
    k1, k2 = cfg["k1"], cfg["k2"]
    effort_bounds = tuple(cfg.get("effort_bounds_stage2", cfg.get("effort_range", (0.0, 200.0))))
    train_qs = list(train_qs if train_qs is not None else cfg["q_list"])  # type: ignore
    eval_qs = list(eval_qs if eval_qs is not None else train_qs)

    ppo_cfg = PPOConfig(
        steps_per_update=steps_per_update_override or 8192,
        epochs=10,
        minibatch_size=1024,
        state_dim=4,  # [q_norm, k_self_norm, k_opp_norm, wgap_norm]
        hidden=64,
        opponent_sync_interval=1,
        opponent_ema_tau=0.0,
        entropy_coef=0.01,
        lr=3e-4,
        clip_eps=0.25,
    )
    agent = PPOTwoPlayersBandit(effort_bounds=effort_bounds, cfg=ppo_cfg)

    rng = np.random.default_rng(cfg.get("seed", 42))
    total_steps_target = int(episodes)
    steps_done = 0
    update_idx = 0

    # Pre-compute planned updates for progress display
    total_updates = (total_steps_target + ppo_cfg.steps_per_update - 1) // ppo_cfg.steps_per_update
    print(f"Starting PPO training: total_steps={total_steps_target}, steps_per_update={ppo_cfg.steps_per_update}, planned_updates={total_updates}", flush=True)

    while steps_done < total_steps_target:
        steps_this = min(ppo_cfg.steps_per_update, total_steps_target - steps_done)
        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env = DifferentCostEnv(w_h=w_h, w_l=w_l, k1=k1, k2=k2, q=q, effort_bounds=effort_bounds, seed=cfg.get("seed", 42))
            s1 = _state_tensor(agent, q, k1, k2, w_h, w_l)
            s2 = _state_tensor(agent, q, k2, k1, w_h, w_l)
            a1_norm, e1, logp1, v1 = agent.act(s1)
            a2_norm, e2, logp2, v2 = agent.act(s2)
            _, rewards, _, done, _ = env.step((torch.tensor([float(e1.item())]), torch.tensor([float(e2.item())])))
            agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
        agent.update()
        update_idx += 1
        steps_done += steps_this

        # Periodic evaluation / progress print
        if log_every and (update_idx % log_every == 0 or update_idx == 1):
            try:
                q_eval_list = eval_qs if eval_qs else [cfg["q"]]
                for q_eval in q_eval_list:
                    e1_star, e2_star = e_star_two_players_asymmetric_cost(q_eval, w_h, w_l, k1, k2)
                    s1_eval = _state_tensor(agent, float(q_eval), k1, k2, w_h, w_l)
                    s2_eval = _state_tensor(agent, float(q_eval), k2, k1, w_h, w_l)
                    with torch.no_grad():
                        # Player 1 role
                        d1, _ = agent.net.dist(s1_eval)
                        if (d1.concentration1.item() > 1.0 and d1.concentration0.item() > 1.0):
                            a1 = (d1.concentration1 - 1.0) / (d1.concentration1 + d1.concentration0 - 2.0)
                            a1 = float(a1.squeeze().clamp(0,1).item())
                        else:
                            a1 = float(d1.sample((512,)).mean().item())
                        e1_eval = float(effort_bounds[0] + a1 * (effort_bounds[1] - effort_bounds[0]))
                        # Player 2 role
                        d2, _ = agent.net.dist(s2_eval)
                        if (d2.concentration1.item() > 1.0 and d2.concentration0.item() > 1.0):
                            a2 = (d2.concentration1 - 1.0) / (d2.concentration1 + d2.concentration0 - 2.0)
                            a2 = float(a2.squeeze().clamp(0,1).item())
                        else:
                            a2 = float(d2.sample((512,)).mean().item())
                        e2_eval = float(effort_bounds[0] + a2 * (effort_bounds[1] - effort_bounds[0]))
                    gap1 = abs(e1_eval - e1_star)
                    gap2 = abs(e2_eval - e2_star)
                    print(f"[Update {update_idx}/{total_updates}] q={q_eval}: e1*={e1_star:.2f} e1={e1_eval:.2f} gap1={gap1:.2f} | e2*={e2_star:.2f} e2={e2_eval:.2f} gap2={gap2:.2f} | steps={steps_done}/{total_steps_target}", flush=True)
            except Exception:
                pass

    # Build rows for each q in eval
    rows: List[Dict] = []
    for q in eval_qs:
        e1_star, e2_star = e_star_two_players_asymmetric_cost(q, w_h, w_l, k1, k2)

        # Evaluate via Beta mode for each role
        s1_eval = _state_tensor(agent, float(q), k1, k2, w_h, w_l)
        s2_eval = _state_tensor(agent, float(q), k2, k1, w_h, w_l)
        with torch.no_grad():
            dist1, _ = agent.net.dist(s1_eval)
            a1 = (dist1.concentration1 - 1.0) / (dist1.concentration1 + dist1.concentration0 - 2.0)
            a1 = a1.clamp(0.0, 1.0).squeeze().item() if (dist1.concentration1.item() > 1.0 and dist1.concentration0.item() > 1.0) else dist1.sample((512,)).mean().item()
            e1_final = float(effort_bounds[0] + a1 * (effort_bounds[1] - effort_bounds[0]))

            dist2, _ = agent.net.dist(s2_eval)
            a2 = (dist2.concentration1 - 1.0) / (dist2.concentration1 + dist2.concentration0 - 2.0)
            a2 = a2.clamp(0.0, 1.0).squeeze().item() if (dist2.concentration1.item() > 1.0 and dist2.concentration0.item() > 1.0) else dist2.sample((512,)).mean().item()
            e2_final = float(effort_bounds[0] + a2 * (effort_bounds[1] - effort_bounds[0]))

        gap1 = abs(e1_final - e1_star)
        gap2 = abs(e2_final - e2_star)
        rows.append({
            "model": "ppo",
            "q": float(q),
            "k1": k1,
            "k2": k2,
            "w_h": w_h,
            "w_l": w_l,
            "theoretical_e1": e1_star,
            "theoretical_e2": e2_star,
            "final_e1": e1_final,
            "final_e2": e2_final,
            "gap1": gap1,
            "gap2": gap2,
            "avg_gap": (gap1 + gap2) / 2.0,
            "episodes": int(episodes),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Asymmetric-cost two-player (true PPO)")
    parser.add_argument("--method", choices=["gradient", "ppo"], default="ppo")
    parser.add_argument("--q", type=float, help="Optional single q value to run")
    parser.add_argument("--episodes", type=int, default=20000, help="Total environment steps for PPO")
    parser.add_argument("--steps-per-update", type=int, default=None, help="Override PPO steps_per_update for faster progress updates")
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N updates (1 for every update)")
    args = parser.parse_args()

    cfg = dict(base_config)
    out_csv = os.path.join("results", "tables", "asymmetric_cost_two_players.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if args.method == "gradient":
        q_vals = [args.q] if args.q is not None else list(cfg["q_list"])  # type: ignore
        for q in q_vals:
            cfg["q"] = float(q)
            row = run_gradient(cfg)
            save_result(row, out_csv)
    else:
        train_qs = [args.q] if args.q is not None else list(cfg["q_list"])  # type: ignore
        eval_qs = train_qs
        rows = run_ppo(
            cfg,
            episodes=args.episodes,
            train_qs=train_qs,
            eval_qs=eval_qs,
            steps_per_update_override=args.steps_per_update,
            log_every=args.log_every,
        )
        for row in rows:
            save_result(row, out_csv)

    print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    main()
