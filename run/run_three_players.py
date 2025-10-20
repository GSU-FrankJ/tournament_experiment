#!/usr/bin/env python3
"""
One-Stage Three-Player Experiment (gradient + true PPO)

Three identical competitors; one winner, two losers. Stage-1 mapping to our
standard CSV uses stage-2 fields only (stage-1 is 0).
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.one_stage_three_players import config as base_config
from utils.theory import e_star_three_players, clip_stage2
from utils.eval import build_csv_row
from utils.plot import plot_effort_curve
from utils.logger import save_standardized_result
from envs.three_players_env import ThreePlayersEnv
from envs.one_stage_env import OneStageEnv
from agents.ppo_three_players_clean import PPOThreePlayersBandit, PPOConfig
from agents.gradient_solver import gradient_descent_solver


def _symmetric_mc_gradient(env: ThreePlayersEnv, e: float, eps: float = 1e-3) -> float:
    """Central-difference gradient of Eu wrt a single player's effort at
    symmetric profile (e, e, e) using env's Monte Carlo win prob.
    """
    p_plus, _, _ = env._win_probs(e + eps, e, e)
    p_minus, _, _ = env._win_probs(e - eps, e, e)
    u_plus = env.w_l + p_plus * (env.w_h - env.w_l) - env.k * (e + eps) ** 2
    u_minus = env.w_l + p_minus * (env.w_h - env.w_l) - env.k * (e - eps) ** 2
    return (u_plus - u_minus) / (2.0 * eps)


def gradient_descent_three_players(
    cfg: Dict,
    lr: float = 0.05,
    steps: int = 20000,
    eps: float = 1e-3,
    *,
    mc_samples: int = 20000,
    allow_near_symmetric_shortcut: bool = True,
    track_shortcut_stats: bool = False,
) -> tuple[float, ThreePlayersEnv]:
    """Gradient descent on symmetric effort using MC-based win probabilities."""
    env = ThreePlayersEnv(w_h=cfg["w_h"], w_l=cfg["w_l"], k=cfg["k"], q=cfg["q"],
                          effort_bounds=tuple(cfg["effort_bounds_stage2"]),
                          seed=cfg.get("seed", 42), mc_samples=int(mc_samples),
                          allow_near_symmetric_shortcut=allow_near_symmetric_shortcut,
                          track_shortcut_stats=track_shortcut_stats)
    lo, hi = env.effort_range
    # Start from theoretical value (faster) and clamp
    e = float(np.clip(e_star_three_players(cfg["q"], cfg["w_h"], cfg["w_l"], cfg["k"]), lo, hi))
    for _ in range(steps):
        g = _symmetric_mc_gradient(env, e, eps)
        e = float(np.clip(e + lr * g, lo, hi))
    return e, env


def run_gradient(
    cfg: Dict,
    *,
    lr: float = 0.05,
    steps: int = 5000,
    grad_eps: float = 0.1,
    mc_samples: int = 20000,
    disable_shortcut: bool = True,
    print_stats: bool = True,
) -> Dict:
    """Finite-difference gradient descent at symmetric profile.

    Uses the analytical utility wrapper in OneStageEnv (num_players=3) to get a
    smooth utility surface near symmetry.
    """
    w_h, w_l, k, q = cfg["w_h"], cfg["w_l"], cfg["k"], cfg["q"]
    e_theory = clip_stage2(e_star_three_players(q, w_h, w_l, k), tuple(cfg["effort_bounds_stage2"]))

    # Use MC-based gradient descent tailored for 3 players
    e_final, env = gradient_descent_three_players(
        cfg,
        lr=lr,
        steps=steps,
        eps=grad_eps,
        mc_samples=mc_samples,
        allow_near_symmetric_shortcut=not disable_shortcut,
        track_shortcut_stats=True,
    )

    if print_stats:
        stats = env.get_shortcut_stats()
        total_calls = stats["shortcut_hits"] + stats["full_path_calls"]
        pct_shortcut = (stats["shortcut_hits"] / total_calls * 100.0) if total_calls else 0.0
        print(
            f"[gradient] q={q:.3f} shortcut_hits={stats['shortcut_hits']} "
            f"full_path_calls={stats['full_path_calls']} ({pct_shortcut:.2f}% shortcut)"
        )
        probe_eps = max(grad_eps, 0.01 * q)
        probe_points = {
            "theory": float(e_theory),
            "final": float(e_final),
            "midpoint": float((float(e_theory) + float(e_final)) / 2.0),
        }
        for label, probe_e in probe_points.items():
            g_val = _symmetric_mc_gradient(env, probe_e, eps=probe_eps)
            print(f"[gradient] q={q:.3f} probe={label} effort={probe_e:.6f} dU/de={g_val:.6f}")

    row = build_csv_row(
        stage1_weight=cfg["stage1_weight"],
        stage2_weight=cfg["stage2_weight"],
        k1=cfg.get("k1", cfg["k"]),
        k2=cfg.get("k2", cfg["k"]),
        information_revelation=cfg.get("information_revelation", "none"),
        theoretical_stage1_effort=0.0,
        theoretical_stage2_effort=e_theory,
        model_training="gradient",
        final_stage1_effort=0.0,
        final_stage2_effort=float(e_final),
        episodes=0,
    )
    return row


def run_ppo(cfg: Dict, episodes: int = 8000, train_qs: Optional[List[float]] = None, eval_qs: Optional[List[float]] = None) -> List[Dict]:
    """Train PPO via self-play with 3 identical players.

    A single shared policy controls all three players. Each episode generates 3
    independent samples and we store all transitions.
    """
    w_h, w_l, k = cfg["w_h"], cfg["w_l"], cfg["k"]
    effort_bounds = tuple(cfg["effort_bounds_stage2"])  # (0, 200)
    train_qs = list(train_qs if train_qs is not None else cfg["q_list"])
    eval_qs = list(eval_qs if eval_qs is not None else train_qs)

    ppo_cfg = PPOConfig(
        steps_per_update=8192,
        epochs=20,
        minibatch_size=1024,
        state_dim=3,
        hidden=64,
        entropy_coef=0.01,
        lr=3e-4,
        clip_eps=0.25,
    )
    agent = PPOThreePlayersBandit(effort_bounds=effort_bounds, cfg=ppo_cfg)

    rng = np.random.default_rng(cfg.get("seed", 42))
    history: List[float] = []
    steps_target = int(episodes)
    steps_done = 0
    update_idx = 0

    while steps_done < steps_target:
        steps_this = min(ppo_cfg.steps_per_update, steps_target - steps_done)
        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env = ThreePlayersEnv(w_h=w_h, w_l=w_l, k=k, q=q, effort_bounds=effort_bounds, seed=cfg.get("seed", 42))
            s = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            # Sample three players from the same policy (on-policy self-play)
            a1, e1, logp1, v1 = agent.act(s)
            a2, e2, logp2, v2 = agent.act(s)
            a3, e3, logp3, v3 = agent.act(s)

            _, rewards, _, done, _ = env.step((torch.tensor([float(e1.item())]),
                                               torch.tensor([float(e2.item())]),
                                               torch.tensor([float(e3.item())])))
            # Store all 3 experiences
            agent.store(s, a1, logp1, float(rewards[0].item()), v1, bool(done))
            agent.store(s, a2, logp2, float(rewards[1].item()), v2, bool(done))
            agent.store(s, a3, logp3, float(rewards[2].item()), v3, bool(done))

            history.append(float((e1.item() + e2.item() + e3.item()) / 3.0))

        agent.update()
        update_idx += 1
        steps_done += steps_this

    # Evaluate for each q
    rows: List[Dict] = []
    for q in eval_qs:
        e_star_val = clip_stage2(e_star_three_players(q, w_h, w_l, k), effort_bounds)
        s_eval = agent.state_from_params(q=float(q), k=k, w_h=w_h, w_l=w_l)
        with torch.no_grad():
            dist, _ = agent.net.dist(s_eval)
            alpha = dist.concentration1.squeeze()
            beta = dist.concentration0.squeeze()
            if (alpha > 1.0 and beta > 1.0):
                a_eval = (alpha - 1.0) / (alpha + beta - 2.0)
            else:
                samples = dist.sample((512,)).squeeze(-1)
                a_eval = samples.mean()
            a_eval = float(a_eval.detach().cpu().item())
        final_e = float(effort_bounds[0] + a_eval * (effort_bounds[1] - effort_bounds[0]))

        row = build_csv_row(
            stage1_weight=cfg["stage1_weight"],
            stage2_weight=cfg["stage2_weight"],
            k1=cfg.get("k1", cfg["k"]),
            k2=cfg.get("k2", cfg["k"]),
            information_revelation=cfg.get("information_revelation", "none"),
            theoretical_stage1_effort=0.0,
            theoretical_stage2_effort=e_star_val,
            model_training="ppo",
            final_stage1_effort=0.0,
            final_stage2_effort=final_e,
            episodes=episodes,
        )
        rows.append(row)

    # Plot overlay of learning history with theoretical curve for last q list
    plot_effort_curve(
        efforts=history,
        qs=eval_qs,
        e_star_fn=e_star_three_players,
        w_h=w_h,
        w_l=w_l,
        k=k,
        title="One-Stage Three-Player learned effort vs episodes",
        output_png=os.path.join("results", "one_stage_three_players.png"),
        effort_bounds=effort_bounds,
    )

    return rows


def main():
    parser = argparse.ArgumentParser(description="One-Stage Three-Player Experiment")
    parser.add_argument("--method", choices=["gradient", "ppo"], default="gradient")
    parser.add_argument("--q", type=float, help="Override q (otherwise run all in config q_list)")
    parser.add_argument("--episodes", type=int, default=8000, help="Episodes for PPO")
    parser.add_argument(
        "--disable-near-symmetric-shortcut-for-grad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable the near-symmetric shortcut when computing gradients (default: disabled).",
    )
    parser.add_argument(
        "--grad-epsilon",
        type=float,
        default=0.1,
        help="Finite-difference epsilon for ∂U/∂e estimates in gradient descent.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=20000,
        help="Monte Carlo samples for gradient-mode ThreePlayersEnv.",
    )
    args = parser.parse_args()

    cfg = dict(base_config)
    csv_path = os.path.join("results", "one_stage_three_players.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if args.method == "gradient":
        q_values = [args.q] if args.q is not None else list(cfg["q_list"])
        for q in q_values:
            cfg["q"] = float(q)
            if (not args.disable_near_symmetric_shortcut_for_grad) and (args.grad_epsilon < 0.01 * cfg["q"]):
                print(
                    f"[gradient] warning: grad_epsilon={args.grad_epsilon:.3g} < 1% of q={cfg['q']:.3g}; "
                    "near-symmetric shortcut may flatten gradients. "
                    "Use --disable-near-symmetric-shortcut-for-grad to avoid this.",
                    flush=True,
                )
            row = run_gradient(
                cfg,
                lr=0.05,
                steps=5000,
                grad_eps=args.grad_epsilon,
                mc_samples=args.mc_samples,
                disable_shortcut=args.disable_near_symmetric_shortcut_for_grad,
                print_stats=True,
            )
            save_standardized_result(row, csv_path)
    else:
        train_qs = [args.q] if args.q is not None else list(cfg["q_list"])
        eval_qs = train_qs
        rows = run_ppo(cfg, episodes=args.episodes, train_qs=train_qs, eval_qs=eval_qs)
        for row in rows:
            save_standardized_result(row, csv_path)

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
