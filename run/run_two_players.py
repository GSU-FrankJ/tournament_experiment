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


def _symmetric_fd_gradient(env: TwoPlayersEnv, e: float, eps: float = 0.1) -> float:
    """Central-difference ∂Eu/∂e_i at symmetric profile (e, e)."""
    lo, hi = env.effort_low, env.effort_high
    e_plus = max(lo, min(hi, e + eps))
    e_minus = max(lo, min(hi, e - eps))
    u_plus = env.expected_utility(e_plus, e)
    u_minus = env.expected_utility(e_minus, e)
    return (u_plus - u_minus) / (2.0 * eps)


def gradient_descent_two_players(
    cfg: Dict,
    *,
    lr: float = 0.1,
    steps: int = 2000,
    eps: float = 0.1,
    tol: float = 1e-4,
    log: bool = True,
) -> tuple[float, Dict[str, float]]:
    """Symmetric gradient descent to match experiment plan requirements."""
    effort_bounds = tuple(cfg["effort_bounds_stage2"])
    env = TwoPlayersEnv(
        w_h=cfg["w_h"],
        w_l=cfg["w_l"],
        k=cfg["k"],
        q=cfg["q"],
        effort_bounds=effort_bounds,
        seed=cfg.get("seed", 42),
    )
    lo, hi = effort_bounds
    e_theory = float(e_star_two_players(cfg["q"], cfg["w_h"], cfg["w_l"], cfg["k"]))
    e = float(np.clip(e_theory, lo, hi))
    history = {
        "init_e": e,
        "final_grad": 0.0,
        "iterations": 0.0,
    }

    for step in range(1, steps + 1):
        g = _symmetric_fd_gradient(env, e, eps=eps)
        e = float(np.clip(e + lr * g, lo, hi))
        history["iterations"] = float(step)
        history["final_grad"] = float(g)
        if log and (step == 1 or step % 250 == 0 or step == steps):
            print(f"[gradient-2p] step={step:05d} effort={e:.6f} grad={g:.6f}")
        if abs(g) < tol:
            if log:
                print(f"[gradient-2p] converged at step={step} with |grad|={abs(g):.6g}")
            break

    return e, history


def run_gradient(
    cfg: Dict,
    *,
    lr: float = 0.1,
    steps: int = 2000,
    grad_eps: float = 0.1,
    tol: float = 1e-4,
    log: bool = True,
) -> Dict:
    w_h, w_l, k, q = cfg["w_h"], cfg["w_l"], cfg["k"], cfg["q"]
    theoretical_e = clip_stage2(e_star_two_players(q, w_h, w_l, k), tuple(cfg["effort_bounds_stage2"]))
    final_e, meta = gradient_descent_two_players(
        cfg,
        lr=lr,
        steps=steps,
        eps=grad_eps,
        tol=tol,
        log=log,
    )
    if log:
        probes = {
            "theory": theoretical_e,
            "final": final_e,
            "midpoint": 0.5 * (theoretical_e + final_e),
        }
        env = TwoPlayersEnv(w_h=w_h, w_l=w_l, k=k, q=q, effort_bounds=tuple(cfg["effort_bounds_stage2"]), seed=cfg.get("seed", 42))
        for label, effort in probes.items():
            g_val = _symmetric_fd_gradient(env, effort, eps=max(grad_eps, 1e-3))
            print(f"[gradient-2p] probe={label} effort={effort:.6f} dU/de={g_val:.6f}")
        print(f"[gradient-2p] meta: iterations={meta['iterations']:.0f} final_grad={meta['final_grad']:.6f}")

    row = build_csv_row(
        stage1_weight=cfg["stage1_weight"],
        stage2_weight=cfg["stage2_weight"],
        k1=cfg["k1"],
        k2=cfg["k2"],
        information_revelation=cfg.get("information_revelation", "none"),
        theoretical_stage1_effort=0.0,
        theoretical_stage2_effort=theoretical_e,
        model_training="gradient",
        final_stage1_effort=0.0,
        final_stage2_effort=final_e,
        episodes=0,
    )
    row["stage2_gap_unweighted"] = abs(float(row["final_stage2_effort"]) - float(row["theoretical_stage2_effort"]))
    row["gradient_iterations"] = meta["iterations"]
    row["gradient_final_grad"] = meta["final_grad"]
    return row


def run_ppo(cfg: Dict, episodes: int = 921600, train_qs: Optional[List[float]] = None, eval_qs: Optional[List[float]] = None) -> List[Dict]:
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
        steps_per_update=3072,
        epochs=12,
        minibatch_size=448,
        state_dim=3,
        hidden=128,
        opponent_sync_interval=1,
        opponent_ema_tau=0.0,
        entropy_coef=0.015,
        lr=2.2e-4,
        clip_eps=0.28,
    )
    agent = PPOTwoPlayersBandit(effort_bounds=effort_bounds, cfg=ppo_cfg)

    history: List[float] = []
    total_steps_target = int(episodes)
    steps_done = 0
    rng = np.random.default_rng(cfg.get("seed", 42))
    # Entropy / LR schedules: hold high values until ~2/3 progress, then anneal
    entropy_start, entropy_hold, entropy_final = 0.015, 0.01, 0.002
    update_idx = 0
    # Late-phase settings
    total_updates = (total_steps_target + ppo_cfg.steps_per_update - 1) // ppo_cfg.steps_per_update
    hold_fraction = 2.0 / 3.0
    hold_updates = max(1, int(math.ceil(total_updates * hold_fraction)))
    tail_updates = max(1, total_updates - hold_updates)
    # Learning rate schedule: hold at 2.2e-4, then anneal to 1.5e-4
    lr_hold = ppo_cfg.lr
    lr_final = 2.0e-4
    # Clip cosine schedule: 0.28 -> 0.18
    clip_max = 0.28
    clip_min = 0.18
    # Self-play lag schedule: warmup 80 updates then fade across 40
    lag_warmup_updates = 80 if total_updates >= 80 else max(1, total_updates // 2)
    lag_fade_updates = 40 if total_updates >= 120 else max(1, total_updates // 3)

    last_update_metrics: Optional[Dict[str, float]] = None

    while steps_done < total_steps_target:
        # Entropy: hold near 0.01 for first ~2/3 updates, then ramp to zero
        if update_idx < hold_updates:
            if hold_updates > 1:
                hold_progress = float(update_idx) / float(hold_updates - 1)
            else:
                hold_progress = 1.0
            hold_progress = max(0.0, min(1.0, hold_progress))
            agent.cfg.entropy_coef = entropy_start + (entropy_hold - entropy_start) * hold_progress
        else:
            tail_progress = float(update_idx - hold_updates) / float(max(1, tail_updates - 1))
            tail_progress = max(0.0, min(1.0, tail_progress))
            agent.cfg.entropy_coef = entropy_hold + (entropy_final - entropy_hold) * tail_progress
        # Clip cosine schedule across total updates
        if total_updates > 1:
            clip_progress = float(update_idx) / float(total_updates - 1)
        else:
            clip_progress = 1.0
        clip_cosine = 0.5 * (1.0 + math.cos(math.pi * clip_progress))
        agent.cfg.clip_eps = clip_min + (clip_max - clip_min) * clip_cosine
        # LR cosine decay across total updates
        if update_idx < hold_updates:
            new_lr = lr_hold
        else:
            lr_tail_progress = float(update_idx - hold_updates) / float(max(1, tail_updates - 1))
            lr_tail_progress = max(0.0, min(1.0, lr_tail_progress))
            new_lr = lr_hold + (lr_final - lr_hold) * lr_tail_progress
        for g in agent.opt.param_groups:
            g["lr"] = new_lr
        # Determine lagged-opponent mixing probability for this PPO update
        if update_idx < lag_warmup_updates:
            lag_prob = 1.0
        elif update_idx < lag_warmup_updates + lag_fade_updates:
            denom = max(1, lag_fade_updates - 1)
            lag_phase = update_idx - lag_warmup_updates
            lag_prob = max(0.0, 1.0 - (lag_phase / denom))
        else:
            lag_prob = 0.0
        steps_this = min(ppo_cfg.steps_per_update, total_steps_target - steps_done)
        for _ in range(steps_this):
            q = float(rng.choice(train_qs))
            env = TwoPlayersEnv(w_h=w_h, w_l=w_l, k=k, q=q, effort_bounds=effort_bounds, seed=cfg.get("seed", 42))

            # Sampling strategy: early-phase mixes in lagged opponent samples (no stochastic env noise);
            # late-phase switches to fully on-policy symmetric sampling and stores both trajectories.
            s1 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            s2 = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
            a1_norm, e1, logp1, v1 = agent.act(s1)
            use_opponent = (lag_prob > 0.0) and (rng.random() < lag_prob)
            if use_opponent:
                a2_norm, e2, _, _ = agent.act_opponent(s2)
                logp2 = None
                v2 = None
            else:
                a2_norm, e2, logp2, v2 = agent.act(s2)

            _, rewards, _, done, _ = env.step((torch.tensor([float(e1.item())]), torch.tensor([float(e2.item())])))

            agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            if not use_opponent:
                agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
            history.append(float((e1.item() + e2.item()) / 2.0))
        last_update_metrics = agent.update()
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
                    alpha_mean = float(dist.concentration1.mean().item())
                    beta_mean = float(dist.concentration0.mean().item())
                gap = abs(final_e2_eval - e2_star_val)
                kl_val = last_update_metrics.get("approx_kl", float("nan")) if last_update_metrics else float("nan")
                adv_mean = last_update_metrics.get("adv_mean", float("nan")) if last_update_metrics else float("nan")
                print(
                    f"[Update {upd_i}] q={q_eval}: e*={e2_star_val:.2f}, policy={final_e2_eval:.2f}, gap={gap:.2f}, "
                    f"entropy={agent.cfg.entropy_coef:.3f}, lag_prob={lag_prob:.2f}, adv_mean={adv_mean:.4f}, "
                    f"approx_kl={kl_val:.4f}, alpha_mean={alpha_mean:.2f}, beta_mean={beta_mean:.2f}"
                )
        except Exception as _e:
            # Keep training robust to any eval hiccup
            pass
        update_idx += 1
        steps_done += steps_this

    # Build rows for each evaluation q
    rows: List[Dict] = []
    for q in eval_qs:
        e2_star_val = clip_stage2(e_star_two_players(q, w_h, w_l, k), effort_bounds)

        # Evaluate via Beta mean (mode intentionally not used per experiment doc)
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
    parser.add_argument("--episodes", type=int, default=921600, help="Episodes for PPO (default 921600 ≈ 300 updates at 3072 steps/update)")
    parser.add_argument("--grad-lr", type=float, default=0.1, help="Learning rate for gradient descent solver.")
    parser.add_argument("--grad-steps", type=int, default=2000, help="Maximum gradient descent iterations.")
    parser.add_argument("--grad-epsilon", type=float, default=0.1, help="Finite-difference epsilon for gradients.")
    parser.add_argument("--grad-tol", type=float, default=1e-4, help="Terminate when |grad| < tol.")
    parser.add_argument("--k", type=float, help="Override symmetric cost k.")
    parser.add_argument("--w_h", type=float, help="Override high prize w_h.")
    parser.add_argument("--w_l", type=float, help="Override low prize w_l.")
    parser.add_argument("--effort-range", type=float, nargs=2, metavar=("LO", "HI"), help="Override symmetric effort bounds.")
    parser.add_argument("--seed", type=int, help="Override RNG seed.")
    args = parser.parse_args()

    cfg = dict(base_config)
    if args.k is not None:
        cfg["k"] = float(args.k)
        cfg["k1"] = float(args.k)
        cfg["k2"] = float(args.k)
    if args.w_h is not None:
        cfg["w_h"] = float(args.w_h)
    if args.w_l is not None:
        cfg["w_l"] = float(args.w_l)
    if args.effort_range is not None:
        lo, hi = args.effort_range
        bounds = [float(lo), float(hi)]
        cfg["effort_bounds_stage2"] = bounds
        cfg["effort_range"] = bounds
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    csv_path = os.path.join("results", "one_stage_two_players.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if args.method == "gradient":
        q_values = [args.q] if args.q is not None else list(cfg["q_list"])
        for q in q_values:
            cfg["q"] = float(q)
            if args.grad_epsilon < 1e-4:
                print(f"[gradient-2p] warning: grad-epsilon={args.grad_epsilon:.2e} may be too small for stable finite differences.", flush=True)
            row = run_gradient(
                cfg,
                lr=args.grad_lr,
                steps=args.grad_steps,
                grad_eps=args.grad_epsilon,
                tol=args.grad_tol,
                log=True,
            )
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

 
