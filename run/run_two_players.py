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


def run_ppo(
    cfg: Dict,
    episodes: Optional[int] = None,
    train_qs: Optional[List[float]] = None,
    eval_qs: Optional[List[float]] = None,
    *,
    eval_symmetric: bool = True,
    eval_vs_opponent: bool = False,
    eval_vs_history: bool = False,
) -> List[Dict]:
    """Train PPO via self-play with conditioning on (q, k, w_gap).

    - Trains over ``train_qs`` (defaults to cfg["q_list" ]).
    - Returns a list of CSV rows, one per q in ``eval_qs`` (defaults to train_qs).
    """
    if episodes is None:
        episodes = int(cfg.get("episodes", 1_800_000))
    else:
        episodes = int(episodes)

    w_h, w_l, k = cfg["w_h"], cfg["w_l"], cfg["k"]
    effort_bounds = tuple(cfg["effort_bounds_stage2"])  # (0, 200)
    # Respect CLI-provided training set; default to config q_list
    train_qs = list(train_qs if train_qs is not None else cfg["q_list"])
    eval_qs = list(eval_qs if eval_qs is not None else train_qs)

    # PPO agent with 3-dim state: [q, k, w_gap]
    ppo_cfg = PPOConfig(
        steps_per_update=int(cfg.get("steps_per_update", 4096)),
        epochs=int(cfg.get("update_epochs", 6)),
        minibatch_size=int(cfg.get("minibatch_size", 1024)),
        state_dim=3,
        hidden=128,
        opponent_mode=cfg.get("opponent_mode", "periodic"),
        opponent_sync_interval=int(cfg.get("opponent_sync_interval", 2)),
        opponent_ema_tau=float(cfg.get("opponent_ema_tau", 0.20)),
        opponent_snapshot_keep=int(cfg.get("opponent_snapshot_keep", 10)),
        opponent_history_sample_p=float(cfg.get("opponent_history_sample_p", 0.3)),
        entropy_coef=float(cfg.get("entropy_coef_start", 0.02)),
        lr=float(cfg.get("lr_start", 3e-4)),
        clip_eps=float(cfg.get("clip_range_start", 0.30)),
    )
    agent = PPOTwoPlayersBandit(effort_bounds=effort_bounds, cfg=ppo_cfg)
    agent.cfg.entropy_coef = float(cfg.get("entropy_coef_start", agent.cfg.entropy_coef))
    agent.cfg.clip_eps = float(cfg.get("clip_range_start", agent.cfg.clip_eps))
    for g in agent.opt.param_groups:
        g["lr"] = float(cfg.get("lr_start", ppo_cfg.lr))

    history: List[float] = []
    total_steps_target = int(episodes)
    steps_done = 0
    rng = np.random.default_rng(cfg.get("seed", 42))
    # Entropy / LR schedules: hold high values until ~2/3 progress, then anneal
    entropy_start = float(cfg.get("entropy_coef_start", agent.cfg.entropy_coef))
    entropy_hold = float(cfg.get("entropy_coef_hold", entropy_start))
    entropy_final = float(cfg.get("entropy_coef_end", 0.005))
    update_idx = 0
    # Late-phase settings
    total_updates = (total_steps_target + ppo_cfg.steps_per_update - 1) // ppo_cfg.steps_per_update
    hold_fraction = 2.0 / 3.0
    hold_updates = max(1, int(math.ceil(total_updates * hold_fraction)))
    tail_updates = max(1, total_updates - hold_updates)
    # Learning rate schedule: hold at starting value, then anneal to final value
    lr_hold = float(cfg.get("lr_start", agent.cfg.lr))
    lr_final = float(cfg.get("lr_end", 2e-4))
    # Clip schedule parameters
    clip_max = float(cfg.get("clip_range_start", agent.cfg.clip_eps))
    clip_min = float(cfg.get("clip_range_end", 0.15))
    # Self-play lag schedule: short warmup then fade
    lag_warmup_updates = max(0, int(cfg.get("lag_warmup_updates", 10)))
    lag_fade_cfg = cfg.get("lag_fade_updates")
    lag_fade_updates = max(0, int(lag_fade_cfg)) if lag_fade_cfg is not None else max(1, total_updates // 3)

    history_prob_start = float(cfg.get("opponent_history_sample_p", agent.opponent_history_sample_p))
    history_prob_end = float(cfg.get("opponent_history_sample_p_end", history_prob_start))
    agent.opponent_history_sample_p = history_prob_start

    clip_factor = 1.0
    lr_factor = 1.0
    clip_floor = 0.10
    clip_ceiling = 0.45
    min_lr = 5e-5
    max_lr = 5e-4
    target_kl = float(cfg.get("target_kl", 0.01))
    kl_low = 0.5 * target_kl
    kl_high = 3.0 * target_kl

    last_update_metrics: Optional[Dict[str, float]] = None
    eval_every = int(cfg.get("eval_every_updates", 20) or 0)
    es_abs = float(cfg.get("early_stop_abs_err", 1.0))
    es_pat = int(cfg.get("early_stop_patience", 5) or 0)
    es_counter = 0
    early_stop_triggered = False

    while steps_done < total_steps_target:
        if total_updates > 1:
            hist_progress = float(update_idx) / float(total_updates - 1)
            hist_progress = max(0.0, min(1.0, hist_progress))
        else:
            hist_progress = 1.0
        agent.opponent_history_sample_p = history_prob_start + (history_prob_end - history_prob_start) * hist_progress

        # Entropy: hold high for first ~2/3 updates, then ramp down
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
        # Clip schedule with adaptive scaling
        if update_idx < hold_updates:
            clip_base = clip_max
        else:
            tail_progress = float(update_idx - hold_updates) / float(max(1, tail_updates - 1))
            tail_progress = max(0.0, min(1.0, tail_progress))
            clip_base = clip_max + (clip_min - clip_max) * tail_progress
        clip_val = clip_base * clip_factor
        clip_val = max(clip_floor, min(clip_ceiling, clip_val))
        agent.cfg.clip_eps = clip_val
        clip_base_current = clip_base
        # Learning rate schedule with adaptive scaling
        if update_idx < hold_updates:
            lr_base = lr_hold
        else:
            lr_tail_progress = float(update_idx - hold_updates) / float(max(1, tail_updates - 1))
            lr_tail_progress = max(0.0, min(1.0, lr_tail_progress))
            lr_base = lr_hold + (lr_final - lr_hold) * lr_tail_progress
        lr_val = lr_base * lr_factor
        lr_val = max(min_lr, min(max_lr, lr_val))
        for g in agent.opt.param_groups:
            g["lr"] = lr_val
        lr_base_current = lr_base
        # Determine probability of sampling lagged-opponent paths for this update
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
            # Early phase: with prob=lag_prob, draw opponent action from lagged/historical policy.
            # Late phase: fully on-policy symmetric sampling.
            use_opponent = (lag_prob > 0.0) and (rng.random() < lag_prob)
            if use_opponent:
                a2_norm, e2, logp2, _ = agent.act_opponent(s2)
                v2 = agent.value_only(s2)
            else:
                a2_norm, e2, logp2, v2 = agent.act(s2)

            _, rewards, _, done, _ = env.step((torch.tensor([float(e1.item())]), torch.tensor([float(e2.item())])))

            agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
            agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
            history.append(float((e1.item() + e2.item()) / 2.0))
        last_update_metrics = agent.update()
        kl_val = float(last_update_metrics.get("approx_kl", 0.0) if last_update_metrics else 0.0)
        if not math.isfinite(kl_val):
            kl_val = 0.0
        if kl_val < kl_low:
            clip_factor = min(clip_factor * 1.2, 1.5)
            lr_factor = min(lr_factor * 1.25, 1.5)
        elif kl_val > kl_high:
            clip_factor = max(clip_factor * 0.8, 0.5)
            lr_factor = max(lr_factor * 0.8, 0.2)
        clip_val = max(clip_floor, min(clip_ceiling, clip_base_current * clip_factor))
        agent.cfg.clip_eps = clip_val
        lr_val = max(min_lr, min(max_lr, lr_base_current * lr_factor))
        for g in agent.opt.param_groups:
            g["lr"] = lr_val
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
                hist_size = last_update_metrics.get("opponent_history_size", float(len(agent._opponent_history))) if last_update_metrics else float(len(agent._opponent_history))
                last_sync = last_update_metrics.get("opponent_last_sync", float(agent._last_sync_step)) if last_update_metrics else float(agent._last_sync_step)
                print(
                    f"[Update {upd_i}] q={q_eval}: e*={e2_star_val:.2f}, policy={final_e2_eval:.2f}, gap={gap:.2f}, "
                    f"entropy={agent.cfg.entropy_coef:.3f}, lag_prob={lag_prob:.2f}, adv_mean={adv_mean:.4f}, "
                    f"approx_kl={kl_val:.4f}, alpha_mean={alpha_mean:.2f}, beta_mean={beta_mean:.2f}, "
                    f"opp_mode={agent.opponent_mode}, last_sync={last_sync:.0f}, opp_hist_size={hist_size:.0f}"
                )
        except Exception as _e:
            # Keep training robust to any eval hiccup
            pass
        update_idx += 1
        steps_done += steps_this

        if eval_every > 0 and es_pat > 0 and (update_idx % eval_every == 0):
            abs_errs = []
            for q_eval in eval_qs:
                e2_star_val = clip_stage2(e_star_two_players(q_eval, w_h, w_l, k), effort_bounds)
                state_eval = agent.state_from_params(q=float(q_eval), k=k, w_h=w_h, w_l=w_l)
                e_eval = agent.mean_effort(state_eval)
                abs_errs.append(abs(e_eval - e2_star_val))
            mean_abs_err = float(np.mean(abs_errs)) if abs_errs else float("inf")
            if mean_abs_err < es_abs:
                es_counter += 1
            else:
                es_counter = 0
            print(f"[EarlyStopProbe] updates={update_idx} mean_abs_err={mean_abs_err:.3f} ({es_counter}/{es_pat})")
            if es_counter >= es_pat:
                print("[EarlyStop] satisfied mean_abs_err threshold and patience. Stopping training.")
                early_stop_triggered = True
                break

    # Build rows for each evaluation q
    rows: List[Dict] = []
    for q in eval_qs:
        e2_star_val = clip_stage2(e_star_two_players(q, w_h, w_l, k), effort_bounds)

        # Helper utilities for evaluation
        env_eval = TwoPlayersEnv(
            w_h=w_h,
            w_l=w_l,
            k=k,
            q=q,
            effort_bounds=effort_bounds,
            seed=cfg.get("seed", 42),
        )

        def _compute_effort(policy_net: torch.nn.Module) -> float:
            state = agent.state_from_params(q=float(q), k=k, w_h=w_h, w_l=w_l)
            with torch.no_grad():
                dist, _ = policy_net.dist(state)
                a_mean = dist.mean.squeeze().clamp(0.0, 1.0)
                return float(effort_bounds[0] + a_mean.detach().cpu().item() * (effort_bounds[1] - effort_bounds[0]))

        def _evaluate_pair(policy_net: torch.nn.Module, opponent_net: torch.nn.Module) -> Dict[str, float]:
            effort_self = _compute_effort(policy_net)
            effort_opp = _compute_effort(opponent_net)
            _, rewards, _, _, _ = env_eval.step(
                (
                    torch.tensor([effort_self], dtype=torch.float32),
                    torch.tensor([effort_opp], dtype=torch.float32),
                )
            )
            reward_self = float(rewards[0].item())
            reward_opp = float(rewards[1].item())
            return {
                "effort_self": effort_self,
                "effort_opp": effort_opp,
                "reward_self": reward_self,
                "reward_opp": reward_opp,
            }

        s_agent = agent.state_from_params(q=float(q), k=k, w_h=w_h, w_l=w_l)
        with torch.no_grad():
            dist_agent, _ = agent.net.dist(s_agent)
            a_agent = dist_agent.mean.squeeze().clamp(0.0, 1.0)
            final_e2 = float(effort_bounds[0] + a_agent.detach().cpu().item() * (effort_bounds[1] - effort_bounds[0]))
            alpha_eval = float(dist_agent.concentration1.mean().item())
            beta_eval = float(dist_agent.concentration0.mean().item())
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
        row["abs_err"] = stage2_gap
        row["opp_mode"] = agent.opponent_mode
        row["opp_sync_interval"] = agent.opponent_sync_interval
        row["opp_ema_tau"] = agent.opponent_ema_tau
        row["opp_hist_size"] = len(agent._opponent_history)
        row["last_sync_step"] = agent._last_sync_step
        row["approx_kl"] = last_update_metrics.get("approx_kl", float("nan")) if last_update_metrics else float("nan")
        row["batch_entropy"] = last_update_metrics.get("batch_entropy", float("nan")) if last_update_metrics else float("nan")
        row["alpha_mean"] = alpha_eval
        row["beta_mean"] = beta_eval

        if eval_symmetric:
            sym_eval = _evaluate_pair(agent.net, agent.net)
            row["eval_symmetric_effort"] = sym_eval["effort_self"]
            row["eval_symmetric_reward"] = sym_eval["reward_self"]
            row["eval_symmetric_abs_err"] = abs(sym_eval["effort_self"] - e2_star_val)

        if eval_vs_opponent:
            opp_eval = _evaluate_pair(agent.net, agent.opponent_policy)
            row["eval_vs_opponent_effort"] = opp_eval["effort_self"]
            row["eval_vs_opponent_reward"] = opp_eval["reward_self"]
            row["eval_vs_opponent_opp_effort"] = opp_eval["effort_opp"]
            row["eval_vs_opponent_abs_err"] = abs(opp_eval["effort_self"] - e2_star_val)

        if eval_vs_history:
            history_nets = list(agent._opponent_history)
            if history_nets:
                hist_results = [_evaluate_pair(agent.net, hist_net) for hist_net in history_nets]
                efforts = np.array([res["effort_self"] for res in hist_results], dtype=np.float32)
                rewards = np.array([res["reward_self"] for res in hist_results], dtype=np.float32)
                row["eval_vs_history_effort_mean"] = float(efforts.mean())
                row["eval_vs_history_effort_std"] = float(efforts.std(ddof=0)) if efforts.size > 1 else 0.0
                row["eval_vs_history_reward_mean"] = float(rewards.mean())
                row["eval_vs_history_reward_std"] = float(rewards.std(ddof=0)) if rewards.size > 1 else 0.0
                row["eval_vs_history_abs_err_mean"] = float(np.mean(np.abs(efforts - e2_star_val)))
            else:
                row["eval_vs_history_effort_mean"] = float("nan")
                row["eval_vs_history_effort_std"] = float("nan")
                row["eval_vs_history_reward_mean"] = float("nan")
                row["eval_vs_history_reward_std"] = float("nan")
                row["eval_vs_history_abs_err_mean"] = float("nan")

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
    parser.add_argument(
        "--episodes",
        type=int,
        default=base_config.get("episodes", 1_800_000),
        help="Episodes for PPO (default config value, e.g. 2.4e6 ≈ 585 updates at 4096 steps/update)",
    )
    parser.add_argument("--grad-lr", type=float, default=0.1, help="Learning rate for gradient descent solver.")
    parser.add_argument("--grad-steps", type=int, default=2000, help="Maximum gradient descent iterations.")
    parser.add_argument("--grad-epsilon", type=float, default=0.1, help="Finite-difference epsilon for gradients.")
    parser.add_argument("--grad-tol", type=float, default=1e-4, help="Terminate when |grad| < tol.")
    parser.add_argument("--eval-vs-opponent", action="store_true", help="Evaluate trained policy against lagged opponent policy.")
    parser.add_argument("--eval-vs-history", action="store_true", help="Evaluate policy against each opponent snapshot and report averages.")
    parser.add_argument("--eval-symmetric", dest="eval_symmetric", action="store_true", help="Evaluate policy against itself (default enabled).")
    parser.add_argument("--no-eval-symmetric", dest="eval_symmetric", action="store_false", help="Disable symmetric self-play evaluation.")
    parser.set_defaults(eval_symmetric=True)
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
    cfg["episodes"] = int(args.episodes)

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
        rows = run_ppo(
            cfg,
            episodes=args.episodes,
            train_qs=train_qs,
            eval_qs=eval_qs,
            eval_symmetric=args.eval_symmetric,
            eval_vs_opponent=args.eval_vs_opponent,
            eval_vs_history=args.eval_vs_history,
        )
        for row in rows:
            save_standardized_result(row, csv_path)

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()

 
