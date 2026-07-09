#!/usr/bin/env python3
"""Training runner for the multi-stage tournament (Claim-B candidate generator).

Self-play PPO with exploring starts produces a candidate effort function
e_hat_t(d); the independent DP verifier (phase03) certifies it. This runner:

  - validates the parameter set against the closed-form validity region
    (q_crit) BEFORE any training minute (owner decision 2026-07-09);
  - runs a VECTORIZED self-play rollout (batched policy forward + batched
    env transition) with exploring starts, storing each player's episode as
    its own trajectory (the trajectory-aware GAE contract);
  - evaluates periodically with the DP verifier (root EXP + reachable-set
    Δ certificate) and tracks the best checkpoint by validation dReach;
  - writes a convergence JSON + log under results/multi_stage/.

Long runs must be launched in tmux (repo rule):
    tmux new-session -d -s ms_q50 \
        "python run/run_multi_stage.py --q 50 --T 2 --seed 42 --updates 4000"

Example (quick CPU smoke of the pipeline, NOT a real run):
    python run/run_multi_stage.py --q 50 --T 2 --updates 200 --episodes 128 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.ppo_multi_stage import MultiStagePPO, MultiStagePPOConfig  # noqa: E402
from config.multi_stage_two_players import config as base_config  # noqa: E402
from config.multi_stage_two_players import validate as validate_config  # noqa: E402
from envs.multi_stage_env import MultiStageEnv  # noqa: E402
from utils.dp_verifier import verify  # noqa: E402
from utils.theory_multistage import g1_two_stage, g2_two_stage  # noqa: E402


def collect_rollout(env: MultiStageEnv, agent: MultiStagePPO, n_episodes: int) -> int:
    """Run ``n_episodes`` self-play episodes into the agent buffer (vectorized).

    All episodes advance synchronously; heterogeneous exploring-start stages
    mean some envs finish earlier, so only still-active envs are stepped each
    round. Each env's two players are stored as two separate trajectories.

    Args:
        env: The environment (provides batched transition + start sampler).
        agent: The PPO agent (provides batched policy + buffer).
        n_episodes: Number of parallel episodes to roll.

    Returns:
        Total transitions stored (both players, all stages).
    """
    T = env.T
    t0, d0 = env.sample_exploring_starts_batch(n_episodes)
    stage = t0.copy()
    gap = d0.astype(float)
    active = np.ones(n_episodes, dtype=bool)
    # Per-env, per-player transition lists: (state, a_norm, logp, reward, value)
    traj0: List[List] = [[] for _ in range(n_episodes)]
    traj1: List[List] = [[] for _ in range(n_episodes)]

    while active.any():
        idx = np.nonzero(active)[0]
        s0 = env.obs_batch(stage[idx], gap[idx])          # player 0 observations
        s1 = env.obs_batch(stage[idx], -gap[idx])         # player 1 observations
        a0n, e0, lp0, v0 = agent.act_batch(s0)
        a1n, e1, lp1, v1 = agent.act_batch(s1)
        out = env.step_batch(stage[idx], gap[idx], e0, e1)
        r0, r1 = out["reward0"], out["reward1"]
        for j, i in enumerate(idx):
            traj0[i].append((s0[j], a0n[j], lp0[j], float(r0[j]), v0[j]))
            traj1[i].append((s1[j], a1n[j], lp1[j], float(r1[j]), v1[j]))
        gap[idx] = out["gap_next"]
        stage[idx] += 1
        active[idx] = ~out["terminal"]

    n_stored = 0
    for i in range(n_episodes):
        for traj in (traj0[i], traj1[i]):
            agent.buffer.start_trajectory()
            for (s, a, lp, r, v) in traj:
                agent.buffer.add_np(s, a, lp, r, v)
                n_stored += 1
            agent.buffer.end_trajectory()
    return n_stored


def evaluate(agent: MultiStagePPO, cfg: Dict, q: float, T: int) -> Dict:
    """Run the DP verifier on the current learned policy.

    Args:
        agent: The trained agent.
        cfg: Config dict (prizes, cost, bounds, verifier settings).
        q: Noise half-width.
        T: Horizon.

    Returns:
        Verifier summary (EXP, EXP/DW, dReach, dFull, on-path, certified).
    """
    def policy(t, d):
        return agent.effort_function(t, d, T=T, q=q)

    r = verify(
        policy, w_h=cfg["w_h"], w_l=cfg["w_l"], k=cfg["k"], q=q, T=T,
        e_bar=cfg["effort_range"][1],
        d_grid_size=cfg["verifier"]["d_grid_sizes"][-1],
        e_grid_size=cfg["verifier"]["e_grid_size"],
        epsilon_over_dw=cfg["verifier"]["epsilon_over_dw"],
    )
    return {
        "exp": r.exp,
        "exp_over_dw": r.exp_over_dw,
        "delta_sum_reachable": r.delta_sum_reachable,
        "delta_sum_full": r.delta_sum_full,
        "delta_onpath_sum": r.delta_onpath_sum,
        "certified": bool(r.certified),
    }


def effort_snapshot(agent: MultiStagePPO, q: float, T: int) -> Dict:
    """Record learned vs closed-form effort at diagnostic states (T=2 only)."""
    snap = {"stage1_at_0": float(agent.effort_function(1, np.array([0.0]), T=T, q=q)[0])}
    d_probe = np.array([-2 * q, -q, 0.0, q, 2 * q])
    snap["stage2_probe_d"] = d_probe.tolist()
    snap["stage2_learned"] = agent.effort_function(2, d_probe, T=T, q=q).tolist()
    return snap


def main() -> int:
    """Parse args, validate params, train, evaluate, and write results."""
    p = argparse.ArgumentParser(description="Multi-stage tournament PPO trainer")
    p.add_argument("--q", type=float, default=base_config["q"])
    p.add_argument("--T", type=int, default=base_config["T"])
    p.add_argument("--seed", type=int, default=base_config["seed"])
    p.add_argument("--updates", type=int, default=4000, help="PPO updates")
    p.add_argument("--episodes", type=int, default=256, help="episodes per update")
    p.add_argument("--eval-every", type=int, default=100, help="verifier eval frequency")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--gae-lambda", type=float, default=1.0)
    p.add_argument("--es-on-path-fraction", type=float, default=base_config["es_on_path_fraction"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    cfg = dict(base_config)
    cfg.update({"q": args.q, "T": args.T, "seed": args.seed,
                "exploring_starts": True, "es_on_path_fraction": args.es_on_path_fraction})

    # Owner decision: no training minute on parameters outside the validity region.
    single = dict(cfg)
    single["q_list"] = [args.q]
    validate_config(single, strict=True)  # raises on q <= q_crit

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = MultiStageEnv(cfg, seed=args.seed)
    agent = MultiStagePPO(
        effort_bounds=tuple(cfg["effort_range"]),
        cfg=MultiStagePPOConfig(
            lr=args.lr, entropy_coef=args.entropy_coef, gae_lambda=args.gae_lambda,
            gamma=1.0, seed=args.seed,
        ),
        device=args.device,
    )

    history: List[Dict] = []
    best = {"delta_sum_reachable": float("inf"), "update": -1, "state_dict": None, "eval": None}
    t_start = time.time()

    print(f"[run] q={args.q} T={args.T} seed={args.seed} device={agent.device} "
          f"updates={args.updates} episodes/update={args.episodes}")
    for u in range(1, args.updates + 1):
        collect_rollout(env, agent, args.episodes)
        diag = agent.update()

        if u % args.eval_every == 0 or u == args.updates:
            ev = evaluate(agent, cfg, args.q, args.T)
            rec = {"update": u, **diag, **{f"eval_{k}": v for k, v in ev.items()}}
            history.append(rec)
            if ev["delta_sum_reachable"] < best["delta_sum_reachable"]:
                best = {
                    "delta_sum_reachable": ev["delta_sum_reachable"],
                    "update": u,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in agent.net.state_dict().items()},
                    "eval": ev,
                }
            print(f"[u{u:>5}] EXP={ev['exp']:.4f} EXP/DW={ev['exp_over_dw']:.4f} "
                  f"dReach={ev['delta_sum_reachable']:.4f} cert={ev['certified']} | "
                  f"pl={diag['policy_loss']:.3f} vl={diag['value_loss']:.2f} "
                  f"ent={diag['entropy']:.3f} kl={diag['approx_kl']:.4f}")

    elapsed = time.time() - t_start
    final_eval = evaluate(agent, cfg, args.q, args.T)
    final_snap = effort_snapshot(agent, args.q, args.T)

    # Closed-form targets for T=2 reporting.
    cf = None
    if args.T == 2:
        g1 = g1_two_stage(args.q, cfg["w_h"], cfg["w_l"], cfg["k"])
        cf = {
            "g1": g1,
            "stage2_probe": g2_two_stage(np.array(final_snap["stage2_probe_d"]),
                                         args.q, cfg["w_h"], cfg["w_l"], cfg["k"],
                                         cfg["effort_range"][1]).tolist(),
        }

    out_dir = os.path.join("results", "multi_stage", "convergence")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    fname = f"ms_T{args.T}_q{args.q:g}_seed{args.seed}{tag}_convergence.json"
    result = {
        "params": {"q": args.q, "T": args.T, "seed": args.seed,
                   "w_h": cfg["w_h"], "w_l": cfg["w_l"], "k": cfg["k"],
                   "effort_range": cfg["effort_range"],
                   "gamma": 1.0, "gae_lambda": args.gae_lambda,
                   "es_on_path_fraction": args.es_on_path_fraction,
                   "updates": args.updates, "episodes_per_update": args.episodes},
        "ppo_config": asdict(agent.cfg),
        "elapsed_sec": elapsed,
        "history": history,
        "final_eval": final_eval,
        "final_effort": final_snap,
        "best_checkpoint": {"update": best["update"], "eval": best["eval"]},
        "closed_form": cf,
    }
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[done] {elapsed:.1f}s | final EXP={final_eval['exp']:.4f} "
          f"dReach={final_eval['delta_sum_reachable']:.4f} cert={final_eval['certified']} | "
          f"best dReach={best['delta_sum_reachable']:.4f} @u{best['update']}")
    if cf is not None:
        print(f"[effort] stage1(0)={final_snap['stage1_at_0']:.2f} (g1={cf['g1']:.2f}) | "
              f"stage2 learned={[round(x, 1) for x in final_snap['stage2_learned']]} "
              f"vs CF={[round(x, 1) for x in cf['stage2_probe']]} @ d={final_snap['stage2_probe_d']}")
    print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
