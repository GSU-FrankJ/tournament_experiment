"""End-to-end smoke test for the multi-stage PPO trainer (step 2).

Runs a short CPU self-play training loop on the T=2 game with exploring
starts and checks that the actor-critic + PPO update are numerically sane
and wired to the env and the verifier:

  1. update() diagnostics are finite across all updates.
  2. Learned effort stays within [low, high]; the policy is not degenerate
     (not pinned to a bound) and entropy drops from its random init.
  3. effort_function feeds utils.dp_verifier.verify without error and yields
     a finite exploitability.

This is a SMOKE test (correctness of plumbing + sanity), not the production
training loop (step 3) and not the acceptance gate (step 4): the budget is
tiny, so convergence to the closed form is NOT asserted.

Run:
    python tools/smoke_multi_stage_ppo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.ppo_multi_stage import MultiStagePPO, MultiStagePPOConfig  # noqa: E402
from config.multi_stage_two_players import config as base_cfg  # noqa: E402
from envs.multi_stage_env import MultiStageEnv  # noqa: E402
from utils.dp_verifier import verify  # noqa: E402


def collect_episode(env: MultiStageEnv, agent: MultiStagePPO) -> None:
    """Roll one self-play episode from an exploring start into the buffer.

    Each player's stage sequence is stored as its OWN trajectory (never
    interleaved), which is what the trajectory-aware GAE relies on.
    """
    obs0, obs1 = env.reset_exploring()
    p0: list = []
    p1: list = []
    done = False
    while not done:
        a0n, e0, lp0, v0 = agent.act(obs0)
        a1n, e1, lp1, v1 = agent.act(obs1)
        res = env.step((e0, e1))
        r0, r1 = float(res.rewards[0].item()), float(res.rewards[1].item())
        p0.append((obs0, a0n, lp0, r0, v0))
        p1.append((obs1, a1n, lp1, r1, v1))
        obs0, obs1 = res.obs
        done = res.done
    for traj in (p0, p1):
        agent.buffer.start_trajectory()
        for (s, a, lp, r, v) in traj:
            agent.buffer.add(s, a, lp, r, v)
        agent.buffer.end_trajectory()


def main() -> int:
    """Run the smoke training loop and checks; return 0 on success."""
    torch.manual_seed(0)
    np.random.seed(0)
    q = 50.0
    cfg = dict(base_cfg)
    cfg.update({"q": q, "T": 2, "exploring_starts": True, "es_on_path_fraction": 0.4})
    env = MultiStageEnv(cfg, seed=0)

    agent = MultiStagePPO(
        effort_bounds=tuple(cfg["effort_range"]),
        cfg=MultiStagePPOConfig(entropy_coef=0.005, epochs=8, minibatch_size=256, seed=0),
        device="cpu",
    )

    d_probe = np.array([0.0])
    e2_init = float(agent.effort_function(2, d_probe, T=2, q=q)[0])
    ent_first = None
    failures = 0

    n_updates = 150
    episodes_per_update = 32
    for u in range(n_updates):
        for _ in range(episodes_per_update):
            collect_episode(env, agent)
        diag = agent.update()
        if ent_first is None:
            ent_first = diag["entropy"]
        finite = all(np.isfinite(v) for k, v in diag.items() if k != "transitions")
        if not finite:
            failures += 1
            print(f"  update {u}: non-finite diagnostics {diag}")
            break
        eff = agent.effort_function(2, np.linspace(-2 * q, 2 * q, 41), T=2, q=q)
        if not (np.all(eff >= 0.0) and np.all(eff <= 100.0)):
            failures += 1
            print(f"  update {u}: effort out of bounds [{eff.min():.2f}, {eff.max():.2f}]")
            break

    diag = agent.update() if len(agent.buffer) else diag
    e1_final = float(agent.effort_function(1, np.array([0.0]), T=2, q=q)[0])
    e2_final = float(agent.effort_function(2, d_probe, T=2, q=q)[0])
    ent_last = diag["entropy"]

    print(f"stage-2 effort at d=0: init={e2_init:.2f} -> final={e2_final:.2f}")
    print(f"stage-1 effort at d=0: final={e1_final:.2f}  (closed form g1=46.67, g2(0)=70.0)")
    print(f"entropy: first={ent_first:.3f} -> last={ent_last:.3f}")
    print(f"final update diag: policy_loss={diag['policy_loss']:.4f} "
          f"value_loss={diag['value_loss']:.4f} approx_kl={diag['approx_kl']:.4f} "
          f"clip_frac={diag['clip_frac']:.3f} grad_norm={diag['grad_norm']:.3f}")

    # Check A: diagnostics finite throughout (already guarded in the loop)
    print(f"[{'OK ' if failures == 0 else 'BAD'}] finite diagnostics + in-bounds effort throughout")

    # Check B: policy not degenerate (not pinned to a bound) and entropy dropped
    non_degenerate = 1.0 < e2_final < 99.0 and 1.0 < e1_final < 99.0
    entropy_dropped = ent_last < ent_first
    failures += check("policy non-degenerate (not pinned to a bound)", non_degenerate,
                      f"e1={e1_final:.2f} e2={e2_final:.2f}")
    failures += check("entropy decreased from random init", entropy_dropped,
                      f"{ent_first:.3f} -> {ent_last:.3f}")

    # Check C: verifier hook works and yields a finite EXP
    def learned_policy(t, d):
        return agent.effort_function(t, d, T=2, q=q)

    r = verify(learned_policy, w_h=cfg["w_h"], w_l=cfg["w_l"], k=cfg["k"], q=q, T=2,
               e_bar=cfg["effort_range"][1], epsilon_over_dw=cfg["verifier"]["epsilon_over_dw"])
    exp_finite = np.isfinite(r.exp) and np.isfinite(r.delta_sum_reachable)
    failures += check("verifier hook returns finite EXP", exp_finite,
                      f"EXP={r.exp:.4f} dReach={r.delta_sum_reachable:.4f}")
    print(f"     verifier on learned policy: EXP={r.exp:.4f} EXP/DW={r.exp_over_dw:.4f} "
          f"dReach={r.delta_sum_reachable:.4f} certified={r.certified}")

    print("\nPASS" if failures == 0 else f"\nFAIL ({failures} checks)")
    return 0 if failures == 0 else 1


def check(name: str, cond: bool, detail: str = "") -> int:
    """Print a pass/fail line; return 1 on failure."""
    print(f"[{'OK ' if cond else 'BAD'}] {name}" + (f"  {detail}" if detail and not cond else ""))
    return 0 if cond else 1


if __name__ == "__main__":
    sys.exit(main())
