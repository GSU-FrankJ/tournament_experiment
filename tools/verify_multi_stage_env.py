"""Independent self-check of envs/multi_stage_env.py against the closed form.

Drives the env end-to-end (sampled outcomes only) under the closed-form
two-stage policy and checks that the empirical statistics reproduce the
analytic benchmark from utils/theory_multistage. The env and the theory
module are independent code paths (the env never imports theory), so an
agreement here is genuine cross-validation, not a tautology.

Checks (T=2, canonical params, q=50):
  1. Mean total payoff  ~ U_eq = 2.678.
  2. E[stage-2 effort under g2] ~ g1 = 46.67 (on-path expectation identity).
  3. Terminal win rate from gap d ~ F_xi(d) when both play the even benchmark.
  4. Transition sanity: E[gap increment | equal efforts] ~ 0, Var ~ 2 q^2/3.

Run:
    python tools/verify_multi_stage_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.multi_stage_two_players import config  # noqa: E402
from envs.multi_stage_env import MultiStageEnv  # noqa: E402
from utils.theory_multistage import (  # noqa: E402
    F_xi,
    eq_utility_two_stage,
    g1_two_stage,
    g2_two_stage,
)

Q = 50.0
N = 400_000


def make_env(q: float = Q, seed: int = 0) -> MultiStageEnv:
    """Build a canonical T=2 env at the given noise level."""
    cfg = dict(config)
    cfg.update({"q": q, "T": 2, "exploring_starts": False})
    return MultiStageEnv(cfg, seed=seed)


def closed_form_policy(env: MultiStageEnv):
    """Shared symmetric closed-form policy (t, d) -> effort for the T=2 game."""
    g1 = g1_two_stage(env.q, env.w_h, env.w_l, env.k)

    def policy(t: int, d: float) -> float:
        if t >= env.T:
            return float(g2_two_stage(np.asarray(d), env.q, env.w_h, env.w_l, env.k, env.effort_high))
        return g1

    return policy


def main() -> int:
    """Run all env self-checks; return 0 on success, 1 on any failure."""
    env = make_env()
    g1 = g1_two_stage(env.q, env.w_h, env.w_l, env.k)
    u_eq = eq_utility_two_stage(env.q, env.w_h, env.w_l, env.k)
    failures = 0

    # --- Checks 1 & 2: payoff and on-path stage-2 effort identity ---
    out = env.rollout_policy(closed_form_policy(env), N, start=(1, 0.0))
    mean_payoff = float(out["payoff0"].mean())
    se_payoff = float(out["payoff0"].std() / np.sqrt(N))
    mean_e2 = float(out["effort0"][:, 1].mean())  # stage-2 effort under on-path d2

    ok1 = abs(mean_payoff - u_eq) < 4.0 * se_payoff + 5e-3
    ok2 = abs(mean_e2 - g1) < 0.2
    failures += (not ok1) + (not ok2)
    print(f"[{'OK ' if ok1 else 'BAD'}] mean payoff = {mean_payoff:.4f} +/- {se_payoff:.4f} "
          f"(U_eq = {u_eq:.4f})")
    print(f"[{'OK ' if ok2 else 'BAD'}] E[stage-2 effort] = {mean_e2:.3f}  (g1 = {g1:.3f})")

    # --- Check 3: terminal win rate from gap d equals F_xi(d) ---
    print("     terminal win rate vs F_xi(d):")
    for d in (-60.0, -30.0, 0.0, 30.0, 60.0):
        env2 = make_env(seed=123)
        g2d = float(g2_two_stage(np.asarray(d), env2.q, env2.w_h, env2.w_l, env2.k, env2.effort_high))
        # both play the even benchmark at the final stage => efforts cancel
        wins = 0
        M = 200_000
        env2.reset(2, d)
        for _ in range(M):
            if env2._done:
                env2.reset(2, d)
            res = env2.step((g2d, g2d))
            wins += 1 if res.info["winner"] == 0 else 0
        emp = wins / M
        theo = float(F_xi(np.asarray(d), env2.q)[0])
        ok = abs(emp - theo) < 0.005
        failures += not ok
        print(f"     [{'OK ' if ok else 'BAD'}] d={d:+6.1f}: emp={emp:.4f} F_xi={theo:.4f}")

    # --- Check 4: transition moments (equal efforts) ---
    env3 = make_env(seed=7)
    incs = []
    env3.reset(1, 0.0)
    for _ in range(N):
        if env3._done:
            env3.reset(1, 0.0)
        d_in = env3.gap
        res = env3.step((30.0, 30.0))  # equal efforts => increment is pure xi
        incs.append(res.info["gap_out"] - d_in)
    incs = np.asarray(incs)
    mean_inc, var_inc = float(incs.mean()), float(incs.var())
    var_theo = 2.0 * Q * Q / 3.0  # Var(eps_i - eps_j) = 2 Var(U(-q,q)) = 2 q^2/3
    ok4 = abs(mean_inc) < 0.2 and abs(var_inc - var_theo) < 0.02 * var_theo
    failures += not ok4
    print(f"[{'OK ' if ok4 else 'BAD'}] gap increment (equal effort): "
          f"mean={mean_inc:+.3f} var={var_inc:.1f} (theory {var_theo:.1f})")

    # --- Check 5: step_batch matches scalar step under common random numbers ---
    env4 = make_env(seed=99)
    rng = np.random.default_rng(5)
    max_abs = 0.0
    for _ in range(2000):
        t = int(rng.integers(1, 3))
        d = float(rng.uniform(-120, 120))
        a0 = float(rng.uniform(0, 100))
        a1 = float(rng.uniform(0, 100))
        eps0 = float(rng.uniform(-Q, Q))
        eps1 = float(rng.uniform(-Q, Q))
        env4.reset(t, d)
        res = env4.step((a0, a1), noise=(eps0, eps1))
        b = env4.step_batch(np.array([t]), np.array([d]), np.array([a0]),
                            np.array([a1]), noise=(np.array([eps0]), np.array([eps1])))
        # gap and per-player rewards must match exactly (prizes deterministic given gap sign)
        max_abs = max(max_abs,
                      abs(res.info["gap_out"] - b["gap_next"][0]),
                      abs(res.rewards[0].item() - b["reward0"][0]),
                      abs(res.rewards[1].item() - b["reward1"][0]))
    ok5 = max_abs < 1e-5
    failures += not ok5
    print(f"[{'OK ' if ok5 else 'BAD'}] step_batch == scalar step (CRN): max abs diff {max_abs:.2e}")

    print("PASS" if failures == 0 else f"FAIL ({failures} checks)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
