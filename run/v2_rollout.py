"""v2 rollout: ``run.run_final_dp_br.collect_batch`` plus the v2 flags.

With ``reward_mode="sampled"``, ``frozen=None`` and ``continuation_action_mode="stochastic"`` the
arithmetic and the RNG calls are exactly those of ``collect_batch`` (same order, same shapes).

Flags (paired-arm RNG alignment rule A6: every draw of the default path is still made):
  - ``frozen``: an actor used for BOTH players' stage-T actions (learner and opponent). Stage < T
    actions are unchanged (learner: live actor; opponent: the lagged copy ``agent.opponent``).
  - ``continuation_action_mode="mean"`` (frozen only): the stage-T Beta draws are made and
    discarded; both players execute the Beta mean e_min + range * alpha/(alpha+beta) (float64
    from the float32 alpha, beta, as in evaluation). The stored action is that mean clipped to
    [c, 1-c] in float32 and its log-prob is evaluated under the frozen Beta; the discarded draws
    enter nothing.
  - ``reward_mode="expected"``: the terminal shocks are still drawn (and move d, which is not
    used after T); the terminal reward is the conditional expectation given the stage-T state and
    both executed efforts, w_l + DW F_xi(d + e_own - e_opp) - k e_own^2. Non-terminal stage
    rewards are unchanged.
The returned dict additionally carries ``stage`` (stage index of every stored transition).

R1 refinement additions (all with defaults that leave every draw, stored value and return of the
locked pipeline unchanged):
  - ``cont_table`` (a ``utils.v2_continuation.ContinuationTable``; requires ``frozen`` and
    ``continuation_action_mode="mean"`` and gamma = lambda = 1): the sampled continuation reward of
    the stage-1 rows is replaced by the table value of y = e_1 - e_1^opp (both efforts as executed);
    stage-1 return = r_1 + table(y), stage-1 advantage = return - V(s_1). The stage-2 rows, their
    returns and advantages, and every random draw are unchanged.
  - ``batch["d1"]``: clamp-hit counts of the raw ``rng.beta`` draws (before the clip), per stage and
    per player, and for the learner at the final stage split by |d| < 2q; ``batch["d1_buf"]``: the
    raw draws and the (alpha, beta) of the learner rows, aligned with ``states``. Bookkeeping only.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from agents.ppo_curriculum import BetaActor, CurriculumPPO
from envs.curriculum_env import GameSpec, gap_bin_index, stage_reward, step_gap
from utils.theory_multistage import F_xi

REWARD_MODES = ("sampled", "expected")
CONT_MODES = ("stochastic", "mean")


def _draw(agent: CurriculumPPO, alpha: np.ndarray, beta: np.ndarray, rng: np.random.Generator):
    """``agent.sample_actions`` with the raw (unclipped) draw kept: the same numpy calls."""
    raw = rng.beta(alpha.astype(float), beta.astype(float))
    c = agent.cfg.action_clamp
    return raw, np.clip(raw, c, 1.0 - c).astype(np.float32)


def expected_terminal_reward(spec: GameSpec, d: np.ndarray, e_own: np.ndarray,
                             e_opp: np.ndarray) -> np.ndarray:
    """E[r_T | d_T, e_own, e_opp] = w_l + DW F_xi(d + e_own - e_opp) - k e_own^2."""
    y = np.asarray(d, dtype=float) + np.asarray(e_own, dtype=float) - np.asarray(e_opp, dtype=float)
    return spec.w_l + spec.dw * F_xi(y, spec.q) - spec.k * np.asarray(e_own, dtype=float) ** 2


def collect_batch_v2(spec: GameSpec, agent: CurriculumPPO, t0: np.ndarray, d0: np.ndarray,
                     roles: np.ndarray, rng_env: np.random.Generator,
                     rng_learn: np.random.Generator, rng_opp: np.random.Generator,
                     gamma: float, lam: float, bin_width: float,
                     reward_mode: str = "sampled", frozen: Optional[BetaActor] = None,
                     continuation_action_mode: str = "stochastic",
                     cont_table=None) -> Dict[str, object]:
    """Roll complete learner episodes (see module docstring for the flags)."""
    if reward_mode not in REWARD_MODES or continuation_action_mode not in CONT_MODES:
        raise ValueError(f"bad flags {reward_mode!r} {continuation_action_mode!r}")
    if continuation_action_mode == "mean" and frozen is None:
        raise ValueError("continuation_action_mode='mean' requires a frozen stage-T actor")
    if cont_table is not None and (frozen is None or continuation_action_mode != "mean"
                                   or gamma != 1.0 or lam != 1.0):
        raise ValueError("an expected-continuation table requires a frozen stage-T actor, "
                         "continuation_action_mode='mean' and gamma = lambda = 1")
    n = t0.size
    T = spec.T
    d_learner = np.where(roles == 0, d0, -d0).astype(float)
    rewards = np.zeros((n, T + 1))
    values = np.zeros((n, T + 1))
    per_stage: Dict[int, Dict[str, np.ndarray]] = {}
    effort_by_stage: Dict[int, float] = {}
    visitation: Dict[str, np.ndarray] = {}
    c_clamp = agent.cfg.action_clamp
    d1: Dict[str, int] = {}
    for t in range(1, T + 1):
        for who in ("L", "O"):
            for suf in ("n", "lo", "hi"):
                d1[f"d1_{who}_s{t}_{suf}"] = 0
    for reg in ("in", "out"):
        for suf in ("n", "lo", "hi"):
            d1[f"d1_L_s{T}_{reg}_{suf}"] = 0
    for t in range(1, T + 1):
        idx = np.nonzero(t0 <= t)[0]
        if idx.size == 0:
            continue
        dL = d_learner[idx]
        obs_L = spec.encode_obs(t, dL)
        obs_O = spec.encode_obs(t, -dL)
        use_frozen = frozen is not None and t == T
        a_L, b_L = agent.beta_params(obs_L, frozen if use_frozen else None)
        raw_L, act_L = _draw(agent, a_L, b_L, rng_learn)
        a_O, b_O = agent.beta_params(obs_O, frozen if use_frozen else agent.opponent)
        raw_O, act_O = _draw(agent, a_O, b_O, rng_opp)
        lo_L, hi_L = raw_L < c_clamp, raw_L > 1.0 - c_clamp
        for who, lo, hi in (("L", lo_L, hi_L), ("O", raw_O < c_clamp, raw_O > 1.0 - c_clamp)):
            d1[f"d1_{who}_s{t}_n"] = int(idx.size)
            d1[f"d1_{who}_s{t}_lo"] = int(lo.sum())
            d1[f"d1_{who}_s{t}_hi"] = int(hi.sum())
        if t == T:
            inside = np.abs(dL) < 2.0 * spec.q
            for reg, m in (("in", inside), ("out", ~inside)):
                d1[f"d1_L_s{T}_{reg}_n"] = int(m.sum())
                d1[f"d1_L_s{T}_{reg}_lo"] = int((lo_L & m).sum())
                d1[f"d1_L_s{T}_{reg}_hi"] = int((hi_L & m).sum())
        eps0 = rng_env.uniform(-spec.q, spec.q, size=idx.size)
        eps1 = rng_env.uniform(-spec.q, spec.q, size=idx.size)
        r = roles[idx]
        eps_L = np.where(r == 0, eps0, eps1)
        eps_O = np.where(r == 0, eps1, eps0)
        if use_frozen and continuation_action_mode == "mean":
            m_L = a_L.astype(float) / (a_L.astype(float) + b_L.astype(float))
            m_O = a_O.astype(float) / (a_O.astype(float) + b_O.astype(float))
            e_L = spec.e_min + spec.e_range * m_L
            e_O = spec.e_min + spec.e_range * m_O
            c = agent.cfg.action_clamp
            act_L = np.clip(m_L, c, 1.0 - c).astype(np.float32)   # the sampled draw is discarded
        else:
            e_L = spec.effort_from_action(act_L)
            e_O = spec.effort_from_action(act_O)
        d_next = step_gap(spec, dL, e_L, e_O, eps_L, eps_O)
        if t >= T and reward_mode == "expected":
            rew = expected_terminal_reward(spec, dL, e_L, e_O)
        else:
            rew = stage_reward(spec, t, e_L, d_next)
        logp = agent.log_prob(a_L, b_L, act_L)
        val = agent.value(obs_L)
        rewards[idx, t] = rew
        values[idx, t] = val
        per_stage[t] = {"idx": idx, "obs": obs_L, "act": act_L, "logp": logp, "val": val,
                        "raw": raw_L, "alpha": a_L, "beta": b_L, "y": e_L - e_O}
        effort_by_stage[t] = float(e_L.mean())
        if t >= 2:
            for src_name, src_mask in (("root_path", t0[idx] == 1), ("direct_es", t0[idx] == t)):
                if src_mask.any():
                    b, nb = gap_bin_index(spec, t, dL[src_mask], bin_width)
                    key = f"stage{t}_{src_name}"
                    visitation[key] = np.bincount(b, minlength=nb)
        d_learner[idx] = d_next
    adv = np.zeros((n, T + 1))
    ret = np.zeros((n, T + 1))
    last = np.zeros(n)
    next_v = np.zeros(n)
    for t in range(T, 0, -1):
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
        ret[:, t] = adv[:, t] + values[:, t]
        next_v = values[:, t]
    if cont_table is not None and 1 in per_stage:
        i1 = per_stage[1]["idx"]
        adv[i1, 1] = rewards[i1, 1] + cont_table.lookup(per_stage[1]["y"]) - values[i1, 1]
        ret[i1, 1] = adv[i1, 1] + values[i1, 1]
    states = np.concatenate([per_stage[t]["obs"] for t in sorted(per_stage)])
    actions = np.concatenate([per_stage[t]["act"] for t in sorted(per_stage)])
    logps = np.concatenate([per_stage[t]["logp"] for t in sorted(per_stage)])
    returns = np.concatenate([ret[per_stage[t]["idx"], t] for t in sorted(per_stage)])
    advs = np.concatenate([adv[per_stage[t]["idx"], t] for t in sorted(per_stage)])
    stage = np.concatenate([np.full(per_stage[t]["idx"].size, t) for t in sorted(per_stage)])
    ep_return = np.array([ret[i, t0[i]] for i in range(n)])
    return {
        "states": states.astype(np.float32), "actions": actions.astype(np.float32),
        "logp": logps.astype(np.float32), "returns": returns.astype(np.float32),
        "advantages": advs.astype(np.float32), "stage": stage,
        "n_episodes": int(n), "n_transitions": int(states.shape[0]),
        "mean_episode_return": float(ep_return.mean()),
        "mean_effort_by_stage": effort_by_stage,
        "visitation": visitation,
        "d1": d1,
        "d1_buf": {"raw": np.concatenate([per_stage[t]["raw"] for t in sorted(per_stage)]),
                   "alpha": np.concatenate([per_stage[t]["alpha"] for t in sorted(per_stage)]),
                   "beta": np.concatenate([per_stage[t]["beta"] for t in sorted(per_stage)])},
    }
