"""Local copy of W's ``run.run_final_dp_br.collect_batch`` with full coverage counts.

The training math and the RNG draw order are the same as in W (learner Beta
action, opponent Beta action, then two environment shocks per stage; GAE with
zero terminal bootstrap). The added statistics only read ``t0`` and the
pre-action learner-signed gaps; they never draw random numbers. A unit test
checks that states/actions/log-probs/returns/advantages and all three RNG
states are identical to W's collector.

Added return fields:
    visitation_by_start_stage  {"s{s}_t{t}": counts on D_t bins} for all s <= t
    start_counts_actual        {"1","2","3": episodes starting there}
    start_bins_raw             {"s{s}": player-0 start gap bins, before the role flip}
    start_bins_learner         {"s{s}": learner-signed start gap bins}
    stage_transition_counts    {"1","2","3": learner transitions at that stage}
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402,F401

from agents.ppo_curriculum import CurriculumPPO  # noqa: E402
from envs.curriculum_env import GameSpec, gap_bin_index, stage_reward, step_gap  # noqa: E402
from metrics import n_stage_bins, strict_gap_bins  # noqa: E402


def collect_batch(spec: GameSpec, agent: CurriculumPPO, t0: np.ndarray, d0: np.ndarray,
                  roles: np.ndarray, rng_env: np.random.Generator,
                  rng_learn: np.random.Generator, rng_opp: np.random.Generator,
                  gamma: float, lam: float, bin_width: float, trace: bool = False,
                  coverage_tol: float = 1e-9) -> Dict[str, Any]:
    """Roll complete learner episodes against the frozen snapshot (vectorized).

    Args:
        spec: Game parameters.
        agent: Learner (actor/critic) + frozen opponent.
        t0: Start stage per episode.
        d0: Player-0 gap per episode at the start stage.
        roles: Learner physical role per episode (0 or 1).
        rng_env: Environment noise stream.
        rng_learn: Learner action stream.
        rng_opp: Opponent action stream.
        gamma: Discount.
        lam: GAE lambda.
        bin_width: Visitation bin width on D_t.
        trace: Also return per-stage internals (tests only; no extra RNG draws).
        coverage_tol: Relative domain tolerance for coverage binning.

    Returns:
        Dict with the W fields plus the coverage fields listed in the module doc.
    """
    n = t0.size
    T = spec.T
    d_learner = np.where(roles == 0, d0, -d0).astype(float)
    rewards = np.zeros((n, T + 1))
    values = np.zeros((n, T + 1))
    per_stage: Dict[int, Dict[str, np.ndarray]] = {}
    effort_by_stage: Dict[int, float] = {}
    visitation: Dict[str, np.ndarray] = {}
    # ---- coverage (added; no RNG) -------------------------------------------------
    vis_by_start: Dict[str, np.ndarray] = {}
    start_counts = {str(s): int((t0 == s).sum()) for s in range(1, T + 1)}
    start_raw: Dict[str, np.ndarray] = {}
    start_learner: Dict[str, np.ndarray] = {}
    for s in range(1, T + 1):
        m = t0 == s
        nb = n_stage_bins(spec, s, bin_width)
        start_raw[f"s{s}"] = np.bincount(
            strict_gap_bins(spec, s, d0[m], bin_width, coverage_tol, f"start_raw_s{s}"), minlength=nb)
        start_learner[f"s{s}"] = np.bincount(
            strict_gap_bins(spec, s, d_learner[m], bin_width, coverage_tol, f"start_learner_s{s}"),
            minlength=nb)
    stage_counts: Dict[str, int] = {}
    tr: Dict[str, Any] = {"d_pre": np.full((n, T + 1), np.nan), "stages": {}}
    for t in range(1, T + 1):
        idx = np.nonzero(t0 <= t)[0]
        stage_counts[str(t)] = int(idx.size)
        if idx.size == 0:
            continue
        dL = d_learner[idx]
        # coverage of the pre-action learner-signed gap, by start stage (no RNG)
        nb = n_stage_bins(spec, t, bin_width)
        src_stage = t0[idx]
        for s in range(1, t + 1):
            m = src_stage == s
            b = strict_gap_bins(spec, t, dL[m], bin_width, coverage_tol, f"start{s}_stage{t}")
            vis_by_start[f"s{s}_t{t}"] = np.bincount(b, minlength=nb)
        if trace:
            tr["d_pre"][idx, t] = dL
        obs_L = spec.encode_obs(t, dL)
        obs_O = spec.encode_obs(t, -dL)
        a_L, b_L = agent.beta_params(obs_L)
        act_L = agent.sample_actions(a_L, b_L, rng_learn)
        a_O, b_O = agent.beta_params(obs_O, agent.opponent)
        act_O = agent.sample_actions(a_O, b_O, rng_opp)
        eps0 = rng_env.uniform(-spec.q, spec.q, size=idx.size)
        eps1 = rng_env.uniform(-spec.q, spec.q, size=idx.size)
        r = roles[idx]
        eps_L = np.where(r == 0, eps0, eps1)
        eps_O = np.where(r == 0, eps1, eps0)
        e_L = spec.effort_from_action(act_L)
        e_O = spec.effort_from_action(act_O)
        d_next = step_gap(spec, dL, e_L, e_O, eps_L, eps_O)
        rew = stage_reward(spec, t, e_L, d_next)
        logp = agent.log_prob(a_L, b_L, act_L)
        val = agent.value(obs_L)
        rewards[idx, t] = rew
        values[idx, t] = val
        per_stage[t] = {"idx": idx, "obs": obs_L, "act": act_L, "logp": logp, "val": val}
        effort_by_stage[t] = float(e_L.mean())
        if t >= 2:
            for src_name, src_mask in (("root_path", t0[idx] == 1), ("direct_es", t0[idx] == t)):
                if src_mask.any():
                    b, nbw = gap_bin_index(spec, t, dL[src_mask], bin_width)
                    key = f"stage{t}_{src_name}"
                    visitation[key] = np.bincount(b, minlength=nbw)
        if trace:
            tr["stages"][t] = {"idx": idx, "obs_L": obs_L, "obs_O": obs_O, "e_L": e_L, "e_O": e_O,
                               "eps0": eps0, "eps1": eps1, "eps_L": eps_L, "eps_O": eps_O,
                               "d_next": d_next, "reward": rew}
        d_learner[idx] = d_next
    # GAE with zero terminal bootstrap (V_{T+1} = 0)
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
    states = np.concatenate([per_stage[t]["obs"] for t in sorted(per_stage)])
    actions = np.concatenate([per_stage[t]["act"] for t in sorted(per_stage)])
    logps = np.concatenate([per_stage[t]["logp"] for t in sorted(per_stage)])
    returns = np.concatenate([ret[per_stage[t]["idx"], t] for t in sorted(per_stage)])
    advs = np.concatenate([adv[per_stage[t]["idx"], t] for t in sorted(per_stage)])
    ep_return = np.array([ret[i, t0[i]] for i in range(n)])
    out = {
        "states": states.astype(np.float32), "actions": actions.astype(np.float32),
        "logp": logps.astype(np.float32), "returns": returns.astype(np.float32),
        "advantages": advs.astype(np.float32),
        "n_episodes": int(n), "n_transitions": int(states.shape[0]),
        "mean_episode_return": float(ep_return.mean()),
        "mean_effort_by_stage": effort_by_stage,
        "visitation": visitation,
        "visitation_by_start_stage": vis_by_start,
        "start_counts_actual": start_counts,
        "start_bins_raw": start_raw,
        "start_bins_learner": start_learner,
        "stage_transition_counts": stage_counts,
    }
    if trace:
        tr.update(rewards=rewards, values=values, adv=adv, ret=ret, d_final=d_learner.copy())
        out["trace"] = tr
    return out
