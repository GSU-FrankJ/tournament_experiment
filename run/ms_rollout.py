"""Stage-t rollout of the MS-R1 pipeline for the phases of the non-terminal stages (D2).

In the phase of stage ``t < T`` the learner rolls out stage ``t`` only: one learner action from the
live actor, the opponent's action from the lagged copy ``agent.opponent``, and the stage return

    R_t = -k e_t^2 + V~_{t+1}(d_t + e_t - e_t^opp)

with ``V~_{t+1}`` the shock-integrated table of the frozen suffix (``utils.ms_continuation``; linear
lookup). No shock is drawn (the continuation is already integrated over it), so the ``env`` stream is
not touched. Advantage = return - V(s_t), critic target = the return; every row of the batch is a
stage-``t`` row. The terminal-stage phase does not use this module: it calls
``run.v2_rollout.collect_batch_v2`` exactly as v2.0's Phase A does.

The closed-form equilibrium does not enter.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from agents.ppo_curriculum import CurriculumPPO
from envs.curriculum_env import GameSpec
from utils.v2_continuation import ContinuationTable


def _draw(agent: CurriculumPPO, alpha: np.ndarray, beta: np.ndarray, rng: np.random.Generator):
    """Beta draw with the raw (unclipped) value kept; the stored action is clipped to [c, 1 - c]."""
    raw = rng.beta(alpha.astype(float), beta.astype(float))
    c = agent.cfg.action_clamp
    return raw, np.clip(raw, c, 1.0 - c).astype(np.float32)


def collect_stage_batch(spec: GameSpec, agent: CurriculumPPO, t: int, d0: np.ndarray,
                        roles: np.ndarray, rng_learn: np.random.Generator,
                        rng_opp: np.random.Generator, table_next: ContinuationTable
                        ) -> Dict[str, object]:
    """Roll one learner episode per row of ``d0`` at stage ``t`` (see the module docstring).

    Args:
        spec: Game specification.
        agent: Learner (live actor, critic) and the lagged opponent copy.
        t: The stage of the phase, ``1 <= t < spec.T``.
        d0: Player-0 gap per episode at stage ``t`` (zeros at t = 1).
        roles: Learner physical role per episode (0 or 1).
        rng_learn: Learner action stream.
        rng_opp: Opponent action stream.
        table_next: The table of stage ``t`` (``V~_{t+1}`` as a function of ``d_t + e_t - e_t^opp``).

    Returns:
        A dict with the flattened stage-``t`` transitions (``states``, ``actions``, ``logp``,
        ``returns``, ``advantages``), counts, the clamp-hit counters ``d1`` and ``d1_buf``.
    """
    if not 1 <= int(t) < int(spec.T):
        raise ValueError(f"collect_stage_batch needs 1 <= t < T={spec.T}; got {t}")
    n = int(np.asarray(d0).size)
    d_learner = np.where(roles == 0, d0, -d0).astype(float)
    obs_L = spec.encode_obs(t, d_learner)
    obs_O = spec.encode_obs(t, -d_learner)
    a_L, b_L = agent.beta_params(obs_L, None)
    raw_L, act_L = _draw(agent, a_L, b_L, rng_learn)
    a_O, b_O = agent.beta_params(obs_O, agent.opponent)
    raw_O, act_O = _draw(agent, a_O, b_O, rng_opp)
    c = agent.cfg.action_clamp
    d1: Dict[str, int] = {}
    for who, raw in (("L", raw_L), ("O", raw_O)):
        d1[f"d1_{who}_s{t}_n"] = int(raw.size)
        d1[f"d1_{who}_s{t}_lo"] = int((raw < c).sum())
        d1[f"d1_{who}_s{t}_hi"] = int((raw > 1.0 - c).sum())
    e_L = spec.effort_from_action(act_L)
    e_O = spec.effort_from_action(act_O)
    y_next = d_learner + e_L - e_O
    rew = -spec.k * e_L ** 2 + table_next.lookup(y_next)
    logp = agent.log_prob(a_L, b_L, act_L)
    val = agent.value(obs_L).astype(np.float64)
    adv = rew - val
    ret = adv + val
    return {
        "states": obs_L.astype(np.float32), "actions": act_L.astype(np.float32),
        "logp": logp.astype(np.float32), "returns": ret.astype(np.float32),
        "advantages": adv.astype(np.float32), "stage": np.full(n, int(t)),
        "n_episodes": n, "n_transitions": n, "mean_episode_return": float(ret.mean()),
        "mean_effort_by_stage": {int(t): float(e_L.mean())},
        "d1": d1,
        "d1_buf": {"raw": raw_L, "alpha": a_L, "beta": b_L},
    }
