# Phase 02: multi-stage environment rewrite (COMPLETE 2026-07-09)

## Status

Done. `envs/multi_stage_env.py` + `tools/verify_multi_stage_env.py`
(all self-checks PASS). Env drives end-to-end on sampled outcomes only and
independently reproduces the closed form (env never imports theory, so
agreement is genuine cross-validation, not a tautology — addresses
shared-assumption risk M7).

## Self-check results (T=2, q=50, canonical params)

| Check | Result |
|---|---|
| mean total payoff vs U_eq=2.678 | 2.6816 +/- 0.0032 OK |
| E[stage-2 effort] vs g1=46.67 | 46.693 OK |
| terminal win rate vs F_xi(d), d in {-60..60} | max abs err 0.002 OK |
| gap increment (equal effort): mean~0, var~2q^2/3 | +0.013, 1660 vs 1667 OK |
| T=3 horizon, reward structure, reset(t0>1) remaining stages | smoke OK |
| exploring-starts sampler (on-path frac, d0=0 at t0=1) | smoke OK |

## Carry-forward for phase04 (trainer)

**GAE ordering trap:** `agents/ppo_two_players_clean._compute_gae`
(line 342) chains `next_value` across consecutive storage indices. This is
correct for single-step bandit (all done=True) but will MISBOOTSTRAP if the
multi-stage runner interleaves p0/p1 transitions (p0 stage-1's next_value
would read p1 stage-1's value). The runner must store each player's stage
sequence CONTIGUOUSLY, or GAE must be made trajectory-aware. Must be fixed
and unit-tested before the first T=2 GPU run.


## Objective

`envs/multi_stage_env.py` implementing the plan's game exactly. The
existing `envs/two_stage_env.py` is a DIFFERENT game (per-stage prize
flow, logit win model, expected-value rewards, no gap state) — reference
only, never extend.

## Spec (from plan sections 1, 3.4 + owner decisions)

- T configurable (2..5), N=2, quadratic cost k e^2 (repo convention).
- Stage: y_it = e_it + eps_it, eps ~ U(-q, q) i.i.d.; state (t, d_t) with
  d_{t+1} = d_t + e_i - e_j + (eps_i - eps_j). Public interim feedback.
- Rewards SAMPLED only (repo invariant): r_t = -k e_t^2 for t < T;
  r_T = R(realized d_{T+1}) - k e_T^2 with R = w_h / w_l by realized
  final gap (tie prob-0). No closed-form probability may enter step().
- Normalized observation [t/T, d/(q sqrt(t))] alongside raw (t, d).
- **Exploring-starts reset API**: reset(t0, d0) + sampler for
  (t0, d0) ~ D_train per config knobs (`es_on_path_fraction`,
  `es_d_range_factor`, `es_stage_distribution`). Critical detail:
  episodes started at t0 > 1 must return per-stage costs consistent with
  the truncated horizon (value targets are continuation values).
- Both players' transitions exposed for rollout storage (self-play).

## Acceptance checks

- MC episode simulation at the closed-form policy reproduces U_eq
  (2.678 at q=50) and E[g2] = g1 within MC error.
- Empirical win rate from gap d matches F_xi(d) when both play the
  (even) benchmark.
- Env validated against `utils/theory_multistage.py`, not vice versa
  (independent code paths; shared-assumption risk M7 from the blind-spot
  review).

## Boundaries

- New file only; do not modify one-stage envs.
- `config.multi_stage_two_players.validate()` called at env construction.
