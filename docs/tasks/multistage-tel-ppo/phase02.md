# Phase 02: multi-stage environment rewrite (NOT STARTED)

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
