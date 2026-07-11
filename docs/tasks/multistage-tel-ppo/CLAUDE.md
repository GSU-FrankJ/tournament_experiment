# Task: multistage-tel-ppo

## Goal

Extend TEL-PPO to multi-stage tournaments (T = 2..5) per
`docs/Experiments Plan_Multi-stage.md`, under the **Claim-B framing**:
PPO generates exploration-smoothed candidate effort functions ê_t(d);
an independent DP best-response verifier certifies approximate-MPE
quality; MC-BR polish bridges residual bias where needed. The narrative
is NOT "ê ≈ e*" recovery.

## Owner decisions (2026-07-09, binding)

1. **Claim-B framing** — the one-stage saga (four concordant negatives,
   see `docs/STATE.md`) proved PPO self-play converges to the smoothed
   equilibrium μ*(κ), not e*. Multi-stage makes this worse (sparse
   terminal reward spreads credit over T stages). Do not re-litigate.
2. **Exploring starts, full MPE claim retained** — episodes reset from
   random (t, d) ~ D_train so off-path states get gradient signal.
   Config knobs: `exploring_starts`, `es_on_path_fraction`,
   `es_d_range_factor`, `es_stage_distribution`.
3. **Δ_t(d) is the primary certificate** — state-wise one-step deviation
   gaps give a true upper bound (EXP <= sum_t max_d Δ_t(d)); root-state
   EXP^UCB is reported alongside, not alone. Grid refinement + Richardson
   extrapolation required; terminal winning probability integrated in
   closed form via F_xi(d) (`utils/theory_multistage.F_xi`), NEVER by
   interpolating the step reward R(d).
4. **Env is a from-scratch rewrite** — `envs/two_stage_env.py` implements
   a different game (per-stage prize flow, logit model, expected-value
   rewards, no gap state). Do not extend it; do not delete it without
   owner confirmation.
5. **Canonical parameters** — `config/multi_stage_two_players.py` is the
   single source of truth (w_h=6, w_l=2, k=1/3500, c(e)=k e^2,
   q_list=[45,50,55]). The plan doc's benchmark section uses c=(k/2)e^2
   and a ΔW=2 example — always convert (k -> 2k). Old
   `config/two_stage_two_players.py` is superseded for this task.
6. **No training before validation** — every runner calls
   `config.multi_stage_two_players.validate()` first; q <= q_crit = 41.83
   is forbidden in the main grid (SOC fails; see audit).

## Key files

- `utils/theory_multistage.py` — closed forms, q_crit, validation scan
- `config/multi_stage_two_players.py` — canonical config + validate()
- `tools/verify_two_stage_benchmark.py` — numerical cross-check (PASS)
- `docs/technical/two_stage_benchmark_audit.md` — derivation audit,
  including the stage-1 SOC correction (kink term) and errata for the
  plan Word doc

## Scope boundaries

- Touch: new env/agent/runner/verifier modules for multi-stage; task docs.
- Do NOT touch: one-stage envs/agents/runners, results/, paper/generator
  theory params, `envs/two_stage_env.py` (superseded, kept for reference).
- Repo invariants apply: sampled training rewards only; Beta mean for
  evaluation; both players' transitions stored; tmux for long runs.
- Denominator invariant re-scoped: "denominator 4" applies to the
  ONE-stage game. In the two-stage benchmark, ΔW/(6kq) (stage 1) and
  ΔW f_xi(d)/(2k) (stage 2) are correct — do not "fix" them to 4qk.

## Phase gate

T=2 full pipeline (train -> extract -> verify -> certify) must pass its
pre-registered gate before ANY GPU minute is spent on T >= 3.
