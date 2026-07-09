# multistage-tel-ppo

Status: in-progress
Current phase: phase03 (complete) -> phase04 (multi-step PPO trainer, not started)

## What's done

- **Phase 01 (2026-07-09): theory audit + config validation.**
  - Independently audited the two-stage closed-form derivation added to
    the plan doc (main `8e9433e`). Verdict: substantially correct; found
    and corrected one math error (stage-1 SOC missing the V2* kink term
    at d=0 — SOC holds iff q > q_SOC, not unconditionally), one factor-2
    bookkeeping conflict (ΔW=2 example vs ΔW=4 table), one leftover
    contradiction (Experiment 1 constant stage-2 effort), one gap
    (zero-effort deviation not covered by the outside-option PC).
    Details + errata: `docs/technical/two_stage_benchmark_audit.md`.
  - `utils/theory_multistage.py`: g1, g2(d), V2*, U_eq, q_SOC/q_B1/q_B2/
    q_PC/q_crit, corrected stage1_curvature, validate_two_stage_params
    (analytic screens + numerical global-deviation scan incl. e=0).
  - `config/multi_stage_two_players.py`: canonical parameters (w_h=6,
    w_l=2, k=1/3500, c=ke^2, q_list=[45,50,55]), gamma=1, lambda=1,
    exploring-starts knobs, verifier settings (closed-form terminal
    integration, delta-gap-sum certificate), validate() raises on
    invalid q. **q_crit = 41.83; q=35 and q=40 are invalid (SOC).**
  - `tools/verify_two_stage_benchmark.py`: all numerical checks PASS
    (validity flips exactly at q_SOC; corrected curvature matches
    numerics; U_eq closed form matches to 4+ digits).

- **Phase 02 (2026-07-09): multi-stage env rewrite.**
  - `envs/multi_stage_env.py`: terminal-reward game, SAMPLED rewards only
    (intermediate = -k e^2, terminal realizes winner from accumulated
    sampled gap), Markov state (t, d), normalized obs [t/T, d/(q sqrt t)],
    T configurable, exploring-starts reset API (reset/sample/reset_exploring),
    both players' transitions via symmetric obs.
  - `tools/verify_multi_stage_env.py`: all checks PASS (payoff=U_eq,
    E[stage-2 effort]=g1, win rate=F_xi(d), transition moments; T=3 +
    exploring-starts smoke). Env cross-validated against theory via
    independent code paths.
  - Carry-forward: `_compute_gae` (agent line 342) will misbootstrap if the
    phase04 runner interleaves p0/p1 transitions — store each player's
    stages contiguously or make GAE trajectory-aware. Fix + unit test before
    the first T=2 GPU run.

- **Phase 03 (2026-07-09): independent DP best-response verifier.**
  - `utils/dp_verifier.py`: backward-induction BR (opponent fixed at ê),
    closed-form terminal integration via F_ξ (no interp of the step
    reward), deterministic triangular quadrature via a 1-D smoothed value,
    parabolic polish on the BR argmax, grid refinement + Richardson.
  - Certificate: PRIMARY = dReach (BR-reachable-support Δ-sum), which
    upper-bounds root EXP and excludes unreachable states (fixes the
    stage-1 off-path spurious term the full-grid worst-case shows).
    dFull and on-path Δ also reported.
  - `tools/calibrate_verifier.py` (all checks PASS): closed form floor
    EXP=dReach=0.0001 & certified; 5 bad policies all EXP≫floor & not
    certified; dReach≥EXP for all; Richardson stable. T=3 smoke OK.
  - Imports only F_ξ/f_ξ from theory; never imports env or agent
    (external verdict).

## Known issues / open items

- Plan Word doc still carries the 5 errata (audit doc section
  "Errata") — owner to fold back into the source.
- mean/mode highlights in plan doc resolved to MEAN by repo invariant;
  doc text not yet updated.
- Commit `8e9433e` on main has a mismatched message ("Update print
  statement...") for a plan-doc edit — noted, history not rewritten.

## What's next

- **Phase 02: env rewrite** (`envs/multi_stage_env.py`): terminal-reward
  game, sampled outcomes only, state (t, d), exploring-starts reset API,
  T configurable. Spec in phase02.md.
- Phase 03: DP verifier + Δ_t(d) certificate (calibrate on closed form:
  EXP(e*_CF) ~ error floor; falsification suite).
- Phase 04: multi-step PPO trainer (critic over (t,d), gamma=1 override,
  GAE path validated); curriculum + ablations per plan.
- Pre-register T=2 gate thresholds before first GPU run.
