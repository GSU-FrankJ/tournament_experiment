# multistage-tel-ppo

Status: in-progress
Current phase: phase05 (T=3 verified equilibrium) — GATE PASSED (5/5 certify)

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

- **Phase 04 step 1 (2026-07-09): trajectory-aware GAE + rollout buffer.**
  - `agents/ppo_multi_stage.py`: `compute_gae_single`,
    `compute_gae_trajectories`, `MultiStageRolloutBuffer`. Ordering-
    independent (per-trajectory, zero terminal bootstrap) so the
    interleaving misbootstrap cannot recur.
  - `tools/test_multi_stage_gae.py` (all PASS): reproduces the flat-GAE
    interleaving bug and confirms the fix; hand-computed 2-step;
    gamma=lam=1 == Monte Carlo; ordering independence; buffer guard rails.
  - One-stage agent left untouched (scope).
- **Phase 04 step 2 (2026-07-09): actor-critic + PPO update.**
  - `agents/ppo_multi_stage.py`: `MultiStageActorCritic` (Beta mean/conc,
    state_dim=2), `MultiStagePPOConfig` (gamma=lambda=1.0, no one-stage
    default leak), `MultiStagePPO` (act, mean_effort, effort_function
    verifier hook, clipped PPO update via trajectory-aware GAE). No
    theory_align/opponent-lag.
  - `tools/smoke_multi_stage_ppo.py` (PASS, ~17s CPU): self-play rollout
    with exploring starts (each player's episode its own trajectory),
    finite diagnostics + in-bounds effort throughout, non-degenerate
    policy, entropy drop, verifier hook returns finite EXP. Convergence
    NOT asserted (step 3/4 with real budget).
  - Remaining: step 3 production rollout loop + exploring-starts tuning;
    step 4 extraction + pre-registered T=2 verifier gate.
- **Phase 04 step 3 (2026-07-09): production rollout loop.**
  - `run/run_multi_stage.py`: validates params (q_crit) before training,
    vectorized self-play rollout with exploring starts (each player's
    episode its own trajectory), periodic DP-verifier eval, best-checkpoint
    by dReach, convergence JSON.
  - Vectorized env: `env.step_batch`/`obs_batch`/`sample_exploring_starts_batch`;
    agent `act_batch` + `buffer.add_np`. step_batch == scalar step under CRN
    to 2e-7 (verify check 5).
  - **Validation (T=2, q=50, seed42, 1000upd x 256ep, CPU): CERTIFIES from
    u100 on.** Final EXP/DW=0.46%, dReach 0.033-0.086 (<3% gate). stage-1
    44.7 (g1=46.67). stage-2 hump-shaped peak 59 (CF 70). Findings: peak
    undershoot = exploration-smoothed mu*(kappa) (Claim-B vindicated);
    stage-2 asymmetry (finite-sample, step-4 seed item). JSON not committed.
- **Phase 04 step 4 (2026-07-09): pre-registered T=2 gate -> PASSED.**
  - `utils/multi_stage_metrics.py` (recovery metrics + gate),
    `tools/evaluate_gate.py`, frozen `preregistration_T2.md`. Thresholds
    committed (e73775c) BEFORE the gated run. Runner restores the
    best-dReach checkpoint and saves recovery metrics + EXP^UCB.
  - **Gated run T=2, q=50, seeds 42-46, 2000upd x 512ep, GPU (~30min/seed):
    GATE PASS, 5/5 certify.** dReach/DW mean 0.0063 (max 0.0078 << 0.03).
    EXP/DW mean 0.0016. Recovery also clears targets: RE_1 0.045 (<0.10),
    RPE_2_core 0.052 (<0.15). Recovered stage-2 near-symmetric (seed42
    e2=[7.4,34.6,64.0,31.5,10.1] vs CF [0,35,70,35,0]); peak 64 vs 70 =
    residual smoothing. Step-3 asymmetry did not survive.
    Results: `results/multi_stage/convergence/ms_T2_q50_seed{42..46}_gateT2_convergence.json`.
  - **Per pre-registration, PASS authorizes T=3 GPU spend.**
  - Perf note: tiny 64-hidden net is CPU-bound; `--device cuda` no faster
    than CPU (act_batch torch<->numpy round trips). For T>=3, run CPU or
    optimize the rollout.

- **Phase 05 (2026-07-09): T=3 verified equilibrium — GATE PASSED.**
  - Runner generalized to T>=3 (per-stage effort_curves: learned/BR/Δ/on-path
    for Figs 3-5); verifier exposes br_effort_by_stage. Frozen
    `preregistration_T3.md` (same certificate threshold; no closed form).
  - **Gated run T=3, q=50, seeds 42-46, 3000upd x 512ep, CPU (~25min/seed):
    GATE PASS, 5/5 certify.** dReach/DW mean 0.0097 (max 0.0144), EXP/DW
    mean 0.0027. A certified eps-approx MPE with NO closed-form benchmark
    (the plan's main contribution).
  - Economic patterns (cross-seed): effort increases toward the final stage
    (e_hat_t(0)=43.3/50.9/64.8), hump-shaped in the gap at every stage,
    leader/follower asymmetry at intermediate stages (behind exerts less;
    legitimate for T>=3). stage-3 peak ~65 vs myopic ~70 = residual
    smoothing.
  - Results: `results/multi_stage/convergence/ms_T3_q50_seed{42..46}_gateT3_convergence.json`.

## Known issues / open items

- Plan Word doc still carries the 5 errata (audit doc section
  "Errata") — owner to fold back into the source.
- mean/mode highlights in plan doc resolved to MEAN by repo invariant;
  doc text not yet updated.
- Commit `8e9433e` on main has a mismatched message ("Update print
  statement...") for a plan-doc edit — noted, history not rewritten.

## What's next

Phases 01-04 complete; T=2 gate PASSED (T=3 GPU spend authorized).

- **Phase 05: T=3 verified equilibrium** (plan 5.3, the main contribution).
  No closed form; PPO computes e_hat_1/2/3(d), the DP verifier certifies.
  Pre-register a T=3 gate (dReach/DW threshold; seed robustness). Report
  learned effort functions, BR-vs-learned, Δ_t(d), exploitability
  certificate. Curriculum (T=1->2->3) + no-curriculum ablation.
- Robustness (plan 5.4): grid refinement (already in verifier), seed
  robustness, falsification, optional adversarial-RL BR cross-check.
- Multi-stage extension T=4,5 (plan 5.5) as benchmark extensions.
- Optional: optimize the CPU-bound rollout before larger-T budgets.
