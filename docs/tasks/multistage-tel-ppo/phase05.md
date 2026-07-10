# Phase 05: T=3 verified equilibrium (IN PROGRESS)

The plan's main contribution (section 5.3). No closed-form benchmark at
T=3, so PPO computes candidate effort functions e_hat_1/2/3(d) and the
independent DP verifier is the SOLE certifier. Authorized by the T=2 gate
PASS.

## Setup (COMPLETE 2026-07-09)

- `utils/dp_verifier.py`: expose `br_effort_by_stage` in the result (the
  grid best-response effort per stage) — needed for the BR-vs-learned
  figure.
- `run/run_multi_stage.py`: generalized to T>=3. Saves per-stage
  `effort_curves` (learned e_hat_t(d), BR_t(d), Δ_t(d), on-path dist over
  |d| <= 4q) for the plan's Figures 3-5; T=3 console summary (per-stage
  e_hat_t(0), worst Δ_t) with the closed-form path skipped. T=3 pipeline
  smoke OK (3-stage curves saved, recovery/closed_form None).
- `docs/.../preregistration_T3.md`: frozen thresholds + protocol. Same
  certificate threshold as T=2 (same verifier); no recovery metrics.

## Gated run (COMPLETE 2026-07-09) — GATE PASSED

T=3, q=50, seeds 42-46, 3000 upd x 512 ep, entropy 0.005, CPU (~25 min/seed).

**GATE PASS — 5/5 seeds certify.** dReach/DW mean 0.0097 (max 0.0144, std
0.0026), well under the 0.03 threshold; EXP/DW mean 0.0027. Reproducible.

| seed | EXP/DW | EXP^UCB/DW | dReach/DW | cert | ckpt |
|---|---|---|---|---|---|
| 42 | 0.0030 | 0.0031 | 0.0097 | yes | u1050 |
| 43 | 0.0035 | 0.0036 | 0.0144 | yes | u2100 |
| 44 | 0.0023 | 0.0024 | 0.0084 | yes | u900 |
| 45 | 0.0020 | 0.0021 | 0.0065 | yes | u1800 |
| 46 | 0.0026 | 0.0027 | 0.0096 | yes | u2550 |

This is a numerically-certified epsilon-approximate MPE for a 3-stage
tournament with NO closed-form benchmark — the plan's main contribution.

Results: `results/multi_stage/convergence/ms_T3_q50_seed{42..46}_gateT3_convergence.json`
(each includes per-stage effort_curves: learned / BR / Δ_t / on-path, for
plan Figures 3-5).

### Economic patterns (cross-seed mean, e_hat_t(d))

| stage | d=-100 | d=-50 | d=0 | d=+50 | d=+100 |
|---|---|---|---|---|---|
| 1 | 4.3 | 6.0 | 43.3 | 35.0 | 14.9 |
| 2 | 4.8 | 13.5 | 50.9 | 35.0 | 10.9 |
| 3 | 7.1 | 29.8 | 64.8 | 32.6 | 7.4 |

- **Effort increases toward the final stage** (d=0: 43.3 -> 50.9 -> 64.8).
  Plan Main Q1: YES.
- **Hump-shaped in the gap at every stage** (peak at d=0, low far
  ahead/behind). Plan Main Q3: YES. Stage-3 hump is sharpest (later stages
  more gap-sensitive).
- **Leader/follower asymmetry** at intermediate stages (a player BEHIND
  exerts less than one equally AHEAD, e.g. stage-2 d=-50 -> 13.5 vs
  d=+50 -> 35.0). Legitimate for T=3 (only the final stage is even by the
  myopic argument); reads as discouragement-when-behind / lead-defense.
  Note: stage-1 d != 0 is OFF-PATH (d_1 == 0), so its shape is less pinned
  down and does not affect the certificate (dReach is reachable-set).

### Note

Peak stage-3 effort ~65 (vs the myopic final-stage optimum ~70) is the same
residual mu*(kappa) smoothing seen at T=2 — expected under Claim-B; the
verifier certifies the smoothed candidate.

## Deliverables

- Certificate table (per seed + cross-seed mean/std) — plan Table 2.
- Per-stage effort functions, BR-vs-learned, Δ_t(d) curves in the JSONs
  (plotting is a follow-up) — plan Figures 3-5.
- Qualitative check of expected economic patterns.

## Deferred

- Curriculum (T=1->2->3) ablation, adversarial-RL BR cross-check (plan 5.4).
- T=4,5 benchmark extensions (plan 5.5).
