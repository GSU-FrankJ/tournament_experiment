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

## Gated run (PENDING)

T=3, q=50, seeds 42-46, 3000 upd x 512 ep, entropy 0.005, CPU. Gate:
`dReach/DW <= 0.03`, `>= 4/5 seeds` (frozen). Evaluate with
`tools/evaluate_gate.py`.

## Deliverables

- Certificate table (per seed + cross-seed mean/std) — plan Table 2.
- Per-stage effort functions, BR-vs-learned, Δ_t(d) curves in the JSONs
  (plotting is a follow-up) — plan Figures 3-5.
- Qualitative check of expected economic patterns.

## Deferred

- Curriculum (T=1->2->3) ablation, adversarial-RL BR cross-check (plan 5.4).
- T=4,5 benchmark extensions (plan 5.5).
