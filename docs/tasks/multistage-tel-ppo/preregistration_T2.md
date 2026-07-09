# Pre-registration: T=2 acceptance gate (FROZEN 2026-07-09)

This document fixes the T=2 acceptance criteria and the training protocol
BEFORE the gated multi-seed run, so the criteria cannot be chosen after
seeing the outcome (plan section 5, owner decision). Thresholds are
justified by the independent verifier error floor and the one-stage
precedent — sources that do NOT depend on the multi-stage training result.
The step-3 pipeline validation (single seed) is a plumbing check and is
NOT used to set thresholds.

Threshold constants are frozen in `utils/multi_stage_metrics.py`
(`GATE_DREACH_OVER_DW`, `GATE_MIN_CERT_SEEDS_FRAC`, `TARGET_RE_1`,
`TARGET_RPE_2_CORE`).

## Framing (why the certificate is the gate, not recovery)

Owner decision 2026-07-09: multi-stage adopts the Claim-B framing. PPO
self-play converges to the exploration-smoothed equilibrium mu*(kappa),
not the sharp e* (four concordant one-stage negatives; see docs/STATE.md).
Therefore the deliverable is a CERTIFIED approximate MPE, and the
acceptance test is the independent best-response certificate — not the
recovery error e_hat ~ e* (which the smoothed candidate misses by design,
especially in the tails where g2 -> 0).

## PRIMARY gate (hard; decides "proceed to T=3")

- **G1 certification (per seed):** conservative reachable-set certificate
  `dReach / DW <= 0.03`. Rationale: plan's EXP^UCB/DW band is 1-3%; we take
  the lenient end (3%) and apply it to dReach, which UPPER-BOUNDS root
  exploitability (phase03) so it cannot over-certify. The verifier error
  floor is dReach(e*_CF)/DW ~ 2.5e-5 — five orders of magnitude below the
  threshold, so the gate measures the policy, not numerics.
- **G2 reproducibility:** `>= 4/5 seeds` certify (G1). A robust result must
  certify across seeds, not on a lucky one.

PASS => the T=2 pipeline (train -> extract -> verify -> certify) is
validated and T=3 GPU spend is authorized. FAIL => diagnose before any T=3.

## SECONDARY recovery diagnostics (reported; NOT gating)

Reported for transparency and because the plan's Experiment 1 requests them;
a miss is explained by exploration smoothing, not a pipeline failure.

- RE_1 target <= 0.10  (stage-1 relative error; one-stage precedent 2-4%).
- RPE_2_core target <= 0.15 over |d| <= q (stage-2 relative policy error on
  the interior core; the full |d| <= 2q RPE is also reported but the tails,
  where g2 -> 0 and the smoothed policy over-exerts, dominate it).
- Also reported: AE_1, MAE_2, RMSE_2, RPE_2 (|d|<=2q), PL_2 = U(e*_CF) -
  U(e_hat) and PL_2/DW, and per-seed EXP, EXP^UCB, dReach, dFull, on-path Δ.

## EXP^UCB for a deterministic verifier

The DP verifier is deterministic (quadrature), so there is no Monte Carlo
SE. Numerical uncertainty is estimated from grid refinement:
`EXP^UCB = EXP_201 + |EXP_201 - EXP_101|` (the Richardson residual as an
error band). Certification uses dReach (conservative), with EXP^UCB reported
alongside. Cross-seed mean +/- std is the seed-robustness report.

## Training protocol (frozen)

| Item | Value |
|---|---|
| Parameters | w_h=6, w_l=2, k=1/3500, e_bar=100, q=50 (in [q_crit=41.83, .]) |
| Horizon | T=2 |
| Seeds | 42, 43, 44, 45, 46 (5 seeds) |
| Updates | 2000 |
| Episodes / update | 512 |
| gamma, lambda | 1.0, 1.0 (finite-horizon economic payoff) |
| Policy | Beta (mean/conc), state [t/T, d/(q sqrt t)] |
| lr / clip / entropy | 3e-4 / 0.2 / 0.005 |
| Epochs / minibatch | 10 / 256 |
| Exploring starts | on, es_on_path_fraction=0.5 |
| Extraction | Beta MEAN (repo invariant) |
| Checkpoint rule | lowest validation dReach over training (pre-specified) |
| Verifier | DP, d-grid 201, e-grid 401, closed-form terminal, dReach cert |

The selected checkpoint is the lowest-validation-dReach one (pre-specified
rule), NOT a visual fit to the benchmark.

## Decision rule

The gate is evaluated by `tools/evaluate_gate.py` over the five per-seed
convergence JSONs. Its printed verdict (PASS/FAIL + per-seed table) is the
record. Any deviation from this protocol after freezing is logged in STATE.
