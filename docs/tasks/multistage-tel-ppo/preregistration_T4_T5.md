# Pre-registration: T=4 and T=5 benchmark extensions (FROZEN 2026-07-09)

Fixes the T=4/T=5 acceptance criteria and protocol BEFORE the runs. Per
plan section 5.5, these are BENCHMARK EXTENSIONS, not the main validation
layer (that is T=2 recovery + T=3 verified equilibrium, both PASSED). They
show how the certified-equilibrium method scales with the horizon.

## Same certificate, same verifier

The verifier is unchanged from T=2/T=3 (closed-form terminal integration,
deterministic quadrature, dReach reachable-set certificate). Its
discriminatory power and error floor were established at T=2. Numerical
adequacy at T=4/T=5 was checked before freezing: EXP and dReach change
< 0.3% when the score-gap grid is refined 201 -> 301 -> 401 (the
conservative D_max saturates in the tails, handled exactly by constant
extrapolation). The per-run grid refinement (51/101/201) + Richardson is
reported as the numerical-limitation check the plan requires.

## PRIMARY gate (hard)

- **G1 certification (per seed):** `dReach / DW <= 0.03` (frozen
  `GATE_DREACH_OVER_DW`, unchanged).
- **G2 reproducibility:** `>= 4/5 seeds` certify (frozen
  `GATE_MIN_CERT_SEEDS_FRAC = 0.8`).

Unlike T=2/T=3, a T=4 or T=5 FAIL is NOT a project blocker — these are
benchmark extensions. A miss is reported with the numerical caveats
(grid resolution, interpolation, best-response solver accuracy) per the
plan, and the horizon at which certification degrades is itself a result.

## Reported (NOT gating; no closed form)

- Per-stage learned effort functions e_hat_t(d), BR vs learned, Δ_t(d)
  (saved as effort_curves).
- Exploitability certificate: EXP, EXP/DW, EXP^UCB/DW, dReach/DW, dFull;
  cross-seed mean +/- std.
- On-path expected total effort and cost (onpath_summary) -> Main Q4 and
  the multi-stage summary Table 4 (T=2,3,4,5).
- Economic patterns vs T (Main Questions 1-5).

## Training protocol (frozen)

| Item | T=4 | T=5 |
|---|---|---|
| Updates | 4000 | 5000 |
| Episodes / update | 512 | 512 |
| Seeds | 42-46 | 42-46 |

All other settings identical to T=3: q=50, w_h=6, w_l=2, k=1/3500,
e_bar=100; gamma=lambda=1; lr 3e-4, clip 0.2, entropy 0.005, epochs 10,
minibatch 256; exploring starts on, es_on_path_fraction=0.5; Beta MEAN
extraction; lowest-validation-dReach checkpoint; verifier d-grid 201,
e-grid 401, closed-form terminal; CPU.

Rationale for the budget: updates = 1000 * T (T=4->4000, T=5->5000),
acknowledging the sparser terminal credit assignment over more stages.

## Deferred (unchanged from T=3)

Curriculum ablation, adversarial-RL BR cross-check, figure generation.

## Decision rule

`tools/evaluate_gate.py --glob ".../ms_T{4,5}_q50_seed*_gate{T4,T5}_convergence.json"`.
