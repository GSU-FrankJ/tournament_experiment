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

## T=4 / T=5 benchmark extensions (COMPLETE 2026-07-09) — BOTH GATES PASSED

Plan section 5.5. Pre-registered in `preregistration_T4_T5.md` (frozen
80ee48c, before the runs). Same verifier/threshold; grid-stable at T=4/5
(<0.3% EXP change 201->401). Budget 1000*T updates x 512 ep, seeds 42-46,
CPU (~45 min T=4, ~75 min T=5).

- **T=4: GATE PASS — 5/5 certify.** dReach/DW mean 0.0155 (max 0.0217),
  EXP/DW mean 0.0044.
- **T=5: GATE PASS — 4/5 certify** (exactly the 80% threshold). dReach/DW
  mean 0.0190 (max 0.0321). Seed 46 is the lone non-certifier
  (dReach/DW=0.0321, just over 0.03; EXP/DW still tiny at 0.0062). T=5 is
  where the conservative reachable-Δ certificate starts to bind — the
  expected "certification degrades at larger horizons" (plan). Reported
  with numerical caveats, not a blocker (benchmark extension).

Results: `results/multi_stage/convergence/ms_T{4,5}_q50_seed{42..46}_gate{T4,T5}_convergence.json`.

## Multi-stage summary (plan Table 4; cross-seed mean)

| T | certify | dReach/DW (max) | EXP/DW | total effort | e_hat_t(0) per stage |
|---|---|---|---|---|---|
| 2 | 5/5 | 0.0063 (0.0078) | 0.0016 | 93.3* | ~[46.7, 46.7]* (recovers CF) |
| 3 | 5/5 | 0.0097 (0.0144) | 0.0027 | 108.1 | [43.3, 50.9, 64.8] |
| 4 | 5/5 | 0.0155 (0.0217) | 0.0044 | 116.8 | [40.3, 42.6, 50.2, 64.0] |
| 5 | 4/5 | 0.0190 (0.0321) | 0.0051 | 128.0 | [38.9, 39.4, 41.9, 47.4, 59.8] |

*T=2 total effort = 2*g1 (analytic recovered value); T=2 run predates the
effort_curves/onpath_summary fields.

### Main Questions (plan) answered

- **Q1 (effort rises toward the final stage):** YES at every T —
  e_hat_t(0) is increasing in t within each row.
- **Q4 (total expected effort increases with T):** YES, monotone
  93.3 -> 108.1 -> 116.8 -> 128.0.
- **Q3 (hump-shaped in the gap):** YES at every stage/horizon (verified in
  the T=3 curves; same shape at T=4/5).
- Final-stage effort e_hat_T(0) ~ 60-64 across T (near the myopic ~70, with
  the residual mu*(kappa) smoothing). As T grows, early-stage effort FALLS
  (T=5 stage-1 38.9 vs T=3 stage-1 43.3): more remaining stages => more
  chance to catch up later => less early effort.
- **Certificate degrades monotonically with T** (dReach/DW 0.006 -> 0.010
  -> 0.016 -> 0.019); T=5 is the first horizon with a non-certifying seed.

## Deliverables — status

- Certificate tables per T (per seed + cross-seed) — done (in JSONs + above).
- Per-stage effort / BR-vs-learned / Δ_t(d) curves — saved in the JSONs.
- **Figures + tables generated** (2026-07-09): `tools/make_multistage_figures.py`
  and `tools/make_multistage_tables.py` produce plan Figures 1-5 and
  Tables 1-4 into `paper/multistage/` (PDFs + .tex committed; PNGs
  gitignored). Fig 3 (learned effort functions) is the main result; Fig 4
  shows learned tracking BR; Fig 2 shows the verifier separating equilibrium
  from bad policies. See `paper/multistage/README.md`.
  - Caveat: Fig 1 / Table 4's T=2 columns are sparse/analytic (the T=2 runs
    predate the effort_curves/onpath_summary fields); re-run one T=2 seed
    for a dense stage-2 curve if needed for the paper.

## Deferred

- Curriculum (T=1->2->3) ablation, adversarial-RL BR cross-check (plan 5.4).
- Optional dense-curve re-run of T=2 for a publication-quality Figure 1.
