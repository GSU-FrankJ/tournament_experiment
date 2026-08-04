# r7 results under the unified configuration — two-player families (2026-08-02)

**Scope.** Of the r7 wave, the runs that satisfy the full unified-configuration
requirement (one policy head = mean×concentration, one entropy/conc/var preset,
verifier M=16384, verification-triggered stopping with NO minimum-update floor)
are exactly the four two-player families below. The r7 3P/dc runs carry
`--min-updates 300` and r7 da carries 1000/head deviations, so their
unified-configuration replacements are the `r8_unified` wave (in progress);
they are NOT reported here.

**Configuration (verified from each run's JSON/manifest, identical across all
rows):** 4-dim state [q/60, k/10⁻³, Δw/10, (l_i−l̄₋ᵢ)/10 = 0]; mean×conc head;
entropy 0 entire run; conc/var preset warm-up 200 / ramp 50 (never reached —
all verified stops < 120 updates, so realized conc_min=100, var_coef=0);
verifier M=16384, ε=0.03, patience 5; no floor; budget cap 1500.
Full config: docs/table2_unified_config.md. Seeds 42–46 everywhere.

## Baselines (Table-4 rows, raw endpoints)

| Family | q | e* | ê_raw (mean ± SD) | Raw Err. | Stop update (mean ± SD) | Stop mode | Verifier exploit. at stop (mean ± SD) |
|---|---|---|---|---|---|---|---|
| 2P Set 1 | 35 | 45.45 | 42.70 ± 1.78 | 2.75 | 49.0 ± 0.0 | verified (streak=5) | 0.0109 ± 0.0030 |
| 2P Set 1 | 45 | 35.35 | 35.57 ± 0.79 | 0.22 | 81.0 ± 11.0 | verified (streak=5) | 0.0073 ± 0.0018 |
| 2P Set 1 | 55 | 28.93 | 29.57 ± 0.70 | 0.64 | 91.0 ± 13.0 | verified (streak=5) | 0.0064 ± 0.0008 |
| 2P Set 2 | 35 | 47.62 | 45.43 ± 1.82 | 2.19 | 61.0 ± 8.4 | verified (streak=5) | 0.0105 ± 0.0017 |
| 2P Set 2 | 45 | 37.04 | 36.05 ± 0.66 | 0.98 | 71.0 ± 11.0 | verified (streak=5) | 0.0069 ± 0.0008 |
| 2P Set 2 | 55 | 30.30 | 30.06 ± 1.06 | 0.25 | 87.0 ± 23.9 | verified (streak=5) | 0.0065 ± 0.0013 |

## Ablation arms (Table-3 rows, raw endpoints; base config identical, one component removed)

| Arm | q | e* | ê_raw (mean ± SD) | Raw Err. | Stop update (mean ± SD) | Stop mode |
|---|---|---|---|---|---|---|
| w/o stability screening | 35 | 45.45 | 43.06 ± 0.83 | 2.40 | 59.0 ± 14.1 | verified (streak=5) |
| w/o stability screening | 45 | 35.35 | 35.32 ± 1.24 | 0.04 | 77.0 ± 13.0 | verified (streak=5) |
| w/o stability screening | 55 | 28.93 | 28.93 ± 0.63 | 0.00 | 85.0 ± 5.5 | verified (streak=5) |
| w/o exploitability verification | 35 | 45.45 | 44.15 ± 1.02 | 1.31 | 1500 (budget) | never verifies |
| w/o exploitability verification | 45 | 35.35 | 33.79 ± 1.72 | 1.56 | 1500 (budget) | never verifies |
| w/o exploitability verification | 55 | 28.93 | 26.28 ± 2.86 | 2.64 | 1500 (budget) | never verifies |

## Per-seed raw endpoints (seeds 42→46, for the seed-level dot plots)

| Family | q | per-seed ê_raw |
|---|---|---|
| 2P Set 1 | 35 | 41.84, 40.24, 43.26, 43.18, 45.00 |
| 2P Set 1 | 45 | 35.57, 34.85, 34.73, 36.46, 36.25 |
| 2P Set 1 | 55 | 30.38, 29.40, 29.75, 29.81, 28.49 |
| 2P Set 2 | 35 | 46.76, 45.82, 43.89, 43.23, 47.44 |
| 2P Set 2 | 45 | 35.72, 35.45, 36.53, 35.60, 36.97 |
| 2P Set 2 | 55 | 30.95, 29.06, 31.25, 30.11, 28.93 |
| w/o stability | 35 | 43.91, 43.75, 43.23, 42.31, 42.09 |
| w/o stability | 45 | 35.49, 33.62, 36.54, 36.39, 34.54 |
| w/o stability | 55 | 29.66, 28.70, 29.13, 29.16, 27.97 |
| w/o exploit. | 35 | 45.67, 43.30, 44.34, 44.32, 43.11 |
| w/o exploit. | 45 | 31.99, 35.44, 34.31, 31.96, 35.27 |
| w/o exploit. | 55 | 30.08, 28.27, 23.33, 25.76, 23.97 |

## Polished endpoints (MC-BR, canonical POL; stage A COMPLETE, 60 rows)

Dual-endpoint table, all four two-player families (polish seeds 4000+si
shared across the three Table-3 arms per (q, si) — common random numbers, so
arm-to-arm polished differences are start-driven only):

| Arm | q | raw mean ± SD | polished mean ± SD | Polished Err. |
|---|---|---|---|---|
| TEL-PPO | 35 | 42.70 ± 1.78 | 44.949 ± 0.020 | 0.51 |
| TEL-PPO | 45 | 35.57 ± 0.79 | 35.091 ± 0.031 | 0.26 |
| TEL-PPO | 55 | 29.57 ± 0.70 | 28.762 ± 0.029 | 0.16 |
| w/o stability screening | 35 | 43.06 ± 0.83 | 44.948 ± 0.029 | 0.51 |
| w/o stability screening | 45 | 35.32 ± 1.24 | 35.086 ± 0.030 | 0.27 |
| w/o stability screening | 55 | 28.93 ± 0.63 | 28.762 ± 0.030 | 0.16 |
| w/o exploitability verif. | 35 | 44.15 ± 1.02 | 44.937 ± 0.036 | 0.52 |
| w/o exploitability verif. | 45 | 33.79 ± 1.72 | 35.086 ± 0.020 | 0.27 |
| w/o exploitability verif. | 55 | 26.28 ± 2.86 | 28.761 ± 0.032 | 0.16 |
| 2P Set 2 (baseline) | 35 | 45.43 ± 1.82 | 47.054 ± 0.030 | 0.57 |
| 2P Set 2 (baseline) | 45 | 36.05 ± 0.66 | 36.744 ± 0.042 | 0.29 |
| 2P Set 2 (baseline) | 55 | 30.06 ± 1.06 | 30.126 ± 0.027 | 0.18 |

**Pre-registered prediction CONFIRMED (branch A).** The polished column
collapses across arms: per-q spread across the three Table-3 arms is 0.012 /
0.005 / 0.001 effort units — inside the ±0.02–0.04 polish SDs. Even the
never-verified arm's worst landing (q55: 26.28 ± 2.86) polishes to the same
28.76 as everyone else. The polished column measures the game; the arm
separation (the ablation's evidence) lives entirely in the raw column, as
pre-registered in docs/ablation_narrative_preregistered.md. Set-1 polished
values match the r5 generation to 0.01 (44.95/35.09/28.76 both generations).

Stage-B polish (3P/dc/da from r8_unified) and the MC-BR-only baseline are
still running; their rows extend this file's JSONs when they land.

## Reading notes (relevant to methodology adjustments)

1. **Every baseline/no-stability run is a true verification stop** (stop_reason
   = exploitability, streak exactly 5) — under the unified no-floor rule the
   "floor expiry reported as convergence" issue cannot occur in these rows.
2. **q=35 Set 1 stops uniformly at update 49** (earliest possible streak
   completion) and lands lower (42.70) than the r5/M=8192 generation (43.58).
   Working hypothesis (pre-registered in the task STATE): doubling verifier M
   halves the estimator's maximization bias, so the 0.03 gate passes earlier →
   less training. The polished endpoint (44.95 ± 0.02, identical to r5)
   supports this: the training landing moved, the certified answer did not.
3. **Verifier exploitability at stop is 0.005–0.011** — consistent with the
   M=16384 estimator floor (~0.008 extrapolated); these values measure the
   instrument near its floor, not a true equilibrium gap. The independent
   M=200000 final check for r7/r8 has not been run yet.
4. **w/o exploitability verification degrades and destabilizes raw landings**
   (err up to 2.64, SD up to 2.86, never verifies), and its worst cell moved
   from q45 (r5) to q55 (r7) — budget-exhaustion landings are unstable across
   generations, which is itself evidence for the verification component.

---

# ADDENDUM (r8_unified landed): the compliant 3P / dc / da results

The r8 wave (30 runs) finished 0-failure; with it, ALL four scenarios now have
results under the full unified configuration. Every r8 run is a true
verification stop (stop_reason=exploitability, streak exactly 5).

## Table-4 rows (raw endpoints, unified stopping)

| Scenario | q | e* | ê_raw (mean ± SD) | Raw Err. | Stop updates | vs floored generation (r7/r5-style) |
|---|---|---|---|---|---|---|
| Three-player | 35 | 25.00 | 26.19 ± 0.43 | 1.19 | 43–62 | 23.66 ± 0.30, err 1.34 (floor 300) |
| Three-player | 55 | 15.91 | 19.13 ± 0.31 | **3.22** | 55–88 | 15.46 ± 0.84, err 0.45 (floor 300) |
| Het. cost P1/P2 | 35 | 38.03 / 27.66 | 37.94 ± 1.85 / 28.33 ± 2.06 | 0.09 / 0.67 | 89–129 | 38.51/27.67, err 0.48/0.02 (floor 300) |
| Het. cost P1/P2 | 55 | 26.54 / 19.30 | 25.67 ± 1.03 / 18.31 ± 2.09 | 0.87 / 0.99 | 105–127 | 26.71/19.60, err 0.17/0.30 (floor 300) |
| Het. ability | 35 | 46.43 | 47.56 ± 1.77 | 1.13 | **22–39** | 45.34 ± 1.77, err 1.08 (v2, floor 1000) |
| Het. ability | 55 | 30.37 | 33.80 ± 1.04 | **3.42** | 53–93 | 30.78 ± 1.98, err 0.41 (v2, floor 1000) |

da learned symmetry holds at these very short runs too: |e1−e2| median
0.53 (q35) / 0.76 (q55), max 1.53 — still seed-noise order.

## The finding this exposes (for the methodology discussion)

**The ε=0.03 certificate is wide in effort space when payoffs are flat.**
Curvature near e* is ~k, so the effort band consistent with exploitability
≤ 0.03 has half-width on the order of sqrt(0.03/k): ≈ 5.5 units at k=0.001
(3P), ≈ 7.7 at k=0.0005 (da). The q55 landings (err 3.2–3.4) sit INSIDE that
band — the certificate is satisfied exactly as designed; it just does not pin
the effort tightly where the payoff is flat.

Consequently the old floors were not decoration: r7/r5's better raw accuracy
at 3P/dc/da came from training PAST first certification. The unified rule
makes this visible and forces one scenario-agnostic choice:

1. **Tighten the certificate uniformly** — smaller ε (needs verifier M above
   16384 to stay off the estimator floor: ε=0.01 wants floor ≲0.004 ⇒
   M≈65536), same for all four scenarios; or
2. **One uniform burn-in for all four** — the same min-updates value
   everywhere (2P runner needs the flag added); "same clock for everyone" is
   defensible but stops are then partly clock-driven again; or
3. **Keep the clean rule and let MC-BR polish carry final accuracy** — raw
   column honestly reports the certificate width; the polished column (stage-B
   polish, in progress) is the paper's headline accuracy. Matches the
   pre-registered narrative.

## Realized-configuration simplification

Under unified stopping, every scenario verifies BEFORE the 200-update ramp
warm-up: realized settings are constant and identical everywhere —
conc_min=100, conc_scale=100, var_coef=0, entropy=0, M=16384, ε=0.03,
patience 5, no floor. The conc/var late schedule is inert in this generation.

---

# ADDENDUM 2 (MC-BR-only landed): what polishing does WITHOUT any training

81 rows (`results/one_stage_ablation/mc_br_only.json`): the canonical polish
run from uninformed starts — start=50 (PPO's own init mean, 5 polish seeds,
CRN with the TEL-PPO polish rows) and a 10/30/70/90 ladder. No training
anywhere.

| cell | e* | MC-BR-only (start 50) | err | ladder max dev | TEL-PPO-started polish |
|---|---|---|---|---|---|
| 2P q35 | 45.45 | 44.937 ± 0.024 | 0.52 | 0.54 | 44.949 ± 0.020 |
| 2P q45 | 35.35 | 35.088 ± 0.030 | 0.27 | 0.33 | 35.091 ± 0.031 |
| 2P q55 | 28.93 | 28.772 ± 0.027 | 0.15 | 0.21 | 28.762 ± 0.029 |
| 3P q35 | 25.00 | 24.750 ± 0.015 | 0.25 | 0.27 | (stage B pending) |
| 3P q55 | 15.91 | 15.840 ± 0.008 | 0.07 | 0.08 | (stage B pending) |
| dc q35 P1/P2 | 38.03 / 27.66 | 38.040 ± 0.031 / 27.674 ± 0.063 | 0.01 / 0.01 | 0.06 | (stage B pending) |
| dc q55 P1/P2 | 26.54 / 19.30 | 26.555 ± 0.029 / 19.287 ± 0.046 | 0.02 / 0.01 | 0.03 | (stage B pending) |
| da q35 | 46.43 | 46.461 ± 0.043 | 0.03 | 0.07 | (stage B pending) |
| da q55 | 30.37 | 30.361 ± 0.018 | 0.01 | 0.06 | (stage B pending) |

Three conclusions (all pre-registered branch A):

1. **Polish-only reproduces the polished column from any start.** Where the
   TEL-PPO-started polish exists (2P), MC-BR-only matches it to ≤0.012. The
   ladder shows start-independence (max spread ≤0.08 within any cell). It even
   recovers the ASYMMETRIC het-cost equilibrium (38.04/27.67) from a
   symmetric 50/50 start.
2. **"Polished Err." is the instrument's own bias, not a property of
   TEL-PPO.** The 2P offsets (−0.51/−0.26/−0.15) appear identically from
   trained and untrained starts — they are the polish estimator's systematic
   offset (max-bias + grid), consistent with all polished residuals being
   negative in Table 4.
3. **Therefore the paper must NOT claim TEL-PPO is needed to locate e*.**
   TEL-PPO's evidence is: the raw column (learning agents reach near-e* from
   own-play sampled rewards only — no counterfactual-deviation oracle), the
   online certification of stopping, and the policy object itself. MC-BR is
   the referee/refiner; it requires oracle queries at arbitrary profiles
   (150000 samples per grid point), which real tournament participants do not
   have. Full wording: docs/ablation_narrative_preregistered.md.

---

# FINAL ASSEMBLY (all pipelines complete): the two tables for the paper

## Table 4 — all scenarios under the ONE unified configuration (raw + polished)

2P rows from r7 (already compliant); 3P/dc/da rows from r8_unified.
Every run: verification-triggered stop, streak=5, no floor, M=16384, 4-dim
state, mean×conc head, entropy 0.

| Scenario | q | e* | ê_raw ± SD | Raw Err. | ê_polish ± SD | Polished Err. |
|---|---|---|---|---|---|---|
| Two-player S1 | 35 | 45.45 | 42.70 ± 1.78 | 2.75 | 44.949 ± 0.020 | 0.51 |
| Two-player S1 | 45 | 35.35 | 35.57 ± 0.79 | 0.22 | 35.091 ± 0.031 | 0.26 |
| Two-player S1 | 55 | 28.93 | 29.57 ± 0.70 | 0.64 | 28.762 ± 0.029 | 0.16 |
| Two-player S2 | 35 | 47.62 | 45.43 ± 1.82 | 2.19 | 47.054 ± 0.030 | 0.57 |
| Two-player S2 | 45 | 37.04 | 36.05 ± 0.66 | 0.98 | 36.744 ± 0.042 | 0.29 |
| Two-player S2 | 55 | 30.30 | 30.06 ± 1.06 | 0.25 | 30.126 ± 0.027 | 0.18 |
| Three-player | 35 | 25.00 | 26.19 ± 0.43 | 1.19 | 24.752 ± 0.015 | 0.25 |
| Three-player | 55 | 15.91 | 19.13 ± 0.31 | 3.22 | 15.837 ± 0.008 | 0.07 |
| Het. cost P1 | 35 | 38.03 | 37.94 ± 1.85 | 0.09 | 38.032 ± 0.033 | 0.00 |
| Het. cost P2 | 35 | 27.66 | 28.33 ± 2.06 | 0.67 | 27.668 ± 0.067 | 0.01 |
| Het. cost P1 | 55 | 26.54 | 25.67 ± 1.03 | 0.87 | 26.554 ± 0.034 | 0.01 |
| Het. cost P2 | 55 | 19.30 | 18.31 ± 2.09 | 0.99 | 19.297 ± 0.052 | 0.00 |
| Het. ability | 35 | 46.43 | 47.56 ± 1.77 | 1.13 | 46.451 ± 0.052 | 0.02 |
| Het. ability | 55 | 30.37 | 33.80 ± 1.04 | 3.42 | 30.370 ± 0.029 | 0.00 |

The q55 wide-certificate landings (3P 3.22, da 3.42) polish to 0.07 / 0.00 —
the "clean unified stopping + MC-BR final accuracy" pipeline (option 3 of the
methodology memo) is fully evidenced.

## Table 3 — five-row dual-endpoint component analysis (2P Set 1)

| Arm | q=35 raw | q=45 raw | q=55 raw | q=35 pol | q=45 pol | q=55 pol |
|---|---|---|---|---|---|---|
| TEL-PPO (full) | 42.70 ± 1.78 | 35.57 ± 0.79 | 29.57 ± 0.70 | 44.949 ± 0.020 | 35.091 ± 0.031 | 28.762 ± 0.029 |
| w/o stability screening | 43.06 ± 0.83 | 35.32 ± 1.24 | 28.93 ± 0.63 | 44.948 ± 0.029 | 35.086 ± 0.030 | 28.762 ± 0.030 |
| w/o exploitability verif. (never verifies) | 44.15 ± 1.02 | 33.79 ± 1.72 | 26.28 ± 2.86 | 44.937 ± 0.036 | 35.086 ± 0.020 | 28.761 ± 0.032 |
| MC-BR only (start 50, no training) | 50 (by constr.) | 50 | 50 | 44.937 ± 0.024 | 35.088 ± 0.030 | 28.772 ± 0.027 |
| MC-BR only (ladder 10/30/70/90) | start value | start value | start value | 44.92–44.98 | 35.02–35.11 | 28.71–28.80 |

Reading (pre-registered): the polished columns are indistinguishable across
ALL five rows — they measure the polish instrument (including its systematic
−0.5/−0.26/−0.15 offsets, which appear identically from untrained starts).
The component evidence is entirely in the raw columns and the stop behavior
(verified at 49–91 vs never-verifies; raw SD 0.63–1.78 vs 2.86).
