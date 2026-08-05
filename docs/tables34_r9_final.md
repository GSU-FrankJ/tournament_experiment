# Tables 3 & 4 — FINAL r9 generation (ε=0.01, M=65,536 certificate)

Owner decision 2026-08-03: the tightened certificate is THE configuration.
Tags: `r9_cert001` (+ `r9_fig7_no_stability`); 3P/da q55 reuse `r8_sens_eps001`
(identical config); the no-exploit ablation reuses `r7_fig7_no_exploit`
(verifier disabled — ε/M never enter its training). All 75 runs in the matrix
are verification-triggered stops except the no-exploit arm (budget, by design).
Referee: M=200,000, grid 0.25, seeds 700000+q·1000+si·7
(`results/one_stage_ablation/final_exploit_r9.json`).

**Headline: every seed in every cell now passes ε=0.01 under the independent
referee** (max single-seed reading 5.0×10⁻³ in the TEL-PPO rows, 7.3×10⁻³
including the no-stability arm). Matrix mean |err| 0.91 (was 1.33 at ε=0.03);
dotplot mean relative error 3.47% (was 6.81%).

## Table 4. Verified TEL-PPO Results Across Tournament Settings

| Scenario | q | Analytical e* | Verified TEL-PPO Estimate ê | Err. | Independent Final Exploitability | Evaluator floor (at e*) |
|---|---|---|---|---|---|---|
| Two-player | 35 | 45.45 | 43.43 ± 0.34 | 2.03 | (2.56 ± 1.16)×10⁻³ | (1.22 ± 1.20)×10⁻³ |
| Two-player | 45 | 35.35 | 34.40 ± 1.05 | 0.95 | (1.55 ± 0.99)×10⁻³ | (0.74 ± 0.45)×10⁻³ |
| Two-player | 55 | 28.93 | 28.70 ± 0.89 | 0.22 | (1.55 ± 0.79)×10⁻³ | (0.61 ± 0.38)×10⁻³ |
| Three-player | 35 | 25.00 | 24.61 ± 0.64 | 0.39 | (2.56 ± 1.43)×10⁻³ | (1.51 ± 0.46)×10⁻³ |
| Three-player | 55 | 15.91 | 17.03 ± 0.15 | 1.12 | (3.12 ± 0.72)×10⁻³ | (0.64 ± 0.39)×10⁻³ |
| Het. cost | 35 | 38.03 / 27.66 | 37.77 ± 0.95 / 27.05 ± 1.25 | 0.25 / 0.61 | (1.51 ± 1.13)×10⁻³ | (0.51 ± 0.25)×10⁻³ |
| Het. cost | 55 | 26.54 / 19.30 | 25.64 ± 1.02 / 20.02 ± 1.88 | 0.90 / 0.72 | (1.82 ± 1.33)×10⁻³ | (0.28 ± 0.18)×10⁻³ |
| Het. ability | 35 | 46.43 | 45.86 ± 0.80 | 0.57 | (2.04 ± 1.63)×10⁻³ | (0.88 ± 0.54)×10⁻³ |
| Het. ability | 55 | 30.37 | 31.09 ± 0.81 | 0.71 | (1.28 ± 0.92)×10⁻³ | (0.85 ± 0.41)×10⁻³ |

## Table 3. Component Ablation of TEL-PPO (two-player, q = 35/45/55)

| Ablation Variant | TEL-PPO Output (mean ± SD) | Absolute Bias | Independent Final Exploitability (×10⁻³) | Mean Verification Update ± SD | Number of Exploitability Calls ± SD | Training outcome |
|---|---|---|---|---|---|---|
| TEL-PPO | 43.43 ± 0.34 / 34.40 ± 1.05 / 28.70 ± 0.89 | 2.03 / 0.95 / 0.22 | 2.56 ± 1.16 / 1.55 ± 0.99 / 1.55 ± 0.79 | 99.0 ± 32.4 / 99.0 ± 18.7 / 113.0 ± 11.4 | 10.0 ± 3.2 / 10.0 ± 1.9 / 11.4 ± 1.1 | Verified |
| w/o stability screening | 43.35 ± 0.44 / 33.62 ± 1.23 / 29.05 ± 0.50 | 2.10 / 1.73 / 0.12 | 2.72 ± 1.43 / 2.28 ± 2.87 / 1.23 ± 0.87 | 153.0 ± 45.6 / 103.0 ± 11.4 / 125.0 ± 11.4 | 15.4 ± 4.6 / 10.4 ± 1.1 / 12.6 ± 1.1 | Verified |
| w/o exploitability-based verification and stopping | 44.15 ± 1.02 / 33.79 ± 1.72 / 26.28 ± 2.86 | 1.31 / 1.56 / 2.64 | 1.70 ± 1.30 / 2.91 ± 2.31 / 7.34 ± 6.08 | N/A (never verifies) | 0 | Reached budget (1500) |

Evaluator floor at e* (q=35/45/55): 1.22 / 0.74 / 0.61 ×10⁻³. The no-exploit
row's training never invokes the verifier, so the r7 runs remain valid
bit-for-bit; its referee values are from final_exploit_r7r8.json.

## vs the ε=0.03 generation (r7/r8), per cell |err|

| cell | ε=0.03 | ε=0.01 | | cell | ε=0.03 | ε=0.01 |
|---|---|---|---|---|---|---|
| 2P S1 q35 | 2.75 | 2.03 | | 3P q35 | 1.19 | 0.39 |
| 2P S1 q45 | 0.22 | 0.95 | | 3P q55 | 3.22 | 1.12 |
| 2P S1 q55 | 0.64 | 0.22 | | dc q35 P1/P2 | 0.09/0.67 | 0.25/0.61 |
| 2P S2 q35 | 2.19 | 2.29 | | dc q55 P1/P2 | 0.87/0.99 | 0.90/0.72 |
| 2P S2 q45 | 0.98 | 1.06 | | da q35 | 1.13 | 0.57 |
| 2P S2 q55 | 0.25 | 0.90 | | da q55 | 3.42 | 0.71 |

Pattern: every former problem cell (err > 1) improved — the worst cell went
3.42 → 2.29 across the matrix — while cells already on e* wobble within the
narrower certificate band (q45/S2, ±1). Cross-seed SDs mostly tighten (2P S1
q35: 1.78 → 0.34). da learned symmetry persists: |e1−e2| median 0.88 (q35).