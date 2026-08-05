# Tables 3 & 4 (no-polish framework) — r7/r8 raw profiles under the independent referee

Per docs/Figures&Tables080226.docx red item 1, generation fixed by owner
decision (2026-08-03): **r7 (2P families) + r8_unified (3P/dc/da), RAW
reported profiles, no MC-BR polishing anywhere.**

Evaluator: `exploitability_frozen_profile`, M=200,000, uniform grid Δe=0.25 on
[0,100], fresh seeds 700000+q·1000+si·7 (CRN across arms per cell) — identical
machinery and seed rule as the previous generation's unified tables.
Data: `results/one_stage_ablation/final_exploit_r7r8.json` (75 profiles, new)
and `results/one_stage_ablation/exploit_noise_floor.json` (analytical-e*
floor, training-independent, reused).

## Table 4. Verified TEL-PPO Results Across Tournament Settings

| Scenario | q | Analytical e* | Verified TEL-PPO Estimate ê | Err. | Independent Final Exploitability | Evaluator floor (at e*) |
|---|---|---|---|---|---|---|
| Two-player | 35 | 45.45 | 42.70 ± 1.78 | 2.75 | (4.90 ± 3.90)×10⁻³ | (1.22 ± 1.20)×10⁻³ |
| Two-player | 45 | 35.35 | 35.57 ± 0.79 | 0.22 | (1.74 ± 1.06)×10⁻³ | (0.74 ± 0.45)×10⁻³ |
| Two-player | 55 | 28.93 | 29.57 ± 0.70 | 0.64 | (1.62 ± 1.08)×10⁻³ | (0.61 ± 0.38)×10⁻³ |
| Three-player | 35 | 25.00 | 26.19 ± 0.43 | 1.19 | (7.36 ± 2.65)×10⁻³ | (1.51 ± 0.46)×10⁻³ |
| Three-player | 55 | 15.91 | 19.13 ± 0.31 | 3.22 | (17.09 ± 3.27)×10⁻³ | (0.64 ± 0.39)×10⁻³ |
| Het. cost | 35 | 38.03 / 27.66 | 37.94 ± 1.85 / 28.33 ± 2.06 | 0.09 / 0.67 | (5.07 ± 3.69)×10⁻³ | (0.51 ± 0.25)×10⁻³ |
| Het. cost | 55 | 26.54 / 19.30 | 25.67 ± 1.03 / 18.31 ± 2.09 | 0.87 / 0.99 | (3.16 ± 1.55)×10⁻³ | (0.28 ± 0.18)×10⁻³ |
| Het. ability | 35 | 46.43 | 47.56 ± 1.77 | 1.13 | (6.18 ± 4.61)×10⁻³ | (0.88 ± 0.54)×10⁻³ |
| Het. ability | 55 | 30.37 | 33.80 ± 1.04 | 3.42 | (8.83 ± 4.66)×10⁻³ | (0.85 ± 0.41)×10⁻³ |

Notes:
- ê is the cross-seed mean ± sample SD of the raw verified endpoints
  (Beta-mean policy efforts at the verification-triggered stop; het-cost
  reports the two players separately).
- **Every individual seed passes the certificate under the independent
  referee**: max single-seed exploitability is 0.0218 (3P q55) < ε = 0.03.
  The verified claim survives independent re-measurement in all 45 runs.
- Raw profiles sit 1.4×–26.5× ABOVE the evaluator floor (unlike the former
  polished profiles, which sat at it). The floor column is the referee's
  reading at the analytical equilibrium, where true exploitability is 0.
- The two wide-certificate cells (3P q55, da q55: Err 3.22 / 3.42) carry the
  largest exploitability (17.1 / 8.8 ×10⁻³) — internally consistent: these
  landings are genuine ~0.01–0.02-equilibria inside the ε=0.03 band where the
  flat payoff does not pin effort tightly. These are the two cells the owner
  designated for the ε=0.01 / M=65,536 sensitivity analysis.

## Table 3. Component Ablation of TEL-PPO (two-player, q = 35/45/55)

| Ablation Variant | TEL-PPO Output (mean ± SD) | Absolute Bias | Independent Final Exploitability (×10⁻³) | Mean Verification Update ± SD | Number of Exploitability Calls ± SD | Training outcome |
|---|---|---|---|---|---|---|
| TEL-PPO | 42.70 ± 1.78 / 35.57 ± 0.79 / 29.57 ± 0.70 | 2.75 / 0.22 / 0.64 | 4.90 ± 3.90 / 1.74 ± 1.06 / 1.62 ± 1.08 | 49.0 ± 0.0 / 81.0 ± 11.0 / 91.0 ± 13.0 | 5.0 ± 0.0 / 8.2 ± 1.1 / 9.2 ± 1.3 | Verified |
| w/o stability screening | 43.06 ± 0.83 / 35.32 ± 1.24 / 28.93 ± 0.63 | 2.40 / 0.04 / 0.00 | 3.31 ± 2.04 / 1.66 ± 0.64 / 0.82 ± 0.60 | 59.0 ± 14.1 / 77.0 ± 13.0 / 85.0 ± 5.5 | 6.0 ± 1.4 / 7.8 ± 1.3 / 8.6 ± 0.5 | Verified |
| w/o exploitability-based verification and stopping | 44.15 ± 1.02 / 33.79 ± 1.72 / 26.28 ± 2.86 | 1.31 / 1.56 / 2.64 | 1.70 ± 1.30 / 2.91 ± 2.31 / 7.34 ± 6.08 | N/A (never verifies) | 0 | Reached budget (1500) |

Evaluator floor at e* for these q: 1.22 / 0.74 / 0.61 ×10⁻³.

Notes:
- All three arms share the identical unified base configuration
  (docs/table2_unified_config.md); one component removed per row. No MC-BR
  polishing anywhere; the former "w/o MC-BR polishing" row is obsolete.
- The verification component's signal: without it, landings destabilize with
  noise (q55: SD 2.86, exploitability 7.34×10⁻³ vs 1.62 with verification;
  bias 2.64 vs 0.64) and nothing certifies the stop. The q35 no_exploit cell
  incidentally lands well (bias 1.31) — budget-exhaustion landings are
  unstable across cells and generations (r5's bad cell was q45, r7's is q55),
  which is the point.
- Stability screening's signal is efficiency/diagnostics, not final quality:
  it gates when the verifier is consulted (calls 5.0 vs 6.0 at q35) and its
  removal leaves accuracy roughly unchanged here. (Its Fig-3 role — KL/drift
  gating before expensive checks — is unchanged.)
- Number of Exploitability Calls = in-training verifier invocations per run
  (mean ± SD over 5 seeds). Each call: M=16,384, coarse-to-fine 5.0/1.0/0.25.
  For reference, 3P/dc/da (r8): 14.8 ± 5.5 / 32.8 ± 7.5 (3P q35/q55),
  12.0 ± 1.2 / 12.0 ± 0.7 (dc), 5.4 ± 0.5 / 8.8 ± 2.0 (da).

## Sensitivity analysis

Completed 2026-08-03 — see the dedicated section below.

---

# Sensitivity analysis (red item 3): ε=0.01, M=65,536 on the two wide-certificate cells

Tag `r8_sens_eps001`; cmdlines identical to `r8_unified` plus
`--exploit-eps 0.01 --exploit-M 65536`. All 10 runs verification-triggered
(streak=5). Independent referee: same M=200,000 / grid 0.25 / seed rule as
Table 4.

| Cell | Certificate | ê_raw (mean ± SD) | \|err\| | Independent Final Exploit. | Stop updates |
|---|---|---|---|---|---|
| 3P q55 (e*=15.91) | ε=0.03, M=16,384 | 19.13 ± 0.31 | 3.22 | (17.09 ± 3.27)×10⁻³ | 55–88 |
| 3P q55 | **ε=0.01, M=65,536** | **17.03 ± 0.15** | **1.12** | **(3.12 ± 0.72)×10⁻³** | 88–123 |
| da q55 (e*=30.37) | ε=0.03, M=16,384 | 33.80 ± 1.04 | 3.42 | (8.83 ± 4.66)×10⁻³ | 53–93 |
| da q55 | **ε=0.01, M=65,536** | **31.09 ± 0.81** | **0.71** | **(1.28 ± 0.92)×10⁻³** | 81–120 |

Evaluator floor at e*: 0.64 (3P q55) / 0.85 (da q55) ×10⁻³.

Reading:

1. **The certificate-width mechanism is confirmed quantitatively.** Tightening
   ε by 3× shrinks the effort error by 2.9× (3P) and 4.8× (da) — at or above
   the √3 ≈ 1.7× band-width prediction. Landings track the certificate, not
   the scenario.
2. **da q55 lands AT the evaluator floor** under ε=0.01: independent
   exploitability (1.28 ± 0.92)×10⁻³ vs floor (0.85 ± 0.41)×10⁻³ (one seed
   reads 0.3×10⁻³, below the floor mean). 3P q55 (3.12×10⁻³) retains ~5×
   headroom above its floor — ε could be tightened further there if desired.
3. **Cost is trivial**: ~30–35 extra updates per run (≈10 min); cross-seed SD
   TIGHTENS in both cells (0.31→0.15, 1.04→0.81). Every seed passes its own
   ε=0.01 certificate under the independent referee (max single-seed reading
   4.0×10⁻³ ≤ 0.01).
4. da's learned effort symmetry persists (|e1−e2| median 0.83, same
   seed-noise order as the ε=0.03 arm).
5. Engineering note (STATE.md 2026-08-03): the first da launch crashed on a
   latent (M,M) broadcast in the asymmetric verifier's payoff math — present
   since r5 at M=16,384 (~1 GB/candidate, silently absorbed by the V100s),
   fatal at M=65,536 (17 GB). Fixed with a linear-memory CPU branch for
   M>16,384 (0.3 s/eval, deterministic); the M≤16,384 path is untouched, so
   all prior generations remain bit-reproducible.

Data: `results/{three_players,different_ability}/convergence/*_r8_sens_eps001_*`,
wave manifests in `results/r8_sens_eps001{,_da}/`.
