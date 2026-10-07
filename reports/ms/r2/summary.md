# MS-R2: summary of the round (P0-P3)

Date: 2026-10-07. Branch `ms-r2` (from `origin/ms-r1` `71c58904`). Status: the pilot is done and reported; **STOP** after the push of section 4.3 of the prompt (`pi_record/19_ms_r2_prompt.md`); the next round is the PI's decision. No fresh seed (40501-40520), lock, T = 3 run or second pilot was used or started, no parameter changed after the pre-registration (`02_preregistration.md`), `main` and tags were not touched.

The question: is the stage-2 peak bias (the learned tie effort below the closed form in every one of the 120 runs) the policy-noise floor? A noise landing (Beta concentration scale ramped from 1 to s over local updates 2001-2200, held to 2400, LR decayed 3e-4 to 3e-5 over 2401-2800) crossed with the start sampler (bin-balanced / stratified), s in {1, 4, 16}, q in {50, 60}, seeds 10501-10510: 120 runs.

## Reading order

1. This file, then `05_decision_inputs.md` (the six arms side by side, what the noise landing changed in each component, what the sampler adds, the stop candidates, what is not settled).
2. `04_pilot.md` (checks first, the criterion, one section per arm, the decomposition by arm and segment, secondary tables, predictions against outcome, anomalies).
3. `02_preregistration.md` (design and decisions), `01_decomposition.md` (the premise check), `01b_stop_candidates.md` (D6 on MS-R1), `03_checks.md` (tests, review, launch checks, analysis), `00_housekeeping.md`, `pi_record/01_factcheck.md` (the fact-check ledger).

## Headline numbers (paths: `results/ms_r2/analysis/`; tables by `reports/ms/r2/report_scripts/pilot_tables.py`)

- **Checks:** 120 of 120 runs exit 0, manifests at the launch commit and clean, applied scale equals the D2 schedule in 120 of 120, C-NL 80 of 80, C-MS3 20 of 20, C-MS4 20 of 20, start shares 360 tests with none flagged (`results/ms_r2/pilot/launch_checks.json`). All 120 runs pass G-A and G-N(eta).
- **Primary criterion (D5): not met in any of the four rows.** Part (a) fails in all eight (sampler, s, q) cells; in one cell (`NL_bb_s4`, q = 60) the interval lies above 0 (+0.0114 [+0.0034, +0.0185]); part (b) holds (every run passes). Block `primary`:

| arm | baseline | (a) q=50 mean [95% CI] | met q=50 | (a) q=60 mean [95% CI] | met q=60 | pairs | (b) | overall |
|---|---|---|---|---|---|---|---|---|
| `NL_bb_s4` | `NL_bb_s1` | -0.00563 [-0.02330, +0.00995] | no | +0.01136 [+0.00335, +0.01849] * | no | 10+10 | holds | not met |
| `NL_bb_s16` | `NL_bb_s1` | -0.00087 [-0.01324, +0.01022] | no | +0.00715 [-0.00376, +0.01742] | no | 10+10 | holds | not met |
| `NL_st_s4` | `NL_st_s1` | +0.00134 [-0.01297, +0.01591] | no | -0.00589 [-0.01762, +0.00532] | no | 10+10 | holds | not met |
| `NL_st_s16` | `NL_st_s1` | -0.00323 [-0.01421, +0.00787] | no | -0.00461 [-0.01917, +0.00990] | no | 10+10 | holds | not met |

- **The smoothing part fell as predicted; the remainder rose and offset at least 46 % of the smoothing reduction in every cell.** sigma_2(0) fell by 1.14-1.78 effort units and the smoothing part by 0.64-1.39 (1.1-2.0 percentage points of `e_2*(0)`) in all eight cells, 10 of 10 seeds each; the measured sigma(s) is 1.01-1.10 times the 1/sqrt(s) prediction in every run and the smoothing part equals `e_2*(0) sigma_2(0) / (sqrt(pi) q)` to 0.083 % at worst. The remainder rose in every cell (+0.30 to +1.40 effort units; intervals above 0 in four of eight, none below), the gap changed by -0.39 to +0.66 (no interval below 0), the fraction of the smoothing reduction that reached the gap lay between -104 % and +54 % (`transmission`). The mean tie effort of the learned policy followed the sharper target only partly and not consistently (`tie_effort`).
- **Elsewhere:** RMSE_pos / `e_2*(0)`, eta_2 and R were lower under s > 1 in seven of eight cells (intervals below 0 in 3, 2, 3 cells); the tail mean rose slightly (four intervals above 0, at most +0.0007).
- **Sampler** (`sampler`, descriptive): the stratified arms have the lower remainder, gap and |peak| in five of six (s, q) cells, two of them with intervals excluding 0; the interaction of D5 excludes 0 only at s = 4, q = 60 (|peak| -0.0173 [-0.0324, -0.0033]), where the bin-balanced arm's |peak| rose.
- **Against `parents_A` with MS-R1's criterion** (descriptive): `NL_st_s4` and `NL_st_s16` are "met", `NL_st_s1` is not; the comparison confounds budget, sampler and scale.
- **Stop candidates** (D6 on the 120 pilot runs): C1 = R0 ranks |peak| as before (Spearman 0.991); the C2 floor falls with s while |c2| rises (at s = 16 the median |c2| is 5.1-6.7 times the median floor); no threshold is selected (`05_decision_inputs.md` section 5).

## What this does not show

An interval that contains 0 with ten seeds does not show the absence of an effect; the primary intervals (half-widths 0.008-0.017) are about as wide as the change the smoothing reduction alone would give (0.011-0.020). The round does not say why the remainder grows (the PPO diagnostics co-vary with s; the LR was not varied at fixed s). Nothing here selects an arm or proposes a protocol.

## Deviations and recorded departures

1. **A flag moved out of the rule dataclass** (P1, found by the full suite: it changed the logged rule parameters of MS-R1 runs; `03_checks.md` section 1): `legacy_global_sampler` is a `StageController` constructor argument, MS-R1 tests unchanged.
2. **C-MS4 `p_digest_next`** (P1, independent review MAJOR, fixed): the one column that carries the polishing sampler's digest is not compared where the MS-R1 reference polishes; `p_digest` still is.
3. **Transmission sign** (decision 10 of `02_preregistration.md`): the ratio is written without the prompt's minus sign (the prompt's sign contradicts its own endpoints).
4. **Report supplements not in D5** (labelled in `04_pilot.md`): the `sampler` block (same-s comparison of the samplers), the `window` block (the paired changes averaged over the last eight checks, post hoc), the `tie_effort`, `directions` and `side_by_side` blocks. They add no test to the pre-registered ones.
5. **Local ref of `main`** of the primary checkout is behind `origin/main` as before (owner-side); the local ref `ms-r1` of the P1 worktree is behind `origin/ms-r1` (recorded in `00_housekeeping.md`, not touched).

## Commands

```
# analysis (written to results/ms_r2/analysis/; blind recomputation: tools/ms/r2_blind_criterion.py --analysis-dir <out>)
python tools/ms/r2_analysis.py --pilot-root results/ms_r2/pilot --ms-r1-pilot-root results/ms_r1/pilot \
  --parents-root <v2-t2-refine>/results/v2_refine/parents_A --rehearsal-root <v2-t2-refine>/results/v2_T2_locked/rehearsal_v2_0 \
  --calibration-root results/ms_r1/calibration --out results/ms_r2/analysis
# tables
python reports/ms/r2/report_scripts/pilot_tables.py --list
python reports/ms/r2/report_scripts/stop_tables.py --dir results/ms_r2/stop_calibration_pilot --block <block>
```
