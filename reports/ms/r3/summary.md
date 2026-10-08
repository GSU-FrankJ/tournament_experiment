# MS-R3: summary of the round (P0-P3)

Date: 2026-10-08. Branch `ms-r3` (from `origin/ms-r2` `e8eb9a08`). Status: the pilot is done and reported; **STOP** after the push of section 4.3 of the prompt (`pi_record/20_ms_r3_prompt.md`); the next round is the PI's decision. No fresh seed (40501-40520), lock, T = 3 experiment or second pilot was used or started, no parameter changed after the pre-registration (`02_preregistration.md`), `main` and tags were not touched.

The question: is the stage-2 tie deficit set by the actor's resolution at the kink of the equilibrium tent? The actor was replaced (the critic was not): `relu` (ReLU hidden units) and `t10` (tanh on a ten times finer d input) against `t1` (the current actor), crossed with the start sampler (bin-balanced / stratified) and the terminal-stage noise landing (s in {1, 16}), q in {50, 60}, seeds 10501-10510: 12 arms, 240 runs.

## Reading order

1. This file, then `05_decision_inputs.md` (the twelve arms side by side, what the actor changes in each component, whether the noise landing transmits, R0, the RL fit against the supervised screen, what is not settled).
2. `04_pilot.md` (checks first, the criterion, one section per actor, the noise landing, the quadrature check, the predictions against the outcome, resolution metrics, along the run, strata, stage 1, anomalies; the figures are embedded).
3. `01_supervised_screen.md` (the offline screen and the premise check), `01b_rl_actor_diagnostics.md` (the d-weights of the MS-R1 and MS-R2 actors), `02_preregistration.md` (design and decisions), `03_checks.md` (tests, review, C-R6), `00_housekeeping.md`, `pi_record/01_factcheck.md` (the fact-check ledger).

## Headline numbers (paths: `results/ms_r3/analysis/`; tables by `reports/ms/r3/report_scripts/pilot_tables.py`)

- **Premise check (P1): PASS.** Supervised fit at the RL budget, bin-balanced, median tip deficit q = 50 / 60: `t1` 1.64 / 6.32, `relu` 0.48 / 0.03, `t10` 0.53 / 0.48 effort units (`results/ms_r3/supervised_screen/premise_check.json`). C-R6: the unchanged v2.0 entry point reproduces `rehearsal_v2_0`, 20 of 20.
- **Checks (P2):** 240 of 240 runs exit 0; scale 240/240, C-INIT 20/20, C-NL 120/120, **C-MS5 80/80** (the four `t1` arms are bit-identical re-runs of MS-R2's `NL_*` arms over the whole run), start shares 720 tests with 0 flagged (`results/ms_r3/pilot/launch_checks.json`).
- **Primary criterion (D6): not met in any of the eight rows.** Block `primary`:

| arm | baseline | (a) q=50 mean [95% CI] | met q=50 | (a) q=60 mean [95% CI] | met q=60 | pairs | (b) | overall |
|---|---|---|---|---|---|---|---|---|
| `relu_bb_s1` | `t1_bb_s1` | +0.06907 [-0.03629, +0.26956] | no | -0.00535 [-0.02069, +0.01116] | no | 10+10 | violated (q50/10504 q50/10506) | not met |
| `relu_bb_s16` | `t1_bb_s16` | +0.06906 [-0.04565, +0.27448] | no | -0.03024 [-0.03721, -0.02308] * | yes | 10+10 | violated (q50/10504) | not met |
| `relu_st_s1` | `t1_st_s1` | -0.01077 [-0.03037, +0.00587] | no | -0.02339 [-0.03867, -0.00821] * | yes | 10+10 | violated (q50/10506) | not met |
| `relu_st_s16` | `t1_st_s16` | -0.01039 [-0.01625, -0.00467] * | yes | -0.02059 [-0.03090, -0.00684] * | yes | 10+10 | violated (q50/10506) | not met |
| `t10_bb_s1` | `t1_bb_s1` | -0.01330 [-0.03590, +0.00859] | no | +0.01456 [+0.00187, +0.02646] * | no | 10+10 | holds | not met |
| `t10_bb_s16` | `t1_bb_s16` | -0.01451 [-0.02852, -0.00249] * | yes | -0.00235 [-0.01285, +0.00998] | no | 10+10 | holds | not met |
| `t10_st_s1` | `t1_st_s1` | +0.00721 [-0.00705, +0.02204] | no | -0.00681 [-0.02749, +0.01108] | no | 10+10 | holds | not met |
| `t10_st_s16` | `t1_st_s16` | -0.00381 [-0.01302, +0.00667] | no | -0.01358 [-0.02552, +0.00059] | no | 10+10 | holds | not met |

  Part (a) holds at both q only for `relu_st_s16`; part (b) is violated in all four `relu` rows (runs that pass G-A under `t1` fail it under `relu`) and holds in all four `t10` rows.
- **`relu`: the typical run has the smaller tie deficit, a few runs fail.** Median gap 0.84-2.36 effort units against 2.11-3.77 for `t1` (lower in all eight cells); 3-6 of 10 runs per q = 60 arm have a gap of at most 1 effort unit (`t1`: 0-1); the smoothing part, sigma_2(0) and the tail mean are lower in all eight cells. Five of the 40 `relu` runs at q = 50 fail G-A (one run lost its tie effort after 825 updates, effort 1e-4 at the tie; one run has a dead region in the middle stratum), none of the 40 at q = 60, none of the 160 `t1` and `t10` runs; the mean-based criterion is dominated by that collapsed run in the two bin-balanced q = 50 rows (+0.0691).
- **`t10`: no change of the tie deficit.** The sharpest first-layer unit bends over 24-30 units of d (`t1`: 159-173), the rounding width is 3.4-6.6 units of d (`t1`: 4.0-5.7); mean |peak error| within 0.015 of `t1`'s in every arm. The tail mean is lower in all eight cells. The supervised screen's advantage of `t10` (0.34-0.53 against 0.71-6.32) does not appear in the RL runs.
- **The noise landing** lowers sigma_2(0) and the smoothing part under every actor. Transmission of the smoothing reduction to the gap: `t1` -0.43 to +0.28 (as in MS-R2), `t10` 0.11, 0.60, 0.72, 0.69 (two intervals exclude 0), `relu` 0.06, 1.53, 0.14, 0.33 (none excludes 0). No arm reaches the smoothing floor at s = 16 (|peak| / floor 3.1-7.4 for `relu` and `t10`, 5.6-7.6 for `t1`). The quadrature model is closer to the observed gap(s16) than the additive one in 10 of 12 cells.
- **R0** (the closed-form-free tie residual) ranks |peak error| at the freeze with Spearman 1.000 for `t1` and `t10` and 0.993 / 0.970 for `relu`.
- **The RL gap is larger than the supervised screen's tip deficit** in 11 of 12 cells (factors 2.1-59).

## What this does not show

Ten seeds per cell: an interval that contains 0 does not show the absence of an effect, and one failed run moves a mean of ten. The mechanism of the two `relu` failures and the reason why `t10` does not transfer from the screen are not established (14-28 of 64 first-layer `relu` units are never active on D_2 in good and failed runs alike). Nothing here selects an actor or proposes a protocol.

## Deviations and recorded departures

1. **The review's minor findings** (P1): six, fixed or answered before the code commit; the code that ran is the code commit `b41ecefd` (run code unchanged since `0a9698b6`), `03_checks.md` section 2.
2. **Order at the start:** the prompt was saved as `pi_record/20_ms_r3_prompt.md` right after the branch was created and the PI's sandbox files (read from `origin/main`, hashes as in the prompt) were copied into `pi_record/`, not before them.
3. **Report supplements not in D5 or D6** (labelled in the reports): the median-based `primary_median`, `robust` and `failures` tables, the hidden-unit counts of the `relu` actors (`relu_units.py`), the per-run tie-profile figure (`tie_profile_runs.py`), and the blocks `directions`, `side_by_side`, `tie_effort`, `screen_vs_rl`. They add no test to the pre-registered ones.
4. **"Every weight export"** (D5) is all 136 exports of a run (`first_layer_weights.csv`, with a `stage` column); the summaries use the terminal stage (`02_preregistration.md` decision 13).
5. **Local refs** of the primary checkout's `main` and of the MS-R1 P1 worktree's `ms-r1` are behind their `origin` refs as before (owner-side, `00_housekeeping.md`); not touched.

## Commands

```
# analysis (written to results/ms_r3/analysis/; blind recomputation: tools/ms/r3_blind_criterion.py --analysis-dir <out>)
python tools/ms/r3_analysis.py --pilot-root results/ms_r3/pilot --ms-r2-pilot-root <MS-R2 pilot root> \
  --parents-root <v2-t2-refine>/results/v2_refine/parents_A --rehearsal-root <v2-t2-refine>/results/v2_T2_locked/rehearsal_v2_0 \
  --calibration-root results/ms_r1/calibration --actors t1 relu t10 --out results/ms_r3/analysis
# tables
python reports/ms/r3/report_scripts/pilot_tables.py --list
python reports/ms/r3/report_scripts/screen_tables.py --block <block>
python reports/ms/r3/report_scripts/preamble_numbers.py --block <block>
python reports/ms/r3/report_scripts/relu_units.py --block summary
```
