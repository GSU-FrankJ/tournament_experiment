# MS-R1 P1c: offline calibration of the development rule on the v2.0 weight exports

Date: 2026-10-07. Spec: `reports/ms/r1/pi_record/17_ms_r1_prompt.md` §2.4 and D8. No training was run for this report. Tool: `tools/ms/replay_dev_rule.py` (`replay`, then `analyze`; tests `tests/test_ms_replay.py`); data: `results/ms_r1/calibration/` (one CSV per run, `manifest.json`, `validation_summary.csv`, `facts.json`, `tables/*.csv`); figures `reports/ms/r1/figures/cal_fig1..5_*.png`; one extra check `tools/ms/residual_asymmetry_example.py` (output `results/ms_r1/calibration/asymmetry_example.txt`).

## Method and validation

The development verifier (`utils.dp_br_verifier.DEV_CONFIG`) was replayed on every stored weight export of the 20 `rehearsal_v2_0` and the 40 `confirmation_v2_0` runs (88 exports per run, u0025..u2200 every 25; the roots are in `results/ms_r1/calibration/manifest.json`: `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/{rehearsal_v2_0,confirmation_v2_0}`). Phase-A exports are evaluated as the candidate "this network at every stage" (v2.0's Phase A has no frozen stage) with the D3 stage t = 2; Phase-B exports (u1625..u2200) as the export at t = 1 with the run's u1600 export frozen at t = 2, D3 stage t = 1. The exports are loaded into a `BetaActor` (float32, bit-identical forward pass) and evaluated with `utils.v2_metrics.evaluate`; the D3 quantities come from `utils.ms_residual.stage_diag`, the rule logic from `utils.ms_rule` (`is_eligible`, `classify_block_end`, `ema_update`), the closed-form errors from `evaluate` (reporting only). The final tier is computed at Phase-A u1600 only (Table 3).

**Validation of the replay.** At every logged verifier call of the 60 runs that has an export, 12 fields of the replay (`eta_T_over_dw`, signed and absolute peak error, RMSE_pos, tail mean and max, `e1_at_0`, `e2_at_0`, `stage1_rel_err_signed`, `Gmax_full_over_dw`, `EXP_root_over_dw`, `conc_max_std_norm`) were compared with the run's own `v2_checkpoints_{A,B}.csv`: 26,484 comparisons, all bit for bit equal (maximum difference 0.0); the u1600 gate (`eta`, peak, RMSE) is met by 60 of 60 runs (`results/ms_r1/calibration/validation_summary.csv`). A second replay with 4 instead of 8 workers reproduced all 368,460 non-timing cells exactly; a re-run of `analyze` reproduces the tables byte for byte. Three runs have logged calls at updates without an export (rehearsal q50 seed 10506, rehearsal q60 seed 10506, confirmation q60 seed 30505); those calls were not replayed, the u1600 call exists in every run.

## 0. The three facts of the preamble

(`results/ms_r1/calibration/facts.json`, `tables/fact2_*.csv`)

**Fact 1: confirmed.** The original rule is in `run/run_v2_stagewise.py` (line 833 `phase_thr_over_dw`, 827 `conc_thr`, 886 `k_phase`; line 889 `if not self.fixed:` guards the exit; line 324 `self.fixed`); all 60 `run_config.json` carry `fixed_budget` true, `verifier_timeout.A` 100, `phase_thr_over_dw.A` 0.02, `k_phase` 3, `conc_thr` 0.04, phase cap A 1600. The Phase-A calls are 100 updates apart in 897 of 903 gaps (in 4 runs a `reason=stability` call shifted the later calls). `verifier_sec` over the 963 Phase-A calls: median 9.0 ms, mean 9.75 ms. The locked Phase-A threshold is 0.02; the 0.005 of the prompt is a replay variant.

**Fact 2: confirmed in substance, one stated bound is missed by 50 updates.** The original rule (3 consecutive eligible calls, threshold 0.02 or 0.005 on `max_D2dev_Delta2_over_dw`, concentration <= 0.04) replayed on the 60 runs' `v2_checkpoints_A.csv` (calls every 100 updates):

**Fact 2. The original rule (3 consecutive eligible calls, calls of the v2.0 log every 100 updates) replayed on the runs' own v2_checkpoints_A.csv**

| threshold | cell | fired | fire local update med [min,max] | |peak error| at the firing call | |peak error| at u1600 (fired runs) | tail mean at the firing call | runs with tail mean > 0.02 | runs with |peak| at fire > |peak| at u1600 |
|---|---|---|---|---|---|---|---|---|
| 0.02 | pooled | 60/60 | 600 [500, 1300] | 0.0759 [0.0039, 0.1248] | 0.0624 (0.0624 all) | 0.0153 [0.0083, 0.0235] | 7 | 43/60 |
| 0.02 | q50 | 30/30 | 700 [600, 1300] | 0.0753 [0.0039, 0.1248] | 0.0615 (0.0615 all) | 0.0146 [0.0083, 0.0235] | 1 | 21/30 |
| 0.02 | q60 | 30/30 | 600 [500, 1100] | 0.0776 [0.0320, 0.1214] | 0.0648 (0.0648 all) | 0.0170 [0.0096, 0.0228] | 6 | 22/30 |
| 0.02 | rehearsal_v2_0/q50 | 10/10 | 650 [600, 1100] | 0.0839 [0.0447, 0.1166] | 0.0625 (0.0625 all) | 0.0138 [0.0083, 0.0235] | 1 | 7/10 |
| 0.02 | rehearsal_v2_0/q60 | 10/10 | 600 [600, 900] | 0.0600 [0.0320, 0.1142] | 0.0497 (0.0497 all) | 0.0165 [0.0096, 0.0220] | 1 | 7/10 |
| 0.02 | confirmation_v2_0/q50 | 20/20 | 700 [600, 1300] | 0.0719 [0.0039, 0.1248] | 0.0615 (0.0615 all) | 0.0146 [0.0087, 0.0191] | 0 | 14/20 |
| 0.02 | confirmation_v2_0/q60 | 20/20 | 600 [500, 1100] | 0.0861 [0.0390, 0.1214] | 0.0678 (0.0678 all) | 0.0182 [0.0105, 0.0228] | 5 | 15/20 |
| 0.005 | pooled | 60/60 | 700 [500, 1500] | 0.0760 [0.0320, 0.1214] | 0.0624 (0.0624 all) | 0.0151 [0.0073, 0.0228] | 6 | 42/60 |
| 0.005 | q50 | 30/30 | 800 [600, 1500] | 0.0754 [0.0327, 0.1180] | 0.0615 (0.0615 all) | 0.0127 [0.0073, 0.0191] | 0 | 20/30 |
| 0.005 | q60 | 30/30 | 600 [500, 1100] | 0.0776 [0.0320, 0.1214] | 0.0648 (0.0648 all) | 0.0170 [0.0096, 0.0228] | 6 | 22/30 |
| 0.005 | rehearsal_v2_0/q50 | 10/10 | 800 [600, 1100] | 0.0841 [0.0487, 0.1115] | 0.0625 (0.0625 all) | 0.0125 [0.0073, 0.0169] | 0 | 6/10 |
| 0.005 | rehearsal_v2_0/q60 | 10/10 | 600 [600, 1100] | 0.0600 [0.0320, 0.0987] | 0.0497 (0.0497 all) | 0.0165 [0.0096, 0.0220] | 1 | 7/10 |
| 0.005 | confirmation_v2_0/q50 | 20/20 | 700 [600, 1500] | 0.0730 [0.0327, 0.1180] | 0.0615 (0.0615 all) | 0.0135 [0.0078, 0.0191] | 0 | 14/20 |
| 0.005 | confirmation_v2_0/q60 | 20/20 | 600 [500, 1100] | 0.0861 [0.0390, 0.1214] | 0.0678 (0.0678 all) | 0.0182 [0.0105, 0.0228] | 5 | 15/20 |

Against the preamble: median firing update 600-750 (obtained: 600 to 700 pooled, q50 700 and 800, q60 600; the q50 cell at threshold 0.005, 800, is 50 above the stated range); median |peak error| at the firing call 0.070-0.078 (obtained 0.0753 to 0.0776, inside) against 0.056-0.067 at u1600 (obtained 0.0615 / 0.0648 / 0.0624, inside); tail mean at the firing call reaching 0.0228-0.0235 (obtained 0.0235 at q50 and 0.0228 at q60, exactly). All 60 runs fire; the firing call has the larger |peak error| than u1600 in 43 of 60 runs (threshold 0.02). Checks of the replay itself: the eligibility at 0.02 equals the logged column in 963 of 963 calls; the fire update equals `would_have_fired.A` in `v2_run_summary.json` in 60 of 60 runs; the same rule on the exports at K = 100 equals the log in 120 of 120 (run, threshold) pairs.

**Fact 3: confirmed.** `confirmation_v2_0/q50/seed30501` at u1600 (`facts.json`, key `fact3`; recomputed independently from `F_xi`, the verifier at both tiers agrees to 1e-15): e_hat_2(0) = 66.4463 (prompt 66.45), e2*(0) = 70, relative error -5.077 % (-5.1 %), one-step deviation gain at d = 0 5.3061e-4 DW (5.3e-4), one-step best-response effort 68.5367 (68.54), best response minus policy 2.0904 effort units = 2.986 % of e2*(0) (2.09, 3.0 %), the gain is 0.106 of the G-A limit 0.005 ("a tenth").

## 1. Spearman correlation of the closed-form errors with the residuals (u >= 400)

**Table 1. Spearman correlation of the closed-form errors with the residuals (Phase-A exports, dev tier; bold = the pair asked for in 2.4 item 1; R0 = r_2(0)/s_2 at the node d = 0 and R_near = max of rho_b over the four near-tie bins are supplementary, not rule quantities)**

| subset | group | x vs y | n pairs | Spearman (pooled) | within-run median [min, max] |
|---|---|---|---|---|---|
| **u>=400** | pooled | |peak| vs R | 2940 | **0.179** | 0.056 [-0.460, 0.645] |
| **u>=400** | pooled | |peak| vs Delta | 2940 | **0.413** | 0.419 [-0.227, 0.772] |
| u>=400 | pooled | |peak| vs R0 | 2940 | 0.985 | 1.000 [0.928, 1.000] |
| u>=400 | pooled | |peak| vs R_near | 2940 | 0.224 | 0.130 [-0.542, 0.760] |
| u>=400 | pooled | RMSE_pos vs R | 2940 | 0.727 | 0.793 [0.530, 0.923] |
| u>=400 | pooled | RMSE_pos vs Delta | 2940 | 0.784 | 0.795 [0.612, 0.925] |
| u>=400 | pooled | tail mean vs R_tail | 2940 | 0.998 | 0.997 [0.992, 0.999] |
| u>=400 | pooled | tail mean vs Delta | 2940 | 0.106 | 0.362 [-0.305, 0.734] |
| u>=400 | pooled | R vs Delta | 2940 | 0.906 | 0.888 [0.344, 0.948] |
| **u>=400** | q50 | |peak| vs R | 1470 | **0.056** | -0.062 [-0.460, 0.266] |
| **u>=400** | q50 | |peak| vs Delta | 1470 | **0.353** | 0.255 [-0.227, 0.665] |
| u>=400 | q50 | |peak| vs R0 | 1470 | 0.996 | 1.000 [0.928, 1.000] |
| u>=400 | q50 | |peak| vs R_near | 1470 | 0.040 | -0.105 [-0.542, 0.759] |
| u>=400 | q50 | RMSE_pos vs R | 1470 | 0.747 | 0.712 [0.530, 0.857] |
| u>=400 | q50 | RMSE_pos vs Delta | 1470 | 0.813 | 0.782 [0.612, 0.925] |
| u>=400 | q50 | tail mean vs R_tail | 1470 | 0.998 | 0.996 [0.992, 0.999] |
| u>=400 | q50 | tail mean vs Delta | 1470 | 0.248 | 0.294 [-0.305, 0.678] |
| u>=400 | q50 | R vs Delta | 1470 | 0.867 | 0.831 [0.344, 0.940] |
| **u>=400** | q60 | |peak| vs R | 1470 | **0.350** | 0.309 [-0.193, 0.645] |
| **u>=400** | q60 | |peak| vs Delta | 1470 | **0.544** | 0.532 [-0.007, 0.772] |
| u>=400 | q60 | |peak| vs R0 | 1470 | 0.998 | 1.000 [0.949, 1.000] |
| u>=400 | q60 | |peak| vs R_near | 1470 | 0.403 | 0.391 [-0.224, 0.760] |
| u>=400 | q60 | RMSE_pos vs R | 1470 | 0.851 | 0.835 [0.688, 0.923] |
| u>=400 | q60 | RMSE_pos vs Delta | 1470 | 0.840 | 0.812 [0.617, 0.903] |
| u>=400 | q60 | tail mean vs R_tail | 1470 | 0.999 | 0.998 [0.994, 0.999] |
| u>=400 | q60 | tail mean vs Delta | 1470 | 0.300 | 0.419 [0.066, 0.734] |
| u>=400 | q60 | R vs Delta | 1470 | 0.916 | 0.900 [0.809, 0.948] |
| 400<=u<=1200 (constant LR) | pooled | |peak| vs R | 1980 | 0.171 | 0.107 [-0.496, 0.713] |
| 400<=u<=1200 (constant LR) | pooled | |peak| vs Delta | 1980 | 0.432 | 0.391 [-0.337, 0.871] |
| 400<=u<=1200 (constant LR) | pooled | |peak| vs R0 | 1980 | 0.987 | 1.000 [0.949, 1.000] |
| 400<=u<=1200 (constant LR) | pooled | |peak| vs R_near | 1980 | 0.251 | 0.185 [-0.419, 0.961] |
| 400<=u<=1200 (constant LR) | pooled | RMSE_pos vs R | 1980 | 0.707 | 0.766 [0.492, 0.924] |
| 400<=u<=1200 (constant LR) | pooled | RMSE_pos vs Delta | 1980 | 0.769 | 0.783 [0.543, 0.910] |
| 400<=u<=1200 (constant LR) | pooled | tail mean vs R_tail | 1980 | 0.998 | 0.997 [0.985, 0.999] |
| 400<=u<=1200 (constant LR) | pooled | tail mean vs Delta | 1980 | 0.034 | 0.259 [-0.275, 0.704] |
| 400<=u<=1200 (constant LR) | pooled | R vs Delta | 1980 | 0.893 | 0.861 [0.273, 0.963] |
| 400<=u<=1200 (constant LR) | q50 | |peak| vs R | 990 | 0.046 | -0.054 [-0.496, 0.292] |
| 400<=u<=1200 (constant LR) | q50 | |peak| vs Delta | 990 | 0.380 | 0.296 [-0.337, 0.718] |
| 400<=u<=1200 (constant LR) | q50 | |peak| vs R0 | 990 | 1.000 | 1.000 [0.999, 1.000] |
| 400<=u<=1200 (constant LR) | q50 | |peak| vs R_near | 990 | 0.081 | -0.075 [-0.419, 0.764] |
| 400<=u<=1200 (constant LR) | q50 | RMSE_pos vs R | 990 | 0.732 | 0.679 [0.492, 0.878] |
| 400<=u<=1200 (constant LR) | q50 | RMSE_pos vs Delta | 990 | 0.810 | 0.776 [0.543, 0.908] |
| 400<=u<=1200 (constant LR) | q50 | tail mean vs R_tail | 990 | 0.998 | 0.997 [0.991, 0.999] |
| 400<=u<=1200 (constant LR) | q50 | tail mean vs Delta | 990 | 0.169 | 0.243 [-0.275, 0.704] |
| 400<=u<=1200 (constant LR) | q50 | R vs Delta | 990 | 0.841 | 0.807 [0.273, 0.940] |
| 400<=u<=1200 (constant LR) | q60 | |peak| vs R | 990 | 0.345 | 0.368 [-0.249, 0.713] |
| 400<=u<=1200 (constant LR) | q60 | |peak| vs Delta | 990 | 0.558 | 0.602 [-0.038, 0.871] |
| 400<=u<=1200 (constant LR) | q60 | |peak| vs R0 | 990 | 0.998 | 1.000 [0.949, 1.000] |
| 400<=u<=1200 (constant LR) | q60 | |peak| vs R_near | 990 | 0.411 | 0.418 [-0.259, 0.961] |
| 400<=u<=1200 (constant LR) | q60 | RMSE_pos vs R | 990 | 0.853 | 0.842 [0.596, 0.924] |
| 400<=u<=1200 (constant LR) | q60 | RMSE_pos vs Delta | 990 | 0.835 | 0.786 [0.655, 0.910] |
| 400<=u<=1200 (constant LR) | q60 | tail mean vs R_tail | 990 | 0.998 | 0.997 [0.985, 0.999] |
| 400<=u<=1200 (constant LR) | q60 | tail mean vs Delta | 990 | 0.180 | 0.276 [-0.013, 0.648] |
| 400<=u<=1200 (constant LR) | q60 | R vs Delta | 990 | 0.908 | 0.887 [0.793, 0.963] |

Supplementary (not rule quantities): `R0 = r_2(0)/s_2`, the residual at the single node d = 0.

**Table 1b (supplementary). |peak error| / R0 with R0 = r_2(0)/s_2, Phase-A exports u >= 400**

| group | n | |peak|/R0 q10 / q50 / q90 | R0 q10 / q50 / q90 | R0 where |peak| = 0.05 (median ratio) | exports with R0 <= 0.03 | exports with |peak| <= 0.05 |
|---|---|---|---|---|---|---|
| pooled | 2940 | 1.440 / 1.582 / 1.666 | 0.022 / 0.044 / 0.075 | 0.0316 | 0.201 | 0.238 |
| q50 | 1470 | 1.617 / 1.651 / 1.674 | 0.022 / 0.042 / 0.072 | 0.0303 | 0.224 | 0.226 |
| q60 | 1470 | 1.433 / 1.453 / 1.469 | 0.024 / 0.046 / 0.077 | 0.0344 | 0.178 | 0.250 |

Reading. Pooled over 2940 (run, export) pairs the rank correlation of |peak error| with R is 0.179 (q50 0.056, q60 0.350) and with Delta 0.413 (q50 0.353, q60 0.544): **R, the maximum over the whole non-tail region, is not a good rank predictor of the peak error**, because its maximum sits elsewhere than at d = 0 (Table 4). The residual at the node d = 0 is: Spearman 0.985 with |peak error| (within-run median 1.000); |peak|/R0 has median 1.58 (q50 1.65, q60 1.45), so |peak| = 0.05 corresponds to R0 of about 0.030 (q50) and 0.034 (q60). R against RMSE_pos 0.727, R_tail against the tail mean 0.998. The constant-LR subset (400 <= u <= 1200) gives the same picture. Figure: `reports/ms/r1/figures/cal_fig1_scatter_R_Delta_vs_peak.png`.

## 2. Candidate rules: when and how well they fire on the v2.0 trajectories

For every candidate rule of the grid epsilon = 0.005 x rho in {0.02, 0.03, 0.05} x tau = 0.02 x M = 3 x K = 25 (the check at export u is local update u), and the original second-order rule (epsilon in {0.02, 0.005}, no first-order part) at K = 25; the K = 100 rows are the preamble's cadence. Closed-form errors at the fire export against the same runs at u1600.

**Table 2. Fire export of every candidate rule (M = 3, K = 25, tau = 0.02, eps = 0.005 for the new rules; 'orig' = second-order only), closed-form errors at the fire export against u1600 (medians over the runs that fire)**

| rule | group | fired | never | fire update med [min,max] | |peak| at fire | |peak| u1600 (same runs) | RMSE_pos at fire | RMSE_pos u1600 | tail mean at fire | tail mean u1600 | fire with tail<=0.02 & RMSE<=0.05 | that share of ALL runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| orig eps=0.02 | pooled | 60/60 | 0 | 450 [325, 1100] | 0.0896 [0.0354, 0.1637] | 0.0624 | 0.0323 [0.0223, 0.0581] | 0.0229 | 0.0194 [0.0094, 0.0343] | 0.0091 | 31/60 | 0.517 |
| orig eps=0.02 | q50 | 30/30 | 0 | 512 [375, 1100] | 0.0890 [0.0354, 0.1637] | 0.0615 | 0.0361 [0.0228, 0.0581] | 0.0232 | 0.0178 [0.0094, 0.0300] | 0.0080 | 18/30 | 0.600 |
| orig eps=0.02 | q60 | 30/30 | 0 | 450 [325, 875] | 0.0905 [0.0457, 0.1214] | 0.0648 | 0.0310 [0.0223, 0.0463] | 0.0221 | 0.0219 [0.0111, 0.0343] | 0.0102 | 13/30 | 0.433 |
| orig eps=0.005 | pooled | 60/60 | 0 | 500 [325, 1525] | 0.0874 [0.0354, 0.1430] | 0.0624 | 0.0306 [0.0223, 0.0549] | 0.0229 | 0.0194 [0.0078, 0.0338] | 0.0091 | 31/60 | 0.517 |
| orig eps=0.005 | q50 | 30/30 | 0 | 525 [400, 1525] | 0.0836 [0.0354, 0.1331] | 0.0615 | 0.0314 [0.0228, 0.0549] | 0.0232 | 0.0165 [0.0078, 0.0264] | 0.0080 | 18/30 | 0.600 |
| orig eps=0.005 | q60 | 30/30 | 0 | 450 [325, 900] | 0.0905 [0.0457, 0.1430] | 0.0648 | 0.0301 [0.0223, 0.0481] | 0.0221 | 0.0219 [0.0111, 0.0338] | 0.0102 | 13/30 | 0.433 |
| new rho=0.02 | pooled | 0/60 | 60 | - | - | - | - | - | - | - | - | 0.000 |
| new rho=0.02 | q50 | 0/30 | 30 | - | - | - | - | - | - | - | - | 0.000 |
| new rho=0.02 | q60 | 0/30 | 30 | - | - | - | - | - | - | - | - | 0.000 |
| new rho=0.03 | pooled | 0/60 | 60 | - | - | - | - | - | - | - | - | 0.000 |
| new rho=0.03 | q50 | 0/30 | 30 | - | - | - | - | - | - | - | - | 0.000 |
| new rho=0.03 | q60 | 0/30 | 30 | - | - | - | - | - | - | - | - | 0.000 |
| new rho=0.05 | pooled | 1/60 | 59 | 1500 [1500, 1500] | 0.0481 [0.0481, 0.0481] | 0.0469 | 0.0170 [0.0170, 0.0170] | 0.0133 | 0.0084 [0.0084, 0.0084] | 0.0083 | 1/1 | 0.017 |
| new rho=0.05 | q50 | 0/30 | 30 | - | - | - | - | - | - | - | - | 0.000 |
| new rho=0.05 | q60 | 1/30 | 29 | 1500 [1500, 1500] | 0.0481 [0.0481, 0.0481] | 0.0469 | 0.0170 [0.0170, 0.0170] | 0.0133 | 0.0084 [0.0084, 0.0084] | 0.0083 | 1/1 | 0.033 |
| orig eps=0.02 K=100 | pooled | 60/60 | 0 | 600 [500, 1300] | 0.0759 [0.0039, 0.1248] | 0.0624 | 0.0293 [0.0173, 0.0527] | 0.0229 | 0.0153 [0.0083, 0.0235] | 0.0091 | 52/60 | 0.867 |
| orig eps=0.02 K=100 | q50 | 30/30 | 0 | 700 [600, 1300] | 0.0753 [0.0039, 0.1248] | 0.0615 | 0.0304 [0.0175, 0.0527] | 0.0232 | 0.0146 [0.0083, 0.0235] | 0.0080 | 28/30 | 0.933 |
| orig eps=0.02 K=100 | q60 | 30/30 | 0 | 600 [500, 1100] | 0.0776 [0.0320, 0.1214] | 0.0648 | 0.0285 [0.0173, 0.0399] | 0.0221 | 0.0170 [0.0096, 0.0228] | 0.0102 | 24/30 | 0.800 |
| orig eps=0.005 K=100 | pooled | 60/60 | 0 | 700 [500, 1500] | 0.0760 [0.0320, 0.1214] | 0.0624 | 0.0283 [0.0173, 0.0497] | 0.0229 | 0.0151 [0.0073, 0.0228] | 0.0091 | 54/60 | 0.900 |
| orig eps=0.005 K=100 | q50 | 30/30 | 0 | 800 [600, 1500] | 0.0754 [0.0327, 0.1180] | 0.0615 | 0.0280 [0.0175, 0.0497] | 0.0232 | 0.0127 [0.0073, 0.0191] | 0.0080 | 30/30 | 1.000 |
| orig eps=0.005 K=100 | q60 | 30/30 | 0 | 600 [500, 1100] | 0.0776 [0.0320, 0.1214] | 0.0648 | 0.0283 [0.0173, 0.0399] | 0.0221 | 0.0170 [0.0096, 0.0228] | 0.0102 | 24/30 | 0.800 |

Supplementary: a wider rho range (new rule, epsilon = 0.005, tau = 0.02, M = 3, K = 25).

**Table 2c (supplementary). The new rule for a wider rho range (eps = 0.005, tau = 0.02, M = 3, K = 25): how often and when it fires on the v2.0 trajectories**

| rho | group | fired | fire update med [min,max] | |peak| at fire | |peak| u1600 (same runs) | RMSE_pos at fire | tail mean at fire | fire with tail<=0.02 & RMSE<=0.05 | fires after u1200 (inside v2.0's LR decay) |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | pooled | 0/60 | - | - | - | - | - | - | 0 |
| 0.02 | q50 | 0/30 | - | - | - | - | - | - | 0 |
| 0.02 | q60 | 0/30 | - | - | - | - | - | - | 0 |
| 0.03 | pooled | 0/60 | - | - | - | - | - | - | 0 |
| 0.03 | q50 | 0/30 | - | - | - | - | - | - | 0 |
| 0.03 | q60 | 0/30 | - | - | - | - | - | - | 0 |
| 0.05 | pooled | 1/60 | 1500 [1500, 1500] | 0.0481 [0.0481, 0.0481] | 0.0469 | 0.0170 [0.0170, 0.0170] | 0.0084 [0.0084, 0.0084] | 1/1 | 1 |
| 0.05 | q50 | 0/30 | - | - | - | - | - | - | 0 |
| 0.05 | q60 | 1/30 | 1500 [1500, 1500] | 0.0481 [0.0481, 0.0481] | 0.0469 | 0.0170 [0.0170, 0.0170] | 0.0084 [0.0084, 0.0084] | 1/1 | 1 |
| 0.06 | pooled | 9/60 | 1350 [925, 1600] | 0.0493 [0.0429, 0.0589] | 0.0499 | 0.0187 [0.0170, 0.0221] | 0.0125 [0.0084, 0.0161] | 9/9 | 6 |
| 0.06 | q50 | 0/30 | - | - | - | - | - | - | 0 |
| 0.06 | q60 | 9/30 | 1350 [925, 1600] | 0.0493 [0.0429, 0.0589] | 0.0499 | 0.0187 [0.0170, 0.0221] | 0.0125 [0.0084, 0.0161] | 9/9 | 6 |
| 0.08 | pooled | 35/60 | 1125 [625, 1600] | 0.0613 [0.0344, 0.0867] | 0.0628 | 0.0225 [0.0127, 0.0330] | 0.0116 [0.0058, 0.0172] | 35/35 | 15 |
| 0.08 | q50 | 6/30 | 1400 [1225, 1600] | 0.0655 [0.0519, 0.0731] | 0.0615 | 0.0212 [0.0127, 0.0241] | 0.0071 [0.0058, 0.0119] | 6/6 | 6 |
| 0.08 | q60 | 29/30 | 1000 [625, 1425] | 0.0596 [0.0344, 0.0867] | 0.0628 | 0.0225 [0.0176, 0.0330] | 0.0119 [0.0075, 0.0172] | 29/29 | 9 |
| 0.1 | pooled | 44/60 | 750 [575, 1600] | 0.0661 [0.0143, 0.1040] | 0.0616 | 0.0248 [0.0158, 0.0396] | 0.0137 [0.0067, 0.0187] | 44/44 | 8 |
| 0.1 | q50 | 15/30 | 1225 [600, 1600] | 0.0694 [0.0345, 0.0813] | 0.0578 | 0.0230 [0.0158, 0.0359] | 0.0095 [0.0067, 0.0135] | 15/15 | 8 |
| 0.1 | q60 | 29/30 | 750 [575, 975] | 0.0605 [0.0143, 0.1040] | 0.0628 | 0.0253 [0.0191, 0.0396] | 0.0150 [0.0103, 0.0187] | 29/29 | 0 |
| 0.12 | pooled | 57/60 | 725 [550, 1400] | 0.0694 [0.0175, 0.1107] | 0.0620 | 0.0246 [0.0157, 0.0391] | 0.0139 [0.0073, 0.0187] | 57/57 | 4 |
| 0.12 | q50 | 27/30 | 800 [550, 1400] | 0.0667 [0.0461, 0.1063] | 0.0611 | 0.0230 [0.0164, 0.0343] | 0.0118 [0.0073, 0.0165] | 27/27 | 4 |
| 0.12 | q60 | 30/30 | 675 [550, 1075] | 0.0728 [0.0175, 0.1107] | 0.0648 | 0.0257 [0.0157, 0.0391] | 0.0161 [0.0109, 0.0187] | 30/30 | 0 |
| 0.15 | pooled | 59/60 | 650 [525, 1550] | 0.0718 [0.0175, 0.1253] | 0.0620 | 0.0272 [0.0157, 0.0427] | 0.0157 [0.0061, 0.0188] | 59/59 | 1 |
| 0.15 | q50 | 29/30 | 650 [525, 1550] | 0.0740 [0.0354, 0.1129] | 0.0611 | 0.0272 [0.0170, 0.0378] | 0.0142 [0.0061, 0.0181] | 29/29 | 1 |
| 0.15 | q60 | 30/30 | 650 [550, 975] | 0.0695 [0.0175, 0.1253] | 0.0648 | 0.0276 [0.0157, 0.0427] | 0.0172 [0.0122, 0.0188] | 30/30 | 0 |
| 0.2 | pooled | 60/60 | 625 [525, 1525] | 0.0774 [0.0175, 0.1430] | 0.0624 | 0.0292 [0.0157, 0.0509] | 0.0162 [0.0086, 0.0188] | 59/60 | 1 |
| 0.2 | q50 | 30/30 | 600 [525, 1525] | 0.0806 [0.0354, 0.1331] | 0.0615 | 0.0298 [0.0223, 0.0450] | 0.0155 [0.0086, 0.0181] | 30/30 | 1 |
| 0.2 | q60 | 30/30 | 625 [525, 900] | 0.0730 [0.0175, 0.1430] | 0.0648 | 0.0287 [0.0157, 0.0509] | 0.0172 [0.0111, 0.0188] | 29/30 | 0 |

Which component binds (fraction of Phase-A exports passing each eligibility component):

**Table 2b. Fraction of Phase-A exports passing each eligibility component (R_int = R without the two outermost non-tail bins; 'best R per run' = the minimum of R over the exports in scope, median [min, max] over runs)**

| scope | group | n | valid | Delta<=0.005 | R_tail<=0.02 | C<=0.04 | R<=0.02 | R<=0.03 | R<=0.05 | R_int<=0.02 | R_int<=0.03 | R_int<=0.05 | all (rho=0.02) | all (rho=0.03) | all (rho=0.05) | R q10 / q50 / q90 | best R per run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u>=400 | pooled | 2940 | 1.000 | 0.953 | 0.904 | 0.929 | 0.000 | 0.000 | 0.013 | 0.000 | 0.000 | 0.014 | 0.000 | 0.000 | 0.013 | 0.066 / 0.102 / 0.178 | 0.059 [0.039, 0.141] |
| u>=400 | q50 | 1470 | 1.000 | 0.914 | 0.923 | 0.915 | 0.000 | 0.000 | 6.80e-04 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 6.80e-04 | 0.081 / 0.129 / 0.200 | 0.067 [0.050, 0.141] |
| u>=400 | q60 | 1470 | 1.000 | 0.992 | 0.885 | 0.942 | 0.000 | 0.000 | 0.026 | 0.000 | 0.000 | 0.027 | 0.000 | 0.000 | 0.026 | 0.059 / 0.083 / 0.126 | 0.050 [0.039, 0.084] |
| u1600 only | pooled | 60 | 1.000 | 0.983 | 1.000 | 1.000 | 0.000 | 0.000 | 0.083 | 0.000 | 0.000 | 0.083 | 0.000 | 0.000 | 0.083 | 0.057 / 0.088 / 0.140 | 0.088 [0.044, 0.174] |
| u1600 only | q50 | 30 | 1.000 | 0.967 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.067 / 0.106 / 0.161 | 0.106 [0.063, 0.174] |
| u1600 only | q60 | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.167 | 0.000 | 0.000 | 0.167 | 0.000 | 0.000 | 0.167 | 0.046 / 0.077 / 0.094 | 0.077 [0.044, 0.134] |

Reading. **On the v2.0 trajectories the new rule is inert for every rho of the grid**: it fires in 0 of 60 runs at rho = 0.02 and 0.03 and in 1 of 60 at rho = 0.05 (q60, export u1500). R is the binding component: R <= 0.03 holds at 0 of 2940 exports, R <= 0.05 at 39 (1.3 %); the other components pass at 95.3 % (Delta <= 0.005), 90.4 % (R_tail <= 0.02) and 92.9 % (C <= 0.04). The best R per run (minimum over u >= 400) has median 0.0586 [0.0391, 0.1411]; no run reaches 0.03, 26.7 % reach 0.05 (q50 3.3 %, q60 50 %). The second-order rule alone, at the pipeline cadence K = 25, fires in 60 of 60 runs at median 450 / 500 (epsilon 0.02 / 0.005) with |peak error| 0.0896 / 0.0874 against 0.0624 at u1600 and tail mean up to 0.0343 (fact 2 again, at the finer cadence). The one rho = 0.05 fire and 6 of the 9 fires at rho = 0.06 lie after u1200, inside v2.0's own LR decay, which the MS pipeline does not use before a stop. The wider range shows what a stop at a higher rho would do: rho = 0.08 fires in 35 of 60 runs (median u1125) with |peak| 0.0613 against 0.0628 at u1600 in the same runs; rho = 0.10 fires in 44 of 60 (median u750) with 0.0661 against 0.0616, i.e. earlier and worse, the failure mode of fact 2. Figure: `cal_fig5_fire_exports.png`.

## 3. Development tier against final tier at u1600 (grid noise of the first-order residual)

**Table 3. Dev tier vs final tier at u1600 (state step 4 vs 2, effort step 1 vs 0.5, GL 16 vs 32 nodes per half): grid noise of the first-order residual**

| group | runs | |R_final - R_dev| med [min,max] | |R_tail diff| med [min,max] | max over bins |rho_final,b - rho_dev,b|: med [min,max] over runs | that bin is outermost / near-tie / other middle | median signed R diff | |Delta diff| med [min,max] | R<=rho flips dev->final (rho .02/.03/.05) |
|---|---|---|---|---|---|---|---|---|
| pooled | 60 | 0.0007 [6.38e-16, 0.0130] | 0.00023 [0.00015, 0.00043] | 0.0148 [0.0065, 0.0244] | 27/15/18 | 0.0007 | 0.00008 [0.00000, 0.00025] | 0/0/1 |
| q50 | 30 | 0.0005 [6.38e-16, 0.0130] | 0.00025 [0.00015, 0.00038] | 0.0155 [0.0077, 0.0244] | 1/13/16 | 0.0005 | 0.00005 [0.00000, 0.00022] | 0/0/0 |
| q60 | 30 | 0.0009 [5.33e-15, 0.0122] | 0.00021 [0.00018, 0.00043] | 0.0141 [0.0065, 0.0231] | 26/2/2 | 0.0009 | 0.00012 [0.00000, 0.00025] | 0/0/1 |

The final tier is computed for this table only. |R_final - R_dev| has median 0.0007 (maximum 0.0130); the worst per-bin difference has median 0.0148 over runs, and sits in the outermost non-tail bin in 27 of 60 runs (26 of 30 at q60). The verdict R <= rho flips between the tiers in 0, 0 and 1 runs for rho = 0.02, 0.03, 0.05. Separate evidence from the unit tests (`tests/test_ms_residual.py`): the development-tier best-response search is not exact (its parabola refinement does not reach a best response in (0, 1) below the first effort node and is off by up to 0.47 effort units at a kink between two nodes), so R carries a noise floor of up to about 0.0075 (q50) and 0.0088 (q60), a quarter of rho = 0.03.

## 4. Localized / broad classification and where the residual sits

At a fire export R <= rho holds by construction, so the pipeline classification there is always broad; the table therefore gives the classification at the block ends of N_block = 400 (u400, u800, u1200, u1600) for each rho, with loc_frac 0.25 (|S| <= 5 of 20 non-tail bins at q50, <= 6 of 24 at q60) and the EMA (beta 0.5, initialised at u0025). 'Near-tie' = bins intersecting (-20, 20), 'middle' = the other non-tail bins.

**Table 4. Localized / broad classification (loc_frac 0.25, EMA beta 0.5 from u0025) at the block ends of N_block = 400, at u1600 and at each rho's fire export; at the fire export R <= rho by construction, so the pipeline classification is 'broad' there**

| rho | where | group | n | localized / broad | |S| med [min,max] | |S|=0 | |S|>cap | |S| in 1..cap (R clause ignored) | S bins near-tie / middle (total) | share of near-tie / of middle bins in S | S bins d<0 / d>0 (total) | R argmax near-tie / middle / boundary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | u1600 | pooled | 60 | 0/60 | 13.0 [7, 22] | 0 | 60 | 0 | 192/573 | 0.80 / 0.53 | 596/169 | 36/24/0 |
| 0.02 | u1600 | q50 | 30 | 0/30 | 13.0 [7, 19] | 0 | 30 | 0 | 97/282 | 0.81 / 0.59 | 291/88 | 15/15/0 |
| 0.02 | u1600 | q60 | 30 | 0/30 | 13.0 [8, 22] | 0 | 30 | 0 | 95/291 | 0.79 / 0.48 | 305/81 | 21/9/0 |
| 0.02 | u400 | pooled | 60 | 0/60 | 17.5 [11, 24] | 0 | 60 | 0 | 215/867 | 0.90 / 0.80 | 646/436 | 40/19/1 |
| 0.02 | u400 | q50 | 30 | 0/30 | 17.0 [11, 20] | 0 | 30 | 0 | 108/384 | 0.90 / 0.80 | 300/192 | 15/14/1 |
| 0.02 | u400 | q60 | 30 | 0/30 | 20.0 [13, 24] | 0 | 30 | 0 | 107/483 | 0.89 / 0.81 | 346/244 | 25/5/0 |
| 0.02 | u800 | pooled | 60 | 0/60 | 15.0 [11, 23] | 0 | 60 | 0 | 210/705 | 0.88 / 0.65 | 621/294 | 25/35/0 |
| 0.02 | u800 | q50 | 30 | 0/30 | 14.0 [11, 18] | 0 | 30 | 0 | 103/323 | 0.86 / 0.67 | 298/128 | 9/21/0 |
| 0.02 | u800 | q60 | 30 | 0/30 | 16.0 [11, 23] | 0 | 30 | 0 | 107/382 | 0.89 / 0.64 | 323/166 | 16/14/0 |
| 0.02 | u1200 | pooled | 60 | 0/60 | 15.0 [11, 21] | 0 | 60 | 0 | 212/690 | 0.88 / 0.64 | 628/274 | 27/33/0 |
| 0.02 | u1200 | q50 | 30 | 0/30 | 14.0 [11, 19] | 0 | 30 | 0 | 106/328 | 0.88 / 0.68 | 299/135 | 12/18/0 |
| 0.02 | u1200 | q60 | 30 | 0/30 | 15.0 [11, 21] | 0 | 30 | 0 | 106/362 | 0.88 / 0.60 | 329/139 | 15/15/0 |
| 0.03 | u1600 | pooled | 60 | 11/49 | 9.0 [4, 17] | 0 | 49 | 11 | 159/371 | 0.66 / 0.34 | 469/61 | 36/24/0 |
| 0.03 | u1600 | q50 | 30 | 1/29 | 10.0 [5, 14] | 0 | 29 | 1 | 81/221 | 0.68 / 0.46 | 272/30 | 15/15/0 |
| 0.03 | u1600 | q60 | 30 | 10/20 | 7.0 [4, 17] | 0 | 20 | 10 | 78/150 | 0.65 / 0.25 | 197/31 | 21/9/0 |
| 0.03 | u400 | pooled | 60 | 1/59 | 14.0 [5, 22] | 0 | 59 | 1 | 189/624 | 0.79 / 0.58 | 605/208 | 40/19/1 |
| 0.03 | u400 | q50 | 30 | 0/30 | 13.0 [11, 16] | 0 | 30 | 0 | 94/298 | 0.78 / 0.62 | 300/92 | 15/14/1 |
| 0.03 | u400 | q60 | 30 | 1/29 | 15.0 [5, 22] | 0 | 29 | 1 | 95/326 | 0.79 / 0.54 | 305/116 | 25/5/0 |
| 0.03 | u800 | pooled | 60 | 0/60 | 11.0 [7, 17] | 0 | 60 | 0 | 184/487 | 0.77 / 0.45 | 541/130 | 25/35/0 |
| 0.03 | u800 | q50 | 30 | 0/30 | 11.0 [9, 16] | 0 | 30 | 0 | 92/261 | 0.77 / 0.54 | 292/61 | 9/21/0 |
| 0.03 | u800 | q60 | 30 | 0/30 | 10.0 [7, 17] | 0 | 30 | 0 | 92/226 | 0.77 / 0.38 | 249/69 | 16/14/0 |
| 0.03 | u1200 | pooled | 60 | 0/60 | 11.0 [7, 18] | 0 | 60 | 0 | 181/492 | 0.75 / 0.46 | 553/120 | 27/33/0 |
| 0.03 | u1200 | q50 | 30 | 0/30 | 11.0 [8, 15] | 0 | 30 | 0 | 90/251 | 0.75 / 0.52 | 290/51 | 12/18/0 |
| 0.03 | u1200 | q60 | 30 | 0/30 | 11.0 [7, 18] | 0 | 30 | 0 | 91/241 | 0.76 / 0.40 | 263/69 | 15/15/0 |
| 0.05 | u1600 | pooled | 60 | 37/23 | 4.0 [0, 11] | 3 | 17 | 40 | 102/151 | 0.42 / 0.14 | 238/15 | 36/24/0 |
| 0.05 | u1600 | q50 | 30 | 13/17 | 6.0 [2, 11] | 0 | 17 | 13 | 58/123 | 0.48 / 0.26 | 177/4 | 15/15/0 |
| 0.05 | u1600 | q60 | 30 | 24/6 | 2.0 [0, 6] | 3 | 0 | 27 | 44/28 | 0.37 / 0.05 | 61/11 | 21/9/0 |
| 0.05 | fire | pooled | 1 | 0/1 | 0.0 [0, 0] | 1 | 0 | 0 | 0/0 | 0.00 / 0.00 | 0/0 | 0/1/0 |
| 0.05 | fire | q60 | 1 | 0/1 | 0.0 [0, 0] | 1 | 0 | 0 | 0/0 | 0.00 / 0.00 | 0/0 | 0/1/0 |
| 0.05 | u400 | pooled | 60 | 5/55 | 10.0 [3, 14] | 0 | 55 | 5 | 173/401 | 0.72 / 0.37 | 504/70 | 40/19/1 |
| 0.05 | u400 | q50 | 30 | 0/30 | 11.0 [8, 14] | 0 | 30 | 0 | 89/232 | 0.74 / 0.48 | 284/37 | 15/14/1 |
| 0.05 | u400 | q60 | 30 | 5/25 | 8.5 [3, 14] | 0 | 25 | 5 | 84/169 | 0.70 / 0.28 | 220/33 | 25/5/0 |
| 0.05 | u800 | pooled | 60 | 25/35 | 7.0 [1, 13] | 0 | 35 | 25 | 130/266 | 0.54 / 0.25 | 370/26 | 25/35/0 |
| 0.05 | u800 | q50 | 30 | 2/28 | 9.0 [4, 11] | 0 | 28 | 2 | 69/188 | 0.57 / 0.39 | 246/11 | 9/21/0 |
| 0.05 | u800 | q60 | 30 | 23/7 | 4.0 [1, 13] | 0 | 7 | 23 | 61/78 | 0.51 / 0.13 | 124/15 | 16/14/0 |
| 0.05 | u1200 | pooled | 60 | 27/33 | 6.0 [1, 11] | 0 | 32 | 28 | 124/251 | 0.52 / 0.23 | 350/25 | 27/33/0 |
| 0.05 | u1200 | q50 | 30 | 3/27 | 8.0 [3, 11] | 0 | 27 | 3 | 63/173 | 0.53 / 0.36 | 228/8 | 12/18/0 |
| 0.05 | u1200 | q60 | 30 | 24/6 | 4.0 [1, 11] | 0 | 5 | 25 | 61/78 | 0.51 / 0.13 | 122/17 | 15/15/0 |

Where the maximum of R sits:

**Table 4 (where the maximum of R sits: near-tie |d|<20, boundary band |d| >= 2q-10, middle = the rest)**

| scope | group | n | near-tie | middle | boundary band | argmax d < 0 | median |argmax d|/2q | median R | median R_int |
|---|---|---|---|---|---|---|---|---|---|
| u>=400 | pooled | 2940 | 0.431 | 0.551 | 0.018 | 0.995 | 0.200 | 0.1020 | 0.1017 |
| u>=400 | q50 | 1470 | 0.329 | 0.654 | 0.017 | 0.999 | 0.240 | 0.1293 | 0.1293 |
| u>=400 | q60 | 1470 | 0.533 | 0.447 | 0.020 | 0.991 | 0.133 | 0.0835 | 0.0834 |
| u1600 | pooled | 60 | 0.600 | 0.400 | 0.000 | 1.000 | 0.160 | 0.0881 | 0.0881 |
| u1600 | q50 | 30 | 0.500 | 0.500 | 0.000 | 1.000 | 0.180 | 0.1060 | 0.1060 |
| u1600 | q60 | 30 | 0.700 | 0.300 | 0.000 | 1.000 | 0.033 | 0.0768 | 0.0768 |

Sensitivity of the classification to the localized fraction, to the EMA weight and the effect of v2.0's 400-update decay (descriptive; `tables/table4_*.csv`):

**Table 4d (descriptive). Runs classified 'localized' at the block ends u400 / u800 / u1200 / u1600 for other localized fractions (pipeline value 0.25)**

| rho | localized fraction | u400 | u800 | u1200 | u1600 |
|---|---|---|---|---|---|
| 0.02 | 0.15 | 0/60 | 0/60 | 0/60 | 0/60 |
| 0.02 | 0.25 | 0/60 | 0/60 | 0/60 | 0/60 |
| 0.02 | 0.35 | 0/60 | 0/60 | 0/60 | 5/60 |
| 0.02 | 0.5 | 0/60 | 3/60 | 4/60 | 16/60 |
| 0.03 | 0.15 | 0/60 | 0/60 | 0/60 | 2/60 |
| 0.03 | 0.25 | 1/60 | 0/60 | 0/60 | 11/60 |
| 0.03 | 0.35 | 2/60 | 11/60 | 11/60 | 27/60 |
| 0.03 | 0.5 | 9/60 | 29/60 | 27/60 | 46/60 |
| 0.05 | 0.15 | 2/60 | 16/60 | 18/60 | 27/60 |
| 0.05 | 0.25 | 5/60 | 25/60 | 27/60 | 37/60 |
| 0.05 | 0.35 | 19/60 | 35/60 | 42/60 | 46/60 |
| 0.05 | 0.5 | 40/60 | 54/60 | 57/60 | 53/60 |

**Table 4c (descriptive). Per-bin map, exports u >= 400: change of the EMA between consecutive exports (25 updates) and its gap to the raw map rho_b, by EMA weight beta (beta = 0 is the raw map)**

| beta | (run, bin, export) triples | median |change| / 25 updates | q90 |change| | median |EMA - raw| | q90 |EMA - raw| |
|---|---|---|---|---|---|
| 0 | 64680 | 0.0140 | 0.0520 | 0.0000 | 0.0000 |
| 0.5 | 64680 | 0.0062 | 0.0226 | 0.0062 | 0.0226 |
| 0.8 | 64680 | 0.0027 | 0.0095 | 0.0107 | 0.0381 |

**Table 4b. Effect of v2.0's 400-update LR decay on the Phase-A candidate (u1200 = last constant-LR export, u1600 = end of the decay): medians over runs**

| group | runs | |peak| | RMSE_pos | tail mean | R | R_tail | Delta |
|---|---|---|---|---|---|---|---|
| pooled | 60 | 0.0586 -> 0.0624 | 0.0283 -> 0.0229 | 0.0099 -> 0.0091 | 0.0987 -> 0.0881 | 0.0105 -> 0.0097 | 0.0016 -> 0.0011 |
| q50 | 30 | 0.0570 -> 0.0615 | 0.0290 -> 0.0232 | 0.0090 -> 0.0080 | 0.1330 -> 0.1060 | 0.0097 -> 0.0086 | 0.0020 -> 0.0013 |
| q60 | 30 | 0.0682 -> 0.0648 | 0.0283 -> 0.0221 | 0.0110 -> 0.0102 | 0.0835 -> 0.0768 | 0.0116 -> 0.0108 | 0.0012 -> 0.0008 |

Reading. The maximum of R is not at the support boundary (1.8 % of exports in the band |d| >= 2q - 10, 43.1 % at near-tie, 55.1 % in the middle, 0 of 60 at u1600); R without the two outermost bins has median 0.1017 against 0.1020 with them, so the boundary does not decide whether rho = 0.03 is attainable. **The maximum sits at d < 0 in 99.5 % of the exports (60 of 60 at u1600), and at rho = 0.03 and u1600, 469 of the 530 bins in S are on the d < 0 side.** With rho = 0.03 the classification at u1600 is localized in 11 of 60 runs (q50 1, q60 10), broad in 49 (mostly |S| above the cap), at u400 1 of 60, at u800 and u1200 0 of 60; with rho = 0.05 it is localized in 5 / 25 / 27 / 37 of 60 at u400 / 800 / 1200 / 1600; with rho = 0.02 never. **So at rho = 0.03 the localized branch (targeted polishing) of the rule is practically unreachable at the calibrated operating point; at rho = 0.05 it is reached.** Near-tie bins enter S more often than middle bins (66 % against 34 % of bins at rho = 0.03, u1600). The EMA at beta = 0.5 halves the check-to-check change of the per-bin map (median 0.0062 against 0.0140 per 25 updates). v2.0's 400-update decay (u1200 to u1600) moves the medians |peak| 0.0586 -> 0.0624, RMSE_pos 0.0283 -> 0.0229, R 0.0987 -> 0.0881. Figures: `cal_fig2_trajectories.png`, `cal_fig3_residual_map_u1600.png`.

### 4.1 Why the residual is largest at d < 0 (an analytic note, checked on a stored export)

The policy itself is nearly symmetric in d (the maximum of |e_hat(d) - e_hat(-d)| over |d| < 2q is 3.2 / 3.3 effort units in the two runs below), the one-step best response is not. Linearising the first-order condition 2 k e = DW f_xi(d + e - e_opp) around a symmetric profile, a deviation c of the opponent from equilibrium is answered by -c a/(2k - a) for d < 0 (where the density f_xi is increasing: strategic complements) and by +c a/(2k + a) for d > 0 (substitutes), with a = DW/(4 q^2). A policy that is off by c everywhere and plays against itself therefore has the residual c 2k/(2k - a) at d < 0 and c 2k/(2k + a) at d > 0: **3.33 c and 0.59 c at q = 50, 1.95 c and 0.67 c at q = 60** (the same factors are what `tests/test_ms_residual.py` finds for an exactly shifted candidate). The first-order residual is therefore an error measure that is amplified on the d < 0 side; the amplification is larger at q = 50, which is why R is larger at q = 50 (median 0.106) than at q = 60 (0.077). In terms of a uniform policy error c at s ~ 68 (q = 50) / 57 (q = 60), R <= 0.03 means c <= about 0.6 / 0.9 effort units (0.9 % / 1.5 % of e2*(0)) and R <= 0.05 means c <= about 1.0 / 1.5 units (1.5 % / 2.5 %); the development-tier search resolves a best response only to a few tenths of an effort unit (Section 3). Evidence on the stored exports: `results/ms_r1/calibration/asymmetry_example.txt` (rehearsal_v2_0 q50 seed 10501 u1600: r(-8) = 8.83 against r(+8) = 2.46, r(-80) = 8.68 against r(+80) = 1.01; q60 seed 10503 u1600: r(-4) = 4.09 against r(+4) = 1.78); the observed ratios are of the order of the predicted ones, not equal to them (the policy error is not uniform).

## 5. Phase B: the stage-1 residual

Stage 1 (the root) has one node; R_1 = |e_hat_1(0) - a_dev_1(0)|/a_dev_1(0) (the first-order analogue of G-S), no tail term; Delta_1 is tiny (median 3e-5, maximum 4.3e-4) and C_1 <= 0.04 at every export, so the stage-1 rule reduces to `R_1 <= rho` for M checks.

**Table 5a. Phase B (stage 1, u1625..u2200, every export): stage-1 residual r_1(0)/s_1 against the closed-form stage-1 error (signed residual = (e_hat_1 - s_1)/s_1; learning proxy = (e_hat_1 - s_1)/g1)**

| group | n pairs | exports with R_1 = 0 exactly | exports with C_1 <= 0.04 | Spearman R vs |err| | Pearson R vs |err| | Spearman signed res vs err | Pearson signed res vs err | Pearson learning proxy vs err | within-run median Spearman R vs |err| | within-run median Spearman signed |
|---|---|---|---|---|---|---|---|---|---|---|
| pooled | 1440 | 108 | 1.000 | 0.735 | 0.901 | 0.900 | 0.901 | 0.962 | 0.877 | 0.999 |
| q50 | 720 | 102 | 1.000 | 0.652 | 0.959 | 0.857 | 0.942 | 0.974 | 0.710 | 0.994 |
| q60 | 720 | 6 | 1.000 | 0.862 | 0.992 | 0.957 | 0.992 | 0.994 | 0.935 | 1.000 |

**Table 5b. Stage 1 at u2200 (the end of v2.0's 600-update decay), median [min, max] over runs**

| group | runs | R_1 = r_1(0)/s_1 | Delta_1 | C_1 | signed stage-1 error | median |error| | inherited proxy (s_1 - g1)/g1 | learning proxy (e_hat_1 - s_1)/g1 |
|---|---|---|---|---|---|---|---|---|
| pooled | 60 | 0.0158 [0.0000, 0.0792] | 0.00003 [0.00000, 0.00043] | 0.0271 [0.0246, 0.0342] | -0.0024 [-0.0407, 0.0464] | 0.0128 | 0.0039 [-0.0305, 0.0286] | -0.0086 [-0.0480, 0.0768] |
| q50 | 30 | 0.0148 [0.0000, 0.0792] | 0.00002 [0.00000, 0.00043] | 0.0282 [0.0260, 0.0342] | -0.0065 [-0.0407, 0.0464] | 0.0131 | 0.0043 [-0.0305, 0.0286] | -0.0127 [-0.0480, 0.0768] |
| q60 | 30 | 0.0188 [0.0000, 0.0399] | 0.00003 [0.00000, 0.00017] | 0.0260 [0.0246, 0.0303] | 1.77e-05 [-0.0324, 0.0280] | 0.0117 | 0.0039 [-0.0134, 0.0195] | -0.0052 [-0.0402, 0.0381] |

**Table 5c. Stage-1 rule (eps = 0.005, M = 3, K = 25, no tail term): fire export in local updates of Phase B, training updates saved against v2.0's fixed 600, total updates with a 400-update landing window, and the closed-form |stage-1 error| at the fire export**

| rho | group | fired | fire local update med [min,max] | 600 - fire (training updates saved) | fire + 400 (median) | |err_1| at fire | |err_1| at u2200 (same runs) | runs with |err_1|>0.05 | fires with R_1 = 0 in a firing check |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | pooled | 58/60 | 288 [100, 575] | 312 [25, 500] | 688 | 0.0088 [0.0004, 0.0258] | 0.0130 | 0 | 19 |
| 0.02 | q50 | 28/30 | 312 [100, 575] | 288 [25, 500] | 712 | 0.0124 [0.0014, 0.0258] | 0.0133 | 0 | 18 |
| 0.02 | q60 | 30/30 | 262 [100, 575] | 338 [25, 500] | 662 | 0.0069 [0.0004, 0.0215] | 0.0117 | 0 | 1 |
| 0.03 | pooled | 60/60 | 188 [75, 550] | 412 [50, 525] | 588 | 0.0122 [0.0004, 0.0322] | 0.0128 | 0 | 17 |
| 0.03 | q50 | 30/30 | 225 [100, 550] | 375 [50, 500] | 625 | 0.0132 [0.0014, 0.0258] | 0.0131 | 0 | 16 |
| 0.03 | q60 | 30/30 | 175 [75, 400] | 425 [200, 525] | 575 | 0.0100 [0.0004, 0.0322] | 0.0117 | 0 | 1 |
| 0.05 | pooled | 60/60 | 125 [75, 300] | 475 [300, 525] | 525 | 0.0152 [0.0003, 0.0407] | 0.0128 | 0 | 9 |
| 0.05 | q50 | 30/30 | 150 [75, 300] | 450 [300, 525] | 550 | 0.0152 [0.0003, 0.0296] | 0.0131 | 0 | 9 |
| 0.05 | q60 | 30/30 | 100 [75, 200] | 500 [400, 525] | 500 | 0.0154 [0.0005, 0.0407] | 0.0117 | 0 | 0 |

**Table 5d. Stage-1 error noise by v2.0 learning-rate band (descriptive; v2.0 decays 3e-4 -> 3e-5 over the 600 local updates of Phase B; the MS stage-1 phase keeps 3e-4 until the stop and then decays over N_land = 400)**

| local updates | v2.0 LR | group | exports | median |err_1| | q90 |err_1| | median |err_1(u) - err_1(u-25)| | q90 swing | median R_1 | q90 R_1 |
|---|---|---|---|---|---|---|---|---|---|
| 1-200 | 3.00e-04 -> 2.10e-04 | pooled | 480 | 0.0189 | 0.0714 | 0.0257 | 0.0897 | 0.0256 | 0.1162 |
| 1-200 | 3.00e-04 -> 2.10e-04 | q50 | 240 | 0.0205 | 0.0720 | 0.0283 | 0.0897 | 0.0321 | 0.1363 |
| 1-200 | 3.00e-04 -> 2.10e-04 | q60 | 240 | 0.0178 | 0.0678 | 0.0229 | 0.0871 | 0.0237 | 0.0813 |
| 201-400 | 2.10e-04 -> 1.20e-04 | pooled | 480 | 0.0147 | 0.0330 | 0.0166 | 0.0463 | 0.0194 | 0.0463 |
| 201-400 | 2.10e-04 -> 1.20e-04 | q50 | 240 | 0.0172 | 0.0350 | 0.0185 | 0.0469 | 0.0194 | 0.0526 |
| 201-400 | 2.10e-04 -> 1.20e-04 | q60 | 240 | 0.0123 | 0.0315 | 0.0154 | 0.0415 | 0.0200 | 0.0399 |
| 401-600 | 1.20e-04 -> 3.00e-05 | pooled | 480 | 0.0113 | 0.0268 | 0.0135 | 0.0329 | 0.0157 | 0.0367 |
| 401-600 | 1.20e-04 -> 3.00e-05 | q50 | 240 | 0.0128 | 0.0294 | 0.0137 | 0.0334 | 0.0161 | 0.0402 |
| 401-600 | 1.20e-04 -> 3.00e-05 | q60 | 240 | 0.0103 | 0.0257 | 0.0130 | 0.0318 | 0.0150 | 0.0341 |

Reading. R_1 follows the closed-form stage-1 error (Pearson 0.90 between the signed residual and the signed error, Spearman 0.735 between R_1 and |error|). At u2200 R_1 has median 0.0158 [0, 0.0792] and the median |error| 0.0128; all 60 runs are within 0.05. The stage-1 rule with rho = 0.03 fires in 60 of 60 replayed trajectories at median local update 188 (rho = 0.02: 58 of 60 at 288; rho = 0.05: 60 of 60 at 125), saving a median 412 training updates against v2.0's 600 (312 / 475); with a 400-update landing the total is about 588 (688 / 525). The closed-form |error| at the fire export at rho = 0.03 is 0.0122 median [0.0004, 0.0322]. Two cautions: (i) the replay favours the stage-1 stop: v2.0's Phase B decays the LR from the first update, so these fires come at LR about 2.2e-4, whereas the MS stage-1 phase checks at the constant 3e-4, where the export-to-export swing of the stage-1 error is a median 0.026 (q90 0.090) in v2.0's top LR band against 0.0135 (q90 0.033) in the bottom band; the landing window (400 updates of decay) is what removes this noise in the MS pipeline, as the last 400 updates do in v2.0. (ii) **Anomaly:** 108 of the 1440 Phase-B exports (102 of 720 at q50, 6 of 720 at q60) have R_1 = 0 exactly: the development-tier search returns the policy's own action (`a_dev_source = mean_action`, Delta_1 = 0; e.g. rehearsal q50 seed 10501 at u1800, where the final tier gives R_1 = 0.0068); in 17 of the 60 fires at rho = 0.03 one of the three firing checks has R_1 = 0. This is a property of the verifier's one-step search (the policy's own action is a candidate and wins when the objective is flat), not of the new modules; at stage 2 R = 0 occurs in 0 of 3840 exports. Figure: `cal_fig4_phaseB_stage1.png`.

## 6. Cost

**Table 6. Wall-clock of one verifier call at T = 2 (milliseconds)**

| source | calls | median ms | mean ms | q10 - q90 ms | min - max ms |
|---|---|---|---|---|---|
| replay dev tier, Phase A (stage 2) | 3840 | 9.1 | 9.9 | 7.9 - 13.5 | 7.1 - 31.4 |
| replay dev tier, Phase B (stage 1) | 1440 | 8.9 | 9.8 | 7.7 - 13.1 | 7.1 - 32.1 |
| replay final tier (u1600) | 60 | 19.9 | 20.6 | 17.3 - 24.9 | 16.1 - 33.0 |
| replay D3 diag + concentration (dev, Phase A) | 3840 | 0.5 | 0.6 | 0.5 - 0.7 | 0.4 - 5.9 |
| v2.0 own log v2_checkpoints_A.csv (dev tier) | 963 | 9.0 | 9.8 | 8.4 - 10.8 | 8.0 - 25.1 |
| v2.0 own log v2_checkpoints_B.csv (dev tier) | 1260 | 8.8 | 9.0 | 8.2 - 9.9 | 7.8 - 14.8 |

One development-tier call costs about 9 ms at T = 2 (the preamble's 10 ms); the D3 diagnostics add 0.5 ms. At K = 25 the checks of a 2400-update stage cost about one second.

## 7. What the calibration says about the parameters (D8)

This section is neutral; the choice is made in `02_preregistration.md` section 3.

- `epsilon_t = 0.005` (fixed by D8): Delta_t <= 0.005 passes at 95 % of the Phase-A exports and at essentially all stage-1 exports; it does not bind.
- `rho_2` (terminal stage): the candidates of the grid (0.02, 0.03, 0.05) make the stop inert on v2.0-like trajectories (0, 0, 1 of 60 fire) and make the localized classification practically unreachable at 0.02 (0 of 60 at every block end) and at 0.03 (11 of 60 at u1600, at most 1 of 60 earlier); at 0.05 it is reached (5, 25, 27, 37 of 60). A rho at which the stop fires at v2.0's quality is about 0.06 to 0.08 (Table 2c: 9 of 60 fire at 0.06 with |peak| 0.0493 against 0.0499; 35 of 60 at 0.08 with 0.0613 against 0.0628); at 0.10 or above it fires earlier and worse (fact 2). The analytic note (4.1) gives the reading of these numbers as a tolerance on the policy error.
- `rho_1` (stage 1): 0.03 fires in 60 of 60 trajectories with |error| at the fire of 0.0122 (maximum 0.0322), within the G-S limit 0.05.
- `tau_t = 0.02`: R_tail <= 0.02 passes at 90 % of the exports (median at u1600 0.0097, maximum 0.0179); the G-A limit is 0.02.
- `M = 3`, `K = 25`: K equals the export cadence of the replay; no table argues for a change. If `N_block` is not a multiple of `K`, the block-end checks add checks at spacing below K (the defaults 400 / 200 and 25 are aligned).
- `N_block`, `U_cap`: the classification table is at the block ends 400, 800, 1200, 1600; `U_cap = 2000` lies beyond the last Phase-A export, so it cannot be calibrated offline.
- `N_land = 400`: v2.0's 400-update decay improves RMSE_pos (0.0283 -> 0.0229), tail mean and R, and moves the median |peak| from 0.0586 to 0.0624 (Table 4 landing).
- `alpha_polish`: cannot be calibrated offline (the replay does not simulate the sampler). `beta`: Table 4 (EMA). Localized fraction: Table 4 (sensitivity).

## 8. Limits of the calibration

The replay applies the rule to v2.0 trajectories: sampler, polishing and landing are not simulated, so the tables say when the rule would fire on bin-balanced Phase-A training and what the residual looks like there, not what it will do under the new sampler. Exports after u1200 come from v2.0's decaying LR, which the MS pipeline does not use before the stop. The final tier exists only at Phase-A u1600. The residual statistics of a trajectory trained with a different start distribution may differ.

## Reproduce

```
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/fjiang4/tournament_experiment/.venv/bin/python
ROOT_V2=/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked
$PY tools/ms/replay_dev_rule.py replay  --root-v2 $ROOT_V2 --out results/ms_r1/calibration --workers 8
$PY tools/ms/replay_dev_rule.py analyze --root-v2 $ROOT_V2 --out results/ms_r1/calibration --fig-dir reports/ms/r1/figures --md-dir <dir>   # the tables above are the markdown rendering
$PY tools/ms/residual_asymmetry_example.py --root-v2 $ROOT_V2 > results/ms_r1/calibration/asymmetry_example.txt
$PY -m pytest tests/test_ms_replay.py -p no:cacheprovider -q
```
