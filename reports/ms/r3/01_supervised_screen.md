# MS-R3 P1: the offline supervised screen, the premise check and the preamble numbers

Date: 2026-10-07. Spec: `reports/ms/r3/pi_record/20_ms_r3_prompt.md` sections 2.3 (a), 2.4 and 2.5. Tool `tools/ms/r3_supervised_screen.py` (tests `tests/test_ms_r3_screen.py`), outputs `results/ms_r3/supervised_screen/` (`cells/*.json`, `summary_by_cell.csv`, `summary_median.csv`, `summary_extended.csv`, `premise_check.json`, `run_manifest.json`, `run.log`). Every table below is printed by `python reports/ms/r3/report_scripts/screen_tables.py --block <name>` (screen blocks) or `python reports/ms/r3/report_scripts/preamble_numbers.py --block <name>` (section 4). The screen is an evaluation-side diagnostic: the closed-form tent is its fitting target, it enters no RL run.

## 1. What was run

For each actor variant (`t1`, `relu`, `t10`), the **real** `BetaActor` of `agents/ppo_curriculum.py` (hidden 64, the record's `c_min` and `mu_clamp`), initialised as the runner initialises it for (q, seed) (the torch generator seeded from `SeedSequence([seed, q, namespace init])`, the actor being the first draw; the variant is set afterwards, so the initial weights are identical across the three actors for every (q, seed): `premise_check.json` `init_identical_across_actors` is true), is fit by MSE in effort units to the closed-form stage-2 tent `e2*(d)` (`utils.theory_multistage.g2_two_stage`, the function behind the verifier's `g2_at_0`): the loss is `mean((e_min + e_range mu(d) - e2*(d))^2)` with mu the clamped sigmoid of the first output of the variant's own forward (the concentration output receives no gradient). Optimisation matched to the RL terminal stage: 56,000 Adam steps with the record's betas and eps, learning rate 3e-4 for the first 48,000 steps then linear to 3e-5 over the last 8,000 (changed at every step, as in the PI's sandbox; the RL runs change it once per 20-step update), minibatch 256, global gradient-norm clip at the record's `max_grad_norm`. Inputs d are drawn each step from the start distribution on D_2 with the repository's `StartSampler`: bin-balanced (`balanced`: a bin uniformly, then uniform within it) or stratified (`stratified_priority(stratified_bin_probs(2, lambda_P 0.35, near-tie half-width 20, alpha 0))`, i.e. the `NL_st` shares without a verifier focus); the d stream has its own generator `SeedSequence([seed, q, 30001])`, recorded in every cell. Metrics are the verifier's own (`utils.v2_metrics.recovery_metrics` on the policy function of the fitted actor, recovery grid of the record): tip deficit `e2*(0) - e_hat(0)` (signed, effort units), RMSE_pos (over |d| < 2q), tail mean, `w_eff` = tip deficit / (`e2*(0)` / 2q) in units of d, and max |w| of the first layer on the d input in units of d / B (ten times the stored weight for `t10`).

Grid: 3 actors x 2 starts x 2 q x 10 seeds (10501-10510) = 120 cells at 56,000 steps, plus the extended cell (`t1`, bin-balanced, 10 seeds x 2 q, 224,000 steps: LR 3e-4 for 216,000 steps then linear to 3e-5 over 8,000; checkpoints every 56,000). Run record (`results/ms_r3/supervised_screen/run.log`): 140 cells, 140 ok, 0 failed, 831.8 s of wall time with 40 single-threaded workers (command in `run_manifest.json`). The `wall_sec` of the cells (`cells/*.json`) is 133-178 s (median 158 s) for the 120 main cells and 588-657 s for the 20 extended cells, measured at a machine load average of about 50 (other jobs were running).

## 2. Results at the RL budget (56,000 steps) and on the way there

### 2.1 Tip deficit `e2*(0) - e_hat(0)` (effort units), median [min, max] over the ten seeds (block `grid`)

| actor | starts | q | 16k | 32k | 48k | 56k | n seeds |
|---|---|---|---|---|---|---|---|
| t1 | bin-balanced | 50 | 2.110 [1.145, 9.909] | 1.671 [0.914, 8.596] | 1.514 [0.642, 9.411] | 1.643 [0.847, 8.590] | 10 |
| t1 | bin-balanced | 60 | 5.398 [1.384, 6.918] | 5.092 [1.016, 7.627] | 6.122 [0.816, 8.002] | 6.320 [0.876, 6.739] | 10 |
| t1 | stratified | 50 | 0.946 [0.712, 1.748] | 1.048 [0.583, 1.783] | 0.842 [0.457, 1.954] | 0.712 [0.624, 1.610] | 10 |
| t1 | stratified | 60 | 0.904 [0.721, 2.187] | 1.172 [0.677, 1.626] | 0.989 [0.547, 1.914] | 1.169 [0.594, 1.357] | 10 |
| relu | bin-balanced | 50 | 0.505 [0.400, 0.593] | 0.487 [0.119, 0.567] | 0.457 [0.086, 0.563] | 0.483 [0.091, 0.514] | 10 |
| relu | bin-balanced | 60 | 0.037 [-0.019, 0.087] | 0.056 [0.029, 0.097] | 0.035 [0.008, 0.095] | 0.032 [0.022, 0.053] | 10 |
| relu | stratified | 50 | 0.352 [0.019, 0.472] | 0.105 [0.026, 0.484] | 0.098 [0.021, 0.421] | 0.081 [0.029, 0.420] | 10 |
| relu | stratified | 60 | 0.048 [0.014, 0.104] | 0.048 [0.014, 0.066] | 0.039 [-0.006, 0.101] | 0.025 [0.006, 0.031] | 10 |
| t10 | bin-balanced | 50 | 0.967 [0.635, 1.067] | 0.582 [0.489, 1.070] | 0.565 [0.460, 0.887] | 0.528 [0.370, 0.695] | 10 |
| t10 | bin-balanced | 60 | 0.835 [0.369, 1.092] | 0.648 [0.400, 0.930] | 0.533 [0.215, 0.662] | 0.478 [0.398, 0.569] | 10 |
| t10 | stratified | 50 | 0.824 [0.485, 1.059] | 0.676 [0.325, 0.751] | 0.492 [0.221, 0.698] | 0.437 [0.369, 0.553] | 10 |
| t10 | stratified | 60 | 0.657 [0.378, 0.987] | 0.540 [0.271, 0.706] | 0.508 [0.082, 0.634] | 0.335 [0.243, 0.445] | 10 |

### 2.2 The other reported metrics at 56,000 steps (block `metrics`; RMSE_pos and tail mean in effort units, w_eff in units of d, max |w| in units of d / B)

| actor | starts | q | RMSE_pos | tail mean | w_eff | max abs w on d (d / B) | tip deficit / e2*(0) |
|---|---|---|---|---|---|---|---|
| t1 | bin-balanced | 50 | 0.256 [0.089, 2.691] | 0.083 [0.039, 0.663] | 2.35 [1.21, 12.27] | 1.30 [1.18, 3.63] | 0.0235 [0.0121, 0.1227] |
| t1 | bin-balanced | 60 | 2.098 [0.100, 2.120] | 0.592 [0.057, 0.607] | 13.00 [1.80, 13.86] | 1.16 [0.99, 2.53] | 0.1084 [0.0150, 0.1155] |
| t1 | stratified | 50 | 0.104 [0.078, 0.944] | 0.069 [0.050, 0.289] | 1.02 [0.89, 2.30] | 1.51 [1.39, 1.93] | 0.0102 [0.0089, 0.0230] |
| t1 | stratified | 60 | 0.356 [0.114, 0.561] | 0.158 [0.076, 0.206] | 2.40 [1.22, 2.79] | 1.31 [1.06, 2.48] | 0.0200 [0.0102, 0.0233] |
| relu | bin-balanced | 50 | 0.079 [0.027, 0.081] | 0.002 [0.001, 0.002] | 0.69 [0.13, 0.73] | 0.86 [0.71, 0.95] | 0.0069 [0.0013, 0.0073] |
| relu | bin-balanced | 60 | 0.018 [0.014, 0.023] | 0.001 [0.001, 0.001] | 0.07 [0.04, 0.11] | 0.79 [0.72, 0.94] | 0.0005 [0.0004, 0.0009] |
| relu | stratified | 50 | 0.039 [0.029, 0.112] | 0.002 [0.001, 0.003] | 0.12 [0.04, 0.60] | 0.87 [0.79, 0.98] | 0.0012 [0.0004, 0.0060] |
| relu | stratified | 60 | 0.022 [0.016, 0.024] | 0.001 [0.001, 0.001] | 0.05 [0.01, 0.06] | 0.82 [0.74, 0.98] | 0.0004 [0.0001, 0.0005] |
| t10 | bin-balanced | 50 | 0.064 [0.044, 0.078] | 0.022 [0.019, 0.026] | 0.75 [0.53, 0.99] | 14.64 [10.08, 29.37] | 0.0075 [0.0053, 0.0099] |
| t10 | bin-balanced | 60 | 0.058 [0.044, 0.067] | 0.021 [0.017, 0.026] | 0.98 [0.82, 1.17] | 17.82 [12.69, 34.83] | 0.0082 [0.0068, 0.0097] |
| t10 | stratified | 50 | 0.073 [0.061, 0.085] | 0.040 [0.034, 0.045] | 0.62 [0.53, 0.79] | 13.06 [9.18, 24.35] | 0.0062 [0.0053, 0.0079] |
| t10 | stratified | 60 | 0.053 [0.044, 0.077] | 0.034 [0.026, 0.048] | 0.69 [0.50, 0.92] | 17.94 [8.60, 37.23] | 0.0057 [0.0042, 0.0076] |

The same four metrics at the four checkpoints (blocks `grid_rmse`, `grid_tail`, `grid_weff`, `grid_maxw`; median [min, max] over the ten seeds).

RMSE_pos (effort units):

| actor | starts | q | 16k | 32k | 48k | 56k | n seeds |
|---|---|---|---|---|---|---|---|
| t1 | bin-balanced | 50 | 0.501 [0.204, 2.967] | 0.394 [0.139, 2.714] | 0.448 [0.126, 2.816] | 0.256 [0.089, 2.691] | 10 |
| t1 | bin-balanced | 60 | 2.114 [0.211, 2.443] | 2.109 [0.171, 2.661] | 2.114 [0.115, 2.396] | 2.098 [0.100, 2.120] | 10 |
| t1 | stratified | 50 | 0.388 [0.236, 1.486] | 0.307 [0.187, 1.279] | 0.260 [0.103, 1.023] | 0.104 [0.078, 0.944] | 10 |
| t1 | stratified | 60 | 0.463 [0.286, 1.446] | 0.413 [0.180, 0.749] | 0.427 [0.164, 0.807] | 0.356 [0.114, 0.561] | 10 |
| relu | bin-balanced | 50 | 0.115 [0.093, 0.177] | 0.092 [0.047, 0.173] | 0.088 [0.043, 0.122] | 0.079 [0.027, 0.081] | 10 |
| relu | bin-balanced | 60 | 0.048 [0.033, 0.164] | 0.040 [0.028, 0.064] | 0.032 [0.027, 0.074] | 0.018 [0.014, 0.023] | 10 |
| relu | stratified | 50 | 0.133 [0.073, 0.181] | 0.083 [0.045, 0.161] | 0.084 [0.049, 0.131] | 0.039 [0.029, 0.112] | 10 |
| relu | stratified | 60 | 0.054 [0.039, 0.108] | 0.037 [0.030, 0.055] | 0.047 [0.028, 0.078] | 0.022 [0.016, 0.024] | 10 |
| t10 | bin-balanced | 50 | 0.162 [0.142, 0.219] | 0.112 [0.084, 0.279] | 0.104 [0.072, 0.206] | 0.064 [0.044, 0.078] | 10 |
| t10 | bin-balanced | 60 | 0.176 [0.140, 0.208] | 0.116 [0.088, 0.202] | 0.106 [0.073, 0.140] | 0.058 [0.044, 0.067] | 10 |
| t10 | stratified | 50 | 0.236 [0.148, 0.297] | 0.162 [0.124, 0.196] | 0.123 [0.093, 0.179] | 0.073 [0.061, 0.085] | 10 |
| t10 | stratified | 60 | 0.198 [0.134, 0.312] | 0.126 [0.094, 0.178] | 0.118 [0.073, 0.160] | 0.053 [0.044, 0.077] | 10 |

Tail mean (effort units):

| actor | starts | q | 16k | 32k | 48k | 56k | n seeds |
|---|---|---|---|---|---|---|---|
| t1 | bin-balanced | 50 | 0.118 [0.083, 0.622] | 0.089 [0.053, 0.661] | 0.090 [0.042, 0.652] | 0.083 [0.039, 0.663] | 10 |
| t1 | bin-balanced | 60 | 0.611 [0.118, 0.670] | 0.596 [0.076, 0.645] | 0.569 [0.061, 0.617] | 0.592 [0.057, 0.607] | 10 |
| t1 | stratified | 50 | 0.123 [0.096, 0.579] | 0.088 [0.064, 0.428] | 0.078 [0.053, 0.319] | 0.069 [0.050, 0.289] | 10 |
| t1 | stratified | 60 | 0.276 [0.128, 0.477] | 0.209 [0.103, 0.330] | 0.174 [0.089, 0.234] | 0.158 [0.076, 0.206] | 10 |
| relu | bin-balanced | 50 | 0.009 [0.007, 0.011] | 0.003 [0.002, 0.004] | 0.002 [0.001, 0.002] | 0.002 [0.001, 0.002] | 10 |
| relu | bin-balanced | 60 | 0.006 [0.005, 0.008] | 0.002 [0.002, 0.002] | 0.001 [0.001, 0.002] | 0.001 [0.001, 0.001] | 10 |
| relu | stratified | 50 | 0.012 [0.010, 0.017] | 0.004 [0.002, 0.005] | 0.002 [0.001, 0.003] | 0.002 [0.001, 0.003] | 10 |
| relu | stratified | 60 | 0.007 [0.005, 0.009] | 0.002 [0.001, 0.003] | 0.001 [0.001, 0.001] | 0.001 [0.001, 0.001] | 10 |
| t10 | bin-balanced | 50 | 0.058 [0.054, 0.066] | 0.033 [0.029, 0.037] | 0.025 [0.020, 0.028] | 0.022 [0.019, 0.026] | 10 |
| t10 | bin-balanced | 60 | 0.059 [0.050, 0.063] | 0.032 [0.027, 0.038] | 0.023 [0.019, 0.028] | 0.021 [0.017, 0.026] | 10 |
| t10 | stratified | 50 | 0.089 [0.077, 0.092] | 0.056 [0.048, 0.062] | 0.044 [0.038, 0.050] | 0.040 [0.034, 0.045] | 10 |
| t10 | stratified | 60 | 0.086 [0.073, 0.098] | 0.048 [0.041, 0.065] | 0.037 [0.029, 0.052] | 0.034 [0.026, 0.048] | 10 |

w_eff (units of d):

| actor | starts | q | 16k | 32k | 48k | 56k | n seeds |
|---|---|---|---|---|---|---|---|
| t1 | bin-balanced | 50 | 3.01 [1.64, 14.16] | 2.39 [1.31, 12.28] | 2.16 [0.92, 13.44] | 2.35 [1.21, 12.27] | 10 |
| t1 | bin-balanced | 60 | 11.10 [2.85, 14.23] | 10.47 [2.09, 15.69] | 12.59 [1.68, 16.46] | 13.00 [1.80, 13.86] | 10 |
| t1 | stratified | 50 | 1.35 [1.02, 2.50] | 1.50 [0.83, 2.55] | 1.20 [0.65, 2.79] | 1.02 [0.89, 2.30] | 10 |
| t1 | stratified | 60 | 1.86 [1.48, 4.50] | 2.41 [1.39, 3.34] | 2.03 [1.12, 3.94] | 2.40 [1.22, 2.79] | 10 |
| relu | bin-balanced | 50 | 0.72 [0.57, 0.85] | 0.70 [0.17, 0.81] | 0.65 [0.12, 0.80] | 0.69 [0.13, 0.73] | 10 |
| relu | bin-balanced | 60 | 0.08 [-0.04, 0.18] | 0.12 [0.06, 0.20] | 0.07 [0.02, 0.20] | 0.07 [0.04, 0.11] | 10 |
| relu | stratified | 50 | 0.50 [0.03, 0.67] | 0.15 [0.04, 0.69] | 0.14 [0.03, 0.60] | 0.12 [0.04, 0.60] | 10 |
| relu | stratified | 60 | 0.10 [0.03, 0.21] | 0.10 [0.03, 0.13] | 0.08 [-0.01, 0.21] | 0.05 [0.01, 0.06] | 10 |
| t10 | bin-balanced | 50 | 1.38 [0.91, 1.52] | 0.83 [0.70, 1.53] | 0.81 [0.66, 1.27] | 0.75 [0.53, 0.99] | 10 |
| t10 | bin-balanced | 60 | 1.72 [0.76, 2.25] | 1.33 [0.82, 1.91] | 1.10 [0.44, 1.36] | 0.98 [0.82, 1.17] | 10 |
| t10 | stratified | 50 | 1.18 [0.69, 1.51] | 0.97 [0.46, 1.07] | 0.70 [0.32, 1.00] | 0.62 [0.53, 0.79] | 10 |
| t10 | stratified | 60 | 1.35 [0.78, 2.03] | 1.11 [0.56, 1.45] | 1.05 [0.17, 1.30] | 0.69 [0.50, 0.92] | 10 |

Max abs first-layer d-weight (units of d / B; ten times the stored weight for `t10`):

| actor | starts | q | 16k | 32k | 48k | 56k | n seeds |
|---|---|---|---|---|---|---|---|
| t1 | bin-balanced | 50 | 1.14 [1.04, 1.41] | 1.25 [1.08, 1.46] | 1.29 [1.13, 3.46] | 1.30 [1.18, 3.63] | 10 |
| t1 | bin-balanced | 60 | 1.12 [0.99, 1.84] | 1.13 [0.99, 2.43] | 1.17 [0.99, 2.50] | 1.16 [0.99, 2.53] | 10 |
| t1 | stratified | 50 | 1.18 [1.02, 1.71] | 1.31 [1.24, 2.00] | 1.48 [1.35, 1.93] | 1.51 [1.39, 1.93] | 10 |
| t1 | stratified | 60 | 1.16 [0.99, 1.51] | 1.22 [1.01, 2.29] | 1.30 [1.05, 2.38] | 1.31 [1.06, 2.48] | 10 |
| relu | bin-balanced | 50 | 0.86 [0.71, 0.95] | 0.86 [0.71, 0.95] | 0.86 [0.71, 0.95] | 0.86 [0.71, 0.95] | 10 |
| relu | bin-balanced | 60 | 0.79 [0.72, 0.93] | 0.79 [0.72, 0.93] | 0.79 [0.72, 0.93] | 0.79 [0.72, 0.94] | 10 |
| relu | stratified | 50 | 0.86 [0.77, 0.97] | 0.87 [0.79, 0.97] | 0.87 [0.79, 0.98] | 0.87 [0.79, 0.98] | 10 |
| relu | stratified | 60 | 0.80 [0.73, 0.97] | 0.81 [0.73, 0.98] | 0.82 [0.74, 0.98] | 0.82 [0.74, 0.98] | 10 |
| t10 | bin-balanced | 50 | 10.70 [7.77, 15.44] | 12.60 [9.22, 23.22] | 13.98 [9.86, 27.82] | 14.64 [10.08, 29.37] | 10 |
| t10 | bin-balanced | 60 | 11.58 [8.13, 18.18] | 15.11 [10.70, 26.87] | 17.01 [12.42, 32.47] | 17.82 [12.69, 34.83] | 10 |
| t10 | stratified | 50 | 8.77 [8.09, 12.80] | 11.38 [8.72, 19.47] | 12.05 [9.02, 23.24] | 13.06 [9.18, 24.35] | 10 |
| t10 | stratified | 60 | 11.31 [6.64, 12.65] | 14.69 [7.72, 27.64] | 17.31 [8.26, 35.51] | 17.94 [8.60, 37.23] | 10 |

### 2.3 The ten per-seed tip deficits at 56,000 steps, bin-balanced starts (block `seeds`)

| actor | q | 10501 | 10502 | 10503 | 10504 | 10505 | 10506 | 10507 | 10508 | 10509 | 10510 | seeds with deficit > 4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t1 | 50 | 3.15 | 1.06 | 8.59 | 1.26 | 0.87 | 1.04 | 3.32 | 0.85 | 2.03 | 8.52 | 2 |
| t1 | 60 | 1.99 | 6.18 | 6.53 | 1.63 | 6.74 | 6.54 | 6.46 | 6.54 | 0.88 | 1.47 | 6 |
| relu | 50 | 0.09 | 0.10 | 0.49 | 0.49 | 0.10 | 0.48 | 0.49 | 0.10 | 0.49 | 0.51 | 0 |
| relu | 60 | 0.04 | 0.03 | 0.04 | 0.03 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.05 | 0 |
| t10 | 50 | 0.47 | 0.49 | 0.55 | 0.70 | 0.50 | 0.51 | 0.62 | 0.37 | 0.60 | 0.57 | 0 |
| t10 | 60 | 0.52 | 0.41 | 0.52 | 0.57 | 0.51 | 0.49 | 0.44 | 0.46 | 0.40 | 0.44 | 0 |

Reading (descriptive). The current actor (`t1`) leaves a median tip deficit of 1.64 (q = 50) and 6.32 (q = 60) effort units under bin-balanced starts, and the distribution over seeds is split: at q = 50 two of ten seeds end near 8.5 and the other eight between 0.85 and 3.32; at q = 60 six of ten seeds end between 6.18 and 6.74 and four between 0.88 and 1.99. Under stratified starts its median deficit is lower at both q (0.71 and 1.17). Both kink-capable variants leave less: `relu` 0.48 and 0.03 (bin-balanced), `t10` 0.53 and 0.48; under stratified starts 0.081 / 0.025 and 0.437 / 0.335. The `relu` cell at q = 50 splits as well (four seeds at 0.09-0.10, six at 0.48-0.51). RMSE_pos, the tail mean and w_eff follow the same order (table 2.2); the first-layer d-weight of `t1` has a median of 1.30 / 1.16 (bin-balanced) and 1.51 / 1.31 (stratified), `relu` 0.86 / 0.79 and 0.87 / 0.82, `t10` 13.1-17.9 in units of d / B (the stored weights times ten, i.e. about 1.3-1.8 in the stored unit).

## 3. The premise check (prompt section 2.5; block `premise`; bin-balanced starts, 56,000 steps)

| condition | variant | q | median deficit | limit | result |
|---|---|---|---|---|---|
| (i) median deficit of t1 >= 1.0 | t1 | 50 | 1.6430 | >= 1 | pass |
| (i) median deficit of t1 >= 1.0 | t1 | 60 | 6.3205 | >= 1 | pass |
| (ii) median deficit <= 0.5 x that of t1 | relu | 50 | 0.4827 | <= 0.8215 (ratio 0.294) | pass |
| (ii) median deficit <= 0.5 x that of t1 | relu | 60 | 0.0318 | <= 3.1602 (ratio 0.005) | pass |
| (ii) median deficit <= 0.5 x that of t1 | t10 | 50 | 0.5279 | <= 0.8215 (ratio 0.321) | pass |
| (ii) median deficit <= 0.5 x that of t1 | t10 | 60 | 0.4777 | <= 3.1602 (ratio 0.076) | pass |
Outcome: **PASS** ((i) holds and both variants pass (ii)). Seeds per group: [10]; complete: True; initial weights identical across the three actors for every (q, seed): True.

(i) holds (the median tip deficit of `t1` is at least 1.0 effort unit at both q) and both variants meet (ii) (at most half of `t1`'s median at both q), with margin: the weakest ratio is 0.321 (`t10` at q = 50). **Outcome: PASS. No variant is dropped; the pilot runs all twelve arms (240 runs).** The stratified cells are reported (2.1, 2.2), not gated.

## 4. Reproduction of the preamble numbers (prompt section 2.4)

From `results/ms_r2/analysis/per_run.csv` of `origin/ms-r2` (the file of this worktree, `ms-r3` is `ms-r2` plus this round's commits), script `reports/ms/r3/report_scripts/preamble_numbers.py`. The preamble is not edited; a number that differs is marked.

**Item 1 and the tie-effort table** (seed means of `e_hat_2(0)` at s = 1 / 4 / 16; block `tie_effort`):

| q | bin-balanced: reproduced | bin-balanced: quoted | bin-balanced: match | stratified: reproduced | stratified: quoted | stratified: match |
|---|---|---|---|---|---|---|
| 50 | 66.04 / 66.44 / 66.10 | 66.04 / 66.44 / 66.10 | ok | 66.97 / 66.87 / 67.19 | 66.97 / 66.87 / 67.19 | ok |
| 60 | 56.00 / 55.33 / 55.58 | 56.00 / 55.33 / 55.58 | ok | 55.64 / 55.99 / 55.91 | 55.64 / 55.99 / 55.91 | ok |

**Item 1, the stratified sampler's shift of the seed-mean tie effort** (stratified minus bin-balanced; block `sampler`):

| q | st - bb at s = 1 / 4 / 16 | reproduced range (1 dp) | quoted | match |
|---|---|---|---|---|
| 50 | +0.923 / +0.436 / +1.089 | +0.4 to +1.1 | +0.4 to +1.1 | ok |
| 60 | -0.354 / +0.653 / +0.332 | -0.4 to +0.7 | -0.4 to +0.7 | ok |

**Item 1, the budget** (`MS_base2400` against `parents_A`, 1600 -> 2400 updates; block `budget`):

| q | MS_base2400 | parents_A | difference | quoted | match |
|---|---|---|---|---|---|
| 50 | 66.287 | 65.408 | +0.879 | +0.88 | ok |
| 60 | 55.716 | 55.221 | +0.495 | +0.50 | differs |

At q = 50 the quoted +0.88 is reproduced; at q = 60 the reproduced difference is +0.495 (+0.4949 to four decimals), which rounds to +0.49, not the quoted +0.50 (a difference in the second decimal). Nothing else about the argument depends on it.

**Item 2, gap / tent slope** (slope = `e2*(0)` / 2q, averaged over the six MS-R2 arms; block `slope`):

| q | mean over the 60 runs | arm means | quoted | match |
|---|---|---|---|---|
| 50 | 4.855 | NL_bb_s1 5.654, NL_bb_s4 5.091, NL_bb_s16 5.568, NL_st_s1 4.335, NL_st_s4 4.469, NL_st_s16 4.012 | 4.85 | ok |
| 60 | 5.333 | NL_bb_s1 4.809, NL_bb_s4 6.173, NL_bb_s16 5.667, NL_st_s1 5.536, NL_st_s4 4.830, NL_st_s16 4.983 | 5.33 | ok |

**Item 3, the additive and the quadrature model at s = 16** (arm means; block `models`):

| gap(s = 16) | bin-balanced q50, q60; stratified q50, q60 (reproduced) | quoted | match |
|---|---|---|---|
| additive | 2.57 / 1.36 / 1.66 / 1.71 | 2.57 / 1.36 / 1.66 / 1.71 | ok |
| quadrature | 3.51 / 1.95 / 2.44 / 2.36 | 3.51 / 1.95 / 2.44 / 2.36 | ok |
| observed | 3.90 / 2.75 / 2.81 / 2.42 | 3.90 / 2.75 / 2.81 / 2.42 | ok |
Quadrature prediction below the observed gap in every cell: True; closer than the additive one in every cell: True; F^2 < 0 flags: ['none', 'none', 'none', 'none'].

All quoted numbers of items 1-3 reproduce except the +0.50 above (+0.495 reproduced). The quadrature prediction is below the observed gap in all four cells, as the preamble says, and closer than the additive one in all four.

## 5. The extended cell: does the current actor close the tip with four times the budget? (block `extended`; `t1`, bin-balanced, 224,000 steps; reported, not gated)

| q | deficit @ 16k | deficit @ 32k | deficit @ 48k | deficit @ 56k | deficit @ 112k | deficit @ 168k | deficit @ 224k |
|---|---|---|---|---|---|---|---|
| 50 | 2.110 [1.145, 9.909] | 1.671 [0.914, 8.596] | 1.514 [0.642, 9.411] | 1.890 [0.816, 8.629] | 1.018 [0.584, 2.035] | 0.969 [0.551, 1.676] | 0.654 [0.508, 1.665] |
| 60 | 5.398 [1.384, 6.918] | 5.092 [1.016, 7.627] | 6.122 [0.816, 8.002] | 5.317 [0.687, 7.050] | 2.498 [0.853, 7.746] | 1.181 [0.612, 6.239] | 0.867 [0.478, 1.411] |

| q | RMSE_pos @ 16k | RMSE_pos @ 32k | RMSE_pos @ 48k | RMSE_pos @ 56k | RMSE_pos @ 112k | RMSE_pos @ 168k | RMSE_pos @ 224k |
|---|---|---|---|---|---|---|---|
| 50 | 0.501 [0.204, 2.967] | 0.394 [0.139, 2.714] | 0.448 [0.126, 2.816] | 0.372 [0.097, 2.670] | 0.223 [0.067, 0.763] | 0.183 [0.071, 0.541] | 0.080 [0.043, 0.365] |
| 60 | 2.114 [0.211, 2.443] | 2.109 [0.171, 2.661] | 2.114 [0.115, 2.396] | 2.136 [0.177, 2.354] | 0.800 [0.140, 2.322] | 0.274 [0.076, 2.100] | 0.120 [0.049, 0.217] |

The median tip deficit of `t1` falls with the budget (1.89 -> 0.65 at q = 50 and 5.32 -> 0.87 at q = 60 between 56,000 and 224,000 steps; the 56,000 checkpoint of the extended cell is at constant LR, so it differs from the 56,000-step cell of 2.1, which ends its LR decay) and RMSE_pos falls with it (0.37 -> 0.08 and 2.14 -> 0.12). At 224,000 steps the medians (0.65 and 0.87) are still above those that `t10` (0.53 and 0.48) and `relu` (0.48 and 0.03) reach at 56,000 steps. The upper ends of the ranges are still 1.67 (q = 50) and 1.41 (q = 60) effort units. This says that, in a supervised fit with exact targets, more training also lowers the deficit of the current actor; it does not say that RL training behaves this way.

## 6. The repository screen next to the PI's sandbox (the values are quoted from the prompt, 3 seeds per cell; not repository evidence; block `sandbox`)

| actor | starts | q = 50: repository median (10 seeds) | q = 50: PI sandbox (3 seeds, quoted) | q = 60: repository median (10 seeds) | q = 60: PI sandbox (3 seeds, quoted) |
|---|---|---|---|---|---|
| t1 | bin-balanced | 1.64 | 1.90 | 6.32 | 1.49 |
| t1 | stratified | 0.71 | 0.76 | 1.17 | 0.76 |
| relu | bin-balanced | 0.48 | 0.49 | 0.03 | 0.04 |
| relu | stratified | 0.08 | 0.08 | 0.02 | 0.03 |
| t10 | bin-balanced | 0.53 | 0.54 | 0.48 | 0.45 |
| t10 | stratified | 0.44 | 0.44 | 0.33 | 0.39 |

| actor | starts | q = 50: repository RMSE_pos median | q = 50: PI sandbox (quoted) | q = 60: repository RMSE_pos median | q = 60: PI sandbox (quoted) |
|---|---|---|---|---|---|
| t1 | bin-balanced | 0.26 | 0.54 | 2.10 | 0.39 |
| relu | bin-balanced | 0.08 | 0.08 | 0.02 | 0.02 |
| t10 | bin-balanced | 0.06 | 0.07 | 0.06 | 0.07 |

| actor | starts | PI sandbox: largest first-layer d-weight (d / B), quoted | q = 50: repository median max abs w | q = 60: repository median max abs w |
|---|---|---|---|---|
| t1 | bin-balanced | 1.7 / 1.7 | 1.30 | 1.16 |
| t1 | stratified | 1.4-1.5 | 1.51 | 1.31 |
| relu | bin-balanced | 0.76-0.92 | 0.86 | 0.79 |
| t10 | bin-balanced | not quoted | 14.64 | 17.82 |

Agreement is close for `relu` and `t10` (the medians differ by at most 0.06 effort units) and for `t1` under stratified starts at q = 50 (0.71 against 0.76). The disagreements are for `t1` under bin-balanced starts: at q = 50 1.64 against 1.90 and, large, at q = 60 6.32 against 1.49 (and RMSE_pos 2.10 against 0.39). The prompt notes that one of the sandbox's three q = 60 seeds left 6.4; here six of ten seeds do (table 2.3), so a three-seed median at q = 60 is fragile for the current actor; the repository value rests on ten seeds. Stratified `t1` at q = 60 is also higher here (1.17 against 0.76). The largest first-layer d-weights are close to the quoted ones for `relu` (0.86 / 0.79 against 0.76-0.92) lower for `t1` under bin-balanced starts (1.30 / 1.16 against 1.7) and close to the quoted range under stratified starts (1.51 / 1.31 against 1.4-1.5).

## 7. What this does and does not show

The screen shows, with the real actor class, the runner's initialisation and a matched optimiser budget, that with exact targets and no RL noise (a) the current actor's median tip deficit is 1.6-6.3 effort units under bin-balanced starts and 0.7-1.2 under stratified starts, with a split distribution over seeds; (b) the `relu` actor and the `t10` actor leave 0.03-0.53 effort units under bin-balanced starts; (c) four times the budget lowers the deficit of the current actor to 0.65-0.87. It does not show that an RL run with sampled rewards and policy noise resolves the tip better with `relu` or `t10`: the prompt's expectation that the RL fit is worse than the supervised one, and the primary criterion, are for the pilot.
