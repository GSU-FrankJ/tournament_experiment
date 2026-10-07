# MS-R2: inputs for the decision after the pilot

Date: 2026-10-07. Branch `ms-r2`. This report collects what the 120 runs of `04_pilot.md` and the repeated stop-candidate calibration bear on. It selects nothing and proposes no protocol: the decision on the next round is the PI's (prompt section 5). Every number is read from `results/ms_r2/analysis/` or `results/ms_r2/stop_calibration_pilot/` through the scripts under `reports/ms/r2/report_scripts/` (`pilot_tables.py` blocks `side_by_side`, `directions`, `tie_effort`, `sampler`, `window`, `transmission`, `interaction`; `stop_tables.py` blocks `freeze_arm`, `spearman_arm`, `fire_arm` and the pooled blocks); `04_pilot.md` has the tests and the full tables.

## 1. The six arms side by side (block `side_by_side`; means over ten seeds, medians where marked; effort units for the decomposition)

`rehearsal_v2_0` stands for the `parents_A` candidate (same terminal stage, MS-R1 C-MS1; its row has the decomposition).

| arm | q | mean abs(peak) | abs(peak)<=0.05 | sigma_2(0) | smoothing part | remainder | gap | RMSE_pos/e2*(0) | tail mean/e2*(0) | eta2/DW | R0 (median) | R (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NL_bb_s1 | 50 | 0.0565 | 5/10 | 2.414 | 1.905 | 2.053 | 3.958 | 0.0199 | 0.0064 | 0.00127 | 0.0301 | 0.0916 |
| NL_bb_s1 | 60 | 0.0401 | 8/10 | 2.450 | 1.343 | 0.995 | 2.338 | 0.0176 | 0.0078 | 0.00061 | 0.0247 | 0.0652 |
| NL_bb_s4 | 50 | 0.0509 | 5/10 | 1.255 | 0.990 | 2.574 | 3.564 | 0.0173 | 0.0066 | 0.00112 | 0.0303 | 0.0750 |
| NL_bb_s4 | 60 | 0.0514 | 4/10 | 1.284 | 0.703 | 2.297 | 3.001 | 0.0168 | 0.0080 | 0.00053 | 0.0362 | 0.0573 |
| NL_bb_s16 | 50 | 0.0557 | 3/10 | 0.651 | 0.514 | 3.383 | 3.897 | 0.0186 | 0.0069 | 0.00133 | 0.0324 | 0.0961 |
| NL_bb_s16 | 60 | 0.0472 | 6/10 | 0.665 | 0.365 | 2.390 | 2.755 | 0.0166 | 0.0084 | 0.00055 | 0.0325 | 0.0567 |
| NL_st_s1 | 50 | 0.0434 | 7/10 | 2.378 | 1.877 | 1.158 | 3.035 | 0.0190 | 0.0074 | 0.00123 | 0.0228 | 0.0886 |
| NL_st_s1 | 60 | 0.0461 | 6/10 | 2.438 | 1.337 | 1.355 | 2.691 | 0.0197 | 0.0081 | 0.00060 | 0.0299 | 0.0597 |
| NL_st_s4 | 50 | 0.0447 | 7/10 | 1.240 | 0.978 | 2.150 | 3.128 | 0.0191 | 0.0072 | 0.00090 | 0.0256 | 0.0781 |
| NL_st_s4 | 60 | 0.0402 | 9/10 | 1.271 | 0.697 | 1.651 | 2.348 | 0.0157 | 0.0087 | 0.00037 | 0.0282 | 0.0488 |
| NL_st_s16 | 50 | 0.0401 | 8/10 | 0.641 | 0.506 | 2.303 | 2.809 | 0.0146 | 0.0075 | 0.00070 | 0.0255 | 0.0786 |
| NL_st_s16 | 60 | 0.0415 | 7/10 | 0.655 | 0.359 | 2.064 | 2.422 | 0.0142 | 0.0088 | 0.00043 | 0.0297 | 0.0503 |
| MS_base2400 | 50 | 0.0530 | 5/10 | 2.533 | 2.000 | 1.713 | 3.713 | 0.0215 | 0.0069 | 0.00147 | 0.0298 | 0.0936 |
| MS_base2400 | 60 | 0.0449 | 7/10 | 2.589 | 1.419 | 1.198 | 2.617 | 0.0218 | 0.0085 | 0.00072 | 0.0251 | 0.0715 |
| MS_s35a5 | 50 | 0.0343 | 8/10 | 2.495 | 1.970 | 0.430 | 2.399 | 0.0205 | 0.0078 | 0.00101 | 0.0182 | 0.0945 |
| MS_s35a5 | 60 | 0.0435 | 8/10 | 2.567 | 1.407 | 1.128 | 2.535 | 0.0222 | 0.0089 | 0.00071 | 0.0278 | 0.0567 |
| rehearsal_v2_0 | 50 | 0.0656 | 2/10 | 2.918 | 2.304 | 2.288 | 4.592 | 0.0225 | 0.0081 | 0.00162 | 0.0377 | 0.1005 |
| rehearsal_v2_0 | 60 | 0.0533 | 5/10 | 2.990 | 1.640 | 1.472 | 3.112 | 0.0201 | 0.0097 | 0.00089 | 0.0340 | 0.0764 |

## 2. What the noise landing changed in each component

### 2.1 Direction counts over the eight (arm, q) cells (block `directions`; the change is arm - the same sampler's s = 1 arm, paired by (q, seed))

| metric | cells | mean change < 0 | interval below 0 | interval above 0 | range of the mean changes |
|---|---|---|---|---|---|
| smoothing part | 8 | 8 | 8 | 0 | -1.3916 to -0.6395 |
| remainder | 8 | 0 | 0 | 4 | +0.2962 to +1.3955 |
| gap | 8 | 5 | 0 | 1 | -0.3940 to +0.6629 |
| sigma_2(0) | 8 | 8 | 8 | 0 | -1.7844 to -1.1382 |
| abs(peak error) | 8 | 5 | 0 | 1 | -0.0059 to +0.0114 |
| signed peak error | 8 | 3 | 1 | 0 | -0.0114 to +0.0059 |
| RMSE_pos/e2*(0) | 8 | 7 | 3 | 0 | -0.0054 to +0.0000 |
| tail mean/e2*(0) | 8 | 1 | 0 | 4 | -0.0002 to +0.0007 |
| eta_2/DW | 8 | 7 | 2 | 0 | -0.0005 to +0.0001 |
| R0 | 8 | 5 | 0 | 1 | -0.0041 to +0.0078 |
| R | 8 | 7 | 3 | 0 | -0.0218 to +0.0079 |

- **Policy noise and the smoothing part: as predicted.** sigma_2(0) and the smoothing part fell in all eight cells with all ten seeds lower in each; sigma_2(0) by 1.14-1.78 effort units, the smoothing part by 0.64-1.39 (1.1-2.0 percentage points of `e_2*(0)`). The ratio to the 1/sqrt(s) prediction was 1.01-1.10 per run and the smoothing part equals `e_2*(0) sigma_2(0) / (sqrt(pi) q)` to 0.083 % at worst (`04_pilot.md` section 6).
- **The remainder: it rose in every cell** (+0.30 to +1.40 effort units; interval above 0 in four of eight cells, below 0 in none; at the late-window mean of `04_pilot.md` 4.6, in six of eight).
- **The gap and the peak error: no net change established.** The gap changed by -0.39 to +0.66 (five of eight means negative, no interval below 0, one above); |peak| by -0.0059 to +0.0114 with the same signs. The tie effort of the learned policy followed the target upward only partly (the seed-mean `e_hat_2(0)` rose by 4-54 % of the target's rise in five cells and fell in three) (block `tie_effort`, below): at the freeze `e_sigma(0)` rose by 0.64-1.39 effort units against s = 1 in every cell while the mean `e_hat_2(0)` changed by -0.66 to +0.39 effort units (the change of `e_hat_2(0)` is minus the change of the gap, `e_2*(0)` being fixed).
- **Global accuracy moved the other way, modestly.** RMSE_pos / `e_2*(0)` was lower in seven of eight cells (three intervals below 0, none above; up to -0.0054, against a level of 0.014-0.020); eta_2 / DW lower in seven of eight (two intervals below 0); R lower in seven of eight (three below 0).
- **The tail mean rose slightly** (four intervals above 0, none below; at most +0.0007 of `e_2*(0)`).

### 2.2 The tie effort: the target moved, the policy followed only partly (block `tie_effort`; means over ten seeds)

| arm | q | e*(0) | e_sigma(0) | e_hat_2(0) | e_hat_2(0) seed SD |
|---|---|---|---|---|---|
| NL_bb_s1 | 50 | 70.00 | 68.09 | 66.04 | 1.56 |
| NL_bb_s1 | 60 | 58.33 | 56.99 | 56.00 | 0.91 |
| NL_bb_s4 | 50 | 70.00 | 69.01 | 66.44 | 1.51 |
| NL_bb_s4 | 60 | 58.33 | 57.63 | 55.33 | 0.45 |
| NL_bb_s16 | 50 | 70.00 | 69.49 | 66.10 | 1.37 |
| NL_bb_s16 | 60 | 58.33 | 57.97 | 55.58 | 0.71 |
| NL_st_s1 | 50 | 70.00 | 68.12 | 66.97 | 1.47 |
| NL_st_s1 | 60 | 58.33 | 57.00 | 55.64 | 1.01 |
| NL_st_s4 | 50 | 70.00 | 69.02 | 66.87 | 0.77 |
| NL_st_s4 | 60 | 58.33 | 57.64 | 55.99 | 0.50 |
| NL_st_s16 | 50 | 70.00 | 69.49 | 67.19 | 0.75 |
| NL_st_s16 | 60 | 58.33 | 57.97 | 55.91 | 0.73 |

`e_sigma(0)` is the smoothed target at the freeze (the tie effort of a policy that best responds to the noisy opponent with spread sigma_2(0); `01_decomposition.md`). Against s = 1 it rose by 0.64-1.39 effort units (minus the change of the smoothing part, `04_pilot.md` 4.2: bin-balanced 0.915 and 0.639 at s = 4, 1.392 and 0.978 at s = 16, q = 50 and 60; stratified 0.899 and 0.640, 1.371 and 0.978), while the seed-mean `e_hat_2(0)` moved by (minus the change of the gap, same order) +0.394, -0.663 (bin-balanced, s = 4), +0.061, -0.417 (s = 16), -0.094, +0.344 (stratified, s = 4), +0.226, +0.269 (s = 16). The seed SD of `e_hat_2(0)` at one arm and q (0.45-1.56 effort units) is as large as or larger than these moves. It is lower at s > 1 than at s = 1 in every sampler and q: stratified 1.47 to 0.77 and 0.75 at q = 50, 1.01 to 0.50 and 0.73 at q = 60; bin-balanced 0.91 to 0.45 and 0.71 at q = 60, and only slightly at q = 50 (1.56 to 1.51 and 1.37).

### 2.3 Transmission and interaction (descriptive; `04_pilot.md` 4.3 and 4.4)

| arm | q | n pairs | mean change of the gap | mean change of the smoothing part | ratio [95% CI] | per-seed ratios |
|---|---|---|---|---|---|---|
| `NL_bb_s4` | 50 | 10 | -0.394 | -0.915 | +0.431 [-0.793, +1.753] | 10501:-0.42488;10502:-2.3969;10503:4.05576;10504:-2.33398;10505:-0.708495;10506:0.834965;10507:2.70429;10508:2.99569;10509:-0.46871;10510:-0.679565 |
| `NL_bb_s4` | 60 | 10 | +0.663 | -0.639 | -1.037 [-1.704, -0.300] | 10501:-0.213028;10502:-0.660954;10503:1.36846;10504:-0.483314;10505:-1.92918;10506:-2.05898;10507:-2.55931;10508:-0.46581;10509:-2.14056;10510:-1.45896 |
| `NL_bb_s16` | 50 | 10 | -0.061 | -1.392 | +0.044 [-0.519, +0.662] | 10501:-0.428025;10502:-0.89633;10503:1.21709;10504:0.446876;10505:-0.735523;10506:-0.30843;10507:1.63567;10508:1.26183;10509:-0.670999;10510:-1.2003 |
| `NL_bb_s16` | 60 | 10 | +0.417 | -0.978 | -0.426 [-1.052, +0.222] | 10501:0.584722;10502:0.611987;10503:1.36217;10504:-0.981928;10505:-0.107116;10506:-0.791748;10507:-1.59553;10508:-0.320781;10509:-1.18782;10510:-2.04951 |
| `NL_st_s4` | 50 | 10 | +0.094 | -0.899 | -0.104 [-1.250, +1.006] | 10501:-1.04043;10502:2.27942;10503:-1.15897;10504:-1.73193;10505:-3.98963;10506:1.65301;10507:1.40458;10508:2.13548;10509:-0.418167;10510:-0.351138 |
| `NL_st_s4` | 60 | 10 | -0.344 | -0.640 | +0.537 [-0.486, +1.606] | 10501:-1.8108;10502:-0.558716;10503:2.05595;10504:0.814271;10505:-1.48962;10506:2.30274;10507:3.60391;10508:-1.00569;10509:0.129376;10510:1.41814 |
| `NL_st_s16` | 50 | 10 | -0.226 | -1.371 | +0.165 [-0.405, +0.730] | 10501:0.72767;10502:1.67666;10503:-0.542848;10504:-0.708453;10505:-1.59839;10506:0.705104;10507:0.640353;10508:1.13073;10509:0.0786392;10510:-0.501866 |
| `NL_st_s16` | 60 | 10 | -0.269 | -0.978 | +0.275 [-0.591, +1.151] | 10501:-1.53218;10502:-0.848969;10503:0.847533;10504:1.07688;10505:-1.71986;10506:1.21197;10507:2.4892;10508:-0.614965;10509:-0.302731;10510:2.21463 |

The ratio of the mean change of the gap to the mean change of the smoothing part lies between -1.04 and +0.54 in the eight cells; none is near 1, and the interval excludes 0 only for `NL_bb_s4` at q = 60. The interaction ("does tip weighting let the mean follow the sharper target?") on the remainder is negative in three of four cells (stratified lower) and positive at s = 4, q = 50; only s = 4 at q = 60 excludes 0 (-1.006 [-1.884, -0.202]), and that cell is driven by the bin-balanced arm's rise (`04_pilot.md` 4.4).

## 3. What the sampler adds at each s (block `sampler`; paired `NL_st_s` - `NL_bb_s`)

| s | metric | q=50: mean [95% CI] (st - bb) | q=50: seeds lower | q=60: mean [95% CI] (st - bb) | q=60: seeds lower |
|---|---|---|---|---|---|
| 1 | smoothing part | -0.028 [-0.131, +0.068] | 4/10 | -0.006 [-0.053, +0.037] | 5/10 |
| 1 | remainder | -0.895 [-2.061, +0.262] | 7/10 | +0.360 [-0.505, +1.344] | 6/10 |
| 1 | gap | -0.923 [-2.109, +0.251] | 7/10 | +0.354 [-0.523, +1.346] | 6/10 |
| 1 | abs(peak error) | -0.0132 [-0.0301, +0.0036] | 7/10 | +0.0061 [-0.0090, +0.0231] | 6/10 |
| 1 | RMSE_pos/e2*(0) | -0.0008 [-0.0053, +0.0032] | 3/10 | +0.0021 [-0.0005, +0.0049] | 3/10 |
| 1 | tail mean/e2*(0) | +0.0010 [+0.0000, +0.0020] * | 3/10 | +0.0003 [-0.0004, +0.0011] | 5/10 |
| 4 | smoothing part | -0.012 [-0.063, +0.036] | 5/10 | -0.007 [-0.031, +0.016] | 5/10 |
| 4 | remainder | -0.424 [-1.187, +0.460] | 7/10 | -0.646 [-1.099, -0.211] * | 8/10 |
| 4 | gap | -0.436 [-1.212, +0.443] | 7/10 | -0.653 [-1.104, -0.218] * | 8/10 |
| 4 | abs(peak error) | -0.0062 [-0.0173, +0.0063] | 7/10 | -0.0112 [-0.0189, -0.0037] * | 8/10 |
| 4 | RMSE_pos/e2*(0) | +0.0018 [-0.0024, +0.0065] | 5/10 | -0.0011 [-0.0034, +0.0010] | 5/10 |
| 4 | tail mean/e2*(0) | +0.0005 [-0.0005, +0.0015] | 3/10 | +0.0007 [-0.0002, +0.0016] | 3/10 |
| 16 | smoothing part | -0.008 [-0.040, +0.022] | 4/10 | -0.006 [-0.019, +0.006] | 5/10 |
| 16 | remainder | -1.080 [-2.147, -0.058] * | 7/10 | -0.327 [-1.152, +0.403] | 7/10 |
| 16 | gap | -1.089 [-2.176, -0.042] * | 7/10 | -0.332 [-1.155, +0.395] | 7/10 |
| 16 | abs(peak error) | -0.0156 [-0.0311, -0.0006] * | 7/10 | -0.0057 [-0.0198, +0.0068] | 7/10 |
| 16 | RMSE_pos/e2*(0) | -0.0040 [-0.0083, -0.0002] * | 9/10 | -0.0024 [-0.0048, -0.0002] * | 6/10 |
| 16 | tail mean/e2*(0) | +0.0006 [-0.0004, +0.0017] | 4/10 | +0.0004 [-0.0008, +0.0017] | 4/10 |

At every s the smoothing part does not differ detectably between the samplers. The remainder, gap and |peak| are lower under the stratified sampler in five of six (s, q) cells, with intervals excluding 0 in two cells each (|peak|: s = 4 at q = 60, s = 16 at q = 50); RMSE_pos is lower at s = 16 at both q. At s = 1 no interval excludes 0 for |peak| (-0.0132 [-0.0301, +0.0036] at q = 50, +0.0061 [-0.0090, +0.0231] at q = 60). The tail mean is not lower under the stratified sampler at any s (+0.0003 to +0.0010).

## 4. The arms against the references

- **`parents_A` with MS-R1's criterion** (`04_pilot.md` 7.1): met for `NL_st_s4` and `NL_st_s16` only. `NL_st_s1` is not met (q = 60 interval contains 0), so the two "met" rows cannot be attributed to the noise landing; they confound the 2800-update budget, the stratified sampler and the scale, and the three stratified arms have overlapping intervals.
- **The s = 1 arms against `MS_base2400` and `MS_s35a5`** (`04_pilot.md` 7.2): no detectable change in |peak|; sigma_2(0) lower by 0.12-0.14 effort units, the smoothing part by 0.07-0.09, the tail mean lower (four intervals below 0): 400 more updates at constant LR was followed by a lower sigma_2(0) and tail error and no detectable change of the tie (the pairs differ in the LR schedule, `MS_s35a5` also in its landing and polishing blocks; ten seeds).

## 5. The stop candidates on the 120 pilot runs (D6 repeated; `results/ms_r2/stop_calibration_pilot/`)

The replay matched every logged check row of the 120 runs: 13,440 valid terminal-stage exports, maximum absolute difference 0 (`results/ms_r2/stop_calibration_pilot/validation_summary.csv`). No candidate enters any run. The three candidates are the ones of D6; the thresholds of the fire tables are the grids recorded before the MS-R1 fire tables (`reports/ms/r2/stop_candidate_grids.json`, commit `93d8be5d`), unchanged.

### 5.1 Facts (block `facts`)

- exports replayed: 13440 valid terminal-stage exports of 120 runs
- q = 50: R0 / |peak| over the exports with u >= 400: median 0.603, 5th-95th percentile [0.588, 0.617] (the linearised value is 2k / (2k + a))
- q = 60: R0 / |peak| over the exports with u >= 400: median 0.685, 5th-95th percentile [0.634, 0.698] (the linearised value is 2k / (2k + a))
- q = 50, at the freeze (60 runs): |c2| below its noise floor sigma_2(0) / (sqrt(pi) q) in 14 runs; c2 < 0 (the learned tie effort above e_sigma(0)) in 4 runs
- q = 60, at the freeze (60 runs): |c2| below its noise floor sigma_2(0) / (sqrt(pi) q) in 13 runs; c2 < 0 (the learned tie effort above e_sigma(0)) in 1 runs

R0 / |peak| has the same median as in MS-R1 (0.603 at q = 50, 0.685 at q = 60) and C1 is |peak| rescaled, whatever s.

### 5.2 Correlation with the closed-form errors, pooled over the 120 runs (block `spearman`; MS-R1's 140 runs in `01b_stop_candidates.md` section 3)

| q | candidate | all exports: \|peak\| | all exports: RMSE_pos | constant-LR exports: \|peak\| | constant-LR exports: RMSE_pos |
|---|---|---|---|---|---|
| 50 | C1 | 0.991 | 0.350 | 0.992 | 0.321 |
| 50 | C2 | 0.862 | 0.229 | 0.884 | 0.273 |
| 50 | C2 signed | 0.911 | 0.162 | 0.940 | 0.198 |
| 50 | C3 | 0.876 | 0.505 | 0.887 | 0.455 |
| 50 | R | 0.093 | 0.761 | 0.071 | 0.742 |
| 50 | Delta | 0.304 | 0.829 | 0.289 | 0.811 |
| 60 | C1 | 0.991 | 0.213 | 0.990 | 0.190 |
| 60 | C2 | 0.875 | 0.159 | 0.889 | 0.206 |
| 60 | C2 signed | 0.939 | 0.053 | 0.963 | 0.083 |
| 60 | C3 | 0.713 | 0.602 | 0.728 | 0.558 |
| 60 | R | 0.141 | 0.835 | 0.118 | 0.820 |
| 60 | Delta | 0.291 | 0.850 | 0.282 | 0.832 |

The ranking of |peak| by C1 is unchanged (0.991 at both q). Against MS-R1's 140 runs, for |peak| at q = 50 / 60: C2 is 0.862 / 0.875 (MS-R1: 0.891 / 0.889), C3 0.876 / 0.713 (0.856 / 0.675), the signed C2 0.911 / 0.939 (0.984 / 0.988), R and Delta stay low (R 0.093 / 0.141). The signed C2 is the one that moved. A reading that is an arithmetic consequence of the definitions, not a tested explanation: |peak| = (smoothing part + remainder) / `e_2*(0)` and c2 = remainder / `e_sigma(0)`, so a pooled sample in which the smoothing part varies by a factor of about four between arms within a q (4.8 at q = 50 and 4.3 at q = 60 per run) correlates less with a quantity that excludes it. The per-arm values (block `spearman_arm`) are 0.74-0.94 for C2 and 0.62-0.92 for C3, 0.98-1.00 for C1:

| arm | q | C1 vs \|peak\| | C2 vs \|peak\| | C3 vs \|peak\| | C1 vs RMSE_pos | C2 vs RMSE_pos | C3 vs RMSE_pos |
|---|---|---|---|---|---|---|---|
| NL_bb_s1 | 50 | 0.991 | 0.942 | 0.911 | 0.360 | 0.368 | 0.477 |
| NL_bb_s1 | 60 | 0.984 | 0.920 | 0.769 | 0.216 | 0.273 | 0.559 |
| NL_bb_s4 | 50 | 0.994 | 0.904 | 0.922 | 0.434 | 0.304 | 0.525 |
| NL_bb_s4 | 60 | 0.992 | 0.910 | 0.767 | 0.239 | 0.205 | 0.601 |
| NL_bb_s16 | 50 | 0.987 | 0.851 | 0.925 | 0.426 | 0.200 | 0.509 |
| NL_bb_s16 | 60 | 0.991 | 0.880 | 0.783 | 0.247 | 0.128 | 0.595 |
| NL_st_s1 | 50 | 0.992 | 0.870 | 0.820 | 0.276 | 0.316 | 0.458 |
| NL_st_s1 | 60 | 0.994 | 0.870 | 0.624 | 0.178 | 0.254 | 0.593 |
| NL_st_s4 | 50 | 0.996 | 0.827 | 0.803 | 0.306 | 0.177 | 0.516 |
| NL_st_s4 | 60 | 0.993 | 0.845 | 0.636 | 0.220 | 0.136 | 0.648 |
| NL_st_s16 | 50 | 0.990 | 0.736 | 0.805 | 0.372 | 0.124 | 0.596 |
| NL_st_s16 | 60 | 0.991 | 0.816 | 0.671 | 0.230 | 0.045 | 0.651 |

### 5.3 Distribution at the freeze, per arm: C2 and its noise floor (block `freeze_arm`; median [min, max] over ten seeds)

| arm | q | C1 | C2 (\|c2\|) | C2 floor | C3 | \|peak\| |
|---|---|---|---|---|---|---|
| NL_bb_s1 | 50 | 0.0301 [0.0179, 0.0510] | 0.0223 [0.0052, 0.0591] | 0.0266 [0.0250, 0.0322] | 0.0706 [0.0421, 0.1452] | 0.0502 [0.0301, 0.0837] |
| NL_bb_s1 | 60 | 0.0247 [0.0098, 0.0501] | 0.0139 [0.0068, 0.0498] | 0.0230 [0.0214, 0.0247] | 0.0445 [0.0366, 0.1073] | 0.0362 [0.0153, 0.0727] |
| NL_bb_s4 | 50 | 0.0303 [0.0073, 0.0525] | 0.0369 [0.0025, 0.0704] | 0.0136 [0.0130, 0.0169] | 0.0531 [0.0350, 0.1209] | 0.0504 [0.0125, 0.0861] |
| NL_bb_s4 | 60 | 0.0362 [0.0282, 0.0434] | 0.0416 [0.0298, 0.0509] | 0.0121 [0.0111, 0.0130] | 0.0537 [0.0418, 0.0645] | 0.0528 [0.0413, 0.0632] |
| NL_bb_s16 | 50 | 0.0324 [0.0168, 0.0547] | 0.0474 [0.0206, 0.0815] | 0.0071 [0.0067, 0.0088] | 0.0551 [0.0436, 0.1380] | 0.0538 [0.0283, 0.0896] |
| NL_bb_s16 | 60 | 0.0325 [0.0199, 0.0467] | 0.0417 [0.0229, 0.0619] | 0.0063 [0.0057, 0.0068] | 0.0507 [0.0353, 0.0872] | 0.0476 [0.0293, 0.0679] |
| NL_st_s1 | 50 | 0.0228 [0.0085, 0.0440] | 0.0111 [0.0005, 0.0482] | 0.0264 [0.0252, 0.0290] | 0.0484 [0.0254, 0.1558] | 0.0381 [0.0143, 0.0726] |
| NL_st_s1 | 60 | 0.0299 [0.0175, 0.0537] | 0.0205 [0.0033, 0.0571] | 0.0232 [0.0218, 0.0239] | 0.0503 [0.0297, 0.1098] | 0.0438 [0.0258, 0.0777] |
| NL_st_s4 | 50 | 0.0256 [0.0185, 0.0375] | 0.0297 [0.0170, 0.0492] | 0.0137 [0.0132, 0.0152] | 0.0469 [0.0343, 0.1211] | 0.0428 [0.0310, 0.0621] |
| NL_st_s4 | 60 | 0.0282 [0.0162, 0.0347] | 0.0300 [0.0116, 0.0397] | 0.0121 [0.0113, 0.0125] | 0.0429 [0.0285, 0.0516] | 0.0413 [0.0239, 0.0507] |
| NL_st_s16 | 50 | 0.0255 [0.0138, 0.0332] | 0.0360 [0.0160, 0.0487] | 0.0070 [0.0068, 0.0079] | 0.0450 [0.0234, 0.0779] | 0.0425 [0.0234, 0.0552] |
| NL_st_s16 | 60 | 0.0297 [0.0082, 0.0378] | 0.0379 [0.0070, 0.0491] | 0.0063 [0.0058, 0.0065] | 0.0441 [0.0329, 0.0562] | 0.0435 [0.0133, 0.0552] |

The floor `sigma_2(0) / (sqrt(pi) q)` falls with s as designed (bin-balanced q = 50: 0.027, 0.014, 0.007). The median |c2| does not fall with it; it rises (s = 1, 4, 16; bin-balanced q = 50: 0.0223, 0.0369, 0.0474; q = 60: 0.0139, 0.0416, 0.0417; stratified q = 50: 0.0111, 0.0297, 0.0360; q = 60: 0.0205, 0.0300, 0.0379). At s = 1 the median |c2| is below its median floor in all four (sampler, q) cells; at s = 16 the ratio of the two medians is 5.1-6.7. This is the criterion-level statement of the remainder's growth: C2 measures it, C1 and |peak| do not distinguish it from the floor. C1 moves little with s (medians at s = 1, 4, 16: bin-balanced q = 50 0.0301, 0.0303, 0.0324, q = 60 0.0247, 0.0362, 0.0325; stratified q = 50 0.0228, 0.0256, 0.0255, q = 60 0.0299, 0.0282, 0.0297; at most 0.011 between two s), as |peak| does not.

### 5.4 When would a rule have fired (block `fire`, pooled; MS-R1's pooled table in `01b_stop_candidates.md` section 6)

Rule: candidate <= theta at three consecutive exports (K = 25), first fire per run; closed-form errors at the fire against the same runs' values at the end of their schedule, which includes the LR decay that the fire values do not.

| candidate | theta | q | fired | fire update median [min, max] | \|peak\| at fire | \|peak\| at the end (same runs) | \|peak\| <= 0.05 at fire |
|---|---|---|---|---|---|---|---|
| C1 = R0 | 0.02 | 50 | 44/60 | 1450 [1000, 2600] | 0.0224 [0.0017, 0.0334] | 0.0428 [0.0125, 0.0848] | 44/44 |
| C1 = R0 | 0.02 | 60 | 44/60 | 1475 [575, 2700] | 0.0170 [0.0012, 0.0291] | 0.0419 [0.0133, 0.0777] | 44/44 |
| C1 = R0 | 0.03 | 50 | 56/60 | 975 [675, 2500] | 0.0317 [0.0046, 0.0485] | 0.0441 [0.0125, 0.0848] | 56/56 |
| C1 = R0 | 0.03 | 60 | 57/60 | 1025 [450, 1975] | 0.0328 [0.0026, 0.0436] | 0.0420 [0.0133, 0.0777] | 57/57 |
| C1 = R0 | 0.04 | 50 | 58/60 | 725 [350, 2200] | 0.0497 [0.0154, 0.0640] | 0.0442 [0.0125, 0.0848] | 30/58 |
| C1 = R0 | 0.04 | 60 | 59/60 | 725 [450, 2775] | 0.0380 [0.0146, 0.0567] | 0.0423 [0.0133, 0.0777] | 41/59 |
| C1 = R0 | 0.05 | 50 | 60/60 | 538 [275, 1075] | 0.0595 [0.0267, 0.0790] | 0.0446 [0.0125, 0.0896] | 9/60 |
| C1 = R0 | 0.05 | 60 | 60/60 | 600 [275, 1025] | 0.0417 [0.0199, 0.0705] | 0.0434 [0.0133, 0.0777] | 33/60 |
| C2 = \|c2\| | 0.005 | 50 | 10/60 | 1550 [825, 2775] | 0.0327 [0.0217, 0.0334] | 0.0415 [0.0234, 0.0692] | 10/10 |
| C2 = \|c2\| | 0.005 | 60 | 11/60 | 1025 [900, 2650] | 0.0329 [0.0130, 0.0371] | 0.0419 [0.0153, 0.0777] | 11/11 |
| C2 = \|c2\| | 0.01 | 50 | 37/60 | 1675 [825, 2700] | 0.0337 [0.0098, 0.0474] | 0.0415 [0.0143, 0.0831] | 37/37 |
| C2 = \|c2\| | 0.01 | 60 | 39/60 | 1075 [475, 2725] | 0.0264 [0.0030, 0.0371] | 0.0419 [0.0133, 0.0777] | 39/39 |
| C2 = \|c2\| | 0.02 | 50 | 54/60 | 850 [350, 2625] | 0.0466 [0.0215, 0.0552] | 0.0441 [0.0125, 0.0848] | 39/54 |
| C2 = \|c2\| | 0.02 | 60 | 52/60 | 900 [475, 2725] | 0.0328 [0.0134, 0.0517] | 0.0419 [0.0133, 0.0777] | 43/52 |
| C2 = \|c2\| | 0.03 | 50 | 58/60 | 675 [350, 2275] | 0.0522 [0.0154, 0.0685] | 0.0442 [0.0125, 0.0896] | 22/58 |
| C2 = \|c2\| | 0.03 | 60 | 58/60 | 600 [350, 2775] | 0.0505 [0.0235, 0.0638] | 0.0422 [0.0133, 0.0777] | 28/58 |
| C3 = R_defl | 0.03 | 50 | 0/60 | - | - | - | 0/0 |
| C3 = R_defl | 0.03 | 60 | 0/60 | - | - | - | 0/0 |
| C3 = R_defl | 0.04 | 50 | 11/60 | 2450 [2075, 2750] | 0.0267 [0.0102, 0.0386] | 0.0368 [0.0234, 0.0686] | 11/11 |
| C3 = R_defl | 0.04 | 60 | 9/60 | 2500 [2275, 2800] | 0.0293 [0.0215, 0.0355] | 0.0378 [0.0133, 0.0543] | 9/9 |
| C3 = R_defl | 0.05 | 50 | 41/60 | 2375 [950, 2800] | 0.0340 [0.0161, 0.0481] | 0.0415 [0.0125, 0.0848] | 41/41 |
| C3 = R_defl | 0.05 | 60 | 50/60 | 1825 [1000, 2800] | 0.0348 [0.0169, 0.0490] | 0.0422 [0.0133, 0.0777] | 50/50 |
| C3 = R_defl | 0.07 | 50 | 56/60 | 975 [600, 2300] | 0.0451 [0.0258, 0.0590] | 0.0441 [0.0125, 0.0848] | 41/56 |
| C3 = R_defl | 0.07 | 60 | 58/60 | 925 [375, 2775] | 0.0473 [0.0059, 0.0611] | 0.0422 [0.0133, 0.0777] | 34/58 |
| C2s = c2 (signed) | 0.005 | 50 | 41/60 | 975 [750, 2425] | 0.0286 [0.0046, 0.0366] | 0.0441 [0.0125, 0.0848] | 41/41 |
| C2s = c2 (signed) | 0.005 | 60 | 48/60 | 1012 [450, 2650] | 0.0204 [0.0006, 0.0380] | 0.0419 [0.0133, 0.0777] | 48/48 |
| C2s = c2 (signed) | 0.01 | 50 | 47/60 | 900 [675, 2325] | 0.0334 [0.0046, 0.0474] | 0.0441 [0.0125, 0.0848] | 47/47 |
| C2s = c2 (signed) | 0.01 | 60 | 53/60 | 975 [450, 2125] | 0.0312 [0.0012, 0.0422] | 0.0419 [0.0133, 0.0777] | 53/53 |
| C2s = c2 (signed) | 0.02 | 50 | 54/60 | 825 [350, 2625] | 0.0448 [0.0154, 0.0550] | 0.0441 [0.0125, 0.0848] | 45/54 |
| C2s = c2 (signed) | 0.02 | 60 | 57/60 | 850 [450, 1975] | 0.0371 [0.0026, 0.0517] | 0.0420 [0.0133, 0.0777] | 51/57 |
| C2s = c2 (signed) | 0.03 | 50 | 58/60 | 675 [350, 2275] | 0.0522 [0.0154, 0.0685] | 0.0442 [0.0125, 0.0896] | 22/58 |
| C2s = c2 (signed) | 0.03 | 60 | 58/60 | 600 [350, 2775] | 0.0482 [0.0235, 0.0638] | 0.0422 [0.0133, 0.0777] | 31/58 |

### 5.5 One grid row per candidate, per arm (block `fire_arm`; rows shown for illustration, no threshold is selected)

| candidate | theta | arm | q | fired | fire update median | \|peak\| at fire (median) | \|peak\| at the end, same runs (median) |
|---|---|---|---|---|---|---|---|
| C1 = R0 | 0.03 | NL_bb_s1 | 50 | 9/10 | 1025 | 0.0289 | 0.0450 |
| C1 = R0 | 0.03 | NL_bb_s1 | 60 | 9/10 | 975 | 0.0330 | 0.0362 |
| C1 = R0 | 0.03 | NL_bb_s4 | 50 | 9/10 | 1025 | 0.0322 | 0.0496 |
| C1 = R0 | 0.03 | NL_bb_s4 | 60 | 9/10 | 975 | 0.0330 | 0.0513 |
| C1 = R0 | 0.03 | NL_bb_s16 | 50 | 8/10 | 1000 | 0.0278 | 0.0538 |
| C1 = R0 | 0.03 | NL_bb_s16 | 60 | 9/10 | 975 | 0.0330 | 0.0463 |
| C1 = R0 | 0.03 | NL_st_s1 | 50 | 10/10 | 875 | 0.0326 | 0.0381 |
| C1 = R0 | 0.03 | NL_st_s1 | 60 | 10/10 | 1038 | 0.0320 | 0.0438 |
| C1 = R0 | 0.03 | NL_st_s4 | 50 | 10/10 | 875 | 0.0326 | 0.0428 |
| C1 = R0 | 0.03 | NL_st_s4 | 60 | 10/10 | 1038 | 0.0320 | 0.0413 |
| C1 = R0 | 0.03 | NL_st_s16 | 50 | 10/10 | 875 | 0.0326 | 0.0425 |
| C1 = R0 | 0.03 | NL_st_s16 | 60 | 10/10 | 1038 | 0.0320 | 0.0435 |
| C2 = \|c2\| | 0.01 | NL_bb_s1 | 50 | 5/10 | 1675 | 0.0343 | 0.0357 |
| C2 = \|c2\| | 0.01 | NL_bb_s1 | 60 | 7/10 | 1075 | 0.0253 | 0.0339 |
| C2 = \|c2\| | 0.01 | NL_bb_s4 | 50 | 3/10 | 975 | 0.0366 | 0.0415 |
| C2 = \|c2\| | 0.01 | NL_bb_s4 | 60 | 7/10 | 1075 | 0.0211 | 0.0513 |
| C2 = \|c2\| | 0.01 | NL_bb_s16 | 50 | 4/10 | 1325 | 0.0354 | 0.0569 |
| C2 = \|c2\| | 0.01 | NL_bb_s16 | 60 | 7/10 | 1075 | 0.0211 | 0.0488 |
| C2 = \|c2\| | 0.01 | NL_st_s1 | 50 | 8/10 | 1712 | 0.0336 | 0.0439 |
| C2 = \|c2\| | 0.01 | NL_st_s1 | 60 | 7/10 | 1825 | 0.0312 | 0.0473 |
| C2 = \|c2\| | 0.01 | NL_st_s4 | 50 | 9/10 | 1875 | 0.0334 | 0.0414 |
| C2 = \|c2\| | 0.01 | NL_st_s4 | 60 | 6/10 | 1425 | 0.0261 | 0.0378 |
| C2 = \|c2\| | 0.01 | NL_st_s16 | 50 | 8/10 | 1712 | 0.0336 | 0.0425 |
| C2 = \|c2\| | 0.01 | NL_st_s16 | 60 | 5/10 | 1025 | 0.0312 | 0.0378 |
| C3 = R_defl | 0.05 | NL_bb_s1 | 50 | 2/10 | 2575 | 0.0263 | 0.0427 |
| C3 = R_defl | 0.05 | NL_bb_s1 | 60 | 9/10 | 1725 | 0.0333 | 0.0362 |
| C3 = R_defl | 0.05 | NL_bb_s4 | 50 | 6/10 | 2488 | 0.0284 | 0.0400 |
| C3 = R_defl | 0.05 | NL_bb_s4 | 60 | 9/10 | 1725 | 0.0319 | 0.0513 |
| C3 = R_defl | 0.05 | NL_bb_s16 | 50 | 7/10 | 2400 | 0.0388 | 0.0523 |
| C3 = R_defl | 0.05 | NL_bb_s16 | 60 | 8/10 | 1588 | 0.0390 | 0.0436 |
| C3 = R_defl | 0.05 | NL_st_s1 | 50 | 8/10 | 2250 | 0.0312 | 0.0381 |
| C3 = R_defl | 0.05 | NL_st_s1 | 60 | 6/10 | 2112 | 0.0369 | 0.0495 |
| C3 = R_defl | 0.05 | NL_st_s4 | 50 | 8/10 | 2412 | 0.0341 | 0.0428 |
| C3 = R_defl | 0.05 | NL_st_s4 | 60 | 9/10 | 2275 | 0.0332 | 0.0423 |
| C3 = R_defl | 0.05 | NL_st_s16 | 50 | 10/10 | 2175 | 0.0350 | 0.0425 |
| C3 = R_defl | 0.05 | NL_st_s16 | 60 | 9/10 | 2300 | 0.0402 | 0.0419 |

Reading (descriptive). C1 at theta = 0.03 fires in 8-10 of 10 runs in every arm at median updates 875-1038, with |peak| at the fire 0.028-0.033 and |peak| at the end of the same runs 0.036-0.054: C1 fires before the landing (medians 875-1038 against 2001), where the arms of a sampler are identical by construction, so its fire rows cannot depend on s here. C2 at theta = 0.01 fires in 3-9 of 10 runs per arm and the number moves with s and sampler (bin-balanced q = 50: 5, 3, 4 of 10 for s = 1, 4, 16; stratified q = 50: 8, 9, 8),; c2 (the remainder over `e_sigma(0)`) grows with s but the counts are not monotone in s in every cell (bin-balanced q = 60: 7, 7, 7). C3 at theta = 0.05 fires in 2-10 of 10 per arm, late (medians 1588-2575). The fire tables do not say that stopping at the fire is better than continuing.

## 6. What is not settled

1. **Why the remainder grows with s.** It rose in all eight cells and the PPO diagnostics co-vary with s (KL per update 0.0061, 0.0091, 0.0166 in the decay segment of the bin-balanced q = 50 arms for s = 1, 4, 16; clip fraction 0.057, 0.078, 0.114; advantage SD 0.042, 0.022, 0.012), but the LR was held at the same schedule for every s, so the round cannot tell a step-size mechanism from others. No run in this round varied the LR at fixed s, the batch size or the number of updates at fixed s.
2. **Whether more updates or a slower decay would close the remainder at s > 1.** The seed-mean remainder of the s = 16 arms between the end of the hold (update 2400) and the end of the decay (update 2800, `04_pilot.md` 5.1) fell by 0.15-0.32 effort units in three of the four arms (bin-balanced q = 60: 2.54 to 2.39; stratified q = 50: 2.46 to 2.30, q = 60: 2.39 to 2.06) and rose by 0.59 in the fourth (bin-balanced q = 50: 2.80 to 3.38); the seed SD of the remainder (0.43-1.49 across arms) and the check-to-check variation of up to 1.9 effort units at the six listed updates (2.08 over all consecutive constant-LR checks) at constant LR limit what a single check says. The late-window comparison (`04_pilot.md` 4.6) is post hoc.
3. **How large an effect ten seeds can show.** The primary intervals have half-widths 0.008-0.017 against an expected change from the smoothing part alone of 0.011-0.020: a change of the size the noise floor would give is at the edge of what a ten-seed paired comparison resolves. An interval containing 0 here does not show the absence of an effect.
4. **Whether the samplers differ at s = 1.** Not at |peak| (the intervals contain 0); the stratified sampler's lower remainder and RMSE_pos at s = 16 is one comparison among many (the `sampler` block lists 36 intervals; none of its tests were pre-registered).
5. **The "met" status against `parents_A`** of `NL_st_s4` and `NL_st_s16` is shared with `NL_st_s1` at q = 50 and differs from it at q = 60 by an amount inside `NL_st_s1`'s own interval (`04_pilot.md` 7.1).
6. **Stage 1 and later stages.** All stage-1 gates pass in all 120 runs; whether a noise landing at the terminal stage helps or harms T = 3 was not examined (not in scope).
7. **Fresh seeds** (40501-40520), a lock, a T = 3 run and a second pilot were not run (prompt section 5).
