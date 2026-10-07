# MS-R2 P1: the decomposition of the d = 0 gap, reproduced from the MS-R1 data (premise check, section 2.4)

Date: 2026-10-07. Tool `tools/ms/r2_decomposition.py` (formulas in `utils/ms_noise.py`), input `results/ms_r1/analysis/per_run.csv` of `origin/ms-r1` (`71c58904`; rows `role = ms_arm`, status `done`: 140 runs), Freeze arrays of the MS-R1 runs for the Beta recomputation (D8 roots, `00_housekeeping.md`). Outputs `results/ms_r2/decomposition/` (`decomposition_per_run.csv`, `decomposition_table.csv`, `decomposition_paired.csv`, `decomposition_beta_recompute.csv`, `decomposition_checks.json`). Tables below are printed by `python reports/ms/r2/report_scripts/decomposition_tables.py --block <premise|arms|paired|facts>`. The preamble of the prompt is not edited; where a number differs it is reported as such (section 4).

**Result: the premise holds; the stop condition of section 2.4 is not met.** The ratio check passes in all 140 runs (every ratio inside [0.99, 1.01]) and the table of the preamble is reproduced to its second decimal in all six rows.

## 1. The decomposition and the formula

With the closed-form tie effort e*(0) = DW / (4 k q) (70 at q = 50, 58.33 at q = 60), e_hat_2(0) = `e2_at_0` and e_sigma(0) = `smoothed_e_pred_0` (the tie first-order condition of the game both players play with the learned noise, `run/run_v2_T2_locked.py:smoothed_share`, reimplemented without the run object in `utils/ms_noise.py:smoothed_tie_prediction`):

    gap       = e*(0) - e_hat_2(0)  =  [e*(0) - e_sigma(0)]  +  [e_sigma(0) - e_hat_2(0)]
              =  smoothing part     +  remainder.

**Derivation of the smoothing part.** The shock difference of the two players has the triangular density f_xi(x) = (2q - |x|) / (4 q^2) on |x| <= 2q, with its kink at x = 0. With independent centred policy noise n_L, n_O on the efforts (the learned Beta noise at d = 0), the tie first-order condition of the noisy game is DW * E[f_xi(n_L - n_O)] = 2 k e, and, as |n_L - n_O| <= 2q, E[f_xi(n_L - n_O)] = 1 / (2q) - E|n_L - n_O| / (4 q^2). Hence e_sigma(0) = e*(0) (1 - E|n_L - n_O| / (2q)) and the smoothing part is e*(0) * E|n_L - n_O| / (2q). For Gaussian noise of standard deviation sigma, n_L - n_O ~ N(0, 2 sigma^2) and E|n_L - n_O| = 2 sigma / sqrt(pi), so

    smoothing part = e*(0) * sigma_2(0) / (sqrt(pi) * q).

Away from the tip f_xi is linear (slope +-1 / (4 q^2) on each side of 0), so E[f_xi(y + n_L - n_O)] = f_xi(y) for symmetric noise and |y| larger than the noise range: the noise has no first-order effect there, the bias is local to the peak.

## 2. The checks of section 2.4

- runs: 140 (role `ms_arm`, status done); ratio smoothing part / [e2*(0) sigma_2(0) / (sqrt(pi) q)]: min 0.99927, max 0.99971; runs outside [0.99, 1.01]: 0
- table of the preamble reproduced to its second decimal: yes
- learned peak below e2*(0): 139 of 140 runs; median remainder of `MS_s35a5` at q = 50: +0.138
- seed SD of the remainder over the 14 (arm, q) cells: 0.59 to 2.31
- the four sampler arms against `MS_base2400`: largest |change of the smoothing part| 0.034; change of the remainder -1.28 to -0.71 at q = 50 (every interval contains 0: True), -0.44 to +0.04 at q = 60 (True)
- budget (`MS_base2400` - `MS_base`), q = 50: sigma_2(0) -0.385 [-0.425, -0.352]; smoothing part -0.304 [-0.336, -0.278]; remainder -0.575 [-1.714, +0.388]
- budget (`MS_base2400` - `MS_base`), q = 60: sigma_2(0) -0.401 [-0.423, -0.380]; smoothing part -0.220 [-0.232, -0.209]; remainder -0.275 [-0.893, +0.457]
- learned Beta noise at d = 0 (140 runs, alpha and beta from `freeze_stage2_final.npz`): skewness -0.095 to -0.012, excess kurtosis -0.025 to -0.004 (a Gaussian has 0 and 0); e_sigma(0) recomputed from alpha, beta differs from `smoothed_e_pred_0` by at most 3.29e-07; the node standard deviation differs from `sigma_effort_at_0_t2` by at most 0.0053 effort units

How close the learned noise is to Gaussian: the Beta(alpha, beta) of the policy at d = 0 has |skewness| <= 0.095 and |excess kurtosis| <= 0.025 in all 140 runs (concentration alpha + beta between 199 and 427, noise variance 5.4 to 11.8 effort units^2), and the exact quantile computation of e_sigma(0) (400 midpoints per player) gives a smoothing part within 0.03 to 0.07 % of the Gaussian formula in every run (ratio 0.99927 to 0.99971). What produces that shortfall (the skewness, the 400-node discretisation, or the truncation of f_xi at 2q) is not separated here.

## 3. The table of the preamble

Effort units, means over the 10 development seeds.

| q | arm | preamble (sigma, gap, smoothing, remainder) | reproduced (2 dp) | reproduced (4 dp) | match |
|---|---|---|---|---|---|
| 50 | `MS_base` | 2.92, 4.59, 2.30, 2.29 | 2.92, 4.59, 2.30, 2.29 | 2.9179, 4.5917, 2.3035, 2.2882 | yes |
| 50 | `MS_base2400` | 2.53, 3.71, 2.00, 1.71 | 2.53, 3.71, 2.00, 1.71 | 2.5332, 3.7127, 1.9996, 1.7131 | yes |
| 50 | `MS_s35a5` | 2.50, 2.40, 1.97, 0.43 | 2.50, 2.40, 1.97, 0.43 | 2.4954, 2.3993, 1.9697, 0.4296 | yes |
| 60 | `MS_base` | 2.99, 3.11, 1.64, 1.47 | 2.99, 3.11, 1.64, 1.47 | 2.9903, 3.1119, 1.6396, 1.4723 | yes |
| 60 | `MS_base2400` | 2.59, 2.62, 1.42, 1.20 | 2.59, 2.62, 1.42, 1.20 | 2.5889, 2.6169, 1.4193, 1.1976 | yes |
| 60 | `MS_s35a5` | 2.57, 2.54, 1.41, 1.13 | 2.57, 2.54, 1.41, 1.13 | 2.5666, 2.5352, 1.4071, 1.1280 | yes |

All six rows, four columns each, agree with the preamble to the second decimal. The full table, with the other four MS-R1 arms:

| q | arm | sigma_2(0) | gap | smoothing part | remainder | remainder median | remainder seed SD |
|---|---|---|---|---|---|---|---|
| 50 | `MS_base` | 2.92 | 4.59 | 2.30 | 2.29 | 2.08 | 1.86 |
| 50 | `MS_base2400` | 2.53 | 3.71 | 2.00 | 1.71 | 1.60 | 1.99 |
| 50 | `MS_rule` | 2.50 | 3.91 | 1.97 | 1.93 | 1.53 | 2.31 |
| 50 | `MS_s25a0` | 2.52 | 2.77 | 1.99 | 0.78 | 0.85 | 1.03 |
| 50 | `MS_s25a5` | 2.53 | 3.00 | 2.00 | 1.00 | 0.84 | 1.33 |
| 50 | `MS_s35a0` | 2.49 | 2.65 | 1.97 | 0.69 | 0.62 | 1.24 |
| 50 | `MS_s35a5` | 2.50 | 2.40 | 1.97 | 0.43 | 0.14 | 1.07 |
| 60 | `MS_base` | 2.99 | 3.11 | 1.64 | 1.47 | 1.32 | 1.17 |
| 60 | `MS_base2400` | 2.59 | 2.62 | 1.42 | 1.20 | 0.80 | 1.16 |
| 60 | `MS_rule` | 2.66 | 3.22 | 1.46 | 1.76 | 1.72 | 1.10 |
| 60 | `MS_s25a0` | 2.61 | 2.67 | 1.43 | 1.24 | 0.95 | 1.15 |
| 60 | `MS_s25a5` | 2.60 | 2.54 | 1.43 | 1.11 | 0.83 | 0.59 |
| 60 | `MS_s35a0` | 2.57 | 2.16 | 1.41 | 0.76 | 0.87 | 1.12 |
| 60 | `MS_s35a5` | 2.57 | 2.54 | 1.41 | 1.13 | 1.01 | 0.65 |

The paired differences of section 2 of the preamble (budget: `MS_base2400` - `MS_base`; samplers: arm - `MS_base2400`), percentile bootstrap, 10,000 resamples, a fresh `numpy.random.default_rng(20261007)` per call (descriptive):

| comparison | arm - baseline | q | quantity | mean [95 % CI] | interval contains 0 |
|---|---|---|---|---|---|
| budget | `MS_base2400` - `MS_base` | 50 | sigma_effort_at_0_t2 | -0.385 [-0.425, -0.352] | no |
| budget | `MS_base2400` - `MS_base` | 50 | smoothing | -0.304 [-0.336, -0.278] | no |
| budget | `MS_base2400` - `MS_base` | 50 | remainder | -0.575 [-1.714, +0.388] | yes |
| budget | `MS_base2400` - `MS_base` | 50 | gap | -0.879 [-2.045, +0.103] | yes |
| budget | `MS_base2400` - `MS_base` | 60 | sigma_effort_at_0_t2 | -0.401 [-0.423, -0.380] | no |
| budget | `MS_base2400` - `MS_base` | 60 | smoothing | -0.220 [-0.232, -0.209] | no |
| budget | `MS_base2400` - `MS_base` | 60 | remainder | -0.275 [-0.893, +0.457] | yes |
| budget | `MS_base2400` - `MS_base` | 60 | gap | -0.495 [-1.118, +0.241] | yes |
| sampler | `MS_s25a0` - `MS_base2400` | 50 | smoothing | -0.007 [-0.085, +0.068] | yes |
| sampler | `MS_s25a0` - `MS_base2400` | 50 | remainder | -0.935 [-2.182, +0.327] | yes |
| sampler | `MS_s25a0` - `MS_base2400` | 50 | gap | -0.942 [-2.210, +0.338] | yes |
| sampler | `MS_s25a5` - `MS_base2400` | 50 | smoothing | +0.001 [-0.076, +0.078] | yes |
| sampler | `MS_s25a5` - `MS_base2400` | 50 | remainder | -0.709 [-1.702, +0.364] | yes |
| sampler | `MS_s25a5` - `MS_base2400` | 50 | gap | -0.709 [-1.713, +0.347] | yes |
| sampler | `MS_s35a0` - `MS_base2400` | 50 | smoothing | -0.034 [-0.145, +0.063] | yes |
| sampler | `MS_s35a0` - `MS_base2400` | 50 | remainder | -1.028 [-2.642, +0.491] | yes |
| sampler | `MS_s35a0` - `MS_base2400` | 50 | gap | -1.061 [-2.764, +0.494] | yes |
| sampler | `MS_s35a5` - `MS_base2400` | 50 | smoothing | -0.030 [-0.131, +0.063] | yes |
| sampler | `MS_s35a5` - `MS_base2400` | 50 | remainder | -1.283 [-2.632, +0.178] | yes |
| sampler | `MS_s35a5` - `MS_base2400` | 50 | gap | -1.313 [-2.704, +0.149] | yes |
| sampler | `MS_s25a0` - `MS_base2400` | 60 | smoothing | +0.010 [-0.037, +0.064] | yes |
| sampler | `MS_s25a0` - `MS_base2400` | 60 | remainder | +0.045 [-0.768, +0.938] | yes |
| sampler | `MS_s25a0` - `MS_base2400` | 60 | gap | +0.054 [-0.766, +0.946] | yes |
| sampler | `MS_s25a5` - `MS_base2400` | 60 | smoothing | +0.007 [-0.052, +0.067] | yes |
| sampler | `MS_s25a5` - `MS_base2400` | 60 | remainder | -0.089 [-0.769, +0.558] | yes |
| sampler | `MS_s25a5` - `MS_base2400` | 60 | gap | -0.082 [-0.786, +0.587] | yes |
| sampler | `MS_s35a0` - `MS_base2400` | 60 | smoothing | -0.012 [-0.062, +0.042] | yes |
| sampler | `MS_s35a0` - `MS_base2400` | 60 | remainder | -0.441 [-1.311, +0.450] | yes |
| sampler | `MS_s35a0` - `MS_base2400` | 60 | gap | -0.454 [-1.337, +0.451] | yes |
| sampler | `MS_s35a5` - `MS_base2400` | 60 | smoothing | -0.012 [-0.061, +0.034] | yes |
| sampler | `MS_s35a5` - `MS_base2400` | 60 | remainder | -0.070 [-0.877, +0.647] | yes |
| sampler | `MS_s35a5` - `MS_base2400` | 60 | gap | -0.082 [-0.901, +0.652] | yes |

## 4. Where this differs from the preamble

- The largest change of the remainder at q = 60 among the four sampler arms is +0.0445 (`MS_s25a0`), which rounds to +0.04; the preamble has "+0.05". Every other range and every interval statement of the preamble (smoothing part change at most 0.04 in absolute value: 0.034; remainder -0.71 to -1.28 at q = 50 with every interval containing 0; -0.44 to +0.04 at q = 60; budget effect on sigma_2(0) of -0.39 / -0.40; on the smoothing part -0.30 [-0.34, -0.28] and -0.22 [-0.23, -0.21]; seed SD of the remainder 0.6 to 2.3; 139 of 140 learned peaks below e*(0); median remainder +0.14 of `MS_s35a5` at q = 50) is reproduced.
- The preamble's bootstrap is "PI-side"; this tool uses `default_rng(20261007)` freshly per (q, statistic) (the seed of this round's criterion), so interval ends may differ in the third decimal from the PI's.
- Nothing else differs. The analysis does not use the decomposition for any decision; it is the premise of the round and the yardstick of its reporting.
