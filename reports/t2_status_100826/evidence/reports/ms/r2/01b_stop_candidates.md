# MS-R2 P1: three closed-form-free stop criteria for the terminal stage, calibrated offline on MS-R1 (D6, report-only)

Date: 2026-10-07. Tool `tools/ms/r2_stop_candidates.py` (tests `tests/test_ms_r2_stop_candidates.py`), outputs `results/ms_r2/stop_calibration/` (per-run CSVs under `base/` and `pilot/`, `manifest.json`, `validation_summary.csv`, `tables/*.csv`, `tables_nofire.md`, `tables_fire.md`). Every table below is printed by `python reports/ms/r2/report_scripts/stop_tables.py --dir results/ms_r2/stop_calibration --block <spearman|freeze|jcheck|diag|fire|facts>`; the full set (per source and arm) is in the two `.md` files. **Nothing in any run depends on this calibration; the candidates are not part of the pre-registered criterion.** They are descriptive, 140 runs that share the v2.0 pipeline and differ in sampler and budget; the rules were not tuned.

## 1. What was computed

The 140 MS-R1 runs (20 `MS_base`, root `.../ms-r1-multistage-development-afebf2/results/ms_r1/base`; 120 pilot runs of `MS_rule`, `MS_s25a0`, `MS_s25a5`, `MS_s35a0`, `MS_s35a5`, `MS_base2400`, root `.../p2-gate-ms-base2400-e41857/results/ms_r1/pilot`) were replayed export by export (every 25 updates of the terminal stage: 12,800 exports) with the development verifier, as in the MS-R1 calibration. Per export: the D3 quantities (Delta, s, R, R_tail, C) and

- **C1** = R0 = r_2(0)/s_2 = |e_hat_2(0) - a_dev_2(0)| / a_dev_2(0) (MS-R1 A3(a));
- **C2** = c2 = (e_sigma(0) - e_hat_2(0)) / e_sigma(0), with the noise floor sigma_2(0) / (sqrt(pi) q) next to it; the fire rule uses |c2| (column `C2`) and, as a variant, the signed value (`C2s`);
- **C3** = R_defl = max over the non-tail nodes of r_2(d) / (s_2 |1 - J(d)|), J the central finite difference (h = 0.5) of the exact one-step best response against the opponent's mean action shifted by +-h; the best response is the root of the strictly decreasing derivative Q' = DW f_xi(d + e - o) - 2 k e (2k > a = DW / (4 q^2) in both games), solved by bisection to 1e-10 and verified against a dense grid with parabolic refinement (agreement 3.6e-10, `tests/test_ms_r2_stop_candidates.py`). No node was excluded by |1 - J| < 0.1.

The closed-form errors (|peak|, RMSE_pos, tail mean) are columns for reporting. **Validation:** every export coincides with a row of the run's own `ms_checks_stage2.csv`; the replayed Delta, s, R, R_tail, C and the closed-form columns equal the logged values with maximum absolute difference 0 (`validation_summary.csv`); no export was invalid. Tool SHA-256 `deed0e2b71146ac2...` (`manifest.json`).

## 2. Facts

- exports replayed: 12800 valid terminal-stage exports of 140 runs
- q = 50: R0 / |peak| over the exports with u >= 400: median 0.603, 5th-95th percentile [0.588, 0.617] (the linearised value is 2k / (2k + a))
- q = 60: R0 / |peak| over the exports with u >= 400: median 0.685, 5th-95th percentile [0.648, 0.698] (the linearised value is 2k / (2k + a))
- q = 50, at the freeze (70 runs): |c2| below its noise floor sigma_2(0) / (sqrt(pi) q) in 52 runs; c2 < 0 (the learned tie effort above e_sigma(0)) in 20 runs
- q = 60, at the freeze (70 runs): |c2| below its noise floor sigma_2(0) / (sqrt(pi) q) in 46 runs; c2 < 0 (the learned tie effort above e_sigma(0)) in 5 runs

## 3. Correlation with the closed-form errors

Spearman correlation, pooled over the 70 runs of each q and all exports with u >= 400 (and the constant-LR exports separately); within-run medians and the per-arm versions are in `tables_nofire.md`.

| q | candidate | all exports: \|peak\| | all exports: RMSE_pos | constant-LR exports: \|peak\| | constant-LR exports: RMSE_pos |
|---|---|---|---|---|---|
| 50 | C1 | 0.991 | 0.251 | 0.991 | 0.221 |
| 50 | C2 | 0.891 | 0.293 | 0.895 | 0.276 |
| 50 | C2 signed | 0.984 | 0.191 | 0.986 | 0.176 |
| 50 | C3 | 0.856 | 0.427 | 0.867 | 0.384 |
| 50 | R | -0.003 | 0.738 | -0.046 | 0.726 |
| 50 | Delta | 0.201 | 0.803 | 0.166 | 0.791 |
| 60 | C1 | 0.991 | 0.173 | 0.990 | 0.160 |
| 60 | C2 | 0.889 | 0.246 | 0.885 | 0.247 |
| 60 | C2 signed | 0.988 | 0.113 | 0.991 | 0.109 |
| 60 | C3 | 0.675 | 0.580 | 0.687 | 0.556 |
| 60 | R | 0.107 | 0.829 | 0.093 | 0.824 |
| 60 | Delta | 0.256 | 0.847 | 0.245 | 0.842 |

Reading. C1, C2 and C3 rank |peak error| (0.86-0.99 at q = 50, 0.68-0.99 at q = 60) and not RMSE_pos (0.11-0.58); R and Delta, the quantities of the MS-R1 rule, do the opposite (R: -0.00 / 0.11 with |peak|, 0.74 / 0.83 with RMSE_pos). The criteria answer different questions: the stop of MS-R1 read the global accuracy of the policy over the non-tail region, the three candidates read the tie. C1 is |peak| rescaled: R0 / |peak| has median 0.603 (q = 50) and 0.685 (q = 60), the linearised 2k / (2k + a) = 0.588 / 0.673, so C1 includes the noise floor of the learned policy: a policy sitting on e_sigma(0) has C1 of about 0.6 times the floor (0.017 at sigma_2(0) = 2.5, i.e. a floor of 2.8 % of e*(0)). C2 removes the floor: its signed version has Spearman 0.984-0.991 with |peak|.

## 4. Distribution at the freeze (the last terminal-stage export)

| q | quantity | n runs | min | median | max |
|---|---|---|---|---|---|
| 50 | C1 | 70 | 0.0026 | 0.0262 | 0.0757 |
| 50 | C2 (\|c2\|) | 70 | 0.0003 | 0.0200 | 0.0867 |
| 50 | C2 signed | 70 | -0.0254 | 0.0173 | 0.0867 |
| 50 | C2 floor | 70 | 0.0262 | 0.0283 | 0.0389 |
| 50 | C3 | 70 | 0.0268 | 0.0544 | 0.2257 |
| 50 | R | 70 | 0.0508 | 0.1009 | 0.2212 |
| 50 | Delta | 70 | 0.0004 | 0.0011 | 0.0051 |
| 50 | C1 exact | 70 | 0.0015 | 0.0262 | 0.0757 |
| 50 | \|peak\| | 70 | 0.0026 | 0.0438 | 0.1222 |
| 50 | RMSE_pos | 70 | 0.0109 | 0.0208 | 0.0417 |
| 60 | C1 | 70 | 0.0000 | 0.0292 | 0.0644 |
| 60 | C2 (\|c2\|) | 70 | 0.0008 | 0.0184 | 0.0702 |
| 60 | C2 signed | 70 | -0.0234 | 0.0183 | 0.0702 |
| 60 | C2 floor | 70 | 0.0226 | 0.0248 | 0.0305 |
| 60 | C3 | 70 | 0.0269 | 0.0527 | 0.1362 |
| 60 | R | 70 | 0.0374 | 0.0623 | 0.1155 |
| 60 | Delta | 70 | 0.0002 | 0.0006 | 0.0017 |
| 60 | C1 exact | 70 | 0.0003 | 0.0292 | 0.0644 |
| 60 | \|peak\| | 70 | 0.0001 | 0.0428 | 0.0928 |
| 60 | RMSE_pos | 70 | 0.0126 | 0.0200 | 0.0358 |

## 5. Checks of the construction

J against the linearisation J - 1 = -2k / (2k -+ a) (mean over the non-tail nodes with |d| > 8):

| q | side | n exports | mean J | linearised J | mean abs deviation | max abs deviation |
|---|---|---|---|---|---|---|
| 50 | d < -8 | 6400 | -2.2219 | -2.3333 | 1.11e-01 | 1.78e+00 |
| 50 | d > +8 | 6400 | 0.4118 | 0.4118 | 9.58e-16 | 4.22e-15 |
| 60 | d < -8 | 6400 | -0.9213 | -0.9459 | 2.46e-02 | 5.17e-01 |
| 60 | d > +8 | 6400 | 0.3271 | 0.3271 | 7.69e-16 | 3.94e-15 |

On the d > 0 side J equals the linearisation to 1e-15; on the d < 0 side the mean J is 0.11 (q = 50) and 0.02 (q = 60) away on average, up to 1.8 and 0.5 in single early exports where the best response clips at the effort bound or the opponent shift leaves the shock support. Development-tier best-response grid error and the C3 exclusions:

| q | n exports | \|s - s_exact\| median | \|s - s_exact\| max | \|R0 - R0_exact\| max | non-tail nodes | excluded nodes (total) |
|---|---|---|---|---|---|---|
| 50 | 6400 | 2.6e-13 | 0.198 | 0.0029 | 49 | 0 |
| 60 | 6400 | 2.8e-13 | 0.177 | 0.0030 | 59 | 0 |

## 6. When would a rule have fired (grids recorded before the fire tables)

The threshold grids are in `reports/ms/r2/stop_candidate_grids.json`, committed as `93d8be5d` (SHA-256 `ef0a01c41b686a7c...`) before any fire table was computed; they were chosen from the freeze distributions of section 4 (no fire table existed). Rule: candidate <= theta at M = 3 consecutive terminal-stage exports (K = 25); the first fire per run; the closed-form errors at the fire are compared with the same fired runs' values at the end of their run (their terminal-stage freeze, including the LR decay of that run's schedule).

| candidate | theta | q | fired | fire update median [min, max] | \|peak\| at fire | \|peak\| at the end (same runs) | \|peak\| <= 0.05 at fire |
|---|---|---|---|---|---|---|---|
| C1 = R0 | 0.02 | 50 | 45/70 | 1450 [700, 2300] | 0.0217 [0.0016, 0.0334] | 0.0377 [0.0026, 0.1026] | 45/45 |
| C1 = R0 | 0.02 | 60 | 42/70 | 1462 [500, 2300] | 0.0196 [0.0014, 0.0288] | 0.0392 [0.0001, 0.0928] | 42/42 |
| C1 = R0 | 0.03 | 50 | 63/70 | 900 [400, 2275] | 0.0391 [0.0046, 0.0493] | 0.0420 [0.0026, 0.1026] | 63/63 |
| C1 = R0 | 0.03 | 60 | 66/70 | 975 [425, 1925] | 0.0327 [0.0044, 0.0437] | 0.0413 [0.0001, 0.0928] | 66/66 |
| C1 = R0 | 0.04 | 50 | 68/70 | 662 [325, 1375] | 0.0495 [0.0018, 0.0640] | 0.0435 [0.0026, 0.1026] | 37/68 |
| C1 = R0 | 0.04 | 60 | 67/70 | 700 [275, 1800] | 0.0402 [0.0043, 0.0567] | 0.0415 [0.0001, 0.0928] | 51/67 |
| C1 = R0 | 0.05 | 50 | 70/70 | 525 [275, 1075] | 0.0599 [0.0088, 0.0814] | 0.0438 [0.0026, 0.1222] | 20/70 |
| C1 = R0 | 0.05 | 60 | 70/70 | 562 [275, 1475] | 0.0553 [0.0217, 0.0716] | 0.0428 [0.0001, 0.0928] | 30/70 |
| C2 = \|c2\| | 0.005 | 50 | 11/70 | 2025 [575, 2325] | 0.0320 [0.0261, 0.0391] | 0.0377 [0.0039, 0.0745] | 11/11 |
| C2 = \|c2\| | 0.005 | 60 | 14/70 | 1712 [475, 2400] | 0.0290 [0.0200, 0.0371] | 0.0422 [0.0240, 0.0928] | 14/14 |
| C2 = \|c2\| | 0.01 | 50 | 46/70 | 1438 [425, 2400] | 0.0333 [0.0214, 0.0474] | 0.0359 [0.0026, 0.1026] | 46/46 |
| C2 = \|c2\| | 0.01 | 60 | 46/70 | 1100 [475, 2350] | 0.0297 [0.0198, 0.0400] | 0.0429 [0.0001, 0.0928] | 46/46 |
| C2 = \|c2\| | 0.02 | 50 | 65/70 | 825 [325, 2275] | 0.0453 [0.0163, 0.0613] | 0.0430 [0.0026, 0.1026] | 46/65 |
| C2 = \|c2\| | 0.02 | 60 | 66/70 | 888 [275, 2125] | 0.0406 [0.0137, 0.0553] | 0.0413 [0.0001, 0.0928] | 56/66 |
| C2 = \|c2\| | 0.03 | 50 | 68/70 | 612 [325, 1225] | 0.0522 [0.0119, 0.0685] | 0.0435 [0.0026, 0.1222] | 27/68 |
| C2 = \|c2\| | 0.03 | 60 | 68/70 | 600 [275, 1625] | 0.0483 [0.0151, 0.0656] | 0.0421 [0.0001, 0.0928] | 40/68 |
| C3 = R_defl | 0.03 | 50 | 0/70 | - | - | - | 0/0 |
| C3 = R_defl | 0.03 | 60 | 0/70 | - | - | - | 0/0 |
| C3 = R_defl | 0.04 | 50 | 3/70 | 2375 [2075, 2400] | 0.0214 [0.0192, 0.0378] | 0.0229 [0.0214, 0.0377] | 3/3 |
| C3 = R_defl | 0.04 | 60 | 9/70 | 2350 [1500, 2400] | 0.0327 [0.0253, 0.0393] | 0.0357 [0.0001, 0.0550] | 9/9 |
| C3 = R_defl | 0.05 | 50 | 29/70 | 2075 [775, 2400] | 0.0296 [0.0057, 0.0479] | 0.0294 [0.0026, 0.0839] | 29/29 |
| C3 = R_defl | 0.05 | 60 | 41/70 | 1725 [1000, 2400] | 0.0366 [0.0101, 0.0476] | 0.0411 [0.0001, 0.0854] | 41/41 |
| C3 = R_defl | 0.07 | 50 | 65/70 | 1000 [600, 2350] | 0.0418 [0.0057, 0.0601] | 0.0430 [0.0026, 0.1026] | 49/65 |
| C3 = R_defl | 0.07 | 60 | 68/70 | 838 [375, 2250] | 0.0472 [0.0059, 0.0670] | 0.0421 [0.0001, 0.0928] | 42/68 |
| C2s = c2 (signed) | 0.005 | 50 | 53/70 | 1075 [575, 2025] | 0.0271 [0.0016, 0.0443] | 0.0395 [0.0026, 0.1026] | 53/53 |
| C2s = c2 (signed) | 0.005 | 60 | 54/70 | 1038 [450, 2300] | 0.0231 [0.0013, 0.0380] | 0.0413 [0.0001, 0.0928] | 54/54 |
| C2s = c2 (signed) | 0.01 | 50 | 57/70 | 925 [400, 2025] | 0.0317 [0.0016, 0.0487] | 0.0395 [0.0026, 0.1026] | 57/57 |
| C2s = c2 (signed) | 0.01 | 60 | 62/70 | 988 [425, 2300] | 0.0311 [0.0013, 0.0428] | 0.0421 [0.0001, 0.0928] | 62/62 |
| C2s = c2 (signed) | 0.02 | 50 | 65/70 | 750 [325, 2275] | 0.0434 [0.0018, 0.0613] | 0.0430 [0.0026, 0.1026] | 49/65 |
| C2s = c2 (signed) | 0.02 | 60 | 66/70 | 838 [275, 1800] | 0.0393 [0.0044, 0.0553] | 0.0413 [0.0001, 0.0928] | 59/66 |
| C2s = c2 (signed) | 0.03 | 50 | 68/70 | 612 [325, 1175] | 0.0520 [0.0018, 0.0685] | 0.0435 [0.0026, 0.1222] | 29/68 |
| C2s = c2 (signed) | 0.03 | 60 | 68/70 | 600 [275, 1625] | 0.0478 [0.0001, 0.0656] | 0.0421 [0.0001, 0.0928] | 41/68 |

Reading (descriptive). With C1 at theta = 0.03 the rule fires in 63 of 70 runs at q = 50 and 66 of 70 at q = 60, at a median update of 900 / 975 against terminal-stage budgets of 1600-2400, and every fired run has |peak error| <= 0.05 at the fire; the median |peak| at the fire (0.039 / 0.033) is not larger than at the end of the same runs (0.042 / 0.041). At theta = 0.05 the fire comes at medians 525 / 562 and |peak| at the fire (0.060 / 0.055) exceeds the value at the end of the same runs (0.044 / 0.043); at theta = 0.04 this holds at q = 50 only (0.050 against 0.044): for C1 the threshold moves the fire along the learning curve. C2 (|c2|) at theta = 0.01 fires in 46 of 70 at both q (medians 1438 / 1100) with |peak| 0.033 / 0.030 at the fire; C3 fires rarely below theta = 0.05 (29 and 41 of 70 at 0.05, 3 and 9 of 70 at 0.04, none at 0.03) and late (medians 2075 / 1725); R_defl at the freeze has median 0.054 / 0.053. The median RMSE_pos at the fire is 0.027-0.033 for C1 and 0.024-0.030 for C2, above the end-of-run value of the same runs (0.017-0.021) in every C1 and C2 row (C3: 0.016-0.027, not in every row): the tie criteria fire before the global accuracy has converged, consistent with section 3. The runs are the pooled arms of MS-R1 (budgets 1600-2400, different samplers); the fire update is a property of those trajectories, not of the criteria alone.

## 7. What this does not establish

No threshold is selected and none is proposed here. The fire tables do not say that stopping at the fire is better than continuing: the end-of-run values include the LR decay of each arm's schedule, the fire values do not. The MS-R2 pilot repeats this calibration on its 120 runs (section 3.3 of the prompt), where the floor of C2 moves with the concentration scale and C1 is expected to move with it as well.
