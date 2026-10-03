### 1a own curvature

Source: `results/v2_pilots/pilot4/analysis/root_game/own_curvature.csv`

| q | tier | fit_half_width | fit_points | BR_minus_e1star | own_curv_over_2k | ev2pp_over_2k | ref_ev2pp_over_2k | slope_implied_by_curv |
|---|---|---|---|---|---|---|---|---|
| 50 | final | 1 | 5 | 0.03256 | -0.4869 | 0.5131 | 0.49 | -1.054 |
| 50 | final | 2 | 9 | 0.0025 | -0.528 | 0.472 | 0.49 | -0.894 |
| 50 | final | 4 | 17 | -0.04676 | -0.5239 | 0.4761 | 0.49 | -0.9088 |
| 50 | fine | 1 | 9 | -0.004796 | -0.5076 | 0.4924 | 0.49 | -0.9699 |
| 50 | fine | 2 | 17 | -0.01149 | -0.5205 | 0.4795 | 0.49 | -0.9212 |
| 50 | fine | 4 | 33 | -0.04506 | -0.5223 | 0.4777 | 0.49 | -0.9146 |
| 60 | final | 1 | 5 | 0.01227 | -0.7871 | 0.2129 | 0.236 | -0.2706 |
| 60 | final | 2 | 9 | -0.01177 | -0.7917 | 0.2083 | 0.236 | -0.2631 |
| 60 | final | 4 | 17 | -0.01861 | -0.7781 | 0.2219 | 0.236 | -0.2853 |
| 60 | fine | 1 | 9 | -0.001071 | -0.768 | 0.232 | 0.236 | -0.302 |
| 60 | fine | 2 | 17 | -0.005135 | -0.7663 | 0.2337 | 0.236 | -0.305 |
| 60 | fine | 4 | 33 | -0.01868 | -0.7693 | 0.2307 | 0.236 | -0.2999 |

### 1a BR slope by h

Source: `results/v2_pilots/pilot4/analysis/root_game/br_slope.csv`

| q | tier | fit_half_width | h=0.25 | h=0.5 | h=1 | h=2 | h=4 | reference |
|---|---|---|---|---|---|---|---|---|
| 50 | final | 1 | -1.057 | -0.8924 | -0.9232 | -0.9269 | -0.8876 | -0.961 |
| 50 | final | 2 | -0.9741 | -0.9243 | -0.9288 | -0.8966 | -0.8585 | -0.961 |
| 50 | final | 4 | -0.9628 | -0.8741 | -0.9113 | -0.8825 | -0.8533 | -0.961 |
| 50 | fine | 1 | -0.9471 | -0.9226 | -0.9248 | -0.8993 | -0.8567 | -0.961 |
| 50 | fine | 2 | -0.9324 | -0.9288 | -0.9224 | -0.8981 | -0.8611 | -0.961 |
| 50 | fine | 4 | -0.9002 | -0.9079 | -0.9021 | -0.8876 | -0.8537 | -0.961 |
| 60 | final | 1 | -0.3322 | -0.332 | -0.2723 | -0.3061 | -0.2792 | -0.309 |
| 60 | final | 2 | -0.2644 | -0.2942 | -0.3138 | -0.3019 | -0.2958 | -0.309 |
| 60 | final | 4 | -0.2928 | -0.2922 | -0.3034 | -0.2944 | -0.2894 | -0.309 |
| 60 | fine | 1 | -0.3108 | -0.2968 | -0.2998 | -0.3041 | -0.2978 | -0.309 |
| 60 | fine | 2 | -0.3022 | -0.3019 | -0.3015 | -0.3004 | -0.2928 | -0.309 |
| 60 | fine | 4 | -0.2965 | -0.3001 | -0.298 | -0.2965 | -0.2926 | -0.309 |

### 1b ACF every-20 record, segment u>=650, centred (median [IQR] over 10 seeds)

Source: `results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_every20.csv`

| q | arm | lag 20 | lag 40 | lag 60 | lag 80 | lag 100 | lag 120 | lag 140 | lag 160 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | mean | 0.59 [0.19, 0.72] | 0.30 [0.17, 0.48] | 0.05 [-0.07, 0.32] | 0.06 [0.02, 0.15] | 0.01 [-0.15, 0.09] | -0.11 [-0.19, 0.00] | -0.19 [-0.27, -0.06] | -0.23 [-0.27, -0.10] |
| 50 | stochastic | 0.48 [0.40, 0.69] | 0.29 [0.13, 0.41] | 0.10 [0.01, 0.26] | 0.02 [-0.02, 0.14] | -0.09 [-0.20, 0.02] | -0.19 [-0.26, -0.09] | -0.20 [-0.28, -0.06] | -0.25 [-0.34, -0.16] |
| 60 | mean | 0.51 [0.41, 0.56] | 0.23 [0.05, 0.27] | -0.06 [-0.21, 0.01] | -0.24 [-0.26, -0.11] | -0.10 [-0.29, -0.01] | -0.11 [-0.22, -0.09] | -0.11 [-0.25, -0.08] | -0.06 [-0.11, 0.00] |
| 60 | stochastic | 0.60 [0.56, 0.67] | 0.28 [0.04, 0.40] | -0.06 [-0.15, 0.24] | -0.08 [-0.40, 0.02] | -0.25 [-0.34, 0.01] | -0.20 [-0.31, -0.09] | -0.18 [-0.27, -0.06] | -0.07 [-0.20, 0.05] |

### 1b ACF every-20 record, segment u>=650, about_e1star (median [IQR] over 10 seeds)

Source: `results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_every20.csv`

| q | arm | lag 20 | lag 40 | lag 60 | lag 80 | lag 100 | lag 120 | lag 140 | lag 160 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | mean | 0.77 [0.57, 0.87] | 0.54 [0.41, 0.78] | 0.44 [0.30, 0.66] | 0.37 [0.29, 0.56] | 0.33 [0.24, 0.47] | 0.29 [0.09, 0.42] | 0.22 [0.03, 0.30] | 0.13 [-0.00, 0.22] |
| 50 | stochastic | 0.82 [0.76, 0.87] | 0.65 [0.60, 0.72] | 0.51 [0.47, 0.57] | 0.38 [0.29, 0.54] | 0.29 [0.17, 0.48] | 0.15 [0.01, 0.42] | 0.12 [0.05, 0.34] | 0.11 [-0.04, 0.25] |
| 60 | mean | 0.71 [0.68, 0.77] | 0.55 [0.32, 0.63] | 0.46 [0.08, 0.57] | 0.29 [0.10, 0.56] | 0.29 [0.08, 0.51] | 0.30 [0.02, 0.46] | 0.24 [0.02, 0.40] | 0.27 [0.02, 0.40] |
| 60 | stochastic | 0.72 [0.65, 0.87] | 0.48 [0.27, 0.74] | 0.28 [-0.05, 0.62] | 0.15 [-0.06, 0.50] | 0.12 [-0.18, 0.44] | 0.06 [-0.17, 0.41] | 0.03 [-0.10, 0.38] | 0.14 [-0.02, 0.34] |

### 1b ACF every-20 record, segment u>=700, centred (median [IQR] over 10 seeds)

Source: `results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_every20.csv`

| q | arm | lag 20 | lag 40 | lag 60 | lag 80 | lag 100 | lag 120 | lag 140 | lag 160 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | mean | 0.55 [0.23, 0.70] | 0.22 [0.10, 0.39] | 0.07 [-0.15, 0.16] | 0.02 [-0.03, 0.12] | -0.07 [-0.16, 0.04] | -0.14 [-0.17, -0.07] | -0.17 [-0.29, -0.03] | -0.24 [-0.31, -0.01] |
| 50 | stochastic | 0.44 [0.39, 0.65] | 0.24 [0.14, 0.44] | 0.18 [-0.00, 0.21] | -0.01 [-0.06, 0.05] | -0.11 [-0.17, -0.01] | -0.14 [-0.21, -0.09] | -0.23 [-0.32, -0.08] | -0.27 [-0.40, -0.19] |
| 60 | mean | 0.51 [0.39, 0.58] | 0.23 [0.04, 0.26] | -0.07 [-0.23, 0.03] | -0.25 [-0.32, -0.11] | -0.07 [-0.29, -0.01] | -0.09 [-0.21, -0.02] | -0.07 [-0.18, -0.03] | -0.05 [-0.11, -0.02] |
| 60 | stochastic | 0.59 [0.48, 0.62] | 0.30 [0.13, 0.39] | -0.02 [-0.15, 0.28] | -0.05 [-0.33, 0.06] | -0.24 [-0.37, -0.07] | -0.21 [-0.33, -0.13] | -0.16 [-0.27, -0.06] | -0.12 [-0.23, 0.07] |

### 1b ACF every-20 record, segment u>=700, about_e1star (median [IQR] over 10 seeds)

Source: `results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_every20.csv`

| q | arm | lag 20 | lag 40 | lag 60 | lag 80 | lag 100 | lag 120 | lag 140 | lag 160 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | mean | 0.76 [0.62, 0.87] | 0.58 [0.40, 0.75] | 0.43 [0.29, 0.62] | 0.33 [0.23, 0.55] | 0.28 [0.24, 0.40] | 0.25 [0.12, 0.35] | 0.19 [0.12, 0.31] | 0.14 [0.10, 0.24] |
| 50 | stochastic | 0.79 [0.74, 0.85] | 0.62 [0.60, 0.71] | 0.46 [0.40, 0.55] | 0.29 [0.24, 0.51] | 0.25 [0.12, 0.41] | 0.12 [0.01, 0.35] | 0.10 [0.03, 0.29] | 0.11 [-0.06, 0.25] |
| 60 | mean | 0.76 [0.70, 0.84] | 0.58 [0.41, 0.69] | 0.48 [0.11, 0.60] | 0.34 [0.09, 0.54] | 0.32 [0.07, 0.52] | 0.35 [-0.01, 0.47] | 0.26 [0.04, 0.42] | 0.26 [0.01, 0.37] |
| 60 | stochastic | 0.71 [0.60, 0.81] | 0.53 [0.20, 0.67] | 0.32 [0.05, 0.53] | 0.24 [0.01, 0.42] | 0.17 [-0.11, 0.41] | 0.08 [-0.13, 0.39] | 0.14 [-0.12, 0.33] | 0.19 [0.07, 0.24] |

### 1b ACF of the 25-update exports, centred

Source: `results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_exports25.csv`

| segment_start | q | arm | lag 25 | lag 50 | lag 75 | lag 100 |
|---|---|---|---|---|---|---|
| 650 | 50 | mean | 0.50 [0.35, 0.63] | 0.30 [0.24, 0.36] | 0.00 [-0.14, 0.12] | -0.02 [-0.13, 0.13] |
| 650 | 50 | stochastic | 0.46 [0.30, 0.59] | 0.25 [-0.06, 0.32] | 0.04 [-0.11, 0.16] | -0.15 [-0.23, -0.03] |
| 650 | 60 | mean | 0.44 [0.32, 0.60] | 0.11 [0.00, 0.12] | -0.15 [-0.25, -0.08] | -0.13 [-0.16, -0.06] |
| 650 | 60 | stochastic | 0.52 [0.44, 0.56] | 0.11 [-0.03, 0.30] | -0.08 [-0.38, 0.06] | -0.22 [-0.32, -0.01] |
| 700 | 50 | mean | 0.42 [0.15, 0.62] | 0.19 [0.05, 0.30] | -0.08 [-0.25, 0.02] | -0.09 [-0.22, 0.03] |
| 700 | 50 | stochastic | 0.44 [0.35, 0.60] | 0.20 [0.06, 0.32] | 0.05 [-0.14, 0.12] | -0.12 [-0.28, -0.03] |
| 700 | 60 | mean | 0.38 [0.35, 0.63] | 0.09 [-0.03, 0.20] | -0.12 [-0.31, 0.02] | -0.11 [-0.16, -0.02] |
| 700 | 60 | stochastic | 0.51 [0.36, 0.53] | 0.13 [-0.02, 0.27] | -0.03 [-0.22, 0.07] | -0.19 [-0.34, -0.03] |

### 1b window regression

Source: `results/v2_pilots/pilot4/analysis/fluctuation/window_regression.csv`

| segment_start | q | arm | n_windows | n_runs | slope | boot_ci95_lo | boot_ci95_hi | per_run_slope_median | per_run_slope_q25 | per_run_slope_q75 |
|---|---|---|---|---|---|---|---|---|---|---|
| 650 | 50 | mean | 170 | 10 | -0.1537 | -0.2937 | -0.07901 | -0.3597 | -0.7876 | -0.1581 |
| 650 | 50 | stochastic | 170 | 10 | -0.1457 | -0.236 | -0.1189 | -0.3964 | -0.5056 | -0.2987 |
| 650 | 60 | mean | 170 | 10 | -0.198 | -0.3171 | -0.133 | -0.4153 | -0.4345 | -0.3702 |
| 650 | 60 | stochastic | 170 | 10 | -0.1999 | -0.287 | -0.1617 | -0.3279 | -0.3691 | -0.2669 |
| 650 | 50 | both | 340 | 20 | -0.149 | -0.2196 | -0.1068 | -0.3964 | -0.6047 | -0.2102 |
| 650 | 60 | both | 340 | 20 | -0.1965 | -0.2638 | -0.1503 | -0.3686 | -0.4173 | -0.3081 |
| 700 | 50 | mean | 150 | 10 | -0.1358 | -0.3082 | -0.05948 | -0.4324 | -0.7476 | -0.1558 |
| 700 | 50 | stochastic | 150 | 10 | -0.1428 | -0.2261 | -0.09964 | -0.3681 | -0.5404 | -0.2423 |
| 700 | 60 | mean | 150 | 10 | -0.1675 | -0.2795 | -0.1059 | -0.3867 | -0.4098 | -0.3207 |
| 700 | 60 | stochastic | 150 | 10 | -0.2355 | -0.3224 | -0.1524 | -0.3283 | -0.3938 | -0.2541 |
| 700 | 50 | both | 300 | 20 | -0.1386 | -0.2143 | -0.09393 | -0.3833 | -0.5933 | -0.2372 |
| 700 | 60 | both | 300 | 20 | -0.2033 | -0.2707 | -0.1485 | -0.3754 | -0.4032 | -0.2585 |

### 1c Phase A extension u1600, stage-2 candidates (medians over 10 seeds)

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv`

| q | arm | K | stage2_peak_rel_err_signed | stage2_peak_rel_err_abs | stage2_peak_locfree_rel_err | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | stage2_tail_max | stage2_tail_mean_over_g2_0 | stage2_tail_max_over_g2_0 | stage2_sym_err_max | eta_T_over_dw | DeltaT_over_dw_on_max | DeltaT_over_dw_off_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | ext | 1 | -0.06835 | 0.06835 | -0.06826 | 0.02605 | 0.515 | 2.22 | 0.007358 | 0.03172 | 2.638 | 0.001469 | 0.001469 | 0.0003395 |
| 50 | ext | 4 | -0.06294 | 0.06294 | -0.06247 | 0.0201 | 0.5552 | 2.334 | 0.007931 | 0.03334 | 1.809 | 0.001168 | 0.001168 | 0.0003489 |
| 50 | ext | 8 | -0.07022 | 0.07022 | -0.06972 | 0.01917 | 0.5619 | 2.34 | 0.008027 | 0.03343 | 1.444 | 0.001285 | 0.001285 | 0.0003247 |
| 50 | ext | 16 | -0.0667 | 0.0667 | -0.06667 | 0.01669 | 0.5778 | 2.436 | 0.008255 | 0.03479 | 1.426 | 0.0009973 | 0.0009973 | 0.0003602 |
| 60 | ext | 1 | -0.06003 | 0.06003 | -0.05585 | 0.02389 | 0.5312 | 1.901 | 0.009106 | 0.03259 | 2.39 | 0.0009012 | 0.0009012 | 0.0002439 |
| 60 | ext | 4 | -0.06243 | 0.06243 | -0.06217 | 0.0179 | 0.5404 | 2.064 | 0.009264 | 0.03538 | 1.165 | 0.0007207 | 0.0007207 | 0.0002679 |
| 60 | ext | 8 | -0.05595 | 0.05595 | -0.0551 | 0.01711 | 0.551 | 2.065 | 0.009446 | 0.0354 | 1.453 | 0.0006428 | 0.0006428 | 0.0002787 |
| 60 | ext | 16 | -0.05869 | 0.05869 | -0.05825 | 0.01635 | 0.5782 | 2.029 | 0.009911 | 0.03479 | 1.01 | 0.0005634 | 0.0005634 | 0.0002893 |

### 1c Pilot 3 u1000, stage-1 candidates (medians over 10 seeds)

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv`

| q | arm | K | e1_cand | stage1_rel_err_signed | stage1_rel_err_abs | learning_rel | inherited_rel | Gmax_full_over_dw | EXP_root_over_dw | dReach_over_dw |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | mean | 1 | 45.34 | -0.02835 | 0.06763 | -0.02974 | 0.001393 | 0.00487 | 0.002336 | 0.006458 |
| 50 | mean | 4 | 45.28 | -0.02979 | 0.03462 | -0.02829 | 0.001393 | 0.004812 | 0.00197 | 0.005732 |
| 50 | mean | 8 | 45.2 | -0.03138 | 0.03943 | -0.03448 | 0.001393 | 0.004606 | 0.001781 | 0.005545 |
| 50 | mean | 12 | 45.23 | -0.03069 | 0.03434 | -0.03683 | 0.001393 | 0.004606 | 0.001635 | 0.005673 |
| 50 | stochastic | 1 | 47.58 | 0.01962 | 0.06479 | 0.01553 | 0.001393 | 0.004606 | 0.001705 | 0.005795 |
| 50 | stochastic | 4 | 47.18 | 0.01107 | 0.04979 | -0.001145 | 0.001393 | 0.004606 | 0.001726 | 0.006225 |
| 50 | stochastic | 8 | 46.7 | 0.0007315 | 0.04744 | 0.001482 | 0.001393 | 0.004606 | 0.001882 | 0.005691 |
| 50 | stochastic | 12 | 45.64 | -0.02197 | 0.0532 | -0.01993 | 0.001393 | 0.004606 | 0.001943 | 0.005769 |
| 60 | mean | 1 | 37.98 | -0.02346 | 0.08852 | -0.02655 | 0.0006429 | 0.002321 | 0.001452 | 0.003369 |
| 60 | mean | 4 | 40.06 | 0.03012 | 0.05756 | 0.03038 | 0.0006429 | 0.002043 | 0.0009127 | 0.00252 |
| 60 | mean | 8 | 40.9 | 0.05184 | 0.06137 | 0.05171 | 0.0006429 | 0.001833 | 0.0009411 | 0.002687 |
| 60 | mean | 12 | 40.52 | 0.04187 | 0.05933 | 0.04175 | 0.0006429 | 0.001833 | 0.0008846 | 0.002341 |
| 60 | stochastic | 1 | 39.27 | 0.009708 | 0.04948 | 0.007651 | 0.0006429 | 0.001833 | 0.0007316 | 0.002383 |
| 60 | stochastic | 4 | 39.95 | 0.0272 | 0.04061 | 0.02391 | 0.0006429 | 0.001833 | 0.0006055 | 0.002162 |
| 60 | stochastic | 8 | 40.53 | 0.0421 | 0.0421 | 0.04443 | 0.0006429 | 0.001833 | 0.0006545 | 0.002261 |
| 60 | stochastic | 12 | 40.12 | 0.03167 | 0.03167 | 0.02961 | 0.0006429 | 0.001833 | 0.0006495 | 0.002472 |

### 2a end-of-phase candidates u1600 (medians over 10 seeds)

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv`

| q | arm | K | stage2_peak_rel_err_signed | stage2_peak_rel_err_abs | stage2_peak_locfree_rel_err | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | stage2_tail_max | stage2_tail_mean_over_g2_0 | stage2_tail_max_over_g2_0 | stage2_sym_err_max | eta_T_over_dw | DeltaT_over_dw_on_max | DeltaT_over_dw_off_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | constant | 1 | -0.06835 | 0.06835 | -0.06826 | 0.02605 | 0.515 | 2.22 | 0.007358 | 0.03172 | 2.638 | 0.001469 | 0.001469 | 0.0003395 |
| 50 | constant | 4 | -0.06294 | 0.06294 | -0.06247 | 0.0201 | 0.5552 | 2.334 | 0.007931 | 0.03334 | 1.809 | 0.001168 | 0.001168 | 0.0003489 |
| 50 | constant | 8 | -0.07022 | 0.07022 | -0.06972 | 0.01917 | 0.5619 | 2.34 | 0.008027 | 0.03343 | 1.444 | 0.001285 | 0.001285 | 0.0003247 |
| 50 | decay | 1 | -0.0625 | 0.0625 | -0.06243 | 0.02007 | 0.5622 | 2.584 | 0.008032 | 0.03692 | 2.265 | 0.001082 | 0.001082 | 0.0004454 |
| 50 | decay | 4 | -0.06911 | 0.06911 | -0.06831 | 0.01735 | 0.5905 | 2.616 | 0.008436 | 0.03737 | 1.622 | 0.001036 | 0.001036 | 0.0004245 |
| 50 | decay | 8 | -0.06673 | 0.06673 | -0.06634 | 0.01713 | 0.5991 | 2.501 | 0.008559 | 0.03572 | 1.583 | 0.001126 | 0.001126 | 0.0003807 |
| 60 | constant | 1 | -0.06003 | 0.06003 | -0.05585 | 0.02389 | 0.5312 | 1.901 | 0.009106 | 0.03259 | 2.39 | 0.0009012 | 0.0009012 | 0.0002439 |
| 60 | constant | 4 | -0.06243 | 0.06243 | -0.06217 | 0.0179 | 0.5404 | 2.064 | 0.009264 | 0.03538 | 1.165 | 0.0007207 | 0.0007207 | 0.0002679 |
| 60 | constant | 8 | -0.05595 | 0.05595 | -0.0551 | 0.01711 | 0.551 | 2.065 | 0.009446 | 0.0354 | 1.453 | 0.0006428 | 0.0006428 | 0.0002787 |
| 60 | decay | 1 | -0.04971 | 0.04971 | -0.04943 | 0.01982 | 0.5479 | 2.111 | 0.009393 | 0.03619 | 1.643 | 0.0008081 | 0.0008081 | 0.0003076 |
| 60 | decay | 4 | -0.05929 | 0.05929 | -0.059 | 0.01774 | 0.5726 | 2.122 | 0.009815 | 0.03637 | 1.455 | 0.0006526 | 0.0006526 | 0.0002998 |
| 60 | decay | 8 | -0.05884 | 0.05884 | -0.05858 | 0.01627 | 0.5719 | 2.122 | 0.009804 | 0.03638 | 1.557 | 0.0005716 | 0.0005716 | 0.0003144 |

### 2b end-of-phase candidates u2200 (medians over 10 seeds)

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv`

| q | arm | K | e1_cand | stage1_rel_err_signed | stage1_rel_err_abs | learning_rel | inherited_rel | Gmax_full_over_dw | EXP_root_over_dw | dReach_over_dw |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | constant | 1 | 48.95 | 0.04889 | 0.06281 | 0.0544 | -0.003 | 0.002282 | 0.00146 | 0.003414 |
| 50 | constant | 4 | 49.17 | 0.05367 | 0.05367 | 0.05365 | -0.003 | 0.002805 | 0.001366 | 0.003098 |
| 50 | constant | 8 | 48.51 | 0.03946 | 0.04398 | 0.03715 | -0.003 | 0.002482 | 0.00114 | 0.00318 |
| 50 | constant | 12 | 49.39 | 0.05844 | 0.06485 | 0.05622 | -0.003 | 0.002623 | 0.001875 | 0.003559 |
| 50 | decay | 1 | 47.59 | 0.01989 | 0.04317 | 0.01167 | -0.003 | 0.001508 | 0.00113 | 0.002115 |
| 50 | decay | 4 | 46.78 | 0.002392 | 0.04553 | 0.01113 | -0.003 | 0.001933 | 0.0009912 | 0.0025 |
| 50 | decay | 8 | 46.13 | -0.01146 | 0.0356 | -0.001815 | -0.003 | 0.001874 | 0.0008426 | 0.002479 |
| 50 | decay | 12 | 46.61 | -0.00132 | 0.02476 | 0.008322 | -0.003 | 0.001564 | 0.0007759 | 0.002187 |
| 60 | constant | 1 | 40.3 | 0.0362 | 0.05486 | 0.0425 | -0.003343 | 0.0009536 | 0.0008544 | 0.001534 |
| 60 | constant | 4 | 40.48 | 0.04084 | 0.07151 | 0.0447 | -0.003343 | 0.001256 | 0.001142 | 0.001861 |
| 60 | constant | 8 | 41.31 | 0.06228 | 0.07859 | 0.06794 | -0.003343 | 0.001424 | 0.001339 | 0.002124 |
| 60 | constant | 12 | 41.43 | 0.06527 | 0.08221 | 0.07297 | -0.003343 | 0.001187 | 0.001049 | 0.001701 |
| 60 | decay | 1 | 38.82 | -0.00188 | 0.03381 | 0.005062 | -0.003343 | 0.0009012 | 0.0003763 | 0.001035 |
| 60 | decay | 4 | 38.94 | 0.001275 | 0.0374 | 0.003975 | -0.003343 | 0.0009012 | 0.0003822 | 0.001125 |
| 60 | decay | 8 | 39.07 | 0.004768 | 0.04758 | 0.007211 | -0.003343 | 0.0009101 | 0.0006352 | 0.001155 |
| 60 | decay | 12 | 39.25 | 0.009381 | 0.04318 | 0.01544 | -0.003343 | 0.000971 | 0.0006163 | 0.001288 |

### 1c/2b location (t*, d*) of Gmax_full

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv`

| family | q | arm | K | t*=1 count | t*=2 count | d* values |
|---|---|---|---|---|---|---|
| 2b | 50 | constant | 1 | 3 | 7 | -60 -40 -32 -24 -16 0 32 |
| 2b | 50 | constant | 4 | 4 | 6 | -32 -24 -16 -4 0 32 |
| 2b | 50 | constant | 8 | 4 | 6 | -32 -24 -16 -4 0 32 |
| 2b | 50 | constant | 12 | 4 | 6 | -32 -24 -16 -4 0 32 |
| 2b | 50 | decay | 1 | 2 | 8 | -40 -32 -24 -16 -4 0 32 |
| 2b | 50 | decay | 4 | 2 | 8 | -40 -32 -24 -16 -4 0 32 |
| 2b | 50 | decay | 8 | 2 | 8 | -40 -32 -24 -16 -4 0 32 |
| 2b | 50 | decay | 12 | 3 | 7 | -40 -32 -24 -16 -4 0 |
| 2b | 60 | constant | 1 | 5 | 5 | -20 -16 0 |
| 2b | 60 | constant | 4 | 6 | 4 | -20 -16 0 |
| 2b | 60 | constant | 8 | 6 | 4 | -20 -16 0 |
| 2b | 60 | constant | 12 | 4 | 6 | -28 -20 -16 0 |
| 2b | 60 | decay | 1 | 2 | 8 | -28 -20 -16 -4 0 |
| 2b | 60 | decay | 4 | 1 | 9 | -28 -20 -16 -4 0 |
| 2b | 60 | decay | 8 | 2 | 8 | -28 -20 -16 -4 0 |
| 2b | 60 | decay | 12 | 4 | 6 | -28 -20 -16 -4 0 |
| p3 | 50 | mean | 1 | 1 | 9 | -4 0 |
| p3 | 50 | mean | 4 | 2 | 8 | -4 0 |
| p3 | 50 | mean | 8 | 1 | 9 | -4 0 |
| p3 | 50 | mean | 12 | 0 | 10 | -4 100 |
| p3 | 50 | stochastic | 1 | 1 | 9 | -4 0 |
| p3 | 50 | stochastic | 4 | 1 | 9 | -4 0 100 |
| p3 | 50 | stochastic | 8 | 0 | 10 | -4 100 |
| p3 | 50 | stochastic | 12 | 0 | 10 | -4 100 |
| p3 | 60 | mean | 1 | 5 | 5 | -4 0 |
| p3 | 60 | mean | 4 | 2 | 8 | -32 -4 0 40 |
| p3 | 60 | mean | 8 | 0 | 10 | -32 -4 40 |
| p3 | 60 | mean | 12 | 0 | 10 | -32 -4 40 |
| p3 | 60 | stochastic | 1 | 0 | 10 | -32 -4 40 |
| p3 | 60 | stochastic | 4 | 0 | 10 | -32 -4 40 |
| p3 | 60 | stochastic | 8 | 0 | 10 | -32 -4 40 |
| p3 | 60 | stochastic | 12 | 0 | 10 | -32 -4 40 |

### Paired K vs K=1

Source: `results/v2_pilots/pilot4/analysis/paired_summary.csv`

| family | q | a | b | metric | median | n_better | n_neg | n_pos | n_zero | boot_ci95_lo | boot_ci95_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ext | 50 | ext K=4 | ext K=1 | stage2_peak_rel_err_signed | -0.002328 | — | 6 | 4 | 0 | -0.01162 | 0.0191 |
| ext | 50 | ext K=4 | ext K=1 | stage2_peak_rel_err_abs | 0.002328 | 4 | 4 | 6 | 0 | -0.01962 | 0.01169 |
| ext | 50 | ext K=4 | ext K=1 | stage2_peak_locfree_rel_err | -0.003148 | — | 6 | 4 | 0 | -0.01141 | 0.01964 |
| ext | 50 | ext K=4 | ext K=1 | stage2_peak_locfree_rel_err_abs | 0.003148 | 4 | 4 | 6 | 0 | -0.01999 | 0.0114 |
| ext | 50 | ext K=4 | ext K=1 | stage2_rmse_pos_over_g2_0 | -0.002994 | 8 | 8 | 2 | 0 | -0.01513 | -0.001532 |
| ext | 50 | ext K=4 | ext K=1 | stage2_tail_mean | 0.01792 | 1 | 1 | 9 | 0 | 0.004275 | 0.03488 |
| ext | 50 | ext K=4 | ext K=1 | stage2_tail_max | 0.114 | 4 | 4 | 6 | 0 | -0.2691 | 0.1441 |
| ext | 50 | ext K=4 | ext K=1 | stage2_tail_mean_over_g2_0 | 0.0002559 | 1 | 1 | 9 | 0 | 6.401e-05 | 0.0004936 |
| ext | 50 | ext K=4 | ext K=1 | stage2_tail_max_over_g2_0 | 0.001628 | 4 | 4 | 6 | 0 | -0.003921 | 0.002019 |
| ext | 50 | ext K=4 | ext K=1 | stage2_sym_err_max | -0.4989 | 7 | 7 | 3 | 0 | -1.886 | -0.1045 |
| ext | 50 | ext K=4 | ext K=1 | eta_T_over_dw | -0.0003245 | 8 | 8 | 2 | 0 | -0.002218 | 0.0002004 |
| ext | 50 | ext K=4 | ext K=1 | DeltaT_over_dw_on_max | -0.0003245 | 8 | 8 | 2 | 0 | -0.002207 | 0.0001858 |
| ext | 50 | ext K=4 | ext K=1 | DeltaT_over_dw_off_max | 2.497e-05 | 3 | 3 | 7 | 0 | -0.0001233 | 4.7e-05 |
| ext | 60 | ext K=4 | ext K=1 | stage2_peak_rel_err_signed | -0.007549 | — | 6 | 4 | 0 | -0.02033 | 0.002442 |
| ext | 60 | ext K=4 | ext K=1 | stage2_peak_rel_err_abs | 0.007549 | 4 | 4 | 6 | 0 | -0.002239 | 0.01907 |
| ext | 60 | ext K=4 | ext K=1 | stage2_peak_locfree_rel_err | -0.008282 | — | 6 | 4 | 0 | -0.02094 | 0.0008526 |
| ext | 60 | ext K=4 | ext K=1 | stage2_peak_locfree_rel_err_abs | 0.008282 | 4 | 4 | 6 | 0 | -0.0009105 | 0.01987 |
| ext | 60 | ext K=4 | ext K=1 | stage2_rmse_pos_over_g2_0 | -0.005099 | 9 | 9 | 1 | 0 | -0.01091 | -0.003869 |
| ext | 60 | ext K=4 | ext K=1 | stage2_tail_mean | -0.005074 | 5 | 5 | 5 | 0 | -0.01176 | 0.0297 |
| ext | 60 | ext K=4 | ext K=1 | stage2_tail_max | -0.001586 | 5 | 5 | 5 | 0 | -0.102 | 0.1706 |
| ext | 60 | ext K=4 | ext K=1 | stage2_tail_mean_over_g2_0 | -8.699e-05 | 5 | 5 | 5 | 0 | -0.0002041 | 0.0005145 |
| ext | 60 | ext K=4 | ext K=1 | stage2_tail_max_over_g2_0 | -2.718e-05 | 5 | 5 | 5 | 0 | -0.001773 | 0.002856 |
| ext | 60 | ext K=4 | ext K=1 | stage2_sym_err_max | -0.8878 | 9 | 9 | 1 | 0 | -1.131 | -0.4444 |
| ext | 60 | ext K=4 | ext K=1 | eta_T_over_dw | -0.0001711 | 7 | 7 | 3 | 0 | -0.0005195 | 3.155e-05 |
| ext | 60 | ext K=4 | ext K=1 | DeltaT_over_dw_on_max | -0.0001711 | 7 | 7 | 3 | 0 | -0.0005113 | 2.807e-05 |
| ext | 60 | ext K=4 | ext K=1 | DeltaT_over_dw_off_max | 5.78e-06 | 5 | 5 | 5 | 0 | -2.95e-05 | 6.324e-05 |
| ext | 50 | ext K=8 | ext K=1 | stage2_peak_rel_err_signed | -0.002852 | — | 7 | 3 | 0 | -0.01573 | 0.02057 |
| ext | 50 | ext K=8 | ext K=1 | stage2_peak_rel_err_abs | 0.002852 | 3 | 3 | 7 | 0 | -0.02056 | 0.01583 |
| ext | 50 | ext K=8 | ext K=1 | stage2_peak_locfree_rel_err | -0.003215 | — | 7 | 3 | 0 | -0.01589 | 0.02071 |
| ext | 50 | ext K=8 | ext K=1 | stage2_peak_locfree_rel_err_abs | 0.003215 | 3 | 3 | 7 | 0 | -0.02092 | 0.01611 |
| ext | 50 | ext K=8 | ext K=1 | stage2_rmse_pos_over_g2_0 | -0.004485 | 8 | 8 | 2 | 0 | -0.01735 | -0.002112 |
| ext | 50 | ext K=8 | ext K=1 | stage2_tail_mean | 0.03522 | 2 | 2 | 8 | 0 | 0.00715 | 0.04643 |
| ext | 50 | ext K=8 | ext K=1 | stage2_tail_max | 0.1532 | 4 | 4 | 6 | 0 | -0.269 | 0.3244 |
| ext | 50 | ext K=8 | ext K=1 | stage2_tail_mean_over_g2_0 | 0.0005031 | 2 | 2 | 8 | 0 | 9.5e-05 | 0.0006569 |
| ext | 50 | ext K=8 | ext K=1 | stage2_tail_max_over_g2_0 | 0.002189 | 4 | 4 | 6 | 0 | -0.003954 | 0.004513 |
| ext | 50 | ext K=8 | ext K=1 | stage2_sym_err_max | -0.7758 | 10 | 10 | 0 | 0 | -1.926 | -0.6146 |
| ext | 50 | ext K=8 | ext K=1 | eta_T_over_dw | -0.0002704 | 6 | 6 | 4 | 0 | -0.002172 | 0.0003157 |
| ext | 50 | ext K=8 | ext K=1 | DeltaT_over_dw_on_max | -0.0002704 | 6 | 6 | 4 | 0 | -0.002159 | 0.000296 |
| ext | 50 | ext K=8 | ext K=1 | DeltaT_over_dw_off_max | 1.662e-05 | 4 | 4 | 6 | 0 | -0.0001469 | 0.0001033 |
| ext | 60 | ext K=8 | ext K=1 | stage2_peak_rel_err_signed | 0.000478 | — | 5 | 5 | 0 | -0.01853 | 0.007403 |
| ext | 60 | ext K=8 | ext K=1 | stage2_peak_rel_err_abs | -0.000478 | 5 | 5 | 5 | 0 | -0.007502 | 0.0167 |
| ext | 60 | ext K=8 | ext K=1 | stage2_peak_locfree_rel_err | -0.003021 | — | 6 | 4 | 0 | -0.01916 | 0.007025 |
| ext | 60 | ext K=8 | ext K=1 | stage2_peak_locfree_rel_err_abs | 0.003021 | 4 | 4 | 6 | 0 | -0.006952 | 0.01722 |
| ext | 60 | ext K=8 | ext K=1 | stage2_rmse_pos_over_g2_0 | -0.005749 | 9 | 9 | 1 | 0 | -0.01197 | -0.004031 |
| ext | 60 | ext K=8 | ext K=1 | stage2_tail_mean | 0.01456 | 4 | 4 | 6 | 0 | -0.0001094 | 0.04174 |
| ext | 60 | ext K=8 | ext K=1 | stage2_tail_max | 0.07841 | 4 | 4 | 6 | 0 | -0.07806 | 0.1731 |
| ext | 60 | ext K=8 | ext K=1 | stage2_tail_mean_over_g2_0 | 0.0002495 | 4 | 4 | 6 | 0 | 4.411e-06 | 0.0007168 |
| ext | 60 | ext K=8 | ext K=1 | stage2_tail_max_over_g2_0 | 0.001344 | 4 | 4 | 6 | 0 | -0.001304 | 0.002958 |
| ext | 60 | ext K=8 | ext K=1 | stage2_sym_err_max | -0.9645 | 8 | 8 | 2 | 0 | -1.268 | -0.1683 |
| ext | 60 | ext K=8 | ext K=1 | eta_T_over_dw | -8.237e-05 | 9 | 9 | 1 | 0 | -0.00067 | -6.843e-05 |
| ext | 60 | ext K=8 | ext K=1 | DeltaT_over_dw_on_max | -8.237e-05 | 9 | 9 | 1 | 0 | -0.0006631 | -7.063e-05 |
| ext | 60 | ext K=8 | ext K=1 | DeltaT_over_dw_off_max | 3.111e-05 | 3 | 3 | 7 | 0 | -2.062e-05 | 6.224e-05 |
| ext | 50 | ext K=16 | ext K=1 | stage2_peak_rel_err_signed | -0.001551 | — | 6 | 4 | 0 | -0.01196 | 0.01789 |
| ext | 50 | ext K=16 | ext K=1 | stage2_peak_rel_err_abs | 0.001551 | 4 | 4 | 6 | 0 | -0.0181 | 0.01207 |
| ext | 50 | ext K=16 | ext K=1 | stage2_peak_locfree_rel_err | -0.001994 | — | 6 | 4 | 0 | -0.01221 | 0.01766 |
| ext | 50 | ext K=16 | ext K=1 | stage2_peak_locfree_rel_err_abs | 0.001994 | 4 | 4 | 6 | 0 | -0.01784 | 0.01252 |
| ext | 50 | ext K=16 | ext K=1 | stage2_rmse_pos_over_g2_0 | -0.00787 | 8 | 8 | 2 | 0 | -0.0184 | -0.00357 |
| ext | 50 | ext K=16 | ext K=1 | stage2_tail_mean | 0.05823 | 2 | 2 | 8 | 0 | 0.03029 | 0.06848 |
| ext | 50 | ext K=16 | ext K=1 | stage2_tail_max | 0.2154 | 4 | 4 | 6 | 0 | -0.1789 | 0.3619 |
| ext | 50 | ext K=16 | ext K=1 | stage2_tail_mean_over_g2_0 | 0.0008319 | 2 | 2 | 8 | 0 | 0.0004276 | 0.0009738 |
| ext | 50 | ext K=16 | ext K=1 | stage2_tail_max_over_g2_0 | 0.003077 | 4 | 4 | 6 | 0 | -0.002665 | 0.005124 |
| ext | 50 | ext K=16 | ext K=1 | stage2_sym_err_max | -1.145 | 9 | 9 | 1 | 0 | -1.893 | -0.6272 |
| ext | 50 | ext K=16 | ext K=1 | eta_T_over_dw | -0.0007141 | 7 | 7 | 3 | 0 | -0.002246 | 9.297e-05 |
| ext | 50 | ext K=16 | ext K=1 | DeltaT_over_dw_on_max | -0.0007141 | 7 | 7 | 3 | 0 | -0.002238 | 7.063e-05 |
| ext | 50 | ext K=16 | ext K=1 | DeltaT_over_dw_off_max | 4.474e-05 | 4 | 4 | 6 | 0 | -0.0001124 | 0.0001123 |
| ext | 60 | ext K=16 | ext K=1 | stage2_peak_rel_err_signed | -0.004511 | — | 6 | 4 | 0 | -0.02216 | 0.004843 |
| ext | 60 | ext K=16 | ext K=1 | stage2_peak_rel_err_abs | 0.004511 | 4 | 4 | 6 | 0 | -0.00521 | 0.02049 |
| ext | 60 | ext K=16 | ext K=1 | stage2_peak_locfree_rel_err | -0.007241 | — | 6 | 4 | 0 | -0.02304 | 0.004638 |
| ext | 60 | ext K=16 | ext K=1 | stage2_peak_locfree_rel_err_abs | 0.007241 | 4 | 4 | 6 | 0 | -0.004634 | 0.02095 |
| ext | 60 | ext K=16 | ext K=1 | stage2_rmse_pos_over_g2_0 | -0.007772 | 10 | 10 | 0 | 0 | -0.01301 | -0.005947 |
| ext | 60 | ext K=16 | ext K=1 | stage2_tail_mean | 0.04384 | 3 | 3 | 7 | 0 | 0.01478 | 0.07106 |
| ext | 60 | ext K=16 | ext K=1 | stage2_tail_max | 0.115 | 4 | 4 | 6 | 0 | -0.06068 | 0.248 |
| ext | 60 | ext K=16 | ext K=1 | stage2_tail_mean_over_g2_0 | 0.0007516 | 3 | 3 | 7 | 0 | 0.0002559 | 0.001217 |
| ext | 60 | ext K=16 | ext K=1 | stage2_tail_max_over_g2_0 | 0.001971 | 4 | 4 | 6 | 0 | -0.001064 | 0.004269 |
| ext | 60 | ext K=16 | ext K=1 | stage2_sym_err_max | -1.282 | 9 | 9 | 1 | 0 | -1.473 | -0.4131 |
| ext | 60 | ext K=16 | ext K=1 | eta_T_over_dw | -0.0001704 | 7 | 7 | 3 | 0 | -0.0006599 | -3.558e-05 |
| ext | 60 | ext K=16 | ext K=1 | DeltaT_over_dw_on_max | -0.0001704 | 7 | 7 | 3 | 0 | -0.0006538 | -3.622e-05 |
| ext | 60 | ext K=16 | ext K=1 | DeltaT_over_dw_off_max | 4.83e-05 | 3 | 3 | 7 | 0 | -1.247e-05 | 7.929e-05 |
| p3 | 50 | stochastic K=4 | stochastic K=1 | stage1_rel_err_signed | 0.02342 | — | 4 | 6 | 0 | -0.01418 | 0.04163 |
| p3 | 50 | stochastic K=4 | stochastic K=1 | stage1_rel_err_abs | -0.01962 | 7 | 7 | 3 | 0 | -0.03222 | 0.01211 |
| p3 | 50 | stochastic K=4 | stochastic K=1 | learning_rel | 0.02342 | — | 4 | 6 | 0 | -0.01453 | 0.04169 |
| p3 | 50 | stochastic K=4 | stochastic K=1 | learning_rel_abs | -0.01211 | 6 | 6 | 4 | 0 | -0.03203 | 0.0133 |
| p3 | 50 | stochastic K=4 | stochastic K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| p3 | 50 | stochastic K=4 | stochastic K=1 | Gmax_full_over_dw | 0 | 1 | 1 | 1 | 8 | -0.0001017 | 0.0001269 |
| p3 | 50 | stochastic K=4 | stochastic K=1 | EXP_root_over_dw | -0.0005122 | 6 | 6 | 4 | 0 | -0.00107 | 0.0005606 |
| p3 | 50 | stochastic K=4 | stochastic K=1 | dReach_over_dw | -0.00052 | 6 | 6 | 4 | 0 | -0.001058 | 0.0005908 |
| p3 | 60 | stochastic K=4 | stochastic K=1 | stage1_rel_err_signed | 0.03871 | — | 2 | 8 | 0 | 0.01141 | 0.06013 |
| p3 | 60 | stochastic K=4 | stochastic K=1 | stage1_rel_err_abs | -0.01737 | 6 | 6 | 4 | 0 | -0.04356 | 0.01444 |
| p3 | 60 | stochastic K=4 | stochastic K=1 | learning_rel | 0.03871 | — | 2 | 8 | 0 | 0.01087 | 0.06024 |
| p3 | 60 | stochastic K=4 | stochastic K=1 | learning_rel_abs | -0.0173 | 6 | 6 | 4 | 0 | -0.04484 | 0.01434 |
| p3 | 60 | stochastic K=4 | stochastic K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| p3 | 60 | stochastic K=4 | stochastic K=1 | Gmax_full_over_dw | 0 | 0 | 0 | 0 | 10 | 0 | 0 |
| p3 | 60 | stochastic K=4 | stochastic K=1 | EXP_root_over_dw | -0.000172 | 6 | 6 | 4 | 0 | -0.0008616 | 0.0001188 |
| p3 | 60 | stochastic K=4 | stochastic K=1 | dReach_over_dw | -0.0001733 | 6 | 6 | 4 | 0 | -0.0008596 | 0.0001153 |
| p3 | 50 | stochastic K=8 | stochastic K=1 | stage1_rel_err_signed | 0.005617 | — | 4 | 6 | 0 | -0.03914 | 0.04246 |
| p3 | 50 | stochastic K=8 | stochastic K=1 | stage1_rel_err_abs | -7.672e-05 | 5 | 5 | 5 | 0 | -0.04211 | 0.01252 |
| p3 | 50 | stochastic K=8 | stochastic K=1 | learning_rel | 0.005617 | — | 4 | 6 | 0 | -0.03929 | 0.04319 |
| p3 | 50 | stochastic K=8 | stochastic K=1 | learning_rel_abs | -0.0005459 | 5 | 5 | 5 | 0 | -0.04226 | 0.01385 |
| p3 | 50 | stochastic K=8 | stochastic K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| p3 | 50 | stochastic K=8 | stochastic K=1 | Gmax_full_over_dw | 0 | 1 | 1 | 0 | 9 | -0.0001017 | 0 |
| p3 | 50 | stochastic K=8 | stochastic K=1 | EXP_root_over_dw | -5.66e-05 | 5 | 5 | 5 | 0 | -0.001316 | 0.0002218 |
| p3 | 50 | stochastic K=8 | stochastic K=1 | dReach_over_dw | -7.159e-05 | 5 | 5 | 5 | 0 | -0.001299 | 0.0002297 |
| p3 | 60 | stochastic K=8 | stochastic K=1 | stage1_rel_err_signed | 0.06283 | — | 2 | 8 | 0 | 0.02336 | 0.08156 |
| p3 | 60 | stochastic K=8 | stochastic K=1 | stage1_rel_err_abs | -0.009612 | 5 | 5 | 5 | 0 | -0.04202 | 0.02581 |
| p3 | 60 | stochastic K=8 | stochastic K=1 | learning_rel | 0.06283 | — | 2 | 8 | 0 | 0.02306 | 0.08179 |
| p3 | 60 | stochastic K=8 | stochastic K=1 | learning_rel_abs | -0.004727 | 5 | 5 | 5 | 0 | -0.04424 | 0.02653 |
| p3 | 60 | stochastic K=8 | stochastic K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| p3 | 60 | stochastic K=8 | stochastic K=1 | Gmax_full_over_dw | 0 | 0 | 0 | 0 | 10 | 0 | 0 |
| p3 | 60 | stochastic K=8 | stochastic K=1 | EXP_root_over_dw | -0.0001188 | 5 | 5 | 5 | 0 | -0.0008881 | 0.0002891 |
| p3 | 60 | stochastic K=8 | stochastic K=1 | dReach_over_dw | -0.0001196 | 5 | 5 | 5 | 0 | -0.0008755 | 0.000287 |
| p3 | 50 | stochastic K=12 | stochastic K=1 | stage1_rel_err_signed | -0.01305 | — | 5 | 5 | 0 | -0.05518 | 0.02727 |
| p3 | 50 | stochastic K=12 | stochastic K=1 | stage1_rel_err_abs | -0.01246 | 6 | 6 | 4 | 0 | -0.03948 | 0.01627 |
| p3 | 50 | stochastic K=12 | stochastic K=1 | learning_rel | -0.01305 | — | 5 | 5 | 0 | -0.05441 | 0.02745 |
| p3 | 50 | stochastic K=12 | stochastic K=1 | learning_rel_abs | -0.0193 | 7 | 7 | 3 | 0 | -0.04151 | 0.01795 |
| p3 | 50 | stochastic K=12 | stochastic K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| p3 | 50 | stochastic K=12 | stochastic K=1 | Gmax_full_over_dw | 0 | 1 | 1 | 0 | 9 | -0.0001017 | 0 |
| p3 | 50 | stochastic K=12 | stochastic K=1 | EXP_root_over_dw | -0.0003638 | 7 | 7 | 3 | 0 | -0.001283 | 0.00034 |
| p3 | 50 | stochastic K=12 | stochastic K=1 | dReach_over_dw | -0.0003796 | 7 | 7 | 3 | 0 | -0.001278 | 0.0003785 |
| p3 | 60 | stochastic K=12 | stochastic K=1 | stage1_rel_err_signed | 0.05117 | — | 1 | 9 | 0 | 0.02126 | 0.08723 |
| p3 | 60 | stochastic K=12 | stochastic K=1 | stage1_rel_err_abs | 0.009994 | 4 | 4 | 6 | 0 | -0.04747 | 0.02785 |
| p3 | 60 | stochastic K=12 | stochastic K=1 | learning_rel | 0.05117 | — | 1 | 9 | 0 | 0.0219 | 0.08768 |
| p3 | 60 | stochastic K=12 | stochastic K=1 | learning_rel_abs | 0.01371 | 4 | 4 | 6 | 0 | -0.04719 | 0.02951 |
| p3 | 60 | stochastic K=12 | stochastic K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| p3 | 60 | stochastic K=12 | stochastic K=1 | Gmax_full_over_dw | 0 | 0 | 0 | 0 | 10 | 0 | 0 |
| p3 | 60 | stochastic K=12 | stochastic K=1 | EXP_root_over_dw | 0.0001121 | 4 | 4 | 6 | 0 | -0.0008656 | 0.000425 |
| p3 | 60 | stochastic K=12 | stochastic K=1 | dReach_over_dw | 0.0001115 | 4 | 4 | 6 | 0 | -0.0008673 | 0.0004139 |
| p3 | 50 | mean K=4 | mean K=1 | stage1_rel_err_signed | 0.01742 | — | 3 | 7 | 0 | -0.004867 | 0.02548 |
| p3 | 50 | mean K=4 | mean K=1 | stage1_rel_err_abs | -0.01143 | 6 | 6 | 4 | 0 | -0.025 | 0.007191 |
| p3 | 50 | mean K=4 | mean K=1 | learning_rel | 0.01742 | — | 3 | 7 | 0 | -0.004524 | 0.02573 |
| p3 | 50 | mean K=4 | mean K=1 | learning_rel_abs | -0.01143 | 6 | 6 | 4 | 0 | -0.02507 | 0.007038 |
| p3 | 50 | mean K=4 | mean K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| p3 | 50 | mean K=4 | mean K=1 | Gmax_full_over_dw | 0 | 1 | 1 | 1 | 8 | -0.0008032 | 0.0001236 |
| p3 | 50 | mean K=4 | mean K=1 | EXP_root_over_dw | -0.0002253 | 6 | 6 | 4 | 0 | -0.001097 | 0.0001205 |
| p3 | 50 | mean K=4 | mean K=1 | dReach_over_dw | -0.0002296 | 6 | 6 | 4 | 0 | -0.001115 | 9.793e-05 |
| p3 | 60 | mean K=4 | mean K=1 | stage1_rel_err_signed | 0.03527 | — | 3 | 7 | 0 | -0.006606 | 0.05267 |
| p3 | 60 | mean K=4 | mean K=1 | stage1_rel_err_abs | -0.03628 | 8 | 8 | 2 | 0 | -0.05218 | -0.007271 |
| p3 | 60 | mean K=4 | mean K=1 | learning_rel | 0.03527 | — | 3 | 7 | 0 | -0.006691 | 0.05336 |
| p3 | 60 | mean K=4 | mean K=1 | learning_rel_abs | -0.03679 | 8 | 8 | 2 | 0 | -0.05216 | -0.006805 |
| p3 | 60 | mean K=4 | mean K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| p3 | 60 | mean K=4 | mean K=1 | Gmax_full_over_dw | -9.555e-05 | 5 | 5 | 1 | 4 | -0.0008859 | -0.000128 |
| p3 | 60 | mean K=4 | mean K=1 | EXP_root_over_dw | -0.0005842 | 8 | 8 | 2 | 0 | -0.001214 | -0.000126 |
| p3 | 60 | mean K=4 | mean K=1 | dReach_over_dw | -0.0005973 | 8 | 8 | 2 | 0 | -0.001208 | -0.0001301 |
| p3 | 50 | mean K=8 | mean K=1 | stage1_rel_err_signed | 0.01277 | — | 4 | 6 | 0 | -0.02102 | 0.02816 |
| p3 | 50 | mean K=8 | mean K=1 | stage1_rel_err_abs | -0.00976 | 5 | 5 | 5 | 0 | -0.03193 | 0.007732 |
| p3 | 50 | mean K=8 | mean K=1 | learning_rel | 0.01277 | — | 4 | 6 | 0 | -0.02088 | 0.02861 |
| p3 | 50 | mean K=8 | mean K=1 | learning_rel_abs | -0.00976 | 5 | 5 | 5 | 0 | -0.03634 | 0.006487 |
| p3 | 50 | mean K=8 | mean K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| p3 | 50 | mean K=8 | mean K=1 | Gmax_full_over_dw | 0 | 1 | 1 | 0 | 9 | -0.001006 | 0 |
| p3 | 50 | mean K=8 | mean K=1 | EXP_root_over_dw | -0.0001136 | 5 | 5 | 5 | 0 | -0.00133 | 6.12e-05 |
| p3 | 50 | mean K=8 | mean K=1 | dReach_over_dw | -0.000127 | 5 | 5 | 5 | 0 | -0.001366 | 4.357e-05 |
| p3 | 60 | mean K=8 | mean K=1 | stage1_rel_err_signed | 0.04539 | — | 4 | 6 | 0 | 0.002325 | 0.07496 |
| p3 | 60 | mean K=8 | mean K=1 | stage1_rel_err_abs | -0.02862 | 9 | 9 | 1 | 0 | -0.05997 | -0.004026 |
| p3 | 60 | mean K=8 | mean K=1 | learning_rel | 0.04539 | — | 4 | 6 | 0 | 0.001494 | 0.07486 |
| p3 | 60 | mean K=8 | mean K=1 | learning_rel_abs | -0.02419 | 9 | 9 | 1 | 0 | -0.06021 | -0.003613 |
| p3 | 60 | mean K=8 | mean K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| p3 | 60 | mean K=8 | mean K=1 | Gmax_full_over_dw | -9.555e-05 | 5 | 5 | 0 | 5 | -0.001117 | -0.0001344 |
| p3 | 60 | mean K=8 | mean K=1 | EXP_root_over_dw | -0.0004625 | 9 | 9 | 1 | 0 | -0.00149 | -0.0001973 |
| p3 | 60 | mean K=8 | mean K=1 | dReach_over_dw | -0.0004727 | 9 | 9 | 1 | 0 | -0.001468 | -0.0002025 |
| p3 | 50 | mean K=12 | mean K=1 | stage1_rel_err_signed | 0.003711 | — | 5 | 5 | 0 | -0.03058 | 0.03349 |
| p3 | 50 | mean K=12 | mean K=1 | stage1_rel_err_abs | -0.01911 | 6 | 6 | 4 | 0 | -0.03932 | 0.01319 |
| p3 | 50 | mean K=12 | mean K=1 | learning_rel | 0.003711 | — | 5 | 5 | 0 | -0.03042 | 0.03331 |
| p3 | 50 | mean K=12 | mean K=1 | learning_rel_abs | -0.01911 | 6 | 6 | 4 | 0 | -0.04247 | 0.01118 |
| p3 | 50 | mean K=12 | mean K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| p3 | 50 | mean K=12 | mean K=1 | Gmax_full_over_dw | 0 | 1 | 1 | 0 | 9 | -0.001074 | 0 |
| p3 | 50 | mean K=12 | mean K=1 | EXP_root_over_dw | -0.0004704 | 6 | 6 | 4 | 0 | -0.001625 | 3.972e-05 |
| p3 | 50 | mean K=12 | mean K=1 | dReach_over_dw | -0.0004652 | 6 | 6 | 4 | 0 | -0.001669 | 2.204e-05 |
| p3 | 60 | mean K=12 | mean K=1 | stage1_rel_err_signed | 0.03808 | — | 4 | 6 | 0 | -0.007908 | 0.07813 |
| p3 | 60 | mean K=12 | mean K=1 | stage1_rel_err_abs | -0.03715 | 8 | 8 | 2 | 0 | -0.06685 | -0.004595 |
| p3 | 60 | mean K=12 | mean K=1 | learning_rel | 0.03808 | — | 4 | 6 | 0 | -0.008291 | 0.07841 |
| p3 | 60 | mean K=12 | mean K=1 | learning_rel_abs | -0.03715 | 8 | 8 | 2 | 0 | -0.06597 | -0.001899 |
| p3 | 60 | mean K=12 | mean K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| p3 | 60 | mean K=12 | mean K=1 | Gmax_full_over_dw | -9.555e-05 | 5 | 5 | 0 | 5 | -0.001117 | -0.0001344 |
| p3 | 60 | mean K=12 | mean K=1 | EXP_root_over_dw | -0.0007192 | 8 | 8 | 2 | 0 | -0.001652 | -0.000187 |
| p3 | 60 | mean K=12 | mean K=1 | dReach_over_dw | -0.0007191 | 8 | 8 | 2 | 0 | -0.001616 | -0.0001861 |
| 2a | 50 | constant K=4 | constant K=1 | stage2_peak_rel_err_signed | -0.002328 | — | 6 | 4 | 0 | -0.01162 | 0.0191 |
| 2a | 50 | constant K=4 | constant K=1 | stage2_peak_rel_err_abs | 0.002328 | 4 | 4 | 6 | 0 | -0.01962 | 0.01169 |
| 2a | 50 | constant K=4 | constant K=1 | stage2_peak_locfree_rel_err | -0.003148 | — | 6 | 4 | 0 | -0.01141 | 0.01964 |
| 2a | 50 | constant K=4 | constant K=1 | stage2_peak_locfree_rel_err_abs | 0.003148 | 4 | 4 | 6 | 0 | -0.01999 | 0.0114 |
| 2a | 50 | constant K=4 | constant K=1 | stage2_rmse_pos_over_g2_0 | -0.002994 | 8 | 8 | 2 | 0 | -0.01513 | -0.001532 |
| 2a | 50 | constant K=4 | constant K=1 | stage2_tail_mean | 0.01792 | 1 | 1 | 9 | 0 | 0.004275 | 0.03488 |
| 2a | 50 | constant K=4 | constant K=1 | stage2_tail_max | 0.114 | 4 | 4 | 6 | 0 | -0.2691 | 0.1441 |
| 2a | 50 | constant K=4 | constant K=1 | stage2_tail_mean_over_g2_0 | 0.0002559 | 1 | 1 | 9 | 0 | 6.401e-05 | 0.0004936 |
| 2a | 50 | constant K=4 | constant K=1 | stage2_tail_max_over_g2_0 | 0.001628 | 4 | 4 | 6 | 0 | -0.003921 | 0.002019 |
| 2a | 50 | constant K=4 | constant K=1 | stage2_sym_err_max | -0.4989 | 7 | 7 | 3 | 0 | -1.886 | -0.1045 |
| 2a | 50 | constant K=4 | constant K=1 | eta_T_over_dw | -0.0003245 | 8 | 8 | 2 | 0 | -0.002218 | 0.0002004 |
| 2a | 50 | constant K=4 | constant K=1 | DeltaT_over_dw_on_max | -0.0003245 | 8 | 8 | 2 | 0 | -0.002207 | 0.0001858 |
| 2a | 50 | constant K=4 | constant K=1 | DeltaT_over_dw_off_max | 2.497e-05 | 3 | 3 | 7 | 0 | -0.0001233 | 4.7e-05 |
| 2a | 60 | constant K=4 | constant K=1 | stage2_peak_rel_err_signed | -0.007549 | — | 6 | 4 | 0 | -0.02033 | 0.002442 |
| 2a | 60 | constant K=4 | constant K=1 | stage2_peak_rel_err_abs | 0.007549 | 4 | 4 | 6 | 0 | -0.002239 | 0.01907 |
| 2a | 60 | constant K=4 | constant K=1 | stage2_peak_locfree_rel_err | -0.008282 | — | 6 | 4 | 0 | -0.02094 | 0.0008526 |
| 2a | 60 | constant K=4 | constant K=1 | stage2_peak_locfree_rel_err_abs | 0.008282 | 4 | 4 | 6 | 0 | -0.0009105 | 0.01987 |
| 2a | 60 | constant K=4 | constant K=1 | stage2_rmse_pos_over_g2_0 | -0.005099 | 9 | 9 | 1 | 0 | -0.01091 | -0.003869 |
| 2a | 60 | constant K=4 | constant K=1 | stage2_tail_mean | -0.005074 | 5 | 5 | 5 | 0 | -0.01176 | 0.0297 |
| 2a | 60 | constant K=4 | constant K=1 | stage2_tail_max | -0.001586 | 5 | 5 | 5 | 0 | -0.102 | 0.1706 |
| 2a | 60 | constant K=4 | constant K=1 | stage2_tail_mean_over_g2_0 | -8.699e-05 | 5 | 5 | 5 | 0 | -0.0002041 | 0.0005145 |
| 2a | 60 | constant K=4 | constant K=1 | stage2_tail_max_over_g2_0 | -2.718e-05 | 5 | 5 | 5 | 0 | -0.001773 | 0.002856 |
| 2a | 60 | constant K=4 | constant K=1 | stage2_sym_err_max | -0.8878 | 9 | 9 | 1 | 0 | -1.131 | -0.4444 |
| 2a | 60 | constant K=4 | constant K=1 | eta_T_over_dw | -0.0001711 | 7 | 7 | 3 | 0 | -0.0005195 | 3.155e-05 |
| 2a | 60 | constant K=4 | constant K=1 | DeltaT_over_dw_on_max | -0.0001711 | 7 | 7 | 3 | 0 | -0.0005113 | 2.807e-05 |
| 2a | 60 | constant K=4 | constant K=1 | DeltaT_over_dw_off_max | 5.78e-06 | 5 | 5 | 5 | 0 | -2.95e-05 | 6.324e-05 |
| 2a | 50 | constant K=8 | constant K=1 | stage2_peak_rel_err_signed | -0.002852 | — | 7 | 3 | 0 | -0.01573 | 0.02057 |
| 2a | 50 | constant K=8 | constant K=1 | stage2_peak_rel_err_abs | 0.002852 | 3 | 3 | 7 | 0 | -0.02056 | 0.01583 |
| 2a | 50 | constant K=8 | constant K=1 | stage2_peak_locfree_rel_err | -0.003215 | — | 7 | 3 | 0 | -0.01589 | 0.02071 |
| 2a | 50 | constant K=8 | constant K=1 | stage2_peak_locfree_rel_err_abs | 0.003215 | 3 | 3 | 7 | 0 | -0.02092 | 0.01611 |
| 2a | 50 | constant K=8 | constant K=1 | stage2_rmse_pos_over_g2_0 | -0.004485 | 8 | 8 | 2 | 0 | -0.01735 | -0.002112 |
| 2a | 50 | constant K=8 | constant K=1 | stage2_tail_mean | 0.03522 | 2 | 2 | 8 | 0 | 0.00715 | 0.04643 |
| 2a | 50 | constant K=8 | constant K=1 | stage2_tail_max | 0.1532 | 4 | 4 | 6 | 0 | -0.269 | 0.3244 |
| 2a | 50 | constant K=8 | constant K=1 | stage2_tail_mean_over_g2_0 | 0.0005031 | 2 | 2 | 8 | 0 | 9.5e-05 | 0.0006569 |
| 2a | 50 | constant K=8 | constant K=1 | stage2_tail_max_over_g2_0 | 0.002189 | 4 | 4 | 6 | 0 | -0.003954 | 0.004513 |
| 2a | 50 | constant K=8 | constant K=1 | stage2_sym_err_max | -0.7758 | 10 | 10 | 0 | 0 | -1.926 | -0.6146 |
| 2a | 50 | constant K=8 | constant K=1 | eta_T_over_dw | -0.0002704 | 6 | 6 | 4 | 0 | -0.002172 | 0.0003157 |
| 2a | 50 | constant K=8 | constant K=1 | DeltaT_over_dw_on_max | -0.0002704 | 6 | 6 | 4 | 0 | -0.002159 | 0.000296 |
| 2a | 50 | constant K=8 | constant K=1 | DeltaT_over_dw_off_max | 1.662e-05 | 4 | 4 | 6 | 0 | -0.0001469 | 0.0001033 |
| 2a | 60 | constant K=8 | constant K=1 | stage2_peak_rel_err_signed | 0.000478 | — | 5 | 5 | 0 | -0.01853 | 0.007403 |
| 2a | 60 | constant K=8 | constant K=1 | stage2_peak_rel_err_abs | -0.000478 | 5 | 5 | 5 | 0 | -0.007502 | 0.0167 |
| 2a | 60 | constant K=8 | constant K=1 | stage2_peak_locfree_rel_err | -0.003021 | — | 6 | 4 | 0 | -0.01916 | 0.007025 |
| 2a | 60 | constant K=8 | constant K=1 | stage2_peak_locfree_rel_err_abs | 0.003021 | 4 | 4 | 6 | 0 | -0.006952 | 0.01722 |
| 2a | 60 | constant K=8 | constant K=1 | stage2_rmse_pos_over_g2_0 | -0.005749 | 9 | 9 | 1 | 0 | -0.01197 | -0.004031 |
| 2a | 60 | constant K=8 | constant K=1 | stage2_tail_mean | 0.01456 | 4 | 4 | 6 | 0 | -0.0001094 | 0.04174 |
| 2a | 60 | constant K=8 | constant K=1 | stage2_tail_max | 0.07841 | 4 | 4 | 6 | 0 | -0.07806 | 0.1731 |
| 2a | 60 | constant K=8 | constant K=1 | stage2_tail_mean_over_g2_0 | 0.0002495 | 4 | 4 | 6 | 0 | 4.411e-06 | 0.0007168 |
| 2a | 60 | constant K=8 | constant K=1 | stage2_tail_max_over_g2_0 | 0.001344 | 4 | 4 | 6 | 0 | -0.001304 | 0.002958 |
| 2a | 60 | constant K=8 | constant K=1 | stage2_sym_err_max | -0.9645 | 8 | 8 | 2 | 0 | -1.268 | -0.1683 |
| 2a | 60 | constant K=8 | constant K=1 | eta_T_over_dw | -8.237e-05 | 9 | 9 | 1 | 0 | -0.00067 | -6.843e-05 |
| 2a | 60 | constant K=8 | constant K=1 | DeltaT_over_dw_on_max | -8.237e-05 | 9 | 9 | 1 | 0 | -0.0006631 | -7.063e-05 |
| 2a | 60 | constant K=8 | constant K=1 | DeltaT_over_dw_off_max | 3.111e-05 | 3 | 3 | 7 | 0 | -2.062e-05 | 6.224e-05 |
| 2a | 50 | decay K=4 | decay K=1 | stage2_peak_rel_err_signed | -0.003899 | — | 7 | 3 | 0 | -0.009432 | 0.00263 |
| 2a | 50 | decay K=4 | decay K=1 | stage2_peak_rel_err_abs | 0.003899 | 3 | 3 | 7 | 0 | -0.002589 | 0.009436 |
| 2a | 50 | decay K=4 | decay K=1 | stage2_peak_locfree_rel_err | -0.003946 | — | 6 | 4 | 0 | -0.009471 | 0.003016 |
| 2a | 50 | decay K=4 | decay K=1 | stage2_peak_locfree_rel_err_abs | 0.003946 | 4 | 4 | 6 | 0 | -0.002987 | 0.00951 |
| 2a | 50 | decay K=4 | decay K=1 | stage2_rmse_pos_over_g2_0 | -0.003088 | 7 | 7 | 3 | 0 | -0.004937 | -0.0009109 |
| 2a | 50 | decay K=4 | decay K=1 | stage2_tail_mean | 0.003872 | 5 | 5 | 5 | 0 | -0.01073 | 0.01949 |
| 2a | 50 | decay K=4 | decay K=1 | stage2_tail_max | -0.02423 | 5 | 5 | 5 | 0 | -0.1595 | 0.2028 |
| 2a | 50 | decay K=4 | decay K=1 | stage2_tail_mean_over_g2_0 | 5.531e-05 | 5 | 5 | 5 | 0 | -0.0001556 | 0.0002727 |
| 2a | 50 | decay K=4 | decay K=1 | stage2_tail_max_over_g2_0 | -0.0003461 | 5 | 5 | 5 | 0 | -0.002286 | 0.002874 |
| 2a | 50 | decay K=4 | decay K=1 | stage2_sym_err_max | -0.4241 | 9 | 9 | 1 | 0 | -0.8625 | -0.1618 |
| 2a | 50 | decay K=4 | decay K=1 | eta_T_over_dw | -9.258e-05 | 6 | 6 | 4 | 0 | -0.0004348 | 0.0001118 |
| 2a | 50 | decay K=4 | decay K=1 | DeltaT_over_dw_on_max | -9.258e-05 | 6 | 6 | 4 | 0 | -0.0004331 | 0.0001169 |
| 2a | 50 | decay K=4 | decay K=1 | DeltaT_over_dw_off_max | -3.486e-05 | 6 | 6 | 4 | 0 | -4.768e-05 | 8.953e-05 |
| 2a | 60 | decay K=4 | decay K=1 | stage2_peak_rel_err_signed | -0.007091 | — | 8 | 2 | 0 | -0.01404 | 0.001158 |
| 2a | 60 | decay K=4 | decay K=1 | stage2_peak_rel_err_abs | 0.007091 | 2 | 2 | 8 | 0 | -0.001115 | 0.014 |
| 2a | 60 | decay K=4 | decay K=1 | stage2_peak_locfree_rel_err | -0.006708 | — | 7 | 3 | 0 | -0.01398 | 0.001365 |
| 2a | 60 | decay K=4 | decay K=1 | stage2_peak_locfree_rel_err_abs | 0.006708 | 3 | 3 | 7 | 0 | -0.001521 | 0.01394 |
| 2a | 60 | decay K=4 | decay K=1 | stage2_rmse_pos_over_g2_0 | -0.003414 | 6 | 6 | 4 | 0 | -0.004621 | -0.0009541 |
| 2a | 60 | decay K=4 | decay K=1 | stage2_tail_mean | -0.007318 | 6 | 6 | 4 | 0 | -0.01107 | 0.01166 |
| 2a | 60 | decay K=4 | decay K=1 | stage2_tail_max | -0.01694 | 8 | 8 | 2 | 0 | -0.1005 | 0.05854 |
| 2a | 60 | decay K=4 | decay K=1 | stage2_tail_mean_over_g2_0 | -0.0001254 | 6 | 6 | 4 | 0 | -0.0001923 | 0.0001951 |
| 2a | 60 | decay K=4 | decay K=1 | stage2_tail_max_over_g2_0 | -0.0002904 | 8 | 8 | 2 | 0 | -0.001717 | 0.00103 |
| 2a | 60 | decay K=4 | decay K=1 | stage2_sym_err_max | -0.1544 | 5 | 5 | 5 | 0 | -0.838 | -0.02315 |
| 2a | 60 | decay K=4 | decay K=1 | eta_T_over_dw | -3.422e-05 | 7 | 7 | 3 | 0 | -0.00035 | 3.504e-05 |
| 2a | 60 | decay K=4 | decay K=1 | DeltaT_over_dw_on_max | -3.422e-05 | 7 | 7 | 3 | 0 | -0.0003472 | 3.603e-05 |
| 2a | 60 | decay K=4 | decay K=1 | DeltaT_over_dw_off_max | -1.354e-05 | 7 | 7 | 3 | 0 | -1.89e-05 | 2.104e-05 |
| 2a | 50 | decay K=8 | decay K=1 | stage2_peak_rel_err_signed | 0.0003223 | — | 5 | 5 | 0 | -0.006417 | 0.0059 |
| 2a | 50 | decay K=8 | decay K=1 | stage2_peak_rel_err_abs | -0.0003223 | 5 | 5 | 5 | 0 | -0.00582 | 0.006366 |
| 2a | 50 | decay K=8 | decay K=1 | stage2_peak_locfree_rel_err | 0.0005691 | — | 5 | 5 | 0 | -0.006426 | 0.006462 |
| 2a | 50 | decay K=8 | decay K=1 | stage2_peak_locfree_rel_err_abs | -0.0005691 | 5 | 5 | 5 | 0 | -0.006461 | 0.006571 |
| 2a | 50 | decay K=8 | decay K=1 | stage2_rmse_pos_over_g2_0 | -0.003814 | 9 | 9 | 1 | 0 | -0.005817 | -0.001634 |
| 2a | 50 | decay K=8 | decay K=1 | stage2_tail_mean | 0.004785 | 4 | 4 | 6 | 0 | -0.009996 | 0.02488 |
| 2a | 50 | decay K=8 | decay K=1 | stage2_tail_max | -0.0757 | 6 | 6 | 4 | 0 | -0.2201 | 0.09421 |
| 2a | 50 | decay K=8 | decay K=1 | stage2_tail_mean_over_g2_0 | 6.836e-05 | 4 | 4 | 6 | 0 | -0.0001417 | 0.0003584 |
| 2a | 50 | decay K=8 | decay K=1 | stage2_tail_max_over_g2_0 | -0.001081 | 6 | 6 | 4 | 0 | -0.003141 | 0.001331 |
| 2a | 50 | decay K=8 | decay K=1 | stage2_sym_err_max | -0.4192 | 7 | 7 | 3 | 0 | -1.198 | -0.1489 |
| 2a | 50 | decay K=8 | decay K=1 | eta_T_over_dw | -1.578e-05 | 6 | 6 | 4 | 0 | -0.0003724 | 6.562e-05 |
| 2a | 50 | decay K=8 | decay K=1 | DeltaT_over_dw_on_max | -1.578e-05 | 6 | 6 | 4 | 0 | -0.0003724 | 7.195e-05 |
| 2a | 50 | decay K=8 | decay K=1 | DeltaT_over_dw_off_max | -4.384e-05 | 6 | 6 | 4 | 0 | -7.75e-05 | 3.678e-05 |
| 2a | 60 | decay K=8 | decay K=1 | stage2_peak_rel_err_signed | 0.000981 | — | 5 | 5 | 0 | -0.01426 | 0.0043 |
| 2a | 60 | decay K=8 | decay K=1 | stage2_peak_rel_err_abs | -0.000981 | 5 | 5 | 5 | 0 | -0.004308 | 0.01418 |
| 2a | 60 | decay K=8 | decay K=1 | stage2_peak_locfree_rel_err | 0.0009445 | — | 5 | 5 | 0 | -0.01382 | 0.004957 |
| 2a | 60 | decay K=8 | decay K=1 | stage2_peak_locfree_rel_err_abs | -0.0009445 | 5 | 5 | 5 | 0 | -0.004887 | 0.01404 |
| 2a | 60 | decay K=8 | decay K=1 | stage2_rmse_pos_over_g2_0 | -0.001783 | 7 | 7 | 3 | 0 | -0.005592 | -0.0009025 |
| 2a | 60 | decay K=8 | decay K=1 | stage2_tail_mean | -0.002486 | 6 | 6 | 4 | 0 | -0.0105 | 0.01321 |
| 2a | 60 | decay K=8 | decay K=1 | stage2_tail_max | -0.01357 | 5 | 5 | 5 | 0 | -0.103 | 0.02382 |
| 2a | 60 | decay K=8 | decay K=1 | stage2_tail_mean_over_g2_0 | -4.262e-05 | 6 | 6 | 4 | 0 | -0.0001835 | 0.0002243 |
| 2a | 60 | decay K=8 | decay K=1 | stage2_tail_max_over_g2_0 | -0.0002326 | 5 | 5 | 5 | 0 | -0.001758 | 0.0003778 |
| 2a | 60 | decay K=8 | decay K=1 | stage2_sym_err_max | -0.2063 | 9 | 9 | 1 | 0 | -0.713 | -0.01596 |
| 2a | 60 | decay K=8 | decay K=1 | eta_T_over_dw | -0.0001136 | 8 | 8 | 2 | 0 | -0.0004409 | -6.497e-05 |
| 2a | 60 | decay K=8 | decay K=1 | DeltaT_over_dw_on_max | -0.0001136 | 8 | 8 | 2 | 0 | -0.0004338 | -6.502e-05 |
| 2a | 60 | decay K=8 | decay K=1 | DeltaT_over_dw_off_max | -1.79e-05 | 7 | 7 | 3 | 0 | -2.475e-05 | 1.144e-05 |
| 2b | 50 | constant K=4 | constant K=1 | stage1_rel_err_signed | -0.00911 | — | 6 | 4 | 0 | -0.02116 | 0.02898 |
| 2b | 50 | constant K=4 | constant K=1 | stage1_rel_err_abs | -0.01922 | 7 | 7 | 3 | 0 | -0.02942 | 0.0189 |
| 2b | 50 | constant K=4 | constant K=1 | learning_rel | -0.00911 | — | 6 | 4 | 0 | -0.02142 | 0.02907 |
| 2b | 50 | constant K=4 | constant K=1 | learning_rel_abs | -0.01285 | 6 | 6 | 4 | 0 | -0.02713 | 0.02067 |
| 2b | 50 | constant K=4 | constant K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 50 | constant K=4 | constant K=1 | Gmax_full_over_dw | 0 | 3 | 3 | 2 | 5 | -0.001447 | 0.001233 |
| 2b | 50 | constant K=4 | constant K=1 | EXP_root_over_dw | -0.0002072 | 6 | 6 | 4 | 0 | -0.001565 | 0.001242 |
| 2b | 50 | constant K=4 | constant K=1 | dReach_over_dw | -0.0002066 | 6 | 6 | 4 | 0 | -0.001576 | 0.001241 |
| 2b | 60 | constant K=4 | constant K=1 | stage1_rel_err_signed | 0.01723 | — | 3 | 7 | 0 | -0.01339 | 0.03986 |
| 2b | 60 | constant K=4 | constant K=1 | stage1_rel_err_abs | 0.00794 | 3 | 3 | 7 | 0 | -0.01327 | 0.03597 |
| 2b | 60 | constant K=4 | constant K=1 | learning_rel | 0.01723 | — | 3 | 7 | 0 | -0.0132 | 0.03953 |
| 2b | 60 | constant K=4 | constant K=1 | learning_rel_abs | 0.00911 | 3 | 3 | 7 | 0 | -0.01312 | 0.03589 |
| 2b | 60 | constant K=4 | constant K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 60 | constant K=4 | constant K=1 | Gmax_full_over_dw | 3.382e-05 | 2 | 2 | 5 | 3 | -0.0002077 | 0.0009938 |
| 2b | 60 | constant K=4 | constant K=1 | EXP_root_over_dw | 7.376e-05 | 3 | 3 | 7 | 0 | -0.0002126 | 0.001051 |
| 2b | 60 | constant K=4 | constant K=1 | dReach_over_dw | 7.643e-05 | 3 | 3 | 7 | 0 | -0.0002336 | 0.001033 |
| 2b | 50 | constant K=8 | constant K=1 | stage1_rel_err_signed | -0.01222 | — | 6 | 4 | 0 | -0.03877 | 0.03824 |
| 2b | 50 | constant K=8 | constant K=1 | stage1_rel_err_abs | -0.01475 | 6 | 6 | 4 | 0 | -0.04389 | 0.02401 |
| 2b | 50 | constant K=8 | constant K=1 | learning_rel | -0.01222 | — | 6 | 4 | 0 | -0.03887 | 0.03881 |
| 2b | 50 | constant K=8 | constant K=1 | learning_rel_abs | -0.01475 | 6 | 6 | 4 | 0 | -0.04264 | 0.02509 |
| 2b | 50 | constant K=8 | constant K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 50 | constant K=8 | constant K=1 | Gmax_full_over_dw | 0 | 3 | 3 | 3 | 4 | -0.002435 | 0.002203 |
| 2b | 50 | constant K=8 | constant K=1 | EXP_root_over_dw | -0.0002435 | 5 | 5 | 5 | 0 | -0.002547 | 0.002026 |
| 2b | 50 | constant K=8 | constant K=1 | dReach_over_dw | -0.0002263 | 5 | 5 | 5 | 0 | -0.002581 | 0.002057 |
| 2b | 60 | constant K=8 | constant K=1 | stage1_rel_err_signed | 0.02372 | — | 4 | 6 | 0 | -0.02077 | 0.05393 |
| 2b | 60 | constant K=8 | constant K=1 | stage1_rel_err_abs | 0.02147 | 3 | 3 | 7 | 0 | -0.0114 | 0.05008 |
| 2b | 60 | constant K=8 | constant K=1 | learning_rel | 0.02372 | — | 4 | 6 | 0 | -0.02074 | 0.0552 |
| 2b | 60 | constant K=8 | constant K=1 | learning_rel_abs | 0.02243 | 3 | 3 | 7 | 0 | -0.0111 | 0.05119 |
| 2b | 60 | constant K=8 | constant K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 60 | constant K=8 | constant K=1 | Gmax_full_over_dw | 0 | 3 | 3 | 4 | 3 | -0.0004138 | 0.001651 |
| 2b | 60 | constant K=8 | constant K=1 | EXP_root_over_dw | 0.0002884 | 3 | 3 | 7 | 0 | -0.0003682 | 0.001756 |
| 2b | 60 | constant K=8 | constant K=1 | dReach_over_dw | 0.0002868 | 3 | 3 | 7 | 0 | -0.000368 | 0.001763 |
| 2b | 50 | constant K=12 | constant K=1 | stage1_rel_err_signed | 0.005882 | — | 5 | 5 | 0 | -0.03457 | 0.04867 |
| 2b | 50 | constant K=12 | constant K=1 | stage1_rel_err_abs | -0.005882 | 5 | 5 | 5 | 0 | -0.039 | 0.03247 |
| 2b | 50 | constant K=12 | constant K=1 | learning_rel | 0.005882 | — | 5 | 5 | 0 | -0.03459 | 0.04967 |
| 2b | 50 | constant K=12 | constant K=1 | learning_rel_abs | -0.005882 | 5 | 5 | 5 | 0 | -0.03931 | 0.03408 |
| 2b | 50 | constant K=12 | constant K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 50 | constant K=12 | constant K=1 | Gmax_full_over_dw | 0 | 3 | 3 | 3 | 4 | -0.002433 | 0.001939 |
| 2b | 50 | constant K=12 | constant K=1 | EXP_root_over_dw | -4.606e-05 | 5 | 5 | 5 | 0 | -0.002476 | 0.001946 |
| 2b | 50 | constant K=12 | constant K=1 | dReach_over_dw | -4.756e-05 | 5 | 5 | 5 | 0 | -0.002504 | 0.001955 |
| 2b | 60 | constant K=12 | constant K=1 | stage1_rel_err_signed | 0.0325 | — | 4 | 6 | 0 | -0.02509 | 0.06221 |
| 2b | 60 | constant K=12 | constant K=1 | stage1_rel_err_abs | 0.02274 | 4 | 4 | 6 | 0 | -0.02374 | 0.04833 |
| 2b | 60 | constant K=12 | constant K=1 | learning_rel | 0.0325 | — | 4 | 6 | 0 | -0.0246 | 0.06309 |
| 2b | 60 | constant K=12 | constant K=1 | learning_rel_abs | 0.02031 | 4 | 4 | 6 | 0 | -0.02323 | 0.04823 |
| 2b | 60 | constant K=12 | constant K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 60 | constant K=12 | constant K=1 | Gmax_full_over_dw | 0 | 3 | 3 | 3 | 4 | -0.00071 | 0.001065 |
| 2b | 60 | constant K=12 | constant K=1 | EXP_root_over_dw | 0.000221 | 5 | 5 | 5 | 0 | -0.0006633 | 0.001274 |
| 2b | 60 | constant K=12 | constant K=1 | dReach_over_dw | 0.0002223 | 5 | 5 | 5 | 0 | -0.0006625 | 0.001249 |
| 2b | 50 | decay K=4 | decay K=1 | stage1_rel_err_signed | 0.001307 | — | 5 | 5 | 0 | -0.01267 | 0.01483 |
| 2b | 50 | decay K=4 | decay K=1 | stage1_rel_err_abs | 0.003672 | 3 | 3 | 7 | 0 | -0.00164 | 0.01749 |
| 2b | 50 | decay K=4 | decay K=1 | learning_rel | 0.001307 | — | 5 | 5 | 0 | -0.01274 | 0.01483 |
| 2b | 50 | decay K=4 | decay K=1 | learning_rel_abs | 0.004334 | 2 | 2 | 8 | 0 | 0.0005226 | 0.01909 |
| 2b | 50 | decay K=4 | decay K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 50 | decay K=4 | decay K=1 | Gmax_full_over_dw | 0 | 1 | 1 | 1 | 8 | -1.599e-05 | 0.0002548 |
| 2b | 50 | decay K=4 | decay K=1 | EXP_root_over_dw | 9.694e-05 | 2 | 2 | 8 | 0 | -3.789e-06 | 0.0005348 |
| 2b | 50 | decay K=4 | decay K=1 | dReach_over_dw | 9.631e-05 | 2 | 2 | 8 | 0 | -8.026e-06 | 0.0005447 |
| 2b | 60 | decay K=4 | decay K=1 | stage1_rel_err_signed | 0.00324 | — | 4 | 6 | 0 | -0.002474 | 0.02285 |
| 2b | 60 | decay K=4 | decay K=1 | stage1_rel_err_abs | -0.001691 | 6 | 6 | 4 | 0 | -0.01427 | 0.01285 |
| 2b | 60 | decay K=4 | decay K=1 | learning_rel | 0.00324 | — | 4 | 6 | 0 | -0.002784 | 0.02294 |
| 2b | 60 | decay K=4 | decay K=1 | learning_rel_abs | -0.001691 | 6 | 6 | 4 | 0 | -0.01409 | 0.01385 |
| 2b | 60 | decay K=4 | decay K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 60 | decay K=4 | decay K=1 | Gmax_full_over_dw | 0 | 2 | 2 | 0 | 8 | -1.237e-05 | 0 |
| 2b | 60 | decay K=4 | decay K=1 | EXP_root_over_dw | -1.345e-05 | 6 | 6 | 4 | 0 | -0.000166 | 0.0001275 |
| 2b | 60 | decay K=4 | decay K=1 | dReach_over_dw | -1.328e-05 | 6 | 6 | 4 | 0 | -0.0001634 | 0.0001272 |
| 2b | 50 | decay K=8 | decay K=1 | stage1_rel_err_signed | 0.005829 | — | 4 | 6 | 0 | -0.0205 | 0.01163 |
| 2b | 50 | decay K=8 | decay K=1 | stage1_rel_err_abs | -0.001334 | 5 | 5 | 5 | 0 | -0.01115 | 0.01321 |
| 2b | 50 | decay K=8 | decay K=1 | learning_rel | 0.005829 | — | 4 | 6 | 0 | -0.02062 | 0.01139 |
| 2b | 50 | decay K=8 | decay K=1 | learning_rel_abs | 0.002854 | 5 | 5 | 5 | 0 | -0.01244 | 0.01494 |
| 2b | 50 | decay K=8 | decay K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 50 | decay K=8 | decay K=1 | Gmax_full_over_dw | 0 | 0 | 0 | 2 | 8 | 0 | 0.0002928 |
| 2b | 50 | decay K=8 | decay K=1 | EXP_root_over_dw | 0.0001368 | 5 | 5 | 5 | 0 | -0.0001553 | 0.0004649 |
| 2b | 50 | decay K=8 | decay K=1 | dReach_over_dw | 0.0001302 | 5 | 5 | 5 | 0 | -0.0001581 | 0.0004703 |
| 2b | 60 | decay K=8 | decay K=1 | stage1_rel_err_signed | 0.03462 | — | 2 | 8 | 0 | -0.001266 | 0.03524 |
| 2b | 60 | decay K=8 | decay K=1 | stage1_rel_err_abs | 0.03077 | 3 | 3 | 7 | 0 | -0.001071 | 0.03349 |
| 2b | 60 | decay K=8 | decay K=1 | learning_rel | 0.03462 | — | 2 | 8 | 0 | -0.001372 | 0.03521 |
| 2b | 60 | decay K=8 | decay K=1 | learning_rel_abs | 0.03386 | 3 | 3 | 7 | 0 | -0.0007904 | 0.0342 |
| 2b | 60 | decay K=8 | decay K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 60 | decay K=8 | decay K=1 | Gmax_full_over_dw | 0 | 1 | 1 | 2 | 7 | -2.08e-06 | 0.0002669 |
| 2b | 60 | decay K=8 | decay K=1 | EXP_root_over_dw | 0.0002618 | 3 | 3 | 7 | 0 | -1.32e-05 | 0.0004462 |
| 2b | 60 | decay K=8 | decay K=1 | dReach_over_dw | 0.0002593 | 3 | 3 | 7 | 0 | -1.547e-05 | 0.0004447 |
| 2b | 50 | decay K=12 | decay K=1 | stage1_rel_err_signed | 0.01712 | — | 4 | 6 | 0 | -0.01331 | 0.02488 |
| 2b | 50 | decay K=12 | decay K=1 | stage1_rel_err_abs | -0.003455 | 5 | 5 | 5 | 0 | -0.01405 | 0.016 |
| 2b | 50 | decay K=12 | decay K=1 | learning_rel | 0.01712 | — | 4 | 6 | 0 | -0.01384 | 0.02466 |
| 2b | 50 | decay K=12 | decay K=1 | learning_rel_abs | 0.00535 | 4 | 4 | 6 | 0 | -0.01488 | 0.01878 |
| 2b | 50 | decay K=12 | decay K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 50 | decay K=12 | decay K=1 | Gmax_full_over_dw | 0 | 0 | 0 | 3 | 7 | 0 | 0.0003468 |
| 2b | 50 | decay K=12 | decay K=1 | EXP_root_over_dw | 3.212e-05 | 5 | 5 | 5 | 0 | -0.0001982 | 0.0007266 |
| 2b | 50 | decay K=12 | decay K=1 | dReach_over_dw | 3.134e-05 | 5 | 5 | 5 | 0 | -0.0002026 | 0.0007428 |
| 2b | 60 | decay K=12 | decay K=1 | stage1_rel_err_signed | 0.03882 | — | 2 | 8 | 0 | 0.002289 | 0.0518 |
| 2b | 60 | decay K=12 | decay K=1 | stage1_rel_err_abs | 0.03107 | 3 | 3 | 7 | 0 | -0.01174 | 0.04349 |
| 2b | 60 | decay K=12 | decay K=1 | learning_rel | 0.03882 | — | 2 | 8 | 0 | 0.002535 | 0.05183 |
| 2b | 60 | decay K=12 | decay K=1 | learning_rel_abs | 0.03415 | 3 | 3 | 7 | 0 | -0.01086 | 0.04405 |
| 2b | 60 | decay K=12 | decay K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 60 | decay K=12 | decay K=1 | Gmax_full_over_dw | 0 | 1 | 1 | 4 | 5 | 6.93e-06 | 0.0003992 |
| 2b | 60 | decay K=12 | decay K=1 | EXP_root_over_dw | 0.000357 | 3 | 3 | 7 | 0 | -3.959e-05 | 0.0006481 |
| 2b | 60 | decay K=12 | decay K=1 | dReach_over_dw | 0.0003587 | 3 | 3 | 7 | 0 | -3.787e-05 | 0.0006426 |

### Paired decay - constant (candidates)

Source: `results/v2_pilots/pilot4/analysis/paired_summary.csv`

| family | q | a | b | metric | median | n_better | n_neg | n_pos | n_zero | boot_ci95_lo | boot_ci95_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2a | 50 | decay K=1 | constant K=1 | stage2_peak_rel_err_signed | 0.006467 | — | 4 | 6 | 0 | -0.0157 | 0.02293 |
| 2a | 50 | decay K=1 | constant K=1 | stage2_peak_rel_err_abs | -0.006467 | 6 | 6 | 4 | 0 | -0.02297 | 0.01624 |
| 2a | 50 | decay K=1 | constant K=1 | stage2_peak_locfree_rel_err | 0.00646 | — | 4 | 6 | 0 | -0.01592 | 0.02275 |
| 2a | 50 | decay K=1 | constant K=1 | stage2_peak_locfree_rel_err_abs | -0.00646 | 6 | 6 | 4 | 0 | -0.02327 | 0.01642 |
| 2a | 50 | decay K=1 | constant K=1 | stage2_rmse_pos_over_g2_0 | -0.005423 | 6 | 6 | 4 | 0 | -0.01418 | 0.0009097 |
| 2a | 50 | decay K=1 | constant K=1 | stage2_tail_mean | 0.04541 | 3 | 3 | 7 | 0 | 0.00161 | 0.05989 |
| 2a | 50 | decay K=1 | constant K=1 | stage2_tail_max | 0.2839 | 3 | 3 | 7 | 0 | -0.07688 | 0.4555 |
| 2a | 50 | decay K=1 | constant K=1 | stage2_tail_mean_over_g2_0 | 0.0006488 | 3 | 3 | 7 | 0 | 9.723e-06 | 0.0008571 |
| 2a | 50 | decay K=1 | constant K=1 | stage2_tail_max_over_g2_0 | 0.004056 | 3 | 3 | 7 | 0 | -0.001151 | 0.006433 |
| 2a | 50 | decay K=1 | constant K=1 | stage2_sym_err_max | -0.5444 | 6 | 6 | 4 | 0 | -1.031 | 0.3524 |
| 2a | 50 | decay K=1 | constant K=1 | eta_T_over_dw | -0.000118 | 7 | 7 | 3 | 0 | -0.002086 | 0.0003807 |
| 2a | 50 | decay K=1 | constant K=1 | DeltaT_over_dw_on_max | -0.000118 | 7 | 7 | 3 | 0 | -0.002069 | 0.0003606 |
| 2a | 50 | decay K=1 | constant K=1 | DeltaT_over_dw_off_max | 9.135e-05 | 3 | 3 | 7 | 0 | -8.225e-05 | 0.0001479 |
| 2a | 60 | decay K=1 | constant K=1 | stage2_peak_rel_err_signed | -0.001493 | — | 6 | 4 | 0 | -0.0154 | 0.01308 |
| 2a | 60 | decay K=1 | constant K=1 | stage2_peak_rel_err_abs | 0.001493 | 4 | 4 | 6 | 0 | -0.0135 | 0.01429 |
| 2a | 60 | decay K=1 | constant K=1 | stage2_peak_locfree_rel_err | -0.001534 | — | 5 | 5 | 0 | -0.01703 | 0.01257 |
| 2a | 60 | decay K=1 | constant K=1 | stage2_peak_locfree_rel_err_abs | 0.001534 | 5 | 5 | 5 | 0 | -0.01309 | 0.01543 |
| 2a | 60 | decay K=1 | constant K=1 | stage2_rmse_pos_over_g2_0 | -0.005117 | 8 | 8 | 2 | 0 | -0.01021 | -0.001109 |
| 2a | 60 | decay K=1 | constant K=1 | stage2_tail_mean | 0.03792 | 3 | 3 | 7 | 0 | -0.006257 | 0.07044 |
| 2a | 60 | decay K=1 | constant K=1 | stage2_tail_max | 0.133 | 3 | 3 | 7 | 0 | -0.05343 | 0.3384 |
| 2a | 60 | decay K=1 | constant K=1 | stage2_tail_mean_over_g2_0 | 0.00065 | 3 | 3 | 7 | 0 | -8.652e-05 | 0.001218 |
| 2a | 60 | decay K=1 | constant K=1 | stage2_tail_max_over_g2_0 | 0.002281 | 3 | 3 | 7 | 0 | -0.0009417 | 0.005796 |
| 2a | 60 | decay K=1 | constant K=1 | stage2_sym_err_max | -0.3929 | 7 | 7 | 3 | 0 | -0.8841 | 0.1322 |
| 2a | 60 | decay K=1 | constant K=1 | eta_T_over_dw | 1.221e-05 | 4 | 4 | 6 | 0 | -0.000534 | 0.0002955 |
| 2a | 60 | decay K=1 | constant K=1 | DeltaT_over_dw_on_max | 1.221e-05 | 4 | 4 | 6 | 0 | -0.0005199 | 0.0002955 |
| 2a | 60 | decay K=1 | constant K=1 | DeltaT_over_dw_off_max | 5.949e-05 | 2 | 2 | 8 | 0 | -2.943e-05 | 9.039e-05 |
| 2a | 50 | decay K=4 | constant K=4 | stage2_peak_rel_err_signed | -0.00456 | — | 5 | 5 | 0 | -0.01125 | 0.008218 |
| 2a | 50 | decay K=4 | constant K=4 | stage2_peak_rel_err_abs | 0.00456 | 5 | 5 | 5 | 0 | -0.008283 | 0.01132 |
| 2a | 50 | decay K=4 | constant K=4 | stage2_peak_locfree_rel_err | -0.004147 | — | 5 | 5 | 0 | -0.01104 | 0.007953 |
| 2a | 50 | decay K=4 | constant K=4 | stage2_peak_locfree_rel_err_abs | 0.004147 | 5 | 5 | 5 | 0 | -0.007667 | 0.01102 |
| 2a | 50 | decay K=4 | constant K=4 | stage2_rmse_pos_over_g2_0 | -0.002315 | 7 | 7 | 3 | 0 | -0.004518 | 0.001301 |
| 2a | 50 | decay K=4 | constant K=4 | stage2_tail_mean | 0.02606 | 4 | 4 | 6 | 0 | -0.01532 | 0.04382 |
| 2a | 50 | decay K=4 | constant K=4 | stage2_tail_max | 0.2179 | 1 | 1 | 9 | 0 | 0.1376 | 0.3974 |
| 2a | 50 | decay K=4 | constant K=4 | stage2_tail_mean_over_g2_0 | 0.0003723 | 4 | 4 | 6 | 0 | -0.000209 | 0.0006353 |
| 2a | 50 | decay K=4 | constant K=4 | stage2_tail_max_over_g2_0 | 0.003113 | 1 | 1 | 9 | 0 | 0.001965 | 0.005673 |
| 2a | 50 | decay K=4 | constant K=4 | stage2_sym_err_max | -0.2604 | 7 | 7 | 3 | 0 | -0.554 | 1.026 |
| 2a | 50 | decay K=4 | constant K=4 | eta_T_over_dw | -4.212e-05 | 6 | 6 | 4 | 0 | -0.0003759 | 0.0003777 |
| 2a | 50 | decay K=4 | constant K=4 | DeltaT_over_dw_on_max | -4.212e-05 | 6 | 6 | 4 | 0 | -0.0003743 | 0.0003758 |
| 2a | 50 | decay K=4 | constant K=4 | DeltaT_over_dw_off_max | 6.588e-05 | 1 | 1 | 9 | 0 | 2.782e-05 | 0.0001445 |
| 2a | 60 | decay K=4 | constant K=4 | stage2_peak_rel_err_signed | 0.002187 | — | 5 | 5 | 0 | -0.008909 | 0.01108 |
| 2a | 60 | decay K=4 | constant K=4 | stage2_peak_rel_err_abs | -0.002187 | 5 | 5 | 5 | 0 | -0.01103 | 0.009165 |
| 2a | 60 | decay K=4 | constant K=4 | stage2_peak_locfree_rel_err | 0.002443 | — | 5 | 5 | 0 | -0.008544 | 0.01117 |
| 2a | 60 | decay K=4 | constant K=4 | stage2_peak_locfree_rel_err_abs | -0.002443 | 5 | 5 | 5 | 0 | -0.01137 | 0.008745 |
| 2a | 60 | decay K=4 | constant K=4 | stage2_rmse_pos_over_g2_0 | -0.0007302 | 7 | 7 | 3 | 0 | -0.001775 | -0.0002079 |
| 2a | 60 | decay K=4 | constant K=4 | stage2_tail_mean | 0.03037 | 4 | 4 | 6 | 0 | -0.0001166 | 0.05101 |
| 2a | 60 | decay K=4 | constant K=4 | stage2_tail_max | 0.1118 | 3 | 3 | 7 | 0 | -0.05454 | 0.2235 |
| 2a | 60 | decay K=4 | constant K=4 | stage2_tail_mean_over_g2_0 | 0.0005207 | 4 | 4 | 6 | 0 | 4.256e-07 | 0.0008804 |
| 2a | 60 | decay K=4 | constant K=4 | stage2_tail_max_over_g2_0 | 0.001917 | 3 | 3 | 7 | 0 | -0.0009129 | 0.003963 |
| 2a | 60 | decay K=4 | constant K=4 | stage2_sym_err_max | 0.05743 | 4 | 4 | 6 | 0 | -0.4937 | 0.4925 |
| 2a | 60 | decay K=4 | constant K=4 | eta_T_over_dw | -0.0001338 | 6 | 6 | 4 | 0 | -0.0001989 | 0.0001496 |
| 2a | 60 | decay K=4 | constant K=4 | DeltaT_over_dw_on_max | -0.0001338 | 6 | 6 | 4 | 0 | -0.0001961 | 0.0001542 |
| 2a | 60 | decay K=4 | constant K=4 | DeltaT_over_dw_off_max | 2.873e-05 | 2 | 2 | 8 | 0 | -2.584e-05 | 6.043e-05 |
| 2a | 50 | decay K=8 | constant K=8 | stage2_peak_rel_err_signed | 0.001494 | — | 5 | 5 | 0 | -0.006011 | 0.01243 |
| 2a | 50 | decay K=8 | constant K=8 | stage2_peak_rel_err_abs | -0.001494 | 5 | 5 | 5 | 0 | -0.01208 | 0.006091 |
| 2a | 50 | decay K=8 | constant K=8 | stage2_peak_locfree_rel_err | 0.001458 | — | 5 | 5 | 0 | -0.005799 | 0.01259 |
| 2a | 50 | decay K=8 | constant K=8 | stage2_peak_locfree_rel_err_abs | -0.001458 | 5 | 5 | 5 | 0 | -0.01237 | 0.005816 |
| 2a | 50 | decay K=8 | constant K=8 | stage2_rmse_pos_over_g2_0 | -0.00121 | 8 | 8 | 2 | 0 | -0.003252 | 0.0006788 |
| 2a | 50 | decay K=8 | constant K=8 | stage2_tail_mean | 0.02428 | 4 | 4 | 6 | 0 | -0.0122 | 0.03262 |
| 2a | 50 | decay K=8 | constant K=8 | stage2_tail_max | 0.2018 | 3 | 3 | 7 | 0 | -0.06326 | 0.2543 |
| 2a | 50 | decay K=8 | constant K=8 | stage2_tail_mean_over_g2_0 | 0.0003468 | 4 | 4 | 6 | 0 | -0.0001596 | 0.0004786 |
| 2a | 50 | decay K=8 | constant K=8 | stage2_tail_max_over_g2_0 | 0.002883 | 3 | 3 | 7 | 0 | -0.0008953 | 0.003597 |
| 2a | 50 | decay K=8 | constant K=8 | stage2_sym_err_max | 0.07683 | 3 | 3 | 7 | 0 | -0.1205 | 0.6469 |
| 2a | 50 | decay K=8 | constant K=8 | eta_T_over_dw | -0.0001913 | 7 | 7 | 3 | 0 | -0.0003438 | 0.0002331 |
| 2a | 50 | decay K=8 | constant K=8 | DeltaT_over_dw_on_max | -0.0001913 | 7 | 7 | 3 | 0 | -0.0003466 | 0.0002432 |
| 2a | 50 | decay K=8 | constant K=8 | DeltaT_over_dw_off_max | 5.815e-05 | 2 | 2 | 8 | 0 | -2.284e-05 | 8.294e-05 |
| 2a | 60 | decay K=8 | constant K=8 | stage2_peak_rel_err_signed | -0.001175 | — | 6 | 4 | 0 | -0.004597 | 0.00252 |
| 2a | 60 | decay K=8 | constant K=8 | stage2_peak_rel_err_abs | 0.001175 | 4 | 4 | 6 | 0 | -0.002543 | 0.00455 |
| 2a | 60 | decay K=8 | constant K=8 | stage2_peak_locfree_rel_err | -0.0006851 | — | 6 | 4 | 0 | -0.004572 | 0.002863 |
| 2a | 60 | decay K=8 | constant K=8 | stage2_peak_locfree_rel_err_abs | 0.0006851 | 4 | 4 | 6 | 0 | -0.002951 | 0.004569 |
| 2a | 60 | decay K=8 | constant K=8 | stage2_rmse_pos_over_g2_0 | -0.000235 | 6 | 6 | 4 | 0 | -0.001959 | 0.0006873 |
| 2a | 60 | decay K=8 | constant K=8 | stage2_tail_mean | 0.01052 | 3 | 3 | 7 | 0 | -0.008044 | 0.03728 |
| 2a | 60 | decay K=8 | constant K=8 | stage2_tail_max | 0.07873 | 2 | 2 | 8 | 0 | -0.04027 | 0.1373 |
| 2a | 60 | decay K=8 | constant K=8 | stage2_tail_mean_over_g2_0 | 0.0001803 | 3 | 3 | 7 | 0 | -0.0001382 | 0.0006439 |
| 2a | 60 | decay K=8 | constant K=8 | stage2_tail_max_over_g2_0 | 0.00135 | 2 | 2 | 8 | 0 | -0.0006594 | 0.002352 |
| 2a | 60 | decay K=8 | constant K=8 | stage2_sym_err_max | 0.0294 | 5 | 5 | 5 | 0 | -0.3517 | 0.4248 |
| 2a | 60 | decay K=8 | constant K=8 | eta_T_over_dw | 3.817e-05 | 4 | 4 | 6 | 0 | -0.0001365 | 7.415e-05 |
| 2a | 60 | decay K=8 | constant K=8 | DeltaT_over_dw_on_max | 3.817e-05 | 4 | 4 | 6 | 0 | -0.0001383 | 7.409e-05 |
| 2a | 60 | decay K=8 | constant K=8 | DeltaT_over_dw_off_max | 2.894e-05 | 3 | 3 | 7 | 0 | -2.786e-05 | 3.53e-05 |
| 2b | 50 | decay K=1 | constant K=1 | stage1_rel_err_signed | 0.009315 | — | 4 | 6 | 0 | -0.0949 | 0.009888 |
| 2b | 50 | decay K=1 | constant K=1 | stage1_rel_err_abs | -0.01504 | 6 | 6 | 4 | 0 | -0.04946 | 0.002933 |
| 2b | 50 | decay K=1 | constant K=1 | learning_rel | 0.009315 | — | 4 | 6 | 0 | -0.09543 | 0.01038 |
| 2b | 50 | decay K=1 | constant K=1 | learning_rel_abs | -0.01779 | 7 | 7 | 3 | 0 | -0.05704 | -0.0009278 |
| 2b | 50 | decay K=1 | constant K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 50 | decay K=1 | constant K=1 | Gmax_full_over_dw | 0 | 3 | 3 | 1 | 6 | -0.003116 | 5.831e-05 |
| 2b | 50 | decay K=1 | constant K=1 | EXP_root_over_dw | -0.0002665 | 7 | 7 | 3 | 0 | -0.003401 | -7.588e-06 |
| 2b | 50 | decay K=1 | constant K=1 | dReach_over_dw | -0.0002669 | 7 | 7 | 3 | 0 | -0.003384 | -4.485e-06 |
| 2b | 60 | decay K=1 | constant K=1 | stage1_rel_err_signed | -0.0523 | — | 7 | 3 | 0 | -0.09272 | 0.003189 |
| 2b | 60 | decay K=1 | constant K=1 | stage1_rel_err_abs | -0.02345 | 5 | 5 | 5 | 0 | -0.07633 | 0.005704 |
| 2b | 60 | decay K=1 | constant K=1 | learning_rel | -0.0523 | — | 7 | 3 | 0 | -0.09017 | 0.004732 |
| 2b | 60 | decay K=1 | constant K=1 | learning_rel_abs | -0.03476 | 6 | 6 | 4 | 0 | -0.07936 | 0.003849 |
| 2b | 60 | decay K=1 | constant K=1 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 60 | decay K=1 | constant K=1 | Gmax_full_over_dw | -1.937e-05 | 5 | 5 | 2 | 3 | -0.001307 | -0.0001012 |
| 2b | 60 | decay K=1 | constant K=1 | EXP_root_over_dw | -0.0004069 | 6 | 6 | 4 | 0 | -0.001726 | -8.255e-05 |
| 2b | 60 | decay K=1 | constant K=1 | dReach_over_dw | -0.0003976 | 6 | 6 | 4 | 0 | -0.001704 | -9.414e-05 |
| 2b | 50 | decay K=4 | constant K=4 | stage1_rel_err_signed | -0.02476 | — | 8 | 2 | 0 | -0.09226 | 0.01045 |
| 2b | 50 | decay K=4 | constant K=4 | stage1_rel_err_abs | -0.01252 | 7 | 7 | 3 | 0 | -0.02885 | 0.01898 |
| 2b | 50 | decay K=4 | constant K=4 | learning_rel | -0.02476 | — | 8 | 2 | 0 | -0.09146 | 0.01071 |
| 2b | 50 | decay K=4 | constant K=4 | learning_rel_abs | -0.01595 | 7 | 7 | 3 | 0 | -0.0334 | 0.01021 |
| 2b | 50 | decay K=4 | constant K=4 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 50 | decay K=4 | constant K=4 | Gmax_full_over_dw | 0 | 4 | 4 | 0 | 6 | -0.001991 | -0.0001734 |
| 2b | 50 | decay K=4 | constant K=4 | EXP_root_over_dw | -0.000474 | 7 | 7 | 3 | 0 | -0.002094 | 2.645e-05 |
| 2b | 50 | decay K=4 | constant K=4 | dReach_over_dw | -0.0003918 | 7 | 7 | 3 | 0 | -0.002069 | 7.799e-05 |
| 2b | 60 | decay K=4 | constant K=4 | stage1_rel_err_signed | -0.01596 | — | 8 | 2 | 0 | -0.1066 | 0.003156 |
| 2b | 60 | decay K=4 | constant K=4 | stage1_rel_err_abs | -0.02015 | 7 | 7 | 3 | 0 | -0.09095 | -0.006937 |
| 2b | 60 | decay K=4 | constant K=4 | learning_rel | -0.01596 | — | 8 | 2 | 0 | -0.1057 | 0.003156 |
| 2b | 60 | decay K=4 | constant K=4 | learning_rel_abs | -0.02015 | 7 | 7 | 3 | 0 | -0.09343 | -0.007939 |
| 2b | 60 | decay K=4 | constant K=4 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 60 | decay K=4 | constant K=4 | Gmax_full_over_dw | -0.0002732 | 6 | 6 | 0 | 4 | -0.001954 | -0.0002295 |
| 2b | 60 | decay K=4 | constant K=4 | EXP_root_over_dw | -0.0003599 | 7 | 7 | 3 | 0 | -0.002414 | -0.0002393 |
| 2b | 60 | decay K=4 | constant K=4 | dReach_over_dw | -0.0003603 | 7 | 7 | 3 | 0 | -0.002367 | -0.0002238 |
| 2b | 50 | decay K=8 | constant K=8 | stage1_rel_err_signed | -0.03636 | — | 6 | 4 | 0 | -0.08375 | 0.002733 |
| 2b | 50 | decay K=8 | constant K=8 | stage1_rel_err_abs | -0.002776 | 6 | 6 | 4 | 0 | -0.0363 | 0.01716 |
| 2b | 50 | decay K=8 | constant K=8 | learning_rel | -0.03636 | — | 6 | 4 | 0 | -0.08305 | 0.002258 |
| 2b | 50 | decay K=8 | constant K=8 | learning_rel_abs | -0.01506 | 8 | 8 | 2 | 0 | -0.04153 | 0.0119 |
| 2b | 50 | decay K=8 | constant K=8 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 50 | decay K=8 | constant K=8 | Gmax_full_over_dw | 0 | 4 | 4 | 0 | 6 | -0.002305 | -9.345e-05 |
| 2b | 50 | decay K=8 | constant K=8 | EXP_root_over_dw | -0.0003143 | 8 | 8 | 2 | 0 | -0.002486 | 0.0001641 |
| 2b | 50 | decay K=8 | constant K=8 | dReach_over_dw | -0.0002483 | 8 | 8 | 2 | 0 | -0.002411 | 0.0001797 |
| 2b | 60 | decay K=8 | constant K=8 | stage1_rel_err_signed | -0.01401 | — | 7 | 3 | 0 | -0.1051 | 0.008002 |
| 2b | 60 | decay K=8 | constant K=8 | stage1_rel_err_abs | -0.02006 | 6 | 6 | 4 | 0 | -0.08449 | 0.003139 |
| 2b | 60 | decay K=8 | constant K=8 | learning_rel | -0.01401 | — | 7 | 3 | 0 | -0.1048 | 0.007797 |
| 2b | 60 | decay K=8 | constant K=8 | learning_rel_abs | -0.02509 | 6 | 6 | 4 | 0 | -0.0881 | 0.001391 |
| 2b | 60 | decay K=8 | constant K=8 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 60 | decay K=8 | constant K=8 | Gmax_full_over_dw | -0.0004331 | 6 | 6 | 1 | 3 | -0.002441 | -0.000139 |
| 2b | 60 | decay K=8 | constant K=8 | EXP_root_over_dw | -0.0005393 | 6 | 6 | 4 | 0 | -0.002815 | -7.705e-05 |
| 2b | 60 | decay K=8 | constant K=8 | dReach_over_dw | -0.0005391 | 6 | 6 | 4 | 0 | -0.002789 | -6.488e-05 |
| 2b | 50 | decay K=12 | constant K=12 | stage1_rel_err_signed | -0.04004 | — | 7 | 3 | 0 | -0.07578 | -0.002561 |
| 2b | 50 | decay K=12 | constant K=12 | stage1_rel_err_abs | -0.01745 | 6 | 6 | 4 | 0 | -0.04338 | 0.004083 |
| 2b | 50 | decay K=12 | constant K=12 | learning_rel | -0.04004 | — | 7 | 3 | 0 | -0.07572 | -0.003277 |
| 2b | 50 | decay K=12 | constant K=12 | learning_rel_abs | -0.03266 | 7 | 7 | 3 | 0 | -0.04645 | 3.718e-05 |
| 2b | 50 | decay K=12 | constant K=12 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 50 | decay K=12 | constant K=12 | Gmax_full_over_dw | 0 | 4 | 4 | 1 | 5 | -0.002038 | -4.928e-06 |
| 2b | 50 | decay K=12 | constant K=12 | EXP_root_over_dw | -0.000779 | 6 | 6 | 4 | 0 | -0.002293 | 6.023e-05 |
| 2b | 50 | decay K=12 | constant K=12 | dReach_over_dw | -0.0007346 | 6 | 6 | 4 | 0 | -0.002221 | 9.024e-05 |
| 2b | 60 | decay K=12 | constant K=12 | stage1_rel_err_signed | -0.01833 | — | 7 | 3 | 0 | -0.08882 | 0.01132 |
| 2b | 60 | decay K=12 | constant K=12 | stage1_rel_err_abs | -0.017 | 6 | 6 | 4 | 0 | -0.06928 | 0.006862 |
| 2b | 60 | decay K=12 | constant K=12 | learning_rel | -0.01833 | — | 7 | 3 | 0 | -0.0897 | 0.01113 |
| 2b | 60 | decay K=12 | constant K=12 | learning_rel_abs | -0.02831 | 6 | 6 | 4 | 0 | -0.0734 | 0.00576 |
| 2b | 60 | decay K=12 | constant K=12 | inherited_rel | 0 | — | 0 | 0 | 10 | 0 | 0 |
| 2b | 60 | decay K=12 | constant K=12 | Gmax_full_over_dw | 0 | 3 | 3 | 3 | 4 | -0.001735 | 0.0001284 |
| 2b | 60 | decay K=12 | constant K=12 | EXP_root_over_dw | -0.0004093 | 6 | 6 | 4 | 0 | -0.002056 | 6.294e-05 |
| 2b | 60 | decay K=12 | constant K=12 | dReach_over_dw | -0.0004017 | 6 | 6 | 4 | 0 | -0.002046 | 7.282e-05 |

### Paired decay - constant (run records)

Source: `results/v2_pilots/pilot4/analysis/paired_summary_run_records.csv`

| family | q | metric | median | n_better | n_neg | n_pos | boot_ci95_lo | boot_ci95_hi |
|---|---|---|---|---|---|---|---|---|
| 2a | 50 | kl_final_epoch | -0.006124 | — | 10 | 0 | -0.008637 | -0.004025 |
| 2a | 50 | clip_frac | -0.02871 | — | 7 | 3 | -0.0618 | -0.01422 |
| 2a | 50 | phase_wall_sec | 0.7803 | 2 | 2 | 8 | -3.388 | 2.966 |
| 2a | 60 | kl_final_epoch | -0.00175 | — | 7 | 3 | -0.008996 | 3.337e-05 |
| 2a | 60 | clip_frac | -0.0708 | — | 10 | 0 | -0.1251 | -0.05746 |
| 2a | 60 | phase_wall_sec | -0.04 | 6 | 6 | 4 | -0.3521 | 7.167 |
| 2b | 50 | kl_final_epoch | -0.003552 | — | 6 | 4 | -0.01787 | -0.00051 |
| 2b | 50 | clip_frac | -0.07348 | — | 9 | 1 | -0.1651 | -0.04704 |
| 2b | 50 | phase_wall_sec | 1.312 | 4 | 4 | 6 | -3.506 | 7.346 |
| 2b | 50 | within_run_sd_e1_last5 | -0.7811 | 7 | 7 | 3 | -1.215 | -0.2705 |
| 2b | 50 | within_run_range_e1_last5 | -2.115 | 7 | 7 | 3 | -2.957 | -0.4928 |
| 2b | 60 | kl_final_epoch | -0.007815 | — | 7 | 3 | -0.01483 | -0.002866 |
| 2b | 60 | clip_frac | -0.08662 | — | 9 | 1 | -0.1371 | -0.06401 |
| 2b | 60 | phase_wall_sec | -0.82 | 6 | 6 | 4 | -2.107 | 5.974 |
| 2b | 60 | within_run_sd_e1_last5 | -0.5096 | 6 | 6 | 4 | -1.198 | -0.07722 |
| 2b | 60 | within_run_range_e1_last5 | -1.284 | 6 | 6 | 4 | -2.984 | -0.2492 |

### 2a run records

Source: `results/v2_pilots/pilot4/analysis/run_records.csv`

| q | arm | n | commit | dirty | lr_first | lr_last | n_would_fire | wf_median | wf_min | wf_max | wall_median | kl_median | clip_median | adv_s1_std | adv_used_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | constant | 10 | c92ee74 | False | 0.0003 | 0.0003 | 10 | 1500 | 1500 | 1500 | 69.4 | 0.006153 | 0.06694 | — | 0.04867 |
| 50 | decay | 10 | c92ee74 | False | 0.0003 | 3e-05 | 10 | 1500 | 1460 | 1500 | 70.58 | 0.004399 | 0.04634 | — | 0.04852 |
| 60 | constant | 10 | c92ee74 | False | 0.0003 | 0.0003 | 10 | 1500 | 1500 | 1500 | 69.23 | 0.006121 | 0.07466 | — | 0.04248 |
| 60 | decay | 10 | c92ee74 | False | 0.0003 | 3e-05 | 10 | 1500 | 1480 | 1500 | 69.13 | 0.004414 | 0.04883 | — | 0.043 |

### 2b run records

Source: `results/v2_pilots/pilot4/analysis/run_records.csv`

| q | arm | n | commit | dirty | lr_first | lr_last | n_would_fire | wf_median | wf_min | wf_max | wall_median | kl_median | clip_median | adv_s1_std | adv_used_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | constant | 10 | c92ee74 | False | 0.0003 | 0.0003 | 10 | 1762 | 1750 | 1925 | 153.1 | 0.004271 | 0.08654 | 1.214 | 1.214 |
| 50 | decay | 10 | c92ee74 | False | 0.0003 | 3e-05 | 10 | 1775 | 1750 | 1925 | 155.9 | 0.003357 | 0.05102 | 1.217 | 1.217 |
| 60 | constant | 10 | c92ee74 | False,True | 0.0003 | 0.0003 | 10 | 1750 | 1750 | 1850 | 144.7 | 0.004525 | 0.09883 | 1.182 | 1.182 |
| 60 | decay | 10 | c92ee74 | False,True | 0.0003 | 3e-05 | 10 | 1750 | 1750 | 1850 | 151.4 | 0.003417 | 0.05912 | 1.182 | 1.182 |

### 2b final checkpoint u2200, all runs

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv + run_records.csv`

| q | seed | arm | e1_cand | stage1_rel_err_signed | learning_rel | learning_band | inherited_rel | inherited_band | sigma_effort_at_0_t1 | within_run_sd_e1_last5 | within_run_range_e1_last5 | Gmax_full_over_dw | Gmax_full_t | Gmax_full_d | EXP_root_over_dw | dReach_over_dw | Deltamax_all_over_dw | dFull_over_dw | kl_final_epoch | clip_frac | phase_wall_sec | would_fire_update |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | constant | 47.92 | 0.02692 | 0.01342 | [0.0106, 0.0134] | 0.0135 | [0.0135, 0.0163] | 3.483 | 1.662 | 3.63 | 0.003373 | 2 | 32 | 0.001325 | 0.003378 | 0.003373 | 0.003378 | 0.008529 | 0.3595 | 153.4 | 1925 |
| 50 | 10501 | decay | 49.67 | 0.06427 | 0.05077 | [0.0480, 0.0508] | 0.0135 | [0.0135, 0.0163] | 3.565 | 1.615 | 4.134 | 0.003373 | 2 | 32 | 0.001972 | 0.004021 | 0.003373 | 0.004021 | 0.001559 | 0.006699 | 155.8 | 1925 |
| 50 | 10502 | constant | 50.27 | 0.07724 | 0.07532 | [0.0702, 0.0753] | 0.001929 | [0.0019, 0.0071] | 2.662 | 2.192 | 5.477 | 0.001985 | 2 | -60 | 0.001832 | 0.003451 | 0.001985 | 0.003451 | 0.007622 | 0.07101 | 147.4 | 1750 |
| 50 | 10502 | decay | 50.77 | 0.08802 | 0.08609 | [0.0809, 0.0861] | 0.001929 | [0.0019, 0.0071] | 2.732 | 1.024 | 2.67 | 0.002369 | 1 | 0 | 0.002369 | 0.003994 | 0.002009 | 0.003994 | 0.0001787 | 0 | 166 | 1750 |
| 50 | 10503 | constant | 44.7 | -0.04215 | -0.03443 | [-0.0385, -0.0344] | -0.007714 | [-0.0077, -0.0036] | 2.864 | 1.884 | 5.002 | 0.0007199 | 2 | 0 | 0.0004443 | 0.001049 | 0.0007199 | 0.001049 | 0.0005474 | 0.05013 | 158.3 | 1900 |
| 50 | 10503 | decay | 45.83 | -0.01801 | -0.0103 | [-0.0144, -0.0103] | -0.007714 | [-0.0077, -0.0036] | 2.961 | 0.6204 | 1.594 | 0.0007199 | 2 | 0 | 0.0001825 | 0.0007861 | 0.0007199 | 0.0007861 | 0.002811 | 0.001726 | 157.7 | 1875 |
| 50 | 10504 | constant | 50.24 | 0.07661 | 0.06611 | [0.0623, 0.0661] | 0.0105 | [0.0105, 0.0144] | 2.665 | 1.1 | 2.874 | 0.001423 | 1 | 0 | 0.001423 | 0.002307 | 0.001237 | 0.002307 | 0.002541 | 0.1099 | 145.3 | 1750 |
| 50 | 10504 | decay | 47.6 | 0.01998 | 0.00948 | [0.0056, 0.0095] | 0.0105 | [0.0105, 0.0144] | 2.736 | 1.136 | 2.556 | 0.001237 | 2 | -4 | 0.0003404 | 0.001237 | 0.001237 | 0.001237 | 0.008082 | 0.09091 | 142.5 | 1750 |
| 50 | 10505 | constant | 43.77 | -0.06216 | -0.05958 | [-0.0649, -0.0596] | -0.002571 | [-0.0026, 0.0028] | 2.605 | 1.444 | 3.587 | 0.00258 | 2 | -24 | 0.001497 | 0.003581 | 0.00258 | 0.003581 | 0.001812 | 0.0759 | 151 | 1750 |
| 50 | 10505 | decay | 44.16 | -0.05369 | -0.05111 | [-0.0565, -0.0511] | -0.002571 | [-0.0026, 0.0028] | 2.691 | 0.8607 | 2.017 | 0.00258 | 2 | -24 | 0.001226 | 0.003311 | 0.00258 | 0.003311 | 0.001678 | 0.006072 | 158 | 1750 |
| 50 | 10506 | constant | 54.96 | 0.1777 | 0.1895 | [0.1841, 0.1895] | -0.01179 | [-0.0118, -0.0064] | 3.119 | 2.787 | 5.743 | 0.01079 | 1 | 0 | 0.01079 | 0.01162 | 0.01058 | 0.01162 | 0.04471 | 0.207 | 156.4 | 1850 |
| 50 | 10506 | decay | 43.1 | -0.07646 | -0.06468 | [-0.0700, -0.0647] | -0.01179 | [-0.0118, -0.0064] | 3.187 | 1.007 | 2.436 | 0.001374 | 1 | 0 | 0.001374 | 0.002229 | 0.001181 | 0.002229 | 0.0005444 | 0.04531 | 143.2 | 1850 |
| 50 | 10507 | constant | 51.2 | 0.09724 | 0.1088 | [0.1052, 0.1088] | -0.01157 | [-0.0116, -0.0079] | 2.675 | 2.112 | 5.797 | 0.003423 | 1 | 0 | 0.003423 | 0.004305 | 0.003225 | 0.004305 | 0.0003759 | 0.1103 | 152.8 | 1750 |
| 50 | 10507 | decay | 47.59 | 0.0198 | 0.03137 | [0.0277, 0.0314] | -0.01157 | [-0.0116, -0.0079] | 2.783 | 0.1915 | 0.4713 | 0.00108 | 2 | -4 | 0.0004071 | 0.001284 | 0.00108 | 0.001284 | 0.001246 | 0 | 155.7 | 1750 |
| 50 | 10508 | constant | 49.63 | 0.06346 | 0.06689 | [0.0626, 0.0669] | -0.003429 | [-0.0034, 0.0009] | 3.129 | 1.281 | 3.456 | 0.006971 | 2 | -32 | 0.004072 | 0.008039 | 0.006971 | 0.008039 | 0.02404 | 0.1331 | 144.3 | 1775 |
| 50 | 10508 | decay | 44.71 | -0.04185 | -0.03842 | [-0.0427, -0.0384] | -0.003429 | [-0.0034, 0.0009] | 3.206 | 0.3024 | 0.7955 | 0.006971 | 2 | -32 | 0.003149 | 0.007311 | 0.006971 | 0.007311 | 0.003117 | 0.05569 | 156 | 1800 |
| 50 | 10509 | constant | 48.27 | 0.03433 | 0.04268 | [0.0380, 0.0427] | -0.008357 | [-0.0084, -0.0036] | 3.03 | 1.846 | 3.96 | 0.001296 | 2 | -40 | 0.0007628 | 0.00173 | 0.001296 | 0.00173 | 0.01332 | 0.0818 | 162.3 | 1825 |
| 50 | 10509 | decay | 48.74 | 0.04449 | 0.05285 | [0.0481, 0.0528] | -0.008357 | [-0.0084, -0.0036] | 3.094 | 2.01 | 5.286 | 0.001296 | 2 | -40 | 0.001035 | 0.002001 | 0.001296 | 0.002001 | 0.001154 | 0.005847 | 154.2 | 1825 |
| 50 | 10510 | constant | 46 | -0.01437 | -0.0253 | [-0.0311, -0.0253] | 0.01093 | [0.0109, 0.0167] | 2.553 | 1.052 | 2.913 | 0.001642 | 2 | -16 | 0.0006243 | 0.001842 | 0.001642 | 0.001842 | 0.001157 | 0.03785 | 157 | 1750 |
| 50 | 10510 | decay | 47.82 | 0.02479 | 0.01386 | [0.0081, 0.0139] | 0.01093 | [0.0109, 0.0167] | 2.677 | 1.274 | 3.236 | 0.001642 | 2 | -16 | 0.0004369 | 0.001649 | 0.001642 | 0.001649 | 0.003737 | 0.04741 | 157.3 | 1750 |
| 60 | 10501 | constant | 41.16 | 0.05846 | 0.06978 | [0.0677, 0.0703] | -0.01131 | [-0.0118, -0.0093] | 2.858 | 1.86 | 4.445 | 0.001006 | 2 | -20 | 0.0009515 | 0.001678 | 0.001006 | 0.001678 | 0.0001008 | 0.1202 | 152.6 | 1750 |
| 60 | 10501 | decay | 36.45 | -0.06274 | -0.05143 | [-0.0535, -0.0509] | -0.01131 | [-0.0118, -0.0093] | 2.933 | 0.3592 | 0.9205 | 0.001006 | 2 | -20 | 0.0006652 | 0.001409 | 0.001006 | 0.001409 | 0.004872 | 0.02707 | 154 | 1750 |
| 60 | 10502 | constant | 37.72 | -0.03005 | -0.02953 | [-0.0311, -0.0290] | -0.0005143 | [-0.0010, 0.0010] | 2.55 | 2.171 | 5.725 | 0.002156 | 2 | -16 | 0.0006679 | 0.002329 | 0.002156 | 0.002329 | 0.003401 | 0.07872 | 162.2 | 1750 |
| 60 | 10502 | decay | 40.26 | 0.03534 | 0.03585 | [0.0343, 0.0364] | -0.0005143 | [-0.0010, 0.0010] | 2.697 | 0.2731 | 0.6438 | 0.002156 | 2 | -16 | 0.0006981 | 0.002352 | 0.002156 | 0.002352 | 0.004684 | 0.01277 | 160.6 | 1750 |
| 60 | 10503 | constant | 39.02 | 0.003319 | 0.007433 | [0.0056, 0.0077] | -0.004114 | [-0.0044, -0.0023] | 2.939 | 2.403 | 5.675 | 0.0006834 | 2 | -20 | 0.0002893 | 0.0006955 | 0.0006834 | 0.0006955 | 0.01618 | 0.08706 | 154.1 | 1850 |
| 60 | 10503 | decay | 41.26 | 0.06087 | 0.06498 | [0.0632, 0.0652] | -0.004114 | [-0.0044, -0.0023] | 3.08 | 0.9357 | 2.462 | 0.0008746 | 1 | 0 | 0.0008746 | 0.001273 | 0.0006834 | 0.001273 | 0.001001 | 0.00961 | 150.3 | 1850 |
| 60 | 10504 | constant | 33.58 | -0.1366 | -0.1258 | [-0.1279, -0.1251] | -0.0108 | [-0.0116, -0.0087] | 2.445 | 1.045 | 2.798 | 0.002378 | 1 | 0 | 0.002378 | 0.003088 | 0.002186 | 0.003088 | 0.00949 | 0.07687 | 144.4 | 1750 |
| 60 | 10504 | decay | 35.75 | -0.08066 | -0.06986 | [-0.0719, -0.0691] | -0.0108 | [-0.0116, -0.0087] | 2.562 | 1.297 | 3.064 | 0.0009013 | 2 | -28 | 0.0008868 | 0.001587 | 0.0009013 | 0.001587 | 0.003683 | 0.01826 | 161.6 | 1750 |
| 60 | 10505 | constant | 44.43 | 0.1424 | 0.1442 | [0.1419, 0.1445] | -0.0018 | [-0.0021, 0.0005] | 3.15 | 1.428 | 3.742 | 0.00295 | 1 | 0 | 0.00295 | 0.003774 | 0.002802 | 0.003774 | 0.03449 | 0.2457 | 157.8 | 1825 |
| 60 | 10505 | decay | 39.38 | 0.01261 | 0.01441 | [0.0121, 0.0147] | -0.0018 | [-0.0021, 0.0005] | 3.157 | 0.5676 | 1.409 | 0.0009715 | 2 | -4 | 0.0001646 | 0.0009939 | 0.0009715 | 0.0009939 | 0.006137 | 0.009487 | 152.5 | 1750 |
| 60 | 10506 | constant | 39.71 | 0.02113 | 0.02293 | [0.0206, 0.0232] | -0.0018 | [-0.0021, 0.0005] | 2.487 | 2.813 | 6.654 | 0.0009011 | 2 | 0 | 0.0001762 | 0.0009632 | 0.0009011 | 0.0009632 | 0.005269 | 0.1505 | 144.9 | 1750 |
| 60 | 10506 | decay | 37.63 | -0.03228 | -0.03048 | [-0.0328, -0.0302] | -0.0018 | [-0.0021, 0.0005] | 2.541 | 1.102 | 2.292 | 0.0009011 | 2 | 0 | 0.0002842 | 0.001076 | 0.0009011 | 0.001076 | 0.003331 | 0.01396 | 154.7 | 1750 |
| 60 | 10507 | constant | 39 | 0.00297 | -0.0006299 | [-0.0024, -0.0004] | 0.0036 | [0.0033, 0.0054] | 2.64 | 0.6151 | 1.448 | 0.0003882 | 2 | 0 | 0.0001222 | 0.0004013 | 0.0003882 | 0.0004013 | 0.01147 | 0.1428 | 87.81 | 1800 |
| 60 | 10507 | decay | 37.41 | -0.03795 | -0.04155 | [-0.0434, -0.0413] | 0.0036 | [0.0033, 0.0054] | 2.687 | 0.8808 | 2.207 | 0.0003986 | 1 | 0 | 0.0003986 | 0.0006764 | 0.0003882 | 0.0006764 | 0.001647 | 0.01196 | 87.79 | 1800 |
| 60 | 10508 | constant | 40.88 | 0.05127 | 0.06207 | [0.0603, 0.0631] | -0.0108 | [-0.0118, -0.0090] | 3.03 | 1.209 | 2.72 | 0.0008816 | 1 | 0 | 0.0008816 | 0.00139 | 0.0008429 | 0.00139 | 0.001186 | 0.04108 | 87.8 | 1825 |
| 60 | 10508 | decay | 38.89 | 9.084e-05 | 0.01089 | [0.0091, 0.0119] | -0.0108 | [-0.0118, -0.0090] | 3.076 | 1.051 | 2.721 | 0.0008429 | 2 | 0 | 0.000354 | 0.0008637 | 0.0008429 | 0.0008637 | 0.001308 | 0.04205 | 88.24 | 1800 |
| 60 | 10509 | constant | 45.13 | 0.1604 | 0.1635 | [0.1612, 0.1637] | -0.003086 | [-0.0033, -0.0008] | 2.8 | 1.259 | 3.004 | 0.003753 | 1 | 0 | 0.003753 | 0.004542 | 0.003623 | 0.004542 | 0.02037 | 0.1419 | 88.46 | 1750 |
| 60 | 10509 | decay | 38.74 | -0.003852 | -0.000766 | [-0.0031, -0.0005] | -0.003086 | [-0.0033, -0.0008] | 2.818 | 1.346 | 2.768 | 0.000919 | 2 | -4 | 0.0001361 | 0.0009266 | 0.000919 | 0.0009266 | 0.004474 | 0.04437 | 86.18 | 1750 |
| 60 | 10510 | constant | 41.54 | 0.06826 | 0.07186 | [0.0698, 0.0721] | -0.0036 | [-0.0039, -0.0015] | 2.791 | 1.314 | 3.375 | 0.0008271 | 1 | 0 | 0.0008271 | 0.001158 | 0.0007178 | 0.001158 | 0.01787 | 0.107 | 87.81 | 1750 |
| 60 | 10510 | decay | 39.22 | 0.008496 | 0.0121 | [0.0100, 0.0124] | -0.0036 | [-0.0039, -0.0015] | 2.832 | 1.994 | 4.943 | 0.0004406 | 2 | -20 | 0.0001357 | 0.0004675 | 0.0004406 | 0.0004675 | 0.002812 | 0.02694 | 86.03 | 1750 |

### 2a final u1600, all runs

Source: `results/v2_pilots/pilot4/analysis/candidates_all.csv`

| q | seed | arm | stage2_peak_rel_err_signed | stage2_peak_rel_err_abs | stage2_peak_locfree_rel_err | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | stage2_tail_max | stage2_tail_mean_over_g2_0 | stage2_tail_max_over_g2_0 | stage2_sym_err_max | eta_T_over_dw | DeltaT_over_dw_on_max | DeltaT_over_dw_off_max | sigma_effort_at_0_t2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | constant | -0.07134 | 0.07134 | -0.07134 | 0.03371 | 0.566 | 4.907 | 0.008086 | 0.07009 | 5.473 | 0.003373 | 0.003373 | 0.001545 | 3.28 |
| 50 | 10501 | decay | -0.1222 | 0.1222 | -0.1221 | 0.03966 | 0.5229 | 4.28 | 0.007471 | 0.06114 | 3.276 | 0.004014 | 0.004014 | 0.001155 | 3.446 |
| 50 | 10502 | constant | -0.05659 | 0.05659 | -0.05545 | 0.02527 | 0.8831 | 2.665 | 0.01262 | 0.03807 | 2.903 | 0.001985 | 0.001985 | 0.0003671 | 2.63 |
| 50 | 10502 | decay | -0.06803 | 0.06803 | -0.06719 | 0.01974 | 0.8309 | 3.004 | 0.01187 | 0.04291 | 1.813 | 0.0009844 | 0.0009844 | 0.000472 | 2.732 |
| 50 | 10503 | constant | -0.05913 | 0.05913 | -0.05883 | 0.01422 | 0.4423 | 2.085 | 0.006319 | 0.02978 | 1.469 | 0.0007199 | 0.0007199 | 0.0003052 | 2.781 |
| 50 | 10503 | decay | -0.02357 | 0.02357 | -0.02246 | 0.01997 | 0.4841 | 2.489 | 0.006916 | 0.03555 | 1.47 | 0.001639 | 0.001639 | 0.0004409 | 2.818 |
| 50 | 10504 | constant | -0.07236 | 0.07236 | -0.07113 | 0.02924 | 0.5181 | 2.173 | 0.007401 | 0.03104 | 2.628 | 0.001237 | 0.001237 | 0.0002502 | 2.674 |
| 50 | 10504 | decay | -0.07125 | 0.07125 | -0.07111 | 0.01909 | 0.602 | 2.402 | 0.0086 | 0.03432 | 2.359 | 0.001159 | 0.001159 | 0.000328 | 2.772 |
| 50 | 10505 | constant | -0.06799 | 0.06799 | -0.06799 | 0.02683 | 0.512 | 2.395 | 0.007314 | 0.03421 | 1.447 | 0.00258 | 0.00258 | 0.0003923 | 2.642 |
| 50 | 10505 | decay | -0.05432 | 0.05432 | -0.05385 | 0.02018 | 0.5821 | 2.218 | 0.008316 | 0.03168 | 2.471 | 0.0009195 | 0.0009195 | 0.0002605 | 2.703 |
| 50 | 10506 | constant | -0.0687 | 0.0687 | -0.06852 | 0.02068 | 0.4967 | 2.117 | 0.007095 | 0.03024 | 2.649 | 0.001048 | 0.001048 | 0.0003199 | 3.005 |
| 50 | 10506 | decay | -0.09843 | 0.09843 | -0.09831 | 0.02399 | 0.5423 | 2.762 | 0.007748 | 0.03946 | 1.787 | 0.002643 | 0.002643 | 0.0005444 | 3.145 |
| 50 | 10507 | constant | -0.06958 | 0.06958 | -0.06958 | 0.02189 | 0.6452 | 2.925 | 0.009217 | 0.04178 | 3.737 | 0.00108 | 0.00108 | 0.0005526 | 2.69 |
| 50 | 10507 | decay | -0.05776 | 0.05776 | -0.05668 | 0.0266 | 0.6903 | 2.652 | 0.009862 | 0.03788 | 4.862 | 0.001026 | 0.001026 | 0.0005013 | 2.748 |
| 50 | 10508 | constant | -0.1286 | 0.1286 | -0.1286 | 0.05976 | 0.3528 | 2.242 | 0.00504 | 0.03203 | 2.085 | 0.006971 | 0.006971 | 0.000359 | 3.085 |
| 50 | 10508 | decay | -0.06563 | 0.06563 | -0.0655 | 0.02341 | 0.3466 | 2.257 | 0.004951 | 0.03225 | 1.266 | 0.001002 | 0.001002 | 0.0003631 | 3.111 |
| 50 | 10509 | constant | -0.05643 | 0.05643 | -0.05575 | 0.02274 | 0.3906 | 2.198 | 0.005579 | 0.0314 | 1.627 | 0.001296 | 0.001296 | 0.0002461 | 2.915 |
| 50 | 10509 | decay | -0.03538 | 0.03538 | -0.03526 | 0.01742 | 0.4467 | 2.689 | 0.006381 | 0.03841 | 2.859 | 0.001138 | 0.001138 | 0.000385 | 2.994 |
| 50 | 10510 | constant | -0.03741 | 0.03741 | -0.0368 | 0.02713 | 0.5574 | 1.656 | 0.007962 | 0.02365 | 3.799 | 0.001642 | 0.001642 | 0.0001463 | 2.598 |
| 50 | 10510 | decay | -0.05937 | 0.05937 | -0.05937 | 0.01502 | 0.642 | 2.517 | 0.009171 | 0.03595 | 2.171 | 0.0007256 | 0.0007256 | 0.0004498 | 2.711 |
| 60 | 10501 | constant | -0.02907 | 0.02907 | -0.02775 | 0.03376 | 0.5004 | 2.32 | 0.008578 | 0.03977 | 2.558 | 0.001006 | 0.001006 | 0.0003005 | 2.985 |
| 60 | 10501 | decay | -0.04693 | 0.04693 | -0.04683 | 0.01333 | 0.4856 | 2.11 | 0.008325 | 0.03617 | 1.143 | 0.0003602 | 0.0003602 | 0.000314 | 3.078 |
| 60 | 10502 | constant | 0.003541 | 0.003541 | 0.003559 | 0.0318 | 0.5769 | 1.743 | 0.009889 | 0.02988 | 1.199 | 0.002156 | 0.002156 | 0.000217 | 2.721 |
| 60 | 10502 | decay | -0.03193 | 0.03193 | -0.03149 | 0.0182 | 0.6296 | 2.26 | 0.01079 | 0.03874 | 1.693 | 0.0005909 | 0.0005909 | 0.0003609 | 2.841 |
| 60 | 10503 | constant | -0.03993 | 0.03993 | -0.03967 | 0.03087 | 0.6248 | 2.556 | 0.01071 | 0.04382 | 2.411 | 0.0006834 | 0.0006834 | 0.000465 | 3.025 |
| 60 | 10503 | decay | -0.07216 | 0.07216 | -0.07202 | 0.02393 | 0.5382 | 2.134 | 0.009227 | 0.03658 | 3.357 | 0.0008517 | 0.0008517 | 0.0002514 | 3.159 |
| 60 | 10504 | constant | -0.07134 | 0.07134 | -0.07039 | 0.02505 | 0.5342 | 2.078 | 0.009157 | 0.03563 | 2.467 | 0.0009013 | 0.0009013 | 0.0002812 | 2.704 |
| 60 | 10504 | decay | -0.04855 | 0.04855 | -0.04838 | 0.02267 | 0.5996 | 2.113 | 0.01028 | 0.03621 | 1.373 | 0.0009172 | 0.0009172 | 0.0003167 | 2.793 |
| 60 | 10505 | constant | -0.07572 | 0.07572 | -0.07572 | 0.02274 | 0.4143 | 1.925 | 0.007102 | 0.033 | 1.238 | 0.0009715 | 0.0009715 | 0.0002393 | 3.135 |
| 60 | 10505 | decay | -0.07591 | 0.07591 | -0.07558 | 0.01628 | 0.5128 | 2.507 | 0.008791 | 0.04297 | 0.9295 | 0.00098 | 0.00098 | 0.0003399 | 3.23 |
| 60 | 10506 | constant | -0.07423 | 0.07423 | -0.07422 | 0.01864 | 0.6374 | 1.825 | 0.01093 | 0.03129 | 0.8313 | 0.0009011 | 0.0009011 | 0.0002377 | 2.671 |
| 60 | 10506 | decay | -0.04289 | 0.04289 | -0.04223 | 0.01728 | 0.7317 | 2.057 | 0.01254 | 0.03527 | 1.593 | 0.0003829 | 0.0003829 | 0.0003012 | 2.738 |
| 60 | 10507 | constant | -0.04872 | 0.04872 | -0.04815 | 0.02106 | 0.5345 | 1.708 | 0.009163 | 0.02927 | 2.651 | 0.0003882 | 0.0003882 | 0.0001492 | 2.76 |
| 60 | 10507 | decay | -0.05151 | 0.05151 | -0.05136 | 0.01729 | 0.5576 | 2.068 | 0.009559 | 0.03544 | 2.266 | 0.000434 | 0.000434 | 0.0002273 | 2.846 |
| 60 | 10508 | constant | -0.07179 | 0.07179 | -0.06356 | 0.03297 | 0.4881 | 1.865 | 0.008367 | 0.03197 | 3.963 | 0.0008429 | 0.0008429 | 0.0002485 | 3.142 |
| 60 | 10508 | decay | -0.09174 | 0.09174 | -0.09052 | 0.02454 | 0.61 | 2.318 | 0.01046 | 0.03974 | 2.833 | 0.001497 | 0.001497 | 0.0003812 | 3.242 |
| 60 | 10509 | constant | -0.07492 | 0.07492 | -0.07459 | 0.02083 | 0.5027 | 1.877 | 0.008618 | 0.03218 | 2.354 | 0.000919 | 0.000919 | 0.0001825 | 2.874 |
| 60 | 10509 | decay | -0.05088 | 0.05088 | -0.05048 | 0.02145 | 0.5067 | 1.879 | 0.008686 | 0.03222 | 1.952 | 0.0007645 | 0.0007645 | 0.000238 | 2.978 |
| 60 | 10510 | constant | -0.04158 | 0.04158 | -0.03975 | 0.01872 | 0.5282 | 2.12 | 0.009056 | 0.03634 | 2.369 | 0.0004406 | 0.0004406 | 0.0003207 | 2.892 |
| 60 | 10510 | decay | -0.02097 | 0.02097 | -0.02071 | 0.02651 | 0.5035 | 2.034 | 0.008632 | 0.03486 | 1.037 | 0.001453 | 0.001453 | 0.0002849 | 2.999 |

### 2b stability

Source: `results/v2_pilots/pilot4/analysis/stability_2b.csv`

| q | arm | across_seed_sd_final_e1 | across_seed_iqr_final_e1 | median_sigma1_0 | median_within_run_sd_last5 | median_within_run_range_last5 |
|---|---|---|---|---|---|---|
| 50 | constant | 3.327 | 3.786 | 2.77 | 1.754 | 3.795 |
| 50 | decay | 2.487 | 3.521 | 2.872 | 1.016 | 2.496 |
| 60 | constant | 3.307 | 2.44 | 2.795 | 1.371 | 3.558 |
| 60 | decay | 1.696 | 1.871 | 2.825 | 0.9931 | 2.377 |

### RNG divergence decay vs constant (global update)

Source: `results/v2_pilots/pilot4/analysis/rng_divergence.csv`

| family | q | env | learn | opp | start | minibatch |
|---|---|---|---|---|---|---|
| 2a | 50 | never (10/10) | 10/10; min 1204, median 1209, max 1240 | 10/10; min 1221, median 1222, max 1236 | never (10/10) | never (10/10) |
| 2a | 60 | never (10/10) | 10/10; min 1205, median 1208, max 1229 | 10/10; min 1221, median 1224, max 1231 | never (10/10) | never (10/10) |
| 2b | 50 | never (10/10) | 10/10; min 1613, median 1672, max 1805 | 10/10; min 1619, median 1659, max 1909 | never (10/10) | never (10/10) |
| 2b | 60 | never (10/10) | 10/10; min 1607, median 1680, max 2168 | 10/10; min 1628, median 1714, max 1978 | never (10/10) | never (10/10) |

### 1d supervised fits (each init)

Source: `results/v2_pilots/pilot4/analysis/repr_floor/fits.csv`

| q | init_seed | steps | stop | final_loss_mse | stage2_peak_rel_err_signed | stage2_peak_locfree_rel_err | stage2_peak_locfree_argmax_d | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | stage2_tail_max | stage2_sym_err_max | eta_T_over_dw | Gmax_full_over_dw | max_abs_resid | max_abs_resid_at_d | tail_min_fit | tail_max_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 0 | 300000 | max_steps | 0.001073 | -2.309e-05 | -2.309e-05 | 0 | 0.004416 | 0.003162 | 0.217 | 1.509 | 3.046e-06 | 3.046e-06 | 0.217 | -100 | 0.0001 | 0.217 |
| 50 | 1 | 300000 | max_steps | 0.000153 | -1.9e-05 | 0.001318 | 0.5 | 0.002146 | 0.001697 | 0.07285 | 0.6851 | 3.314e-07 | 3.314e-07 | 0.07285 | 100 | 0.0001 | 0.07285 |
| 50 | 2 | 300000 | max_steps | 0.0003907 | 1.453e-05 | 1.453e-05 | 0 | 0.004221 | 0.00211 | 0.1383 | 0.4177 | 8.892e-07 | 8.892e-07 | 0.1383 | 100 | 0.0001 | 0.1383 |
| 50 | 3 | 300000 | max_steps | 0.0002663 | -2.003e-05 | 8.849e-06 | 0.5 | 0.004034 | 0.001444 | 0.09482 | 0.3049 | 5.222e-07 | 5.222e-07 | 0.09482 | 100 | 0.0001 | 0.09482 |
| 50 | 4 | 300000 | max_steps | 0.005794 | 0.000759 | 0.000759 | 0 | 0.001855 | 0.007546 | 0.1142 | 0.126 | 8.245e-07 | 8.245e-07 | 0.1142 | 100 | 0.0002335 | 0.1142 |
| 60 | 0 | 300000 | max_steps | 6.847e-05 | -4.632e-06 | -4.632e-06 | 0 | 0.001311 | 0.003303 | 0.05792 | 0.1169 | 2.086e-07 | 2.086e-07 | 0.05792 | 120 | 0.001143 | 0.05792 |
| 60 | 1 | 300000 | max_steps | 0.000538 | -0.0007393 | -0.0007393 | 0 | 0.001623 | 0.003653 | 0.04235 | 0.3997 | 2.356e-07 | 2.356e-07 | 0.04757 | -40 | 0.001066 | 0.04235 |
| 60 | 2 | 300000 | max_steps | 5.414e-05 | 1.006e-05 | 1.006e-05 | 0 | 0.001275 | 0.002987 | 0.03844 | 0.1323 | 9.443e-08 | 9.443e-08 | 0.03844 | 120 | 0.001118 | 0.03844 |
| 60 | 3 | 300000 | max_steps | 0.0001314 | -0.000961 | -0.000961 | 0 | 0.001323 | 0.004828 | 0.05454 | 0.07127 | 4.834e-07 | 4.834e-07 | 0.07087 | 16 | 0.00175 | 0.05454 |
| 60 | 4 | 300000 | max_steps | 0.0002798 | -0.0005967 | -0.0005967 | 0 | 0.00132 | 0.004659 | 0.04804 | 0.0927 | 2.117e-07 | 2.117e-07 | 0.04804 | -120 | 0.002063 | 0.04804 |

### 1d supervised fits (median over 5 inits)

Source: `results/v2_pilots/pilot4/analysis/repr_floor/fits.csv`

| q | stage2_peak_rel_err_signed | stage2_peak_locfree_rel_err | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | stage2_tail_max | stage2_sym_err_max | eta_T_over_dw | Gmax_full_over_dw |
|---|---|---|---|---|---|---|---|---|
| 50 | -1.9e-05 | 1.453e-05 | 0.004034 | 0.00211 | 0.1142 | 0.4177 | 8.245e-07 | 8.245e-07 |
| 60 | -0.0005967 | -0.0005967 | 0.00132 | 0.003653 | 0.04804 | 0.1169 | 2.117e-07 | 2.117e-07 |

### 1d RL u1600 (median over 10 seeds), same metrics

Source: `results/v2_pilots/pilot4/analysis/repr_floor/rl_u1600.csv`

| q | stage2_peak_rel_err_signed | stage2_peak_locfree_rel_err | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | stage2_tail_max | stage2_sym_err_max | eta_T_over_dw |
|---|---|---|---|---|---|---|---|
| 50 | -0.06835 | -0.06826 | 0.02605 | 0.515 | 2.22 | 2.638 | 0.001469 |
| 60 | -0.06003 | -0.05585 | 0.02389 | 0.5312 | 1.901 | 2.39 | 0.0009012 |

### 1d three-way peak gap at d = 0 (effort units)

Source: `results/v2_pilots/pilot4/analysis/repr_floor/three_way_peak_gap.csv`

| q | quantity | n | median | min | max | median_over_e2star0 |
|---|---|---|---|---|---|---|
| 50 | rl_peak_gap_d0 | 10 | 4.784 | 2.619 | 9.002 | 0.06835 |
| 50 | smoothing_predicted_gap_d0 | 10 | 2.159 | 2.05 | 2.59 | 0.03085 |
| 50 | supervised_floor_gap_d0 | 5 | 0.00133 | -0.05313 | 0.001616 | 1.9e-05 |
| 50 | rl_locfree_rel_err | 10 | -0.06826 | -0.1286 | -0.0368 | — |
| 50 | floor_locfree_rel_err | 5 | 1.453e-05 | -2.309e-05 | 0.001318 | — |
| 60 | rl_peak_gap_d0 | 10 | 3.502 | -0.2065 | 4.417 | 0.06003 |
| 60 | smoothing_predicted_gap_d0 | 10 | 1.581 | 1.464 | 1.723 | 0.0271 |
| 60 | supervised_floor_gap_d0 | 5 | 0.0348 | -0.0005871 | 0.05606 | 0.0005967 |
| 60 | rl_locfree_rel_err | 10 | -0.05585 | -0.07572 | 0.003559 | — |
| 60 | floor_locfree_rel_err | 5 | -0.0005967 | -0.000961 | 1.006e-05 | — |

### Distribution tables, Phase A

Source: `results/v2_pilots/pilot4/analysis/gate_distribution_tables.csv`

| arm | K | q | metric | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| constant | 1 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| constant | 4 | 50 | eta_T_over_dw | 0.0004987 | 0.0006821 | 0.001018 | 0.001168 | 0.001855 | 0.00223 | 0.002563 |
| constant | 8 | 50 | eta_T_over_dw | 0.0005877 | 0.000866 | 0.001068 | 0.001285 | 0.001526 | 0.002608 | 0.00304 |
| decay | 1 | 50 | eta_T_over_dw | 0.0007256 | 0.0009001 | 0.0009888 | 0.001082 | 0.001519 | 0.00278 | 0.004014 |
| decay | 4 | 50 | eta_T_over_dw | 0.0006195 | 0.0006471 | 0.0006597 | 0.001036 | 0.001521 | 0.002795 | 0.003426 |
| decay | 8 | 50 | eta_T_over_dw | 0.0004457 | 0.0006706 | 0.0007642 | 0.001126 | 0.001284 | 0.002363 | 0.004195 |
| constant | 1 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| constant | 4 | 60 | eta_T_over_dw | 0.0004663 | 0.0005175 | 0.0005915 | 0.0007207 | 0.0007453 | 0.0009099 | 0.001142 |
| constant | 8 | 60 | eta_T_over_dw | 0.0003168 | 0.0003858 | 0.0004095 | 0.0006428 | 0.0007601 | 0.0008653 | 0.00097 |
| decay | 1 | 60 | eta_T_over_dw | 0.0003602 | 0.0003806 | 0.0004732 | 0.0008081 | 0.0009643 | 0.001458 | 0.001497 |
| decay | 4 | 60 | eta_T_over_dw | 0.000354 | 0.0003982 | 0.0004425 | 0.0006526 | 0.0008819 | 0.0009596 | 0.001131 |
| decay | 8 | 60 | eta_T_over_dw | 0.00037 | 0.0003752 | 0.0004112 | 0.0005716 | 0.0007766 | 0.0008276 | 0.0008496 |
| constant | 1 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| constant | 4 | 50 | stage2_peak_locfree_rel_err | -0.09311 | -0.09274 | -0.08075 | -0.06247 | -0.05435 | -0.04822 | -0.04065 |
| constant | 8 | 50 | stage2_peak_locfree_rel_err | -0.1069 | -0.09748 | -0.07593 | -0.06972 | -0.05426 | -0.03868 | -0.03396 |
| decay | 1 | 50 | stage2_peak_locfree_rel_err | -0.1221 | -0.1007 | -0.07013 | -0.06243 | -0.05456 | -0.03398 | -0.02246 |
| decay | 4 | 50 | stage2_peak_locfree_rel_err | -0.1115 | -0.09958 | -0.07868 | -0.06831 | -0.05189 | -0.03908 | -0.03662 |
| decay | 8 | 50 | stage2_peak_locfree_rel_err | -0.1205 | -0.09201 | -0.07369 | -0.06634 | -0.04799 | -0.034 | -0.02867 |
| constant | 1 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| constant | 4 | 60 | stage2_peak_locfree_rel_err | -0.08172 | -0.07425 | -0.06654 | -0.06217 | -0.05594 | -0.05117 | -0.03125 |
| constant | 8 | 60 | stage2_peak_locfree_rel_err | -0.0769 | -0.0681 | -0.0653 | -0.0551 | -0.04632 | -0.04383 | -0.04222 |
| decay | 1 | 60 | stage2_peak_locfree_rel_err | -0.09052 | -0.07707 | -0.06686 | -0.04943 | -0.04338 | -0.03041 | -0.02071 |
| decay | 4 | 60 | stage2_peak_locfree_rel_err | -0.08043 | -0.0752 | -0.07275 | -0.059 | -0.0498 | -0.04167 | -0.03799 |
| decay | 8 | 60 | stage2_peak_locfree_rel_err | -0.07081 | -0.06853 | -0.06799 | -0.05858 | -0.04811 | -0.0441 | -0.03961 |
| constant | 1 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| constant | 4 | 50 | stage2_peak_rel_err_abs | 0.04077 | 0.04837 | 0.05453 | 0.06294 | 0.08174 | 0.09316 | 0.09494 |
| constant | 8 | 50 | stage2_peak_rel_err_abs | 0.03448 | 0.03956 | 0.05445 | 0.07022 | 0.07666 | 0.0978 | 0.107 |
| decay | 1 | 50 | stage2_peak_rel_err_abs | 0.02357 | 0.0342 | 0.05518 | 0.0625 | 0.07044 | 0.1008 | 0.1222 |
| decay | 4 | 50 | stage2_peak_rel_err_abs | 0.03757 | 0.03928 | 0.05234 | 0.06911 | 0.07868 | 0.1002 | 0.1116 |
| decay | 8 | 50 | stage2_peak_rel_err_abs | 0.02928 | 0.03464 | 0.04943 | 0.06673 | 0.07379 | 0.09278 | 0.1212 |
| constant | 1 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| constant | 4 | 60 | stage2_peak_rel_err_abs | 0.03148 | 0.0512 | 0.05679 | 0.06243 | 0.06688 | 0.07435 | 0.08176 |
| constant | 8 | 60 | stage2_peak_rel_err_abs | 0.04332 | 0.04394 | 0.04679 | 0.05595 | 0.06533 | 0.06988 | 0.07701 |
| decay | 1 | 60 | stage2_peak_rel_err_abs | 0.02097 | 0.03083 | 0.0439 | 0.04971 | 0.067 | 0.0775 | 0.09174 |
| decay | 4 | 60 | stage2_peak_rel_err_abs | 0.0385 | 0.04256 | 0.04982 | 0.05929 | 0.07342 | 0.07551 | 0.08114 |
| decay | 8 | 60 | stage2_peak_rel_err_abs | 0.0403 | 0.04479 | 0.0482 | 0.05884 | 0.06891 | 0.07113 | 0.07207 |
| constant | 1 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| constant | 4 | 50 | stage2_peak_rel_err_signed | -0.09494 | -0.09316 | -0.08174 | -0.06294 | -0.05453 | -0.04837 | -0.04077 |
| constant | 8 | 50 | stage2_peak_rel_err_signed | -0.107 | -0.0978 | -0.07666 | -0.07022 | -0.05445 | -0.03956 | -0.03448 |
| decay | 1 | 50 | stage2_peak_rel_err_signed | -0.1222 | -0.1008 | -0.07044 | -0.0625 | -0.05518 | -0.0342 | -0.02357 |
| decay | 4 | 50 | stage2_peak_rel_err_signed | -0.1116 | -0.1002 | -0.07868 | -0.06911 | -0.05234 | -0.03928 | -0.03757 |
| decay | 8 | 50 | stage2_peak_rel_err_signed | -0.1212 | -0.09278 | -0.07379 | -0.06673 | -0.04943 | -0.03464 | -0.02928 |
| constant | 1 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| constant | 4 | 60 | stage2_peak_rel_err_signed | -0.08176 | -0.07435 | -0.06688 | -0.06243 | -0.05679 | -0.0512 | -0.03148 |
| constant | 8 | 60 | stage2_peak_rel_err_signed | -0.07701 | -0.06988 | -0.06533 | -0.05595 | -0.04679 | -0.04394 | -0.04332 |
| decay | 1 | 60 | stage2_peak_rel_err_signed | -0.09174 | -0.0775 | -0.067 | -0.04971 | -0.0439 | -0.03083 | -0.02097 |
| decay | 4 | 60 | stage2_peak_rel_err_signed | -0.08114 | -0.07551 | -0.07342 | -0.05929 | -0.04982 | -0.04256 | -0.0385 |
| decay | 8 | 60 | stage2_peak_rel_err_signed | -0.07207 | -0.07113 | -0.06891 | -0.05884 | -0.0482 | -0.04479 | -0.0403 |
| constant | 1 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| constant | 4 | 50 | stage2_rmse_pos_over_g2_0 | 0.01642 | 0.01778 | 0.01888 | 0.0201 | 0.02321 | 0.02718 | 0.02746 |
| constant | 8 | 50 | stage2_rmse_pos_over_g2_0 | 0.01438 | 0.01691 | 0.01776 | 0.01917 | 0.02075 | 0.02488 | 0.02906 |
| decay | 1 | 50 | stage2_rmse_pos_over_g2_0 | 0.01502 | 0.01718 | 0.01925 | 0.02007 | 0.02384 | 0.0279 | 0.03966 |
| decay | 4 | 50 | stage2_rmse_pos_over_g2_0 | 0.01472 | 0.01511 | 0.01635 | 0.01735 | 0.02154 | 0.02582 | 0.03071 |
| decay | 8 | 50 | stage2_rmse_pos_over_g2_0 | 0.01416 | 0.0146 | 0.0159 | 0.01713 | 0.01899 | 0.02224 | 0.03369 |
| constant | 1 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| constant | 4 | 60 | stage2_rmse_pos_over_g2_0 | 0.01626 | 0.01717 | 0.0173 | 0.0179 | 0.01925 | 0.02039 | 0.02177 |
| constant | 8 | 60 | stage2_rmse_pos_over_g2_0 | 0.01452 | 0.01515 | 0.01559 | 0.01711 | 0.01975 | 0.02153 | 0.02156 |
| decay | 1 | 60 | stage2_rmse_pos_over_g2_0 | 0.01333 | 0.01598 | 0.01729 | 0.01982 | 0.02362 | 0.02474 | 0.02651 |
| decay | 4 | 60 | stage2_rmse_pos_over_g2_0 | 0.01363 | 0.01423 | 0.01716 | 0.01774 | 0.01848 | 0.01885 | 0.02059 |
| decay | 8 | 60 | stage2_rmse_pos_over_g2_0 | 0.01388 | 0.01434 | 0.01549 | 0.01627 | 0.01794 | 0.02132 | 0.02282 |
| constant | 1 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| constant | 4 | 50 | stage2_tail_max_over_g2_0 | 0.02726 | 0.0292 | 0.03218 | 0.03334 | 0.03433 | 0.03809 | 0.06256 |
| constant | 8 | 50 | stage2_tail_max_over_g2_0 | 0.03011 | 0.03043 | 0.03183 | 0.03343 | 0.03557 | 0.04449 | 0.06118 |
| decay | 1 | 50 | stage2_tail_max_over_g2_0 | 0.03168 | 0.03219 | 0.03463 | 0.03692 | 0.03919 | 0.04473 | 0.06114 |
| decay | 4 | 50 | stage2_tail_max_over_g2_0 | 0.03141 | 0.03368 | 0.03489 | 0.03737 | 0.03866 | 0.04335 | 0.06362 |
| decay | 8 | 50 | stage2_tail_max_over_g2_0 | 0.03114 | 0.03266 | 0.03447 | 0.03572 | 0.03713 | 0.04069 | 0.06423 |
| constant | 1 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| constant | 4 | 60 | stage2_tail_max_over_g2_0 | 0.02999 | 0.03025 | 0.03212 | 0.03538 | 0.03753 | 0.03901 | 0.03912 |
| constant | 8 | 60 | stage2_tail_max_over_g2_0 | 0.03089 | 0.03222 | 0.03332 | 0.0354 | 0.03765 | 0.03827 | 0.03831 |
| decay | 1 | 60 | stage2_tail_max_over_g2_0 | 0.03222 | 0.0346 | 0.03531 | 0.03619 | 0.0382 | 0.04007 | 0.04297 |
| decay | 4 | 60 | stage2_tail_max_over_g2_0 | 0.03181 | 0.03409 | 0.03514 | 0.03637 | 0.038 | 0.03912 | 0.04044 |
| decay | 8 | 60 | stage2_tail_max_over_g2_0 | 0.03158 | 0.03254 | 0.03449 | 0.03638 | 0.03803 | 0.03936 | 0.04041 |
| constant | 1 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| constant | 4 | 50 | stage2_tail_mean_over_g2_0 | 0.005495 | 0.005756 | 0.006867 | 0.007931 | 0.008077 | 0.009736 | 0.0127 |
| constant | 8 | 50 | stage2_tail_mean_over_g2_0 | 0.005799 | 0.006122 | 0.006908 | 0.008027 | 0.008366 | 0.009555 | 0.0123 |
| decay | 1 | 50 | stage2_tail_mean_over_g2_0 | 0.004951 | 0.006238 | 0.007055 | 0.008032 | 0.009028 | 0.01006 | 0.01187 |
| decay | 4 | 50 | stage2_tail_mean_over_g2_0 | 0.00525 | 0.006395 | 0.006783 | 0.008436 | 0.009089 | 0.009798 | 0.01154 |
| decay | 8 | 50 | stage2_tail_mean_over_g2_0 | 0.005382 | 0.006385 | 0.006755 | 0.008559 | 0.009089 | 0.009868 | 0.01177 |
| constant | 1 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| constant | 4 | 60 | stage2_tail_mean_over_g2_0 | 0.007741 | 0.008168 | 0.008731 | 0.009264 | 0.00975 | 0.01038 | 0.01107 |
| constant | 8 | 60 | stage2_tail_mean_over_g2_0 | 0.008003 | 0.008152 | 0.008927 | 0.009446 | 0.009934 | 0.01057 | 0.01157 |
| decay | 1 | 60 | stage2_tail_mean_over_g2_0 | 0.008325 | 0.008601 | 0.008712 | 0.009393 | 0.01041 | 0.01097 | 0.01254 |
| decay | 4 | 60 | stage2_tail_mean_over_g2_0 | 0.008178 | 0.008306 | 0.008503 | 0.009815 | 0.01042 | 0.01093 | 0.01259 |
| decay | 8 | 60 | stage2_tail_mean_over_g2_0 | 0.00826 | 0.008264 | 0.008435 | 0.009804 | 0.0105 | 0.01101 | 0.01274 |

### Distribution tables, Phase B

Source: `results/v2_pilots/pilot4/analysis/gate_distribution_tables.csv`

| arm | K | q | metric | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| constant | 1 | 50 | EXP_root_over_dw | 0.0004443 | 0.0006063 | 0.0009034 | 0.00146 | 0.003025 | 0.004744 | 0.01079 |
| constant | 4 | 50 | EXP_root_over_dw | 0.0001632 | 0.0005439 | 0.0008332 | 0.001366 | 0.003712 | 0.00619 | 0.006218 |
| constant | 8 | 50 | EXP_root_over_dw | 0.0002234 | 0.0004095 | 0.0008106 | 0.00114 | 0.002876 | 0.004201 | 0.00978 |
| constant | 12 | 50 | EXP_root_over_dw | 0.0002052 | 0.0004748 | 0.000858 | 0.001875 | 0.003061 | 0.004138 | 0.008448 |
| decay | 1 | 50 | EXP_root_over_dw | 0.0001825 | 0.0003246 | 0.0004146 | 0.00113 | 0.001822 | 0.002447 | 0.003149 |
| decay | 4 | 50 | EXP_root_over_dw | 0.0001957 | 0.0004584 | 0.0006984 | 0.0009912 | 0.002293 | 0.003169 | 0.003208 |
| decay | 8 | 50 | EXP_root_over_dw | 0.000164 | 0.0002011 | 0.0004178 | 0.0008426 | 0.002665 | 0.003014 | 0.003042 |
| decay | 12 | 50 | EXP_root_over_dw | 0.0001345 | 0.0001996 | 0.0004992 | 0.0007759 | 0.002575 | 0.003082 | 0.004163 |
| constant | 1 | 60 | EXP_root_over_dw | 0.0001222 | 0.0001708 | 0.0003839 | 0.0008544 | 0.002021 | 0.00303 | 0.003753 |
| constant | 4 | 60 | EXP_root_over_dw | 0.0001466 | 0.0001827 | 0.0003352 | 0.001142 | 0.002641 | 0.003948 | 0.005093 |
| constant | 8 | 60 | EXP_root_over_dw | 0.0001447 | 0.0002217 | 0.0006332 | 0.001339 | 0.001967 | 0.002933 | 0.007975 |
| constant | 12 | 60 | EXP_root_over_dw | 0.0001129 | 0.0001392 | 0.0006298 | 0.001049 | 0.001947 | 0.003107 | 0.005886 |
| decay | 1 | 60 | EXP_root_over_dw | 0.0001357 | 0.000136 | 0.0001945 | 0.0003763 | 0.0006899 | 0.0008758 | 0.0008868 |
| decay | 4 | 60 | EXP_root_over_dw | 0.0001233 | 0.0001402 | 0.0001969 | 0.0003822 | 0.0007302 | 0.0007819 | 0.0008369 |
| decay | 8 | 60 | EXP_root_over_dw | 0.0002407 | 0.000273 | 0.0003133 | 0.0006352 | 0.0007736 | 0.001233 | 0.001683 |
| decay | 12 | 60 | EXP_root_over_dw | 0.0001868 | 0.0002048 | 0.0002897 | 0.0006163 | 0.001147 | 0.001595 | 0.001712 |
| constant | 1 | 50 | Gmax_full_over_dw | 0.0007199 | 0.001239 | 0.001478 | 0.002282 | 0.00341 | 0.007353 | 0.01079 |
| constant | 4 | 50 | Gmax_full_over_dw | 0.0007199 | 0.001176 | 0.001339 | 0.002805 | 0.005484 | 0.006293 | 0.006971 |
| constant | 8 | 50 | Gmax_full_over_dw | 0.0008063 | 0.001052 | 0.001339 | 0.002482 | 0.00329 | 0.007252 | 0.00978 |
| constant | 12 | 50 | Gmax_full_over_dw | 0.00108 | 0.001221 | 0.00158 | 0.002623 | 0.003328 | 0.007119 | 0.008448 |
| decay | 1 | 50 | Gmax_full_over_dw | 0.0007199 | 0.001044 | 0.001252 | 0.001508 | 0.002527 | 0.003733 | 0.006971 |
| decay | 4 | 50 | Gmax_full_over_dw | 0.0007199 | 0.001044 | 0.001252 | 0.001933 | 0.002514 | 0.003733 | 0.006971 |
| decay | 8 | 50 | Gmax_full_over_dw | 0.0007199 | 0.001044 | 0.001252 | 0.001874 | 0.002784 | 0.003733 | 0.006971 |
| decay | 12 | 50 | Gmax_full_over_dw | 0.0007199 | 0.001044 | 0.001252 | 0.001564 | 0.002866 | 0.004444 | 0.006971 |
| constant | 1 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0006539 | 0.0008407 | 0.0009536 | 0.002322 | 0.00303 | 0.003753 |
| constant | 4 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0004353 | 0.000939 | 0.001256 | 0.00286 | 0.003948 | 0.005093 |
| constant | 8 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0004353 | 0.001008 | 0.001424 | 0.002148 | 0.002933 | 0.007975 |
| constant | 12 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0004353 | 0.0009012 | 0.001187 | 0.002011 | 0.003107 | 0.005886 |
| decay | 1 | 60 | Gmax_full_over_dw | 0.0003986 | 0.0004364 | 0.0008508 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| decay | 4 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0004353 | 0.0008384 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| decay | 8 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0006556 | 0.0008574 | 0.0009101 | 0.0009974 | 0.00173 | 0.002156 |
| decay | 12 | 60 | Gmax_full_over_dw | 0.0003882 | 0.0008386 | 0.0009057 | 0.000971 | 0.001156 | 0.001757 | 0.002156 |
| constant | 1 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| constant | 4 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| constant | 8 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| constant | 12 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| decay | 1 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| decay | 4 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| decay | 8 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| decay | 12 | 50 | eta_T_over_dw | 0.0007199 | 0.001015 | 0.001119 | 0.001469 | 0.002431 | 0.003733 | 0.006971 |
| constant | 1 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| constant | 4 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| constant | 8 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| constant | 12 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| decay | 1 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| decay | 4 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| decay | 8 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| decay | 12 | 60 | eta_T_over_dw | 0.0003882 | 0.0004353 | 0.0007233 | 0.0009012 | 0.0009584 | 0.001121 | 0.002156 |
| constant | 1 | 50 | stage1_rel_err_abs | 0.01437 | 0.02567 | 0.03628 | 0.06281 | 0.07709 | 0.1053 | 0.1777 |
| constant | 4 | 50 | stage1_rel_err_abs | 0.01014 | 0.01354 | 0.018 | 0.05367 | 0.08246 | 0.1337 | 0.1436 |
| constant | 8 | 50 | stage1_rel_err_abs | 0.005005 | 0.01277 | 0.02768 | 0.04398 | 0.07093 | 0.09842 | 0.1791 |
| constant | 12 | 50 | stage1_rel_err_abs | 0.001757 | 0.003798 | 0.04771 | 0.06485 | 0.0804 | 0.1003 | 0.1674 |
| decay | 1 | 50 | stage1_rel_err_abs | 0.01801 | 0.01962 | 0.02118 | 0.04317 | 0.06162 | 0.07762 | 0.08802 |
| decay | 4 | 50 | stage1_rel_err_abs | 0.01994 | 0.02157 | 0.02749 | 0.04553 | 0.07779 | 0.09817 | 0.09909 |
| decay | 8 | 50 | stage1_rel_err_abs | 0.008304 | 0.009129 | 0.01734 | 0.0356 | 0.0819 | 0.0963 | 0.09713 |
| decay | 12 | 50 | stage1_rel_err_abs | 0.009957 | 0.01233 | 0.01734 | 0.02476 | 0.07281 | 0.1006 | 0.1155 |
| constant | 1 | 60 | stage1_rel_err_abs | 0.00297 | 0.003284 | 0.02336 | 0.05486 | 0.1195 | 0.1442 | 0.1604 |
| constant | 4 | 60 | stage1_rel_err_abs | 0.01084 | 0.01095 | 0.01963 | 0.07151 | 0.1219 | 0.1645 | 0.1882 |
| constant | 8 | 60 | stage1_rel_err_abs | 0.01045 | 0.01754 | 0.0513 | 0.07859 | 0.1079 | 0.1365 | 0.2366 |
| constant | 12 | 60 | stage1_rel_err_abs | 0.00774 | 0.009696 | 0.04406 | 0.08221 | 0.1015 | 0.1298 | 0.2027 |
| decay | 1 | 60 | stage1_rel_err_abs | 9.084e-05 | 0.003476 | 0.009526 | 0.03381 | 0.05514 | 0.06453 | 0.08066 |
| decay | 4 | 60 | stage1_rel_err_abs | 0.0009903 | 0.003286 | 0.01699 | 0.0374 | 0.04757 | 0.05894 | 0.06158 |
| decay | 8 | 60 | stage1_rel_err_abs | 0.02181 | 0.02449 | 0.03328 | 0.04758 | 0.06705 | 0.07335 | 0.09611 |
| decay | 12 | 60 | stage1_rel_err_abs | 0.01097 | 0.01255 | 0.02345 | 0.04318 | 0.08383 | 0.08799 | 0.09712 |
| constant | 1 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| constant | 4 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| constant | 8 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| constant | 12 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| decay | 1 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| decay | 4 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| decay | 8 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| decay | 12 | 50 | stage2_peak_locfree_rel_err | -0.1286 | -0.07707 | -0.07074 | -0.06826 | -0.05652 | -0.05358 | -0.0368 |
| constant | 1 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| constant | 4 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| constant | 8 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| constant | 12 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| decay | 1 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| decay | 4 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| decay | 8 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| decay | 12 | 60 | stage2_peak_locfree_rel_err | -0.07572 | -0.07471 | -0.07326 | -0.05585 | -0.03969 | -0.02462 | 0.003559 |
| constant | 1 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| constant | 4 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| constant | 8 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| constant | 12 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| decay | 1 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| decay | 4 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| decay | 8 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| decay | 12 | 50 | stage2_peak_rel_err_abs | 0.03741 | 0.05453 | 0.05722 | 0.06835 | 0.0709 | 0.07798 | 0.1286 |
| constant | 1 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| constant | 4 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| constant | 8 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| constant | 12 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| decay | 1 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| decay | 4 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| decay | 8 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| decay | 12 | 60 | stage2_peak_rel_err_abs | 0.003541 | 0.02652 | 0.04034 | 0.06003 | 0.07362 | 0.075 | 0.07572 |
| constant | 1 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| constant | 4 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| constant | 8 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| constant | 12 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| decay | 1 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| decay | 4 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| decay | 8 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| decay | 12 | 50 | stage2_peak_rel_err_signed | -0.1286 | -0.07798 | -0.0709 | -0.06835 | -0.05722 | -0.05453 | -0.03741 |
| constant | 1 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| constant | 4 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| constant | 8 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| constant | 12 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| decay | 1 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| decay | 4 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| decay | 8 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| decay | 12 | 60 | stage2_peak_rel_err_signed | -0.07572 | -0.075 | -0.07362 | -0.06003 | -0.04034 | -0.02581 | 0.003541 |
| constant | 1 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| constant | 4 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| constant | 8 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| constant | 12 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| decay | 1 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| decay | 4 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| decay | 8 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| decay | 12 | 50 | stage2_rmse_pos_over_g2_0 | 0.01422 | 0.02004 | 0.0221 | 0.02605 | 0.02871 | 0.03632 | 0.05976 |
| constant | 1 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| constant | 4 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| constant | 8 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| constant | 12 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| decay | 1 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| decay | 4 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| decay | 8 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| decay | 12 | 60 | stage2_rmse_pos_over_g2_0 | 0.01864 | 0.01871 | 0.02089 | 0.02389 | 0.03157 | 0.03305 | 0.03376 |
| constant | 1 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| constant | 4 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| constant | 8 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| constant | 12 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| decay | 1 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| decay | 4 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| decay | 8 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| decay | 12 | 50 | stage2_tail_max_over_g2_0 | 0.02365 | 0.02917 | 0.03044 | 0.03172 | 0.03711 | 0.04461 | 0.07009 |
| constant | 1 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| constant | 4 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| constant | 8 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| constant | 12 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| decay | 1 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| decay | 4 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| decay | 8 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| decay | 12 | 60 | stage2_tail_max_over_g2_0 | 0.02927 | 0.02982 | 0.03146 | 0.03259 | 0.03616 | 0.04017 | 0.04382 |
| constant | 1 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| constant | 4 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| constant | 8 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| constant | 12 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| decay | 1 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| decay | 4 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| decay | 8 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| decay | 12 | 50 | stage2_tail_mean_over_g2_0 | 0.00504 | 0.005525 | 0.006513 | 0.007358 | 0.008055 | 0.009557 | 0.01262 |
| constant | 1 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| constant | 4 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| constant | 8 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| constant | 12 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| decay | 1 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| decay | 4 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| decay | 8 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
| decay | 12 | 60 | stage2_tail_mean_over_g2_0 | 0.007102 | 0.008241 | 0.008588 | 0.009106 | 0.009708 | 0.01073 | 0.01093 |
