### calibration

Source: `results/v2_T2_locked/calibration/calibration_locked.csv`

| q | policy | tier | Gmax_full_over_dw | Gmax_full_t | Gmax_full_d | G_max_t1_over_dw | eta_T_over_dw | EXP_root_over_dw | dReach_over_dw | Deltamax_all_over_dw | dFull_over_dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | analytic_eq | final | 2.22e-16 | 2 | -24 | 0 | 2.22e-16 | 0 | 2.22e-16 | 2.22e-16 | 2.22e-16 |
| 50 | analytic_eq | development | 2.22e-16 | 2 | -16 | 0 | 2.22e-16 | 0 | 2.22e-16 | 2.22e-16 | 2.22e-16 |
| 50 | zero | final | 0.2593 | 2 | -26 | 0.2428 | 0.2593 | 0.2428 | 0.3938 | 0.2593 | 0.3938 |
| 50 | zero | development | 0.259 | 2 | -24 | 0.2428 | 0.259 | 0.2428 | 0.3934 | 0.259 | 0.3934 |
| 60 | analytic_eq | final | 3.511e-07 | 1 | 0 | 3.511e-07 | 2.22e-16 | 3.511e-07 | 3.511e-07 | 3.511e-07 | 3.511e-07 |
| 60 | analytic_eq | development | 2.22e-16 | 2 | 40 | 0 | 2.22e-16 | 0 | 2.22e-16 | 2.22e-16 | 2.22e-16 |
| 60 | zero | final | 0.1955 | 2 | -24 | 0.1911 | 0.1955 | 0.1911 | 0.2951 | 0.1955 | 0.2951 |
| 60 | zero | development | 0.1955 | 2 | -24 | 0.191 | 0.1955 | 0.191 | 0.2951 | 0.1955 | 0.2951 |

### calibration vs Phase 1

Source: `results/v2_T2_locked/calibration/calibration_locked.csv`

| q | policy | tier | max_abs_diff_vs_phase1 | diff_vs_phase1_Gmax_full_over_dw | diff_vs_phase1_EXP_root_over_dw | diff_vs_phase1_dReach_over_dw |
|---|---|---|---|---|---|---|
| 50 | analytic_eq | final | 0 | 0 | 0 | 0 |
| 50 | analytic_eq | development | 0 | 0 | 0 | 0 |
| 50 | zero | final | 5.551e-17 | 5.551e-17 | 0 | 0 |
| 50 | zero | development | 2.776e-17 | 0 | 2.776e-17 | 0 |
| 60 | analytic_eq | final | 5.294e-23 | -5.294e-23 | -5.294e-23 | -5.294e-23 |
| 60 | analytic_eq | development | 0 | 0 | 0 | 0 |
| 60 | zero | final | 8.327e-17 | 0 | 8.327e-17 | 5.551e-17 |
| 60 | zero | development | 5.551e-17 | 0 | 2.776e-17 | 5.551e-17 |

### calibration argmax vs Phase 1

Source: `results/v2_T2_locked/calibration/calibration_locked.csv`

| q | policy | tier | Gmax_full_t | phase1_Gmax_full_t | Gmax_full_d | phase1_Gmax_full_d |
|---|---|---|---|---|---|---|
| 50 | analytic_eq | final | 2 | 2 | -24 | -24 |
| 50 | analytic_eq | development | 2 | 2 | -16 | -16 |
| 50 | zero | final | 2 | 2 | -26 | -26 |
| 50 | zero | development | 2 | 2 | -24 | -24 |
| 60 | analytic_eq | final | 1 | 1 | 0 | 0 |
| 60 | analytic_eq | development | 2 | 2 | 40 | 40 |
| 60 | zero | final | 2 | 2 | -24 | -24 |
| 60 | zero | development | 2 | 2 | -24 | -24 |

### zero policy vs PI reference

Source: `results/v2_T2_locked/calibration/calibration_locked.csv`

| q | tier | Gmax_full_over_dw | pi_ref_Gmax | diff_vs_pi_Gmax | Gmax_full_t | pi_ref_t | Gmax_full_d | pi_ref_d | EXP_root_over_dw | pi_ref_root_gain | diff_vs_pi_root_gain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | final | 0.2593 | 0.2593 | -4.118e-05 | 2 | 2 | -26 | -26 | 0.2428 | 0.243 | -0.0001909 |
| 50 | development | 0.259 | 0.2593 | -0.0003353 | 2 | 2 | -24 | -26 | 0.2428 | 0.243 | -0.0002471 |
| 60 | final | 0.1955 | 0.1955 | 1.402e-05 | 2 | 2 | -24 | -23.5 | 0.1911 | 0.191 | 6.098e-05 |
| 60 | development | 0.1955 | 0.1955 | 1.402e-05 | 2 | 2 | -24 | -23.5 | 0.191 | 0.191 | 2.011e-05 |

### check 1 counts

Source: `results/v2_T2_locked/rehearsal_analysis/check1_phaseA_vs_stitched.csv`

| field | identical (of 20) |
|---|---|
| actor_identical | 20 |
| critic_identical | 20 |
| opponent_identical | 20 |
| opt_actor_identical | 20 |
| opt_critic_identical | 20 |
| rng_minibatch_identical | 20 |
| rng_streams_identical | 20 |
| torch_generator_identical | 20 |
| torch_global_rng_identical | 0 |
| numpy_global_rng_identical | 0 |
| python_random_identical | 0 |
| exports_A_identical | 20 |
| stage2_metrics_u1600_identical | 20 |

### check 2

Source: `results/v2_T2_locked/rehearsal_analysis/check2_phaseB_vs_launcher.csv`

| field | q50 s10503 | q60 s10503 |
|---|---|---|
| q | 50 | 60 |
| seed | 10503 | 10503 |
| history_B_identical | yes | yes |
| stability_B_identical | yes | yes |
| verifier_calls_B_identical | yes | yes |
| n_exports_B | 24 | 24 |
| exports_B_identical | yes | yes |
| final_weights_identical | yes | yes |
| rng_positions_B_identical | yes | yes |
| end_state_actor_identical | yes | yes |
| end_state_critic_identical | yes | yes |
| end_state_opponent_identical | yes | yes |
| end_state_frozen_identical | yes | yes |
| end_state_opt_actor_identical | yes | yes |
| end_state_opt_critic_identical | yes | yes |
| end_state_rng_minibatch_identical | yes | yes |
| end_state_rng_identical | yes | yes |
| checkpoint_metrics_B_identical | yes | yes |

### per-run gates (final tier; dev tier in parentheses columns)

Source: `results/v2_T2_locked/rehearsal_analysis/gates_per_run.csv`

| q | seed | eta_T_over_dw_final | eta_T_over_dw_dev | stage2_rmse_pos_over_g2_0_final | stage2_tail_mean_over_g2_0_final | G-A | Gmax_full_over_dw_final | Gmax_full_over_dw_dev | stage1_rel_err_abs_final | G-F | run_pass | outcome | G-A_dev | G-F_dev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | 0.004014 | 0.004014 | 0.03966 | 0.007471 | yes | 0.004014 | 0.004014 | 0.009488 | yes | yes | pass | yes | yes |
| 50 | 10502 | 0.001171 | 0.0009844 | 0.01974 | 0.01187 | yes | 0.001171 | 0.0009844 | 0.0267 | yes | yes | pass | yes | yes |
| 50 | 10503 | 0.001721 | 0.001639 | 0.01997 | 0.006916 | yes | 0.002122 | 0.002079 | 0.08738 | yes | yes | pass | yes | yes |
| 50 | 10504 | 0.001327 | 0.001159 | 0.01909 | 0.0086 | yes | 0.001327 | 0.001159 | 0.01427 | yes | yes | pass | yes | yes |
| 50 | 10505 | 0.0009605 | 0.0009195 | 0.02018 | 0.008316 | yes | 0.0009605 | 0.0009195 | 0.03294 | yes | yes | pass | yes | yes |
| 50 | 10506 | 0.002643 | 0.002643 | 0.02399 | 0.007748 | yes | 0.002643 | 0.002643 | 0.04206 | yes | yes | pass | yes | yes |
| 50 | 10507 | 0.001063 | 0.001026 | 0.0266 | 0.009862 | yes | 0.002131 | 0.002114 | 0.1027 | no | no | fail_G-F | yes | no |
| 50 | 10508 | 0.001141 | 0.001002 | 0.02341 | 0.004951 | yes | 0.001141 | 0.001002 | 0.03777 | yes | yes | pass | yes | yes |
| 50 | 10509 | 0.001222 | 0.001138 | 0.01742 | 0.006381 | yes | 0.001222 | 0.001138 | 0.01567 | yes | yes | pass | yes | yes |
| 50 | 10510 | 0.0009461 | 0.0007256 | 0.01502 | 0.009171 | yes | 0.0009461 | 0.0007256 | 0.04482 | yes | yes | pass | yes | yes |
| 60 | 10501 | 0.0004121 | 0.0003602 | 0.01333 | 0.008325 | yes | 0.0004121 | 0.0003602 | 0.01206 | yes | yes | pass | yes | yes |
| 60 | 10502 | 0.0005909 | 0.0005909 | 0.0182 | 0.01079 | yes | 0.000756 | 0.000762 | 0.06947 | yes | yes | pass | yes | yes |
| 60 | 10503 | 0.001021 | 0.0008517 | 0.02393 | 0.009227 | yes | 0.001093 | 0.001092 | 0.07487 | yes | yes | pass | yes | yes |
| 60 | 10504 | 0.0009294 | 0.0009172 | 0.02267 | 0.01028 | yes | 0.0009294 | 0.0009172 | 0.007296 | yes | yes | pass | yes | yes |
| 60 | 10505 | 0.00118 | 0.00098 | 0.01628 | 0.008791 | yes | 0.002256 | 0.002277 | 0.126 | no | no | fail_G-F | yes | no |
| 60 | 10506 | 0.0004016 | 0.0003829 | 0.01728 | 0.01254 | yes | 0.0004896 | 0.0005092 | 0.04916 | yes | yes | pass | yes | yes |
| 60 | 10507 | 0.0005143 | 0.000434 | 0.01729 | 0.009559 | yes | 0.0005143 | 0.000434 | 0.01354 | yes | yes | pass | yes | yes |
| 60 | 10508 | 0.001597 | 0.001497 | 0.02454 | 0.01046 | yes | 0.001597 | 0.001497 | 0.01648 | yes | yes | pass | yes | yes |
| 60 | 10509 | 0.0007645 | 0.0007645 | 0.02145 | 0.008686 | yes | 0.0007645 | 0.0007645 | 0.04113 | yes | yes | pass | yes | yes |
| 60 | 10510 | 0.001453 | 0.001453 | 0.02651 | 0.008632 | yes | 0.002115 | 0.002157 | 0.113 | no | no | fail_G-F | yes | no |

### pass counts

Source: `results/v2_T2_locked/rehearsal_analysis/pass_counts.csv`

| q | n | G_A_pass | G_F_pass | run_pass | G_A_pass_dev | G_F_pass_dev |
|---|---|---|---|---|---|---|
| 50 | 10 | 10 | 9 | 9 | 10 | 9 |
| 60 | 10 | 10 | 8 | 8 | 10 | 8 |

### reported stage-2 metrics at the end of A (final tier)

Source: `results/v2_T2_locked/rehearsal_analysis/gates_per_run.csv`

| q | seed | A_stage2_peak_rel_err_signed | A_stage2_peak_locfree_rel_err | A_stage2_peak_locfree_argmax_d | A_stage2_sym_err_max | A_stage2_tail_max | A_DeltaT_over_dw_on_max | A_DeltaT_over_dw_off_max | A_sigma_effort_at_0_t2 | A_smoothed_share_peak_gap_d0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | -0.1222 | -0.1221 | -0.5 | 3.276 | 4.28 | 0.004014 | 0.001155 | 3.446 | 0.318 |
| 50 | 10502 | -0.06803 | -0.06719 | -2 | 1.813 | 3.004 | 0.001171 | 0.000472 | 2.732 | 0.453 |
| 50 | 10503 | -0.02357 | -0.02246 | -1.5 | 1.47 | 2.489 | 0.001721 | 0.0004409 | 2.818 | 1.348 |
| 50 | 10504 | -0.07125 | -0.07111 | -0.5 | 2.359 | 2.402 | 0.001327 | 0.000328 | 2.772 | 0.4388 |
| 50 | 10505 | -0.05432 | -0.05385 | -1.5 | 2.471 | 2.218 | 0.0009605 | 0.0002605 | 2.703 | 0.5612 |
| 50 | 10506 | -0.09843 | -0.09831 | 0.5 | 1.787 | 2.762 | 0.002643 | 0.0005444 | 3.145 | 0.3604 |
| 50 | 10507 | -0.05776 | -0.05668 | -2 | 4.862 | 2.652 | 0.001063 | 0.0005013 | 2.748 | 0.5365 |
| 50 | 10508 | -0.06563 | -0.0655 | -0.5 | 1.266 | 2.257 | 0.001141 | 0.0003631 | 3.111 | 0.5345 |
| 50 | 10509 | -0.03538 | -0.03526 | 0.5 | 2.859 | 2.689 | 0.001222 | 0.000385 | 2.994 | 0.9544 |
| 50 | 10510 | -0.05937 | -0.05937 | 0 | 2.171 | 2.517 | 0.0009461 | 0.0004498 | 2.711 | 0.515 |
| 60 | 10501 | -0.04693 | -0.04683 | 0.5 | 1.143 | 2.11 | 0.0004121 | 0.000314 | 3.078 | 0.6166 |
| 60 | 10502 | -0.03193 | -0.03149 | -1.5 | 1.693 | 2.26 | 0.0005909 | 0.0003609 | 2.841 | 0.8365 |
| 60 | 10503 | -0.07216 | -0.07202 | -1 | 3.357 | 2.134 | 0.001021 | 0.0002697 | 3.159 | 0.4115 |
| 60 | 10504 | -0.04855 | -0.04838 | -1 | 1.373 | 2.113 | 0.0009294 | 0.0003167 | 2.793 | 0.5406 |
| 60 | 10505 | -0.07591 | -0.07558 | 1.5 | 0.9295 | 2.507 | 0.00118 | 0.0003494 | 3.23 | 0.3999 |
| 60 | 10506 | -0.04289 | -0.04223 | -1.5 | 1.593 | 2.057 | 0.0004016 | 0.0003012 | 2.738 | 0.5999 |
| 60 | 10507 | -0.05151 | -0.05136 | 1 | 2.266 | 2.068 | 0.0005143 | 0.0002528 | 2.846 | 0.5193 |
| 60 | 10508 | -0.09174 | -0.09052 | -2.5 | 2.833 | 2.318 | 0.001597 | 0.0003812 | 3.242 | 0.3323 |
| 60 | 10509 | -0.05088 | -0.05048 | -1.5 | 1.952 | 1.879 | 0.0007645 | 0.000238 | 2.978 | 0.5502 |
| 60 | 10510 | -0.02097 | -0.02071 | -1 | 1.037 | 2.034 | 0.001453 | 0.0002849 | 2.999 | 1.344 |

### reported stage-1 / full-policy metrics at the end of B (final tier)

Source: `results/v2_T2_locked/rehearsal_analysis/gates_per_run.csv`

| q | seed | B_e1_at_0 | B_stage1_rel_err_signed | dec_learning_rel | dec_learning_rel_lo | dec_learning_rel_hi | dec_inherited_rel | dec_inherited_rel_lo | dec_inherited_rel_hi | B_EXP_root_over_dw | B_dReach_over_dw | B_Deltamax_all_over_dw | B_dFull_over_dw | B_Gmax_full_t | B_Gmax_full_d | B_sigma_effort_at_0_t1 | dec_band_contiguous | dec_e1_inside_sweep | drift_test_pass | wall_sec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | 46.22 | -0.009488 | -0.01677 | -0.02063 | -0.01677 | 0.007286 | 0.007286 | 0.01114 | 0.0009042 | 0.004092 | 0.004014 | 0.004092 | 2 | -4 | 3.623 | no | yes | yes | 343.4 |
| 50 | 10502 | 47.91 | 0.0267 | 0.03763 | 0.0342 | 0.03763 | -0.01093 | -0.01093 | -0.0075 | 0.0005191 | 0.001487 | 0.001171 | 0.001487 | 2 | -2 | 2.781 | yes | yes | yes | 340 |
| 50 | 10503 | 42.59 | -0.08738 | -0.07881 | -0.08374 | -0.07881 | -0.008571 | -0.008571 | -0.003643 | 0.002122 | 0.003594 | 0.001873 | 0.003594 | 1 | 0 | 3.043 | no | yes | yes | 341.7 |
| 50 | 10504 | 47.33 | 0.01427 | 0.02648 | 0.02027 | 0.02648 | -0.01221 | -0.01221 | -0.006 | 0.0003792 | 0.001484 | 0.001327 | 0.001484 | 2 | -2 | 2.874 | no | yes | yes | 341.7 |
| 50 | 10505 | 45.13 | -0.03294 | -0.0218 | -0.02544 | -0.0218 | -0.01114 | -0.01114 | -0.0075 | 0.0003605 | 0.001135 | 0.0009605 | 0.001135 | 2 | -18 | 2.772 | yes | yes | yes | 335.9 |
| 50 | 10506 | 44.7 | -0.04206 | -0.03285 | -0.03649 | -0.03285 | -0.009214 | -0.009214 | -0.005571 | 0.0007696 | 0.002966 | 0.002643 | 0.002966 | 2 | -4 | 3.295 | yes | yes | yes | 337.6 |
| 50 | 10507 | 41.87 | -0.1027 | -0.08106 | -0.08449 | -0.08106 | -0.02164 | -0.02164 | -0.01821 | 0.002131 | 0.002905 | 0.001842 | 0.002905 | 1 | 0 | 2.794 | yes | yes | yes | 340.2 |
| 50 | 10508 | 48.43 | 0.03777 | 0.0412 | 0.0352 | 0.0412 | -0.003429 | -0.003429 | 0.002571 | 0.0007559 | 0.001539 | 0.001141 | 0.001539 | 2 | -2 | 3.335 | no | yes | yes | 341.4 |
| 50 | 10509 | 47.4 | 0.01567 | 0.008385 | 0.003028 | 0.008385 | 0.007286 | 0.007286 | 0.01264 | 0.0002716 | 0.001231 | 0.001222 | 0.001231 | 2 | -14 | 3.208 | no | yes | yes | 338.6 |
| 50 | 10510 | 48.76 | 0.04482 | 0.04353 | 0.04011 | 0.04353 | 0.001286 | 0.001286 | 0.004714 | 0.0005954 | 0.001403 | 0.0009461 | 0.001403 | 2 | -2 | 2.826 | yes | yes | yes | 338.6 |
| 60 | 10501 | 39.36 | 0.01206 | 0.01541 | 0.01361 | 0.01566 | -0.003343 | -0.0036 | -0.001543 | 0.0001084 | 0.000447 | 0.0004121 | 0.000447 | 2 | -2 | 3.02 | yes | yes | yes | 350 |
| 60 | 10502 | 41.59 | 0.06947 | 0.06922 | 0.06716 | 0.06947 | 0.0002571 | 0 | 0.002314 | 0.000756 | 0.001241 | 0.0006498 | 0.001241 | 1 | 0 | 2.781 | yes | yes | yes | 329.1 |
| 60 | 10503 | 35.98 | -0.07487 | -0.08233 | -0.08438 | -0.08207 | 0.007457 | 0.0072 | 0.009514 | 0.001093 | 0.001961 | 0.001021 | 0.001961 | 1 | 0 | 3.122 | yes | yes | yes | 329.4 |
| 60 | 10504 | 39.17 | 0.007296 | 0.008067 | 0.006267 | 0.008582 | -0.0007714 | -0.001286 | 0.001029 | 0.0002269 | 0.0009395 | 0.0009294 | 0.0009395 | 2 | -22 | 2.669 | yes | yes | yes | 332.3 |
| 60 | 10505 | 33.99 | -0.126 | -0.1247 | -0.1268 | -0.1242 | -0.001286 | -0.0018 | 0.0007714 | 0.002256 | 0.003335 | 0.002154 | 0.003335 | 1 | 0 | 3.107 | yes | yes | yes | 330.9 |
| 60 | 10506 | 40.8 | 0.04916 | 0.05482 | 0.05302 | 0.05507 | -0.005657 | -0.005914 | -0.003857 | 0.0004896 | 0.0008076 | 0.000406 | 0.0008076 | 1 | 0 | 2.659 | yes | yes | yes | 329.9 |
| 60 | 10507 | 38.36 | -0.01354 | -0.01791 | -0.01997 | -0.0174 | 0.004371 | 0.003857 | 0.006429 | 0.0001692 | 0.0005769 | 0.0005143 | 0.0005769 | 2 | -2 | 2.767 | yes | yes | yes | 331.8 |
| 60 | 10508 | 38.25 | -0.01648 | -0.0134 | -0.01571 | -0.01314 | -0.003086 | -0.003343 | -0.0007714 | 0.0002676 | 0.001633 | 0.001597 | 0.001633 | 2 | -2 | 3.156 | yes | yes | yes | 329.8 |
| 60 | 10509 | 37.29 | -0.04113 | -0.03265 | -0.0347 | -0.03239 | -0.008486 | -0.008743 | -0.006429 | 0.0003091 | 0.0009341 | 0.0007645 | 0.0009341 | 2 | -24 | 2.898 | yes | yes | yes | 334.9 |
| 60 | 10510 | 43.28 | 0.113 | 0.1143 | 0.1122 | 0.1145 | -0.001286 | -0.001543 | 0.0007714 | 0.002115 | 0.003258 | 0.001805 | 0.003258 | 1 | 0 | 2.975 | yes | yes | yes | 334.4 |

### gate-metric distributions (final tier)

Source: `results/v2_T2_locked/rehearsal_analysis/gate_distributions.csv`

| q | metric | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|
| 50 | Gmax_full_over_dw_final | 0.0009461 | 0.000959 | 0.001148 | 0.001274 | 0.002129 | 0.00278 | 0.004014 |
| 60 | Gmax_full_over_dw_final | 0.0004121 | 0.0004818 | 0.0005747 | 0.0008469 | 0.001471 | 0.002129 | 0.002256 |
| 50 | eta_T_over_dw_final | 0.0009461 | 0.000959 | 0.001083 | 0.001196 | 0.001623 | 0.00278 | 0.004014 |
| 60 | eta_T_over_dw_final | 0.0004016 | 0.0004111 | 0.0005334 | 0.0008469 | 0.001141 | 0.001468 | 0.001597 |
| 50 | stage1_rel_err_abs_final | 0.009488 | 0.01379 | 0.01843 | 0.03536 | 0.04413 | 0.08892 | 0.1027 |
| 60 | stage1_rel_err_abs_final | 0.007296 | 0.01159 | 0.01427 | 0.04515 | 0.07352 | 0.1143 | 0.126 |
| 50 | stage2_rmse_pos_over_g2_0_final | 0.01502 | 0.01718 | 0.01925 | 0.02007 | 0.02384 | 0.0279 | 0.03966 |
| 60 | stage2_rmse_pos_over_g2_0_final | 0.01333 | 0.01598 | 0.01729 | 0.01982 | 0.02362 | 0.02474 | 0.02651 |
| 50 | stage2_tail_mean_over_g2_0_final | 0.004951 | 0.006238 | 0.007055 | 0.008032 | 0.009028 | 0.01006 | 0.01187 |
| 60 | stage2_tail_mean_over_g2_0_final | 0.008325 | 0.008601 | 0.008712 | 0.009393 | 0.01041 | 0.01097 | 0.01254 |

### gate-metric distributions (dev tier)

Source: `results/v2_T2_locked/rehearsal_analysis/gate_distributions.csv`

| q | metric | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|
| 50 | Gmax_full_over_dw_dev | 0.0007256 | 0.0009001 | 0.0009888 | 0.001149 | 0.002105 | 0.00278 | 0.004014 |
| 60 | Gmax_full_over_dw_dev | 0.0003602 | 0.0004266 | 0.0005724 | 0.0008409 | 0.001396 | 0.002169 | 0.002277 |
| 50 | eta_T_over_dw_dev | 0.0007256 | 0.0009001 | 0.0009888 | 0.001082 | 0.001519 | 0.00278 | 0.004014 |
| 60 | eta_T_over_dw_dev | 0.0003602 | 0.0003806 | 0.0004732 | 0.0008081 | 0.0009643 | 0.001458 | 0.001497 |
| 50 | stage1_rel_err_abs_dev | 0.009488 | 0.01379 | 0.01843 | 0.03536 | 0.04413 | 0.08892 | 0.1027 |
| 60 | stage1_rel_err_abs_dev | 0.007296 | 0.01159 | 0.01427 | 0.04515 | 0.07352 | 0.1143 | 0.126 |
| 50 | stage2_rmse_pos_over_g2_0_dev | 0.01502 | 0.01718 | 0.01925 | 0.02007 | 0.02384 | 0.0279 | 0.03966 |
| 60 | stage2_rmse_pos_over_g2_0_dev | 0.01333 | 0.01598 | 0.01729 | 0.01982 | 0.02362 | 0.02474 | 0.02651 |
| 50 | stage2_tail_mean_over_g2_0_dev | 0.004951 | 0.006238 | 0.007055 | 0.008032 | 0.009028 | 0.01006 | 0.01187 |
| 60 | stage2_tail_mean_over_g2_0_dev | 0.008325 | 0.008601 | 0.008712 | 0.009393 | 0.01041 | 0.01097 | 0.01254 |

### dev - final differences of the gate metrics

Source: `results/v2_T2_locked/rehearsal_analysis/gate_distributions.csv`

| q | metric | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|
| 50 | Gmax_full_over_dw_dev_minus_final | -0.0002205 | -0.0001896 | -0.0001607 | -6.333e-05 | -2.306e-05 | -9.992e-17 | 0 |
| 60 | Gmax_full_over_dw_dev_minus_final | -0.0001 | -8.228e-05 | -4.201e-05 | -2.8e-07 | 1.618e-05 | 2.315e-05 | 4.187e-05 |
| 50 | eta_T_over_dw_dev_minus_final | -0.0002205 | -0.0001896 | -0.0001607 | -8.259e-05 | -3.783e-05 | -9.992e-17 | 0 |
| 60 | eta_T_over_dw_dev_minus_final | -0.0002002 | -0.0001728 | -9.509e-05 | -3.534e-05 | -3.037e-06 | -9.992e-17 | 0 |
| 50 | stage1_rel_err_abs_dev_minus_final | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60 | stage1_rel_err_abs_dev_minus_final | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 50 | stage2_rmse_pos_over_g2_0_dev_minus_final | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60 | stage2_rmse_pos_over_g2_0_dev_minus_final | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 50 | stage2_tail_mean_over_g2_0_dev_minus_final | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60 | stage2_tail_mean_over_g2_0_dev_minus_final | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### dev - final differences of every reported metric

Source: `results/v2_T2_locked/rehearsal_analysis/gates_per_run.csv`

| q | metric | min | median | max | max_abs |
|---|---|---|---|---|---|
| 50 | end-of-A stage2_peak_rel_err_signed | 0 | 0 | 0 | 0 |
| 50 | end-of-A stage2_peak_rel_err_abs | 0 | 0 | 0 | 0 |
| 50 | end-of-A stage2_sym_err_max | 0 | 0 | 0 | 0 |
| 50 | end-of-A stage2_tail_max | 0 | 0 | 0 | 0 |
| 50 | end-of-A stage2_tail_max_over_g2_0 | 0 | 0 | 0 | 0 |
| 50 | end-of-A stage2_tail_mean | 0 | 0 | 0 | 0 |
| 50 | end-of-A DeltaT_over_dw_on_max | -0.0002205 | -8.259e-05 | 0 | 0.0002205 |
| 50 | end-of-A DeltaT_over_dw_off_max | 0 | 0 | 0 | 0 |
| 50 | end-of-A DeltaT_over_dw_on_mean_cellmass_weighted | -1.116e-06 | 3.808e-07 | 8.93e-07 | 1.116e-06 |
| 50 | end-of-A sigma_effort_at_0_t2 | 0 | 0 | 0 | 0 |
| 50 | end-of-A e2_at_0 | 0 | 0 | 0 | 0 |
| 50 | end-of-A g2_at_0 | 0 | 0 | 0 | 0 |
| 50 | end-of-A stage2_rmse_pos_over_g2_0 | 0 | 0 | 0 | 0 |
| 50 | end-of-A stage2_tail_mean_over_g2_0 | 0 | 0 | 0 | 0 |
| 50 | end-of-A eta_T_over_dw | -0.0002205 | -8.259e-05 | 0 | 0.0002205 |
| 50 | end-of-A stage2_peak_locfree_rel_err | 0 | 0 | 0 | 0 |
| 50 | end-of-B Gmax_full_over_dw | -0.0002205 | -6.333e-05 | 0 | 0.0002205 |
| 50 | end-of-B Gmax_full_t | 0 | 0 | 0 | 0 |
| 50 | end-of-B Gmax_full_d | -2 | 0 | 2 | 2 |
| 50 | end-of-B EXP_root_over_dw | -4.325e-05 | -9.37e-06 | 8.482e-06 | 4.325e-05 |
| 50 | end-of-B dReach_over_dw | -0.0002187 | -0.0001111 | 1.261e-05 | 0.0002187 |
| 50 | end-of-B Deltamax_all_over_dw | -0.0002205 | -6.552e-05 | 0 | 0.0002205 |
| 50 | end-of-B dFull_over_dw | -0.0002187 | -0.0001111 | 1.261e-05 | 0.0002187 |
| 50 | end-of-B stage1_rel_err_signed | 0 | 0 | 0 | 0 |
| 50 | end-of-B stage1_rel_err_abs | 0 | 0 | 0 | 0 |
| 50 | end-of-B e1_at_0 | 0 | 0 | 0 | 0 |
| 50 | end-of-B g1 | 0 | 0 | 0 | 0 |
| 50 | end-of-B sigma_effort_at_0_t1 | 0 | 0 | 0 | 0 |
| 50 | end-of-B eta_T_over_dw | -0.0002205 | -8.259e-05 | 0 | 0.0002205 |
| 60 | end-of-A stage2_peak_rel_err_signed | 0 | 0 | 0 | 0 |
| 60 | end-of-A stage2_peak_rel_err_abs | 0 | 0 | 0 | 0 |
| 60 | end-of-A stage2_sym_err_max | 0 | 0 | 0 | 0 |
| 60 | end-of-A stage2_tail_max | 0 | 0 | 0 | 0 |
| 60 | end-of-A stage2_tail_max_over_g2_0 | 0 | 0 | 0 | 0 |
| 60 | end-of-A stage2_tail_mean | 0 | 0 | 0 | 0 |
| 60 | end-of-A DeltaT_over_dw_on_max | -0.0002002 | -3.534e-05 | 0 | 0.0002002 |
| 60 | end-of-A DeltaT_over_dw_off_max | -2.553e-05 | 0 | 0 | 2.553e-05 |
| 60 | end-of-A DeltaT_over_dw_on_mean_cellmass_weighted | -6.909e-07 | -3.286e-07 | 6.401e-07 | 6.909e-07 |
| 60 | end-of-A sigma_effort_at_0_t2 | 0 | 0 | 0 | 0 |
| 60 | end-of-A e2_at_0 | 0 | 0 | 0 | 0 |
| 60 | end-of-A g2_at_0 | 0 | 0 | 0 | 0 |
| 60 | end-of-A stage2_rmse_pos_over_g2_0 | 0 | 0 | 0 | 0 |
| 60 | end-of-A stage2_tail_mean_over_g2_0 | 0 | 0 | 0 | 0 |
| 60 | end-of-A eta_T_over_dw | -0.0002002 | -3.534e-05 | 0 | 0.0002002 |
| 60 | end-of-A stage2_peak_locfree_rel_err | 0 | 0 | 0 | 0 |
| 60 | end-of-B Gmax_full_over_dw | -0.0001 | -2.8e-07 | 4.187e-05 | 0.0001 |
| 60 | end-of-B Gmax_full_t | 0 | 0 | 0 | 0 |
| 60 | end-of-B Gmax_full_d | -2 | 0 | 2 | 2 |
| 60 | end-of-B EXP_root_over_dw | -4.621e-06 | 1.535e-05 | 4.187e-05 | 4.187e-05 |
| 60 | end-of-B dReach_over_dw | -0.0001774 | -3.419e-05 | 4.06e-05 | 0.0001774 |
| 60 | end-of-B Deltamax_all_over_dw | -0.0001 | -6.073e-06 | 4.06e-05 | 0.0001 |
| 60 | end-of-B dFull_over_dw | -0.0001774 | -3.419e-05 | 4.06e-05 | 0.0001774 |
| 60 | end-of-B stage1_rel_err_signed | 0 | 0 | 0 | 0 |
| 60 | end-of-B stage1_rel_err_abs | 0 | 0 | 0 | 0 |
| 60 | end-of-B e1_at_0 | 0 | 0 | 0 | 0 |
| 60 | end-of-B g1 | 0 | 0 | 0 | 0 |
| 60 | end-of-B sigma_effort_at_0_t1 | 0 | 0 | 0 | 0 |
| 60 | end-of-B eta_T_over_dw | -0.0002002 | -3.534e-05 | 0 | 0.0002002 |

### cusp thresholds

Source: `results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv`

| q | quantity | n_reached | median | min | max |
|---|---|---|---|---|---|
| 50 | step_abs_peak_lt_0.05 | 5 | 8500 | 7500 | 9000 |
| 50 | rmse_at_peak_lt_0.05 | 5 | 0.01597 | 0.0153 | 0.01717 |
| 50 | locfree_at_peak_lt_0.05 | 5 | -0.03457 | -0.04402 | -0.03131 |
| 50 | step_abs_locfree_lt_0.05 | 5 | 8000 | 7500 | 8500 |
| 50 | step_abs_peak_lt_0.03 | 5 | 16000 | 8500 | 17000 |
| 50 | step_abs_locfree_lt_0.03 | 5 | 16000 | 8500 | 16500 |
| 50 | step_abs_peak_lt_0.01 | 5 | 19000 | 14500 | 55000 |
| 50 | step_abs_locfree_lt_0.01 | 5 | 19000 | 14500 | 55000 |
| 60 | step_abs_peak_lt_0.05 | 5 | 9500 | 8500 | 10500 |
| 60 | rmse_at_peak_lt_0.05 | 5 | 0.0154 | 0.009977 | 0.02182 |
| 60 | locfree_at_peak_lt_0.05 | 5 | -0.04126 | -0.04694 | -0.03473 |
| 60 | step_abs_locfree_lt_0.05 | 5 | 9500 | 8500 | 10500 |
| 60 | step_abs_peak_lt_0.03 | 5 | 12000 | 9500 | 16000 |
| 60 | step_abs_locfree_lt_0.03 | 5 | 12000 | 9500 | 16000 |
| 60 | step_abs_peak_lt_0.01 | 5 | 19500 | 18000 | 35000 |
| 60 | step_abs_locfree_lt_0.01 | 5 | 19500 | 18000 | 35000 |

### cusp thresholds per init

Source: `results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv`

| q | init_seed | step_abs_peak_lt_0.05 | rmse_at_peak_lt_0.05 | locfree_at_peak_lt_0.05 | step_abs_locfree_lt_0.05 | step_abs_peak_lt_0.03 | step_abs_locfree_lt_0.03 | step_abs_peak_lt_0.01 | step_abs_locfree_lt_0.01 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 0 | 8500 | 0.0153 | -0.03283 | 8500 | 16000 | 16000 | 16000 | 16000 |
| 50 | 1 | 8000 | 0.01627 | -0.03131 | 8000 | 8500 | 8500 | 21000 | 21000 |
| 50 | 2 | 7500 | 0.01597 | -0.03457 | 7500 | 17000 | 16500 | 55000 | 55000 |
| 50 | 3 | 8500 | 0.01537 | -0.03472 | 8500 | 10000 | 10000 | 14500 | 14500 |
| 50 | 4 | 9000 | 0.01717 | -0.04402 | 8000 | 16500 | 16500 | 19000 | 19000 |
| 60 | 0 | 9500 | 0.0154 | -0.04126 | 9500 | 12500 | 12500 | 18500 | 18500 |
| 60 | 1 | 10000 | 0.02182 | -0.04158 | 10000 | 16000 | 16000 | 35000 | 35000 |
| 60 | 2 | 8500 | 0.009977 | -0.03707 | 8500 | 9500 | 9500 | 19500 | 19500 |
| 60 | 3 | 9000 | 0.0175 | -0.04694 | 9000 | 12000 | 12000 | 21000 | 21000 |
| 60 | 4 | 10500 | 0.01139 | -0.03473 | 10500 | 11000 | 11000 | 18000 | 18000 |

### cusp identity vs Pilot 4

Source: `results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv`

| q | init_seed | n_common_steps | n_identical_float32 | identical_to_pilot4 |
|---|---|---|---|---|
| 50 | 0 | 300 | 300 | yes |
| 50 | 1 | 300 | 300 | yes |
| 50 | 2 | 300 | 300 | yes |
| 50 | 3 | 300 | 300 | yes |
| 50 | 4 | 300 | 300 | yes |
| 60 | 0 | 300 | 300 | yes |
| 60 | 1 | 300 | 300 | yes |
| 60 | 2 | 300 | 300 | yes |
| 60 | 3 | 300 | 300 | yes |
| 60 | 4 | 300 | 300 | yes |
