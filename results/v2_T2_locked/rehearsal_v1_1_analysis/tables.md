### verdict

| root | seeds | rule | per_q | overall | n_agreement_rows | n_all_agree |
|---|---|---|---|---|---|---|
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | [10501, 10510] | each q >= 18 of 20 runs pass; both q | {"50": false, "60": false} | not applicable (rule needs 20 seeds per q) | 20 | 20 |

### pass counts

| q | n_expected | n_completed | n_missing | n_failed_exception | n_incomplete | n_global_rng_violation | n_pass | pass_rate | cp95_lo | cp95_hi | rule | q_passes_rule | n_G-A_pass | n_G-F_pass | n_G-N_pass | n_v1_0_run_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10 | 10 | 0 | 0 | 0 | 0 | 10 | 1 | 0.6915 | 1 | not applicable (n = 10 != 20) |  | 10 | 10 | 10 | 9 |
| 60 | 10 | 10 | 0 | 0 | 0 | 0 | 10 | 1 | 0.6915 | 1 | not applicable (n = 10 != 20) |  | 10 | 10 | 10 | 8 |

### per-run verdicts

| q | seed | status | exit_code | status_info | crashed_attempts | eta_final | eta_dev | eta_dev_minus_final | rmse | tail | gmax_final | gmax_dev | gmax_dev_minus_final | s1 | stage1_rel_err_signed | G-A_pass | G-F_pass | G-N_pass | S1_pass | eta_N_pass | gmax_N_pass | global_rng | run_pass | v1_0_G-F | v1_0_run_pass | wall_sec | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | completed | 0 |  |  | 0.004014 | 0.004014 | 0 | 0.03966 | 0.007471 | 0.004014 | 0.004014 | 0 | 0.009488 | -0.009488 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 322.3 | pass |
| 50 | 10502 | completed | 0 |  |  | 0.001171 | 0.0009844 | -0.0001861 | 0.01974 | 0.01187 | 0.001171 | 0.0009844 | -0.0001861 | 0.0267 | 0.0267 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 393.5 | pass |
| 50 | 10503 | completed | 0 |  |  | 0.001721 | 0.001639 | -8.178e-05 | 0.01997 | 0.006916 | 0.002122 | 0.002079 | -4.325e-05 | 0.08738 | -0.08738 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 326.5 | pass |
| 50 | 10504 | completed | 0 |  |  | 0.001327 | 0.001159 | -0.000168 | 0.01909 | 0.0086 | 0.001327 | 0.001159 | -0.000168 | 0.01427 | 0.01427 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 321.1 | pass |
| 50 | 10505 | completed | 0 |  |  | 0.0009605 | 0.0009195 | -4.101e-05 | 0.02018 | 0.008316 | 0.0009605 | 0.0009195 | -4.101e-05 | 0.03294 | -0.03294 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 326.6 | pass |
| 50 | 10506 | completed | 0 |  |  | 0.002643 | 0.002643 | -1.11e-16 | 0.02399 | 0.007748 | 0.002643 | 0.002643 | -1.11e-16 | 0.04206 | -0.04206 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 322.8 | pass |
| 50 | 10507 | completed | 0 |  |  | 0.001063 | 0.001026 | -3.677e-05 | 0.0266 | 0.009862 | 0.002131 | 0.002114 | -1.708e-05 | 0.1027 | -0.1027 | yes | yes | yes | no | yes | yes | ok | yes | no | no | 321 | pass |
| 50 | 10508 | completed | 0 |  |  | 0.001141 | 0.001002 | -0.0001389 | 0.02341 | 0.004951 | 0.001141 | 0.001002 | -0.0001389 | 0.03777 | 0.03777 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 321.8 | pass |
| 50 | 10509 | completed | 0 |  |  | 0.001222 | 0.001138 | -8.34e-05 | 0.01742 | 0.006381 | 0.001222 | 0.001138 | -8.34e-05 | 0.01567 | 0.01567 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 320 | pass |
| 50 | 10510 | completed | 0 |  |  | 0.0009461 | 0.0007256 | -0.0002205 | 0.01502 | 0.009171 | 0.0009461 | 0.0007256 | -0.0002205 | 0.04482 | 0.04482 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 321.1 | pass |
| 60 | 10501 | completed | 0 |  |  | 0.0004121 | 0.0003602 | -5.196e-05 | 0.01333 | 0.008325 | 0.0004121 | 0.0003602 | -5.196e-05 | 0.01206 | 0.01206 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 320.7 | pass |
| 60 | 10502 | completed | 0 |  |  | 0.0005909 | 0.0005909 | -1.11e-16 | 0.0182 | 0.01079 | 0.000756 | 0.000762 | 5.976e-06 | 0.06947 | 0.06947 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 319.1 | pass |
| 60 | 10503 | completed | 0 |  |  | 0.001021 | 0.0008517 | -0.0001698 | 0.02393 | 0.009227 | 0.001093 | 0.001092 | -5.6e-07 | 0.07487 | -0.07487 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 316.6 | pass |
| 60 | 10504 | completed | 0 |  |  | 0.0009294 | 0.0009172 | -1.215e-05 | 0.02267 | 0.01028 | 0.0009294 | 0.0009172 | -1.215e-05 | 0.007296 | 0.007296 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 323.6 | pass |
| 60 | 10505 | completed | 0 |  |  | 0.00118 | 0.00098 | -0.0002002 | 0.01628 | 0.008791 | 0.002256 | 0.002277 | 2.107e-05 | 0.126 | -0.126 | yes | yes | yes | no | yes | yes | ok | yes | no | no | 314.8 | pass |
| 60 | 10506 | completed | 0 |  |  | 0.0004016 | 0.0003829 | -1.873e-05 | 0.01728 | 0.01254 | 0.0004896 | 0.0005092 | 1.958e-05 | 0.04916 | 0.04916 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 316.3 | pass |
| 60 | 10507 | completed | 0 |  |  | 0.0005143 | 0.000434 | -8.031e-05 | 0.01729 | 0.009559 | 0.0005143 | 0.000434 | -8.031e-05 | 0.01354 | -0.01354 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 316.3 | pass |
| 60 | 10508 | completed | 0 |  |  | 0.001597 | 0.001497 | -0.0001 | 0.02454 | 0.01046 | 0.001597 | 0.001497 | -0.0001 | 0.01648 | -0.01648 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 318.1 | pass |
| 60 | 10509 | completed | 0 |  |  | 0.0007645 | 0.0007645 | 0 | 0.02145 | 0.008686 | 0.0007645 | 0.0007645 | 0 | 0.04113 | -0.04113 | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | 312.8 | pass |
| 60 | 10510 | completed | 0 |  |  | 0.001453 | 0.001453 | -1.11e-16 | 0.02651 | 0.008632 | 0.002115 | 0.002157 | 4.187e-05 | 0.113 | 0.113 | yes | yes | yes | no | yes | yes | ok | yes | no | no | 314.5 | pass |

### agreement with gates.json

| q | seed | absdiff_eta_final | absdiff_eta_dev | absdiff_rmse | absdiff_tail | absdiff_gmax_final | absdiff_gmax_dev | absdiff_s1 | G-A | G-F | G-N | S1 | v1_0_G-F | v1_0_run_pass | run_pass | outcome | all_agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10502 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10503 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10504 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10505 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10506 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10507 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10508 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10509 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10510 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10501 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10502 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10503 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10504 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10505 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10506 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10507 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10508 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10509 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10510 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes |

### S1

| q | n | S1_pass | cp95_lo | cp95_hi | mean_signed | boot95_lo | boot95_hi | median_signed | sd_signed | bootstrap_resamples | bootstrap_seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10 | 9 | 0.555 | 0.9975 | -0.01354 | -0.04535 | 0.01545 | 0.00239 | 0.05153 | 10000 | 20261002 |
| 60 | 10 | 8 | 0.4439 | 0.9748 | -0.002107 | -0.04395 | 0.03882 | -0.003121 | 0.06976 | 10000 | 20261002 |

### distributions

| root | q | metric | n | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 50 | eta_final | 10 | 0.0009461 | 0.000959 | 0.001083 | 0.001196 | 0.001623 | 0.00278 | 0.004014 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 50 | eta_dev | 10 | 0.0007256 | 0.0009001 | 0.0009888 | 0.001082 | 0.001519 | 0.00278 | 0.004014 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 50 | eta_dev_minus_final | 10 | -0.0002205 | -0.0001896 | -0.0001607 | -8.259e-05 | -3.783e-05 | -9.992e-17 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 50 | rmse | 10 | 0.01502 | 0.01718 | 0.01925 | 0.02007 | 0.02384 | 0.0279 | 0.03966 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 50 | tail | 10 | 0.004951 | 0.006238 | 0.007055 | 0.008032 | 0.009028 | 0.01006 | 0.01187 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 50 | gmax_final | 10 | 0.0009461 | 0.000959 | 0.001148 | 0.001274 | 0.002129 | 0.00278 | 0.004014 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 50 | gmax_dev | 10 | 0.0007256 | 0.0009001 | 0.0009888 | 0.001149 | 0.002105 | 0.00278 | 0.004014 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 50 | gmax_dev_minus_final | 10 | -0.0002205 | -0.0001896 | -0.0001607 | -6.333e-05 | -2.306e-05 | -9.992e-17 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 50 | s1 | 10 | 0.009488 | 0.01379 | 0.01843 | 0.03536 | 0.04413 | 0.08892 | 0.1027 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 50 | stage1_rel_err_signed | 10 | -0.1027 | -0.08892 | -0.03978 | 0.00239 | 0.02394 | 0.03848 | 0.04482 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 60 | eta_final | 10 | 0.0004016 | 0.0004111 | 0.0005334 | 0.0008469 | 0.001141 | 0.001468 | 0.001597 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 60 | eta_dev | 10 | 0.0003602 | 0.0003806 | 0.0004732 | 0.0008081 | 0.0009643 | 0.001458 | 0.001497 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 60 | eta_dev_minus_final | 10 | -0.0002002 | -0.0001728 | -9.509e-05 | -3.534e-05 | -3.037e-06 | -9.992e-17 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 60 | rmse | 10 | 0.01333 | 0.01598 | 0.01729 | 0.01982 | 0.02362 | 0.02474 | 0.02651 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 60 | tail | 10 | 0.008325 | 0.008601 | 0.008712 | 0.009393 | 0.01041 | 0.01097 | 0.01254 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 60 | gmax_final | 10 | 0.0004121 | 0.0004818 | 0.0005747 | 0.0008469 | 0.001471 | 0.002129 | 0.002256 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 60 | gmax_dev | 10 | 0.0003602 | 0.0004266 | 0.0005724 | 0.0008409 | 0.001396 | 0.002169 | 0.002277 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 60 | gmax_dev_minus_final | 10 | -0.0001 | -8.228e-05 | -4.201e-05 | -2.8e-07 | 1.618e-05 | 2.315e-05 | 4.187e-05 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 60 | s1 | 10 | 0.007296 | 0.01159 | 0.01427 | 0.04515 | 0.07352 | 0.1143 | 0.126 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1 | 60 | stage1_rel_err_signed | 10 | -0.126 | -0.07999 | -0.03497 | -0.003121 | 0.03989 | 0.07382 | 0.113 |

### reported metrics

| q | seed | A_stage2_peak_rel_err_signed | A_stage2_peak_locfree_rel_err | A_stage2_peak_locfree_argmax_d | A_stage2_sym_err_max | A_stage2_tail_max | A_DeltaT_over_dw_on_max | A_DeltaT_over_dw_off_max | A_sigma_effort_at_0_t2 | A_smoothed_share_peak_gap_d0 | B_e1_at_0 | B_stage1_rel_err_signed | B_EXP_root_over_dw | B_dReach_over_dw | B_Deltamax_all_over_dw | B_dFull_over_dw | B_Gmax_full_t | B_Gmax_full_d | B_sigma_effort_at_0_t1 | dec_e_tilde | dec_band_lo | dec_band_hi | dec_learning_rel | dec_learning_rel_lo | dec_learning_rel_hi | dec_inherited_rel | dec_inherited_rel_lo | dec_inherited_rel_hi | dec_learning_contains_0 | dec_inherited_contains_0 | dec_band_contiguous | dec_e1_inside_sweep | drift_test_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | -0.1222 | -0.1221 | -0.5 | 3.276 | 4.28 | 0.004014 | 0.001155 | 3.446 | 0.318 | 46.22 | -0.009488 | 0.0009042 | 0.004092 | 0.004014 | 0.004092 | 2 | -4 | 3.623 | 47.01 | 47.01 | 47.19 | -0.01677 | -0.02063 | -0.01677 | 0.007286 | 0.007286 | 0.01114 | no | no | no | yes | yes |
| 50 | 10502 | -0.06803 | -0.06719 | -2 | 1.813 | 3.004 | 0.001171 | 0.000472 | 2.732 | 0.453 | 47.91 | 0.0267 | 0.0005191 | 0.001487 | 0.001171 | 0.001487 | 2 | -2 | 2.781 | 46.16 | 46.16 | 46.32 | 0.03763 | 0.0342 | 0.03763 | -0.01093 | -0.01093 | -0.0075 | no | no | yes | yes | yes |
| 50 | 10503 | -0.02357 | -0.02246 | -1.5 | 1.47 | 2.489 | 0.001721 | 0.0004409 | 2.818 | 1.348 | 42.59 | -0.08738 | 0.002122 | 0.003594 | 0.001873 | 0.003594 | 1 | 0 | 3.043 | 46.27 | 46.27 | 46.5 | -0.07881 | -0.08374 | -0.07881 | -0.008571 | -0.008571 | -0.003643 | no | no | no | yes | yes |
| 50 | 10504 | -0.07125 | -0.07111 | -0.5 | 2.359 | 2.402 | 0.001327 | 0.000328 | 2.772 | 0.4388 | 47.33 | 0.01427 | 0.0003792 | 0.001484 | 0.001327 | 0.001484 | 2 | -2 | 2.874 | 46.1 | 46.1 | 46.39 | 0.02648 | 0.02027 | 0.02648 | -0.01221 | -0.01221 | -0.006 | no | no | no | yes | yes |
| 50 | 10505 | -0.05432 | -0.05385 | -1.5 | 2.471 | 2.218 | 0.0009605 | 0.0002605 | 2.703 | 0.5612 | 45.13 | -0.03294 | 0.0003605 | 0.001135 | 0.0009605 | 0.001135 | 2 | -18 | 2.772 | 46.15 | 46.15 | 46.32 | -0.0218 | -0.02544 | -0.0218 | -0.01114 | -0.01114 | -0.0075 | no | no | yes | yes | yes |
| 50 | 10506 | -0.09843 | -0.09831 | 0.5 | 1.787 | 2.762 | 0.002643 | 0.0005444 | 3.145 | 0.3604 | 44.7 | -0.04206 | 0.0007696 | 0.002966 | 0.002643 | 0.002966 | 2 | -4 | 3.295 | 46.24 | 46.24 | 46.41 | -0.03285 | -0.03649 | -0.03285 | -0.009214 | -0.009214 | -0.005571 | no | no | yes | yes | yes |
| 50 | 10507 | -0.05776 | -0.05668 | -2 | 4.862 | 2.652 | 0.001063 | 0.0005013 | 2.748 | 0.5365 | 41.87 | -0.1027 | 0.002131 | 0.002905 | 0.001842 | 0.002905 | 1 | 0 | 2.794 | 45.66 | 45.66 | 45.82 | -0.08106 | -0.08449 | -0.08106 | -0.02164 | -0.02164 | -0.01821 | no | no | yes | yes | yes |
| 50 | 10508 | -0.06563 | -0.0655 | -0.5 | 1.266 | 2.257 | 0.001141 | 0.0003631 | 3.111 | 0.5345 | 48.43 | 0.03777 | 0.0007559 | 0.001539 | 0.001141 | 0.001539 | 2 | -2 | 3.335 | 46.51 | 46.51 | 46.79 | 0.0412 | 0.0352 | 0.0412 | -0.003429 | -0.003429 | 0.002571 | no | yes | no | yes | yes |
| 50 | 10509 | -0.03538 | -0.03526 | 0.5 | 2.859 | 2.689 | 0.001222 | 0.000385 | 2.994 | 0.9544 | 47.4 | 0.01567 | 0.0002716 | 0.001231 | 0.001222 | 0.001231 | 2 | -14 | 3.208 | 47.01 | 47.01 | 47.26 | 0.008385 | 0.003028 | 0.008385 | 0.007286 | 0.007286 | 0.01264 | no | no | no | yes | yes |
| 50 | 10510 | -0.05937 | -0.05937 | 0 | 2.171 | 2.517 | 0.0009461 | 0.0004498 | 2.711 | 0.515 | 48.76 | 0.04482 | 0.0005954 | 0.001403 | 0.0009461 | 0.001403 | 2 | -2 | 2.826 | 46.73 | 46.73 | 46.89 | 0.04353 | 0.04011 | 0.04353 | 0.001286 | 0.001286 | 0.004714 | no | no | yes | yes | yes |
| 60 | 10501 | -0.04693 | -0.04683 | 0.5 | 1.143 | 2.11 | 0.0004121 | 0.000314 | 3.078 | 0.6166 | 39.36 | 0.01206 | 0.0001084 | 0.000447 | 0.0004121 | 0.000447 | 2 | -2 | 3.02 | 38.76 | 38.75 | 38.83 | 0.01541 | 0.01361 | 0.01566 | -0.003343 | -0.0036 | -0.001543 | no | no | yes | yes | yes |
| 60 | 10502 | -0.03193 | -0.03149 | -1.5 | 1.693 | 2.26 | 0.0005909 | 0.0003609 | 2.841 | 0.8365 | 41.59 | 0.06947 | 0.000756 | 0.001241 | 0.0006498 | 0.001241 | 1 | 0 | 2.781 | 38.9 | 38.89 | 38.98 | 0.06922 | 0.06716 | 0.06947 | 0.0002571 | 0 | 0.002314 | no | yes | yes | yes | yes |
| 60 | 10503 | -0.07216 | -0.07202 | -1 | 3.357 | 2.134 | 0.001021 | 0.0002697 | 3.159 | 0.4115 | 35.98 | -0.07487 | 0.001093 | 0.001961 | 0.001021 | 0.001961 | 1 | 0 | 3.122 | 39.18 | 39.17 | 39.26 | -0.08233 | -0.08438 | -0.08207 | 0.007457 | 0.0072 | 0.009514 | no | no | yes | yes | yes |
| 60 | 10504 | -0.04855 | -0.04838 | -1 | 1.373 | 2.113 | 0.0009294 | 0.0003167 | 2.793 | 0.5406 | 39.17 | 0.007296 | 0.0002269 | 0.0009395 | 0.0009294 | 0.0009395 | 2 | -22 | 2.669 | 38.86 | 38.84 | 38.93 | 0.008067 | 0.006267 | 0.008582 | -0.0007714 | -0.001286 | 0.001029 | no | yes | yes | yes | yes |
| 60 | 10505 | -0.07591 | -0.07558 | 1.5 | 0.9295 | 2.507 | 0.00118 | 0.0003494 | 3.23 | 0.3999 | 33.99 | -0.126 | 0.002256 | 0.003335 | 0.002154 | 0.003335 | 1 | 0 | 3.107 | 38.84 | 38.82 | 38.92 | -0.1247 | -0.1268 | -0.1242 | -0.001286 | -0.0018 | 0.0007714 | no | yes | yes | yes | yes |
| 60 | 10506 | -0.04289 | -0.04223 | -1.5 | 1.593 | 2.057 | 0.0004016 | 0.0003012 | 2.738 | 0.5999 | 40.8 | 0.04916 | 0.0004896 | 0.0008076 | 0.000406 | 0.0008076 | 1 | 0 | 2.659 | 38.67 | 38.66 | 38.74 | 0.05482 | 0.05302 | 0.05507 | -0.005657 | -0.005914 | -0.003857 | no | no | yes | yes | yes |
| 60 | 10507 | -0.05151 | -0.05136 | 1 | 2.266 | 2.068 | 0.0005143 | 0.0002528 | 2.846 | 0.5193 | 38.36 | -0.01354 | 0.0001692 | 0.0005769 | 0.0005143 | 0.0005769 | 2 | -2 | 2.767 | 39.06 | 39.04 | 39.14 | -0.01791 | -0.01997 | -0.0174 | 0.004371 | 0.003857 | 0.006429 | no | no | yes | yes | yes |
| 60 | 10508 | -0.09174 | -0.09052 | -2.5 | 2.833 | 2.318 | 0.001597 | 0.0003812 | 3.242 | 0.3323 | 38.25 | -0.01648 | 0.0002676 | 0.001633 | 0.001597 | 0.001633 | 2 | -2 | 3.156 | 38.77 | 38.76 | 38.86 | -0.0134 | -0.01571 | -0.01314 | -0.003086 | -0.003343 | -0.0007714 | no | no | yes | yes | yes |
| 60 | 10509 | -0.05088 | -0.05048 | -1.5 | 1.952 | 1.879 | 0.0007645 | 0.000238 | 2.978 | 0.5502 | 37.29 | -0.04113 | 0.0003091 | 0.0009341 | 0.0007645 | 0.0009341 | 2 | -24 | 2.898 | 38.56 | 38.55 | 38.64 | -0.03265 | -0.0347 | -0.03239 | -0.008486 | -0.008743 | -0.006429 | no | no | yes | yes | yes |
| 60 | 10510 | -0.02097 | -0.02071 | -1 | 1.037 | 2.034 | 0.001453 | 0.0002849 | 2.999 | 1.344 | 43.28 | 0.113 | 0.002115 | 0.003258 | 0.001805 | 0.003258 | 1 | 0 | 2.975 | 38.84 | 38.83 | 38.92 | 0.1143 | 0.1122 | 0.1145 | -0.001286 | -0.001543 | 0.0007714 | no | yes | yes | yes | yes |

### EXP_root vs stage-1 error

| q | seed | stage1_rel_err_signed | stage1_rel_err_sq | EXP_root_over_dw |
|---|---|---|---|---|
| 50 | 10501 | -0.009488 | 9.003e-05 | 0.0009042 |
| 50 | 10502 | 0.0267 | 0.000713 | 0.0005191 |
| 50 | 10503 | -0.08738 | 0.007636 | 0.002122 |
| 50 | 10504 | 0.01427 | 0.0002036 | 0.0003792 |
| 50 | 10505 | -0.03294 | 0.001085 | 0.0003605 |
| 50 | 10506 | -0.04206 | 0.001769 | 0.0007696 |
| 50 | 10507 | -0.1027 | 0.01055 | 0.002131 |
| 50 | 10508 | 0.03777 | 0.001427 | 0.0007559 |
| 50 | 10509 | 0.01567 | 0.0002456 | 0.0002716 |
| 50 | 10510 | 0.04482 | 0.002009 | 0.0005954 |
| 60 | 10501 | 0.01206 | 0.0001455 | 0.0001084 |
| 60 | 10502 | 0.06947 | 0.004826 | 0.000756 |
| 60 | 10503 | -0.07487 | 0.005606 | 0.001093 |
| 60 | 10504 | 0.007296 | 5.323e-05 | 0.0002269 |
| 60 | 10505 | -0.126 | 0.01588 | 0.002256 |
| 60 | 10506 | 0.04916 | 0.002417 | 0.0004896 |
| 60 | 10507 | -0.01354 | 0.0001833 | 0.0001692 |
| 60 | 10508 | -0.01648 | 0.0002717 | 0.0002676 |
| 60 | 10509 | -0.04113 | 0.001692 | 0.0003091 |
| 60 | 10510 | 0.113 | 0.01277 | 0.002115 |

### EXP_root vs err^2 fit

| q | n | intercept | slope_on_err_sq | r2 | model |
|---|---|---|---|---|---|
| 50 | 10 | 0.0004166 | 0.1805 | 0.8791 | EXP_root/DW = a + b * (stage-1 rel. err)^2, ordinary least squares |
| 60 | 10 | 0.0001627 | 0.1406 | 0.9837 | EXP_root/DW = a + b * (stage-1 rel. err)^2, ordinary least squares |
