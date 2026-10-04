### verdict

| root | seeds | rule | per_q | overall | n_agreement_rows | n_all_agree | n_expected | n_completed | n_missing | n_failed_exception | n_incomplete | n_global_rng_violation | n_clean_tree | commits | rehearsal_safety | paired_root |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | [10501, 10510] | each q >= 18 of 20 runs pass; both q | {"50": false, "60": false} | not applicable (rule needs 20 seeds per q) | 20 | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 20 | ['f2d616cc921e2c6be46c48336d31372dbbf58481'] | {"pass": true, "pooled_n_G-S_pass": 20, "per_q_P": {"50": 0.9958733577727633, "60": 0.9995272279276824}} | {"root": "/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1", "label": "rehearsal_v1_1", "n_pairs": 20} |

### rehearsal safety condition (D4)

| q | n_runs | n_G-S_pass | mean_signed | sd_ddof1 | per_run_pass_prob | P_ge_k_of_n | meets_probability |
|---|---|---|---|---|---|---|---|
| 50 | 10 | 10 | -0.011811 | 0.0178499 | 0.983533 | 0.995873 | yes |
| 60 | 10 | 10 | 0.000906032 | 0.0187419 | 0.992294 | 0.999527 | yes |

### pass counts

| q | n_expected | n_completed | n_missing | n_failed_exception | n_incomplete | n_global_rng_violation | n_pass | pass_rate | cp95_lo | cp95_hi | rule | q_passes_rule | n_G-A_pass | n_G-F_pass | n_G-N_pass | n_G-S_pass | n_S1_0.10_pass | n_v1_1_run_pass | n_v1_0_run_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10 | 10 | 0 | 0 | 0 | 0 | 10 | 1 | 0.691503 | 1 | not applicable (n = 10 != 20) |  | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| 60 | 10 | 10 | 0 | 0 | 0 | 0 | 10 | 1 | 0.691503 | 1 | not applicable (n = 10 != 20) |  | 10 | 10 | 10 | 10 | 10 | 10 | 10 |

### per-run verdicts

| q | seed | status | exit_code | status_info | crashed_attempts | eta_final | eta_dev | eta_dev_minus_final | rmse | rmse_dev | rmse_dev_minus_final | tail | tail_dev | tail_dev_minus_final | gmax_final | gmax_dev | gmax_dev_minus_final | s1 | s1_dev | s1_dev_minus_final | stage1_rel_err_signed | e1_at_0 | G-A_pass | G-A_eta_pass | G-A_rmse_pass | G-A_tail_pass | G-F_pass | G-N_pass | G-N_eta_pass | G-N_gmax_pass | G-S_pass | S1_pass | global_rng | run_pass | v1_1_run_pass | v1_0_G-F | v1_0_run_pass | wall_sec | outcome | table_sha256 | table_build_sec | table_sha256_disk | clean_tree | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | completed | 0 |  |  | 0.00401444 | 0.00401444 | 0 | 0.0396621 | 0.0396621 | 0 | 0.00747055 | 0.00747055 | 0 | 0.00401444 | 0.00401444 | 0 | 0.0134169 | 0.0134169 | 0 | 0.0134169 | 47.2928 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 331.253 | pass | c72e4156e3ce08f25a071f61f285e55500fc4ee8b8258eb71f34462bbec40243 | 7.79056 | c72e4156e3ce08f25a071f61f285e55500fc4ee8b8258eb71f34462bbec40243 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10502 | completed | 0 |  |  | 0.00117057 | 0.000984432 | -0.000186138 | 0.0197389 | 0.0197389 | 0 | 0.01187 | 0.01187 | 0 | 0.00117057 | 0.000984432 | -0.000186138 | 0.0102045 | 0.0102045 | 0 | -0.0102045 | 46.1905 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 329.866 | pass | 9cfc1a75bceacf13bd1de865a8f93c613a65c96ae948ca85cc735f0c27d0cfd8 | 7.83428 | 9cfc1a75bceacf13bd1de865a8f93c613a65c96ae948ca85cc735f0c27d0cfd8 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10503 | completed | 0 |  |  | 0.00172113 | 0.00163935 | -8.178e-05 | 0.019966 | 0.019966 | 0 | 0.00691612 | 0.00691612 | 0 | 0.00172113 | 0.00163935 | -8.178e-05 | 0.0322856 | 0.0322856 | 0 | -0.0322856 | 45.16 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 332.358 | pass | fac688aca008fe72bcad328958ffa0e81b553d955669a06a3dbed718398d0076 | 7.76992 | fac688aca008fe72bcad328958ffa0e81b553d955669a06a3dbed718398d0076 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10504 | completed | 0 |  |  | 0.00132726 | 0.00115929 | -0.000167971 | 0.0190868 | 0.0190868 | 0 | 0.00860028 | 0.00860028 | 0 | 0.00132726 | 0.00115929 | -0.000167971 | 0.00920279 | 0.00920279 | 0 | 0.00920279 | 47.0961 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 331.474 | pass | 7caccc1db7054996995378326c70d23e8a5b9615fbac915d3944900a45a8a78d | 7.68434 | 7caccc1db7054996995378326c70d23e8a5b9615fbac915d3944900a45a8a78d | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10505 | completed | 0 |  |  | 0.000960484 | 0.000919478 | -4.10057e-05 | 0.0201792 | 0.0201792 | 0 | 0.00831625 | 0.00831625 | 0 | 0.000960484 | 0.000919478 | -4.10057e-05 | 0.0186761 | 0.0186761 | 0 | -0.0186761 | 45.7951 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 328.472 | pass | b60f95f614d0e7f4f279da3babc22e6bbe856f8fb43530b6c0b443fa729d84ea | 7.97087 | b60f95f614d0e7f4f279da3babc22e6bbe856f8fb43530b6c0b443fa729d84ea | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10506 | completed | 0 |  |  | 0.002643 | 0.002643 | -1.11022e-16 | 0.0239887 | 0.0239887 | 0 | 0.00774763 | 0.00774763 | 0 | 0.002643 | 0.002643 | -1.11022e-16 | 0.040692 | 0.040692 | 0 | -0.040692 | 44.7677 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 330.548 | pass | e0742ba707095b809704533c9fab1c32d3727f2a203d9a05ae84eb5db37fbabc | 7.96374 | e0742ba707095b809704533c9fab1c32d3727f2a203d9a05ae84eb5db37fbabc | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10507 | completed | 0 |  |  | 0.00106326 | 0.00102649 | -3.67692e-05 | 0.0265976 | 0.0265976 | 0 | 0.00986186 | 0.00986186 | 0 | 0.00106326 | 0.00102649 | -3.67692e-05 | 0.0246492 | 0.0246492 | 0 | -0.0246492 | 45.5164 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 328.511 | pass | a8422ec85f950f255c5b9963ffdbacc08cf9b1191381756fabf46bcb90d6c44a | 7.97388 | a8422ec85f950f255c5b9963ffdbacc08cf9b1191381756fabf46bcb90d6c44a | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10508 | completed | 0 |  |  | 0.00114095 | 0.00100209 | -0.000138854 | 0.0234103 | 0.0234103 | 0 | 0.00495108 | 0.00495108 | 0 | 0.00114095 | 0.00100209 | -0.000138854 | 0.00963182 | 0.00963182 | 0 | -0.00963182 | 46.2172 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 328.058 | pass | 8e06e912a5dccdc419d72be2f09e48691c4b67d3c819f7d7f90e72bdd0519308 | 7.85391 | 8e06e912a5dccdc419d72be2f09e48691c4b67d3c819f7d7f90e72bdd0519308 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10509 | completed | 0 |  |  | 0.00122168 | 0.00113828 | -8.33983e-05 | 0.0174234 | 0.0174234 | 0 | 0.00638139 | 0.00638139 | 0 | 0.00122168 | 0.00113828 | -8.33983e-05 | 0.0105367 | 0.0105367 | 0 | -0.0105367 | 46.175 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 328.959 | pass | 0504a66d02387aeae7c27c07e6e85843e66e4a3b4e5940b42782d6d139f948e2 | 7.81602 | 0504a66d02387aeae7c27c07e6e85843e66e4a3b4e5940b42782d6d139f948e2 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10510 | completed | 0 |  |  | 0.000946099 | 0.000725629 | -0.00022047 | 0.0150213 | 0.0150213 | 0 | 0.00917122 | 0.00917122 | 0 | 0.000946099 | 0.000725629 | -0.00022047 | 0.00594587 | 0.00594587 | 0 | 0.00594587 | 46.9441 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 332.287 | pass | ce17763bdc6023fe25c0344c3bf4ea6851f84949d8dcb3d6ad48c806b22a0a82 | 7.8393 | ce17763bdc6023fe25c0344c3bf4ea6851f84949d8dcb3d6ad48c806b22a0a82 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10501 | completed | 0 |  |  | 0.00041212 | 0.000360161 | -5.19596e-05 | 0.0133277 | 0.0133277 | 0 | 0.00832454 | 0.00832454 | 0 | 0.00041212 | 0.000360161 | -5.19596e-05 | 0.0314414 | 0.0314414 | 0 | -0.0314414 | 37.6662 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 324.356 | pass | 26b50d8b855489f38ab5ffdac4e4c263902743c90dbd0f334543e85b32847614 | 9.24697 | 26b50d8b855489f38ab5ffdac4e4c263902743c90dbd0f334543e85b32847614 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10502 | completed | 0 |  |  | 0.000590891 | 0.000590891 | -1.11022e-16 | 0.0182003 | 0.0182003 | 0 | 0.0107927 | 0.0107927 | 0 | 0.000590891 | 0.000590891 | -1.11022e-16 | 0.0280365 | 0.0280365 | 0 | 0.0280365 | 39.9792 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 325.906 | pass | 3a7cd63058871f2666a2efd00611cd198bb7a6468dd75f8689c3caae42af8169 | 9.28325 | 3a7cd63058871f2666a2efd00611cd198bb7a6468dd75f8689c3caae42af8169 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10503 | completed | 0 |  |  | 0.00102148 | 0.000851703 | -0.000169772 | 0.0239348 | 0.0239348 | 0 | 0.00922653 | 0.00922653 | 0 | 0.00102148 | 0.000851703 | -0.000169772 | 0.0219821 | 0.0219821 | 0 | 0.0219821 | 39.7437 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 320.779 | pass | 0b75162d3e77e9a068c7c5602c2877bb764a41f3b61c24f6ce231916af7ed10a | 9.43834 | 0b75162d3e77e9a068c7c5602c2877bb764a41f3b61c24f6ce231916af7ed10a | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10504 | completed | 0 |  |  | 0.000929381 | 0.000917235 | -1.21464e-05 | 0.0226681 | 0.0226681 | 0 | 0.0102783 | 0.0102783 | 0 | 0.000929381 | 0.000917235 | -1.21464e-05 | 0.00486167 | 0.00486167 | 0 | -0.00486167 | 38.6998 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 334.988 | pass | 6e955f7af5b1ce45f7e6f6f325959eecfce317b892c981d40e7218e8e133af54 | 8.77537 | 6e955f7af5b1ce45f7e6f6f325959eecfce317b892c981d40e7218e8e133af54 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10505 | completed | 0 |  |  | 0.00118019 | 0.00098001 | -0.000200184 | 0.0162783 | 0.0162783 | 0 | 0.00879114 | 0.00879114 | 0 | 0.00118019 | 0.00098001 | -0.000200184 | 0.0243908 | 0.0243908 | 0 | -0.0243908 | 37.9404 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 321.681 | pass | 99e824e56240103d6726eb7fa3270b53db0b32db5d466841daa3ed63426c76c1 | 9.40059 | 99e824e56240103d6726eb7fa3270b53db0b32db5d466841daa3ed63426c76c1 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10506 | completed | 0 |  |  | 0.000401645 | 0.000382915 | -1.87303e-05 | 0.0172828 | 0.0172828 | 0 | 0.0125433 | 0.0125433 | 0 | 0.000401645 | 0.000382915 | -1.87303e-05 | 0.0130148 | 0.0130148 | 0 | 0.0130148 | 39.395 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 323.724 | pass | d15f4eacb7b965eff473f5d0a3cd38dc2210dc56d072d32ec3efc40bb617655d | 9.30185 | d15f4eacb7b965eff473f5d0a3cd38dc2210dc56d072d32ec3efc40bb617655d | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10507 | completed | 0 |  |  | 0.000514279 | 0.000433965 | -8.03144e-05 | 0.0172917 | 0.0172917 | 0 | 0.00955929 | 0.00955929 | 0 | 0.000514279 | 0.000433965 | -8.03144e-05 | 0.00726242 | 0.00726242 | 0 | 0.00726242 | 39.1713 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 320.84 | pass | 73a63fd60a86a063724045653b486114fd7ae5f55ae6caa04e2bab8ea0552438 | 9.35181 | 73a63fd60a86a063724045653b486114fd7ae5f55ae6caa04e2bab8ea0552438 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10508 | completed | 0 |  |  | 0.00159729 | 0.00149728 | -0.000100009 | 0.0245414 | 0.0245414 | 0 | 0.0104563 | 0.0104563 | 0 | 0.00159729 | 0.00149728 | -0.000100009 | 0.00245915 | 0.00245915 | 0 | -0.00245915 | 38.7933 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 346.943 | pass | 1ba536c273dcbef980d2c4d3ffc79e98c31a2d473c95935fc214deb0e1f9e16b | 9.02847 | 1ba536c273dcbef980d2c4d3ffc79e98c31a2d473c95935fc214deb0e1f9e16b | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10509 | completed | 0 |  |  | 0.000764476 | 0.000764476 | 0 | 0.021446 | 0.021446 | 0 | 0.00868628 | 0.00868628 | 0 | 0.000764476 | 0.000764476 | 0 | 0.00657856 | 0.00657856 | 0 | 0.00657856 | 39.1447 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 320.783 | pass | fccc71ffe9f6afd174dca951826f6641e73b13f607cd3b1e6c022217205684c8 | 9.38379 | fccc71ffe9f6afd174dca951826f6641e73b13f607cd3b1e6c022217205684c8 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10510 | completed | 0 |  |  | 0.00145321 | 0.00145321 | -1.11022e-16 | 0.0265131 | 0.0265131 | 0 | 0.00863194 | 0.00863194 | 0 | 0.00145321 | 0.00145321 | -1.11022e-16 | 0.00466098 | 0.00466098 | 0 | -0.00466098 | 38.7076 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 345.389 | pass | ca645fd67a97de2e6a8241629ead3d1f7df989ebdb9c51fa539d50580e6fd200 | 9.25676 | ca645fd67a97de2e6a8241629ead3d1f7df989ebdb9c51fa539d50580e6fd200 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |

### agreement with gates.json

| q | seed | absdiff_eta_final | absdiff_eta_dev | absdiff_rmse | absdiff_tail | absdiff_gmax_final | absdiff_gmax_dev | absdiff_s1 | absdiff_rmse_dev | absdiff_tail_dev | absdiff_s1_dev | G-A | G-F | G-N | S1 | v1_0_G-F | v1_0_run_pass | G-S | v1_1_run_pass | run_pass | outcome | protocol_version | protocol_sha256 | table_sha256 | table_rule | all_agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10502 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10503 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10504 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10505 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10506 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10507 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10508 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10509 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 10510 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10501 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10502 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10503 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10504 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10505 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10506 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10507 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10508 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10509 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 10510 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |

### stage-1 error: G-S (0.05) and S1 (0.10)

| q | criterion | threshold | n | n_pass | cp95_lo | cp95_hi | mean_signed | boot95_lo | boot95_hi | median_signed | sd_signed | bootstrap_resamples | bootstrap_seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | G-S | 0.05 | 10 | 10 | 0.691503 | 1 | -0.011811 | -0.0224141 | -0.0014633 | -0.0103706 | 0.0178499 | 10000 | 20261002 |
| 50 | S1 | 0.1 | 10 | 10 | 0.691503 | 1 | -0.011811 | -0.0224141 | -0.0014633 | -0.0103706 | 0.0178499 | 10000 | 20261002 |
| 60 | G-S | 0.05 | 10 | 10 | 0.691503 | 1 | 0.000906032 | -0.0103165 | 0.0114422 | 0.00205971 | 0.0187419 | 10000 | 20261002 |
| 60 | S1 | 0.1 | 10 | 10 | 0.691503 | 1 | 0.000906032 | -0.0103165 | 0.0114422 | 0.00205971 | 0.0187419 | 10000 | 20261002 |

### distributions

| root | q | metric | n | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | eta_final | 10 | 0.000946099 | 0.000959045 | 0.00108268 | 0.00119612 | 0.00162266 | 0.00278015 | 0.00401444 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | eta_dev | 10 | 0.000725629 | 0.000900093 | 0.000988848 | 0.00108238 | 0.00151934 | 0.00278015 | 0.00401444 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | eta_dev_minus_final | 10 | -0.00022047 | -0.000189571 | -0.000160692 | -8.25891e-05 | -3.78284e-05 | -9.99201e-17 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | rmse | 10 | 0.0150213 | 0.0171832 | 0.0192498 | 0.0200726 | 0.0238441 | 0.027904 | 0.0396621 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | rmse_dev | 10 | 0.0150213 | 0.0171832 | 0.0192498 | 0.0200726 | 0.0238441 | 0.027904 | 0.0396621 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | rmse_dev_minus_final | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | tail | 10 | 0.00495108 | 0.00623836 | 0.00705473 | 0.00803194 | 0.00902849 | 0.0100627 | 0.01187 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | tail_dev | 10 | 0.00495108 | 0.00623836 | 0.00705473 | 0.00803194 | 0.00902849 | 0.0100627 | 0.01187 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | tail_dev_minus_final | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | gmax_final | 10 | 0.000946099 | 0.000959045 | 0.00108268 | 0.00119612 | 0.00162266 | 0.00278015 | 0.00401444 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | gmax_dev | 10 | 0.000725629 | 0.000900093 | 0.000988848 | 0.00108238 | 0.00151934 | 0.00278015 | 0.00401444 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | gmax_dev_minus_final | 10 | -0.00022047 | -0.000189571 | -0.000160692 | -8.25891e-05 | -3.78284e-05 | -9.99201e-17 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | s1 | 10 | 0.00594587 | 0.0088771 | 0.009775 | 0.0119768 | 0.0231559 | 0.0331262 | 0.040692 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | s1_dev | 10 | 0.00594587 | 0.0088771 | 0.009775 | 0.0119768 | 0.0231559 | 0.0331262 | 0.040692 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | s1_dev_minus_final | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 50 | stage1_rel_err_signed | 10 | -0.040692 | -0.0331262 | -0.0231559 | -0.0103706 | 0.00205145 | 0.0096242 | 0.0134169 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | eta_final | 10 | 0.000401645 | 0.000411073 | 0.000533432 | 0.000846929 | 0.00114051 | 0.00146762 | 0.00159729 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | eta_dev | 10 | 0.000360161 | 0.00038064 | 0.000473196 | 0.00080809 | 0.000964316 | 0.00145761 | 0.00149728 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | eta_dev_minus_final | 10 | -0.000200184 | -0.000172813 | -9.50857e-05 | -3.53449e-05 | -3.03661e-06 | -9.99201e-17 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | rmse | 10 | 0.0133277 | 0.0159833 | 0.017285 | 0.0198232 | 0.0236181 | 0.0247385 | 0.0265131 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | rmse_dev | 10 | 0.0133277 | 0.0159833 | 0.017285 | 0.0198232 | 0.0236181 | 0.0247385 | 0.0265131 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | rmse_dev_minus_final | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | tail | 10 | 0.00832454 | 0.0086012 | 0.0087125 | 0.00939291 | 0.0104118 | 0.0109677 | 0.0125433 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | tail_dev | 10 | 0.00832454 | 0.0086012 | 0.0087125 | 0.00939291 | 0.0104118 | 0.0109677 | 0.0125433 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | tail_dev_minus_final | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | gmax_final | 10 | 0.000401645 | 0.000411073 | 0.000533432 | 0.000846929 | 0.00114051 | 0.00146762 | 0.00159729 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | gmax_dev | 10 | 0.000360161 | 0.00038064 | 0.000473196 | 0.00080809 | 0.000964316 | 0.00145761 | 0.00149728 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | gmax_dev_minus_final | 10 | -0.000200184 | -0.000172813 | -9.50857e-05 | -3.53449e-05 | -3.03661e-06 | -9.99201e-17 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | s1 | 10 | 0.00245915 | 0.0044408 | 0.0052909 | 0.0101386 | 0.0237886 | 0.028377 | 0.0314414 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | s1_dev | 10 | 0.00245915 | 0.0044408 | 0.0052909 | 0.0101386 | 0.0237886 | 0.028377 | 0.0314414 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | s1_dev_minus_final | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 | 60 | stage1_rel_err_signed | 10 | -0.0314414 | -0.0250959 | -0.0048115 | 0.00205971 | 0.0115767 | 0.0225875 | 0.0280365 |

### reported metrics

| q | seed | A_stage2_peak_rel_err_signed | A_stage2_peak_locfree_rel_err | A_stage2_peak_locfree_argmax_d | A_stage2_sym_err_max | A_stage2_tail_max | A_DeltaT_over_dw_on_max | A_DeltaT_over_dw_off_max | A_sigma_effort_at_0_t2 | A_smoothed_share_peak_gap_d0 | B_e1_at_0 | B_stage1_rel_err_signed | B_EXP_root_over_dw | B_dReach_over_dw | B_Deltamax_all_over_dw | B_dFull_over_dw | B_Gmax_full_t | B_Gmax_full_d | B_sigma_effort_at_0_t1 | dec_e_tilde | dec_band_lo | dec_band_hi | dec_learning_rel | dec_learning_rel_lo | dec_learning_rel_hi | dec_inherited_rel | dec_inherited_rel_lo | dec_inherited_rel_hi | dec_learning_contains_0 | dec_inherited_contains_0 | dec_band_contiguous | dec_e1_inside_sweep | drift_test_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | -0.122225 | -0.122131 | -0.5 | 3.27607 | 4.27999 | 0.00401444 | 0.00115537 | 3.4457 | 0.318004 | 47.2928 | 0.0134169 | 0.000842715 | 0.00401725 | 0.00401444 | 0.00401725 | 2 | -4 | 3.35565 | 47.0067 | 47.0067 | 47.1867 | 0.00613121 | 0.00227407 | 0.00613121 | 0.00728571 | 0.00728571 | 0.0111429 | no | no | no | yes | yes |
| 50 | 10502 | -0.0680283 | -0.0671942 | -2 | 1.81303 | 3.00361 | 0.00117057 | 0.000471972 | 2.73242 | 0.452963 | 46.1905 | -0.0102045 | 0.000198533 | 0.00117057 | 0.00117057 | 0.00117057 | 2 | -2 | 2.66045 | 46.1567 | 46.1567 | 46.3167 | 0.00072404 | -0.00270453 | 0.00072404 | -0.0109286 | -0.0109286 | -0.0075 | yes | no | yes | yes | yes |
| 50 | 10503 | -0.0235707 | -0.0224627 | -1.5 | 1.47002 | 2.48877 | 0.00172113 | 0.000440938 | 2.81754 | 1.34795 | 45.16 | -0.0322856 | 0.0004736 | 0.0019319 | 0.00172113 | 0.0019319 | 2 | -14 | 2.85935 | 46.2667 | 46.2667 | 46.4967 | -0.0237142 | -0.0286427 | -0.0237142 | -0.00857143 | -0.00857143 | -0.00364286 | no | no | no | yes | yes |
| 50 | 10504 | -0.0712494 | -0.0711073 | -0.5 | 2.35864 | 2.40237 | 0.00132726 | 0.000328007 | 2.77197 | 0.438752 | 47.0961 | 0.00920279 | 0.000316441 | 0.00142158 | 0.00132726 | 0.00142158 | 2 | -2 | 2.69509 | 46.0967 | 46.0967 | 46.3867 | 0.0214171 | 0.0152028 | 0.0214171 | -0.0122143 | -0.0122143 | -0.006 | no | no | no | yes | yes |
| 50 | 10505 | -0.0543192 | -0.0538472 | -1.5 | 2.4712 | 2.21792 | 0.000960484 | 0.000260532 | 2.70308 | 0.561176 | 45.7951 | -0.0186761 | 0.000216983 | 0.000987866 | 0.000960484 | 0.000987866 | 2 | -18 | 2.66826 | 46.1467 | 46.1467 | 46.3167 | -0.00753321 | -0.0111761 | -0.00753321 | -0.0111429 | -0.0111429 | -0.0075 | no | no | yes | yes | yes |
| 50 | 10506 | -0.0984256 | -0.0983059 | 0.5 | 1.78685 | 2.76186 | 0.002643 | 0.000544408 | 3.14462 | 0.36035 | 44.7677 | -0.040692 | 0.000747264 | 0.00294328 | 0.002643 | 0.00294328 | 2 | -4 | 3.03351 | 46.2367 | 46.2367 | 46.4067 | -0.0314777 | -0.0351206 | -0.0314777 | -0.00921429 | -0.00921429 | -0.00557143 | no | no | yes | yes | yes |
| 50 | 10507 | -0.0577588 | -0.0566822 | -2 | 4.86227 | 2.6516 | 0.00106326 | 0.000501285 | 2.74774 | 0.536485 | 45.5164 | -0.0246492 | 0.000293631 | 0.00106877 | 0.00106326 | 0.00106877 | 2 | -18 | 2.70128 | 45.6567 | 45.6567 | 45.8167 | -0.0030063 | -0.00643487 | -0.0030063 | -0.0216429 | -0.0216429 | -0.0182143 | no | no | yes | yes | yes |
| 50 | 10508 | -0.0656348 | -0.0654965 | -0.5 | 1.26619 | 2.25745 | 0.00114095 | 0.000363148 | 3.11072 | 0.53452 | 46.2172 | -0.00963182 | 0.00036348 | 0.00116081 | 0.00114095 | 0.00116081 | 2 | -2 | 3.0993 | 46.5067 | 46.5067 | 46.7867 | -0.00620325 | -0.0122032 | -0.00620325 | -0.00342857 | -0.00342857 | 0.00257143 | no | yes | no | yes | yes |
| 50 | 10509 | -0.035379 | -0.0352603 | 0.5 | 2.85899 | 2.68895 | 0.00122168 | 0.000384956 | 2.9942 | 0.954407 | 46.175 | -0.0105367 | 0.000372701 | 0.00133357 | 0.00122168 | 0.00133357 | 2 | -14 | 3.0043 | 47.0067 | 47.0067 | 47.2567 | -0.0178224 | -0.0231795 | -0.0178224 | 0.00728571 | 0.00728571 | 0.0126429 | no | no | no | yes | yes |
| 50 | 10510 | -0.0593674 | -0.0593674 | 0 | 2.17063 | 2.51676 | 0.000946099 | 0.000449828 | 2.7113 | 0.515024 | 46.9441 | 0.00594587 | 0.000138118 | 0.000946265 | 0.000946099 | 0.000946265 | 2 | -2 | 2.65953 | 46.7267 | 46.7267 | 46.8867 | 0.00466016 | 0.00123159 | 0.00466016 | 0.00128571 | 0.00128571 | 0.00471429 | no | no | yes | yes | yes |
| 60 | 10501 | -0.0469268 | -0.0468315 | 0.5 | 1.14297 | 2.11008 | 0.00041212 | 0.000314045 | 3.07808 | 0.616562 | 37.6662 | -0.0314414 | 0.000208413 | 0.00054703 | 0.00041212 | 0.00054703 | 2 | -2 | 2.78233 | 38.7589 | 38.7489 | 38.8289 | -0.0280986 | -0.0298986 | -0.0278414 | -0.00334286 | -0.0036 | -0.00154286 | no | no | yes | yes | yes |
| 60 | 10502 | -0.0319254 | -0.0314894 | -1.5 | 1.69289 | 2.25965 | 0.000590891 | 0.000360931 | 2.84118 | 0.836459 | 39.9792 | 0.0280365 | 0.00020993 | 0.000695616 | 0.000590891 | 0.000695616 | 2 | -16 | 2.53409 | 38.8989 | 38.8889 | 38.9789 | 0.0277794 | 0.0257222 | 0.0280365 | 0.000257143 | 0 | 0.00231429 | no | yes | yes | yes | yes |
| 60 | 10503 | -0.0721634 | -0.0720225 | -1 | 3.35659 | 2.1341 | 0.00102148 | 0.000269717 | 3.15893 | 0.411487 | 39.7437 | 0.0219821 | 0.000188485 | 0.00105428 | 0.00102148 | 0.00105428 | 2 | -2 | 2.85601 | 39.1789 | 39.1689 | 39.2589 | 0.0145249 | 0.0124678 | 0.0147821 | 0.00745714 | 0.0072 | 0.00951429 | no | no | yes | yes | yes |
| 60 | 10504 | -0.0485514 | -0.0483811 | -1 | 1.37311 | 2.1125 | 0.000929381 | 0.000316732 | 2.79263 | 0.540622 | 38.6998 | -0.00486167 | 0.000222875 | 0.000936498 | 0.000929381 | 0.000936498 | 2 | -22 | 2.48626 | 38.8589 | 38.8389 | 38.9289 | -0.00409025 | -0.00589025 | -0.00357596 | -0.000771429 | -0.00128571 | 0.00102857 | no | yes | yes | yes | yes |
| 60 | 10505 | -0.0759133 | -0.0755798 | 1.5 | 0.929512 | 2.5066 | 0.00118019 | 0.000349411 | 3.22952 | 0.399911 | 37.9404 | -0.0243908 | 0.000203772 | 0.0012796 | 0.00118019 | 0.0012796 | 2 | -2 | 2.88088 | 38.8389 | 38.8189 | 38.9189 | -0.0231051 | -0.0251622 | -0.0225908 | -0.00128571 | -0.0018 | 0.000771429 | no | yes | yes | yes | yes |
| 60 | 10506 | -0.0428923 | -0.0422296 | -1.5 | 1.59292 | 2.0574 | 0.000401645 | 0.000301203 | 2.7375 | 0.59986 | 39.395 | 0.0130148 | 0.000132701 | 0.000451923 | 0.000401645 | 0.000451923 | 2 | -18 | 2.46708 | 38.6689 | 38.6589 | 38.7389 | 0.018672 | 0.016872 | 0.0189291 | -0.00565714 | -0.00591429 | -0.00385714 | no | no | yes | yes | yes |
| 60 | 10507 | -0.051511 | -0.0513581 | 1 | 2.26594 | 2.06761 | 0.000514279 | 0.000252842 | 2.84582 | 0.519272 | 39.1713 | 0.00726242 | 0.000107607 | 0.000515272 | 0.000514279 | 0.000515272 | 2 | -2 | 2.55032 | 39.0589 | 39.0389 | 39.1389 | 0.00289099 | 0.000833844 | 0.00340527 | 0.00437143 | 0.00385714 | 0.00642857 | no | no | yes | yes | yes |
| 60 | 10508 | -0.0917359 | -0.0905231 | -2.5 | 2.83273 | 2.31835 | 0.00159729 | 0.000381182 | 3.24238 | 0.332255 | 38.7933 | -0.00245915 | 0.000232389 | 0.00159729 | 0.00159729 | 0.00159729 | 2 | -2 | 2.87449 | 38.7689 | 38.7589 | 38.8589 | 0.000626567 | -0.00168772 | 0.00088371 | -0.00308571 | -0.00334286 | -0.000771429 | yes | no | yes | yes | yes |
| 60 | 10509 | -0.0508767 | -0.0504808 | -1.5 | 1.95239 | 1.87929 | 0.000764476 | 0.000237976 | 2.97803 | 0.550195 | 39.1447 | 0.00657856 | 0.000174035 | 0.000796481 | 0.000764476 | 0.000796481 | 2 | -24 | 2.68804 | 38.5589 | 38.5489 | 38.6389 | 0.0150643 | 0.0130071 | 0.0153214 | -0.00848571 | -0.00874286 | -0.00642857 | no | no | yes | yes | yes |
| 60 | 10510 | -0.0209694 | -0.0207133 | -1 | 1.03723 | 2.03359 | 0.00145321 | 0.000284862 | 2.99923 | 1.3444 | 38.7076 | -0.00466098 | 0.000305789 | 0.00145935 | 0.00145321 | 0.00145935 | 2 | -20 | 2.71055 | 38.8389 | 38.8289 | 38.9189 | -0.00337526 | -0.00543241 | -0.00311812 | -0.00128571 | -0.00154286 | 0.000771429 | no | yes | yes | yes | yes |

### EXP_root vs stage-1 error

| q | seed | stage1_rel_err_signed | stage1_rel_err_sq | EXP_root_over_dw |
|---|---|---|---|---|
| 50 | 10501 | 0.0134169 | 0.000180014 | 0.000842715 |
| 50 | 10502 | -0.0102045 | 0.000104132 | 0.000198533 |
| 50 | 10503 | -0.0322856 | 0.00104236 | 0.0004736 |
| 50 | 10504 | 0.00920279 | 8.46913e-05 | 0.000316441 |
| 50 | 10505 | -0.0186761 | 0.000348796 | 0.000216983 |
| 50 | 10506 | -0.040692 | 0.00165584 | 0.000747264 |
| 50 | 10507 | -0.0246492 | 0.000607581 | 0.000293631 |
| 50 | 10508 | -0.00963182 | 9.27719e-05 | 0.00036348 |
| 50 | 10509 | -0.0105367 | 0.000111022 | 0.000372701 |
| 50 | 10510 | 0.00594587 | 3.53534e-05 | 0.000138118 |
| 60 | 10501 | -0.0314414 | 0.000988565 | 0.000208413 |
| 60 | 10502 | 0.0280365 | 0.000786045 | 0.00020993 |
| 60 | 10503 | 0.0219821 | 0.000483212 | 0.000188485 |
| 60 | 10504 | -0.00486167 | 2.36359e-05 | 0.000222875 |
| 60 | 10505 | -0.0243908 | 0.000594912 | 0.000203772 |
| 60 | 10506 | 0.0130148 | 0.000169385 | 0.000132701 |
| 60 | 10507 | 0.00726242 | 5.27427e-05 | 0.000107607 |
| 60 | 10508 | -0.00245915 | 6.04741e-06 | 0.000232389 |
| 60 | 10509 | 0.00657856 | 4.32774e-05 | 0.000174035 |
| 60 | 10510 | -0.00466098 | 2.17247e-05 | 0.000305789 |

### EXP_root vs err^2 fit

| q | n | intercept | slope_on_err_sq | r2 | model |
|---|---|---|---|---|---|
| 50 | 10 | 0.000301188 | 0.223242 | 0.264624 | EXP_root/DW = a + b * (stage-1 rel. err)^2, ordinary least squares |
| 60 | 10 | 0.000197847 | 0.00237524 | 0.000255216 | EXP_root/DW = a + b * (stage-1 rel. err)^2, ordinary least squares |

### paired by (q, seed): this root - rehearsal_v1_1

| q | metric | n_pairs | mean_diff | median_diff | n_new_lower | n_new_higher | n_equal | mean_ref | mean_new | sd_ref_ddof1 | sd_new_ddof1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | eta_final | 10 | 0 | 0 | 0 | 0 | 10 | 0.00162089 | 0.00162089 | 0.000981212 | 0.000981212 |
| 50 | rmse | 10 | 0 | 0 | 0 | 0 | 10 | 0.0225074 | 0.0225074 | 0.00688924 | 0.00688924 |
| 50 | tail | 10 | 0 | 0 | 0 | 0 | 10 | 0.00812864 | 0.00812864 | 0.00193059 | 0.00193059 |
| 50 | gmax_final | 10 | -0.00014688 | 0 | 2 | 0 | 8 | 0.00176777 | 0.00162089 | 0.000979783 | 0.000981212 |
| 50 | s1 | 10 | -0.0238577 | -0.0153821 | 9 | 1 | 0 | 0.0413818 | 0.0175241 | 0.0309167 | 0.0115065 |
| 50 | stage1_rel_err_signed | 10 | 0.00172461 | -0.00184645 | 5 | 5 | 0 | -0.0135356 | -0.011811 | 0.0515267 | 0.0178499 |
| 50 | e1_at_0 | 10 | 0.080482 | -0.0861676 | 5 | 5 | 0 | 46.035 | 46.1155 | 2.40458 | 0.832993 |
| 50 | eta_dev_minus_final | 10 | 0 | 0 | 0 | 0 | 10 | -9.56386e-05 | -9.56386e-05 | 7.88516e-05 | 7.88516e-05 |
| 50 | gmax_dev_minus_final | 10 | -5.82185e-06 | 0 | 2 | 0 | 8 | -8.98167e-05 | -9.56386e-05 | 8.22447e-05 | 7.88516e-05 |
| 60 | eta_final | 10 | 0 | 0 | 0 | 0 | 10 | 0.000886496 | 0.000886496 | 0.000426045 | 0.000426045 |
| 60 | rmse | 10 | 0 | 0 | 0 | 0 | 10 | 0.0201484 | 0.0201484 | 0.00426672 | 0.00426672 |
| 60 | tail | 10 | 0 | 0 | 0 | 0 | 10 | 0.00972904 | 0.00972904 | 0.00130412 | 0.00130412 |
| 60 | gmax_final | 10 | -0.000206185 | -3.56197e-05 | 5 | 0 | 5 | 0.00109268 | 0.000886496 | 0.000671821 | 0.000426045 |
| 60 | s1 | 10 | -0.0378342 | -0.0353488 | 9 | 1 | 0 | 0.052303 | 0.0144688 | 0.0428007 | 0.0109345 |
| 60 | stage1_rel_err_signed | 10 | 0.0030127 | 0.000933258 | 5 | 5 | 0 | -0.00210667 | 0.000906032 | 0.0697605 | 0.0187419 |
| 60 | e1_at_0 | 10 | 0.117161 | 0.0362934 | 5 | 5 | 0 | 38.807 | 38.9241 | 2.71291 | 0.728853 |
| 60 | eta_dev_minus_final | 10 | 0 | 0 | 0 | 0 | 10 | -6.33117e-05 | -6.33117e-05 | 7.33812e-05 | 7.33812e-05 |
| 60 | gmax_dev_minus_final | 10 | -4.76624e-05 | -2.98798e-06 | 5 | 0 | 5 | -1.56493e-05 | -6.33117e-05 | 4.65084e-05 | 7.33812e-05 |

### pass/fail flips by (q, seed): this root vs rehearsal_v1_1

| q | flag | n_pairs | n_ref_pass | n_new_pass | n_ref_pass_new_fail | n_ref_fail_new_pass |
|---|---|---|---|---|---|---|
| 50 | G-A_pass | 10 | 10 | 10 | 0 | 0 |
| 50 | G-F_pass | 10 | 10 | 10 | 0 | 0 |
| 50 | G-N_pass | 10 | 10 | 10 | 0 | 0 |
| 50 | G-S_pass | 10 | 8 | 10 | 0 | 2 |
| 50 | S1_pass | 10 | 9 | 10 | 0 | 1 |
| 50 | v1_1_run_pass | 10 | 10 | 10 | 0 | 0 |
| 50 | run_pass | 10 | 8 | 10 | 0 | 2 |
| 60 | G-A_pass | 10 | 10 | 10 | 0 | 0 |
| 60 | G-F_pass | 10 | 10 | 10 | 0 | 0 |
| 60 | G-N_pass | 10 | 10 | 10 | 0 | 0 |
| 60 | G-S_pass | 10 | 6 | 10 | 0 | 4 |
| 60 | S1_pass | 10 | 8 | 10 | 0 | 2 |
| 60 | v1_1_run_pass | 10 | 10 | 10 | 0 | 0 |
| 60 | run_pass | 10 | 6 | 10 | 0 | 4 |
