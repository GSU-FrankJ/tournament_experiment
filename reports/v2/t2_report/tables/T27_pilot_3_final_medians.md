# T27: Pilot 3 final medians

- priority: core; status: generated; tier: final and development
- sources: `results/v2_pilots/pilot3/analysis/final_table.csv`, `Pilot 3 final_v2.json (40 runs)` (40 files), `results/v2_pilots/pilot3/analysis/decomposition_residual_band.csv`
- built by: `tools/v2/report/sec_pilot23.py:build_t27`; base commit `cb0b541`
- transformation: Median, q25, q75 (numpy linear interpolation), min, max and n over the 10 seeds (10501-10510) per (q, arm) at global u1000. Development-tier values from results/v2_pilots/pilot3/analysis/final_table.csv (last training-time checkpoint; equal to final_v2.json['development'] within 9.9e-17); final-tier values from final_v2.json['final']. learning/inherited terms: residual-band decomposition with the parent band (final tier); final_table.csv equals the u1000 weight-export rows of decomposition_residual_band.csv (max abs diff 7.10543e-15). Within-run SD (ddof = 1) and range of e_hat_1(0) over the weight exports u900, 925, 950, 975, 1000 (n_last5 = [5]). Learning band excludes 0 in 39 of 40 final rows.

Pilot 3 at global u1000, per q and arm (stochastic vs mean continuation), n = 10 seeds (10501-10510). e_hat_1(0), within-run SD/range and sigma_1(0) in effort units (raw); errors and decomposition terms relative to e_1*(0); deviation metrics divided by Delta W = 4.

| q | arm | arm_short | metric | tier | median | q25 | q75 | min | max | n | n_true | argmax_t_d_counts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | B2_frozen_s1norm | stochastic | e1_at_0 | tier-independent | 47.58 | 43.3 | 48.48 | 42.23 | 51.7 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | stage1_rel_err_signed | tier-independent | 0.01962 | -0.07218 | 0.03879 | -0.09501 | 0.1078 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | stage1_rel_err_abs | tier-independent | 0.06479 | 0.03304 | 0.09121 | 0.01374 | 0.1078 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | learning_rel | final | 0.01553 | -0.0717 | 0.04769 | -0.09586 | 0.1059 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | learning_rel_abs | final | 0.06725 | 0.03619 | 0.09464 | 0.0141 | 0.1059 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | inherited_rel | final | 0.001393 | -0.003375 | 0.009321 | -0.01179 | 0.01607 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | inherited_rel_abs | final | 0.006643 | 0.003268 | 0.01077 | 0.0008571 | 0.01607 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | learning_contains_0 | final |  |  |  |  |  | 10 | 0 |  |
| 50 | B2_frozen_s1norm | stochastic | inherited_contains_0 | final |  |  |  |  |  | 10 | 1 |  |
| 50 | B2_frozen_s1norm | stochastic | within_run_sd_e1_last5 | tier-independent | 1.947 | 1.435 | 2.416 | 1.273 | 3.121 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | within_run_range_e1_last5 | tier-independent | 5.004 | 3.701 | 5.908 | 2.665 | 6.718 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | sigma_effort_at_0_t1 | tier-independent | 3.646 | 3.407 | 3.859 | 3.274 | 4.107 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | Gmax_full_over_dw | development | 0.004606 | 0.004373 | 0.005746 | 0.00185 | 0.006638 | 10 |  | (t=2, d=-4) x9; (t=1, d=0) x1 |
| 50 | B2_frozen_s1norm | stochastic | Gmax_full_over_dw | final | 0.004606 | 0.004373 | 0.005746 | 0.001865 | 0.006718 | 10 |  | (t=2, d=-4) x8; (t=1, d=0) x1; (t=2, d=-6) x1 |
| 50 | B2_frozen_s1norm | stochastic | EXP_root_over_dw | development | 0.001705 | 0.001278 | 0.003441 | 0.0009865 | 0.004917 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | EXP_root_over_dw | final | 0.001707 | 0.001268 | 0.003448 | 0.0009543 | 0.004912 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | dReach_over_dw | development | 0.005795 | 0.004887 | 0.00687 | 0.002998 | 0.009355 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | dReach_over_dw | final | 0.00579 | 0.004874 | 0.006855 | 0.003043 | 0.00942 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | Deltamax_all_over_dw | development | 0.004606 | 0.004373 | 0.005746 | 0.001511 | 0.006638 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | Deltamax_all_over_dw | final | 0.004606 | 0.004373 | 0.005746 | 0.001544 | 0.006718 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | dFull_over_dw | development | 0.005795 | 0.004887 | 0.00687 | 0.002998 | 0.009355 | 10 |  |  |
| 50 | B2_frozen_s1norm | stochastic | dFull_over_dw | final | 0.00579 | 0.004874 | 0.006855 | 0.003043 | 0.00942 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | e1_at_0 | tier-independent | 45.34 | 43.05 | 46.98 | 40.06 | 51.57 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | stage1_rel_err_signed | tier-independent | -0.02835 | -0.07754 | 0.006756 | -0.1416 | 0.105 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | stage1_rel_err_abs | tier-independent | 0.06763 | 0.02321 | 0.07926 | 0.005032 | 0.1416 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | learning_rel | final | -0.02974 | -0.08605 | 0.001721 | -0.1382 | 0.09539 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | learning_rel_abs | final | 0.07384 | 0.02487 | 0.09122 | 0.0001035 | 0.1382 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | inherited_rel | final | 0.001393 | -0.003375 | 0.009321 | -0.01179 | 0.01607 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | inherited_rel_abs | final | 0.006643 | 0.003268 | 0.01077 | 0.0008571 | 0.01607 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | learning_contains_0 | final |  |  |  |  |  | 10 | 1 |  |
| 50 | B2_frozen_s1norm_mean | mean | inherited_contains_0 | final |  |  |  |  |  | 10 | 1 |  |
| 50 | B2_frozen_s1norm_mean | mean | within_run_sd_e1_last5 | tier-independent | 1.044 | 0.7452 | 1.796 | 0.5107 | 2.36 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | within_run_range_e1_last5 | tier-independent | 2.378 | 1.744 | 4.614 | 1.282 | 6.279 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | sigma_effort_at_0_t1 | tier-independent | 3.624 | 3.436 | 3.747 | 3.298 | 4.195 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | Gmax_full_over_dw | development | 0.00487 | 0.004565 | 0.005746 | 0.003827 | 0.006638 | 10 |  | (t=2, d=-4) x9; (t=1, d=0) x1 |
| 50 | B2_frozen_s1norm_mean | mean | Gmax_full_over_dw | final | 0.004879 | 0.004565 | 0.005746 | 0.003827 | 0.006718 | 10 |  | (t=2, d=-4) x8; (t=1, d=0) x1; (t=2, d=-6) x1 |
| 50 | B2_frozen_s1norm_mean | mean | EXP_root_over_dw | development | 0.002336 | 0.001778 | 0.003233 | 0.0006932 | 0.005093 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | EXP_root_over_dw | final | 0.002318 | 0.00179 | 0.003215 | 0.0007006 | 0.005111 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | dReach_over_dw | development | 0.006458 | 0.005429 | 0.006804 | 0.004308 | 0.007742 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | dReach_over_dw | final | 0.006452 | 0.005426 | 0.006826 | 0.004308 | 0.007718 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | Deltamax_all_over_dw | development | 0.004691 | 0.004565 | 0.005746 | 0.003827 | 0.006638 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | Deltamax_all_over_dw | final | 0.0047 | 0.004565 | 0.005746 | 0.003827 | 0.006718 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | dFull_over_dw | development | 0.006458 | 0.005429 | 0.006804 | 0.004308 | 0.007742 | 10 |  |  |
| 50 | B2_frozen_s1norm_mean | mean | dFull_over_dw | final | 0.006452 | 0.005426 | 0.006826 | 0.004308 | 0.007718 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | e1_at_0 | tier-independent | 39.27 | 36.96 | 39.9 | 33.31 | 41.78 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | stage1_rel_err_signed | tier-independent | 0.009708 | -0.04969 | 0.02604 | -0.1434 | 0.07444 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | stage1_rel_err_abs | tier-independent | 0.04948 | 0.02456 | 0.06674 | 0.002742 | 0.1434 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | learning_rel | final | 0.007651 | -0.05715 | 0.02295 | -0.1439 | 0.06827 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | learning_rel_abs | final | 0.05282 | 0.0207 | 0.06684 | 0.00377 | 0.1439 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | inherited_rel | final | 0.0006429 | -0.0008357 | 0.005914 | -0.005657 | 0.01157 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | inherited_rel_abs | final | 0.005014 | 0.0008357 | 0.006043 | 0.0002571 | 0.01157 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | learning_contains_0 | final |  |  |  |  |  | 10 | 0 |  |
| 60 | B2_frozen_s1norm | stochastic | inherited_contains_0 | final |  |  |  |  |  | 10 | 3 |  |
| 60 | B2_frozen_s1norm | stochastic | within_run_sd_e1_last5 | tier-independent | 1.802 | 1.257 | 2.187 | 0.6852 | 3.488 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | within_run_range_e1_last5 | tier-independent | 4.61 | 2.983 | 5.349 | 1.746 | 7.904 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | sigma_effort_at_0_t1 | tier-independent | 3.485 | 3.357 | 3.654 | 3.315 | 3.756 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | Gmax_full_over_dw | development | 0.001833 | 0.001767 | 0.002251 | 0.001181 | 0.003512 | 10 |  | (t=2, d=-4) x8; (t=2, d=-32) x1; (t=2, d=40) x1 |
| 60 | B2_frozen_s1norm | stochastic | Gmax_full_over_dw | final | 0.001891 | 0.001837 | 0.002291 | 0.001243 | 0.003512 | 10 |  | (t=2, d=-2) x6; (t=2, d=-4) x2; (t=2, d=-30) x1; (t=2, d=40) x1 |
| 60 | B2_frozen_s1norm | stochastic | EXP_root_over_dw | development | 0.0007316 | 0.0005208 | 0.001034 | 0.0003708 | 0.003287 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | EXP_root_over_dw | final | 0.0007127 | 0.0005239 | 0.001034 | 0.000371 | 0.003289 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | dReach_over_dw | development | 0.002383 | 0.001946 | 0.002556 | 0.00125 | 0.006242 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | dReach_over_dw | final | 0.002422 | 0.002022 | 0.002623 | 0.001261 | 0.006245 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | Deltamax_all_over_dw | development | 0.001833 | 0.001767 | 0.002251 | 0.001181 | 0.003512 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | Deltamax_all_over_dw | final | 0.001891 | 0.001837 | 0.002291 | 0.001243 | 0.003512 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | dFull_over_dw | development | 0.002383 | 0.001946 | 0.002556 | 0.00125 | 0.006242 | 10 |  |  |
| 60 | B2_frozen_s1norm | stochastic | dFull_over_dw | final | 0.002422 | 0.002022 | 0.002623 | 0.001261 | 0.006245 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | e1_at_0 | tier-independent | 37.98 | 37.02 | 42.73 | 32.1 | 43.92 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | stage1_rel_err_signed | tier-independent | -0.02346 | -0.04795 | 0.09885 | -0.1746 | 0.1293 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | stage1_rel_err_abs | tier-independent | 0.08852 | 0.04154 | 0.1262 | 0.02108 | 0.1746 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | learning_rel | final | -0.02655 | -0.04712 | 0.09505 | -0.1754 | 0.1231 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | learning_rel_abs | final | 0.08608 | 0.038 | 0.1187 | 0.02005 | 0.1754 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | inherited_rel | final | 0.0006429 | -0.0008357 | 0.005914 | -0.005657 | 0.01157 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | inherited_rel_abs | final | 0.005014 | 0.0008357 | 0.006043 | 0.0002571 | 0.01157 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | learning_contains_0 | final |  |  |  |  |  | 10 | 0 |  |
| 60 | B2_frozen_s1norm_mean | mean | inherited_contains_0 | final |  |  |  |  |  | 10 | 3 |  |
| 60 | B2_frozen_s1norm_mean | mean | within_run_sd_e1_last5 | tier-independent | 1.988 | 1.327 | 2.171 | 1.001 | 2.607 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | within_run_range_e1_last5 | tier-independent | 4.659 | 3.261 | 5.515 | 2.595 | 6.232 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | sigma_effort_at_0_t1 | tier-independent | 3.498 | 3.4 | 3.702 | 3.208 | 3.849 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | Gmax_full_over_dw | development | 0.002321 | 0.002082 | 0.003164 | 0.00177 | 0.00445 | 10 |  | (t=1, d=0) x5; (t=2, d=-4) x5 |
| 60 | B2_frozen_s1norm_mean | mean | Gmax_full_over_dw | final | 0.002324 | 0.002081 | 0.00316 | 0.001854 | 0.004449 | 10 |  | (t=1, d=0) x5; (t=2, d=-2) x3; (t=2, d=-4) x2 |
| 60 | B2_frozen_s1norm_mean | mean | EXP_root_over_dw | development | 0.001452 | 0.0007128 | 0.0023 | 0.0005325 | 0.00445 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | EXP_root_over_dw | final | 0.001445 | 0.0006941 | 0.002267 | 0.0005095 | 0.004449 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | dReach_over_dw | development | 0.003369 | 0.002514 | 0.003767 | 0.00235 | 0.005943 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | dReach_over_dw | final | 0.003425 | 0.002518 | 0.003795 | 0.002373 | 0.00604 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | Deltamax_all_over_dw | development | 0.002249 | 0.001893 | 0.003093 | 0.00146 | 0.004111 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | Deltamax_all_over_dw | final | 0.002252 | 0.001896 | 0.003105 | 0.00145 | 0.004112 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | dFull_over_dw | development | 0.003369 | 0.002514 | 0.003767 | 0.00235 | 0.005943 | 10 |  |  |
| 60 | B2_frozen_s1norm_mean | mean | dFull_over_dw | final | 0.003425 | 0.002518 | 0.003795 | 0.002373 | 0.00604 | 10 |  |  |
