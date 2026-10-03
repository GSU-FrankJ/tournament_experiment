# T44: Stage-2 tail averaging

- priority: supp; status: generated; tier: development
- sources: `results/v2_pilots/pilot4/analysis/candidates_all.csv`, `results/v2_pilots/pilot4/analysis/paired_summary.csv`
- built by: `tools/v2/report/sec_ext.py:build_t44`; base commit `cb0b541`
- transformation: Blocks: candidates = median/IQR/min/max over seeds 10501-10510 of the tail-averaged candidates (candidate_K = pointwise average of the last K weight-export mappings, K = 1 the last iterate; extension at u1600 with K = 16 covering u1225-u1600; 2a arms at u1600); paired_K_vs_K1 = paired difference (K) - (K = 1) per (q, seed) from paired_summary.csv (bootstrap: 95% percentile CI of the mean paired difference, 10,000 resamples, numpy seed 20261001; n_better counts pairs with K < K=1 for smaller-is-better metrics). Development tier for eta_2 / Delta_2 (state step 4, effort step 1, GL 16); recovery metrics tier-independent; sigma is not defined for K > 1 (the average is not a Beta). The 2a constant arm is bit-identical to the extension over u1201-u1600, so its candidates equal the extension's for K = 1, 4, 8 (max abs diff 0 over the 13 metrics, 60 per-run rows).

Pilot 4 sections 1c.2 (extension, u1600) and 2a (u1600): tail-averaged stage-2 candidates.

| block | family | arm | K | q | metric | tier | quantity | median | q25 | q75 | min | max | n | n_pairs | mean | n_neg | n_pos | n_zero | boot_ci95_lo | boot_ci95_hi | n_better | better_if |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| candidates | ext | ext | 1 | 50 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.06835 | -0.0709 | -0.05722 | -0.1286 | -0.03741 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 50 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.06835 | 0.05722 | 0.0709 | 0.03741 | 0.1286 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 50 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.06826 | -0.07074 | -0.05652 | -0.1286 | -0.0368 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.06826 | 0.05652 | 0.07074 | 0.0368 | 0.1286 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.02605 | 0.0221 | 0.02871 | 0.01422 | 0.05976 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 50 | stage2_tail_mean | tier-independent | candidate value | 0.515 | 0.4559 | 0.5639 | 0.3528 | 0.8831 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 50 | stage2_tail_max | tier-independent | candidate value | 2.22 | 2.131 | 2.597 | 1.656 | 4.907 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.007358 | 0.006513 | 0.008055 | 0.00504 | 0.01262 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 50 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03172 | 0.03044 | 0.03711 | 0.02365 | 0.07009 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 50 | stage2_sym_err_max | tier-independent | candidate value | 2.638 | 1.742 | 3.529 | 1.447 | 5.473 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 50 | eta_T_over_dw | development | candidate value | 0.001469 | 0.001119 | 0.002431 | 0.0007199 | 0.006971 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 50 | DeltaT_over_dw_on_max | development | candidate value | 0.001469 | 0.001119 | 0.002431 | 0.0007199 | 0.006971 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 50 | DeltaT_over_dw_off_max | development | candidate value | 0.0003395 | 0.000264 | 0.000386 | 0.0001463 | 0.001545 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.06294 | -0.08174 | -0.05453 | -0.09494 | -0.04077 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.06294 | 0.05453 | 0.08174 | 0.04077 | 0.09494 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.06247 | -0.08075 | -0.05435 | -0.09311 | -0.04065 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.06247 | 0.05435 | 0.08075 | 0.04065 | 0.09311 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.0201 | 0.01888 | 0.02321 | 0.01642 | 0.02746 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | stage2_tail_mean | tier-independent | candidate value | 0.5552 | 0.4807 | 0.5654 | 0.3847 | 0.889 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | stage2_tail_max | tier-independent | candidate value | 2.334 | 2.252 | 2.403 | 1.908 | 4.379 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.007931 | 0.006867 | 0.008077 | 0.005495 | 0.0127 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03334 | 0.03218 | 0.03433 | 0.02726 | 0.06256 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | stage2_sym_err_max | tier-independent | candidate value | 1.809 | 1.628 | 2.328 | 0.6365 | 2.425 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | eta_T_over_dw | development | candidate value | 0.001168 | 0.001018 | 0.001855 | 0.0004987 | 0.002563 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | DeltaT_over_dw_on_max | development | candidate value | 0.001168 | 0.001018 | 0.001855 | 0.0004987 | 0.002563 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 50 | DeltaT_over_dw_off_max | development | candidate value | 0.0003489 | 0.0002959 | 0.0003808 | 0.0002174 | 0.001187 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.07022 | -0.07666 | -0.05445 | -0.107 | -0.03448 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.07022 | 0.05445 | 0.07666 | 0.03448 | 0.107 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.06972 | -0.07593 | -0.05426 | -0.1069 | -0.03396 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.06972 | 0.05426 | 0.07593 | 0.03396 | 0.1069 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.01917 | 0.01776 | 0.02075 | 0.01438 | 0.02906 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | stage2_tail_mean | tier-independent | candidate value | 0.5619 | 0.4836 | 0.5856 | 0.4059 | 0.861 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | stage2_tail_max | tier-independent | candidate value | 2.34 | 2.228 | 2.49 | 2.108 | 4.282 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.008027 | 0.006908 | 0.008366 | 0.005799 | 0.0123 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03343 | 0.03183 | 0.03557 | 0.03011 | 0.06118 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | stage2_sym_err_max | tier-independent | candidate value | 1.444 | 1.278 | 1.87 | 0.6973 | 2.318 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | eta_T_over_dw | development | candidate value | 0.001285 | 0.001068 | 0.001526 | 0.0005877 | 0.00304 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | DeltaT_over_dw_on_max | development | candidate value | 0.001285 | 0.001068 | 0.001526 | 0.0005877 | 0.00304 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 50 | DeltaT_over_dw_off_max | development | candidate value | 0.0003247 | 0.0003087 | 0.0004052 | 0.0002631 | 0.001091 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.0667 | -0.07333 | -0.0568 | -0.09952 | -0.04044 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.0667 | 0.0568 | 0.07333 | 0.04044 | 0.09952 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.06667 | -0.07314 | -0.05573 | -0.09952 | -0.03976 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.06667 | 0.05573 | 0.07314 | 0.03976 | 0.09952 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.01669 | 0.01594 | 0.01888 | 0.01516 | 0.03054 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | stage2_tail_mean | tier-independent | candidate value | 0.5778 | 0.5144 | 0.6152 | 0.4155 | 0.882 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | stage2_tail_max | tier-independent | candidate value | 2.436 | 2.297 | 2.555 | 2.142 | 4.415 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.008255 | 0.007349 | 0.008789 | 0.005936 | 0.0126 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03479 | 0.03281 | 0.0365 | 0.0306 | 0.06307 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | stage2_sym_err_max | tier-independent | candidate value | 1.426 | 1.179 | 1.693 | 0.9799 | 2.645 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | eta_T_over_dw | development | candidate value | 0.0009973 | 0.0009155 | 0.001286 | 0.0006005 | 0.002655 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | DeltaT_over_dw_on_max | development | candidate value | 0.0009973 | 0.0009155 | 0.001286 | 0.0006005 | 0.002655 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 50 | DeltaT_over_dw_off_max | development | candidate value | 0.0003602 | 0.0003316 | 0.0003902 | 0.0003082 | 0.00116 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.06003 | -0.07362 | -0.04034 | -0.07572 | 0.003541 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.06003 | 0.04034 | 0.07362 | 0.003541 | 0.07572 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.05585 | -0.07326 | -0.03969 | -0.07572 | 0.003559 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.05585 | 0.03969 | 0.07326 | 0.003559 | 0.07572 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.02389 | 0.02089 | 0.03157 | 0.01864 | 0.03376 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | stage2_tail_mean | tier-independent | candidate value | 0.5312 | 0.501 | 0.5663 | 0.4143 | 0.6374 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | stage2_tail_max | tier-independent | candidate value | 1.901 | 1.835 | 2.109 | 1.708 | 2.556 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.009106 | 0.008588 | 0.009708 | 0.007102 | 0.01093 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03259 | 0.03146 | 0.03616 | 0.02927 | 0.04382 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | stage2_sym_err_max | tier-independent | candidate value | 2.39 | 1.517 | 2.535 | 0.8313 | 3.963 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | eta_T_over_dw | development | candidate value | 0.0009012 | 0.0007233 | 0.0009584 | 0.0003882 | 0.002156 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | DeltaT_over_dw_on_max | development | candidate value | 0.0009012 | 0.0007233 | 0.0009584 | 0.0003882 | 0.002156 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 1 | 60 | DeltaT_over_dw_off_max | development | candidate value | 0.0002439 | 0.0002221 | 0.0002957 | 0.0001492 | 0.000465 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.06243 | -0.06688 | -0.05679 | -0.08176 | -0.03148 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.06243 | 0.05679 | 0.06688 | 0.03148 | 0.08176 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.06217 | -0.06654 | -0.05594 | -0.08172 | -0.03125 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.06217 | 0.05594 | 0.06654 | 0.03125 | 0.08172 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.0179 | 0.0173 | 0.01925 | 0.01626 | 0.02177 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | stage2_tail_mean | tier-independent | candidate value | 0.5404 | 0.5093 | 0.5688 | 0.4516 | 0.6459 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | stage2_tail_max | tier-independent | candidate value | 2.064 | 1.874 | 2.189 | 1.749 | 2.282 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.009264 | 0.008731 | 0.00975 | 0.007741 | 0.01107 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03538 | 0.03212 | 0.03753 | 0.02999 | 0.03912 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | stage2_sym_err_max | tier-independent | candidate value | 1.165 | 1.027 | 1.605 | 0.7139 | 3.27 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | eta_T_over_dw | development | candidate value | 0.0007207 | 0.0005915 | 0.0007453 | 0.0004663 | 0.001142 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | DeltaT_over_dw_on_max | development | candidate value | 0.0007207 | 0.0005915 | 0.0007453 | 0.0004663 | 0.001142 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 4 | 60 | DeltaT_over_dw_off_max | development | candidate value | 0.0002679 | 0.000237 | 0.0003269 | 0.0002186 | 0.0003719 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.05595 | -0.06533 | -0.04679 | -0.07701 | -0.04332 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.05595 | 0.04679 | 0.06533 | 0.04332 | 0.07701 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.0551 | -0.0653 | -0.04632 | -0.0769 | -0.04222 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.0551 | 0.04632 | 0.0653 | 0.04222 | 0.0769 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.01711 | 0.01559 | 0.01975 | 0.01452 | 0.02156 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | stage2_tail_mean | tier-independent | candidate value | 0.551 | 0.5207 | 0.5795 | 0.4668 | 0.675 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | stage2_tail_max | tier-independent | candidate value | 2.065 | 1.944 | 2.196 | 1.802 | 2.235 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.009446 | 0.008927 | 0.009934 | 0.008003 | 0.01157 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.0354 | 0.03332 | 0.03765 | 0.03089 | 0.03831 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | stage2_sym_err_max | tier-independent | candidate value | 1.453 | 1.01 | 1.862 | 0.724 | 2.08 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | eta_T_over_dw | development | candidate value | 0.0006428 | 0.0004095 | 0.0007601 | 0.0003168 | 0.00097 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | DeltaT_over_dw_on_max | development | candidate value | 0.0006428 | 0.0004095 | 0.0007601 | 0.0003168 | 0.00097 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 8 | 60 | DeltaT_over_dw_off_max | development | candidate value | 0.0002787 | 0.0002651 | 0.00031 | 0.000221 | 0.0003558 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.05869 | -0.06625 | -0.05241 | -0.07993 | -0.0475 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.05869 | 0.05241 | 0.06625 | 0.0475 | 0.07993 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.05825 | -0.06582 | -0.05218 | -0.07809 | -0.04663 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.05825 | 0.05218 | 0.06582 | 0.04663 | 0.07809 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.01635 | 0.01419 | 0.01775 | 0.01334 | 0.02085 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | stage2_tail_mean | tier-independent | candidate value | 0.5782 | 0.5242 | 0.6077 | 0.493 | 0.7135 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | stage2_tail_max | tier-independent | candidate value | 2.029 | 1.979 | 2.227 | 1.837 | 2.397 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.009911 | 0.008987 | 0.01042 | 0.008451 | 0.01223 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03479 | 0.03393 | 0.03818 | 0.03149 | 0.04109 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | stage2_sym_err_max | tier-independent | candidate value | 1.01 | 0.8361 | 1.664 | 0.7145 | 2.072 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | eta_T_over_dw | development | candidate value | 0.0005634 | 0.000454 | 0.0007205 | 0.000371 | 0.001045 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | DeltaT_over_dw_on_max | development | candidate value | 0.0005634 | 0.000454 | 0.0007205 | 0.000371 | 0.001045 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | ext | ext | 16 | 60 | DeltaT_over_dw_off_max | development | candidate value | 0.0002893 | 0.0002784 | 0.0003089 | 0.000241 | 0.0003909 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.06835 | -0.0709 | -0.05722 | -0.1286 | -0.03741 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.06835 | 0.05722 | 0.0709 | 0.03741 | 0.1286 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.06826 | -0.07074 | -0.05652 | -0.1286 | -0.0368 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.06826 | 0.05652 | 0.07074 | 0.0368 | 0.1286 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.02605 | 0.0221 | 0.02871 | 0.01422 | 0.05976 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | stage2_tail_mean | tier-independent | candidate value | 0.515 | 0.4559 | 0.5639 | 0.3528 | 0.8831 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | stage2_tail_max | tier-independent | candidate value | 2.22 | 2.131 | 2.597 | 1.656 | 4.907 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.007358 | 0.006513 | 0.008055 | 0.00504 | 0.01262 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03172 | 0.03044 | 0.03711 | 0.02365 | 0.07009 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | stage2_sym_err_max | tier-independent | candidate value | 2.638 | 1.742 | 3.529 | 1.447 | 5.473 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | eta_T_over_dw | development | candidate value | 0.001469 | 0.001119 | 0.002431 | 0.0007199 | 0.006971 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | DeltaT_over_dw_on_max | development | candidate value | 0.001469 | 0.001119 | 0.002431 | 0.0007199 | 0.006971 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 50 | DeltaT_over_dw_off_max | development | candidate value | 0.0003395 | 0.000264 | 0.000386 | 0.0001463 | 0.001545 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.06294 | -0.08174 | -0.05453 | -0.09494 | -0.04077 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.06294 | 0.05453 | 0.08174 | 0.04077 | 0.09494 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.06247 | -0.08075 | -0.05435 | -0.09311 | -0.04065 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.06247 | 0.05435 | 0.08075 | 0.04065 | 0.09311 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.0201 | 0.01888 | 0.02321 | 0.01642 | 0.02746 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | stage2_tail_mean | tier-independent | candidate value | 0.5552 | 0.4807 | 0.5654 | 0.3847 | 0.889 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | stage2_tail_max | tier-independent | candidate value | 2.334 | 2.252 | 2.403 | 1.908 | 4.379 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.007931 | 0.006867 | 0.008077 | 0.005495 | 0.0127 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03334 | 0.03218 | 0.03433 | 0.02726 | 0.06256 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | stage2_sym_err_max | tier-independent | candidate value | 1.809 | 1.628 | 2.328 | 0.6365 | 2.425 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | eta_T_over_dw | development | candidate value | 0.001168 | 0.001018 | 0.001855 | 0.0004987 | 0.002563 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | DeltaT_over_dw_on_max | development | candidate value | 0.001168 | 0.001018 | 0.001855 | 0.0004987 | 0.002563 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 50 | DeltaT_over_dw_off_max | development | candidate value | 0.0003489 | 0.0002959 | 0.0003808 | 0.0002174 | 0.001187 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.07022 | -0.07666 | -0.05445 | -0.107 | -0.03448 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.07022 | 0.05445 | 0.07666 | 0.03448 | 0.107 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.06972 | -0.07593 | -0.05426 | -0.1069 | -0.03396 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.06972 | 0.05426 | 0.07593 | 0.03396 | 0.1069 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.01917 | 0.01776 | 0.02075 | 0.01438 | 0.02906 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | stage2_tail_mean | tier-independent | candidate value | 0.5619 | 0.4836 | 0.5856 | 0.4059 | 0.861 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | stage2_tail_max | tier-independent | candidate value | 2.34 | 2.228 | 2.49 | 2.108 | 4.282 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.008027 | 0.006908 | 0.008366 | 0.005799 | 0.0123 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03343 | 0.03183 | 0.03557 | 0.03011 | 0.06118 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | stage2_sym_err_max | tier-independent | candidate value | 1.444 | 1.278 | 1.87 | 0.6973 | 2.318 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | eta_T_over_dw | development | candidate value | 0.001285 | 0.001068 | 0.001526 | 0.0005877 | 0.00304 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | DeltaT_over_dw_on_max | development | candidate value | 0.001285 | 0.001068 | 0.001526 | 0.0005877 | 0.00304 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 50 | DeltaT_over_dw_off_max | development | candidate value | 0.0003247 | 0.0003087 | 0.0004052 | 0.0002631 | 0.001091 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.06003 | -0.07362 | -0.04034 | -0.07572 | 0.003541 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.06003 | 0.04034 | 0.07362 | 0.003541 | 0.07572 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.05585 | -0.07326 | -0.03969 | -0.07572 | 0.003559 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.05585 | 0.03969 | 0.07326 | 0.003559 | 0.07572 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.02389 | 0.02089 | 0.03157 | 0.01864 | 0.03376 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | stage2_tail_mean | tier-independent | candidate value | 0.5312 | 0.501 | 0.5663 | 0.4143 | 0.6374 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | stage2_tail_max | tier-independent | candidate value | 1.901 | 1.835 | 2.109 | 1.708 | 2.556 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.009106 | 0.008588 | 0.009708 | 0.007102 | 0.01093 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03259 | 0.03146 | 0.03616 | 0.02927 | 0.04382 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | stage2_sym_err_max | tier-independent | candidate value | 2.39 | 1.517 | 2.535 | 0.8313 | 3.963 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | eta_T_over_dw | development | candidate value | 0.0009012 | 0.0007233 | 0.0009584 | 0.0003882 | 0.002156 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | DeltaT_over_dw_on_max | development | candidate value | 0.0009012 | 0.0007233 | 0.0009584 | 0.0003882 | 0.002156 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 1 | 60 | DeltaT_over_dw_off_max | development | candidate value | 0.0002439 | 0.0002221 | 0.0002957 | 0.0001492 | 0.000465 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.06243 | -0.06688 | -0.05679 | -0.08176 | -0.03148 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.06243 | 0.05679 | 0.06688 | 0.03148 | 0.08176 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.06217 | -0.06654 | -0.05594 | -0.08172 | -0.03125 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.06217 | 0.05594 | 0.06654 | 0.03125 | 0.08172 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.0179 | 0.0173 | 0.01925 | 0.01626 | 0.02177 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | stage2_tail_mean | tier-independent | candidate value | 0.5404 | 0.5093 | 0.5688 | 0.4516 | 0.6459 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | stage2_tail_max | tier-independent | candidate value | 2.064 | 1.874 | 2.189 | 1.749 | 2.282 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.009264 | 0.008731 | 0.00975 | 0.007741 | 0.01107 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03538 | 0.03212 | 0.03753 | 0.02999 | 0.03912 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | stage2_sym_err_max | tier-independent | candidate value | 1.165 | 1.027 | 1.605 | 0.7139 | 3.27 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | eta_T_over_dw | development | candidate value | 0.0007207 | 0.0005915 | 0.0007453 | 0.0004663 | 0.001142 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | DeltaT_over_dw_on_max | development | candidate value | 0.0007207 | 0.0005915 | 0.0007453 | 0.0004663 | 0.001142 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 4 | 60 | DeltaT_over_dw_off_max | development | candidate value | 0.0002679 | 0.000237 | 0.0003269 | 0.0002186 | 0.0003719 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.05595 | -0.06533 | -0.04679 | -0.07701 | -0.04332 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.05595 | 0.04679 | 0.06533 | 0.04332 | 0.07701 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.0551 | -0.0653 | -0.04632 | -0.0769 | -0.04222 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.0551 | 0.04632 | 0.0653 | 0.04222 | 0.0769 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.01711 | 0.01559 | 0.01975 | 0.01452 | 0.02156 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | stage2_tail_mean | tier-independent | candidate value | 0.551 | 0.5207 | 0.5795 | 0.4668 | 0.675 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | stage2_tail_max | tier-independent | candidate value | 2.065 | 1.944 | 2.196 | 1.802 | 2.235 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.009446 | 0.008927 | 0.009934 | 0.008003 | 0.01157 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.0354 | 0.03332 | 0.03765 | 0.03089 | 0.03831 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | stage2_sym_err_max | tier-independent | candidate value | 1.453 | 1.01 | 1.862 | 0.724 | 2.08 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | eta_T_over_dw | development | candidate value | 0.0006428 | 0.0004095 | 0.0007601 | 0.0003168 | 0.00097 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | DeltaT_over_dw_on_max | development | candidate value | 0.0006428 | 0.0004095 | 0.0007601 | 0.0003168 | 0.00097 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | constant | 8 | 60 | DeltaT_over_dw_off_max | development | candidate value | 0.0002787 | 0.0002651 | 0.00031 | 0.000221 | 0.0003558 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.0625 | -0.07044 | -0.05518 | -0.1222 | -0.02357 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.0625 | 0.05518 | 0.07044 | 0.02357 | 0.1222 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.06243 | -0.07013 | -0.05456 | -0.1221 | -0.02246 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.06243 | 0.05456 | 0.07013 | 0.02246 | 0.1221 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.02007 | 0.01925 | 0.02384 | 0.01502 | 0.03966 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | stage2_tail_mean | tier-independent | candidate value | 0.5622 | 0.4938 | 0.632 | 0.3466 | 0.8309 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | stage2_tail_max | tier-independent | candidate value | 2.584 | 2.424 | 2.744 | 2.218 | 4.28 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.008032 | 0.007055 | 0.009028 | 0.004951 | 0.01187 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03692 | 0.03463 | 0.03919 | 0.03168 | 0.06114 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | stage2_sym_err_max | tier-independent | candidate value | 2.265 | 1.793 | 2.762 | 1.266 | 4.862 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | eta_T_over_dw | development | candidate value | 0.001082 | 0.0009888 | 0.001519 | 0.0007256 | 0.004014 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | DeltaT_over_dw_on_max | development | candidate value | 0.001082 | 0.0009888 | 0.001519 | 0.0007256 | 0.004014 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 50 | DeltaT_over_dw_off_max | development | candidate value | 0.0004454 | 0.0003686 | 0.000494 | 0.0002605 | 0.001155 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.06911 | -0.07868 | -0.05234 | -0.1116 | -0.03757 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.06911 | 0.05234 | 0.07868 | 0.03757 | 0.1116 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.06831 | -0.07868 | -0.05189 | -0.1115 | -0.03662 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.06831 | 0.05189 | 0.07868 | 0.03662 | 0.1115 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.01735 | 0.01635 | 0.02154 | 0.01472 | 0.03071 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | stage2_tail_mean | tier-independent | candidate value | 0.5905 | 0.4748 | 0.6363 | 0.3675 | 0.8077 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | stage2_tail_max | tier-independent | candidate value | 2.616 | 2.442 | 2.706 | 2.199 | 4.453 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.008436 | 0.006783 | 0.009089 | 0.00525 | 0.01154 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03737 | 0.03489 | 0.03866 | 0.03141 | 0.06362 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | stage2_sym_err_max | tier-independent | candidate value | 1.622 | 1.433 | 1.883 | 0.8464 | 4.251 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | eta_T_over_dw | development | candidate value | 0.001036 | 0.0006597 | 0.001521 | 0.0006195 | 0.003426 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | DeltaT_over_dw_on_max | development | candidate value | 0.001036 | 0.0006597 | 0.001521 | 0.0006195 | 0.003426 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 50 | DeltaT_over_dw_off_max | development | candidate value | 0.0004245 | 0.0003976 | 0.0005165 | 0.0003052 | 0.001202 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.06673 | -0.07379 | -0.04943 | -0.1212 | -0.02928 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.06673 | 0.04943 | 0.07379 | 0.02928 | 0.1212 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.06634 | -0.07369 | -0.04799 | -0.1205 | -0.02867 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.06634 | 0.04799 | 0.07369 | 0.02867 | 0.1205 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.01713 | 0.0159 | 0.01899 | 0.01416 | 0.03369 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | stage2_tail_mean | tier-independent | candidate value | 0.5991 | 0.4729 | 0.6363 | 0.3767 | 0.8239 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | stage2_tail_max | tier-independent | candidate value | 2.501 | 2.413 | 2.599 | 2.18 | 4.496 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.008559 | 0.006755 | 0.009089 | 0.005382 | 0.01177 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03572 | 0.03447 | 0.03713 | 0.03114 | 0.06423 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | stage2_sym_err_max | tier-independent | candidate value | 1.583 | 1.349 | 2.153 | 1.093 | 3.278 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | eta_T_over_dw | development | candidate value | 0.001126 | 0.0007642 | 0.001284 | 0.0004457 | 0.004195 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | DeltaT_over_dw_on_max | development | candidate value | 0.001126 | 0.0007642 | 0.001284 | 0.0004457 | 0.004195 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 50 | DeltaT_over_dw_off_max | development | candidate value | 0.0003807 | 0.0003561 | 0.0004476 | 0.0003085 | 0.001181 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.04971 | -0.067 | -0.0439 | -0.09174 | -0.02097 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.04971 | 0.0439 | 0.067 | 0.02097 | 0.09174 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.04943 | -0.06686 | -0.04338 | -0.09052 | -0.02071 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.04943 | 0.04338 | 0.06686 | 0.02071 | 0.09052 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.01982 | 0.01729 | 0.02362 | 0.01333 | 0.02651 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | stage2_tail_mean | tier-independent | candidate value | 0.5479 | 0.5082 | 0.6074 | 0.4856 | 0.7317 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | stage2_tail_max | tier-independent | candidate value | 2.111 | 2.06 | 2.228 | 1.879 | 2.507 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.009393 | 0.008712 | 0.01041 | 0.008325 | 0.01254 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03619 | 0.03531 | 0.0382 | 0.03222 | 0.04297 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | stage2_sym_err_max | tier-independent | candidate value | 1.643 | 1.201 | 2.188 | 0.9295 | 3.357 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | eta_T_over_dw | development | candidate value | 0.0008081 | 0.0004732 | 0.0009643 | 0.0003602 | 0.001497 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | DeltaT_over_dw_on_max | development | candidate value | 0.0008081 | 0.0004732 | 0.0009643 | 0.0003602 | 0.001497 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 1 | 60 | DeltaT_over_dw_off_max | development | candidate value | 0.0003076 | 0.0002597 | 0.0003341 | 0.0002273 | 0.0003812 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.05929 | -0.07342 | -0.04982 | -0.08114 | -0.0385 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.05929 | 0.04982 | 0.07342 | 0.0385 | 0.08114 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.059 | -0.07275 | -0.0498 | -0.08043 | -0.03799 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.059 | 0.0498 | 0.07275 | 0.03799 | 0.08043 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.01774 | 0.01716 | 0.01848 | 0.01363 | 0.02059 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | stage2_tail_mean | tier-independent | candidate value | 0.5726 | 0.496 | 0.6076 | 0.477 | 0.7347 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | stage2_tail_max | tier-independent | candidate value | 2.122 | 2.05 | 2.216 | 1.856 | 2.359 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.009815 | 0.008503 | 0.01042 | 0.008178 | 0.01259 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03637 | 0.03514 | 0.038 | 0.03181 | 0.04044 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | stage2_sym_err_max | tier-independent | candidate value | 1.455 | 1.12 | 1.794 | 0.7329 | 1.918 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | eta_T_over_dw | development | candidate value | 0.0006526 | 0.0004425 | 0.0008819 | 0.000354 | 0.001131 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | DeltaT_over_dw_on_max | development | candidate value | 0.0006526 | 0.0004425 | 0.0008819 | 0.000354 | 0.001131 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 4 | 60 | DeltaT_over_dw_off_max | development | candidate value | 0.0002998 | 0.0002739 | 0.0003275 | 0.0002168 | 0.0003856 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | stage2_peak_rel_err_signed | tier-independent | candidate value | -0.05884 | -0.06891 | -0.0482 | -0.07207 | -0.0403 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | stage2_peak_rel_err_abs | tier-independent | candidate value | 0.05884 | 0.0482 | 0.06891 | 0.0403 | 0.07207 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | stage2_peak_locfree_rel_err | tier-independent | candidate value | -0.05858 | -0.06799 | -0.04811 | -0.07081 | -0.03961 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | candidate value | 0.05858 | 0.04811 | 0.06799 | 0.03961 | 0.07081 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | candidate value | 0.01627 | 0.01549 | 0.01794 | 0.01388 | 0.02282 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | stage2_tail_mean | tier-independent | candidate value | 0.5719 | 0.492 | 0.6122 | 0.4818 | 0.7431 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | stage2_tail_max | tier-independent | candidate value | 2.122 | 2.012 | 2.218 | 1.842 | 2.357 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | candidate value | 0.009804 | 0.008435 | 0.0105 | 0.00826 | 0.01274 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | stage2_tail_max_over_g2_0 | tier-independent | candidate value | 0.03638 | 0.03449 | 0.03803 | 0.03158 | 0.04041 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | stage2_sym_err_max | tier-independent | candidate value | 1.557 | 1.001 | 1.832 | 0.5704 | 2.319 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | eta_T_over_dw | development | candidate value | 0.0005716 | 0.0004112 | 0.0007766 | 0.00037 | 0.0008496 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | DeltaT_over_dw_on_max | development | candidate value | 0.0005716 | 0.0004112 | 0.0007766 | 0.00037 | 0.0008496 | 10 |  |  |  |  |  |  |  |  |  |
| candidates | 2a | decay | 8 | 60 | DeltaT_over_dw_off_max | development | candidate value | 0.0003144 | 0.0002832 | 0.0003196 | 0.0001973 | 0.0003629 | 10 |  |  |  |  |  |  |  |  |  |
| paired_K_vs_K1 | ext | ext | 4 | 50 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | -0.002328 |  |  | -0.02258 | 0.07018 |  | 10 | 0.001827 | 6 | 4 | 0 | -0.01162 | 0.0191 |  | no preferred direction |
| paired_K_vs_K1 | ext | ext | 4 | 50 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.002328 |  |  | -0.07018 | 0.02258 |  | 10 | -0.001827 | 4 | 6 | 0 | -0.01962 | 0.01169 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 50 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | -0.003148 |  |  | -0.02199 | 0.07069 |  | 10 | 0.001996 | 6 | 4 | 0 | -0.01141 | 0.01964 |  | no preferred direction |
| paired_K_vs_K1 | ext | ext | 4 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.003148 |  |  | -0.07069 | 0.02199 |  | 10 | -0.001996 | 4 | 6 | 0 | -0.01999 | 0.0114 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.002994 |  |  | -0.04095 | 0.002194 |  | 10 | -0.006807 | 8 | 2 | 0 | -0.01513 | -0.001532 | 8 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 50 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | 0.01792 |  |  | -0.03456 | 0.05709 |  | 10 | 0.02052 | 1 | 9 | 0 | 0.004275 | 0.03488 | 1 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 50 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | 0.114 |  |  | -0.6561 | 0.3592 |  | 10 | -0.05779 | 4 | 6 | 0 | -0.2691 | 0.1441 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.0002559 |  |  | -0.0004937 | 0.0008156 |  | 10 | 0.0002931 | 1 | 9 | 0 | 6.401e-05 | 0.0004936 | 1 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 50 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.001628 |  |  | -0.009373 | 0.005132 |  | 10 | -0.0008255 | 4 | 6 | 0 | -0.003921 | 0.002019 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 50 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -0.4989 |  |  | -3.501 | 0.956 |  | 10 | -0.9515 | 7 | 3 | 0 | -1.886 | -0.1045 | 7 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 50 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -0.0003245 |  |  | -0.006268 | 0.001326 |  | 10 | -0.0008027 | 8 | 2 | 0 | -0.002218 | 0.0002004 | 8 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 50 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -0.0003245 |  |  | -0.006268 | 0.001326 |  | 10 | -0.0008027 | 8 | 2 | 0 | -0.002207 | 0.0001858 | 8 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 50 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | 2.497e-05 |  |  | -0.0003581 | 0.0001178 |  | 10 | -2.911e-05 | 3 | 7 | 0 | -0.0001233 | 4.7e-05 | 3 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 50 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | -0.002852 |  |  | -0.03567 | 0.07517 |  | 10 | 0.0005914 | 7 | 3 | 0 | -0.01573 | 0.02057 |  | no preferred direction |
| paired_K_vs_K1 | ext | ext | 8 | 50 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.002852 |  |  | -0.07517 | 0.03567 |  | 10 | -0.0005914 | 3 | 7 | 0 | -0.02056 | 0.01583 | 3 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 50 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | -0.003215 |  |  | -0.03561 | 0.07542 |  | 10 | 0.0005974 | 7 | 3 | 0 | -0.01589 | 0.02071 |  | no preferred direction |
| paired_K_vs_K1 | ext | ext | 8 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.003215 |  |  | -0.07542 | 0.03561 |  | 10 | -0.0005974 | 3 | 7 | 0 | -0.02092 | 0.01611 | 3 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.004485 |  |  | -0.04538 | 0.002963 |  | 10 | -0.008116 | 8 | 2 | 0 | -0.01735 | -0.002112 | 8 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 50 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | 0.03522 |  |  | -0.02669 | 0.06835 |  | 10 | 0.02771 | 2 | 8 | 0 | 0.00715 | 0.04643 | 2 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 50 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | 0.1532 |  |  | -0.7925 | 0.8676 |  | 10 | 0.02424 | 4 | 6 | 0 | -0.269 | 0.3244 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.0005031 |  |  | -0.0003813 | 0.0009765 |  | 10 | 0.0003959 | 2 | 8 | 0 | 9.5e-05 | 0.0006569 | 2 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 50 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.002189 |  |  | -0.01132 | 0.01239 |  | 10 | 0.0003463 | 4 | 6 | 0 | -0.003954 | 0.004513 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 50 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -0.7758 |  |  | -3.155 | -0.1652 |  | 10 | -1.223 | 10 | 0 | 0 | -1.926 | -0.6146 | 10 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 50 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -0.0002704 |  |  | -0.006383 | 0.001512 |  | 10 | -0.0007159 | 6 | 4 | 0 | -0.002172 | 0.0003157 | 6 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 50 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -0.0002704 |  |  | -0.006383 | 0.001512 |  | 10 | -0.0007159 | 6 | 4 | 0 | -0.002159 | 0.000296 | 6 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 50 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | 1.662e-05 |  |  | -0.0004535 | 0.0003003 |  | 10 | -1.501e-05 | 4 | 6 | 0 | -0.0001469 | 0.0001033 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 50 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | -0.001551 |  |  | -0.03081 | 0.06328 |  | 10 | 0.001327 | 6 | 4 | 0 | -0.01196 | 0.01789 |  | no preferred direction |
| paired_K_vs_K1 | ext | ext | 16 | 50 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.001551 |  |  | -0.06328 | 0.03081 |  | 10 | -0.001327 | 4 | 6 | 0 | -0.0181 | 0.01207 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 50 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | -0.001994 |  |  | -0.031 | 0.06328 |  | 10 | 0.001224 | 6 | 4 | 0 | -0.01221 | 0.01766 |  | no preferred direction |
| paired_K_vs_K1 | ext | ext | 16 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.001994 |  |  | -0.06328 | 0.031 |  | 10 | -0.001224 | 4 | 6 | 0 | -0.01784 | 0.01252 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.00787 |  |  | -0.0439 | 0.001576 |  | 10 | -0.009616 | 8 | 2 | 0 | -0.0184 | -0.00357 | 8 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 50 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | 0.05823 |  |  | -0.003303 | 0.1007 |  | 10 | 0.05028 | 2 | 8 | 0 | 0.03029 | 0.06848 | 2 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 50 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | 0.2154 |  |  | -0.6918 | 0.7891 |  | 10 | 0.09313 | 4 | 6 | 0 | -0.1789 | 0.3619 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.0008319 |  |  | -4.718e-05 | 0.001439 |  | 10 | 0.0007182 | 2 | 8 | 0 | 0.0004276 | 0.0009738 | 2 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 50 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.003077 |  |  | -0.009883 | 0.01127 |  | 10 | 0.00133 | 4 | 6 | 0 | -0.002665 | 0.005124 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 50 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -1.145 |  |  | -2.828 | 0.3777 |  | 10 | -1.249 | 9 | 1 | 0 | -1.893 | -0.6272 | 9 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 50 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -0.0007141 |  |  | -0.00599 | 0.001606 |  | 10 | -0.0009119 | 7 | 3 | 0 | -0.002246 | 9.297e-05 | 7 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 50 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -0.0007141 |  |  | -0.00599 | 0.001606 |  | 10 | -0.0009119 | 7 | 3 | 0 | -0.002238 | 7.063e-05 | 7 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 50 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | 4.474e-05 |  |  | -0.0003851 | 0.0002729 |  | 10 | 6.427e-06 | 4 | 6 | 0 | -0.0001124 | 0.0001123 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 60 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | -0.007549 |  |  | -0.03691 | 0.01677 |  | 10 | -0.008899 | 6 | 4 | 0 | -0.02033 | 0.002442 |  | no preferred direction |
| paired_K_vs_K1 | ext | ext | 4 | 60 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.007549 |  |  | -0.01677 | 0.03691 |  | 10 | 0.008191 | 4 | 6 | 0 | -0.002239 | 0.01907 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 60 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | -0.008282 |  |  | -0.03806 | 0.01689 |  | 10 | -0.009943 | 6 | 4 | 0 | -0.02094 | 0.0008526 |  | no preferred direction |
| paired_K_vs_K1 | ext | ext | 4 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.008282 |  |  | -0.01689 | 0.03806 |  | 10 | 0.009231 | 4 | 6 | 0 | -0.0009105 | 0.01987 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.005099 |  |  | -0.0175 | 0.0009559 |  | 10 | -0.007265 | 9 | 1 | 0 | -0.01091 | -0.003869 | 9 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 60 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | -0.005074 |  |  | -0.02409 | 0.08279 |  | 10 | 0.007186 | 5 | 5 | 0 | -0.01176 | 0.0297 | 5 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 60 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | -0.001586 |  |  | -0.2737 | 0.3234 |  | 10 | 0.03276 | 5 | 5 | 0 | -0.102 | 0.1706 | 5 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -8.699e-05 |  |  | -0.000413 | 0.001419 |  | 10 | 0.0001232 | 5 | 5 | 0 | -0.0002041 | 0.0005145 | 5 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 60 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -2.718e-05 |  |  | -0.004692 | 0.005545 |  | 10 | 0.0005615 | 5 | 5 | 0 | -0.001773 | 0.002856 | 5 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 60 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -0.8878 |  |  | -1.531 | 0.305 |  | 10 | -0.8025 | 9 | 1 | 0 | -1.131 | -0.4444 | 9 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 60 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -0.0001711 |  |  | -0.001406 | 0.0002916 |  | 10 | -0.0002042 | 7 | 3 | 0 | -0.0005195 | 3.155e-05 | 7 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 60 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -0.0001711 |  |  | -0.001406 | 0.0002916 |  | 10 | -0.0002042 | 7 | 3 | 0 | -0.0005113 | 2.807e-05 | 7 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 4 | 60 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | 5.78e-06 |  |  | -9.302e-05 | 0.0001587 |  | 10 | 1.625e-05 | 5 | 5 | 0 | -2.95e-05 | 6.324e-05 | 5 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 60 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | 0.000478 |  |  | -0.05259 | 0.02819 |  | 10 | -0.004727 | 5 | 5 | 0 | -0.01853 | 0.007403 |  | no preferred direction |
| paired_K_vs_K1 | ext | ext | 8 | 60 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | -0.000478 |  |  | -0.02819 | 0.04551 |  | 10 | 0.004018 | 5 | 5 | 0 | -0.007502 | 0.0167 | 5 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 60 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | -0.003021 |  |  | -0.05254 | 0.02879 |  | 10 | -0.005514 | 6 | 4 | 0 | -0.01916 | 0.007025 |  | no preferred direction |
| paired_K_vs_K1 | ext | ext | 8 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.003021 |  |  | -0.02879 | 0.04542 |  | 10 | 0.004803 | 4 | 6 | 0 | -0.006952 | 0.01722 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.005749 |  |  | -0.01924 | 0.002884 |  | 10 | -0.007904 | 9 | 1 | 0 | -0.01197 | -0.004031 | 9 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 60 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | 0.01456 |  |  | -0.02386 | 0.09088 |  | 10 | 0.01972 | 4 | 6 | 0 | -0.0001094 | 0.04174 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 60 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | 0.07841 |  |  | -0.3239 | 0.3099 |  | 10 | 0.05271 | 4 | 6 | 0 | -0.07806 | 0.1731 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.0002495 |  |  | -0.0004091 | 0.001558 |  | 10 | 0.000338 | 4 | 6 | 0 | 4.411e-06 | 0.0007168 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 60 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.001344 |  |  | -0.005552 | 0.005312 |  | 10 | 0.0009036 | 4 | 6 | 0 | -0.001304 | 0.002958 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 60 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -0.9645 |  |  | -1.936 | 1.248 |  | 10 | -0.761 | 8 | 2 | 0 | -1.268 | -0.1683 | 8 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 60 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -8.237e-05 |  |  | -0.001763 | 6.868e-05 |  | 10 | -0.0003072 | 9 | 1 | 0 | -0.00067 | -6.843e-05 | 9 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 60 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -8.237e-05 |  |  | -0.001763 | 6.868e-05 |  | 10 | -0.0003072 | 9 | 1 | 0 | -0.0006631 | -7.063e-05 | 9 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 8 | 60 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | 3.111e-05 |  |  | -0.0001091 | 0.000134 |  | 10 | 2.174e-05 | 3 | 7 | 0 | -2.062e-05 | 6.224e-05 | 3 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 60 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | -0.004511 |  |  | -0.05545 | 0.02673 |  | 10 | -0.007906 | 6 | 4 | 0 | -0.02216 | 0.004843 |  | no preferred direction |
| paired_K_vs_K1 | ext | ext | 16 | 60 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.004511 |  |  | -0.02673 | 0.04837 |  | 10 | 0.007198 | 4 | 6 | 0 | -0.00521 | 0.02049 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 60 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | -0.007241 |  |  | -0.05538 | 0.0276 |  | 10 | -0.008754 | 6 | 4 | 0 | -0.02304 | 0.004638 |  | no preferred direction |
| paired_K_vs_K1 | ext | ext | 16 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.007241 |  |  | -0.0276 | 0.04826 |  | 10 | 0.008042 | 4 | 6 | 0 | -0.004634 | 0.02095 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.007772 |  |  | -0.01972 | -0.0007829 |  | 10 | -0.009343 | 10 | 0 | 0 | -0.01301 | -0.005947 | 10 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 60 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | 0.04384 |  |  | -0.01701 | 0.1319 |  | 10 | 0.04203 | 3 | 7 | 0 | 0.01478 | 0.07106 | 3 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 60 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | 0.115 |  |  | -0.3042 | 0.4772 |  | 10 | 0.09236 | 4 | 6 | 0 | -0.06068 | 0.248 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.0007516 |  |  | -0.0002915 | 0.002261 |  | 10 | 0.0007205 | 3 | 7 | 0 | 0.0002559 | 0.001217 | 3 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 60 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.001971 |  |  | -0.005214 | 0.00818 |  | 10 | 0.001583 | 4 | 6 | 0 | -0.001064 | 0.004269 | 4 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 60 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -1.282 |  |  | -1.891 | 1.038 |  | 10 | -0.9912 | 9 | 1 | 0 | -1.473 | -0.4131 | 9 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 60 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -0.0001704 |  |  | -0.001715 | 0.0002021 |  | 10 | -0.000297 | 7 | 3 | 0 | -0.0006599 | -3.558e-05 | 7 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 60 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -0.0001704 |  |  | -0.001715 | 0.0002021 |  | 10 | -0.000297 | 7 | 3 | 0 | -0.0006538 | -3.622e-05 | 7 | diff < 0 |
| paired_K_vs_K1 | ext | ext | 16 | 60 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | 4.83e-05 |  |  | -0.0001166 | 0.0001425 |  | 10 | 3.492e-05 | 3 | 7 | 0 | -1.247e-05 | 7.929e-05 | 3 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | -0.002328 |  |  | -0.02258 | 0.07018 |  | 10 | 0.001827 | 6 | 4 | 0 | -0.01162 | 0.0191 |  | no preferred direction |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.002328 |  |  | -0.07018 | 0.02258 |  | 10 | -0.001827 | 4 | 6 | 0 | -0.01962 | 0.01169 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | -0.003148 |  |  | -0.02199 | 0.07069 |  | 10 | 0.001996 | 6 | 4 | 0 | -0.01141 | 0.01964 |  | no preferred direction |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.003148 |  |  | -0.07069 | 0.02199 |  | 10 | -0.001996 | 4 | 6 | 0 | -0.01999 | 0.0114 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.002994 |  |  | -0.04095 | 0.002194 |  | 10 | -0.006807 | 8 | 2 | 0 | -0.01513 | -0.001532 | 8 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | 0.01792 |  |  | -0.03456 | 0.05709 |  | 10 | 0.02052 | 1 | 9 | 0 | 0.004275 | 0.03488 | 1 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | 0.114 |  |  | -0.6561 | 0.3592 |  | 10 | -0.05779 | 4 | 6 | 0 | -0.2691 | 0.1441 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.0002559 |  |  | -0.0004937 | 0.0008156 |  | 10 | 0.0002931 | 1 | 9 | 0 | 6.401e-05 | 0.0004936 | 1 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.001628 |  |  | -0.009373 | 0.005132 |  | 10 | -0.0008255 | 4 | 6 | 0 | -0.003921 | 0.002019 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -0.4989 |  |  | -3.501 | 0.956 |  | 10 | -0.9515 | 7 | 3 | 0 | -1.886 | -0.1045 | 7 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -0.0003245 |  |  | -0.006268 | 0.001326 |  | 10 | -0.0008027 | 8 | 2 | 0 | -0.002218 | 0.0002004 | 8 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -0.0003245 |  |  | -0.006268 | 0.001326 |  | 10 | -0.0008027 | 8 | 2 | 0 | -0.002207 | 0.0001858 | 8 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 50 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | 2.497e-05 |  |  | -0.0003581 | 0.0001178 |  | 10 | -2.911e-05 | 3 | 7 | 0 | -0.0001233 | 4.7e-05 | 3 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | -0.002852 |  |  | -0.03567 | 0.07517 |  | 10 | 0.0005914 | 7 | 3 | 0 | -0.01573 | 0.02057 |  | no preferred direction |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.002852 |  |  | -0.07517 | 0.03567 |  | 10 | -0.0005914 | 3 | 7 | 0 | -0.02056 | 0.01583 | 3 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | -0.003215 |  |  | -0.03561 | 0.07542 |  | 10 | 0.0005974 | 7 | 3 | 0 | -0.01589 | 0.02071 |  | no preferred direction |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.003215 |  |  | -0.07542 | 0.03561 |  | 10 | -0.0005974 | 3 | 7 | 0 | -0.02092 | 0.01611 | 3 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.004485 |  |  | -0.04538 | 0.002963 |  | 10 | -0.008116 | 8 | 2 | 0 | -0.01735 | -0.002112 | 8 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | 0.03522 |  |  | -0.02669 | 0.06835 |  | 10 | 0.02771 | 2 | 8 | 0 | 0.00715 | 0.04643 | 2 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | 0.1532 |  |  | -0.7925 | 0.8676 |  | 10 | 0.02424 | 4 | 6 | 0 | -0.269 | 0.3244 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.0005031 |  |  | -0.0003813 | 0.0009765 |  | 10 | 0.0003959 | 2 | 8 | 0 | 9.5e-05 | 0.0006569 | 2 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.002189 |  |  | -0.01132 | 0.01239 |  | 10 | 0.0003463 | 4 | 6 | 0 | -0.003954 | 0.004513 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -0.7758 |  |  | -3.155 | -0.1652 |  | 10 | -1.223 | 10 | 0 | 0 | -1.926 | -0.6146 | 10 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -0.0002704 |  |  | -0.006383 | 0.001512 |  | 10 | -0.0007159 | 6 | 4 | 0 | -0.002172 | 0.0003157 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -0.0002704 |  |  | -0.006383 | 0.001512 |  | 10 | -0.0007159 | 6 | 4 | 0 | -0.002159 | 0.000296 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 50 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | 1.662e-05 |  |  | -0.0004535 | 0.0003003 |  | 10 | -1.501e-05 | 4 | 6 | 0 | -0.0001469 | 0.0001033 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | -0.007549 |  |  | -0.03691 | 0.01677 |  | 10 | -0.008899 | 6 | 4 | 0 | -0.02033 | 0.002442 |  | no preferred direction |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.007549 |  |  | -0.01677 | 0.03691 |  | 10 | 0.008191 | 4 | 6 | 0 | -0.002239 | 0.01907 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | -0.008282 |  |  | -0.03806 | 0.01689 |  | 10 | -0.009943 | 6 | 4 | 0 | -0.02094 | 0.0008526 |  | no preferred direction |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.008282 |  |  | -0.01689 | 0.03806 |  | 10 | 0.009231 | 4 | 6 | 0 | -0.0009105 | 0.01987 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.005099 |  |  | -0.0175 | 0.0009559 |  | 10 | -0.007265 | 9 | 1 | 0 | -0.01091 | -0.003869 | 9 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | -0.005074 |  |  | -0.02409 | 0.08279 |  | 10 | 0.007186 | 5 | 5 | 0 | -0.01176 | 0.0297 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | -0.001586 |  |  | -0.2737 | 0.3234 |  | 10 | 0.03276 | 5 | 5 | 0 | -0.102 | 0.1706 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -8.699e-05 |  |  | -0.000413 | 0.001419 |  | 10 | 0.0001232 | 5 | 5 | 0 | -0.0002041 | 0.0005145 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -2.718e-05 |  |  | -0.004692 | 0.005545 |  | 10 | 0.0005615 | 5 | 5 | 0 | -0.001773 | 0.002856 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -0.8878 |  |  | -1.531 | 0.305 |  | 10 | -0.8025 | 9 | 1 | 0 | -1.131 | -0.4444 | 9 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -0.0001711 |  |  | -0.001406 | 0.0002916 |  | 10 | -0.0002042 | 7 | 3 | 0 | -0.0005195 | 3.155e-05 | 7 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -0.0001711 |  |  | -0.001406 | 0.0002916 |  | 10 | -0.0002042 | 7 | 3 | 0 | -0.0005113 | 2.807e-05 | 7 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 4 | 60 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | 5.78e-06 |  |  | -9.302e-05 | 0.0001587 |  | 10 | 1.625e-05 | 5 | 5 | 0 | -2.95e-05 | 6.324e-05 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | 0.000478 |  |  | -0.05259 | 0.02819 |  | 10 | -0.004727 | 5 | 5 | 0 | -0.01853 | 0.007403 |  | no preferred direction |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | -0.000478 |  |  | -0.02819 | 0.04551 |  | 10 | 0.004018 | 5 | 5 | 0 | -0.007502 | 0.0167 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | -0.003021 |  |  | -0.05254 | 0.02879 |  | 10 | -0.005514 | 6 | 4 | 0 | -0.01916 | 0.007025 |  | no preferred direction |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.003021 |  |  | -0.02879 | 0.04542 |  | 10 | 0.004803 | 4 | 6 | 0 | -0.006952 | 0.01722 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.005749 |  |  | -0.01924 | 0.002884 |  | 10 | -0.007904 | 9 | 1 | 0 | -0.01197 | -0.004031 | 9 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | 0.01456 |  |  | -0.02386 | 0.09088 |  | 10 | 0.01972 | 4 | 6 | 0 | -0.0001094 | 0.04174 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | 0.07841 |  |  | -0.3239 | 0.3099 |  | 10 | 0.05271 | 4 | 6 | 0 | -0.07806 | 0.1731 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.0002495 |  |  | -0.0004091 | 0.001558 |  | 10 | 0.000338 | 4 | 6 | 0 | 4.411e-06 | 0.0007168 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 0.001344 |  |  | -0.005552 | 0.005312 |  | 10 | 0.0009036 | 4 | 6 | 0 | -0.001304 | 0.002958 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -0.9645 |  |  | -1.936 | 1.248 |  | 10 | -0.761 | 8 | 2 | 0 | -1.268 | -0.1683 | 8 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -8.237e-05 |  |  | -0.001763 | 6.868e-05 |  | 10 | -0.0003072 | 9 | 1 | 0 | -0.00067 | -6.843e-05 | 9 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -8.237e-05 |  |  | -0.001763 | 6.868e-05 |  | 10 | -0.0003072 | 9 | 1 | 0 | -0.0006631 | -7.063e-05 | 9 | diff < 0 |
| paired_K_vs_K1 | 2a | constant | 8 | 60 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | 3.111e-05 |  |  | -0.0001091 | 0.000134 |  | 10 | 2.174e-05 | 3 | 7 | 0 | -2.062e-05 | 6.224e-05 | 3 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | -0.003899 |  |  | -0.02097 | 0.01066 |  | 10 | -0.003322 | 7 | 3 | 0 | -0.009432 | 0.00263 |  | no preferred direction |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.003899 |  |  | -0.01066 | 0.02097 |  | 10 | 0.003322 | 3 | 7 | 0 | -0.002589 | 0.009436 | 3 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | -0.003946 |  |  | -0.02144 | 0.01063 |  | 10 | -0.003245 | 6 | 4 | 0 | -0.009471 | 0.003016 |  | no preferred direction |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.003946 |  |  | -0.01063 | 0.02144 |  | 10 | 0.003245 | 4 | 6 | 0 | -0.002987 | 0.00951 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.003088 |  |  | -0.008951 | 0.001339 |  | 10 | -0.002917 | 7 | 3 | 0 | -0.004937 | -0.0009109 | 7 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | 0.003872 |  |  | -0.02412 | 0.04683 |  | 10 | 0.003949 | 5 | 5 | 0 | -0.01073 | 0.01949 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | -0.02423 |  |  | -0.4648 | 0.476 |  | 10 | 0.01735 | 5 | 5 | 0 | -0.1595 | 0.2028 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 5.531e-05 |  |  | -0.0003445 | 0.000669 |  | 10 | 5.642e-05 | 5 | 5 | 0 | -0.0001556 | 0.0002727 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.0003461 |  |  | -0.00664 | 0.006801 |  | 10 | 0.0002479 | 5 | 5 | 0 | -0.002286 | 0.002874 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -0.4241 |  |  | -1.625 | 0.3081 |  | 10 | -0.4933 | 9 | 1 | 0 | -0.8625 | -0.1618 | 9 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -9.258e-05 |  |  | -0.00102 | 0.0004361 |  | 10 | -0.0001491 | 6 | 4 | 0 | -0.0004348 | 0.0001118 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -9.258e-05 |  |  | -0.00102 | 0.0004361 |  | 10 | -0.0001491 | 6 | 4 | 0 | -0.0004331 | 0.0001169 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 50 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | -3.486e-05 |  |  | -0.0001009 | 0.0002558 |  | 10 | 1.493e-05 | 6 | 4 | 0 | -4.768e-05 | 8.953e-05 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | 0.0003223 |  |  | -0.01676 | 0.01911 |  | 10 | -0.0003642 | 5 | 5 | 0 | -0.006417 | 0.0059 |  | no preferred direction |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | -0.0003223 |  |  | -0.01911 | 0.01676 |  | 10 | 0.0003642 | 5 | 5 | 0 | -0.00582 | 0.006366 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | 0.0005691 |  |  | -0.0177 | 0.0204 |  | 10 | -0.0001175 | 5 | 5 | 0 | -0.006426 | 0.006462 |  | no preferred direction |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | -0.0005691 |  |  | -0.0204 | 0.0177 |  | 10 | 0.0001175 | 5 | 5 | 0 | -0.006461 | 0.006571 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.003814 |  |  | -0.009251 | 0.003057 |  | 10 | -0.00377 | 9 | 1 | 0 | -0.005817 | -0.001634 | 9 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | 0.004785 |  |  | -0.03259 | 0.07017 |  | 10 | 0.006344 | 4 | 6 | 0 | -0.009996 | 0.02488 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | -0.0757 |  |  | -0.5341 | 0.2628 |  | 10 | -0.06023 | 6 | 4 | 0 | -0.2201 | 0.09421 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | 6.836e-05 |  |  | -0.0004656 | 0.001002 |  | 10 | 9.063e-05 | 4 | 6 | 0 | -0.0001417 | 0.0003584 | 4 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.001081 |  |  | -0.007629 | 0.003754 |  | 10 | -0.0008605 | 6 | 4 | 0 | -0.003141 | 0.001331 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -0.4192 |  |  | -2.561 | 0.4412 |  | 10 | -0.6352 | 7 | 3 | 0 | -1.198 | -0.1489 | 7 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -1.578e-05 |  |  | -0.0008956 | 0.0002617 |  | 10 | -0.0001375 | 6 | 4 | 0 | -0.0003724 | 6.562e-05 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -1.578e-05 |  |  | -0.0008956 | 0.0002617 |  | 10 | -0.0001375 | 6 | 4 | 0 | -0.0003724 | 7.195e-05 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 50 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | -4.384e-05 |  |  | -0.0001325 | 0.0001189 |  | 10 | -2.197e-05 | 6 | 4 | 0 | -7.75e-05 | 3.678e-05 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | -0.007091 |  |  | -0.02552 | 0.02026 |  | 10 | -0.006821 | 8 | 2 | 0 | -0.01404 | 0.001158 |  | no preferred direction |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.007091 |  |  | -0.02026 | 0.02552 |  | 10 | 0.006821 | 2 | 8 | 0 | -0.001115 | 0.014 | 2 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | -0.006708 |  |  | -0.02546 | 0.02107 |  | 10 | -0.00667 | 7 | 3 | 0 | -0.01398 | 0.001365 |  | no preferred direction |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | 0.006708 |  |  | -0.02107 | 0.02546 |  | 10 | 0.00667 | 3 | 7 | 0 | -0.001521 | 0.01394 | 3 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.003414 |  |  | -0.008661 | 0.0007955 |  | 10 | -0.00274 | 6 | 4 | 0 | -0.004621 | -0.0009541 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | -0.007318 |  |  | -0.0275 | 0.0411 |  | 10 | -0.0007783 | 6 | 4 | 0 | -0.01107 | 0.01166 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | -0.01694 |  |  | -0.2841 | 0.2467 |  | 10 | -0.02187 | 8 | 2 | 0 | -0.1005 | 0.05854 | 8 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.0001254 |  |  | -0.0004714 | 0.0007047 |  | 10 | -1.334e-05 | 6 | 4 | 0 | -0.0001923 | 0.0001951 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.0002904 |  |  | -0.00487 | 0.004229 |  | 10 | -0.0003749 | 8 | 2 | 0 | -0.001717 | 0.00103 | 8 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -0.1544 |  |  | -1.573 | 0.2961 |  | 10 | -0.4067 | 5 | 5 | 0 | -0.838 | -0.02315 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -3.422e-05 |  |  | -0.0007741 | 0.0002795 |  | 10 | -0.0001424 | 7 | 3 | 0 | -0.00035 | 3.504e-05 | 7 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -3.422e-05 |  |  | -0.0007741 | 0.0002795 |  | 10 | -0.0001424 | 7 | 3 | 0 | -0.0003472 | 3.603e-05 | 7 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 4 | 60 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | -1.354e-05 |  |  | -3.549e-05 | 6.885e-05 |  | 10 | -1.128e-06 | 7 | 3 | 0 | -1.89e-05 | 2.104e-05 | 7 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | stage2_peak_rel_err_signed | tier-independent | paired difference (K) - (K = 1) | 0.000981 |  |  | -0.03213 | 0.01966 |  | 10 | -0.004828 | 5 | 5 | 0 | -0.01426 | 0.0043 |  | no preferred direction |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | stage2_peak_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | -0.000981 |  |  | -0.01966 | 0.03213 |  | 10 | 0.004828 | 5 | 5 | 0 | -0.004308 | 0.01418 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | stage2_peak_locfree_rel_err | tier-independent | paired difference (K) - (K = 1) | 0.0009445 |  |  | -0.03219 | 0.02225 |  | 10 | -0.004375 | 5 | 5 | 0 | -0.01382 | 0.004957 |  | no preferred direction |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | stage2_peak_locfree_rel_err_abs | tier-independent | paired difference (K) - (K = 1) | -0.0009445 |  |  | -0.02225 | 0.03219 |  | 10 | 0.004375 | 5 | 5 | 0 | -0.004887 | 0.01404 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | stage2_rmse_pos_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.001783 |  |  | -0.01212 | 0.0005568 |  | 10 | -0.003012 | 7 | 3 | 0 | -0.005592 | -0.0009025 | 7 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | stage2_tail_mean | tier-independent | paired difference (K) - (K = 1) | -0.002486 |  |  | -0.03101 | 0.03878 |  | 10 | 0.0008467 | 6 | 4 | 0 | -0.0105 | 0.01321 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | stage2_tail_max | tier-independent | paired difference (K) - (K = 1) | -0.01357 |  |  | -0.2173 | 0.1157 |  | 10 | -0.03739 | 5 | 5 | 0 | -0.103 | 0.02382 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | stage2_tail_mean_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -4.262e-05 |  |  | -0.0005316 | 0.0006648 |  | 10 | 1.451e-05 | 6 | 4 | 0 | -0.0001835 | 0.0002243 | 6 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | stage2_tail_max_over_g2_0 | tier-independent | paired difference (K) - (K = 1) | -0.0002326 |  |  | -0.003725 | 0.001984 |  | 10 | -0.000641 | 5 | 5 | 0 | -0.001758 | 0.0003778 | 5 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | stage2_sym_err_max | tier-independent | paired difference (K) - (K = 1) | -0.2063 |  |  | -1.492 | 0.6753 |  | 10 | -0.3515 | 9 | 1 | 0 | -0.713 | -0.01596 | 9 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | eta_T_over_dw | development | paired difference (K) - (K = 1) | -0.0001136 |  |  | -0.0009921 | 2.903e-05 |  | 10 | -0.0002295 | 8 | 2 | 0 | -0.0004409 | -6.497e-05 | 8 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | DeltaT_over_dw_on_max | development | paired difference (K) - (K = 1) | -0.0001136 |  |  | -0.0009921 | 2.903e-05 |  | 10 | -0.0002295 | 8 | 2 | 0 | -0.0004338 | -6.502e-05 | 8 | diff < 0 |
| paired_K_vs_K1 | 2a | decay | 8 | 60 | DeltaT_over_dw_off_max | development | paired difference (K) - (K = 1) | -1.79e-05 |  |  | -4.092e-05 | 6.319e-05 |  | 10 | -8.554e-06 | 7 | 3 | 0 | -2.475e-05 | 1.144e-05 | 7 | diff < 0 |
