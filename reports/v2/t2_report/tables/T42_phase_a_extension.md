# T42: Phase A extension

- priority: core; status: generated; tier: final and development
- sources: `results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv`, `results/v2_pilots/phaseA_ext/analysis/curves_weights_every25.csv`, `results/v2_pilots/phaseA_ext/analysis/runs.csv`, `recovery arrays at u400 (Pilot-1 parent) and u800/u1200/u1600` (80 files), `final-tier values saved at u400 (Pilot 1) and u1600 (extension)` (40 files), `weight exports re-evaluated on the final tier` (80 files), `results/v2_pilots/pilot4/analysis/candidates_all.csv`
- built by: `tools/v2/report/sec_ext.py:build_t42`; base commit `cb0b541`
- transformation: Rows with tier development / tier-independent and source results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv are that file's rows verbatim (recomputed from the per-run values of curves_weights_every25.csv: max abs diff 0, n equal: True). Added rows: location-free peak error and its argmax d from the saved recovery arrays (d = 0 value of the arrays equals the curves' signed peak error, max abs diff 0; u1600 location-free equals candidates_all.csv ext K=1, max abs diff 0); final-tier rows of the five tier-dependent stage-2 metrics (u400: Pilot-1 final_v2.json, the same policy as the u400 parent export; u1600: extension final_v2.json; u800/u1200: final-tier verifier evaluation of the weight exports, logged in reevaluations.csv; the same evaluation reproduces the saved u400/u1600 values, max abs diff 0). Statistics over 10 seeds per q (10501-10510), numpy linear percentiles. The u400 row is the Pilot-1 expected parent (seeds are the same runs). Development tier = state step 4, effort step 1, GL 16; final = 2, 0.5, 32.

Long format: one row per (q, checkpoint u, metric, tier). Raw effort units for tail, symmetry, sigma and e_pred/e_learned; *_over_g2_0 and peak errors are fractions of e2*(0); eta_2 and Delta_2 are /Delta W.

| q | update | metric | tier | median | q25 | q75 | min | max | n | source |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 400 | stage2_peak_rel_err_signed | tier-independent | -0.1293 | -0.1426 | -0.1254 | -0.1562 | -0.07954 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | stage2_peak_rel_err_signed | tier-independent | -0.07907 | -0.09373 | -0.06996 | -0.1115 | -0.04872 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | stage2_peak_rel_err_signed | tier-independent | -0.06002 | -0.0861 | -0.053 | -0.1121 | -0.03316 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | stage2_peak_rel_err_signed | tier-independent | -0.06835 | -0.0709 | -0.05722 | -0.1286 | -0.03741 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | stage2_peak_rel_err_abs | tier-independent | 0.1293 | 0.1254 | 0.1426 | 0.07954 | 0.1562 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | stage2_peak_rel_err_abs | tier-independent | 0.07907 | 0.06996 | 0.09373 | 0.04872 | 0.1115 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | stage2_peak_rel_err_abs | tier-independent | 0.06002 | 0.053 | 0.0861 | 0.03316 | 0.1121 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | stage2_peak_rel_err_abs | tier-independent | 0.06835 | 0.05722 | 0.0709 | 0.03741 | 0.1286 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | stage2_peak_locfree_rel_err | tier-independent | -0.1289 | -0.1423 | -0.1233 | -0.1562 | -0.07678 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 50 | 800 | stage2_peak_locfree_rel_err | tier-independent | -0.07846 | -0.09365 | -0.06958 | -0.1094 | -0.04696 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 50 | 1200 | stage2_peak_locfree_rel_err | tier-independent | -0.05401 | -0.07959 | -0.05145 | -0.1119 | -0.03311 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 50 | 1600 | stage2_peak_locfree_rel_err | tier-independent | -0.06826 | -0.07074 | -0.05652 | -0.1286 | -0.0368 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 50 | 400 | stage2_peak_locfree_argmax_d | tier-independent | 0.5 | -0.25 | 1.375 | -3 | 4.5 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 50 | 800 | stage2_peak_locfree_argmax_d | tier-independent | -0.25 | -0.875 | 0.5 | -2.5 | 2.5 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 50 | 1200 | stage2_peak_locfree_argmax_d | tier-independent | -0.5 | -1.375 | 0.375 | -6.5 | 6 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 50 | 1600 | stage2_peak_locfree_argmax_d | tier-independent | 0 | -0.875 | 1.125 | -1.5 | 2 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 50 | 400 | stage2_rmse_pos_over_g2_0 | tier-independent | 0.04367 | 0.03551 | 0.04881 | 0.03362 | 0.05957 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | stage2_rmse_pos_over_g2_0 | tier-independent | 0.03083 | 0.02859 | 0.03521 | 0.01949 | 0.04158 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | stage2_rmse_pos_over_g2_0 | tier-independent | 0.02896 | 0.02332 | 0.0309 | 0.02118 | 0.0411 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | stage2_rmse_pos_over_g2_0 | tier-independent | 0.02605 | 0.0221 | 0.02871 | 0.01422 | 0.05976 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | stage2_tail_mean | tier-independent | 1.477 | 1.352 | 1.693 | 1.008 | 2.092 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | stage2_tail_mean | tier-independent | 0.8706 | 0.727 | 0.8805 | 0.5781 | 1.254 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | stage2_tail_mean | tier-independent | 0.6386 | 0.5335 | 0.6965 | 0.4294 | 0.9571 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | stage2_tail_mean | tier-independent | 0.515 | 0.4559 | 0.5639 | 0.3528 | 0.8831 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | stage2_tail_mean_over_g2_0 | tier-independent | 0.0211 | 0.01932 | 0.02419 | 0.01439 | 0.02989 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | stage2_tail_mean_over_g2_0 | tier-independent | 0.01244 | 0.01039 | 0.01258 | 0.008258 | 0.01791 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | stage2_tail_mean_over_g2_0 | tier-independent | 0.009123 | 0.007621 | 0.009951 | 0.006134 | 0.01367 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | stage2_tail_mean_over_g2_0 | tier-independent | 0.007358 | 0.006513 | 0.008055 | 0.00504 | 0.01262 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | stage2_tail_max | tier-independent | 4.76 | 4.264 | 5.108 | 3.193 | 6.134 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | stage2_tail_max | tier-independent | 3.144 | 2.792 | 3.444 | 2.304 | 5.734 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | stage2_tail_max | tier-independent | 2.813 | 2.585 | 3.003 | 2.309 | 4.23 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | stage2_tail_max | tier-independent | 2.22 | 2.131 | 2.597 | 1.656 | 4.907 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | stage2_tail_max_over_g2_0 | tier-independent | 0.068 | 0.06092 | 0.07297 | 0.04562 | 0.08763 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | stage2_tail_max_over_g2_0 | tier-independent | 0.04492 | 0.03988 | 0.0492 | 0.03292 | 0.08192 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | stage2_tail_max_over_g2_0 | tier-independent | 0.04019 | 0.03693 | 0.04289 | 0.03298 | 0.06042 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | stage2_tail_max_over_g2_0 | tier-independent | 0.03172 | 0.03044 | 0.03711 | 0.02365 | 0.07009 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | stage2_sym_err_max | tier-independent | 2.981 | 2.351 | 3.878 | 1.128 | 6.462 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | stage2_sym_err_max | tier-independent | 3.467 | 2.267 | 4.268 | 1.452 | 4.752 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | stage2_sym_err_max | tier-independent | 2.855 | 2.233 | 4.053 | 0.8764 | 7.866 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | stage2_sym_err_max | tier-independent | 2.638 | 1.742 | 3.529 | 1.447 | 5.473 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | eta_T_over_dw | development | 0.004606 | 0.004373 | 0.005746 | 0.001511 | 0.006638 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | eta_T_over_dw | development | 0.002664 | 0.001909 | 0.003183 | 0.001172 | 0.003667 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | eta_T_over_dw | development | 0.002556 | 0.001709 | 0.002941 | 0.0005958 | 0.003502 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | eta_T_over_dw | development | 0.001469 | 0.001119 | 0.002431 | 0.0007199 | 0.006971 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | eta_T_over_dw | final | 0.004606 | 0.004373 | 0.005746 | 0.001544 | 0.006718 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 800 | eta_T_over_dw | final | 0.002664 | 0.001937 | 0.003183 | 0.001393 | 0.003667 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 1200 | eta_T_over_dw | final | 0.002565 | 0.001709 | 0.002942 | 0.000748 | 0.003502 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 1600 | eta_T_over_dw | final | 0.001591 | 0.001297 | 0.002481 | 0.0009051 | 0.006971 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 400 | DeltaT_over_dw_on_max | development | 0.004606 | 0.004373 | 0.005746 | 0.001451 | 0.006638 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | DeltaT_over_dw_on_max | development | 0.002664 | 0.001909 | 0.003183 | 0.001172 | 0.003667 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | DeltaT_over_dw_on_max | development | 0.002556 | 0.001709 | 0.002941 | 0.0005958 | 0.003502 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | DeltaT_over_dw_on_max | development | 0.001469 | 0.001119 | 0.002431 | 0.0007199 | 0.006971 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | DeltaT_over_dw_on_max | final | 0.004606 | 0.004373 | 0.005746 | 0.001524 | 0.006718 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 800 | DeltaT_over_dw_on_max | final | 0.002664 | 0.001937 | 0.003183 | 0.001393 | 0.003667 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 1200 | DeltaT_over_dw_on_max | final | 0.002565 | 0.001709 | 0.002942 | 0.000748 | 0.003502 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 1600 | DeltaT_over_dw_on_max | final | 0.001591 | 0.001297 | 0.002481 | 0.0009051 | 0.006971 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 400 | DeltaT_over_dw_on_mean_cellmass_weighted | development | 0.001086 | 0.0007081 | 0.001434 | 0.0003797 | 0.002074 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | DeltaT_over_dw_on_mean_cellmass_weighted | development | 0.0004923 | 0.0003321 | 0.0006156 | 0.0002042 | 0.0006879 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | DeltaT_over_dw_on_mean_cellmass_weighted | development | 0.0005036 | 0.000286 | 0.0007265 | 0.0002222 | 0.0008146 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | DeltaT_over_dw_on_mean_cellmass_weighted | development | 0.0003412 | 0.0002337 | 0.0005086 | 0.0001176 | 0.002879 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | DeltaT_over_dw_on_mean_cellmass_weighted | final | 0.001085 | 0.0007082 | 0.001434 | 0.0003806 | 0.002073 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 800 | DeltaT_over_dw_on_mean_cellmass_weighted | final | 0.0004917 | 0.0003319 | 0.0006152 | 0.0002052 | 0.0006868 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 1200 | DeltaT_over_dw_on_mean_cellmass_weighted | final | 0.0005037 | 0.000286 | 0.000726 | 0.0002218 | 0.0008139 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 1600 | DeltaT_over_dw_on_mean_cellmass_weighted | final | 0.0003406 | 0.0002331 | 0.0005079 | 0.0001168 | 0.002879 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 400 | DeltaT_over_dw_off_max | development | 0.00144 | 0.001238 | 0.001757 | 0.0005562 | 0.002626 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | DeltaT_over_dw_off_max | development | 0.0006315 | 0.0005293 | 0.0007409 | 0.0003459 | 0.001735 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | DeltaT_over_dw_off_max | development | 0.0004991 | 0.0003865 | 0.0006277 | 0.0003265 | 0.001006 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | DeltaT_over_dw_off_max | development | 0.0003395 | 0.000264 | 0.000386 | 0.0001463 | 0.001545 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | DeltaT_over_dw_off_max | final | 0.001457 | 0.001238 | 0.001757 | 0.0005664 | 0.002626 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 800 | DeltaT_over_dw_off_max | final | 0.0006315 | 0.0005315 | 0.0007474 | 0.0003459 | 0.001829 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 1200 | DeltaT_over_dw_off_max | final | 0.0004991 | 0.0003865 | 0.0006277 | 0.0003265 | 0.001009 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 1600 | DeltaT_over_dw_off_max | final | 0.0003395 | 0.000264 | 0.0003867 | 0.0001463 | 0.001545 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 400 | sigma_effort_at_0_t2 | tier-independent | 3.935 | 3.811 | 4.031 | 3.741 | 4.352 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | sigma_effort_at_0_t2 | tier-independent | 3.365 | 3.229 | 3.591 | 3.16 | 3.836 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | sigma_effort_at_0_t2 | tier-independent | 2.981 | 2.897 | 3.224 | 2.814 | 3.524 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | sigma_effort_at_0_t2 | tier-independent | 2.735 | 2.65 | 2.982 | 2.598 | 3.28 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | sigma2_effort_mean_pos | development | 3.476 | 3.348 | 3.556 | 3.21 | 3.748 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | sigma2_effort_mean_pos | development | 3.007 | 2.857 | 3.174 | 2.797 | 3.489 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | sigma2_effort_mean_pos | development | 2.644 | 2.609 | 2.894 | 2.588 | 3.164 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | sigma2_effort_mean_pos | development | 2.469 | 2.377 | 2.646 | 2.324 | 2.955 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | sigma2_effort_mean_pos | final | 3.459 | 3.33 | 3.538 | 3.191 | 3.731 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 800 | sigma2_effort_mean_pos | final | 2.989 | 2.839 | 3.155 | 2.779 | 3.472 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 1200 | sigma2_effort_mean_pos | final | 2.627 | 2.593 | 2.876 | 2.572 | 3.147 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 1600 | sigma2_effort_mean_pos | final | 2.453 | 2.361 | 2.629 | 2.308 | 2.938 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 50 | 400 | e_pred_0 | tier-independent | 66.89 | 66.82 | 66.99 | 66.56 | 67.05 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | e_pred_0 | tier-independent | 67.34 | 67.16 | 67.45 | 66.97 | 67.51 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | e_pred_0 | tier-independent | 67.65 | 67.45 | 67.71 | 67.22 | 67.78 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | e_pred_0 | tier-independent | 67.84 | 67.65 | 67.91 | 67.41 | 67.95 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | e_learned_0 | tier-independent | 60.95 | 60.02 | 61.22 | 59.06 | 64.43 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | e_learned_0 | tier-independent | 64.47 | 63.44 | 65.1 | 62.19 | 66.59 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | e_learned_0 | tier-independent | 65.8 | 63.97 | 66.29 | 62.15 | 67.68 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | e_learned_0 | tier-independent | 65.22 | 65.04 | 65.99 | 61 | 67.38 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 400 | share_peak_gap_explained | tier-independent | 0.3306 | 0.3154 | 0.3559 | 0.2944 | 0.5718 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 800 | share_peak_gap_explained | tier-independent | 0.4808 | 0.4202 | 0.5757 | 0.3296 | 0.8083 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1200 | share_peak_gap_explained | tier-independent | 0.5824 | 0.3847 | 0.686 | 0.3377 | 0.9572 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 50 | 1600 | share_peak_gap_explained | tier-independent | 0.5059 | 0.4365 | 0.5288 | 0.2706 | 0.783 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | stage2_peak_rel_err_signed | tier-independent | -0.09872 | -0.1065 | -0.09011 | -0.1351 | -0.06882 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | stage2_peak_rel_err_signed | tier-independent | -0.0577 | -0.07548 | -0.03899 | -0.169 | -0.01642 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | stage2_peak_rel_err_signed | tier-independent | -0.05439 | -0.08235 | -0.05049 | -0.1103 | -0.03897 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | stage2_peak_rel_err_signed | tier-independent | -0.06003 | -0.07362 | -0.04034 | -0.07572 | 0.003541 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | stage2_peak_rel_err_abs | tier-independent | 0.09872 | 0.09011 | 0.1065 | 0.06882 | 0.1351 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | stage2_peak_rel_err_abs | tier-independent | 0.0577 | 0.03899 | 0.07548 | 0.01642 | 0.169 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | stage2_peak_rel_err_abs | tier-independent | 0.05439 | 0.05049 | 0.08235 | 0.03897 | 0.1103 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | stage2_peak_rel_err_abs | tier-independent | 0.06003 | 0.04034 | 0.07362 | 0.003541 | 0.07572 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | stage2_peak_locfree_rel_err | tier-independent | -0.098 | -0.1055 | -0.08931 | -0.1342 | -0.06879 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 60 | 800 | stage2_peak_locfree_rel_err | tier-independent | -0.05768 | -0.07185 | -0.03656 | -0.1652 | -0.01008 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 60 | 1200 | stage2_peak_locfree_rel_err | tier-independent | -0.05411 | -0.07824 | -0.04893 | -0.1067 | -0.03234 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 60 | 1600 | stage2_peak_locfree_rel_err | tier-independent | -0.05585 | -0.07326 | -0.03969 | -0.07572 | 0.003559 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 60 | 400 | stage2_peak_locfree_argmax_d | tier-independent | 0.25 | -1.375 | 2.5 | -3 | 5 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 60 | 800 | stage2_peak_locfree_argmax_d | tier-independent | 0 | -4 | 0.375 | -6 | 4.5 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 60 | 1200 | stage2_peak_locfree_argmax_d | tier-independent | -2 | -4.75 | 0 | -8 | 1.5 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 60 | 1600 | stage2_peak_locfree_argmax_d | tier-independent | -1 | -2.5 | -0.5 | -7 | 1.5 | 10 | recovery arrays (D04): Pilot-1 final_development.npz at u400, extension checkpoints/u*.npz at u800-u1600 |
| 60 | 400 | stage2_rmse_pos_over_g2_0 | tier-independent | 0.03254 | 0.03051 | 0.03613 | 0.02905 | 0.03852 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | stage2_rmse_pos_over_g2_0 | tier-independent | 0.0309 | 0.02391 | 0.03802 | 0.01857 | 0.06408 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | stage2_rmse_pos_over_g2_0 | tier-independent | 0.0261 | 0.02242 | 0.03097 | 0.01894 | 0.03818 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | stage2_rmse_pos_over_g2_0 | tier-independent | 0.02389 | 0.02089 | 0.03157 | 0.01864 | 0.03376 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | stage2_tail_mean | tier-independent | 1.339 | 1.041 | 1.443 | 0.9137 | 1.878 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | stage2_tail_mean | tier-independent | 0.7954 | 0.7138 | 0.8359 | 0.646 | 1.106 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | stage2_tail_mean | tier-independent | 0.6048 | 0.5467 | 0.6556 | 0.4867 | 0.7983 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | stage2_tail_mean | tier-independent | 0.5312 | 0.501 | 0.5663 | 0.4143 | 0.6374 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | stage2_tail_mean_over_g2_0 | tier-independent | 0.02295 | 0.01784 | 0.02473 | 0.01566 | 0.03219 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | stage2_tail_mean_over_g2_0 | tier-independent | 0.01363 | 0.01224 | 0.01433 | 0.01107 | 0.01896 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | stage2_tail_mean_over_g2_0 | tier-independent | 0.01037 | 0.009372 | 0.01124 | 0.008343 | 0.01369 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | stage2_tail_mean_over_g2_0 | tier-independent | 0.009106 | 0.008588 | 0.009708 | 0.007102 | 0.01093 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | stage2_tail_max | tier-independent | 3.568 | 3.435 | 3.64 | 3.061 | 3.894 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | stage2_tail_max | tier-independent | 2.56 | 2.369 | 2.886 | 2.107 | 3.407 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | stage2_tail_max | tier-independent | 2.346 | 2.095 | 2.531 | 1.894 | 2.709 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | stage2_tail_max | tier-independent | 1.901 | 1.835 | 2.109 | 1.708 | 2.556 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | stage2_tail_max_over_g2_0 | tier-independent | 0.06116 | 0.05889 | 0.06241 | 0.05247 | 0.06675 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | stage2_tail_max_over_g2_0 | tier-independent | 0.04388 | 0.04061 | 0.04948 | 0.03612 | 0.05841 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | stage2_tail_max_over_g2_0 | tier-independent | 0.04022 | 0.03592 | 0.04339 | 0.03246 | 0.04645 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | stage2_tail_max_over_g2_0 | tier-independent | 0.03259 | 0.03146 | 0.03616 | 0.02927 | 0.04382 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | stage2_sym_err_max | tier-independent | 2.887 | 2.016 | 3.47 | 0.9355 | 5.179 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | stage2_sym_err_max | tier-independent | 3.079 | 2.576 | 3.801 | 0.6723 | 4.763 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | stage2_sym_err_max | tier-independent | 2.893 | 1.979 | 3.052 | 0.7366 | 7.713 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | stage2_sym_err_max | tier-independent | 2.39 | 1.517 | 2.535 | 0.8313 | 3.963 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | eta_T_over_dw | development | 0.001833 | 0.001767 | 0.002251 | 0.001181 | 0.003512 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | eta_T_over_dw | development | 0.001113 | 0.0008293 | 0.002318 | 0.0007333 | 0.006252 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | eta_T_over_dw | development | 0.001059 | 0.0006352 | 0.001684 | 0.0004335 | 0.0022 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | eta_T_over_dw | development | 0.0009012 | 0.0007233 | 0.0009584 | 0.0003882 | 0.002156 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | eta_T_over_dw | final | 0.001891 | 0.001837 | 0.002291 | 0.001243 | 0.003512 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 800 | eta_T_over_dw | final | 0.001169 | 0.0008874 | 0.00232 | 0.0007333 | 0.006332 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 1200 | eta_T_over_dw | final | 0.001078 | 0.0006569 | 0.001716 | 0.0004512 | 0.002212 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 1600 | eta_T_over_dw | final | 0.000991 | 0.0007434 | 0.001091 | 0.0004406 | 0.002156 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 400 | DeltaT_over_dw_on_max | development | 0.001833 | 0.001767 | 0.002251 | 0.001181 | 0.003512 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | DeltaT_over_dw_on_max | development | 0.001113 | 0.0008293 | 0.002318 | 0.0007333 | 0.006252 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | DeltaT_over_dw_on_max | development | 0.001059 | 0.0006352 | 0.001684 | 0.0003918 | 0.0022 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | DeltaT_over_dw_on_max | development | 0.0009012 | 0.0007233 | 0.0009584 | 0.0003882 | 0.002156 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | DeltaT_over_dw_on_max | final | 0.001891 | 0.001837 | 0.002291 | 0.001243 | 0.003512 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 800 | DeltaT_over_dw_on_max | final | 0.001169 | 0.0008874 | 0.00232 | 0.0007333 | 0.006332 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 1200 | DeltaT_over_dw_on_max | final | 0.001078 | 0.0006569 | 0.001716 | 0.0004512 | 0.002212 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 1600 | DeltaT_over_dw_on_max | final | 0.000991 | 0.0007434 | 0.001091 | 0.0004406 | 0.002156 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 400 | DeltaT_over_dw_on_mean_cellmass_weighted | development | 0.000397 | 0.0003308 | 0.0005366 | 0.0002937 | 0.0005735 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | DeltaT_over_dw_on_mean_cellmass_weighted | development | 0.0002832 | 0.0001797 | 0.0004761 | 0.0001116 | 0.001451 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | DeltaT_over_dw_on_mean_cellmass_weighted | development | 0.0002222 | 0.0001444 | 0.0003065 | 0.0001153 | 0.0006444 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | DeltaT_over_dw_on_mean_cellmass_weighted | development | 0.0001711 | 0.0001165 | 0.0002749 | 0.0001092 | 0.0004971 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | DeltaT_over_dw_on_mean_cellmass_weighted | final | 0.0003975 | 0.0003312 | 0.0005368 | 0.0002932 | 0.0005733 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 800 | DeltaT_over_dw_on_mean_cellmass_weighted | final | 0.000283 | 0.0001803 | 0.0004764 | 0.0001114 | 0.001451 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 1200 | DeltaT_over_dw_on_mean_cellmass_weighted | final | 0.0002228 | 0.0001448 | 0.0003068 | 0.0001159 | 0.0006446 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 1600 | DeltaT_over_dw_on_mean_cellmass_weighted | final | 0.0001707 | 0.0001163 | 0.0002751 | 0.0001097 | 0.0004974 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 400 | DeltaT_over_dw_off_max | development | 0.0008603 | 0.0008057 | 0.0009269 | 0.0006519 | 0.00105 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | DeltaT_over_dw_off_max | development | 0.0004195 | 0.0003855 | 0.0005562 | 0.000242 | 0.0008193 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | DeltaT_over_dw_off_max | development | 0.000313 | 0.0002443 | 0.0004059 | 0.0002324 | 0.0004535 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | DeltaT_over_dw_off_max | development | 0.0002439 | 0.0002221 | 0.0002957 | 0.0001492 | 0.000465 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | DeltaT_over_dw_off_max | final | 0.0008813 | 0.0008295 | 0.0009269 | 0.0006519 | 0.00105 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 800 | DeltaT_over_dw_off_max | final | 0.0004195 | 0.0003855 | 0.0005614 | 0.0002492 | 0.0008193 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 1200 | DeltaT_over_dw_off_max | final | 0.0003224 | 0.0002443 | 0.0004059 | 0.0002324 | 0.0004535 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 1600 | DeltaT_over_dw_off_max | final | 0.0002439 | 0.0002221 | 0.0002957 | 0.0001595 | 0.000465 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 400 | sigma_effort_at_0_t2 | tier-independent | 3.975 | 3.929 | 4.213 | 3.823 | 4.257 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | sigma_effort_at_0_t2 | tier-independent | 3.499 | 3.378 | 3.693 | 3.219 | 3.807 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | sigma_effort_at_0_t2 | tier-independent | 3.166 | 2.991 | 3.311 | 2.895 | 3.422 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | sigma_effort_at_0_t2 | tier-independent | 2.883 | 2.731 | 3.015 | 2.671 | 3.142 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | sigma2_effort_mean_pos | development | 3.31 | 3.259 | 3.461 | 3.147 | 3.549 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | sigma2_effort_mean_pos | development | 2.908 | 2.795 | 3.058 | 2.703 | 3.119 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | sigma2_effort_mean_pos | development | 2.596 | 2.51 | 2.764 | 2.391 | 2.799 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | sigma2_effort_mean_pos | development | 2.393 | 2.283 | 2.543 | 2.206 | 2.576 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | sigma2_effort_mean_pos | final | 3.295 | 3.244 | 3.445 | 3.133 | 3.532 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 800 | sigma2_effort_mean_pos | final | 2.893 | 2.781 | 3.043 | 2.689 | 3.104 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 1200 | sigma2_effort_mean_pos | final | 2.582 | 2.496 | 2.749 | 2.378 | 2.784 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 1600 | sigma2_effort_mean_pos | final | 2.379 | 2.271 | 2.529 | 2.194 | 2.563 | 10 | D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; re-evaluated weight exports at u800 and u1200 |
| 60 | 400 | e_pred_0 | tier-independent | 56.15 | 56.02 | 56.18 | 56 | 56.24 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | e_pred_0 | tier-independent | 56.41 | 56.31 | 56.48 | 56.25 | 56.57 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | e_pred_0 | tier-independent | 56.6 | 56.52 | 56.69 | 56.46 | 56.75 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | e_pred_0 | tier-independent | 56.75 | 56.68 | 56.84 | 56.61 | 56.87 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | e_learned_0 | tier-independent | 52.57 | 52.12 | 53.08 | 50.45 | 54.32 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | e_learned_0 | tier-independent | 54.97 | 53.93 | 56.06 | 48.47 | 57.38 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | e_learned_0 | tier-independent | 55.16 | 53.53 | 55.39 | 51.9 | 56.06 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | e_learned_0 | tier-independent | 54.83 | 54.04 | 55.98 | 53.92 | 58.54 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 400 | share_peak_gap_explained | tier-independent | 0.3759 | 0.3547 | 0.4143 | 0.2959 | 0.5817 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 800 | share_peak_gap_explained | tier-independent | 0.5791 | 0.4306 | 0.834 | 0.2118 | 1.843 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1200 | share_peak_gap_explained | tier-independent | 0.508 | 0.3888 | 0.5591 | 0.2713 | 0.8043 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
| 60 | 1600 | share_peak_gap_explained | tier-independent | 0.4003 | 0.3573 | 0.6235 | -7.224 | 0.9654 | 10 | results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv |
