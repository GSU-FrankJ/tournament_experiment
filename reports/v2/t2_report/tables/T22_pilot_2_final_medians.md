# T22: Pilot 2 final medians

- priority: core; status: generated; tier: final and development
- sources: `results/v2_pilots/pilot2/analysis/final_table.csv`, `Pilot 2 final_v2.json (60 runs)` (60 files), `Pilot 2 final_development.npz (60 runs)` (60 files), `results/v2_pilots/pilot2/analysis/decomposition_residual_band.csv`
- built by: `tools/v2/report/sec_pilot23.py:build_t22`; base commit `cb0b541`
- transformation: Median, q25, q75 (numpy linear interpolation), min, max and n over the 10 seeds (10501-10510) per (q, arm) at global u1000. Development-tier values: results/v2_pilots/pilot2/analysis/final_table.csv (last training-time checkpoint; equal to final_v2.json['development'] within 1e-16); final-tier values: final_v2.json['final'] (studies.final_tier_columns). Stage-2 drift = |e_hat_2 - e_hat_2^parent| of the candidate's stage-2 mapping on the dev D_2 grid (A: live network; B1/B2: frozen snapshot), effort units (final_table.csv equals final_v2.json['drift_vs_parent'] within 8.9e-16). Location-free peak error from the recovery arrays of final_development.npz (formula of tools/v2/pilot4_common.py:location_free). dec_* = revised stage-1 decomposition (residual-band method, final tier; pilot2_freeze.md section 6), rows source == weights, update == 1000 of decomposition_residual_band.csv; the superseded stage1_learning_err_rel / stage1_inherited_err_rel of final_table.csv are not used. 7 early decomposition rows (u425/u450, B1/B2, q=60) have e_hat_1 outside the parent sweep; no u1000 row is affected. (t*, d*) counts per location are also in T25.

Pilot 2 at global u1000 (end of Phase B), per q and arm, n = 10 seeds (10501-10510). Drift and tail mean in effort units (raw); errors relative to e_1*(0) or e_2*(0); Gmax_full, EXP_root, dReach divided by Delta W = 4. Column tier names the verifier tier of each row.

| q | arm | arm_short | metric | tier | median | q25 | q75 | min | max | n | n_true | argmax_t_d_counts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | A_joint | A | stage2_drift_cand_on_max | development | 6.28 | 5.469 | 6.771 | 4.441 | 7.034 | 10 |  |  |
| 50 | A_joint | A | stage2_drift_cand_on_mean_cellmass_weighted | development | 2.97 | 2.659 | 3.178 | 2.488 | 3.874 | 10 |  |  |
| 50 | A_joint | A | stage2_drift_cand_off_max | development | 2.076 | 1.41 | 3.06 | 0.4008 | 5.002 | 10 |  |  |
| 50 | A_joint | A | stage2_drift_cand_off_mean_unweighted | development | 0.9847 | 0.5751 | 1.582 | 0.347 | 2.471 | 10 |  |  |
| 50 | A_joint | A | stage2_peak_rel_err_signed | tier-independent | -0.06038 | -0.07007 | -0.05263 | -0.1015 | -0.04108 | 10 |  |  |
| 50 | A_joint | A | stage2_peak_rel_err_abs | tier-independent | 0.06038 | 0.05263 | 0.07007 | 0.04108 | 0.1015 | 10 |  |  |
| 50 | A_joint | A | stage2_peak_locfree_rel_err | tier-independent | -0.06 | -0.06922 | -0.05247 | -0.1011 | -0.04108 | 10 |  |  |
| 50 | A_joint | A | stage2_peak_locfree_rel_err_abs | tier-independent | 0.06 | 0.05247 | 0.06922 | 0.04108 | 0.1011 | 10 |  |  |
| 50 | A_joint | A | stage2_peak_locfree_argmax_d | tier-independent | -0.5 | -1.375 | 0.375 | -2 | 2 | 10 |  |  |
| 50 | A_joint | A | stage2_tail_mean | tier-independent | 2.584 | 2.175 | 2.82 | 1.691 | 4.567 | 10 |  |  |
| 50 | A_joint | A | stage2_tail_mean_over_g2_0 | tier-independent | 0.03692 | 0.03107 | 0.04029 | 0.02415 | 0.06525 | 10 |  |  |
| 50 | A_joint | A | stage1_rel_err_signed | tier-independent | -0.03841 | -0.08368 | -0.01297 | -0.1986 | 0.03061 | 10 |  |  |
| 50 | A_joint | A | stage1_rel_err_abs | tier-independent | 0.03841 | 0.02434 | 0.08368 | 0.004075 | 0.1986 | 10 |  |  |
| 50 | A_joint | A | Gmax_full_over_dw | development | 0.003227 | 0.002478 | 0.004611 | 0.001565 | 0.01068 | 10 |  | (t=1, d=0) x4; (t=2, d=-104) x1; (t=2, d=-100) x1; (t=2, d=-96) x1; (t=2, d=-92) x1; (t=2, d=-76) x1; (t=2, d=-72) x1 |
| 50 | A_joint | A | Gmax_full_over_dw | final | 0.003273 | 0.002488 | 0.004691 | 0.001542 | 0.01071 | 10 |  | (t=1, d=0) x4; (t=2, d=-102) x1; (t=2, d=-100) x1; (t=2, d=-98) x1; (t=2, d=-92) x1; (t=2, d=-78) x1; (t=2, d=-72) x1 |
| 50 | A_joint | A | EXP_root_over_dw | development | 0.00129 | 0.0003982 | 0.002586 | 0.0003487 | 0.01068 | 10 |  |  |
| 50 | A_joint | A | EXP_root_over_dw | final | 0.00128 | 0.0004281 | 0.002609 | 0.0003556 | 0.01071 | 10 |  |  |
| 50 | A_joint | A | dReach_over_dw | development | 0.003801 | 0.002649 | 0.005666 | 0.001722 | 0.01212 | 10 |  |  |
| 50 | A_joint | A | dReach_over_dw | final | 0.003808 | 0.002667 | 0.005712 | 0.001797 | 0.01227 | 10 |  |  |
| 50 | A_joint | A | dec_learning_rel | final | -0.04366 | -0.08561 | -0.009968 | -0.2059 | 0.02332 | 10 |  |  |
| 50 | A_joint | A | dec_learning_rel_abs | final | 0.04366 | 0.02124 | 0.08561 | 0.0006464 | 0.2059 | 10 |  |  |
| 50 | A_joint | A | dec_inherited_rel | final | -0.002571 | -0.006482 | 0.007286 | -0.01414 | 0.018 | 10 |  |  |
| 50 | A_joint | A | dec_inherited_rel_abs | final | 0.007286 | 0.004393 | 0.01071 | 0.001714 | 0.018 | 10 |  |  |
| 50 | A_joint | A | dec_learning_contains_0 | final |  |  |  |  |  | 10 | 0 |  |
| 50 | A_joint | A | dec_inherited_contains_0 | final |  |  |  |  |  | 10 | 3 |  |
| 50 | B1_frozen_allnorm | B1 | stage2_drift_cand_on_max | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | stage2_drift_cand_on_mean_cellmass_weighted | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | stage2_drift_cand_off_max | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | stage2_drift_cand_off_mean_unweighted | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | stage2_peak_rel_err_signed | tier-independent | -0.1293 | -0.1426 | -0.1254 | -0.1562 | -0.07954 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | stage2_peak_rel_err_abs | tier-independent | 0.1293 | 0.1254 | 0.1426 | 0.07954 | 0.1562 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | stage2_peak_locfree_rel_err | tier-independent | -0.1289 | -0.1423 | -0.1233 | -0.1562 | -0.07678 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | stage2_peak_locfree_rel_err_abs | tier-independent | 0.1289 | 0.1233 | 0.1423 | 0.07678 | 0.1562 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | stage2_peak_locfree_argmax_d | tier-independent | 0.5 | -0.25 | 1.375 | -3 | 4.5 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | stage2_tail_mean | tier-independent | 1.477 | 1.352 | 1.693 | 1.008 | 2.092 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | stage2_tail_mean_over_g2_0 | tier-independent | 0.0211 | 0.01932 | 0.02419 | 0.01439 | 0.02989 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | stage1_rel_err_signed | tier-independent | 0.01941 | -0.04321 | 0.04382 | -0.09082 | 0.08262 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | stage1_rel_err_abs | tier-independent | 0.04792 | 0.03281 | 0.07503 | 0.007959 | 0.09082 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | Gmax_full_over_dw | development | 0.004606 | 0.004373 | 0.005746 | 0.001511 | 0.006638 | 10 |  | (t=2, d=-4) x9; (t=2, d=100) x1 |
| 50 | B1_frozen_allnorm | B1 | Gmax_full_over_dw | final | 0.004606 | 0.004373 | 0.005746 | 0.001544 | 0.006718 | 10 |  | (t=2, d=-4) x8; (t=2, d=-6) x1; (t=2, d=102) x1 |
| 50 | B1_frozen_allnorm | B1 | EXP_root_over_dw | development | 0.001979 | 0.001563 | 0.002125 | 0.0008418 | 0.002408 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | EXP_root_over_dw | final | 0.00198 | 0.001541 | 0.002117 | 0.0008513 | 0.002397 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | dReach_over_dw | development | 0.005651 | 0.005349 | 0.006373 | 0.001903 | 0.006646 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | dReach_over_dw | final | 0.005644 | 0.005341 | 0.006359 | 0.001986 | 0.006719 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | dec_learning_rel | final | 0.02005 | -0.05558 | 0.04176 | -0.08373 | 0.07426 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | dec_learning_rel_abs | final | 0.0434 | 0.03572 | 0.07155 | 0.006031 | 0.08373 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | dec_inherited_rel | final | 0.001393 | -0.003375 | 0.009321 | -0.01179 | 0.01607 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | dec_inherited_rel_abs | final | 0.006643 | 0.003268 | 0.01077 | 0.0008571 | 0.01607 | 10 |  |  |
| 50 | B1_frozen_allnorm | B1 | dec_learning_contains_0 | final |  |  |  |  |  | 10 | 0 |  |
| 50 | B1_frozen_allnorm | B1 | dec_inherited_contains_0 | final |  |  |  |  |  | 10 | 1 |  |
| 50 | B2_frozen_s1norm | B2 | stage2_drift_cand_on_max | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | stage2_drift_cand_on_mean_cellmass_weighted | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | stage2_drift_cand_off_max | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | stage2_drift_cand_off_mean_unweighted | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | stage2_peak_rel_err_signed | tier-independent | -0.1293 | -0.1426 | -0.1254 | -0.1562 | -0.07954 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | stage2_peak_rel_err_abs | tier-independent | 0.1293 | 0.1254 | 0.1426 | 0.07954 | 0.1562 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | stage2_peak_locfree_rel_err | tier-independent | -0.1289 | -0.1423 | -0.1233 | -0.1562 | -0.07678 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | stage2_peak_locfree_rel_err_abs | tier-independent | 0.1289 | 0.1233 | 0.1423 | 0.07678 | 0.1562 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | stage2_peak_locfree_argmax_d | tier-independent | 0.5 | -0.25 | 1.375 | -3 | 4.5 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | stage2_tail_mean | tier-independent | 1.477 | 1.352 | 1.693 | 1.008 | 2.092 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | stage2_tail_mean_over_g2_0 | tier-independent | 0.0211 | 0.01932 | 0.02419 | 0.01439 | 0.02989 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | stage1_rel_err_signed | tier-independent | 0.01962 | -0.07218 | 0.03879 | -0.09501 | 0.1078 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | stage1_rel_err_abs | tier-independent | 0.06479 | 0.03304 | 0.09121 | 0.01374 | 0.1078 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | Gmax_full_over_dw | development | 0.004606 | 0.004373 | 0.005746 | 0.00185 | 0.006638 | 10 |  | (t=2, d=-4) x9; (t=1, d=0) x1 |
| 50 | B2_frozen_s1norm | B2 | Gmax_full_over_dw | final | 0.004606 | 0.004373 | 0.005746 | 0.001865 | 0.006718 | 10 |  | (t=2, d=-4) x8; (t=1, d=0) x1; (t=2, d=-6) x1 |
| 50 | B2_frozen_s1norm | B2 | EXP_root_over_dw | development | 0.001705 | 0.001278 | 0.003441 | 0.0009865 | 0.004917 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | EXP_root_over_dw | final | 0.001707 | 0.001268 | 0.003448 | 0.0009543 | 0.004912 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | dReach_over_dw | development | 0.005795 | 0.004887 | 0.00687 | 0.002998 | 0.009355 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | dReach_over_dw | final | 0.00579 | 0.004874 | 0.006855 | 0.003043 | 0.00942 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | dec_learning_rel | final | 0.01553 | -0.0717 | 0.04769 | -0.09586 | 0.1059 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | dec_learning_rel_abs | final | 0.06725 | 0.03619 | 0.09464 | 0.0141 | 0.1059 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | dec_inherited_rel | final | 0.001393 | -0.003375 | 0.009321 | -0.01179 | 0.01607 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | dec_inherited_rel_abs | final | 0.006643 | 0.003268 | 0.01077 | 0.0008571 | 0.01607 | 10 |  |  |
| 50 | B2_frozen_s1norm | B2 | dec_learning_contains_0 | final |  |  |  |  |  | 10 | 0 |  |
| 50 | B2_frozen_s1norm | B2 | dec_inherited_contains_0 | final |  |  |  |  |  | 10 | 1 |  |
| 60 | A_joint | A | stage2_drift_cand_on_max | development | 4.001 | 2.937 | 5.565 | 2.534 | 5.94 | 10 |  |  |
| 60 | A_joint | A | stage2_drift_cand_on_mean_cellmass_weighted | development | 1.537 | 1.302 | 2.368 | 1.186 | 2.703 | 10 |  |  |
| 60 | A_joint | A | stage2_drift_cand_off_max | development | 1.558 | 1.017 | 2.834 | 0.2298 | 5.903 | 10 |  |  |
| 60 | A_joint | A | stage2_drift_cand_off_mean_unweighted | development | 0.9922 | 0.5097 | 1.261 | 0.1568 | 2.369 | 10 |  |  |
| 60 | A_joint | A | stage2_peak_rel_err_signed | tier-independent | -0.06171 | -0.06645 | -0.05603 | -0.1206 | -0.02658 | 10 |  |  |
| 60 | A_joint | A | stage2_peak_rel_err_abs | tier-independent | 0.06171 | 0.05603 | 0.06645 | 0.02658 | 0.1206 | 10 |  |  |
| 60 | A_joint | A | stage2_peak_locfree_rel_err | tier-independent | -0.06037 | -0.06575 | -0.05585 | -0.1206 | -0.02655 | 10 |  |  |
| 60 | A_joint | A | stage2_peak_locfree_rel_err_abs | tier-independent | 0.06037 | 0.05585 | 0.06575 | 0.02655 | 0.1206 | 10 |  |  |
| 60 | A_joint | A | stage2_peak_locfree_argmax_d | tier-independent | -0.25 | -1.375 | 0.375 | -4 | 1.5 | 10 |  |  |
| 60 | A_joint | A | stage2_tail_mean | tier-independent | 2.192 | 2.14 | 2.439 | 1.602 | 3.295 | 10 |  |  |
| 60 | A_joint | A | stage2_tail_mean_over_g2_0 | tier-independent | 0.03758 | 0.03669 | 0.04181 | 0.02747 | 0.05648 | 10 |  |  |
| 60 | A_joint | A | stage1_rel_err_signed | tier-independent | -0.001616 | -0.03625 | 0.06261 | -0.1818 | 0.1525 | 10 |  |  |
| 60 | A_joint | A | stage1_rel_err_abs | tier-independent | 0.05527 | 0.03068 | 0.1353 | 0.001444 | 0.1818 | 10 |  |  |
| 60 | A_joint | A | Gmax_full_over_dw | development | 0.002666 | 0.002038 | 0.003402 | 0.001089 | 0.005673 | 10 |  | (t=2, d=-120) x4; (t=1, d=0) x3; (t=2, d=-124) x1; (t=2, d=-4) x1; (t=2, d=124) x1 |
| 60 | A_joint | A | Gmax_full_over_dw | final | 0.002666 | 0.002065 | 0.003412 | 0.001089 | 0.005673 | 10 |  | (t=1, d=0) x3; (t=2, d=-120) x2; (t=2, d=-124) x1; (t=2, d=-122) x1; (t=2, d=-118) x1; (t=2, d=-4) x1; (t=2, d=122) x1 |
| 60 | A_joint | A | EXP_root_over_dw | development | 0.0006633 | 0.0003975 | 0.002634 | 0.0001916 | 0.004394 | 10 |  |  |
| 60 | A_joint | A | EXP_root_over_dw | final | 0.0006536 | 0.0003868 | 0.002627 | 0.0001952 | 0.004385 | 10 |  |  |
| 60 | A_joint | A | dReach_over_dw | development | 0.002931 | 0.002121 | 0.004436 | 0.001234 | 0.005035 | 10 |  |  |
| 60 | A_joint | A | dReach_over_dw | final | 0.002942 | 0.002115 | 0.004617 | 0.001332 | 0.005152 | 10 |  |  |
| 60 | A_joint | A | dec_learning_rel | final | 0.006099 | -0.03106 | 0.05547 | -0.1787 | 0.1502 | 10 |  |  |
| 60 | A_joint | A | dec_learning_rel_abs | final | 0.0486 | 0.02682 | 0.1333 | 0.0007839 | 0.1787 | 10 |  |  |
| 60 | A_joint | A | dec_inherited_rel | final | -0.001929 | -0.002957 | 0.001993 | -0.01286 | 0.008486 | 10 |  |  |
| 60 | A_joint | A | dec_inherited_rel_abs | final | 0.002829 | 0.002314 | 0.007136 | 0.001029 | 0.01286 | 10 |  |  |
| 60 | A_joint | A | dec_learning_contains_0 | final |  |  |  |  |  | 10 | 1 |  |
| 60 | A_joint | A | dec_inherited_contains_0 | final |  |  |  |  |  | 10 | 1 |  |
| 60 | B1_frozen_allnorm | B1 | stage2_drift_cand_on_max | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | stage2_drift_cand_on_mean_cellmass_weighted | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | stage2_drift_cand_off_max | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | stage2_drift_cand_off_mean_unweighted | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | stage2_peak_rel_err_signed | tier-independent | -0.09872 | -0.1065 | -0.09011 | -0.1351 | -0.06882 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | stage2_peak_rel_err_abs | tier-independent | 0.09872 | 0.09011 | 0.1065 | 0.06882 | 0.1351 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | stage2_peak_locfree_rel_err | tier-independent | -0.098 | -0.1055 | -0.08931 | -0.1342 | -0.06879 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | stage2_peak_locfree_rel_err_abs | tier-independent | 0.098 | 0.08931 | 0.1055 | 0.06879 | 0.1342 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | stage2_peak_locfree_argmax_d | tier-independent | 0.25 | -1.375 | 2.5 | -3 | 5 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | stage2_tail_mean | tier-independent | 1.339 | 1.041 | 1.443 | 0.9137 | 1.878 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | stage2_tail_mean_over_g2_0 | tier-independent | 0.02295 | 0.01784 | 0.02473 | 0.01566 | 0.03219 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | stage1_rel_err_signed | tier-independent | 0.05607 | -0.01696 | 0.08817 | -0.07548 | 0.1313 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | stage1_rel_err_abs | tier-independent | 0.06941 | 0.05164 | 0.08817 | 0.03886 | 0.1313 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | Gmax_full_over_dw | development | 0.002039 | 0.001767 | 0.002642 | 0.001181 | 0.003512 | 10 |  | (t=2, d=-4) x7; (t=1, d=0) x2; (t=2, d=-32) x1 |
| 60 | B1_frozen_allnorm | B1 | Gmax_full_over_dw | final | 0.002074 | 0.001846 | 0.002633 | 0.001243 | 0.003512 | 10 |  | (t=2, d=-2) x5; (t=1, d=0) x2; (t=2, d=-4) x2; (t=2, d=-30) x1 |
| 60 | B1_frozen_allnorm | B1 | EXP_root_over_dw | development | 0.001027 | 0.0008026 | 0.00134 | 0.0006155 | 0.002772 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | EXP_root_over_dw | final | 0.001004 | 0.0008025 | 0.001343 | 0.0005865 | 0.002745 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | dReach_over_dw | development | 0.002668 | 0.002141 | 0.003717 | 0.001588 | 0.004702 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | dReach_over_dw | final | 0.002709 | 0.002212 | 0.003688 | 0.001729 | 0.004697 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | dec_learning_rel | final | 0.0553 | -0.01728 | 0.08226 | -0.07599 | 0.1362 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | dec_learning_rel_abs | final | 0.06534 | 0.05488 | 0.08357 | 0.03963 | 0.1362 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | dec_inherited_rel | final | 0.0006429 | -0.0008357 | 0.005914 | -0.005657 | 0.01157 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | dec_inherited_rel_abs | final | 0.005014 | 0.0008357 | 0.006043 | 0.0002571 | 0.01157 | 10 |  |  |
| 60 | B1_frozen_allnorm | B1 | dec_learning_contains_0 | final |  |  |  |  |  | 10 | 0 |  |
| 60 | B1_frozen_allnorm | B1 | dec_inherited_contains_0 | final |  |  |  |  |  | 10 | 3 |  |
| 60 | B2_frozen_s1norm | B2 | stage2_drift_cand_on_max | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | stage2_drift_cand_on_mean_cellmass_weighted | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | stage2_drift_cand_off_max | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | stage2_drift_cand_off_mean_unweighted | development | 0 | 0 | 0 | 0 | 0 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | stage2_peak_rel_err_signed | tier-independent | -0.09872 | -0.1065 | -0.09011 | -0.1351 | -0.06882 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | stage2_peak_rel_err_abs | tier-independent | 0.09872 | 0.09011 | 0.1065 | 0.06882 | 0.1351 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | stage2_peak_locfree_rel_err | tier-independent | -0.098 | -0.1055 | -0.08931 | -0.1342 | -0.06879 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | stage2_peak_locfree_rel_err_abs | tier-independent | 0.098 | 0.08931 | 0.1055 | 0.06879 | 0.1342 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | stage2_peak_locfree_argmax_d | tier-independent | 0.25 | -1.375 | 2.5 | -3 | 5 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | stage2_tail_mean | tier-independent | 1.339 | 1.041 | 1.443 | 0.9137 | 1.878 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | stage2_tail_mean_over_g2_0 | tier-independent | 0.02295 | 0.01784 | 0.02473 | 0.01566 | 0.03219 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | stage1_rel_err_signed | tier-independent | 0.009708 | -0.04969 | 0.02604 | -0.1434 | 0.07444 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | stage1_rel_err_abs | tier-independent | 0.04948 | 0.02456 | 0.06674 | 0.002742 | 0.1434 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | Gmax_full_over_dw | development | 0.001833 | 0.001767 | 0.002251 | 0.001181 | 0.003512 | 10 |  | (t=2, d=-4) x8; (t=2, d=-32) x1; (t=2, d=40) x1 |
| 60 | B2_frozen_s1norm | B2 | Gmax_full_over_dw | final | 0.001891 | 0.001837 | 0.002291 | 0.001243 | 0.003512 | 10 |  | (t=2, d=-2) x6; (t=2, d=-4) x2; (t=2, d=-30) x1; (t=2, d=40) x1 |
| 60 | B2_frozen_s1norm | B2 | EXP_root_over_dw | development | 0.0007316 | 0.0005208 | 0.001034 | 0.0003708 | 0.003287 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | EXP_root_over_dw | final | 0.0007127 | 0.0005239 | 0.001034 | 0.000371 | 0.003289 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | dReach_over_dw | development | 0.002383 | 0.001946 | 0.002556 | 0.00125 | 0.006242 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | dReach_over_dw | final | 0.002422 | 0.002022 | 0.002623 | 0.001261 | 0.006245 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | dec_learning_rel | final | 0.007651 | -0.05715 | 0.02295 | -0.1439 | 0.06827 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | dec_learning_rel_abs | final | 0.05282 | 0.0207 | 0.06684 | 0.00377 | 0.1439 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | dec_inherited_rel | final | 0.0006429 | -0.0008357 | 0.005914 | -0.005657 | 0.01157 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | dec_inherited_rel_abs | final | 0.005014 | 0.0008357 | 0.006043 | 0.0002571 | 0.01157 | 10 |  |  |
| 60 | B2_frozen_s1norm | B2 | dec_learning_contains_0 | final |  |  |  |  |  | 10 | 0 |  |
| 60 | B2_frozen_s1norm | B2 | dec_inherited_contains_0 | final |  |  |  |  |  | 10 | 3 |  |
