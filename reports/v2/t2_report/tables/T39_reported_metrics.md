# T39: Reported metrics

- priority: core; status: generated; tier: final
- sources: `results/v2_T2_locked/confirmation_analysis/reported_metrics.csv`, `results/v2_T2_locked/confirmation/q*/seed*/gates.json` (40 files), `results/v2_T2_locked/confirmation/q*/seed*/drift_test.json` (40 files), `results/v2_T2_locked/confirmation/q*/seed*/gateA_final.npz` (40 files)
- built by: `tools/v2/report/sec_locked_b.py:build_t39`; base commit `cb0b541`
- transformation: Per-run values: results/v2_T2_locked/confirmation_analysis/reported_metrics.csv (the pre-registered analysis script's copy of each run's gates.json 'reported' block); tail max/e2*(0), the dev - final differences (gates.json reported.*.dev_minus_final, named A_dmf_*/B_dmf_*) and the argmin-at-sweep-edge flag from each run's gates.json; symmetry/e2*(0) = A_stage2_sym_err_max / g2_at_0 (gates.json); drift magnitudes and the bit-identity flag from drift_test.json. Statistics over the 20 seeds per q: median, q25, q75 (numpy linear interpolation), min, max; count rows give the number of runs (out of n) for which the stated condition holds. End of A = stage-2 last iterate at u1600; end of B = live stage-1 actor at u2200 with the frozen end-of-A stage-2 snapshot. CSVs parsed with pandas float_precision='round_trip' (exact float64 values). Checks: reported_metrics.csv equals gates.json 'reported' in every copied field (max abs diff 0.0, 40 runs); location-free peak error, its argmax d and the peak error at d=0 recomputed from gateA_final.npz (recovery_e2, recovery_g2; tools/v2/pilot4_common.py:location_free) equal reported_metrics.csv (max abs diff 0.0). Report cross-check (4.5): 1080 cells, 0 mismatches.

Confirmation (protocol v1.1, fresh seeds 20501-20520), per q: the reported (not gated) metrics of the locked protocol, median, IQR, min and max over the 20 runs, and counts of (t*, d*), of decomposition bands containing 0, of band diagnostics and of the drift test.

| q | group | quantity | column | tier | units | n | median | q25 | q75 | min | max | count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | stage 2, end of Phase A (u1600) | peak error at d = 0, signed | A_stage2_peak_rel_err_signed | tier-independent | fraction of e2*(0) | 20 | -0.06082 | -0.06998 | -0.0518 | -0.08072 | -0.00649 |  |
| 50 | stage 2, end of Phase A (u1600) | location-free peak error, (max_d e_hat_2(d) - e2*(0))/e2*(0) | A_stage2_peak_locfree_rel_err | tier-independent | fraction of e2*(0) | 20 | -0.06065 | -0.06998 | -0.04995 | -0.08072 | -0.004309 |  |
| 50 | stage 2, end of Phase A (u1600) | argmax d of e_hat_2 on the recovery grid (location-free peak) | A_stage2_peak_locfree_argmax_d | tier-independent | effort units (gap d) | 20 | 0 | -0.625 | 0.5 | -2.5 | 1 |  |
| 50 | stage 2, end of Phase A (u1600) | symmetry error max_d \|e_hat_2(d) - e_hat_2(-d)\| / e2*(0) | A_stage2_sym_err_max_over_g2_0 | tier-independent | fraction of e2*(0) | 20 | 0.03357 | 0.02957 | 0.0421 | 0.01564 | 0.05921 |  |
| 50 | stage 2, end of Phase A (u1600) | symmetry error, raw | A_stage2_sym_err_max | tier-independent | effort units, raw | 20 | 2.35 | 2.07 | 2.947 | 1.095 | 4.145 |  |
| 50 | stage 2, end of Phase A (u1600) | tail max of e_hat_2 on \|d\| >= 2q / e2*(0) | A_stage2_tail_max_over_g2_0 | tier-independent | fraction of e2*(0) | 20 | 0.03618 | 0.03459 | 0.03776 | 0.0297 | 0.05016 |  |
| 50 | stage 2, end of Phase A (u1600) | tail max, raw | A_stage2_tail_max | tier-independent | effort units [0, 100], raw | 20 | 2.532 | 2.422 | 2.643 | 2.079 | 3.511 |  |
| 50 | stage 2, end of Phase A (u1600) | Delta_2 on path, max / dW | A_DeltaT_over_dw_on_max | final | Delta W (dimensionless) | 20 | 0.001365 | 0.0009915 | 0.001701 | 0.000804 | 0.00329 |  |
| 50 | stage 2, end of Phase A (u1600) | Delta_2 off path, max / dW | A_DeltaT_over_dw_off_max | final | Delta W (dimensionless) | 20 | 0.0004113 | 0.0003709 | 0.0004639 | 0.0002991 | 0.0008293 |  |
| 50 | stage 2, end of Phase A (u1600) | sigma_2(0), SD of the stage-2 Beta action at d = 0 | A_sigma_effort_at_0_t2 | tier-independent | effort units [0, 100] | 20 | 2.836 | 2.727 | 2.881 | 2.638 | 3.117 |  |
| 50 | stage 2, end of Phase A (u1600) | smoothed-game share of the d = 0 peak gap | A_smoothed_share_peak_gap_d0 | tier-independent | fraction of the peak gap e2*(0) - e_hat_2(0) | 20 | 0.5366 | 0.4518 | 0.6318 | 0.3818 | 5.086 |  |
| 50 | stage 1 and full policy, end of Phase B (u2200) | e_hat_1(0), raw | B_e1_at_0 | tier-independent | effort units [0, 100], raw | 20 | 46.72 | 44.59 | 48.05 | 43.08 | 50.69 |  |
| 50 | stage 1 and full policy, end of Phase B (u2200) | stage-1 error (e_hat_1(0) - e1*(0))/e1*(0), signed | B_stage1_rel_err_signed | tier-independent | fraction of e1*(0) | 20 | 0.00111 | -0.04447 | 0.0297 | -0.07694 | 0.08619 |  |
| 50 | stage 1 and full policy, end of Phase B (u2200) | EXP_root / dW | B_EXP_root_over_dw | final | Delta W (dimensionless) | 20 | 0.0006927 | 0.0003392 | 0.001214 | 0.0001568 | 0.002735 |  |
| 50 | stage 1 and full policy, end of Phase B (u2200) | dReach / dW | B_dReach_over_dw | final | Delta W (dimensionless) | 20 | 0.001961 | 0.001551 | 0.002385 | 0.0009819 | 0.00376 |  |
| 50 | stage 1 and full policy, end of Phase B (u2200) | Delta_max_all / dW | B_Deltamax_all_over_dw | final | Delta W (dimensionless) | 20 | 0.001476 | 0.001122 | 0.00176 | 0.000804 | 0.00329 |  |
| 50 | stage 1 and full policy, end of Phase B (u2200) | dFull / dW | B_dFull_over_dw | final | Delta W (dimensionless) | 20 | 0.001961 | 0.001551 | 0.002385 | 0.0009819 | 0.00376 |  |
| 50 | stage 1 and full policy, end of Phase B (u2200) | sigma_1(0), SD of the stage-1 Beta action at d = 0 | B_sigma_effort_at_0_t1 | tier-independent | effort units [0, 100] | 20 | 2.928 | 2.798 | 3.005 | 2.74 | 3.257 |  |
| 50 | stage-1 decomposition (residual band, final tier) | learning term (e_hat_1(0) - e~1)/e1*(0), signed | dec_learning_rel | final | fraction of e1*(0) | 20 | -0.001085 | -0.04688 | 0.02844 | -0.06965 | 0.09189 |  |
| 50 | stage-1 decomposition (residual band, final tier) | inherited term (e~1 - e1*(0))/e1*(0), signed | dec_inherited_rel | final | fraction of e1*(0) | 20 | -0.003429 | -0.009857 | 0.003911 | -0.02164 | 0.008571 |  |
| 50 | dev - final difference of the tier-dependent reported metrics | Delta_2 on path, max / dW: dev - final | A_dmf_DeltaT_over_dw_on_max | dev - final | Delta W (dimensionless) | 20 | -7.204e-05 | -0.0001932 | -1.491e-05 | -0.000261 | 0 |  |
| 50 | dev - final difference of the tier-dependent reported metrics | Delta_2 off path, max / dW: dev - final | A_dmf_DeltaT_over_dw_off_max | dev - final | Delta W (dimensionless) | 20 | 0 | 0 | 0 | -5.831e-06 | 0 |  |
| 50 | dev - final difference of the tier-dependent reported metrics | EXP_root / dW: dev - final | B_dmf_EXP_root_over_dw | dev - final | Delta W (dimensionless) | 20 | -9.133e-06 | -2.044e-05 | 4.1e-06 | -5.561e-05 | 2.943e-05 |  |
| 50 | dev - final difference of the tier-dependent reported metrics | dReach / dW: dev - final | B_dmf_dReach_over_dw | dev - final | Delta W (dimensionless) | 20 | -0.0001038 | -0.000206 | -1.679e-05 | -0.0002729 | 3.275e-05 |  |
| 50 | dev - final difference of the tier-dependent reported metrics | Delta_max_all / dW: dev - final | B_dmf_Deltamax_all_over_dw | dev - final | Delta W (dimensionless) | 20 | -2.009e-05 | -0.0001564 | 0 | -0.000261 | 3.275e-05 |  |
| 50 | dev - final difference of the tier-dependent reported metrics | dFull / dW: dev - final | B_dmf_dFull_over_dw | dev - final | Delta W (dimensionless) | 20 | -0.0001038 | -0.000206 | -1.679e-05 | -0.0002729 | 3.275e-05 |  |
| 50 | drift test of the frozen stage-2 snapshot | max \|snapshot - end-of-A mapping\|, Beta mean | drift_maxabs_mean | n/a | effort units [0, 100] | 20 | 0 | 0 | 0 | 0 | 0 |  |
| 50 | drift test of the frozen stage-2 snapshot | max \|snapshot - end-of-A mapping\|, alpha | drift_maxabs_alpha | n/a | Beta parameter units | 20 | 0 | 0 | 0 | 0 | 0 |  |
| 50 | drift test of the frozen stage-2 snapshot | max \|snapshot - end-of-A mapping\|, beta | drift_maxabs_beta | n/a | Beta parameter units | 20 | 0 | 0 | 0 | 0 | 0 |  |
| 50 | location (t*, d*) of Gmax_full (final tier) | runs with t* = 1 (root, d = 0) | B_Gmax_full_t | final | count of runs | 20 |  |  |  |  |  | 5 |
| 50 | location (t*, d*) of Gmax_full (final tier) | runs with t* = 2 | B_Gmax_full_t | final | count of runs | 20 |  |  |  |  |  | 15 |
| 50 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (1, 0) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 5 |
| 50 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (2, -82) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 1 |
| 50 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (2, -62) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 1 |
| 50 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (2, -38) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 1 |
| 50 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (2, -34) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 1 |
| 50 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (2, -32) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 1 |
| 50 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (2, -16) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 2 |
| 50 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (2, -14) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 1 |
| 50 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (2, -2) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 7 |
| 50 | stage-1 decomposition (residual band, final tier) | runs whose learning-term band interval contains 0 | dec_learning_contains_0 | final | count of runs | 20 |  |  |  |  |  | 0 |
| 50 | stage-1 decomposition (residual band, final tier) | runs whose inherited-term band interval contains 0 | dec_inherited_contains_0 | final | count of runs | 20 |  |  |  |  |  | 5 |
| 50 | stage-1 decomposition (residual band, final tier) | runs whose induced band is contiguous | dec_band_contiguous | final | count of runs | 20 |  |  |  |  |  | 4 |
| 50 | stage-1 decomposition (residual band, final tier) | runs with e_hat_1(0) inside the sweep range | dec_e1_inside_sweep | final | count of runs | 20 |  |  |  |  |  | 20 |
| 50 | stage-1 decomposition (residual band, final tier) | runs whose Delta_1 argmin lies at the sweep edge | dec_argmin_at_sweep_edge | final | count of runs | 20 |  |  |  |  |  | 0 |
| 50 | drift test of the frozen stage-2 snapshot | runs passing the drift test | drift_test_pass | n/a | count of runs | 20 |  |  |  |  |  | 20 |
| 50 | drift test of the frozen stage-2 snapshot | runs whose snapshot parameters are bit-identical to the end-of-A actor | drift_snapshot_bit_identical | n/a | count of runs | 20 |  |  |  |  |  | 20 |
| 60 | stage 2, end of Phase A (u1600) | peak error at d = 0, signed | A_stage2_peak_rel_err_signed | tier-independent | fraction of e2*(0) | 20 | -0.05948 | -0.07755 | -0.04827 | -0.1087 | -0.02498 |  |
| 60 | stage 2, end of Phase A (u1600) | location-free peak error, (max_d e_hat_2(d) - e2*(0))/e2*(0) | A_stage2_peak_locfree_rel_err | tier-independent | fraction of e2*(0) | 20 | -0.05928 | -0.07672 | -0.04827 | -0.1087 | -0.02054 |  |
| 60 | stage 2, end of Phase A (u1600) | argmax d of e_hat_2 on the recovery grid (location-free peak) | A_stage2_peak_locfree_argmax_d | tier-independent | effort units (gap d) | 20 | -0.25 | -1 | 0.625 | -4.5 | 2 |  |
| 60 | stage 2, end of Phase A (u1600) | symmetry error max_d \|e_hat_2(d) - e_hat_2(-d)\| / e2*(0) | A_stage2_sym_err_max_over_g2_0 | tier-independent | fraction of e2*(0) | 20 | 0.02759 | 0.02152 | 0.03511 | 0.009556 | 0.06898 |  |
| 60 | stage 2, end of Phase A (u1600) | symmetry error, raw | A_stage2_sym_err_max | tier-independent | effort units, raw | 20 | 1.609 | 1.255 | 2.048 | 0.5574 | 4.024 |  |
| 60 | stage 2, end of Phase A (u1600) | tail max of e_hat_2 on \|d\| >= 2q / e2*(0) | A_stage2_tail_max_over_g2_0 | tier-independent | fraction of e2*(0) | 20 | 0.03626 | 0.03366 | 0.04126 | 0.02796 | 0.06424 |  |
| 60 | stage 2, end of Phase A (u1600) | tail max, raw | A_stage2_tail_max | tier-independent | effort units [0, 100], raw | 20 | 2.115 | 1.964 | 2.407 | 1.631 | 3.748 |  |
| 60 | stage 2, end of Phase A (u1600) | Delta_2 on path, max / dW | A_DeltaT_over_dw_on_max | final | Delta W (dimensionless) | 20 | 0.0007876 | 0.0005148 | 0.001252 | 0.0004143 | 0.002318 |  |
| 60 | stage 2, end of Phase A (u1600) | Delta_2 off path, max / dW | A_DeltaT_over_dw_off_max | final | Delta W (dimensionless) | 20 | 0.0002942 | 0.0002491 | 0.0003846 | 0.0001501 | 0.0008322 |  |
| 60 | stage 2, end of Phase A (u1600) | sigma_2(0), SD of the stage-2 Beta action at d = 0 | A_sigma_effort_at_0_t2 | tier-independent | effort units [0, 100] | 20 | 2.941 | 2.839 | 3.1 | 2.771 | 3.538 |  |
| 60 | stage 2, end of Phase A (u1600) | smoothed-game share of the d = 0 peak gap | A_smoothed_share_peak_gap_d0 | tier-independent | fraction of the peak gap e2*(0) - e_hat_2(0) | 20 | 0.4546 | 0.3833 | 0.5773 | 0.2981 | 1.048 |  |
| 60 | stage 1 and full policy, end of Phase B (u2200) | e_hat_1(0), raw | B_e1_at_0 | tier-independent | effort units [0, 100], raw | 20 | 38.82 | 37.39 | 40.71 | 34.48 | 44.84 |  |
| 60 | stage 1 and full policy, end of Phase B (u2200) | stage-1 error (e_hat_1(0) - e1*(0))/e1*(0), signed | B_stage1_rel_err_signed | tier-independent | fraction of e1*(0) | 20 | -0.00175 | -0.03856 | 0.0468 | -0.1135 | 0.153 |  |
| 60 | stage 1 and full policy, end of Phase B (u2200) | EXP_root / dW | B_EXP_root_over_dw | final | Delta W (dimensionless) | 20 | 0.0004513 | 0.0002835 | 0.0006757 | 9.253e-05 | 0.003638 |  |
| 60 | stage 1 and full policy, end of Phase B (u2200) | dReach / dW | B_dReach_over_dw | final | Delta W (dimensionless) | 20 | 0.001194 | 0.0008344 | 0.001643 | 0.0005165 | 0.005505 |  |
| 60 | stage 1 and full policy, end of Phase B (u2200) | Delta_max_all / dW | B_Deltamax_all_over_dw | final | Delta W (dimensionless) | 20 | 0.0008897 | 0.0005287 | 0.001254 | 0.0004322 | 0.003187 |  |
| 60 | stage 1 and full policy, end of Phase B (u2200) | dFull / dW | B_dFull_over_dw | final | Delta W (dimensionless) | 20 | 0.001194 | 0.0009113 | 0.001643 | 0.0005165 | 0.005505 |  |
| 60 | stage 1 and full policy, end of Phase B (u2200) | sigma_1(0), SD of the stage-1 Beta action at d = 0 | B_sigma_effort_at_0_t1 | tier-independent | effort units [0, 100] | 20 | 2.942 | 2.73 | 3.089 | 2.631 | 3.596 |  |
| 60 | stage-1 decomposition (residual band, final tier) | learning term (e_hat_1(0) - e~1)/e1*(0), signed | dec_learning_rel | final | fraction of e1*(0) | 20 | -0.002007 | -0.02811 | 0.05171 | -0.1091 | 0.1553 |  |
| 60 | stage-1 decomposition (residual band, final tier) | inherited term (e~1 - e1*(0))/e1*(0), signed | dec_inherited_rel | final | fraction of e1*(0) | 20 | -0.004243 | -0.006493 | -0.001864 | -0.018 | 0.005914 |  |
| 60 | dev - final difference of the tier-dependent reported metrics | Delta_2 on path, max / dW: dev - final | A_dmf_DeltaT_over_dw_on_max | dev - final | Delta W (dimensionless) | 20 | -9.087e-05 | -0.0001648 | -3.05e-05 | -0.000225 | 1.11e-16 |  |
| 60 | dev - final difference of the tier-dependent reported metrics | Delta_2 off path, max / dW: dev - final | A_dmf_DeltaT_over_dw_off_max | dev - final | Delta W (dimensionless) | 20 | 0 | -2.072e-11 | 0 | -7.122e-05 | 0 |  |
| 60 | dev - final difference of the tier-dependent reported metrics | EXP_root / dW: dev - final | B_dmf_EXP_root_over_dw | dev - final | Delta W (dimensionless) | 20 | 1.709e-05 | 7.159e-06 | 2.339e-05 | -3.174e-06 | 2.991e-05 |  |
| 60 | dev - final difference of the tier-dependent reported metrics | dReach / dW: dev - final | B_dmf_dReach_over_dw | dev - final | Delta W (dimensionless) | 20 | -5.213e-05 | -0.0001329 | 3.011e-07 | -0.0001985 | 2.943e-05 |  |
| 60 | dev - final difference of the tier-dependent reported metrics | Delta_max_all / dW: dev - final | B_dmf_Deltamax_all_over_dw | dev - final | Delta W (dimensionless) | 20 | -7.188e-06 | -9.822e-05 | 1.037e-06 | -0.000225 | 2.029e-05 |  |
| 60 | dev - final difference of the tier-dependent reported metrics | dFull / dW: dev - final | B_dmf_dFull_over_dw | dev - final | Delta W (dimensionless) | 20 | -5.213e-05 | -0.0001329 | 3.011e-07 | -0.0001985 | 2.943e-05 |  |
| 60 | drift test of the frozen stage-2 snapshot | max \|snapshot - end-of-A mapping\|, Beta mean | drift_maxabs_mean | n/a | effort units [0, 100] | 20 | 0 | 0 | 0 | 0 | 0 |  |
| 60 | drift test of the frozen stage-2 snapshot | max \|snapshot - end-of-A mapping\|, alpha | drift_maxabs_alpha | n/a | Beta parameter units | 20 | 0 | 0 | 0 | 0 | 0 |  |
| 60 | drift test of the frozen stage-2 snapshot | max \|snapshot - end-of-A mapping\|, beta | drift_maxabs_beta | n/a | Beta parameter units | 20 | 0 | 0 | 0 | 0 | 0 |  |
| 60 | location (t*, d*) of Gmax_full (final tier) | runs with t* = 1 (root, d = 0) | B_Gmax_full_t | final | count of runs | 20 |  |  |  |  |  | 7 |
| 60 | location (t*, d*) of Gmax_full (final tier) | runs with t* = 2 | B_Gmax_full_t | final | count of runs | 20 |  |  |  |  |  | 13 |
| 60 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (1, 0) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 7 |
| 60 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (2, -120) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 1 |
| 60 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (2, -20) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 1 |
| 60 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (2, -18) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 2 |
| 60 | location (t*, d*) of Gmax_full (final tier) | runs with (t*, d*) = (2, -2) | B_Gmax_full_t, B_Gmax_full_d | final | count of runs | 20 |  |  |  |  |  | 9 |
| 60 | stage-1 decomposition (residual band, final tier) | runs whose learning-term band interval contains 0 | dec_learning_contains_0 | final | count of runs | 20 |  |  |  |  |  | 0 |
| 60 | stage-1 decomposition (residual band, final tier) | runs whose inherited-term band interval contains 0 | dec_inherited_contains_0 | final | count of runs | 20 |  |  |  |  |  | 2 |
| 60 | stage-1 decomposition (residual band, final tier) | runs whose induced band is contiguous | dec_band_contiguous | final | count of runs | 20 |  |  |  |  |  | 20 |
| 60 | stage-1 decomposition (residual band, final tier) | runs with e_hat_1(0) inside the sweep range | dec_e1_inside_sweep | final | count of runs | 20 |  |  |  |  |  | 20 |
| 60 | stage-1 decomposition (residual band, final tier) | runs whose Delta_1 argmin lies at the sweep edge | dec_argmin_at_sweep_edge | final | count of runs | 20 |  |  |  |  |  | 0 |
| 60 | drift test of the frozen stage-2 snapshot | runs passing the drift test | drift_test_pass | n/a | count of runs | 20 |  |  |  |  |  | 20 |
| 60 | drift test of the frozen stage-2 snapshot | runs whose snapshot parameters are bit-identical to the end-of-A actor | drift_snapshot_bit_identical | n/a | count of runs | 20 |  |  |  |  |  | 20 |
