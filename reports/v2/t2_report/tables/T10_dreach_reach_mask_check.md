# T10: dReach reach-mask check

- priority: supp; status: found; tier: final and development
- source file: `results/v2_pilots/dreach_mask_check/dreach_mask_check.csv` (sha256 `28fb832bd0db...`; byte-identical copy)
- built by: `tools/v2/report/sec_p0p1.py:build_t10`; base commit `cb0b541`
- transformation: summary block in the .md computed from the same CSV (groupby q, tier); tier per row in column tier

Summary (computed from the CSV below):

| q | tier | evaluations | valid | total_holes | with_0_extras | with_1_extra | with_2_extras | checkpoint_evaluations | checkpoints_with_2_extras | dreach_diff_nonzero | dreach_diff_max_over_dw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | development | 72 | 72 | 0 | 69 | 0 | 3 | 70 | 2 | 1 | 0.001408 |
| 50 | final | 72 | 72 | 0 | 71 | 0 | 1 | 70 | 0 | 0 | 0 |
| 60 | development | 12 | 12 | 0 | 11 | 0 | 1 | 10 | 0 | 0 | 0 |
| 60 | final | 12 | 12 | 0 | 12 | 0 | 0 | 10 | 0 | 0 | 0 |

- evaluations: 168 (168 valid); candidates: analytic, zero and 80 legacy checkpoints, each on 2 tiers
- extra nodes in total: 10; distinct distances to the support edge: 0
- (official - reference-mask) dReach/DW: exactly 0 in 167 evaluations; nonzero in 1; negative in 0
- nonzero case: two_stage_q50_restarts_20260924/tel_q50_s10224 (development tier): difference 0.001408; drift_BR 0.000000; max Delta_2/DW over R_2 0.009661 vs over the reference support 0.008253; official dReach/DW 0.009667 vs reference-mask 0.008259
- the same checkpoint on the final tier: drift_BR -0.108828, 0 extra nodes, difference 0

| candidate | q | tier | valid | n_grid | a_br_1 | e_hat_1 | drift_BR | n_R2 | n_ref | n_holes | n_extra | holes_d | extra_d | extra_dist_to_support_edge | dreach_official_over_dw | dreach_refmask_over_dw | dreach_official_minus_refmask_over_dw | max_delta2_R2_over_dw | max_delta2_ref_over_dw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| analytic | 50 | development | True | 101 | 46.67 | 46.67 | 0 | 51 | 49 | 0 | 2 |  | -100 100 | 0 0 | 2.22e-16 | 2.22e-16 | 0 | 2.22e-16 | 2.22e-16 |
| analytic | 50 | final | True | 201 | 46.67 | 46.67 | 0 | 101 | 99 | 0 | 2 |  | -100 100 | 0 0 | 2.22e-16 | 2.22e-16 | 0 | 2.22e-16 | 2.22e-16 |
| zero | 50 | development | True | 101 | 29.21 | 0 | 29.21 | 50 | 50 | 0 | 0 |  |  |  | 0.3934 | 0.3934 | 0 | 0.259 | 0.259 |
| zero | 50 | final | True | 201 | 29.28 | 0 | 29.28 | 100 | 100 | 0 | 0 |  |  |  | 0.3938 | 0.3938 | 0 | 0.2593 | 0.2593 |
| analytic | 60 | development | True | 111 | 38.89 | 38.89 | 0 | 61 | 59 | 0 | 2 |  | -120 120 | 0 0 | 2.22e-16 | 2.22e-16 | 0 | 2.22e-16 | 2.22e-16 |
| analytic | 60 | final | True | 221 | 38.83 | 38.89 | -0.05509 | 120 | 120 | 0 | 0 |  |  |  | 3.511e-07 | 3.511e-07 | 0 | 2.22e-16 | 2.22e-16 |
| zero | 60 | development | True | 111 | 29.27 | 0 | 29.27 | 60 | 60 | 0 | 0 |  |  |  | 0.2951 | 0.2951 | 0 | 0.1955 | 0.1955 |
| zero | 60 | final | True | 221 | 29.27 | 0 | 29.27 | 120 | 120 | 0 | 0 |  |  |  | 0.2951 | 0.2951 | 0 | 0.1955 | 0.1955 |
| two_stage_confirmation_T2_20260922/tel_q50_s10101 | 50 | development | True | 101 | 47.74 | 45.62 | 2.118 | 50 | 50 | 0 | 0 |  |  |  | 0.008798 | 0.008798 | 0 | 0.008472 | 0.008472 |
| two_stage_confirmation_T2_20260922/tel_q50_s10101 | 50 | final | True | 201 | 47.8 | 45.62 | 2.18 | 100 | 100 | 0 | 0 |  |  |  | 0.008851 | 0.008851 | 0 | 0.008589 | 0.008589 |
| two_stage_confirmation_T2_20260922/tel_q50_s10102 | 50 | development | True | 101 | 46.64 | 46.81 | -0.1687 | 50 | 50 | 0 | 0 |  |  |  | 0.009423 | 0.009423 | 0 | 0.009423 | 0.009423 |
| two_stage_confirmation_T2_20260922/tel_q50_s10102 | 50 | final | True | 201 | 46.5 | 46.81 | -0.3098 | 100 | 100 | 0 | 0 |  |  |  | 0.009425 | 0.009425 | 0 | 0.009423 | 0.009423 |
| two_stage_confirmation_T2_20260922/tel_q50_s10103 | 50 | development | True | 101 | 42.36 | 48.78 | -6.413 | 50 | 50 | 0 | 0 |  |  |  | 0.007626 | 0.007626 | 0 | 0.005824 | 0.005824 |
| two_stage_confirmation_T2_20260922/tel_q50_s10103 | 50 | final | True | 201 | 42.34 | 48.78 | -6.435 | 100 | 100 | 0 | 0 |  |  |  | 0.007736 | 0.007736 | 0 | 0.005976 | 0.005976 |
| two_stage_confirmation_T2_20260922/tel_q50_s10104 | 50 | development | True | 101 | 45.63 | 49.76 | -4.129 | 50 | 50 | 0 | 0 |  |  |  | 0.02425 | 0.02425 | 0 | 0.02352 | 0.02352 |
| two_stage_confirmation_T2_20260922/tel_q50_s10104 | 50 | final | True | 201 | 45.42 | 49.76 | -4.339 | 100 | 100 | 0 | 0 |  |  |  | 0.02421 | 0.02421 | 0 | 0.02352 | 0.02352 |
| two_stage_confirmation_T2_20260922/tel_q50_s10105 | 50 | development | True | 101 | 47.48 | 37.65 | 9.833 | 50 | 50 | 0 | 0 |  |  |  | 0.02858 | 0.02858 | 0 | 0.02217 | 0.02217 |
| two_stage_confirmation_T2_20260922/tel_q50_s10105 | 50 | final | True | 201 | 47.48 | 37.65 | 9.826 | 100 | 100 | 0 | 0 |  |  |  | 0.02858 | 0.02858 | 0 | 0.02217 | 0.02217 |
| two_stage_confirmation_T2_20260922/tel_q50_s10106 | 50 | development | True | 101 | 47 | 46.24 | 0.7585 | 50 | 50 | 0 | 0 |  |  |  | 0.007947 | 0.007947 | 0 | 0.007897 | 0.007897 |
| two_stage_confirmation_T2_20260922/tel_q50_s10106 | 50 | final | True | 201 | 46.61 | 46.24 | 0.3694 | 100 | 100 | 0 | 0 |  |  |  | 0.007911 | 0.007911 | 0 | 0.007897 | 0.007897 |
| two_stage_confirmation_T2_20260922/tel_q50_s10107 | 50 | development | True | 101 | 48.08 | 47.99 | 0.08571 | 50 | 50 | 0 | 0 |  |  |  | 0.03148 | 0.03148 | 0 | 0.03139 | 0.03139 |
| two_stage_confirmation_T2_20260922/tel_q50_s10107 | 50 | final | True | 201 | 48 | 47.99 | 0.00882 | 100 | 100 | 0 | 0 |  |  |  | 0.03148 | 0.03148 | 0 | 0.03146 | 0.03146 |
| two_stage_confirmation_T2_20260922/tel_q50_s10108 | 50 | development | True | 101 | 47.59 | 44.15 | 3.446 | 50 | 50 | 0 | 0 |  |  |  | 0.008838 | 0.008838 | 0 | 0.008094 | 0.008094 |
| two_stage_confirmation_T2_20260922/tel_q50_s10108 | 50 | final | True | 201 | 48.03 | 44.15 | 3.88 | 100 | 100 | 0 | 0 |  |  |  | 0.008904 | 0.008904 | 0 | 0.008154 | 0.008154 |
| two_stage_confirmation_T2_20260922/tel_q50_s10109 | 50 | development | True | 101 | 48 | 44.04 | 3.958 | 50 | 50 | 0 | 0 |  |  |  | 0.008178 | 0.008178 | 0 | 0.007449 | 0.007449 |
| two_stage_confirmation_T2_20260922/tel_q50_s10109 | 50 | final | True | 201 | 47.82 | 44.04 | 3.776 | 100 | 100 | 0 | 0 |  |  |  | 0.008185 | 0.008185 | 0 | 0.007449 | 0.007449 |
| two_stage_confirmation_T2_20260922/tel_q50_s10110 | 50 | development | True | 101 | 48.29 | 47.47 | 0.8206 | 50 | 50 | 0 | 0 |  |  |  | 0.009006 | 0.009006 | 0 | 0.00893 | 0.00893 |
| two_stage_confirmation_T2_20260922/tel_q50_s10110 | 50 | final | True | 201 | 48 | 47.47 | 0.5283 | 100 | 100 | 0 | 0 |  |  |  | 0.009047 | 0.009047 | 0 | 0.009021 | 0.009021 |
| two_stage_confirmation_T2_20260922/tel_q60_s10111 | 60 | development | True | 111 | 38 | 35.82 | 2.181 | 60 | 60 | 0 | 0 |  |  |  | 0.009871 | 0.009871 | 0 | 0.009453 | 0.009453 |
| two_stage_confirmation_T2_20260922/tel_q60_s10111 | 60 | final | True | 221 | 38.25 | 35.82 | 2.429 | 120 | 120 | 0 | 0 |  |  |  | 0.009923 | 0.009923 | 0 | 0.009504 | 0.009504 |
| two_stage_confirmation_T2_20260922/tel_q60_s10112 | 60 | development | True | 111 | 38 | 37.77 | 0.2279 | 60 | 60 | 0 | 0 |  |  |  | 0.009635 | 0.009635 | 0 | 0.009627 | 0.009627 |
| two_stage_confirmation_T2_20260922/tel_q60_s10112 | 60 | final | True | 221 | 38.02 | 37.77 | 0.2444 | 120 | 120 | 0 | 0 |  |  |  | 0.009814 | 0.009814 | 0 | 0.009808 | 0.009808 |
| two_stage_confirmation_T2_20260922/tel_q60_s10113 | 60 | development | True | 111 | 39.42 | 38.02 | 1.401 | 60 | 60 | 0 | 0 |  |  |  | 0.009294 | 0.009294 | 0 | 0.009139 | 0.009139 |
| two_stage_confirmation_T2_20260922/tel_q60_s10113 | 60 | final | True | 221 | 39.28 | 38.02 | 1.265 | 120 | 120 | 0 | 0 |  |  |  | 0.009514 | 0.009514 | 0 | 0.009387 | 0.009387 |
| two_stage_confirmation_T2_20260922/tel_q60_s10114 | 60 | development | True | 111 | 37.63 | 37.8 | -0.1705 | 60 | 60 | 0 | 0 |  |  |  | 0.007567 | 0.007567 | 0 | 0.007563 | 0.007563 |
| two_stage_confirmation_T2_20260922/tel_q60_s10114 | 60 | final | True | 221 | 37.5 | 37.8 | -0.3008 | 120 | 120 | 0 | 0 |  |  |  | 0.008714 | 0.008714 | 0 | 0.008708 | 0.008708 |
| two_stage_confirmation_T2_20260922/tel_q60_s10115 | 60 | development | True | 111 | 39.81 | 34.98 | 4.832 | 60 | 60 | 0 | 0 |  |  |  | 0.008202 | 0.008202 | 0 | 0.006754 | 0.006754 |
| two_stage_confirmation_T2_20260922/tel_q60_s10115 | 60 | final | True | 221 | 39.7 | 34.98 | 4.716 | 120 | 120 | 0 | 0 |  |  |  | 0.009113 | 0.009113 | 0 | 0.007679 | 0.007679 |
| two_stage_confirmation_T2_20260922/tel_q60_s10116 | 60 | development | True | 111 | 37.62 | 41.67 | -4.055 | 60 | 60 | 0 | 0 |  |  |  | 0.00949 | 0.00949 | 0 | 0.008519 | 0.008519 |
| two_stage_confirmation_T2_20260922/tel_q60_s10116 | 60 | final | True | 221 | 37.5 | 41.67 | -4.174 | 120 | 120 | 0 | 0 |  |  |  | 0.009483 | 0.009483 | 0 | 0.008519 | 0.008519 |
| two_stage_confirmation_T2_20260922/tel_q60_s10117 | 60 | development | True | 111 | 39.17 | 38.96 | 0.2141 | 60 | 60 | 0 | 0 |  |  |  | 0.0086 | 0.0086 | 0 | 0.008595 | 0.008595 |
| two_stage_confirmation_T2_20260922/tel_q60_s10117 | 60 | final | True | 221 | 39.07 | 38.96 | 0.1163 | 120 | 120 | 0 | 0 |  |  |  | 0.008693 | 0.008693 | 0 | 0.008692 | 0.008692 |
| two_stage_confirmation_T2_20260922/tel_q60_s10118 | 60 | development | True | 111 | 39.81 | 37.31 | 2.496 | 60 | 60 | 0 | 0 |  |  |  | 0.007346 | 0.007346 | 0 | 0.006912 | 0.006912 |
| two_stage_confirmation_T2_20260922/tel_q60_s10118 | 60 | final | True | 221 | 40 | 37.31 | 2.686 | 120 | 120 | 0 | 0 |  |  |  | 0.007341 | 0.007341 | 0 | 0.006912 | 0.006912 |
| two_stage_confirmation_T2_20260922/tel_q60_s10119 | 60 | development | True | 111 | 39 | 41.72 | -2.716 | 60 | 60 | 0 | 0 |  |  |  | 0.006742 | 0.006742 | 0 | 0.00622 | 0.00622 |
| two_stage_confirmation_T2_20260922/tel_q60_s10119 | 60 | final | True | 221 | 38.8 | 41.72 | -2.917 | 120 | 120 | 0 | 0 |  |  |  | 0.006724 | 0.006724 | 0 | 0.00622 | 0.00622 |
| two_stage_confirmation_T2_20260922/tel_q60_s10120 | 60 | development | True | 111 | 39.43 | 40.06 | -0.6296 | 60 | 60 | 0 | 0 |  |  |  | 0.00804 | 0.00804 | 0 | 0.008016 | 0.008016 |
| two_stage_confirmation_T2_20260922/tel_q60_s10120 | 60 | final | True | 221 | 39.44 | 40.06 | -0.6162 | 120 | 120 | 0 | 0 |  |  |  | 0.008094 | 0.008094 | 0 | 0.008062 | 0.008062 |
| two_stage_E1_q50_p_20260923/tel_q50_s10121 | 50 | development | True | 101 | 45.74 | 48.02 | -2.284 | 50 | 50 | 0 | 0 |  |  |  | 0.007265 | 0.007265 | 0 | 0.006995 | 0.006995 |
| two_stage_E1_q50_p_20260923/tel_q50_s10121 | 50 | final | True | 201 | 45.61 | 48.02 | -2.418 | 100 | 100 | 0 | 0 |  |  |  | 0.007237 | 0.007237 | 0 | 0.006995 | 0.006995 |
| two_stage_E1_q50_p_20260923/tel_q50_s10122 | 50 | development | True | 101 | 43.41 | 48.89 | -5.479 | 50 | 50 | 0 | 0 |  |  |  | 0.008945 | 0.008945 | 0 | 0.007965 | 0.007965 |
| two_stage_E1_q50_p_20260923/tel_q50_s10122 | 50 | final | True | 201 | 43.93 | 48.89 | -4.964 | 100 | 100 | 0 | 0 |  |  |  | 0.008928 | 0.008928 | 0 | 0.007965 | 0.007965 |
| two_stage_E1_q50_p_20260923/tel_q50_s10123 | 50 | development | True | 101 | 42.18 | 48.64 | -6.458 | 50 | 50 | 0 | 0 |  |  |  | 0.007956 | 0.007956 | 0 | 0.006583 | 0.006583 |
| two_stage_E1_q50_p_20260923/tel_q50_s10123 | 50 | final | True | 201 | 42.56 | 48.64 | -6.083 | 100 | 100 | 0 | 0 |  |  |  | 0.008109 | 0.008109 | 0 | 0.006755 | 0.006755 |
| two_stage_E1_q50_p_20260923/tel_q50_s10124 | 50 | development | True | 101 | 46.47 | 50.51 | -4.04 | 50 | 50 | 0 | 0 |  |  |  | 0.009259 | 0.009259 | 0 | 0.008469 | 0.008469 |
| two_stage_E1_q50_p_20260923/tel_q50_s10124 | 50 | final | True | 201 | 46.4 | 50.51 | -4.115 | 100 | 100 | 0 | 0 |  |  |  | 0.009266 | 0.009266 | 0 | 0.008489 | 0.008489 |
| two_stage_E1_q50_p_20260923/tel_q50_s10125 | 50 | development | True | 101 | 47.94 | 46.13 | 1.812 | 50 | 50 | 0 | 0 |  |  |  | 0.009037 | 0.009037 | 0 | 0.00875 | 0.00875 |
| two_stage_E1_q50_p_20260923/tel_q50_s10125 | 50 | final | True | 201 | 48.34 | 46.13 | 2.216 | 100 | 100 | 0 | 0 |  |  |  | 0.009139 | 0.009139 | 0 | 0.008848 | 0.008848 |
| two_stage_E1_q50_p_20260923/tel_q50_s10126 | 50 | development | True | 101 | 46.66 | 41.7 | 4.964 | 50 | 50 | 0 | 0 |  |  |  | 0.007867 | 0.007867 | 0 | 0.006694 | 0.006694 |
| two_stage_E1_q50_p_20260923/tel_q50_s10126 | 50 | final | True | 201 | 46.5 | 41.7 | 4.802 | 100 | 100 | 0 | 0 |  |  |  | 0.007854 | 0.007854 | 0 | 0.006694 | 0.006694 |
| ... 108 more rows in the CSV | | | | | | | | | | | | | | | | | | | |
