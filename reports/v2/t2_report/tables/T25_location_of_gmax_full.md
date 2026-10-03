# T25: Location of Gmax_full

- priority: supp; status: generated; tier: final and development
- sources: `results/v2_pilots/pilot2/analysis/gmax_location_final.csv`, `results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv`, `results/v2_pilots/pilot2/analysis/final_table.csv`, `Pilot 2 final_v2.json (60 runs)` (60 files), `Pilot 2 v2_checkpoints.csv (60 runs)` (60 files)
- built by: `tools/v2/report/sec_pilot23.py:build_t25`; base commit `cb0b541`
- transformation: Location classes: stage1 (t* = 1, always d* = 0), stage2_onpath (t* = 2, |d*| < 2q; the root drift is 0 in every checkpoint), stage2_offpath (t* = 2, |d*| >= 2q). Rows with a gmax_location_*.csv source are the existing counts (development tier); recounting them from final_table.csv and the 60 v2_checkpoints.csv files gives max abs diff 0. Final-tier classes and all exact (t*, d*) pairs are generated from final_table.csv (development tier) and final_v2.json['final'] (final tier; d* on the 2-step grid). The (t*, d*) of the two tiers differ in 22 of 60 runs: t* in 0, d* by at most 2 effort units; the location class differs in 1 run(s).

Where Gmax_full is attained in Pilot 2: counts per (q, arm), n = 10 runs (seeds 10501-10510) at u1000 or 210 training-time checkpoints per (q, arm).

| block | q | arm | arm_short | tier | scope | location | t_star | d_star | count | n | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| location class | 50 | A_joint | A | development | final checkpoint u1000 | stage1 |  |  | 4 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 50 | A_joint | A | development | final checkpoint u1000 | stage2_offpath |  |  | 2 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 50 | A_joint | A | development | final checkpoint u1000 | stage2_onpath |  |  | 4 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 50 | B1_frozen_allnorm | B1 | development | final checkpoint u1000 | stage1 |  |  | 0 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 50 | B1_frozen_allnorm | B1 | development | final checkpoint u1000 | stage2_offpath |  |  | 1 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 50 | B1_frozen_allnorm | B1 | development | final checkpoint u1000 | stage2_onpath |  |  | 9 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 50 | B2_frozen_s1norm | B2 | development | final checkpoint u1000 | stage1 |  |  | 1 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 50 | B2_frozen_s1norm | B2 | development | final checkpoint u1000 | stage2_offpath |  |  | 0 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 50 | B2_frozen_s1norm | B2 | development | final checkpoint u1000 | stage2_onpath |  |  | 9 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 60 | A_joint | A | development | final checkpoint u1000 | stage1 |  |  | 3 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 60 | A_joint | A | development | final checkpoint u1000 | stage2_offpath |  |  | 6 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 60 | A_joint | A | development | final checkpoint u1000 | stage2_onpath |  |  | 1 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 60 | B1_frozen_allnorm | B1 | development | final checkpoint u1000 | stage1 |  |  | 2 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 60 | B1_frozen_allnorm | B1 | development | final checkpoint u1000 | stage2_offpath |  |  | 0 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 60 | B1_frozen_allnorm | B1 | development | final checkpoint u1000 | stage2_onpath |  |  | 8 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 60 | B2_frozen_s1norm | B2 | development | final checkpoint u1000 | stage1 |  |  | 0 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 60 | B2_frozen_s1norm | B2 | development | final checkpoint u1000 | stage2_offpath |  |  | 0 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 60 | B2_frozen_s1norm | B2 | development | final checkpoint u1000 | stage2_onpath |  |  | 10 | 10 | results/v2_pilots/pilot2/analysis/gmax_location_final.csv |
| location class | 50 | A_joint | A | development | 21 training-time checkpoints u500-u1000 | stage1 |  |  | 51 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 50 | A_joint | A | development | 21 training-time checkpoints u500-u1000 | stage2_offpath |  |  | 58 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 50 | A_joint | A | development | 21 training-time checkpoints u500-u1000 | stage2_onpath |  |  | 101 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 50 | B1_frozen_allnorm | B1 | development | 21 training-time checkpoints u500-u1000 | stage1 |  |  | 34 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 50 | B1_frozen_allnorm | B1 | development | 21 training-time checkpoints u500-u1000 | stage2_offpath |  |  | 9 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 50 | B1_frozen_allnorm | B1 | development | 21 training-time checkpoints u500-u1000 | stage2_onpath |  |  | 167 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 50 | B2_frozen_s1norm | B2 | development | 21 training-time checkpoints u500-u1000 | stage1 |  |  | 39 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 50 | B2_frozen_s1norm | B2 | development | 21 training-time checkpoints u500-u1000 | stage2_offpath |  |  | 10 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 50 | B2_frozen_s1norm | B2 | development | 21 training-time checkpoints u500-u1000 | stage2_onpath |  |  | 161 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 60 | A_joint | A | development | 21 training-time checkpoints u500-u1000 | stage1 |  |  | 77 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 60 | A_joint | A | development | 21 training-time checkpoints u500-u1000 | stage2_offpath |  |  | 78 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 60 | A_joint | A | development | 21 training-time checkpoints u500-u1000 | stage2_onpath |  |  | 55 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 60 | B1_frozen_allnorm | B1 | development | 21 training-time checkpoints u500-u1000 | stage1 |  |  | 57 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 60 | B1_frozen_allnorm | B1 | development | 21 training-time checkpoints u500-u1000 | stage2_offpath |  |  | 0 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 60 | B1_frozen_allnorm | B1 | development | 21 training-time checkpoints u500-u1000 | stage2_onpath |  |  | 153 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 60 | B2_frozen_s1norm | B2 | development | 21 training-time checkpoints u500-u1000 | stage1 |  |  | 64 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 60 | B2_frozen_s1norm | B2 | development | 21 training-time checkpoints u500-u1000 | stage2_offpath |  |  | 0 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 60 | B2_frozen_s1norm | B2 | development | 21 training-time checkpoints u500-u1000 | stage2_onpath |  |  | 146 | 210 | results/v2_pilots/pilot2/analysis/gmax_location_all_checkpoints.csv |
| location class | 50 | A_joint | A | final | final checkpoint u1000 | stage1 |  |  | 4 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 50 | A_joint | A | final | final checkpoint u1000 | stage2_offpath |  |  | 2 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 50 | A_joint | A | final | final checkpoint u1000 | stage2_onpath |  |  | 4 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 50 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage1 |  |  | 0 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 50 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage2_offpath |  |  | 1 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 50 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage2_onpath |  |  | 9 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 50 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage1 |  |  | 1 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 50 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage2_offpath |  |  | 0 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 50 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage2_onpath |  |  | 9 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 60 | A_joint | A | final | final checkpoint u1000 | stage1 |  |  | 3 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 60 | A_joint | A | final | final checkpoint u1000 | stage2_offpath |  |  | 5 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 60 | A_joint | A | final | final checkpoint u1000 | stage2_onpath |  |  | 2 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 60 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage1 |  |  | 2 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 60 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage2_offpath |  |  | 0 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 60 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage2_onpath |  |  | 8 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 60 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage1 |  |  | 0 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 60 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage2_offpath |  |  | 0 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| location class | 60 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage2_onpath |  |  | 10 | 10 | final_v2.json['final'] (Gmax_full_t, Gmax_full_d) |
| (t*, d*) pair | 50 | A_joint | A | development | final checkpoint u1000 | stage1 | 1 | 0 | 4 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 50 | A_joint | A | development | final checkpoint u1000 | stage2_offpath | 2 | -104 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 50 | A_joint | A | development | final checkpoint u1000 | stage2_offpath | 2 | -100 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 50 | A_joint | A | development | final checkpoint u1000 | stage2_onpath | 2 | -96 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 50 | A_joint | A | development | final checkpoint u1000 | stage2_onpath | 2 | -92 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 50 | A_joint | A | development | final checkpoint u1000 | stage2_onpath | 2 | -76 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 50 | A_joint | A | development | final checkpoint u1000 | stage2_onpath | 2 | -72 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 50 | B1_frozen_allnorm | B1 | development | final checkpoint u1000 | stage2_onpath | 2 | -4 | 9 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 50 | B1_frozen_allnorm | B1 | development | final checkpoint u1000 | stage2_offpath | 2 | 100 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 50 | B2_frozen_s1norm | B2 | development | final checkpoint u1000 | stage2_onpath | 2 | -4 | 9 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 50 | B2_frozen_s1norm | B2 | development | final checkpoint u1000 | stage1 | 1 | 0 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 60 | A_joint | A | development | final checkpoint u1000 | stage2_offpath | 2 | -120 | 4 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 60 | A_joint | A | development | final checkpoint u1000 | stage1 | 1 | 0 | 3 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 60 | A_joint | A | development | final checkpoint u1000 | stage2_offpath | 2 | -124 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 60 | A_joint | A | development | final checkpoint u1000 | stage2_onpath | 2 | -4 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 60 | A_joint | A | development | final checkpoint u1000 | stage2_offpath | 2 | 124 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 60 | B1_frozen_allnorm | B1 | development | final checkpoint u1000 | stage2_onpath | 2 | -4 | 7 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 60 | B1_frozen_allnorm | B1 | development | final checkpoint u1000 | stage1 | 1 | 0 | 2 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 60 | B1_frozen_allnorm | B1 | development | final checkpoint u1000 | stage2_onpath | 2 | -32 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 60 | B2_frozen_s1norm | B2 | development | final checkpoint u1000 | stage2_onpath | 2 | -4 | 8 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 60 | B2_frozen_s1norm | B2 | development | final checkpoint u1000 | stage2_onpath | 2 | -32 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 60 | B2_frozen_s1norm | B2 | development | final checkpoint u1000 | stage2_onpath | 2 | 40 | 1 | 10 | results/v2_pilots/pilot2/analysis/final_table.csv |
| (t*, d*) pair | 50 | A_joint | A | final | final checkpoint u1000 | stage1 | 1 | 0 | 4 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 50 | A_joint | A | final | final checkpoint u1000 | stage2_offpath | 2 | -102 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 50 | A_joint | A | final | final checkpoint u1000 | stage2_offpath | 2 | -100 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 50 | A_joint | A | final | final checkpoint u1000 | stage2_onpath | 2 | -98 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 50 | A_joint | A | final | final checkpoint u1000 | stage2_onpath | 2 | -92 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 50 | A_joint | A | final | final checkpoint u1000 | stage2_onpath | 2 | -78 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 50 | A_joint | A | final | final checkpoint u1000 | stage2_onpath | 2 | -72 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 50 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage2_onpath | 2 | -4 | 8 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 50 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage2_onpath | 2 | -6 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 50 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage2_offpath | 2 | 102 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 50 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage2_onpath | 2 | -4 | 8 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 50 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage1 | 1 | 0 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 50 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage2_onpath | 2 | -6 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | A_joint | A | final | final checkpoint u1000 | stage1 | 1 | 0 | 3 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | A_joint | A | final | final checkpoint u1000 | stage2_offpath | 2 | -120 | 2 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | A_joint | A | final | final checkpoint u1000 | stage2_offpath | 2 | -124 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | A_joint | A | final | final checkpoint u1000 | stage2_offpath | 2 | -122 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | A_joint | A | final | final checkpoint u1000 | stage2_onpath | 2 | -118 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | A_joint | A | final | final checkpoint u1000 | stage2_onpath | 2 | -4 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | A_joint | A | final | final checkpoint u1000 | stage2_offpath | 2 | 122 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage2_onpath | 2 | -2 | 5 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage1 | 1 | 0 | 2 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage2_onpath | 2 | -4 | 2 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | B1_frozen_allnorm | B1 | final | final checkpoint u1000 | stage2_onpath | 2 | -30 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage2_onpath | 2 | -2 | 6 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage2_onpath | 2 | -4 | 2 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage2_onpath | 2 | -30 | 1 | 10 | final_v2.json['final'] |
| (t*, d*) pair | 60 | B2_frozen_s1norm | B2 | final | final checkpoint u1000 | stage2_onpath | 2 | 40 | 1 | 10 | final_v2.json['final'] |
