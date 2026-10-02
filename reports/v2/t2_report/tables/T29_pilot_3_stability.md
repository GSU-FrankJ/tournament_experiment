# T29: Pilot 3 stability

- priority: core; status: generated; tier: tier-independent
- sources: `results/v2_pilots/pilot3/analysis/final_table.csv`, `results/v2_pilots/pilot3/analysis/stability.csv`, `results/v2_pilots/pilot3/analysis/curves_weights_every25.csv`
- built by: `tools/v2/report/sec_pilot23.py:build_t29`; base commit `cb0b541`
- transformation: Per (q, arm) over the 10 seeds (10501-10510): median, q25, q75 (numpy linear interpolation), min, max, n, sd (ddof = 1) and iqr = q75 - q25 of: within-run SD (ddof = 1) and range of e_hat_1(0) over the 5 weight exports u900, 925, 950, 975, 1000 (final_table.csv within_run_*; recomputed from curves_weights_every25.csv e1_at_0: max abs diff 4.9e-15 / 7.1e-15); the final e_hat_1(0) (u1000; its sd and iqr are the across-seed SD and IQR); sigma_1(0) (u1000). The cells (e1_at_0, sd), (e1_at_0, iqr), (within_run_sd_e1_last5, median), (within_run_range_e1_last5, median), (sigma_effort_at_0_t1, median) reproduce results/v2_pilots/pilot3/analysis/stability.csv: max abs diff 0. Note: stability.csv computes the IQR with pandas quantile (linear), the same as numpy linear interpolation.

Pilot 3 stage-1 stability per q and arm, n = 10 seeds (10501-10510); all values in effort units (raw).

| q | arm | arm_short | metric | tier | median | q25 | q75 | min | max | n | sd | iqr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | B2_frozen_s1norm | stochastic | within_run_sd_e1_last5 | tier-independent | 1.947 | 1.435 | 2.416 | 1.273 | 3.121 | 10 | 0.6396 | 0.9816 |
| 50 | B2_frozen_s1norm | stochastic | within_run_range_e1_last5 | tier-independent | 5.004 | 3.701 | 5.908 | 2.665 | 6.718 | 10 | 1.415 | 2.207 |
| 50 | B2_frozen_s1norm | stochastic | e1_at_0 | tier-independent | 47.58 | 43.3 | 48.48 | 42.23 | 51.7 | 10 | 3.506 | 5.178 |
| 50 | B2_frozen_s1norm | stochastic | sigma_effort_at_0_t1 | tier-independent | 3.646 | 3.407 | 3.859 | 3.274 | 4.107 | 10 | 0.2771 | 0.4522 |
| 50 | B2_frozen_s1norm_mean | mean | within_run_sd_e1_last5 | tier-independent | 1.044 | 0.7452 | 1.796 | 0.5107 | 2.36 | 10 | 0.6795 | 1.051 |
| 50 | B2_frozen_s1norm_mean | mean | within_run_range_e1_last5 | tier-independent | 2.378 | 1.744 | 4.614 | 1.282 | 6.279 | 10 | 1.878 | 2.871 |
| 50 | B2_frozen_s1norm_mean | mean | e1_at_0 | tier-independent | 45.34 | 43.05 | 46.98 | 40.06 | 51.57 | 10 | 3.405 | 3.934 |
| 50 | B2_frozen_s1norm_mean | mean | sigma_effort_at_0_t1 | tier-independent | 3.624 | 3.436 | 3.747 | 3.298 | 4.195 | 10 | 0.2637 | 0.3105 |
| 60 | B2_frozen_s1norm | stochastic | within_run_sd_e1_last5 | tier-independent | 1.802 | 1.257 | 2.187 | 0.6852 | 3.488 | 10 | 0.8261 | 0.9303 |
| 60 | B2_frozen_s1norm | stochastic | within_run_range_e1_last5 | tier-independent | 4.61 | 2.983 | 5.349 | 1.746 | 7.904 | 10 | 1.917 | 2.366 |
| 60 | B2_frozen_s1norm | stochastic | e1_at_0 | tier-independent | 39.27 | 36.96 | 39.9 | 33.31 | 41.78 | 10 | 2.576 | 2.945 |
| 60 | B2_frozen_s1norm | stochastic | sigma_effort_at_0_t1 | tier-independent | 3.485 | 3.357 | 3.654 | 3.315 | 3.756 | 10 | 0.1699 | 0.2968 |
| 60 | B2_frozen_s1norm_mean | mean | within_run_sd_e1_last5 | tier-independent | 1.988 | 1.327 | 2.171 | 1.001 | 2.607 | 10 | 0.5528 | 0.8431 |
| 60 | B2_frozen_s1norm_mean | mean | within_run_range_e1_last5 | tier-independent | 4.659 | 3.261 | 5.515 | 2.595 | 6.232 | 10 | 1.367 | 2.254 |
| 60 | B2_frozen_s1norm_mean | mean | e1_at_0 | tier-independent | 37.98 | 37.02 | 42.73 | 32.1 | 43.92 | 10 | 4.11 | 5.709 |
| 60 | B2_frozen_s1norm_mean | mean | sigma_effort_at_0_t1 | tier-independent | 3.498 | 3.4 | 3.702 | 3.208 | 3.849 | 10 | 0.2012 | 0.3014 |
