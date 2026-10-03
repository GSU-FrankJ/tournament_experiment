# T50: Anatomy of the stage-1 fluctuation

- priority: core; status: generated; tier: tier-independent
- sources: `results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_every20.csv`, `results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_exports25.csv`, `results/v2_pilots/pilot4/analysis/fluctuation/acf_per_run_every20.csv`, `results/v2_pilots/pilot4/analysis/fluctuation/window_regression.csv`, `results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv`, `results/v2_pilots/pilot4/analysis/fluctuation/export_series_e1.csv`, `results/v2_pilots/pilot4/analysis/fluctuation/windows.csv`, `results/v2_pilots/pilot4/analysis/root_game/br_slope.csv`, `tools/v2/pilot4_fluctuation.py`, `reports/v2/pilot4_stabilization.md`
- built by: `tools/v2/report/sec_stage1.py:build_t50`; base commit `cb0b541`
- transformation: Blocks: record and definitions (from the series CSVs and the tool docstring; text marked source: report text); verbatim copies of acf_summary_every20.csv (both versions, both segments, lags 20-160), acf_summary_exports25.csv and window_regression.csv; comparison of the pooled/per-run window slopes with the root-game BR slope (br_slope.csv). ACF summaries re-derived from acf_per_run_every20.csv with numpy percentiles: max abs diff 0. The four every-20 ACF tables of pilot4 section 1b carry no segment/kind label; matched by value: every-20 table 1 (report line 134) = segment u>=650, centred; every-20 table 2 (report line 143) = segment u>=700, centred; every-20 table 3 (report line 152) = segment u>=650, about_e1star; every-20 table 4 (report line 161) = segment u>=700, about_e1star; exports-25 table (report line 172) = both segments, centred. Values are tier-independent (Beta-mean policy queries).

Anatomy of the stage-1 fluctuation in the Pilot 3 runs (both arms): ACF of e_hat_1(0) - e1* from the 20-update stability log, the 25-update export ACF, and the 20-update window regression.

| block | segment_start | kind | q | arm | lag_updates | median | q25 | q75 | n_runs | n_windows | slope | boot_ci95_lo | boot_ci95_hi | per_run_slope_median | per_run_slope_q25 | per_run_slope_q75 | quantity | value | unit | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| record and definitions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | series | e_hat_1(0) (Beta mean) from the stability log of train_history.json, every 20 updates; no per-update record of the root policy mean exists (history mean_effort_by_stage is the batch mean of sampled efforts) (source: report text) | text | results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; reports/v2/pilot4_stabilization.md section 1b |
| record and definitions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | runs | 40 | count (Pilot 3: 2 arms x 2 q x 10 seeds) | results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; reports/v2/pilot4_stabilization.md section 1b |
| record and definitions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | points per run | 30 | count | results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; reports/v2/pilot4_stabilization.md section 1b |
| record and definitions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | first and last update | u420 .. u1000 | global update | results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; reports/v2/pilot4_stabilization.md section 1b |
| record and definitions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | spacing | 20 | updates | results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; reports/v2/pilot4_stabilization.md section 1b |
| record and definitions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ACF segment u>=650: points per run | 18 | count (u660..u1000) | results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; reports/v2/pilot4_stabilization.md section 1b |
| record and definitions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ACF segment u>=700: points per run | 16 | count (u700..u1000) | results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; reports/v2/pilot4_stabilization.md section 1b |
| record and definitions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ACF definitions | per run, x = e_hat_1(0) - e1* on the segment; 'centred' = sample ACF about the segment mean (denominator n c0); 'about_e1star' = same with x not demeaned; summary = median and q25/q75 over the 10 runs per (q, arm) | text | results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; reports/v2/pilot4_stabilization.md section 1b |
| record and definitions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | secondary series | 25-update weight exports u425..u1000 (24 per run), lags 25-100 | text | results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; reports/v2/pilot4_stabilization.md section 1b |
| record and definitions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | window regression | y = e_hat_1(u+20) - e_hat_1(u) on x = e_hat_1(u) - e1* (opponent refreshed to the actor at u), OLS with intercept pooled over runs per (q, arm) and per q; 95% CI by a cluster bootstrap over runs (10,000 resamples, numpy seed 20261001) | text | results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; reports/v2/pilot4_stabilization.md section 1b |
| record and definitions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | windows (all segments) | 1160 | count | results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; reports/v2/pilot4_stabilization.md section 1b |
| record and definitions |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | recomputed every-20 summary from acf_per_run_every20.csv (numpy percentiles) vs acf_summary_every20.csv: max abs diff | 0 | ACF (dimensionless) | results/v2_pilots/pilot4/analysis/fluctuation/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; reports/v2/pilot4_stabilization.md section 1b |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | mean | 20 | 0.7729 | 0.5727 | 0.8722 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | mean | 40 | 0.5363 | 0.4134 | 0.7833 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | mean | 60 | 0.4414 | 0.295 | 0.6637 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | mean | 80 | 0.3691 | 0.2903 | 0.5634 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | mean | 100 | 0.3345 | 0.243 | 0.4721 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | mean | 120 | 0.2936 | 0.08561 | 0.4213 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | mean | 140 | 0.2175 | 0.03087 | 0.2969 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | mean | 160 | 0.1298 | -0.0006538 | 0.2205 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | stochastic | 20 | 0.8217 | 0.7608 | 0.8666 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | stochastic | 40 | 0.6542 | 0.6017 | 0.7154 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | stochastic | 60 | 0.5062 | 0.4741 | 0.5692 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | stochastic | 80 | 0.3757 | 0.2859 | 0.5413 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | stochastic | 100 | 0.2933 | 0.1719 | 0.4842 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | stochastic | 120 | 0.1529 | 0.01169 | 0.4215 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | stochastic | 140 | 0.1247 | 0.04506 | 0.3374 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 50 | stochastic | 160 | 0.112 | -0.0381 | 0.2546 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | mean | 20 | 0.7073 | 0.6823 | 0.7739 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | mean | 40 | 0.5461 | 0.3247 | 0.6276 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | mean | 60 | 0.4593 | 0.08043 | 0.5677 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | mean | 80 | 0.2873 | 0.1024 | 0.5635 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | mean | 100 | 0.2914 | 0.08153 | 0.5109 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | mean | 120 | 0.3026 | 0.01636 | 0.4622 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | mean | 140 | 0.2404 | 0.01848 | 0.3965 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | mean | 160 | 0.2724 | 0.02122 | 0.4021 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | stochastic | 20 | 0.7155 | 0.649 | 0.8735 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | stochastic | 40 | 0.4787 | 0.2669 | 0.7401 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | stochastic | 60 | 0.2829 | -0.04897 | 0.6167 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | stochastic | 80 | 0.1544 | -0.0609 | 0.4953 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | stochastic | 100 | 0.1185 | -0.1838 | 0.4383 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | stochastic | 120 | 0.06004 | -0.1673 | 0.4083 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | stochastic | 140 | 0.03289 | -0.09846 | 0.3784 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | about_e1star | 60 | stochastic | 160 | 0.1384 | -0.01948 | 0.3355 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | mean | 20 | 0.5894 | 0.1938 | 0.7224 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | mean | 40 | 0.2968 | 0.1699 | 0.4827 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | mean | 60 | 0.04625 | -0.0713 | 0.3233 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | mean | 80 | 0.05744 | 0.02426 | 0.1479 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | mean | 100 | 0.01407 | -0.1527 | 0.08973 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | mean | 120 | -0.1078 | -0.1853 | 0.0006946 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | mean | 140 | -0.1873 | -0.2712 | -0.05696 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | mean | 160 | -0.2316 | -0.2737 | -0.1039 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | stochastic | 20 | 0.4828 | 0.3975 | 0.6917 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | stochastic | 40 | 0.2877 | 0.1316 | 0.4072 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | stochastic | 60 | 0.1017 | 0.0122 | 0.2584 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | stochastic | 80 | 0.02426 | -0.01918 | 0.1374 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | stochastic | 100 | -0.09166 | -0.1969 | 0.02273 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | stochastic | 120 | -0.1909 | -0.259 | -0.08871 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | stochastic | 140 | -0.204 | -0.2848 | -0.05951 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 50 | stochastic | 160 | -0.245 | -0.3359 | -0.1621 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | mean | 20 | 0.5105 | 0.4106 | 0.5592 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | mean | 40 | 0.2272 | 0.04982 | 0.2722 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | mean | 60 | -0.05513 | -0.2107 | 0.007331 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | mean | 80 | -0.2399 | -0.2631 | -0.1113 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | mean | 100 | -0.09986 | -0.2888 | -0.01455 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | mean | 120 | -0.1061 | -0.2175 | -0.08645 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | mean | 140 | -0.1077 | -0.2452 | -0.08441 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | mean | 160 | -0.06168 | -0.1116 | 0.003583 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | stochastic | 20 | 0.6024 | 0.5571 | 0.6733 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | stochastic | 40 | 0.2762 | 0.04149 | 0.4027 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | stochastic | 60 | -0.06379 | -0.1482 | 0.2401 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | stochastic | 80 | -0.08073 | -0.4026 | 0.02027 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | stochastic | 100 | -0.2471 | -0.3355 | 0.01374 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | stochastic | 120 | -0.2035 | -0.3111 | -0.08896 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | stochastic | 140 | -0.1807 | -0.2741 | -0.05501 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 650 | centred | 60 | stochastic | 160 | -0.06887 | -0.2013 | 0.04542 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | mean | 20 | 0.7576 | 0.6172 | 0.8701 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | mean | 40 | 0.5805 | 0.4046 | 0.7503 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | mean | 60 | 0.4308 | 0.2855 | 0.6209 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | mean | 80 | 0.334 | 0.2317 | 0.549 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | mean | 100 | 0.2847 | 0.2389 | 0.4012 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | mean | 120 | 0.2458 | 0.1158 | 0.354 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | mean | 140 | 0.1854 | 0.1237 | 0.3104 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | mean | 160 | 0.1428 | 0.09987 | 0.2445 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | stochastic | 20 | 0.795 | 0.7406 | 0.8458 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | stochastic | 40 | 0.6199 | 0.5953 | 0.7094 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | stochastic | 60 | 0.4597 | 0.4047 | 0.5492 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | stochastic | 80 | 0.2861 | 0.239 | 0.5141 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | stochastic | 100 | 0.2531 | 0.1175 | 0.4091 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | stochastic | 120 | 0.1204 | 0.01119 | 0.3527 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | stochastic | 140 | 0.09974 | 0.02533 | 0.2919 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 50 | stochastic | 160 | 0.1109 | -0.0645 | 0.2544 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | mean | 20 | 0.763 | 0.7023 | 0.8396 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | mean | 40 | 0.5798 | 0.4149 | 0.6925 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | mean | 60 | 0.4832 | 0.1112 | 0.5972 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | mean | 80 | 0.3423 | 0.09224 | 0.5436 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | mean | 100 | 0.3229 | 0.07093 | 0.5194 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | mean | 120 | 0.3487 | -0.008712 | 0.4668 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | mean | 140 | 0.2594 | 0.04468 | 0.4173 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | mean | 160 | 0.257 | 0.009249 | 0.3718 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | stochastic | 20 | 0.7129 | 0.5972 | 0.8087 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | stochastic | 40 | 0.5344 | 0.2021 | 0.6663 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | stochastic | 60 | 0.3184 | 0.05401 | 0.5306 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | stochastic | 80 | 0.243 | 0.01494 | 0.423 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | stochastic | 100 | 0.1685 | -0.1071 | 0.4065 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | stochastic | 120 | 0.08453 | -0.1265 | 0.3924 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | stochastic | 140 | 0.1418 | -0.1172 | 0.3311 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | about_e1star | 60 | stochastic | 160 | 0.193 | 0.0662 | 0.2417 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | mean | 20 | 0.5498 | 0.2257 | 0.7002 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | mean | 40 | 0.2227 | 0.1029 | 0.3898 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | mean | 60 | 0.07083 | -0.1473 | 0.1569 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | mean | 80 | 0.01973 | -0.03425 | 0.1151 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | mean | 100 | -0.07484 | -0.1648 | 0.03998 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | mean | 120 | -0.1361 | -0.1675 | -0.07149 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | mean | 140 | -0.1734 | -0.2932 | -0.02652 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | mean | 160 | -0.2361 | -0.3105 | -0.005232 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | stochastic | 20 | 0.4434 | 0.3855 | 0.6525 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | stochastic | 40 | 0.2361 | 0.1444 | 0.4386 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | stochastic | 60 | 0.1834 | -0.002551 | 0.2074 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | stochastic | 80 | -0.01316 | -0.05515 | 0.05086 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | stochastic | 100 | -0.1098 | -0.1664 | -0.007 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | stochastic | 120 | -0.1438 | -0.2138 | -0.08755 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | stochastic | 140 | -0.2264 | -0.3155 | -0.08442 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 50 | stochastic | 160 | -0.2708 | -0.3994 | -0.1929 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | mean | 20 | 0.5099 | 0.3926 | 0.5795 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | mean | 40 | 0.2279 | 0.04351 | 0.2594 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | mean | 60 | -0.07172 | -0.2326 | 0.0334 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | mean | 80 | -0.2486 | -0.3174 | -0.1053 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | mean | 100 | -0.07407 | -0.2939 | -0.009328 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | mean | 120 | -0.08737 | -0.2087 | -0.01759 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | mean | 140 | -0.06769 | -0.1846 | -0.03024 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | mean | 160 | -0.05045 | -0.1071 | -0.02272 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | stochastic | 20 | 0.5855 | 0.4845 | 0.6214 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | stochastic | 40 | 0.2987 | 0.1263 | 0.3929 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | stochastic | 60 | -0.01788 | -0.1471 | 0.2769 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | stochastic | 80 | -0.05139 | -0.3308 | 0.06065 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | stochastic | 100 | -0.2381 | -0.3676 | -0.06923 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | stochastic | 120 | -0.2131 | -0.33 | -0.1277 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | stochastic | 140 | -0.1646 | -0.2652 | -0.05948 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF every 20 updates (acf_summary_every20.csv) | 700 | centred | 60 | stochastic | 160 | -0.1202 | -0.2346 | 0.07366 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 50 | mean | 25 | 0.7008 | 0.5158 | 0.872 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 50 | mean | 50 | 0.4669 | 0.3575 | 0.7317 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 50 | mean | 75 | 0.3022 | 0.1407 | 0.5798 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 50 | mean | 100 | 0.2882 | 0.2299 | 0.4575 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 50 | stochastic | 25 | 0.7603 | 0.7034 | 0.8426 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 50 | stochastic | 50 | 0.5518 | 0.4727 | 0.6391 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 50 | stochastic | 75 | 0.3899 | 0.2828 | 0.5255 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 50 | stochastic | 100 | 0.2463 | 0.1659 | 0.4843 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 60 | mean | 25 | 0.7156 | 0.6326 | 0.7641 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 60 | mean | 50 | 0.4464 | 0.2699 | 0.6192 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 60 | mean | 75 | 0.3641 | 0.0782 | 0.5374 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 60 | mean | 100 | 0.2779 | 0.06576 | 0.5599 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 60 | stochastic | 25 | 0.6245 | 0.5794 | 0.8349 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 60 | stochastic | 50 | 0.4283 | 0.1242 | 0.6884 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 60 | stochastic | 75 | 0.1921 | -0.08239 | 0.5285 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | about_e1star | 60 | stochastic | 100 | 0.1089 | -0.1695 | 0.4515 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 50 | mean | 25 | 0.4966 | 0.3473 | 0.6258 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 50 | mean | 50 | 0.2997 | 0.2388 | 0.3647 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 50 | mean | 75 | 0.004596 | -0.1405 | 0.1241 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 50 | mean | 100 | -0.02029 | -0.1338 | 0.1311 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 50 | stochastic | 25 | 0.4609 | 0.2959 | 0.5881 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 50 | stochastic | 50 | 0.2487 | -0.06372 | 0.3221 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 50 | stochastic | 75 | 0.03655 | -0.1147 | 0.1616 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 50 | stochastic | 100 | -0.1483 | -0.2306 | -0.03003 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 60 | mean | 25 | 0.4378 | 0.3223 | 0.5984 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 60 | mean | 50 | 0.1081 | 0.004571 | 0.1186 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 60 | mean | 75 | -0.1486 | -0.2484 | -0.08424 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 60 | mean | 100 | -0.1259 | -0.1617 | -0.06055 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 60 | stochastic | 25 | 0.5179 | 0.4363 | 0.5631 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 60 | stochastic | 50 | 0.1052 | -0.03 | 0.3013 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 60 | stochastic | 75 | -0.075 | -0.3795 | 0.05586 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 650 | centred | 60 | stochastic | 100 | -0.2174 | -0.3222 | -0.009634 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 50 | mean | 25 | 0.6777 | 0.5344 | 0.8533 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 50 | mean | 50 | 0.539 | 0.3638 | 0.6713 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 50 | mean | 75 | 0.3705 | 0.08797 | 0.5107 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 50 | mean | 100 | 0.277 | 0.2316 | 0.3574 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 50 | stochastic | 25 | 0.7422 | 0.7317 | 0.8277 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 50 | stochastic | 50 | 0.5412 | 0.4856 | 0.6163 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 50 | stochastic | 75 | 0.2787 | 0.2324 | 0.5419 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 50 | stochastic | 100 | 0.1905 | 0.1147 | 0.4183 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 60 | mean | 25 | 0.7378 | 0.7077 | 0.8119 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 60 | mean | 50 | 0.4916 | 0.3394 | 0.7045 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 60 | mean | 75 | 0.3951 | 0.1342 | 0.575 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 60 | mean | 100 | 0.3456 | 0.03432 | 0.5002 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 60 | stochastic | 25 | 0.6103 | 0.5358 | 0.7373 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 60 | stochastic | 50 | 0.3897 | 0.1335 | 0.5935 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 60 | stochastic | 75 | 0.2475 | 0.06072 | 0.4365 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | about_e1star | 60 | stochastic | 100 | 0.1567 | -0.04988 | 0.4355 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 50 | mean | 25 | 0.4192 | 0.152 | 0.622 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 50 | mean | 50 | 0.1943 | 0.05172 | 0.2992 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 50 | mean | 75 | -0.07663 | -0.2505 | 0.01977 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 50 | mean | 100 | -0.08922 | -0.2186 | 0.02661 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 50 | stochastic | 25 | 0.4424 | 0.3527 | 0.5963 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 50 | stochastic | 50 | 0.1967 | 0.06373 | 0.3248 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 50 | stochastic | 75 | 0.04985 | -0.1374 | 0.1242 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 50 | stochastic | 100 | -0.1224 | -0.2765 | -0.03207 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 60 | mean | 25 | 0.3825 | 0.3537 | 0.6299 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 60 | mean | 50 | 0.09014 | -0.02666 | 0.2003 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 60 | mean | 75 | -0.1244 | -0.3069 | 0.02449 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 60 | mean | 100 | -0.1087 | -0.1637 | -0.02425 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 60 | stochastic | 25 | 0.5084 | 0.359 | 0.5344 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 60 | stochastic | 50 | 0.1334 | -0.01597 | 0.2686 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 60 | stochastic | 75 | -0.02871 | -0.2176 | 0.0718 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| ACF of the 25-update exports, secondary (acf_summary_exports25.csv) | 700 | centred | 60 | stochastic | 100 | -0.1906 | -0.3416 | -0.03175 | 10 |  |  |  |  |  |  |  |  |  |  |  |
| window regression (window_regression.csv) | 650 |  | 50 | mean |  |  |  |  | 10 | 170 | -0.1537 | -0.2937 | -0.07901 | -0.3597 | -0.7876 | -0.1581 |  |  |  |  |
| window regression (window_regression.csv) | 650 |  | 50 | stochastic |  |  |  |  | 10 | 170 | -0.1457 | -0.236 | -0.1189 | -0.3964 | -0.5056 | -0.2987 |  |  |  |  |
| window regression (window_regression.csv) | 650 |  | 60 | mean |  |  |  |  | 10 | 170 | -0.198 | -0.3171 | -0.133 | -0.4153 | -0.4345 | -0.3702 |  |  |  |  |
| window regression (window_regression.csv) | 650 |  | 60 | stochastic |  |  |  |  | 10 | 170 | -0.1999 | -0.287 | -0.1617 | -0.3279 | -0.3691 | -0.2669 |  |  |  |  |
| window regression (window_regression.csv) | 650 |  | 50 | both |  |  |  |  | 20 | 340 | -0.149 | -0.2196 | -0.1068 | -0.3964 | -0.6047 | -0.2102 |  |  |  |  |
| window regression (window_regression.csv) | 650 |  | 60 | both |  |  |  |  | 20 | 340 | -0.1965 | -0.2638 | -0.1503 | -0.3686 | -0.4173 | -0.3081 |  |  |  |  |
| window regression (window_regression.csv) | 700 |  | 50 | mean |  |  |  |  | 10 | 150 | -0.1358 | -0.3082 | -0.05948 | -0.4324 | -0.7476 | -0.1558 |  |  |  |  |
| window regression (window_regression.csv) | 700 |  | 50 | stochastic |  |  |  |  | 10 | 150 | -0.1428 | -0.2261 | -0.09964 | -0.3681 | -0.5404 | -0.2423 |  |  |  |  |
| window regression (window_regression.csv) | 700 |  | 60 | mean |  |  |  |  | 10 | 150 | -0.1675 | -0.2795 | -0.1059 | -0.3867 | -0.4098 | -0.3207 |  |  |  |  |
| window regression (window_regression.csv) | 700 |  | 60 | stochastic |  |  |  |  | 10 | 150 | -0.2355 | -0.3224 | -0.1524 | -0.3283 | -0.3938 | -0.2541 |  |  |  |  |
| window regression (window_regression.csv) | 700 |  | 50 | both |  |  |  |  | 20 | 300 | -0.1386 | -0.2143 | -0.09393 | -0.3833 | -0.5933 | -0.2372 |  |  |  |  |
| window regression (window_regression.csv) | 700 |  | 60 | both |  |  |  |  | 20 | 300 | -0.2033 | -0.2707 | -0.1485 | -0.3754 | -0.4032 | -0.2585 |  |  |  |  |
| comparison with the root-game BR slope |  |  | 50 |  |  |  |  |  |  |  |  | -0.2196 | -0.1068 |  |  |  | pooled window slope (u>=650, both arms) | -0.149 | dimensionless | results/v2_pilots/pilot4/analysis/fluctuation/window_regression.csv |
| comparison with the root-game BR slope |  |  | 50 |  |  |  |  |  |  |  |  |  |  |  |  |  | per-run median window slope (u>=650, both arms) | -0.3964 | dimensionless | results/v2_pilots/pilot4/analysis/fluctuation/window_regression.csv |
| comparison with the root-game BR slope |  |  | 50 |  |  |  |  |  |  |  |  |  |  |  |  |  | root-game BR slope, PI reference | -0.961 | dimensionless | results/v2_pilots/pilot4/analysis/root_game/br_slope.csv (ref_slope) |
| comparison with the root-game BR slope |  |  | 50 |  |  |  |  |  |  |  |  |  |  |  |  |  | root-game BR slope, fine tier, h = 1, W = 2 | -0.9224 | dimensionless | results/v2_pilots/pilot4/analysis/root_game/br_slope.csv |
| comparison with the root-game BR slope |  |  | 50 |  |  |  |  |  |  |  |  |  |  |  |  |  | root-game BR slope, final tier, min over W and h | -1.057 | dimensionless | results/v2_pilots/pilot4/analysis/root_game/br_slope.csv |
| comparison with the root-game BR slope |  |  | 50 |  |  |  |  |  |  |  |  |  |  |  |  |  | root-game BR slope, final tier, max over W and h | -0.8533 | dimensionless | results/v2_pilots/pilot4/analysis/root_game/br_slope.csv |
| comparison with the root-game BR slope |  |  | 60 |  |  |  |  |  |  |  |  | -0.2638 | -0.1503 |  |  |  | pooled window slope (u>=650, both arms) | -0.1965 | dimensionless | results/v2_pilots/pilot4/analysis/fluctuation/window_regression.csv |
| comparison with the root-game BR slope |  |  | 60 |  |  |  |  |  |  |  |  |  |  |  |  |  | per-run median window slope (u>=650, both arms) | -0.3686 | dimensionless | results/v2_pilots/pilot4/analysis/fluctuation/window_regression.csv |
| comparison with the root-game BR slope |  |  | 60 |  |  |  |  |  |  |  |  |  |  |  |  |  | root-game BR slope, PI reference | -0.309 | dimensionless | results/v2_pilots/pilot4/analysis/root_game/br_slope.csv (ref_slope) |
| comparison with the root-game BR slope |  |  | 60 |  |  |  |  |  |  |  |  |  |  |  |  |  | root-game BR slope, fine tier, h = 1, W = 2 | -0.3015 | dimensionless | results/v2_pilots/pilot4/analysis/root_game/br_slope.csv |
| comparison with the root-game BR slope |  |  | 60 |  |  |  |  |  |  |  |  |  |  |  |  |  | root-game BR slope, final tier, min over W and h | -0.3322 | dimensionless | results/v2_pilots/pilot4/analysis/root_game/br_slope.csv |
| comparison with the root-game BR slope |  |  | 60 |  |  |  |  |  |  |  |  |  |  |  |  |  | root-game BR slope, final tier, max over W and h | -0.2644 | dimensionless | results/v2_pilots/pilot4/analysis/root_game/br_slope.csv |
