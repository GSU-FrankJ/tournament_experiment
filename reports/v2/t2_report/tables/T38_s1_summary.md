# T38: S1 summary

- priority: core; status: generated; tier: tier-independent (stage-1 error is a direct policy query)
- sources: `results/v2_T2_locked/confirmation_analysis/s1_summary.csv`, `results/v2_T2_locked/confirmation_analysis/per_run.csv`, `tools/v2/confirmation_analysis.py`
- built by: `tools/v2/report/sec_locked_a.py:build_t38`; base commit `cb0b541`
- transformation: Per-q rows: confirmation_analysis/s1_summary.csv as is (bootstrap 10,000 resamples, numpy seed 20261002, a fresh generator per q, q ascending; exact Clopper-Pearson CI), plus the number and seeds of the S1 failures; 'S1 failure' rows: the failing runs from per_run.csv with their signed error, Gmax_full/DW (final tier) and run pass under v1.1 and v1.0. Check: S1 count, CP bounds, mean, bootstrap CI (common.bootstrap_mean_ci, same method and seed), median and SD recomputed from per_run.csv equal the file (max abs diff 0.0).

S1 (secondary; no pass rule): per q pass count with exact 95% CI, mean signed stage-1 error with its bootstrap 95% CI, median and SD; then the runs that fail S1.

| row_type | q | n | S1_pass | cp95_lo | cp95_hi | mean_signed | boot95_lo | boot95_hi | median_signed | sd_signed | bootstrap_resamples | bootstrap_seed | n_S1_fail | S1_fail_seeds | seed | stage1_rel_err_signed | s1 | gmax_final | run_pass | v1_0_run_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| per-q summary | 50 | 20 | 20 | 0.8316 | 1 | -0.005066 | -0.02592 | 0.01608 | 0.00111 | 0.04937 | 10000 | 20261002 | 0 |  |  |  |  |  |  |  |
| per-q summary | 60 | 20 | 18 | 0.683 | 0.9877 | 0.007697 | -0.01845 | 0.03493 | -0.00175 | 0.06254 | 10000 | 20261002 | 2 | 20510; 20515 |  |  |  |  |  |  |
| S1 failure | 60 |  |  |  |  |  |  |  |  |  |  |  |  |  | 20510 | 0.153 | 0.153 | 0.003638 | True | False |
| S1 failure | 60 |  |  |  |  |  |  |  |  |  |  |  |  |  | 20515 | -0.1135 | 0.1135 | 0.00181 | True | False |
