# Round 2 (concentration fix, warmup=200) — Full Results

Revised 2026-04-14: switched final-effort metric from last-50 avg (Metric C)
to policy_mean_effort[-1] (Metric B). See docs/round2_metric_decision.md for rationale.

Parameters: Set 1, k=0.00055, w_H=6.5, w_L=3.0, W=3.5

Round 2 data: `results/two_players/convergence/ppo_q*.0_seed*_convergence.json`

Baseline data: `_archive_pre_warmup_fix/ppo_q*.0_seed*_entropy_end_0.002_convergence.json`

## Section 1: Per-seed detail

| q | seed | final_effort | e* | abs_gap | rel_gap_% | exploit_final | conc_final | conc_peak | n_updates | stop_reason |
|--:|-----:|-------------:|-------:|--------:|----------:|--------------:|-----------:|----------:|----------:|-------------|
| 35 | 42 | 43.46 | 45.45 | 2.00 | 4.40 | 0.0072 | 196 | 196 | 69 | exploitability |
| 35 | 43 | 44.26 | 45.45 | 1.19 | 2.63 | 0.0074 | 234 | 234 | 49 | exploitability |
| 35 | 44 | 44.34 | 45.45 | 1.11 | 2.44 | 0.0085 | 184 | 184 | 49 | exploitability |
| 35 | 45 | 41.49 | 45.45 | 3.96 | 8.72 | 0.0089 | 193 | 193 | 49 | exploitability |
| 35 | 46 | 46.96 | 45.45 | 1.51 | 3.32 | 0.0226 | 204 | 204 | 49 | exploitability |
| 45 | 42 | 35.82 | 35.35 | 0.47 | 1.33 | 0.0075 | 237 | 237 | 69 | exploitability |
| 45 | 43 | 34.37 | 35.35 | 0.98 | 2.77 | 0.0092 | 165 | 166 | 69 | exploitability |
| 45 | 44 | 33.51 | 35.35 | 1.84 | 5.21 | 0.0057 | 223 | 223 | 59 | exploitability |
| 45 | 45 | 35.48 | 35.35 | 0.13 | 0.36 | 0.0088 | 199 | 199 | 69 | exploitability |
| 45 | 46 | 35.61 | 35.35 | 0.26 | 0.73 | 0.0073 | 220 | 220 | 69 | exploitability |
| 55 | 42 | 29.29 | 28.93 | 0.36 | 1.25 | 0.0064 | 254 | 254 | 99 | exploitability |
| 55 | 43 | 29.60 | 28.93 | 0.67 | 2.32 | 0.0060 | 250 | 253 | 89 | exploitability |
| 55 | 44 | 27.84 | 28.93 | 1.08 | 3.75 | 0.0050 | 250 | 250 | 109 | exploitability |
| 55 | 45 | 28.20 | 28.93 | 0.73 | 2.52 | 0.0068 | 224 | 224 | 89 | exploitability |
| 55 | 46 | 29.49 | 28.93 | 0.56 | 1.95 | 0.0057 | 222 | 222 | 79 | exploitability |

Row count: 15

## Section 2: Per-q summary statistics

| q | n_seeds | mean_abs_gap | std_abs_gap | min_gap | max_gap | mean_rel_% | mean_exploit | mean_updates |
|--:|--------:|-------------:|------------:|--------:|--------:|-----------:|-------------:|-------------:|
| 35 | 5 | 1.96 | 1.18 | 1.11 | 3.96 | 4.30 | 0.0109 | 53 |
| 45 | 5 | 0.74 | 0.70 | 0.13 | 1.84 | 2.08 | 0.0077 | 67 |
| 55 | 5 | 0.68 | 0.26 | 0.36 | 1.08 | 2.36 | 0.0060 | 93 |

## Section 3: Comparison vs baseline

Baseline source: `_archive_pre_warmup_fix/ppo_q*.0_seed*_entropy_end_0.002_convergence.json`

| q | baseline_mean_gap | round2_mean_gap | baseline_mean_exploit | round2_mean_exploit |
|--:|------------------:|----------------:|----------------------:|--------------------:|
| 35 | MISSING | 1.96 | MISSING | 0.0109 |
| 45 | 4.27 | 0.74 | 0.0211 | 0.0077 |
| 55 | 7.58 | 0.68 | 0.0474 | 0.0060 |

## Section 4: Raw trajectory checkpoints

`policy_mean_effort` at given fraction of total updates.

| q | seed | t=0 | t=25% | t=50% | t=75% | t=final |
|--:|-----:|----:|------:|------:|------:|--------:|
| 35 | 42 | 57.90 | 49.94 | 45.03 | 43.82 | 43.46 |
| 35 | 43 | 49.05 | 44.53 | 44.21 | 44.02 | 44.26 |
| 35 | 44 | 48.62 | 47.39 | 45.51 | 45.13 | 44.34 |
| 35 | 45 | 45.91 | 40.81 | 40.46 | 42.30 | 41.49 |
| 35 | 46 | 46.48 | 47.91 | 47.80 | 48.03 | 46.96 |
| 45 | 42 | 50.52 | 42.67 | 38.28 | 36.19 | 35.82 |
| 45 | 43 | 49.38 | 43.30 | 38.54 | 36.59 | 34.37 |
| 45 | 44 | 46.74 | 40.40 | 36.17 | 34.54 | 33.51 |
| 45 | 45 | 50.39 | 44.26 | 39.04 | 36.16 | 35.48 |
| 45 | 46 | 50.94 | 42.56 | 37.25 | 34.89 | 35.61 |
| 55 | 42 | 47.17 | 41.39 | 34.25 | 31.59 | 29.29 |
| 55 | 43 | 45.53 | 39.62 | 33.41 | 30.74 | 29.60 |
| 55 | 44 | 53.29 | 42.94 | 35.50 | 30.97 | 27.84 |
| 55 | 45 | 46.16 | 40.50 | 33.52 | 29.81 | 28.20 |
| 55 | 46 | 45.94 | 37.89 | 32.52 | 30.42 | 29.49 |

Row count: 15

## Verification

- Section 1: 15 rows (expected 15)
- Section 2: 3 rows (expected 3)
- Section 3: 3 rows (expected 3)
- Section 4: 15 rows (expected 15)
