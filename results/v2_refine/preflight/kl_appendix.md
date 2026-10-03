### Appendix: KL statistics of the locked v1.1 baseline (descriptive)

Source: `history[*]` of `train_history.json` of the 20 `rehearsal_v1_1` runs (q in {50, 60} x seeds 10501-10510), canonical worktree `results/v2_T2_locked/rehearsal_v1_1/q*/seed*/`. Every update logs `kl_epochs`, the whole-buffer KL over the policy rows after each of the 10 PPO epochs, mean((r - 1) - log r) with r = pi_new / pi_old, and `kl_final_epoch` = `kl_epochs[-1]`. Phase A has 1600 updates and phase B 600 per run, so each table row pools 16000 (A) or 6000 (B) updates per q. Percentiles use linear interpolation (`numpy.percentile`).

Stopping rule of the target-KL arms (D5): after an epoch, if the KL exceeds the target (strict >) no further epoch runs; the epoch that exceeded the target has run. The number of epochs run is therefore the 1-based index of the first epoch with KL > target, or 10 if none exceeds it. The tables apply that rule to the baseline `kl_epochs`. They are exact for each baseline update taken in isolation (up to and including the first stop the update is identical to the baseline), but counterfactual for the run as a whole: after the first stop the trajectory of a target-KL run differs from the baseline, so later updates would see different buffers and policies. They are not a prediction of the share of stopped updates in a target-KL run. 'Stopped' means that some epoch's KL exceeded the target; an exceedance at epoch 10 truncates nothing (10 epochs run either way), so the column 'fewer than 10 epochs run' (stop epoch 1 to 9) is the share of updates a target-KL run would actually shorten.

Script: `python tools/v2/refine_preflight.py kl`; CSVs: `results/v2_refine/preflight/kl_final_epoch_distribution.csv`, `results/v2_refine/preflight/kl_target_stopping.csv`.

#### (a) Final-epoch KL (`kl_final_epoch`), all updates of the 10 seeds

| phase | q | updates | min | p10 | median | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| A | 50 | 16000 | 4.6e-05 | 0.00185 | 0.00503 | 0.0116 | 0.0237 | 0.0815 | 0.00616 |
| B | 50 | 6000 | 5.36e-09 | 0.000152 | 0.00326 | 0.0123 | 0.0306 | 0.0709 | 0.0052 |
| A | 60 | 16000 | 1.82e-05 | 0.00174 | 0.00497 | 0.0119 | 0.025 | 0.0823 | 0.0062 |
| B | 60 | 6000 | 1.05e-08 | 0.000188 | 0.00341 | 0.0128 | 0.0335 | 0.0795 | 0.00547 |

#### (b) Counterfactual stopping at each target

| phase | q | target | updates | stopped (n) | stopped (share) | of which fewer than 10 epochs run (share) | stopped share per run, min to max | stop epoch median / p90 (stopped only) | epochs run median / p90 / mean |
|---|---|---|---|---|---|---|---|---|---|
| A | 50 | 0.005 | 16000 | 14379 | 89.87% | 88.32% | 84.9% to 95.9% | 3 / 7 | 3 / 10 / 4.04 |
| A | 50 | 0.01 | 16000 | 8173 | 51.08% | 48.88% | 37.1% to 72.8% | 3 / 8 | 10 / 10 / 7.06 |
| B | 50 | 0.005 | 6000 | 5729 | 95.48% | 94.83% | 93.3% to 97.5% | 2 / 5 | 2 / 6 / 2.87 |
| B | 50 | 0.01 | 6000 | 4153 | 69.22% | 67.13% | 60.7% to 79.5% | 2 / 8 | 5 / 10 / 5.46 |
| A | 60 | 0.005 | 16000 | 14624 | 91.40% | 90.13% | 85.9% to 94.7% | 3 / 6 | 3 / 9 / 3.84 |
| A | 60 | 0.01 | 16000 | 8792 | 54.95% | 52.68% | 36.2% to 69.4% | 4 / 8 | 9 / 10 / 6.84 |
| B | 60 | 0.005 | 6000 | 5704 | 95.07% | 94.33% | 91.5% to 97.0% | 2 / 5 | 2 / 6 / 2.89 |
| B | 60 | 0.01 | 6000 | 4222 | 70.37% | 68.55% | 61.3% to 77.0% | 3 / 7 | 4 / 10 / 5.36 |

#### (c) Stopping epoch, share of all updates (%)

Epoch index is 1-based; 'none' = no epoch exceeded the target (10 epochs run).

| phase | q | target | ep 1 | ep 2 | ep 3 | ep 4 | ep 5 | ep 6 | ep 7 | ep 8 | ep 9 | ep 10 | none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 50 | 0.005 | 11.78 | 27.24 | 19.95 | 10.84 | 6.51 | 4.51 | 3.13 | 2.60 | 1.75 | 1.55 | 10.13 |
| A | 50 | 0.01 | 3.76 | 10.68 | 11.16 | 7.33 | 4.47 | 3.53 | 2.98 | 2.67 | 2.29 | 2.21 | 48.92 |
| B | 50 | 0.005 | 35.47 | 26.30 | 13.38 | 7.65 | 4.58 | 2.88 | 1.88 | 1.27 | 1.42 | 0.65 | 4.52 |
| B | 50 | 0.01 | 18.90 | 15.83 | 9.52 | 5.72 | 4.40 | 4.02 | 3.32 | 2.88 | 2.55 | 2.08 | 30.78 |
| A | 60 | 0.005 | 14.38 | 27.20 | 19.26 | 11.05 | 7.00 | 4.58 | 2.86 | 2.02 | 1.79 | 1.27 | 8.60 |
| A | 60 | 0.01 | 5.01 | 11.63 | 10.42 | 7.47 | 4.72 | 4.44 | 3.54 | 2.72 | 2.73 | 2.27 | 45.05 |
| B | 60 | 0.005 | 34.68 | 26.68 | 13.87 | 7.42 | 4.27 | 3.13 | 1.97 | 1.33 | 0.98 | 0.73 | 4.93 |
| B | 60 | 0.01 | 18.67 | 16.30 | 10.02 | 6.12 | 4.73 | 4.27 | 3.27 | 2.62 | 2.57 | 1.82 | 29.63 |

#### (d) Supplementary: phase A split by learning-rate window

Local 1-1200 has constant LR 3e-4; local 1201-1600 is the linear decay 3e-4 to 3e-5, the window the phase-A continuation arms run in.

| phase | q | updates | min | p10 | median | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| A local 1-1200 | 50 | 12000 | 4.6e-05 | 0.00198 | 0.00524 | 0.012 | 0.0243 | 0.0815 | 0.00641 |
| A local 1201-1600 | 50 | 4000 | 0.000103 | 0.00153 | 0.00443 | 0.0103 | 0.019 | 0.0542 | 0.00539 |
| A local 1-1200 | 60 | 12000 | 1.82e-05 | 0.00187 | 0.00518 | 0.0122 | 0.0254 | 0.0704 | 0.00643 |
| A local 1201-1600 | 60 | 4000 | 6.06e-05 | 0.00139 | 0.00439 | 0.0107 | 0.0215 | 0.0823 | 0.00549 |

| phase | q | target | updates | stopped (n) | stopped (share) | of which fewer than 10 epochs run (share) | stopped share per run, min to max | stop epoch median / p90 (stopped only) | epochs run median / p90 / mean |
|---|---|---|---|---|---|---|---|---|---|
| A local 1-1200 | 50 | 0.005 | 12000 | 11125 | 92.71% | 91.42% | 87.7% to 97.9% | 3 / 6 | 3 / 9 / 3.72 |
| A local 1-1200 | 50 | 0.01 | 12000 | 6650 | 55.42% | 53.08% | 40.3% to 76.8% | 3 / 8 | 8 / 10 / 6.77 |
| A local 1201-1600 | 50 | 0.005 | 4000 | 3254 | 81.35% | 79.03% | 75.8% to 90.0% | 3 / 7 | 4 / 10 / 5.01 |
| A local 1201-1600 | 50 | 0.01 | 4000 | 1523 | 38.07% | 36.25% | 27.5% to 60.5% | 4 / 9 | 10 / 10 / 7.91 |
| A local 1-1200 | 60 | 0.005 | 12000 | 11304 | 94.20% | 93.17% | 88.3% to 97.2% | 3 / 6 | 3 / 8 / 3.52 |
| A local 1-1200 | 60 | 0.01 | 12000 | 7155 | 59.62% | 57.27% | 40.3% to 74.1% | 3 / 8 | 7 / 10 / 6.53 |
| A local 1201-1600 | 60 | 0.005 | 4000 | 3320 | 83.00% | 81.00% | 74.5% to 87.8% | 3 / 7 | 4 / 10 / 4.8 |
| A local 1201-1600 | 60 | 0.01 | 4000 | 1637 | 40.92% | 38.90% | 24.0% to 55.2% | 4 / 9 | 10 / 10 / 7.76 |

| phase | q | target | ep 1 | ep 2 | ep 3 | ep 4 | ep 5 | ep 6 | ep 7 | ep 8 | ep 9 | ep 10 | none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A local 1-1200 | 50 | 0.005 | 12.95 | 29.97 | 21.03 | 10.24 | 6.12 | 4.34 | 2.84 | 2.42 | 1.50 | 1.29 | 7.29 |
| A local 1-1200 | 50 | 0.01 | 4.25 | 11.65 | 12.64 | 8.00 | 4.61 | 3.68 | 3.14 | 2.81 | 2.30 | 2.33 | 44.58 |
| A local 1201-1600 | 50 | 0.005 | 8.25 | 19.07 | 16.70 | 12.65 | 7.67 | 5.03 | 4.00 | 3.15 | 2.50 | 2.33 | 18.65 |
| A local 1201-1600 | 50 | 0.01 | 2.30 | 7.75 | 6.72 | 5.33 | 4.05 | 3.08 | 2.48 | 2.27 | 2.27 | 1.82 | 61.92 |
| A local 1-1200 | 60 | 0.005 | 16.25 | 29.27 | 19.89 | 10.74 | 6.83 | 4.19 | 2.55 | 1.84 | 1.62 | 1.02 | 5.80 |
| A local 1-1200 | 60 | 0.01 | 5.83 | 12.57 | 11.59 | 8.02 | 5.08 | 4.84 | 3.73 | 2.82 | 2.79 | 2.36 | 40.38 |
| A local 1201-1600 | 60 | 0.005 | 8.78 | 21.00 | 17.35 | 11.97 | 7.53 | 5.72 | 3.77 | 2.55 | 2.33 | 2.00 | 17.00 |
| A local 1201-1600 | 60 | 0.01 | 2.52 | 8.80 | 6.90 | 5.83 | 3.67 | 3.23 | 2.98 | 2.42 | 2.55 | 2.02 | 59.08 |
