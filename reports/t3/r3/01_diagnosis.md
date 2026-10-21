# T3-R3, P1: diagnosis of the ten `bw` failures at T = 3

Date: 2026-10-21. Branch `t3-r3`, code commit `4e122a7d` (all diagnosis steps and the replays were run at it, on a tree whose only changes were untracked results and reports). **This file is generated** from `reports/t3/r3/report_scripts/01_diagnosis.template.md` by `build_report.py`; every table is written by `diagnosis_tables.py` from the CSVs of `results/t3_r3/diag/` (path under each table). Nothing was trained except the 14 replays of DG5; no closed-form quantity entered anything below that could trigger anything in a run (the closed form appears only in the stored evaluation columns `eta_3`, `rmse_pos_T_over_eT0`, `Gmax_full` and in the reporting of `e*(0)`).

## 0. Result

1. **The ten failures fall into three working classes, and none is (E) or (U).** (F) formation failure: 1 run (T3-R2 q = 60 seed 60518). (K) clamp regime with a persistent in-band or stage-2 error: 8 runs. (V) last-iterate variability: 1 run (T3-R1 q = 50 seed 50517). The independent calculator reproduces the stored `eta_3` of all 14 replay-set runs to 3e-16 (§5), the replays reproduce the originals bit for bit (§6), so no implementation error was found.
2. **60518 is a formation failure of a different kind than the rest, and it is not a clamp case** (S3 = 0). Every run passes through a collapse of the effort to a low, almost gap-independent level (the minimum of e(0) over updates 100-300 is 6.9-12.8, median 9.4) with a gap-sensitivity that falls to a minimum of 5.5e-5 to 6.3e-2 (median 5.3e-3); 79 of 80 runs break out of it at updates 150-700 (median 175-200). 60518 does not: its in-band range of the effort falls from 0.33 to 0 and its sensitivity from 6e-3 to 7e-10; from update 350 on its second hidden layer saturates (all 64 units |pre-activation| > 3 by update 575), the first layer never does. From update 100 the in-band and out-of-band region gradients are exactly opposed (cosine -0.99999 or -1.000), so the sum is small though each is large. Its signals up to update 100 lie inside, or at the edge of, the range of the passing q = 60 runs (§4).
3. **DP2 is not supported.** In the clamp-regime runs the out-of-band samples (sample share 0.75) carry 0.5-0.7 % of the surrogate gradient (projection share) and the draws below the clamp (sample share 0.53-0.57) 0.3 %, against 0.1 % and -0.008 % in the non-regime runs. The tie-point effort does not fall after the onset (it rises by a median of 3.7 effort units in the next 250 updates at q = 50). What the data do show is a marker and a correlate (§7): S3 > 0.2 coincides with the Beta mean at the out-of-band nodes reaching the actor's mean clamp 1e-6 (21 of 30 regime runs at some export, 0 of 50 others), with a minimum alpha of order 1e-4 (0.4 in the other runs), an entropy of order -1000, 2.7 times larger pre-clip gradient norms, 1.5 times larger KL, and a larger last-iterate drawdown of the tie-point effort. All ten stage-2 stretches are in the regime (10 of 30 against 0 of 50). Whether the regime causes the in-band errors is **not tested** here (§12).
4. **The retrospective trigger table** (§11): a gate-level trigger (development `Delta` > 0.005 at local update 1600) fires on exactly 7 of the 80 runs, all 7 of which failed, and misses 3 failures (the two stage-2 stretches of T3-R1 and 60516); it costs 3.7 % of the compute. A formation-level trigger (`R` > 0.4 ... 1 or `R_tail` > 0.02 ... 0.1) fires on 60518 alone; it costs 0.5 %. Any such threshold is set from one observation.

**One-line answer to the coworker's question (DG7).** The q = 50 tie-point deficits (60501, 60506, 60513 and T3-R1's 50504) are one kind of problem, class (K): a persistent in-band bias (`eta_3` above 0.005 at 14-16 of the last 16 exports) of a run in the clamp regime (S3 0.74-0.79); 60515's excess at d = -36 and 60516's amplitude deficit are the same class with a different shape (S3 0.80 and 0.77); 60518 is not that kind of problem, it is a formation failure (S3 = 0).

## 1. What was run

| step | command | cost (wall, workers) |
|---|---|---|
| DG1-DG4, DG6, DG8, retro, DG7 | `bash results/t3_r3/run_diag.sh` (`tools/t3/t3r3_diag.py <step>`), at `4e122a7d` | 3.7 / 6.2 / 4.4 / 13.6 / 4.2 / 0.2 / 0.2 / 5.0 / 1.1 s for dg1 / dg2 / dg3 / dg6 / dg8 / retro / dg7 / dg4 / dg5, 16 workers (`results/t3_r3/diag/costs.csv`, last block) |
| DG5 | `python tools/t3/t3r3_replay.py all --workers 14` at `4e122a7d` (tmux `t3r3_replay2`) | 325 s wall, 14 workers, 304-322 s per replay (`results/t3_r3/replays/replay_identity.csv`) |

A first set of the same 14 replays was run at `3b774be8` (358 s, all 14 bit-identical); it lacked the SD of the advantage per bin and the raw statistics per clamp group that the prompt lists, so it was repeated at `4e122a7d`. The first set is kept, untracked, in `results/t3_r3/replays_3b774be8/` (for 60518 the columns `share_out`, `share_clamp_lo`, `gnorm_total`, `gnorm_in` and `sens_in_mean` of the two sets are equal in every update).

Roots (recorded in `results/t3_r3/diag/record_*.json`): T3-R1 `.../t3-bw-vs-jt-51e472/results/t3_r1`, T3-R2 `.../t3-bw-failures-audit-6f3a38/results/t3_r2`, v2.0 `.../v2-t2-refine/results/v2_T2_locked/confirmation_v2_0`. The tests of the new tools are `tests/test_t3r3_diag.py` (18) and `tests/test_t3r3_replay.py` (5), see `03_checks.md`.

## 2. DG1, the records

Source: `results/t3_r3/diag/dg1_inventory.csv`, `dg1_inventory_v20.csv`.

| records | tracked files present (of 8 kinds, all runs) | untracked files present (of 8 kinds) | stage-3 exports `u00025`-`u01600` | exports in total | `ms_binmaps_stage3.npz` `probs` all NaN |
|---|---|---|---|---|---|
| T3-R1 `bw` runs (40) | 40-40 of 40 per kind | 40-40 of 40 per kind | 2560 of 2560 | 152-152 per run | 40 of 40 |
| T3-R2 `bw` runs (40) | 40-40 of 40 per kind | 40-40 of 40 per kind | 2560 of 2560 | 152-152 per run | 40 of 40 |
| v2.0 confirmation runs (40) | `v2_updates.csv` 40 of 40 | `v2_updates.csv` 40 of 40 | - | 88-88 per run | - |

Every record exists for all 80 `bw` runs and all 40 v2.0 runs. `ms_binmaps_stage3.npz` holds `probs` = NaN for all 80 runs (bin-balanced starts), as the prompt says, so the starts per bin cannot be read from it.

**The coworker's list and the records that answer it:**

| item | answered by | needs the DG5 replay |
|---|---|---|
| Beta parameters (alpha, beta) | the 64 stage-3 exports `weights/u00025` ... `u01600` (`actor.*`) on any grid (DG2); `train_history.json`, `conc_buffer_min_mean_max` and the entropy series (whole-buffer statistics per update) | no |
| the clamp | `ms_updates.csv` `d1_L_s3_{in,out}_{n,lo,hi}` per update (tracked); the Beta mean at the mean clamp from the exports | per-sample gradients of the clamped draws: yes |
| advantage and critic signals | `ms_updates.csv` `adv_raw_mean`, `adv_raw_std`, `value_loss`, `policy_loss`; the critic `V(d)` from the exports (`critic.*`) | per region and bin: yes |
| gradients | `train_history.json` whole-network gradient norms per update, pre-clip (untracked; present for all 80 runs) | per region, per bin, per draw group: yes |
| state coverage | `ms_updates.csv` `n_start_{tail,near,mid}` (strata counts only) | starts per bin: yes |
| deviation-gain curves | exports with the T3-R2 calculator (DG4); `ms_checks_stage3.csv` (`Delta`, `R`, ...) | no |

## 3. DG2, the stage-3 policy over training (80 runs, 64 exports each)

Source: `results/t3_r3/diag/dg2_exports.csv` (5,120 rows), `dg2_runs.csv`; arrays `results/t3_r3/diag/arrays/dg2_*.npz` (untracked). The grid is the final-tier grid of D_3 (state step 2), `|d| < 2q` is the band.

**Formation.** The formation export is the first export at which F(u) = e(0) - mean{e(d): |d| >= 2q} exceeds half of the median final F of the passing runs of the same q (both rounds pooled).

| q | runs | formed (F exceeds the threshold) | threshold (half the median final F of the passing runs, effort units) | formation export, min / median / max | formed by u250 | formed after u250 | not formed |
|---|---|---|---|---|---|---|---|
| 50 | 40 | 40 of 40 | 32.11 | 150 / 175 / 700 | 36 | 4 |  |
| 60 | 40 | 39 of 40 | 26.78 | 150 / 200 / 325 | 37 | 2 | 60518 |

79 of 80 runs form, at update 150-700 (36 of 40 at q = 50 and 37 of 39 at q = 60 by update 250). 60518 never does.

| q | group | runs | F(u01600), median | e_3(0), median | in-band range, median | out-of-band mean effort, median | share of out nodes with alpha < 1, median | |d mu / d input 2| in band, median | layer-2 saturation in band, median |
|---|---|---|---|---|---|---|---|---|---|
| 50 | passing | 34 | 64.22 | 64.56 | 61.57 | 0.3347 | 0.8576 | 2.494 | 0.4632 |
| 50 | failing, formed | 6 | 59.92 | 60.12 | 56.65 | 0.2467 | 0.9023 | 2.285 | 0.2955 |
| 60 | passing | 36 | 53.56 | 53.83 | 51.45 | 0.3279 | 0.8525 | 1.894 | 0.4867 |
| 60 | failing, formed | 3 | 51.74 | 51.98 | 49.18 | 0.1621 | 0.913 | 1.819 | 0.3615 |
| 60 | failing, not formed (60518) | 1 | -1.78e-15 | 8.155 | 0 | 8.155 | 0 | 6.82e-10 | 1 |

**The trajectory of 60518** against a run that broke out at updates 150-250 (T3-R2 q = 60 seed 60501, formation export 250) and the median of the passing q = 60 runs:

| update | 60518 e(0) | 60518 in-band range | 60518 sens | 60518 sat2 | 60501 (formation export 250) e(0) | 60501 sens | 60501 sat2 | passing q60 median e(0) | median sens |
|---|---|---|---|---|---|---|---|---|---|
| 25 | 35.94 | 0.3321 | 0.006191 | 0 | 37.7 | 0.007359 | 0 | 36.72 | 0.007348 |
| 50 | 21.37 | 0.356 | 0.006637 | 0 | 23.23 | 0.002251 | 0 | 22.45 | 0.003227 |
| 75 | 12.12 | 0.4156 | 0.007748 | 0 | 13.16 | 0.01211 | 0 | 12.61 | 0.008294 |
| 100 | 8.101 | 0.2392 | 0.00446 | 0 | 8.985 | 0.005706 | 0 | 9.044 | 0.005012 |
| 125 | 7.513 | 0.08505 | 0.001586 | 0 | 7.786 | 0.001908 | 0 | 10.3 | 0.03744 |
| 150 | 7.036 | 0.03716 | 0.000693 | 0 | 7.886 | 0.000286 | 0 | 18.75 | 0.2809 |
| 175 | 6.987 | 0.01428 | 0.000266 | 0 | 8.979 | 0.02387 | 0 | 31.51 | 0.8013 |
| 200 | 7.905 | 0.0507 | 0.000945 | 0 | 13.79 | 0.1359 | 0.005121 | 40.43 | 1.229 |
| 250 | 7.407 | 0.03392 | 0.000632 | 0 | 38.24 | 1.11 | 0.02127 | 46.95 | 1.557 |
| 300 | 7.806 | 0.002939 | 5.49e-05 | 0 | 48.9 | 1.636 | 0.07379 | 48.52 | 1.656 |
| 350 | 7.438 | 0.004954 | 9.24e-05 | 0.0625 | 50.64 | 1.747 | 0.1429 | 49.32 | 1.676 |
| 400 | 7.027 | 0.002746 | 5.12e-05 | 0.2169 | 50.16 | 1.751 | 0.1666 | 49.81 | 1.719 |
| 500 | 7.131 | 0.000243 | 4.58e-06 | 0.7969 | 52.43 | 1.843 | 0.2466 | 50.14 | 1.728 |
| 575 | 8.248 | 0.000491 | 9.1e-06 | 1 | 52.04 | 1.783 | 0.2623 | 50.34 | 1.737 |
| 800 | 8.173 | 1.68e-05 | 3.76e-07 | 1 | 51.6 | 1.801 | 0.3567 | 52.14 | 1.811 |
| 1200 | 7.998 | 0 | 6.17e-10 | 1 | 56.26 | 2.01 | 0.4253 | 53.2 | 1.876 |
| 1600 | 8.155 | 0 | 6.82e-10 | 1 | 53.11 | 1.872 | 0.4961 | 53.83 | 1.894 |

All runs collapse to a low effort (6.9-12.8 at its minimum over updates 100-300) with a small in-band range and a sensitivity that reaches its minimum of 5.5e-5 to 6.3e-2 in updates 100-300 (the table below gives the deepest). 60501 breaks out between updates 150 and 250 (its sensitivity rises from 2.9e-4 at update 150 to 0.14 at 200 and 1.1 at 250, its tie-point effort from 7.9 to 13.8 and 38.2). 60518 does not: the range keeps falling (0.037 at 150, 0.003 at 300, 0 from update 1200), the sensitivity with it. The second-layer saturation of 60518 is 0 until update 300 and 0.06, 0.22, 0.80, 1.0 at updates 350, 400, 500, 575. So the saturation is not what prevented the break-out at updates 125-300 (it is zero then); it follows the loss of sensitivity and then locks the state (the first layer is never saturated, `sat1_in` = 0 at every export of the run).

**How deep the bottleneck is predicts when a run breaks out.**

Spearman correlation with the formation export over the 79 runs that formed: sensitivity at u150 -0.921 (p = 2.9e-33), at u125 -0.882, at u100 -0.731 (the formation exports are 150 or later, so these two are measured before any formation); minimum sensitivity over u100-u300 -0.806 (p = 3.0e-19). Over all 80 runs the minimum of e(0) over u100-u300 is 6.9-12.8 (median 9.4) and the minimum sensitivity 5.5e-05-6.3e-02 (median 5.3e-03).

| run | minimum of |d mu / d input 2| over u100-u300 | at update | formation export | pass |
|---|---|---|---|---|
| T3-R2 q60 s60518 | 5.49e-05 | 300 | - | no |
| T3-R1 q50 s50504 | 6.48e-05 | 175 | 400 | no |
| T3-R1 q50 s50515 | 0.000102 | 175 | 700 | yes |
| T3-R1 q60 s50503 | 0.000102 | 125 | 325 | yes |
| T3-R1 q50 s50518 | 0.000149 | 100 | 225 | yes |
| T3-R2 q50 s60511 | 0.000165 | 125 | 250 | yes |

Formation export >= 325 (descriptive): 3 of 5 such runs fail, against 6 of 74 with an earlier formation (Fisher exact p = 0.009; 60518 excluded, it never formed).

The depth of the bottleneck (60518 has the smallest minimum of all 80 runs, 5.5e-5, not much below 6.5e-5 of T3-R1's 50504, which broke out at update 400) predicts the formation export: the sensitivity at update 150 has Spearman -0.92, and the values at updates 125 and 100, which precede every formation, -0.88 and -0.73. The late-formation count above is a post-hoc split at 325 and descriptive.

## 4. DG3, the training signal (stage-3 phase)

Sources: `dg3_checks.csv`, `dg3_windows.csv`, `dg3_history_windows.csv` (from `ms_updates.csv`, `ms_checks_stage3.csv`, `train_history.json`).

**60518 against the 18 passing q = 60 runs of T3-R2, development checks:**

| quantity | 60518 | 18 passing q60 runs of T3-R2: min | median | max |
|---|---|---|---|---|
| Delta, local 25 | 0.0973 | 0.09188 | 0.1005 | 0.1189 |
| Delta, local 50 | 0.07855 | 0.06628 | 0.07308 | 0.07703 |
| Delta, local 100 | 0.1451 | 0.1195 | 0.1393 | 0.1449 |
| Delta, local 200 | 0.1461 | 0.003999 | 0.0162 | 0.1398 |
| Delta, local 400 | 0.1512 | 0.002347 | 0.004602 | 0.01481 |
| Delta, local 800 | 0.1446 | 0.000803 | 0.002729 | 0.007434 |
| Delta, local 1600 | 0.1447 | 0.000525 | 0.001289 | 0.004065 |
| s (best-response effort at d = 0), local 100 | 41.9 | 41.91 | 42.23 | 43.44 |
| s, local 400 | 41.55 | 53.16 | 55.53 | 56.59 |
| s, local 1600 | 41.92 | 55.57 | 56.91 | 57.7 |
| R, local 100 | 1.183 | 1.043 | 1.157 | 1.181 |
| R, local 400 | 1.219 | 0.1275 | 0.1688 | 0.3131 |
| R, local 1600 | 1.18 | 0.06442 | 0.09455 | 0.163 |
| R_tail, local 100 | 0.1944 | 0.1879 | 0.21 | 0.2255 |
| R_tail, local 400 | 0.1691 | 0.01097 | 0.01633 | 0.02385 |
| R_tail, local 1600 | 0.1945 | 0.002779 | 0.005819 | 0.01031 |
| C, local 100 | 0.02548 | 0.0251 | 0.02673 | 0.03046 |
| C, local 400 | 0.01874 | 0.03817 | 0.04095 | 0.04463 |
| C, local 1600 | 0.01477 | 0.02928 | 0.03133 | 0.03938 |
| first check with Delta < 0.05 (local) | - | 150 | 175 | 300 |
| checks with Delta in [0.12, 0.15] (of 64) | 53 | 0 | 2 | 5 |

At local updates 25-100, 60518 is inside, or at the edge of, the passing range; by local 200 the passing runs have left the level of 0.12-0.15 and 60518 never does (53 of 64 checks in that interval, against 0-5). `R` and `R_tail` have the same picture.

**Training signals by window:**

| local updates | quantity | 60518 | passing q60 (T3-R2): min | median | max |
|---|---|---|---|---|---|
| 1-100 | starts in the tail stratum (sum over the window) | 3.716e+04 | 3.706e+04 | 3.719e+04 | 3.746e+04 |
| 1-100 | starts in the near-tie stratum | 2295 | 2237 | 2318 | 2368 |
| 1-100 | starts in the middle stratum | 1.175e+04 | 1.148e+04 | 1.169e+04 | 1.182e+04 |
| 1-100 | mean effort | 24.78 | 24.78 | 25.47 | 26.76 |
| 1-100 | kl_final_epoch | 0.007396 | 0.006337 | 0.006797 | 0.00746 |
| 1-100 | clip_frac | 0.1046 | 0.1012 | 0.1084 | 0.1156 |
| 1-100 | adv_raw_std | 0.3614 | 0.3307 | 0.3579 | 0.3872 |
| 1-100 | value_loss | 0.2627 | 0.2033 | 0.2372 | 0.3517 |
| 1-100 | actor grad norm (pre-clip, mean) | 1.723 | 1.638 | 1.71 | 1.799 |
| 1-100 | critic grad norm (mean) | 0.5361 | 0.4062 | 0.4882 | 0.6371 |
| 1-100 | entropy (effort scale, after the update) | 2.735 | 2.747 | 2.752 | 2.785 |
| 1-100 | concentration alpha + beta (mean over the buffer) | 108.7 | 106.2 | 107.4 | 108.6 |
| 101-400 | starts in the tail stratum (sum over the window) | 1.12e+05 | 1.11e+05 | 1.12e+05 | 1.12e+05 |
| 101-400 | starts in the near-tie stratum | 6987 | 6829 | 6970 | 7103 |
| 101-400 | starts in the middle stratum | 3.484e+04 | 3.448e+04 | 3.49e+04 | 3.518e+04 |
| 101-400 | mean effort | 7.51 | 8.291 | 8.916 | 9.252 |
| 101-400 | kl_final_epoch | 0.002672 | 0.003446 | 0.004265 | 0.005402 |
| 101-400 | clip_frac | 0.03466 | 0.03226 | 0.03684 | 0.05097 |
| 101-400 | adv_raw_std | 0.03371 | 0.04263 | 0.04711 | 0.04837 |
| 101-400 | value_loss | 0.000573 | 0.000907 | 0.00109 | 0.001156 |
| 101-400 | actor grad norm (pre-clip, mean) | 1.594 | 1.858 | 2.078 | 3.514 |
| 101-400 | critic grad norm (mean) | 0.01694 | 0.02111 | 0.02339 | 0.02393 |
| 101-400 | entropy (effort scale, after the update) | 2.145 | -2.083 | 1.709 | 1.896 |
| 101-400 | concentration alpha + beta (mean over the buffer) | 155.1 | 126.3 | 135.3 | 145.8 |
| 401-800 | starts in the tail stratum (sum over the window) | 1.49e+05 | 1.49e+05 | 1.49e+05 | 1.49e+05 |
| 401-800 | starts in the near-tie stratum | 9285 | 9118 | 9295 | 9443 |
| 401-800 | starts in the middle stratum | 4.652e+04 | 4.606e+04 | 4.664e+04 | 4.674e+04 |
| 401-800 | mean effort | 7.677 | 8.035 | 8.277 | 8.598 |
| 401-800 | kl_final_epoch | 0.003176 | 0.003545 | 0.004084 | 0.009297 |
| 401-800 | clip_frac | 0.03963 | 0.03322 | 0.03733 | 0.06334 |
| 401-800 | adv_raw_std | 0.02598 | 0.0377 | 0.03943 | 0.04443 |
| 401-800 | value_loss | 0.000335 | 0.000704 | 0.000768 | 0.000976 |
| 401-800 | actor grad norm (pre-clip, mean) | 2.201 | 2.426 | 2.762 | 9.89 |
| 401-800 | critic grad norm (mean) | 0.01675 | 0.02015 | 0.02156 | 0.02347 |
| 401-800 | entropy (effort scale, after the update) | 1.985 | -928.8 | 0.8958 | 1.246 |
| 401-800 | concentration alpha + beta (mean over the buffer) | 220.9 | 139.9 | 169.5 | 193.7 |
| 1201-1600 | starts in the tail stratum (sum over the window) | 1.49e+05 | 1.48e+05 | 1.49e+05 | 1.5e+05 |
| 1201-1600 | starts in the near-tie stratum | 9210 | 9171 | 9300 | 9625 |
| 1201-1600 | starts in the middle stratum | 4.654e+04 | 4.588e+04 | 4.656e+04 | 4.703e+04 |
| 1201-1600 | mean effort | 7.711 | 8.043 | 8.173 | 8.34 |
| 1201-1600 | kl_final_epoch | 0.003238 | 0.003489 | 0.003952 | 0.006641 |
| 1201-1600 | clip_frac | 0.02984 | 0.02922 | 0.03213 | 0.04919 |
| 1201-1600 | adv_raw_std | 0.02018 | 0.02994 | 0.03222 | 0.04036 |
| 1201-1600 | value_loss | 0.000201 | 0.000442 | 0.00051 | 0.000795 |
| 1201-1600 | actor grad norm (pre-clip, mean) | 2.554 | 2.898 | 3.349 | 10.76 |
| 1201-1600 | critic grad norm (mean) | 0.01085 | 0.01434 | 0.01528 | 0.01857 |
| 1201-1600 | entropy (effort scale, after the update) | 1.79 | -1999 | 0.1406 | 0.8845 |
| 1201-1600 | concentration alpha + beta (mean over the buffer) | 331 | 165.1 | 241.5 | 272.9 |

Up to update 100 the signals of 60518 are inside the passing range or at its edge (`kl_final_epoch`, `clip_frac`, `adv_raw_std`, the gradient norms and the starts per stratum are inside; the mean effort 24.78 is the smallest of the 19 values, and the entropy 2.735 and the concentration 108.7 lie just outside the passing ranges 2.747-2.785 and 106.2-108.6). After update 100 it has a smaller advantage SD, KL and critic loss than the passing runs (there is less to learn when nothing depends on the gap) and a concentration alpha + beta above every passing q = 60 run (331 against 165-273 in the last window), an entropy of 1.8 where the passing runs go down to -1999 (a tail Beta that has collapsed to the lower end, see §7).

**All 80 runs by clamp class, local updates 1201-1600:**

| q | S3 class | runs | failures | kl_final_epoch | clip_frac | actor grad norm, mean | actor grad norm, max | entropy (effort scale) | concentration (mean) | min alpha on the D_3 grid at u01600 |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | regime | 16 | 5 | 0.005878 | 0.04357 | 8.382 | 51.36 | -1387 | 167.9 | 0.000177 |
| 50 | mid | 4 | 0 | 0.003885 | 0.03226 | 3.707 | 18.13 | -0.5402 | 217.7 | 0.1308 |
| 50 | low | 20 | 1 | 0.003779 | 0.0307 | 3.05 | 15.18 | 0.3007 | 236.8 | 0.3839 |
| 60 | regime | 14 | 3 | 0.005778 | 0.04308 | 8.045 | 45.05 | -1036 | 182.8 | 0.000191 |
| 60 | mid | 3 | 0 | 0.004043 | 0.03369 | 4.034 | 20.89 | -0.6211 | 216 | 0.1233 |
| 60 | low | 23 | 1 | 0.003693 | 0.031 | 3.033 | 15.25 | 0.3846 | 243.7 | 0.4515 |

The regime runs have higher KL, clip fraction, pre-clip gradient norm (mean 8.0-8.4 against 3.0-3.1, maximum 45-51 against 15), a concentration 60-70 lower, an entropy of order -1000 and a minimum alpha of 2e-4 against 0.4.

## 5. DG4, the deviation-gain curves (14 runs, exports `u00100`, `u00400`, `u00800`, `u01600`)

Source: `dg4_profiles.csv`, `dg4_curves.csv` (T3-R2 calculator `tools/t3/t3r2_br_calc.py`, default resolution; the export's network at stage 3 against itself).

Over the 952 curves Q_3(d, .) of DG4 (14 runs x 4 exports x up to 17 gaps), 952 have exactly one local maximum on the effort grid of step 0.01 (0 have another number).

**Agreement with the stored verifier values:**

| run | stored eta_3 (final-tier verifier) | calculator max Delta_3 at u01600 | difference |
|---|---|---|---|
| q50 s60501 | 0.005241 | 0.005241 | 0 |
| q50 s60502 | 0.004654 | 0.004654 | 0 |
| q50 s60504 | 0.001502 | 0.001502 | 1e-16 |
| q50 s60505 | 0.002287 | 0.002287 | 3e-16 |
| q50 s60506 | 0.007244 | 0.007244 | 9.97e-17 |
| q50 s60507 | 0.004401 | 0.004401 | 0 |
| q50 s60513 | 0.005466 | 0.005466 | 9.97e-17 |
| q50 s60515 | 0.01138 | 0.01138 | 0 |
| q60 s60501 | 0.00166 | 0.00166 | 0 |
| q60 s60502 | 0.001316 | 0.001316 | 1e-16 |
| q60 s60503 | 0.002194 | 0.002194 | 2e-16 |
| q60 s60509 | 0.002923 | 0.002923 | -9.97e-17 |
| q60 s60516 | 0.003765 | 0.003765 | 0 |
| q60 s60518 | 0.1447 | 0.1447 | 1.11e-16 |

Largest absolute difference over the 14 runs: 3.00e-16 (units of DW).

**The maximum of Delta_3 over D_3 at the four exports** (`max Delta (d*; learned effort, best response)`):

| run | S3 | pass | u100: max Delta (d*; e, BR) | u400: max Delta (d*; e, BR) | u800: max Delta (d*; e, BR) | u1600: max Delta (d*; e, BR) |
|---|---|---|---|---|---|---|
| q50 s60501 | 0.7758 | no | 0.1962 (-22; 9.1, 54.0) | 0.0075 (-38; 48.7, 30.0) | 0.0077 (-64; 22.3, 41.2) | 0.0052 (-4; 60.0, 67.5) |
| q50 s60502 | 0.02438 | yes | 0.1741 (-22; 12.6, 55.4) | 0.0061 (-40; 47.7, 30.8) | 0.0030 (-74; 14.9, 26.7) | 0.0047 (-20; 62.9, 48.2) |
| q50 s60504 | 0.01811 | yes | 0.1906 (-22; 10.0, 54.4) | 0.0149 (-70; 13.8, 40.2) | 0.0044 (-4; 60.9, 67.7) | 0.0015 (-2; 64.7, 68.6) |
| q50 s60505 | 0.6558 | yes | 0.1939 (-22; 9.5, 54.1) | 0.0157 (-36; 51.0, 24.0) | 0.0036 (-74; 17.8, 30.8) | 0.0023 (-2; 63.4, 68.1) |
| q50 s60506 | 0.7887 | no | 0.1933 (-22; 9.6, 54.1) | 0.0349 (-82; 27.4, 0.0) | 0.0139 (-8; 53.6, 66.5) | 0.0072 (-6; 58.1, 67.7) |
| q50 s60507 | 0.7516 | yes | 0.1888 (-22; 10.3, 54.5) | 0.0105 (-38; 51.9, 29.8) | 0.0185 (-36; 57.1, 27.7) | 0.0044 (-74; 13.3, 27.7) |
| q50 s60513 | 0.7439 | no | 0.1894 (-22; 10.2, 54.4) | 0.0144 (-66; 14.4, 40.3) | 0.0083 (-28; 58.4, 38.7) | 0.0055 (-4; 59.8, 67.6) |
| q50 s60515 | 0.7965 | no | 0.1927 (-22; 9.6, 54.2) | 0.0326 (-10; 45.2, 63.9) | 0.0124 (-36; 50.8, 26.7) | 0.0114 (-36; 53.4, 30.3) |
| q60 s60501 | 0.02499 | yes | 0.1401 (-20; 9.0, 48.7) | 0.0037 (-4; 50.3, 56.9) | 0.0027 (-4; 51.3, 57.5) | 0.0017 (-2; 53.0, 57.3) |
| q60 s60502 | 0.02769 | yes | 0.1295 (-20; 10.8, 49.4) | 0.0034 (-4; 50.6, 57.1) | 0.0020 (-2; 52.3, 56.9) | 0.0013 (-2; 53.5, 57.3) |
| q60 s60503 | 0.7308 | yes | 0.1398 (-20; 9.0, 48.8) | 0.0070 (-4; 47.3, 56.2) | 0.0051 (-42; 43.2, 31.5) | 0.0022 (-2; 52.1, 57.0) |
| q60 s60509 | 0.7367 | yes | 0.1196 (-18; 12.7, 49.3) | 0.0064 (-4; 47.8, 56.2) | 0.0021 (52; 37.4, 33.0) | 0.0029 (-32; 47.5, 38.6) |
| q60 s60516 | 0.7736 | no | 0.1381 (-20; 9.3, 48.8) | 0.0069 (-4; 47.3, 56.0) | 0.0064 (-4; 47.8, 56.5) | 0.0038 (-4; 50.2, 56.9) |
| q60 s60518 | 0 | no | 0.1451 (-20; 8.1, 48.5) | 0.1512 (-20; 7.0, 48.1) | 0.1446 (-20; 8.2, 48.5) | 0.1447 (-20; 8.2, 48.5) |

All 14 runs have the same picture at update 100: a policy at 9-13 effort units against a best response of 48-55, Delta 0.12-0.20. The passing runs leave it by update 400; 60518 sits there at every export. The curves at selected gaps:

| run | export | d | learned effort | slope dQ/de there | one-step best response | Delta/DW | local maxima of Q_3(d, .) (efforts) |
|---|---|---|---|---|---|---|---|
| q60 s60518 | u01600 | 0 | 8.155 | 0.02867 | 41.92 | 0.121 | 41.92 |
| q60 s60518 | u01600 | -4 | 8.155 | 0.02756 | 43.23 | 0.1295 | 43.23 |
| q60 s60518 | u01600 | -6 | 8.155 | 0.02701 | 43.88 | 0.133 | 43.88 |
| q60 s60518 | u01600 | -20 | 8.155 | 0.02312 | 48.46 | 0.1447 | 48.46 |
| q60 s60518 | u01600 | -36 | 8.155 | 0.01867 | 53.7 | 0.1301 | 53.7 |
| q60 s60518 | u01600 | 40 | 8.155 | 0.01756 | 28.84 | 0.0454 | 28.84 |
| q60 s60518 | u00100 | 0 | 8.101 | 0.0287 | 41.9 | 0.1213 | 41.9 |
| q60 s60518 | u00100 | -4 | 8.096 | 0.02759 | 43.21 | 0.1298 | 43.21 |
| q60 s60518 | u00100 | -6 | 8.094 | 0.02704 | 43.87 | 0.1333 | 43.87 |
| q60 s60518 | u00100 | -20 | 8.08 | 0.02315 | 48.45 | 0.1451 | 48.45 |
| q60 s60518 | u00100 | -36 | 8.065 | 0.0187 | 53.69 | 0.1306 | 53.69 |
| q60 s60518 | u00100 | 40 | 8.142 | 0.01755 | 28.8 | 0.04532 | 28.8 |
| q50 s60501 | u01600 | 0 | 60.11 | 0.005649 | 65.93 | 0.004106 | 65.93 |
| q50 s60501 | u01600 | -4 | 60.05 | 0.004155 | 67.48 | 0.005241 | 67.48 |
| q50 s60501 | u01600 | -6 | 59.9 | 0.003473 | 68.21 | 0.00508 | 68.21 |
| q50 s60501 | u01600 | -20 | 56.75 | -6.59e-05 | 56.37 | 3.16e-06 | 56.37 |
| q50 s60501 | u01600 | -36 | 48.61 | -0.00149 | 39.91 | 0.00162 | 39.91 |
| q50 s60501 | u01600 | 40 | 43.95 | -0.000368 | 43.58 | 1.75e-05 | 43.58 |
| q50 s60515 | u01600 | 0 | 64 | 0.003431 | 67.53 | 0.001515 | 67.53 |
| q50 s60515 | u01600 | -4 | 63.98 | 0.001939 | 69.07 | 0.001734 | 69.07 |
| q50 s60515 | u01600 | -6 | 63.86 | 0.001253 | 69.79 | 0.001095 | 69.79 |
| q50 s60515 | u01600 | -20 | 61.07 | -0.002392 | 47.12 | 0.004172 | 47.12 |
| q50 s60515 | u01600 | -36 | 53.39 | -0.00395 | 30.35 | 0.01138 | 30.35 |
| q50 s60515 | u01600 | 40 | 48.03 | -0.002375 | 45.58 | 0.000726 | 45.58 |
| q60 s60516 | u01600 | 0 | 50.22 | 0.004637 | 55.68 | 0.003165 | 55.68 |
| q60 s60516 | u01600 | -4 | 50.22 | 0.003569 | 56.94 | 0.003765 | 56.94 |
| q60 s60516 | u01600 | -6 | 50.16 | 0.003069 | 57.55 | 0.003482 | 57.55 |
| q60 s60516 | u01600 | -20 | 48.68 | 0.00019 | 49.32 | 1.54e-05 | 49.32 |
| q60 s60516 | u01600 | -36 | 44.7 | -0.001754 | 38.72 | 0.00131 | 38.72 |
| q60 s60516 | u01600 | 40 | 41.46 | -0.000952 | 40.34 | 0.000133 | 40.34 |

For 60518 the learned effort 8.155 is on the rising side of the single maximum at every gap (slope +0.029 at d = 0, best response 41.9). For the q = 50 failures at u01600 the largest gain is at a small gap and below the best response (60501: slope +0.0042 at d = -4, effort 60.05, best response 67.5); for 60515 it is at d = -36 and above it (effort 53.4, best response 30.3, slope -0.0040: an excess); for 60516 the tent is lower overall (effort 50.2 at d = 0, best response 55.7).

## 6. DG5, the instrumented replays (14 runs)

Source: `results/t3_r3/replays/q*/seed*/` (`replay_updates.csv`, `replay_bins.npz`, `replay_identity.json`), `dg5_*.csv`.

| q | seed | 64 exports | stage-3 rows of ms_updates.csv | ms_checks_stage3.csv | all identical | wall time (s) |
|---|---|---|---|---|---|---|
| 50 | 60501 | identical | identical | identical | yes | 321.8 |
| 50 | 60506 | identical | identical | identical | yes | 306.4 |
| 50 | 60513 | identical | identical | identical | yes | 312.5 |
| 50 | 60515 | identical | identical | identical | yes | 303.8 |
| 60 | 60516 | identical | identical | identical | yes | 319.3 |
| 60 | 60518 | identical | identical | identical | yes | 316 |
| 50 | 60505 | identical | identical | identical | yes | 312.6 |
| 50 | 60507 | identical | identical | identical | yes | 305.3 |
| 60 | 60503 | identical | identical | identical | yes | 319.1 |
| 60 | 60509 | identical | identical | identical | yes | 313.7 |
| 50 | 60502 | identical | identical | identical | yes | 306 |
| 50 | 60504 | identical | identical | identical | yes | 312.3 |
| 60 | 60501 | identical | identical | identical | yes | 317.8 |
| 60 | 60502 | identical | identical | identical | yes | 319.5 |

Freeze arrays `freeze_stage3_{final,development}.npz`, rollout clamp counters against the replay's stored-action counts and global-RNG assertions: 14 of 14 replays bit-identical in all items; global-RNG violations in total 0.

**The instrumentation is passive and the replays are replays**: all 14 reproduce the 64 exports, the stage-3 rows of `ms_updates.csv`, `ms_checks_stage3.csv` and the freeze arrays bit for bit (the passivity test on a reduced config is in `03_checks.md`). The number of draws at the stored clamp value equals the rollout's own `d1_L_s3_{in,out}_{lo,hi}` in every update of every replay.

**Gradient shares** (pooled projection share `sum <g_r, g> / sum |g|^2` over the window against the sample share; medians over the runs of each group):

| group | updates | sample share out of band | gradient share out of band | sample share, draws at the lower clamp | gradient share, draws at the lower clamp | gradient share in band | |g| total | |g_in| | |g_out| |
|---|---|---|---|---|---|---|---|---|---|
| failures (6) | 1-100 | 0.7476 | 1.109 | 0 | 0 | -0.1093 | 2.259 | 1.08 | 3.076 |
| failures (6) | 101-400 | 0.7487 | -0.1117 | 0.04623 | -0.000149 | 1.112 | 2.311 | 3.997 | 2.678 |
| failures (6) | 401-800 | 0.7497 | 0.01695 | 0.3928 | 0.003114 | 0.9831 | 4.305 | 5.203 | 0.5604 |
| failures (6) | 801-1200 | 0.7495 | 0.007904 | 0.5549 | 0.002139 | 0.9921 | 5.925 | 6.618 | 0.467 |
| failures (6) | 1201-1600 | 0.7496 | 0.006692 | 0.5724 | 0.002821 | 0.9933 | 6.014 | 6.617 | 0.4038 |
| passing, clamp regime (4) | 1-100 | 0.7376 | 1.127 | 0 | 0 | -0.1269 | 2.388 | 1.161 | 3.303 |
| passing, clamp regime (4) | 101-400 | 0.7395 | -0.1032 | 0.04655 | -9.98e-05 | 1.103 | 2.096 | 2.946 | 1.259 |
| passing, clamp regime (4) | 401-800 | 0.7384 | 0.01058 | 0.3531 | 0.003019 | 0.9894 | 3.848 | 3.823 | 0.3072 |
| passing, clamp regime (4) | 801-1200 | 0.7382 | 0.00577 | 0.4917 | 0.003097 | 0.9942 | 4.96 | 4.928 | 0.3843 |
| passing, clamp regime (4) | 1201-1600 | 0.7384 | 0.004902 | 0.5325 | 0.002907 | 0.9951 | 5.46 | 5.435 | 0.4123 |
| passing, not in the regime (4) | 1-100 | 0.7368 | 1.114 | 0 | 0 | -0.1137 | 2.44 | 1.018 | 3.228 |
| passing, not in the regime (4) | 101-400 | 0.7383 | -0.1013 | 6.84e-05 | -2.1e-05 | 1.101 | 1.741 | 2.239 | 0.7053 |
| passing, not in the regime (4) | 401-800 | 0.7387 | 0.001415 | 0.003 | -6.99e-05 | 0.9986 | 2.11 | 2.107 | 0.1142 |
| passing, not in the regime (4) | 801-1200 | 0.7391 | 0.000731 | 0.01081 | -0.000122 | 0.9993 | 2.48 | 2.479 | 0.09036 |
| passing, not in the regime (4) | 1201-1600 | 0.738 | 0.001291 | 0.01821 | -8.04e-05 | 0.9987 | 2.581 | 2.579 | 0.07326 |

In the first 100 updates the out-of-band samples (sample share 0.75) carry 111 % of the projection (the in-band samples -11 %) in **all three groups**: the descent from effort 50 to about 8 is driven by the out-of-band starts. From update 400 on, the out-of-band samples carry 0.1-0.7 % and the draws below the clamp 0.3 % (regime) or 0 (non-regime); the draws below the clamp are 53-57 % of the samples of the regime runs. **A disproportionate share is not found in either group.**

**60518 compared with the five formed failures:**

| updates | |g| | |g_in| | |g_out| | cos(g_in, g_out) | share in | share out | sensitivity | layer-1 saturation | layer-2 saturation |
|---|---|---|---|---|---|---|---|---|---|
| 1-100 | 2.435 | 0.9915 | 3.082 | -0.4207 | -0.06599 | 1.066 | 0.005114 | 0 | 0 |
| 101-400 | 1.176 | 5.227 | 5.194 | -1 | 0.7008 | 0.2992 | 0.000468 | 0 | 0.02504 |
| 401-800 | 1.66 | 7.097 | 7.218 | -1 | 0.3648 | 0.6352 | 1.85e-05 | 0 | 0.8661 |
| 801-1200 | 1.97 | 8.503 | 8.575 | -1 | 0.4738 | 0.5262 | 2.03e-07 | 0 | 1 |
| 1201-1600 | 1.945 | 9.389 | 9.354 | -1 | 0.6642 | 0.3358 | 8.02e-09 | 0 | 1 |

Medians of the five other failures (formed, in the clamp regime):

| updates | |g| | |g_in| | |g_out| | cos(g_in, g_out) | sensitivity | layer-2 saturation |
|---|---|---|---|---|---|---|
| 1-100 | 2.252 | 1.09 | 3.07 | -0.5094 | 0.007523 | 0 |
| 101-400 | 2.339 | 3.259 | 1.595 | -0.4723 | 1.102 | 0.03053 |
| 401-800 | 4.334 | 4.292 | 0.441 | 0.0319 | 2.021 | 0.1476 |
| 801-1200 | 5.957 | 5.896 | 0.4614 | 0.02995 | 2.29 | 0.2779 |
| 1201-1600 | 6.113 | 6.07 | 0.3947 | 0.0435 | 2.221 | 0.318 |

For 60518 the in-band and out-of-band region gradients are exactly opposed from update 101 on (cosine -1.000), each of norm 5-9, with a net norm of 1.2-2.0; the formed failures have a cosine of about 0.03 after update 400 and a small out-of-band norm. The network of 60518 is a function of one parameter (the common effort level), on which the two regions push in opposite directions; the directions that would introduce a dependence on d are those blocked by the lost sensitivity.

**Coverage** (updates 1201-1600):

| run | in-band share of starts | bins | in-band bins | min / max bin share of starts | in-band share of the gradient (pooled) | gradient share of the 6 bins with |d| <= 60 |
|---|---|---|---|---|---|---|
| q50 s60501 | 0.2501 | 40 | 10 | 0.0240 / 0.0256 | 0.9931 | 0.9441 |
| q50 s60506 | 0.2484 | 40 | 10 | 0.0245 / 0.0262 | 0.9885 | 0.9182 |
| q50 s60513 | 0.2498 | 40 | 10 | 0.0239 / 0.0258 | 0.9955 | 0.9396 |
| q50 s60515 | 0.2508 | 40 | 10 | 0.0242 / 0.0257 | 0.9935 | 0.928 |
| q60 s60516 | 0.2722 | 44 | 12 | 0.0221 / 0.0231 | 0.9961 | 0.8824 |
| q60 s60518 | 0.2722 | 44 | 12 | 0.0221 / 0.0234 | 0.6642 | 0.6195 |
| q50 s60505 | 0.2479 | 40 | 10 | 0.0244 / 0.0256 | 0.9935 | 0.929 |
| q50 s60507 | 0.2497 | 40 | 10 | 0.0243 / 0.0256 | 0.9981 | 0.9268 |
| q60 s60503 | 0.275 | 44 | 12 | 0.0221 / 0.0234 | 0.9942 | 0.89 |
| q60 s60509 | 0.2735 | 44 | 12 | 0.0223 / 0.0236 | 0.996 | 0.9247 |
| q50 s60502 | 0.2512 | 40 | 10 | 0.0244 / 0.0258 | 0.9991 | 0.9305 |
| q50 s60504 | 0.2487 | 40 | 10 | 0.0242 / 0.0261 | 0.9983 | 0.9288 |
| q60 s60501 | 0.2728 | 44 | 12 | 0.0219 / 0.0236 | 1 | 0.9028 |
| q60 s60502 | 0.274 | 44 | 12 | 0.0219 / 0.0233 | 0.9983 | 0.904 |

The starts are uniform over the bins in all 14 runs (the bin share of starts lies within 5 % of 1/n_bins, in 60518 as in the others). The in-band share of the starts is 0.25 (q = 50) and 0.273 (q = 60) by design, and 98.9-100 % of the gradient is in band in all runs but 60518 (66 %, where the opposed halves nearly cancel). The per-sample gradient norm of the in-band draws is 105 / 102 (failures / passing regime) against 50 (non-regime) in the last window, and for the draws at the lower clamp 10 / 11 against 19 (`dg5_windows.csv`, window 1201-1600, medians).

## 7. DG6, the clamp regime across runs

Source: `dg6_s3_runs.csv`, `dg6_groups.csv`, `dg6_relations.csv`, `dg6_t2_analogue.csv`, `dg2_runs.csv`.

**Recomputation of the preliminary numbers: all reproduced.** S3 > 0.2 in 30 of the 80 runs (16 at q = 50, 14 at q = 60), S3 > 0.4 in 25; 8 of the 10 failures are among the 30 (8 of 30 against 2 of 50); medians by group as in the prompt (q = 50: `eta_3` 0.0033 against 0.0016, `peak_T_signed` -0.104 against -0.063; q = 60: 0.0023 / -0.089 against 0.0011 / -0.071); all ten stage-2 stretches are in the regime, the smallest S3 among them is 0.4356.

| scope | runs | S3 > 0.2 | S3 > 0.4 | S3 < 0.05 | failures in S3 > 0.2 | failures in S3 <= 0.2 | runs with a stage-2 stretch in S3 > 0.2 | runs with a stage-2 stretch in S3 <= 0.2 |
|---|---|---|---|---|---|---|---|---|
| all 80 | 80 | 30 | 25 | 43 | 8 | 2 | 10 | 0 |
| q = 50 | 40 | 16 | 13 | 20 | 5 | 1 | 7 | 0 |
| q = 60 | 40 | 14 | 12 | 23 | 3 | 1 | 3 | 0 |

Failure rate 8 of 30 ([0.123, 0.459]) against 2 of 50 ([0.005, 0.137]), Fisher exact p = 0.0048. Stage-2 stretch 10 of 30 ([0.173, 0.528]) against 0 of 50 ([0.000, 0.071]), Fisher exact p = 1.8e-05.

| scope | group | runs | failures | with a stage-2 stretch | median eta_3 | median peak_T_signed | median rmse_pos_T/e_T(0)* | median Gmax_full |
|---|---|---|---|---|---|---|---|---|
| all | S3 > 0.2 | 30 | 8 | 10 | 0.002612 | -0.09878 | 0.03555 | 0.003061 |
| all | S3 > 0.4 | 25 | 8 | 10 | 0.002923 | -0.1057 | 0.0369 | 0.003451 |
| all | S3 <= 0.2 | 50 | 2 | 0 | 0.001422 | -0.06385 | 0.02468 | 0.001422 |
| all | S3 < 0.05 | 43 | 2 | 0 | 0.001292 | -0.06342 | 0.02427 | 0.001292 |
| q50 | S3 > 0.2 | 16 | 5 | 7 | 0.003299 | -0.1044 | 0.03675 | 0.003299 |
| q50 | S3 > 0.4 | 13 | 5 | 7 | 0.004401 | -0.1072 | 0.04153 | 0.004401 |
| q50 | S3 <= 0.2 | 24 | 1 | 0 | 0.001599 | -0.06268 | 0.0252 | 0.001654 |
| q50 | S3 < 0.05 | 20 | 1 | 0 | 0.001553 | -0.05977 | 0.02437 | 0.001599 |
| q60 | S3 > 0.2 | 14 | 3 | 3 | 0.002271 | -0.08856 | 0.03422 | 0.002802 |
| q60 | S3 > 0.4 | 12 | 3 | 3 | 0.002271 | -0.09757 | 0.03411 | 0.003029 |
| q60 | S3 <= 0.2 | 26 | 1 | 0 | 0.001142 | -0.07105 | 0.02458 | 0.001142 |
| q60 | S3 < 0.05 | 23 | 1 | 0 | 0.001129 | -0.07169 | 0.02427 | 0.001129 |

| scope | quantity | n | Spearman correlation with S3 |
|---|---|---|---|
| all | eta_3 | 80 | 0.5451 |
| all | peak_T_signed | 80 | -0.5331 |
| all | rmse_pos_T_over_eT0 | 80 | 0.585 |
| all | Gmax_full_over_dw | 80 | 0.592 |
| all | fail (1/0) | 80 | 0.2938 |
| q50 | eta_3 | 40 | 0.5977 |
| q50 | peak_T_signed | 40 | -0.6976 |
| q50 | rmse_pos_T_over_eT0 | 40 | 0.6221 |
| q50 | Gmax_full_over_dw | 40 | 0.6144 |
| q50 | fail (1/0) | 40 | 0.4124 |
| q60 | eta_3 | 40 | 0.5391 |
| q60 | peak_T_signed | 40 | -0.3525 |
| q60 | rmse_pos_T_over_eT0 | 40 | 0.5436 |
| q60 | Gmax_full_over_dw | 40 | 0.6063 |
| q60 | fail (1/0) | 40 | 0.1552 |

Timing (reference update = the run's own regime onset, or the median onset of the regime runs of its q):

| scope | quantity (effort units) | regime runs, median | other runs, median | Mann-Whitney p |
|---|---|---|---|---|
| timing q50 | e0_change_ref_plus_10_exports | 3.708 | 2.663 | 0.1992 |
| timing q50 | e0_change_ref_to_final | 5.786 | 3.813 | 0.3696 |
| timing q50 | e0_drawdown_from_max_after_ref | 4.348 | 2.666 | 0.005525 |
| timing q60 | e0_change_ref_plus_10_exports | 3.1 | 2.473 | 0.681 |
| timing q60 | e0_change_ref_to_final | 1.983 | 4.024 | 0.1155 |
| timing q60 | e0_drawdown_from_max_after_ref | 4.088 | 2.121 | 0.01209 |

**The T = 2 analogue** (Sigma `d1_L_s2_out_lo` / Sigma `d1_L_s2_out_n`, phase A local 1201-1600 of the 40 v2.0 runs): maximum 0.1892, below 0.005 in 26 runs, as in the prompt.

| q | runs | max | median | below 0.005 | above 0.05 |
|---|---|---|---|---|---|
| 50 | 20 | 0.1892 | 0.002916 | 13 | 3 |
| 60 | 20 | 0.07376 | 0.000188 | 13 | 3 |
| both | 40 | 0.1892 | 0.000505 | 26 | 6 |

The one failed run of the 40 v2.0 confirmation runs, q = 50 seed 30510, has the T = 2 analogue 0.1892, the largest of the 40 (second largest 0.1297).

**What S3 marks.** A draw falls below the clamp 1e-6 only if alpha is below about 0.1, so S3 > 0.2 means that the Beta at the out-of-band states has collapsed against the lower end. In the regime runs the Beta *mean* itself reaches the actor's mean clamp (mu = clamp(sigmoid(z_0), 1e-6, 1 - 1e-6)), where d mu / d z_0 = 0:

| q | S3 class | runs | runs with a tail node at the mean clamp at some export | smallest Beta mean on the D_3 grid at u01600, median | share of out-of-band nodes at the mean clamp, median | share of out-of-band nodes with mean < 1e-4, median |
|---|---|---|---|---|---|---|
| 50 | regime | 16 | 12 | 1e-06 | 0.3377 | 0.6606 |
| 50 | mid | 4 | 0 | 0.000544 | 0 | 0 |
| 50 | low | 20 | 0 | 0.001566 | 0 | 0 |
| 60 | regime | 14 | 9 | 1e-06 | 0.2407 | 0.6258 |
| 60 | mid | 3 | 0 | 0.000522 | 0 | 0 |
| 60 | low | 23 | 0 | 0.001778 | 0 | 0 |

21 of the 30 regime runs have an out-of-band node at the mean clamp at some export, none of the 50 other runs; at u01600 the smallest Beta mean is 1e-6 in the median regime run and 5e-4 to 1.8e-3 in the other runs.

**Failures by class** (S3, onset of the regime = first 25-update window with an out-of-band share above 0.2, tie-point effort at the onset and at the end):

| run | S3 | regime onset (local update) | e_3(0) at the onset | e_3(0) at u1600 | stage-2 stretch |
|---|---|---|---|---|---|
| T3-R1 q50 s50504 | 0.7772 | 425 | 44.22 | 59.24 | yes |
| T3-R1 q50 s50517 | 0.001646 | - | - | 69.43 | no |
| T3-R1 q60 s50514 | 0.4356 | 625 | 51.33 | 51.98 | yes |
| T3-R1 q60 s50517 | 0.6975 | 750 | 52.43 | 53.1 | yes |
| T3-R2 q50 s60501 | 0.7758 | 475 | 57.95 | 60.11 | no |
| T3-R2 q50 s60506 | 0.7887 | 375 | 45.25 | 58.55 | yes |
| T3-R2 q50 s60513 | 0.7439 | 350 | 57.97 | 60.12 | no |
| T3-R2 q50 s60515 | 0.7965 | 375 | 44.16 | 64 | yes |
| T3-R2 q60 s60516 | 0.7736 | 350 | 48.84 | 50.22 | no |
| T3-R2 q60 s60518 | 0 | - | - | 8.155 | no |

**Timing.** The tie-point effort does not fall or stop rising after the onset. Over the next 250 updates it rises by a median of 3.7 (regime) against 2.7 (others) effort units at q = 50 and by 3.1 against 2.5 at q = 60 (Mann-Whitney p = 0.20 and 0.68). What differs is the drawdown of the tie-point effort from its maximum after the reference update to the last iterate: 4.3 against 2.7 at q = 50 (p = 0.0055) and 4.1 against 2.1 at q = 60 (p = 0.012). The last iterate of a regime run sits lower below its own recent maximum.

**From the replays** (§6): the out-of-band samples and the draws below the clamp do not have a projection share above their sample share in any group (§6 table).

## 8. DG7, the classification of the ten failures

Source: `dg7_failures.csv` (it joins `dg2_runs.csv`, `dg3_checks.csv`, `dg6_s3_runs.csv`, `dg8_phase_entry.csv`, `retro_checks_local1600.csv` and the stored last-16 tables `results/t3_r2/audit/a8_last16.csv`, `results/t3_r2/confirmation/analysis/a8_last16_confirmation.csv`).

| run | failing | eta_3 | rmse_pos | Gmax_full | S3 | F(u01600) | formation export | eta_3 > 0.005 at n of the last 16 exports | rank of the final eta_3 (1 = largest) | dev Delta, local 1600 | stage-2 stretch nodes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T3-R1 q50 s50504 | G-A eta_3 | 0.006227 | 0.04295 | 0.006227 | 0.7772 | 59.02 | 400 | 16 | 13 | 0.006227 | 16 |
| T3-R1 q50 s50517 | G-A eta_3 | 0.005676 | 0.04917 | 0.005676 | 0.001646 | 68.99 | 175 | 1 | 1 | 0.005558 | 0 |
| T3-R1 q60 s50514 | G-F | 0.002348 | 0.02766 | 0.05628 | 0.4356 | 51.74 | 225 | 2 | 10 | 0.002339 | 17 |
| T3-R1 q60 s50517 | G-F | 0.001703 | 0.02616 | 0.01089 | 0.6975 | 52.96 | 200 | 1 | 7 | 0.001568 | 28 |
| T3-R2 q50 s60501 | G-A eta_3 | 0.005241 | 0.04502 | 0.005241 | 0.7758 | 59.97 | 200 | 14 | 14 | 0.005241 | 0 |
| T3-R2 q50 s60506 | G-A eta_3 | 0.007244 | 0.04655 | 0.007244 | 0.7887 | 58.31 | 325 | 14 | 7 | 0.007238 | 6 |
| T3-R2 q50 s60513 | G-A eta_3 | 0.005466 | 0.04758 | 0.005466 | 0.7439 | 59.87 | 200 | 16 | 16 | 0.005466 | 0 |
| T3-R2 q50 s60515 | G-A eta_3;G-A rmse;G-F | 0.01138 | 0.06669 | 0.01138 | 0.7965 | 63.73 | 350 | 11 | 4 | 0.01138 | 5 |
| T3-R2 q60 s60516 | G-A rmse | 0.003765 | 0.05378 | 0.003765 | 0.7736 | 50.06 | 200 | 9 | 14 | 0.003765 | 0 |
| T3-R2 q60 s60518 | G-A eta_3;G-A rmse;G-A tail;G-F | 0.1447 | 0.462 | 0.1447 | 0 | -1.78e-15 | - | 16 | 14 | 0.1447 | 0 |

| run | class | evidence |
|---|---|---|
| T3-R1 q50 s50504 | **K** | S3 0.777; `eta_3` above 0.005 at all 16 of the last stage-3 exports; largest gain at d = -4 below the best response; also a 16-node stage-2 stretch (passing G-F) |
| T3-R1 q50 s50517 | **V** | S3 0.0016; formed at 175, F = 69; `eta_3` above 0.005 only at the last export, which is the largest of the 16 (rank 1) |
| T3-R1 q60 s50514 | **K** | S3 0.436; stage-3 passes (`eta_3` 0.0023); the 17-node stage-2 stretch formed in training; G-F 0.0563 |
| T3-R1 q60 s50517 | **K** | S3 0.698; stage-3 passes; the 28-node stage-2 stretch is present at the phase entry (26 of 28 nodes below 1 at entry); G-F 0.0109 |
| T3-R2 q50 s60501 | **K** | S3 0.776; `eta_3` above 0.005 at 14 of 16; deficit at d = -4 (60.05 against 67.5) |
| T3-R2 q50 s60506 | **K** | S3 0.789; 14 of 16; deficit at d = -6 (58.1 against 67.7); 6-node stage-2 stretch (passing) |
| T3-R2 q50 s60513 | **K** | S3 0.744; 16 of 16 (the final value is the smallest of the 16); deficit at d = -4 (59.8 against 67.6) |
| T3-R2 q50 s60515 | **K** | S3 0.797; 11 of 16; excess at d = -36 (53.4 against 30.3); 5-node stage-2 stretch (passing) |
| T3-R2 q60 s60516 | **K** | S3 0.774; `eta_3` 0.0038 passes, the amplitude is low (`rmse_pos` 0.0538, e(0) 50.2 against the best response 55.7); 9 of 16 above 0.005 |
| T3-R2 q60 s60518 | **F** | S3 = 0; never formed (constant 8.155); F = 0; sensitivity 7e-10; layer 2 saturated |

**The coworker's question, in detail.** (i) 60501, 60506, 60513 and 50504 are one kind: all four are in the regime (S3 0.74-0.79), the largest gain is at a small gap (|d| of 4 to 6) with the learned effort 7.4-9.6 units below the one-step best response, and the deficit is present at 14-16 of the last 16 exports (a persistent bias, not a last-iterate fluctuation). (ii) 60515 has the same regime and the same persistence (11 of 16) but its largest gain is a local excess at d = -36 (the learned effort above the best response, negative slope): the same class, a different shape. (iii) 60516's failure is the RMSE part of G-A only: the whole tent is about 14 % low (e(0) 50.2 against 58.3); the same regime. (iv) 60518 has none of this: S3 = 0, a policy constant in d from update 300. The prompt's DP3 for the classes of these runs holds as stated; the mechanism part of DP2 does not (§6, §7).

**(V).** Only 50517 at q = 50 meets all of D2's conditions (tent formed, not in the regime, failing value at the top of its own last-16 range). It is 1 of the 50 runs outside the regime; with one run nothing more can be said.

## 9. DG8, the stage-2 phase entry

Source: `dg8_phase_entry.csv`, `dg8_by_class.csv` (the stage-3 network `u01600` at the stage-2 input, before any stage-2 update; the stage-2 stretch from `u03200`, which equals the frozen stage 2; in-band nodes of the final-tier stage-2 grid with mean effort below 1).

| S3 class | runs | with a final stage-2 stretch | with >= 1 in-band node below 1 at the phase entry | median in-band nodes below 1 at entry | max | median entry effort over the band |
|---|---|---|---|---|---|---|
| regime | 30 | 10 | 30 | 26.5 | 46 | 6.735 |
| mid | 7 | 0 | 7 | 23 | 59 | 2.379 |
| low | 43 | 0 | 42 | 57 | 71 | 0.7334 |

The ten runs with a final stretch:

| run | S3 | final stretch nodes | of which below 1 at entry | median entry effort at the stretch nodes |
|---|---|---|---|---|
| T3-R1 q50 s50504 | 0.7772 | 16 | 11 | 0.4136 |
| T3-R1 q50 s50510 | 0.5218 | 20 | 20 | 0.01272 |
| T3-R1 q50 s50512 | 0.7436 | 15 | 15 | 0.1484 |
| T3-R1 q50 s50520 | 0.7527 | 11 | 11 | 0.0455 |
| T3-R1 q60 s50514 | 0.4356 | 17 | 0 | 41.94 |
| T3-R1 q60 s50517 | 0.6975 | 28 | 26 | 0.06955 |
| T3-R1 q60 s50519 | 0.7622 | 19 | 19 | 0.01606 |
| T3-R2 q50 s60506 | 0.7887 | 6 | 6 | 0.3123 |
| T3-R2 q50 s60507 | 0.7516 | 7 | 7 | 0.01788 |
| T3-R2 q50 s60515 | 0.7965 | 5 | 5 | 0.3067 |

An in-band node below 1 at the phase entry is **not** specific: 79 of the 80 runs have at least one (the median is 57 nodes in the 43 runs with S3 < 0.05). A stretch at the end appears only in the regime (10 of 30), including 50514 whose stretch formed later (0 of its 17 nodes below 1 at the entry).

## 10. The predictions (D7) against the evidence

| prediction | verdict | evidence |
|---|---|---|
| DP1a: 60518 is a formation failure unrelated to the clamp | **supported** | S3 = 0, no learner draw below or above the clamp at stage 3 in any of the 1600 updates (`d1_L_s3_{in,out}_{lo,hi}` sum to 0 in `results/t3_r3/replays/q60/seed60518/ms_updates.csv`); constant effort 8.155; never forms (§3) |
| DP1b: runs pass a development Delta of 0.12-0.15 near local update 100 and 60518 never left it | **supported** | Delta 0.1451 at local 100 (passing runs 0.1195-0.1449); 53 of 64 checks in 0.12-0.15 against 0-5 (§4) |
| DP1c: its tanh units are saturated, its gap-sensitivity about 0 | **partly supported** | the second layer is saturated (all 64 units from update 575), the first never; the sensitivity is about 0 from update 150 on, **before** any saturation; saturation follows and locks it (§3) |
| DP1d: its in-band surrogate gradient stays small compared with runs that formed | **not supported as stated** | the in-band region gradient is not small (norm 5.2-9.4 against 3.3-6.1 in the formed failures); it is exactly opposed by the out-of-band one (cosine -1.000), so the sum is small (§6) |
| DP2a: out-of-band samples and draws below the clamp have a projection share above their sample share | **not supported** | 0.7 % against 0.75 (out of band) and 0.3 % against 0.57 (below the clamp) in the regime runs (§6) |
| DP2b: the tie-point effort falls or stops rising after the onset | **not supported** | it rises by a median of 3.7 and 3.1 effort units in the next 250 updates; the last iterate sits further below the recent maximum in regime runs (§7) |
| DP3: one class (K) for 60501, 60506, 60513, 50504, 60515, 60516, 50514 and 50517 at q = 60; 50517 at q = 50 is (V) | **supported as a classification**; the mechanism is not shown | §8 |

## 11. The retrospective trigger table (for the proposal)

Source: `retro_triggers.csv`, `retro_checks_local1600.csv` (`ms_checks_stage3.csv` at local update 1600 of the 80 runs; the columns `Delta`, `R`, `R_tail`, `valid` only; extra cost = fired runs times one 1600-update stage-3 phase, over 80 x 3800 updates).

| trigger at local update 1600 | fires on | of which failed | failing runs not caught | extra updates | extra cost (share of the 80 x 3800 updates) |
|---|---|---|---|---|---|
| Delta>0.005 | 7 | 7 | 3 | 11200 | 0.03684 |
| R>0.4 | 1 | 1 | 9 | 1600 | 0.005263 |
| R>0.5 | 1 | 1 | 9 | 1600 | 0.005263 |
| R>0.7 | 1 | 1 | 9 | 1600 | 0.005263 |
| R>1 | 1 | 1 | 9 | 1600 | 0.005263 |
| R_tail>0.02 | 1 | 1 | 9 | 1600 | 0.005263 |
| R_tail>0.05 | 1 | 1 | 9 | 1600 | 0.005263 |
| R_tail>0.1 | 1 | 1 | 9 | 1600 | 0.005263 |
| degenerate | 0 | 0 | 10 | 0 | 0 |

| trigger | fired runs | failing runs not caught |
|---|---|---|
| Delta>0.005 | r1_q50_s50504;r1_q50_s50517;r2_q50_s60501;r2_q50_s60506;r2_q50_s60513;r2_q50_s60515;r2_q60_s60518 | r1_q60_s50514;r1_q60_s50517;r2_q60_s60516 |
| R>0.5 | r2_q60_s60518 | r1_q50_s50504;r1_q50_s50517;r1_q60_s50514;r1_q60_s50517;r2_q50_s60501;r2_q50_s60506;r2_q50_s60513;r2_q50_s60515;r2_q60_s60516 |
| R>1 | r2_q60_s60518 | r1_q50_s50504;r1_q50_s50517;r1_q60_s50514;r1_q60_s50517;r2_q50_s60501;r2_q50_s60506;r2_q50_s60513;r2_q50_s60515;r2_q60_s60516 |
| R_tail>0.05 | r2_q60_s60518 | r1_q50_s50504;r1_q50_s50517;r1_q60_s50514;r1_q60_s50517;r2_q50_s60501;r2_q50_s60506;r2_q50_s60513;r2_q50_s60515;r2_q60_s60516 |

Margins at the check (the 14 largest `Delta`):

| run | pass | Delta | R | R_tail | C |
|---|---|---|---|---|---|
| r2_q60_s60518 | no | 0.1447 | 1.18 | 0.1945 | 0.01477 |
| r2_q50_s60515 | no | 0.01138 | 0.3412 | 0.004153 | 0.04203 |
| r2_q50_s60506 | no | 0.007238 | 0.1956 | 0.003835 | 0.04369 |
| r1_q50_s50504 | no | 0.006227 | 0.1817 | 0.003606 | 0.04222 |
| r1_q50_s50517 | no | 0.005558 | 0.231 | 0.006474 | 0.03084 |
| r2_q50_s60513 | no | 0.005466 | 0.2202 | 0.004088 | 0.03904 |
| r2_q50_s60501 | no | 0.005241 | 0.193 | 0.002257 | 0.03918 |
| r2_q50_s60502 | yes | 0.004654 | 0.2127 | 0.005655 | 0.03153 |
| r1_q50_s50515 | yes | 0.004493 | 0.1851 | 0.003525 | 0.03742 |
| r2_q50_s60507 | yes | 0.004357 | 0.2146 | 0.002516 | 0.03869 |
| r2_q60_s60512 | yes | 0.004065 | 0.1474 | 0.003468 | 0.03862 |
| r2_q60_s60516 | no | 0.003765 | 0.1437 | 0.003072 | 0.03802 |
| r2_q50_s60517 | yes | 0.00337 | 0.1873 | 0.004236 | 0.04037 |
| r1_q50_s50506 | yes | 0.003147 | 0.1305 | 0.003026 | 0.03869 |

Largest values among the other 79 runs (without 60518): R 0.3412 (r2_q50_s60515), R_tail 0.0103 (r2_q60_s60510); 60518: R 1.1800, R_tail 0.1945.

`Delta` > 0.005 separates the 7 failing runs from the 73 others with a margin of 7 % on the passing side (0.004654) and 5 % on the failing side (0.005241). It is **the same quantity as the gate on the development tier**, so a trigger at this level gives a run near the limit a second draw; it raises the pass rate without changing how stage 3 is learned, and any paper must describe it that way. A formation-level trigger catches gross failures only: every threshold between the largest `R` of the other 79 runs (0.341) and 1.18, and between their largest `R_tail` (0.0103) and 0.19, fires on 60518 alone, and the position of the threshold inside those intervals is set by this single observation.

## 12. What is not covered, flags

- **Causation in class (K).** The regime is a marker (the Beta mean at the mean clamp, alpha of order 1e-4, entropy of order -1000, larger gradient norms) and a strong correlate of failures and of every stage-2 stretch. The experiment that would distinguish "the clamp regime causes the in-band error" from "both are products of a sharper tail/in-band trade-off" is an intervention at the same seeds that removes the over-push at the lower end (the censored likelihood of the clamped draws, §2 of `02_proposal.md`); it was not run in P1.
- **One formation failure in 80 runs** (1.25 %, exact 95 % interval [0.0003, 0.068]); the early-phase picture of 60518 is one run's. 60518 sits at or just outside the edge of the passing range on five quantities at update 100 or earlier (§4); this was not tested as a predictor, and the sensitivity at update 125 correlates with the formation time (§3).
- The replay set is 14 runs of T3-R2; T3-R1 runs have no replay.
- The sensitivity of DG2 is the derivative of the Beta mean with respect to the second input at the stage-3 input (1.0, d / 2B); it is an effort-scale derivative per normalised unit, not per gap unit.
- The pooled projection share can be negative or above 1; it is a first-order share of the mean surrogate gradient, not of its variance (the per-sample norms are given beside it). The optimiser is Adam with a global-norm clip at 0.5 and every update is clipped (pre-clip means 3-50), so the movement of a parameter is not proportional to the gradient share: a share of 0.3 % rules out that the clamped draws *dominate* the gradient, not an effect on parameters where the in-band gradient is small.
