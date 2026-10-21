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

{{table:inventory}}

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

{{table:formation}}

79 of 80 runs form, at update 150-700 (36 of 40 at q = 50 and 37 of 39 at q = 60 by update 250). 60518 never does.

{{table:final_shape}}

**The trajectory of 60518** against a run that broke out at updates 150-250 (T3-R2 q = 60 seed 60501, formation export 250) and the median of the passing q = 60 runs:

{{table:traj}}

All runs collapse to a low effort (6.9-12.8 at its minimum over updates 100-300) with a small in-band range and a sensitivity that reaches its minimum of 5.5e-5 to 6.3e-2 in updates 100-300 (the table below gives the deepest). 60501 breaks out between updates 150 and 250 (its sensitivity rises from 2.9e-4 at update 150 to 0.14 at 200 and 1.1 at 250, its tie-point effort from 7.9 to 13.8 and 38.2). 60518 does not: the range keeps falling (0.037 at 150, 0.003 at 300, 0 from update 1200), the sensitivity with it. The second-layer saturation of 60518 is 0 until update 300 and 0.06, 0.22, 0.80, 1.0 at updates 350, 400, 500, 575. So the saturation is not what prevented the break-out at updates 125-300 (it is zero then); it follows the loss of sensitivity and then locks the state (the first layer is never saturated, `sat1_in` = 0 at every export of the run).

**How deep the bottleneck is predicts when a run breaks out.**

{{table:bottleneck}}

The depth of the bottleneck (60518 has the smallest minimum of all 80 runs, 5.5e-5, not much below 6.5e-5 of T3-R1's 50504, which broke out at update 400) predicts the formation export: the sensitivity at update 150 has Spearman -0.92, and the values at updates 125 and 100, which precede every formation, -0.88 and -0.73. The late-formation count above is a post-hoc split at 325 and descriptive.

## 4. DG3, the training signal (stage-3 phase)

Sources: `dg3_checks.csv`, `dg3_windows.csv`, `dg3_history_windows.csv` (from `ms_updates.csv`, `ms_checks_stage3.csv`, `train_history.json`).

**60518 against the 18 passing q = 60 runs of T3-R2, development checks:**

{{table:dg3_checks}}

At local updates 25-100, 60518 is inside, or at the edge of, the passing range; by local 200 the passing runs have left the level of 0.12-0.15 and 60518 never does (53 of 64 checks in that interval, against 0-5). `R` and `R_tail` have the same picture.

**Training signals by window:**

{{table:dg3_signal}}

Up to update 100 the signals of 60518 are inside the passing range or at its edge (`kl_final_epoch`, `clip_frac`, `adv_raw_std`, the gradient norms and the starts per stratum are inside; the mean effort 24.78 is the smallest of the 19 values, and the entropy 2.735 and the concentration 108.7 lie just outside the passing ranges 2.747-2.785 and 106.2-108.6). After update 100 it has a smaller advantage SD, KL and critic loss than the passing runs (there is less to learn when nothing depends on the gap) and a concentration alpha + beta above every passing q = 60 run (331 against 165-273 in the last window), an entropy of 1.8 where the passing runs go down to -1999 (a tail Beta that has collapsed to the lower end, see §7).

**All 80 runs by clamp class, local updates 1201-1600:**

{{table:dg3_all}}

The regime runs have higher KL, clip fraction, pre-clip gradient norm (mean 8.0-8.4 against 3.0-3.1, maximum 45-51 against 15), a concentration 60-70 lower, an entropy of order -1000 and a minimum alpha of 2e-4 against 0.4.

## 5. DG4, the deviation-gain curves (14 runs, exports `u00100`, `u00400`, `u00800`, `u01600`)

Source: `dg4_profiles.csv`, `dg4_curves.csv` (T3-R2 calculator `tools/t3/t3r2_br_calc.py`, default resolution; the export's network at stage 3 against itself).

{{table:dg4_counts}}

**Agreement with the stored verifier values:**

{{table:dg4_agree}}

**The maximum of Delta_3 over D_3 at the four exports** (`max Delta (d*; learned effort, best response)`):

{{table:dg4_profile}}

All 14 runs have the same picture at update 100: a policy at 9-13 effort units against a best response of 48-55, Delta 0.12-0.20. The passing runs leave it by update 400; 60518 sits there at every export. The curves at selected gaps:

{{table:dg4_curves}}

For 60518 the learned effort 8.155 is on the rising side of the single maximum at every gap (slope +0.029 at d = 0, best response 41.9). For the q = 50 failures at u01600 the largest gain is at a small gap and below the best response (60501: slope +0.0042 at d = -4, effort 60.05, best response 67.5); for 60515 it is at d = -36 and above it (effort 53.4, best response 30.3, slope -0.0040: an excess); for 60516 the tent is lower overall (effort 50.2 at d = 0, best response 55.7).

## 6. DG5, the instrumented replays (14 runs)

Source: `results/t3_r3/replays/q*/seed*/` (`replay_updates.csv`, `replay_bins.npz`, `replay_identity.json`), `dg5_*.csv`.

{{table:dg5_identity}}

**The instrumentation is passive and the replays are replays**: all 14 reproduce the 64 exports, the stage-3 rows of `ms_updates.csv`, `ms_checks_stage3.csv` and the freeze arrays bit for bit (the passivity test on a reduced config is in `03_checks.md`). The number of draws at the stored clamp value equals the rollout's own `d1_L_s3_{in,out}_{lo,hi}` in every update of every replay.

**Gradient shares** (pooled projection share `sum <g_r, g> / sum |g|^2` over the window against the sample share; medians over the runs of each group):

{{table:dg5_shares}}

In the first 100 updates the out-of-band samples (sample share 0.75) carry 111 % of the projection (the in-band samples -11 %) in **all three groups**: the descent from effort 50 to about 8 is driven by the out-of-band starts. From update 400 on, the out-of-band samples carry 0.1-0.7 % and the draws below the clamp 0.3 % (regime) or 0 (non-regime); the draws below the clamp are 53-57 % of the samples of the regime runs. **A disproportionate share is not found in either group.**

**60518 compared with the five formed failures:**

{{table:dg5_60518}}

For 60518 the in-band and out-of-band region gradients are exactly opposed from update 101 on (cosine -1.000), each of norm 5-9, with a net norm of 1.2-2.0; the formed failures have a cosine of about 0.03 after update 400 and a small out-of-band norm. The network of 60518 is a function of one parameter (the common effort level), on which the two regions push in opposite directions; the directions that would introduce a dependence on d are those blocked by the lost sensitivity.

**Coverage** (updates 1201-1600):

{{table:dg5_coverage}}

The starts are uniform over the bins in all 14 runs (the bin share of starts lies within 5 % of 1/n_bins, in 60518 as in the others). The in-band share of the starts is 0.25 (q = 50) and 0.273 (q = 60) by design, and 98.9-100 % of the gradient is in band in all runs but 60518 (66 %, where the opposed halves nearly cancel). The per-sample gradient norm of the in-band draws is 105 / 102 (failures / passing regime) against 50 (non-regime) in the last window, and for the draws at the lower clamp 10 / 11 against 19 (`dg5_windows.csv`, window 1201-1600, medians).

## 7. DG6, the clamp regime across runs

Source: `dg6_s3_runs.csv`, `dg6_groups.csv`, `dg6_relations.csv`, `dg6_t2_analogue.csv`, `dg2_runs.csv`.

**Recomputation of the preliminary numbers: all reproduced.** S3 > 0.2 in 30 of the 80 runs (16 at q = 50, 14 at q = 60), S3 > 0.4 in 25; 8 of the 10 failures are among the 30 (8 of 30 against 2 of 50); medians by group as in the prompt (q = 50: `eta_3` 0.0033 against 0.0016, `peak_T_signed` -0.104 against -0.063; q = 60: 0.0023 / -0.089 against 0.0011 / -0.071); all ten stage-2 stretches are in the regime, the smallest S3 among them is 0.4356.

{{table:s3_counts}}

{{table:s3_groups}}

{{table:s3_rel}}

**The T = 2 analogue** (Sigma `d1_L_s2_out_lo` / Sigma `d1_L_s2_out_n`, phase A local 1201-1600 of the 40 v2.0 runs): maximum 0.1892, below 0.005 in 26 runs, as in the prompt.

{{table:t2_analogue}}

{{table:t2_failure}}

**What S3 marks.** A draw falls below the clamp 1e-6 only if alpha is below about 0.1, so S3 > 0.2 means that the Beta at the out-of-band states has collapsed against the lower end. In the regime runs the Beta *mean* itself reaches the actor's mean clamp (mu = clamp(sigmoid(z_0), 1e-6, 1 - 1e-6)), where d mu / d z_0 = 0:

{{table:mu_clamp}}

21 of the 30 regime runs have an out-of-band node at the mean clamp at some export, none of the 50 other runs; at u01600 the smallest Beta mean is 1e-6 in the median regime run and 5e-4 to 1.8e-3 in the other runs.

**Failures by class** (S3, onset of the regime = first 25-update window with an out-of-band share above 0.2, tie-point effort at the onset and at the end):

{{table:s3_failures}}

**Timing.** The tie-point effort does not fall or stop rising after the onset. Over the next 250 updates it rises by a median of 3.7 (regime) against 2.7 (others) effort units at q = 50 and by 3.1 against 2.5 at q = 60 (Mann-Whitney p = 0.20 and 0.68). What differs is the drawdown of the tie-point effort from its maximum after the reference update to the last iterate: 4.3 against 2.7 at q = 50 (p = 0.0055) and 4.1 against 2.1 at q = 60 (p = 0.012). The last iterate of a regime run sits lower below its own recent maximum.

**From the replays** (§6): the out-of-band samples and the draws below the clamp do not have a projection share above their sample share in any group (§6 table).

## 8. DG7, the classification of the ten failures

Source: `dg7_failures.csv` (it joins `dg2_runs.csv`, `dg3_checks.csv`, `dg6_s3_runs.csv`, `dg8_phase_entry.csv`, `retro_checks_local1600.csv` and the stored last-16 tables `results/t3_r2/audit/a8_last16.csv`, `results/t3_r2/confirmation/analysis/a8_last16_confirmation.csv`).

{{table:classes}}

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

{{table:entry}}

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

{{table:retro}}

{{table:retro_runs}}

Margins at the check (the 14 largest `Delta`):

{{table:retro_margins}}

`Delta` > 0.005 separates the 7 failing runs from the 73 others with a margin of 7 % on the passing side (0.004654) and 5 % on the failing side (0.005241). It is **the same quantity as the gate on the development tier**, so a trigger at this level gives a run near the limit a second draw; it raises the pass rate without changing how stage 3 is learned, and any paper must describe it that way. A formation-level trigger catches gross failures only: every threshold between the largest `R` of the other 79 runs (0.341) and 1.18, and between their largest `R_tail` (0.0103) and 0.19, fires on 60518 alone, and the position of the threshold inside those intervals is set by this single observation.

## 12. What is not covered, flags

- **Causation in class (K).** The regime is a marker (the Beta mean at the mean clamp, alpha of order 1e-4, entropy of order -1000, larger gradient norms) and a strong correlate of failures and of every stage-2 stretch. The experiment that would distinguish "the clamp regime causes the in-band error" from "both are products of a sharper tail/in-band trade-off" is an intervention at the same seeds that removes the over-push at the lower end (the censored likelihood of the clamped draws, §2 of `02_proposal.md`); it was not run in P1.
- **One formation failure in 80 runs** (1.25 %, exact 95 % interval [0.0003, 0.068]); the early-phase picture of 60518 is one run's. 60518 sits at or just outside the edge of the passing range on five quantities at update 100 or earlier (§4); this was not tested as a predictor, and the sensitivity at update 125 correlates with the formation time (§3).
- The replay set is 14 runs of T3-R2; T3-R1 runs have no replay.
- The sensitivity of DG2 is the derivative of the Beta mean with respect to the second input at the stage-3 input (1.0, d / 2B); it is an effort-scale derivative per normalised unit, not per gap unit.
- The pooled projection share can be negative or above 1; it is a first-order share of the mean surrogate gradient, not of its variance (the per-sample norms are given beside it). The optimiser is Adam with a global-norm clip at 0.5 and every update is clipped (pre-clip means 3-50), so the movement of a parameter is not proportional to the gradient share: a share of 0.3 % rules out that the clamped draws *dominate* the gradient, not an effect on parameters where the in-band gradient is small.
