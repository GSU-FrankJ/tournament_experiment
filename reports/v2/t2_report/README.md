# T=2 v2 report pack (P0 to P6)

Materials for a report on the T=2 v2 work, from the Phase 0 audit through the fresh-seed confirmation: every table, figure and headline number, with provenance. This directory contains no report prose; captions and notes describe what is shown and where it comes from.

## Provenance

- **Base commit of the pack:** `cb0b541b3ba16c04a0fa5d1a465e9d94c635ed89` (HEAD of branch `v2-stagewise-pilots` in the canonical worktree `.claude/worktrees/pilot-4-stabilization-fb99a2` when the pack was started).
- **Pack commit:** the commit that adds this directory (`git log --diff-filter=A -- reports/v2/t2_report/README.md`).
- **Where it was built:** the pack and its scripts were written in the session worktree `.claude/worktrees/t2-v2-report-pack-f51a78` (branch `claude/t2-v2-report-pack-f51a78`, fast-forwarded to the base commit), not in the canonical worktree, because a hook blocks writes into other worktrees; the canonical worktree's `results/` is the read-only data root. To bring the pack onto `v2-stagewise-pilots`: from the canonical worktree, `git merge --ff-only claude/t2-v2-report-pack-f51a78` (or cherry-pick the pack commits).
- **Builder:** `tools/v2/report/build_t2_report_pack.py` rebuilds the whole pack from `results/`; the module and its SHA-256 that built each item are in `manifest.csv` (`script`). `commit` in the manifest is the base commit above.
- **Data root:** `results/` of the canonical worktree. The arrays, weight exports and full states are untracked there (size, gitignored `.pt`); the builder reads them read-only (`T2_REPORT_RESULTS_ROOT` or `--results-root`). Every source path in the pack is repo-relative with the SHA-256 of the bytes that were read.
- **Rebuild:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/report/build_t2_report_pack.py --results-root <canonical worktree>/results`
- No training was run. Forward passes and verifier evaluations that the pack needed are listed in `reevaluations.csv` (policy source, tier, commit, wall time).

## Files

| file | content |
|---|---|
| `manifest.csv` | one row per item: id, section, title, type, priority, status, files, sources (path:sha256), script, commit, notes |
| `key_numbers.csv` | the headline numbers K01 to K19 with source file, selector and computation |
| `data_dictionary.csv` | every column of every pack table: definition, units, normalization, tier, source function |
| `tables/`, `figures/`, `data/` | the items; `provenance/` lists the files behind items with more than 12 sources |
| `gaps.md` | items not found and not generable, and UNKNOWN values, with reasons |
| `consistency.md` | comparisons with the existing reports and every mismatch |
| `reevaluations.csv` | every forward pass / verifier evaluation performed for the pack |

## Status

| status | meaning | items |
|---|---|---|
| found | an existing file used as is (copied, or referenced when large) | 8 |
| regenerated | rebuilt from existing data for a consistent style; matches the original | 12 |
| generated | new, from saved data or allowed computation | 75 |
| derived | computed from other pack items | 16 |
| missing | could not be found or generated (see gaps.md) | 0 |
| **total** | | **111** |

## Conventions

- **q** is reported separately (50 and 60). Per-run tables list their seeds explicitly. Development seeds: 10501 to 10510; fresh confirmation seeds: 20501 to 20520.
- **Evaluated policy:** the deterministic Beta mean. **Tier:** final unless stated otherwise; every development-tier value is labelled (`tier` column, `_dev` suffix, or the table header). The pilot learning curves are development-tier (re-evaluations of the 25-update weight exports); tier-independent metrics (recovery grid, direct policy queries) are marked as such in `data_dictionary.csv`.
- **Normalization:** deviation metrics (eta_2, Delta, Gmax_full, EXP_root, dReach, Delta_max_all, dFull) are divided by Delta W = 4; recovery errors are fractions of e_1*(0) (46.667 / 38.889) or e_2*(0) (70 / 58.333), signed unless stated; raw effort values are in effort units [0, 100] and labelled raw.
- **Final checkpoint of each study:** Pilot 1 u400; Pilots 2 and 3 global u1000; Phase A extension u400, u800, u1200, u1600; Pilot 4 section 2a u1600; section 2b u2200; locked runs end of A (u1600) and end of B (u2200).
- **Across-seed summaries:** median and IQR (25th to 75th percentile, numpy linear interpolation), plus min and max. **Paired statistics** reuse the existing bootstrap results (10,000 resamples, numpy seed 20261001; S1 uses seed 20261002).
- **Figures:** 7 in wide, no text below 8 pt, PDF (TrueType) plus 300 dpi PNG, each with `_data.csv` and `_caption.md`. Fixed colours (the same in every figure):

| kind | key | colour |
|---|---|---|
| arm | sampled | `#2a78d6` |
| arm | expected | `#eb6834` |
| arm | expected_ext | `#eb6834` |
| arm | A_joint | `#e87ba4` |
| arm | B1_frozen_allnorm | `#008300` |
| arm | B2_frozen_s1norm | `#4a3aa7` |
| arm | stochastic | `#4a3aa7` |
| arm | mean | `#1baf7a` |
| arm | B2_frozen_s1norm_mean | `#1baf7a` |
| arm | constant | `#eda100` |
| arm | B2_mean_constant | `#eda100` |
| arm | decay | `#e34948` |
| arm | B2_mean_decay | `#e34948` |
| q | q=50 | `#2a78d6` |
| q | q=60 | `#eb6834` |
| reference | closed form / target | `#0b0b0b` |
| reference | threshold line (dashed) | `#9c1c1c` |

  A figure colours either by arm or by q, never both; q is also encoded by line style (50 solid, 60 dashed) and marker (circle, square). Palette: the documented 8-slot categorical palette, assignment chosen for the largest worst-pair CVD separation within each group of arms that co-occur (see `tools/v2/report/style.py`).

## Outline

### Front matter

| id | item | priority | status | files |
|---|---|---|---|---|
| T01 | Study inventory | core | generated | [T01_study_inventory.csv](tables/T01_study_inventory.csv), [T01_study_inventory.md](tables/T01_study_inventory.md) |
| T02 | Decision log | core | generated | [T02_decision_log.csv](tables/T02_decision_log.csv), [T02_decision_log.md](tables/T02_decision_log.md) |
| T03 | Plan vs execution | core | generated | [T03_plan_vs_execution.csv](tables/T03_plan_vs_execution.csv), [T03_plan_vs_execution.md](tables/T03_plan_vs_execution.md) |

### Part I, section 2: Setting and baseline audit (P0)

| id | item | priority | status | files |
|---|---|---|---|---|
| T04 | Parameters and derived constants | core | generated | [T04_parameters_and_derived_constants.csv](tables/T04_parameters_and_derived_constants.csv), [T04_parameters_and_derived_constants.md](tables/T04_parameters_and_derived_constants.md) |
| F01 | Closed-form benchmark | core | generated | [F01_closed_form_benchmark.pdf](figures/F01_closed_form_benchmark.pdf), [F01_closed_form_benchmark.png](figures/F01_closed_form_benchmark.png), [F01_closed_form_benchmark_data.csv](figures/F01_closed_form_benchmark_data.csv), [F01_closed_form_benchmark_caption.md](figures/F01_closed_form_benchmark_caption.md) |
| T05 | Baseline audit findings | core | generated | [T05_baseline_audit_findings.csv](tables/T05_baseline_audit_findings.csv), [T05_baseline_audit_findings.md](tables/T05_baseline_audit_findings.md) |
| T06 | Resolved configuration of the locked v1.1 protocol | supp | generated | [T06_resolved_configuration_of_the_locked_v1_1_pr.csv](tables/T06_resolved_configuration_of_the_locked_v1_1_pr.csv), [T06_resolved_configuration_of_the_locked_v1_1_pr.md](tables/T06_resolved_configuration_of_the_locked_v1_1_pr.md) |

### Part I, section 3: Method change (a), the full-domain MPE metric

| id | item | priority | status | files |
|---|---|---|---|---|
| T07 | Metric definitions | core | generated | [T07_metric_definitions.csv](tables/T07_metric_definitions.csv), [T07_metric_definitions.md](tables/T07_metric_definitions.md) |
| T08 | Verifier tiers | supp | generated | [T08_verifier_tiers.csv](tables/T08_verifier_tiers.csv), [T08_verifier_tiers.md](tables/T08_verifier_tiers.md) |
| T09 | Invariant checks | supp | generated | [T09_invariant_checks.csv](tables/T09_invariant_checks.csv), [T09_invariant_checks.md](tables/T09_invariant_checks.md) |
| T10 | dReach reach-mask check | supp | found | [T10_dreach_reach_mask_check.csv](tables/T10_dreach_reach_mask_check.csv), [T10_dreach_reach_mask_check.md](tables/T10_dreach_reach_mask_check.md) |
| T11 | Benchmark consistency | supp | generated | [T11_benchmark_consistency.csv](tables/T11_benchmark_consistency.csv), [T11_benchmark_consistency.md](tables/T11_benchmark_consistency.md) |
| T12 | Calibration | core | generated | [T12_calibration.csv](tables/T12_calibration.csv), [T12_calibration.md](tables/T12_calibration.md) |
| F02 | Zero-effort deviation gains | supp | generated | [F02_zero_effort_deviation_gains.pdf](figures/F02_zero_effort_deviation_gains.pdf), [F02_zero_effort_deviation_gains.png](figures/F02_zero_effort_deviation_gains.png), [F02_zero_effort_deviation_gains_data.csv](figures/F02_zero_effort_deviation_gains_data.csv), [F02_zero_effort_deviation_gains_caption.md](figures/F02_zero_effort_deviation_gains_caption.md) |

### Part I, section 4: Method change (b), stagewise learning with frozen continuation

| id | item | priority | status | files |
|---|---|---|---|---|
| T13 | Flags | core | generated | [T13_flags.csv](tables/T13_flags.csv), [T13_flags.md](tables/T13_flags.md) |
| T14 | Snapshot drift | supp | generated | [T14_snapshot_drift.csv](tables/T14_snapshot_drift.csv), [T14_snapshot_drift.md](tables/T14_snapshot_drift.md) |
| T15 | C7 regression | supp | generated | [T15_c7_regression.csv](tables/T15_c7_regression.csv), [T15_c7_regression.md](tables/T15_c7_regression.md) |
| T16 | Test suite | supp | generated | [T16_test_suite.csv](tables/T16_test_suite.csv), [T16_test_suite.md](tables/T16_test_suite.md) |
| T17 | RNG streams | supp | generated | [T17_rng_streams.csv](tables/T17_rng_streams.csv), [T17_rng_streams.md](tables/T17_rng_streams.md) |
| T18 | Reproducibility ledger | core | generated | [T18_reproducibility_ledger.csv](tables/T18_reproducibility_ledger.csv), [T18_reproducibility_ledger.md](tables/T18_reproducibility_ledger.md) |

### Part I, section 5: Pilots

| id | item | priority | status | files |
|---|---|---|---|---|
| T19 | Pilot design summary (Pilots 1-3) | core | generated | [T19_pilot_design_summary_pilots_1_3.csv](tables/T19_pilot_design_summary_pilots_1_3.csv), [T19_pilot_design_summary_pilots_1_3.md](tables/T19_pilot_design_summary_pilots_1_3.md) |
| T20 | Pilot 1 final medians and IQR | core | generated | [T20_pilot_1_final_medians_and_iqr.csv](tables/T20_pilot_1_final_medians_and_iqr.csv), [T20_pilot_1_final_medians_and_iqr.md](tables/T20_pilot_1_final_medians_and_iqr.md) |
| T21 | Pilot 1 paired differences (expected - sampled) | core | generated | [T21_pilot_1_paired_differences_expected_sampled.csv](tables/T21_pilot_1_paired_differences_expected_sampled.csv), [T21_pilot_1_paired_differences_expected_sampled.md](tables/T21_pilot_1_paired_differences_expected_sampled.md) |
| F03 | Pilot 1 learning curves | core | regenerated | [F03_pilot_1_learning_curves.pdf](figures/F03_pilot_1_learning_curves.pdf), [F03_pilot_1_learning_curves.png](figures/F03_pilot_1_learning_curves.png), [F03_pilot_1_learning_curves_data.csv](figures/F03_pilot_1_learning_curves_data.csv), [F03_pilot_1_learning_curves_caption.md](figures/F03_pilot_1_learning_curves_caption.md) |
| F04 | Pilot 1 stage-2 mapping against the closed form at u400 | core | generated | [F04_pilot_1_stage_2_mapping_against_the_closed_f.pdf](figures/F04_pilot_1_stage_2_mapping_against_the_closed_f.pdf), [F04_pilot_1_stage_2_mapping_against_the_closed_f.png](figures/F04_pilot_1_stage_2_mapping_against_the_closed_f.png), [F04_pilot_1_stage_2_mapping_against_the_closed_f_data.csv](figures/F04_pilot_1_stage_2_mapping_against_the_closed_f_data.csv), [F04_pilot_1_stage_2_mapping_against_the_closed_f_caption.md](figures/F04_pilot_1_stage_2_mapping_against_the_closed_f_caption.md) |
| F05 | Pilot 1 peak error against sigma_2(0)/q | supp | regenerated | [F05_pilot_1_peak_error_against_sigma_2_0_q.pdf](figures/F05_pilot_1_peak_error_against_sigma_2_0_q.pdf), [F05_pilot_1_peak_error_against_sigma_2_0_q.png](figures/F05_pilot_1_peak_error_against_sigma_2_0_q.png), [F05_pilot_1_peak_error_against_sigma_2_0_q_data.csv](figures/F05_pilot_1_peak_error_against_sigma_2_0_q_data.csv), [F05_pilot_1_peak_error_against_sigma_2_0_q_caption.md](figures/F05_pilot_1_peak_error_against_sigma_2_0_q_caption.md) |
| T22 | Pilot 2 final medians | core | generated | [T22_pilot_2_final_medians.csv](tables/T22_pilot_2_final_medians.csv), [T22_pilot_2_final_medians.md](tables/T22_pilot_2_final_medians.md) |
| T23 | Pilot 2 paired comparisons | core | generated | [T23_pilot_2_paired_comparisons.csv](tables/T23_pilot_2_paired_comparisons.csv), [T23_pilot_2_paired_comparisons.md](tables/T23_pilot_2_paired_comparisons.md) |
| T24 | On-path definition and drift decomposition | supp | generated | [T24_on_path_definition_and_drift_decomposition.csv](tables/T24_on_path_definition_and_drift_decomposition.csv), [T24_on_path_definition_and_drift_decomposition.md](tables/T24_on_path_definition_and_drift_decomposition.md) |
| T25 | Location of Gmax_full | supp | generated | [T25_location_of_gmax_full.csv](tables/T25_location_of_gmax_full.csv), [T25_location_of_gmax_full.md](tables/T25_location_of_gmax_full.md) |
| F06 | Pilot 2 learning curves | core | regenerated | [F06_pilot_2_learning_curves.pdf](figures/F06_pilot_2_learning_curves.pdf), [F06_pilot_2_learning_curves.png](figures/F06_pilot_2_learning_curves.png), [F06_pilot_2_learning_curves_data.csv](figures/F06_pilot_2_learning_curves_data.csv), [F06_pilot_2_learning_curves_caption.md](figures/F06_pilot_2_learning_curves_caption.md) |
| F07 | Pilot 2 stage-2 mapping at the end of Phase B against the parent | core | generated | [F07_pilot_2_stage_2_mapping_at_the_end_of_phase.pdf](figures/F07_pilot_2_stage_2_mapping_at_the_end_of_phase.pdf), [F07_pilot_2_stage_2_mapping_at_the_end_of_phase.png](figures/F07_pilot_2_stage_2_mapping_at_the_end_of_phase.png), [F07_pilot_2_stage_2_mapping_at_the_end_of_phase_data.csv](figures/F07_pilot_2_stage_2_mapping_at_the_end_of_phase_data.csv), [F07_pilot_2_stage_2_mapping_at_the_end_of_phase_caption.md](figures/F07_pilot_2_stage_2_mapping_at_the_end_of_phase_caption.md) |
| F08 | Advantage SD ratio | supp | regenerated | [F08_advantage_sd_ratio.pdf](figures/F08_advantage_sd_ratio.pdf), [F08_advantage_sd_ratio.png](figures/F08_advantage_sd_ratio.png), [F08_advantage_sd_ratio_data.csv](figures/F08_advantage_sd_ratio_data.csv), [F08_advantage_sd_ratio_caption.md](figures/F08_advantage_sd_ratio_caption.md) |
| T26 | Pilot 3 reproducibility (stochastic arm vs Pilot 2 B2) | core | regenerated | [T26_pilot_3_reproducibility_stochastic_arm_vs_pi.csv](tables/T26_pilot_3_reproducibility_stochastic_arm_vs_pi.csv), [T26_pilot_3_reproducibility_stochastic_arm_vs_pi.md](tables/T26_pilot_3_reproducibility_stochastic_arm_vs_pi.md) |
| T27 | Pilot 3 final medians | core | generated | [T27_pilot_3_final_medians.csv](tables/T27_pilot_3_final_medians.csv), [T27_pilot_3_final_medians.md](tables/T27_pilot_3_final_medians.md) |
| T28 | Pilot 3 paired differences (mean - stochastic) | core | generated | [T28_pilot_3_paired_differences_mean_stochastic.csv](tables/T28_pilot_3_paired_differences_mean_stochastic.csv), [T28_pilot_3_paired_differences_mean_stochastic.md](tables/T28_pilot_3_paired_differences_mean_stochastic.md) |
| T29 | Pilot 3 stability | core | generated | [T29_pilot_3_stability.csv](tables/T29_pilot_3_stability.csv), [T29_pilot_3_stability.md](tables/T29_pilot_3_stability.md) |
| F09 | Pilot 3 learning curves | core | regenerated | [F09_pilot_3_learning_curves.pdf](figures/F09_pilot_3_learning_curves.pdf), [F09_pilot_3_learning_curves.png](figures/F09_pilot_3_learning_curves.png), [F09_pilot_3_learning_curves_data.csv](figures/F09_pilot_3_learning_curves_data.csv), [F09_pilot_3_learning_curves_caption.md](figures/F09_pilot_3_learning_curves_caption.md) |
| F10 | Stage-1 effort trajectories | supp | generated | [F10_stage_1_effort_trajectories.pdf](figures/F10_stage_1_effort_trajectories.pdf), [F10_stage_1_effort_trajectories.png](figures/F10_stage_1_effort_trajectories.png), [F10_stage_1_effort_trajectories_data.csv](figures/F10_stage_1_effort_trajectories_data.csv), [F10_stage_1_effort_trajectories_caption.md](figures/F10_stage_1_effort_trajectories_caption.md) |

### Part I, section 6: Formal T=2 experiment, locked protocol and fresh-seed confirmation

| id | item | priority | status | files |
|---|---|---|---|---|
| T30 | Locked pipeline (v1.1) | core | generated | [T30_locked_pipeline_v1_1.csv](tables/T30_locked_pipeline_v1_1.csv), [T30_locked_pipeline_v1_1.md](tables/T30_locked_pipeline_v1_1.md) |
| T31 | Criteria | core | generated | [T31_criteria.csv](tables/T31_criteria.csv), [T31_criteria.md](tables/T31_criteria.md) |
| F11 | Learning-rate schedule | supp | generated | [F11_learning_rate_schedule.pdf](figures/F11_learning_rate_schedule.pdf), [F11_learning_rate_schedule.png](figures/F11_learning_rate_schedule.png), [F11_learning_rate_schedule_data.csv](figures/F11_learning_rate_schedule_data.csv), [F11_learning_rate_schedule_caption.md](figures/F11_learning_rate_schedule_caption.md) |
| T32 | Protocol history | core | generated | [T32_protocol_history.csv](tables/T32_protocol_history.csv), [T32_protocol_history.md](tables/T32_protocol_history.md) |
| T33 | Timeline (UTC) | supp | generated | [T33_timeline_utc.csv](tables/T33_timeline_utc.csv), [T33_timeline_utc.md](tables/T33_timeline_utc.md) |
| T34 | Re-rehearsal checks R1-R6 | supp | generated | [T34_re_rehearsal_checks_r1_r6.csv](tables/T34_re_rehearsal_checks_r1_r6.csv), [T34_re_rehearsal_checks_r1_r6.md](tables/T34_re_rehearsal_checks_r1_r6.md) |
| T35 | Confirmation verdict | core | regenerated | [T35_confirmation_verdict.csv](tables/T35_confirmation_verdict.csv), [T35_confirmation_verdict.md](tables/T35_confirmation_verdict.md) |
| T36 | Per-run confirmation table | supp | found | [T36_per_run_confirmation_table.csv](tables/T36_per_run_confirmation_table.csv), [T36_per_run_confirmation_table.md](tables/T36_per_run_confirmation_table.md) |
| T37 | Distributions of the gate metrics | core | generated | [T37_distributions_of_the_gate_metrics.csv](tables/T37_distributions_of_the_gate_metrics.csv), [T37_distributions_of_the_gate_metrics.md](tables/T37_distributions_of_the_gate_metrics.md) |
| T38 | S1 summary | core | generated | [T38_s1_summary.csv](tables/T38_s1_summary.csv), [T38_s1_summary.md](tables/T38_s1_summary.md) |
| T39 | Reported metrics | core | generated | [T39_reported_metrics.csv](tables/T39_reported_metrics.csv), [T39_reported_metrics.md](tables/T39_reported_metrics.md) |
| F12 | Gate metrics against thresholds | core | generated | [F12_gate_metrics_against_thresholds.pdf](figures/F12_gate_metrics_against_thresholds.pdf), [F12_gate_metrics_against_thresholds.png](figures/F12_gate_metrics_against_thresholds.png), [F12_gate_metrics_against_thresholds_data.csv](figures/F12_gate_metrics_against_thresholds_data.csv), [F12_gate_metrics_against_thresholds_caption.md](figures/F12_gate_metrics_against_thresholds_caption.md) |
| F13 | Stage-2 mapping of all 20 runs per q | core | generated | [F13_stage_2_mapping_of_all_20_runs_per_q.pdf](figures/F13_stage_2_mapping_of_all_20_runs_per_q.pdf), [F13_stage_2_mapping_of_all_20_runs_per_q.png](figures/F13_stage_2_mapping_of_all_20_runs_per_q.png), [F13_stage_2_mapping_of_all_20_runs_per_q_data.csv](figures/F13_stage_2_mapping_of_all_20_runs_per_q_data.csv), [F13_stage_2_mapping_of_all_20_runs_per_q_caption.md](figures/F13_stage_2_mapping_of_all_20_runs_per_q_caption.md) |
| F14 | Stage-1 effort and its decomposition | core | generated | [F14_stage_1_effort_and_its_decomposition.pdf](figures/F14_stage_1_effort_and_its_decomposition.pdf), [F14_stage_1_effort_and_its_decomposition.png](figures/F14_stage_1_effort_and_its_decomposition.png), [F14_stage_1_effort_and_its_decomposition_data.csv](figures/F14_stage_1_effort_and_its_decomposition_data.csv), [F14_stage_1_effort_and_its_decomposition_caption.md](figures/F14_stage_1_effort_and_its_decomposition_caption.md) |
| F15 | EXP_root against the squared stage-1 error | core | generated | [F15_exp_root_against_the_squared_stage_1_error.pdf](figures/F15_exp_root_against_the_squared_stage_1_error.pdf), [F15_exp_root_against_the_squared_stage_1_error.png](figures/F15_exp_root_against_the_squared_stage_1_error.png), [F15_exp_root_against_the_squared_stage_1_error_data.csv](figures/F15_exp_root_against_the_squared_stage_1_error_data.csv), [F15_exp_root_against_the_squared_stage_1_error_caption.md](figures/F15_exp_root_against_the_squared_stage_1_error_caption.md) |
| F16 | Learning curves under the locked pipeline | core | generated | [F16_learning_curves_under_the_locked_pipeline.pdf](figures/F16_learning_curves_under_the_locked_pipeline.pdf), [F16_learning_curves_under_the_locked_pipeline.png](figures/F16_learning_curves_under_the_locked_pipeline.png), [F16_learning_curves_under_the_locked_pipeline_data.csv](figures/F16_learning_curves_under_the_locked_pipeline_data.csv), [F16_learning_curves_under_the_locked_pipeline_caption.md](figures/F16_learning_curves_under_the_locked_pipeline_caption.md) |
| F17 | G_t(d) on D_t | supp | generated | [F17_g_t_d_on_d_t.pdf](figures/F17_g_t_d_on_d_t.pdf), [F17_g_t_d_on_d_t.png](figures/F17_g_t_d_on_d_t.png), [F17_g_t_d_on_d_t_data.csv](figures/F17_g_t_d_on_d_t_data.csv), [F17_g_t_d_on_d_t_caption.md](figures/F17_g_t_d_on_d_t_caption.md) |
| T40 | Rehearsal vs confirmation | supp | found | [T40_rehearsal_vs_confirmation.csv](tables/T40_rehearsal_vs_confirmation.csv), [T40_rehearsal_vs_confirmation.md](tables/T40_rehearsal_vs_confirmation.md) |
| T41 | Targets vs results | core | generated | [T41_targets_vs_results.csv](tables/T41_targets_vs_results.csv), [T41_targets_vs_results.md](tables/T41_targets_vs_results.md) |

### Part II, section 7: Stage-2 accuracy beyond 400 updates

| id | item | priority | status | files |
|---|---|---|---|---|
| T42 | Phase A extension | core | generated | [T42_phase_a_extension.csv](tables/T42_phase_a_extension.csv), [T42_phase_a_extension.md](tables/T42_phase_a_extension.md) |
| F18 | Extension learning curves | core | regenerated | [F18_extension_learning_curves.pdf](figures/F18_extension_learning_curves.pdf), [F18_extension_learning_curves.png](figures/F18_extension_learning_curves.png), [F18_extension_learning_curves_data.csv](figures/F18_extension_learning_curves_data.csv), [F18_extension_learning_curves_caption.md](figures/F18_extension_learning_curves_caption.md) |
| T43 | Pilot 4 section 2a | core | generated | [T43_pilot_4_section_2a.csv](tables/T43_pilot_4_section_2a.csv), [T43_pilot_4_section_2a.md](tables/T43_pilot_4_section_2a.md) |
| T44 | Stage-2 tail averaging | supp | generated | [T44_stage_2_tail_averaging.csv](tables/T44_stage_2_tail_averaging.csv), [T44_stage_2_tail_averaging.md](tables/T44_stage_2_tail_averaging.md) |
| F19 | Pilot 4 section 2a learning curves | core | regenerated | [F19_pilot_4_section_2a_learning_curves.pdf](figures/F19_pilot_4_section_2a_learning_curves.pdf), [F19_pilot_4_section_2a_learning_curves.png](figures/F19_pilot_4_section_2a_learning_curves.png), [F19_pilot_4_section_2a_learning_curves_data.csv](figures/F19_pilot_4_section_2a_learning_curves_data.csv), [F19_pilot_4_section_2a_learning_curves_caption.md](figures/F19_pilot_4_section_2a_learning_curves_caption.md) |

### Part II, section 8: Anatomy of the stage-2 peak gap

| id | item | priority | status | files |
|---|---|---|---|---|
| T45 | Smoothed-game share | core | generated | [T45_smoothed_game_share.csv](tables/T45_smoothed_game_share.csv), [T45_smoothed_game_share.md](tables/T45_smoothed_game_share.md) |
| T46 | Representation floor | core | generated | [T46_representation_floor.csv](tables/T46_representation_floor.csv), [T46_representation_floor.md](tables/T46_representation_floor.md) |
| T47 | Cusp diagnostic | core | generated | [T47_cusp_diagnostic.csv](tables/T47_cusp_diagnostic.csv), [T47_cusp_diagnostic.md](tables/T47_cusp_diagnostic.md) |
| F20 | Three-way peak gap | core | generated | [F20_three_way_peak_gap.pdf](figures/F20_three_way_peak_gap.pdf), [F20_three_way_peak_gap.png](figures/F20_three_way_peak_gap.png), [F20_three_way_peak_gap_data.csv](figures/F20_three_way_peak_gap_data.csv), [F20_three_way_peak_gap_caption.md](figures/F20_three_way_peak_gap_caption.md) |
| F21 | Supervised-fit trajectories | core | generated | [F21_supervised_fit_trajectories.pdf](figures/F21_supervised_fit_trajectories.pdf), [F21_supervised_fit_trajectories.png](figures/F21_supervised_fit_trajectories.png), [F21_supervised_fit_trajectories_data.csv](figures/F21_supervised_fit_trajectories_data.csv), [F21_supervised_fit_trajectories_caption.md](figures/F21_supervised_fit_trajectories_caption.md) |

### Part II, section 9: Stage-1 accuracy

| id | item | priority | status | files |
|---|---|---|---|---|
| T48 | Induced-target method | core | generated | [T48_induced_target_method.csv](tables/T48_induced_target_method.csv), [T48_induced_target_method.md](tables/T48_induced_target_method.md) |
| F22 | Stage-1 residual against effort | core | generated | [F22_stage_1_residual_against_effort.pdf](figures/F22_stage_1_residual_against_effort.pdf), [F22_stage_1_residual_against_effort.png](figures/F22_stage_1_residual_against_effort.png), [F22_stage_1_residual_against_effort_data.csv](figures/F22_stage_1_residual_against_effort_data.csv), [F22_stage_1_residual_against_effort_caption.md](figures/F22_stage_1_residual_against_effort_caption.md) |
| T49 | Root stage game | core | generated | [T49_root_stage_game.csv](tables/T49_root_stage_game.csv), [T49_root_stage_game.md](tables/T49_root_stage_game.md) |
| T50 | Anatomy of the stage-1 fluctuation | core | generated | [T50_anatomy_of_the_stage_1_fluctuation.csv](tables/T50_anatomy_of_the_stage_1_fluctuation.csv), [T50_anatomy_of_the_stage_1_fluctuation.md](tables/T50_anatomy_of_the_stage_1_fluctuation.md) |
| F23 | ACF plot | supp | generated | [F23_acf_plot.pdf](figures/F23_acf_plot.pdf), [F23_acf_plot.png](figures/F23_acf_plot.png), [F23_acf_plot_data.csv](figures/F23_acf_plot_data.csv), [F23_acf_plot_caption.md](figures/F23_acf_plot_caption.md) |
| T51 | Pilot 4 section 2b | core | generated | [T51_pilot_4_section_2b.csv](tables/T51_pilot_4_section_2b.csv), [T51_pilot_4_section_2b.md](tables/T51_pilot_4_section_2b.md) |
| F24 | Pilot 4 section 2b learning curves | core | regenerated | [F24_pilot_4_section_2b_learning_curves.pdf](figures/F24_pilot_4_section_2b_learning_curves.pdf), [F24_pilot_4_section_2b_learning_curves.png](figures/F24_pilot_4_section_2b_learning_curves.png), [F24_pilot_4_section_2b_learning_curves_data.csv](figures/F24_pilot_4_section_2b_learning_curves_data.csv), [F24_pilot_4_section_2b_learning_curves_caption.md](figures/F24_pilot_4_section_2b_learning_curves_caption.md) |

### Part II, section 10: From development distributions to pre-registered gates

| id | item | priority | status | files |
|---|---|---|---|---|
| T52 | Pilot 4 section 6 distribution tables | supp | found | [T52_pilot_4_section_6_distribution_tables.csv](tables/T52_pilot_4_section_6_distribution_tables.csv), [T52_pilot_4_section_6_distribution_tables.md](tables/T52_pilot_4_section_6_distribution_tables.md) |
| T53 | v1.0 rehearsal | core | generated | [T53_v1_0_rehearsal.csv](tables/T53_v1_0_rehearsal.csv), [T53_v1_0_rehearsal.md](tables/T53_v1_0_rehearsal.md) |

### Part II, section 11: Protocol engineering and reproducibility

| id | item | priority | status | files |
|---|---|---|---|---|
| T54 | v1.0 Check 1 and Check 2 | supp | generated | [T54_v1_0_check_1_and_check_2.csv](tables/T54_v1_0_check_1_and_check_2.csv), [T54_v1_0_check_1_and_check_2.md](tables/T54_v1_0_check_1_and_check_2.md) |
| T55 | Consolidation | supp | generated | [T55_consolidation.csv](tables/T55_consolidation.csv), [T55_consolidation.md](tables/T55_consolidation.md) |
| T56 | v1.1 global-RNG hardening | core | generated | [T56_v1_1_global_rng_hardening.csv](tables/T56_v1_1_global_rng_hardening.csv), [T56_v1_1_global_rng_hardening.md](tables/T56_v1_1_global_rng_hardening.md) |

### Part II, section 12: Known issues and open questions (T=2 only)

| id | item | priority | status | files |
|---|---|---|---|---|
| T57 | Issue list | core | generated | [T57_issue_list.csv](tables/T57_issue_list.csv), [T57_issue_list.md](tables/T57_issue_list.md) |

### Appendices

| id | item | priority | status | files |
|---|---|---|---|---|
| D01 | Per-run table: Pilot 1 | supp | generated | [D01_per_run_table_pilot_1.csv](data/D01_per_run_table_pilot_1.csv) |
| D02 | Per-run table: Pilot 2 | supp | generated | [D02_per_run_table_pilot_2.csv](data/D02_per_run_table_pilot_2.csv) |
| D03 | Per-run table: Pilot 3 | supp | generated | [D03_per_run_table_pilot_3.csv](data/D03_per_run_table_pilot_3.csv) |
| D04 | Per-run table: Phase A extension | supp | generated | [D04_per_run_table_phase_a_extension.csv](data/D04_per_run_table_phase_a_extension.csv) |
| D05 | Per-run table: Pilot 4 section 2a | supp | generated | [D05_per_run_table_pilot_4_section_2a.csv](data/D05_per_run_table_pilot_4_section_2a.csv) |
| D06 | Per-run table: Pilot 4 section 2b | supp | generated | [D06_per_run_table_pilot_4_section_2b.csv](data/D06_per_run_table_pilot_4_section_2b.csv) |
| D07 | Per-run table: v1.0 rehearsal | supp | found | [D07_per_run_table_v1_0_rehearsal.csv](data/D07_per_run_table_v1_0_rehearsal.csv) |
| D08 | Per-run table: v1.1 re-rehearsal | supp | regenerated | [D08_per_run_table_v1_1_re_rehearsal.csv](data/D08_per_run_table_v1_1_re_rehearsal.csv) |
| D09 | Per-run table: confirmation | supp | regenerated | [D09_per_run_table_confirmation.csv](data/D09_per_run_table_confirmation.csv) |
| T58 | Compute | supp | generated | [T58_compute.csv](tables/T58_compute.csv), [T58_compute.md](tables/T58_compute.md) |
| T59 | Commands to reproduce | supp | generated | [T59_commands_to_reproduce.csv](tables/T59_commands_to_reproduce.csv), [T59_commands_to_reproduce.md](tables/T59_commands_to_reproduce.md) |

### Key numbers (key_numbers.csv)

| id | item | priority | status | files |
|---|---|---|---|---|
| K01 | Confirmation: primary passes per q, exact 95% CI and verdict | core | found | [key_numbers.csv](key_numbers.csv) |
| K02 | Confirmation: Gmax_full/dW, median and max per q | core | derived | [key_numbers.csv](key_numbers.csv) |
| K03 | Confirmation: eta_2, median and max per q | core | derived | [key_numbers.csv](key_numbers.csv) |
| K04 | Confirmation: RMSE/e2*(0), median and max per q | core | derived | [key_numbers.csv](key_numbers.csv) |
| K05 | Confirmation: stage-2 tail mean, median and max per q | core | derived | [key_numbers.csv](key_numbers.csv) |
| K06 | Confirmation: peak error at d=0, median and range per q | core | derived | [key_numbers.csv](key_numbers.csv) |
| K07 | Confirmation: stage-1 error summary per q | core | derived | [key_numbers.csv](key_numbers.csv) |
| K08 | Confirmation: max |dev - final| for eta_2 and Gmax_full | core | derived | [key_numbers.csv](key_numbers.csv) |
| K09 | Confirmation: the v1.0 outcome | core | derived | [key_numbers.csv](key_numbers.csv) |
| K10 | Pilot 1: medians of eta_2 and peak error per arm and q, sign counts | core | derived | [key_numbers.csv](key_numbers.csv) |
| K11 | Pilot 2: drift on and off path; peak error and tail mean per arm | core | derived | [key_numbers.csv](key_numbers.csv) |
| K12 | Pilot 3: paired differences of the stage-1 metrics, within-run SD | core | found | [key_numbers.csv](key_numbers.csv) |
| K13 | Phase A extension: peak error, tail mean and sigma_2(0) trends | core | derived | [key_numbers.csv](key_numbers.csv) |
| K14 | Pilot 4: decay - constant effects | core | found | [key_numbers.csv](key_numbers.csv) |
| K15 | Smoothed-game share: medians across studies | core | derived | [key_numbers.csv](key_numbers.csv) |
| K16 | Supervised-fit floor of the peak error | core | derived | [key_numbers.csv](key_numbers.csv) |
| K17 | Calibration floors, zero-effort Gmax and agreement with the PI references | core | derived | [key_numbers.csv](key_numbers.csv) |
| K18 | Bit-identity checks passed out of the total | core | derived | [key_numbers.csv](key_numbers.csv) |
| K19 | Total runs and total CPU-hours over P0-P6 | core | derived | [key_numbers.csv](key_numbers.csv) |

## Open points of the build

- Mismatches against the existing reports: 42 (`consistency.md`).
- Items with status `missing`: 0 (`gaps.md`).
- Columns without a dictionary entry: 0.
- Re-evaluation rows: 2 (`reevaluations.csv`).
