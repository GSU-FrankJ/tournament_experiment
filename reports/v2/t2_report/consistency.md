# Consistency: pack values against the existing reports

Comparison rule: each pack value is rounded to the number of significant digits that the report shows for the same cell; a difference beyond that rounding is a mismatch. The old reports are not edited.

- cells compared: 22186; mismatches: 42

## Checks performed

| pack item | report | what | tables | cells compared | mismatches | unmatched report rows |
|---|---|---|---|---|---|---|
| T04 | protocols/v2_T2_locked.md | game table: T, B, e1*(0), e2*(0) | 1 | 8 | 0 | 0 |
| T04 | reports/v2/phase0_audit.md | e1*, e2*(0), D_2, grid sizes (sections 1.2, 1.9, 2) | 0 | 12 | 0 | 0 |
| T04 | reports/v2/phase1_verifier.md | B, grid sizes (section 2.2), recovery-region node counts (section 4.3) | 0 | 12 | 0 | 0 |
| T06 | protocols/v2_T2_locked.md | v1 protocol document: budgets and LR windows (section 2) | 0 | 8 | 0 | 0 |
| T06 | reports/v2/phase0_audit.md | legacy as-run values (section 2 table) | 0 | 34 | 0 | 0 |
| T07 | reports/v2/phase2_opening_checks.md | stage-1 drift exactly 0 (section 1c) | 0 | 2 | 0 | 0 |
| T08 | reports/v2/phase0_audit.md | tier grids (section 2 table) | 0 | 12 | 0 | 0 |
| T08 | reports/v2/phase2_opening_checks.md | outermost GL node (section 1c) | 0 | 1 | 0 | 0 |
| T09 | reports/v2/phase1_verifier.md | invariant residuals (section 4.1 tables and prose) | 2 | 44 | 1 | 0 |
| T10 | reports/v2/dreach_reach_mask_check.md | mask comparison table and dReach effect | 1 | 31 | 0 | 0 |
| T11 | reports/v2/phase1_verifier.md | benchmark consistency (section 4.5 table and prose) | 1 | 31 | 0 | 0 |
| T12 | reports/v2/phase1_verifier.md | calibration table and prose (section 4.2) | 1 | 106 | 1 | 0 |
| T12 | reports/v2/pilot4_stabilization.md | root-game references and the range of the fits (section 1a) | 1 | 12 | 0 | 0 |
| T12 | reports/v2/protocol_lock_and_rehearsal.md | lock-time calibration values | 1 | 72 | 0 | 0 |
| T12 | reports/v2/protocol_lock_and_rehearsal.md | zero policy vs PI references | 1 | 40 | 0 | 0 |
| T12 | reports/v2/protocol_lock_and_rehearsal.md | lock - Phase 1 differences as recorded (diff_vs_phase1_* columns; rendering check) | 1 | 32 | 0 | 0 |
| T12 | reports/v2/protocol_lock_and_rehearsal.md | lock - Phase 1 differences, exact (both CSVs parsed round-trip) | 1 | 32 | 14 | 0 |
| T12 | reports/v2/protocol_lock_and_rehearsal.md | argmax (t*, d*) vs Phase 1 | 1 | 32 | 0 | 0 |
| T12 | reports/v2/protocol_lock_and_rehearsal.md | calibration prose (section 3) | 0 | 12 | 1 | 0 |
| T12 | reports/v2/summary.md | zero-policy Gmax at the lock; lock vs Phase 1 bound | 0 | 3 | 0 | 0 |
| T13 | protocols/v2_T2_locked_v1_1.json (pipeline) vs run/run_v2_stagewise.py and run manifests | locked flags: protocol, code and manifests agree | 0 | 20 | 0 | 0 |
| T14 | reports/v2/pilot2_freeze.md | Pilot 2 snapshot integrity per arm (section 2.6) | 1 | 24 | 0 | 0 |
| T14 | reports/v2/pilot2_freeze.md | Pilot 2 median candidate drift on/off path per arm | 1 | 12 | 0 | 0 |
| T14 | reports/v2/pilot2_freeze.md | Pilot 2 snapshot drift and joint-arm drift (prose) | 0 | 5 | 0 | 0 |
| T14 | reports/v2/pilot3_continuation_mode.md | Pilot 3 snapshot drift and drift_test per arm (section 7) | 1 | 8 | 0 | 0 |
| T14 | reports/v2/pilot3_continuation_mode.md | Pilot 3 snapshot drift (prose) | 0 | 2 | 0 | 0 |
| T14 | reports/v2/pilot4_stabilization.md | Pilot 4 2b snapshot drift (prose) | 0 | 2 | 0 | 0 |
| T14 | reports/v2/protocol_lock_and_rehearsal.md | v1.0 rehearsal drift test (prose) | 0 | 1 | 0 | 0 |
| T14 | reports/v2/protocol_v1_1_confirmation.md | confirmation drift_test_pass column (reported-metrics table) | 0 | 1 | 0 | 0 |
| T14 | reports/v2/summary.md | Pilot 2 median candidate drift on/off path per arm | 1 | 12 | 0 | 0 |
| T15 | reports/v2 (C7 statements of each report) | C7 verdict and dirty flag per commit | 0 | 15 | 0 | 0 |
| T15 | reports/v2 (launch commit and dirty statements) | manifest commit and dirty flag of the launched runs vs report statements | 0 | 24 | 0 | 0 |
| T16 | reports/v2/pilot4_stabilization.md, reports/v2/protocol_lock_and_rehearsal.md, reports/v2/protocol_v1_1_confirmation.md and results/v2_T2_locked/rehearsal_v1_1_pytest.txt | known failing test named in each report; R6 pytest counts: log, checks JSON and report | 0 | 9 | 0 | 0 |
| T17 | reports/v2/pilot1_reward_estimator.md | Pilot 1 first divergence per pair (section 3.5) | 1 | 60 | 0 | 0 |
| T17 | reports/v2/pilot1_reward_estimator.md line 232 | Pilot 1 'never' cells (env, start, minibatch) | 0 | 60 | 0 | 0 |
| T17 | reports/v2/pilot1_reward_estimator.md line 256 | Pilot 1 divergence ranges (prose) | 0 | 12 | 0 | 0 |
| T17 | reports/v2/pilot2_freeze.md (table at line 494) | first-divergence summary cells | 0 | 66 | 0 | 0 |
| T17 | reports/v2/pilot2_freeze.md, reports/v2/pilot3_continuation_mode.md, reports/v2/pilot4_stabilization.md (anomaly prose) | desynchronization offsets after branching | 0 | 6 | 0 | 0 |
| T17 | reports/v2/pilot3_continuation_mode.md (table at line 293) | first-divergence summary cells | 0 | 22 | 0 | 0 |
| T17 | reports/v2/pilot4_stabilization.md (table at line 721) | first-divergence summary cells | 0 | 44 | 0 | 0 |
| T17 | reports/v2/summary.md line 213 | learn/opp desynchronization horizon over all pilots (prose) | 0 | 1 | 1 | 0 |
| T17 | results/v2_pilots/*/analysis/rng_divergence.csv | first divergence per stream re-computed from v2_updates.csv vs the existing CSVs | 0 | 840 | 0 | 0 |
| T18 | reports/v2/protocol_lock_and_rehearsal.md | Check 1 field counts | 1 | 13 | 0 | 0 |
| T18 | reports/v2/protocol_lock_and_rehearsal.md, reports/v2/protocol_v1_1_confirmation.md | ledger counts vs report text | 0 | 34 | 0 | 0 |
| T18 | reports/v2/protocol_v1_1_confirmation.md line 162 | R1-R6 verdicts vs rehearsal_v1_1_checks.json | 0 | 6 | 0 | 0 |
| T19 | reports/v2/pilot1_reward_estimator.md | prose: Pilot 1 launcher and Phase A wall per run (min / median / max), runs, workers | 0 | 8 | 0 | 0 |
| T19 | reports/v2/pilot1_reward_estimator.md | text: Pilot 1 header: launch commit and dirty flag of every run | 0 | 2 | 0 | 0 |
| T19 | reports/v2/pilot2_freeze.md | prose: Pilot 2 launcher wall range per run, runs, workers | 0 | 4 | 0 | 0 |
| T19 | reports/v2/pilot2_freeze.md | text: Pilot 2 header: launch commit and dirty flag of every run | 0 | 2 | 0 | 0 |
| T19 | reports/v2/pilot2_freeze.md | Pilot 2 Phase B wall per (q, arm) from v2_run_summary.json (section 2.7) | 1 | 18 | 0 | 0 |
| T19 | reports/v2/pilot3_continuation_mode.md | prose: Pilot 3 launcher wall range per run, runs, workers | 0 | 4 | 0 | 0 |
| T19 | reports/v2/pilot3_continuation_mode.md | text: Pilot 3 header: launch commit and dirty flag of every run | 0 | 2 | 0 | 0 |
| T19 | reports/v2/pilot3_continuation_mode.md | Pilot 3 Phase B wall median per (q, arm) from v2_run_summary.json (section 7) | 1 | 4 | 0 | 0 |
| T19 | reports/v2/summary.md | text: study table: launch commit and runs of Pilots 1-3 | 0 | 6 | 0 | 0 |
| T20 | reports/v2/pilot1_reward_estimator.md | prose: section 5: median signed peak error and median tail mean per arm and q | 0 | 8 | 0 | 0 |
| T20 | reports/v2/summary.md | Pilot 1 final medians (eta_2 development tier) | 1 | 20 | 0 | 0 |
| T21 | reports/v2/pilot1_reward_estimator.md | q=50 paired differences: mean, sd, median, min, max (section 3.2) | 1 | 110 | 0 | 0 |
| T21 | reports/v2/pilot1_reward_estimator.md | q=50 favour counts and CI95 (section 3.2) | 1 | 66 | 0 | 0 |
| T21 | reports/v2/pilot1_reward_estimator.md | q=60 paired differences: mean, sd, median, min, max (section 3.2) | 1 | 110 | 0 | 0 |
| T21 | reports/v2/pilot1_reward_estimator.md | q=60 favour counts and CI95 (section 3.2) | 1 | 66 | 0 | 0 |
| T21 | reports/v2/pilot1_reward_estimator.md | prose: section 5: sign counts of the paired differences | 0 | 7 | 0 | 0 |
| T21 | reports/v2/summary.md | Pilot 1 paired differences in summary.md | 1 | 32 | 0 | 0 |
| F05 | reports/v2/pilot1_reward_estimator.md | Spearman rho of peak error vs sigma_2(0)/q (section 3.4 table) | 1 | 18 | 0 | 0 |
| F05 | reports/v2/pilot1_reward_estimator.md | prose: section 5: Spearman rho within arms | 0 | 4 | 0 | 0 |
| T22 | reports/v2/pilot2_freeze.md | Pilot 2 medians (section 2.1) | 1 | 48 | 0 | 0 |
| T22 | reports/v2/pilot2_freeze.md | revised decomposition medians (section 6) | 1 | 42 | 0 | 0 |
| T22 | reports/v2/pilot2_freeze.md | revised medians vs superseded (section 6) | 1 | 18 | 0 | 0 |
| T22 | reports/v2/summary.md | summary.md Pilot 2 medians | 1 | 48 | 0 | 0 |
| T22 | reports/v2/summary.md | summary.md revised decomposition | 1 | 18 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | paired table q = 50, B1 − A | 1 | 155 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | CI95 and better/sign counts, 'q = 50, B1 − A' (manual comparison) | 1 | 62 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | paired table q = 50, B2 − A | 1 | 155 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | CI95 and better/sign counts, 'q = 50, B2 − A' (manual comparison) | 1 | 62 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | paired table q = 50, B2 − B1 | 1 | 155 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | CI95 and better/sign counts, 'q = 50, B2 − B1' (manual comparison) | 1 | 62 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | paired table q = 60, B1 − A | 1 | 155 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | CI95 and better/sign counts, 'q = 60, B1 − A' (manual comparison) | 1 | 62 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | paired table q = 60, B2 − A | 1 | 155 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | CI95 and better/sign counts, 'q = 60, B2 − A' (manual comparison) | 1 | 62 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | paired table q = 60, B2 − B1 | 1 | 155 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | CI95 and better/sign counts, 'q = 60, B2 − B1' (manual comparison) | 1 | 62 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | revised-term pairs (section 6) | 1 | 120 | 0 | 0 |
| T23 | reports/v2/pilot2_freeze.md | CI95 of the revised-term pairs (section 6) (manual comparison) | 1 | 24 | 0 | 0 |
| T23 | results/v2_pilots/pilot2/analysis/decomposition_residual_band_paired.csv | recomputed revised-decomposition pairs vs existing file (manual comparison) | 1 | 72 | 0 | 0 |
| T24 | reports/v2/pilot2_freeze.md | median live-network stage-2 drift of the frozen arms (section 2.6) | 1 | 4 | 0 | 0 |
| T25 | reports/v2/pilot2_freeze.md | location counts, final checkpoint u1000 (section 2.4) | 1 | 18 | 0 | 0 |
| T25 | reports/v2/pilot2_freeze.md | location counts, 21 training-time checkpoints u500-u1000 (section 2.4) | 1 | 18 | 0 | 0 |
| F08 | reports/v2/pilot2_freeze.md | SD-ratio summary (section 2.5) | 1 | 48 | 0 | 0 |
| T26 | reports/v2/pilot3_continuation_mode.md | reproducibility and snapshot-identity tables (section 2) (manual comparison) | 1 | 160 | 0 | 0 |
| T27 | reports/v2/pilot3_continuation_mode.md | Pilot 3 medians (section 3) | 1 | 32 | 0 | 0 |
| T27 | reports/v2/summary.md | summary.md Pilot 3 medians | 1 | 28 | 0 | 0 |
| T28 | reports/v2/pilot3_continuation_mode.md | paired table 4.1 q = 50 | 1 | 80 | 0 | 0 |
| T28 | reports/v2/pilot3_continuation_mode.md | CI95 and better/sign counts, '4.1 q = 50' (manual comparison) | 1 | 32 | 0 | 0 |
| T28 | reports/v2/pilot3_continuation_mode.md | paired table 4.2 q = 60 | 1 | 80 | 0 | 0 |
| T28 | reports/v2/pilot3_continuation_mode.md | CI95 and better/sign counts, '4.2 q = 60' (manual comparison) | 1 | 32 | 0 | 0 |
| T28 | reports/v2/summary.md | summary.md Pilot 3 paired differences | 1 | 40 | 0 | 0 |
| T29 | reports/v2/pilot3_continuation_mode.md | stability table (section 6) | 1 | 20 | 0 | 0 |
| T29 | reports/v2/summary.md | summary.md stability | 1 | 20 | 0 | 0 |
| T30 | reports/v2/protocol_v1_1_confirmation.md | 1.2 'Nothing else changed': records and pipeline v1.0 vs v1.1 | 0 | 2 | 0 | 0 |
| T31 | protocols/v2_T2_locked_v1_1.md | thresholds JSON vs protocol .md gate table and S1 text | 1 | 7 | 0 | 0 |
| T32 | reports/v2/protocol_lock_and_rehearsal.md | v1.0 rehearsal pass counts (4.3) | 1 | 12 | 0 | 0 |
| T32 | reports/v2/protocol_lock_and_rehearsal.md | 4.3 prose: failing stage-1 errors and their Gmax | 0 | 6 | 0 | 0 |
| T32 | reports/v2/protocol_v1_1_confirmation.md | 1.2 diff (24 entries, ops, paths) and 1.3 pass-probability prose | 0 | 35 | 0 | 0 |
| T32 | reports/v2/summary.md | v1.0 rehearsal G-A / G-F / run pass counts (summary table) | 1 | 6 | 0 | 0 |
| T33 | reports/v2/protocol_v1_1_confirmation.md | 1.5 timeline (UTC) | 1 | 11 | 1 | 0 |
| T34 | reports/v2/protocol_v1_1_confirmation.md | 3 checks table (verdicts and numbers) | 1 | 13 | 0 | 0 |
| T35 | reports/v2/protocol_v1_1_confirmation.md | 4.1 pass counts | 1 | 28 | 0 | 0 |
| T35 | reports/v2/protocol_v1_1_confirmation.md | text cells: rule, q_passes_rule | 1 | 4 | 0 | 0 |
| T35 | reports/v2/protocol_v1_1_confirmation.md | Verdict table (top) and 4.1 verdict row | 2 | 13 | 0 | 0 |
| T35 | reports/v2/summary.md | summary confirmation table: primary passes, CI, v1.0 outcome | 1 | 8 | 0 | 0 |
| T36 | reports/v2/protocol_v1_1_confirmation.md | 4.2 per-run table (numbers) | 1 | 480 | 0 | 0 |
| T36 | reports/v2/protocol_v1_1_confirmation.md | 4.2 per-run table (verdicts and labels) | 1 | 480 | 0 | 0 |
| T37 | reports/v2/protocol_v1_1_confirmation.md | 4.4 distributions | 1 | 160 | 0 | 0 |
| T37 | reports/v2/protocol_v1_1_confirmation.md | 4.4 prose: largest |dev - final| | 0 | 1 | 0 | 0 |
| T38 | reports/v2/protocol_v1_1_confirmation.md | 4.3 S1 table | 1 | 22 | 0 | 0 |
| T38 | reports/v2/protocol_v1_1_confirmation.md | 4.3 formatted S1 table and prose | 1 | 22 | 0 | 0 |
| T38 | reports/v2/summary.md | summary confirmation table: S1 passes and mean signed error [CI] | 1 | 8 | 0 | 0 |
| T39 | reports/v2/protocol_v1_1_confirmation.md | per-run reported metrics (report 4.5) | 1 | 1080 | 0 | 0 |
| F12 | reports/v2/protocol_v1_1_confirmation.md | per-run gate metrics (report 4.2) | 1 | 440 | 0 | 0 |
| F15 | reports/v2/protocol_v1_1_confirmation.md | OLS fit EXP_root/dW = a + b err^2 (report 4.6) | 1 | 8 | 0 | 0 |
| F15 | reports/v2/protocol_v1_1_confirmation.md | per-run stage-1 error and EXP_root (report 4.6) | 1 | 120 | 0 | 0 |
| T40 | reports/v2/protocol_v1_1_confirmation.md | 5 rehearsal vs confirmation | 1 | 280 | 0 | 0 |
| T41 | reports/v2/protocol_v1_1_confirmation.md | median/min/max of rmse, gmax_final, s1 (report 4.4) | 1 | 18 | 0 | 14 |
| T41 | reports/v2/protocol_v1_1_confirmation.md | S1 passes (report 4.3) | 1 | 2 | 0 | 0 |
| T41 | reports/v2/protocol_v1_1_confirmation.md | primary passes (report 4.1) | 1 | 2 | 0 | 0 |
| T42 | reports/v2/phaseA_ext.md | section 2.1: every 'median [IQR] (min-max)' cell, per q, metric, update | 2 | 680 | 0 | 0 |
| T42 | reports/v2/phaseA_ext.md | section 3 prose (plateau description) and section 2.3 stop-rule text | 0 | 42 | 2 | 0 |
| T42 | reports/v2/pilot4_stabilization.md | added rows: u1600 location-free peak-error medians vs section 1c.2 (ext, K=1) | 1 | 2 | 0 | 0 |
| T43 | reports/v2/pilot4_stabilization.md | section 2a paired decay - constant (K = 1): median, better / sign counts, CI95 | 1 | 66 | 0 | 0 |
| T43 | reports/v2/pilot4_stabilization.md | section 2a run-level paired differences (the 10 unmatched family-2b rows of that table belong to T51) | 1 | 32 | 0 | 10 |
| T43 | reports/v2/pilot4_stabilization.md | section 2a run-records summary | 1 | 44 | 0 | 0 |
| T43 | reports/v2/pilot4_stabilization.md | section 2a reproducibility statements (20/20 counts, verifier updates) | 0 | 8 | 0 | 0 |
| T44 | reports/v2/pilot4_stabilization.md | sections 1c.2 and 2a candidate medians per q, arm, K | 2 | 240 | 0 | 0 |
| T44 | reports/v2/pilot4_stabilization.md | section 1c.2 paired K - (K=1), extension: median, better, CI95 of 5 metrics | 1 | 120 | 3 | 0 |
| T45 | reports/v2/phaseA_ext.md | section 2.1 share / e_pred / e_learned cells (extension rows) | 2 | 56 | 0 | 0 |
| T45 | reports/v2/pilot1_reward_estimator.md | section 7 summary: median e_pred(0), e_learned(0), share median [min, max] | 1 | 20 | 0 | 0 |
| T45 | reports/v2/pilot1_reward_estimator.md | per-run inputs: Pilot 1 section 7 per-run table | 1 | 120 | 0 | 0 |
| T45 | reports/v2/protocol_lock_and_rehearsal.md | per-run inputs: lock report 4.4 per-run shares (v1.0 rehearsal) | 1 | 20 | 0 | 0 |
| T45 | reports/v2/protocol_lock_and_rehearsal.md | section 4.4 notes: median share per q, runs with share > 1 | 0 | 3 | 0 | 0 |
| T45 | reports/v2/protocol_v1_1_confirmation.md | per-run inputs: confirmation report 4.5 per-run shares | 1 | 40 | 0 | 0 |
| T45 | reports/v2/summary.md | summary.md Pilot 1 share medians | 1 | 4 | 0 | 0 |
| T46 | reports/v2/pilot4_stabilization.md | section 1d per-init fit table | 1 | 150 | 0 | 0 |
| T46 | reports/v2/pilot4_stabilization.md | section 1d median tables (fits, RL u1600) and the three-way summary cells | 3 | 42 | 0 | 0 |
| T46 | reports/v2/pilot4_stabilization.md | section 1d three-way table | 1 | 46 | 0 | 0 |
| T46 | reports/v2/pilot4_stabilization.md | section 1d prose (loss still falling, fitted tails, clamp, residual locations) | 0 | 11 | 6 | 0 |
| T47 | reports/v2/protocol_lock_and_rehearsal.md | section 5 thresholds_summary table | 1 | 64 | 0 | 0 |
| T47 | reports/v2/protocol_lock_and_rehearsal.md | section 5 'median [range]' table, RL actor-step text, identity count | 1 | 39 | 0 | 0 |
| T48 | reports/v2/pilot2_freeze.md | calibration gate band width | 1 | 2 | 0 | 0 |
| T48 | reports/v2/pilot2_freeze.md | revised decomposition medians per (q, arm) at u1000 | 1 | 42 | 0 | 0 |
| T48 | reports/v2/pilot2_freeze.md | all decomposition rows per arm (Pilot 2) | 1 | 12 | 0 | 0 |
| T48 | reports/v2/pilot2_freeze.md | revised vs superseded medians at u1000 | 1 | 36 | 0 | 0 |
| T48 | reports/v2/pilot2_freeze.md | per-run decomposition at u1000 (60 runs) | 1 | 180 | 0 | 0 |
| T48 | reports/v2/pilot2_freeze.md | band diagnostics and superseded-solver residual (prose) | 0 | 25 | 1 | 0 |
| T48 | reports/v2/pilot3_continuation_mode.md | Pilot 3 sweep coverage and non-contiguous parent bands (prose) | 0 | 4 | 0 | 0 |
| T48 | reports/v2/pilot4_stabilization.md | Pilot 4 2b parent bands (prose) | 0 | 11 | 0 | 0 |
| T48 | reports/v2/protocol_lock_and_rehearsal.md | v1.0 rehearsal bands (prose) | 0 | 2 | 0 | 0 |
| T48 | reports/v2/summary.md | open question 5 (prose) | 0 | 3 | 0 | 0 |
| T49 | reports/v2/pilot4_stabilization.md | own curvature per fit (section 1a) | 1 | 72 | 0 | 0 |
| T49 | reports/v2/pilot4_stabilization.md | BR slope per fit and h (section 1a) | 1 | 72 | 0 | 0 |
| T49 | reports/v2/pilot4_stabilization.md | section 1a comparison with the reference values (ranges) | 0 | 38 | 2 | 0 |
| T49 | reports/v2/summary.md | Pilot 4 bullet 1a (slope ranges) | 0 | 6 | 0 | 0 |
| T50 | reports/v2/pilot4_stabilization.md | section 1b ACF tables (median [q25, q75], 2 decimals) | 0 | 480 | 0 | 0 |
| T50 | reports/v2/pilot4_stabilization.md | window regression (section 1b) | 1 | 96 | 0 | 0 |
| T50 | reports/v2/pilot4_stabilization.md | section 1b comparison table and summary bullets | 0 | 18 | 1 | 0 |
| T50 | reports/v2/summary.md | Pilot 4 bullet 1b | 0 | 5 | 1 | 0 |
| T51 | reports/v2/pilot4_stabilization.md | 2b stability table | 1 | 20 | 0 | 0 |
| T51 | reports/v2/pilot4_stabilization.md | tail-averaged candidates, medians (sections 1c.1 and 2b) | 2 | 256 | 0 | 0 |
| T51 | reports/v2/pilot4_stabilization.md | 2b per-run table at u2200 (40 runs) | 1 | 720 | 0 | 0 |
| T51 | reports/v2/pilot4_stabilization.md | 2b run records table | 1 | 48 | 0 | 0 |
| T51 | reports/v2/pilot4_stabilization.md | run-record paired differences (families 2a and 2b) | 1 | 88 | 0 | 0 |
| T51 | reports/v2/pilot4_stabilization.md | location of Gmax_full (counts) | 1 | 64 | 0 | 0 |
| T51 | reports/v2/pilot4_stabilization.md | 2b and 1c.1 paired tables and prose (sections 1c.1, 2b, 7) | 0 | 206 | 3 | 0 |
| T51 | reports/v2/summary.md | Pilot 4 bullet 2b table | 0 | 14 | 0 | 0 |
| T52 | reports/v2/pilot4_stabilization.md | section 6, phase A table | 1 | 588 | 0 | 0 |
| T52 | reports/v2/pilot4_stabilization.md | section 6, phase B table | 1 | 1120 | 0 | 0 |
| T53 | reports/v2/protocol_lock_and_rehearsal.md | section 4.3 per-run gate values | 1 | 140 | 0 | 0 |
| T53 | reports/v2/protocol_lock_and_rehearsal.md | section 4.3 pass counts | 1 | 12 | 0 | 0 |
| T53 | reports/v2/protocol_lock_and_rehearsal.md | section 4.3 yes/no verdicts and outcome per run; prose on the 3 failures | 1 | 130 | 0 | 0 |
| T54 | reports/v2/protocol_lock_and_rehearsal.md | Check 1 field table | 1 | 13 | 0 | 0 |
| T54 | reports/v2/protocol_lock_and_rehearsal.md | Check 2 field table (yes/no) and section 4.1 prose | 0 | 39 | 0 | 0 |
| T55 | reports/v2/protocol_lock_and_rehearsal.md | section 1.2, 1.3 and 2.1 prose | 0 | 12 | 0 | 0 |
| T56 | reports/v2/protocol_v1_1_confirmation.md | section 2 table and section 1.4 test counts | 0 | 9 | 0 | 0 |
| T57 | reports/v2/pilot2_freeze.md | learn/opp desynchronisation ranges (prose) | 0 | 2 | 0 | 0 |
| T57 | reports/v2/pilot3_continuation_mode.md | learn/opp desynchronisation ranges (prose) | 0 | 2 | 0 | 0 |
| T57 | reports/v2/pilot4_stabilization.md | peak bias evidence (sections 1c, 1d) | 0 | 8 | 0 | 0 |
| T57 | reports/v2/pilot4_stabilization.md | learn/opp desynchronisation in 2a (prose) | 0 | 2 | 0 | 0 |
| T57 | reports/v2/protocol_lock_and_rehearsal.md | stage-1 failures, calibration floor, cusp steps (sections 3, 4.3, 5) | 0 | 9 | 0 | 0 |
| T57 | reports/v2/protocol_v1_1_confirmation.md | S1 summary and S1 failures (section 4.3) | 0 | 7 | 0 | 0 |
| D01 | reports/v2/pilot1_reward_estimator.md | per-run final-checkpoint table (section 3.1) | 1 | 680 | 0 | 0 |
| D01 | reports/v2/pilot1_reward_estimator.md | per-run smoothed-game table (section 7) | 1 | 240 | 0 | 0 |
| D01 | reports/v2/pilot1_reward_estimator.md | prose: sections 1 and 3.1: on-path max vs eta_2, stage1_status, stop rule, checkpoints | 0 | 6 | 0 | 0 |
| D01 | reports/v2/summary.md | median EXP_root/dW (stage1_untrained, development tier) in summary.md | 1 | 4 | 0 | 0 |
| D02 | reports/v2/pilot2_freeze.md | per-run table (section 2.1) | 1 | 1200 | 0 | 0 |
| D02 | reports/v2/pilot2_freeze.md | revised decomposition per run (section 6) | 1 | 180 | 0 | 0 |
| D02 | reports/v2/pilot2_freeze.md | bands and contains-0 flags per run (section 6) (manual comparison) | 1 | 240 | 0 | 0 |
| D03 | reports/v2/pilot3_continuation_mode.md | per-run table (section 3) | 1 | 680 | 0 | 0 |
| D03 | reports/v2/pilot3_continuation_mode.md | learning and inherited bands per run (section 3) (manual comparison) | 1 | 80 | 0 | 0 |
| D04 | reports/v2/phaseA_ext.md | section 2.3 run records (summary of the run-level columns) | 1 | 24 | 0 | 0 |
| D05 | reports/v2/pilot4_stabilization.md | section 2a per-run final metrics (u1600) | 1 | 520 | 0 | 0 |
| D06 | reports/v2/pilot4_stabilization.md | section 2b per-run table (u2200) | 1 | 720 | 0 | 0 |
| D07 | reports/v2/protocol_lock_and_rehearsal.md | 4.3 per-run gates | 1 | 140 | 0 | 0 |
| D07 | reports/v2/protocol_lock_and_rehearsal.md | 4.4 reported metrics, end of A | 1 | 180 | 0 | 0 |
| D07 | reports/v2/protocol_lock_and_rehearsal.md | 4.4 reported metrics, end of B | 1 | 320 | 0 | 0 |
| D07 | reports/v2/protocol_lock_and_rehearsal.md | 4.3 per-run verdicts | 1 | 120 | 0 | 0 |
| D08 | reports/v2/protocol_v1_1_confirmation.md | 3 per-run table (re-rehearsal) | 1 | 240 | 0 | 0 |
| D08 | reports/v2/protocol_v1_1_confirmation.md | 3 per-run verdicts (re-rehearsal) | 1 | 200 | 0 | 0 |
| D08 | reports/v2/protocol_v1_1_confirmation.md | 3: re-rehearsal gate values equal the v1.0 rehearsal (R1) | 0 | 2 | 0 | 0 |
| D09 | reports/v2/protocol_v1_1_confirmation.md | 4.2 per-run table | 1 | 480 | 0 | 0 |
| D09 | reports/v2/protocol_v1_1_confirmation.md | 4.5 reported metrics | 1 | 1080 | 0 | 0 |
| D09 | reports/v2/protocol_v1_1_confirmation.md | 4.5 reported metrics (flags) | 1 | 200 | 0 | 0 |
| K01 | reports/v2/protocol_v1_1_confirmation.md | verdict and S1 table rows of the v1.1 report (K01, K07) | 0 | 22 | 0 | 0 |
| K02 | reports/v2/protocol_v1_1_confirmation.md | medians and maxima of eta_final, rmse, tail, gmax_final (K02-K05) | 1 | 16 | 0 | 12 |
| K10 | reports/v2/summary.md | Pilot 1 final medians (summary.md) | 1 | 24 | 0 | 0 |
| K10 | reports/v2/summary.md | Pilot 1 paired differences (summary.md) | 1 | 32 | 0 | 0 |
| K11 | reports/v2/summary.md | Pilot 2 final medians (summary.md) | 1 | 48 | 0 | 0 |
| K12 | reports/v2/summary.md | Pilot 3 stability (summary.md) | 1 | 20 | 0 | 0 |
| K12 | reports/v2/summary.md | Pilot 3 paired differences (summary.md) | 1 | 40 | 0 | 0 |
| K13 | reports/v2/summary.md | extension medians u400-u1600 (summary.md) | 1 | 48 | 0 | 0 |
| K16 | reports/v2/pilot4_stabilization.md | supervised-fit medians (pilot4 report 1d) | 1 | 16 | 0 | 0 |
| K17 | reports/v2/protocol_lock_and_rehearsal.md | zero-effort calibration vs PI reference (lock report) | 1 | 24 | 0 | 0 |

## Mismatches

| pack item | quantity | pack value | report | report value | comment |
|---|---|---|---|---|---|
| T42 | range of the per-update median symmetry error u400-u1600 (both q) | 1.657 (q=60 u1525) to 4.612 (q=50 u525) | reports/v2/phaseA_ext.md (section 3) | the median stays between 2.4 and 3.5 effort units | holds at the four checkpoints u400/u800/u1200/u1600 (2.39-3.47), not along the 25-update curve |
| T42 | range of the per-update median signed peak error u800-u1600 (both q) | -0.0879 (q=60 u1375) to -0.03191 (q=60 u1425) | reports/v2/phaseA_ext.md (section 3) | fluctuates between about -0.04 and -0.09 |  |
| T44 | eta_T_over_dw boot_ci95_lo [q=50, K=16] | -0.0022455277830538685 | reports/v2/pilot4_stabilization.md (line 300) | −0.0023 | report digit is one off; it equals the pack value rounded first to 3 and then to 2 significant digits (double rounding in the report's rendering) |
| T44 | stage2_peak_rel_err_abs median [q=60, K=4] | 0.00754912206681305 | reports/v2/pilot4_stabilization.md (line 301) | +0.0076 | report digit is one off; it equals the pack value rounded first to 3 and then to 2 significant digits (double rounding in the report's rendering) |
| T44 | stage2_rmse_pos_over_g2_0 median [q=60, K=8] | -0.005748758232271964 | reports/v2/pilot4_stabilization.md (line 302) | −0.0058 | report digit is one off; it equals the pack value rounded first to 3 and then to 2 significant digits (double rounding in the report's rendering) |
| T46 | decrease of the running best loss over the last 20,000 steps (all 10 fits) | q=50: 0.154 to 0.293; q=60: 0.0546 to 0.0814 | reports/v2/pilot4_stabilization.md (section 1d) | The loss was still falling, by about 22% over the last 20,000 steps (init 0: ...) | about 22% is init 0's logged loss at 280k vs 300k; the logged loss oscillates (spikes up to 304 x the last value within the last 20,000 steps); the best logged loss fell in every fit (still falling), by the amounts given here; final loss above the best logged loss in 5 fits (up to 18.1 x) |
| T46 | smallest fitted tail mean over the 10 fits (lower end of 'fitted means are 0.001-0.22') | 9.999999484780954e-05 | reports/v2/pilot4_stabilization.md (section 1d, 'Can the head reach the target?') | 0.001 |  |
| T46 | fits whose smallest fitted tail mean equals the head floor 100*1e-6 = 1e-4 | 4 | reports/v2/pilot4_stabilization.md (section 1d) | the clamp is not binding | 4 fits (q=50 inits 0,1,2,3) reach the clamp floor 1e-4 at some tail node; the effect is at most 1e-4 effort units |
| T46 | largest residual at or near the domain edge (d = +-100 / +-120) | q=60 init 1: d=-40; q=60 init 3: d=16 | reports/v2/pilot4_stabilization.md (section 1d) | The largest residuals sit at or near the domain edges, d = ±100 (q=50) and ±120 (q=60) |  |
| T46 | max over the 10 fits of |peak error at d = 0| | 0.000961 (per-q medians 1.9e-05, 0.000597) | reports/v2/pilot4_stabilization.md (section 7 (1d)) | represent e2* to within 0.06% at the peak | per-q medians <= 0.0006: True; inits above 0.0006: 3 of 10 |
| T46 | max over the 10 fits of RMSE/e2*(0) | 0.004416 | reports/v2/pilot4_stabilization.md (section 7 (1d)) | <= 0.004 RMSE/e2*(0) | q=50 median 0.004034 (rounds to 0.004); inits above 0.004: 3 of 10 |
| T33 | time of 'Confirmation finished (40/40 rc=0)' | 2026-10-02T04:54:12 | reports/v2/protocol_v1_1_confirmation.md (line 114) | 2026-10-02T04:54:27 | pack time = latest status.json end_time (latest end_time of 40 status.json files: results/v2_T2_locked/confirmation/q50/seed20501/status.json, results/v2_T2_locked/confirmation/q50/seed20513/status.json); confirmation/launcher.out last modified 2026-10-02T04:54:13.693; the report time 04:54:27 occurs in 0 of 495 text files of results/v2_T2_locked/confirmation and confirmation_analysis |
| F12 | runs with Gmax_full dev tier > final tier (of 40) | 10 | reports/v2/protocol_v1_1_confirmation.md (4.4) | 'eta_2 and Gmax_full on the dev tier are lower than or equal to final in nearly all runs' | eta_2: 39/40 runs have dev <= final (the other 1 differ by 1.11e-16); Gmax_full: 30/40 runs have dev <= final, 10 have dev > final by up to 2.943e-05 dW (per_run.csv) |
| T17 | max over all pilot pairs of the first learn/opp divergence, updates after branching (largest: Pilot 4 2b q=60 learn) | 568.0 | reports/v2/summary.md line 213 | about 160 ('within tens to about 160 updates') | statement dates from commit 8b35066 (before Pilot 4) and is unchanged in the current summary; the Pilot 4 2b pairs first diverge up to this many updates after the u1600 branch point (pilots 1-3 and 2a: at most 164) |
| T09 | max inv_Gmax_le_dFull_excess_over_dw over Phase 1 legacy checkpoints | 0.0 | reports/v2/phase1_verifier.md (line 163) | -4.41e-08 | the report value equals the maximum over the final-tier rows only (80 rows); the maximum over all 240 rows is 0, attained in 5 rows (dev_2x, development tier) |
| T12 | dev_2x - dev dReach, zero q=60 | 3.8728732595805226e-05 | reports/v2/phase1_verifier.md (section 4.2) | 3.88e-5 | results/v2_pilots/phase1/calibration/calibration.csv dReach_over_dw: dev_2x 0.29509617752789286 - development 0.29505744879529705 |
| T12 | max_abs_diff_vs_phase1 q=50 zero final (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | 5.551e-17 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | diff_vs_phase1_Gmax_full_over_dw q=50 zero final (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | 5.551e-17 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | max_abs_diff_vs_phase1 q=50 zero development (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | 2.776e-17 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | diff_vs_phase1_EXP_root_over_dw q=50 zero development (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | 2.776e-17 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | max_abs_diff_vs_phase1 q=60 analytic_eq final (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | 5.294e-23 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | diff_vs_phase1_Gmax_full_over_dw q=60 analytic_eq final (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | -5.294e-23 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | diff_vs_phase1_EXP_root_over_dw q=60 analytic_eq final (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | -5.294e-23 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | diff_vs_phase1_dReach_over_dw q=60 analytic_eq final (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | -5.294e-23 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | max_abs_diff_vs_phase1 q=60 zero final (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | 8.327e-17 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | diff_vs_phase1_EXP_root_over_dw q=60 zero final (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | 8.327e-17 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | diff_vs_phase1_dReach_over_dw q=60 zero final (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | 5.551e-17 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | max_abs_diff_vs_phase1 q=60 zero development (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | 5.551e-17 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | diff_vs_phase1_EXP_root_over_dw q=60 zero development (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | 2.776e-17 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | diff_vs_phase1_dReach_over_dw q=60 zero development (exact) | 0.0 | reports/v2/protocol_lock_and_rehearsal.md (line 180) | 5.551e-17 | tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| T12 | max |lock - Phase 1| over metrics, exact | 0.0 | reports/v2/protocol_lock_and_rehearsal.md | 8.3e-17 | the report attributes the differences to floating-point summation order; tools/v2/locked_calibration.py read calibration.csv with pandas' default float parser, which returned 20 of 80 Phase 1 values one ulp away from their stored text (the phase1_* columns of calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are identical (0 nonzero exact differences out of 80) |
| F03 | updates of the training-time verifier checkpoints | u100, u200, u300, u400 in 40/40 runs (reasons: timeout, warmup_forced) | reports/v2/pilot1_reward_estimator.md (section 3.3, 'Deviation') | 'Those fall at run-specific updates, because stability-triggered calls move them, so they cannot be aligned across seeds.' | text claim: in Pilot 1 the 4 checkpoints are at the same updates in every run (only 4 points per run, so the 25-update exports remain the finer series) |
| T22 | kind of the Pilot 2 decomposition rows with e_hat_1 outside the parent sweep | 7 rows, all weight exports (updates [425, 450]; arms ['B1_frozen_allnorm', 'B2_frozen_s1norm']; q [60]) | reports/v2/pilot2_freeze.md (section 6, below 'Summary per (q, arm)') | The 7 out-of-sweep rows are early B1/B2 exports and checkpoints | count agrees; none of the rows is a training-time checkpoint row (those start at u500) |
| F08 | SD ratio at the first Phase B update u401 (median over the 10 runs per q, identical in all arms) | q50 1.1502157049301487; q60 1.073670537895672 | reports/v2/pilot2_freeze.md (section 4, 'Normalization scope') | the SD ratio is about √2 in every arm and at every update | prose claim; true from u402 on (min q50 1.3873, q60 1.3730); at u401 the ratio is 0.90-1.32 in every run |
| T48 | sources of the 7 out-of-sweep rows | weights: 7 | reports/v2/pilot2_freeze.md (section 6, after the all-rows table) | early B1/B2 exports and checkpoints | all 7 rows are 25-update weight exports at u425/u450 (column source == weights); no training-time checkpoint row lies outside the sweep |
| T49 | q60 dBR/de_opp, all fit widths W and steps h fine: range end | -0.29256775785060274 | reports/v2/pilot4_stabilization.md (section 1a comparison table (fine)) | −0.292 | pack value = the end of the [min, max] range over the fits closest to the report number (root_game CSVs) |
| T49 | q=60 numeric values vs reference: ranges that do not contain the reference | E[V2'']/(2k) final [0.2083, 0.2219] vs 0.236; own curvature 2a/(2k) final [-0.7917, -0.7781] vs -0.764; E[V2'']/(2k) fine [0.2307, 0.2337] vs 0.236; own curvature 2a/(2k) fine [-0.7693, -0.7663] vs -0.764 | reports/v2/pilot4_stabilization.md (section 1a, bullet q=60) | The numeric values bracket the reference on both tiers | only the BR slope ranges contain the reference -0.309; the curvature ranges (E[V2'']/(2k) and own curvature /2k) exclude 0.236 / -0.764 on both tiers (they are within 0.0053 of it on the fine tier) |
| T50 | median ACF at lags 100-160 (centred, u>=650), largest value | 0.014067609725697523 | reports/v2/pilot4_stabilization.md (section 1b summary) | negative at lags 100-160: medians -0.06 to -0.25 | the largest median at lags 100-160 is +0.0141 (q=50 mean, lag 100), not negative; the other medians lie in [-0.247, -0.062] |
| T50 | median ACF at lags 100-160 (centred, u>=650), largest value | 0.014067609725697523 | reports/v2/summary.md (Pilot 4 bullet 1b) | negative at 100-160 | one median at lags 100-160 is positive (see the pilot4 section 1b record) |
| T51 | q50 K=1 dReach_over_dw: CI hi | -4.4852575623020546e-06 | reports/v2/pilot4_stabilization.md (section 2b paired decay - constant (line 690)) | -0.000003 |  |
| T51 | q60 K=1 stage1_rel_err_abs: median | -0.02344697421398547 | reports/v2/pilot4_stabilization.md (section 2b paired decay - constant (line 690)) | −0.0235 |  |
| T51 | p3 q60 mean K=12 stage1_rel_err_abs: median | -0.03714825625876785 | reports/v2/pilot4_stabilization.md (section 1c.1 paired K - (K=1) (line 264)) | −0.0372 |  |
