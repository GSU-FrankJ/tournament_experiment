# T = 3 line: index

The T = 3 line takes the settled T = 2 solver (protocol v2.0, `reports/t2_refine_100526/`) and the T-generic multistage pipeline (`reports/ms/README.md`) to the first T = 3 experiment. Decision at MS-R4's gate G2 (PI, 2026-10-16): option (i), v2.0 stays the T = 2 solver, MS-R4's two tries are reported as ablations and limitations (a separate task), the seed block 40501-40520 stays reserved, T = 3 starts with T3-R1.

## Rounds

| round | branch | status | start here |
|---|---|---|---|
| T3-R1 | `t3-r1` (from `origin/ms-r4` `db7577ea`) | P0 to P3 done: 160 runs (waves `t3` = 120, `t2j` = 40, one launch, all exit 0, 160 of 160 post-launch checks), evaluation, pre-registered analysis with an independent blind recomputation, the reports, two independent read-only fact-check rounds (the first 3,392 claims, the second 8,475), `handoff.md` for the coworker; code commit `5535d610`, launch `a848114e`. Outcome: H1, H2, H3 (T = 3) and H1-T2 (T = 2) are all supported at both q; `bw` passes all three gates in 18 of 20 runs at both q (the reading rule of D7 is met with no margin). **STOP**: the next round is for the PI and the coworker to decide (no lock, no confirmation, no tuning, no second wave) | `reports/t3/r1/summary.md` (outcome), `reports/t3/r1/04_t3_results.md` (T = 3: checks, arms, per-seed tables, `bw` with its four failures in section 4.1, the comparison with the H1 = H2 + H3 decomposition, where the gains are, best of run, coverage, cost, predictions, caveats), `reports/t3/r1/05_t2_comparison.md` (T = 2), `reports/t3/r1/06_decision_inputs.md` (decides nothing), `reports/t3/r1/handoff.md` (Chinese, for the coworker), `reports/t3/r1/pi_record/01_factcheck.md` (the fact-check ledger), `reports/t3/r1/02_preregistration.md`, `reports/t3/r1/03_checks.md`, `reports/t3/r1/00_housekeeping.md` (section 6: the P3 resume), `reports/t3/r1/pi_record/25_t3_r1_prompt.md` and `26_g1_reply.md` |

Files of the round (`reports/t3/r1/`): `pi_record/25_t3_r1_prompt.md`, `00_housekeeping.md`, `02_preregistration.md`, `03_checks.md`, `04_review_reports.md`, `prereg_criteria.json` (the pre-registered numeric parameters read by the analysis tools), `params_bw_t3.json` (the `bw` parameter file), `run_config_hashes.csv` (SHA-256 of every planned `run_config.json`, 168 rows), `arm_config_differences.csv`, and `report_scripts/` (`config_differences.py`, `prereg_tables.py`, `smoke_cost.py`, `seed_names.py`; P3: `bw_failures.py`, `v20_cpu.py`, `results_tables.py`, `assemble_reports.py` with the templates, `factcheck_ledger.py`, `build_handoff.py`). P3 adds `04_t3_results.md`, `05_t2_comparison.md`, `06_decision_inputs.md`, `summary.md`, `handoff.md`, `tables/` (the generated table fragments) and `pi_record/01_factcheck.md`. Results of the round: `results/t3_r1/` (`runs/`, `runs_t2/`, `eval/`, `analysis/`, `blind/`, `factcheck/`); the weights, freeze arrays, per-run arrays and `.pt` files are untracked and live only in the worktree `.claude/worktrees/t3-bw-vs-jt-51e472` (`bw_failures.py` reads them through `T3R1_UNTRACKED_ROOT`).

## Code of the line

| path | role |
|---|---|
| `run/run_joint.py` | the joint whole-game runner (config `joint_run_config/1`; arms `jt`, `jt_es`, `jt2`; T in {2, 3}) |
| `tools/t3/t3_configs.py` | the configs of the four arms (`bw` through `tools/ms/ms_configs.build_config(..., T=3, arm_table=...)`), the pinned production values of D5, seed guards, run directories |
| `tools/t3/t3_launch.py` | the launcher (waves `smoke` = 8, `t3` = 120, `t2j` = 40; reuses the job, record and host functions of `tools/ms/launch_ms_r1.py`) |
| `tools/t3/t3_metrics.py`, `tools/t3/t3_eval.py` | the D6 metrics (band split, terminal-stage recovery for any T, gates) and the evaluation of the three run formats (`bw`, joint, `v20`) with best of run |
| `tools/t3/t3_launch_checks.py` | C-INIT (plan), C-T2R and the post-launch checks of section 3.2 (budgets, schedules, starts, C-INIT, C-REPLAY) |
| `tools/t3/t3_analysis.py`, `tools/t3/t3_blind.py` | the pre-registered paired comparison and its independent recomputation |
| `tools/t3/t3_track.py`, `tools/t3/compare_runs.py` | lists / stages the files the tracking rules allow; bit-for-bit comparison of two runs of one config (the determinism reruns) |
| `tools/t3/seed_hits.py`, `preflight_seeds.sh`, `check_reference_roots.py` | the P0 scripts |
| `tests/t3_helpers.py`, `tests/test_t3_*.py` | the tests of all of the above |

## Reference roots (untracked; explicit arguments to every tool)

- `v20`, the v2.0 confirmation runs: `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0/q{50,60}/seed{30501..30520}/`.
- the September T = 3 start mix: `MultiStage/three_stage/T3_IMPLEMENTATION_PLAN_20260924.md` (read only); the original plan: `docs/Experiments Plan_Multi-stage.md` section 3.3-3.4 (read only).

## Tracking rules

Results of the line are under `results/t3_r1/` (`preflight/`, `smoke/`, `runs/`, `runs_t2/`, `eval/`, `analysis/`). Tracked: per-run `run_config.json`, `manifest.json`, `status.json`; for `bw` runs what `reports/ms/README.md` tracks for MS runs (`gates.json`, `rule_log.json`, `drift_test.json`, `ms_run_summary.json`, `ms_checks_stage*.csv`, `ms_binmaps_stage*.npz`, `ms_updates.csv`, `continuation_table_stage*.npz`); for joint runs `joint_run_summary.json`, `joint_checks.csv`, `joint_updates.csv`, `visitation.npz`; the evaluation tables, the analysis tables, the launch records and check JSONs. Never tracked: `.pt` files, `train_history.json`, weight exports, `freeze_stage*_*.npz`, `final_eval_*.npz`, the per-run NPZ arrays of `t3_eval.py` (`eval/arrays/`), `run.log`. `tools/t3/t3_track.py` lists or stages exactly these files. Nothing under `protocols/`, `agents/`, `envs/`, `utils/`, `tools/v2/`, `tools/ms/` or the existing files of `run/` changes in this line; seeds 40501-40520 and 10501-10530 appear in no config.
