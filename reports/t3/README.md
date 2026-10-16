# T = 3 line: index

The T = 3 line takes the settled T = 2 solver (protocol v2.0, `reports/t2_refine_100526/`) and the T-generic multistage pipeline (`reports/ms/README.md`) to the first T = 3 experiment. Decision at MS-R4's gate G2 (PI, 2026-10-16): option (i), v2.0 stays the T = 2 solver, MS-R4's two tries are reported as ablations and limitations (a separate task), the seed block 40501-40520 stays reserved, T = 3 starts with T3-R1.

## Rounds

| round | branch | status | start here |
|---|---|---|---|
| T3-R1 | `t3-r1` (from `origin/ms-r4` `db7577ea`) | P0 and P1 done (housekeeping, the joint runner `run/run_joint.py`, the `tools/t3` tools, 126 new tests, checks C-DIFF / C-INIT / C-T2R, the 8-run smoke wave with C-REPLAY and full-scale determinism reruns, independent review with its verification (its reports stored verbatim), pre-registration; final code commit `5535d610`); **STOP at gate G1**: P2 (waves `t3` = 120 runs and `t2j` = 40 runs) starts only after the PI's reply "proceed P2" | `reports/t3/r1/02_preregistration.md` (design as implemented, criteria, predictions, launch plan), `reports/t3/r1/03_checks.md` (checks, tests, smoke wave, review), `reports/t3/r1/00_housekeeping.md`, `reports/t3/r1/04_review_reports.md` (the reviewer's prompt and reports, verbatim), `reports/t3/r1/pi_record/25_t3_r1_prompt.md` (the prompt, verbatim) |

Files of the round (`reports/t3/r1/`): `pi_record/25_t3_r1_prompt.md`, `00_housekeeping.md`, `02_preregistration.md`, `03_checks.md`, `04_review_reports.md`, `prereg_criteria.json` (the pre-registered numeric parameters read by the analysis tools), `params_bw_t3.json` (the `bw` parameter file), `run_config_hashes.csv` (SHA-256 of every planned `run_config.json`, 168 rows), `arm_config_differences.csv`, and `report_scripts/` (`config_differences.py`, `prereg_tables.py`, `smoke_cost.py`, `seed_names.py`).

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
