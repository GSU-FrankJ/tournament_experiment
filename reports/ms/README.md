# Multistage line: index

The multistage line takes the settled T=2 baseline (protocol v2.0: conditional expected reward, expected continuation, backward freeze; `reports/t2_refine_100526/`) to a T-generic pipeline with a development stop rule, verifier-guided starts and targeted polishing.

## Rounds

| round | branch | status | start here |
|---|---|---|---|
| MS-R1 | `ms-r1` (from `origin/main` `4dc604de`) | done: P1 (code, tests, C-R4 and C-MS1, calibration, pre-registration), P2 (120-run pilot, C-MS2 20/20, analysis, reports); **STOP after the §4.3 push**, no criterion arm met, next round is the PI's decision | `reports/ms/r1/summary.md` (headline numbers, deviations, commands), `reports/ms/r1/05_decision_inputs.md` (arms side by side), `reports/ms/r1/04_pilot.md` (all tables), `reports/ms/r1/02_preregistration.md` (design; Addendum 1 = the G1 decision and the arm `MS_base2400`) |
| MS-R2 | `ms-r2` (from `origin/ms-r1` `71c58904`) | done: P1 (noise-landing code, tests, review, decomposition premise check, stop-candidate calibration on MS-R1, pre-registration), P2 (120-run pilot, all launch checks incl. C-NL 80/80, C-MS3 20/20, C-MS4 20/20, analysis, blind recomputation, D6 repeated), P3 reports; **STOP after the §4.3 push**: the primary criterion is met in no row (the smoothing part fell as predicted, the remainder rose), next round is the PI's decision | `reports/ms/r2/summary.md` (headline numbers, deviations, commands), `reports/ms/r2/05_decision_inputs.md` (arms side by side, what the noise landing changed, stop candidates, what is not settled), `reports/ms/r2/04_pilot.md` (all tables), `reports/ms/r2/02_preregistration.md` (design and decisions) |
| MS-R3 | `ms-r3` (from `origin/ms-r2` `e8eb9a08`) | done: P1 (actor variants `relu` and `t10` behind `BetaActor.variant`, tools, 313 new tests, independent review 0 MAJOR / 6 MINOR, the offline supervised screen with its premise check **PASS**, the RL-actor diagnostics, C-R6 20/20, pre-registration), P2 (240-run pilot, all launch checks incl. C-INIT 20/20, C-NL 120/120, C-MS5 80/80, analysis, blind recomputation), P3 reports; **STOP after the §4.3 push**: the primary criterion is met in no row; `relu` has the smaller typical tie deficit but five q = 50 runs (two (q, seed) cases) fail G-A, `t10` shows no detectable change of the mean deficit; next round is the PI's decision | `reports/ms/r3/summary.md` (headline numbers, deviations, commands), `reports/ms/r3/05_decision_inputs.md` (twelve arms side by side, what the actor changes, transmission, what is not settled), `reports/ms/r3/04_pilot.md` (all tables and figures), `reports/ms/r3/02_preregistration.md` |

Files of the round (`reports/ms/r1/`): `pi_record/17_ms_r1_prompt.md` (the round's prompt, verbatim), `00_housekeeping.md`, `01_calibration.md`, `02_preregistration.md`, `prereg_parameters.json` (the pre-registered parameter file), `03_checks.md`, `figures/` (calibration figures), `report_scripts/` (the two small scripts behind the base-wave tables). The pilot reports are `04_pilot.md`, `05_decision_inputs.md` and `summary.md`; `pi_record/18_g1_reply.md` is the PI's reply at gate G1 (verbatim) and `report_scripts/pilot_tables.py` generates the tables of the pilot reports from the analysis CSVs.

Files of the round (`reports/ms/r2/`): `pi_record/19_ms_r2_prompt.md` (the round's prompt, verbatim), `pi_record/01_factcheck.md` (the fact-check ledger of the reports), `00_housekeeping.md`, `01_decomposition.md` (the premise check on MS-R1), `01b_stop_candidates.md` (D6 on MS-R1), `02_preregistration.md`, `stop_candidate_grids.json` (the threshold grids, committed before the fire tables), `03_checks.md`, `04_pilot.md`, `05_decision_inputs.md`, `summary.md`, and `report_scripts/` (`decomposition_tables.py`, `stop_tables.py`, `pilot_tables.py`: every table of the reports is printed by one of them from the CSVs).

Files of the round (`reports/ms/r3/`): `pi_record/20_ms_r3_prompt.md` (the round's prompt, verbatim), `pi_record/sandbox_fit_tip.py` and `sandbox_fit_tip_results.jsonl` (the PI's sandbox, unchanged, not evidence), `pi_record/01_factcheck.md` (the fact-check ledger), `00_housekeeping.md`, `01_supervised_screen.md` (the offline screen, the premise check, the preamble numbers), `01b_rl_actor_diagnostics.md`, `02_preregistration.md`, `03_checks.md`, `04_pilot.md`, `05_decision_inputs.md`, `summary.md`, and `report_scripts/` (`pilot_tables.py`, `screen_tables.py`, `preamble_numbers.py`, `relu_units.py`, `tie_profile_runs.py`: every table of the reports is printed by one of them from the CSVs).

## Code of the line

| path | role |
|---|---|
| `run/run_ms_stagewise.py` | the runner (config `ms_run_config/1`; one stage per phase; rule, freeze, gates) |
| `run/ms_rollout.py`, `utils/ms_continuation.py` | stage-t rollout for t < T, nested expected-continuation tables |
| `utils/ms_residual.py`, `utils/ms_rule.py` | first-order / second-order residual metrics (D3), the development stop rule state machine (D4) |
| `envs/curriculum_env.py` | `StartSampler` additions: strata, `stratified_priority` (D5) |
| `tools/ms/` | `ms_configs.py` (arms incl. the budget-matched control `MS_base2400`), `launch_ms_r1.py` (waves `base`, `v20_repro`, `pilot` = five rule arms + `MS_base2400`), `cms1_compare.py`, `launch_checks.py` (post-launch checks and C-MS2), `replay_dev_rule.py` (offline calibration), `r1_analysis.py` and `blind_criterion.py` (pilot analysis), `residual_asymmetry_example.py` |
| `utils/ms_noise.py`, `tools/ms/r2_*.py` | MS-R2: the noise-floor decomposition and reporting columns (`utils/ms_noise.py`, used by the runner's optional `noise_report`, `conc_scale_schedule` and `fixed_global_sampler` keys), the launch checks C-NL / C-MS3 / C-MS4 (`r2_launch_checks.py`), the decomposition, stop-candidate, analysis and blind-recomputation tools (`r2_decomposition.py`, `r2_stop_candidates.py`, `r2_analysis.py`, `r2_blind_criterion.py`); wave `r2` of `launch_ms_r1.py` |
| `agents/ppo_curriculum.py`, `tools/ms/r3_*.py` | MS-R3: the actor variants (`BetaActor.variant`: `t1` default, `relu`, `t10`; runner keys `actor_variant`, `init_digest`; variant-aware numpy reload), the twelve-arm wave `r3`, the launch checks C-INIT / C-NL / C-MS5 (`r3_launch_checks.py`), the supervised screen, the RL-actor diagnostics, the analysis and the blind recomputation |
| `tests/test_ms_*.py` | the tests of all of the above |

## Reference roots (untracked; explicit arguments to every tool)

- `parents_A`: `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine/parents_A/`
- `rehearsal_v2_0`, `confirmation_v2_0`: `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/`
- canonical worktree for the C7 reference: `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2`

## Tracking rules

Results of the line are under `results/ms_r1/` (`calibration/`, `v20_reproduction/`, `base/`, `pilot/`, `analysis/`) and `results/ms_r2/` (`decomposition/`, `stop_calibration/`, `pilot/`, `analysis/`, `stop_calibration_pilot/`) and `results/ms_r3/` (`supervised_screen/`, `rl_actor_diagnostics/`, `v20_reproduction/`, `pilot/`, `analysis/`). Tracked: per-run `manifest.json`, `run_config.json`, `status.json`, `gates.json`, `rule_log.json`, the check CSVs (`ms_checks_stage*.csv`), the bin maps (`ms_binmaps_stage*.npz`), the per-update CSV (`ms_updates.csv`), the continuation tables, the analysis tables, the calibration CSVs, the launch records and check JSONs. Not tracked: `.pt` files, `train_history.json`, weight exports, the freeze evaluation arrays (`freeze_stage*_*.npz`), `run.log`. Nothing under `protocols/`, `run/run_v2_T2_locked.py`, `utils/v2_continuation.py` or `tools/v2/confirmation_analysis*.py` changes in this line; no fresh seeds (40501-40520 are reserved).
