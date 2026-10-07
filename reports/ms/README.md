# Multistage line: index

The multistage line takes the settled T=2 baseline (protocol v2.0: conditional expected reward, expected continuation, backward freeze; `reports/t2_refine_100526/`) to a T-generic pipeline with a development stop rule, verifier-guided starts and targeted polishing.

## Rounds

| round | branch | status | start here |
|---|---|---|---|
| MS-R1 | `ms-r1` (from `origin/main` `4dc604de`) | P1 done and pushed (code, tests, C-R4 and C-MS1 passed, calibration, pre-registration); gate G1: waiting for the PI's "proceed P2" | `reports/ms/r1/02_preregistration.md` (design, parameters, arms, criterion), `reports/ms/r1/01_calibration.md` (what the v2.0 exports say about the rule), `reports/ms/r1/03_checks.md` (reproduction checks, tests, review) |

Files of the round (`reports/ms/r1/`): `pi_record/17_ms_r1_prompt.md` (the round's prompt, verbatim), `00_housekeeping.md`, `01_calibration.md`, `02_preregistration.md`, `prereg_parameters.json` (the pre-registered parameter file), `03_checks.md`, `figures/` (calibration figures), `report_scripts/` (the two small scripts behind the base-wave tables). The pilot reports (`04_pilot.md`, `05_decision_inputs.md`, `summary.md`) follow the pilot.

## Code of the line

| path | role |
|---|---|
| `run/run_ms_stagewise.py` | the runner (config `ms_run_config/1`; one stage per phase; rule, freeze, gates) |
| `run/ms_rollout.py`, `utils/ms_continuation.py` | stage-t rollout for t < T, nested expected-continuation tables |
| `utils/ms_residual.py`, `utils/ms_rule.py` | first-order / second-order residual metrics (D3), the development stop rule state machine (D4) |
| `envs/curriculum_env.py` | `StartSampler` additions: strata, `stratified_priority` (D5) |
| `tools/ms/` | `ms_configs.py`, `launch_ms_r1.py` (waves `base`, `v20_repro`, `pilot`), `cms1_compare.py`, `launch_checks.py`, `replay_dev_rule.py` (offline calibration), `r1_analysis.py` and `blind_criterion.py` (pilot analysis), `residual_asymmetry_example.py` |
| `tests/test_ms_*.py` | the tests of all of the above |

## Reference roots (untracked; explicit arguments to every tool)

- `parents_A`: `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine/parents_A/`
- `rehearsal_v2_0`, `confirmation_v2_0`: `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/`
- canonical worktree for the C7 reference: `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2`

## Tracking rules

Results of the line are under `results/ms_r1/` (`calibration/`, `v20_reproduction/`, `base/`, `pilot/`, `analysis/`). Tracked: per-run `manifest.json`, `run_config.json`, `status.json`, `gates.json`, `rule_log.json`, the check CSVs (`ms_checks_stage*.csv`), the bin maps (`ms_binmaps_stage*.npz`), the per-update CSV (`ms_updates.csv`), the continuation tables, the analysis tables, the calibration CSVs, the launch records and check JSONs. Not tracked: `.pt` files, `train_history.json`, weight exports, the freeze evaluation arrays (`freeze_stage*_*.npz`), `run.log`. Nothing under `protocols/`, `run/run_v2_T2_locked.py`, `utils/v2_continuation.py` or `tools/v2/confirmation_analysis*.py` changes in this line; no fresh seeds (40501-40520 are reserved).
