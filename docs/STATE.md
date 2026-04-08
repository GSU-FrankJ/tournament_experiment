# Project state

Last updated: 2026-04-08

## Current status
- **Parameter overhaul**: All configs updated to match `docs/experiment_config_040726.md`
- All old results deleted — fresh experiments needed with new parameters
- Two-stage runner deferred to separate task
- Paper generator updated with per-experiment theory params

## Parameter overhaul (2026-04-08)

Updated all experiment configs to new parameters that satisfy SOC and participation constraints.

### Config changes applied

| Experiment | k | (w_h, w_l) | q_list | effort_range |
|---|---|---|---|---|
| 2P Set 1 | 0.00055 | (6.5, 3.0) | [35, 45, 55] | [0, 100] |
| 2P Set 2 | via CLI: --k 0.0006 --w_h 8 --w_l 4 | (8, 4) | [35, 45, 55] | [0, 100] |
| 3P | 0.001 | (6.5, 3.0) | [35, 55] | [0, 100] |
| Diff Cost | k1=0.0004, k2=0.00055 | (8, 5.5) | [35, 55] | [0, 100] |
| Diff Ability | 0.0005 | (6.5, 3.0) | [35, 55] | [0, 100] |
| Two Stage | 0.0004 | (6.5, 3.0) | [25, 40, 55] | stage1=[0,100], stage2=[0,100] |

### Theoretical equilibria (verified)

| Experiment | q=35 | q=45 | q=55 |
|---|---|---|---|
| 2P Set 1 | 45.45 | 35.35 | 28.93 |
| 2P Set 2 | 47.62 | 37.04 | 30.30 |
| 3P | 25.00 | — | 15.91 |
| Diff Cost | e1=38.03, e2=27.66 | — | e1=26.54, e2=19.30 |
| Diff Ability | 46.43 | — | 30.37 |

### Files modified
- `config/one_stage_two_players.py` — k, q, q_list, effort_range
- `config/one_stage_three_players.py` — k, q, q_list, effort_range
- `config/one_stage_different_cost.py` — w_h, w_l, q, q_list, effort_range
- `config/one_stage_different_ability.py` — k, q, q_list, effort_range
- `config/two_stage_two_players.py` — effort_bounds_stage2
- `paper/generator/config.py` — per-experiment THEORY_PARAMS, Q_VALUES, updated e_star defaults
- `paper/generator/extract.py` — per-experiment theory param lookup
- `paper/generator/metrics.py` — per-experiment theory param lookup
- `paper/generator/tables.py` — per-experiment q_values and theory params
- `paper/generator/__init__.py` — export new symbols

### Results deleted
- All convergence JSONs, logs, and summary CSVs from previous parameter runs

## Previous critical finding: interior NE validity (2026-03-28)

Still relevant. The new parameters were chosen to satisfy the participation constraint:
- q_crit(2P, k=0.00055) = sqrt(2*3.5/(16*0.00055)) = 28.2 → q=35 passes
- q_crit(3P, k=0.001) = sqrt(3*3.5/(16*0.001)) = 25.6 → q=35 passes
- All experiments now use q >= 35, satisfying both SOC and participation constraint

## Task status

| Task | Status | Notes |
|------|--------|-------|
| parameter-overhaul | **complete** | All configs match experiment_config_040726.md |
| two-stage-runner | deferred | Config and env exist, runner needs to be built |
| paper-figures-tables-revision | stale | Needs re-run after new experiment data |
| runner-refactor | deferred | Post-project cleanup |

## Next steps
1. Run gradient baselines for all experiment types with new parameters
2. Run PPO experiments (5 seeds each) for all q values
3. Two-player Set 2 via CLI flags: `--k 0.0006 --w_h 8 --w_l 4`
4. Regenerate paper artifacts after results are collected
5. Build two-stage runner (separate task)
