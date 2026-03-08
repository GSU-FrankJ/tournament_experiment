# Session Summary — 2026-03-08

## Goals
1. Update all paper figures and tables with best-result runs
2. Archive old outputs
3. Investigate q=25 convergence problem

## What Was Done

### 1. Best-Only Paper Generator (`--best-only` flag)

Added a `--best-only` flag to the paper generator CLI that selects the single best seed (lowest final effort error) per (experiment, q) across all baseline variants (v1, v2, ablations).

**Files changed:**
- `paper/generator/extract.py` — added `get_final_effort_error_from_json()`
- `paper/generator/run_registry.py` — added `select_best_runs()`
- `paper/generator/__main__.py` — added `--best-only` flag to all subcommands

**Usage:** `python -m paper.generator --best-only make_all`

Note: `--best-only` must come before the subcommand due to argparse ordering.

### 2. Archived Old Outputs

Old figures, tables, and data moved to `paper/archive/2026-03-08/`.

### 3. Bug Found and Fixed: Wrong Theoretical Values

`select_best_runs()` originally used the two-player symmetric formula `e* = (w_H - w_L)/(4qk) = 87.5` for ALL experiments. For asymmetric experiments this is wrong:

| Experiment | Wrong e* | Correct e* (q=25) |
|---|---|---|
| two_players | 87.5 | 87.5 |
| three_players | 87.5 | 87.5 |
| different_cost | 87.5 | ~51.2 (per-agent: 59.2 / 43.1) |
| different_ability | 87.5 | 78.75 |

Fixed by reading theoretical values from the convergence JSON itself.

### 4. q=25 Convergence Investigation

**The "q=25 problem" was largely a measurement artifact.** With correct theoretical values:

| Experiment | Old Reported Err | Corrected Err | Rel% |
|---|---|---|---|
| two_players | 0.01 | 0.01 | 0.01% |
| different_cost | 33.89 | 0.14 | 0.27% |
| different_ability | 10.42 | 1.67 | 2.12% |
| three_players | 5.98 | 5.98 | 6.84% |

**three_players q=25 (~7% gap) is the only real remaining issue.**

Root cause analysis:
- Theoretical formula verified correct (FOC holds at e*=87.5)
- Gradient dp/de_i = 1/(2q) is constant and identical to two-player case
- PPO stalls at ~81.5 consistently across all 5 seeds
- Both 2p and 3p use same reward mechanism (stochastic env.step)
- All 3 players' transitions are stored per update
- Likely cause: higher reward variance from 3 noise draws (vs 2 in two-player), making PPO gradient signal noisier
- Potential fix: increase `steps_per_update` or reduce learning rate for three-player

### 5. Regenerated Paper Artifacts

7 figures (png+pdf) and 5 tables (csv+tex) regenerated with best-only runs.

Two figures skipped due to missing data in selected best runs:
- `beta_snapshots` — no alpha/beta time-series data
- `effort_drift` — no drift data

## Best Runs Selected (q=25)

| Experiment | Seed | Ablation | Error |
|---|---|---|---|
| two_players | 123 | pat_10 | 0.01 |
| three_players | 42 | baseline | 5.98 |
| different_cost | 456 | eps_001 | 0.14 |
| different_ability | 456 | baseline | 1.67 |

## Commits

1. `78084e0` — feat: add --best-only flag to paper generator and regenerate with best runs
2. `05a640b` — fix: use per-experiment theoretical values in select_best_runs
