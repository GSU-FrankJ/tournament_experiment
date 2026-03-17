# Runner Refactor

## Goal

Extract shared logic from the 4 runner scripts (`run/run_*.py`) into a common base, reducing ~60% code duplication.

## Scope

- **Touch:** `run/run_two_players.py`, `run/run_three_players.py`, `run/run_different_cost.py`, `run/run_different_ability.py`
- **Create:** `run/base_runner.py` (or similar shared module)
- **Do NOT touch:** `agents/`, `envs/`, `utils/`, `paper/`, `results/`

## Key files

- `run/run_two_players.py` — canonical/most complete runner (~2000 lines)
- `run/run_three_players.py` — 3-player variant
- `run/run_different_cost.py` — heterogeneous cost variant
- `run/run_different_ability.py` — heterogeneous ability variant

## Constraints

- All CLI flags must remain backward-compatible (same `--method`, `--q`, `--seed`, `--episodes`, ablation flags)
- Output file paths and JSON format must not change
- No changes to experiment logic or training behavior — pure structural refactor
