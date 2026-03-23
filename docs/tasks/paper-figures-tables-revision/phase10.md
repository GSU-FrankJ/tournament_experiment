# Phase 10: Tables 1–2 — Environment & Training Configuration

## Objective

Fill in all ==?== placeholder values in Tables 1–2 by reading them from the codebase.

## Table 1 — Tournament Environment Configuration

Fill in:
- Winner and loser prizes (from `config.py` THEORY_PARAMS)
- Effort bounds (from runner or env code)
- Noise distribution description
- Default benchmark scenario

## Table 2 — TEL-PPO Training and Verification Configuration

Fill in from `agents/ppo_two_players_clean.py` and runner code:
- Samples per update (steps_per_update)
- Random seeds used
- Entropy coefficient (confirm if 0 or omitted)
- GAE lambda (confirm if 1)
- Network architecture (layers, units, activation, shared/separate)
- State input description
- Evaluation interval (N_eval)
- Training budget (max updates / env steps)

## Files to modify

- `paper/generator/tables.py` — table generation functions
- May need to read: `agents/ppo_two_players_clean.py`, `run/run_two_players.py`, `config/one_stage_two_players.py`

## Verification

- Generated LaTeX tables contain no ==?== placeholders
- All values match the code
