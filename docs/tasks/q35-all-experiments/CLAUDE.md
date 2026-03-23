# Task: Run q=35 for three_players, different_cost, different_ability

## Goal
Run PPO (5 seeds) and gradient for q=35 across the three remaining experiment types, so that q=25 can be fully replaced by q=35 in all paper figures.

## Scope
- **In scope**: three_players, different_cost, different_ability — PPO + gradient, q=35 only
- **Out of scope**: two_players (already done), conc_max tuning, paper generator changes, figure regeneration

## Key Files
- `run/run_three_players.py` — three_players runner
- `run/run_different_cost.py` — different_cost runner
- `run/run_different_ability.py` — different_ability runner
- `config/one_stage_three_players.py` — three_players config (episodes=6,144,000)
- `config/one_stage_different_cost.py` — different_cost config (episodes=6,144,000)
- `config/one_stage_different_ability.py` — different_ability config (episodes=6,144,000)

## Constraints
- Training parameters same as q=25 (config defaults, no conc_max override)
- All long-running commands must use tmux
- Never modify existing convergence JSONs
- q=35 theory: e*(35) = (6.5 - 3.0) / (4 * 35 * 0.0004) = 62.5
