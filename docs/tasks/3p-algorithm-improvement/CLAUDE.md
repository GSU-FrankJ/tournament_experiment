# Task: Fix 3-Player PPO Convergence Gap

## Goal

Resolve the systematic ~5-unit gap between learned effort and Nash equilibrium in 3-player symmetric tournaments. The 2-player version converges well; the 3-player version consistently stalls at effort ~57 instead of 62.5 (for q=35).

## Scope

- **In scope**: Algorithm-level changes to `agents/ppo_three_players.py` and `envs/three_players_env.py`
- **In scope**: New training modes in `run/run_three_players.py`
- **Out of scope**: Paper generator, 2-player code, results from other experiments
- **Constraint**: The theoretical equilibrium formula `e* = (w_H - w_L) / (4qk)` is correct (gradient descent confirms gap=0.12)

## Key Files

| File | Role |
|------|------|
| `agents/ppo_three_players.py` | PPO agent: network, GAE, loss, update |
| `envs/three_players_env.py` | 3-player tournament environment |
| `run/run_three_players.py` | Training loop, rollout collection, convergence tracking |
| `config/one_stage_three_players.py` | Default hyperparameters |
| `utils/prob.py` | Win probability math (`p_from_diff`, `win_prob_three_players`) |
| `utils/theory.py` | Equilibrium formulas |
| `utils/exploit_asymmetric.py` | Exploitability evaluation (best-response search) |

## Validation Criteria

A successful fix must:
1. Achieve gap < 2.0 from e*(q=35)=62.5 across 5 seeds
2. Achieve exploitability < 0.05 (current threshold)
3. Not require knowledge of e* (no theory-align cheats)
4. Work for q=40 and q=55 as well
