# AGENTS.md

> Last updated: 2026-02-06

Game-theoretic tournament experiments using PPO reinforcement learning to learn Nash equilibrium effort levels. Python 3.8+, PyTorch, NumPy, pandas, matplotlib.

## Commands (verified)

| Command | Purpose | ~Time |
|---------|---------|-------|
| `pip install -r requirements.txt` | Install dependencies | 30s |
| `python run/run_two_players.py --method ppo --q 25 --episodes 131072` | Train PPO on 2-player symmetric tournament | 2-10min |
| `python run/run_two_players.py --method gradient --q 40` | Gradient baseline for 2-player | <10s |
| `python run/run_three_players.py --method ppo --q 40 --episodes 2048000 --seed 42` | Train PPO on 3-player symmetric tournament | 10-30min |
| `python run/run_different_ability.py --method both --q 25 --episodes 131072` | Train 2-player different-ability | 5-15min |
| `python run/run_different_cost.py --method both --q 25 --episodes 131072` | Train 2-player asymmetric-cost | 5-15min |
| `python tools/plot_convergence.py` | Plot convergence curves from JSON history | <30s |
| `python tools/plot_convergence_detailed.py` | Detailed convergence plots with diagnostics | <30s |
| `python -m paper.generator make_all` | Generate all paper figures and tables | 1-2min |
| `python -m paper.generator --dry-run` | Preview paper artifact generation | <5s |

## File Map

```
run/                  -> Experiment entry points (CLI runners with argparse)
agents/               -> PPO agent implementations (Beta policy, GAE, clipping)
envs/                 -> Tournament game environments (expected utility, win probability)
config/               -> Experiment parameter configs (dicts with PPO/gradient settings)
utils/                -> Core utilities (theory, probability, evaluation, logging, plotting)
tools/                -> Analysis scripts (convergence plots, audits, sweeps)
paper/                -> Paper generation & outputs
  paper/generator/    -> Figure/table generation pipeline (python -m paper.generator)
  paper/figures/      -> Generated figures
  paper/tables/       -> Generated tables
results/              -> Experiment outputs organized by experiment type
  results/two_players/      -> Symmetric 2-player (convergence/, logs/, summary.csv)
  results/three_players/    -> 3-player (convergence/, logs/)
  results/different_cost/   -> Asymmetric cost (convergence/, logs/, summary.csv)
  results/different_ability/ -> Different ability (convergence/, logs/, summary.csv)
  results/ablation/         -> Ablation studies (exploit_params/, mechanism/)
docs/                 -> Documentation (guides/, technical/, archive/)
```

## Architecture

Single-step bandit setting: each episode is one tournament round. No temporal dependencies between episodes (gamma used only for GAE within a single step).

**Data flow:** `config/*.py` -> `run/run_*.py` -> `agents/ppo_*.py` + `envs/*_env.py` -> `results/`

**Key classes:**
- `agents/ppo_two_players_clean.py:ActorCritic` - Shared trunk with alpha/beta heads (Beta distribution policy) and value head
- `agents/ppo_two_players_clean.py:PPOTwoPlayersBandit` - PPO trainer with self-play, opponent lag, entropy/LR/clip schedules
- `envs/two_players_env.py:TwoPlayersEnv` - Closed-form expected utility (no stochastic rollouts)
- `utils/prob.py:p_from_diff` - Exact win probability under Uniform(-q,q) noise
- `utils/theory.py:e_star_two_players` - Theoretical Nash equilibrium: `e*(q) = (w_H - w_L) / (4qk)`

## Golden Samples

| For | Reference | Key patterns |
|-----|-----------|--------------|
| PPO agent | `agents/ppo_two_players_clean.py` | Beta policy, GAE, clipped objective, self-play |
| Environment | `envs/two_players_env.py` | Closed-form utilities via `utils.prob`, gym-like API |
| Runner | `run/run_two_players.py` | argparse CLI, schedule management, convergence eval |
| Config | `config/one_stage_two_players.py` | Dict-based config with PPO hyperparams |
| Convergence output | `results/*/convergence/` | JSON files with per-update effort/gap/entropy traces |

## Utilities (reuse, don't duplicate)

| Need | Use | Location |
|------|-----|----------|
| Win probability | `p_from_diff(d, q)` or `p_from_efforts(e1, e2, q)` | `utils/prob.py` |
| Theoretical equilibrium | `e_star_two_players(q, w_h, w_l, k)` | `utils/theory.py` |
| Asymmetric-cost equilibrium | `e_star_two_players_asymmetric_cost(q, w_h, w_l, k1, k2)` | `utils/theory.py` |
| Different-ability equilibrium | `e_star_two_players_different_ability(q, w_h, w_l, k, l1, l2)` | `utils/theory.py` |
| Save CSV results | `save_standardized_result(data, filename)` | `utils/logger.py` |
| Convergence quality | `convergence_quality_from_gap(gap)` | `utils/eval.py` |
| Plot effort curves | `plot_effort_curve(...)` | `utils/plot.py` |
| Rollout stats accumulator | `RolloutStatsAccumulator` | `utils/rollout_stats.py` |

## Heuristics

| When | Do |
|------|----|
| Adding a new tournament variant | Create env in `envs/`, config in `config/`, runner in `run/`, add theory formula to `utils/theory.py` |
| Changing PPO hyperparameters | Edit `config/*.py` dict or pass CLI overrides to `run/run_*.py` |
| Evaluating convergence quality | Use `utils/eval.py:convergence_quality_from_gap` (Excellent <0.5, Good <1.0, Fair <5.0) |
| Effort is normalized | Agent outputs `a in (0,1)` via Beta distribution; mapped to effort `e = low + a*(high-low)` |
| Checking equilibrium correctness | Compare learned mean effort to `utils/theory.e_star_*` closed-form |
| Running full parameter grid | Use `--grid` flag on runners (e.g., `run_different_ability.py --method both --grid`) |
| Generating paper figures | `python -m paper.generator make_all` reads from `results/` |

## Boundaries

**Always:**
- Use closed-form expected utilities in environments (no stochastic noise during rollouts)
- Evaluate policy via Beta distribution mean (not mode, even when alpha,beta > 1)
- Use denominator 4 for two-player single-stage equilibrium formula
- Store both players' transitions in PPO rollouts
- Use `save_standardized_result` for CSV output

**Ask first:**
- Changing the equilibrium formula or denominator
- Modifying the Beta policy parameterization
- Changing convergence evaluation thresholds
- Adding new dependencies beyond torch/numpy/pandas/matplotlib

**Never:**
- Use stochastic sampling in environment step (use closed-form expected utilities)
- Use Beta mode for evaluation (always use mean)
- Commit experiment results without verifying against theoretical benchmarks
- Mix denominator-4 and denominator-6 formulas within the same experiment track

## Codebase State

- Two-player symmetric track is mature and validated against theory
- Three-player symmetric track is implemented and functional
- Different-ability and different-cost tracks are implemented with gradient + PPO
- `envs/one_stage_env.py`, `envs/shared_payoff.py`, `envs/two_stage_env.py` are legacy/unused environments
- `paper_artifacts/` pipeline generates figures/tables from existing results data
- Config files contain Chinese comments in places (original development language)

## Terminology

| Term | Means |
|------|-------|
| `q` | Noise half-width for Uniform(-q, q) performance noise |
| `k` | Quadratic cost coefficient: cost = k * e^2 |
| `w_H`, `w_L` | Winner and loser prizes |
| `e*` | Theoretical Nash equilibrium effort level |
| `gap` | Absolute difference between learned effort and theoretical e* |
| `Beta policy` | Policy parameterized as Beta(alpha, beta) distribution over normalized effort [0,1] |
| `self-play` | Both tournament players share the same policy network |
| `opponent lag` | Opponent uses a delayed copy of the learner's policy |
| `bandit` | Single-step episode (one tournament round per episode) |
| `exploit` / `exploitability` | Maximum unilateral deviation gain from the learned strategy profile |
| `convergence eval` | Automated assessment of whether PPO has converged to theoretical equilibrium |
