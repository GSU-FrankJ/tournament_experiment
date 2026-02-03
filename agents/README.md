# agents/

## Purpose

Learning agents for game-theoretic tournament experiments. Contains reinforcement learning (PPO) and gradient-based optimization algorithms that learn optimal effort strategies in tournament environments.

## Key Contents

| File | Description |
|------|-------------|
| `ppo_two_players_clean.py` | PPO agent with Beta policy for two-player tournaments. Supports self-play, opponent lag, theory alignment. |
| `mc_fd_crn_solver.py` | Monte Carlo Finite-Difference solver with Common Random Numbers for gradient-based optimization. |

## Entry Points / How to Use

Agents are instantiated by run scripts, not invoked directly:

```python
# PPO agent (from run/run_two_players.py)
from agents.ppo_two_players_clean import PPOTwoPlayersBandit, PPOConfig
agent = PPOTwoPlayersBandit(effort_bounds=(0, 200), config=PPOConfig(), device='cuda')

# Gradient solver (from run/run_mcfd.py)
from agents.mc_fd_crn_solver import MCFDConfig, gradient_ascent_dynamics
config = MCFDConfig(w_h=6.5, w_l=3.0, k=0.0004, sigma1=25.0, sigma2=25.0)
result = gradient_ascent_dynamics(config)
```

## Dependencies & Contracts

**Depends on:**
- `torch` - Neural network operations (PPO)
- `numpy` - Numerical computations
- No direct imports from `envs/` or `config/` (environment-agnostic)

**Provides to system:**
- `PPOTwoPlayersBandit` - Main PPO agent class with `act()`, `update()`, `train()` methods
- `PPOConfig` - Dataclass for PPO hyperparameters
- `ActorCritic`, `ActorCriticMeanConc` - Neural network architectures
- `MCFDConfig`, `gradient_ascent_dynamics()` - Gradient solver interface

## Key Classes and Patterns

### PPOTwoPlayersBandit
- **Policy**: Beta distribution over normalized effort [0,1], mapped to `effort_bounds`
- **Architecture**: Two heads - `ActorCriticMeanConc` (mean+concentration) or `ActorCritic` (alpha/beta)
- **Self-play modes**: `ema` (exponential moving average), `periodic` (sync every N updates), `snapshot`
- **Theory alignment**: Optional `theory_align_v2` mode with variance penalty

### MCFDConfig
- Configuration dataclass with defaults for MC-FD gradient solver
- Parameters: `w_h`, `w_l`, `k`, `sigma1`, `sigma2`, `delta`, `eta`, `num_samples`

## Gotchas / Conventions

- Effort is internally normalized to [0,1] via Beta distribution, then scaled to `effort_bounds`
- PPO stores transitions for **both** agents in self-play mode
- `theory_align_v2` uses mean+concentration parameterization with entropy coefficient forced to 0
- Opponent policy is lagged (not the current learner) to stabilize training

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Added README.md for directory documentation |
