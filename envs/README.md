# envs/

## Purpose

Game environments implementing tournament mechanics. Each environment follows a gym-like API with `reset()` and `step()` methods, computing win probabilities and expected utilities for different tournament scenarios.

## Key Contents

| File | Description |
|------|-------------|
| `two_players_env.py` | **Primary** - Symmetric two-player single-stage tournament (closed-form probabilities) |
| `three_players_env.py` | Three-player one-stage tournament with analytic/MC probability |
| `one_stage_env.py` | Generic one-stage tournament (2 or 3 players, uniform noise) |
| `two_stage_env.py` | Two-stage sequential tournament with information revelation modes |
| `different_ability_env.py` | Two-player with asymmetric ability parameters (l1 ≠ l2) |
| `different_cost_env.py` | Two-player with asymmetric cost parameters (k1 ≠ k2) |
| `shared_payoff.py` | Shared utility function for expected utility calculation |

## Entry Points / How to Use

Environments are instantiated by run scripts:

```python
from envs.two_players_env import TwoPlayersEnv

env = TwoPlayersEnv(q=40.0, k=0.0004, w_h=6.5, w_l=3.0, effort_bounds=(0, 200))
obs = env.reset()
obs, rewards, costs, done, info = env.step([e1, e2])
```

## Dependencies & Contracts

**Depends on:**
- `utils.prob` - Closed-form win probability functions (`p_from_efforts`, `win_prob_three_players`)
- `utils.logger` - Logging (some environments)
- `numpy` - Numerical computations

**Provides to system:**
- Gym-like API: `reset()`, `step(actions)` → `(obs, rewards, costs, done, info)`
- `expected_utility()` - Expected utility given efforts
- `probability_win()` - Win probability calculation

## Core Mechanics

### Win Probability (Two-Player)
For players with efforts `e1, e2` and noise `ε ~ Uniform(-q, q)`:
- Output: `y_i = e_i + ε_i`
- Win probability: `P(y1 > y2)` computed via triangular distribution

### Expected Utility
```
E[u_i] = w_L + P(win) * (w_H - w_L) - k * e_i²
```

### Theoretical Equilibrium
- Two-player: `e* = (w_H - w_L) / (4 * q * k)`
- Three-player: `e* = (w_H - w_L) / (4 * q * k)` (same formula)

## Environment API

```python
class TwoPlayersEnv:
    def __init__(self, q, k, w_h, w_l, effort_bounds):
        """Initialize environment with game parameters."""
    
    def reset(self) -> np.ndarray:
        """Reset environment, return initial observation."""
    
    def step(self, actions: List[float]) -> Tuple[obs, rewards, costs, done, info]:
        """Execute one step with given efforts."""
    
    def expected_utility(self, e1: float, e2: float, player: int) -> float:
        """Compute expected utility for a player."""
```

## Gotchas / Conventions

- Environments are stateless for single-stage games (each step is independent)
- `step()` returns tuple: `(obs, rewards, costs, done, info)` - note separate `costs` field
- `effort_bounds` constrains valid effort range but doesn't clip automatically
- Two-stage environments track state across `step_stage1()` and `step_stage2()` calls
- Different ability environments use `y_i = e_i + l_i + ε_i` (ability term added)

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Added README.md for directory documentation |
