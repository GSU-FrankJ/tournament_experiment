# utils/

## Purpose

Core utility modules providing shared functionality across the codebase. Includes probability calculations, theoretical benchmarks, evaluation metrics, logging, and plotting.

## Key Contents

| File | Description |
|------|-------------|
| `prob.py` | **Core** - Closed-form win probability calculations |
| `theory.py` | **Core** - Theoretical equilibrium effort formulas |
| `eval.py` | Evaluation metrics (gap, quality classification) |
| `evaluation.py` | Extended evaluation (validation, aggregation) |
| `logger.py` | Winston-style logging infrastructure |
| `plot.py` | Plotting utilities for effort curves |
| `rollout_stats.py` | PPO training statistics tracking |
| `logging_example.py` | Example script for logging usage |
| `logging_integration_guide.md` | Guide for integrating logging |

## Entry Points / How to Use

These are library modules, imported by other code:

```python
# Win probability calculation
from utils.prob import p_from_efforts
prob = p_from_efforts(e1=80, e2=90, q=40)

# Theoretical effort
from utils.theory import e_star_two_players
e_star = e_star_two_players(q=40, w_h=6.5, w_l=3.0, k=0.0004)

# Evaluation metrics
from utils.eval import convergence_quality_from_gap
quality = convergence_quality_from_gap(gap=0.3)  # "Excellent"

# Logging
from utils.logger import get_logger
logger = get_logger("my_experiment")
```

## Dependencies & Contracts

**Depends on:**
- `numpy` - Numerical computations
- `torch` - For rollout_stats.py (PPO integration)
- `matplotlib` - For plot.py

**Provides to system:**
- `envs/` uses `prob.py` for win probabilities
- `run/` uses `theory.py`, `eval.py`, `logger.py`, `plot.py`, `rollout_stats.py`
- `config/` uses `theory.py` for theoretical calculations

## Module Details

### prob.py - Probability Utilities
```python
p_from_diff(d, q)      # P(win) from effort difference
p_from_efforts(e1, e2, q)  # P(player 1 wins)
win_prob_three_players(e, q)  # Three-player probability
win_prob_three_players_grad(e, q)  # Gradient of three-player
```

### theory.py - Theoretical Benchmarks
```python
e_star_two_players(q, w_h, w_l, k)  # Two-player equilibrium
e_star_three_players(q, w_h, w_l, k)  # Three-player equilibrium
e_star_two_players_asymmetric_cost(q, w_h, w_l, k1, k2)  # Asymmetric cost
```

### eval.py - Evaluation Metrics
```python
gap_from_theoretical(actual, theoretical)  # Absolute gap
convergence_quality_from_gap(gap)  # Quality buckets
build_csv_row(...)  # Standardized CSV row builder
```

### rollout_stats.py - Training Statistics
```python
WelfordAccumulator  # Online mean/variance
RolloutStatsAccumulator  # Effort/reward tracking
PPOUpdateStats  # Per-update statistics
```

## Quality Classification

| Quality | Gap from e* |
|---------|-------------|
| Excellent | < 0.5 |
| Good | < 1.0 |
| Fair | < 5.0 |
| Poor | ≥ 5.0 |

## Gotchas / Conventions

- `eval.py` and `evaluation.py` have overlapping functions - use `eval.py` for basic metrics
- `theory.py` uses denominator 4 for two-player (not 6)
- `prob.py` assumes uniform noise distribution
- Logger uses Winston-style hierarchical logging

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Added README.md for directory documentation |
