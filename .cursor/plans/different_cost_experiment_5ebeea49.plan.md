---
name: Different Cost Experiment
overview: Implement a one-stage, two-player tournament experiment with asymmetric cost parameters (k1 < k2, l1 = l2) by creating a new dedicated runner script run/run_different_cost.py that integrates the existing DifferentCostEnv environment.
todos:
  - id: create-runner
    content: Create run/run_different_cost.py - new dedicated runner for asymmetric cost experiments
    status: pending
  - id: create-config
    content: Create config/one_stage_different_cost.py with parameterized defaults and theoretical effort helpers
    status: pending
  - id: implement-gradient
    content: Implement gradient_descent_different_cost() in the new runner (no symmetry enforcement)
    status: pending
  - id: implement-ppo
    content: Implement run_ppo_different_cost() with per-player theoretical tracking and convergence
    status: pending
  - id: implement-outputs
    content: Implement convergence JSON and CSV output with per-player metrics
    status: pending
  - id: update-plotting
    content: Update plot_convergence.py to support asymmetric scenarios with two theoretical lines
    status: pending
isProject: false
---

# One-Stage Different Cost Experiment Implementation

## Overview

This plan implements experiment type "III.2.b Two Players with Different Cost Functions" from [docs/experiment_plan.md](docs/experiment_plan.md), where:

- Cost functions: `C_i(e) = k_i * e^2` with `k1 < k2`
- Ability parameters: `l1 = l2` (equal, so ability doesn't affect win probability)
- Theoretical equilibrium efforts differ per player

## Existing Components (No Changes Needed)

The following components already exist and are compatible:

- **Environment**: [envs/different_cost_env.py](envs/different_cost_env.py) - `DifferentCostEnv` class with k1, k2 support
- **Theory Functions**: [utils/theory.py](utils/theory.py) - `e_star_two_players_asymmetric_cost()` and `eu_two_players_asymmetric_cost()`
- **Config Template**: [config/asymmetric_cost_two_players.py](config/asymmetric_cost_two_players.py) - k1=0.0004, k2=0.00055
- **PPO Agent**: [agents/ppo_two_players_clean.py](agents/ppo_two_players_clean.py) - reusable with state encoding `[q, k, w_gap]`

## Implementation Tasks

### 1. Create New Runner Script `run/run_different_cost.py`

Create a dedicated runner script (following the pattern of [run/run_three_players.py](run/run_three_players.py)):

```python
#!/usr/bin/env python3
"""
One-Stage Two-Player Different Cost Experiment (k1 < k2, l1 = l2)

Usage:
    # Gradient baseline
    python run/run_different_cost.py --method gradient --q 40
    
    # PPO training
    python run/run_different_cost.py --method ppo --q 40 --episodes 2048000 --seed 42
"""
```

**Key imports:**

- `from envs.different_cost_env import DifferentCostEnv`
- `from config.one_stage_different_cost import config as base_config`
- `from utils.theory import e_star_two_players_asymmetric_cost`
- `from agents.ppo_two_players_clean import PPOTwoPlayersBandit, PPOConfig`

### 2. Create Config File `config/one_stage_different_cost.py`

New config file with parameterized defaults (following pattern of [config/one_stage_two_players.py](config/one_stage_two_players.py)):

```python
config = {
    # Asymmetric cost parameters
    "k1": 0.0004,      # Player 1 cost (lower)
    "k2": 0.00055,     # Player 2 cost (higher)
    
    # Game parameters
    "w_h": 6.5,
    "w_l": 3.0,
    "q": 40.0,
    "q_list": [25.0, 40.0, 55.0],
    "effort_range": [0, 200],
    "effort_bounds_stage2": [0, 200],
    "seed": 42,
    "num_players": 2,
    
    # PPO hyperparameters (same as symmetric baseline)
    "steps_per_update": 4096,
    "minibatch_size": 1024,
    "update_epochs": 6,
    "episodes": 2_048_000,
    # ... (copy convergence settings from one_stage_two_players.py)
}

def get_theoretical_efforts(q, k1, k2, w_h, w_l):
    """Compute per-player equilibrium efforts."""
    from utils.theory import e_star_two_players_asymmetric_cost
    return e_star_two_players_asymmetric_cost(q, w_h, w_l, k1, k2)
```

### 3. Implement Gradient Solver in Runner

`gradient_descent_different_cost()` function:

```python
def gradient_descent_different_cost(cfg, *, lr, steps, eps, tol, num_samples, log=True):
    """Two-player gradient ascent for k1 != k2 (no symmetry enforcement)."""
    env = DifferentCostEnv(
        w_h=cfg["w_h"], w_l=cfg["w_l"],
        k1=cfg["k1"], k2=cfg["k2"],
        q=cfg["q"], effort_bounds=tuple(cfg["effort_bounds_stage2"]),
        seed=cfg.get("seed", 42)
    )
    e1_star, e2_star = e_star_two_players_asymmetric_cost(cfg["q"], cfg["w_h"], cfg["w_l"], cfg["k1"], cfg["k2"])
    
    # Initialize near but not at theoretical values
    e1 = e1_star * 0.8
    e2 = e2_star * 1.2
    
    # Gradient ascent loop (NO symmetry enforcement - players have different optima)
    for step in range(steps):
        g1, g2 = _compute_gradients(env, e1, e2, eps, num_samples)
        e1 = clip(e1 + lr * g1, bounds)
        e2 = clip(e2 + lr * g2, bounds)
        
        # Convergence check: both players near their respective theoretical values
        gap1, gap2 = abs(e1 - e1_star), abs(e2 - e2_star)
        if max(gap1, gap2) < tol:
            break
    
    return (e1, e2), {"gap1": gap1, "gap2": gap2, ...}
```

### 4. Implement PPO Training Loop

`run_ppo_different_cost()` function with per-player tracking:

```python
def run_ppo_different_cost(cfg, episodes, train_qs, eval_qs, **kwargs):
    """Train PPO for asymmetric cost scenario."""
    k1, k2 = cfg["k1"], cfg["k2"]
    w_h, w_l = cfg["w_h"], cfg["w_l"]
    
    # Two agents with different k values in their state encoding
    # Option A: Single shared agent, player-specific state [q, k_i, w_gap]
    # Option B: Two separate agents (more flexible for asymmetric scenarios)
    
    env = DifferentCostEnv(w_h=w_h, w_l=w_l, k1=k1, k2=k2, q=q, ...)
    
    # Track per-player convergence
    for q in train_qs:
        e1_star, e2_star = e_star_two_players_asymmetric_cost(q, w_h, w_l, k1, k2)
        # Training loop...
        # Log: effort_agent1, effort_agent2, gap1, gap2
```

### 5. Convergence History JSON Format

Output to `results/convergence_history/different_cost_{method}_q{q}_seed{seed}_convergence.json`:

```json
{
  "config": {
    "q": 40.0, "k1": 0.0004, "k2": 0.00055,
    "w_h": 6.5, "w_l": 3.0, "seed": 42
  },
  "scenario": "different_cost",
  "theoretical": {
    "effort1": 49.0,
    "effort2": 35.6
  },
  "history": {
    "effort_agent1": [...],
    "effort_agent2": [...],
    "gap_agent1": [...],
    "gap_agent2": [...],
    "update_idx": [...]
  },
  "final": {
    "effort1": 48.9,
    "effort2": 35.5,
    "gap1": 0.1,
    "gap2": 0.1
  }
}
```

### 6. CSV Output Schema

Output to `results/different_cost_two_players.csv` with columns:

- `q`, `k1`, `k2`, `w_h`, `w_l`, `seed`
- `theoretical_effort1`, `theoretical_effort2`
- `final_effort1`, `final_effort2`
- `gap1`, `gap2`, `max_gap`
- `method`, `episodes`, `converged`

### 7. Update Plotting Support

Update [tools/plot_convergence.py](tools/plot_convergence.py) to:

- Detect `different_cost` scenario from JSON config
- Plot two theoretical reference lines (e1*, e2*) with different colors
- Show per-agent convergence trajectories
- Label plot: "Different Cost (k1=0.0004, k2=0.00055)"

## File Changes Summary


| File                                 | Action                                                |
| ------------------------------------ | ----------------------------------------------------- |
| `run/run_different_cost.py`          | **Create** - new dedicated runner script              |
| `config/one_stage_different_cost.py` | **Create** - dedicated config for k1 < k2 experiments |
| `tools/plot_convergence.py`          | Update for per-player theoretical lines               |


## Usage Examples

```bash
# Gradient baseline for different cost
python run/run_different_cost.py --method gradient --q 40

# PPO training for different cost
python run/run_different_cost.py --method ppo --q 40 --episodes 2048000 --seed 42

# PPO with convergence evaluation
python run/run_different_cost.py --method ppo --q 40 --episodes 2048000 --seed 42 \
    --enable-convergence-eval --cheap-gate-profile relaxed

# Custom k1, k2 values (CLI override)
python run/run_different_cost.py --method ppo --q 40 --k1 0.0003 --k2 0.0006

# Sweep all q values
python run/run_different_cost.py --method gradient
```

## Theoretical Equilibrium Reference

For k1 < k2, the equilibrium efforts are:

```
e1* = 2 k2 q (w_H - w_L) / (8 k1 k2 q^2 - (k1 - k2)(w_H - w_L))
e2* = 2 k1 q (w_H - w_L) / (8 k1 k2 q^2 - (k1 - k2)(w_H - w_L))
```

With default parameters (k1=0.0004, k2=0.00055, w_h=6.5, w_l=3.0):

- q=25: e1* ≈ 78.3, e2* ≈ 56.9
- q=40: e1* ≈ 49.0, e2* ≈ 35.6
- q=55: e1* ≈ 35.7, e2* ≈ 26.0

(Player 1 with lower cost exerts more effort in equilibrium)