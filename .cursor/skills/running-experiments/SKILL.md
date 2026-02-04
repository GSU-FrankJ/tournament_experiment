---
name: running-experiments
description: Run tournament game theory experiments with PPO or gradient methods. Use when the user wants to run experiments, train agents, set up custom parameter sweeps, analyze convergence, or asks about run_two_players.py, experiment configuration, or training commands.
---

# Running Experiments

## Quick Start

### PPO Training (Recommended)

```bash
# Default modern config - selfplay with theory alignment
python run/run_two_players.py --method ppo --q 40 --episodes 2048000 --seed 42

# Multiple q values (omit --q to sweep all)
python run/run_two_players.py --method ppo --episodes 2048000 --seed 42
```

### Gradient Baseline

```bash
python run/run_two_players.py --method gradient --q 40
```

<!-- MC-FD Solver (备用，目前不常用)
### MC-FD Solver

```bash
python run/run_mcfd.py --w-h 6.5 --w-l 3.0 --k 0.0004 --sigma1 25.0
```
-->

## Core CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | `ppo` | Algorithm: `ppo` or `gradient` |
| `--q` | (sweeps all) | Noise parameter (single value) |
| `--episodes` | 2048000 | Total environment steps |
| `--seed` | 42 | Random seed |

### PPO-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--rollout-mode` | `selfplay` | `selfplay` or `vs_opponent` |
| `--theory-align-v2` | True | Mean+concentration policy head |
| `--enable-convergence-eval` | True | Early stopping on convergence |
| `--cheap-gate-profile` | `relaxed` | KL threshold profile |

### Disabling Defaults

```bash
# Disable theory alignment
python run/run_two_players.py --method ppo --no-theory-align-v2

# Disable convergence evaluation
python run/run_two_players.py --method ppo --no-convergence-eval
```

## Configuration Files

Configuration lives in `config/one_stage_two_players.py`. Key parameters:

```python
config = {
    # Game parameters
    "k": 0.0004,          # Quadratic cost coefficient
    "w_h": 6.5,           # High prize
    "w_l": 3.0,           # Low prize
    "q_list": [25.0, 40.0, 55.0],  # Noise values to sweep
    
    # PPO hyperparameters
    "steps_per_update": 4096,
    "minibatch_size": 1024,
    "update_epochs": 6,
    "episodes": 2_048_000,
    
    # Learning rate schedule
    "lr_start": 3e-4,
    "lr_end": 2e-4,
    
    # Entropy schedule
    "entropy_coef_start": 0.03,
    "entropy_coef_end": 0.015,
    
    # Convergence settings
    "convergence": {
        "enabled": True,
        "cheap_gate_profile": "relaxed",
    }
}
```

## Theoretical Equilibrium

For two-player single-stage tournament, equilibrium effort:

```
e* = (w_h - w_l) / (4 * k * q)
```

Examples with default `w_h=6.5, w_l=3.0, k=0.0004`:
- q=25: e* ≈ 87.5
- q=40: e* ≈ 54.69
- q=55: e* ≈ 39.77

## Custom Parameter Experiments

### Method 1: CLI Override

```bash
python run/run_two_players.py --method ppo --q 40 --seed 50 \
    --episodes 4096000 --rollout-mode vs_opponent
```

### Method 2: Custom Script

Create a script like `run/run_ppo_custom_params.py`:

```python
from config.one_stage_two_players import config as base_config
from run.run_two_players import run_ppo

cfg = base_config.copy()
cfg["k"] = 0.0005        # Custom cost
cfg["w_h"] = 8.0         # Custom high prize
cfg["w_l"] = 3.0         # Custom low prize
cfg["seed"] = 42

results = run_ppo(
    cfg=cfg,
    episodes=2_048_000,
    train_qs=[25.0, 40.0],
    eval_qs=[25.0, 40.0],
    rollout_mode="selfplay",
    ablation_name="my_experiment",
)
```

## Output Files

| Output | Location |
|--------|----------|
| Convergence JSON | `results/convergence_history/{method}_q{q}_seed{seed}_{ablation}_convergence.json` |
| Metadata JSON | `results/convergence_history/*_metadata.json` |
| Results CSV | `results/one_stage_two_players_v2.csv` |
| Training logs | `results/logs/` |

### Convergence JSON Structure

```json
{
  "config": { "q": 40.0, "seed": 42, ... },
  "history": {
    "effort_agent1": [50.1, 51.2, ...],
    "effort_agent2": [49.8, 51.0, ...],
    "kl_divergence": [0.01, 0.008, ...],
    "update_idx": [0, 1, 2, ...]
  },
  "final": {
    "theoretical_effort": 54.69,
    "final_effort": 54.2,
    "gap": 0.49
  }
}
```

## Analysis Tools

### Plotting Convergence

```bash
# Multi-algorithm comparison
python tools/plot_convergence.py

# Detailed per-agent plots
python tools/plot_convergence_detailed.py --algorithm PPO --q 25.0
```

### Hyperparameter Sweeps

```bash
# Run sweep
python tools/sweep_one_stage_vs_opponent.py

# Analyze results
python tools/collect_and_pick_best.py
```

## Experiment Workflow

### Standard Experiment

1. **Verify setup**:
   ```bash
   python tools/verify_rollout_modes.py
   ```

2. **Run experiment**:
   ```bash
   python run/run_two_players.py --method ppo --q 40 --seed 42
   ```

3. **Check convergence**:
   ```bash
   python tools/plot_convergence_detailed.py --q 40.0
   ```

### Ablation Study

1. **Create ablation script** with modified config
2. **Run with ablation name**:
   ```python
   run_ppo(..., ablation_name="my_ablation")
   ```
3. **Compare results** in `results/convergence_history/`

## Convergence Profiles

| Profile | Use Case |
|---------|----------|
| `relaxed` | Default, tolerates higher KL variance |
| `default` | Standard thresholds |
| `conservative` | Stricter convergence criteria |
| `aggressive` | Fast early stopping |

```bash
python run/run_two_players.py --method ppo --cheap-gate-profile conservative
```

## Common Issues

### High KL Divergence

- Reduce learning rate: modify `lr_start/lr_end` in config
- Use conservative profile: `--cheap-gate-profile conservative`

### Slow Convergence

- Increase episodes: `--episodes 4096000`
- Adjust entropy: modify `entropy_coef_*` in config

### Reproducibility

- Always set `--seed` for reproducible results
- Check git SHA in metadata files for version tracking

## Additional Resources

For detailed implementation:
- [run/README.md](../../../run/README.md) - Runner documentation
- [config/README.md](../../../config/README.md) - Configuration guide
- [tools/README.md](../../../tools/README.md) - Analysis tools
