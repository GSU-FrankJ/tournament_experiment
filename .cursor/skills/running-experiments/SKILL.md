---
name: running-experiments
description: Run tournament game theory experiments with PPO or gradient methods. Use when the user wants to run experiments, train agents, set up custom parameter sweeps, analyze convergence, or asks about run_two_players.py, experiment configuration, or training commands.
---

# Running Experiments

## Available Experiment Types

| Experiment | Script | Config | Description |
|------------|--------|--------|-------------|
| **Two-Player Symmetric** | `run/run_two_players.py` | `config/one_stage_two_players.py` | Identical players (k1=k2, l1=l2) |
| **Different Cost** | `run/run_different_cost.py` | `config/one_stage_different_cost.py` | Asymmetric costs (k1 < k2, l1=l2) |
| **Different Ability** | `run/run_different_ability.py` | `config/one_stage_different_ability.py` | Asymmetric abilities (k1=k2, l1 > l2) |
| **Three Players** | `run/run_three_players.py` | `config/one_stage_three_players.py` | Three identical players |

## Quick Start

### Two-Player Symmetric (Default)

```bash
# PPO Training (Recommended)
python run/run_two_players.py --method ppo --q 40 --episodes 2048000 --seed 42

# Gradient Baseline
python run/run_two_players.py --method gradient --q 40
```

### Different Ability Experiment

```bash
# Gradient baseline (l1=10, l2=5 by default)
python run/run_different_ability.py --method gradient --q 40

# PPO training
python run/run_different_ability.py --method ppo --q 40 --episodes 2048000 --seed 42

# Custom ability parameters
python run/run_different_ability.py --method ppo --q 40 --l1 15 --l2 5
```

### Different Cost Experiment

```bash
# Gradient baseline (k1=0.0004, k2=0.00055 by default)
python run/run_different_cost.py --method gradient --q 40

# PPO training
python run/run_different_cost.py --method ppo --q 40 --episodes 2048000 --seed 42
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

### Convergence & Exploitability Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--exploit-every-updates` | 10 | Max interval between exploitability evaluations |
| `--disable-cheap-gate` | False | Gate always ON: exploitability eval eligible every update |
| `--disable-exploitability` | False | Never evaluate exploitability; converge on effort gap only |

**Cheap Gate**: 决定何时触发 exploitability 检查的"门控"机制，基于 KL divergence 和 policy drift 是否稳定。

**Exploitability**: 衡量当前策略的 ε-Nash 近似程度。如果对手可以通过单方面偏离获得超过 ε 的收益，则策略尚未收敛。

```bash
# 每5个update评估一次exploitability，禁用cheap gate门控
python run/run_two_players.py --method ppo --q 40 \
    --exploit-every-updates 5 --disable-cheap-gate

# 完全禁用exploitability评估（仅基于effort gap收敛）
python run/run_two_players.py --method ppo --q 40 --disable-exploitability
```

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

### Two-Player Symmetric

```
e* = (w_h - w_l) / (4 * k * q)
```

Examples with default `w_h=6.5, w_l=3.0, k=0.0004`:
- q=25: e* ≈ 87.5
- q=40: e* ≈ 54.69
- q=55: e* ≈ 39.77

### Different Ability (Additive Model)

Model: `y_i = e_i + l_i + ε_i` where `l1 > l2`

```
e* = ((2q - (l1 - l2)) * (w_h - w_l)) / (8 * k * q²)
```

Both players exert **same effort** at equilibrium; player 1 wins more often due to ability advantage.

Examples with `l1=10, l2=5, k=0.0004, w_h=6.5, w_l=3.0`:
- q=25: e* ≈ 78.75, P(p1 wins) ≈ 0.68
- q=40: e* ≈ 51.27, P(p1 wins) ≈ 0.56
- q=55: e* ≈ 38.07, P(p1 wins) ≈ 0.54

### Different Cost (Asymmetric Cost)

```
e1* = 2 k2 q (w_H - w_L) / (8 k1 k2 q² - (k1 - k2)(w_H - w_L))
e2* = 2 k1 q (w_H - w_L) / (8 k1 k2 q² - (k1 - k2)(w_H - w_L))
```

Player with lower cost (k1) exerts more effort at equilibrium.

## Custom Parameter Experiments

### Method 1: CLI Override

```bash
python run/run_two_players.py --method ppo --q 40 --seed 50 \
    --episodes 4096000 --rollout-mode vs_opponent
```

```

## Output Files

### Two-Player Symmetric
| Output | Location |
|--------|----------|
| Convergence JSON | `results/convergence_history/{method}_q{q}_seed{seed}_{ablation}_convergence.json` |
| Results CSV | `results/one_stage_two_players_v2.csv` |
| Training logs | `results/logs/one_stage_two_players_*.log` |

### Different Ability
| Output | Location |
|--------|----------|
| Convergence JSON | `results/convergence_history/different_ability_{method}_q{q}_convergence.json` |
| Results CSV | `results/different_ability_two_players.csv` |
| Training logs | `results/logs/different_ability_*.log` |

### Different Cost
| Output | Location |
|--------|----------|
| Convergence JSON | `results/convergence_history/different_cost_{method}_q{q}_convergence.json` |
| Results CSV | `results/different_cost_two_players.csv` |
| Training logs | `results/logs/different_cost_*.log` |

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
