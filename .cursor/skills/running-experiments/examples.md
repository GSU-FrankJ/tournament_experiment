# Experiment Examples

## Basic Experiments

### Single Q Value Training

```bash
# Train PPO on q=40 with default settings
python run/run_two_players.py --method ppo --q 40 --episodes 2048000 --seed 42

# Train gradient baseline for comparison
python run/run_two_players.py --method gradient --q 40
```

### Full Q Sweep

```bash
# Train on all q values in config (25, 40, 55)
python run/run_two_players.py --method ppo --episodes 2048000 --seed 42
```

## Custom Parameters

### Modified Game Parameters

```python
# custom_game_params.py
import sys
sys.path.insert(0, '.')

from config.one_stage_two_players import config as base_config
from run.run_two_players import run_ppo

cfg = base_config.copy()

# Custom game settings
cfg["k"] = 0.0005       # Higher cost
cfg["w_h"] = 8.0        # Higher prize spread
cfg["w_l"] = 2.0
cfg["seed"] = 50

# Theoretical: e* = (8-2)/(4*0.0005*q) = 3000/q
# q=25: e*=120, q=40: e*=75, q=55: e*≈54.5

results = run_ppo(
    cfg=cfg,
    episodes=2_048_000,
    train_qs=[25.0, 40.0],
    rollout_mode="selfplay",
    ablation_name="high_spread",
)
```

### Modified PPO Hyperparameters

```python
# custom_ppo.py
cfg = base_config.copy()

# More aggressive exploration
cfg["entropy_coef_start"] = 0.05
cfg["entropy_coef_end"] = 0.02

# Larger batch sizes
cfg["steps_per_update"] = 8192
cfg["minibatch_size"] = 2048

# Longer training
cfg["episodes"] = 4_096_000

results = run_ppo(cfg=cfg, ...)
```

## Ablation Studies

### Rollout Mode Comparison

```bash
# Selfplay (agents train against themselves)
python run/run_two_players.py --method ppo --q 40 --seed 42 \
    --rollout-mode selfplay

# vs_opponent (train against lagged opponent)
python run/run_two_players.py --method ppo --q 40 --seed 42 \
    --rollout-mode vs_opponent
```

### Convergence Profile Comparison

```bash
# Test different convergence profiles
for profile in relaxed default conservative aggressive; do
    python run/run_two_players.py --method ppo --q 40 --seed 42 \
        --cheap-gate-profile $profile
done
```

### Seed Sweep

```python
# seed_sweep.py
seeds = [42, 50, 68, 99, 123]

for seed in seeds:
    results = run_ppo(
        cfg=cfg,
        episodes=2_048_000,
        train_qs=[40.0],
        rollout_mode="selfplay",
        ablation_name=f"seed_sweep",
    )
```

## Analysis Workflows

### Compare Methods

```bash
# Run both methods
python run/run_two_players.py --method gradient --q 40
python run/run_two_players.py --method ppo --q 40 --seed 42

# Plot comparison
python tools/plot_convergence.py
```

### Detailed Analysis

```bash
# Generate detailed plots for specific run
python tools/plot_convergence_detailed.py --algorithm PPO --q 40.0

# Output files:
# results/convergence_plots/k5e4_wh8_wl3/q40_1_effort_comparison.png
# results/convergence_plots/k5e4_wh8_wl3/q40_2_agent_efforts.png
# ...
```

## Debugging Runs

### Quick Sanity Check

```bash
# Short run to verify setup
python run/run_two_players.py --method ppo --q 40 --episodes 51200 --seed 42

# Verify rollout modes work correctly
python tools/verify_rollout_modes.py
```

### Verbose Logging

```python
# The runner automatically saves logs to results/logs/
# Check latest log:
# results/logs/one_stage_two_players_ppo_q40_ep2048000_seed42_YYYYMMDD_HHMMSS.log
```

## Production Experiments

### Full Experiment Suite

```bash
#!/bin/bash
# run_full_suite.sh

# Gradient baselines
for q in 25 40 55; do
    python run/run_two_players.py --method gradient --q $q
done

# PPO with multiple seeds
for seed in 42 50 68; do
    for q in 25 40 55; do
        python run/run_two_players.py --method ppo --q $q \
            --episodes 2048000 --seed $seed
    done
done

# Generate all plots
python tools/plot_convergence.py
python tools/plot_convergence_detailed.py
```

### Parallel Execution

```bash
# Run different q values in parallel (on different GPUs/machines)
# Machine 1:
python run/run_two_players.py --method ppo --q 25 --seed 42

# Machine 2:
python run/run_two_players.py --method ppo --q 40 --seed 42

# Machine 3:
python run/run_two_players.py --method ppo --q 55 --seed 42
```

## Using the Quick Experiment Script

```bash
# Debug preset (fast, no convergence check)
python .cursor/skills/running-experiments/scripts/quick_experiment.py \
    --preset debug --q 40

# Production preset
python .cursor/skills/running-experiments/scripts/quick_experiment.py \
    --preset production --q 40 --seed 42 --ablation-name my_exp

# Custom game parameters
python .cursor/skills/running-experiments/scripts/quick_experiment.py \
    --preset production --k 0.0005 --w-h 8.0 --w-l 2.0 --q 40
```

## Expected Results

### Convergence Quality

| Quality | Gap from e* | KL Behavior |
|---------|-------------|-------------|
| Excellent | < 1.0 | Stable, low variance |
| Good | < 3.0 | Some variance, stable mean |
| Fair | < 5.0 | Higher variance |
| Poor | > 5.0 | Unstable or not converged |

### Typical Training Time

| Episodes | Steps/Update | Approx. Time |
|----------|--------------|--------------|
| 204,800 | 2048 | ~2-5 min |
| 512,000 | 4096 | ~10-15 min |
| 2,048,000 | 4096 | ~30-60 min |
| 4,096,000 | 4096 | ~1-2 hours |

Times vary based on hardware and convergence behavior.
