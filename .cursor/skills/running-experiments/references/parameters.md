# Experiment Parameters Reference

## Game Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | float | 0.0004 | Quadratic cost coefficient |
| `w_h` | float | 6.5 | High prize (winner) |
| `w_l` | float | 3.0 | Low prize (loser) |
| `q` | float | 40.0 | Noise parameter (uniform ±q) |
| `effort_range` | [lo, hi] | [0, 200] | Effort bounds |

## PPO Hyperparameters

### Training Schedule

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps_per_update` | 4096 | Steps collected before each update |
| `minibatch_size` | 1024 | Minibatch size for SGD |
| `update_epochs` | 6 | Epochs per update |
| `episodes` | 2,048,000 | Total environment steps |
| `max_updates` | 500 | Maximum number of updates |

### Learning Rate

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_start` | 3e-4 | Initial learning rate |
| `lr_end` | 2e-4 | Final learning rate (linear decay) |

### Entropy

| Parameter | Default | Description |
|-----------|---------|-------------|
| `entropy_coef_start` | 0.03 | Initial entropy coefficient |
| `entropy_coef_hold` | 0.03 | Entropy coefficient during hold phase |
| `entropy_coef_end` | 0.015 | Final entropy coefficient |

### Clipping

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_range_start` | 0.50 | Initial PPO clip range |
| `clip_range_end` | 0.35 | Final PPO clip range |
| `target_kl` | 0.08 | Target KL divergence for adaptive updates |

## Convergence Settings

### Cheap Gate Profiles

All profiles use `window_size: 20`.

| Profile | `mean_kl_thresh` | `std_kl_thresh` | `drift_effort_thresh` | `patience_drift` |
|---------|------------------|-----------------|----------------------|------------------|
| `relaxed` | 0.015 | 0.012 | 8.0 | 1 |
| `default` | 0.0045 | 0.0035 | 2.0 | 2 |
| `conservative` | 0.0038 | 0.0030 | 1.5 | 3 |
| `aggressive` | 0.0060 | 0.0075 | 5.5 | 1 |

### Exploitability Check

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exploit_eps` | 0.05 | Exploitability threshold |
| `patience_exploit` | 5 | Patience for exploit check |
| `M` | 8192 | Monte Carlo samples |

## Gradient Method Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gradient_lr` | 0.08 | Learning rate |
| `gradient_steps` | 1500 | Maximum iterations |
| `gradient_delta` | 0.5 | Finite difference step |
| `gradient_tol` | 1e-4 | Convergence tolerance |
| `gradient_num_samples` | 64 | MC samples per gradient |
| `gradient_init_perturb` | 1.0 | Initial perturbation |

## Recommended Configurations

### Fast Iteration (Debugging)

```python
cfg = {
    "episodes": 204800,        # 10x fewer steps
    "steps_per_update": 2048,  # Smaller batches
    "eval_every_updates": 5,   # Frequent eval
}
```

### Production (Full Training)

```python
cfg = {
    "episodes": 4_096_000,     # Extended training
    "steps_per_update": 4096,
    "entropy_coef_start": 0.03,
    "convergence": {"enabled": True, "cheap_gate_profile": "relaxed"},
}
```

### Hyperparameter Search

```python
# Sweep over multiple seeds
for seed in [42, 50, 68, 99]:
    run_ppo(..., seed=seed, ablation_name=f"sweep_seed{seed}")
```
