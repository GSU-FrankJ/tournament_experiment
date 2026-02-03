# tools/

## Purpose

Diagnostic, analysis, and utility scripts. These are standalone tools for debugging, verification, data processing, and visualization that support the main experiment pipeline.

## Key Contents

### Visualization Tools
| File | Description |
|------|-------------|
| `plot_convergence.py` | Multi-algorithm comparison plots |
| `plot_convergence_detailed.py` | Detailed single-algorithm plots with per-agent breakdown |

### Diagnostic Tools
| File | Description |
|------|-------------|
| `audit_rollout_modes.py` | Rigorous audit of rollout modes implementation |
| `verify_rollout_modes.py` | Quick sanity check for rollout mode behavior |
| `diagnose_data_provenance.py` | Debug tool for data mixing issues |
| `check_env_noise_determinism.py` | Test environment noise behavior |

### Analysis Tools
| File | Description |
|------|-------------|
| `collect_and_pick_best.py` | Analyze hyperparameter sweeps, find best config |
| `parse_logs_to_json.py` | Parse training logs to JSON format |
| `backfill_params_from_dirname.py` | Extract hyperparams from directory names |

### Experiment Tools
| File | Description |
|------|-------------|
| `sweep_one_stage_vs_opponent.py` | Automated hyperparameter sweep runner |

## Entry Points / How to Use

### Plotting

```bash
# Generate multi-algorithm comparison plots
python tools/plot_convergence.py

# Generate detailed per-agent plots
python tools/plot_convergence_detailed.py

# Filter by algorithm or q value
python tools/plot_convergence_detailed.py --algorithm PPO --q 25.0
```

### Verification

```bash
# Quick rollout mode check
python tools/verify_rollout_modes.py

# Full audit (more comprehensive)
python tools/audit_rollout_modes.py
```

### Analysis

```bash
# Find best hyperparameters from sweep
python tools/collect_and_pick_best.py

# Parse logs to JSON
python tools/parse_logs_to_json.py results/logs/experiment.log
```

### Sweeps

```bash
# Run automated hyperparameter sweep
python tools/sweep_one_stage_vs_opponent.py
```

## Dependencies & Contracts

**Depends on:**
- `results/convergence_history/` - Source data for plotting
- `agents/`, `envs/`, `config/` - For verification tools
- `numpy`, `pandas`, `matplotlib` - External dependencies

**Provides to system:**
- Verification of implementation correctness
- Visualization of training results
- Analysis of hyperparameter sweeps

## Tool Categories

### Pre-Experiment
- `verify_rollout_modes.py` - Sanity check before training
- `audit_rollout_modes.py` - Full verification of implementation

### Post-Experiment
- `plot_convergence*.py` - Visualize results
- `collect_and_pick_best.py` - Analyze sweep results
- `parse_logs_to_json.py` - Extract metrics from logs

### Debugging
- `diagnose_data_provenance.py` - Track action origins
- `check_env_noise_determinism.py` - Debug environment behavior

## Gotchas / Conventions

- Tools are standalone scripts, not imported as modules
- Most tools use argparse for CLI options
- Plotting tools read from `results/convergence_history/`
- Verification tools may modify global state - run with care

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Added README.md; moved plot scripts from results/ |
