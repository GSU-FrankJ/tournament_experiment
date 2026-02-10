# tools/

## Purpose

Diagnostic, analysis, and utility scripts. These are standalone tools for debugging, verification, data processing, visualization, and ablation sweep orchestration that support the main experiment pipeline.

## Key Contents

### Visualization Tools
| File | Description |
|------|-------------|
| `plot_convergence.py` | Multi-algorithm comparison plots across q values; supports symmetric, asymmetric, and 3-player scenarios |
| `plot_convergence_detailed.py` | Detailed single-algorithm plots with per-agent breakdown; filterable by algorithm/q |
| `plot_exploit_ablation.py` | Exploit ablation result visualizations: setting comparison bars, stop-reason stacked bars, scatter plots, heatmaps, convergence curves |
| `plot_mechanism_ablation.py` | Mechanism ablation result visualizations: convergence speed, final error box plots, exploitability bars, learning curve overlays |

### Sweep / Ablation Tools
| File | Description |
|------|-------------|
| `sweep_exploit_ablation.py` | Orchestrate exploit-parameter ablation sweep (8 settings x 4 experiments x 3 q x 3 seeds = 288 runs); parallel execution, resume support, dry-run, smoke-test |
| `sweep_mechanism_ablation.py` | Orchestrate mechanism ablation sweep (4 settings x 4 experiments x 3 q x 3 seeds = 144 runs); same features as exploit sweep |

### Diagnostic Tools
| File | Description |
|------|-------------|
| `audit_rollout_modes.py` | Rigorous 5-point audit of rollout modes (selfplay vs vs_opponent) |
| `verify_rollout_modes.py` | Quick sanity check for rollout mode transition counts and buffer integrity |
| `diagnose_data_provenance.py` | Debug tool to prove opponent-generated actions don't mix into learner's PPO update |
| `check_env_noise_determinism.py` | Test whether env noise becomes deterministic due to re-seeding |

### Analysis Tools
| File | Description |
|------|-------------|
| `collect_and_pick_best.py` | Analyze hyperparameter sweeps at 200 updates, score convergence, emit 500-update command |
| `parse_logs_to_json.py` | Parse PPO training log files into structured JSON (alpha/beta, KL, effort metrics) |
| `backfill_params_from_dirname.py` | Extract hyperparameters from directory names into convergence JSON files |
| `generate_inventory.py` | Scan results dirs and generate INVENTORY.md with experiment coverage matrix |

### Migration Tools
| File | Description |
|------|-------------|
| `migrate_structure.py` | One-time migration script that reorganized flat results/ into experiment-type subdirectories |

## Entry Points / How to Use

### Ablation Sweeps

```bash
# Full exploit-parameter ablation sweep (288 runs, ~24h with 4 workers)
python tools/sweep_exploit_ablation.py --parallel 4

# Dry run (preview all commands)
python tools/sweep_exploit_ablation.py --dry-run

# Resume interrupted sweep
python tools/sweep_exploit_ablation.py --resume --parallel 4

# Smoke test (reduced episodes)
python tools/sweep_exploit_ablation.py --smoke-test --parallel 2

# Specific experiments only
python tools/sweep_exploit_ablation.py --experiments two_players,different_cost

# Full mechanism ablation sweep (144 runs)
python tools/sweep_mechanism_ablation.py --parallel 4

# Mechanism sweep with specific settings
python tools/sweep_mechanism_ablation.py --settings baseline,no_entropy
```

### Ablation Plotting

```bash
# Plot exploit ablation results (reads summary.csv)
python tools/plot_exploit_ablation.py --input results/ablation/exploit_params/

# Plot mechanism ablation results (reads runs/*.json or summary.json)
python tools/plot_mechanism_ablation.py --input results/ablation/mechanism/

# Filter to specific experiments
python tools/plot_mechanism_ablation.py --input results/ablation/mechanism/ \
    --experiments two_players,different_cost
```

### Convergence Plotting

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

# Full 5-point audit (more comprehensive)
python tools/audit_rollout_modes.py

# Check env noise determinism
python tools/check_env_noise_determinism.py

# Diagnose data provenance (opponent vs learner buffer)
python tools/diagnose_data_provenance.py
```

### Analysis

```bash
# Find best hyperparameters from sweep
python tools/collect_and_pick_best.py

# Parse training logs to JSON
python tools/parse_logs_to_json.py results/two_players/logs/experiment.log

# Parse all known log files
python tools/parse_logs_to_json.py --all

# Backfill params from directory names
python tools/backfill_params_from_dirname.py

# Generate experiment coverage inventory
python tools/generate_inventory.py
python tools/generate_inventory.py --output INVENTORY.md
```

## Dependencies & Contracts

**Depends on:**
- `results/*/convergence/` - Convergence JSON files for plotting
- `results/ablation/*/runs/` - Per-run result JSONs from sweeps
- `results/ablation/*/summary.csv` - Aggregated sweep results for plotting
- `run/run_*.py` - Sweep tools invoke runners as subprocesses
- `agents/`, `envs/`, `config/` - For verification/diagnostic tools
- `numpy`, `pandas`, `matplotlib` - External dependencies

**Provides to system:**
- Ablation sweep orchestration with parallel execution and resume support
- Verification of implementation correctness (rollout modes, buffer integrity)
- Visualization of training results and ablation experiments
- Analysis of hyperparameter sweeps and convergence quality
- Experiment coverage tracking (INVENTORY.md)

## Tool Categories

### Pre-Experiment
- `verify_rollout_modes.py` - Sanity check before training
- `audit_rollout_modes.py` - Full 5-point verification of implementation
- `check_env_noise_determinism.py` - Validate environment randomness

### Experiment Orchestration
- `sweep_exploit_ablation.py` - Run exploit-parameter ablation grid
- `sweep_mechanism_ablation.py` - Run mechanism ablation grid

### Post-Experiment
- `plot_convergence.py` - Multi-algorithm convergence comparison
- `plot_convergence_detailed.py` - Detailed per-agent convergence
- `plot_exploit_ablation.py` - Exploit ablation analysis figures
- `plot_mechanism_ablation.py` - Mechanism ablation analysis figures
- `collect_and_pick_best.py` - Score and rank hyperparameter configs
- `generate_inventory.py` - Coverage matrix generation

### Debugging
- `diagnose_data_provenance.py` - Track action origins in PPO buffer
- `check_env_noise_determinism.py` - Debug environment RNG behavior

### Data Processing
- `parse_logs_to_json.py` - Log file -> JSON conversion
- `backfill_params_from_dirname.py` - Extract params from directory naming convention
- `migrate_structure.py` - One-time directory reorganization (already applied)

## Gotchas / Conventions

- Tools are standalone scripts, not imported as modules
- Most tools use argparse for CLI options; run with `--help` for full usage
- Sweep tools support `--dry-run` to preview commands without executing
- Sweep tools support `--resume` to skip already-completed runs
- Sweep tools save per-run results immediately for crash resilience
- Plotting tools read from `results/*/convergence/` or `results/ablation/*/`
- Verification tools import from `agents/`, `envs/`, `config/` - ensure repo root is on PYTHONPATH
- `migrate_structure.py` is idempotent but intended as a one-time migration

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-09 | Updated README.md; added sweep, ablation plotting, inventory, and migration tools |
| 2026-02-03 | Added README.md; moved plot scripts from results/ |
