# run/

## Purpose

Experiment entry points and runners. These scripts orchestrate training by combining agents, environments, and configurations. They handle CLI parsing, logging, and result collection.

## Key Contents

| File | Description |
|------|-------------|
| `run_two_players.py` | **Primary entry point** - Two-player tournament experiments (PPO + gradient) |
| `run_mcfd.py` | Standalone MC-FD gradient solver |
| `run_ppo_custom_params.py` | Custom parameter sweeps for convergence analysis |

## Entry Points / How to Use

### Main Experiments (run_two_players.py)

```bash
# PPO with default modern config (recommended)
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 50

# Gradient baseline
python run/run_two_players.py --method gradient --q 40

# PPO with custom options
python run/run_two_players.py --method ppo --q 55 --episodes 2048000 --seed 42 \
  --no-theory-align-v2
```

### MC-FD Solver (run_mcfd.py)

```bash
python run/run_mcfd.py --w-h 6.5 --w-l 3.0 --k 0.0004 --sigma1 25.0 --sigma2 25.0
```

### Custom Parameter Sweeps (run_ppo_custom_params.py)

```bash
# Runs predefined parameter combinations
python run/run_ppo_custom_params.py
```

## Dependencies & Contracts

**Depends on:**
- `agents/` - Learning algorithms
- `envs/` - Game environments
- `config/` - Experiment configurations
- `utils/` - Theory, evaluation, logging, plotting

**Provides to system:**
- CLI interface for experiments
- Result files in `results/`
- Convergence history in `results/convergence_history/`

## CLI Arguments (run_two_players.py)

### Core Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | `ppo` | Algorithm: `ppo` or `gradient` |
| `--q` | None | Noise parameter (sweeps q_list if omitted) |
| `--episodes` | from config | Total environment steps |
| `--seed` | 42 | Random seed |

### PPO-Specific (auto-enabled for --method ppo)
| Argument | Default | Description |
|----------|---------|-------------|
| `--theory-align-v2` | `True` | Mean+concentration policy head |
| `--enable-convergence-eval` | `True` | Early stopping on convergence |
| `--cheap-gate-profile` | `relaxed` | KL threshold profile |

### Override Flags
| Argument | Description |
|----------|-------------|
| `--no-theory-align-v2` | Disable theory alignment |
| `--no-convergence-eval` | Disable early stopping |

## Output Files

| Output | Location |
|--------|----------|
| Convergence JSON | `results/convergence_history/{method}_q{q}_seed{seed}_{ablation}_convergence.json` |
| Metadata JSON | `results/convergence_history/*_metadata.json` |
| Results CSV | `results/one_stage_two_players_v2.csv` |
| Training logs | Printed to console (or results/logs/ if enabled) |

## Gotchas / Conventions

- PPO defaults to modern config (selfplay, theory-align-v2, convergence-eval)
- Use `--no-*` flags to disable defaults
- `--episodes` should be multiple of `steps_per_update` (default 16384)
- Gradient method uses deterministic optimization (no seed needed)
- Results are appended to CSV, not overwritten

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Added README.md for directory documentation |
