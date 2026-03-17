
# Results Folder Guide

## Overview

The `results/` directory stores all experiment outputs: convergence histories, training logs, summary CSVs, ablation studies, and diagnostic plots. Each experiment has its own subdirectory with a consistent internal structure.

## Directory Structure

```
results/
├── two_players/                    # Symmetric 2-player tournament
│   ├── convergence/                # 221 files (115 convergence + 106 metadata)
│   ├── logs/                       # Raw training logs
│   └── summary.csv                 # Aggregated results table
├── three_players/                  # Symmetric 3-player tournament
│   ├── convergence/                # 14 files
│   └── logs/
├── different_cost/                 # Asymmetric cost (k1 != k2)
│   ├── convergence/                # 16 files
│   ├── logs/
│   └── summary.csv
├── different_ability/              # Asymmetric ability (l1 > l2)
│   ├── convergence/                # 16 files
│   ├── logs/
│   └── summary.csv
├── ablation/                       # Ablation studies
│   ├── mechanism/runs/             # Toggle cheap-gate / exploit / entropy
│   └── exploit_params/runs/        # Sweep eps / patience parameters
```

## Convergence JSON Files

These are the primary data files. Two formats exist:

### Flat Format (two_players, three_players)

Filename pattern: `{method}_q{q}[_seed{seed}][_{ablation}]_convergence.json`

```json
{
  "steps": [4096, 8192, ...],
  "agent1_effort": [50.2, 55.1, ...],
  "agent2_effort": [49.8, 54.9, ...],
  "policy_mean_effort": [50.0, 55.0, ...],
  "approx_kl": [0.01, 0.008, ...],
  "batch_entropy": [-2.1, -2.3, ...],
  "alpha_mean": [2.5, 3.0, ...],
  "beta_mean": [2.5, 3.0, ...],
  "exploitability": [5.0, null, null, 3.2, ...],
  "exploitability_is_valid": [true, false, false, true, ...],
  "theoretical": {"effort": 87.5},
  "seed": 42,
  "ablation_name": "baseline",
  "q": 25.0
}
```

### Nested Format (different_cost, different_ability)

Filename pattern: `{experiment}_{method}_q{q}_seed{seed}[_{ablation}]_convergence.json`

```json
{
  "scenario": "different_cost",
  "q": 25.0,
  "seed": 42,
  "theoretical": {"effort1": 59.23, "effort2": 43.08},
  "history": {
    "steps": [4096, 8192, ...],
    "agent1_effort": [30.0, 35.2, ...],
    "agent2_effort": [28.5, 33.1, ...],
    "approx_kl_agent1": [0.01, ...],
    "approx_kl_agent2": [0.009, ...],
    "batch_entropy_agent1": [-2.0, ...],
    "batch_entropy_agent2": [-1.9, ...]
  },
  "exploit_history": [
    {"update": 100, "exploit_max": 3.5},
    {"update": 200, "exploit_max": 1.2}
  ],
  "final_exploit_max": 0.8
}
```

### Metadata JSON Files

Companion `*_metadata.json` files store hyperparameters and run configuration:

```json
{
  "seed": 42,
  "ablation_name": "baseline",
  "q": 25.0,
  "steps_per_update": 4096,
  "max_updates": 1500,
  "entropy_coef_end": 0.005
}
```

## Filename Convention

```
{prefix}_{method}_q{q}_seed{seed}_{ablation}_convergence.json
```

| Component | Values | Examples |
|-----------|--------|---------|
| prefix | (none) for two_players, `3p` for three_players, `different_cost`, `different_ability` | `ppo_q25.0_...`, `ppo_3p_q25.0_...`, `different_cost_ppo_q25.0_...` |
| method | `ppo`, `gradient` | |
| q | `25.0`, `40.0`, `55.0` | |
| seed | `42`, `50`, `68`, `123`, `456`, `789`, `1024` | |
| ablation | `baseline`, `eps_001`, `eps_003`, `eps_010`, `eps_020`, `pat_01`, `pat_03`, `pat_10`, `no_cheap_gate`, `no_exploitability` | |

## Ablation Naming

| Ablation | Meaning |
|----------|---------|
| `baseline` | Default hyperparameters |
| `eps_001` / `eps_003` / `eps_010` / `eps_020` | Exploitability epsilon sweep |
| `pat_01` / `pat_03` / `pat_10` | Exploitability patience sweep |
| `no_cheap_gate` | Cheap-gate mechanism disabled |
| `no_exploitability` | Exploitability check disabled |
| `wh8_wl4` | Alternative prize weights (w_H=8, w_L=4) |

## Summary CSV Files

Per-experiment `summary.csv` files aggregate final results. Typical columns:

```
method, q, seed, ablation_name, theoretical_effort, final_effort_mean,
agent1_effort, agent2_effort, abs_error, rel_error_pct, max_updates,
converged, convergence_step
```

## Experiment Parameters

| Experiment | Players | Cost | Ability | Theoretical e* (q=25) |
|---|---|---|---|---|
| two_players | 2 symmetric | k=0.0004 | equal | 87.50 |
| three_players | 3 symmetric | k=0.0004 | equal | 87.50 |
| different_cost | 2 asymmetric | k1=0.0004, k2=0.0006 | equal | e1*=59.23, e2*=43.08 |
| different_ability | 2 asymmetric | k=0.0004 | l1=10, l2=5 | 78.75 (shared) |

Common: w_H=6.5, w_L=3.0, q in {25, 40, 55}, effort bounds [0, 200].

## How to Use

### Load data programmatically

```python
# Load all runs across experiments
from paper.generator.run_registry import discover_runs
from paper.generator.extract import load_multiple_runs, get_final_values

runs = discover_runs()  # Scans all 4 convergence dirs
df = load_multiple_runs(runs)
final = get_final_values(df)
```

### Select best run per experiment x q

```python
from paper.generator.run_registry import discover_runs, select_best_runs

runs = discover_runs()
best = select_best_runs(runs)  # Lowest error per (experiment, q)
```

### Generate paper artifacts

```bash
# All figures and tables (all seeds)
python -m paper.generator make_all

# Best-only (one seed per experiment x q)
python -m paper.generator --best-only make_all

# Dry run (list discovered runs)
python -m paper.generator --dry-run
```

