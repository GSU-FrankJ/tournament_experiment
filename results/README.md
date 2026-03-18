# results/

## Purpose

Experiment output directory containing convergence history files, training logs, ablation study outputs, and summary CSVs. This is the primary output location for all training runs across every experiment track.

## Directory Structure

```
results/
├── two_players/                    # Symmetric 2-player experiment track
│   ├── convergence/                #   Convergence + metadata JSONs (221 files)
│   ├── logs/                       #   Training log files (191 files)
│   └── summary.csv                 #   Aggregated results summary
├── three_players/                  # 3-player symmetric experiment track
│   ├── convergence/                #   Convergence JSONs (14 files)
│   └── logs/                       #   Training log files (118 files)
├── different_ability/              # Different-ability experiment track
│   ├── convergence/                #   Convergence JSONs (16 files)
│   ├── logs/                       #   Training log files (118 files)
│   └── summary.csv                 #   Aggregated results summary
├── different_cost/                 # Asymmetric-cost experiment track
│   ├── convergence/                #   Convergence JSONs (16 files)
│   ├── logs/                       #   Training log files (117 files)
│   └── summary.csv                 #   Aggregated results summary
├── ablation/                       # Ablation studies
│   ├── exploit_params/runs/        #   Exploit-parameter ablation runs (234 files)
│   └── mechanism/runs/             #   Mechanism ablation runs (empty)
├── exploit_ablation/               # Standalone exploit ablation sweep
│   ├── runs/                       #   Per-run JSON results (291 files)
│   ├── summary.csv                 #   Ablation summary
│   ├── summary.json                #   Ablation summary (JSON)
│   └── RECOMMENDATION.md           #   Per-setting recommendations
├── logs/                           # Legacy flat training logs (231 files)
└── README.md
```

**Total files:** ~1,378

## Entry Points / How to Use

**Generated automatically** by run scripts:

```bash
# Symmetric 2-player experiments
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 42

# 3-player experiments
python run/run_three_players.py --method ppo --q 40 --episodes 2048000 --seed 42

# Different-ability experiments
python run/run_different_ability.py --method both --q 25 --episodes 131072

# Asymmetric-cost experiments
python run/run_different_cost.py --method both --q 25 --episodes 131072

# Generate convergence plots from history
python tools/plot_convergence.py
python tools/plot_convergence_detailed.py

# Generate paper figures/tables from results
python -m paper.generator make_all
```

## Dependencies & Contracts

**Depends on:**
- `run/run_two_players.py` - Generates `two_players/`
- `run/run_three_players.py` - Generates `three_players/`
- `run/run_different_ability.py` - Generates `different_ability/`
- `run/run_different_cost.py` - Generates `different_cost/`
- `tools/plot_convergence.py` - Generates convergence plots
- `tools/plot_convergence_detailed.py` - Generates detailed diagnostic plots

**Provides to system:**
- Raw experiment data for analysis (`**/convergence/*.json`)
- Summary CSVs for paper generation (`**/summary.csv`)
- Ablation study results for robustness analysis (`ablation/`, `exploit_ablation/`)

## Data Flow

```
run/run_two_players.py
    → results/two_players/convergence/*_convergence.json
    → results/two_players/logs/*.log
    → results/two_players/summary.csv

run/run_three_players.py
    → results/three_players/convergence/*_convergence.json
    → results/three_players/logs/*.log

run/run_different_ability.py
    → results/different_ability/convergence/*_convergence.json
    → results/different_ability/logs/*.log
    → results/different_ability/summary.csv

run/run_different_cost.py
    → results/different_cost/convergence/*_convergence.json
    → results/different_cost/logs/*.log
    → results/different_cost/summary.csv

paper/generator/
    reads: results/**  →  paper/figures/, paper/tables/
```

## CSV Formats

### summary.csv (per experiment subdirectory)
Compact per-experiment summary with key metrics for quick aggregation. Present in `two_players/`, `different_ability/`, and `different_cost/`.

## Naming Conventions

| Pattern | Example | Meaning |
|---------|---------|---------|
| `{method}_q{q}_convergence.json` | `gradient_q25.0_convergence.json` | Gradient baseline, q=25 |
| `ppo_q{q}_seed{s}_convergence.json` | `ppo_q40.0_seed42_convergence.json` | PPO baseline run |
| `ppo_q{q}_seed{s}_eps_{v}_convergence.json` | `ppo_q25.0_seed42_eps_003_convergence.json` | PPO with clip epsilon ablation |
| `ppo_q{q}_seed{s}_pat_{v}_convergence.json` | `ppo_q25.0_seed42_pat_01_convergence.json` | PPO with patience ablation |
| `ppo_q{q}_seed{s}_no_{mech}_convergence.json` | `ppo_q25.0_seed42_no_cheap_gate_convergence.json` | PPO with mechanism disabled |
| `ppo_q{q}_seed{s}_wh{w}_wl{w}_convergence.json` | `ppo_q25.0_seed42_wh8_wl4_convergence.json` | PPO with alternate wage params |
| `*_metadata.json` | `ppo_q25.0_seed42_metadata.json` | Hyperparameters & run config |
| `{track}_ppo_q{q}_seed{s}_baseline_convergence.json` | `different_ability_ppo_q25.0_seed42_baseline_convergence.json` | Track-specific PPO baseline |
| `ppo_3p_q{q}_seed{s}_baseline_convergence.json` | `ppo_3p_q40.0_seed42_baseline_convergence.json` | 3-player PPO baseline |
| `gradient_3p_q{q}_convergence.json` | `gradient_3p_q25.0_convergence.json` | 3-player gradient baseline |

## Gotchas / Conventions

- Legacy flat logs in `logs/` duplicate data also stored in per-experiment `*/logs/` subdirectories
- Legacy files without seed/ablation in name default to seed=42, ablation="baseline"
- Do not manually edit CSV/JSON - regenerate via run scripts
- `ablation/mechanism/runs/` exists but is currently empty (placeholder)
- `exploit_ablation/` is a standalone sweep separate from `ablation/exploit_params/` (291 vs 234 run files; exploit_ablation includes additional runs)
- `three_players/` has no `summary.csv`

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-03-18 | Updated README to reflect current structure: removed deleted legacy dirs (convergence_history, convergence_plots, plots, archive), removed top-level CSVs/PNGs, updated file counts |
| 2026-02-09 | Updated README to reflect full directory structure with all experiment tracks, ablation studies, and current file counts |
| 2026-02-03 | Added README.md; moved plot scripts to tools/ |
