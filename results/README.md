# results/

## Purpose

Experiment output directory containing raw results data, convergence history files, training logs, ablation study outputs, and quick-analysis plots. This is the primary output location for all training runs across every experiment track.

## Directory Structure

```
results/
├── two_players/                    # Symmetric 2-player experiment track
│   ├── convergence/                #   Convergence JSON files (168 files)
│   ├── logs/                       #   Training log files (63 files)
│   ├── summary.csv                 #   Aggregated results summary
│   └── summary_legacy.csv          #   Legacy summary (pre-reorganization)
├── three_players/                  # 3-player symmetric experiment track
│   ├── convergence/                #   Convergence JSON files (2 files)
│   └── logs/                       #   Training log files (70 files)
├── different_ability/              # Different-ability experiment track
│   ├── convergence/                #   Convergence JSON files (20 files)
│   ├── logs/                       #   Training log files (26 files)
│   └── summary.csv                 #   Aggregated results summary
├── different_cost/                 # Asymmetric-cost experiment track
│   ├── convergence/                #   Convergence JSON files (73 files)
│   ├── logs/                       #   Training log files (72 files)
│   └── summary.csv                 #   Aggregated results summary
├── ablation/                       # Ablation studies
│   ├── exploit_params/runs/        #   Exploit-parameter ablation runs (234 files)
│   └── mechanism/runs/             #   Mechanism ablation runs (empty)
├── exploit_ablation/               # Standalone exploit ablation sweep
│   └── runs/                       #   Per-run JSON results (234 files)
├── convergence_history/            # Legacy flat convergence JSONs (264 files)
├── convergence_plots/              # Generated convergence plots (36 files)
│   ├── gradient/                   #   Gradient method plots
│   ├── ppo/                        #   PPO method plots
│   └── k5e4_wh8_wl3/              #   Detailed diagnostic plots
├── plots/                          # Additional analysis plots (40 files)
│   ├── gradient/                   #   Gradient method plots
│   ├── ppo/                        #   PPO method plots
│   └── k5e4_wh8_wl3/              #   Detailed diagnostic plots
├── logs/                           # Legacy flat training logs (231 files)
├── one_stage_two_players.csv       # Legacy 2-player results CSV
├── one_stage_two_players_v2.csv    # Current 2-player results CSV (full metrics)
├── different_ability_two_players.csv  # Different-ability results CSV
├── different_cost_two_players.csv  # Asymmetric-cost results CSV
├── convergence_comparison.png      # Cross-method comparison plot
├── convergence_separate_agents.png # Per-agent convergence plot
├── one_stage_two_players.png       # 2-player summary plot
└── exploit_ablation_sweep.log      # Ablation sweep execution log
```

**Total files:** ~1,546

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
- `run/run_two_players.py` - Generates `two_players/`, `convergence_history/`, `logs/`, top-level CSVs
- `run/run_three_players.py` - Generates `three_players/`
- `run/run_different_ability.py` - Generates `different_ability/`, `different_ability_two_players.csv`
- `run/run_different_cost.py` - Generates `different_cost/`, `different_cost_two_players.csv`
- `tools/plot_convergence.py` - Generates `convergence_plots/`, `plots/`
- `tools/plot_convergence_detailed.py` - Generates detailed diagnostic plots

**Provides to system:**
- Raw experiment data for analysis (`**/convergence/*.json`)
- Summary CSVs for paper generation (`*.csv`, `**/summary.csv`)
- Convergence history for figure generation (`convergence_history/`)
- Ablation study results for robustness analysis (`ablation/`, `exploit_ablation/`)
- Quick-look plots for debugging (`convergence_plots/`, `plots/`)

## Data Flow

```
run/run_two_players.py
    → results/two_players/convergence/*_convergence.json
    → results/two_players/logs/*.log
    → results/two_players/summary.csv
    → results/convergence_history/*_convergence.json  (legacy flat copy)
    → results/logs/*.log                              (legacy flat copy)
    → results/one_stage_two_players_v2.csv

run/run_three_players.py
    → results/three_players/convergence/*_convergence.json
    → results/three_players/logs/*.log

run/run_different_ability.py
    → results/different_ability/convergence/*_convergence.json
    → results/different_ability/logs/*.log
    → results/different_ability/summary.csv
    → results/different_ability_two_players.csv

run/run_different_cost.py
    → results/different_cost/convergence/*_convergence.json
    → results/different_cost/logs/*.log
    → results/different_cost/summary.csv
    → results/different_cost_two_players.csv

tools/plot_convergence.py
    → results/convergence_plots/{gradient,ppo,k5e4_wh8_wl3}/*.png
    → results/plots/{gradient,ppo,k5e4_wh8_wl3}/*.png

paper/generator/
    reads: results/**  →  paper/figures/, paper/tables/
```

## CSV Formats

### one_stage_two_players_v2.csv (symmetric 2-player)
Standard columns: `method`, `q`, `seed`, `ablation_name`, `theoretical_effort`, `final_effort`, `gap`, `quality`, `convergence_step`, `total_steps`, plus PPO hyperparameters.

### different_ability_two_players.csv / different_cost_two_players.csv
Per-player columns with theoretical and learned efforts for each player (asymmetric equilibria).

### summary.csv (per experiment subdirectory)
Compact per-experiment summary with key metrics for quick aggregation.

## Naming Conventions

| Pattern | Example | Meaning |
|---------|---------|---------|
| `{method}_q{q}_convergence.json` | `gradient_q25.0_convergence.json` | Gradient baseline, q=25 |
| `ppo_q{q}_seed{s}_convergence.json` | `ppo_q40.0_seed42_convergence.json` | PPO baseline run |
| `ppo_q{q}_seed{s}_eps_{v}_convergence.json` | `ppo_q25.0_seed42_eps_003_convergence.json` | PPO with clip epsilon ablation |
| `ppo_q{q}_seed{s}_pat_{v}_convergence.json` | `ppo_q25.0_seed42_pat_01_convergence.json` | PPO with patience ablation |
| `*_metadata.json` | `ppo_q25.0_seed42_metadata.json` | Hyperparameters & run config |
| `{track}_ppo_q{q}_seed{s}_baseline_convergence.json` | `different_ability_ppo_q25.0_seed42_baseline_convergence.json` | Track-specific PPO baseline |

## Gotchas / Conventions

- Legacy files in `convergence_history/` and `logs/` duplicate data now also stored in organized subdirectories (`two_players/`, etc.)
- Legacy files without seed/ablation in name default to seed=42, ablation="baseline"
- Plots are organized by algorithm/ablation in subdirectories
- Do not manually edit CSV/JSON - regenerate via run scripts
- `ablation/mechanism/runs/` exists but is currently empty (placeholder)
- `exploit_ablation/` is a standalone sweep separate from `ablation/exploit_params/` (both contain 234 run files)

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-09 | Updated README to reflect full directory structure with all experiment tracks, ablation studies, and current file counts |
| 2026-02-03 | Added README.md; moved plot scripts to tools/ |
