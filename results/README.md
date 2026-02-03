# results/

## Purpose

Experiment output directory containing raw results data, convergence history files, and quick-analysis plots. This is the primary output location for training runs.

## Key Contents

| Path | Description |
|------|-------------|
| `convergence_history/` | JSON files with step-by-step training metrics |
| `convergence_plots/` | Quick-analysis plots organized by algorithm/ablation |
| `logs/` | Training log files (if enabled) |
| `one_stage_two_players.csv` | Legacy results CSV |
| `one_stage_two_players_v2.csv` | Current results CSV with full metrics |
| `*.png` | Quick comparison plots |

## Entry Points / How to Use

**Generated automatically** by run scripts:

```bash
# Run experiment (generates results)
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 50

# Generate convergence plots from history
python tools/plot_convergence.py
python tools/plot_convergence_detailed.py
```

## Dependencies & Contracts

**Depends on:**
- `run/` scripts - Generate results during training
- `tools/plot_*.py` - Generate visualization plots

**Provides to system:**
- Raw experiment data for analysis
- Convergence history for paper generation
- Quick-look plots for debugging

## Data Flow

```
run/run_two_players.py
    ↓ (during training)
results/convergence_history/*_convergence.json
results/convergence_history/*_metadata.json
results/one_stage_two_players_v2.csv
    ↓ (via tools/)
results/convergence_plots/
    ↓ (via paper_artifacts/)
paper_out/
```

## CSV Format (one_stage_two_players_v2.csv)

Standard columns:
- `method`, `q`, `seed`, `ablation_name`
- `theoretical_effort`, `final_effort`, `gap`, `quality`
- `convergence_step`, `total_steps`
- PPO hyperparameters

## Gotchas / Conventions

- JSON files use naming convention: `{method}_q{q}_seed{seed}_{ablation}_convergence.json`
- Legacy files (without seed/ablation in name) default to seed=42, ablation="baseline"
- Plots are organized by algorithm/ablation in subdirectories
- Do not manually edit CSV/JSON - regenerate via run scripts

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Added README.md; moved plot scripts to tools/ |
