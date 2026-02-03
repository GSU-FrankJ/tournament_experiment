# paper_out/data/

## Purpose

Underlying data files used to generate paper figures. These CSV files contain the processed data that feeds into matplotlib plots.

## Key Contents

| File | Description |
|------|-------------|
| `convergence_main.csv` | Effort convergence data (steps, agent efforts, theoretical values) |
| `kl_dynamics.csv` | KL divergence over training |
| `exploitability_dynamics.csv` | Exploitability measurements over training |
| `beta_evolution.csv` | Beta distribution parameters (alpha, beta) over training |
| `ablation_comparison.csv` | Comparison data across ablation variants |

## Entry Points / How to Use

**Generated automatically** - do not edit manually:

```bash
# Regenerate all data
python -m paper_artifacts make_all --out-dir paper_out
```

To use for custom analysis:

```python
import pandas as pd
df = pd.read_csv("paper_out/data/convergence_main.csv")
```

## Dependencies & Contracts

**Depends on:** `paper_artifacts/extract.py` - Data extraction module

**Provides to system:** Intermediate data for figure generation

## Gotchas / Conventions

- CSV files use standard pandas format
- Column names match metric names from convergence history
- Data is aggregated across seeds where applicable
- NaN values indicate missing exploitability evaluations

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Added README.md for directory documentation |
