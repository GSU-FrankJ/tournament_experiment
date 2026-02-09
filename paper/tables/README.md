# paper_out/tables/

## Purpose

Publication-ready tables for the TEL-PPO research paper. Each table is provided in both CSV (data) and LaTeX (.tex) formats.

## Key Contents

| File | Description |
|------|-------------|
| `summary_metrics.*` | Summary of convergence metrics across all experiments |
| `ablation_results.*` | Results from ablation studies (cheap-gate, exploitability) |
| `final_summary.*` | Final summary table for paper |
| `convergence_comparison.*` | Comparison of convergence across methods/parameters |

## Entry Points / How to Use

**Generated automatically** - do not edit manually:

```bash
# Regenerate all tables
python -m paper_artifacts make_all

# Regenerate specific table
python -m paper_artifacts table summary_metrics --out-dir paper_out/tables
```

## Dependencies & Contracts

**Depends on:**
- `paper_artifacts/tables.py` - Table generation module
- `results/convergence_history/` - Source data

**Provides to system:** Publication-ready tables for papers

## File Formats

- **CSV**: Raw data for analysis or custom formatting
- **TEX**: LaTeX-formatted tables ready for `\input{}` in papers

## Quality Classification (used in tables)

| Quality | Gap from e* |
|---------|-------------|
| Excellent | < 0.5 |
| Good | < 1.0 |
| Fair | < 5.0 |
| Poor | ≥ 5.0 |

## Gotchas / Conventions

- Tables auto-generated - manual edits will be overwritten
- LaTeX tables use booktabs style
- CSV files use standard pandas format

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Added README.md for directory documentation |
