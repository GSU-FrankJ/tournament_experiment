# paper/

## Purpose

Output directory and generation pipeline for publication-ready artifacts for the TEL-PPO research paper. Contains the `generator/` Python package, plus generated figures, tables, and underlying data.

## Key Contents

| Path | Description |
|------|-------------|
| `generator/` | Python package — figure/table generation pipeline (`python -m paper.generator`) |
| `figures/` | Publication figures in PNG (300 DPI) and PDF (vector) formats |
| `tables/` | LaTeX tables (`.tex`) and CSV data files |
| `data/` | Underlying CSV data used to generate figures |

## Entry Points / How to Use

**Generated automatically** — do not edit output files manually.

```bash
# Generate all paper artifacts (figures + tables)
python -m paper.generator make_all

# Preview discovered runs without generating
python -m paper.generator --dry-run

# Generate a specific figure
python -m paper.generator plot convergence_main
python -m paper.generator plot kl_dynamics
python -m paper.generator plot exploitability_dynamics
python -m paper.generator plot beta_evolution
python -m paper.generator plot beta_snapshots
python -m paper.generator plot ablation_comparison

# Generate a specific table
python -m paper.generator table summary_metrics
python -m paper.generator table ablation_results
python -m paper.generator table final_summary
python -m paper.generator table convergence_comparison
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--runs-dir` | scan all `results/*/convergence/` | Path to convergence directory |
| `--csv` | `results/two_players/summary.csv` | Path to results CSV |
| `--out-dir` | `paper/` | Output directory |
| `--q` | per-experiment (e.g. `35,45,55`) | Comma-separated q values |
| `--dry-run` | — | List discovered runs without generating |

## Dependencies & Contracts

**Depends on:**
- `results/*/convergence/` — Source convergence JSON data (two_players, three_players, different_cost, different_ability, ablation)
- `results/two_players/summary.csv` — Summary CSV for table generation
- `utils/theory.py` — Theoretical equilibrium formulas

**Provides to system:**
- Publication-ready figures for papers (PNG + PDF)
- LaTeX-compatible tables (`.tex` + `.csv`)
- Reproducible data exports

## Generator Modules

| Module | Purpose |
|--------|---------|
| `generator/__main__.py` | CLI entry point (argparse dispatcher) |
| `generator/__init__.py` | Package init, exports `RESULTS_DIR`, `OUTPUT_DIR`, `Q_VALUES`, `THEORY_PARAMS` |
| `generator/config.py` | Paths, per-experiment theory params (`EXPERIMENT_THEORY_PARAMS`), plot styles, quality thresholds |
| `generator/extract.py` | Data loading — JSON→DataFrame, multi-run aggregation, seed aggregation, forward-fill |
| `generator/metrics.py` | Convergence detection, cheap-gate stats, summary metrics computation |
| `generator/plots.py` | All figure generation (matplotlib, publication style, PNG+PDF output) |
| `generator/tables.py` | All table generation (DataFrame→CSV + LaTeX) |
| `generator/run_registry.py` | Run discovery — filename parsing, metadata lookup, filtering, grouping |

## Output Structure

```
paper/
├── generator/              # Generation pipeline (Python package)
│   ├── __init__.py
│   ├── __main__.py         # CLI: python -m paper.generator
│   ├── config.py           # Paths, theory params, plot styles
│   ├── extract.py          # Data loading & aggregation
│   ├── metrics.py          # Convergence detection & metrics
│   ├── plots.py            # Figure generation
│   ├── tables.py           # Table generation
│   ├── run_registry.py     # Run discovery & filtering
│   └── README.md
├── figures/
│   ├── convergence_main.png/pdf    # Main convergence (1×3 grid by q)
│   ├── kl_dynamics.png/pdf         # KL divergence over training
│   ├── exploitability_dynamics.png/pdf  # Exploitability over training
│   ├── beta_evolution.png/pdf      # Alpha/beta parameter evolution (2×3 grid)
│   ├── beta_snapshots.png/pdf      # Beta distribution snapshots
│   ├── ablation_comparison.png/pdf # Ablation study comparison
│   └── README.md
├── tables/
│   ├── summary_metrics.csv/tex     # Summary metrics for all runs
│   ├── ablation_results.csv/tex    # Ablation study results
│   ├── final_summary.csv/tex       # Final paper comparison table
│   ├── convergence_comparison.csv/tex  # Convergence comparison across methods
│   └── README.md
├── data/
│   ├── convergence_main.csv        # Main convergence data
│   ├── kl_dynamics.csv             # KL divergence data
│   ├── exploitability_dynamics.csv # Exploitability data
│   ├── beta_evolution.csv          # Beta parameter evolution data
│   ├── ablation_comparison.csv     # Ablation comparison data
│   └── README.md
└── README.md
```

## Gotchas / Conventions

- **Do not edit generated files directly** — regenerate via `python -m paper.generator`
- PNG files are 300 DPI for publication quality
- PDF files are vector format for LaTeX inclusion
- Tables are provided in both CSV (data) and LaTeX (formatted) formats
- Run discovery supports both legacy (`ppo_q40.0_convergence.json`) and new (`ppo_q40.0_seed42_baseline_convergence.json`) filename formats
- Sparse exploitability values are forward-filled during extraction
- Quality classification: Excellent (<0.5 gap), Good (<1.0), Fair (<5.0), Poor (≥5.0)
- Cheap-gate thresholds: `mean_kl < 0.0045`, `std_kl < 0.0035`, `drift < 2.0`

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-09 | Updated README to reflect current `paper/generator/` pipeline, corrected commands and structure |
| 2026-02-03 | Added README.md for directory documentation |
