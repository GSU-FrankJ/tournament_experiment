# paper_out/figures/

## Purpose

Publication-ready figures for the TEL-PPO research paper. Each figure is generated in both PNG (for preview) and PDF (for LaTeX) formats.

## Key Contents

| File | Description |
|------|-------------|
| `convergence_main.*` | Main convergence plot showing effort trajectories toward e* |
| `kl_dynamics.*` | KL divergence over training (stability metric) |
| `exploitability_dynamics.*` | Exploitability measurements (equilibrium quality) |
| `beta_evolution.*` | Evolution of Beta distribution parameters |
| `beta_snapshots.*` | Snapshots of Beta distribution at key training points |
| `ablation_comparison.*` | Comparison across ablation variants |

## Entry Points / How to Use

**Generated automatically** - do not edit manually:

```bash
# Regenerate all figures
python -m paper_artifacts make_all

# Regenerate specific figure
python -m paper_artifacts plot convergence_main --out-dir paper_out/figures
```

## Dependencies & Contracts

**Depends on:**
- `paper_artifacts/plots.py` - Plotting module
- `paper_out/data/` - Source data

**Provides to system:** Publication-ready figures for papers

## Figure Specifications

- **Resolution**: 300 DPI (PNG)
- **Format**: PNG (preview) + PDF (LaTeX)
- **Style**: matplotlib default with custom colors
- **Size**: Optimized for single-column paper width

## Gotchas / Conventions

- PNG files for quick preview, PDF for paper inclusion
- Figures auto-generated - manual edits will be overwritten
- See `paper_artifacts/config.py` for style settings

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Added README.md for directory documentation |
