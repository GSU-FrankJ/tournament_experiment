# results/convergence_plots/

## Purpose

Quick-analysis convergence plots generated from training history. Organized by algorithm and ablation variant for easy comparison.

## Key Contents

| Path | Description |
|------|-------------|
| `gradient/` | Convergence plots for gradient solver |
| `ppo/` | Convergence plots for PPO agent |
| `k5e4_wh8_wl3/` | Plots for specific parameter ablation (k=0.0005, w_h=8, w_l=3) |

## Plot Types

| Pattern | Description |
|---------|-------------|
| `q{q}_combined.png` | Both agents on single plot |
| `q{q}_separated.png` | Separate subplot per agent |
| `q{q}_1_effort_comparison.png` | Effort comparison plot |
| `q{q}_2_agent_efforts.png` | Individual agent efforts |
| `q{q}_3_gap_comparison.png` | Gap from theoretical |
| `q{q}_4_kl_divergence.png` | KL divergence over training |
| `q{q}_5_alpha_beta.png` | Beta distribution parameters |
| `q{q}_6_concentration.png` | Policy concentration |

## Entry Points / How to Use

**Generated automatically** - do not edit manually:

```bash
# Generate all plots
python tools/plot_convergence_detailed.py

# Generate for specific algorithm
python tools/plot_convergence_detailed.py --algorithm PPO

# Generate for specific q value
python tools/plot_convergence_detailed.py --q 25.0
```

## Dependencies & Contracts

**Depends on:**
- `tools/plot_convergence_detailed.py` - Plotting script
- `results/convergence_history/` - Source data

**Provides to system:** Quick visual analysis of training runs

## Gotchas / Conventions

- Plots are organized by algorithm/ablation in subdirectories
- Files are overwritten when regenerating
- For publication figures, use `paper_out/figures/` instead
- Combined plots show both agents; separated plots show each agent's trajectory

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Added README.md for directory documentation |
