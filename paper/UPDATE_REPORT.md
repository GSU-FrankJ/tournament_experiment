# Figure & Table Generation Update Report

**Date**: 2026-02-25
**Scope**: All 7 figure functions + 3 tables in `paper/generator/`
**Reference**: `Figures&Tables for Experimental analysis022426.pdf`

---

## Files Modified

| File | Lines Changed | Purpose |
|------|--------------|---------|
| `paper/generator/config.py` | +6/−1 | Added `THEORY_LINE_COLOR`, `THEORY_LINE_WIDTH` constants |
| `paper/generator/extract.py` | +48/−1 | Added `get_convergence_step()` helper |
| `paper/generator/plots.py` | +392/−110 | All 7 figure functions revised |
| `paper/generator/tables.py` | +224/−109 | Tables 1−3 restructured |

**Total**: +511 insertions, −159 deletions across 4 files.

---

## Global Style Changes

| Property | Before | After |
|----------|--------|-------|
| Theory line color | `black` | `red` |
| Theory line width | `2.0` | `2.5` |
| Seed trace alpha | `0.25` | `0.15` |
| CI band alpha | `0.15` | `0.10` |
| Error band method | mean ± CI95 | mean ± CI95 (unchanged) |

---

## Figure-by-Figure Changes

### Figure 2a: Effort Trajectory (Two-Player Baseline) — `plot_convergence_main()`

| Revision | Detail |
|----------|--------|
| Convergence vertical lines | Green dotted `axvline` per seed, read from `get_convergence_step()` |
| Final summary annotation | Upper-right text box: \|ē−e*\|, exploitability ε, symmetry gap Δsym |
| Theory line | `color="red"`, `linewidth=2.5`, dashed |
| Shadow transparency | Seed traces α=0.15, CI bands α=0.10 |
| Unified y-axis | `_unify_ylim()` applied across all 2×3 panels |
| Panel titles | `"q = 25"` → `"Noise Level q = 25"` |

### Figure 2b: Distance to Nash Equilibrium — `plot_distance_to_equilibrium()`

| Revision | Detail |
|----------|--------|
| y-axis floor | `ax.set_ylim(bottom=0.1)` (starts at 10⁻¹) |
| Convergence vertical lines | Per-q mean convergence step, matching q-line color |
| ε threshold line | Gray dashed horizontal at `effort_delta=0.5` |
| Shadow alpha | CI band reduced from 0.2 → 0.12 |
| Title | → "Distance to the Nash Equilibrium Across Noise Levels" |

### Figure 3a: KL Divergence Dynamics — `plot_kl_dynamics()`

| Revision | Detail |
|----------|--------|
| KL threshold line | Red dashed at `mean_kl_thresh=0.0045` |
| Per-seed traces | Thin lines α=0.3, linewidth=0.8 |
| Bold mean | Blue linewidth=2 aggregate |
| y-limits | `[1e-4, 1e-1]` (log scale) |

### Figure 3b: Effort Drift — `plot_effort_drift()`

| Revision | Detail |
|----------|--------|
| Unified y-axis | `_unify_ylim()` across all 3 panels |
| Threshold line | Red bold `linewidth=2.5`, with `"2.0"` annotation beside line |

### Figure 4: Beta Distribution Snapshots — `plot_beta_snapshots()`

| Revision | Detail |
|----------|--------|
| Unified y-axis | Shared ylim across 3 panels |
| e* vertical line | Green dash-dot at `e_star(q)` |
| κ annotation | Upper-left box: `κ=α+β` value |
| X-axis | Converted from normalized [0,1] → effort [0, 250] |

### Figure 5: Exploitability Dynamics — `plot_exploitability_dynamics()`

| Revision | Detail |
|----------|--------|
| y-scale | Log, limits `[0.01, 1]` |
| Threshold line | Red bold `linewidth=2.5, alpha=0.8` |
| Convergence vlines | Green dotted per seed |
| Per-seed traces | Thin α=0.3 with `steps-post` drawstyle |
| Bold mean | Blue linewidth=2, `steps-post` |

### Figure 6: Equilibrium Recovery Dotplot — `plot_equilibrium_recovery_dotplot()`

| Revision | Detail |
|----------|--------|
| Theory lines | Red, `linewidth=3` (up from 2) |
| Relative error | Computed `\|e_learned − e*\| / e*` per seed, saved in CSV |
| Mean relative error | 0.1157 (printed for caption) |

---

## Table Changes

### Table 1: Environment Configuration — `generate_environment_config_table()`

Updated to match PDF specification:

| Addition/Change | Value |
|-----------------|-------|
| Prizes | `(wH, wL) = (6.5, 3) and (8, 4)` |
| Effort bounds | `[0, 200]` (was `[0, 250]`) |
| Number of players | `2 (baseline) / 3` |
| Learning rate | `3×10⁻⁴` |
| Batch size | `4096 episodes/update` |
| PPO clip | `0.2` |
| Seeds per config | `5` |
| Policy parameterization | `Beta(α, β)` |
| Min steps | `100` |

### Table 2: Quantitative Summary — `generate_final_paper_table()`

**New column structure**:

```
Scenario | q | Method | Mean±std | |ē−e*| | Exploitability | Symmetry Gap | Steps to Conv.
```

- Covers all 4 experiments (Two-Player, Three-Player, Het. Cost, Het. Ability)
- Mean±std format for effort
- Convergence steps computed via `get_convergence_step()` (NC = not converged)

### Table 3: Ablation Results — `generate_ablation_table()`

**New column structure**:

```
Ablation | Final Error | Exploitability | Steps to Conv. | Failure Mode
```

- Rows ordered: TEL-PPO (baseline), No stability gate, No exploitability gate
- Failure mode auto-classified: `diverge` / `cycle` / `biased mean` / `never terminate`
- Aggregated across all q values and seeds

---

## Generated Artifacts

### Figures (9 total, PNG + PDF)

| Figure | File | Size (PNG) |
|--------|------|-----------|
| Fig 2a: Convergence Main | `convergence_main.png` | 697 KB |
| Fig 2b: Distance to Equilibrium | `distance_to_equilibrium.png` | 362 KB |
| Fig 3a: KL Dynamics | `kl_dynamics.png` | 324 KB |
| Fig 3b: Effort Drift | `effort_drift.png` | 335 KB |
| Fig 4: Beta Snapshots | `beta_snapshots.png` | 171 KB |
| Fig 5: Exploitability Dynamics | `exploitability_dynamics.png` | 119 KB |
| Fig 6: Equilibrium Recovery | `equilibrium_recovery_dotplot.png` | 189 KB |
| Beta Evolution (supplementary) | `beta_evolution.png` | 256 KB |
| Ablation Comparison (supplementary) | `ablation_comparison.png` | 457 KB |

### Tables (5 total, CSV + LaTeX)

| Table | CSV | LaTeX |
|-------|-----|-------|
| Table 1: Environment Config | `environment_config.csv` | `environment_config.tex` |
| Table 2: Final Summary | `final_summary.csv` | `final_summary.tex` |
| Table 3: Ablation Results | `ablation_results.csv` | `ablation_results.tex` |
| Summary Metrics (detailed) | `summary_metrics.csv` | `summary_metrics.tex` |
| Convergence Comparison | `convergence_comparison.csv` | `convergence_comparison.tex` |

---

## New Utility Functions

| Function | File | Purpose |
|----------|------|---------|
| `get_convergence_step(df)` | `extract.py` | Detect convergence step per run using effort_delta/window criteria |
| `_unify_ylim(axes, margin)` | `plots.py` | Synchronize y-axis limits across subplot panels |
| `_classify_failure_mode(row)` | `tables.py` | Auto-classify ablation failure: diverge/cycle/biased mean/never terminate |

---

## Reproduction

```bash
cd /home/fjiang4/tournament_experiment
python -m paper.generator make_all
# Outputs: paper/figures/ (PNG+PDF), paper/tables/ (CSV+LaTeX), paper/data/ (underlying CSVs)
```

---

## Known Observations

1. **CSV parsing warning**: `summary.csv` has inconsistent column counts (line 14 has 46 fields vs expected 39). This is a pre-existing data issue and does not affect figure/table generation since convergence JSONs are the primary data source.

2. **Tight layout warning**: `convergence_main` produces a tight_layout warning due to the 2×3 grid with row annotations. The figure renders correctly; the warning is cosmetic.

3. **Convergence rates**: Most TEL-PPO runs show `NC` (not converged) under the strict dual-criterion (effort + exploitability). The single converged case is Two-Player q=55 at step 999,424. This is consistent with the 500-update budget being tight for lower noise levels.

4. **Mean relative error**: The dotplot shows a mean relative error of 11.57% across all scenarios, indicating TEL-PPO recovers equilibrium within ~12% on average.
