# TEL-PPO Paper Artifacts Pipeline

This module generates all figures, tables, and underlying data for the TEL-PPO research paper.

## Quick Start

```bash
# Dry-run: list discovered runs
python -m paper_artifacts.make_all --dry-run

# Generate all artifacts
python -m paper_artifacts.make_all \
    --runs-dir results/convergence_history \
    --csv results/one_stage_two_players_v2.csv \
    --out-dir paper_out \
    --q 25,40,55

# Generate single figure
python -m paper_artifacts.plots convergence_main --out-dir paper_out/figures
```

## Definitions

### Theoretical Equilibrium Effort (e*)

The Nash equilibrium effort for the symmetric two-player tournament is:

```
e* = (w_H - w_L) / (4 * q * k)
```

Where:
- `w_H = 6.5` (high prize)
- `w_L = 3.0` (low prize)
- `k = 0.0004` (cost coefficient)
- `q` = noise parameter (Uniform(-q, q) noise on performance)

Reference: `utils/theory.py:e_star_two_players()`

### Exploitability (ε)

Exploitability measures how much an agent can gain by deviating from the learned policy:

```
ε = max_{e_dev} E[u(e_dev, π)] - E[u(π, π)]
```

Where:
- `u(e, π)` = expected utility of playing effort `e` against opponent policy `π`
- The expectation is over game noise and opponent's stochastic policy

**Computation Method:**
- Monte Carlo with Common Random Numbers (CRN) for variance reduction
- Grid search: coarse (step=5.0) → medium (step=1.0) → fine (step=0.25)
- M = 8192 samples per evaluation

Reference: `run/run_two_players.py:eval_exploitability()` (lines 242-337)

### Convergence Criterion

A run is considered converged when BOTH conditions are met:
1. Effort is within δ=0.5 of e* for W=20 consecutive updates
2. Exploitability < ε_threshold=0.05 for P=5 consecutive evaluations
   (unless `--disable-exploitability` is set)

### Cheap-Gate Mechanism

The cheap-gate controls when to trigger expensive exploitability evaluations based on inexpensive stability metrics:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `mean_kl_window` | < 0.0045 | Mean KL divergence over sliding window |
| `std_kl_window` | < 0.0035 | Std of KL divergence over sliding window |
| `drift_effort` | < 2.0 | Policy effort drift over window |

The gate must pass for `patience_drift=2` consecutive evaluations before triggering exploitability.

Reference: `run/run_two_players.py:CheapGateTracker` (lines 138-170)

## Ablation Flags

### `--exploit-every-updates N` (default: 10)

Maximum interval between exploitability evaluations. Caps worst-case computational cost when cheap-gate is disabled. The gate can still trigger evaluation earlier if it passes.

### `--disable-cheap-gate`

Gate is always ON → exploitability evaluation is eligible every update. Combined with `--exploit-every-updates N`, evaluations occur every N updates guaranteed.

**Use case:** Ablation to measure the benefit of the cheap-gate mechanism.

### `--disable-exploitability`

Completely skip exploitability computation. Convergence is based on effort gap only. All exploitability values are NaN with `exploitability_is_valid = False`.

**Note:** This flag overrides `--exploit-every-updates` (no evaluations regardless).

**Use case:** Ablation to measure the benefit of the exploitability term.

### `--ablation-name <name>` (default: "baseline")

Tag for this ablation variant. Required in every output record (JSON, CSV, metadata).

### Flag Interaction Matrix

| disable-cheap-gate | disable-exploitability | exploit-every-updates | Behavior |
|-------------------|------------------------|----------------------|----------|
| False | False | 10 | Normal: gate controls, but at least every 10 updates |
| True | False | 10 | Eval every 10 updates (gate always passes) |
| True | False | 1 | Eval every update (expensive, for debugging) |
| False | True | any | Never eval, converge on effort only |
| True | True | any | Never eval (disable-exploitability wins) |

## Output Structure

```
paper_out/
├── figures/
│   ├── convergence_main.png
│   ├── convergence_main.pdf
│   ├── kl_dynamics.png
│   ├── exploitability_dynamics.png
│   ├── beta_evolution.png
│   └── ablation_comparison.png
├── tables/
│   ├── summary_metrics.csv
│   ├── summary_metrics.tex
│   ├── ablation_results.csv
│   ├── ablation_results.tex
│   ├── final_summary.csv
│   └── final_summary.tex
└── data/
    ├── convergence_main.csv
    ├── kl_dynamics.csv
    └── ...
```

## Module Structure

```
paper_artifacts/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point
├── config.py            # Constants, paths, style settings
├── run_registry.py      # Discover runs, map to (method, q, seed, ablation)
├── extract.py           # Load logs -> tidy DataFrames
├── metrics.py           # Convergence step, exploitability summary, gaps
├── plots.py             # matplotlib figures (PNG/PDF)
├── tables.py            # CSV + LaTeX table generation
└── README.md            # This file
```

## Backward Compatibility

### Old Filename Patterns (Legacy)

```
ppo_q40.0_convergence.json
gradient_q25.0_convergence.json
```

Inferred as: seed=42 (default), ablation="baseline"

### New Filename Patterns

```
ppo_q40.0_seed42_convergence.json
ppo_q40.0_seed42_no_cheap_gate_convergence.json
ppo_q40.0_seed123_baseline_convergence.json
```

### Fallback Strategy for Run Registry

1. Try to find matching `*_metadata.json` for full info
2. Parse filename for method, q, seed, ablation
3. For old files: join with CSV on (method, q) to get seed if available
4. If still unknown: seed=42, ablation="baseline", log warning

## Quality Classification

| Quality | Gap from e* |
|---------|-------------|
| Excellent | < 0.5 |
| Good | < 1.0 |
| Fair | < 5.0 |
| Poor | ≥ 5.0 |

All experiments must achieve at least "Good" quality across all test conditions.

## Dependencies

- numpy
- pandas
- matplotlib
- scipy (optional, for confidence intervals)
