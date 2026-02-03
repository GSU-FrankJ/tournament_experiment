# results/convergence_history/

## Purpose

Step-by-step convergence data from training runs. Each experiment produces a JSON file with effort trajectories, KL divergence, exploitability, and other metrics over training.

## Key Contents

| Pattern | Description |
|---------|-------------|
| `ppo_q{q}_seed{seed}_{ablation}_convergence.json` | PPO convergence data |
| `gradient_q{q}_convergence.json` | Gradient solver convergence data |
| `*_metadata.json` | Run metadata (hyperparameters, config) |

## File Format

### Convergence JSON Structure

```json
{
  "steps": [0, 1, 2, ...],
  "agent1_effort": [50.0, 55.2, ...],
  "agent2_effort": [50.0, 54.8, ...],
  "theoretical_effort": 87.5,
  "exploitability": [NaN, NaN, 0.12, ...],
  "exploitability_is_valid": [false, false, true, ...],
  "approx_kl": [0.001, 0.002, ...],
  "alpha_mean": [2.5, 2.8, ...],
  "beta_mean": [3.0, 3.2, ...]
}
```

### Metadata JSON Structure

```json
{
  "method": "ppo",
  "q": 25.0,
  "seed": 50,
  "ablation_name": "baseline",
  "w_h": 6.5,
  "w_l": 3.0,
  "k": 0.0004
}
```

## Entry Points / How to Use

**Generated automatically** - do not edit manually:

```bash
# Generate via training
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 50

# Load for analysis
import json
with open("results/convergence_history/ppo_q25.0_seed50_baseline_convergence.json") as f:
    data = json.load(f)
```

## Dependencies & Contracts

**Depends on:** `run/run_two_players.py` - Generates files during training

**Provides to system:**
- `paper_artifacts/` - Source data for paper figures
- `tools/plot_*.py` - Source data for convergence plots

## Naming Convention

### New Format (explicit)
`{method}_q{q}_seed{seed}_{ablation}_convergence.json`

Example: `ppo_q25.0_seed50_baseline_convergence.json`

### Legacy Format (inferred)
`{method}_q{q}_convergence.json`

Example: `ppo_q25.0_convergence.json` (seed=42, ablation="baseline" inferred)

## Gotchas / Conventions

- `exploitability` contains NaN for steps where evaluation was skipped (cheap-gate)
- Use `exploitability_is_valid` to filter valid measurements
- Legacy files lack seed/ablation - paper_artifacts handles fallback
- Files are overwritten if re-running with same parameters

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Added README.md for directory documentation |
