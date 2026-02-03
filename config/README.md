# config/

## Purpose

Experiment configuration files defining parameters for different tournament scenarios. Each file exports a `config` dictionary with game parameters, PPO hyperparameters, and convergence settings.

## Key Contents

| File | Description |
|------|-------------|
| `one_stage_two_players.py` | **Primary config** - One-stage, two-player symmetric tournament (active track) |
| `one_stage_three_players.py` | One-stage, three-player symmetric tournament |
| `two_stage_two_players.py` | Two-stage sequential tournament with information revelation |
| `asymmetric_cost_two_players.py` | Two-player with different cost parameters (k1 ≠ k2) |
| `asymmetric_ability_two_players.py` | Two-player with different ability parameters (l1 ≠ l2) |
| `different_ability_two_players.py` | Extended config with parameter grid builder for ability experiments |

## Entry Points / How to Use

Configs are imported by run scripts:

```python
from config.one_stage_two_players import config as base_config

# Override specific parameters via CLI or code
cfg = dict(base_config)
cfg["q"] = 40.0
cfg["episodes"] = 2048000
```

## Dependencies & Contracts

**Depends on:**
- `utils.theory` - For theoretical effort calculations (some configs)

**Provides to system:**
- `config` dict with standardized keys for all experiment parameters
- Theoretical effort formulas: `e* = (w_H - w_L) / (4 * q * k)` for two-player

## Standard Config Keys

### Game Parameters
- `w_h`, `w_l` - Prize values (high/low)
- `k` - Cost coefficient (quadratic cost: `k * e²`)
- `q` - Noise parameter (Uniform(-q, q) distribution)
- `effort_bounds` - Effort range `[min, max]`

### PPO Parameters
- `gamma`, `gae_lambda`, `clip_eps`, `lr` - Standard PPO hyperparameters
- `steps_per_update`, `epochs`, `minibatch_size` - Training schedule
- `entropy_schedule`, `lr_schedule`, `clip_schedule` - Annealing schedules

### Convergence Settings
- `cheap_gate_profiles` - KL/drift thresholds for convergence evaluation
- `convergence` - Window sizes, delta thresholds, patience settings

## Gotchas / Conventions

- All configs export a dict named `config`
- `q_list = [25.0, 40.0, 55.0]` is the standard test sweep
- Theoretical effort: `e*(q) = (w_H - w_L) / (4 * q * k)` (denominator 4 for two-player)
- CLI arguments override config values (see `run/run_two_players.py`)

## Change Log (local)

| Date | Change |
|------|--------|
| 2026-02-03 | Added README.md for directory documentation |
