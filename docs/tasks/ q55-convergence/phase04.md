# Phase 04: Pairwise Signal Gate (conditional)

## Precondition
Only execute if Phase 03 adaptive entropy shows instability (oscillating entropy_coef, or convergence for some seeds but not others).

## Objective
Add a binary safety gate that prevents entropy from decreasing when pairwise learning signal is unhealthy.

## Design
```python
signal_healthy = (
    var_e1_minus_e2 > threshold_var_e
    and var_r1_minus_r2 > threshold_var_r
)

if signal_healthy:
    log_alpha += eta_alpha * (H_target - H_batch)  # normal update
else:
    log_alpha += max(0, eta_alpha * (H_target - H_batch))  # only allow increase
```

Gate thresholds calibrated from q=35 successful runs (minimum values observed during convergence).

## Files to modify
- `agents/ppo_two_players_clean.py` — add gate logic
- `run/run_two_players.py` — compute and log pairwise metrics

## Verification
- q=55 converges without oscillation
- Gate fires during early transport phase, releases during contraction
- q=35/40 gate rarely fires (signal always healthy)
