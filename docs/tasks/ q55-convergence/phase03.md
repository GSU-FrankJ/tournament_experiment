# Phase 03: Simple Adaptive Entropy (conditional)

## Precondition
Only execute if Phase 02 shows q=35/40 regress under fixed high entropy, meaning different q values need different entropy levels.

## Objective
Implement minimal SAC-style adaptive entropy: a single learnable log_α that tracks a fixed target entropy H_target.

## Design
```python
# Once per rollout (after collecting batch, before PPO epochs):
H_batch = mean(entropy_of_beta(alpha, beta))
log_alpha += eta_alpha * (H_target - H_batch)
log_alpha = clip(log_alpha, log_alpha_min, log_alpha_max)
entropy_coef = exp(log_alpha)
# Freeze entropy_coef for all PPO epochs in this update
```

## Key parameters
| Parameter | Initial value | Source |
|-----------|--------------|--------|
| H_target | TBD | Entropy at q=35 mid-transport |
| eta_alpha | 0.01 | Conservative; tune if too slow |
| log_alpha_min | -5 | entropy_coef ≈ 0.007 |
| log_alpha_max | 0 | entropy_coef ≈ 1.0 |

## Files to modify
- `agents/ppo_two_players_clean.py` — add log_alpha state, adaptive update method
- `run/run_two_players.py` — call adaptive update, log entropy_coef trajectory

## Verification
- q=55 seed=42 converges
- q=35, q=40 seed=42 still converge
- entropy_coef trajectory: high during transport, decays during contraction
