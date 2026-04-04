# Phase 04: Adaptive Entropy (conditional)

## Precondition
Only execute if Phase 03 shows q=35/40 regress under standard mode + entropy_end=0.002,
meaning different q values need different entropy levels.

## Objective
Implement minimal SAC-style adaptive entropy: a learnable log_α that tracks
a fixed target entropy H_target, so the entropy coefficient self-adjusts per q.

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
| log_alpha_min | -7 | entropy_coef ≈ 0.001 |
| log_alpha_max | 0 | entropy_coef ≈ 1.0 |

## Files to modify
- `agents/ppo_two_players_clean.py` — add log_alpha state, adaptive update method
- `run/run_two_players.py` — call adaptive update, log entropy_coef trajectory

## Verification
- q=55 seed=42 converges with gap < 2
- q=35, q=40 seed=42 still converge with gap < 2
- entropy_coef trajectory: high during transport, decays during contraction
