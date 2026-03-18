# Policy and Scale Diagnostics Guide

This document describes the instrumentation added to track policy definitions and scale statistics during PPO training.

## 1. Policy Definition

### What is "policy"?

The `policy=` value represents the **Beta distribution mean** mapped to the effort range:

```
policy_mean_effort = effort_low + (alpha / (alpha + beta)) * (effort_high - effort_low)
```

### Code References

| Location | File | Description |
|----------|------|-------------|
| Beta distribution | `agents/ppo_two_players_clean.py` | `ActorCritic.dist()` returns Beta(alpha, beta) |
| Policy computation | `run/run_two_players.py` | `dist.mean` scaled to effort range |
| Helper function | `utils/rollout_stats.py` | `compute_policy_mean_effort()` |
| Verification | `run/run_two_players.py` | `policy_mean_check_err` validation |

### Verification

The `[PolicyCheck]` line (at update 1 and every 100 updates) confirms:
- `policy_mean_check_err < 0.01` means policy equals Beta mean scaled to effort range

## 2. Rollout Sample Metrics

### Definitions

| Metric | Definition | Data Source |
|--------|------------|-------------|
| `sample_avg_effort` | Mean of sampled efforts during rollout | Learner-generated actions only |
| `mean_vs_sample_gap` | `policy_mean_effort - sample_avg_effort` | Comparison metric |
| `effort_sample_count` | Number of effort samples in the update | Count of learner transitions |

### Data Provenance

Both P1 and P2 sampled efforts are included. `effort_sample_count` ~ 2 x `steps_per_update`.

### Code References

| Location | File | Description |
|----------|------|-------------|
| P1 effort tracking | `run/run_two_players.py` | Always tracked |
| P2 effort tracking | `run/run_two_players.py` | Always tracked |
| Accumulator class | `utils/rollout_stats.py` | `RolloutStatsAccumulator` |

## 3. Scale Statistics

All statistics computed per PPO update on stored learner transitions:

| Metric | Definition | Tensor Source |
|--------|------------|---------------|
| `state_mean` | Global mean across all state elements | `storage["states"]` |
| `state_std` | Global std (population, unbiased=False) | `storage["states"]` |
| `reward_mean` | Mean of rewards | `storage["rewards"]` |
| `reward_std` | Std of rewards (population) | `storage["rewards"]` |
| `adv_mean` | Mean of raw advantages (before normalization) | `_compute_gae()` output |
| `adv_std` | Std of raw advantages (before normalization) | `_compute_gae()` output |
| `adv_norm_std` | Std of normalized advantages (should be ~1.0) | After normalization |
| `value_mean` | Mean of value estimates | `storage["values"]` |
| `value_std` | Std of value estimates | `storage["values"]` |

### Code References

| Location | File | Description |
|----------|------|-------------|
| State/reward/value stats | `agents/ppo_two_players_clean.py` | Computed in `update()` |
| Advantage computation | `agents/ppo_two_players_clean.py` | Raw advantage mean/std |
| Advantage normalization | `agents/ppo_two_players_clean.py` | Normalization |
| Welford accumulator | `utils/rollout_stats.py` | Numerically stable |

## 4. Console Output Format

Each PPO update logs:

```
[Update N] q=X: e*=Y, policy=Z, gap=W, entropy=E, lag_prob=L, approx_kl=K, alpha_mean=A, beta_mean=B
  [Rollout] sample_avg_effort=S, mean_vs_sample_gap=G, effort_samples=C
  [Scale] state_mean=M, state_std=D, reward_mean=R, reward_std=T, adv_mean=V, adv_std=U, adv_norm_std=N
```

At updates 1, 100, 200, ...:
```
  [PolicyCheck] policy_mean_check_err=E (expected <0.01; confirms policy=alpha/(alpha+beta) scaled to effort_range)
```

## 5. CSV Columns Added

| Column | Type | Description |
|--------|------|-------------|
| `policy_mean_effort` | float | Beta mean mapped to effort range |
| `sample_avg_effort` | float | Mean sampled effort from last update |
| `mean_vs_sample_gap` | float | policy_mean_effort - sample_avg_effort |
| `effort_sample_count` | int | Number of effort samples |
| `state_mean` | float | Global state mean |
| `state_std` | float | Global state std |
| `reward_mean` | float | Mean reward |
| `reward_std` | float | Reward std |
| `adv_mean` | float | Raw advantage mean |
| `adv_std` | float | Raw advantage std |
| `adv_norm_std` | float | Normalized advantage std (~1.0) |
| `value_mean` | float | Value estimate mean |
| `value_std` | float | Value estimate std |

## 6. Sanity Check Commands

### Short Sanity Run (20k episodes)

```bash
python run/run_two_players.py --method ppo --q 40 --episodes 20000 --seed 42
```

### Verification Checklist

1. `sample_avg_effort` is finite (not nan/inf)
2. `effort_samples` ~ 2 x steps_per_update (e.g., 8192 for 4096 steps)
3. `policy_mean_check_err < 0.01`
4. `adv_norm_std` ~ 1.0

### Verification Commands

```bash
# Check sample_avg_effort is finite
grep "sample_avg_effort" results/logs/*.log | head -5

# Check policy verification
grep "policy_mean_check_err" results/logs/*.log

# Check scale stats
grep "Scale" results/logs/*.log | head -5
```

## 7. Interpretation Guide

| Condition | `mean_vs_sample_gap` | Interpretation |
|-----------|---------------------|----------------|
| Near 0 | ~ 0 | Policy mean matches sample average (expected converged) |
| Large positive | >> 0 | Policy predicts higher effort than sampled (high entropy) |
| Large negative | << 0 | Policy predicts lower effort than sampled (rare) |

| Metric | Expected | Concern If |
|--------|----------|------------|
| `state_std` | 0.3-1.5 | Very small (<0.01): states collapsing |
| `reward_std` | > 0 | Near 0: degenerate rewards |
| `adv_std` | 0.1-10 | Very small (<0.01): no learning signal |
| `adv_norm_std` | ~1.0 | Not ~1.0: bug in normalization |

## 8. Files Modified

| File | Changes |
|------|---------|
| `utils/rollout_stats.py` | NEW - RolloutStatsAccumulator and helpers |
| `agents/ppo_two_players_clean.py` | Extended `update()` return metrics |
| `run/run_two_players.py` | Integrated rollout stats, console output, CSV columns |
| `docs/POLICY_SCALE_DIAGNOSTICS.md` | NEW - This documentation |

---

*Last updated: 2024-12-18*

