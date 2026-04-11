# q=45/55 Convergence Fix

Status: in-progress
Current phase: Round 1.5 — concentration freeze fix

## Step 1 Findings: Concentration Growth Mechanism

### Q1: What outputs concentration?

`ActorCriticMeanConc.forward()` (agents/ppo_two_players_clean.py:96-106):
```python
mean = sigmoid(mean_head(h))           # ∈ (0, 1)
conc = softplus(conc_head(h)) * scale + conc_min  # ∈ [conc_min, conc_max]
alpha = mean * conc
beta = (1 - mean) * conc
```

theory_align_v2 defaults (run/run_two_players.py:1882-1891):
- conc_min ramps from 100 → 1000 over updates 20-70
- conc_scale ramps from 100 → 10000 over updates 20-70
- conc_max = 100000
- **entropy = 0 literally** (lines 1878-1880: start/hold/end all set to 0.0)

Value range of conc: `softplus(x) * 10000 + 1000` where softplus ≥ 0.
At softplus(0) ≈ 0.69, conc ≈ 0.69 × 10000 + 1000 = **7900**.
The network doesn't even need to learn anything for concentration to be huge —
the scale and floor parameters push it there by construction.

### Q2: Is entropy literally 0?

**Yes.** Lines 1878-1880:
```python
cfg["entropy_coef_start"] = 0.0
cfg["entropy_coef_hold"] = 0.0
cfg["entropy_coef_end"] = 0.0
```
Line 528 of PPO loss: `loss = policy_loss + value_coef * value_loss - 0.0 * entropy`
The entropy term contributes exactly zero to the gradient.

### Q3: What drives concentration up?

Two forces:

**Force 1 (dominant): conc_min/conc_scale ramp.** The architecture imposes
`conc ≥ conc_min` and scales the conc_head output by `conc_scale`. During
updates 20-70, conc_min ramps 100→1000 and conc_scale ramps 100→10000.
This FORCES concentration to grow regardless of gradient signal. Even if
the conc_head output stays constant, concentration goes from ~170 to ~7900+.

**Force 2 (secondary): var_coef loss.** Line 541:
```python
var_loss = var_coef * var_effort.mean()
```
where `var_effort = alpha*beta / (conc² * (conc+1)) * range²`. This loss
PENALIZES variance — it pushes concentration UP to reduce policy spread.
var_coef ramps from 0.0 to 0.05 over updates 20-70.

**Force 3 (absent): entropy.** With entropy_coef=0, there is NO counterforce
that would encourage exploration or resist concentration growth.

### Q4: Actual concentration in q=35 baseline (seed=42)

| Update | Concentration | Effort | Ramp phase |
|--------|--------------|--------|------------|
| 0 | 170 | 51.16 | warmup (conc_min=100, scale=100) |
| 10 | 179 | 47.48 | warmup |
| 20 | 383 | 44.57 | ramp starts |
| 30 | 2694 | 44.29 | mid-ramp (scale rising fast) |
| 49 | 8580 | 44.51 | ramp complete, conc ≈ softplus × 10000 + 1000 |
| 58 | 12318 | 44.50 | still growing |

Key observation: by update 20 (end of warmup), effort has already moved
from 51 to 44.6 — nearly at e*=45.45. The first 20 updates with low
concentration (conc < 400) are when ALL the learning happens. After the
ramp kicks in, effort barely changes (44.57 → 44.50 over 38 updates).

**This means the agent has a ~20 update window to find e* before the policy
freezes.** At ~0.3 effort units per update of descent (from 51→44.6 in 20
updates), the maximum travel distance is ~6-7 units. This perfectly explains
the init-distance threshold: q=35 (dist=4.6) passes, q=45 (dist=6.9) just
barely fails, q=55 (dist=21) has no chance.

## What's done
- Step 1 complete: concentration mechanism fully characterized
- Root cause confirmed: forced conc ramp + zero entropy = 20-update freeze window

## Step 2: Concentration Fix Design

**Approach:** Extend the concentration ramp warmup period via CLI flag.

**Principle:** During warmup, conc_min=100, conc_scale=100, var_coef=0.0.
This keeps concentration at ~170, allowing the Beta mean to move freely.
Currently warmup=20 updates → ~6 effort units of travel. Extending to
200 updates → ~60 effort units of travel (more than [0,100] range).

**Implementation:**
- Added `--override-conc-ramp-warmup N` flag to `run/run_two_players.py`
  - arg definition: line ~2115 (after --override-conc-max)
  - config override: line ~1926 (after override_conc_max handling)
- Default: None (uses existing cfg value = 20, unchanged behavior)
- When set: overrides `cfg["theory_align_v2_ramp_warmup"]`

**Files modified:** `run/run_two_players.py` only (2 insertions, 0 deletions)

**Expected effect on concentration trajectory:**
- warmup=20 (current): conc hits 2700 by update 30 → policy frozen
- warmup=200: conc stays at ~170 until update 200 → 200 updates of free movement
- After warmup ends, ramp proceeds as before (50 steps to reach full scale)

**Experiment flags:** `--override-conc-ramp-warmup 200`

## What's next
- Step 3: Verify default behavior unchanged (1-update dry-run)
- Step 4: Run experiments with --override-conc-ramp-warmup 200
