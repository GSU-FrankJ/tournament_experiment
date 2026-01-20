# Data Mixing Issue - Quick Summary

## Verdict: ✗ CONFIRMED HIGH-RISK MIXING

---

## The Problem in One Sentence

**Opponent-generated actions (with opponent's log-probs) are being stored in the PPO buffer and then used to compute PPO ratios against the learner's current policy, creating meaningless gradient signals.**

---

## Evidence Table

| Finding | File:Lines | What Happens | Risk |
|---------|-----------|--------------|------|
| Player 2 sometimes uses opponent | `run/run_two_players.py:448-450` | `if use_opponent:`<br>`agent.act_opponent(s2)` | Generates action from lagged/historical opponent network |
| Both players always stored | `run/run_two_players.py:457-458` | `agent.store(s1, ...)`<br>`agent.store(s2, ...)` | **No filtering** — opponent samples go into buffer |
| PPO uses all data | `agents/ppo_two_players_clean.py:243-245` | Loads all stored transitions | No provenance checking |
| Invalid ratio computation | `agents/ppo_two_players_clean.py:277` | `ratio = exp(learner_logp - opponent_logp)` | **Meaningless cross-policy comparison** |

---

## Diagnostic Results

**Test Run**: 200 steps, 50% opponent probability

```
Total transitions:     400
Learner-generated:     296 (74%)
Opponent-generated:    104 (26%) ← These cause the problem!

All 400 used in PPO update with no filtering
```

---

## Why This Is High-Risk

1. **PPO theory violated**: Ratio should be π_new(a|s) / π_old(a|s) from same policy; instead we get π_learner / π_opponent
2. **Invalid clipping**: Can't bound policy updates when comparing different policies
3. **Distorted advantages**: GAE assumes consistent behavior policy
4. **Misleading metrics**: `approx_kl` mixes two different KL measurements
5. **Training instability**: Contaminated gradients, especially during early training (lag_prob=1.0 → 50% opponent samples)

---

## Recommended Fixes

### ✅ Option A: Store Only Learner (Cleanest)
**File**: `run/run_two_players.py:457-458`

```python
# Current (WRONG):
agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))

# Fixed:
agent.store(s1, a1_norm, logp1, float(rewards[0].item()), v1, bool(done))
if not use_opponent:  # Only store player 2 when using learner policy
    agent.store(s2, a2_norm, logp2, float(rewards[1].item()), v2, bool(done))
```

**Trade-off**: Less data during lag phase (may need more steps_per_update)

---

### ✅ Option D: Remove Lag Entirely (Simplest)
**File**: `run/run_two_players.py:448-453`

```python
# Current (WRONG):
use_opponent = (lag_prob > 0.0) and (rng.random() < lag_prob)
if use_opponent:
    a2_norm, e2, logp2, _ = agent.act_opponent(s2)
    v2 = agent.value_only(s2)
else:
    a2_norm, e2, logp2, v2 = agent.act(s2)

# Fixed (pure symmetric self-play):
a2_norm, e2, logp2, v2 = agent.act(s2)  # Always use learner
```

**Trade-off**: Removes lag stabilization (if it was actually helping)

---

## Action Items

1. **Choose fix**: Option A (if want "learner vs env") or Option D (if want "symmetric self-play")
2. **Implement**: Modify `run/run_two_players.py` as shown above
3. **Re-run experiments**: Get valid baselines with fixed code
4. **Verify**: Add debug logging to confirm only learner samples in PPO update

---

## Full Details

See: `docs/data_provenance_investigation.md`



