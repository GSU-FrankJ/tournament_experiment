# The 3-Player PPO Convergence Problem: Complete Context

## 1. The Symptom

In a symmetric 3-player single-stage tournament game, PPO agents learn effort levels that are systematically ~5 units below the Nash equilibrium. This gap is robust across seeds, hyperparameters, and training duration.

| Setting | e*(theory) | PPO final effort | Gap | Converged? |
|---------|-----------|-----------------|-----|-----------|
| 3p q=35 (5 seeds) | 62.50 | 56.9-57.5 | 5.0-5.7 | NO |
| 3p q=40 (3 seeds) | 54.69 | 47.4-54.6 | 0.1-7.8 | 1/3 seeds |
| 3p q=55 (3 seeds) | 39.77 | 35.8-43.7 | 3.8-4.5 | NO |
| **2p q=35 (5 seeds)** | **62.50** | **63.0-64.4** | **0.5-1.9** | **YES** |
| **2p q=40 (5 seeds)** | **54.69** | **55.5-58.3** | **0.8-3.6** | **YES** |

The 2-player version converges. The gradient descent solver converges perfectly for 3p (gap=0.12). The problem is specific to **PPO learning dynamics in the 3-player setting**.

## 2. What Was Tried (All Failed)

### Entropy ablations
| Variant | Change | Gap | Verdict |
|---------|--------|-----|---------|
| baseline | entropy 0.03→0.005 | 5.34 | — |
| no_entropy | entropy=0 throughout | 4.54 | Slightly better gap, but exploit worse (0.198) |
| ent_start_01 | entropy 0.01→0.005 | 4.79 | Slightly better gap, exploit worse (0.187) |

**Conclusion**: Entropy is not the cause. Removing it slightly helps the gap but worsens exploitability. The policy isn't being pushed away from equilibrium by entropy — it simply can't find its way there.

### Optimization ablations
| Variant | Change | Gap | Verdict |
|---------|--------|-----|---------|
| spu2731 | steps_per_update 4096→2731 | 6.06 | Worse |
| no_adv_norm | disable advantage normalization | 7.44 | Much worse |
| adam_reset100 | reset optimizer at update 100 | 5.35 | No change |
| 5000upd | train for 5000 updates (3.3x longer) | 5.35 | No change |
| 5000upd_v2 | even longer (killed at 4776 updates) | ~5.3 | No change |

**Conclusion**: More training doesn't help. The policy is stuck, not slow. Reducing gradient steps worsens it. Advantage normalization removal destroys performance.

### Architecture ablations
| Variant | Change | Gap | Verdict |
|---------|--------|-----|---------|
| binary_h64 | stochastic binary rewards + hidden=64 | 9.88 | Much worse |
| binary_h128 | stochastic binary rewards + hidden=128 | 9.27 | Much worse |
| smooth_h64 | expected utility + hidden=64 | 5.98 | Slightly worse |
| mean_conc | mean+concentration parameterization | 5.43 | No change |

**Conclusion**: Switching to binary rewards makes things much worse (more variance, less signal). Network architecture changes have no effect.

### Summary of eliminated hypotheses
1. **Entropy too high** → No (removing it doesn't fix it)
2. **Over-optimization per update** → No (reducing gradient steps is worse)
3. **Insufficient training** → No (5000 updates = no change from 1500)
4. **Network architecture** → No (mean_conc, different hidden sizes = no change)
5. **Optimizer stuck in local minimum** → No (resetting Adam doesn't help)
6. **Advantage normalization** → No (disabling it is much worse)
7. **Stochastic vs expected rewards** → No (binary rewards are much worse)

## 3. Root Cause Analysis

### 3.1 The 2-player vs 3-player reward signal

**2-player**: Each game has exactly 1 winner and 1 loser. Every transition gets a clear signal:
- Winner: reward = w_H - cost → positive advantage (do more of this)
- Loser: reward = w_L - cost → negative advantage (do less of this)
- **Signal clarity: 100% of transitions carry useful gradient information**

**3-player**: Each game has 1 winner and 2 losers. The transitions:
- Winner (rank 1): reward = w_H - cost → clear positive advantage
- Loser A (rank 2): reward = w_L - cost → negative, but close to baseline
- Loser B (rank 3): reward = w_L - cost → same as rank 2

**Critical issue**: In the current implementation with `use_binary_rewards=False` (expected utility mode), ALL THREE players get continuous rewards:
```python
utility_i = w_L + p_i(e_i, e_j, e_k) * (w_H - w_L) - k * e_i^2
```

At symmetric play (all efforts equal), `p_i = 1/3` for all players:
```
utility = 3.0 + (1/3)(3.5) - k*e^2 = 4.167 - k*e^2
```

The **gradient of utility w.r.t. effort at symmetric play** is:
```
∂u/∂e = (w_H - w_L) * ∂p/∂e - 2ke
```

For the 2-player case: `∂p/∂e = 1/(2q)` (derivative of piecewise linear CDF)
For the 3-player case: `∂p/∂e = 1/(2q)` (same! verified in `utils/theory.py`)

So the **marginal incentive gradient is identical** between 2p and 3p at the equilibrium. The difference must be in how PPO processes this signal.

### 3.2 The PPO processing bottleneck

The advantage function is: `A(s,a) = R(s,a) - V(s)`

In 2-player:
- V(s) ≈ average utility at current effort level
- For the winner: A > 0 (effort was good)
- For the loser: A < 0 (effort was bad, or unlucky)
- **Variance of A**: depends on win/lose spread = `w_H - w_L = 3.5`

In 3-player (expected utility mode):
- All 3 players get nearly identical rewards when efforts are close to symmetric
- V(s) ≈ the common utility level
- A ≈ small perturbation around 0 for all players
- **Variance of A**: much smaller because rewards are continuous, not binary

This is the core problem: **the advantage signal variance collapses in 3-player expected-utility mode**.

When advantages are small and tightly clustered:
1. Advantage normalization maps them to ~N(0,1), but the **direction** is unreliable
2. PPO's clipping prevents large updates, so small advantages → tiny policy changes
3. The policy converges to a plateau where the gradient signal is too weak to escape

### 3.3 Why the policy gets stuck at ~57 instead of ~62.5

At effort=57 (below equilibrium 62.5), a player who increases effort gains:
- Higher win probability → higher expected payoff
- But higher cost → lower utility

The **net marginal benefit** of increasing effort from 57→58:
```
∂u/∂e = 3.5 * (1/70) - 2*0.0004*57 = 0.05 - 0.0456 = 0.0044
```

This is a **tiny** gradient (0.4% of the effort value). The PPO advantage estimate has much more noise than this signal. The policy cannot distinguish "increase effort by 1" from random fluctuation.

At equilibrium (e=62.5):
```
∂u/∂e = 3.5 * (1/70) - 2*0.0004*62.5 = 0.05 - 0.05 = 0
```

So the gradient landscape near equilibrium is extremely flat. The difference between effort=57 and effort=62.5 in terms of marginal utility gradient is only 0.0044 — PPO cannot resolve this.

### 3.4 Why 2-player doesn't have this problem

In 2-player with **binary rewards** (stochastic winner/loser):
- Winner gets utility ≈ 6.5 - cost ≈ 5.0
- Loser gets utility ≈ 3.0 - cost ≈ 1.5
- **Advantage spread** ≈ 3.5 (the full prize gap)

This large discrete jump means the advantage function has **high signal-to-noise ratio**. PPO can clearly identify which effort levels win more often.

In 3-player with **expected utility**:
- All 3 players get utility ≈ 4.17 - cost (when symmetric)
- Advantage spread ≈ 0.01-0.05 (tiny deviations from V(s))
- **Signal is buried in value function estimation noise**

## 4. Architecture Details (for the implementer)

### Network (agents/ppo_three_players.py)
```
Input: state (3,) = [q/60, k/1e-3, (w_h-w_l)/10]
       ↓
Linear(3, 64) → Tanh
Linear(64, 64) → Tanh
       ↓
┌──────┼──────────┐
alpha_head(64,1)  beta_head(64,1)  value_head(64,1)
Softplus + 1.0    Softplus + 1.0
       ↓               ↓
    Beta(α, β) distribution on [0,1]
    effort = low + sample * (high - low)   # low=0, high=200
```

### Rollout collection (run/run_three_players.py, ~line 842)
```python
# Per environment step:
state = agent.state_from_params(q=q, k=k, w_h=w_h, w_l=w_l)
a1, e1, logp1, v1 = agent.act(state)  # 3 independent samples
a2, e2, logp2, v2 = agent.act(state)  # from the SAME policy
a3, e3, logp3, v3 = agent.act(state)

_, rewards, _, done, _ = env.step((e1, e2, e3))

agent.store(state, a1, logp1, rewards[0], v1, done)  # 3 transitions stored
agent.store(state, a2, logp2, rewards[1], v2, done)
agent.store(state, a3, logp3, rewards[2], v3, done)
```

### GAE (agents/ppo_three_players.py, ~line 415)
Standard GAE with γ=0.99, λ=0.95. Advantages normalized to mean=0, std=1.

### PPO Loss (agents/ppo_three_players.py, ~line 573)
Standard clipped surrogate + 0.5 * value loss - entropy_coef * entropy.
Clip schedule: 0.50 → 0.35. Entropy schedule: 0.03 (hold 2/3) → 0.005.

### Environment (envs/three_players_env.py)
Default mode: **expected utility** (analytical win probabilities, no sampling noise).
```python
p_i = win_prob_three_players(e_i, e_j, e_k, q)  # from utils/prob.py
utility_i = w_L + p_i * (w_H - w_L) - k * e_i^2
```

### Key hyperparameters (config/one_stage_three_players.py)
- steps_per_update = 4096 (→ 12288 transitions/update with 3 players)
- minibatch_size = 1024
- update_epochs = 6
- max_updates = 1500
- lr: 3e-4 → 2e-4
- GAE: γ=0.99, λ=0.95
- Exploit eval: M=16384 MC samples, eps=0.05, patience=5

### Comparison with 2-player
| Aspect | 2-Player | 3-Player |
|--------|----------|----------|
| Reward type | Binary (stochastic winner) | Expected utility (continuous) |
| Transitions/step | 2 | 3 |
| Advantage variance | High (~3.5 prize gap) | Low (~0.01 utility deviation) |
| Equilibrium formula | (w_H-w_L)/(4qk) | Same |
| Convergence | YES (gap < 2) | NO (gap ≈ 5) |

## 5. The Gradient Solver Works Perfectly

`run/run_three_players.py --method gradient` uses analytical gradients:
```python
dp_de = win_prob_three_players_grad(e_i, e_j, e_k, q)
grad_i = (w_H - w_L) * dp_de_i - 2*k*e_i
e_i += lr * grad_i  # direct gradient ascent
```

This converges to gap=0.12 in a few hundred steps. The issue is entirely in PPO's ability to estimate and follow the gradient through its RL pipeline.
