# Algorithm Improvement Proposals for 3-Player PPO

## Priority ranking

| # | Proposal | Expected impact | Implementation effort | Risk |
|---|----------|----------------|----------------------|------|
| 1 | Pairwise advantage decomposition | High | Medium | Low — pure reward reshaping |
| 2 | Counterfactual baseline (COMA-style) | High | Medium | Medium — new value head |
| 3 | Hybrid binary + expected rewards | High | Low | Low — env flag already exists |
| 4 | Opponent-marginal advantage | Medium | High | Medium — requires opponent modeling |
| 5 | Increased batch + lower lr | Low-Medium | Trivial | Low — already tried partially |
| 6 | Policy gradient with analytical grad injection | Medium | Medium | High — changes RL paradigm |

---

## Proposal 1: Pairwise Advantage Decomposition (RECOMMENDED FIRST)

### Diagnosis it addresses

The 3-player advantage `A = R - V(s)` has low variance because all 3 players get similar expected utility. By decomposing the reward into pairwise comparisons, each sub-signal has the binary high-variance structure that works in 2-player.

### Idea

Instead of computing one advantage from the full 3-player reward, decompose each player's experience into TWO pairwise duels:

```
Player i's reward in a 3-player game with players i, j, k:
  R_i = w_L + p_i * (w_H - w_L) - k * e_i^2

Decompose into pairwise rewards:
  R_i_vs_j = w_L + p_ij * (w_H - w_L) - k * e_i^2    (i vs j, ignoring k)
  R_i_vs_k = w_L + p_ik * (w_H - w_L) - k * e_i^2    (i vs k, ignoring j)

where p_ij = p_from_diff(e_i - e_j, q) is the 2-player win probability.

Shaped reward:
  R_i_shaped = 0.5 * R_i_vs_j + 0.5 * R_i_vs_k
```

### Why this should work

- Each pairwise reward has the **same variance structure** as the 2-player case
- The gradient of `0.5 * p_ij + 0.5 * p_ik` at symmetric play equals `p_i`'s gradient (preserves equilibrium)
- The advantage computed from these rewards will have ~2x the variance → stronger signal
- **Does NOT require knowledge of e***

### Implementation

In `envs/three_players_env.py`, add a reward mode:

```python
def step(self, efforts):
    e1, e2, e3 = efforts

    if self.reward_mode == "pairwise_decomposed":
        # 2-player win probs for each pair
        p12 = p_from_diff(e1 - e2, self.q)
        p13 = p_from_diff(e1 - e3, self.q)
        p21 = p_from_diff(e2 - e1, self.q)
        p23 = p_from_diff(e2 - e3, self.q)
        p31 = p_from_diff(e3 - e1, self.q)
        p32 = p_from_diff(e3 - e2, self.q)

        # Average pairwise expected utility
        u1 = 0.5 * (self.w_l + p12 * self.w_gap - self.k * e1**2) \
           + 0.5 * (self.w_l + p13 * self.w_gap - self.k * e1**2)
        # ... similarly for u2, u3
```

Or equivalently (simpler):
```python
        # Player i's shaped reward: average of "how do I do vs each opponent"
        p1_avg = 0.5 * (p12 + p13)  # average pairwise win rate
        u1 = self.w_l + p1_avg * self.w_gap - self.k * e1**2
```

### Equilibrium preservation proof sketch

At symmetric play (e_i = e_j = e_k = e*):
- p_ij = p_ik = 0.5 (each pairwise duel is 50/50)
- p_avg = 0.5
- Shaped utility = w_L + 0.5 * (w_H - w_L) - k*e^2

FOC: `(w_H - w_L) * ∂p_avg/∂e = 2ke`

At symmetric play: `∂p_ij/∂e = 1/(2q)`, so `∂p_avg/∂e = 0.5 * 1/(2q) + 0.5 * 1/(2q) = 1/(2q)`

This gives: `(w_H - w_L)/(2q) = 2ke` → `e* = (w_H - w_L)/(4qk)` ✓ Same equilibrium.

### Verification

Run with q=35, seed=42, 1500 updates. If gap drops from ~5.3 to <2.0, proceed to 5-seed validation.

### Risk

Very low. This is a **potential shaping** (Ng et al. 1999) — it preserves optimal policy under mild conditions. The equilibrium is unchanged by construction.

---

## Proposal 2: Counterfactual Baseline (COMA-style)

### Diagnosis it addresses

Standard advantage `A = R - V(s)` conflates "my action was good" with "my opponents happened to play in a way that benefited me." In 3-player, this attribution problem is worse because there are 2 opponents.

### Idea

Replace the standard value baseline with a **counterfactual baseline**: "what would my expected reward be if I played a random action, while my opponents play as they did?"

```
A_i^CF = R_i(a_i, a_{-i}) - E_{a'_i ~ π}[R_i(a'_i, a_{-i})]
```

This isolates the effect of player i's specific action from the environment dynamics.

### Implementation

**Option A: Analytical counterfactual** (exact, efficient):

Since the environment is closed-form, we can compute the counterfactual analytically:

```python
# In the rollout, after getting rewards:
# R_i = utility(e_i, e_j, e_k)

# Counterfactual: expected utility if player i plays according to current policy
# E_{e'_i ~ Beta(α,β)}[utility(e'_i, e_j, e_k)]
# = w_L + E[p(e'_i, e_j, e_k)] * w_gap - k * E[e'^2_i]

# Approximate with K=32 Monte Carlo samples from current policy:
e_samples = agent.sample_efforts(state, K=32)  # K samples from Beta(α,β)
cf_utilities = [utility(e_s, e_j, e_k) for e_s in e_samples]
cf_baseline = mean(cf_utilities)

advantage_i = R_i - cf_baseline  # counterfactual advantage
```

**Option B: Learned counterfactual value function**:

Add a second value head that takes (state, e_j, e_k) as input — conditions on opponents' actions:

```python
class ActorCriticCF(nn.Module):
    def __init__(self, state_dim=3, hidden=64):
        ...
        # Standard value head (for GAE returns target)
        self.value_head = nn.Linear(hidden, 1)

        # Counterfactual value head (conditions on opponent efforts)
        self.cf_shared = nn.Sequential(
            nn.Linear(state_dim + 2, hidden),  # +2 for opponent efforts
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def counterfactual_value(self, state, opponent_efforts):
        """V(s, a_{-i}): expected utility given opponents' actions."""
        cf_input = torch.cat([state, opponent_efforts], dim=-1)
        return self.cf_shared(cf_input)
```

Then the advantage becomes:
```
A_i = R_i - V_CF(s, e_j, e_k)
```

### Why this should work

- Removes opponent-induced variance from the advantage
- Each player's advantage purely reflects **how their effort compared to what the policy would have done on average**
- In 2-player, this is less critical because the binary reward already provides strong signal
- In 3-player, where the reward is continuous and differences are tiny, removing opponent variance is crucial

### Risk

Medium. Option A (analytical) is safe but adds computation. Option B requires training a second network which may itself have estimation errors.

---

## Proposal 3: Hybrid Binary + Expected Rewards

### Diagnosis it addresses

The expected-utility reward mode produces low-variance advantages. Binary rewards have high variance but provide clearer signal. A hybrid approach could get the best of both.

### Idea

Use binary (stochastic) rewards during early training (for strong gradient signal), then switch to expected utility (for fine-tuning near equilibrium):

```python
if update_idx < switch_update:
    env.use_binary_rewards = True   # high variance, strong signal
else:
    env.use_binary_rewards = False  # low variance, fine-tuning
```

Or use a weighted combination throughout:

```python
# In env.step():
u_expected = w_L + p_i * w_gap - k * e_i^2
u_binary = (w_H if won else w_L) - k * e_i^2

alpha = schedule(update_idx)  # 1.0 → 0.0 over training
u_i = alpha * u_binary + (1 - alpha) * u_expected
```

### Why this might work

- Binary rewards gave gap=9-10 alone (too much variance at q=35)
- Expected utility gives gap=5.3 (too little variance)
- A curriculum could use binary to get close, then expected to fine-tune
- The existing `use_binary_rewards` flag makes this trivial to implement

### Implementation

Add to runner:
```python
parser.add_argument("--reward-curriculum", action="store_true")
parser.add_argument("--binary-fraction", type=float, default=0.5,
                    help="Fraction of training to use binary rewards")
```

### Risk

Low implementation risk. Unclear if this is better than Proposal 1. The binary reward mode performed much worse in isolation (gap=9-10), which suggests the noise may overwhelm the signal even in a curriculum.

### Alternative: Binary rewards with variance reduction

Instead of raw binary rewards, use binary rewards but subtract an action-independent baseline:

```python
# Binary reward with control variate
u_binary = (w_H if won else w_L) - k * e_i^2
control = w_L + (1/3) * w_gap - k * e_i^2  # expected utility at symmetric play
u_shaped = u_binary - control + control  # same expected value, lower variance
# Equivalently: u_shaped = (w_gap * (1{won} - 1/3)) + control
```

This preserves the binary signal direction but centers it, potentially giving both high signal AND manageable variance.

---

## Proposal 4: Opponent-Marginal Advantage

### Idea

Instead of computing V(s) from all experiences, compute separate value functions for different opponent action bins:

```python
# Discretize opponent effort levels into bins
opp_bin = discretize(mean(e_j, e_k), n_bins=10)

# Separate value estimate per opponent bin
V_i(s, opp_bin) = learned value conditioned on opponent behavior range

# Advantage
A_i = R_i - V_i(s, opp_bin)
```

This is a simpler version of Proposal 2 that doesn't require a new network — just stratified value estimation.

### Implementation

Use a value head with opponent-effort conditioning via a lookup table or small embedding.

### Risk

Medium. Binning opponent efforts is crude. May not help enough if the fundamental issue is the reward signal strength rather than value estimation.

---

## Proposal 5: Increased Batch + Lower Learning Rate

### Idea

If the gradient signal is 27% weaker (or 10x noisier), compensate by:
- 4x larger batch: `steps_per_update = 16384` (→ 49152 transitions with 3 players)
- Proportionally lower lr: `lr = 1e-4`
- More epochs: `update_epochs = 10`

### Why it might help

With 4x more data per update, the advantage estimates have 2x lower standard error. Combined with lower lr, each update is more reliable even if smaller.

### Why it might not

The previous `spu2731` experiment (reducing steps_per_update) made things worse, but going in the **opposite** direction (larger batch) was not tested. The relationship may be nonlinear.

### Implementation

Trivial — just change config values. Test with:
```bash
python run/run_three_players.py --method ppo --q 35 --seed 42 \
    --steps-per-update 16384 --minibatch-size 4096 --update-epochs 10 \
    --episodes 24576000
```

### Risk

Low risk, low expected reward. Likely to reduce gap by 1-2 units at most.

---

## Proposal 6: Policy Gradient with Analytical Gradient Injection

### Idea

Exploit the fact that the environment has a known, differentiable reward function. Instead of relying purely on PPO's advantage estimation, inject the analytical policy gradient as a regularization term:

```python
# Analytical gradient: ∂u/∂e = (w_H-w_L) * ∂p/∂e - 2ke
# This is computable from utils/prob.py

# During PPO update, add a supervised term:
policy_mean = alpha / (alpha + beta) * (high - low) + low
analytical_grad = compute_expected_utility_gradient(policy_mean, policy_mean, policy_mean, q, k, w_h, w_l)

# Loss term that pushes policy mean in the direction of the analytical gradient
grad_injection_loss = -grad_injection_coef * (policy_mean * analytical_grad.detach())
loss = ppo_loss + grad_injection_loss
```

### Why this might work

This directly provides the gradient signal that PPO cannot estimate. It's like a "hint" that says "the reward function slope at your current effort points upward."

### Why this might NOT work

- It requires computing the analytical gradient, which may feel like "cheating"
- It doesn't compute the gradient from the RL objective — it computes it from the game theory objective
- For a paper about RL learning in games, this may undermine the contribution

### Risk

High philosophical risk (is this still RL?), but low technical risk. Should be highly effective if the goal is just convergence, but may not be appropriate for the paper's narrative.

---

## Recommended Execution Order

### Phase 1: Proposal 1 (Pairwise Decomposition)
- Highest expected impact with lowest risk
- Pure reward shaping, no network changes
- Preserves equilibrium by construction
- If gap drops to <2.0: declare success, run 5-seed validation

### Phase 2 (if Phase 1 insufficient): Proposal 3 variant (Binary with control variate)
- Combine binary rewards with variance reduction
- Tests whether high-variance-but-directional signal helps

### Phase 3 (if still insufficient): Proposal 2 Option A (Analytical counterfactual)
- More invasive but principled
- Removes opponent variance from advantage estimate

### Each phase: single seed first (seed=42, q=35), then 5-seed validation if gap < 2.0.

---

## Implementation Notes

### Testing a new reward mode

1. Add the reward mode to `envs/three_players_env.py`:
   - Add `reward_mode` parameter to `__init__`
   - Add branch in `step()`
   - Verify with `python -c "from envs.three_players_env import ...; ..."`

2. Add CLI flag to `run/run_three_players.py`:
   - `--reward-mode {expected, binary, pairwise_decomposed, hybrid}`
   - Pass through to env constructor

3. Run experiment:
   ```bash
   tmux new-session -d -s 3p_pairwise \
     "python run/run_three_players.py --method ppo --q 35 --seed 42 \
      --reward-mode pairwise_decomposed --episodes 6144000"
   ```

4. Check results:
   ```python
   import json
   with open("results/three_players/convergence/ppo_3p_q35.0_seed42_pairwise_convergence.json") as f:
       data = json.load(f)
   print(f"Final effort: {data['policy_mean_effort'][-1]:.2f}")
   print(f"Gap: {abs(data['policy_mean_effort'][-1] - 62.5):.2f}")
   ```

### Verifying equilibrium preservation

For any reward shaping, verify that the gradient solver still converges:
```bash
python run/run_three_players.py --method gradient --q 35 --reward-mode pairwise_decomposed
```
The gradient solver should still reach gap < 0.5.
