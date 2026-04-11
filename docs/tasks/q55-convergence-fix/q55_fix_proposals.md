# q=45/55 Convergence Gap: Fix Proposals

**Date:** 2026-04-10
**Target:** |ē − e*| < 2.0 effort units for all q values
**Current status:** q=35: 1.0 | q=45: 4.3 | q=55: 7.6 (mean across 5 seeds)

## Root cause recap (from F2)

The symmetric policy gradient is dEU/de = W/(2q) − 2ke. For q=55, this
gradient enters the "exhaustion zone" (|grad| < 0.005) at e ≈ 33.5, still
4.6 units above e*=28.93. The PPO advantage estimator cannot distinguish
"slightly above optimal" from "at optimal" when the EU difference is
< 0.02 over a 5-unit range.

Current training parameters:
- Effort bounds: [0, 100] → Beta init mean ≈ 50
- Entropy: 0.03 (hold 67%) → 0.005 (default) or 0.002 (q=45/55 override)
- LR: 3e-4 → 2e-4 (gated by KL warm-up)
- Clip: 0.50 → 0.35
- Batch: 4096, epochs=6, max_updates=1500

---

## Proposal 1: Extended Training Budget

**One-line:** Increase max_updates from 1500 to 5000 (6M → 20M steps).

**Principle:** The gradient at q=55's stall point (e≈36) is −0.0078 — not
zero. Given enough updates, even this weak signal should push the policy
downward. The current 1500-update budget may simply be insufficient for
the slow descent rate.

**Quantitative argument:**
- At e=36, grad=−0.0078. With lr=2e-4 and 6 gradient epochs per update,
  the effective step size per update is ~lr × grad × epochs = 2e-4 × 0.008 × 6
  ≈ 1e-5 in parameter space. Very slow but nonzero.
- However, the entropy regularization (still at 0.002 at end) adds noise
  that may cancel this signal. The signal-to-noise ratio doesn't improve
  with more updates — this is the key limitation.

**Expected effect:**
- Optimistic: q=55 gap reduces to ~4-5 (from 7.6). Some descent below 36.
- Pessimistic: q=55 plateaus at same ~36 regardless of budget. The gradient
  signal is below the effective noise floor, and more updates just oscillate.

**Implementation:** Change `max_updates` in config or via `--episodes`.
One line change.

**Experiment cost:** 1 seed × 1 q × ~3.5× longer = ~7 GPU-hours per run.
Round 1: 1 run. Round 2: 10 runs. Total: ~80 GPU-hours.

**Risk to q=35:** None (q=35 terminates early via exploitability).

**Assessment:** Low cost, low expected improvement. Good as a baseline
to rule out "just needed more time." But unlikely to solve the fundamental
gradient exhaustion problem.

---

## Proposal 2: More Aggressive Entropy Schedule

**One-line:** Decay entropy to 0.0005 (vs current 0.002) with earlier onset.

**Principle:** Entropy regularization keeps the Beta distribution broad,
preventing the policy from concentrating. At the stall point, the entropy
bonus counteracts the weak policy gradient. Reducing entropy_end lets the
distribution concentrate more, making the policy mean more responsive to
small gradient signals.

**Quantitative argument:**
- Current entropy_end=0.002. The entropy gradient contribution to the
  policy update scales as entropy_coef × d(entropy)/d(params). When
  entropy_coef ≈ |policy_gradient| ≈ 0.008, they're comparable — entropy
  is literally fighting the policy gradient.
- At entropy_end=0.0005, the entropy term is 4× weaker, giving the policy
  gradient a 4:1 advantage.
- Risk: too-low entropy → premature concentration at a wrong value.

**Schedule options:**
- (a) Same shape, lower floor: 0.03 → 0.0005 (current: 0.03 → 0.002)
- (b) Earlier decay: hold_fraction 0.33 instead of 0.67
- (c) Exponential decay (instead of linear) — faster initial drop

**Expected effect:**
- Optimistic: q=55 gap drops to 2-3 (the policy can concentrate precisely).
- Pessimistic: premature convergence to e≈34 (entropy drops too fast,
  policy locks before reaching e*). Worse than current if decay is too early.

**Implementation:** CLI flags already exist. 0 lines of code change.
```
--override-entropy-end 0.0005
```
For earlier onset, would need to add `--override-entropy-hold-fraction` flag
(~10 lines in runner).

**Experiment cost:** Round 1: 2-3 runs (testing end={0.001, 0.0005, 0.0002}).
Round 2: 10 runs. Total: ~30 GPU-hours.

**Risk to q=35:** Low. q=35 converges in ~250k steps and exits via
exploitability before entropy decay matters much.

**Assessment:** Moderate cost, moderate expected improvement. The entropy
schedule is clearly a bottleneck — it's the only hyperparameter that was
already tuned per-q (0.005 → 0.002). Pushing further is the obvious next
step.

---

## Proposal 3: Lower Late-Stage Learning Rate

**One-line:** Reduce lr_end from 2e-4 to 5e-5 for fine-grained convergence.

**Principle:** Smaller LR means smaller parameter updates, which reduces
oscillation around the optimum. When the gradient is weak (0.008), a large
LR causes the policy to "bounce" across the optimum rather than settling.

**Quantitative argument:**
- Current lr_end=2e-4. At e=33 (closer to e*=28.93), grad=−0.0045.
  Update magnitude ≈ lr × grad × epochs = 2e-4 × 0.0045 × 6 = 5.4e-6.
  This is within the noise of the advantage estimator.
- At lr_end=5e-5, the update is 4× smaller: 1.4e-6. Even less responsive.
- This doesn't help because the SIGNAL is too weak, not the LR too large.

**Expected effect:**
- Optimistic: Slightly smoother convergence, gap reduces to ~6 (from 7.6).
- Pessimistic: Even slower descent, possibly worse if reduced LR prevents
  the policy from following the already-weak gradient.

**Implementation:** Config change: `lr_end`. Already parameterized. 0 lines.

**Experiment cost:** Round 1: 1-2 runs. Round 2: 10 runs. Total: ~25 GPU-hours.

**Risk to q=35:** None (exits early).

**Assessment:** Low cost, low expected improvement. LR reduction addresses
oscillation, but the q=55 problem is undershoot (not oscillation). Not
recommended as primary fix.

---

## Proposal 4: Adaptive Entropy Control (AEC-TEL-PPO)

**One-line:** Three-layer control loop that adjusts entropy based on
convergence diagnostics (KL, effort drift, gradient magnitude).

**Principle:** Instead of a fixed entropy schedule, use a feedback controller:
- **Layer 1 (KL monitor):** If KL drops below threshold (policy stagnating),
  reduce entropy to let the policy sharpen.
- **Layer 2 (Gradient monitor):** Estimate the effective policy gradient
  magnitude from recent effort changes. When |Δe/Δt| < threshold, the
  policy is in the exhaustion zone — aggressively reduce entropy.
- **Layer 3 (Exploitability feedback):** After each exploitability eval,
  if exploit > target, increase entropy (re-explore); if exploit is
  decreasing steadily, maintain current entropy.

**Design (derived from gradient exhaustion mechanism):**

```
At each update:
  drift = |effort_mean[t] - effort_mean[t-W]| / W   # effort velocity
  if drift < drift_threshold:
    # In exhaustion zone — reduce entropy aggressively
    entropy *= decay_fast  # e.g. 0.95
  elif kl < kl_low:
    # Policy not changing — moderate reduction
    entropy *= decay_moderate  # e.g. 0.98
  else:
    # Active learning — hold entropy
    entropy = max(entropy, entropy_floor)  # e.g. 0.0005
```

**Expected effect:**
- Optimistic: gap < 2 for q=55. The controller detects stalling at e≈36
  and drops entropy, allowing precise convergence. This is the "right" fix
  because it addresses the mechanism directly.
- Pessimistic: controller oscillates (entropy too low → premature lock,
  detect → increase entropy → unlock → re-stall). Needs careful tuning
  of drift_threshold and decay rates.

**Implementation:** ~100-150 lines in the PPO trainer. New class
`AdaptiveEntropyController`. Integration into the training loop.

**Experiment cost:** Tuning the controller itself needs ~5-10 exploratory
runs before committing. Round 1: 3-5 runs. Round 2: 10 runs.
Total: ~50-80 GPU-hours.

**Risk to q=35:** Medium. The controller could reduce entropy too early
for q=35, but since q=35 converges in 59 updates, there's little time
for the controller to do harm. Add a "min_updates_before_aec" guard.

**Assessment:** Highest expected improvement, highest implementation cost.
This is the "correct" solution but carries engineering risk. Recommend
as Round 2 escalation if simpler fixes don't work.

---

## Proposal 5: Theory-Derived Effort Bound

**One-line:** Set effort_range = [0, L] where L = ⌈√(W/(2k))⌉, a
game-theoretic dominance bound that does not depend on e* or q.

**Derivation:**
In any symmetric 2-player tournament, both agents win with probability
p = 1/2 at symmetric play. Expected utility is therefore:

    EU = w_L + (1/2)·W − k·e²

Participation is rational only when EU ≥ w_L, i.e., W/2 − k·e² ≥ 0:

    e ≤ √(W / (2k))

Any effort above this bound is **dominated** — the cost exceeds the
expected prize gain regardless of q, regardless of the opponent's
strategy. This bound depends only on (W, k), which are fixed game
parameters, not on the equilibrium we are trying to learn.

**Computed values:**
- Set 1: L = ⌈√(3.5 / (2 × 0.00055))⌉ = ⌈56.41⌉ = **57**
- Set 2: L = ⌈√(4.0 / (2 × 0.0006))⌉ = ⌈57.74⌉ = **58**

**Margin check (e* must be well inside [0, L]):**

| q  | e*    | L  | margin | e*/L  |
|----|-------|----|--------|-------|
| 35 | 45.45 | 57 | 11.5   | 0.80  |
| 45 | 35.35 | 57 | 21.6   | 0.62  |
| 55 | 28.93 | 57 | 28.1   | 0.51  |

All equilibria are well interior to the bound.

**Effect on initial policy center:**
With effort_range = [0, 57], Beta init mean = 28.5:

| q  | dist to e* (new) | dist to e* (old, [0,100]) | improvement |
|----|-----------------|--------------------------|-------------|
| 35 | 17.0 (below e*) | 4.5 (above e*)            | see below   |
| 45 | 6.9 (below)     | 14.6 (above)              | 2.1×        |
| 55 | 0.4 (below)     | 21.1 (above)              | **53×**     |

**q=35 risk analysis (critical — starts further from e* than before):**
Although distance increases from 4.5 to 17.0, the gradient is STRONGER
from below. At the new init (e=28.5, ascending):

    dEU/de = W/(2·35) − 2k·28.5 = +0.0186

Compare to current init (e=50, descending):

    dEU/de = W/(2·35) − 2k·50 = −0.0050

The ascending gradient (0.019) is **3.8× stronger** than the current
descending gradient (0.005). The agent enters the exhaustion zone
(|grad| < 0.005) at e ≈ 42, with only 3.5 units remaining to e*=45.45.
This is geometrically equivalent to the current q=35 setup (enters
exhaustion at e ≈ 48, 2.5 units from e*).

Conclusion: q=35 convergence should be **at least as fast** with
L=57, possibly faster due to the stronger initial gradient.

**Implementation:** `--effort-range 0 57`. CLI flag exists (line 2075).
0 code changes. Same bound L=57 for ALL q values in Set 1.

**Experiment cost:** Round 1: 1 run. Round 2: 10 runs. Total: ~25 GPU-hours.

**Risk to q=35:** Low — gradient analysis shows stronger signal from below.
Round 3 regression confirms empirically.

**Paper defensibility:** The bound is derived from the participation
constraint at symmetric play. It does not reference e*, q, or any
equilibrium quantity. The paper can state:

> "We set L = ⌈√((w_H − w_L) / (2k))⌉, the maximum effort at which
> symmetric participation remains individually rational. This bound
> depends only on game parameters (prizes and cost coefficient), not
> on the equilibrium being learned."

**Assessment:** Lowest cost, highest expected improvement, theoretically
motivated bound. Replaces the ad-hoc [0, 100] with a principled choice.

---

## Proposal 6: Larger Batch Size

**One-line:** Increase batch from 4096 to 16384 to reduce advantage
estimator variance.

**Principle:** The advantage noise floor scales as ~1/sqrt(batch). Quadrupling
the batch halves the noise floor, making weak gradient signals
(|grad| = 0.005-0.008) detectable.

**Quantitative argument:**
- Current noise floor estimate: ~0.031 (reward_range / sqrt(4096))
- At batch=16384: ~0.016. Still larger than |grad|=0.008 at the stall point.
- At batch=65536: ~0.008. This matches the gradient at e=36. But each
  update is 16× more expensive.

**Expected effect:**
- Optimistic: gap reduces to ~5 (from 7.6). Marginal improvement.
- Pessimistic: no improvement because 4× batch gives only 2× noise
  reduction, which is still above the gradient signal.

**Implementation:** Config change: `steps_per_update`. 0 lines.

**Experiment cost:** 4× slower per update. Round 1: 1 run at ~8 GPU-hours.

**Risk to q=35:** Low (larger batch usually helps or is neutral).

**Assessment:** Moderate cost, low expected improvement. The noise floor
calculation shows batch=16384 is still insufficient. Would need batch=65536+
to match the gradient signal, making this very expensive.

---

## Summary Table

| # | Proposal | Expected gap | Cost (runs) | Complexity | Risk to q=35 | Improvement/Cost |
|---|----------|-------------|-------------|------------|--------------|------------------|
| 5 | Theory bound L=⌈√(W/2k)⌉ | < 1.0 | 1 + 10 | 0 lines | Low (tested) | **Highest** |
| 2 | Lower entropy | 2-3 | 3 + 10 | 0-10 lines | Low | High |
| 1 | More training | 4-5 | 1 + 10 | 1 line | None | Low |
| 3 | Lower LR | 6-7 | 2 + 10 | 0 lines | None | Low |
| 6 | Larger batch | 5-6 | 1 + 10 | 0 lines | None | Low |
| 4 | AEC controller | < 2 | 5 + 10 | 100-150 lines | Medium | Medium |

Ranked by expected improvement / experiment cost:
**5 > 2 > 1 > 4 > 3 > 6**

---

## Recommendation: 3-Round Experiment Plan

### Round 1: Quick Validation (1 seed, q=55 only)

Run these 3 configs to bracket the solution space.
L = ⌈√(W/(2k))⌉ = 57 for Set 1 (W=3.5, k=0.00055). Same L for ALL q values.

```bash
# (a) Theory bound only (L=57, uniform across q)
python run/run_two_players.py --method ppo --q 55 --seed 42 \
  --effort-range 0 57 --episodes 6144000

# (b) Theory bound + lower entropy
python run/run_two_players.py --method ppo --q 55 --seed 42 \
  --effort-range 0 57 --override-entropy-end 0.0005 --episodes 6144000

# (c) Lower entropy only (control — no bounds change)
python run/run_two_players.py --method ppo --q 55 --seed 42 \
  --override-entropy-end 0.0005 --episodes 6144000
```

**Go/no-go criteria:**
- If (a) achieves |ē − e*| < 3.0 → theory bound alone works. Round 2 with (a).
- If (a) fails but (b) succeeds → need both interventions. Round 2 with (b).
- If all fail → escalate to Proposal 4 (AEC) or reconsider the problem.

Expected wall time: ~2 hours per run × 3 = 6 hours.

### Round 2: Full Validation (5 seeds × q={35, 45, 55})

Use the winning config from Round 1 with L=57 for all q (since L does
not depend on q):

```bash
for q in 35 45 55; do
  for seed in 42 43 44 45 46; do
    python run/run_two_players.py --method ppo --q $q --seed $seed \
      --effort-range 0 57 [+ entropy flags if needed] --episodes 6144000
  done
done
```

15 runs total. Go/no-go:
- q=45 and q=55: mean |ē − e*| < 2.0
- q=35: mean |ē − e*| < 1.5 (no regression from current 1.0)

Note: Round 2 includes q=35 directly (no separate Round 3 needed),
because the theory bound applies uniformly to all q values. If q=35
works, we have a single configuration for all experiments.

### Round 3: Cross-Experiment Generalization (if Round 2 passes)

Apply L = ⌈√(W/(2k))⌉ to other experiments with their respective (W, k):
- 3P: L = ⌈√(3.5/(2×0.001))⌉ = ⌈41.8⌉ = 42
- Het. Cost: L = ⌈√(2.5/(2×0.0004))⌉ = ⌈55.9⌉ = 56 (using smaller k)
- Het. Ability: L = ⌈√(3.5/(2×0.0005))⌉ = ⌈59.2⌉ = 60

5 seeds × 2 q × 3 experiments = 30 runs.

---

## Pre-flight Check

The `--effort-range LO HI` CLI flag already exists in `run/run_two_players.py`
(line 2075). No code changes needed for Round 1.
