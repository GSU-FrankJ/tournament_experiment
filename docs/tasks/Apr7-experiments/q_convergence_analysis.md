# Why q=35 Converges Much Faster Than q=45 and q=55

**Date:** 2026-04-09
**Scope:** Two-player Set 1 (k=0.00055, w_H=6.5, w_L=3.0), PPO baseline runs

## Empirical Observations

| Metric | q=35 | q=45 | q=55 |
|--------|------|------|------|
| Theoretical e* | 45.45 | 35.35 | 28.93 |
| Mean final effort | 44.55 | 39.62 | 36.51 |
| Mean gap from theory | 3.6% | 12.1% | 26.2% |
| Mean updates to stop | 55 | 793 | 1458 |
| Exploitability pass rate | 5/5 | 5/5 | 1/5 |
| Mean exploit_max | 0.003 | 0.021 | 0.047 |

Note: q=35 uses default entropy_end=0.005; q=45/55 use `--override-entropy-end 0.002`.

## Root Cause Analysis

Three reinforcing factors explain the pattern. All derive from the same
first principle: **higher q flattens the marginal return to effort**.

### Factor 1: Distance from Initial Policy (Dominant)

The Beta policy initializes with mean effort around 47-55 (mid-range of [0, 100]).
The equilibrium targets are:

- q=35: e* = 45.45 — **~5 units from init** (agent barely needs to learn)
- q=45: e* = 35.35 — ~15 units from init
- q=55: e* = 28.93 — **~21 units from init**

This alone accounts for most of the convergence gap. The q=35 agent is
essentially "born" near the Nash equilibrium.

### Factor 2: Flattening Reward Landscape

The Hessian of expected utility at the symmetric NE:

```
d²EU/de² = -(w_H - w_L)/(4q²) - 2k
```

| q  | \|d²EU/de²\| | EU penalty at e*+10 |
|----|-------------|---------------------|
| 35 | 0.001814    | 0.091               |
| 45 | 0.001532    | 0.077               |
| 55 | 0.001389    | 0.070               |

The payoff "bowl" around Nash equilibrium is 30% flatter for q=55 than q=35.
A 10-unit overshoot costs only 0.070 EU for q=55 — roughly 1.6% of total EU
(~4.29). This is a weak signal for PPO to act on.

**Note on the "2nd order" column in the experiment config:** Those values are
(w_H - w_L) - 8kq², which gets more negative with q (e.g., -1.89 at q=35 vs
-9.81 at q=55). This metric captures the SOC margin in scaled units and
confirms the NE is a valid maximum. But for RL convergence, what matters is the
**raw curvature per unit of effort** (d²EU/de²), which is flatter for higher q.

### Factor 3: Vanishing Gradient During Descent

In symmetric self-play, the policy gradient signal is:

```
dEU/de = (w_H - w_L)/(2q) - 2k·e
```

Tracing the gradient as each agent descends from e=50:

| Position | q=35 gradient | q=55 gradient | Notes |
|----------|--------------|--------------|-------|
| e = 50   | -0.005       | -0.023       | q=55 starts stronger |
| e = 40   | -            | -0.012       | halved |
| e = 35   | at target    | -0.007       | q=55 still 6 above target |
| e = 30   | -            | -0.001       | near noise floor |

The q=55 agent follows a strong initial gradient but "runs out of steam."
By e ≈ 36, the gradient (-0.007) approaches the PPO noise floor. The remaining
7 units to e*=28.93 produce almost no reward improvement, so the agent stalls.

For q=35, the gradient is weak throughout (-0.005 at start), but the target is
only 5 units away — the agent arrives before the signal fades.

## EU Landscape Visualization

Expected utility when opponent plays e* (unilateral deviation):

```
q=35 (e*=45.45):
  e=50: EU=3.595  (ΔEU = -0.019)
  e=45: EU=3.614  (≈ optimum)
  e=40: EU=3.608  (ΔEU = -0.006)    ← flat below equilibrium
  e=35: EU=3.593  (ΔEU = -0.021)

q=55 (e*=28.93):
  e=50: EU=3.981  (ΔEU = -0.309)
  e=40: EU=4.205  (ΔEU = -0.085)
  e=35: EU=4.264  (ΔEU = -0.026)    ← almost no penalty
  e=30: EU=4.289  (ΔEU = -0.001)    ← indistinguishable from optimal
  e=25: EU=4.284  (ΔEU = -0.006)
```

For q=55, the EU curve from e=25 to e=35 spans only 0.020 — the agent is
nearly indifferent across a 10-unit range around equilibrium.

## Verification: Not a Code Bug

Gradient descent (numerical optimization) achieves **Excellent** convergence
for all q values, confirming the theory formulas and environment implementation
are correct. The difficulty is specific to PPO as a reinforcement learning
algorithm operating on a flat landscape.

## Conclusion

**This is expected behavior, not a bug.** The pattern follows directly from
tournament theory: high noise (large q) reduces the marginal return to effort,
which simultaneously (a) pushes the equilibrium further from the natural policy
initialization, and (b) flattens the payoff landscape that PPO must navigate.
These effects compound multiplicatively.

Potential mitigations (not necessarily needed for the paper):

1. **Narrower effort bounds for high q** — e.g., [0, 60] instead of [0, 100]
   for q=55, placing the initialization closer to the target
2. **Larger training budget** — q=55 uses 1500 max updates; more may help
3. **Adaptive entropy scheduling** — already partially addressed via
   `--override-entropy-end 0.002` for q=45/55
