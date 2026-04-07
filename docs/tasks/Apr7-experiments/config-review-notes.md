# Experiment Configuration Review Notes (2026-04-07)

## Background

Review of `Experiment configuration uniform.md` — the revised experiment parameter table.
User adjusted the table to maintain q-value consistency across experiments, which constrains
the tuning space and results in some effort values being close together.

---

## Summary of Changes vs. Current Codebase

### 1. q-grid unification

| | Old | New |
|---|---|---|
| All single-stage | {25, 40, 55} (3 values) | {30, 35, 40, 45, 50, 55} (6 values, step=5) |
| Two-stage | {25, 40, 55} | {25, 40, 55} (unchanged) |

Starting value raised from 25 to 30 to avoid q=25 participation constraint violation
(old k=0.0004 had q_min~33.07; new k=0.0005 has q_min~29.58, so q=30 just passes).

### 2. k parameter changes

| Experiment | Old k | New k | Change |
|---|---|---|---|
| Two players Set 1 | 0.0004 | 0.0005 | +25% |
| Two players Set 2 | — | 0.00065 | new parameter set |
| Three players | 0.0004 | 0.001 | x2.5 |
| Different cost (k1) | 0.0004 | 0.0004 | unchanged |
| Different cost (k2) | 0.00055 | 0.00055 | unchanged |
| Different ability | 0.0004 | 0.0005 | +25% |
| Two-stage | 0.0004 | 0.0004 | unchanged |

Core purpose of k increase: lower equilibrium effort so q=30 satisfies participation constraint (EU >= w_L).

### 3. Prize structure changes

| Experiment | Old (w_h, w_l) | New (w_h, w_l) | Change |
|---|---|---|---|
| Different cost | (6.5, 3.0) | (8, 5.5) | gap 3.5 -> 2.5 |
| Two players Set 2 | — | (8, 4) | new (from existing wh8_wl4 variant) |
| All others | (6.5, 3.0) | (6.5, 3.0) | unchanged |

### 4. New additions

- **Two players split into two Sets**: Set 1 (k=0.0005, w_h=6.5/w_l=3) + Set 2 (k=0.00065, w_h=8/w_l=4)
- **Two-stage adds REINFORCE** method alongside Gradient and PPO

---

## First-Principles Analysis of Issues

### Issue 1: q=30 is the weakest point across all single-stage experiments

SOC condition `(w_H - w_L) - 8kq^2` evaluates to **-0.10** for Set 1 / Different Ability (k=0.0005),
and the 3-player SOC `(w_H - w_L) - 4kq^2` also gives **-0.10** (k=0.001).
This is not a copy-paste error — it's because `8 * k_2p = 4 * k_3p = 0.004`.

**-0.10 means the equilibrium payoff surface is nearly flat.** Consequences for PPO:

- Extremely weak gradient signal near NE -> slow, noisy convergence
- Exploitability evaluation unreliable (best response nearly indifferent)
- This data point risks becoming a reviewer target rather than algorithmic evidence

**Participation constraint is razor-thin at q=30:**

| Experiment | EU at q=30 | w_l | Margin |
|---|---|---|---|
| Set 1 | 3.05 | 3.0 | **0.05** |
| Different Ability (P2) | 3.04 | 3.0 | **0.04** |
| Different Cost (P2) | 5.80 | 5.5 | 0.30 |
| Three players | 3.32 | 3.0 | 0.32 |
| Set 2 | 4.29 | 4.0 | 0.29 |

Set 1 and Different Ability have margins < 0.05. Any effort overshoot during training
pushes EU below w_l, making the agent theoretically indifferent to participating.

### Issue 2: Different Cost asymmetry signal weakened

Prize gap reduced from 3.5 to 2.5 (29% decrease).

- e1 - e2 gap is smaller, harder to distinguish from training noise
- Core value of this experiment is demonstrating PPO learns **asymmetric equilibrium**
- With smaller gap, the story is less compelling

| q | e1 | e2 | gap | gap/e1 |
|---|---|---|---|---|
| 30 | 42.11 | 30.63 | 11.48 | 27% |
| 55 | 26.54 | 19.30 | 7.24 | 27% |

Proportional gap is constant (~27%), but absolute values are small, especially at q>=50.

### Issue 3: High-q effort compression

Due to e* ~ 1/q, equal-spaced q grid produces diminishing effort differences at high q:

| q step | Set 1 delta | 3p delta |
|---|---|---|
| 30->35 | 8.3 | 4.2 |
| 35->40 | 6.3 | 3.1 |
| 45->50 | 3.9 | 1.9 |
| 50->55 | **3.2** | **1.6** |

Three-player q=50 vs q=55 differs by only 1.6 effort units. Typical PPO training noise
may exceed this — the two points become experimentally indistinguishable.

### Issue 4: Three-player action space mismatch

effort_range = [0, 200], but target efforts are only 15.9-29.2 (8%-15% of the space).
PPO must precisely locate a small target in a very wide action space.

### Issue 5: Two-stage REINFORCE addition

REINFORCE in bandit-like settings typically has high variance. Need to consider:
- Whether there's a clear theoretical motivation
- Additional hyperparameter tuning cost
- Significantly longer training time may be needed

---

## Core Trade-off

**q-grid consistency vs. per-point signal quality.**

Unified {30, 35, ..., 55} grid enables cross-experiment comparison, but:
- q=30 is at the theoretical validity edge in multiple experiments
- q=50/55 produce near-identical efforts in some experiments

The fundamental tension: the paper needs each data point to be a convincing
demonstration of NE recovery, not a borderline case that invites doubt.
