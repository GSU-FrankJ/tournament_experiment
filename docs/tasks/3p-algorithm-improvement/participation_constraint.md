# Participation Constraint in N-Player Tournament Games

## Executive Summary

The symmetric interior Nash equilibrium `e* = (w_H - w_L) / (4qk)` is **not globally valid** for all parameter combinations. When noise `q` is too small relative to the number of players `N`, the equilibrium cost exceeds the expected prize gain, making unilateral deviation to near-zero effort strictly profitable. This invalidates the interior NE as a global equilibrium.

**Validity condition:**

```
q >= q_crit = sqrt(N * w_gap / (16k))
```

With our parameters (`w_H=6.5, w_L=3.0, k=0.0004`):

| N | q_crit | q=25 | q=35 | q=40 | q=55 |
|---|--------|------|------|------|------|
| 2 | 33.07  | FAIL | pass | pass | pass |
| 3 | 40.50  | FAIL | FAIL | FAIL | pass |

This explains why 3-player PPO experiments at q=25/35/40 never converge to the interior equilibrium: **the equilibrium does not exist as a global NE at these parameters**.

---

## 1. Setup

Consider an N-player symmetric tournament:

- Player i chooses effort `e_i >= 0`
- Output: `y_i = e_i + eps_i`, where `eps_i ~ U(-q, q)` i.i.d.
- Highest output wins prize `w_H`; all losers receive `w_L`
- Cost: `C(e_i) = k * e_i^2`
- Expected utility: `EU_i = w_L + p_i(e) * (w_H - w_L) - k * e_i^2`

where `p_i(e)` is the probability that player i has the highest output.

## 2. Interior Equilibrium (FOC)

At a symmetric interior equilibrium, all players choose the same effort `e*`. By symmetry, each player's win probability is `1/N`.

The first-order condition for player i:

```
dEU_i/de_i = w_gap * (dp_i/de_i) - 2k*e_i = 0
```

At symmetric play, the marginal win probability is:

```
dp_i/de_i |_{symmetric} = 1/(2q)    (for any N)
```

This can be verified analytically: at symmetric play, the density of the "gap to beat" at zero is `1/(2q)` regardless of N, because each pairwise noise difference `eps_i - eps_j` has a symmetric triangular distribution on `[-2q, 2q]` with peak density `1/(2q)` at zero.

Solving the FOC:

```
w_gap / (2q) = 2k * e*
e* = w_gap / (4qk)
```

This is a **local** equilibrium — no player can improve by an infinitesimal deviation. But is it **global**?

## 3. The Participation Constraint

At the interior NE, each player's expected utility is:

```
EU_NE = w_L + w_gap/N - k*(e*)^2
      = w_L + w_gap/N - w_gap^2/(16*q^2*k)
```

A player can always deviate to `e = 0` (shirking). With zero effort against opponents playing `e*`, the deviator:
- Pays zero cost
- Still receives at least `w_L` (the loser's prize)
- May also win with some small probability if noise is large enough

At minimum, the deviation utility is:

```
EU_deviate(0) >= w_L
```

For the interior NE to survive, we need at least:

```
EU_NE >= w_L
```

This requires:

```
w_gap/N - w_gap^2/(16*q^2*k) >= 0
w_gap/N >= w_gap^2/(16*q^2*k)
16*q^2*k >= N*w_gap
q^2 >= N*w_gap/(16k)
```

**Participation constraint:**

```
q >= q_crit = sqrt(N * w_gap / (16k))
```

When this fails, the equilibrium cost exceeds the expected prize gain — every player at the NE is worse off than if they simply exerted zero effort and accepted the loser's payoff. The interior equilibrium is self-defeating.

## 4. Numerical Verification

Parameters: `w_H = 6.5`, `w_L = 3.0`, `k = 0.0004`, `w_gap = 3.5`.

### 4.1 Critical q values

| N | q_crit |
|---|--------|
| 2 | sqrt(2 * 3.5 / 0.0064) = sqrt(1093.75) = **33.07** |
| 3 | sqrt(3 * 3.5 / 0.0064) = sqrt(1640.63) = **40.50** |

### 4.2 EU comparison (3-player)

| q | e* | Cost k(e*)^2 | Gain w_gap/3 | EU_NE | vs w_L=3.0 |
|---|------|-------------|-------------|-------|------------|
| 25 | 87.50 | 3.0625 | 1.1667 | 1.104 | **EU_NE << w_L** |
| 35 | 62.50 | 1.5625 | 1.1667 | 2.604 | **EU_NE < w_L** |
| 40 | 54.69 | 1.1963 | 1.1667 | 2.970 | **EU_NE < w_L** (barely) |
| 55 | 39.77 | 0.6327 | 1.1667 | 3.534 | EU_NE > w_L |

### 4.3 Best-response scan (3-player)

For each q, we computed the best response against two opponents playing e*, scanning over e in [0, 200]:

| q | e* (interior NE) | Best response e | BR utility | EU_NE | Globally valid? |
|---|-------------------|-----------------|------------|-------|-----------------|
| 25 | 87.50 | **0.00** | 3.000 | 1.104 | **NO** — shirking dominates |
| 35 | 62.50 | **0.90** | 3.002 | 2.604 | **NO** — shirking dominates |
| 40 | 54.69 | **11.72** | 3.061 | 2.970 | **NO** — partial shirking dominates |
| 55 | 39.77 | **39.76** | 3.534 | 3.534 | **YES** — BR = NE |

At q=55, the best response to opponents playing e*=39.77 is to also play ~39.77 — confirming the interior NE is globally valid. At q=25/35/40, the best response is to deviate far below e*, towards zero.

### 4.4 EU comparison (2-player, for reference)

| q | e* | Cost k(e*)^2 | Gain w_gap/2 | EU_NE | vs w_L | Best response |
|---|------|-------------|-------------|-------|--------|---------------|
| 25 | 87.50 | 3.0625 | 1.7500 | 1.688 | EU < w_L | e=0 (**NOT GLOBAL**) |
| 35 | 62.50 | 1.5625 | 1.7500 | 3.188 | EU > w_L | e=62.5 (GLOBAL) |
| 40 | 54.69 | 1.1963 | 1.7500 | 3.554 | EU > w_L | e=54.7 (GLOBAL) |
| 55 | 39.77 | 0.6327 | 1.7500 | 4.117 | EU > w_L | e=39.8 (GLOBAL) |

Note: 2-player q=25 also fails — `q_crit(N=2) = 33.07 > 25`.

## 5. Intuition

The participation constraint captures a simple economic trade-off:

- **Benefit of participating**: win probability `1/N` times prize gap `w_gap`, yielding expected gain `w_gap/N`
- **Cost of participating**: `k * (e*)^2 = w_gap^2 / (16 q^2 k)`

As `N` increases, the per-player expected gain shrinks (`w_gap/N`), but the equilibrium effort `e*` (and hence cost) stays the same (because the FOC marginal condition is independent of N). Eventually, cost exceeds gain, and rational players prefer to "free-ride" by exerting zero effort.

Higher noise `q` lowers e* (and hence cost), restoring viability. This is why q=55 works for 3-player but q=35/40 do not.

## 6. Implications

### For the paper

1. **3-player results at q=25/35/40 should not be expected to converge to the interior NE**, because that NE is not globally valid. PPO's "failure" here is actually correct behavior — the agent discovers that shirking (or partial shirking) is more profitable.

2. **3-player q=55 is the only valid test case** with our current parameters. PPO does converge there (gap ~2.5 with standard mode + entropy).

3. The gradient descent solver appears to converge at q=35 (gap=0.12), but this is misleading — gradient descent follows **local** gradients and finds the local FOC solution. It does not explore the global best response (e=0).

### Options for the paper

- **(a)** Restrict 3-player experiments to q >= q_crit (i.e., q=55 only). Report the constraint as a theoretical finding.
- **(b)** Change parameters so NE is valid at more q values (e.g., lower k or w_gap, or raise q_list).
- **(c)** Characterize the mixed-strategy or asymmetric equilibrium that exists when q < q_crit.

### For 2-player

The same constraint explains 2-player q=25 anomalies (q_crit=33.07 > 25). All other 2-player results (q=35/40/55) are unaffected.

## 7. Derivation Notes

The constraint `q >= sqrt(N * w_gap / (16k))` is a **necessary** condition (EU_NE >= w_L). It is also empirically **sufficient** for our parameters: the full best-response scan confirms that whenever the constraint passes, the BR coincides with e*. This is because the utility function `EU(e) = w_L + p_win(e, e*, ..., e*) * w_gap - k*e^2` is concave near e* when q is large enough, with no secondary maximum near zero.

However, sufficiency is parameter-dependent. A formal proof of sufficiency would require showing the utility function has a unique global maximum at e* for all q >= q_crit — this amounts to verifying concavity of the best-response function over the full effort range, not just locally.

---

*Verified numerically on 2026-04-05 using `utils/prob.py:win_prob_three_players()` with fine-grained best-response scan (10,001 points over [0, 200]).*
