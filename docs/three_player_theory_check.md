# Three-Player Theory Check

## 1. Theory layer

### Setup
- n=3 symmetric players, shared k
- Output: y_i = e_i + epsilon_i, epsilon_i ~ Uniform(-q, q) iid
- Player i wins if y_i > y_j AND y_i > y_k
- EU_i = w_L + P_i * (w_H - w_L) - k * e_i^2

### Win probability at symmetric point
Conditioning on epsilon_i:

P_i = integral_{-q}^{q} F(eps + a) * F(eps + b) * (1/(2q)) deps

where a = e_i - e_j, b = e_i - e_k, and F(x) = (x+q)/(2q) clamped to [0,1].

At symmetric point (a=b=0): P_i = integral_0^1 u^2 du = 1/3.

### Derivative at symmetric point
dP_i/de_i = dP_i/da + dP_i/db (since da/de_i = db/de_i = 1)

dP_i/da|_{a=b=0} = integral_{-q}^{q} f(eps) * F(eps) * (1/(2q)) deps
                  = (1/(4q^2)) * integral_{-q}^{q} F(eps) deps
                  = (1/(4q^2)) * q
                  = 1/(4q)

By symmetry dP_i/db = 1/(4q), so dP_i/de_i = 1/(2q).

This is the SAME derivative as the two-player case.

### FOC
W * (1/(2q)) - 2k*e* = 0

e* = W / (4qk)

For W=3.5, q=35, k=0.001: e* = 3.5 / 0.14 = **25.00**

### Normalization does not affect the result
The env normalizes win probs (p1+p2+p3 = 1). At the symmetric point the raw probs already sum to 1, and dS/de_i = 0 by symmetry, so the normalized derivative equals the raw derivative.

## 2. Code layer

### Formula
`utils/theory.py:46-56`: `e_star_three_players(q, w_h, w_l, k)` returns `(w_h - w_l) / (4.0 * q * k)`.

For q=35, w_h=6.5, w_l=3.0, k=0.001: returns 25.00. **Matches Section 1.**

### Usage in runner
`run/run_three_players.py` calls `e_star_three_players()` at lines 484 (gradient), 973 (BR target), 988 (PPO reference), 1142 (convergence eval).

### Config
`config/one_stage_three_players.py:133-138` precomputes `effort = (w_h - w_l) / (4 * k * q) = 25.0`.

## 3. Env layer

### Win probability implementation
`utils/prob.py:150-156`: `win_prob_three_players(e_i, e_j, e_k, q)` uses analytic integration via `_integrate_product`.

`envs/three_players_env.py:60-103`: computes per-player win probs via `win_prob_three_players`, then normalizes `p1 + p2 + p3 = 1`.

### Numerical verification

| check | expected | actual |
|-------|----------|--------|
| P_i(25, 25, 25) | 0.333333 | 0.333333 |
| dP_i/de_i at sym | 1/(2*35) = 0.014286 | 0.014286 |
| dEU/de at e*=25 | 0.0 | 0.000000 |
| EU at e*=25 | w_L + W/3 - k*e*^2 = 3.5417 | 3.5417 |
| win_prob sum (30,25,25) | 1.0 | 1.000000 |

Best-response landscape (opponents at e*=25):

| e_i | EU |
|----:|-------:|
| 10 | 3.4659 |
| 20 | 3.5341 |
| 24 | 3.5414 |
| 25 | 3.5417 |
| 26 | 3.5407 |
| 30 | 3.5162 |
| 40 | 3.3052 |

EU is uniquely maximized at e*=25. Second-order condition satisfied.

## 4. Conclusion

**e* = 25.00 is correct.** The formula, code implementation, and environment are all consistent. The three-player convergence gap (~3 units, 12%) is not caused by a theory or env error.
