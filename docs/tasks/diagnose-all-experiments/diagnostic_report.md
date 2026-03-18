# Diagnostic Report: three_players, different_cost, different_ability

Generated from `tools/diagnose_all.py`

## 0. Probability Cross-Validation (prob.py vs independent)

| Experiment | Test | prob.py | Independent | Match? |
|---|---|---|---|---|
| two_players | p_from_diff(d=0.0, q=25.0) | 0.500000 | 0.500000 | YES |
| two_players | p_from_diff(d=10.0, q=40.0) | 0.617188 | 0.617188 | YES |
| two_players | p_from_diff(d=-30.0, q=55.0) | 0.264463 | 0.264463 | YES |
| two_players | p_from_diff(d=87.5, q=25.0) | 1.000000 | 1.000000 | YES |
| three_players | win_prob_3p(50,50,50,q=40) | 0.333333 | 0.333383 | YES |
| three_players | win_prob_3p(0,87.5,87.5,q=25) | 0.000000 | 0.000000 | YES |
| three_players | win_prob_3p(54.7,54.7,54.7,q=40) | 0.333333 | 0.333383 | YES |
| different_ability | p_win_da(e1=50,e2=50,l=10/5,q=40) | 0.560547 | 0.560547 | YES |
| different_ability | p_win_da(e1=0,e2=50,l=10/5,q=40) | 0.095703 | 0.095703 | YES |
| different_ability | p_win_da(e1=78.75,e2=78.75,l=10/5,q=25) | 0.595000 | 0.595000 | YES |

---
## three_players

### 1. Participation Constraint

| q | Player | e* | EU(e*) | EU(0) | P(win\|0) | Dev gain | Valid? |
|---|--------|-----|--------|-------|-----------|---------|--------|
| 25 | symmetric | 87.50 | 1.1042 | 3.0000 | 0.000000 | 1.8958 | **NO** |
| 40 | symmetric | 54.69 | 2.9704 | 3.0370 | 0.010564 | 0.0666 | **NO** |
| 55 | symmetric | 39.77 | 3.5339 | 3.3037 | 0.086760 | -0.2303 | YES |

### 2. Convergence Status

| q | seed | stop | effort | e* | gap | updates |
|---|------|------|--------|-----|-----|---------|
| 25 | 0 | max_updates | 81.46 | 87.50 | 6.04 | 1500 |
| 25 | 0 | max_updates | 81.27 | 87.50 | 6.23 | 1500 |
| 25 | 0 | max_updates | 64.66 | 87.50 | 22.84 | 1500 |
| 25 | 0 | max_updates | 81.34 | 87.50 | 6.16 | 1500 |
| 25 | 0 | max_updates | 81.28 | 87.50 | 6.22 | 1500 |
| 40 | 0 | exploitability | 45.60 | 54.69 | 9.08 | 79 |
| 40 | 0 | unknown | 54.57 | 54.69 | 0.11 | 500 |
| 40 | 0 | exploitability | 46.98 | 54.69 | 7.71 | 93 |
| 55 | 0 | exploitability | 35.26 | 39.77 | 4.51 | 64 |
| 55 | 0 | exploitability | 37.08 | 39.77 | 2.69 | 72 |
| 55 | 0 | exploitability | 37.18 | 39.77 | 2.59 | 82 |

#### Summary per q

| q | seeds | converged | mean_gap |
|---|-------|-----------|----------|
| 25 | 5 | 0/5 | 9.50 |
| 40 | 3 | 2/3 | 5.64 |
| 55 | 3 | 3/3 | 3.26 |

### 3. Data Integrity

- **ppo_3p_q40.0_seed42_baseline_convergence.json**: Truncated: 500/1500 updates, stop_reason=unknown
- **ppo_3p_q40.0_seed42_baseline_convergence.json**: Missing stop_reason field

### 4. Theory Consistency

| q | Player | JSON | theory.py | Formula | Consistent? |
|---|--------|------|-----------|---------|-------------|
| 25 | symmetric | 87.5000 | 87.5000 | 87.5000 | YES |
| 40 | symmetric | 54.6875 | 54.6875 | 54.6875 | YES |
| 55 | symmetric | 39.7727 | 39.7727 | 39.7727 | YES |

---
## different_cost

### 1. Participation Constraint

| q | Player | e* | EU(e*) | EU(0) | P(win\|0) | Dev gain | Valid? |
|---|--------|-----|--------|-------|-----------|---------|--------|
| 25 | P1 (k=0.0004) | 59.23 | 4.2948 | 3.0336 | 0.009586 | -1.2612 | YES |
| 25 | P2 (k=0.00055) | 43.08 | 2.7813 | 3.0000 | 0.000000 | 0.2187 | **NO** |
| 40 | P1 (k=0.0004) | 46.09 | 4.4069 | 3.5907 | 0.168759 | -0.8163 | YES |
| 40 | P2 (k=0.00055) | 33.52 | 3.6251 | 3.3143 | 0.089814 | -0.3108 | YES |
| 55 | P1 (k=0.0004) | 36.20 | 4.5258 | 4.0125 | 0.289289 | -0.5133 | YES |
| 55 | P2 (k=0.00055) | 26.33 | 4.0687 | 3.7876 | 0.225043 | -0.2810 | YES |

### 2. Convergence Status

| q | seed | stop | e1 | e1* | gap1 | e2 | e2* | gap2 | exploit | streak |
|---|------|------|-----|------|------|-----|------|------|---------|--------|
| 25 | 42 | max_updates | 55.97 | 59.23 | 3.26 | 38.26 | 43.08 | 4.82 | 0.1330 | 0 |
| 25 | 123 | max_updates | 58.25 | 59.23 | 0.98 | 40.56 | 43.08 | 2.52 | 0.1234 | 0 |
| 25 | 456 | max_updates | 59.54 | 59.23 | 0.31 | 47.68 | 43.08 | 4.60 | 0.1620 | 0 |
| 25 | 789 | max_updates | 56.01 | 59.23 | 3.22 | 39.96 | 43.08 | 3.12 | 0.0921 | 0 |
| 25 | 1024 | max_updates | 56.47 | 59.23 | 2.76 | 40.71 | 43.08 | 2.37 | 0.0805 | 0 |
| 40 | 42 | exploitability | 44.32 | 46.09 | 1.78 | 31.00 | 33.52 | 2.52 | 0.0260 | 5 |
| 40 | 123 | exploitability | 45.44 | 46.09 | 0.65 | 33.19 | 33.52 | 0.33 | 0.0244 | 5 |
| 40 | 456 | exploitability | 44.30 | 46.09 | 1.80 | 32.19 | 33.52 | 1.33 | 0.0263 | 5 |
| 40 | 789 | exploitability | 47.23 | 46.09 | 1.13 | 32.99 | 33.52 | 0.54 | 0.0276 | 5 |
| 40 | 1024 | exploitability | 45.03 | 46.09 | 1.07 | 32.46 | 33.52 | 1.07 | 0.0245 | 5 |
| 55 | 42 | exploitability | 34.44 | 36.20 | 1.77 | 24.31 | 26.33 | 2.02 | 0.0290 | 5 |
| 55 | 123 | exploitability | 36.07 | 36.20 | 0.14 | 25.99 | 26.33 | 0.34 | 0.0239 | 5 |
| 55 | 456 | exploitability | 35.48 | 36.20 | 0.72 | 25.07 | 26.33 | 1.26 | 0.0243 | 5 |
| 55 | 789 | exploitability | 34.82 | 36.20 | 1.39 | 26.40 | 26.33 | 0.07 | 0.0261 | 5 |
| 55 | 1024 | exploitability | 35.03 | 36.20 | 1.17 | 25.98 | 26.33 | 0.35 | 0.0271 | 5 |

#### Summary per q

| q | seeds | converged | mean_gap |
|---|-------|-----------|----------|
| 25 | 5 | 0/5 | 2.11 |
| 40 | 5 | 5/5 | 1.28 |
| 55 | 5 | 5/5 | 1.04 |

### 3. Data Integrity

No issues found.

### 4. Theory Consistency

| q | Player | JSON | theory.py | Formula | Consistent? |
|---|--------|------|-----------|---------|-------------|
| 25 | P1 (k1) | 59.2308 | 59.2308 | 59.2308 | YES |
| 25 | P2 (k2) | 43.0769 | 43.0769 | 43.0769 | YES |
| 40 | P1 (k1) | 46.0940 | 46.0940 | 46.0940 | YES |
| 40 | P2 (k2) | 33.5229 | 33.5229 | 33.5229 | YES |
| 55 | P1 (k1) | 36.2028 | 36.2028 | 36.2028 | YES |
| 55 | P2 (k2) | 26.3293 | 26.3293 | 26.3293 | YES |

---
## different_ability

### 1. Participation Constraint

| q | Player | e* | EU(e*) | EU(0) | P(win\|0) | Dev gain | Valid? |
|---|--------|-----|--------|-------|-----------|---------|--------|
| 25 | P1 (l=10, stronger) | 78.75 | 2.6019 | 3.0000 | 0.000000 | 0.3981 | **NO** |
| 25 | P2 (l=5, weaker) | 78.75 | 1.9369 | 3.0000 | 0.000000 | 1.0631 | **NO** |
| 40 | P1 (l=10, stronger) | 51.27 | 3.9105 | 3.3111 | 0.088886 | -0.5994 | YES |
| 40 | P2 (l=5, weaker) | 51.27 | 3.4867 | 3.1540 | 0.043995 | -0.3327 | YES |
| 55 | P1 (l=10, stronger) | 37.96 | 4.3289 | 3.8583 | 0.245224 | -0.4707 | YES |
| 55 | P2 (l=5, weaker) | 37.96 | 4.0180 | 3.6499 | 0.185690 | -0.3681 | YES |

### 2. Convergence Status

| q | seed | stop | effort | e* | gap | exploit | streak |
|---|------|------|--------|-----|-----|---------|--------|
| 25 | 42 | max_updates | 75.69 | 78.75 | 3.06 | 0.7466 | 0 |
| 25 | 123 | max_updates | 72.87 | 78.75 | 5.88 | 0.7862 | 0 |
| 25 | 456 | max_updates | 77.08 | 78.75 | 1.67 | 1.0161 | 0 |
| 25 | 789 | max_updates | 74.33 | 78.75 | 4.42 | 0.9012 | 0 |
| 25 | 1024 | max_updates | 74.22 | 78.75 | 4.53 | 0.7425 | 0 |
| 40 | 42 | exploitability | 47.27 | 51.27 | 3.99 | 0.0261 | 5 |
| 40 | 123 | exploitability | 48.85 | 51.27 | 2.42 | 0.0272 | 5 |
| 40 | 456 | exploitability | 49.40 | 51.27 | 1.87 | 0.0264 | 5 |
| 40 | 789 | exploitability | 48.60 | 51.27 | 2.67 | 0.0267 | 5 |
| 40 | 1024 | exploitability | 47.66 | 51.27 | 3.61 | 0.0235 | 5 |
| 55 | 42 | exploitability | 35.51 | 37.96 | 2.46 | 0.0267 | 5 |
| 55 | 123 | exploitability | 37.58 | 37.96 | 0.38 | 0.0265 | 5 |
| 55 | 456 | exploitability | 38.05 | 37.96 | 0.08 | 0.0274 | 5 |
| 55 | 789 | exploitability | 33.98 | 37.96 | 3.99 | 0.0258 | 5 |
| 55 | 1024 | exploitability | 35.91 | 37.96 | 2.06 | 0.0238 | 5 |

#### Summary per q

| q | seeds | converged | mean_gap |
|---|-------|-----------|----------|
| 25 | 5 | 0/5 | 3.91 |
| 40 | 5 | 5/5 | 2.91 |
| 55 | 5 | 5/5 | 1.79 |

### 3. Data Integrity

No issues found.

### 4. Theory Consistency

| q | Player | JSON | theory.py | Formula | Consistent? |
|---|--------|------|-----------|---------|-------------|
| 25 | symmetric | 78.7500 | 78.7500 | 78.7500 | YES |
| 40 | symmetric | 51.2695 | 51.2695 | 51.2695 | YES |
| 55 | symmetric | 37.9649 | 37.9649 | 37.9649 | YES |
