# Experiment Configuration (04/07/26)

## 1. One Stage — Two Identical Players (Set 1)

k = 0.00055, w_h = 6.5, w_l = 3

| # | q | EU | 2nd order | Cost of effort | Effort |
|---|----|------|-------|------|--------|
| 1 | 35 | 3.61 | -1.89 | 1.14 | 45.45 |
| 2 | 45 | 4.06 | -5.41 | 0.69 | 35.35 |
| 3 | 55 | 4.29 | -9.81 | 0.46 | 28.93 |

## 2. One Stage — Two Identical Players (Set 2)

k = 0.0006, w_h = 8, w_l = 4

| # | q | EU | 2nd order | Cost of effort | Effort |
|---|----|------|--------|------|--------|
| 1 | 35 | 4.64 | -1.88  | 1.36 | 47.62 |
| 2 | 45 | 5.18 | -5.72  | 0.82 | 37.04 |
| 3 | 55 | 5.45 | -10.52 | 0.55 | 30.30 |

## 3. One Stage — Three Identical Players

k = 0.001, w_h = 6.5, w_l = 3

| # | q | EU | 2nd order | Cost of effort | Effort |
|---|----|------|-------|------|-------|
| 1 | 35 | 3.54 | -1.40 | 0.63 | 25.00 |
| 2 | 55 | 3.91 | -8.60 | 0.25 | 15.91 |

## 4. One Stage — Two Players with Different Cost (k₁ < k₂, l₁ = l₂)

k₁ = 0.0004, k₂ = 0.00055, w_h = 8, w_l = 5.5

| # | q | EU₁ | EU₂ | 2nd order k₁ | 2nd order k₂ | c(e₁) | c(e₂) | e₁ | e₂ |
|---|----|------|------|-------|--------|------|------|-------|-------|
| 1 | 35 | 6.51 | 5.99 | -1.42 | -2.89  | 0.58 | 0.42 | 38.03 | 27.66 |
| 2 | 55 | 6.63 | 6.39 | -7.18 | -10.81 | 0.28 | 0.20 | 26.54 | 19.30 |

(Noted value: 1.375)

## 5. One Stage — Two Players with Different Ability (l₁ > l₂, k₁ = k₂)

k = 0.0005, l₁ = 10, l₂ = 5, w_h = 6.5, w_l = 3

| # | q | EU₁ | EU₂ | 2nd order | Cost of effort | Effort |
|---|----|------|------|-------|------|-------|
| 1 | 35 | 3.91 | 3.43 | -1.40 | 1.08 | 46.43 (→50.00) |
| 2 | 55 | 4.29 | 4.13 | -8.60 | 0.46 | 30.37 |

## 6. Two Stage

k = 0.0004, w_h = 6.5, w_l = 3

| # | q | Cost (stage 1) | Cost (stage 2) | Effort stage 1 | Effort stage 2 | Model training |
|---|----|------|------|-------|-------|------|
| 1 | 25 | 1.36 | 1.36 | 58.33 | 58.33 | Gradient / REINFORCE / PPO |
| 2 | 40 | 0.53 | 0.53 | 36.46 | 36.46 | Gradient / REINFORCE / PPO |
| 3 | 55 | 0.28 | 0.28 | 26.52 | 26.52 | Gradient / REINFORCE / PPO |

**Notes:**
- Try to avoid second-order values being too close to 0.
- Effort range is set to [0, 100] for both stages — the values are not that large.
- Record: learning rate, batch size, clip ε, ...
- Charts: (episode, effort), (episode, utility)