# Experiment Configuration & Concentration Control

---

## Part 1: Experiment Configuration (Uniform)

### 1.1 One Stage: Two Identical Players (Set 1)

Parameters: $k = 0.0004$, $w_h = 6.5$, $w_l = 3$

| # | q  | EU   | Second Order | Cost of Effort | Effort | Model Training     |
|---|-----|------|--------------|----------------|--------|--------------------|
| 1 | 25  | 1.69 | 1.5          | 3.06           | 87.5   | Gradient, PPO      |
| 2 | 35  | 3.19 | −0.42        | 1.56           | 62.5   | —                  |
| 3 | 40  | 3.55 | −1.62        | 1.20           | 54.69  | Gradient, PPO      |
| 4 | 55  | 4.12 | −6.18        | 0.63           | 39.77  | Gradient, PPO      |

### 1.2 One Stage: Two Identical Players (Set 2)

Parameters: $k = 0.0005$, $w_h = 8$, $w_l = 4$

| # | q  | EU   | Second Order | Cost of Effort | Effort | Model Training              |
|---|-----|------|--------------|----------------|--------|-----------------------------|
| 1 | 25  | 2.80 | 1.5          | 3.20           | 80.00  | Gradient, REINFORCE         |
| 2 | 35  | 4.37 | −0.9         | 1.63           | 57.14  | PPO                         |
| 3 | 40  | 4.75 | −2.4         | 1.25           | 50.00  | Gradient, REINFORCE, PPO    |
| 4 | 55  | 5.34 | −8.1         | 0.66           | 36.36  | Gradient, REINFORCE, PPO    |

### 1.3 One Stage: Three Identical Players

Parameters: $k = 0.0004$, $w_h = 6.5$, $w_l = 3$

| # | q  | EU   | Second Order | Cost of Effort | Effort | Model Training |
|---|-----|------|--------------|----------------|--------|----------------|
| 1 | 25  | 1.10 | 1.5          | 3.06           | 87.50  | Gradient, PPO  |
| 2 | 35  | 2.60 | −0.42        | 1.56           | 62.50  | —              |
| 3 | 40  | 2.97 | −1.62        | 1.20           | 54.69  | Gradient, PPO  |
| 4 | 55  | 3.53 | −6.18        | 0.63           | 39.77 / 38.60 (PPO) | Gradient, PPO |

### 1.4 One Stage: Two Players with Different Cost ($k_1 < k_2$, $l_1 = l_2$)

Parameters: $k_1 = 0.0004$, $k_2 = 0.00055$, $w_h = 8$, $w_l = 5.5$

| # | q  | EU₁  | EU₂  | Second Order (k₁) | Second Order (k₂) | c(e₁) | c(e₂) | e₁    | e₂    | Model Training |
|---|-----|------|------|--------------------|--------------------|--------|--------|-------|-------|----------------|
| 1 | 35  | 1.01 | 0.49 | −1.42              | −2.89              | 0.58   | 0.42   | 38.03 | 27.66 | Gradient, PPO  |
| 2 | 40  | 1.05 | 0.63 | −2.62              | −4.54              | 0.48   | 0.35   | 34.47 | 25.07 | Gradient, PPO  |
| 3 | 55  | 1.13 | 0.89 | −7.18              | −10.81             | 0.28   | 0.20   | 26.54 | 19.30 | Gradient, PPO  |

### 1.5 One Stage: Two Players with Different Ability ($l_1 > l_2$, $k_1 = k_2$)

Parameters: $k = 0.0004$, $l_1 = 10$, $l_2 = 5$, $w_h = 6.5$, $w_l = 3$

| # | q  | EU₁  | EU₂  | Second Order | Cost of Effort | Effort | Model Training              |
|---|-----|------|------|--------------|----------------|--------|-----------------------------|
| 1 | 35  | 3.64 | 3.16 | −0.42        | 1.35           | 58.04  | Gradient, REINFORCE, PPO    |
| 2 | 40  | 3.70 | 0.49 | −1.62        | 1.05           | 51.27  | Gradient, REINFORCE, PPO    |
| 3 | 55  | 4.17 | 1.02 | −6.18        | 0.58           | 37.96  | Gradient, REINFORCE, PPO    |

### 1.6 Two Stage

Parameters: $k = 0.0004$, $w_h = 6.5$, $w_l = 3$

| # | q  | Cost (Stage 1) | Cost (Stage 2) | Effort (Stage 1) | Effort (Stage 2) | Model Training              |
|---|-----|----------------|----------------|-------------------|-------------------|-----------------------------|
| 1 | 25  | 1.36           | 1.36           | 58.33             | 58.33             | Gradient, REINFORCE, PPO    |
| 2 | 40  | 0.53           | 0.53           | 36.46             | 36.46             | Gradient, REINFORCE, PPO    |
| 3 | 55  | 0.28           | 0.28           | 26.52             | 26.52             | Gradient, REINFORCE, PPO    |

> **Notes (from sheet):**
> - Record: learning rate, batch size, clip ε, …
> - Chart: (episode, effort), (episode, utility)

---

## Part 2: Concentration Control — Prioritization

### Priority 1: Variance Floor (Do This Now)

This is the most practical first step because it targets the failure mechanism most directly. In the $q=55$ investigation, the main problem is not the absence of equilibrium, but the fact that policy concentration becomes too high too early. Once concentration is very large, the policy variance collapses, the two self-play agents produce nearly identical efforts, reward differences become extremely small, and PPO loses directional signal. The investigation shows that gradient direction accuracy falls to about 53–54% in the high-concentration regime at $q=55$.

A variance-floor rule addresses this problem at the right level: instead of trying to guess when concentration should grow, it simply prevents the sampling variance from becoming too small. This makes it the most direct and lowest-risk fix. It is also closely aligned with the existing evidence, since the successful $\text{conc\_max}=1000$ intervention can be interpreted as an implicit way of preserving sufficient policy variance.

**Downsides:** Variance floor is somewhat more "engineering-style" than "theory-style." It is easy to implement and likely to work, but on its own it may look like a robustness patch rather than a full methodological contribution. It also requires choosing a minimum variance threshold, so there is still a tuning decision, although this is likely to be more stable than tuning a separate concentration cap for each $q$.

### Priority 2: Progress-Aware Cap (Do This Next)

This is the best candidate for a more general cross-$q$ method. The core design flaw in the default schedule is that concentration grows only as a function of update count, even though the distance from initialization to equilibrium grows with $q$. That is why $q=35$ and $q=40$ usually converge before the dangerous regime, while $q=55$ often enters the high-concentration regime while still far from equilibrium.

A progress-aware cap addresses exactly this mismatch: concentration would remain low while the agent is still far from the target region, and would only be allowed to sharpen as training makes relative progress toward equilibrium. This gives a strong paper story: the issue is not that $q=55$ is special in itself, but that the schedule should depend on where training is relative to the solution, not just on elapsed updates.

**Downsides:** Harder to define and implement robustly. It requires a progress metric, and depending on how progress is measured, the rule may become noisy or too dependent on problem structure. In the current setting, where theoretical equilibrium is known, this is manageable.

### Priority 3: Signal-Quality-Based Cap (Leave This for Later)

This is the most ambitious idea, but also the most difficult. In principle, it is very attractive because the diagnosed failure mechanism is ultimately about signal quality: once concentration becomes too large, PPO updates are driven by noise rather than meaningful reward differences.

A signal-quality-based cap would adapt concentration directly based on whether the learning signal is still reliable. However, signal quality is much harder to measure robustly than either variance or progress. Advantage variance, reward-difference magnitude, or other proxies can all become small for different reasons, including legitimate late-stage convergence. That means a signal-based controller is more prone to false triggers and harder to tune and explain.

**Verdict:** Promising as an advanced extension or appendix method, especially for versions that don't rely on knowing the theoretical equilibrium — but too high-risk to serve as the main immediate fix for $q=55$.

### Roadmap Summary

1. **Variance Floor** → Stabilize $q=55$ first with variance control (most direct fix for policy-variance collapse).
2. **Progress-Aware Cap** → Generalize the method with progress-aware concentration (fixes the deeper design flaw of update-count-only schedule).
3. **Signal-Quality Cap** → Explore as a more advanced extension later.
