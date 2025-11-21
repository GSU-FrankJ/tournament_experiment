# MC-FD Sweep Report

_Generated: 2025-11-21_

## 1. Overview

- **Total runs:** 270  
- **Sigma (σ):** 20.0 for most grid; additional control runs at σ = 15 (legacy)  
- **Effort range:** [0, 100]  
- **Seed:** 42  
- **Recorded columns:** `sigma, delta, eta, num_samples, final_effort, mcfd_iterations, mcfd_tol, mcfd_effort_min, mcfd_effort_max, seed`

Final effort statistics:

| Metric | Value |
| --- | --- |
| Mean | 60.119 |
| Std Dev | 0.663 |
| Min | 58.341 |
| Max | 60.641 |
| Median iterations | 200 |
| Max iterations | 1000 |

## 2. Best & Worst Configurations

| Rank | σ | δ | η | N | tol | Iterations | Final Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Best** | 15.0 | 0.75 | 0.02 | 64 | 1e-3 | 1 | **60.641** |
| **Worst** | 20.0 | 0.75 | 0.10 | 32 | 1e-3 | 1000 | **58.341** |

Notes:
- The best runs use a very small learning rate (η=0.02) and achieve convergence almost immediately.
- The worst cases combine aggressive step sizes (η=0.1) with small batches (N=32), forcing the solver to hit the max iteration cap without reaching a high-effort equilibrium.

## 3. Parameter Insights

### 3.1 Delta (Finite-Difference Step)

| δ | Mean Effort | Best Effort | Worst Effort |
| --- | --- | --- | --- |
| 0.50 | 60.064 | 60.641 | 58.345 |
| 0.75 | 60.086 | 60.641 | 58.341 |
| 1.00 | **60.206** | 60.641 | 58.429 |

> Larger deltas (1.0) slightly improve average effort without hurting stability.

### 3.2 Eta (Gradient Step Size)

| η | Mean Effort | Best Effort | Worst Effort |
| --- | --- | --- | --- |
| 0.02 | **60.542** | 60.641 | 60.156 |
| 0.05 | 60.243 | 60.637 | 59.396 |
| 0.06 | 60.235 | 60.638 | 59.699 |
| 0.10 | 59.572 | 60.635 | **58.341** |

> η=0.02 clearly dominates; η=0.10 consistently drifts to lower efforts.

### 3.3 Monte Carlo Batch Size

| N | Mean Effort | Best Effort | Worst Effort |
| --- | --- | --- | --- |
| 32 | 59.938 | 60.641 | **58.341** |
| 64 | 60.134 | 60.641 | 58.455 |
| 128 | **60.284** | 60.641 | 58.410 |

> Larger batches reduce variance and deliver better averages, but even N=64 is close to N=128 while being ~2× cheaper.

### 3.4 Tolerance vs Iterations

| tol | Avg Iterations |
| --- | --- |
| 1e-3 | 159.9 |
| 5e-4 | 339.5 |
| 1e-4 | 566.7 |

> Halving the tolerance roughly doubles the iteration count. If you need more stable policies, use `tol ≤ 5e-4` and raise `max_iters ≥ 500`.

## 4. Recommendations

1. **Use η = 0.02** for symmetric cases; higher values reduce effort and often hit the iteration cap.
2. **Set tol to 5e-4 or 1e-4** when you want >200 iterations of refinement. Combine with `max_iters ≥ 500`.
3. **Prefer batch sizes ≥ 64**; they offer better average performance with modest additional cost.
4. **Δ = 1.0** is a safe default; smaller deltas only help marginally.
5. **Monitor iterations**: If you routinely see `mcfd_iterations = 1000`, tighten tol or reduce η until convergence happens sooner.

## 5. Next Steps

- Rerun the sweep focusing on promising regions (e.g., η ∈ {0.015, 0.02, 0.03} with tol = 1e-4).  
- Track final effort histograms to understand variability.  
- Add reward variance metrics to the CSV to capture stability, not just mean effort.

---

_Report auto-generated from `results/one_stage_two_players.csv`._

