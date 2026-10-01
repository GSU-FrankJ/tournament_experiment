# Pilot 1 — Terminal reward estimator (Phase A only)

Date: 2026-10-01. Branch `v2-stagewise-pilots`.

| Item | Value |
|---|---|
| **Base commit** | `657f54a` |
| **Code commit for every Pilot 1 run** | **`89cd600`** (40/40 manifests: `git.short = 89cd600`, `dirty = false`) |
| Analysis tool | `tools/v2/pilot1_analysis.py` (commit `77f1fe6`) |
| Run records | commit `246d550` |

`P1/` means `results/v2_pilots/pilot1/` and `A/` means `P1/analysis/`. ΔW = 4.

These are descriptive results. **The estimator is not chosen here.**

---

## 1. What was done

### Changes before the pilot (§1 of the request)

All in commit `89cd600`.

**1a. On-path rule.**
- `utils/v2_metrics.py` defines on-path as {d ∈ D₂ : |d − drift| < 2q}, with `drift = ê₁(0) − ê₁(−0)` computed from the verifier's stage-1 arrays.
- Weights are exact cell masses F_ξ(b_{i+1} − drift) − F_ξ(b_i − drift), with cell boundaries at the midpoints between nodes and at the domain edges. They are normalized after reporting the captured total.
- The node-based GL PMF is kept only as an array (`v_t2_cand_pmf`).
- Tests:
  - new passing tests `test_onpath_rule_equals_open_support` (analytic, zero and 4 existing final checkpoints per q, on 3 tiers) and `test_cell_masses_sum_to_one`;
  - a separately named strict xfail, `test_node_pmf_support_mismatch_documented`.
- **Results across Pilot 1:** drift is exactly 0 at all 160 verifier checkpoints of the 40 runs, and the captured mass total is 1.0 at every checkpoint (min = max = 1.0). Source: `A/analysis_meta.json`.

**1b. Seeds 10501–10510.**
- They are disjoint from the Phase 0 inventory.
- A fresh scan finds 10501 only in this branch's own approved Phase 1/2 regression and smoke runs (`results/v2_pilots/phase1_regression`, `phase2_regression`, `_smoke`). Seeds 10502–10510 appear nowhere.

**1c. RNG stream positions.**
- `run/run_v2_stagewise.py` logs the exact PCG64 position (128-bit state, cached-uint32 flag and value) of the env, learn, opp, start and minibatch streams after every update, in `v2_updates.csv` columns `rngpos_*`.
- `tools/v2/rng_divergence.py` reports, per stream, the first update at which two runs differ.

### Verification of these changes at `89cd600` on a clean tree

- Tests: `53 passed, 2 xfailed` (`tests/test_v2_verifier.py` + `tests/test_v2_infra.py`).
- C7 regression against the existing runner: **IDENTICAL** (`results/v2_pilots/phase2_regression/v2_full_89cd600.compare.txt`; that run's manifest shows `dirty: false`). The new on-path rule changes reporting only.

### The dReach reach-mask check (§2)

It is reported separately in `reports/v2/dreach_reach_mask_check.md`:
- no holes in R₂;
- extras only at exactly ±2q from drift_BR (closed vs open interval);
- a dReach difference in 1 of 168 evaluations, +1.41e−3·ΔW.

### The pilot

**Design.** Phase A only, 400 updates, `fixed_budget=true`; arms `reward_mode` = sampled vs expected. q ∈ {50, 60} × seeds 10501–10510, paired: same seed, so the same `SeedSequence` streams and the same initial network. 40 runs.

**Launch.**
- Command: `tmux new-session -d -s v2_pilot1 "... tools/v2/launch_pilot.py --pilot pilot1 --phase A --qs 50 60 --seeds 10501 … 10510 --arms sampled expected --workers 40"`.
- Machine: `nproc` = 64. Load average was 0.90 / 0.63 / 0.35 at launch and 18.52 / 6.81 / 2.57 at the end (`P1/launch_20261001_185352.json`, fields `nproc`, `loadavg_at_start`, `loadavg_at_end`).
- 40 workers, one single-threaded process per run (OMP/MKL/OPENBLAS = 1, torch threads 1).
- **Outcome:** 40/40 returncode 0. Launcher wall per run 47.5 / 50.6 / 68.1 s (min / median / max). Phase A wall per run 41.34 / 45.03 / 62.72 s (min / median / max; `v2_run_summary.json → phase_timing.A.wall_sec`, summarized from `A/final_table.csv`).
- **Existing Phase A stop rule:** it would not have fired in any of the 40 runs (`would_have_fired.A = null` in every `P1/q*/seed*/*/v2_run_summary.json`).
- **Checkpoints:** every run saved a full-state `state_end_A.pt` (on disk; not committed, `.pt` is gitignored). These are the parents for Pilots 2 and 3.

**Metrics and conventions.**
- All metrics come from `utils/v2_metrics.evaluate` on the development tier (state step 4, effort step 1, GL 16/half), with the evaluated policy = Beta mean.
- Recovery metrics use the 0.5-step recovery grid.
- Full-policy metrics (EXP_root, dReach, Ĝmax_full) are **stage1_untrained**: `stage1_status` is `stage1_untrained` in all 40 final rows, and the stage-1 policy is the untrained initial network.
- Each run has 4 training-time verifier checkpoints (160 rows in total; `A/analysis_meta.json`). The final checkpoint is update 400 in every run; the analysis checks this and stops otherwise.

---

## 2. Tests

No new tests in this phase beyond the §1 changes listed above.

**Data consistency check.** Re-evaluating the u400 weight export reproduces each run's own update-400 checkpoint row to within 8.3e−17 (peak error) and 9.9e−17 (η₂), over 40 runs (`A/analysis_meta.json` → `weights_u400_vs_final_checkpoint`).

---

## 3. Results

### 3.1 Final-checkpoint table (update 400), all 40 runs

Source: `A/final_table.csv`, built from the last row of each `P1/q<q>/seed<seed>/<arm>/v2_checkpoints.csv`; wall time from `v2_run_summary.json`.

Columns:
- **Recovery:** peak rel = (ê₂(0) − e₂*(0))/e₂*(0). RMSE, tail and sym are in effort units.
- **Residuals:** η₂, Δ₂ on wmean and Δ₂ off max are /ΔW.
  - "Δ₂ on wmean" is the cell-mass-weighted mean of Δ₂ over the on-path set.
  - The on-path unweighted max equals η₂ in 39 of 40 runs. The exception is q50 s10508 expected, where the max sits off-path (0.001511 vs 0.001451 on-path).
- **Spread:** σ is in effort units.
- **Diagnostics:** KL and clip are from the last update.
- **Full-policy, stage1_untrained:** the columns marked * (EXP*, dReach*, Ĝmax*) are /ΔW.

Normalized columns (RMSE/e₂*(0), tail/e₂*(0), sym/e₂*(0)) and |peak| are in `A/final_table.csv`. e₂*(0) is 70 at q=50 and 58.333 at q=60.

| q | seed | arm | peak rel (signed) | RMSE | RMSE/e2*(0) | tail mean | tail max | sym | η₂ | Δ₂ on wmean | Δ₂ off max | σ₂(0) | σ̄₂(\|d\|<2q) | KL | clip | wall s | EXP* | dReach* | Ĝmax* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | expected | -0.1562 | 4.17 | 0.05957 | 1.382 | 6.134 | 2.626 | 0.006638 | 0.002074 | 0.002626 | 4.352 | 3.748 | 0.02994 | 0.06074 | 43.45 | 0.1435 | 0.1477 | 0.1435 |
| 50 | 10501 | sampled | -0.4 | 12.83 | 0.1833 | 7.115 | 11.45 | 13.18 | 0.04635 | 0.01615 | 0.007326 | 4.572 | 3.957 | 0.009848 | 0.03418 | 52.68 | 0.0375 | 0.06699 | 0.04635 |
| 50 | 10502 | expected | -0.1324 | 3.479 | 0.04969 | 2.092 | 4.437 | 5.218 | 0.004646 | 0.001458 | 0.001369 | 3.741 | 3.231 | 0.004514 | 0.03184 | 41.83 | 0.01444 | 0.0175 | 0.01444 |
| 50 | 10502 | sampled | -0.5058 | 18.08 | 0.2582 | 5.685 | 8.459 | 8.667 | 0.07268 | 0.03498 | 0.004824 | 4.196 | 3.348 | 0.005812 | 0.03418 | 58.68 | 0.03512 | 0.07273 | 0.07268 |
| 50 | 10503 | expected | -0.1285 | 3.232 | 0.04616 | 1.535 | 5.089 | 6.462 | 0.005119 | 0.001009 | 0.001797 | 4.023 | 3.53 | 0.00315 | 0.05078 | 43.81 | 0.1765 | 0.1808 | 0.1765 |
| 50 | 10503 | sampled | -0.7201 | 25.43 | 0.3632 | 19.62 | 20.96 | 2.641 | 0.1336 | 0.06769 | 0.03139 | 3.509 | 3.509 | 0.0008109 | 0.008789 | 42.23 | 0.06919 | 0.1376 | 0.1336 |
| 50 | 10504 | expected | -0.1216 | 2.413 | 0.03447 | 1.642 | 4.644 | 2.572 | 0.003827 | 0.0006558 | 0.001283 | 3.847 | 3.422 | 0.004391 | 0.02617 | 54.75 | 0.1031 | 0.1064 | 0.1031 |
| 50 | 10504 | sampled | -0.2763 | 7.98 | 0.114 | 7.627 | 9.59 | 3.647 | 0.02045 | 0.007376 | 0.006441 | 4.452 | 3.766 | 0.002473 | 0.0166 | 56.3 | 0.03469 | 0.04701 | 0.03469 |
| 50 | 10505 | expected | -0.1286 | 2.471 | 0.0353 | 1.71 | 5.141 | 2.277 | 0.004565 | 0.0007318 | 0.001784 | 3.822 | 3.373 | 0.002726 | 0.04863 | 43.5 | 0.06673 | 0.07061 | 0.06673 |
| 50 | 10505 | sampled | -0.4569 | 15.05 | 0.215 | 9.3 | 14.23 | 15.87 | 0.05643 | 0.02302 | 0.01261 | 4.343 | 3.758 | 0.00923 | 0.07402 | 42.15 | 0.03076 | 0.06343 | 0.05643 |
| 50 | 10506 | expected | -0.1534 | 2.946 | 0.04208 | 1.008 | 3.962 | 1.431 | 0.00635 | 0.001163 | 0.001039 | 4.16 | 3.607 | 0.0123 | 0.03438 | 52.46 | 0.1963 | 0.2018 | 0.1963 |
| 50 | 10506 | sampled | -0.3059 | 9.586 | 0.1369 | 7.216 | 8.437 | 16.62 | 0.02415 | 0.01615 | 0.004765 | 4.458 | 3.784 | 0.005021 | 0.03906 | 55.42 | 0.0383 | 0.04738 | 0.0383 |
| 50 | 10507 | expected | -0.1459 | 3.168 | 0.04526 | 1.826 | 4.207 | 1.128 | 0.005955 | 0.001361 | 0.001223 | 3.808 | 3.339 | 0.002878 | 0.04102 | 44.06 | 0.05394 | 0.05857 | 0.05394 |
| 50 | 10507 | sampled | -0.365 | 12.53 | 0.179 | 8.23 | 8.928 | 13.05 | 0.03798 | 0.02537 | 0.005692 | 4.456 | 3.678 | 0.003544 | 0.0459 | 45.09 | 0.03526 | 0.04732 | 0.03798 |
| 50 | 10508 | expected | -0.07954 | 2.353 | 0.03362 | 1.418 | 5.115 | 3.941 | 0.001511 | 0.0003797 | 0.001511 | 4.032 | 3.554 | 0.002073 | 0.04258 | 55.59 | 0.1272 | 0.1281 | 0.1272 |
| 50 | 10508 | sampled | -0.3991 | 11.97 | 0.171 | 7.836 | 9.176 | 10.24 | 0.0392 | 0.02113 | 0.006 | 4.351 | 3.73 | 0.0093 | 0.07832 | 42.55 | 0.03578 | 0.05368 | 0.0392 |
| 50 | 10509 | expected | -0.1244 | 2.529 | 0.03613 | 1.332 | 4.875 | 3.687 | 0.004308 | 0.0007002 | 0.001678 | 4.029 | 3.556 | 0.002899 | 0.03125 | 42.95 | 0.1135 | 0.1172 | 0.1135 |
| 50 | 10509 | sampled | -0.3436 | 9.953 | 0.1422 | 5.877 | 10.62 | 11.4 | 0.03132 | 0.01373 | 0.007033 | 4.542 | 3.911 | 0.008189 | 0.06875 | 42.37 | 0.03447 | 0.05228 | 0.03447 |
| 50 | 10510 | expected | -0.1299 | 3.886 | 0.05551 | 1.342 | 3.193 | 3.336 | 0.004566 | 0.001465 | 0.0005562 | 3.754 | 3.21 | 0.002141 | 0.0248 | 51.53 | 0.02892 | 0.03171 | 0.02892 |
| 50 | 10510 | sampled | -0.405 | 14.25 | 0.2035 | 6.341 | 8.511 | 7.702 | 0.04695 | 0.022 | 0.004754 | 4.397 | 3.533 | 0.007963 | 0.06543 | 62.72 | 0.02526 | 0.04895 | 0.04695 |
| 60 | 10501 | expected | -0.09853 | 1.841 | 0.03155 | 1.248 | 3.789 | 0.9355 | 0.001831 | 0.000354 | 0.000798 | 4.11 | 3.395 | 0.0009414 | 0.1158 | 42.14 | 0.03439 | 0.03585 | 0.03439 |
| 60 | 10501 | sampled | -0.5798 | 16.04 | 0.2749 | 10.93 | 14.05 | 4.781 | 0.0656 | 0.02918 | 0.01376 | 3.794 | 3.481 | 0.00538 | 0.05352 | 44.98 | 0.02972 | 0.06594 | 0.0656 |
| 60 | 10502 | expected | -0.1057 | 1.759 | 0.03016 | 1.281 | 3.061 | 3.324 | 0.00225 | 0.0003295 | 0.0006519 | 3.941 | 3.263 | 0.003928 | 0.05977 | 45.99 | 0.08232 | 0.08434 | 0.08232 |
| 60 | 10502 | sampled | -0.2272 | 6.152 | 0.1055 | 7.94 | 9.679 | 4.979 | 0.01005 | 0.004282 | 0.006643 | 4.577 | 3.757 | 0.004838 | 0.04395 | 42.3 | 0.0287 | 0.03443 | 0.0287 |
| 60 | 10503 | expected | -0.06882 | 1.89 | 0.0324 | 0.9716 | 3.53 | 3.011 | 0.001235 | 0.0004957 | 0.0008888 | 4.257 | 3.549 | 0.005153 | 0.08281 | 47.5 | 0.1903 | 0.1912 | 0.1903 |
| 60 | 10503 | sampled | -0.6365 | 18.27 | 0.3132 | 13.46 | 16.03 | 2.81 | 0.07934 | 0.03636 | 0.01691 | 3.649 | 3.449 | 0.01294 | 0.08672 | 55.44 | 0.03729 | 0.0799 | 0.07934 |
| 60 | 10504 | expected | -0.08302 | 1.695 | 0.02905 | 1.448 | 3.227 | 2.685 | 0.001181 | 0.0002937 | 0.0007048 | 3.901 | 3.23 | 0.0004878 | 0.01836 | 42.25 | 0.06581 | 0.06669 | 0.06581 |
| 60 | 10504 | sampled | -0.2761 | 5.824 | 0.09983 | 6.839 | 9.325 | 5.803 | 0.01389 | 0.004259 | 0.005545 | 4.532 | 3.858 | 0.005345 | 0.04844 | 41.51 | 0.0512 | 0.06136 | 0.0512 |
| 60 | 10505 | expected | -0.1068 | 2.129 | 0.0365 | 0.9137 | 3.413 | 1.793 | 0.002252 | 0.0005502 | 0.0008319 | 4.247 | 3.507 | 0.006297 | 0.04297 | 43.03 | 0.1583 | 0.1601 | 0.1583 |
| 60 | 10505 | sampled | -0.2553 | 6.611 | 0.1133 | 6.815 | 8.955 | 8.994 | 0.01298 | 0.003864 | 0.005081 | 4.491 | 3.687 | 0.003689 | 0.04238 | 49.91 | 0.02643 | 0.03536 | 0.02643 |
| 60 | 10506 | expected | -0.0979 | 1.757 | 0.03011 | 1.878 | 3.501 | 1.686 | 0.00177 | 0.000294 | 0.0008288 | 3.823 | 3.147 | 0.003928 | 0.02187 | 41.9 | 0.0376 | 0.03906 | 0.0376 |
| 60 | 10506 | sampled | -0.5631 | 15.3 | 0.2623 | 9.791 | 13.32 | 4.646 | 0.06106 | 0.02699 | 0.01158 | 3.874 | 3.498 | 0.01174 | 0.127 | 46.91 | 0.03047 | 0.06418 | 0.06106 |
| 60 | 10507 | expected | -0.08751 | 2.126 | 0.03645 | 1.45 | 3.894 | 5.179 | 0.001835 | 0.0005718 | 0.00105 | 3.924 | 3.258 | 0.003886 | 0.03574 | 56.31 | 0.1158 | 0.1172 | 0.1158 |
| 60 | 10507 | sampled | -0.5055 | 14.11 | 0.2419 | 10.92 | 16.03 | 12.27 | 0.04732 | 0.02318 | 0.017 | 3.99 | 3.604 | 0.006651 | 0.05859 | 41.34 | 0.03045 | 0.05459 | 0.04732 |
| 60 | 10508 | expected | -0.1351 | 2.247 | 0.03852 | 0.9294 | 3.627 | 2.763 | 0.003512 | 0.0005735 | 0.0009213 | 4.249 | 3.483 | 0.0002482 | 0.06523 | 55.61 | 0.1203 | 0.1234 | 0.1203 |
| 60 | 10508 | sampled | -0.5986 | 16.86 | 0.289 | 14.91 | 18.88 | 7.121 | 0.07187 | 0.0303 | 0.02482 | 3.791 | 3.557 | 0.001946 | 0.008398 | 41.49 | 0.03511 | 0.07585 | 0.07187 |
| 60 | 10509 | expected | -0.1256 | 2.05 | 0.03515 | 1.397 | 3.606 | 3.519 | 0.003374 | 0.0004401 | 0.0009288 | 3.976 | 3.318 | 0.009899 | 0.04883 | 56.18 | 0.0421 | 0.04509 | 0.0421 |
| 60 | 10509 | sampled | -0.2952 | 7.374 | 0.1264 | 8.09 | 9.235 | 9.991 | 0.01702 | 0.006813 | 0.005961 | 4.356 | 3.642 | 0.003329 | 0.0166 | 56.2 | 0.02245 | 0.03295 | 0.02245 |
| 60 | 10510 | expected | -0.0989 | 1.907 | 0.03269 | 1.427 | 3.645 | 5.009 | 0.001766 | 0.0003346 | 0.0009479 | 3.974 | 3.303 | 0.008742 | 0.04609 | 54.61 | 0.0521 | 0.05359 | 0.0521 |
| 60 | 10510 | sampled | -0.4397 | 11.45 | 0.1963 | 10.29 | 11.4 | 5.468 | 0.03777 | 0.01454 | 0.009211 | 4.229 | 3.67 | 0.01148 | 0.08262 | 42.17 | 0.02263 | 0.04545 | 0.03777 |

### 3.2 Paired differences (expected − sampled) per (q, seed)

Per-pair values are in `A/paired_differences.csv`; the summaries below are in `A/paired_summary.csv`.

- **Bootstrap:** 95% percentile CI of the mean paired difference, 10,000 resamples of the 10 pairs, numpy seed 20261001.
- **"Favour":**
  - For metrics where smaller is better, it is the number of pairs with expected − sampled < 0. This applies to |peak|, RMSE, tail, symmetry, η₂, Δ₂, wall time, and the stage1_untrained full-policy metrics.
  - Signed peak error, σ, KL and clip have no preferred direction; for them the counts of negative/positive differences are shown instead.

#### q = 50

| label | mean | sd | median | min | max | favour | CI95 |
|---|---|---|---|---|---|---|---|
| peak rel. err. at d=0 (signed) | 0.2877 | 0.129 | 0.2594 | 0.1524 | 0.5915 | n/a (0−/10+) | [0.2215, 0.3718] |
| \|peak rel. err.\| at d=0 | -0.2877 | 0.129 | -0.2594 | -0.5915 | -0.1524 | 10/10 | [-0.3697, -0.2199] |
| RMSE over \|d\|<2q (effort) | -10.7 | 4.849 | -9.49 | -22.19 | -5.567 | 10/10 | [-13.86, -8.214] |
| RMSE / e2*(0) | -0.1529 | 0.06927 | -0.1356 | -0.3171 | -0.07953 | 10/10 | [-0.1963, -0.117] |
| tail mean effort | -6.956 | 4.066 | -6.097 | -18.09 | -3.593 | 10/10 | [-9.637, -5.226] |
| tail mean / e2*(0) | -0.09937 | 0.05809 | -0.0871 | -0.2584 | -0.05133 | 10/10 | [-0.1374, -0.07405] |
| tail max effort | -6.357 | 3.644 | -5.131 | -15.87 | -4.023 | 10/10 | [-8.742, -4.689] |
| tail max / e2*(0) | -0.09081 | 0.05206 | -0.0733 | -0.2268 | -0.05747 | 10/10 | [-0.1252, -0.06699] |
| symmetry err. max (effort) | -7.033 | 5.963 | -7.006 | -15.19 | 3.821 | 9/10 | [-10.46, -3.525] |
| symmetry err. / e2*(0) | -0.1005 | 0.08519 | -0.1001 | -0.2169 | 0.05459 | 9/10 | [-0.1496, -0.04791] |
| eta_2 = max Delta_2 / DW | -0.04617 | 0.03277 | -0.0387 | -0.1285 | -0.01662 | 10/10 | [-0.0673, -0.02966] |
| Delta_2/DW on-path max | -0.04617 | 0.03276 | -0.03873 | -0.1285 | -0.01662 | 10/10 | [-0.06769, -0.03012] |
| Delta_2/DW on-path cell-mass mean | -0.02366 | 0.01678 | -0.02064 | -0.06669 | -0.00672 | 10/10 | [-0.03444, -0.01555] |
| Delta_2/DW off-path max | -0.007598 | 0.008005 | -0.004594 | -0.02959 | -0.003455 | 10/10 | [-0.01281, -0.004309] |
| sigma_2(0) (effort) | -0.371 | 0.3446 | -0.484 | -0.6479 | 0.5137 | n/a (9−/1+) | [-0.5379, -0.1412] |
| mean sigma_2 over \|d\|<2q | -0.2404 | 0.1309 | -0.2661 | -0.3855 | 0.02133 | n/a (9−/1+) | [-0.3122, -0.1593] |
| KL (final epoch, last update) | 0.0004816 | 0.008313 | -0.0009824 | -0.007227 | 0.02009 | n/a (6−/4+) | [-0.003809, 0.005806] |
| clip fraction (last update) | -0.007305 | 0.028 | -0.004785 | -0.04063 | 0.04199 | n/a (7−/3+) | [-0.02324, 0.009863] |
| Phase A wall-clock (s) | -2.625 | 8.246 | -1.292 | -16.84 | 13.04 | 6/10 | [-7.494, 2.238] |
| EXP_root/DW [stage1_untrained] | 0.06477 | 0.05488 | 0.07372 | -0.02068 | 0.158 | 1/10 | [0.0325, 0.09648] |
| dReach/DW [stage1_untrained] | 0.0423 | 0.05883 | 0.05129 | -0.05523 | 0.1545 | 2/10 | [0.00782, 0.07686] |
| Gmax_full/DW [stage1_untrained] | 0.04833 | 0.06289 | 0.05564 | -0.05824 | 0.158 | 2/10 | [0.01128, 0.08499] |

#### q = 60

| label | mean | sd | median | min | max | favour | CI95 |
|---|---|---|---|---|---|---|---|
| peak rel. err. at d=0 (signed) | 0.3369 | 0.1646 | 0.3794 | 0.1215 | 0.5677 | n/a (0−/10+) | [0.2397, 0.4329] |
| \|peak rel. err.\| at d=0 | -0.3369 | 0.1646 | -0.3794 | -0.5677 | -0.1215 | 10/10 | [-0.4301, -0.2364] |
| RMSE over \|d\|<2q (effort) | -9.859 | 4.88 | -10.77 | -16.38 | -4.129 | 10/10 | [-12.68, -7.017] |
| RMSE / e2*(0) | -0.169 | 0.08365 | -0.1846 | -0.2808 | -0.07078 | 10/10 | [-0.2174, -0.1204] |
| tail mean effort | -8.704 | 2.817 | -8.39 | -13.99 | -5.391 | 10/10 | [-10.44, -7.137] |
| tail mean / e2*(0) | -0.1492 | 0.0483 | -0.1438 | -0.2397 | -0.09242 | 10/10 | [-0.1793, -0.1227] |
| tail max effort | -9.162 | 3.37 | -8.79 | -15.26 | -5.542 | 10/10 | [-11.23, -7.264] |
| tail max / e2*(0) | -0.1571 | 0.05777 | -0.1507 | -0.2615 | -0.095 | 10/10 | [-0.1911, -0.125] |
| symmetry err. max (effort) | -3.696 | 2.639 | -3.481 | -7.201 | 0.2009 | 9/10 | [-5.247, -2.15] |
| symmetry err. / e2*(0) | -0.06336 | 0.04524 | -0.05968 | -0.1234 | 0.003444 | 9/10 | [-0.09007, -0.03665] |
| eta_2 = max Delta_2 / DW | -0.03959 | 0.02702 | -0.04075 | -0.07811 | -0.007798 | 10/10 | [-0.05542, -0.02373] |
| Delta_2/DW on-path max | -0.03959 | 0.02702 | -0.04075 | -0.07811 | -0.007798 | 10/10 | [-0.05561, -0.02367] |
| Delta_2/DW on-path cell-mass mean | -0.01755 | 0.0126 | -0.01841 | -0.03587 | -0.003314 | 10/10 | [-0.0248, -0.01019] |
| Delta_2/DW off-path max | -0.0108 | 0.006422 | -0.009509 | -0.0239 | -0.004249 | 10/10 | [-0.01473, -0.007322] |
| sigma_2(0) (effort) | -0.08812 | 0.432 | -0.1546 | -0.6355 | 0.6077 | n/a (7−/3+) | [-0.3323, 0.1648] |
| mean sigma_2 over \|d\|<2q | -0.2751 | 0.2158 | -0.3352 | -0.6272 | 0.09923 | n/a (9−/1+) | [-0.4038, -0.1487] |
| KL (final epoch, last update) | -0.002382 | 0.004435 | -0.00275 | -0.007811 | 0.00657 | n/a (8−/2+) | [-0.004826, 0.0003773] |
| clip fraction (last update) | -0.003066 | 0.04958 | -0.00166 | -0.1051 | 0.0623 | n/a (5−/5+) | [-0.03319, 0.0248] |
| Phase A wall-clock (s) | 2.327 | 8.704 | 0.3583 | -7.941 | 14.97 | 5/10 | [-2.654, 7.504] |
| EXP_root/DW [stage1_untrained] | 0.05846 | 0.05332 | 0.04154 | 0.004677 | 0.153 | 0/10 | [0.02867, 0.09184] |
| dReach/DW [stage1_untrained] | 0.03665 | 0.05266 | 0.02983 | -0.03009 | 0.1247 | 2/10 | [0.006261, 0.06877] |
| Gmax_full/DW [stage1_untrained] | 0.04073 | 0.05308 | 0.03406 | -0.03121 | 0.1319 | 2/10 | [0.01065, 0.07277] |

### 3.3 Learning curves

Figures:
- `reports/v2/figures/pilot1/curves_q50.png`
- `reports/v2/figures/pilot1/curves_q60.png`

Data: `A/curves_weights_every25.csv`.
- Each panel shows the per-arm median and IQR across the 10 seeds, for peak relative error (signed), RMSE/e₂*(0), tail mean, η₂ and σ₂(0).
- The x-axis is the weight exports every 25 updates (u25 … u400), re-evaluated with the same `evaluate` call on the development tier.
- **Deviation:** I used the 25-update exports rather than the training-time verifier checkpoints. Those fall at run-specific updates, because stability-triggered calls move them, so they cannot be aligned across seeds.

### 3.4 Descriptive scatter: peak relative error vs σ₂(0)/q

Figure: `reports/v2/figures/pilot1/scatter_peakerr_vs_sigma.png`. It plots all 160 training-time verifier checkpoints (4 per run), coloured by arm, one panel per q. Spearman ρ is computed with scipy `spearmanr` and stored in `A/spearman_peakerr_vs_sigma.csv`. No functional form is fitted.

| q | arm | n_points | spearman_rho | p_value |
|---|---|---|---|---|
| 50 | sampled | 40 | 0.873 | 2.072e-13 |
| 50 | expected | 40 | -0.4492 | 0.003646 |
| 50 | pooled | 80 | -0.0004923 | 0.9965 |
| 60 | sampled | 40 | 0.9293 | 5.064e-18 |
| 60 | expected | 40 | -0.1797 | 0.2671 |
| 60 | pooled | 80 | 0.5419 | 2.094e-07 |

### 3.5 RNG alignment per pair

Source: `A/rng_divergence.csv`, from `tools/v2/rng_divergence.py` on `v2_updates.csv`. Each entry is the first update at which a stream's position differs, or "never".

| q | seed | n_updates_compared | env | learn | opp | start | minibatch |
|---|---|---|---|---|---|---|---|
| 50 | 10501 | 400 | never | 28 | 44 | never | never |
| 50 | 10502 | 400 | never | 69 | 36 | never | never |
| 50 | 10503 | 400 | never | 40 | 50 | never | never |
| 50 | 10504 | 400 | never | 37 | 42 | never | never |
| 50 | 10505 | 400 | never | 11 | 73 | never | never |
| 50 | 10506 | 400 | never | 29 | 38 | never | never |
| 50 | 10507 | 400 | never | 61 | 22 | never | never |
| 50 | 10508 | 400 | never | 11 | 46 | never | never |
| 50 | 10509 | 400 | never | 50 | 30 | never | never |
| 50 | 10510 | 400 | never | 16 | 50 | never | never |
| 60 | 10501 | 400 | never | 28 | 24 | never | never |
| 60 | 10502 | 400 | never | 16 | 29 | never | never |
| 60 | 10503 | 400 | never | 49 | 42 | never | never |
| 60 | 10504 | 400 | never | 13 | 35 | never | never |
| 60 | 10505 | 400 | never | 63 | 22 | never | never |
| 60 | 10506 | 400 | never | 32 | 60 | never | never |
| 60 | 10507 | 400 | never | 15 | 50 | never | never |
| 60 | 10508 | 400 | never | 32 | 37 | never | never |
| 60 | 10509 | 400 | never | 41 | 69 | never | never |
| 60 | 10510 | 400 | never | 36 | 22 | never | never |

- **env, start and minibatch:** never diverge in any of the 20 pairs.
- **learn:** diverges in every pair, at update 11–69 (median 33) at q=50 and 13–63 (median 32) at q=60.
- **opp:** diverges in every pair, at update 22–73 (median 43) at q=50 and 22–69 (median 36) at q=60.

---

## 4. Anomalies and deviations from the request

1. **RNG alignment of the action streams is lost early.** In all 20 pairs the learner and opponent Beta streams desynchronize by update 73 at the latest (§3.5).
   - The source is the one identified in Phase 2: numpy's rejection-based Beta sampler consumes a parameter-dependent number of draws, and the two arms' policies differ after the first update.
   - The shock (env), start/role and minibatch streams stay aligned for all 400 updates.
   - So beyond the first tens of updates, "paired" means the same initialization and the same shock, start and minibatch streams, but different action-noise draws.
   - Per the instruction, no sampler was changed.
2. **Learning curves** use the 25-update weight exports instead of the training-time checkpoints (§3.3). The two agree at u400 to within 1e−16 (§2).
3. **Seed 10501** was already used by this branch's own approved regression and smoke runs, with reduced budgets (§1, 1b). Those runs are not part of Pilot 1.
4. **Off-path maximum in one run.** In q50 s10508 expected, the on-path unweighted max of Δ₂ (0.001451) is below η₂ (0.001511): the global maximum lies off-path.
5. **Parents not committed.** The full-state parents `P1/q*/seed*/*/state_end_A.pt` are on disk and not in git, because `.pt` is gitignored. The same applies to the per-checkpoint NPZs, the weight exports and `train_history.json` (107 MB of NPZ in total), to keep the repository small. The committed records are the configs, manifests, status files, summaries, CSV logs, final JSON and plots.

---

## 5. Interpretation (labelled; not a choice of estimator)

- **Stage-2 recovery: lower error with expected in every pair, at both q.**
  - All of these favour `expected` in 10/10 pairs at both q, with bootstrap CIs that exclude 0: |peak error|, RMSE, tail mean and tail max, η₂, and the on- and off-path Δ₂.
  - The `sampled` arm under-exerts at the peak (median signed peak error −0.40 at q=50 and −0.47 at q=60) and keeps substantial effort in the theoretical zero region (median tail mean 7.4 and 10.0). The `expected` arm is closer on both counts (−0.13 and −0.10; tail mean 1.5 and 1.3). Medians are computed from `A/final_table.csv`.
  - In the q=50 curves the separation builds between roughly update 75 and 150. After that `sampled` keeps improving slowly while `expected` has largely flattened.
- **Stage1_untrained full-policy metrics: the opposite direction.**
  - EXP_root, dReach and Ĝmax_full are higher with `expected` in 8–10 of 10 pairs.
  - With an untrained stage 1, these measure how badly the *initial* stage-1 action fits the learned stage 2. A more accurate stage 2 can raise them. They are not evidence about the stage-2 estimator itself, and Pilot 2 is where they become meaningful.
- **σ₂ is mostly lower with expected.** σ₂(0) is lower in 9/10 pairs at q=50 and 7/10 at q=60; mean σ₂ over |d| < 2q is lower in 9/10 at both q.
- **Peak error vs σ₂(0)/q.**
  - Within `sampled`, peak error rises (less under-exertion) with σ₂(0)/q: ρ = 0.87 at q=50 and 0.93 at q=60.
  - Within `expected`, the relation is weak or negative: ρ = −0.45 and −0.18.
  - This is descriptive only, from 4 checkpoints per run.
- **Pairing.** Because the action-noise streams desynchronize early (§4.1), the paired design controls initialization and environment noise but not the action draws. The consistency of the sign across all 10 pairs is the more robust descriptive fact than any single paired magnitude.

---

## 6. Commands to reproduce

```bash
tmux new-session -d -s v2_pilot1 "cd /home/fjiang4/tournament_experiment/.claude/worktrees/tournament-v2-stagewise-pilots-f97609 && /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/launch_pilot.py --pilot pilot1 --phase A --qs 50 60 --seeds 10501 10502 10503 10504 10505 10506 10507 10508 10509 10510 --arms sampled expected --workers 40"
```

```bash
OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/pilot1_analysis.py
```

```bash
/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/rng_divergence.py results/v2_pilots/pilot1/q50/seed10501/sampled results/v2_pilots/pilot1/q50/seed10501/expected
```

---

**STOP — Pilot 1 complete. Waiting for your choice of reward estimator.**
