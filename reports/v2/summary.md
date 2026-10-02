# v2 T=2 pilots — summary

Date: 2026-10-01. Branch `v2-stagewise-pilots`, base commit `657f54a`. ΔW = 4.

All values are descriptive. Each table names its source file; the full reports carry every number and its provenance.

| Study | Report | Launch commit | Runs |
|---|---|---|---|
| Phase 0 audit | `reports/v2/phase0_audit.md` | — | — |
| Phase 1 verifier | `reports/v2/phase1_verifier.md` | — | — |
| Phase 2 infrastructure | `reports/v2/phase2_infra.md`, `reports/v2/phase2_opening_checks.md` | — | smoke only |
| dReach mask check | `reports/v2/dreach_reach_mask_check.md` | `89cd600` (tool) | — |
| Pilot 1: reward estimator | `reports/v2/pilot1_reward_estimator.md` | `89cd600` | 40 |
| Pilot 2: joint vs frozen | `reports/v2/pilot2_freeze.md` | `1791687` (+ §6 recompute at `cd760fd`) | 60 |
| Pilot 3: continuation mode | `reports/v2/pilot3_continuation_mode.md` | `cd760fd` | 40 |
| Phase A extension | `reports/v2/phaseA_ext.md` | `cd760fd` | 20 |
| Pilot 4: stabilization (1a–1d analyses, LR decay 2a/2b) | `reports/v2/pilot4_stabilization.md` | `c92ee74` | 80 |

**Decisions taken so far:**
- reward estimator = `expected` (after Pilot 1);
- frozen variant = B2 (`adv_norm_scope = stage1_rows`) (after Pilot 2);
- ẽ₁ by residual minimization on the final tier (decision D2);
- on-path set = {|d − drift| < 2q}, weighted with exact cell masses;
- (2026-10-02, before Pilot 4) D1 continuation mode = `mean`; D2 Phase A = fixed 1600 updates + end-of-phase pass/fail gate on η₂ and tail/RMSE (thresholds pending); D3 joint training and Phase C dropped from v2; D4 no sampler change.

---

## Pilot 1 — sampled vs expected terminal reward (Phase A, 400 updates)

Final-checkpoint medians. Source: `results/v2_pilots/pilot1/analysis/final_table.csv`. EXP_root is stage1_untrained.

| q | arm | stage2_peak_rel_err_signed | stage2_rmse_pos_over_g2_0 | stage2_tail_mean | eta_T_over_dw | sigma_effort_at_0_t2 | EXP_root_over_dw |
|---|---|---|---|---|---|---|---|
| 50 | expected | -0.1293 | 0.04367 | 1.477 | 0.004606 | 3.935 | 0.1083 |
| 50 | sampled | -0.3995 | 0.1811 | 7.421 | 0.04278 | 4.424 | 0.03519 |
| 60 | expected | -0.09872 | 0.03254 | 1.339 | 0.001833 | 3.975 | 0.07407 |
| 60 | sampled | -0.4726 | 0.2191 | 10.04 | 0.04255 | 4.109 | 0.03008 |

Paired differences (expected − sampled). Source: `results/v2_pilots/pilot1/analysis/paired_summary.csv`. `n_favour_expected` is out of 10. Bootstrap: 10,000 resamples, seed 20261001.

| q | metric | median | n_favour_expected | boot_ci95_lo | boot_ci95_hi |
|---|---|---|---|---|---|
| 50 | stage2_peak_rel_err_abs | -0.2594 | 10 | -0.3697 | -0.2199 |
| 50 | stage2_rmse_pos_over_g2_0 | -0.1356 | 10 | -0.1963 | -0.117 |
| 50 | stage2_tail_mean | -6.097 | 10 | -9.637 | -5.226 |
| 50 | eta_T_over_dw | -0.0387 | 10 | -0.0673 | -0.02966 |
| 60 | stage2_peak_rel_err_abs | -0.3794 | 10 | -0.4301 | -0.2364 |
| 60 | stage2_rmse_pos_over_g2_0 | -0.1846 | 10 | -0.2174 | -0.1204 |
| 60 | stage2_tail_mean | -8.39 | 10 | -10.44 | -7.137 |
| 60 | eta_T_over_dw | -0.04075 | 10 | -0.05542 | -0.02373 |

Share of the d=0 peak gap explained by action-noise smoothing (median over seeds). Source: `results/v2_pilots/pilot1/analysis/smoothed_game/per_run.csv`.

| q | arm | share_peak_gap_explained |
|---|---|---|
| 50 | expected | 0.3306 |
| 50 | sampled | 0.126 |
| 60 | expected | 0.3759 |
| 60 | sampled | 0.08229 |

## Pilot 2 — joint (A) vs frozen_allnorm (B1) vs frozen_s1norm (B2) (Phase B, 600 updates)

Final medians. Source: `results/v2_pilots/pilot2/analysis/final_table.csv`. Drift is in effort units against the parent's stage 2; Ĝmax, EXP and dReach are /ΔW.

| q | arm | stage2_drift_cand_on_max | stage2_drift_cand_off_max | stage2_peak_rel_err_signed | stage2_tail_mean | stage1_rel_err_signed | Gmax_full_over_dw | EXP_root_over_dw | dReach_over_dw |
|---|---|---|---|---|---|---|---|---|---|
| 50 | A | 6.28 | 2.076 | -0.06038 | 2.584 | -0.03841 | 0.003227 | 0.00129 | 0.003801 |
| 50 | B1 | 0 | 0 | -0.1293 | 1.477 | 0.01941 | 0.004606 | 0.001979 | 0.005651 |
| 50 | B2 | 0 | 0 | -0.1293 | 1.477 | 0.01962 | 0.004606 | 0.001705 | 0.005795 |
| 60 | A | 4.001 | 1.558 | -0.06171 | 2.192 | -0.001616 | 0.002666 | 0.0006633 | 0.002931 |
| 60 | B1 | 0 | 0 | -0.09872 | 1.339 | 0.05607 | 0.002039 | 0.001027 | 0.002668 |
| 60 | B2 | 0 | 0 | -0.09872 | 1.339 | 0.009708 | 0.001833 | 0.0007316 | 0.002383 |

Revised stage-1 decomposition at update 1000, /e₁* (residual band, final tier). Source: `results/v2_pilots/pilot2/analysis/decomposition_residual_band.csv`. `inherited_band_contains0` is out of 10.

| q | arm | median_abs_learning | median_inherited | inherited_band_contains0 |
|---|---|---|---|---|
| 50 | A | 0.04366 | -0.002571 | 3 |
| 50 | B1 | 0.0434 | 0.001393 | 1 |
| 50 | B2 | 0.06725 | 0.001393 | 1 |
| 60 | A | 0.0486 | -0.001929 | 1 |
| 60 | B1 | 0.06534 | 0.0006429 | 3 |
| 60 | B2 | 0.05282 | 0.0006429 | 3 |

## Pilot 3 — frozen B2, continuation stochastic vs mean (Phase B, 600 updates)

**Reproducibility:** the stochastic arm is bit-identical to Pilot 2 B2 in **20/20** runs (`results/v2_pilots/pilot3/analysis/reproducibility_vs_pilot2_B2.csv`).

Final medians. Source: `results/v2_pilots/pilot3/analysis/final_table.csv`. The learning and inherited terms are /e₁*; within-run SD is in effort units over exports u900–u1000.

| q | arm | stage1_rel_err | abs_stage1_err | learning | inherited | within_run_sd_last5 | EXP | dReach |
|---|---|---|---|---|---|---|---|---|
| 50 | mean | -0.02835 | 0.06763 | -0.02974 | 0.001393 | 1.044 | 0.002336 | 0.006458 |
| 50 | stochastic | 0.01962 | 0.06479 | 0.01553 | 0.001393 | 1.947 | 0.001705 | 0.005795 |
| 60 | mean | -0.02346 | 0.08852 | -0.02655 | 0.0006429 | 1.988 | 0.001452 | 0.003369 |
| 60 | stochastic | 0.009708 | 0.04948 | 0.007651 | 0.0006429 | 1.802 | 0.0007316 | 0.002383 |

Stability. Source: `results/v2_pilots/pilot3/analysis/stability.csv`.

| q | arm | across_seed_sd_final_e1 | across_seed_iqr_final_e1 | median_within_run_sd_last5 | median_within_run_range_last5 | median_sigma1_0 |
|---|---|---|---|---|---|---|
| 50 | stochastic | 3.506 | 5.178 | 1.947 | 5.004 | 3.646 |
| 50 | mean | 3.405 | 3.934 | 1.044 | 2.378 | 3.624 |
| 60 | stochastic | 2.576 | 2.945 | 1.802 | 4.61 | 3.485 |
| 60 | mean | 4.11 | 5.709 | 1.988 | 4.659 | 3.498 |

Paired differences (mean − stochastic). Source: `results/v2_pilots/pilot3/analysis/paired_summary.csv`. `n_mean_better` is out of 10.

| q | metric | median | n_mean_better | boot_ci95_lo | boot_ci95_hi |
|---|---|---|---|---|---|
| 50 | stage1_rel_err_abs | -0.001984 | 6 | -0.03213 | 0.02782 |
| 50 | learning_rel_abs | -0.001984 | 6 | -0.03484 | 0.03169 |
| 50 | EXP_root_over_dw | -9.664e-05 | 6 | -0.0008869 | 0.001156 |
| 50 | dReach_over_dw | -9.698e-05 | 6 | -0.0008788 | 0.001157 |
| 50 | within_run_sd_e1_last5 | -0.5063 | 9 | -1.099 | -0.4001 |
| 60 | stage1_rel_err_abs | 0.04944 | 3 | -0.0009213 | 0.06754 |
| 60 | learning_rel_abs | 0.04406 | 2 | -0.002521 | 0.06367 |
| 60 | EXP_root_over_dw | 0.0007288 | 2 | -0.0001783 | 0.001573 |
| 60 | dReach_over_dw | 0.0007421 | 2 | -0.0001899 | 0.001589 |
| 60 | within_run_sd_e1_last5 | 0.09531 | 5 | -0.6642 | 0.5959 |

## Phase A extension — stage 2 only, u400 → u1600

Medians across 10 seeds. Source: `results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv` (the IQR and min/max are in `reports/v2/phaseA_ext.md`).

| metric | q | 400 | 800 | 1200 | 1600 |
|---|---|---|---|---|---|
| eta_T_over_dw | 50 | 0.004606 | 0.002664 | 0.002556 | 0.001469 |
| eta_T_over_dw | 60 | 0.001833 | 0.001113 | 0.001059 | 0.0009012 |
| share_peak_gap_explained | 50 | 0.3306 | 0.4808 | 0.5824 | 0.5059 |
| share_peak_gap_explained | 60 | 0.3759 | 0.5791 | 0.508 | 0.4003 |
| sigma_effort_at_0_t2 | 50 | 3.935 | 3.365 | 2.981 | 2.735 |
| sigma_effort_at_0_t2 | 60 | 3.975 | 3.499 | 3.166 | 2.883 |
| stage2_peak_rel_err_signed | 50 | -0.1293 | -0.07907 | -0.06002 | -0.06835 |
| stage2_peak_rel_err_signed | 60 | -0.09872 | -0.0577 | -0.05439 | -0.06003 |
| stage2_rmse_pos_over_g2_0 | 50 | 0.04367 | 0.03083 | 0.02896 | 0.02605 |
| stage2_rmse_pos_over_g2_0 | 60 | 0.03254 | 0.0309 | 0.0261 | 0.02389 |
| stage2_tail_mean | 50 | 1.477 | 0.8706 | 0.6386 | 0.515 |
| stage2_tail_mean | 60 | 1.339 | 0.7954 | 0.6048 | 0.5312 |

Plateau description (from `reports/v2/phaseA_ext.md` §3):
- the peak error flattens at about −0.06 from about u800;
- tail effort, σ₂ and off-path Δ₂ keep decreasing through u1600;
- η₂ drifts down noisily.

## Pilot 4 — stabilization round (2026-10-02)

Launch commit `c92ee74`; C7 IDENTICAL at that commit. 2a: 40 runs (u1200→u1600, `constant` vs linear `decay` 3e−4→3e−5). 2b: 40 runs (B2 + `mean`, parent u1600, u1601→u2200, `constant` vs `decay`). Full report: `reports/v2/pilot4_stabilization.md`.

- **1a** The reference BR slope / E[V₂″]/(2k) values are reproduced within fit and step variation (q=50: slope −0.85 to −1.06, ref −0.961; q=60: −0.26 to −0.33, ref −0.309). Source: `results/v2_pilots/pilot4/analysis/root_game/`.
- **1b** ê₁(0) − e₁*: median lag-20 ACF 0.48–0.60, ≈ 0 at lag 60, negative at 100–160. Window slope −0.15 (q=50) / −0.20 (q=60), pooled. Source: `.../analysis/fluctuation/`.
- **1c** Tail-averaging stage 2 (K up to 16) lowers RMSE and the symmetry error, and does not move the peak error (≈ −0.06). Source: `.../analysis/candidates_all.csv`, `paired_summary.csv`.
- **1d** The supervised floor of the stage-2 actor at d=0 is ≤ 0.06% of e₂*(0). The RL gap is 6–7%, of which action-noise smoothing predicts ≈ 45%. Source: `.../analysis/repr_floor/`.
- **2a** `constant` is bit-identical to the extension (20/20). Decay lowers RMSE (q=60 8/10) and KL/clip; peak and η₂ are unchanged; tail mean is slightly higher.
- **2b** Decay vs constant, medians:

| q | within-run SD last5 | across-seed SD final ê₁ | |stage-1 err| (K=1) | EXP_root/ΔW (K=1) |
|---|---|---|---|---|
| 50 | 1.02 vs 1.75 (CI excl. 0) | 2.49 vs 3.33 | 0.043 vs 0.063 (CI incl. 0) | lower in 7/10 (CI excl. 0) |
| 60 | 0.99 vs 1.37 (CI excl. 0) | 1.70 vs 3.31 | 0.034 vs 0.055 (CI incl. 0) | lower in 6/10 (CI excl. 0) |

Gate-setting distribution tables (quantiles over seeds, no thresholds): `reports/v2/pilot4_stabilization.md` §6, `results/v2_pilots/pilot4/analysis/gate_distribution_tables.csv`.

---

## Open questions

1. **Phase A budget and precision criterion.** These are your decision; the extension data are descriptive only. The peak error flattens from about u800, while tail and σ₂ keep improving.
2. **Stage-1 accuracy.** In every Phase B arm, ê₁(0) still oscillates by a few percent around e₁* at u1000, and the learning term dominates the decomposition. Not tested yet:
   - longer Phase B;
   - learning-rate decay;
   - a stopping rule for stage 1.
3. **The stage-2 peak keeps improving under joint training (Pilot 2 A), at the cost of the tail and off-path region.** Interaction with a longer Phase A is untested.
4. **RNG pairing.** The learner/opponent Beta streams desynchronize in every pair within tens to about 160 updates, because numpy's Beta sampler is rejection-based. Pairing therefore controls initialization, shocks, starts and minibatches, but not action noise. An inverse-CDF sampler would align them, but it would break C7 against the existing runner; that is your call.
5. **Induced-target bands.** 167 of 480 A_joint export bands are non-contiguous; the outermost points are used. The parent sweep range [0.4, 1.7]·e₁* misses ê₁ at 13 early rows (Pilot 2: 7, Pilot 3: 6).
6. **The q=60 calibration floor** (3.5e−7·ΔW at the root on the finer tiers) is still unexplained (Phase 1). e₁* lies off-grid at both q.
7. **Phase C and T=3** are out of scope and not run (Phase C dropped from v2, D3).
8. **Gate thresholds and protocol lock** (after Pilot 4): your decision. The distribution tables are in `reports/v2/pilot4_stabilization.md` §6.
9. **Eight Pilot-4 2b manifests carry `dirty: true`** (untracked analysis scripts present at launch; tracked code = `c92ee74`). A clean re-run of one of them would confirm this.
