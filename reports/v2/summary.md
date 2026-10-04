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
| Protocol lock + dress rehearsal (dev seeds) + cusp diagnostic | `reports/v2/protocol_lock_and_rehearsal.md` | lock `4bd2214`; rehearsal `5b07293` | 20 + 2 + 1 |
| Protocol v1.1 + re-rehearsal + **fresh-seed confirmation** | `reports/v2/protocol_v1_1_confirmation.md` | lock `431474d`; rehearsal `95c000e`; confirmation `f6838ec` | 20 + 40 |

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

## Protocol lock and dress rehearsal (2026-10-02)

- **Locked protocol v1:** `protocols/v2_T2_locked.json` / `.md`.
  - Lock commit `4bd2214`, recorded in `protocols/LOCK` (`7412c41`).
  - Protocol SHA-256 `cf7b6929…`.
  - Entry point `run/run_v2_T2_locked.py`, which takes only `--q`, `--seed`, `--out-dir`.
  - C7 is IDENTICAL at the lock.
- **Canonical worktree:** `.claude/worktrees/pilot-4-stabilization-fb99a2`, on branch `v2-stagewise-pilots` (fast-forwarded). Canonical development results: its `results/v2_pilots/`; all 10,034 original files are checksum-identical.
- **Calibration at the lock** matches Phase 1 to ≤ 1e−16. Zero policy: 0.25926 / 0.19551 at stage 2, which matches the PI reference.
- **Rehearsal on the development seeds** (not evidence for the protocol):

| q | G-A | G-F | run pass | median η₂/ΔW | median Ĝmax_full/ΔW | median \|stage-1 err\| |
|---|---|---|---|---|---|---|
| 50 | 10/10 | 9/10 | 9/10 | 0.0012 | 0.0013 | 0.035 |
| 60 | 10/10 | 8/10 | 8/10 | 0.00085 | 0.00085 | 0.045 |

Source: `results/v2_T2_locked/rehearsal_analysis/`.
- All 3 failures are |stage-1 err| > 0.10.
- Check 2 (Phase B code path): bit-identical, 2/2.
- **Check 1** (Phase A vs the stitched development state): every training state is bit-identical in 20/20. Three never-consumed, process-global RNG states (torch / numpy legacy / python) differ, so the literal criterion fails. Your decision is pending.
- **Cusp diagnostic:** the exact-target fit first drops below 5% peak error after 8,500–9,500 full-batch steps, with RMSE ≈ 0.016 by then. RL Phase A gives 32,000 actor steps (8,000 in the last 400 updates).
- **Proposed confirmation seed block:** 20501–20520 (0 collisions); awaiting your confirmation. The confirmation was not run.

## Protocol v1.1 and the confirmation (2026-10-02)

- **v1.1 changes:** the stage-1 criterion moves from G-F to the secondary S1; gate G-N is added (|dev − final| of η₂ and Ĝmax ≤ 0.001·ΔW); the process-global RNGs are hardened.
- **Lock:** commit `431474d` (`protocols/v2_T2_locked_v1_1.json`, SHA-256 `21d85983…`), recorded in `protocols/LOCK` at `95c000e`. v1.0 is unchanged.
- **Re-rehearsal on the development seeds:** R1–R6 all pass (`results/v2_T2_locked/rehearsal_v1_1_checks.json`). The training state is bit-identical to v1.0 in 20/20.
- **Confirmation** (seeds 20501–20520, launch commit `f6838ec`) — **verdict: PASS.**

| q | primary passes | exact 95% CI | S1 passes | mean signed stage-1 err [boot 95% CI] | v1.0 outcome |
|---|---|---|---|---|---|
| 50 | 20/20 | [0.832, 1] | 20/20 | −0.0051 [−0.0259, 0.0161] | 20/20 |
| 60 | 20/20 | [0.832, 1] | 18/20 | +0.0077 [−0.0185, 0.0349] | 18/20 |

Source: `results/v2_T2_locked/confirmation_analysis/` (pre-registered script, run unchanged). 40/40 runs had exit code 0, no global-RNG violations and no crashes.

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
8. **Gate thresholds and protocol lock**: done (locked v1, `protocols/LOCK`).
9. **Eight Pilot-4 2b manifests carry `dirty: true`**: resolved. A clean re-run of q60/10507 `B2_mean_constant` is bit-identical.
10. **Rehearsal Check 1:** accepted (D1, v1.1 round).
11. **Confirmation seed block 20501–20520:** confirmed and run; the confirmation passed.

## R1 accuracy-refinement round (2026-10-03, branch v2-t2-refine)

A follow-up round on the locked v1.1 pipeline, without changing the protocol: two diagnostics (clipped-Beta likelihood, verifier sensitivity) and six single-factor pilots (polishing, batch, target-KL, concentration annealing, deterministic-mean fine-tuning, expected continuation) on the development seeds, paired with the locked baseline. The unchanged entry point reproduces the rehearsal in 20/20 runs; among the six methods only the expected-continuation arm meets its pre-registered criterion at both q (S1 error, with 0.3x the across-seed spread of the signed error), under a check of the continuation table that is open as stated. Summary and reading order: `reports/v2/refine/summary.md`; decision inputs: `reports/v2/refine/06_decision_inputs.md`.

## T=2 protocol v2.0: re-rehearsal and fresh-seed confirmation (2026-10-04, branch v2-t2-refine)

Protocol v2.0 (lock commit `1d6d4d0`, LOCK record `f2d616c`) is v1.1 plus the expected continuation in Phase B (the R1 arm `B_expcont`), the gate G-S (stage-1 error ≤ 0.05, in the run pass) and the seed block 30501–30520; Phase A is unchanged. Check (ii) of the continuation table, left open in the R1 round, was settled before the lock by decision D3: the tests (ii-a), (ii-b) and (ii-c) all pass (`results/v2_T2_locked/v2_0/continuation_check_v2_0.json`). The re-rehearsal on the development seeds passed R1–R6 (20/20 each; R3b: 20/20 under G-S, at least 19 needed, and a normal-model P(≥ 18 of 20 fresh runs pass G-S) of 0.9959 and 0.9995 against the limit 0.90). R7 was false as the checks tool computed it, because the tool's C7 comparison read a reference whose gitignored `checkpoint.pt` is not in that worktree; the C7 comparison against the canonical reference is IDENTICAL including `checkpoint.pt`, and the owner accepted R7 as met (decision D-R7). The confirmation (launch commit `d2e377d`, 40 runs, all exit 0, no global-RNG violation): **PASS**. q=50 19/20 and q=60 20/20 pass G-A ∧ G-F ∧ G-N ∧ G-S against the rule of at least 18 of 20 at each q; the one failure is a Phase-A stage-2 failure (q=50, seed 30510, G-A η₂ 0.0058 > 0.005). G-S passed 20/20 and 20/20. The checks tool's reference path was fixed after the confirmation (`3fedaa2`); its R7 re-run at that commit gives `pass` True. Report: [protocol_v2_0_confirmation.md](protocol_v2_0_confirmation.md); data: `results/v2_T2_locked/confirmation_v2_0*` and `rehearsal_v2_0*`. Deviations are listed in §6 of the report.

## Addendum (2026-10-04): pushed state

Addendum (2026-10-04, P0 of round R2b): the v2.0 commits are no longer unpushed. `origin/v2-t2-refine` is at `e89b61d` (pushed on 2026-10-04; `git ls-remote origin v2-t2-refine`). Annotated tags for the publication: `t2-v2-lock-v2.0` -> `1d6d4d0` (the v2.0 lock commit), `t2-v2-confirmation-v2.0` -> `d2e377d` (the confirmation launch commit) and `t2-v2-main-v2.0` -> the commit that adds `reports/v2/refine_r2b/00_housekeeping.md` (the head of `v2-t2-refine` after the P0 housekeeping commits). The fast-forward of `main` and the push of `main` and the tags are recorded in `reports/v2/refine_r2b/00_housekeeping.md`. The "not pushed" wording above is left as written; it was true when it was written.

## R2b terminal-stage follow-up round (2026-10-04, branch v2-t2-r2b, not pushed)

After the v2.0 confirmation (PASS, 19/20 and 20/20; the failed run is q=50 seed 30510), a round on the development seeds tested three mechanisms for the stage-2 peak, one at a time and paired with the existing runs: peak-focused exploring starts (shares 0.25 and 0.50), censored likelihood of clamped Beta draws, and pathwise terminal fine-tuning with 20 steps per update (model-based; ablation only). The unchanged v2.0 entry point run from the new branch reproduces `rehearsal_v2_0` in 20/20 (C-R2). Seed 30510 was diagnosed read-only: under the pre-registered rules none of H1-H4 is supported, and the descriptive facts (early divergence of the concentration, a plateau below the pack from about update 900) are in the report. In the pilots no arm meets both parts of the pre-registered criterion: peak-focused starts at share 0.50 improve the absolute peak error at both q (CI of the mean difference excludes 0; 8 and 7 of 10 seeds improve) but 3 of 10 q=60 runs that passed G-A fail its tail-mean limit; the censored likelihood and the pathwise arms do not separate from their comparators. Nothing is decided and nothing was combined. Summary and reading order: `reports/v2/refine_r2b/summary.md`; decision inputs: `reports/v2/refine_r2b/05_decision_inputs.md`.
