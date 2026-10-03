# Pilot 1 accepted: estimator = `expected`. Run Pilot 2 (joint vs frozen).

Pilot 1 is accepted. **The reward estimator for all further development is `reward_mode = expected`.**

All earlier instructions still apply unless changed here:
- phase gates and provenance discipline;
- closed form used for evaluation only;
- the on-path rule (open interval, exact cell masses);
- the A6 RNG conventions;
- no deletion of results.

**Scope.** Run Pilot 2 (§1–§3). Optionally run the side analysis in §4. Then **STOP**.

---

## 1. Design

**Parents.** The 20 full-state end-of-Phase-A checkpoints from Pilot 1 with `reward_mode = expected`: q ∈ {50, 60} × seeds 10501–10510. Before launching, verify each parent:
- its manifest says `reward_mode = expected` and commit `89cd600`;
- it restores cleanly, including the lagged-opponent copy and its refresh counter.

**Phase B settings**
- **Budget:** existing Phase B, **600 updates, fixed budget**. No early exit. Record the update at which the existing Phase B stop rule would have fired.
- **Reward:** `reward_mode = expected`, the same as the parent.
- **Start state:** the root `(t=1, d=0)`, as in the existing Phase B.
- **Stage-1 opponent:** the existing lagged copy, refreshed every 20 updates.
- **Continuation action mode:** `stochastic`.

**Arms**

| Arm | `stage2_update_mode` | `adv_norm_scope` |
|---|---|---|
| A `joint` | joint | all_rows |
| B1 `frozen_allnorm` | frozen | all_rows |
| B2 `frozen_s1norm` | frozen | stage1_rows |

**Runs.** 20 parents × 3 arms = **60 runs**. All three arms of a parent restore the same checkpoint. Use single-threaded processes in tmux, with parallelism chosen from `nproc` and the current load, and report both.

**Code state.** Commit any new analysis code before launching. Every manifest must carry the launch commit hash and a clean-tree flag.

---

## 2. Metrics

Log these at every verifier checkpoint, plus the weight exports every 25 updates, as in Pilot 1. The evaluated policy is the Beta mean.

**Stage-2 drift relative to the parent**
- `ê_2` change, split into on-path (`|d| < 2q`) and off-path, as:
  - max |Δ| (unweighted);
  - cell-mass-weighted mean |Δ|.
- Stage-2 recovery: signed and absolute peak error, normalized RMSE, tail mean and tail max (raw and divided by `e2*(0)`), and symmetry error.
- Frozen-stage output drift: the snapshot's mean, α, and β. These must be exactly 0 in B1 and B2.
- In A, the drift of the live stage-2 mapping.

**Stage-1 recovery**
- `(ê_1(0) − e1*(0)) / e1*(0)`, signed and absolute.
- `σ_1(0)`.

**Strategic quality**
- `Ĝmax_full/ΔW`, with its argmax `(t*, d*)`.
- `EXP_root`, `dReach`, `Δmax_all`, `dFull`.
- `η_2`, and `Δ_2` split on-path / off-path.

**Induced stage-1 target.** This is analysis only; it never enters training.
- Definition: for the candidate's current stage-2 mapping `ê_2`, compute `ẽ_1[ê_2]`, the symmetric fixed point of the root stage game. It is the value `e` such that `e = argmax_{e'} Q_1(0, e' | opponent plays e at stage 1, continuation ê_2 for both players)`.
- Method:
  - Use the verifier's own `Q_1` machinery.
  - Solve by fixed-point iteration on the stage-1 action grid, followed by a local refinement.
  - Report convergence: the number of iterations and the final step size.
- Calibration test: for `ê_2 = e2*`, `ẽ_1` must equal `e1*(0)` to within the verifier floor.
- Decomposition to report: `ê_1 − e1* = (ê_1 − ẽ_1[ê_2]) + (ẽ_1[ê_2] − e1*)`. The first term is the stage-1 learning error; the second is the error inherited from the stage-2 continuation. Report both terms, signed and relative to `e1*(0)`.

**Optimization diagnostics**
- The C3 advantage statistics: mean and standard deviation over stage-1 rows and over all rows, at every update, in all arms.
- KL and clip fraction.
- Wall-clock time.
- RNG stream divergence per arm pair, as the first update at which each stream diverges.

---

## 3. Report: `reports/v2/pilot2_freeze.md`

Cite a source path for every number. The report contains:

1. **Final-checkpoint table.** One row per run, for all 60 runs.

2. **Paired comparisons** per (q, seed), for every metric in §2:
   - Comparisons: **B1 − A** (the freeze effect with identical normalization scope), **B2 − A**, and **B2 − B1** (the normalization-scope effect within frozen mode).
   - For each q:
     - the mean, SD, median, min, and max;
     - the sign count out of 10, with the direction of "better" stated per metric;
     - a 95% bootstrap CI of the mean paired difference, with the number of resamples and the bootstrap seed.

   These are descriptive statistics only.

3. **Learning curves vs update**, as the per-arm median and IQR across seeds, separately for each q:
   - on-path and off-path stage-2 drift;
   - Stage-1 relative error;
   - `Ĝmax_full/ΔW`;
   - `EXP_root`;
   - `dReach`;
   - both terms of the stage-1 decomposition.

4. **Location of `Ĝmax_full`.** The distribution of `(t*, d*)` per arm: how often the maximum sits at stage 1 or stage 2, and whether it falls on-path or off-path.

5. **Normalization scope.** Per update, the ratio of the stage-1 advantage SD computed over stage-1 rows to the SD computed over all rows, per arm. This quantifies how much the B1 and B2 updates differ in effective step size.

6. **Snapshot integrity.** Confirmation that the output drift is 0 in every B1 and B2 run.

7. **Records.** RNG divergence, would-have-fired updates, and wall-clock times.

8. **Interpretation.** In a separate, labeled section. Do not choose the frozen variant for Pilot 3.

**STOP.** I will choose between B1 and B2 for Pilot 3.

---

## 4. Optional side analysis on Pilot 1 data (no new runs)

**Delete this section if I have not asked for it.**

Question: how much of the remaining stage-2 peak underestimation in the `expected` arm comes from exploration noise? Under `expected`, the median signed peak error is still −0.13 at q=50 and −0.10 at q=60.

**Data.** For each of the 20 Pilot 1 `expected` final checkpoints, and for comparison the 20 `sampled` ones, use the saved Beta α(d) and β(d) on the evaluation grid.

**Prediction.** Compute the smoothed-game prediction

```
ê_pred(d) = (ΔW / 2k) · E[ f_ξ(d + a_i − a_j) ]
```

- `a_i` and `a_j` are drawn from the learned Beta policies at `d` and at `−d`, each centred on its own mean: that is, `a − mean`. `d + a_i − a_j` is therefore `d` plus the zero-mean action-noise difference.
- Evaluate the expectation by quadrature over both Beta distributions; report the node count. If the grid is not symmetric about 0, interpolate α and β to `−d` and say so.
- This is the stochastic-policy first-order condition (location-shift approximation). If the learner has converged to the equilibrium of the game whose noise is smoothed by its own action noise, then `ê_pred ≈ ê_learned`.

**Report** (append it to `reports/v2/pilot1_reward_estimator.md` as a separate section):
- per run, the RMSE over `|d| < 2q` of `ê_learned − ê_pred` and of `ê_learned − e2*`;
- at `d = 0`, the share of the peak gap explained: `(e2*(0) − ê_pred(0)) / (e2*(0) − ê_learned(0))`;
- one overlay plot per q of `e2*`, `ê_pred` (median across seeds), and `ê_learned` (median across seeds).

These are descriptive statistics only. If running this delays Pilot 2, do it after launching Pilot 2.
