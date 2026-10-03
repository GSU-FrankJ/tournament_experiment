# Phase 2 approved: three small changes, one report-only check, then Pilot 1

Phase 2 is approved. Everything from the earlier messages still applies unless changed here.

**Scope.** Do §1, then §2, then run Pilot 1 (§3). Then **STOP**. Do not choose the reward estimator.

---

## 1. Changes before Pilot 1

Commit all three changes, re-run the full test suite and the C7 regression on a clean tree, and record the new commit hash. Every Pilot 1 manifest must carry that hash.

### 1a. On-path rule: option (b), with exact cell masses for weighting

**On-path set.** It is the open interval

```
on_path = { d ∈ D_2 : |d − drift| < 2q }
```

- `drift` is the stage-1 effort difference under the evaluated policy, learner minus opponent at the root.
- **Compute `drift`; do not hardcode it.** Assert that it is 0 for the symmetric deterministic evaluation. If it is not, report the value.
- The nodes at exactly ±2q are off-path.
- Off-path is `D_2 \ on_path`.

**Cell masses for the weighted versions.** Use exact masses from `F_ξ`, not quadrature nodes:
- Cell boundaries are the midpoints between adjacent grid nodes. The outermost boundaries are the edges of `D_2`.
- The mass of cell i is `F_ξ(b_{i+1} − drift) − F_ξ(b_i − drift)`.
- Report the total mass captured on `D_2` (it should be 1 up to rounding), then normalize.

**Tests**
- Replace the current xfail test with these passing tests:
  - the on-path set equals `{|d| < 2q}` for the analytical equilibrium, for `ê ≡ 0`, and for a sample of the existing final checkpoints;
  - the cell masses sum to 1 within rounding.
- Keep a separate, clearly named xfail test that documents the node-based PMF mismatch found in check 1c.

**No effect on training.** This change affects reporting only. Confirm that C7 is still bit-exact.

### 1b. Seeds: expand to 10

- The seeds are **10501–10510**.
- Check that they are disjoint from every seed in the Phase 0 inventory. If any overlaps, stop and tell me before running anything.

### 1c. RNG stream positions

- After every update, log the position, or a state fingerprint, of each of the five random streams, per run.
- Add a utility that takes two paired runs and reports, per stream, the first update at which the positions differ. Report "never" if they stay aligned.
- This is diagnostic only. Do not change any sampler.

---

## 2. dReach reach-mask check (report only; do not change dReach)

The node-based forward PMF left holes inside the support. Check whether the reach mask `R_t` that dReach uses has the same problem.

**Candidates.** Run the check on:
- the analytical equilibrium, for q ∈ {50, 60};
- `ê ≡ 0`;
- the 80 existing final checkpoints.

**Reference support.** The reference is the continuous support of the root-BR path at stage 2: `{ d : |d − drift_BR| < 2q }`. Here `drift_BR` is the deviator's stage-1 BR effort minus `ê_1(0)`.

**What to report**
- **Mask comparison:** the number of grid points inside the reference support that are excluded from `R_2` (holes), and the number included from outside the support.
- **Effect on dReach:** next to the official dReach, compute an alternative value that uses the reference support as the mask, and report the difference in units of ΔW. This is for comparison only. The official dReach and its code stay unchanged.

Write the results to `reports/v2/dreach_reach_mask_check.md`. This check does not block Pilot 1; you may run it while Pilot 1 is running.

---

## 3. Pilot 1: terminal reward estimator (Phase A only)

**Design.** Phase A only, 400 updates, fixed budget. Every other setting is exactly as in the existing runner.

| | |
|---|---|
| Arms | `reward_mode = sampled` vs `expected` (A6 alignment: shocks are still drawn in `expected`) |
| q | 50, 60 |
| Seeds | 10501–10510, paired across arms |
| Runs | 2 × 10 × 2 = **40** |

**Launch**
- Run single-threaded processes in tmux.
- Choose the parallelism from `nproc` and the current load, and report both.
- Every run saves a **full-state end-of-Phase-A checkpoint**. These are the parents for Pilots 2 and 3.

**Per-checkpoint metrics** (all from `utils/v2_metrics.py`; evaluated policy = Beta mean):
- **Stage-2 recovery:**
  - peak relative error at `d = 0`, signed and absolute;
  - positive-region RMSE, raw and divided by `e2*(0)`;
  - tail mean and tail max effort, raw and divided by `e2*(0)`;
  - terminal symmetry error.
- **Residuals and policy spread:**
  - `η_2 = max_{d∈D_2} Δ_2(d) / ΔW`;
  - `Δ_2` split on-path / off-path with the §1a rule: unweighted max, and cell-mass-weighted mean;
  - `σ_2(0)`, and mean `σ_2` over `|d| < 2q`.
- **Diagnostics and cost:**
  - existing KL and clip-fraction diagnostics;
  - wall-clock time;
  - the update at which the existing Phase A stop rule would have fired, if it ever would;
  - full-policy metrics (`EXP_root`, `dReach`, `Ĝmax_full`), labeled `stage1_untrained`.

**Report:** `reports/v2/pilot1_reward_estimator.md`. Cite a source path for every number. It contains:

1. **Final-checkpoint table.** One row per run, for all 40 runs.
2. **Paired differences**, `expected − sampled`, per (q, seed), for every metric above. For each q, give:
   - the mean, SD, median, min, and max;
   - the sign count: how many of the 10 pairs favour `expected`, with the direction of "favour" stated per metric;
   - a 95% bootstrap CI of the mean paired difference. State the number of resamples and the bootstrap seed.

   These are descriptive statistics. Do not make significance claims beyond them.
3. **Learning curves** vs update, for peak error, normalized RMSE, tail mean, `η_2`, and `σ_2(0)`. Plot the per-arm median and IQR across seeds, separately for each q.
4. **Descriptive scatter.** Plot peak relative error against `σ_2(0)/q`, using every checkpoint of every run, coloured by arm, one panel per q. Report the Spearman correlation. Do not fit or claim any functional form.
5. **RNG alignment.** For each pair, report the first update at which each stream diverged.
6. **Anomalies and deviations from this prompt.**

Keep interpretation in a separate, labeled section.

**STOP.** I will choose the reward estimator.

---

## 4. Updated plan, for reference only (do not run)

| Pilot | Phase | Updates | Parents / arms | Runs |
|---|---|---|---|---|
| Pilot 2 | B | 600 | the 20 end-of-Phase-A parents produced with my chosen estimator (2 q × 10 seeds); arms A `joint` / B1 `frozen_allnorm` / B2 `frozen_s1norm` | 60 |
| Pilot 3 | B | 600 | the same 20 parents; my chosen frozen variant; `stochastic` vs `mean` | 40 |

- **Phase B reward mode:** in Pilots 2 and 3, Phase B uses the same reward estimator as the parent, as you proposed.
- **On-path definition:** Pilot 2 uses the on-path rule from §1a.
