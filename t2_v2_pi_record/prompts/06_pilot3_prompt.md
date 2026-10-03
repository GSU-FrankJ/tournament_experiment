# Pilot 2 accepted: B2 for Pilot 3, new ẽ₁ method, and a Phase A extension study in parallel

Pilot 2 is accepted. Everything from the earlier messages still applies unless changed here: phase gates, provenance discipline, closed form used for evaluation only, the on-path rule, the A6 RNG conventions, and no deletion of results.

**Scope.** Do §1 (no new runs). Then launch §2 (Pilot 3) and §3 (Phase A extension) in parallel. Then write §4. Then **STOP**.

## 0. Decisions

- **D1. Frozen variant for Pilot 3:** **B2**, that is `stage2_update_mode = frozen` with `adv_norm_scope = stage1_rows`.
- **D2. Induced stage-1 target ẽ₁.** Replace the fixed-point solver with residual minimization on the final tier, using the verifier's own one-step residual (§1). The old bracketing + Brent results stay in the Pilot 2 report as an appendix labeled "superseded".
- **D3. Phase A extension study.** Run it in parallel with Pilot 3 (§3).

Commit all new code before launching anything. Every manifest carries the launch commit hash and a clean-tree flag. Re-run the test suite and the C7 regression on that commit; C7 must still be bit-exact.

---

## 1. Re-compute ẽ₁ for Pilot 2 (analysis only, no new runs)

### Definition

For a candidate with stage-2 mapping `ê_2`, at the **final tier**:

1. **Sweep.** Sweep a stage-1 root effort `e` over a fine grid. The grid must cover the current `ê_1(0)` and `e1*(0)`, with margin on both sides. Report its range and resolution.
2. **Residual at each `e`.** For each `e`, set the stage-1 root action of **both** players to `e`. Keep `ê_2` as the continuation for both players, and evaluate the **verifier's own** one-step root residual:
   ```
   Δ_1(e; ê_2) = max_{e'} Q_1(0, e' | opponent plays e, continuation ê_2) − Q_1(0, e | opponent plays e, continuation ê_2)
   ```
   Use the verifier's existing action search for the max. Do not add a finer refinement than the verifier uses.
3. **Point estimate.** `ẽ_1 = argmin_e Δ_1(e; ê_2)`. Also report `Δ_1,min`.
4. **Uncertainty band.** The band is the set of sweep points with `Δ_1(e) ≤ Δ_1,min + floor`, where `floor = max(calibration floor for that q on the final tier, 1e-12·ΔW)`. The calibration floor is `Δ_1` at `(e1*, e2*)`. Report the band's endpoints, and its width relative to `e1*(0)`.

### Calibration gate

With `ê_2 = e2*`, the band must contain `e1*(0)`, at both q on the final tier. If it does not, stop and report before using the method anywhere.

### Decomposition

Re-compute the decomposition for every Pilot 2 checkpoint:
- **Learning term:** `ê_1 − ẽ_1`, reported as a point value plus an interval from the band.
- **Inherited term:** `ẽ_1 − e1*`, reported the same way.

Report both terms relative to `e1*(0)`, and state for each term whether its interval contains 0.

Append the revised decomposition to `reports/v2/pilot2_freeze.md` as a new section. Also say whether any Pilot 2 statement changes as a result.

---

## 2. Pilot 3: continuation action mode (stochastic vs mean)

### Design

- **Parents.** The same 20 `expected` end-of-Phase-A parents: q ∈ {50, 60} × seeds 10501–10510.
- **Phase B.** 600 updates, fixed budget. `reward_mode = expected`. Root start. The stage-1 opponent is the lagged copy, refreshed every 20 updates.
- **Mode.** Frozen B2: `stage2_update_mode = frozen`, `adv_norm_scope = stage1_rows`.
- **Arms:** `continuation_action_mode`:
  - `stochastic`: both players' stage-2 actions are sampled from the frozen Beta.
  - `mean`: both players' stage-2 actions are the frozen Beta mean. Draws are still consumed and discarded, per A6.
- **Runs:** 20 × 2 = **40**.

### Reproducibility check

The `stochastic` arm uses exactly the configuration of Pilot 2 arm B2. Its training history and final weights must be bit-identical to the corresponding Pilot 2 B2 runs. If they are not, stop and report before interpreting anything.

### Metrics

Log these at every verifier checkpoint, plus the weight exports every 25 updates. The evaluated policy is the Beta mean.

**Stage-1 accuracy**
- Relative error `(ê_1(0) − e1*(0)) / e1*(0)`, signed and absolute.
- The decomposition from §1, with bands. Stage 2 is frozen from the same parent, so the inherited term must be identical in both arms of a pair. Verify this numerically; arm differences in total Stage-1 error then equal differences in the learning term.

**Stage-1 stability**
- Within a run: the SD and range of `ê_1(0)` over the last 5 weight exports (updates 500, 525, 550, 575, 600).
- Across seeds: the SD and IQR of the final `ê_1(0)`, per arm and q.

**Policy spread:** `σ_1(0)`.

**Strategic quality:** `Ĝmax_full/ΔW` with `(t*, d*)`, `EXP_root`, `dReach`, `Δmax_all`, `dFull`.

**Snapshot integrity:** the frozen stage-2 output drift must be 0.

**Optimization diagnostics:** advantage statistics, KL and clip fraction, the would-have-fired update of the existing Phase B stop rule, RNG divergence, and wall-clock time.

### Note to include in the report

The verifier always evaluates the deterministic mapping. The `mean` arm therefore trains stage 1 against the same continuation that the verifier evaluates. The `stochastic` arm trains against a different continuation: the sampled one.

ẽ₁ is defined with the deterministic `ê_2`. So in the `stochastic` arm, the learning term also includes this mode mismatch.

### Report: `reports/v2/pilot3_continuation_mode.md`

It contains:
1. **Final table.** One row per run, for all 40 runs.
2. **Paired differences**, `mean − stochastic`, per (q, seed). For each q, give the mean, SD, median, min, max, the sign count out of 10 (with the direction of "better"), and a 95% bootstrap CI (state the number of resamples and the bootstrap seed).
3. **Learning curves.** Per-arm median and IQR across seeds, per q, for Stage-1 relative error, the learning term, `σ_1(0)`, `Ĝmax_full/ΔW`, and `EXP_root`.
4. **Stability table** (as defined under Metrics).
5. **Reproducibility check result.**
6. **Interpretation**, in a separate labeled section.

---

## 3. Phase A extension study (in parallel with Pilot 3)

**Question.** Does stage 2 keep improving after 400 Phase A updates? If so, where do peak error, `η_2`, and tail effort plateau?

### Design

- From the same 20 `expected` end-of-Phase-A parents, **continue Phase A**: stage-2-only, same start-state distribution, `reward_mode = expected`, same hyperparameters.
- Continue to **1600 total updates**.
- Save full-state checkpoints at 800, 1200, and 1600, and weight exports every 25 updates.
- **Runs:** 20.

### Schedules

First, report every schedule that depends on the update counter or on the Phase A budget: learning rate, entropy coefficient, clip range, anything else.

Beyond update 400, **hold every schedule at its end-of-Phase-A value**. If that needs a code change, add it as a flag that is off by default, and confirm that C7 is still bit-exact. Report exactly what was held.

### Metrics

Log at every verifier checkpoint and weight export:
- signed and absolute peak error;
- normalized RMSE;
- tail mean and tail max (raw and divided by `e2*(0)`);
- symmetry error;
- `η_2`;
- `Δ_2` on-path / off-path, unweighted max and cell-mass-weighted;
- `σ_2(0)`, and mean `σ_2` over `|d| < 2q`;
- KL and clip fraction;
- the would-have-fired update of the existing Phase A stop rule.

### Side analysis

Repeat the smoothed-game prediction from the Pilot 1 side analysis at updates 400, 800, 1200, and 1600. Report how the share of the peak gap it explains changes as training continues.

### Report: `reports/v2/phaseA_extension.md`

It contains:
- a table at 400 / 800 / 1200 / 1600: per-q median, IQR, and min/max across seeds;
- learning curves from 400 to 1600, as the median and IQR across seeds, per q;
- a description of where each metric plateaus, if it does.

These are descriptive statistics only. **Do not propose or set a precision criterion or a Phase A budget.** I will decide that.

---

## 4. Summary and stop

Write `reports/v2/summary.md`. It collects:
- Pilot 1, Pilot 2 (with the revised ẽ₁ decomposition), Pilot 3, and the Phase A extension;
- the key tables, each with source paths;
- an open-questions list.

Commit the lightweight records, following the Pilot 1 convention.

**STOP.**
