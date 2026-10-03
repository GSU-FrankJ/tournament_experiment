# Pilots 1–3 closed: stabilization round (Pilot 4) before the protocol lock

Pilot 3, the Phase A extension, and the revised ẽ₁ decomposition are accepted. Everything from earlier messages still applies unless changed here: phase gates, provenance discipline, closed form for evaluation only, the on-path rule, the A6 RNG conventions, and no deletion of results.

**Scope.** Do the analyses in §1 (no new runs) and the runs in §2. Write the report in §3. Then **STOP**.
- Do not lock the protocol.
- Do not propose gate thresholds.
- Do not run any formal or fresh-seed experiment.

## 0. Decisions (binding from now on)

- **D1. Continuation mode = `mean`** for all further work.
- **D2. Phase A** is a fixed budget of **1600 updates**, followed by an end-of-phase **pass/fail gate** on η₂ and the tail/RMSE metrics. I will set the thresholds after this round. The single-point peak error at d = 0 is **not** a gate metric until §1d is in.
- **D3. Joint training is dropped from the v2 protocol**, and so is Phase C.
- **D4. No sampler change.** The RNG streams stay as they are, and inverse-CDF sampling is not implemented.
- **Already decided:**
  - `reward_mode = expected`;
  - frozen variant B2 (`adv_norm_scope = stage1_rows`);
  - ẽ₁ by residual minimization on the final tier;
  - on-path set = `{|d − drift| < 2q}` with exact cell masses.
- **Seeds.** This round still uses the development seeds 10501–10510 and q ∈ {50, 60}.

Commit all new code before launching anything. Re-run the full test suite and the C7 regression on the launch commit (C7 must still be bit-exact), and record that commit in every manifest.

---

## 1. Analyses on existing data (no new runs)

### 1a. Check the analytic reference values for the root stage game

The PI side derived the following from the closed form. They are **reference values to be checked, not established facts**. With `e2*` as the continuation for both players, at `e1*(0)`:

| q | BR slope dBR/de_opp | E[V₂″] / (2k) | own curvature ∂²Q₁/∂e_i² |
|---|---|---|---|
| 50 | −0.961 | 0.490 | −2k·(1 − 0.490) |
| 60 | −0.309 | 0.236 | −2k·(1 − 0.236) |

**Check them numerically** with the verifier's own `Q_1` on the final tier.

**Best response.** `BR(e_opp) = argmax_e Q_1(0, e | opponent plays e_opp at stage 1, continuation e2* for both players)`. The verifier's grid search makes BR piecewise constant, so for this diagnostic only, locate the maximum at sub-grid resolution with a local quadratic fit of `Q_1` around the grid maximum. Report the method.

**Slope.** Central differences at `e_opp = e1* ± h`, for a few step sizes `h`. Report each `h` used.

**Own curvature.** Take it from the same quadratic fit at `(e1*, e1*)` and express it relative to `2k`.

**Report** the numeric values next to the reference values. If they disagree beyond the finite-difference variation, say so plainly.

### 1b. Anatomy of the stage-1 fluctuation (Pilot 3 runs, both arms)

**Record.** Use the densest record of `ê_1(0)` that exists, and state which one you used:
1. a per-update training log of the root policy mean;
2. otherwise the verifier checkpoints;
3. otherwise the 25-update weight exports.

**Segment.** Use global updates ≥ 650; the Pilot 3 curves reach a median of about 0 by roughly u600–650. Also report the results when the segment starts at u700.

**Report**
- **Autocorrelation:** the ACF of `ê_1(0) − e1*` at the available lags, per q and arm, as the median and IQR across seeds.
- **Window regression** (only if per-update data exist):
  - Within each 20-update opponent window, regress the learner's change in `ê_1(0)` on the opponent's deviation `e_opp − e1*` at the start of the window.
  - Report the slope with a bootstrap CI, per q.
  - Put it next to the BR slope from 1a, descriptively only. The learner moves only part of the way toward BR within a window, so the two are not expected to be equal.
- **If only the 25-update exports exist:** state that the 20-update refresh cycle cannot be resolved, and report the ACF at lags 25, 50, 75, and 100 only.

### 1c. Tail-averaged candidates (post hoc)

**Definition**
- `candidate_K` is the pointwise average of the deterministic (Beta-mean) mapping over the last K weight exports of a run, for each stage trained in that run.
- `K = 1` is the last iterate.
- Frozen stages are unchanged.
- Build `candidate_K` as a deterministic policy object the verifier can evaluate. It is exact at grid points; off-grid, use the same interpolation the verifier uses for any other policy. State how this is done.

**Apply it to**
1. **Pilot 3, both arms, stage 1:** K ∈ {1, 4, 8, 12}. K = 12 covers u725–u1000, inside the converged segment.
   - Report: Stage-1 relative error (signed and absolute), the decomposition against the parent band (the inherited term is unchanged), `Ĝmax_full/ΔW` with `(t*, d*)`, `EXP_root`, and `dReach`.
2. **Phase A extension at u1600, stage 2:** K ∈ {1, 4, 8, 16}.
   - Report the stage-2 metrics only: signed and absolute peak error, location-free peak error (defined in 1d), normalized RMSE, tail mean and tail max (raw and divided by `e2*(0)`), symmetry error, `η_2`, and `Δ_2` on/off-path.
   - Stage 1 is untrained in these runs, so do not report full-policy metrics.

**Paired comparisons:** K versus K = 1 per (q, seed), with the median, sign count, and 95% bootstrap CI.

### 1d. Representation floor of the stage-2 actor (diagnostic only)

This is a diagnostic. It is never a training target, never used to initialize any run, and never enters the method.

**Setup**
- Use the **same** stage-2 actor architecture, input encoding, initialization scheme, and output head, including the Beta parameterization, the concentration bounds, and any clipping.
- Fit the **Beta mean** to `e2*(d)` by least squares over the development `D_2` grid, with uniform weights.
- Train with Adam until the loss plateaus. Report the learning rate, the number of steps, and the plateau rule.
- Run 5 random initializations per q.

**Report, per fit and as the median**
- signed peak error at `d = 0`;
- **location-free peak error**, `(max_d ê_2(d) − e2*(0)) / e2*(0)`, with the argmax location;
- RMSE divided by `e2*(0)`;
- tail mean and tail max;
- symmetry error;
- the verifier's `η_2` and `Ĝmax_full/ΔW` for the complete candidate (stage 1 = `e1*(0)`, stage 2 = the fitted mapping).

If the output head cannot reach the target anywhere (for example near 0 in the tails), report where.

**Comparison** with the RL stage-2 policies at u1600 from the extension. Per q, show three things side by side:
- the RL peak gap at `d = 0`;
- the part predicted by action-noise smoothing (the Pilot 1 / extension method);
- the supervised-fit floor.

These are descriptive only.

---

## 2. New runs: learning-rate decay (one variable per comparison)

**Common settings**
- `reward_mode = expected`, fixed budgets, single-threaded processes in tmux.
- Choose the parallelism from `nproc` and the current load, and report both.
- Every manifest carries the launch commit hash.

**Decay schedule for the `decay` arms**
- Use the **existing Phase C linear schedule form** (`lr_at`) for both actor and critic, mapped onto the arm's decay window.
- Report the exact start and end LR values and the code location.
- Do not introduce new LR values. If the existing end value is greater than 0, keep it as is.

### 2a. Phase A end-of-phase decay

- **Parents:** the Phase A extension's `state_u01200.pt` (20 runs).
- **Arms**
  - `constant`: re-run u1200→u1600 exactly as in the extension.
  - `decay`: LR decays over u1200→u1600.
- **Runs:** 40.
- **Reproducibility check.** The `constant` arm's u1600 weights must be bit-identical to the extension's u1600 weights. Verifier-cadence differences on re-entry should not matter, because the verifier consumes no RNG; confirm this. If the weights differ, stop and report.
- **Metrics:** the stage-2 set at every export and at the end. Also tail-averaged candidates with K ∈ {1, 4, 8} for both arms.

### 2b. Phase B decay

- **Parents:** the Phase A extension's `state_u01600.pt` (20 runs, constant-LR Phase A).
  - Stage 2 is frozen as a snapshot at u1600.
  - The 2a `decay` states are **not** used. This keeps the comparison to a single variable, and allows 2a and 2b to launch in parallel.
- **Settings:** B2 normalization, `continuation_action_mode = mean`, 600 Phase B updates (global 1601–2200), root start, lagged opponent refreshed every 20 updates.
- **Arms**
  - `constant`
  - `decay`: LR decays over the whole of Phase B.
- **Runs:** 40.
- **Induced-target bands.** Compute the ẽ₁ residual bands for the 20 new parent snapshots with the same method (final tier). Widen the sweep, or extend it adaptively, so that **no** row of either arm falls outside it. Report the range used.
- **Metrics**
  - everything reported in Pilot 3: Stage-1 relative error, the decomposition, within-run SD and range over the last 5 exports, across-seed SD, `σ_1(0)`, `Ĝmax_full/ΔW` with `(t*, d*)`, `EXP_root`, `dReach`, `Δmax_all`, `dFull`, advantage statistics, KL and clip fraction, the would-have-fired update, RNG divergence, and wall-clock time;
  - tail-averaged candidates with K ∈ {1, 4, 8, 12} for both arms.

---

## 3. Report: `reports/v2/pilot4_stabilization.md`

Cite a source path for every number. All statistics are descriptive. The report contains:

1. **1a:** reference versus numeric table.
2. **1b:** fluctuation anatomy.
3. **1c:** tail-averaging tables, with paired comparisons against K = 1.
4. **1d:** representation-floor table, and the three-way comparison of the peak gap.
5. **2a and 2b:**
   - final tables;
   - paired differences (`decay − constant`) per (q, seed): median, sign count with the direction of "better", and 95% bootstrap CI;
   - learning curves, as the median and IQR across seeds;
   - the reproducibility check result.
6. **Distribution tables for setting gates** (no threshold proposals).
   - **Candidates:** each end-of-phase candidate:
     - Phase A ∈ {constant, decay} × K ∈ {1, 4, 8};
     - Phase B ∈ {constant, decay} × K ∈ {1, 4, 8, 12}.
   - **Quantiles:** min, 10%, 25%, median, 75%, 90%, max, across the 10 seeds of each q.
   - **Metrics** (where defined): `η_2`, RMSE/`e2*(0)`, tail mean/`e2*(0)`, tail max/`e2*(0)`, signed and absolute peak error, location-free peak error, absolute Stage-1 relative error, `Ĝmax_full/ΔW`, `EXP_root/ΔW`.
7. **Interpretation**, in a separate labeled section.

Update `reports/v2/summary.md` with the new rows and decisions. Commit the lightweight records, following the earlier convention.

**STOP.**
