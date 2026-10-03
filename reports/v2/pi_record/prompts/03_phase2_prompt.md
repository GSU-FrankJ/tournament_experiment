# Phase 1 approved — two opening checks, then Phase 2 only

Phase 1 is approved. Two numbers from it are now the recorded numerical floor of the verifier for analytical-equilibrium policies:

- **q = 50:** ≤ 2.2e-16·ΔW.
- **q = 60:** up to 3.5e-7·ΔW, on the finer grids, at the root.

Everything from the original task prompt and from the Phase 0 approval message (sections A–E) still applies, except where this message changes it.

**Scope of this message.** Run the two opening checks (§1), then Phase 2 (§3), then **STOP**. Run no pilots.

---

## 1. Opening checks (before any Phase 2 code)

### 1a. Numerical check of the verifier at non-equilibrium policies

Phase 1 established two things: the verifier returns ≈0 at the equilibrium, and a large value for the zero-effort policy. It did not establish that the *values* it returns away from equilibrium are correct.

Check that now against an independent reference, at the terminal stage, where `Δ_2(d)` has a one-dimensional closed form.

**Reference implementation**
- Write the reference as a separate function that does **not** import or reuse any verifier search or interpolation code.
- It uses the environment's exact `F_ξ`, which was already validated against Monte Carlo in Phase 1.
- It does a dense search over `e ∈ [0, 100]`, followed by a local refinement (bounded scalar minimization, or golden-section search) around the best grid point. Report the resolution you used.

**Formula.** For a test policy `ê_2`, with the opponent playing `ê_2(−d)`:

```
Δ_2^ref(d) = max_{e∈[0,100]} [ ΔW·F_ξ(d + e − ê_2(−d)) − k e² ]  −  [ ΔW·F_ξ(d + ê_2(d) − ê_2(−d)) − k ê_2(d)² ]
```

(`W_L` cancels.)

**Test policies**
1. `ê_2 ≡ 0`.
2. An **asymmetric** policy, so that the opponent-mirroring logic `ê_2(−d)` is actually exercised. Use the closed form shifted by `q/2`: `ê_2(d) = e2*(d − q/2)`. This stays inside `[0, 100]`.

**Comparison**
- Compare `Δ_2^ref(d)` with the verifier's `Δ_2(d)` at every grid point of `D_2`.
- Do this for q ∈ {50, 60}, at the default grid and at the 2× finer grid.
- Report, in units of ΔW:
  - the maximum absolute difference and its location;
  - the sign pattern. A grid-search maximum should not exceed the continuous maximum by more than rounding. A positive excess beyond rounding is a bug and must be reported.
  - whether the difference shrinks on the finer grid.

**Gate.** If the differences are not explained by action- and state-grid resolution, stop and report before going any further.

### 1b. q = 60 floor: quick look only

Report whether `e1*(0)` lies exactly on the stage-1 action grid, for q = 50 and q = 60, at the default grid and at both finer variants.

No code changes, and no further investigation. This only checks whether the grid can explain why the floor is non-zero only on the finer grids for q = 60.

### 1c. On-path rule: confirmed, and add one test

Keep the rule as it is: a state is on-path only if its probability is strictly greater than zero, with no threshold.

Add a test that this on-path set equals `{d ∈ D_2 : |d| < 2q}`, for the following candidates:
- the analytical equilibrium;
- `ê ≡ 0`;
- a sample of the 80 existing final checkpoints.

The rationale: at evaluation, both players play the same deterministic `ê_1(0)` at the root, so `d_2 = ξ_1`. This identity requires stage-1 and stage-2 shocks to be identically distributed; confirm that from the code. If the sets differ, report where and why.

Write the results of 1a–1c into `reports/v2/phase2_opening_checks.md`. Then continue to Phase 2, unless the gate in 1a fired.

---

## 2. Decision: Pilot 2 now has three arms

The two frozen arms differ only in the advantage-normalization scope. This separates the effect of freezing from the effect of the normalization scope. Pilot 2 is therefore:

| Arm | `stage2_update_mode` | `adv_norm_scope` | Stage-2 actions (both players) | Stage-2 rows in policy loss | Normalization statistics computed over |
|---|---|---|---|---|---|
| A `joint` | joint | all_rows | live policy (learner) / lagged copy (opponent), sampled, as now | yes | all rows, as now |
| B1 `frozen_allnorm` | frozen | all_rows | frozen snapshot | **no** | all rows (stage-1 + stage-2), same computation as joint |
| B2 `frozen_s1norm` | frozen | stage1_rows | frozen snapshot | **no** | stage-1 rows only |

- **Common to all arms:** stage-1 opponent = the existing lagged copy, refreshed every 20 updates. Continuation action mode = `stochastic`.
- **Stage-2 advantages in B1.** B1 still computes advantages for the stage-2 rows exactly as joint does, from the same GAE and critic. They enter the mean/std only, never the gradient. In B1 and A the normalization statistics are computed identically; they differ only because the stage-2 data come from different policies. Document this in the Pilot 2 report.
- **Valid combinations.** `joint` is valid only with `adv_norm_scope=all_rows`; the entry point must refuse `joint + stage1_rows`.
- **Pilot 3.** Pilot 3 uses whichever frozen variant (B1 or B2) I choose after Pilot 2.

---

## 3. Phase 2: training infrastructure (no pilot runs)

Implement the original Phase 2, the Phase 0 amendments C1–C8, and the changes below.

### 3a. Flags

- `stage2_update_mode ∈ {joint, frozen}`
- `adv_norm_scope ∈ {all_rows, stage1_rows}`, default `all_rows`
- `reward_mode ∈ {sampled, expected}`
- `continuation_action_mode ∈ {stochastic, mean}`

All of them are recorded in the manifest. Defaults reproduce the existing runner bit-exactly (C7).

### 3b. Additional tests

On top of those listed in C1–C7:

1. **Gradient isolation in both frozen arms.** Perturb only the stage-2 rows' advantages: the policy-loss gradient must be exactly zero. Perturb only the stage-1 rows: the gradient must be nonzero.
2. **Snapshot immutability.** After N updates in both frozen arms, the snapshot's parameters, its normalizer statistics, and its stage-2 mean/α/β on the `D_2` grid are bit-identical to their values at freeze time.
3. **Normalization scope.** In B1 the normalization mean/std equal those computed over all rows. In B2 they equal those over stage-1 rows only. Check both against a direct recomputation.
4. **Joint mode unchanged.** With `stage2_update_mode=joint`, policy-loss values and gradients are bit-identical to the existing runner on the same batch.
5. **RNG alignment** (A6). After one episode, and again after 20 updates, the RNG states are identical:
   - between `sampled` and `expected`;
   - between `stochastic` and `mean`;
   - across A, B1 and B2, when all three start from the same parent checkpoint. (They consume randomness the same way, since stage-2 actions are drawn in every arm.)

   If any pair cannot be aligned exactly, report the source.
6. **Branch reproducibility** (C1). Two branches from the same end-of-Phase-A checkpoint with identical flags give bit-identical Phase B metrics for at least 20 updates. Run this for each of A, B1 and B2.

### 3c. Smoke runs only

- Using a throwaway end-of-Phase-A checkpoint (q = 50, seed 10501, reduced Phase A budget), run each arm for about 20 Phase B updates. The arms are A, B1 and B2, plus `frozen + mean` and `reward_mode=expected` in Phase A.
- Purpose: confirm that everything runs, logs the C3 advantage statistics and the per-phase wall-clock time, writes manifests, NPZ/CSV files, and plots, and passes the drift test.
- Put the outputs under `results/v2_pilots/_smoke/`. These are not pilot results; do not interpret them.

### 3d. Report and stop

- Commit on `v2-stagewise-pilots`.
- Write `reports/v2/phase2_infra.md` containing: changed files, all test results, the smoke-run checklist, the measured per-phase wall-clock times, and the planned parallelism (with `nproc` and the current machine load).
- **STOP** and wait for approval before Pilot 1.

---

## 4. Updated pilot plan, for reference only (do not run)

- **Pilot 1:** Phase A, 400 updates. `sampled` vs `expected`. q ∈ {50, 60} × seeds {10501, 10502, 10503} = 12 runs. Each run saves a full-state end-of-Phase-A checkpoint.
- **Pilot 2:** Phase B, 600 updates, fixed budget. Three arms A / B1 / B2, branched from the 6 parents produced with the estimator I choose = 18 runs.
- **Pilot 3:** Phase B, 600 updates. My chosen frozen variant. `stochastic` vs `mean`, from the same 6 parents = 12 runs.
- **Not in scope:** Phase C, formal runs, and T=3.
