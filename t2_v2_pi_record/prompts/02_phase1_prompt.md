# Phase 0 approved — decisions, amendments, and go-ahead for Phase 1 only

Thanks for the audit. Phase 0 is approved. The decisions and amendments below are **binding for every remaining phase**. Where they conflict with the original task prompt, they override it. Everything in the original prompt that is not changed here still applies: phase gates, provenance discipline, "closed form for evaluation only", no deletion of old results, and per-phase reports.

**Execute Phase 1 only, then STOP.** Sections B–E record how the later phases must be done, so that Phase 1 builds what they need. Do not start Phase 2.

---

## A. Decisions (binding)

**A1. Repository and branch base**
- The repository is `/home/fjiang4/tournament_experiment`.
- Create `v2-stagewise-pilots` from commit `657f54a` (branch `codex/multistage-experiments-20260927`), not from `main`.
- Record the base commit in every manifest and every report.

**A2. Pilot grid**
- q ∈ {50, 60}.
- Seeds ∈ {10501, 10502, 10503}.
- Before using these seeds, confirm they are disjoint from every seed in your Phase 0 inventory. If any overlaps, stop and tell me.

**A3. Phase mapping**
- **Stage-2-only training** = existing **Phase A**: stage 2 only, 400 updates.
- **Stage-1 training** for Pilots 2 and 3 = existing **Phase B only**: both stages from the root, run at its full budget of 600 updates under fixed-budget mode.
- **Phase C is not run** in any pilot.
- Pilot 2 and Pilot 3 branch from full-state checkpoints saved at the **end of Phase A**.

**A4. Frozen mode** (`stage2_update_mode=frozen`, applies in Phase B only)
- **Snapshot.** Take the stage-2 snapshot from the end-of-Phase-A state.
- **Stage-2 actions.** Both players' stage-2 actions, the learner's and the opponent's, come from the frozen snapshot.
- **Stage-1 opponent.** The opponent's stage-1 action keeps coming from the existing lagged copy of the policy, refreshed every 20 updates exactly as now.
- **Policy loss.** Stage-2 rows are excluded from the policy loss.
- **Advantages.** Advantages are normalized over the stage-1 rows only.
- **Critic.** The critic/value loss is unchanged.

**A5. Joint mode** (`stage2_update_mode=joint`)
- Must stay exactly as it is now, including advantage normalization over both stages together and the lagged opponent for both stages.

**A6. Random-number alignment between paired arms**
- In `reward_mode=expected`, the terminal performance shocks are still drawn, then not used for the reward. This keeps RNG consumption identical to `sampled` mode.
- In `continuation_action_mode=mean`, stage-2 actions are still drawn from the frozen Beta, then discarded in favour of the Beta mean. This also keeps RNG consumption identical.
- Discarded draws must not enter rewards, stored actions, log-probs, logs, or any metric.

**A7. Test tooling**
- Install `pytest` into the repo's `.venv` (`.venv/bin/python -m pip install pytest`), never into system Python.
- Record the installed version in the Phase 1 report.
- If the install fails (no network, permissions), stop and report the exact error.

**A8. Compute**
- Runs are CPU-only and single-threaded. Keep each process single-threaded, matching the existing runs; pin thread environment variables if the existing launcher does.
- You may run independent runs as parallel processes. Choose the degree of parallelism from `nproc` and the current load on the machine, and report both.
- Add per-phase wall-clock logging to the v2 path, so that phase-level cost is no longer UNKNOWN.

---

## B. Phase 1 — what to do now (verifier and metrics; no training-loop changes)

These are amendments to the original Phase 1, in light of the audit.

**B1. Primary metric by aggregation**
- Ĝmax_full is the aggregation of the existing full-grid `V_t^BR` and `V_t^ê` arrays.
- Before aggregating, document from the code exactly what these arrays are:
  - the deviator re-optimizes from `(t, d)` through every later stage;
  - the opponent is fixed at `ê` and plays `ê_t(−d)` from its own perspective;
  - the action grid or search used;
  - the interpolation used.
- Report the argmax location `(t*, d*)` together with the value.
- Make sure `d = 0` is an exact grid point, so the peak error is evaluated at 0 and not at an interpolated point.

**B2. Evaluated policy**
- Use the Beta mean, as confirmed in the audit, for all recovery and verifier computations.

**B3. Invariant tests**
- Implement the invariant tests from the original prompt (§2, Phase 1 item 5) as pytest tests and as runtime checks inside the verifier.
- For `Δmax_all ≤ Ĝmax_full`, a violation is expected from the different action candidates the two searches use. For this relation:
  - report the maximum violation in units of ΔW, and where it occurs `(t, d)`;
  - report whether the violation shrinks on the 2× finer action grid;
  - do not clamp it, and do not change either search to hide it.
- Report every other relation the same way: maximum violation, location, and units of ΔW.

**B4. On/off-path decomposition** (needed for Pilot 2)
- Implement the decomposition now. On-path = grid states in `D_2` with positive probability under the candidate's own stage-2 state distribution from the root, using the verifier's forward PMF. Off-path = the rest of `D_2`.
- Provide both unweighted and PMF-weighted versions of the stage-2 drift and of `Δ_2`.
- State in the report how "positive probability" is decided numerically: the exact threshold, or exact zeros.

**B5. σ(d) logging and recovery metrics**
- Implement them as specified in the original Phase 1 (items 2–4), on the existing evaluation grid.

**B6. Benchmark consistency and verifier calibration** (original Phase 1 items 6–7)
- Run both for **q = 50 and q = 60**, using the as-run ΔW, k, and action bounds from the audit.
- Calibration covers:
  - the analytical equilibrium, at the default grid and at the 2× finer state and action grid;
  - the zero-effort policy `ê ≡ 0`.

**B7. No side effects on training**
- Phase 1 must not change any training trajectory.
- Confirm that the verifier does not draw from any training RNG stream.
- Then run the existing runner on one short configuration (q = 50, seed 10501, reduced budget) before and after your Phase 1 changes, and show that all training-side logs are identical.

**B8. Commit and stop**
- Commit on `v2-stagewise-pilots`.
- Write `reports/v2/phase1_verifier.md` (format as in the original §3).
- **STOP** and wait for approval.

---

## C. Amendments to Phase 2 (infrastructure) — do not start yet

- **C1. Full-state checkpoints.** Save one at the end of Phase A, and at any phase boundary a branch may start from. It contains:
  - model parameters;
  - optimizer state;
  - **all five RNG states** identified in the audit;
  - update counters;
  - normalizer statistics, if any;
  - schedule positions;
  - **the lagged opponent copy, together with its refresh counter / next-refresh step.**

  Test: restoring the checkpoint into two branches with identical flags gives identical Phase B metrics for at least 20 updates. On CPU this should be bit-exact; any discrepancy is a bug to report.
- **C2. Reward mode.** `reward_mode=expected` must follow A6. Tests:
  - the RNG state after an episode is identical in `sampled` and `expected` mode;
  - the mean of many `sampled` terminal rewards matches `r̄` within MC error on a grid of `(d, e_i, e_j)`, for both players.
- **C3. Frozen and joint modes.** Implement them exactly as A4 and A5 specify. In both modes, log at every update:
  - the advantage mean and standard deviation over stage-1 rows only;
  - the same statistics over all rows.

  This is diagnostic only, so that the normalization-scope difference between the arms can be quantified.
- **C4. Continuation action mode.** `continuation_action_mode=mean` follows A6. Test: RNG state alignment between `stochastic` and `mean` after an episode.
- **C5. Output-drift test.** Run it at the end of Phase B.
  - Frozen mode: the snapshot's stage-2 mean, α, and β must be unchanged.
  - Joint mode: log the drift of the live network's stage-2 mapping, split on-path / off-path (B4).
- **C6. Fixed budget.**
  - Phase A runs 400 updates. Report confirmation that the existing stop rule would not have fired, or the update at which it would have.
  - Phase B runs 600 updates, with no early exit. Record the update at which the existing Phase B stop rule would have fired.
- **C7. Regression test.** With every new flag at its default, the v2 entry point reproduces the existing runner bit-exactly on CPU for a short run (q = 50, seed 10501).
- **C8. Pilot launcher.** Provide a launcher that can:
  - run Phase A only;
  - run Phase B only from a named parent checkpoint;
  - write manifests under `results/v2_pilots/<pilot>/q<q>/seed<seed>/<arm>/`.

---

## D. Amendments to Phases 3–5 (pilots) — do not start yet

- **D1. Pilot 1.** Phase A only, 400 updates. Arms: `sampled` vs `expected`. q ∈ {50, 60}, seeds {10501, 10502, 10503}, so 12 runs. Save a full-state end-of-Phase-A checkpoint for every run. I choose the estimator afterwards.
- **D2. Pilot 2.** Branch Phase B (600 updates) from the 6 end-of-Phase-A checkpoints produced with my chosen estimator. Arms: `joint` vs `frozen`. The continuation action mode is the existing default, `stochastic`. In the report, include the per-update advantage-statistics comparison from C3, so the normalization-scope difference is visible next to the freeze effect.
- **D3. Pilot 3.** Phase B (600 updates) from the same 6 parents, `frozen` in both arms. Arms: `stochastic` vs `mean`. The mode applies to both players' stage-2 actions.
- **D4. Not in scope.** Phase C, formal T=2 runs, and T=3 runs are all out of scope.

---

## E. Reminder

- Every number in a report carries its source path.
- Interpretation is labeled and kept separate from results.
- Anything unexpected: stop and ask.
