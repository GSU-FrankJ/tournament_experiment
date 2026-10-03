# Pilot 4 accepted: lock the v2 T=2 protocol, dress rehearsal on dev seeds, cusp diagnostic

Pilot 4 is accepted. This round turns the development results into a **locked protocol**. It then runs that protocol end to end on the **development** seeds as a dress rehearsal, plus one diagnostic that needs no RL runs.

**Do not run the fresh-seed confirmation.** It comes in the next round, after I approve this one.

Everything from earlier messages still applies unless changed here: provenance discipline, closed form used for evaluation only, no deletion of results, and reports with source paths.

**Scope:** §1 → §2 → §3 → §4, with §5 in parallel. Then write the report (§6) and **STOP**.

---

## 0. Decisions (binding; these define the locked protocol)

### D1. Training pipeline
- **Reward:** `reward_mode = expected`.
- **Phase A** (stage 2 only):
  - 1600 updates;
  - bin-balanced exploring starts on D₂, as now;
  - actor and critic LR: **3e−4 constant** for local updates 1–1200, then **linear 3e−4 → 3e−5** over local updates 1201–1600, using the existing `lr_at` linear form. This is the Pilot 4 §2a `decay` arm.
- **Freeze:** a frozen stage-2 snapshot at the end of Phase A, with the B2 machinery.
- **Phase B** (stage 1 only):
  - 600 updates;
  - `adv_norm_scope = stage1_rows`;
  - `continuation_action_mode = mean`;
  - root start;
  - lagged opponent refreshed every 20 updates;
  - actor and critic LR **linear 3e−4 → 3e−5** over local updates 1–600. This is the Pilot 4 §2b `decay` arm.
- **Not used:** joint training, Phase C, and tail averaging. **The evaluated candidate is the last iterate** (Beta mean): stage 2 at the end of A, stage 1 at the end of B.
- **Unchanged:**
  - RNG streams and samplers;
  - the on-path rule (open interval, exact cell masses);
  - the ẽ₁ residual-band method (final tier).

### D2. Gates (pre-registered)

Gates are applied with the **final-tier** verifier. Dev-tier values are also reported.

- **G-A, at the end of Phase A.** The candidate is the stage-2 last iterate. It passes only if all three hold:
  - `η_2 ≤ 0.005·ΔW`;
  - `RMSE_pos / e2*(0) ≤ 0.05`;
  - `tail_mean / e2*(0) ≤ 0.02`.
- **G-F, at the end of Phase B.** The candidate is the full last iterate. It passes only if both hold:
  - `Ĝmax_full / ΔW ≤ 0.01`;
  - `|ê_1(0) − e1*(0)| / e1*(0) ≤ 0.10`.
- **Run outcome.** A run **passes** if and only if it passes G-A and G-F.
  - A run that fails G-A is recorded as a **stage-2 failure**.
  - Its Phase B is still executed and reported as a diagnostic, but the run counts as failed.
- **Reported, not gated:**
  - stage 2: signed and location-free peak error at d = 0, symmetry error, tail max, `η_2` on-path and off-path, `σ_2(0)`, and the smoothed-game noise share at d = 0;
  - stage 1 and full policy: `EXP_root`, `dReach`, `Δmax_all`, `dFull`, the stage-1 decomposition (learning and inherited terms, with bands), and `σ_1(0)`;
  - numerics: dev-tier minus final-tier differences of every gate and reported metric.
- **Change from the 0929 plan.** Its development targets of 5% for stage-1 error and peak error are replaced by the gates above. This change is made **before** any confirmation run. The protocol file must state it, together with the Pilot 4 evidence (§6 distribution tables, and the 2b stability results).

### D3. Confirmation design

It is pre-registered now and executed only in the next round.
- q ∈ {50, 60}.
- **20 fresh seeds**, used for both q (40 runs).
- The seeds must be disjoint from every seed used so far: the Phase 0 inventory, 10501–10510, and anything else in any v2 manifest. **Propose a block (for example 20501–20520) and verify it is disjoint; I will confirm it.**
- **Pass rule:** for each q, at least **18 of 20** runs pass both gates.
- Every run is reported, including failures. Nothing may change after the lock.

### D4. Peak error

It is not gated. Report it with its decomposition. The diagnostic in §5 checks the part not explained by action-noise smoothing.

---

## 1. Consolidation and housekeeping

1. **One branch.** Bring the Pilot 4 commits onto `v2-stagewise-pilots` without content changes: a fast-forward if possible, otherwise a conflict-free merge, and report which. From now on there is **one** canonical worktree and branch; state its path.
2. **One development results root.** The original worktree and the Pilot 4 worktree (filled by rsync) both hold `results/v2_pilots/`. Declare one canonical copy. Verify with checksums that every parent file used in Pilot 4 is byte-identical between the two copies. Delete neither.
   - Locked-protocol outputs go to a **new** root, `results/v2_T2_locked/`: `rehearsal/` now, `confirmation/` in the next round.
3. **Dirty-flag runs.** Re-run **one** of the 8 dirty-flagged runs in a clean tree: Pilot 4 §2b, q = 60, seed 10507, `B2_mean_constant`. Confirm it is bit-identical to the original: history, weights, and end state.
4. **Known test failure.** `test_registry_canonicalization` fails on `main` as well and is unrelated to v2. Leave the test and the paper-registry data unchanged, and record the failure in the protocol file as known and pre-existing.

---

## 2. Lock the protocol

1. **Protocol files.**
   - Write `protocols/v2_T2_locked.json` (machine-readable) and `protocols/v2_T2_locked.md` (human-readable).
   - Include every setting in D1–D4, plus every **resolved** hyperparameter: PPO, network and Beta parameterization, grids and verifier tiers, verifier cadence, budgets, LR schedule endpoints, and seeds.
   - Also include: exact gate-metric definitions with the function each comes from; the pass rule; the proposed fresh-seed block; evidence pointers to the reports and sections behind every decision; and the change log versus the 0929 targets.
2. **Locked entry point.** One command runs the whole pipeline in **one process**: Phase A (1600, with the decay window) → G-A → freeze stage 2 → Phase B (600, decay) → G-F → final-tier evaluation.
   - **Writes:** a manifest (commit, clean-tree flag, SHA-256 of the protocol JSON, q, seed), a gates JSON, per-checkpoint metrics, and full-state checkpoints at the end of A and the end of B.
   - **Configuration:** it reads its configuration **only** from `protocols/v2_T2_locked.json`. It refuses to run if that file's hash differs from the hash recorded at the lock commit, or if any override is given other than q, seed and output directory.
3. **Tests.**
   - **Refusals:** the entry point refuses a modified protocol file and refuses extra overrides.
   - **LR schedule:** the per-update LR equals the locked schedule at every update of A and B, checked against `lr_at`.
   - **Legacy unchanged:** C7 is still bit-exact; the legacy runner is untouched.
   - **Full suite:** passes, apart from the known failure noted in §1.4.
4. **Lock record.** This commit is the **lock commit**. In a follow-up commit, add `protocols/LOCK` containing the lock commit hash, the protocol JSON hash, and the date. Nothing in `protocols/` may change after that without a new version number and a stated reason.

---

## 3. Calibration at the lock commit (no training)

Evaluate on the final tier and the dev tier, for both q.

**1. Analytic equilibrium `(e1*, e2*)`.**
- Report `Ĝmax_full/ΔW` (the floor), `η_2`, `EXP_root`, `dReach`, `Δmax_all`, and `dFull`.

**2. Zero-effort policy `ê ≡ 0`.**
- Report `Ĝmax_full/ΔW` and its location.
- Reference values from an independent PI-side computation (no repo code): **0.2593 at q = 50, maximum at stage 2, d ≈ −26**; **0.1955 at q = 60, maximum at stage 2, d ≈ −23.5**. The root dynamic gains were 0.243 and 0.191.
- Phase 1 reported 0.259 and 0.196. Confirm agreement.

**3. Comparison with Phase 1.** Report the differences from the Phase 1 values.

---

## 4. Dress rehearsal on the development seeds (not a confirmation)

**Runs.** q ∈ {50, 60} × seeds 10501–10510 = **20 runs**, through the locked entry point, end to end from scratch, written under `results/v2_T2_locked/rehearsal/`.

### Check 1: Phase A reproduces the development path

For each (q, seed), the rehearsal's end-of-Phase-A full state must be **bit-identical** to the stitched development state: Pilot 1 `expected` (u400) → Phase A extension (u1200) → Pilot 4 §2a `decay` (u1600).
- Compare: actor, critic, both optimizer states, the lagged opponent, all RNG states, and the stage-2 metrics at u1600.
- This is expected to hold, because re-entering at u400 and u1200 was shown not to affect training.
- **If it fails, stop and explain before doing anything else.**

### Check 2: Phase B code path

For two (q, seed) pairs, one per q, run Phase B with the pilot launcher (`B2_mean_decay`) from the stitched 2a-decay u1600 state. It must be bit-identical to the rehearsal's Phase B.

### Report per run

- G-A and G-F: each criterion's value (final tier and dev tier) and pass or fail;
- the run's overall outcome;
- every reported metric.

### Report per q

- the number of passes out of 10;
- the distribution of every gate metric (min, quantiles, max);
- the dev-tier minus final-tier differences.

These seeds were used for development, so **this is not evidence for the protocol**. It checks that the locked pipeline reproduces the development path. Whatever the outcome, change nothing; report it.

---

## 5. Cusp diagnostic (no RL runs; can run in parallel with §4)

**Setup.** Re-run the Pilot 4 §1d supervised fit with the same setup: same network and head, same grid, full-batch Adam with lr 1e−3, the same 5 initializations per q, and a cap of 300k steps.

**Log every 500 optimizer steps:**
- signed peak error at d = 0;
- location-free peak error;
- RMSE/`e2*(0)`.

**Report:**
- **Peak thresholds:** per q, the first step at which |peak error| falls below 0.05, 0.03 and 0.01, as the median and range over the inits.
- **RMSE:** the RMSE at the step where the peak error first falls below 0.05.
- **RL comparison:** the number of actor optimizer steps that stage 2 actually receives in RL Phase A, computed from the locked config as updates × epochs × minibatches per update. Give it for all 1600 updates and for the last 400.

**Purpose.** This tests, descriptively, whether the part of the RL peak gap that action-noise smoothing does not explain is consistent with this network learning the cusp slowly (spectral bias). It changes nothing in the method.

---

## 6. Report: `reports/v2/protocol_lock_and_rehearsal.md`

Sections:
1. consolidation and housekeeping, including the checksum and dirty-run results;
2. the lock: commit, files, hashes, and the change log;
3. calibration;
4. rehearsal: both bit-identity checks, the per-run gate table, pass counts, distributions, and dev-tier minus final-tier differences;
5. cusp diagnostic;
6. anomalies and deviations;
7. commands to reproduce.

Update `reports/v2/summary.md`.

**STOP.** Do not launch the confirmation.
