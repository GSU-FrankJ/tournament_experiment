# Lock round accepted: protocol v1.1, re-rehearsal, then the confirmation

The lock and rehearsal round is accepted, with the decisions in §0. This round has four steps:
1. Amend the protocol to **v1.1** and lock it (§1).
2. Re-run the rehearsal on the development seeds under v1.1, with automatic checks (§2).
3. **Only if every check passes**, launch the fresh-seed confirmation without waiting for me (§3).
4. Write the report (§4), then **STOP**.

Work in the canonical worktree `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2`, on branch `v2-stagewise-pilots`.

Everything from earlier messages still applies unless changed here:
- provenance discipline;
- the closed form is used for evaluation only;
- no deletion or overwriting of results or checkpoints;
- pytest only in `.venv`;
- the paper-registry test and its data stay unchanged;
- single-threaded processes in tmux;
- no push.

**If any step fails, or anything does not match this prompt, stop at that point and report. Do not work around it.**

---

## 0. Decisions (binding)

### D1. Check 1 is accepted

In 20/20 runs, the training state was bit-identical to the stitched development path. The three process-global RNG states (torch global, numpy legacy global, Python `random`) were never consumed by training. They are diagnostics only.

**Training-relevant state.** From now on, every bit-identity check compares exactly this:
- the actor, the critic, the lagged opponent, the frozen stage-2 snapshot (Phase B), and both Adam states;
- the five training RNG streams (env, learn, opp, start, minibatch) and the torch generator;
- all weight exports and the final weights;
- training histories and per-update logs, without their wall-clock fields;
- checkpoint metrics, gate-metric values, and the evaluation outputs (`gateA_*.npz`, `final_*.npz`, `induced_band.json`, `band_sweep.npz`, `drift_test.json`).

Excluded: verdict fields, manifests, and the three process-global RNG states.

Record this decision as a dated addendum appended to `reports/v2/protocol_lock_and_rehearsal.md`. Do not edit the existing text.

### D2. Primary run pass (v1.1)

All values are on the final tier unless stated otherwise. Every "≤" is inclusive. Compare at full precision, with no rounding.

- **G-A** (unchanged): the stage-2 last iterate at the end of Phase A, with three criteria.
  - `η_2/ΔW ≤ 0.005`.
  - `RMSE_pos/e2*(0) ≤ 0.05`.
  - `tail_mean/e2*(0) ≤ 0.02`.
- **G-F (v1.1)**: the full last iterate at the end of Phase B, with one criterion.
  - `Ĝmax_full/ΔW ≤ 0.01`.
  - The stage-1 criterion leaves G-F and becomes S1 (D3).
- **G-N (new: numerical refinement)**, with two criteria.
  - `|η_2(dev) − η_2(final)| / ΔW ≤ 0.001`, using the G-A η₂ at the end of A.
  - `|Ĝmax_full(dev) − Ĝmax_full(final)| / ΔW ≤ 0.001`, using the G-F Ĝmax at the end of B.
- **A run passes if and only if it passes G-A, G-F and G-N.**
  - As before, a G-A failure is recorded as a stage-2 failure.
  - That run's Phase B still runs and is reported, and the run counts as failed.

### D3. Secondary criterion S1 (pre-registered; not part of run pass)

- The criterion: `S1: |ê_1(0) − e1*(0)| / e1*(0) ≤ 0.10`.
- **Report per run:** the value, and pass or fail.
- **Report per q:**
  - the S1 pass count out of 20, with an exact (Clopper–Pearson) 95% CI;
  - the mean signed stage-1 relative error, with a 95% percentile bootstrap CI (10,000 resamples; the bootstrap seed is fixed in the v1.1 protocol);
  - the median and the SD.
- S1 has no pass rule.

### D4. Pass rule, seeds, and the v1.0 outcome

- **Pass rule (unchanged).** For each q, at least **18 of 20** runs must pass (D2).
  - The confirmation passes if and only if both q do.
  - Report each q's pass rate with an exact 95% CI.
- **Seeds.** The confirmation block **20501–20520 is confirmed**. It is used for both q (40 runs).
- **v1.0 outcome.** For every run, also report the v1.0 outcome: G-A plus the v1.0 G-F, which includes the stage-1 criterion. It is reported only and decides nothing.

### D5. Hardening of the process-global RNGs

- **Seeding.**
  - Where: at the start of the pipeline function (the one the in-process smoke test also calls), before any object is constructed.
  - What: seed all three with the run seed (`torch.manual_seed`, `np.random.seed`, `random.seed`).
  - Record the seeds in the manifest.
- **Reference state.** Each RNG's state right before the first Phase A update.
  - Also record the post-seed state.
  - For each RNG, report whether anything between seeding and the first update drew from it, and where.
  - For example, module construction can draw from the torch global RNG through PyTorch's default initializer, even when the weights are then set from the torch generator.
- **Assertion.** Each of the three states must equal its reference state at four points:
  - the end of Phase A;
  - after G-A;
  - the end of Phase B;
  - the end of the run.

  On a violation:
  - record the RNG and the point in `gates.json`;
  - let the run finish and write all outputs, then exit with a nonzero code;
  - count the run as failed.
- **Digests.** In the manifest, record a SHA-256 digest of each RNG state at seeding, at the reference point, and at each assertion point.
- **No change to training.** Training-relevant state must not change; §2 R1 checks this.

### D6. Automatic continuation

- If every check in §2 passes, launch §3 immediately.
- If any check fails, stop and report. Do not launch.

### Unchanged from v1.0

- The training pipeline (D1 of the lock round).
- The verifier and its tiers.
- All metric definitions, the on-path rule, and the ẽ₁ band method.
- The reported-not-gated list.
- The evaluated candidate: the last iterate.

---

## 1. Protocol v1.1 and its lock

### 1.1 Protocol files

- **Add** two files: `protocols/v2_T2_locked_v1_1.json` and `protocols/v2_T2_locked_v1_1.md`.
- **Generate** the v1.1 JSON from the v1.0 JSON by applying only D1–D5.
  - Produce a machine diff from v1.0 to v1.1.
  - Confirm that nothing else changed.
- **Leave the v1.0 files byte-identical.** v1.0 stays reproducible from its lock commit `4bd2214`.
- **The `.md` contains:**
  - the v1.1 gates, S1, the pass rule, the confirmed seed block, and the bootstrap settings;
  - the definition of training-relevant state;
  - the hardening rule;
  - the change log (§1.2).

### 1.2 Change log, v1.0 → v1.1

For each change, give what changed, why, and the evidence with paths. State that the PI decided every change **after** the development-seed rehearsal and **before** any confirmation seed was run.

**1. The stage-1 criterion moves from G-F to S1.** The evidence comes from the v1.0 rehearsal (`reports/v2/protocol_lock_and_rehearsal.md` §4.3–§4.5).
- **Failures.** All 3 failures were on the stage-1 criterion (0.103, 0.126, 0.113).
  - Their Ĝmax_full/ΔW was 0.0021–0.0023, at (t = 1, d = 0), well under 0.01.
  - G-A passed 20/20.
- **Spread.** The signed stage-1 error had SD 0.052 (q = 50) and 0.070 (q = 60). The 90th percentile of |error| was 0.089 and 0.114.
- **Origin of the threshold.** The 0.10 threshold was set from the Pilot 4 distributions, whose Phase B parents had a constant-LR Phase A. The locked pipeline's distribution has a heavier upper tail.
- **PI-side estimate** of the v1.0 confirmation pass probability (both q reaching ≥ 18/20):
  - about 0.14, taking the rehearsal pass rates (9/10, 8/10) as the true rates;
  - about 0.35 under a normal model of the signed stage-1 error fitted to the rehearsal (mean −0.014 / −0.002, SD 0.052 / 0.070). This model treats G-A and the Ĝmax criterion as always passing, as they did in 20/20 rehearsal runs.

  Recompute both with a script and cite its path. Report any disagreement.

**2. G-N is added.** In the rehearsal, the dev − final differences of η₂ and Ĝmax were at most 2.2e−4·ΔW (§4.5).

**3. Hardening.** This is motivated by Check 1 (§4.1).

### 1.3 Entry point

`run/run_v2_T2_locked.py`:
- reads only the v1.1 JSON, and embeds its hash;
- keeps every v1.0 refusal;
- writes the D2–D5 fields into `gates.json` and the manifest, with protocol version 1.1.

### 1.4 Pre-registered analysis script

Write `tools/v2/confirmation_analysis.py` now. It:
- takes a results root and expects every (q, seed) of the block, marking missing or crashed runs as described in §3;
- recomputes every verdict from the metric values, and checks each against that run's `gates.json`;
- produces every table in §4.

It is committed with the lock and used **unchanged** on the re-rehearsal (§2) and on the confirmation (§3).

### 1.5 Tests (`tests/test_v2_locked.py`)

- **Hash and refusals for v1.1:** a modified protocol, extra overrides, and q ∉ {50, 60}.
- **LR schedule:** checked at every update of A and B, as before.
- **Gate logic**, on synthetic values, including values exactly at each threshold:
  - run pass = G-A ∧ G-F ∧ G-N;
  - S1 and the v1.0 outcome do not affect it.
- **Global RNGs:**
  - after seeding, each state equals the state from a fresh seeding with the run seed;
  - the reduced-budget in-process pipeline records no violation;
  - a single injected draw from each of the three RNGs, one at a time, is detected and recorded in `gates.json`, and makes the entry point exit with a nonzero code.

  If the reduced-budget run shows that anything after the reference point draws from a global RNG, stop before the lock commit and report the call site.
- **Full suite:** it passes, apart from the known `test_registry_canonicalization` failure.
- **C7:** bit-exact.

### 1.6 Lock

1. The commit that holds all of the above is the **v1.1 lock commit**.
2. In a follow-up commit, append a v1.1 record to `protocols/LOCK`. It contains:
   - the lock commit hash and the protocol hash;
   - the SHA-256 of the analysis script;
   - the version, the date, and the reason.

   Keep the v1.0 record verbatim.
3. From then on, nothing changes in `protocols/`, the entry point, the pipeline code, or the analysis script.

---

## 2. Re-rehearsal under v1.1 (development seeds)

### Runs

- **Design:** q ∈ {50, 60} × seeds 10501–10510, 20 runs, through the v1.1 entry point, from scratch.
- **Output:** `results/v2_T2_locked/rehearsal_v1_1/q*/seed*/`. Leave the v1.0 rehearsal directory untouched.
- **Launch commit:** its diff from the v1.1 lock commit may touch only `protocols/LOCK` and `results/`.
- **Settings:** single-threaded (OMP / MKL / OpenBLAS = 1). Choose parallelism from `nproc` and the current load, and report both.

### Automatic checks

Write each verdict to `results/v2_T2_locked/rehearsal_v1_1_checks.json`.

- **R1. Bit-identity with the v1.0 rehearsal.** All training-relevant state (D1) is equal for every (q, seed), in 20/20 runs. Gate-metric values are compared wherever both versions record them.
- **R2. Primary criterion.** All 20 runs pass G-A, G-F and G-N.
- **R3. Global RNGs.** No violation is recorded, and the return code is 0, in 20/20 runs.
- **R4. Manifests.** All 20 show the v1.1 protocol hash and version, the launch commit, and `clean_tree: true`.
- **R5. Analysis script.** It runs on the rehearsal root, and its recomputed verdicts agree with `gates.json` in 20/20 runs.
- **R6. Tests and C7 at the launch commit.** The full suite passes as described in §1.5, and C7 is bit-exact.

**If all six pass,** commit the rehearsal records and the checks file, then go to §3.

**If any check fails,** stop and report. Do not launch §3.

---

## 3. Confirmation (only if §2 passes)

- **Runs.**
  - q ∈ {50, 60} × seeds 20501–20520: 40 runs.
  - Each runs through the v1.1 entry point, from scratch, with the same settings as §2.
  - Output: `results/v2_T2_locked/confirmation/q*/seed*/`.
- **Launch commit.**
  - The same rule as in §2 applies. Record `git diff --stat` against the v1.1 lock commit.
  - Every manifest must show the v1.1 hash and `clean_tree: true`.
- **Before launch.** Check `nproc`, the load, and free disk space, and report all three. Do not kill jobs you did not start.
- **Crashes.**
  - **Infrastructure kill** (out of memory, disk full, process killed):
    - move the run's directory to `confirmation/crashed/q{q}/seed{s}_attempt1/` (do not delete it);
    - re-run once into the original path;
    - if the re-run also fails for an infrastructure reason, stop and report, and do not count that run either way.
  - **Any other nonzero exit** (an exception in the pipeline, or a global-RNG violation): the run counts as failed. Report it with its traceback.
- **After launch, nothing changes:** not the code, the protocol, the thresholds, or the analysis script.
  - No extra seeds, and no re-runs beyond the crash rule.
  - If the analysis script turns out to have a bug, do not edit it in place. Fix it in a new commit, run both versions, and report both outputs with the reason.
- **Analysis.** When all runs have finished, run `tools/v2/confirmation_analysis.py` unchanged on the confirmation root.

---

## 4. Report: `reports/v2/protocol_v1_1_confirmation.md`

Cite a source path for every number. State the verdict exactly as computed.

1. **The v1.1 lock:**
   - commits, files, and hashes;
   - the v1.0 → v1.1 JSON diff and the change log;
   - the test and C7 results;
   - a timeline with commit hashes and timestamps: lock commit, LOCK record, rehearsal launch, checks commit, confirmation launch.
2. **Hardening:**
   - where each global RNG's reference state was taken;
   - any draw before the first update, with its call site;
   - the test results.
3. **Re-rehearsal:**
   - the R1–R6 verdicts;
   - a per-run table with the v1.1 criteria, S1, and the v1.0 outcome.
4. **Confirmation:**
   - **Verdict first.** Per q: the primary passes out of 20 against the ≥ 18/20 rule, with an exact 95% CI. Then the overall verdict.
   - **Per-run table** for all 40 runs, including failures and crashes:
     - every G-A, G-F and G-N value (final and dev) with pass or fail, and the run outcome;
     - S1 and the v1.0 outcome;
     - the return code, the global-RNG status, and the wall time.
   - **S1 per q:**
     - the pass count with its exact CI;
     - the mean signed stage-1 error with its bootstrap CI;
     - the median and the SD.
   - **Distributions** (min, p10, p25, median, p75, p90, max) of every gate metric, plus the dev − final differences.
   - **Every reported metric** from §4.4 of the v1.0 report, including the stage-1 decomposition with bands.
   - **Stage-1 error vs root gain** (descriptive only). Per q:
     - a table of `EXP_root/ΔW` against the stage-1 relative error;
     - a least-squares fit of `EXP_root/ΔW` on the squared relative error, with an intercept.
5. **Rehearsal vs confirmation:** side-by-side distributions, descriptive only. The rehearsal seeds were used for development.
6. **Anomalies and deviations.**
7. **Commands to reproduce.**

Update `reports/v2/summary.md`, and commit the lightweight records. Do not push.

**STOP.** Do not start any T = 3 work. Change nothing after the confirmation.
