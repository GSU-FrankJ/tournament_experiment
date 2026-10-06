> **Provenance of this file (read first).** This is **not** the PI's delivered file. The PI inbox (`.claude/pi_inbox/`) did not exist when this folder was built, and no copy of prompt 15 exists in any worktree or results folder. What follows is the assistant's transcription of the prompt text as it was pasted into the R2c session on 2026-10-05, which the assistant saved as scratch for conformance checks during that round. Line wrapping and emphasis markers may differ from the delivered file; the section structure and decisions (D1-D7) are as pasted. The line `(Verbatim copy of the owner's prompt ...)` of the scratch copy was replaced by this banner. If the PI supplies the delivered file, it should replace this one (the `SHA256SUMS` of `pi_record/` then changes).

# R2b accepted: sampler pilot R2c, then protocol v2.1 (peak-focused starts) — lock, re-rehearsal and confirmation under a pre-registered selection rule

The R2b round and its audit are accepted (`reports/v2/refine_r2b/`). Its result that matters here: `A_peak50` is the only mechanism that moved the stage-2 peak (paired difference of the absolute peak error −0.023 [−0.041, −0.004] at q = 50 and −0.012 [−0.023, −0.001] at q = 60; runs within the 0.05 target 2 → 8 and 5 → 8), and it failed criterion part (b) only because 3 of 10 q = 60 runs crossed the G-A tail-mean limit (0.020–0.022 against 0.009–0.010 under the baseline) when the tail bins' sample share fell from about 45% to 25%. The pathwise arms with a matched optimiser budget did not beat PPO at the same LR, and `A_censored` had no effect. Read: the remaining peak gap is a weighting problem (payoff flatness at the cusp, 10% of the samples), not a noise, smoothing or optimiser problem; the fix is the start distribution, and the open question is the share and the timing that keep the tail inside its gate.

This round answers that question on the development seeds and, **if a pre-registered selection rule picks an arm, continues without waiting for me** to protocol v2.1, its re-rehearsal and its fresh-seed confirmation on 40501–40520. If no arm is selected, it stops after the pilot report.

Steps:
1. **P0** housekeeping and the pushes authorised in §1.
2. **P1** the scheduled sampler, tests, C-R3, pre-registration (§2).
3. **P2** the pilot wave S and its pre-registered analysis and selection (§3).
4. **P3** only if an arm is selected: protocol v2.1, lock, re-rehearsal, confirmation (§4).
5. **P4** reports (§5), then **STOP** (§6).

Repository `/home/fjiang4/tournament_experiment`; Python `/home/fjiang4/tournament_experiment/.venv/bin/python` only; `pytest tests` (never a bare `pytest`). P0 runs where stated; P1–P4 run on a new branch `v2-t2-r2c` taken from the head of `v2-t2-r2b`, in a new worktree `.claude/worktrees/v2-t2-r2c`. Results root `results/v2_refine_r2c/` for the pilot and `results/v2_T2_locked/` for the lock round; reports `reports/v2/refine_r2c/` and `reports/v2/protocol_v2_1_confirmation.md`.

Everything from the earlier rounds still applies: provenance discipline; the closed-form equilibrium enters evaluation only; no deletion or overwriting of results or checkpoints; the paper-registry test and its data unchanged; single-threaded processes in tmux; do not kill jobs you did not start. Pushing is authorised only where §1 and §4.6 say so.

**If any step fails, or anything does not match this prompt, stop at that point and report. Do not work around it.**

---

## 0. Decisions (binding)

### D1. R2b is closed

- Method 5 (pathwise fine-tuning) is closed as a negative result at matched budgets; `clamp_likelihood` stays `density` (no measurable effect; the likelihood inconsistency remains a recorded known issue); the seed-30510 diagnostic stays descriptive. These three statements go into the v2.1 change log / known issues (§4.2).
- The R2b open items: the report-pack refresh of the R2b prompt §1.2 is **waived** (record the waiver and the builder's failure diagnosis from `00_housekeeping.md` Addendum 2 as a known issue; no repair); the over-long commit subjects stay (they are cited by hash); the R2b worktree repair is accepted.

### D2. Pilot wave S: four start-distribution arms, one factor each, PPO-internal

`start_weights` gains `local_first` (default 1): for local updates before `local_first` the sampler is the locked bin-balanced one, bit-identical in draws and stream consumption; from `local_first` on it is `peak_focused` as implemented in R2b (half-width 20, share s, uniform remainder). Full Phase A from scratch (mode `phase_A`, 1600 updates, locked Phase-A settings, `fixed_budget=true`, `full_state_at: [1200]`), baseline `results/v2_refine/parents_A` (paired by (q, seed)):

| arm | `start_weights` |
|---|---|
| `A_peak35` | peak_focused, share 0.35, local_first 1 |
| `A_peak40` | peak_focused, share 0.40, local_first 1 |
| `A_peak50_late400` | peak_focused, share 0.50, local_first 1201 |
| `A_peak50_late800` | peak_focused, share 0.50, local_first 801 |

4 arms × 20 (q, seed) = 80 runs. Expected by construction and to be verified: the first 1200 updates of `A_peak50_late400` are bit-identical to `parents_A` (compare `state_u01200.pt` and the weight exports up to u1200), and the first 800 updates of `A_peak50_late800` are (weight exports up to u0800).

### D3. Pre-registered criterion and selection rule

- **Criterion per arm** (as in R1/R2b): (a) the 95% percentile bootstrap CI of the mean paired difference of the absolute signed peak error against `parents_A` excludes 0 in the improving direction at **both** q; (b) no run that passed G-A and its G-N part under the baseline fails it under the arm. Bootstrap: 10,000 resamples, `numpy.random.default_rng(20261005)`, one fresh generator per (q, statistic) in table order.
- **Selection rule.** Among the arms meeting (a) and (b): the largest total number of runs with |peak error| ≤ 0.05 over both q; tie → the smaller mean |peak error| over both q; further tie → the smaller change from the locked sampler (later `local_first`, then smaller share). The selected arm's `start_weights` is the one pipeline change of v2.1.
- **No arm meets (a) and (b):** write the pilot reports and stop (§6). Do not relax the rule, do not add arms.

### D4. Protocol v2.1 = v2.0 plus the selected `start_weights` in Phase A, nothing else

Unchanged from v2.0: Phase B in full (expected continuation, the locked table rule), budgets 1600 / 600, LR schedules, G-A / G-F / G-N / G-S and their thresholds, S1 at 0.10 and the earlier outcomes reported, the verifier and tiers, every metric definition, the evaluated candidate, the global-RNG hardening, the training-relevant-state definition, the bootstrap settings (seed 20261002 for the confirmation statistics). v2.0 stays reproducible from `1d6d4d0`; the v2.0 protocol files stay byte-identical.

### D5. New gate G-P (stage-2 peak recovery), admitted by a pre-registered rule

- **G-P:** |ê₂(0) − e₂*(0)| / e₂*(0) ≤ 0.10 on the end-of-A last iterate (the stage-2 analogue of G-S; 0.10, not 0.05, because the selected arm's development distribution will not support 0.05 at ≥ 18/20).
- **Rule (evaluated on the v2.1 re-rehearsal, §4.4 R3b):** G-P enters run pass iff (i) ≥ 19 of the 20 rehearsal runs satisfy it and (ii) under a normal model fitted per q to the rehearsal's signed peak errors (mean, SD with ddof = 1; the other gates treated as passing) the probability that ≥ 18 of 20 fresh runs satisfy G-P is ≥ 0.90 at both q. The outcome is recorded in `results/v2_T2_locked/v2_1/gp_rule.json` **before** the confirmation launch, and the v2.1 analysis script reads it. If the rule fails, G-P is reported, not gated, and run pass stays G-A ∧ G-F ∧ G-N ∧ G-S.
- Reported for every run either way: G-P, the peak error with its decomposition, and the v2.0 / v1.1 / v1.0 outcomes.

### D6. Pass rule, seeds, crashes, continuation

- For each q at least **18 of 20** runs pass; both q must pass; exact Clopper–Pearson CIs. Seed block **40501–40520** for both q (40 runs); re-run `tools/v2/seed_inventory.py` at the lock commit (with the protocol's own declaration excluded, as in v2.0) and record 0 collisions. Crash rule as in v2.0 (`confirmation_v2_1/crashed/…`).
- Automatic continuation: pilot → (selection) → lock → rehearsal → (all checks pass) → confirmation. Any failed check stops the round.

### D7. Reference roots (untracked files live where they were produced)
`rehearsal_v1_1` and the v1.1 confirmation: `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/`; `rehearsal_v2_0`, the v2.0 confirmation, `parents_A` and the R1 runs: `.claude/worktrees/v2-t2-refine/results/`; the R2b runs: `.claude/worktrees/v2-t2-r2b/results/`. Every comparison tool takes its reference root as an explicit argument and records it; a missing reference file is a stop-and-report.

---

## 1. P0 — housekeeping and pushes

Write `reports/v2/refine_r2c/00_housekeeping.md` as you go (commands, hashes, verbatim outputs).

1. **Push `v2-t2-r2b`** (fast-forward, no force) so the R2b record is on GitHub; record `git ls-remote`.
2. **Fast-forward `main`**, in the primary checkout `/home/fjiang4/tournament_experiment`: record `git status --porcelain` **verbatim** in the housekeeping record. If it shows any tracked modification, skip this item and report (do not stash, reset or merge). Otherwise `git fetch origin`; `git merge --ff-only origin/main` (f02a256); `git merge --ff-only t2-v2-main-v2.0` (b55d389); verify `git rev-parse main` = `b55d389`; `git push origin main` (fast-forward only). Record `git ls-remote origin main`.
3. Record the D1 waivers and acceptances.
4. Create the branch and worktree: `git worktree add .claude/worktrees/v2-t2-r2c -b v2-t2-r2c v2-t2-r2b`.

---

## 2. P1 — code, tests, C-R3, pre-registration (on `v2-t2-r2c`)

### 2.1 Code
`local_first` for `start_weights` (D2) in the sampler and the runner; written to `run_config.json` and `manifest.json`; the launcher gains wave S. Nothing else changes in the pipeline code.

### 2.2 Tests (`tests/test_v2_r2c.py`, plus the existing suites)
- Defaults: with `local_first` absent or 1 and `scheme = bin_balanced`, the smoke pipeline is bit-identical to the C7 reference and to the v2.0 reduced-budget in-process run.
- Window: with `local_first = L`, the start draws and the stream positions for local < L equal the bin-balanced sampler's exactly; from L on the peak share is as specified (10⁶-start empirical check, 3 binomial SE).
- Launcher: the four wave-S configs differ from the `parents_A` config only in `start_weights`.
- Full suite (`pytest tests`) passes apart from the known registry failure; C7 bit-exact.

### 2.3 C-R3 — the v2.0 entry point still reproduces the v2.0 rehearsal
Run the unchanged `run/run_v2_T2_locked.py` from the new branch for q ∈ {50, 60} × seeds 10501–10510 into `results/v2_refine_r2c/v20_reproduction/` and compare the training-relevant state against `rehearsal_v2_0` (reference root per D7) with the R2b comparison tool: 20/20 identical, or stop and report.

### 2.4 Pre-registration and the code commit
`reports/v2/refine_r2c/01_preregistration.md`: the arms (D2), the criterion and selection rule (D3), the v2.1 definition (D4), the G-P rule (D5), the pass rule and seed block (D6), the design basis quoted from R2b with paths (the `A_peak50` paired differences, the per-run tail means of its three violating runs, the pathwise and censored outcomes), and the bootstrap seeds. Commit code, tests and the pre-registration (**code commit**); later launch commits touch only `results/`, `protocols/LOCK` (at the lock) and the report folders.

---

## 3. P2 — pilot wave S and the selection

- Launch the 80 runs through the launcher (≤ 40 single-threaded workers in tmux; `nproc`, load and disk recorded; the crash rule; nothing changed after launch), output `results/v2_refine_r2c/waveS/q*/seed*/<arm>/`, one launch record. Post-launch checks: every manifest at the code commit with `dirty: false`; the prefix identities of D2 (`state_u01200.pt` of `A_peak50_late400` against `parents_A`, and the u0800 weight export of `A_peak50_late800` against `parents_A`'s) in 20/20 each.
- Analysis with the R2b tool extended for wave S: the metrics of the R2b stage-2 analysis (primary absolute signed peak error; location-free peak with its argmax d; RMSE_pos; tail mean and max with the G-A verdict; η₂ with the G-A and G-N verdicts; symmetry error; σ₂(0); smoothed-game share; the tail statistics: max |peak|, number of runs with |peak| ≤ 0.05, number with η₂/ΔW > 0.004; peak-set visitation share; KL, clip, wall time), paired against `parents_A` per q; the criterion parts (a) and (b); the selection rule applied mechanically, with its outcome written to `results/v2_refine_r2c/analysis/selection.json`.
- Reports `02_pilot_waveS.md` (checks first, then one section per arm with paired tables, criterion outcome, tail statistics, figures, anomalies, commands) and `03_selection.md` (the rule, the inputs, the outcome). If no arm is selected: `summary.md`, the paragraph in `reports/v2/summary.md`, commit, **STOP**.

---

## 4. P3 — protocol v2.1 (only if an arm was selected)

### 4.1 Files
`protocols/v2_T2_locked_v2_1.json` and `.md`, generated from the v2.0 JSON by `tools/v2/make_locked_protocol_v2_1.py` applying only D4–D6 (the selected `start_weights`, gate G-P with its rule and a placeholder for the rule outcome, the seed block, the known issues, the change log); machine diff v2.0 → v2.1; v2.0 files byte-identical.

### 4.2 Change log, v2.0 → v2.1
With evidence paths: the sampler change (R2b `A_peak50` and the wave-S selection, with the paired numbers); the closed items of D1 (method 5 negative at matched budgets: `reports/v2/refine_r2b/04_pilot_waveP.md`; censored likelihood no effect: `03_pilot_waveA.md`; the seed-30510 diagnostic: `02_seed30510_diagnostic.md`); the R2b/R2c pipeline-code changes with C-R2 / C-R3 20/20 as the evidence that v2.0 is unaffected; G-P and its admission rule; the seed block. State that the PI decided every change after R2b and before any confirmation seed was run.

### 4.3 Entry point, analysis script, tests, lock
- `run/run_v2_T2_locked.py` reads only the v2.1 JSON, embeds its hash, keeps every refusal, applies the sampler schedule in Phase A, writes G-P (value, pass, gated-or-reported per `gp_rule.json`), run pass (v2.1), and the v2.0 / v1.1 / v1.0 outcomes into `gates.json` with `protocol_version` 2.1.
- `tools/v2/confirmation_analysis_v2_1.py` (new file; the v2.0 script untouched): as the v2.0 script, plus G-P, the rule evaluation on a rehearsal root (writing `gp_rule.json`), and the §5 tables.
- `tests/test_v2_locked_v2_1.py`: hash and refusals (including a changed sampler parameter); gate logic with G-P gated and reported; the sampler schedule inside the entry point equals the pilot arm's draws; global RNGs; full suite; C7.
- Lock commit, then the LOCK record appended: lock commit, protocol SHA-256 (JSON, MD), entry point, analysis script, `utils/v2_continuation.py`, `envs/curriculum_env.py` (the sampler), version, date, reason, `supersedes` v2.0. From then on nothing in `protocols/`, the entry point, the pipeline code, the two modules or the analysis script changes.

### 4.4 Re-rehearsal under v2.1 (development seeds)
20 runs through the v2.1 entry point into `results/v2_T2_locked/rehearsal_v2_1/`; the launch commit's diff from the lock commit touches only `protocols/LOCK` and `results/`. Checks → `rehearsal_v2_1_checks.json`:
- **R1** the end-of-A training-relevant state equals the selected pilot arm's runs in 20/20 (Check-1 relation; reference root D7);
- **R2** Phase B: the end-of-B state is reproduced by a `phase_B` branch from the rehearsal's own end-of-A state in 2 of 2 spot checks (Check-2 relation); the stage-1 metrics are reported next to `rehearsal_v2_0`'s, paired by (q, seed), descriptive only;
- **R3** all 20 pass G-A, G-F, G-N, G-S; **R3b** the G-P rule (D5) evaluated and `gp_rule.json` written;
- **R4** global RNGs; **R5** manifests (v2.1 hash, launch commit, `clean_tree: true`, sampler configuration present); **R6** analysis script agrees with `gates.json` 20/20; **R7** `pytest tests` and C7 at the launch commit, C7 against the canonical reference by explicit path.
If all pass, commit the records and go to §4.5; otherwise stop and report.

### 4.5 Confirmation (seeds 40501–40520)
40 runs through the v2.1 entry point into `results/v2_T2_locked/confirmation_v2_1/`, same settings; launch-commit rule; `nproc`, load, disk recorded; the crash rule; nothing changes after launch; then `tools/v2/confirmation_analysis_v2_1.py` unchanged on the confirmation root.

### 4.6 Pushes after the confirmation
Push `v2-t2-r2c` (fast-forward, no force) once the report (§5) is committed, so that I can read it. Do not push `main` again and do not create tags; those follow my review.

---

## 5. P4 — reports

- Pilot: `reports/v2/refine_r2c/00_housekeeping.md`, `01_preregistration.md`, `02_pilot_waveS.md`, `03_selection.md`, `summary.md`.
- Lock round: `reports/v2/protocol_v2_1_confirmation.md` in the layout of the v2.0 report: the lock (commits, hashes, diff, change log, tests, C7, timeline); the re-rehearsal (R1–R7, R3b with the fitted model and probabilities, per-run table); the confirmation (**verdict first**; per-run table with every gate incl. G-P and the earlier outcomes; S1 / G-S and G-P summaries with exact and bootstrap CIs; distributions; reported metrics with the stage-1 decomposition; stage-1 error vs root gain); v2.0 vs v2.1 descriptive comparisons (same development seeds paired; fresh seeds 30501–30520 vs 40501–40520 unpaired); anomalies and deviations; commands.
- Append paragraphs to `reports/v2/summary.md`, rows to `reports/v2/README.md` (reading order, launch commits; a sentence for the tags, which I create), update `docs/STATE.md`. Commit the lightweight records with the usual tracking rules (the 4001-node tables are tracked; no `.pt`, no `train_history.json`, no weight exports).

---

## 6. STOP

Stop after the reports and the §4.6 push (or after the pilot report if no arm was selected). No T=3, no further arms, no changes after the confirmation. The next round is mine to decide.
