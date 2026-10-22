# T3-R3, P1-P3: checks

Dates: 2026-10-21 (P1, sections 1-3), 2026-10-22 (P2 and P3, sections 4-5). Code commits of P1: `3b774be8` (diagnosis tool, replay driver, tests), `6221341f` (replay summary and timing outputs), `475cac6f` (mean-clamp markers), `4e122a7d` (changes after the independent review, §2), `39f4e393` (the replay tool loads `tools/t3/compare_runs.py` by path: a full test session imports `tools/v2/compare_runs.py` first and the bare import resolved to it; found by the sharded full-suite run, §3). The diagnosis steps and the second set of replays ran at `4e122a7d`, which differs from `39f4e393` in that one import (no other test imports the replay tool). No existing file of `agents/`, `envs/`, `utils/`, `run/`, `protocols/` or `tools/` was changed (`git diff --name-status c3289df9 HEAD -- agents envs utils run protocols tools` lists four added files under `tools/t3/`, nothing else).

## 1. Tests

New: `tests/test_t3r3_diag.py` (18 tests, 3 s) and `tests/test_t3r3_replay.py` (5 tests, about 2 min, two of them train reduced-budget runs in temporary directories at seed 50500, which the prompt exempts).

| test | what it pins |
|---|---|
| S3 on a synthetic `ms_updates.csv` (3 tests), the T = 2 analogue, the onset window, the class boundaries | stage-3 rows and local updates 1201-1600 only; ratio of sums; phase A only; strict "above 0.2" |
| `forward_parts` | the manual forward equals `BetaActor.forward` exactly (`alpha`, `beta`) |
| constant network, saturated network, bump network, network at the mean clamp | F(u) = 0 and no sensitivity for a constant network; the second-layer saturation share is 1.0 and the sensitivity is 100 times smaller for the saturated one; F(u) > 50 for a bump; the share of nodes at the 1e-6 mean clamp and zero gradient through it |
| formation export and threshold | half the median of the passing runs' final F; `None` for a constant network |
| entry stretch | runs of consecutive in-band nodes below 1 |
| trigger flags and retrospective table | counts, fired and missed runs and the cost fraction on a hand-made table; the source of `trigger_flags` and `retro_table` names none of `rmse`, `tail_mean`, `peak`, `Gmax`, `eT_`, `g2_`, `theory` |
| replay summaries | the pooled share is weighted by the squared gradient norm; the window ratios; the cosine; the per-bin coverage |
| per-sample gradients, `update_diag` (hand computation of 6 samples: regions, clamp groups, bins, shares summing to 1, the actor untouched) | `vmap` gradients equal the one-backward-per-sample loop; the projection shares of any partition sum to 1 |
| passivity (reduced `bw` config, stage 3 at 50 updates) | the replay driver with the instrumentation equals the same driver without it bit for bit (exports, rows, checks, freeze arrays, stage-3 state incl. optimizer and streams), the global-RNG assertion holds, the rollout's clamp counters equal the counts of stored actions |
| replay equals the original | the driver on the config of the reduced legacy `bw` run (the session cache of `tests/t3_helpers.py`) reproduces its stage-3 files bit for bit |

Full suite: `results/t3_r3/tests/pytest_full.txt` (see §3).

## 2. Independent review

One read-only review by an agent that wrote none of the code or reports (brief: the prompt's §2.5; it re-derived numbers from the raw tracked CSVs, the stored weights and the untracked originals without importing `t3r3_diag.py`, and spot-checked four replays against their originals). **No blocker.** Fifteen findings, all dispositioned:

| # | severity | finding | disposition |
|---|---|---|---|
| 1 | should-fix | the retry's "generator calls of the original construction" would move the process-global torch RNG (`nn.Linear.__init__`) and a new optimiser would trip the `optimizer object replaced` guard | `02_proposal.md` §2.1 now specifies an in-place re-initialisation (`nn.init.orthogonal_(..., generator=g2)` on the existing modules, Adam states cleared in place) and a P2 test that the global RNG states are unchanged across the retry |
| 2 | should-fix | attempt 1's `state_end_stage3.pt` is written by `freeze_stage` only; the in-memory histories and counters of attempt 1 were not in the specification | §2.1 now lists the explicit dumps (`state_attempt1_end.pt`, `ms_updates_stage3.csv`, `train_history_stage3.json`) and the counters that are reset; attempt 1 is not frozen |
| 3 | minor | the trajectory table labelled 60501 (q = 60) "formed at u200"; its formation export is 250 | the label is now generated from `dg2_runs.csv` |
| 4 | minor | "every run collapses to about 8 with sensitivity 1e-4 to 1e-3" over-generalised | replaced by the distributions (minimum e(0) 6.9-12.8; minimum sensitivity 5.5e-5 to 6.3e-2, median 5.3e-3) |
| 5 | minor | "5 bins around d = 0" selects 6 bins | header changed to "6 bins with |d| <= 60" |
| 6 | minor | the reference update for runs without an onset used the median over all runs with any onset, the text says the regime runs | code aligned with the text (`4e122a7d`); q = 50 "others" median 2.66, p = 0.20 (was 2.07, 0.10); the conclusion is unchanged |
| 7 | minor | "3.3-6.6" should read 3.3-6.1 for the formed failures | fixed |
| 8 | minor | "persistent in-band bias 14-16 of 16" holds for 4 of the 8 K runs | the class table of `02_proposal.md` now says which |
| 9 | minor | `valid` is not purely a stage-3 quantity (it includes stages 1 and 2 of the composite candidate) | the trigger now reads `Delta`, `s` and `C` only (`degenerate`: not finite, or `s` <= 0); `trigger_flags`, the test and the retrospective table were updated; no effect on the recorded data (`valid` is true at all 5,120 stage-3 checks) |
| 10 | minor | the replay did not store the SD of the advantages per bin and the raw statistics per clamp group that DG5 lists; `s` and the start counts were not in the report | `update_diag` now stores them; the 14 replays were repeated at `4e122a7d` (14 of 14 bit-identical again); the DG3 tables of `01_diagnosis.md` §4 now show `s` and the starts per stratum |
| 11 | minor | the "64 exports" column was hard-coded; `retro()` was untested; the passivity test's comment said "optimizer" but compared none | the column comes from the data; `test_retro_reads_the_local_1600_row_of_the_check_files` added; the passivity test now compares both Adam states |
| 12 | note | "a mechanism through the mean gradient is not what the data show" is stronger than the share supports (Adam, global-norm clip) | one sentence added in both reports: the share rules out domination of the gradient, not an effect on parameters where the in-band gradient is small |
| 13 | note | "inside the passing range up to update 100" omits entropy, concentration, mean effort, `Delta`, `R` | wording corrected in §4 and §12 of the diagnosis |
| 14 | note | the Spearman -0.92 uses the sensitivity at u150, partly after formation | the values at u125 (-0.88) and u100 (-0.73), which precede every formation, are quoted |
| 15 | note | commit hygiene: `replay_bins.npz` and `diag/arrays` must not be tracked | the P1 commit uses an explicit file list (csv / json of `diag/` and `replays/`; no npz, no weights) |

Not re-checked by the reviewer: the five references (verified by web search by the author: Fujita & Maeda ICML 2018, PMLR 80:1597-1606; Chou et al. ICML 2017, PMLR 70:834-843; Gomes et al. J. Automated Reasoning 24:67-100, 2000; Henderson et al. AAAI 2018; Luby, Sinclair & Zuckerman 1993, whose journal volume and pages were not confirmed, the proposal says so) and the 1.17-fold cost of the censored update (one measurement on a synthetic batch by the author).

What the reviewer reproduced independently: S3 and every count of §7 of the diagnosis, the T = 2 analogue, DG2 for all 5,120 exports (differences at most 1e-5, float32), the 80 formation exports, DG8, the retrospective table, the operating-characteristic and projection tables, the budgets, the 32 generated tables from scratch, update 1 of two replays from the config (about 1e-8 relative), and bit-identity of four replays with their originals (all 64 exports, the freeze arrays, 1,600 rows, the check file) plus, over all 14, the bin maps, the continuation tables and the whole stage-3 state.

## 3. Full suite

`pytest tests` (never a bare `pytest`) as 12 parallel shards by test file (`results/t3_r3/tests/run_shards.py`, tmux sessions `t3r3_sh0` ... `t3r3_sh11`, one thread each, the files of each shard in `shard{n}.files`, output in `shard{n}.txt`), started at `4e122a7d`:

| shard | result | | shard | result |
|---|---|---|---|---|
| 0 | 129 passed (3:58) | | 6 | 92 passed (0:54) |
| 1 | 80 passed (3:04) | | 7 | 157 passed (8:57) |
| 2 | 170 passed, 2 xfailed (12:20) | | 8 | 153 passed, 1 skipped (4:54) |
| 3 | 110 passed, 3 skipped (2:37) | | 9 | 164 passed (2:21) |
| 4 | 213 passed (6:31) | | 10 | 168 passed, **1 failed** (4:13) |
| 5 | first run: 3 failed, 129 passed, 2 skipped (17:23); **re-run at `39f4e393`: 131 passed, 2 skipped, 1 failed (16:10)** | | 11 | 179 passed (6:26) |

**Total with the shard-5 re-run: 1,746 passed, 6 skipped, 2 xfailed, 2 failed.** The two failures are not caused by this round:

- `tests/test_registry_canonicalization.py::test_registry_canonicalization` (shard 5): the known failure ("expected 15 Set-2 gradient runs, got 0"; recorded in `results/t3_r1/preflight/tests/two_non_t3_failures.txt` as failing also on `origin/ms-r4`).
- `tests/test_ms_r4_launcher.py::test_the_existing_launcher_is_unchanged` (shard 10): it asserts that `git diff --stat be2881ff -- ... protocols` is empty; the diff it prints is `protocols/LOCK` and `protocols/v2_T3_locked_v1_0.{json,md}`, the T3-R2 lock files. `git diff --stat be2881ff c3289df9 -- protocols` (the commit this round started from) prints the same three files, and `git diff --name-status c3289df9 HEAD -- protocols` is empty, so the test failed before this round and T3-R2's record of "1 failed" predates its own lock commit. It is not edited here (no existing test changes in this round).

The first run of shard 5 had two further failures, both of the new replay tests (`test_instrumentation_is_passive_and_bit_identical`, `test_replay_reproduces_the_original_stage3_bit_for_bit`): `import compare_runs` resolved to `tools/v2/compare_runs.py` because another test of the session had imported it first (`module 'compare_runs' has no attribute 'same_array'`). Fixed in `39f4e393` by loading the T3 file by path; the two tests pass alone (112 s for both new test files together) and in the shard re-run. The earlier partial full run (`pytest_full_partial_475cac6f.txt`, stopped at 24 % because the working tree had changed under it) is kept for the record.

Counts against T3-R2's full suite (1,724 passed, 1 failed, 6 skipped, 2 xfailed): 1,724 + 23 new tests - 1 (the launcher test, which fails since the T3-R2 lock) = 1,746 passed.

## 4. P2: the scheme, its tests, the independent review, C-LEGACY

**Code.** `run/ms_retry.py` (new), additive key-gated changes in `run/run_ms_stagewise.py` (`stage3_retry`, `terminal_clamp_likelihood`, declared in `OPTIONAL` and validated; a bookkeeping attribute `last_check`; the manifest records the keys when present), `tools/t3/t3r3_{launch,checks,pair,eval,analysis,blind,legacy}.py`. Commits: `26ada045` (implementation and tools), `727116a3` (the review's fixes and the C-LEGACY driver; **the final code commit**: C-LEGACY, the plan and the launch use it). `git diff --name-status c3289df9 HEAD -- agents envs utils run protocols tools` lists only new files and `run/run_ms_stagewise.py`; the ten tool files of the T3-1.0 lock have unchanged SHA-256 (the launcher's lock check passes). Candidate B needed nothing beyond wiring `utils/beta_tail.py`, `collect_batch_v2(..., clamp_likelihood="censored")` and `CurriculumPPOv2.update(..., clamp_side=...)` into the runner behind the key (G1 reply 1: no stop).

**New tests (65):** `tests/test_t3r3_retry.py` (30), `test_t3r3_launch.py` (20), `test_t3r3_analysis.py` (8), `test_t3r3_blind.py` (7). They pin: the keys' validation (refused with `actor_variant`, `conc_scale_schedule`); the trigger on 12 conditions (strict `>`, non-finite `Delta` / `s` / `C`, `s` <= 0, no reading of `valid`, `R`, `R_tail`, none of the closed-form names in its source); the in-place re-initialisation equals the constructor; the derived seed and the five streams with hard-coded names; the restart on reduced runs (the process-global generators and the optimiser identity untouched, nothing overwritten, a second restart refused, counters, costs and logs reset); **attempt 2 equals a fresh phase built by hand** from the original constructors and the derived seeds (rows, networks, optimiser state, streams); a fired reduced run: attempt 1 equals the `bw` run's stage 3 bit for bit, the frozen path has the `bw` layout and files, totals 140 updates; a run that does not fire and the censored key without a clipped draw equal the `bw` run (with run names that differ); **the required B test** (the stored log-prob of every flagged row equals the censored log-mass of `utils/beta_tail.py`, every unflagged row the density path; more than 100 flagged and 10 unflagged rows) and its control (the density path without the key, `update` called without `clamp_side`); negative controls of the pair comparator (one export, one cell of the stage-3 rows, one cell of the check file, the bin maps, a missing `attempt1/` file, a stray file, an `attempt1/` without `retry.json`); the checks on reduced runs in the launcher's layout (config hash, init digest, commit, trigger consistency, pair, replay); the launcher (122 runs, the seed guards for 11 seeds, config differences, the lock check, the plan digests, infrastructure failures); the analysis (counts, target, selection order, McNemar, the first-attempt convention and its stop, the registered bootstrap, S3 of the first attempt, costs, fired table) and the blind recomputation (inclusive pass rule, per-q and total thresholds, missing runs, ledger).

**Independent review of the P2 code** (one read-only agent that wrote none of it; it read the diff line by line, ran the new tests, 14 mutation checks on copies, a launcher dry run and an independent reference for attempt 2): **no finding in `run/ms_retry.py` or `run/run_ms_stagewise.py`**; the legacy default path is the old one (a reduced `bw` run at `c3289df9` and at HEAD are identical, same file list). Fourteen findings, all in the post-launch tools and the tests, dispositioned in `727116a3`:

| # | severity | finding | disposition |
|---|---|---|---|
| 1 | blocker | the check files carry the run name, which differs between arms and rounds: C-PAIR would fail on every candidate run | the pair comparator drops `run` (and `verifier_sec`) from the check CSVs; the fixtures now use distinct run names |
| 2 | blocker | `config_equals_plan` took the plan root from `parents[2]` (the T3-R2 layout): false for all 122 runs; the test had the same error | the root of the launch is used; the test uses the launcher's layout `<root>/dev/q50/seed50500/<arm>` |
| 3 | should-fix | the tests did not pin the attempt-2 reset (mutations of the minibatch and numpy streams, `costs`, `weights_log` survived) | the hand-built reference test; `costs`, `weights_log`, `phase_timing` asserted empty after the restart |
| 4 | should-fix | the stream-name test iterated the code's own mapping | hard-coded mapping |
| 5 | should-fix | the pair comparator had no negative control | the negative-control test above |
| 6 | should-fix | analysis I/O untested (S3 of the first attempt) | `first_attempt_rows` factored out and tested for a fired and a non-fired run |
| 7 | minor | the bootstrap-seed test was insensitive (3 distinct values) | continuous synthetic differences; a different seed gives another interval |
| 8 | minor | an `or` escape in a Clopper-Pearson assertion | exact bounds (0.8308, 0.9937) |
| 9 | minor | the blind total threshold was untested alone | 37 of 40 (19 / 18) fails, 38 of 40 passes |
| 10 | minor | an exception in the checks of one run would abort the pool | `check_run` records `error` and leaves the verdicts it did not reach unevaluated (fails `checks_ok`) |
| 11 | note | the launcher test needed the thread environment; `--plan-csv` too | the launcher and the test set it with `setdefault` |
| 12 | note | `stage3_retry` with `conc_scale_schedule` / a non-`t1` actor / ... is not reset by the restart | the validator refuses the combination (tested) |
| 13 | note | the first-attempt sentence of the G1 reply says "equals the pass count of `bw`" | pre-registered as the per-seed invariant under the same convention (`04_preregistration.md` section 7); the raw `bw` count and the number of `bw` runs that fire and pass are reported beside it |
| 14 | note | `fired` is read from `retry.json` (written at the end); the diagnostic C-INIT compared with the plan only | `attempt1/` without `retry.json` fails `files_complete`; the diagnostic runs' digest is also compared with T3-R2's manifest |

**C-LEGACY at `727116a3`** (`results/t3_r3/legacy/`, `tools/t3/t3r3_legacy.py`): T3-R2's runs (q = 50, seed 60502) and (q = 60, seed 60518) re-run from their own `run_config.json`: **both bit-identical** to their originals by `tools/t3/compare_runs.py bw` (162 npz files, the four CSVs without wall-clock columns, the final networks, frozen stages and stream states; 424 s and 436 s).

**Full suite at `727116a3`** (12 shards, `results/t3_r3/tests_p2/`): **1,811 passed, 6 skipped, 2 xfailed, 2 failed.** The two failures are the known ones: `tests/test_registry_canonicalization.py::test_registry_canonicalization` and `tests/test_ms_r4_launcher.py::test_the_existing_launcher_is_unchanged` (the PI recorded it as known in the G1 reply; it has failed since the T3-1.0 lock changed `protocols/`). 1,746 (P1) + 65 new tests = 1,811. The MS-R3 tests that compare a run without the new keys with a recorded state (`test_without_the_new_keys_...`) pass, which is further evidence for the unchanged default path.

## 5. P3: the checks of the wave, the deviations, the final suite

**The wave** (`05_development.md` section 1): 122 runs from the pre-registration commit `b565f0e8`, all exit 0, no infrastructure event; the post-launch checks (`results/t3_r3/dev/launch_checks_absroot.json`) pass on all 122 runs except `clean_tree` (82 runs). The evaluation, the analysis and the blind recomputation agree (`results/t3_r3/dev/analysis/blind.json`: no difference).

**Deviations (PI reply `pi_record/28c_dirty_tree_reply.md`; evidence in `05_development.md` section 1):**

- **D-1.** 82 manifests record a dirty tree, because a report script (`reports/t3/r3/report_scripts/results_tables.py`) was written into the worktree at 02:21:15 UTC, about 73 s after the launch (`launch_start.txt` 02:20:02 UTC; 71 s after the launcher's own start stamp 02:20:04 UTC). The flag follows the file's creation without exception (40 clean runs started at 02:20:08-02:20:09, 82 dirty ones at 02:29:08-02:48:43); every manifest carries `b565f0e8`. Five clean-tree re-runs (two with a fired trigger; no `bw` run) are bit-identical to their originals (`results/t3_r3/dev/verify_clean_tree/`). The 122 runs remain the counted records.
- **D-2.** The first post-launch check used a relative `--root` (`config_equals_plan` 0 of 122, the same defect as T3-R2's F10); `launch_checks_absroot.json` is the record, `launch_checks.json` is kept. Commit `e6902c28`: the four tools with `--root` refuse a relative path (tested).
- **D-3.** `results/t3_r3/post_launch.sh` ran the evaluation, the analysis and the blind recomputation after the checks had exited 1. Commit `e6902c28`: `tools/t3/t3r3_post_launch.sh` stops at the first nonzero exit (tested).
- The `pair_ok` column is a check for `bwA` and the diagnostic runs only; for `bwAB` it is reported and decides nothing.

**New code and tests in P3.** `e6902c28` (the root check in `t3r3_{launch,eval,checks,analysis}.py`, `t3r3_post_launch.sh`, 5 tests in `tests/test_t3r3_launch.py`), `87ca2144` (the two verification drivers `t3r3_vct_boundary.py` and `t3r3_vct_rerun.py`, committed as they stand after the second, compare-only pass), `11354e1f` (`t3r3_last16.py`, 3 tests in `tests/test_t3r3_last16.py`). Nothing under `agents/`, `envs/`, `utils/`, `run/` or `protocols/` changed after `727116a3`.

**Full suite at `11354e1f`** (12 shards, `results/t3_r3/tests_p3/`): **1,819 passed, 6 skipped, 2 xfailed, 2 failed** (1,811 at `727116a3` plus 8 new tests). The two failures are the known ones: `tests/test_registry_canonicalization.py::test_registry_canonicalization` and `tests/test_ms_r4_launcher.py::test_the_existing_launcher_is_unchanged`.

**Independent fact-check** of the reports against the CSVs: `pi_record/01_factcheck.md`.
