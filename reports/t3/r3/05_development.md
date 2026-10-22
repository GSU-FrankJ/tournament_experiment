# T3-R3, P2: the development wave (results)

Date: 2026-10-22. Branch `t3-r3`. **Generated** from `reports/t3/r3/report_scripts/05_development.template.md` by `build_report.py`; every table by `reports/t3/r3/report_scripts/results_tables.py` from `results/t3_r3/dev/launch_checks_absroot.json`, `results/t3_r3/dev/verify_clean_tree/` and `results/t3_r3/dev/analysis/` (written by `tools/t3/t3r3_analysis.py`, `t3r3_blind.py`, `t3r3_last16.py`). The binding texts are the prompt (`pi_record/28_t3_r3_prompt.md`), the PI's replies at G1 (`28b_g1_reply.md`) and on the dirty-tree flag (`28c_dirty_tree_reply.md`), and the pre-registration (`04_preregistration.md`, pushed before the launch). Nothing in this report was tuned after the launch. Nothing is locked, and no confirmation seed was trained.

**In one paragraph.** One wave of 122 runs (120 development runs, seeds 70501-70520 at q = 50 and q = 60, in three arms, and two diagnostic runs at q = 60, seed 60518) ran to completion with exit code 0 and no infrastructure event. `bw` (the control) passes 37 of 40 runs, `bwA` 39 of 40 and `bwAB` 38 of 40; both candidates reach the target (at least 38 of 40 and at least 18 of 20 at each q), so by the pre-registered order **`bwA` is proposed**. The 2-run difference between `bwA` and `bw` is two seeds at q = 50 where the trigger fired and the second attempt passed; every other pair of runs is bit-identical (exact McNemar p = 0.5). The causal reading of B is "mixed", with no causal claim. 82 of the 122 manifests record a dirty working tree because a report script was written into the worktree about 73 s after the launch; the verification the PI asked for (section 1.3) found the five clean-tree re-runs bit-identical to their originals.

## 1. The checks

### 1.1 What was checked

The checks are those of `04_preregistration.md` section 5, run by `tools/t3/t3r3_checks.py` on all 122 runs: exit code and files, manifest commit and clean tree, the global-RNG assertion, the config against the plan, the lock check, the trigger recomputed from the check file it read, budgets and schedules (a run that fired also has 5,400 updates, 2,764,800 policy rows, 108,000 minibatch steps and 216 development checks), C-INIT, C-REPLAY and C-PAIR. The record is `results/t3_r3/dev/launch_checks_absroot.json`; the first call (`launch_checks.json`, kept) used a relative `--root` (D-2 below), and its counts are shown in the last column.

| check | pass | fail | not evaluated | first call (relative `--root`): pass / fail |
|---|---|---|---|---|
| exit_ok | 122 | 0 | 0 | 122 / 0 |
| files_complete | 122 | 0 | 0 | 122 / 0 |
| manifest_commit_ok | 122 | 0 | 0 | 122 / 0 |
| clean_tree | 40 | 82 | 0 | 40 / 82 |
| global_rng_ok | 122 | 0 | 0 | 122 / 0 |
| config_equals_plan | 122 | 0 | 0 | 0 / 122 |
| config_production_ok | 122 | 0 | 0 | 122 / 0 |
| trigger_consistent | 122 | 0 | 0 | 122 / 0 |
| budget_ok | 122 | 0 | 0 | 122 / 0 |
| lr_ok | 122 | 0 | 0 | 122 / 0 |
| refresh_ok | 122 | 0 | 0 | 122 / 0 |
| exports_ok | 122 | 0 | 0 | 122 / 0 |
| checks_ok | 122 | 0 | 0 | 122 / 0 |
| retry_budget_ok | 122 | 0 | 0 | 122 / 0 |
| init_equals_plan | 122 | 0 | 0 | 122 / 0 |
| replay_ok | 122 | 0 | 0 | 122 / 0 |
| pair_ok (binding for `bwA` and the diagnostic runs only) | 122 | 0 | 0 | 122 / 0 |

| arm | runs | trigger fired | trigger would fire (recomputed) | failed checks (other than `pair_ok`) | C-PAIR against `bw` |
|---|---|---|---|---|---|
| bw | 40 | 0 | 2 | clean_tree: 40 | - |
| bwA | 41 | 3 | 3 | clean_tree: 40 | 40 of 40 development runs as required (identical over the whole run where the trigger did not fire, same stage 3 up to the check where it fired); 1 of 1 diagnostic run equal to T3-R2's original up to the check |
| bwAB | 41 | 4 | 4 | clean_tree: 2 | 0 of 40 development runs identical to `bw` (not a check: `bwAB` differs from `bw` by construction); 1 of 1 diagnostic run equal to T3-R2's original up to the check |

Infrastructure ledger present: True; events: 0; stopped without verdict: []. Cross-arm C-INIT: attempt-1 digest differs across arms in 0 (q, seed); attempt-2 digests of `bwA` and `bwAB` differ in 0.

**Reading.** Every check passes except `clean_tree` (82 of 122 manifests record a dirty tree: D-1, examined in section 1.3). The `pair_ok` row counts a `bwAB` run as passing by construction: `bwAB` differs from `bw` wherever B changes a stage-3 update, so its pair with `bw` is computed and reported but is not a check (the arm table shows how many `bwAB` runs are nevertheless identical to `bw`). Only `bwA` and the diagnostic runs are held to C-PAIR, and they all pass it. No infrastructure failure happened, no run was re-run or replaced, and the seeds 80501-80540 reserved for the later confirmation were never trained.

### 1.2 Deviations from the pre-registered procedure

- **D-1. 82 manifests record a dirty tree.** I wrote the untracked file `reports/t3/r3/report_scripts/results_tables.py` into the worktree at 02:21:15 UTC, about 73 s after the launch (`launch_start.txt` 02:20:02 UTC; 71 s after the launcher's own start stamp 02:20:04 UTC). The runner's definition of `clean_tree` (`run/run_v2_stagewise.git_state`: tracked changes or untracked files anywhere except `results/`) turned false for every run that started after that moment. The evidence is in section 1.3. The launcher had checked the tree before the launch, as pre-registered; the error is mine, a write during the wave.
- **D-2. The first post-launch check used a relative `--root`** and reported `config_equals_plan` as 0 of 122 (the plan hashes the output directory relative to the root's parent, which a relative root cannot do). The absolute-root call (`launch_checks_absroot.json`) is the record, and the first file is kept. It is the same defect as T3-R2's F10, now in `t3r3_checks.py`. Since commit `e6902c28` the four tools that take `--root` (`t3r3_launch`, `t3r3_eval`, `t3r3_checks`, `t3r3_analysis`) refuse a relative path (tested), and so does `t3r3_last16` (commit `11354e1f`). One consequence: `analysis/diagnostic.csv` took the C-PAIR result of the diagnostic runs from the first-call `launch_checks.json`; `pair_ok` of the two diagnostic runs is the same in both check files, so no number changes.
- **D-3. `results/t3_r3/post_launch.sh` ran the evaluation, the analysis and the blind recomputation after the checks had exited 1.** Nothing was committed or published in between, and those steps only compute from the data; the script is kept as it ran. `tools/t3/t3r3_post_launch.sh` (commit `e6902c28`) stops at the first nonzero exit and refuses a relative root (tested).
- **The `pair_ok` column** is labelled in the tables as binding for `bwA` and the diagnostic runs only; for `bwAB` it is not a check (the first report of the checks showed 40 `bwAB` failures of it).
- **Tools changed after the launch.** `git diff b565f0e8 HEAD -- tools run agents envs utils protocols` lists, besides new files, only the root check in the four tools above; the evaluation, the analysis and the blind recomputation ran with the tools as launched (`b565f0e8`), and the logic of those tools is unchanged by that commit.

### 1.3 The dirty-tree flag, verified (reply 28c)

All outputs are under `results/t3_r3/dev/verify_clean_tree/`.

**1. The boundary.** From the 122 manifests and `status.json` start times:

| manifests record | runs | `bw` | `bwA` | `bwAB` | first start (UTC) | last start (UTC) |
|---|---|---|---|---|---|---|
| a clean tree | 40 | 0 | 1 | 39 | 2026-10-22T02:20:08.000+00:00 | 2026-10-22T02:20:09.000+00:00 |
| a dirty tree | 82 | 40 | 40 | 2 | 2026-10-22T02:29:08.000+00:00 | 2026-10-22T02:48:43.000+00:00 |

`reports/t3/r3/report_scripts/results_tables.py` was created at 2026-10-22 02:21:15.679567227 +0000 (birth time recorded in `boundary.json`); the wave was launched at 02:20:02 UTC. Runs whose recorded flag disagrees with 'started after the file was created': 0. All 122 manifests carry the commit `b565f0e8`; the diagnostic runs are all in the clean group (0 dirty). Table: `results/t3_r3/dev/verify_clean_tree/boundary.csv`.

The recorded flag follows the creation of the file without exception (the creation time is the birth time recorded in `boundary.json` before the file was moved for the re-runs; `stat` of the restored file now shows the time of the copy). The manifests store only the boolean, not the entries of `git status`, so 'the only entry is that file' is established from the file system instead: the search for paths outside `results/` and `.git` modified in the wave window 02:20:00-02:55:47 UTC finds that file and (in the search at 03:10 UTC, before the verification moved it) its parent directory, nothing else (`files_outside_results_in_wave_window.txt`); the same `git status --porcelain -- . ':(exclude)results'` the runner uses listed that file and the saved PI reply `28c_dirty_tree_reply.md` (written after the wave) when run after the re-runs with both restored (`files_outside_results_in_wave_window.txt`), and nothing while both were moved out (`rerun_stdout.txt`: 'tree clean outside results/'). Every manifest carries the commit `b565f0e8`. The runner opens no file under `reports/` (`run/run_ms_stagewise.py`, `run/ms_retry.py`, `run/run_v2_stagewise.py`, `run/v2_rollout.py`, `agents/`, `utils/` and `envs/` mention `reports/` only in one docstring).

**2. Pairs across the boundary.**

Pairs of `bwA` and `bw` of the same seed: 40. Pairs in which the two runs record different flags: 0. Of the 40 pairs, 38 are bit-identical over the whole run (the trigger did not fire) and 2 have the same stage 3 up to the check (it fired); bit-identical with different flags: 0; same stage 3 with different flags: 0. Table: `results/t3_r3/dev/verify_clean_tree/pairs_across_boundary.csv`.

The count of `bwA`-`bw` pairs that are bit-identical while one run recorded a clean tree and the other a dirty one is therefore **0**: all 40 `bw` runs and the 40 development `bwA` runs are dirty-flagged, and the one clean `bwA` run is the diagnostic run, which has no `bw` partner. The pairs do not cross the boundary and carry no evidence on it; the re-runs below do.

**3. Clean-tree re-runs (verification only, never counted).** `reports/t3/r3/report_scripts/results_tables.py` and the saved reply `28c_dirty_tree_reply.md` were moved out of the worktree for the re-runs and restored afterwards; at the start HEAD was `b565f0e8` and `git status --porcelain` showed nothing outside `results/`. Each run was re-run from its own `run_config.json` into `verify_clean_tree/q{q}/seed{s}/{arm}/` (single-threaded, in tmux, `tools/t3/t3r3_vct_rerun.py`) and compared with its original bit for bit by `tools/t3/compare_runs.py bw`, by the set of files, by `attempt1/` and `retry.json` where present, and by the small JSON records. The selection follows the reply: every dirty-flagged `bwA` run whose trigger fired (two; there is no dirty-flagged `bwAB` run that fired, and the two diagnostic runs are clean-flagged already), and the lowest-numbered dirty-flagged `bwAB` run that did not fire at each q. **Only q = 60 has such a run: 38 of the 40 development `bwAB` runs (all 20 of q = 50 and the q = 60 seeds 70501-70518) are clean-flagged, and only q = 60 seeds 70519 and 70520 are dirty.** To cover the dirty-flagged non-fired case at q = 50 as well, I added the lowest-numbered dirty-flagged non-fired `bwA` run at each q (marked 'addition'; `bwA` without a trigger is bit-identical to `bw`).

| run | why this run | flag, original | flag, re-run | `compare_runs.py bw` exit code | npz files identical | same file set | `attempt1/` (64 exports, CSVs, bin maps, state) and `retry.json` | gates / rule log / drift / summary JSON | re-run wall time (s, driver clock) |
|---|---|---|---|---|---|---|---|---|---|
| `t3r3_q50_s70508_bwA` | dirty-flagged bwA run whose trigger fired | dirty | clean | 0 | 162 | yes | identical | identical | 496 |
| `t3r3_q50_s70516_bwA` | dirty-flagged bwA run whose trigger fired | dirty | clean | 0 | 162 | yes | identical | identical | 494 |
| `t3r3_q60_s70519_bwAB` | lowest-numbered dirty-flagged bwAB run at q=60 whose trigger did not fire | dirty | clean | 0 | 162 | yes | no `attempt1/` | identical | 368 |
| `t3r3_q50_s70501_bwA` | (addition) lowest-numbered dirty-flagged bwA run at q=50 whose trigger did not fire | dirty | clean | 0 | 162 | yes | no `attempt1/` | identical | 377 |
| `t3r3_q60_s70501_bwA` | (addition) lowest-numbered dirty-flagged bwA run at q=60 whose trigger did not fire | dirty | clean | 0 | 162 | yes | no `attempt1/` | identical | 363 |

**Result: all five are identical to their originals** (`compare_runs.py bw` exit code 0 for each: 162 npz files, the four CSVs without their wall-clock columns, the final networks, frozen stages and stream states; the same set of files; for the two fired runs also the 64 exports, the stage-3 rows and checks, the bin maps and the state of `attempt1/`, and `retry.json` apart from wall-clock fields; the gates, rule log, drift test and summary JSON apart from the `clean_tree` flag, which is the thing under test and reads `false` in the original and `true` in the re-run). The re-run manifests record the commit `b565f0e8` and a clean tree. **The flag is recorded as a deviation (D-1); the 122 runs remain the counted records exactly as they are.**

*A note on the comparison itself.* The first pass of the comparison (`rerun_results_first_comparison.json`, kept) reported 'not ok' for all five, only because my JSON comparison treated `NaN != NaN` (the `R_tail` of stages 1 and 2 is NaN in both runs). With NaN equal to NaN and the `clean_tree` flag excluded, the comparison (`rerun_results_final.json`, `--compare-only`, no retraining) passes; `compare_runs.py` itself returned 0 in both passes. The committed driver `tools/t3/t3r3_vct_rerun.py` is the version of that second pass; the training itself ran with the version before the NaN rule and the `--compare-only` option.

### 1.4 The full suite

`pytest tests` as 12 parallel shards at the commit `11354e1f` (`results/t3_r3/tests_p3/`): 1,819 passed, 6 skipped, 2 xfailed, 2 failed. The two failures are the known ones recorded at G1: `tests/test_registry_canonicalization.py::test_registry_canonicalization` and `tests/test_ms_r4_launcher.py::test_the_existing_launcher_is_unchanged` (it has failed since the T3-1.0 lock changed `protocols/`). Nothing else fails.

## 2. The arms and the code version

| arm | scheme keys | runs | role |
|---|---|---|---|
| `bw` | none (the locked T3-1.0 settings) | 40 | control; not eligible |
| `bwA` | `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}` | 40 (+ 1 diagnostic) | candidate A; first in the selection order |
| `bwAB` | `stage3_retry` as above and `terminal_clamp_likelihood = "censored"` | 40 (+ 1 diagnostic) | candidates A and B; second |

The final code commit of the scheme is `727116a3` (`run/ms_retry.py`, the additive key-gated changes in `run/run_ms_stagewise.py`, `tools/t3/t3r3_*.py`); C-LEGACY ran at it (two T3-R2 runs re-run bit-identically, `04_preregistration.md` section 8). The wave was launched at 02:20:02 UTC from the pre-registration commit `b565f0e8` (`results/t3_r3/launch_commit.txt`), 40 workers, single-threaded, in tmux; it ended at 02:55:46 UTC (`launch_start.txt`, `launch_end.txt`); all 122 runs exited 0. The seeds are the same in every arm; `reports/t3/r3/dev_plan.csv` holds the planned config hashes and init digests (all matched: C-INIT).

## 3. Per-seed results

Each cell is the run's verdict under the pass rule of T3-1.0 (G-A, G-F, G-N, exit code, global-RNG assertion, valid evaluation on both tiers), `eta_3` and `Gmax_full` in units of the prize gap. `[fired]` marks a candidate run whose trigger fired (the verdict is that of the frozen second attempt); `[would fire]` marks a `bw` run to which the trigger rule applied to its own local-1600 check would fire. The medians and the worst values are in section 4.

**q = 50**

| seed | bw | bw eta_3 | bw Gmax | bwA | bwA eta_3 | bwA Gmax | bwAB | bwAB eta_3 | bwAB Gmax |
|---|---|---|---|---|---|---|---|---|---|
| 70501 | pass | 0.002083 | 0.002083 | pass | 0.002083 | 0.002083 | pass | 0.00166 | 0.00166 |
| 70502 | pass | 0.001004 | 0.001092 | pass | 0.001004 | 0.001092 | pass | 0.001203 | 0.001203 |
| 70503 | pass | 0.001631 | 0.001631 | pass | 0.001631 | 0.001631 | pass | 0.001373 | 0.001373 |
| 70504 | pass | 0.00115 | 0.00133 | pass | 0.00115 | 0.00133 | pass | 0.003613 | 0.003613 |
| 70505 | pass | 0.002917 | 0.002917 | pass | 0.002917 | 0.002917 | pass | 0.003579 | 0.003579 |
| 70506 | pass | 0.001006 | 0.001705 | pass | 0.001006 | 0.001705 | pass | 0.00158 | 0.00158 |
| 70507 | pass | 0.001217 | 0.001217 | pass | 0.001217 | 0.001217 | pass | 0.000876 | 0.001051 |
| 70508 | FAIL (gate_G_A) [would fire] | 0.00644 | 0.00644 | pass [fired] | 0.00441 | 0.00441 | FAIL (gate_G_A) | 0.003334 | 0.003334 |
| 70509 | pass | 0.004605 | 0.004605 | pass | 0.004605 | 0.004605 | pass | 0.00394 | 0.00394 |
| 70510 | pass | 0.004274 | 0.004274 | pass | 0.004274 | 0.004274 | pass [fired] | 0.003782 | 0.003782 |
| 70511 | pass | 0.002957 | 0.002957 | pass | 0.002957 | 0.002957 | pass | 0.001347 | 0.001347 |
| 70512 | pass | 0.001129 | 0.001129 | pass | 0.001129 | 0.001129 | pass | 0.004338 | 0.004338 |
| 70513 | pass | 0.003171 | 0.003171 | pass | 0.003171 | 0.003171 | pass | 0.004148 | 0.004148 |
| 70514 | pass | 0.003677 | 0.003677 | pass | 0.003677 | 0.003677 | pass [fired] | 0.001856 | 0.001856 |
| 70515 | pass | 0.001791 | 0.001791 | pass | 0.001791 | 0.001791 | pass | 0.00204 | 0.00204 |
| 70516 | FAIL (gate_G_A) [would fire] | 0.006252 | 0.006252 | pass [fired] | 0.002022 | 0.002022 | pass [fired] | 0.003212 | 0.003212 |
| 70517 | pass | 0.002266 | 0.002266 | pass | 0.002266 | 0.002266 | pass | 0.001297 | 0.002133 |
| 70518 | pass | 0.00288 | 0.00288 | pass | 0.00288 | 0.00288 | pass | 0.001986 | 0.001986 |
| 70519 | pass | 0.002007 | 0.002007 | pass | 0.002007 | 0.002007 | pass | 0.001808 | 0.001808 |
| 70520 | pass | 0.001214 | 0.001214 | pass | 0.001214 | 0.001214 | FAIL (gate_G_A) | 0.004129 | 0.004129 |

**q = 60**

| seed | bw | bw eta_3 | bw Gmax | bwA | bwA eta_3 | bwA Gmax | bwAB | bwAB eta_3 | bwAB Gmax |
|---|---|---|---|---|---|---|---|---|---|
| 70501 | pass | 0.000527 | 0.000527 | pass | 0.000527 | 0.000527 | pass | 0.002188 | 0.002188 |
| 70502 | pass | 0.000792 | 0.000792 | pass | 0.000792 | 0.000792 | pass | 0.00039 | 0.00039 |
| 70503 | pass | 0.003563 | 0.003927 | pass | 0.003563 | 0.003927 | pass | 0.002772 | 0.002772 |
| 70504 | pass | 0.000844 | 0.000844 | pass | 0.000844 | 0.000844 | pass | 0.000462 | 0.000462 |
| 70505 | pass | 0.001833 | 0.001833 | pass | 0.001833 | 0.001833 | pass | 0.002996 | 0.002996 |
| 70506 | pass | 0.000446 | 0.000446 | pass | 0.000446 | 0.000446 | pass | 0.000649 | 0.000649 |
| 70507 | pass | 0.002423 | 0.002423 | pass | 0.002423 | 0.002423 | pass | 0.00283 | 0.00283 |
| 70508 | pass | 0.001287 | 0.001287 | pass | 0.001287 | 0.001287 | pass | 0.002365 | 0.002365 |
| 70509 | pass | 0.000655 | 0.000655 | pass | 0.000655 | 0.000655 | pass | 0.000702 | 0.000777 |
| 70510 | pass | 0.000831 | 0.000831 | pass | 0.000831 | 0.000831 | pass | 0.001475 | 0.001475 |
| 70511 | FAIL (gate_G_A;gate_G_F) | 0.003955 | 0.04639 | FAIL (gate_G_A;gate_G_F) | 0.003955 | 0.04639 | pass | 0.001866 | 0.001866 |
| 70512 | pass | 0.000894 | 0.000894 | pass | 0.000894 | 0.000894 | pass | 0.001557 | 0.001557 |
| 70513 | pass | 0.00222 | 0.00222 | pass | 0.00222 | 0.00222 | pass | 0.00144 | 0.00144 |
| 70514 | pass | 0.001576 | 0.001576 | pass | 0.001576 | 0.001576 | pass | 0.001556 | 0.001556 |
| 70515 | pass | 0.000886 | 0.000886 | pass | 0.000886 | 0.000886 | pass | 0.001123 | 0.001123 |
| 70516 | pass | 0.000769 | 0.000769 | pass | 0.000769 | 0.000769 | pass | 0.000703 | 0.000703 |
| 70517 | pass | 0.002264 | 0.002264 | pass | 0.002264 | 0.002264 | pass | 0.002408 | 0.002408 |
| 70518 | pass | 0.001359 | 0.001359 | pass | 0.001359 | 0.001359 | pass | 0.002818 | 0.002818 |
| 70519 | pass | 0.001274 | 0.001274 | pass | 0.001274 | 0.001274 | pass | 0.001476 | 0.001476 |
| 70520 | pass | 0.000686 | 0.000686 | pass | 0.000686 | 0.000686 | pass | 0.000999 | 0.000999 |

## 4. The target outcome per arm and the selection

An arm reaches the target iff at least 38 of its 40 runs pass and at least 18 of 20 at each q. Only `bwA` and `bwAB` are eligible; in that order the first arm that reaches the target is proposed. Exact (Clopper-Pearson) 95 % intervals.

| arm | q | runs | pass | exact 95 % interval | worst Gmax_full (run) | median eta_3 | median rmse_pos/e*(0) | median tail_mean/e*(0) | median peak_T_signed | median Gmax_full | median EXP_root |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bw | 50 | 20 | 18 | [0.683, 0.988] | 0.00644 (t3r3_q50_s70508_bw) | 0.002175 | 0.02771 | 0.005029 | -0.0703 | 0.002175 | 0.000586 |
| bw | 60 | 20 | 19 | [0.751, 0.999] | 0.04639 (t3r3_q60_s70511_bw) | 0.001084 | 0.02669 | 0.004785 | -0.07353 | 0.001084 | 0.000255 |
| bw | both | 40 | 37 | [0.796, 0.984] | 0.04639 (t3r3_q60_s70511_bw) | 0.001603 | 0.0277 | 0.004912 | -0.07033 | 0.001668 | 0.000413 |
| bwA | 50 | 20 | 20 | [0.832, 1.000] | 0.00460 (t3r3_q50_s70509_bwA) | 0.002052 | 0.02771 | 0.005029 | -0.06675 | 0.002052 | 0.000577 |
| bwA | 60 | 20 | 19 | [0.751, 0.999] | 0.04639 (t3r3_q60_s70511_bwA) | 0.001084 | 0.02669 | 0.004785 | -0.07353 | 0.001084 | 0.000255 |
| bwA | both | 40 | 39 | [0.868, 0.999] | 0.04639 (t3r3_q60_s70511_bwA) | 0.001603 | 0.0277 | 0.004912 | -0.06751 | 0.001668 | 0.000402 |
| bwAB | 50 | 20 | 18 | [0.683, 0.988] | 0.00434 (t3r3_q50_s70512_bwAB) | 0.002013 | 0.02756 | 0.005093 | -0.07039 | 0.002086 | 0.000519 |
| bwAB | 60 | 20 | 20 | [0.832, 1.000] | 0.00300 (t3r3_q60_s70505_bwAB) | 0.001516 | 0.02811 | 0.005147 | -0.07843 | 0.001516 | 0.000289 |
| bwAB | both | 40 | 38 | [0.831, 0.994] | 0.00434 (t3r3_q50_s70512_bwAB) | 0.001832 | 0.02756 | 0.005098 | -0.07052 | 0.001861 | 0.000319 |

| arm | eligible | pass q50 | pass q60 | pass total | total >= 38 | q50 >= 18 | q60 >= 18 | reaches the target |
|---|---|---|---|---|---|---|---|---|
| bw | no (control) | 18 of 20 | 19 of 20 | 37 of 40 | no | yes | yes | no |
| bwA | yes | 20 of 20 | 19 of 20 | 39 of 40 | yes | yes | yes | yes |
| bwAB | yes | 18 of 20 | 20 of 20 | 38 of 40 | yes | yes | yes | yes |

**Selection:** bwA is the first candidate arm in the order ['bwA', 'bwAB'] that reaches the target; both arms reach it, so bwA is proposed (proposed: `bwA`; both arms reach the target: True).

| arm | pass q50 | pass q60 | pass total | reaches the target |
|---|---|---|---|---|
| bw | 18 | 19 | 37 | no |
| bwA | 20 | 19 | 39 | yes |
| bwAB | 18 | 20 | 38 | yes |

Blind recomputation (`tools/t3/t3r3_blind.py`): proposed `bwA`; agrees with the analysis: True; differences: [].

**First-attempt outcome per arm (G1 reply 2.1).** The pass count when every fired run counts as a failure of its first attempt. For `bwA` it must equal `bw`'s, seed by seed, under the same convention (a `bw` run counts as fired when the trigger rule applied to its own check fires): it does, with no mismatch, so the run does not stop.

| arm | first-attempt pass (fired = failure) | pass of the run | trigger fired |
|---|---|---|---|
| bw | 37 | 37 | 2 |
| bwA | 37 | 39 | 2 |
| bwAB | 35 | 38 | 3 |

Invariant `bwA` = `bw` per seed under the same convention: mismatches 0; stop: False. Raw pass count of `bw`: 37; `bw` runs that would fire and nevertheless pass the gates: 0.

**What the selection says, and what it does not.** By the pre-registered order `bwA` is proposed: both arms reach the target and `bwA` comes first. The PI may lock `bwAB` instead only with a stated reason taken from the items of section 2.2 of the G1 reply; section 8 reports those items. `bwAB` reaches the target with no margin (38 of 40, and exactly 18 of 20 at q = 50). The development evidence is 40 runs per arm: the exact interval of the pass rate of `bwA` is [0.868, 0.999] and of `bwAB` [0.831, 0.994], and the operating characteristics in `04_preregistration.md` section 10 show that a rate of 0.95 reaches this target with probability 0.68 and a rate of 0.90 with probability 0.22.

## 5. The paired comparison with `bw`

Paired by seed (same q and seed). `Gmax_full` and `eta_3` in units of the prize gap; intervals from the paired bootstrap (10,000 resamples, seed 20261021).

| comparison | statistic | q | pairs | mean a | mean b | mean difference | paired bootstrap 95 % interval | seeds with a < b / a > b |
|---|---|---|---|---|---|---|---|---|
| bwA - bw | Gmax_full_over_dw | 50 | 20 | 0.002419 | 0.002732 | -0.000313 | [-0.00084, 0.00000] | 2 / 0 |
| bwA - bw | Gmax_full_over_dw | 60 | 20 | 0.003594 | 0.003594 | 0 | [0.00000, 0.00000] | 0 / 0 |
| bwA - bw | Gmax_full_over_dw | both | 40 | 0.003007 | 0.003163 | -0.000156 | [-0.00042, 0.00000] | 2 / 0 |
| bwA - bw | eta_T_over_dw | 50 | 20 | 0.002371 | 0.002684 | -0.000313 | [-0.00084, 0.00000] | 2 / 0 |
| bwA - bw | eta_T_over_dw | 60 | 20 | 0.001454 | 0.001454 | 0 | [0.00000, 0.00000] | 0 / 0 |
| bwA - bw | eta_T_over_dw | both | 40 | 0.001912 | 0.002069 | -0.000156 | [-0.00042, 0.00000] | 2 / 0 |
| bwAB - bw | Gmax_full_over_dw | 50 | 20 | 0.002606 | 0.002732 | -0.000126 | [-0.00082, 0.00059] | 13 / 7 |
| bwAB - bw | Gmax_full_over_dw | 60 | 20 | 0.001643 | 0.003594 | -0.001952 | [-0.00659, 0.00051] | 7 / 13 |
| bwAB - bw | Gmax_full_over_dw | both | 40 | 0.002124 | 0.003163 | -0.001039 | [-0.00349, 0.00038] | 20 / 20 |
| bwAB - bw | eta_T_over_dw | 50 | 20 | 0.002555 | 0.002684 | -0.000129 | [-0.00083, 0.00061] | 12 / 8 |
| bwAB - bw | eta_T_over_dw | 60 | 20 | 0.001639 | 0.001454 | 0.000185 | [-0.00019, 0.00053] | 7 / 13 |
| bwAB - bw | eta_T_over_dw | both | 40 | 0.002097 | 0.002069 | 2.8e-05 | [-0.00038, 0.00044] | 19 / 21 |
| bwAB - bwA | Gmax_full_over_dw | 50 | 20 | 0.002606 | 0.002419 | 0.000187 | [-0.00037, 0.00079] | 12 / 8 |
| bwAB - bwA | Gmax_full_over_dw | 60 | 20 | 0.001643 | 0.003594 | -0.001952 | [-0.00659, 0.00051] | 7 / 13 |
| bwAB - bwA | Gmax_full_over_dw | both | 40 | 0.002124 | 0.003007 | -0.000883 | [-0.00332, 0.00049] | 19 / 21 |
| bwAB - bwA | eta_T_over_dw | 50 | 20 | 0.002555 | 0.002371 | 0.000184 | [-0.00040, 0.00081] | 11 / 9 |
| bwAB - bwA | eta_T_over_dw | 60 | 20 | 0.001639 | 0.001454 | 0.000185 | [-0.00019, 0.00053] | 7 / 13 |
| bwAB - bwA | eta_T_over_dw | both | 40 | 0.002097 | 0.001912 | 0.000184 | [-0.00016, 0.00055] | 18 / 22 |

Pass / fail cross-tables over the 40 seeds:

| arm | reference | both pass | reference only | arm only | both fail | exact McNemar p (two-sided) |
|---|---|---|---|---|---|---|
| bwA | bw | 37 | 0 | 2 | 1 | 0.5 |
| bwAB | bw | 36 | 1 | 2 | 1 | 1 |
| bwAB | bwA | 37 | 2 | 1 | 0 | 1 |

**Reading.** `bwA` and `bw` are bit-identical on 38 of 40 seeds (the table shows exactly zero differences there) and differ on the 2 seeds where A fired, where the second attempt passes and `bw` fails: 2 seeds better, none worse, exact McNemar p = 0.5. `bwAB` differs from `bw` on all 40 seeds (no `bwAB` run is identical to its `bw` run) because B changes the stage-3 update itself; its paired differences of `eta_3` and `Gmax_full` have intervals that contain 0 at both q, its pass / fail cross-table against `bw` is 2 better, 1 worse (p = 1) and against `bwA` 1 better, 2 worse (p = 1).

## 6. The failures of every arm, classified as in D2

The class is assigned by the D2 definitions: (F) formation failure; (K) the run is in the stage-3 clamp regime (S3 of the frozen attempt above 0.2) and its failing quantity is a stage-3 in-band bias or a stage-2 near-zero stretch; (V) the tent forms, the run is not in the regime, and the failing value sits at the top of its own range over the last 16 exports of the phase. 'Last 16 exports' is `tools/t3/t3r3_last16.py` (the exports `u01225` ... `u01600` evaluated on the final tier; `u01600` reproduces each run's evaluated value to 1e-9).

| run | failing gates | eta_3 | rmse_pos/e*(0) | tail_mean/e*(0) | Gmax_full | S3 (first attempt) | S3 (frozen) | trigger fired | stage-2 stretch nodes | dev Delta at local 1600 (frozen attempt) | last 16 exports (failing quantities) | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `t3r3_q50_s70508_bwAB` | gate_G_A | 0.003334 | 0.05319 | 0.004076 | 0.003334 | 0.701 | 0.701 | no | 0 | 0.003334 | rmse_pos 0.05319 > 0.05: rank 6 of 16 from the top, 6 of 16 above the limit | K |
| `t3r3_q50_s70520_bwAB` | gate_G_A | 0.004129 | 0.05428 | 0.005184 | 0.004129 | 0.009769 | 0.009769 | no | 0 | 0.004096 | rmse_pos 0.05428 > 0.05: rank 1 of 16 from the top, 1 of 16 above the limit | V |
| `t3r3_q60_s70511_bwA` | gate_G_A;gate_G_F | 0.003955 | 0.05253 | 0.003422 | 0.04639 | 0.5235 | 0.5235 | no | 28 | 0.003955 | rmse_pos 0.05253 > 0.05: rank 3 of 16 from the top, 4 of 16 above the limit | K |
| `t3r3_q50_s70508_bw` | gate_G_A | 0.00644 | 0.05348 | 0.002348 | 0.00644 | 0.7769 | 0.7769 | no | 0 | 0.00644 | eta_3 0.00644 > 0.005: rank 12 of 16 from the top, 13 of 16 above the limit; rmse_pos 0.05348 > 0.05: rank 12 of 16 from the top, 15 of 16 above the limit | K |
| `t3r3_q50_s70516_bw` | gate_G_A | 0.006252 | 0.05186 | 0.002441 | 0.006252 | 0.7553 | 0.7553 | no | 0 | 0.006252 | eta_3 0.00625 > 0.005: rank 15 of 16 from the top, 16 of 16 above the limit; rmse_pos 0.05186 > 0.05: rank 14 of 16 from the top, 15 of 16 above the limit | K |
| `t3r3_q60_s70511_bw` | gate_G_A;gate_G_F | 0.003955 | 0.05253 | 0.003422 | 0.04639 | 0.5235 | 0.5235 | no | 28 | 0.003955 | rmse_pos 0.05253 > 0.05: rank 3 of 16 from the top, 4 of 16 above the limit | K |

There are four distinct failing seeds and six failing runs. **Five of the six are class (K)**: `bw` seeds 70508 and 70516 at q = 50 (in-band bias, S3 0.78 and 0.76), `bw` and `bwA` seed 70511 at q = 60 (a stage-2 near-zero stretch of 28 nodes, `Gmax_full` 0.046, S3 0.52), `bwAB` seed 70508 (S3 0.70). **One is class (V)**: `bwAB` seed 70520 at q = 50 (S3 0.0098; `rmse_pos` 0.0543 against the limit 0.05 is the largest of its last 16 exports, 1 of 16 above the limit). No failure is class (F) (the formation measure F(u01600) is above both thresholds in all 120 development runs); I assigned none to (E) or (U), which are judgements. The two `bw` failures at q = 50 are the two that A repairs (section 7). The q = 60 stretch failure (seed 70511) is invisible to the trigger: its stage-3 development `Delta` at local 1600 is 0.0040, below 0.005, and a stage-2 stretch is not seen by the stage-3 check; A cannot reach it. In `bwAB` the same seed passes, but its stage-3 first attempt differs from `bw`'s, so that is not evidence of an effect of B on the stretch.

## 7. The triggers that fired

| run | kind | trigger Delta | s | C | reason | S3 of attempt 1 | S3 of the frozen attempt 2 | attempt-2 final Delta | outcome | eta_3 | Gmax_full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `t3r3_q60_s60518_bwA` | diagnostic | 0.1447 | 41.92 | 0.01477 | delta_gt | 0 | 0.01215 | 0.001297 | pass | 0.0013 | 0.0013 |
| `t3r3_q60_s60518_bwAB` | diagnostic | 0.1447 | 41.92 | 0.01477 | delta_gt | 0 | 0.01539 | 0.00356 | pass | 0.00356 | 0.00356 |
| `t3r3_q50_s70510_bwAB` | dev | 0.005802 | 67.35 | 0.04065 | delta_gt | 0.7467 | 0.000748 | 0.003759 | pass | 0.003782 | 0.003782 |
| `t3r3_q50_s70514_bwAB` | dev | 0.005646 | 65.91 | 0.03676 | delta_gt | 0.6754 | 0.01116 | 0.001788 | pass | 0.001856 | 0.001856 |
| `t3r3_q50_s70516_bwAB` | dev | 0.005096 | 66.15 | 0.03645 | delta_gt | 0.7119 | 0.05736 | 0.003212 | pass | 0.003212 | 0.003212 |
| `t3r3_q50_s70508_bwA` | dev | 0.00644 | 65.53 | 0.04046 | delta_gt | 0.7769 | 0.7556 | 0.00441 | pass | 0.00441 | 0.00441 |
| `t3r3_q50_s70516_bwA` | dev | 0.006252 | 65.58 | 0.0377 | delta_gt | 0.7553 | 0.07741 | 0.001958 | pass | 0.002022 | 0.002022 |

Five development runs fired (`bwA`: seeds 70508 and 70516 at q = 50; `bwAB`: seeds 70510, 70514 and 70516 at q = 50) and the two diagnostic runs. **Every fired run passes on its second attempt** (5 of 5 development, 2 of 2 diagnostic). The two `bwA` fires are the two `bw` runs that the trigger rule would fire on and that fail in `bw` (the `bw` runs that would fire and nevertheless pass: 0). None fired at q = 60 outside the diagnostic runs. Two notes: (i) the second attempt of `bwA` seed 70508 is itself in the clamp regime (S3 0.756) and passes with `eta_3` 0.00441 against the limit 0.005, so a restart does not always leave the regime; (ii) the first attempts of `bwAB` differ from `bw`'s, so `bwAB` fired on seeds 70510 and 70514, which `bw` did not.

## 8. S3 and the stage-2 stretches; the causal reading of B

| arm | q | runs | S3 > 0.2, first attempt | S3 > 0.2, frozen attempt | median S3, frozen | runs with a stage-2 stretch | stretch nodes, total | passing runs with S3 > 0.2 (frozen) | failing runs with S3 > 0.2 (frozen) |
|---|---|---|---|---|---|---|---|---|---|
| bw | 50 | 20 | 4 | 4 | 0.02129 | 1 | 10 | 2 | 2 |
| bw | 60 | 20 | 8 | 8 | 0.04763 | 2 | 42 | 7 | 1 |
| bw | both | 40 | 12 | 12 | 0.02912 | 3 | 52 | 9 | 3 |
| bwA | 50 | 20 | 4 | 3 | 0.02129 | 1 | 10 | 3 | 0 |
| bwA | 60 | 20 | 8 | 8 | 0.04763 | 2 | 42 | 7 | 1 |
| bwA | both | 40 | 12 | 11 | 0.02912 | 3 | 52 | 10 | 1 |
| bwAB | 50 | 20 | 4 | 1 | 0.01162 | 0 | 0 | 0 | 1 |
| bwAB | 60 | 20 | 9 | 9 | 0.06936 | 1 | 2 | 9 | 0 |
| bwAB | both | 40 | 13 | 10 | 0.02306 | 1 | 2 | 9 | 1 |

The items of section 2.2 of the G1 reply, paired by seed on first attempts (for a fired run, `attempt1/`):

| arm | q | runs | runs with S3 > 0.2 (first attempt) | median S3 (first) | runs with a tail node at the mean clamp (u01600) | median share of tail nodes at the mean clamp | median minimum alpha | median dev Delta, local 1600 | fire rate | runs with a stage-2 stretch |
|---|---|---|---|---|---|---|---|---|---|---|
| bw | 50 | 20 | 4 | 0.02129 | 3 | 0 | 0.305 | 0.002145 | 2 of 20 | 1 |
| bw | 60 | 20 | 8 | 0.04763 | 6 | 0 | 0.2722 | 0.000973 | 0 of 20 | 2 |
| bw | both | 40 | 12 | 0.02912 | 9 | 0 | 0.2722 | 0.001505 | 2 of 40 | 3 |
| bwA | 50 | 20 | 4 | 0.02129 | 3 | 0 | 0.305 | 0.002145 | 2 of 20 | 1 |
| bwA | 60 | 20 | 8 | 0.04763 | 6 | 0 | 0.2722 | 0.000973 | 0 of 20 | 2 |
| bwA | both | 40 | 12 | 0.02912 | 9 | 0 | 0.2722 | 0.001505 | 2 of 40 | 3 |
| bwAB | 50 | 20 | 4 | 0.02067 | 4 | 0 | 0.3046 | 0.00268 | 3 of 20 | 0 |
| bwAB | 60 | 20 | 9 | 0.06936 | 3 | 0 | 0.2078 | 0.001472 | 0 of 20 | 1 |
| bwAB | both | 40 | 13 | 0.04254 | 7 | 0 | 0.2479 | 0.001759 | 3 of 40 | 1 |

Paired by seed, on first attempts:

| pair | q | regime runs a / b | McNemar: a only / b only | exact p | seeds with no trigger in either arm | eta_3 mean difference [95 %] | peak_T_signed mean difference [95 %] |
|---|---|---|---|---|---|---|---|
| bwAB vs bw | 50 | 4 / 4 | 0 / 0 | 1 | 16 | 0.00037 [-0.00027, 0.00106] | -0.0079 [-0.0287, 0.0139] |
| bwAB vs bw | 60 | 9 / 8 | 1 / 0 | 1 | 20 | 0.00018 [-0.00019, 0.00053] | 0.0017 [-0.0094, 0.0128] |
| bwAB vs bw | both | 13 / 12 | 1 / 0 | 1 | 36 | 0.00027 [-0.00008, 0.00064] | -0.0025 [-0.0137, 0.0090] |
| bwAB vs bwA | 50 | 4 / 4 | 0 / 0 | 1 | 16 | 0.00037 [-0.00027, 0.00106] | -0.0079 [-0.0287, 0.0139] |
| bwAB vs bwA | 60 | 9 / 8 | 1 / 0 | 1 | 20 | 0.00018 [-0.00019, 0.00053] | 0.0017 [-0.0094, 0.0128] |
| bwAB vs bwA | both | 13 / 12 | 1 / 0 | 1 | 36 | 0.00027 [-0.00008, 0.00064] | -0.0025 [-0.0137, 0.0090] |
| bwA vs bw | 50 | 4 / 4 | 0 / 0 | 1 | 18 | 0.00000 [0.00000, 0.00000] | 0.0000 [0.0000, 0.0000] |
| bwA vs bw | 60 | 8 / 8 | 0 / 0 | 1 | 20 | 0.00000 [0.00000, 0.00000] | 0.0000 [0.0000, 0.0000] |
| bwA vs bw | both | 12 / 12 | 0 / 0 | 1 | 38 | 0.00000 [0.00000, 0.00000] | 0.0000 [0.0000, 0.0000] |

**Pre-registered reading** (`04_preregistration.md` section 7): (i) the number of first-attempt runs with S3 > 0.2 is smaller in `bwAB` than in `bw` and in `bwA`, with the exact McNemar p < 0.05 on the pooled pairs; (ii) the number of runs with a stage-2 stretch is smaller in `bwAB` than in `bw` and in `bwA`; (iii) on the seeds where the trigger fired in neither arm, the paired bootstrap interval of the difference of `eta_3` (`bwAB` - `bw`, pooled) lies entirely below 0. All three: 'more than a marker'; none: 'a marker'; otherwise 'mixed', with no causal claim.

- (i) **not met**: 13 first-attempt runs with S3 > 0.2 in `bwAB` against 12 in `bw` and in `bwA`; the exact McNemar test has 1 discordant pair (the regime in `bwAB` only), p = 1.
- (ii) **met**: 1 run with a stage-2 stretch in `bwAB` (2 nodes) against 3 in `bw` and in `bwA` (52 nodes). Descriptively, an exact Fisher test of 3 of 40 against 1 of 40 gives p = 0.62; the counts are small.
- (iii) **not met**: the interval is [-0.00008, 0.00064] on 36 seeds (pooled), which contains 0; the interval of `peak_T_signed` is [-0.0137, 0.0090].

**The reading is 'mixed', and no causal claim is made.** The other items are also without a visible effect of B: the share of out-of-band nodes at the actor's mean clamp at `u01600` has median 0 in every arm (runs with at least one such tail node: 9 in `bw` and `bwA`, 7 in `bwAB`), the median minimum alpha is 0.272 (`bw`) and 0.248 (`bwAB`), the median development `Delta` at local 1600 is 0.0015 (`bw`) and 0.0018 (`bwAB`), and the fire rate is 2 of 40 for `bw` (would fire), 2 of 40 for `bwA` and 3 of 40 for `bwAB`. The frozen-attempt count of runs with S3 > 0.2 is 12 (`bw`), 11 (`bwA`) and 10 (`bwAB`); it differs from the first-attempt count only through the restarts: of the two `bwA` restarts one leaves the regime (seed 70516) and one stays in it (seed 70508), and all three `bwAB` restarts leave it.

## 9. The diagnostic runs at (q = 60, seed 60518)

Two runs, one per candidate arm, never counted. Seed 60518 is the formation failure of T3-R2 (class F in `01_diagnosis.md`).

| arm | A fired | trigger Delta / s / C | attempt 1 equals T3-R2's original (C-PAIR) | S3 (attempt 1) | F(u01600) of the frozen attempt | formation export (threshold 26.78) | sensitivity at u01600 | minimum sensitivity u100-u300 | passes | eta_3 | rmse_pos/e*(0) | Gmax_full | fail reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bwA | yes | 0.14468 / 41.92 / 0.0148 | yes | 0 | 54.08 | 225 | 1.904 | 0.000777 | yes | 0.0013 | 0.03198 | 0.0013 | - |
| bwAB | yes | 0.14468 / 41.92 / 0.0148 | yes | 0 | 50.16 | 225 | 1.757 | 0.000777 | yes | 0.00356 | 0.03548 | 0.00356 | - |

- **Attempt 1 equals T3-R2's original stage 3** (the 64 exports, the stage-3 rows and the checks) in both arms (C-PAIR with the original run; the planned attempt-1 init digest equals the digest in the original manifest).
- **A fired in both arms**: the development `Delta` at local 1600 is 0.1447, far above 0.005.
- **Attempt 2 forms and passes in both arms.** The formation measure F(u) of the frozen attempt at `u01600` is 54.1 (`bwA`) and 50.2 (`bwAB`), against the formation threshold 26.78; the formation export is `u00225` in both; the gap-sensitivity at `u01600` is 1.9 and 1.8 (6.8e-10 in the failed original, `01_diagnosis.md` sections 3 and 8). The final gates pass in both (`eta_3` 0.0013 and 0.0036, `rmse_pos` 0.032 and 0.035, `Gmax_full` 0.0013 and 0.0036).

## 10. The cost

Fixed in advance (`04_preregistration.md` section 4): 3,800 updates, 1,945,600 policy rows, 76,000 minibatch steps and 152 development checks for a run that does not fire; 5,400 updates, 2,764,800 policy rows, 108,000 minibatch steps and 216 checks for a run that fires (a factor 1.421).

| arm | runs | mean updates | mean policy rows | mean minibatch steps | mean dev checks | total wall (s) | mean wall (s) | median wall (s) | mean verifier (s) | ratio updates | ratio policy rows | ratio wall (total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bw | 40 | 3800 | 1.95e+06 | 7.6e+04 | 152 | 2.198e+04 | 549.6 | 555.5 | 6.087 | 1 | 1 | 1 |
| bwA | 40 | 3880 | 1.99e+06 | 7.76e+04 | 155.2 | 2.377e+04 | 594.2 | 582.7 | 6.661 | 1.021 | 1.021 | 1.081 |
| bwAB | 40 | 3920 | 2.01e+06 | 7.84e+04 | 156.8 | 2.64e+04 | 659.9 | 646.8 | 6.692 | 1.032 | 1.032 | 1.201 |

The counts are as planned: `bwA` fired on 2 of 40 runs, so its mean cost is 1.021 times `bw`'s in updates and policy rows (pre-registered expectation 1.037 at the recorded fire rate of 7 of 80; the expected rate 0.0875 lies inside the exact interval [0.006, 0.169] of the observed 2 of 40 = 0.05); `bwAB` fired on 3 of 40, 1.032 times.

**Measured wall time.** The totals above give 1.081 (`bwA`) and 1.201 (`bwAB`) times `bw`'s. They are not the cost of the scheme alone: the arms ran in one concurrent wave of 40 workers in the order `bwAB`, `bwA`, `bw` (the control last, partly in the draining phase), and the 38 seeds where `bwA` did not fire are bit-identical computations to `bw`'s, yet took 1.065 times as long.

| comparison | seeds | mean wall a (s) | mean wall b (s) | ratio a / b |
|---|---|---|---|---|
| `bwA` against `bw`, the seeds where `bwA` did not fire (bit-identical computations) | 38 | 582.5 | 546.8 | 1.065 |
| `bwA` against `bw`, the seeds where `bwA` fired (stage 3 trained twice) | 2 | 817.8 | 602.2 | 1.358 |
| `bwAB` against `bw`, the seeds where neither `bwA` nor `bwAB` fired | 36 | 635.5 | 544.1 | 1.168 |
| `bwAB` against `bwA`, the seeds where neither fired | 36 | 635.5 | 582 | 1.092 |

Read against that baseline: the two fired `bwA` runs took 818 s on average against 602 s for the same seeds in `bw`, 216 s more (36 % of a `bw` run, in line with the stage-3 phase of about 235 s, `04_preregistration.md` section 4); the policy-row ratio of `bwA` is 1.021, and 1.021 times the 1.065 measured on the bit-identical computations (run order and load) is 1.087, close to the measured 1.081. On the 36 seeds where neither candidate fired, `bwAB` took 1.092 times as long as `bwA` (1.168 times `bw`); the synthetic estimate for the censored update was 1.17 times the plain update (which is part of a run). The measured ratios are reported as measured; a clean ratio would need the arms run in an interleaved order, which this wave did not do.

## 11. The predictions

D7 of the prompt (the PI's, written before P1) was evaluated on the recorded runs in `01_diagnosis.md` section 10 (DP1a-b supported, DP1c-d partly or not, DP2 not supported, DP3 supported as a classification, DP4 no prediction of the outcome). The proposal's own predictions, which this wave can contradict (`04_preregistration.md` section 11), against the wave:

| prediction (`04_preregistration.md` section 11) | observed |
|---|---|
| A fires on about 9 % of the runs (7 of 80 in the recorded runs) | `bw` would fire in 2 of 40 = 0.050 [0.006, 0.169]; `bwA` fired in 2 of 40 = 0.050 [0.006, 0.169]; `bwAB` fired in 3 of 40 = 0.075 [0.016, 0.204] (exact 95 % intervals) |
| `bwA` and `bw` are bit-identical wherever A does not fire | 38 of 38 non-fired pairs are bit-identical over the whole run; the 2 fired pairs have the same stage 3 up to the check |
| the diagnostic run of seed 60518 fires A in both arms | fired in 2 of 2 diagnostic runs (`bwA`, `bwAB`); Delta at local 1600 = 0.14468; attempt 1 equals T3-R2's original: True; both second attempts pass: True |
| B is not expected to change the fire rate if the clamp regime is only a marker | fire rate of `bwAB` 3 of 40 against `bwA` 2 of 40; the seeds that fired are not the same ones (see the table of fired triggers); no test at these counts |

None of the proposal's predictions is contradicted. DP3 concerns specific runs of T3-R1 and T3-R2, which the wave does not contain; what the wave shows is that the same classification assigns five of the six failing runs to class (K) and one to class (V) (section 6). DP4 made no prediction of the development outcome: the outcome is the result of section 4.

## 12. What the wave shows and what it does not

- Both candidates reach the pre-registered development target and `bwA` is proposed; the evidence is 40 runs per arm on fresh seeds, and the improvement of `bwA` over `bw` is two seeds (exact McNemar p = 0.5). The wave is a development result and carries no confirmation: nothing is locked and the reserved seeds 80501-80540 were not trained.
- A repairs failures that show at the stage-3 check, and it does so by a restart; it repairs 2 of the 3 `bw` failures here. It does not touch a failure that the stage-3 check does not show (a stage-2 stretch: seed 70511 at q = 60, which fails in `bw` and `bwA` and passes in `bwAB`).
- The pre-registered indicators for B give 'mixed' (one of the three is met, (ii); (i) and (iii) are not), with no causal claim; `bwAB` reached the target with two failures, both at q = 50 (seed 70508, class (K), and seed 70520, class (V)); it passes 20 of 20 at q = 60.
- The checks pass except the dirty-tree flag, which is a recorded deviation caused by a write during the wave. It was checked on five clean-tree re-runs (two with a fired trigger), all bit-identical to their originals, and by the boundary argument (the flag follows the file's creation exactly); no `bw` run was re-run, since `bwA` without a trigger is bit-identical to `bw`.

## 13. Files

- Checks: `results/t3_r3/dev/launch_checks_absroot.json` (the record), `launch_checks.json` (first call), `launch_checks.out.txt`.
- Verification of the flag: `results/t3_r3/dev/verify_clean_tree/` (`boundary.csv`, `boundary.json`, `pairs_across_boundary.csv`, `files_outside_results_in_wave_window.txt`, `rerun_results_first_comparison.json`, `rerun_results_final.json`, `compare_*.txt`, and the five re-run directories, of which the manifests and statuses are tracked).
- Evaluation and analysis: `results/t3_r3/dev/eval/per_run.csv`, `results/t3_r3/diagnostic/eval/per_run.csv`, `results/t3_r3/dev/analysis/` (`headline.json`, `blind.json`, `per_run.csv`, `counts.csv`, `paired.csv`, `crosstab.csv`, `fired.csv`, `first_attempt.csv`, `causal_b.csv`, `b_pairs.csv`, `cost.csv`, `diagnostic.csv`, `last16.csv`, `last16_summary.csv`).
- The runs: `results/t3_r3/dev/q{q}/seed{s}/{arm}/` and `results/t3_r3/diagnostic/q60/seed60518/{arm}/`.
- Launch record: `results/t3_r3/launch_*.json`, `launch_commit.txt`, `launch_start.txt`, `launch_end.txt`, `launch_stdout.txt`, `infra_events.json` (no events).
- Tools: `tools/t3/t3r3_{launch,checks,pair,eval,analysis,blind,last16,post_launch,vct_boundary,vct_rerun}.*`; tests `tests/test_t3r3_*.py`.
