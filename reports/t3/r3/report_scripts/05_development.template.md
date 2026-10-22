# T3-R3, P2: the development wave (results)

Date: {{date}}. Branch `t3-r3`. **Generated** from `reports/t3/r3/report_scripts/05_development.template.md` by `build_report.py`; every table by `reports/t3/r3/report_scripts/results_tables.py` from `results/t3_r3/dev/launch_checks_absroot.json`, `results/t3_r3/dev/verify_clean_tree/` and `results/t3_r3/dev/analysis/` (written by `tools/t3/t3r3_analysis.py`, `t3r3_blind.py`, `t3r3_last16.py`). The binding texts are the prompt (`pi_record/28_t3_r3_prompt.md`), the PI's replies at G1 (`28b_g1_reply.md`) and on the dirty-tree flag (`28c_dirty_tree_reply.md`), and the pre-registration (`04_preregistration.md`, pushed before the launch). Nothing in this report was tuned after the launch. Nothing is locked, and no confirmation seed was trained.

**In one paragraph.** One wave of 122 runs (120 development runs, seeds 70501-70520 at q = 50 and q = 60, in three arms, and two diagnostic runs at q = 60, seed 60518) ran to completion with exit code 0 and no infrastructure event. `bw` (the control) passes 37 of 40 runs, `bwA` 39 of 40 and `bwAB` 38 of 40; both candidates reach the target (at least 38 of 40 and at least 18 of 20 at each q), so by the pre-registered order **`bwA` is proposed**. The 2-run difference between `bwA` and `bw` is two seeds at q = 50 where the trigger fired and the second attempt passed; every other pair of runs is bit-identical (exact McNemar p = 0.5). The causal reading of B is "mixed", with no causal claim. 82 of the 122 manifests record a dirty working tree because a report script was written into the worktree 71 s after the launch; the verification the PI asked for (section 1.3) found the clean-tree re-runs bit-identical to their originals.

## 1. The checks

### 1.1 What was checked

The checks are those of `04_preregistration.md` section 5, run by `tools/t3/t3r3_checks.py` on all 122 runs: exit code and files, manifest commit and clean tree, the global-RNG assertion, the config against the plan, the lock check, the trigger recomputed from the check file it read, budgets and schedules (a run that fired also has 5,400 updates, 2,764,800 policy rows, 108,000 minibatch steps and 216 development checks), C-INIT, C-REPLAY and C-PAIR. The record is `results/t3_r3/dev/launch_checks_absroot.json`; the first call (`launch_checks.json`, kept) used a relative `--root` (D-2 below), and its counts are shown in the last column.

{{table:res_checks}}

**Reading.** Every check passes except `clean_tree` (82 of 122 manifests record a dirty tree: D-1, verified harmless in section 1.3). The `pair_ok` row counts a `bwAB` run as passing by construction: `bwAB` differs from `bw` wherever B changes a stage-3 update, so its pair with `bw` is computed and reported but is not a check (the arm table shows how many `bwAB` runs are nevertheless identical to `bw`). Only `bwA` and the diagnostic runs are held to C-PAIR, and they all pass it. No infrastructure failure happened, no run was re-run or replaced, and the seeds 80501-80540 reserved for the later confirmation were never trained.

### 1.2 Deviations from the pre-registered procedure

- **D-1. 82 manifests record a dirty tree.** I wrote the untracked file `reports/t3/r3/report_scripts/results_tables.py` into the worktree at 02:21:15 UTC, 71 s after the launch at 02:20:02. The runner's definition of `clean_tree` (`run/run_v2_stagewise.git_state`: tracked changes or untracked files anywhere except `results/`) turned false for every run that started after that moment. The evidence is in section 1.3. The launcher had checked the tree before the launch, as pre-registered; the error is mine, a write during the wave.
- **D-2. The first post-launch check used a relative `--root`** and reported `config_equals_plan` as 0 of 122 (the plan hashes the output directory relative to the root's parent, which a relative root cannot do). The absolute-root call (`launch_checks_absroot.json`) is the record, and the first file is kept. It is the same defect as T3-R2's F10, now in `t3r3_checks.py`. Since commit `e6902c28` the four tools that take `--root` (`t3r3_launch`, `t3r3_eval`, `t3r3_checks`, `t3r3_analysis`) refuse a relative path (tested).
- **D-3. `results/t3_r3/post_launch.sh` ran the evaluation, the analysis and the blind recomputation after the checks had exited 1.** Nothing was committed or published in between, and those steps only compute from the data; the script is kept as it ran. `tools/t3/t3r3_post_launch.sh` (commit `e6902c28`) stops at the first nonzero exit and refuses a relative root (tested).
- **The `pair_ok` column** is labelled in the tables as binding for `bwA` and the diagnostic runs only; for `bwAB` it is not a check (the first report of the checks showed 40 `bwAB` failures of it).
- **Tools changed after the launch.** `git diff b565f0e8 HEAD -- tools run agents envs utils protocols` lists, besides new files, only the root check in the four tools above; the evaluation, the analysis and the blind recomputation ran with the tools as launched (`b565f0e8`), and the logic of those tools is unchanged by that commit.

### 1.3 The dirty-tree flag, verified (reply 28c)

All outputs are under `results/t3_r3/dev/verify_clean_tree/`.

**1. The boundary.** From the 122 manifests and `status.json` start times:

{{table:res_boundary}}

The recorded flag follows the creation of the file without exception. The manifests store only the boolean, not the entries of `git status`, so 'the only entry is that file' is established from the file system instead: the search for paths outside `results/` and `.git` modified in the wave window 02:20:00-02:55:47 UTC finds that file and (in the search at 03:10 UTC, before the verification moved it) its parent directory, nothing else (`files_outside_results_in_wave_window.txt`); the same `git status --porcelain -- . ':(exclude)results'` the runner uses lists only that file at verification time. Every manifest carries the commit `b565f0e8`, and `git diff HEAD` was empty. The runner opens no file under `reports/` (`run/run_ms_stagewise.py`, `run/ms_retry.py`, `run/run_v2_stagewise.py`, `run/v2_rollout.py`, `agents/`, `utils/` and `envs/` mention `reports/` only in one docstring).

**2. Pairs across the boundary.**

{{table:res_pairs}}

The count of `bwA`-`bw` pairs that are bit-identical while one run recorded a clean tree and the other a dirty one is therefore **0**: all 40 `bw` runs and the 40 development `bwA` runs are dirty-flagged, and the one clean `bwA` run is the diagnostic run, which has no `bw` partner. The pairs do not cross the boundary and carry no evidence on it; the re-runs below do.

**3. Clean-tree re-runs (verification only, never counted).** `reports/t3/r3/report_scripts/results_tables.py` and the saved reply `28c_dirty_tree_reply.md` were moved out of the worktree for the re-runs and restored afterwards; at the start HEAD was `b565f0e8` and `git status --porcelain` showed nothing outside `results/`. Each run was re-run from its own `run_config.json` into `verify_clean_tree/q{q}/seed{s}/{arm}/` (single-threaded, in tmux, `tools/t3/t3r3_vct_rerun.py`) and compared with its original bit for bit by `tools/t3/compare_runs.py bw`, by the set of files, by `attempt1/` and `retry.json` where present, and by the small JSON records. The selection follows the reply: every dirty-flagged `bwA` run whose trigger fired (two; there is no dirty-flagged `bwAB` run that fired, and the two diagnostic runs are clean-flagged already), and the lowest-numbered dirty-flagged `bwAB` run that did not fire at each q. **Only q = 60 has such a run: 38 of the 40 development `bwAB` runs (all 20 of q = 50 and the q = 60 seeds 70501-70518) are clean-flagged, and only q = 60 seeds 70519 and 70520 are dirty.** To cover the dirty-flagged non-fired case at q = 50 as well, I added the lowest-numbered dirty-flagged non-fired `bwA` run at each q (marked 'addition'; `bwA` without a trigger is bit-identical to `bw`).

{{table:res_rerun}}

**Result: all five are identical to their originals** (`compare_runs.py bw` exit code 0 for each: 162 npz files, the four CSVs without their wall-clock columns, the final networks, frozen stages and stream states; the same set of files; for the two fired runs also the 64 exports, the stage-3 rows and checks, the bin maps and the state of `attempt1/`, and `retry.json` apart from wall-clock fields; the gates, rule log, drift test and summary JSON apart from the `clean_tree` flag, which is the thing under test and reads `false` in the original and `true` in the re-run). The re-run manifests record the commit `b565f0e8` and a clean tree. **The flag is recorded as a deviation (D-1); the 122 runs remain the counted records exactly as they are.**

*A note on the comparison itself.* The first pass of the comparison (`rerun_results_first_comparison.json`, kept) reported 'not ok' for all five, only because my JSON comparison treated `NaN != NaN` (the `R_tail` of stages 1 and 2 is NaN in both runs). With NaN equal to NaN and the `clean_tree` flag excluded, the comparison (`rerun_results_final.json`, `--compare-only`, no retraining) passes; `compare_runs.py` itself returned 0 in both passes. The committed driver `tools/t3/t3r3_vct_rerun.py` is the version of that second pass; the training itself ran with the version before the NaN rule and the `--compare-only` option.

### 1.4 The full suite

`pytest tests` as 12 parallel shards at the commit `{{suite_commit}}` (`results/t3_r3/tests_p3/`): {{suite_result}} The two failures are the known ones recorded at G1: `tests/test_registry_canonicalization.py::test_registry_canonicalization` and `tests/test_ms_r4_launcher.py::test_the_existing_launcher_is_unchanged` (it has failed since the T3-1.0 lock changed `protocols/`). Nothing else fails.

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

{{table:res_perseed_q50}}

**q = 60**

{{table:res_perseed_q60}}

## 4. The target outcome per arm and the selection

An arm reaches the target iff at least 38 of its 40 runs pass and at least 18 of 20 at each q. Only `bwA` and `bwAB` are eligible; in that order the first arm that reaches the target is proposed. Exact (Clopper-Pearson) 95 % intervals.

{{table:res_counts}}

{{table:res_target}}

{{table:res_blind}}

**First-attempt outcome per arm (G1 reply 2.1).** The pass count when every fired run counts as a failure of its first attempt. For `bwA` it must equal `bw`'s, seed by seed, under the same convention (a `bw` run counts as fired when the trigger rule applied to its own check fires): it does, with no mismatch, so the run does not stop.

{{table:res_firstattempt}}

**What the selection says, and what it does not.** By the pre-registered order `bwA` is proposed: both arms reach the target and `bwA` comes first. The PI may lock `bwAB` instead only with a stated reason taken from the items of section 2.2 of the G1 reply; section 8 reports those items. The development evidence is 40 runs per arm: the exact interval of the pass rate of `bwA` is [0.868, 0.999] and of `bwAB` [0.831, 0.994], and the operating characteristics in `04_preregistration.md` section 10 show that a rate of 0.95 reaches this target with probability 0.68 and a rate of 0.90 with probability 0.22.

## 5. The paired comparison with `bw`

Paired by seed (same q and seed). `Gmax_full` and `eta_3` in units of the prize gap; intervals from the paired bootstrap (10,000 resamples, seed 20261021).

{{table:res_paired}}

**Reading.** `bwA` and `bw` are bit-identical on 38 of 40 seeds (the table shows exactly zero differences there) and differ on the 2 seeds where A fired, where the second attempt passes and `bw` fails: 2 seeds better, none worse, exact McNemar p = 0.5. `bwAB` differs from `bw` on all 40 seeds (no `bwAB` run is identical to its `bw` run) because B changes the stage-3 update itself; its paired differences of `eta_3` and `Gmax_full` have intervals that contain 0 at both q, its pass / fail cross-table against `bw` is 2 better, 1 worse (p = 1) and against `bwA` 1 better, 2 worse (p = 1).

## 6. The failures of every arm, classified as in D2

The class is assigned by the D2 definitions: (F) formation failure; (K) the run is in the stage-3 clamp regime (S3 of the frozen attempt above 0.2) and its failing quantity is a stage-3 in-band bias or a stage-2 near-zero stretch; (V) the tent forms, the run is not in the regime, and the failing value sits at the top of its own range over the last 16 exports of the phase. 'Last 16 exports' is `tools/t3/t3r3_last16.py` (the exports `u01225` ... `u01600` evaluated on the final tier; `u01600` reproduces each run's evaluated value to 1e-9).

{{table:res_failures}}

There are four distinct failing seeds and six failing runs. **Five of the six are class (K)**: `bw` seeds 70508 and 70516 at q = 50 (in-band bias, S3 0.78 and 0.76), `bw` and `bwA` seed 70511 at q = 60 (a stage-2 near-zero stretch of 28 nodes, `Gmax_full` 0.046, S3 0.52), `bwAB` seed 70508 (S3 0.70). **One is class (V)**: `bwAB` seed 70520 at q = 50 (S3 0.0098; `rmse_pos` 0.0543 against the limit 0.05 is the largest of its last 16 exports, 1 of 16 above the limit). No failure is class (F), (E) or (U). The two `bw` failures at q = 50 are the two that A repairs (section 7). The q = 60 stretch failure (seed 70511) is invisible to the trigger: its stage-3 development `Delta` at local 1600 is 0.0040, below 0.005, and a stage-2 stretch is not seen by the stage-3 check; A cannot reach it. In `bwAB` the same seed passes, but its stage-3 first attempt differs from `bw`'s, so that is not evidence of an effect of B on the stretch.

## 7. The triggers that fired

{{table:res_fired}}

Five development runs fired (`bwA`: seeds 70508 and 70516 at q = 50; `bwAB`: seeds 70510, 70514 and 70516 at q = 50) and the two diagnostic runs. **Every fired run passes on its second attempt** (5 of 5 development, 2 of 2 diagnostic). The two `bwA` fires are the two `bw` runs that the trigger rule would fire on and that fail in `bw` (the `bw` runs that would fire and nevertheless pass: 0). None fired at q = 60 outside the diagnostic runs. Two notes: (i) the second attempt of `bwA` seed 70508 is itself in the clamp regime (S3 0.756) and passes with `eta_3` 0.00441 against the limit 0.005, so a restart does not always leave the regime; (ii) the first attempts of `bwAB` differ from `bw`'s, so `bwAB` fired on seeds 70510 and 70514, which `bw` did not.

## 8. S3 and the stage-2 stretches; the causal reading of B

{{table:res_s3}}

The items of section 2.2 of the G1 reply, paired by seed on first attempts (for a fired run, `attempt1/`):

{{table:res_causal}}

**Pre-registered reading** (`04_preregistration.md` section 7): (i) the number of first-attempt runs with S3 > 0.2 is smaller in `bwAB` than in `bw` and in `bwA`, with the exact McNemar p < 0.05 on the pooled pairs; (ii) the number of runs with a stage-2 stretch is smaller in `bwAB` than in `bw` and in `bwA`; (iii) on the seeds where the trigger fired in neither arm, the paired bootstrap interval of the difference of `eta_3` (`bwAB` - `bw`, pooled) lies entirely below 0. All three: 'more than a marker'; none: 'a marker'; otherwise 'mixed', with no causal claim.

- (i) **not met**: 13 first-attempt runs with S3 > 0.2 in `bwAB` against 12 in `bw` and in `bwA`; the exact McNemar test has 1 discordant pair (the regime in `bwAB` only), p = 1.
- (ii) **met**: 1 run with a stage-2 stretch in `bwAB` (2 nodes) against 3 in `bw` and in `bwA` (52 nodes). Descriptively, an exact Fisher test of 3 of 40 against 1 of 40 gives p = 0.62; the counts are small.
- (iii) **not met**: the interval is [-0.00008, 0.00064] on 36 seeds (pooled), which contains 0; the interval of `peak_T_signed` is [-0.0137, 0.0090].

**The reading is 'mixed', and no causal claim is made.** The other items are also without a visible effect of B: the share of out-of-band nodes at the actor's mean clamp at `u01600` has median 0 in every arm (runs with at least one such tail node: 9 in `bw` and `bwA`, 7 in `bwAB`), the median minimum alpha is 0.272 (`bw`) and 0.248 (`bwAB`), the median development `Delta` at local 1600 is 0.0015 (`bw`) and 0.0018 (`bwAB`), and the fire rate is 2 of 40 for `bw` (would fire), 2 of 40 for `bwA` and 3 of 40 for `bwAB`. The frozen-attempt count of runs with S3 > 0.2 is 12 (`bw`), 11 (`bwA`) and 10 (`bwAB`); it differs from the first-attempt count only through the restarts: of the two `bwA` restarts one leaves the regime (seed 70516) and one stays in it (seed 70508), and all three `bwAB` restarts leave it.

## 9. The diagnostic runs at (q = 60, seed 60518)

Two runs, one per candidate arm, never counted. Seed 60518 is the formation failure of T3-R2 (class F in `01_diagnosis.md`).

{{table:res_diag}}

- **Attempt 1 equals T3-R2's original stage 3** (the 64 exports, the stage-3 rows and the checks) in both arms (C-PAIR with the original run; the planned attempt-1 init digest equals the digest in the original manifest).
- **A fired in both arms**: the development `Delta` at local 1600 is 0.1447, far above 0.005.
- **Attempt 2 forms and passes in both arms.** The formation measure F(u) of the frozen attempt at `u01600` is 54.1 (`bwA`) and 50.2 (`bwAB`), against the formation threshold 26.78; the formation export is `u00225` in both; the gap-sensitivity at `u01600` is 1.9 and 1.8 (7e-10 in the failed original, `01_diagnosis.md` section 10). The final gates pass in both (`eta_3` 0.0013 and 0.0036, `rmse_pos` 0.032 and 0.035, `Gmax_full` 0.0013 and 0.0036).

## 10. The cost

Fixed in advance (`04_preregistration.md` section 4): 3,800 updates, 1,945,600 policy rows, 76,000 minibatch steps and 152 development checks for a run that does not fire; 5,400 updates, 2,764,800 policy rows, 108,000 minibatch steps and 216 checks for a run that fires (a factor 1.421).

{{table:res_cost}}

The counts are as planned: `bwA` fired on 2 of 40 runs, so its mean cost is 1.021 times `bw`'s in updates and policy rows (pre-registered expectation 1.037 at the recorded fire rate of 7 of 80; the expected rate 0.0875 lies inside the exact interval [0.006, 0.169] of the observed 2 of 40 = 0.05); `bwAB` fired on 3 of 40, 1.032 times.

**Measured wall time.** The totals above give 1.081 (`bwA`) and 1.201 (`bwAB`) times `bw`'s. They are not the cost of the scheme alone: the arms ran in one concurrent wave of 40 workers in the order `bwAB`, `bwA`, `bw` (the control last, partly in the draining phase), and the 38 seeds where `bwA` did not fire are bit-identical computations to `bw`'s, yet took 1.065 times as long.

{{table:res_cost_identical}}

Read against that baseline: the two fired `bwA` runs took 818 s on average against 602 s for the same seeds in `bw`, 216 s more (36 % of a `bw` run, in line with the stage-3 phase of about 235 s, `04_preregistration.md` section 4); the policy-row ratio of `bwA` is 1.021, and 1.021 times the environmental 1.065 is 1.087, close to the measured 1.081. On the 36 seeds where neither candidate fired, `bwAB` took 1.092 times as long as `bwA` (1.168 times `bw`); the synthetic estimate for the censored update was 1.17 times the plain update (which is part of a run). The measured ratios are reported as measured; a clean ratio would need the arms run in an interleaved order, which this wave did not do.

## 11. The predictions

D7 of the prompt (the PI's, written before P1) was evaluated on the recorded runs in `01_diagnosis.md` section 10 (DP1a-b supported, DP1c-d partly or not, DP2 not supported, DP3 supported as a classification, DP4 no prediction of the outcome). The proposal's own predictions, which this wave can contradict (`04_preregistration.md` section 11), against the wave:

{{table:res_predictions}}

None of the proposal's predictions is contradicted. DP3 as a classification holds in the wave as well: five of the six failing runs are class (K), one is class (V), none is (F), (E) or (U) (section 6). DP4 made no prediction of the development outcome: the outcome is the result of section 4.

## 12. What the wave shows and what it does not

- Both candidates reach the pre-registered development target and `bwA` is proposed; the evidence is 40 runs per arm on fresh seeds, and the improvement of `bwA` over `bw` is two seeds (exact McNemar p = 0.5). The wave is a development result and carries no confirmation: nothing is locked and the reserved seeds 80501-80540 were not trained.
- A repairs failures that show at the stage-3 check, and it does so by a restart; it repairs 2 of the 3 `bw` failures here. It does not touch a failure that the stage-3 check does not show (a stage-2 stretch: seed 70511 at q = 60, which fails in `bw` and `bwA` and passes in `bwAB`).
- B is not distinguishable from a marker by the pre-registered indicators ('mixed'); `bwAB` reached the target with two failures, both at q = 50 (seed 70508, class (K), and seed 70520, class (V)); it passes 20 of 20 at q = 60.
- The checks pass except the dirty-tree flag, which is a recorded deviation caused by a write during the wave and verified to leave the results unchanged on the five re-runs chosen.

## 13. Files

- Checks: `results/t3_r3/dev/launch_checks_absroot.json` (the record), `launch_checks.json` (first call), `launch_checks.out.txt`.
- Verification of the flag: `results/t3_r3/dev/verify_clean_tree/` (`boundary.csv`, `boundary.json`, `pairs_across_boundary.csv`, `files_outside_results_in_wave_window.txt`, `rerun_results_first_comparison.json`, `rerun_results_final.json`, `compare_*.txt`, and the five re-run directories, of which the manifests and statuses are tracked).
- Evaluation and analysis: `results/t3_r3/dev/eval/per_run.csv`, `results/t3_r3/diagnostic/eval/per_run.csv`, `results/t3_r3/dev/analysis/` (`headline.json`, `blind.json`, `per_run.csv`, `counts.csv`, `paired.csv`, `crosstab.csv`, `fired.csv`, `first_attempt.csv`, `causal_b.csv`, `b_pairs.csv`, `cost.csv`, `diagnostic.csv`, `last16.csv`, `last16_summary.csv`).
- The runs: `results/t3_r3/dev/q{q}/seed{s}/{arm}/` and `results/t3_r3/diagnostic/q60/seed60518/{arm}/`.
- Launch record: `results/t3_r3/launch_*.json`, `launch_commit.txt`, `launch_start.txt`, `launch_end.txt`, `launch_stdout.txt`, `infra_events.json` (no events).
- Tools: `tools/t3/t3r3_{launch,checks,pair,eval,analysis,blind,last16,post_launch,vct_boundary,vct_rerun}.*`; tests `tests/test_t3r3_*.py`.
