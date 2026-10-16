# T3-R1 checks (P1)

Date: 2026-10-16. Commands and verbatim outputs; machine-readable copies in `results/t3_r1/preflight/` and `results/t3_r1/smoke_checks.json`. Each check below is a stop-and-report if it fails; none failed.

## 1. C-DIFF: nothing frozen changes

`git diff --cached --stat db7577ea -- agents envs utils run protocols tools/v2 tools/ms` with the code commit staged (`results/t3_r1/preflight/c_diff_staged.txt`):

```
$ git diff --cached --stat db7577ea -- agents envs utils run protocols tools/v2 tools/ms
 run/run_joint.py | 699 +++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 699 insertions(+)
```

After the commit, `git diff --stat db7577ea <code commit> -- agents envs utils run protocols tools/v2 tools/ms` (`results/t3_r1/preflight/c_diff_commit.txt`):

```
$ git diff --stat db7577ea da4b1115f518555125c4fab0a99c20e12243634a -- agents envs utils run protocols tools/v2 tools/ms
 run/run_joint.py | 699 +++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 699 insertions(+)
```

Only the new `run/run_joint.py` is listed. The same command at the final code commit `5535d6104899526ae1790d50806cc66392549994` (`results/t3_r1/preflight/c_diff_final.txt`) gives the same single line, `A run/run_joint.py`; between the two commits the review fixes (section 6) touch `tools/t3/` and `tests/` and, in `run/run_joint.py`, only the placement of the development check; the follow-up `5535d610` touches `tools/t3/` and `tests/` only.

## 2. C-INIT (plan): initial weights depend on (q, seed) only

`python tools/t3/t3_launch_checks.py c-init-plan --out results/t3_r1/preflight/c_init_plan.csv` constructs every planned run without training (`MSRun` for `bw`, `JointRun` for `jt`, `jt_es`, `jt2`; plus `MSRun` at T = 2 built as `build_config(proto, q, seed, "MS_base", out_dir, params=reports/ms/r1/prereg_parameters.json)` with `init_digest` true) and records the SHA-256 of the initial actor and critic weights (`results/t3_r1/preflight/c_init_plan.out.txt`, 210 rows):

```
{"rows": 210, "t3_groups": 42, "t3_equal": 42, "t2_groups": 42, "t2_equal": 42}
```

- 42 of 42 (q, seed) groups (smoke 2 + wave `t3` 40) have equal digests across `bw`, `jt` and `jt_es`; 42 of 42 groups (smoke 2 + wave `t2j` 40) have `jt2` equal to `MSRun` at T = 2. In the smoke groups all five digests (`bw`, `jt`, `jt_es`, `jt2` and the T = 2 reference) are one value, so the initial weights are also equal across T. The 42 groups have 42 distinct digests.

## 3. C-T2R: the evaluator reproduces the 40 recorded v2.0 runs

`python tools/t3/t3_launch_checks.py c-t2r --v20-root <D9 root> --out-dir results/t3_r1/preflight` re-evaluates the 40 `v20` runs (`u01600` export at t = 2, `u02200` export at t = 1) with `tools/t3/t3_eval.py` and compares with each run's `gates.json` (`results/t3_r1/preflight/c_t2r.out.txt`, `c_t2r.csv`, `c_t2r_per_run.csv`):

```
{
 "runs": 40,
 "identical": 40,
 "per_field_identical": {
  "eta_final": 40,
  "eta_dev": 40,
  "rmse": 40,
  "tail": 40,
  "gmax_final": 40,
  "gmax_dev": 40,
  "s1": 40,
  "G-A": 40,
  "G-F": 40,
  "G-N": 40,
  "G-S": 40
 }
}
```

All seven gate values (`eta_final`, `eta_dev`, `rmse`, `tail`, `gmax_final`, `gmax_dev`, `s1`) and the four verdicts (G-A, G-F, G-N, G-S) are identical bit for bit in 40 of 40 runs, including seed 30510 at q = 50 (the one run that failed G-A). The evaluator, the loaders and the band-free part of the metrics are therefore the v2.0 pipeline's.

## 4. Tests

`/home/fjiang4/tournament_experiment/.venv/bin/python -m pytest tests/test_t3_*.py -p no:cacheprovider -q` (single thread, at the final code commit `5535d610`): **126 passed in 114 s** (`results/t3_r1/preflight/tests/t3_tests_at_head.out.txt`). New files: `tests/test_t3_joint_runner.py` (52 tests: config validation and budgets, the draw order of the starts, the LR schedule, the visitation, the development check, the bit-for-bit equality of a stage-3-only joint run with `MSRun`'s stage-3 phase, the global-RNG hardening, exit codes, determinism), `tests/test_t3_metrics_eval.py` (24: band split, terminal recovery against `evaluate()` at T = 2, the gates, the three loaders, best of run, replay), `tests/test_t3_launcher.py` (33: waves, seed guards, plan, config hashes and differences, the post-launch checks, C-INIT, C-T2R), `tests/test_t3_analysis.py` (14: the criterion and secondary tables, the bootstrap against an independent implementation, the reading rule, the figures' content, the blind tool against the analysis), `tests/test_t3_track.py` (3: the tracking rules, the run comparison), and the helpers `tests/t3_helpers.py`. Tests that run training use reduced configurations (a few dozen episodes); no test writes outside a temporary directory.

**The whole suite** (`tests/test_*.py`, 12 parallel shards of one thread each, over the 56 test files that existed when the shards were started; the T3 files were re-run on the final HEAD above; `results/t3_r1/preflight/tests/full_suite_shards.txt`): **1,644 passed, 2 failed, 6 skipped, 2 xfailed** (the 2 xfailed are `strict` documentation tests of `tests/test_v2_verifier.py`; the 6 skips test for MS-R2 / MS-R1 pilot results that are not in this clone). The two failures are outside this round (`results/t3_r1/preflight/tests/two_non_t3_failures.txt`); I looked at both before calling them unrelated, and the second reproduces on the base commit:

1. `tests/test_ms_analysis.py::test_cli_end_to_end_complete_world_writes_every_output`: `assert np.float64(inf) == '9e2443526928'`. The analysis writes a 12-hex-digit `roots_id` (the first 12 hex digits of a SHA-256 of the resolved root paths, which lie in the pytest temporary directory, whose numbered name differs from run to run) and the test reads the CSV back with `pandas.read_csv`, which parses a digits-`e`-digits id such as `9e2443526928` as a float (`9e2443526928` = inf). It is a latent flake of the existing test, decided by the temporary-directory name (about 1 id in a few hundred is parsed as a number), not by anything of this round: neither the test nor `tools/ms/` differs from `origin/ms-r4`, and the same test alone at HEAD afterwards (new temporary directory) **passed (52.8 s)**. Not fixed here (the prompt allows no change in `tools/ms/` and none of `tests/` outside this round's files).
2. `tests/test_registry_canonicalization.py::test_registry_canonicalization`: `expected 15 Set-2 gradient runs, got 0` (`:115`). The test counts the Set-2 (`wh8_wl4`) gradient runs that the paper generator's run registry finds under `results/two_players/`; it finds none. The test fails with the identical assertion in a `git archive` of `origin/ms-r4` (`db7577ea`, no file of this round), so it is independent of this round; I did not investigate the cause (the registry's selection of the Set-2 gradient rows), because the prompt allows no change outside this round's files.

## 5. The smoke wave (cost and correctness only; never analysed)

The wave `smoke` (seed 50500, both q, the four arms at full budget, 8 runs) was launched with `python tools/t3/t3_launch.py --wave smoke --workers 8 --code-commit da4b1115f518555125c4fab0a99c20e12243634a` in tmux on HEAD `91a9f514` (no tracked file changed; six result files untracked, listed in the launch record's `status_porcelain` and not counted by the manifests' `clean_tree` flag, which ignores `results/`) at 22:41 UTC on 2026-10-16, under the load of other users (load average 12.3 / 11.9 / 11.1 at the start and 15.7 / 16.1 / 13.6 at the end of the launch, `results/t3_r1/smoke/launch_20261016_224126.json`): **8 of 8 runs exit 0** (`results/t3_r1/smoke/launch_20261016_224126.json`, `results/t3_r1/smoke_launch_stdout.txt`). The outputs serve cost and correctness only; they are never analysed or reported as results.

**Cost** (`results/t3_r1/smoke_cost.csv`; `report_scripts/smoke_cost.py`, `prereg_tables.py`). Wall and CPU seconds per run and per category; for `bw` per phase below the table. The MS runner records no peak RSS: it was measured on the scratch reruns, 934,632 kB (q = 50) and 933,112 kB (q = 60); the joint arms record theirs.

| arm | q | updates | policy rows | minibatch steps | wall (s) | CPU (s) | train_rollout (s) | train_update (s) | dev_verifier (s) | final_eval (s) | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bw` | 50 | 3,800 | 1,945,600 | 76,000 | 387 | 353 | 7.8 | 340.2 | 4.1 | nan | nan |
| `jt` | 50 | 1,267 | 1,946,112 | 76,020 | 336 | 336 | 7.3 | 326.5 | 1.4 | 0.2 | 499 |
| `jt_es` | 50 | 1,754 | 1,945,186 | 87,700 | 387 | 385 | 9.5 | 372.7 | 1.9 | 0.2 | 499 |
| `jt2` | 50 | 1,100 | 1,126,400 | 44,000 | 202 | 201 | 4.6 | 195.9 | 0.4 | 0.0 | 493 |
| `bw` | 60 | 3,800 | 1,945,600 | 76,000 | 399 | 355 | 7.8 | 341.9 | 4.3 | nan | nan |
| `jt` | 60 | 1,267 | 1,946,112 | 76,020 | 338 | 337 | 7.4 | 327.7 | 1.5 | 0.2 | 499 |
| `jt_es` | 60 | 1,754 | 1,945,186 | 87,700 | 386 | 385 | 9.5 | 372.6 | 2.1 | 0.2 | 502 |
| `jt2` | 60 | 1,100 | 1,126,400 | 44,000 | 204 | 204 | 4.6 | 198.7 | 0.4 | 0.0 | 492 |

`bw` q = 50 per phase (wall s): stage 3 154, stage 2 147, stage 1 53; CPU s: 153 / 147 / 53

`bw` q = 60 per phase (wall s): stage 3 156, stage 2 151, stage 1 52; CPU s: 156 / 147 / 52

**Estimate for the launch** (longest-processing-time scheduling of the smoke wall times of each arm and q onto 40 workers): the 160 runs of waves `t3` and `t2j` take about **22 minutes** in one launch (wave `t3` alone 19 minutes, wave `t2j` alone 3 minutes); the sum of the run wall times is 14.7 hours. The memory is 40 x 0.5-0.9 GB; the disk 74 MB per 8 runs, about 1.5 GB for 160 runs (about 200 MB of it tracked).

**Post-launch checks on the smoke runs** (`python tools/t3/t3_launch_checks.py post-launch --root results/t3_r1 --waves smoke --commit 91a9f514... --plan-digests results/t3_r1/preflight/c_init_plan.csv`, `results/t3_r1/smoke_checks.json`; 5 seconds): every verdict passes in 8 of 8 runs, including **C-REPLAY** (the evaluation of `t3_eval.py` reproduces each run's own final-evaluation arrays, `freeze_stage1_{final,development}.npz` for `bw` and `final_eval_{final,development}.npz` for the joint arms, bit for bit) and the budgets of D5 exactly (updates, policy rows by stage, minibatch steps, exports 152 / 50 / 70 / 44, checks 64 / 64 / 24 and 50 / 70 / 44, the LR at every update, the refreshes).

| check | pass | fail | not evaluated |
|---|---|---|---|
| exit_ok | 8 | 0 | 0 |
| files_complete | 8 | 0 | 0 |
| manifest_commit_ok | 8 | 0 | 0 |
| clean_tree | 8 | 0 | 0 |
| global_rng_ok | 8 | 0 | 0 |
| config_equals_plan | 8 | 0 | 0 |
| config_production_ok | 8 | 0 | 0 |
| budget_ok | 8 | 0 | 0 |
| lr_ok | 8 | 0 | 0 |
| refresh_ok | 8 | 0 | 0 |
| exports_ok | 8 | 0 | 0 |
| checks_ok | 8 | 0 | 0 |
| starts_ok | 8 | 0 | 0 |
| replay_ok | 8 | 0 | 0 |

C-INIT within (q, seed): {'groups': 2, 'equal': 2}; equals the plan: {'runs': 8, 'equal': 8}; start-share flags: {'cells': 262, 'flags': 1, 'expected_under_null': 0.7074}

**Full-scale determinism** (same config, a second run in a scratch root, `tools/t3/compare_runs.py`): the two `jt2` smoke runs (47 NPZ files each: 44 exports, 2 final evaluations, `visitation.npz`), the CSVs without their wall-clock columns, the final networks and the stream states are identical (`results/t3_r1/preflight/rerun_determinism_jt2.txt`); the two `bw` smoke runs (162 NPZ files each) likewise (`rerun_determinism_bw.txt`); with the final code (`run/run_joint.py` of `d993f612`) a full-scale rerun of `jt`, `jt_es` and `jt2` at q = 50 (53, 73 and 47 NPZ files) is bit-identical to the smoke runs made with the first code commit (`rerun_determinism_final_code_joint.txt`), so the changes between the two commits moved no number.

## 6. Independent review

The reviewer's prompt and both of its reports are stored verbatim in `04_review_reports.md`; the counts and timings quoted here (tool calls, minutes, passed tests, decoding errors, wall-clock seconds of the check) are the reviewer's own.

A read-only agent that did not write the code (a general-purpose subagent, Opus, 78 tool calls, about 22 minutes) reviewed `git diff db7577ea..da4b1115` against the prompt, covering conformance with D2-D7, the JT runner against `MSRun`/`run_pipeline` in everything D2 calls shared, the budget arithmetic of D5 and C-INIT, the closed-form-free training, the band split / terminal recovery / loaders / analysis definitions, the launcher and the launch checks, and the tests. While it worked, the working tree was edited (uncommitted launcher changes); it reviewed the committed files (`git show da4b1115:<path>`) and ran `pytest tests/test_t3_*.py` on a `git archive` extract: **111 passed**. **Findings: 2 MAJOR (one procedural, one descriptive), 5 MINOR, 1 INFO group.** None would change a trained policy, a recorded number or a pre-registered verdict when the inputs are correct. All are fixed in `d993f612` (tools and tests only; **no file of `agents/`, `envs/`, `utils/`, `protocols/`, `tools/v2/`, `tools/ms/` or an existing file of `run/` changed; the only change to `run/run_joint.py` is that the development check now runs after the per-update row, which moves no random number and no weight, see section 5**).

| # | severity | finding | disposition |
|---|---|---|---|
| 1 | MAJOR (procedural) | the committed launcher takes one wave per call, so "one launch: waves `t3` and `t2j`" (section 3.1) needed two pools or two launches | fixed (`d993f612`): `--wave t3 t2j` is one launch (one pool of at most 40 workers, one record `launch_t3_t2j_<stamp>.json`); explicit `--seeds` / `--arms` are refused with several waves (an `--arms` filter was silently ignored in the uncommitted version); test `test_two_waves_are_one_launch_with_one_record_and_one_plan`, `test_several_waves_refuse_explicit_arms_...` |
| 2 | MAJOR (descriptive) | the figures covered part of section 3.3 / D6 "Curves": no closed form or g1 in the effort figure though it was documented, q = 50 only, T = 3 arms only, the paired figure showed Gmax_full only, the curves lacked stages 1-2 and `jt2`, the visitation lacked `jt2`; the figure test only checked the file size | fixed: `figure_effort` (both rows T = 3 / T = 2, all stages, the closed form at the terminal stage, g1 at stage 1 of the T = 2 row), `figure_gains` (T = 3 and T = 2 arms, median and maximum, band edges), `figure_paired` (Gmax_full, EXP_root, the in / out-of-band maxima of G_t and Delta_t, four pairs, both q), `figure_curves` (Gmax_full, EXP_root, eta_t and G_max_t of every stage, `bw`'s phase Delta, a T = 2 figure for `jt2`), `figure_visitation` (T = 3 stages 2-3 with `bw`'s design, T = 2 stage 2); one file per q; `test_figures_have_the_pre_registered_content` inspects titles, labels, the closed-form curve, the band-edge lines and the panel counts |
| 3 | MINOR | a C-INIT failure did not change the post-launch exit status although section 3.2 calls every deterministic check a stop-and-report | fixed: `post_launch_ok` (verdicts, C-INIT within (q, seed), C-INIT against the plan; the z flags stay descriptive); test `test_the_deterministic_checks_decide_the_exit_status` |
| 4 | MINOR | a missing v20 reference did not stop the evaluation, H1-T2 came out "not supported" with 0 pairs (D9: a missing file is a stop-and-report) | fixed: `t3_eval.py` requires `--v20-root` with wave `t2j` and stops if any of the 200 reference files (status, gates, config, the two exports of each of the 40 runs) is missing; test `test_a_missing_reference_file_stops_the_evaluation` |
| 5 | MINOR | two error paths crashed instead of reporting the run: `bw_gate_table` for a `bw` run that exited 0 but whose evaluation raised; `read_checks` when every development evaluation of a joint run failed | fixed: the diagnosis reads "no result (status ..., exit code ..., evaluation error: ...)"; `read_checks` uses `.get`; tests `test_a_run_with_a_clean_exit_but_no_metrics_is_reported_not_a_crash`, `test_a_check_file_without_any_successful_evaluation_has_no_valid_candidate` |
| 6 | MINOR | D5's check of the v2.0 step count was done by no tool (the cost table computed it, nothing compared it with 56,000) | fixed: `budget_check` in the analysis (`budget_check.csv`: updates, policy rows by stage and in total, minibatch steps against `expected_budget`, `headline.json["budget_check_all_equal"]`); the 40 recorded runs were verified now (`results/t3_r1/preflight/v20_budget_check.txt`: 2200 updates, 56,000 steps, 1,126,400 policy rows in all 40) |
| 7 | MINOR | tests that cannot fail: `manifest_commit_ok` with `commit=None`, `clean_tree in (True, False)`; the decoded-observation test checked the array length and a total that holds by construction; the bw allow-list of q-differences included `pipeline` and `derived` | fixed: the commit test passes the manifest's commit (True) and a wrong one (False) and compares the flag with the manifest; `test_visitation_sources_equal_the_bins_of_the_true_gaps` records the true float64 gaps at the `encode_obs` calls and compares every source's bins (exact and approximate) with them; the allow-list is trimmed |
| 8 | INFO | pre-registered values written into code that match the criteria file today (band tolerance, the 95 % percentiles of the mean interval, the secondary list, the run-pass gates of the blind tool, the thresholds in the diagnosis text); `--commit` of the post-launch checks defaults to HEAD, which moves after the launch commit; `starts_ok` is always true for `bw`; the last iterate's development value is read from the summary JSON | fixed or tied by tests: the 95 % interval takes the criteria percentiles, the blind tool takes the run-pass gates from the criteria file, the diagnosis text takes the thresholds from it, tests equate `BAND_TOL` and the secondary lists with `prereg_criteria.json`; `--commit` is required; the `bw` `starts_ok` is documented (bin-balanced by the pinned config; only the descriptive z flags apply); the summary JSON holds the scalars of `final_eval_development.npz`; `run_joint.py` imports `validate_two_stage_params` as `MSRun` does (a closed-form validity check that can only refuse a config) |

**Verified OK by the reviewer** (with what it ran): (1) the JT runner against `MSRun`: streams and namespaces, torch generator, init digest, LR expression and when it is set, entry refresh then every 20, exports every 25, the pure development check, global-RNG hardening (reference after the entry refresh; assertions at the end of training, after the final evaluation and at the end of the run), manifest, status, exit codes; **a `run_joint` run with only stage-3 starts (48 episodes, 45 updates, window 31-45) against `run_ms_stagewise`'s stage-3 phase: all 9 weight exports bit-identical, the positions of the five streams, the LR, KL, clip fraction, both losses, the advantage statistics and the mean return identical at all 45 updates, the same init digest** (now also a test); injected faults give exit 5 (`global_rng_violation`) and exit 1 (status failed), and the development check moves no stream, no torch generator and no global RNG. (2) The draw order from `starts_roles` against an independent re-implementation of D4; `t0` int64; stage blocks sorted and in ascending episode order; the exact visitation keys equal bincounts of the recorded gaps; all rows enter the update, no mask; the approximate decode differs from the true bins in 0 bins at q = 50 and 60 (largest decoding error 1.1e-5 / 1.3e-5). (3) D5: `expected_budget` and `decay_window` give 951 / 1316 / 826; the smoke runs show the updates, policy rows by stage, minibatch steps, export counts (50 / 70 / 44 / 152), check counts (50 / 70 / 44, 64 / 64 / 24), refreshes (63 / 87 / 55 plus the entry refresh) and the LR (actor and critic) equal to the window formula at every update. (4) C-INIT: 42/42 and 42/42; all four smoke arms at q = 50 share one digest. (5) Nothing closed-form enters training, the start sampler, the schedule or any run decision (F_xi only in the expected terminal reward; the strata and tail mask are used for reporting only: the legacy controller returns before any rule decision when the rule is disabled). (6) Metrics and evaluation: band split including edge nodes, terminal recovery on the grid of D_T (C-T2R 40/40), inclusive gates with the protocol's thresholds, the three loaders (C-REPLAY on the 8 smoke runs), the best-of-run rules. (7) Analysis and blind tool: pairing by (q, seed), dropped pairs counted, bootstrap indices created afresh with S = 20261016 / 20261017, percentile intervals, the median interval from the same indices, Clopper-Pearson counts, the strict "< 0" support rule; the blind tool imports nothing from the analysis. (8) The launcher reuses the nine functions of `launch_ms_r1.py`; seed guards, job counts 8 / 120 / 40, run directories and names, `--plan-csv`, validation by each runner; `params_bw_t3.json` equals D3 and `prereg_criteria.json` equals D6 and D7. (9) C-DIFF: in the frozen paths the commit only adds `run/run_joint.py`. (10) Every item of the section 2.2 test list is covered.

**Verification of the fixes** (the same reviewer, resumed, read-only, `git diff da4b1115..d993f612`, `pytest tests/test_t3_*.py` on a `git archive` extract of `d993f612`: **123 passed in 118 s**): findings 1-8 are all closed. Finding 3 had one leftover (MINOR): `--plan-digests` was still optional, so without it the "equal to the planned digests" half of C-INIT was skipped and the tool exited 0. One new INFO item: `tools/t3/compare_runs.py` looked for the second run's files only from the first run's directory (a file present only in the second run was not reported; a file present only in the first crashed). Both are fixed in `5535d610` (tools and tests only): `--plan-digests` is required for `post-launch` (together with `--commit`), and `compare_runs.py` compares the two file sets symmetrically and reports every difference (`test_compare_runs_flags_a_changed_array_and_csv_but_not_wall_clock`). The reviewer also confirmed that moving the development check after the per-update row changes no number (reduced `jt_es` and `jt2` runs with the runners of `da4b1115` and `d993f612`: exports, final-evaluation arrays, `visitation.npz`, both CSVs without wall-clock columns, `state_end.pt` and the summary fields bit-identical; the only difference is `update_wall_sec` at a check update, 0.089 s before and 0.055 s after), that `test_a_stage3_only_joint_run_equals_the_ms_stage_3_phase_bit_for_bit` is a genuine bit-for-bit test (40 updates, decay window 21-40, refreshes at 20 and 40, against `MSRun.run_stage(3)`), and that the `seed_hits.py` classes added for the 30501-30520 listing hold no training-run files. Its remaining INFO: the figure test inspects q = 50 only, the q = 60 figures are checked for existence and size. The full-scale reruns of section 5 add the numerical evidence for the runner change.

## 7. Fact-check of the documents against the prompt, the code and the records

After the review of the code, the documents of this commit sequence (`02_preregistration.md`, `00_housekeeping.md`, `03_checks.md` as far as it was filled, `reports/t3/README.md`, `docs/STATE.md`) were read once more by 12 independent read-only auditors, one per section (the commits and seeds, D2 to D9, the config hashes and launch plan, the departures, the housekeeping against P0, and a checklist of the P1 deliverables against the prompt), each finding being re-derived by two skeptics who tried to refute it (86 agents; `results/t3_r1/preflight/doc_audit_result.json`). The auditors checked 538 claims and reported 37 findings, 12 of which both skeptics refuted (the document already said it, or it was a wording point); **25 survived, all minor, none about a number a run produces or a rule of the pre-registration**: two cross-references to the wrong section, the `tests/test_t3_track.py` listed as uncommitted, "the working tree was clean" at the smoke launch (the launch record lists six untracked result files that the manifests' `clean_tree` flag ignores), the thread-variable check named against the record instead of the config, `512 / 512 / 512` for the single-stage arms where D5 says "512 in its stage", the D5 learning-rate column and policy-row definition, the source of the winner's evaluation in best of run, the bw cost sources, the hypothesis readings of D7, the missing `weights/` prefix and the D9 rule, the two wall-clock columns excluded from the determinism comparison, the advantage normalisation of the recorded v2.0 Phase B (now section 13.8 of the pre-registration), the first listing's CSVs that the re-run replaced (copies kept), numbers of the reviewer that were not stored (now `04_review_reports.md`), and four stale `claude/*` branches missing from the ref list. All 25 are corrected in the documents; nothing in the code or in a run changed.

