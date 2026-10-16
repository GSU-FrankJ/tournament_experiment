# T3-R1: the independent review, verbatim

Provenance. One read-only reviewer that did not write the code: an Agent of type `general-purpose`, model `opus`, started in the background with the prompt of section 1 (the code commit `da4b1115` under review). Its first report (section 2) took 78 tool calls and 21.6 minutes; the reviewer was then resumed (same agent) with the request of section 3 and gave the verification report of section 4 (18 tool calls, 4.7 minutes). The texts are copied from the session transcript without change (HTML entities decoded); the numbers quoted in `03_checks.md` section 6 are the reviewer's own (its scratch directory `/tmp/t3review.*` is not kept). The reviewer's scratch results were not stored; every finding was re-checked by the implementer before it was marked fixed, and the fixes are tests in `tests/test_t3_*.py`.

## 1. The prompt given to the reviewer

You are an independent, READ-ONLY code reviewer for a research repository (game-theory / PPO experiments). You did not write this code. Do NOT modify, create or delete any file inside the repository or its worktrees, do NOT run `git clean`, `git checkout`, `git stash`, `git commit` or anything that changes git state, and do NOT launch training runs. You may read files, run `git diff`/`git show`/`git log`, and run small throw-away Python snippets that import repo modules and write only to your own scratch directory (use `mktemp -d` under /tmp, not the repo). Always use the interpreter `/home/fjiang4/tournament_experiment/.venv/bin/python` and set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. If you run pytest, run only `.../python -m pytest tests/test_t3_*.py -p no:cacheprovider -q` (it takes a few minutes; other long jobs are running on this machine, do not kill anything).

Repository / worktree: /home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472 (branch `t3-r1`). The change to review is the commit `da4b1115f518555125c4fab0a99c20e12243634a` on top of `db7577ea5a85910ecfc43db142e872fc1950bf75` (`git diff db7577ea da4b1115`): a new runner `run/run_joint.py`, new tools `tools/t3/*.py` (+ `preflight_seeds.sh`), new tests `tests/t3_helpers.py`, `tests/test_t3_*.py`, and two JSON files under `reports/t3/r1/` (`params_bw_t3.json`, `prereg_criteria.json`). The specification of the round is the PI's prompt, saved verbatim in the repository at `reports/t3/r1/pi_record/25_t3_r1_prompt.md` — read its decisions D1-D9 and sections 2.1-2.5 and 3.2 first; the code must conform to them. The existing pipeline that the new code reuses or must mirror: `run/run_ms_stagewise.py` (class `MSRun`, `validate_config`, `run_pipeline`), `run/v2_rollout.py` (`collect_batch_v2`), `run/run_v2_T2_locked.py` (global-RNG hardening), `agents/ppo_curriculum.py`, `agents/ppo_curriculum_v2.py`, `utils/v2_metrics.py`, `utils/dp_br_verifier.py`, `tools/ms/ms_configs.py`, `tools/ms/launch_ms_r1.py`, `tools/ms/replay_dev_rule.py`, `tools/ms/r4_common.py`.

Review scope (this is the list the PI asked for):
1. Conformance of the code with D2-D7 of the prompt (game, shared settings, arms, the JT runner `run/run_joint.py` D4, the budget rule D5, what is measured D6, criteria D7).
2. `run/run_joint.py` against `MSRun`/`run_pipeline` in everything D2 says is shared: RNG streams and namespaces, torch generator, learning-rate expression and when it is set, snapshot refresh order (entry refresh, every-20), weight exports (every 25), development checks (purity: no RNG moved; `DomainError`/`FloatingPointError`/`ValueError` handling), global-RNG hardening (when the reference states are taken; assertion points; exit code 5), initial-weight digest, `manifest.json` content, exit codes 0/1/3/4/5, status files. Look for any difference that could change a trained policy or a recorded number relative to what the prompt specifies, e.g. ordering of draws from the `starts_roles` stream, which rows enter the PPO update, the observation/gap decoding of the approximate visitation counts, the `t0` dtype/ordering assumptions about `collect_batch_v2`'s stage blocks.
3. The budget arithmetic of D5 (policy rows per stage, minibatch steps, decay windows = last 25 % rounded up) as implemented in `tools/t3/t3_configs.py` (`JOINT_BUDGETS`, `expected_budget`, `decay_window`, `production_problems`), and check C-INIT (`tools/t3/t3_launch_checks.py: c_init_plan`).
4. That no closed-form equilibrium (e_T*, g1, `g2_two_stage`, `g1_two_stage`) enters training, the start sampler, the schedule or anything a run decides (the expected terminal reward's shock CDF F_xi is part of the game and allowed). Note `StartSampler.strata`/`tail_mask` in `envs/curriculum_env.py` (pre-existing, unchanged) are used only for reporting under bin-balanced starts.
5. The band split, the terminal-stage recovery for any T, the gates (`tools/t3/t3_metrics.py`), the loaders and the evaluation of the three run formats (`tools/t3/t3_eval.py`: `bw` composite from `state_end_stage1.pt`, joint runs from `state_end.pt`, `v20` from the `u01600` export at t=2 and the `u02200` export at t=1; the best-of-run selection rules), and the analysis definitions (`tools/t3/t3_analysis.py`: pairing by (q, seed), failed runs/missing pairs dropped and counted, bootstrap indices `numpy.random.default_rng(S).integers(0, n, size=(10000, n))` created afresh per call, percentile intervals, median interval from the same indices, Clopper-Pearson counts, the hypothesis support rule) and the independent blind tool (`tools/t3/t3_blind.py`, which must not import the analysis module and must follow `reports/t3/r1/prereg_criteria.json` literally).
6. The launcher `tools/t3/t3_launch.py` and the post-launch checks `tools/t3/t3_launch_checks.py` against section 2.1 and 3.2 (waves smoke/t3/t2j and their seed guards, run directories, the 120/40/8 job counts, `--plan-csv`, which functions of `tools/ms/launch_ms_r1.py` are reused, the check list: budgets, schedules including the export counts 152/50/70/44 and check counts 64/64/24 for bw, starts, C-INIT, C-REPLAY).
7. The tests: do they test what they claim? Are there tests that pass vacuously (e.g. assertions that cannot fail, `or True`, tolerance too loose)? Missing coverage of anything in the prompt's section 2.2 list?

Report format: a numbered list of findings, each with severity (MAJOR = would change a result or violate the spec; MINOR = incorrect/misleading but no effect on results; INFO), file:line, what the specification says (quote the section), what the code does, and a concrete failure scenario or the evidence (command + output) that supports it. Verify every claim you make by reading the code or running a snippet; do not report speculation as fact. If you find nothing in an area, say so explicitly ("checked X, no finding"). Be concise: findings first, then a short list of what you checked and found conformant. Do not suggest new gates, hashes or baselines; the PI's rule is that guards are added only at irreversible, cross-system, security or release boundaries.

## 2. First report (on `da4b1115`)

I reviewed commit `da4b1115` against `db7577ea` and found two spec-conformance gaps (rated MAJOR under your definition), several MINOR issues and some INFO notes. Nothing found would change a trained policy, a recorded number, or a pre-registered verdict when the inputs are correct.

While I worked, another session was editing the working tree (uncommitted changes to `tools/t3/t3_launch.py` and several tests, plus cleanup elsewhere). So I reviewed the committed files (`git show da4b1115:<path>`; line numbers below are the commit's) and ran the tests on a `git archive` extract of the commit: `pytest tests/test_t3_*.py` gave **111 passed in 74 s**. My scripts are in `/tmp/t3review.rLabxx/`.

## Findings

**1. MAJOR (procedural only): the committed launcher cannot launch t3 and t2j as one launch.** `tools/t3/t3_launch.py:171,184`.
- Spec, §3.1: "One launch through `tools/t3/t3_launch.py`: waves `t3` and `t2j` (160 runs), at most 40 single-threaded workers".
- Code: `p.add_argument("--wave", choices=tc.WAVES, required=True)` accepts one wave. The two waves need two calls, so two pools and two records: either more than 40 workers in total or two sequential launches.
- The uncommitted working tree changes this to `nargs="+"` (one pool, one record) and adds `test_two_waves_are_one_launch_with_one_record_and_one_plan`. That is not part of `da4b1115`. In that version `--arms` is silently ignored when several waves are given.

**2. MAJOR (descriptive only): the figures in `t3_analysis.py` cover only part of §3.3 and of D6 "Curves".** `tools/t3/t3_analysis.py:401-542`. §2.1 assigns "the figures of §3.3" to this tool.
- `figure_effort` draws no closed form and no g1, although §3.3 asks for both and its own docstring (402-403) says it does. It plots q = 50 only (line 411).
- `figure_gains` plots the T = 3 arms only (no `v20`, no `jt2`) and q = 50 only.
- `figure_paired` shows only Gmax_full. §3.3 also asks for EXP_root and the in-band and out-of-band maxima per stage.
- `figure_curves` shows Gmax_full, EXP_root and η_3 for `jt`/`jt_es`, and only stage 3 for `bw`. D6 asks for η_t and G_max_t for every stage; `jt2` is missing.
- `figure_visitation` has no `jt2` and q = 50 only.
- `test_figures_run_on_the_reduced_runs` only checks that each file is larger than 1000 bytes.
- Note: after G1 only `reports/t3/` and `results/t3_r1/` may change, so any fix would have to come before G1 or be done by report scripts.

**3. MINOR: a C-INIT failure does not change the post-launch exit status.** `tools/t3/t3_launch_checks.py:264-271, 388-391`.
- `bad` is computed from the per-run verdicts only. `c_init_within_q_seed` and `c_init_equals_plan` are written and printed, but a mismatch still exits 0.
- §3.2 says "A failed deterministic check is a stop-and-report", and C-INIT is deterministic.

**4. MINOR: a missing `v20` reference does not stop the evaluation, so H1-T2 comes out "not supported".** `tools/t3/t3_eval.py:192-199, 348`; `t3_analysis.py:113-121`; `t3_blind.py:96-115`.
- Spec, D9: "A missing file is a stop-and-report."
- Code: `--v20-root` is optional ("omit to skip"), and a run directory without `status.json` becomes a "missing" row.
- I ran a per-run table without `v20` rows through both tools. Each gave H1-T2 at q = 50 and 60 with n_pairs 0 and dropped_pairs 20, `hypothesis_supported False`, fails at "50;60". The blind check agrees with the analysis, so it does not catch this.
- C-T2R does stop in this case, because it reads `gates.json` directly.

**5. MINOR: two error paths crash the analysis instead of reporting the run.** I triggered both with a snippet.
- `t3_analysis.py:225`: `int(r["Gmax_full_t"])` raises `ValueError: cannot convert float NaN to integer` for a `bw` run that exited 0 but whose evaluation raised (`t3_eval.py:208-209` stores the error and leaves the metrics empty). The analysis stops, against D7's "every planned run is reported".
- `t3_eval.py:219`: `r["valid"]` raises `KeyError` when every development check of a joint run failed. In that case `run_joint.py:434-438` writes only the base columns, so the best-of-run pool aborts.

**6. MINOR: D5's check of the `v20` step count is not done by any tool.**
- Spec, D5: "Verify this from each run's `v2_updates.csv`".
- `t3_analysis.py:340-351` computes the steps into `cost.csv`, but nothing compares them with 56,000. `expected_budget("v20")` (`t3_configs.py:244-248`) is never used, and the post-launch checks have no `v20` jobs (`t3_launch_checks.py:385`).
- The fact itself holds: all 40 runs have 1600 Phase-A rows with `n_rows` 512, 600 Phase-B rows with `n_rows` 1024, `n_policy_rows` 512, `n_epochs_run` 10, and 56,000 steps.

**7. MINOR: some tests cannot fail, or cannot detect what they say they test.**
- `tests/test_t3_launcher.py:291`: `res["manifest_commit_ok"]` is always True when `commit=None` (`t3_launch_checks.py:127`), and `res["clean_tree"] in (True, False)` is always true.
- `tests/test_t3_joint_runner.py:260-266`: the "decoded observation" test checks only the array length (80) and the total (8·U). The total holds by construction (`run_joint.py:381-387`), so wrong rows would still pass.
  - My own check shows the implementation is correct. On one 256/85/171 batch I compared the bins of the true float64 gaps (recorded inside `encode_obs`) with the runner's approximate bins: 0 bins differ at q = 50 and 60; the largest decoding error is 1.1e-5 and 1.3e-5.
- `tests/test_t3_launcher.py:182`: for `bw`, the allowed q-differences include `pipeline` and `derived`, so a q-dependent pipeline would not be caught. The actual differences are only q, `record_sha256`, `run` and the record fields that depend on q (B, `domain_half_stage2`, `es_bins_stage2`, `game.q`).

**8. INFO.**
- `run/run_joint.py:71, 251-256` imports `validate_two_stage_params`, a closed-form validity check at T = 2, exactly as `MSRun` does (lines 402-407). It can only refuse a config; it computes nothing used in training.
- Some pre-registered values are written into the code rather than read from `prereg_criteria.json`. All match the file today:
  - `t3_metrics.py:30`, `BAND_TOL`;
  - `t3_analysis.py:87`, the 95 % interval uses `r4_common`'s default (2.5, 97.5);
  - `t3_analysis.py:128-138`, the list of secondary statistics;
  - `t3_blind.py:150`, the gate names of a run pass;
  - `t3_analysis.py:219-233`, the diagnosis text.
- `t3_launch_checks.py:367`: `--commit` defaults to the current HEAD. HEAD is now `4de166b2` and the smoke manifests carry `91a9f514`, so the launch commit must be passed explicitly. Otherwise every run fails `manifest_commit_ok` (a false failure, never a false pass).
- `t3_eval.py:230-235` reads the last iterate's development value from `joint_run_summary.json`. These are the scalars of the same evaluation; `final_eval_development.npz` holds only arrays.
- `t3_launch_checks.py:189`: `starts_ok` is always True for `bw`. §3.2 requires only the descriptive z counts for `bw`.

## Checked and found conformant (no finding)

- **JT runner against `MSRun` (point 2).** These match: streams and namespaces, the torch generator, the initial-weight digest, the LR expression and when it is set, entry refresh then every 20, exports every 25, the development check, global-RNG hardening (reference after the entry refresh; asserts at end of training, after the final evaluation and at end of run), the manifest and status files, and exit codes 0/1/3/4/5.
  - I ran `run_joint` with only stage-3 starts (48 episodes, 45 updates, decay window 31-45) against `run_ms_stagewise`'s stage-3 phase. All 9 weight exports were bit-identical. At all 45 updates the positions of the five streams, the LR, KL, clip fraction, both losses, the advantage statistics and the mean return were identical, and the init digest was the same (`check_equiv_ms.py`).
  - Injected faults gave exit 5 (`global_rng_violation`) and exit 1 (status failed). The development check moved no stream, no torch generator and no global RNG (`check_exit_paths.py`).
- **Starts and rollout blocks.** The draw order from `starts_roles` matches an independent re-implementation of D4. `t0` is int64; the stage blocks are sorted and in ascending episode order; the exact visitation keys equal bincounts of the recorded gaps. All rows enter the update, with no mask.
- **D5 budgets.** `expected_budget` and `decay_window` give 951/1316/826. The smoke runs show:
  - updates 1267/1754/1100;
  - policy rows by stage, minibatch steps 76,020/87,700/44,000, and 76,000 for `bw`;
  - exports 50/70/44/152;
  - checks 50/70/44, and 64/64/24 for `bw`;
  - refreshes 63/87/55 plus the entry refresh;
  - the LR, both actor and critic, exactly equal to the window formula at every update.
- **C-INIT.** The planning logic is correct. The recorded outputs are 42/42 and 42/42, and all four smoke arms at q = 50 share one digest.
- **Closed form.** Nothing closed-form enters training, the start sampler, the schedule or any run decision. F_ξ appears only in the expected terminal reward. The strata and tail mask are used only for reporting: the legacy controller returns before any rule decision when the rule is disabled (`utils/ms_rule.py:299-305`), and the z-test only reads them.
- **Metrics and evaluation.**
  - The band split, including edge nodes.
  - Terminal recovery on the grid of D_T; C-T2R reproduced 40/40.
  - Gates: inclusive comparisons, with the protocol's thresholds.
  - The `bw`, joint and `v20` loaders; C-REPLAY passed on the 8 smoke runs.
  - The best-of-run rules.
- **Analysis and blind tool.**
  - Pairing by (q, seed); dropped pairs are counted; bootstrap indices are created afresh with S = 20261016 / 20261017.
  - Percentile intervals; the median interval uses the same indices; Clopper-Pearson counts; the strict "< 0" support rule.
  - The blind tool imports nothing from the analysis.
- **Launcher.** It reuses the nine functions of `launch_ms_r1.py`; seed guards, job counts 8/120/40, run directories and names, `--plan-csv`, and validation by each runner. `params_bw_t3.json` matches D3, and `prereg_criteria.json` matches D6 and D7.
- **C-DIFF.** In the frozen paths the commit only adds `run/run_joint.py`.
- **Tests.** Every item of the §2.2 list is covered, with the weaknesses noted in finding 7.

## 3. The request to verify the fixes (on `d993f612`)

Please verify the fixes for your review (read-only again; same rules as before: change nothing in the repository, use your own scratch directory, set OMP/MKL/OPENBLAS_NUM_THREADS=1, interpreter /home/fjiang4/tournament_experiment/.venv/bin/python).

The fix commit is d993f6124f18445dfb040d06c83a03725ba457e6 on branch t3-r1 (`git diff da4b1115..d993f612`). Review exactly that diff; the working tree of the worktree may contain uncommitted documents (reports/), ignore them. Run the tests on a `git archive` extract of d993f612, as you did before (`pytest tests/test_t3_*.py`).

For each of your numbered findings say whether the fix closes it, and look for anything new the fix introduced:
1. launcher: `--wave t3 t2j` is one launch (one pool, one record), `--seeds` / `--arms` refused with several waves.
2. figures in tools/t3/t3_analysis.py (figure_effort, figure_gains, figure_paired, figure_curves, figure_visitation) against section 3.3 and D6 "Curves"; check that the tests really inspect their content.
3. C-INIT now decides the post-launch exit status (post_launch_ok) and `--commit` is required.
4. a missing v20 reference file stops tools/t3/t3_eval.py (D9).
5. the two crash paths (bw_gate_table with a run that exited 0 but has no metrics; read_checks when every development evaluation failed).
6. the v20 step count against D5 (budget_check in the analysis; also results/t3_r1/preflight/v20_budget_check.txt).
7. the vacuous assertions (manifest_commit_ok / clean_tree test; the visitation decode test; the bw q-difference allow-list).
8. INFO items: the 95 % interval now comes from the criteria percentiles; the blind tool takes the run-pass gates from prereg_criteria.json; the diagnosis text takes its thresholds from the criteria file.
Also please re-check two things I changed in run/run_joint.py: the development check now runs after the per-update row is appended (so `update_wall_sec` excludes the check, as in MSRun) -- confirm nothing numerical changes; and the new test `test_a_stage3_only_joint_run_equals_the_ms_stage_3_phase_bit_for_bit`.

Report only: per finding closed / not closed (with evidence), and any new finding with severity. Keep it short.

## 4. Verification report (on `d993f612`)

Every finding is closed by `d993f612`. One small leftover remains on finding 3, and the fix introduced nothing new that changes a number. I reviewed exactly `git diff da4b1115..d993f612`; on a `git archive` extract of `d993f612`, `pytest tests/test_t3_*.py` gave **123 passed in 118 s**.

## Per finding

1. **Closed.** `--wave` now takes several waves (`t3_launch.py:171-205`), all jobs go through one `run_pool`, and the single record is `<root>/launch_t3_t2j_<stamp>.json`. `--seeds` or `--arms` with several waves is refused. Both new tests check this and pass.

2. **Closed.** Each figure now matches §3.3 and D6 "Curves", at both q:
   - effort functions: T = 3 and T = 2 rows, the closed form at the terminal stage, g1 at stage 1 of T = 2;
   - deviation gains: every stage, median and maximum, band edges marked;
   - paired differences: Gmax_full, EXP_root and the in-band / out-of-band maxima of G_t and Δ_t, for the four pairs;
   - learning curves: η_t and G_max_t for every stage, `jt2`, and each `bw` phase's own Delta;
   - visitation: `jt2` added, plus `bw`'s design coverage.

   `test_figures_have_the_pre_registered_content` really inspects content: panel titles and counts, legend entries, the closed-form line (d from −400 to 400, peak 70 at q = 50), the band-edge positions, the rows per paired panel (8 or 6), and the log scale. It inspects q = 50 only; the q = 60 figures are checked only for existence and size (INFO).

3. **Closed, with a leftover (MINOR).** `post_launch_ok` now counts C-INIT within each (q, seed), and `--commit` is required. But `--plan-digests` is still optional. Without it, the "equal to the planned digests" half of C-INIT is skipped and the tool still exits 0 (`t3_launch_checks.py:368`, `post_launch_ok`).

4. **Closed.** `t3_eval.py` now stops if `--v20-root` is missing for wave `t2j`, or if any of the 5 needed files of the 40 `v20` runs is missing. It stops before creating the output directory, and the test passes.

5. **Closed.** `bw_gate_table` reports "no result (…, evaluation error: …)" when a run exited 0 but has no metrics. `read_checks` and `best_candidates` return no candidate when every development evaluation failed. Both are tested.

6. **Closed.** `budget_check.csv` compares each run's updates, policy rows (per stage and total) and minibatch steps with `expected_budget`; for `v20` they come from `v2_updates.csv`. The headline gets `budget_check_all_equal`. `results/t3_r1/preflight/v20_budget_check.txt` is untracked (not in `d993f612`). Its values (40 runs, 2200 updates, 56,000 steps, 1,126,400 rows, 819,200 / 307,200 by stage) match my own check.

7. **Closed.**
   - `clean_tree` is now compared with the manifest's flag, and `manifest_commit_ok` is shown to be False for a wrong commit.
   - The new visitation test compares all five source arrays with the bins of the true float64 gaps captured at the `encode_obs` calls, so it can catch a wrong row selection.
   - `pipeline` and `derived` were removed from the `bw` q-difference allow-list.

8. **Closed.**
   - The 95 % interval now takes its percentiles from the criteria file; the test checks it against an independent bootstrap and against changed percentiles.
   - The blind tool takes the run-pass gates from `gates.run_pass` in the criteria file.
   - The diagnosis text takes its thresholds from the criteria file.
   - New tests pin `BAND_TOL` and the secondary-statistics list to the criteria file.

## The two runner items

**Moving the development check after the per-update row changes no number.** I ran reduced `jt_es` (T = 3, 24 / 8 / 16 starts) and `jt2` (T = 2, 32 / 16) configs (U = 30, 48 episodes) with the runner of `da4b1115` and of `d993f612`. In each case the 3 exports, the 2 final-evaluation arrays and `visitation.npz` were bit-identical. So were both CSVs without their wall-clock columns, `state_end.pt` (actor, critic, opponent, streams, counters) and the summary fields. `tools/t3/compare_runs.py` agrees. The only change is `update_wall_sec` at a check update: 0.089 s before, 0.055 s now, the verifier's 0.036 s no longer included.

**`test_a_stage3_only_joint_run_equals_the_ms_stage_3_phase_bit_for_bit` is a genuine bit-for-bit test.** It covers 40 updates with the decay window 21-40 and snapshot refreshes at 20 and 40. It compares, against `MSRun.run_stage(3)`:
- the actor, critic, opponent, both optimizer states, the minibatch stream and the refresh count;
- the four stream states;
- every per-update LR, loss, KL, clip fraction, advantage statistic, mean return and stream position;
- the u00025 export, the init digest, and η_T and Gmax at the first check.

It passes.

## New findings

- **INFO:** `tools/t3/compare_runs.py` only looks for files from the first run directory in the second. A file present only in the second is not reported, and a file present only in the first makes it crash rather than report a difference.
- No other new issues. The `seed_hits.py` classes added for the 30501-30520 seed-hit listing (`results/ms_r1/calibration/`, `diag_30510/`) hold no training-run files: the only manifest there belongs to the offline replay tool.

## 5. Afterwards

The leftover of finding 3 (`--plan-digests` optional) and the INFO item (`compare_runs.py`, one-sided file lookup) were fixed in `5535d610` (tools and tests only) and covered by tests; `5535d610` was not reviewed again by the reviewer.
