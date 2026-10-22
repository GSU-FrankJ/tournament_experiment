Round T3-R4: lock `bwA` as protocol T3-1.1 and confirm it on fresh seeds
Where the project stands

* Model. The T-stage rank-order tournament of `reports/t3/r1/pi_record/25_t3_r1_prompt.md` (D2).
   * k = 1/3500, w_H = 6, w_L = 2, ΔW = 4.
   * The shock ξ is triangular on [-2q, 2q]. B = 100 + 2q, and D_t = [-(t-1)B, (t-1)B].
   * T = 3 and q ∈ {50, 60}.
* Gates (`reports/t3/r1/prereg_criteria.json`, computed by `tools/t3/t3_metrics.py`):
   * G-A: η_3 ≤ 0.005, `rmse_pos_T_over_eT0` ≤ 0.05 and `tail_mean_T_over_eT0` ≤ 0.02;
   * G-F: `Gmax_full_over_dw` ≤ 0.01;
   * G-N: the development-final differences of η_3 and of Gmax_full are ≤ 0.001.
   * A run passes iff it exits with code 0, its global-RNG assertion is `ok`, its evaluation is valid on both tiers (`valid`, `dev_valid`), and it passes G-A, G-F and G-N. Every "≤" is inclusive at full float64 precision.
   * Tiers: the final tier `FINAL_CONFIG` and the development tier `DEV_CONFIG` of `utils/dp_br_verifier.py`.
* The solver `bw`, protocol T3-1.0 (`protocols/v2_T3_locked_v1_0.{json,md}`):
   * backward-induction training of stage 3, then stage 2, then stage 1, through `run/run_ms_stagewise.py`;
   * configs built by `tools/t3/t3_configs.build_bw_config` with `reports/t3/r1/params_bw_t3.json`;
   * per-run budget: updates 1600 / 1600 / 600, policy rows 819,200 / 819,200 / 307,200, 76,000 minibatch steps, 152 exports and 64 / 64 / 24 development checks.
Its confirmation in round T3-R2 failed: 34 of 40 runs passed, against a rule of at least 36 (`reports/t3/r2/04_confirmation.md`).
* Round T3-R3 (branch `t3-r3`, final commit `df2365d9278f0a0b7f3ab5efa5ec8cab04eaf1e5`) diagnosed the failures (`reports/t3/r3/01_diagnosis.md`) and tested two candidates on development seeds 70501-70520 (`reports/t3/r3/05_development.md`).
   * `bw` passed 37 of 40, `bwA` 39 of 40 and `bwAB` 38 of 40.
   * Under the pre-registered order `bwA` was proposed. At gate G2 the PI decided to lock `bwA`.
* `bwA` is `bw` plus the config key `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`. The code is `run/ms_retry.py` and `run/run_ms_stagewise.py`; `git diff b565f0e8 df2365d9 -- agents envs utils run tools/ms tools/v2` is empty. The behaviour (`reports/t3/r3/02_proposal.md` §2.1):
   * Trigger. At the development check at local update 1600 of the stage-3 phase, the trigger fires iff `Delta` > 0.005, or `Delta`, `s` or `C` is not finite, or `s` ≤ 0.
   * Retry. One in-place re-initialisation of stage 3 with derived seeds and streams, and one retraining. The second attempt is frozen, whatever its own check says.
   * Records.
      * The artefacts of attempt 1 are moved to `attempt1/`: `attempt1_record.json`, `ms_binmaps_stage3.npz`, `ms_checks_stage3.csv` and `ms_updates_stage3.csv` are tracked; the exports and `.pt` files are untracked. Nothing is overwritten or deleted.
      * `retry.json` records the trigger values and the derived seeds.
      * A run whose trigger does not fire is bit-identical to `bw`.
   * Budget of a fired run: 5,400 updates, 2,764,800 policy rows, 108,000 minibatch steps, 216 checks, 152 exports plus 64 in `attempt1/`.
* Paper wording for A (`reports/t3/r3/04_preregistration.md` §3): "a verifier-gated restart of the terminal stage: when the development-tier one-step deviation gain at the end of the stage-3 phase exceeds the gate's limit 0.005, stage 3 is re-initialised once and retrained, and the second attempt is kept; its cost is part of the method's cost."
* The confirmation rule was fixed before any development run, in T3-R3's pre-registration (`reports/t3/r3/04_preregistration.md` §9, D6), and is binding here:
   * Seeds: 80501-80540, the same 40 seeds at q = 50 and at q = 60, for 80 runs.
   * Rule: the confirmation passes iff at most 6 of the 80 runs fail and at most 4 of the 40 runs fail at each q.
   * Operating characteristic (independent per-run failure probability f, the same at both q): pass probability 0.982 at f = 0.03, 0.865 at 0.05, 0.271 at 0.10 and 0.030 at 0.15.
   * Failure counting:
      * Infrastructure failures: a negative return code, an out-of-memory kill, or ENOSPC, recorded with evidence in `infra_events.json`. Such a run is re-run once; a second infrastructure failure stops the confirmation without a verdict.
      * Other failures: any other nonzero exit, and any global-RNG violation, is a failed run.
      * Evaluation problems: an evaluation error, or an invalid evaluation, is a failed run unless a re-run of the locked evaluator succeeds.
      * The scheme's own steps: a re-initialisation inside a run is part of the run, never a re-run.
* Untracked roots, read-only:
   * T3-R3: `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-diagnosis-25cb13/results/t3_r3/`;
   * T3-R2: `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38/results/t3_r2/`.

Steps

1. P0: housekeeping, committed on its own (§1).
2. P1:
   * the tools and tests;
   * an independent review;
   * C-REPRO;
   * the lock drafts and `01_lock.md`;
   * the lock, the LOCK record and their push (§2).
3. P2: the 80-run wave, its checks, evaluation, analysis and blind recomputation (§3).
4. P3: reports, fact-check and push, then STOP (§4).

There is no gate inside this round: the PI took the decision at T3-R3's G2. Every stop-and-report rule below still applies.
Environment and standing rules

* Repository and Python. The repository is `/home/fjiang4/tournament_experiment`. Python is `/home/fjiang4/tournament_experiment/.venv/bin/python` only. Run `pytest tests`, never a bare `pytest`.
* Runs. Every run is single-threaded (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) and runs in tmux. Do not kill jobs you did not start.
* Branch and worktree.
   * Work in a new worktree, on a new branch `t3-r4` created from `origin/t3-r3` at `df2365d9278f0a0b7f3ab5efa5ec8cab04eaf1e5`. Record the worktree path.
   * The first file written there is this prompt, verbatim, as `reports/t3/r4/pi_record/29_t3_r4_prompt.md`.
* Other worktrees. Do not touch other worktrees, and never run `git clean`.
* Where files go.
   * Results go under `results/t3_r4/`, and reports under `reports/t3/r4/`.
   * New tools are new files `tools/t3/t3r4_*.py` (and `.sh`), and new tests are `tests/test_t3r4_*.py`.
   * Existing tools are reused by import, unchanged.
* Provenance: every number in a report cites a path.
* The closed form (`utils.theory_multistage.g2_two_stage`) enters evaluation and reporting only, never anything a run decides.
* Frozen files.
   * No existing file changes in `agents/`, `envs/`, `utils/`, `run/`, `tools/` or `tests/`.
   * `protocols/` gains only the two T3-1.1 files and one appended LOCK record. No existing protocol file or record changes.
* Tracking.
   * Tracking follows `reports/ms/README.md` and `tools/t3/t3_track.py`.
   * Never tracked: `.pt` files, `train_history.json`, weight exports, `freeze_stage*_*.npz`, `final_eval_*.npz` and evaluation arrays.
   * Never delete or overwrite results.
* Process rules, from T3-R3's deviations D-1 to D-3:
   * While the runs of a wave are starting or running, write nothing into the worktree outside `results/`.
   * A script that chains checks, evaluation and analysis stops at the first nonzero exit.
   * Every tool that takes `--root` refuses a relative path.
* Pushing happens only at the lock push (§2.6) and the final push (§4.3). `main` and tags are not touched, and no pull request is opened.

If any step fails, or anything does not match this prompt, stop at that point and report. Do not work around it.
1. P0 — housekeeping (`reports/t3/r4/00_housekeeping.md`; committed on its own)

1. Refs.
   * Run `git ls-remote origin` for `main`, `t3-r2`, `t3-r3` and the tags.
   * Verify that `origin/t3-r3` = `df2365d9278f0a0b7f3ab5efa5ec8cab04eaf1e5`.
   * Verify that `git diff --stat b565f0e8 df2365d9 -- agents envs utils run tools/ms tools/v2` is empty.
2. Branch and worktree. Create them, and save this prompt.
3. Seeds.
   * Run `tools/v2/seed_inventory.py --block 80501 80540` into `results/t3_r4/preflight/`, and list every hit with `tools/t3/seed_hits.py`.
   * Stop-and-report on a hit in a training-run record (`run_config.json`, `manifest.json`, `status.json`, `gates.json` or a run directory, in any worktree).
   * Classify all other hits. T3-R3's own inventory and pre-registration name the block; those hits are expected.
4. Roots. Verify the two untracked roots. For the C-REPRO runs (§2.3), record that the T3-R3 originals have their 152 exports, `state_end_stage{1,2,3}.pt`, and for the fired run `attempt1/`.
5. Machine. Record `nproc`, load, free disk, and the Python, torch and numpy versions.
6. A known erratum, recorded here only. `reports/t3/r3/01_diagnosis.md` says that from update 400 on the out-of-band gradient share is 0.1-0.7 %. Its own table shows 1.06-1.7 % in the window 401-800. The conclusion is unchanged. Do not edit T3-R3's files.
7. Commit P0.

2. P1 — tools, C-REPRO and the lock
2.1 Tools (new files; reuse the `t3r3_*`, `t3c_*` and `t3_*` tools by import where they apply)

* `tools/t3/t3r4_make_lock.py`.
   * It writes `v2_T3_locked_v1_1.json` and `.md` into the directory given by `--out`. They contain:
      * the T3-1.0 settings with `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}` added;
      * the reference config SHA-256 per q, normalised as `tools/t3/t3c_launch.py` does for T3-1.0, with the key added;
      * the paper wording above, verbatim;
      * the confirmation rule, the operating characteristic, the failure counting and the seed block;
      * the SHA-256 of `run/ms_retry.py`, `run/run_ms_stagewise.py` and every tool that the confirmation runs.
   * The files hold no timestamp, no commit hash and no absolute path.
* `tools/t3/t3r4_launch.py`.
   * Arm `bwA` only; seeds exactly 80501-80540 at q ∈ {50, 60}.
   * Run names are `t3r4_q{q}_s{seed}_bwA`, run directories `results/t3_r4/confirmation/q{q}/seed{s}/bwA/`, and `pilot` = "t3_r4".
   * Each config is the T3-1.1 reference with only `seed`, `run`, `record.run`, `record.seed`, `record.output_dir` and `pilot` set.
   * `--plan-csv` writes the run name, output directory, config SHA-256 and planned init digest of every run, and checks against the lock drafts.
   * At launch, it refuses unless all of these hold:
      * the T3-1.1 record is the last record of `protocols/LOCK`;
      * the lock JSON and MD and every listed file match their SHA-256;
      * HEAD is the LOCK-record commit;
      * `git status --porcelain` shows nothing outside `results/`;
      * `git diff --stat b565f0e8 HEAD -- agents envs utils run tools/ms tools/v2` is empty;
      * `--root` is absolute.
   * Its `infra_events.json` ledger works as in T3-R2's `tools/t3/t3c_launch.py`.
* `tools/t3/t3r4_checks.py`: the post-launch checks of §3.2.
* `tools/t3/t3r4_eval.py`: `tools/t3/t3_eval.evaluate_run` on both tiers for the 80 runs, as `tools/t3/t3r3_eval.py` does for `bwA`.
* `tools/t3/t3r4_analysis.py`: the verdict and the reported items of §3.3.
* `tools/t3/t3r4_blind.py`: an independent recomputation of the counts and the verdict from the evaluation table and `infra_events.json`. It does not import the analysis tool.
* `tools/t3/t3r4_post_launch.sh`: checks, then evaluation, then analysis, then blind recomputation, stopping at the first nonzero exit.

2.2 Tests (`tests/test_t3r4_*.py`)

* The launcher guards: seeds, arm, relative root, lock or tool mismatch, a missing record, a dirty tree, a nonempty code diff.
* Configs: each config differs from the reference only in the six run fields, checked with `tools/t3/t3_configs.config_diff`.
* The lock maker is deterministic: two calls give byte-identical files.
* The analysis and blind tools, on synthetic tables with known verdicts:
   * a pass at 6 failures (3 + 3);
   * a fail at 7 (4 + 3);
   * a fail at 5 failures with 5 at one q (5 + 0);
   * a pass at 4 + 2;
   * an evaluation error counted as a failure;
   * a stop without a verdict on a second infrastructure failure.
* The full suite. `pytest tests` passes, apart from the known `test_registry_canonicalization` and `tests/test_ms_r4_launcher.py::test_the_existing_launcher_is_unchanged`.

2.3 C-REPRO (after the review fixes, at the final code commit)

* Re-run T3-R3's `bwA` runs (q = 50, seed 70508) and (q = 60, seed 70501) from their own `run_config.json` into `results/t3_r4/repro/`. The first fired its trigger; the second did not.
* Compare each with its original in the T3-R3 untracked root: `tools/t3/compare_runs.py bw`, plus `attempt1/` and `retry.json` for 70508.
* Both must be bit-identical. A mismatch is a stop.

2.4 Independent review
One read-only review of the new tools and tests by an agent that did not write them. It checks:

* the launcher's guards;
* the lock contents against this prompt;
* the checks of §3.2;
* the rule and the failure counting in the analysis and blind tools.

Record it in `reports/t3/r4/02_checks.md`, and fix the findings before C-REPRO.
2.5 `01_lock.md` and the lock drafts (committed before the lock)

* Run `t3r4_make_lock.py --out reports/t3/r4/lock_draft/`.
* `reports/t3/r4/01_lock.md` holds:
   * the settings;
   * the rule, the operating characteristic (recomputed by a script under `reports/t3/r4/report_scripts/`) and the failure counting;
   * the paper wording;
   * the plan CSV (`t3r4_launch.py --plan-csv`);
   * the config differences against T3-R3's `bwA` configs of the same q;
   * the seed inventory and C-REPRO.

2.6 The lock

1. Lock commit. Run `t3r4_make_lock.py --out protocols/`. The files must equal the drafts byte for byte. Commit them.
2. LOCK record. Append one record to `protocols/LOCK`, in the format of its existing records, and commit it separately. It holds:
   * protocol T3-1.1, the lock commit, the JSON and MD SHA-256 and the date;
   * the entry: `run/run_ms_stagewise.py` through `tools/t3/t3r4_launch.py`;
   * the reason: "PI decision at gate G2 of round T3-R3: lock `bwA`";
   * the rule "nothing in protocols/ changes after this lock without a new version number and a stated reason".
3. Push both commits.

3. P2 — the confirmation wave
3.1 Launch

* One wave of 80 runs, at most 40 single-threaded workers, in tmux, from the LOCK-record commit, on a clean tree.
* Record `nproc`, load, free disk, HEAD and `git status --porcelain` at the launch.
* Then write nothing outside `results/` until every run has finished.
* When all runs have finished, run `t3r4_post_launch.sh`.

3.2 Post-launch checks (`results/t3_r4/confirmation/launch_checks.json`; deterministic failures are a stop)

* Basics: exit codes; complete files; manifests at the launch commit with a clean tree; the global-RNG assertions; `infra_events.json` present.
* Configs: each config equals the plan.
* Budgets and schedules, per run:
   * Not fired: the `bw` budgets above, the learning rate at every update, 152 exports, 64 / 64 / 24 checks, and no `retry.json` or `attempt1/`.
   * Fired: the fired budgets above, `retry.json` present and consistent with the attempt-1 check, `attempt1/` complete, and the attempt-2 init digest equal to the derived one.
* C-INIT: the recorded init digests equal the planned ones.
* C-REPLAY: the evaluator reproduces each run's freeze arrays bit for bit.

3.3 Evaluation, verdict and reported items

* Evaluation with `t3r4_eval.py`, then the verdict under the rule with `t3r4_analysis.py`. `t3r4_blind.py` must agree with it.
* Reported, deciding nothing:
   * Pass counts: per q and in total, with exact Clopper-Pearson 95 % intervals.
   * Values: the worst Gmax_full and its run; the medians of η_3, `rmse_pos_T_over_eT0`, `tail_mean_T_over_eT0`, `peak_T_signed`, Gmax_full and EXP_root.
   * The trigger: its fire rate, with the runs; and the first-attempt outcome, the pass count with every fired run counted as a failure of its first attempt.
   * Each failure: the failing gates and values, the stage of the failing quantity, S3, the stage-2 stretch, and the last-16-export range of the failing stage-3 quantity (with `tools/t3/t3r3_last16.py`). Classify it as in T3-R3's D2 (`reports/t3/r3/pi_record/28_t3_r3_prompt.md`): (F), (K), (V), (E) or (U).
      * S3 = Σ `d1_L_s3_out_lo` / Σ `d1_L_s3_out_n` over the stage-3 rows with local update 1201-1600 of the frozen attempt.
      * A stage-2 stretch is a run of consecutive in-band (|d| < 2q) stage-2 nodes of the final-tier grid with learned mean effort below 1.
   * Cost: updates, policy rows and wall time per run, and the share of fired runs.
   * Descriptive comparisons:
      * this confirmation pooled with `bwA`'s 40 development runs (39 of 40);
      * T3-1.0's three blocks of `bw` (36, 34 and 37 of 40);
      * the pass rate of T3-1.0 implied by the first-attempt outcome.

4. P3 — reports, fact-check, push
4.1 Reports (`reports/t3/r4/`)

* `03_confirmation.md`, in this order:
   * the checks;
   * the locked settings and the code version;
   * the per-seed tables;
   * the verdict under the rule;
   * the trigger and the first-attempt outcome;
   * the failures, classified;
   * the cost;
   * the descriptive comparisons.
* `summary.md`.
* `handoff.md`, in Chinese only:
   * the verdict with the counts, the trigger rate, the failures and the cost, and what the paper must say about A;
   * at most five GitHub links in the text, at the commit that holds the reports;
   * no name or contact detail.

Every table is generated by a script under `reports/t3/r4/report_scripts/`. Add a T3-R4 row to `reports/t3/README.md`, and a dated section to `docs/STATE.md`.
4.2 Fact-check
An independent, read-only fact-check checks every number against the CSVs. Its ledger is `reports/t3/r4/pi_record/01_factcheck.md`. Corrections go in before the push.
4.3 Push and STOP

* Push `t3-r4` (fast-forward). Create no tags, leave `main` untouched, and open no pull request.
* STOP. Report the verdict, the counts, the trigger rate and anything flagged.
* If the confirmation fails, nothing is changed or tried in this round; the PI decides the next step.
