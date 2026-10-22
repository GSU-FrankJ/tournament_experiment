# T3-R4: the confirmation of protocol T3-1.1 (`bwA`) on fresh seeds

Date: {{date}}. Branch `t3-r4`. **Generated** from `reports/t3/r4/report_scripts/03_confirmation.template.md` by `build_report.py`; every table by `reports/t3/r4/report_scripts/confirmation_tables.py` from `results/t3_r4/confirmation/launch_checks.json`, `eval/per_run.csv` and `analysis/` (written by `tools/t3/t3r4_checks.py`, `t3r4_eval.py`, `t3r4_analysis.py`, `t3r4_blind.py`). The binding text is the prompt (`pi_record/29_t3_r4_prompt.md`); the rule was fixed before any development run in T3-R3's pre-registration (`reports/t3/r3/04_preregistration.md` section 9, D6). Nothing was changed or tuned after the lock.

**In one paragraph.** One wave of 80 runs of the locked protocol T3-1.1 (the arm `bwA`: `bw` plus the verifier-gated restart of stage 3) on the fresh seeds 80501-80540 at q = 50 and q = 60 ran to completion: all 80 exit codes 0, no infrastructure event, every post-launch check passes. **79 of the 80 runs pass (q = 50: 40 of 40; q = 60: 39 of 40; exact 95 % interval of the pass rate [0.932, 1.000]). The rule (at most 6 failures of 80 and at most 4 of 40 at each q) is met: the confirmation PASSES**, with one failure against a limit of six. The independent blind recomputation agrees. The trigger fired in 5 of the 80 runs (6.25 %, all at q = 50) and all five second attempts pass; with every fired run counted as a failure of its first attempt, 74 of 80 pass. The one failure (q = 60, seed 80502) is a stage-2 near-zero stretch in the stage-3 clamp regime, class (K), which the stage-3 trigger cannot see (its `Delta` at local update 1600 is 0.0008). The restart costs 1.026 times the updates of a run that does not fire.

## 1. The checks

The checks are those of prompt 3.2, run by `tools/t3/t3r4_checks.py` (through `tools/t3/t3r4_post_launch.sh`, which stops at the first nonzero exit; it ran to its end) on all 80 runs: exit codes; complete files (a run that fired also has `retry.json` and a complete `attempt1/`; a run that did not has neither); the manifests carry the launch commit and a clean tree; the global-RNG assertions; each config equals the plan and the lock; the budgets and schedules (not fired: 3,800 updates, 1,945,600 policy rows, 76,000 minibatch steps, the learning rate at every update, 152 exports, 64 / 64 / 24 checks; fired: 5,400 updates, 2,764,800 policy rows, 108,000 minibatch steps, 216 checks, 64 exports and 64 checks in `attempt1/`); the trigger recomputed from the check file it read and compared with `retry.json` (both attempts); C-INIT (attempt 1 and, for a fired run, attempt 2); C-REPLAY (the evaluator reproduces each run's freeze arrays bit for bit).

{{table:conf_checks}}

**The launch.** Launched by `tools/t3/t3r4_launch.py --workers 40` in tmux (session `t3r4_conf`, `results/t3_r4/launch_confirmation.sh`) at {{launch_start}} UTC from the LOCK-record commit `{{launch_commit}}`, ended {{launch_end}} UTC (`results/t3_r4/confirmation/launch_start.txt`, `launch_end.txt`); `launch_machine.txt` records, 21 s before the launch, `nproc` 64, load average 8.9 / 11.5 / 12.9, 789 GB free on `/home`, HEAD and an empty `git status --porcelain` outside `results/`. The launcher itself checked, and passed, every guard of prompt 2.1 (the T3-1.1 record last in `protocols/LOCK`, the lock files and every listed file by SHA-256, HEAD equal to the LOCK-record commit, a clean tree, an empty diff of the frozen code paths since `b565f0e8`, an absolute root). Nothing was written outside `results/` between the launch and the end of the wave. The start-count diagnostic of the checks flags one of the five cells of one run (q = 50 seed 80536, |z| > 3) and `starts_ok` holds for all 80 runs. The evaluation needed no re-run (`eval/eval_reruns.json` is empty), and the infrastructure ledger `infra_events.json` holds no event: no run was repeated, replaced or re-drawn.

**Deviations from the prompt.** None that touches a result. The launch commit and the machine state were recorded by separate commands just before the launch (the tool environment refuses a launch script that calls `git`), and a first pass of C-REPRO with a weaker comparison was kept beside the final one (`02_checks.md` section 4).

## 2. The locked settings and the code version

- **Protocol T3-1.1** (`protocols/v2_T3_locked_v1_1.{json,md}`, lock commit `98355d8a`, JSON SHA-256 `{{json_sha}}`, MD SHA-256 `{{md_sha}}`; the record in `protocols/LOCK` is the commit `141b5189`, which is HEAD of the launch). It is protocol T3-1.0 (`protocols/v2_T3_locked_v1_0.json`, unchanged) plus the config key `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`. The lock lists the SHA-256 of {{n_listed}} files (`run/ms_retry.py`, `run/run_ms_stagewise.py` and every tool the confirmation runs).
- **Code.** `run/ms_retry.py` and `run/run_ms_stagewise.py` are the code of T3-R3 (`git diff --stat b565f0e8 HEAD -- agents envs utils run tools/ms tools/v2` is empty). Behaviour: the trigger is read at the development check at local update 1600 of the stage-3 phase and fires iff `Delta` > 0.005, or `Delta`, `s` or `C` is not finite, or `s` <= 0; a fired run re-initialises stage 3 once in place and retrains it, the second attempt is frozen whatever its own check says, the artefacts of attempt 1 go to `attempt1/`, and `retry.json` records the trigger values and the derived seeds; a run whose trigger does not fire is bit-identical to `bw` (C-REPRO re-ran T3-R3's `bwA` runs of seeds 70508 (fired) and 70501 (not fired) at the final code commit: both bit-identical to their originals, `02_checks.md` section 4).
- **Rule.** The confirmation passes iff at most 6 of the 80 runs fail and at most 4 of the 40 runs fail at each q. A run passes iff it exits with code 0, its global-RNG assertion is ok, its evaluation is valid on both tiers, and G-A (`eta_3` <= 0.005, `rmse_pos_T_over_eT0` <= 0.05, `tail_mean_T_over_eT0` <= 0.02), G-F (`Gmax_full_over_dw` <= 0.01) and G-N (development minus final, `eta_3` and `Gmax_full`, <= 0.001) hold, every `<=` inclusive at full float64 precision. Operating characteristic (independent per-run failure probability f at both q): the rule is met with probability 0.982 at f = 0.03, 0.865 at 0.05, 0.271 at 0.10 and 0.030 at 0.15 (`01_lock.md` section 2).
- **Paper wording for A (verbatim):** "a verifier-gated restart of the terminal stage: when the development-tier one-step deviation gain at the end of the stage-3 phase exceeds the gate's limit 0.005, stage 3 is re-initialised once and retrained, and the second attempt is kept; its cost is part of the method's cost."

## 3. Per-seed results

Each row is one run under the pass rule; `eta_3`, `rmse_pos` and `tail_mean` are the G-A quantities (the last two relative to the tie-point effort), `Gmax_full` the G-F quantity, in units of the prize gap where the gate says so; `S3` is the out-of-band clamp share of the frozen attempt; `[fired]` marks a run whose trigger fired (the row is the frozen second attempt).

**q = 50**

{{table:conf_perseed_q50}}

**q = 60**

{{table:conf_perseed_q60}}

## 4. The verdict under the rule

{{table:conf_verdict}}

{{table:conf_values}}

The rule is met with a wide margin: 1 failure against the limit 6 in all and the limit 4 at each q. The exact 95 % interval of the failure rate is [0.000, 0.068]. Among the 79 passing runs the closest approach to a limit is `eta_3` = 0.004897 (98 % of 0.005; q = 60 seed 80528), then `rmse_pos` = 0.0454 (91 % of 0.05; q = 60 seed 80536); 6 runs have `eta_3` and 8 runs have `rmse_pos` above 80 % of their limits; `tail_mean`, `Gmax_full` and the development-final differences reach at most 47.1 %, 49 % and 24.1 % of theirs (`analysis/per_run.csv`).

## 5. The trigger and the first-attempt outcome

{{table:conf_trigger}}

All five fires are at q = 50 (5 of 40 = 0.125), none at q = 60 (0 of 40); the trigger values are `Delta` = 0.0052-0.0094 against the limit 0.005. In four of the five runs attempt 1 was in the stage-3 clamp regime (S3 of 0.46-0.78) and the frozen attempt is out of it in three (S3 0.010, 0.0004, 0.056) and still above 0.2 in one (seed 80535: 0.459 to 0.267); seed 80532 is the other way round: attempt 1 was outside the regime (S3 = 0.060, `Delta` = 0.0094) and the frozen attempt is inside it (S3 = 0.78) and passes with `eta_3` = 0.004663, 93 % of the limit. So a restart does not need to leave the regime to pass, and a restart can enter it. Three runs that did not fire are within 5 % of the trigger limit (`Delta` = 0.004897 for q = 60 seed 80528, 0.004825 for q = 50 seed 80510, 0.004758 for q = 50 seed 80520); all three pass. The fact-check re-evaluated the stage-3 network of attempt 1 of the five fired runs on the final tier: `eta_3` = 0.0064, 0.0075, 0.0052, 0.0094 and 0.0067, all above 0.005, so each first attempt would have failed G-A (`pi_record/01_factcheck.md`). Cost of the fires: section 7.

**First-attempt outcome.** The pass count with every fired run counted as a failure of its first attempt (the convention of the G1 reply of round T3-R3; a fired run's first attempt is the run `bw` would have produced up to the check) is **74 of 80** (0.925, exact 95 % interval [0.844, 0.972]; q = 50: 35 of 40, q = 60: 39 of 40). It is the `bw`-equivalent pass rate of this block: the 79 passes of `bwA` are the 74 first-attempt passes plus the 5 fired runs whose second attempt passes.

## 6. The failure, classified

{{table:conf_failures}}

The only failing run is q = 60 seed 80502 (`gate_G_F` and `gate_G_N`). Its stage-3 quantities pass (`eta_3` = 0.00093, `rmse_pos` = 0.0214, `tail_mean` = 0.0032: G-A holds), so the last-16-export table of the failing stage-3 quantity is empty (no stage-3 quantity failed). Over its last 16 stage-3 exports `rmse_pos` reaches 0.0505 (above its limit 0.05 at one export, the final export being 0.0214) and `eta_3` reaches 0.0042, both at update 1300 (`analysis/last16_summary.csv`; the table was computed for the failing run only, so there is no comparison with the passing runs). The failing quantity lives at **stage 2**: the maximal one-step gain `Gmax_full` = 0.0586 at d = -14 (limit 0.01), and the development-final difference of `Gmax_full` is 0.0010748 (limit 0.001, by 7.5 %). A **stage-2 near-zero stretch** of 5 in-band nodes (the longest at d in [-22, -14]) coincides with it: the maximal gain is at d = -14, the end of the stretch. **Class (K)** of T3-R3's D2: S3 of the frozen attempt = 0.536 (above 0.2: the run is in the clamp regime) and the failing quantity is a stage-2 near-zero stretch. It is not (F) (F(u01600) = 54.0 against the threshold 26.8 of q = 60: the policy formed), not (V) (no stage-3 quantity fails), and I found no implementation error (E): the run passes C-INIT and C-REPLAY like the others and, its trigger not having fired, it is by construction (C-REPRO) the `bw` run of its seed. **The trigger does not see it:** `Delta` at local update 1600 = 0.0008 (limit 0.005). It is the second failure of this kind among the 120 `bwA` runs of T3-R3's development and this confirmation (T3-R3 development: q = 60 seed 70511, stage-2 stretch of 28 nodes, also unreached by A), both at q = 60.

Other descriptive facts: 28 of the 80 runs are in the clamp regime (S3 of the frozen attempt > 0.2) and 27 of them pass; 3 runs have a stage-2 stretch (14, 8 and 5 nodes) and 2 of them pass (the stretches of the passing runs carry `Gmax_full` of 0.0045 and 0.0046).

## 7. The cost

{{table:conf_cost}}

A run that does not fire costs 3,800 updates, 1,945,600 policy rows, 76,000 minibatch steps and 152 development checks; a fired run 5,400 updates, 2,764,800 rows, 108,000 steps and 216 checks (a factor 1.421; pre-registered in `reports/t3/r3/04_preregistration.md` section 4). With 5 fires in 80 runs the mean is **3,900 updates (1.026 times), 2.0e6 policy rows, 78,000 minibatch steps and 156 checks**; the pre-registered expectation at the recorded fire rate of 7 of 80 was 1.037, and the observed fire rate 0.0625 lies inside the exact 95 % interval [0.036, 0.172] of that rate (`reports/t3/r3/04_preregistration.md` section 4). At q = 50, where all fires are, the factor is 1.053. **Measured wall time** (one concurrent wave of 40 workers, single-threaded runs on a machine shared with other jobs: the launch record `launch_20261022_071148.json` gives load averages (1 / 5 / 15 min) of 8.6 / 11.2 / 12.8 at its start and 14.8 / 33.3 / 35.3 at its end; `launch_machine.txt`, taken 21 s before the launch, has 8.9 / 11.5 / 12.9): 597.6 s per run on average (median 591.3 s), 47,810 s of summed run wall time for the wave (13.3 hours; 12.1 hours of recorded training-phase CPU time), 22.9 minutes at 40 workers; a fired run took 801.2 s on average (766-853 s) against 584.1 s for a run that did not fire, 217 s or 37 % more, in line with the stage-3 phase of about 235 s (`reports/t3/r3/04_preregistration.md` section 4) and with T3-R3's 36 %. The share of fired runs is 0.0625 (5 of 80).

## 8. Descriptive comparisons (they decide nothing)

{{table:conf_comparisons}}

- This confirmation pooled with `bwA`'s 40 development runs of T3-R3 (39 of 40): **118 of 120 pass** (0.983, exact 95 % interval [0.941, 0.998]; q = 50: 60 of 60, q = 60: 58 of 60).
- T3-1.0's three blocks of `bw`: 36 of 40 (T3-R1 development), 34 of 40 (T3-R2 confirmation, which failed its rule), 37 of 40 (T3-R3's development control); pooled **107 of 120** (0.892, [0.822, 0.941]; q = 50: 52 of 60, q = 60: 55 of 60). The 118 of 120 of `bwA` is 11 passes more than 107 of 120 on the same number of runs; the blocks have different seeds, so this is a description, not a test.
- Held to the same rule, the first-attempt counts (5 failures at q = 50, 1 at q = 60, 6 in all) would meet the limit of 6 in all exactly and exceed the limit of 4 at q = 50 (5 > 4); this is a description under the first-attempt convention, not a test.
- The pass rate of T3-1.0 implied by the first-attempt outcome: **74 of 80 = 0.925** [0.844, 0.972], in the range of the three `bw` blocks (0.850-0.925); pooled with T3-R3's development control (`bw` 37 of 40, whose first-attempt count under the same convention is also 37) it is 111 of 120.

## 9. What the confirmation shows and what it does not

- The pre-registered rule is met: at most 6 failures of 80 and at most 4 of 40 per q, with one failure. The checks, the blind recomputation and the lock guards all hold, and no run was repeated or replaced.
- The restart is exercised: 5 of 80 runs fired and all five pass on their second attempt; it costs 2.6 % more updates (3.7 % at the pre-registered expectation). This block has no control arm: a second attempt is a fresh draw, and a fresh draw passes at the block's base rate, so five of five second attempts passing is not by itself a measure of an effect beyond that of a second draw.
- It acts at q = 50 in these 120 `bwA` runs (7 fires of 60 runs at q = 50 in the development and this confirmation, none of 60 at q = 60, apart from the diagnostic run of seed 60518 in T3-R3), and it does not reach the failure type that remains: a stage-2 near-zero stretch in the clamp regime at q = 60 (2 of the 60 `bwA` runs at q = 60 of the two blocks).
- This is one confirmation block of 80 runs with one failure; the mechanism by which the restart helps is not tested here (T3-R3's diagnosis found the clamp regime a marker, not shown to be a cause, and a restart that stays in the regime still passes); nothing was tuned, and no other scheme was tried.

## 10. Files

- Lock and plan: `protocols/v2_T3_locked_v1_1.{json,md}`, the record in `protocols/LOCK`, `reports/t3/r4/01_lock.md`, `confirmation_plan.csv`; checks of the tools: `02_checks.md`.
- The wave: `results/t3_r4/confirmation/q{q}/seed{s}/bwA/` (tracked: manifest, config, status, gates, rule log, drift test, summary, check and update CSVs, bin maps, continuation tables, and for fired runs `retry.json` and the small records of `attempt1/`; untracked: weights, `.pt`, freeze and evaluation arrays, train histories), `launch_*.json`, `launch_commit.txt`, `launch_machine.txt`, `launch_start.txt`, `launch_end.txt`, `launch_stdout.txt`, `infra_events.json`, `launch_checks.json`, `post_launch_stdout.txt`.
- Evaluation and analysis: `results/t3_r4/confirmation/eval/per_run.csv`, `eval_reruns.json`, `analysis/` (`headline.json`, `blind.json`, `per_run.csv`, `counts.csv`, `fired.csv`, `failures.csv`, `cost.csv`, `comparisons.csv`, `last16.csv`, `last16_summary.csv`).
- Tools: `tools/t3/t3r4_{launch,make_lock,checks,eval,analysis,blind,repro,post_launch,preflight_seeds}.*`; tests `tests/test_t3r4_*.py`.
