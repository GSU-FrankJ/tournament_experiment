# T3-R4, P1: the lock drafts (protocol T3-1.1) and the plan of the confirmation

Date: {{date}}. Branch `t3-r4` (from `origin/t3-r3` `df2365d9`). **Generated** from `reports/t3/r4/report_scripts/01_lock.template.md` by `build_report.py`; the tables by `lock_tables.py`. This file is committed **before** the lock (section 9). The binding text is the prompt (`pi_record/29_t3_r4_prompt.md`); the decision to lock `bwA` is the PI's at gate G2 of round T3-R3 (`reports/t3/r3/summary.md`, `05_development.md`: `bw` 37 of 40, `bwA` 39 of 40, `bwAB` 38 of 40 on development seeds 70501-70520).

## 1. What is locked

**Protocol T3-1.1 = protocol T3-1.0 plus one config key.** T3-1.0 (`protocols/v2_T3_locked_v1_0.{json,md}`, unchanged) is the locked `bw`: backward-induction training of stage 3, then stage 2, then stage 1 through `run/run_ms_stagewise.py`. T3-1.1 adds `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}` (`run/ms_retry.py`, `run/run_ms_stagewise.py`). The code is the code of round T3-R3: the diff `git diff --stat b565f0e8 df2365d9 -- agents envs utils run tools/ms tools/v2` is empty (`00_housekeeping.md` section 1), and the launcher checks the same diff against HEAD at every launch. Behaviour (`reports/t3/r3/02_proposal.md` section 2.1): the trigger is read at the development check at local update 1600 of the stage-3 phase and fires iff `Delta` > 0.005, or `Delta`, `s` or `C` is not finite, or `s` <= 0; a fired run re-initialises stage 3 once in place (derived seeds and streams) and retrains it, and the second attempt is frozen whatever its own check says; the artefacts of attempt 1 go to `attempt1/` and `retry.json` records the trigger values and the derived seeds; a run whose trigger does not fire is bit-identical to `bw`. Budget of a fired run: 5,400 updates, 2,764,800 policy rows, 108,000 minibatch steps, 216 development checks, 152 exports plus 64 in `attempt1/`.

**Paper wording for A (verbatim):** "a verifier-gated restart of the terminal stage: when the development-tier one-step deviation gain at the end of the stage-3 phase exceeds the gate's limit 0.005, stage 3 is re-initialised once and retrained, and the second attempt is kept; its cost is part of the method's cost."

The files (`reports/t3/r4/lock_draft/v2_T3_locked_v1_1.{json,md}`, written by `tools/t3/t3r4_make_lock.py`; SHA-256 of the JSON `{{json_sha}}`, of the MD `{{md_sha}}`) hold no timestamp, no commit hash and no absolute path; the lock written into `protocols/` must equal them byte for byte.

{{table:lock_settings}}

## 2. The confirmation rule, the operating characteristic and the failure counting

The rule was fixed before any development run, in T3-R3's pre-registration (`reports/t3/r3/04_preregistration.md` section 9, D6), and is binding: **the confirmation passes iff at most 6 of the 80 runs fail and at most 4 of the 40 runs fail at each q** (seeds 80501-80540, the same 40 seeds at q = 50 and q = 60). A run passes iff it exits with code 0, its global-RNG assertion is ok, its evaluation is valid on both tiers, and G-A (`eta_3` <= 0.005, `rmse_pos_T_over_eT0` <= 0.05, `tail_mean_T_over_eT0` <= 0.02), G-F (`Gmax_full_over_dw` <= 0.01) and G-N (development minus final: `eta_3` and `Gmax_full` <= 0.001) hold, every `<=` inclusive at full float64 precision (`reports/t3/r1/prereg_criteria.json`).

Operating characteristic, recomputed here with `scipy.stats.binom` (not with `t3r4_make_lock.py`) and set against the values in the lock; independent per-run failure probability f, the same at both q:

{{table:lock_oc}}

(The prompt quotes 0.982 at f = 0.03, 0.865 at 0.05, 0.271 at 0.10 and 0.030 at 0.15.)

**Failure counting** (in the lock, section 5): every planned run is reported and no seed is replaced or re-drawn. Infrastructure failures (a negative return code, an out-of-memory kill, ENOSPC) are recorded with evidence in `results/t3_r4/confirmation/infra_events.json`; such a run is moved to `crashed/` and re-run once, and a second infrastructure failure stops the confirmation without a verdict. Any other nonzero exit and any global-RNG violation is a failed run. An evaluation error or an invalid evaluation is a failed run unless a re-run of the locked evaluator on the same weights succeeds. A re-initialisation inside a run is part of the run, never a re-run. If the confirmation fails, nothing is changed or tried in this round: the PI decides the next step.

## 3. The tools, and what the launcher refuses

`tools/t3/t3r4_launch.py` (arm `bwA` only; seeds exactly 80501-80540 at q in {50, 60}); `t3r4_checks.py` (the post-launch checks), `t3r4_eval.py`, `t3r4_analysis.py`, `t3r4_blind.py`, `t3r4_post_launch.sh` (checks, evaluation, analysis and blind recomputation, stopping at the first nonzero exit), `t3r4_make_lock.py`, `t3r4_repro.py` (C-REPRO). They reuse by import, unchanged, the `t3_*`, `t3c_*`, `t3r2_*` and `t3r3_*` tools; the lock lists the SHA-256 of every file the confirmation runs. At a launch the launcher refuses unless: the T3-1.1 record is the last record of `protocols/LOCK`; the lock JSON and MD match its SHA-256; every listed file matches; HEAD is the LOCK-record commit; `git status --porcelain` shows nothing outside `results/`; the diff of `agents envs utils run tools/ms tools/v2` since `b565f0e8` is empty; `--root` is absolute; each config equals the T3-1.1 reference but for the six run fields. Tests: `tests/test_t3r4_*.py` (see `02_checks.md`).

## 4. The plan of the 80 runs

`reports/t3/r4/confirmation_plan.csv` (written by `t3r4_launch.py --plan-csv`, which checks every config against the lock drafts; SHA-256 of the file `{{plan_sha}}`): run name, output directory, config SHA-256 (as written and with the output directory made relative to the root), the planned init digest of attempt 1 and of attempt 2 (the run is constructed without training). The post-launch checks compare the recorded digests with these.

{{table:lock_plan}}

## 5. Config differences against T3-R3's `bwA` configs of the same q

A confirmation config differs from T3-R3's `bwA` development config of the same q in the six run fields only:

{{table:lock_cfgdiff}}

(The launcher also tests this for all 80 configs against the T3-1.1 reference: `tests/test_t3r4_launch.py`.)

## 6. Seed inventory (P0, `00_housekeeping.md` section 3)

{{table:lock_seeds}}

## 7. C-REPRO (at the final code commit)

T3-R3's `bwA` development runs (q = 50, seed 70508), whose trigger fired, and (q = 60, seed 70501), whose trigger did not, re-run from their own `run_config.json` with `tools/t3/t3r4_repro.py` and compared with their originals in the T3-R3 untracked root (`tools/t3/compare_runs.py bw`, plus `attempt1/` and `retry.json` for 70508):

{{table:lock_repro}}

## 8. Rehearsal

Before the lock, the post-launch tools were run once on 80 run directories whose files are symlinks to T3-R3's 40 `bwA` development runs (`results/t3_r4/rehearsal/`, mapped to the seeds 80501-80540; not a lock input): the evaluation, the analysis, the blind recomputation and the report tables run end to end on production-size runs, including four fired runs; the budgets, schedules, exports, checks, retry budgets, trigger and C-REPLAY checks pass, and the config / plan / init / clean-tree verdicts fail by construction. The details are in `02_checks.md`.

## 9. The steps that follow

1. The lock: `t3r4_make_lock.py --out protocols/` (byte-identical to the drafts), committed alone.
2. The LOCK record appended to `protocols/LOCK` (protocol T3-1.1, the lock commit, the JSON and MD SHA-256, the date; entry `run/run_ms_stagewise.py` through `tools/t3/t3r4_launch.py`; reason "PI decision at gate G2 of round T3-R3: lock `bwA`"; the rule of change), committed separately; both commits pushed.
3. The wave of 80 runs from the LOCK-record commit (`results/t3_r4/confirmation/`), then `t3r4_post_launch.sh`.
