# T3-R4: summary (the confirmation of protocol T3-1.1, `bwA`)

Date: {{date}}. Branch `t3-r4`. Generated from `reports/t3/r4/report_scripts/summary.template.md`; the full report is `03_confirmation.md`, the lock `01_lock.md`, the checks of the tools `02_checks.md`. **STOP: the confirmation is done and passed; nothing was changed or tried after the lock; the next step is the PI's.**

## The verdict

Protocol T3-1.1 is T3-1.0 (the locked `bw`) plus the verifier-gated restart of stage 3 (`stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`); it was locked at `98355d8a` (record `141b5189`, pushed before the launch) and confirmed on the fresh seeds 80501-80540 at q = 50 and q = 60 (80 runs, one wave, {{launch_start}}-{{launch_end}} UTC).

{{table:conf_verdict}}

- **The confirmation PASSES** the rule fixed in T3-R3's pre-registration (at most 6 of the 80 runs fail and at most 4 of the 40 at each q): **79 of 80 pass, one failure** (q = 50: 40 of 40; q = 60: 39 of 40; exact 95 % interval of the pass rate [0.932, 1.000]). The independent blind recomputation agrees.
- **All checks pass** on all 80 runs (exit codes, files, manifests at the launch commit with a clean tree, global-RNG assertions, configs against the plan, budgets and schedules, init digests, bit-exact replay of the freeze arrays); no infrastructure event; no run repeated, replaced or re-drawn.
- **The trigger** fired in **5 of 80 runs (6.25 %**, exact 95 % interval [0.021, 0.140]; all five at q = 50, none at q = 60), and **all five second attempts pass**. With every fired run counted as a failure of its first attempt, **74 of 80 pass** (0.925), the `bw`-equivalent rate of this block.
- **The one failure** (q = 60, seed 80502; `gate_G_F` and `gate_G_N`) is a stage-2 near-zero stretch (5 nodes, `Gmax_full` = 0.0586 at d = -14) in the stage-3 clamp regime (S3 = 0.536): class (K) of T3-R3's D2. The stage-3 trigger cannot see it (`Delta` at local update 1600 = 0.0008, limit 0.005), and its stage-3 quantities pass.
- **Cost:** a run that does not fire costs 3,800 updates; a fired run 5,400 (1.421 times). The mean over the 80 runs is 3,900 updates (**1.026 times**), 2.0e6 policy rows, 156 development checks. Measured wall time: 597.6 s per run on average, 801 s for a fired run against 584 s for one that did not (+37 %); the wave took 22.9 minutes at 40 workers (13.3 hours of summed run wall time).
- **Descriptive comparisons** (they decide nothing): this block pooled with `bwA`'s 40 development runs: 118 of 120 (0.983, [0.941, 0.998]); T3-1.0's three `bw` blocks: 36, 34 and 37 of 40, pooled 107 of 120 (0.892, [0.822, 0.941]); the first-attempt-implied pass rate of T3-1.0 in this block: 74 of 80 (0.925, [0.844, 0.972]).

## What to put in the paper about A

- **The wording** (`protocols/v2_T3_locked_v1_1.md`): "a verifier-gated restart of the terminal stage: when the development-tier one-step deviation gain at the end of the stage-3 phase exceeds the gate's limit 0.005, stage 3 is re-initialised once and retrained, and the second attempt is kept; its cost is part of the method's cost."
- **Report with it:** the trigger rate (5 of 80 in the confirmation, 7 of 120 over development and confirmation at q = 50 and q = 60 together, none at q = 60), the cost (1.026 times the updates; a fired run 1.42 times the updates and about 1.37 times the wall time), the confirmation outcome (79 of 80, rule met with one failure against six), the first-attempt rate (74 of 80) beside T3-1.0's three blocks, and that the rule and the seeds were fixed before the development runs and the lock.
- **State the limits:** the restart acts on failures that show at the stage-3 check (this block has no control arm, and a second attempt is a fresh draw, so five of five second attempts passing is not by itself a measure of an effect beyond that of a second draw); the remaining failure type, a stage-2 near-zero stretch at q = 60, is not reached by it (2 of the 60 `bwA` runs at q = 60 over development and confirmation); whether and how the restart helps beyond a second draw is not established (a restart can stay inside the clamp regime, or enter it, and pass).

## Flagged

1. **All trigger fires are at q = 50** (7 of 60 runs over development and confirmation; 0 of 60 at q = 60, apart from T3-R3's diagnostic run of seed 60518). At q = 60 the restart has not been needed in 60 `bwA` runs, and the failures that remain there are outside its reach.
2. **Margins.** Among the 79 passing runs the closest approach to a gate is `eta_3` = 0.004897 (98 % of 0.005; q = 60 seed 80528) and `rmse_pos` = 0.0454 (91 % of 0.05); six runs have `eta_3` above 80 % of its limit. A fired run passed with `eta_3` = 0.004663 (seed 80532, whose second attempt is in the clamp regime).
3. **The tools were reviewed twice** by an independent agent (no blocker; 18 + 9 findings, all dispositioned in `02_checks.md`); the pre-lock rehearsal on 80 symlinked production runs found no defect; C-REPRO re-ran T3-R3's fired (seed 70508) and unfired (seed 70501) `bwA` runs at the final code commit, bit-identical to their originals; 109 new tests, the full suite gives 1,928 passed, 6 skipped, 2 xfailed, 2 failed (the two known failures).
4. **No deviation from the prompt** that touches a result (`03_confirmation.md` section 1). The one-failure outcome is a single block of 80 runs.

## For the PI

The prompt ends here: the confirmation passed, so nothing in this round is to be changed or tried. The next step (the paper, a second confirmation block, or anything else) is yours. `reports/t3/r4/handoff.md` is the Chinese note for the coworker.
