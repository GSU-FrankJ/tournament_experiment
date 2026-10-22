# T3-R4: the confirmation of protocol T3-1.1 (`bwA`) on fresh seeds

Date: 2026-10-22. Branch `t3-r4`. **Generated** from `reports/t3/r4/report_scripts/03_confirmation.template.md` by `build_report.py`; every table by `reports/t3/r4/report_scripts/confirmation_tables.py` from `results/t3_r4/confirmation/launch_checks.json`, `eval/per_run.csv` and `analysis/` (written by `tools/t3/t3r4_checks.py`, `t3r4_eval.py`, `t3r4_analysis.py`, `t3r4_blind.py`). The binding text is the prompt (`pi_record/29_t3_r4_prompt.md`); the rule was fixed before any development run in T3-R3's pre-registration (`reports/t3/r3/04_preregistration.md` section 9, D6). Nothing was changed or tuned after the lock.

**In one paragraph.** One wave of 80 runs of the locked protocol T3-1.1 (the arm `bwA`: `bw` plus the verifier-gated restart of stage 3) on the fresh seeds 80501-80540 at q = 50 and q = 60 ran to completion: all 80 exit codes 0, no infrastructure event, every post-launch check passes. **79 of the 80 runs pass (q = 50: 40 of 40; q = 60: 39 of 40; exact 95 % interval of the pass rate [0.932, 1.000]). The rule (at most 6 failures of 80 and at most 4 of 40 at each q) is met: the confirmation PASSES**, with one failure against a limit of six. The independent blind recomputation agrees. The trigger fired in 5 of the 80 runs (6.25 %, all at q = 50) and all five second attempts pass; with every fired run counted as a failure of its first attempt, 74 of 80 pass. The one failure (q = 60, seed 80502) is a stage-2 near-zero stretch in the stage-3 clamp regime, class (K), which the stage-3 trigger cannot see (its `Delta` at local update 1600 is 0.0008). The restart costs 1.026 times the updates of a run that does not fire.

## 1. The checks

The checks are those of prompt 3.2, run by `tools/t3/t3r4_checks.py` (through `tools/t3/t3r4_post_launch.sh`, which stops at the first nonzero exit; it ran to its end) on all 80 runs: exit codes; complete files (a run that fired also has `retry.json` and a complete `attempt1/`; a run that did not has neither); the manifests carry the launch commit and a clean tree; the global-RNG assertions; each config equals the plan and the lock; the budgets and schedules (not fired: 3,800 updates, 1,945,600 policy rows, 76,000 minibatch steps, the learning rate at every update, 152 exports, 64 / 64 / 24 checks; fired: 5,400 updates, 2,764,800 policy rows, 108,000 minibatch steps, 216 checks, 64 exports and 64 checks in `attempt1/`); the trigger recomputed from the check file it read and compared with `retry.json` (both attempts); C-INIT (attempt 1 and, for a fired run, attempt 2); C-REPLAY (the evaluator reproduces each run's freeze arrays bit for bit).

| check | pass | fail | not evaluated |
|---|---|---|---|
| exit_ok | 80 | 0 | 0 |
| files_complete | 80 | 0 | 0 |
| manifest_commit_ok | 80 | 0 | 0 |
| clean_tree | 80 | 0 | 0 |
| global_rng_ok | 80 | 0 | 0 |
| config_equals_plan | 80 | 0 | 0 |
| config_production_ok | 80 | 0 | 0 |
| trigger_consistent | 80 | 0 | 0 |
| budget_ok | 80 | 0 | 0 |
| lr_ok | 80 | 0 | 0 |
| refresh_ok | 80 | 0 | 0 |
| exports_ok | 80 | 0 | 0 |
| checks_ok | 80 | 0 | 0 |
| retry_budget_ok | 80 | 0 | 0 |
| init_equals_plan | 80 | 0 | 0 |
| replay_ok | 80 | 0 | 0 |

Runs checked: 80; trigger fired: 5; trigger would fire (recomputed from the check file it read): 5; check errors: 0. Infrastructure ledger present: True; events: 0; stopped without verdict: [].

**The launch.** Launched by `tools/t3/t3r4_launch.py --workers 40` in tmux (session `t3r4_conf`, `results/t3_r4/launch_confirmation.sh`) at 07:11:46 UTC from the LOCK-record commit `141b5189`, ended 07:34:42 UTC (`results/t3_r4/confirmation/launch_start.txt`, `launch_end.txt`); `launch_machine.txt` records, 21 s before the launch, `nproc` 64, load average 8.9 / 11.5 / 12.9, 789 GB free on `/home`, HEAD and an empty `git status --porcelain` outside `results/`. The launcher itself checked, and passed, every guard of prompt 2.1 (the T3-1.1 record last in `protocols/LOCK`, the lock files and every listed file by SHA-256, HEAD equal to the LOCK-record commit, a clean tree, an empty diff of the frozen code paths since `b565f0e8`, an absolute root). Nothing was written outside `results/` between the launch and the end of the wave. The evaluation needed no re-run (`eval/eval_reruns.json` is empty), and the infrastructure ledger `infra_events.json` holds no event: no run was repeated, replaced or re-drawn.

**Deviations from the prompt.** None that touches a result. The launch commit and the machine state were recorded by separate commands just before the launch (the tool environment refuses a launch script that calls `git`), and a first pass of C-REPRO with a weaker comparison was kept beside the final one (`02_checks.md` section 4).

## 2. The locked settings and the code version

- **Protocol T3-1.1** (`protocols/v2_T3_locked_v1_1.{json,md}`, lock commit `98355d8a`, JSON SHA-256 `97bb9146c379c957356706f463361c2ff48c2724fac31b9114e7aba5a2a7e38e`, MD SHA-256 `90a1047b745174209a0d05c8826cce868b8506d89cb1c1fc3239104ef0063c47`; the record in `protocols/LOCK` is the commit `141b5189`, which is HEAD of the launch). It is protocol T3-1.0 (`protocols/v2_T3_locked_v1_0.json`, unchanged) plus the config key `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`. The lock lists the SHA-256 of 33 files (`run/ms_retry.py`, `run/run_ms_stagewise.py` and every tool the confirmation runs).
- **Code.** `run/ms_retry.py` and `run/run_ms_stagewise.py` are the code of T3-R3 (`git diff --stat b565f0e8 HEAD -- agents envs utils run tools/ms tools/v2` is empty). Behaviour: the trigger is read at the development check at local update 1600 of the stage-3 phase and fires iff `Delta` > 0.005, or `Delta`, `s` or `C` is not finite, or `s` <= 0; a fired run re-initialises stage 3 once in place and retrains it, the second attempt is frozen whatever its own check says, the artefacts of attempt 1 go to `attempt1/`, and `retry.json` records the trigger values and the derived seeds; a run whose trigger does not fire is bit-identical to `bw` (C-REPRO re-ran T3-R3's `bwA` runs of seeds 70508 (fired) and 70501 (not fired) at the final code commit: both bit-identical to their originals, `02_checks.md` section 4).
- **Rule.** The confirmation passes iff at most 6 of the 80 runs fail and at most 4 of the 40 runs fail at each q. A run passes iff it exits with code 0, its global-RNG assertion is ok, its evaluation is valid on both tiers, and G-A (`eta_3` <= 0.005, `rmse_pos_T_over_eT0` <= 0.05, `tail_mean_T_over_eT0` <= 0.02), G-F (`Gmax_full_over_dw` <= 0.01) and G-N (development minus final, `eta_3` and `Gmax_full`, <= 0.001) hold, every `<=` inclusive at full float64 precision. Operating characteristic (independent per-run failure probability f at both q): the rule is met with probability 0.982 at f = 0.03, 0.865 at 0.05, 0.271 at 0.10 and 0.030 at 0.15 (`01_lock.md` section 2).
- **Paper wording for A (verbatim):** "a verifier-gated restart of the terminal stage: when the development-tier one-step deviation gain at the end of the stage-3 phase exceeds the gate's limit 0.005, stage 3 is re-initialised once and retrained, and the second attempt is kept; its cost is part of the method's cost."

## 3. Per-seed results

Each row is one run under the pass rule; `eta_3`, `rmse_pos` and `tail_mean` are the G-A quantities (the last two relative to the tie-point effort), `Gmax_full` the G-F quantity, in units of the prize gap where the gate says so; `S3` is the out-of-band clamp share of the frozen attempt; `[fired]` marks a run whose trigger fired (the row is the frozen second attempt).

**q = 50**

| seed | verdict | eta_3 | rmse_pos/e*(0) | tail_mean/e*(0) | Gmax_full | S3 (frozen) | stage-2 stretch nodes | wall (s) |
|---|---|---|---|---|---|---|---|---|
| 80501 | pass [fired] | 0.002128 | 0.0293 | 0.004623 | 0.002128 | 0.009967 | 0 | 804.8 |
| 80502 | pass | 0.001073 | 0.02785 | 0.003856 | 0.001073 | 0.3327 | 0 | 564.7 |
| 80503 | pass | 0.001483 | 0.01969 | 0.004412 | 0.001483 | 0.05337 | 0 | 621.5 |
| 80504 | pass | 0.001102 | 0.02225 | 0.005428 | 0.001115 | 0.09842 | 0 | 633.4 |
| 80505 | pass | 0.001139 | 0.02468 | 0.006038 | 0.001139 | 0.00052 | 0 | 589.8 |
| 80506 | pass | 0.002608 | 0.03209 | 0.005292 | 0.004557 | 0.01064 | 0 | 523.9 |
| 80507 | pass | 0.002574 | 0.02418 | 0.007777 | 0.002574 | 5.21e-05 | 0 | 539.7 |
| 80508 | pass | 0.000695 | 0.01962 | 0.005551 | 0.000695 | 0.01924 | 0 | 550.3 |
| 80509 | pass | 0.002059 | 0.0266 | 0.005195 | 0.002059 | 0.02524 | 0 | 587.6 |
| 80510 | pass | 0.004825 | 0.03902 | 0.002912 | 0.004825 | 0.5646 | 0 | 548.7 |
| 80511 | pass | 0.001234 | 0.02954 | 0.004756 | 0.001234 | 0.01563 | 0 | 592 |
| 80512 | pass | 0.001 | 0.02446 | 0.003229 | 0.001 | 0.5165 | 0 | 579.5 |
| 80513 | pass | 0.001498 | 0.02415 | 0.004912 | 0.001498 | 0.007374 | 0 | 620.5 |
| 80514 | pass | 0.001491 | 0.02525 | 0.002998 | 0.001491 | 0.5059 | 0 | 533.4 |
| 80515 | pass [fired] | 0.001906 | 0.03889 | 0.006913 | 0.001906 | 0.00039 | 0 | 852.9 |
| 80516 | pass | 0.001782 | 0.01917 | 0.003769 | 0.001782 | 0.376 | 0 | 574.1 |
| 80517 | pass | 0.002207 | 0.03081 | 0.006537 | 0.002207 | 0.000716 | 0 | 568.6 |
| 80518 | pass | 0.001487 | 0.02196 | 0.005161 | 0.001487 | 0.003932 | 0 | 597.8 |
| 80519 | pass | 0.003943 | 0.04369 | 0.003074 | 0.004469 | 0.7851 | 14 | 619.9 |
| 80520 | pass | 0.004758 | 0.04036 | 0.003851 | 0.004758 | 0.6677 | 0 | 628.3 |
| 80521 | pass | 0.00157 | 0.02876 | 0.00316 | 0.00157 | 0.3165 | 0 | 552.9 |
| 80522 | pass | 0.001197 | 0.02172 | 0.005227 | 0.001197 | 0.02079 | 0 | 553 |
| 80523 | pass | 0.004118 | 0.04355 | 0.003059 | 0.004118 | 0.7611 | 0 | 566.3 |
| 80524 | pass | 0.003279 | 0.02665 | 0.006336 | 0.003279 | 0.000312 | 0 | 630.9 |
| 80525 | pass | 0.003247 | 0.03276 | 0.005006 | 0.003247 | 0.003574 | 0 | 584 |
| 80526 | pass [fired] | 0.002516 | 0.02366 | 0.004822 | 0.002516 | 0.05593 | 0 | 766.2 |
| 80527 | pass | 0.002767 | 0.02686 | 0.004243 | 0.002767 | 0.2266 | 0 | 627.3 |
| 80528 | pass | 0.004615 | 0.04422 | 0.003291 | 0.004615 | 0.7725 | 8 | 612.4 |
| 80529 | pass | 0.001682 | 0.02519 | 0.006232 | 0.003311 | 0.05539 | 0 | 632.8 |
| 80530 | pass | 0.002011 | 0.02538 | 0.005553 | 0.002011 | 0.00596 | 0 | 573.7 |
| 80531 | pass | 0.001001 | 0.0184 | 0.004485 | 0.001001 | 0.04045 | 0 | 553.3 |
| 80532 | pass [fired] | 0.004663 | 0.04119 | 0.003066 | 0.004663 | 0.7799 | 0 | 780.2 |
| 80533 | pass | 0.002924 | 0.03554 | 0.004179 | 0.002924 | 0.01369 | 0 | 619.7 |
| 80534 | pass | 0.000722 | 0.01854 | 0.002791 | 0.000722 | 0.5692 | 0 | 630 |
| 80535 | pass [fired] | 0.002132 | 0.03045 | 0.004435 | 0.002132 | 0.2668 | 0 | 802 |
| 80536 | pass | 0.003715 | 0.03782 | 0.008423 | 0.003715 | 1.95e-05 | 0 | 609.4 |
| 80537 | pass | 0.002031 | 0.02991 | 0.003027 | 0.002031 | 0.4111 | 0 | 545.1 |
| 80538 | pass | 0.002468 | 0.02372 | 0.005672 | 0.002468 | 0.0102 | 0 | 548.3 |
| 80539 | pass | 0.003382 | 0.03186 | 0.002928 | 0.003382 | 0.5969 | 0 | 610.7 |
| 80540 | pass | 0.002119 | 0.02945 | 0.004453 | 0.002119 | 0.01716 | 0 | 563.5 |

**q = 60**

| seed | verdict | eta_3 | rmse_pos/e*(0) | tail_mean/e*(0) | Gmax_full | S3 (frozen) | stage-2 stretch nodes | wall (s) |
|---|---|---|---|---|---|---|---|---|
| 80501 | pass | 0.001544 | 0.02972 | 0.005427 | 0.001544 | 0.0042 | 0 | 612.2 |
| 80502 | FAIL (gate_G_F;gate_G_N) | 0.00093 | 0.02141 | 0.003178 | 0.05864 | 0.5358 | 5 | 593 |
| 80503 | pass | 0.001108 | 0.03337 | 0.002827 | 0.001108 | 0.7597 | 0 | 607.3 |
| 80504 | pass | 0.002137 | 0.02415 | 0.005671 | 0.002137 | 0.008028 | 0 | 601.7 |
| 80505 | pass | 0.001126 | 0.02357 | 0.007177 | 0.001126 | 0.003606 | 0 | 603.2 |
| 80506 | pass | 0.000844 | 0.01781 | 0.005738 | 0.000844 | 0.002317 | 0 | 598.9 |
| 80507 | pass | 0.001724 | 0.02061 | 0.005877 | 0.001724 | 0.009996 | 0 | 595.1 |
| 80508 | pass | 0.001146 | 0.02749 | 0.0075 | 0.001146 | 0.000758 | 0 | 648.4 |
| 80509 | pass | 0.001381 | 0.02317 | 0.006186 | 0.001381 | 0.003083 | 0 | 602.6 |
| 80510 | pass | 0.001487 | 0.02392 | 0.006015 | 0.001487 | 0.01311 | 0 | 629 |
| 80511 | pass | 0.00241 | 0.04345 | 0.003614 | 0.00241 | 0.7712 | 0 | 547.6 |
| 80512 | pass | 0.003378 | 0.03725 | 0.003266 | 0.003378 | 0.772 | 0 | 595.3 |
| 80513 | pass | 0.002583 | 0.0359 | 0.004034 | 0.002583 | 0.3201 | 0 | 638.2 |
| 80514 | pass | 0.001424 | 0.02297 | 0.0074 | 0.001424 | 0.000296 | 0 | 599.8 |
| 80515 | pass | 0.002221 | 0.02855 | 0.006145 | 0.002221 | 0.007565 | 0 | 617.2 |
| 80516 | pass | 0.000925 | 0.02521 | 0.006087 | 0.000925 | 0.01357 | 0 | 595.3 |
| 80517 | pass | 0.001938 | 0.02399 | 0.006729 | 0.001938 | 0.001774 | 0 | 592.4 |
| 80518 | pass | 0.003733 | 0.03886 | 0.004036 | 0.003733 | 0.7426 | 0 | 589.2 |
| 80519 | pass | 0.003941 | 0.03984 | 0.003596 | 0.003941 | 0.747 | 0 | 590.7 |
| 80520 | pass | 0.00101 | 0.02131 | 0.005933 | 0.00101 | 0.01302 | 0 | 588.2 |
| 80521 | pass | 0.001354 | 0.0211 | 0.006292 | 0.001354 | 0.005501 | 0 | 608.7 |
| 80522 | pass | 0.000603 | 0.01755 | 0.005818 | 0.000603 | 0.03582 | 0 | 597.8 |
| 80523 | pass | 0.001541 | 0.02148 | 0.006483 | 0.001541 | 0.009348 | 0 | 590.2 |
| 80524 | pass | 0.001435 | 0.02394 | 0.007293 | 0.001435 | 0.03483 | 0 | 545 |
| 80525 | pass | 0.001767 | 0.03664 | 0.002526 | 0.001767 | 0.7183 | 0 | 563.7 |
| 80526 | pass | 0.00188 | 0.0285 | 0.003253 | 0.00188 | 0.5954 | 0 | 603.1 |
| 80527 | pass | 0.002476 | 0.02988 | 0.007361 | 0.002476 | 0.009333 | 0 | 601.5 |
| 80528 | pass | 0.004897 | 0.04484 | 0.003392 | 0.004897 | 0.7586 | 0 | 565.9 |
| 80529 | pass | 0.001508 | 0.02908 | 0.009418 | 0.001508 | 0 | 0 | 545.4 |
| 80530 | pass | 0.001771 | 0.02933 | 0.00657 | 0.001771 | 0.01506 | 0 | 576.1 |
| 80531 | pass | 0.001555 | 0.02329 | 0.007711 | 0.001555 | 0.03 | 0 | 576.5 |
| 80532 | pass | 0.001097 | 0.02823 | 0.005004 | 0.001097 | 0.02345 | 0 | 585.7 |
| 80533 | pass | 0.001068 | 0.02288 | 0.006817 | 0.001068 | 0.0301 | 0 | 560.6 |
| 80534 | pass | 0.000803 | 0.01839 | 0.00623 | 0.000803 | 0.0275 | 0 | 558.6 |
| 80535 | pass | 0.001559 | 0.0263 | 0.005453 | 0.001559 | 0.06251 | 0 | 574.2 |
| 80536 | pass | 0.002881 | 0.0454 | 0.003605 | 0.002881 | 0.7626 | 0 | 524.4 |
| 80537 | pass | 0.002159 | 0.02808 | 0.002982 | 0.002159 | 0.6358 | 0 | 545.1 |
| 80538 | pass | 0.001301 | 0.03204 | 0.008662 | 0.001301 | 2.02e-05 | 0 | 522.2 |
| 80539 | pass | 0.00071 | 0.01971 | 0.006901 | 0.00071 | 0.000458 | 0 | 517.6 |
| 80540 | pass | 0.002529 | 0.03286 | 0.006773 | 0.002529 | 0.003021 | 0 | 509.9 |

## 4. The verdict under the rule

| q | runs | pass | fail | limit on failures | exact 95 % interval of the pass rate | condition met |
|---|---|---|---|---|---|---|
| 50 | 40 | 40 | 0 | 4 | [0.912, 1.000] | yes |
| 60 | 40 | 39 | 1 | 4 | [0.868, 0.999] | yes |
| both | 80 | 79 | 1 | 6 | [0.932, 1.000] | yes |

**Verdict under the rule: PASS.** at most 6 of the 80 runs fail and at most 4 of the 40 at each q. Blind recomputation (`tools/t3/t3r4_blind.py`): verdict pass, failures 0 + 1 = 1; agrees with the analysis: True; differences: [].

| q | runs | pass | worst Gmax_full (run) | median eta_3 | median rmse_pos/e*(0) | median tail_mean/e*(0) | median peak_T_signed | median Gmax_full | median EXP_root |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 40 | 40 | 0.00483 (t3r4_q50_s80510_bwA) | 0.002089 | 0.02735 | 0.004554 | -0.07421 | 0.002124 | 0.000462 |
| 60 | 40 | 39 | 0.05864 (t3r4_q60_s80502_bwA) | 0.001543 | 0.0269 | 0.005974 | -0.08694 | 0.00155 | 0.000276 |
| both | 80 | 79 | 0.05864 (t3r4_q60_s80502_bwA) | 0.001769 | 0.02717 | 0.005211 | -0.08297 | 0.001831 | 0.000372 |

The rule is met with a wide margin: 1 failure against the limit 6 in all and the limit 4 at each q. The exact 95 % interval of the failure rate is [0.000, 0.068]. Among the 79 passing runs the closest approach to a limit is `eta_3` = 0.004897 (98 % of 0.005; q = 60 seed 80528), then `rmse_pos` = 0.0454 (91 % of 0.05; q = 60 seed 80536); 6 runs have `eta_3` and 8 runs have `rmse_pos` above 80 % of their limits; `tail_mean`, `Gmax_full` and the development-final differences stay below 50 %, 49 % and 24 % of theirs (`analysis/per_run.csv`).

## 5. The trigger and the first-attempt outcome

Trigger fired in 5 of 80 runs (rate 0.0625, exact 95 % interval [0.0206, 0.1399]); 5 of the 5 fired runs pass on their frozen second attempt. **First-attempt outcome** (every fired run counted as a failure of its first attempt): 74 of 80 pass (rate 0.9250).

| run | q | trigger Delta | s | C | reason | S3 of attempt 1 | S3 of the frozen attempt 2 | attempt-2 final Delta | outcome | eta_3 | Gmax_full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `t3r4_q50_s80501_bwA` | 50 | 0.006397 | 68.28 | 0.04354 | delta_gt | 0.761 | 0.009967 | 0.002072 | pass | 0.002128 | 0.002128 |
| `t3r4_q50_s80515_bwA` | 50 | 0.007393 | 66.2 | 0.04337 | delta_gt | 0.7243 | 0.00039 | 0.001892 | pass | 0.001906 | 0.001906 |
| `t3r4_q50_s80526_bwA` | 50 | 0.005173 | 65.97 | 0.04346 | delta_gt | 0.7754 | 0.05593 | 0.002444 | pass | 0.002516 | 0.002516 |
| `t3r4_q50_s80532_bwA` | 50 | 0.009369 | 68.78 | 0.03264 | delta_gt | 0.06046 | 0.7799 | 0.004663 | pass | 0.004663 | 0.004663 |
| `t3r4_q50_s80535_bwA` | 50 | 0.006715 | 68.52 | 0.0362 | delta_gt | 0.459 | 0.2668 | 0.002121 | pass | 0.002132 | 0.002132 |

All five fires are at q = 50 (5 of 40 = 0.125), none at q = 60 (0 of 40); the trigger values are `Delta` = 0.0052-0.0094 against the limit 0.005. In four of the five runs attempt 1 was in the stage-3 clamp regime (S3 of 0.46-0.78) and the frozen attempt is out of it in three (S3 0.010, 0.0004, 0.056) and still above 0.2 in one (seed 80535: 0.459 to 0.267); seed 80532 is the other way round: attempt 1 was outside the regime (S3 = 0.060, `Delta` = 0.0094) and the frozen attempt is inside it (S3 = 0.78) and passes with `eta_3` = 0.004663, 93 % of the limit. So a restart does not need to leave the regime to pass, and a restart can enter it. Cost of the fires: section 7.

**First-attempt outcome.** The pass count with every fired run counted as a failure of its first attempt (the convention of the G1 reply of round T3-R3; a fired run's first attempt is the run `bw` would have produced up to the check) is **74 of 80** (0.925, exact 95 % interval [0.844, 0.972]; q = 50: 35 of 40, q = 60: 39 of 40). It is the `bw`-equivalent pass rate of this block: the 79 passes of `bwA` are the 74 first-attempt passes plus the 5 fired runs whose second attempt passes.

## 6. The failure, classified

| run | failing gates | failing quantities (stage) | S3 (first / frozen) | fired | stage-2 stretch nodes | F(u01600) (threshold) | last 16 exports | class (suggested by rule) |
|---|---|---|---|---|---|---|---|---|
| `t3r4_q60_s80502_bwA` | gate_G_F;gate_G_N | Gmax_full = 0.0586406 (limit 0.01) at stage 2; \|dev - final\| of Gmax_full = 0.0010748 (limit 0.001) at stage 2 | 0.5358 / 0.5358 | no | 5 | 54.04 (26.78) | - | K |

The only failing run is q = 60 seed 80502 (`gate_G_F` and `gate_G_N`). Its stage-3 quantities pass (`eta_3` = 0.00093, `rmse_pos` = 0.0214, `tail_mean` = 0.0032: G-A holds), so the last-16-export range of the stage-3 quantity is empty (no stage-3 quantity failed). The failing quantity lives at **stage 2**: the maximal one-step gain `Gmax_full` = 0.0586 at d = -14 (limit 0.01), and the development-final difference of `Gmax_full` is 0.0010748 (limit 0.001, by 7 %). A **stage-2 near-zero stretch** of 5 in-band nodes (the longest at d in [-22, -14]) explains it. **Class (K)** of T3-R3's D2: S3 of the frozen attempt = 0.536 (above 0.2: the run is in the clamp regime) and the failing quantity is a stage-2 near-zero stretch. It is not (F) (F(u01600) = 54.0 against the threshold 26.8 of q = 60: the policy formed), not (V) (no stage-3 quantity fails), and I found no implementation error (E): the run passes C-INIT and C-REPLAY like the others and, its trigger not having fired, it is by construction (C-REPRO) the `bw` run of its seed. **The trigger does not see it:** `Delta` at local update 1600 = 0.0008 (limit 0.005). It is the second failure of this kind among the 120 `bwA` runs of T3-R3's development and this confirmation (T3-R3 development: q = 60 seed 70511, stage-2 stretch of 28 nodes, also unreached by A), both at q = 60.

Other descriptive facts: 28 of the 80 runs are in the clamp regime (S3 of the frozen attempt > 0.2) and 27 of them pass; 3 runs have a stage-2 stretch (14, 8 and 5 nodes) and 2 of them pass (the stretches of the passing runs carry `Gmax_full` of 0.0045 and 0.0046).

## 7. The cost

| q | runs | fired | fired share | mean updates | mean policy rows | mean minibatch steps | mean dev checks | ratio updates (to a run that does not fire) | total wall (s) | mean wall (s) | median wall (s) | mean wall, fired (s) | mean wall, not fired (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 40 | 5 | 0.125 | 4000 | 2.05e+06 | 8e+04 | 160 | 1.053 | 2.449e+04 | 612.3 | 590.9 | 801.2 | 585.3 |
| 60 | 40 | 0 | 0 | 3800 | 1.95e+06 | 7.6e+04 | 152 | 1 | 2.332e+04 | 582.9 | 591.5 | - | 582.9 |
| both | 80 | 5 | 0.0625 | 3900 | 2e+06 | 7.8e+04 | 156 | 1.026 | 4.781e+04 | 597.6 | 591.3 | 801.2 | 584.1 |

A run that does not fire costs 3,800 updates, 1,945,600 policy rows, 76,000 minibatch steps and 152 development checks; a fired run 5,400 updates, 2,764,800 rows, 108,000 steps and 216 checks (a factor 1.421; pre-registered in `reports/t3/r3/04_preregistration.md` section 4). With 5 fires in 80 runs the mean is **3,900 updates (1.026 times), 2.0e6 policy rows, 78,000 minibatch steps and 156 checks**; the pre-registered expectation at the recorded fire rate of 7 of 80 was 1.037, and the observed fire rate 0.0625 lies inside the exact 95 % interval [0.036, 0.172] of that rate (`reports/t3/r3/04_preregistration.md` section 4). At q = 50, where all fires are, the factor is 1.053. **Measured wall time** (one concurrent wave of 40 workers, single-threaded runs, load average 8.9 before and up to about 50 during the wave): 597.6 s per run on average (median 591.3 s), 47,810 s for the wave (13.3 CPU-hours); a fired run took 801.2 s on average (766-853 s) against 584.1 s for a run that did not fire, 217 s or 37 % more, in line with the stage-3 phase of about 235 s (`reports/t3/r3/04_preregistration.md` section 4) and with T3-R3's 36 %. The share of fired runs is 0.0625 (5 of 80).

## 8. Descriptive comparisons (they decide nothing)

| set | q | runs | passes | rate | exact 95 % interval |
|---|---|---|---|---|---|
| t3r4_confirmation_bwA | all | 80 | 79 | 0.9875 | [0.932, 1.000] |
| t3r4_confirmation_bwA | 50 | 40 | 40 | 1 | [0.912, 1.000] |
| t3r4_confirmation_bwA | 60 | 40 | 39 | 0.975 | [0.868, 0.999] |
| t3r3_development_bwA | all | 40 | 39 | 0.975 | [0.868, 0.999] |
| t3r3_development_bwA | 50 | 20 | 20 | 1 | [0.832, 1.000] |
| t3r3_development_bwA | 60 | 20 | 19 | 0.95 | [0.751, 0.999] |
| pooled_bwA_120 | all | 120 | 118 | 0.9833 | [0.941, 0.998] |
| pooled_bwA_120 | 50 | 60 | 60 | 1 | [0.940, 1.000] |
| pooled_bwA_120 | 60 | 60 | 58 | 0.9667 | [0.885, 0.996] |
| t3r1_development_bw | all | 40 | 36 | 0.9 | [0.763, 0.972] |
| t3r1_development_bw | 50 | 20 | 18 | 0.9 | [0.683, 0.988] |
| t3r1_development_bw | 60 | 20 | 18 | 0.9 | [0.683, 0.988] |
| t3r2_confirmation_bw | all | 40 | 34 | 0.85 | [0.702, 0.943] |
| t3r2_confirmation_bw | 50 | 20 | 16 | 0.8 | [0.563, 0.943] |
| t3r2_confirmation_bw | 60 | 20 | 18 | 0.9 | [0.683, 0.988] |
| t3r3_development_bw | all | 40 | 37 | 0.925 | [0.796, 0.984] |
| t3r3_development_bw | 50 | 20 | 18 | 0.9 | [0.683, 0.988] |
| t3r3_development_bw | 60 | 20 | 19 | 0.95 | [0.751, 0.999] |
| pooled_bw_120 | all | 120 | 107 | 0.8917 | [0.822, 0.941] |
| pooled_bw_120 | 50 | 60 | 52 | 0.8667 | [0.754, 0.941] |
| pooled_bw_120 | 60 | 60 | 55 | 0.9167 | [0.816, 0.972] |
| t3r4_first_attempt_bwA_equivalent | all | 80 | 74 | 0.925 | [0.844, 0.972] |
| t3r4_first_attempt_bwA_equivalent | 50 | 40 | 35 | 0.875 | [0.732, 0.958] |
| t3r4_first_attempt_bwA_equivalent | 60 | 40 | 39 | 0.975 | [0.868, 0.999] |

- This confirmation pooled with `bwA`'s 40 development runs of T3-R3 (39 of 40): **118 of 120 pass** (0.983, exact 95 % interval [0.941, 0.998]; q = 50: 60 of 60, q = 60: 58 of 60).
- T3-1.0's three blocks of `bw`: 36 of 40 (T3-R1 development), 34 of 40 (T3-R2 confirmation, which failed its rule), 37 of 40 (T3-R3's development control); pooled **107 of 120** (0.892, [0.822, 0.941]; q = 50: 52 of 60, q = 60: 55 of 60). The 118 of 120 of `bwA` is 11 passes more than 107 of 120 on the same number of runs; the blocks have different seeds, so this is a description, not a test.
- The pass rate of T3-1.0 implied by the first-attempt outcome: **74 of 80 = 0.925** [0.844, 0.972], in the range of the three `bw` blocks (0.850-0.925); pooled with T3-R3's development control (`bw` 37 of 40, whose first-attempt count under the same convention is also 37) it is 111 of 120.

## 9. What the confirmation shows and what it does not

- The pre-registered rule is met: at most 6 failures of 80 and at most 4 of 40 per q, with one failure. The checks, the blind recomputation and the lock guards all hold, and no run was repeated or replaced.
- The restart is exercised: 5 of 80 runs fired and all five pass on their second attempt; it costs 2.6 % more updates (3.7 % at the pre-registered expectation).
- It acts at q = 50 in these 120 `bwA` runs (7 fires of 60 runs at q = 50 in the development and this confirmation, none of 60 at q = 60, apart from the diagnostic run of seed 60518 in T3-R3), and it does not reach the failure type that remains: a stage-2 near-zero stretch in the clamp regime at q = 60 (2 of the 60 `bwA` runs at q = 60 of the two blocks).
- This is one confirmation block of 80 runs with one failure; the mechanism by which the restart helps is not tested here (T3-R3's diagnosis found the clamp regime a marker, not shown to be a cause, and a restart that stays in the regime still passes); nothing was tuned, and no other scheme was tried.

## 10. Files

- Lock and plan: `protocols/v2_T3_locked_v1_1.{json,md}`, the record in `protocols/LOCK`, `reports/t3/r4/01_lock.md`, `confirmation_plan.csv`; checks of the tools: `02_checks.md`.
- The wave: `results/t3_r4/confirmation/q{q}/seed{s}/bwA/` (tracked: manifest, config, status, gates, rule log, drift test, summary, check and update CSVs, bin maps, continuation tables, and for fired runs `retry.json` and the small records of `attempt1/`; untracked: weights, `.pt`, freeze and evaluation arrays, train histories), `launch_*.json`, `launch_commit.txt`, `launch_machine.txt`, `launch_start.txt`, `launch_end.txt`, `launch_stdout.txt`, `infra_events.json`, `launch_checks.json`, `post_launch_stdout.txt`.
- Evaluation and analysis: `results/t3_r4/confirmation/eval/per_run.csv`, `eval_reruns.json`, `analysis/` (`headline.json`, `blind.json`, `per_run.csv`, `counts.csv`, `fired.csv`, `failures.csv`, `cost.csv`, `comparisons.csv`, `last16.csv`, `last16_summary.csv`).
- Tools: `tools/t3/t3r4_{launch,make_lock,checks,eval,analysis,blind,repro,post_launch,preflight_seeds}.*`; tests `tests/test_t3r4_*.py`.
