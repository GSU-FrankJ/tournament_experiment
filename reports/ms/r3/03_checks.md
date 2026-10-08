# MS-R3 P1: tests, independent review, C-R6 and the checks of the pilot

Date: 2026-10-07. Spec: `reports/ms/r3/pi_record/20_ms_r3_prompt.md` sections 2.2, 2.6 and 3.2. Everything below was run on the tree that becomes the code commit (the pre-registration, `02_preregistration.md`, names it); between that commit and the launch only `results/` and the report folder change.

## 1. Tests

Full suite `pytest tests` on the tree of the code commit (the tests were run on the working tree whose content is this commit), as **four groups of invocations over a partition of all 1376 collected tests** (2026-10-07 23:14 to 2026-10-08 00:25, tmux; the partition was checked: 570 + 365 + 128 + 313 = 1376 = `pytest tests --collect-only`):

```
pytest tests --ignore-glob='tests/test_ms_*'                       1 failed, 567 passed, 2 xfailed, 2 warnings in 1004.95s (0:16:44)
pytest <the MS-R1 test files, tests/test_ms_*.py minus r2 / r3>   365 passed in 670.55s (0:11:10)
pytest tests/test_ms_r2_*.py                                       128 passed, 1 warning in 603.93s (0:10:03)
pytest tests/test_ms_r3_actor.py                                    31 passed, 1 warning in 487.85s (0:08:07)
pytest tests/test_ms_r3_analysis.py                                129 passed in 219.77s (0:03:39)
pytest tests/test_ms_r3_{launcher,launch_checks,screen,actor_diagnostics}.py   153 passed, 12 warnings in 146.19s (0:02:26)
```

Together **1373 passed, 1 failed, 2 xfailed**; the one failure is the known `tests/test_registry_canonicalization.py::test_registry_canonicalization` (as in MS-R1 and MS-R2). All 365 MS-R1 and all 128 MS-R2 tests pass unchanged (no existing test file differs from `e8eb9a08`: `git diff --name-only e8eb9a08 -- tests` lists only the six new files). The C7 tests (the regression pipeline against the canonical reference: `tests/test_ms_runner.py`, `tests/test_v2_r2c.py`, `tests/test_v2_refine.py`) are in the first two groups and pass.

New test files (all single-threaded, CPU, no network):

| file | tests | what it proves |
|---|---|---|
| `tests/test_ms_r3_actor.py` | 31 | without the new keys the runner equals commit `e8eb9a08` (reduced `NL_st_s16`, `NL_bb_s4` and an `MS_s35a5` rule run, each in a `git archive` checkout of that commit and in the current tree: states, every weight export, `ms_updates.csv`, check rows, no new column; with `init_digest: true` only the manifest differs); the forward of each variant against an independent float64 D2 reference; `t10(x) == t1(x * (1, 10))` exactly, through the continuation-table and verifier policy functions; identical initial actor / critic weights and recorded digest across variants (C-INIT); the digest covers actor and critic and draws no random number; 5 configuration refusals; exports carry the variant (t1: none); the numpy reload equals the torch forward and a stripped export would be read as tanh; the opponent, every refresh, the frozen snapshot, the full state and its round trip carry the variant; the stage-1 table from a variant's frozen snapshot equals a direct evaluation through the reload; C-NL for relu and t10; reduced T = 3 smoke runs with relu and with t10; the MS-R1 residual example refuses a non-t1 export (behavioural) |
| `tests/test_ms_r3_launcher.py` | 14 | wave `r3` (240 jobs, `pilot/q*/seed*/<arm>`, default root `results/ms_r3`), the `--actors` filter, the twelve configs (t1 arms equal MS-R2's `NL_*` arms apart from labels and `init_digest`; the actors differ only in the variant key), every MS-R1 and MS-R2 `build_config` output and arm table unchanged against `e8eb9a08` |
| `tests/test_ms_r3_launch_checks.py` | 51 | a genuine reduced 12-arm wave passes every check; tampered scale, init digest, one component at a time of C-MS5 (24 cases), a missing run, the C-NL "export differs" half (network arrays, not labels), the `--actors` filter, the CLI exit code |
| `tests/test_ms_r3_screen.py` | 47 | the supervised screen: runner initialisation identical across variants, LR schedule, stratified shares, `e_hat(0)` against the reload, a float64 numpy re-implementation of the update, the d stream identical across actors, no overwriting, `summarise` and the premise check on constructed cells (PASS, DROP relu, STOP (i), STOP (both fail (ii))) |
| `tests/test_ms_r3_actor_diagnostics.py` | 41 | the diagnostics tool: formulas by hand, the variant-aware reload (relu / t10 exports are never read as tanh), quantile and count arithmetic, missing roots, overwrite refusal, `--screen-dir`, a genuine mini run, one real MS-R2 run; 9 deliberate bugs seeded into the tool are each caught |
| `tests/test_ms_r3_analysis.py` | 129 | the criterion parts (a) and (b) by hand, every bootstrap-producing table against a manual recomputation with a fresh `default_rng(20261008)` per interval (a 10-seed world; a spy on every generator; a changed seed changes every interval column), transmission, interaction, quadrature (including F^2 < 0), predictions P1-P4 by independent formulas, strata, the first-layer extraction (t10 reports ten times the stored weight; every export with a `stage` column), the symmetry-error window, the blind script (agreement, tampering, independence by an AST scan), extraction on a genuine reduced wave |

## 2. Independent review (prompt 2.6)

One read-only review of the diff by an agent that did not write it (conformance with D2-D6, the initialisation identity, bit-identity of the defaults, the variant semantics, the analysis and the checks), run on the working tree on top of `e8eb9a08` before the first MS-R3 commit. Result: **0 MAJOR, 6 MINOR, 17 INFO**. None of the MINOR findings touched `agents/`, `run/`, `utils/`, `envs/` or `protocols/`.

| finding (severity) | disposition |
|---|---|
| **M1 (minor)** the "its `u02025` export differs" half of C-NL cannot fail: the s = 16 export carries a `conc_scale` entry that the s = 1 export lacks, so a key-set difference alone made it pass (reproduced on the reduced wave: s = 1 networks copied into the s = 16 export still passed); the same in `tests/test_ms_r3_actor.py` | **fixed** (`tools/ms/r3_launch_checks.py:_weights_differ` compares the `actor.*` and `critic.*` arrays only; `tests/test_ms_r3_launch_checks.py::test_c_nl_export_differs_half_compares_the_network_arrays_not_the_labels`; `tests/test_ms_r3_actor.py::same_networks`) |
| **M2 (minor)** tests that cannot fail or cannot see what their name claims (analysis: a slice compared with itself, n <= 4 pairs so a partly changed bootstrap seed survives, degenerate transmission and interaction rows, weak prediction asserts, symmetric sign counts, empty reference roots in the CLI test, the |d| < 2q window, t10 evaluated at d = 0; screen: a structural test, an assertion over all four outcomes, a mutated module fixture; diagnostics: a hard-coded absolute path; actor: a source-text grep) | **fixed** by a test-strengthening pass (an agent that edited the test files only): a 10-seed world (18 arms x 2 q) run through the real `run_analysis`, every table equal to a manual recomputation with a fresh `default_rng(20261008)` per interval, a spy on every generator the run creates, a second run with shifted seeds changing every interval column; hand values for transmission, interaction and P1-P4; asymmetric sign counts; a genuine `parents_A` and `rehearsal_v2_0` root in the CLI test; the symmetry-error window pinned at both edges; the t10 reload checked at d != 0; screen: a behavioural d-stream test, specific premise outcomes, a copied fixture; diagnostics: no absolute path; actor: a behavioural residual-example test. Every new or strengthened test fails under at least one in-memory mutation of the code it covers (about 80 mutations); no tool bug was found. One observation, no change: `r3_analysis.transmission_ratio` binds its seed as a default argument at import (patching `BOOT_SEED` at run time would not reach it; editing the constant in source does) |
| **M3 (minor)** no test shows that the C-INIT digest covers the critic (an actor-only digest passed) | **fixed** (`tests/test_ms_r3_actor.py::test_the_init_digest_covers_the_actor_and_the_critic_and_consumes_no_rng`: six perturbations of actor and critic weights change the digest, the digest draws no random number) |
| **M4 (minor)** D5 asks for the first-layer weights "at every weight export"; `first_layer_weights.csv` had the 112 terminal-stage exports of 136 | **fixed** (`tools/ms/r3_analysis.py`: the file now holds every export, 112 terminal-stage + 24 stage-1, with a `stage` column; `first_layer_summary.csv` and the last-export columns of `per_run.csv` use the terminal stage; tests updated) |
| **M5 (minor)** `01_supervised_screen.md` showed four of the five metrics only at 56k steps | **fixed** (blocks `grid_rmse`, `grid_tail`, `grid_weff`, `grid_maxw` of `reports/ms/r3/report_scripts/screen_tables.py`; section 2.2) |
| **M6 (minor)** `01b_rl_actor_diagnostics.md` did not embed the |w| distribution | **fixed** (blocks `diag_dist`, `diag_dist_facts`; section 2.1) |
| INFO 1: the digest is taken before `set_actor_variant` | not changed: the digest hashes weights, which do not depend on the variant; the variant is recorded separately in the manifest |
| INFO 2: `CurriculumPPO.state()`, `save()`, `load_weights()` carry no variant | not changed: no MS path uses them (the runner saves `full_state`, which carries the variant) |
| INFO 3: premise (ii) compares the signed deficit | noted; all variant medians are positive (smallest 0.032) so it has no effect |
| INFO 4: two rounding slips in `01_supervised_screen.md` | **fixed** (0.025 and 0.335) |
| INFO 5: pre-registration placeholders; D1 not restated; `results/ms_r1/calibration` missing from the roots | **fixed** in `02_preregistration.md` |
| INFO 6: the "small-weight regime" reference is an all-seed median; across the ten `t1` bin-balanced screen cells the Spearman of max abs w against the deficit is +0.70 (q = 50) and -0.60 (q = 60) | noted in `01b_rl_actor_diagnostics.md` section 4 (a monotone small-weight regime is not established by the screen) |
| INFO 7: `results/ms_r3/supervised_screen/run.log` is git-ignored | **fixed**: the log is force-added (12 KB) with the offline outputs |
| INFO 8: a launch without `--params` would use the default stage-2 rho 0.03 instead of 0.05 | no gate added (the launch command of `02_preregistration.md` section 11 has the flag; the launch record has `params_sha256`) |
| INFO 9: untracked files outside `results/` make `git_state().dirty` true | handled by procedure: nothing is created or committed in the worktree while the launcher is starting runs |
| INFO 10: start-share exceedances are reported, not gating | as in MS-R1 and MS-R2 |
| INFO 11: the repo identity tests compare 4 of 9 C-MS5 parts | the reviewer ran all 9 on a reduced pair; the pilot's C-MS5 compares all 9 on the real runs |
| INFO 12-14, 16: an equivalent mutant in `refresh_snapshot`; the documented `variant=` override of `mean_effort_numpy`; metrics without a finite pair omitted from a paired table; R3's type-strict comparison against R2's | noted, no change |
| INFO 15: loader inventory | every loader used in this round reads the variant or refuses (the `tools/v2/` loaders would read a relu / t10 export as tanh d / B; they are unchanged and not used by any r3 tool) |
| INFO 17: D5 coverage | all items present in the analysis outputs (mapping in the review notes) |

Checks the reviewer ran and reported as passing: the D2 forward of all three variants against an independent float64 reference; C-INIT by construction (`CurriculumPPOv2` built three ways: all actor, critic and opponent parameters identical); the configs (t1 arms equal MS-R2's NL arms except labels and `init_digest`; relu / t10 differ from t1 only in `actor_variant` and labels, all 8 combinations at both q); the launcher dry run (240 jobs, 160 with `--actors t1,t10`); the constrained paths byte-identical to `e8eb9a08` (`protocols`, `envs`, `run/run_v2_*.py`, `run/v2_rollout.py`, `utils/v2_*`, `utils/dp_br_verifier.py`, `tools/v2`); no closed-form identifier in the tracked diff; reduced `NL_st_s16` against a `git archive e8eb9a08` tree (all nine C-MS5 parts identical: 409 arrays, 267 gate values); full-size first 200 updates identical to the real MS-R2 outputs (`NL_bb_s1`, `NL_st_s16`); **three full-length t1 re-runs against the real MS-R2 outputs through the C-MS5 CLI path, ALL true** (`t1_bb_s1` q50 seed 10501: 1839 arrays; `t1_st_s16` q60 seed 10501: 1873; `t1_bb_s16` q50 seed 10502: 1871; 136 exports, 3400 update rows, 136 check rows and 267 gate values each); full-scale C-NL for `relu_bb` q50 and `t10_st` q60; the variant-aware reload against the verifier freeze (about 1e-5 effort units; reading the same exports as tanh d / B would be off by up to 97 for relu and 54 for t10); the analysis on a stand-in wave (MS-R2 data under the new arm names): the blind script agrees on 684 numbers and an independent recomputation of the criterion, transmission, quadrature and predictions tables has 0 mismatches; an independent float64 re-fit of the supervised screen reproduces its distributions and the premise check (PASS).

Not verified by the reviewer: C-R6, C7 and the full suite (section 1 and 3), the real 240-run wave.

## 3. C-R6: the unchanged v2.0 entry point reproduces `rehearsal_v2_0`

The unchanged `run/run_v2_T2_locked.py --q Q --seed S` from this branch, q in {50, 60} x seeds 10501-10510, into `results/ms_r3/v20_reproduction/q{Q}/seed{S}`:

```
python tools/ms/launch_ms_r1.py --wave v20_repro --root results/ms_r3 --workers 20 --code-commit f25a3cfc   # tmux session ms_r3_cr6
python tools/v2/cr2_compare.py --ref /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 \
    --new results/ms_r3/v20_reproduction --out results/ms_r3/v20_reproduction_checks.json
C-R2 identical 20/20 ALL=True
```

**PASS, 20 of 20 identical.** `cr2_compare` compares the v2.0 training-relevant state (the states with actor, critic, opponent, frozen snapshot, both Adam states and every stream, every weight export and `checkpoint_weights.npz`, the histories, `v2_updates.csv` and the checkpoint CSVs on the common columns, the gate values, the evaluation NPZs, the induced band and drift test, the continuation table and the verdict fields; manifests, wall-clock fields and the process-global RNG states are excluded). Launch record `results/ms_r3/v20_reproduction/launch_20261007_230601.json`: 20 runs, all exit 0, wall 366-385 s (median 374 s), nproc 64, load average at start 14.5 / 13.0 / 12.4, HEAD `f25a3cfc` equal to the code-commit argument, `diff_stat_run_code_to_head` empty, `git status --porcelain` listing only the 1,680 untracked arrays and weights of the MS-R1 and MS-R2 pilots under `results/`. The run code of that launch (`agents/`, `run/`, `utils/`, `envs/`, `protocols/`) is byte-identical to the first MS-R3 commit `0a9698b6` (`git diff --stat 0a9698b6 f25a3cfc -- agents run utils envs protocols` is empty) and differs from `e8eb9a08` only in `agents/ppo_curriculum.py`, `agents/ppo_curriculum_v2.py` and `run/run_ms_stagewise.py`; the paths the prompt forbids changing (`protocols/`, `envs/`, `run/run_v2_T2_locked.py`, `run/run_v2_stagewise.py`, `run/v2_rollout.py`, `utils/v2_continuation.py`, `utils/v2_metrics.py`, `utils/dp_br_verifier.py`, `tools/v2/`) are byte-identical to `e8eb9a08`. The locked entry point uses the tanh d / B actor path that the new code leaves untouched; C-R6 is the full-scale proof for it (the full-length C-MS5 re-runs of the review are the proof for the MS runner).

## 4. Offline outputs

`results/ms_r3/supervised_screen/` (140 cells, premise check PASS: `reports/ms/r3/01_supervised_screen.md`) and `results/ms_r3/rl_actor_diagnostics/` (260 runs, 32,389 exports: `reports/ms/r3/01b_rl_actor_diagnostics.md`).

## 5. P2: launch checks, identities, analysis, blind recomputation (2026-10-08)

Run after the pilot (`results/ms_r3/pilot`, launched 00:24:23 from HEAD `b41ecefd` = the code commit; the worktree was not modified between the launch and the end of the pilot: `git status --porcelain` at the launch lists only the untracked arrays of the MS-R1 and MS-R2 pilots under `results/`, and every manifest has `clean_tree`). Tables: `python reports/ms/r3/report_scripts/pilot_tables.py --block checks`.

| check | result | path |
|---|---|---|
| status exit 0 / manifest at the launch commit, clean tree / files complete / global-RNG assertions / tail-share coverage | 240/240 each | `results/ms_r3/pilot/launch_checks.json` (`base.n_ok_per_check`; `base.all_ok` true) |
| start shares per stratum within 3 binomial SE | 720 tests, 0 flagged (1.9 expected by chance) | `launch_checks.json`, `base.start_share_tests` |
| applied scale equals the D3 schedule at every terminal-stage update; 1.0 throughout stage 1 | 240/240 | `summary.scale_ok` |
| C-INIT (one `init_state_sha256` across all twelve arms of every (q, seed)) | 20/20 | `summary.C_INIT_pass` |
| C-NL (s = 16 against s = 1 within each (actor, starts, q, seed) through update 2001; `u02025` differs in the network arrays) | 120/120 | `summary.C_NL_pass` |
| **C-MS5** (each `t1` arm against MS-R2's `NL_*` arm over the whole run: all 136 weight exports, `train_history`, the whole `ms_updates.csv`, both check tables, the freeze arrays, bin maps, stage-1 table, gate values) | **80/80** | `summary.C_MS5_pass`; what is and is not compared: `ignored` in the same file |
| analysis tool: the decomposition against the runner's own record, the arm / export / config variant agreement | no flag raised: the `flags` column of `per_run.csv` is empty in all 360 rows; 360 of 360 rows done (240 arms + 80 MS-R2 `NL_*` reference rows + `parents_A` and `rehearsal_v2_0`) | `results/ms_r3/analysis/per_run.csv`, `completeness.csv`, `analysis_info.json` |
| independent recomputation of `criterion.csv`, `transmission.csv` and `interaction.csv` from `per_run.csv` (`tools/ms/r3_blind_criterion.py`, imports nothing from `tools/ms/`) | "ALL 684 numbers agree ... to 1e-12 (floats) / exactly (counts, flags, lists)" | `results/ms_r3/analysis/blind_recomputation.txt` |

All checks pass; no stop condition of the prompt was met. The checks are those of prompt section 3.2; no check was added after the pre-registration. The four `t1` arms of the pilot are therefore bit-identical re-runs of MS-R2's `NL_bb_s1`, `NL_bb_s16`, `NL_st_s1`, `NL_st_s16`: their rows in `results/ms_r3/analysis/` equal MS-R2's.
