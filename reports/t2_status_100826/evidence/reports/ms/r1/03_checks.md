# MS-R1 P1: reproduction checks, tests, independent review

Date: 2026-10-07. Spec: `reports/ms/r1/pi_record/17_ms_r1_prompt.md` §2.2 and §2.3. Both reproduction checks were run on real runs (development seeds 10501-10510, q in {50, 60}, full budgets) at commit `c95a2af4` with a clean tree. Between that commit and the code commit of the pre-registration only `reports/`, `results/` and `docs/` change (`git diff --stat c95a2af4 <code commit> -- . ':(exclude)reports' ':(exclude)results' ':(exclude)docs'` is empty; section 5).

## 1. C-R4: the unchanged v2.0 entry point reproduces `rehearsal_v2_0` (PASS, 20 of 20)

The unchanged `run/run_v2_T2_locked.py --q Q --seed S` from this branch, q in {50, 60} x seeds 10501-10510, into `results/ms_r1/v20_reproduction/q{Q}/seed{S}`:

```
python tools/ms/launch_ms_r1.py --wave v20_repro --workers 20 --code-commit c95a2af4     # tmux session ms_cr4
python tools/v2/cr2_compare.py --ref /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 \
    --new results/ms_r1/v20_reproduction --out results/ms_r1/v20_reproduction_checks.json
C-R2 identical 20/20 ALL=True
```

48 fields per run are compared (`state_end_A.pt` / `state_end_B.pt` with actor, critic, opponent, frozen snapshot, both Adam states, the minibatch stream, the four numpy streams and the torch generator; every weight export and `checkpoint_weights.npz`; histories, stability, verifier calls, curriculum, snapshots; `v2_updates.csv` and `v2_checkpoints_{A,B}.csv`; the gate values; the evaluation NPZs; `induced_band.json`, `drift_test.json`; the continuation table `continuation_table.npz` bit for bit and its SHA-256; the v2.0 verdict fields of `gates.json`; status done with exit 0); failing fields: none (`results/ms_r1/v20_reproduction_checks.json`). Every run: exit 0, manifest at `c95a2af4`, `clean_tree` true, outcome `pass`.

Launch record `results/ms_r1/v20_reproduction/launch_20261007_030049.json`: nproc 64, load average at start 22.46 / 14.52 / 13.45, free disk 1026 GB, HEAD `c95a2af4`, `git diff --stat c95a2af4 HEAD` empty, `git status --porcelain` at the launch: only the sibling wave's directory `?? results/ms_r1/base/` (results are excluded from the runs' own dirty flag), 20 workers (the base wave ran at the same time with 20 more: 40 single-threaded workers in total), wall per run median 500 s, maximum 529 s (loaded host: the load average rose to 30-45).

This is the proof that the shared-module edit (`envs/curriculum_env.py`, new `StartSampler` methods only) left every existing path byte-identical in effect, at full budget; the existing suites (section 3) are the second proof.

## 2. C-MS1: the terminal stage of `MS_base` equals `parents_A` (PASS, 20 of 20)

The 20 `MS_base` runs (`start_weights` bin_balanced, `rule.enabled` false, fixed budgets 1600 / 600, v2.0's LR windows), into `results/ms_r1/base/q{Q}/seed{S}/MS_base/`:

```
python tools/ms/launch_ms_r1.py --wave base --workers 20 --code-commit c95a2af4          # tmux session ms_cms1
python tools/ms/cms1_compare.py --ref .../v2-t2-refine/results/v2_refine/parents_A --ref-kind parentsA \
    --new results/ms_r1/base --out results/ms_r1/base_checks_parents_A.json
C-MS1 (parentsA) identical 20/20 ALL=True
python tools/ms/cms1_compare.py --ref .../v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0 --ref-kind rehearsal \
    --new results/ms_r1/base --out results/ms_r1/base_checks_rehearsal_v2_0.json
C-MS1 (rehearsal) identical 20/20 ALL=True
```

17 fields per run, all identical in all 20 runs, against `parents_A` (the prompt's reference) and also against `rehearsal_v2_0` (D6: the same candidate): `state_end_stage2.pt` against `state_end_A.pt` for the actor, critic, lagged opponent, frozen snapshot (none on both sides), both Adam states, the minibatch stream, `snapshot_refreshes`, the concentration scales, the env / learner / opponent / start streams, the torch generator and the counters (global update 1600, episodes, transitions); all 64 weight exports `u0025 ... u1600` bit for bit; the per-update training series of `train_history.json` for updates 1..1600 (losses, KL, clip fraction, advantage statistics, gradient norms, return, LR, snapshot flag, the learner's mean effort); the five stream positions after every update and the losses of `v2_updates.csv` / `ms_updates.csv`; and the end-of-phase evaluation arrays on both verifier tiers (`freeze_stage2_{final,development}.npz` against `final_{final,development}.npz` / `gateA_*.npz`). The tool and its negative controls are in `tests/test_ms_cms1.py` (a changed weight bit, a stream state, a missing evaluation file, a tampered series are each found).

Launch record `results/ms_r1/base/launch_20261007_030049.json` (same host record as above; `git status --porcelain` empty): 20 of 20 runs exit 0, wall per run median 446 s, maximum 478 s. Post-launch checks on this wave with the tool of the pilot (`tools/ms/launch_checks.py`, `results/ms_r1/base/launch_checks.json`): manifests at the commit and clean 20 of 20, files complete 20 of 20, global-RNG assertions 20 of 20, tail-mass coverage 20 of 20, 60 start-share tests of which 0 flagged (0.16 expected by chance).

**One launch error, recorded.** The base wave was launched without `--params`, so its configs carry the D4 default thresholds (rho_2 = 0.03), not the pre-registered file (rho_2 = 0.05). In the legacy arm the thresholds decide nothing (the training state is identical, which is what C-MS1 shows); they only define the `would_fire_local` entry of `rule_log.json`. The entry was recomputed from the logged checks with the pre-registered thresholds (`reports/ms/r1/report_scripts/base_would_fire.py`, output `results/ms_r1/base_would_fire.csv` / `.txt`): stage 2 would fire in 0 of 10 runs at q = 50 and 1 of 10 at q = 60 (local update 1500), in 1 of 20 against the logged 0 of 20 at rho_2 = 0.03 (calibration, Table 2: 1 of 60 at rho = 0.05); stage 1 would fire in 20 of 20 runs, median local update 125.

**Stage 1 of `MS_base` is not bit-identical to v2.0's Phase B** (its batch holds stage-1 rows only; D2). Descriptive comparison with `rehearsal_v2_0` (`results/ms_r1/base_vs_rehearsal_stage1.csv`, script `reports/ms/r1/report_scripts/base_vs_rehearsal_stage1.py`): the closed-form stage-1 error |e1| has median 0.0093 (maximum 0.0163) at q = 50 against 0.0120 (0.0407) for `rehearsal_v2_0`, and 0.0154 (0.0261) at q = 60 against 0.0101 (0.0314); G-S 20 of 20 in both; G-F 20 of 20 in both; the v2.0 combination (G-A, G-F, G-N, G-S) 20 of 20 in both. The first-order stage-1 residual R_1 at the freeze of `MS_base` has median 0.0161 (final tier) and 0.0164 (development tier). All 20 `MS_base` runs have the outcome `pass`.

## 3. Tests

- **Existing suites, before the MS tests** (`pytest tests --ignore=tests/test_ms_*.py`, 2026-10-07 02:24-02:41, 1000 s): `1 failed, 567 passed, 2 xfailed`; the failure is the known `tests/test_registry_canonicalization.py::test_registry_canonicalization` (the same state as at the end of the publication round).
- **MS tests** (`pytest tests/test_ms_*.py`, 559 s): `314 passed`. Per file: `test_ms_sampler.py` 65, `test_ms_residual.py` 43, `test_ms_rule.py` 49, `test_ms_continuation.py` 22, `test_ms_runner.py` 46, `test_ms_launcher.py` 8, `test_ms_launch_checks.py` 7, `test_ms_cms1.py` 5, `test_ms_replay.py` 31, `test_ms_analysis.py` 38.
- **Full suite** `pytest tests`: `1 failed, 881 passed, 2 xfailed, 2 warnings in 1699 s` (2026-10-07 03:00-03:29, `pytest tests`): the failure is the known `test_registry_canonicalization`; 881 = 567 pre-existing + 314 MS tests.
- The reduced-budget identities required by §2.2 are in `tests/test_ms_runner.py`: the terminal phase under legacy settings equals v2.0's `phase_A` bit for bit for q = 50 and q = 60 (actor, critic, opponent, both Adam states, the minibatch stream, the five streams, the torch generator, every weight export, the per-update series and the five stream positions after every update), the C7 reference itself (`test_c7_reference_is_unchanged`), the reduced locked run of the current tree against the v2.0 lock commit (`test_the_locked_v20_pipeline_equals_the_lock_commit_on_a_reduced_run`), the stage-1 table against the v2.0 builder with no RNG movement, the T = 3 full pipeline on reduced budgets (three phases, nested tables rebuilt independently from the frozen snapshots, the stop path and the forced-landing path), the closed-form stub invariance, the start shares, the global-RNG path (exit code 5), 36 configuration refusals.
- Numbers from the module tests worth recording (`tests/test_ms_continuation.py`, `tests/test_ms_residual.py`; reported by the agent that wrote them, reproduced by the passing tests): the T = 2 table equals `utils.v2_continuation.build_continuation_table` bitwise (difference 0.0 DW); the T = 3 nested self-convergence (panel width halved, nodes doubled) changes V~_2 by 3.58e-9 DW (limit 1e-8); the agreement of the tables with the verifier on the refined tier (state step 0.25, 64 GL nodes per half) is 4.4e-7 DW (V~_2) and 4.2e-7 DW (V~_3 against the stage-2 Q) (limit 1e-6); the T = 3 strata: D_2 40 / 44 bins with no tail bin, D_3 80 / 88 bins with 60 / 64 tail bins; a candidate equal to the best response gives R = 2.9e-14; the development-tier best-response search has a noise floor of up to 0.47 effort units (R up to 0.0075 at q = 50), see `01_calibration.md` Section 3.

## 4. Independent review before the code commit

Four agents wrote the tests / calibration / analysis code (their reports are summarised in the sections above and in `01_calibration.md`), and three further agents reviewed the implementation read-only, each with scripts of its own (no repository edits): (a) conformance with D2-D7 and §2.1-2.2, (b) RNG and numerical identity (C-MS1 loop equivalence, purity of the checks, frozen snapshots, table builds, global-RNG hardening, boundary floats), (c) the state machine and the sampler by fuzzing (43,000 random check sequences against an independent reference simulator written from the D4 text, 0 mismatches; 12 million sampler draws; 390 random-policy `stage_diag` evaluations). No blocker and no major finding. Dispositions:

| finding (severity) | disposition |
|---|---|
| the final global-RNG assertion preceded the stage-1 decomposition and the writers (minor, RNG review) | **fixed**: `check("end_of_run")` now follows `build_gates` and the drift test, as in the locked entry point; `tests/test_ms_runner.py::test_a_global_rng_draw_after_the_stage_1_phase_is_caught_and_labelled` |
| an exit-5 run kept the gate-based outcome label (minor) | **fixed**: the outcome is `global_rng_violation`, the gate label is kept as `outcome_gates`; same test |
| `gap_bin_index` puts one development node (q = 60: d = 80; final tier d = -70, 80) in the bin below the sampler's edge convention by float rounding (minor, two reviewers) | documented (02_preregistration.md section 3): D3 prescribes `gap_bin_index`; the effect is one node of one bin of the per-bin map; R, R_tail, Delta, C are unaffected |
| classification of a block end from an invalid last check (minor) | documented (section 3); reachable only if the verifier flags the result invalid |
| the check CSV does not hold the per-bin map (minor) | documented (section 4 item 11): the map is in `ms_binmaps_stage{t}.npz`, `r_t(d)` in `freeze_stage{t}_*.npz` |
| `rule_log.json` block field `S` has two meanings (minor) | documented (section 4 item 12) |
| `lambda_M > 0` refused on a float difference, inexact at the boundary (minor) | not changed: the smallest `lambda_M` among the arms is 0.15 |
| landing LR end and the strata half-width 20 are module constants, not config keys (minor) | documented (section 4 item 14) |
| tail nodes at exactly `2q (T - t + 1)` are in no bin map; the T = 3 stage-2 end nodes belong to neither R nor R_tail (notes) | documented (sections 2 and 3); T = 3 is a test only |
| the landing sampler setting is the followed block's setting as of the block end (note) | documented (section 3) |
| `a_dev` candidate set includes `a_br` (note) | documented (section 3) |
| block-end checks count in the M-streak, so a non-aligned `N_block` shortens the streak span (note) | the pre-registered `N_block` (400 / 200) and `K` (25) are aligned |
| bare `NaN` tokens in `gates.json` / `rule_log.json` (note) | not changed (Python reads them; the repository's writers do the same) |
| bit identity depends on the host's MKL / oneDNN path (note) | not an issue here: the references and the new runs were produced on the same host (`vector2` in every manifest) with the same versions (python 3.12.3, torch 2.5.1+cu121, numpy 2.5.0); C-R4 (unchanged v2.0 code) is the control that would separate a host effect from a code effect, and it passes |
| style: lines over 100 characters, two public members without docstring (notes) | docstrings added for `StageController.finished` and `is_check_due`; line lengths follow the surrounding code |

## 5. What changed after the checks

Nothing in `run/`, `utils/`, `envs/`, `agents/`, `tools/`, `tests/` or `protocols/` after commit `c95a2af4`. `git diff --stat c95a2af4 <code commit> -- . ':(exclude)reports' ':(exclude)results' ':(exclude)docs'` is recorded in `02_preregistration.md` section 9.

---

# Addendum 1 (2026-10-07): P2 checks

Nothing above is edited.

## 6. Tests after the addendum code (`9b719715`)

Full suite `pytest tests` on the final tree (2026-10-07 05:21-05:49, tmux): `1 failed, 932 passed, 2 xfailed, 2 warnings in 1651.91s (0:27:31)`; the failure is the known `tests/test_registry_canonicalization.py::test_registry_canonicalization`. 932 = 881 (P1) + 51 new tests (C-MS2 with tampered copies and the CLI, the real LR schedule of `MS_base2400`, the launcher with 120 runs and the arm's config, the 7-arm analysis, the secondary table and its blind recomputation, A3 (a)-(d)). Independent read-only review of the diff by an agent that did not write it: highest severity MINOR (four findings, all fixed before the commit; listed in `02_preregistration.md` Addendum 1, section A.7).

## 7. Launch checks of the pilot wave (`results/ms_r1/pilot/launch_checks.json`)

```
python tools/ms/launch_checks.py --root results/ms_r1/pilot --arms MS_rule MS_s25a0 MS_s25a5 MS_s35a0 MS_s35a5 MS_base2400 \
    --code-commit f969d55026b59edaaaac34be633b42b447f9af3c --out results/ms_r1/pilot/launch_checks.json \
    --cms2-ref /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine/parents_A --cms2-arm MS_base2400
all_ok true; status 120, manifest 120, files 120, global_rng 120, tail_share_coverage 120 (of 120)
start-share tests: 1860, flagged (|z| > 3) 3, expected under the null 5.02
C-MS2 (MS_base2400 against parents_A through update 1201) identical 20/20 ALL=True
```

Launch record `results/ms_r1/pilot/launch_20261007_055045.json`: 120 runs, 40 workers, all exit 0, HEAD `f969d550`, `git diff --stat c95a2af4 HEAD -- run utils envs agents protocols` empty, `git status --porcelain` empty, parameter file SHA-256 `0fbfc01857c5abe339ab1926361cd57d6e988da20e754fd6dea3ffa4e076b406`, wall per run 503-654 s (median 550 s). The three flagged start-share tests are q = 60 seeds 10506 (`MS_s25a0`, `MS_s25a5`: landing window, middle stratum, z = -3.13, -3.00) and 10507 (`MS_rule`: polishing block, near-tie stratum, z = -3.10).

C-MS2 detail (the tool and its negative controls are `tools/ms/launch_checks.py:cms2_run` and `tests/test_ms_cms1.py`): through update 1201 the 48 weight exports `u0025 ... u1200`, the per-update series of updates 1..1201 (including the learning rate: 3e-4 at update 1201 on both sides) and the five stream positions after every update are equal bit for bit in 20 of 20 runs, and the export `u1225` differs from `parents_A`'s in 20 of 20.

## 8. Analysis and blind recomputation

```
python tools/ms/r1_analysis.py --base-root <P1 worktree>/results/ms_r1/base --pilot-root results/ms_r1/pilot \
    --parents-root .../v2_refine/parents_A --rehearsal-root .../v2_T2_locked/rehearsal_v2_0 \
    --calibration-root results/ms_r1/calibration --out results/ms_r1/analysis        # exit 0: every planned run is done
python tools/ms/blind_criterion.py --analysis-dir results/ms_r1/analysis
ALL 255 numbers agree with criterion.csv and criterion_vs_MS_base2400.csv to 1e-12 (floats) / exactly (counts, flags, lists)
```

(`results/ms_r1/analysis/blind_recomputation.txt`; the blind script imports nothing from the analysis tool.)
