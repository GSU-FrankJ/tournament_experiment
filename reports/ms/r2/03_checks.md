# MS-R2 P1: tests, independent review, decomposition and calibration checks

Date: 2026-10-07. Spec: `reports/ms/r2/pi_record/19_ms_r2_prompt.md` section 2.2 and 2.3. Everything below was run on the tree that becomes the code commit (the pre-registration, `02_preregistration.md`, names it); between that commit and the launch only `results/` and the report folder change.

## 1. Tests

Full suite `pytest tests` on the tree of the code commit, run as **three parallel invocations over a partition of all 1063 collected tests** (2026-10-07 09:30-09:41, tmux; the partition was checked: 570 + 365 + 128 = 1063 = `pytest tests --collect-only`):

```
pytest tests --ignore-glob='tests/test_ms_*'            1 failed, 567 passed, 2 xfailed, 2 warnings in 984.26s (0:16:24)
pytest <the MS-R1 test files, tests/test_ms_*.py minus test_ms_r2_*>      365 passed in 671.47s (0:11:11)
pytest tests/test_ms_r2_*.py                            128 passed, 1 warning in 581.47s (0:09:41)
```

Together **1060 passed, 1 failed, 2 xfailed**; the one failure is the known `tests/test_registry_canonicalization.py::test_registry_canonicalization`. All 365 MS-R1 tests pass unchanged (no existing test file differs from `71c58904`). An earlier single-invocation run of the whole suite on the preceding tree (`2 failed, 1058 passed, 2 xfailed in 2224s`) found a second failure, the MS-R1 test `tests/test_ms_rule.py::test_record_content_and_json`, which asserts that the logged rule parameters are exactly the `StageRule` dataclass fields: the flag `legacy_global_sampler` had been added to the dataclass. It was moved to a constructor argument of `StageController` (the logged parameters of MS-R1 are again unchanged; `tests/test_ms_r2_runner.py::test_the_legacy_global_sampler_flag_is_a_controller_argument_not_a_rule_field`), and the three-part run above is the run on the final tree.

New test files (all single-threaded, CPU, no network):

| file | tests | what it proves |
|---|---|---|
| `tests/test_ms_r2_runner.py` | 26 | without the new keys the MS runner equals commit `71c58904` (a reduced legacy run and a reduced rule run, each run in a `git archive` checkout of that commit and in the current tree: `state_end_stage2/1.pt` (actor, critic, opponent, both Adam states, the streams, the torch generator, counters), every weight export, `ms_updates.csv` and the check rows equal, no new column); the D2 scale table, applied to the live actor and the lagged opponent, carried by the snapshot refresh and the frozen snapshot, reset at stage-1 entry before the refresh; C-NL, C-MS3, C-MS4 on reduced budgets (the fixed-budget stratified path against MS-R1's rule mode with `rho = 0` and a localized fraction of 1e-9; the bin-balanced path against the `MS_base2400` window); the reporting columns; 12 configuration refusals |
| `tests/test_ms_r2_launch_checks.py` | 9 | a genuine reduced wave passes every check; tampered scale, tampered export scale, a set scale where none is expected, a changed export / series value / stream position, an identical u-after-bound export, wrong references, a missing run, the `p_digest_next` rule of C-MS4, the CLI exit code |
| `tests/test_ms_r2_launcher.py` | 7 | wave `r2` (120 jobs, `pilot/q*/seed*/<arm>`, default root `results/ms_r2`), the six configs differ only in `start_weights` / `derived` and the two optional keys, the stratified settings equal `MS_s35a5`'s, the MS-R1 waves and arm table unchanged, the dry-run record |
| `tests/test_ms_r2_decomposition.py` | 10 | the formulas of `utils/ms_noise.py` (the noise report equals the locked `smoothed_share` on the same network to 1e-12), the decomposition tool on synthetic rows (the ratio check and the stop exit code 3), the Beta recomputation from a stored MS-R1 export, the preamble table reproduced from `results/ms_r1/analysis/per_run.csv` |
| `tests/test_ms_r2_stop_candidates.py` | 23 | the exact best response against a dense grid with parabolic refinement (3.6e-10), J against the linearisation, C1-C3 on a stored MS-R1 export equal to its logged check rows, the `conc_scale` export, the fire rule, the tables, refusals (`--fire` without `--grids`, overwriting) |
| `tests/test_ms_r2_analysis.py` | 53 | the criterion parts (a) and (b) by hand (a violation, a missing run), a fresh bootstrap generator per (q, statistic), transmission and interaction on constructed data, the decomposition columns, the blind script (agreement, tampering, independence), extraction on a genuine reduced wave; 17 deliberate mutations of the tools were caught |

Numbers asked for in section 2.2 of the prompt (measured by the tests, `-s`):

- Beta mean under the scale: the maximum difference of the mean effort between scale s and scale 1 over 301 states and s in {1.5, 4, 16, 100} is 5.96e-6 effort units (limit 1e-4).
- Action spread at d = 0: the sample standard deviation of 10^6 draws equals sigma = e_range sqrt(mu (1 - mu) / (c s + 1)) within 3 standard errors for s in {1, 4, 16}.
- Stage-1 continuation table from a scaled frozen snapshot (s = 4 and 16): the maximum difference from the table at scale 1 is 0.0 DW (limit 1e-6 DW); the table is built from the Beta means, which the scale leaves unchanged here to the last bit.

## 2. Independent review (prompt 2.3)

One read-only review of the diff by an agent that did not write it (conformance with D2-D7, RNG and identity, the scale semantics). Highest severity **MAJOR**, one finding; the others minor or informational. Dispositions:

| finding (severity) | disposition |
|---|---|
| **C-MS4 would fail in 14 of 20 comparisons although the runs are identical** (major): the check row at the block end before a polishing block of MS-R1's `MS_s35a5` carries the polishing sampler's digest in `p_digest_next`; `NL_st_s1` never polishes. Reproduced on real data (q50 seed 10508, update 400: reference `582a3653636d0dc2`, the global setting's digest recomputed from the stored `rho_bar` is `071ba2858bed25cc`) and on a reduced pair. The tests had missed it because the reduced reference never polished | **fixed** in `tools/ms/r2_launch_checks.py`: where the reference polishes (`first_polish_local` is not None) `p_digest_next` is not compared; `p_digest` (the probabilities in force at every update) still is; `tests/test_ms_r2_launch_checks.py::test_c_ms4_ignores_p_digest_next_only_where_the_reference_polishes` (the column is compared where the reference does not polish, a changed `p_digest` fails in both cases); pre-registration section 4 |
| segment boundaries of the analysis differed from D2 (minor): training 1-2001 and ramp 2002-2200 instead of 1-2000 and 2001-2200 | **fixed** (`tools/ms/r2_analysis.py:Windows.segments`, the test, pre-registration 9.8) |
| the pre-registration and the code disagreed on the transmission formula (minor): the prompt's minus sign contradicts its own endpoints | **documented** as decision 10 of section 9 of the pre-registration (the ratio without the minus sign; the reviewer's sign analysis agrees) |
| `state_end_stage2.pt` and `full_state` do not store `conc_scale` (info) | **not an issue**: `state_end_stage2.pt` stores it (`agent['conc_scale'] = {'actor': 4.0, 'opponent': 4.0, 'frozen': None}` in a reduced `NL_st_s4` run); recorded as decision 12 |
| the concentration statistic C scales as 1/sqrt(s), so the would-fire eligibility is easier at s > 1 (info) | recorded (decision 13); the would-fire record decides nothing |
| pre-registration section 10 still had the tests placeholder (info) | filled |
| the full suite found a second failure after the review: the MS-R1 test `test_ms_rule.py::test_record_content_and_json` (the flag as a `StageRule` field changed the logged rule parameters of every run) | **fixed**: the flag is a `StageController` constructor argument; all 365 MS-R1 tests pass unchanged (section 1) |

Checks the reviewer ran and reported as passing: the constrained paths (`protocols`, `envs`, `agents`, `run/run_v2_*`, `run/v2_rollout.py`, `utils/v2_continuation.py`, `utils/v2_metrics.py`, `utils/dp_br_verifier.py`, `tools/v2`) are byte-identical to `71c58904`; no existing test file differs from HEAD; the closed form enters only the reporting columns; every new branch of `run/run_ms_stagewise.py` and `utils/ms_rule.py` is guarded by an absent-key condition; the runner's scale function equals the literal D2 table at every update j in 1..2800 for s = 4 and 16 (0 mismatches); the reporting columns are pure (reduced `NL_st_s4` and `NL_bb_s16` with and without `noise_report`: weight exports, the five stream positions, the common check columns, `state_end` and the global torch / numpy RNG states identical; `gap == g2_0 - e2_at_0` exactly in 288 rows); `build_config` hashes of all MS-R1 arms (84 combinations) and the `base` / `pilot` job lists identical at HEAD and in the working tree; the analysis ran on the mini wave and the blind recomputation agreed (220 numbers); `tie_noise_report` takes 3 ms per call on a real export (112 checks per run: 0.3 s per run).

## 3. Decomposition (section 2.4) and offline calibration (2.5)

The premise holds and the stop condition is not met: `reports/ms/r2/01_decomposition.md` (ratio check 0.99927-0.99971 in all 140 runs; the six rows of the preamble table reproduced to the second decimal; one difference of the second decimal in a quoted range, section 4 of that report). The calibration of the three closed-form-free stop criteria on the 140 MS-R1 runs: `reports/ms/r2/01b_stop_candidates.md`; the replay reproduces every logged check row of the 12,800 exports with maximum absolute difference 0; the grids were committed (`93d8be5d`) before the fire tables were computed.

## 4. Provenance of the tools

`tools/ms/r2_decomposition.py`, `r2_stop_candidates.py`, `r2_analysis.py`, `r2_blind_criterion.py`, `r2_launch_checks.py` and the extended `launch_ms_r1.py` / `ms_configs.py` are in the code commit; the formulas shared by the runner and the tools are `utils/ms_noise.py`. The independent blind recomputation (`r2_blind_criterion.py`) imports nothing from `tools/ms/` (checked by an AST scan in the tests and by the reviewer).
