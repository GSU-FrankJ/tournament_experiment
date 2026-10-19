# T3-R2, P1: checks (tests, the independent review, dispositions)

Date: 2026-10-19. Branch `t3-r2`. Code commits: `67336a38cca03215c467096e28d3717a54cf32fb` (the new tools and tests) and `897982f28b820108ba7a8d1e9ebd3ec6407f4141` (the fixes of section 3). No existing file of `agents/`, `envs/`, `utils/`, `run/`, `protocols/`, `tools/ms/`, `tools/v2/` or `tools/t3/` was changed (`git diff --name-status be2b7e9f HEAD -- agents envs utils run protocols tools/ms tools/v2 tools/t3` lists additions only, all `tools/t3/t3c_*` and `tools/t3/t3r2_*`; the same filter the launcher applies, `t3c_launch.tree_problems`).

## 1. New tests

76 tests in five files (`pytest --collect-only`): all pass at `897982f2` (`pytest tests/test_t3r2_*.py`: 76 passed in 141 s).

| file | tests | what it checks |
|---|---|---|
| `tests/test_t3r2_calc.py` | 14 | AST independence of the calculator; `F_xi` / `f_xi` against numerical integration; the shock rule; the T = 2 closed form at q = 50 and q = 60 (Δ, G at the root and at 50 states of D_2 at most 1e-6 ΔW; actual ~1e-15); a perturbed closed form with an exactly computable best response and gain; the T = 3 continuation against direct nested adaptive quadrature; the vectorised refinement against `scipy` bounded Brent; both effort bounds; the convergence helper; agreement with the verifier on a synthetic T = 3 policy; an independent brute-force T = 3 chain (added after the review, N2) |
| `tests/test_t3r2_semantics.py` | 6 | A4: the observation at (t = 2, T = 3), the role-1 sign, the stage-2 return, the out-of-grid refusal, the lagged opponent, the stage-3 reward |
| `tests/test_t3r2_audit.py` | 12 | the A7 core on a planted zero-effort stretch given as a sequence of synthetic exports (the stretch nodes, the formation export, the slope of Q at the learned effort, the class, an equilibrium-like node, a node that never forms); A2 and A3 on a reduced `bw` run (identity, the detection of a wrong export); `first_below`, `local_maxima`, the training-window helper, the text comparison of A1; the lock (determinism, no timestamp / commit hash / absolute path) |
| `tests/test_t3r2_launch.py` | 22 | the 40 jobs; the seed and arm guards (eight bad blocks); the identity with T3-R1's `bw` configs outside the D4 fields (against `build_bw_config` and against the 20 recorded configs' reference); the refusal of eleven edited configs; the lock checks (a missing LOCK record, a modified lock JSON or MD, a modified or missing listed tool or input file); the CLI refusals; the plan CSV with its init digests (equal to the digest the run records at construction); the infrastructure classification and the ledger with the single repeat and the stop after a second failure; the git guards (only new `t3c_*` / `t3r2_*` files; untracked `results/` allowed); the real launch path with a fake pool; `t3c_eval` on a reduced run reproducing its freeze arrays |
| `tests/test_t3r2_analysis.py` | 22 | the confirmation rule on synthetic tables (35 = 18 + 17 fails; 36 = 16 + 20 fails by the per-q floor alone; 18 + 18 and 17 + 19 pass; 34 = 17 + 17 fails by the total; 20 + 16 fails), thirteen kinds of failed run, the inclusive limits (also one ulp above), the stop without a verdict (from the list and from an attempt-2 event), a missing ledger, the Clopper-Pearson intervals of the two tools, the pooled / gate-value outputs, the diagnosis on a reduced run, an exception of the evaluation as a failed run, and `t3c_checks` |

Both verdict tools are tested on the same synthetic tables and compared (`bl.compare`) in every case: they agree on counts, intervals and verdict in all of them.

The run-pass rule of `t3c_analysis` and `t3c_blind` also reproduces T3-R1's own gate counts, run for run (`01_audit.md` section 6, `tables/a05g_recount.md`): bw 18 / 18, jt 0 / 0, jt_es 0 / 0.

A defect found by my own tests, before the review: `t3c_analysis.run_row` treated an empty `eval_error` cell (read by pandas as NaN) as an error text, so that every run would have been counted as failed; the synthetic confirmation-rule test failed on it and the fix (`math.isnan` on the cell) went into the first code commit.

## 2. The full test suite

The whole suite, run at the second code commit (`897982f2`) as `pytest` over 12 shards (round-robin shares of the 64 `tests/test_*.py` files, run concurrently, single-threaded each, in the tmux session `t3r2_suite`; each shard file starts with its test files and its start time; summary `full_suite_shards.txt` printed by `report_scripts/suite_summary.py`):

**1,724 passed, 1 failed, 6 skipped, 2 xfailed, 0 errors.** The one failure is the known `tests/test_registry_canonicalization.py::test_registry_canonicalization` ("expected 15 Set-2 gradient runs, got 0"; it fails the same way on `origin/ms-r4`, `reports/t3/r1/00_housekeeping.md`). The second failure that T3-R1 recorded (`test_ms_analysis.py::test_cli_end_to_end_complete_world_writes_every_output`, a latent flake of a temporary-directory name read as a float) did not occur in this run. The skipped and xfailed tests are those of earlier rounds (their skip conditions were not examined). A first single-process run of the suite, started on the first code commit, was stopped after about 55 minutes because it was slow and the code had changed; nothing of it is kept.

| shard | files | result | exit |
|---|---|---|---|
| 00 | 6 | 123 passed, 2 warnings in 215 s | 0 |
| 01 | 6 | 127 passed, 2 xfailed in 194 s | 0 |
| 02 | 5 | 123 passed, 1 warning in 566 s | 0 |
| 03 | 5 | 88 passed in 89 s | 0 |
| 04 | 5 | 136 passed, 3 skipped in 141 s | 0 |
| 05 | 5 | 179 passed, 1 warning in 721 s | 0 |
| 06 | 5 | 215 passed in 286 s | 0 |
| 07 | 5 | 148 passed in 1,039 s | 0 |
| 08 | 5 | 104 passed in 51 s | 0 |
| 09 | 5 | 1 failed, 127 passed, 2 skipped, 2 warnings in 61 s | 1 |
| 10 | 5 | 156 passed, 1 warning in 768 s | 0 |
| 11 | 5 | 198 passed, 1 skipped in 312 s | 0 |

(The 76 new tests are in shards 02 to 06; they pass. The shard script is `results/t3_r2/preflight/tests/shards.sh`.)

## 3. The independent review

One read-only review of the diff and of `01_audit.md` by an agent that wrote none of it (a general-purpose agent, told not to modify anything and to verify against the code and the data; it was given the prompt as the specification and my work, not my conclusions). Scope as §2.5: the calculator, the audit and its report, the launcher / lock / configs, and the rule and the failure counting. It reported **no MAJOR finding and nine MINOR**, plus nine notes. It spot-checked more than 30 report numbers against the CSV / JSON (all matched), re-implemented the T = 3 chain independently (agreement to 3e-7 ΔW), checked that all 800 (planned, recorded) config pairs differ only in the six D4 fields, reproduced the lock drafts byte for byte and the plan CSV, and checked the recount run for run (0 mismatches over 120 runs). Findings and my dispositions (each finding was checked before it was accepted):

| # | finding (severity) | verified | disposition |
|---|---|---|---|
| F1 | `t3c_analysis.py` read the tables with pandas' default float parser, which is not round-trip: on `per_run.csv` 189 to 194 of 200 values per gate column differ from `float(cell)`, and a value one ulp above a limit is counted as passing by the analysis tool but not by the blind tool (MINOR, a departure from "at full float64 precision") | yes: `0.010000000000000002` is read as `0.01` by the default parser (pandas 3.0.3), exactly by `float_precision="round_trip"`; 189 of 200 `Gmax_full_over_dw` values differ by default, 0 round-trip | **fixed** (`897982f2`): `float_precision="round_trip"` at both calls; new test with values one ulp above each limit and exactly on the limits |
| F2 | `t3c_launch.tree_problems` used plain `git status --porcelain`; the runner defines "dirty" without `results/` (`git_state`, the `clean_tree` flag of every manifest), and the tree holds untracked files under `results/` that the tracking rules forbid committing (re-run weights, freeze arrays), so the real launch would always have refused (MINOR, fail-closed, blocks P2) | yes | **fixed** (`897982f2`): `git status --porcelain -- . ':(exclude)results'`, the runner's definition; tests: untracked `results/` allowed, a dirty `reports/` file refused |
| F3 | `01_audit.md` called AP3 "confirmed" although seed 50504 is above 0.005 at 16 of 16 exports (MINOR, wording) | yes (min 0.00510) | **fixed**: AP3 is "partly confirmed" (section 11), and section 9 says what holds for each seed |
| F4 | section 10 of `01_audit.md` cited a full-suite record that was a 32-byte timestamp (MINOR, premature) | yes | **fixed**: the statement now cites the sharded run of section 2 |
| F5 | D3 items reported only partly: A7.4 `adv_raw_mean`, the realised starts and the formation-window signals; A8.3 min / median / max of the shape parts; A5 slope and both best-response efforts at every evaluated state (MINOR) | yes | **fixed**: tables `a07h`, `a07i` (both windows per stretch run, the starts per stratum, `adv_raw_mean`), `a08c` (per run); `t3r2_audit.py` now writes the learned effort, both best-response efforts, the gain and the slope at the Δ_T state and at the root as well; A5 re-run at `897982f2` (its first outputs are kept in `results/t3_r2/audit/at_67336a38/`; every shared column is identical) |
| F6 | the text said the stored best-response efforts agree to 0.018; that holds for the dynamic best response, the one-step argmax differs by up to 0.029 (MINOR) | yes (34.43 against 34.40) | **fixed**: both numbers are given; the explanation is marked as the effort resolution of the verifier's search, not tested further |
| F7 | the comparison table labelled the nodes of the 33 other runs "policy error" although the class is set from the one-step best response alone (MINOR) | yes: those runs have no in-band node below 1, so the label does not apply | **fixed**: the columns are "one-step argmax >= 1 / < 1", with the lowest learned effort among the nodes and the median gap to the best response added |
| F8 | `t3c_eval` called `evaluate_run` through a pool, which catches only `DomainError`, `FloatingPointError` and `ValueError`: any other exception would abort the evaluation of all 40 runs, whereas D4 says an evaluation error is a failed run (MINOR, loud not silent) | yes | **fixed**: `safe_evaluate_run` records any exception as that run's `eval_error`; the re-run rule of D4 applies to it; test with a corrupted checkpoint |
| F9 | "stopped without a verdict" was read only from the key `stopped_without_verdict`; an operator who records only the events would not stop the verdict (MINOR) | yes | **fixed** in both tools: an event of attempt 2 or later also stops the confirmation; test |
| N1 | the convergence test covered only the q = 60 stretch nodes | yes | the four q = 50 stretch runs (5 nodes each) were added as extra rows; all converge (effort change at most 0.00099, values 3e-9 ΔW) |
| N2 | a calculator test was a tautology; no unit test of the T = 3 best-response chain against brute force | yes | the tautology was replaced; the brute-force chain test was added |
| N3 | "not an accident of the machine or the day": the original and the re-runs ran on the same host | yes | wording corrected (a second day on the same host, single-threaded CPU) |
| N4 | A4 "hand-computed": tests 3 and 5 replay the same random draws | yes | wording corrected |
| N5 | the recount compared counts per arm and q, not run for run | yes | now run for run, also against the stored `gates_pass` column (reviewer: 0 mismatches) |
| N6 | the explanation of the `dReach` change (grid-defined reachable set) | the reviewer checked one run: the maximiser sits at the edge of the reachable set | kept; `dReach` is not gated and the (E) rule of D2 for A6 applies to verdicts |
| N7 | audit steps ran while untracked report scripts existed | disclosed in section 1 of `01_audit.md` | no change |
| N8 | `main()` success path untested; the out-of-memory regex goes beyond "OOM kill" | yes | test of the real launch path with a fake pool added; the regex stays (it reads only the run's own log, and D4 allows the log) |
| N9 | any fix to a `t3c_*` file changes the tool hashes of the lock | yes | the lock drafts and §6 of the pre-registration were regenerated after the fixes (the plan CSV did not change) |

Re-run after the fixes (§2.5: a review fix that changes code re-runs the audit parts it touches): A5 and its convergence test, at `897982f2` (`results/t3_r2/a5_rerun_stdout.txt`, `audit/costs.csv`); A1, A2, A3, A6, A7 and A8 use unchanged code paths and were not repeated.

## 4. Departures from the prompt and things to know

- The calculator refines the tabulated best responses (V^BR on whole grids) with a vectorised golden-section search on the same brackets; the individually reported states use `scipy`'s bounded Brent as the prompt says; a test checks that the two agree.
- `t3c_launch.py` writes the plan CSV with its own writer (`write_plan`): `t3_launch.write_plan_csv` has fixed field names and cannot carry the init-digest column the prompt asks for. The other five reused functions are imported unchanged.
- The steps A1, A2, A3 and A5 ran with only `results/` untracked; the later steps ran while the report scripts under `reports/` were untracked (no tool path dirty). The re-run manifests say `clean_tree` true.
- The pandas version here (3.0.3) is not round-trip by default for floats; every table the confirmation tools read is read round-trip, the blind tool through `csv` and `float()`.
- The new tests of the reduced runs take about 2 minutes; the whole new set 141 s.

## 5. P2: notes on the lock, the launch and the checks

- **The lock** (`protocols/v2_T3_locked_v1_0.{json,md}`, lock commit `72bd773f784f40bcd190113acbeb2dd8450d5b9a`, then the LOCK record `2b63ffe4`): written by `tools/t3/t3r2_make_lock.py --out protocols/`, byte-identical to the P1 drafts (`cmp`; SHA-256 `4f02c7ae...` and `14f8e2e7...`); the launcher read the T3 record from `protocols/LOCK` and checked both files and the SHA-256 of every listed tool file. Nothing under `tools/`, `tests/`, `agents/`, `envs/`, `utils/` or `run/` changed between the lock commit and the end of the round, and under `protocols/` only the LOCK record was appended (`git diff --stat 72bd773f HEAD` over those paths shows `protocols/LOCK` with 11 added lines and nothing else).
- **The launch** (`results/t3_r2/confirmation/launch_preconditions.txt`, `launch_*.json`): from `2b63ffe4`, clean tree (the runner's definition), `git diff --stat be2b7e9f HEAD -- agents envs utils run tools/ms tools/v2` empty; 40 workers, 40 of 40 exit 0, `infra_events.json` empty.
- **F10 (found in P2, NOTE; the tool is locked and was not edited):** `t3c_checks.py` compares the config hash with the output directory made relative to the parent of `--root`; with a relative `--root` that step silently keeps the absolute path, so `config_equals_plan` is 0 of 40 on a correct launch. The first invocation used a relative root and reported exactly that (`results/t3_r2/confirmation/launch_checks_first_invocation_relative_root.json`, all other 13 checks 40 of 40); the written configs were verified by hand against the plan (identical SHA-256), and the same tool was run again with an absolute `--root` (`launch_checks.json`, 14 of 14 checks 40 of 40). The review did not find it because its tests and its read-through used absolute paths. A later round should resolve `--root` in the tool.
- **Push:** the commit with the 40 runs, the evaluation and the analysis (`f6b38354`) was pushed right after the analysis, before the reports (the prompt's push points are §3.1 and §4.3). A fast-forward of the round's own branch.
