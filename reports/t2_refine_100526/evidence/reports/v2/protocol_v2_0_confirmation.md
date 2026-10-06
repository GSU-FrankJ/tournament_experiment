# v2 T=2 — protocol v2.0, re-rehearsal, and fresh-seed confirmation

Date: 2026-10-04. Worktree `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine`, branch `v2-t2-refine`. `origin/v2-t2-refine` is at `155cdec` (the end of the R1 round); every commit named below after it is not pushed.

Path conventions:
- `LK/` means `results/v2_T2_locked/`; `V20/` means `LK/v2_0/`.
- `CA/` means `LK/confirmation_v2_0_analysis/`: the pre-registered `tools/v2/confirmation_analysis_v2_0.py` on the confirmation root, invoked exactly as in `protocols/v2_T2_locked_v2_0.md` §9.
- `CB/` means `LK/confirmation_v2_0_analysis_vs_v1_1/`: the same script with `--compare-root` (the v1.1 confirmation) added. Its outputs equal those of `CA/` byte for byte, except `tables.md` and `verdict.json`, which gain the comparison; `CB/` also holds one extra file, `compare_distributions.csv` (`LK/confirmation_v2_0_checks.txt`).
- `RA/` means `LK/rehearsal_v2_0_analysis/`: the same script on the re-rehearsal root.
- `CHK/` means `LK/rehearsal_v2_0_checks/`.

Tables marked "verbatim" are copied from the `tables.md` of the directory named. Every other number is read from the file named next to it.

## Verdict

**Confirmation verdict (protocol v2.0, as computed): PASS** (`CA/verdict.json`).

| q | primary passes (G-A ∧ G-F ∧ G-N ∧ G-S) | rule | exact 95% CI (Clopper–Pearson) | q passes |
|---|---|---|---|---|
| 50 | **19 / 20** | >= 18 of 20 | [0.7513, 0.9987] | yes |
| 60 | **20 / 20** | >= 18 of 20 | [0.8316, 1.0000] | yes |

Source: `CA/pass_counts.csv`.
- **Overall: PASS**: both q pass.
- The one failed run is q=50 seed 30510: outcome `stage2_failure`. Its G-A η₂ is 0.00580334 against the threshold 0.005 (`G-A_eta_pass` False); its RMSE (0.0451) and tail (0.008822) pass; G-F, G-N and G-S pass (stage-1 error +0.008284). `CA/per_run.csv`. It was not re-run (§4).
- 40/40 runs completed with exit code 0. There were no global-RNG violations (0), no exceptions (0), no incomplete or missing runs, and no infrastructure crash (`LK/confirmation_v2_0/crashed/` does not exist; `LK/confirmation_v2_0/launcher.out`: 40 lines, 40 with rc=0).
- The script's recomputed values and verdicts agree with every run's `gates.json` in 40/40, with a maximum absolute value difference of 0 (`CA/agreement.csv`).
- Other criteria, reported and not gated (`CA/pass_counts.csv`): G-S 20/20 and 20/20; S1 at 0.10 20/20 and 20/20; v1.1 outcome (G-A ∧ G-F ∧ G-N) 19/20 and 20/20; v1.0 outcome 19/20 and 20/20 (q=50, q=60).
- Re-rehearsal checks: R1–R6 pass; R7 is false as the tool computed it and is accepted as met on the canonical-reference comparison by owner decision D-R7 (§3 and §6, deviation 1).

---

## 1. The v2.0 lock

### 1.1 Commits, files, hashes

| Item | Value |
|---|---|
| end of the R1 round (parent of all v2.0 work) | `155cdec` |
| R1 acceptance addendum (PI decisions D1 and D3) | `ce0573f` (2026-10-03T23:35:17+00:00) |
| **v2.0 lock commit** | **`1d6d4d0`** (`1d6d4d00736b265a18ae71b91ecb77b19e4915b9`) |
| LOCK record commit | `f2d616c`: the v2.0 record appended to `protocols/LOCK` (the v1.0 and v1.1 records unchanged), the two seed-inventory scans at the lock commit, the launcher `LK/launch_one_v2_0.sh`, the two `jobs.txt` |
| `protocols/v2_T2_locked_v2_0.json` | SHA-256 `6ca0a6f95ee8109aeca7c787186fb40f5fec80a09c113b880fe80d20f1d47fd3` (current file: equal) |
| `protocols/v2_T2_locked_v2_0.md` | SHA-256 `efbb4f5af832494b8b2f7e992d3b05b178a2570861b0959cb4135687d3a95937` (current file: equal) |
| `run/run_v2_T2_locked.py` | SHA-256 at the lock `b276a2ce5e1b6439f32d34bc980968a1482c787645b4afb0ccbfb8c00773c356` (current file: equal) |
| `tools/v2/confirmation_analysis_v2_0.py` | SHA-256 `9cdbaf7c1fe85787ae99a8abdd3bab5987bd3b7e3742c6a1564ac80bf0c5fa90` (current file: equal) |
| `utils/v2_continuation.py` | SHA-256 `513fef4b304e46606fd2bc436c628c90a4893b36979a454cb00924163bb50b4a` (current file: equal) |
| v1.0 and v1.1 files | `protocols/v2_T2_locked.json`, `protocols/v2_T2_locked_v1_1.json` and `.md`, `tools/v2/confirmation_analysis.py` are byte-identical; a test asserts it (`test_earlier_protocol_files_untouched`). v1.1 stays reproducible from `431474d`, v1.0 from `4bd2214` |

The full values are in `protocols/LOCK` (the v2.0 record is its last record).

**What the lock commit contains** (`git show --name-only 1d6d4d0`: 18 paths):
- `protocols/v2_T2_locked_v2_0.json`
- `protocols/v2_T2_locked_v2_0.md`
- `run/run_v2_T2_locked.py`
- `tests/test_v2_locked.py` (deleted: replaced by `tests/test_v2_locked_v2_0.py`, §6 deviation 5)
- `tests/test_v2_locked_v2_0.py`
- `tests/test_v2_refine_continuation.py`
- `tools/v2/confirmation_analysis_v2_0.py`
- `tools/v2/make_locked_protocol_v2_0.py`
- `tools/v2/seed_inventory.py`
- `tools/v2/v2_0_pass_probability.py`
- `tools/v2/v2_0_rehearsal_checks.py`
- and 7 records under `V20/`: `continuation_check_v2_0.json`, `continuation_check_v2_0.out`, `protocol_diff_v1_1_to_v2_0.json`, `seed_inventory_pre_lock.csv`, `seed_inventory_pre_lock.out`, `v2_0_pass_probability.json`, `verifier_numerics_note.md`.

**Nothing locked changed afterwards.**
- Outside `results/`, `git diff --name-only 1d6d4d0 d2e377d` lists only: `protocols/LOCK`; the whole diff is 270 paths, 269 of them under `results/` (73432 insertions(+); `LK/confirmation_v2_0/launch_env.txt`).
- Up to the fix commit `3fedaa2` (made after the confirmation, §6 deviation 1) the list is: `protocols/LOCK`, `tools/v2/v2_0_rehearsal_checks.py`.

### 1.2 v1.1 → v2.0 JSON diff

Source: `V20/protocol_diff_v1_1_to_v2_0.json`, 30 entries, generated by `tools/v2/make_locked_protocol_v2_0.py` (grouping by decision is mine).

| Group | Entries |
|---|---|
| version bump | 8: changed `/date`; changed `/name`; changed `/schema`; changed `/supersedes/file`; changed `/supersedes/lock_commit`; changed `/supersedes/sha256`; changed `/supersedes/version`; changed `/version` |
| D2 expected continuation | 7: added `/pipeline/continuation_table`; changed `/pipeline/entry_point_reads`; added `/pipeline/flags/continuation_value_mode`; changed `/pipeline/flags_note`; changed `/pipeline/outputs`; added `/pipeline/phase_B/continuation_value_mode`; changed `/training_relevant_state/compared` |
| D4 gate G-S | 3: added `/gates/G-S`; changed `/gates/run_outcome`; added `/v1_1_outcome_reported` |
| D5 confirmation block | 9: changed `/confirmation/analysis_script`; changed `/confirmation/crash_rule`; changed `/confirmation/disjointness_check`; changed `/confirmation/pass_rule`; added `/confirmation/rehearsal`; added `/confirmation/root`; changed `/confirmation/seed_block`; changed `/confirmation/seed_block_status`; changed `/confirmation/status` |
| change log, evidence, known issues | 3: changed `/change_log`; added `/evidence/v2.0 changes`; changed `/known_issues` |

**Nothing else changed.** The per-q records (game, PPO, verifier, cadence, budgets, RNG namespaces), the Phase-B flags other than `continuation_value_mode`, the LR windows, G-A, G-F, G-N, the decomposition method and the reported list are identical (the diff above is the complete list).

### 1.3 Change log

Verbatim from `protocols/v2_T2_locked_v2_0.md` §7.

**The PI decided every change after the R1 round and before any confirmation seed was run.**

1. **Phase B runs with continuation_value_mode=expected: for the stage-1 rows only, the sampled continuation in the stage-1 return is replaced by the table value V2~(y), y = e1 - e1_opp, built once at Phase-B entry from the frozen stage-2 Beta mean by utils/v2_continuation.py with the R1 default rule (composite Gauss-Legendre, 6 nodes per panel, panels of width 1 aligned at z = 0 and z = +-2q, y-grid step 0.05, float64, linear interpolation). Phase A is unchanged**
   - Why: the only R1 arm that met the pre-registered criterion at both q (method 6, arm B_expcont): paired difference (arm - B_base) of |relative stage-1 error| -0.02386 [-0.04061, -0.009831] (q=50) and -0.03783 [-0.06321, -0.01472] (q=60); across-seed SD of the signed error 0.3464x and 0.2687x the baseline's; mean per-pair ratio of the stage-1 advantage SD 0.0656 and 0.0531; Phase-B wall time 1.084x (including the table build). No other R1 arm enters the protocol; target-KL changes cost, not accuracy, and is not adopted.
   - Evidence: `reports/v2/refine/04_pilot_stage1.md`; `reports/v2/refine/06_decision_inputs.md section 1`; `results/v2_refine/analysis/stage1_criterion.csv`; `results/v2_refine/analysis/stage1_dispersion.csv`; `results/v2_refine/analysis/stage1_adv_ratio.csv`; `results/v2_refine/analysis/stage1_cost.csv`; `reports/v2/refine/01_preregistration.md section 4 item 6 and Addendum 1`.

2. **the pipeline code changed at 32a8c21 (R1 flags, every new key at its default behaves as before)**
   - Why: needed to run the expected continuation; v1.1 is unaffected: the unchanged v1.1 entry point reproduces rehearsal_v1_1 in 20/20 runs on the changed code (C-R1), and parents_A, B_base and A_base reproduce it in 20/20 each.
   - Evidence: `results/v2_refine/v11_reproduction_checks.json`; `results/v2_refine/parents_A_checks.json`; `results/v2_refine/stage1_base_checks.json`; `results/v2_refine/stage2_base_checks.json`; `results/v2_refine/c7/code_check.compare.txt`.

3. **check (ii) of the continuation table settled: the literal R1 criterion (<= 1e-6 DW against the verifier's standard final tier) is replaced by (ii-a) self-convergence <= 1e-8 DW, (ii-b) refined-verifier agreement <= 1e-6 DW (state step 0.25, 64 Gauss-Legendre nodes) and (ii-c) training-side sensitivity: shift of the stage-1 optimum <= 1e-3 e1* when the convergence difference profile is added**
   - Why: the standard final tier interpolates V2 onto a state grid of step 2; the gap it shows (3.229e-05 DW at q=50, 1.96e-05 DW at q=60) is of the order of the dev - final differences G-N exists for, while the table is converged: (ii-a) 4.76e-09 / 3.22e-09 DW, (ii-b) 4.98e-07 / 4.1e-07 DW, (ii-c) shift of the stage-1 optimum 2.14e-05 / 0 e1* (q=50 / q=60; limit 1e-3). Verifier numerics item (reported, not gated): the standard-tier gap 3.229e-05 / 1.96e-05 DW would imply a stage-1 optimum shift of 0.00214 / 0.00342 e1* if it were a table error.
   - Evidence: `results/v2_T2_locked/v2_0/continuation_check_v2_0.json`; `results/v2_T2_locked/v2_0/verifier_numerics_note.md`; `results/v2_refine/continuation_check.json`; `tests/test_v2_refine_continuation.py`.

4. **gate G-S added: |e_hat_1(0) - e1*(0)| / e1*(0) <= 0.05 on the end-of-B last iterate (the v1.1 secondary criterion S1 with threshold 0.05, the 0929 target); run pass = G-A and G-F and G-N and G-S**
   - Why: threshold evidence from the locked pipeline's own distribution before any confirmation seed: arm B_expcont on the development seeds (the v2.0 rehearsal must reproduce it bit-identically: check R2): 10/10 and 10/10 within 0.05, max |error| 0.04069 (q=50) and 0.03144 (q=60), signed error mean -0.01181 / 0.000906, SD (ddof=1) 0.01785 / 0.01874. Normal model (G-A, G-F, G-N treated as passing): P(>= 18 of 20 fresh runs pass G-S) = 0.9959 (q=50) and 0.9995 (q=60). S1 at 0.10, the v1.1 outcome and the v1.0 outcome are reported for every run and decide nothing. Safety condition on the v2.0 rehearsal: >= 19/20 under G-S and a recomputed normal-model probability >= 0.90 at each q.
   - Evidence: `results/v2_T2_locked/v2_0/v2_0_pass_probability.json (tools/v2/v2_0_pass_probability.py)`; `results/v2_refine/analysis/stage1_per_run.csv`.

5. **confirmation seed block 30501-30520, used for both q (40 runs); pass rule per q >= 18 of 20 runs pass (G-A, G-F, G-N, G-S), both q, exact Clopper-Pearson 95% CIs**
   - Why: seeds reserved and unused; the inventory of every seed value recorded in the tournament_experiment tree (305 distinct values in 47088 json and 8782 csv files) found 0 collisions with the block, taken before the block appeared in any JSON or CSV seed record (the block had been reserved and unused since the R1 pre-flight: reports/v2/refine/00_preflight.md, results/v2_refine/preflight/seed_inventory_30501_30520.out).
   - Evidence: `results/v2_T2_locked/v2_0/seed_inventory_pre_lock.csv`; `results/v2_T2_locked/v2_0/seed_inventory_pre_lock.out`; `tools/v2/seed_inventory.py`.

### 1.4 Tests and C7

- **Full suite** (`.venv`, `-p no:cacheprovider`, single thread):
  - at the lock content, before the lock commit: `1 failed, 390 passed, 1 skipped, 2 xfailed, 2 warnings in 771.43s (0:12:51)` (`V20/fullsuite_prelock.txt`; the one skip is the LOCK-record test, which waits for the record written in `f2d616c`);
  - at the launch-tree code (R7 of the re-rehearsal, run at `f2d616c`): `1 failed, 391 passed, 2 xfailed, 2 warnings in 757.28s (0:12:37)` (`LK/rehearsal_v2_0_pytest.txt`);
  - at the fix commit `3fedaa2` (tool re-run, §6 deviation 1): `1 failed, 391 passed, 2 xfailed, 2 warnings in 803.07s (0:13:23)` (`CHK/tool_fix_rerun/rehearsal_v2_0_pytest.txt`).
  - The single failure in each is the known `tests/test_registry_canonicalization.py::test_registry_canonicalization` (paper registry vs data on disk; pre-existing, unrelated to v2; test and data unchanged).
- **`tests/test_v2_locked_v2_0.py`**: 34 test functions (the v1.1 file had 15, all ported, §6 deviation 5). They cover: hash and refusals (modified protocol, changed table-rule parameter, another table rule in the code), the LR schedule at every update of A and B, gate logic on synthetic values (values exactly at each threshold; run pass = G-A ∧ G-F ∧ G-N ∧ G-S; S1 at 0.10, the v1.1 and the v1.0 outcome do not enter it), the table written by the entry point against `utils.v2_continuation` called directly on the frozen snapshot, no RNG moved by the build, the global-RNG hardening as in v1.1, the analysis script on synthetic and smoke roots, the safety condition.
- **`tests/test_v2_refine_continuation.py`**: (ii-a), (ii-b), (ii-c) in place of the two strict expected failures (§2).
- **C7** (the v2 runner at default flags equals the pre-refactor reference, bit for bit):
  - at the re-rehearsal launch commit `f2d616c` (code identical to the lock commit apart from `protocols/LOCK`, as at the confirmation launch commit `d2e377d`): the tool's own comparison is **false** (reference `checkpoint.pt` not in this worktree); against the canonical reference it is **IDENTICAL including `checkpoint.pt`** (`CHK/c7_canonical_reference.txt`, two candidates, both manifests `dirty: false`; owner decision D-R7).
  - at the fix commit `3fedaa2`, by the fixed tool: c7_identical **True**, c7_commit `3fedaa2`, c7_dirty False, R7 pass **True** (`CHK/tool_fix_rerun/r7_result.json`; `results/v2_pilots/phase2_regression/v2_full_3fedaa2.compare.txt`).

### 1.5 Timeline (UTC)

| Event | Commit | Time |
|---|---|---|
| R1 acceptance addendum (PI decisions D1 and D3) | `ce0573f` | 2026-10-03T23:35:17+00:00 |
| **v2.0 lock commit** | `1d6d4d0` | 2026-10-03T23:35:28+00:00 |
| LOCK v2.0 record, seed inventories, launcher | `f2d616c` | 2026-10-03T23:36:57+00:00 |
| Re-rehearsal launch (at `f2d616c`) | — | 2026-10-03T23:37:12Z (`LK/rehearsal_v2_0/launch_time_utc.txt`) |
| Re-rehearsal finished (20/20 rc=0) | — | 2026-10-03T23:43:07Z (latest `end_time` in `LK/rehearsal_v2_0/q*/seed*/status.json`; the host's local time is UTC) |
| Checks R1–R7 finished; R7 false as computed; confirmation not launched; stop | — | 2026-10-03T23:59:15Z (`LK/rehearsal_v2_0_checks.out`, mtime) |
| Owner decision D-R7: R7 accepted as met on the canonical-reference comparison | — | 2026-10-04 |
| C7 canonical-reference record written | — | 2026-10-04T04:23:05Z (`CHK/c7_canonical_reference.txt`) |
| Re-rehearsal records, checks, D-R7 record (results only) | `d2e377d` | 2026-10-04T04:24:27+00:00 |
| **Confirmation launch** (at `d2e377d`) | — | 2026-10-04T04:24:47Z (`LK/confirmation_v2_0/launch_time_utc.txt`) |
| Confirmation finished (40/40 rc=0) | — | 2026-10-04T04:37:16Z (latest `end_time` in `LK/confirmation_v2_0/q*/seed*/status.json`) |
| Pre-registered analysis run | — | 2026-10-04T04:37:35Z (`LK/confirmation_v2_0_analysis_start_utc.txt`) |
| Confirmation records and analysis output | `85c294e` | 2026-10-04T04:38:49+00:00 |
| Checks-tool fix (reference path) | `3fedaa2` | 2026-10-04T04:39:28+00:00 |
| R7 re-run at the fix commit | — | 2026-10-04T04:39:53Z to 2026-10-04T04:53:40Z (`CHK/tool_fix_rerun/`) |

Launch-commit rule (prompt §4):
- `git diff --name-only 1d6d4d0 d2e377d` outside `results/` lists only `protocols/LOCK`; `git diff --stat 1d6d4d0 d2e377d`: 270 files changed, 73432 insertions(+), 270 paths of which 269 are under `results/` (`LK/confirmation_v2_0/launch_env.txt`, `launch_diff_stat.txt`).
- The same holds for the rehearsal launch commit `f2d616c` (`LK/rehearsal_v2_0/launch_env.txt`).

---

## 2. Check (ii) of the continuation table

Decision D3 replaced the literal R1 criterion (≤ 1e-6·ΔW against the verifier's standard final tier) by three tests. Source: `V20/continuation_check_v2_0.json` (`all_pass` True; written by `tests/test_v2_refine_continuation.py --write`, 329 s).

| test | criterion | q = 50 | q = 60 | passes |
|---|---|---|---|---|
| (ii-a) self-convergence: panel width 1 → 0.5, nodes 6 → 12 | max change ≤ 1e-8·ΔW | 4.76e-09 | 3.22e-09 | yes |
| (ii-b) refined verifier (state step 0.25, 64 GL nodes), fixed stage-1 opponent 40 / 45 | max over the effort grid of \|Ṽ₂(e − ê₁(0)) − (Q₁(0, e) + k e²)\| ≤ 1e-6·ΔW | 4.98e-07 | 4.1e-07 | yes |
| (ii-c) training-side sensitivity: finest rule (panel 0.25, 24 nodes) minus default added to the table, shift of argmax_e[−k e² + Ṽ₂(e − ê₁(0))] on a 0.001 grid | shift ≤ 1e-3·e₁* | -0.001 = 2.14e-05·e₁* | 0 = 0·e₁* | yes |

**Verifier numerics item** (reported, not gated; `V20/verifier_numerics_note.md`, key `verifier_numerics` of the record above): on the standard final tier (state step 2, 32 GL nodes per half interval) the stage-1 value of the verifier differs from the table by 3.229e-05·ΔW (q = 50) and 1.96e-05·ΔW (q = 60). Were that a table error it would move the stage-1 optimum by -0.1 and -0.133 effort units (0.00214·e₁* and 0.00342·e₁*). It bounds the precision of the verifier's own ẽ₁ on that tier.

(Test T7 of the pre-lock review: (ii-a)–(ii-c) do not cover the interpolation error of the step-0.05 linear lookup between table nodes, so "converged" refers to the node values; §6 deviation 3, finding T7.)

---

## 3. Re-rehearsal under v2.0 (development seeds)

**Design.**
- 20 runs: q ∈ {50, 60} × seeds 10501–10510, through `run/run_v2_T2_locked.py` from scratch, into `LK/rehearsal_v2_0/`.
- Launch commit `f2d616c`; `nproc` 64; `23:37:12 up 5 days,  7:17,  7 users,  load average: 2.31, 1.98, 1.84`; `/dev/md0        5.5T  3.1T  2.2T  60% /home` (`LK/rehearsal_v2_0/launch_env.txt`).
- 20 parallel single-threaded processes (OMP/MKL/OpenBLAS = 1), `xargs -P 20` in tmux, through `LK/launch_one_v2_0.sh`.

**Checks** (`tools/v2/v2_0_rehearsal_checks.py` → `LK/rehearsal_v2_0_checks.json`; the tool's own verdict `ALL_PASS: False`, from R7 alone; the tool's original file is kept in `CHK/rehearsal_v2_0_checks.tool_output.json`, SHA-256 `981724df27daaa0ff3f3514e07b8d9308d21e1751e849b2282eea60190d1b57f`):

| Check | Verdict | Detail |
|---|---|---|
| R1 Phase A equals `rehearsal_v1_1` | **pass** | 20/20 identical in every compared field (`LK/rehearsal_v2_0_checks_R1_details.csv`); snapshot-refresh counter (end of A): 82 in the new run against 82 in `rehearsal_v1_1`, equal in 20/20 (`R1.snapshot_refreshes` in `LK/rehearsal_v2_0_checks.json`) |
| R2 Phase B equals the R1 pilot `B_expcont` | **pass** | 20/20 identical, including the gate-metric values and the table (rebuilt from the pilot's frozen snapshot) (`LK/rehearsal_v2_0_checks_R2_details.csv`); snapshot-refresh counter (end of B), reported separately: 113 against 113 in the pilot, equal in 20/20 (`R2.snapshot_refreshes_reported_separately` in `LK/rehearsal_v2_0_checks.json`) |
| R3 G-A, G-F, G-N, G-S | **pass** | 20/20 runs pass |
| R3b (D4 safety condition) | **pass** | 20/20 under G-S (needed 19); recomputed normal-model P(≥ 18 of 20 fresh runs pass G-S) = 0.9959 (q=50; mean -0.0118, SD 0.0178) and 0.9995 (q=60; mean +0.00091, SD 0.0187); limit 0.90 (`RA/rehearsal_safety.json`) |
| R4 global RNGs, return codes | **pass** | 20/20 |
| R5 manifests (v2.0 hash and version, launch commit, `clean_tree: true`, table SHA-256) | **pass** | 20/20 |
| R6 analysis script agrees with `gates.json` | **pass** | 20/20, max abs value difference 0 (`RA/agreement.csv`) |
| R7 tests and C7 | **tool: false; accepted as met (D-R7)** | pytest `1 failed, 391 passed, 2 xfailed, 2 warnings in 757.28s (0:12:37)`, only the known registry failure (`pytest_known_failure_only` True); the tool's C7 `c7_tool` False; against the canonical reference `c7_canonical_reference` **IDENTICAL** (`results/v2_T2_locked/rehearsal_v2_0_checks/c7_canonical_reference.txt`); `owner_decision`: D-R7 accepted (owner, 2026-10-04): R7 is met on the canonical-reference comparison; the literal failure is a reference-path defect of the checks tool, not of the pipeline (precedent: Check 1 of the v1.0 lock round). The tool is not edited before the confirmation. |

R7 is discussed in §6, deviation 1. The R7 block's last eight keys were appended after the tool run (`amended` key in `LK/rehearsal_v2_0_checks.json`); every other key, including `R7.pass` and `ALL_PASS`, is the tool's own output.

**Re-rehearsal pass counts and safety condition, verbatim from `RA/tables.md`:**

| q | n_runs | n_G-S_pass | mean_signed | sd_ddof1 | per_run_pass_prob | P_ge_k_of_n | meets_probability |
|---|---|---|---|---|---|---|---|
| 50 | 10 | 10 | -0.011811 | 0.0178499 | 0.983533 | 0.995873 | yes |
| 60 | 10 | 10 | 0.000906032 | 0.0187419 | 0.992294 | 0.999527 | yes |

| q | n_expected | n_completed | n_missing | n_failed_exception | n_incomplete | n_global_rng_violation | n_pass | pass_rate | cp95_lo | cp95_hi | rule | q_passes_rule | n_G-A_pass | n_G-F_pass | n_G-N_pass | n_G-S_pass | n_S1_0.10_pass | n_v1_1_run_pass | n_v1_0_run_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10 | 10 | 0 | 0 | 0 | 0 | 10 | 1 | 0.691503 | 1 | not applicable (n = 10 != 20) |  | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| 60 | 10 | 10 | 0 | 0 | 0 | 0 | 10 | 1 | 0.691503 | 1 | not applicable (n = 10 != 20) |  | 10 | 10 | 10 | 10 | 10 | 10 | 10 |

**Per-run table (re-rehearsal), verbatim from `RA/tables.md`.** Final-tier values; `*_dev` and `*_dev_minus_final` are the development tier and its difference from the final tier. `G-A_pass`, `G-F_pass`, `G-N_pass`, `G-S_pass` are the v2.0 gates; `S1_pass` is S1 at 0.10; `v1_1_run_pass` and `v1_0_*` are the v1.1 and v1.0 outcomes; `table_sha256` and `table_build_sec` are the continuation table's.

| q | seed | status | exit_code | status_info | crashed_attempts | eta_final | eta_dev | eta_dev_minus_final | rmse | rmse_dev | rmse_dev_minus_final | tail | tail_dev | tail_dev_minus_final | gmax_final | gmax_dev | gmax_dev_minus_final | s1 | s1_dev | s1_dev_minus_final | stage1_rel_err_signed | e1_at_0 | G-A_pass | G-A_eta_pass | G-A_rmse_pass | G-A_tail_pass | G-F_pass | G-N_pass | G-N_eta_pass | G-N_gmax_pass | G-S_pass | S1_pass | global_rng | run_pass | v1_1_run_pass | v1_0_G-F | v1_0_run_pass | wall_sec | outcome | table_sha256 | table_build_sec | table_sha256_disk | clean_tree | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | completed | 0 |  |  | 0.00401444 | 0.00401444 | 0 | 0.0396621 | 0.0396621 | 0 | 0.00747055 | 0.00747055 | 0 | 0.00401444 | 0.00401444 | 0 | 0.0134169 | 0.0134169 | 0 | 0.0134169 | 47.2928 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 331.253 | pass | c72e4156e3ce08f25a071f61f285e55500fc4ee8b8258eb71f34462bbec40243 | 7.79056 | c72e4156e3ce08f25a071f61f285e55500fc4ee8b8258eb71f34462bbec40243 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10502 | completed | 0 |  |  | 0.00117057 | 0.000984432 | -0.000186138 | 0.0197389 | 0.0197389 | 0 | 0.01187 | 0.01187 | 0 | 0.00117057 | 0.000984432 | -0.000186138 | 0.0102045 | 0.0102045 | 0 | -0.0102045 | 46.1905 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 329.866 | pass | 9cfc1a75bceacf13bd1de865a8f93c613a65c96ae948ca85cc735f0c27d0cfd8 | 7.83428 | 9cfc1a75bceacf13bd1de865a8f93c613a65c96ae948ca85cc735f0c27d0cfd8 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10503 | completed | 0 |  |  | 0.00172113 | 0.00163935 | -8.178e-05 | 0.019966 | 0.019966 | 0 | 0.00691612 | 0.00691612 | 0 | 0.00172113 | 0.00163935 | -8.178e-05 | 0.0322856 | 0.0322856 | 0 | -0.0322856 | 45.16 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 332.358 | pass | fac688aca008fe72bcad328958ffa0e81b553d955669a06a3dbed718398d0076 | 7.76992 | fac688aca008fe72bcad328958ffa0e81b553d955669a06a3dbed718398d0076 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10504 | completed | 0 |  |  | 0.00132726 | 0.00115929 | -0.000167971 | 0.0190868 | 0.0190868 | 0 | 0.00860028 | 0.00860028 | 0 | 0.00132726 | 0.00115929 | -0.000167971 | 0.00920279 | 0.00920279 | 0 | 0.00920279 | 47.0961 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 331.474 | pass | 7caccc1db7054996995378326c70d23e8a5b9615fbac915d3944900a45a8a78d | 7.68434 | 7caccc1db7054996995378326c70d23e8a5b9615fbac915d3944900a45a8a78d | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10505 | completed | 0 |  |  | 0.000960484 | 0.000919478 | -4.10057e-05 | 0.0201792 | 0.0201792 | 0 | 0.00831625 | 0.00831625 | 0 | 0.000960484 | 0.000919478 | -4.10057e-05 | 0.0186761 | 0.0186761 | 0 | -0.0186761 | 45.7951 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 328.472 | pass | b60f95f614d0e7f4f279da3babc22e6bbe856f8fb43530b6c0b443fa729d84ea | 7.97087 | b60f95f614d0e7f4f279da3babc22e6bbe856f8fb43530b6c0b443fa729d84ea | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10506 | completed | 0 |  |  | 0.002643 | 0.002643 | -1.11022e-16 | 0.0239887 | 0.0239887 | 0 | 0.00774763 | 0.00774763 | 0 | 0.002643 | 0.002643 | -1.11022e-16 | 0.040692 | 0.040692 | 0 | -0.040692 | 44.7677 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 330.548 | pass | e0742ba707095b809704533c9fab1c32d3727f2a203d9a05ae84eb5db37fbabc | 7.96374 | e0742ba707095b809704533c9fab1c32d3727f2a203d9a05ae84eb5db37fbabc | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10507 | completed | 0 |  |  | 0.00106326 | 0.00102649 | -3.67692e-05 | 0.0265976 | 0.0265976 | 0 | 0.00986186 | 0.00986186 | 0 | 0.00106326 | 0.00102649 | -3.67692e-05 | 0.0246492 | 0.0246492 | 0 | -0.0246492 | 45.5164 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 328.511 | pass | a8422ec85f950f255c5b9963ffdbacc08cf9b1191381756fabf46bcb90d6c44a | 7.97388 | a8422ec85f950f255c5b9963ffdbacc08cf9b1191381756fabf46bcb90d6c44a | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10508 | completed | 0 |  |  | 0.00114095 | 0.00100209 | -0.000138854 | 0.0234103 | 0.0234103 | 0 | 0.00495108 | 0.00495108 | 0 | 0.00114095 | 0.00100209 | -0.000138854 | 0.00963182 | 0.00963182 | 0 | -0.00963182 | 46.2172 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 328.058 | pass | 8e06e912a5dccdc419d72be2f09e48691c4b67d3c819f7d7f90e72bdd0519308 | 7.85391 | 8e06e912a5dccdc419d72be2f09e48691c4b67d3c819f7d7f90e72bdd0519308 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10509 | completed | 0 |  |  | 0.00122168 | 0.00113828 | -8.33983e-05 | 0.0174234 | 0.0174234 | 0 | 0.00638139 | 0.00638139 | 0 | 0.00122168 | 0.00113828 | -8.33983e-05 | 0.0105367 | 0.0105367 | 0 | -0.0105367 | 46.175 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 328.959 | pass | 0504a66d02387aeae7c27c07e6e85843e66e4a3b4e5940b42782d6d139f948e2 | 7.81602 | 0504a66d02387aeae7c27c07e6e85843e66e4a3b4e5940b42782d6d139f948e2 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 50 | 10510 | completed | 0 |  |  | 0.000946099 | 0.000725629 | -0.00022047 | 0.0150213 | 0.0150213 | 0 | 0.00917122 | 0.00917122 | 0 | 0.000946099 | 0.000725629 | -0.00022047 | 0.00594587 | 0.00594587 | 0 | 0.00594587 | 46.9441 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 332.287 | pass | ce17763bdc6023fe25c0344c3bf4ea6851f84949d8dcb3d6ad48c806b22a0a82 | 7.8393 | ce17763bdc6023fe25c0344c3bf4ea6851f84949d8dcb3d6ad48c806b22a0a82 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10501 | completed | 0 |  |  | 0.00041212 | 0.000360161 | -5.19596e-05 | 0.0133277 | 0.0133277 | 0 | 0.00832454 | 0.00832454 | 0 | 0.00041212 | 0.000360161 | -5.19596e-05 | 0.0314414 | 0.0314414 | 0 | -0.0314414 | 37.6662 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 324.356 | pass | 26b50d8b855489f38ab5ffdac4e4c263902743c90dbd0f334543e85b32847614 | 9.24697 | 26b50d8b855489f38ab5ffdac4e4c263902743c90dbd0f334543e85b32847614 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10502 | completed | 0 |  |  | 0.000590891 | 0.000590891 | -1.11022e-16 | 0.0182003 | 0.0182003 | 0 | 0.0107927 | 0.0107927 | 0 | 0.000590891 | 0.000590891 | -1.11022e-16 | 0.0280365 | 0.0280365 | 0 | 0.0280365 | 39.9792 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 325.906 | pass | 3a7cd63058871f2666a2efd00611cd198bb7a6468dd75f8689c3caae42af8169 | 9.28325 | 3a7cd63058871f2666a2efd00611cd198bb7a6468dd75f8689c3caae42af8169 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10503 | completed | 0 |  |  | 0.00102148 | 0.000851703 | -0.000169772 | 0.0239348 | 0.0239348 | 0 | 0.00922653 | 0.00922653 | 0 | 0.00102148 | 0.000851703 | -0.000169772 | 0.0219821 | 0.0219821 | 0 | 0.0219821 | 39.7437 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 320.779 | pass | 0b75162d3e77e9a068c7c5602c2877bb764a41f3b61c24f6ce231916af7ed10a | 9.43834 | 0b75162d3e77e9a068c7c5602c2877bb764a41f3b61c24f6ce231916af7ed10a | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10504 | completed | 0 |  |  | 0.000929381 | 0.000917235 | -1.21464e-05 | 0.0226681 | 0.0226681 | 0 | 0.0102783 | 0.0102783 | 0 | 0.000929381 | 0.000917235 | -1.21464e-05 | 0.00486167 | 0.00486167 | 0 | -0.00486167 | 38.6998 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 334.988 | pass | 6e955f7af5b1ce45f7e6f6f325959eecfce317b892c981d40e7218e8e133af54 | 8.77537 | 6e955f7af5b1ce45f7e6f6f325959eecfce317b892c981d40e7218e8e133af54 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10505 | completed | 0 |  |  | 0.00118019 | 0.00098001 | -0.000200184 | 0.0162783 | 0.0162783 | 0 | 0.00879114 | 0.00879114 | 0 | 0.00118019 | 0.00098001 | -0.000200184 | 0.0243908 | 0.0243908 | 0 | -0.0243908 | 37.9404 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 321.681 | pass | 99e824e56240103d6726eb7fa3270b53db0b32db5d466841daa3ed63426c76c1 | 9.40059 | 99e824e56240103d6726eb7fa3270b53db0b32db5d466841daa3ed63426c76c1 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10506 | completed | 0 |  |  | 0.000401645 | 0.000382915 | -1.87303e-05 | 0.0172828 | 0.0172828 | 0 | 0.0125433 | 0.0125433 | 0 | 0.000401645 | 0.000382915 | -1.87303e-05 | 0.0130148 | 0.0130148 | 0 | 0.0130148 | 39.395 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 323.724 | pass | d15f4eacb7b965eff473f5d0a3cd38dc2210dc56d072d32ec3efc40bb617655d | 9.30185 | d15f4eacb7b965eff473f5d0a3cd38dc2210dc56d072d32ec3efc40bb617655d | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10507 | completed | 0 |  |  | 0.000514279 | 0.000433965 | -8.03144e-05 | 0.0172917 | 0.0172917 | 0 | 0.00955929 | 0.00955929 | 0 | 0.000514279 | 0.000433965 | -8.03144e-05 | 0.00726242 | 0.00726242 | 0 | 0.00726242 | 39.1713 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 320.84 | pass | 73a63fd60a86a063724045653b486114fd7ae5f55ae6caa04e2bab8ea0552438 | 9.35181 | 73a63fd60a86a063724045653b486114fd7ae5f55ae6caa04e2bab8ea0552438 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10508 | completed | 0 |  |  | 0.00159729 | 0.00149728 | -0.000100009 | 0.0245414 | 0.0245414 | 0 | 0.0104563 | 0.0104563 | 0 | 0.00159729 | 0.00149728 | -0.000100009 | 0.00245915 | 0.00245915 | 0 | -0.00245915 | 38.7933 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 346.943 | pass | 1ba536c273dcbef980d2c4d3ffc79e98c31a2d473c95935fc214deb0e1f9e16b | 9.02847 | 1ba536c273dcbef980d2c4d3ffc79e98c31a2d473c95935fc214deb0e1f9e16b | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10509 | completed | 0 |  |  | 0.000764476 | 0.000764476 | 0 | 0.021446 | 0.021446 | 0 | 0.00868628 | 0.00868628 | 0 | 0.000764476 | 0.000764476 | 0 | 0.00657856 | 0.00657856 | 0 | 0.00657856 | 39.1447 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 320.783 | pass | fccc71ffe9f6afd174dca951826f6641e73b13f607cd3b1e6c022217205684c8 | 9.38379 | fccc71ffe9f6afd174dca951826f6641e73b13f607cd3b1e6c022217205684c8 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |
| 60 | 10510 | completed | 0 |  |  | 0.00145321 | 0.00145321 | -1.11022e-16 | 0.0265131 | 0.0265131 | 0 | 0.00863194 | 0.00863194 | 0 | 0.00145321 | 0.00145321 | -1.11022e-16 | 0.00466098 | 0.00466098 | 0 | -0.00466098 | 38.7076 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 345.389 | pass | ca645fd67a97de2e6a8241629ead3d1f7df989ebdb9c51fa539d50580e6fd200 | 9.25676 | ca645fd67a97de2e6a8241629ead3d1f7df989ebdb9c51fa539d50580e6fd200 | yes | f2d616cc921e2c6be46c48336d31372dbbf58481 |

---

## 4. Confirmation (fresh seeds 30501–30520)

**Design.**
- 40 runs (q ∈ {50, 60} × seeds 30501–30520), through the v2.0 entry point from scratch, into `LK/confirmation_v2_0/`, same settings as §3.
- Launch commit `d2e377d` (`d2e377d0da702e8e162662f74576d1bff5b42a56`). All 40 manifests show the v2.0 version, SHA-256 `6ca0a6f95ee8109aeca7c787186fb40f5fec80a09c113b880fe80d20f1d47fd3`, commit `d2e377d` and `clean_tree: true` (`LK/confirmation_v2_0_checks.txt`; `n_clean_tree` 40 in `CA/verdict.json`).

**Before launch** (`LK/confirmation_v2_0/launch_env.txt`):
- `nproc` 64; `04:24:47 up 5 days, 12:05,  8 users,  load average: 6.93, 6.99, 6.45`; `/dev/md0 5.5T 3.1T 2.1T 61% /home`; memory 754 GB total, 728 GB available;
- the load average before launch (6.93 / 6.99 / 6.45) was higher than at the rehearsal launch (2.31, 1.98, 1.84); its source was not identified, and no job of anyone else was started or stopped by this work.

20 parallel single-threaded processes (`xargs -P 20 -L 1 bash LK/launch_one_v2_0.sh <root>`) ran in tmux session `confirm_v2_0` (the command line is in `LK/confirmation_v2_0/launch_command.txt`, recorded after the launch): the first 20 jobs are q=50, the next 20 q=60 (`LK/confirmation_v2_0/jobs.txt`). **No crash rule was needed:** 40/40 rc = 0, and `LK/confirmation_v2_0/crashed/` does not exist. Wall time per run: min 346 s, median 356 s, max 373 s (the re-rehearsal: min 321 s, median 329 s, max 347 s; `RA/per_run.csv`); table build: min 7.73 s, median 8.96 s, max 10.04 s, q=50 7.73–8.60 s and q=60 9.32–10.04 s (`CA/per_run.csv`). First start 2026-10-04T04:24:57Z, last end 2026-10-04T04:37:16Z.

**After launch nothing changed** until all runs had finished:
- no code, protocol, threshold or analysis-script edit; no extra seeds; no re-run. The tree held no change outside `results/`: the launch record shows none at launch (`LK/confirmation_v2_0/launch_env.txt`), every manifest, read at each run's start, shows `clean_tree: true`, and `git log` shows no commit touching anything outside `results/` between `d2e377d` and the end of the last run (`85c294e` is results-only; the checks-tool fix `3fedaa2` came after). `git status --porcelain -- . ':(exclude)results'` was also empty when read by hand after the last run; that output is not recorded in a file;
- the pre-registered analysis ran unchanged (script SHA-256 `9cdbaf7c1fe85787ae99a8abdd3bab5987bd3b7e3742c6a1564ac80bf0c5fa90` = the LOCK record).

### 4.1 Verdict

| root | seeds | rule | per_q | overall | n_agreement_rows | n_all_agree | n_expected | n_completed | n_missing | n_failed_exception | n_incomplete | n_global_rng_violation | n_clean_tree | commits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| results/v2_T2_locked/confirmation_v2_0 | [30501, 30520] | each q >= 18 of 20 runs pass; both q | {"50": true, "60": true} | PASS | 40 | 40 | 40 | 40 | 0 | 0 | 0 | 0 | 40 | ['d2e377d0da702e8e162662f74576d1bff5b42a56'] |

| q | n_expected | n_completed | n_missing | n_failed_exception | n_incomplete | n_global_rng_violation | n_pass | pass_rate | cp95_lo | cp95_hi | rule | q_passes_rule | n_G-A_pass | n_G-F_pass | n_G-N_pass | n_G-S_pass | n_S1_0.10_pass | n_v1_1_run_pass | n_v1_0_run_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 20 | 20 | 0 | 0 | 0 | 0 | 19 | 0.95 | 0.751267 | 0.998735 | >= 18 of 20 | yes | 19 | 20 | 20 | 20 | 20 | 19 | 19 |
| 60 | 20 | 20 | 0 | 0 | 0 | 0 | 20 | 1 | 0.831567 | 1 | >= 18 of 20 | yes | 20 | 20 | 20 | 20 | 20 | 20 | 20 |

### 4.2 Per-run table, all 40 runs

Verbatim from `CA/tables.md`. `eta_final`, `eta_dev`, `rmse`, `tail` are G-A (RMSE and tail are tier-independent; `rmse_dev`, `tail_dev` are their development-tier columns); `gmax_final`, `gmax_dev` are G-F; the `*_dev_minus_final` columns are G-N; `s1` is |ê₁(0) − e₁*(0)|/e₁*(0), G-S at 0.05 and S1 at 0.10; `exit_code` and `global_rng` give the run status; `wall_sec` is the wall time; `table_sha256` and `table_build_sec` are the continuation table's; `clean_tree` and `commit` come from the manifest.

| q | seed | status | exit_code | status_info | crashed_attempts | eta_final | eta_dev | eta_dev_minus_final | rmse | rmse_dev | rmse_dev_minus_final | tail | tail_dev | tail_dev_minus_final | gmax_final | gmax_dev | gmax_dev_minus_final | s1 | s1_dev | s1_dev_minus_final | stage1_rel_err_signed | e1_at_0 | G-A_pass | G-A_eta_pass | G-A_rmse_pass | G-A_tail_pass | G-F_pass | G-N_pass | G-N_eta_pass | G-N_gmax_pass | G-S_pass | S1_pass | global_rng | run_pass | v1_1_run_pass | v1_0_G-F | v1_0_run_pass | wall_sec | outcome | table_sha256 | table_build_sec | table_sha256_disk | clean_tree | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 30501 | completed | 0 |  |  | 0.000974035 | 0.000970333 | -3.70239e-06 | 0.0198581 | 0.0198581 | 0 | 0.00971214 | 0.00971214 | 0 | 0.000974035 | 0.000970333 | -3.70239e-06 | 0.000643774 | 0.000643774 | 0 | 0.000643774 | 46.6967 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 355.236 | pass | ef532a7546fbb32624f0544f1b82da0bdd12310c9975fcffd817c5c41871ad24 | 8.48423 | ef532a7546fbb32624f0544f1b82da0bdd12310c9975fcffd817c5c41871ad24 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30502 | completed | 0 |  |  | 0.00160659 | 0.00147389 | -0.000132695 | 0.0226588 | 0.0226588 | 0 | 0.00869106 | 0.00869106 | 0 | 0.00160659 | 0.00147389 | -0.000132695 | 0.0205636 | 0.0205636 | 0 | -0.0205636 | 45.707 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 355.418 | pass | c52654dff37f04aa5a9c7c6cf2d4494107a0ff79abadb508228e8aa6f8334ae5 | 8.40479 | c52654dff37f04aa5a9c7c6cf2d4494107a0ff79abadb508228e8aa6f8334ae5 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30503 | completed | 0 |  |  | 0.00181288 | 0.00181288 | 0 | 0.0217572 | 0.0217572 | 0 | 0.0122467 | 0.0122467 | 0 | 0.00181288 | 0.00181288 | 0 | 0.00529941 | 0.00529941 | 0 | -0.00529941 | 46.4194 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 346.341 | pass | 0d68f7ec30f13b46a8af2773a352dd9b4fe1a1b4c4cdb75546d1b1da1f4ff7e6 | 7.96609 | 0d68f7ec30f13b46a8af2773a352dd9b4fe1a1b4c4cdb75546d1b1da1f4ff7e6 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30504 | completed | 0 |  |  | 0.000898606 | 0.000777959 | -0.000120647 | 0.0192646 | 0.0192646 | 0 | 0.00785592 | 0.00785592 | 0 | 0.000898606 | 0.000777959 | -0.000120647 | 0.0463558 | 0.0463558 | 0 | 0.0463558 | 48.8299 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 361.895 | pass | fb6d98351032ebb1cace5b9c0868edfab1f646b7691b407d36400ea568e967d3 | 8.31149 | fb6d98351032ebb1cace5b9c0868edfab1f646b7691b407d36400ea568e967d3 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30505 | completed | 0 |  |  | 0.0022939 | 0.00228301 | -1.08838e-05 | 0.0359387 | 0.0359387 | 0 | 0.00792492 | 0.00792492 | 0 | 0.0022939 | 0.00228301 | -1.08838e-05 | 0.000910725 | 0.000910725 | 0 | 0.000910725 | 46.7092 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 355.046 | pass | 2d8e25a6952befeed679819735ac44254b5ce5cd6d807c50fab4d77c24f97bfb | 7.72644 | 2d8e25a6952befeed679819735ac44254b5ce5cd6d807c50fab4d77c24f97bfb | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30506 | completed | 0 |  |  | 0.00132588 | 0.0013024 | -2.34818e-05 | 0.0236995 | 0.0236995 | 0 | 0.00707731 | 0.00707731 | 0 | 0.00132588 | 0.0013024 | -2.34818e-05 | 0.00151485 | 0.00151485 | 0 | 0.00151485 | 46.7374 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 361.98 | pass | 6fa554079e542fb365df3e5244f0040a7bfd8ccb8f53a2e87b2a6e75e3125a9a | 8.59866 | 6fa554079e542fb365df3e5244f0040a7bfd8ccb8f53a2e87b2a6e75e3125a9a | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30507 | completed | 0 |  |  | 0.00114376 | 0.00108331 | -6.04571e-05 | 0.0240118 | 0.0240118 | 0 | 0.00799612 | 0.00799612 | 0 | 0.00114376 | 0.00108331 | -6.04571e-05 | 0.0341006 | 0.0341006 | 0 | -0.0341006 | 45.0753 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 348.6 | pass | 3cb9a9883fffe8a2dd618d1f04c25c4557df61fe7fe5ad9a06886f4d236e9118 | 8.00748 | 3cb9a9883fffe8a2dd618d1f04c25c4557df61fe7fe5ad9a06886f4d236e9118 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30508 | completed | 0 |  |  | 0.00153562 | 0.00145277 | -8.28529e-05 | 0.0226018 | 0.0226018 | 0 | 0.00750555 | 0.00750555 | 0 | 0.00153562 | 0.00145277 | -8.28529e-05 | 0.0156833 | 0.0156833 | 0 | 0.0156833 | 47.3986 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 350.03 | pass | 9d6123032557a1e9a9c18ff20170672ed20ce88d22a7eb592ad5e1626dd422a0 | 8.00261 | 9d6123032557a1e9a9c18ff20170672ed20ce88d22a7eb592ad5e1626dd422a0 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30509 | completed | 0 |  |  | 0.00115358 | 0.00113955 | -1.40255e-05 | 0.0247397 | 0.0247397 | 0 | 0.00916147 | 0.00916147 | 0 | 0.00115358 | 0.00113955 | -1.40255e-05 | 0.0249523 | 0.0249523 | 0 | -0.0249523 | 45.5022 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 353.18 | pass | 27bb8aa7799101d5c43243783040df2a26181d00f5cfa0e7bf7095cd93c7d784 | 8.2639 | 27bb8aa7799101d5c43243783040df2a26181d00f5cfa0e7bf7095cd93c7d784 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30510 | completed | 0 |  |  | 0.00580334 | 0.00580334 | 0 | 0.0451017 | 0.0451017 | 0 | 0.00882212 | 0.00882212 | 0 | 0.00580334 | 0.00580334 | 0 | 0.00828428 | 0.00828428 | 0 | 0.00828428 | 47.0533 | no | no | yes | yes | yes | yes | yes | yes | yes | yes | ok | no | no | yes | no | 373.182 | stage2_failure | 0c526eaa1fc7b2ed45de092dfe7ef62c93f1d767753ef5e2913f5421db370bb6 | 8.0333 | 0c526eaa1fc7b2ed45de092dfe7ef62c93f1d767753ef5e2913f5421db370bb6 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30511 | completed | 0 |  |  | 0.000730676 | 0.000562397 | -0.000168279 | 0.0133075 | 0.0133075 | 0 | 0.00651095 | 0.00651095 | 0 | 0.000730676 | 0.000562397 | -0.000168279 | 0.0127825 | 0.0127825 | 0 | -0.0127825 | 46.0702 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 355.586 | pass | 1d2f9d7b42e0ad0053611b018fe3d737d7bcdb64ca7d37058a2894edc60b4c8e | 8.43172 | 1d2f9d7b42e0ad0053611b018fe3d737d7bcdb64ca7d37058a2894edc60b4c8e | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30512 | completed | 0 |  |  | 0.00145875 | 0.00137759 | -8.11612e-05 | 0.0230111 | 0.0230111 | 0 | 0.00728 | 0.00728 | 0 | 0.00145875 | 0.00137759 | -8.11612e-05 | 0.00480452 | 0.00480452 | 0 | -0.00480452 | 46.4425 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 348.247 | pass | 41228dd828a31e81c39780932a3f709cbff4c8c8f284617eb22998265ad6ce83 | 8.02446 | 41228dd828a31e81c39780932a3f709cbff4c8c8f284617eb22998265ad6ce83 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30513 | completed | 0 |  |  | 0.00193495 | 0.00193495 | 0 | 0.0286553 | 0.0286553 | 0 | 0.00988788 | 0.00988788 | 0 | 0.00193495 | 0.00193495 | 0 | 0.017725 | 0.017725 | 0 | -0.017725 | 45.8395 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 347.828 | pass | 5bdc846bf4af760abdaa92e303242639af528cf972f84318f3283e64b25ff5b2 | 8.04469 | 5bdc846bf4af760abdaa92e303242639af528cf972f84318f3283e64b25ff5b2 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30514 | completed | 0 |  |  | 0.00309877 | 0.00309877 | 0 | 0.028967 | 0.028967 | 0 | 0.00761254 | 0.00761254 | 0 | 0.00309877 | 0.00309877 | 0 | 0.00502922 | 0.00502922 | 0 | -0.00502922 | 46.432 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 345.928 | pass | a28df0fcab256d3ba9a0c170204006d090e225061bca36c98ad115042c9b52ae | 8.02373 | a28df0fcab256d3ba9a0c170204006d090e225061bca36c98ad115042c9b52ae | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30515 | completed | 0 |  |  | 0.00234454 | 0.00225352 | -9.10148e-05 | 0.0211348 | 0.0211348 | 0 | 0.00907354 | 0.00907354 | 0 | 0.00234454 | 0.00225352 | -9.10148e-05 | 0.00775767 | 0.00775767 | 0 | -0.00775767 | 46.3046 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 357.637 | pass | 62edcae7f06ecbf7edea83195981e33f114a0c5fbd7ddeaf40b19b8f3317ecf2 | 8.21616 | 62edcae7f06ecbf7edea83195981e33f114a0c5fbd7ddeaf40b19b8f3317ecf2 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30516 | completed | 0 |  |  | 0.00135845 | 0.00114892 | -0.000209529 | 0.0277918 | 0.0277918 | 0 | 0.00871455 | 0.00871455 | 0 | 0.00135845 | 0.00114892 | -0.000209529 | 0.0195428 | 0.0195428 | 0 | -0.0195428 | 45.7547 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 356.285 | pass | 6cd8186afe5715b79bfde61389099b7a712ff7c8c5aa03c5f29e8913e0854f82 | 8.31886 | 6cd8186afe5715b79bfde61389099b7a712ff7c8c5aa03c5f29e8913e0854f82 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30517 | completed | 0 |  |  | 0.00135616 | 0.00126994 | -8.62272e-05 | 0.0229897 | 0.0229897 | 0 | 0.00685848 | 0.00685848 | 0 | 0.00135616 | 0.00126994 | -8.62272e-05 | 0.032236 | 0.032236 | 0 | 0.032236 | 48.171 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 357.035 | pass | f4ab1175012d06c220052baad8c0b4621438e0c5977f1f2abdad9a0e30089715 | 8.37501 | f4ab1175012d06c220052baad8c0b4621438e0c5977f1f2abdad9a0e30089715 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30518 | completed | 0 |  |  | 0.00297871 | 0.00297871 | 0 | 0.0335186 | 0.0335186 | 0 | 0.00676424 | 0.00676424 | 0 | 0.00297871 | 0.00297871 | 0 | 0.0129067 | 0.0129067 | 0 | 0.0129067 | 47.269 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 357.797 | pass | ff0845502a7716d52bb697472cfee932dcb04b59e806202c1cef7f953deb0fc6 | 8.37352 | ff0845502a7716d52bb697472cfee932dcb04b59e806202c1cef7f953deb0fc6 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30519 | completed | 0 |  |  | 0.00165171 | 0.00165171 | 0 | 0.029929 | 0.029929 | 0 | 0.00916547 | 0.00916547 | 0 | 0.00165171 | 0.00165171 | 0 | 0.0400337 | 0.0400337 | 0 | -0.0400337 | 44.7984 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 361.221 | pass | e5d838cf9ecadccf17ce506b9f42082e4f33171bd90200c7e99a1a8d48d92b6c | 7.74577 | e5d838cf9ecadccf17ce506b9f42082e4f33171bd90200c7e99a1a8d48d92b6c | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 50 | 30520 | completed | 0 |  |  | 0.00212602 | 0.00211915 | -6.87008e-06 | 0.0386725 | 0.0386725 | 0 | 0.00684092 | 0.00684092 | 0 | 0.00212602 | 0.00211915 | -6.87008e-06 | 0.0132744 | 0.0132744 | 0 | 0.0132744 | 47.2861 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 357.595 | pass | e4bf5c0e9999af9339dfc14ec516de7da78bace7cdb2e7e1ed257292648429cc | 8.4331 | e4bf5c0e9999af9339dfc14ec516de7da78bace7cdb2e7e1ed257292648429cc | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30501 | completed | 0 |  |  | 0.000861162 | 0.000742716 | -0.000118446 | 0.0236536 | 0.0236536 | 0 | 0.00938399 | 0.00938399 | 0 | 0.000861162 | 0.000742716 | -0.000118446 | 0.00160547 | 0.00160547 | 0 | -0.00160547 | 38.8265 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 356.823 | pass | ad8cf93f016ae20defd5139547ca9bef7070a34ae5ba24f81e3ff8504a619e6a | 9.70092 | ad8cf93f016ae20defd5139547ca9bef7070a34ae5ba24f81e3ff8504a619e6a | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30502 | completed | 0 |  |  | 0.00142891 | 0.00131203 | -0.000116881 | 0.0257316 | 0.0257316 | 0 | 0.00983573 | 0.00983573 | 0 | 0.00142891 | 0.00131203 | -0.000116881 | 0.0142033 | 0.0142033 | 0 | -0.0142033 | 38.3365 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 359.659 | pass | c4589b9855c8f3f2c928d60bd3db7425a788a7e954c8c115d925f768b97c936c | 9.57285 | c4589b9855c8f3f2c928d60bd3db7425a788a7e954c8c115d925f768b97c936c | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30503 | completed | 0 |  |  | 0.000403486 | 0.000354395 | -4.90903e-05 | 0.0177315 | 0.0177315 | 0 | 0.0122179 | 0.0122179 | 0 | 0.000403486 | 0.000354395 | -4.90903e-05 | 0.0211874 | 0.0211874 | 0 | 0.0211874 | 39.7128 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 349.615 | pass | 09235d7a69a22b0be0b01a21149887308d57449efbf96957565d9cd57e19762e | 9.62629 | 09235d7a69a22b0be0b01a21149887308d57449efbf96957565d9cd57e19762e | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30504 | completed | 0 |  |  | 0.00142436 | 0.00125432 | -0.000170044 | 0.0199138 | 0.0199138 | 0 | 0.0075688 | 0.0075688 | 0 | 0.00142436 | 0.00125432 | -0.000170044 | 0.028034 | 0.028034 | 0 | 0.028034 | 39.9791 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 358.53 | pass | 6fefb4ea6df8573668761da754d01a550f65a79c9f689b9cbc2f990527a007ef | 9.59021 | 6fefb4ea6df8573668761da754d01a550f65a79c9f689b9cbc2f990527a007ef | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30505 | completed | 0 |  |  | 0.0009744 | 0.000727564 | -0.000246837 | 0.0352438 | 0.0352438 | 0 | 0.0112427 | 0.0112427 | 0 | 0.0009744 | 0.000727564 | -0.000246837 | 0.0219569 | 0.0219569 | 0 | 0.0219569 | 39.7428 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 356.286 | pass | 575dbbedd314533fb7640d572dee38b08fcec9961cb75ef0bc0e57b71976b668 | 9.41698 | 575dbbedd314533fb7640d572dee38b08fcec9961cb75ef0bc0e57b71976b668 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30506 | completed | 0 |  |  | 0.000469731 | 0.000407044 | -6.26867e-05 | 0.0152854 | 0.0152854 | 0 | 0.012832 | 0.012832 | 0 | 0.000469731 | 0.000407044 | -6.26867e-05 | 0.0127892 | 0.0127892 | 0 | -0.0127892 | 38.3915 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 353.869 | pass | 77d998ed2617ac8924e946838083c95e84a5f0d2836109d32e26708454c067a3 | 9.43151 | 77d998ed2617ac8924e946838083c95e84a5f0d2836109d32e26708454c067a3 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30507 | completed | 0 |  |  | 0.000596736 | 0.000474571 | -0.000122165 | 0.0199821 | 0.0199821 | 0 | 0.0103122 | 0.0103122 | 0 | 0.000596736 | 0.000474571 | -0.000122165 | 0.00463505 | 0.00463505 | 0 | 0.00463505 | 39.0691 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 364.298 | pass | 92765590819787d583e2fc1348f3e3813910d58649f91e8e27bc1719142df964 | 9.6403 | 92765590819787d583e2fc1348f3e3813910d58649f91e8e27bc1719142df964 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30508 | completed | 0 |  |  | 0.00123147 | 0.00110656 | -0.00012491 | 0.0302516 | 0.0302516 | 0 | 0.0086379 | 0.0086379 | 0 | 0.00123147 | 0.00110656 | -0.00012491 | 0.0234091 | 0.0234091 | 0 | -0.0234091 | 37.9785 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 353.309 | pass | d37a1a42d5e6e6f6979c2c183eb080c88a3b9d202e82b93d7612ecc95790bad9 | 9.82042 | d37a1a42d5e6e6f6979c2c183eb080c88a3b9d202e82b93d7612ecc95790bad9 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30509 | completed | 0 |  |  | 0.00107484 | 0.00107484 | -1.11022e-16 | 0.0240313 | 0.0240313 | 0 | 0.0105452 | 0.0105452 | 0 | 0.00107484 | 0.00107484 | -1.11022e-16 | 0.00241008 | 0.00241008 | 0 | -0.00241008 | 38.7952 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 352.36 | pass | 77d602a98e8284295b9f48548e6d841317321f1ca0e9dbe7f73187cdc8f6ed19 | 9.57702 | 77d602a98e8284295b9f48548e6d841317321f1ca0e9dbe7f73187cdc8f6ed19 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30510 | completed | 0 |  |  | 0.000891601 | 0.000771761 | -0.00011984 | 0.0236582 | 0.0236582 | 0 | 0.0109159 | 0.0109159 | 0 | 0.000891601 | 0.000771761 | -0.00011984 | 0.0323913 | 0.0323913 | 0 | -0.0323913 | 37.6292 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 356.055 | pass | 22b58bec1b4182d26585040ebfa87c31aa5767e92b09d515c5510de562907c75 | 9.81895 | 22b58bec1b4182d26585040ebfa87c31aa5767e92b09d515c5510de562907c75 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30511 | completed | 0 |  |  | 0.00118577 | 0.00117799 | -7.78044e-06 | 0.0277458 | 0.0277458 | 0 | 0.00869869 | 0.00869869 | 0 | 0.00118577 | 0.00117799 | -7.78044e-06 | 0.00164082 | 0.00164082 | 0 | 0.00164082 | 38.9527 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 355.561 | pass | 9dcbaed4d17b3c11db799156d686b666f6a051cdc5e4678f64ccd4e8279e644c | 9.94303 | 9dcbaed4d17b3c11db799156d686b666f6a051cdc5e4678f64ccd4e8279e644c | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30512 | completed | 0 |  |  | 0.000928969 | 0.000760078 | -0.000168891 | 0.0174797 | 0.0174797 | 0 | 0.0110519 | 0.0110519 | 0 | 0.000928969 | 0.000760078 | -0.000168891 | 0.00189203 | 0.00189203 | 0 | -0.00189203 | 38.8153 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 357.111 | pass | 40dc36f4ce59cb0a1b1dcf6b9df3a53f54d51c67d8c0033645aa27b30c014aed | 9.73002 | 40dc36f4ce59cb0a1b1dcf6b9df3a53f54d51c67d8c0033645aa27b30c014aed | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30513 | completed | 0 |  |  | 0.000733635 | 0.000612091 | -0.000121544 | 0.0177426 | 0.0177426 | 0 | 0.00950923 | 0.00950923 | 0 | 0.000733635 | 0.000612091 | -0.000121544 | 0.0101024 | 0.0101024 | 0 | -0.0101024 | 38.496 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 353.574 | pass | d8eec1db68435ec888331d04fc563209f5dcbf7e2a712a271ce4af2faa19ee00 | 9.89225 | d8eec1db68435ec888331d04fc563209f5dcbf7e2a712a271ce4af2faa19ee00 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30514 | completed | 0 |  |  | 0.00122082 | 0.00108335 | -0.000137477 | 0.0203459 | 0.0203459 | 0 | 0.00925947 | 0.00925947 | 0 | 0.00122082 | 0.00108335 | -0.000137477 | 0.00510017 | 0.00510017 | 0 | -0.00510017 | 38.6905 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 355.524 | pass | 052844dd049956900fc88e04927bb7ab5eae6614f9412c1cd568c6221cdb85e7 | 10.0423 | 052844dd049956900fc88e04927bb7ab5eae6614f9412c1cd568c6221cdb85e7 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30515 | completed | 0 |  |  | 0.00160519 | 0.00148972 | -0.000115479 | 0.0228834 | 0.0228834 | 0 | 0.0100368 | 0.0100368 | 0 | 0.00160519 | 0.00148972 | -0.000115479 | 0.0105766 | 0.0105766 | 0 | 0.0105766 | 39.3002 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 353.002 | pass | c675f6ff7890171a8e651c770c13ea93def5f8657472b6bfb4beeaee0ce78419 | 9.71963 | c675f6ff7890171a8e651c770c13ea93def5f8657472b6bfb4beeaee0ce78419 | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30516 | completed | 0 |  |  | 0.0011427 | 0.000963956 | -0.000178741 | 0.0229304 | 0.0229304 | 0 | 0.0111109 | 0.0111109 | 0 | 0.0011427 | 0.000963956 | -0.000178741 | 0.00601656 | 0.00601656 | 0 | 0.00601656 | 39.1229 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 355.745 | pass | 6b0fd9cfbdebad7c51e43f917f9b4fbc7dbf88a3cdfc611b72be5fb886faf01e | 9.76094 | 6b0fd9cfbdebad7c51e43f917f9b4fbc7dbf88a3cdfc611b72be5fb886faf01e | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30517 | completed | 0 |  |  | 0.000671917 | 0.000554159 | -0.000117758 | 0.0161496 | 0.0161496 | 0 | 0.0110087 | 0.0110087 | 0 | 0.000671917 | 0.000554159 | -0.000117758 | 0.0260273 | 0.0260273 | 0 | -0.0260273 | 37.8767 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 349.331 | pass | 20e610b89a8b485d38e252d47f899fcc52ddd58b9315e5f1ccbc029284f14efa | 9.86022 | 20e610b89a8b485d38e252d47f899fcc52ddd58b9315e5f1ccbc029284f14efa | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30518 | completed | 0 |  |  | 0.000804334 | 0.000645416 | -0.000158918 | 0.0165907 | 0.0165907 | 0 | 0.010662 | 0.010662 | 0 | 0.000804334 | 0.000645416 | -0.000158918 | 0.0162493 | 0.0162493 | 0 | 0.0162493 | 39.5208 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 358.971 | pass | 7e6d04a6c73e7652b0a8cb3ad4394f23e3786113d8d37750650433f09c56023d | 9.44857 | 7e6d04a6c73e7652b0a8cb3ad4394f23e3786113d8d37750650433f09c56023d | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30519 | completed | 0 |  |  | 0.00177531 | 0.00171994 | -5.53695e-05 | 0.0347667 | 0.0347667 | 0 | 0.00842805 | 0.00842805 | 0 | 0.00177531 | 0.00171994 | -5.53695e-05 | 0.0230671 | 0.0230671 | 0 | 0.0230671 | 39.7859 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 358.561 | pass | b17a9e777bac1c22bb1dd39ea3a924fe761ba6b3ae283e8cbd0d64ae17b6b59a | 9.75596 | b17a9e777bac1c22bb1dd39ea3a924fe761ba6b3ae283e8cbd0d64ae17b6b59a | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |
| 60 | 30520 | completed | 0 |  |  | 0.00319467 | 0.00319467 | 1.11022e-16 | 0.0374264 | 0.0374264 | 0 | 0.0164521 | 0.0164521 | 0 | 0.00319467 | 0.00319467 | 1.11022e-16 | 0.00843709 | 0.00843709 | 0 | 0.00843709 | 39.217 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | ok | yes | yes | yes | yes | 361.212 | pass | ce361dae46b185ff8bd92a4529e52956b016fbc654368c8fd0fc691ee8442fcc | 9.32282 | ce361dae46b185ff8bd92a4529e52956b016fbc654368c8fd0fc691ee8442fcc | yes | d2e377d0da702e8e162662f74576d1bff5b42a56 |

Agreement of the script's recomputation with each run's `gates.json` (verbatim):

| q | seed | absdiff_eta_final | absdiff_eta_dev | absdiff_rmse | absdiff_tail | absdiff_gmax_final | absdiff_gmax_dev | absdiff_s1 | absdiff_rmse_dev | absdiff_tail_dev | absdiff_s1_dev | G-A | G-F | G-N | S1 | v1_0_G-F | v1_0_run_pass | G-S | v1_1_run_pass | run_pass | outcome | protocol_version | protocol_sha256 | table_sha256 | table_rule | all_agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 30501 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30502 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30503 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30504 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30505 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30506 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30507 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30508 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30509 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30510 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30511 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30512 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30513 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30514 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30515 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30516 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30517 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30518 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30519 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 50 | 30520 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30501 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30502 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30503 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30504 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30505 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30506 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30507 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30508 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30509 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30510 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30511 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30512 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30513 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30514 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30515 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30516 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30517 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30518 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30519 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| 60 | 30520 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |

### 4.3 Stage-1 error: G-S at 0.05 and S1 at 0.10

Verbatim from `CA/tables.md` (source `CA/s1_summary.csv`; bootstrap 10,000 resamples, seed 20261002, one fresh generator per q):

| q | criterion | threshold | n | n_pass | cp95_lo | cp95_hi | mean_signed | boot95_lo | boot95_hi | median_signed | sd_signed | bootstrap_resamples | bootstrap_seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | G-S | 0.05 | 20 | 20 | 0.831567 | 1 | -0.00303907 | -0.0117933 | 0.00619827 | -0.00491687 | 0.0211154 | 10000 | 20261002 |
| 50 | S1 | 0.1 | 20 | 20 | 0.831567 | 1 | -0.00303907 | -0.0117933 | 0.00619827 | -0.00491687 | 0.0211154 | 10000 | 20261002 |
| 60 | G-S | 0.05 | 20 | 20 | 0.831567 | 1 | 0.000593533 | -0.00684287 | 0.00765795 | 1.76751e-05 | 0.0170593 | 10000 | 20261002 |
| 60 | S1 | 0.1 | 20 | 20 | 0.831567 | 1 | 0.000593533 | -0.00684287 | 0.00765795 | 1.76751e-05 | 0.0170593 | 10000 | 20261002 |

| q | criterion | passes | exact 95% CI | mean signed stage-1 error [bootstrap 95% CI] | median | SD (ddof = 1) |
|---|---|---|---|---|---|---|
| 50 | G-S (0.05) | 20 / 20 | [0.832, 1.000] | -0.0030 [-0.0118, +0.0062] | -0.0049 | 0.0211 |
| 50 | S1 (0.1) | 20 / 20 | [0.832, 1.000] | -0.0030 [-0.0118, +0.0062] | -0.0049 | 0.0211 |
| 60 | G-S (0.05) | 20 / 20 | [0.832, 1.000] | +0.0006 [-0.0068, +0.0077] | +0.0000 | 0.0171 |
| 60 | S1 (0.1) | 20 / 20 | [0.832, 1.000] | +0.0006 [-0.0068, +0.0077] | +0.0000 | 0.0171 |

The mean, median and SD are the same for G-S and S1: they are one set of signed errors, the two criteria differ only in the threshold (`CA/s1_summary.csv`).

### 4.4 Distributions of the gate metrics and dev − final differences

Verbatim from `CA/tables.md` (source `CA/distributions.csv`):

| root | q | metric | n | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| results/v2_T2_locked/confirmation_v2_0 | 50 | eta_final | 20 | 0.000730676 | 0.000966492 | 0.0012828 | 0.00157111 | 0.00216799 | 0.00299072 | 0.00580334 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | eta_dev | 20 | 0.000562397 | 0.000951095 | 0.00114658 | 0.00146333 | 0.00215274 | 0.00299072 | 0.00580334 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | eta_dev_minus_final | 20 | -0.000209529 | -0.000136254 | -8.74241e-05 | -1.87537e-05 | 0 | 0 | 0 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | rmse | 20 | 0.0133075 | 0.0197987 | 0.0223906 | 0.0238556 | 0.0292075 | 0.0362121 | 0.0451017 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | rmse_dev | 20 | 0.0133075 | 0.0197987 | 0.0223906 | 0.0238556 | 0.0292075 | 0.0362121 | 0.0451017 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | rmse_dev_minus_final | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | tail | 20 | 0.00651095 | 0.00683325 | 0.00722933 | 0.00796052 | 0.00909552 | 0.00972971 | 0.0122467 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | tail_dev | 20 | 0.00651095 | 0.00683325 | 0.00722933 | 0.00796052 | 0.00909552 | 0.00972971 | 0.0122467 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | tail_dev_minus_final | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | gmax_final | 20 | 0.000730676 | 0.000966492 | 0.0012828 | 0.00157111 | 0.00216799 | 0.00299072 | 0.00580334 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | gmax_dev | 20 | 0.000562397 | 0.000951095 | 0.00114658 | 0.00146333 | 0.00215274 | 0.00299072 | 0.00580334 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | gmax_dev_minus_final | 20 | -0.000209529 | -0.000136254 | -8.74241e-05 | -1.87537e-05 | 0 | 0 | 0 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | s1 | 20 | 0.000643774 | 0.00145444 | 0.00523186 | 0.0130905 | 0.0216608 | 0.0346939 | 0.0463558 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | s1_dev | 20 | 0.000643774 | 0.00145444 | 0.00523186 | 0.0130905 | 0.0216608 | 0.0346939 | 0.0463558 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | s1_dev_minus_final | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| results/v2_T2_locked/confirmation_v2_0 | 50 | stage1_rel_err_signed | 20 | -0.0400337 | -0.0258672 | -0.0181794 | -0.00491687 | 0.00943987 | 0.0173385 | 0.0463558 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | eta_final | 20 | 0.000403486 | 0.000584036 | 0.000786659 | 0.00102462 | 0.00127969 | 0.00162221 | 0.00319467 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | eta_dev | 20 | 0.000354395 | 0.000467818 | 0.000637085 | 0.000867859 | 0.00119707 | 0.00151274 | 0.00319467 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | eta_dev_minus_final | 20 | -0.000246837 | -0.000170914 | -0.000142837 | -0.000119143 | -6.08574e-05 | -7.0024e-06 | 1.11022e-16 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | rmse | 20 | 0.0152854 | 0.0165465 | 0.0177399 | 0.0229069 | 0.0262351 | 0.0348144 | 0.0374264 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | rmse_dev | 20 | 0.0152854 | 0.0165465 | 0.0177399 | 0.0229069 | 0.0262351 | 0.0348144 | 0.0374264 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | rmse_dev_minus_final | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | tail | 20 | 0.0075688 | 0.00861691 | 0.00935286 | 0.0104287 | 0.0110666 | 0.0122793 | 0.0164521 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | tail_dev | 20 | 0.0075688 | 0.00861691 | 0.00935286 | 0.0104287 | 0.0110666 | 0.0122793 | 0.0164521 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | tail_dev_minus_final | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | gmax_final | 20 | 0.000403486 | 0.000584036 | 0.000786659 | 0.00102462 | 0.00127969 | 0.00162221 | 0.00319467 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | gmax_dev | 20 | 0.000354395 | 0.000467818 | 0.000637085 | 0.000867859 | 0.00119707 | 0.00151274 | 0.00319467 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | gmax_dev_minus_final | 20 | -0.000246837 | -0.000170914 | -0.000142837 | -0.000119143 | -6.08574e-05 | -7.0024e-06 | 1.11022e-16 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | s1 | 20 | 0.00160547 | 0.00186691 | 0.00498389 | 0.0116829 | 0.0222344 | 0.026228 | 0.0323913 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | s1_dev | 20 | 0.00160547 | 0.00186691 | 0.00498389 | 0.0116829 | 0.0222344 | 0.026228 | 0.0323913 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | s1_dev_minus_final | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| results/v2_T2_locked/confirmation_v2_0 | 60 | stage1_rel_err_signed | 20 | -0.0323913 | -0.0236709 | -0.0107741 | 1.76751e-05 | 0.0119948 | 0.0220679 | 0.028034 |

### 4.5 Reported metrics

The reported metrics of the v1.1 report (its §4.5, which reproduces the metrics of §4.4 of the v1.0 report), final tier, including the stage-1 decomposition with bands (source `CA/reported_metrics.csv`). Verbatim:

| q | seed | A_stage2_peak_rel_err_signed | A_stage2_peak_locfree_rel_err | A_stage2_peak_locfree_argmax_d | A_stage2_sym_err_max | A_stage2_tail_max | A_DeltaT_over_dw_on_max | A_DeltaT_over_dw_off_max | A_sigma_effort_at_0_t2 | A_smoothed_share_peak_gap_d0 | B_e1_at_0 | B_stage1_rel_err_signed | B_EXP_root_over_dw | B_dReach_over_dw | B_Deltamax_all_over_dw | B_dFull_over_dw | B_Gmax_full_t | B_Gmax_full_d | B_sigma_effort_at_0_t1 | dec_e_tilde | dec_band_lo | dec_band_hi | dec_learning_rel | dec_learning_rel_lo | dec_learning_rel_hi | dec_inherited_rel | dec_inherited_rel_lo | dec_inherited_rel_hi | dec_learning_contains_0 | dec_inherited_contains_0 | dec_band_contiguous | dec_e1_inside_sweep | drift_test_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 30501 | -0.0507668 | -0.0502328 | -1.5 | 1.17152 | 2.84585 | 0.000974035 | 0.000429902 | 2.76429 | 0.614044 | 46.6967 | 0.000643774 | 0.00021163 | 0.000974269 | 0.000974035 | 0.000974269 | 2 | -66 | 2.71225 | 46.5067 | 46.5067 | 46.6767 | 0.00407235 | 0.000429488 | 0.00407235 | -0.00342857 | -0.00342857 | 0.000214286 | no | yes | no | yes | yes |
| 50 | 30502 | -0.079053 | -0.0786955 | -1.5 | 3.06583 | 2.19312 | 0.00160659 | 0.000299074 | 2.73568 | 0.390265 | 45.707 | -0.0205636 | 0.000278049 | 0.0016129 | 0.00160659 | 0.0016129 | 2 | -2 | 2.65075 | 46.0267 | 46.0267 | 46.1667 | -0.00684928 | -0.00984928 | -0.00684928 | -0.0137143 | -0.0137143 | -0.0107143 | no | no | no | yes | yes |
| 50 | 30503 | -0.0520319 | -0.0519954 | 0.5 | 0.903148 | 2.80419 | 0.00181288 | 0.000447851 | 2.77291 | 0.600985 | 46.4194 | -0.00529941 | 0.000354711 | 0.00182162 | 0.00181288 | 0.00182162 | 2 | -68 | 2.71647 | 46.5967 | 46.5967 | 46.8967 | -0.00379941 | -0.010228 | -0.00379941 | -0.0015 | -0.0015 | 0.00492857 | no | yes | no | yes | yes |
| 50 | 30504 | -0.0573839 | -0.0573839 | 0 | 1.73799 | 2.75309 | 0.000898606 | 0.000450167 | 2.8807 | 0.566135 | 48.8299 | 0.0463558 | 0.000618544 | 0.00132409 | 0.000898606 | 0.00132409 | 2 | -2 | 2.84955 | 46.9767 | 46.9767 | 46.9867 | 0.0397129 | 0.0394986 | 0.0397129 | 0.00664286 | 0.00664286 | 0.00685714 | no | no | yes | yes | yes |
| 50 | 30505 | -0.0894583 | -0.0878579 | 2 | 3.80106 | 2.43401 | 0.0022939 | 0.000420658 | 2.93288 | 0.369751 | 46.7092 | 0.000910725 | 0.000552711 | 0.00235022 | 0.0022939 | 0.00235022 | 2 | -2 | 2.86594 | 47.2267 | 47.2267 | 47.4967 | -0.0110893 | -0.016875 | -0.0110893 | 0.012 | 0.012 | 0.0177857 | no | no | no | yes | yes |
| 50 | 30506 | -0.0619695 | -0.0608187 | -2 | 1.86681 | 2.10424 | 0.00132588 | 0.000234409 | 2.88998 | 0.525932 | 46.7374 | 0.00151485 | 0.000203007 | 0.00132588 | 0.00132588 | 0.00132588 | 2 | -86 | 2.87684 | 46.6367 | 46.6367 | 46.9567 | 0.00215771 | -0.00469943 | 0.00215771 | -0.000642857 | -0.000642857 | 0.00621429 | yes | yes | no | yes | yes |
| 50 | 30507 | -0.0329269 | -0.0328609 | -0.5 | 4.4142 | 3.11579 | 0.00114376 | 0.000634644 | 2.97409 | 1.01858 | 45.0753 | -0.0341006 | 0.000381866 | 0.00130356 | 0.00114376 | 0.00130356 | 2 | -14 | 3.00225 | 46.0467 | 46.0467 | 46.3367 | -0.0208149 | -0.0270292 | -0.0208149 | -0.0132857 | -0.0132857 | -0.00707143 | no | no | no | yes | yes |
| 50 | 30508 | -0.075891 | -0.075877 | -0.5 | 0.578258 | 2.10792 | 0.00153562 | 0.000244365 | 2.75941 | 0.410052 | 47.3986 | 0.0156833 | 0.00046819 | 0.00162102 | 0.00153562 | 0.00162102 | 2 | -2 | 2.66044 | 46.5067 | 46.5067 | 46.7367 | 0.0191118 | 0.0141833 | 0.0191118 | -0.00342857 | -0.00342857 | 0.0015 | no | yes | no | yes | yes |
| 50 | 30509 | -0.048471 | -0.0476591 | -1.5 | 3.29606 | 2.7717 | 0.00115358 | 0.000520104 | 2.72216 | 0.633317 | 45.5022 | -0.0249523 | 0.000303674 | 0.00116419 | 0.00115358 | 0.00116419 | 2 | -18 | 2.69605 | 45.6867 | 45.6867 | 45.8767 | -0.00395235 | -0.00802378 | -0.00395235 | -0.021 | -0.021 | -0.0169286 | no | no | yes | yes | yes |
| 50 | 30510 | -0.145813 | -0.145544 | 1.5 | 2.37212 | 3.88102 | 0.00580334 | 0.000818975 | 3.61088 | 0.279363 | 47.0533 | 0.00828428 | 0.00115141 | 0.00580363 | 0.00580334 | 0.00580363 | 2 | -4 | 3.41816 | 47.1167 | 47.1167 | 47.1367 | -0.00135858 | -0.00178715 | -0.00135858 | 0.00964286 | 0.00964286 | 0.0100714 | no | no | yes | yes | yes |
| 50 | 30511 | -0.0522651 | -0.0522487 | 0.5 | 1.23162 | 2.56816 | 0.000730676 | 0.000454211 | 3.09365 | 0.667547 | 46.0702 | -0.0127825 | 0.000203838 | 0.00078745 | 0.000730676 | 0.00078745 | 2 | -2 | 3.10876 | 46.5967 | 46.5967 | 46.8967 | -0.0112825 | -0.017711 | -0.0112825 | -0.0015 | -0.0015 | 0.00492857 | no | yes | no | yes | yes |
| 50 | 30512 | -0.0760429 | -0.0751203 | -1.5 | 1.5041 | 2.44144 | 0.00145875 | 0.000316151 | 2.90918 | 0.431457 | 46.4425 | -0.00480452 | 0.000288073 | 0.00146727 | 0.00145875 | 0.00146727 | 2 | -2 | 2.81156 | 46.5967 | 46.5967 | 46.9267 | -0.00330452 | -0.0103759 | -0.00330452 | -0.0015 | -0.0015 | 0.00557143 | no | yes | no | yes | yes |
| 50 | 30513 | -0.0611293 | -0.0589055 | -2.5 | 3.9355 | 2.03777 | 0.00193495 | 0.000239473 | 2.68002 | 0.494406 | 45.8395 | -0.017725 | 0.000381814 | 0.00194764 | 0.00193495 | 0.00194764 | 2 | -84 | 2.59887 | 46.0767 | 46.0767 | 46.3967 | -0.00508209 | -0.0119392 | -0.00508209 | -0.0126429 | -0.0126429 | -0.00578571 | no | no | no | yes | yes |
| 50 | 30514 | -0.0165318 | -0.014963 | -2 | 3.45762 | 2.80652 | 0.00309877 | 0.000556763 | 2.95741 | 2.01728 | 46.432 | -0.00502922 | 0.000476677 | 0.00310782 | 0.00309877 | 0.00310782 | 2 | -16 | 3.01081 | 46.0067 | 46.0067 | 46.2767 | 0.00911364 | 0.00332792 | 0.00911364 | -0.0141429 | -0.0141429 | -0.00835714 | no | no | no | yes | yes |
| 50 | 30515 | -0.0192522 | -0.0190073 | -1 | 1.47121 | 3.12729 | 0.00234454 | 0.000675119 | 2.70887 | 1.58662 | 46.3046 | -0.00775767 | 0.000402861 | 0.00234919 | 0.00234454 | 0.00234919 | 2 | -14 | 2.73679 | 46.5067 | 46.5067 | 46.6767 | -0.0043291 | -0.00797196 | -0.0043291 | -0.00342857 | -0.00342857 | 0.000214286 | no | yes | no | yes | yes |
| 50 | 30516 | -0.0706372 | -0.0705603 | 0.5 | 3.02789 | 2.50307 | 0.00135845 | 0.000334842 | 2.79963 | 0.446969 | 45.7547 | -0.0195428 | 0.000561385 | 0.00164699 | 0.00135845 | 0.00164699 | 2 | -2 | 2.71467 | 47.1267 | 47.1267 | 47.4367 | -0.0293999 | -0.0360428 | -0.0293999 | 0.00985714 | 0.00985714 | 0.0165 | no | no | no | yes | yes |
| 50 | 30517 | -0.0751793 | -0.0723522 | -3 | 4.41483 | 2.6627 | 0.00135616 | 0.000386441 | 3.17385 | 0.476145 | 48.171 | 0.032236 | 0.000519108 | 0.00160527 | 0.00135616 | 0.00160527 | 2 | -2 | 3.11755 | 46.6167 | 46.6167 | 46.8767 | 0.0333074 | 0.027736 | 0.0333074 | -0.00107143 | -0.00107143 | 0.0045 | no | yes | no | yes | yes |
| 50 | 30518 | -0.0797861 | -0.0797765 | 0.5 | 1.91861 | 2.09381 | 0.00297871 | 0.000286693 | 2.89895 | 0.40977 | 47.269 | 0.0129067 | 0.00059379 | 0.00298206 | 0.00297871 | 0.00298206 | 2 | -40 | 2.83467 | 46.9067 | 46.9067 | 47.1367 | 0.0077638 | 0.00283522 | 0.0077638 | 0.00514286 | 0.00514286 | 0.0100714 | no | no | no | yes | yes |
| 50 | 30519 | -0.0306122 | -0.025858 | -4 | 3.98863 | 2.90352 | 0.00165171 | 0.000576891 | 2.79434 | 1.02936 | 44.7984 | -0.0400337 | 0.0004137 | 0.00175673 | 0.00165171 | 0.00175673 | 2 | -16 | 2.80124 | 45.5867 | 45.5867 | 45.9167 | -0.0168908 | -0.0239623 | -0.0168908 | -0.0231429 | -0.0231429 | -0.0160714 | no | no | no | yes | yes |
| 50 | 30520 | -0.0857782 | -0.0836272 | 2.5 | 3.15378 | 2.44799 | 0.00212602 | 0.000427914 | 3.03865 | 0.39953 | 47.2861 | 0.0132744 | 0.000581742 | 0.00212602 | 0.00212602 | 0.00212602 | 2 | -46 | 3.0111 | 47.2667 | 47.2667 | 47.4467 | 0.000417297 | -0.00343985 | 0.000417297 | 0.0128571 | 0.0128571 | 0.0167143 | yes | no | yes | yes | yes |
| 60 | 30501 | -0.0673882 | -0.0664328 | -1.5 | 3.3718 | 1.93995 | 0.000861162 | 0.000204383 | 2.86858 | 0.400107 | 38.8265 | -0.00160547 | 0.000164558 | 0.000862142 | 0.000861162 | 0.000862142 | 2 | -2 | 2.54942 | 38.8689 | 38.8489 | 38.9489 | -0.00109118 | -0.00314833 | -0.000576897 | -0.000514286 | -0.00102857 | 0.00154286 | no | yes | yes | yes | yes |
| 60 | 30502 | -0.0856846 | -0.0853783 | -1 | 4.10864 | 2.33288 | 0.00142891 | 0.000388075 | 3.17533 | 0.348356 | 38.3365 | -0.0142033 | 0.0002493 | 0.00146094 | 0.00142891 | 0.00146094 | 2 | -2 | 2.80962 | 38.8189 | 38.7989 | 38.8889 | -0.0124033 | -0.0142033 | -0.011889 | -0.0018 | -0.00231429 | 0 | no | yes | yes | yes | yes |
| 60 | 30503 | -0.0465497 | -0.0464483 | 0.5 | 1.70983 | 2.03322 | 0.000403486 | 0.00022636 | 2.76708 | 0.558705 | 39.7128 | 0.0211874 | 0.000143647 | 0.000444601 | 0.000403486 | 0.000444601 | 2 | -2 | 2.48019 | 39.0489 | 39.0289 | 39.1289 | 0.0170732 | 0.015016 | 0.0175874 | 0.00411429 | 0.0036 | 0.00617143 | no | no | yes | yes | yes |
| 60 | 30504 | -0.0820461 | -0.0802324 | 3 | 1.75741 | 2.24303 | 0.00142436 | 0.000275154 | 3.18954 | 0.365433 | 39.9791 | 0.028034 | 0.00023656 | 0.00151128 | 0.00142436 | 0.00151128 | 2 | -2 | 2.88924 | 38.9089 | 38.8989 | 39.0089 | 0.0275197 | 0.0249483 | 0.0277769 | 0.000514286 | 0.000257143 | 0.00308571 | no | no | yes | yes | yes |
| 60 | 30505 | -0.0666973 | -0.0641975 | 2.5 | 4.99576 | 2.25466 | 0.0009744 | 0.000359799 | 2.7917 | 0.393408 | 39.7428 | 0.0219569 | 0.000296187 | 0.00098214 | 0.0009744 | 0.00098214 | 2 | -2 | 2.50253 | 39.4689 | 39.4689 | 39.5389 | 0.00704257 | 0.00524257 | 0.00704257 | 0.0149143 | 0.0149143 | 0.0167143 | no | no | yes | yes | yes |
| 60 | 30506 | -0.0498877 | -0.0498877 | 0 | 0.701544 | 2.24052 | 0.000469731 | 0.000276238 | 2.80382 | 0.528251 | 38.3915 | -0.0127892 | 0.000116936 | 0.000504161 | 0.000469731 | 0.000504161 | 2 | -2 | 2.50214 | 38.8489 | 38.8389 | 38.9289 | -0.0117606 | -0.0138178 | -0.0115035 | -0.00102857 | -0.00128571 | 0.00102857 | no | yes | yes | yes | yes |
| 60 | 30507 | -0.0538671 | -0.0520491 | 2.5 | 2.43257 | 2.29892 | 0.000596736 | 0.000313824 | 2.7674 | 0.482866 | 39.0691 | 0.00463505 | 0.000138727 | 0.000604865 | 0.000596736 | 0.000604865 | 2 | -2 | 2.4881 | 39.2089 | 39.1989 | 39.2689 | -0.00359352 | -0.00513638 | -0.00333638 | 0.00822857 | 0.00797143 | 0.00977143 | no | no | yes | yes | yes |
| 60 | 30508 | -0.08087 | -0.0793455 | -2.5 | 2.60488 | 2.19301 | 0.00123147 | 0.000284258 | 3.13774 | 0.36472 | 37.9785 | -0.0234091 | 0.000370123 | 0.0013454 | 0.00123147 | 0.0013454 | 2 | -2 | 2.78457 | 39.0289 | 39.0189 | 39.1089 | -0.0270091 | -0.0290662 | -0.026752 | 0.0036 | 0.00334286 | 0.00565714 | no | no | yes | yes | yes |
| 60 | 30509 | -0.0306894 | -0.0306894 | 0 | 1.2423 | 2.21304 | 0.00107484 | 0.000287783 | 3.0211 | 0.925299 | 38.7952 | -0.00241008 | 0.000270989 | 0.00108143 | 0.00107484 | 0.00108143 | 2 | -20 | 2.72196 | 38.9189 | 38.9089 | 38.9989 | -0.00318151 | -0.00523866 | -0.00292437 | 0.000771429 | 0.000514286 | 0.00282857 | no | no | yes | yes | yes |
| 60 | 30510 | -0.0686933 | -0.0675792 | -2.5 | 1.77486 | 3.07096 | 0.000891601 | 0.000669588 | 3.28006 | 0.448867 | 37.6292 | -0.0323913 | 0.000378583 | 0.00103781 | 0.000891601 | 0.00103781 | 2 | -2 | 2.90536 | 38.8389 | 38.8189 | 38.9189 | -0.0311056 | -0.0331628 | -0.0305913 | -0.00128571 | -0.0018 | 0.000771429 | no | yes | yes | yes | yes |
| 60 | 30511 | -0.0154128 | -0.0153924 | -0.5 | 1.13749 | 1.94399 | 0.00118577 | 0.000269643 | 2.94661 | 1.79693 | 38.9527 | 0.00164082 | 0.000334659 | 0.00118679 | 0.00118577 | 0.00118679 | 2 | -18 | 2.66775 | 38.8489 | 38.8289 | 38.9189 | 0.00266939 | 0.00086939 | 0.00318368 | -0.00102857 | -0.00154286 | 0.000771429 | no | yes | yes | yes | yes |
| 60 | 30512 | -0.0681714 | -0.0681714 | 0 | 0.652955 | 2.24092 | 0.000928969 | 0.000294874 | 2.76533 | 0.381264 | 38.8153 | -0.00189203 | 0.00010799 | 0.000928969 | 0.000928969 | 0.000928969 | 2 | -2 | 2.46429 | 38.7989 | 38.7889 | 38.8889 | 0.000422255 | -0.00189203 | 0.000679397 | -0.00231429 | -0.00257143 | 0 | yes | yes | yes | yes | yes |
| 60 | 30513 | -0.061176 | -0.0611527 | -0.5 | 1.1888 | 2.13285 | 0.000733635 | 0.000315021 | 2.8781 | 0.4422 | 38.496 | -0.0101024 | 0.000113144 | 0.000761835 | 0.000733635 | 0.000761835 | 2 | -2 | 2.56339 | 38.8989 | 38.8889 | 38.9789 | -0.0103595 | -0.0124166 | -0.0101024 | 0.000257143 | 0 | 0.00231429 | no | yes | yes | yes | yes |
| 60 | 30514 | -0.0787893 | -0.0786407 | -1 | 0.901634 | 1.7762 | 0.00122082 | 0.00019606 | 2.8936 | 0.345198 | 38.6905 | -0.00510017 | 0.000164114 | 0.00122591 | 0.00122082 | 0.00122591 | 2 | -2 | 2.59853 | 38.8089 | 38.7889 | 38.8889 | -0.00304302 | -0.00510017 | -0.00252874 | -0.00205714 | -0.00257143 | 0 | no | yes | yes | yes | yes |
| 60 | 30515 | -0.0885239 | -0.0880552 | 1.5 | 1.17718 | 2.20939 | 0.00160519 | 0.00028616 | 2.86301 | 0.303989 | 39.3002 | 0.0105766 | 0.000212953 | 0.00162205 | 0.00160519 | 0.00162205 | 2 | -2 | 2.53605 | 38.8589 | 38.8489 | 38.9489 | 0.0113481 | 0.00903379 | 0.0116052 | -0.000771429 | -0.00102857 | 0.00154286 | no | yes | yes | yes | yes |
| 60 | 30516 | -0.0735145 | -0.0727503 | 1.5 | 1.75807 | 2.22135 | 0.0011427 | 0.000352455 | 2.87766 | 0.367927 | 39.1229 | 0.00601656 | 0.000172436 | 0.00114724 | 0.0011427 | 0.00114724 | 2 | -2 | 2.56614 | 38.8989 | 38.8889 | 38.9889 | 0.00575942 | 0.00344513 | 0.00601656 | 0.000257143 | 0 | 0.00257143 | no | yes | yes | yes | yes |
| 60 | 30517 | -0.058209 | -0.0581722 | 0.5 | 0.71967 | 2.07559 | 0.000671917 | 0.000277338 | 2.88163 | 0.465311 | 37.8767 | -0.0260273 | 0.000196407 | 0.000772153 | 0.000671917 | 0.000772153 | 2 | -2 | 2.5603 | 38.7789 | 38.7689 | 38.8589 | -0.0231987 | -0.0252559 | -0.0229416 | -0.00282857 | -0.00308571 | -0.000771429 | no | no | yes | yes | yes |
| 60 | 30518 | -0.0628192 | -0.0621627 | 2 | 1.0143 | 2.12075 | 0.000804334 | 0.000271108 | 2.9007 | 0.434017 | 39.5208 | 0.0162493 | 0.000162052 | 0.000847029 | 0.000804334 | 0.000847029 | 2 | -2 | 2.59651 | 38.7989 | 38.7889 | 38.8889 | 0.0185636 | 0.0162493 | 0.0188208 | -0.00231429 | -0.00257143 | 0 | no | yes | yes | yes | yes |
| 60 | 30519 | -0.0934257 | -0.0932388 | 1 | 2.04108 | 1.95453 | 0.00177531 | 0.000216625 | 2.95316 | 0.297117 | 39.7859 | 0.0230671 | 0.00040777 | 0.00181794 | 0.00177531 | 0.00181794 | 2 | -2 | 2.67575 | 39.0789 | 39.0689 | 39.1689 | 0.0181814 | 0.0158671 | 0.0184386 | 0.00488571 | 0.00462857 | 0.0072 | no | no | yes | yes | yes |
| 60 | 30520 | -0.123656 | -0.12166 | 3.5 | 3.13216 | 4.41785 | 0.00319467 | 0.00113015 | 3.37767 | 0.256786 | 39.217 | 0.00843709 | 0.000491745 | 0.00319467 | 0.00319467 | 0.00319467 | 2 | -4 | 3.02791 | 39.2089 | 39.1889 | 39.2989 | 0.000208518 | -0.00210577 | 0.000722804 | 0.00822857 | 0.00771429 | 0.0105429 | yes | no | yes | yes | yes |

Counts of the decomposition and band flags (computed from `CA/reported_metrics.csv`; `yes` runs out of 20 per q):

| q | dec_learning_contains_0 | dec_inherited_contains_0 | dec_band_contiguous | dec_e1_inside_sweep | drift_test_pass |
|---|---|---|---|---|---|
| 50 | 2 / 20 | 8 / 20 | 4 / 20 | 20 / 20 | 20 / 20 |
| 60 | 2 / 20 | 11 / 20 | 20 / 20 | 20 / 20 | 20 / 20 |

### 4.6 Stage-1 error versus root gain (descriptive only)

Verbatim from `CA/tables.md` (sources `CA/exp_vs_s1.csv`, `CA/exp_vs_s1_fit.csv`):

| q | seed | stage1_rel_err_signed | stage1_rel_err_sq | EXP_root_over_dw |
|---|---|---|---|---|
| 50 | 30501 | 0.000643774 | 4.14445e-07 | 0.00021163 |
| 50 | 30502 | -0.0205636 | 0.00042286 | 0.000278049 |
| 50 | 30503 | -0.00529941 | 2.80837e-05 | 0.000354711 |
| 50 | 30504 | 0.0463558 | 0.00214886 | 0.000618544 |
| 50 | 30505 | 0.000910725 | 8.29419e-07 | 0.000552711 |
| 50 | 30506 | 0.00151485 | 2.29478e-06 | 0.000203007 |
| 50 | 30507 | -0.0341006 | 0.00116285 | 0.000381866 |
| 50 | 30508 | 0.0156833 | 0.000245965 | 0.00046819 |
| 50 | 30509 | -0.0249523 | 0.00062262 | 0.000303674 |
| 50 | 30510 | 0.00828428 | 6.86293e-05 | 0.00115141 |
| 50 | 30511 | -0.0127825 | 0.000163391 | 0.000203838 |
| 50 | 30512 | -0.00480452 | 2.30834e-05 | 0.000288073 |
| 50 | 30513 | -0.017725 | 0.000314174 | 0.000381814 |
| 50 | 30514 | -0.00502922 | 2.52931e-05 | 0.000476677 |
| 50 | 30515 | -0.00775767 | 6.01815e-05 | 0.000402861 |
| 50 | 30516 | -0.0195428 | 0.000381921 | 0.000561385 |
| 50 | 30517 | 0.032236 | 0.00103916 | 0.000519108 |
| 50 | 30518 | 0.0129067 | 0.000166582 | 0.00059379 |
| 50 | 30519 | -0.0400337 | 0.0016027 | 0.0004137 |
| 50 | 30520 | 0.0132744 | 0.000176211 | 0.000581742 |
| 60 | 30501 | -0.00160547 | 2.57753e-06 | 0.000164558 |
| 60 | 30502 | -0.0142033 | 0.000201733 | 0.0002493 |
| 60 | 30503 | 0.0211874 | 0.000448908 | 0.000143647 |
| 60 | 30504 | 0.028034 | 0.000785906 | 0.00023656 |
| 60 | 30505 | 0.0219569 | 0.000482104 | 0.000296187 |
| 60 | 30506 | -0.0127892 | 0.000163563 | 0.000116936 |
| 60 | 30507 | 0.00463505 | 2.14837e-05 | 0.000138727 |
| 60 | 30508 | -0.0234091 | 0.000547986 | 0.000370123 |
| 60 | 30509 | -0.00241008 | 5.8085e-06 | 0.000270989 |
| 60 | 30510 | -0.0323913 | 0.0010492 | 0.000378583 |
| 60 | 30511 | 0.00164082 | 2.69229e-06 | 0.000334659 |
| 60 | 30512 | -0.00189203 | 3.57978e-06 | 0.00010799 |
| 60 | 30513 | -0.0101024 | 0.000102058 | 0.000113144 |
| 60 | 30514 | -0.00510017 | 2.60117e-05 | 0.000164114 |
| 60 | 30515 | 0.0105766 | 0.000111865 | 0.000212953 |
| 60 | 30516 | 0.00601656 | 3.6199e-05 | 0.000172436 |
| 60 | 30517 | -0.0260273 | 0.00067742 | 0.000196407 |
| 60 | 30518 | 0.0162493 | 0.000264041 | 0.000162052 |
| 60 | 30519 | 0.0230671 | 0.000532093 | 0.00040777 |
| 60 | 30520 | 0.00843709 | 7.11845e-05 | 0.000491745 |

| q | n | intercept | slope_on_err_sq | r2 | model |
|---|---|---|---|---|---|
| 50 | 20 | 0.000432953 | 0.033238 | 0.00873923 | EXP_root/DW = a + b * (stage-1 rel. err)^2, ordinary least squares |
| 60 | 20 | 0.000201112 | 0.127637 | 0.127658 | EXP_root/DW = a + b * (stage-1 rel. err)^2, ordinary least squares |

Per q, an ordinary-least-squares fit of `EXP_root/ΔW = a + b·err²`: q=50: a = 0.000433, b = 0.0332, R² = 0.00874; q=60: a = 0.000201, b = 0.128, R² = 0.128 (`CA/exp_vs_s1_fit.csv`). This describes these 40 runs; it is not a model claim.
For comparison, the same fit on the v1.1 confirmation: q=50: a = 0.000254, b = 0.27, R² = 0.832; q=60: a = 0.000139, b = 0.144, R² = 0.978 (`LK/confirmation_analysis/exp_vs_s1_fit.csv`; the v1.1 report §4.6 shows the same fit).

---

## 5. v1.1 versus v2.0, descriptive

### 5a. The same development seeds: `rehearsal_v1_1` versus `rehearsal_v2_0`, paired by (q, seed)

Phase A is bit-identical to v1.1 by construction (R1 20/20), so the pairs differ only through Phase B and the gates. Verbatim from `RA/tables.md` (`this root` = `rehearsal_v2_0`; sources `RA/paired_summary.csv`, `RA/paired_gate_flips.csv`, `RA/paired_per_run.csv`):

| q | metric | n_pairs | mean_diff | median_diff | n_new_lower | n_new_higher | n_equal | mean_ref | mean_new | sd_ref_ddof1 | sd_new_ddof1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | eta_final | 10 | 0 | 0 | 0 | 0 | 10 | 0.00162089 | 0.00162089 | 0.000981212 | 0.000981212 |
| 50 | rmse | 10 | 0 | 0 | 0 | 0 | 10 | 0.0225074 | 0.0225074 | 0.00688924 | 0.00688924 |
| 50 | tail | 10 | 0 | 0 | 0 | 0 | 10 | 0.00812864 | 0.00812864 | 0.00193059 | 0.00193059 |
| 50 | gmax_final | 10 | -0.00014688 | 0 | 2 | 0 | 8 | 0.00176777 | 0.00162089 | 0.000979783 | 0.000981212 |
| 50 | s1 | 10 | -0.0238577 | -0.0153821 | 9 | 1 | 0 | 0.0413818 | 0.0175241 | 0.0309167 | 0.0115065 |
| 50 | stage1_rel_err_signed | 10 | 0.00172461 | -0.00184645 | 5 | 5 | 0 | -0.0135356 | -0.011811 | 0.0515267 | 0.0178499 |
| 50 | e1_at_0 | 10 | 0.080482 | -0.0861676 | 5 | 5 | 0 | 46.035 | 46.1155 | 2.40458 | 0.832993 |
| 50 | eta_dev_minus_final | 10 | 0 | 0 | 0 | 0 | 10 | -9.56386e-05 | -9.56386e-05 | 7.88516e-05 | 7.88516e-05 |
| 50 | gmax_dev_minus_final | 10 | -5.82185e-06 | 0 | 2 | 0 | 8 | -8.98167e-05 | -9.56386e-05 | 8.22447e-05 | 7.88516e-05 |
| 60 | eta_final | 10 | 0 | 0 | 0 | 0 | 10 | 0.000886496 | 0.000886496 | 0.000426045 | 0.000426045 |
| 60 | rmse | 10 | 0 | 0 | 0 | 0 | 10 | 0.0201484 | 0.0201484 | 0.00426672 | 0.00426672 |
| 60 | tail | 10 | 0 | 0 | 0 | 0 | 10 | 0.00972904 | 0.00972904 | 0.00130412 | 0.00130412 |
| 60 | gmax_final | 10 | -0.000206185 | -3.56197e-05 | 5 | 0 | 5 | 0.00109268 | 0.000886496 | 0.000671821 | 0.000426045 |
| 60 | s1 | 10 | -0.0378342 | -0.0353488 | 9 | 1 | 0 | 0.052303 | 0.0144688 | 0.0428007 | 0.0109345 |
| 60 | stage1_rel_err_signed | 10 | 0.0030127 | 0.000933258 | 5 | 5 | 0 | -0.00210667 | 0.000906032 | 0.0697605 | 0.0187419 |
| 60 | e1_at_0 | 10 | 0.117161 | 0.0362934 | 5 | 5 | 0 | 38.807 | 38.9241 | 2.71291 | 0.728853 |
| 60 | eta_dev_minus_final | 10 | 0 | 0 | 0 | 0 | 10 | -6.33117e-05 | -6.33117e-05 | 7.33812e-05 | 7.33812e-05 |
| 60 | gmax_dev_minus_final | 10 | -4.76624e-05 | -2.98798e-06 | 5 | 0 | 5 | -1.56493e-05 | -6.33117e-05 | 4.65084e-05 | 7.33812e-05 |

| q | flag | n_pairs | n_ref_pass | n_new_pass | n_ref_pass_new_fail | n_ref_fail_new_pass |
|---|---|---|---|---|---|---|
| 50 | G-A_pass | 10 | 10 | 10 | 0 | 0 |
| 50 | G-F_pass | 10 | 10 | 10 | 0 | 0 |
| 50 | G-N_pass | 10 | 10 | 10 | 0 | 0 |
| 50 | G-S_pass | 10 | 8 | 10 | 0 | 2 |
| 50 | S1_pass | 10 | 9 | 10 | 0 | 1 |
| 50 | v1_1_run_pass | 10 | 10 | 10 | 0 | 0 |
| 50 | run_pass | 10 | 8 | 10 | 0 | 2 |
| 60 | G-A_pass | 10 | 10 | 10 | 0 | 0 |
| 60 | G-F_pass | 10 | 10 | 10 | 0 | 0 |
| 60 | G-N_pass | 10 | 10 | 10 | 0 | 0 |
| 60 | G-S_pass | 10 | 6 | 10 | 0 | 4 |
| 60 | S1_pass | 10 | 8 | 10 | 0 | 2 |
| 60 | v1_1_run_pass | 10 | 10 | 10 | 0 | 0 |
| 60 | run_pass | 10 | 6 | 10 | 0 | 4 |

### 5b. Fresh seeds: confirmation v1.1 (20501–20520) versus confirmation v2.0 (30501–30520), unpaired

Side-by-side distributions only (different seeds, so no pairing and no test). Columns `*_root` are the v2.0 confirmation (n = 20 per q); columns `*_compare_root` are the v1.1 confirmation (n = 20 per q), read from the canonical worktree (`.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/confirmation`). Verbatim from `CB/tables.md` (source `CB/compare_distributions.csv`):

| q | metric | max_compare_root | max_root | median_compare_root | median_root | min_compare_root | min_root | n_compare_root | n_root | p10_compare_root | p10_root | p25_compare_root | p25_root | p75_compare_root | p75_root | p90_compare_root | p90_root |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | eta_dev | 0.00327081 | 0.00580334 | 0.00119899 | 0.00146333 | 0.000674568 | 0.000562397 | 20 | 20 | 0.000751248 | 0.000951095 | 0.000784946 | 0.00114658 | 0.00167661 | 0.00215274 | 0.00223823 | 0.00299072 |
| 50 | eta_dev_minus_final | 0 | 0 | -7.20367e-05 | -1.87537e-05 | -0.000261028 | -0.000209529 | 20 | 20 | -0.000237792 | -0.000136254 | -0.000193224 | -8.74241e-05 | -1.49132e-05 | 0 | 0 | 0 |
| 50 | eta_final | 0.00328969 | 0.00580334 | 0.00136472 | 0.00157111 | 0.000803954 | 0.000730676 | 20 | 20 | 0.000897929 | 0.000966492 | 0.000991526 | 0.0012828 | 0.00170076 | 0.00216799 | 0.00225599 | 0.00299072 |
| 50 | gmax_dev | 0.00327081 | 0.00580334 | 0.00147012 | 0.00146333 | 0.000711661 | 0.000562397 | 20 | 20 | 0.000772144 | 0.000951095 | 0.00105158 | 0.00114658 | 0.00179454 | 0.00215274 | 0.00280599 | 0.00299072 |
| 50 | gmax_dev_minus_final | 2.94325e-05 | 0 | -1.86015e-05 | -1.87537e-05 | -0.000261028 | -0.000209529 | 20 | 20 | -0.000237792 | -0.000136254 | -0.000156361 | -8.74241e-05 | 0 | 0 | 1.97976e-06 | 0 |
| 50 | gmax_final | 0.00328969 | 0.00580334 | 0.0015268 | 0.00157111 | 0.000917337 | 0.000730676 | 20 | 20 | 0.000973873 | 0.000966492 | 0.00112196 | 0.0012828 | 0.00181156 | 0.00216799 | 0.00278866 | 0.00299072 |
| 50 | rmse | 0.0340274 | 0.0451017 | 0.0218022 | 0.0238556 | 0.0143631 | 0.0133075 | 20 | 20 | 0.0166528 | 0.0197987 | 0.0190477 | 0.0223906 | 0.0264882 | 0.0292075 | 0.0287211 | 0.0362121 |
| 50 | rmse_dev | 0.0340274 | 0.0451017 | 0.0218022 | 0.0238556 | 0.0143631 | 0.0133075 | 20 | 20 | 0.0166528 | 0.0197987 | 0.0190477 | 0.0223906 | 0.0264882 | 0.0292075 | 0.0287211 | 0.0362121 |
| 50 | rmse_dev_minus_final | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 50 | s1 | 0.0861947 | 0.0463558 | 0.0377618 | 0.0130905 | 0.00405762 | 0.000643774 | 20 | 20 | 0.0131262 | 0.00145444 | 0.0245384 | 0.00523186 | 0.0599097 | 0.0216608 | 0.0770202 | 0.0346939 |
| 50 | s1_dev | 0.0861947 | 0.0463558 | 0.0377618 | 0.0130905 | 0.00405762 | 0.000643774 | 20 | 20 | 0.0131262 | 0.00145444 | 0.0245384 | 0.00523186 | 0.0599097 | 0.0216608 | 0.0770202 | 0.0346939 |
| 50 | s1_dev_minus_final | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 50 | stage1_rel_err_signed | 0.0861947 | 0.0463558 | 0.00110981 | -0.00491687 | -0.07694 | -0.0400337 | 20 | 20 | -0.0618396 | -0.0258672 | -0.0444678 | -0.0181794 | 0.0296979 | 0.00943987 | 0.0540794 | 0.0173385 |
| 50 | tail | 0.0121523 | 0.0122467 | 0.00887049 | 0.00796052 | 0.00499628 | 0.00651095 | 20 | 20 | 0.00543606 | 0.00683325 | 0.0081479 | 0.00722933 | 0.010061 | 0.00909552 | 0.010622 | 0.00972971 |
| 50 | tail_dev | 0.0121523 | 0.0122467 | 0.00887049 | 0.00796052 | 0.00499628 | 0.00651095 | 20 | 20 | 0.00543606 | 0.00683325 | 0.0081479 | 0.00722933 | 0.010061 | 0.00909552 | 0.010622 | 0.00972971 |
| 50 | tail_dev_minus_final | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60 | eta_dev | 0.00228346 | 0.00319467 | 0.000722168 | 0.000867859 | 0.000403927 | 0.000354395 | 20 | 20 | 0.000420756 | 0.000467818 | 0.00047468 | 0.000637085 | 0.00106857 | 0.00119707 | 0.00142381 | 0.00151274 |
| 60 | eta_dev_minus_final | 1.11022e-16 | 1.11022e-16 | -7.05464e-05 | -0.000119143 | -0.000224962 | -0.000246837 | 20 | 20 | -0.0001898 | -0.000170914 | -0.000153053 | -0.000142837 | -9.74011e-06 | -6.08574e-05 | 0 | -7.0024e-06 |
| 60 | eta_final | 0.00231803 | 0.00319467 | 0.000787586 | 0.00102462 | 0.000432158 | 0.000403486 | 20 | 20 | 0.000467131 | 0.000584036 | 0.000514803 | 0.000786659 | 0.00125163 | 0.00127969 | 0.00152925 | 0.00162221 |
| 60 | gmax_dev | 0.0036486 | 0.00319467 | 0.000837027 | 0.000867859 | 0.000403927 | 0.000354395 | 20 | 20 | 0.000438093 | 0.000467818 | 0.000577119 | 0.000637085 | 0.00120733 | 0.00119707 | 0.00184654 | 0.00151274 |
| 60 | gmax_dev_minus_final | 2.05778e-05 | 1.11022e-16 | -4.07178e-06 | -0.000119143 | -0.000224962 | -0.000246837 | 20 | 20 | -0.0001898 | -0.000170914 | -9.82191e-05 | -0.000142837 | 2.88449e-06 | -6.08574e-05 | 1.18731e-05 | -7.0024e-06 |
| 60 | gmax_final | 0.00363757 | 0.00319467 | 0.000934119 | 0.00102462 | 0.000447105 | 0.000403486 | 20 | 20 | 0.000488991 | 0.000584036 | 0.000578668 | 0.000786659 | 0.00125417 | 0.00127969 | 0.00182917 | 0.00162221 |
| 60 | rmse | 0.0375636 | 0.0374264 | 0.0181575 | 0.0229069 | 0.0138912 | 0.0152854 | 20 | 20 | 0.0153708 | 0.0165465 | 0.0158407 | 0.0177399 | 0.0249454 | 0.0262351 | 0.0352276 | 0.0348144 |
| 60 | rmse_dev | 0.0375636 | 0.0374264 | 0.0181575 | 0.0229069 | 0.0138912 | 0.0152854 | 20 | 20 | 0.0153708 | 0.0165465 | 0.0158407 | 0.0177399 | 0.0249454 | 0.0262351 | 0.0352276 | 0.0348144 |
| 60 | rmse_dev_minus_final | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60 | s1 | 0.153016 | 0.0323913 | 0.0454838 | 0.0116829 | 0.00142243 | 0.00160547 | 20 | 20 | 0.0102934 | 0.00186691 | 0.024534 | 0.00498389 | 0.0698786 | 0.0222344 | 0.087325 | 0.026228 |
| 60 | s1_dev | 0.153016 | 0.0323913 | 0.0454838 | 0.0116829 | 0.00142243 | 0.00160547 | 20 | 20 | 0.0102934 | 0.00186691 | 0.024534 | 0.00498389 | 0.0698786 | 0.0222344 | 0.087325 | 0.026228 |
| 60 | s1_dev_minus_final | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60 | stage1_rel_err_signed | 0.153016 | 0.028034 | -0.00174997 | 1.76751e-05 | -0.113466 | -0.0323913 | 20 | 20 | -0.0549015 | -0.0236709 | -0.0385638 | -0.0107741 | 0.0468021 | 0.0119948 | 0.0745687 | 0.0220679 |
| 60 | tail | 0.0137824 | 0.0164521 | 0.00939548 | 0.0104287 | 0.00613089 | 0.0075688 | 20 | 20 | 0.00755905 | 0.00861691 | 0.00866217 | 0.00935286 | 0.0107473 | 0.0110666 | 0.012303 | 0.0122793 |
| 60 | tail_dev | 0.0137824 | 0.0164521 | 0.00939548 | 0.0104287 | 0.00613089 | 0.0075688 | 20 | 20 | 0.00755905 | 0.00861691 | 0.00866217 | 0.00935286 | 0.0107473 | 0.0110666 | 0.012303 | 0.0122793 |
| 60 | tail_dev_minus_final | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

---

## 6. Anomalies and deviations

The deviations below were accepted by the owner (prompt of 2026-10-04) unless marked otherwise.

### 1. R7 failed as the tool computed it; D-R7; the checks tool fixed after the confirmation

- **What failed.** R7 of `tools/v2/v2_0_rehearsal_checks.py` (full suite and C7 at the launch commit) returned `pass: false` (`LK/rehearsal_v2_0_checks.json`, original in `CHK/rehearsal_v2_0_checks.tool_output.json`). Pytest was fine: `1 failed, 391 passed, 2 xfailed, 2 warnings in 757.28s (0:12:37)`, only the known registry failure. C7's comparison crashed: the tool compared the candidate with the reference run through a path relative to the worktree, and in this worktree that directory holds only the tracked files; the reference's `checkpoint.pt` (gitignored `*.pt`) is not in this worktree (it is in the canonical worktree, and one older worktree, `tournament-v2-stagewise-pilots-f97609`, holds a byte-identical copy, SHA-256 `2eb8bb33…4374`), so `tools/v2/compare_runs.py` raised `FileNotFoundError` in `torch.load` after the six other files had compared with 0 differences, and the tool recorded `c7_identical: false` (`results/v2_pilots/phase2_regression/v2_full_f2d616c.compare.txt` stops after `phase_B_exit_arrays.npz`; the tool kept only stdout, so the stderr was repeated afterwards: `CHK/c7_tool_error_reproduction.txt`). The R1 reference of the same tool is read from the canonical worktree, so R1 was unaffected.
- **Precision on the wording of the decision.** The reference *directory* is present in this worktree: six of the seven compared reference files are there and their SHA-256 equal the canonical copies' (`CHK/c7_tool_error_reproduction.txt`, table at the end; the canonical hashes are also the `reference (canonical)` column of `CHK/c7_canonical_reference.txt`); the file that is absent is the reference's `checkpoint.pt`. (The comment at `C7_REF` in the tool, the message of `3fedaa2` and the header of `CHK/c7_canonical_reference.txt` say the file exists "only" in the canonical worktree, which overstates that: the canonical worktree is the one the checks tool and the earlier rounds use as the reference.)
- **Owner decision D-R7 (2026-10-04):** R7 is accepted as met on the canonical-reference comparison; the literal failure is a reference-path defect of the checks tool, not of the pipeline; the precedent is Check 1 of the v1.0 lock round.
- **The record.** From `f2d616c`'s code on a clean tree (`git status --porcelain` empty apart from untracked `results/`), the C7 comparison against the canonical reference ends **IDENTICAL, including `checkpoint.pt` (0 differing tensors)**: `CHK/c7_canonical_reference.txt` holds the full output, the exact commands, the reference run's path, the SHA-256 of every compared file of the reference and of both candidates, and the commit hash. Candidate B is a fresh run (21 s, `CHK/c7_run_start_utc.txt`, `c7_run_end_utc.txt`), candidate A the one the tool's own R7 attempt produced; both give IDENTICAL, and their `checkpoint.pt` files are byte-identical to each other. The record explains why the reference's `checkpoint.pt` differs from the candidates' in bytes (three extra top-level keys in the newer file: `frozen`, `rng_minibatch`, `snapshot_refreshes`; `cfg` equal; `compare_runs.py` compares only actor, critic, opponent and the two optimizer states).
- **R7 in the checks file** records both: `c7_tool: false` with the tool's error (`FileNotFoundError: [Errno 2] No such file or directory: 'results/v2_pilots/phase1_regression/before/FINAL_A400_B25_C25/tel_q50_s10501/checkpoint.pt'`; the same message with the full traceback is in `CHK/c7_tool_error_reproduction.txt`), `c7_canonical_reference: IDENTICAL` with the path of the record, and `owner_decision`. `R7.pass` and `ALL_PASS` stay `false`, as the tool wrote them.
- **Launch.** The confirmation was launched from `d2e377d`, a results-only commit on top of the LOCK record (§1.5): the launch-commit rule holds.
- **The fix, after the confirmation and its analysis** (`3fedaa2`, `tools/v2/v2_0_rehearsal_checks.py` only: 4 hunks): C7 now compares against `C7_REF`, resolved from the canonical worktree as the R1 reference is; R7's record carries the path it used. No new check or guard, no CLI argument. Nothing in `protocols/`, the entry point, the pipeline code, `utils/v2_continuation.py` or the analysis script changed.
- **Both outputs.** (1) The tool at `f2d616c`'s version: R7 `pass: false`, `c7_identical: false` (above). (2) The fixed tool, `run_check_7` called unchanged at `3fedaa2` through `CHK/tool_fix_rerun/driver.py` (which only points the tool's log directory `LK` at `CHK/tool_fix_rerun/` so that the committed `rehearsal_v2_0_pytest.txt` is not overwritten): `{"pass": true, "pytest_summary": "1 failed, 391 passed, 2 xfailed, 2 warnings in 803.07s (0:13:23)", "pytest_failed": ["tests/test_registry_canonicalization.py::test_registry_canonicalization"], "pytest_errors": [], "c7_identical": true, "c7_commit": "3fedaa2", "c7_dirty": false, "c7_reference": "/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_pilots/phase1_regression/before/FINAL_A400_B25_C25/tel_q50_s10501"}` (`CHK/tool_fix_rerun/r7_result.json`, `driver.out`; started and ended as in §1.5). Only R7 was re-run: R1–R6 do not depend on the fix, and R5 would not hold at another launch commit by construction.

### 2. The worktree switch

The v2.0 round's session (prompt of 2026-10-03) started in the worktree `protocol-v2-r1-acceptance-e881de` and entered `v2-t2-refine` with `EnterWorktree`, because the prompt names `.claude/worktrees/v2-t2-refine` as the place to work; the session of 2026-10-04 started in `r7-canonical-reference-checks-bd0e75` and did the same. All commits are on `v2-t2-refine`. No file of the two starting worktrees was changed. The only file operation outside `v2-t2-refine` is the authorised deletion of the four stray files in a third worktree, `t2-accuracy-refinement-r1-32d472` (deviation 6).

### 3. The six-lens pre-lock review

Before the lock a read-only review ran with six lenses (protocol spec, analysis script, entry point, bit-identity and hazards, tests and check (ii), rehearsal checks) and one skeptic per major or minor finding. It produced **44 findings**; **22** were verified by a skeptic (**13 confirmed**, 9 partly, none refuted); the other 22 were nits and were not verified. While the confirmation was running (2026-10-04, 04:27–04:37 UTC; read-only), a second pass established what the files at the lock commit show for every finding, with a second agent trying to refute each row (`V20/prelock_review_ledger.md` and `.json`; the two workflow journals they come from are named in the JSON). The second agent corrected 4 ledger rows (P7, A2, T2, RC-F8); none of them is a confirmed finding, and the record keeps both labels.

**Commits.** Every review fix to a locked or code file (protocol, entry point, analysis script, tests, tools) was made in the working tree on top of `155cdec` and went into the single lock commit **`1d6d4d0`**; there is no separate pre-lock fix commit. The only other commit before the lock is `ce0573f` (docs only: the R1 acceptance addendum; no review fix). The launcher and job lists (BI-F3) and the two seed-inventory scans (P1, BI-F1) are `results/` files committed after the lock in `f2d616c`; the same commit appends the v2.0 record to `protocols/LOCK` (T4), the only file in it outside `results/`. No locked file changed there.

**The 13 confirmed findings, their fixes and commits** (each line: id, lens, severity as rated by the reviewer → by the skeptic; the finding; what was done, with the label of the ledger; the commit). The *Fix* texts describe what the files at the named commit showed when the ledger was made (while the confirmation was running), not the present state of the working tree. Full text of every finding, the skeptic verdicts and the evidence (path:line at the commit): `V20/prelock_review_ledger.json`; one ledger wording that a later check showed to be wrong (BI-F2: two modified files, in fact three) is corrected there, with the original kept in `correction_after_factcheck`.

- **P1** (protocol-spec; major → minor). *Finding:* D5 requires re-running tools/v2/seed_inventory.py at the lock commit and recording 0 collisions. That cannot be satisfied literally. The stock scanner's KEY_RE matches any JSON key containing 'seed' with an int or list value, and it scans the whole tournament_experiment tree. The new protocols/v2_T2_locked_v2_0.json declares the block as `"seed_block": [30501, ...]`, so the re-run will count its own declaration as collisions. *Fix (fixed_differently):* The reviewer asked for a procedure only (run the stock tool, record that every hit is the protocol's own seed_block). The lock did that and also added an opt-in --exclude flag to tools/v2/seed_inventory.py. Both scans were run and recorded: with the declaring files excluded, 0 collisions among 305 values (exit 0); stock, 20 hits whose single source is protocols/v2_T2_locked_v2_0.json key seed_block (exit 1). The locked JSON, MD and diff describe both runs. *Commit:* 1d6d4d0 (tool --exclude and the protocol text); f2d616c (the two scan records, LOCK record).
- **P2** (protocol-spec; minor → minor). *Finding:* The locked text says the pre-lock inventory was taken 'before the block was declared in any file'. That is false. The block 30501-30520 was already declared, as 'reserved for a later round', in tracked R1 files. The same claim is repeated in change_log entry 5 (line 714) and in the MD (lines 53 and 129). *Fix (fixed):* Every occurrence was reworded to 'before the block appeared in any JSON/CSV seed record (the block had been reserved and unused since the R1 pre-flight: reports/v2/refine/00_preflight.md, results/v2_refine/preflight/seed_inventory_30501_30520.out)'. The generator was changed and the JSON, MD and machine diff regenerated, so the three outputs agree. *Commit:* 1d6d4d0.
- **A1** (analysis-script; minor → nit). *Finding:* The script crashes with AttributeError when no (q, seed) is in DONE (all runs failed_exception, incomplete or missing), so verdict.json and tables.md are never written. *Fix (fixed):* New helper ensure_columns() adds every gate-metric column (NaN) and every gate-flag column (False) to the per-run frame, and analyse() now returns ensure_columns(pd.DataFrame(rows)). The later tables (s1 summary, rehearsal_safety, pass counts) therefore always find their columns. Same idea as the reviewer's optional suggestion (a frame that always has the column), no new gate. *Commit:* 1d6d4d0.
- **BI-F1** (bit-identity-and-hazards; major → major). *Finding:* The D5 step 're-run tools/v2/seed_inventory.py at the lock commit and record 0 collisions' cannot be satisfied literally. The scan counts the protocol's own declaration of the block as 20 collisions and exits 1. *Fix (fixed_differently):* The reviewer's primary suggestion was wording only (n_sources==1, single source = the protocol's own seed_block), with an exclusion option for the unlocked tool as the alternative. The alternative was taken, and the stock result is kept next to it. Tool at 1d6d4d0: tools/v2/seed_inventory.py gained '--exclude SUFFIX ...' (path-suffix match, p.endswith; excluded files are not counted in the scanned totals); with no --exclude the behaviour is identical to 155cdec (the diff touches only the exclude handling and a print under 'if a.exclude'); it still returns 1 iff any block seed is found. Two files were excluded: (1) protocols/v2_T2_locked_v2_0.json, which declares confirmation.seed_block [30501..30520] (the only file the stock scan attributes the 20 hits to, key seed_block); (2) results/v2_T2_locked/v2_0/protocol_diff_v1_1_to_v2_0.json, which restates the block under path /confirmation/seed_block old/new (lines ~219-262) but under keys the tool's KEY_RE does not match, so it contributes 0 hits either way and its exclusion does not change the count. Re-run at the lock commit, with exclusion: 0 collisions, 305 distinct values, EXIT=0 (seed_inventory.csv/.out). Stock scan without exclusion: 47092 json, 8783 csv, 325 distinct values, 20 collisions, each of 30501-30520 with the single source protocols/v2_T2_locked_v2_0.json key seed_block, EXIT=1 (seed_inventory_stock.csv/.out). This reproduces the reviewer's scratch run exactly (47092 / 8783 / 325 / 20). 325 - 305 = 20, the block. The locked JSON was not changed after the lock: the wording was fixed before the lock, in the JSON confirmation.disjointness_check and md section 3 (protocol JSON line 258, quoted below), which forward-cite seed_inventory.csv and seed_inventory_stock.out; the generator tools/v2/make_locked_protocol_v2_0.py:281-284 and 390-393 produce that text. The pre-lock inventory (seed_inventory_pre_lock.out: 305 values, 0 collisions, 47088 json/8782 csv, no 'excluded' line) is untouched. For the deviations list: without the exclusion the inventory counts exactly the 20 seeds the v2.0 protocol declares (stock .out lists 30501..30520 each with one source; 'recorded seeds within +-1000 of the block' is exactly 30501..30520). *Commit:* 1d6d4d0 (tool change, protocol/md wording, generator); f2d616c (the two inventory records and the commit message).
- **BI-F2** (bit-identity-and-hazards; minor → minor). *Finding:* clean_tree is evaluated once per run, at that run's start, over everything outside the top-level `results/`. Any repo write or commit between the F commit and the start of the last run makes that run's manifest dirty or gives it a different commit. *Fix (procedure_followed):* No code change (reviewer: procedure only, no new guard). git_state() in run/run_v2_stagewise.py is byte-identical between 155cdec and 1d6d4d0 (not in the diffstat). The procedure was followed: (1) L (1d6d4d0) contains all 7 formerly untracked non-results files (protocols/v2_T2_locked_v2_0.json and .md, tests/test_v2_locked_v2_0.py, tools/v2/confirmation_analysis_v2_0.py, make_locked_protocol_v2_0.py, v2_0_pass_probability.py, v2_0_rehearsal_checks.py), the staged deletion of tests/test_v2_locked.py, and the 3 modified files (run/run_v2_T2_locked.py, tests/test_v2_refine_continuation.py, tools/v2/seed_inventory.py) - all in 'git diff --stat 155cdec 1d6d4d0'. (2) The LOCK record was committed as F (f2d616c) before any launch; after f2d616c, git diff --name-only 1d6d4d0 d2e377d over protocols, run, tools, tests, utils, agents, envs, config, docs, reports lists only protocols/LOCK. (3) Rehearsal launched from f2d616c with 'git status --porcelain outside results/: []' recorded; all 20 rehearsal manifests have commit f2d616c and clean_tree true; R5 passes 20/20 in rehearsal_v2_0_checks.json. (4) The confirmation was launched from d2e377d (the results-only commit of the rehearsal records and checks), whose launch_env.txt records 'git diff --name-only <lock> HEAD, outside results/: protocols/LOCK' and 'git status --porcelain outside results/: []'; the confirmation runs on 20 workers (xargs -P 20), so the second wave exists, and at the time of this read every confirmation manifest present (including wave-2 q60 runs already started) has clean_tree true and commit d2e377d, none has false. The quiet-tree rule after launch is not written in any repo file at 1d6d4d0 or f2d616c; it is the owner's step 3 ('results/ only, so the launch-commit rule holds') and was followed in observable outcome. *Commit:* 1d6d4d0 (L content); f2d616c (F record); no code fix.
- **BI-F4** (bit-identity-and-hazards; minor → minor). *Finding:* The L commit must contain results/v2_T2_locked/v2_0/*. Nothing detects an omission, because results/ is excluded from clean_tree. A blanket `git add results/` or `-A` would also pull in R1's untracked heavy files. *Fix (procedure_followed):* No code change. L (1d6d4d0) was staged by explicit path: its only results/ entries are the 7 files of results/v2_T2_locked/v2_0 (continuation_check_v2_0.json, continuation_check_v2_0.out, protocol_diff_v1_1_to_v2_0.json, seed_inventory_pre_lock.csv, seed_inventory_pre_lock.out, v2_0_pass_probability.json, verifier_numerics_note.md) - exactly the list the skeptic gave, and the ones tests/test_v2_locked_v2_0.py::test_generator_reproduces_the_committed_protocol (via G.load_numbers and G.DIFF) needs. No results/v2_refine/** file and no train_history.json or weight/npz export is in 1d6d4d0; the heavy parents_A files and the rehearsal's train_history.json/weights/*.npz are still untracked in git status, and the later results commits did not take them (d2e377d is 'Results only' and its stat lists no train_history). The post-lock inventories went into F (f2d616c) as suggested: seed_inventory.{csv,out} and seed_inventory_stock.{csv,out}. Now 'git status --porcelain results/v2_T2_locked/v2_0' is empty and 'git ls-tree -r f2d616c results/v2_T2_locked/v2_0' lists all 11 files. *Commit:* 1d6d4d0 (L, explicit paths); f2d616c (post-lock inventories); no code fix.
- **T1** (tests-and-check-ii; minor → nit). *Finding:* The explicit assertion that S1 at exactly 0.10 passes was lost in the port. test_gates_at_exact_thresholds_pass now uses s1=0.05, which sits at the G-S threshold and is trivially below the S1 threshold. *Fix (fixed):* test_gates_at_exact_thresholds_pass now asserts S1 at 0.10 passes (with the v1.0 outcome passing and G-S and run_pass failing) and S1 at nextafter(0.10, 1.0) fails (v1.0 outcome fails, v1.1 outcome passes). This is the skeptic's suggested pair, with extra independent asserts. The v1.1 test at 431474d had s1=0.10; the 300-case cross-check still exists. *Commit:* 1d6d4d0.
- **T5** (tests-and-check-ii; minor → minor). *Finding:* Deleting tests/test_v2_locked.py leaves dangling references. The report-pack builder reads that path from the working tree and raises FileNotFoundError in build_t56 when run at this head. reports/v2/README.md:43 still lists it as the tests file. *Fix (documented_only):* The deletion is committed with its replacement stated in the commit message and in the new test file's docstring: tests/test_v2_locked.py is replaced by tests/test_v2_locked_v2_0.py, every test ported, v1.1 suite at 431474d. No code or doc references were updated. The report-pack builder still reads tests/test_v2_locked.py and would raise FileNotFoundError in build_t56 at this head. reports/v2/README.md:43 and docs/STATE.md are unchanged. No v2.0 'Anomalies and deviations' section exists in the repo yet; the owner's R7 message assigns the change-log entry to the report and the builder update to the pack rebuild (out of scope). *Commit:* 1d6d4d0.
- **T7** (tests-and-check-ii; minor → nit). *Finding:* None of (ii-a), (ii-b) or (ii-c) covers the table's linear-lookup error between y-nodes (step 0.05). That error is larger than the (ii-a) limit. The phrase 'table converged to <= 5e-9·ΔW' holds for node values only. *Fix (not_changed):* No test, code or wording change, which matches the skeptic's 'no test or code change'. The locked texts still say the table is converged without 'at its nodes'. The protocol states 'linear interpolation at lookup' but gives no between-node error figure, and no lookup-error figure appears in the check record. The 'at its nodes' qualifier for the report was not applied in the repo. *Commit:* none.
- **RC-F1** (rehearsal-checks; minor → minor). *Finding:* The R3/R4/R5 loop appends to r3 (and r4) inside the try block before later statements that can raise. The except branch then appends False to all three lists, so the lists become misaligned and the reported counts are wrong. *Fix (fixed):* At 1d6d4d0 run_checks_345 initialises b3 = b4 = b5 = False per run, assigns them inside the try, and appends each list exactly once after the try/except, so r3, r4, r5 always have one entry per run (n = 20). The except branch is a bare 'pass', so b3/b4 keep any value computed before the exception instead of being reset to False; the skeptic's suggested fix reset all three in except. *Commit:* 1d6d4d0.
- **RC-F2** (rehearsal-checks; minor → minor). *Finding:* R1's end-of-A gate-value comparison (_gates_a) is skipped without a failed field when the new run's gates.json does not exist. R1 can then report ALL=True with the gate-value part not compared. *Fix (fixed):* At 1d6d4d0 _gates_a no longer has an exists() guard: it always registers the three checks (reported.end_of_A, metric_values_A, dev_tier_values.G-A), and the gates.json files are loaded lazily inside the check lambdas via both(). C.Result.check turns any exception (missing file) into a failed field, so a missing gates.json fails R1. This is option (a) of the skeptic's fix. *Commit:* 1d6d4d0.
- **RC-F3** (rehearsal-checks; minor → minor). *Finding:* The R2 table rebuild is a bit-identity comparison computed in the checker process, but the tool neither sets torch.set_num_threads(1) nor checks the thread environment variables. The entry point and Run refuse to run unless OMP/MKL/OPENBLAS are 1 and set torch threads. *Fix (fixed):* At 1d6d4d0 main() does both things: refuses to run (exit code 4) unless OMP/MKL/OPENBLAS_NUM_THREADS are all '1', and then calls torch.set_num_threads(1). The module docstring states the requirement. The reviewer offered either measure and the skeptic preferred set_num_threads; the commit has both. *Commit:* 1d6d4d0.
- **RC-F6** (rehearsal-checks; minor → minor). *Finding:* R7 decides test success only from '^FAILED' lines. pytest reports fixture or setup errors as 'ERROR ...' lines, which are ignored, and the summary line is not used. The check passes whenever the only FAILED test is the known registry test, even if other tests ERROR. This is the same pattern as v1.1 R6. *Fix (fixed):* At 1d6d4d0 run_check_7 also collects `^ERROR ` lines and requires `failed == [KNOWN_FAILURE] and not errors`. The error list is recorded as pytest_errors in the R7 record, and the module and function docstrings now say 'no other failure, no error'. This is the skeptic's suggested fix. *Commit:* 1d6d4d0.

P1 and BI-F1 are the same defect (found by two lenses); it is the only defect the reviewer rated major (the skeptic of P1 rated it minor, the skeptic of BI-F1 kept major). The nine verified findings rated "partly" and the 22 unverified nits are in the same record with their dispositions; most were fixed in `1d6d4d0`, a few were left unchanged or only documented, as recorded there.

### 4. `tools/v2/seed_inventory.py --exclude`

The option is new and opt-in (the stock behaviour is unchanged); it was added before the lock (`1d6d4d0`).
- **The two files excluded:** `protocols/v2_T2_locked_v2_0.json` and `results/v2_T2_locked/v2_0/protocol_diff_v1_1_to_v2_0.json`. Both declare the seed block 30501–30520 in their content (the protocol as its `seed_block` entry, the machine diff as the old and new value of that entry).
- **Why:** D5 asks for the inventory to be re-run at the lock commit with 0 collisions, but the scanner counts any JSON key containing `seed` with an integer or list value, and the locked protocol itself declares the block. Without the exclusion the scan counts the protocol's own declaration: scanned 47092 json, 8783 csv files; 325 distinct seed values; block 30501-30520: 20 collisions (`V20/seed_inventory_stock.out`; exit 1), and each of 30501–30520 has `n_sources` = 1, the source being `protocols/v2_T2_locked_v2_0.json`, key `seed_block` (`V20/seed_inventory_stock.csv`). **Without the exclusion the inventory therefore counts exactly the 20 seeds the v2.0 protocol itself declares, and nothing else.** (The diff file is not a hit of the stock scan: its entries are under the keys `old` and `new`, which the scanner's key pattern does not match; excluding it changes no count.)
- **With the exclusion:** scanned 47090 json, 8783 csv files; 305 distinct seed values; block 30501-30520: 0 collisions (`V20/seed_inventory.out`; exit 0), i.e. the stock scan's 325 distinct values minus the 20 declarations = the 305 distinct values of the pre-lock scan.
- **Taken before the block appeared in any JSON or CSV seed record** (`V20/seed_inventory_pre_lock.out`): scanned 47088 json, 8782 csv files; 305 distinct seed values; block 30501-30520: 0 collisions. The block had been reserved, and unused, in prose and in `tools/v2/refine_preflight.py` since the R1 pre-flight (`reports/v2/refine/00_preflight.md`, `docs/STATE.md`); the locked text says exactly that (pre-lock review finding P2).
- The scan records of the lock commit were written in `f2d616c`; the locked text, hashed at `1d6d4d0`, names them before they exist.

### 5. `tests/test_v2_locked.py` replaced by `tests/test_v2_locked_v2_0.py`

**Change-log entry.** The lock commit `1d6d4d0` deletes `tests/test_v2_locked.py` and adds `tests/test_v2_locked_v2_0.py`. Every v1.1 test is ported: of the 15 test functions of the v1.1 file, 13 keep their name and 2 were renamed to match the v2.0 content (`test_v1_0_protocol_untouched` → `test_earlier_protocol_files_untouched`, which now asserts the v1.0 and v1.1 files and the v1.1 analysis script; `test_run_pass_is_GA_GF_GN_only` → `test_run_pass_is_GA_GF_GN_GS_only`, with the v1.1 behaviour pinned by the new `test_verdicts_of_the_v1_1_protocol_are_unchanged`). The v2.0 file has 34 test functions. The v1.1 file stays at `431474d` (identical at `155cdec`). The report-pack builder still reads the old path (`tools/v2/report/sec_stage1.py`; the pack's T56 cites it); it is updated at the pack rebuild, which is out of scope for this round. The pack's T57 is stale for a different reason: it cites `docs/STATE.md` by SHA-256, and that file changed.
- Files outside `results/` that name `tests/test_v2_locked.py` in the working tree (read from `git grep -l`; `docs/STATE.md` names it only in the uncommitted edit of this round, where it records the replacement): `docs/STATE.md`, `reports/v2/README.md`, `reports/v2/pi_record/prompts/09_v1_1_confirmation_prompt.md`, `reports/v2/protocol_lock_and_rehearsal.md`, `reports/v2/protocol_v1_1_confirmation.md`, `reports/v2/t2_report/provenance/T56_sources.csv`, `reports/v2/t2_report/tables/T56_v1_1_global_rng_hardening.csv`, `reports/v2/t2_report/tables/T56_v1_1_global_rng_hardening.md`, `tests/test_v2_locked_v2_0.py`, `tools/v2/report/sec_stage1.py`. The pack-builder dependency is `tools/v2/report/sec_stage1.py` (pre-lock review finding T5).

### 6. The four stray files

Four untracked early copies sat in the old session worktree `.claude/worktrees/t2-accuracy-refinement-r1-32d472` (`git status --porcelain --untracked-files=all` there lists exactly these four, all `??`; its HEAD `f02a256`). Recorded in `V20/stray_files_record.json` before deletion, compared with the tracked version at `155cdec`:

| file | SHA-256 | bytes | byte-identical to the tracked version at `155cdec` | `git diff --stat` against `155cdec` |
|---|---|---|---|---|
| `utils/v2_continuation.py` | `4af50d84e50632ea520afc2bee11425b0b661e797dc6e61f6d77441e78530c0e` | 12415 | **no** (tracked SHA-256 `513fef4b304e4660…`) | utils/v2_continuation.py / 6 +-----;  1 file changed, 1 insertion(+), 5 deletions(-) |
| `agents/ppo_pathwise.py` | `e244d5fa8fcfbcba734217589af40ac8f0f1b864820ae97536c7e4d9c4c44579` | 9736 | yes | — |
| `tests/test_v2_refine_pathwise.py` | `a9d61299e82620f71af8936d47bf9d77654a2793d03a4d4bf9ca3e5bf63ffd40` | 19600 | yes | — |
| `tools/v2/refine_preflight.py` | `1f4aa8a63f07be33679c69439da9318b16357426d7ca6eaf60e085ecdbbee073` | 35732 | **no** (tracked SHA-256 `74f9288e84e06aac…`) | tools/v2/refine_preflight.py / 83 ++++-------------------------------;  1 file changed, 8 insertions(+), 75 deletions(-) |

The full diffs of the two that differ are in `V20/stray_files_diffs.patch`. As authorised, all four were then deleted from that worktree (not a git change; nothing else in it was touched). `utils/v2_continuation.py` at `155cdec` is the locked module (its SHA-256 is the one in the LOCK record).

### 7. Other anomalies

- **One failed run in the confirmation** (q=50 seed 30510, a G-A η₂ failure in Phase A, which v2.0 leaves unchanged), reported as computed; no re-run (D5: only an infrastructure kill is re-run).
- **Known issues at the lock** (`protocols/v2_T2_locked_v2_0.md` §8): the registry test failure; the Phase-A clamp flags M1 and M2 of the R1 diagnostic D1 (recorded, not fixed); the verifier numerics item (§2); the module docstring of `utils/v2_continuation.py` still calling check (ii) open (superseded by the protocol text).
- **Two analysis invocations.** `CA/` is the pre-registered invocation; `CB/` adds `--compare-root` for §5b. They agree on every common output (`LK/confirmation_v2_0_checks.txt`).
- **`LK/confirmation_v2_0_checks.txt`** is a plain read of the 40 manifests, statuses and the launcher log (a script of this session, not a gate); it is the source of the 40/40 statements about the v2.0 hash, `clean_tree`, commit and table SHA-256.
- **Housekeeping, in a separate commit after this report** (authorised): dated addenda in `docs/STATE.md` (after the R1 section) and at the end of `reports/v2/refine/summary.md` recording that `origin/v2-t2-refine` is at `155cdec`; the original lines are unchanged. The `main` checkout was not touched and the report pack (`reports/v2/t2_report`) was not rebuilt (its T56 and T57 are stale, §6 deviation 5).
- **Provenance of two records.** `V20/fullsuite_prelock.txt` is a byte-for-byte copy (SHA-256 `36f86d2f2504928f4e32f9dfecb6a94248a52cce63c9de7e6a08040bff4fb1c7`) of the log the lock session wrote to its scratch directory before the lock commit; `V20/prelock_review_ledger.json` is generated from two workflow journals of Claude Code sessions that are not in the repository (their paths are in the JSON). Both are evidence copied into the repository for this report, not outputs of a repository tool.
- **Load at launch.** The load average before the confirmation launch was higher than before the rehearsal launch (§4); its source was not identified. Both used 20 workers; the wall times are in §4.

---

## 7. Commands to reproduce

In the order they were run. `PY` is `/home/fjiang4/tournament_experiment/.venv/bin/python`; `T1` is `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`; `<worktree>` is `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine`; `<canonical>` is `.../worktrees/pilot-4-stabilization-fb99a2`.

**Before the lock** (2026-10-03, the lock session; these commands are from its transcript, not from a repository file):

```bash
PY tools/v2/seed_inventory.py --block 30501 30520 --out results/v2_T2_locked/v2_0/seed_inventory_pre_lock.csv > results/v2_T2_locked/v2_0/seed_inventory_pre_lock.out 2>&1
```
The pre-lock scan was run (22:30 UTC) before the block appeared in any JSON or CSV seed record; it cannot be re-created identically now, because the protocol JSON declares the block (a re-run needs `--exclude`, below).

```bash
T1 PY tests/test_v2_refine_continuation.py --write > results/v2_T2_locked/v2_0/continuation_check_v2_0.out 2>&1
PY tools/v2/v2_0_pass_probability.py
PY tools/v2/make_locked_protocol_v2_0.py
```
`make_locked_protocol_v2_0.py` reads the v1.1 protocol JSON (the base it changes), the check (ii) record, `v2_0_pass_probability.json`, the pre-lock inventory, the R1 record `results/v2_refine/continuation_check.json` and `utils/v2_continuation.py`, and writes `protocols/v2_T2_locked_v2_0.json` and `.md`, `V20/protocol_diff_v1_1_to_v2_0.json` and `V20/verifier_numerics_note.md`; it was run before the lock commit.

Each redirected command below was followed by `echo EXIT=$? >> <the same .out>` (the two analysis runs by `echo rc=$? | tee -a <the same .out>`); the exit codes quoted from the `.out` files are those lines. The lock session's full-suite log went to a scratch file (copied as `V20/fullsuite_prelock.txt`).

```bash
T1 PY -m pytest tests/ -q -p no:cacheprovider
```

**At the lock commit** (`1d6d4d0`; scans, launcher and the LOCK record written in `f2d616c`; per the lock session's transcript, the v2.0 record in `protocols/LOCK` was appended by a short script of that session that computes the hashes from the files and takes the lock commit as argument):

```bash
PY tools/v2/seed_inventory.py --block 30501 30520 --out results/v2_T2_locked/v2_0/seed_inventory.csv --exclude protocols/v2_T2_locked_v2_0.json results/v2_T2_locked/v2_0/protocol_diff_v1_1_to_v2_0.json > results/v2_T2_locked/v2_0/seed_inventory.out 2>&1
PY tools/v2/seed_inventory.py --block 30501 30520 --out results/v2_T2_locked/v2_0/seed_inventory_stock.csv > results/v2_T2_locked/v2_0/seed_inventory_stock.out 2>&1
```

**Re-rehearsal** (launch commit `f2d616c`; command as issued in the lock session, not recorded in a repository file):

```bash
tmux new-session -d -s rehearsal "cd <worktree> && cat results/v2_T2_locked/rehearsal_v2_0/jobs.txt | xargs -P 20 -L 1 bash results/v2_T2_locked/launch_one_v2_0.sh results/v2_T2_locked/rehearsal_v2_0"
```

```bash
T1 PY tools/v2/v2_0_rehearsal_checks.py --launch-commit f2d616cc921e2c6be46c48336d31372dbbf58481 > results/v2_T2_locked/rehearsal_v2_0_checks.out 2>&1
```
Run with the tool as it was at `f2d616c` (before the fix of `3fedaa2`, which changed only C7). The tool writes `RA/` itself, by running in a subprocess: `PY tools/v2/confirmation_analysis_v2_0.py --root results/v2_T2_locked/rehearsal_v2_0 --seeds 10501 10510 --rehearsal-safety --paired-root <canonical>/results/v2_T2_locked/rehearsal_v1_1 --paired-label rehearsal_v1_1 --out results/v2_T2_locked/rehearsal_v2_0_analysis`.

**C7 against the canonical reference** (the comparison of `CHK/c7_canonical_reference.txt`, run at `f2d616c`; the tool's own failing call, repeated, is in `CHK/c7_tool_error_reproduction.txt`):

```bash
T1 PY -B run/run_v2_stagewise.py --config results/v2_pilots/phase2_regression/run_config.json --out-dir <candidate dir>
PY tools/v2/compare_runs.py <canonical>/results/v2_pilots/phase1_regression/before/FINAL_A400_B25_C25/tel_q50_s10501 <candidate dir>
```

**Confirmation** (launch commit `d2e377d`; the command line is in `LK/confirmation_v2_0/launch_command.txt`):

```bash
tmux new-session -d -s confirm_v2_0 "cd <worktree> && cat results/v2_T2_locked/confirmation_v2_0/jobs.txt | xargs -P 20 -L 1 bash results/v2_T2_locked/launch_one_v2_0.sh results/v2_T2_locked/confirmation_v2_0"
```

```bash
PY tools/v2/confirmation_analysis_v2_0.py --root results/v2_T2_locked/confirmation_v2_0 --out results/v2_T2_locked/confirmation_v2_0_analysis > results/v2_T2_locked/confirmation_v2_0_analysis.out 2>&1
PY tools/v2/confirmation_analysis_v2_0.py --root results/v2_T2_locked/confirmation_v2_0 --out results/v2_T2_locked/confirmation_v2_0_analysis_vs_v1_1 --compare-root <canonical>/results/v2_T2_locked/confirmation --compare-seeds 20501 20520 > results/v2_T2_locked/confirmation_v2_0_analysis_vs_v1_1.out 2>&1
```

**After the confirmation:** the fix is the commit `3fedaa2`; the R7 re-run calls the tool's `run_check_7` unchanged (the driver only points the tool's log directory at `CHK/tool_fix_rerun/`):

```bash
PY -B results/v2_T2_locked/rehearsal_v2_0_checks/tool_fix_rerun/driver.py 3fedaa2a28e53455fc0b11a0bc17b8a366ca9dfa
```

Several record files were written by short scripts of the two sessions that are not in the repository: `CHK/c7_canonical_reference.txt`, `CHK/c7_tool_error_reproduction.txt`, the amendment of `LK/rehearsal_v2_0_checks.json`, `LK/confirmation_v2_0_checks.txt`, `V20/prelock_review_ledger.*`, `V20/stray_files_record.json` and `.patch`, and the `launch_env.txt`, `launch_diff_stat.txt` and `launch_time_utc.txt` of the confirmation (and, in the lock session, the `launch_env.txt` and `launch_time_utc.txt` of the re-rehearsal, by inline shell). The two C7 records give their commands and sources, the checks-file amendment and the ledger name their sources, and `LK/confirmation_v2_0/launch_command.txt` describes the confirmation's launch files; `LK/confirmation_v2_0_checks.txt`, `launch_time_utc.txt`, `launch_diff_stat.txt` and the `.patch` carry no statement of source or command (the checks file reads `manifest.json`, `gates.json`, `status.json`, the continuation-table files and `launcher.out` of the 40 runs, and compares the two analysis directories).

## Addendum (2026-10-04): pushed state

Addendum (2026-10-04, P0 of round R2b): the v2.0 commits are no longer unpushed. `origin/v2-t2-refine` is at `e89b61d` (pushed on 2026-10-04; `git ls-remote origin v2-t2-refine`). Annotated tags for the publication: `t2-v2-lock-v2.0` -> `1d6d4d0` (the v2.0 lock commit), `t2-v2-confirmation-v2.0` -> `d2e377d` (the confirmation launch commit) and `t2-v2-main-v2.0` -> the commit that adds `reports/v2/refine_r2b/00_housekeeping.md` (the head of `v2-t2-refine` after the P0 housekeeping commits). The fast-forward of `main` and the push of `main` and the tags are recorded in `reports/v2/refine_r2b/00_housekeeping.md`. The "not pushed" wording above is left as written; it was true when it was written.
