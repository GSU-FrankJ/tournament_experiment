# P0 pre-flight for the T=2 accuracy-refinement round R1

Date of the measurements: 2026-10-03 (UTC times below). Spec: PROMPT.md section 1, items 1-7. Every number below cites the file or command it comes from. The generated tables are written by `tools/v2/refine_preflight.py` and pasted here unchanged; their sources are the files named in each section.

## 0. Status

| item | requirement | outcome |
|---|---|---|
| 1 | `main` clean, head `f02a256`, three tags | head and tags match. The local `main` ref and its checkout do **not** match (stale ref, dirty checkout). The branch was taken from `f02a256` = `origin/main` = tag `t2-v2-main`. Deviation, see section 1. |
| 2 | worktree and branch, host facts, venv versions | done, section 2. Versions match the embedded-record requirement. |
| 3 | full suite and C7 at the branch point | 103 passed, 1 failed (the known registry test), 2 xfailed. C7 identical. Section 3. These two records come from runs made earlier in this session by the lead, not from this fix pass. |
| 4 | parents inventory | 20/20 parents present, 20/20 show the v1.1 protocol hash and launch commit `95c000e`, section 4. `results/v2_refine/parents.csv`. |
| 5 | seed inventory | section 5. |
| 6 | baseline reference numbers | section 6. |
| 7 | KL appendix | section 7. |

## 1. Git state (item 1)

Commands run in the branch worktree `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine`, 2026-10-03 18:06 UTC.

| what | command | result |
|---|---|---|
| branch head | `git rev-parse HEAD` | `f02a256095dcaa69dffbd011718d337f84183890` (`chore: refresh report pack T05 and T57 after adding the v2 index`, committed 2026-10-03T03:18:04+00:00) |
| `origin/main` (local remote-tracking ref, not fetched) | `git rev-parse origin/main` | `f02a256095dcaa69dffbd011718d337f84183890` |
| tag `t2-v2-main` | `git rev-parse t2-v2-main^{commit}` | `f02a256095dcaa69dffbd011718d337f84183890` (annotated tag object `70fae9fb...`) |
| tag `t2-v2-lock-v1.1` | `git rev-parse t2-v2-lock-v1.1^{commit}` | `431474d18259ff7534520b8a6d52616beb36c800` (`feat: lock v2 T=2 protocol v1.1 ...`) |
| tag `t2-v2-confirmation` | `git rev-parse t2-v2-confirmation^{commit}` | `f6838ec2550dbb7ce946a4085afd9189f03c8f5c` (`chore: add v1.1 re-rehearsal records and R1-R6 checks (all pass)`) |
| launch commit of the parents | `git rev-parse 95c000e` | `95c000e0670072ee0a382439c3a2c5ddfda8fcce` (`chore: record v2 T=2 protocol v1.1 lock (commit 431474d) and C7 at the lock`) |
| branch creation | `git reflog show v2-t2-refine` | `f02a256 v2-t2-refine@{0}: branch: Created from f02a256` |

**Deviation from item 1.** The local `main` ref is `d1b84437d3be32daf40a58d77e044201b7ba09d0` (2026-08-29, `Add files via upload`). It is an ancestor of `f02a256` and 52 commits behind (`git rev-list --count main..HEAD` = 52, `git merge-base --is-ancestor main HEAD` true). So "`git status` on `main` is clean, head `f02a256`" cannot be satisfied literally: the local `main` ref does not point at `f02a256`. The checkout at `/home/fjiang4/tournament_experiment` also carries other sessions' uncommitted files (12 entries: one modified file `SESSION_STATE.md`; untracked `MultiStage/Discussion/`, `MultiStage/Plan/`, `experiments/`, and 8 smoke JSON files under `results/*/convergence/`). That list is from `git -C /home/fjiang4/tournament_experiment status --short` run earlier in this session by the first pass of this worker; it was not re-measured in the fix pass, because the session guard refuses `git -C` to the shared checkout. The worktree was created from `f02a256`, which is also `origin/main` and the tag `t2-v2-main`. Nothing in this round touches the local `main` ref or its checkout.

Code identity between the launch commit and the branch point: `git diff --name-only 95c000e f02a256 -- run agents utils envs protocols` prints nothing, so the code that produced the parents is the code at the branch point.

## 2. Worktree and host (item 2)

| what | value | source |
|---|---|---|
| worktree | `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine` | `git worktree`, session environment |
| branch | `v2-t2-refine`, created from `f02a256` | `git reflog show v2-t2-refine` |
| time of the measurements | 2026-10-03T18:06:32Z | `date -u` |
| `nproc --all` | 64 (2 sockets, Intel Xeon Gold 6242 @ 2.80 GHz) | `nproc --all`, `lscpu`. Plain `nproc` prints 1 when `OMP_NUM_THREADS=1` is exported (GNU `nproc` honours it), so use `--all`. |
| load average (1, 5, 15 min) | 4.11, 2.98, 1.76 | `/proc/loadavg`, 18:06:32Z. Other workers of this round were running tests and probes at that time. An earlier reading in this session at 17:51Z was 0.35, 0.57, 0.56. |
| RAM | 754 GiB total, 737 GiB available | `free -g` |
| disk `/home` | 5,952,313,540,608 B total, 3,335,510,786,048 B used, 2,316,746,440,704 B available (60 % used, about 2.1 TiB free) | `df -B1 /home` |
| Python | 3.12.3 | `.venv/bin/python` |
| torch | 2.5.1+cu121 | same |
| numpy | 2.5.0 | same |
| scipy / pandas (information) | 1.18.0 / 3.0.3 | same |

The required versions (Python 3.12.3, torch 2.5.1+cu121, numpy 2.5.0) are exactly the ones found.

## 3. Test suite and C7 at the branch point (item 3)

Both records were produced earlier in this session and are labelled by the lead as branch-point facts (`DESIGN.md`, "Branch-point facts"); the file times are 17:43 (suite) and 17:44 to 17:54 (C7) on 2026-10-03. This pass did not repeat them and did not verify whether other workers' uncommitted edits were already present in the worktree when they ran (the suite run covers `HEAD` = `f02a256` plus whatever the working tree held at that time). The suite is not re-run in this fix pass because other workers' files in the worktree are mid-edit.

- Full suite: `results/v2_refine/preflight_pytest.txt`. Last lines: `FAILED tests/test_registry_canonicalization.py::test_registry_canonicalization`, `1 failed, 103 passed, 2 xfailed in 127.91s (0:02:07)`, `EXIT=1`. The only failure is the known registry test. The file does not record the command line; the lock-round command is `python -m pytest tests/ -q -p no:cacheprovider` (`reports/v2/protocol_lock_and_rehearsal.md`, command block that ends with the suite run).
- C7 (`run/run_v2_stagewise.py --config results/v2_pilots/phase2_regression/run_config.json`, compared with `tools/v2/compare_runs.py`; the reference run is `results/v2_pilots/phase1_regression/before/FINAL_A400_B25_C25/tel_q50_s10501` in the canonical worktree, where its `checkpoint.pt` exists, as stated in the lead's DESIGN.md; this report did not re-run the comparison): `results/v2_refine/c7/branchpoint_f02a256.compare.txt`. Result: `train_history.json`: 0 differing leaves; `final_eval.json`: 0 differing leaves; `arrays.npz`: 90 arrays, 0 differ; `checkpoint_weights.npz`: 12 arrays, 0 differ; `phase_A_exit_arrays.npz` and `phase_B_exit_arrays.npz`: 41 arrays each, 0 differ; `checkpoint.pt`: 0 differing tensors; final line `IDENTICAL`. Run output: `results/v2_refine/c7/branchpoint_f02a256.run.out`.

## 4. Parents inventory (item 4)

Command: `python tools/v2/refine_preflight.py parents` (single-threaded). It reads, read-only, `state_end_A.pt`, `state_end_B.pt`, `manifest.json` and `status.json` of every run of `rehearsal_v1_1` in the canonical worktree
`/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/rehearsal_v1_1/q{50,60}/seed{10501..10510}/`
and writes `results/v2_refine/parents.csv` (20 rows; columns: absolute paths, byte sizes and full SHA-256 of both states, `protocol_sha256`, `commit`, `git_dirty`, `clean_tree`, and the three checks). It exits with status 2 and lists the problem if a state file or manifest is missing or the protocol hash or commit does not match; it never regenerates anything. Exit status here: 0.

Result (`results/v2_refine/parents.csv`):

- 20 of 20 runs have `status.json` state `done`, both state files present, 20 distinct SHA-256 values for `state_end_A.pt` and 20 for `state_end_B.pt`.
- `manifest.json` `locked_protocol.sha256` is `21d85983f2a2bebc0998e99fcf665c2c729f060996c529fa3162b3d62222a40f` in 20 of 20. It equals `sha256sum protocols/v2_T2_locked_v1_1.json` in this worktree (`21d85983f2a2bebc0998e99fcf665c2c729f060996c529fa3162b3d62222a40f`). `protocol_version` is `1.1` in 20 of 20.
- `manifest.json` `commit` is `95c000e0670072ee0a382439c3a2c5ddfda8fcce` in 20 of 20 (`git.short` `95c000e`, `git.dirty` false, `clean_tree` true).
- Sizes: `state_end_A.pt` 164126 bytes in 14 runs and 164062 bytes in 6; `state_end_B.pt` 184996 bytes in 12 runs and 184932 bytes in 8.
- Per D6 every run config of this round must carry the absolute path and the SHA-256 of the parent it uses; the full hashes are in `parents.csv`.

Compact view (first 16 hex digits of each SHA-256; generated into `results/v2_refine/preflight/parents_table.md`):

<!-- generated by tools/v2/refine_preflight.py parents; full hashes in results/v2_refine/parents.csv -->

| q | seed | state_end_A.pt bytes | state_end_A.pt SHA-256 (first 16) | state_end_B.pt bytes | state_end_B.pt SHA-256 (first 16) | protocol hash = v1.1 | commit 95c000e | git dirty |
|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | 164126 | 1b7eb86879b31f53 | 184996 | 6f4c60293b28b830 | yes | yes | no |
| 50 | 10502 | 164126 | 3867eec41b948b29 | 184996 | 4fe10c38396b57a9 | yes | yes | no |
| 50 | 10503 | 164126 | 61466a5395d29dde | 184932 | 3854ce165f4e58db | yes | yes | no |
| 50 | 10504 | 164062 | e0cac13bc743785d | 184932 | 1d105c6eb872e940 | yes | yes | no |
| 50 | 10505 | 164062 | bbea224987273cea | 184932 | 778999d2e5f24fa4 | yes | yes | no |
| 50 | 10506 | 164126 | 8e108dc6e8a738f0 | 184996 | 4c2ed8d4867de431 | yes | yes | no |
| 50 | 10507 | 164126 | 80e23aa46d768a4b | 184996 | 0244f33dff34f834 | yes | yes | no |
| 50 | 10508 | 164062 | 410fb01b5a135467 | 184932 | 848f9a6b6b514f3d | yes | yes | no |
| 50 | 10509 | 164126 | 6116b23196f8c027 | 184996 | e31cc49d5e4a1302 | yes | yes | no |
| 50 | 10510 | 164126 | c68e23c893345fd8 | 184996 | 8b9a0fba4f0f9c0e | yes | yes | no |
| 60 | 10501 | 164126 | fce4fdb1ed84538b | 184996 | b3770ffdb5bd084f | yes | yes | no |
| 60 | 10502 | 164126 | b05dd265c6f5f11a | 184996 | 0bb6abddf4623e1a | yes | yes | no |
| 60 | 10503 | 164126 | 3571b7c48660f1c6 | 184932 | c5e0c5f574c17495 | yes | yes | no |
| 60 | 10504 | 164062 | d25af3f9dc76307b | 184932 | efdc8f39a4dc309b | yes | yes | no |
| 60 | 10505 | 164062 | d6829b8463ac7b7e | 184932 | 6b1e3c64b51c267f | yes | yes | no |
| 60 | 10506 | 164126 | 79a829e85e27fec3 | 184996 | 6d90b84686196ca4 | yes | yes | no |
| 60 | 10507 | 164126 | b04be9e13a6f3690 | 184996 | a5f4aaf98bd1cc7a | yes | yes | no |
| 60 | 10508 | 164062 | 7ab7e8bdc710a5ab | 184932 | a46711ef60d11195 | yes | yes | no |
| 60 | 10509 | 164126 | 0fc3893b52105516 | 184996 | ee989a2237274d86 | yes | yes | no |
| 60 | 10510 | 164126 | 816011319ac0c90d | 184996 | 00377e55c9ae8578 | yes | yes | no |

## 5. Seed inventory (item 5)

Command (in tmux session `p0fix_seeds`, single-threaded, about 3 minutes): `python tools/v2/refine_preflight.py seeds`. It runs the stock `tools/v2/seed_inventory.py` twice, with the same extra roots as the lock-round inventory (`/home/fjiang4/tournament_experiment_upload_20260927`, `/home/fjiang4/TEL_PPO`; lock-round command in `reports/v2/protocol_lock_and_rehearsal.md`, the `seed_inventory.py --block 20501 20520` line):

```
python tools/v2/seed_inventory.py --block 10501 10510 --out <csv> --extra-root /home/fjiang4/tournament_experiment_upload_20260927 /home/fjiang4/TEL_PPO
python tools/v2/seed_inventory.py --block 30501 30520 --out <csv> --extra-root /home/fjiang4/tournament_experiment_upload_20260927 /home/fjiang4/TEL_PPO
```

and then one more in-process scan with the same roots, to list for each seed of the blocks 10501-10510, 30501-30520 and (information only) 20501-20520 every source file (collapsed across checkouts). Stock outputs: `results/v2_refine/preflight/seed_inventory_10501_10510.{csv,out}` and `seed_inventory_30501_30520.{csv,out}`; per-seed source listings: `seed_sources_*.csv`, `seed_source_categories_*.csv`; commands and counts: `seed_inventory_summary.json`. The stock tool scans every `*.json` and `*.csv` under `/home/fjiang4/tournament_experiment` (all worktrees; `.git` and `.venv` skipped) plus the two extra roots, and `reports/v2/phase0_audit.md`.

| block | role | stock exit code | stock result | source: |
|---|---|---|---|---|
| 10501-10510 | development seeds | 1 (= every seed of the block is recorded, expected) | 10 collisions (all 10 seeds); 42243 json and 6802 csv files scanned; 288 distinct seed values | `seed_inventory_10501_10510.out` |
| 30501-30520 | reserved for a later round | **0** | **0 collisions**; recorded seeds within +-1000 of the block: none | `seed_inventory_30501_30520.out` |
| 20501-20520 | confirmation seeds (information only, not an item-5 requirement) | not run through the stock CLI | all 20 seeds recorded, 510 distinct (seed, path, key) entries, all under `protocols/`, `reports/v2` and `results/v2_T2_locked/confirmation*` | `seed_source_categories_20501_20520.csv` |

**10501-10510 are the development seeds.** `protocols/v2_T2_locked_v1_1.json`, key `development_seeds`, is exactly `[10501, ..., 10510]` (read by the tool, `seed_inventory_summary.json` -> `development_seeds_in_protocol.equals_10501_to_10510` = true; also `protocols/v2_T2_locked.md` line 130). The seeds are recorded in 1909 distinct (seed, path, key) entries: the two protocol files, `reports/v2`, the rehearsal and pilot results (`results/v2_T2_locked/rehearsal*`, `results/v2_pilots/pilot1` to `pilot4*`, `phaseA_ext`, `induced_band`, regression and smoke runs), and this round's own `results/v2_refine/c7` and `results/v2_refine/parents.csv` (`seed_source_categories_10501_10510.csv`). All 1909 entries are under v2 paths (protocols, reports/v2, results/v2_*); no other experiment in the scanned roots records them.

**30501-30520 collide with nothing.** The stock CLI reports 0 collisions; the detail scan finds 0 sources for each of the 20 seeds (`seed_sources_30501_30520.csv` has only its header; `seed_inventory_summary.json` `sources_30501_30520.n_seeds_with_any_source` = 0). No recorded seed value lies between 20521 and 734999 (the smallest recorded value above 20520 is 735000; `seed_inventory_30501_30520.csv`), so the block is also far from anything recorded. The block stays reserved and is not used in this round. The scan was made on 2026-10-03 at about 18:05-18:09 UTC with the other workers of this round active in the same tree; their files record only 10501-10510 where they record seeds at all. Seed 10501-10510 entries of this round's own outputs under `results/v2_refine/preflight/` are skipped by the detail scan and cannot affect the 30501-30520 result (they hold no 30501-30520 value).

## 6. Baseline reference numbers (item 6)

Sources, all read-only in the canonical worktree
`/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/`:

- Per-run values of the paired baseline (`rehearsal_v1_1`, q in {50, 60} x seeds 10501-10510): `rehearsal_v1_1/q*/seed*/gates.json` (`reported.end_of_A`, `reported.end_of_B`, `metric_values`, `G-A`, `G-F`, `G-N`, `S1`, `outcome`) and `v2_run_summary.json` (`total_wall_sec`). Written to `results/v2_refine/preflight/baseline_rehearsal_v1_1_per_run.csv` (one row per run, 41 columns) and, as medians, `baseline_rehearsal_v1_1_per_q_median.csv`; pass counts: `baseline_rehearsal_v1_1_gate_counts.csv`.
- Confirmation medians (n = 20 per q, seeds 20501-20520): `confirmation_analysis/per_run.csv`, `confirmation_analysis/reported_metrics.csv`; each row below names its source file and column. Written to `baseline_confirmation_medians.csv`. Where `confirmation_analysis/distributions.csv` holds the same metric, the median agrees to 0.0 (`baseline_confirmation_medians_check.json`).
- Transcription check: the per-run values taken from `gates.json` equal the columns of `rehearsal_v1_1_analysis/per_run.csv` and `reported_metrics.csv` exactly (maximum absolute difference 0.0 over 17 metric pairs, outcome string equal in 20/20; `baseline_rehearsal_v1_1_crosscheck.json`). This shows that nothing was mistyped. It is not an independent recomputation, because those analysis files are themselves written from `gates.json`.

Definitions (`utils/v2_metrics.py`, `recovery_metrics`): stage-1 signed error = (e1(0) - g1)/g1; signed peak error = (e2(0) - g2(0))/g2(0) at d = 0; RMSE_pos = root mean square of e2 - g2 over |d| < 2q, divided by g2(0); tail = the part of D_2 with |d| >= 2q, tail mean = mean of e2 there (absolute and divided by g2(0)), tail max = max of e2 there (effort units). g2(0) is written `e2*(0)` in the tables. eta_2 and Gmax_full are the verifier values divided by Delta W; "final" and "development" are the two verifier tiers; "dev - final" is the G-N quantity. "S1" is the stage-1 criterion |e1(0) - e1*|/e1* <= 0.10, a secondary criterion without a pass rule in v1.1 (`reports/v2/protocol_v1_1_confirmation.md` section 1.3, change 1, and section 4.3); the run outcome (`pass`) is G-A and G-F and G-N. The 0929 target is |stage-1 error| <= 0.05.

Reading notes (all from the tables below):
- Rehearsal v1.1 run outcome: pass in 10/10 at q = 50 and 10/10 at q = 60 (G-A, G-F and G-N each 10/10 at both q). S1 passes in 9/10 (q = 50) and 8/10 (q = 60). The three S1 failures are q = 50 seed 10507 (|err| 0.1027) and q = 60 seeds 10505 (0.1260) and 10510 (0.1130); all three runs still pass the run outcome.
- The 0929 target (|err| <= 0.05) is met by 8/10 (q = 50) and 6/10 (q = 60) rehearsal runs and by 12/20 and 13/20 confirmation runs (`baseline_s1_target_counts.csv`; the confirmation counts agree with `reports/v2/t2_report/tables/T57_issue_list.md`, issue 2).
- The location (t*, d*) of the largest Gmax_full on the final tier is stage 2 at a d between -24 and -2 in 13 of 20 runs and stage 1 at d = 0 in 7 of 20 runs (per-run table below); this is reported, not interpreted.
- Wall time of one locked run (A1600 + B600) is 312.8 s to 393.5 s in the table (`v2_run_summary.json` `total_wall_sec`); the 393.5 s run is q = 50 seed 10502.

Generated tables (`results/v2_refine/preflight/baseline_tables.md`, command `python tools/v2/refine_preflight.py baseline`):

<!-- generated by tools/v2/refine_preflight.py baseline -->

#### Rehearsal v1.1, per-q medians (n = 10 per q)

| metric (column of the per-run CSV) | q=50 median (min, max) | q=60 median (min, max) |
|---|---|---|
| stage-1 signed relative error (`stage1_rel_err_signed`) | 0.00239 (-0.1027, 0.04482) | -0.003121 (-0.126, 0.113) |
| stage-1 absolute relative error (S1) (`stage1_rel_err_abs`) | 0.03536 (0.009488, 0.1027) | 0.04515 (0.007296, 0.126) |
| stage-2 peak error, signed (d = 0) (`peak_rel_err_signed`) | -0.0625 (-0.1222, -0.02357) | -0.04971 (-0.09174, -0.02097) |
| stage-2 peak error, location-free (`peak_rel_err_locfree`) | -0.06243 (-0.1221, -0.02246) | -0.04943 (-0.09052, -0.02071) |
| RMSE_pos / e2*(0) (`rmse_pos_over_e2star0`) | 0.02007 (0.01502, 0.03966) | 0.01982 (0.01333, 0.02651) |
| tail mean / e2*(0) (`tail_mean_over_e2star0`) | 0.008032 (0.004951, 0.01187) | 0.009393 (0.008325, 0.01254) |
| tail max (effort units) (`tail_max`) | 2.584 (2.218, 4.28) | 2.111 (1.879, 2.507) |
| eta_2 / DW, final tier (`eta2_final`) | 0.001196 (0.0009461, 0.004014) | 0.0008469 (0.0004016, 0.001597) |
| eta_2 / DW, development tier (`eta2_dev`) | 0.001082 (0.0007256, 0.004014) | 0.0008081 (0.0003602, 0.001497) |
| eta_2 dev - final (`eta2_dev_minus_final`) | -8.259e-05 (-0.0002205, 0) | -3.534e-05 (-0.0002002, 0) |
| Gmax_full / DW, final tier (`gmax_final`) | 0.001274 (0.0009461, 0.004014) | 0.0008469 (0.0004121, 0.002256) |
| Gmax_full / DW, development tier (`gmax_dev`) | 0.001149 (0.0007256, 0.004014) | 0.0008409 (0.0003602, 0.002277) |
| Gmax_full dev - final (`gmax_dev_minus_final`) | -6.333e-05 (-0.0002205, 0) | -2.8e-07 (-0.0001, 4.187e-05) |
| EXP_root / DW (final tier) (`exp_root`) | 0.0006756 (0.0002716, 0.002131) | 0.0003993 (0.0001084, 0.002256) |
| dReach / DW (final tier) (`dreach`) | 0.001513 (0.001135, 0.004092) | 0.00109 (0.000447, 0.003335) |
| Deltamax_all / DW (final tier) (`deltamax_all`) | 0.001274 (0.0009461, 0.004014) | 0.0008469 (0.000406, 0.002154) |
| dFull / DW (final tier) (`dfull`) | 0.001513 (0.001135, 0.004092) | 0.00109 (0.000447, 0.003335) |
| sigma_effort at d = 0, stage 2 (`sigma_effort_at_0_t2`) | 2.795 (2.703, 3.446) | 2.989 (2.738, 3.242) |
| e1_hat(0) (`e1_at_0`) | 46.78 (41.87, 48.76) | 38.77 (33.99, 43.28) |

#### Rehearsal v1.1, number of runs passing, per q

| q | n runs | run | G_A | G_F | G_N | S1 | outcome pass |
|---|---|---|---|---|---|---|---|
| 50 | 10 | 10 | 10 | 10 | 10 | 9 | 10 |
| 60 | 10 | 10 | 10 | 10 | 10 | 8 | 10 |

#### Stage-1 error against the 0.05 target and the S1 threshold 0.10

| set | q | n runs | |err| <= 0.05 | |err| <= 0.10 | max |err| |
|---|---|---|---|---|---|
| rehearsal_v1_1 | 50 | 10 | 8 | 9 | 0.1027 |
| confirmation | 50 | 20 | 12 | 20 | 0.08619 |
| rehearsal_v1_1 | 60 | 10 | 6 | 8 | 0.126 |
| confirmation | 60 | 20 | 13 | 18 | 0.153 |

#### Rehearsal v1.1, per run (the paired baseline of this round)

| q | seed | outcome | S1 | stage-1 signed err | peak err (signed) | RMSE_pos/e2*(0) | tail mean/e2*(0) | eta_2/DW (final) | Gmax/DW (final) | Gmax (t*, d*) | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 10501 | pass | yes | -0.009488 | -0.1222 | 0.03966 | 0.007471 | 0.004014 | 0.004014 | (2, -4) | 322.3 |
| 50 | 10502 | pass | yes | 0.0267 | -0.06803 | 0.01974 | 0.01187 | 0.001171 | 0.001171 | (2, -2) | 393.5 |
| 50 | 10503 | pass | yes | -0.08738 | -0.02357 | 0.01997 | 0.006916 | 0.001721 | 0.002122 | (1, 0) | 326.5 |
| 50 | 10504 | pass | yes | 0.01427 | -0.07125 | 0.01909 | 0.0086 | 0.001327 | 0.001327 | (2, -2) | 321.1 |
| 50 | 10505 | pass | yes | -0.03294 | -0.05432 | 0.02018 | 0.008316 | 0.0009605 | 0.0009605 | (2, -18) | 326.6 |
| 50 | 10506 | pass | yes | -0.04206 | -0.09843 | 0.02399 | 0.007748 | 0.002643 | 0.002643 | (2, -4) | 322.8 |
| 50 | 10507 | pass | no | -0.1027 | -0.05776 | 0.0266 | 0.009862 | 0.001063 | 0.002131 | (1, 0) | 321 |
| 50 | 10508 | pass | yes | 0.03777 | -0.06563 | 0.02341 | 0.004951 | 0.001141 | 0.001141 | (2, -2) | 321.8 |
| 50 | 10509 | pass | yes | 0.01567 | -0.03538 | 0.01742 | 0.006381 | 0.001222 | 0.001222 | (2, -14) | 320 |
| 50 | 10510 | pass | yes | 0.04482 | -0.05937 | 0.01502 | 0.009171 | 0.0009461 | 0.0009461 | (2, -2) | 321.1 |
| 60 | 10501 | pass | yes | 0.01206 | -0.04693 | 0.01333 | 0.008325 | 0.0004121 | 0.0004121 | (2, -2) | 320.7 |
| 60 | 10502 | pass | yes | 0.06947 | -0.03193 | 0.0182 | 0.01079 | 0.0005909 | 0.000756 | (1, 0) | 319.1 |
| 60 | 10503 | pass | yes | -0.07487 | -0.07216 | 0.02393 | 0.009227 | 0.001021 | 0.001093 | (1, 0) | 316.6 |
| 60 | 10504 | pass | yes | 0.007296 | -0.04855 | 0.02267 | 0.01028 | 0.0009294 | 0.0009294 | (2, -22) | 323.6 |
| 60 | 10505 | pass | no | -0.126 | -0.07591 | 0.01628 | 0.008791 | 0.00118 | 0.002256 | (1, 0) | 314.8 |
| 60 | 10506 | pass | yes | 0.04916 | -0.04289 | 0.01728 | 0.01254 | 0.0004016 | 0.0004896 | (1, 0) | 316.3 |
| 60 | 10507 | pass | yes | -0.01354 | -0.05151 | 0.01729 | 0.009559 | 0.0005143 | 0.0005143 | (2, -2) | 316.3 |
| 60 | 10508 | pass | yes | -0.01648 | -0.09174 | 0.02454 | 0.01046 | 0.001597 | 0.001597 | (2, -2) | 318.1 |
| 60 | 10509 | pass | yes | -0.04113 | -0.05088 | 0.02145 | 0.008686 | 0.0007645 | 0.0007645 | (2, -24) | 312.8 |
| 60 | 10510 | pass | no | 0.113 | -0.02097 | 0.02651 | 0.008632 | 0.001453 | 0.002115 | (1, 0) | 314.5 |

#### Confirmation, per-q medians (n = 20 per q)

| metric | source file | source column | q=50 median (min, max) | q=60 median (min, max) |
|---|---|---|---|---|
| stage-1 signed relative error | `per_run.csv` | `stage1_rel_err_signed` | 0.00111 (-0.07694, 0.08619) | -0.00175 (-0.1135, 0.153) |
| stage-1 absolute relative error (S1) | `per_run.csv` | `s1` | 0.03776 (0.004058, 0.08619) | 0.04548 (0.001422, 0.153) |
| stage-2 peak error, signed (d = 0) | `reported_metrics.csv` | `A_stage2_peak_rel_err_signed` | -0.06082 (-0.08072, -0.00649) | -0.05948 (-0.1087, -0.02498) |
| stage-2 peak error, location-free | `reported_metrics.csv` | `A_stage2_peak_locfree_rel_err` | -0.06065 (-0.08072, -0.004309) | -0.05928 (-0.1087, -0.02054) |
| RMSE_pos / e2*(0) | `per_run.csv` | `rmse` | 0.0218 (0.01436, 0.03403) | 0.01816 (0.01389, 0.03756) |
| tail mean / e2*(0) | `per_run.csv` | `tail` | 0.00887 (0.004996, 0.01215) | 0.009395 (0.006131, 0.01378) |
| tail max (effort units) | `reported_metrics.csv` | `A_stage2_tail_max` | 2.532 (2.079, 3.511) | 2.115 (1.631, 3.748) |
| eta_2 / DW, final tier | `per_run.csv` | `eta_final` | 0.001365 (0.000804, 0.00329) | 0.0007876 (0.0004322, 0.002318) |
| eta_2 / DW, development tier | `per_run.csv` | `eta_dev` | 0.001199 (0.0006746, 0.003271) | 0.0007222 (0.0004039, 0.002283) |
| eta_2 dev - final | `per_run.csv` | `eta_dev_minus_final` | -7.204e-05 (-0.000261, 0) | -7.055e-05 (-0.000225, 1.11e-16) |
| Gmax_full / DW, final tier | `per_run.csv` | `gmax_final` | 0.001527 (0.0009173, 0.00329) | 0.0009341 (0.0004471, 0.003638) |
| Gmax_full / DW, development tier | `per_run.csv` | `gmax_dev` | 0.00147 (0.0007117, 0.003271) | 0.000837 (0.0004039, 0.003649) |
| Gmax_full dev - final | `per_run.csv` | `gmax_dev_minus_final` | -1.86e-05 (-0.000261, 2.943e-05) | -4.072e-06 (-0.000225, 2.058e-05) |
| EXP_root / DW (final tier) | `reported_metrics.csv` | `B_EXP_root_over_dw` | 0.0006927 (0.0001568, 0.002735) | 0.0004513 (9.253e-05, 0.003638) |
| dReach / DW (final tier) | `reported_metrics.csv` | `B_dReach_over_dw` | 0.001961 (0.0009819, 0.00376) | 0.001194 (0.0005165, 0.005505) |
| Deltamax_all / DW (final tier) | `reported_metrics.csv` | `B_Deltamax_all_over_dw` | 0.001476 (0.000804, 0.00329) | 0.0008897 (0.0004322, 0.003187) |
| dFull / DW (final tier) | `reported_metrics.csv` | `B_dFull_over_dw` | 0.001961 (0.0009819, 0.00376) | 0.001194 (0.0005165, 0.005505) |
| sigma_effort at d = 0, stage 2 | `reported_metrics.csv` | `A_sigma_effort_at_0_t2` | 2.836 (2.638, 3.117) | 2.941 (2.771, 3.538) |
| e1_hat(0) | `reported_metrics.csv` | `B_e1_at_0` | 46.72 (43.08, 50.69) | 38.82 (34.48, 44.84) |

## 7. KL statistics of the baseline (item 7; appendix of the pre-registration)

The canonical `train_history.json` files are present for all 20 rehearsal runs, so the C-R1 reproduction runs are not needed for this item. Command: `python tools/v2/refine_preflight.py kl`. The tool reads `history[*]` (2200 entries per run: 1600 of phase A, 600 of phase B) and raises an error if any entry has a `kl_epochs` array whose length is not 10, a non-finite value, or `kl_final_epoch != kl_epochs[-1]`; it finished without error, so the 20 x 2200 = 44000 entries all have 10 logged epochs. The semantics of `kl_epochs` were read from `git show HEAD:agents/ppo_curriculum_v2.py` (`for _ in range(cfg.epochs)`, one value appended after each epoch, `kl_final_epoch = kl_epochs[-1]`). Files: `results/v2_refine/preflight/kl_final_epoch_distribution.csv`, `kl_target_stopping.csv`, `kl_meta.json`, `kl_appendix.md`.

Check of the tool: the number of stopped updates and the mean number of epochs run of the eight (phase, q, target) rows of `kl_target_stopping.csv` were recomputed with a separate plain-Python loop over the same `train_history.json` files and agree in 8/8 rows (script `p0/kl_check.py` in the session scratchpad; not part of the repository).

Short reading (numbers from the tables below): with target 0.005 the KL of the baseline's own `kl_epochs` exceeds the target at some epoch in 89.9 % / 91.4 % of the phase-A updates (q = 50 / q = 60) and 95.5 % / 95.1 % of the phase-B updates; the updates that a target-KL run would actually shorten (stop epoch 1 to 9, an exceedance only at epoch 10 truncates nothing) are 88.3 % / 90.1 % (A) and 94.8 % / 94.3 % (B), with a median of 3 (A) and 2 (B) epochs run. With target 0.01 the exceedance shares are 51.1 % / 55.0 % (A) and 69.2 % / 70.4 % (B), and the shortened shares 48.9 % / 52.7 % (A) and 67.1 % / 68.6 % (B). The median final-epoch KL is 0.0050 (A) and 0.0033 to 0.0034 (B). These are per-update statistics of the baseline, not predictions for a target-KL run (see the caveat in the appendix).

The appendix, generated (`results/v2_refine/preflight/kl_appendix.md`):

### Appendix: KL statistics of the locked v1.1 baseline (descriptive)

Source: `history[*]` of `train_history.json` of the 20 `rehearsal_v1_1` runs (q in {50, 60} x seeds 10501-10510), canonical worktree `results/v2_T2_locked/rehearsal_v1_1/q*/seed*/`. Every update logs `kl_epochs`, the whole-buffer KL over the policy rows after each of the 10 PPO epochs, mean((r - 1) - log r) with r = pi_new / pi_old, and `kl_final_epoch` = `kl_epochs[-1]`. Phase A has 1600 updates and phase B 600 per run, so each table row pools 16000 (A) or 6000 (B) updates per q. Percentiles use linear interpolation (`numpy.percentile`).

Stopping rule of the target-KL arms (D5): after an epoch, if the KL exceeds the target (strict >) no further epoch runs; the epoch that exceeded the target has run. The number of epochs run is therefore the 1-based index of the first epoch with KL > target, or 10 if none exceeds it. The tables apply that rule to the baseline `kl_epochs`. They are exact for each baseline update taken in isolation (up to and including the first stop the update is identical to the baseline), but counterfactual for the run as a whole: after the first stop the trajectory of a target-KL run differs from the baseline, so later updates would see different buffers and policies. They are not a prediction of the share of stopped updates in a target-KL run. 'Stopped' means that some epoch's KL exceeded the target; an exceedance at epoch 10 truncates nothing (10 epochs run either way), so the column 'fewer than 10 epochs run' (stop epoch 1 to 9) is the share of updates a target-KL run would actually shorten.

Script: `python tools/v2/refine_preflight.py kl`; CSVs: `results/v2_refine/preflight/kl_final_epoch_distribution.csv`, `results/v2_refine/preflight/kl_target_stopping.csv`.

#### (a) Final-epoch KL (`kl_final_epoch`), all updates of the 10 seeds

| phase | q | updates | min | p10 | median | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| A | 50 | 16000 | 4.6e-05 | 0.00185 | 0.00503 | 0.0116 | 0.0237 | 0.0815 | 0.00616 |
| B | 50 | 6000 | 5.36e-09 | 0.000152 | 0.00326 | 0.0123 | 0.0306 | 0.0709 | 0.0052 |
| A | 60 | 16000 | 1.82e-05 | 0.00174 | 0.00497 | 0.0119 | 0.025 | 0.0823 | 0.0062 |
| B | 60 | 6000 | 1.05e-08 | 0.000188 | 0.00341 | 0.0128 | 0.0335 | 0.0795 | 0.00547 |

#### (b) Counterfactual stopping at each target

| phase | q | target | updates | stopped (n) | stopped (share) | of which fewer than 10 epochs run (share) | stopped share per run, min to max | stop epoch median / p90 (stopped only) | epochs run median / p90 / mean |
|---|---|---|---|---|---|---|---|---|---|
| A | 50 | 0.005 | 16000 | 14379 | 89.87% | 88.32% | 84.9% to 95.9% | 3 / 7 | 3 / 10 / 4.04 |
| A | 50 | 0.01 | 16000 | 8173 | 51.08% | 48.88% | 37.1% to 72.8% | 3 / 8 | 10 / 10 / 7.06 |
| B | 50 | 0.005 | 6000 | 5729 | 95.48% | 94.83% | 93.3% to 97.5% | 2 / 5 | 2 / 6 / 2.87 |
| B | 50 | 0.01 | 6000 | 4153 | 69.22% | 67.13% | 60.7% to 79.5% | 2 / 8 | 5 / 10 / 5.46 |
| A | 60 | 0.005 | 16000 | 14624 | 91.40% | 90.13% | 85.9% to 94.7% | 3 / 6 | 3 / 9 / 3.84 |
| A | 60 | 0.01 | 16000 | 8792 | 54.95% | 52.68% | 36.2% to 69.4% | 4 / 8 | 9 / 10 / 6.84 |
| B | 60 | 0.005 | 6000 | 5704 | 95.07% | 94.33% | 91.5% to 97.0% | 2 / 5 | 2 / 6 / 2.89 |
| B | 60 | 0.01 | 6000 | 4222 | 70.37% | 68.55% | 61.3% to 77.0% | 3 / 7 | 4 / 10 / 5.36 |

#### (c) Stopping epoch, share of all updates (%)

Epoch index is 1-based; 'none' = no epoch exceeded the target (10 epochs run).

| phase | q | target | ep 1 | ep 2 | ep 3 | ep 4 | ep 5 | ep 6 | ep 7 | ep 8 | ep 9 | ep 10 | none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 50 | 0.005 | 11.78 | 27.24 | 19.95 | 10.84 | 6.51 | 4.51 | 3.13 | 2.60 | 1.75 | 1.55 | 10.13 |
| A | 50 | 0.01 | 3.76 | 10.68 | 11.16 | 7.33 | 4.47 | 3.53 | 2.98 | 2.67 | 2.29 | 2.21 | 48.92 |
| B | 50 | 0.005 | 35.47 | 26.30 | 13.38 | 7.65 | 4.58 | 2.88 | 1.88 | 1.27 | 1.42 | 0.65 | 4.52 |
| B | 50 | 0.01 | 18.90 | 15.83 | 9.52 | 5.72 | 4.40 | 4.02 | 3.32 | 2.88 | 2.55 | 2.08 | 30.78 |
| A | 60 | 0.005 | 14.38 | 27.20 | 19.26 | 11.05 | 7.00 | 4.58 | 2.86 | 2.02 | 1.79 | 1.27 | 8.60 |
| A | 60 | 0.01 | 5.01 | 11.63 | 10.42 | 7.47 | 4.72 | 4.44 | 3.54 | 2.72 | 2.73 | 2.27 | 45.05 |
| B | 60 | 0.005 | 34.68 | 26.68 | 13.87 | 7.42 | 4.27 | 3.13 | 1.97 | 1.33 | 0.98 | 0.73 | 4.93 |
| B | 60 | 0.01 | 18.67 | 16.30 | 10.02 | 6.12 | 4.73 | 4.27 | 3.27 | 2.62 | 2.57 | 1.82 | 29.63 |

#### (d) Supplementary: phase A split by learning-rate window

Local 1-1200 has constant LR 3e-4; local 1201-1600 is the linear decay 3e-4 to 3e-5, the window the phase-A continuation arms run in.

| phase | q | updates | min | p10 | median | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| A local 1-1200 | 50 | 12000 | 4.6e-05 | 0.00198 | 0.00524 | 0.012 | 0.0243 | 0.0815 | 0.00641 |
| A local 1201-1600 | 50 | 4000 | 0.000103 | 0.00153 | 0.00443 | 0.0103 | 0.019 | 0.0542 | 0.00539 |
| A local 1-1200 | 60 | 12000 | 1.82e-05 | 0.00187 | 0.00518 | 0.0122 | 0.0254 | 0.0704 | 0.00643 |
| A local 1201-1600 | 60 | 4000 | 6.06e-05 | 0.00139 | 0.00439 | 0.0107 | 0.0215 | 0.0823 | 0.00549 |

| phase | q | target | updates | stopped (n) | stopped (share) | of which fewer than 10 epochs run (share) | stopped share per run, min to max | stop epoch median / p90 (stopped only) | epochs run median / p90 / mean |
|---|---|---|---|---|---|---|---|---|---|
| A local 1-1200 | 50 | 0.005 | 12000 | 11125 | 92.71% | 91.42% | 87.7% to 97.9% | 3 / 6 | 3 / 9 / 3.72 |
| A local 1-1200 | 50 | 0.01 | 12000 | 6650 | 55.42% | 53.08% | 40.3% to 76.8% | 3 / 8 | 8 / 10 / 6.77 |
| A local 1201-1600 | 50 | 0.005 | 4000 | 3254 | 81.35% | 79.03% | 75.8% to 90.0% | 3 / 7 | 4 / 10 / 5.01 |
| A local 1201-1600 | 50 | 0.01 | 4000 | 1523 | 38.07% | 36.25% | 27.5% to 60.5% | 4 / 9 | 10 / 10 / 7.91 |
| A local 1-1200 | 60 | 0.005 | 12000 | 11304 | 94.20% | 93.17% | 88.3% to 97.2% | 3 / 6 | 3 / 8 / 3.52 |
| A local 1-1200 | 60 | 0.01 | 12000 | 7155 | 59.62% | 57.27% | 40.3% to 74.1% | 3 / 8 | 7 / 10 / 6.53 |
| A local 1201-1600 | 60 | 0.005 | 4000 | 3320 | 83.00% | 81.00% | 74.5% to 87.8% | 3 / 7 | 4 / 10 / 4.8 |
| A local 1201-1600 | 60 | 0.01 | 4000 | 1637 | 40.92% | 38.90% | 24.0% to 55.2% | 4 / 9 | 10 / 10 / 7.76 |

| phase | q | target | ep 1 | ep 2 | ep 3 | ep 4 | ep 5 | ep 6 | ep 7 | ep 8 | ep 9 | ep 10 | none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A local 1-1200 | 50 | 0.005 | 12.95 | 29.97 | 21.03 | 10.24 | 6.12 | 4.34 | 2.84 | 2.42 | 1.50 | 1.29 | 7.29 |
| A local 1-1200 | 50 | 0.01 | 4.25 | 11.65 | 12.64 | 8.00 | 4.61 | 3.68 | 3.14 | 2.81 | 2.30 | 2.33 | 44.58 |
| A local 1201-1600 | 50 | 0.005 | 8.25 | 19.07 | 16.70 | 12.65 | 7.67 | 5.03 | 4.00 | 3.15 | 2.50 | 2.33 | 18.65 |
| A local 1201-1600 | 50 | 0.01 | 2.30 | 7.75 | 6.72 | 5.33 | 4.05 | 3.08 | 2.48 | 2.27 | 2.27 | 1.82 | 61.92 |
| A local 1-1200 | 60 | 0.005 | 16.25 | 29.27 | 19.89 | 10.74 | 6.83 | 4.19 | 2.55 | 1.84 | 1.62 | 1.02 | 5.80 |
| A local 1-1200 | 60 | 0.01 | 5.83 | 12.57 | 11.59 | 8.02 | 5.08 | 4.84 | 3.73 | 2.82 | 2.79 | 2.36 | 40.38 |
| A local 1201-1600 | 60 | 0.005 | 8.78 | 21.00 | 17.35 | 11.97 | 7.53 | 5.72 | 3.77 | 2.55 | 2.33 | 2.00 | 17.00 |
| A local 1201-1600 | 60 | 0.01 | 2.52 | 8.80 | 6.90 | 5.83 | 3.67 | 3.23 | 2.98 | 2.42 | 2.55 | 2.02 | 59.08 |

## 8. Deviations, remarks and reproduction

Deviations and remarks:

1. **Local `main`** is not at `f02a256` and its checkout is not clean (section 1). The branch was taken from `f02a256` = `origin/main` = tag `t2-v2-main`. The statement about the dirty checkout was not re-measured in the fix pass.
2. **Item 3** (suite and C7) is reported from the records of the earlier run in this session (`results/v2_refine/preflight_pytest.txt`, `results/v2_refine/c7/branchpoint_f02a256.compare.txt`); this pass did not repeat them.
3. **Extras beyond the spec**, all labelled in the generated tables: the phase-A split into local 1-1200 and 1201-1600 in the KL appendix (the second window is the one the phase-A continuation arms run in); the per-run baseline table; the counts of the 0.05 target and the S1 threshold (`baseline_s1_target_counts.csv`); the seed-source listing for 20501-20520 (information only; the confirmation block is not used in this round).
4. **Seed inventory** is the stock tool run once per block plus one in-process scan for the per-file listing (three scans in total). The `seeds` outputs were produced at 18:05 to 18:09 UTC; `tools/v2/refine_preflight.py` was edited afterwards only in `cmd_parents`, `main`, `_crosscheck` (docstring) and `cmd_baseline`, so the `seeds` code path is the one that ran. `parents`, `baseline` and `kl` were re-run with the final file.
5. **Cross-check of the baseline values** is a transcription check only (section 6).
6. No result, checkpoint or protocol file was written or changed by this pre-flight; the only outputs are the files listed below. The first draft of `tools/v2/refine_preflight.py` was written into the wrong checkout (`t2-accuracy-refinement-r1-32d472/tools/v2/`, untracked) before the file was copied here. That stray copy was left untouched, because it lies outside this worktree; it is not used by any command in this report.

Reproduce (from the worktree root, single-threaded; `seeds` takes about 3 minutes and was run in tmux):

```bash
cd /home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/refine_preflight.py parents
/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/refine_preflight.py baseline
/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/refine_preflight.py kl
tmux new-session -d -s p0_seeds "/home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/refine_preflight.py seeds"
```

Files written by the pre-flight:

- `tools/v2/refine_preflight.py`
- `results/v2_refine/parents.csv`
- `results/v2_refine/preflight/parents_table.md`
- `results/v2_refine/preflight/baseline_rehearsal_v1_1_per_run.csv`, `baseline_rehearsal_v1_1_per_q_median.csv`, `baseline_rehearsal_v1_1_gate_counts.csv`, `baseline_rehearsal_v1_1_crosscheck.json`, `baseline_confirmation_medians.csv`, `baseline_confirmation_medians_check.json`, `baseline_s1_target_counts.csv`, `baseline_tables.md`
- `results/v2_refine/preflight/kl_final_epoch_distribution.csv`, `kl_target_stopping.csv`, `kl_meta.json`, `kl_appendix.md`
- `results/v2_refine/preflight/seed_inventory_10501_10510.{csv,out}`, `seed_inventory_30501_30520.{csv,out}`, `seed_inventory_summary.json`, `seed_sources_{10501_10510,20501_20520,30501_30520}.csv`, `seed_source_categories_{10501_10510,20501_20520,30501_30520}.csv`
- `reports/v2/refine/00_preflight.md` (this file)
