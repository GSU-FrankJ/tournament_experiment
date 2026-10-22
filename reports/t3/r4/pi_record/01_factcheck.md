# T3-R4: independent fact-check of the P3 reports

Round 1, by one independent read-only agent that wrote none of the reports, against commit `87ae90c8` (documents checked: `reports/t3/r4/03_confirmation.md`, `summary.md`, `01_lock.md`, `02_checks.md`, the T3-R4 row, files paragraph and tools row of `reports/t3/README.md`, and the T3-R4 section of `docs/STATE.md`). Its report is reproduced verbatim below; my dispositions follow it. The Chinese handoff was written after this round and is covered by round 2 (end of this file).

---

## (1) Ledger

All values were recomputed from raw run files (`status.json`, `manifest.json`, `gates.json`, `retry.json`, `ms_checks_stage3.csv`, `ms_updates.csv`, `attempt1/*`, `freeze_stage*_*.npz`, `eval/per_run.csv`). Table cells in the documents were parsed from the markdown text itself, not from the generator CSVs. Scratch scripts are in `.../scratchpad/factcheck_r4/`. Nothing in the repository was modified; `git status` shows no tracked changes and no new files outside `results/`.

| # | Document and section | Claim | How checked | Verdict |
|---|---|---|---|---|
| 1 | 03 §3, per-seed tables (q50, q60) | 80 rows x 8 cells: verdict, eta_3, rmse_pos, tail_mean, Gmax_full, S3, stretch nodes, wall | Pass/fail redone with the gate rule from `per_run.csv` (inclusive, float64) plus status, exit code, rng, valid and dev_valid. S3 from `ms_updates.csv` (local 1201-1600). Stretch from `freeze_stage2_final.npz`. Wall from `status.json`. | CONFIRMED. 640 cells, 0 mismatches. Six S3 cells are shown at 3 significant digits (e.g. "0.00039" for 0.0003897). Rounding only. |
| 2 | 03 §1 checks table (16 checks x 80) | All pass, 0 fail, 0 not evaluated | `launch_checks.json` read. Independently recomputed on 80 runs: exit/state, commit, clean tree, global RNG, config sha vs plan, config differs from the reference in the six fields only, budgets (3,800 rows, 1600/1600/600; fired totals 5,400 / 2,764,800 / 108,000 / 216), lr at every update (attempt 1 too), 152 exports, 64/64/24 checks, 64 attempt-1 exports, init digests incl. attempt 2, snapshot refresh (`state_end_stage1.pt` log), trigger vs `retry.json`. | CONFIRMED |
| 3 | 03 §1, C-REPLAY | The evaluator reproduces freeze arrays bit for bit | Ran `te.replay_mismatches` on all 80 runs. Also recomputed eta_3 and Gmax_full directly from the freeze arrays, both tiers. | CONFIRMED. 0 mismatches. Array-derived values equal `per_run.csv` exactly (max diff 0.0). |
| 4 | 03 §1, summary counts | Fired 5, would fire 5, errors 0, ledger present, 0 events | `launch_checks.json`, `infra_events.json` | CONFIRMED |
| 5 | 03 §1, launch facts | Launch 07:11:46, end 07:34:42 UTC; commit `141b5189`; machine record 21 s earlier (nproc 64, load 8.9/11.5/12.9, 789 GB, empty porcelain outside results) | `launch_start/end/commit/machine.txt`, launch JSON, `status.json` times (last end 07:34:41, max 40 concurrent) | CONFIRMED |
| 6 | 03 §1, "nothing written outside results/ between launch and end of wave" | `find -newermt` outside `results/` for 07:11:46-07:35:00 | CONFIRMED. No files. The first non-results writes are the report files after 07:34:42. |
| 7 | 03 §1, guards of prompt 2.1 | Launcher passed all guards | Launch JSON: head = code_commit, both diff stats empty, absolute root, porcelain entries all under `results/`. LOCK record was last (see 14). | CONFIRMED |
| 8 | 03 §1, post-launch script "ran to its end"; eval re-runs empty | `post_launch_stdout.txt` ends "done 07:35:38Z"; `eval_reruns.json` is `[]` | CONFIRMED |
| 9 | 03 §2, lock facts | JSON/MD SHA-256, lock commit `98355d8a`, record `141b5189`, 33 listed files, T3-1.0 unchanged | `sha256sum`; `git rev-parse`; `tool_sha256` has 33 entries and every SHA matches the file at HEAD; draft = protocols byte-identical; v1_0 sha matches `builds_on`; LOCK diff is +11 lines, 0 deleted. | CONFIRMED |
| 10 | 03 §2, code claims | `git diff --stat b565f0e8 HEAD -- agents envs utils run tools/ms tools/v2` is empty; trigger rule | Ran the diff (also `b565f0e8..df2365d9`). Read `run/ms_retry.py:trigger`. | CONFIRMED |
| 11 | 03 §2, rule, OC, paper wording | Rule 6/80 and 4/40; OC 0.982/0.865/0.271/0.030; wording verbatim | OC recomputed with `scipy.stats.binom`: 0.999893, 0.981819, 0.864763, 0.270786, 0.029886. Wording identical in the prompt, lock JSON, lock MD, 01, 03 and summary. Rule matches prereg §9/D6. | CONFIRMED |
| 12 | 03 §4 verdict table; Clopper-Pearson intervals | 40/40 [0.912,1.000]; 39/40 [0.868,0.999]; 79/80 [0.932,1.000]; failure-rate interval [0.000,0.068] | scipy Beta quantiles | CONFIRMED |
| 13 | 03 §4 blind recomputation | verdict pass, 0+1, agrees, differences [] | `blind.json`. The blind tool imports only scipy. | CONFIRMED |
| 14 | 03 §4 values table | Worst Gmax_full 0.00483 (s80510) at q=50 and 0.05864 (s80502) at q=60; 3 x 6 medians | Recomputed medians: 0.002089/0.02735/0.004554/-0.07421/0.002124/0.000462 for q50 and the corresponding q60 and both rows. Both-row EXP_root median is 0.0003725 (shown as 0.000372). | CONFIRMED |
| 15 | 03 §4 margins text | eta_3 0.004897 = 98 % (q60 s80528); rmse 0.0454 = 91 % (s80536); 6 and 8 runs above 80 %; tail < 50 %, Gmax < 49 %, dev-final < 24 % | Recomputed over the 79 passing runs: eta 97.9 %, rmse 90.8 %, counts 6 and 8; tail max 47.1 %; Gmax max 48.97 %; dev-final max 24.02 %. | CORRECTION NEEDED (E7, trivial) |
| 16 | 03 §5 trigger table (5 rows x 11 cells) | Delta, s, C, reason, S3 first and frozen, attempt-2 Delta, eta_3, Gmax | Attempt-1 check at local 1600; S3 from `attempt1/ms_updates_stage3.csv`; attempt-2 Delta from `ms_checks_stage3.csv` | CONFIRMED. 0 mismatches. |
| 17 | 03 §5 text | Fire rate 5/80, interval [0.0206,0.1399]; all q50 (5/40); Delta 0.0052-0.0094; "four of five in regime, S3 0.46-0.78"; 80535 0.459 to 0.267; 80532 S3 0.060 to 0.78, eta 0.004663 = 93 % | Recomputed. Fired iff Delta > 0.005 holds for all 80. `retry.json` agrees with the check rows for all 5. | CONFIRMED |
| 18 | 03 §5, first-attempt outcome | 74/80 [0.844,0.972]; q50 35/40; q60 39/40; "the bw-equivalent pass rate" | Recomputed. Extra test: evaluated each fired run's attempt-1 `u01600` on the final tier. eta_3 = 0.00640, 0.00745, 0.00517, 0.00937, 0.00672, so all five fail G-A. Unfired runs are bit-identical to `bw`. | CONFIRMED. The convention is exact at stage 3. |
| 19 | 03 §6 failure row and text | s80502 q60: Gmax_full 0.0586406 at stage 2, d=-14; dev-final 0.0010748; stretch 5 nodes, longest [-22,-14]; S3 0.536 / 0.536; F(u01600)=54.04 vs threshold 26.78; Delta@1600=0.0008; class K | `per_run.csv` fields (stage 2, d=-14, dev d=-12); stretch d = -22..-14; F recomputed from `freeze_stage3_final.npz` = 54.0396; threshold = half the median F_final of passing q60 runs in `dg2_runs.csv` = 26.7806; trigger Delta 0.000809; K matches T3-R3 D2. | CONFIRMED. Minor wording points E8 and O1. |
| 20 | 03 §6 last 16 exports "-" | No failing stage-3 quantity, so range empty | `failures.csv`, `last16_summary.csv` | CONFIRMED (but see O3) |
| 21 | 03 §6, descriptive facts | 28/80 in regime, 27 pass; 3 stretches (14, 8, 5); 2 pass with Gmax 0.0045/0.0046 | Recomputed (q50: 16, q60: 12 in regime) | CONFIRMED |
| 22 | 03 §6, T3-R3 comparison | "Second failure of this kind in 120 bwA runs; T3-R3 q60 s70511, 28 nodes, unreached by A" | `results/t3_r3/dev/analysis/per_run.csv`: bwA fail only s70511, stretch 28, S3 0.52, trig Delta 0.00396 | CONFIRMED |
| 23 | 03 §7 cost table (3 rows x 13 cols) | Updates, rows, steps, checks, ratios, wall totals, means, medians, fired and unfired means | Recomputed: 4000/2.05e6/8e4/160/1.053; 3800/.../1; 3900/2e6/7.8e4/156/1.026; walls 2.449e4, 2.332e4, 4.781e4; 612.3, 582.9, 597.6; 590.9, 591.5, 591.3; 801.2, 585.3, 584.1. | CONFIRMED |
| 24 | 03 §7 text | 1.421; 1.037 for 7/80; interval [0.036,0.172]; fired 766-853 s; +217 s or 37 %; "36 %" and "~235 s" from T3-R3 | 5400/3800; 7/80 interval; fired range; 217.2 s = 37.2 %. Prereg §4 gives 235 s, and the 36 % is in `05_development.md`. | CONFIRMED |
| 25 | 03 §7, "47,810 s (13.3 CPU-hours)" | Summed per-run wall time | Wall sum 47,810.8 s = 13.28 h. Recorded process CPU (`phase_timing` + attempt 1) = 12.10 h. | CORRECTION NEEDED (E5) |
| 26 | 03 §7, "load average 8.9 before and up to about 50 during" | Launch JSON and machine record | Recorded: start 8.64/11.24/12.77; end 14.8/33.3/35.3 (1/5/15 min). Nothing records "about 50". | CORRECTION NEEDED (E6) |
| 27 | 03 §8, comparisons table (24 rows) and bullets | Pass counts, rates, intervals for confirmation, T3-R3 bwA, pooled, T3-R1/R2/R3 bw, pooled bw, first attempt; 118/120; 107/120; 111/120 | Recomputed from `results/t3_r1/eval/per_run.csv` (gate rule applied), `t3_r2/.../per_run_table.csv`, `t3_r3/dev/analysis/per_run.csv` | CONFIRMED. 24/24 rows. |
| 28 | 03 §9 and summary | 7 fires of 60 at q50 over dev + confirmation (2 + 5); none at q60 except diagnostic s60518; 2 of 60 q60 failures; 2.6 % and 3.7 % | T3-R3 bwA fires: q50 2 (s70508, s70516), q60 0; diagnostic s60518 fired | CONFIRMED |
| 29 | 03 §10 and tracked/untracked file lists | Files exist; tracked set | `git ls-files` for a fired run's dir (20 files incl. `retry.json`, attempt1 records); untracked arrays, weights, train histories; all listed analysis files exist | CONFIRMED |
| 30 | summary.md | Verdict table, trigger, cost, comparisons, "pushed before the launch", "22.9 min", "+37 %", "~1.37x wall", "18 + 9 findings", "109 tests", suite totals | Recomputed. `git reflog origin/t3-r4`: pushed 07:10:33 to `141b5189` (before 07:11:46). 22 min 56 s. Wall ratio 1.372. Review counts and suite totals checked under 36 and 38. | CONFIRMED, except "13.3 CPU-hours" (E5) |
| 31 | 01_lock §1 and §2 | Settings table; lock files; OC table (5 values to 1e-6); failure counting vs lock section 5 and prompt | Compared with lock JSON and MD and the prompt | CONFIRMED. Wording issues E9 and E10. |
| 32 | 01_lock §4 plan table | 80 x 3 digests (16 hex) and plan CSV SHA-256 `3e3d4456...` | Parsed against `confirmation_plan.csv`: 240/240 cells; file SHA matches; on-disk `run_config.json` SHA = plan for all 80 | CONFIRMED |
| 33 | 01_lock §5 and §6 | Config differences vs T3-R3 bwA of the same q; seed inventory | Differences are exactly the six fields at both q. Inventory: 231,947 json, 87,424 csv, 437 values, 0 collisions, 0 hits. | CONFIRMED |
| 34 | 01_lock §7 and 02 §4, C-REPRO table | Commit `631c19f1`, exit 0, 162 npz, same file set, states equal, JSON/manifest/train-history diffs empty, attempt1/retry identical, walls 575.3 and 434.8 | `compare_*.json` and `.txt`; manifests (commit `631c19f1`, clean). First pass: `5af02fd7`, tree dirty, ok. | CONFIRMED. Path point E4. |
| 35 | 01_lock §8 and 02 §3, rehearsal | 80 symlinked runs, 4 fired, 78/80 pass, 12 checks pass, 4 fail by construction, 0 eval re-runs | Rehearsal `launch_checks.json`, `headline.json`, `eval_reruns.json` | CONFIRMED |
| 36 | 02 §1, test counts | 34+11+35+11+3+5+10 = 109 | `pytest --collect-only -q` per file | CONFIRMED |
| 37 | 02 §1, file-by-file test descriptions | e.g. "3,000 random rows", "eight more cases" | Test source and collected ids | CORRECTION NEEDED (E2). Other descriptions match test names. |
| 38 | 02 §2, review rounds | 18 round-1 findings (7 should-fix, 6 minor, 5 note); 9 new in round 2; "27"; dispositions vs code | Counted rows. Spot-checked code for findings 1, 6, 7, 11, 12, 13. R2-4 (162 = 152+6+2+2) matches. R2-5 "15 perturbations" matches the test dict (15; 2 harmless). | CONFIRMED. Internally consistent. |
| 39 | 02 §2, process detail | 56 mutants, 13 sandbox launches, 37 mutants, Monte Carlo of the OC | No record in the repository | NOT DATA-CHECKABLE |
| 40 | 02 header, commit roles | `7cc101b0` .. `631c19f1` roles; "no existing file changed" | `git show --stat` per commit: roles match. The "only new files" claim fails at HEAD (E3). `5af02fd7` touches only `reports/`, `631c19f1` only 3 draft/plan files. | CORRECTION NEEDED (E3) for the diff claim; roles CONFIRMED |
| 41 | 02 §5, suite totals | 1,928 passed, 6 skipped, 2 xfailed, 2 failed; per-shard table; 1,819 + 109 | Parsed `shard*.txt`: sums 1928/6/2/2, all 12 cells match. T3-R3: 1,819/6/2/2. `pytest_commit.txt` = `631c19f1`. | CONFIRMED (table). Shard attribution wrong (E1). |
| 42 | README T3-R4 row, tools row; STATE section | Numbers and hashes identical to 03 and summary | Compared item by item. README tests row (see E12). | CONFIRMED, except E11 and E12 |
| 43 | 03 §1, process statements | tmux session `t3r4_conf`; "tool environment refuses a launch script that calls git"; "nothing tuned"; "no implementation error (E)"; suite/repro "tree clean outside results" during pytest | Only the script comment and `launch_confirmation.sh` exist. | NOT DATA-CHECKABLE |

## (2) Errors and corrections

**E1. `02_checks.md` §5: the two known failures are in swapped shards. Wrong fact.**
- Now: "`tests/test_registry_canonicalization.py::test_registry_canonicalization` (shard 9) and `tests/test_ms_r4_launcher.py::test_the_existing_launcher_is_unchanged` (shard 10; ...".
- Correct: the launcher test is in shard 9; `test_registry_canonicalization` is in shard 10.
- Evidence: `results/t3_r4/tests/shard9.txt` has "FAILED tests/test_ms_r4_launcher.py::...", and `shard10.txt` has "FAILED tests/test_registry_canonicalization.py::...". `grep -l` on `shard*.files` agrees. The table (shard 9: 193 passed, 1 failed; shard 10: 143 passed, 1 failed) is correct.

**E2. `02_checks.md` §1, `test_t3r4_analysis.py` row. Wrong number.**
- Now: "the rule at 3 + 3 (pass), 4 + 3 (fail), 5 + 0 (fail), 4 + 2 (pass) and eight more cases".
- Correct: "six more cases".
- Evidence: `CASES` in `tests/test_t3r4_analysis.py` and `--collect-only` list 10 parametrised cases (the four named, plus 0+0, 4+4, 0+5, a repeated 4+2, 1+6, 6+0).

**E3. `02_checks.md` header. Imprecise at HEAD.**
- Now: "`git diff --name-status df2365d9 HEAD -- agents envs utils run protocols tools tests` lists only new files, nothing else".
- At `87ae90c8` the command prints 18 `A` and 1 `M` (`protocols/LOCK`, 11 lines appended).
- Correct: "...lists only new files, apart from the LOCK record appended at the lock (`protocols/LOCK`, +11 lines, prompt 2.6)". The sentence was true at `631c19f1`/`31639d6c`, before the lock.

**E4. `02_checks.md` §4. Path imprecision.**
- Now: "outputs in `results/t3_r4/repro/` (... `repro_stdout.txt`)".
- Correct: `repro_stdout.txt` is `results/t3_r4/repro_stdout.txt`. `results/t3_r4/repro/` holds only the compare files and `q50`/`q60`.

**E5. `03_confirmation.md` §7 and `summary.md` (Cost bullet). Imprecise.**
- Now: "47,810 s for the wave (13.3 CPU-hours)"; summary "(13.3 CPU-hours)".
- 47,810.8 s is the sum of per-run `total_wall_sec` (13.28 h). Process CPU time summed from `ms_run_summary.json` `phase_timing` plus attempt 1 in `retry.json` is 12.10 h.
- Correct: "13.3 hours of summed run wall time (12.1 h of recorded process CPU time)", or drop "CPU". The wave itself took 22.9 min.

**E6. `03_confirmation.md` §7. Unsupported number.**
- Now: "load average 8.9 before and up to about 50 during the wave".
- The records give 8.64/11.24/12.77 (launcher start, `launch_20261022_071148.json`; 8.89/11.47/12.87 in `launch_machine.txt`) and 14.8/33.3/35.3 at the end (1/5/15-min). No record of "about 50" exists. Max concurrency was 40 runs.
- Correct: state only the recorded values (e.g. "1-min load 8.6 at the start, 15-min load 35.3 at the end"), or mark "about 50" as an estimate (40 workers + about 9 background).

**E7. `03_confirmation.md` §4. Trivially imprecise.**
- Now: "the development-final differences stay below 50 %, 49 %, and 24 % of theirs".
- Among passing runs the largest dev-final difference is 0.00024016 (24.02 % of 0.001, q50 s80503), so it is not below 24 %. tail_mean max is 47.1 % and Gmax_full max is 48.97 %.
- Correct: "at most 47 %, 49 % and 24.1 %".

**E8. `03_confirmation.md` §6. Trivially imprecise.**
- Now: "(limit 0.001, by 7 %)".
- Correct: "by 7.5 %". 0.0010748 / 0.001 = 1.0748.

**E9. `01_lock.md` §1 (also the lock test description in `02_checks.md` §1). Misleading wording.**
- Now: the lock files "hold no timestamp, no commit hash and no absolute path".
- Both lock files contain the abbreviated commit `b565f0e8` (`settings.code`, and MD line 16). The test `test_the_files_hold_no_timestamp_no_commit_and_no_absolute_path` only excludes 40-hex strings.
- Correct: "no timestamp, no full commit hash (only the base commit `b565f0e8` of the frozen-code diff) and no absolute path".

**E10. `01_lock.md` §1 table, row "records". Imprecise.**
- Now: "`game.T` = 3".
- In the lock JSON, `records.{50,60}.game.T` = 2 and `records.*.T` = 2. T = 3 is applied by the config builder (`run_config.record.game.T` = 3). The same wording is in the lock MD, so it is inherited.
- Suggest: "the v2.0 record of each q (game.T = 2 there), with game.T = 3 set in the run configs".

**E11. README T3-R4 row, STATE top section, `summary.md` ("For the PI"). Premature statements at commit `87ae90c8`.**
- Now: README row "P0-P3 done" listing `handoff.md` and `pi_record/01_factcheck.md`; STATE "Done: ... P3: ... `handoff.md` (Chinese), the independent fact-check (`pi_record/01_factcheck.md`) and the push of `t3-r4` (fast-forward ...)"; summary "`reports/t3/r4/handoff.md` is the Chinese note".
- At this commit `reports/t3/r4/handoff.md` and `reports/t3/r4/pi_record/01_factcheck.md` do not exist, and `origin/t3-r4` is still `141b5189` (reflog: pushed 07:10:33).
- Correct: make these true before the final push, or reword them as pending. The same files are named in the README "Files of the round" paragraph.

**E12. `reports/t3/README.md` tests row. Omission.**
- The row "`tests/t3_helpers.py`, `tests/test_t3_*.py`, `tests/test_t3r2_*.py`, `tests/test_t3r3_*.py` | the tests of all of the above" does not list `tests/test_t3r4_*.py`, although the tools row now claims T3-R4 tools.

## (3) Overstated relative to the data, or omitted

- **O1. 03 §6, "A stage-2 near-zero stretch of 5 in-band nodes ... explains it".** Gmax_full at stage 2 peaks at d = -14, the last node of the stretch [-22, -14], so the location coincides. No test separates cause from coincidence. "Is located at" or "coincides with" would be accurate. This follows the repo rule against elevating a plausible factor to root cause.
- **O2. summary "State the limits: the restart repairs failures that show at the stage-3 check".** The data show 5 of 5 fired first attempts failing G-A on the final tier and 5 of 5 second attempts passing. The second attempt is a fresh draw, so this equals a retry of an independent run. With the block's base rate (about 94-97 % pass), 5 of 5 would also occur by chance with probability about 0.73-0.86. There is no control arm in this block. 03 §9 already says the mechanism is not tested; the summary wording is stronger.
- **O3. Failure run s80502 omits a related fact.** Among its last 16 stage-3 exports, `rmse_pos` is above the 0.05 limit at one export (max 0.0505, `last16_summary.csv`), and eta_3 reaches 0.0042. The final export passes, so "stage-3 quantities pass" is true, but the last-iterate variability of the same run is not mentioned.
- **O4. Not reported but in the checks file.** The start-sampler diagnostic has `z_flags` = 1 (|z| > 3) in 1 of 5 cells of q50 s80536; `starts_ok` is true for all 80. A non-fired q60 run (s80528) had trigger Delta = 0.004897, 98 % of the 0.005 trigger limit, and it is also the run closest to the G-A eta limit. Neither is wrong to omit, but both bear on how close the "invisible" margins are.
- **O5. 03 §9, "The rule is met with a wide margin".** This is one block of 80 runs with one failure against a limit of six. The document says so in the next sentence. The first-attempt count (74/80) would still have met the rule (6 failures against a limit of 6) at exactly the boundary.

---

# Dispositions of round 1 (mine)

No result number was wrong. Everything below was corrected in the templates (`report_scripts/*.template.md`, `lock_tables.py`) or the hand-written files, and the reports regenerated, before the push. The tools and the lock were not touched (they are locked; the corrections concern wording).

| item | disposition |
|---|---|
| E1 | `02_checks.md` section 5 now names the shard of each known failure correctly (the launcher test in shard 9, `test_registry_canonicalization` in shard 10). |
| E2 | 'six more cases'. |
| E3 | The header sentence now says 'only new files, apart from the one record appended to `protocols/LOCK` at the lock (+11 lines, prompt 2.6)'. |
| E4 | `results/t3_r4/repro_stdout.txt` is named with its path. |
| E5 | '47,810 s of summed run wall time (13.3 hours; 12.1 hours of recorded process CPU time), 22.9 minutes at 40 workers' in `03_confirmation.md`; 'hours of summed run wall time' in `summary.md`. The CPU-time figure is the checker's (it summed `phase_timing` and attempt 1); I did not recompute it. |
| E6 | The load sentence now gives the recorded values: 8.6 / 11.2 / 12.8 at the start and 14.8 / 33.3 / 35.3 at the end (`launch_20261022_071148.json`), on a machine shared with other jobs. |
| E7, E8 | 'at most 47 %, 49 % and 24.1 %'; 'by 7.5 %'. |
| E9 | 'no full commit hash (only the abbreviated base commit `b565f0e8` of the frozen-code diff)' in `01_lock.md`, and in the description of the test in `02_checks.md`. The docstring of the (locked) `t3r4_make_lock.py` keeps its older wording. |
| E10 | The settings table says 'the record itself has T = 2, the run configs set `game.T` = 3' (I confirmed `records["50"]["game"]["T"] == 2` in the lock JSON and `record.game.T == 3` in the run configs). The lock MD, which is locked, keeps its wording; `01_lock.md` says so through the table. |
| E11 | Made true: `handoff.md` and this file are written and the push follows the commit that holds them. |
| E12 | `tests/test_t3r4_*.py` added to the tests row of the README. |
| O1 | 'coincides with it: the maximal gain is at d = -14, the end of the stretch'. |
| O2 | `03_confirmation.md` section 9 and `summary.md` now say that the block has no control arm, a second attempt is a fresh draw, and five of five second attempts passing is not by itself a measure of an effect beyond that of a second draw. The checker's chance figures (0.73-0.86) are in the ledger; I did not recompute them and do not quote them. |
| O3 | Section 6 now reports the last-iterate variability of seed 80502 (`rmse_pos` up to 0.0505, `eta_3` up to 0.0042 over its last 16 stage-3 exports; I checked `last16_summary.csv`). |
| O4 | Section 1 reports the one start-count flag (q = 50 seed 80536, one cell); section 5 names the three non-fired runs within 5 % of the trigger limit (`Delta` 0.004897, 0.004825, 0.004758; I checked `per_run.csv`) and says all three pass. |
| O5 | The checker's reading is not right for the per-q limit: counted as first-attempt failures the block has 6 failures, 5 at q = 50 and 1 at q = 60 (`per_run.csv`), so it would meet the total limit of 6 exactly and exceed the limit of 4 at q = 50. Section 8 now says exactly that (as a description under the convention, not a test). The five fired runs' attempt-1 final-tier `eta_3` values (all above 0.005) are cited to this ledger. |

# Round 2

Same agent, resumed, at commit `dddfdd6e` (the corrections of round 1) with the new `handoff.md` read from disk. Its report verbatim:

---

## Round 2 ledger

I checked commit `dddfdd6e`, with `handoff.md` read from disk (untracked). Numbers were recomputed from raw run files, not from the generated CSVs. The tree is unchanged by me: `git diff HEAD` is empty for `reports/`, `docs/`, `protocols/` and `tools/`, and the only untracked file outside `results/` is `handoff.md`.

### (a) Round-1 corrections E1-E12 and O1-O5, as they now stand in the regenerated files

| Item | Where | Verdict |
|---|---|---|
| E1 | `02_checks` §5: launcher test in shard 9, `test_registry_canonicalization` in shard 10 | FIXED. Matches `shard9.txt` and `shard10.txt`. |
| E2 | `02_checks` §1 test row | FIXED. "six more cases" matches the 10 parametrised cases. |
| E3 | `02_checks` header | FIXED. At HEAD the command gives 18 A and 1 M (`protocols/LOCK`, +11 lines), as the text now says. |
| E4 | `02_checks` §4 | FIXED. `results/t3_r4/repro_stdout.txt` is named with its path. |
| E5 | `03` §7, `summary` | FIXED, and the new wording is correct. 47,810.85 s of summed `total_wall_sec` is 13.28 h. Summed `phase_timing` process CPU plus attempt-1 CPU is 12.097 h. The wave took 22 min 56 s. |
| E6 | `03` §7 | FIXED, and the values are correct. Launch record start (8.64, 11.24, 12.77) shows as 8.6, 11.2, 12.8. End (14.76, 33.25, 35.32) shows as 14.8, 33.3, 35.3. |
| E7 | `03` §4 | FIXED, but one number is off (see Corrections 1). |
| E8 | `03` §6 | FIXED. 0.0010748 / 0.001 = 1.0748, so "7.5 %" is right. |
| E9 | `01_lock` §1, `02_checks` test row | FIXED. The only short hash in either lock file is `b565f0e8`. The test excludes 40-hex strings only, so "full commit hash" is accurate. |
| E10 | `01_lock` settings table and `tables/lock_settings.md` | FIXED. "The record itself has T = 2, the run configs set `game.T` = 3" matches the lock JSON (`records.*.T` = 2, `records.*.game.T` = 2) and the run configs (`record.game.T` = 3). |
| E11 | README row, STATE, `summary` "For the PI" | NOT CHANGED (see Corrections 2). `docs/STATE.md` is not in the diff, and the README row is unchanged. |
| E12 | README tests row | FIXED. `tests/test_t3r4_*.py` is now listed. |
| O1 | `03` §6 | FIXED, and correct. The stretch nodes are d = -22, -20, -18, -16, -14, and the final-tier maximum of the stage-2 gain is at d = -14. The development-tier peak is at d = -12, one node outside the stretch. |
| O2 | `03` §9, `summary` limits bullet | FIXED. |
| O3 | `03` §6 | FIXED, numbers correct (see Overstated 1). `last16.csv` covers updates 1225-1600 (16 exports). The maximum `rmse_pos` is 0.050523 and the maximum `eta_3` is 0.004193, both at update 1300. The final export has `eta_3` 0.00093 and `rmse_pos` 0.0214; `last16_summary.csv` has `n_above_limit` = 1 for `rmse_pos`. |
| O4 | `03` §1 and §5 | FIXED, numbers correct. `launch_checks.json` has `z_flags` = 1 in q50 s80536, `z_cells` = 5, and `starts_ok` for all 80. Non-fired runs with Delta >= 0.00475 are exactly three: 0.004897 (q60 s80528), 0.004825 (q50 s80510), 0.004758 (q50 s80520), all passing. The next is 0.004583, which is 91.7 % of the limit. |
| O5 | `03` §8 new bullet | Correct. My round-1 O5 was wrong (see below). |

**§5 sentence on the attempt-1 re-evaluation.** My scratch results, by final-tier `eta_3` of attempt-1 `u01600`, in trigger-table order, are:

| Seed | `eta_3` |
|---|---|
| 80501 | 0.006397 |
| 80515 | 0.007454 |
| 80526 | 0.005173 |
| 80532 | 0.009369 |
| 80535 | 0.006715 |

These round to the quoted 0.0064, 0.0075, 0.0052, 0.0094 and 0.0067, the order matches the trigger table, and all five are above 0.005.

**§8 first-attempt bullet.** Failures are 40 - 35 = 5 at q = 50, 1 at q = 60, 6 in all. 6 <= 6 meets the total limit exactly, and 5 > 4 exceeds the per-q limit. The bullet is correct.

**My round-1 O5 was wrong.** I wrote that the first-attempt count would "still have met the rule at exactly the boundary". That ignored the per-q limit of 4. The coordinator's disposition is right, and I withdraw that sentence.

### (b) `handoff.md`, claim by claim

All of the following are CONFIRMED against the data and the reports.

- **Verdict and counts:** 79/80; per-q 40/40 and 39/40; intervals; limits 6 and 4.
- **Operating characteristic:** 0.982 / 0.865 / 0.271 / 0.030 at f = 0.03 / 0.05 / 0.10 / 0.15.
- **Lock and code:** the empty frozen-code diff against the T3-R3 pre-registration commit `b565f0e8`; 33 listed files; commits `98355d8a` and `141b5189` pushed before the launch (reflog 07:10:33, launch 07:11:46); the guard list.
- **Trigger:** 5/80 (6.25 %, [0.021, 0.140]), all at q = 50, 5/5 second attempts pass. First attempt 74/80; the 6 first-attempt failures are 5 at q = 50 and 1 at q = 60. 7/120 over development plus confirmation.
- **Comparisons:** 118/120 (0.983, [0.941, 0.998]); `bw` 36/34/37 of 40, pooled 107/120 (0.892, [0.822, 0.941]).
- **Cost:** 3,800 / 5,400 (1.421); mean 3,900 (1.026); 597.6 s per run; 801 s fired against 584 s unfired (+37 %, wall ratio 1.37); 22.9 min with 40 concurrent runs; 47,810 s summed wall.
- **Failure paragraph:**
  - run and mechanism: s80502, 5 nodes, d in [-22, -14];
  - values: `Gmax_full` 0.0586 at d = -14, S3 0.536, Delta 0.000809, `rmse_pos` 0.0505 at one of the last 16 exports, final 0.0214;
  - T3-R3 comparison: the second failure of its kind after T3-R3 s70511, both at q = 60.
- **Margins:** `eta_3` 0.004897 = 98 %; `rmse_pos` 0.0454 = 91 %; 6 runs above 80 %; three non-fired runs within 5 % of the limit.
- **Tools and checks:** C-REPRO 162 npz; 109 tests; 27 findings; 1,928 passed, 6 skipped, 2 xfailed, 2 failed; the known failure explanation.
- **Remote state:** main is still `15e5b71a` and the remote tag list is unchanged (20 lines), so "no tags, main untouched" holds at this moment. "No pull request" is NOT DATA-CHECKABLE.
- **Format:**
  - exactly 5 GitHub links, all on one line at the full hash `dddfdd6ee26d...`;
  - every link target exists at that commit;
  - no e-mail and no personal name; the URLs carry only the account handle, as in the T3-R2 handoff;
  - no English sentence (a regex scan found no run of 3 or more English words outside code spans and URLs; only identifiers such as `bwA`, G-A, C-REPRO, SHA-256 and link texts remain).

### (c) STATE and README at HEAD

- `docs/STATE.md` and the README T3-R4 row are byte-identical to the versions I fact-checked in round 1. Still untrue at `dddfdd6e`:
  - "P0-P3 done" and "Done: ... P3: ... `handoff.md` ... and the push of `t3-r4`": `handoff.md` is untracked and `origin/t3-r4` is still `141b5189`.
  - The README "start here" cell lists `handoff.md` in the same way.
  - `summary.md` "`handoff.md` is the Chinese note for the coworker" is true only once it is committed.
- Everything else in STATE and the README (numbers, hashes, 27 findings, 109 tests, 1,928 passed) matches the data.
- `141b5189` is an ancestor of HEAD, so the planned push is a fast-forward (3 commits ahead).

## Corrections (exact old/new text)

1. **E7 residue, `03_confirmation.md` §4. Trivial.**
   - Old: "the development-final differences reach at most 47 %, 49 % and 24.1 % of theirs"
   - New: "... at most 47.1 %, 49 % and 24.1 % of theirs". The `tail_mean` maximum is 0.4709 of its limit (q60 s80529), so "47 %" is 0.09 percentage points too low. This inherited number came from my own round-1 suggestion.
2. **E11 residue, `docs/STATE.md` top section and README T3-R4 row. Premature, until the final commit and push.**
   - Old: "P3: ... `handoff.md` (Chinese), the independent fact-check (`pi_record/01_factcheck.md`) and the push of `t3-r4` (fast-forward; ...)", and the README "P0-P3 done".
   - Needed: after `handoff.md` and the round-2 ledger are committed and pushed, these hold. If STATE is committed before the push, write "pushed" only once `git ls-remote origin t3-r4` shows the new head. Also state the second fact-check round, since the ledger's "Round 2" section is still a placeholder at `dddfdd6e`.
3. **`handoff.md` §4, last bullet. Understated.**
   - Old: "没有发现错误的结果数字，只有措辞和归属上的更正，已全部处理"
   - New: "没有发现错误的结果数字；更正涉及措辞、归属、一处测试用例数目（'八个'应为'六个'）以及成本/负载的两处表述（'CPU-hours'实为墙钟之和、'约 50'的负载无记录）"
   - Reason: E2 was a wrong count and E5/E6 were a mislabelled unit and an unsupported figure, so "only wording and attribution" is too small. Also, "已全部处理" is not yet true for E11 until the final commit and push.
4. **`handoff.md` §2, failure paragraph. Ambiguous wording, inherited from the English summary.**
   - Old: "它处在第 3 阶段的夹紧区间里（S3 = 0.536）"
   - New: "该 run 处在第 3 阶段的夹紧区间里（S3 = 0.536）"
   - Reason: the clamp regime is a property of the run (stage 3, S3), not of the stage-2 stretch at d in [-22, -14].
5. **`handoff.md` §2, trigger paragraph. Possible misreading.**
   - Old: "在最终层上重新评估"
   - New: "在最终验证层级（final tier）上重新评估"
   - Reason: "最终层" can read as the final network layer.

## Overstated or loosely supported

1. **`03` §6 (O3 text): "The same run is not quiet in its last iterates".** The last-16 table was computed only for the failing run (`last16.csv` has one run). Nothing shows this run is noisier than passing runs, so "not quiet" implies an untested comparison. The figures are right; "reaches 0.0505 and 0.0042" is enough. Both maxima fall at the same export (update 1300).
2. **`handoff.md` §3 and `summary.md` limits bullet: "重启为什么有效的机制没有被确立" / "the mechanism by which the restart helps".** This presupposes an effect, which the same paragraph says cannot be measured without a control arm. "Whether and how the restart helps" is accurate.
3. **`03` §1 versus §7 load figures.** §1 quotes the `launch_machine.txt` snapshot (8.9 / 11.5 / 12.9, 21 s before launch). §7 quotes the launch record (8.6 / 11.2 / 12.8 "at the start"). Both are correctly sourced, but a reader will see two different "start" loads. One clause naming the sources would remove the confusion.
4. **"Recorded process CPU time" (12.1 h).** This sums the three training phases plus attempt 1. It excludes the continuation-table builds and process start-up, about 40 s per run, since a fired run's wall time of 804.8 s exceeds its summed phases of 764 s. "12.1 hours of recorded training-phase CPU time" is exact.
5. **`03` §4 "wide margin".** This still stands. It is one block of 80 runs, and the new §8 bullet shows the first-attempt counts would not meet the per-q limit.

---

## Dispositions of round 2 (mine)

| item | disposition |
|---|---|
| (a) E1-E10, E12, O1-O4 | confirmed fixed by the checker; nothing to do. The checker withdrew its round-1 O5 (the per-q limit of 4 makes the first-attempt count miss the rule at q = 50). |
| Correction 1 | `03_confirmation.md` section 4 now says 'at most 47.1 %, 49 % and 24.1 %'. |
| Correction 2 (E11 residue) | The README row, the STATE section and `summary.md` were true only after the final commit and the push; both were made true (the commit that adds `handoff.md` and this round-2 section is pushed last, fast-forward from `141b5189`), and STATE now names the two fact-check rounds. |
| Correction 3 | The last bullet of section 4 of `handoff.md` now lists what was corrected (wording, attribution, one test count, one unit, one unsupported load figure). |
| Corrections 4, 5 | '该 run 处在……夹紧区间里'; '最终验证层级（final tier）'. |
| Overstated 1 | The sentence 'not quiet in its last iterates' is replaced by the figures and a note that the last-16 table was computed for the failing run only, with no comparison with the passing runs. |
| Overstated 2 | `summary.md` and `handoff.md` now say 'whether and how the restart helps beyond a second draw is not established' ('重启是否以及如何起作用，没有被确立'). |
| Overstated 3 | `03_confirmation.md` section 7 names the sources of the two load pairs (the launch record at its start and end; `launch_machine.txt` 21 s before the launch). |
| Overstated 4 | '12.1 hours of recorded training-phase CPU time'. |
| Overstated 5 | unchanged: the text of section 4 states the margin against the limits and section 8 and 9 state the caveats (one block, first-attempt counts). |
