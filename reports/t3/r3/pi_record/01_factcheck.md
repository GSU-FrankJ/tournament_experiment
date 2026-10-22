# T3-R3: independent fact-check of the P3 reports

Round 1, by one independent read-only agent that wrote none of the reports, against commit `be9fe3bc` (documents checked: `reports/t3/r3/05_development.md`, `summary.md`, section 5 of `03_checks.md`). Its report is reproduced verbatim below; my dispositions follow it. The Chinese handoff, the README row and the `docs/STATE.md` section were written after this round and are covered by round 2 (end of this file).

---

# T3-R3 fact-check ledger (be9fe3bc): `05_development.md`, `summary.md`, `03_checks.md` §5

The data in the three documents is correct. I found no wrong result number. The corrections are 3 imprecise or wrong statements about process and cross-references, 1 misleading wording, and 3 smaller points (E1–E7 below).

I recomputed from the rawest data: `dev/eval/per_run.csv`, `diagnostic/eval/per_run.csv`, and each run directory's `status.json`, `manifest.json`, `retry.json`, `ms_checks_stage3.csv`, `ms_updates.csv` and `weights/*.npz`.
- Pass/fail uses my own implementation of the T3 gate rule (G-A, G-F, G-N, exit 0, global-RNG, valid evaluation).
- Bit-identity is `np.array_equal` on all exports and freeze files, plus value-equality of the CSVs.
- Bootstrap intervals use `default_rng(20261021)` with 10,000 resamples. The CP, McNemar and Fisher p-values come from scipy.
- S3 is `Σ d1_L_s3_out_lo / Σ d1_L_s3_out_n` over stage 3, local 1201–1600.
- For the stage-2 stretch, the tail-node and alpha-min columns, and F(u) I called the repository's `network_on_grid` / `export_summary`. These are partly independent, since the code is shared.

Scratch scripts are in `/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-t3-bw-failures-diagnosis-25cb13/d068e02a-164b-4c01-950c-16d86f43269c/scratchpad/factcheck/`. Nothing in the repository was touched.

## 1. Ledger

| # | document, section | claim | how checked | verdict |
|---|---|---|---|---|
| 1 | 05 intro; summary; 03§5 | 122 runs = 120 dev + 2 diagnostic; all exit 0; no infra event; 80501–80540 never trained | `status.json` ×122 (state done, exit 0); launcher JSON (122 returncode 0); `infra_events.json`; `find results -name 'seed805*'` is empty | CONFIRMED |
| 2 | 05 intro, §4, §12; summary | pass 37 / 39 / 38 of 40; bwA proposed; both arms reach target | own pass rule on 122 rows; reproduces `headline.json` and `blind.json` (`differences: []`) | CONFIRMED |
| 3 | 05 intro, D-1 (line 45); 03§5 D-1; summary D-1 | script written "71 s after the launch at 02:20:02" | `launch_start.txt` = 02:20:02Z. Script birth (from `boundary.json`) = 02:21:15.680, so 73.7 s after that. 71.7 s is measured from the launcher's own stamp 02:20:04 (`launch_20261022_022004.json`). The PI's 28c reply also says "71 s". | CORRECTION NEEDED (E1) |
| 4 | 05 §1.1 checks table (17 rows × 5 cols) | counts incl. clean_tree 40/82; first call config_equals_plan 0/122 | `verdicts` and raw per-run booleans of `launch_checks_absroot.json` and `launch_checks.json` | CONFIRMED |
| 5 | 05 §1.1 arm table | runs 40/41/41; fired 0/3/4; would-fire 2/3/4; failed checks; C-PAIR 40/40 bwA, 0/40 bwAB identical, 1/1 diagnostic | `by_arm`, `pair_detail`. Own comparison: bwA vs bw has 0 differing exports in all 40 pairs (38 over all 152 exports, 2 over the first-attempt 64). bwAB differs from bw in 26–56 of 64 stage-3 exports in all 40 pairs. | CONFIRMED |
| 6 | 05 §1.1 prose | ledger present, 0 events; C-INIT 0 / 0 differences; fired runs have 5,400 / 2,764,800 / 108,000 / 216 | `manifest.init_state_sha256` and `retry.json` digests: 0 of 41 (q,seed) differ; both-fired attempt-2 digests equal; `retry.json totals` for all 7 fired runs | CONFIRMED |
| 7 | 05 §1.1 "Reading", §1.2 pair_ok note | pair_ok verdict row counts bwAB as passing; raw first report had 40 bwAB failures | raw `pair_ok` False ×40 (bwAB), `by_arm` bwAB `pair_ok: 40`, verdict row 122/0 | CONFIRMED |
| 8 | 05 §1.2 D-2, D-3, "tools changed after launch" | relative `--root` gives 0/122; `post_launch.sh` continued after checks exit=1; `git diff b565f0e8 HEAD` shows only the root check | `launch_checks.json`; `launch_checks.out.txt` ends `exit=1` at 02:56:09; eval started 02:56:09; `git diff` read in full; F10 text in `t3/r2/03_checks.md`; nothing committed or pushed in between (origin at b565f0e8 until 03:21+) | CONFIRMED |
| 9 | 05 §1.2 D-2 | "the four tools that take `--root` … refuse a relative path (tested)" | true at e6902c28 (4-way parametrised test). `t3r3_last16.py` (11354e1f) is a fifth, also refusing. | CONFIRMED (note E7) |
| 10 | 05 §1.3(1) boundary table | clean 40 (0 bw / 1 bwA / 39 bwAB), starts 02:20:08–09; dirty 82 (40/40/2), starts 02:29:08–02:48:43; 0 disagreements; all b565f0e8; diagnostic all clean | `status.json` start_time, `manifest.git.dirty`, cut 02:21:15.68. Dirty bwAB = q60 s70519 and s70520. | CONFIRMED |
| 11 | 05 §1.3(1) "created at … 02:21:15.679567227 (birth time, `stat`)" | | value is in `boundary.json`; `stat` today gives birth 03:20:40.627, mtime 03:39:47 (file moved and edited since) | CORRECTION NEEDED (E3) |
| 12 | 05 §1.3(1) "git status … lists only that file at verification time" | | `files_outside_…txt` lists two untracked paths (the script and `pi_record/28c_dirty_tree_reply.md`) | CORRECTION NEEDED (E2) |
| 13 | 05 §1.3(1) wave-window search (02:20:00–02:55:47) | only that file and its parent directory | recorded in the txt. Search can't be rerun (file since modified); other tracked/untracked files in the repo have mtimes outside the window. | CONFIRMED from record |
| 14 | 05 §1.3(1) "runner opens no file under `reports/`" | | `grep "reports"` over `run/ agents/ utils/ envs/`: one docstring in `run/ms_retry.py`; no `reports` in any run's config, manifest or log; `git_state` code read | CONFIRMED |
| 15 | 05 §1.3(2) pairs across boundary | 40 pairs, 0 flags differ, 38 bit-identical + 2 same stage 3; one clean bwA = diagnostic | `pairs_across_boundary.csv`; own bit comparison | CONFIRMED |
| 16 | 05 §1.3(3) re-run table (5 rows) | flags dirty→clean, rc 0, 162 npz, same file set, attempt1/retry.json identical, JSONs identical, commit b565f0e8 | `rerun_results_final.json`, `compare_*.txt`, re-run `manifest.json` / `status.json`. Wall-time column: see E4. | CONFIRMED (E4) |
| 17 | 05 §1.3(3) NaN note | first comparison "not ok" only through NaN≠NaN in `R_tail` | `rerun_results_first_comparison.json`: all diffs ∈ {clean_tree, R_tail}; original and re-run `gates.json` R_tail = NaN in both; `compare_runs` rc 0 in both passes | CONFIRMED |
| 18 | 05 §1.3(3) selection | two fired dirty bwA; no dirty bwAB fired; diagnostic clean; lowest dirty non-fired bwAB at q60 = 70519; none at q50; additions bwA 70501 at q50, q60 | recomputed from manifests, retry.json and starts | CONFIRMED |
| 19 | 05 §1.4; 03§5 | 12 shards at 11354e1f; 1,819 passed / 6 skipped / 2 xfailed / 2 failed; the two named failures | `tests_p3/shard*.txt` summed: 1819 / 6 / 2 / 2; `pytest_commit.txt`. P2: 1,811. New tests: 4+1 and 3 → +8. | CONFIRMED |
| 20 | 05 §2 | arm keys; same seeds; launched 02:20:02, ended 02:55:46, commit b565f0e8, 40 workers, 1 thread; pre-reg pushed before launch | `run_config.json` ×122; `launch_*.txt`; `torch_threads` 1 / OMP/MKL/OPENBLAS 1; max concurrency 40; `reflog origin/t3-r3` push 02:19:50Z | CONFIRMED |
| 21 | 05 §2 / §1.3 "in tmux" | | no record in repo files | NOT DATA-CHECKABLE |
| 22 | 05 §3 per-seed tables (q50, q60: 40 rows × 9 cells = 360 cells) | verdict, failing gates, `[fired]` / `[would fire]`, eta_3, Gmax | own pass rule, retry.json, local-1600 Delta | CONFIRMED (0 mismatches at printed precision) |
| 23 | 05 §4 counts table (9 rows × 12 cols) | pass, CP intervals, worst Gmax + run, medians | own recomputation | CONFIRMED |
| 24 | 05 §4 target tables, selection, first-attempt table (37/37/2; 37/39/2; 35/38/3), invariant (0 mismatches, `bw` fired-and-pass = 0) | | own recomputation | CONFIRMED |
| 25 | 05 §4 "0.95 → 0.68, 0.90 → 0.22" | | binomial recomputation 0.6767 / 0.2228 | CONFIRMED |
| 26 | 05 §5 paired table (18 rows × 8 cols) | means, differences, bootstrap intervals, n_neg/n_pos | own bootstrap (default_rng(20261021), 10,000 resamples) | CONFIRMED |
| 27 | 05 §5 cross-tables + reading | 37/0/2/1 p=0.5; 36/1/2/1 p=1; 37/2/1/0 p=1; "intervals contain 0" | exact binomial | CONFIRMED |
| 28 | 05 §6 failure table (6 rows × 13 cols) | gate values, S3 first/frozen, stretch nodes, Delta@1600, last-16 ranks, class | recomputed; last16 `u01600` equals the evaluated value (diff 0.0); ranks / n_above recomputed from `last16.csv`; 3 exports re-evaluated, diffs ≤ 1e-16. F(u01600) ≥ 50 in all 120 dev runs (thresholds 32.11 / 26.78), so no (F). | CONFIRMED |
| 29 | 05 §6 prose | 4 distinct seeds, 6 runs; 5 K, 1 V; 70511 Delta 0.0040 < 0.005 | | CONFIRMED |
| 30 | 05 §6 "(E), (U) none" | | classification judgement | NOT DATA-CHECKABLE |
| 31 | 05 §7 fired table (7 rows × 12 cols) + prose | | `retry.json`, S3 of attempt1/frozen, `a2` final Delta; all fired runs at q50 except the diagnostics | CONFIRMED |
| 32 | 05 §8 three tables (27 + 27 + 27 rows) | S3 counts, medians, stretches (10 / 42 / 52 nodes etc.), tail-at-clamp, alpha_min, Delta@1600, fire rates, McNemar counts, 16/20/36/18/38, eta and peak differences + intervals | own recomputation; stretch / alpha / mu via repo functions | CONFIRMED |
| 33 | 05 §8 reading (i)(ii)(iii), Fisher p = 0.62, restarts leave/stay regime | | Fisher [[3,37],[1,39]] = 0.6153; S3 per fired run | CONFIRMED |
| 34 | 05 §9 diagnostic table + prose | Delta / s / C, F(u1600) 54.08 / 50.16, thr 26.78, formation export 225, sens 1.904 / 1.757, min-sens 0.000777, eta, rmse, Gmax; attempt 1 = T3-R2 original; init digest fb3fde18 = original manifest | recomputed F, first export > thr, sens; `pair_detail`; original manifest exists | CONFIRMED |
| 35 | 05 §9 "(7e-10 in the failed original, `01_diagnosis.md` section 10)" | | value is 6.82e-10 in §3 (line 67) and §8 (line 511); §10 says only "about 0" | CORRECTION NEEDED (E5) |
| 36 | 05 §10 cost table (3 rows × 13 cols) + factor 1.421 | means, totals, medians, verifier s, ratios | `status.json`, `ms_run_summary.json`, `retry.json` | CONFIRMED |
| 37 | 05 §10 wall tables (4 rows) + prose (818 vs 602, +216 s = 36 %, 1.021 × 1.065 = 1.087, 1.092, 1.168, order bwAB→bwA→bw, 1.037 at 7/80, [0.006, 0.169]) | | recomputed: 582.5 / 546.8 / 1.065; 817.8 / 602.2 / 1.358; 635.5 / 544.1 / 1.168; 635.5 / 582.0 / 1.092; start ranks by arm; CP | CONFIRMED |
| 38 | 05 §10 "partly in the draining phase" | | last job starts 02:48:43; 37 of 40 bw runs end later | CONFIRMED |
| 39 | 05 §11 predictions | fire rates with CP [0.006, 0.169] / [0.016, 0.204]; 38 of 38 non-fired pairs identical; diagnostics fire | | CONFIRMED |
| 40 | 05 §11 "DP3 … holds in the wave as well" | | DP3 names specific T3-R1/R2 runs; the wave only reuses the class scheme | see §3 below |
| 41 | 05 §12 bullets | | each recomputed above | CONFIRMED, except the wording flagged in E6 |
| 42 | 05 §13 files | all named files exist and are tracked | `git ls-files` | CONFIRMED |
| 43 | summary: result table, intervals, gain = 2 seeds (70508, 70516), 13 vs 12, 1 vs 3, [-0.00008, 0.00064], 6 runs / 4 seeds / 5K + 1V, diagnostic numbers, cost numbers, D-1…D-3 text, suite counts, G2 asks | | as above | CONFIRMED |
| 44 | summary "most of the wall-time gap of bwA is the environment" | | gap 1,785 s; 1,354 s (76 %) is in the 38 bit-identical seeds | CONFIRMED (inference, see §3) |
| 45 | 03§5 wave paragraph, D-1…D-3, P3 code list, suite | clean 02:20:08–09, dirty 02:29:08–02:48:43; e6902c28 / 87ca2144 / 11354e1f contents; "nothing under agents/ envs/ utils/ run/ protocols/ changed after 727116a3" | `git show --stat`; `git diff 727116a3 HEAD -- agents envs utils run protocols` is empty; test counts 5 / 3 | CONFIRMED |
| 46 | 03§5 last line: "Independent fact-check … `pi_record/01_factcheck.md`" | | file does not exist at be9fe3bc (`pi_record/` holds only 28*.md) | CORRECTION NEEDED (E8) |
| 47 | 03 header "T3-R3, P1: checks, Date: 2026-10-21" | | §4–§5 cover P2/P3 and 2026-10-22 events | CORRECTION NEEDED (E9, trivial) |
| 48 | "git diff HEAD was empty"; "search at 03:10 UTC"; "training ran with the pre-NaN-rule driver version" | | not reproducible from repo | NOT DATA-CHECKABLE |

## 2. Errors and corrections

**E1 (imprecise number; 05 line 5 and line 45, 03§5 D-1, summary D-1).** Now: "71 s after the launch at 02:20:02".
- Correct: the script's birth time 02:21:15.68 is 73.7 s after `launch_start.txt` (02:20:02Z). It is 71.7 s after the launcher's own start stamp 02:20:04 (`launch_20261022_022004.json`; first run starts 02:20:08).
- Suggested text: "about 73 s after `launch_start.txt` (02:20:02Z; 71 s after the launcher's own stamp 02:20:04Z)". The PI's 28c reply has the same "71 s".

**E2 (imprecise; 05 line 64).** Now: the runner's `git status --porcelain …` "lists only that file at verification time".
- Evidence: `files_outside_results_in_wave_window.txt` lists two untracked paths: `results_tables.py` and `reports/t3/r3/pi_record/28c_dirty_tree_reply.md` (saved after the wave).
- Correct: "lists that file and the 28c reply (saved after the wave); with both moved out, nothing outside `results/`".

**E3 (imprecise; 05 line 62).** Now: "created at … 02:21:15.679567227 (birth time, `stat`)".
- A `stat` today gives birth 03:20:40.627 (the file was moved and restored) and mtime 03:39:47.
- The 02:21:15.679567227 value is the one saved in `boundary.json` before the move. Cite `boundary.json`, not `stat`.

**E4 (imprecise label; 05 line 74 column "re-run wall time (s)").**
- The values 496 / 494 / 368 / 377 / 363 are the driver's subprocess wall clock (`rerun_results_first_comparison.json`).
- The re-runs' own `status.json total_wall_sec` are 492.6 / 491.4 / 364.8 / 373.8 / 359.7. `rerun_results_final.json` carries `wall_sec ≈ 1e-4` (compare-only pass).
- Say which clock is meant, or use `total_wall_sec`.

**E5 (wrong cross-reference; 05 §9, line 323).** Now: "(7e-10 in the failed original, `01_diagnosis.md` section 10)".
- Correct: the 6.82e-10 (≈ 7e-10) is in §3 (DG2 table, line 67) and §8 (line 511), and in §0. §10 says only "about 0".

**E6 (misleading wording; 05 §12, line 365).** Now: "B is not distinguishable from a marker by the pre-registered indicators ('mixed')".
- Pre-registration §7: none of (i)–(iii) met = "a marker"; all three = "more than a marker"; otherwise "mixed". Here (ii) is met, and (i) and (iii) are not, so the label is "mixed".
- Correct: "the pre-registered indicators give 'mixed' (one of three met: (ii); (i) and (iii) not met); no causal claim".

**E7 (note; 05 D-2).**
- "The four tools" is right at e6902c28. `t3r3_last16.py` (11354e1f) also refuses a relative `--root`, so five tools do now.
- Also, `post_launch.sh` fed `analysis/diagnostic.csv`'s `attempt1_equals_T3R2_original` from the first-call `launch_checks.json` (relative root), not from `launch_checks_absroot.json`, which the text calls the record. `pair_ok` is identical in both files for the diagnostic runs, so there is no numeric effect. Worth one sentence.

**E8 (dangling reference; 03§5, last line).** `pi_record/01_factcheck.md` does not exist in be9fe3bc. Create it before the push, or reword.

**E9 (trivial; 03_checks.md header).** The title is "P1: checks" and the date is 2026-10-21, but §4 (P2) and §5 (P3) document 2026-10-22 events.

## 3. Overstated relative to the data

- **"Verified harmless" (05 §1.2 Reading, summary D-1).**
  - Direct evidence is 5 clean-tree re-runs of 82 dirty-flagged runs: 2 fired bwA, 1 bwAB, and 2 non-fired bwA added as stand-ins for bw.
  - No `bw` run (40 of the 82) was re-run.
  - The rest rests on two arguments: the flag follows the file's creation time exactly, and the runner only records the boolean. Say "verified on five re-runs plus the boundary argument".
- **"B not distinguishable from a marker."** See E6.
- **"DP3 holds in the wave as well" (05 §11).** DP3 concerns specific T3-R1/R2 runs. The wave only shows that the same class scheme assigns K to 5 and V to 1 of the 6 failures.
- **"Most of the wall-time gap … is the environment" (summary); "environmental 1.065" (05 §10).**
  - The numbers are right: 76 % of the bwA − bw wall gap sits in the 38 bit-identical computations.
  - "Environment" is inferred from run order and load, not measured. 05 §10 itself says an interleaved run would be needed, so the summary should be as hedged.
- **"Both candidates reach the target."** True, but bwAB reaches it with zero margin (38/40 and exactly 18/20 at q = 50). The gate is inclusive, so it is not an error.
- **"No failure is (E) or (U)."** Only (F) is data-excluded (F(u01600) ≥ 50 in all 120 dev runs). (E) and (U) are judgements.

---

# Dispositions of round 1 (mine)

No result number was wrong. Every item below was corrected in the templates and the reports regenerated before the push (`report_scripts/05_development.template.md`, `summary.template.md`, `results_tables.py`, `03_checks.md`).

| item | disposition |
|---|---|
| E1 (71 s) | Corrected everywhere to 'about 73 s after `launch_start.txt` (71 s after the launcher's own stamp 02:20:04)' in `05_development.md`, `summary.md`, `03_checks.md`, `handoff.md`, `reports/t3/README.md` and `docs/STATE.md`. The PI's reply `28c` keeps its own '71 s' (saved verbatim). |
| E2 | Reworded (and corrected again after round 2, C1): the porcelain output listed the report script and the saved reply `28c` when run after the re-runs with both restored, and nothing while both were moved out (`rerun_stdout.txt`). |
| E3 | The creation time is now cited to `boundary.json` (recorded before the file was moved), with a sentence that `stat` of the restored file shows the time of the copy. |
| E4 | The column is labelled 'driver clock'. |
| E5 | Corrected to '6.8e-10, `01_diagnosis.md` sections 3 and 8'. |
| E6 | Reworded to 'the pre-registered indicators give mixed (one of the three met, (ii); (i) and (iii) not met), with no causal claim'. |
| E7 | One sentence added to D-2: five tools now refuse a relative `--root`, and `analysis/diagnostic.csv` took its diagnostic C-PAIR from the first-call check file (same `pair_ok`, no number changes). |
| E8 | This file. |
| E9 | `03_checks.md` title and date line now cover P1-P3. |
| 'verified harmless' | Replaced by 'checked on five re-runs (two with a fired trigger, no `bw` run) plus the boundary argument'; the report says that no `bw` run was re-run because `bwA` without a trigger is bit-identical to `bw`. |
| DP3 wording | Replaced by the statement that the wave only shows the same classification assigning five failures to (K) and one to (V). |
| wall time 'environment' | Hedged in the summary and the handoff ('appears to be', inferred from run order and load, not measured). |
| `bwAB` without margin | Added to section 4 of the report and to the summary. |
| '(E) and (U) none' | Reworded: (F) is excluded by data; (E) and (U) are judgements, 'I assigned none'. |

# Round 2

Same agent, resumed, on the working tree after the round-1 corrections (the corrected passages of `05_development.md`, `summary.md`, `03_checks.md`, the new `handoff.md`, the README rows, the STATE section). Its report verbatim:

---

# Round 2 fact-check: corrections E1–E9, handoff.md, README rows, STATE.md (working tree, read-only)

Corrections E1–E9 and the overstated items are mostly fixed. No result number is wrong in any new text. The handoff diagnosis and results numbers all check out. What remains is 7 wording or timing issues (C1–C7), 2 stale-text notes and 4 smaller observations (C8–C13).

I regenerated `05_development.md` and `summary.md` in memory from their templates and the tables. They match the files on disk line for line, and `t_rerun()` matches `tables/res_rerun.md`. Nothing was written to the repository.

## 1. Ledger

| group | claim | check | verdict |
|---|---|---|---|
| (a) E1 | "about 73 s … (`launch_start.txt` 02:20:02; 71 s after the launcher's own stamp 02:20:04)" in 05 intro and D-1, summary D-1, 03§5 D-1, README, STATE, handoff | 02:21:15.68 − 02:20:02 = 73.7 s; − 02:20:04 (`launch_20261022_022004.json`) = 71.7 s. The stamps are whole seconds, so "about 73" and "71" are consistent. | CONFIRMED |
| (a) E2 | porcelain "listed that file and the 28c reply … at the start of the verification, and nothing once both were moved out" | `files_outside_…txt` says the listing was taken "after the verification re-runs, with the file restored". `rerun_stdout.txt` gives "tree clean outside results/" for the moved-out state. No record of a listing at the start. | CORRECTION NEEDED (C1) |
| (a) E3 | creation time cited to `boundary.json`; `stat` shows the copy's time | `stat` today: birth 03:20:40.627. The next sentence is right. | PARTLY FIXED (C2) |
| (a) E4 | column "re-run wall time (s, driver clock)" | values unchanged (496 / 494 / 368 / 377 / 363 = driver clock); template, `results_tables.py` and `res_rerun.md` agree | CONFIRMED |
| (a) E5 | "6.8e-10, `01_diagnosis.md` sections 3 and 8" | §3 table has 6.82e-10; §8 writes 7e-10 | CONFIRMED |
| (a) E6 | "indicators give 'mixed' (one of three met, (ii); (i) and (iii) not)" | 13 vs 12, p = 1; 1 vs 3; interval contains 0 | CONFIRMED |
| (a) E7 | `diagnostic.csv` took its C-PAIR from the first-call `launch_checks.json`; `pair_ok` same in both; last16 refuses a relative root | `post_launch.sh` passes `--checks results/t3_r3/dev/launch_checks.json`. `diagnostic_rows` reads `checks["runs"][…]["pair_ok"]`. `diagnostic.csv` mtime is 02:56:29, before the absroot file (02:59:42). `pair_ok` is True for both diagnostic runs in both files. `t3r3_last16.py` calls `require_absolute`, with a test. | CONFIRMED |
| (a) E8 | `pi_record/01_factcheck.md` exists | on disk, untracked (to be committed) | CONFIRMED |
| (a) E9 | title "P1-P3", dates line | read on disk | CONFIRMED |
| (a) "bwAB reaches the target with no margin (38 of 40, exactly 18 of 20 at q = 50)" | counts 38 / 18 | CONFIRMED |
| (a) "F(u01600) above both thresholds in all 120 dev runs" | F(u01600) minimum is 58.98 at q = 50 (threshold 32.11) and 50.13 at q = 60 (threshold 26.78); computed with the repo's `export_summary` | CONFIRMED |
| (a) "no `bw` run was re-run, since `bwA` without a trigger is bit-identical to `bw`" | the 5 re-runs are 4 `bwA` and 1 `bwAB`; the 38 non-fired pairs are bit-identical (own comparison of exports, CSV values and freeze files) | CONFIRMED |
| (a) "verified harmless", DP3, "(E)/(U)" rewording | grep and read: gone from 05, summary, 03, README and handoff | CONFIRMED |
| (a) wall-time hedge | summary and handoff are hedged. 05 §10 and STATE.md are not (C3). | PARTLY FIXED |
| (b) handoff L3, L24–36 | 10 failures (4 from T3-R1 + 6 from T3-R2); seeds; 122 runs; results table and intervals; 38 identical seeds; McNemar p = 0.5; 37 = 37; 13 vs 12 (p = 1); 1 vs 3; [-0.00008, 0.00064] on 36 seeds; RMSE 0.0543, 1 of 16 above the limit, the largest; 28 nodes; Δ 0.0040; Δ 0.1447; F 54.1 / 50.2 against 26.78 | recomputed | CONFIRMED |
| (b) L9–11 | classes F 1 / K 8 / V 1 (60518; 50504, 50514, 50517@q60, 60501, 60506, 60513, 60515, 60516; 50517@q50); constant effort 8.155; S3 = 0 | `dg7_failures.csv`, `dg6_s3_runs.csv`, §3 | CONFIRMED |
| (b) L10 S3 definition | "last 400 updates, out-of-band starts, learner actions below the action clamp 1e-6, > 20 %" | `s3_of_rows` = Σ`d1_L_s3_out_lo` / Σ`d1_L_s3_out_n` over stage 3, local 1201–1600 (the last 400 of 1600). `run/ms_rollout.py`: `lo = raw < action_clamp`. Run configs: `action_clamp` = 1e-6. The prompt's definition says "at out-of-band starts (|d| ≥ 2q)". | CONFIRMED |
| (b) L13 | 30 of 80 in the regime | `dg6_s3_runs.csv`: 30 (25 above 0.4) | CONFIRMED |
| (b) L13 | 21 of 30 vs 0 of 50 with the Beta mean at the 1e-6 clamp | `dg2_runs.csv`: `first_export_tail_mu_at_clamp` is set in 21 of 30 and 0 of 50. The same split holds for `out_mu_at_clamp_final`. The report says "at some export"; the handoff omits that. | CONFIRMED (C6) |
| (b) L13 | 0.3 % and 0.5–0.7 % gradient shares; "DP2 not supported" | wording is `01_diagnosis.md` §0 item 3 (also §10, README, STATE). `dg5_group_medians.csv`: clamp-draw share 0.21–0.31 % from update 400 on; out-of-band share 0.49–0.79 % for updates 801–1600. | CONFIRMED as the report's own wording (note C9) |
| (b) L15 | trigger fires on 7 of 80, all 7 failures, misses 3, +3.7 % | `retro_triggers.csv`: n_fired 7, fired_and_failed 7, missed r1_q60_s50514, r1_q60_s50517 (stage-2 stretches, G-F) and r2_q60_s60516 (G-A rmse only); cost fraction 0.0368 | CONFIRMED |
| (b) L19–20 | A and B descriptions | prereg §3; own bit comparison | CONFIRMED |
| (b) L40 | 3800 / 3880 / 3920, 1.021, 1.032, 5400, 1.421, wall 1.081 / 1.201, 1.065, order bwAB → bwA → bw | recomputed | CONFIRMED |
| (b) L44–47 | 82 manifests, about 73 s, 5 re-runs (2 fired, no `bw`), flag follows the creation time; relative root and 0/122; 1,819 / 6 / 2 / 2 | recomputed | CONFIRMED except C5 |
| (b) L51 | 6 of 80 / 4 of 40 per q; only a G1 §2.2 reason may change the choice | prereg §9 and the G1 reply | CONFIRMED |
| (b) form | 5 GitHub links (exactly the limit), each with `COMMIT` | counted | CONFIRMED |
| (b) form | no email, phone or personal name | The URLs carry the account handle `GSU-FrankJ` (a first name and initial). The T3-R2 handoff uses the same pattern. | NOTE: your call |
| (b) form | Chinese only | The only Latin text is identifiers, code spans (`bw`, `C-PAIR`, `McNemar`, `RMSE`, `manifest`, `PI`) and link labels. No English sentence. | CONFIRMED |
| (c) README T3-R3 row | numbers: 37/40, 39/40, 38/40, 122 runs from `b565f0e8`, p = 0.5, 1.021 / 1.032, 1.081 / 1.201, "about 73 s", classes 1 / 8 / 1, `c3289df9` = `origin/t3-r2`; all named paths exist | `git rev-parse`; `ls` | CONFIRMED |
| (c) README tools row | all 13 `tools/t3/t3r3_*` files, `t3r3_preflight_seeds.sh`, `t3r3_post_launch.sh`, `run/ms_retry.py`, `tests/test_t3r3_*.py`, `tests/t3_helpers.py` exist | `ls` | CONFIRMED |
| (c) README line 17 "Files of the round (`reports/t3/r3/`)" | not part of the new rows | still lists only P1 files | STALE (C10) |
| (d) STATE.md | `28b`, `03_checks.md` section 4 (14 findings), C-LEGACY at `727116a3`, pre-registration `b565f0e8`, 122 runs, 40 workers, 02:20–02:55, results and intervals, 13 vs 12, stretches 1 vs 3, 1.021 / 1.032, 1.081 / 1.201, 1.065, D-1…D-3, 1,819 / 6 / 2 / 2, the two failing tests, 70511 | recomputed | CONFIRMED |
| (d) STATE.md | "2.7 GB for the wave" | `du` on `dev` + `diagnostic` = 2.66 GiB (2.86 GB decimal) | CONFIRMED |
| (d) STATE.md | untracked / tracked file lists | `git ls-files` | CONFIRMED |
| (d) STATE.md | "the push of `t3-r3` (fast-forward …)" listed as done | `origin/t3-r3` is still `b565f0e8`; local HEAD is `be9fe3bc` plus uncommitted files. A push would be a fast-forward. | NOT YET TRUE (C4) |
| (d) STATE.md | "most of that is the environment" | unhedged | CORRECTION NEEDED (C3) |

## 2. Corrections needed

**C1 (05 line 64; same in `01_factcheck.md` disposition E2).**
- Old: "…listed that file and the saved PI reply `28c_dirty_tree_reply.md` (written after the wave) at the start of the verification, and nothing once both were moved out."
- New: "…listed that file and the saved PI reply `28c_dirty_tree_reply.md` (written after the wave) when run after the re-runs with both restored (`files_outside_results_in_wave_window.txt`), and nothing while both were moved out (`rerun_stdout.txt`: 'tree clean outside results/')."
- Evidence: the txt's own header; no record of a listing at the start. Also fix the template.

**C2 (05 line 62 and template).**
- Old: "(birth time, `stat`)". New: "(birth time recorded in `boundary.json`)".
- The clarifier two lines later is correct; the first sentence still points readers at `stat`.

**C3 (05 §10 line 346 and template; STATE.md, Result paragraph).**
- Old: "1.021 times the environmental 1.065 is 1.087". New: "1.021 times the 1.065 measured on the bit-identical computations (run order and load) is 1.087".
- STATE old: "so most of that is the environment". New: "so most of that appears to be the environment (run order and load; inferred, not measured)".
- 76 % of the gap (1,354 of 1,785 s) lies in the 38 identical seeds, so "most" is right. The inference is not measured, which is what the summary and handoff now say.

**C4 (STATE.md, Done).**
- "…and the push of `t3-r3` (fast-forward; no tag, no pull request, `main` untouched)" is not true yet.
- Make it true by pushing before the file is read as state, or move it to "Next".

**C5 (handoff L45).**
- Old: "绝对路径重跑后通过；…四个带 `--root` 的工具现在拒绝相对路径。"
- New: "绝对路径重跑后 `config_equals_plan` 为 122/122（`clean_tree` 仍有 82 个失败，见上一条）；…五个带 `--root` 的工具（launch、eval、checks、analysis、last16）现在拒绝相对路径。"
- Reason: after the absolute-root rerun the overall check still fails on `clean_tree`. 05 D-2 now says five tools.
- STATE.md Known issues (D-2) also says "the four tools"; the same fix applies.

**C6 (handoff L13).**
- Old: "（30 个里 21 个，其余 50 个里 0 个）". New: "（30 个里 21 个在某个导出点出现，其余 50 个里 0 个）".
- The report says "at some export".

**C7 (handoff L7, mild overstatement).**
- Old: "也没有无法解释的失败". New: "没有归入 (U)（无法解释）的失败；(K) 类的机制并未被证明，见下面".
- `01_diagnosis.md` §10 gives the DP3 verdict as "supported as a classification; the mechanism is not shown".

**C8 (handoff L36, imprecise).**
- Old: "第一次尝试与 T3-R2 的原始运行逐位相同". New: "第一次尝试（第 3 阶段）与 T3-R2 原始运行的第 3 阶段（64 个导出、第 3 阶段各行与检查、分箱图和状态）逐位相同".
- Only stage 3 up to the check was compared.

## 3. Notes and overstated items

- **C9, gradient shares (`01_diagnosis.md` §0 and §6; the handoff inherits the wording).**
  - "From update 400 on the out-of-band samples carry 0.1–0.7 %" does not match the report's own table: the 401–800 window is 1.7 % for failures and 1.06 % for regime runs.
  - 0.49–0.79 % holds only from update 801 on. The conclusion (nowhere near the 0.75 sample share) is unaffected.
  - The handoff gives the shares without the sample shares (0.75, 0.53–0.57); adding them would show that "只占" is relative to those.
- **C10, README line 17.** "Files of the round (`reports/t3/r3/`)" lists only the P1 files (no `04`, `05`, `summary`, `handoff`, `pi_record/28b`, `28c`, `01_factcheck`). Its results part lists only `preflight/`, `diag/`, `replays/` and `tests/`; it misses `dev/`, `diagnostic/`, `legacy/`, `tests_p2/`, `tests_p3/` and the launch records. The new row carries these, so the line is incomplete rather than wrong.
- **C11, 05 heading "### 1.3 The dirty-tree flag, verified (reply 28c)".** "Verified" is the PI's verb. The body is hedged, so this is fine.
- **C12, unverifiable.** The new sentence "`git diff HEAD` was empty when I checked at 03:10 UTC" is a process claim with no record in the repository.
- **C13, "about 73 s".** The exact value is 73.7 s, so "about 74" would round correctly. At the whole-second resolution of the stamps, "about 73" and "71" are consistent.

---

# Dispositions of round 2 (mine)

| item | disposition |
|---|---|
| C1 | Corrected in the template (and so in `05_development.md`) and in the round-1 disposition above. |
| C2 | Corrected in `results_tables.py` (`boundary` table text): 'birth time recorded in `boundary.json`'. |
| C3 | Corrected in the template and in `docs/STATE.md` ('appears to be the environment: run order and load, inferred and not measured'). |
| C4 | `docs/STATE.md` now says the final push 'follows the commit that holds this file'; the push is made right after the commits. |
| C5 | The handoff and `docs/STATE.md` now say that `config_equals_plan` is 122/122 after the absolute-root call while the overall check still fails on `clean_tree`, and that five tools refuse a relative `--root`. |
| C6, C7, C8 | Corrected in the handoff as proposed. |
| C9 | The handoff now gives the sample shares beside the gradient shares. The wording in `01_diagnosis.md` §0 and §6 (the accepted P1 report, committed at G1) is not changed here: its table shows the windows 401-800 at 1.06 % and 1.7 %; this is noted here as a known imprecision of P1, with no effect on the conclusion that the shares are far below the 0.75 sample share. |
| C10 | The README 'Files of the round' paragraph now lists the P2/P3 files and result directories. |
| C11 | The heading keeps the PI's verb; the body states what was and was not re-run. |
| C12 | The sentence ("`git diff HEAD` was empty when I checked at 03:10 UTC") is removed: no record of it in the repository. |
| C13 | 'about 73 s' kept (the stamps have whole-second resolution; the PI's 71 s is the figure relative to the launcher's own stamp). |
| account handle in the URLs | The links follow the repository's address (`github.com/GSU-FrankJ/...`), as in the T3-R1 and T3-R2 handoffs; the text itself contains no name or contact detail. |
