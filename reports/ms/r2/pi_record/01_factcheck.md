# MS-R2 fact-check ledger (independent read-only agent, 2026-10-07)

Scope: reports/ms/r2/{04_pilot,05_decision_inputs,summary}.md, the P2 sections of 03_checks.md and 00_housekeeping.md, the README row and the STATE.md section, against results/ms_r2/. Result: 208 rows, 170 OK, 8 WRONG, 21 OVERCLAIM, 9 UNSUPPORTED (6 / 14 / 5 distinct issues); all 48 script-generated tables byte-identical to their scripts.

Disposition: every WRONG number corrected (tail-mean range -0.0002..+0.0007; smoothing-part factor; C3 0.92; C1 0.011; 0.59; remainder SD 0.43-1.49); overclaims reworded (scope of the blind recomputation, "only gap interval", 2.08 check-to-check maximum, the "+1" does not explain the sigma excess (max 1.0024, found by the check), causal wording for the 400-update comparison, "did not follow" -> "followed only partly", C1 fire rows identical by construction); unsupported statements removed or qualified (load attribution, "as likely a feature of ten seeds"). Not added (thin against D4, data in results/ms_r2/analysis CSVs): middle/tail strata tables, tail max, R0/R/Delta/C along the run, S1 and stage-1 difference against rehearsal_v2_0, the figures are named not shown. Not checked by the agent: tmux use, load attribution, 365/128 test counts. The push status was pending at check time.

# MS-R2 fact-check ledger (independent, read-only)

Repository: `/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857`, HEAD `2227fc3a` (branch `ms-r2`), working tree with the uncommitted report files named in the task.
Method: (1) every table of `04_pilot.md`, `05_decision_inputs.md`, `summary.md` was parsed and compared byte for byte with the output of `pilot_tables.py --block NAME [--arm ARM]` / `stop_tables.py --dir results/ms_r2/stop_calibration_pilot --block NAME` run with the venv python (outputs and the comparison script are in the scratchpad `blocks/`, `blocks2/`, `cmp_tables.py`); (2) every prose number was compared with the CSV/JSON sources or recomputed from `per_run.csv`, `paired_secondary.csv`, `trajectory_checks.csv`, `predictions.csv`, `transmission*.csv`, `strata_summary.csv`, the `stop_calibration_pilot/tables/*.csv`, the launch JSONs and `git`; (3) the `sampler` and `window` blocks (computed by the script, not stored in a CSV) were additionally re-implemented independently from `per_run.csv` / `trajectory_checks.csv` (bootstrap with `default_rng(20261007)` per (q, statistic)) and agree to the printed digits.
Verdicts: OK / WRONG (a number or statement that does not match the source) / UNSUPPORTED (no source found, or not verifiable read-only) / OVERCLAIM (true in part, but stated more strongly or more generally than the data allow, a cause asserted for a co-variation, or inconsistent with another place in the reports).
Line numbers are those of the files as they are now. Where a paragraph is one line, all claims of the paragraph share the line number.

## A. Tables (script-generated: byte-identical check)

| id | file:line | table | source / command | verdict | note |
|---|---|---|---|---|---|
| T01 | 04_pilot.md:11 | launch checks | `pilot_tables.py --block checks` | OK | byte-identical |
| T02 | 04_pilot.md:23-26 (+ line 21 header) | launch bullets | `--block launch` | OK | byte-identical (line 25 is a 59,751-character line: the full list of 840 untracked `results/ms_r1/pilot/...` entries; correct but unreadable; presentation only) |
| T03 | 04_pilot.md:28 | item / value / path table | none (hand-written) | OK | no script produces it; content verified in P14-P17 below. Contradicts the claim of 04:3 that every table is script-printed (see P03) |
| T04 | 04_pilot.md:41 | primary | `--block primary` | OK | byte-identical |
| T05 | 04_pilot.md:56, 70, 88, 106, 120, 138 | overview rows of NL_bb_s1, NL_bb_s4, NL_bb_s16, NL_st_s1, NL_st_s4, NL_st_s16 | `--block overview --arm <arm>` | OK | byte-identical (6 tables) |
| T06 | 04_pilot.md:61, 75, 93, 111, 125, 143 | decomp rows of the six arms | `--block decomp --arm <arm>` | OK | byte-identical (6 tables) |
| T07 | 04_pilot.md:80, 98, 130, 148 | primary rows of NL_bb_s4, NL_bb_s16, NL_st_s4, NL_st_s16 | `--block primary --arm <arm>` | OK | byte-identical (4 tables) |
| T08 | 04_pilot.md:158, 163, 168 | overview rows MS_base2400, MS_s35a5, parents_A | `--block overview --arm <arm>` | OK | byte-identical (3 tables); `rehearsal_v2_0` rows are not printed (see P07) |
| T09 | 04_pilot.md:177 | decomp, all arms | `--block decomp` | OK | byte-identical |
| T10 | 04_pilot.md:200 | secondary | `--block secondary` | OK | byte-identical |
| T11 | 04_pilot.md:251 | transmission | `--block transmission` | OK | byte-identical |
| T12 | 04_pilot.md:268 | interaction | `--block interaction` | OK | byte-identical |
| T13 | 04_pilot.md:289 | sampler | `--block sampler` | OK | byte-identical; independently re-implemented from per_run.csv: all 36 intervals and counts agree |
| T14 | 04_pilot.md:316 | window | `--block window` | OK | byte-identical; independently re-implemented from trajectory_checks.csv (means of the 8 checks 2625-2800, paired differences, counts): agree |
| T15 | 04_pilot.md:343 | segments | `--block segments` | OK | byte-identical |
| T16 | 04_pilot.md:398 | trajectory | `--block trajectory` | OK | byte-identical |
| T17 | 04_pilot.md:417 | predictions | `--block predictions` | OK | byte-identical |
| T18 | 04_pilot.md:443 | vs_parents | `--block vs_parents` | OK | byte-identical |
| T19 | 04_pilot.md:458 | refs | `--block refs` | OK | byte-identical |
| T20 | 04_pilot.md:489 | strata_near | `--block strata_near` | OK | byte-identical |
| T21 | 04_pilot.md:514 | stage1 | `--block stage1` | OK | byte-identical |
| T22 | 04_pilot.md:539 | gates | `--block gates` | OK | byte-identical (the `nan` and `10.0` cells are printed as the script prints them) |
| T23 | 04_pilot.md:564 | shares | `--block shares` | OK | byte-identical |
| T24 | 04_pilot.md:583 | budget | `--block budget` | OK | byte-identical |
| T25 | 04_pilot.md:610 | "Where the numbers are" | none (hand-written) | OK | all cited files exist (23 CSVs, `figures/` with the five named PNGs, `blind_recomputation.txt`, `analysis_info.json`, `summary.txt`) (P70) |
| T26 | 05_decision_inputs.md:9 | side_by_side | `--block side_by_side` | OK | byte-identical |
| T27 | 05_decision_inputs.md:34 | directions | `--block directions` | OK | byte-identical |
| T28 | 05_decision_inputs.md:56 | tie_effort | `--block tie_effort` | OK | byte-identical |
| T29 | 05_decision_inputs.md:75 | transmission | `--block transmission` | OK | byte-identical |
| T30 | 05_decision_inputs.md:90 | sampler | `--block sampler` | OK | byte-identical |
| T31 | 05_decision_inputs.md:124-128 | facts bullets | `stop_tables.py --block facts` | OK | byte-identical |
| T32 | 05_decision_inputs.md:134 | spearman | `stop_tables.py --block spearman` | OK | byte-identical |
| T33 | 05_decision_inputs.md:151 | spearman_arm | `stop_tables.py --block spearman_arm` | OK | byte-identical |
| T34 | 05_decision_inputs.md:168 | freeze_arm | `stop_tables.py --block freeze_arm` | OK | byte-identical |
| T35 | 05_decision_inputs.md:189 | fire | `stop_tables.py --block fire` | OK | byte-identical |
| T36 | 05_decision_inputs.md:226 | fire_arm | `stop_tables.py --block fire_arm` | OK | byte-identical |
| T37 | summary.md:18 | primary | `--block primary` | OK | byte-identical |
| T38 | 03_checks.md:62-72 | launch/identity/analysis check table (section 5) | none (hand-written) | OK | content verified in C01-C09 |

Result: 48 script-generated table instances (04_pilot.md 37, 05_decision_inputs.md 10, summary.md 1) plus two bullet blocks (`launch`, `facts`), covering 29 distinct blocks (`strata_mid` and `strata_tail` are named but never printed), all byte-identical to the script output, 0 differences. Three hand-written tables (T03 = 04:28, T25 = 04:610, T38 = 03_checks.md:62) cannot be matched to a script. The line 04:483 starts with `|peak|` but is prose.

## B. 04_pilot.md prose and hand-written tables

| id | file:line | claim (short quote) | source checked | verdict | correct value / note |
|---|---|---|---|---|---|
| P01 | 04:3 | six arms, 20 runs per arm, q in {50,60}, seeds 10501-10510, terminal stage fixed 2800, stage 1 600 updates | per_run.csv (arm counts), budget.csv (2800 / 600 / 1,740,800 = 3400 x 512) | OK | |
| P02 | 04:3 | "recomputed independently by `r2_blind_criterion.py`, agreement of the 220 numbers" (said of everything read from `analysis/`) | blind_recomputation.txt last line: "ALL 220 numbers agree with criterion.csv and interaction.csv" | OVERCLAIM | the independent recomputation covers only `criterion.csv` and the `interaction.csv` summary (as 03_checks.md:71 says correctly); secondary, transmission, sampler, window, decomposition tables are not independently recomputed by that tool |
| P03 | 04:3 | "every table is printed by `pilot_tables.py --block <name>` (blocks named at each table)" | 04:28 and 04:610 tables are hand-written; §1 checks table (04:11) does not name its block `checks` | OVERCLAIM | two hand-written tables in 04 (and one in 03_checks.md §5) |
| P04 | 04:5 | signed peak error negative in all 120 runs (table overview, `signed<0`) | per_run.csv: 120/120 NL runs have stage2_peak_rel_err_signed < 0 (also all 80 reference rows) | OK | |
| P05 | 04:5 | "|peak error| equals the gap divided by e_2*(0)"; compare `gap % of e*(0)` with `mean abs(peak)` | per_run: max abs(gap/g2_at_0 - abs peak) = 1.0e-16 | OK | |
| P06 | 04:5 | e_2*(0) = 70.0 (q=50), 58.33 (q=60) | g2_at_0 | OK | 70.000, 58.333 |
| P07 | 04:5 | `parents_A` and `rehearsal_v2_0` "print identical rows below" | overview and strata rows identical (checked all 4 / all 8 rows); decomp (04:177) has only `rehearsal_v2_0`; stage1 (04:514) only `rehearsal_v2_0`; gates (04:539) differ (v2.0 combination `nan` vs `10.0`) | OVERCLAIM | identical only in `overview` and `strata`; `parents_A` has no stage 1 / decomposition rows and a `nan` in `gates` |
| P08 | 04:5 | C-MS1 of MS-R1 `03_checks.md` makes them the same terminal stage | reports/ms/r1/03_checks.md:22-33 (C-MS1 parentsA 20/20, rehearsal 20/20 identical) | OK | |
| P09 | 04:9 | tool `tools/ms/r2_launch_checks.py` "which the launch commit contains" | `git ls-tree 99857513`, HEAD 9b93b50c | OK | |
| P10 | 04:23 | wave r2, 120 planned, 120 finished, state done, workers 40, started 20261007_094801 | launch JSON | OK | |
| P11 | 04:24 | nonzero exits 0; wall min 555, median 620, max 711 s | JSON runs[].wall_sec: 555.09 / 620.02 (upper median 620.36) / 710.59 | OK | |
| P12 | 04:25 | HEAD 9b93b50c; code commit 99857513...; diff of run/utils/envs/agents/protocols empty | JSON head, code_commit, diff_stat_run_code_to_head = "" | OK | |
| P13 | 04:26 | SHA-256 0fbfc018...b406; nproc 64; load 14.8, 13.7, 13.6 -> 22.6, 42.2, 44.0; free disk 1014 GB | JSON params_sha256, nproc, loadavg_*, disk_free_bytes 1014330994688; `sha256sum reports/ms/r1/prereg_parameters.json` equals it | OK | |
| P14 | 04:30 | launch "2026-10-07 09:48:01, tmux, 40 single-threaded workers, 120 runs, state done" | JSON: started, workers 40, n_planned 120, state done; "tmux" appears in no JSON/manifest/config | UNSUPPORTED | tmux (and "single-threaded") is not recorded in any repository file; tmux server of this user not running (cannot confirm); everything else OK |
| P15 | 04:31 | `git diff --name-only 99857513 9b93b50c` lists 156 files, all under `results/ms_r2/`, none under run, utils, envs, agents, protocols, tools, tests | `git diff --name-only 99857513 9b93b50c` = 156 lines, 0 outside `results/ms_r2/` | OK | |
| P16 | 04:31 | `status_porcelain` holds 840 entries, all untracked under `results/ms_r1/pilot` | JSON status_porcelain: 840, all start with `?? results/ms_r1/pilot/` | OK | |
| P17a | 04:32 | host: 64 cores, load 14.8/13.7/13.6, 1.01e12 bytes free of 5.95e12 | JSON nproc 64, loadavg_at_start, disk_free_bytes 1.0143e12, disk_total_bytes 5.9523e12 | OK | |
| P17b | 04:32 | "(other users' work)" for the load average | not recorded in any repository file | UNSUPPORTED | minor attribution |
| P18 | 04:33 | no infrastructure kill, no non-zero exit, `results/ms_r2/pilot/crashed/` does not exist, no run re-run | `ls` (absent), JSON returncodes all 0, 120 run dirs | OK | |
| P19 | 04:35 | "No check failed and no stop condition was met" | launch_checks.json all_ok true | OK | |
| P20 | 04:48 | part (a) fails in all eight cells: no interval entirely below 0 | criterion.csv: all ci_hi >= 0 | OK | |
| P21 | 04:48 | NL_bb_s4 q=60 interval entirely above 0; |peak| larger under s=4 in 9 of 10 seeds; signed -0.0114 [-0.0185, -0.0034] | paired_secondary: abs peak n_neg = 1/10; signed row | OK | |
| P22 | 04:48 | part (b) holds in the trivial sense: all 120 runs pass G-A and G-N(eta) | per_run gate_pass True x120; gates.csv | OK | |
| P23 | 04:48 | five of eight means negative (bb_s4 q50, bb_s16 q50, st_s4 q60, st_s16 both), three positive (bb_s4 q60, bb_s16 q60, st_s4 q50) | criterion.csv means | OK | |
| P24 | 04:48 | interval half-widths 0.008-0.017; smoothing reduction alone 0.011-0.020 of e*(0) | half-widths 0.00757-0.01663; smoothing_rel changes -0.0110 to -0.0199 | OK | |
| P25 | 04:66 | NL_bb_s1: 5/10 and 8/10 within 0.05; gap 5.65/4.01 %, smoothing 2.72/2.30 %, remainder 2.93/1.71 %; remainder SD 1.47/0.88; vs MS_base2400 +0.0035 [-0.0166,+0.0269], -0.0048 [-0.0200,+0.0088] | per_run, refs table | OK | |
| P26 | 04:66 | "constant LR 3e-4 to update 2400, then linear decay to 3e-5" | prompt D2; trajectory_checks lr column (2400: 3.0e-4; 2600: 1.65e-4; 2800: 3.0e-5) | OK | |
| P27 | 04:84 | sigma 2.414/2.450 -> 1.255/1.284; smoothing 1.905/1.343 -> 0.990/0.703 (10/10 seeds lower); remainder 2.053/0.995 -> 2.574/2.297; gap -0.394 [-1.631,+0.697] and +0.663 [+0.196,+1.078]; 5/10, 4/10 within 0.05; R0, R changes | decomp, secondary | OK | |
| P28 | 04:84 | "+0.663 ... the only gap interval in the round that excludes 0" | other gap intervals that exclude 0 in the same report: 04:289 sampler (s=4 q=60 [-1.104,-0.218]; s=16 q=50 [-2.176,-0.042]) and 04:316 window (NL_bb_s16 q=60 +0.761 [+0.387,+1.208], which 04:335 itself names) | OVERCLAIM | true only for the pre-registered secondary table against s=1 (04:200); contradicts 04:310/04:335 |
| P29 | 04:102 | NL_bb_s16 numbers (0.651/0.665; 0.514/0.365; -1.392/-0.978; +1.331 [+0.468,+2.102], +1.395 [+0.768,+1.987]; 3/10, 1/10 lower; gap 3.897/2.755, -0.061/+0.417; 3/10, 6/10 within 0.05; tail +0.0005, +0.0006, 2/10, 3/10) | decomp, secondary, overview | OK | |
| P30 | 04:116 | NL_st_s1: 7/10, 6/10; gap 4.34/4.61 %, smoothing 2.68/2.29 %, remainder 1.65/2.32 %; vs MS_s35a5 +0.0091 [-0.0099,+0.0277], +0.0027 [-0.0076,+0.0130], remainder +0.728, +0.227 | decomp, refs | OK | lambda_P 0.35 / alpha_global 0.5 consistent with `shares` design column |
| P31 | 04:134 | NL_st_s4: sigma 1.240/1.271; smoothing 0.978/0.697 (-0.899, -0.640); remainder 2.150/1.651 (+0.992 [-0.005,+2.013], +0.296 [-0.389,+0.950]); gap 3.128/2.348 (+0.094, -0.344); 7/10, 9/10; q=60 RMSE -0.0040 [-0.0064,-0.0019], eta -0.0002 [-0.0004,-0.0001], R -0.012 [-0.021,-0.005], tail +0.0006 [+0.0002,+0.0010] | decomp, secondary | OK | |
| P32 | 04:152 | NL_st_s16: sigma 0.641/0.655; smoothing 0.506/0.359 (-1.371,-0.978); remainder 2.303/2.064 (+1.145 [+0.371,+1.914], +0.709 [-0.147,+1.556]); gap 2.809/2.422 (-0.226,-0.269); 8/10, 7/10; RMSE -0.0044 [-0.0083,-0.0006], -0.0054 [-0.0085,-0.0028]; eta -0.0005 [-0.0010,-0.0001] q50; R -0.022 [-0.044,-0.001] q50; tail +0.0007 [+0.0002,+0.0013] q60 | decomp, secondary | OK | |
| P33 | 04:156 | 3.7 "block overview and decomp" | only `overview` rows are printed in 3.7 (decomp rows of the references are in 4.1) | OK | wording imprecise only |
| P34 | 04:249 | ratio = mean(d gap)/mean(d smoothing); denominator negative in all eight rows (smoothing fell in every pair) | transmission_seed_level.csv: d_smoothing < 0 in 80/80 pairs | OK | |
| P35 | 04:262 | eight estimates between -1.04 and +0.54; no ratio near 1; interval excludes 0 only for NL_bb_s4 q=60 (-1.037 [-1.704,-0.300]); per-seed ratios swing in sign in every row | transmission.csv (-1.0367 .. +0.5371; only that row has ci_hi<0); seed-level neg/pos counts 6/4, 9/1, 6/4, 7/3, 6/4, 4/6, 4/6, 5/5 | OK | |
| P36 | 04:283 | only s=4 q=60 excludes 0: |peak| -0.0173 [-0.0324,-0.0033], remainder -1.006 [-1.884,-0.202]; paired changes +0.0114 and -0.0059 ([-0.0176,+0.0053]); smoothing interactions +0.016, -0.000, +0.020, +0.000, all contain 0 | interaction.csv, secondary | OK | -0.0059 - 0.0114 = -0.0173 |
| P37 | 04:310 | smoothing does not differ between samplers (six intervals contain 0, |mean| <= 0.028) | sampler block (-0.028 max) | OK | |
| P38 | 04:310 | remainder and gap lower under st in 5 of 6 (s,q) cells; |peak| lower in 5 of 6 (exception s=1 q=60); |peak| interval excludes 0 in two cells (s=4 q=60 -0.0112 [-0.0189,-0.0037]; s=16 q=50 -0.0156 [-0.0311,-0.0006]); remainder in two (same cells); RMSE lower at s=16 at both q (excl. 0); tail mean not lower at any s (+0.0003..+0.0010; +0.0005, +0.0007; +0.0006, +0.0004) | sampler block, independent recompute | OK | |
| P39 | 04:314 | checks 2625, 2650, ..., 2800; LR falls from 1.5e-4 to 3e-5 over them | trajectory_checks: 8 checks per run; lr(2625) = 1.48e-4, lr(2800) = 3.0e-5 | OK | |
| P40 | 04:335 | smoothing lower in every pair in both tables; remainder higher in all eight cells of both tables, interval excludes 0 in six of eight (window) / four of eight (freeze); gap and |peak| contain 0 in 7 of 8 cells, exception NL_bb_s16 q=60 (+0.761 [+0.387,+1.208]; +0.0125 [+0.0058,+0.0201]) | window block, secondary | OK | |
| P41 | 04:341 | segment columns: means over updates and seeds; scale [min,max]; last four columns = seed means at last check of the segment | segments.csv | OK | |
| P42 | 04:394 | KL, clip fraction up and advantage SD down with s: hold bb q50 KL 0.0083/0.0150/0.0344, clip 0.077/0.112/0.171, adv 0.043/0.022/0.012; decay KL 0.0061/0.0091/0.0166; "same order at q=60 and for the stratified arms"; st q50 hold 0.0083/0.0133/0.0306, 0.073/0.105/0.159, 0.051/0.026/0.014 | segments.csv: KL and clip strictly increasing and adv SD strictly decreasing in s in all 12 (sampler, q, segment in ramp/hold/decay) combinations; training rows identical across s | OK | "training" rows identical through update 2000, consistent with C-NL |
| P43 | 04:394 | "it does not say that this causes the remainder" | text only (no data claim) | OK | appropriately hedged |
| P44 | 04:413 | NL_bb_s1 q=50 seed-mean remainder 1.66, 3.35, 1.44, 2.58, 2.03, 2.05 at 1800...2800; 1800 and 2000 values common to all three bb arms | trajectory table | OK | |
| P45 | 04:413 | "check-to-check variation ... at constant LR ... up to 1.9 effort units between two checks of the same arm" | trajectory_checks.csv: max over the six listed updates 1.91 (NL_bb_s1 q50, 2000->2200); max over all consecutive stored checks (25 apart) at constant LR 2.08 (NL_bb_s1 q=60, update 2150->2175); the listed 2600 and 2800 values are in the decay segment (LR 1.65e-4, 3e-5) | OVERCLAIM | "up to 1.9" holds only for the six listed checks; across all stored checks it is 2.08 (same sentence at 04:605 and 05:270) |
| P46 | 04:413 | freeze-level changes "0.3-1.4 effort units" | remainder changes +0.296..+1.395 | OK | |
| P47 | 04:432 | mean sigma ratios 1.040/1.048, 1.078/1.086, 1.042/1.043, 1.077/1.074; per-run min 1.013, max 1.102; "within 1.3-10.2 % per run"; larger than predicted in every run | predictions.csv; independent per-run ratios (min 1.0129, max 1.1023; all > 1) | OK | |
| P48 | 04:432 | "The Beta standard deviation is e_range sqrt(mu(1-mu)/(c s+1)) ..., so a ratio above 1 for any finite concentration c is what the formula gives; the numerical size of the excess was not checked against the c of the runs" | 01_decomposition.md:31 and decomposition_beta_recompute.csv: alpha+beta of the runs = 198.5-426.6; sqrt(s(c+1)/(sc+1)) = 1.0007-1.0019 (s=4), 1.0011-1.0024 (s=16) | OVERCLAIM | sign is right, but the "+1" can account for at most 0.24 % against the observed 1.3-10.2 % (min per-run 1.013); the size check is a one-line calculation with data already in the repository and it does not support the "+1" as the explanation of the size (same implication in 04:601) |
| P49 | 04:433 | smoothing/formula mean ratio 0.99917-0.99946 in every row; 0 of 120 outside 0.5 %; per-run min 0.99917, max 0.99952 | predictions.csv; per_run: min 0.9991716, max 0.9995155 | OK | |
| P50 | 04:434 | floors: 1.41/0.73 (q50 s=4/16 bb), 1.21/0.63 (q60); planning 1.41/0.71 and 1.18/0.59; s=1: 2.72 % bb and 2.68 % st (q50), 2.30 % and 2.29 % (q60), planning 2.82/2.35 (sigma 2.5) | predictions.csv (1.4150, 1.2060, 0.7340, 0.6250 -> 0.63 (0.625044)) | OK | |
| P51 | 04:435 | remainder grew in every cell (+0.296..+1.395; 1 to 4 of 10 seeds lower); interval excludes 0 in four (bb_s4 q60, bb_s16 q50 and q60, st_s16 q50); transmission -104 % .. +54 %; remainder offset at least 46 % | paired_secondary, transmission (1 - 0.537 = 0.463) | OK | quoted D5 sentence matches the prompt verbatim |
| P52 | 04:441 | parents_A terminal stage 1600 updates (1200 + 400 decay) | reports/ms/r1/04_pilot.md:5; per_run t2_updates 1600 | OK | |
| P53 | 04:452 | "met": NL_st_s4, NL_st_s16; NL_st_s1 not met (q60 -0.00721 [-0.02618,+0.01050]); -0.0222 at q=50 excludes 0; upper bounds -0.00048 and -0.00032; point estimates -0.0131, -0.0118 inside NL_st_s1's interval (-0.0072 [-0.0262,+0.0105]); bb rows: NL_bb_s1 met at q=60 only, NL_bb_s4 q=50 only, NL_bb_s16 neither | criterion_vs_parents_A.csv | OK | the 4-decimal interval is the rounding of the 5-decimal table cell |
| P54 | 04:452 | "the 'met' status ... is therefore not evidence that the noise landing helped: the comparison confounds budget, sampler and scale" | criterion_vs_parents_A.csv, per_run t2_updates (1600 vs 2800) | OK | hedged correctly |
| P55 | 04:456 | NL_st_s1 vs MS_s35a5 "in the words of D5, the effect of 400 more constant-LR updates"; MS_s35a5 terminal stage ran 2400 updates | per_run: MS_s35a5 t2_updates 2400, t2_n_updates_landing 400, 14 of 20 runs have polishing blocks (28 in total); MS_base2400 2400, no landing | OK | the quote of D5 is accurate; the text does say NL_st_s1 has no stop/no polishing, but not that MS_s35a5 polished in 14 of 20 runs (see P56) |
| P56 | 04:483 | "a slow additional contraction of the policy noise and of the tail error during the longer constant-LR phase" (and sigma lower by 0.117-0.139, smoothing by 0.070-0.094, tail by 0.0005-0.0008, 7-9 of 10 seeds; |peak| intervals contain 0; RMSE significant only for NL_bb_s1 q60 -0.0042 [-0.0069,-0.0014]) | refs table; numbers all OK | OVERCLAIM | numbers OK, but the cause is asserted: the two pairs also differ in the LR schedule (decay in 2002-2400 vs 2401-2800) and, for the stratified pair, in 400-update landing and polishing blocks in 14/20 MS_s35a5 runs; the data show co-variation with the arm change, not the attribution to "the longer constant-LR phase". Same statement at 05:116 |
| P57 | 04:487 | heading: "all three [strata] in `results/ms_r2/analysis/strata.csv`" | strata.csv has near/mid/tail; the blocks read `strata_summary.csv` | OK | minor: the tables are produced from `strata_summary.csv`, not `strata.csv` |
| P58 | 04:535 | all stage-1 gates 10/10 in every row; stage-1 signed interval excludes 0 only for rehearsal_v2_0 q=50 (-0.0118 [-0.0224,-0.0014]) | stage1.csv | OK | |
| P59 | 04:535 | stage-1 columns differ across arms "because stage 1 is trained against each run's own frozen terminal-stage policy" | design (stage-1 table built from the frozen stage-2 mean) | OK | design statement, not tested |
| P60 | 04:579 | measured start shares within 3 binomial SE in all 360 tests (section 1) | launch_checks.json: 360 tests, 0 flagged | OK | the `design lambda_P / lambda_M` columns of 04:564 are parameters, not expected shares, so the agreement cannot be read off that table |
| P61 | 04:579 | clamp share falls with s: bb q50 0.0199/0.0147/0.0128 | shares block (bb q60 .0059/.0037/.0036; st q50 .0066/.0043/.0041; st q60 .0013/.0007/.0007) | OK | non-increasing in all four rows (equal at s=4,16 for st q60) |
| P62 | 04:600 | item 1: |peak| rose under s=4 bb q=60 (+0.0114 [+0.0034,+0.0185], 9/10), not under s=16 (+0.0072 [-0.0038,+0.0174]); window table opposite order (s=4 +0.0050 [-0.0002,+0.0108]; s=16 +0.0125 [+0.0058,+0.0201]) | secondary, window | OK | |
| P63 | 04:600 | "The non-monotone pattern in s is as likely a feature of ten seeds as of the mechanism" | no analysis of likelihood anywhere | UNSUPPORTED | a judgement of equal likelihood has no data behind it; the supported statement is that the window table has the opposite order |
| P64 | 04:601 | item 2: sigma above the 1/sqrt(s) prediction in every run, "by an amount the formula's '+1' accounts for in sign; the size was not checked" | see P48 | OVERCLAIM | in sign yes; in size not (max 1.0024 vs min observed 1.013) |
| P65 | 04:602 | item 3: "`tail mean / e_2*(0)` +0.0001 to +0.0007; intervals exclude 0 in four of the eight cells: NL_bb_s16 both q, NL_st_s4 and NL_st_s16 at q = 60; at most 0.07 % of e*(0)" | secondary: tail-mean changes +0.0003, +0.0002, +0.0005, +0.0006, **-0.0002** (NL_st_s4 q50: -0.0002 [-0.0006,+0.0001]), +0.0006, +0.0001, +0.0007; starred: bb_s16 q50, bb_s16 q60, st_s4 q60, st_s16 q60 | WRONG | range is -0.0002 to +0.0007 (7 of 8 positive; 05:34 table says "-0.0002 to +0.0007", and "rose slightly with s" is then 7 of 8 cells); the four starred cells and "at most 0.07 %" are right |
| P66 | 04:603 | item 4: PPO diagnostics move with s; LR not varied at fixed s | segments | OK | |
| P67 | 04:604 | item 5: NL_st_s4 q=50 d<0 middle stratum -1.152, largest in magnitude of the 20 values; next +1.106 (MS_s35a5 q50) | strata_summary final/mid/d<0: 20 rows; |-1.152|, |+1.106|, |-0.576|, |+0.477| | OK | |
| P68 | 04:605 | item 6: check-to-check variation up to 1.9 | see P45 | OVERCLAIM | 2.08 over all stored consecutive checks |
| P69 | 04:606 | item 7: no run lost/re-run/flagged; criterion (b) cannot fail | completeness.csv (n_done 10 per arm and q), launch JSON, gates.csv | OK | |
| P70 | 04:610 | "Where the numbers are": 22 files + 5 figures + blind_recomputation.txt + analysis_info.json + summary.txt | all exist in `results/ms_r2/analysis/` (figures: the five named PNGs) | OK | |

## C. 05_decision_inputs.md

| id | file:line | claim (short quote) | source checked | verdict | correct value / note |
|---|---|---|---|---|---|
| D01 | 05:3 | blocks used; "selects nothing and proposes no protocol" | grep of 04/05/summary for recommend/should/suggest/propose/prefer: none found | OK | descriptive throughout |
| D02 | 05:7 | `rehearsal_v2_0` stands for `parents_A`, its row has the decomposition | T26; C-MS1 | OK | |
| D03 | 05:48 | sigma and smoothing fell in all 8 cells, 10/10 seeds; sigma by 1.14-1.78, smoothing 0.64-1.39 (1.1-2.0 pct points); ratio 1.01-1.10 per run; smoothing = formula to 0.083 % at worst | directions (-1.7844..-1.1382; -1.3916..-0.6395); per_run 1-0.9991716 = 0.0828 % | OK | |
| D04 | 05:49 | remainder rose in every cell (+0.30..+1.40); four intervals above 0, none below; window: six of eight | directions, window | OK | |
| D05 | 05:50 | gap -0.39..+0.66, 5 of 8 negative, none below, one above; |peak| -0.0059..+0.0114 same signs | directions | OK | |
| D06 | 05:50 | "e_sigma(0) rose by 0.64-1.39 ... while the mean e_hat_2(0) changed by -0.66 to +0.39 (change of e_hat = minus change of gap)" | secondary (gap -0.394..+0.663); e* fixed | OK | |
| D07 | 05:50 | "The tie effort of the learned policy did not follow the target upward" | in 5 of 8 cells the seed-mean e_hat_2(0) rose (by 0.06-0.39 effort units, i.e. 4 %-54 % of the target's rise: the positive transmission ratios +0.044, +0.165, +0.275, +0.431, +0.537); in the other 3 it fell (-0.09, -0.42, -0.66); no gap interval is below 0, one is above | OVERCLAIM | "did not follow" (same as 05:54 title and summary.md:25) is a summary of: followed partially in 5 cells, moved the wrong way in 3, nothing distinguishable from 0 except the one cell where the gap rose |
| D08 | 05:51 | RMSE_pos lower in 7 of 8 (3 intervals below 0, none above, up to -0.0054, level 0.014-0.020); eta_2 7 of 8 (2 below); R 7 of 8 (3 below) | directions; overview means 0.0142-0.0199 | OK | |
| D09 | 05:52 | tail mean rose slightly (4 intervals above 0, none below; at most +0.0007) | directions (1 of 8 negative, 4 above) | OK | |
| D10 | 05:71 | e_hat changes "+0.394, -0.663 (bb s=4), +0.061, -0.417 (s=16), -0.094, +0.344 (st s=4), +0.226, +0.269 (s=16)"; smoothing changes 0.915, 0.639, 1.392, 0.978; 0.899, 0.640, 1.371, 0.978 | secondary (minus gap / minus smoothing) | OK | |
| D11 | 05:71 | "The seed SD of e_hat_2(0) at one arm and q (0.45-1.56) is as large as or larger than these moves" | per_run SDs: 1.563, 0.913, 1.515, 0.447, 1.371, 0.709, 1.465, 1.010, 0.774, 0.498, 0.752, 0.731; the 0.45-1.56 range is right; NL_bb_s4 q=60: arm SD 0.447 < move 0.663 (paired SD of the difference 0.760) | OVERCLAIM | not true for NL_bb_s4 at q=60, the one cell whose gap interval excludes 0 |
| D12 | 05:71 | SD lower at s>1 in every sampler and q: st 1.47 -> 0.77, 0.75 (q50), 1.01 -> 0.50, 0.73 (q60); bb 0.91 -> 0.45, 0.71 (q60); q50 1.56 -> 1.51, 1.37 | per_run | OK | |
| D13 | 05:86 | ratios between -1.04 and +0.54, none near 1, only NL_bb_s4 q60 excludes 0; interaction on remainder negative in 3 of 4 cells, positive at s=4 q=50; only s=4 q=60 excludes 0 (-1.006 [-1.884,-0.202]) | transmission, interaction | OK | |
| D14 | 05:111 | "At every s the smoothing part is the same under both samplers" | sampler: differences -0.028..-0.006 effort units, intervals contain 0 | OVERCLAIM | minor: "does not differ detectably (|mean| <= 0.028)" |
| D15 | 05:111 | remainder, gap, |peak| lower under st in 5 of 6 cells, intervals exclude 0 in two cells each (|peak|: s=4 q60, s=16 q50); RMSE lower at s=16 both q; s=1 no |peak| interval excludes 0 (-0.0132 [-0.0301,+0.0036], +0.0061 [-0.0090,+0.0231]); tail not lower at any s (+0.0003..+0.0010) | sampler | OK | |
| D16 | 05:115 | `parents_A` criterion met for NL_st_s4 and NL_st_s16 only; NL_st_s1 not met | criterion_vs_parents_A | OK | |
| D17 | 05:116 | s=1 arms vs refs: no |peak| change; sigma lower by 0.12-0.14, smoothing 0.07-0.09, tail lower (four intervals below 0) | refs (0.117-0.139; 0.070-0.094; four starred) | OK | |
| D18 | 05:116 | "400 more updates at constant LR contracted the policy noise and the tail error and did not move the tie" | see P56; "did not move the tie" rests on |peak|/gap intervals containing 0 with 10 seeds, which 05:271 itself says does not show absence | OVERCLAIM | same issue as P56 |
| D19 | 05:120 | replay matched every logged check row: 13,440 valid exports, max abs difference 0 | validation_summary.csv: 120 rows, n_exports 112 each, n_matched 112, max_abs_d3 = max_abs_closed = 0, mismatches 0; manifest n_rows 13440 | OK | |
| D20 | 05:120 | thresholds = grids recorded before the MS-R1 fire tables (`stop_candidate_grids.json`, commit 93d8be5d), unchanged | file committed 93d8be5d (07:53:10), fire.csv written 07:53:15; no later commit; thresholds in fire.csv equal the grids | OK | |
| D21 | 05:125-128 | R0/|peak| medians 0.603 / 0.685, 5-95th [0.588,0.617], [0.634,0.698]; |c2| < floor in 14 and 13 runs; c2 < 0 in 4 and 1 runs | facts block (T31); independent recompute from per_run: 14, 13, 4, 1 | OK | "in 1 runs" typo only |
| D22 | 05:130 | same medians as MS-R1 (0.603, 0.685); "C1 is |peak| rescaled, whatever s" | 01b_stop_candidates.md:41 (0.603, 0.685); 5-95th percentiles [0.588, 0.617] (q50) and [0.634, 0.698] (q60), i.e. -7 % / +2 % around the median at q=60 | OK | "rescaled" is approximate at q=60 (0.634-0.698) |
| D23 | 05:149 | C1 0.991 at both q "unchanged"; MS-R1: C2 0.891/0.889, C3 0.856/0.675, signed C2 0.984/0.988; pilot C2 0.862/0.875, C3 0.876/0.713, signed C2 0.911/0.939, R 0.093/0.141 | 01b_stop_candidates.md:29-41 and T32 | OK | |
| D24 | 05:149 | "the smoothing part varies by a factor of about five between arms (0.36 to 1.91 effort units at the freeze, s = 16 to s = 1)" | arm-mean smoothing: min 0.359 (NL_st_s16, q=60), max 1.905 (NL_bb_s1, q=50): the two ends come from different q, and the Spearman correlations are pooled per q. Within q=50: 0.506-1.905 (x3.77); within q=60: 0.359-1.343 (x3.74); per-run within q: x4.8 (q50), x4.3 (q60) | WRONG | within each q the arm means differ by a factor of about 3.7-3.8 (4.3-4.8 per run), not "about five" |
| D25 | 05:149 | per-arm Spearman values "0.74-0.94 for C2, 0.62-0.93 for C3, 0.98-1.00 for C1" | a_spearman.csv per-arm |peak|: C2 0.7357-0.9420, C3 0.6238-**0.9246**, C1 0.9835-0.9957 | WRONG | C3 upper end is 0.92 (0.9246; "0.93" is a double rounding of 0.925); C2 and C1 ranges are right |
| D26 | 05:183 | floor 0.027, 0.014, 0.007 (bb q50); |c2| medians rise (bb q50 0.0223/0.0369/0.0474; q60 0.0139/0.0416/0.0417; st q50 0.0111/0.0297/0.0360; q60 0.0205/0.0300/0.0379); s=1 |c2| below floor in all four cells; at s=16 ratio of medians 5.1-6.7 | b_freeze.csv: 5.108, 6.682, 6.610, 6.055; s=1 cells 0.0223<0.0266, 0.0139<0.0230, 0.0111<0.0264, 0.0205<0.0232 | OK | |
| D27 | 05:183 | C1 medians at s=1,4,16 listed; "at most 0.012 between two s" | b_freeze.csv medians: bb q60 0.02467, 0.03617, 0.03252 -> max spread 0.011494 | WRONG | 0.011 (0.0115 from the 4-decimal values, 0.01149 exact): double rounding |
| D28 | 05:183 | "This is the criterion-level statement of the remainder's growth: C2 measures it, C1 and |peak| do not distinguish it from the floor" | definitions (c2 = remainder / e_sigma(0)) | OK | interpretation, follows from the definitions |
| D29 | 05:265 | C1 at theta 0.03: fires 8-10 of 10 in every arm, median updates 875-1038, |peak| at fire 0.028-0.033, at end 0.036-0.054 | fire_arm table | OK | |
| D30 | 05:265 | "the fire statistics of C1 do not depend on s" | C1 fires (medians 875-1038) occur before the landing; by C-NL the three arms of a sampler are bit-identical through update 2001, so the rows for s = 1, 4, 16 of a sampler are identical by construction (st arms: 875/1038 in all three s) | OVERCLAIM | true by construction for fires before update 2002, not evidence; the report does not say so |
| D31 | 05:265 | C2 theta 0.01 fires 3-9 of 10; counts bb q50 5,3,4, st q50 8,9,8 "because c2 ... grows with s" | fire_arm: counts 5,7 / 3,7 / 4,7 / 8,7 / 9,6 / 8,5 (s=1,4,16; q=50,60) | OVERCLAIM | ranges and quoted counts right; "the number moves with s ... because c2 grows with s" asserts a cause and does not hold for bb q60 (7,7,7), st q60 (7,6,5 monotone but one cell) or st q50 (8,9,8) |
| D32 | 05:265 | C3 theta 0.05 fires 2-10 of 10, medians 1588-2575 | fire_arm | OK | |
| D33 | 05:269 | KL 0.0061, 0.0091, 0.0166; clip 0.057, 0.078, 0.114; adv SD 0.042, 0.022, 0.012 (decay segment, bb q50) | segments | OK | |
| D34 | 05:270 | s=16 seed-mean remainder 2400 -> 2800 fell by 0.15-0.32 in three arms (bb q60 2.54 -> 2.39; st q50 2.46 -> 2.30; st q60 2.39 -> 2.06) | trajectory_checks: -0.1534, -0.1548, -0.3236 | OK | |
| D35 | 05:270 | "... and rose by 0.58 in the fourth (bin-balanced q = 50: 2.80 to 3.38)" | +0.5851 (2.7984 -> 3.3835) | WRONG | 0.59 (rounding; 0.585 from the printed 3-decimal numbers rounds either way, the exact value is 0.5851) |
| D36 | 05:270 | "the seed SD of the remainder (0.51-1.49 across arms)" | per_run seed SDs of the remainder, 12 NL cells: 1.471, 0.875, 1.486, 0.427, 1.342, 0.707, 1.478, 1.032, 0.765, 0.510, 0.761, 0.735 | WRONG | range is 0.43-1.49 (min 0.427, NL_bb_s4 q=60; 0.510 is NL_st_s4 q=60); same SD column in T09 |
| D37 | 05:271 | half-widths 0.008-0.017 vs 0.011-0.020 | P24 | OK | |
| D38 | 05:272 | `sampler` block lists 36 intervals; none pre-registered | 3 s x 6 metrics x 2 q = 36 | OK | |
| D39 | 05:273 | NL_st_s4, NL_st_s16 "met" status shared with NL_st_s1 at q=50 and differs at q=60 by an amount inside NL_st_s1's interval | criterion_vs_parents_A | OK | |
| D40 | 05:274-275 | stage 1 gates pass in all 120 runs; fresh seeds, lock, T=3, second pilot not run | gates.csv; remote refs (no new tag, `main` unchanged) | OK | |

## D. summary.md

| id | file:line | claim (short quote) | source checked | verdict | correct value / note |
|---|---|---|---|---|---|
| S01 | summary:3 | branch from `origin/ms-r1` 71c58904; STOP after the push of section 4.3; no fresh seed/lock/T=3/second pilot; no parameter changed after the pre-registration; main and tags not touched | `git ls-remote`: ms-r1 71c58904, main 4dc604de (unchanged), no ms tag; prereg file unchanged since 99857513; params SHA unchanged | OK | the push of 4.3 has not happened yet (see ST01) |
| S02 | summary:5 | "stage-2 peak bias (the learned tie effort below the closed form in every run)" | MS-R2: 120/120 (and 80/80 references) negative; MS-R1: 139 of 140 (`01_decomposition.md:24`; MS_s35a0 q60 seed 10501 has +0.000143) | OVERCLAIM | minor: true for this round's data, 139 of 140 for the MS-R1 runs the question refers to |
| S03 | summary:5 | landing schedule: scale 1 to s over 2001-2200, hold to 2400, LR 3e-4 to 3e-5 over 2401-2800 | prompt D2; trajectory_checks | OK | |
| S04 | summary:15 | 120/120 exit 0, manifests/clean, scale 120/120, C-NL 80/80, C-MS3 20/20, C-MS4 20/20, 360 tests none flagged, all runs pass G-A and G-N(eta) | launch_checks.json, per_run | OK | |
| S05 | summary:16 | primary criterion not met in any row; part (a) fails in all eight cells; NL_bb_s4 q60 above 0 (+0.0114 [+0.0034,+0.0185]); (b) holds; table | criterion.csv | OK | |
| S06 | summary:25 | sigma fell by 1.14-1.78, smoothing by 0.64-1.39 (1.1-2.0 pct points), all eight cells, 10/10; sigma(s) 1.01-1.10 times prediction in every run; smoothing = formula to 0.083 % at worst; remainder +0.30..+1.40, four intervals above 0, none below; gap -0.39..+0.66; fraction reaching the gap -104 %..+54 % | see D03-D05, P47-P51 | OK | |
| S07 | summary:25 | "The smoothing part fell as predicted; the remainder rose by about as much." | per-cell (remainder rise)/(smoothing fall): 0.57, 2.04, 0.96, 1.43, 1.10, 0.46, 0.84, 0.73; aggregate (sum of rises / sum of falls) 0.98 | OVERCLAIM | holds in aggregate (98 %) and for two cells; per cell the remainder offset 46 %-204 % of the smoothing reduction (04:435 states "at least 46 %" correctly) |
| S08 | summary:25 | "The mean tie effort of the learned policy did not follow the sharper target" | see D07 | OVERCLAIM | same as D07 |
| S09 | summary:26 | RMSE_pos, eta_2, R lower in 7 of 8 cells (intervals below 0 in 3, 2, 3); tail mean rose (four intervals above 0, at most +0.0007) | directions | OK | |
| S10 | summary:27 | stratified arms lower remainder, gap, |peak| in 5 of 6 cells, two with intervals excluding 0; interaction excludes 0 only at s=4 q=60 (-0.0173 [-0.0324,-0.0033]) | sampler, interaction | OK | |
| S11 | summary:28 | NL_st_s4, NL_st_s16 "met" vs `parents_A`; NL_st_s1 not; confounded | criterion_vs_parents_A | OK | |
| S12 | summary:29 | C1 ranks |peak| as before (0.991); C2 floor falls with s while |c2| rises (5.1-6.7 times at s=16); no threshold selected | T32, D26, 01b table | OK | |
| S13 | summary:33 | half-widths 0.008-0.017; change 0.011-0.020; "does not say why the remainder grows" | P24 | OK | |
| S14 | summary:37-41 | deviations 1-5 (flag moved to a constructor argument, C-MS4 `p_digest_next` MAJOR fixed, transmission sign decision 10, report supplements, local refs behind) | 03_checks.md:1 and :32-40, 02_preregistration.md:98 (decision 10), 00_housekeeping.md:1.1 | OK | |
| S15 | summary:47-52 | commands equal `analysis_info.json` command | analysis_info.json "command" | OK | |
| S16 | summary:11 | `pi_record/01_factcheck.md` named as an existing ledger | file does not exist yet | UNSUPPORTED | expected (it is to be created from this ledger); same in README.md and STATE.md |

## E. 00_housekeeping.md section 2, 03_checks.md section 5, README.md, STATE.md

| id | file:line | claim (short quote) | source checked | verdict | correct value / note |
|---|---|---|---|---|---|
| H01 | 00:49 | launch command (`--wave r2 --params ... --workers 40 --code-commit 99857513...`) from HEAD 9b93b50c | JSON argv | OK | |
| H02 | 00:49 | "tmux session `ms_r2_pilot`" | no repository record | UNSUPPORTED | see P14 |
| H03 | 00:49 | 64 cores, load 14.8/13.7/13.6, 1.01e12 of 5.95e12 free, empty diff, 840 untracked, state done, 120/120, no non-zero exit; each run 555-711 s | JSON | OK | |
| H04 | 00:49 | "Nothing was changed between the launch and the end of the pilot (a clean tree outside `results/`)" | launch_checks manifest `clean_tree` true for 120/120 runs; no commit between 9b93b50c (09:47:41) and 75533f2e (10:25:44) | OK | |
| H05 | 00:50 | roots used recorded in `analysis_info.json`; `parents_A`/`rehearsal_v2_0` in `.claude/worktrees/v2-t2-refine/results/`; calibration `results/ms_r1/calibration`; `MS_base` not needed | analysis_info.json command; paths exist | OK | |
| H06 | 00:51 | three commits: 75533f2e (13 files per run x 120 + launch record + launch_checks.json = 1562 files, 235 MB), a37cdd58 (`analysis/`), 2227fc3a (`stop_calibration_pilot/`); untracked: .pt, train_history.json, weights, freeze npz, band_sweep.npz, run.log | `git show --stat`: 1562 files; `git ls-tree -l` sum 235,432,806 bytes; 13 distinct files per run; a37cdd58 34 files in analysis/; 2227fc3a 132 files in stop_calibration_pilot/ | OK | |
| H07 | 00:52 | departures: none beyond summary.md | summary.md:37-41, prompt D1-D8 vs launch record | OK | |
| H08 | 00:53 | main, tags not touched | `git ls-remote`: main 4dc604de, no new tag | OK | |
| C01 | 03:60 | launched 09:48:01 from HEAD 9b93b50c on code commit 99857513; no file under run/utils/envs/agents/protocols/tools/tests differs | `git diff --name-only 99857513 9b93b50c`: 156 files, all `results/ms_r2/` | OK | |
| C02 | 03:64-66 | 120/120 each check; 360 tests, 0 flagged (1.0 expected); scale 120/120 | launch_checks.json (`base.n_ok_per_check`, `base.all_ok`, `base.start_share_tests`, `summary.scale_ok`) | OK | |
| C03 | 03:67-69 | C-NL 80/80, C-MS3 20/20, C-MS4 20/20 (`p_digest_next` not compared where the reference polishes) | `summary.*` | OK | |
| C04 | 03:70 | analysis cross-check against `rule_log.json` stages[2].freeze.noise; 1e-9 absolute, 1e-5 relative; flags column empty in all 200 rows | `r2_analysis.py` NOISE_TOL = 1e-9, SIGMA_RTOL = 1e-5; per_run `flags` non-empty 0/200; rule_log has `stages['2']['freeze']['noise']` | OK | |
| C05 | 03:71 | blind recomputation: "ALL 220 numbers agree ... to 1e-12 (floats) / exactly" | blind_recomputation.txt | OK | |
| C06 | 03:72 | stop-candidate replay: 13,440 exports, 120 runs, 0 invalid / 0 errors / 0 mismatches; max abs diff 0; tool SHA-256 `deed0e2b71146ac2...` same as P1 | validation_summary.csv, stop_calibration_pilot/manifest.json; `tool_sha256` of results/ms_r2/stop_calibration/manifest.json is the same 64-hex value (`deed0e2b71146ac21652609d...`) | OK | |
| C07 | 03:74 | "All checks pass; no stop condition of the prompt was met" | launch_checks.json all_ok true | OK | |
| R01 | README:10 | row MS-R2: branch from `origin/ms-r1` 71c58904; C-NL 80/80, C-MS3 20/20, C-MS4 20/20; primary criterion met in no row; smoothing fell as predicted, remainder rose | 04/05 | OK | "remainder rose" in all eight cells; see S07 for "by as much" (not in README) |
| R02 | README:14 (files paragraph) | files list incl. `pi_record/01_factcheck.md`, `stop_candidate_grids.json`, `report_scripts/` with three scripts | `ls` | UNSUPPORTED | all exist except `pi_record/01_factcheck.md` (to be created) |
| R03 | README:25 | `utils/ms_noise.py`, `tools/ms/r2_*.py`; keys `noise_report`, `conc_scale_schedule`, `fixed_global_sampler`; wave `r2` of `launch_ms_r1.py` | files exist; keys quoted in STATE | OK | |
| R04 | README:36 | results under `results/ms_r2/` (decomposition/, stop_calibration/, pilot/, analysis/, stop_calibration_pilot/) | `ls results/ms_r2` | OK | |
| ST01 | STATE:5 | heading "branch ms-r2, pushed; STOP after the push" | `git ls-remote origin refs/heads/ms-r2` = 9b93b50c (the P1 push); HEAD is 2227fc3a; the three P2 commits and all report files are not on the remote | UNSUPPORTED | premature at the time of the check: only the section 2.6 push exists; the section 4.3 push is pending |
| ST02 | STATE:7 | P1 content: branch from 71c58904; keys `conc_scale_schedule`, `fixed_global_sampler`, `noise_report`; defaults bit-identical (checked against a `git archive` checkout); 128 new tests; independent review one MAJOR fixed; ratio check 0.99927-0.99971 in 140 runs | 03_checks.md:7-14 (128 passed; archive checkout tests), :32 review; decomposition_per_run.csv ratio 0.999266-0.999708 | OK | |
| ST03 | STATE:7 | P2: 120/120 exit 0, launch checks pass, analysis and blind recomputation agree on 220 numbers, D6 repeated | launch_checks.json, blind_recomputation.txt, validation_summary.csv | OK | |
| ST04 | STATE:8 | "sigma_2(0) 1/sqrt(s) within 1.3-10.2 % per run, smoothing part equal to the formula to 0.083 %"; remainder +0.30..+1.40 in all eight cells; paired |peak| -0.0059..+0.0114; NL_bb_s4 q60 +0.0114 [+0.0034,+0.0185]; RMSE/eta/R lower in 7 of 8; stratified lower remainder and |peak| in 5 of 6; C1 Spearman 0.991; C2 floor falls, |c2| rises | P47, P49, D03-D05, T32 | OK | |
| ST05 | STATE:9 | data under `results/ms_r2/pilot/` (13 records per run; arrays and weights untracked) | `git ls-tree`: 13 files per run | OK | |
| ST06 | STATE:10 | known issues: registry test fails; local `main` of primary checkout and `ms-r1` of P1 worktree behind origin; `sampler` and `window` post hoc | 00_housekeeping.md:1.1; 03_checks.md:1 | OK | |

## F. Commits, paths, remote state

| id | file:line | claim | source | verdict | note |
|---|---|---|---|---|---|
| X01 | 04, 00, 05 | commit `9b93b50c` (calibration outputs, 09:47:41), `99857513` (code), `75533f2e`, `a37cdd58`, `2227fc3a`, `93d8be5d`, `71c58904` | `git cat-file -t` = commit for all seven (also d1b84437, 21139215, 4dc604de quoted in 00 section 1) | OK | |
| X02 | 04:3 etc. | report_scripts extended after the code commit | `git diff HEAD --stat reports/ms/r2/report_scripts`: +112 / +62 lines uncommitted | OK | expected per task |
| X03 | all | every cited repository path exists (analysis CSVs, figures, launch JSONs, tools, scripts, stop_candidate_grids.json, tests dirs, reference roots under `.claude/worktrees/`) | existence scan | OK | only `reports/ms/r2/pi_record/01_factcheck.md` is absent (S16, R02, ST) |
| X04 | 00:1.1 | `origin/main` 4dc604de, `origin/ms-r1` 71c58904, no ms tag | `git ls-remote` | OK | `origin/ms-r2` exists at 9b93b50c |

## G. Cross-document consistency (same quantity in 04, 05, summary, STATE, README)

| id | quantity | places | verdict | note |
|---|---|---|---|---|
| G01 | 120/120, C-NL 80/80, C-MS3 20/20, C-MS4 20/20, 360 tests 0 flagged, 220 numbers | 04:11/28, 03:64-71, summary:15, STATE:7, README:10 | OK | consistent |
| G02 | sigma fall 1.14-1.78; smoothing 0.64-1.39; 1.1-2.0 pct points; ratios 1.01-1.10 (1.3-10.2 %); 0.083 % | 04:432-433, 05:48, summary:25, STATE:8 | OK | consistent (1.013-1.102 vs 1.3-10.2 %) |
| G03 | remainder +0.30..+1.40, 4 of 8 above 0, 6 of 8 window | 04:435, 05:49, summary:25, STATE:8 | OK | |
| G04 | gap -0.39..+0.66, 5 of 8 negative; |peak| -0.0059..+0.0114 | 04:48, 04:84, 05:50, summary:16, STATE:8 | OK | |
| G05 | only one gap interval excludes 0 | 04:84 ("in the round") vs 04:310, 04:335, 05:111 | OVERCLAIM | see P28 |
| G06 | tail-mean change range | 04:602 (+0.0001..+0.0007) vs 05:34 table (-0.0002..+0.0007), 05:52, summary:26 | WRONG | see P65 |
| G07 | "up to 1.9" check-to-check | 04:413, 04:605, 05:270 vs trajectory_checks.csv (2.08) | OVERCLAIM | see P45 |
| G08 | "400 more constant-LR updates" attribution | 04:456/483, 05:116 | OVERCLAIM | see P56 |
| G09 | transmission -104 %..+54 %, ratios -1.04..+0.54, only NL_bb_s4 q60 excludes 0 | 04:262, 04:435, 05:86, summary:25 | OK | |
| G10 | sampler: 5 of 6, two intervals each | 04:310, 05:111, summary:27, STATE:8 | OK | |
| G11 | `parents_A` criterion: met for NL_st_s4, NL_st_s16 only | 04:452, 05:115, 05:273, summary:28 | OK | |
| G12 | seed SD of remainder 0.51-1.49 | 05:270 vs 04 decomp table (0.427 min) | WRONG | see D36 |
| G13 | `pi_record/01_factcheck.md` | summary:11, README, STATE:7 | UNSUPPORTED | file absent |
| G14 | "pushed" | STATE:5 vs `git ls-remote` | UNSUPPORTED | see ST01 |

## H. Spec coverage (prompt section 4.1 and D4/D5)

04_pilot.md (required: checks first; one section per arm; decomposition by arm and segment; criterion and secondary tables; D5 predictions against outcome; anomalies):
- Checks first (section 1) present; one section per arm (3.1-3.6, plus 3.7 for the references); decomposition by arm (4.1, 3.x) and by segment (5.1, 5.2); criterion (2) and secondary tables (4.2-4.6, 7); D5 predictions against outcome (6); anomalies (9): all present.
- Thin or missing against D4 (listed for the PI, not required by 4.1): per-stratum errors are printed only for the near-tie stratum (8.1); the middle and tail strata tables (blocks `strata_mid`, `strata_tail`) are not printed (only mentioned); tail max is not reported (only tail mean); eta_2 is on the final tier only (no development tier); along the run (D4, every check 1800-2800) only e_hat_2(0), sigma_2(0), smoothing, remainder at six updates are printed, R0, R, Delta_2, C_2 along the run are not (they are in `trajectory_checks.csv`); stage 1: S1 is not reported and the stage-1 error "against `rehearsal_v2_0`" (D4) is shown as the arms' own signed error next to the `rehearsal_v2_0` row, not as a paired difference; raw-draw clamp counts are given as one mean share of the learner's draws (`t2_clamp_L_frac`), the opponent's share is not printed; the three figures requested in 3.3 are only named (section 10), not shown.
- The "checks" table of section 1 does not name its block (04:3 says blocks are named at each table).

05_decision_inputs.md (required: six arms side by side; what the noise landing changed in each component; what the sampler adds at each s; the stop candidates; what is not settled): all five present (sections 1, 2, 3, 5, 6; section 4 adds the references). "What changed in each component" covers smoothing, remainder, gap, sigma, |peak|, signed peak, RMSE_pos, tail mean, eta_2, R0, R as a pooled direction table (2.1) plus the tie-effort and transmission tables; there is no per-arm-per-component table in 05 itself (it is in 04:4.2).
summary.md: headline numbers, what is not shown, deviations, commands present; names a ledger file that does not yet exist.
00_housekeeping.md section 2 and 03_checks.md section 5 present as addenda; `reports/ms/README.md` row and `docs/STATE.md` top section present and dated.
Descriptive/no recommendation: no sentence selects an arm, proposes a protocol, or recommends a next round (grep for recommend/should/suggest/propose/prefer/better/improve found only the definition "negative when the arm is better" at 04:5).
