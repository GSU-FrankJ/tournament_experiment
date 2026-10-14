# MS-R4: fact-check ledger of the reports

Date: 2026-10-14. Scope: `reports/ms/r4/04_e1_actor.md`, `04_e2_feedback.md`, `05_decision_inputs.md`, `summary.md`, `handoff.md` (Chinese and English), the P2 / P3 additions to `00_housekeeping.md` and `03_checks.md` (section 9), the MS-R4 entries of `reports/ms/README.md` and `docs/STATE.md`. Prompt section 4.2.

**Who checked.** A read-only agent that did not write the reports (a fresh agent with the report files and the paths of the CSVs, not the author's conclusions; it modified nothing in the repository, scratch files in `/tmp/factcheck_ms_r4/`). It read the analysis CSVs, the per-run `gates.json` files, the launch record and checks, the curves, `prereg_criteria.json`, the protocol and the PI's reply, recomputed the numbers and compared them with the prose and the printed tables. The author verified each correction below against the data before applying it (the column and code reads are cited), and the same agent then re-checked the corrected text (second pass below).

## First pass

- **Numbers checked:** about 500 in the prose (438 recomputed by script, about 60 counts, hashes, commit ids, times and file counts checked by hand or with git) and about 2,500 table cells compared; every generated table equals the current output of `report_scripts/pilot_tables.py`; about 2,300 cells were re-derived independently from `per_run.csv`, `curves_all.csv`, the raw `gates.json` of the 280 runs and the launch records; the criterion intervals, `paired_e1.csv` (624 values) and `secondary_e2.csv` (456 values) were re-bootstrapped with 0 differences; all 280 `run_config.json` hashes equal `run_config_hashes.csv`; 3,973 copied files are byte-identical to the launch worktree; `per_run.csv` equals the raw `gates.json` for all 280 runs.
- **Discrepancies:** 1 WRONG (one root cause, 31 printed numbers in four files plus the script), 4 UNSUPPORTED, 14 MINOR.
- **A2:** the block in `05_decision_inputs.md` section 5 is character-identical to `pi_record/24_g1_reply.md` and to Addendum 2 of `02_preregistration.md`; the E1 restatement matches rules (i)-(iii); the only gap was that the E2 branch and the combination were not restated next to their outcomes (row 19).
- **Chinese and English halves of `handoff.md`:** they say the same thing; every number, arm and q agrees (and both carried the row-1 error).
- **Order of sections** follows section 4.1 of the prompt in both pilot reports, with a "0. Result" section before the checks and, in `04_e2_feedback.md`, a post hoc section 7 between the weights and the cost.

| # | where | the report said | the data say | severity | disposition |
|---|---|---|---|---|---|
| 1 | `pilot_tables.py` blocks `gmax_equality`, `e2_stage1_gain`; `04_e2` section 7, `05`, `handoff.md` | stage-1 deviation gain Delta_1/DW: median 5.66e-06 / 8.99e-06, maximum 1.0e-4 / 1.3e-4; "at most 1.4e-4 ... more than seven times below the margin" | the script divided `t1_Delta_final` by 4.0 (DW), but the column is already divided by DW (`utils/ms_residual.py`: `Delta` = `delta_over_dw`; for every run `t2_Delta_final` equals `eta_T_over_dw`). Correct: median 2.27e-5 (E1) and 3.60e-5 (E2), maximum 4.1e-4 and 5.4e-4, about half of the margin; paired mean differences at most +5.6e-5 | WRONG | **fixed**: script, tables and every number in `04_e2` section 7, `05`, `handoff.md` (zh, en); no verdict depends on it ((E2-a) fails for both decision arms whatever (E2-c) says) |
| 2 | `04_e1` section 4 | "The controls behave as in MS-R3." | q = 60 matches (-0.0207 against -0.0206); at q = 50 `relu` - `t1` is -0.0033 [-0.0157, +0.0127] against -0.0104 [-0.0162, -0.0047] | UNSUPPORTED | **fixed**: "partly reproduce", q = 50 effect smaller with an interval containing 0 |
| 3 | `04_e1` section 6 | `relu` and `lrsm` have a lower median abs peak error "in the second half of the stage" | at q = 50 `t1`'s median is lower at updates 1600 and 2000; from update 1400 on `relu` is lower at 44 (q = 50) and 52 (q = 60) of 57 exports, `lrsm` at 50 and 50 | UNSUPPORTED | **fixed**: "at the end of the stage and at 44-52 of the 57 exports from update 1400 on", with the `t1` exceptions named |
| 4 | `04_e1` section 5.1, `05`, `summary.md` | in all seven failures RMSE_pos(u2000) > RMSE_pos(u1200), "so the degradation is under way before the ramp" | the same rise occurs in 56 of 113 passing runs; the observation did not separate the groups | UNSUPPORTED | **fixed**: replaced by the comparison (block `e1_rmse_rise`, post hoc): ratio u2000 / u1200 is 2.3-8.7 in all seven failing runs (median 3.1) and in 3 of 113 passing runs (median 1.00), labelled as a description selected after the fact |
| 5 | `05` (option B, combination bullet), `handoff.md` | "the plan allows a combination only after each try is effective"; "no confirmation or combined run is called for under the plan" | Appendix A says only that, even if each try is effective, the combination must first be validated on development seeds; it is silent on tries that missed their targets | UNSUPPORTED | **fixed**: phrased as a reading of Appendix A ("silent on a combination of tries that missed their targets"); the handoff leaves the decision to the coworker and the PI |
| 6 | `04_e1` section 5.1 | RMSE_pos "below 0.06 up to update 1600" | below 0.06 from update 75 to 1600 (0.26-0.28 at update 25, 0.14-0.17 at update 50) | MINOR | fixed |
| 7 | `04_e2` section 3 | "1.4-4.2 % above the limit" | +1.31 %, +4.20 %, +3.29 % | MINOR | fixed: 1.3-4.2 % |
| 8 | `04_e2` section 6, `05` | `fb_rx` median TV "0.12" | 0.1251 (q = 50) and 0.1218 (q = 60) | MINOR | fixed: 0.122-0.125 |
| 9 | `05` section 4 | "the E2 area intervals are wide at q = 60" | widths at q = 50 and q = 60 are 0.064 / 0.057 (`fb_r`), 0.074 / 0.061 (`fb_rg`), 0.081 / 0.081 (`fb_rx`) | MINOR | fixed: "widths 0.06-0.08" |
| 10 | `05` section 1 | "E1 at q = 60 is a clear, small improvement" | interval [-0.0219, -0.0016], 14 of 20 pairs lower; the clear case is `relu` - `t1` at q = 60 | MINOR | fixed |
| 11 | `03_checks` section 9 | status `ok` "in every row" | `ok` in the per-q rows; the "both" and DECISION rows hold the verdict text | MINOR | fixed: "every per-q row" |
| 12 | checks table (`04_e1`, `04_e2`, `03_checks`) | "strata masses equal the design to 1e-12: 280 / 280" | true for the tail mass in all 280 runs; the near-tie and middle masses equal the design only in the 120 runs of `fb_a0`, `fb_r`, `fb_rg` (the cross-stratum runs deviate by construction) | MINOR | fixed: two rows, 280 / 280 (tail) and 120 / 120 (near-tie and middle) |
| 13 | `summary`, `04_e1` | supplements "labelled in the reports" | the early-share table and the multiplicity count in `04_e2` were not labelled; `e1_units` is a D5 item | MINOR | fixed: labelled post hoc; the sentence now says D5 itself calls the hidden-unit counts post hoc |
| 14 | `STATE`, `summary`, `README`, `00_housekeeping` | the branch is "pushed", the ledger is cited | at the time of the check the report files were uncommitted, `origin/ms-r4` was `2f618531`, and this ledger did not exist | MINOR (pending) | resolved by the commits and the push of section 4.3 (see the end of this file) |
| 15 | `04_e2` section 8 | arms "differ in cost by noise only" | all six paired wall-time intervals contain 0 | MINOR | fixed: "no detectable difference" |
| 16 | `04_e1` section 6, `04_e2` section 8 | offline cost "reported in `01b_curve_tool.md` and `curves_info.json`: 625 s and 314 s" | the two values are only in `curves_info.json`; `01b` holds the MS-R3 validation (47.2 s and 22.6 s) | MINOR | fixed |
| 17 | `04_e2` section 3 | `fb_rx` "the lowest of the four from about update 300 at q = 50" | lowest at 46 of the 53 exports from update 300 on | MINOR | fixed |
| 18 | `04_e1` section 1 | "4.1 standard errors, in 5 of 280 runs" | the maximum |z| = 4.10 is one run; five runs have a flagged test | MINOR | fixed |
| 19 | `05` | A2 "is stated next to each outcome" | only the E1 branch restated the rule | MINOR | fixed: the E2 branch and the combined scheme restate their A2 clauses |

### Coverage (first pass)

Covered: every D5 item for E1 at the freeze on the final tier (blocks `e1_per_seed`, `e1_overview`, `e1_paired_peak`, `e1_decomp` x3, `e1_locfree`, `e1_strata`), the four gates, the failures (trajectories, tie at the bound, unit shares), the learning curves of eta_2/DW and abs peak error, cost, E2 curves, gates and end points, the sampling record, all D6 parts including the 97.5 % intervals, `fb_rx`, the secondaries and the predictions, and every deliverable of Appendix A (configuration and code version, per-seed results, paired intervals, peak, learning curves, final gates, cost). Not covered by a table or statement in the first version, and the disposition:

| # | item | disposition |
|---|---|---|
| 1 | E1 learning curve of e_hat_2(0) | **added** (`04_e1` section 6, block `e1_curves metric=e2`) |
| 2 | "on both tiers": no development-tier table | **added** (`04_e1` section 5.3 and `04_e2` section 3, block `tiers`: eta_2/DW and Gmax_full/DW on both tiers, the G-N differences; the development-tier strata stay in `strata_summary.csv`) |
| 3 | G-N(Gmax) values | **added** (same block) |
| 4 | E2 development-tier online values | **partly**: the area `A_dev_online` (secondary table) and the end values (block `tiers exp=e2`); the per-check development-tier series stay in the online check files that C-OFF ties to the offline curves |
| 5 | realised near-tie / middle shares of the E1 arms (prompt 3.2) | **added** (`04_e1` section 5.3, block `e1_shares`) |
| 6 | per-seed E2 values other than A | **added** the per-seed abs peak error (block `e2_peak_seed`); the other per-seed values (gap, RMSE_pos, tail mean, stage-1 error) stay in `per_run.csv` |
| 7 | stage-1 cost categories | **added** the stage-1 verifier time; the stage-1 rollout and update times of E1 stay in `cost_e1.csv` and `per_run.csv` |
| 8 | per-update start counts (`n_start_*`) | **not tabulated**: shares by update are in `strata_shares_by_update.csv` and the figure, the counts in `ms_updates.csv` |

## Second pass

The same agent, resumed, re-checked the corrected text.

- **Numbers:** 60 new or changed claims recomputed by script, 60 of 60 correct, plus about 25 ratios, thresholds and counts checked by hand; all 484 decimal tokens of the five reports' prose matched against the ground-truth values (only the "1e-12" tolerances have no match); 56 of the 62 prose intervals of the form "x [lo, hi]" match a triple in the CSVs (the other six are the MS-R3 priors, verified in the first pass, and the P1 stress-test numbers of `03_checks.md` section 8); all 52 printed blocks equal the current output of `pilot_tables.py` (0 of 701 lines missing from the reports); 730 cells of the changed or new blocks were recomputed independently with 0 mismatches; `t1_Delta_final` equals the raw `gates.json` values for all 280 runs (largest stage-1 gain 5.354e-4).
- **Result:** WRONG 0, UNSUPPORTED 0, MINOR 7 (four carried over, three new). Of the 19 first-pass rows, 18 were resolved; row 14 stayed open until the push (it is resolved by the commits and the push recorded below).
- **A2:** `05_decision_inputs.md` section 5 is still character-identical to `pi_record/24_g1_reply.md` and to Addendum 2; the E1 and E2 restatements match.
- **Strata-mass facts in the checks tables** are correct (tail mass 280 / 280; near-tie and middle masses 120 / 120 for `fb_a0`, `fb_r`, `fb_rg`).

| # | where | the report said | the data say | disposition |
|---|---|---|---|---|
| R1 (= row 14) | `STATE`, `README`, `summary`, `00_housekeeping`, the citations of this ledger | "pushed", the ledger cited | not yet pushed, no ledger | **resolved** by the commits and the push (end of this file) |
| R2 | `04_e2` section 7 | "Gmax_full/DW is the maximum over the stages of the final-tier deviation gain ... because the stage-1 deviation gain is smaller" | `Gmax` is the maximum over the stages of the best-response value gap v_br - v_mean (`utils/v2_metrics.py`); the one-step gap Delta_1 of the table is a different quantity; equality with eta_2/DW implies that the stage-1 best-response gap is at most eta_2/DW, and Delta_1/eta_2 is at most 0.49 (E1) and 0.45 (E2) | **fixed**: the definition is stated as the best-response gap; the table is labelled the one-step stage-1 deviation gain, with a column for the largest Delta_1 / eta_2; the advice for a later round now says "say which stage-1 quantity" |
| R3 | `05` section 4, `summary` | "the seven failing runs degrade before the ramp (RMSE_pos rises by update 2000 in all seven)" | the same rise (ratio above 1) occurs in 56 of 113 passing runs | **fixed**: replaced by the factor 2.3-8.7 against a median ratio of 1.0 in the passing runs (post hoc) |
| R4 | `summary`, `README`, `STATE` | "strata masses 280/280" | only the tail mass is 280 / 280; near-tie and middle masses are checked in 120 runs | **fixed** |
| N1 | `04_e1` section 5.1, `summary` | "3 of 113 passing runs reach the ratio 2.29, the smallest among the failing runs" | the failing minimum is 2.294; one passing `relu` run (q = 60, seed 10511) has 2.2935, below it; at the exact threshold 2 of 113 passing runs reach it (seeds 10521: 2.49 and 10522: 2.72, both `relu` at q = 60) | **fixed**: the block uses the exact minimum of the failing runs and lists the passing runs that reach it; text and summary say 2 of 113 and name the third run |
| N2 | `04_e1` section 5.1, block `e1_failure_traj` | one table of e_hat_2(0), RMSE_pos, R0 and the symmetry error, tier not named | e_hat_2(0), RMSE_pos and R0 are the online development-tier checks, the symmetry error is the offline final-tier curve (`tools/ms/r4_e1_analysis.py`, `failure_trajectories`); e_hat_2(0) and RMSE_pos are tier-independent | **fixed**: a sentence names the tiers; the tier-independence of the five recovery metrics is stated with its check (largest |final - development| over all 23,680 curve points: 0, block `recovery_tier_check`) |
| N3 | `05` section 2 (the quotations in the intro and in the combination bullet), `handoff.md` | quotation marks around condensed translations of the note and of the prompt | the quoted fragments did not match the English rendering or the prompt verbatim | **fixed**: replaced by the exact words of the English rendering (`pi_record/Multistage101126(1).md`) and a note saying so; the exact "stage-1 deviation gains do not become worse" in `04_e2` and `handoff.md`; "If the goal is not met, stop this line" in `05` |

The corrections of this second table were verified by the author against the data (script output, the code lines the agent named, the quoted source files); the agent was not run a third time.

### Coverage (second pass)

Covered now: items 1 (e_hat_2(0) curve), 3 (G-N values) and 5 (E1 shares). Item 2 (both tiers) is covered for the verifier-based quantities (block `tiers`) and, for the recovery metrics, by the statement and check that the two tiers give identical values (block `recovery_tier_check`); the remaining development-tier numbers of D5 are in `per_run.csv` and `strata_summary.csv`. Items 6 and 7 are partly covered (per-seed peak; stage-1 verifier time) and items 4 and 8 stay in the CSV files, as listed in the first-pass table; none of them enters a criterion.


## Not verifiable by the checker

- The tmux session `msr4_pilot` and its closure (no file records it); the launch record and the stdout of the launcher do.
- The SHA-256 of the coworker's `.docx` (never seen; `02_preregistration.md` Addendum 1).
- The P1 numbers (155 tests, suite counts, review counts) were cross-read against `03_checks.md` only.
- The GitHub links of `handoff.md` (no network access from the checker; the files exist and the remote URL matches `git remote -v`).
- The v2.0 numbers (0.0630 / 0.0678 at q = 50 / 60, 5 of 20 and 4 of 20 within 0.05) and the MS-R3 priors (-0.0104 / -0.0206; 5 of 40; 1 of 10) are not in the round's own CSVs; the checker confirmed them from the raw confirmation `gates.json` in the `v2-t2-refine` worktree and the status-pack text, and from the tracked `results/ms_r3/analysis/` tables.

## Closing record

The reports, the table script, this ledger and the index entries (`reports/ms/README.md`, `docs/STATE.md`) are committed together in one `docs:` commit of branch `ms-r4` (`git log -- reports/ms/r4/pi_record/01_factcheck.md` names it), and `ms-r4` is pushed as a fast-forward (prompt section 4.3; no force, no tag, no pull request, `main` untouched). Row 14 and R1 are therefore resolved by that push.
