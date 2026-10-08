# Fact-check ledger (P3)

Five agents that did not write the folder checked it read-only, each on its own slice, against the records (they recomputed tables from the CSV/JSON copies, not from `tables.py`). The agents were told to give one verdict per statement: OK, WRONG, OVERCLAIM, UNSUPPORTED or MISSING. They ran on commit `5b67f487`; the corrections are in the commit that follows it.

**Deviation from the prompt (D3 of section 4, "one row per checked statement").** The agents returned one row per statement (82, 46, 57, 73 and 94 rows for the slices below, 352 in all). This file lists every row that is not OK with its disposition and, for the OK rows, a count per slice and what was recomputed; the full per-row tables were returned in the agents' final messages and are not copied here.

| slice | what was checked | rows | OK | WRONG | OVERCLAIM | UNSUPPORTED | MISSING |
|---|---|---|---|---|---|---|---|
| S1: report sections 0-6 | every number, parameter, definition, check count, branch/head, prompt reference and PI attribution; Chinese summary against the body; section 2 against `BG-01`, `BG-02`, `BG-03`, `T2R:PL-01` | 82 | 75 | 1 | 5 | 1 | 0 |
| S2: sections 7.1-7.4 | tables `accuracy_a/b`, `signs`, `nonneg`, `formula`, `share`, `quad`, `eta`, `interventions` (44 rows), `budget` recomputed from the per-run and criterion CSVs; labels | 46 | 42 | 0 | 3 | 0 | 1 |
| S3: sections 7.5-7.7 | 13 tables (166 rows) recomputed; figure captions against the PNGs; labels | 57 | 50 | 1 | 4 | 1 | 1 |
| S4: sections 8-11, coverage, README, handoff, provenance | D3, D4, D6, D7, D5, B5, B7-B12 coverage; measures and workload tables recomputed from the launch JSONs | 73 | 51 | 6 | 7 | 3 | 6 |
| S5: Appendix B of the prompt, against the records only | every number and statement of B1-B12 and Appendix A | 94 | 86 (incl. 3 RANGE and 1 PI reasoning rows listed below) | 2 | 2 | 1 | 0 |

(S5 "OK" counts the rows the agent marked OK; its three RANGE rows and its one UNSUPPORTED row are listed below.) All 166 table rows of S3 and the tables recomputed by S2 agree with `tables.py` at the last printed digit; the defect found by S2 and S3 was that the table blocks of `report.md` were empty (below).

## Rows that are not OK, and their disposition

| id | where | verdict | finding | disposition |
|---|---|---|---|---|
| S1-11 | section 0, "严格等于" | OVERCLAIM | the record is a first-order derivation plus agreement within 0.083%, with two collapsed `relu` runs excluded | **fixed**: "与公式一致, 偏差至多 0.083%, 不含 2 个塌缩的 relu run"; handoff likewise |
| S1-14, S1-74 | section 0 and the rounds table, MS-R2 "the gap does not follow" | OVERCLAIM | the gap fell in some cells; no interval excluded 0 | **fixed**: "no detectable fall of the gap" |
| S1-22 | section 1, forward reference to `01_factcheck.md` | UNSUPPORTED | the file did not exist | **fixed**: this file |
| S1-28 | section 2, "the pilot that chose the estimator" | WRONG | `pilot1_reward_estimator.md` says the estimator is not chosen there | **fixed**: the pilot compared the estimators descriptively; the choice is in the protocol |
| S1-33 | section 2, closed form "never enters ... any criterion" | OVERCLAIM | it defines the gates G-A, G-S and the criterion's primary metric, as evaluation | **fixed**: stated |
| S1-36 | section 2, "enters ... twice" | OVERCLAIM | other departures from the invariant (exploring starts, verifier-guided sampler, lagged opponent) | **fixed**: "at least twice" plus the further departures |
| S2-1, S3-53 | `report.md`, every `<!-- TBL -->` block | MISSING | the blocks were empty: `tables.py` matched blocks with a newline between the markers only, so `--verify` and `check_numbers` passed vacuously | **fixed**: `inject()` rewritten, an empty block now counts as a difference; 33 blocks injected (426 table rows); `--verify` and `check_numbers` pass on the filled file |
| S2-16 | 7.3 "i.e." median gap = mean \|peak\| | OVERCLAIM | median gap 4.31 is 6.15% of e2*(0), mean \|peak\| is 0.0630 | **fixed**: "while" |
| S2-24 | 7.3 smoothing share at s = 16 | OVERCLAIM | 13-18% holds for the four `t1` cells only | **fixed**: "in the four `t1` cells" |
| S2-43 | 7.4 MS-R2 bullet labelled [Verified] | OVERCLAIM | a descriptive pilot measurement, not a pre-registered test | **fixed**: [Verified, descriptive] |
| S3-1 | 7.5 intro "all numbers are development seeds, n = 10" | OVERCLAIM | the pilot-4 value 0.00133 is q = 50, five initialisations, other seeds | **fixed**: exception stated |
| S3-12 | 7.5 sigma_2(0) cited to a table without that column | UNSUPPORTED | true in the record | **fixed**: cites `M3-08` |
| S3-38, S3-39 | 7.5 trajectory "plateau"; "1.7 rounded" | OVERCLAIM | range up to 3.55; maximum drift +1.83, not about 1.7 | **fixed**: "does not trend strongly"; the PI's 1.7 is stated as understating the maximum |
| S3-44 | 7.6 "only rho_2 = 0.05 was run" | OVERCLAIM | other thresholds and three candidate criteria were replayed offline | **fixed**: clause with `M1-05`, `M2-06` |
| S3-49 | FIG-09 caption "development seeds" | WRONG | the figure shows 20 development + 40 fresh-seed runs (it calibrates the rule) | **fixed**: caption states it and the D3 mixing exception |
| S3 (labels) | 7.5 seed bullets without a label; 7.6 [Verified] for a rule count; "no transfer" heading | wording | D3 labels | **fixed**: [Verified, descriptive]; heading "no detectable change in RL" |
| S3-54..57 | figure captions without q or budget | MISSING (D5) | captions must give seeds, q, arms, budget, ids | **fixed** for FIG-01, 02, 04, 05, 07, 08, 10, 12, 13 |
| S3-50 | 7.7 text omits the one MS-R1 stage-2 failure | wording | only in the table | **fixed** |
| S3-21 | H3: no discriminating experiment in the text | note | candidate fixes only | **kept**: section 9 U4 names the test (a `relu` run with fixes on more seeds) |
| S4-3 | section 8 item 3 points to section 9 for the withdrawn readings | WRONG | section 9 had no such entry | **fixed**: new row U13 (history of the readings) and a corrected cross-reference |
| S4-8 | section 8 item 9 cites only MS-R3 for T=3 smoke tests | UNSUPPORTED | MS-R1: tests only | **fixed**: `M1-02`, `M1-06` |
| S4-12 | U1 labelled [Hypothesis] while 7.3/8 say [Insufficient evidence] | WRONG | label inconsistency | **fixed**: mechanism [Insufficient evidence], candidate reading [Hypothesis] H1 |
| S4-15 | section 9 lacks B7's history of the PI readings | MISSING | | **fixed**: U13 |
| S4-18 | compare table, Path A "cost: the write-up only" | OVERCLAIM | D6: no invented cost | **fixed**: "no runs; person-time cannot be estimated" |
| S4-19 | Path A "its small effect on eta_2" | OVERCLAIM | causal; remainder missing from the list | **fixed**: "measured eta_2, small (why: [Hypothesis])"; unexplained remainder added |
| S4-25, S4-57 | section 10 labels, seed set and n in reference rows | MISSING (D3) | | **fixed**: labels, seed sets and n added; gain cells tagged [Insufficient evidence] |
| S4-34 | measure 3 workload "MS-R1 ... 503-654 s" | WRONG | pilot only; base wave 386-478 s; R2c wall times not in the pack | **fixed** |
| S4-37, S4-40 | measures 4 and 5: workload "one MS-R2-sized wave" | OVERCLAIM | an assumed size, not a record | **fixed**: "for scale only ... not determined" |
| S4-38 | measure 4 "the critic is shared by all stages" | UNSUPPORTED | no citation | **fixed**: replaced by a statement about acting in every stage, no claim of sharing |
| S4-41 | measure 5 "large per-run spread [TBL-fd]" | UNSUPPORTED | the table shows arm-mean vs per-run-median | **fixed**: reworded to the difference between the two |
| S4-42 | measure 5 "would inform T=3 design only" | OVERCLAIM | | **fixed**: "only" dropped |
| S4-44 | workload table omits the 140 supervised fits, 20 C-R6 runs, R2c | MISSING (B12) | | **fixed**: "Not in this table" note with the screen's recorded wall time |
| S4-48 | PI reasons "about one effort unit or less" | OVERCLAIM | MS-R3 records `relu` median-gap reductions of 0.24-2.35 | **fixed**: record added next to the PI reason |
| S4-55 | section 11 differences list lacks the trajectory drift | MISSING | | **fixed** |
| S4-63 | README: FIG-06 copied but not embedded | WRONG | | **fixed**: embedded in 7.5 |
| S4-64 | README link to the ledger | MISSING | | **fixed**: this file |
| S4-68 | handoff "no admissible tip fix exists" | OVERCLAIM | only tested interventions | **fixed** |
| S4-69, S4-70 | handoff: three sentences of context; English 226 words | WRONG | D7: two sentences, about 200 words | **fixed**: one context sentence each; English 209 words including URLs |
| S5-2 (B1) | "below the closed form in every run of every round" | WRONG | eight runs have a non-negative signed error (R2b 1, R2c 1, MS-R1 1, MS-R3 5, all `relu`) | **reported**: report 7.3 and 11; `TBL-nonneg` |
| S5-68 (B6) | trajectory "up to about 1.7" | WRONG | maximum +1.83 | **reported**: report 7.5 and 11 |
| S5-29, S5-30 (B4) | smoothing floor "2.3-2.7%" and "0.6-0.7%" | RANGE | the first is the `t1` range (2.29-2.72); with `t10` 2.24-2.72; `relu` arms reach 1.81 and 0.54 | **reported**: report 11 and the reference-point table |
| S5-67 (B6) | trajectory "about one effort unit between checks" | RANGE | largest step per arm 0.77-2.08; typical step below 0.5 | **reported**: the record values are given in 7.5 |
| S5-80 (B9) | "tie weighting and budget help a little" | OVERCLAIM | budget effect unresolved; the stratified gain is resolved in one cell | **fixed** in the report's H1 sentence |
| S5-91 (B12) | "per-run wall times are in `budget.csv`" | OVERCLAIM | `budget.csv` has aggregates; per-run times are in `per_run.csv` and the launch JSONs | **reported**: the workload table reads the launch JSONs |
| S5-16 (B3) | "the deficit costs little payoff because the payoff is flat" | UNSUPPORTED | PI reasoning, no record | **kept**, labelled [Hypothesis] with its test (not run) |

## OK rows (what was recomputed)

S1 (75 OK): parameters, protocol gate limits, bootstrap seeds, round heads and dates against `git ls-remote`, check counts C-R4, C-MS1..5, C-NL, C-INIT, C-R6, criterion outcomes of every round, the PI-gate table against prompts 17, 19, 20 and the G1 reply. S2 (42 OK): the fresh-seed accuracy (0.0630, 0.0678, 5 and 4 of 20, signed errors), the 20 rows each of Tables 1 and 2, the sign table, the formula table (0.99917-0.99971), the share table, the quadrature table (10 of 12), and all 44 intervention rows (means within 1e-6). S3 (50 OK): premise check, RL against screen (11 of 12), `relu` typical-run and failure tables, `relu_units`, `t10`, F_d (24 rows, arm means and per-run medians), transmission (20 rows), trajectory (24 rows), stop-rule and R0 tables, gates and the 42-row arms table. S4 (51 OK): the reference points, the workload table (runs, workers, per-run wall minimum, median, maximum, sums), measure 1-3 record numbers, coverage of D3-D7 and of B5, B7-B12, README, handoff numbers. S5 (86 OK): B2 (fresh seeds recomputed), B3, B4 transmission and quadrature, the B5 table, B6 premise and `relu` and `t10` numbers, B8 F_d ranges, B9 clamp line and unit counts, B11, B12.

## After the corrections

`check_numbers.py`, `check_links.py` (working tree; against the pushed commit in `00_build_log.md` P4) and `build_pack.py --check` pass; the commands and outputs are in `00_build_log.md`.
