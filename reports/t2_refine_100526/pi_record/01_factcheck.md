# Fact-check of `100526report.md`, `README.md` and the evidence pack

Date: 2026-10-06. This is the ledger the publication prompt asks for (section 4): the statements checked, the disagreements found, and how each was resolved. The full ledgers of the reviewers are in `factcheck_ledgers/` (one CSV per reviewer, every checked statement with its evidence locator and recomputed value); `factcheck_ledgers/findings_and_resolutions.csv` lists every row that was not `VERIFIED` with its resolution.

## 1. Who checked, and what independence means here

- **Reviewers.** Nine reviewing agents, A to I, started fresh for this check, each with a written brief (the method below). They did not write the report and were told not to look at the drafting agents' scripts or ledgers. After the corrections, two more agents, R1 and R2, checked every changed passage (section 5). **These are agents of the same session and model family as the drafting agents, not a human reviewer;** that limit is listed as item 24(d) of the report's section 7.
- **Method.** For every statement that carries a number, a quoted wording, a verdict word, an attributed decision, a path, a hash or a count, or a universal claim ("all", "none", "only"): locate the cited evidence item through `evidence/manifest.csv`, recompute the value with the reviewer's own code from the pack file (for the bootstrap intervals: `default_rng(seed)`, 10,000 resamples of the 10 paired seeds, 2.5 and 97.5 percentiles; exact Clopper-Pearson intervals from the beta quantiles), compare at the printed precision, check signs, q, comparator, seed and tier, check verdict words and quotations character by character, count universal claims over the full tables, and resolve every cross-reference, figure link and tag. Every table cell was checked by script. A hint file (the number-presence backstop of `report_scripts/check_report_numbers.py`) was given to the reviewers as a hint only: it shows that a printed number occurs somewhere in the pack, not that it is attached to the right statement.
- **Verdicts.** VERIFIED; DISAGREES (the report's value or statement differs from the evidence); UNVERIFIABLE (the check needs something that did not exist yet or is not in the repository); NUANCE (literally true but imprecise, overstated or misleading).

| reviewer | slice (line numbers of the report as checked) | rows | VERIFIED | NUANCE | UNVERIFIABLE | DISAGREES |
|---|---|---|---|---|---|---|
| A | lines 1-44 (Chinese summary, section 1) and `README.md` | 111 | 90 | 12 | 8 | 1 |
| B | lines 45-229 (section 2, sections 3.1-3.4) | 188 | 175 | 13 | 0 | 0 |
| C | lines 230-278 and 580-630 (sections 3.5-3.7, 4.6) | 104 | 96 | 7 | 0 | 1 |
| D | lines 279-520 (sections 4.1-4.4) | 202 | 194 | 8 | 0 | 0 |
| E | lines 521-579 and 631-844 (section 4.5, section 5) | 221 | 208 | 9 | 1 | 3 |
| F | lines 845-988 (sections 6.1-6.3) | 110 | 95 | 12 | 0 | 3 |
| G | lines 989-1113 and 1181-1194 (sections 6.4-6.6, 8) | 152 | 135 | 16 | 0 | 1 |
| H | lines 1114-1180 and 1195-end (sections 7, 9), `pi_record/README.md`, `plans/README.md`, the publication log | 168 | 138 | 22 | 6 | 2 |
| I | mechanical checks: manifest, SHA256SUMS, figures, builder rerun, links | 232 | 214 | 14 | 3 | 1 |
| **total** | | **1488** | **1345** | **113** | **18** | **12** |

Line numbers in the ledgers and in sections 2 to 4 of this file refer to the report as it was checked, before the corrections; the corrections moved no section.

## 2. Mechanical checks (reviewer I, before the corrections)

- **Manifest:** all 180 rows. Item IDs are unique. 167 copied, 10 referenced (untracked), 2 referenced (tracked, large), 1 generated. For every copied item the SHA-256 of the copy equals that of the source file and, for tracked sources, the blob at `HEAD`; sizes agree; `source_commit` is the last commit that touched the path, contains the same blob and is an ancestor of `HEAD`. The 10 untracked items exist in the worktrees named in `location` with the recorded size and SHA-256.
- **Tags:** 911 tag tokens (129 distinct) in the report, 2 in the README, 0 in `pi_record/*.md`: all resolve to manifest rows.
- **`sha256sum -c`:** `evidence/SHA256SUMS` 150 of 150 OK, `figures/SHA256SUMS` 19 of 19 OK; nothing unlisted, nothing missing.
- **Builder:** `python tools/v2/report/build_t2_refine_pack.py --check-against reports/t2_refine_100526` printed "rebuilt 180 items" and "reproduced byte for byte".
- **Figures and links:** all 19 figures viewed; every number in a figure title recomputed from the pack; 21 relative links, 1 broken (the link to this file, which did not exist yet).
- **Folder contents:** no `.pt`, `.pth`, `train_history.json`, weight export, `__pycache__` or `.pyc`; no file above 1 MiB (largest 721,510 bytes).

## 3. Findings and how each was resolved

### 3.1 DISAGREES (12), all corrected

| # | reviewer, line as checked | what was wrong | resolution |
|---|---|---|---|
| 1 | A, README line 59 | the canonical worktree directory `results/v2_T2_locked/` was said to hold the C7 reference; it holds `rehearsal_v1_1` and the v1.1 confirmation, the C7 reference run is `results/v2_pilots/phase1_regression/before/FINAL_A400_B25_C25/tel_q50_s10501` | README table: the C7 reference is its own row with that path and size (2.2 MiB; `results/v2_pilots/` is 1.8 GiB) |
| 2 | C, report line 277 | cross-reference: the sentence said section 4.6 holds the other five methods and D1/D2; section 4.6 is method 6 only | "the chain of evidence is in section 4.6; the other five methods are in sections 4.1-4.5 and the D1 and D2 diagnostics in section 5" |
| 3 | E, line 702 | "stage-2 arms branch from the u1200 state ... 4,000 (run, update) pairs per q group": `A_ctrl200` continues the u1600 state for 200 updates and has 2,000 pairs | exception and the 2,000 added (R1-36 `n_updates`) |
| 4 | E, line 738 | "violates G-F and G-A first at 10% / 15%": true for G-F and the eta part of G-A; the RMSE part of G-A is exceeded from 10% at both q | reworded with the eta part and the RMSE part (Table 5.2d) |
| 5 | E, line 738 | "a tail offset up to tau = 2 reaches none of G-F, G-A or G-N": the tail-mean part of G-A is exceeded at tau = 2 (0.02857 at q = 50, 0.03429 at q = 60 against 0.02) | reworded: none of G-F, the eta part of G-A, G-N; the tail-mean part is exceeded (Table 5.2d) |
| 6 | F, line 855 | "each was tested at one setting": two settings for polishing, batch, target-KL and annealing, three pathwise arms | "one or two settings (pathwise fine-tuning: three arms)" |
| 7 | F, line 913 | the bootstrap paragraph of RR-08 is in its section 5, not section 6 | "RR-08, section 5, and RR-09, section 6" |
| 8 | F, line 919 | "with the single exception noted above": two exceptions (`A_batch_mb256` at q = 50; `A_anneal4` at q = 60, worsening side, +0.01181 [0.005123, 0.01949]) | both named |
| 9 | G, line 1068 | Table 6.5b, `R2b_A_peak25` at q = 50, "mean diff / baseline": -0.1814; exact -0.011903677 / 0.065595814 = -0.18147 | -0.1815 |
| 10 | H, line 1126 | RR-03 "section 7 ('Housekeeping')": section 7 is "Commands to reproduce"; the bullet is in section 6, deviation 7 ('Other anomalies') | corrected |
| 11 | H, line 1201 | the worktrees holding the untracked files are not listed in `reports/v2/README.md` (it names only the canonical one) | pointer changed to this folder's `README.md`, section "Large files that are not in git", and the manifest `location` |
| 12 | I, README line 23 | link to `pi_record/01_factcheck.md`, which did not exist | this file written |

### 3.2 UNVERIFIABLE (18)

All 18 were things that did not exist yet when the reviewers looked, or session facts: `pi_record/01_factcheck.md` and the two `SHA256SUMS` files (written; the checksum files are generated last); the tag, the push and the publication log section 5 (written at the merge and push); the addenda to `reports/v2/summary.md` and `reports/v2/README.md`; one statement in the report about the pilot arms and D1 (E, line 702: "not designed to test D1", no source; removed); and three statements of the publication log that rest on the session (the guard's refusal and the owner's answers), which are not checkable from the repository and are labelled as records of the session. Per-row dispositions are in `factcheck_ledgers/findings_and_resolutions.csv`.

### 3.3 NUANCE (113)

All were reworded in the report, the README, `pi_record/README.md` or the log, with the reviewers' proposed wording or a close adaptation; `factcheck_ledgers/findings_and_resolutions.csv` has one row for each with the reviewer's note and the resolution. The ones that change how a result reads:

- **Part (b) is trivial in R1** (B, C): all 160 stage-1 and 220 stage-2 runs pass their gates, so the verdict of R1 rests on part (a); stated in sections 2, 3.7 and 4.6.
- **"Reduction of spread, not a shift of the mean"** (C): with n = 10 the signed-difference intervals cannot exclude shifts of 0.02-0.04; now "no shift of the mean is detected (n = 10)".
- **Wall ratio 1.084** (C): a shared-machine wall clock, indicative only (RR-08 Addendum 1, item 5).
- **Method 5, "does not separate from the control"** (E, F): `P20_lr3e-4` is worse than its control at q = 50; `A_detmean` and `P20` arms lower the FOC residual and RMSE_pos, `A_detmean` RMSE_pos at q = 50 contains 0.
- **Section 6 wording** (F): share response not monotone at q = 50; "far outside the others" replaced by the numbers (-0.1458 against -0.0895 for the next); the q-asymmetry is absent from the confirmation runs on median, mean and count but the most negative single error is still at q = 50; the representation floor is a median (largest of five fits 0.0531); heading 6.2 now "did not meet part (a)"; the share response is "lower for every share tested", with the intervals.
- **Seed 30510** (G, H): the rule text of R2b was written at 07:03 UTC before the first diagnostic output at 07:20, but the author saw a diagnostic agent's own readings before committing the file (RR-09 section 5); "none of H1-H4 (early or unstable departure)" is the label of seed 30510 alone among the 40 runs, while H2 fires on 5 passing runs.
- **Section 8** (G): the items are "open questions for the PI (listed by this report; none is a recorded pending decision)"; the `clamp_likelihood` setting is the PI's R2c decision D1.
- **Chinese summary** (A): the sentences that claimed more than the body ("the only effective lever", "compatible with this reading") now say what the body says; the heading "建议" is now "决定与开放问题".
- **Statements that described the push and the tag before they happened** (H, A): the push of `origin/main` and the tag are now cited to log section 5, written after the push.

Not changed, noted by the reviewers as correct but worth knowing: the v1.1 maximum at q = 60 is printed with three significant digits (0.153) in places; the R2b summary's FOC means are printed with two digits; the table build "7 to 8 s" of the R1 report is a report-only number and the pack residual is about 9-10 s under shared load.

## 4. Disagreements with round reports, and PI-side readings

### 4.1 Round reports: dated addenda (commit `d19c5649`; nothing above the addenda was edited)

| report | statement | evidence | addendum |
|---|---|---|---|
| `reports/v2/refine/summary.md` row 03 of the reading order | "D2: 152 perturbed closed-form candidates" | `03_d2_verifier_sensitivity.md`: "evaluations (rows) / errors" = "152 / 0"; `evaluations.csv`: 38 candidates x 2 q x 2 tiers | addendum item 1 |
| `reports/v2/refine/06_decision_inputs.md`, annealing paragraph | "-0.0766 [-1.026, 0.837]" | `stage2_annealing.csv`: the lower end is -1.0254895, which rounds to -1.025 | addendum item 2 (in the summary) |
| `reports/v2/refine/01_preregistration.md` section 4 item 6 (and the verifier numerics note) | "falls by about 4x per halving of the verifier state step" | `continuation_check.json` sweep, 32 nodes: ratios 4.29, 3.79, 2.87, 1.76 (q = 50), 4.11, 3.83, 2.56, 1.83 (q = 60) | addendum item 3 |
| `reports/v2/refine_r2c/summary.md` observation "Part (b) holds" | "3.2% and 4.3% below" the limit 0.02 | `tail.csv`: 0.019374 is 3.13% below, 0.019143 is 4.29% below | addendum in that summary |

The evidence-pack copies of the two summaries were refreshed after the addenda (items RR-01 and RR-06) and the pack was rebuilt byte for byte.

### 4.2 PI-side readings (publication prompt, section 3, "PI-side readings to verify, not to copy")

Every reading was compared with its evidence item by the drafting side (`report_scripts/pi_readings/`) and the reviewers cross-checked those in their slices (A: the `A_peak50` endpoints; E: D1, D2 and the method-5 readings; F, G: the R2b and R2c arms). Of 38 readings (R-01 to R-34e: 34 AGREE, 4 DISAGREE) and 17 commit labels (H-01 to H-17: 16 AGREE, 1 DISAGREE):

- **AGREE** (to the digits the PI wrote): all of R-01 to R-22, R-24 to R-28, R-30, R-32, R-33, R-34b to R-34e, and 16 of the 17 commit labels.
- **DISAGREE, reported as such, and the report uses the evidence values:**

| reading | the PI's value | the evidence | difference |
|---|---|---|---|
| R-23, R2b `A_peak50`, q = 50 lower end | -0.0410 | -0.0414478 (R2B-02) | 0.0004 |
| R-23, q = 50 upper end | -0.0040 | -0.0037586 | 0.0002 |
| R-23, q = 60 lower end | -0.0230 | -0.0227322 | 0.0003 |
| R-29, R2c `A_peak35`, q = 50 upper end | -0.0136 | -0.0135488 (R2C-02), rounds to -0.0135 | 0.0001 |
| R-31, R2c `A_peak50_late400`, q = 50 lower end | -0.0303 | -0.0302462 (R2C-02), rounds to -0.0302 | 0.0001 |
| R-34a, D1 shorthand "meets the criterion at q = 50 only" | read for R2b `A_peak50` | `A_peak50` meets part (a) at both q (-0.02346 [-0.04145, -0.003759], -0.01176 [-0.02273, -0.001032]); the criterion is not met because part (b) is violated at q = 60 | the shorthand is exact only for part (a) of three R2c arms (`A_peak35`, `A_peak40`, `A_peak50_late800`); the report states both readings (section 6.6) |
| H-14, `3ad1b07` as the R2c "results" commit | results | it is the commit that the wave-S manifests record (C-R3 reproduction, tool validation and review); the wave-S runs, launch checks and tables are in `3492cac4` | label corrected in the README |

- **Two further label notes, both in the README:** `d2e377d` ("rehearsal/launch") is the confirmation launch commit and holds the re-rehearsal records (the re-rehearsal itself was launched at `f2d616c`); `d581b3c` ("launch" of R2b) is the commit recorded in the pilot manifests and its subject is the addendum to the pre-registration.
- **R-27 (FOC and RMSE):** the PI's numbers are medians over the 10 seeds (5.05e-04 -> 3.6e-04 for `P20_lr3e-5` at q = 50, RMSE 0.0201 -> 0.0149 at q = 50 for `P20_lr3e-5` and 0.0198 -> 0.0134 at q = 60 for `P20_lr3e-4`); the round reports and Table 3 of the report give means (for example 5.2e-4 -> 3.9e-4 at q = 50). Both are right; they are different statistics, and the two RMSE pairs belong to different arms and q. The report prints the means.
- **R-04, R-03:** 1.084 and 0.0656 / 0.0531 agree; the wall ratio is a shared-machine wall clock.

## 5. Second pass on the corrections and final checks

**Second pass.** After the corrections of sections 3 and 4, two further reviewing agents checked every changed passage (the editor listed each passage with a search anchor and its new claim; the agents recomputed the numbers and compared each new wording with its source):

- **R1** (report sections 0 to 6): 112 rows, 102 VERIFIED, 10 NUANCE (9 distinct findings), no DISAGREES, no UNVERIFIABLE. All 56 tables of the report have consistent cell counts; 264 table cells of Tables 3, 5.2d, 6.2, 6.3a-d and 6.5a-b were compared by script with the pack. Findings, all applied: the "G-A" labels of the D2 row (section 2 and the Chinese summary) now say "eta part" and name the RMSE and tail-mean parts; "lower for every share" is limited to the shares applied from update 1 (the late schedules are in section 6.5); the Chinese summary says "among the mechanisms judged on the stage-2 peak error", gives the q-asymmetry qualifier (median, mean and count; the most negative single error is still at q = 50) and no longer says that the two measurements "do not support" the weighting reading; the seed-30510 sampling statement cites the peak-set share range of the other 19 seeds (0.09942 to 0.10029) and the bin shares against the design value; "second order" is the PI's wording; the verifier sweep ratios are those at 32 Gauss-Legendre nodes per half interval; "the prefix identities (R2c prompt D2)".
- **R2** (report sections 7 to 9, line 3, line 41, `README.md`, `pi_record/`, the publication log, the round-report addenda, and the mechanical checks of the folder): 192 rows, 157 VERIFIED, 22 NUANCE, 12 UNVERIFIABLE, 1 DISAGREES. The DISAGREES: report line 43 said that the publication changes "no earlier report"; it appends dated addenda to two (corrected: the sentence lists the addenda and the new section of `docs/STATE.md` and says that no existing line is edited). The 12 UNVERIFIABLE were forward references (the push, the tag, log section 5, `pi_record/SHA256SUMS`, `factcheck_ledgers/`, the placeholder of this section), resolved by the steps of the publication. The NUANCE findings, all applied: the training-relevant state for `A_peak50_late400` (exports up to update 1200 and 1200 per-update rows); the source of "Proceed per D4" (the log); nine agents plus two second-pass agents; section 8 "open questions for the PI that this report raises"; "no shift of the signed error detected (n = 10)"; M2 exceeded in the pooled reading only; the T=3 scope of the `clamp_likelihood` decision (no record says); the figure descriptions (FG-02 and FG-04 stage overviews, FG-03 and FG-14 per-arm paired-difference plots); the scope of "not in git" (the result roots of these four rounds track only the 60 continuation tables); RR-03 section 6, finding RC-F3 for the refusal to run without the single-thread settings; README line 3 (the addenda) and line 28 (log sections 1.1 and 1.6); the mapping of the PI decisions in `pi_record/README.md` (R1 addendum items; Pub-D1 is the only decision on the rounds); the dates of the publication log; the caution that all 54 colliding files stop a fast-forward; the label of seed 30510 in this file; the wording of the `reports/v2/summary.md` addendum; and the indentation of row 13 of `reports/v2/README.md` (a table nested in a list item).

**Final checks** (after the last edit of the report, run by the editor with `report_scripts/final_checks.py`; `pi_record/SHA256SUMS` is written last and is checked by `sha256sum -c` from its folder):

- **Links:** the 21 relative links of the report and the README (10 and 11) resolve.
- **Tags:** every item tag of the report (134 distinct), the README (7), `pi_record/README.md` and this file (12 in all) resolves in `evidence/manifest.csv` (180 rows).
- **Tables:** every Markdown table of the report, the README, `pi_record/README.md`, this file and the log has the cell count of its header (66 tables, 409 data rows).
- **Checksum files:** `evidence/SHA256SUMS` 150 of 150 OK, `figures/SHA256SUMS` 19 of 19, `report_scripts/SHA256SUMS` 20 of 20; no file of those folders is unlisted.
- **Builder:** `python tools/v2/report/build_t2_refine_pack.py --check-against reports/t2_refine_100526` rebuilt 180 items and reproduced the pack byte for byte (after the two copies of the round summaries were refreshed for the addenda).
- **Number backstop** (`report_scripts/check_report_numbers.py`): of 2020 decimal numbers of the report, 2001 occur in a table of the pack at the printed precision, 11 only in a round report, 8 nowhere; the 8 are values that the report computes (two SD ratios, a mean, three ratios against the baseline, and the drawing position 887.5 of a line in a figure), each recomputed by a reviewer.
- **Folder contents:** no `.pt`, `.pth`, `.pyc`, `train_history.json` or weight export, and no file of 1 MiB or more.

## 6. Files

- `factcheck_ledgers/A_ledger.csv` to `I_ledger.csv`: the nine reviewers' ledgers (`id,line,section,statement,evidence_item,locator,recomputed_value,verdict,note`); `R1_ledger.csv`, `R2_ledger.csv`: the second pass.
- `factcheck_ledgers/findings_and_resolutions.csv`: every row that was not `VERIFIED`, with its resolution.
- `../../../../tools/v2/report/build_t2_refine_pack.py` and `../report_scripts/`: the builder and the drafting scripts the checks were run against.
