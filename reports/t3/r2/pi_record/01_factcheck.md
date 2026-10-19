# T3-R2: fact-check ledger of the reports

Date: 2026-10-19. Four independent, read-only checkers (general-purpose agents that wrote none of the reports; told not to modify any file and to recompute every number from the CSV / JSON / code rather than trust the generated tables). Each was given the specification (`pi_record/27_t3_r2_prompt.md`) and a set of files; none saw the corrected text. The checkers' counts are their own. Every problem was checked against the data before it was accepted; all 52 were accepted; the corrections are in the report templates / scripts (`report_scripts/`) and the reports were regenerated, so the corrected text is what was committed.

| checker | files | claims checked (their count) | correct | problems reported |
|---|---|---|---|---|
| 1 | `04_confirmation.md`, `summary.md` | about 200 (about 900 table cells recomputed) | 185 | 15 |
| 2 | `01_audit.md` sections 0 to 6 | about 270 | about 255 | 12 |
| 3 | `01_audit.md` sections 7 to 11 | about 235 | about 226 | 9 |
| 4 | `handoff.md` (Chinese), `02_preregistration.md`, `03_checks.md`, `00_housekeeping.md` | about 220 | about 205 | 16 |

Not checkable by the checkers (stated by them): tmux session names and whether the launch ran in tmux; the review agent's own transcript; the P0 load average and free disk; the timings of the post-launch tools (`time` output, not stored; now labelled so in the report); the mechanism statements that the reports mark as untested.

## Checker 1: `04_confirmation.md`, `summary.md`

| # | problem (severity) | verified | disposition |
|---|---|---|---|
| 1 | the means table (`c09_means`) held the stage-of-maximum rows: the generating script overwrote its row list (WRONG) | yes (script bug) | fixed in `confirmation_tables.py`; the table now shows the mean `Gmax_full` and `EXP_root` (0.00325 / 0.00249 and 0.00078 / 0.00058 at q = 50; 0.00897 / 0.00476 and 0.00447 / 0.00084 at q = 60) |
| 2 | "a failure of the rule is the more likely outcome" at 0.90 is false (IMPRECISE) | yes: pass probability 0.607 at 0.90 | rewritten: failure probability 0.75 at 0.85, 0.59 at 0.875, even near 0.89, passing more likely above it |
| 3 | AP3 second part was said to hold for the marginal excesses, contradicting section 5 (UNSUPPORTED) | yes (60501 and 60506 above 0.005 at 14 of 16 exports) | rewritten: not supported for any of the four q = 50 failures |
| 4 | "the stage-2 gain it inherits" / "none of the q = 60 failures is a stage-2 failure" for seed 60518 (UNSUPPORTED) | yes: one-step gain at stage 2 is 0.0369 at d = −12, G_max,2 = 0.113 | rewritten with both numbers; "inherits" removed; the maximum is at stage 3 but stage 2 is also far from a best response |
| 5 | AP2 stated as holding although alpha < 1 and the clamp share were not examined on the confirmation nodes (IMPRECISE) | yes | now limited to the objective part |
| 6 | "agree to about 1e-5 dW" (IMPRECISE) | yes | the three numbers are quoted |
| 7 | "the audit of section 3 found none" cites the wrong section (IMPRECISE) | yes | now section 0 |
| 8 | "a few effort units below the best response over a wide band" (IMPRECISE) | yes: 5.8 to 6.7 at d = 0, 10.2 to 11.7 at d = −10 (recomputed from `freeze_stage3_final.npz`) | numbers and the band added |
| 9 | "(1)" for a boolean (IMPRECISE) | yes | now text |
| 10 | the `git status --porcelain` row against `launch_preconditions.txt` (IMPRECISE) | not a discrepancy: the command that wrote the file filtered out the untracked audit re-run files | the row names its source and the filter is stated |
| 11 | timings 6.5 s, 6.3 s, 17 s not stored (UNSUPPORTED) | yes (shell `time`) | labelled as such |
| 12 | "none of the failures is a resolution artefact" rests on the two tiers only (IMPRECISE) | yes | scoped to the two tiers; the refined tier and the calculator were not run |
| 13 | `summary.md`: "None is a stage-2 stretch" next to "three q = 50 runs have a short stretch" (IMPRECISE) | yes | rewritten |
| 14 | `summary.md`: the medians are "close" although RMSE_pos is about 20 % higher (IMPRECISE) | yes | rewritten |
| 15 | the ledger did not exist; the figure's y axis cuts off seeds 60518 and 60515's peak without a note (IMPRECISE) | yes | this file; the caption now says so |

## Checker 2: `01_audit.md` sections 0 to 6

| # | problem (severity) | verified | disposition |
|---|---|---|---|
| 1 | "A1, A2, A3 and A5 ran with only results/ untracked" is false for the A5 re-run (WRONG) | yes (its `record_a5.json` lists modified tracked `reports/` files) | rewritten; no tool path was dirty |
| 2 | "at the root the objective is nearly flat, so efforts converge slowest there" (UNSUPPORTED) | yes: the largest effort change is at a q = 50 stretch node (0.000995) | sentence replaced by the numbers |
| 3 | "~1e-15 dW" for the closed-form test (UNSUPPORTED) | yes (2e-16 to 6e-16 in a rerun) | corrected |
| 4 | "every number comes from audit_numbers.json" (IMPRECISE) | yes (a few typed values, other source tables) | softened |
| 5 | "persistent small excess" for seed 50504 (IMPRECISE) | yes (0.0051 to 0.0227) | "persistent excess" with the range |
| 6 | "far inside the G-N tolerance" for EXP_root at 9.97e-5 (IMPRECISE) | yes | rewritten |
| 7 | the no-flip statement is only at the stored states (IMPRECISE, scope) | yes | qualifier added in section 0 |
| 8 | A3 covers 8 of 40 runs; "trained against exactly" rests on reading the runner (IMPRECISE, scope) | yes | softened |
| 9 | v2.0's check (ii-b) found 1e-6 (it is the limit; values 4.98e-7 / 4.1e-7) (IMPRECISE) | yes | corrected |
| 10 | "hand-computed" for tests that replay draws (IMPRECISE) | yes | section 0 and the table row reworded |
| 11 | "`protocols/` are those of `be2b7e9f`" (IMPRECISE) | yes (P2 added files) | "at the time of the audit" |
| 12 | "takes only `f_xi` and `F_xi`" (IMPRECISE) | yes (a test helper imports `g1_two_stage` / `g2_two_stage`) | stated |

## Checker 3: `01_audit.md` sections 7 to 11

| # | problem (severity) | verified | disposition |
|---|---|---|---|
| 1 | "did not recover" / "not recovered" for the stretches (WRONG for seed 50517, IMPRECISE for others) | **yes, a real error of the first draft:** in seed 50517 (q = 60) the failure node d = −66 has effort 0.07 at `u01625`, 9.4 to 15.9 at `u01750` to `u02950` and falls below 1 again only at `u03125`; 11 of the 28 nodes are below 1 at `u02000`; 47 of the 126 nodes left 1 and came back | section 0 table, section 8 (new recovery table `a07j`, new paragraph) and section 11 corrected; the summary, handoff and STATE do not say "did not recover" (the handoff gained the sentence) |
| 2 | dReach spread "up to 1.7e-3 in four jt runs (up to 4.6e-4 in the others)" (WRONG/IMPRECISE) | yes (other jt runs reach 9.8e-4) | numbers generated by the script |
| 3 | the reason alpha >= 1 at 14 nodes (IMPRECISE) | yes (concentration is at least 100 at all nodes; those 14 have alpha + beta of 218 or more) | corrected |
| 4 | "same kind as the others" for seed 50504 (UNSUPPORTED) | yes | removed; the tail position on eta_3 and RMSE is stated |
| 5 | the clamp share is a run-level aggregate, not at the stretch nodes (UNSUPPORTED) | yes | stated; the attribution is marked untested |
| 6 | "upper end" of kl / clip / adv-std (IMPRECISE) | yes (ranks given by the checker: clip_frac of seed 50514 is 10 of 20) | reworded with the exception |
| 7 | "2.48e+03" (IMPRECISE) | yes | prints 2475 |
| 8 | "far from the best response (2.1 to 34.4)" is the best-response range, not the distance (IMPRECISE) | yes (distance 2.10 to 33.89) | both stated |
| 9 | "Brent for reported states" but A7 uses the golden-section path (IMPRECISE) | yes (`brent=False` in `objective_at_node`) | stated in section 10 |

## Checker 4: `handoff.md`, `02_preregistration.md`, `03_checks.md`, `00_housekeeping.md`

| # | problem (severity) | verified | disposition |
|---|---|---|---|
| 1 | "64 test files" (WRONG) | yes (62) | corrected |
| 2 | wrong citation for the known failure (WRONG) | yes | `reports/t3/r1/03_checks.md` section 4 and `two_non_t3_failures.txt` |
| 3 | "never trained and not used" for seeds that earlier rounds used (WRONG as worded) | yes | `02_preregistration.md` is not edited after its commit; the wording is corrected in `03_checks.md` section 5 (the lock JSON key has the same wording and is locked) |
| 4 | "76 passed in 141 s" not stored (UNSUPPORTED) | yes | timing removed |
| 5 | the `git diff` statement holds only at `897982f2` (IMPRECISE) | yes | scoped |
| 6 | "189 to 194" (IMPRECISE) | yes (167 to 194) | corrected |
| 7 | "hand-computed" still in `01_audit.md` section 0 / table and `handoff.md` (IMPRECISE) | yes | reworded in both (the docstring of `tests/test_t3r2_semantics.py` still says "hand-computed examples"; a test file, not edited after the commit) |
| 8 | F6 says "marked as not tested" but the report did not (IMPRECISE) | yes | the report now says so |
| 9 | the clean-tree statement in `03_checks.md` section 4 (IMPRECISE) | yes | corrected like checker 2 #1 |
| 10 | "against the 20 recorded configs' reference" (IMPRECISE) | yes (two recorded configs in the test; 800 pairs by the reviewer) | reworded |
| 11 | the pre-registered launch command shows `--code-commit <lock commit>`, the launch used `be2b7e9f` (IMPRECISE) | yes | noted in `03_checks.md` section 5 (the flag is only recorded) |
| 12 | handoff: "0.0002 到 0.002" (IMPRECISE) | yes (0.00224) | "0.0002 到 0.0022" |
| 13 | handoff: "all checks passed" omits the first invocation (IMPRECISE) | yes | the relative-root episode is stated |
| 14 | handoff: "全部 CSV" excludes the wall-clock columns (IMPRECISE) | yes | stated |
| 15 | handoff: two of the three stretch runs are among the failing runs (IMPRECISE) | yes | stated |
| 16 | the handoff links resolve only after the push; a GitHub URL contains the owner handle (note) | yes | noted in `03_checks.md` section 5 |

## Summary

52 problems reported (3 of them substantive errors of fact or of the first draft's reading: the means table of the confirmation report, the "did not recover" statement for the stretches, and the stage-2 gain of seed 60518), all accepted and corrected before the push. No number of a result table (verdict, counts, intervals, per-seed values, gate values, calculator differences) was found wrong; the checkers recomputed the verdict, the intervals, the pooled counts, the 24 hypothesis rows, the flips and the 40 per-seed rows.
