# Gaps: items not found and not generable, and UNKNOWN values

## T09: test_invariants: maximum residual and number of positive residuals

- reason: UNKNOWN: tests/test_v2_verifier.py::test_invariants asserts every residual <= 1e-12 DW on 36 evaluations but writes no values; no file records them (the logged sets of T09 cover the same relations)

## T16: test-suite result at cd760fd (Pilot 3 and Phase A extension launch)

- reason: UNKNOWN: no report or log states it; re-running the suite at old commits is not allowed

## T16: test-suite result at 5d50a9d (dirty-flag re-run launch)

- reason: UNKNOWN: no report or log states it; re-running the suite at old commits is not allowed

## T16: test-suite result at 5b07293 (v1.0 rehearsal and Check 2 launch)

- reason: UNKNOWN: no report or log states it; re-running the suite at old commits is not allowed

## T16: test-suite result at f6838ec (confirmation launch)

- reason: UNKNOWN: no report or log states it; re-running the suite at old commits is not allowed

## T18: commit / tree of the v1.0 pre-lock insurance run

- reason: UNKNOWN: run in scratch before the lock commit and not kept; the report does not record the tree

## T18: commit / tree and compared-field list of the v1.1 pre-lock insurance run

- reason: UNKNOWN: run in scratch before the lock commit and not kept; the report states only 'all training-relevant state matched'

## T57: impact of the clipped-Beta likelihood issue (issue 7)

- reason: UNKNOWN: no run log records the number of Beta draws that hit the [1e-6, 1-1e-6] clamp (v2_updates.csv and train_history.json fields checked) and no report analyses it

## T59: commands to reproduce: Phase 2 opening checks

- reason: UNKNOWN: reports/v2/phase2_opening_checks.md has no fenced command block under a 'reproduce' heading (the row says so; no command is inferred)

## T59: commands to reproduce: Pilot 3

- reason: UNKNOWN: reports/v2/pilot3_continuation_mode.md has no fenced command block under a 'reproduce' heading (the row says so; no command is inferred)

