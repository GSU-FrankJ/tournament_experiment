# Publication validation (2026-09-27)

These checks validate the relocated publication code and supplied evidence. They do not add formal seeds, improve the reported solver success rates, or establish new equilibrium results.

| Check | Observed result |
|---|---|
| Existing T2 reporting regressions | 12 tests passed |
| Existing T3 numerical/collector/state-machine/reporting tests | 60 tests passed, no skips |
| T2 real smoke | Seven PPO updates (A2/B2/C3), final evaluation and artifacts completed; exit 0 |
| T3 real smoke, q50 and q60 | Both completed A/B/C, six updates each, final verification and both economic evaluation modes; exit 0 |
| T3 smoke report completeness | Both runs ok; no missing fields or unavailable checks |
| Archived T2 table recount | Reproduced all four primary-cohort counts, separate pooled counts and fixed restart-group results |
| T3 saved-weight replay | All 46 runs, 138 verifier evaluations; six checked metrics each; maximum absolute difference 0 |
| T3 compact formal summary regeneration | Completed for all 40 formal runs |

The original pilot q60 seed 10401 training history is supplied so the existing three-update regression can compare training losses, KL and returns with the original run. It passed without changing the numerical source.

The shared PPO, environment, verifier, analytic utility and base rollout/evaluation module match the original source byte-for-byte. T3 collection, metrics, training runner and its existing tests also match byte-for-byte. Other publication edits concern paths, interpreter discovery, output relocation, documentation and report interfaces.

Smoke outputs and regenerated reports are intentionally ignored and are not included as new scientific results. Full formal training was not repeated for publication. No source experiment or raw result directory was changed.

Re-run the commands in the [T2 guide](two_stage/README.md) and [T3 guide](three_stage/README.md). The checks were performed in the [recorded numerical environment](REPRODUCIBILITY.md); identical training on another software stack is not asserted.
