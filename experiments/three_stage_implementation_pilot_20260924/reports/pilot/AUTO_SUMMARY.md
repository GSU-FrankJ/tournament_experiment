# AUTO_SUMMARY — cohort `pilot`

Generated 2026-09-24T17:36:46+0000 by make_report.py. All values come from the CSVs in this directory; interpretation is in PILOT_REPORT.md.

## q=50: counts (implementation/debug outcome (not a reliability estimate))

N_planned=3, N_started=3, N_completed=3, N_operational_failed=0, N_pending=0

- candidate_discovery: 0/3  p=0  Wilson95=[0, 0.5615]
- conditional_certification: 0/0  p=—  Wilson95=[—, —] no candidates (N/A)
- end_to_end: 0/3  p=0  Wilson95=[0, 0.5615]

## q=60: counts (implementation/debug outcome (not a reliability estimate))

N_planned=3, N_started=3, N_completed=3, N_operational_failed=0, N_pending=0

- candidate_discovery: 0/3  p=0  Wilson95=[0, 0.5615]
- conditional_certification: 0/0  p=—  Wilson95=[—, —] no candidates (N/A)
- end_to_end: 0/3  p=0  Wilson95=[0, 0.5615]

## Runs

| run | state | outcome | B exit (local) | C local | final dReach/DW | refine dReach | refine EXP | dense conc | certification | min C dReach/DW (u) | complete |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t3_pilot_q50_s10411 | done | no_candidate_budget_exhausted | budget_forced (600) | 1800 | 0.045918 | 2.1679e-05 | 9.2545e-06 | 0.030039 | not_applicable_no_candidate | 0.026149 (2625) | True |
| t3_pilot_q50_s10412 | done | no_candidate_budget_exhausted | budget_forced (600) | 1800 | 0.068772 | 4.8986e-06 | 3.1433e-05 | 0.026508 | not_applicable_no_candidate | 0.061411 (2125) | True |
| t3_pilot_q50_s10413 | done | no_candidate_budget_exhausted | budget_forced (600) | 1800 | 0.076438 | 0.00077486 | 3.0736e-05 | 0.025407 | not_applicable_no_candidate | 0.058421 (2050) | True |
| t3_pilot_q60_s10401 | done | no_candidate_budget_exhausted | budget_forced (600) | 1800 | 0.02236 | 3.1482e-05 | 2.0681e-05 | 0.028422 | not_applicable_no_candidate | 0.022392 (2800) | True |
| t3_pilot_q60_s10402 | done | no_candidate_budget_exhausted | budget_forced (600) | 1800 | 0.030906 | 0.00010931 | 4.751e-05 | 0.026207 | not_applicable_no_candidate | 0.024756 (2475) | True |
| t3_pilot_q60_s10403 | done | no_candidate_budget_exhausted | budget_forced (600) | 1800 | 0.030323 | 4.3442e-05 | 6.2036e-05 | 0.029864 | not_applicable_no_candidate | 0.024258 (2450) | True |

## Completeness

- t3_pilot_q50_s10411: ok=True; failures=none; notes=none; unavailable=none
- t3_pilot_q50_s10412: ok=True; failures=none; notes=none; unavailable=none
- t3_pilot_q50_s10413: ok=True; failures=none; notes=none; unavailable=none
- t3_pilot_q60_s10401: ok=True; failures=none; notes=none; unavailable=none
- t3_pilot_q60_s10402: ok=True; failures=none; notes=none; unavailable=none
- t3_pilot_q60_s10403: ok=True; failures=none; notes=none; unavailable=none

## Files

runs.csv, phases.csv, verifier_calls.csv, verifier_summary.csv, verifier_stage_metrics.csv, deviations.csv, policy_profiles.csv, policy_asymmetry.csv, stage_metrics.csv, state_visitation.csv, coverage.csv, coverage_summary.csv, failure_diagnostics.csv, resources.csv, rates.csv, completeness.json, figures/*.png|pdf|csv. Per-run copies: runs/<run_id>/tables/.
