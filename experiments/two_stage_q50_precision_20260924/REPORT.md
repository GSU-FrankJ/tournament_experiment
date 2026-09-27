# q=50, T=2 supplemental precision sample

All ten scheduled new runs are reported; new operation errors: 0. Parallel batch elapsed: 314.322 seconds.

This ten-run supplement was registered after the previous 30 outcomes were known. The new ten are reported separately and the combined 40 are a descriptive update. The original report and its ten restart groups remain unchanged; these ten seeds are not regrouped.

## Single-run results

| Cohort | Candidate discovery | Conditional joint certification | End-to-end |
|---|---|---|---|
| new10 | 4/10 (40.0%; Wilson 95% 16.8%–68.7%) | 3/4 (75.0%; Wilson 95% 30.1%–95.4%) | 3/10 (30.0%; Wilson 95% 10.8%–60.3%) |
| previous30 | 25/30 (83.3%; Wilson 95% 66.4%–92.7%) | 22/25 (88.0%; Wilson 95% 70.0%–95.8%) | 22/30 (73.3%; Wilson 95% 55.6%–85.8%) |
| pooled40 | 29/40 (72.5%; Wilson 95% 57.2%–83.9%) | 25/29 (86.2%; Wilson 95% 69.4%–94.5%) | 25/40 (62.5%; Wilson 95% 47.0%–75.8%) |

Every candidate receives the unchanged locked final verifier. Candidate discovery uses the first valid C development call with dReach/DeltaW <=0.01 and concentration <=0.04. Joint certification additionally requires valid development/final tiers, final dReach/DeltaW <=0.01, both refinement differences <=0.002, and dense C_all <=0.04. A no-candidate terminal evaluation is diagnostic only.

## Precision and interpretation

| Metric | Prior30 denominator | Pooled40 denominator | Prior width (pp) | Realized pooled width (pp) | Change (pp) | Prior-rate arithmetic reference width (pp) |
|---|---:|---:|---:|---:|---:|---:|
| candidate_discovery | 30 | 40 | 26.228 | 26.727 | 0.499 | 22.823 |
| conditional_joint_certification | 25 | 29 | 25.789 | 25.062 | -0.727 | 23.940 |
| end_to_end | 30 | 40 | 30.265 | 28.745 | -1.521 | 26.497 |

Negative width change means a narrower realized interval. More observations do not guarantee a narrower realized interval because the estimated rate and conditional candidate denominator can change. The arithmetic reference keeps the previous observed rate and uses the realized pooled denominator; its fractional implied count is only a formula comparison, not observed data or an additional interval estimate.

The decision to add these ten followed inspection of the previous outcomes. Pooled Wilson intervals are descriptive fixed-final-N summaries; nominal fixed-sample coverage is not guaranteed under an unmodeled outcome-dependent sampling-extension rule. These results are not an independent fixed-N confirmation and do not imply significance, publication readiness, guaranteed improvement, or three-stage performance.

## Every new run

| Seed | Operation | Candidate | Certification | C update | Candidate final dReach | Wall seconds | Failure reason |
|---|---|---|---|---:|---:|---:|---|
| 10231 | done | True | passed | 175 | 0.009946327 | 177.959 |  |
| 10232 | done | True | passed | 425 | 0.009053801 | 232.079 |  |
| 10233 | done | False | not_applicable_no_candidate | None | N/A | 307.119 | C budget exhausted without an eligible candidate; every C check was numerically valid and concentration passed; BR criterion never passed |
| 10234 | done | False | not_applicable_no_candidate | None | N/A | 311.135 | C budget exhausted without an eligible candidate; every C check was numerically valid and concentration passed; BR criterion never passed |
| 10235 | done | False | not_applicable_no_candidate | None | N/A | 304.285 | C budget exhausted without an eligible candidate; every C check was numerically valid and concentration passed; BR criterion never passed |
| 10236 | done | True | passed | 600 | 0.007507325 | 261.862 |  |
| 10237 | done | False | not_applicable_no_candidate | None | N/A | 295.657 | C budget exhausted without an eligible candidate; every C check was numerically valid and concentration passed; BR criterion never passed |
| 10238 | done | True | failed | 150 | 0.010865317 | 183.085 | Final rejection: main_pass |
| 10239 | done | False | not_applicable_no_candidate | None | N/A | 300.274 | C budget exhausted without an eligible candidate; every C check was numerically valid and concentration passed; BR criterion never passed |
| 10240 | done | False | not_applicable_no_candidate | None | N/A | 306.289 | C budget exhausted without an eligible candidate; every C check was numerically valid and concentration passed; BR criterion never passed |

## Computing cost

Summed process wall time is not parallel batch elapsed time. Costs include failed searches and rejected candidates.

| Metric | New10 sum | Previous30 sum | Pooled40 sum |
|---|---:|---:|---:|
| total_updates | 17050 | 42200 | 59250 |
| total_episodes | 8.7296e+06 | 2.16064e+07 | 3.0336e+07 |
| total_transitions | 1.35296e+07 | 3.33952e+07 | 4.69248e+07 |
| dev_calls | 502 | 1149 | 1651 |
| training_seconds | 2659.26 | 6610.08 | 9269.34 |
| final_eval_seconds | 20.3908 | 61.0156 | 81.4064 |
| total_wall_seconds | 2679.74 | 6671.46 | 9351.2 |
| process_cpu_seconds | 2678.48 | 6666.95 | 9345.43 |

## Verification and files

All 40 raw configurations, histories, stopping records and final results were rechecked using the previously validated read_run/rate helpers. Operation errors: 0. New parameters equal the previous source except run name, seed and output path. The recomputed old rates reproduce the unchanged original summary.

- PREREGISTRATION.md: design written before these ten started.
- manifest.json and launch.py: exact inputs and unchanged launcher.
- summary.json: all cohort rates, Wilson intervals, cost summaries, raw-derived rows and width comparisons.
- runs.csv: all ten new seeds, including failures and costs.
- pooled_runs.csv: all 40 audited rows with cohort labels.
- runs/ and logs/: original per-run records and execution logs.

This supplement makes no new restart grouping or restart-policy decision and launches no T3 runs.
