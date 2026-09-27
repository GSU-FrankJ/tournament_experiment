# q=50, T=2 prospective random-restart evaluation

All 30 scheduled seeds were assigned to ten fixed non-overlapping triples before outcomes. k=1,2,3 use their fixed prefixes. All runs are evaluated even after an earlier success.

Candidate = first valid development C call with dReach/Δw ≤0.01 and concentration ≤0.04. Joint certification additionally requires valid development and final tiers, final dReach/Δw ≤0.01, both dReach and exploitability refinement differences ≤0.002, and dense concentration ≤0.04. No-candidate terminal evaluations are diagnostic only.

- candidate_discovery: 25/30 (83.3%; Wilson 95% 66.4%–92.7%)
- conditional_certification: 22/25 (88.0%; Wilson 95% 70.0%–95.8%)
- end_to_end: 22/30 (73.3%; Wilson 95% 55.6%–85.8%)

| Fixed budget | Candidate discovery | Joint certification | Reconstructed attempts |
|---|---|---|---|
| k=1 | 8/10 (80.0%; Wilson 95% 49.0%–94.3%) | 8/10 (80.0%; Wilson 95% 49.0%–94.3%) | 10 |
| k=2 | 10/10 (100.0%; Wilson 95% 72.2%–100.0%) | 9/10 (90.0%; Wilson 95% 59.6%–98.2%) | 12 |
| k=3 | 10/10 (100.0%; Wilson 95% 72.2%–100.0%) | 9/10 (90.0%; Wilson 95% 59.6%–98.2%) | 13 |

Rescued groups: 1. Operation errors: 0.
Prespecified decision: **retain_useful_T2_pilot**.

This is a descriptive engineering decision with ten paired group replications of one configuration. Prefix results are dependent; Wilson intervals are marginal intervals, not an interval for the gain. There is no significance or reliability-guarantee claim. The per-run success rate is unchanged by regrouping. T2 results cannot certify T3; any T3 pilot requires a separate preregistration.

## Resource costs

Actual evaluation costs sum all scheduled runs. Reconstructed sequential costs stop after the first jointly certified candidate, or exhaust k attempts. They are counterfactual costs from the observed runs; sums of process wall time are not parallel batch elapsed time. Missing costs are explicit.

| Metric | Actual all runs | Reconstructed k1 | Reconstructed k2 | Reconstructed k3 |
|---|---:|---:|---:|---:|
| total_updates | 42200 (missing 0) | 14400 (missing 0) | 16875 (missing 0) | 18750 (missing 0) |
| total_episodes | 2.16064e+07 (missing 0) | 7.3728e+06 (missing 0) | 8.64e+06 (missing 0) | 9.6e+06 (missing 0) |
| total_transitions | 3.33952e+07 (missing 0) | 1.13472e+07 (missing 0) | 1.32864e+07 (missing 0) | 1.47456e+07 (missing 0) |
| dev_calls | 1149 (missing 0) | 397 (missing 0) | 460 (missing 0) | 517 (missing 0) |
| training_seconds | 6610.08 (missing 0) | 2227.01 (missing 0) | 2612.42 (missing 0) | 2902.74 (missing 0) |
| final_eval_seconds | 61.0156 (missing 0) | 20.1532 (missing 0) | 24.179 (missing 0) | 26.2281 (missing 0) |
| total_wall_seconds | 6671.46 (missing 0) | 2247.25 (missing 0) | 2636.71 (missing 0) | 2929.09 (missing 0) |
| process_cpu_seconds | 6666.95 (missing 0) | 2246.01 (missing 0) | 2635.19 (missing 0) | 2927.41 (missing 0) |

Actual parallel batch elapsed seconds: 828.035521030426.

## Fixed groups

| Group | Seeds in fixed order | First certified position | k1 | k2 | k3 |
|---|---|---:|---|---|---|
| 1 | [10201, 10202, 10203] | 1 | True | True | True |
| 2 | [10204, 10205, 10206] | 1 | True | True | True |
| 3 | [10207, 10208, 10209] | 1 | True | True | True |
| 4 | [10210, 10211, 10212] | 1 | True | True | True |
| 5 | [10213, 10214, 10215] | 1 | True | True | True |
| 6 | [10216, 10217, 10218] | none | False | False | False |
| 7 | [10219, 10220, 10221] | 1 | True | True | True |
| 8 | [10222, 10223, 10224] | 2 | False | True | True |
| 9 | [10225, 10226, 10227] | 1 | True | True | True |
| 10 | [10228, 10229, 10230] | 1 | True | True | True |

## Every scheduled run

| Seed | Operation | Candidate | Certification | C update | Final dReach | Dense concentration | Issue |
|---|---|---|---|---:|---:|---:|---|
| 10201 | done | True | passed | 100 | 0.007565682790183437 | 0.03847547667606658 |  |
| 10202 | done | True | passed | 650 | 0.009606689194546159 | 0.03360234771538974 |  |
| 10203 | done | True | passed | 175 | 0.007792655936961768 | 0.0371903453997168 |  |
| 10204 | done | True | passed | 500 | 0.00917618378209073 | 0.0340341326878152 |  |
| 10205 | done | True | passed | 175 | 0.00941560713182421 | 0.03610095749742718 |  |
| 10206 | done | True | passed | 350 | 0.009740443482676997 | 0.03403138290910413 |  |
| 10207 | done | True | passed | 600 | 0.009664590639106296 | 0.033545666823284984 |  |
| 10208 | done | False | not_applicable_no_candidate | None | 0.04832431607689558 | 0.030600182651423873 |  |
| 10209 | done | False | not_applicable_no_candidate | None | 0.026591791203259607 | 0.03091322469295935 |  |
| 10210 | done | True | passed | 600 | 0.00913097321138423 | 0.03314728929323344 |  |
| 10211 | done | True | failed | 475 | 0.010125868817483563 | 0.035369405723328015 |  |
| 10212 | done | True | passed | 150 | 0.008580071682567136 | 0.03827353879922965 |  |
| 10213 | done | True | passed | 175 | 0.00904636091346589 | 0.037072887497719025 |  |
| 10214 | done | True | passed | 575 | 0.008658660571890686 | 0.03399667527338193 |  |
| 10215 | done | True | passed | 175 | 0.009883635142275637 | 0.036567467678930834 |  |
| 10216 | done | False | not_applicable_no_candidate | None | 0.023543933895788682 | 0.03134930224333215 |  |
| 10217 | done | True | failed | 100 | 0.01006584710625541 | 0.03682954353598147 |  |
| 10218 | done | False | not_applicable_no_candidate | None | 0.04954554576447667 | 0.03227264036603717 |  |
| 10219 | done | True | passed | 600 | 0.008340904038973918 | 0.03389593298902504 |  |
| 10220 | done | True | passed | 425 | 0.009483411028606503 | 0.034571987456886964 |  |
| 10221 | done | True | failed | 800 | 0.010059877490573743 | 0.03165924429256883 |  |
| 10222 | done | False | not_applicable_no_candidate | None | 0.011607040302830662 | 0.031242598799761555 |  |
| 10223 | done | True | passed | 625 | 0.009370750113401938 | 0.03367351889990567 |  |
| 10224 | done | True | passed | 150 | 0.009661414208266328 | 0.0372195013334327 |  |
| 10225 | done | True | passed | 550 | 0.009698258137134652 | 0.033949696026971066 |  |
| 10226 | done | True | passed | 150 | 0.00973785041932096 | 0.03743137445449745 |  |
| 10227 | done | True | passed | 100 | 0.008633099411607392 | 0.03856546457464658 |  |
| 10228 | done | True | passed | 150 | 0.009054865364358067 | 0.03793800223366693 |  |
| 10229 | done | True | passed | 775 | 0.009329741933231439 | 0.03351068452272401 |  |
| 10230 | done | True | passed | 225 | 0.009242235833234413 | 0.036833806000861474 |  |

## Recovery diagnostics by first certified position

These describe the returned candidates only and have no acceptance threshold. All available numerical recovery fields, counts and seed identities are included in summary.json.

| Position | Candidates | Stage1 absolute error mean | Stage2 positive-region RMSE mean | On-path RMSE mean |
|---|---:|---:|---:|---:|
| 1 | 8 | 1.93392 | 4.95682 | 4.92426 |
| 2 | 1 | 4.79251 | 5.08579 | 4.87233 |
| 3 | 0 | N/A | N/A | N/A |
