# T35: Confirmation verdict

- priority: core; status: regenerated; tier: final (G-N compares development and final)
- sources: `results/v2_T2_locked/confirmation_analysis/pass_counts.csv`, `results/v2_T2_locked/confirmation_analysis/verdict.json`, `results/v2_T2_locked/confirmation_analysis/per_run.csv`, `tools/v2/confirmation_analysis.py`
- built by: `tools/v2/report/sec_locked_a.py:build_t35`; base commit `cb0b541`
- transformation: Per-q rows: results/v2_T2_locked/confirmation_analysis/pass_counts.csv, values as in the file; overall row: verdict.json (rule, per_q, overall, agreement counts). Check: n_pass, n_expected and the Clopper-Pearson bounds recounted from per_run.csv with tools/v2/confirmation_analysis.py:clopper_pearson (scipy beta quantiles) equal the file (max abs diff 0.0); overall = both q pass is consistent with per-q verdicts: True.

Confirmation verdict under protocol v1.1: per q passes out of 20 with the exact 95% Clopper-Pearson CI and the rule, then the overall verdict.

| scope | q | n_expected | n_completed | n_missing | n_failed_exception | n_incomplete | n_global_rng_violation | n_pass | pass_rate | cp95_lo | cp95_hi | rule | q_passes_rule | n_G-A_pass | n_G-F_pass | n_G-N_pass | n_v1_0_run_pass | verdict | per_q | seeds | n_agreement_rows | n_all_agree | root |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q=50 | 50 | 20 | 20 | 0 | 0 | 0 | 0 | 20 | 1 | 0.8316 | 1 | >= 18 of 20 | True | 20 | 20 | 20 | 20 | pass |  | 20501-20520 |  |  | results/v2_T2_locked/confirmation |
| q=60 | 60 | 20 | 20 | 0 | 0 | 0 | 0 | 20 | 1 | 0.8316 | 1 | >= 18 of 20 | True | 20 | 20 | 20 | 18 | pass |  | 20501-20520 |  |  | results/v2_T2_locked/confirmation |
| overall (both q) |  |  |  |  |  |  |  |  |  |  |  | each q >= 18 of 20 runs pass; both q |  |  |  |  |  | PASS | {"50": true, "60": true} | 20501-20520 | 40 | 40 | results/v2_T2_locked/confirmation |
