# T37: Distributions of the gate metrics

- priority: core; status: generated; tier: final and development
- sources: `results/v2_T2_locked/confirmation_analysis/distributions.csv`, `results/v2_T2_locked/confirmation_analysis/per_run.csv`, `confirmation gates.json` (40 files)
- built by: `tools/v2/report/sec_locked_a.py:build_t37`; base commit `cb0b541`
- transformation: Rows with row_source 'found' are confirmation_analysis/distributions.csv as is (values equal the file; recomputed from per_run.csv with the same quantile method, max abs diff 0.0). Added rows: the G-N gate metrics |eta_dev - eta_final| and |gmax_dev - gmax_final| from per_run.csv, and dev - final of the tier-independent gate metrics (RMSE, tail, S1) from gates.json reported.end_of_A / end_of_B dev_minus_final; quantiles with pandas linear interpolation (= numpy linear) as in the analysis script.

Confirmation distributions (n = 20 runs per q) of every gate metric and of the dev - final differences; metric units as in T36.

| root | q | metric | gate_role | n | min | p10 | p25 | median | p75 | p90 | max | row_source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| results/v2_T2_locked/confirmation | 50 | eta_final | G-A (final tier) | 20 | 0.000804 | 0.0008979 | 0.0009915 | 0.001365 | 0.001701 | 0.002256 | 0.00329 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 50 | eta_dev | development tier (G-N input) | 20 | 0.0006746 | 0.0007512 | 0.0007849 | 0.001199 | 0.001677 | 0.002238 | 0.003271 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 50 | eta_dev_minus_final | dev - final, signed (G-N uses the absolute value) | 20 | -0.000261 | -0.0002378 | -0.0001932 | -7.204e-05 | -1.491e-05 | 0 | 0 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 50 | rmse | G-A (tier-independent) | 20 | 0.01436 | 0.01665 | 0.01905 | 0.0218 | 0.02649 | 0.02872 | 0.03403 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 50 | tail | G-A (tier-independent) | 20 | 0.004996 | 0.005436 | 0.008148 | 0.00887 | 0.01006 | 0.01062 | 0.01215 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 50 | gmax_final | G-F (final tier) | 20 | 0.0009173 | 0.0009739 | 0.001122 | 0.001527 | 0.001812 | 0.002789 | 0.00329 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 50 | gmax_dev | development tier (G-N input) | 20 | 0.0007117 | 0.0007721 | 0.001052 | 0.00147 | 0.001795 | 0.002806 | 0.003271 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 50 | gmax_dev_minus_final | dev - final, signed (G-N uses the absolute value) | 20 | -0.000261 | -0.0002378 | -0.0001564 | -1.86e-05 | 0 | 1.98e-06 | 2.943e-05 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 50 | s1 | S1 (secondary) | 20 | 0.004058 | 0.01313 | 0.02454 | 0.03776 | 0.05991 | 0.07702 | 0.08619 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 50 | stage1_rel_err_signed | S1 signed value (reported) | 20 | -0.07694 | -0.06184 | -0.04447 | 0.00111 | 0.0297 | 0.05408 | 0.08619 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 50 | eta_dev_minus_final_abs | G-N criterion 1 | 20 | 0 | 0 | 1.491e-05 | 7.204e-05 | 0.0001932 | 0.0002378 | 0.000261 | generated: per_run.csv \|eta_dev_minus_final\| |
| results/v2_T2_locked/confirmation | 50 | gmax_dev_minus_final_abs | G-N criterion 2 | 20 | 0 | 0 | 9.049e-06 | 2.009e-05 | 0.0001564 | 0.0002378 | 0.000261 | generated: per_run.csv \|gmax_dev_minus_final\| |
| results/v2_T2_locked/confirmation | 50 | rmse_dev_minus_final | dev - final of a tier-independent gate metric | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | generated: gates.json reported.*.dev_minus_final |
| results/v2_T2_locked/confirmation | 50 | tail_dev_minus_final | dev - final of a tier-independent gate metric | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | generated: gates.json reported.*.dev_minus_final |
| results/v2_T2_locked/confirmation | 50 | s1_dev_minus_final | dev - final of a tier-independent criterion | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | generated: gates.json reported.*.dev_minus_final |
| results/v2_T2_locked/confirmation | 60 | eta_final | G-A (final tier) | 20 | 0.0004322 | 0.0004671 | 0.0005148 | 0.0007876 | 0.001252 | 0.001529 | 0.002318 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 60 | eta_dev | development tier (G-N input) | 20 | 0.0004039 | 0.0004208 | 0.0004747 | 0.0007222 | 0.001069 | 0.001424 | 0.002283 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 60 | eta_dev_minus_final | dev - final, signed (G-N uses the absolute value) | 20 | -0.000225 | -0.0001898 | -0.0001531 | -7.055e-05 | -9.74e-06 | 0 | 1.11e-16 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 60 | rmse | G-A (tier-independent) | 20 | 0.01389 | 0.01537 | 0.01584 | 0.01816 | 0.02495 | 0.03523 | 0.03756 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 60 | tail | G-A (tier-independent) | 20 | 0.006131 | 0.007559 | 0.008662 | 0.009395 | 0.01075 | 0.0123 | 0.01378 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 60 | gmax_final | G-F (final tier) | 20 | 0.0004471 | 0.000489 | 0.0005787 | 0.0009341 | 0.001254 | 0.001829 | 0.003638 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 60 | gmax_dev | development tier (G-N input) | 20 | 0.0004039 | 0.0004381 | 0.0005771 | 0.000837 | 0.001207 | 0.001847 | 0.003649 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 60 | gmax_dev_minus_final | dev - final, signed (G-N uses the absolute value) | 20 | -0.000225 | -0.0001898 | -9.822e-05 | -4.072e-06 | 2.884e-06 | 1.187e-05 | 2.058e-05 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 60 | s1 | S1 (secondary) | 20 | 0.001422 | 0.01029 | 0.02453 | 0.04548 | 0.06988 | 0.08733 | 0.153 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 60 | stage1_rel_err_signed | S1 signed value (reported) | 20 | -0.1135 | -0.0549 | -0.03856 | -0.00175 | 0.0468 | 0.07457 | 0.153 | found: confirmation_analysis/distributions.csv |
| results/v2_T2_locked/confirmation | 60 | eta_dev_minus_final_abs | G-N criterion 1 | 20 | 0 | 9.992e-17 | 9.74e-06 | 7.055e-05 | 0.0001531 | 0.0001898 | 0.000225 | generated: per_run.csv \|eta_dev_minus_final\| |
| results/v2_T2_locked/confirmation | 60 | gmax_dev_minus_final_abs | G-N criterion 2 | 20 | 0 | 1.885e-07 | 4.808e-06 | 1.889e-05 | 9.822e-05 | 0.0001898 | 0.000225 | generated: per_run.csv \|gmax_dev_minus_final\| |
| results/v2_T2_locked/confirmation | 60 | rmse_dev_minus_final | dev - final of a tier-independent gate metric | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | generated: gates.json reported.*.dev_minus_final |
| results/v2_T2_locked/confirmation | 60 | tail_dev_minus_final | dev - final of a tier-independent gate metric | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | generated: gates.json reported.*.dev_minus_final |
| results/v2_T2_locked/confirmation | 60 | s1_dev_minus_final | dev - final of a tier-independent criterion | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | generated: gates.json reported.*.dev_minus_final |
