# T47: Cusp diagnostic

- priority: core; status: generated; tier: tier-independent
- sources: `results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv`, `results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv`, `results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json`, `results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv`, `results/v2_T2_locked/cusp_diagnostic/curves_every500.csv`, `results/v2_T2_locked/rehearsal_analysis/lr_schedule_check.csv`
- built by: `tools/v2/report/sec_ext.py:build_t47`; base commit `cb0b541`
- transformation: Supervised fit of Pilot 4 section 1d re-run with logging every 500 Adam steps on the recovery grid (tools/v2/cusp_diagnostic.py). Blocks: first_crossing_summary = thresholds_summary.csv (median/min/max over 5 inits of the first logged step with |peak error at d = 0| < 0.05 / 0.03 / 0.01, the location-free variants, and RMSE/e2*(0) and the location-free error at the 0.05 crossing); first_crossing_per_init = thresholds_per_init.csv in long format (recomputed here from curves_every500.csv: max abs diff 0); rl_actor_steps = RL actor (minibatch) steps of the locked Phase A computed from the protocol (512 stage-2 rows per update, minibatch 256, 10 epochs); rl_actor_steps_check = v1.0 rehearsal runs whose histories give the same counts; identity_vs_pilot4 = float32 loss logs identical to the Pilot 4 fits at every common step. Tier-independent (recovery grid).

Cusp diagnostic (no RL runs): how fast the supervised fit of the stage-2 actor reaches the d = 0 peak, against the RL actor-step counts.

| block | q | init_seed | quantity | value | n_reached | median | min | max | n | source |
|---|---|---|---|---|---|---|---|---|---|---|
| first_crossing_summary | 50 |  | step_abs_peak_lt_0.05 |  | 5 | 8500 | 7500 | 9000 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 50 |  | rmse_at_peak_lt_0.05 |  | 5 | 0.01597 | 0.0153 | 0.01717 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 50 |  | locfree_at_peak_lt_0.05 |  | 5 | -0.03457 | -0.04402 | -0.03131 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 50 |  | step_abs_locfree_lt_0.05 |  | 5 | 8000 | 7500 | 8500 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 50 |  | step_abs_peak_lt_0.03 |  | 5 | 1.6e+04 | 8500 | 1.7e+04 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 50 |  | step_abs_locfree_lt_0.03 |  | 5 | 1.6e+04 | 8500 | 1.65e+04 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 50 |  | step_abs_peak_lt_0.01 |  | 5 | 1.9e+04 | 1.45e+04 | 5.5e+04 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 50 |  | step_abs_locfree_lt_0.01 |  | 5 | 1.9e+04 | 1.45e+04 | 5.5e+04 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 60 |  | step_abs_peak_lt_0.05 |  | 5 | 9500 | 8500 | 1.05e+04 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 60 |  | rmse_at_peak_lt_0.05 |  | 5 | 0.0154 | 0.009977 | 0.02182 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 60 |  | locfree_at_peak_lt_0.05 |  | 5 | -0.04126 | -0.04694 | -0.03473 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 60 |  | step_abs_locfree_lt_0.05 |  | 5 | 9500 | 8500 | 1.05e+04 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 60 |  | step_abs_peak_lt_0.03 |  | 5 | 1.2e+04 | 9500 | 1.6e+04 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 60 |  | step_abs_locfree_lt_0.03 |  | 5 | 1.2e+04 | 9500 | 1.6e+04 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 60 |  | step_abs_peak_lt_0.01 |  | 5 | 1.95e+04 | 1.8e+04 | 3.5e+04 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_summary | 60 |  | step_abs_locfree_lt_0.01 |  | 5 | 1.95e+04 | 1.8e+04 | 3.5e+04 |  | results/v2_T2_locked/cusp_diagnostic/thresholds_summary.csv |
| first_crossing_per_init | 50 | 0 | step_abs_peak_lt_0.05 | 8500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 1 | step_abs_peak_lt_0.05 | 8000 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 2 | step_abs_peak_lt_0.05 | 7500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 3 | step_abs_peak_lt_0.05 | 8500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 4 | step_abs_peak_lt_0.05 | 9000 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 0 | step_abs_peak_lt_0.05 | 9500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 1 | step_abs_peak_lt_0.05 | 1e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 2 | step_abs_peak_lt_0.05 | 8500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 3 | step_abs_peak_lt_0.05 | 9000 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 4 | step_abs_peak_lt_0.05 | 1.05e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 0 | rmse_at_peak_lt_0.05 | 0.0153 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 1 | rmse_at_peak_lt_0.05 | 0.01627 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 2 | rmse_at_peak_lt_0.05 | 0.01597 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 3 | rmse_at_peak_lt_0.05 | 0.01537 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 4 | rmse_at_peak_lt_0.05 | 0.01717 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 0 | rmse_at_peak_lt_0.05 | 0.0154 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 1 | rmse_at_peak_lt_0.05 | 0.02182 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 2 | rmse_at_peak_lt_0.05 | 0.009977 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 3 | rmse_at_peak_lt_0.05 | 0.0175 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 4 | rmse_at_peak_lt_0.05 | 0.01139 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 0 | locfree_at_peak_lt_0.05 | -0.03283 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 1 | locfree_at_peak_lt_0.05 | -0.03131 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 2 | locfree_at_peak_lt_0.05 | -0.03457 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 3 | locfree_at_peak_lt_0.05 | -0.03472 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 4 | locfree_at_peak_lt_0.05 | -0.04402 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 0 | locfree_at_peak_lt_0.05 | -0.04126 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 1 | locfree_at_peak_lt_0.05 | -0.04158 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 2 | locfree_at_peak_lt_0.05 | -0.03707 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 3 | locfree_at_peak_lt_0.05 | -0.04694 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 4 | locfree_at_peak_lt_0.05 | -0.03473 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 0 | step_abs_locfree_lt_0.05 | 8500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 1 | step_abs_locfree_lt_0.05 | 8000 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 2 | step_abs_locfree_lt_0.05 | 7500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 3 | step_abs_locfree_lt_0.05 | 8500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 4 | step_abs_locfree_lt_0.05 | 8000 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 0 | step_abs_locfree_lt_0.05 | 9500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 1 | step_abs_locfree_lt_0.05 | 1e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 2 | step_abs_locfree_lt_0.05 | 8500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 3 | step_abs_locfree_lt_0.05 | 9000 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 4 | step_abs_locfree_lt_0.05 | 1.05e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 0 | step_abs_peak_lt_0.03 | 1.6e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 1 | step_abs_peak_lt_0.03 | 8500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 2 | step_abs_peak_lt_0.03 | 1.7e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 3 | step_abs_peak_lt_0.03 | 1e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 4 | step_abs_peak_lt_0.03 | 1.65e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 0 | step_abs_peak_lt_0.03 | 1.25e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 1 | step_abs_peak_lt_0.03 | 1.6e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 2 | step_abs_peak_lt_0.03 | 9500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 3 | step_abs_peak_lt_0.03 | 1.2e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 4 | step_abs_peak_lt_0.03 | 1.1e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 0 | step_abs_locfree_lt_0.03 | 1.6e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 1 | step_abs_locfree_lt_0.03 | 8500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 2 | step_abs_locfree_lt_0.03 | 1.65e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 3 | step_abs_locfree_lt_0.03 | 1e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 4 | step_abs_locfree_lt_0.03 | 1.65e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 0 | step_abs_locfree_lt_0.03 | 1.25e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 1 | step_abs_locfree_lt_0.03 | 1.6e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 2 | step_abs_locfree_lt_0.03 | 9500 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 3 | step_abs_locfree_lt_0.03 | 1.2e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 4 | step_abs_locfree_lt_0.03 | 1.1e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 0 | step_abs_peak_lt_0.01 | 1.6e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 1 | step_abs_peak_lt_0.01 | 2.1e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 2 | step_abs_peak_lt_0.01 | 5.5e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 3 | step_abs_peak_lt_0.01 | 1.45e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 4 | step_abs_peak_lt_0.01 | 1.9e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 0 | step_abs_peak_lt_0.01 | 1.85e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 1 | step_abs_peak_lt_0.01 | 3.5e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 2 | step_abs_peak_lt_0.01 | 1.95e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 3 | step_abs_peak_lt_0.01 | 2.1e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 4 | step_abs_peak_lt_0.01 | 1.8e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 0 | step_abs_locfree_lt_0.01 | 1.6e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 1 | step_abs_locfree_lt_0.01 | 2.1e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 2 | step_abs_locfree_lt_0.01 | 5.5e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 3 | step_abs_locfree_lt_0.01 | 1.45e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 50 | 4 | step_abs_locfree_lt_0.01 | 1.9e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 0 | step_abs_locfree_lt_0.01 | 1.85e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 1 | step_abs_locfree_lt_0.01 | 3.5e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 2 | step_abs_locfree_lt_0.01 | 1.95e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 3 | step_abs_locfree_lt_0.01 | 2.1e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| first_crossing_per_init | 60 | 4 | step_abs_locfree_lt_0.01 | 1.8e+04 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/thresholds_per_init.csv |
| rl_actor_steps | 50 |  | phase_A_updates | 1600 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 50 |  | rows_per_update | 512 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 50 |  | minibatch | 256 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 50 |  | epochs | 10 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 50 |  | minibatches_per_update | 2 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 50 |  | actor_steps_per_update | 20 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 50 |  | actor_steps_all_A | 32000 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 50 |  | actor_steps_last_400 | 8000 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 60 |  | phase_A_updates | 1600 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 60 |  | rows_per_update | 512 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 60 |  | minibatch | 256 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 60 |  | epochs | 10 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 60 |  | minibatches_per_update | 2 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 60 |  | actor_steps_per_update | 20 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 60 |  | actor_steps_all_A | 32000 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps | 60 |  | actor_steps_last_400 | 8000 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/rl_actor_steps.json (from protocols/v2_T2_locked.json) |
| rl_actor_steps_check | 50 |  | rehearsal runs with actor_minibatch_steps_A = 32000 | 10 |  |  |  |  | 10 | results/v2_T2_locked/rehearsal_analysis/lr_schedule_check.csv |
| rl_actor_steps_check | 50 |  | rehearsal runs with actor_minibatch_steps_A_last400 = 8000 | 10 |  |  |  |  | 10 | results/v2_T2_locked/rehearsal_analysis/lr_schedule_check.csv |
| rl_actor_steps_check | 60 |  | rehearsal runs with actor_minibatch_steps_A = 32000 | 10 |  |  |  |  | 10 | results/v2_T2_locked/rehearsal_analysis/lr_schedule_check.csv |
| rl_actor_steps_check | 60 |  | rehearsal runs with actor_minibatch_steps_A_last400 = 8000 | 10 |  |  |  |  | 10 | results/v2_T2_locked/rehearsal_analysis/lr_schedule_check.csv |
| identity_vs_pilot4 | 50 | 0 | n_common_steps | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 1 | n_common_steps | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 2 | n_common_steps | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 3 | n_common_steps | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 4 | n_common_steps | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 0 | n_common_steps | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 1 | n_common_steps | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 2 | n_common_steps | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 3 | n_common_steps | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 4 | n_common_steps | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 0 | n_identical_float32 | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 1 | n_identical_float32 | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 2 | n_identical_float32 | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 3 | n_identical_float32 | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 4 | n_identical_float32 | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 0 | n_identical_float32 | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 1 | n_identical_float32 | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 2 | n_identical_float32 | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 3 | n_identical_float32 | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 4 | n_identical_float32 | 300 |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 0 | identical_to_pilot4 | True |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 1 | identical_to_pilot4 | True |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 2 | identical_to_pilot4 | True |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 3 | identical_to_pilot4 | True |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 50 | 4 | identical_to_pilot4 | True |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 0 | identical_to_pilot4 | True |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 1 | identical_to_pilot4 | True |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 2 | identical_to_pilot4 | True |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 3 | identical_to_pilot4 | True |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
| identity_vs_pilot4 | 60 | 4 | identical_to_pilot4 | True |  |  |  |  |  | results/v2_T2_locked/cusp_diagnostic/identity_vs_pilot4.csv |
