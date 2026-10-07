# MS-R3 P1: the first-layer d-weights and the rounding width of the RL actors of MS-R1 and MS-R2

Date: 2026-10-07. Spec: `reports/ms/r3/pi_record/20_ms_r3_prompt.md` section 2.3 (b). Tool `tools/ms/r3_actor_diagnostics.py` (tests `tests/test_ms_r3_actor_diagnostics.py`), outputs `results/ms_r3/rl_actor_diagnostics/` (`per_export.csv`, `per_arm.csv`, `relation.csv`, `regime.csv`, `summary.txt`, `manifest.json`). Every table below is printed by `python reports/ms/r3/report_scripts/screen_tables.py --block <diag_...>` (default `--diag-dir results/ms_r3/rl_actor_diagnostics`). No RL training was run for this report: the tool reads the weight exports (every 25 updates) of existing runs.

## 1. What was read

| root | runs | exports | terminal-stage exports |
|---|---|---|---|
| MS-R2 pilot, `.../p2-gate-ms-base2400-e41857/results/ms_r2/pilot` | 120 | 16,320 | 13,440 |
| MS-R1 pilot, `.../p2-gate-ms-base2400-e41857/results/ms_r1/pilot` | 120 | 14,309 | 11,520 |
| MS-R1 base (`MS_base`), `.../ms-r1-multistage-development-afebf2/results/ms_r1/base` | 20 | 1,760 | 1,280 |
| total | 260 | 32,389 | 26,240 |

(`summary.txt` and `manifest.json` of the output; the terminal-stage exports are those of stage 2, tagged by `ms_updates.csv`; the MS-R1 rule arms stop their stage 1 early, not their terminal stage, which runs 2400 updates.) All exports are `t1` exports (none carries an `actor_variant` entry), read by the variant-aware reload of `agents/ppo_curriculum.py:mean_effort_numpy`.

Per export: the first-layer weights on the d input in units of d / B (column 1 of `actor.l1.weight`; max |w|, the quantiles 0-100 % of |w| over the 64 units and the number of units above 1, 2, 3, 5); `e_hat_2(0)` from the export at the stage-2 observation of d = 0; `e2*(0)` the closed form (`utils.theory_multistage.g2_two_stage`, equal to the verifier's `g2_at_0`: 70.0 at q = 50, 58.333 at q = 60); the gap and `w_eff = gap / (e2*(0) / 2q)` in units of d. `w_eff` is computed for terminal-stage exports only (after the freeze the live actor is trained on stage 1). The reload agrees with the verifier's `e2_at_0` of the analysis tables at the final terminal-stage export of every run (block `diag_check`):

| source | runs compared | max abs diff of e_hat_2(0) (effort units) | max abs diff of w_eff (units of d) | analysis table |
|---|---|---|---|---|
| ms_r2_pilot | 120 | 1.85e-05 | 3.80e-05 | results/ms_r2/analysis/per_run.csv |
| ms_r1_pilot | 120 | 2.36e-05 | 3.81e-05 | results/ms_r1/analysis/per_run.csv |
| ms_r1_base | 20 | 1.32e-05 | 2.07e-05 | results/ms_r1/analysis/per_run.csv |

## 2. The first-layer d-weights and the rounding width at the freeze and along training (block `diag_arm`; medians over the ten seeds; "u" is the terminal-stage export used)

| source | arm | q | u400: max abs w / w_eff | u1200: max abs w / w_eff | u2000: max abs w / w_eff | final: max abs w / w_eff |
|---|---|---|---|---|---|---|
| ms_r2_pilot | NL_bb_s1 | 50 | 1.104 / 12.93 (u400) | 1.210 / 6.00 (u1200) | 1.243 / 8.19 (u2000) | 1.260 / 5.02 (u2800) |
| ms_r2_pilot | NL_bb_s1 | 60 | 1.122 / 11.85 (u400) | 1.213 / 6.53 (u1200) | 1.252 / 6.60 (u2000) | 1.284 / 4.35 (u2800) |
| ms_r2_pilot | NL_bb_s4 | 50 | 1.104 / 12.93 (u400) | 1.210 / 6.00 (u1200) | 1.243 / 8.19 (u2000) | 1.255 / 5.04 (u2800) |
| ms_r2_pilot | NL_bb_s4 | 60 | 1.122 / 11.85 (u400) | 1.213 / 6.53 (u1200) | 1.252 / 6.60 (u2000) | 1.280 / 6.34 (u2800) |
| ms_r2_pilot | NL_bb_s16 | 50 | 1.104 / 12.93 (u400) | 1.210 / 6.00 (u1200) | 1.243 / 8.19 (u2000) | 1.255 / 5.38 (u2800) |
| ms_r2_pilot | NL_bb_s16 | 60 | 1.122 / 11.85 (u400) | 1.213 / 6.53 (u1200) | 1.252 / 6.60 (u2000) | 1.270 / 5.71 (u2800) |
| ms_r2_pilot | NL_st_s1 | 50 | 1.046 / 7.67 (u400) | 1.154 / 5.93 (u1200) | 1.203 / 4.69 (u2000) | 1.237 / 3.81 (u2800) |
| ms_r2_pilot | NL_st_s1 | 60 | 1.106 / 8.29 (u400) | 1.216 / 6.83 (u1200) | 1.271 / 5.43 (u2000) | 1.303 / 5.26 (u2800) |
| ms_r2_pilot | NL_st_s4 | 50 | 1.046 / 7.67 (u400) | 1.154 / 5.93 (u1200) | 1.203 / 4.69 (u2000) | 1.231 / 4.28 (u2800) |
| ms_r2_pilot | NL_st_s4 | 60 | 1.106 / 8.29 (u400) | 1.216 / 6.83 (u1200) | 1.271 / 5.43 (u2000) | 1.298 / 4.95 (u2800) |
| ms_r2_pilot | NL_st_s16 | 50 | 1.046 / 7.67 (u400) | 1.154 / 5.93 (u1200) | 1.203 / 4.69 (u2000) | 1.225 / 4.25 (u2800) |
| ms_r2_pilot | NL_st_s16 | 60 | 1.106 / 8.29 (u400) | 1.216 / 6.83 (u1200) | 1.271 / 5.43 (u2000) | 1.291 / 5.22 (u2800) |
| ms_r1_pilot | MS_base2400 | 50 | 1.104 / 12.93 (u400) | 1.210 / 6.00 (u1200) | 1.243 / 8.19 (u2000) | 1.252 / 4.97 (u2400) |
| ms_r1_pilot | MS_base2400 | 60 | 1.122 / 11.85 (u400) | 1.213 / 6.53 (u1200) | 1.252 / 6.60 (u2000) | 1.263 / 4.42 (u2400) |
| ms_r1_pilot | MS_rule | 50 | 1.106 / 9.62 (u400) | 1.187 / 5.95 (u1200) | 1.229 / 5.49 (u2000) | 1.238 / 4.94 (u2400) |
| ms_r1_pilot | MS_rule | 60 | 1.227 / 13.97 (u400) | 1.302 / 7.48 (u1200) | 1.344 / 6.26 (u2000) | 1.357 / 6.51 (u2400) |
| ms_r1_pilot | MS_s25a0 | 50 | 1.109 / 7.07 (u400) | 1.243 / 3.21 (u1200) | 1.303 / 4.09 (u2000) | 1.318 / 3.97 (u2400) |
| ms_r1_pilot | MS_s25a0 | 60 | 1.055 / 8.05 (u400) | 1.141 / 5.83 (u1200) | 1.195 / 5.31 (u2000) | 1.206 / 4.82 (u2400) |
| ms_r1_pilot | MS_s25a5 | 50 | 1.184 / 9.56 (u400) | 1.269 / 5.17 (u1200) | 1.325 / 4.04 (u2000) | 1.338 / 4.16 (u2400) |
| ms_r1_pilot | MS_s25a5 | 60 | 1.174 / 7.98 (u400) | 1.276 / 4.36 (u1200) | 1.320 / 5.35 (u2000) | 1.324 / 4.70 (u2400) |
| ms_r1_pilot | MS_s35a0 | 50 | 1.118 / 7.51 (u400) | 1.256 / 5.89 (u1200) | 1.307 / 4.04 (u2000) | 1.328 / 3.62 (u2400) |
| ms_r1_pilot | MS_s35a0 | 60 | 1.131 / 8.98 (u400) | 1.261 / 6.00 (u1200) | 1.321 / 6.66 (u2000) | 1.335 / 4.77 (u2400) |
| ms_r1_pilot | MS_s35a5 | 50 | 1.046 / 7.67 (u400) | 1.151 / 5.88 (u1200) | 1.200 / 4.00 (u2000) | 1.210 / 3.05 (u2400) |
| ms_r1_pilot | MS_s35a5 | 60 | 1.106 / 8.29 (u400) | 1.225 / 5.35 (u1200) | 1.270 / 6.11 (u2000) | 1.285 / 4.90 (u2400) |
| ms_r1_base | MS_base | 50 | 1.104 / 12.93 (u400) | 1.210 / 6.00 (u1200) | 1.220 / 6.25 (u1600) | 1.220 / 6.25 (u1600) |
| ms_r1_base | MS_base | 60 | 1.122 / 11.85 (u400) | 1.213 / 6.53 (u1200) | 1.220 / 5.97 (u1600) | 1.220 / 5.97 (u1600) |

Ranges over the 26 source / arm / q rows (block `diag_ranges`):

| quantity | min | max | rows |
|---|---|---|---|
| median max abs w on d (d / B), u400 | 1.046 | 1.227 | 26 |
| median w_eff (units of d), u400 | 7.07 | 13.97 | 26 |
| median max abs w on d (d / B), final | 1.206 | 1.357 | 26 |
| median w_eff (units of d), final | 3.05 | 6.51 | 26 |
| B / median max abs w, final, q = 50 (B = 200; units of d) | 149 | 165 | 13 |
| B / median max abs w, final, q = 60 (B = 220; units of d) | 162 | 182 | 13 |
| growth of the median max abs w from u400 to final (fraction) | 0.088 | 0.188 | 26 |
| across the ten final exports: Spearman(max abs w, w_eff) | -0.333 | 0.842 | 26 |
| within-run median Spearman(max abs w, w_eff) | -0.710 | -0.418 | 26 |
| pooled Spearman(max abs w, w_eff) | -0.356 | -0.162 | 26 |
| share of final exports with max abs w <= the screen reference | 0.10 | 0.70 | 26 |

### 2.1 The distribution of |w| over the 64 first-layer units (blocks `diag_dist`, `diag_dist_facts`; medians over the seeds of the quantiles and of the counts)

| source | arm | q | u400 | final |
|---|---|---|---|---|
| ms_r2_pilot | NL_bb_s1 | 50 | |w| q25/q50/q90/max 0.11/0.31/0.72/1.10; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.15/0.38/0.80/1.26; units > 1/2/3/5: 3/0/0/0 |
| ms_r2_pilot | NL_bb_s1 | 60 | |w| q25/q50/q90/max 0.09/0.26/0.76/1.12; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.09/0.31/0.86/1.28; units > 1/2/3/5: 4/0/0/0 |
| ms_r2_pilot | NL_bb_s4 | 50 | |w| q25/q50/q90/max 0.11/0.31/0.72/1.10; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.15/0.40/0.80/1.25; units > 1/2/3/5: 2/0/0/0 |
| ms_r2_pilot | NL_bb_s4 | 60 | |w| q25/q50/q90/max 0.09/0.26/0.76/1.12; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.09/0.31/0.85/1.28; units > 1/2/3/5: 4/0/0/0 |
| ms_r2_pilot | NL_bb_s16 | 50 | |w| q25/q50/q90/max 0.11/0.31/0.72/1.10; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.15/0.39/0.80/1.26; units > 1/2/3/5: 2/0/0/0 |
| ms_r2_pilot | NL_bb_s16 | 60 | |w| q25/q50/q90/max 0.09/0.26/0.76/1.12; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.09/0.31/0.85/1.27; units > 1/2/3/5: 4/0/0/0 |
| ms_r2_pilot | NL_st_s1 | 50 | |w| q25/q50/q90/max 0.13/0.32/0.74/1.05; units > 1/2/3/5: 1/0/0/0 | |w| q25/q50/q90/max 0.17/0.40/0.84/1.24; units > 1/2/3/5: 4/0/0/0 |
| ms_r2_pilot | NL_st_s1 | 60 | |w| q25/q50/q90/max 0.10/0.30/0.70/1.11; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.12/0.35/0.79/1.30; units > 1/2/3/5: 4/0/0/0 |
| ms_r2_pilot | NL_st_s4 | 50 | |w| q25/q50/q90/max 0.13/0.32/0.74/1.05; units > 1/2/3/5: 1/0/0/0 | |w| q25/q50/q90/max 0.17/0.41/0.83/1.23; units > 1/2/3/5: 4/0/0/0 |
| ms_r2_pilot | NL_st_s4 | 60 | |w| q25/q50/q90/max 0.10/0.30/0.70/1.11; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.12/0.35/0.80/1.30; units > 1/2/3/5: 4/0/0/0 |
| ms_r2_pilot | NL_st_s16 | 50 | |w| q25/q50/q90/max 0.13/0.32/0.74/1.05; units > 1/2/3/5: 1/0/0/0 | |w| q25/q50/q90/max 0.16/0.41/0.84/1.23; units > 1/2/3/5: 4/0/0/0 |
| ms_r2_pilot | NL_st_s16 | 60 | |w| q25/q50/q90/max 0.10/0.30/0.70/1.11; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.11/0.35/0.79/1.29; units > 1/2/3/5: 3/0/0/0 |
| ms_r1_pilot | MS_base2400 | 50 | |w| q25/q50/q90/max 0.11/0.31/0.72/1.10; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.15/0.39/0.79/1.25; units > 1/2/3/5: 2/0/0/0 |
| ms_r1_pilot | MS_base2400 | 60 | |w| q25/q50/q90/max 0.09/0.26/0.76/1.12; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.09/0.31/0.85/1.26; units > 1/2/3/5: 4/0/0/0 |
| ms_r1_pilot | MS_rule | 50 | |w| q25/q50/q90/max 0.11/0.31/0.73/1.11; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.17/0.37/0.84/1.24; units > 1/2/3/5: 4/0/0/0 |
| ms_r1_pilot | MS_rule | 60 | |w| q25/q50/q90/max 0.07/0.27/0.77/1.23; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.07/0.34/0.87/1.36; units > 1/2/3/5: 4/0/0/0 |
| ms_r1_pilot | MS_s25a0 | 50 | |w| q25/q50/q90/max 0.13/0.32/0.74/1.11; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.14/0.41/0.86/1.32; units > 1/2/3/5: 4/0/0/0 |
| ms_r1_pilot | MS_s25a0 | 60 | |w| q25/q50/q90/max 0.10/0.28/0.73/1.06; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.10/0.33/0.81/1.21; units > 1/2/3/5: 4/0/0/0 |
| ms_r1_pilot | MS_s25a5 | 50 | |w| q25/q50/q90/max 0.11/0.29/0.76/1.18; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.12/0.35/0.85/1.34; units > 1/2/3/5: 4/0/0/0 |
| ms_r1_pilot | MS_s25a5 | 60 | |w| q25/q50/q90/max 0.10/0.31/0.72/1.17; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.11/0.33/0.82/1.32; units > 1/2/3/5: 4/0/0/0 |
| ms_r1_pilot | MS_s35a0 | 50 | |w| q25/q50/q90/max 0.11/0.32/0.74/1.12; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.15/0.42/0.85/1.33; units > 1/2/3/5: 3/0/0/0 |
| ms_r1_pilot | MS_s35a0 | 60 | |w| q25/q50/q90/max 0.11/0.33/0.68/1.13; units > 1/2/3/5: 1/0/0/0 | |w| q25/q50/q90/max 0.12/0.38/0.78/1.33; units > 1/2/3/5: 3/0/0/0 |
| ms_r1_pilot | MS_s35a5 | 50 | |w| q25/q50/q90/max 0.13/0.32/0.74/1.05; units > 1/2/3/5: 1/0/0/0 | |w| q25/q50/q90/max 0.14/0.39/0.82/1.21; units > 1/2/3/5: 4/0/0/0 |
| ms_r1_pilot | MS_s35a5 | 60 | |w| q25/q50/q90/max 0.10/0.30/0.70/1.11; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.13/0.35/0.78/1.28; units > 1/2/3/5: 3/0/0/0 |
| ms_r1_base | MS_base | 50 | |w| q25/q50/q90/max 0.11/0.31/0.72/1.10; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.14/0.36/0.78/1.22; units > 1/2/3/5: 2/0/0/0 |
| ms_r1_base | MS_base | 60 | |w| q25/q50/q90/max 0.09/0.26/0.76/1.12; units > 1/2/3/5: 2/0/0/0 | |w| q25/q50/q90/max 0.09/0.31/0.83/1.22; units > 1/2/3/5: 3/0/0/0 |

At the final terminal-stage export of every run:

- final terminal-stage exports: 260 runs
- units with |w| > 1: min 0, median 4, max 9 (of 64 units)
- units with |w| > 2: min 0, median 0, max 0 (of 64 units)
- units with |w| > 3: min 0, median 0, max 0 (of 64 units)
- units with |w| > 5: min 0, median 0, max 0 (of 64 units)
- median |w| over the units: min 0.169, median over runs 0.364, max 0.524
- 90 % quantile of |w|: min 0.666, median over runs 0.829, max 1.069
- max |w|: min 0.811, median over runs 1.280, max 1.693

Reading (descriptive). The largest first-layer weight on d / B of the RL actors is about 1.0-1.2 at update 400 and 1.2-1.4 at the freeze (medians over seeds, all 26 rows): its median grows by 9-19 % between update 400 and the freeze (row by row). The rounding width `w_eff` falls over the same updates, from 7-14 units of d at update 400 to 3-6.5 at the freeze. At the freeze the sharpest tanh unit of the first layer bends over `B / max |w|` = 149-165 units of d (q = 50) and 162-182 units (q = 60), against a rounding width of 3-6.5 units of d. Over the rows the median `w_eff` at the freeze varies by a factor of about two (3.05-6.51) and the median max |w| by about 12 % (1.206-1.357): the MS-R1 sampler arms and the MS-R2 noise-landing arms differ more in the width than in the weight.

## 3. How the weights and the width move together (block `diag_relation`; Spearman correlation of max |w| with `w_eff` over the terminal-stage exports)

| source | arm | q | runs | pooled rho | within-run median rho | final export, across seeds: rho |
|---|---|---|---|---|---|---|
| ms_r2_pilot | NL_bb_s1 | 50 | 10 | -0.199 | -0.535 | 0.479 |
| ms_r2_pilot | NL_bb_s1 | 60 | 10 | -0.194 | -0.504 | 0.479 |
| ms_r2_pilot | NL_bb_s4 | 50 | 10 | -0.180 | -0.612 | 0.236 |
| ms_r2_pilot | NL_bb_s4 | 60 | 10 | -0.172 | -0.560 | 0.564 |
| ms_r2_pilot | NL_bb_s16 | 50 | 10 | -0.203 | -0.564 | 0.236 |
| ms_r2_pilot | NL_bb_s16 | 60 | 10 | -0.186 | -0.503 | 0.042 |
| ms_r2_pilot | NL_st_s1 | 50 | 10 | -0.272 | -0.559 | -0.079 |
| ms_r2_pilot | NL_st_s1 | 60 | 10 | -0.265 | -0.522 | -0.333 |
| ms_r2_pilot | NL_st_s4 | 50 | 10 | -0.238 | -0.635 | -0.079 |
| ms_r2_pilot | NL_st_s4 | 60 | 10 | -0.299 | -0.529 | -0.321 |
| ms_r2_pilot | NL_st_s16 | 50 | 10 | -0.310 | -0.633 | -0.321 |
| ms_r2_pilot | NL_st_s16 | 60 | 10 | -0.267 | -0.418 | -0.176 |
| ms_r1_pilot | MS_base2400 | 50 | 10 | -0.221 | -0.583 | 0.079 |
| ms_r1_pilot | MS_base2400 | 60 | 10 | -0.162 | -0.558 | 0.503 |
| ms_r1_pilot | MS_rule | 50 | 10 | -0.273 | -0.624 | -0.164 |
| ms_r1_pilot | MS_rule | 60 | 10 | -0.244 | -0.596 | 0.127 |
| ms_r1_pilot | MS_s25a0 | 50 | 10 | -0.303 | -0.582 | 0.224 |
| ms_r1_pilot | MS_s25a0 | 60 | 10 | -0.249 | -0.432 | -0.042 |
| ms_r1_pilot | MS_s25a5 | 50 | 10 | -0.179 | -0.603 | 0.842 |
| ms_r1_pilot | MS_s25a5 | 60 | 10 | -0.210 | -0.559 | 0.406 |
| ms_r1_pilot | MS_s35a0 | 50 | 10 | -0.191 | -0.612 | -0.067 |
| ms_r1_pilot | MS_s35a0 | 60 | 10 | -0.298 | -0.539 | 0.648 |
| ms_r1_pilot | MS_s35a5 | 50 | 10 | -0.290 | -0.595 | -0.042 |
| ms_r1_pilot | MS_s35a5 | 60 | 10 | -0.273 | -0.516 | -0.042 |
| ms_r1_base | MS_base | 50 | 10 | -0.356 | -0.710 | 0.309 |
| ms_r1_base | MS_base | 60 | 10 | -0.233 | -0.701 | 0.624 |

Within a run the two are negatively rank-correlated (median over seeds -0.42 to -0.71 over the rows): where the weight is larger the width is smaller, which in a single run mostly reflects that both drift with the update count. The pooled correlations over seeds are weaker (-0.16 to -0.36). The correlation across the ten final exports of an arm (last column) has no consistent sign (from -0.33 to +0.84 over the rows; ten points each). The table does not separate training time from the weight itself.

## 4. Do the RL actors sit in the small-weight regime of the screen? (block `diag_regime`)

The reference is the median over the ten seeds of max |w| of the supervised `t1` actors at 56,000 steps, bin-balanced starts (`01_supervised_screen.md` table 2.2): 1.299 at q = 50 and 1.163 at q = 60. The table gives the share of the exports of each arm at or below that value (all exports, terminal-stage exports, and the ten final exports).

| source | arm | q | reference max abs w | all exports | terminal-stage exports | final exports (of 10) |
|---|---|---|---|---|---|---|
| ms_r2_pilot | NL_bb_s1 | 50 | 1.299 | 0.698 | 0.719 | 0.60 |
| ms_r2_pilot | NL_bb_s1 | 60 | 1.163 | 0.307 | 0.352 | 0.10 |
| ms_r2_pilot | NL_bb_s4 | 50 | 1.299 | 0.698 | 0.719 | 0.60 |
| ms_r2_pilot | NL_bb_s4 | 60 | 1.163 | 0.303 | 0.346 | 0.10 |
| ms_r2_pilot | NL_bb_s16 | 50 | 1.299 | 0.698 | 0.719 | 0.60 |
| ms_r2_pilot | NL_bb_s16 | 60 | 1.163 | 0.304 | 0.348 | 0.10 |
| ms_r2_pilot | NL_st_s1 | 50 | 1.299 | 0.593 | 0.613 | 0.50 |
| ms_r2_pilot | NL_st_s1 | 60 | 1.163 | 0.399 | 0.420 | 0.30 |
| ms_r2_pilot | NL_st_s4 | 50 | 1.299 | 0.592 | 0.612 | 0.50 |
| ms_r2_pilot | NL_st_s4 | 60 | 1.163 | 0.399 | 0.420 | 0.30 |
| ms_r2_pilot | NL_st_s16 | 50 | 1.299 | 0.598 | 0.619 | 0.50 |
| ms_r2_pilot | NL_st_s16 | 60 | 1.163 | 0.399 | 0.420 | 0.30 |
| ms_r1_pilot | MS_base2400 | 50 | 1.299 | 0.711 | 0.739 | 0.60 |
| ms_r1_pilot | MS_base2400 | 60 | 1.163 | 0.328 | 0.384 | 0.10 |
| ms_r1_pilot | MS_rule | 50 | 1.299 | 0.789 | 0.830 | 0.70 |
| ms_r1_pilot | MS_rule | 60 | 1.163 | 0.306 | 0.339 | 0.20 |
| ms_r1_pilot | MS_s25a0 | 50 | 1.299 | 0.626 | 0.650 | 0.50 |
| ms_r1_pilot | MS_s25a0 | 60 | 1.163 | 0.518 | 0.552 | 0.40 |
| ms_r1_pilot | MS_s25a5 | 50 | 1.299 | 0.552 | 0.599 | 0.40 |
| ms_r1_pilot | MS_s25a5 | 60 | 1.163 | 0.367 | 0.406 | 0.20 |
| ms_r1_pilot | MS_s35a0 | 50 | 1.299 | 0.595 | 0.648 | 0.40 |
| ms_r1_pilot | MS_s35a0 | 60 | 1.163 | 0.418 | 0.443 | 0.30 |
| ms_r1_pilot | MS_s35a5 | 50 | 1.299 | 0.619 | 0.640 | 0.60 |
| ms_r1_pilot | MS_s35a5 | 60 | 1.163 | 0.404 | 0.432 | 0.30 |
| ms_r1_base | MS_base | 50 | 1.299 | 0.769 | 0.795 | 0.70 |
| ms_r1_base | MS_base | 60 | 1.163 | 0.457 | 0.516 | 0.30 |

The shares of final exports at or below the reference lie between 0.10 and 0.70 over the 26 rows; at q = 50 they are 0.4-0.7 and at q = 60 0.1-0.4. The RL actors' final max |w| (medians 1.21-1.36) lies in the neighbourhood of the supervised `t1` actors' median at the RL budget (1.30 and 1.16): the RL actors are in the same weight range as the supervised actors that leave a rounded tip (median tip deficit 1.64 and 6.32 effort units, `01_supervised_screen.md` section 2.1). For a ReLU unit the weight is not the relevant quantity (the unit has a kink at any weight); the supervised `relu` actors have max |w| of 0.79-0.87.

## 5. The preamble figure

The six-arm mean of `w_eff` at the freeze of the MS-R2 runs is 4.855 at q = 50 and 5.333 at q = 60 (`summary.txt`, last table), next to the preamble's 4.85 and 5.33 ("gap / tent slope averages 4.85 at q = 50 and 5.33 at q = 60"): the tool, which reloads the exports, and the analysis table `results/ms_r2/analysis/per_run.csv` (`reports/ms/r3/01_supervised_screen.md` section 4, item 2) agree.
