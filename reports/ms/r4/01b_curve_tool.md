# MS-R4 validation of the offline curve tool on MS-R3's `t1_st_s16` runs

Date 2026-10-13. Tool: `tools/ms/r4_curves.py` (code commit of the tool: see `02_preregistration.md`). Prompt §2.3 (b). Offline, no training; the closed form enters only the evaluation columns (`g2_at_0`, the peak and RMSE errors), never a run.

## 1. What the tool does

For every stage-2 weight export `weights/u<N>.npz` (N = 25, 50, ..., the terminal-stage budget) it builds the composite candidate with the export at every stage, exactly as the live actor is used during the stage-2 phase (`tools/ms/replay_dev_rule.build_actor`: a `BetaActor` carrying the exported float32 weights, `conc_scale` and `actor_variant` read from the export, an unknown variant refused; `candidate_fns` with the same network for stages 1 and 2) and runs `utils.v2_metrics.evaluate` on the FINAL tier (the curve) and on the DEVELOPMENT tier (check C-OFF). It is pure: it moves no RNG stream (the process-global torch RNG, which `nn.Linear`'s default initialisation would draw from, is forked around the actor construction), writes nothing under a run directory, and runs one single-threaded process per run (at most 40 workers).

## 2. C-OFF on real exports

```
python tools/ms/r4_curves.py --pilot-root <MS-R3 worktree>/results/ms_r3/pilot --out results/ms_r4/curves_validation \
    --arms t1_st_s16 --seeds 10501 ... 10510 --workers 20
```

(`<MS-R3 worktree>` = `/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857`, HEAD `be4fd202`; stdout in `results/ms_r4/curves_validation.stdout`, per-run record `results/ms_r4/curves_validation/c_off.csv`, all points `curves_all.csv`, `curves_info.json`.) All tables below are printed by `reports/ms/r4/report_scripts/curve_validation_tables.py` from those CSVs.

**Result: PASS.** 20 of 20 runs (2 q x 10 seeds), 2,240 stage-2 exports (112 per run), **24,640 comparisons** (11 online-check columns at each of the 2,240 points: `Delta`, `eta_T_over_dw`, `stage2_peak_rel_err_signed`, `stage2_peak_rel_err_abs`, `stage2_rmse_pos_over_g2_0`, `stage2_tail_mean_over_g2_0`, `stage2_tail_max_over_g2_0`, `e1_at_0`, `e2_at_0`, `stage1_rel_err_signed`, `Gmax_full_over_dw`), **0 mismatches, maximum absolute difference 0.0** (the development-tier value computed offline from the export equals the value the run wrote at the same update bit for bit). At the last export (u2800) the final-tier eta_2/DW, the development-tier eta_2/DW and the final-tier |peak error| equal the freeze record of `gates.json` in 20 of 20 runs. Cost of the offline evaluation, reported apart from training: 47.2 s (final tier) and 22.6 s (development tier) summed over the 2,240 exports, 7.1 s wall with 20 workers.

## 3. Descriptive: the final-tier curves of these 20 runs on the E2 grid

The 64 fixed checkpoints u25 ... u1600 (the grid of the E2 area; these runs go on to u2800 under the noise landing, which the area ignores). `A` = the trapezoidal area of the final-tier eta_2/DW over u25 ... u1600 divided by 1575, i.e. the mean height of the curve.

| q | runs | mean A (final) | mean A (dev) | share of the final-tier area over u <= 200: mean [min, max] | dev share: mean | eta final at u1600: median | dev / final per run: median ratio (min over runs - max over runs) | first export with eta <= 0.005: median |
|---|---|---|---|---|---|---|---|---|
| 50 | 10 | 0.00652 | 0.00648 | 0.619 [0.565, 0.682] | 0.622 | 2.14e-03 | 0.999 (0.741 - 1.000) | 300 |
| 60 | 10 | 0.00466 | 0.00462 | 0.745 [0.707, 0.776] | 0.751 | 1.05e-03 | 0.995 (0.742 - 1.000) | 150 |
| both | 20 | 0.00559 | 0.00555 | 0.682 [0.565, 0.776] | 0.687 | 1.29e-03 | | 225 |

Seed-median final-tier / development-tier eta_2/DW at selected updates:

- q = 50: u25 1.12e-01 / 1.12e-01; u50 9.18e-02 / 9.18e-02; u100 1.64e-02 / 1.64e-02; u200 9.63e-03 / 9.52e-03; u400 2.16e-03 / 2.14e-03; u800 1.87e-03 / 1.87e-03; u1200 1.91e-03 / 1.90e-03; u1600 2.14e-03 / 2.04e-03; u2000 2.05e-03 / 2.02e-03; u2800 6.97e-04 / 6.36e-04.
- q = 60: u25 1.07e-01 / 1.07e-01; u50 6.42e-02 / 6.41e-02; u100 2.17e-02 / 2.17e-02; u200 3.73e-03 / 3.73e-03; u400 2.22e-03 / 2.20e-03; u800 1.40e-03 / 1.40e-03; u1200 7.71e-04 / 7.44e-04; u1600 1.05e-03 / 8.83e-04; u2000 6.52e-04 / 6.13e-04; u2800 4.05e-04 / 3.95e-04.

Reading (descriptive, no test): the first 200 updates hold 57-78 % of the final-tier area of these runs (the prompt's PI-side figure for the development tier of MS-R1 was 49-75 %; adjustment 5), so a reduction of the area by 20 % has to come mostly from the early transient, which the sampling feedback can influence only after the first check (update 25) and only through the starts. The development tier tracks the final tier closely (median ratio 0.995-0.999 per run); it never exceeds the final tier (largest ratio 1.0000000000004) and falls below 0.9 of it at 158 of the 1,280 points of the E2 grid, all of them at final-tier values of at most 1.8e-3 of DW (smallest ratio 0.741; 5th percentile 0.826; 1st percentile 0.769).
