# R2b P2: read-only diagnostic of q=50 seed 30510 (v2.0 confirmation)

Produced by `tools/v2/diag_seed30510.py` (commit `71eca87`, applying `01_preregistration.md` section 5 literally) from files that were only read. No run was repeated, no threshold or protocol value was changed, no training was started. Every number below is stored with its source in `results/v2_refine_r2b/diag_30510/numbers.json`; tables carry a source line (CSVs under `results/v2_refine_r2b/diag_30510/tables/`, figures under `figures/`). Association is not cause: where this report compares seeds it says so. The report text is the tool's output; the only hand edits are this header, the pointers to the pre-registration in section 7 and the file list in section 10. Seed 30510 was not re-run (D1).

## 0. Summary

- Failure (reproduced from the files): eta_2/DW = 0.00580 against the G-A threshold 0.005; stage-2 peak error -0.1458; eta_2 maximum on-path (0.00580) against off-path (0.000819); smoothed-game share 0.279. Match with the four numbers quoted in the request (to the digits quoted): yes, all four.
- Peak error of seed 30510: -0.1458. The other 19 q=50 seeds: median -0.0611, 10th percentile -0.0810, 90th percentile -0.0283 (range -0.0895 .. -0.0165). Rank of seed 30510 from the lowest: 1 of 20 q=50 runs, 1 of 40 runs; the lowest of the other 39 is q=60 seed 30520 at -0.1237 (its G-A verdict: pass).
- Pre-registered departure update: u_leave = 900 (the first export from which S_target(u) < p10_pack(S, u) holds at that export and at every later one; resolution 25 updates). The LR-decay window starts at update 1201; u_leave >= 1225: False. Descriptive facts on the same series: last in-band export 875; below the band at 56 of 64 exports in total.
- Pre-registered rules H1-H4 applied literally (section 7; verdict words are produced from computed booleans, `numbers.json` keys `rules.target.*`): H1 not supported, H2 not supported, H3 not supported, H4 not supported. Classification label: **none of H1-H4 (early or unstable departure)**. Inputs: max over all exports of S = -0.0582 at u=825 against p10_pack(S, 1600) = -0.0810; out_S(1600) = True; u_leave = 900; consecutive exports not out of the band ending at u_leave - 25: 1 (the H3 plateau condition needs 16).
- Early signature (descriptive): the concentration alpha+beta at d=0 leaves the other seeds' band at update 125 and never returns; the actor gradient norm from update 150, sigma_2(0) from 175. Whether this is a cause or a co-symptom of the low peak cannot be decided from these files (section 8).

### Six key numbers

| # | quantity | value |
|---|---|---|
| 1 | peak error of seed 30510 (signed, relative) | -0.1458 |
| 2 | median of the other 19 q=50 seeds | -0.0611 |
| 3 | 10th percentile of the other 19 | -0.0810 |
| 4 | 90th percentile of the other 19 | -0.0283 |
| 5 | u_leave: update from which 30510 is out of the pack's band (S < p10_pack(S, u)) at every later export (pre-registered definition) | 900 |
| 6 | pre-registered rules H1 / H2 / H3 / H4 and label | not supported / not supported / not supported / not supported; none of H1-H4 (early or unstable departure) |

## Source legend

All run files are under `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0/q{50,60}/seed{30501..30520}/`.

| tag | file(s) | content used |
|---|---|---|
| [G] | `gates.json` | `reported.end_of_A` (final-tier and development-tier metrics, smoothed game), `metric_values`, `G-A` |
| [W] | `weights/u00025.npz` .. `u01600.npz` | actor weights every 25 updates (64 exports in Phase A); the stage-2 mean, alpha, beta are recomputed with the repo's float32 forward pass |
| [A] | `gateA_final.npz` | end-of-A recovery grid (e2_hat, e2*) and final-tier verifier arrays (Delta_2, on-path mask, sigma_2, alpha, beta) |
| [C] | `v2_checkpoints_A.csv` | development-tier verifier calls in Phase A (16 per run; runs with another count: [[60, 30505]]) |
| [U] | `v2_updates.csv` | per-update KL, clip fraction, advantage SD, losses, D1 clamp counts (Phase-A rows) |
| [H] | `train_history.json` | per-update actor grad norms, entropy, buffer concentration; `verifier_calls[A].visitation_cumulative_phase.stage2_direct_es` |
| [S] | `v2_run_summary.json` | `would_have_fired` |
| [R] | `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0_analysis/reported_metrics.csv`, `per_run.csv` | the confirmation analysis tables (cross-check of the owner's numbers) |
| [F] | `results/v2_pilots/pilot4/analysis/repr_floor/fits.csv`, `three_way_peak_gap.csv` (v2-t2-refine worktree) | supervised-fit floor of the actor (5 inits, pilot 4 section 1d) and pilot-4 reference gaps |
| [V] | this tool + `utils/dp_br_verifier.py` (read-only import) | development-tier eta_2 recomputed on every weight export |

## 1. Methods and reproduction of the owner's numbers

**Methods.** For every one of the 40 runs (q in {50, 60} x seeds 30501-30520) the 64 Phase-A weight exports were read and the stage-2 actor re-evaluated in float32 exactly as `agents/ppo_curriculum.mean_effort_numpy` does (input [1, d/B], B = 100 + 2q; mean = 100 alpha/(alpha+beta)) on the recovery grid (step 0.5, 801 points for q=50, 881 for q=60) as in `utils/v2_metrics.recovery_metrics`. e2_hat(0) is the Beta mean at d=0. The location-free peak is the maximum of the mean over that grid, as `run/run_v2_T2_locked.stage2_extra` defines it (relative error (max - e2*(0))/e2*(0)). The closed form e2*(d) = DW f_xi(d)/(2k) (DW=4, k=0.000286, e2*(0)=70 at q=50; e2*(0)=58.33 at q=60) is used for evaluation only. The smoothed-game prediction is the repo's (`smoothed_share`: 400 Beta quantile nodes at d=0). End-of-A arrays come from the runs' own `gateA_final.npz`. The development-tier eta_2 was recomputed on every export by importing the repo verifier read-only (bytecode writing disabled); nothing was written inside the repository or any results directory. 'Others' means the 19 other q=50 seeds; the band is the 10th-90th percentile (numpy linear) of their values at the same update. 'Below the band' means below the 10th percentile (the errors are negative, so below = a larger error). Ratios and percentages written in the text are computed from the listed registry values. Percentiles of 19 values are coarse: any single seed is below the band at roughly 10-15% of exports by construction, and exports of a run are autocorrelated, so run-level comparisons below use the other seeds' own leave-one-out frequencies.

**Reproduction.** Sources: [G], [W], [A], [R], [V].

(The column 'owner's message' is quoted from the request, not read from a file.)

| quantity | owner's message | gates.json [G] | recomputed here | abs diff |
|---|---|---|---|---|
| peak error signed (final tier) | -0.1458 | -0.145813 | -0.145813 | 0.0e+00 |
| eta_2 on-path max, final tier [G] / dev tier [G] / dev tier recomputed [V] | 0.0058 | 0.0058033 / 0.0058033 | 0.0058033 |  |
| eta_2 off-path max, final tier [G] / dev tier [G] / dev tier recomputed [V] | 0.00082 | 0.0008190 / 0.0008184 | 0.0008184 |  |
| smoothed-game share of the d=0 gap | 0.279 | 0.279363 | 0.279363 | 0.0e+00 |
| e2_hat(0) (effort units) | n/a | 59.79309 | 59.79309 | 0.0e+00 |
| location-free peak error | n/a | -0.145544 | -0.145545 | 1.9e-07 |
| sigma_2(0) (effort units) | n/a | 3.61088 | 3.61088 | 0.0e+00 |

The same comparison over all 40 runs: largest |difference| in the signed peak error 2.9e-07, in the smoothed share 2.3e-05, in sigma_2(0) 2.2e-07; the recovery-grid mean from the u1600 weights differs from the run's own `recovery_e2` by at most 2.9e-05 effort units (float32 rounding of numpy against torch). The recomputed development-tier eta_2 matches `v2_checkpoints_A.csv` at every Phase-A call of every run to within 4.3e-08.
Match with the four numbers quoted in the request (file value rounded to the digits quoted): peak error True, on-path maximum True, off-path maximum True, smoothed share True; all four: True. Final-tier and development-tier on-path maxima identical in the files: True; off-path maxima final 0.0008190, development 0.0008184; the development-tier on-path maximum recomputed here equals the stored one to 5.3e-09.

## 2. Peak trajectory (figures/fig1_peak_trajectory_q50.png, figures/fig2_peak_trajectory_q60_reference.png, figures/fig3b_profile_evolution_near_peak.png)

### 2.1 Definitions used in this section

For every export u in {25, ..., 1600}: z(u) = relative error of e2_hat(0) = S(u) (and, separately, of the location-free peak = L(u)) of seed 30510; p10(u), median(u), p90(u) over the 19 other q=50 seeds. Below the band: z(u) < p10(u) (the pre-registered out_S(u) for S). Reported: (a) first export below the band; (b) the first export of a run of at least 8 consecutive below-band exports (200 updates; not pre-registered); (c) the last in-band export; (d) u_leave, the pre-registered departure update: the earliest export u0 such that z(u) < p10(u) at every export u0, ..., 1600. In-band exports are listed so that the reader can apply another rule. The pre-registered rules H1-H4 are applied in section 7.

### 2.2 Snapshot of e2_hat(0) (relative error)  [W]

| update | seed 30510 | others median | others p10 .. p90 | rank of 30510 among 20 (1 = lowest) |
|---|---|---|---|---|
| 100 | -0.7566 | -0.4962 | -0.6789 .. -0.3569 | 1 |
| 250 | -0.1396 | -0.1470 | -0.1873 .. -0.1245 | 12 |
| 325 | -0.1542 | -0.1224 | -0.1563 .. -0.1031 | 3 |
| 400 | -0.1603 | -0.1069 | -0.1320 .. -0.0720 | 1 |
| 600 | -0.1783 | -0.0719 | -0.0959 .. -0.0551 | 1 |
| 800 | -0.1337 | -0.0670 | -0.1071 .. -0.0306 | 1 |
| 825 | -0.0582 | -0.0586 | -0.0943 .. -0.0495 | 11 |
| 850 | -0.1217 | -0.0753 | -0.0976 .. -0.0447 | 1 |
| 900 | -0.1300 | -0.0594 | -0.0916 .. -0.0412 | 1 |
| 1200 | -0.1402 | -0.0569 | -0.0898 .. -0.0364 | 2 |
| 1600 | -0.1458 | -0.0611 | -0.0810 .. -0.0283 | 1 |

### 2.3 Band-exit statistics  [W]

| series | exports below band | first below | start of first run of >= 8 | last in band | sustained departure | in band at u >= 400 | below band at u >= 900 | rank 1 (lowest of 20) at u >= 900 |
|---|---|---|---|---|---|---|---|---|
| e2_hat(0) | 56 / 64 | 50 | 550 | 875 | 900 | 4 / 49 | 29 / 29 | 24 / 29 |
| location-free peak | 55 / 64 | 100 | 550 | 875 | 900 | 4 / 49 | 29 / 29 | 26 / 29 |

In-band exports of the e2_hat(0) series: [25, 75, 250, 325, 500, 525, 825, 875]. Location-free series: [25, 50, 75, 250, 325, 500, 525, 825, 875].

### 2.4 Trajectory facts (descriptive; any wording that compares values is the author's reading, not pre-registered)

- Early values. At update 100: -0.7566 against a median of -0.4962 (p10 -0.6789, rank 1 of 20 from the lowest). At update 250: -0.1396 (rank 12; median -0.1470); at 325: -0.1542 (rank 3).
- Later values. Pack median -0.1069 at 400, -0.0719 at 600, -0.0609 as the median over exports from 900 on; seed 30510: -0.1603 at 400, -0.1783 at 600, -0.1300 at 900, -0.1402 at 1200, -0.1458 at 1600.
- First export at or above -0.10: update 825 for 30510, against a median of 400 for the other seeds (p10 345, p90 485, latest 550; all 19 others reach it). That export (u=825) has S = -0.0582 (pack median -0.0586, rank 11), between -0.1337 (u=800) and -0.1217 (u=850). At that export the dev-tier eta_2/DW is 0.01374 (the others' median at u=825: 0.00224); the largest eta_2/DW of 30510 from u=400 on is at u=825 (0.01374).
- Late level (updates 900-1600): mean -0.1362 for 30510 against a median of -0.0620 for the other seeds' means (range -0.0830 .. -0.0485): 0 of the 19 others have a late mean at or below it. Its best late export (-0.0944 at u=1325) is at the pooled 10th percentile of the others' late exports (-0.0949); only 1 of its 29 late exports reaches it.
- Leave-one-out comparison: the fraction of late exports below a seed's own leave-one-out p10 is 1.00 for 30510; for the other seeds the median is 0.07 and the maximum 0.31; 0 of them reach 0.5.
- Volatility: SD of the late series 0.0222 (others' median 0.0222; target inside the others' range); SD of consecutive-export changes 0.0322 (others' median 0.0239, maximum 0.0355; not above the others' maximum: True); largest single step 0.0619 (others' median of the largest step 0.0539, maximum 0.0848; not above the others' maximum: True).
- LR-decay window: mean(1225-1600) minus mean(900-1200) is +0.0142 for 30510 (rank 13 of 20 from the lowest; inside the others' range); the other seeds: median +0.0062, p10 -0.0114, p90 +0.0234. OLS slope of the late series: +0.0027 per 100 updates (others' median +0.0015).
- q=60 pack (figures/fig2_peak_trajectory_q60_reference.png; 20 runs): median -0.0678, p10 -0.0890, p90 -0.0450 at u=1600; 30510 (-0.1458) is below that p10: True.

Per-seed trajectory statistics (source: `tables/tab_per_seed_trajectory_stats.csv`, from [W]):

| statistic | seed 30510 | others median | others p10 .. p90 | others min .. max | rank of 30510 (1 = lowest) |
|---|---|---|---|---|---|
| late mean of e2_hat(0) error (u 900-1600) | -0.1362 | -0.0620 | -0.0684 .. -0.0546 | -0.0830 .. -0.0485 | 1 |
| late maximum | -0.0944 | -0.0194 | -0.0313 .. -0.0088 | -0.0386 .. +0.0064 | 1 |
| late SD | 0.0222 | 0.0222 | 0.0195 .. 0.0302 | 0.0165 .. 0.0375 | 10 |
| mean(1225-1600) - mean(900-1200) | +0.0142 | +0.0062 | -0.0114 .. +0.0234 | -0.0258 .. +0.0285 | 13 |
| OLS slope per 100 updates (u 900-1600) | +0.0027 | +0.0015 | -0.0032 .. +0.0058 | -0.0043 .. +0.0084 | 12 |
| fraction of late exports below own leave-one-out p10 | 1.000 | 0.069 | 0.000 .. 0.179 | 0.000 .. 0.310 | 20 |

### 2.5 Shape of the peak  [W]

| quantity at u=1600 | seed 30510 | others median | others p10 .. p90 |
|---|---|---|---|
| e2_hat(0) error | -0.1458 | -0.0611 | -0.0810 .. -0.0283 |
| error at |d|=10 (mean of +-10, relative to e2*(10)) | -0.0671 | +0.0086 | -0.0156 .. +0.0431 |
| error at |d|=20 | -0.0057 | +0.0202 | -0.0139 .. +0.0487 |
| error at |d|=40 | +0.0316 | -0.0199 | -0.0641 .. +0.0057 |
| cusp depth: error at 0 minus error at |d|=10 | -0.0787 | -0.0691 | -0.0731 .. -0.0613 |

The location-free peak of 30510 is -0.1455 at d=1.5; L - S at u=1600 is +0.0003. The error at |d|=10 is -0.0671 (others' median +0.0086) and at |d|=40 +0.0316 (others' median -0.0199) (figures/fig3b_profile_evolution_near_peak.png). The depth of the cusp (error at d=0 minus error at |d|=10) is -0.0787 (others' median -0.0691, p10 -0.0731). e2_hat(0) in effort units at 400/800/1200/1600: 58.8 / 60.6 / 60.2 / 59.8 for 30510, against medians of 62.5 / 65.3 / 66.0 / 65.7 (e2*(0) = 70).

## 3. End-of-A profile (figures/fig3_endA_profile.png)

Contrast seeds: the two other q=50 seeds nearest to the median signed peak error of the others (-0.0611): 30513 and 30506. Sources: [A], [G].

| quantity | seed 30510 | seed 30513 | seed 30506 |
|---|---|---|---|
| peak error (signed) | -0.1458 | -0.0611 | -0.0620 |
| location-free peak error | -0.1455 | -0.0589 | -0.0608 |
| location-free argmax d | 1.5 | -2.5 | -2.0 |
| e2_hat(0) | 59.793 | 65.721 | 65.662 |
| eta_2/DW (final tier) | 0.005803 | 0.001935 | 0.001326 |
| argmax d* of Delta_2 | -4 | -84 | -86 |
| on-path max Delta_2/DW | 0.005803 | 0.001935 | 0.001326 |
| off-path max Delta_2/DW | 0.000819 | 0.000239 | 0.000234 |
| argmax d (off-path) | 102 | 100 | 100 |
| symmetry error max | 2.372 | 3.935 | 1.867 |
| |d| of the symmetry maximum | 52.5 | 22.5 | 68.0 |
| sigma_2(0) | 3.611 | 2.680 | 2.890 |
| mean sigma_2 over |d|<2q | 3.069 | 2.350 | 2.544 |
| tail max (|d|>=2q) | 3.881 | 2.038 | 2.104 |
| tail argmax d | 100 | 100 | 100 |
| RMSE on |d|<2q / e2*(0) | 0.0451 | 0.0287 | 0.0237 |
| tail mean / e2*(0) | 0.00882 | 0.00989 | 0.00708 |
| error at +-10 (effort) | -4.228 | +0.543 | +0.598 |
| error at +-20 (effort) | -0.321 | +1.129 | +1.253 |
| fraction of on-path nodes with e2_hat below e2* | 0.654 | 0.737 | 0.729 |

All 20 q=50 runs (source: `tables/tab_endA_scalars_all_runs.csv`, from [G] and [A]):

| quantity | seed 30510 | others median | others p10 .. p90 | others min .. max | rank (1 = lowest) |
|---|---|---|---|---|---|
| eta_2/DW (= on-path max) | 0.00580 | 0.00154 | 0.00096 .. 0.00247 | 0.00073 .. 0.00310 | 20 |
| off-path max Delta_2/DW | 0.000819 | 0.000428 | 0.000243 .. 0.000588 | 0.000234 .. 0.000675 | 20 |
| sigma_2(0) | 3.611 | 2.881 | 2.719 .. 3.050 | 2.680 .. 3.174 | 20 |
| mean sigma_2 over |d|<2q | 3.069 | 2.544 | 2.444 .. 2.722 | 2.350 .. 2.822 | 20 |
| RMSE on |d|<2q / e2*(0) | 0.0451 | 0.0237 | 0.0197 .. 0.0340 | 0.0133 .. 0.0387 | 20 |
| tail max | 3.881 | 2.568 | 2.102 .. 2.946 | 2.038 .. 3.127 | 20 |
| symmetry error max | 2.372 | 3.028 | 1.118 .. 4.074 | 0.578 .. 4.415 | 10 |
| tail mean / e2*(0) | 0.00882 | 0.00792 | 0.00683 .. 0.00975 | 0.00651 .. 0.01225 | 14 |

- On/off-path split: the maximum of Delta_2 is on-path at d*=-4 (0.00580); the off-path maximum is 0.000819 at d=102. Ranks from the lowest of the 20 q=50 runs: on-path 20, off-path 20 (20 = largest). Others: on-path max 0.00154 median, 0.00310 maximum; off-path 0.000428 median, 0.000675 maximum. On-path value above the G-A threshold 0.005: True; off-path value above it: False.
- Location of d*: the on-path argmax is within |d|<=20 for 13 of the 19 other q=50 seeds (median |d*| 14) and for 20 of 20 q=60 runs; for seed 30510: -4. The on-path maximum of seed 30510 is 0.00580 against a maximum of 0.00310 for the others. At the development-tier calls from u=400 on, the on-path argmax of 30510 is within |d|<=20 at 12 of 13 calls (the others: [(1400, -32.0)]), against a fraction of 0.64 of the other seeds' calls.
- Policy noise: sigma_2(0) = 3.611 (rank 20 of 20 from the lowest; others 2.881 median, 3.174 maximum). Mean sigma_2 over |d|<2q: 3.069 against 2.544 (others' median; rank 20); see panel (e).
- Symmetry error: rank 10 of 20 from the lowest; 2.372 against a median of 3.028. RMSE 0.0451 (rank 20) and tail max 3.881 (rank 20); gate G-A(RMSE) pass (threshold 0.05), gate G-A(tail) pass (threshold 0.02; tail mean/e2*(0) = 0.00882).

## 4. Optimisation (figures/fig4_optimisation_q50.png, figures/fig5_eta2_calls.png)

### 4.1 Window means (sources: [U] for KL, clip fraction, advantage SD, losses; [H] for grad norms; [W] for concentration and sigma_2(0))

| quantity | updates | seed 30510 | others median | others p10 .. p90 | rank (1 = lowest) |
|---|---|---|---|---|---|
| KL after the 10 epochs | 1-100 | 0.00578 | 0.00618 | 0.00573 .. 0.00692 | 4 |
| KL after the 10 epochs | 101-400 | 0.00760 | 0.00553 | 0.00497 .. 0.00587 | 20 |
| KL after the 10 epochs | 401-900 | 0.00769 | 0.00637 | 0.00610 .. 0.00662 | 20 |
| KL after the 10 epochs | 901-1200 | 0.00815 | 0.00706 | 0.00658 .. 0.00746 | 19 |
| KL after the 10 epochs | 1201-1600 | 0.00587 | 0.00541 | 0.00516 .. 0.00578 | 20 |
| clip fraction | 1-100 | 0.0803 | 0.0870 | 0.0822 .. 0.0899 | 2 |
| clip fraction | 101-400 | 0.0738 | 0.0516 | 0.0464 .. 0.0590 | 20 |
| clip fraction | 401-900 | 0.0764 | 0.0610 | 0.0558 .. 0.0675 | 20 |
| clip fraction | 901-1200 | 0.0791 | 0.0653 | 0.0599 .. 0.0756 | 20 |
| clip fraction | 1201-1600 | 0.0568 | 0.0489 | 0.0462 .. 0.0543 | 19 |
| actor grad norm, mean pre-clip (clip level 0.5) | 1-100 | 1.75 | 1.89 | 1.80 .. 1.96 | 1 |
| actor grad norm, mean pre-clip (clip level 0.5) | 101-400 | 4.80 | 2.28 | 2.19 .. 2.65 | 20 |
| actor grad norm, mean pre-clip (clip level 0.5) | 401-900 | 6.37 | 3.14 | 2.84 .. 4.38 | 20 |
| actor grad norm, mean pre-clip (clip level 0.5) | 901-1200 | 7.03 | 3.90 | 3.23 .. 5.45 | 20 |
| actor grad norm, mean pre-clip (clip level 0.5) | 1201-1600 | 6.45 | 3.64 | 3.36 .. 5.11 | 20 |
| advantage SD (raw) | 1-100 | 0.2800 | 0.2897 | 0.2690 .. 0.2950 | 5 |
| advantage SD (raw) | 101-400 | 0.0811 | 0.0779 | 0.0763 .. 0.0796 | 20 |
| advantage SD (raw) | 401-900 | 0.0727 | 0.0608 | 0.0583 .. 0.0638 | 20 |
| advantage SD (raw) | 901-1200 | 0.0658 | 0.0530 | 0.0512 .. 0.0570 | 20 |
| advantage SD (raw) | 1201-1600 | 0.0610 | 0.0491 | 0.0476 .. 0.0532 | 20 |
| critic value loss | 1-100 | 0.17624 | 0.19925 | 0.16706 .. 0.24426 | 4 |
| critic value loss | 101-400 | 0.00323 | 0.00303 | 0.00289 .. 0.00314 | 19 |
| critic value loss | 401-900 | 0.00262 | 0.00184 | 0.00169 .. 0.00202 | 20 |
| critic value loss | 901-1200 | 0.00214 | 0.00138 | 0.00129 .. 0.00159 | 20 |
| critic value loss | 1201-1600 | 0.00184 | 0.00119 | 0.00111 .. 0.00139 | 20 |
| concentration alpha+beta at d=0 (mean of exports) | 1-400 | 110.6 | 128.6 | 125.2 .. 136.8 | 1 |
| concentration alpha+beta at d=0 (mean of exports) | 900-1600 | 167.4 | 246.0 | 221.4 .. 273.3 | 1 |
| sigma_2(0) (mean of exports) | 1-400 | 4.444 | 4.196 | 4.148 .. 4.298 | 20 |
| sigma_2(0) (mean of exports) | 900-1600 | 3.774 | 3.023 | 2.869 .. 3.188 | 20 |

The actor gradient is clipped to norm 0.5 (20 minibatch steps per update); the others' pre-clip window means are of the order of 3-4 (10th percentile 3.23 in the 901-1200 window), i.e. 6.5 times the clip level; the pre-clip norm of 30510 is 14.1 times it. Clipping bounds the step, so a larger pre-clip norm is not by itself a larger step. Ranks (1 = lowest of the 20, 20 = highest) of 30510 in the 1-100 window: KL 4, clip fraction 2, actor grad norm 1, advantage SD 5, critic loss 4 (no excess in this window); in the 101-400 window: KL 20, clip fraction 20, actor grad norm 20, advantage SD 20, critic loss 19. The separation starts right after the first 100 updates.

### 4.2 Where the optimisation footprint leaves the others' band (same rules as 2.1; the 'beyond' side is above the 90th percentile, or below the 10th for the concentration)  [W], [U], [H]

| series | beyond-band side | first export beyond | start of first run of >= 8 | sustained departure | last in band | fraction beyond (all exports) | fraction beyond (u >= 900) |
|---|---|---|---|---|---|---|---|
| concentration alpha+beta at d=0 | below p10 | 125 | 125 | 125 | 100 | 0.94 | 1.00 |
| actor grad norm (mean, pre-clip) | above p90 | 150 | 150 | 150 | 125 | 0.92 | 1.00 |
| sigma_2(0) | above p90 | 175 | 175 | 175 | 150 | 0.91 | 1.00 |
| advantage SD | above p90 | 200 | 275 | 275 | 250 | 0.88 | 1.00 |
| critic value loss | above p90 | 200 | 275 | 275 | 250 | 0.86 | 1.00 |
| clip fraction | above p90 | 150 | 150 | n/a | 1600 | 0.69 | 0.59 |
| KL | above p90 | 150 | 200 | n/a | 1600 | 0.45 | 0.38 |

Spearman correlation across the 19 other seeds between the late level of the peak error and each of these (mean over updates >= 900, n=19): concentration at d=0 +0.15, sigma_2(0) -0.23, actor grad norm -0.10, KL +0.05, advantage SD -0.05, critic loss -0.07 (with 30510 included, n=20: +0.27, -0.34, -0.23). Source: `tables/tab_late_means_per_seed_q50.csv`. Largest absolute coefficient among the 19 others: 0.23. Descriptive only; association is not cause.

### 4.3 Dev-tier eta_2/DW at every Phase-A verifier call  [C]

| update (call) | seed 30510 | others median | others min .. max | rank (1 = lowest, 20 = highest eta_2) |
|---|---|---|---|---|
| 100 | 0.14827 | 0.06494 | 0.02319 .. 0.13817 | 20 |
| 200 | 0.01900 | 0.00890 | 0.00697 .. 0.01478 | 20 |
| 300 | 0.01310 | 0.00555 | 0.00255 .. 0.01077 | 20 |
| 400 | 0.00673 | 0.00315 | 0.00216 .. 0.00868 | 18 |
| 500 | 0.00337 | 0.00318 | 0.00129 .. 0.00571 | 13 |
| 600 | 0.00838 | 0.00189 | 0.00095 .. 0.00385 | 20 |
| 700 | 0.00522 | 0.00259 | 0.00085 .. 0.00867 | 18 |
| 800 | 0.00485 | 0.00246 | 0.00112 .. 0.00717 | 17 |
| 900 | 0.00461 | 0.00217 | 0.00091 .. 0.00536 | 19 |
| 1000 | 0.00460 | 0.00172 | 0.00087 .. 0.00516 | 19 |
| 1100 | 0.00604 | 0.00208 | 0.00086 .. 0.00815 | 19 |
| 1200 | 0.00531 | 0.00192 | 0.00074 .. 0.00712 | 19 |
| 1300 | 0.00409 | 0.00185 | 0.00108 .. 0.00887 | 18 |
| 1400 | 0.00364 | 0.00181 | 0.00066 .. 0.00439 | 17 |
| 1500 | 0.00332 | 0.00127 | 0.00074 .. 0.00583 | 17 |
| 1600 | 0.00580 | 0.00145 | 0.00056 .. 0.00310 | 20 |

- 30510 is above the others' median at 16 of 16 calls, the highest of the 20 at 5, and above every other seed at 5. From u=400 on, its value is 0.00332 at the lowest and 0.00838 at the highest; it is above 0.005 at 6 of 13 calls, with the on-path maximum above the off-path maximum at every one (flag: True).
- For the other seeds, 0.073 of their calls from u=400 on exceed 0.005 (largest value 0.00887); 12 of the 19 had at least one such call. At the last call (u=1600) the others' values are 0.00145 median and 0.00310 maximum (all below 0.005: True).
- Calls of 30510 at u=1300 (0.00409), 1400 (0.00364) and 1500 (0.00332) against the 0.005 threshold (all below: True); the end-of-A value 0.00580 is above it: True.
- Recomputed at every export from u=400 on [V]: 30510 has eta_2/DW between 0.00332 and 0.01374; from u=900 on it is above 0.005 at 15 of 29 exports (median 0.00510); the others' pooled exports from u=900 on: median 0.00178, 90th percentile 0.00382, 99th percentile 0.00728, maximum 0.01108; 0.038 of them exceed 0.005. 30510 is above the others' 90th percentile at 0.83 of the exports from u=900 on. Fraction of 30510's exports from u=900 on above 0.005: 0.52; ratio of the target's median to the others' pooled median over the same exports: 2.9; ratio of the end-of-A values (target over the others' median): 3.8.
- `would_have_fired` ([S], fixed-budget run): update 1300 for 30510; the other seeds: median 700, range 600 .. 900. The Phase-A rule fires after 3 consecutive eligible calls; a call is eligible when eta_2/DW <= 0.02 (not the G-A threshold 0.005) and the maximum over the D2 grid of the normalised policy std is <= 0.04 (= 4.0 effort units). For 30510 eta_2/DW is <= 0.02 from the call at u=200 (others: median 200, latest 200), but the maximum std is above the threshold at calls up to u=1000 (others: last such call, median 400, latest 600); its first eligible call is u=1100 (others: median 500, range 400 .. 700). Of its 10 non-eligible calls, 9 have eta_2/DW <= 0.02 and 9 have the std above the threshold; non-eligible calls with eta_2/DW <= 0.02 and the std at or below the threshold: 0 (figures/fig5_eta2_calls.png panel d). The G-A verdict was fail.

## 5. Sampling (figures/fig6_sampling.png)

Phase A draws exploring starts bin-balanced on D_2 (bin chosen uniformly, then uniform inside the bin; bins of width 10: 40 bins for q=50). The stored visitation counts the learner's starting-gap bins (819200 learner rows over 1600 updates). Peak set: bins [18, 19, 20, 21] (intervals meeting (-20, 20)). Sources: [H] `verifier_calls[A].visitation_cumulative_phase.stage2_direct_es`; `tables/tab_visitation_peak_share_by_call.csv`.

| quantity | seed 30510 | others |
|---|---|---|
| peak-set share at u=1600 | 0.10019 | median 0.09998, range 0.09942 .. 0.10029 |
| design value (4 of 40 bins) | 0.10000 |  |
| binomial z-score of the share | +0.57 | range -1.75 .. +0.88; SD of the 20 q=50 z-scores 0.76, of the 20 q=60 z-scores 1.05 |
| smallest / largest bin share | 0.0246 / 0.0253 | design 0.0250 |

Largest |binomial z| of the cumulative peak-set share of seed 30510 over the Phase-A calls: 1.16 (within +-2: True); at u=1600 the 19 other q=50 runs have z between -1.75 and +0.88.

D1 clamp counts, Phase A, summed over updates 1-1600 (raw Beta draws below 1e-6 or above 1-1e-6 before the clip; source [U] `d1_*` columns):

| count | seed 30510 | other 19 q=50: median | other 19: maximum | rank of 30510 (1 = lowest) |
|---|---|---|---|---|
| learner stage-2 rows | 819200 | 819200 | 819200 |  |
| learner draws below 1e-6 | 58402 | 457 | 23255 | 20 |
| learner draws above 1-1e-6 | 0 | 0 | 0 |  |
|   of which inside |d|<2q (lo + hi) | 0 + 0 | 0 | 0 |  |
| opponent draws below 1e-6 | 58307 |  | 23046 |  |
| learner rows with alpha < 1 | 235140 | 110917 | 221887 | 20 |
| learner rows with beta < 1 | 0 | 0 | 0 |  |
| smallest alpha seen | 0.0252 |  | (others' smallest: 0.0328) |  |

All clamp hits of 30510 are lower-clamp hits at stage-2 states outside the support (|d| >= 2q, where e2* = 0 and the learned mean is small: tail mean 0.62 effort units, alpha < 1). Over all 40 runs: 0 hits inside |d| < 2q and 0 at the upper clamp (all-run total of lo+hi counts 566829, of which q=60 maximum per run 21633). The count of 30510 (58402, 7.1% of its learner rows, 14.3% of its out-of-support rows) is 128 times the median of the others and 2.5 times the next largest (23255); 1 other q=50 seed exceeds 20,000. Its share of rows with alpha<1 is 28.7% against a median of 13.5%. The counts are produced by the policy's draws at the stored starting states, whose bin distribution is the same design in every run (above). Spearman correlation of the number of clamp hits with the peak error -0.03 among the 19 others, -0.26 over all 40 runs (descriptive).

## 6. Three-way decomposition of the d=0 peak gap (figures/fig7_decomposition.png)

Definitions (the repo's: `run/run_v2_T2_locked.smoothed_share`, pilot 4 section 1d). RL gap = e2*(0) - e2_hat(0) (effort units). Smoothing-predicted gap = e2*(0) - e_pred(0), where e_pred(0) = DW/(2k) E[f_xi(x_i - x_j)] and x_i, x_j are independent draws of the policy's own d=0 noise (Beta quantile nodes centred on its mean), i.e. what a best responder would play against the policy's own randomness. Remainder = e_pred(0) - e2_hat(0) = RL gap - smoothing-predicted gap (the 'smoothing-free' part). Share = smoothing-predicted gap / RL gap (`smoothed_share_peak_gap_d0`). Supervised floor = e2*(0) - e2_fit(0) for the same actor class fitted to e2* by least squares (pilot 4 section 1d, 5 inits, 300,000 steps; the plateau rule never fired, so these values are upper bounds of the floor; this is not a fit of this seed). Sources: [G] (`reported.end_of_A`), [F].

| component (effort units) | seed 30510 | others median | others p10 .. p90 | others min .. max | rank (1 = lowest) |
|---|---|---|---|---|---|
| RL gap  e2*(0) - e2_hat(0) | 10.207 | 4.279 | 1.984 .. 5.669 | 1.157 .. 6.262 | 20 |
| smoothing-predicted gap | 2.851 | 2.274 | 2.147 .. 2.408 | 2.116 .. 2.506 | 20 |
| remainder (smoothing-free) | 7.355 | 2.056 | -0.208 .. 3.420 | -1.177 .. 3.947 | 20 |
| share = smoothing gap / RL gap | 0.279 | 0.526 | 0.398 .. 1.141 | 0.370 .. 2.017 | 1 |
| e_pred(0) | 67.149 | 67.726 | 67.592 .. 67.853 | 67.494 .. 67.884 | 1 |

- Supervised floor (same actor class, fitted to e2*): median 0.00133 effort units over 5 initialisations (range -0.0531 .. 0.0016); the RL gap of 30510 is about 7674 times that median. The pilot-4 reference (Phase-A extension, 10 other seeds at u1600, [F]): RL gap median 4.784, smoothing-predicted 2.159. Largest absolute deviation of the five floor fits at d=0: 0.0531 effort units.
- RL gap of 30510 is 2.39 times the others' median (6.26 maximum). Its smoothing-predicted part is 2.851, larger than any other seed's (2.506 maximum); the prediction is a function of sigma_2(0), which is the largest; it covers 0.279 of the gap against a median of 0.526 for the others (rank 1 from the lowest). The remainder, 7.355, is 3.6 times the others' median (2.056; maximum 3.947).
- q=60 for scale (20 runs): RL gap median 3.954 (p10 2.623, p90 5.192, max 7.213); remainder median 2.359 (max 5.361); share median 0.397.
- Across the 19 other q=50 seeds, OLS of RL gap on smoothing-predicted gap: slope 2.76, intercept -2.14; the line predicts 5.73 for 30510, observed 10.21 (residual +4.48 = 2.9 residual SDs; descriptive, 19 points, an extrapolation in the smoothing-gap axis). Panel (b) shows the same.

## 7. Pre-registered hypothesis rules H1-H4, applied literally

The rules below are those of `reports/v2/refine_r2b/01_preregistration.md` section 5, committed in `1ff99bd` (the draft text was written at 07:03 UTC, before any output of the diagnostic existed; the pre-registration states which diagnostic results the author had seen before committing it) and are applied literally by `apply_rules()` in `tools/v2/diag_seed30510.py` (tested on synthetic trajectories in `tests/test_v2_r2b_diag.py`; the tool does not read the pre-registration file). Nothing was changed after the data were seen, including where a literal rule is awkward. Every verdict word in this section and in section 0 is produced from the computed booleans (`numbers.json` keys `rules.target.*`; `tables/hypothesis_rules.csv`).

Notation. S(u) = (e2_hat(0; u) - e2*(0)) / e2*(0): signed peak error at d=0 from the weight export of update u (u = 25, ..., 1600). L(u) = (max_d e2_hat(d; u) - e2*(0)) / e2*(0) on the recovery grid (location-free peak error, `pilot4_common.location_free`). Pack = the 19 other q=50 runs of confirmation_v2_0. p10_pack(., u), p90_pack(., u): 10th / 90th percentile (numpy.percentile, linear) over the pack at the same export. out_S(u) := S_30510(u) < p10_pack(S, u). u_leave := the first export from which out_S holds at that export and at every later export up to 1600 (undefined if out_S(1600) is false). Resolution 25 updates.

- **H1**: peak never rose to the pack's level: max over ALL exports u of S_target(u) < p10_pack(S, 1600)
- **H2**: rose, then decayed in the LR-decay window: not H1; out_S(1600); u_leave >= 1225
- **H3**: late excursion after a stable plateau: not H1; out_S(1600); u_leave <= 1200; and the target is NOT out_S at >= 16 consecutive exports ending at u_leave - 25
- **H4**: plateau at the pack's level of peak HEIGHT with an unusually rounded cusp: out_S(1600); L_target(u) >= p10_pack(L, u) at every export u in 1225..1600; and (L - S)_target(1600) > p90_pack(L - S, 1600)
- **NONE**: if out_S(1600), not H1, u_leave <= 1200 and no 400-update (16-export) plateau: 'none of H1-H4 (early or unstable departure)'

Verdict per hypothesis: 'supported' when the rule holds, 'not supported' when it does not, 'undetermined' if an input is missing. The rules are not exclusive by construction.

| hypothesis | computed inputs | verdict |
|---|---|---|
| H1 | max over all 64 exports of S = -0.0582 (at u=825); p10_pack(S, 1600) = -0.0810; max < p10: False | not supported |
| H2 | not H1: True; out_S(1600): True; u_leave = 900, u_leave >= 1225: False | not supported |
| H3 | not H1: True; out_S(1600): True; u_leave = 900, u_leave <= 1200: True; consecutive exports not out_S ending at u_leave - 25: 1, >= 16: False | not supported |
| H4 | out_S(1600): True; L >= p10_pack(L, u) at every export 1225..1600: False (16 of 16 exports below; smallest margin L - p10_pack(L) = -0.0929); (L - S)(1600) = +0.00027 against p90_pack(L - S, 1600) = +0.00234, exceeds: False | not supported |

**Label (from the rules): none of H1-H4 (early or unstable departure).** The 'none' rule: out_S(1600) True, not H1 True, u_leave <= 1200 True, no 16-export (400-update) plateau True: all four hold: True.

Awkward-looking features of the literal rules, reported as they come out (not changed): H1 compares the maximum over ALL exports, early transient included, with p10_pack(S, 1600); for seed 30510 the margin max S - p10_pack(S, 1600) is +0.0228 (H1 needs it to be negative), the maximum being at u=825. H3 counts the exports not out_S that end at u_leave - 25 = 875; the run is 1 export(s) long. Consistency check: u_leave equals the 'sustained departure' of section 2.3: True.

### 7.1 The same rules applied to the other runs (how they behave on runs that passed)

Each of the 20 q=50 runs taken as the target against the other 19 q=50 runs (the seed 30510 row repeats the target above), and each of the 20 q=60 runs against the other 19 q=60 runs. Source: `tables/hypothesis_rules.csv` (`numbers.json` keys `rules.table.*`). Columns: G-A = the run's own G-A verdict; S = S(1600); p10 = p10_pack(S, 1600); out = out_S(1600); max S (u) = H1's statistic; plateau = consecutive exports not out_S ending at u_leave - 25.

**q=50 block.** Labels (none of H1-H4 (not out of the pack at u=1600): 17; supported: H2: 2; none of H1-H4 (early or unstable departure): 1). Runs with out_S(1600): 3 of 20; supported counts H1 / H2 / H3 / H4: 0 / 2 / 0 / 0.

| seed | G-A | S | p10 | out | u_leave | max S (u) | plateau | H1 | H2 | H3 | H4 | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30501 | pass | -0.0508 | -0.0865 | False | n/a | -0.0370 (1275) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30502 | pass | -0.0791 | -0.0865 | False | n/a | -0.0194 (1150) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30503 | pass | -0.0520 | -0.0865 | False | n/a | -0.0152 (1150) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30504 | pass | -0.0574 | -0.0865 | False | n/a | -0.0027 (1450) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30505 | pass | -0.0895 | -0.0810 | True | 1600 | -0.0369 (425) | 14 | not supported | supported | not supported | not supported | supported: H2 |
| 30506 | pass | -0.0620 | -0.0865 | False | n/a | -0.0013 (800) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30507 | pass | -0.0329 | -0.0865 | False | n/a | -0.0117 (1300) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30508 | pass | -0.0759 | -0.0865 | False | n/a | +0.0064 (1100) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30509 | pass | -0.0485 | -0.0865 | False | n/a | -0.0194 (1350) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30510 | fail | -0.1458 | -0.0810 | True | 900 | -0.0582 (825) | 1 | not supported | not supported | not supported | not supported | none of H1-H4 (early or unstable departure) |
| 30511 | pass | -0.0523 | -0.0865 | False | n/a | -0.0266 (450) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30512 | pass | -0.0760 | -0.0865 | False | n/a | -0.0039 (700) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30513 | pass | -0.0611 | -0.0865 | False | n/a | -0.0191 (1075) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30514 | pass | -0.0165 | -0.0865 | False | n/a | -0.0103 (1300) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30515 | pass | -0.0193 | -0.0865 | False | n/a | -0.0186 (1525) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30516 | pass | -0.0706 | -0.0865 | False | n/a | -0.0215 (575) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30517 | pass | -0.0752 | -0.0865 | False | n/a | -0.0299 (1025) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30518 | pass | -0.0798 | -0.0865 | False | n/a | -0.0199 (1150) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30519 | pass | -0.0306 | -0.0865 | False | n/a | -0.0198 (1450) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30520 | pass | -0.0858 | -0.0817 | True | 1600 | -0.0143 (1450) | 8 | not supported | supported | not supported | not supported | supported: H2 |

**q=60 block.** Labels (none of H1-H4 (not out of the pack at u=1600): 17; supported: H2: 3). Runs with out_S(1600): 3 of 20; supported counts H1 / H2 / H3 / H4: 0 / 3 / 0 / 0.

| seed | G-A | S | p10 | out | u_leave | max S (u) | plateau | H1 | H2 | H3 | H4 | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30501 | pass | -0.0674 | -0.0895 | False | n/a | -0.0127 (1300) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30502 | pass | -0.0857 | -0.0895 | False | n/a | -0.0437 (1000) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30503 | pass | -0.0465 | -0.0895 | False | n/a | -0.0019 (1225) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30504 | pass | -0.0820 | -0.0895 | False | n/a | -0.0015 (600) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30505 | pass | -0.0667 | -0.0895 | False | n/a | -0.0163 (1575) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30506 | pass | -0.0499 | -0.0895 | False | n/a | +0.0021 (675) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30507 | pass | -0.0539 | -0.0895 | False | n/a | -0.0252 (1550) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30508 | pass | -0.0809 | -0.0895 | False | n/a | -0.0200 (575) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30509 | pass | -0.0307 | -0.0895 | False | n/a | -0.0164 (950) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30510 | pass | -0.0687 | -0.0895 | False | n/a | -0.0339 (925) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30511 | pass | -0.0154 | -0.0895 | False | n/a | -0.0076 (775) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30512 | pass | -0.0682 | -0.0895 | False | n/a | -0.0267 (675) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30513 | pass | -0.0612 | -0.0895 | False | n/a | -0.0069 (900) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30514 | pass | -0.0788 | -0.0895 | False | n/a | -0.0245 (825) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30515 | pass | -0.0885 | -0.0872 | True | 1600 | -0.0083 (900) | 13 | not supported | supported | not supported | not supported | supported: H2 |
| 30516 | pass | -0.0735 | -0.0895 | False | n/a | -0.0081 (1075) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30517 | pass | -0.0582 | -0.0895 | False | n/a | -0.0187 (1400) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30518 | pass | -0.0628 | -0.0895 | False | n/a | -0.0187 (825) | n/a | not supported | not supported | not supported | not supported | none of H1-H4 (not out of the pack at u=1600) |
| 30519 | pass | -0.0934 | -0.0863 | True | 1600 | -0.0095 (950) | 39 | not supported | supported | not supported | not supported | supported: H2 |
| 30520 | pass | -0.1237 | -0.0863 | True | 1525 | -0.0158 (825) | 1 | not supported | supported | not supported | not supported | supported: H2 |

### 7.2 Post hoc readings (not pre-registered)

These are the author's own readings, written after the data were seen. They are not the pre-registered rules and carry no verdict; each line gives a rule, its computed inputs and whether it holds.

- H1, strict from u >= 400 (S below p10_pack at every export u >= 400): does not hold; in band at 4 of 49 exports ([500, 525, 825, 875]).
- H1, plateau reading (S below p10_pack at every export u >= 900): holds (29 of 29); late mean -0.1362 against the others' median -0.0620; first export with S >= -0.10 at u=825 (others' median 400).
- H2-like, in band at u=1200 and out of band at u=1600: does not hold (S(1200) = -0.1402 against p10 -0.0898; fraction of exports 900-1200 in band 0.00; last in-band export 875, i.e. 326 updates before the decay window). Decay-window change mean(1225-1600) - mean(900-1200) = +0.0142, inside the others' range (-0.0258 .. +0.0285).
- H3-like, at least half of the exports 900-1200 in band followed by a drop out of band: does not hold (fraction in band 0.00); largest single step 0.0619 (others' maximum 0.0848; not above it: True); late slope +0.0027 per 100 updates (others' median +0.0015).
- H4-like shape readings at u=1600: location-free peak at or above the others' p10 (-0.1455 against -0.0805): does not hold; error at |d|=10 at or above the others' p10 (-0.0671 against -0.0156): does not hold; cusp depth (error at 0 minus error at |d|=10) outside the others' p10..p90 (-0.0787 against -0.0731 .. -0.0613): holds; smoothing-free remainder of the d=0 gap above the others' maximum (7.36 against 3.95): holds.
- Timing of the optimisation footprint (descriptive): sustained departure of the concentration at update 125, actor grad norm 150, sigma_2(0) 175, advantage SD and critic loss 275; these dates say when series leave their bands, not which one drives which.

## 8. What cannot be distinguished, and which experiment would

1. Cause or co-symptom of the early concentration/noise divergence. The footprint series leave their bands for good at updates 125-275; the pre-registered u_leave of the peak is 900. The smoothing prediction accounts for 28% of the gap; among the other 19 seeds the Spearman coefficients between the late peak error and these quantities are at most 0.23 in absolute value. The files contain no manipulation that separates 'a lower concentration produced the low peak' from 'both follow from an earlier state of the network'. An experiment that would: re-run seed 30510 (the pipeline is seeded, so the same streams are intended to reproduce it; not verified here) with a full-state checkpoint at update 100 (`full_state_at`), then branch from it with the minibatch and sampling streams reseeded (same network state, different noise): if every branch ends at the same plateau, the plateau is fixed by the state at update ~100; if only some do, it is noise-driven. A second branch with the concentration manipulated at that state (the repo has a `conc_anneal` mechanism for phase A continuation) would test the concentration route directly. Not run.
2. Whether the plateau is permanent. Over updates 900-1600 the late slope is +0.0027 per 100 updates (others +0.0015). `state_end_A.pt` of the run exists, so a Phase-A continuation from it (mode `phase_A_continue`) would show whether it moves; not run.
3. How much of the G-A failure is the end-iterate draw. The per-export eta_2 of 30510 is above 0.005 at 15 of 29 exports from u=900 on; the ratio of its median to the others' pooled median is 2.9 (over the same exports). A tail-averaged candidate (pilot 4 section 1c) would be the experiment, and it is outside this diagnostic.
4. The clamp counts. A conjecture, untested: rows whose raw draw is clipped at 1e-6 have a log-density of the stored action that is large, and the gradient of that log-density with respect to alpha is of order |ln 1e-6| = 13.8 per such row; 30510 has more of them, and its pre-clip actor gradient norm is larger. The files show only the co-occurrence (section 5 lists the counts and their Spearman correlation with the peak error). Re-running with the clip level moved would test it; not run.
5. The export at u=825 is a single point at a 25-update resolution; the data cannot say whether it is a transient of the policy mean or a move that was reversed.

## 9. Not reproduced, inconsistencies, caveats

- Reproduction of the four numbers quoted in the request: all four match to the digits quoted.
- The location-free peak error recomputed from the weights differs from `gates.json` in the 7th digit (float32 rounding: numpy against torch); the `recovery_e2` arrays agree to 3e-5 effort units.
- The weight exports are every 25 updates; any statement about an update that is not a multiple of 25 is not possible from the exports. Verifier calls are every 100 updates (dev tier) with these exceptions (runs whose Phase-A calls are off that grid or not labelled timeout/warmup, as update (reason)): q=50 seed 30517: 1600 (stability); q=60 seed 30505: 540 (stability), 640 (timeout), 740 (timeout), 840 (timeout), 940 (timeout), 1040 (timeout), 1140 (timeout), 1240 (timeout), 1340 (timeout), 1440 (timeout), 1540 (timeout), 1600 (phase_end). For q=60 seed 30505 a stability-triggered call at u=540 shifts the later calls to u=640, 740, ...; those calls have no weight export and are left out of the check of the recomputed eta_2 against the CSV (they remain in `tables/tab_eta2_verifier_calls_all_runs.csv`). No q=50 call used in the tables above is affected except that seed 30517's call at u=1600 is labelled `stability` (the same update).
- The supervised floor and the pilot-4 reference values are read from the pilot-4 tables [F]; they were not recomputed here and do not belong to these seeds. The floor fits stopped at the step cap and are upper bounds.
- 'Others' are 19 values: percentiles are coarse, and band membership at a single export carries little information; run-level statements use leave-one-out frequencies and ranks. Exports of a run are autocorrelated, so no binomial tail probability is quoted.
- Provenance of the verifier: `utils/dp_br_verifier.py` (sha256 `3054e5500eea8876...`) and `utils/theory_multistage.py` (`eff8d2f70e3fbc3a...`) were imported read-only from the worktree given with `--repo`, not from the commit that produced the runs (d2e377d); the agreement with the stored CSV to 4.3e-08 at every call shows that it behaves as the code that was used.
- The per-export eta_2 (section 4.3, last bullet) is the development tier (state step 4); the gate uses the final tier. For 30510 at u=1600 the two on-path values are identical.
- The mean over buffer states of the entropy (`entropy_post_update_effort_scale`) and of the concentration (`conc_buf_mean`) are computed over a state-uniform buffer that is half out-of-support; they are in `tables/tab_opt_window100_all_runs.csv` but are not interpreted here.

## 10. Files

Figures (pdf with fonts type 42, and png): `reports/v2/refine_r2b/figures/` (the report refers to them as figures/figN.png). Everything else under `results/v2_refine_r2b/diag_30510/`: `report_draft.md` (the tool's own copy of this text before the header edits), `numbers.json` (values and sources), `tables/hypothesis_rules.csv` (pre-registered rules for the target and the 40 runs), `tables/*.csv` (every plotted series: `tab_peak_trajectory_band_q50.csv`, `tab_peak_trajectory_band_q60.csv`, `tab_exports_all_runs.csv`, `tab_endA_profile_*.csv`, `tab_opt_trailing25_band_q50.csv`, `tab_opt_window100_all_runs.csv`, `tab_opt_target_per_update.csv`, `tab_eta2_verifier_calls_all_runs.csv`, `tab_eta2_every_export_q50.csv`, `tab_visitation_*.csv`, `tab_d1_clamp_counts_phaseA_sum.csv`, `tab_decomposition_*.csv`, `tab_per_seed_trajectory_stats.csv`, `tab_late_means_per_seed_q50.csv`, `tab_profile_evolution_near_peak.csv`, `tab_would_have_fired_q50.csv`).

Run commit of the data: `d2e377d0da702e8e162662f74576d1bff5b42a56` (clean tree: True). Tool sha256 `0c832417d822d656...`. Command, as run from the repository root:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B tools/v2/diag_seed30510.py \
    --out results/v2_refine_r2b/diag_30510 --fig-dir reports/v2/refine_r2b/figures
```
(`--run-root` defaults to the v2-t2-refine worktree's `results/v2_T2_locked/confirmation_v2_0`, `--repo` to this worktree for the read-only import of the verifier, `--floor-dir` to the v2-t2-refine pilot-4 floor tables.)

---

## Addendum 1 (2026-10-04, after the pilots, from the conformance audit against the prompt)

Nothing above was changed; the tool output (`results/v2_refine_r2b/diag_30510/`, `report_draft.md`, `numbers.json`, figure 6) is left as produced. Three corrections and two clarifications; the evidence for each is in the audit record `results/v2_refine_r2b/review/post_stop_prompt_audit_journal.jsonl` and was re-checked here.

1. **Correction (section 5, last paragraph).** The all-run clamp total "566829" double-counts the learner's out-of-support hits: in `tables/tab_d1_clamp_counts_phaseA_sum.csv` the column `d1_L_s2_out_lo_sum` equals `d1_L_s2_lo_sum` in every run, and the tool summed every column that ends in `_lo_sum` or `_hi_sum`. Recounted from the same table: learner stage-2 draws below the clamp 189,170, opponent stage-2 draws below the clamp 188,489, all other counts 0 (no upper-clamp hit, no hit inside |d| < 2q, none at stage 1), i.e. **377,659 distinct clamped draws over the 40 runs**. The statements "0 hits inside |d| < 2q and 0 at the upper clamp" and the counts of seed 30510 (58,402 learner draws below the clamp; the next largest run 23,255) are unaffected. Figure 6, panel (c), carries the old total.
2. **Correction (section 3, "Location of d*").** In "(the others: [(1400, -32.0)])" the bracket is the one development-tier call of seed 30510 itself whose on-path argmax lies outside |d| <= 20 (update 1400, d* = -32), not a statement about the other seeds.
3. **Plain statement per hypothesis** (the report has the evidence in sections 7, 7.1, 7.2 and 8). *Under the pre-registered literal rules:* H1, H2, H3 and H4 are all not supported for seed 30510, label "none of H1-H4 (early or unstable departure)". *Under the post hoc readings (not pre-registered, section 7.2):* H1 holds in its plateau form (below the pack's 10th percentile at 29 of 29 exports from update 900 on; it does not hold from update 400 on, the run being in band at 4 of 49 exports); H2 and H3 do not hold (no in-band stretch of exports 900-1200, last in-band export 875, the change over the LR-decay window is inside the other seeds' range); the level claim of H4 does not hold (the location-free peak, -0.1455, and the error at |d| = 10 are below the others' 10th percentiles), while its "large smoothing-free gap" clause holds (remainder 7.36 effort units against a maximum of 3.95 for the other seeds). *What the files cannot distinguish* (section 8): whether the early divergence of the concentration, the gradient norm and sigma_2(0) (updates 125-275) caused the low plateau or is a co-symptom of an earlier state of the network; whether the plateau is permanent; how much of the G-A failure is the draw of the end iterate (eta_2 above 0.005 at 15 of 29 exports from update 900 on); and whether the single export at update 825 is a transient.
4. **Clarification (H4).** The pre-registered H4 rule uses the location-free peak height and the cusp rounding L - S (`01_preregistration.md` section 5), a choice the prompt's wording ("a plateau at the pack's level with an unusually rounded cusp (large smoothing-free gap)") left open; the quantity the prompt names in parentheses, the smoothing-free remainder of the d = 0 gap, is in section 6 and in the post hoc reading above. H4 is not supported because the peak height is low, not because the cusp is not unusually rounded.
5. **Clarification (what the literal rules show).** The literal rules fire (H2 "supported") on 5 of the other 39 confirmation runs, all of which passed G-A (q=50 seeds 30505 and 30520; q=60 seeds 30515, 30519, 30520; section 7.1), so they do not separate seed 30510 from passing runs; the plateau reading of section 7.2 is what does.
