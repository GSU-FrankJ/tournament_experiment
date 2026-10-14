# MS-R4: summary of the round (P0-P3)

Date: 2026-10-14. Branch `ms-r4` (from `origin/t2-status-pack` `be2881ff`). Status: the pilot is done and reported; **STOP** after the push of section 4.3 of the prompt (`pi_record/23_ms_r4_prompt.md`); the next round is the PI's decision with the coworker. No fresh seed (40501-40520), lock, confirmation, combined E1 + E2 run, T = 3 run or second pilot was used or started; no parameter changed after the pre-registration (`02_preregistration.md`; Addendum 1 records the coworker's note, Addendum 2 the PI's decision at gate G1); `main` and tags were not touched.

The questions (the coworker's two tries at T = 2):

- **E1:** does an actor with LeakyReLU hidden units and a smooth, unclamped mean map (`lrsm`), on `relu_st_s16`'s recipe, bring the mean |peak error| to <= 0.04 at both q with a paired improvement over `t1_st_s16` and no gate failure?
- **E2:** does the development DP-BR help PPO training, i.e. does sampling feedback lower the area under the stage-2 eta_2/DW curve by at least 20 % against fixed stratified sampling at the t1 actor and v2.0's budgets?

Seven arms (E1: `t1_st_s16`, `relu_st_s16`, `lrsm_st_s16`; E2: `fb_a0`, `fb_r`, `fb_rg`, `fb_rx`) x q in {50, 60} x seeds 10511-10530 = 280 runs.

## Reading order

1. This file, then `05_decision_inputs.md` (the outcomes against the coworker's targets, what the plan of Appendix A implies next, the confirmation rule A2, what is not settled).
2. `04_e1_actor.md` and `04_e2_feedback.md` (checks first, per-arm configuration, per-seed tables, the target, failures and decomposition or secondaries, curves, cost, predictions against the outcome; the figures are embedded).
3. `02_preregistration.md` (design, criteria, predictions, Addenda), `03_checks.md` (P1 tests, review, C-R7, C-MS6; section 9: the P2 checks), `00_housekeeping.md`, `01_preamble_numbers.md`, `01b_curve_tool.md`, `pi_record/01_factcheck.md` (the fact-check ledger), `handoff.md` (the message to the coworker, Chinese then English; nothing was sent).

## Headline numbers (paths: `results/ms_r4/analysis/`; tables by `reports/ms/r4/report_scripts/pilot_tables.py`)

- **Checks (P2):** 280 of 280 runs exit 0, tail share 280/280, tail mass 280/280, near-tie and middle masses 120/120 in the three within-stratum arms, scale 280/280, C-INIT 40/40, **C-FB25 120/120**, **C-X2 40/40**, sampling weights 160/160, **C-OFF 280/280 runs, 260,480 comparisons, 0 mismatches**; start-share flags 8 of 840 tests against 2.27 expected (five runs; reported, not failures); both blind recomputations agree (0 differences at 1e-12) (`results/ms_r4/pilot/launch_checks.json`, `results/ms_r4/curves/c_off.csv`). Config gate before the launch: the regenerated plan equals the 280 `r4` rows of `run_config_hashes.csv` (Addendum 2).
- **E1: not met** (the line stops; no other slope, map or variant follows). q = 60: all three parts hold (mean |peak error| 0.0259; paired difference to `t1_st_s16` -0.0122 [-0.0219, -0.0016]; 0 of 20 runs failing). q = 50: (a) holds (0.0329), (b) fails (-0.0025 [-0.0175, +0.0196]), (c) fails: 4 of 20 `lrsm` runs fail a gate (`t1_st_s16` 0 of 20, `relu_st_s16` 3 of 20). Three of the four `lrsm` failures have a peak error of 0.011-0.025 and fail G-A on eta_2/DW and RMSE_pos. *Post hoc:* the RMSE_pos at update 2000 is 2.3-8.7 times its value at update 1200 in all seven failing E1 runs (`relu` and `lrsm`) and in 2 of 113 passing runs. `lrsm` - `relu` is +0.0008 [-0.0212, +0.0263] and +0.0086 [-0.0000, +0.0174]. The effect is that of the combined change; no part is attributed to either.
- **E2: neither `fb_r` nor `fb_rg` meets the target; the DP-BR stays a verifier and no training improvement is claimed.** The area reduction rho against `fb_a0` (target >= 0.20) is +0.020 [-0.014, +0.051] and -0.020 [-0.048, +0.008] for `fb_r`, +0.013 [-0.024, +0.049] and -0.009 [-0.038, +0.023] for `fb_rg` (q = 50, 60); every interval lies below 0.06. `fb_r` also violates (E2-b) at q = 50 (one run at eta_2/DW = 0.00516 against 0.005). (E2-c) holds for both.
- **Descriptive:** `fb_rx` (cross-stratum sampler) has rho +0.054 [+0.014, +0.094] at q = 50 and -0.126 [-0.166, -0.085] at q = 60, a smaller late-window area at both q (+0.312, +0.153), and moves 0.12 of the sampling mass from the near-tie to the middle stratum. `fb_rg` has lower end-of-stage values at q = 50 (eta_2/DW at update 1600 -0.00074 [-0.00124, -0.00029], |peak error| -0.0069 [-0.0127, -0.0013]) and none at q = 60; 15 of 60 secondary intervals exclude 0 against about 3 expected by chance.
- **Predictions (D7):** E1-P1 held in direction and fell short in size; E1-P2 held for `relu` (3 of 20 fail) and failed for `lrsm` (4 of 20); E2-P1, E2-P2 and E2-P3 held (`fb_rx` area +12.6 % at q = 60; no arm reaches 20 %; the checks take at most 0.44 % of the stage-2 wall time).
- **Cost:** the `lrsm` forward costs nothing measurable (median total wall time 616 s and 590 s against 623 s and 600 s for `t1`); the development verifier takes 0.40-0.42 % of the stage-2 wall time in every arm. The wave took about 64 minutes with 40 workers (launch record `results/ms_r4/pilot/launch_20261013_235029.json`).

## What this does not show

Twenty seeds per q: an interval that contains 0 does not show the absence of an effect, and one failed run moves a mean of twenty by about 0.01. The cause of the `relu` and `lrsm` failures at q = 50 is not established (in all seven failing runs RMSE_pos at update 2000 is 2.3-8.7 times its value at update 1200, against a median ratio of 1.0 in the 113 passing runs, a post hoc description; nothing was varied to test why). E2 tested sampling feedback at one strength inside strata (alpha 0.5, two focus maps); it does not say that no sampling feedback can help. `fb_rx`'s late-window gain cannot be separated from its change of the strata shares. The stop rule and polishing were not tested again.

## Coverage

The coworker's deliverables per arm and D5:

| item | where |
|---|---|
| configuration and code version of every arm | `04_e1_actor.md` and `04_e2_feedback.md` section 2 (blocks `config_arms`, `config_diffs`; manifest commit `2f618531`) |
| per-seed results | section 3 of both (blocks `e1_per_seed`, `e2_area_seed`; every run is a row of `per_run.csv`) |
| paired intervals | E1 sections 4 and 5.2; E2 sections 4 and 5 |
| peak (|peak error|, signed error, location-free peak, argmax, symmetry error, per-stratum errors) | E1 sections 3, 4, 5.2 (blocks `e1_paired_peak`, `e1_decomp`, `e1_locfree`, `e1_strata`); E2 section 5 (`peak_abs`, `gap`) |
| learning curves | E1 section 6; E2 section 3 (blocks `e1_curves`, `e2_curves`; all points in `learning_curves_*.csv`) |
| final gates | E1 sections 4 and 5.1; E2 section 3 (block `e2_gate_counts`) |
| run cost | E1 section 6; E2 section 8 (block `cost`; per phase and by category: `cost_e1.csv`, `cost_e2.csv`) |
| D5 E1: failures (trajectories, tie at the bound, per-unit shares) | E1 section 5.1 (blocks `e1_failure_traj`, `e1_failed_runs`, `e1_units`) |
| D5 E2: sampling (strata counts per update, weights per check, rho_bar, g_bar) | E2 section 6 (blocks `e2_strata`, `e2_weights`; `strata_shares_by_update.csv`, `sampling_weights.csv`, heat maps) |
| both tiers | `04_e1_actor.md` section 5.3 and `04_e2_feedback.md` section 3 (block `tiers`: eta_2/DW and Gmax_full/DW on both tiers and the G-N differences); the development tier of every stratum is in `strata_summary.csv`, the rest in `per_run.csv` (`*_dev` columns) and, per update, in the online checks that C-OFF ties to the offline curves |
| learned tie effort e_hat_2(0) along the stage | `04_e1_actor.md` section 6 (block `e1_curves metric=e2`) |
| realised start shares of the E1 arms (prompt 3.2) | `04_e1_actor.md` section 5.3 (block `e1_shares`) |
| per-seed peak of the E2 arms | `04_e2_feedback.md` section 3 (block `e2_peak_seed`) |

## Deviations and recorded departures

1. **P3 was authored in the session's own worktree, not in the launch worktree.** The launch (steps A1.1-A1.5) ran in `.claude/worktrees/ms-r4-e1-e2-experiments-2da78b`, as the PI's reply required (the config hashes contain its absolute path), and so did the post-launch checks, the curves and the analyses (their outputs are in that worktree's `results/ms_r4/`). The reports, the table script and the commits of P3 were made in this session's worktree `.claude/worktrees/gate-g1-decision-launch-55df61` on the local branch `claude/ms-r4-p3` (from `origin/ms-r4` `2f618531`), because the file tools of this session write only inside that worktree. The tracked record files were copied (`rsync`) from the launch worktree and are byte-identical (SHA-256 compared for all 3,643 pilot files and all 330 analysis and curve files). `ms-r4` is pushed from that branch as a fast-forward. The local branch `ms-r4` of the launch worktree stays at `2f618531` until it is fast-forwarded to `origin/ms-r4` (owner-side; its untracked result files would have to be removed or left out first, since identical copies are now tracked).
2. **`python` is not on the PATH of the machine; every command used `/home/fjiang4/tournament_experiment/.venv/bin/python`** (torch 2.5.1+cu121), as the prompt prescribes.
3. **The launch tree:** `git status --porcelain` at the launch listed 840 entries, all untracked files under `results/ms_r4/` (the heavier C-MS6 and C-R7 files, which P1 did not commit); the tracked tree was clean, all 280 manifests record a clean tree, and the same state held at the C-R7 launch. The prompt's "clean tree" is read as the tracked tree.
4. **E2-c measures the stage-2 deviation gain**, not the stage-1 one (`Gmax_full/DW` equals `eta_2/DW` in all 280 runs); reported in `04_e2_feedback.md` section 7. No verdict depends on it; the pre-registered definition was not changed.
5. **Report supplements not in D5 or D6** (labelled in the reports): the pass / fail split of |peak error| (`e1_pass_fail_split`), the hidden-unit counts (a D5 item, post hoc), the stage-1 deviation gain table (`e2_stage1_gain`), the early-area share (`e2_early_share`), the multiplicity count (`e2_secondary_count`), the RMSE_pos rise of the failing runs (`e1_rmse_rise`) and the illustration of rule (i) of A2 (`a2_rule_i`). None adds a test to the pre-registered ones.
6. **Crash rule:** not needed (0 nonzero exits, no run moved to `crashed/`, no re-run).
7. **The six skipped tests** of the P1 suite (MS-R1 / MS-R2 pilot arrays absent in the P1 worktree) were accepted by the PI at gate G1; they were not re-run in P3.
8. **Local refs** (the primary checkout's `main`, `ms-r1` and `t2-status-pack` behind their `origin` refs) are as `00_housekeeping.md` section 2 records; not touched.

## Commands

```
# launch (in tmux, in the launch worktree; from commit 2f618531, tree clean)
tools/ms/r4_launch.py --wave r4 --params reports/ms/r1/prereg_parameters.json --workers 40 --code-commit 2f618531be7c8a9a1a9e30c79014650416b945f3
# post-launch checks, curves (C-OFF), analyses, blind recomputations (results/ms_r4/...)
tools/ms/r4_launch_checks.py --mode pilot --root results/ms_r4/pilot --code-commit 2f618531... --out results/ms_r4/pilot/launch_checks.json
tools/ms/r4_curves.py --pilot-root results/ms_r4/pilot --out results/ms_r4/curves --workers 40
tools/ms/r4_e1_analysis.py --pilot-root results/ms_r4/pilot --curves-root results/ms_r4/curves --criteria reports/ms/r4/prereg_criteria.json --out results/ms_r4/analysis/e1
tools/ms/r4_e2_analysis.py --pilot-root results/ms_r4/pilot --curves-root results/ms_r4/curves --criteria reports/ms/r4/prereg_criteria.json --out results/ms_r4/analysis/e2
tools/ms/r4_blind_e1.py --per-run results/ms_r4/analysis/e1/per_run.csv --criteria reports/ms/r4/prereg_criteria.json --out results/ms_r4/analysis/e1_blind --compare results/ms_r4/analysis/e1/criterion_e1.csv --check-gates-json
tools/ms/r4_blind_e2.py --per-run results/ms_r4/analysis/e2/per_run.csv --curves results/ms_r4/curves/curves_all.csv --criteria reports/ms/r4/prereg_criteria.json --out results/ms_r4/analysis/e2_blind --compare results/ms_r4/analysis/e2/criterion_e2.csv --analysis-info results/ms_r4/analysis/e2/analysis_info.json --check-gates-json
# tables of the reports
python reports/ms/r4/report_scripts/pilot_tables.py --list
python reports/ms/r4/report_scripts/pilot_tables.py --block <name> [key=value]
```

![paired differences of |peak error| (E1)](../../../results/ms_r4/analysis/e1/figures/paired_differences.png)

![eta_2/DW per arm (E2)](../../../results/ms_r4/analysis/e2/figures/eta_curves_log.png)
