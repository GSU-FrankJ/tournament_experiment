# MS-R2 pre-registration

Date: 2026-10-07. Branch `ms-r2` (from `origin/ms-r1` = `71c58904`). Prompt: `reports/ms/r2/pi_record/19_ms_r2_prompt.md`; `§` and `D#` refer to it. This file fixes, before any pilot run, the schedule, the arms, the identities, the measures, the criterion and the secondary analyses, the predictions, the offline calibration, the reference roots, the tests and the launch plan. Nothing in it changes after the launch except through a dated addendum. It is part of the **code commit** (the commit that adds this file together with the code, the tests and the reports 01-03; the decomposition and calibration outputs under `results/ms_r2/` follow in the next commit, results only, as the repository's commit rules separate code from results); the code commit's hash is the `--code-commit` argument of the launch and is recorded in the launch record and in `reports/ms/r2/04_pilot.md`.

Provenance: every number cites a path. Premise: `reports/ms/r2/01_decomposition.md` (reproduced from `results/ms_r1/analysis/per_run.csv`, the stop condition of §2.4 is not met). Offline calibration: `reports/ms/r2/01b_stop_candidates.md`. Host and roots: `reports/ms/r2/00_housekeeping.md`.

## 1. The question and the design

MS-R1 left the terminal-stage peak below the closed form in 139 of 140 runs. The PI's reading (preamble of the prompt, checked in `01_decomposition.md`): the d = 0 gap is the sum of a noise-smoothing part, which equals e*(0) sigma_2(0) / (sqrt(pi) q) to within 0.07 % in all 140 runs, and a remainder; the budget and the policy noise move the first, the samplers the second. This round lowers sigma_2(0) directly (a **noise landing**: the concentration scale of the Beta actor ramped at constant LR, held, then the LR decay) and crosses it with the start sampler, with the stop, the classification and polishing switched off.

## 2. The terminal-stage schedule (D2), as implemented

Every arm: terminal stage fixed 2800 updates, then stage 1 fixed 600 (LR linear 3e-4 -> 3e-5 over local 1-600, no rule, scale reset to 1.0 at the phase entry before the snapshot refresh).

| local updates | LR | concentration scale |
|---|---|---|
| 1-2000 | 3e-4 | 1 (never set) |
| 2001-2200 (ramp) | 3e-4 | before update j: 1 for j <= 2001, `1 + (s - 1) (j - 2001) / 199` for 2001 < j < 2200, s for j >= 2200 (the form of `run/run_v2_stagewise.py:Run.conc_scale_for`, first 2001, last 2200); first update with a scale different from 1: 2002 |
| 2201-2400 (hold) | 3e-4 | s |
| 2401-2800 (decay) | `lr_linear` 3e-4 -> 3e-5, first 2401, last 2800 (`pipeline.lr_windows["2"] = [{first 2401, last 2800, start 3e-4, end 3e-5}]`) | s |

Implementation (`run/run_ms_stagewise.py`; three new optional config keys, all absent = the MS-R1 behaviour bit for bit, proved by `tests/test_ms_r2_runner.py::test_without_the_new_keys_*`: a reduced legacy run and a reduced rule run in a checkout of `71c58904` end in the same state, weight exports, per-update series and check rows, with no new column):

- `conc_scale_schedule = {stage: 2, local_first: 2001, local_last: 2200, scale_first: 1.0, scale_last: s}` (absent for s = 1: no scale is ever set). The scale is set on the live actor and the lagged opponent before every update with local update > local_first, constant within the update (rollout and PPO update); the snapshot refresh copies it to the opponent (`agents/ppo_curriculum.py:refresh_snapshot`, unchanged); the frozen stage-2 snapshot (a deep copy at the freeze) keeps s; weight exports carry a `conc_scale` array when it differs from 1.
- `fixed_global_sampler = true` (with `rule.enabled = false` and `stratified_priority` starts): the legacy controller (`utils/ms_rule.py`, new constructor argument `StageController(..., legacy_global_sampler=True)`; not a `StageRule` field, so the logged rule parameters of MS-R1 are unchanged) returns the setting of a global block at every update: alpha = alpha_global, focus = the EMA of the per-bin residual map `rho_bar` (updated at every valid check, which is every K = 25 local updates), `p_PM` before the first check; no classification, no polishing; the block label stays `legacy`.
- `noise_report = true`: reporting only. `ms_updates.csv` gets `conc_scale` (the scale applied in that update); every check of the terminal stage gets `conc_scale, sigma_0, e_sigma_0, g2_0, smoothing, remainder, gap, R0` (`utils/ms_noise.py`; `e_sigma_0` with the Beta-quantile computation of `smoothed_share`, 400 midpoints per player; the learned tie effort of the decomposition is the verifier's `e2_at_0`, so that gap = e*(0) - e2_at_0 exactly as the closed-form pipeline reports it); the freeze record `rule_log.json` `stages["2"]["freeze"]["noise"]` has the same quantities, the ratio smoothing / [e*(0) sigma_2(0) / (sqrt(pi) q)], the scale and R0 on both tiers.
- The closed form (e*(0)) enters only the reporting columns; the schedule, the sampler and the stop thresholds read nothing of it.

Checks every K = 25 local updates are recorded in all arms (pure, no RNG, final tier only at the freeze); the would-fire record uses the thresholds of `reports/ms/r1/prereg_parameters.json` (passed with `--params`; SHA-256 `0fbfc01857c5abe339ab1926361cd57d6e988da20e754fd6dea3ffa4e076b406`: K = 25, M = 3, rho_2 = 0.05, rho_1 = 0.03, eps = 0.005, tau = 0.02, concentration limit 0.04, EMA weight 0.5, near-tie half-width 20) and decides nothing.

## 3. Arms (D3), each 20 runs: q in {50, 60} x seeds 10501-10510

| arm | terminal-stage starts | s | config keys beyond the common ones |
|---|---|---|---|
| `NL_bb_s1` | `bin_balanced` (the locked `StartSampler.balanced` call) | 1 | `noise_report` |
| `NL_bb_s4` | same | 4 | `noise_report`, `conc_scale_schedule` (scale_last 4) |
| `NL_bb_s16` | same | 16 | `noise_report`, `conc_scale_schedule` (scale_last 16) |
| `NL_st_s1` | `stratified_priority`: lambda_P 0.35, alpha_global 0.5, alpha_polish 0.5 (unused), EMA beta 0.5, near-tie half-width 20, lambda_T the bin-balanced tail share | 1 | `noise_report`, `fixed_global_sampler` |
| `NL_st_s4` | same | 4 | + `conc_scale_schedule` (4) |
| `NL_st_s16` | same | 16 | + `conc_scale_schedule` (16) |

Strata shares of the stratified arms (`derived` in `run_config.json`): q = 50: 40 bins, tail 20, near-tie 4, middle 16, lambda_T 0.5000, lambda_P 0.35, lambda_M 0.15; q = 60: 44 bins, tail 20, near-tie 4, middle 20, lambda_T 0.4545, lambda_P 0.35, lambda_M 0.1955. The six configs differ only in `start_weights` / `derived` and the two optional keys (`tests/test_ms_r2_launcher.py`); the stratified settings equal `MS_s35a5`'s (tested). Config builder `tools/ms/ms_configs.py` (`R2_ARM_TABLE`, kept apart from the MS-R1 arm table), wave `r2` of `tools/ms/launch_ms_r1.py` (default root `results/ms_r2`, output `results/ms_r2/pilot/q{q}/seed{s}/<arm>`), run names `msr2_q{q}_s{seed}_{arm}`.

## 4. Identities (D3; checked by `tools/ms/r2_launch_checks.py`; any failure is a stop-and-report)

"Equal" = the weight exports, the per-update series of `ms_updates.csv` (which holds the five stream positions after every update `rngpos_*`, the losses, KL and clip fraction, the start counts and the probability digest), the per-update history of `train_history.json`, and the columns common to both runs of the check rows of `ms_checks_stage2.csv`; wall-clock columns (`update_wall_sec`, `verifier_sec`) and the `run` / `arm` labels are excluded, and for C-MS4 also `block_id`, `block_type` and `mode` (the labels of the controller that differs by construction). A missing file is a failure, never a skip.

- **C-NL**: within each sampler and (q, seed), `NL_*_s4` and `NL_*_s16` equal `NL_*_s1` through update 2001 (the scale is still 1), and their `u02025` export differs: 2 samplers x 2 values of s x 20 = 80 comparisons.
- **C-MS3**: `NL_bb_s1` equals MS-R1's `MS_base2400` through update 2001 (both run LR 3e-4 up to 2001; `MS_base2400` decays from 2002) and `u02025` differs: 20 comparisons (the s = 4 and 16 runs equal `NL_bb_s1` through 2001 by C-NL).
- **C-MS4**: `NL_st_s1` equals MS-R1's `MS_s35a5` through update 2001 where that run's terminal stage had no polishing block, otherwise through the last update before its first polishing block (`first_local - 1` of the first block of type `polish` in its `rule_log.json`): 20 comparisons. The first export after that update is reported to differ (informational). Where the reference polishes, the check row at the block end before its first polishing block carries the polishing sampler's digest in `p_digest_next` (the next block's sampler, which `NL_st_s1` never has); that one column is not compared there (found by the independent review; `p_digest`, the probabilities in force at every update, is). MS-R1's `MS_s35a5` landing starts at update 2001 at LR 3e-4 (`lr_linear` returns `start` exactly at the first window update), with the sampler setting of the block it follows, which is the setting `NL_st_s1` has there.
- **Scale**: the applied scale (`conc_scale` of `ms_updates.csv`) equals the D2 table at every terminal-stage update (1 up to 2001, the ramp, s) and is 1.0 throughout stage 1; the recorded schedule (`run_config.json`, `manifest.json`) equals D2; every exported `conc_scale` equals the table at that update.
- The usual checks (`tools/ms/launch_checks.py`): status exit 0; manifest at the launch commit with `clean_tree`; files complete (rule record of both stages, check tables, bin maps, weight exports every 25 updates); process-global RNG assertions; the tail share lambda_T of every probability vector in force to 1e-12 and the measured start shares per stratum within 3 binomial standard errors per block (a few flags among the few thousand tests are expected by chance and are counted).

Tested on reduced budgets with a genuine reduced wave, tampered copies and the CLI: `tests/test_ms_r2_launch_checks.py`; the same relations on the runner: `tests/test_ms_r2_runner.py` (sections 3 and 4: the fixed-budget stratified path against MS-R1's rule mode with `rho = 0` and a localized fraction of 1e-9, so that stop and polishing are impossible; the bin-balanced path against `MS_base2400`'s window).

## 5. What is measured (D4)

At the terminal-stage freeze, on both tiers where applicable, per run: signed and absolute peak error; RMSE_pos; tail mean and max; eta_2 (final and development) with G-A and G-N(eta); sigma_2(0) (`sigma_effort_at_0_t2`); e_sigma(0) (`smoothed_e_pred_0`) and the decomposition gap = smoothing part + remainder, in effort units and relative to e*(0) (`g2_at_0`); the ratio smoothing / [e*(0) sigma_2(0) / (sqrt(pi) q)]; R0 and R (`t2_R0_*`, `t2_R_*`); the per-stratum errors of MS-R1 A3(b) (near-tie / middle / tail x d < 0 / d > 0); start shares per stratum; raw-draw clamp counts. Along the run, at every check from local update 1800 to 2800: e_hat_2(0), sigma_2(0), e_sigma(0), the decomposition, R0, R, Delta_2, C_2, the scale applied, and per segment (training 1-2000, ramp 2001-2200, hold 2201-2400, decay 2401-2800) the PPO diagnostics (KL, clip fraction, advantage SD, value loss) from `ms_updates.csv`. Stage 1, descriptive: G-S, S1, G-F, G-N(Gmax), the stage-1 error against `rehearsal_v2_0`, R_1. Gates reported as in MS-R1 D7: G-A and G-N(eta) on the final tier at the terminal freeze; G-F, G-N(Gmax), G-S, S1 at the end of stage 1; the v2.0 combination for information; no run-pass rule.

## 6. Criterion, comparisons, predictions (D5; descriptive, not gates)

**Primary** (`tools/ms/r2_analysis.py` -> `results/ms_r2/analysis/criterion.csv`, recomputed by the independent `tools/ms/r2_blind_criterion.py`): for each sampler (bb, st) and s in {4, 16}: |peak error| (`stage2_peak_rel_err_abs`) at the terminal freeze, paired by (q, seed) against the same sampler's s = 1 arm. (a) The 95 % percentile bootstrap interval (10,000 resamples, a fresh `numpy.random.default_rng(20261007)` per (q, statistic), indices `rng.integers(0, n, size=(10000, n))`, percentiles 2.5 / 97.5 of the resampled mean) of the mean paired difference lies below 0 at both q. (b) No run that passes G-A with its G-N(eta) part under s = 1 fails it under s. Four rows; the verdict per row is "met", "not met" or "incomplete", as in MS-R1.

**Secondary**, descriptive, same pairing and bootstrap: the paired changes of the smoothing part, the remainder, sigma_2(0), RMSE_pos, tail mean and eta_2 (`paired_secondary.csv`); the transmission ratio per arm and q: (change of the gap) / (change of the smoothing part), the ratio of the mean paired changes (changes = arm - the same sampler's s = 1 arm) with a percentile bootstrap interval of the ratio (`transmission.csv`; 1 = the whole smoothing reduction reaches the gap, 0 = the remainder offsets it; equivalently reduction of the gap / reduction of the smoothing part; see section 9, item 10); the interaction (`NL_st_s` - `NL_st_s1`) - (`NL_bb_s` - `NL_bb_s1`) per (q, seed) on |peak| and on the remainder (`interaction.csv`); every arm against `parents_A` with MS-R1's criterion (`criterion_vs_parents_A.csv`); the s = 1 arms against their MS-R1 references, `NL_bb_s1` - `MS_base2400` and `NL_st_s1` - `MS_s35a5` (`paired_vs_ms_r1_refs.csv`: the effect of 400 more constant-LR updates and of the extended schedule).

**Predictions, written before the launch.**
1. sigma_2(0) at the freeze equals sigma_2(0) at s = 1 divided by sqrt(s), up to the change of the mean (the ratio sigma(s) / [sigma(1) / sqrt(s)] is reported per arm and q, `predictions.csv`). At the planning value sigma_2(0) ~ 2.5 (MS-R1: 2.50-2.99) this gives sigma_2(0) ~ 1.25 (s = 4) and ~ 0.625 (s = 16).
2. The smoothing part equals e*(0) sigma_2(0) / (sqrt(pi) q): the ratio is reported per arm and q, expected within 0.5 % of 1 (MS-R1: 0.9993-0.9997).
3. The floor, the smoothing part in percent of e*(0), is therefore about 1.4 % / 0.7 % at q = 50 and 1.2 % / 0.6 % at q = 60 for s = 4 / 16 (2.8 % at q = 50 and 2.4 % at q = 60 for s = 1).
4. The open question is the remainder. If it does not grow when sigma falls, |peak| falls by about the smoothing reduction (transmission ratio near 1); R1's `A_anneal4` (bin-balanced starts, the ramp inside the LR decay) suggests it may grow under bin-balanced starts, which would give a transmission ratio below 1; whether tip weighting (the stratified arms) lets the mean follow the sharper target is the interaction. No prediction is made about the sign or size of the interaction.
No selection rule follows from the round; the decision is the PI's.

## 7. Offline calibration of stop criteria (D6, report-only)

`tools/ms/r2_stop_candidates.py`; results `results/ms_r2/stop_calibration/`; report `reports/ms/r2/01b_stop_candidates.md`. Candidates C1 = R0, C2 = |c2| with the noise floor (and the signed c2 as a variant), C3 = R_defl; the exact one-step best response is the root of the strictly decreasing derivative of the terminal objective (bisection to 1e-10), checked against a dense grid with parabolic refinement (3.6e-10), which is the accurate solution of the problem D6 states. Threshold grids `reports/ms/r2/stop_candidate_grids.json` were committed (`93d8be5d`) before any fire table was computed. The pilot repeats it on the 120 MS-R2 runs (§3.3).

## 8. Reference roots (D8; every tool takes them as explicit arguments and records them)

| name | path |
|---|---|
| MS-R1 pilot (`MS_base2400`, `MS_s35a5`, other arms for D6) | `/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857/results/ms_r1/pilot` (this worktree) |
| MS-R1 base wave (`MS_base`) | `/home/fjiang4/tournament_experiment/.claude/worktrees/ms-r1-multistage-development-afebf2/results/ms_r1/base` |
| `parents_A` | `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine/parents_A` |
| `rehearsal_v2_0` | `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0` |
| canonical worktree for C7 | `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2` |

## 9. Choices where the prompt is silent (fixed here, before the launch)

1. The stratified arms are expressed as a flag of the legacy (fixed-budget) controller (`fixed_global_sampler`), not through the rule controller; C-MS4 is the proof of the equivalence.
2. The scale is applied from local update 2002 (`local > local_first`), and the key is absent for s = 1 (no scale is ever set).
3. `noise_report` is a separate optional key so that the MS-R1 configs keep their columns exactly; it is true in all six arms (the s = 1 arms record `conc_scale` = 1.0 per update).
4. The decomposition uses the verifier's `e2_at_0` and `g2_at_0` (the closed-form pipeline's values), and `e_sigma(0)` from the Beta parameters of the candidate at d = 0 (single-point evaluation, as `smoothed_share`).
5. C-MS3 is evaluated for `NL_bb_s1` (20 comparisons); the s = 4 and s = 16 runs equal `NL_bb_s1` through 2001 by C-NL.
6. For C-MS4 the compared columns exclude the controller labels (`block_id`, `block_type`, `mode`).
7. C3 uses the exact best response (bisection) rather than a dense grid; C2 uses |c2| in the fire rule (signed c2 as a variant), because a signed residual would fire on an overshoot.
8. Segment boundaries of the analysis: training = local 1-2000 (the scale is 1 through 2001), ramp = 2001-2200, hold = 2201-2400, decay = 2401-2800 (D2's table; the ramp segment includes update 2001 at scale 1).
9. Stage 1 is bit-identical in structure to MS-R1's `MS_base`; its comparison with `rehearsal_v2_0` is descriptive.
10. **The sign of the transmission ratio.** The prompt writes "-(change of the gap) / (change of the smoothing part)" and defines 1 as "the whole smoothing reduction reaches the gap" and 0 as "the remainder offsets it". With changes defined as arm minus the same sampler's s = 1 arm both changes are negative when the smoothing reduction reaches the gap, so the literal formula gives -1 for full transmission and 0 only when the gap does not change: its sign contradicts the stated endpoints. The ratio without the minus sign, (change of the gap) / (change of the smoothing part), has the stated endpoints (1 and 0) and is what `r2_analysis.py` computes (`analysis_info.json` and the module docstring say so); it equals 1 + (change of the remainder) / (change of the smoothing part).
11. The seed 20261007 is used for every bootstrap interval of the MS-R2 tools, also in the descriptive tables (MS-R1's tools used 20261006).
12. `state_end_stage2.pt` stores the concentration scale (`agent.conc_scale`), as do the weight exports, `rule_log.json` and `manifest.json`.
13. The would-fire record of the arms with s > 1 is not comparable across s: the concentration statistic C_2 scales as 1/sqrt(s), so eligibility is easier there. It decides nothing.

## 10. Tests and review (filled at the code commit)

Full suite `pytest tests` on the tree of the code commit (three parallel invocations over a partition of all 1063 collected tests): 1060 passed, 1 failed, 2 xfailed; the failure is the known `test_registry_canonicalization`; all 365 MS-R1 tests pass unchanged, the 128 new tests (`tests/test_ms_r2_{runner,launch_checks,launcher,decomposition,stop_candidates,analysis}.py`: 26, 9, 7, 10, 23, 53) pass. Independent read-only review: highest severity MAJOR (C-MS4 `p_digest_next`), fixed before this commit; the dispositions are in `reports/ms/r2/03_checks.md` section 2. The same file has the measured maxima asked for in section 2.2 of the prompt.

## 11. Launch plan

After the pre-registration commit is pushed (`git push origin ms-r2:ms-r2`, new branch, no force) and if every test, check and the review are clean, from a clean tree at its head, in tmux, root `results/ms_r2`:

    python tools/ms/launch_ms_r1.py --wave r2 --params reports/ms/r1/prereg_parameters.json --workers 40 --code-commit <code commit>

(120 runs: six arms x 20; one launch record `results/ms_r2/pilot/launch_<stamp>.json` with nproc, load, free disk, HEAD, `git diff --stat <code commit> HEAD`, the same limited to `run utils envs agents protocols`, `git status --porcelain` and the parameter file's SHA-256; at most 40 single-threaded workers; crash rule: an infrastructure kill is re-run once after moving the first attempt to `results/ms_r2/pilot/crashed/`, any other non-zero exit counts as a failed run and is reported). Nothing is edited or committed in the worktree while the launcher is still starting runs. Post-launch checks: `python tools/ms/r2_launch_checks.py --root results/ms_r2/pilot --ms-r1-pilot-root <MS-R1 pilot root> --code-commit <HEAD at launch> --out results/ms_r2/pilot/launch_checks.json`. Analysis: `python tools/ms/r2_analysis.py ...` then `python tools/ms/r2_blind_criterion.py --analysis-dir results/ms_r2/analysis`; D6 on the 120 runs with `python tools/ms/r2_stop_candidates.py replay ...`. Wall estimate: 3400 updates per run (2800 + 600) at 0.18 s per update on this shared host, about 11 minutes per run, three waves of 40 workers (MS-R1: 3000 updates in 503-654 s).
