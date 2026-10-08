# MS-R1 pre-registration

Date: 2026-10-07. Branch `ms-r1` (from `origin/main` = `4dc604de`, the head of the `t2-refine-100526` publication). This file fixes, before any pilot run, the pipeline, the metrics, every rule and sampler parameter with the calibration table that supports it, the arms, the criterion and the comparators, the gates that are reported, the reference roots, the two reproduction checks and the launch plan. The prompt is `reports/ms/r1/pi_record/17_ms_r1_prompt.md`; `§` and `D#` refer to it. Nothing in this file changes after the launch except through a dated addendum (§ 2.6: the PI's reply at gate G1).

Provenance: every number cites a path. Code: commit `c95a2af4` and its ancestors back to `170b7d6d` (section 9); the two reproduction checks ran at `c95a2af4`. Parameter file: `reports/ms/r1/prereg_parameters.json` (SHA-256 recorded in every launch record).

## 1. Pipeline structure (D2) and what differs from v2.0

Entry point `run/run_ms_stagewise.py`; nested tables `utils/ms_continuation.py`; stage-t rollout `run/ms_rollout.py`; rule `utils/ms_rule.py`; residual metrics `utils/ms_residual.py`; sampler additions in `envs/curriculum_env.py`. Config schema `ms_run_config/1` (`run/run_ms_stagewise.py:validate_config`: sections `pipeline`, `start_weights`, `rule`, the embedded v2.0 record, the gate thresholds of the protocol, `derived`; unknown or missing keys are refused).

| | v2.0 (`run/run_v2_T2_locked.py`, locked, untouched) | MS-R1 (`run/run_ms_stagewise.py`) |
|---|---|---|
| structure | Phase A: stage 2 alone, 1600 updates; freeze; Phase B: stage 1, 600 updates, whose batch still holds the 512 stage-2 rows (critic-fitted, masked out of the actor loss) | phases t = T, ..., 1, one stage per phase; in the phase of stage t the batch holds stage-t rows only (512 episodes per update) |
| budget | fixed 1600 / 600 (the original stop rule is disabled by `fixed_budget`) | the rule of D4 per stage (blocks, development stop, landing) or, for `MS_base`, the fixed 1600 / 600 with v2.0's LR windows |
| terminal stage rollout | `collect_batch_v2`, `reward_mode=expected` | the same function with the same arguments (`t0 = T` for every row): the streams are consumed exactly as in v2.0 |
| stage t < T rollout | stage 1 with the table value of the frozen stage 2; the stage-2 rows are rolled and critic-fitted | `ms_rollout.collect_stage_batch`: learner action from the live actor, opponent action from the lagged copy (refreshed at phase entry and every 20 global updates), return `-k e^2 + V~_{t+1}(d_t + e_t - e_t^opp)` (linear lookup in the table), advantage = return - V(s_t), critic target = the return; no shock is drawn (the table integrates it), so the `env` stream is untouched; roles are still drawn after the starts |
| continuation | `utils/v2_continuation.py`, T = 2 only | `utils/ms_continuation.py`, nested for any frozen suffix; at T = 2 the table of stage 1 equals v2.0's bit for bit (`tests/test_ms_runner.py::test_the_stage_1_table_equals_the_v20_builder_and_is_built_without_rng`); built once at the freeze of the stage above, pure (asserted: no training stream and no global RNG moves), written to `continuation_table_stage{t}.npz` |
| starts | bin-balanced (`StartSampler.balanced`) | bin-balanced for `MS_base` (the locked call, unchanged); `stratified_priority` (D5) for the rule arms; root (d = 0) at stage 1 |
| checks | every 100 updates (Phase A) | every K local updates and at every block end: one `utils.v2_metrics.evaluate` on the development tier (pure, no RNG); final tier only at the freeze |
| freeze | deep copy of the actor after Phase A | after the landing window of each stage: `state_end_stage{t}.pt`, final- and development-tier evaluation (`freeze_stage{t}_{final,development}.npz`), a deep-copied snapshot (no grad, eval mode, in no optimizer), the table of the next phase |

Kept from v2.0 (read from the embedded record, `protocols/v2_T2_locked_v2_0.json` `records[q]`, SHA-256 in each manifest): the game, the 2 -> 64 -> 64 tanh actor and critic, PPO settings (LR 3e-4, 10 epochs, minibatch 256, clip 0.2, gradient clip 0.5, entropy 0, gamma = lambda = 1, Adam state preserved across phases), 512 episodes per update, the five RNG streams and the torch generator (`SeedSequence([seed, q, namespace])`), the process-global RNG hardening (seeded with the run seed; reference state before the first update; digests asserted at the end of every stage, after every freeze evaluation, around every table build and at the end of the run; a violation is recorded in `gates.json` and the run exits with code 5 and counts as failed), ES bin width 10, `clamp_likelihood = density` (any other value is refused), weight exports every 25 global updates, the verifier tiers, every metric of `utils.v2_metrics`, concentration threshold 0.04.

Legacy identity (check C-MS1, section 8): with `rule.enabled = false`, `start_weights = {scheme: bin_balanced}`, the fixed budget 1600 and the window `1201-1600` (3e-4 -> 3e-5), the terminal-stage phase ends in the training-relevant state of `parents_A`. The stage-1 phase is not bit-identical to v2.0's Phase B (the batch differs, see the table); its end-of-phase metrics are compared with `rehearsal_v2_0` descriptively.

Shared modules gained code only behind new names: `envs/curriculum_env.py` (new `StartSampler` methods, the existing ones byte-identical), nothing else. Check C-R4 (section 8) is the proof for the unchanged v2.0 path.

## 2. Metrics (D3), as implemented (`utils/ms_residual.py`)

On the composite candidate (live actor at the stage being trained, frozen snapshots above it, the live actor's untrained output below it) one development-tier `verify` call gives per node d of the stage-t grid the one-step deviation gain `delta`, the policy mean `e_hat` and the one-step best-response effort `a_dev` (argmax of the one-step deviation search, policy action and parabola-vertex candidates included). With `thr = 2q (T - t + 1)`:

- `Delta_t = full_delta_max[t] / dw`; `r_t(d) = |e_hat_t(d) - a_dev_t(d)|`; `s_t = a_dev_t(0)` (the exact zero node of the grid);
- `R_t = max { r_t(d) / s_t : |d| < thr }` (nodes within 1e-9 of `thr` belong to the tail side); `R_t^tail = mean { e_hat_t(d) : |d| >= thr } / s_t`, void at t = 1 and in a stage without a tail bin (stage 2 of T = 3);
- per-bin map `rho_b = max { r_t(d)/s_t }` over the non-tail-region nodes assigned to the non-tail bin b by `gap_bin_index` (a node of the tail region is in no bin map, also when `gap_bin_index` puts it on the edge of a non-tail bin); EMA `rho_bar <- beta rho_bar + (1 - beta) rho_b`, initialised at the first check;
- `C_t = concentration_stats(...)["max_std_norm"]` on the stage-t development grid.

A check is `valid` iff the verifier result is valid, the concentration statistics are finite and `s_t > 0`; an invalid check, a NaN, or a failed verifier call is never eligible. The closed-form quantities (peak error, RMSE_pos, tail mean, stage-1 error) are computed at every check for reporting only and are never read by the rule or the sampler: `tests/test_ms_runner.py::test_decisions_and_sampler_do_not_depend_on_the_closed_form` replaces `recovery_metrics` by a stub and finds identical check columns, blocks, sampler digests and training state.

## 3. Rule and sampler parameters

The parameters are in `reports/ms/r1/prereg_parameters.json` (SHA-256 `0fbfc01857c5abe339ab1926361cd57d6e988da20e754fd6dea3ffa4e076b406`), passed to the launcher with `--params`; the code does not change when a value does. The defaults of D4 and D5 apply except where a row says **changed**. Tables are in `reports/ms/r1/01_calibration.md`.

| parameter | terminal stage (t = 2) | stage 1 | default (D4/D5) | calibration support |
|---|---|---|---|---|
| cadence K (local updates) | 25 | 25 | 25 | equals the export cadence of the replay; no table argues for another value |
| consecutive eligible checks M | 3 | 3 | 3 | unchanged |
| epsilon_t (Delta_t) | 0.005 | 0.005 | 0.005 | fixed by D8; passes at 95 % of the Phase-A exports and at essentially all stage-1 exports (Table 2b, Section 5) |
| **rho_t (R_t)** | **0.05 (changed)** | 0.03 | 0.03 | see below |
| tau_t (R_t^tail) | 0.02 | void | 0.02 | passes at 90 % of the exports, median at u1600 0.0097, maximum 0.0179 (Tables 0, 2b); the G-A limit |
| concentration limit C_t | 0.04 | 0.04 | 0.04 | 60 of 60 runs at or below 0.04 at u1600 (Table 0) |
| block length N_block | 400 | 200 | 400 / 200 | the classification tables are at the block ends 400, 800, 1200, 1600 (Table 4) |
| training cap U_cap | 2000 | 600 | 2000 / 600 | beyond the last Phase-A export, not calibratable offline |
| landing length N_land | 400 | 400 | 400 / 400 | v2.0's 400-update decay improves RMSE_pos 0.0283 -> 0.0229 (Table 4 landing) |
| localized fraction | 0.25 (5 of 20 bins at q = 50, 6 of 24 at q = 60) | n/a | 0.25 | Table 4d |
| alpha_polish | 0.5 | n/a | 0.5 | cannot be calibrated offline |
| EMA weight beta | 0.5 | n/a | 0.5 | halves the check-to-check change of the per-bin map (median 0.0062 against 0.0140 per 25 updates; Table 4c) |
| near-tie half-width | 20 | n/a | 20 (R2b) | |
| alpha_global | per arm (0 or 0.5) | n/a | per arm | D6 |
| lambda_P | per arm | n/a | per arm | D6 |
| lambda_T | n_tail / n_bins: 0.5000 (q = 50), 0.4545 (q = 60) | n/a | the bin-balanced tail share | D5, derived, never configurable |

**The one departure: rho_2 = 0.05 instead of 0.03.** Reasons, all from the calibration tables: (i) with rho in {0.02, 0.03} the stop of the terminal stage fires in 0 of 60 v2.0 trajectories (1 of 60 at 0.05; Table 2) because R is the binding eligibility component (R <= 0.03 holds at 0 of 2940 Phase-A exports, Table 2b); (ii) with rho = 0.03 the localized branch (targeted polishing, the PI's rule) is classified in 1 of 60 runs at u400 and 0 of 60 at u800 and u1200 (11 of 60 at u1600), and with rho = 0.02 never, i.e. it is practically unreachable at the calibrated operating point, while at rho = 0.05 it is reached in 5, 25, 27 and 37 of 60 runs at the four block ends (Table 4, 4d); (iii) the first-order residual is an amplified error measure on the d < 0 side (Section 4.1 of the calibration report: factor 3.33 at q = 50, 1.95 at q = 60), so R <= 0.03 corresponds to a uniform policy error of about 0.6 (q = 50) and 0.9 (q = 60) effort units, of the order of the development-tier best-response resolution (Section 3); R <= 0.05 to about 1.0 and 1.5 units. rho = 0.05 is a value of the prompt's own grid. **Consequence to expect:** on trajectories like v2.0's the stop will rarely fire (1 of 60 in the replay), so most terminal-stage runs are expected to end by the cap (`budget_forced = true`, 2000 training updates plus the 400-update landing); the stop is then effectively a budget, and the pilot compares arms at that budget. A rho at which the stop fires at v2.0's quality is about 0.06 to 0.08 (Table 2c); it is not pre-registered. Stage 1 keeps 0.03 (60 of 60 trajectories fire, at |error| 0.0122 median and 0.0322 maximum, Table 5c).

Reading of the D4 text that the code implements (stated here so that nothing is decided after the pilot): the classification of a block end uses the last check's R and the EMA map as logged, including when that check is invalid (reachable only if the verifier flags the result invalid or the concentration is non-finite; the EMA is updated by valid checks only); the landing window's frozen sampler setting is the followed block's setting at the end of that block (its focus is `rho_bar` after the stopping or capping check); the per-bin map assigns a node to the bin given by `gap_bin_index`, which at nodes lying exactly on a bin edge can differ from the sampler's half-open convention by float rounding (q = 60: the node d = 80 goes to bin 29, the sampler's convention gives bin 30; one node of one bin); the verifier's `a_dev` candidate set also contains the dynamic-BR action `a_br`, which at t = T is the same maximiser and at t < T can be returned when it ties or beats the grid and vertex candidates.

## 4. Choices where the prompt is silent (fixed here, before the pilot)

1. The streak of consecutive eligible checks is not reset by a block boundary; one ineligible check resets it. The first check is at local update K.
2. Landing: checks continue at local updates that are multiples of K (reported, deciding nothing); the per-bin EMA keeps being updated for reporting; the sampler setting of the landing window is the setting of the block it follows with the focus vector frozen at the first landing update.
3. A polishing block's set S_t is fixed at the block start (the classification of the previous block end); its focus is `rho_bar` restricted to S_t at every update (`rho_bar` keeps evolving); before the first check, or when `rho_bar` is all zero, the focus is `p_PM`.
4. Block lengths: a block is `min(N_block, U_cap - updates so far)` updates; a stop or a cap at a block end is one event (the stop wins).
5. At stage 1 (no sampler, no bins) the classification of a block end without a stop is recorded as `degenerate`, the next block is global; the rule reduces to stop / continue / forced landing.
6. `MS_rule` takes lambda_P = n_near / n_bins (4/40 at q = 50, 4/44 at q = 60), so its bin probabilities are 1/n_bins up to rounding (the draw mechanics of D5, not the locked `balanced` call).
7. The stage-t rollout for t < T draws nothing from the `env` stream (the shock is integrated in the table); roles are drawn after the starts as in v2.0.
8. The eligibility thresholds are recorded in the `MS_base` config as well (the `would_fire_local` of the legacy arm uses them); `MS_base` decides nothing from its checks.
9. The tail-share coverage constraint: `lambda_T = n_tail / n_bins` is derived, written to `run_config.json` and `manifest.json` (`derived`) and re-derived by the runner; a `lambda_T` key in the config is accepted for T = 2 only and must equal the derived value. At T = 3 the terminal stage has 60 of 80 tail bins (q = 50), so lambda_P < 0.25 there; the T = 3 code is exercised in tests only.
10. The start counts per stratum (tail / near-tie / middle) are logged for every update in `ms_updates.csv` (`n_start_*`), and the bin probabilities in force in `ms_binmaps_stage{t}.npz` (`probs_first_update`, `probs_table`), also for the bin-balanced arm (strata with half-width 20).
11. 'All D3 quantities' of a check row: `ms_checks_stage{t}.csv` carries the scalars (Delta, s, R, R_tail, tail_term, C, the argmax node of R, eligibility, streak, would-fire, the sampler digests, the closed-form reporting columns); the per-bin map `rho_b` and its EMA `rho_bar_b` of every check are in `ms_binmaps_stage{t}.npz` (aligned by `update`, with the bin probabilities in force), and `r_t(d)` per node at the freeze in `freeze_stage{t}_{final,development}.npz` (`v_t{t}_e_hat`, `v_t{t}_a_dev`).
12. `rule_log.json`: in a block that ends by `block_end` or `cap` the field `S` is the set S_t of that block's own classification (which is the focus set of the next, polishing, block); a block that ends by `development_stop` keeps the focus set it trained on; the focus set of a polishing block is therefore the `S` of its predecessor. `S` holds bin indices of D_t. The log is written at every freeze; `gates.json` once, at the end of the run (key names differ from v2.0: `reported.end_of_stage2` / `end_of_stage1`, `continuation_tables`).
13. A development stop at the check at which the cap is reached is a stop (`budget_forced` false).
14. Two constants are module constants of the runner rather than config keys: the landing LR end 3e-5 (recorded in `rule_log.json` `params.lr_end`) and the half-width 20 of the strata used to measure start shares in the bin-balanced arm.
15. `ms_updates.csv`, `train_history.json` and `ms_run_summary.json` are written when the run completes; the check CSVs, bin maps and `rule_log.json` are written as the run goes. The smoothed-game share of the d = 0 gap is `run.run_v2_T2_locked.smoothed_share` (as in R2b's `gates.json`); `tools/v2/decomposition.py` is the stage-1 error decomposition, reported through `induced_band.json`.

## 5. Arms, comparators and criterion (D6)

Six arms, each 20 runs (q in {50, 60} x seeds 10501-10510), the full pipeline (terminal stage, then stage 1) in one process (`tools/ms/ms_configs.py:ARMS`; `tools/ms/launch_ms_r1.py` waves `base` and `pilot`):

| arm | sampler in global blocks | rule | notes |
|---|---|---|---|
| `MS_base` | bin_balanced (the locked `StartSampler.balanced`) | disabled; fixed 1600 / 600 with v2.0's LR windows (1201-1600 and 1-600, 3e-4 -> 3e-5); checks every 25 reported, would-fire recorded | regression arm; its terminal-stage end state must equal `parents_A` (C-MS1) |
| `MS_rule` | `stratified_priority`, lambda_P = n_near / n_bins (4/40, 4/44), alpha_global 0 | D4 with polishing (alpha_polish 0.5) | stop rule and polishing alone; bin-balanced in distribution |
| `MS_s25a0` | `stratified_priority`, lambda_P 0.25, alpha_global 0 | D4 | |
| `MS_s25a5` | `stratified_priority`, lambda_P 0.25, alpha_global 0.5 | D4 | |
| `MS_s35a0` | `stratified_priority`, lambda_P 0.35, alpha_global 0 | D4 | |
| `MS_s35a5` | `stratified_priority`, lambda_P 0.35, alpha_global 0.5 | D4 | |

The `MS_base` arm of the pilot is the C-MS1 wave (`results/ms_r1/base`, 20 runs); it is not run again in P2, which runs the five rule arms. The five rule arms differ from `MS_rule` only in `start_weights` (`tests/test_ms_launcher.py::test_the_five_pilot_arms_differ_from_ms_rule_only_in_start_weights`). Strata at q = 50 (q = 60): 40 (44) bins, tail 20 (20), near-tie 4 (4), middle 16 (20); lambda_M = 1 - lambda_P - lambda_T is positive in every arm (from 0.15 at q = 50 with lambda_P = 0.35 up to 0.4545 at q = 60 for `MS_rule`; the values are in `derived` of each `run_config.json`). The terminal stage of T = 3 (60 of 80 tail bins) refuses lambda_P >= 0.25: the arms are T = 2 experiments.

**Comparators**, paired by (q, seed): the terminal-stage candidate against `parents_A` (v2.0's end of Phase A, identical to `rehearsal_v2_0`'s); the stage-1 candidate against `rehearsal_v2_0`'s end of Phase B. Every rule arm is also reported against `MS_rule` (the sampler's effect net of the rule).

**Criterion** (descriptive, not a gate): primary metric |peak error| (`stage2_peak_rel_err_abs`) of the frozen terminal-stage candidate; (a) the 95 % percentile bootstrap interval (10,000 resamples, `numpy.random.default_rng(20261006)`, one fresh generator per (q, statistic) in table order) of the mean paired difference arm - `parents_A` lies below 0 at **both** q; (b) no run that passes G-A with its G-N part (eta) under `parents_A` fails it under the arm; the `parents_A` verdicts are taken from its `final_v2.json` (final tier: eta_2/DW <= 0.005, RMSE_pos <= 0.05, tail mean <= 0.02, and |eta_dev - eta_final| <= 0.001 from its two tiers). No selection rule and no protocol change follow; the decision is the PI's. Everything else in the analysis is descriptive, and an interval that contains 0 with ten seeds does not show that a mechanism has no effect.

**Reported per arm and q** (paired where a comparator exists): the signed peak error with its interval (whether it contains 0), runs with |peak error| <= 0.05, RMSE_pos, tail mean and max with the G-A verdicts, eta_2 on both tiers with G-N, sigma_2(0) and the smoothed-game share of the d = 0 gap, R_t, R_t^tail and Delta_t at the freeze on both tiers, the budget to the freeze (updates, episodes, optimiser steps, wall time), the rule record (fire update or `budget_forced`, number and types of blocks, localized / broad classifications, |S_t|), the measured start shares per stratum and the raw-draw clamp counts (the D1 columns), and for stage 1: the G-S error, r_1(0)/s_1, the induced-target decomposition (learning against inherited), G-F, the stage-1 budget and rule record. Tool `tools/ms/r1_analysis.py` (section 9); an independent recomputation of `criterion.csv` by `tools/ms/blind_criterion.py`.

## 6. Gates reported (D7)

At the terminal-stage freeze, final tier: G-A (eta_2/DW <= 0.005, RMSE_pos/e2*(0) <= 0.05, tail mean/e2*(0) <= 0.02) and the eta part of G-N (|dev - final| <= 0.001). At the end of the stage-1 phase: G-F (Gmax_full/DW <= 0.01), the Gmax part of G-N, G-S (|e_hat_1(0) - e1*(0)|/e1*(0) <= 0.05), S1 at 0.10. `gates.json` (`run/run_ms_stagewise.py:build_gates`) carries every value and verdict (thresholds from the embedded protocol gates), the v2.0 combination G-A and G-F and G-N and G-S for information (`v20_combination_pass`), the global-RNG record and the continuation-table records, the D3 quantities at both freezes on both tiers, the budgets and the rule record. There is no run-pass rule in this round.

## 7. Reference roots (D9)

| name | path |
|---|---|
| `parents_A` | `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine/parents_A/q{50,60}/seed{10501..10510}` |
| `rehearsal_v2_0`, `confirmation_v2_0` | `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/{rehearsal_v2_0,confirmation_v2_0}/q*/seed*` |
| canonical worktree for C7 | `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2` |

Every tool takes its roots as explicit arguments and records them; a missing reference file is a stop-and-report.

## 8. Reproduction checks and the test-suite summary

Details, commands and outputs are in `reports/ms/r1/03_checks.md`.

- **C-R4** (the unchanged `run/run_v2_T2_locked.py` from this branch, q in {50, 60} x seeds 10501-10510, into `results/ms_r1/v20_reproduction/`, compared with `rehearsal_v2_0` by `tools/v2/cr2_compare.py`): **identical 20 of 20** (48 fields per run, including the continuation table bit for bit), `results/ms_r1/v20_reproduction_checks.json`.
- **C-MS1** (the 20 `MS_base` runs into `results/ms_r1/base/q*/seed*/MS_base/`, `tools/ms/cms1_compare.py`): the terminal-stage end state `state_end_stage2.pt` (actor, critic, lagged opponent, both Adam states, the five streams, the torch generator, the counters), all 64 weight exports, the per-update series and the end-of-phase evaluation arrays on both tiers equal `parents_A`'s end of Phase A in **20 of 20**, and equal `rehearsal_v2_0`'s in 20 of 20 (`results/ms_r1/base_checks_parents_A.json`, `base_checks_rehearsal_v2_0.json`). The `MS_base` wave was launched without `--params` (its would-fire record carries rho_2 = 0.03; recomputed at the pre-registered thresholds in `03_checks.md`); its training is unaffected.
- **Tests:** the pre-existing suites `1 failed, 567 passed, 2 xfailed` (the known `test_registry_canonicalization` failure), the 314 MS tests all pass, full suite: `1 failed, 881 passed, 2 xfailed, 2 warnings in 1699 s` (2026-10-07 03:00-03:29, `pytest tests`): the failure is the known `test_registry_canonicalization`; 881 = 567 pre-existing + 314 MS tests.
- **Independent review** (conformance, RNG / numerical identity, state machine and sampler fuzzing): no blocker or major finding; two minor findings fixed (final global-RNG assertion after the decomposition; outcome label of an exit-5 run), the others documented in sections 3 and 4 (`03_checks.md` section 4).

## 9. Launch plan

Commits (`git log --oneline 4dc604de..`): code `170b7d6d` (sampler), `c856cb9b` (nested tables), `684479aa` (residual metrics and rule), `411ea438` (runner), `4eb8f185` (launcher, config builder, C-MS1 comparison, launch checks), `db5e5d3c` (calibration replay and analysis tools), tests `01da192e`; results `24862b57` (calibration outputs); reports `c95a2af4`; the later commits of this round (the check results, this file, the report folder) change only `reports/`, `results/` and `docs/`. The pilot is launched from the head of `ms-r1` at that time, a clean tree whose code is that of `c95a2af4`: the launch record carries `git diff --stat c95a2af4 HEAD` (expected: `reports/`, `results/`, `docs/` only) and every manifest must be at the launch commit with `dirty: false`. Nothing under `run/`, `utils/`, `envs/`, `agents/`, `tools/`, `tests/` or `protocols/` changes after `c95a2af4`.

1. After the PI's "proceed P2" (and any addendum to this file, dated, no edits): from a clean tree at the head of `ms-r1`, in tmux, with the explicit root `results/ms_r1`:
   `python tools/ms/launch_ms_r1.py --wave pilot --params reports/ms/r1/prereg_parameters.json --workers 40 --code-commit c95a2af4` (100 runs, one launch record `results/ms_r1/pilot/launch_<stamp>.json` with nproc, load, free disk, HEAD, `git diff --stat c95a2af4 HEAD`, `git status --porcelain`, the parameter file's SHA-256). Nothing is started before the go, not even a dry run that writes into `results/ms_r1/pilot/`.
2. Crash rule as in v2.0: an infrastructure kill is re-run once into the original path after moving the first attempt to `results/ms_r1/pilot/crashed/`; any other non-zero exit (pipeline exception, global-RNG violation, verifier error) counts as a failed run and is reported, not re-run.
3. Post-launch checks: `python tools/ms/launch_checks.py --root results/ms_r1/pilot --arms MS_rule MS_s25a0 MS_s25a5 MS_s35a0 MS_s35a5 --code-commit <HEAD at launch, the `head` of the launch record> --out results/ms_r1/pilot/launch_checks.json` (manifests at the launch commit and clean; rule record for both stages, check tables, bin maps, weight exports; global-RNG assertions; the tail share lambda_T of every probability vector in force to 1e-12 and the measured start shares per stratum within 3 binomial standard errors per block and per landing window, with the polishing blocks compared with their own focus; at 3 standard errors a few flags are expected by chance among the several thousand tests, and the summary says how many).
4. Analysis: `python tools/ms/r1_analysis.py --base-root results/ms_r1/base --pilot-root results/ms_r1/pilot --parents-root <parents_A> --rehearsal-root <rehearsal_v2_0> --calibration-root results/ms_r1/calibration --out results/ms_r1/analysis` (the calibration root fills the D3 quantities of the two comparators), then `python tools/ms/blind_criterion.py --analysis-dir results/ms_r1/analysis` (independent recomputation of `criterion.csv`, agreement to 1e-12).
5. Wall estimate: about 6-9 minutes per run at 0.12 s per update, 100 runs on at most 40 workers.
6. Push: `ms-r1` is pushed after this file is committed (so that the PI can read it) and again after the reports; `main` and tags are not touched.


---

# Addendum 1 (2026-10-07): the PI's decision at gate G1, the arm `MS_base2400`, the expectations, the analysis additions

Everything above this line is unchanged (no existing line was edited). Source: the PI's reply at gate G1, verbatim in `reports/ms/r1/pi_record/18_g1_reply.md`; `A1`-`A3` below are its items. This addendum is committed and pushed before any pilot run (the pilot directory `results/ms_r1/pilot/` does not exist when it is written). The numbers quoted in A2 were checked against `reports/ms/r1/01_calibration.md` (Tables 1, 2, 4) and `reports/t2_refine_100526/100526report.md` (the R2b control row) before they were adopted: they agree.

## A.1 Decision at G1

- P1 is accepted, including the three recorded deviations (the work tree is the session worktree, not `.claude/worktrees/ms-r1`; the primary checkout's `main` was not fast-forwarded, which is owner-side and not needed for this round; the `MS_base` wave was launched without `--params`, its would-fire entry recomputed in `03_checks.md`).
- `rho_2 = 0.05` stands; `rho_1 = 0.03` and every other value of `prereg_parameters.json` stand. The parameter file is unchanged (SHA-256 `0fbfc01857c5abe339ab1926361cd57d6e988da20e754fd6dea3ffa4e076b406`); sections 3 to 8 of this file stand.
- "proceed P2", after this addendum. The pilot is launched with `--params reports/ms/r1/prereg_parameters.json`.

## A.2 A1: the added arm `MS_base2400` (budget-matched control)

Why (the PI's text): at `rho_2 = 0.05` the terminal-stage stop is expected to be rare (1 of 60 v2.0 trajectories, `01_calibration.md` Table 2), so most rule-arm runs train 2000 updates at LR 3e-4 plus the 400-update landing, against 1200 + 400 for `parents_A`. Extra training at 3e-4 alone moved the q = 50 peak in R2b (`A_ctrl200_lr3e-4` against its parent: -0.01901 [-0.03334, -0.004326]). Without a matched-budget control the comparison with `parents_A` cannot separate the rule, polishing and samplers from the budget.

| | `MS_base` (section 5) | `MS_base2400` |
|---|---|---|
| `start_weights` | `{scheme: bin_balanced}` (the locked `StartSampler.balanced` call) | the same |
| `rule.enabled` | false | false |
| terminal stage | fixed 1600; LR 3e-4 to local 1200, linear 3e-4 -> 3e-5 over 1201-1600 | **fixed 2400; LR 3e-4 to local 2000, linear 3e-4 -> 3e-5 over 2001-2400** (`pipeline.lr_windows["2"] = [{first: 2001, last: 2400, start: 3e-4, end: 3e-5}]`: the same `lr_linear` form and end value as the rule arms' landing) |
| stage 1 | fixed 600, window 1-600 | exactly as `MS_base` |
| checks | every K = 25 reported, would-fire recorded | the same, would-fire at the pre-registered thresholds (rho_2 = 0.05; launched with `--params`) |
| runs | q in {50, 60} x seeds 10501-10510 = 20 | the same |

- **Output root** (recorded): `results/ms_r1/pilot/q{q}/seed{s}/MS_base2400/`, the same root as the five rule arms: the pilot wave is now six arms, 120 runs, one launch record. `run/run_ms_stagewise.py:validate_config` accepts the config (checked on all 120 configs by a dry run into a scratch directory before this addendum was written; the runner itself is not changed; if the runner refuses the config at run time, the launch stops and is reported).
- **Check C-MS2** (post-launch, `tools/ms/launch_checks.py --cms2-ref`; a failure is a stop-and-report). Against `parents_A` (v2.0 `phase_A` runs, root in section 7), per (q, seed), through update 1201, the first update of `parents_A`'s LR window (which still runs at 3e-4: `lr_at` of the linear form returns `start` exactly at the window's first update):
  1. the weight exports `u0025 ... u1200` (48 files, both sides must hold exactly that set) equal bit for bit;
  2. the per-update training series of updates 1..1201 (`train_history.json`, every key both writers share, the learning rate included, so update 1201 is at 3e-4 on both sides) equal;
  3. the five stream positions after every update 1..1201, and the losses, KL and clip fraction (`v2_updates.csv` against `ms_updates.csv`) equal;
  4. the export `u1225` exists on both sides and differs in at least one array.
  Pass = 20 of 20 on all four.
- **Comparisons**, fixed before any pilot run. `MS_base2400` enters the primary table against `parents_A` like every other arm (`criterion.csv`, `paired_vs_parents_A.csv`, ...): this row is the budget effect, exactly paired up to update 1201. **Secondary table:** each of the five rule arms minus `MS_base2400` (rule, polishing and sampler at matched budget), paired by (q, seed), with the D6 statistics (10,000 resamples, a fresh `numpy.random.default_rng(20261006)` per (q, statistic)) and parts (a) and (b) computed against `MS_base2400` (a: the 95 % interval of the mean paired difference of `stage2_peak_rel_err_abs`, arm minus `MS_base2400`, lies below 0 at both q; b: no run that passes G-A with its G-N eta part under `MS_base2400` fails it under the arm); descriptive, also covered by the blind recomputation. `MS_base2400` is not part of the "vs `MS_rule`" comparisons. The primary criterion is otherwise unchanged.
- **Code**: authorised in `tools/ms/` and the matching tests only (done: `ms_configs.py` arm table and a per-arm legacy pipeline, `launch_ms_r1.py`, C-MS2 in `launch_checks.py`, `r1_analysis.py`, `blind_criterion.py`). `run/`, `utils/`, `envs/`, `agents/`, `protocols/` stay at their `c95a2af4` content: `git diff c95a2af4 <code commit> -- run utils envs agents protocols` is empty (section A.7). The launch record carries `git diff --stat c95a2af4 HEAD -- run utils envs agents protocols` (`diff_stat_run_code_to_head`; expected empty: the pilot runs the run code of the C-R4 and C-MS1 runs).

## A.3 A2: expectations, recorded before the launch

All from `01_calibration.md`; recorded so that they are not read post hoc.

1. The terminal-stage stop is expected to be rare at rho_2 = 0.05 (Table 2: 1 of 60 v2.0 trajectories), so the rule arms are expected to end mostly by the cap (`budget_forced = true`).
2. The localized branch is expected mainly at q = 60: at the constant-LR block ends u400 / u800 / u1200 the v2.0 trajectories classify localized in 0 / 2 / 3 of 30 runs at q = 50 and 5 / 23 / 24 of 30 at q = 60 (Table 4). The q = 50 comparisons therefore test mainly the global samplers and the budget, and at q = 50 `MS_rule` minus `MS_base2400` is expected to differ little.
3. R, the maximum over the non-tail region, sits at d < 0 in 99.5 % of the exports and ranks the peak error poorly (Table 1: Spearman 0.179 pooled, 0.056 at q = 50), while the residual at d = 0 tracks it (0.985). The stop is a global accuracy criterion; the peak is addressed by lambda_P and by S_t when S_t contains near-tie bins.

## A.4 A3: analysis additions (descriptive; no change to any run), as implemented in `tools/ms/r1_analysis.py`

All are labelled descriptive and computed from files already in each run directory (the comparators where their files carry the quantity, else NaN, listed in `analysis_info.json` under `a3_not_stored_by_design`).

- **(a) R0** = `r_2(0)/s_2` at the terminal-stage freeze on both tiers, from `freeze_stage2_{final,development}.npz`: at the node d = 0 of `v_t2_d_grid`, `|v_t2_e_hat - v_t2_a_dev| / v_t2_a_dev` (the D3 definition; a test asserts equality with `utils.ms_residual.stage_diag`). Per-run columns `t2_R0_final`, `t2_R0_dev`, `t2_peak_over_R0_final`, `t2_peak_over_R0_dev` (|peak error| / R0, NaN if R0 = 0); table `r0.csv` per (arm, q) with the references repeated per row: calibration median 1.65 (q = 50) / 1.45 (q = 60) and the linearised factor `(2k + a)/(2k)`, `a = DW/(4 q^2)`, which gives 1.700 / 1.486.
- **(b) Strata** at the freeze, on each tier: near-tie |d| < 20, tail |d| >= 2q, middle = the rest, each split d < 0 / d > 0 (the node d = 0 itself belongs to neither side; it is the peak of (a)). Per run, tier, stratum and side (`strata.csv`; per-arm summary `strata_summary.csv`): the closed-form error of the learned effort is `recovery_e2 - recovery_g2` on the `recovery_d_grid` (801 points at q = 50, 881 at q = 60) (mean signed error, RMSE, max |error|, in effort units and divided by e_2*(0) = `recovery_g2(0)`); the maximum of `r_2/s_2` over the verifier nodes of the stratum and side. The middle stratum is reported explicitly (its start share is 0.15 at q = 50 under lambda_P = 0.35, 0.40 bin-balanced).
- **(c) Blocks and classifications**: `rule.csv` adds the number of global blocks to the existing polishing / fire / cap / localized / broad counts; `rule_blocks.csv` and the new `classifications.csv` give, for every classification, `|S_t|` and its split into near-tie / middle / tail bins (labels from `StartSampler.stratum_labels(2, 20)`, as `launch_checks.py` does); `rule.csv` counts the classifications and runs whose `S_t` contains near-tie bins.
- **(d) Stage 1**: per run, the number of stage-1 checks with `R_1 = 0` exactly (`t1_n_checks_R0`) and whether the firing streak (the M = 3 consecutive checks ending at `fire_local`, or at `would_fire_local` for the two legacy arms, labelled `would_fire`) contains such a check (`t1_streak_kind`, `t1_streak_has_R0`); `R_1` on the final tier at every stage-1 freeze (`t1_R_final`); tables `stage1_R1_runs.csv` and `stage1_R1.csv` (reference: `01_calibration.md` section 5, 17 of 60 replayed fires at rho = 0.03 contained such a check).
- **Secondary table** (A.2): `paired_vs_MS_base2400.csv`, `criterion_vs_MS_base2400.csv`, extra rows in `paired_seed_level.csv`, `figures/paired_primary_vs_MS_base2400.png`.

## A.5 Launch plan (replaces items 1, 3 and 4 of section 9; everything else of section 9 stands)

1. After this addendum is committed and pushed, from a clean tree at the head of `ms-r1` (code of `9b719715`, run code of `c95a2af4`), in tmux, root `results/ms_r1`:
   `python tools/ms/launch_ms_r1.py --wave pilot --params reports/ms/r1/prereg_parameters.json --workers 40 --code-commit c95a2af4`
   (120 runs: the five rule arms and `MS_base2400`; one launch record `results/ms_r1/pilot/launch_<stamp>.json` with nproc, load, free disk, HEAD, `git diff --stat c95a2af4 HEAD`, `git diff --stat c95a2af4 HEAD -- run utils envs agents protocols`, `git status --porcelain` and the parameter file's SHA-256). At most 40 single-threaded workers. Crash rule as in section 9 item 2.
2. Post-launch checks:
   `python tools/ms/launch_checks.py --root results/ms_r1/pilot --arms MS_rule MS_s25a0 MS_s25a5 MS_s35a0 MS_s35a5 MS_base2400 --code-commit <HEAD at launch> --out results/ms_r1/pilot/launch_checks.json --cms2-ref <parents_A root> --cms2-arm MS_base2400`
   (the checks of section 9 item 3 for all six arms, and C-MS2 for `MS_base2400`: pass 20 of 20, else stop and report).
3. Analysis: `python tools/ms/r1_analysis.py --base-root <the MS_base wave root> --pilot-root results/ms_r1/pilot --parents-root <parents_A> --rehearsal-root <rehearsal_v2_0> --calibration-root results/ms_r1/calibration --out results/ms_r1/analysis` (seven arms: `MS_base` from the base root, the other six from the pilot root). The base root is `/home/fjiang4/tournament_experiment/.claude/worktrees/ms-r1-multistage-development-afebf2/results/ms_r1/base`: the freeze arrays that A3 (a) and (b) read (`freeze_stage2_*.npz`) are untracked and exist only where the wave ran (section 7 rule: explicit roots, recorded in `analysis_info.json`); this worktree's `results/ms_r1/base` holds the tracked records only, then `python tools/ms/blind_criterion.py --analysis-dir results/ms_r1/analysis` (the primary `criterion.csv` and the secondary `criterion_vs_MS_base2400.csv`, agreement to 1e-12).
4. Wall estimate: 0.12-0.2 s per update on this shared host (load average about 12 of 64 at the time of writing): about 10-17 minutes per run (3000 updates for `MS_base2400`, up to 3400 for a rule arm), three waves of 40 workers.
5. Then sections 3-4 of the prompt unchanged (reports, tracking), the push of section 4.3, and the STOP.

## A.6 Choices where the PI's text is silent (fixed here, before the launch)

1. `MS_base2400` is launched inside the pilot wave (same root, one launch record), not as a separate wave.
2. C-MS2 compares the series and stream positions for updates 1..1201 inclusive and the exports `u0025 ... u1200`; the first export after the through-update is `u1225` (= `(1201 // 25 + 1) * 25`). The tool is `tools/ms/launch_checks.py` (`cms2_run`, `cms2_root`; options `--cms2-ref`, `--cms2-arm`, `--cms2-through`, the last only for tests); the result is written into `launch_checks.json` under `c_ms2`, and the exit code is non-zero if it fails.
3. The would-fire entry of `MS_base2400` is evaluated at local updates of its own terminal phase (checks every 25 up to local update 2400); it decides nothing.
4. The secondary table pairs the arms that are `done` on both sides, exactly as the primary table does; a pending or failed pair is reported (`b_pending`, `b_violations`), not dropped silently.
5. For the strata of A3 (b) the d = 0 node is excluded from both sides; strata use the run's own q; `n_nodes` counts nodes of the recovery grid (801 points at q = 50, 881 at q = 60).
6. The firing streak of A3 (d) has M checks, M read from the run config (`rule.M`).

## A.7 Code commit, tests, review (filled before the launch)

Code commit: **`9b719715`** (`9b7197154c9ee5e92e93cdfb35144c059f0107d6`, "feat: add MS_base2400 arm, C-MS2, secondary table and A3 analysis"), on top of the pre-registration commits of P1. Its diff outside `reports/`, `results/` and `docs/` is `tools/ms/{ms_configs,launch_ms_r1,launch_checks,r1_analysis,blind_criterion}.py` and `tests/test_ms_{analysis,analysis_a3,cms1,launcher}.py` only (9 files, +1746 / -124); `git diff c95a2af4 9b719715 -- run utils envs agents protocols` is empty.

- **Tests.** Full suite `pytest tests` on the final tree (2026-10-07 05:21-05:49, tmux): `1 failed, 932 passed, 2 xfailed, 2 warnings in 1651.91s (0:27:31)`; the one failure is the known `tests/test_registry_canonicalization.py::test_registry_canonicalization`. 932 = 881 (P1) + 51 new tests. Added: C-MS2 on reduced real runs (an identical pair, a tampered export / series value / stream position, the export after the window start made identical, a missing file or run, the CLI, a C-MS2 failure alone giving a non-zero exit, and the real LR schedule: 3e-4 through local 2001, parents_A's window starting at exactly 3e-4 at update 1201); the launcher (120 runs, the arm's config differs from `MS_base` only in the terminal-stage budget and window, the run-code diff stat limited to the five paths); the analysis (seven arms, the secondary table by hand, one fresh generator per (q, statistic), the blind recomputation of both tables with tampering, A3 (a)-(d) on constructed inputs).
- **Real-data checks before the launch** (read-only on the P1 results, outputs in a scratch directory): on the 20 real `MS_base` runs the analysis runs to completion, the blind recomputation agrees (28 numbers), the median |peak error|/R0 is 1.656 (q = 50) and 1.462 (q = 60) against the calibration's 1.65 / 1.45 (the linearised factors are 1.700 / 1.486; nothing was tuned to match), and 7 of the 10 stage-1 would-fire streaks at q = 50 contain an exact R_1 = 0 check (0 of 10 at q = 60).
- **Independent read-only review** of the diff by an agent that did not write it: highest severity MINOR, no blocker or major finding. Verified there: `MS_base` configs byte-identical at HEAD and in the working tree (72 of 72, and equal to the 20 committed base-wave `run_config.json`), `MS_base2400` differs from `MS_base` only in `arm`, `run`, the terminal budget and its window; the pilot wave plans 120 runs and refuses `--arms MS_base`; C-MS2 compares updates 1..1201 and exactly the 48 exports `u0025 ... u1200`, with the real `parents_A` at lr 3e-4 at update 1201 and below it at 1202 in 20 of 20 runs; real `MS_base` against `parents_A` passes the first three fields and fails the fourth (`u1225` identical) in 20 of 20, as it must; the primary criterion tables of the six old arms are `assert_frame_equal`-identical at HEAD and in the working tree; the strata (max abs difference 0) and the streak logic agree with independent recomputations on real files, and R0 reproduces the calibration's ratio. The reviewer also ran one real `MS_base2400` run (q = 50, seed 10501) in a scratch directory on the working tree: the runner accepted the config (no refusal), the run finished with 3000 updates, the LR trace is 3e-4 through local 2001 and 3.0e-5 at 2400, and C-MS2 against the real `parents_A` passed all four fields for it. That run is outside `results/`, was made on a dirty tree, and is used for nothing else. Findings, all fixed before the commit: (1) the "vs MS_rule" figure drew an empty `MS_base2400` row (the arm is now excluded; test added); (2) the blind script skipped the secondary table when its file was absent (now a disagreement when `per_run.csv` holds `MS_base2400` and rule arms; test added); (3) two weak tests strengthened (the run-code path limit of the launch record; a C-MS2 failure alone giving a non-zero exit); (4) "801-point recovery grid" is 881 points at q = 60 (wording).
- **Launch record** (`results/ms_r1/pilot/launch_<stamp>.json`) will carry `diff_stat_run_code_to_head` = `git diff --stat c95a2af4 HEAD -- run utils envs agents protocols`; expected empty.
