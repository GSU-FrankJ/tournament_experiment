# Multistage round MS-R4: the coworker's two tries at T=2 — a robust kink-capable actor (E1) and whether development-DP-BR sampling feedback helps PPO training (E2) — new development seeds 10511-10530

The T=2 status pack is closed (`reports/t2_status_100826/`, branch `t2-status-pack` at `be2881ff`; the PI's reply is `reports/t2_status_100826/pi_record/22_pi_reply.md`). The coworker who decides on T=2 has answered (Appendix A, verbatim): continue with two tries on new development seeds, then settle the T=2 protocol and start T=3. Solver v2.0 (`protocols/v2_T2_locked_v2_0.json`, tags `t2-v2-lock-v2.0`, `t2-v2-confirmation-v2.0`) stays the locked baseline throughout this round.

This prompt implements the coworker's plan with the adjustments listed below. The adjustments are the PI's; the coworker's note allows adjustments based on experience. The reports must name every adjustment and its reason.

## The coworker's plan in short (Appendix A is binding wherever this list is silent)

- **E1, the stage-2 tie bias.** Start from MS-R3's `relu_st_s16` and test exactly one robustness candidate, fixed in advance: LeakyReLU hidden units and a smooth, bounded mean map without the hard clamp. Everything else stays as in MS-R3: stratified starts, the s = 16 noise landing, 2800 stage-2 updates. `relu_st_s16` and `t1_st_s16` are re-run on the same seeds as controls. Target at both q: mean |peak error| <= 0.04, a paired improvement over `t1_st_s16` in the 95 % interval, and no G-A, G-F, G-N or G-S failure. Only the combined change is reported, with no attribution to either part. If the target is missed, the line stops.
- **E2, whether the development DP-BR helps PPO training.** The t1 actor, training settings and budget are fixed (stage 2 1600 updates, stage 1 600); no dynamic stop and no polishing. Only the sampling feedback differs:
  - (1) fixed stratified sampling (alpha = 0);
  - (2) the existing residual-guided sampling (alpha = 0.5);
  - (3) the per-state deviation gains of the development DP-BR, aggregated per state bin and combined with the existing action residual (alpha = 0.5).

  All arms use the same seeds, check frequency and strata shares. The sampling weights are recorded after every check, and (1) runs the same checks. The closed form is used for offline evaluation only. Target against (1) at both q: the area under the stage-2 eta_2/DW learning curve at fixed checkpoints falls by at least 20 % and the paired 95 % interval supports the improvement, while the final gates and the stage-1 deviation gain do not get worse. If (2) meets the target, (3) is not needed to show that the feedback works. If neither meets it, the DP-BR stays a verifier and no training improvement is claimed.
- **Seeds.** q in {50, 60}. New development seeds 10511-10530: 20 per q, paired across arms. 30501-30520 (the v2.0 confirmation block) are not new seeds. 40501-40520 stay reserved for the confirmation after a lock.
- **Deliverables per arm:** the configuration and code version, per-seed results, paired intervals, the peak, learning curves, the final gates and the run cost. Locking one complete scheme and its fresh-seed confirmation come after this round. The combination of E1 and E2 would first need its own development run. If the confirmation fails, v2.0 stays and both tries are reported as ablations and limitations.

## The PI's adjustments (binding)

1. **Strata shares in E2.**
   - The existing sampler (`StartSampler.stratified_bin_probs`, `envs/curriculum_env.py`) is p(b) = lambda_T u_tail(b) + (1 - lambda_T) [(1 - alpha) p_PM(b) + alpha f(b)], with the focus f normalised over all non-tail bins. Only the tail share is fixed: at alpha = 0.5 the near-tie and middle shares move with the focus.
   - Recorded shares (arm means of `t2_share_near_all` / `t2_share_mid_all`):

     | arm | q = 50 near / mid | q = 60 near / mid | source |
     |---|---|---|---|
     | design | 0.35 / 0.15 | 0.35 / 0.195 | |
     | MS-R1 `MS_s35a0` (alpha 0) | 0.348 / 0.152 | 0.310 / 0.236 (polishing blocks at q = 60) | `results/ms_r1/analysis/per_run.csv` |
     | MS-R1 `MS_s35a5` (alpha 0.5) | 0.234 / 0.266 | 0.259 / 0.286 | `results/ms_r1/analysis/per_run.csv` |
     | MS-R3 `t1_st_s16` | 0.232 / 0.268 | 0.236 / 0.309 | `results/ms_r3/analysis/per_run.csv` |

   - At alpha = 0.5 the existing sampler therefore moves a quarter to a third of the near-tie mass into the middle stratum. Arm (2) minus arm (1) would then mix the feedback with a change of the strata shares, against the coworker's "same strata shares".
   - Arms (2) and (3) therefore use a **within-stratum focus** (D4). In `fb_a0`, `fb_r` and `fb_rg` the tail, near-tie and middle shares are exactly lambda_T, lambda_P, lambda_M at every update, and the feedback only moves samples inside a stratum. The price is a weaker lever than the cross-stratum sampler.
   - The existing cross-stratum sampler is kept as a fourth, descriptive arm, `fb_rx`. It is the sampler inside E1's arms, and `fb_rx` equals E1's `t1_st_s16` bit for bit through local update 1201 (check C-X2).
2. **E1's arms already contain DP-BR feedback.** MS-R3's `*_st_*` arms use `fixed_global_sampler` with `alpha_global` 0.5, i.e. the sampler of `fb_rx`. E2's outcome governs the *claim* about DP-BR feedback. It does not remove the feedback from E1's configuration, because that would be an untested configuration. The reports say so.
3. **The learning curve is evaluated on the final tier, offline.** Arm (3) samples by the development-grid deviation gains, and the development-tier eta_2 is the maximum of those same gains. Scoring E2 with it would score (3) by its own target. The primary curve is therefore the final-tier eta_2/DW, evaluated offline on the weight exports at fixed checkpoints. The development-tier values of the online checks are a secondary statistic (D6).
4. **Operational definitions** of the coworker's targets are given in D6: the area statistic, the two "not worse" conditions (the stage-1 one with a non-inferiority margin), and the order of (2) and (3). The parameters of the E1 candidate are in D3.
5. **Prior information for E2** (PI-side; P1 reproduces it, §2.3 (a)). Source: the tracked MS-R1 check tables, `results/ms_r1/pilot/q*/seed*/<arm>/ms_checks_stage2.csv`. Quantity: the development-tier `Delta` = eta_2/DW every 25 updates, local <= 1600. Arms `MS_s25a0`, `MS_s25a5`, `MS_s35a0`, `MS_s35a5`, 10 seeds per q. These arms ran the stop rule, with polishing mostly at q = 60.
   - The first 200 updates hold 49-75 % of the trapezoidal area. Arm means of the per-run share: 0.61, 0.66, 0.49, 0.62 at q = 50; 0.72, 0.75, 0.67, 0.73 at q = 60.
   - The reduction 1 - mean(A_alpha0.5) / mean(A_alpha0), where negative means the area grew:

     | window | q | lambda_P 0.25 | lambda_P 0.35 |
     |---|---|---|---|
     | full | 50 | -0.067 [-0.110, -0.029] | +0.067 [-0.000, +0.131] |
     | full | 60 | -0.192 [-0.245, -0.150] | -0.146 [-0.171, -0.117] |
     | local 200-1600 | 50 | +0.088 [-0.000, +0.155] | +0.311 [+0.219, +0.390] |
     | local 200-1600 | 60 | -0.044 [-0.183, +0.076] | +0.078 [+0.043, +0.106] |

   - The linear area is dominated by the early transient, and the cross-stratum alpha 0.5 sampler did not lower it. The coworker's area stays the primary statistic. The secondary statistics of D6 are pre-registered so that a miss can be read.
6. **Power of E1's zero-failure target.** Zero failures in 40 runs bound the per-run failure rate by 8.8 % (exact Clopper-Pearson 95 % upper limit). A candidate with `relu_st_s16`'s observed q = 50 rate (1 of 10 on seeds 10501-10510) would pass all 20 q = 50 runs with probability 0.9^20 = 0.12. Passing E1 screens out relu-sized failure rates; it does not establish robustness. The reports say so.
7. **The cost of the checks** (PI-side; P1 reproduces it). In the MS-R1 pilot a development check took 0.0165 s at the median (n = 11,520; column `verifier_sec` of `ms_checks_stage2.csv`). A stage-2 update took 0.161 s at the median (n = 288,000; column `update_wall_sec` of `ms_updates.csv`). One check per 25 updates is about 0.4 % of the stage-2 wall time.

Steps:

1. **P0** housekeeping (§1).
2. **P1** code, tests, offline work (the preamble numbers, the curve tool), the reproduction checks C-R7 and C-MS6, the review, the pre-registration and the push (§2); then **STOP at gate G1**.
3. **P2** (after my reply "proceed P2"): the launch of wave `r4` (280 runs), its checks, the offline curves and the analysis (§3).
4. **P3** reports, fact-check and push, then **STOP** (§4, §5).

Repository `/home/fjiang4/tournament_experiment`; Python `/home/fjiang4/tournament_experiment/.venv/bin/python` only; `pytest tests` (never a bare `pytest`); every run single-threaded (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) in tmux; do not kill jobs you did not start. Work in your session's own worktree on a new branch `ms-r4` created from `origin/t2-status-pack` (`be2881ff`); record the worktree path. Results root `results/ms_r4/`; reports `reports/ms/r4/`; new tools under `tools/ms/` (prefix `r4_`); new tests `tests/test_ms_r4_*.py`.

Everything from the earlier rounds still applies:

- **Provenance:** every number in a report cites a path.
- **Closed form:** the closed-form equilibrium enters evaluation, reporting and the offline curves only. It never enters RL training, the sampler or its focus, the schedule, or anything a run decides.
- **Results:** no deletion or overwriting of results or checkpoints; the paper-registry test and its data are unchanged; no `.pt`, `train_history.json`, weight exports or freeze arrays go into git; `clamp_likelihood` stays `density`.
- **Frozen files:** nothing changes in `protocols/`, `run/run_v2_T2_locked.py`, `run/run_v2_stagewise.py`, `run/v2_rollout.py`, `utils/v2_continuation.py`, `utils/v2_metrics.py`, `utils/dp_br_verifier.py`, `envs/` or `tools/v2/`.
- **New code behind keys only:** `agents/`, `utils/ms_residual.py`, `utils/ms_rule.py` and `run/run_ms_stagewise.py` may gain code only behind the new variant and the new keys. Their absence leaves every existing path bit-identical; the default-path tests, C-R7 and C-MS6 are the proof.
- **Pushing** is authorised only where §2.7 and §4.3 say so. `main` and tags are not touched; no pull request.
- Save this prompt verbatim as `reports/ms/r4/pi_record/23_ms_r4_prompt.md` before anything else.

**If any step fails, or anything does not match this prompt, stop at that point and report. Do not work around it.**

---

## 0. Decisions (binding)

### D1. Scope

- **Seeds and runs.** T = 2. New development seeds 10511-10530 × q ∈ {50, 60}: 20 runs per arm and q, paired by (q, seed) across arms. Reproduction runs use 10501-10510 (C-R7, C-MS6). No fresh seeds: 30501-30520 and 40501-40520 appear in no config of this round.
- **Out of scope:** no lock, no confirmation, no combined E1+E2 arm, no T = 3 training run (reduced T = 3 smoke tests only) and no protocol change. v2.0 stays the T = 2 solver.
- **Rule and checks.** No arm has a development stop, a classification or polishing. Every arm runs the checks every K = 25 local updates in every stage and records them. The would-fire record uses the thresholds of `reports/ms/r1/prereg_parameters.json`, passed explicitly; SHA-256 `0fbfc01857c5abe339ab1926361cd57d6e988da20e754fd6dea3ffa4e076b406`. That file also supplies K, M, the EMA weight 0.5 and the near-tie half-width 20.
- **Critic:** unchanged (tanh on d / B) in every arm.

### D2. Arms (seven; 20 runs per arm and q; 280 runs)

| experiment | arm | actor | starts | terminal-stage budget and landing |
|---|---|---|---|---|
| E1 | `t1_st_s16` (control) | `t1` | MS-R3 stratified (cross-stratum focus, alpha 0.5) | 2800, s = 16 |
| E1 | `relu_st_s16` (control) | `relu` | same | same |
| E1 | `lrsm_st_s16` (candidate) | `lrsm` (D3) | same | same |
| E2 | `fb_a0` = (1) | `t1` | stratified, alpha 0 | 1600, s = 1 |
| E2 | `fb_r` = (2) | `t1` | stratified, alpha 0.5, within-stratum focus = residual map (D4) | same |
| E2 | `fb_rg` = (3) | `t1` | stratified, alpha 0.5, within-stratum focus = residual + gain map (D4) | same |
| E2 | `fb_rx` (descriptive) | `t1` | stratified, alpha 0.5, the existing cross-stratum focus | same |

**E1 settings** are those of MS-R3's `*_st_s16` arms (`tools/ms/ms_configs.py`, `R3_ARM_TABLE`):

- Starts: `stratified_priority`, lambda_P 0.35, `alpha_global` 0.5, `alpha_polish` 0.5 (unused), `ema_beta` 0.5, `near_tie_half_width` 20, lambda_T = the bin-balanced tail share (0.5 at q = 50, 20/44 at q = 60); `fixed_global_sampler` true; rule disabled.
- Terminal stage, fixed 2800 updates:
  - local 1-2000: LR 3e-4, scale 1;
  - 2001-2200: LR 3e-4, concentration scale linear 1 -> 16 (`conc_scale_schedule` = {stage 2, local_first 2001, local_last 2200, scale_first 1, scale_last 16});
  - 2201-2400: LR 3e-4, scale 16;
  - 2401-2800: LR linear 3e-4 -> 3e-5, scale 16.
- Stage 1: fixed 600 updates, LR linear 3e-4 -> 3e-5 over 1-600, scale reset to 1 at stage-1 entry.
- `noise_report` true; `init_digest` true.
- The configs of the two controls differ from MS-R3's arms of the same name only in the round tag, the run name and the seed (and fields derived from them).

**E2 settings:**

- Actor `t1`, no `actor_variant` key.
- Starts: `stratified_priority`, lambda_P 0.35, `ema_beta` 0.5, `near_tie_half_width` 20, lambda_T the bin-balanced tail share; `fixed_global_sampler` true in every E2 arm (`fb_a0` included: at alpha 0 it leaves the probabilities bitwise unchanged, and the four E2 configs then differ only in the sampler keys); rule disabled.
- Budgets of `MS_base`, i.e. v2.0's (`LEGACY_PIPELINE`): terminal stage fixed 1600 updates with LR 3e-4 to local 1200 and linear 3e-4 -> 3e-5 over 1201-1600; stage 1 fixed 600 updates with LR linear 3e-4 -> 3e-5.
- No `conc_scale_schedule` (s = 1); `noise_report` true; `init_digest` true.

Run names `msr4_q{q}_s{seed}_{arm}`; outputs `results/ms_r4/pilot/q{q}/seed{s}/{arm}/`.

### D3. The E1 candidate actor `lrsm` (fixed before any run; no other value is tried)

- **Hidden units:** LeakyReLU with negative slope 0.01 in both hidden layers (`torch.nn.functional.leaky_relu(x, 0.01)`). The d component of the input is d / ((t - 1) B), as now, with no scaling.
- **Mean:** mu = m + (1 - 2m) · sigmoid(z0) with m = `mu_clamp` = 1e-6, computed in float32. There is no clamp anywhere on the mean of this variant. The range is that of the current clamp. In float32 the derivative with respect to z0 is positive for about -88 < z0 < 17 (below, sigmoid underflows; above, it rounds to exactly 1), against -13.8 < z0 < 13.8 under the current clamp; the upper end corresponds to efforts near 100, far above every equilibrium effort of this game (e2*(0) = 70 at q = 50, 58.33 at q = 60).
- **Unchanged:** the architecture 2 -> 64 -> 64 -> 2; the concentration c = c_min + softplus(z1), times `conc_scale`; the critic; the clipping of sampled actions to [1e-6, 1 - 1e-6]; `clamp_likelihood` = density.
- **Initial weights:** the variant is set after construction, as for `relu` and `t10`. So the initial weights come from the same generator calls in the same order as `t1`, and the initial actor and critic weights are identical across `t1`, `relu` and `lrsm` within a (q, seed) (C-INIT).
- **Other variants:** the forwards of `t1`, `relu` and `t10` stay operation-for-operation unchanged; the clamp stays in their path.
- **Records:** `actor_variant` = "lrsm" is recorded in `run_config.json`, `manifest.json`, every weight export and the full states. The opponent, the refresh and the frozen snapshot carry it. `mean_effort_numpy`, or a variant-aware equivalent, reproduces the forward to float32 rounding.
- **No silent reload.** Every loader used in this round, including `tools/ms/replay_dev_rule.build_actor` and the tools that reuse it, either builds the `lrsm` forward from the export or refuses the export. It never falls back to a tanh or ReLU forward with the clamp. The loaders in `tools/v2/` stay unchanged and are not used on MS-R4 exports.
- **Where:** implement in `agents/ppo_curriculum.py` (`BetaActor`, `CurriculumPPO.set_actor_variant`, `mean_effort_numpy`). Do **not** change the tuple `ACTOR_VARIANTS`: MS-R3 tests iterate it and compare it with `tools/ms/r3_analysis.R3.ACTORS` (`tests/test_ms_r3_screen.py`, `tests/test_ms_r3_analysis.py`, `tests/test_ms_r3_actor.py`). Register `lrsm` in a separate tuple (e.g. `ACTOR_VARIANTS_R4 = ("lrsm",)`) and accept the union wherever a variant is validated or built: the actor, the numpy reload, the runner and the loaders used in this round. The runner accepts `actor_variant` "lrsm" besides "relu" and "t10".

### D4. The E2 sampler (new optional runner keys; absent = the MS-R3 behaviour, bit for bit)

- **`focus_scope` = "within_strata".** Allowed only with `stratified_priority`, `fixed_global_sampler` true and the rule disabled; otherwise it is a config error.
  - Notation: stage t >= 2; the strata of `StartSampler.strata(t, 20)` (tail T, near-tie P, middle M; counts n_T, n_P, n_M); lambda_T = n_T / n_bins, lambda_P = 0.35, lambda_M = 1 - lambda_P - lambda_T; alpha = `alpha_global`; a per-bin weight w >= 0.
  - The bin probabilities are

    p(b) = lambda_T · 1[b in T] / n_T + sum over S in {P, M} of lambda_S · [ (1 - alpha) · 1[b in S] / n_S + alpha · f_S(b) ],

    with f_S(b) = w(b) / (sum of w over S) for b in S and 0 elsewhere. f_S is uniform on S when that sum is 0 or not finite. When a stage has no tail bin (n_T = 0, e.g. stage 2 of T = 3), the tail term is absent and lambda_T = 0, as in `stratified_bin_probs`.
  - Before the first valid check (no map yet), p is exactly `StartSampler.stratified_bin_probs(t, 0.35, 20.0, alpha=0.0)`: call it, so that the floats are the same. Every E2 arm is then bit-identical through local update 25 (C-FB25).
- **The weight w for `fb_r`:** the existing residual map rho_bar, i.e. the EMA of rho_b = max over the development nodes d of non-tail bin b of r_t(d) / s_t (`utils/ms_residual.py`), with NaN read as 0.
- **`focus_map` = "residual_gain"** (requires `focus_scope`).
  - From the **same** development verifier call, with no extra call: g_b = max over the same nodes of Delta_t(d) / DW, where Delta_t(d) is `res.stages[t].delta` (non-negative by construction, `utils/dp_br_verifier.py`).
  - g_bar is the EMA of g_b with the same weight 0.5, initialised at the first valid check, with NaN handling as in `ema_update`.
  - Within each stratum S: f_S(b) = 1/2 · rho_bar(b) / (sum of rho_bar over S) + 1/2 · g_bar(b) / (sum of g_bar over S).
  - Fallbacks: if the g_bar sum over S is 0, use the normalised rho_bar term alone; if the rho_bar sum is 0, the g_bar term alone; if both are 0, uniform on S.
- **No closed form.** Nothing here reads the closed form. The stub test covers both keys (§2.2).
- **Where the code lives.** The probability function goes in a new module (e.g. `utils/ms_focus.py`). The g_bar EMA and the choice of focus go behind constructor arguments of `utils.ms_rule.StageController`, not `StageRule` fields, so that the logged rule parameters of MS-R1..R3 are unchanged. `envs/` is not changed, and the draw stays `StartSampler.stratified_priority(t, n, rng, probs)`.
- **Records.** For an arm with a new key, `ms_binmaps_stage{t}.npz` additionally holds, per check:
  - `focus_within` (f_S on the near-tie and middle bins, 0 on the tail bins);
  - `probs_next` (the bin probabilities in force from the next update; NaN at the last check of the stage, which has no next update);
  - with `focus_map`, also `gain` and `gain_bar`.

  For an arm without the new keys, every output array is unchanged.
- **T-generic:** defined for every t >= 2 (reduced T = 3 smoke, §2.2).

### D5. What is measured

**E1**, for each of the three arms:

- **At the terminal freeze**, on both tiers, everything MS-R3 measured: |peak error| and the signed error, the gap, the smoothing part, the remainder, sigma_2(0), RMSE_pos/e2*(0), tail mean/e2*(0), eta_2/DW, R0, w_eff, the location-free peak and its argmax, the symmetry error, and the per-stratum errors. Stage 1: G-S, the stage-1 error, G-F (Gmax_full/DW) and G-N(Gmax). The four gates.
- **Learning curves:** on every stage-2 export (u25 ... u2800, 112 points), the final-tier eta_2/DW, |peak error| and e_hat_2(0). The closed form is used for evaluation only.
- **Failures:** for every run that fails a gate or exits nonzero:
  - its trajectories of e_hat_2(0), R0, RMSE_pos/e2*(0) and the symmetry error;
  - whether the tie mean sits at the bound of the mean map;
  - for `relu` and `lrsm` runs, per first- and second-layer unit, the share of the development nodes of D_2 with a positive pre-activation at the freeze (post hoc, descriptive).
- **Cost:** per-run wall time, per phase and by category (`train_rollout`, `train_update`, `dev_verifier`, from `ms_run_summary.json`).

**E2**, for each of the four arms:

- **Curves:** the final-tier eta_2/DW on every stage-2 export u25 ... u1600 (64 points; these are the "fixed checkpoints"), and the development-tier values of the online checks.
- **Gates and end points:** the gates; Gmax_full/DW and the stage-1 error at the end of stage 1; |peak error|, gap, RMSE_pos/e2*(0) and tail mean/e2*(0) at the stage-2 freeze.
- **Sampling:**
  - per update, the realised start counts per stratum (`n_start_tail`, `n_start_near`, `n_start_mid`);
  - per check, the sampling weights in force after the check: from `probs_next` where recorded, otherwise reconstructed from `probs_first_update` / `probs_table` and verified against `p_digest_next`;
  - per check, rho_bar, and for `fb_rg` also g_bar.
- **Cost:** as for E1, plus the development verifier's share of the stage-2 wall time. The offline evaluation cost is reported separately, since it is evaluation, not training.

### D6. Criteria and decision rules (pre-registered)

- **Bootstrap:** 10,000 resamples of the 20 seeds of a q, with replacement; a fresh `numpy.random.default_rng(seed)` per (q, statistic), in table order; percentile intervals. Seed 20261013 for E1 and 20261014 for E2.
- **Pairing:** by (q, seed).
- **Failed runs:** every planned run is reported. A run that exits nonzero (after the crash rule) is a failed run.

**E1 target.** All three parts must hold at q = 50 and at q = 60:

- (E1-a) the mean |peak error| of `lrsm_st_s16` over its 20 runs is <= 0.04;
- (E1-b) the 95 % interval of the mean paired difference of |peak error|, `lrsm_st_s16` - `t1_st_s16`, lies entirely below 0;
- (E1-c) none of the 40 `lrsm_st_s16` runs fails G-A, G-F, G-N or G-S, and none exits nonzero. The gates are those of `protocols/v2_T2_locked_v2_0.json`, as the runner writes them to `gates.json`.

Rules for E1:

- **Met** iff (a), (b) and (c) hold at both q.
- **Not met:** the line stops, and no other slope, map or variant follows.
- **Attribution:** met or not, the effect is reported as that of the combined change. No part of it is attributed to the LeakyReLU or to the mean map.

Descriptive for E1:

- the same paired statistics for `lrsm_st_s16` - `relu_st_s16` and for `relu_st_s16` - `t1_st_s16`;
- gate failures per arm and q, with exact Clopper-Pearson 95 % intervals;
- the paired changes of the gap, the smoothing part, the remainder, sigma_2(0), RMSE_pos, tail mean, eta_2, R0, w_eff, the stage-1 error and Gmax_full/DW.

**E2 target**, for X in {`fb_r`, `fb_rg`} against `fb_a0`:

- **Area per run:** A = (1 / 1575) × the trapezoidal area of the final-tier eta_2/DW over local updates 25 ... 1600 (the 64 exports).
- **Reduction per q:** rho_X = 1 - mean(A_X) / mean(A_fb_a0), with means over the 20 seeds. Its interval comes from the paired bootstrap: resample the seeds jointly for X and `fb_a0` and recompute the ratio of means.
- (E2-a) rho_X >= 0.20 and the lower end of its 95 % interval > 0, at both q.
- (E2-b) no (q, seed) that passes all four gates under `fb_a0` fails any gate under X.
- (E2-c) Gmax_full/DW at the end of stage 1 (final tier): the upper end of the 95 % interval of the mean paired difference X - `fb_a0` is <= +0.001 at both q. The margin is the G-N tolerance, used as the non-inferiority margin. For scale, MS-R3's `t1_st_s1` had medians of 0.00105 at q = 50 and 0.00048 at q = 60 (`results/ms_r3/analysis/per_run.csv`, `Gmax_full_over_dw`).
- **Met:** X meets the target iff (a), (b) and (c) hold.

**Order of the E2 decision:**

1. `fb_r` is evaluated first. If it meets the target, the statement "development-DP-BR sampling feedback lowers the area under the stage-2 eta_2/DW curve at this budget" rests on `fb_r`, and `fb_rg` is reported descriptively.
2. If `fb_r` does not meet the target, `fb_rg` is evaluated. If `fb_rg` meets it, the statement rests on `fb_rg`.
3. If neither meets the target, the development DP-BR is kept as a verifier only, and no training improvement is claimed.

Two arms had a chance at one statement, so 97.5 % intervals are reported next to the 95 % ones (descriptive).

**Other E2 statistics:**

- `fb_rx` against `fb_a0`: the same statistics, descriptive, outside the decision.
- Secondary statistics (descriptive):
  - the mean of log10(eta_2/DW) over the 64 points;
  - the area over local 200-1600, normalised by 1400;
  - the first export with eta_2/DW <= 0.005;
  - eta_2/DW at u1600;
  - the area of the development-tier online values on the same 64 updates;
  - |peak error|, gap, RMSE_pos and tail mean at the freeze;
  - the stage-1 error;
  - the realised strata shares;
  - the cost.

**Parameter file.** The numeric parameters of D6 are written to `reports/ms/r4/prereg_criteria.json` before the launch, and the analysis tools read them from there. They are: 0.04, 0.20, 0.001, the checkpoint grids, the windows, and the bootstrap seeds and sizes.

### D7. Predictions (written before the launch; expectations, not tests; nothing is selected by them)

- **(E1-P1)** `lrsm_st_s16` has a lower mean |peak error| than `t1_st_s16` at both q, about as `relu_st_s16` had on seeds 10501-10510. That comparison was paired -0.0104 [-0.0162, -0.0047] at q = 50 and -0.0206 [-0.0309, -0.0068] at q = 60 (`results/ms_r3/analysis/criterion.csv`, row `relu_st_s16`).
- **(E1-P2)** `relu_st_s16` fails at least one gate at q = 50 on the new seeds, and `lrsm_st_s16` fails fewer. On 10501-10510, 5 of the 40 `relu` runs at q = 50 failed G-A, from two (q, seed) cases; for `relu_st_s16` it was 1 of 10.
- **(E2-P1)** `fb_rx` has a larger area than `fb_a0` at q = 60 (adjustment 5).
- **(E2-P2)** `fb_r` falls short of a 20 % reduction at both q. Any gain of `fb_r` or `fb_rg` shows more in the area over local 200-1600 than in the full area.
- **(E2-P3)** The development checks take under 1 % of the stage-2 wall time in every arm.

### D8. Reference roots

Every tool takes these as explicit arguments and records them; a missing reference file is a stop-and-report.

- **MS-R3 runs** (C-MS6, §2.3 (b)): `/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857/results/ms_r3/pilot`. This is the worktree in which MS-R3 ran, as `reports/ms/r3/00_housekeeping.md` records; verify the path.
- **Tracked MS-R1 and MS-R3 records** (§2.3 (a)): `results/ms_r1/pilot/`, `results/ms_r1/analysis/per_run.csv` and `results/ms_r3/analysis/per_run.csv` of the branch.
- **`rehearsal_v2_0`** (C-R7): `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0`.
- **The C7 canonical worktree:** `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2`.

---

## 1. P0 — housekeeping

Write `reports/ms/r4/00_housekeeping.md` as you go (commands, hashes, verbatim outputs).

1. **Save the prompt.** Save this prompt verbatim as `reports/ms/r4/pi_record/23_ms_r4_prompt.md`. If the coworker's document `Multistage1011261.docx` is provided, store it unchanged in `reports/ms/r4/pi_record/` and check its SHA-256 against `699f73f30010bc59777469c8cada4c8f81c2fa3bbde6fa39e28349b1126e10e8`; Appendix A is its text.
2. **Refs and branch.**
   - Run `git ls-remote origin` for `main`, `ms-r3`, `t2-status-pack` and the tags.
   - Verify `origin/t2-status-pack` = `be2881ff31474ce5147bef29f9ff0e785e1d2b95` and `origin/ms-r3` = `be4fd2021e5aee52625a994b3566dc18726e0586`.
   - Create `ms-r4` from `origin/t2-status-pack` in your session's own worktree and record its path.
   - Do not touch `main` or any other worktree. Record, but do not fix, local refs that are behind.
3. **Seeds.**
   - Run `python tools/v2/seed_inventory.py --block 10511 10530 --out results/ms_r4/preflight/seed_inventory_10511_10530.csv` and record the command and its output.
   - A collision with a record of the T = 2 v2 or MS lines (`results/v2_*`, `results/ms_r*`, `reports/v2/`, `reports/ms/`, `reports/t2_*`, `protocols/`, in any worktree) is a stop-and-report.
   - Collisions with records of other experiments are listed and classified, and they do not stop the round.
4. **Reference roots.** Verify the reference roots of D8 and record what this round reads from them. For C-MS6 that is the 20 `t1_st_s16` and 20 `relu_st_s16` runs of seeds 10501-10510, with their weight exports (136 per run), freeze arrays, per-update series, check tables and gates; record the counts.
5. **Machine.** Record `nproc`, load, free disk, and the Python and package versions against the embedded record.

## 2. P1 — code, tests, offline work, pre-registration (on `ms-r4`)

### 2.1 Code

- `agents/ppo_curriculum.py`: the variant `lrsm` (D3).
- `run/run_ms_stagewise.py`:
  - accept `actor_variant` "lrsm";
  - the optional keys `focus_scope` and `focus_map` (D4), with their validation and records;
  - absent keys mean bit-identical behaviour.
- `utils/ms_residual.py`: the per-bin gain map from the same verifier result, as a new field with an empty default; the CSV scalars are unchanged.
- `utils/ms_rule.py`: the g_bar EMA and the focus choice behind constructor arguments.
- The new module with the within-stratum probabilities.
- `tools/ms/r4_configs.py` (new; `tools/ms/ms_configs.py` changes only additively, if at all, because its tests pin several of its tables):
  - `R4_E1_ARM_TABLE`: MS-R3's two `*_st_s16` entries copied with the round tag `ms_r4`, plus `lrsm_st_s16`;
  - `R4_E2_ARM_TABLE`: the four E2 arms;
  - round tag `ms_r4`, run prefix `msr4`, `R4_SEEDS` = 10511-10530;
  - `ms_configs.build_config` resolves `t1_st_s16` and `relu_st_s16` from `R3_ARM_TABLE` by name (round tag `ms_r3`). MS-R4 configs therefore come from the MS-R4 tables, through an additive option of `build_config` (absent = the current lookup) or a thin wrapper, never through the MS-R3 lookup by name. The training-relevant content of the two controls equals MS-R3's; C-MS6 proves it.
- `tools/ms/r4_launch.py` (new launcher; it reuses the job, pool and record functions of `tools/ms/launch_ms_r1.py` by import; `launch_ms_r1.py` is not changed, because its tests pin `WAVES` and `WAVE_DIR`):
  - wave `r4`: the seven arms × q × 10511-10530 into `results/ms_r4/pilot`;
  - wave `r4_cms6`: `t1_st_s16` and `relu_st_s16` from `R4_E1_ARM_TABLE` × q × 10501-10510 into `results/ms_r4/cms6`. Their configs differ from MS-R3's only in fields that do not enter training (round tag, run name, output directory);
  - seed guards: wave `r4` accepts exactly 10511-10530 and `r4_cms6` exactly 10501-10510; no wave accepts any of 30501-30520 or 40501-40520;
  - C-R7 uses the existing wave `v20_repro` of `launch_ms_r1.py` with `--root results/ms_r4`.
- `tools/ms/r4_curves.py`, the offline curve tool (D5):
  - load each stage-2 export with a variant-aware loader;
  - build the composite candidate with the export at every stage, as the live actor is used during the stage-2 phase;
  - run `utils.v2_metrics.evaluate` on the final tier, and on the development tier for C-OFF;
  - write, per export, eta_2/DW, |peak error|, e_hat_2(0) and the evaluation seconds;
  - pure (it moves no RNG); parallel over runs with at most 40 workers; output `results/ms_r4/curves/`.
- `tools/ms/r4_launch_checks.py`: the checks of §3.2.
- `tools/ms/r4_e1_analysis.py`, `tools/ms/r4_e2_analysis.py`, and the independent `tools/ms/r4_blind_e1.py`, `tools/ms/r4_blind_e2.py` (§3.3).
- `tools/ms/r4_preamble.py` (§2.3 (a)).

### 2.2 Tests (`tests/test_ms_r4_*.py`, plus every existing suite)

- **Defaults.** Without the new keys and the new variant, everything is bit-identical to `be2881ff`: every MS-R1, MS-R2 and MS-R3 test passes unchanged, and C7 passes.
- **`lrsm`:**
  - the forward matches D3: LeakyReLU 0.01 in both layers, the smooth mean map, no clamp;
  - the mean's gradient with respect to z0 is nonzero where the current clamp gives zero (z0 = -20 and z0 = +15);
  - the initial weights equal those of `t1` and `relu` for a (q, seed);
  - the numpy reload reproduces the torch forward to float32 rounding;
  - the opponent, the refresh and the frozen snapshot carry the variant;
  - each loader used in this round gives the `lrsm` forward or an error for an `lrsm` export;
  - the stage-1 continuation table built from an `lrsm` snapshot agrees with direct evaluation of that snapshot's mean.
- **E2 sampler:**
  - the per-stratum mass is exactly lambda_T, lambda_P, lambda_M (to 1e-12) for any focus and alpha;
  - the within-stratum normalisation and the fallbacks behave as in D4;
  - before the first valid check the probabilities equal the alpha-0 floats bit for bit;
  - the gain map makes no extra verifier call;
  - a new test in `tests/test_ms_r4_*.py` repeats `tests/test_ms_runner.py::test_decisions_and_sampler_do_not_depend_on_the_closed_form` with both new keys: the sampler decisions are identical when the closed-form functions are replaced by stubs (the existing test file is not changed);
  - config validation refuses `focus_map` without `focus_scope`, and `focus_scope` without its preconditions.
- **The curve tool:** on a short run, the development-tier values equal the online check `Delta` (bit for bit, or to 1e-12), and the last stage-2 point equals the final-tier eta_2 of the freeze record.
- **Reduced-budget runs:** C-FB25 on short runs of the four E2 arms; a reduced T = 3 smoke with `lrsm` and one with `fb_rg`.
- **Launcher** (`tools/ms/r4_launch.py`): the seed guards above; the configs within an experiment differ only in the intended keys; wave `r4` has 280 jobs and `r4_cms6` has 40; the existing launcher's tests pass unchanged.
- **Analysis and blind tools:** tested on synthetic rows with known answers for each criterion part, including a failed run and a missing pair.
- **Full suite** `pytest tests`: everything passes apart from the known `test_registry_canonicalization`; record the summary line.

### 2.3 Offline work (no training)

**(a) Preamble numbers.** Reproduce them with `tools/ms/r4_preamble.py` into `reports/ms/r4/01_preamble_numbers.md`:

- **Strata shares** (adjustment 1): arm means of `t2_share_near_all` and `t2_share_mid_all` for MS-R1 `MS_s35a0` and `MS_s35a5` and MS-R3 `t1_st_s16`, both q.
- **MS-R1 areas** (adjustment 5):
  - per run, take the rows of `ms_checks_stage2.csv` with a finite `Delta` and `local` <= 1600, and compute the trapezoidal area of `Delta` over `local`;
  - compute the share of the area over local <= 200;
  - per q and lambda_P in {0.25, 0.35}, compute 1 - mean(A_a5) / mean(A_a0) with the paired bootstrap interval, over the full window and over local >= 200.
- **How the PI-side intervals were drawn.** One `numpy.random.default_rng(12345)` was shared in this order: for q in (50, 60), for lambda_P 0.25 and then 0.35, the full window and then the window >= 200. Each interval draws `rng.integers(0, 10, size=(10000, 10))` over the seeds sorted ascending and takes the 2.5 and 97.5 percentiles. Reproduce that exactly, then report next to it the intervals under the convention of D6 (a fresh generator per statistic).
- **Check cost** (adjustment 7): the median `verifier_sec` over the stage-2 checks and the median `update_wall_sec` over the stage-2 updates of the MS-R1 pilot.
- **Zero-failure power** (adjustment 6): the Clopper-Pearson bound and 0.9^20.

A number that differs is reported as such. The preamble is not edited, and a difference is not a stop condition.

**(b) Validate the curve tool** on MS-R3's `t1_st_s16` (20 runs, seeds 10501-10510, from the reference root):

- **C-OFF on real exports:** the development tier equals the online checks at all 112 stage-2 points of all 20 runs, and the last point equals the freeze record on the final tier.
- **Then, descriptive:** the final-tier curves of those runs, the share of their area over the first 200 updates, and the final-tier curve against the development-tier curve.

Report: `reports/ms/r4/01b_curve_tool.md`.

### 2.4 Reproduction checks on real runs (each is a stop-and-report if it fails)

- **C-R7:** run the unchanged `run/run_v2_T2_locked.py` from `ms-r4` for q in {50, 60} × seeds 10501-10510 into `results/ms_r4/v20_reproduction/`. Compare with `rehearsal_v2_0` using `tools/v2/cr2_compare.py`: 20/20 identical.
- **C-MS6:** run wave `r4_cms6` (40 runs) and compare with the MS-R3 runs of the reference root, as C-MS5 did: every array of every weight export, the per-update series, the stream positions after every update, the check rows' common columns, the freeze arrays and the gate values. 40/40 identical.

### 2.5 Independent review

One read-only review of the diff by an agent that did not write it. It covers:

- conformance with D1-D6;
- the initialisation identity;
- the bit-identity of the defaults;
- the `lrsm` semantics and loaders;
- the within-stratum focus and the gain map;
- the closed-form-free sampler;
- the curve tool.

Findings and dispositions go in `reports/ms/r4/03_checks.md`.

### 2.6 Pre-registration

`reports/ms/r4/02_preregistration.md` contains:

- D1-D8 as implemented; the SHA-256 of every run's `run_config.json` (a CSV), and per arm the keys in which its configs differ from those of its comparator;
- the PI's adjustments and their reasons;
- the criteria and decision rules, and `prereg_criteria.json` with its SHA-256;
- the secondary analyses;
- the predictions;
- the reference roots;
- the test summary;
- the C-R7 and C-MS6 results;
- the preamble reproduction and the curve-tool validation;
- the launch plan: wave `r4`, 280 runs, at most 40 workers.

### 2.7 Commits, push, STOP at G1

1. Commit the code, tests, the offline outputs and reports, and the pre-registration (the **code commit**). Later commits touch only `results/` and `reports/ms/r4/`.
2. Add a row for MS-R4 to `reports/ms/README.md` and a dated section at the top of `docs/STATE.md` (P1 done, gate G1).
3. Push `ms-r4` (fast-forward, no force).
4. **STOP.** P2 starts only after my reply "proceed P2" (gate G1). My reply may add an addendum to the pre-registration; nothing else changes after it.

## 3. P2 — the pilot (after gate G1)

### 3.1 Launch

- One launch through `tools/ms/r4_launch.py`: wave `r4` (280 runs), at most 40 single-threaded workers, in tmux.
- Recorded: `nproc`, load, free disk, HEAD, `git diff --stat <code commit> HEAD` and `git status --porcelain`. Nothing changes after the launch.
- Crash rule as before:
  - an infrastructure kill is re-run once, after moving the attempt to `results/ms_r4/pilot/crashed/`;
  - any other nonzero exit counts as a failed run and is reported, not re-run.

### 3.2 Post-launch checks

Output: `results/ms_r4/pilot/launch_checks.json`. Any failure of a deterministic check is a stop-and-report; the statistical start-share flags are reported against their chance expectation, as in MS-R1..R3. Wall-clock columns (`update_wall_sec`, `verifier_sec` and the like) are excluded from every bit-identity comparison, here and in C-MS6.

- **Basics:** manifests at the launch commit and clean; files complete; the global-RNG assertions.
- **Start shares:**
  - deterministic, from the recorded probabilities: at every update the probability mass of the tail stratum equals lambda_T, and for `fb_a0`, `fb_r` and `fb_rg` the masses of the near-tie and middle strata equal lambda_P and lambda_M (to 1e-12);
  - statistical, as `tools/ms/launch_checks.py` (`start_share_tests`) does: z-scores of the realised counts per (stage, block, stratum) against the probabilities in force, and the number of |z| > 3 next to its expectation under the null. Flags are not failures by themselves, as in the earlier rounds;
  - for `fb_rx` and the E1 arms, the realised near and middle shares are reported.
- **Scale:** the applied scale equals the D2 schedule at every terminal-stage update of the E1 arms. It is 1.0 throughout stage 1 and throughout the E2 arms.
- **C-INIT:** the initial actor and critic weights are identical across `t1_st_s16`, `relu_st_s16` and `lrsm_st_s16` of every (q, seed) (`init_state_sha256`).
- **C-FB25:** within every (q, seed), `fb_a0`, `fb_r`, `fb_rg` and `fb_rx` are bit-identical through local update 25:
  - the u25 export, the per-update series and stream positions of updates 1-25, and the check row at local 25 are the same;
  - the columns that record the configured alpha and the next-update digest are excepted;
  - their u50 exports differ.
- **C-X2:** within every (q, seed), `fb_rx` equals `t1_st_s16` bit for bit through local update 1201:
  - exports u25 ... u1200, the per-update series and stream positions of updates 1-1201, and the check rows' common columns through local 1200 are the same;
  - their u1225 exports differ.
- **Sampling weights:** for `fb_r` and `fb_rg`, `probs_next` is consistent with `p_digest_next` at every check except the last one of the stage (there the runner writes an empty `p_digest_next`).

### 3.3 Offline curves and analysis

**Curves.** Run `tools/ms/r4_curves.py` on all 280 runs into `results/ms_r4/curves/`, with C-OFF on every run; a mismatch is a stop-and-report.

**E1 analysis.** `tools/ms/r4_e1_analysis.py` writes to `results/ms_r4/analysis/e1/`:

- tables: the per-run table; `criterion_e1.csv` (E1-a, E1-b and E1-c per q, and the verdict); the descriptive paired tables; failures with Clopper-Pearson intervals; the failure diagnostics; the cost;
- figures:
  - |peak error| per run by arm and q, with the 0.04 line;
  - paired differences with their intervals;
  - e_hat_2(d) against e2*(d) for |d| <= 30 per arm (seed median with bands);
  - learning curves (median and inter-quartile band) of eta_2/DW and |peak error|;
  - the trajectories of the failed runs.

**E2 analysis.** `tools/ms/r4_e2_analysis.py` writes to `results/ms_r4/analysis/e2/`:

- tables:
  - the per-run table (A, the secondaries, gates, stage 1, freeze metrics, cost);
  - `criterion_e2.csv` (rho_X with 95 % and 97.5 % intervals; E2-a, E2-b and E2-c per q; the verdict; the decision by the order of D6);
  - the `fb_rx` comparison (descriptive);
  - `sampling_weights.csv` (per check);
  - the strata shares per update;
- figures:
  - eta_2/DW curves (median and band, on log and linear scales) per arm and q;
  - per-seed A paired against `fb_a0`;
  - rho_bar and g_bar over the checks (seed-mean heat maps, bins × checks);
  - the realised strata shares over the updates.

**Blind recomputation.** `tools/ms/r4_blind_e1.py` and `r4_blind_e2.py` recompute the criterion tables from the per-run tables and the curve files without importing the analysis tools; they must agree to 1e-12.

## 4. P3 — reports, fact-check, push

### 4.1 Reports (`reports/ms/r4/`)

- `00_housekeeping.md`, `01_preamble_numbers.md`, `01b_curve_tool.md`, `02_preregistration.md`, `03_checks.md`.
- `04_e1_actor.md`, in this order:
  - the checks;
  - per arm, the configuration and code version;
  - the per-seed tables;
  - the target and the controls;
  - the failures, the decomposition and stage 1;
  - the learning curves and the cost;
  - the predictions against the outcome;
  - the combined-change caveat and the power caveat of adjustment 6.
- `04_e2_feedback.md`, in this order:
  - the checks;
  - per arm, the configuration and code version;
  - the per-seed tables and the learning curves;
  - the target and the decision by the order of D6;
  - `fb_rx` and the secondaries;
  - the sampling weights and the strata shares;
  - the cost;
  - the predictions against the outcome;
  - what the statement covers: sampling feedback only. The stop rule and polishing were not tested again; MS-R1's record stands (at rho_2 = 0.05 the terminal-stage stop fired in 0 of 100 runs).
- `05_decision_inputs.md`: E1 and E2 against the coworker's targets, and what the plan of Appendix A implies next for each outcome. The options are: lock one complete scheme and confirm it on 40501-40520; run the combination on development seeds first; or keep v2.0 and report both tries as ablations. Nothing in this file decides.
- `summary.md`.
- `handoff.md`: a message to the coworker (Chinese, then English) with the outcome against the coworker's targets and the links. Nothing is sent, and no name or contact detail is written.

Every table is generated by a script under `reports/ms/r4/report_scripts/` from the CSVs, and the key figures are embedded. Update the MS-R4 row in `reports/ms/README.md` and add a dated section at the top of `docs/STATE.md`.

### 4.2 Fact-check and tracking

- An independent read-only fact-check checks every number in the reports against the CSVs, including coverage of every D5 item and of the deliverables the coworker listed.
- Ledger: `reports/ms/r4/pi_record/01_factcheck.md`. Corrections go in before the push.
- Commit the lightweight records under the tracking rules of `reports/ms/README.md`.

### 4.3 Push

Push `ms-r4` (fast-forward, no force) once the reports are committed. Do not touch `main`, do not create tags, and open no pull request.

## 5. STOP

Stop after the §4.3 push. No fresh seeds, no lock, no confirmation, no combined run, no T = 3 run, no second pilot, and no change after the pre-registration. The next round is decided by the PI with the coworker.

## Appendix A — the coworker's decision (verbatim)

Text of `Multistage1011261.docx` (SHA-256 `699f73f30010bc59777469c8cada4c8f81c2fa3bbde6fa39e28349b1126e10e8`), paragraphs in order. Bold and highlighting are not reproduced.

```text
我们还是可以尝试一下降低stage-2的偏差，另外确定development DP-BR在训练中的作用。我现在的想法是先做下面这两项尝试，完成后就确定 T=2 协议、开始 T=3的实验验证。

下面的具体实验细节供参考，你可以根据你的经验做调整。

锁定的 v2.0 保留作基准。两项实验都跑 q=50、60，使用新的开发种子 10511–10530（每个 q 20 个，组间配对）。30501–30520 已用于此前确认，不再算新种子；40501–40520 留到候选方案锁定后确认。

1. 尝试降低 stage-2 平局点偏差。 以 relu_st_s16 为起点，只做一个预先固定的稳健性候选：隐藏层改用 LeakyReLU，均值输出改为不依赖硬 clamp 的平滑、有界映射。其余保持 MS-R3 的分层起点、s16 噪声着陆和 2800 次 stage-2 更新。用同一批种子重跑 relu_st_s16 和 t1_st_s16 作对照。目标是在两个 q 上，平均 |peak 误差| 均 ≤0.04，且相对 t1_st_s16 的配对改善在 95% 区间内成立；开发实验的 G-A、G-F、G-N、G-S 不出现失败。现有 ReLU 失败的成因尚未确定，因此即使候选有效，也只报告组合改动的效果，不归因于某一个改动。未达目标就停止这条线。

2. 检验 development DP-BR 能否实际帮助 PPO训练。 固定原 t1 actor、训练设置和预算（stage-2 1600 次、stage-1 600 次），关闭动态停止和 polishing，只比较采样反馈：①固定分层采样（α=0）；②现有残差引导采样（α=0.5）；③把 development DP-BR 的逐状态偏离收益汇总到各状态 bin，与现有动作残差一起决定下批起点（α=0.5）。三组使用相同种子、检查频率和状态分层比例，并记录每次检查后的采样权重；①也运行同样的检查，以便比较计算成本。闭式解仅用于离线评价，不进入采样。目标是相对①，在两个 q 上使固定检查点的 stage-2 η₂/DW 学习曲线面积平均下降至少 20%，配对 95% 区间支持改善，同时最终完整门槛和 stage-1 偏离收益不变差。②若已达目标，就不需要③来证明反馈有效；均未达标则只把 DP-BR 保留为验证器，不声称它改善了训练。

请交付每组的配置和代码版本、逐种子结果、配对区间、peak、学习曲线、最终各项门槛及运行成本。开发结果通过后，先锁定要采用的一个完整方案，再用 40501–40520 做 fresh-seed 确认；两项各自有效也不能直接假定合用有效，合用需先在开发种子上单独验证。确认不过就保留 v2.0，并把两项结果作为消融和局限报告。
```
