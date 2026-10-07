# Multistage round MS-R3: does the actor's resolution at the kink set the stage-2 peak bias? Kink-capable actors (ReLU; tanh with a finer d input) against the current actor, crossed with the sampler and the noise landing — T=2, development seeds

MS-R2 is closed (`reports/ms/r2/summary.md`, `reports/ms/r2/05_decision_inputs.md`; `origin/ms-r2` at `e8eb9a08`). The noise landing lowered sigma_2(0) as designed (about fourfold at s = 16) and the smoothing part by 0.64-1.39 effort units (s = 4 and 16), but the learned tie effort did not follow: the remainder rose by +0.30 to +1.40 in every cell and no row of the primary criterion was met. The PI's reading after MS-R1, that the noise floor is what binds, is **withdrawn**: the tie effort is essentially invariant to the policy noise (`05_decision_inputs.md` section 2.2, seed means of e_hat_2(0) at s = 1 / 4 / 16):

| q | bin-balanced | stratified |
|---|---|---|
| 50 | 66.04 / 66.44 / 66.10 | 66.97 / 66.87 / 67.19 |
| 60 | 56.00 / 55.33 / 55.58 | 55.64 / 55.99 / 55.91 |

The pathwise fine-tuning of R1 and R2b (method 5: exact gradients of the expected payoff at the Beta means, no policy noise at all) had already failed to lower the peak error against its matched PPO controls, although it lowered RMSE_pos and the first-order residual (`reports/t2_refine_100526/100526report.md` section 4.5); so had R1's concentration annealing (section 4.4).

**The PI's new reading (post hoc; this round tests it): the tie deficit is set mainly by the actor's resolution at the kink.** The equilibrium e2*(d) is a tent with a cusp at d = 0 (the triangular difference density has its kink at the tie). The current actor is a 2 → 64 → 64 tanh network whose d input is d / B (B = 100 + 2q). A tanh unit with first-layer weight w on d / B bends over |d| ≈ B / w, about a hundred units of d or more at the weights the supervised fit of item 4 reaches, while the observed deficit is about five units of d wide (item 2). The evidence so far:

1. The tie level does not respond to removing the noise (MS-R2, R1 annealing, R1/R2b pathwise), and does respond to more weight near the tie (R2b, R2c, MS-R1, MS-R2: in MS-R2 the stratified sampler moves the seed-mean tie effort by +0.4 to +1.1 effort units at q = 50, by −0.4 to +0.7 at q = 60) and to more training (MS-R1: `MS_base2400` against `parents_A`, 1600 → 2400 updates, +0.88 / +0.50 at q = 50 / 60).
2. Measured in units of d, the deficit is about the same at both q: gap / tent slope (slope = e2*(0) / 2q) averages 4.85 at q = 50 and 5.33 at q = 60 over the six MS-R2 arms. This is what a fixed input-space resolution would give.
3. An additive model (gap = fit part + smoothing part, the reading of MS-R1) misses MS-R2: at s = 16 it predicts gaps of 2.57 / 1.36 / 1.66 / 1.71 (bin-balanced q50, q60; stratified q50, q60) against observed 3.90 / 2.75 / 2.81 / 2.42. A quadrature model, in which the fit rounding and the noise smoothing combine like two kernel widths, gap(s) = sqrt(F^2 + smoothing(s)^2) with F fixed from s = 1, predicts 3.51 / 1.95 / 2.44 / 2.36: closer in all four cells, still below the observed gap in each. Under that model, while the fit term F dominates, removing the noise barely moves the peak; once F is small, the noise term dominates and the noise landing should work.
4. **PI-side sandbox (not repository evidence).** A numpy copy of the actor (2 → 64 → 64, orthogonal init gain sqrt(2), zero output layer, mean = sigmoid(z0), effort = 100 · mean, input (1, d/B)) was fit by supervised MSE to the closed-form tent at the RL budget: 56,000 Adam steps (the 2400 + 400 terminal updates × 20 minibatch steps; `t2_minibatch_steps` = 56,000 in the MS-R2 runs), LR 3e-4 then linear to 3e-5 over the last 8,000, minibatch 256, global gradient-norm clip 0.5, starts drawn bin-balanced or stratified (lambda_P 0.35), 3 seeds per cell. Median tip deficit e2*(0) − e_hat(0), effort units, q = 50 / 60:

   | actor | bin-balanced | stratified | RMSE_pos, bin-balanced |
   |---|---|---|---|
   | tanh, input d/B (current) | 1.90 / 1.49 | 0.76 / 0.76 | 0.54 / 0.39 |
   | ReLU, input d/B | 0.49 / 0.04 | 0.08 / 0.03 | 0.08 / 0.02 |
   | tanh, input 10 · d/B | 0.54 / 0.45 | 0.44 / 0.39 | 0.07 / 0.07 |

   Even with exact targets and no RL noise, the current actor leaves 1.5-1.9 effort units at the tip under bin-balanced starts (one of the three q = 60 seeds left 6.4). Its largest first-layer weight on the d input (in units of d / B) ended at a median of 1.7 at both q (1.4-1.5 under stratified starts), so its sharpest unit bends over about 120-130 units of d. Tip weighting halves the deficit. Both kink-capable variants remove most of it; the ReLU actor does so with smaller d-weights (0.76-0.92), since a ReLU unit has a kink at any weight.

   The sandbox script and its output accompany this prompt: `sandbox_fit_tip.py` (SHA-256 `a02308287ba149c63678063fc995a5c7296d32150a392d6fb6134bba86269f36`) and `sandbox_fit_tip_results.jsonl` (`3038885bc3ff69b688425a1728a4bac21dcbec7fe0305c7a4f00cd5fa5257f8a`). If they are provided, store them unchanged in `reports/ms/r3/pi_record/` and record the hashes; they are not re-run and are not evidence.

Unless another path is named, the numbers of items 1-3 were computed on the PI side from the arm means of `results/ms_r2/analysis/per_run.csv` of `origin/ms-r2` (columns `e2_at_0`, `gap`, `smoothing`, `g2_at_0`; the `MS_base2400` and `parents_A` rows for the budget comparison). P1 reproduces them (§2.4). The supervised screen of P1 is the repository version of item 4.

Also carried from MS-R2: R0 = r_2(0)/s_2 estimates the peak error without the closed form. Over the 13,440 terminal-stage exports of MS-R2 its Spearman correlation with |peak error| is 0.991 at each q, and R0 / |peak| has median 0.603 / 0.685 against the linearised 2k/(2k + a) = 0.588 / 0.673, a = DW/(4q^2) (`reports/ms/r2/05_decision_inputs.md` section 5.1). It is reported in this round as the terminal-stage tie-accuracy metric (not a stop rule).

Steps:

1. **P0** housekeeping (§1).
2. **P1** the actor variants in code, tests, the offline supervised screen, the RL-actor diagnostics on existing exports, the reproduction of the preamble numbers, the pre-registration and push (§2). If the premise check of §2.5 passes and every test, check and the review are clean, continue to P2 without waiting for me; otherwise stop and report.
3. **P2** the RL pilot (up to 240 runs), its checks and pre-registered analysis (§3).
4. **P3** reports, fact-check, push, then **STOP** (§4, §5).

Repository `/home/fjiang4/tournament_experiment`; Python `/home/fjiang4/tournament_experiment/.venv/bin/python` only; `pytest tests` (never a bare `pytest`); every run single-threaded (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) in tmux; do not kill jobs you did not start. Work in your session's own worktree on a new branch `ms-r3` created from `origin/ms-r2` (`e8eb9a08`); record the worktree path. Results root `results/ms_r3/`; reports `reports/ms/r3/`; new tools under `tools/ms/` (prefix `r3_`); new tests `tests/test_ms_r3_*.py`.

Everything from the earlier rounds still applies: provenance discipline (every number in a report cites a path); the closed-form equilibrium enters evaluation, reporting and the offline supervised screen only — never RL training, the sampler, the schedule or any criterion of the runs; no deletion or overwriting of results or checkpoints; the paper-registry test and its data unchanged; no `.pt`, `train_history.json`, weight exports or freeze arrays in git; `clamp_likelihood` stays `density`. Nothing changes in `protocols/`, `run/run_v2_T2_locked.py`, `run/run_v2_stagewise.py`, `run/v2_rollout.py`, `utils/v2_continuation.py`, `utils/v2_metrics.py`, `utils/dp_br_verifier.py`, `envs/` or `tools/v2/`. `agents/` may gain code only behind new keys whose absence leaves every existing path bit-identical (C-R6 and C-MS5 are the proof). Pushing is authorised only where §2.6 and §4.3 say so; `main` and tags are not touched. Save this prompt verbatim as `reports/ms/r3/pi_record/20_ms_r3_prompt.md` before anything else.

**If any step fails, or anything does not match this prompt, stop at that point and report. Do not work around it.**

---

## 0. Decisions (binding)

### D1. Scope

- T=2, development seeds 10501-10510 × q ∈ {50, 60}, 20 runs per arm. No fresh seeds (40501-40520 stay reserved), no lock, no confirmation, no T=3 run, no protocol change. The round ends with a pilot report and my decision.
- As in MS-R2: no development stop, no classification-driven change of the sampler, no polishing. The checks still run every K = 25 and are recorded. The would-fire record uses the thresholds of `reports/ms/r1/prereg_parameters.json` (passed explicitly, SHA-256 recorded), which also supplies K, M, the EMA weight and the near-tie half-width.
- Only the actor changes. The critic keeps the current architecture and its input (d / B).

### D2. Actor variants

| key | hidden activation | d component of the actor's input |
|---|---|---|
| `t1` (control; the current actor, no new key) | tanh, tanh | d / ((t − 1) B), as now |
| `relu` | ReLU, ReLU | d / ((t − 1) B), as now |
| `t10` | tanh, tanh | 10 · d / ((t − 1) B) (the stage feature is not scaled) |

- Everything else as the current `BetaActor` (`agents/ppo_curriculum.py`): 2 → 64 → 64 → 2, l1 and l2 orthogonal with gain sqrt(2) from the explicit torch generator, biases and output layer zero, mean = clamp(sigmoid(z0)), concentration = c_min + softplus(z1), times `conc_scale`.
- **The variants draw their initial weights with the same generator calls in the same order as the current actor**, so that within a (q, seed) the initial l1 and l2 matrices, and the critic's initial weights, are identical across `t1`, `relu` and `t10` (check C-INIT).
- The input scaling of `t10` happens inside the actor's forward, so every caller (rollout, opponent, frozen snapshot, policy functions of the verifier, continuation table, numpy reload) sees the same function.
- The variant is recorded in `run_config.json`, `manifest.json` and every weight export of a `relu` or `t10` run. A `t1` run writes no variant field anywhere (absence means `t1`, as in every existing export), so its configs, exports and arrays are those of the current code. The numpy reload (`mean_effort_numpy` or a variant-aware equivalent) reproduces each variant's forward to float32 rounding. The opponent and the frozen snapshot carry the variant.
- **No silent tanh reload.** Several loaders rebuild the actor from an export with the tanh, d / B forward hard-coded (`agents/ppo_curriculum.py:mean_effort_numpy`, the `BetaActor` built by `tools/ms/replay_dev_rule.py` and the tools that reuse it, the loaders in `tools/v2/`). Every loader used in this round reads the variant from the export and builds that forward, or refuses an export whose variant is not `t1`. The loaders in `tools/v2/` stay unchanged and are not used on MS-R3 exports. Test: a `relu` and a `t10` export fed to each loader used in this round give that variant's forward or an error, never the tanh d / B forward.
- Implement the variants either in `agents/ppo_curriculum.py` behind new keys (absent = bit-identical) or in a new module used only by the MS runner; record which.

### D3. Schedule (unchanged from MS-R2 D2)

- Terminal stage, fixed 2800 updates:
  - local 1-2000: LR 3e-4, scale 1;
  - 2001-2200: LR 3e-4, concentration scale linear 1 → s (`Run.conc_scale_for` form, scale 1 at 2001, s at 2200);
  - 2201-2400: LR 3e-4, scale s;
  - 2401-2800: LR linear 3e-4 → 3e-5, scale s.
- The scale is set on the live actor and the lagged opponent, carried by the refresh and the frozen snapshot. For s = 1 no scale is ever set.
- Stage 1: fixed 600 updates, LR linear 3e-4 → 3e-5 over 1-600, no rule, scale reset to 1 at stage-1 entry.
- Everything else as MS-R2's `NL_*` arms.

### D4. Arms (twelve, 20 runs each)

actor ∈ {`t1`, `relu`, `t10`} × starts ∈ {bin-balanced (the locked `StartSampler.balanced` call), stratified (MS-R2 `NL_st` settings: lambda_P 0.35, alpha_global 0.5, EMA beta 0.5, near-tie half-width 20, lambda_T the bin-balanced tail share, the MS-R1 global-block sampler at every update)} × s ∈ {1, 16}.

- Names: `{actor}_{bb|st}_s{1|16}`, e.g. `t1_bb_s1`, `relu_st_s16`, `t10_bb_s16`.
- The four `t1` arms are MS-R2's `NL_bb_s1`, `NL_bb_s16`, `NL_st_s1` and `NL_st_s16` re-run on the new code: check C-MS5.
- If the premise check (§2.5) drops a variant, its four arms are not run.

### D5. What is measured

Everything of MS-R2 D4, at the terminal freeze on both tiers and along the run from local update 1800 to 2800, plus:

- **Tie metrics.** R0 at the freeze on both tiers, and R0 / |peak| against the linearised 2k / (2k + a). The location-free peak and its argmax d (where the learned kink sits). The symmetry error max |e_hat_2(d) − e_hat_2(−d)| over |d| < 2q on the recovery grid.
- **Resolution metrics.** The effective rounding width w_eff = gap / (e2*(0) / 2q) in units of d. The first-layer weights on the d input in units of d / B (for `t10`, ten times the stored weight), max |w| and the distribution, at every weight export.
- **Per-stratum errors** (near-tie / middle / tail × d < 0 / d > 0), as in MS-R1 A3(b).
- **Decomposition** (smoothing part, remainder) and sigma_2(0).
- **Along the run:** R0, R, Delta_2, C_2, w_eff, the PPO diagnostics per segment.
- **Stage 1,** paired against the same sampler's and s's `t1` arm and against `rehearsal_v2_0`, descriptive: G-S, S1, G-F, G-N(Gmax), the stage-1 error, R_1. The actor is shared across stages, so the variant can change stage 1.

**The reports must cover every item of this list.** MS-R2's fact-check ledger lists items that were left thin; do not repeat that. Embed the key figures in the reports instead of only naming them.

### D6. Criterion, comparisons, predictions (pre-registered; descriptive, not gates)

- **Primary.** For each variant v ∈ {`relu`, `t10`}, each starts and each s ∈ {1, 16}, the measure is |peak error| at the terminal freeze, paired by (q, seed) against the `t1` arm with the same starts and s (8 rows).
  - (a) The 95 % percentile bootstrap interval of the mean paired difference lies below 0 at both q.
  - (b) No run that passes G-A with its G-N(eta) part under `t1` fails it under v.
  - Bootstrap: 10,000 resamples, a fresh `numpy.random.default_rng(20261008)` per (q, statistic), in table order.
- **Secondary** (same statistics, descriptive):
  - the paired changes of the gap, the smoothing part, the remainder, sigma_2(0), RMSE_pos, tail mean, eta_2, R0, R and w_eff;
  - within each (actor, starts): the noise-landing effect s = 16 against s = 1 (C-NL pairs), with the transmission ratio = (change of the gap) / (change of the smoothing part), where 1 means the whole smoothing reduction reached the gap;
  - the interaction (v_s16 − v_s1) − (t1_s16 − t1_s1) on |peak| and on the gap;
  - the starts effect within each actor;
  - every arm against `parents_A` with MS-R1's criterion;
  - the stage-1 comparisons of D5.
- **The quadrature check** (the post-hoc model of the preamble, now written down before the runs). Per (actor, starts, q), from arm means:
  - F^2 = gap(s1)^2 − smoothing(s1)^2 (if negative: F = 0, flagged);
  - quadrature prediction gap(s16) = sqrt(F^2 + smoothing(s16)^2);
  - additive prediction gap(s16) = gap(s1) − (smoothing(s1) − smoothing(s16));
  - report the observed gap(s16) next to both and which is closer, descriptive.
- **Predictions written before the launch:**
  - (P1) `relu` and `t10` have lower |peak| than `t1` at s = 1 under both starts.
  - (P2) For `relu` and `t10` at s = 1, the remainder is small against the smoothing part, i.e. e_hat_2(0) is close to e_sigma(0).
  - (P3) For `relu` and `t10`, the noise landing transmits (transmission ratio near 1), so that at s = 16 |peak| approaches the smoothing floor sigma_2(0) / (sqrt(pi) q), relative to e2*(0) (in MS-R2 at s = 16: about 0.7 % at q = 50 and 0.6 % at q = 60, against an observed |peak| of 4.0-5.6 %).
  - (P4) RMSE_pos is lower for `relu` and `t10`.
  - The RL fit will be worse than the supervised screen; these are expectations, not tests. No selection rule follows; the decision is mine.

### D7. Reference roots

Every tool takes these as explicit arguments and records them; a missing reference file is a stop-and-report.

- **MS-R2 runs** (for C-MS5 and §2.3 (b)): `/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857/results/ms_r2/pilot`. This is the worktree in which MS-R2 ran (its fact-check ledger names it); verify the path.
- **MS-R1 runs** (for §2.3 (b)): pilot `/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857/results/ms_r1/pilot`; base `/home/fjiang4/tournament_experiment/.claude/worktrees/ms-r1-multistage-development-afebf2/results/ms_r1/base`.
- **`parents_A`:** `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine/parents_A`.
- **`rehearsal_v2_0`:** `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0`.
- **The C7 canonical worktree:** `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2`.

---

## 1. P0 — housekeeping

Write `reports/ms/r3/00_housekeeping.md` as you go (commands, hashes, verbatim outputs).

1. `git ls-remote origin` for `main`, `ms-r2` and the tags; verify `origin/ms-r2` = `e8eb9a08`; create `ms-r3` from it in your session's worktree. Do not touch `main` or any other worktree; record (do not fix) local refs that are behind.
2. Verify the reference roots of D7 and what this round reads from them (the MS-R2 weight exports, freeze arrays, per-update series and check tables; the MS-R1 exports); record the counts.
3. Record `nproc`, load, free disk, and the Python and package versions against the embedded record.

## 2. P1 — code, tests, offline work, pre-registration (on `ms-r3`)

### 2.1 Code

- The actor variants of D2 (in `agents/ppo_curriculum.py` behind new keys, or in a new module), wired into `run/run_ms_stagewise.py` through a new optional config key. Absent key = the current actor, bit for bit.
- The variant is recorded in configs, manifests and exports; a variant-aware numpy reload is added.
- `tools/ms/`:
  - the twelve arms and a wave `r3` in the config builder and the launcher;
  - `r3_supervised_screen.py` (§2.3 (a));
  - `r3_actor_diagnostics.py` (§2.3 (b));
  - `r3_launch_checks.py` (C-INIT, C-NL, C-MS5, the scale record, and the usual manifest, files, global-RNG and start-share checks);
  - `r3_analysis.py` and an independent `r3_blind_criterion.py` (D5, D6).

### 2.2 Tests (`tests/test_ms_r3_*.py`, plus every existing suite)

- **Defaults.** Without the new key everything is bit-identical to `e8eb9a08`: every MS-R1 and MS-R2 test passes unchanged, and C7 passes.
- **Variants.**
  - The forward of each variant matches D2.
  - The `t10` scaling applies to the d component only, and everywhere the actor is evaluated.
  - The initial l1 and l2 weights and the critic's are identical across variants for a given (q, seed) (C-INIT).
  - The numpy reload reproduces the torch forward of each variant to float32 rounding.
  - The opponent, the refresh and the frozen snapshot carry the variant.
  - The stage-1 continuation table from a variant's frozen snapshot is consistent with direct evaluation of that snapshot's mean.
- **Reduced-budget runs.** C-NL for each variant; a reduced T=3 smoke with `relu` and one with `t10`, so that nothing at T=3 breaks.
- **Tools.** The screen and the diagnostics tool on reduced budgets; the analysis and the blind recomputation on synthetic rows; the launcher (the configs of an actor group differ only in the variant key, the sampler and the scale key, as intended).
- **Full suite** `pytest tests`: everything passes apart from the known `test_registry_canonicalization`; record the summary line.

### 2.3 Offline work (no RL training)

**(a) Supervised screen.** For each variant in D2, with the real actor class and the runner's initialisation for (q, seed):

- **Target:** the Beta-mean effort fit to the closed-form tent e2*(d) on D_2 by MSE in effort units. This is an evaluation-side diagnostic, as Pilot 4's representation floor was.
- **Optimisation, matched to the RL terminal stage:** 56,000 Adam steps (the record's betas and eps), LR 3e-4 for 48,000 then linear to 3e-5 over 8,000, minibatch 256, global gradient-norm clip at the record's `max_grad_norm`.
- **Inputs:** d drawn from the start distribution. Bin-balanced: a bin uniformly, then a position uniformly within it, as `StartSampler.balanced`. Stratified: the stratum shares of the `NL_st` arms (lambda_P 0.35 on the near-tie bins of half-width 20, lambda_T the bin-balanced tail share, the rest on the middle) with uniform weights within each stratum, i.e. `stratified_bin_probs` with alpha 0, since a supervised fit has no verifier. The d stream comes from its own generator, seeded from (q, seed) and recorded.
- **Grid:** 10 seeds × q ∈ {50, 60} × both starts, for each variant.
- **One extended cell (reported, not gated):** `t1`, bin-balanced, 10 seeds × both q, 224,000 steps (LR 3e-4 for 216,000, then linear to 3e-5 over 8,000), checkpoints every 56,000. It shows whether the current actor closes the tip with four times the budget, i.e. whether more training alone is a remedy.
- **Reported at steps 16k, 32k, 48k and 56k:** the tip deficit e2*(0) − e_hat(0), RMSE_pos, tail mean, w_eff, max |w| of the first layer on the d input (units of d / B).

Output `results/ms_r3/supervised_screen/`, report `reports/ms/r3/01_supervised_screen.md`.

**(b) RL-actor diagnostics on existing exports** (MS-R1 and MS-R2 runs, every 25 updates):

- the first-layer weights on the d input (max |w| and the distribution);
- w_eff at every export (e_hat_2(0) from the export, the closed form for e2*(0));
- their relation over training, per arm;
- whether the RL actors sit in the small-weight regime that the screen associates with a rounded tip (this regime matters for tanh units; a ReLU unit has a kink at any weight).

Report `reports/ms/r3/01b_rl_actor_diagnostics.md`.

### 2.4 Reproduce the preamble numbers

From `results/ms_r2/analysis/per_run.csv` of `origin/ms-r2`: the tie-effort table, the sampler and budget differences of item 1, the gap / slope averages (4.85, 5.33), and the additive and quadrature predictions at s = 16. Add a section to `01_supervised_screen.md`. A number that differs is reported as such (the preamble is not edited); this is not a stop condition.

### 2.5 Premise check (pre-registered; decides what P2 runs)

From the supervised screen at 56,000 steps, bin-balanced starts:

- (i) the median tip deficit of `t1` is at least 1.0 effort unit at both q;
- (ii) for each variant v, the median tip deficit of v is at most half that of `t1` at both q.

Outcomes:

- If (i) fails, or both variants fail (ii): **stop and report**.
- If one variant fails (ii): drop its four arms from P2 and record why.
- The stratified cells are reported, not gated.

### 2.6 Independent review, pre-registration, push

- **Review.** One read-only review of the diff by an agent that did not write it: conformance with D2-D6, the initialisation identity, bit-identity of the defaults, the variant semantics. Findings and dispositions go in `reports/ms/r3/03_checks.md`.
- **C-R6.** The unchanged `run/run_v2_T2_locked.py` from `ms-r3`, q ∈ {50, 60} × seeds 10501-10510, into `results/ms_r3/v20_reproduction/`, compared with `rehearsal_v2_0` by `tools/v2/cr2_compare.py`: 20/20 identical, or stop and report.
- **Pre-registration.** `reports/ms/r3/02_preregistration.md`:
  - D1-D7 as implemented;
  - the premise-check outcome and the arms that will run;
  - the criterion, the secondary analyses and the quadrature check;
  - the predictions;
  - the reference roots, the test summary and the launch plan.
- **Commits.** Commit code, tests, the offline outputs and reports, and the pre-registration (**code commit**); later commits touch only `results/` and the report folder.
- **Push** `ms-r3` (fast-forward, no force). If every test, check and the review are clean and the premise check passed, continue to P2 without waiting for me.

## 3. P2 — the RL pilot

### 3.1 Launch

- One launch through the launcher: wave `r3`, the arms kept by §2.5 (240 runs if both variants pass), at most 40 single-threaded workers in tmux.
- Recorded: nproc, load, free disk, HEAD, `git diff --stat <code commit> HEAD` and `git status --porcelain`. Nothing changes after the launch.
- Output `results/ms_r3/pilot/q*/seed*/<arm>/`.
- Crash rule as before: an infrastructure kill is re-run once after moving the attempt to `results/ms_r3/pilot/crashed/`; any other nonzero exit counts as a failed run and is reported, not re-run.

### 3.2 Post-launch checks (`results/ms_r3/pilot/launch_checks.json`; any failure is a stop-and-report)

- Manifests at the launch commit and clean; files complete; global-RNG assertions.
- Start shares per stratum within 3 binomial SE (the stratified arms; the tail share lambda_T in force at every update).
- The applied scale equals the D3 schedule at every terminal-stage update and is 1.0 throughout stage 1.
- **C-INIT:** the initial actor and critic weights are identical across the variants of every (q, seed) (from an update-0 export, or a hash of the initial state recorded by the runner; record which).
- **C-NL:** within each (actor, starts, q, seed), the s = 16 run equals the s = 1 run bit for bit through update 2001, and its u2025 export differs.
- **C-MS5:** each `t1` arm equals the corresponding MS-R2 `NL_*` arm bit for bit over the whole run: every array of every weight export, the per-update series, stream positions after every update, the check rows' common columns, freeze arrays, the gate values; 80 of 80.

### 3.3 Analysis

`tools/ms/r3_analysis.py` → `results/ms_r3/analysis/`:

- the per-run table;
- the primary and secondary paired tables, `criterion.csv`, the quadrature check;
- the decomposition at the freeze and along the run;
- strata, stage 1, gates;
- figures:
  - paired differences per row;
  - e_hat_2(0), the smoothing part, the remainder and w_eff against updates 1800-2800 per arm;
  - e_hat_2(d) against e2*(d) near the tie (|d| ≤ 30) for each actor at the freeze, seed means with bands;
  - first-layer d-weights over training per actor.

`tools/ms/r3_blind_criterion.py` recomputes `criterion.csv` and the transmission and interaction tables from `per_run.csv` without importing the analysis tool (agreement to 1e-12).

## 4. P3 — reports, fact-check, push

### 4.1 Reports (`reports/ms/r3/`)

- `00_housekeeping.md`.
- `01_supervised_screen.md`, `01b_rl_actor_diagnostics.md`.
- `02_preregistration.md`.
- `03_checks.md`.
- `04_pilot.md`: checks first; the criterion; one section per actor; the noise landing within each actor; the quadrature check; the predictions against the outcome; stage 1; anomalies.
- `05_decision_inputs.md`: the twelve arms side by side; what the actor changes in each component of the gap and in global accuracy; whether the noise landing transmits under each actor; R0 as the tie metric; what is not settled.
- `summary.md`.

Add a row to `reports/ms/README.md` and a dated section at the top of `docs/STATE.md`. Every table is generated by a script under `reports/ms/r3/report_scripts/` from the CSVs, and the key figures are embedded.

### 4.2 Fact-check and tracking

An independent read-only fact-check of every number in the reports against the CSVs, including coverage of every D5 item. Ledger: `reports/ms/r3/pi_record/01_factcheck.md`. Corrections go in before the push. Commit the lightweight records with the usual tracking rules.

### 4.3 Push

Push `ms-r3` (fast-forward, no force) once the reports are committed. Do not touch `main`, do not create tags.

## 5. STOP

Stop after the §4.3 push. No fresh seeds, no lock, no T=3 run, no second pilot, no change after the pre-registration. The next round is mine to decide.
