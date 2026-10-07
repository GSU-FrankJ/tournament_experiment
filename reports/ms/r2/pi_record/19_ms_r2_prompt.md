# Multistage round MS-R2: is the stage-2 peak bias the policy-noise floor? A noise landing (concentration-scale ramp, hold, LR decay) crossed with the start sampler — T=2, development seeds

MS-R1 is closed (`reports/ms/r1/summary.md`, `reports/ms/r1/05_decision_inputs.md`; `origin/ms-r1` at `71c58904`): no arm met the pre-registered criterion, the terminal-stage stop fired in 0 of 100 runs at rho_2 = 0.05, and at matched budget (against `MS_base2400`) no interval of |peak error| excluded 0. What follows is the PI's reading of the MS-R1 data; this round tests it. Unless another path is named, every number below was computed on the PI side from `results/ms_r1/analysis/per_run.csv` of `origin/ms-r1` (columns `g2_at_0`, `e2_at_0`, `smoothed_e_pred_0`, `sigma_effort_at_0_t2`; rows `role = ms_arm`, 140 runs). P1 reproduces them with a committed script (§2.4) before anything is built on them.

1. **The d = 0 gap splits into a noise-smoothing part and a remainder.** gap = e2*(0) − e_hat_2(0) = [e2*(0) − e_sigma(0)] + [e_sigma(0) − e_hat_2(0)], where e_sigma(0) = `smoothed_e_pred_0` = (DW / 2k) · E[f_xi(n_L − n_O)], n the centred learned Beta noise at d = 0 (`run/run_v2_T2_locked.py:smoothed_share`): the tie first-order condition of the game both players actually play when their actions carry the policy's own noise. The triangular density f_xi has its kink at x = 0, so averaging the marginal benefit over the noise lowers it at the tie by E|n_L − n_O| / (4q^2), first order in sigma: the smoothing part equals e2*(0) · sigma_2(0) / (sqrt(pi) · q) to within 0.07 % in all 140 MS-R1 runs (ratio 0.9993 to 0.9997). Away from the tip f_xi is linear and the noise has no first-order effect, so this bias is local to the peak.

   | q | arm | sigma_2(0) | gap | smoothing part | remainder |
   |---|---|---|---|---|---|
   | 50 | `MS_base` (= `parents_A`, 1600 updates) | 2.92 | 4.59 | 2.30 | 2.29 |
   | 50 | `MS_base2400` | 2.53 | 3.71 | 2.00 | 1.71 |
   | 50 | `MS_s35a5` | 2.50 | 2.40 | 1.97 | 0.43 |
   | 60 | `MS_base` | 2.99 | 3.11 | 1.64 | 1.47 |
   | 60 | `MS_base2400` | 2.59 | 2.62 | 1.42 | 1.20 |
   | 60 | `MS_s35a5` | 2.57 | 2.54 | 1.41 | 1.13 |

   Effort units, means over the 10 development seeds; e2*(0) = 70 at q = 50 and 58.33 at q = 60.

2. **Only sigma_2(0) moves the smoothing part; the samplers move only the remainder.** Paired against `MS_base2400`, the four sampler arms change the smoothing part by at most 0.04 effort units in absolute value, and the remainder by −0.71 to −1.28 at q = 50 (every interval contains 0) and −0.44 to +0.05 at q = 60. The 800 extra updates (`MS_base2400` − `MS_base`) lowered sigma_2(0) by 0.39 / 0.40 and the smoothing part by 0.30 [0.28, 0.34] / 0.22 [0.21, 0.23] at q = 50 / 60 (PI-side percentile bootstrap, descriptive): resolved with 10 seeds, while the remainder (seed SD 0.6 to 2.3) is not. In 139 of the 140 MS-R1 runs the learned peak is below e2*(0); in `MS_s35a5` at q = 50 the median remainder is +0.14, i.e. the learned peak sits on e_sigma(0) and what is left of the bias, about 2.0 effort units (2.8 %), is the noise floor. This is why every signed-peak interval of MS-R1 lies below 0.

3. **R1's concentration annealing was not this experiment.** `A_anneal4` (`reports/t2_refine_100526/100526report.md` section 4.4) lowered sigma_2(0) from 2.795 to 1.432 (q = 50) and cut the predicted smoothing part by −1.132 [−1.197, −1.079] (q = 60: −0.8045 [−0.8345, −0.7759]), but the observed gap changed by −0.07658 [−1.025, 0.837] (q = 60: +0.6889 [0.2988, 1.137]): the remainder grew by about 1.06 / 1.49. That round used bin-balanced starts and ramped the scale over the same 400 updates in which the LR decayed to 3e-5. A candidate reading, not established: a smaller noise sharpens the target at the tip, and with little tip weight and a vanishing LR the fit does not follow. MS-R1 shows that near-tie weighting lowers the remainder. Tip weighting combined with a noise reduction at constant LR, followed by the decay, has not been run.

This round runs it: a noise landing (concentration scale ramped at constant LR, held, then the LR decay) crossed with the start sampler, with the stop, the classification and polishing switched off, exact prefix pairing, and the T=2 closed form as the yardstick. Alongside, it calibrates three closed-form-free stop criteria offline (report-only).

Steps:

1. **P0** housekeeping (§1).
2. **P1** code, tests, the reproduction of the decomposition, the offline stop-criterion calibration on MS-R1, the pre-registration, push (§2). If every test, check and the review are clean, continue to P2 without waiting for me; if anything fails, stop and report.
3. **P2** the 120-run pilot, its checks and pre-registered analysis, the calibration on the new trajectories (§3).
4. **P3** reports, fact-check, push, then **STOP** (§4, §5).

Repository `/home/fjiang4/tournament_experiment`; Python `/home/fjiang4/tournament_experiment/.venv/bin/python` only; `pytest tests` (never a bare `pytest`); every run single-threaded (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) in tmux; do not kill jobs you did not start. Work in your session's own worktree on a new branch `ms-r2` created from `origin/ms-r1` (`71c58904`); record the worktree path. Results root `results/ms_r2/`; reports `reports/ms/r2/`; new tools under `tools/ms/` (prefix `r2_`); new tests `tests/test_ms_r2_*.py`.

Everything from the earlier rounds still applies: provenance discipline (every number in a report cites a path); the closed-form equilibrium enters evaluation and reporting only, never training, the sampler, the schedule or any criterion; no deletion or overwriting of results or checkpoints; the paper-registry test and its data unchanged; no `.pt`, `train_history.json`, weight exports or freeze arrays in git; `clamp_likelihood` stays `density`. Nothing changes in `protocols/`, `envs/`, `agents/`, `run/run_v2_T2_locked.py`, `run/run_v2_stagewise.py`, `run/v2_rollout.py`, `utils/v2_continuation.py`, `utils/v2_metrics.py`, `utils/dp_br_verifier.py` or `tools/v2/`. Pushing is authorised only where §2.6 and §4.3 say so; `main` and tags are not touched. Save this prompt verbatim as `reports/ms/r2/pi_record/19_ms_r2_prompt.md` before anything else.

**If any step fails, or anything does not match this prompt, stop at that point and report. Do not work around it.**

---

## 0. Decisions (binding)

### D1. Scope

- T=2, development seeds 10501-10510 × q ∈ {50, 60}, 20 runs per arm. No fresh seeds (40501-40520 stay reserved), no lock, no confirmation, no T=3 run, no protocol change. The round ends with a pilot report and my decision.
- Off in every arm: the development stop, any classification-driven change of the sampler, targeted polishing. The development checks still run every K = 25 local updates and are recorded; the would-fire record uses the thresholds of `reports/ms/r1/prereg_parameters.json` (passed explicitly, its SHA-256 recorded), which also supplies K, M, the EMA weight and the near-tie half-width.

### D2. Terminal stage: fixed budget 2800 with a noise landing

Every arm trains the terminal stage for 2800 updates:

| local updates | LR | concentration scale |
|---|---|---|
| 1-2000 | 3e-4, constant | 1 |
| 2001-2200 (ramp) | 3e-4, constant | linear from 1 to s, in the form of `run/run_v2_stagewise.py:Run.conc_scale_for` with local_first 2001, local_last 2200 (scale 1 at update 2001, s at 2200) |
| 2201-2400 (hold) | 3e-4, constant | s |
| 2401-2800 (decay) | linear 3e-4 → 3e-5 (`lr_linear`, first 2401, last 2800) | s |

- The scale multiplies the Beta concentration (`agents/ppo_curriculum.py`, `BetaActor.conc_scale`; the Beta mean does not depend on it). It is set before every update on the live actor and on the lagged opponent (both players' noise shrinks), constant within the update (rollout and PPO update), as R1's `conc_anneal` did; the snapshot refresh carries it (`refresh_snapshot` copies it to the opponent). The frozen stage-2 snapshot keeps s; `sigma_effort_at_0_t2` and every concentration statistic at the freeze are the scaled values. For s = 1 no scale is ever set (the key is absent).
- Why a hold at constant LR: in R1 the scale reached its end value while the LR was already at its minimum; here the mean has 200 updates at 3e-4 after the ramp to re-equilibrate before the decay.
- Stage 1, after the freeze: as `MS_base` — fixed 600 updates, LR linear 3e-4 → 3e-5 over local 1-600, no rule. At the stage-1 phase entry the live actor's and the lagged opponent's scale are reset to 1.0 (before the phase-entry snapshot refresh); the stage-1 continuation table is built from the frozen stage-2 mean as before.
- Everything else as in MS-R1's `MS_base2400` / `MS_s35a5`: the embedded v2.0 record, PPO settings, 512 episodes per update, the five streams and the torch generator, the global-RNG hardening, weight exports every 25 updates, the verifier tiers, development checks every K = 25 (pure, no RNG), concentration limit 0.04, freeze evaluation on both tiers.

### D3. Arms (six, 20 runs each)

| arm | terminal-stage starts | s |
|---|---|---|
| `NL_bb_s1` | `bin_balanced`, the locked `StartSampler.balanced` call | 1 |
| `NL_bb_s4` | same | 4 |
| `NL_bb_s16` | same | 16 |
| `NL_st_s1` | `stratified_priority` with MS-R1 `MS_s35a5`'s settings: lambda_P 0.35, alpha_global 0.5, EMA beta 0.5, near-tie half-width 20, lambda_T the bin-balanced tail share | 1 |
| `NL_st_s4` | same | 4 |
| `NL_st_s16` | same | 16 |

- The stratified arms use the sampler of an MS-R1 global block at every terminal-stage update (focus f ∝ rho_bar_b over the non-tail bins, the EMA updated at every valid check, p_PM before the first check) under the fixed budget; no classification changes the sampler. How the runner expresses this (a new key, or the rule controller with stop and polishing disabled) is your choice, provided C-MS4 holds.
- Identities that must hold (checks of §3.2; compared: the weight exports, the per-update series of `ms_updates.csv` and `train_history.json` without wall-clock fields, the five stream positions after every update, and the columns of the check rows common to both runs):
  - **C-NL**: within each sampler and (q, seed), the s = 4 and s = 16 runs equal the s = 1 run bit for bit through update 2001 (the scale is still 1 there), and their u2025 export differs.
  - **C-MS3**: `NL_bb_*` equals MS-R1's `MS_base2400` through update 2001 (both run LR 3e-4 up to 2001; `MS_base2400` decays from 2002), and the u2025 export differs.
  - **C-MS4**: `NL_st_*` equals MS-R1's `MS_s35a5` through update 2001 where that run's terminal stage had no polishing block, and otherwise through the last update before its first polishing block.

### D4. What is measured

- At the terminal-stage freeze, on both tiers, per run: signed and absolute peak error; RMSE_pos; tail mean and max; eta_2 (both tiers) with G-A and G-N(eta); sigma_2(0); e_sigma(0) and the decomposition gap = smoothing part + remainder, in effort units and relative to e2*(0); the ratio smoothing part / [e2*(0) · sigma_2(0) / (sqrt(pi) · q)]; R0 = r_2(0)/s_2 and R; the per-stratum errors of MS-R1 A3(b) (near-tie / middle / tail × d < 0 / d > 0); start shares per stratum; raw-draw clamp counts.
- Along the run, at every check from local update 1800 to 2800: e_hat_2(0), sigma_2(0), e_sigma(0) (computed at the check with the same Beta-quantile computation as `smoothed_share`; reporting only), the decomposition, R0, R, Delta_2, C_2; and per segment (training, ramp, hold, decay) the PPO diagnostics (KL, clip fraction, advantage SD, value loss) and the scale actually applied.
- Stage 1, descriptive: G-S, S1, G-F, G-N(Gmax), the stage-1 error against `rehearsal_v2_0`, R_1.

### D5. Criterion, comparisons, predictions (pre-registered; descriptive, not gates)

- **Primary**, per sampler, for s ∈ {4, 16}: |peak error| at the terminal freeze, paired by (q, seed) against the same sampler's s = 1 arm. (a) The 95 % percentile bootstrap interval of the mean paired difference lies below 0 at both q. (b) No run that passes G-A with its G-N(eta) part under s = 1 fails it under s. Bootstrap: 10,000 resamples, a fresh `numpy.random.default_rng(20261007)` per (q, statistic), in table order.
- **Secondary**, same statistics, descriptive:
  - the paired changes of the smoothing part, the remainder, sigma_2(0), RMSE_pos, tail mean and eta_2;
  - the transmission ratio −(change of the gap) / (change of the smoothing part) per arm and q (1 = the whole smoothing reduction reaches the gap, 0 = the remainder offsets it);
  - the interaction (`NL_st_s` − `NL_st_s1`) − (`NL_bb_s` − `NL_bb_s1`) per (q, seed), on |peak| and on the remainder: does tip weighting let the mean follow the sharper target?
  - every arm against `parents_A` with MS-R1's criterion;
  - the s = 1 arms against their MS-R1 references (`NL_bb_s1` − `MS_base2400`, `NL_st_s1` − `MS_s35a5`: the effect of 400 more constant-LR updates).
- **Predictions, written before the launch**: sigma_2(0) at the freeze ≈ sigma_2(0) at s = 1 divided by sqrt(s) (report the ratio); smoothing part ≈ e2*(0) · sigma_2(0) / (sqrt(pi) · q) (report the ratio; expected within 0.5 %). At the planning value sigma_2(0) ≈ 2.5 for s = 1 this puts the floor at about 1.4 % / 0.7 % at q = 50 and 1.2 % / 0.6 % at q = 60 for s = 4 / 16. The open question is the remainder: if it does not grow, |peak| falls by about the smoothing reduction; R1 suggests it may grow under bin-balanced starts. No selection rule follows; the decision is mine.

### D6. Offline calibration of stop criteria (report-only; nothing in the runs depends on it)

Three closed-form-free candidates for the terminal stage, computed from the weight exports every 25 updates with the development verifier:

- **C1**: R0 = r_2(0)/s_2 (MS-R1 A3(a)).
- **C2**: the training residual at the tie, (e_sigma(0) − e_hat_2(0)) / e_sigma(0), reported together with the noise floor sigma_2(0) / (sqrt(pi) · q); both use only the game and the policy.
- **C3**: the deflated residual R_defl = max over the non-tail nodes of r_2(d) / (s_2 · |1 − J(d)|), where J(d) = ∂e_tilde_2(d)/∂e_opp is a central finite difference (h = 0.5 effort units) of the one-step best response at d with the opponent's action at −d shifted by ±h, maximising the same terminal one-step objective as the verifier but accurately (a dense effort grid with local refinement, not the development tier's effort grid, whose best-response error reaches 0.47 effort units). Nodes with |1 − J| < 0.1 are excluded and counted. Under symmetric errors r ≈ (J − 1) · delta, so R_defl estimates the policy error. Check J against the linearisation: J − 1 = −2k/(2k − a) on the d < 0 side and −2k/(2k + a) on the d > 0 side, a = DW/(4q^2).

For each candidate, per q: the Spearman correlation with |peak error| and with RMSE_pos (all exports with u ≥ 400, and the constant-LR exports separately); the distribution at the freeze; and, for a small threshold grid per candidate (chosen and recorded before the fire tables are computed), when a rule "candidate ≤ theta at M = 3 consecutive checks, K = 25" would fire, and the closed-form errors at the fire against the end of the run. P1 runs this on the 140 MS-R1 runs (20 `MS_base` and 120 pilot runs); P2 repeats it on the 120 MS-R2 runs, where the floor of C2 moves with s.

### D7. Gates reported

As MS-R1 D7: G-A and G-N(eta) on the final tier at the terminal freeze; G-F, G-N(Gmax), G-S and S1 at the end of stage 1; the v2.0 combination for information. No run-pass rule.

### D8. Reference roots

Every tool takes these as explicit arguments and records them; a missing reference file is a stop-and-report.

- MS-R1 pilot runs (`MS_base2400`, `MS_s35a5`, and the other arms for D6): `/home/fjiang4/tournament_experiment/.claude/worktrees/p2-gate-ms-base2400-e41857/results/ms_r1/pilot`.
- MS-R1 base wave (`MS_base`): `/home/fjiang4/tournament_experiment/.claude/worktrees/ms-r1-multistage-development-afebf2/results/ms_r1/base`.
- `parents_A`: `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine/parents_A`; `rehearsal_v2_0`: `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/rehearsal_v2_0`.
- The canonical worktree for C7: `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2`.

---

## 1. P0 — housekeeping

Write `reports/ms/r2/00_housekeeping.md` as you go (commands, hashes, verbatim outputs).

1. `git ls-remote origin` for `main`, `ms-r1` and the tags; verify `origin/ms-r1` = `71c58904`; create `ms-r2` from it in your session's worktree. Do not touch `main` or any other worktree; record (do not fix) that the P1 worktree's local `ms-r1` and the primary checkout's `main` are behind.
2. Verify that the reference roots of D8 exist and hold what this round reads (weight exports, freeze arrays, per-update series, check tables of the MS-R1 runs); record the counts.
3. Record `nproc`, load, free disk, and the Python and package versions against the embedded record.

## 2. P1 — code, tests, decomposition, calibration, pre-registration (on `ms-r2`)

### 2.1 Code

- `run/run_ms_stagewise.py` (and `utils/ms_rule.py` if needed): the terminal-stage concentration-scale schedule of D2 as a new optional config key (absent = the MS-R1 behaviour, bit for bit); the stage-1 scale reset; the fixed-budget stratified sampler of D3; the per-check reporting columns of D4 (e_sigma(0), the decomposition, the scale); the scale per update in `ms_updates.csv`; the schedule in `run_config.json` and `manifest.json`.
- `tools/ms/`: the six arms and a wave `r2` in the config builder and the launcher (every key written out); `r2_launch_checks.py` (C-NL, C-MS3, C-MS4, the scale record, and the usual manifest, files, global-RNG and start-share checks); `r2_analysis.py` and an independent `r2_blind_criterion.py` (D4, D5); `r2_stop_candidates.py` (D6); `r2_decomposition.py` (§2.4).

### 2.2 Tests (`tests/test_ms_r2_*.py`, plus every existing suite)

- Defaults: without the new keys the MS runner is bit-identical to `71c58904` (every MS-R1 test passes unchanged).
- Scale schedule: the scale before each update equals the D2 table (ramp endpoints included); it is applied to the live actor and the lagged opponent, constant within an update, carried by the snapshot refresh and by the frozen snapshot, and reset at stage-1 entry. The Beta mean is unchanged by the scale up to float32 rounding (report the maximum difference in effort units, at most 1e-4). The empirical action spread at d = 0 scales as 1/sqrt(c · s + 1) (10^6 draws, within 3 SE). The stage-1 table built from a scaled frozen snapshot differs from the one built at scale 1 by at most 1e-6 · DW (report the maximum).
- Prefix identities on reduced budgets: C-NL (s > 1 against s = 1 through the first ramp update); the fixed-budget stratified path against MS-R1's rule mode with stop and polishing made impossible (rho = 0, localized fraction 0) through the cap and the first landing update (the C-MS4 relation); the bin-balanced fixed-budget path against `MS_base2400`'s (the C-MS3 relation).
- The analysis, the blind recomputation, the decomposition script and the calibration tool on synthetic rows and on stored MS-R1 exports; the launcher (the six configs differ only in `start_weights` and the scale key, as intended).
- Full suite `pytest tests`: everything passes apart from the known `test_registry_canonicalization`; record the summary line.

### 2.3 Independent review

One read-only review of the diff by an agent that did not write it (conformance with D2-D7, RNG and identity, the scale semantics); findings and dispositions in `reports/ms/r2/03_checks.md`.

### 2.4 Reproduce the decomposition (a premise check, before anything is built on it)

`tools/ms/r2_decomposition.py` on `results/ms_r1/analysis/per_run.csv` of `origin/ms-r1`: the table of the preamble, the ratio check over the 140 runs, the paired smoothing and remainder differences (budget, samplers) with intervals, the count of learned peaks below e2*(0). Report in `reports/ms/r2/01_decomposition.md`, with the derivation of the smoothing formula in a few lines (triangular f_xi, the noise difference, E|n_L − n_O| = 2 sigma / sqrt(pi) for Gaussian noise) and how close the learned Beta noise is to Gaussian. A number that differs from the preamble is reported as such (the preamble is not edited). **If the ratio check fails (any run outside 0.99 to 1.01) or the table is not reproduced to its second decimal, stop and report before going further.**

### 2.5 Offline calibration on MS-R1, and the pre-registration

- D6 on the 140 MS-R1 runs → `results/ms_r2/stop_calibration/`; report `reports/ms/r2/01b_stop_candidates.md` (the threshold grids recorded before the fire tables).
- `reports/ms/r2/02_preregistration.md`: D1-D8 as implemented; the schedule; the arms; the identities; the criterion and the secondary analyses; the predictions of D5 with their planning values; the reference roots; the test summary; the launch plan. Commit code, tests, the decomposition report, the calibration outputs and the pre-registration (**code commit**); later commits touch only `results/` and the report folder.

### 2.6 Push and continue

Push `ms-r2` (fast-forward, no force) after the pre-registration commit. If every test, check and the review are clean, continue to P2 without waiting for me.

## 3. P2 — the pilot (120 runs)

### 3.1 Launch

The six arms × 20 runs in one launch through the launcher (wave `r2`, at most 40 single-threaded workers in tmux; nproc, load, free disk, HEAD, `git diff --stat <code commit> HEAD` and `git status --porcelain` recorded; nothing changes after the launch), output `results/ms_r2/pilot/q*/seed*/<arm>/`. Crash rule as before: an infrastructure kill is re-run once after moving the first attempt to `results/ms_r2/pilot/crashed/`; any other nonzero exit counts as a failed run and is reported, not re-run.

### 3.2 Post-launch checks (`results/ms_r2/pilot/launch_checks.json`; any failure is a stop-and-report)

Manifests at the launch commit and clean; files complete; global-RNG assertions; start shares per stratum within 3 binomial SE (the stratified arms; the tail share lambda_T in force at every update); the applied scale equals the D2 schedule at every terminal-stage update of every run and is 1.0 throughout stage 1; C-NL in 80 of 80 comparisons (2 samplers × 2 values of s × 20 (q, seed)); C-MS3 in 20 of 20; C-MS4 in 20 of 20 at the defined update.

### 3.3 Analysis

`tools/ms/r2_analysis.py` → `results/ms_r2/analysis/`: the per-run table; the paired tables (primary, secondary, transmission ratio, interaction, references); `criterion.csv`; the decomposition at the freeze and along the landing; strata; stage 1; gates; figures (paired differences per arm; e_hat_2(0), sigma_2(0), smoothing part and remainder against updates 1800-2800 per arm with the segments marked; the per-run change of the remainder against the change of the smoothing part, with the line on which the two cancel). `tools/ms/r2_blind_criterion.py` recomputes `criterion.csv` and the interaction table from `per_run.csv` without importing the analysis tool (agreement to 1e-12). D6 repeated on the 120 runs.

## 4. P3 — reports, fact-check, push

### 4.1 Reports (`reports/ms/r2/`)

`00_housekeeping.md`, `01_decomposition.md`, `01b_stop_candidates.md`, `02_preregistration.md`, `03_checks.md`, `04_pilot.md` (checks first; one section per arm; the decomposition by arm and segment; the criterion and the secondary tables; the predictions of D5 against the outcome; anomalies), `05_decision_inputs.md` (the six arms side by side; what the noise landing changed in each component; what the sampler adds at each s; the stop candidates; what is not settled), `summary.md`. Add a row to `reports/ms/README.md` and a dated section at the top of `docs/STATE.md`. Every table is generated by a script under `reports/ms/r2/report_scripts/` from the CSVs.

### 4.2 Fact-check and tracking

An independent read-only fact-check of every number in the reports against the CSVs, ledger `reports/ms/r2/pi_record/01_factcheck.md`; corrections before the push. Commit the lightweight records with the usual tracking rules.

### 4.3 Push

Push `ms-r2` (fast-forward, no force) once the reports are committed. Do not touch `main`, do not create tags.

## 5. STOP

Stop after the §4.3 push. No fresh seeds, no lock, no T=3 run, no second pilot, no change after the pre-registration. The next round is mine to decide.
