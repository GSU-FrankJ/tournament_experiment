# R1 accuracy-refinement round: pre-registration

Written before any pilot run of this round was launched. It is not edited after the code commit; corrections go in a dated addendum at the end. Source of every requirement: the PI prompt of this round (decisions D1-D9, sections 1-7), cited below as "D#" / "§#". Where this file states a choice the prompt left open, it says so ("Choice").

## 1. Status and provenance

| item | value | source |
|---|---|---|
| branch / worktree | `v2-t2-refine`, `.claude/worktrees/v2-t2-refine` | D6 |
| branched from | `f02a256` = `origin/main` = tag `t2-v2-main` | `reports/v2/refine/00_preflight.md` |
| local `main` ref | stale (`d1b8443`), its checkout holds other sessions' uncommitted files; the branch was taken from the commit the prompt names, not from that ref | `00_preflight.md`, deviation section |
| protocol | v1.1, `protocols/v2_T2_locked_v1_1.json` (SHA-256 `21d85983…`), unchanged; `protocols/`, `run/run_v2_T2_locked.py`, `tools/v2/confirmation_analysis.py` are not edited | D1, D8 |
| development seeds | 10501-10510 x q in {50, 60}; nothing else is run | D6 |
| parents | `rehearsal_v1_1/q*/seed*/state_end_A.pt` of the canonical worktree (absolute paths and SHA-256 in every run config; inventory `results/v2_refine/parents.csv`) | D6, §1 item 4 |

Code under change (all defaults bit-identical to the locked behaviour, D8): `run/run_v2_stagewise.py`, `run/v2_rollout.py`, `agents/ppo_curriculum.py`, `agents/ppo_curriculum_v2.py`; new modules `utils/v2_continuation.py`, `agents/ppo_pathwise.py`; new tools under `tools/v2/`; tests `tests/test_v2_refine*.py`. The launch commits that follow the code commit touch only `results/` and `reports/v2/refine/`; every launch record stores `git diff --stat <code commit> HEAD`; every manifest must show `dirty: false`.

## 2. Design

- **One factor at a time, paired by (q, seed)** (D7). Every arm differs from its baseline arm in exactly one setting. Stage-1 arms branch from the same end-of-A parent (same frozen stage 2); stage-2 arms branch from the same u1200 state (the method-5 pair from the baseline u1600 state). All pilot runs use `fixed_budget=true`. Budgets stay 1600 / 600; polishing is carved out of the existing windows (D2); the only appended phase is method 5's 200 updates, always reported next to its matched control.
- **Baseline** = the locked v1.1 pipeline (D2): `reward_mode=expected`, `stage2_update_mode=frozen`, `adv_norm_scope=stage1_rows`, `continuation_action_mode=mean`; Phase A 1600 updates (LR 3e-4 constant to local 1200, then linear 3e-4 -> 3e-5 over 1201-1600); Phase B 600 updates (linear 3e-4 -> 3e-5 over 1-600); candidate = the last iterate.
- Paired-branching correctness checks (required, §4.1-4.3): `parents_A` end-of-A state identical to `rehearsal_v1_1` in 20/20; `B_base` identical to `state_end_B.pt` and the rehearsal end-of-B gate values in 20/20; `A_base` identical to `state_end_A.pt` in 20/20 (snapshot-refresh counter reported separately). A failure stops the round.
- **C-R1** (§2.4): the unchanged `run/run_v2_T2_locked.py` run from this branch on the 20 development runs must reproduce `rehearsal_v1_1` in 20/20 (`tools/v2/cr1_compare.py`, `results/v2_refine/v11_reproduction_checks.json`). These runs are also the D1 data of the locked pipeline.

## 3. Arms and their exact settings (D5, §4.2, §4.3)

All values are written explicitly in every `run_config.json` (arm tables: `tools/v2/launch_refine.py`, constants `STAGE1_ARMS`, `STAGE2_ARMS`, `PARENTS_A_ARMS`; the programmatic difference of each arm from its base arm is asserted in `tests/test_v2_refine_tools.py`).

### 3.1 Stage-1 wave: mode `phase_B`, parent `rehearsal_v1_1 state_end_A.pt`, locked flags, cap B = 600, 8 arms x 20 = 160 runs

| arm | the one change |
|---|---|
| `B_base` | none |
| `B_polish1` | LR windows local 1-400: 3e-4 -> 3e-5, local 401-600: 3e-5 -> 3e-6 |
| `B_polish2` | LR window local 1-600: 3e-4 -> 3e-6 |
| `B_batch` | `episodes_per_update` 2048, `minibatch` 1024 (40 optimizer steps per update, unchanged) |
| `B_batch_mb256` | `episodes_per_update` 2048, `minibatch` 256 (secondary; 160 steps) |
| `B_kl005` | `target_kl` 0.005 |
| `B_kl010` | `target_kl` 0.01 |
| `B_expcont` | `continuation_value_mode=expected` (method 6) |

### 3.2 Stage-2 wave: 11 arms x 20 = 220 runs

Mode `phase_A_continue`, parent `parents_A/q*/seed*/state_u01200.pt`, `reward_mode=expected`, cap A = 400, window local 1-400 linear 3e-4 -> 3e-5 unless stated (local 1-400 = global 1201-1600):

| arm | the one change |
|---|---|
| `A_base` | none |
| `A_polish1` | LR windows local 1-250: 3e-4 -> 3e-5, local 251-400: 3e-5 -> 3e-6 |
| `A_polish2` | LR window local 1-400: 3e-4 -> 3e-6 |
| `A_batch` | `episodes_per_update` 2048, `minibatch` 1024 (20 optimizer steps per update, unchanged) |
| `A_batch_mb256` | `episodes_per_update` 2048, `minibatch` 256 (secondary; 80 steps) |
| `A_kl005` | `target_kl` 0.005 |
| `A_kl010` | `target_kl` 0.01 |
| `A_anneal2` | `conc_anneal` 1.0 -> 2.0, linear in the local update over 1-400 |
| `A_anneal4` | `conc_anneal` 1.0 -> 4.0, linear in the local update over 1-400 |

Method-5 pair, parent = baseline u1600 state (`rehearsal_v1_1 state_end_A.pt`):

| arm | definition |
|---|---|
| `A_ctrl200` | mode `phase_A_continue`, cap 200, LR constant 3e-5 (one window local 1-200), locked Phase-A flags |
| `A_detmean` | mode `phase_P`, cap 200, LR constant 3e-5, pathwise deterministic-mean updates (§4.5); dev-tier verifier every 50 updates and at the end |

No Phase B is run for stage-2 arms. Stage-2 arms start from u1200, whose training state is shared by all arms; `A_base` must reproduce the end-of-A state of the locked pipeline.

## 4. Implementation of the six methods (what the code does)

1. **Polishing / LR windows.** `lr_decay` accepts several contiguous, non-overlapping windows per phase (the next `local_first` = the previous `local_last` + 1; the last window ends at the phase cap); inside a window lr(j) = start + (end - start)(j - local_first)/(local_last - local_first), applied to both optimizers before each update, constant within an update, Adam state preserved. `start_lr != ab_lr` is accepted for a window starting at local 1 of a continued phase (`phase_A_continue`, `phase_P`) and for later windows; every other refusal is kept. A single window behaves exactly as before.
2. **Larger batch.** `budget_overrides.episodes_per_update` and `ppo_overrides.minibatch`. The `phase_c_root + phase_c_es == episodes_per_update` check applies only when phase C is run.
3. **Target-KL.** After each PPO epoch the whole-buffer KL over the policy rows (the quantity logged as `kl_epochs`, mean((r-1) - log r)) is computed as before; if it is **strictly greater** than `target_kl` no further epoch runs in this update (neither actor nor critic steps); the epoch that exceeded the target has run. The permutation of every skipped epoch is still drawn from the minibatch stream and discarded (rule A6), so the minibatch stream position after the update equals the baseline's. `n_epochs_run` is a column of `v2_updates.csv` (and a key of the history entry only when `target_kl` is set). Both the masked update (Phase B) and the original update (Phase A) implement it.
4. **Concentration annealing.** `BetaActor.conc_scale` (default 1.0; the multiplication is skipped at exactly 1.0): c = (c_min + softplus(z_c)) * conc_scale. The runner sets scale(j) = scale_first + (scale_last - scale_first)(j - local_first)/(local_last - local_first) on the live actor and the lagged opponent before each rollout; the refresh copies the scale to the opponent; a frozen snapshot keeps its freeze-time scale. At the end of the phase the actor carries the final scale. The scale is stored in the full state (key `conc_scale`), the weight exports (array `conc_scale`, float64, the exact Python float) and the manifest. **Choice:** the full-state key and the export array are written only when a scale differs from 1.0, so that default files stay key-identical to the locked ones (absent = unscaled); `mean_effort_numpy(..., conc_scale=...)` reproduces exported (alpha, beta) outside torch.
5. **Deterministic-mean terminal fine-tuning (`phase_P`, ablation only).** Exact-gradient ascent on the conditional expected terminal payoff R(d, e, e_opp) = w_l + DW F_xi(d + e - e_opp) - k e^2; learner e(d) = e_min + range alpha/(alpha+beta) formed in float64 from the float32 (alpha, beta), opponent = the lagged copy's mean at -d (no gradient); loss = -mean R; the existing actor Adam (state preserved), LR 3e-5, clip 0.5; no critic update, PPO ratio, clipping or entropy. Exploring starts and roles come from the `start` stream with the Phase-A calls; no shock or action draw is made; the opponent is refreshed at phase entry and every 20 global updates. `agents/ppo_pathwise.torch_F_xi` equals `utils.theory_multistage.F_xi` to 1e-12 (tested). **Choice (concentration head):** the gradient of output row 1 of `actor.out` is zeroed explicitly, and because the preserved Adam moments of those parameters (left by Phase A) would still move them, their weight and bias are saved before and restored bit-exactly after every optimizer step; at the end of the phase the run asserts and records (`phaseP_checks.json`) that the head, the critic and the critic's Adam state are bit-identical to the parent's. The payoff function is the known game (as the locked expected-reward estimator); the closed-form equilibrium is not used.
6. **Expected continuation.** For the stage-1 rows only, the sampled continuation r_2 in the stage-1 return is replaced by the table value V~_2(y), y = e_1 - e_1^opp (both efforts as executed, float64): stage-1 return = r_1 + V~_2(y), stage-1 advantage = that return - V(s_1). Stage-2 rows, their critic targets and every random draw are unchanged (shocks still drawn, d_2 still computed, rule A6). Valid only with `stage2_update_mode=frozen`, `continuation_action_mode=mean` and gamma = lambda = 1. The opponent's stage-1 action stays sampled.
   - **Integration rule (Choice).** V~_2(y) = E_z[g_2(y + z)] with g_2(d) = w_l + DW F_xi(d + e^_2(d) - e^_2(-d)) - k e^_2(d)^2, e^_2 the frozen stage-2 Beta mean on the float path of `continuation_action_mode=mean`; z has the triangular density (2q - |z|)/(4q^2) on [-2q, 2q]. The frozen actor is evaluated **directly** at the quadrature nodes d = y + z (g_2 is never interpolated). The z-integral is a composite Gauss-Legendre rule with 6 nodes per panel on panels of width 1 aligned at z = 0 and z = +-2q (the density's kink points), 200 panels (q = 50) / 240 panels (q = 60), weights including the density, summing to 1 to rounding. The y-grid is symmetric on [-100, 100] with step 0.05 (4001 nodes), float64, linear interpolation at lookup (refuses y outside the grid beyond float noise). Built once at Phase-B entry from the frozen snapshot (7-8 s single-threaded). Convergence: halving the panel width and doubling the nodes changes the table by at most 2.1e-8 DW (parent actors, from 2.0/3 nodes) and at most 5e-9 DW (from the default rule) (`results/v2_refine/continuation_check.json`, key `convergence`). The module imports neither `utils.dp_br_verifier` nor `utils.v2_metrics`.
   - **Check (ii) of §2.3, outcome and decision.** The table was compared with the verifier's stage-1 Q^e^(0, e) + k e^2 at every effort-grid node (parent seed 10501, both q, stage-1 mean fixed at 40 / 45). On the **final** tier the maximum difference is 3.23e-5 DW (q = 50) and 1.96e-5 DW (q = 60) against the required <= 1e-6 DW: **the literal criterion is not met**. Development tier: 1.33e-4 / 8.5e-5 DW. Evidence on the cause: the verifier's stage-2 values equal the table's g_2 at the verifier's own nodes (<= 1e-12 DW, tested), and the difference falls by about 4x per halving of the verifier state step (sweep in the JSON); on a refined verifier configuration (state step 0.25, 64 Gauss-Legendre nodes) it is 5.0e-7 DW (q = 50) and 4.1e-7 DW (q = 60), within 1e-6 DW. The PI was asked and decided to keep method 6 in the round and to record both results here and in `04_pilot_stage1.md`; the literal test stays in the suite as a strict expected failure and no tolerance was loosened.

7. **Mid-phase-A parent (runner change).** The stage-2 arms branch from `parents_A/.../state_u01200.pt`, a state saved inside an uninterrupted phase A (`phases_done == []`, `phase_done == "A"`). `execute()` for mode `phase_A_continue` refused such a state (it required `phases_done == ['A']`, which only holds for states of a continued phase); it now also accepts `phases_done == []` with `phase_done == "A"`. Nothing else about the continuation changes (phase-entry snapshot refresh at u1200 is a no-op because u1200 is a multiple of 20; the phase-local verifier and stop-rule counters restart; `A_base` must reproduce the locked end-of-A state, §2).

## 5. Metrics and analysis (§5)

Final tier for every gate metric; recovery metrics are tier-independent. All differences are **arm - baseline**, paired by (q, seed); per q: median, mean, `n_better` out of 10, and 95% percentile bootstrap CIs of the mean and of the median. Every run is reported, failures included. Tool: `tools/v2/refine_analysis.py` (reuses `pilot4_common.py`, `pilot4_analysis.py`, `decomposition.py`, `induced_band.py`, `rng_divergence.py`, `pilot1_smoothed_game.py`).

**Bootstrap (D5).** 10,000 percentile resamples of the 10 paired seeds with replacement; `numpy.random.default_rng(20261003)`; one fresh generator per (q, statistic) in table order (so the resample indices are drawn identically for every statistic of the same size, which makes the intervals reproducible and comparable); the CI is the 2.5 / 97.5 percentiles of the resampled mean (respectively median). The dispersion ratio arm/baseline of an across-seed SD uses paired resampling (the same resampled seeds for both arms).

**Stage-1 arms** (candidate = end-of-B last iterate; baseline `B_base`):
- primary: |e^_1(0) - e_1*| / e_1* (the S1 value); also the signed error;
- dispersion: across-seed SD of the signed error and of e^_1(0), ratio arm/baseline with a paired-resampling CI; within-run SD and range of e^_1(0) over the last 5 weight exports;
- the learning / inherited decomposition with residual bands (e~_1 is the same for all arms of a (q, seed) because stage 2 is shared: stated and verified);
- Gmax_full/DW with (t*, d*) and the G-F verdict, EXP_root, dReach, Deltamax_all, dFull, |dev - final| of Gmax (G-N part), S1 pass, the v1.1 run outcome given the parent's G-A, and the 0929 target |error| <= 0.05;
- optimisation diagnostics: KL, clip fraction, grad norms, `n_actor_steps`, `n_epochs_run` (target-KL arms: distribution of the stopping epoch), advantage SD of the stage-1 rows (for `B_expcont`: the variance-reduction measurement, ratio to the baseline), RNG divergence (the batch arms diverge at the first update by construction), wall time.

**Stage-2 arms** (candidate = stage-2 last iterate at the end of the continued phase; baseline `A_base`; for the method-5 pair also the parent u1600 candidate and `A_ctrl200`):
- primary: the signed peak error; also the location-free peak error with its argmax d;
- RMSE_pos/e_2*(0), tail mean and max, eta_2/DW with the G-A verdict, |dev - final| of eta_2 (G-N part), the symmetry error, sigma_2(0) (annealing arms: the annealed value), the smoothed-game share of the d = 0 gap (Pilot-1 method, with the actor's own Beta at the end of the phase);
- annealing arms: the smoothing-predicted peak gap before and after annealing next to the observed change (the test of the smoothing explanation);
- `A_detmean`: the loss trajectory, the first-order-condition residual |dR/de| at the end (mean and max over the exploring-start distribution) and the profile |e^_2(d) - e_2*(d)| on the recovery grid (diagnostic only);
- optimisation diagnostics as above, and wall time.

**Definitions of verdicts.** G-A = eta_2/DW <= 0.005 and RMSE_pos/e_2*(0) <= 0.05 and tail mean/e_2*(0) <= 0.02 (final tier); G-F = Gmax_full/DW <= 0.01 (final tier); G-N = |dev - final| <= 0.001 for eta_2/DW (end of A) and for Gmax_full/DW (end of B); run pass = G-A and G-F and G-N (protocol v1.1). For a stage-1 arm the G-A and the eta_2 part of G-N are those of the shared parent (`rehearsal_v1_1/.../gates.json`); the arm contributes G-F and the Gmax part of G-N. For a stage-2 arm the criterion's gate is G-A with its G-N part (eta_2).

**Per-arm criterion (descriptive, not a gate).** (a) The bootstrap CI of the mean paired difference of the primary metric excludes 0 in the improving direction at **both** q, and (b) no run that passed its gate (G-F for stage-1 arms, G-A for stage-2 arms, each with its G-N part) under the baseline fails it under the arm. Both parts are reported separately. **Choice (improving direction):** stage 1: a decrease of |e^_1(0) - e_1*|/e_1*; stage 2: a decrease of |signed peak error| (the baseline peak errors are negative, so this equals an increase of the signed error); the signed differences are reported as well.

## 6. Diagnostics

**D1 (T57 issue 7).** `tools/v2/d1_clamp_analysis.py` on the 20 C-R1 runs plus the pilot runs as they finish; all runs log the `d1_*` columns of `v2_updates.csv` (raw `rng.beta` draws below `action_clamp` or above 1 - `action_clamp`, counted before the clip, per stage, for learner and opponent, and for the learner at the final stage split by |d| < 2q; plus min alpha, min beta and the numbers of learner policy rows with alpha < 1 and beta < 1) and save the full buffer at local update 25, at the phase midpoint rounded to a multiple of 25 and at the phase cap (`d1_buffers/u{global_update}.npz`). The buffer saved at global update u was collected before update u; the weights export of the same update is post-update (the analysis states the resulting ratio deviation). Materiality flags (report-only): **M1** = the median over runs of the per-phase clamp fraction among learner policy rows exceeds 1e-3; **M2** = the gradient share of the clamped rows exceeds 1% at any saved update in more than 2 of the 20 runs. Report: `02_d1_clamp.md`.

**D2 (T57 issue 8).** `tools/v2/verifier_sensitivity.py`: pure evaluation with `utils.v2_metrics.evaluate` on both tiers and both q of closed-form perturbed candidates, families (a)-(e) as specified in §3.2. Detection limits: the smallest perturbation at which Gmax_full/DW > 0.01 (G-F), eta_2/DW > 0.005 (G-A), |dev - final| > 0.001 DW (G-N), or "not reached on the grid". Descriptive; it changes no threshold. Report: `03_d2_verifier_sensitivity.md`.

## 7. Verification state at the code commit

- Full suite on the code of the code commit: **328 passed, 1 failed, 4 xfailed** in 386 s (`results/v2_refine/code_pytest_full.txt`). The one failure is the known `tests/test_registry_canonicalization.py::test_registry_canonicalization`; the 4 xfailed are the 2 pre-existing ones and the 2 strict expected failures of check (ii) (§4 item 6, final tier, q = 50 and q = 60).
- C7 on the final code: **IDENTICAL** (`results/v2_refine/c7/code_check.compare.txt`; run output `results/v2_refine/c7/code_check.run.out`; reference = the existing runner's output `results/v2_pilots/phase1_regression/before/FINAL_A400_B25_C25/tel_q50_s10501`).
- `tests/test_v2_refine*.py` cover the items of §2.3: defaults identical to the C7 reference and to the old `v2_updates.csv` columns, piecewise LR closed forms for B and A (P1, P2) and the constant window rule, steps per update (20/40 and 80/160), target-KL (one epoch, stream position, mid-update stops, `n_epochs_run`), annealing (identity at 1.0, scaling, log-prob, opponent scale, export reload), continuation table ((i) Monte Carlo, (iii) stream positions, (iv) stage-2 rows unchanged, lookup), `phase_P` (F_xi to 1e-12, finite-difference gradient, head and critic bit-identical, start stream equal to an equally long Phase-A continuation, other streams unmoved), D1 counts (alpha = 0.3 against `scipy.stats.beta`), the launcher arm tables and the C-R1 comparison tool, the D1 and D2 tools.
- Branch-point baseline (before any change): 103 passed, 1 failed (the same known test), 2 xfailed (`results/v2_refine/preflight_pytest.txt`); C7 IDENTICAL (`results/v2_refine/c7/branchpoint_f02a256.compare.txt`).
- **Independent adversarial review of the core diff** (five reviewers, one lens each, report-only: bit-identity, configuration and schedules, phase P, expected continuation, D1 and annealing). Verdict: no blocker. A probe of the **unchanged** `run/run_v2_T2_locked.py` for q = 50 seed 10501 on the reviewed code reproduced `rehearsal_v1_1` in all 45 compared fields (`tools/v2/cr1_compare.py`), and a bitwise A/B run against the HEAD sources (locked mode, 30+30 updates) showed 0 differences; the full C-R1 check on the 20 runs follows the launch. Changes made because of the review before this commit: (1) each D1 buffer now also stores the pre-update actor (`actor_pre.*`, `conc_scale_pre`) that generated it, and the gradient share of the clamped rows is computed with it (ratio exactly 1; the post-update export is kept only as a secondary `postexport_*` comparison); (2) `manifest.json` records `conc_scale_initial` after the parent state is restored; (3) `verifier_timeout["P"]` is checked when a phase-P run is constructed; (4) the `phase_c_root + phase_c_es == episodes_per_update` check of the record is kept for every mode, and the post-override check applies when phase C is run; (5) `mean_effort_numpy` reads the exported `conc_scale` when none is passed; (6) every config carries `conc_anneal` explicitly (null where unused) and the launcher refuses seeds outside 10501-10510; (7) the continuation-table build record is written to `v2_run_summary.json`.
- **Facts for reading the results (from the review, not defects).** (a) `A_detmean` rows have their own schema: `v2_updates.csv` holds `loss`, `grad_norm_pre_clip`, `foc_abs_mean`, `foc_abs_max`, `e0` (the learner's mean effort at d = 0), `actor_lr`, the stream positions, and no `kl_*`, `clip_frac`, `n_epochs_run`, `conc_scale` or `d1_*` columns (phase P draws no action, so it has no D1 data by construction); its checkpoint file is `v2_checkpoints_P.csv`, and because cap 200 is a multiple of the timeout 50 the last call is labelled `reason = timeout` (the end-of-phase row is the one with `local = cap`). The per-update `loss` and `foc_abs_*` are evaluated on that update's 512 fresh exploring-start rows, so they are noisy; the loss trajectory in the report is therefore also computed offline on a fixed grid from the weight exports. The flags of a `phase_P` config carry no information. (b) For annealed arms the verifier-side `conc_pass`, `eligible` and `would_have_fired` use the annealed concentration, so they are not comparable with the baseline arms. (c) Existing loaders (`pilot1_analysis.load_actor_policy`, `pilot2_analysis._actor_fns`, `induced_band`) ignore `conc_scale`; the analysis of this round uses a loader that applies it. (d) The reload check `mean_effort_numpy` reproduces (alpha, beta) to float32 rounding (rtol about 1e-6), as it did before this round, not bit for bit. (e) The continuation table assumes a root gap of 0 (true for Phase B at T = 2); it is looked up at e_1 - e_1^opp only.

## 8. Appendix: descriptive KL statistics of the locked baseline (P0 item 7)

(The tables are produced by `python tools/v2/refine_preflight.py kl` from the 20 `rehearsal_v1_1` `train_history.json` files; fragment `results/v2_refine/preflight/kl_appendix.md`, pasted below.)

### KL statistics of the locked v1.1 baseline (descriptive)

Source: `history[*]` of `train_history.json` of the 20 `rehearsal_v1_1` runs (q in {50, 60} x seeds 10501-10510), canonical worktree `results/v2_T2_locked/rehearsal_v1_1/q*/seed*/`. Every update logs `kl_epochs`, the whole-buffer KL over the policy rows after each of the 10 PPO epochs, mean((r - 1) - log r) with r = pi_new / pi_old, and `kl_final_epoch` = `kl_epochs[-1]`. Phase A has 1600 updates and phase B 600 per run, so each table row pools 16000 (A) or 6000 (B) updates per q. Percentiles use linear interpolation (`numpy.percentile`).

Stopping rule of the target-KL arms (D5): after an epoch, if the KL exceeds the target (strict >) no further epoch runs; the epoch that exceeded the target has run. The number of epochs run is therefore the 1-based index of the first epoch with KL > target, or 10 if none exceeds it. The tables apply that rule to the baseline `kl_epochs`. They are exact for each baseline update taken in isolation (up to and including the first stop the update is identical to the baseline), but counterfactual for the run as a whole: after the first stop the trajectory of a target-KL run differs from the baseline, so later updates would see different buffers and policies. They are not a prediction of the share of stopped updates in a target-KL run. 'Stopped' means that some epoch's KL exceeded the target; an exceedance at epoch 10 truncates nothing (10 epochs run either way), so the column 'fewer than 10 epochs run' (stop epoch 1 to 9) is the share of updates a target-KL run would actually shorten.

Script: `python tools/v2/refine_preflight.py kl`; CSVs: `results/v2_refine/preflight/kl_final_epoch_distribution.csv`, `results/v2_refine/preflight/kl_target_stopping.csv`.

#### (a) Final-epoch KL (`kl_final_epoch`), all updates of the 10 seeds

| phase | q | updates | min | p10 | median | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| A | 50 | 16000 | 4.6e-05 | 0.00185 | 0.00503 | 0.0116 | 0.0237 | 0.0815 | 0.00616 |
| B | 50 | 6000 | 5.36e-09 | 0.000152 | 0.00326 | 0.0123 | 0.0306 | 0.0709 | 0.0052 |
| A | 60 | 16000 | 1.82e-05 | 0.00174 | 0.00497 | 0.0119 | 0.025 | 0.0823 | 0.0062 |
| B | 60 | 6000 | 1.05e-08 | 0.000188 | 0.00341 | 0.0128 | 0.0335 | 0.0795 | 0.00547 |

#### (b) Counterfactual stopping at each target

| phase | q | target | updates | stopped (n) | stopped (share) | of which fewer than 10 epochs run (share) | stopped share per run, min to max | stop epoch median / p90 (stopped only) | epochs run median / p90 / mean |
|---|---|---|---|---|---|---|---|---|---|
| A | 50 | 0.005 | 16000 | 14379 | 89.87% | 88.32% | 84.9% to 95.9% | 3 / 7 | 3 / 10 / 4.04 |
| A | 50 | 0.01 | 16000 | 8173 | 51.08% | 48.88% | 37.1% to 72.8% | 3 / 8 | 10 / 10 / 7.06 |
| B | 50 | 0.005 | 6000 | 5729 | 95.48% | 94.83% | 93.3% to 97.5% | 2 / 5 | 2 / 6 / 2.87 |
| B | 50 | 0.01 | 6000 | 4153 | 69.22% | 67.13% | 60.7% to 79.5% | 2 / 8 | 5 / 10 / 5.46 |
| A | 60 | 0.005 | 16000 | 14624 | 91.40% | 90.13% | 85.9% to 94.7% | 3 / 6 | 3 / 9 / 3.84 |
| A | 60 | 0.01 | 16000 | 8792 | 54.95% | 52.68% | 36.2% to 69.4% | 4 / 8 | 9 / 10 / 6.84 |
| B | 60 | 0.005 | 6000 | 5704 | 95.07% | 94.33% | 91.5% to 97.0% | 2 / 5 | 2 / 6 / 2.89 |
| B | 60 | 0.01 | 6000 | 4222 | 70.37% | 68.55% | 61.3% to 77.0% | 3 / 7 | 4 / 10 / 5.36 |

#### (c) Stopping epoch, share of all updates (%)

Epoch index is 1-based; 'none' = no epoch exceeded the target (10 epochs run).

| phase | q | target | ep 1 | ep 2 | ep 3 | ep 4 | ep 5 | ep 6 | ep 7 | ep 8 | ep 9 | ep 10 | none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 50 | 0.005 | 11.78 | 27.24 | 19.95 | 10.84 | 6.51 | 4.51 | 3.13 | 2.60 | 1.75 | 1.55 | 10.13 |
| A | 50 | 0.01 | 3.76 | 10.68 | 11.16 | 7.33 | 4.47 | 3.53 | 2.98 | 2.67 | 2.29 | 2.21 | 48.92 |
| B | 50 | 0.005 | 35.47 | 26.30 | 13.38 | 7.65 | 4.58 | 2.88 | 1.88 | 1.27 | 1.42 | 0.65 | 4.52 |
| B | 50 | 0.01 | 18.90 | 15.83 | 9.52 | 5.72 | 4.40 | 4.02 | 3.32 | 2.88 | 2.55 | 2.08 | 30.78 |
| A | 60 | 0.005 | 14.38 | 27.20 | 19.26 | 11.05 | 7.00 | 4.58 | 2.86 | 2.02 | 1.79 | 1.27 | 8.60 |
| A | 60 | 0.01 | 5.01 | 11.63 | 10.42 | 7.47 | 4.72 | 4.44 | 3.54 | 2.72 | 2.73 | 2.27 | 45.05 |
| B | 60 | 0.005 | 34.68 | 26.68 | 13.87 | 7.42 | 4.27 | 3.13 | 1.97 | 1.33 | 0.98 | 0.73 | 4.93 |
| B | 60 | 0.01 | 18.67 | 16.30 | 10.02 | 6.12 | 4.73 | 4.27 | 3.27 | 2.62 | 2.57 | 1.82 | 29.63 |

#### (d) Supplementary: phase A split by learning-rate window

Local 1-1200 has constant LR 3e-4; local 1201-1600 is the linear decay 3e-4 to 3e-5, the window the phase-A continuation arms run in.

| phase | q | updates | min | p10 | median | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| A local 1-1200 | 50 | 12000 | 4.6e-05 | 0.00198 | 0.00524 | 0.012 | 0.0243 | 0.0815 | 0.00641 |
| A local 1201-1600 | 50 | 4000 | 0.000103 | 0.00153 | 0.00443 | 0.0103 | 0.019 | 0.0542 | 0.00539 |
| A local 1-1200 | 60 | 12000 | 1.82e-05 | 0.00187 | 0.00518 | 0.0122 | 0.0254 | 0.0704 | 0.00643 |
| A local 1201-1600 | 60 | 4000 | 6.06e-05 | 0.00139 | 0.00439 | 0.0107 | 0.0215 | 0.0823 | 0.00549 |

| phase | q | target | updates | stopped (n) | stopped (share) | of which fewer than 10 epochs run (share) | stopped share per run, min to max | stop epoch median / p90 (stopped only) | epochs run median / p90 / mean |
|---|---|---|---|---|---|---|---|---|---|
| A local 1-1200 | 50 | 0.005 | 12000 | 11125 | 92.71% | 91.42% | 87.7% to 97.9% | 3 / 6 | 3 / 9 / 3.72 |
| A local 1-1200 | 50 | 0.01 | 12000 | 6650 | 55.42% | 53.08% | 40.3% to 76.8% | 3 / 8 | 8 / 10 / 6.77 |
| A local 1201-1600 | 50 | 0.005 | 4000 | 3254 | 81.35% | 79.03% | 75.8% to 90.0% | 3 / 7 | 4 / 10 / 5.01 |
| A local 1201-1600 | 50 | 0.01 | 4000 | 1523 | 38.07% | 36.25% | 27.5% to 60.5% | 4 / 9 | 10 / 10 / 7.91 |
| A local 1-1200 | 60 | 0.005 | 12000 | 11304 | 94.20% | 93.17% | 88.3% to 97.2% | 3 / 6 | 3 / 8 / 3.52 |
| A local 1-1200 | 60 | 0.01 | 12000 | 7155 | 59.62% | 57.27% | 40.3% to 74.1% | 3 / 8 | 7 / 10 / 6.53 |
| A local 1201-1600 | 60 | 0.005 | 4000 | 3320 | 83.00% | 81.00% | 74.5% to 87.8% | 3 / 7 | 4 / 10 / 4.8 |
| A local 1201-1600 | 60 | 0.01 | 4000 | 1637 | 40.92% | 38.90% | 24.0% to 55.2% | 4 / 9 | 10 / 10 / 7.76 |

| phase | q | target | ep 1 | ep 2 | ep 3 | ep 4 | ep 5 | ep 6 | ep 7 | ep 8 | ep 9 | ep 10 | none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A local 1-1200 | 50 | 0.005 | 12.95 | 29.97 | 21.03 | 10.24 | 6.12 | 4.34 | 2.84 | 2.42 | 1.50 | 1.29 | 7.29 |
| A local 1-1200 | 50 | 0.01 | 4.25 | 11.65 | 12.64 | 8.00 | 4.61 | 3.68 | 3.14 | 2.81 | 2.30 | 2.33 | 44.58 |
| A local 1201-1600 | 50 | 0.005 | 8.25 | 19.07 | 16.70 | 12.65 | 7.67 | 5.03 | 4.00 | 3.15 | 2.50 | 2.33 | 18.65 |
| A local 1201-1600 | 50 | 0.01 | 2.30 | 7.75 | 6.72 | 5.33 | 4.05 | 3.08 | 2.48 | 2.27 | 2.27 | 1.82 | 61.92 |
| A local 1-1200 | 60 | 0.005 | 16.25 | 29.27 | 19.89 | 10.74 | 6.83 | 4.19 | 2.55 | 1.84 | 1.62 | 1.02 | 5.80 |
| A local 1-1200 | 60 | 0.01 | 5.83 | 12.57 | 11.59 | 8.02 | 5.08 | 4.84 | 3.73 | 2.82 | 2.79 | 2.36 | 40.38 |
| A local 1201-1600 | 60 | 0.005 | 8.78 | 21.00 | 17.35 | 11.97 | 7.53 | 5.72 | 3.77 | 2.55 | 2.33 | 2.00 | 17.00 |
| A local 1201-1600 | 60 | 0.01 | 2.52 | 8.80 | 6.90 | 5.83 | 3.67 | 3.23 | 2.98 | 2.42 | 2.55 | 2.02 | 59.08 |

