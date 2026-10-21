# Round T3-R3: diagnosis of the T = 3 BW failures, then a bounded improvement tested on fresh development seeds

## Where the project stands

- **Model.** Two identical players play a T-stage rank-order tournament with terminal prizes (`reports/t3/r1/pi_record/25_t3_r1_prompt.md`, D2).
  - Each stage a player chooses effort e ∈ [0, 100] at cost k e², with k = 1/3500, and draws a shock U[-q, q].
  - Prizes are w_H = 6 and w_L = 2, so ΔW = 4.
  - The gap moves as d' = d + e - e_opp + ξ, with ξ triangular on [-2q, 2q].
  - B = 100 + 2q and D_t = [-(t-1)B, (t-1)B]. At T = 3 the terminal domain D_3 = [-2B, 2B] is twice as wide as the terminal domain D_2 = [-B, B] at T = 2.
  - The actor's input is [(t-1)/(T-1), d/((t-1)B)] (`envs/curriculum_env.GameSpec.encode_obs`).
- **Policy.** `agents/ppo_curriculum.BetaActor` (variant `t1`): a mean/concentration Beta policy on two tanh hidden layers.
  - μ = clamp(sigmoid(z_0), 1e-6, 1 - 1e-6), c = c_min + softplus(z_1) with c_min = 100, α = μc, β = (1 - μ)c.
  - The runner trains with `agents/ppo_curriculum_v2.CurriculumPPOv2`, whose `update` calls `CurriculumPPO.update` through `super()`.
  - Learner draws below the action clamp 1e-6 are counted per update in `ms_updates.csv` (`d1_*_lo`). At the terminal stage the count is split at |d_L| < 2q (`d1_L_s3_{in,out}_{n,lo,hi}`; `run/v2_rollout.py`).
  - Weight exports `weights/u{u:05d}.npz`, every 25 updates, hold the actor (`actor.*`) and the critic (`critic.*`).
  - `train_history.json`, untracked, holds per update everything `CurriculumPPO.update` returns, including `grad_norm_actor_mean/max`, `grad_norm_critic_mean/max` and the entropy and concentration statistics.
- **Verifier tiers** (`utils/dp_br_verifier.py`):
  - the final tier `FINAL_CONFIG` has state step 2, effort step 0.5 and 32 Gauss-Legendre nodes per half of the shock support;
  - the development tier `DEV_CONFIG` has state step 4, effort step 1 and 16 nodes per half.
- **Gains.**
  - η_t is the largest one-step deviation gain at stage t over the stage-t grid, divided by ΔW (`Delta_max_t{t}_over_dw`).
  - G_t(d) = V_t^BR(d) - V_t^mean(d), and Gmax_full is its maximum over t and states, divided by ΔW (`Gmax_full_over_dw`).
  - At the terminal stage, G_3 = Δ_3.
- **Development checks.** Every 25 updates the runner evaluates the candidate on the development tier and writes `ms_checks_stage{t}.csv`. The columns `Delta`, `s`, `R`, `R_tail` and `C` are defined in `utils/ms_residual.py`, and the closed-form equilibrium does not enter them:
  - `Delta` = η_t of the phase's stage;
  - `s` = the one-step best-response effort at d = 0;
  - `R` = the largest |learned effort - one-step best-response effort| / s over |d| < 2q(T - t + 1);
  - `R_tail` = the mean learned effort outside that region / s;
  - `C` = the largest normalised Beta standard deviation on the development grid.

  Caution: during the stage-3 phase the checked candidate plays the live stage-3 network at every stage. So the `Gmax_full_over_dw` column of `ms_checks_stage3.csv` includes stages 1 and 2 played by that network, and it is not a stage-3 quantity.
- **Gates** (`reports/t3/r1/prereg_criteria.json`, computed by `tools/t3/t3_metrics.py`):
  - G-A: η_T ≤ 0.005, `rmse_pos_T_over_eT0` ≤ 0.05 and `tail_mean_T_over_eT0` ≤ 0.02. The last two compare the terminal-stage effort with the closed-form tent.
  - G-F: `Gmax_full_over_dw` ≤ 0.01.
  - G-N: the development-final differences of η_T and of Gmax_full are ≤ 0.001.
  - A run passes iff it exits with code 0, its global-RNG assertion is `ok`, its evaluation is valid on both tiers (`valid` and `dev_valid`), and it passes G-A, G-F and G-N. Every "≤" is inclusive at full float64 precision.
- **T = 2 is settled.** Solver v2.0 passed its fresh-seed confirmation with 39 of 40 runs (seeds 30501-30520).
- **The T = 3 backward-induction solver `bw`.** It runs through `run/run_ms_stagewise.py`, with configs built by `tools/t3/t3_configs.build_bw_config` and `reports/t3/r1/params_bw_t3.json`.
  - Stage 3 is trained for 1600 updates on bin-balanced starts on D_3 (bin width 10) and frozen.
  - Stage 2 is then trained for 1600 updates against the exact expected continuation of the frozen stage 3, continuing from the stage-3 network, and frozen.
  - Stage 1 is trained for 600 updates.
  - Every update has 512 episodes. The learning rate is 3e-4, decaying linearly to 3e-5 over local updates 1201-1600 (stages 3 and 2) and 1-600 (stage 1).
  - **Per-run budget:** updates 1600 / 1600 / 600, policy rows 819,200 / 819,200 / 307,200, 76,000 minibatch steps, 152 exports `u00025` ... `u03800`, and 64 / 64 / 24 development checks.
- **Round T3-R1** (branch `t3-r1`, `be2b7e9f`; seeds 50501-50520):
  - `bw` passed in 36 of 40 runs.
  - The pre-registered comparison with joint whole-game training supported H1, H2, H3 and H1-T2 at both q. It is not touched in this round.
- **Round T3-R2** (branch `t3-r2`, final commit `c3289df9afbcd69d995c2b8c6d5b497a9dddeebe`):
  - **Audit.** It found no implementation error (`reports/t3/r2/01_audit.md`). An independent best-response calculator, `tools/t3/t3r2_br_calc.py`, reproduced the stored gains of 200 runs to at most 2.5e-5 ΔW.
  - **Stage-2 near-zero stretches.** A stretch is a run of consecutive in-band (|d| < 2q) stage-2 nodes of the final-tier grid where the learned mean effort is below 1 (`01_audit.md` §8). The seven stretches of T3-R1 are learning failures. Six were present 25 updates after the stage-2 phase entry; the stretch of (q = 60, seed 50514) formed in updates 2475-2750.
  - **Confirmation.** The locked `bw` (protocol T3-1.0, `protocols/v2_T3_locked_v1_0.{json,md}`) failed its pre-registered confirmation on seeds 60501-60520: 34 of 40 runs passed, 16 of 20 at q = 50 and 18 of 20 at q = 60, against a rule of at least 36 and at least 17 per q (`reports/t3/r2/04_confirmation.md`).
- **The ten failures of the 80 `bw` runs.** Sources:
  - `results/t3_r1/analysis/bw_failures/points.csv`;
  - `results/t3_r2/confirmation/analysis/failures.csv`;
  - both `per_run.csv`;
  - for the last-16-export statements, `results/t3_r2/audit/a8_last16.csv` and `results/t3_r2/confirmation/analysis/a8_last16_confirmation.csv`.

  S3 is defined under "Preliminary observations" below.

  | round | q | seed | failing gate | stage of the failing quantity | values | S3 |
  |---|---|---|---|---|---|---|
  | T3-R1 | 50 | 50504 | G-A (η_3) | 3 | η_3 0.00623, above 0.005 at all of the last 16 stage-3 exports | 0.78 |
  | T3-R1 | 50 | 50517 | G-A (η_3) | 3 | η_3 0.00568, above 0.005 only at the last export | 0.00 |
  | T3-R1 | 60 | 50514 | G-F | 2 (stretch formed in training) | Gmax_full 0.05628 | 0.44 |
  | T3-R1 | 60 | 50517 | G-F | 2 (stretch present at phase entry) | Gmax_full 0.01089 | 0.70 |
  | T3-R2 | 50 | 60501 | G-A (η_3) | 3 | η_3 0.00524; learned e_3(0) 60.11 against e*(0) = 70 | 0.78 |
  | T3-R2 | 50 | 60506 | G-A (η_3) | 3 | η_3 0.00724; e_3(0) 58.55 | 0.79 |
  | T3-R2 | 50 | 60513 | G-A (η_3) | 3 | η_3 0.00547; e_3(0) 60.12 | 0.74 |
  | T3-R2 | 50 | 60515 | G-A (η_3, RMSE), G-F | 3 | η_3 = Gmax_full = 0.01138 at d = -36 (learned 53.4, best response 30.3); RMSE_pos 0.0667 | 0.80 |
  | T3-R2 | 60 | 60516 | G-A (RMSE only) | 3 | RMSE_pos 0.0538, η_3 0.00376 | 0.77 |
  | T3-R2 | 60 | 60518 | G-A (all three), G-F | 3 | Gmax_full 0.14468; learned e_3(0) 8.16 against e*(0) = 58.33 | 0.00 |

- **The terminal stage is less accurate at T = 3 than at T = 2 under the same per-stage training.** Medians at q = 50 / q = 60, from `results/t3_r1/eval/per_run.csv` (arm `v20` for T = 2) and both rounds' `bw` rows. `peak_T_abs` = |ê_T(0) - e*(0)| / e*(0), and the median signed value is negative in all four groups.

  | | T = 2 (40 v2.0 runs) | T = 3 (80 `bw` runs) |
  |---|---|---|
  | η_T | 0.00157 / 0.00102 | 0.00201 / 0.00137 |
  | `rmse_pos_T_over_eT0` | 0.0239 / 0.0229 | 0.0294 / 0.0269 |
  | `peak_T_abs` | 0.0615 / 0.0678 | 0.0804 / 0.0785 |
  | runs with η_T > 0.005 | 1 of 20 / 0 of 20 | 6 of 40 / 1 of 40 |

- **Preliminary observations.** The PI made these from the tracked logs before this round, and the review of this prompt added to them. They are leads to test, not results.
  - **S3, the stage-3 clamp share of a run.** S3 = Σ `d1_L_s3_out_lo` / Σ `d1_L_s3_out_n` over the stage-3 rows of `ms_updates.csv` with local update 1201-1600. It is the share of the learner's stage-3 draws at out-of-band starts (|d| ≥ 2q, where the terminal equilibrium effort is 0) that fall below the action clamp.
    - **At T = 3,** 30 of the 80 `bw` runs have S3 > 0.2, 25 of them above 0.4. At q = 50 that is 16 of 40 runs, and at q = 60 it is 14 of 40.
    - **At T = 2,** among the 40 v2.0 runs, the analogue (Σ `d1_L_s2_out_lo` / Σ `d1_L_s2_out_n` over phase-A local 1201-1600 of `results/v2_T2_locked/confirmation_v2_0/q{q}/seed{s}/v2_updates.csv`, tracked on this branch) is at most 0.19, and below 0.005 in 26 runs.
  - **Failures by S3.** 8 of the 10 failures are among the 30 runs with S3 > 0.2. The failure rate is 8 of 30 there, against 2 of 50 in the other runs.
  - **Accuracy by S3.**
    - At q = 50, runs with S3 > 0.2 have median η_3 0.0033 and median `peak_T_signed` -0.104. The other runs have 0.0016 and -0.063, which is about T = 2's level.
    - At q = 60 the corresponding values are 0.0023 / -0.089 and 0.0011 / -0.071.
  - **Stretches by S3.** All ten `bw` runs with a stage-2 stretch (seven in T3-R1, three in T3-R2) have S3 ≥ 0.436.
  - **The development check at the end of stage 3** (local 1600 of `ms_checks_stage3.csv`):
    - `Delta` > 0.005 in exactly seven of the 80 runs, all failing: T3-R1 (50, 50504) and (50, 50517); T3-R2 (50, 60501), (50, 60506), (50, 60513), (50, 60515) and (60, 60518).
    - `R` is 1.18 and `R_tail` 0.195 for (60, 60518). The largest values among the other 79 runs are 0.341 and 0.0103.
  - **Seed 60518 (q = 60) looks like a different case.** Sources: `results/t3_r2/confirmation/eval/per_run.csv`; `results/t3_r2/confirmation/q60/seed60518/bw/ms_updates.csv` and `ms_checks_stage3.csv`; `reports/t3/r2/04_confirmation.md` §5.
    - S3 is 0 and no stage-3 learner draw falls below the clamp.
    - `eT_at_0` = `tail_max_T` = 8.155370679773789. Its stored `rmse_pos_T_over_eT0`, 0.4620386542963486, equals to all 16 digits the value computed for a constant effort 8.155370679773789 against the tent. So the frozen stage-3 policy appears to be exactly constant in d.
    - Its development `Delta` was 0.1451 at local update 100 and stayed between 0.144 and 0.154 to the end. At q = 60 other runs pass through a similar level near local update 100 and then leave it: seed 60501 had 0.140-0.147 at local 100-150.
    - Its `C` fell to 0.0148 at local 1600, against 0.0318 for seed 60501.
- **Untracked files.** They live only in the worktrees of earlier rounds:
  - T3-R1: `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472/results/t3_r1/`;
  - T3-R2: `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38/results/t3_r2/`, which holds the 40 confirmation runs' weights, `state_end*.pt`, `train_history.json`, freeze arrays and evaluation arrays, and the audit's re-run files;
  - v2.0: `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0/`.

  These are read-only reference roots.

## The plan for this round (binding; from the coworker's note to the PI, with the PI's decisions)

1. **Diagnose 60518 first, from the existing records.** Compare it with the passing q = 60 runs. Examine:
   - the stage-3 Beta parameters;
   - the clamp;
   - the advantage and critic signals;
   - the gradients;
   - the state coverage;
   - the deviation-gain curves.

   Where a key record is missing, a diagnostic replay on the original seed is allowed. Also examine the three q = 50 runs with too little effort near the tie point (60501, 60506, 60513), and 60515's excess effort at d = -36, and decide whether they are the same kind of problem.
2. **Choose a bounded improvement from the diagnosis.**
   - Fixing a stage 3 that does not form, as in 60518, comes first.
   - One candidate: at the end of stage 3, check with the development verifier, and on failure re-initialize and retrain at most once by a fixed rule. The check may use only closed-form-free stage-3 quantities (see the closed-form rule and D4).
   - If the diagnosis points to a clearer cause, use a better-founded scheme.
   - If, once that is fixed, a persistent error remains at q = 50, try one precision change aimed at the effort curve across states.
   - Every trigger, count, budget and cost is fixed in advance.
   - When stage 3 changes, its continuation is rebuilt and stage 2 and stage 1 are trained by the original procedure.
3. **Test on fresh development seeds.**
   - Use 20 per q at q = 50 and q = 60, paired with the locked old `bw` on the same seeds.
   - Evaluate with G-A, G-F and G-N. Report per q the pass counts, the worst deviation gain and the compute cost. Every planned run counts.
   - Once a development result reaches the target fixed in advance, lock only one complete scheme and confirm it on another unused block of seeds. The lock and the confirmation are a later round.
   - 60518 may be used for diagnosis, but never as fresh-seed evidence for the new scheme.
4. **The PI's decisions, fixed now.**
   - **Development target:** an arm reaches it iff at least 38 of its 40 runs pass and at least 18 of 20 at each q. With 20 runs per q, the per-q condition follows from the total.
   - **The later confirmation:** 40 seeds per q (80 runs). It passes iff at most 6 of the 80 runs fail and at most 4 of the 40 fail at each q (D6).

## Steps

1. **P0:** housekeeping, committed on its own first (§1).
2. **P1:**
   - the diagnosis code and tests;
   - the diagnosis DG1-DG8 and the proposal;
   - an independent review;
   - the push (§2).

   Then **STOP at gate G1**.
3. **P2,** only after my reply "proceed P2", which names the approved arms and any change to the proposal:
   - the implementation, its tests and an independent review;
   - C-LEGACY at the final code commit;
   - the pre-registration, committed and pushed before any development run;
   - the development wave, its checks, evaluation, analysis and blind recomputation (§3).
4. **P3:** reports, fact-check, push, then **STOP at G2** (§4, §5).

## Environment and standing rules

- **Repository and Python.** The repository is `/home/fjiang4/tournament_experiment`. Python is `/home/fjiang4/tournament_experiment/.venv/bin/python` only. Run `pytest tests`, never a bare `pytest`.
- **Runs.** Every run is single-threaded (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) and runs in tmux. Do not kill jobs you did not start.
- **Branch and worktree.**
  - Work in a new worktree, on a new branch `t3-r3` created from `origin/t3-r2` at `c3289df9afbcd69d995c2b8c6d5b497a9dddeebe`. Record the worktree path.
  - The first file written there is this prompt, verbatim, as `reports/t3/r3/pi_record/28_t3_r3_prompt.md`.
- **Other worktrees.** Do not touch the worktrees of earlier rounds, and never run `git clean` in any of them. Read their untracked files in place. Do not copy them into git.
- **Where files go.**
  - Results go under `results/t3_r3/`, and reports under `reports/t3/r3/`.
  - New tools are new files `tools/t3/t3r3_*.py`, and new tests are `tests/test_t3r3_*.py`.
- **Provenance:** every number in a report cites a path.
- **The closed form.** The closed-form equilibrium (the tent `utils.theory_multistage.g2_two_stage`; at T = 2 also g1 = ΔW/(6kq)) enters evaluation, reporting, diagnosis and calculator tests only. It never enters training or anything a run decides.
  - This includes any trigger, threshold, selection between attempts and start distribution.
  - In particular `rmse_pos_T_over_eT0`, `tail_mean_T_over_eT0` and the peak errors use the closed form, so they cannot trigger anything in a run.
- **Results and tracking.**
  - Never delete or overwrite results or checkpoints.
  - Tracking follows `reports/ms/README.md` and `tools/t3/t3_track.py`.
  - Never tracked: `.pt` files, `train_history.json`, weight exports, `freeze_stage*_*.npz`, `final_eval_*.npz`, per-run evaluation arrays, replay arrays and `run.log`.
- **Frozen files.**
  - **In P1:** no existing file changes in `agents/`, `envs/`, `utils/`, `run/`, `protocols/` or `tools/`.
  - **In P2:** the approved scheme may change existing files in `agents/` or `run/`, only where it must. Each change is additive and sits behind a config key whose absence gives exactly the old behaviour (`run/run_ms_stagewise.py` validates config keys, so a new key must be declared there), and C-LEGACY must pass. Prefer new files where that is practical.
  - **Never in this round:** `envs/`, `utils/`, `protocols/`, `tools/v2/`, `tools/ms/` and the existing `tools/t3/` files do not change. There is no lock in this round.
  - Anything else that seems to need a change: stop and report.
- **Pushing** happens only at three points: the G1 push (§2.6), the pre-registration push before the development launch (§3.3), and the final push (§4.3). `main` and tags are not touched, and no pull request is opened.
- **Implementation errors.** If the work finds an implementation error in the code or the verifier, stop and report it with its impact, as round T3-R2 did (`reports/t3/r2/pi_record/27_t3_r2_prompt.md`, D2).

**If any step fails, or anything does not match this prompt, stop at that point and report. Do not work around it.**

---

## 0. Decisions (binding)

### D1. Scope and seeds

- **P1.** No new seed is trained. The only training is the instrumented replays of DG5: the stage-3 phase of recorded runs, into `results/t3_r3/replays/`.
- **P2.**
  - The development block is seeds 70501-70520, the same 20 seeds at q = 50 and at q = 60.
  - It is trained by the control arm `bw` and by at most two candidate arms approved at G1, all on the same seeds.
  - Each candidate arm also runs once at (q = 60, seed 60518) as a diagnostic run, reported separately and never counted (D5).
  - C-LEGACY re-runs two T3-R2 runs (D5).
- **Reserved:** 80501-80540, for the later confirmation. Inventoried in P0 and never trained in this round.
- **Never trained,** except for the replays, the diagnostic runs and the C-LEGACY re-runs named here: 40501-40520, 10501-10530, 30501-30520, 50500-50520 and 60501-60520. Unit tests that train reduced-budget runs in temporary directories (for example at seed 50500 in `tests/t3_helpers.py`) are exempt.
- **Out of scope:**
  - T = 4 and the paper's figures;
  - any change to the gates, the thresholds, the evaluator or the verifier;
  - a change to stage 2 or stage 1 beyond rebuilding their continuation from the changed stage 3;
  - the lock and the confirmation (a later round, D6).

### D2. Working classes of failure

Each of the ten failures above gets one class, with its evidence. The diagnosis may split or merge classes, if it says why.

- **(F) Formation failure.** The stage-3 policy never develops its dependence on d.
- **(K) Clamp regime.** The run is in the stage-3 clamp regime (S3 > 0.2), and its failing quantity is either a stage-3 in-band bias or a stage-2 near-zero stretch. A stretch counts whether it was present at the stage-2 phase entry or formed later.
- **(V) Last-iterate variability.** The tent forms, the run is not in the clamp regime, and the failing value sits at the top of its own range over the last 16 exports of the phase.
- **(E) Implementation error.** Stop and report.
- **(U) Unexplained.**

### D3. The diagnosis (P1; reads records and evaluates stored weights; the only training is DG5)

**DG1. Records inventory.**

- For every run used, list which records exist:
  - tracked: `ms_updates.csv`, `ms_checks_stage{t}.csv`, `ms_binmaps_stage{t}.npz`, `run_config.json`, `gates.json`;
  - untracked: the 152 exports, `state_end_stage{t}.pt`, `train_history.json`, the freeze arrays and the evaluation arrays.
- Map each item of the coworker's list to the records that answer it: Beta parameters, clamp, advantage and critic signals, gradients, state coverage, deviation-gain curves.
  - Whole-network gradient norms per update are in `train_history.json`, if it is present.
  - `ms_binmaps_stage{t}.npz` does not hold realised starts per bin: under bin-balanced starts its `probs` is NaN.
- Say which items need the replay of DG5. These are the per-region and per-bin quantities: the gradients by region and the starts per bin.

**DG2. The stage-3 policy over training, all 80 `bw` runs.** Compute everything from the 64 stage-3 exports `u00025` ... `u01600`, on the final-tier grid of D_3, without training.

- **At every export:**
  - ê_3(d), α(d), β(d), the normalised standard deviation and the critic V(d);
  - the sensitivity |∂μ/∂(input 2)| at the in-band nodes, by autograd on the loaded network;
  - the share of first- and second-layer tanh units with |pre-activation| > 3 at the in-band nodes, which measures saturation.
- **Per export, the summaries:**
  - the in-band (|d| < 2q) range of ê_3 and its mean slopes on [-2q, 0) and on (0, 2q];
  - ê_3(0);
  - the out-of-band mean effort;
  - the share of out-of-band nodes with α < 1;
  - the formation measure F(u) = ê_3(0) - mean{ê_3(d) : |d| ≥ 2q}, which is closed-form-free.
- **The formation export** of a run is the first export at which F(u) exceeds half of the median final F(u01600) of the passing runs of the same q. A run that never reaches it is reported as not formed.

Compare within q. Report 60518 against the other 39 runs of q = 60.

**DG3. Training signal, coverage and the development checks.** Use the stage-3 rows of `ms_updates.csv`:

- `mean_effort`, `kl_final_epoch`, `clip_frac`, `policy_loss`, `value_loss`, `adv_raw_mean` and `adv_raw_std`;
- the starts per stratum, `n_start_tail`, `n_start_near` and `n_start_mid`. Coverage per bin comes from the DG5 replays.
- the clamp counters `d1_L_s3_{in,out}_{lo,n}`;
- the 64 checks of `ms_checks_stage3.csv` (`Delta`, `s`, `R`, `R_tail`, `C`);
- where `train_history.json` exists, its gradient-norm, entropy and concentration series.

Compare 60518 with the 18 passing q = 60 runs of T3-R2, and summarise all 80 runs.

**DG4. Deviation-gain curves.** Use the T3-R2 calculator `tools/t3/t3r2_br_calc.py`, or the final-tier verifier, on the export with the export's own network as the opponent.

- **Where:** at the exports `u00100`, `u00400`, `u00800` and `u01600`.
- **Runs:** 60518 and the DG5 set.
- **What:** Δ_3(d) over D_3, and the objective Q_3(d, e) for e ∈ [0, 100] at d ∈ {0, ±4, ±6, ±10, ±20, ±36, ±40, ±80} and at the run's own stored maximiser.
- **Report:** the slope at the learned effort, the best-response effort, and every local maximum.

**DG5. Instrumented replays of the stage-3 phase.**

*Set (14 runs, all from T3-R2 `results/t3_r2/confirmation/`, S3 as computed in DG6):*

| group | q = 50 | q = 60 |
|---|---|---|
| the six failures | 60501, 60506, 60513, 60515 | 60516, 60518 |
| passing, clamp regime (S3 > 0.4) | 60505, 60507 | 60503, 60509 |
| passing, not in the regime (S3 < 0.05) | 60502, 60504 | 60501, 60502 |

*Driver.* `tools/t3/t3r3_replay.py` drives the runner in-process. The CLI of `run/run_ms_stagewise.py` always runs all three phases, and it writes `ms_updates.csv` only at the end.

1. Do the set-up of `run_pipeline` (`run/run_ms_stagewise.py`): output directory, config copy, `seed_globals(seed)`, `MSRun(cfg, out_dir)`, the phase-start hook and the RNG check.
2. Call `rec = run.run_stage(3)` and `run.freeze_stage(3, rec, check)`.
3. Call `write_histories(run, out_dir)`, and stop.

The same driver, without the instrumentation, is used in the passivity test (§2.2).

*Passive instrumentation.*

- It wraps the update for the stage-3 phase (`CurriculumPPOv2.update`, which reaches `CurriculumPPO.update`).
- It draws no random numbers. The only RNG `update` itself consumes is the minibatch permutation per epoch.
- It changes no tensor, optimizer state or generator of the training.
- It computes on detached deep copies, using `torch.autograd.grad`: deep copies carry stale `.grad` tensors.

*Inputs inside `update`.*

- The update sees float32 observations, so d_L = obs[:, 1] · 2B at stage 3.
- A draw below the clamp has action equal to float32(1e-6). Cross-check the counts against `d1_L_s3_{in,out}_{lo,n}`, or wrap `run/v2_rollout.collect_batch_v2` to read them directly.
- At stage 3, return - V equals the raw advantage exactly, so the critic's error needs no separate item.

*Bit-identity.* The replay must reproduce the original bit for bit:

- the 64 stage-3 exports;
- the stage-3 rows of `ms_updates.csv`, compared as dicts without `update_wall_sec`;
- `ms_checks_stage3.csv` without `verifier_sec`;
- `freeze_stage3_{final,development}.npz`.

If it does not, it is not a replay: report that and drop the item. Do not stop the round.

*Logged at every stage-3 update, on the batch at the start of the update (θ_old):*

- **Regions:** in (|d_L| < 2q) and out (|d_L| ≥ 2q), plus 20-unit bins of d_L.
- **Per region r:**
  - n_r, the number of samples, which gives the starts per bin;
  - the mean and standard deviation of the raw advantage and of the normalised advantage Â (as `update` uses it);
  - g_r = (1/N) Σ_{i∈r} Â_i ∇_θ log π(a_i | s_i) over the actor parameters, with N the batch size;
  - ‖g_r‖ and the projection share ⟨g_r, g⟩ / ‖g‖², where g = Σ_r g_r. The shares sum to 1 over the in/out partition, and the baseline for "disproportionate" is n_r / N.
- **The mean ‖∇_θ log π‖** for draws below the clamp and for the other draws.
- **The actor:** its sensitivity |∂μ/∂(input 2)| over the in-band samples, and the saturation share of DG2.

**DG6. The clamp regime across runs.**

- **Recomputation.** Recompute S3 for the 80 `bw` runs, and the T = 2 analogue for the 40 v2.0 runs (definitions above). Every preliminary number above must be reproduced or corrected.
- **Relations to report:**
  - between S3 and η_3, `peak_T_signed`, `rmse_pos_T_over_eT0`, Gmax_full and pass/fail;
  - between S3 and the stage-2 stretches: T3-R1's seven runs (`results/t3_r1/analysis/bw_failures/zero_effort.csv`) and T3-R2's three (`results/t3_r2/confirmation/analysis/stretches.csv`).
- **Timing.** The onset of the regime is the first 25-update window in which the out-of-band share exceeds 0.2. Set it against ê_3(0) over the exports, run by run.
- **From the replays:** the projection shares of DG5 for the out-of-band samples and for the draws below the clamp, against their sample shares, in the clamp-regime runs and in the others.

**DG7. Classification.** Give each of the ten failures a class under D2, with its evidence. Answer the coworker's question:

- Are the q = 50 tie-point deficits (60501, 60506, 60513; and T3-R1's 50504) one kind of problem?
- Is 60515's excess at d = -36 the same kind?
- How do they relate to 60516's shape failure and to 60518?

**DG8. Stage-2 phase entry.** Relate S3 to the stage-2 stretches. At the phase entry, evaluate the stage-3 network `u01600` at the stage-2 input for all 80 runs, as T3-R2's A7 did for its runs (`reports/t3/r2/01_audit.md` §8).

### D4. The proposal (`reports/t3/r3/02_proposal.md`, at G1)

1. **The mechanism(s)** that the evidence supports, by class.
2. **At most two candidate changes.**
   - **(A) A safeguard against formation failure.** This is the priority. The coworker's candidate is a check at the end of the stage-3 phase plus at most one re-initialization and retraining of stage 3 by a fixed rule. A better-founded alternative is welcome if the diagnosis points to one.
   - **(B) One precision change** aimed at the stage-3 effort curve across states, if the evidence supports one, for example one that acts on the clamp regime.
   - The diagnosis may instead show that one change covers both. Then propose one change.
3. **For each candidate:**
   - **What it changes and why:** the evidence from DG1-DG8, and at least one real published reference for the method, using the terms used there. If you cannot verify a reference, say so.
   - **The exact specification:**
     - The trigger quantities and thresholds, closed-form-free stage-3 quantities only: the development `Delta` of the stage-3 phase (= Δ_3 = G_3), `R`, `R_tail`, `C` or `s`, or the verifier's stage-3 best-response efforts. Never RMSE_pos, the tail mean or a peak error. Never the `Gmax_full_over_dw` column of `ms_checks_stage3.csv`, which is not a stage-3 quantity.
     - When the trigger is checked, and how many times it may fire at most.
     - For a re-initialization: what is re-initialized (actor, critic, optimizer state) and from which derived seed and RNG streams.
     - Which attempt is frozen. Any choice between attempts uses only closed-form-free development-tier stage-3 quantities.
     - Where the artefacts of a discarded attempt are stored: in new files or an `attempt1/` subdirectory, never overwritten, so that a non-firing run writes exactly the files, columns and rows of `bw`.
     - The budgets (updates and policy rows) in the normal case and the worst case, and the expected compute cost.
     - The config keys and their legacy defaults.
     - How the continuation is rebuilt, so that stage 2 and stage 1 follow by the original procedure.
4. **For a check-and-retrain rule, a retrospective table on the 80 recorded runs,** using `ms_checks_stage3.csv` at local update 1600. Show at least two thresholds:
   - one at the gate level (`Delta` > 0.005);
   - one aimed at formation failure only, for example on `R` or `R_tail`.

   For each, give the runs it would have fired on, how many of those failed, the failing runs it would miss, and the extra cost. State plainly what each threshold does:
   - A gate-level trigger gives runs near the limit a second draw. It raises the pass rate without changing how a stage is learned, and a paper must describe it that way.
   - A formation-level trigger catches gross failures only.
5. **The development arms.**
   - Default: the control `bw`, `bwA` (A alone) and `bwAB` (A and B). Propose a different set if the evidence supports one, with reasons.
   - Give the order in which the selection rule of D5 takes the candidate arms, and the expected cost of each arm.
6. **A draft of the P2 pre-registration** (D5 and D6 filled in).

### D5. The development wave (P2)

- **Arms:** the control `bw` and the candidate arms named in my "proceed P2" reply.
- **Seeds:** 70501-70520 at q = 50 and q = 60, the same seeds in every arm. Run names are `t3r3_q{q}_s{seed}_{arm}`, and run directories `results/t3_r3/dev/q{q}/seed{s}/{arm}/`.
- **Control configs.** The settings are exactly the locked T3-1.0 settings.
  - Build each config with `tools/t3/t3_configs.build_bw_config`, as the lock's launcher did, then set `seed`, `run`, `record.run`, `record.seed`, `record.output_dir` and `pilot` (= "t3_r3").
  - Check it against the lock's `reference_config_sha256` with the method of `tools/t3/t3c_launch.py` (`normalise` and `config_problems`), reusing those functions by import where they apply.
- **Candidate configs** differ from the control's config of the same (q, seed) only in the scheme's own keys and the run fields. A test checks this with `tools/t3/t3_configs.config_diff`.
- **Diagnostic runs.** Each candidate arm runs at (q = 60, seed 60518) into `results/t3_r3/diagnostic/q60/seed60518/{arm}/`. They are reported separately and never counted. C-INIT and C-PAIR compare them with T3-R2's original run of that seed.
- **Before the launch, in this order:**
  1. **The independent review** of the P2 code, with any fixes committed.
  2. **C-LEGACY,** at that final code commit. Re-run T3-R2's runs (q = 50, seed 60502) and (q = 60, seed 60518) from their own `run_config.json` into `results/t3_r3/legacy/`. Compare each with its original in the T3-R2 untracked root using `tools/t3/compare_runs.py bw`. Both must be bit-identical. A mismatch is a stop.
  3. **The pre-registration,** committed and pushed (§3.3). From the C-LEGACY commit to the launch, commits change only `reports/`, `results/`, `docs/STATE.md` and `reports/t3/README.md`.
- **The launch.** One wave of 40 × (number of arms) runs plus the diagnostic runs, at most 40 single-threaded workers, in tmux, from the commit that holds the pre-registration, on a clean tree.
- **Post-launch checks** (`results/t3_r3/dev/launch_checks.json`), per arm, as listed in `reports/t3/r2/pi_record/27_t3_r2_prompt.md` §3.2:
  - **Basics:** exit codes; complete files; manifests at the launch commit with a clean tree; the global-RNG assertions.
  - **Budgets:** for `bw`, updates 1600 / 1600 / 600, policy rows 819,200 / 819,200 / 307,200 and 76,000 minibatch steps. For a candidate, the budgets it specifies, including any retraining.
  - **Schedules:** the learning rate at every update; 152 exports; 64 / 64 / 24 checks.
  - **C-INIT:** the recorded init digests equal the planned ones, and they are equal across arms at the same seed unless the scheme changes the initial network.
  - **C-REPLAY:** the evaluator reproduces the freeze arrays bit for bit.
  - **C-PAIR,** for a candidate that differs from `bw` only by a check-and-retrain rule:
    - at every seed where its trigger did not fire, the candidate run equals the `bw` run bit for bit (`tools/t3/compare_runs.py bw`), with the scheme's own records only in new files;
    - where it fired, a new comparator `tools/t3/t3r3_pair.py` shows that the stored first attempt equals the `bw` run's stage 3 up to the check: the stage-3 exports, and the stage-3 rows of `ms_updates.csv` and `ms_checks_stage3.csv` without wall-clock columns.
- **Evaluation:** `tools/t3/t3_eval.evaluate_run` on the final and development tiers, through a new driver. The evaluated policy is the one the scheme freezes. The gates and the run pass are as above. For a retrained run, S3 is computed on the frozen attempt.
- **The target (the PI's):** a candidate arm reaches it iff at least 38 of its 40 runs pass and at least 18 of 20 at each q.
  - Pass probabilities for comparison, with independent runs at per-run pass probability p: 0.882 at p = 0.97, 0.677 at 0.95, 0.223 at 0.90 and 0.108 at 0.875.
  - Every planned run counts. Failures are counted as in D6.
- **Selection.** Only candidate arms are eligible. The first candidate arm, in the order fixed at G1, that reaches the target is the scheme proposed for the lock. If none reaches it, no scheme is proposed. Nothing is locked in this round.
- **Reported, deciding nothing:**
  - per arm and q:
    - the pass counts with exact intervals;
    - the worst Gmax_full and its run;
    - the medians of η_3, `rmse_pos_T_over_eT0`, `tail_mean_T_over_eT0`, `peak_T_signed`, Gmax_full and EXP_root;
  - against `bw`, per seed:
    - the paired differences of Gmax_full and η_3, with paired bootstrap intervals (10,000 resamples, seed 20261021);
    - the pass/fail cross-table;
  - S3 of every run;
  - the stage-2 stretches;
  - every trigger that fired, with the run and the outcome;
  - the compute cost per arm (updates, policy rows, wall time, verifier seconds) and its ratio to `bw`.
- **Blind recomputation:** an independent tool recomputes the counts and the target outcome from the evaluation table.

### D6. The later confirmation (fixed now; executed only in a later round, after the PI's decision at G2)

- **What is confirmed:** the one scheme proposed under D5, locked in that round as protocol T3-1.1.
- **Seeds:** 80501-80540, the same 40 seeds at q = 50 and at q = 60, for 80 runs.
- **The rule:** the confirmation passes iff at most 6 of the 80 runs fail and at most 4 of the 40 runs fail at each q.
- **Operating characteristic** (with an independent per-run failure probability f, the same at both q): pass probability 0.982 at f = 0.03, 0.865 at 0.05, 0.271 at 0.10 and 0.030 at 0.15. A script recomputes and prints it in the pre-registration.
- **Failure counting,** as in T3-R2's D4:
  - **Infrastructure failures:** a negative return code, an out-of-memory kill, or ENOSPC, recorded with evidence in an `infra_events.json`. Such a run is re-run once. A second infrastructure failure stops the confirmation without a verdict.
  - **Other failures:** any other nonzero exit, and any global-RNG violation, is a failed run.
  - **Evaluation problems:** an evaluation error, or an invalid evaluation, is a failed run unless a re-run of the locked evaluator succeeds.
  - **The scheme's own steps:** a re-initialization or retraining inside a run is part of the run, never a re-run.

### D7. Predictions (the PI's, written before P1; expectations, not tests)

- **DP1.** 60518 is a formation failure, unrelated to the clamp.
  - At q = 60, runs pass through a level of development `Delta` of about 0.12-0.15 near local update 100. 60518 never left it.
  - Its stage-3 network is saturated: its tanh units are saturated over D_3, its gap-sensitivity is about 0, and its in-band surrogate gradient stays small, compared with runs that formed.
- **DP2.** The clamp regime is a mechanism, not only a correlate.
  - In the clamp-regime runs, the out-of-band samples and the draws below the clamp have a projection share of the surrogate gradient above their sample share.
  - The tie-point effort ê_3(0) falls, or stops rising, after the regime's onset.
- **DP3.** The following are one class, (K):
  - the q = 50 tie-point deficits (60501, 60506, 60513, 50504);
  - 60515's excess at d = -36;
  - 60516's shape failure;
  - the two T3-R1 stage-2 stretch failures. 50517's stretch was present at the phase entry, while 50514's formed in updates 2475-2750, but both runs are in the regime.

  T3-R1's 50517 at q = 50 is (V).
- **DP4.** No prediction of the development outcome.

### D8. Reference roots

Every tool takes these as explicit arguments and records them. A missing file is a stop-and-report.

- **T3-R1, untracked:** `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472/results/t3_r1/`. Runs are in `runs/q{q}/seed{s}/bw/`.
- **T3-R2, untracked:** `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38/results/t3_r2/`. Runs are in `confirmation/q{q}/seed{s}/bw/`.
- **v2.0 confirmation runs:** `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0/q{50,60}/seed{30501..30520}/`. Their tracked logs are also on this branch.
- **Tracked records,** on the branch: `results/t3_r1/`, `results/t3_r2/`, `results/v2_T2_locked/confirmation_v2_0/`, `reports/t3/r1/` and `reports/t3/r2/`.

---

## 1. P0 — housekeeping (committed on its own, before any code)

Write `reports/t3/r3/00_housekeeping.md` as you go: commands, hashes and verbatim outputs.

1. **Refs.**
   - Run `git ls-remote origin` for `main`, `t3-r1`, `t3-r2` and the tags.
   - Verify that `origin/t3-r2` = `c3289df9afbcd69d995c2b8c6d5b497a9dddeebe`.
   - Record, but do not fix, local refs that are behind.
2. **Branch and worktree.** Create `t3-r3` and its worktree, record the path, and save this prompt there first.
3. **Seeds.**
   - Run `tools/v2/seed_inventory.py --block 70501 70520` and `--block 80501 80540` into `results/t3_r3/preflight/`, and list every hit with `tools/t3/seed_hits.py`.
   - **Stop-and-report** on a hit in a training-run record of the project's tournament lines (`run_config.json`, `manifest.json`, `status.json`, `gates.json` or a run directory, in any worktree).
   - Classify all other hits.
4. **Reference roots.** Verify D8 and record:
   - for the 40 T3-R1 and 40 T3-R2 `bw` runs: 152 exports, `state_end_stage{1,2,3}.pt`, the freeze arrays, and whether `train_history.json` is present;
   - for the 40 v2.0 runs: 88 exports and `v2_updates.csv`.
5. **Machine.** Record `nproc`, load, free disk, and the Python, torch and numpy versions.
6. **Commit** P0: the prompt, `00_housekeeping.md` and the preflight files.

## 2. P1 — the diagnosis and the proposal (on `t3-r3`)

### 2.1 Code (new files only)

- **`tools/t3/t3r3_diag.py`:** DG1-DG4 and DG6-DG8, and the retrospective trigger table of D4.
  - It is read-only and moves no RNG.
  - It runs in parallel with at most 40 workers.
  - Its output goes to `results/t3_r3/diag/`.
- **`tools/t3/t3r3_replay.py`:** DG5, with the bit-identity comparison built in. Output goes to `results/t3_r3/replays/`.

### 2.2 Tests (`tests/test_t3r3_*.py`, plus every existing suite)

- **Passivity of the instrumentation.** Use a reduced-budget `bw` config, for example stage 3 at 50 updates, in a temporary directory. The DG5 driver with the instrumentation equals the same driver without it, bit for bit.
- **The per-region quantities of DG5** on a hand-made batch equal a hand computation, including that the projection shares sum to 1.
- **S3** on a synthetic `ms_updates.csv`.
- **F(u), the formation export and the saturation share** on synthetic networks, including a network whose output is constant in d.
- **The retrospective trigger table** on a synthetic `ms_checks_stage3.csv`.
- **The full suite.** `pytest tests` passes, apart from the known `test_registry_canonicalization` and the failures that T3-R1 recorded as known (`results/t3_r1/preflight/tests/two_non_t3_failures.txt`).

### 2.3 Run the diagnosis

- Commit the code and tests first, then run DG1-DG8 at that commit on a clean tree.
- The replays are 14 runs of the stage-3 phase, with at most 14 workers.
- Record the cost of every step.

### 2.4 Reports

- **`reports/t3/r3/01_diagnosis.md`:**
  - DG1-DG8, every number with its path;
  - the class table of the ten failures;
  - the predictions DP1-DP3 against the evidence;
  - a one-line answer to the coworker's question in DG7.
- **`reports/t3/r3/02_proposal.md`:** D4.

Tables are generated by scripts under `reports/t3/r3/report_scripts/`.

### 2.5 Independent review

One read-only review of the code, of `01_diagnosis.md` and of `02_proposal.md`, by an agent that wrote none of them. It checks:

- the passivity and bit-identity of the replays;
- the definitions of D3;
- that every proposed trigger is a closed-form-free stage-3 quantity;
- the retrospective table.

Findings and dispositions go in `reports/t3/r3/03_checks.md`. A fix that changes code makes a new code commit, and the affected items are re-run.

### 2.6 Commit, push, STOP at G1

1. **Commit order:** P0 (already committed), then the code, then the diagnosis records and reports.
2. **Indexes.** Add a T3-R3 row to `reports/t3/README.md` and a dated section at the top of `docs/STATE.md`.
3. **Push** `t3-r3` (fast-forward).
4. **STOP.** Report:
   - the class table;
   - the one-line answer to DG7;
   - the proposed candidates with their triggers, budgets and costs;
   - anything flagged.

   P2 starts only after my reply "proceed P2".

## 3. P2 — implementation, pre-registration and the development wave (after G1)

### 3.1 Implementation

- Implement the approved scheme as D4 specifies, with any changes from my reply.
- **Tests:**
  - the trigger fires exactly on the specified condition, and never reads a closed-form quantity;
  - a re-initialization uses the specified derived seed and streams;
  - the artefacts of a discarded attempt go to new files, and nothing is overwritten;
  - the continuation is rebuilt from the frozen stage 3;
  - the legacy default reproduces the old path.
- **New tools:**
  - `tools/t3/t3r3_launch.py`: the arms; seeds 70501-70520 plus the diagnostic (60, 60518) runs; guards against any other seed; a plan CSV with config SHA-256 and planned init digests;
  - `tools/t3/t3r3_checks.py`, `tools/t3/t3r3_pair.py`, `tools/t3/t3r3_eval.py` and `tools/t3/t3r3_analysis.py`;
  - `tools/t3/t3r3_blind.py`, which does not import the analysis tool.
- **The independent review** of this code, appended to `03_checks.md`. Its fixes are committed before C-LEGACY.

### 3.2 C-LEGACY

Run it as D5 says, at the final code commit, and record it in `results/t3_r3/legacy/`.

### 3.3 The pre-registration (committed and pushed before any development run)

`reports/t3/r3/04_preregistration.md` contains:

- D5 and D6, filled in with the approved arms, keys, budgets, costs and the selection order;
- D7, verbatim;
- the operating-characteristic tables of the target and the later confirmation, computed by a script;
- the plan CSV;
- the config differences of each arm against `bw`;
- the seed inventories;
- C-LEGACY.

### 3.4 Launch, checks, evaluation, analysis

- Launch, check, evaluate and analyse as D5 says.
- The blind recomputation must agree with the analysis.
- Results go to `results/t3_r3/dev/` and `results/t3_r3/diagnostic/`.

## 4. P3 — reports, fact-check, push

### 4.1 Reports (`reports/t3/r3/`)

- **`05_development.md`,** in this order:
  - the checks;
  - the arms and the code version;
  - the per-seed tables;
  - the target outcome per arm and the selection;
  - the paired comparison with `bw`;
  - the failures of every arm, classified as in D2;
  - the triggers that fired;
  - S3 and the stage-2 stretches;
  - the diagnostic 60518 runs;
  - the cost;
  - the predictions.
- **`summary.md`.**
- **`handoff.md`,** in Chinese only:
  - the diagnosis in brief, the target outcome per arm, and the cost;
  - at most five GitHub links, in the text, at the commit that holds the reports;
  - no name or contact detail.

Every table is generated by a script under `reports/t3/r3/report_scripts/`. Update the T3-R3 row of `reports/t3/README.md`, and add a dated section to `docs/STATE.md`.

### 4.2 Fact-check

An independent, read-only fact-check checks every number in the reports against the CSVs. Its ledger is `reports/t3/r3/pi_record/01_factcheck.md`. Corrections go in before the push.

### 4.3 Push

Push `t3-r3` (fast-forward). Create no tags, leave `main` untouched, and open no pull request.

## 5. STOP at G2

Stop after the §4.3 push.

- Nothing is locked, no confirmation seed is trained, and nothing is tuned after the pre-registration.
- The PI decides at G2 whether the proposed scheme goes to the lock and confirmation of D6 in the next round.
