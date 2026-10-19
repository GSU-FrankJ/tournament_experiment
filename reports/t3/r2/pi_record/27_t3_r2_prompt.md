# Round T3-R2: audit of the four T = 3 BW failures from the records, then (only if no implementation error is found) the lock of BW at T = 3 and its fresh-seed confirmation

## Where the project stands

- **Model.** Two identical players play a T-stage rank-order tournament with terminal prizes (round T3-R1's prompt `reports/t3/r1/pi_record/25_t3_r1_prompt.md`, D2).
  - Each stage a player chooses effort e ∈ [0, 100] at cost k e², with k = 1/3500.
  - Each player draws a shock U[-q, q] per stage.
  - Prizes are w_H = 6 and w_L = 2, so ΔW = 4.
  - The gap moves as d' = d + e - e_opp + ξ, with ξ triangular on [-2q, 2q].
  - B = 100 + 2q, and D_t = [-(t-1)B, (t-1)B].
- **Verifier tiers** (`utils/dp_br_verifier.py`):
  - the final tier `FINAL_CONFIG` has state step 2, effort step 0.5 and 32 Gauss-Legendre nodes per half of the shock support;
  - the development tier `DEV_CONFIG` has state step 4, effort step 1 and 16 nodes per half.
- **Per-stage gains.** η_t is the largest one-step deviation gain at stage t over the stage-t grid, divided by ΔW (column `Delta_max_t{t}_over_dw`). G_t(d) = V_t^BR(d) - V_t^mean(d) is the gain of a dynamic best response. Gmax_full is its maximum over t and states, divided by ΔW (`Gmax_full_over_dw`).
- **T = 2 is settled.** Solver v2.0 (`protocols/v2_T2_locked_v2_0.json`) passed its fresh-seed confirmation on seeds 30501-30520: 19 of 20 runs at q = 50 and 20 of 20 at q = 60.
- **Round T3-R1** (branch `t3-r1`, final commit `be2b7e9fe3e0aad34f7481a1d8306af7cc2963e7`) ran v2.0's design at T = 3: backward-induction training, arm `bw`, by `run/run_ms_stagewise.py` with legacy settings.
  - **Training:**
    - stage 3 is trained for 1600 updates and frozen;
    - stage 2 is trained for 1600 updates against the exact expected continuation of the frozen stage 3, then frozen;
    - stage 1 is trained for 600 updates;
    - every update has 512 episodes, with bin-balanced starts on D_3 and on D_2, and root starts at stage 1.
  - **Configs** are built by `tools/t3/t3_configs.build_bw_config` with `reports/t3/r1/params_bw_t3.json`.
  - **Gates:** `bw` passed G-A ∧ G-F ∧ G-N in 18 of 20 runs at q = 50 (seeds 50501-50520) and 18 of 20 at q = 60 (`results/t3_r1/analysis/gate_counts.csv`).
  - **The four failures** (`results/t3_r1/analysis/bw_failures/points.csv`; `reports/t3/r1/04_t3_results.md` §4.1). "Best response" here is the verifier's dynamic best-response action (column `br`):

    | q | seed | gate failed | value | stage, d | learned mean effort | best-response effort |
    |---|---|---|---|---|---|---|
    | 50 | 50504 | G-A, η_3 | 0.00623 (limit 0.005) | 3, -4 | 59.15 | 67.14 |
    | 50 | 50517 | G-A, η_3 | 0.00568 | 3, -18 | 64.29 | 48.02 |
    | 60 | 50514 | G-F, Gmax_full | 0.05628 (limit 0.01) | 2, -14 | 0.52 | 34.44 |
    | 60 | 50517 | G-F | 0.01089 | 2, -66 | 0.16 | 14.71 |

  - **Near-zero effort.** In the two q = 60 failures the trailing player's stage-2 mean effort is below 1 over a stretch of gaps inside the band |d| < 2q. Seven of the 40 `bw` runs have such a stretch (`results/t3_r1/analysis/bw_failures/zero_effort.csv`). In seed 50514 the development-tier stage-2 gain `Delta` was 0.00551 at local update 400, 0.01611 at 800 and 0.06122 at 1200, and it stayed at or above 0.05505 through the learning-rate decay. T3-R1 tested no cause.
  - **Clamp counters (a preliminary look while this prompt was reviewed).** `ms_updates.csv` counts, per update, the stage-2 learner draws below the action clamp (`d1_L_s2_lo` of `d1_L_s2_n`). Summed over the learning-rate decay of the stage-2 phase (local updates 1201-1600, 204,800 draws), the count was 11,142 to 71,877 in the seven runs with a stretch and at most 13,641 in the other 33 (`results/t3_r1/runs/q{q}/seed{s}/bw/ms_updates.csv`). Early in the phase (local 1-400), high counts also occurred in runs that ended without a stretch.
- **The comparison of training schemes.** T3-R1's comparison (H1, H2, H3, H1-T2; `reports/t3/r1/04_t3_results.md`, `05_t2_comparison.md`; definitions in `reports/t3/r1/prereg_criteria.json`, recomputation in `tools/t3/t3_blind.py`) is not changed by this round. It is only checked, through the audit, against implementation errors.
- **T3-R1's untracked files** live only in its worktree `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472`: weights, `.pt` files, freeze arrays and evaluation arrays. That worktree is a read-only reference root here.

## The PI's plan for this round (binding)

1. **Audit (P1).** Investigate the four T = 3 `bw` failures from the existing records, above all the near-zero effort in the two q = 60 stage-2 failures, and establish whether any T3-R1 result is affected by an error in the code or in the verifier.
   - This is a check of the conclusions and an analysis. No new batch of seeds is trained.
   - The only training is the four bit-identity re-runs of A1.
2. **If no implementation error is found:**
   - keep the current T = 3 `bw` settings unchanged and lock them;
   - check a block of unused seeds;
   - fix the confirmation rule and the failure-counting rule in writing before any run;
   - train and verify `bw` anew on 20 seeds per q for q = 50 and 60, and report the result (P2, P3).
3. **If an implementation error is found:** stop at G1 with an assessment of its impact. The fix, a test on new development seeds and the independent confirmation follow in a later round.
4. **After the T = 3 confirmation,** the paper's analysis and figures follow, in a later round.

## Steps

1. **P0:** housekeeping (§1).
2. **P1:**
   - the code and tests, committed first;
   - the audit A1-A8, run at the code commit;
   - the drafted lock and pre-registration;
   - the independent review and the push (§2).

   Then **STOP at gate G1**.
3. **P2,** only after my reply "proceed P2": the lock, the confirmation wave (40 runs), its checks and evaluation (§3).
4. **P3:** reports, fact-check and push, then **STOP** (§4, §5).

## Environment and standing rules

- **Repository and Python.** The repository is `/home/fjiang4/tournament_experiment`. Python is `/home/fjiang4/tournament_experiment/.venv/bin/python` only. Run `pytest tests`, never a bare `pytest`.
- **Runs.** Every run is single-threaded (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) and runs in tmux. Do not kill jobs you did not start.
- **Branch and worktree.**
  - Work in a new worktree, on a new branch `t3-r2` created from `origin/t3-r1` at `be2b7e9fe3e0aad34f7481a1d8306af7cc2963e7`. Record the worktree path.
  - The first file written there is this prompt, verbatim, as `reports/t3/r2/pi_record/27_t3_r2_prompt.md`.
- **Other worktrees.**
  - Do not touch T3-R1's worktree (above), MS-R4's launch worktree `.claude/worktrees/ms-r4-e1-e2-experiments-2da78b`, or any other worktree.
  - Never run `git clean` in any of them.
  - Read T3-R1's untracked files from its worktree. Do not copy them into git.
- **Where files go.**
  - Results go under `results/t3_r2/`, and reports under `reports/t3/r2/`.
  - New tools go under `tools/t3/`, as new files only, with names `t3r2_*` or `t3c_*`.
  - New tests go in `tests/test_t3r2_*.py`.
- **Provenance:** every number in a report cites a path.
- **Closed form.** The closed-form equilibrium is the final-stage tent `utils.theory_multistage.g2_two_stage` and, at T = 2, g1 = ΔW/(6kq). It enters evaluation, reporting and calculator tests only. It never enters training or anything a run decides.
- **Results and tracking.**
  - Never delete or overwrite results or checkpoints.
  - Tracking follows `reports/ms/README.md` and the standing rules of `reports/t3/r1/pi_record/25_t3_r1_prompt.md`.
  - Never tracked: `.pt` files, `train_history.json`, weight exports, `freeze_stage*_*.npz`, `final_eval_*.npz`, per-run evaluation arrays and `run.log`.
- **Frozen files.**
  - No existing file changes in `agents/`, `envs/`, `utils/`, `run/`, `protocols/`, `tools/v2/`, `tools/ms/` or `tools/t3/`.
  - The one exception is P2's lock (§3.1): it adds two new files to `protocols/` and appends one record to `protocols/LOCK`. No existing protocol file or record changes.
  - If the audit needs a change to an existing file, stop and report.
- **Pushing** happens only where §2.6, §3.1 and §4.3 say. `main` and tags are not touched, and no pull request is opened.

**If any step fails, or anything does not match this prompt, stop at that point and report. Do not work around it.**

---

## 0. Decisions (binding)

### D1. Scope and seeds

- **P1, the audit.**
  - No new seed is trained.
  - The only training runs are the four bit-identity re-runs of A1, for `bw` at (q = 50, 50504), (50, 50517), (60, 50514) and (60, 50517), into `results/t3_r2/reruns/q{q}/seed{s}/bw/`.
  - Everything else in P1 reads records and stored weights and evaluates them.
- **P2, the confirmation.**
  - Arm `bw` only, with the settings locked in §3.1. These are identical to T3-R1's `bw`.
  - q ∈ {50, 60}, seeds 60501-60520, the same 20 seeds at both q: 40 runs.
  - The block is subject to the P0 inventory.
  - No other arm, no tuning, and no change of a setting, gate or threshold.
- **Never trained:** 40501-40520, the MS development seeds 10501-10530, 30501-30520, and the T3-R1 seeds 50500-50520 (except the four A1 re-runs).
- **Out of scope:** T = 4, the paper's figures, and any remedy for a failure.

### D2. Classification of findings (at G1)

The specification of everything the audit checks is:

- T3-R1's prompt, D2-D7;
- the verifier's documented definitions, `MultiStage/METHODS.md` section "Verification quantities" only. The rest of that file describes earlier protocols. Where it differs from T3-R1, T3-R1 holds.

Every finding, each of the four failures and each of the seven stretches is classified as one of the following:

- **(E) Implementation error.** The code or the verifier departs from its specification as discretised. Examples:
  - a sign, a stage or role index, or an observation encoding;
  - a continuation table, or the snapshot that was evaluated;
  - the configuration a run executed, or an analysis step that computes a reported value;
  - verifier numerics outside their documented discretisation.

  Every (E) triggers the error path (plan item 3), whether or not it changes a verdict.
- **(N) Numerical resolution.** The implementation matches its specification, and a value differs from the independent calculator or the refined tier only within a stated resolution. A verdict flip of a value that lies within 1e-4 ΔW of its limit also counts here. (N) is reported and does not trigger the error path.
- **(R) Record or report error.** The error is confined to a record or a report text and changes no computed value. It is logged in `01_audit.md` and does not trigger the error path. T3-R1's files are not edited.
- **(L) Learning failure.** The implementation is correct, but training produced a policy that is far from a best response somewhere.
- **(U) Unexplained.**

Each (E) comes with its impact:

- the runs, arms, metrics, gates and hypotheses it affects;
- recomputed values, where the stored weights allow it.

No fix is made in this round.

### D3. The audit A1-A8 (P1; diagnostic; no recorded value or verdict of T3-R1 is changed)

**A1. Reproduction.**

- Re-run each of the four failing `bw` runs from its own T3-R1 `run_config.json`, passed unchanged, with `--out-dir results/t3_r2/reruns/q{q}/seed{s}/bw/`.
- Compare each re-run with its T3-R1 original in two ways:
  - with `tools/t3/compare_runs.py`, which covers the 152 weight exports, the freeze arrays, both continuation tables, the bin maps, the CSVs without their wall-clock columns, and the final full state;
  - by running `tools/t3/t3_eval.evaluate_run` on the re-run and comparing it with the run's row in `results/t3_r1/eval/per_run.csv`, on every metric column.
- Both comparisons must agree bit for bit.

**A2. Snapshot identity, for all 40 T3-R1 `bw` runs.** In `state_end_stage1.pt`, each of the following must hold bit for bit:

- `frozen_stages["3"]` equals the export `weights/u01600.npz`;
- `frozen_stages["2"]` equals `weights/u03200.npz`;
- the live actor (`agent`) equals `weights/u03800.npz`.

**A3. Continuation tables, for the four failing runs and four passing runs** (q = 50 seeds 50501 and 50502; q = 60 seeds 50501 and 50502).

1. **Decisive check.** Rebuild `continuation_table_stage2.npz` and `continuation_table_stage1.npz` from the frozen snapshots with `utils.ms_continuation.build_nested_tables`. The rebuilt tables must be equal bit for bit.
2. **Descriptive check.** Compare each table at 201 points spread evenly over its y-grid with A5's calculator, run at the settings that passed A5's convergence test.
   - For scale: v2.0's check (ii-b) found agreement within 1e-6 ΔW at T = 2 (`protocols/v2_T2_locked_v2_0.md`, section 4).
   - A difference above 1e-5 ΔW is flagged and examined.

**A4. Stage-2 training semantics at T = 3.** Unit tests with hand-computed examples, in `tests/test_t3r2_semantics.py`. Returns are stored in float32; compare with a relative tolerance of 1e-5.

- The observation at (t = 2, T = 3) is [0.5, d/B].
- A learner in role 1 sees -d.
- The stage-2 return of `run.ms_rollout.collect_stage_batch` equals -k e_L² + Ṽ_3(d + e_L - e_O).
- The table lookup refuses an out-of-grid y, as specified.
- The opponent's stage-2 action comes from the lagged copy.
- The stage-3 phase reward is w_L + ΔW F_ξ(d + e - e_opp) - k e².

**A5. The independent best-response calculator** (`tools/t3/t3r2_br_calc.py`).

*Independence.*

- The calculator must not call or reuse the verifier's or the metrics module's numerics: `verify`, `gl_shock_rule`, `stage_grid`, `effort_grid` and `evaluate` of `utils/dp_br_verifier.py` and `utils/v2_metrics.py`.
- An AST test checks that none of these names occurs in `t3r2_br_calc.py`.
- Imports that come in through the network loaders and the mean-effort functions are allowed: `tools/t3/t3_eval.py` (`load_bw`, `load_joint`, `load_v20`, `net_from_export`), `tools/ms/replay_dev_rule.py` (`load_export`, `build_actor`, `candidate_fns`) and `run.run_final_dp_br.make_policy_fns`. The audit tool `t3r2_audit.py`, unlike the calculator, may call the verifier, for its comparisons and for A6 and A8.
- The calculator may use `f_xi` and `F_xi` from `utils.theory_multistage`. A test checks them against numerical integration of the triangular density.

*Policy.* The mean effort ê_t(d) of the loaded networks: float32 networks, float64 means, as in the training evaluation. The opponent plays ê_t(-d).

*Definitions.*

| quantity | definition |
|---|---|
| Q_T(x, e) | w_L + ΔW F_ξ(x + e - ê_T(-x)) - k e² |
| V_T^mean(x) | Q_T(x, ê_T(x)) |
| V_T^BR(x) | max_e Q_T(x, e) |
| W_{t+1}^·(y), t < T | E_ξ[V_{t+1}^·(y + ξ)], for · = mean and · = BR |
| Q_t^m(d, e) | -k e² + W_{t+1}^mean(d + e - ê_t(-d)) |
| V_t^mean(d) | Q_t^m(d, ê_t(d)) |
| Δ_t(d) | max_e Q_t^m(d, e) - V_t^mean(d), with argmax the one-step best-response effort |
| V_t^BR(d) | max_e {-k e² + W_{t+1}^BR(d + e - ê_t(-d))}, with argmax the dynamic best-response effort |
| G_t(d) | V_t^BR(d) - V_t^mean(d) |
| EXP_root | G_1(0) |

For every evaluated state, report Δ, G, both best-response efforts, the learned effort, and the slope ∂Q_t^m/∂e at the learned effort.

*Numerics.* Pure recursion is infeasible at T = 3, so the calculator tabulates.

- **Terminal stage.** Tabulate V_T^mean exactly, and V_T^BR by per-point maximisation, on a grid of D_T with step ≤ 0.05.
- **Expectations.** Tabulate W_{t+1} on a y-grid with step ≤ 0.05. Use composite Gauss-Legendre on panels of width 1 aligned at z = 0 and z = ±2q, with at least 4 nodes per panel and the density in the weights. Interpolate V_{t+1} cubically from its table.
- **Earlier stages.** Compute stage-t values by maximising over e with cubic interpolation of W_{t+1}, and recurse down to t = 1.
- **Effort maximisation.** Search a grid of step 0.01 on [0, 100], then refine with bounded Brent around the best grid point and at both bounds.
- **Convergence test.** The test set is the four failure points, the root of the five runs named below, and 20 stretch nodes of A7. Halve every step and double the nodes per panel. The values Δ, G and V must change by at most 1e-6 ΔW, and the efforts by at most 1e-3. If a value does not converge, refine and report it; this is not a stop.

*Tests.*

- **T = 2 closed form.** For the closed-form policy (e_2* = the tent, e_1*(0) = g1, as functions), Δ_t and G_t must be at most 1e-6 ΔW at the root and at 50 states of D_2. The closed form is an equilibrium for q > q_crit = 41.83 (`utils/theory_multistage.validate_two_stage_params`).
- **Agreement with the verifier.** On five runs, compare the calculator's V_1^mean(0) and V_t^mean at 20 states per stage with the final-tier verifier arrays of the same candidate (keys `v_t{t}_d_grid` and `v_t{t}_v_mean`). For `bw` these are the run's `freeze_stage1_final.npz`, and for `jt` and `jt_es` its `final_eval_final.npz`; T3-R1's C-REPLAY found both equal to the current verifier's output bit for bit (`results/t3_r1/launch_checks.json`, `replay_ok` in 160 of 160 runs). T3-R1 stored no such arrays for `v20` (its `eval/arrays/` files hold only `fin_t{t}_{d_grid,e_hat,delta,G}`), so the audit tool evaluates `t3_eval.load_v20` with `t3_eval.evaluate_tier` on `FINAL_CONFIG`. The five runs are:
  - `bw` q = 50 seed 50501;
  - `bw` q = 60 seed 50501;
  - `jt` q = 50 seed 50501;
  - `jt_es` q = 60 seed 50501;
  - `v20` q = 50 seed 30501.

  Report the differences. If one exceeds 1e-4 ΔW, examine it.

*Where the calculator is applied.*

- (a) The four failure points.
- (b) For every one of the 200 runs of `results/t3_r1/eval/per_run.csv` (160 runs of T3-R1 plus the 40 recorded v2.0 runs):
  - G and Δ at its state (`Gmax_full_t`, `Gmax_full_d`);
  - Δ_T at (T, `Delta_argmax_d_t{T}`).
- (c) EXP_root of all 200 runs.
- (d) Every node of the seven stretches (A7).

*Comparison with the verifier's final tier.* For each value, report the difference to the stored verifier value.

- **(E) or (N).** A difference above 0.001 ΔW (the G-N tolerance) is a candidate (E) and is examined. Smaller differences are (N).
- **Flips.** If the calculator's value would flip a gate verdict of a run, or the verdict of H1, H2, H3 or H1-T2, the flip is reported. It is (N) if the stored value lies within 1e-4 ΔW of its limit, and a candidate (E) otherwise.

**A6. A refined verifier tier, for all 200 runs.** Same code, finer grid: `VerifierConfig("refined", state_step=1.0, effort_step=0.25, gl_half=64)`.

- Report Gmax_full, η_T, EXP_root and the per-stage maxima, with |refined - final|.
- Report the G-A and G-F verdicts on the refined tier. G-N compares the development and final tiers and is not defined for a third tier.
- The refined tier is diagnostic and changes no recorded verdict. A verdict that differs from the final tier is classified as in A5.

**A7. The near-zero stretches.** For the seven runs of `zero_effort.csv`, with the other 33 `bw` runs as comparison:

1. **Formation.** From the 64 stage-2 exports of each run (`u01625` ... `u03200`, the live actor during the stage-2 phase), evaluate ê_2(d) on the final-tier grid of D_2.
   - Per stretch node, report the first export with mean effort below 1, and how the stretch grew.
   - Figure: nodes against updates.
2. **The objective.** At every stretch node, compute with A5 Q_2^m(d, e) over e ∈ [0, 100]. Use the frozen stage 3 (`u01600`) and the opponent of the same export. Do this at two exports: the first at which the node's mean effort is below 1, and `u03200`. Report:
   - the slope at the learned effort;
   - whether there is a local maximum at or near e = 0;
   - the distance to the global maximum, and how much higher it is.
3. **The policy's spread.** At the stretch nodes, over the exports: the Beta parameters α and β and the normalised standard deviation.
4. **The training signal,** from `ms_updates.csv`, in the formation window and in the decay window (local 1201-1600):
   - `kl_final_epoch`, `clip_frac`, `adv_raw_mean`, `adv_raw_std`;
   - the realised starts per stratum;
   - the clamp counters `d1_L_s2_lo` / `d1_L_s2_n`, the share of stage-2 learner draws below the action clamp.
5. **Classification of nodes.** A stretch node is "equilibrium-like" where the one-step best-response effort is below 1, and a "policy error" otherwise. Report the counts.
6. **The other runs.** The same quantities at the same nodes for the 33 runs without a stretch.

**A8. The q = 50 G-A failures and last-iterate variability.**

1. A5's Δ_3 at the two failure points.
2. For all 40 `bw` runs, the final-tier η_3 at each of the last 16 stage-3 exports (`u01225` ... `u01600`), with the export's network used at every stage (η_3 depends only on stage 3). Report per run the minimum, median and maximum. Report per q the number of runs with η_3 above 0.005 at one or more of the 16 exports.
3. The same for the shape parts of G-A, `rmse_pos_T_over_eT0` and `tail_mean_T_over_eT0`.

**Order and stops.**

- The audit runs at the code commit (§2.6), on a clean tree.
- A mismatch in A1, A2 or A3 step 1 is a stop-and-report. Complete the read-only parts that do not depend on the mismatching item, then stop at G1 with it.
- A part that a review fix touches is re-run, and the re-run is recorded.

**The G1 report.** `reports/t3/r2/01_audit.md` contains:

- each of A1-A8 with its outcome and the paths of its data;
- the D2 class of every finding, of each failure and of each stretch;
- for any (E), its impact;
- one line: either "No implementation error found" or "Implementation error found: ...".

### D4. The confirmation (P2; written in P1, before any confirmation run)

- **Locked settings: exactly T3-R1's `bw`.**
  - The v2.0 record of q, with `game.T` = 3.
  - `reports/t3/r1/params_bw_t3.json`; record its SHA-256.
  - Budgets {3: 1600, 2: 1600, 1: 600}.
  - LR 3e-4, linear 3e-4 -> 3e-5 over local updates 1201-1600 at stages 3 and 2, and over 1-600 at stage 1.
  - Bin-balanced starts, the rule disabled, `init_digest` true.
  - The code of `agents/`, `envs/`, `utils/`, `run/`, `tools/ms/` and the existing files of `protocols/` as at `be2b7e9f`. These are unchanged since T3-R1's launch.
- **The confirmation configs.**
  - They differ from T3-R1's `bw` configs of the same q only in `seed`, `run`, `record.run`, `record.seed`, `record.output_dir` and `pilot` (= "t3_r2").
  - Run names are `t3r2_q{q}_s{seed}_bw`.
  - A test checks this.
- **Evaluation and gates.** Exactly T3-R1's: D6 items 1-8 of its prompt; `tools/t3/t3_eval.py` and `tools/t3/t3_metrics.py`; the gates part of `reports/t3/r1/prereg_criteria.json`.
  - The evaluated policy is the last iterate, as Beta means, on the final and development tiers.
  - G-A: η_3 ≤ 0.005, `rmse_pos_T_over_eT0` ≤ 0.05 and `tail_mean_T_over_eT0` ≤ 0.02.
  - G-F: `Gmax_full_over_dw` ≤ 0.01.
  - G-N: the dev-final differences of η_3 and of Gmax_full are ≤ 0.001.
  - Every "≤" is inclusive, at full float64 precision.
- **When a run passes.** It exits with code 0, its global-RNG assertion is `ok`, it has a valid evaluation, and it passes G-A, G-F and G-N.
- **The confirmation rule** (the PI's decision, fixed now).
  - The confirmation **passes** iff at least 36 of the 40 runs pass, **and** at least 17 of 20 at q = 50 pass, **and** at least 17 of 20 at q = 60 pass. Otherwise it fails.
  - Rationale: the confirmation must reproduce T3-R1's development pass rate (36 of 40), with neither q below 17 of 20.
  - Operating characteristic. With per-run pass probability p, the same at both q and independent between the q blocks, the rule passes with probability 0.071 at p = 0.80, 0.249 at 0.85, 0.607 at 0.90 and 0.942 at 0.95. A script recomputes these and prints them in the pre-registration.
- **Failure counting.**
  - **Reporting.** Every planned run is reported. No seed is replaced or re-drawn.
  - **Infrastructure failures.** An attempt is an infrastructure failure only if one of these holds:
    - a negative return code (a signal, e.g. SIGKILL);
    - an out-of-memory kill recorded in the kernel log or the run's log;
    - `OSError` ENOSPC recorded in the run's `status.json` or traceback.

    The launcher, or the operator, records every such attempt with its evidence in `results/t3_r2/confirmation/infra_events.json`. The analysis and the blind tool both read that ledger.
  - **Re-run after an infrastructure failure.** The attempt is moved to `results/t3_r2/crashed/` and the run is repeated once into the original path. If the repeat also fails for an infrastructure reason, the confirmation stops without a verdict, and that run is counted neither way.
  - **Other failures.** Any other nonzero exit, and any global-RNG violation, is a failed run.
  - **Evaluation problems.** An evaluation error, or an evaluation with `valid` = False, counts as a failed run unless a re-run of the locked evaluator on the same weights succeeds. The re-run is reported. A completed run is never re-trained.
- **Reported, deciding nothing.**
  - The pass counts per q, with exact Clopper-Pearson 95 % intervals.
  - Per failing run, the diagnosis of `reports/t3/r1/04_t3_results.md` §4.1, plus the A7 analysis applied to it.
  - The near-zero stretches of all 40 runs.
  - The pass rate pooled with T3-R1's 40 development runs (80 runs), with its interval, labelled descriptive.
  - The gate values set against T3-R1's.
- **If the confirmation fails,** nothing is changed or tried in this round. The PI decides the next step.

### D5. Predictions (the PI's, written before P1; expectations, not tests)

- **AP1.** The audit finds no implementation error. A1, A2 and A3 step 1 match bit for bit, and A5 and A6 produce no (E).
- **AP2.** The seven stretches, the two q = 60 failures among them, are learning failures and not features of the game: the stage-2 Beta policy collapsed onto the lower action clamp and did not recover.
  - At most stretch nodes the one-step best-response effort is above 1. There, Q_2^m still increases with effort at the learned effort: there is no local maximum near zero.
  - At these nodes the Beta policy has α < 1, so that a large share of its draws falls below the clamp.
- **AP3.** At q = 50, several runs besides the two failures have η_3 above 0.005 at one or more of their last 16 exports. The two G-A failures are last-iterate variability, not a different kind of policy.
- **CP1.** No prediction of the confirmation's outcome. Its operating characteristic is given in D4.

### D6. Reference roots

Every tool takes these as explicit arguments and records them. A missing file is a stop-and-report.

- **T3-R1 untracked root:** `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472/results/t3_r1/`. It holds:
  - `runs/q{q}/seed{s}/{bw,jt,jt_es}/` and `runs_t2/q{q}/seed{s}/jt2/`, with weights, `state_end*.pt` and freeze arrays;
  - `eval/arrays/`.

  T3-R1's report script `reports/t3/r1/report_scripts/bw_failures.py` reads this root through the variable `T3R1_UNTRACKED_ROOT`. The tools of this round take it as an argument.
- **T3-R1 tracked records,** on the branch: `results/t3_r1/` and `reports/t3/r1/`.
- **The v2.0 confirmation runs:** `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0/q{50,60}/seed{30501..30520}/`.

---

## 1. P0 — housekeeping

Write `reports/t3/r2/00_housekeeping.md` as you go: commands, hashes and verbatim outputs.

1. **Refs.**
   - Run `git ls-remote origin` for `main`, `t3-r1`, `ms-r4` and the tags.
   - Verify that `origin/t3-r1` = `be2b7e9fe3e0aad34f7481a1d8306af7cc2963e7`.
   - Record, but do not fix, local refs that are behind. T3-R1's worktree has a local `t3-r1` at `ea727dcb`; leave it.
2. **Branch and worktree.** Create `t3-r2` from `origin/t3-r1` in a new worktree, record its path, and save this prompt there first.
3. **Seeds.**
   - Create `results/t3_r2/preflight/`.
   - Run `python tools/v2/seed_inventory.py --block 60501 60520 --out results/t3_r2/preflight/seed_inventory_60501_60520.csv`. Exit code 1 means it found hits.
   - List every hit file with `tools/t3/seed_hits.py` (or an equivalent script of your own) into `results/t3_r2/preflight/seed_hits_60501_60520.csv`.
   - **Stop-and-report** on a hit in a training run record of the project's tournament lines. Such a record is a `run_config.json`, `manifest.json`, `status.json`, `gates.json` or run directory under `results/`, `experiments/` or `MultiStage/`, in any worktree.
   - Seed-inventory outputs are not collisions by themselves. List and classify all other hits.
4. **Reference roots.** Verify D6 and record:
   - the 40 `bw`, 40 `jt`, 40 `jt_es` and 40 `jt2` run directories of T3-R1, with their export counts (152, 50, 70 and 44) and their `state_end*.pt`;
   - the 40 `v20` directories, with 88 exports each;
   - `eval/arrays/`.
5. **Machine.** Record `nproc`, load and free disk, and the Python, torch and numpy versions against the v2.0 record (3.12.3, 2.5.1+cu121, 2.5.0).

## 2. P1 — code, audit and the drafted confirmation (on `t3-r2`)

### 2.1 Code (new files only)

- **`tools/t3/t3r2_br_calc.py`:** the calculator of A5.
- **`tools/t3/t3r2_audit.py`:** A2, A3 and A5-A8.
  - It is pure and moves no RNG.
  - It runs in parallel over runs, with at most 40 workers.
  - Its output goes to `results/t3_r2/audit/`.
- **`tools/t3/t3c_launch.py`:** the confirmation launcher, wave `t3c`.
  - **Reused from `tools/t3/t3_launch.py` by import, unchanged:** `job_command`, `run_pool`, `plan_rows`, `relpath_sha256`, `write_plan_csv` and `check_params_file`.
  - **Its own seed guard and production check.** `t3_configs.check_seeds`, `WAVES` and `production_problems` reject t3c by design.
  - **The configs** come from `t3_configs.build_bw_config`, with the D4 fields then set.
  - **Guards:**
    - seeds exactly 60501-60520 at both q;
    - the arm is `bw` only;
    - a config that differs from the lock outside the D4 fields is refused.
  - **`--plan-csv`** (used in P1) writes, for every run, the run name, the output directory, the config SHA-256 and the planned init digest. The init digest is computed by constructing the run without training.
  - **At launch,** the launcher reads the T3 record from `protocols/LOCK` and refuses if the record is absent. It then refuses unless:
    - the lock JSON and MD match the record's SHA-256;
    - every tool file the lock JSON lists matches its recorded SHA-256.

    With `--plan-csv` it checks against the drafts in `reports/t3/r2/lock_draft/` instead.
  - **Run directories:** `results/t3_r2/confirmation/q{q}/seed{s}/bw/`.
- **`tools/t3/t3c_eval.py`:** evaluation of the confirmation runs.
  - It builds the 40 `t3c` jobs and calls `tools/t3/t3_eval.py`'s `evaluate_run`, `write_rows` and `replay_mismatches`.
  - `t3_eval.py`'s own CLI accepts only T3-R1's waves.
- **`tools/t3/t3c_checks.py`:** the post-launch checks of §3.2.
- **`tools/t3/t3c_analysis.py`:** the verdict and the reported items of D4. It reads `infra_events.json`.
- **`tools/t3/t3c_blind.py`:** an independent recomputation of the counts and the verdict from the evaluation table and `infra_events.json`. It does not import `t3c_analysis.py`.
- **`tools/t3/t3r2_make_lock.py`:**
  - It writes `v2_T3_locked_v1_0.json` and `v2_T3_locked_v1_0.md` into a directory given by `--out`.
  - The files contain the settings of D4, with the embedded v2.0 records of both q and the parameters file, plus the gates, the confirmation rule, the failure counting and the seed block.
  - They also contain the SHA-256 of every tool file that the confirmation uses: `tools/t3/t3c_*.py`, `tools/t3/t3_eval.py`, `tools/t3/t3_metrics.py`, `tools/t3/t3_configs.py`, `tools/t3/t3_launch.py` and `tools/ms/ms_configs.py`.
  - The files hold no timestamp, no commit hash and no absolute path, so P1's draft and P2's lock are byte-identical.

### 2.2 Tests (`tests/test_t3r2_*.py`, plus every existing suite)

- **Calculator:**
  - the tests named in A5;
  - F_ξ and f_ξ against numerical integration;
  - the convergence test;
  - the AST independence test.
- **A4's semantics tests.**
- **Audit tool.** Build a T = 3 game with hand-made stage-2 and stage-3 policies, with a planted zero-effort stretch in the stage-2 policy, given as a sequence of synthetic exports. The tool must detect the stretch, the export at which it formed, the slope of Q at the learned effort, and its classification.
- **Launcher (`t3c`):**
  - the seed and arm guards, and 40 jobs;
  - each config differs from T3-R1's `bw` config of the same q only in the D4 fields (compared with `tools/t3/t3_configs.config_diff`);
  - a modified lock file, a modified listed tool, and a missing LOCK record are each refused;
  - `--plan-csv` emits the init digests.
- **`t3c_eval.py`:** on a reduced `bw` run it reproduces that run's freeze arrays bit for bit.
- **Analysis and blind tools,** on synthetic tables with known verdicts:
  - a fail at 18 + 17 = 35 of 40;
  - a fail at 16 + 20 = 36 of 40 (the per-q floor alone);
  - a pass at 18 + 18;
  - a pass at 17 + 19;
  - an evaluation error counted as failed;
  - a stop without a verdict on a second infrastructure failure, read from `infra_events.json`.
- **The full suite.** `pytest tests` passes, apart from:
  - the known `test_registry_canonicalization`;
  - the failures that T3-R1 recorded as known (`results/t3_r1/preflight/tests/two_non_t3_failures.txt`).

  Record the summary line.

### 2.3 The audit

- **Running it.** Run A1-A8 (D3) at the code commit, on a clean tree.
  - A1 is the only training: four runs, at most 4 workers, in tmux.
  - Record the cost of every step.
- **The report.** Write `reports/t3/r2/01_audit.md`, holding:
  - the tables and figures of A1-A8, every number with its path;
  - the D2 classification;
  - the one-line statement.

### 2.4 The drafted confirmation (committed in P1, before any confirmation run)

- **The lock drafts.** Run `tools/t3/t3r2_make_lock.py --out reports/t3/r2/lock_draft/`.
- **The pre-registration,** `reports/t3/r2/02_preregistration.md`. It contains:
  - D4 and D5, verbatim;
  - the operating-characteristic table, computed by a script under `reports/t3/r2/report_scripts/`;
  - the planned 40 configs with their SHA-256 and planned init digests (`confirmation_plan.csv`, from `t3c_launch.py --plan-csv`);
  - the config difference to T3-R1's `bw` for each q;
  - the lock drafts with their SHA-256;
  - the seed inventory;
  - the launch plan: 40 runs, at most 40 workers.

### 2.5 Independent review

One read-only review of the diff and of `01_audit.md`, by an agent that did not write them. It covers:

- the calculator's definitions against D3 and `MultiStage/METHODS.md` "Verification quantities", its numerics, its tests, and its independence;
- the audit's comparisons and its D2 classification;
- the launcher's guards and lock checks, and the identity of the confirmation configs with T3-R1's;
- the rule and the failure counting in the analysis and blind tools.

Findings and dispositions go in `reports/t3/r2/03_checks.md`. A review fix that changes code makes a new code commit, and the audit parts it touches are re-run.

### 2.6 Commits, push, STOP at G1

1. **Commit order.**
   - First the code and tests (the code commit).
   - Then the audit records and `01_audit.md`.
   - Then the pre-registration and the lock drafts.

   After the code commit, commits touch only `results/t3_r2/`, `reports/t3/`, `protocols/` (P2's lock only) and `docs/STATE.md`, except for code commits from review fixes.
2. **Indexes.** Add the T3-R2 row to `reports/t3/README.md`, and a dated section at the top of `docs/STATE.md` (P1 done, gate G1).
3. **Push** `t3-r2` (fast-forward, no force).
4. **STOP.** Report three things: the audit's one-line statement, the D2 classification table, and anything flagged.
   - P2 starts only after my reply "proceed P2".
   - If the audit finds an (E), there is no P2 in this round.

## 3. P2 — the lock and the confirmation (after gate G1)

### 3.1 The lock

1. **Write the lock files.** Run `tools/t3/t3r2_make_lock.py --out protocols/`. The files must equal the P1 drafts byte for byte.
2. **Lock commit.** Commit the two files.
3. **LOCK record.** Append one JSON record to `protocols/LOCK`, in the format of its existing records, and commit it separately (a commit cannot name its own hash). The record holds:
   - the protocol and MD SHA-256;
   - the lock commit;
   - the date;
   - the entry: `run/run_ms_stagewise.py` through `tools/t3/t3c_launch.py`;
   - the rule "nothing in protocols/ changes after this lock without a new version number and a stated reason".

   No existing record changes.
4. **Push** both commits.

### 3.2 Launch and checks

- **Launch.** Wave `t3c`, 40 runs, from the commit that holds the LOCK record, on a clean tree, with at most 40 single-threaded workers, in tmux.
- **Recorded at launch:**
  - `nproc`, load, free disk, HEAD and `git status --porcelain`;
  - `git diff --stat be2b7e9f HEAD -- agents envs utils run tools/ms tools/v2`, which must be empty. Under `tools/t3/` only new `t3c_*` and `t3r2_*` files may appear.
- **Crash rule:** as in D4.
- **Post-launch checks,** in `results/t3_r2/confirmation/launch_checks.json`. Deterministic failures are a stop-and-report. Wall-clock columns are excluded.
  - **Basics:** manifests at the launch commit with a clean tree; exit codes; complete files; the global-RNG assertions; `infra_events.json` present (empty if there was no event).
  - **Budgets:**
    - updates 1600, 1600 and 600 per phase;
    - policy rows 819,200, 819,200 and 307,200;
    - minibatch steps 76,000.
  - **Schedules:**
    - the LR at every update;
    - snapshot refreshes at the phase entries and at multiples of 20;
    - 152 exports (`u00025` ... `u03800`);
    - 64, 64 and 24 checks per stage.
  - **C-INIT:** each recorded init digest equals the planned one in `confirmation_plan.csv`.
  - **C-REPLAY:** `tools/t3/t3c_eval.py` reproduces each run's `freeze_stage1_{final,development}.npz` bit for bit.

### 3.3 Evaluation and verdict

- **Evaluation.** Run `tools/t3/t3c_eval.py` on the 40 runs, into `results/t3_r2/confirmation/eval/`.
- **Analysis.** Run `tools/t3/t3c_analysis.py`, into `results/t3_r2/confirmation/analysis/`. It produces:
  - the per-run table;
  - the pass counts per q with their intervals;
  - the verdict under D4;
  - the per-failure diagnosis, using the A7 tools on the new runs;
  - the stretches of all 40 runs;
  - the pooled descriptive table;
  - the gate values against T3-R1's.
- **Blind recomputation.** `tools/t3/t3c_blind.py` recomputes the counts and the verdict. The two must agree.

## 4. P3 — reports, fact-check, push

### 4.1 Reports (`reports/t3/r2/`)

- `00_housekeeping.md`, `01_audit.md`, `02_preregistration.md` and `03_checks.md`.
- **`04_confirmation.md`,** in this order:
  - the checks;
  - the locked settings and the code version;
  - the per-seed table;
  - the verdict under the rule;
  - the failures, diagnosed and classified;
  - the stretches;
  - the comparison with T3-R1's 40 runs (descriptive);
  - the cost;
  - the predictions against the outcome.
- **`summary.md`.**
- **`handoff.md`,** for the coworker, in Chinese only.
  - It gives the audit's outcome and the confirmation's verdict with the counts.
  - It has at most five GitHub links, placed in the text and pointing at the commit that holds the reports.
  - It contains no name or contact detail.

Every table is generated by a script under `reports/t3/r2/report_scripts/`. Update the T3-R2 row of `reports/t3/README.md`, and add a dated section to `docs/STATE.md`.

### 4.2 Fact-check

An independent, read-only fact-check checks every number in the reports against the CSVs. Its ledger is `reports/t3/r2/pi_record/01_factcheck.md`. Corrections go in before the push.

### 4.3 Push

- Push `t3-r2` (fast-forward, no force).
- Do not touch `main`, create no tags, and open no pull request.

## 5. STOP

Stop after the §4.3 push, or at G1 if the audit finds an implementation error.

- No tuning, no second confirmation wave, no other arm.
- No change after the pre-registration.
- The next round is decided by the PI with the coworker.
