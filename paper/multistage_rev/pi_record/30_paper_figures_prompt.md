# Paper figures and table data (round P-FIG): produce exactly the items listed, nothing else

You produce five figures and the data of four tables for the revised paper, from existing records. Each item must be traceable to its source. **Produce only what is listed here:** no other figures, tables, analyses, runs or prose. New computation is limited to evaluating stored policies, plus the new diagnostic evaluations of F2. Nothing is trained.

## Context (all facts you need)

- **Model.** A T-stage, two-player rank-order tournament.
  - Effort e ∈ [0, 100] at cost k e², with k = 1/3500.
  - Prizes w_H = 6 and w_L = 2, so ΔW = 4.
  - The score gap moves as d' = d + e - e_opp + ξ, with ξ triangular on [-2q, 2q].
  - B = 100 + 2q, and the state domain at stage t is D_t = [-(t-1)B, (t-1)B]. Stage 1 has the single state d = 0.
  - q ∈ {50, 60}.
- **The closed-form equilibrium** (`utils/theory_multistage.py`):
  - the terminal stage follows the tent `g2_two_stage`, which is positive for |d| < 2q and zero for |d| ≥ 2q;
  - at T = 2, stage 1 has g1 = ΔW/(6kq) (`g1_two_stage`);
  - it exists only for the terminal stage (and stage 1 at T = 2).
- **Verifier.** `utils/dp_br_verifier.py` with `utils/v2_metrics.py`. The final tier `FINAL_CONFIG` has state step 2, effort step 0.5 and 32 Gauss-Legendre nodes per half of the shock support; the development tier `DEV_CONFIG` has 4, 1 and 16.
- **Quantities and their stored names:**

  | quantity | stored name |
  |---|---|
  | one-step deviation gain Δ_t(d) | `delta` |
  | dynamic best-response gain G_t(d) = V_t^BR(d) - V_t^mean(d) | `G` |
  | dynamic best-response effort | `a_br` |
  | one-step best-response effort | `a_dev` |
  | learned mean effort | `e_hat` |
  | BR-reachable mask | `reach` |
  | η_T, the largest Δ_T over D_T, / ΔW | `Delta_max_t{T}_over_dw` |
  | G_max,full, the largest G over all stages and states, / ΔW | `Gmax_full_over_dw` |
  | root exploitability, G_1(0) / ΔW | `EXP_root_over_dw` |
  | terminal recovery | `rmse_pos_T_over_eT0`, `tail_mean_T_over_eT0`, `peak_T_signed`, `peak_T_abs` |
  | T = 2 stage-1 error | `stage1_rel_err_signed`, `stage1_rel_err_abs` |

  All of these are defined in `tools/t3/t3_metrics.py`, `utils/v2_metrics.py` and `MultiStage/METHODS.md` ("Verification quantities").
- **Protocols:**
  - **T = 2 v2.0** (`protocols/v2_T2_locked_v2_0.{json,md}`). Confirmation seeds 30501-30520 at both q: 19 of 20 pass at q = 50 (the failed run is seed 30510) and 20 of 20 at q = 60. The evaluated policy takes stage 2 from export `u01600` and stage 1 from `u02200` (`tools/t3/t3_eval.load_v20`).
  - **T3-1.0** (`protocols/v2_T3_locked_v1_0.{json,md}`): the backward-induction `bw` settings of round T3-R1, locked in round T3-R2.
  - **T3-1.1** (`protocols/v2_T3_locked_v1_1.{json,md}`): `bw` plus the verifier-gated restart of stage 3, `stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`. Its confirmation on seeds 80501-80540 at both q passed: 40 of 40 at q = 50, 39 of 40 at q = 60, 79 of 80 overall. The only failure is q = 60, seed 80502. The restart fired in 5 runs, all at q = 50, so the first-attempt count is 74 of 80.
- **Seed blocks:**

  | seeds | round | what ran there |
  |---|---|---|
  | 50501-50520 | T3-R1 | `bw`, `jt`, `jt_es` (paired) |
  | 60501-60520 | T3-R2 | T3-1.0 confirmation: 34 of 40, not passed |
  | 70501-70520 | T3-R3 | development |
  | 80501-80540 | T3-R4 | T3-1.1 confirmation |

- **Untracked roots,** read-only; never clean them:
  - T3-R1: `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472/results/t3_r1/` (`runs/q{q}/seed{s}/{bw,jt,jt_es}/`, `eval/arrays/`);
  - T3-R4: `/home/fjiang4/tournament_experiment/.claude/worktrees/t3-r4-lock-confirm/results/t3_r4/confirmation/q{q}/seed{s}/bwA/`;
  - v2.0 runs: `/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0/q{q}/seed{s}/`.
- **Tracked records** (on the branch):
  - `results/t3_r1/eval/per_run.csv`: arms `bw`, `jt`, `jt_es`, `jt2`, and `v20` for the 40 v2.0 confirmation runs evaluated with the same evaluator;
  - `results/t3_r1/analysis/{gate_counts,criterion,budget_check,cost}.csv`;
  - `results/t3_r4/confirmation/eval/per_run.csv` and `results/t3_r4/confirmation/analysis/{counts,cost,fired,failures,headline}.*`;
  - `results/v2_T2_locked/calibration/calibration_locked.csv`, the lock-time verifier calibration of the exact equilibrium (`analytic_eq`) and the zero-effort policy (`zero`), q = 50 and 60, final and development tiers; `reports/v2/t2_report/tables/T12_calibration.md` describes it.

## Rules

- **Repository and Python.** The repository is `/home/fjiang4/tournament_experiment`. Python is `.venv/bin/python` only, and every process runs single-threaded.
- **Branch and location.**
  - Create a new worktree and branch `paper-figs-1026` from `origin/t3-r4` at `49cd402e882cb62ab3dcadbe85e5ebdd464c749d`.
  - Save this prompt verbatim as `paper/multistage_rev/pi_record/30_paper_figures_prompt.md`.
  - All outputs go under `paper/multistage_rev/` only. Do not modify `paper/multistage/`, `paper/generator/` or any existing file.
- **No training, no changes to code or records.**
  - Load stored networks with the existing loaders: `tools/t3/t3_eval.load_v20`, `load_bw`, `load_joint`; mean effort through `tools/ms/replay_dev_rule.candidate_fns`.
  - Recompute final-tier verifier results with `tools/t3/t3_eval.evaluate_tier(cand, FINAL_CONFIG)` where needed.
  - Any recomputed `Gmax_full`, η_T or `EXP_root` must equal the stored per-run value exactly. A mismatch is a stop-and-report.
- **New diagnostic evaluations** (F2 only) are labelled "new diagnostic evaluation" in the figure, in its data file and in PROVENANCE.
- **Labels and notation** use exactly these terms: score gap d, mean effort, closed-form equilibrium, tie point (d = 0), positive-effort region (|d| < 2q at the terminal stage), theoretical zero-effort tails (|d| ≥ 2q at the terminal stage), dynamic best-response effort, one-step deviation gain Δ_t(d)/ΔW, state-wise dynamic gain G_t(d)/ΔW, BR-reachable region, state-visitation distribution, root exploitability, full-domain dynamic best-response gain G_max,full/ΔW, η_T/ΔW. Do not coin other terms.
- **Figure format.**
  - matplotlib; vector PDF plus PNG at 300 dpi.
  - Full width 7 in, fonts at least 8 pt, a colour-blind-safe palette, no titles inside the figures.
  - q = 50 and q = 60 side by side.
  - For every figure, `paper/multistage_rev/data/F{n}.csv` holds the plotted values, and `paper/multistage_rev/scripts/F{n}.py` regenerates the figure from that data file.
- **Table format.** For each table, `paper/multistage_rev/tables/T{n}.csv` and `T{n}.md`. Every value has a source column (file path and column, or protocol file and key).
- **Provenance.** `paper/multistage_rev/PROVENANCE.md` lists, per figure and table:
  - the runs used, the source files and columns, and the commit;
  - every transformation;
  - every new diagnostic evaluation.
- **Checks against the facts above.** Pass counts that differ from 19/20, 20/20, 40/40, 39/40, 79/80 and 74/80 (first attempt) are a stop-and-report.

## Figures

**F1. T = 2 analytical recovery.**

- **Runs:** the 40 v2.0 confirmation runs (20 per q). Mean effort from `load_v20` and `candidate_fns`.
- **Stage 2:** learned mean effort e_2(d) on D_2 at step 0.5, for all 20 runs per q. Draw them as thin lines, and seed 30510 at q = 50 (the failed run) in a distinct style. Add the closed-form equilibrium `g2_two_stage`.
- **Stage 1:** the single value e_1(0) per run, as points (the failed run marked), against g1.
- **Mark** the tie point, the positive-effort region and the theoretical zero-effort tails.

**F2. T = 2 verifier calibration** (final tier). Compare η_2/ΔW, G_max,full/ΔW and root exploitability across these policy types:

- **Exact equilibrium and zero effort,** at both q: the existing records in `results/v2_T2_locked/calibration/calibration_locked.csv`, tier `final`.
  - Before using them, re-evaluate both policies with the code you use for the new evaluations. It must reproduce the stored values to at most 1e-12 ΔW; otherwise stop and report.
- **Frozen v2.0 policies:** the 40 `v20` rows of `results/t3_r1/eval/per_run.csv` (final-tier columns), 20 points per q, with seed 30510 marked.
- **Clearly misspecified policies, new diagnostic evaluations,** at both q, with efforts clipped to [0, 100]:
  - (M1) both stages at 0.8 × the closed-form equilibrium;
  - (M2) both stages at 1.2 × the closed-form equilibrium;
  - (M3) stage 2 constant at g2(0) for every d, and stage 1 at g1.
- **Layout:** a log axis. Show values below 1e-8 at a floor marked "≤ 1e-8". Mark the gate limits η_T ≤ 0.005 and G_max,full ≤ 0.01.

**F3. T = 3 effort policies** (T3-1.1 confirmation, 40 runs per q).

- **Mean effort:** per stage against the score gap, as all runs thin, the median bold, and seed 80502 marked.
  - stage 3 on D_3 at step 1, with the closed-form equilibrium tent as reference;
  - stage 2 on D_2 at step 1;
  - stage 1 as the single value e_1(0) per run.
- **The state-visitation distribution** of d at stages 2 and 3, below the effort panels.
  - It is the on-path distribution when both players play the evaluated mean-effort policy from d = 0 at stage 1: `utils/v2_metrics.candidate_pmf` of the final-tier result. It is not the verifier's `pmf` (the BR chain), and not the training starts.
  - Show the median across runs, with the 10-90 % band.
- **Mark** the tie point, and at stage 3 the positive-effort region.

**F4. T = 3 best-response diagnostics** (T3-1.1 confirmation, final tier).

- **Runs:** per q, the passing run with the median G_max,full (the lower seed on a tie), plus the failure q = 60, seed 80502.
- **For each run and each stage t = 1, 2, 3:**
  - learned mean effort against the dynamic best-response effort;
  - the state-wise dynamic gain G_t(d)/ΔW;
  - the one-step deviation gain Δ_t(d)/ΔW, as a separately labelled curve;
  - the BR-reachable region (`reach`), shaded and labelled.

  Stage 1 is shown as points at d = 0.
- **Mark** the G_max,full location of 80502, at stage 2.

**F5. Method comparison** (T3-R1, paired).

- **What:** per q, G_max,full/ΔW of `bw`, `jt` and `jt_es` for seeds 50501-50520, on a log axis.
- **Pairing:** lines connect the three values of the same seed.
- **Mark** the G-F limit 0.01.
- **Source:** `results/t3_r1/eval/per_run.csv`.

## Tables (data only: CSV + Markdown with sources)

**T1. Experimental protocols and gates.** One column each for T = 2 v2.0, T3-1.0 (T3-R1's `bw`) and T3-1.1. The rows:

- **Model parameters:** k, w_H, w_L, ΔW, the effort range, the shock, B and D_t.
- **q values and seed blocks:** development, comparison and confirmation blocks with their outcomes, including T3-1.0's unpassed confirmation of 34 of 40.
- **Training:** stage budgets (updates per stage, episodes per update, policy rows per stage and in total), the learning-rate schedule and the start design.
- **The continuation-value calculation:** the expected-continuation table and its integration rule (`utils/ms_continuation.py`, `utils/v2_continuation.py`).
- **The retry rule:** T3-1.1 only.
- **The evaluated mean-effort policy:** which snapshot or export is evaluated at each stage.
- **The verifier tiers.**
- **The pass criteria as actually applied:** T = 2 uses G-A, G-F, G-N and G-S; T = 3 uses G-A, G-F and G-N. Give the thresholds and the run-pass definition.

Sources: the protocol JSON and MD files, `reports/t3/r1/prereg_criteria.json`, `reports/t3/r1/params_bw_t3.json` and the run configs.

**T2. T = 2 recovery and confirmation** (40 `v20` rows of `results/t3_r1/eval/per_run.csv`).

- **Per q:** the pass count, then the median [min, max] over the 20 runs of:
  - the stage-1 error (`stage1_rel_err_signed`, `stage1_rel_err_abs`);
  - the stage-2 RMSE (`rmse_pos_T_over_eT0`) and tail effort (`tail_mean_T_over_eT0`);
  - the signed and absolute tie-point error (`peak_T_signed`, `peak_T_abs`);
  - η_2/ΔW;
  - G_max,full/ΔW.
- **A separate row** for the failed run, seed 30510 at q = 50, with its failing gate.
- **A row** for the gate limits.

**T3. T = 3 fresh-seed confirmation** (T3-1.1; `results/t3_r4/confirmation/`).

- **Pass counts:** per q and overall (40/40, 39/40, 79/80), with the pre-registered failure limits (at most 4 per q, at most 6 of 80).
- **Per q, median [min, max]** of:
  - the terminal-stage positive-region RMSE (`rmse_pos_T_over_eT0`);
  - the mean effort in the theoretical zero-effort tails (`tail_mean_T_over_eT0`);
  - the signed and absolute tie-point error;
  - η_3/ΔW;
  - root exploitability.
- **G_max,full/ΔW:** the median and the worst, per q.
- **Retry counts per q,** and the cost: mean updates and policy rows per run, the ratio to the 3,800 updates of a run without a retry, and the median wall time per run.
- **A row** for seed 80502 as the sole failure: its failing gates and values, and the stage of its G_max,full.

**T4. Method comparison and cost.**

- **The T3-R1 comparison,** per q:
  - pass counts for `bw`, `jt` and `jt_es` (`results/t3_r1/analysis/gate_counts.csv`);
  - the mean paired differences of G_max,full/ΔW with their pre-registered 95 % intervals, for H1 = `bw` - `jt`, H2 = `bw` - `jt_es` and H3 = `jt_es` - `jt` (`results/t3_r1/analysis/criterion.csv`; bootstrap seed 20261016).
- **Per arm:**
  - policy rows and updates (`results/t3_r1/analysis/budget_check.csv`);
  - runtime, as the median wall time per run (`results/t3_r1/analysis/cost.csv`);
  - the network structure (from the run configs): hidden layers and their width, activation, the Beta parameterisation, and whether stages share one network or `bw` freezes one network per stage;
  - the state-start design: `bw` uses bin-balanced starts on D_3 and D_2 and the root at stage 1; `jt` uses root-only whole-game episodes; `jt_es` uses root starts plus exploring starts at stages 2 and 3, with the shares from its config.
- **The T3-1.1 retry cost,** in its own rows (`results/t3_r4/confirmation/analysis/`):
  - fired runs per q;
  - updates and policy rows per run;
  - the first-attempt count of 74 of 80 next to the final count of 79 of 80, stated as the same confirmation block.

## Check, push, stop

1. **Fact-check.** One independent, read-only check of every table value and every count in PROVENANCE against its source, by an agent that did not produce them. Record it in `paper/multistage_rev/pi_record/factcheck.md`, and correct before the push.
2. **Commit and push.** Commit only `paper/multistage_rev/`. Push `paper-figs-1026` (fast-forward). Leave `main` untouched, create no tags and open no pull request.
3. **STOP.** Report the list of files, anything that did not match the facts above, and nothing else.
