# Project state

Last updated: 2026-04-24

## Current status
- **Round 3/4 COMPLETE (2026-04-20)**: all non-2P scenarios now <3% mean rel gap (Metric B). See §Round 3/4 results below and `docs/round3_round4_report.md`.
- **2P (Round 2, unchanged)**: q=35 rel 4.3%, q=45 rel 2.1%, q=55 rel 2.4%
- **Parameter overhaul (2026-04-08)**: All configs match `docs/experiment_config_040726.md`
- **Pending**: paper artifacts regeneration with Round 3/4 data (next step)
- Two-stage runner deferred to separate task

## Round 3/4 results (2026-04-20) — COMPLETE

All non-2P scenarios brought below 3% mean relative gap (Metric B, 5 seeds each). Detailed per-scenario diagnosis + fixes in `docs/round3_round4_report.md`.

### Final per-scenario rel% (5 seeds each)

| Scenario | q=35 | q=45 | q=55 | Fix applied |
|----------|-----:|-----:|-----:|------|
| Two-Player | 4.30% | 2.08% | 2.36% | (Round 2 unchanged) |
| Three-Player | 0.57% | — | 0.84% | streak fix + ramp + min_updates=300 |
| Het. Cost | 1.07% | — | 0.97% | ramp + eps=0.03 + min_updates=300 |
| Het. Ability | 1.61% | — | **0.64%** | eps=0.03 + min_updates=1000 |

### da q=55 re-run incident (2026-04-20)

First q=55 batch ran in worktree `hardcore-diffie-49ca45` (started ~10:51 Apr 19). Three seeds (s42/45/46) completed and wrote JSONs, one (s43) exited without JSON, one (s44) still training at upd=906 — at which point the entire worktree directory was externally deleted (not by the running Claude session). All JSONs/logs in the deleted tree were lost; only one FD on the s44 tee process briefly held a readable inode, which was lost before salvage succeeded.

Batch restarted in the committed worktree `jovial-leakey-c2c923`. All 5 seeds (42-46) completed cleanly: mean 0.64% ± 0.89% (max 2.22% on s43), all stopped via `exploitability` at upd=1000 (min_updates floor), streak 960-972/1000.

**Mitigation for future long-running batches**: run inside the worktree you intend to keep; do not launch long GPU jobs from a throwaway worktree.

## Figure pipeline (current, 2026-04-24)

`python -m paper.generator make_all` produces 10 figures at `paper/figures/*.{pdf,png}` plus companion CSVs in `paper/data/` and 5 tables in `paper/tables/`. All consumers (BASELINE_OVERRIDES, filename parsing, Metric B) live in `paper/generator/`.

| Figure | Purpose |
|--------|---------|
| convergence_main | Effort-vs-step trajectories per q (PPO vs Gradient, all experiments) |
| kl_dynamics | Approx-KL over training, per q |
| exploitability_dynamics | Periodic exploitability over training |
| beta_evolution | Beta policy α/β parameters over training |
| beta_snapshots | Beta PDF snapshots at 10%/50%/90% of training (2P q=45) |
| ablation_comparison | Entropy / cheap-gate / exploit-disable ablation sweep |
| hyperparam_sensitivity | Appendix — hyperparameter sweep |
| distance_to_equilibrium | Log-scale \|e−e\*\| over training |
| effort_drift | Post-convergence drift statistics |
| equilibrium_recovery_dotplot | Per-seed final effort vs e\* across experiments |

Tables: `environment_config`, `summary_metrics`, `ablation_results`, `final_summary`, `convergence_comparison`.

### Historical note — Phase 3 pipeline (2026-04-09, superseded)

Earlier pipeline produced 12 figures (F1–F4 hero + 8 FA_\* appendix) at `paper/generator/output/figures/`. Files no longer exist on disk; the generator was restructured to write flat names (`convergence_main`, `beta_snapshots`, etc.) into `paper/figures/`. If any paper draft still references `F1_equilibrium_recovery.pdf` etc., update to the current names.

## Parameter overhaul (2026-04-08)

Updated all experiment configs to new parameters that satisfy SOC and participation constraints.

### Config changes applied

| Experiment | k | (w_h, w_l) | q_list | effort_range |
|---|---|---|---|---|
| 2P Set 1 | 0.00055 | (6.5, 3.0) | [35, 45, 55] | [0, 100] |
| 2P Set 2 | via CLI: --k 0.0006 --w_h 8 --w_l 4 | (8, 4) | [35, 45, 55] | [0, 100] |
| 3P | 0.001 | (6.5, 3.0) | [35, 55] | [0, 100] |
| Diff Cost | k1=0.0004, k2=0.00055 | (8, 5.5) | [35, 55] | [0, 100] |
| Diff Ability | 0.0005 | (6.5, 3.0) | [35, 55] | [0, 100] |
| Two Stage | 0.0004 | (6.5, 3.0) | [25, 40, 55] | stage1=[0,100], stage2=[0,100] |

### Theoretical equilibria (verified)

| Experiment | q=35 | q=45 | q=55 |
|---|---|---|---|
| 2P Set 1 | 45.45 | 35.35 | 28.93 |
| 2P Set 2 | 47.62 | 37.04 | 30.30 |
| 3P | 25.00 | — | 15.91 |
| Diff Cost | e1=38.03, e2=27.66 | — | e1=26.54, e2=19.30 |
| Diff Ability | 46.43 | — | 30.37 |

### Files modified
- `config/one_stage_two_players.py` — k, q, q_list, effort_range
- `config/one_stage_three_players.py` — k, q, q_list, effort_range
- `config/one_stage_different_cost.py` — w_h, w_l, q, q_list, effort_range
- `config/one_stage_different_ability.py` — k, q, q_list, effort_range
- `config/two_stage_two_players.py` — effort_bounds_stage2
- `paper/generator/config.py` — per-experiment THEORY_PARAMS, Q_VALUES, updated e_star defaults
- `paper/generator/extract.py` — per-experiment theory param lookup
- `paper/generator/metrics.py` — per-experiment theory param lookup
- `paper/generator/tables.py` — per-experiment q_values and theory params
- `paper/generator/__init__.py` — export new symbols

### Results deleted
- All convergence JSONs, logs, and summary CSVs from previous parameter runs

## Previous critical finding: interior NE validity (2026-03-28)

Still relevant. The new parameters were chosen to satisfy the participation constraint:
- q_crit(2P, k=0.00055) = sqrt(2*3.5/(16*0.00055)) = 28.2 → q=35 passes
- q_crit(3P, k=0.001) = sqrt(3*3.5/(16*0.001)) = 25.6 → q=35 passes
- All experiments now use q >= 35, satisfying both SOC and participation constraint

## Task status

| Task | Status | Notes |
|------|--------|-------|
| parameter-overhaul | **complete** | All configs match experiment_config_040726.md |
| round3-round4-fixes | **complete** | All non-2P scenarios <3% mean rel gap (2026-04-20) |
| paper-figures-tables-revision | **in progress** | Regenerating with Round 3/4 data (2026-04-20) |
| two-stage-runner | deferred | Config and env exist, runner needs to be built |
| runner-refactor | deferred | Post-project cleanup |

## Phase 2 (Metric B Migration) — Complete

### Loader schema (two formats)

| format | experiments | policy_mean_effort | sample_effort_mean |
|--------|------------|-------------------|-------------------|
| flat | two_players, three_players | from JSON `policy_mean_effort` field (deterministic Beta mean) | `(agent1_effort + agent2_effort) / 2` (sample rollout averages) |
| nested | different_cost, different_ability | `(agent1_effort + agent2_effort) / 2` (agents store policy means directly) | NaN (not recorded by nested runners) |
| flat (gradient) | all (gradient method) | `(agent1_effort + agent2_effort) / 2` (deterministic, same as sample) | same as policy_mean_effort |

### Changes

- Renamed column `effort_mean` → `sample_effort_mean` (4 files, 48 call sites)
- Switched all paper-reporting metrics from sample_effort_mean to `policy_mean_effort`:
  - `extract.py`: effort_error (lines 479, 527), convergence step detection (line 608)
  - `metrics.py`: effort_series (line 391)
  - `tables.py`: final summary Mean/Std (lines 348-349, 368-369)
  - `plots.py`: best-seed selection (line 907), F4 distance curve (line 1331), dotplot (lines 1657, 1772), all trajectory plots (lines 1056, 1073, 1212, 1226, 1229)
- Fixed nested loader (extract.py): `policy_mean_effort` computed from agent policy means, `sample_effort_mean` set to NaN
- Fixed flat loader (extract.py): raises ValueError for missing `policy_mean_effort` in PPO runs; falls back to agent average for gradient runs
- All convergence trajectory plots now use `policy_mean_effort` (removes sample noise)

### Verification

- Schema verification passed (verify_loader_schema.py): 6/6 assertions
- Metric B sanity check passed (verify_metric_b.py): 6/6 assertions (flat gap=0.362, nested gap=2.233)

### Regenerated artifacts

- Tables: final_summary.tex, convergence_comparison.tex, summary_metrics.tex, ablation_results.tex, environment_config.tex
- Data: convergence_main.csv, equilibrium_recovery_dotplot.csv, distance_to_equilibrium.csv, + 6 others
- Figures: convergence_main.pdf, equilibrium_recovery_dotplot.pdf, distance_to_equilibrium.pdf, + 6 others

### Verified numbers

| q | decision doc | regenerated table | match? |
|---|-------------|-------------------|--------|
| 35 | 4.3% | 4.30% | yes |
| 45 | 2.1% | 2.08% | yes |
| 55 | 2.4% | 2.36% | yes |

### Known items NOT done this round

- `docs/metric_diagnosis.md` — intentionally preserves old-metric context
- `docs/round2_metric_decision.md` — intentionally preserves old-vs-new comparison

### Commit suggestion

`refactor: switch paper reporting to Metric B (policy_mean_effort[-1])`

## Conc-ramp warmup port: three_players (in progress)

Ported concentration ramp logic from `run_two_players.py:889-910` to `run_three_players.py`.
Added CLI flag `--override-conc-ramp-warmup`. Default stays at 20 (as in theory_align_v2 defaults block).

### Agent attribute mapping (two_players vs three_players)

| attribute | PPOTwoPlayersBandit | PPOThreePlayersBandit | match? |
|-----------|--------------------|-----------------------|--------|
| `agent.net.conc_min` | yes (ActorCriticMeanConc:80) | yes (ActorCriticMeanConc:100) | identical |
| `agent.net.conc_scale` | yes (:81) | yes (:101) | identical |
| `agent.net.conc_max` | yes (:82) | yes (:102) | identical |
| `agent.opponent_policy.conc_min` | yes (deepcopy of net) | yes (deepcopy of net) | identical |
| `agent.cfg.theory_align_v2_var_coef` | yes (PPOConfig:145-146) | yes (PPOConfig:167-168) | identical |

All attributes match 1:1. The ramp code is an exact copy.

### Not yet ported (pending results from 3p)

- `run_different_cost.py`: uses TWO agents (agent1, agent2) — ramp needs to apply to both. Deferred.
- `run_different_ability.py`: single shared agent, similar to two_players. Deferred.

## Next steps
1. Regenerate paper artifacts with Round 3/4 data: `python -m paper.generator make_all`
2. Verify regenerated figures/tables reflect new numbers (esp. Het. Ability q=55 at 0.64%, up from "in progress")
3. Address `round3_round4_report.md` §8 open questions:
   - Whether da ~1.6% at q=35 is a shared-policy architectural floor (separate-policy variant TBD)
   - Whether 3P q=55 std=0.35% holds under broader conditions
4. Build two-stage runner (separate task)
