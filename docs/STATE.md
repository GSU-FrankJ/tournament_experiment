# Project state

Last updated: 2026-07-07

## Claim-A κ-continuation 5-seed run + Gate C (2026-07-07)

- Pilot #2 (seed 42) completed 2026-07-04: clean exit, all 6 ladder stages to
  κ=400, 0 forced advances, final 24.30 — borderline vs Gate C, owner chose the
  5-seed run. **Gate C scored on 5 FRESH seeds 43–47** (pilot 42 excluded to avoid
  pilot-selection bias; owner-confirmed); tmux `c3_s43`–`c3_s47`, GPUs 0–4, params
  byte-identical to pilot #2.
- **All 5 seeds clean** (exploitability stop, forced_advances=0, exploit
  0.004–0.011). **Gate C verdict: BORDERLINE — no branch fired.** Final-snapshot
  metric: cross-seed mean 24.034 (|err| 3.86%, KILL line is 4%), std 0.688
  (success ≤0.5, KILL >1.0).
- **Decomposition is decisive though**: snapshot spread ≈ within-run κ=400
  diffusion sampled at 1 update ("done" stops on its first update). Time-averaged
  (κ=400 stage, last 30 upd): cross-seed mean **24.29, std 0.146** (SE 0.065) —
  variance solved (~11× tighter than c2's 1.67), but mean misses the 24.5 success
  line by ~3 SE. **Undershoot is systematic bias, not noise; strong Claim A is
  dead in this parameterization.** All 6 runs land 24.1–24.4, ~0.4 below
  μ*(400)=24.7.
- Recommendation: adopt Claim B final form (PPO → smoothed equilibrium μ*(κ);
  continuation tracks it reproducibly; MC-BR bridges the last 0.7). No more GPU
  on Claim A without new variance-reduction evidence — this is now four
  concordant negatives (r5, c2, design analysis, c3 5-seed).
- Details + per-seed table: `docs/tasks/claim-a-nonlocking-continuation/STATE.md`.
  Data: `results/three_players/convergence/ppo_3p_q35.0_seed{42..47}_c3_cont_convergence.json`.

## Current status
- **Parameter overhaul**: All configs updated to match `docs/experiment_config_040726.md`
- **Concentration fix (major)**: `--override-conc-ramp-warmup 200` resolves q=45/55 convergence
  - Root cause: theory_align_v2 concentration ramp froze policy in ~20 updates, before effort reached e*
  - Fix: extend warmup from 20→200, giving policy time to descend to e* before concentration rises
  - Results (Round 2, 5 seeds, Metric B): q=35 rel 4.3%, q=45 rel 2.1%, q=55 rel 2.4%
  - Old entropy_end_0.002 files archived to `results/two_players/convergence/_archive_pre_warmup_fix/`
- **Figure pipeline**: All 12 figures regenerated with warmup=200 results
- **Pending**: q=35 seeds 44/45, q=45 seed 44 retrying (OOM from parallel run)
- Two-stage runner deferred to separate task

## Figure pipeline (2026-04-09)

### Step 0: Style gate — PASSED
- Updated `paper/generator/config.py`: Wong palette, IEEE/ACM font sizes (8/9/10pt),
  TrueType embedding (pdf.fonttype=42), horizontal-only grid
- Gate checks: no Type 3 fonts, PDF width 6.70" (within 6.75±0.05"), all Wong colors verified
- Output: `paper/generator/output/style_test/style_test.{pdf,png}`

### F1: Equilibrium Recovery (hero) — COMPLETE
- Patched `run_registry.py`: `BASELINE_ALIASES = {"entropy_end_0.002"}` (line ~283)
  so q=45/55 Set 1 runs classified as baseline
- All 5 experiment groups × all q values plotted (14 conditions total)
- Bootstrap percentile 95% CI (n=5, n_resamples=2000)
- All seed counts = 5, all CI widths < 5 effort units
- q=55 2P Set 1 gray band + "See Fig. 2" annotation
- Output: `paper/generator/output/figures/F1_equilibrium_recovery.{pdf,png}`

### F2: EU Landscape & Gradient Signal — COMPLETE
- 3 panels: (a) EU landscape, (b) symmetric gradient, (c) stall-point bars at e=36
- Theory values verified: e*, gradient at e*=0, gradient at 36 matches plan
- Output: `paper/generator/output/figures/F2_eu_landscape.{pdf,png}`

### F3: Training Diagnostics — COMPLETE
- 2×3 grid: KL (top) + exploitability (bottom) × q=35/45/55
- Output: `paper/generator/output/figures/F3_training_diagnostics.{pdf,png}`

### F4: Distance to Equilibrium — COMPLETE
- Single-column, log-scale |e-e*| with terminal gaps (Metric B): q=35: 1.96, q=45: 0.74, q=55: 0.68
- Output: `paper/generator/output/figures/F4_distance_to_equilibrium.{pdf,png}`

### F5: Ablation — BLOCKED (no data)

### Appendix (8 figures) — ALL COMPLETE
- FA_2p, FA_set2, FA_gvp, FA_3p, FA_het, FA_beta, FA_snap, FA_drift
- All in `paper/generator/output/figures/`

## Figure Manifest (Phase 3 complete)

| fig_id | filename | width | fonts | status |
|--------|----------|-------|-------|--------|
| F1 | F1_equilibrium_recovery.pdf | 6.70" | TT | done |
| F2 | F2_eu_landscape.pdf | 6.77" | TT | done |
| F3 | F3_training_diagnostics.pdf | 6.69" | TT | done |
| F4 | F4_distance_to_equilibrium.pdf | 3.20" | TT | done |
| FA_2p | FA_2p_convergence.pdf | 6.69" | TT | done |
| FA_3p | FA_3p_convergence.pdf | 3.20" | TT | done |
| FA_beta | FA_beta_evolution.pdf | 6.67" | TT | done |
| FA_drift | FA_drift_post_convergence.pdf | 6.69" | TT | done |
| FA_gvp | FA_gvp_comparison.pdf | 6.69" | TT | done |
| FA_het | FA_het_convergence.pdf | 6.69" | TT | done |
| FA_set2 | FA_set2_convergence.pdf | 6.69" | TT | done |
| FA_snap | FA_snap_beta_pdf.pdf | 6.70" | TT | done |

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
| two-stage-runner | deferred | Config and env exist, runner needs to be built |
| paper-figures-tables-revision | stale | Needs re-run after new experiment data |
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
1. Run gradient baselines for all experiment types with new parameters
2. Run PPO experiments (5 seeds each) for all q values
3. Two-player Set 2 via CLI flags: `--k 0.0006 --w_h 8 --w_l 4`
4. Regenerate paper artifacts after results are collected
5. Build two-stage runner (separate task)
