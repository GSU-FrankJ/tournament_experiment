# Project state

Last updated: 2026-04-10

## Current status
- **Parameter overhaul**: All configs updated to match `docs/experiment_config_040726.md`
- **Concentration fix (major)**: `--override-conc-ramp-warmup 200` resolves q=45/55 convergence
  - Root cause: theory_align_v2 concentration ramp froze policy in ~20 updates, before effort reached e*
  - Fix: extend warmup from 20→200, giving policy time to descend to e* before concentration rises
  - Results (Round 2, 5 seeds): q=35 gap 0.3%, q=45 gap 0.1%, q=55 gap 0.1%
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
- Single-column, log-scale |e-e*| with terminal gaps: q=35: 1.6, q=45: 4.3, q=55: 7.6
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

## Next steps
1. Run gradient baselines for all experiment types with new parameters
2. Run PPO experiments (5 seeds each) for all q values
3. Two-player Set 2 via CLI flags: `--k 0.0006 --w_h 8 --w_l 4`
4. Regenerate paper artifacts after results are collected
5. Build two-stage runner (separate task)
