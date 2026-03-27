# Project state

Last updated: 2026-03-27

## Current status
- Paper figures/tables revision complete: 11 figures + 5 tables generated via `python -m paper.generator make_all`
- q=35 experiments complete for all 4 scenarios (two_players, three_players, different_cost, different_ability)
- 3p PPO convergence gap (~5 units) accepted as paper limitation — discussed as NC in final_summary
- All tmux sessions terminated

## What was done (2026-03-27, session 2)
1. Generated missing gradient baseline for q=35 wh8_wl4 (w_H=8, w_L=4, e*=71.43, gap=0.17)
2. Regenerated baseline gradient for q=35 (w_H=6.5, w_L=3.0, e*=62.5, gap=0.15) — required 50k steps
3. Regenerated convergence_main figure — all 6 panels now populated
4. Closed q35-all-experiments task (3p limitation accepted)
5. Created 3p-algorithm-improvement task with full problem diagnosis and 6 proposals
6. Killed running tmux session (3p_5000upd_v2)

## What was done (2026-03-27, session 1)

### Paper figures & tables revision (phases 03–11)
1. **Phase 03**: Effort drift — SHADE_ALPHA, thicker threshold, standardized legend
2. **Phase 04**: KL dynamics — SHADE_ALPHA, "Reference threshold" label
3. **Phase 05**: Distance to equilibrium — new title, y-axis label, q=35 color
4. **Phase 06**: Beta snapshots/evolution — auto-select best seed (42), lighter shading
5. **Phase 07**: Exploitability 6a — renamed labels; new Figure 6b for q=25
6. **Phase 08**: Ablation — ABLATION_LABELS/LINEWIDTHS, prominent Theory line, unified y-axis
7. **Phase 09**: Dotplot — alternating backgrounds, smaller per-seed dots, renamed labels
8. **Phase 10–11**: Tables — q=25 excluded, q=35 included, data loader fix for make_all

### q=35 3p diagnostic (phase 03 — closed)
- Tested 11 variants (entropy, adv norm, network arch, optimizer reset, 5000 updates)
- All failed: gap remains ~5 units from equilibrium
- Root cause: 3-player rank-order reward produces structurally weaker gradient signal than 2-player
- Gradient descent converges perfectly (gap=0.12), confirming theory is correct
- Decision: accept as limitation

## Data inventory
- two_players: baseline + wh8_wl4 + ablations + sweeps (~115 convergence JSONs)
- three_players: baseline (5 seeds) + gradient + 11 diagnostic variants for q=35
- different_cost: baseline (5 seeds per q) + gradient
- different_ability: baseline (5 seeds per q) + gradient

## Known tech debt
- Runner files (run/run_*.py) have ~60% code duplication — extract shared base (task: `docs/tasks/runner-refactor/`)
- No tests exist for any module
- No CI/CD pipeline
- summary.csv in results/two_players/ has a parsing issue (line 14 has 46 fields, expected 39)
- Vestigial opponent lag code in agents/ppo_two_players_clean.py (deepcopy, sync logic, act_opponent — all unused)
- q=35 baseline gradient solver needs 50k steps (MC gradient noisy for low-q; existing q=40/55 files were generated with different params)

## Task status

| Task | Status | Notes |
|------|--------|-------|
| paper-figures-tables-revision | complete | 11 phases done, all figures/tables regenerated, q=35 wh8_wl4 panel filled |
| q35-all-experiments | closed | 3p limitation accepted |
| perfect-exploitability-figure | closed | decisions resolved, work transferred |
| diagnose-all-experiments | complete | — |
| runner-refactor | in-progress (phase01) | low priority, duplication audit pending |
| 3p-algorithm-improvement | not-started | proposals documented, pending implementation |

## Next steps
- 3p algorithm improvement: implement pairwise advantage decomposition (Proposal 1 in `docs/tasks/3p-algorithm-improvement/PROPOSALS.md`)
- Runner refactor phase 01: audit duplication across the 4 runners
- Add basic tests for theory.py, prob.py, paper generator
- Consider git filter-repo to remove large files from history (~85 MB .git)
