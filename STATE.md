# Project state

Last updated: 2026-03-17

## Current status
- Data pruning pass 2 complete: 161 convergence JSONs retained (figure-essential only)
- All 10 paper figures verified reproducible with `python -m paper.generator make_all` (no --best-only)
- Figures match e3de29a (2026-03-09) versions; generation is deterministic across runs
- 13 post-e3de29a files archived to `results/archive/` (largeb + new diff_ability sweeps)
- 187 unused files deleted (baseline_v2, diff_cost/ability eps/pat sweeps, k5e4_wh8_wl3, q25_seed68)
  - All recoverable from git history at commit e3de29a

## What was done (2026-03-17)
1. Identified which convergence JSONs each of the 10 figures actually uses
2. Restored 116 files from e3de29a that were missing after prior prune (needed for multi-seed aggregation)
3. Restored all data to exact e3de29a versions (byte-identical)
4. Archived 13 post-e3de29a files to `results/archive/`
5. Deleted 187 unused files (210 total including metadata) — verified figures unchanged

## Data inventory (161 convergence JSONs)
- two_players: baseline (19), wh8_wl4 (9+3 gradient), no_cheap_gate (9), no_exploitability (9), eps_*/pat_* sweeps (63), gradient baseline (3)
- three_players: baseline TEL-PPO (11) + gradient (3)
- different_cost: baseline TEL-PPO (15) + gradient (1)
- different_ability: baseline TEL-PPO (15) + gradient (1)

## Known tech debt
- Runner files (run/run_*.py) have ~60% code duplication — extract shared base
- No tests exist for any module
- No CI/CD pipeline
- summary.csv in results/two_players/ has a parsing issue (line 14 has 46 fields, expected 39)
- baseline_v2 runs had a bug (q parameter not effective) — all deleted, documented here for reference

## Next steps
- Phase 2: Code refactoring (extract shared runner, reduce duplication)
- Add basic tests for theory.py, prob.py, paper generator
- Consider git filter-repo to remove large files from history (~85 MB .git)
