# Project state

Last updated: 2026-03-13

## Current status
- Phase 1 cleanup complete (branch cleanup/prune-results merged)
- 239 convergence JSONs retained (best runs + gradient + ablation + hyperparam + weight variants)
- Paper generator verified: --best-only --dry-run passes

## Known tech debt
- Runner files (run/run_*.py) have ~60% code duplication — extract shared base
- No tests exist for any module
- No CI/CD pipeline
- summary.csv in results/two_players/ has a parsing issue (line 14 has 46 fields, expected 39)

## Next steps
- Phase 2: Code refactoring (extract shared runner, reduce duplication)
- Add basic tests for theory.py, prob.py, paper generator
- Consider git filter-repo to remove large files from history (~85 MB .git)
