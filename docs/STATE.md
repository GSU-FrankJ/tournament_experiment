# Project state

Last updated: 2026-03-18

## Current status
- Data pruning pass 2 complete: 161 convergence JSONs retained (figure-essential only)
- All 10 paper figures verified reproducible with `python -m paper.generator make_all`
- Figures match e3de29a (2026-03-09) versions; generation is deterministic across runs
- Task pipeline established: `docs/tasks/` for durable, git-tracked multi-phase planning
- STATE.md moved from repo root into `docs/` (referenced by `.claude/CLAUDE.md`)
- vs_opponent rollout mode fully removed from codebase (runner-refactor phase 02)

## What was done (2026-03-18, session 2)
1. Runner-refactor phase 02: removed vs_opponent rollout mode entirely
   - Runners: removed rollout_mode param, vs_opponent branches, eval_vs_opponent, --rollout-mode CLI arg from run_two_players.py, run_three_players.py, run_three_players_largeb.py
   - Utils: removed eval_vs_opponent_* columns from logger.py, updated rollout_stats.py docstring
   - Tools: cleaned verify_rollout_modes.py, audit_rollout_modes.py, sweep_mechanism_ablation.py, sweep_exploit_ablation.py
   - Docs: updated 9 markdown files across docs/technical/, docs/guides/, run/, tools/, AGENTS.md
   - Cursor: updated skills, examples, quick_experiment.py, 2 plan files
   - Verified: zero grep hits for vs_opponent in code/docs, --help confirms no --rollout-mode flag

## What was done (2026-03-18, session 1)
1. Moved `STATE.md` → `docs/STATE.md` as project-level state tracker
2. Created `docs/tasks/README.md` with pipeline conventions and templates
3. Seeded `docs/tasks/runner-refactor/` as first task (CLAUDE.md, STATE.md, phase01.md)
4. Updated `.claude/CLAUDE.md` with Task Pipeline section and corrected STATE.md path
5. Updated `docs/README.md` tree to include `tasks/` and `STATE.md`
6. Added `.gitignore` exception for `docs/tasks/` (was blocked by `tasks/` rule)

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
- Runner files (run/run_*.py) have ~60% code duplication — extract shared base (task: `docs/tasks/runner-refactor/`)
- No tests exist for any module
- No CI/CD pipeline
- summary.csv in results/two_players/ has a parsing issue (line 14 has 46 fields, expected 39)
- baseline_v2 runs had a bug (q parameter not effective) — all deleted, documented here for reference

## Next steps
- Runner refactor phase 01: audit duplication across the 4 runners (see `docs/tasks/runner-refactor/phase01.md`)
  - Phase 02 (vs_opponent removal) done; phase 01 (duplication audit) still pending
- Add basic tests for theory.py, prob.py, paper generator
- Consider git filter-repo to remove large files from history (~85 MB .git)
