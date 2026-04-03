# Runner Refactor

Status: deferred
Current phase: phase01 (not started)

## What's done
- Task folder created with scope and constraints documented
- Phase 02: Removed vs_opponent rollout mode from entire codebase
  - Runners: removed rollout_mode param, vs_opponent branches, eval_vs_opponent, --rollout-mode CLI arg from run_two_players.py and run_three_players.py (+ largeb variant)
  - Utils: removed eval_vs_opponent_* columns from logger.py, updated rollout_stats.py docstring
  - Tools: cleaned verify_rollout_modes.py, audit_rollout_modes.py, sweep_mechanism_ablation.py, sweep_exploit_ablation.py
  - Docs: updated 9 markdown files (rollout_modes.md, POLICY_SCALE_DIAGNOSTICS.md, audit_theory_align_v2.md, ppo_defaults.md, plot_convergence.md, asymmetric_init.md, run/README.md, AGENTS.md, tools/README.md)
  - Cursor: updated skills, examples, quick_experiment.py, 2 plan files
- Phase 03: Removed --best-only flag from paper generator
  - Code: removed CLI flag, select_best_runs(), get_final_effort_error_from_json(), if/else branches in 3 command handlers
  - Docs: updated CLAUDE.md, docs/STATE.md, results-folder-guide.md, 3 task docs
- Fix: Corrected misleading "lagged opponent" references across docs and config
  - Phase 02 removed the vs_opponent rollout mode, but a separate concept — opponent_sync_interval / lagged opponent network — was left intact and misleading
  - `.claude/CLAUDE.md`: "self-play, lagged opponent" → "pure self-play; opponent lag exists in code but never used for action selection"
  - `config/one_stage_two_players.py`: set opponent_sync_interval to 0, marked all opponent lag settings as VESTIGIAL with explanatory comment
  - `README.md`: removed two-phase "early: lagged opponent / late: symmetric" description, replaced with accurate single-line "pure self-play" description
  - `run/run_two_players.py`: marked lag_prob schedule comment as vestigial (computed/logged but never affects action selection)
  - Not touched: `docs/technical/rollout_modes.md` (already accurate), `docs/tasks/perfect-exploitability-figure/` (already notes lag unused)
  - Remaining dead code: `agents/ppo_two_players_clean.py` still creates `opponent_policy = deepcopy(net)` at init; with sync_interval now 0 the sync branch no longer fires, but the initial deepcopy still wastes memory. Candidate for future cleanup.

## What's next
- Phase 01: Audit duplication across the 4 runners, identify shared logic

## Known issues
- Vestigial opponent lag code in `agents/ppo_two_players_clean.py`: deepcopy of network at init, sync logic in `update()`, `act_opponent()` method — all unused in selfplay mode. Low priority cleanup candidate.

## Blockers
- (none)
