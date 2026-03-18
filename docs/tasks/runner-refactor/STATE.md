# Runner Refactor

Status: in-progress
Current phase: phase01

## What's done
- Task folder created with scope and constraints documented
- Phase 02: Removed vs_opponent rollout mode from entire codebase
  - Runners: removed rollout_mode param, vs_opponent branches, eval_vs_opponent, --rollout-mode CLI arg from run_two_players.py and run_three_players.py (+ largeb variant)
  - Utils: removed eval_vs_opponent_* columns from logger.py, updated rollout_stats.py docstring
  - Tools: cleaned verify_rollout_modes.py, audit_rollout_modes.py, sweep_mechanism_ablation.py, sweep_exploit_ablation.py
  - Docs: updated 9 markdown files (rollout_modes.md, POLICY_SCALE_DIAGNOSTICS.md, audit_theory_align_v2.md, ppo_defaults.md, plot_convergence.md, asymmetric_init.md, run/README.md, AGENTS.md, tools/README.md)
  - Cursor: updated skills, examples, quick_experiment.py, 2 plan files

## What's next
- Phase 01: Audit duplication across the 4 runners, identify shared logic

## Blockers
- (none)
