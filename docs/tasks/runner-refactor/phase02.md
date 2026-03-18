# Phase 02: Remove vs_opponent Rollout Mode

## Objective

Completely remove the `vs_opponent` rollout mode from the codebase. This mode is no longer used in any experiment. Only `selfplay` should remain. All console output, CLI flags, docs, and tool scripts referencing `vs_opponent` must be cleaned up so nothing confuses future users.

## Inventory (87+ references)

### Code (must change)

| File | What to do |
|------|-----------|
| `run/run_two_players.py` | Remove `rollout_mode` parameter and all branching on it; remove `eval_vs_opponent` flag and related eval columns; remove `--rollout-mode` CLI arg; hardcode selfplay behavior; clean up console prints |
| `run/run_three_players.py` | Same: remove `rollout_mode` parameter, branching, lag schedule for vs_opponent, `--rollout-mode` CLI arg |
| `utils/logger.py` | Remove `eval_vs_opponent_*` columns from CSV header and row dict |
| `utils/rollout_stats.py` | Remove docstring reference to vs_opponent |

### Tools (must change)

| File | What to do |
|------|-----------|
| `tools/verify_rollout_modes.py` | Remove vs_opponent test cases; rename/simplify if only selfplay remains |
| `tools/audit_rollout_modes.py` | Remove `audit_risk_point_2_vs_opponent()` and all vs_opponent branches |

### Documentation (must change)

| File | What to do |
|------|-----------|
| `docs/technical/rollout_modes.md` | Remove vs_opponent section; simplify to selfplay-only doc |
| `docs/technical/POLICY_SCALE_DIAGNOSTICS.md` | Remove vs_opponent references |
| `docs/technical/audit_theory_align_v2.md` | Remove vs_opponent mention |
| `docs/technical/README.md` | Update table description |
| `docs/guides/ppo_defaults.md` | Remove vs_opponent examples and table row |
| `docs/README.md` | Update reference line |
| `run/README.md` | Remove `--rollout-mode` from table and examples |
| `AGENTS.md` | Remove mention of "self-play vs opponent-lag behavior" |

### Cursor skills/plans (must change)

| File | What to do |
|------|-----------|
| `.cursor/skills/running-experiments/SKILL.md` | Remove `--rollout-mode` from table and examples |
| `.cursor/skills/running-experiments/examples.md` | Remove vs_opponent example |
| `.cursor/skills/running-experiments/scripts/quick_experiment.py` | Remove `--rollout-mode` arg |
| `.cursor/plans/three-player_experiment_*.plan.md` | Remove vs_opponent references |
| `.cursor/plans/exploit_ablation_experiment_*.plan.md` | Remove vs_opponent tool reference |

### Not touched

- `run/run_different_cost.py` — no vs_opponent references
- `run/run_different_ability.py` — no vs_opponent references
- `agents/`, `envs/`, `paper/`, `results/` — no references

## Steps

1. **Runners first:** Edit `run/run_two_players.py` and `run/run_three_players.py`:
   - Remove `rollout_mode` parameter from `train_ppo_*` functions; hardcode selfplay logic
   - Remove `eval_vs_opponent` parameter and related eval code
   - Remove `--rollout-mode` and `--eval-vs-opponent` CLI args
   - Remove all `if rollout_mode == "vs_opponent"` branches
   - Remove lag schedule code that only served vs_opponent
   - Clean up console prints that mention vs_opponent
2. **Utils:** Remove `eval_vs_opponent_*` columns from `utils/logger.py`; fix docstring in `utils/rollout_stats.py`
3. **Tools:** Clean up `tools/verify_rollout_modes.py` and `tools/audit_rollout_modes.py`
4. **Docs:** Update all markdown files listed above
5. **Cursor config:** Update skill files and plans
6. **Verify:** `grep -r "vs_opponent" .` returns zero hits (excluding git history)

## Verification

- `grep -rn "vs_opponent" --include="*.py" --include="*.md" .` returns no matches
- `python run/run_two_players.py --method ppo --q 40 --episodes 4096 --seed 42` runs without errors
- `python run/run_three_players.py --method ppo --q 40 --episodes 4096 --seed 42` runs without errors
- No `--rollout-mode` flag in `--help` output
- Console output during training mentions only "selfplay" (or no mode at all)
