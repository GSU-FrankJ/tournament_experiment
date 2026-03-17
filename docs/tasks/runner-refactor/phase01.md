# Phase 01: Audit Duplication

## Objective

Map the shared vs. unique logic across all 4 runners to determine what can be extracted into a common base.

## Steps

1. Read all 4 runner files and catalogue their top-level structure (functions, classes, CLI args)
2. Diff shared sections: argument parsing, training loop, logging, checkpoint saving, convergence JSON output
3. Identify experiment-specific divergences (env creation, reward computation, evaluation)
4. Produce a summary table: function/block → shared | unique per runner
5. Propose an extraction plan (what goes into `base_runner.py`, what stays)

## Files to read

- `run/run_two_players.py`
- `run/run_three_players.py`
- `run/run_different_cost.py`
- `run/run_different_ability.py`

## Verification

- Summary table written to this task's STATE.md or a separate `audit.md`
- Extraction plan reviewed and approved before moving to phase 02
