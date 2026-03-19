# Phase 03: Remove --best-only flag

## Objective

Remove the `--best-only` filtering entirely. Always load all runs without seed selection.

## Changes

### Code (`paper/generator/__main__.py`)
1. Remove `--best-only` argparse argument (lines 347-351)
2. Remove `select_best_runs` import (line 39)
3. Remove `load_multiple_runs` import in `cmd_make_all`, `cmd_plot`, `cmd_table`
4. Simplify `cmd_make_all` (lines 149-167): remove if/else, keep only `load_all_convergence_data` path
5. Simplify `cmd_plot` (lines 214-229): same
6. Simplify `cmd_table` (lines 271-286): same

### Code (`paper/generator/run_registry.py`)
7. Remove `select_best_runs()` function (lines 394-433)

### Code (`paper/generator/extract.py`)
8. Remove `get_final_effort_error_from_json()` if no other callers exist
9. Remove `load_multiple_runs()` if no other callers exist

### Docs
10. Update `.claude/CLAUDE.md`: remove `--best-only` from example commands
11. Update `docs/guides/results-folder-guide.md`, `docs/STATE.md`, task docs referencing `--best-only`

## Verification

- `python -m paper.generator make_all` works without `--best-only` flag
- `python -m paper.generator --help` no longer shows `--best-only`
- `grep -r "best.only" .` returns no hits in code (docs ok if cleaned)
