# Task: Diagnose All Non-Two-Player Experiments

## Goal
Run comprehensive diagnostics on three_players, different_cost, and different_ability experiments. Identify participation constraint violations, convergence issues, data integrity problems, and theory inconsistencies.

## Scope
- **In scope**: Diagnostic analysis and report only. No code fixes, no retraining.
- **Out of scope**: Fixing any issues found (separate tasks). Two_players experiment (already diagnosed).

## Key Files
- `results/{three_players,different_cost,different_ability}/convergence/` — data
- `utils/theory.py` — equilibrium formulas
- `utils/prob.py` — win probability functions
- `tools/diagnose_all.py` — diagnostic script (to create)

## Constraints
- Do not modify any result files
- Independent P(win) implementation for cross-validation against prob.py
- Three-way cross-check: theory.py vs JSON theoretical_effort vs formula
- Report which specific player violates participation constraint (not just "violated")
