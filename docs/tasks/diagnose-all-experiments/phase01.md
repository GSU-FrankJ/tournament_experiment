# Phase 01: Diagnostic Script and Report

## Objective
Create `tools/diagnose_all.py` that diagnoses three_players, different_cost, and different_ability experiments.

## Checks
1. **Participation constraint** — per experiment type, per q, per player where asymmetric
2. **Convergence status** — effort gap, final exploitability, stop reason per (q, seed)
3. **Data integrity** — truncated runs, missing fields, NaN values
4. **Theory consistency** — three-way cross-check: theory.py vs JSON vs formula

## Design
- Independent P(win) implementation (not relying on prob.py) for cross-validation
- Modular: load_results(), check_participation(), check_convergence(), check_integrity(), check_theory()
- Unified DiagnosticResult output per check
- Markdown report saved to task folder

## Verification
Run script, review report for each experiment type.
