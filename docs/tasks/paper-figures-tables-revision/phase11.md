# Phase 11: Tables 3–4 — Quantitative Summary Tables

## Objective

Update quantitative summary tables to include q=35 data and ensure all metrics are current.

## Changes

1. Add q=35 rows for all scenarios (two-player, three-player, het. cost, het. ability)
2. Ensure q=25 rows remain for analysis (may be in a separate appendix table or kept inline)
3. Verify all metric values against latest convergence JSON files:
   - Mean effort +/- std
   - |e_bar - e*|
   - Relative error
   - Exploitability
   - Structure gap
   - Steps to convergence
4. Generate both CSV and LaTeX output

## Files to modify

- `paper/generator/tables.py`
- `paper/generator/metrics.py` (if computation changes needed)

## Verification

- Tables include q=35 data
- All values match `results/*/convergence/*.json`
- `python -m paper.generator make_all` generates updated tables
