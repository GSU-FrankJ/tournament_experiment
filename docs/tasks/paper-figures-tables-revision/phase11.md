# Phase 11: Tables 3–4 — Quantitative Summary Tables

## Objective

Update quantitative summary tables: replace q=25 rows with q=35 and ensure all metrics are current.

## Changes

1. Replace q=25 rows with q=35 for all scenarios (two-player, three-player, het. cost, het. ability)
2. q=25 does NOT appear in Tables 3–4 (discussed separately in text)
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
