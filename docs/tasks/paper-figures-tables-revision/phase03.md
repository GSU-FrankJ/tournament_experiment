# Phase 03: Figure 2 — Effort Drift Across Noise Levels

## Objective

Restyle Figure 2 per revision notes: lighter shading, thicker threshold line, standardized legend.

## Changes

1. Lighten shading (use global `SHADE_ALPHA`)
2. Make threshold line thicker
3. Standardize legend entries:
   - Median drift
   - 10–90% interval
   - Drift threshold (2.0)
   - Detected convergence step

## Files to modify

- `paper/generator/plots.py` — effort drift figure function

## Verification

- Generate figure, visually inspect shading, threshold line weight, legend entries
