# Phase 04: Figure 3 — KL Divergence Across Noise Levels

## Objective

Restyle Figure 3 per revision notes: lighter shading, standardized legend.

## Changes

1. Lighten shading (use global `SHADE_ALPHA`)
2. Standardize legend entries:
   - Median KL
   - 10–90% interval
   - Reference threshold

## Files to modify

- `paper/generator/plots.py` — KL divergence figure function

## Verification

- Generate figure, visually inspect shading and legend entries
