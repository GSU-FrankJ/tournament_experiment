# Phase 09: Figure 8 — Equilibrium Recovery Across Scenarios and Noise Levels

## Objective

Restyle dotplot: clarify grouping, rename labels, adjust dot styling.

## Changes

1. Rename figure: **Equilibrium Recovery Across Scenarios and Noise Levels**
2. Clarify x-axis grouping: add grey vertical separator or increase spacing; or use two background regions for symmetric vs. heterogeneous
3. Strengthen Theory line differentiation (grey-black?)
4. Rename labels:
   - "Per-seed" → **Per-seed estimate**
   - "Seed mean" → **Across-seed mean**
5. Per-seed estimate: use lighter, smaller dots
6. Move scenario labels closer to q values

## Files to modify

- `paper/generator/plots.py` — equilibrium recovery dotplot function

## Verification

- Generate figure, verify grouping clarity, label names, dot sizes, Theory line visibility
