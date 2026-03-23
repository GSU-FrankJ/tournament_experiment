# Phase 05: Figure 4 — Convergence Error to the Analytical Equilibrium

## Objective

Restyle and rename Figure 4 per revision notes.

## Changes

1. Rename figure: **Convergence Error to the Analytical Equilibrium**
2. y-axis label: Equilibrium error |·|
3. Add target error threshold (epsilon = 0.5) as horizontal line
4. Vertical line at detected convergence step (make thinner than current)
5. Lighten shading (use global `SHADE_ALPHA`)
6. Standardize legend entries:
   - q = 35
   - q = 40
   - q = 55
   - Target error threshold
   - Detected convergence step

## Files to modify

- `paper/generator/plots.py` — convergence error figure function

## Verification

- Generate figure, verify title, y-axis label, threshold line, convergence vertical line, legend
