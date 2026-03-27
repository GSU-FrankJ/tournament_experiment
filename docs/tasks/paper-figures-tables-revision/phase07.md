# Phase 07: Figures 6a/6b — Exploitability and Approximate Equilibrium Verification

## Objective

Rename labels in Figure 6a; create separate Figure 6b panel for the excluded low-noise case.

## Changes — Figure 6a (q=35, 40, 55)

1. Panels: q=35, q=40, q=55 (replace q=25)
2. Rename labels:
   - "Cheap gate" → **Stability screening passed**
   - "Nash convergence" → **Approx. Nash verified**
   - "Threshold" → **Tolerance threshold**

## Changes — Figure 6b (q=25 only)

1. Create a separate single-panel figure showing exploitability in the excluded low-noise case (q=25)
2. Match styling with Figure 6a

## Files to modify

- `paper/generator/plots.py` — exploitability figure function(s)

## Verification

- Generate both 6a and 6b, verify renamed labels, visual consistency between panels
