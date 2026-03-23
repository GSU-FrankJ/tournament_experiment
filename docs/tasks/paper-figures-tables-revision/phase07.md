# Phase 07: Figures 6a/6b — Exploitability and Approximate Equilibrium Verification

## Objective

Rename labels in Figure 6a; create separate Figure 6b panel for the excluded low-noise case.

## Changes — Figure 6a

1. Rename labels:
   - "Cheap gate" → **Stability screening passed**
   - "Nash convergence" → **Approx. Nash verified**
   - "Threshold" → **Tolerance threshold**

## Changes — Figure 6b

1. Create a separate panel showing exploitability in the excluded low-noise scenario (q=25 or q=35 depending on which is excluded)
2. Match styling with Figure 6a

## Files to modify

- `paper/generator/plots.py` — exploitability figure function(s)

## Verification

- Generate both 6a and 6b, verify renamed labels, visual consistency between panels
