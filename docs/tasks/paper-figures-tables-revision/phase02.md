# Phase 02: Figure 1 — Convergence of TEL-PPO under Increasing Noise

## Objective

Restyle Figure 1 per revision notes: clearer labels, lighter shading, convergence vertical line, reconsider error box. Columns are now q=35/40/55 (q=25 removed).

## Changes

1. Columns: q=35, q=40, q=55 (remove q=25 column)
2. Labels `(w_H, w_L) = (6.5, 3)` and `(8, 4)`: add `k` value; make clearer (place on top or bold)
3. Lighten shading (use global `SHADE_ALPHA`)
4. Add a vertical line at first update satisfying convergence criteria
5. Consider removing error box (check with user after generating draft)
6. Clarify what the specific metric on y-axis refers to (add axis label or annotation)

## Files to modify

- `paper/generator/plots.py` — convergence figure function

## Verification

- Generate figure, visually inspect label clarity, shading opacity, vertical line placement
- `python -m paper.generator make_all` succeeds
