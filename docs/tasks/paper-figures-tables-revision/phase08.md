# Phase 08: Figure 7 — Component Ablation of TEL-PPO

## Objective

Major restyle of ablation figure: unified y-axis, renamed labels, line weight differentiation, x-axis formatting, optional summary panel. Panels are q=35/40/55 (q=25 removed).

## Changes

1. Unify y-axis across panels
2. Make Theory line more prominent
3. Rename labels:
   - "Baseline" → **TEL-PPO**
   - "No cheap gate" → **No stability screening**
   - "No exploitability" → **No exploitability verification**
4. TEL-PPO line thicker; ablation lines slightly thinner
5. x-axis: Training steps (x10^6)
6. q = 35, 40, 55 (remove `.0` suffix)
7. Consider adding a small summary panel with:
   - Terminal absolute error
   - Final exploitability
   - NC rate
   - Time-to-pass verification

## Files to modify

- `paper/generator/plots.py` — ablation figure function
- `paper/generator/config.py` — ablation label/color mappings (if not done in phase01)

## Verification

- Generate figure, verify unified y-axis, label names, line weights, x-axis format
- If summary panel added: verify metric values match convergence data
