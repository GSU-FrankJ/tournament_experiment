# Phase 06: Figure 5 — Evolution of the Learned Policy Distribution

## Objective

Update Figure 5 to use q=40, (w_H, w_L) = (6.5, 3). Evaluate whether a seed with smaller error exists; if not, consider removing.

## Steps

1. Load all q=40 convergence data for baseline seeds
2. Identify the seed with smallest final effort error |e - e*|
3. If a seed has acceptably small error, generate beta distribution snapshots from that seed
4. If no seed has small error, flag for removal and confirm with user

## Files to modify

- `paper/generator/plots.py` — beta evolution figure function
- `paper/generator/extract.py` — if seed selection logic needed

## Verification

- Generate figure (or produce report recommending removal)
- If generated: verify q=40, (w_H, w_L) = (6.5, 3), visually inspect distribution quality
