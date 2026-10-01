# dReach reach-mask check (report only; dReach unchanged)

Date: 2026-10-01. Branch `v2-stagewise-pilots`, base commit `657f54a`, code at `89cd600`. ΔW = 4.

| Item | Value |
|---|---|
| Script | `tools/v2/dreach_mask_check.py` (new, read-only) |
| Data | `results/v2_pilots/dreach_mask_check/dreach_mask_check.csv`, one row per (candidate, tier) |

**Candidates.** 168 (candidate, tier) evaluations, all `valid=True`:
- the analytic equilibrium and ê ≡ 0, each at q = 50 and q = 60;
- all 80 existing final checkpoints (70 at q=50, 10 at q=60), read only from `/home/fjiang4/tournament_experiment/experiments/`;
- each on the development tier (state step 4) and the final tier (state step 2).

**What is compared.**
- **Official mask R₂:** `utils/dp_br_verifier.py:438-469`. It is the grid nodes covered by the **closed** interval [a^BR₁(0) − ê₁(0) − 2q, a^BR₁(0) − ê₁(0) + 2q] ∩ D₂, with a 1e−9 tolerance.
- **Reference support:** the **open** set {d : |d − drift_BR| < 2q}, with drift_BR = a^BR₁(0) − ê₁(0). The opponent plays ê₁(−0) = ê₁(0).
- **Alternative dReach:** Δ₁(0) + max over the reference support of Δ₂. It is computed for comparison only. The official dReach and its code are unchanged.

## Results

### Mask comparison

Holes are reference nodes not in R₂; extras are R₂ nodes outside the reference support.

| q | Tier | Evaluations | Total holes | Evaluations with 0 / 1 / 2 extras |
|---|---|---|---|---|
| 50 | development | 72 | **0** | 69 / 0 / 3 |
| 50 | final | 72 | **0** | 71 / 0 / 1 |
| 60 | development | 12 | **0** | 11 / 0 / 1 |
| 60 | final | 12 | **0** | 12 / 0 / 0 |

These counts include the analytic and zero candidates; the 80 checkpoints alone account for 0 holes and 2 two-extra cases (q=50, development).

- **No holes anywhere.** The official mask is an interval cover, not a node-based PMF, so it does not reproduce the problem found in the forward PMF.
- **Every extra node lies exactly on the support boundary.** All 10 extra nodes are at distance 0.0 from |d − drift_BR| = 2q (column `extra_dist_to_support_edge`).
  - They appear only when drift_BR ± 2q falls exactly on a grid node. Examples: the analytic equilibrium at q=50 on both tiers (±100) and at q=60 on the development tier (±120).
  - These nodes are in R₂ because R₂ uses a closed interval, while the reference is open.

### Effect on dReach

| Quantity | Value |
|---|---|
| (official − reference-mask)/ΔW, 167 of 168 evaluations | exactly 0 |
| (official − reference-mask)/ΔW, the one exception | **+1.408e−3** |

The exception is `two_stage_q50_restarts_20260924/tel_q50_s10224`, development tier:
- drift_BR = 0.000000: the BR action at the root equals ê₁(0) on that effort grid.
- So R₂ includes the boundary nodes d = ±100.
- max Δ₂/ΔW is 0.009661 over R₂ but 0.008253 over the reference support. The maximum sits at a boundary node.
- Official dReach/ΔW is 0.009667; the reference-mask value is 0.008259.
- On the final tier the same checkpoint has drift_BR = −0.108828, no extras and no difference.

No evaluation has a negative difference: the official dReach is never smaller than the reference-mask value.

## Interpretation (labelled)

The official reach mask has no holes. Its only deviation from the continuous support is the closed-versus-open boundary convention, which adds at most the two boundary nodes. That raised dReach in 1 of 168 evaluations, by 1.4e−3·ΔW. That amount is below the C-phase threshold of 0.01·ΔW, but it is not negligible relative to it. No change is proposed; dReach stays as is.

## Reproduce

```bash
cd tools/v2 && OMP_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python dreach_mask_check.py --raw /home/fjiang4/tournament_experiment/experiments --out ../../results/v2_pilots/dreach_mask_check/dreach_mask_check.csv
```
