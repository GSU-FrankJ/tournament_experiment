# Phase 2 opening checks (1a–1c)

Date: 2026-10-01. Branch `v2-stagewise-pilots`, base commit `657f54a`. ΔW = 4.

`OPEN/` means `results/v2_pilots/phase2_opening/`.

Scripts:
- `tools/v2/reference_delta2.py` (1a)
- `tools/v2/onpath_check.py` (1b, 1c)

They change no training or verifier code.

---

## 1a. Verifier Δ₂ against an independent reference

### Reference

`delta2_reference` in `tools/v2/reference_delta2.py`.

**What it imports.** Only `F_xi`, the CDF validated against the environment's Monte Carlo in Phase 1, and `g2_two_stage`, which is used only to build the test policy. It reuses no verifier search, quadrature or interpolation.

**Resolution.**
- A dense effort grid of 100,001 points on [0, 100], spacing 0.001.
- Then scipy `minimize_scalar` (method `bounded`, `xatol` 1e−12) on [e_j − 0.001, e_j + 0.001] around the best dense point e_j. The larger of the two values is kept.
- The refinement gains at most 2.7e−11·ΔW over the dense grid (`OPEN/delta2_reference.csv`, column `refine_gain_over_dw`).

### Test policies

The stage-1 action is set to e₁* (stage 1 does not enter Δ₂).
1. ê₂ ≡ 0.
2. ê₂(d) = e₂*(d − q/2). This is asymmetric, so the verifier's ê₂(−d) opponent lookup is exercised.

### Result

Source: `OPEN/delta2_reference.csv`, one row per grid point. diff = (Δ₂^verifier − Δ₂^ref)/ΔW.

| q | Policy | Tier (state, effort step) | Grid points | max \|diff\| | at d | max diff (positive = verifier above the continuous max) | max Δ₂^ref |
|---|---|---|---|---|---|---|---|
| 50 | zero | development (4, 1) | 101 | 2.2e−16 | −28 | 2.2e−16 | 0.25897 |
| 50 | zero | 2× (2, 0.5) | 201 | 3.3e−16 | 14 | 3.3e−16 | 0.25926 |
| 50 | shifted | development | 101 | 4.4e−16 | 36 | 4.4e−16 | 0.07412 |
| 50 | shifted | 2× | 201 | 4.4e−16 | 32 | 4.4e−16 | 0.07412 |
| 60 | zero | development | 111 | 2.2e−16 | −32 | 1.1e−16 | 0.19551 |
| 60 | zero | 2× | 221 | **1.66e−08** | **−58** | 2.2e−16 | 0.19551 |
| 60 | shifted | development | 111 | 4.4e−16 | 88 | 3.3e−16 | 0.03976 |
| 60 | shifted | 2× | 221 | 3.3e−16 | −8 | 3.3e−16 | 0.03976 |

### Sign pattern

- The largest positive difference is 4.4e−16·ΔW, which is rounding. **No positive excess beyond rounding.**
- Across all 1,168 grid points, only one difference exceeds rounding: q=60, zero policy, 2× tier, d = −58. There the verifier is **below** the continuous max by 1.66e−8·ΔW.
- At that point the opponent plays 0, so the kink of F_ξ (at y = 0) lies at e = 58.
  - The reference optimum is 58.2243 (column `a_ref`). The verifier's parabola vertex is 58.2118 (column `a_dev_verifier`).
  - The vertex comes from the triple (57.5, 58, 58.5), which straddles the kink.
- **Why agreement is otherwise exact.** The objective ΔW·F_ξ(y) − k e² is piecewise quadratic in e, so the verifier's parabola vertex is exact whenever the three grid points lie in one piece.

### Does it shrink on the finer grid?

No. Both tiers sit at rounding level everywhere except at that single kink-straddling point. d = −58 is not a node of the development grid (step 4), so the effect cannot appear there.

### Gate

The only difference beyond rounding has the expected sign (grid search ≤ continuous max) and is explained by action-grid resolution at a kink of F_ξ. **The gate did not fire.**

---

## 1b. Is e₁*(0) on the stage-1 action grid?

Source: `OPEN/effort_grid_membership.csv`.

| q | e₁* | development (step 1) | dev_2x (step 0.5) | final (step 0.5) |
|---|---|---|---|---|
| 50 | 46.6667 | no (nearest 47, distance 0.333) | no (46.5, 0.167) | no (46.5, 0.167) |
| 60 | 38.8889 | no (39, 0.111) | no (39, 0.111) | no (39, 0.111) |

e₁* is off-grid at both q and on every tier. Grid membership of e₁* therefore does **not** distinguish q=60 from q=50, and cannot by itself explain why the floor is non-zero only at q=60 on the finer grids. As instructed, I made no further investigation.

---

## 1c. On-path set vs {d ∈ D₂ : |d| < 2q}

### Identity check from the code

- **Training environment.** It draws a fresh eps ~ U(−q, q) for each player at **every** stage, with the same q and the same call (`run/run_final_dp_br.py:190-191`, inside the stage loop that starts at line 179).
- **Verifier.** It uses one GL rule for every stage's continuation (`utils/dp_br_verifier.py:338`) and for the forward PMF (lines 474–483).
- So stage-1 and stage-2 shocks are identically distributed.
- **Stage-1 drift.** At the root both players query ê₁ at d = 0 and at −0. `encode_obs` sets the stage-1 gap input to 0 (`envs/curriculum_env.py:74-75`), so the two actions are identical.
  - Measured drift ê₁(0) − ê₁(−0): exactly 0 in all 252 cases (`OPEN/onpath_sets.csv`, column `stage1_drift`).
- Therefore d₂ = ξ₁, and the **continuous** support of the candidate's stage-2 distribution is exactly (−2q, 2q).

### Test result

The candidates were the analytic equilibrium, ê ≡ 0 and **all** 80 final checkpoints, at both q and on 3 tiers: 252 cases in total. Source: `OPEN/onpath_sets.csv`.

**The exact-positivity on-path set differs from {|d| < 2q} in all 252 cases.** The difference depends only on (q, tier), not on the candidate, because the drift is always 0.

| q | Tier | On-path nodes | Nodes with \|d\| < 2q | On-path but \|d\| ≥ 2q | \|d\| < 2q but zero mass (holes) |
|---|---|---|---|---|---|
| 50 | development | 47 | 49 | ±100 | ±40, ±60 |
| 50 | dev_2x | 59 | 99 | ±100 | 42 nodes, e.g. ±10, ±16, ±22, …, ±90 |
| 50 | final | 95 | 99 | ±100 | ±36, ±50, ±64 |
| 60 | development | 51 | 59 | ±120 | ±28, ±48, ±60, ±72, ±92 |
| 60 | dev_2x | 59 | 119 | ±120 | 62 nodes |
| 60 | final | 103 | 119 | ±120 | 18 nodes |

### Why the sets differ

The forward PMF places mass only at the 2·n_GL Gauss–Legendre nodes of ξ, then splits each node's mass linearly onto its two neighbouring grid nodes. This produces two artefacts.

1. **Endpoints.** The outermost GL nodes lie just inside ±2q. With 16 nodes per half, the outermost is at about ±99.5 for q = 50. Part of their mass is split onto the grid node at exactly ±2q, where the true density is 0. So ±2q is on-path.
2. **Holes.** GL nodes cluster near the ends and centre of each half-interval and are sparse in between (spacing up to about 10 at q = 50 with 16 nodes per half). Where two consecutive GL nodes fall in grid cells that are not adjacent, the grid nodes in between receive no mass. They are inside the true support but count as off-path.
   - This is worst on the dev_2x tier: a fine state grid (step 2) combined with only 16 GL nodes per half.

**Correction to `reports/v2/phase1_verifier.md` §2.4.** There I wrote that "the nodes nearest ±2q can receive zero mass". That statement was wrong. The truth is the opposite: the nodes at ±2q receive mass, and interior nodes can receive none.

### Test

`tests/test_v2_verifier.py::test_onpath_equals_open_support` checks set equality as you requested.

- It is marked `xfail(strict=True)`, citing this report, so the suite records the known mismatch rather than hiding it.
- `strict=True` means it will turn into a hard failure if the sets ever start to agree, for example if the rule changes.
- It covers the analytic and zero candidates on all 3 tiers. The 80 checkpoints are covered by the script output above.

### Decision needed before Pilot 2

The rule is kept as is, as you instructed, but it does not produce the set the rationale describes. Options:
- **(a)** Keep the rule. On-path is then a GL-node artefact: it has holes and includes ±2q.
- **(b)** Define on-path as the continuous support of the candidate's stage-2 law. With the stage-1 drift identically 0 under symmetric evaluation, this is exactly {|d| < 2q}; in general it is {|d − drift| < 2q}. No numerical threshold is involved.
- **(c)** Compute the PMF with exact cell masses, i.e. differences of F_ξ over each grid cell, instead of GL nodes. Its support is then {|d − drift| < 2q + h}.

Every per-checkpoint NPZ stores `v_t2_cand_pmf` and the grid, so the Pilot 2 decomposition can be computed afterwards under whichever rule you choose. Phase 2 does not depend on this choice.
