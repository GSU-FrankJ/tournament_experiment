# Phase 01: Phase A — zero-GPU feasibility screen

## Objective
Decide, without spending any GPU/training budget, whether a best-response-distance
trigger can fire NEAR e*=25 (not at mode≈18 like the gain trigger), and how long
the ramp window must be. Output a Gate-A recommendation.

## The three analyses
1. **Deterministic BR curve** (signal-in-principle, high-κ regime): sweep a
   symmetric deterministic center c∈[16,26]; compute payoff-gain(c) and best
   response BR(c) via `exploitability_frozen_profile`. Expect gain flat, distance
   `|BR(c)−c|` steep near 25. If distance is ALSO flat → design dead, stop.
2. **Explore-κ stochastic backtest** (decisive): at each c2 seed's stored
   (mode, kappa) trajectory, reconstruct BR against opponents SAMPLED from the
   actual explore-κ Beta policy (κ=20), and read the BR-distance at the historical
   gain-trigger point (mode≈18) and along the trajectory. If distance is already
   <~2 at mode≈18 during explore → distance trigger fires early too, same failure.
3. **Window calibration**: from c2 ramp segments, measure mode velocity
   (units/update) per κ stage; estimate updates needed to cross the residual
   distance (trigger→e*); set `--kappa-stage-hold`.

## Files
- Add: `tools/claim_a_phase_a_screen.py` (diagnostic; matches `tools/phase0_*.py`).
- Write findings: `docs/tasks/claim-a-dev-trigger-retrain/phase01_findings.md`
  + raw JSON `phase01_screen.json` in the task folder.
- Read-only: `results/three_players/convergence/ppo_3p_q35.0_seed{42..46}_c2_mode_conc_convergence.json`.

## Verification
- Script runs to exit 0, prints a report, all numbers trace to the JSON dump.
- Gate-A recommendation written with the pre-registered kill condition applied.

## Parameters (3P q35)
k=0.001, w_H=6.5, w_L=3.0, q=35, l=[0,0,0], n=3, bounds=[0,100], e*=25.
