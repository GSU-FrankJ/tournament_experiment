# Claim-A dev-distance-trigger retrain

## Goal
A second, redesigned attempt at **Claim A** ("PPO self-play learning itself
converges to equilibrium effort"), after Component-2 (exploitability-gain trigger)
was falsified. The redesign replaces the trigger *observable*: fire the κ ramp on
**best-response distance** `|best_dev_effort − policy_mean| < τ_dist` (a sampled,
in-loop quantity already computed by `eval_exploitability_3p`), not on payoff-gain
`EXP_raw < 0.05`. Rationale below.

## Why the observable, not the parameters, must change
Verified diagnosis of the Component-2 negative result (see
`docs/phase0_response_to_revision_plan.md` §6.4, corrected 2026-07-02 against the
stored mode/mean trajectories):
- The gain trigger `EXP_raw<0.05` fired at mode≈17.9–18.2 (~7 effort units below
  e*=25) on ALL four triggered seeds, because on the flat payoff plateau
  (Finding B) local deviations look unprofitable far from e*. Gain has no signal
  in [18, 25].
- After firing, the κ ramp did NOT freeze the policy — mode kept moving +2.7..+7.2
  toward e* — but the fixed 80-update window was too short to cover the residual
  distance, and per-seed travel variance produced the 6.5× std blow-up.
- Therefore: (a) need a trigger observable that is steep in [18, 25] so it fires
  NEAR e*, and (b) need a longer/adaptive ramp window.

Best-response distance `|BR − mean|` is the candidate: at mode=18 the sampled BR
is ~25 (distance ~7); at mode≈25 the BR≈mean (distance→0). It is sampled-only
(no analytic e*), so it does not violate the CLAUDE.md train/eval split or hand
the agent the equilibrium.

## Scope — what to touch
- Phase A (current): analysis-only. New diagnostic `tools/claim_a_phase_a_screen.py`
  + findings in this folder. Reads existing c2/r5 JSONs. ZERO GPU, ZERO training.
- Phase B+ (only if Gate A passes): a NEW trigger mode behind a flag on the
  existing Component-2 code path in `run/run_three_players.py` + `agents/ppo_three_players.py`.

## Scope — what NOT to touch
- Do NOT modify `results/*/convergence/` (read-only inputs).
- Do NOT modify the `theory_align_v2`, r5, or existing Component-2 (`--mode-conc-ramp`
  gain-trigger) paths — all must stay reproducible. The new trigger is an ADDITIVE
  mode, default off.
- Do NOT introduce analytic e* into any trigger, reward, or policy update.
- Do NOT touch `paper/generator/config.py` theory params.

## Locked decisions (pre-registered before any GPU)
- **New trigger**: `|best_dev_effort − policy_mean_effort| < τ_dist` (τ_dist≈1.0)
  for `ramp_trigger_patience` consecutive evals, ANDed with the existing
  `EXP_raw < ramp_trigger_exp` guard (defence against coarse-grid near-distance
  false positives). Both are already-computed sampled quantities.
- **Window**: lengthen `--kappa-stage-hold` (20→~60) and/or make it adaptive;
  final value set by Phase-A Analysis 3.
- **Reporting metric**: Beta mean (CLAUDE.md invariant), mode as diagnostic — same
  as Component-2.
- **Kill conditions** (pre-registered, honour them — do not iterate to fit):
  - Gate A (end of Phase A, zero GPU): if the BR-distance observable is NOT steep
    in [18,25] under BOTH the deterministic (high-κ) AND the explore-κ stochastic
    regimes — i.e. distance is already <~2 at mode≈18 during explore — then a
    distance trigger fires early too, same failure. STOP, write negative result,
    no GPU.
  - Gate C (after 5-seed q35): triggered seeds with std > 1.0 or mean|err| > 4%
    ⇒ Claim A is dead in this parameterization; no further retrain iterations
    without new authorization.

## Hard rules carried over
- tmux for any run >1 min; no nohup/bare background.
- No fabricated numbers — every reported value traces to a JSON or a logged script run.
- Confirm params with owner before any GPU launch (Phase C).
- Split commits: analysis code vs results vs docs.
