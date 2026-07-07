# Component-2 mode-conc retrain

Status: complete (negative result — Claim A not supported; disposition recorded in
SESSION_STATE.md and docs/phase0_response_to_revision_plan.md §6. Successor analyses:
docs/tasks/claim-a-dev-trigger-retrain/ → docs/tasks/claim-a-nonlocking-continuation/.)
Current phase: closed

## What's done
- Recovered Phase-0 scripts + reran definitive 6-cell verify (6/6 PASS); results
  in `results/phase0_verify_20260701_1941.log`, consolidated into SESSION_STATE.md.
- Owner authorized Component-2 retrain (walk toward Claim A), 2026-07-01.
- Read integration points: `ActorCriticMeanConc` (ppo_three_players.py:75),
  training loop + `theory_align_v2` time-ramp (run_three_players.py:784-805),
  in-loop exploitability eval + stop (run_three_players.py:1000-1075).
- Reporting metric decided: mean (owner).

## What's done (cont.)
- Implemented `ActorCriticModeConc` + PPOConfig fields + init branch (agents/ppo_three_players.py).
- Implemented ramp state machine + 4 CLI flags + mode/kappa/ramp_phase diagnostic
  columns + stop gating (run/run_three_players.py).
- Smoke-tested on CPU (loose params): explore→ramping→done, κ 20→50→100→200 at
  stage_hold, α+β−2 pins to κ exactly, mean+mode logged, JSON well-formed, no crash.
- ENV FIX: vector2 GPUs are V100 (sm_70); torch 2.12.1+cu130 lacked sm_70 kernels
  (no kernel image error). Reinstalled torch==2.5.1+cu121 (owner-approved), verified
  sm_70 in arch list + real GPU matmul + GPU smoke marches the ramp identically.
  Re-pinned requirements.lock. numpy 2.5.0 unchanged.

## What's done (cont.) — RESULT
- Launched 3P q35, seeds 42-46, tag `c2_mode_conc`, spec params (ramp-trigger-exp
  0.05 x3, stage-hold 20, exploit-every 10, full episodes), one seed per GPU (0-4).
  All 5 completed (2026-07-02, ~7h wall for the batch).
- **Raw PPO effort at final (no polish), vs e*=25:**
  seed42=22.698(-9.21%), seed43=22.989(-8.04%), seed44=25.663(+2.65%),
  seed45=20.865(-16.54%), seed46=21.417(-14.33%, ramp NEVER triggered — ran the
  full 1500-update budget stuck in "explore" at kappa=20).
- **vs r5 baseline (old ramp, raw PPO, same metric):**
  r5 mean=22.993 std=0.255 (tight, [22.75,23.42]) mean|err|=8.03%
  C2  mean=22.726 std=1.666 (wide,  [20.86,25.66]) mean|err|=10.16%
- **VERDICT: Claim A NOT supported.** Mean did not move toward e* (statistically
  indistinguishable from r5, arguably worse). Std exploded 6.5x — the retrain does
  not reliably converge, it scatters. 1/5 seeds never even triggered the ramp.
  Mechanism diagnosis (revised 2026-07-02 against stored mode/mean trajectories):
  the `EXP_raw<0.05` trigger fires wherever the flat payoff plateau (Finding B)
  makes local deviations look unprofitable, not specifically near e* — all four
  triggered seeds fired at mode ~17.9-18.2 (mean ~20.8-21.1), ~7 units below e*,
  at near-identical positions. The kappa ramp does NOT freeze the policy there:
  within the ~80-update ramp window the mode kept moving toward e* by +2.7..+7.2
  units (s42 17.9->22.4, s43 18.2->22.7, s44 18.2->25.4 overshooting, s45
  17.9->20.6). Actual failure mode: premature trigger + fixed-length ramp window
  too short to cover the remaining distance; how far each seed travels inside
  the window varies, which is where the 6.5x std comes from. RETRACTED (do not
  resurrect): the earlier "ramp FREEZES whatever mean was there / same failure
  mode as the theory_align_v2 lock" wording is contradicted by the trajectories;
  the two designs share only the outcome (PPO alone does not reliably reach e*).
- Component-2 disposition: **evidence AGAINST Claim A** (not merely "insufficient
  evidence for A" — the std/mean comparison actively argues against the retrain
  approach fixing the undershoot). Code is kept (do not delete — it's a real,
  reproducible negative result with full provenance), gated behind
  `--mode-conc-ramp` (default off), does not touch r5/theory_align_v2 paths.

## What's next
1. Reported to owner (this session) — Claim B stands as the paper's central claim
   pending owner's decision (see SESSION_STATE.md "OPEN DECISIONS" — needs re-vote
   now that decision 3 has an answer: Component-2 retrain does NOT rescue Claim A).
2. If owner still wants to pursue A: would need a fundamentally different retrain
   design (e.g., a trigger tied to effort proximity rather than exploitability, or
   a much longer/gentler ramp) — out of scope for this task without new authorization.
3. Convergence JSONs are the source of truth for these numbers:
   `results/three_players/convergence/ppo_3p_q35.0_seed{42..46}_c2_mode_conc_convergence.json`

## Blockers
- None technically. Awaiting owner's call on Claim A vs B given this result.

## Open questions for after the run
- If raw policy-mean still undershoots e*=25 at κ=200 → Claim A NOT supported;
  report back and revisit A-vs-B with owner. Do not massage numbers to fit A.
