# Phase 04: multi-stage PPO trainer (IN PROGRESS)

The candidate GENERATOR (Claim-B framing). PPO produces exploration-smoothed
effort functions; the phase03 DP verifier certifies them. Built incrementally.

## Step 1: trajectory-aware GAE + rollout buffer (COMPLETE 2026-07-09)

`agents/ppo_multi_stage.py` (`compute_gae_single`, `compute_gae_trajectories`,
`MultiStageRolloutBuffer`) + `tools/test_multi_stage_gae.py` (all PASS).

**Bug fixed.** The one-stage `PPOTwoPlayersBandit._compute_gae` (line 342)
chains `next_value` across consecutive storage indices, resetting only on
`done`. Correct for single-step bandit and for contiguous trajectories, but
it MISBOOTSTRAPS on interleaved p0/p1 storage: p0 stage-1's bootstrap reads
p1 stage-1's value. The unit test reproduces this (flat GAE gives the wrong
p0-stage-1 advantage) and shows the trajectory-aware GAE gives the correct
per-player values on the same data.

**Design.** The new GAE is ordering-independent: advantages are computed per
trajectory with a zero terminal bootstrap (finite horizon), so the caller
cannot reintroduce the interleaving bug regardless of storage order. The
one-stage agent is untouched (out of scope, precious results).

Unit-test coverage: hand-computed 2-step (gamma=1, lam=0.5); gamma=lam=1
== Monte Carlo return (the plan's main spec); ordering independence;
interleaving-bug reproduction + fix; buffer guard rails.

## Step 2: actor-critic + PPO update (COMPLETE 2026-07-09)

`agents/ppo_multi_stage.py` extended with `MultiStageActorCritic`,
`MultiStagePPOConfig`, `MultiStagePPO`. `tools/smoke_multi_stage_ppo.py`
(all checks PASS, ~17s CPU).

- Beta policy, mean/conc parametrization (Beta MEAN extraction = repo
  invariant), critic over the 2-D state [t/T, d/(q sqrt t)]. Fresh network,
  state_dim=2; NO theory_align ramp or opponent-lag (one-stage stabilizers).
- `MultiStagePPOConfig` defaults gamma=gae_lambda=1.0 (finite-horizon
  economic payoff) — deliberately not the one-stage 0.99/0.95; no leak.
- Standard clipped-surrogate + value-MSE + entropy update; advantages from
  the trajectory-aware GAE (`buffer.compute`), so storage order is safe.
- `effort_function(t, d, T, q)` builds the normalized state and returns the
  Beta-mean effort — the object the phase03 verifier consumes.

Smoke result (T=2, q=50, 150 updates x 32 episodes, tiny budget): finite
diagnostics + in-bounds effort throughout; policy non-degenerate (stage-1
effort ~49.5 near g1=46.67, stage-2 moving); entropy drops (sharpening);
verifier hook yields EXP=0.073 (EXP/DW=1.8%), dReach=0.50, uncertified.
Convergence to the closed form is NOT asserted here (step 3/4 with a real
budget); this step verifies the plumbing is correct and numerically sane.

## Step 3: self-play rollout loop + exploring starts (NOT STARTED)

- Both players share the symmetric policy; player j observes -d. Store each
  player's episode as its OWN trajectory in the buffer (never interleave).
- Exploring starts via `env.reset_exploring()` (phase02) so off-path (t, d)
  get gradient signal -> supports the full approximate-MPE claim.
- Sampled rewards only (env already enforces).

## Step 4: extraction + verifier hook + pre-registered T=2 gate (NOT STARTED)

- Extract e_hat_t(d) (Beta mean) on a d-grid; feed the phase03 verifier.
- Pre-register the T=2 acceptance gate (RE_1, RPE_2 vs closed form; EXP^UCB
  / dReach thresholds) BEFORE the first GPU run. No GPU on T>=3 until T=2
  passes.

## Constraints

- Long runs in tmux (repo rule). Sampled training rewards only. Both
  players' transitions stored. Do not touch one-stage code.
