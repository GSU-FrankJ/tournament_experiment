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

## Step 2: actor-critic + PPO update (NOT STARTED)

- Beta policy (repo invariant), critic over the 2-D state [t/T, d/(q sqrt t)].
  state_dim=2 (per-cell training drops the constant params). Reuse the Beta
  head / mean extraction pattern from the one-stage `ActorCriticMeanConc`,
  but a fresh network (different input dim, no theory_align concentration
  ramp — that was a one-stage stabilizer).
- gamma=1, lambda=1 (override the agent defaults 0.99/0.95 — they must NOT
  leak in). Clip, entropy, epochs per the training protocol.
- Policy extraction = Beta MEAN (repo invariant).

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
