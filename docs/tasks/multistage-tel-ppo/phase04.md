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

## Step 3: production rollout loop + exploring starts (COMPLETE 2026-07-09)

`run/run_multi_stage.py` + vectorized env support (`env.step_batch`,
`env.obs_batch`, `env.sample_exploring_starts_batch`) + agent batch methods
(`act_batch`, `buffer.add_np`).

- VECTORIZED self-play rollout: N parallel episodes, batched policy forward
  + batched env transition; heterogeneous exploring-start stages handled by
  stepping only still-active envs. Each env's two players stored as two
  SEPARATE trajectories (the trajectory-aware GAE contract).
- `env.step_batch` verified to match the scalar `step` under common random
  numbers to 2e-7 (`tools/verify_multi_stage_env.py` check 5) -- single
  source of truth for the game math, no divergence.
- Param validation (`config.validate`) runs BEFORE training: q <= q_crit
  raises. Periodic DP-verifier eval; best checkpoint tracked by validation
  dReach; convergence JSON under results/multi_stage/.

### Validation run (T=2, q=50, seed 42, 1000 upd x 256 ep, entropy 0.005, CPU ~4.6 min)

**The pipeline works and the policy CERTIFIES from update 100 on.**
Final EXP=0.018 (EXP/DW=0.46%), dReach 0.033-0.086 (<< 3% gate),
cert=True. stage-1(0)=44.7 (g1=46.67, RE_1~4%). stage-2 learned
[13.8, 45.1, 59.2, 28.3, 7.5] vs CF [0, 35, 70, 35, 0] at
d=[-100,-50,0,50,100]: clearly hump-shaped, peak 59 at d=0.

Two honest findings (NOT blockers; the certificate holds):
1. **Peak undershoot** (59 vs 70): the exploration-smoothed mu*(kappa)
   effect from the one-stage saga, in the multi-stage setting. Exactly why
   the Claim-B framing (verifier certifies; not "e_hat ~ e*") is correct.
   The verifier certifies the smoothed candidate as an approx-MPE.
2. **Stage-2 asymmetry** (45 at d=-50 vs 28 at d=+50) where the closed form
   is even. Certificate still holds because BR-reachable mass concentrates
   near d=0. Likely finite-sample/single-seed; revisit under step-4 seed
   robustness / longer budget / optional symmetry regularization.

Validation JSON not committed (single-seed CPU characterization); the real
gated multi-seed runs are step 4.

## Step 4: extraction + verifier hook + pre-registered T=2 gate (NOT STARTED)

- Extract e_hat_t(d) (Beta mean) on a d-grid; feed the phase03 verifier.
- Pre-register the T=2 acceptance gate (RE_1, RPE_2 vs closed form; EXP^UCB
  / dReach thresholds) BEFORE the first GPU run. No GPU on T>=3 until T=2
  passes.

## Constraints

- Long runs in tmux (repo rule). Sampled training rewards only. Both
  players' transitions stored. Do not touch one-stage code.
