# Rollout Mode: Selfplay

## Overview

PPO training in `run/run_two_players.py` uses selfplay mode, where both players use the learner policy.

## Selfplay Mode

Both players always act from the current learner policy. The opponent lag mechanism is disabled for action selection.

- **Transitions stored:** Both P1 and P2 every step (2 transitions/step)
- **Buffer contents:** 100% learner-generated, all PPO ratios are valid
- **Lag schedule:** Ignored for action selection (opponent never used)

## Historical: Data Mixing Bug (Dec 2025)

Before the rollout mode fix, `run_two_players.py` selected P2's action from either the learner or a lagged opponent based on `use_opponent`, but then unconditionally called `agent.store()` for both cases. The PPO update then computed `ratio = exp(learner_logp - opponent_logp)` — a cross-policy ratio that violates PPO's on-policy requirement. Diagnostics confirmed ~26% of stored transitions in a test run (104/400 at 50% opponent probability) were opponent-generated, causing invalid clipping, distorted GAE advantages, and training instability worst during warmup when `lag_prob=1.0`. The fix ensures selfplay mode never uses the opponent and all stored transitions are learner-generated.

## Code References

| Component | Location | Description |
|-----------|----------|-------------|
| Mode selection | `run/run_two_players.py` (`run_ppo()`) | `rollout_mode` parameter |
| Rollout loop | `run/run_two_players.py` (`run_ppo()`) | Selfplay stores both P1 and P2 transitions |
| PPO update | `agents/ppo_two_players_clean.py` (`update()`) | Loads buffer, computes ratio — assumes all transitions are learner-generated |
