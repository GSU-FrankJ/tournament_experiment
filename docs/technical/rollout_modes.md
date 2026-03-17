# Rollout Modes: Selfplay vs vs_opponent

## Overview

The `--rollout-mode` flag in `run/run_two_players.py` controls how the two players interact during PPO training. Two modes exist:

- **`selfplay`** (default for PPO) — Both players use the learner policy
- **`vs_opponent`** — Player 1 uses the learner; Player 2 may use a lagged opponent

The flag was introduced to fix a data mixing bug where opponent-generated transitions contaminated the PPO buffer. Selfplay avoids the issue entirely and is the recommended mode.

## Selfplay Mode

Both players always act from the current learner policy. The opponent lag mechanism is disabled for action selection.

- **Transitions stored:** Both P1 and P2 every step (2 transitions/step)
- **Buffer contents:** 100% learner-generated, all PPO ratios are valid
- **Lag schedule:** Ignored for action selection (opponent never used)

This is the default when `--method ppo` is specified.

## vs_opponent Mode

Player 1 always uses the learner policy. Player 2 probabilistically uses a lagged copy of the policy based on the `lag_prob` schedule.

- **P1 transitions:** Always stored
- **P2 transitions:** Stored **only** when P2 used the learner; opponent-generated steps are treated as environment dynamics and discarded
- **Transitions/step:** 1-2 (average ~1.5 at lag_prob=0.5)

This mode is useful for testing robustness against non-stationary opponents but produces fewer training samples per step.

## Historical: Data Mixing Bug (Dec 2025)

Before the rollout mode fix, `run_two_players.py` selected P2's action from either the learner or the lagged opponent based on `use_opponent`, but then unconditionally called `agent.store()` for both cases. The PPO update then computed `ratio = exp(learner_logp - opponent_logp)` — a cross-policy ratio that violates PPO's on-policy requirement. Diagnostics confirmed ~26% of stored transitions in a test run (104/400 at 50% opponent probability) were opponent-generated, causing invalid clipping, distorted GAE advantages, and training instability worst during warmup when `lag_prob=1.0`. The fix separates the two modes explicitly: selfplay never uses the opponent, and vs_opponent discards opponent-generated transitions.

## Code References

| Component | Location | Description |
|-----------|----------|-------------|
| Mode selection | `run/run_two_players.py` (`run_ppo()`) | `rollout_mode` parameter, CLI flag `--rollout-mode` |
| Rollout loop | `run/run_two_players.py` (`run_ppo()`) | Mode-branched loop: selfplay stores both; vs_opponent conditionally stores P2 |
| PPO update | `agents/ppo_two_players_clean.py` (`update()`) | Loads buffer, computes ratio — assumes all transitions are learner-generated |
