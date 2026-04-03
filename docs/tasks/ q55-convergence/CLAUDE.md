# Task: q55-convergence

## Goal
Fix PPO convergence at q=55 for both 2-player and 3-player experiments. Currently 2p baseline converges 1/12 seeds, 3p has gap ~2.6-4.5. The conc_max=1000 workaround achieves 5/5 for 2p but is a per-q manual tuning — we want a principled, general fix.

## Scope
- **In scope**: entropy schedule, concentration control, adaptive entropy, 2p and 3p q=55 experiments
- **Out of scope**: q=25 (NE invalid), q=35/40 (already converge), different_cost, different_ability, paper figures

## Root Cause Analysis
In self-play, learning signal comes entirely from policy stochasticity. As concentration grows, Var(e₁-e₂) → 0, reward differences become indistinguishable from noise, and PPO loses directional signal. q=55 fails because the distance from initialization (~100) to equilibrium (39.77) is larger than q=35/40, so the agent needs more updates with good signal — but concentration grows on a fixed schedule that doesn't account for this.

## Key Files
- `agents/ppo_two_players_clean.py` — PPO trainer, ActorCritic / ActorCriticMeanConc
- `run/run_two_players.py` — entropy schedule (lines 732-904), conc ramp
- `run/run_three_players.py` — 3p runner
- `config/one_stage_two_players.py` — default hyperparameters
- `envs/three_players_env.py` — 3p environment

## Current Entropy Schedule
```
entropy_coef: 0.03 → 0.03 (hold ~67%) → 0.005 (final)
```
Concentration growth is emergent from gradient updates in standard mode (no explicit schedule).

## Key Findings
- conc_max=1000 fixes 2p q=55 (5/5 converge, gap 0.5-3.5)
- 2p baseline: 11/12 seeds stuck at effort ~48-55, exploitability oscillates 0.05-0.4
- 3p q=55: NE is valid (q > q_crit=40.5), gap 2.6-4.5, all stopped by exploitability
- Theoretical equilibrium e*=39.77 is the same for 2p and 3p (verified from FOC)

## Constraints
- Never modify convergence JSONs without confirmation
- Long-running experiments in tmux
- One logical change per commit
- Verify q=35/40 not regressed after any change
