# Audit Report: Theory Align V2 (One-Stage Two-Player PPO)

## Code Map (Step 0)
- A) Theory e* source: `utils/theory.py:e_star_two_players` (closed-form), echoed in `config/one_stage_two_players.py` and logged in `run/run_two_players.py` (uses denominator 4).
- B) Environment reward: `envs/two_players_env.py:expected_utility` uses `utils/prob.py:p_from_efforts` and cost term `-k*e^2`.
- C) Effort parameterization: `agents/ppo_two_players_clean.py:ActorCritic.dist` (Beta on [0,1]) and `PPOTwoPlayersBandit.act` mapping to effort via `low + a*(high-low)`; policy mean logged via `utils/rollout_stats.compute_policy_mean_effort`.
- D) PPO update: `agents/ppo_two_players_clean.py:PPOTwoPlayersBandit.update` (log_prob/ratio/clip/advantage; early-stop on ratio/KL).
- E) Self-play data collection: `run/run_two_players.py` (selfplay stores both; vs_opponent stores only learner-generated transitions).

## Evidence & Root-Cause Analysis (Falsifiable)
1) **Theory matches env?**  
   Hypothesis: e*=54.69 is correct for the current env reward.  
   Evidence chain: env utility is `w_l + p(e_i,e_j)(w_h-w_l) - k e_i^2`; `p_from_efforts` has slope 1/(2q) at d=0. This implies e*=(w_h-w_l)/(4kq).  
   Falsifiable check: `run/exp_sanity_theory_q40.py` grid scan prints argmax vs theory. If |argmax-e*|>0.5, the theory is not aligned with env.

2) **Why conc=100 still under-shoots? (Jensen bias + parameterization)**  
   Evidence chain from logs: updates 480-500 show `conc_mean=100`, `entropy=0`, policy mean effort ~44-50 vs e*=54.69, `mean_vs_sample_gap` ~1-3, and frequent `ratio_exceeded` early-stops.  
   Hypothesis: with effort range [0,200], conc=100 implies std~8-9 effort units, so Jensen bias in `-k e^2` and nonlinear `p_from_efforts` shifts the stochastic optimum below the deterministic e*.  
   Falsifiable check: `run/exp_jensen_gap.py` reports var/std and the best-response mean vs e* for each concentration. If conc=100 yields |best_mean - e*|>0.5, variance is sufficient to explain the under-shoot.  
   Additional structural cause: alpha/beta heads entangle mean and concentration, making it hard for PPO to raise concentration without shifting mean (seen as conc stuck at 100 despite `conc_weight`).

3) **Update aggressiveness (ratio early-stops) adds oscillation**  
   Evidence chain: ratio_max repeatedly exceeds threshold (2.2 with clip_eps=0.6), triggering early-stop; policy mean jumps around.  
   Falsifiable check: compare early-stop rate and mean drift before/after v2 stability tweaks (logged in `[TheoryAlignV2]`).

## Fix Strategy (Minimal & Behind `--theory-align-v2`)
- **Mean+Concentration head (Strategy 1)**: new `ActorCriticMeanConc` decouples mean and concentration to let PPO shrink variance without shifting mean.  
  Files: `agents/ppo_two_players_clean.py` (new class + v2 switch).
- **Variance penalty (Strategy 2)**: add `L_var = lambda * mean(Var_effort)` to loss to suppress Jensen bias while preserving sampling PPO.  
  Files: `agents/ppo_two_players_clean.py` (v2 var penalty).
- **Stability tweaks (Strategy 3, v2 only)**: mild reductions in LR/epochs/clip/target_kl to reduce ratio spikes and early-stop oscillations.  
  Files: `run/run_two_players.py` (v2-only overrides).
- **New diagnostics**: `[TheoryAlignV2]` logs concentration, variance, and early-stop rate.  
  Files: `run/run_two_players.py`.

## Minimal Experiments
1) Theory sanity (Step 1):  
   `python run/exp_sanity_theory_q40.py --q 40 --opp-effort 54.69 --e-min 0 --e-max 100 --step 0.1`
2) Jensen gap (Step 2):  
   `python run/exp_jensen_gap.py --q 40 --num-samples 20000 --grid-window 10 --grid-step 0.5`
