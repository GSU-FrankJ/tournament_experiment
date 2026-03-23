# Phase 01: Run q=35 training for three experiments

## Objective
Run PPO (5 seeds each) and gradient for q=35 across three_players, different_cost, and different_ability. All runs use config defaults (episodes=6,144,000 for PPO, i.e. 1500 updates).

## Inventory

### Existing q=25 data (to be replaced by q=35)

| Experiment | PPO seeds | Gradient |
|---|---|---|
| three_players | 42, 123, 456, 789, 1024 | Yes (q=25, q=40, q=55) |
| different_cost | 42, 123, 456, 789, 1024 | No q=25 (only q=40) |
| different_ability | 42, 123, 456, 789, 1024 | No q=25 (only q=40) |

### Runs to execute

**PPO — 15 runs total** (5 seeds x 3 experiments, ~1500 updates each)

```bash
# three_players (5 runs)
python run/run_three_players.py --method ppo --q 35 --seed 42 --episodes 6144000
python run/run_three_players.py --method ppo --q 35 --seed 123 --episodes 6144000
python run/run_three_players.py --method ppo --q 35 --seed 456 --episodes 6144000
python run/run_three_players.py --method ppo --q 35 --seed 789 --episodes 6144000
python run/run_three_players.py --method ppo --q 35 --seed 1024 --episodes 6144000

# different_cost (5 runs)
python run/run_different_cost.py --method ppo --q 35 --seed 42 --episodes 6144000
python run/run_different_cost.py --method ppo --q 35 --seed 123 --episodes 6144000
python run/run_different_cost.py --method ppo --q 35 --seed 456 --episodes 6144000
python run/run_different_cost.py --method ppo --q 35 --seed 789 --episodes 6144000
python run/run_different_cost.py --method ppo --q 35 --seed 1024 --episodes 6144000

# different_ability (5 runs)
python run/run_different_ability.py --method ppo --q 35 --seed 42 --episodes 6144000
python run/run_different_ability.py --method ppo --q 35 --seed 123 --episodes 6144000
python run/run_different_ability.py --method ppo --q 35 --seed 456 --episodes 6144000
python run/run_different_ability.py --method ppo --q 35 --seed 789 --episodes 6144000
python run/run_different_ability.py --method ppo --q 35 --seed 1024 --episodes 6144000
```

**Gradient — 3 runs total** (1 per experiment, deterministic so no seeds needed)

```bash
python run/run_three_players.py --method gradient --q 35
python run/run_different_cost.py --method gradient --q 35
python run/run_different_ability.py --method gradient --q 35
```

## Execution plan

1. Gradient runs are fast (seconds) — run sequentially first
2. PPO runs are long — launch in tmux sessions, parallelism depends on available GPUs
   - Naming: `tmux new-session -d -s 3p_q35_s42 "python run/run_three_players.py --method ppo --q 35 --seed 42 --episodes 6144000"`
   - Convention: `{exp}_{q}_{seed}` e.g. `3p_q35_s42`, `dc_q35_s123`, `da_q35_s789`
3. Monitor with `tmux ls` and spot-check logs

## Expected output files

```
results/three_players/convergence/ppo_3p_q35.0_seed{42,123,456,789,1024}_baseline_convergence.json
results/three_players/convergence/gradient_3p_q35.0_convergence.json

results/different_cost/convergence/different_cost_ppo_q35.0_seed{42,123,456,789,1024}_baseline_convergence.json
results/different_cost/convergence/different_cost_gradient_q35.0_convergence.json

results/different_ability/convergence/different_ability_ppo_q35.0_seed{42,123,456,789,1024}_baseline_convergence.json
results/different_ability/convergence/different_ability_gradient_q35.0_convergence.json
```

## Theoretical efforts at q=35

| Experiment | Formula | e* |
|---|---|---|
| two_players | (w_H - w_L) / (4qk) | 62.50 |
| three_players | (w_H - w_L) / (4qk) | 62.50 (same formula) |
| different_cost (k1=0.0004, k2=0.00055) | asymmetric cost formula | e1*=50.26, e2*=36.55 |
| different_ability (l1=10, l2=5) | ((2q-(l1-l2))*(w_H-w_L))/(8kq^2) | 58.04 (symmetric) |

Participation constraint: EU(e*)=3.19 > w_l=3.0 — satisfied (unlike q=25).

## Verification

1. All 18 expected convergence JSON files exist
2. Each JSON has `"q": 35.0` and correct theoretical effort per experiment type (see table above)
3. Gradient runs converge to their respective e* (exact or near-exact)
4. PPO convergence quality: check gap from respective e* per seed
5. Compare convergence rate with q=25 results (q=35 should be easier since it's further from participation constraint boundary at q_min=33.07)
