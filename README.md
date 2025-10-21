# Tournament Experiment

This repository contains game‑theoretic tournament experiments. The current active track focuses on the one‑stage, two‑player symmetric tournament and verifies whether a PPO agent can learn the theoretical symmetric effort.

## Project Structure

```
tournament_experiment/
├── config/              # Experiment configurations
├── envs/                # Environments (two-player one-stage active)
├── agents/              # Agents (clean PPO for two-player is active)
├── run/                 # Entry points (two-player runner active)
├── results/             # Output CSVs and figures
├── utils/               # Probability, theory, plotting, logging
└── backup/              # Archived legacy scripts/environments
```

Legacy multi-player / two‑stage files have been moved to `backup/` to keep the two‑player track minimal.

## Installation

We recommend Python 3.8+ and a virtual environment.

```
pip install -r requirements.txt
```

## Two‑Player One‑Stage PPO (Denominator 4)

- Theory (two identical players): `e*(q) = (w_H − w_L) / (4 q k)`.
- Implementation:
  - Environment: `envs/two_players_env.py` (win probability via `utils/prob.p_from_diff`).
  - PPO agent: `agents/ppo_two_players_clean.py` (Beta policy over normalized effort; GAE; clipping; opponent lag support).
  - Runner: `run/run_two_players.py` (adds late‑phase schedules and self‑play sampling control).

### Runner Defaults and Schedules

- gamma: 0.99; gae_lambda: 0.95; value_coef: 0.5; max_grad_norm: 0.5
- steps_per_update: 16384; epochs: 20; minibatch_size: 1024; hidden: 64
- clip schedule: 0.25 early; late‑phase anneal 0.35 → 0.25
- entropy schedule: 0.02 → 0.002 over first 50 updates; forced to 0.0 in last ~30 updates
- learning rate: 3e‑4 base; boosted to 4e‑4 in last ~50 updates
- self‑play sampling:
  - Early phase: learner vs lagged opponent; only learner transitions stored
  - Late phase: fully on‑policy symmetric sampling; store both players’ transitions
- state encoding (3‑D): `[q/60, k/1e‑3, (w_h−w_l)/10]`
- Assumptions documented for this track:
  - Two-player environment uses closed-form expected utilities (no stochastic noise is sampled during rollouts).
  - Evaluation converts the learned Beta policy to actions via the distribution mean; we do **not** switch to the mode even when α,β>1.

### Config (defaults)

- Prizes: `w_H=6.5`, `w_L=3.0`
- Cost: `k=0.0004` (fixed)
- Noise list: `q_list=[25.0, 40.0, 55.0]`
- Effort bounds: `[0, 200]`

### Train and Evaluate

- Train once across all q’s (`config/one_stage_two_players.py:q_list=[25,40,55]`) and evaluate each q:

```
python3 run/run_two_players.py --method ppo --episodes 131072
```

- Train only on a specific q (and evaluate that q):

```
python3 run/run_two_players.py --method ppo --q 40 --episodes 131072
```

Tips:
- `--episodes` is the total number of environment steps (bandit: one step per episode). Set it to a multiple of `steps_per_update=16384` for clean updates (e.g., 65536, 131072, 262144).
- Override the training/eval set via `--q`; omit it to sweep all values in `q_list`.

- Closed‑form baseline rows (denominator 4):

```
python3 run/run_two_players.py --method gradient --q 40
```

### Outputs

- CSV: `results/one_stage_two_players.csv` (standardized header via `utils.logger.save_standardized_result`).
- Plot: `results/one_stage_two_players.png` with learning curve and e*(q) overlays.

During training, the runner also prints per‑update diagnostics for each evaluation `q`: theoretical `e*`, policy implied effort (via Beta mode if defined, else sample mean), absolute gap, and current entropy.

## Different Ability (l1 > l2, k1 = k2)

The `run/run_different_ability.py` entry point aligns the “Different ability” plan:

- Closed-form baseline from `config/different_ability_two_players.py` (two ability tiers, equal cost).
- Gradient solver: `agents/different_ability_solver.py` (adaptive LR, ability-aware gradients).
- PPO agent: `agents/ppo_two_players_clean.PPOTwoPlayersBandit` (Beta policy, dynamic clip, entropy/learning-rate schedules, both players stored every rollout).
- Environment: `envs/different_ability_env.DifferentAbilityEnv` (now seeded with true theoretical efforts on construction).

### Reproducing the experiments

- **Single configuration** (default k=0.0004, w_H=6.5, w_L=3.0, q=25, effort range [0,200]):

```
python3 run/run_different_ability.py --method both --q 25 --episodes 131072 --steps-per-update 8192 --epochs 20 --log-interval 5
```

Adjust `--steps-per-update`, `--epochs`, `--updates`, `--episodes`, and `--minibatch-size` for sensitivity tests.  
Use `--skip-history` when you only need final CSV rows (skips gap trace files).

- **Full parameter grid** (k,w pairs {(0.0004,6.5,3.0), (0.0005,8.0,3.0)}, q∈{25,40,55}, effort ranges {[0,100],[0,200]}):

```
python3 run/run_different_ability.py --method both --grid --updates 160 --steps-per-update 8192 --log-interval 4
```

This command produces gradient + PPO rows for all 12 combinations with consistent schedules. Override `--seed` for repeats.

### Outputs

- CSV aggregates: `results/tables/different_ability_two_players.csv` (columns: method, parameters, theoretical efforts, final efforts, episodes, updates, max_gap, quality, effort bounds, PPO hyperparameters).
- Per-run traces: `results/traces/different_ability/*.csv` (update-by-update gap diagnostics for gradient and PPO).
- Gap plots: `results/plots/different_ability/*_gap.png`.
- Parameter sensitivity summary: `results/plots/different_ability/parameter_sensitivity.png` (mean final max-gap vs q for each parameter tuple).

Per-update logging now prints theoretical efforts, current policy efforts, max gap, entropy coefficient, clip threshold, and instantaneous learning rate to ease convergence checks. PPO stores both players’ trajectories from the first update to improve stability (matching the asymmetric-cost runner behaviour).

## Contact

For questions or contributions: `fjiang4@student.gsu.edu`.
