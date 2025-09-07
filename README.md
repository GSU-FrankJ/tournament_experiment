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
  - PPO agent: `agents/ppo_two_players_clean.py` (Beta policy, log_prob, ratio, clip, GAE).
  - Runner: `run/run_two_players.py`.

### Hyperparameters (PPO)

- gamma: 0.99
- gae_lambda: 0.95
- clip_eps: 0.2
- lr: 3e-4 (Adam)
- value_coef: 0.5
- entropy_coef: 0.01
- max_grad_norm: 0.5
- steps_per_update: 2048
- epochs: 15
- minibatch_size: 256 (agent default; runner uses 128)
- state_dim: 3 with features `[q_norm, k_norm, w_gap_norm]`
- hidden: 64

### Config (defaults)

- Prizes: `w_H=6.5`, `w_L=3.0`
- Cost: `k=0.0004` (fixed)
- Noise list: `q_list=[25.0, 40.0, 55.0]`
- Effort bounds: `[0, 200]`

### Train and Evaluate

- Train once across all q’s and evaluate each q:

```
python3 run/run_two_players.py --method ppo --episodes 100000
```

- Train only on a specific q (and evaluate that q):

```
python3 run/run_two_players.py --method ppo --q 40 --episodes 100000
```

- Closed‑form baseline rows (uses denominator 4):

```
python3 run/run_two_players.py --method gradient --q 40
```

### Outputs

- CSV: `results/one_stage_two_players.csv` (standardized header via `utils.logger.save_standardized_result`).
- Plot: `results/one_stage_two_players.png` with learning curve and e*(q) overlays.

## Contact

For questions or contributions: `fjiang4@student.gsu.edu`.
