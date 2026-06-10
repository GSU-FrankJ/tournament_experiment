# Tournament Experiment

Game-theoretic tournament experiments with PPO reinforcement learning. The active track verifies whether a PPO agent can learn the theoretical Nash equilibrium effort in two-player symmetric tournaments.

## New Agent Quickstart

**For AI agents exploring this codebase:** Each directory contains a `README.md` with purpose, key files, and usage patterns. Start here:

| Directory | Purpose | Key Entry Point |
|-----------|---------|-----------------|
| [`run/`](run/README.md) | Experiment runners | `python run/run_two_players.py --method ppo --q 40` |
| [`agents/`](agents/README.md) | Learning algorithms | `PPOTwoPlayersBandit`, `MCFDConfig` |
| [`envs/`](envs/README.md) | Game environments | `TwoPlayersEnv` |
| [`config/`](config/README.md) | Experiment configs | `one_stage_two_players.py` |
| [`utils/`](utils/README.md) | Core utilities | `prob.py`, `theory.py`, `eval.py` |
| [`tools/`](tools/README.md) | Analysis & diagnostics | `plot_convergence.py` |
| [`results/`](results/README.md) | Experiment outputs | JSON convergence data, CSV results |
| [`docs/`](docs/README.md) | Documentation | Guides, technical docs, archive |
| [`paper/generator/`](paper/generator/README.md) | Paper generation | `python -m paper.generator make_all` |

**Quick commands:**
```bash
# Run PPO experiment
python run/run_two_players.py --method ppo --q 40 --episodes 2048000 --seed 42

# Run gradient baseline
python run/run_two_players.py --method gradient --q 40

# Generate convergence plots
python tools/plot_convergence_detailed.py

# Generate paper artifacts
python -m paper.generator make_all
```

## Project Structure

```
tournament_experiment/
├── agents/              # Learning algorithms (PPO, MC-FD gradient solver)
├── config/              # Experiment configurations
├── docs/                # Documentation (guides, technical, tasks)
│   ├── STATE.md         # Project-level state tracker
│   ├── guides/          # User guides (PPO defaults, plotting, etc.)
│   ├── technical/       # Implementation docs, audit reports
│   └── tasks/           # Multi-phase task pipeline (per-task STATE.md + phases)
├── envs/                # Game environments
├── paper/               # Paper generation & outputs
│   ├── generator/       # Python package (figure/table pipeline)
│   ├── figures/         # Generated figures
│   ├── tables/          # Generated tables
│   └── data/            # Generated data
├── results/             # Experiment outputs
│   ├── two_players/     # Symmetric 2-player (convergence/, logs/, summary.csv)
│   ├── three_players/   # 3-player (convergence/, logs/)
│   ├── different_cost/  # Asymmetric cost (convergence/, logs/, summary.csv)
│   ├── different_ability/ # Different ability (convergence/, logs/, summary.csv)
│   ├── ablation/        # Ablation studies (exploit_params/, mechanism/)
│   └── plots/           # Diagnostic plots
├── run/                 # Experiment entry points
├── tools/               # Analysis and diagnostic scripts
└── utils/               # Core utility modules
```

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
- self‑play sampling: pure self-play; both players always act from the current learner policy, storing both P1 and P2 transitions every step
- state encoding (3‑D): `[q/60, k/1e‑3, (w_h−w_l)/10]`
- Assumptions documented for this track:
  - Training uses sampled outcomes ONLY: every env step draws fresh uniform noise, ranks the realized outputs, and pays w_H/w_L minus cost. Closed-form win-probability / expected-utility / e* objects are evaluation- and baseline-only and never enter training rewards or policy updates.
  - Evaluation converts the learned Beta policy to actions via the distribution mean; we do **not** switch to the mode even when α,β>1.

### Config (defaults)

- Prizes: `w_H=6.5`, `w_L=3.0`
- Cost: `k=0.0004` (fixed)
- Noise list: `q_list=[35.0, 40.0, 55.0]` (q=25 replaced by q=35; q=25 violates participation constraint)
- Effort bounds: `[0, 200]`

### Train and Evaluate

```
python3 run/run_two_players.py --method ppo --q 40 --episodes 131072
```

Tips:
- `--q` specifies the noise parameter (required for PPO).
- `--episodes` is the total number of environment steps (bandit: one step per episode). Set it to a multiple of `steps_per_update=16384` for clean updates (e.g., 65536, 131072, 262144).

Closed‑form baseline (denominator 4):

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
