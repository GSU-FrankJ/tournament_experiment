# V2 Hyperparameter Retrain — Continuation Summary

Paste this to Claude to resume the work.

---

## What Was Done

### Goal
Minimize theory-experiment gap across all 4 experiments (Two-Player, Three-Player, Het. Cost, Het. Ability) by tuning hyperparameters. The exploitability-based convergence stopping mechanism is kept — no closed-form e* in stopping criterion.

### Config Changes Applied (all 4 configs)

| Parameter | v1 (Old) | v2 (New) |
|-----------|----------|----------|
| `max_updates` | 500 | 1500 |
| `episodes` | 2,048,000 | 6,144,000 |
| `entropy_coef_end` | 0.015 | 0.005 |
| `exploit_eps` | 0.03 | 0.05 |
| `M` (MC samples) | 8192 | 16384 |
| `kl_clip_factor_up` | 1.5 (default) | 1.2 |
| `kl_clip_factor_down` | 0.7 (default) | 0.8 |
| `kl_lr_factor_up` | 1.5 (default) | 1.2 |
| `kl_lr_factor_down` | 0.7 (default) | 0.8 |

`different_cost` and `different_ability` configs also got a new `convergence.exploit` block added (was previously missing).

### Modified Files
- `config/one_stage_two_players.py` — updated hyperparams + KL factors
- `config/one_stage_three_players.py` — updated hyperparams + KL factors
- `config/one_stage_different_cost.py` — updated hyperparams + KL factors + added exploit block
- `config/one_stage_different_ability.py` — updated hyperparams + KL factors + added exploit block
- `run/run_different_cost.py` — `exploit_eps`/`exploit_M` now resolve from config (matching `run_two_players.py` pattern)
- `run/run_different_ability.py` — same as above

### New Files
- `run/retrain_all_v2.sh` — retrain all 4 experiments × 5 seeds with `--ablation-name baseline_v2`
- `run/compare_v1_v2.py` — side-by-side v1 vs v2 comparison, saves `results/comparison_v1_v2.csv`

### v1 vs v2 files do NOT overwrite each other
v2 files include `baseline_v2` in filename (e.g., `ppo_q25.0_seed42_baseline_v2_convergence.json`), so old results are preserved.

---

## Current Retraining Status (as of 2026-02-27)

| Experiment | v2 Runs Completed | Missing Seeds | Status |
|------------|-------------------|---------------|--------|
| Three-Player | 15/15 | — | DONE |
| Two-Player | 9/15 | seed 42, seed 1024 (all 3 q values each) | PARTIAL |
| Different Cost | 0/15 | all | NOT STARTED |
| Different Ability | 0/15 | all | NOT STARTED |

### Preliminary v2 Results

**Three-Player (complete, 5 seeds):**
| q | v1 Rel% | v2 Rel% | Delta |
|---|---------|---------|-------|
| 25 | 26.4% | 24.8% | -1.5% |
| 40 | 10.3% | 9.1% | -1.2% |
| 55 | 8.2% | 5.8% | -2.5% |

Improved but still high at q=25. All runs hit `max_updates=1500` without exploitability-based convergence.

**Two-Player (partial, 3 seeds):**
Example q=25 seed123: final effort ~74.6 vs theoretical 87.5 (gap=12.9, ~14.7%). Also hit max_updates.

---

## What Needs to Be Done

### 1. Complete Missing v2 Runs

```bash
# Two-Player: missing seeds 42 and 1024
python run/run_two_players.py --method ppo --rollout-mode selfplay --seed 42 --ablation-name baseline_v2 --enable-convergence-eval --cheap-gate-profile relaxed
python run/run_two_players.py --method ppo --rollout-mode selfplay --seed 1024 --ablation-name baseline_v2 --enable-convergence-eval --cheap-gate-profile relaxed

# Different Cost: all 5 seeds
for seed in 42 123 456 789 1024; do
  python run/run_different_cost.py --method ppo --seed $seed --ablation-name baseline_v2 --enable-convergence-eval --cheap-gate-profile relaxed
done

# Different Ability: all 5 seeds
for seed in 42 123 456 789 1024; do
  python run/run_different_ability.py --method ppo --seed $seed --ablation-name baseline_v2 --enable-convergence-eval --cheap-gate-profile relaxed
done
```

### 2. Compare Results
```bash
python run/compare_v1_v2.py
```

### 3. If q=25 Still Has >15% Error
Consider:
- Increasing `max_updates` further (e.g., 2500)
- Lowering `entropy_coef_end` further (e.g., 0.001)
- Adjusting learning rate schedule (`lr_end`)

### 4. KL Adaptive Factors Limitation
`kl_clip_factor_up/down` and `kl_lr_factor_up/down` are only read by `run_two_players.py` (lines 788-791). The other 3 scripts do NOT have KL adaptive factor logic in their training loops. To get the benefit, that code would need to be ported.

### 5. Regenerate Paper Artifacts
```bash
python -m paper.generator make_all
```

---

## Key File Paths
- Configs: `config/one_stage_{two_players,three_players,different_cost,different_ability}.py`
- Run scripts: `run/run_{two_players,three_players,different_cost,different_ability}.py`
- Retrain script: `run/retrain_all_v2.sh`
- Comparison: `run/compare_v1_v2.py` → `results/comparison_v1_v2.csv`
- v2 results: `results/{experiment}/convergence/*baseline_v2*`
