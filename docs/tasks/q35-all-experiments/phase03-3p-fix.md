# Phase 03: Fix 3-Player Convergence Gap

## Problem

3p q=35 PPO consistently converges to effort ~57 instead of NE=62.5 (gap ~5.4).
All 5 seeds hit max_updates=1500 without converging. Root cause confirmed in phase02:

1. 3p per-transition gradient signal 27% weaker than 2p (rank-order loser cancellation)
2. entropy_coef=0.03 (tuned for 2p) overpowers the weak signal -> policy deconcentrates
3. Low concentration -> weak nabla log pi -> KL->0 -> policy freezes at false equilibrium

Failed experiments (phase02):
- entropy_end sweep (0.0005/0.001/0.003): zero effect (policy frozen before tail phase)
- gradient steps per update reduction: made it worse

## Design: Round 1 (single seed=42, parallel)

Three experiments targeting the root cause from different angles:

### Exp A1: Lower entropy_start/hold (most direct)

**Rationale**: The entropy_end sweep failed because the damage happens during the
hold phase (2/3 of training at entropy=0.03). Lowering start/hold directly addresses this.

**Change**: entropy_start=0.01, entropy_hold=0.01, entropy_end=0.005
(vs baseline: 0.03/0.03/0.005)

```bash
CUDA_VISIBLE_DEVICES=0 python run/run_three_players.py --method ppo --q 35 --seed 42 \
  --episodes 6144000 --exploit-eps 0.02 \
  --override-entropy-start 0.01 --ablation-name ent_start_01
```

### Exp B3: Disable advantage normalization (signal preservation)

**Rationale**: Advantage normalization divides by std, amplifying noise when raw
advantages are small. Analytical gradient is identical for 2p/3p at all effort levels
(verified in phase02). Without normalization, step size is smaller but direction is correct.

**Change**: Skip `(adv - mean) / (std + 1e-8)` normalization.

```bash
CUDA_VISIBLE_DEVICES=1 python run/run_three_players.py --method ppo --q 35 --seed 42 \
  --episodes 6144000 --exploit-eps 0.02 \
  --disable-adv-norm --ablation-name no_adv_norm
```

### Exp A3: Zero entropy (extreme control)

**Rationale**: If entropy is the root cause, removing it entirely should allow convergence.
This is the most aggressive test — if it fails, the root cause analysis needs revision.

**Change**: entropy=0 throughout training (flag already exists).

```bash
CUDA_VISIBLE_DEVICES=2 python run/run_three_players.py --method ppo --q 35 --seed 42 \
  --episodes 6144000 --exploit-eps 0.02 \
  --disable-entropy --ablation-name no_entropy
```

## Code changes

- `agents/ppo_three_players.py`: Added `normalize_advantages` config field (default True),
  conditional normalization in `update()`.
- `run/run_three_players.py`: Added `--override-entropy-start` and `--disable-adv-norm`
  CLI flags with corresponding config propagation.

## Success criteria

- Gap from NE < 3 (comparable to 2p)
- entropy decreases during training (not increases)
- KL stays above 0.001 (policy keeps updating)

## Round 2 plan (based on results)

- If A1 works: try A1 + sweep entropy_start in {0.005, 0.01, 0.02}, then 5-seed validation
- If B3 works: 5-seed validation, check q=40/q=25 too
- If A3 works but A1 doesn't: find minimum entropy that still allows convergence
- If none work: try larger batch (steps_per_update=8192), or combination A1+B3
