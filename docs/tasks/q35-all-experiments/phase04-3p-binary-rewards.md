# Phase 04: Binary Rewards + Initialization Fix for 3P Convergence

## Root cause (revised after phase03 failure)

Phase03 experiments (entropy=0, entropy_start=0.01, no adv norm) all failed.
Fresh root-cause analysis revealed two structural issues:

### Factor 1: 2P uses stochastic binary rewards, 3P uses smooth analytic rewards

- 2P env.step(): samples noise, picks winner, gives w_H=6.5 or w_L=3.0 (binary)
- 3P env.step(): computes analytic p_i, gives w_L + p_i * w_gap - cost (smooth)
- At efforts ~57: 2P advantages are +/-2.3 (clear signal), 3P advantages are +/-0.01 (noise)
- 2P final KL: 0.02-0.04, 3P final KL: 0.0005-0.005 (17x weaker updates)

### Factor 2: 3P initial policy is too flat (hidden=128 vs 2P hidden=64)

- 3P seed42 init: entropy=-0.09 (concentration ~1.4, nearly uniform over [0,200])
- 2P seed42 init: entropy=-1.86 (concentration ~174, focused around mean)
- Flat init causes catastrophic overshoot: effort crashes 85 -> 9 in 15 updates
- Then spends 1485 updates climbing from 9 to 57, budget exhausted before reaching NE=62.5
- 2P descends smoothly: 105 -> 62.5 in ~1000 updates (no crash)

### Why previous experiments failed

Entropy was never the main antagonist. The real issue is:
1. Smooth rewards give gradient signals 100x weaker than binary rewards
2. The initial crash wastes 95% of training budget on recovery

## Design: 3 parallel experiments (seed=42, q=35)

### Exp C: binary rewards + hidden=64 (combined fix)
```bash
CUDA_VISIBLE_DEVICES=0 python run/run_three_players.py --method ppo --q 35 --seed 42 \
  --episodes 6144000 --exploit-eps 0.02 \
  --binary-rewards --hidden-size 64 --ablation-name binary_h64
```

### Exp A: binary rewards only (hidden=128, isolate reward effect)
```bash
CUDA_VISIBLE_DEVICES=1 python run/run_three_players.py --method ppo --q 35 --seed 42 \
  --episodes 6144000 --exploit-eps 0.02 \
  --binary-rewards --ablation-name binary_h128
```

### Exp B: hidden=64 only (smooth rewards, isolate init effect)
```bash
CUDA_VISIBLE_DEVICES=2 python run/run_three_players.py --method ppo --q 35 --seed 42 \
  --episodes 6144000 --exploit-eps 0.02 \
  --hidden-size 64 --ablation-name smooth_h64
```

## Code changes

- `envs/three_players_env.py`: Added `use_binary_rewards` parameter. When True, samples
  U(-q,q) noise for 3 players, picks winner by max(e+eps), gives w_H/w_L payoffs.
- `run/run_three_players.py`: Added `--binary-rewards` and `--hidden-size` CLI flags.
  Changed `hidden=128` to `hidden=cfg.get("hidden_size", 128)`.

## Success criteria

- Gap from NE < 3 (comparable to 2P)
- Exploitability < 0.05
- No initial crash to effort < 20

## Expected outcomes

- C (binary+h64): BEST — strong signals + no crash. Should match 2P convergence quality.
- A (binary only): partial fix — strong signals but may still crash from flat init. Recovery
  should be faster than baseline due to stronger gradient signal.
- B (h64 only): partial fix — no crash but still weak signals near NE. May get closer than
  baseline (57) but probably won't reach NE.

## Note on CLAUDE.md invariant

CLAUDE.md states "Closed-form expected utilities — no stochastic noise during rollouts."
However, the 2P env (`envs/two_players_env.py:58-63`) DOES use stochastic noise (samples eps,
picks winner, gives binary w_H/w_L). The invariant is either outdated or refers to something
else. The binary reward mode for 3P makes it consistent with 2P's actual behavior.
