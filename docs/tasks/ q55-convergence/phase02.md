# Phase 02: Standard Mode Multi-Seed Validation + Precision Tuning

## Background
Phase 01 Exp A showed standard ActorCritic (no theory_align_v2) + entropy regularization
converges for q=55 seed=42: effort=34.53, gap=5.25, 269 updates. This is the first time
standard mode has been tested for q=55 — all prior baselines used theory_align_v2.

Gap=5.25 is worse than conc_max=1000's 0.54, but the mechanism is fundamentally different:
concentration stays low (~92) via entropy, no hard cap needed.

## Objective
1. Validate standard mode across multiple seeds (is Exp A's success reproducible?)
2. Improve precision (reduce gap from 5.25 toward <2)

## Experiments

### Step 1: Multi-seed validation (5 seeds)
Same config as Exp A, different seeds. Determines convergence rate.

```bash
for SEED in 123 456 789 1024 11; do
  tmux new-session -d -s q55_ph02_s${SEED} \
    "python run/run_two_players.py --method ppo --q 55 --seed ${SEED} \
     --episodes 6144000 --no-theory-align-v2 \
     --ablation-name no_theory_align \
     2>&1 | tee logs/q55_seed${SEED}_no_theory_align.log"
done
```

**Pass criterion**: ≥3/5 seeds converge (exploit < 0.05 streak=5)

### Step 2: Precision tuning (conditional, only if Step 1 passes)
If multi-seed validates, try to close the gap. Options:

**2a. Slower entropy decay** — keep entropy higher for longer so effort can reach 39.77
```bash
tmux new-session -d -s q55_ph02_slow_ent \
  "python run/run_two_players.py --method ppo --q 55 --seed 42 \
   --episodes 6144000 --no-theory-align-v2 \
   --override-entropy-end 0.015 \
   --ablation-name no_tv2_ent015 \
   2>&1 | tee logs/q55_seed42_no_tv2_ent015.log"
```

**2b. More episodes** — give the existing config more budget to refine
```bash
tmux new-session -d -s q55_ph02_long \
  "python run/run_two_players.py --method ppo --q 55 --seed 42 \
   --episodes 12288000 --no-theory-align-v2 \
   --ablation-name no_tv2_long \
   2>&1 | tee logs/q55_seed42_no_tv2_long.log"
```

## Interpretation

| Step 1 result | Step 2 action |
|---------------|---------------|
| ≥3/5 converge, gap < 3 | Standard mode is the fix; proceed to cross-q validation |
| ≥3/5 converge, gap 3-8 | Run Step 2 precision tuning |
| <3/5 converge | Standard mode unreliable; fall back to conc_max approach |

## Files to modify
- None (CLI flags only)

## Verification
- Compare each seed vs theory_align_v2 baseline (same seed)
- Check concentration trajectories stay bounded without explicit cap
- Check q=35/40 regression (after confirming q=55 works)
