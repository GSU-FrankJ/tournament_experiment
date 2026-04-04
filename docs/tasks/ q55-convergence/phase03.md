# Phase 03: Cross-q Validation (q=35, q=40 regression check)

## Precondition
Phase 02 confirms standard mode + entropy_end=0.002 is stable across seeds for q=55.

## Objective
Verify that standard mode (no theory_align_v2) does not regress q=35 and q=40 convergence.
All prior q=35/40 baselines used theory_align_v2 — we have no data for standard mode.

## Experiments
Run q=35 and q=40 with the same config that works for q=55:
- `--no-theory-align-v2 --override-entropy-end 0.002`
- seed=42 for quick validation, then 5 seeds if it looks good

```bash
for Q in 35 40; do
  tmux new-session -d -s q${Q}_std_test \
    "CUDA_VISIBLE_DEVICES=X python run/run_two_players.py --method ppo --q ${Q} --seed 42 \
     --episodes 131072 --no-theory-align-v2 \
     --override-entropy-end 0.002 \
     --ablation-name no_tv2_ent002 \
     2>&1 | tee logs/q${Q}_seed42_no_tv2_ent002.log"
done
```

## Success criteria
- q=35 and q=40 converge with gap < 2 (comparable to theory_align_v2 baseline)
- Convergence speed within 2x of baseline

## Interpretation
| Result | Next step |
|--------|-----------|
| q=35/40 unaffected | Standard mode is a universal fix — Phase 04 unnecessary |
| q=35/40 slower but converge | Acceptable; document tradeoff |
| q=35/40 fail | Different q values need different config → Phase 04 (adaptive entropy) |
