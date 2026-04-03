# Phase 01: Isolate Entropy vs Concentration Control

## Background Discovery
All baseline PPO runs use `theory_align_v2` by default, which:
- Forces `entropy_coef = 0` (start/hold/end all zeroed)
- Uses MeanConc network (separate mean_head + conc_head)
- Sets `conc_min=1000, conc_scale=10000, conc_max=100000`

The entropy schedule (0.03 → 0.03 → 0.005) from config is **never active** in baseline runs.
The conc_max=1000 fix works by hard-clamping the conc_head output, not via entropy.

## Objective
Determine whether entropy regularization alone can fix q=55, and whether it needs
the MeanConc architecture or works with standard ActorCritic.

## Experiments

### Exp A: Standard mode (no theory_align_v2)
Disable theory_align_v2 entirely. Uses standard ActorCritic (direct α/β heads) with
the entropy schedule from config (0.03 → 0.03 → 0.005).

This isolates: **can entropy regularization + standard network prevent concentration collapse?**

```bash
tmux new-session -d -s q55_ph01_expA \
  "python run/run_two_players.py --method ppo --q 55 --seed 42 \
   --episodes 6144000 --no-theory-align-v2 \
   --ablation-name no_theory_align \
   2>&1 | tee logs/q55_seed42_no_theory_align.log"
```

### Exp B: theory_align_v2 with entropy restored
Keep MeanConc network and conc_max=100000, but re-enable entropy regularization.
`--override-entropy-end` is re-applied AFTER theory_align_v2 zeroing (line 1919-1928),
and auto-sets start=hold=end*2 when end>0.

Using `--override-entropy-end 0.015` gives: start=0.03, hold=0.03, end=0.015.
This closely matches the original config spirit (0.03 → 0.03 → decay).

This isolates: **does adding entropy to MeanConc prevent conc_head from growing too fast?**

```bash
tmux new-session -d -s q55_ph01_expB \
  "python run/run_two_players.py --method ppo --q 55 --seed 42 \
   --episodes 6144000 --theory-align-v2 \
   --override-entropy-end 0.015 \
   --ablation-name tv2_entropy_015 \
   2>&1 | tee logs/q55_seed42_tv2_entropy_015.log"
```

## Interpretation Matrix

| Exp A (standard) | Exp B (MeanConc+entropy) | Conclusion |
|-------------------|--------------------------|------------|
| converge | converge | Entropy sufficient; theory_align_v2 not needed |
| converge | fail | Standard mode + entropy is the fix; MeanConc fights entropy |
| fail | converge | Entropy helps but needs MeanConc architecture |
| fail | fail | Entropy alone insufficient for q=55; concentration cap is necessary |

## Success criteria
- **Pass**: seed=42 converges (exploit < 0.05 streak=5) with gap < 5
- **Partial**: effort trending toward 39.77 but doesn't pass threshold
- **Fail**: effort stuck at ~48-55 same as baseline

## What to log and compare
- Final effort, gap, exploitability (vs baseline: effort=48.81, gap=9.04)
- Concentration trajectory (α+β over updates)
- Entropy trajectory (batch_entropy over updates)
- Exploitability trajectory shape (monotonic vs oscillating)

## Files to modify
- Exp A: none (CLI flags only)
- Exp B: may need patch in `run_two_players.py` to prevent entropy zeroing when CLI override is set

## Next phase
Based on results, proceed to:
- Phase 02 (cross-q validation) if either experiment passes
- Revised Phase 03 (adaptive concentration) if both fail
