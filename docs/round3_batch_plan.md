# Round 3 Batch Plan

## 1. Three-Player (10 runs)

### Configuration

```bash
python run/run_three_players.py \
  --method ppo --q Q --seed SEED \
  --theory-align-v2 \
  --override-conc-ramp-warmup 200 \
  --min-updates 300 \
  --output-tag round3 \
  --episodes 6144000
```

### Matrix

| q | seeds | n_runs |
|---|-------|--------|
| 35 | 42, 43, 44, 45, 46 | 5 |
| 55 | 42, 43, 44, 45, 46 | 5 |
| **total** | | **10** |

### Fixes included
- Concentration ramp logic (ported from 2P)
- Streak reset fix (line 1119: only reset on eval fail)
- Alpha/beta logging
- Forced final exploit eval
- Output tag for filename isolation

### Output files
`results/three_players/convergence/ppo_3p_q{Q}_seed{S}_round3_baseline_convergence.json`

## 2. Different-Cost / Different-Ability

**Skipped this round.** Per `docs/dc_da_streak_check.md`:
- dc/da do NOT have the streak reset bug
- dc/da do NOT have concentration ramp ported yet
- Their convergence issues (if any) need separate diagnosis

## 3. Wall time estimate

- Per-run estimate: ~85 min for 305 updates (from 4A). With streak fix, most runs should stop at 300-400 updates.
- Available GPUs: 8
- 10 runs / 8 GPUs = 2 waves (8 + 2)
- **Total wall time: ~3 hours** (wave 1: 8 runs parallel ~2 hrs, wave 2: 2 runs ~2 hrs)

## 4. Launch script

```bash
#!/bin/bash
# round3_3p_batch.sh

for q in 35 55; do
  for seed in 42 43 44 45 46; do
    # Find a free GPU (round-robin)
    gpu_id=$(( (seed - 42 + (q == 55 ? 5 : 0)) % 8 ))
    echo "Launching q=$q seed=$seed on GPU $gpu_id"
    tmux new-session -d -s "3p_r3_q${q}_s${seed}" \
      "CUDA_VISIBLE_DEVICES=$gpu_id python run/run_three_players.py \
        --method ppo --q $q --seed $seed \
        --theory-align-v2 \
        --override-conc-ramp-warmup 200 \
        --min-updates 300 \
        --output-tag round3 \
        --episodes 6144000 \
        2>&1 | tee results/round2_conc_fix/log_3p_r3_q${q}_s${seed}.txt"
  done
done
echo "All 10 runs launched."
```

## 5. Monitoring

```bash
# Progress check: count completed runs
ls results/three_players/convergence/ppo_3p_*_round3_baseline_convergence.json 2>/dev/null | wc -l
# Should reach 10.

# Per-run check:
for f in results/round2_conc_fix/log_3p_r3_q*_s*.txt; do
  echo "$f: $(grep -c 'update=' $f) updates, $(grep 'Final q=' $f)"
done
```

## 6. Evaluation script

```python
import json, glob, statistics

files = sorted(glob.glob("results/three_players/convergence/ppo_3p_*_round3_baseline_convergence.json"))
print(f"Found {len(files)} runs")

for q in [35.0, 55.0]:
    q_files = [f for f in files if f"q{q}" in f]
    gaps, rels, exploits, updates = [], [], [], []
    e_star = 3.5 / (4 * q * 0.001)
    for f in q_files:
        d = json.load(open(f))
        pme = d["policy_mean_effort"][-1]
        gap = abs(pme - e_star)
        gaps.append(gap)
        rels.append(gap / e_star * 100)
        exploit_vals = [v for v in d.get("exploitability", []) if v is not None]
        exploits.append(exploit_vals[-1] if exploit_vals else float('nan'))
        updates.append(d.get("stopped_at_update", len(d.get("policy_mean_effort", []))))
    
    if gaps:
        print(f"\nq={q}: n={len(gaps)}")
        print(f"  gap:     mean={statistics.mean(gaps):.2f} std={statistics.stdev(gaps):.2f}")
        print(f"  rel%:    mean={statistics.mean(rels):.1f}% std={statistics.stdev(rels):.1f}%")
        print(f"  exploit: mean={statistics.mean(exploits):.4f}")
        print(f"  updates: mean={statistics.mean(updates):.0f}")
        print(f"  per-seed: {[f'{g:.2f}' for g in gaps]}")
```

## 7. Pass criteria

| metric | threshold |
|--------|-----------|
| stop_reason | exploitability (all 10 runs) |
| mean gap/e* | < 6% per q |
| max gap/e* | < 15% per q |
| exploitability at stop | < 0.03 (all runs) |
| final_br_effort not null | all 10 runs |
| alpha_mean/beta_mean present | all 10 runs |
