# Phase 02: Round 2 — Full Validation (all q, unified L)

## Objective
Run the winning config from Phase 01 across 5 seeds × q={35, 45, 55} with the
unified theory bound L=57. Since L is q-independent, q=35 regression is tested
simultaneously (no separate Round 3 needed).

## Prerequisite
Phase 01 must identify a config with q=55 gap < 3.0.

## Steps
1. Determine winning config from Phase 01 results
2. Run 15 experiments: 5 seeds × 3 q values, all with `--effort-range 0 57`
3. Aggregate results: mean, std, per-seed gaps

## Commands
```bash
for q in 35 45 55; do
  for seed in 42 43 44 45 46; do
    tmux new-session -d -s "L57_q${q}_s${seed}" \
      "python run/run_two_players.py --method ppo --q $q --seed $seed \
       --effort-range 0 57 --variant-name L57 --episodes 6144000"
  done
done
```

## Go/No-Go Criteria
- q=45 mean |e-e*| < 2.0
- q=55 mean |e-e*| < 2.0
- q=35 mean |e-e*| < 1.5 (no regression from current 1.0)
- No seed with gap > 5.0 (outlier check)

## Pass → Phase 03
Apply L = ⌈√(W/(2k))⌉ to other experiments:
- 3P: L = 42 (k=0.001)
- Het. Cost: L = 56 (k₁=0.0004)
- Het. Ability: L = 60 (k=0.0005)
