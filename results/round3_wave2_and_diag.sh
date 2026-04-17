#!/bin/bash
# Round 3 Wave 2 + dc/da diagnostic
# Run AFTER wave 1 completes

set -e

# q=55: seeds 45-46 on GPU 0-1
for i in 3 4; do
  seed=$((42 + i))
  gpu=$((i - 3))
  echo "Launching 3P q=55 seed=$seed on GPU $gpu"
  tmux new-session -d -s "3p_r3_q55_s${seed}" \
    "CUDA_VISIBLE_DEVICES=$gpu python run/run_three_players.py \
      --method ppo --q 55 --seed $seed \
      --theory-align-v2 \
      --override-conc-ramp-warmup 200 \
      --min-updates 300 \
      --output-tag round3 \
      --episodes 6144000 \
      2>&1 | tee results/round2_conc_fix/log_3p_r3_q55_s${seed}.txt"
done

# dc diagnostic: q=35 seed=42 on GPU 6
echo "Launching dc diagnostic q=35 seed=42 on GPU 6"
tmux new-session -d -s "dc_diag" \
  "CUDA_VISIBLE_DEVICES=6 python run/run_different_cost.py \
    --method ppo --q 35 --seed 42 \
    --ablation-name dc_diag \
    --episodes 6144000 \
    2>&1 | tee results/round2_conc_fix/log_dc_diag_q35_s42.txt"

# da diagnostic: q=35 seed=42 on GPU 7
echo "Launching da diagnostic q=35 seed=42 on GPU 7"
tmux new-session -d -s "da_diag" \
  "CUDA_VISIBLE_DEVICES=7 python run/run_different_ability.py \
    --method ppo --q 35 --seed 42 \
    --ablation-name da_diag \
    --episodes 6144000 \
    2>&1 | tee results/round2_conc_fix/log_da_diag_q35_s42.txt"

echo "Wave 2 (2 runs) + dc/da diag (2 runs) launched on GPU 0-1, 6-7"
