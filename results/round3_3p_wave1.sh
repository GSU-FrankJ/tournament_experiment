#!/bin/bash
# Round 3 Wave 1: 3P batch — 5 q=35 seeds + 3 q=55 seeds on GPU 0-7

set -e

# q=35: seeds 42-46 on GPU 0-4
for i in 0 1 2 3 4; do
  seed=$((42 + i))
  echo "Launching 3P q=35 seed=$seed on GPU $i"
  tmux new-session -d -s "3p_r3_q35_s${seed}" \
    "CUDA_VISIBLE_DEVICES=$i python run/run_three_players.py \
      --method ppo --q 35 --seed $seed \
      --theory-align-v2 \
      --override-conc-ramp-warmup 200 \
      --min-updates 300 \
      --output-tag round3 \
      --episodes 6144000 \
      2>&1 | tee results/round2_conc_fix/log_3p_r3_q35_s${seed}.txt"
done

# q=55: seeds 42-44 on GPU 5-7
for i in 0 1 2; do
  seed=$((42 + i))
  gpu=$((5 + i))
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

echo "Wave 1 launched: 8 runs on GPU 0-7"
echo "Monitor: bash results/round3_monitor.sh"
