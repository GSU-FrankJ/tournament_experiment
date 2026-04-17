#!/bin/bash
# Round 4 Wave 2: da q=55 seeds 43-46
# Launch AFTER wave 1 dc q=55 runs complete (GPU 2-5 free)

for i in 1 2 3 4; do
  seed=$((42 + i))
  gpu=$((i - 1))
  echo "Launching da q=55 seed=$seed on GPU $gpu"
  tmux new-session -d -s "r4_da_q55_s${seed}" \
    "CUDA_VISIBLE_DEVICES=$gpu python run/run_different_ability.py \
      --method ppo --q 55 --seed $seed \
      --exploit-eps 0.03 --min-updates 1000 \
      --ablation-name r4_h1_long --episodes 6144000 \
      2>&1 | tee results/round2_conc_fix/log_r4_da_h1_long_q55_s${seed}.txt"
done
echo "Wave 2: 4 da q=55 runs launched"
