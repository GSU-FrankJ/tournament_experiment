#!/bin/bash
# Phase 03 Round 1: Entropy end-value screen on q=40
# 4 variants × 3 seeds × 1500 updates (full training)
# Each GPU runs 3 seeds serially; 4 GPUs run in parallel
#
# Output files use ablation-name prefix "ent_*"
# Cleanup: rm results/two_players/convergence/ppo_q40.0_seed*_ent_*

set -e
cd "$(dirname "$0")/.."

PYTHON=/home/fjiang4/miniconda3/envs/mio-sd/bin/python
EPISODES=6144000  # 1500 updates × 4096 steps/update
Q=40
METHOD=ppo
SEEDS=(42 123 456)

echo "=== Phase 03 Round 1: q=40 Entropy End-Value Screen ==="
echo "Episodes per run: $EPISODES (1500 updates)"
echo "Seeds: ${SEEDS[*]}"
echo ""

# GPU 0: Baseline entropy (entropy_coef_end=0.005, current default)
run_gpu0() {
    for s in "${SEEDS[@]}"; do
        echo "[GPU 0] ent_baseline (0.005): seed $s..."
        CUDA_VISIBLE_DEVICES=0 $PYTHON run/run_two_players.py \
            --method $METHOD --q $Q --episodes $EPISODES --seed $s \
            --ablation-name ent_baseline
    done
    echo "[GPU 0] Done."
}

# GPU 1: Low entropy (entropy_coef_end=0.001)
run_gpu1() {
    for s in "${SEEDS[@]}"; do
        echo "[GPU 1] ent_001 (0.001): seed $s..."
        CUDA_VISIBLE_DEVICES=1 $PYTHON run/run_two_players.py \
            --method $METHOD --q $Q --episodes $EPISODES --seed $s \
            --override-entropy-end 0.001 \
            --ablation-name ent_001
    done
    echo "[GPU 1] Done."
}

# GPU 2: Very low entropy (entropy_coef_end=0.0001)
run_gpu2() {
    for s in "${SEEDS[@]}"; do
        echo "[GPU 2] ent_0001 (0.0001): seed $s..."
        CUDA_VISIBLE_DEVICES=2 $PYTHON run/run_two_players.py \
            --method $METHOD --q $Q --episodes $EPISODES --seed $s \
            --override-entropy-end 0.0001 \
            --ablation-name ent_0001
    done
    echo "[GPU 2] Done."
}

# GPU 3: Zero entropy (entropy_coef_end=0.0)
run_gpu3() {
    for s in "${SEEDS[@]}"; do
        echo "[GPU 3] ent_zero (0.0): seed $s..."
        CUDA_VISIBLE_DEVICES=3 $PYTHON run/run_two_players.py \
            --method $METHOD --q $Q --episodes $EPISODES --seed $s \
            --override-entropy-end 0.0 \
            --ablation-name ent_zero
    done
    echo "[GPU 3] Done."
}

# Launch all 4 GPUs in parallel
run_gpu0 &
run_gpu1 &
run_gpu2 &
run_gpu3 &

echo "4 GPU jobs launched (3 seeds each, serial per GPU). Waiting..."
wait
echo ""
echo "=== Phase 03 Round 1 complete ==="
echo "Results: ls results/two_players/convergence/ppo_q40.0_seed*_ent_*"
echo "Cleanup: rm results/two_players/convergence/ppo_q40.0_seed*_ent_*"
