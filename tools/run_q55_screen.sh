#!/bin/bash
# Phase 02 Round 1: Quick hyperparameter screen for q=55
# 4 variants × 2 seeds × 500 updates
# Each GPU runs 2 seeds serially; 4 GPUs run in parallel
#
# Output files use ablation-name prefix "screen_*" to avoid
# polluting the main baseline results. Clean up after analysis:
#   rm results/two_players/convergence/ppo_q55.0_seed*_screen_*

set -e
cd "$(dirname "$0")/.."

# Use mio-sd conda env (has torch + CUDA)
PYTHON=/home/fjiang4/miniconda3/envs/mio-sd/bin/python

EPISODES=2048000  # 500 updates × 4096 steps/update
Q=55
METHOD=ppo

echo "=== Phase 02 Round 1: q=55 Hyperparameter Screen ==="
echo "Episodes per run: $EPISODES (500 updates)"
echo "Output: results/two_players/convergence/ppo_q55.0_seed*_screen_*"
echo ""

# GPU 0: Baseline reproduction
run_gpu0() {
    echo "[GPU 0] Baseline: seed 42..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON run/run_two_players.py \
        --method $METHOD --q $Q --episodes $EPISODES --seed 42 \
        --ablation-name screen_baseline
    echo "[GPU 0] Baseline: seed 456..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON run/run_two_players.py \
        --method $METHOD --q $Q --episodes $EPISODES --seed 456 \
        --ablation-name screen_baseline
    echo "[GPU 0] Done."
}

# GPU 1: High entropy (entropy_coef_end: 0.005 -> 0.015)
run_gpu1() {
    echo "[GPU 1] High entropy: seed 42..."
    CUDA_VISIBLE_DEVICES=1 $PYTHON run/run_two_players.py \
        --method $METHOD --q $Q --episodes $EPISODES --seed 42 \
        --override-entropy-end 0.015 \
        --ablation-name screen_high_entropy
    echo "[GPU 1] High entropy: seed 456..."
    CUDA_VISIBLE_DEVICES=1 $PYTHON run/run_two_players.py \
        --method $METHOD --q $Q --episodes $EPISODES --seed 456 \
        --override-entropy-end 0.015 \
        --ablation-name screen_high_entropy
    echo "[GPU 1] Done."
}

# GPU 2: Low learning rate (lr_end: 2e-4 -> 5e-5)
run_gpu2() {
    echo "[GPU 2] Low LR: seed 42..."
    CUDA_VISIBLE_DEVICES=2 $PYTHON run/run_two_players.py \
        --method $METHOD --q $Q --episodes $EPISODES --seed 42 \
        --override-lr-end 5e-5 \
        --ablation-name screen_low_lr
    echo "[GPU 2] Low LR: seed 456..."
    CUDA_VISIBLE_DEVICES=2 $PYTHON run/run_two_players.py \
        --method $METHOD --q $Q --episodes $EPISODES --seed 456 \
        --override-lr-end 5e-5 \
        --ablation-name screen_low_lr
    echo "[GPU 2] Done."
}

# GPU 3: High entropy + Low LR
run_gpu3() {
    echo "[GPU 3] High entropy + Low LR: seed 42..."
    CUDA_VISIBLE_DEVICES=3 $PYTHON run/run_two_players.py \
        --method $METHOD --q $Q --episodes $EPISODES --seed 42 \
        --override-entropy-end 0.015 --override-lr-end 5e-5 \
        --ablation-name screen_high_ent_low_lr
    echo "[GPU 3] High entropy + Low LR: seed 456..."
    CUDA_VISIBLE_DEVICES=3 $PYTHON run/run_two_players.py \
        --method $METHOD --q $Q --episodes $EPISODES --seed 456 \
        --override-entropy-end 0.015 --override-lr-end 5e-5 \
        --ablation-name screen_high_ent_low_lr
    echo "[GPU 3] Done."
}

# Launch all 4 GPUs in parallel, each runs seeds serially
run_gpu0 &
run_gpu1 &
run_gpu2 &
run_gpu3 &

echo "4 GPU jobs launched (2 seeds each, serial per GPU). Waiting..."
wait
echo ""
echo "=== All Round 1 jobs complete ==="
echo "Results: ls results/two_players/convergence/ppo_q55.0_seed*_screen_*"
echo "Cleanup: rm results/two_players/convergence/ppo_q55.0_seed*_screen_*"
