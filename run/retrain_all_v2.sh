#!/usr/bin/env bash
# Retrain all experiments with v2 hyperparameters (3x budget, softened KL, etc.)
# Config changes are already baked into config/*.py files.
# This script just sweeps seeds x q_values for all 4 experiments.
set -euo pipefail

SEEDS="42 123 456 789 1024"
TAG="baseline_v2"

echo "=== Retrain All Experiments (v2 hyperparameters) ==="
echo "Seeds: $SEEDS"
echo "Ablation tag: $TAG"
echo ""

# ------------------------------------------------------------------
# Two-player (each seed runs all q_list internally)
# ------------------------------------------------------------------
echo "[1/4] Two-Player experiments..."
for seed in $SEEDS; do
  python run/run_two_players.py --method ppo --rollout-mode selfplay \
    --seed "$seed" --ablation-name "$TAG" \
    --enable-convergence-eval --cheap-gate-profile relaxed &
done
wait
echo "  Two-Player done."

# ------------------------------------------------------------------
# Three-player
# ------------------------------------------------------------------
echo "[2/4] Three-Player experiments..."
for seed in $SEEDS; do
  python run/run_three_players.py --method ppo \
    --seed "$seed" --ablation-name "$TAG" \
    --enable-convergence-eval --cheap-gate-profile relaxed &
done
wait
echo "  Three-Player done."

# ------------------------------------------------------------------
# Different cost
# ------------------------------------------------------------------
echo "[3/4] Different Cost experiments..."
for seed in $SEEDS; do
  python run/run_different_cost.py --method ppo \
    --seed "$seed" --ablation-name "$TAG" \
    --enable-convergence-eval --cheap-gate-profile relaxed &
done
wait
echo "  Different Cost done."

# ------------------------------------------------------------------
# Different ability
# ------------------------------------------------------------------
echo "[4/4] Different Ability experiments..."
for seed in $SEEDS; do
  python run/run_different_ability.py --method ppo \
    --seed "$seed" --ablation-name "$TAG" \
    --enable-convergence-eval --cheap-gate-profile relaxed &
done
wait
echo "  Different Ability done."

echo ""
echo "=== All v2 retraining complete. ==="
echo "Run 'python run/compare_v1_v2.py' to compare results."
