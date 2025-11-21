#!/bin/bash
#
# Bash script to run Monte Carlo + Finite Difference (MC-FD) solver
# for two-player effort competition experiments
#
# Usage examples:
#   ./run_mcfd.sh                    # Run with defaults
#   ./run_mcfd.sh --sigma1 15 --sigma2 20
#   ./run_mcfd.sh --num-samples 128 --max-iters 1000
#

# Default parameters
METHOD="mcfd"
# Note: MC-FD uses Gaussian noise (σ), not uniform noise (q), so --q is not needed
SIGMA1=20.0
SIGMA2=20.0
DELTA=1.0
ETA=0.1
NUM_SAMPLES=64
MAX_ITERS=500
TOL=1e-3
EFFORT_MIN=0.0
EFFORT_MAX=100.0
SEED=42

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sigma1)
            SIGMA1="$2"
            shift 2
            ;;
        --sigma2)
            SIGMA2="$2"
            shift 2
            ;;
        --delta)
            DELTA="$2"
            shift 2
            ;;
        --eta)
            ETA="$2"
            shift 2
            ;;
        --num-samples|--N)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --max-iters)
            MAX_ITERS="$2"
            shift 2
            ;;
        --tol)
            TOL="$2"
            shift 2
            ;;
        --effort-min)
            EFFORT_MIN="$2"
            shift 2
            ;;
        --effort-max)
            EFFORT_MAX="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "MC-FD Solver Options:"
            echo "  Note: MC-FD uses Gaussian noise (σ), not uniform noise (q)"
            echo "  --sigma1 VALUE          Player 1 noise std dev (15, 20, or 25) (default: 20.0)"
            echo "  --sigma2 VALUE          Player 2 noise std dev (15, 20, or 25) (default: 20.0)"
            echo "  --delta VALUE           Finite-difference perturbation (default: 1.0)"
            echo "  --eta VALUE             Learning rate (default: 0.1)"
            echo "  --num-samples VALUE     Monte Carlo samples N (32, 64, or 128) (default: 64)"
            echo "  --max-iters VALUE       Maximum iterations (default: 500)"
            echo "  --tol VALUE             Convergence tolerance (default: 1e-3)"
            echo "  --effort-min VALUE      Minimum effort bound (default: 0.0)"
            echo "  --effort-max VALUE      Maximum effort bound (default: 100.0)"
            echo "  --seed VALUE            Random seed (default: 42)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --sigma1 15 --sigma2 20 --num-samples 128"
            echo "  $0 --effort-max 200 --max-iters 1000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=========================================="
echo "MC-FD Solver Configuration"
echo "=========================================="
echo "Method:           $METHOD"
echo "Note: Uses Gaussian noise (σ), not uniform noise (q)"
echo "Sigma1 (σ₁):      $SIGMA1"
echo "Sigma2 (σ₂):      $SIGMA2"
echo "Delta (δ):        $DELTA"
echo "Eta (η):          $ETA"
echo "Num Samples (N):  $NUM_SAMPLES"
echo "Max Iterations:   $MAX_ITERS"
echo "Tolerance:        $TOL"
echo "Effort Range:     [$EFFORT_MIN, $EFFORT_MAX]"
echo "Seed:             $SEED"
echo "=========================================="
echo ""

# Run the experiment
python run/run_two_players.py \
    --method "$METHOD" \
    --mcfd-sigma1 "$SIGMA1" \
    --mcfd-sigma2 "$SIGMA2" \
    --mcfd-delta "$DELTA" \
    --mcfd-eta "$ETA" \
    --mcfd-num-samples "$NUM_SAMPLES" \
    --mcfd-max-iters "$MAX_ITERS" \
    --mcfd-tol "$TOL" \
    --mcfd-effort-min "$EFFORT_MIN" \
    --mcfd-effort-max "$EFFORT_MAX" \
    --mcfd-seed "$SEED"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ MC-FD experiment completed successfully!"
    echo "Results saved to: results/one_stage_two_players.csv"
else
    echo ""
    echo "❌ MC-FD experiment failed!"
    exit 1
fi

