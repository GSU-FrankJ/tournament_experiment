#!/bin/bash
#
# MC-FD Parameter Sweep Script
#
# Sweeps over:
#   - Sigma (std of performance shock): 15
#   - Delta (finite-difference perturbation): 0.5, 0.75, 1.0
#   - Eta (gradient ascent step size): 0.02, 0.06, 0.10
#   - Num samples (Monte-Carlo batch size): 32, 64, 128
#
# Total combinations: 1 × 3 × 3 × 3 = 27 runs

set -e  # Exit on error

# Configuration
METHOD="mcfd"
SIGMA=15.0
DELTAS=(0.5 0.75 1.0)
ETAS=(0.02 0.06 0.10)
NUM_SAMPLES=(32 64 128)

# Other fixed parameters
MAX_ITERS=500
TOL=1e-3
EFFORT_MIN=0.0
EFFORT_MAX=100.0
SEED=42

# Output file
CSV_PATH="results/one_stage_two_players.csv"
LOG_DIR="results/logs"
SWEEP_LOG="${LOG_DIR}/mcfd_sweep_$(date +%Y%m%d_%H%M%S).log"

# Create directories
mkdir -p "$(dirname "$CSV_PATH")"
mkdir -p "$LOG_DIR"

# Initialize counters
total_runs=$((${#DELTAS[@]} * ${#ETAS[@]} * ${#NUM_SAMPLES[@]}))
current_run=0

echo "=========================================="
echo "MC-FD Parameter Sweep"
echo "=========================================="
echo "Sigma (σ): $SIGMA"
echo "Deltas (δ): ${DELTAS[@]}"
echo "Etas (η): ${ETAS[@]}"
echo "Num Samples (N): ${NUM_SAMPLES[@]}"
echo "Total combinations: $total_runs"
echo "=========================================="
echo ""

# Log start time
start_time=$(date +%s)
echo "[$(date)] Starting sweep..." | tee -a "$SWEEP_LOG"

# Run sweep
for delta in "${DELTAS[@]}"; do
    for eta in "${ETAS[@]}"; do
        for num_samples in "${NUM_SAMPLES[@]}"; do
            current_run=$((current_run + 1))
            
            echo "" | tee -a "$SWEEP_LOG"
            echo "----------------------------------------" | tee -a "$SWEEP_LOG"
            echo "[$current_run/$total_runs] Running:" | tee -a "$SWEEP_LOG"
            echo "  σ = $SIGMA" | tee -a "$SWEEP_LOG"
            echo "  δ = $delta" | tee -a "$SWEEP_LOG"
            echo "  η = $eta" | tee -a "$SWEEP_LOG"
            echo "  N = $num_samples" | tee -a "$SWEEP_LOG"
            echo "----------------------------------------" | tee -a "$SWEEP_LOG"
            
            # Run MC-FD
            python run/run_two_players.py \
                --method "$METHOD" \
                --mcfd-sigma1 "$SIGMA" \
                --mcfd-sigma2 "$SIGMA" \
                --mcfd-delta "$delta" \
                --mcfd-eta "$eta" \
                --mcfd-num-samples "$num_samples" \
                --mcfd-max-iters "$MAX_ITERS" \
                --mcfd-tol "$TOL" \
                --mcfd-effort-min "$EFFORT_MIN" \
                --mcfd-effort-max "$EFFORT_MAX" \
                --mcfd-seed "$SEED" \
                2>&1 | tee -a "$SWEEP_LOG"
            
            exit_code=${PIPESTATUS[0]}
            
            if [ $exit_code -eq 0 ]; then
                echo "✅ Run $current_run/$total_runs completed successfully" | tee -a "$SWEEP_LOG"
            else
                echo "❌ Run $current_run/$total_runs failed with exit code $exit_code" | tee -a "$SWEEP_LOG"
                echo "Continuing with next combination..." | tee -a "$SWEEP_LOG"
            fi
            
            # Small delay to avoid overwhelming the system
            sleep 0.5
        done
    done
done

# Calculate elapsed time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

echo "" | tee -a "$SWEEP_LOG"
echo "==========================================" | tee -a "$SWEEP_LOG"
echo "Sweep Complete!" | tee -a "$SWEEP_LOG"
echo "Total runs: $total_runs" | tee -a "$SWEEP_LOG"
echo "Elapsed time: ${minutes}m ${seconds}s" | tee -a "$SWEEP_LOG"
echo "Results saved to: $CSV_PATH" | tee -a "$SWEEP_LOG"
echo "Sweep log saved to: $SWEEP_LOG" | tee -a "$SWEEP_LOG"
echo "==========================================" | tee -a "$SWEEP_LOG"

# Generate summary
echo ""
echo "Generating summary..."
python3 << EOF
import csv
import pandas as pd

try:
    df = pd.read_csv('$CSV_PATH')
    mcfd_rows = df[df['Model_training'] == 'mcfd']
    
    if len(mcfd_rows) > 0:
        print(f"\n📊 Summary of MC-FD Results:")
        print(f"   Total MC-FD runs: {len(mcfd_rows)}")
        print(f"   Final effort range: {mcfd_rows['final_stage2_effort'].min():.2f} - {mcfd_rows['final_stage2_effort'].max():.2f}")
        print(f"   Average final effort: {mcfd_rows['final_stage2_effort'].mean():.2f}")
        print(f"   Convergence quality distribution:")
        print(mcfd_rows['Convergence_Quality'].value_counts().to_string())
    else:
        print("No MC-FD results found in CSV")
except Exception as e:
    print(f"Could not generate summary: {e}")
EOF

echo ""
echo "✅ Sweep complete! Check results in: $CSV_PATH"

