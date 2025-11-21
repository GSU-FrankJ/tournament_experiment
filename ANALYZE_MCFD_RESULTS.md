# How to Analyze MC-FD Results

## Your Result (Row 29)

```
3.0,6.5,0.0004,0.0004,none,0.0,nan,mcfd,0.0,60.63175977141099,394.10643851417143,Poor,0,nan
```

## Column-by-Column Analysis

### 1. **Basic Parameters**
- **stage1_weight (w_L)**: `3.0` - Low prize
- **stage2_weight (w_H)**: `6.5` - High prize  
- **k1, k2**: `0.0004` - Cost parameter (quadratic cost: k·e²)
- **information_revelation**: `none` - No information revealed

### 2. **Effort Results**
- **final_stage1_effort**: `0.0` - Stage 1 effort (always 0 for one-stage)
- **final_stage2_effort**: `60.63` - **Average effort from both players**
- **final_weighted_effort**: `394.11` - Weighted effort (w_L·0 + w_H·60.63)

### 3. **Convergence Status**
- **Convergence_Quality**: `Poor` ⚠️
- **episodes**: `0` - MC-FD doesn't use episodes (uses iterations instead)

### 4. **Theoretical Comparison**
- **theoretical_stage2_effort**: `nan` - No benchmark (MC-FD uses Gaussian noise)
- **Gap_from_theoretical**: `nan` - Cannot compare to uniform-noise theory

## What This Result Means

### ✅ **What We Know:**
1. **Final Effort**: Both players converged to an average effort of **~60.63**
2. **Method**: MC-FD with Gaussian noise (σ₁, σ₂)
3. **Convergence**: Marked as "Poor" - may need tuning

### ⚠️ **Issues to Address:**

1. **"Poor" Convergence Quality**
   - The algorithm may not have converged properly
   - Possible causes:
     - Not enough iterations
     - Learning rate too high/low
     - Not enough Monte Carlo samples
     - Tolerance too strict

2. **No Theoretical Benchmark**
   - MC-FD uses Gaussian noise N(0, σ²)
   - Cannot compare to uniform-noise theoretical value
   - This is expected and correct

## How to Improve Results

### Option 1: Increase Iterations
```bash
python run/run_two_players.py \
    --method mcfd \
    --mcfd-max-iters 1000  # Increase from default 500
```

### Option 2: Increase Monte Carlo Samples
```bash
python run/run_two_players.py \
    --method mcfd \
    --mcfd-num-samples 128  # Increase from default 64
```

### Option 3: Adjust Learning Rate
```bash
python run/run_two_players.py \
    --method mcfd \
    --mcfd-eta 0.05  # Try smaller (more stable) or larger (faster)
```

### Option 4: Tighten Tolerance
```bash
python run/run_two_players.py \
    --method mcfd \
    --mcfd-tol 1e-4  # Stricter convergence (default: 1e-3)
```

### Option 5: Combined Optimization
```bash
python run/run_two_players.py \
    --method mcfd \
    --mcfd-sigma1 20 \
    --mcfd-sigma2 20 \
    --mcfd-num-samples 128 \
    --mcfd-max-iters 1000 \
    --mcfd-eta 0.1 \
    --mcfd-tol 1e-4
```

## Expected MC-FD Columns (if saved)

The following columns should be in the CSV (may be in additional columns):
- `mcfd_final_e1`: Player 1's final effort
- `mcfd_final_e2`: Player 2's final effort  
- `mcfd_iterations`: Number of iterations run
- `mcfd_sigma1`: Noise std dev for player 1
- `mcfd_sigma2`: Noise std dev for player 2
- `mcfd_delta`: Finite-difference perturbation
- `mcfd_eta`: Learning rate used
- `mcfd_samples`: Monte Carlo samples per gradient
- `mcfd_tol`: Convergence tolerance

## Interpretation Guide

### Good Result:
- Convergence Quality: "Good" or "Excellent"
- Final effort is reasonable (within effort bounds)
- Both players' efforts are similar (symmetric equilibrium)

### Poor Result (like yours):
- Convergence Quality: "Poor"
- May indicate:
  - Algorithm didn't converge
  - Need more iterations/samples
  - Hyperparameters need tuning

### What to Check:
1. **Effort values**: Are they within bounds [0, 100] or [0, 200]?
2. **Symmetry**: Are e₁ and e₂ similar? (should be for symmetric game)
3. **Iterations**: Did it use all max_iters or converge early?
4. **Stability**: Run multiple times with same seed - do results vary?

## Quick Analysis Command

```bash
python3 analyze_mcfd_result.py 29
```

This will show a detailed breakdown of row 29.


