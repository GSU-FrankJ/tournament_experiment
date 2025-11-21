# MC-FD Parameter Sweep Scripts

## Overview

Two scripts to run a comprehensive parameter sweep for MC-FD solver:

**Sweep Parameters:**
- **Sigma (σ)**: 15 (fixed)
- **Delta (δ)**: 0.5, 0.75, 1.0 (3 values)
- **Eta (η)**: 0.02, 0.06, 0.10 (3 values)
- **Num Samples (N)**: 32, 64, 128 (3 values)

**Total Combinations**: 1 × 3 × 3 × 3 = **27 runs**

## Usage

### Option 1: Python Script (Recommended)
```bash
python3 run_mcfd_sweep.py
```

**Features:**
- Better progress tracking
- Real-time elapsed time and ETA
- Automatic summary generation
- Cleaner output

### Option 2: Bash Script
```bash
./run_mcfd_sweep.sh
```

**Features:**
- Simple bash implementation
- Logs all output to file
- Works without Python dependencies

## Output

### Results File
- **Location**: `results/one_stage_two_players.csv`
- **Format**: CSV with all MC-FD results appended
- **Columns**: Includes all standard columns plus MC-FD specific fields:
  - `mcfd_final_e1`, `mcfd_final_e2`
  - `mcfd_iterations`
  - `mcfd_sigma1`, `mcfd_sigma2`
  - `mcfd_delta`, `mcfd_eta`, `mcfd_samples`
  - `mcfd_tol`, `mcfd_effort_min`, `mcfd_effort_max`

### Log File
- **Location**: `results/logs/mcfd_sweep_YYYYMMDD_HHMMSS.log`
- **Contains**: Full output from each run, including errors

## Customization

To modify sweep parameters, edit the script:

### Python (`run_mcfd_sweep.py`)
```python
SIGMA = 15.0
DELTAS = [0.5, 0.75, 1.0]
ETAS = [0.02, 0.06, 0.10]
NUM_SAMPLES = [32, 64, 128]
```

### Bash (`run_mcfd_sweep.sh`)
```bash
SIGMA=15.0
DELTAS=(0.5 0.75 1.0)
ETAS=(0.02 0.06 0.10)
NUM_SAMPLES=(32 64 128)
```

## Fixed Parameters

These are set in both scripts (can be modified):
- `MAX_ITERS = 500`
- `TOL = 1e-3`
- `EFFORT_MIN = 0.0`
- `EFFORT_MAX = 100.0`
- `SEED = 42`

## Expected Runtime

- **Per run**: ~5-30 seconds (depends on num_samples and convergence)
- **Total (27 runs)**: ~2-15 minutes

## Analysis After Sweep

### Quick Analysis
```bash
python3 analyze_mcfd_result.py <row_number>
```

### Full CSV Analysis
```python
import pandas as pd

df = pd.read_csv('results/one_stage_two_players.csv')
mcfd = df[df['Model_training'] == 'mcfd']

# Best performing runs
print(mcfd.nlargest(5, 'final_stage2_effort'))

# Parameter sensitivity
print(mcfd.groupby('mcfd_delta')['final_stage2_effort'].mean())
print(mcfd.groupby('mcfd_eta')['final_stage2_effort'].mean())
print(mcfd.groupby('mcfd_samples')['final_stage2_effort'].mean())
```

## Troubleshooting

### If a run fails:
- Check the sweep log file for error messages
- The script continues with remaining combinations
- Failed runs are logged but don't stop the sweep

### To stop the sweep:
- Press `Ctrl+C`
- The script will exit gracefully
- Completed runs are already saved

### To resume:
- The script will append to the CSV
- You may get duplicate runs if you restart
- Consider filtering duplicates in post-processing

