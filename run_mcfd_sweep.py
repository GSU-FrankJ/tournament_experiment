#!/usr/bin/env python3
"""
MC-FD parameter sweep helper.

Fixed settings:
    - sigma1 = sigma2 = 20.0
    - effort bounds = [0, 100]
    - seed = 42
    - num_samples in {32, 64, 128}

Sweeps over combinations of:
    - delta (finite-difference step)
    - eta (gradient ascent step size)
    - max_iters (iteration cap)
    - tol (convergence tolerance)

Outputs are appended to results/one_stage_two_players.csv
using the current MC-FD layout (sigma, delta, ... seed).
"""

import itertools
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Fixed parameters
SIGMA = 20.0
NUM_SAMPLES = [32, 64, 128]
EFFORT_MIN = 0.0
EFFORT_MAX = 100.0
SEED = 42

# Sweep grids
DELTAS = [0.5, 0.75, 1.0]
ETAS = [0.02, 0.05, 0.10]
MAX_ITERS = [200, 500, 1000]
TOLS = [1e-3, 5e-4, 1e-4]

RESULTS_CSV = Path("results/one_stage_two_players.csv")
LOG_DIR = Path("results/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)


def run_mcfd(delta: float, eta: float, num_samples: int, max_iters: int, tol: float):
    """Invoke run_two_players with MC-FD settings."""
    cmd = [
        sys.executable,
        "run/run_two_players.py",
        "--method", "mcfd",
        "--mcfd-sigma1", str(SIGMA),
        "--mcfd-sigma2", str(SIGMA),
        "--mcfd-delta", str(delta),
        "--mcfd-eta", str(eta),
        "--mcfd-num-samples", str(num_samples),
        "--mcfd-max-iters", str(max_iters),
        "--mcfd-tol", str(tol),
        "--mcfd-effort-min", str(EFFORT_MIN),
        "--mcfd-effort-max", str(EFFORT_MAX),
        "--mcfd-seed", str(SEED),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    combos = list(itertools.product(DELTAS, ETAS, NUM_SAMPLES, MAX_ITERS, TOLS))
    total_runs = len(combos)
    start_time = time.time()

    summary_name = LOG_DIR / f"mcfd_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(summary_name, "w") as log:
        log.write(f"MC-FD sweep started {datetime.now()}\n")
        log.write(f"Total combinations: {total_runs}\n")
        log.write(f"Sigma: {SIGMA}\n")
        log.write(f"Effort bounds: [{EFFORT_MIN}, {EFFORT_MAX}]\n")
        log.write(f"Seed: {SEED}\n")
        log.write(f"Num samples grid: {NUM_SAMPLES}\n")
        log.write(f"Delta grid: {DELTAS}\n")
        log.write(f"Eta grid: {ETAS}\n")
        log.write(f"Max iters grid: {MAX_ITERS}\n")
        log.write(f"Tol grid: {TOLS}\n")
        log.write("=" * 80 + "\n")

        for idx, (delta, eta, num_samples, max_iters, tol) in enumerate(combos, 1):
            header = (
                f"[{idx}/{total_runs}] σ={SIGMA} δ={delta} η={eta} "
                f"N={num_samples} max_iters={max_iters} tol={tol}"
            )
            print(header)
            log.write(header + "\n")

            result = run_mcfd(delta, eta, num_samples, max_iters, tol)
            if result.returncode == 0:
                status = "SUCCESS"
            else:
                status = f"FAIL ({result.returncode})"

            log.write(f"Status: {status}\n")
            if result.stdout:
                log.write("STDOUT:\n" + result.stdout + "\n")
            if result.stderr:
                log.write("STDERR:\n" + result.stderr + "\n")
            log.write("-" * 80 + "\n")

            if result.returncode != 0:
                print(f"  -> failed, see log for details")

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    print("\nSweep complete.")
    print(f"Results CSV: {RESULTS_CSV}")
    print(f"Log: {summary_name}")
    print(f"Elapsed: {mins}m {secs}s")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
MC-FD Parameter Sweep Script

Sweeps over:
  - Sigma (std of performance shock): 15
  - Delta (finite-difference perturbation): 0.5, 0.75, 1.0
  - Eta (gradient ascent step size): 0.02, 0.06, 0.10
  - Num samples (Monte-Carlo batch size): 32, 64, 128

Total combinations: 1 × 3 × 3 × 3 = 27 runs
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Configuration
SIGMA = 15.0
DELTAS = [0.5, 0.75, 1.0]
ETAS = [0.02, 0.06, 0.10]
NUM_SAMPLES = [32, 64, 128]

# Fixed parameters
MAX_ITERS = 500
TOL = 1e-3
EFFORT_MIN = 0.0
EFFORT_MAX = 100.0
SEED = 42

# Output paths
CSV_PATH = Path("results/one_stage_two_players.csv")
LOG_DIR = Path("results/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

SWEEP_LOG = LOG_DIR / f"mcfd_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def run_mcfd(delta, eta, num_samples, sigma=SIGMA):
    """Run MC-FD with specified parameters."""
    cmd = [
        "python", "run/run_two_players.py",
        "--method", "mcfd",
        "--mcfd-sigma1", str(sigma),
        "--mcfd-sigma2", str(sigma),
        "--mcfd-delta", str(delta),
        "--mcfd-eta", str(eta),
        "--mcfd-num-samples", str(num_samples),
        "--mcfd-max-iters", str(MAX_ITERS),
        "--mcfd-tol", str(TOL),
        "--mcfd-effort-min", str(EFFORT_MIN),
        "--mcfd-effort-max", str(EFFORT_MAX),
        "--mcfd-seed", str(SEED),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    total_runs = len(DELTAS) * len(ETAS) * len(NUM_SAMPLES)
    current_run = 0
    successful_runs = 0
    failed_runs = 0
    
    print("=" * 70)
    print("MC-FD Parameter Sweep")
    print("=" * 70)
    print(f"Sigma (σ): {SIGMA}")
    print(f"Deltas (δ): {DELTAS}")
    print(f"Etas (η): {ETAS}")
    print(f"Num Samples (N): {NUM_SAMPLES}")
    print(f"Total combinations: {total_runs}")
    print(f"Results will be saved to: {CSV_PATH}")
    print(f"Sweep log will be saved to: {SWEEP_LOG}")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    with open(SWEEP_LOG, 'w') as log_file:
        log_file.write(f"MC-FD Parameter Sweep Started: {datetime.now()}\n")
        log_file.write(f"Sigma: {SIGMA}\n")
        log_file.write(f"Deltas: {DELTAS}\n")
        log_file.write(f"Etas: {ETAS}\n")
        log_file.write(f"Num Samples: {NUM_SAMPLES}\n")
        log_file.write(f"Total runs: {total_runs}\n")
        log_file.write("=" * 70 + "\n\n")
        
        for delta in DELTAS:
            for eta in ETAS:
                for num_samples in NUM_SAMPLES:
                    current_run += 1
                    
                    print("-" * 70)
                    print(f"[{current_run}/{total_runs}] Running:")
                    print(f"  σ = {SIGMA}")
                    print(f"  δ = {delta}")
                    print(f"  η = {eta}")
                    print(f"  N = {num_samples}")
                    print("-" * 70)
                    
                    log_file.write(f"\n[{current_run}/{total_runs}] Run started: {datetime.now()}\n")
                    log_file.write(f"  Parameters: σ={SIGMA}, δ={delta}, η={eta}, N={num_samples}\n")
                    log_file.write("-" * 70 + "\n")
                    
                    result = run_mcfd(delta, eta, num_samples)
                    
                    if result.returncode == 0:
                        successful_runs += 1
                        status = "✅ SUCCESS"
                        print(f"{status}")
                    else:
                        failed_runs += 1
                        status = "❌ FAILED"
                        print(f"{status}")
                        print(f"Error: {result.stderr[:200]}")
                    
                    log_file.write(f"Status: {status}\n")
                    log_file.write(f"Return code: {result.returncode}\n")
                    if result.stdout:
                        log_file.write("STDOUT:\n" + result.stdout + "\n")
                    if result.stderr:
                        log_file.write("STDERR:\n" + result.stderr + "\n")
                    log_file.write("-" * 70 + "\n")
                    log_file.flush()
                    
                    # Progress update
                    progress = (current_run / total_runs) * 100
                    elapsed = time.time() - start_time
                    avg_time = elapsed / current_run
                    remaining = avg_time * (total_runs - current_run)
                    
                    print(f"Progress: {progress:.1f}% | "
                          f"Elapsed: {elapsed/60:.1f}m | "
                          f"Remaining: ~{remaining/60:.1f}m")
                    print()
    
    # Summary
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print("=" * 70)
    print("Sweep Complete!")
    print("=" * 70)
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Elapsed time: {minutes}m {seconds}s")
    print(f"Results saved to: {CSV_PATH}")
    print(f"Sweep log saved to: {SWEEP_LOG}")
    print("=" * 70)
    
    # Generate summary
    try:
        import pandas as pd
        if CSV_PATH.exists():
            df = pd.read_csv(CSV_PATH)
            mcfd_rows = df[df['Model_training'] == 'mcfd']
            
            if len(mcfd_rows) > 0:
                print("\n📊 Summary of MC-FD Results:")
                print(f"   Total MC-FD runs in CSV: {len(mcfd_rows)}")
                if 'final_stage2_effort' in mcfd_rows.columns:
                    print(f"   Final effort range: {mcfd_rows['final_stage2_effort'].min():.2f} - {mcfd_rows['final_stage2_effort'].max():.2f}")
                    print(f"   Average final effort: {mcfd_rows['final_stage2_effort'].mean():.2f}")
                if 'Convergence_Quality' in mcfd_rows.columns:
                    print(f"   Convergence quality distribution:")
                    print(mcfd_rows['Convergence_Quality'].value_counts().to_string())
    except ImportError:
        print("\n⚠️  pandas not available - skipping summary")
    except Exception as e:
        print(f"\n⚠️  Could not generate summary: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Sweep interrupted by user")
        sys.exit(1)

