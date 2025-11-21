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
