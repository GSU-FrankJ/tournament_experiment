#!/usr/bin/env python3
"""
Short-run hyperparameter sweep for the different-ability PPO runner.

Runs 16 short (200-update) experiments, scores each by convergence speed on the
gap curve, and prints the best configuration plus the 500-update command to
reproduce it. No full run is launched automatically.
"""

from __future__ import annotations

import itertools
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import re


RUNNER_CANDIDATES: Tuple[Path, ...] = (
    Path("run/run_one_stage_different_ability.py"),
    Path("run/run_different_ability.py"),
)

SHORT_UPDATES = 200
SWEEP_ROOT = Path("results/sweeps")
METRICS_FILENAME = "metrics.csv"


@dataclass
class SweepResult:
    lr_start: float
    lr_final: float
    target_kl: float
    entropy_hold_fraction: float
    clip_range_end: float
    hit_step: float
    final_gap: float
    outdir: Path
    command: List[str]


def find_runner() -> Path:
    for candidate in RUNNER_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate different-ability runner. "
        "Checked: {}".format(", ".join(str(p) for p in RUNNER_CANDIDATES))
    )


def slug(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.3e}".replace("+", "").replace("-", "m").replace(".", "p")


def build_outdir(base: Path, lr_start: float, lr_final: float, target_kl: float,
                 entropy_hold_fraction: float, clip_range_end: float) -> Path:
    name = (
        f"short_lr{slug(lr_start)}_lrf{slug(lr_final)}"
        f"_kl{slug(target_kl)}_ent{slug(entropy_hold_fraction)}"
        f"_clip{slug(clip_range_end)}"
    )
    outdir = base / name
    counter = 1
    while outdir.exists():
        counter += 1
        outdir = base / f"{name}_{counter}"
    return outdir


def build_command(
    runner: Path,
    updates: int,
    lr_start: float,
    lr_final: float,
    target_kl: float,
    entropy_hold_fraction: float,
    clip_range_end: float,
    outdir: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        str(runner),
        "--method",
        "ppo",
        "--updates",
        str(updates),
        "--lr-start",
        str(lr_start),
        "--lr-final",
        str(lr_final),
        "--target-kl",
        str(target_kl),
        "--entropy-hold-fraction",
        str(entropy_hold_fraction),
        "--clip-range-end",
        str(clip_range_end),
        "--seed",
        "42",
        "--outdir",
        str(outdir),
        "--skip-history",
    ]
    return cmd


def find_gap_column(df: pd.DataFrame) -> str:
    lower = {col.lower(): col for col in df.columns}
    if "gap" in lower:
        return lower["gap"]
    pattern = re.compile(r"\bgap\b", flags=re.IGNORECASE)
    for col in df.columns:
        if pattern.search(col):
            return col
    raise ValueError("Unable to locate a gap column in metrics.csv")


def compute_convergence_metrics(df: pd.DataFrame, gap_col: str) -> Tuple[float, float]:
    if "update" in df.columns:
        updates = df["update"].to_numpy()
    else:
        updates = np.arange(1, len(df) + 1)
        df = df.copy()
        df["update"] = updates

    gap_series = df[gap_col].astype(float)
    rolling = gap_series.rolling(window=5, min_periods=1).mean()
    final_gap = gap_series.iloc[-1]
    initial_gap = gap_series.iloc[0]
    if final_gap <= 0.02:
        threshold = 0.02
    else:
        threshold = min(0.02, 0.1 * initial_gap)

    hit_step: float = math.inf
    for idx, value in enumerate(rolling):
        if value <= threshold:
            hit_step = float(updates[idx])
            break

    final_rolling = float(rolling.iloc[-1])
    return hit_step, final_rolling


def run_single(
    runner: Path,
    lr_start: float,
    lr_final: float,
    target_kl: float,
    entropy_hold_fraction: float,
    clip_range_end: float,
) -> SweepResult:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    outdir = build_outdir(
        SWEEP_ROOT, lr_start, lr_final, target_kl, entropy_hold_fraction, clip_range_end
    )
    cmd = build_command(
        runner=runner,
        updates=SHORT_UPDATES,
        lr_start=lr_start,
        lr_final=lr_final,
        target_kl=target_kl,
        entropy_hold_fraction=entropy_hold_fraction,
        clip_range_end=clip_range_end,
        outdir=outdir,
    )
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    metrics_path = outdir / METRICS_FILENAME
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics file missing: {metrics_path}")
    df = pd.read_csv(metrics_path)
    gap_col = find_gap_column(df)
    hit_step, final_gap = compute_convergence_metrics(df, gap_col)

    return SweepResult(
        lr_start=lr_start,
        lr_final=lr_final,
        target_kl=target_kl,
        entropy_hold_fraction=entropy_hold_fraction,
        clip_range_end=clip_range_end,
        hit_step=hit_step,
        final_gap=final_gap,
        outdir=outdir,
        command=cmd,
    )


def main() -> None:
    runner = find_runner()
    lr_pairs = ((3e-4, 1e-4), (4e-4, 2e-4))
    target_kl_values = (0.015, 0.02)
    entropy_hold_values = (0.66, 0.85)
    clip_end_values = (0.15, 0.2)

    combos = list(itertools.product(lr_pairs, target_kl_values, entropy_hold_values, clip_end_values))
    results: List[SweepResult] = []

    try:
        for (lr_start, lr_final), target_kl, entropy_hold_fraction, clip_range_end in combos:
            result = run_single(
                runner,
                lr_start=lr_start,
                lr_final=lr_final,
                target_kl=target_kl,
                entropy_hold_fraction=entropy_hold_fraction,
                clip_range_end=clip_range_end,
            )
            results.append(result)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Sweep aborted: command failed with exit code {exc.returncode}") from exc

    results.sort(key=lambda r: (r.hit_step, r.final_gap))

    header = (
        f"{'lr_start':>10} {'lr_final':>10} {'target_kl':>10} "
        f"{'entropy':>8} {'clip_end':>9} {'hit_step':>10} {'final_gap':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for res in results:
        hit_display = "INF" if math.isinf(res.hit_step) else f"{int(res.hit_step)}"
        print(
            f"{res.lr_start:>10.2e} {res.lr_final:>10.2e} {res.target_kl:>10.3f} "
            f"{res.entropy_hold_fraction:>8.2f} {res.clip_range_end:>9.2f} "
            f"{hit_display:>10} {res.final_gap:>10.4f}"
        )

    best = results[0]
    best_hit = "INF" if math.isinf(best.hit_step) else int(best.hit_step)
    print(
        f"\nBEST COMBO: lr_start={best.lr_start:.2e}, lr_final={best.lr_final:.2e}, "
        f"target_kl={best.target_kl:.3f}, entropy_hold_fraction={best.entropy_hold_fraction:.2f}, "
        f"clip_range_end={best.clip_range_end:.2f} (hit_step={best_hit}, final_gap={best.final_gap:.4f})"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_outdir = SWEEP_ROOT / f"best_full_{timestamp}"
    full_cmd = build_command(
        runner=runner,
        updates=500,
        lr_start=best.lr_start,
        lr_final=best.lr_final,
        target_kl=best.target_kl,
        entropy_hold_fraction=best.entropy_hold_fraction,
        clip_range_end=best.clip_range_end,
        outdir=best_outdir,
    )
    print("500-update command (not executed):")
    print(" ".join(full_cmd))


if __name__ == "__main__":
    main()
