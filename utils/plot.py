"""
Plotting utilities for experiments.

Saves symmetric average effort vs. training steps and overlays closed-form
benchmark lines e*(q) per required configurations.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_effort_curve(
    efforts: List[float],
    qs: List[float],
    e_star_fn,
    w_h: float,
    w_l: float,
    k: float,
    title: str,
    output_png: str,
    effort_bounds: Tuple[float, float] = (0.0, 200.0),
) -> None:
    """Plot effort over episodes with e*(q) overlays.

    Args:
        efforts: Sequence of symmetric average efforts per episode (or checkpoints)
        qs: List of q values to overlay theoretical lines for
        e_star_fn: Callable(q, w_h, w_l, k) -> e*
        w_h, w_l, k: Parameters for theory
        title: Chart title
        output_png: Destination path under results/
        effort_bounds: Bounds for y-axis
    """
    ensure_dir(output_png)
    plt.figure(figsize=(8, 5))
    x = np.arange(len(efforts))
    plt.plot(x, efforts, label="learned effort", color="#1f77b4")

    # Overlay theoretical lines for each q
    for q in qs:
        est = float(e_star_fn(q, w_h, w_l, k))
        plt.hlines(est, xmin=0, xmax=len(efforts) - 1 if len(efforts) > 0 else 1, linestyles="dashed", colors="#d62728", label=f"e*(q={q:.0f})")

    plt.ylim(effort_bounds[0], effort_bounds[1])
    plt.xlabel("episodes")
    plt.ylabel("effort")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()





















