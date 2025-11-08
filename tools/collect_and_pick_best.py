#!/usr/bin/env python3
"""Collect 200-update metrics, score convergence, and emit a 500-update command."""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


RESULTS_ROOT = Path("results")
PARAM_KEYS = [
    "lr_start",
    "lr_final",
    "target_kl",
    "entropy_hold_fraction",
    "clip_range_end",
]
PARAM_PATTERN = re.compile(
    r"lr(?P<lr_start>[\d.eE-]+)_(?P<lr_final>[\d.eE-]+)_"
    r"kl(?P<target_kl>[\d.eE-]+)_ent(?P<entropy_hold_fraction>[\d.eE-]+)_"
    r"clip(?P<clip_range_end>[\d.eE-]+)"
)


@dataclass
class RunRecord:
    run_dir: Path
    params: Dict[str, Optional[float]]
    hit_step: float
    final_gap: float

    def display_params(self) -> Dict[str, str]:
        return {key: format_optional_float(self.params.get(key)) for key in PARAM_KEYS}

    def hit_step_display(self) -> str:
        if math.isinf(self.hit_step):
            return "inf"
        if float(self.hit_step).is_integer():
            return str(int(self.hit_step))
        return f"{self.hit_step:.6g}"

    def final_gap_display(self) -> str:
        return f"{self.final_gap:.6g}"


def format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "<unknown>"
    return f"{value:.6g}"


def coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
    except Exception:  # pragma: no cover - defensive
        return None
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_params_from_json(path: Path) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {key: None for key in PARAM_KEYS}
    if not path.exists():
        return result
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"Warning: Failed to read {path.as_posix()}: {exc}", file=sys.stderr)
        return result
    for key in PARAM_KEYS:
        result[key] = coerce_float(payload.get(key))
    return result


def load_params_from_dir_name(run_dir: Path) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {key: None for key in PARAM_KEYS}
    try:
        relative = run_dir.relative_to(RESULTS_ROOT)
        normalized = relative.as_posix()
    except ValueError:
        normalized = run_dir.as_posix()
    normalized = normalized.replace("/", "_")
    match = PARAM_PATTERN.search(normalized)
    if not match:
        return result
    for key in PARAM_KEYS:
        result[key] = coerce_float(match.group(key))
    return result


def merge_params(primary: Dict[str, Optional[float]], secondary: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    merged = dict(primary)
    for key in PARAM_KEYS:
        if merged.get(key) is None:
            merged[key] = secondary.get(key)
    return merged


def find_gap_column(columns: Iterable[str]) -> Optional[str]:
    for column in columns:
        if column.lower() == "gap":
            return column
    gap_regex = re.compile(r"\bgap\b", re.IGNORECASE)
    for column in columns:
        if gap_regex.search(column):
            return column
    return None


def ensure_update_column(df: pd.DataFrame) -> pd.DataFrame:
    if "update" in df.columns:
        return df
    df = df.copy()
    df["update"] = range(len(df))
    return df


def compute_convergence(df: pd.DataFrame, gap_column: str) -> Tuple[float, float]:
    gap_series = pd.to_numeric(df[gap_column], errors="coerce")
    if gap_series.isna().all():
        raise ValueError("gap column contains only NaNs")
    gap_series = gap_series.ffill().bfill()
    if gap_series.isna().any():
        gap_series = gap_series.fillna(0.0)
    rolling = gap_series.rolling(window=5, min_periods=1).mean()
    first_gap = float(gap_series.iloc[0])
    last_gap = float(gap_series.iloc[-1])
    if last_gap <= 0.02:
        threshold = 0.02
    else:
        threshold = min(0.02, 0.1 * first_gap)
    mask = rolling <= threshold
    hit_step = math.inf
    if mask.any():
        hit_indices = mask[mask].index
        if len(hit_indices) > 0:
            hit_row = int(hit_indices[0])
            update_value = df.loc[hit_row, "update"]
            try:
                hit_step = int(update_value)
            except (TypeError, ValueError):
                hit_step = float(update_value)
    final_gap = float(rolling.iloc[-1])
    return hit_step, final_gap


def process_metrics(metrics_path: Path) -> Optional[RunRecord]:
    run_dir = metrics_path.parent

    params = load_params_from_json(run_dir / "params.json")
    params = merge_params(params, load_params_from_dir_name(run_dir))

    try:
        df = pd.read_csv(metrics_path)
    except Exception as exc:
        print(f"Warning: Failed to read {metrics_path.as_posix()}: {exc}", file=sys.stderr)
        return None
    df = ensure_update_column(df)
    if "update" not in df.columns:
        print(f"Warning: Unable to synthesize update column for {metrics_path.as_posix()}", file=sys.stderr)
        return None
    df = df.sort_values("update").reset_index(drop=True)
    gap_column = find_gap_column(df.columns)
    if gap_column is None:
        print(f"Warning: No gap-like column in {metrics_path.as_posix()}", file=sys.stderr)
        return None

    try:
        hit_step, final_gap = compute_convergence(df, gap_column)
    except Exception as exc:
        print(f"Warning: Failed to compute convergence for {metrics_path.as_posix()}: {exc}", file=sys.stderr)
        return None

    return RunRecord(run_dir=run_dir, params=params, hit_step=hit_step, final_gap=final_gap)


def print_table(records: List[RunRecord]) -> None:
    headers = [
        "run_dir",
        "lr_start",
        "lr_final",
        "target_kl",
        "entropy_hold_fraction",
        "clip_range_end",
        "hit_step",
        "final_gap",
    ]
    table: List[List[str]] = []
    for record in records:
        params_display = record.display_params()
        table.append(
            [
                record.run_dir.as_posix(),
                params_display["lr_start"],
                params_display["lr_final"],
                params_display["target_kl"],
                params_display["entropy_hold_fraction"],
                params_display["clip_range_end"],
                record.hit_step_display(),
                record.final_gap_display(),
            ]
        )

    widths = [len(header) for header in headers]
    for row in table:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def format_row(values: List[str]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in table:
        print(format_row(row))


def determine_runner_script() -> str:
    candidate = Path("run") / "run_one_stage_different_ability.py"
    if candidate.exists():
        return candidate.as_posix()
    fallback = Path("run") / "run_different_ability.py"
    if fallback.exists():
        return fallback.as_posix()
    raise FileNotFoundError("Unable to locate a runner script for different ability experiments.")


def build_command(script_path: str, params: Dict[str, Optional[float]]) -> str:
    args = {
        "--lr-start": format_optional_float(params.get("lr_start")),
        "--lr-final": format_optional_float(params.get("lr_final")),
        "--target-kl": format_optional_float(params.get("target_kl")),
        "--entropy-hold-fraction": format_optional_float(params.get("entropy_hold_fraction")),
        "--clip-range-end": format_optional_float(params.get("clip_range_end")),
    }
    command_lines = [
        f"python {script_path} \\",
        "  --updates 500 \\",
    ]
    for flag, value in args.items():
        command_lines.append(f"  {flag} {value} \\")
    command_lines.extend(
        [
            "  --seed 42 \\",
            "  --outdir results/sweeps/best_full_$(date +%F_%H%M%S)",
        ]
    )
    return "\n".join(command_lines)


def main() -> int:
    if not RESULTS_ROOT.exists():
        print("Error: results/ directory not found.", file=sys.stderr)
        return 1

    metrics_paths = sorted(RESULTS_ROOT.rglob("metrics.csv"))
    if not metrics_paths:
        print("Error: No metrics.csv files found under results/.", file=sys.stderr)
        return 1

    records: List[RunRecord] = []
    for metrics_path in metrics_paths:
        record = process_metrics(metrics_path)
        if record is not None:
            records.append(record)

    if not records:
        print("Error: No valid runs found after filtering.", file=sys.stderr)
        return 1

    records.sort(key=lambda rec: (rec.hit_step, rec.final_gap))

    print_table(records)
    best = records[0]
    best_display = best.display_params()
    print()
    print("BEST COMBO:")
    print(
        "lr_start={lr_start}, lr_final={lr_final}, target_kl={target_kl}, "
        "entropy_hold_fraction={entropy_hold_fraction}, clip_range_end={clip_range_end}".format(**best_display)
    )
    print()
    try:
        runner_script = determine_runner_script()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    command = build_command(runner_script, best.params)
    print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
