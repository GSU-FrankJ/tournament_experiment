"""
Run Registry: Discover and catalog experiment runs.

Provides backward compatibility for both old and new filename formats:
- Old: ppo_q40.0_convergence.json
- New: ppo_q40.0_seed42_baseline_convergence.json

Fallback strategy:
1. Try to find matching *_metadata.json for full info
2. Parse filename for method, q, seed, ablation
3. For old files: join with CSV on (method, q) to get seed if available
4. If still unknown: seed=42, ablation="baseline", log warning
"""

import os
import re
import json
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import pandas as pd

from .config import CONVERGENCE_DIR, CONVERGENCE_DIRS, CSV_PATH, Q_VALUES


@dataclass
class Run:
    """Represents a single experiment run."""
    method: str                # "TEL-PPO" | "Gradient" | "Theory"
    q: float                   # Noise parameter
    seed: int                  # Random seed
    ablation: str              # "baseline" | "no_cheap_gate" | "no_exploitability"
    path: str                  # Path to convergence JSON
    experiment: str = "two_players"        # "two_players" | "three_players" | "different_cost" | "different_ability"
    weight_variant: str = "baseline"       # "baseline" | "wh8_wl4" (parameter set variant)
    metadata_path: Optional[str] = None    # Path to metadata JSON (may be None)
    has_time_series: bool = False          # True if new format with KL/alpha/beta
    is_legacy_format: bool = False         # True if inferred from old filename
    is_nested_format: bool = False         # True if history-based nested JSON

    @property
    def run_key(self) -> Tuple[str, str, float, int, str]:
        """Unique key for this run."""
        return (self.experiment, self.method, self.q, self.seed, self.ablation)

    def __str__(self) -> str:
        legacy_tag = " (legacy)" if self.is_legacy_format else ""
        ts_tag = " [has time-series]" if self.has_time_series else ""
        exp_tag = f" [{self.experiment}]" if self.experiment != "two_players" else ""
        return f"{self.method}_q{self.q}_seed{self.seed}_{self.ablation}{exp_tag}{legacy_tag}{ts_tag}"


def _parse_filename(filename: str) -> Dict[str, any]:
    """
    Parse convergence filename to extract method, q, seed, ablation, and experiment.

    Supports formats:
    - ppo_q40.0_convergence.json (legacy two_players)
    - ppo_q40.0_seed42_convergence.json (two_players baseline)
    - ppo_q40.0_seed42_no_cheap_gate_convergence.json (two_players ablation)
    - gradient_q25.0_convergence.json (two_players gradient)
    - ppo_3p_q40.0_seed42_baseline_convergence.json (three_players)
    - gradient_3p_q40.0_convergence.json (three_players gradient)
    - different_cost_ppo_q40.0_seed42_baseline_convergence.json
    - different_cost_gradient_q40.0_convergence.json
    - different_ability_ppo_q40.0_seed42_baseline_convergence.json
    - different_ability_gradient_q40.0_convergence.json
    """
    result = {
        "method": None,
        "q": None,
        "seed": None,
        "ablation": None,
        "is_legacy": False,
        "experiment": None,
    }

    # Remove _convergence.json suffix
    base = filename.replace("_convergence.json", "")

    # --- Three-player formats ---
    # ppo_3p_q{q}_seed{seed}[_{ablation}]
    pattern_3p_new = r"^(ppo)_3p_q([\d.]+)_seed(\d+)(?:_(.+))?$"
    match = re.match(pattern_3p_new, base, re.IGNORECASE)
    if match:
        result["method"] = "TEL-PPO"
        result["q"] = float(match.group(2))
        result["seed"] = int(match.group(3))
        result["ablation"] = match.group(4) if match.group(4) else "baseline"
        result["experiment"] = "three_players"
        return result

    # gradient_3p_q{q}
    pattern_3p_legacy = r"^(gradient)_3p_q([\d.]+)$"
    match = re.match(pattern_3p_legacy, base, re.IGNORECASE)
    if match:
        result["method"] = "Gradient"
        result["q"] = float(match.group(2))
        result["seed"] = 42
        result["ablation"] = "baseline"
        result["is_legacy"] = True
        result["experiment"] = "three_players"
        return result

    # --- different_cost / different_ability formats ---
    # {prefix}_ppo_q{q}_seed{seed}[_{ablation}]
    pattern_prefix_new = r"^(different_cost|different_ability)_(ppo|gradient)_q([\d.]+)_seed(\d+)(?:_(.+))?$"
    match = re.match(pattern_prefix_new, base, re.IGNORECASE)
    if match:
        result["experiment"] = match.group(1)
        method_raw = match.group(2).upper()
        result["method"] = "TEL-PPO" if method_raw == "PPO" else method_raw.title()
        result["q"] = float(match.group(3))
        result["seed"] = int(match.group(4))
        result["ablation"] = match.group(5) if match.group(5) else "baseline"
        return result

    # {prefix}_gradient_q{q} (legacy gradient for different_cost/different_ability)
    pattern_prefix_legacy = r"^(different_cost|different_ability)_(gradient)_q([\d.]+)$"
    match = re.match(pattern_prefix_legacy, base, re.IGNORECASE)
    if match:
        result["experiment"] = match.group(1)
        result["method"] = "Gradient"
        result["q"] = float(match.group(3))
        result["seed"] = 42
        result["ablation"] = "baseline"
        result["is_legacy"] = True
        return result

    # --- Standard two_players formats ---
    # {method}_q{q}_seed{seed}[_{ablation}]
    new_pattern = r"^(ppo|gradient)_q([\d.]+)_seed(\d+)(?:_(.+))?$"
    match = re.match(new_pattern, base, re.IGNORECASE)
    if match:
        method_raw = match.group(1).upper()
        result["method"] = "TEL-PPO" if method_raw == "PPO" else "Gradient"
        result["q"] = float(match.group(2))
        result["seed"] = int(match.group(3))
        result["ablation"] = match.group(4) if match.group(4) else "baseline"
        result["experiment"] = "two_players"
        return result

    # {method}_q{q}_{ablation} (gradient with ablation, no seed)
    gradient_ablation_pattern = r"^(gradient)_q([\d.]+)_(.+)$"
    match = re.match(gradient_ablation_pattern, base, re.IGNORECASE)
    if match:
        result["method"] = "Gradient"
        result["q"] = float(match.group(2))
        result["seed"] = 42
        result["ablation"] = match.group(3)
        result["experiment"] = "two_players"
        return result

    # {method}_q{q} (legacy)
    legacy_pattern = r"^(ppo|gradient)_q([\d.]+)$"
    match = re.match(legacy_pattern, base, re.IGNORECASE)
    if match:
        method_raw = match.group(1).upper()
        result["method"] = "TEL-PPO" if method_raw == "PPO" else "Gradient"
        result["q"] = float(match.group(2))
        result["seed"] = 42  # Default seed for legacy files
        result["ablation"] = "baseline"  # Default ablation for legacy files
        result["is_legacy"] = True
        result["experiment"] = "two_players"
        return result

    return result


def _check_has_time_series(data: Dict) -> bool:
    """Check if convergence JSON has extended time-series metrics."""
    # Flat format (two_players, three_players)
    required_fields = ["approx_kl", "alpha_mean", "beta_mean"]
    if all(field in data and isinstance(data[field], list) and len(data[field]) > 0
           for field in required_fields):
        return True

    # Nested format (different_cost, different_ability)
    history = data.get("history", {})
    if isinstance(history, dict) and "steps" in history:
        return len(history.get("steps", [])) > 0

    return False


def _check_is_nested_format(data: Dict) -> bool:
    """Check if convergence JSON uses the nested history format."""
    history = data.get("history", {})
    return isinstance(history, dict) and "steps" in history


def _load_csv_for_fallback(csv_path: str) -> Optional[pd.DataFrame]:
    """Load CSV for seed/ablation fallback lookup."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        warnings.warn(f"Failed to load CSV for fallback: {e}")
        return None


def discover_runs(
    convergence_dir: str = None,
    csv_path: str = None,
) -> List[Run]:
    """
    Discover all experiment runs from convergence JSON files.

    Scans all per-experiment convergence directories when convergence_dir
    is not explicitly provided.

    Args:
        convergence_dir: Path to a single convergence directory (or None to scan all)
        csv_path: Path to results CSV for fallback info

    Returns:
        List of Run objects representing discovered runs
    """
    if csv_path is None:
        csv_path = CSV_PATH

    # Determine which directories to scan
    if convergence_dir is not None:
        scan_dirs = [convergence_dir]
    else:
        scan_dirs = list(CONVERGENCE_DIRS.values())

    runs = []

    # Load CSV for fallback
    csv_df = _load_csv_for_fallback(csv_path)

    # Collect all convergence JSON files across directories
    json_entries = []  # (filepath, filename)
    for cdir in scan_dirs:
        if not os.path.exists(cdir):
            continue
        for f in os.listdir(cdir):
            if f.endswith("_convergence.json") and not f.endswith("_metadata.json"):
                json_entries.append((os.path.join(cdir, f), f, cdir))

    json_entries.sort(key=lambda x: x[1])
    
    for filepath, filename, cdir in json_entries:
        # Try to find metadata file
        metadata_filename = filename.replace("_convergence.json", "_metadata.json")
        metadata_path = os.path.join(cdir, metadata_filename)
        has_metadata = os.path.exists(metadata_path)
        
        # Parse filename
        parsed = _parse_filename(filename)
        
        if parsed["method"] is None:
            warnings.warn(f"Could not parse filename: {filename}")
            continue
        
        # Load convergence JSON to check for time-series and get additional info
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load {filename}: {e}")
            continue
        
        # Check for time-series metrics and nested format
        has_time_series = _check_has_time_series(data)
        is_nested = _check_is_nested_format(data)

        # Get seed and ablation from various sources (priority order)
        seed = parsed["seed"]
        ablation = parsed["ablation"]
        is_legacy = parsed["is_legacy"]
        experiment = parsed.get("experiment")
        weight_variant = "baseline"

        # Detect weight_variant from filename-parsed ablation
        # Filename like ppo_q35.0_seed42_wh8_wl4_convergence.json parses ablation="wh8_wl4"
        # This is actually a weight variant, not a code ablation
        KNOWN_WEIGHT_VARIANTS = {"wh8_wl4"}
        if ablation in KNOWN_WEIGHT_VARIANTS:
            weight_variant = ablation
            ablation = "baseline"  # The actual ablation is baseline

        # Filename-derived tag (after weight-variant extraction). Runners'
        # --output-tag renames the FILE (e.g. ..._round3_baseline_...) while
        # leaving JSON ablation_name as "baseline"; a "baseline" claim from
        # metadata/JSON must therefore not erase a more specific filename tag,
        # or distinct batches silently merge under one run identity.
        filename_tag = ablation

        # 1. Try metadata.json if available
        if has_metadata:
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                seed = metadata.get("seed", seed)
                # Use ablation_name from metadata only for real ablations
                meta_ablation = metadata.get("ablation_name", "baseline")
                if meta_ablation not in KNOWN_WEIGHT_VARIANTS:
                    if not (meta_ablation == "baseline" and filename_tag != "baseline"):
                        ablation = meta_ablation
                # Read variant_name from metadata (authoritative source)
                meta_variant = metadata.get("variant_name", "baseline")
                if meta_variant in KNOWN_WEIGHT_VARIANTS:
                    weight_variant = meta_variant
                is_legacy = False  # Has metadata, not legacy
            except Exception:
                pass

        # 2. Try data from JSON itself (new format includes these)
        if "seed" in data:
            seed = int(data["seed"])
            is_legacy = False
        if "ablation_name" in data:
            data_ablation = data["ablation_name"]
            if data_ablation not in KNOWN_WEIGHT_VARIANTS:
                if not (data_ablation == "baseline" and filename_tag != "baseline"):
                    ablation = data_ablation
            is_legacy = False
        
        # 3. If still legacy, try CSV fallback
        if is_legacy and csv_df is not None:
            method_csv = "ppo" if parsed["method"] == "TEL-PPO" else parsed["method"].lower()
            q_val = parsed["q"]
            
            # Look for matching row in CSV
            mask = (csv_df.get("q", pd.Series()) == q_val)
            if "method" in csv_df.columns:
                mask = mask & (csv_df["method"].str.lower() == method_csv)
            
            if mask.any():
                row = csv_df[mask].iloc[0]
                if "seed" in row and pd.notna(row["seed"]):
                    seed = int(row["seed"])
        
        # Infer experiment from directory if not parsed from filename
        if experiment is None:
            for exp_name, exp_dir in CONVERGENCE_DIRS.items():
                if os.path.normpath(cdir) == os.path.normpath(exp_dir):
                    experiment = exp_name
                    break
            if experiment is None:
                experiment = "two_players"

        # Create Run object
        run = Run(
            method=parsed["method"],
            q=parsed["q"],
            seed=seed,
            ablation=ablation,
            path=filepath,
            experiment=experiment,
            weight_variant=weight_variant,
            metadata_path=metadata_path if has_metadata else None,
            has_time_series=has_time_series,
            is_legacy_format=is_legacy,
            is_nested_format=is_nested,
        )
        
        runs.append(run)

    # Duplicate-identity guard: two files must never share the same run key,
    # or downstream groupbys silently merge their trajectories into one run.
    # On collision keep the newest file (mtime) and warn with both paths.
    by_key: Dict[Tuple, Run] = {}
    deduped: List[Run] = []
    for run in runs:
        key = (run.experiment, run.method, run.q, run.seed, run.ablation, run.weight_variant)
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = run
            deduped.append(run)
            continue
        keep, drop = (run, prev) if os.path.getmtime(run.path) >= os.path.getmtime(prev.path) else (prev, run)
        warnings.warn(
            f"Duplicate run identity {key}: keeping newer '{keep.path}', "
            f"excluding '{drop.path}'. Tag batches explicitly (--ablation-name) "
            f"to avoid collisions."
        )
        if keep is run:
            deduped[deduped.index(prev)] = run
            by_key[key] = run

    return deduped


def discover_theory_runs(q_values: List[float] = None) -> List[Run]:
    """
    Create synthetic 'Theory' runs for comparison.
    
    Theory is not trained, so these are placeholder runs for plotting.
    """
    if q_values is None:
        q_values = Q_VALUES
    
    return [
        Run(
            method="Theory",
            q=q,
            seed=0,
            ablation="theory",
            path="",  # No file
            has_time_series=False,
            is_legacy_format=False,
        )
        for q in q_values
    ]


def filter_runs(
    runs: List[Run],
    methods: Optional[List[str]] = None,
    q_values: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    ablations: Optional[List[str]] = None,
    require_time_series: bool = False,
) -> List[Run]:
    """Filter runs by criteria."""
    result = runs
    
    if methods is not None:
        methods_upper = [m.upper() for m in methods]
        result = [r for r in result if r.method.upper() in methods_upper or 
                  (r.method == "TEL-PPO" and "PPO" in methods_upper)]
    
    if q_values is not None:
        result = [r for r in result if r.q in q_values]
    
    if seeds is not None:
        result = [r for r in result if r.seed in seeds]
    
    if ablations is not None:
        result = [r for r in result if r.ablation in ablations]
    
    if require_time_series:
        result = [r for r in result if r.has_time_series]
    
    return result


def group_runs_by(
    runs: List[Run],
    by: str,  # "method", "q", "seed", "ablation"
) -> Dict[any, List[Run]]:
    """Group runs by a specific attribute."""
    groups = {}
    for run in runs:
        key = getattr(run, by)
        if key not in groups:
            groups[key] = []
        groups[key].append(run)
    return groups


def print_discovery_report(runs: List[Run]) -> None:
    """Print a summary report of discovered runs."""
    print(f"\nDiscovered {len(runs)} runs:")
    print("-" * 60)
    
    # Count by format
    legacy_count = sum(1 for r in runs if r.is_legacy_format)
    new_count = len(runs) - legacy_count
    ts_count = sum(1 for r in runs if r.has_time_series)
    
    print(f"  New format: {new_count}")
    print(f"  Legacy format: {legacy_count}")
    print(f"  With time-series: {ts_count}")
    print()
    
    # List runs by experiment then method
    by_experiment = group_runs_by(runs, "experiment")
    for experiment, exp_runs in sorted(by_experiment.items()):
        print(f"\n[{experiment}]:")
        by_method = group_runs_by(exp_runs, "method")
        for method, method_runs in sorted(by_method.items()):
            print(f"  {method}:")
            for run in sorted(method_runs, key=lambda r: (r.q, r.seed, r.ablation)):
                status = "✓" if run.has_time_series else "⚠"
                legacy_tag = " (legacy)" if run.is_legacy_format else ""
                nested_tag = " (nested)" if run.is_nested_format else ""
                print(f"    {status} q={run.q}, seed={run.seed}, ablation={run.ablation}{legacy_tag}{nested_tag}")
    
    # Warnings for missing data
    missing_ts = [r for r in runs if not r.has_time_series and r.method != "Theory"]
    if missing_ts:
        print(f"\n⚠ {len(missing_ts)} runs missing time-series data (need re-run with new logging):")
        for r in missing_ts:
            print(f"    - {r}")


if __name__ == "__main__":
    # Quick test: discover and report
    runs = discover_runs()
    print_discovery_report(runs)
