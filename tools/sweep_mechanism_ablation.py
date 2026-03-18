#!/usr/bin/env python3
"""
Mechanism Ablation Sweep for All Experiment Types

Runs a mechanism ablation study: disable cheap gate, exploitability, or entropy
across all 4 experiment types with parallel execution and resume support.

Mechanism Settings (4 variants):
| Setting ID     | Description                    | CLI Flags                                       |
|----------------|--------------------------------|-------------------------------------------------|
| baseline       | All mechanisms enabled         | --enable-convergence-eval                       |
| no_cheap_gate  | Disable cheap gate pre-filter  | --enable-convergence-eval --disable-cheap-gate   |
| no_exploit     | Disable exploitability eval    | --enable-convergence-eval --disable-exploitability|
| no_entropy     | Zero entropy regularization    | --enable-convergence-eval --disable-entropy       |

Run Matrix:
- 4 experiment types: two_players, three_players, different_cost, different_ability
- 3 q values: 25, 40, 55
- 3 seeds: 42, 123, 456
- 4 mechanism settings
- Total: 4 x 3 x 3 x 4 = 144 runs

Usage:
    # Full sweep
    python tools/sweep_mechanism_ablation.py --parallel 4

    # Specific experiments
    python tools/sweep_mechanism_ablation.py --experiments two_players,different_cost

    # Dry run
    python tools/sweep_mechanism_ablation.py --dry-run

    # Resume interrupted sweep
    python tools/sweep_mechanism_ablation.py --resume

    # Smoke test (reduced episodes)
    python tools/sweep_mechanism_ablation.py --smoke-test --parallel 2
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_SCRIPTS = {
    "two_players": "run/run_two_players.py",
    "three_players": "run/run_three_players.py",
    "different_cost": "run/run_different_cost.py",
    "different_ability": "run/run_different_ability.py",
}

DEFAULT_EPISODES = 2_048_000
DEFAULT_M = 8192
DEFAULT_EXPLOIT_EVERY = 10

# Mechanism ablation settings: (setting_id, extra_cli_flags)
MECHANISM_SETTINGS: List[Tuple[str, List[str]]] = [
    ("baseline",       []),
    ("no_cheap_gate",  ["--disable-cheap-gate"]),
    ("no_exploit",     ["--disable-exploitability"]),
    ("no_entropy",     ["--disable-entropy"]),
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RunTask:
    experiment: str
    q: float
    seed: int
    setting_id: str
    extra_flags: List[str]

    @property
    def task_id(self) -> str:
        return f"{self.experiment}_q{self.q}_seed{self.seed}_{self.setting_id}"

    @property
    def output_filename(self) -> str:
        return f"{self.task_id}.json"


@dataclass
class RunResult:
    task: RunTask
    success: bool
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    stop_reason: Optional[str] = None
    stopped_at_update: Optional[int] = None
    final_exploit_max: Optional[float] = None
    final_effort_1: Optional[float] = None
    final_effort_2: Optional[float] = None
    final_abs_err_max: Optional[float] = None
    theoretical_effort_1: Optional[float] = None
    theoretical_effort_2: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": self.task.experiment,
            "q": self.task.q,
            "seed": self.task.seed,
            "setting_id": self.task.setting_id,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "stop_reason": self.stop_reason,
            "stopped_at_update": self.stopped_at_update,
            "final_exploit_max": self.final_exploit_max,
            "final_effort_1": self.final_effort_1,
            "final_effort_2": self.final_effort_2,
            "final_abs_err_max": self.final_abs_err_max,
            "theoretical_effort_1": self.theoretical_effort_1,
            "theoretical_effort_2": self.theoretical_effort_2,
        }


# =============================================================================
# Task Generation
# =============================================================================

def generate_tasks(
    experiments: List[str],
    q_values: List[float],
    seeds: List[int],
    settings: Optional[List[Tuple[str, List[str]]]] = None,
) -> List[RunTask]:
    if settings is None:
        settings = MECHANISM_SETTINGS

    tasks = []
    for experiment in experiments:
        if experiment not in EXPERIMENT_SCRIPTS:
            raise ValueError(f"Unknown experiment type: {experiment}")
        for q in q_values:
            for seed in seeds:
                for setting_id, extra_flags in settings:
                    tasks.append(RunTask(
                        experiment=experiment,
                        q=q,
                        seed=seed,
                        setting_id=setting_id,
                        extra_flags=list(extra_flags),
                    ))
    return tasks


# =============================================================================
# Command Building
# =============================================================================

def build_command(
    task: RunTask,
    episodes: int = DEFAULT_EPISODES,
    exploit_M: int = DEFAULT_M,
    exploit_every: int = DEFAULT_EXPLOIT_EVERY,
    theory_align_v2: bool = True,
    cheap_gate_profile: str = "relaxed",
) -> List[str]:
    script = EXPERIMENT_SCRIPTS[task.experiment]

    cmd = [
        sys.executable,
        script,
        "--method", "ppo",
        "--q", str(task.q),
        "--seed", str(task.seed),
        "--episodes", str(episodes),
        "--exploit-every-updates", str(exploit_every),
        "--exploit-M", str(exploit_M),
        "--ablation-name", task.setting_id,
        "--enable-convergence-eval",
        "--cheap-gate-profile", cheap_gate_profile,
    ]

    if theory_align_v2:
        cmd.append("--theory-align-v2")

    # Add mechanism-specific flags
    cmd.extend(task.extra_flags)

    return cmd


# =============================================================================
# Result Parsing
# =============================================================================

def find_convergence_json(task: RunTask) -> Optional[str]:
    convergence_dir = os.path.join("results", task.experiment, "convergence")

    patterns = []
    if task.experiment == "different_cost":
        patterns.append(f"different_cost_ppo_q{task.q:.1f}_seed{task.seed}_{task.setting_id}_convergence.json")
    elif task.experiment == "different_ability":
        patterns.append(f"different_ability_ppo_q{task.q:.1f}_seed{task.seed}_{task.setting_id}_convergence.json")
    elif task.experiment == "two_players":
        patterns.append(f"ppo_q{task.q:.1f}_seed{task.seed}_{task.setting_id}_convergence.json")
    elif task.experiment == "three_players":
        patterns.append(f"ppo_3p_q{task.q:.1f}_seed{task.seed}_{task.setting_id}_convergence.json")

    for pattern in patterns:
        path = os.path.join(convergence_dir, pattern)
        if os.path.exists(path):
            return path

    return None


def parse_convergence_json(json_path: str, task: RunTask) -> Dict[str, Any]:
    result = {}
    if not os.path.exists(json_path):
        return result

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        result["stop_reason"] = data.get("stop_reason", "max_updates")
        result["stopped_at_update"] = data.get("stopped_at_update")
        result["final_exploit_max"] = data.get("final_exploit_max")

        final_data = data.get("final", {})
        theoretical = data.get("theoretical", {})

        if task.experiment == "different_cost":
            result["final_effort_1"] = final_data.get("effort1")
            result["final_effort_2"] = final_data.get("effort2")
            result["final_abs_err_max"] = final_data.get("max_gap")
            result["theoretical_effort_1"] = theoretical.get("effort1")
            result["theoretical_effort_2"] = theoretical.get("effort2")
        elif task.experiment == "different_ability":
            effort = final_data.get("effort")
            result["final_effort_1"] = effort
            result["final_effort_2"] = effort
            result["final_abs_err_max"] = final_data.get("gap")
            result["theoretical_effort_1"] = theoretical.get("effort")
            result["theoretical_effort_2"] = theoretical.get("effort")
        else:
            result["final_effort_1"] = final_data.get("effort1") or final_data.get("effort")
            result["final_effort_2"] = final_data.get("effort2") or final_data.get("effort")
            result["final_abs_err_max"] = final_data.get("max_gap") or final_data.get("gap")
            result["theoretical_effort_1"] = theoretical.get("effort1") or theoretical.get("effort")
            result["theoretical_effort_2"] = theoretical.get("effort2") or theoretical.get("effort")

    except Exception as e:
        print(f"[WARN] Failed to parse {json_path}: {e}")

    return result


# =============================================================================
# Task Execution
# =============================================================================

def run_single_task(
    task: RunTask,
    output_dir: str,
    episodes: int = DEFAULT_EPISODES,
    exploit_M: int = DEFAULT_M,
    theory_align_v2: bool = True,
    cheap_gate_profile: str = "relaxed",
    dry_run: bool = False,
) -> RunResult:
    cmd = build_command(
        task,
        episodes=episodes,
        exploit_M=exploit_M,
        theory_align_v2=theory_align_v2,
        cheap_gate_profile=cheap_gate_profile,
    )
    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"[DRY RUN] {task.task_id}: {cmd_str}")
        return RunResult(task=task, success=True)

    start_time = datetime.datetime.now()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        if proc.returncode != 0:
            return RunResult(
                task=task,
                success=False,
                duration_seconds=duration,
                error_message=f"Exit code {proc.returncode}: {proc.stderr[:500]}",
            )

        json_path = find_convergence_json(task)
        parsed = {}
        if json_path:
            parsed = parse_convergence_json(json_path, task)

        return RunResult(
            task=task,
            success=True,
            duration_seconds=duration,
            stop_reason=parsed.get("stop_reason"),
            stopped_at_update=parsed.get("stopped_at_update"),
            final_exploit_max=parsed.get("final_exploit_max"),
            final_effort_1=parsed.get("final_effort_1"),
            final_effort_2=parsed.get("final_effort_2"),
            final_abs_err_max=parsed.get("final_abs_err_max"),
            theoretical_effort_1=parsed.get("theoretical_effort_1"),
            theoretical_effort_2=parsed.get("theoretical_effort_2"),
        )

    except Exception as e:
        return RunResult(
            task=task,
            success=False,
            error_message=f"Exception: {e}\n{traceback.format_exc()[:500]}",
        )


# =============================================================================
# Resume Support
# =============================================================================

def load_completed_runs(output_dir: str) -> set:
    completed = set()
    runs_dir = os.path.join(output_dir, "runs")

    if not os.path.exists(runs_dir):
        return completed

    for filename in os.listdir(runs_dir):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(runs_dir, filename)) as f:
                    data = json.load(f)
                if data.get("success"):
                    task_id = f"{data['experiment']}_q{data['q']}_seed{data['seed']}_{data['setting_id']}"
                    completed.add(task_id)
            except Exception:
                pass

    return completed


def save_run_result(result: RunResult, output_dir: str) -> None:
    runs_dir = os.path.join(output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    output_path = os.path.join(runs_dir, result.task.output_filename)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


# =============================================================================
# Sweep Execution
# =============================================================================

def run_sweep(
    experiments: List[str],
    q_values: List[float],
    seeds: List[int],
    output_dir: str,
    episodes: int = DEFAULT_EPISODES,
    exploit_M: int = DEFAULT_M,
    theory_align_v2: bool = True,
    cheap_gate_profile: str = "relaxed",
    parallel: int = 1,
    resume: bool = False,
    dry_run: bool = False,
    settings: Optional[List[Tuple[str, List[str]]]] = None,
) -> List[RunResult]:
    tasks = generate_tasks(experiments, q_values, seeds, settings)
    print(f"[Sweep] Generated {len(tasks)} tasks")

    if resume and not dry_run:
        completed = load_completed_runs(output_dir)
        tasks = [t for t in tasks if t.task_id not in completed]
        print(f"[Resume] {len(completed)} already done, {len(tasks)} remaining")

    if not tasks:
        print("[Sweep] No tasks to run")
        return []

    os.makedirs(output_dir, exist_ok=True)

    results = []
    start_time = datetime.datetime.now()

    if parallel > 1 and not dry_run:
        print(f"[Sweep] Running {len(tasks)} tasks with {parallel} workers")
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    run_single_task, task, output_dir, episodes, exploit_M,
                    theory_align_v2, cheap_gate_profile, dry_run
                ): task
                for task in tasks
            }

            for i, future in enumerate(as_completed(futures)):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = RunResult(
                        task=task,
                        success=False,
                        error_message=f"Future exception: {e}",
                    )

                results.append(result)

                if not dry_run:
                    save_run_result(result, output_dir)

                status = "OK" if result.success else "FAIL"
                stop = result.stop_reason or "N/A"
                print(f"[{i+1}/{len(tasks)}] {task.task_id}: {status} (stop={stop})")
    else:
        for i, task in enumerate(tasks):
            result = run_single_task(
                task, output_dir, episodes, exploit_M,
                theory_align_v2, cheap_gate_profile, dry_run
            )
            results.append(result)

            if not dry_run:
                save_run_result(result, output_dir)

            status = "OK" if result.success else "FAIL"
            stop = result.stop_reason or "N/A"
            print(f"[{i+1}/{len(tasks)}] {task.task_id}: {status} (stop={stop})")

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    successful = sum(1 for r in results if r.success)
    print(f"\n[Sweep] Complete: {successful}/{len(results)} successful in {duration/3600:.2f}h")

    return results


# =============================================================================
# Output Generation
# =============================================================================

def write_summary_csv(results: List[RunResult], output_path: str) -> None:
    if not results:
        return

    fieldnames = [
        "experiment", "setting_id", "q", "seed", "success",
        "stop_reason", "stopped_at_update",
        "final_exploit_max",
        "final_effort_1", "final_effort_2",
        "final_abs_err_max",
        "theoretical_effort_1", "theoretical_effort_2",
        "duration_seconds", "error_message",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    print(f"[Output] Wrote summary to {output_path}")


def write_summary_json(results: List[RunResult], output_path: str) -> None:
    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_runs": len(results),
        "successful_runs": sum(1 for r in results if r.success),
        "results": [r.to_dict() for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[Output] Wrote summary to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mechanism Ablation Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full sweep with 4 workers
    python tools/sweep_mechanism_ablation.py --parallel 4

    # Only two_players and different_cost
    python tools/sweep_mechanism_ablation.py --experiments two_players,different_cost

    # Dry run
    python tools/sweep_mechanism_ablation.py --dry-run

    # Resume interrupted sweep
    python tools/sweep_mechanism_ablation.py --resume

    # Smoke test (reduced episodes)
    python tools/sweep_mechanism_ablation.py --smoke-test

    # Specific settings only
    python tools/sweep_mechanism_ablation.py --settings baseline,no_entropy
""",
    )

    parser.add_argument(
        "--experiments",
        type=str,
        default="two_players,three_players,different_cost,different_ability",
        help="Comma-separated experiment types (default: all 4)",
    )
    parser.add_argument(
        "--q-values",
        type=str,
        default="25,40,55",
        help="Comma-separated q values (default: 25,40,55)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated seeds (default: 42,123,456)",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default=None,
        help="Comma-separated setting IDs (default: all 4: baseline,no_cheap_gate,no_exploit,no_entropy)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation/mechanism/",
        help="Output directory (default: results/ablation/mechanism/)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Episodes per run (default: {DEFAULT_EPISODES:,})",
    )
    parser.add_argument(
        "--exploit-M",
        type=int,
        default=DEFAULT_M,
        help=f"Monte Carlo samples (default: {DEFAULT_M})",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--theory-align-v2",
        action="store_true",
        default=True,
        help="Enable theory-align-v2 (default: True)",
    )
    parser.add_argument(
        "--no-theory-align-v2",
        action="store_false",
        dest="theory_align_v2",
        help="Disable theory-align-v2",
    )
    parser.add_argument(
        "--cheap-gate-profile",
        type=str,
        default="relaxed",
        choices=["relaxed", "default", "conservative", "aggressive"],
        help="Cheap gate profile (default: relaxed)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous runs (skip completed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick validation with reduced episodes",
    )

    args = parser.parse_args()

    # Parse lists
    experiments = [e.strip() for e in args.experiments.split(",")]
    q_values = [float(q.strip()) for q in args.q_values.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Filter settings if specified
    settings = MECHANISM_SETTINGS
    if args.settings:
        requested = {s.strip() for s in args.settings.split(",")}
        settings = [(sid, flags) for sid, flags in MECHANISM_SETTINGS if sid in requested]
        if not settings:
            print(f"ERROR: No matching settings for: {args.settings}")
            print(f"Available: {', '.join(s[0] for s in MECHANISM_SETTINGS)}")
            sys.exit(1)

    # Smoke test reduces episodes
    episodes = 20_000 if args.smoke_test else args.episodes
    if args.smoke_test:
        print("[SMOKE TEST] Running with reduced episodes (20,000)\n")

    # Run the sweep
    results = run_sweep(
        experiments=experiments,
        q_values=q_values,
        seeds=seeds,
        output_dir=args.output_dir,
        episodes=episodes,
        exploit_M=args.exploit_M,
        theory_align_v2=args.theory_align_v2,
        cheap_gate_profile=args.cheap_gate_profile,
        parallel=args.parallel,
        resume=args.resume,
        dry_run=args.dry_run,
        settings=settings,
    )

    # Generate outputs
    if results and not args.dry_run:
        # Combine with existing results if resuming
        if args.resume:
            runs_dir = os.path.join(args.output_dir, "runs")
            if os.path.exists(runs_dir):
                for filename in os.listdir(runs_dir):
                    if filename.endswith(".json"):
                        try:
                            with open(os.path.join(runs_dir, filename)) as f:
                                data = json.load(f)
                            task_id = f"{data['experiment']}_q{data['q']}_seed{data['seed']}_{data['setting_id']}"
                            if task_id not in {r.task.task_id for r in results}:
                                # Find matching setting flags
                                setting_flags = []
                                for sid, flags in MECHANISM_SETTINGS:
                                    if sid == data["setting_id"]:
                                        setting_flags = flags
                                        break
                                task = RunTask(
                                    experiment=data["experiment"],
                                    q=data["q"],
                                    seed=data["seed"],
                                    setting_id=data["setting_id"],
                                    extra_flags=setting_flags,
                                )
                                result = RunResult(
                                    task=task,
                                    success=data["success"],
                                    duration_seconds=data.get("duration_seconds", 0),
                                    error_message=data.get("error_message"),
                                    stop_reason=data.get("stop_reason"),
                                    stopped_at_update=data.get("stopped_at_update"),
                                    final_exploit_max=data.get("final_exploit_max"),
                                    final_effort_1=data.get("final_effort_1"),
                                    final_effort_2=data.get("final_effort_2"),
                                    final_abs_err_max=data.get("final_abs_err_max"),
                                    theoretical_effort_1=data.get("theoretical_effort_1"),
                                    theoretical_effort_2=data.get("theoretical_effort_2"),
                                )
                                results.append(result)
                        except Exception:
                            pass

        write_summary_csv(results, os.path.join(args.output_dir, "summary.csv"))
        write_summary_json(results, os.path.join(args.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
