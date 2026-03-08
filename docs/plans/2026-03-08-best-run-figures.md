# Best-Run Figures & Tables Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `--best-only` flag to paper generator that selects the single best seed per (experiment, q), archive old outputs, and regenerate all figures/tables.

**Architecture:** Add `select_best_runs()` to `run_registry.py` that loads final effort error from each baseline run's convergence JSON and keeps only the lowest-error seed per (experiment, q). Wire this into `__main__.py` via `--best-only` flag. Archive old paper outputs before regenerating.

**Tech Stack:** Python, pandas, matplotlib (existing paper generator pipeline)

---

### Task 1: Add `get_final_effort_error()` helper to `extract.py`

**Files:**
- Modify: `paper/generator/extract.py` (append after `get_convergence_step`)

**Step 1: Write the helper function**

Add to the end of `extract.py` (before `if __name__`):

```python
def get_final_effort_error_from_json(path: str, theoretical_effort: float) -> float:
    """Load a convergence JSON and return |final_effort_mean - theoretical|.

    For flat format: uses agent1_effort[-1] and agent2_effort[-1].
    For nested format: uses history.agent1_effort[-1] / history.effort[-1].

    Returns float('inf') if the file cannot be loaded or has no data.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception:
        return float('inf')

    # Nested format (different_cost, different_ability)
    history = data.get("history", {})
    if isinstance(history, dict) and "steps" in history:
        if "effort" in history and len(history["effort"]) > 0:
            final_effort = float(history["effort"][-1])
        elif "agent1_effort" in history and "agent2_effort" in history:
            e1 = history["agent1_effort"]
            e2 = history["agent2_effort"]
            if len(e1) > 0 and len(e2) > 0:
                final_effort = (float(e1[-1]) + float(e2[-1])) / 2.0
            else:
                return float('inf')
        else:
            return float('inf')
    else:
        # Flat format (two_players, three_players)
        e1 = data.get("agent1_effort", [])
        e2 = data.get("agent2_effort", [])
        if len(e1) > 0 and len(e2) > 0:
            final_effort = (float(e1[-1]) + float(e2[-1])) / 2.0
        else:
            return float('inf')

    return abs(final_effort - theoretical_effort)
```

**Step 2: Verify import exists**

Ensure `json` is already imported at top of `extract.py` (it is).

**Step 3: Commit**

```bash
git add paper/generator/extract.py
git commit -m "feat: add get_final_effort_error_from_json helper"
```

---

### Task 2: Add `select_best_runs()` to `run_registry.py`

**Files:**
- Modify: `paper/generator/run_registry.py` (append after `filter_runs`)

**Step 1: Add imports and function**

Add import at top of file:
```python
from .extract import get_final_effort_error_from_json
```

Add function after `filter_runs()`:

```python
def select_best_runs(runs: List[Run]) -> List[Run]:
    """For each (experiment, q), keep only the run with the lowest final effort error.

    - Considers all baseline variants (baseline, baseline_v2, etc.)
    - Keeps all Gradient runs unchanged (typically one per experiment x q)
    - Computes theoretical effort from config to evaluate error

    Returns:
        Filtered list of Run objects (one PPO per experiment x q, plus all Gradient)
    """
    from .config import e_star, THEORY_PARAMS

    # Separate gradient and PPO runs
    gradient_runs = [r for r in runs if r.method == "Gradient"]
    ppo_runs = [r for r in runs if r.method != "Gradient"]

    # Group PPO runs by (experiment, q)
    groups: Dict[Tuple[str, float], List[Run]] = {}
    for run in ppo_runs:
        key = (run.experiment, run.q)
        groups.setdefault(key, []).append(run)

    best_runs = []
    for (experiment, q), group in sorted(groups.items()):
        theo = e_star(q, **THEORY_PARAMS)

        # Score each run by final effort error
        scored = []
        for run in group:
            err = get_final_effort_error_from_json(run.path, theo)
            scored.append((err, run))

        scored.sort(key=lambda x: x[0])
        best_err, best_run = scored[0]
        best_runs.append(best_run)
        print(f"  Best run [{experiment}] q={q}: seed={best_run.seed} "
              f"ablation={best_run.ablation} error={best_err:.4f}")

    return gradient_runs + best_runs
```

**Step 2: Commit**

```bash
git add paper/generator/run_registry.py
git commit -m "feat: add select_best_runs to pick lowest-error seed per experiment x q"
```

---

### Task 3: Add `--best-only` flag to `__main__.py`

**Files:**
- Modify: `paper/generator/__main__.py`

**Step 1: Add import**

Add to imports:
```python
from .run_registry import discover_runs, print_discovery_report, select_best_runs
```
(Change existing import line to include `select_best_runs`.)

**Step 2: Add CLI flag**

Add after the `--dry-run` argument:
```python
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Select only the best seed per (experiment, q) based on lowest final effort error",
    )
```

**Step 3: Modify `cmd_make_all` to use best-only filtering**

Replace the data loading section in `cmd_make_all` (lines ~148-153) with:

```python
    # Load data
    print("\nLoading convergence data...")
    if args.best_only:
        from .run_registry import select_best_runs as _select_best
        from .extract import load_multiple_runs
        all_runs = discover_runs(
            convergence_dir=args.runs_dir,
            csv_path=args.csv,
        )
        # Filter to requested q values
        if q_values:
            all_runs = [r for r in all_runs if r.q in q_values]
        print(f"\nSelecting best run per (experiment, q)...")
        best = _select_best(all_runs)
        print_discovery_report(best)
        df = load_multiple_runs(best)
    else:
        df = load_all_convergence_data(
            convergence_dir=args.runs_dir,
            csv_path=args.csv,
            q_values=q_values,
        )
    print(f"Loaded {len(df)} rows from {df['method'].nunique()} methods")
```

**Step 4: Apply same pattern to `cmd_plot` and `cmd_table`**

Same filtering logic in both functions, gated on `args.best_only`.

**Step 5: Commit**

```bash
git add paper/generator/__main__.py
git commit -m "feat: add --best-only flag to paper generator CLI"
```

---

### Task 4: Archive old paper outputs

**Step 1: Create archive directory and move old outputs**

```bash
mkdir -p paper/archive/2026-03-08
mv paper/figures paper/archive/2026-03-08/figures
mv paper/tables paper/archive/2026-03-08/tables
mv paper/data paper/archive/2026-03-08/data
```

**Step 2: Commit**

```bash
git add paper/archive/
git commit -m "chore: archive old paper figures/tables/data before best-only regeneration"
```

---

### Task 5: Regenerate all figures and tables with best-only

**Step 1: Run the generator**

```bash
python -m paper.generator make_all --best-only
```

Expected: generates 9 figures (png+pdf) and 5 tables (csv+tex) in `paper/figures/`, `paper/tables/`, `paper/data/`.

**Step 2: Verify outputs exist**

```bash
ls paper/figures/*.png | wc -l   # expect 9
ls paper/tables/*.csv | wc -l    # expect 5
```

**Step 3: Commit**

```bash
git add paper/figures/ paper/tables/ paper/data/
git commit -m "feat: regenerate all paper figures and tables with best-only runs"
```

---

### Task 6: Final verification

**Step 1: Compare old vs new**

Spot-check that new figures use lower-error runs by examining `paper/data/convergence_main.csv` — each (experiment, q) should have exactly one seed.

**Step 2: Verify old multi-seed pipeline still works**

```bash
python -m paper.generator make_all --dry-run
```

Should still discover all runs (no data was deleted).
