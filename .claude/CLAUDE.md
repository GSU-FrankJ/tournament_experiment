# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Game-theory research project (TEL-PPO paper): PPO agents learn Nash equilibrium effort in tournament games. Python 3.8+ / PyTorch. Theory: `e*(q) = (w_H - w_L) / (4qk)`, defaults: w_H=6.5, w_L=3.0, k=0.0004, q∈{25,40,55}.

4 experiment types: `two_players`, `three_players`, `different_cost`, `different_ability`.

Pipeline: `results/*/convergence/*.json` → `paper/generator/` → `paper/{figures,tables,data}/`

## Common Commands

```bash
pip install -r requirements.txt

# Run experiments
python run/run_two_players.py --method ppo --q 25 --episodes 131072 --seed 42
python run/run_two_players.py --method gradient --q 40
python run/run_three_players.py --method ppo --q 40 --episodes 2048000 --seed 42
python run/run_different_ability.py --method both --q 25 --episodes 131072
python run/run_different_cost.py --method both --q 25 --episodes 131072

# Paper artifacts
python -m paper.generator make_all
python -m paper.generator --dry-run

# Plots and diagnostics
python tools/plot_convergence.py
python tools/plot_convergence_detailed.py
python tools/audit_rollout_modes.py
```

No formal test suite. Validation via theory-alignment and exploitability evaluation.

## Architecture

**Data flow:** `config/*.py` → `run/run_*.py` → `agents/ppo_*.py` + `envs/*_env.py` → `results/`

- **`agents/ppo_two_players_clean.py`** — PPO trainer (`PPOTwoPlayersBandit`), Beta distribution policy over [0,1] scaled to `effort_bounds`
- **`envs/`** — Gym-like environments. Closed-form expected utilities (no stochastic sampling)
- **`utils/prob.py`** — `p_from_diff(d, q)`: piecewise triangular win probability (core math)
- **`utils/theory.py`** — Closed-form equilibrium formulas for all 4 experiment types
- **`utils/eval.py`** — Convergence quality: Excellent <0.5, Good <1.0, Fair <5.0, Poor ≥5.0
- **`run/run_*.py`** — CLI entry points (~1000-2000 lines each, ~60% duplicated logic across runners)
- **`paper/generator/`** — `run_registry.py` discovers runs, `extract.py`/`metrics.py`/`plots.py`/`tables.py` generate outputs

Results: `results/{experiment}/convergence/`, `logs/`, `summary.csv`. Ablation: `results/ablation/`.

## Commit Rules

```
- Conventional commits ONLY: feat:, fix:, docs:, chore:, refactor:, test:, ci:
- Subject line: imperative mood, <72 chars, no trailing period
- NEVER use your response summary as a commit message
  Bad:  "Done. The RelErr column is now in final_summary.csv..."
  Good: "feat: add RelErr column to final_summary.csv"
- NEVER commit messages like "update", "new results", "modify", "changes"
- One logical change per commit. Never mix code fixes + experiment reruns + figure regen
- If a change touches both code and results, split into two commits
```

## Workflow Boundaries

```
- When asked to "fix X", fix ONLY X. Do not refactor, optimize, or "improve" nearby code
- Before deleting any file, list what will be deleted and confirm
- Before running experiments that take >1 minute, confirm parameters first
- Never modify files in results/*/convergence/ without explicit confirmation
- When generating paper artifacts, use: python -m paper.generator make_all
- Read docs/STATE.md (if it exists) before starting any task
```

## Code Style

```
- Python 3.8+ compatible
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- Imports: stdlib, then third-party, then local (separated by blank lines)
- No wildcard imports
- Max line length: 100 chars
```

## Critical Invariants

- **Denominator 4** for two-player equilibrium. Never mix with denominator-6
- **Beta mean for evaluation**, not mode, even when α,β > 1
- **Closed-form expected utilities** — no stochastic noise during rollouts
- **Both players' transitions** stored in rollouts (self-play, lagged opponent)
- Results are precious and irreproducible (GPU training runs). Never delete without confirmation
- Never modify `paper/generator/config.py` theory parameters without confirming — they match the paper's math

## Never Do These

```
- Never commit __pycache__/, *.pyc, or .pt/.pth files
- Never commit to main directly for multi-step changes; use a feature branch
- Never rewrite git history on main (no force push, no filter-repo without explicit ask)
- Never install packages without --break-system-packages flag (system Python)
```

## State Tracking

After completing a task, update `docs/STATE.md` with: what was done, current known issues, suggested next steps. Create it if it doesn't exist.

## Task Pipeline

For non-trivial multi-phase work, create a task folder under `docs/tasks/` following the conventions in `docs/tasks/README.md`. Each task gets its own `CLAUDE.md`, `STATE.md`, and numbered phase files (`phase01.md`–`phase99.md`).

## Ablation Flags

Runners support: `--disable-entropy`, `--disable-cheap-gate`, `--disable-exploitability`, `--exploit-every-updates N`
