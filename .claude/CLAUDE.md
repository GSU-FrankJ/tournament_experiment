# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Game-theory research project (TEL-PPO paper): PPO agents learn Nash equilibrium effort in tournament games. Python 3.8+ / PyTorch. Theory: `e*(q) = (w_H - w_L) / (4qk)`. Parameters vary by experiment (see `docs/experiment_config_040726.md`):
- 2P Set 1: k=0.00055, w_H=6.5, w_L=3.0, q∈{35,45,55}
- 2P Set 2: k=0.0006, w_H=8, w_L=4, q∈{35,45,55} (via CLI flags)
- 3P: k=0.001, w_H=6.5, w_L=3.0, q∈{35,55}
- Different Cost: k1=0.0004, k2=0.00055, w_H=8, w_L=5.5, q∈{35,55}
- Different Ability: k=0.0005, l1=10, l2=5, w_H=6.5, w_L=3.0, q∈{35,55}

5 experiment types: `two_players`, `three_players`, `different_cost`, `different_ability`, `two_stage` (runner deferred).

Pipeline: `results/*/convergence/*.json` → `paper/generator/` → `paper/{figures,tables,data}/`

## Common Commands

```bash
pip install -r requirements.txt

# Run experiments (use config default episodes, never copy quick-run examples)
python run/run_two_players.py --method ppo --q 35 --seed 42
python run/run_two_players.py --method gradient --q 45
python run/run_two_players.py --method ppo --q 35 --k 0.0006 --w_h 8 --w_l 4 --variant-name wh8_wl4 --seed 42  # Set 2
python run/run_three_players.py --method ppo --q 35 --seed 42
python run/run_different_ability.py --method both --q 35
python run/run_different_cost.py --method both --q 35

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
- **`envs/`** — Gym-like environments. Training rewards are SAMPLED one-step outcomes (uniform noise, realized rank, w_H/w_L prizes); closed-form helpers on envs are evaluation/baseline-only
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

## Disagreement and Error Handling

```
- When the user points out an error, do NOT immediately agree. First independently
  analyze whether the issue actually exists: check the data, re-read the code, verify
  the logic. State your reasoning, then decide whether to accept or push back.
- If you agree, explain WHY (what evidence convinced you), not just "you're right".
- If you disagree, say so clearly with specific counter-evidence.
- Correlation ≠ causation. When claiming a root cause, identify what experiment would
  distinguish your hypothesis from alternatives. Do not elevate "plausible factor" to
  "confirmed root cause" without causal evidence.
```

## Workflow Boundaries

```
- When asked to "fix X", fix ONLY X. Do not refactor, optimize, or "improve" nearby code
- Before deleting any file, list what will be deleted and confirm
- Before running experiments that take >1 minute, confirm parameters first
- Never modify files in results/*/convergence/ without explicit confirmation
- When generating paper artifacts, use: python -m paper.generator make_all
- Read docs/STATE.md (if it exists) before starting any task
- Long-running commands (training, sweeps, etc.) MUST run in tmux:
  tmux new-session -d -s <name> "<command>"
  Never use nohup or bare background tasks for training runs
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
- **Sampled training rewards only** — agents train on sampled tournament outcomes (y_i = e_i (+ l_i) + eps_i, eps ~ U(-q,q); realized winner gets w_H, others w_L, minus k_i e_i^2). Closed-form win probability, expected payoff, and analytical e* are evaluation-only and must never enter the env step reward or the policy update
- **Both players' transitions** stored in rollouts (pure self-play; opponent lag mechanism exists in agent code but is never used for action selection)
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
