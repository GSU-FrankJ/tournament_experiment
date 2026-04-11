# q=45/55 Convergence Fix

## Goal
Reduce TEL-PPO convergence gap for q=45 and q=55 from current 12%/26% to |e-e*| < 2.0 effort units, matching q=35 performance.

## Root Cause
High-q reward landscape is flat — policy gradient enters "exhaustion zone" (|dEU/de| < 0.005) at e ≈ 33-35, still 4-7 units above e*. See F2 figure and `q55_fix_proposals.md` for full analysis.

## Approach
Proposal evaluation in `q55_fix_proposals.md` identified **theory-derived effort bound** (Proposal 5) as the highest improvement/cost option.

**Key formula:** L = ⌈√(W/(2k))⌉ — the maximum rational effort at symmetric play. Depends only on (W, k), NOT on e* or q. For Set 1: L=57.

Experiment plan:
- Round 1: 3 configs × 1 seed × q=55 (quick validation)
- Round 2: winning config × 5 seeds × q={35,45,55} (includes regression)

## Scope
- Modify: run commands only (CLI flags `--effort-range`, `--override-entropy-end`)
- Do NOT modify: game formulation, reward function, environment code, theory
- Results go to: `results/two_players/convergence/` with appropriate naming

## Key Files
- `q55_fix_proposals.md` — full proposal evaluation (this directory)
- `run/run_two_players.py` — CLI entry point (`--effort-range` flag at line 2075)
- `config/one_stage_two_players.py` — default hyperparameters
- `agents/ppo_two_players_clean.py` — PPO trainer
- `paper/generator/output/figures/F2_eu_landscape.pdf` — gradient exhaustion visualization

## Go/No-Go Criteria
- Round 1 → Round 2: q=55 gap < 3.0 effort units
- Round 2 pass: mean gap < 2.0 for q={45,55} AND gap < 1.5 for q=35
- Fail: all Round 1 configs fail → escalate to AEC (Proposal 4)
