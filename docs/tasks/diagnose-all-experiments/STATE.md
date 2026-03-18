# Diagnose All Experiments

Status: complete
Current phase: phase01

## What's done
- Created `tools/diagnose_all.py` with modular design (load/participation/convergence/integrity/theory checks)
- Independent P(win) implementation for cross-validation (fixed bug in 3-player conditional probability)
- Generated `diagnostic_report.md` with full results

## Key Findings

### Participation Constraint Violations
- **three_players q=25**: Invalid (gain=+1.90), same as two_players
- **three_players q=40**: Invalid (gain=+0.07), marginal but still violated — q_min for 3-player is higher than 2-player
- **different_cost q=25**: P2 (k=0.00055, higher cost) invalid (gain=+0.22), P1 valid
- **different_ability q=25**: Both players invalid (P1 gain=+0.40, P2 gain=+1.06, weaker player worse)

### Convergence
- different_cost: q=40 5/5, q=55 5/5, q=25 0/5 (NE invalid)
- different_ability: q=40 5/5, q=55 5/5, q=25 0/5 (NE invalid)
- three_players: q=55 3/3, q=40 2/3 (NE invalid anyway), q=25 0/5

### Data Issues
- three_players: seed field missing (all show 0), only 3 seeds for q=40/55 (vs 5 for others)
- three_players q=40 seed42: truncated (500/1500 updates), missing stop_reason
- three_players q=40 converged seeds: effort gap 7-9 units (much worse than other experiments)

### Cross-Validation
- prob.py: all checks passed (2-player, 3-player, different-ability)
- Theory consistency: JSON / theory.py / formula all match for every experiment and q

## What's next
- Issues found here feed into separate tasks (not this task's scope)
- q=25 participation constraint issue applies across ALL 4 experiment types — collaborator discussion covers all

## Blockers
- (none — diagnostic complete)
