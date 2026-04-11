# Phase 01: Round 1 — Quick Validation

## Objective
Test 3 configs on q=55 (1 seed) to identify which intervention reduces the convergence gap below 3.0 effort units.

## Theory-Derived Bound

L = ⌈√(W/(2k))⌉ — the maximum effort at which symmetric participation remains individually rational.

- Derivation: EU = w_L + W/2 − k·e² ≥ w_L ⟹ e ≤ √(W/(2k))
- Set 1 (W=3.5, k=0.00055): L = ⌈√(3181.8)⌉ = ⌈56.41⌉ = **57**
- This bound depends only on (W, k). It does NOT depend on e* or q.
- All e* values are well below L: e*(35)=45.45, e*(45)=35.35, e*(55)=28.93

## Configs

| Label | Command | What it tests |
|-------|---------|---------------|
| R1a | `python run/run_two_players.py --method ppo --q 55 --seed 42 --effort-range 0 57 --episodes 6144000` | Theory bound only |
| R1b | `python run/run_two_players.py --method ppo --q 55 --seed 42 --effort-range 0 57 --override-entropy-end 0.0005 --episodes 6144000` | Theory bound + lower entropy |
| R1c | `python run/run_two_players.py --method ppo --q 55 --seed 42 --override-entropy-end 0.0005 --episodes 6144000` | Lower entropy only (control) |

## Execution
- Run in tmux sessions (per CLAUDE.md rules for long-running jobs)
- All 3 can run in parallel if GPU/CPU allows
- Expected wall time: ~2 hours each
- Use `--variant-name` to distinguish: e.g., `--variant-name L57` for R1a

## Verification
After runs complete, compare final gaps:
```bash
python3 -c "
import json, glob
for label, pattern in [
    ('R1a (L=57)', '*q55.0*L57*convergence.json'),
    ('R1b (L=57+ent)', '*q55.0*L57*ent*convergence.json'),
    ('R1c (ent only)', '*q55.0*entropy_end_0.0005*convergence.json'),
    ('baseline', '*q55.0*seed42*entropy_end_0.002*convergence.json'),
]:
    files = glob.glob(f'results/two_players/convergence/{pattern}')
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        gap = d['final']['gap']
        effort = d['final']['effort']
        print(f'{label}: effort={effort:.2f}, gap={gap:.2f}, e*=28.93')
"
```

## Go/No-Go
- ANY config with gap < 3.0 → proceed to Phase 02 (Round 2) with that config
- ALL configs gap > 3.0 → reassess; consider Proposal 4 (AEC)
- If R1a < 3.0 and R1b < 3.0 → prefer R1a (simpler)

## Files to modify
None. CLI flags only.
