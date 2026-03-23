# q=35 all experiments

Status: in-progress
Current phase: phase02 (3p diagnostic)

## What's done

### Phase 01: Training runs

**Gradient — all 3 converged**
- three_players: gap=0.125, different_cost: gap=0.066, different_ability: gap=0.222

**PPO — different_cost: DONE (gaps 1-3, acceptable)**

| Seed | e1 | e2 | Gaps | Exploit | Stopped |
|------|-----|-----|------|---------|---------|
| 42 | 48.93 | 35.43 | 1.33/1.12 | 0.048 | @174 |
| 123 | 48.83 | 35.81 | 1.44/0.75 | 0.047 | @170 |
| 456 | 49.10 | 35.47 | 1.16/1.08 | 0.046 | @184 |
| 789 | 47.48 | 34.79 | 2.78/1.77 | 0.047 | @181 |
| 1024 | 48.97 | 35.98 | 1.30/0.58 | 0.050 | @181 |

**PPO — different_ability: DONE (gaps 1.8-2.8, acceptable with exploit_eps=0.02)**

| Seed | Effort | Gap from 58.04 | Exploit | Stopped |
|------|--------|----------------|---------|---------|
| 42 | 55.25 | 2.79 | 0.019 | @504 |
| 123 | 56.20 | 1.83 | 0.020 | @561 |
| 456 | 55.55 | 2.48 | 0.019 | @539 |
| 789 | 55.35 | 2.69 | 0.019 | @562 |
| 1024 | 55.24 | 2.80 | 0.018 | @622 |

**PPO — three_players: NOT ACCEPTABLE (gaps 5.0-5.7)**

| Seed | Effort | Gap from 62.5 | Exploit | Stopped |
|------|--------|---------------|---------|---------|
| 42 | 57.06 | 5.44 | 0.171 | @1500 (max) |
| 123 | 56.91 | 5.59 | 0.124 | @1500 (max) |
| 456 | 57.49 | 5.01 | 0.171 | @1500 (max) |
| 789 | 57.53 | 4.97 | 0.173 | @1500 (max) |
| 1024 | 56.82 | 5.68 | 0.141 | @1500 (max) |

### Phase 02: Root cause analysis

**Eliminated hypotheses:**
- H1 (initial entropy collapse): 3p q=25 seeds 123-1024 started concentrated (ent=-3.7), still gap=6.0
- H4 (opponent lag): Code audit shows 2p also uses pure self-play (act_opponent() never called)

**Initial hypothesis: entropy bonus overpowers weak 3p gradient signal**
1. 3p per-transition gradient signal is 27% weaker (rank-order loser cancellation)
2. entropy_coef (tuned for 2p) too strong for 3p → policy deconcentrates
3. 3p q=25 seed=123: entropy went from -3.69 → -2.75 (UP, deconcentrated!)
4. 2p q=35: entropy went from -1.86 → -4.68 (DOWN, concentrated normally)
5. Less concentrated → weaker ∇log π → KL→0 → policy frozen at effort≈NE-5.5

**Experiment A: entropy_coef_end sweep — FAILED (2026-03-22)**

3 entropy_end values, seed=42, q=35, exploit_eps=0.02. All hit max updates (1500).

| entropy_end | Final effort | Gap | Entropy | Exploit |
|-------------|-------------|-----|---------|---------|
| 0.0005 | 57.10 | 5.40 | 0.0005 | 0.190 |
| 0.001 | 57.11 | 5.39 | 0.0010 | 0.190 |
| 0.003 | 57.09 | 5.41 | 0.0030 | 0.189 |

Conclusion: lowering entropy_end has zero effect. Gap and exploit are identical across all 3 values.

**Experiment B: gradient steps per update — FAILED (2026-03-23)**

Two variants to reduce gradient steps from 72→48 (matching 2p), seed=42, q=35, exploit_eps=0.02.
Added `--steps-per-update`, `--minibatch-size`, `--update-epochs` CLI flags to runner.

| Variant | Change | Final effort | Gap | Exploit | Status |
|---------|--------|-------------|-----|---------|--------|
| spu2731 | steps_per_update: 4096→2731 | 56.36 | 6.14 | 0.142 | @1500 (max) |
| mbs1536 | minibatch_size: 1024→1536 | ~56.26 | ~6.24 | ~0.173 | killed @1214 |
| baseline | (no change) | 57.05 | 5.44 | 0.171 | @1500 (max) |

Conclusion: reducing gradient steps per update **worsened** convergence (gap 6.1-6.2 vs 5.4 baseline). Over-optimization is not the root cause.

## What's running now

Nothing. All experiments stopped.

## What's next

Both experiment A (entropy) and B (gradient steps) failed. Remaining options:
1. Combine A+B (low confidence — both individually had no/negative effect)
2. More aggressive changes: different advantage normalization, 1 transition per step
3. Accept 3p gap≈5.4 as a known PPO limitation for rank-order tournaments
4. Try conc_max cap (worked for 2p q=55)

## Blockers
- No viable fix found for 3p convergence gap — user decision needed on next approach
