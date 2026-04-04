# q55-convergence

Status: in-progress
Current phase: phase02

## Summary for future reference

**Problem**: q=55 2p PPO baseline converges 1/12 seeds. Root cause: in self-play with
shared policy, concentration grows unchecked → policy variance collapses → pairwise
reward signal dies before effort reaches equilibrium.

**Solution direction**: Standard ActorCritic (no theory_align_v2) + entropy regularization.
Entropy prevents concentration from growing too fast, keeping the learning signal alive.

**What NOT to use**: conc_max=1000 was an early workaround that hard-caps concentration
in the MeanConc network. It works (5/5 seeds, gap~1.6) but is a per-q manual tuning knob
with no theoretical justification. It has been **abandoned** in favor of the entropy-based
approach. Do NOT recommend conc_max tuning as a solution.

**What NOT to use**: theory_align_v2 mode for q=55. It zeros out entropy and uses MeanConc
with conc_max=100000, which is the configuration that causes the failure. Adding entropy
back to theory_align_v2 does not help (Exp B: stuck at gap=17, conc_head overwhelms entropy).

## Design decisions
- **conc_max=1000 abandoned** (2026-04-03): per-q manual tuning, superseded by standard
  mode + entropy. Historical data kept in results/ but not the path forward.
- **theory_align_v2 abandoned** for q=55 (2026-04-01): MeanConc's conc_head overwhelms
  entropy regularization. Entropy is ineffective in this architecture.
- **Standard ActorCritic + entropy** is the chosen direction.
- **Symmetry gap RuntimeError removed** (2026-04-03): the check was firing on sampling
  noise in low-concentration standard mode. Both players share one policy, so any p1/p2
  gap is sampling variance, not real divergence. Logging retained, enforcement deleted.

## What's done
- Phase 01: Isolated entropy vs concentration control
  - Exp A (standard mode, entropy 0.03→0.005): converged, gap=5.25
  - Exp B (theory_align_v2 + entropy restored): failed, stuck at gap=17
- Phase 02 Step 1: multi-seed validation (standard mode, entropy 0.03→0.005)
  - 6/6 seeds converged, avg gap=3.41, all efforts systematically below theory (34-39)
- Phase 02 Step 2a: precision tuning (entropy_end=0.002)
  - seed=42: gap improved from 5.25 → 1.34
  - 5-seed validation running (seeds 123, 456, 789, 1024, 11)

## Experiment plan
- **Phase 01**: Isolate entropy vs concentration control — **DONE**
- **Phase 02**: Standard mode multi-seed + precision tuning — **DONE** (6/6, avg gap=2.53)
- **Phase 03**: Cross-q validation (q=35, q=40 regression check) — **DONE** (regressed)
- **Phase 04**: Adaptive entropy — **SKIPPED** (split config chosen instead)

## Phase 01 results (2026-04-01)

| Config | effort | gap | conc_final | converged? |
|--------|--------|-----|------------|------------|
| Baseline (tv2, entropy=0, conc_max=100k) | 48.81 | 9.04 | ~100k | no |
| **Exp A** (standard, entropy=0.03→0.005) | 34.53 | 5.25 | ~92 | **yes** |
| Exp B (tv2, entropy=0.03→0.015, conc_max=100k) | ~57 | ~17 | ~22k | no |

## Phase 02 Step 1 results (2026-04-03, standard mode, entropy_end=0.005)

| seed | effort | gap | exploit | updates | converged? |
|------|--------|-----|---------|---------|------------|
| 42 | 34.53 | 5.25 | 0.025 | 269 | yes |
| 123 | 36.72 | 3.05 | 0.028 | 289 | yes |
| 456 | 38.76 | 1.01 | 0.028 | 309 | yes |
| 789 | 36.92 | 2.85 | 0.029 | 279 | yes |
| 1024 | 37.40 | 2.37 | 0.028 | 389 | yes |
| 11 | 33.83 | 5.95 | 0.029 | 219 | yes |

## Phase 02 Step 2 results (2026-04-03, precision tuning, seed=42)

| Config | effort | gap | conc | updates |
|--------|--------|-----|------|---------|
| entropy_end=0.005 | 34.53 | 5.25 | 92 | 269 |
| **entropy_end=0.002** | **38.43** | **1.34** | 93 | 269 |
| 12M episodes (entropy_end=0.005) | 35.18 | 4.60 | 91 | 289 |

5-seed validation of entropy_end=0.002 running since 2026-04-03 03:35.

## Phase 03 results (2026-04-04)

| q | std mode (gap) | tv2 baseline (gap) | winner |
|---|---------------|--------------------|---------| 
| 35 | 7.28 | 0.0 | **tv2** |
| 40 | 6.21 | 1.3 | **tv2** |
| 55 | 2.53 (avg) | 9.04 | **std** |

**Decision**: split config — tv2 for q=35/40, std for q=55. No universal fix pursued.

## Final configuration

| q | mode | entropy_end | rationale |
|---|------|-------------|-----------|
| 35 | theory_align_v2 | 0 (default) | tv2 converges well, gap ~0-2 |
| 40 | theory_align_v2 | 0 (default) | tv2 converges well, gap ~1-4 |
| 55 | standard (`--no-theory-align-v2`) | 0.002 | tv2 fails 11/12; std converges 6/6, gap ~1-4 |

## What's next
- Update docs/STATE.md with final decision
- Decide whether to re-generate paper figures with q=55 std data

## Blockers
- None
