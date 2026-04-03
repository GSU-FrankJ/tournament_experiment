# q55-convergence

Status: in-progress
Current phase: phase01

## What's done
- Diagnosed root cause: self-play signal collapse from premature concentration growth
- Verified conc_max=1000 fixes 2p q=55 (5/5 seeds, gap 0.5-3.5)
- Confirmed 2p and 3p share same equilibrium formula e*=(w_H-w_L)/(4qk) — not a bug
- Generated split exploitability plots (conc1000 vs baseline)
- Evaluated patience=3 vs 5: patience=3 worse (false positives at gap ~9)
- Reviewed three concentration control proposals (variance floor, progress-aware, signal-quality)
- First-principles analysis: adaptive entropy is the most natural mechanism
- Reviewed ChatGPT's three-layer adaptive entropy design; identified over-engineering risk

## Design decision: incremental experiments before adaptive entropy
ChatGPT proposed a three-layer architecture (safety gate + SAC-style controller + PPO). Before building that, we need to answer a simpler question: does just NOT decaying entropy fix q=55? This determines whether adaptive entropy is needed at all.

## Experiment plan
- **Phase 01**: Fixed high entropy — does entropy_coef=0.03 (no decay) fix q=55?
- **Phase 02**: Cross-q validation — does fixed high entropy break q=35/40?
- **Phase 03**: (conditional) Simple adaptive entropy if Phase 01-02 show different q needs different entropy
- **Phase 04**: (conditional) Add pairwise signal gate if Phase 03 oscillates

## Phase 01 results (2026-04-01)

| Config | effort | gap | conc_final | converged? |
|--------|--------|-----|------------|------------|
| Baseline (tv2, entropy=0, conc_max=100k) | 48.81 | 9.04 | ~100k | no |
| **Exp A** (standard, entropy=0.03→0.005) | 34.53 | 5.25 | ~92 | **yes** (update 269) |
| **Exp B** (tv2, entropy=0.03→0.015, conc_max=100k) | ~57 | ~17 | ~22k | no (stuck from update 51) |
| conc_max=1000 (tv2, entropy=0, conc_max=1k) | 39.23 | 0.54 | 1000 | **yes** |

**Key findings:**
- Entropy regularization works in standard mode but not in MeanConc (conc_head overwhelms entropy)
- Standard mode converges but overshoots below equilibrium (effort 34.5 vs theory 39.77, gap=5.25)
- MeanConc + conc_max=1000 remains the best accuracy (gap=0.54)
- The two architectures have fundamentally different dynamics; entropy is not a universal fix

## What's next
- Phase 02 Step 1: multi-seed validation (5 seeds, standard mode)
- Phase 02 Step 2: precision tuning if Step 1 passes

## Blockers
- None
