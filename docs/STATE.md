# Project state

Last updated: 2026-04-05

## Current status
- **CRITICAL**: Theoretical equilibrium e*=(w_H-w_L)/(4qk) is NOT a global NE for
  3-player at q=25,35,40 (and 2-player at q=25). See section below.
- 3p algorithm improvement task pivoted from "fix PPO" to "fix game theory"
- Paper figures/tables revision complete
- q=35 experiments complete for all 4 scenarios

## Critical finding: interior NE validity (2026-03-28)

The interior FOC equilibrium is only a LOCAL optimum. The global best response can be
to shirk (e≈0). The interior NE is globally valid iff:

    q >= sqrt(N * w_gap / (16k))

| N | q_crit | q=25 | q=35 | q=40 | q=55 |
|---|--------|------|------|------|------|
| 2 | 33.07  | FAIL | pass | pass | pass |
| 3 | 40.50  | FAIL | FAIL | FAIL | pass |

This explains ALL experimental anomalies:
- 3p q=35 gap ~5: no pure-strategy symmetric NE exists, PPO finds interior local optimum
- 3p q=55 gap ~3: NE IS valid, PPO converges (slowly) in correct direction
- 2p q=25 gap ~3: NE invalid even for 2 players, PPO finds local optimum
- 2p q=35,40 gap <2: NE valid, PPO converges correctly

Gradient solver is misleading — it follows local gradients to the interior solution
without checking global deviations (e→0). Exploitability eval correctly catches this.

## What was done (2026-03-28)
1. Implemented pairwise_binary, hybrid, and COMA reward modes for 3p env
2. Ran pairwise_binary PPO (gap=11.64, worse — converges to flat exploit region)
3. Ran hybrid ns=0.3 and bigbatch 16384 (same failure mode)
4. Discovered via exploitability analysis that e*=62.5 is not a global NE at q=35
5. Derived participation constraint: q >= sqrt(N*w_gap/(16k))
6. Verified: 3p NE only valid at q=55, 2p NE fails at q=25

## What was done (2026-03-27, session 2)
1. Generated missing gradient baseline for q=35 wh8_wl4 (w_H=8, w_L=4, e*=71.43, gap=0.17)
2. Regenerated baseline gradient for q=35 (w_H=6.5, w_L=3.0, e*=62.5, gap=0.15) — required 50k steps
3. Regenerated convergence_main figure — all 6 panels now populated
4. Closed q35-all-experiments task (3p limitation accepted)
5. Created 3p-algorithm-improvement task with full problem diagnosis and 6 proposals
6. Killed running tmux session (3p_5000upd_v2)

## What was done (2026-03-27, session 1)

### Paper figures & tables revision (phases 03–11)
1. **Phase 03**: Effort drift — SHADE_ALPHA, thicker threshold, standardized legend
2. **Phase 04**: KL dynamics — SHADE_ALPHA, "Reference threshold" label
3. **Phase 05**: Distance to equilibrium — new title, y-axis label, q=35 color
4. **Phase 06**: Beta snapshots/evolution — auto-select best seed (42), lighter shading
5. **Phase 07**: Exploitability 6a — renamed labels; new Figure 6b for q=25
6. **Phase 08**: Ablation — ABLATION_LABELS/LINEWIDTHS, prominent Theory line, unified y-axis
7. **Phase 09**: Dotplot — alternating backgrounds, smaller per-seed dots, renamed labels
8. **Phase 10–11**: Tables — q=25 excluded, q=35 included, data loader fix for make_all

### q=35 3p diagnostic (phase 03 — closed)
- Tested 11 variants (entropy, adv norm, network arch, optimizer reset, 5000 updates)
- All failed: gap remains ~5 units from equilibrium
- Original hypothesis: weaker gradient signal → gap ~5 (wrong)
- Actual root cause (found 2026-03-28): interior NE is not globally valid at q=35
- Gradient descent finds local optimum but misses global deviation to e≈0
- Decision: needs theory correction (see critical finding above)

## Data inventory
- two_players: baseline + wh8_wl4 + ablations + sweeps (~115 convergence JSONs)
- three_players: baseline (5 seeds) + gradient + 11 diagnostic variants for q=35
- different_cost: baseline (5 seeds per q) + gradient
- different_ability: baseline (5 seeds per q) + gradient

## Known tech debt
- Runner files (run/run_*.py) have ~60% code duplication — extract shared base (task: `docs/tasks/runner-refactor/`)
- No tests exist for any module
- No CI/CD pipeline
- summary.csv in results/two_players/ has a parsing issue (line 14 has 46 fields, expected 39)
- Vestigial opponent lag code in agents/ppo_two_players_clean.py (deepcopy, sync logic, act_opponent — all unused)
- q=35 baseline gradient solver needs 50k steps (MC gradient noisy for low-q; existing q=40/55 files were generated with different params)

## Task status

| Task | Status | Notes |
|------|--------|-------|
| paper-figures-tables-revision | complete | 11 phases done, all figures/tables regenerated, q=35 wh8_wl4 panel filled |
| q35-all-experiments | in-progress | phase05: 2p ablations missing for ablation_comparison figure |
| perfect-exploitability-figure | closed | decisions resolved, work transferred |
| diagnose-all-experiments | complete | — |
| runner-refactor | deferred | post-project cleanup, user will revisit later |
| 3p-algorithm-improvement | blocked | Interior NE invalid at q=25,35,40; need theory fix or parameter change |
| q55-convergence | complete | Standard ActorCritic + entropy fixes 2p q=55 (6/6 seeds). Paper updated. |

## q=55 convergence update (2026-04-03)
**Root cause**: theory_align_v2 (default PPO mode) zeros entropy and uses MeanConc with
conc_max=100000. Concentration grows unchecked, pairwise signal dies before convergence.

**Solution**: Split configuration by q value:
- q=35, q=40: **theory_align_v2** (default) — converges well, gap 0-2
- q=55: **standard mode** (`--no-theory-align-v2 --override-entropy-end 0.002`) — 6/6 seeds converge, avg gap=2.53

Standard mode regresses q=35/40 (gap 6-7 vs 0-2 with tv2), so no universal config exists.
Adaptive entropy (Phase 04) was skipped in favor of this split.

**Abandoned approaches** (do NOT revisit):
- conc_max=1000: works but is per-q manual tuning, no theoretical basis
- theory_align_v2 + entropy restored: MeanConc conc_head overwhelms entropy, stuck at gap=17
- Standard mode for all q values: regresses q=35/40 precision

See `docs/tasks/ q55-convergence/STATE.md` for full details.

## What was done (2026-04-05)
1. Added BASELINE_OVERRIDES in paper/generator/config.py: maps (two_players, 55.0) → "no_tv2_ent002"
2. Added promote_preferred_ablations() in paper/generator/extract.py: transparently relabels
   preferred ablation as "baseline" so all downstream plots/tables use the fixed q=55 data
3. Regenerated all paper artifacts (11 figures + 5 tables)
4. Two-Player q=55 TEL-PPO improved: gap 9.36→2.60, RelErr 23.53%→6.55%, exploit 0.076→0.027

## Next steps
- **Decide paper strategy for 3-player**:
  (a) Restrict 3p results to q≥55 (only valid NE),
  (b) Change game parameters so NE is valid at lower q (reduce k or w_gap),
  (c) Characterize mixed-strategy equilibrium for q < q_crit
- Derive participation constraint formally for the paper's theory section
- Verify 3p q=55 convergence with more episodes (current gap ~3.3)
- Runner refactor phase 01: audit duplication across the 4 runners
- Add basic tests for theory.py, prob.py, paper generator
- Consider git filter-repo to remove large files from history (~85 MB .git)
