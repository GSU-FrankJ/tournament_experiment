# STATE — r7-state4-wave

Status: **wave complete** (2026-07-31 17:39 → 2026-08-01 16:45 UTC, 23.1 h;
100/100 jobs, rc=0 everywhere; analysis done, polish pending owner go-ahead)

## Wave results (r7 4-dim vs r5 3-dim, raw endpoints, 5 seeds)

Accuracy verdict for red item 2: **no systematic degradation** — Δ|err| across
the 14 baseline cells is centered on zero (10 improved, 4 worsened, none
beyond seed noise except 2P S1 q35 +0.88 ≈ 1.1 SE). Recommendation: keep
4-dim for ALL scenarios.

| cell | r7 err | r5 err | Δ |
|---|---|---|---|
| 2P S1 q35/45/55 | 2.75 / 0.22 / 0.64 | 1.87 / 0.11 / 0.73 | +0.88 / +0.11 / −0.09 |
| 2P S2 q35/45/55 | 2.19 / 0.98 / 0.25 | 3.01 / 0.58 / 1.06 | −0.82 / +0.41 / −0.81 |
| 3P q35/55 | 1.34 / 0.45 | 2.01 / 0.60 | −0.66 / −0.15 |
| dc q35 P1/P2 | 0.48 / 0.02 | 0.32 / 0.42 | +0.16 / −0.40 |
| dc q55 P1/P2 | 0.17 / 0.30 | 0.14 / 0.27 | +0.03 / +0.02 |
| da std q35/55 | 2.28 / 0.51 | 2.44 / 0.67 | −0.16 / −0.16 |
| da v2 q35/55 | 1.08 / 0.41 | 1.50 / 0.10 | −0.41 / +0.30 |

- **da learned symmetry**: |e1−e2| med 0.44/0.49 (std), 0.77–1.15 (v2) —
  same order as cross-seed SD (0.38–1.98), 1–2% of e*. Honest claim: the
  free network reproduces effort symmetry to within seed-level noise.
- **2P S1 q35 mechanism (hypothesis)**: all 5 r7 seeds stop at update 49
  (r5: 49–69). M=16384 halves the verifier's maximization bias
  (~1.1e-2 → ~7.8e-3), so the 0.03 gate passes earlier → less training →
  landing farther below e*. Polish should absorb it; check after polishing.
- **Floor confound replicated** (expected, same flags): 3P streaks 150–209,
  dc 67–119 at stop. r6_floor_ablation.sh remains the needed experiment.
- **no_exploit instability moved cells** (by design, no verification): r5's
  bad cell was q45 (err 3.31), r7's is q55 (err 2.64) — supports the claim
  that budget-exhaustion landings are unstable without verification.

## Done

- 4-dim state implemented across both agents, four runners, the asymmetric
  exploitability evaluator, the MC adapters, and three diagnostic tools
  (full change table: phase01.md).
- Two-player training verifier M unified to 16384
  (`config/one_stage_two_players.py:110`).
- CPU sanity + 5-path GPU smoke passed (tag `r7smoke*`, seed 999; files left
  in results/*/convergence/, clearly tagged).
- Wave: 100 runs, tags `r7_state4[_std|_v2]`, `r7_fig7_no_{stability,exploit}`;
  seeds 42–46; manifest + per-job logs + code snapshot in `results/r7_state4/`.
  r5 files untouched (3-dim comparison arm).

## Next

1. Owner decision per red item 2 — recommendation: keep 4-dim everywhere
   (no accuracy cost measured).
2. MC-BR polish for r7 baselines AND both fig7 arms (dual-endpoint Table 3).
   NOTE: `tools/one_stage_polish_per_seed.py` writes
   `results/one_stage_ablation/polish_per_seed_all.json` — running it as-is
   would OVERWRITE the r5 polish rows. Needs r7 globs + a new output path
   (e.g. `polish_per_seed_r7.json`) before running. ~90 rows ≈ 3.2 h CPU.
3. `tools/unified_exploitability_tables.py` r7 variant; generator promotion
   (BASELINE_OVERRIDES → r7_*) only after owner signs off.

## Known issues

- Code changes uncommitted at launch; snapshot in
  `results/r7_state4/code_state.{txt,diff}`. Suggested split when committing:
  (1) feat: 4-dim state; (2) chore: verifier M 16384; (3) chore: wave script.
- r5 exploitability trajectories are not comparable to r7 for the two-player
  group (M 8192 vs 16384 changes the estimator floor ~1.1e-2 → ~7.8e-3).
- 5 `r7smoke*` JSONs remain in results/*/convergence/ (cleanup needs owner
  confirmation per repo rules).
