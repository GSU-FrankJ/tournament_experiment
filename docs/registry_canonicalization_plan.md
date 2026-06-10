# Registry canonicalization plan — PREREQUISITE FOR `make_all` REGENERATION

**Status (2026-06-10): APPROVED per scenario and EXECUTED through V1.**
C2 = commit `5999165`, C3 = `58fcc26`, D1 = `d048418`, C1 + V1 acceptance test = `2044608`.
**E (`make_all` regeneration) remains BLOCKED on a separate owner go-ahead — not executed.**

**Owner decisions recorded with the approval:**
1. 3-player → `round3_baseline`, het-cost → `r4_dc_final`, het-ability → `r4_h1_long` — all
   three are closed-form-trained legacy, accepted as TEMPORARY picks solely so `make_all` is
   runnable; the `r5_sampled` wave replaces them.
2. da: `r4_h1_long` is accepted despite being standard-config (NOT theory-align-v2), purely as
   the cleanest complete group. **OPEN DECISION for r5_sampled: run het-ability BOTH
   theory-align-v2 and standard under sampled training and settle the config explicitly.**
3. Provenance footnotes that MUST accompany any artifact generated at E (also recorded as
   comments on `BASELINE_OVERRIDES` in `paper/generator/config.py`):
   - `round3_baseline` ran under the pre-unification exploit_eps=0.05 gate (measured
     exploitability at stop 0.0002–0.0047 also passes the unified 0.03);
   - `r4_dc_final` has no committed launch script (provenance = docs/round3_round4_report.md +
     per-JSON exploit_config + stops at the --min-updates 300 floor).

Date: 2026-06-10 · Branch: `fix/audit-remediation` · Companion: `FIX_CHANGELOG.md` §13-14,
`AUDIT_REPORT.md` §5.5.

---

## 1. How the registry selects runs today (evidence)

- `paper/generator/run_registry.py:206-357` scans `results/{exp}/convergence/*_convergence.json`
  (non-recursive) and classifies each file with precedence: `_metadata.json` →
  **JSON `ablation_name` field (:309-313)** → filename regex. `wh8_wl4` is split off as
  `weight_variant` (:283-286).
- `paper/generator/config.py:145-147`: `BASELINE_OVERRIDES = {}` (empty). When populated,
  `extract.promote_preferred_ablations` **drops** the rows labeled `baseline` for that
  `(experiment, q)` and **relabels** the preferred ablation as `baseline`, so all downstream
  code is untouched.
- Paper tables use only `ablation == "baseline"` (+ `weight_variant == "baseline"` in
  final_summary).

## 2. What is wrong with the current selection (verified read-only)

| # | Problem | Evidence |
|---|---------|----------|
| P1 | **3P collision:** pre-fix `ppo_3p_*_baseline_*.json` AND post-fix `ppo_3p_*_round3_baseline_*.json` both carry JSON `ablation_name: "baseline"` (the Round-3 batch used `--output-tag round3`, which changes only the filename). The registry therefore creates two runs with the identical key `(three_players, TEL-PPO, q, seed, "baseline")` — 10 collisions (5 seeds × q∈{35,55}) — and the loader's groupby silently **merges both trajectories into one run**. | duplicate-key scan: 10 × `('three_players', 'TEL-PPO', q, seed, 'baseline') x 2`; JSON dumps of both seed-42 files show `ablation_name='baseline'` |
| P2 | **dc/da canonical runs excluded:** Round-4 canonical runs are tagged `r4_dc_final` / `r4_h1_long` in their JSON, so the baseline rows currently come from the **pre-fix** Round-2 runs. | JSON dumps; `AUDIT_REPORT.md` §5.5 |
| P3 | **Set 1 / Set 2 merging:** run-level groupbys in `extract.get_convergence_step`, `extract.get_verified_convergence_step`, and `metrics.compute_summary_metrics` key on `(experiment, method, q, seed, ablation)` **without `weight_variant`**, so 2P Set 1 and Set 2 (wh8_wl4) rows merge per seed (18 apparent duplicate keys). final_summary filters weight_variant before its own stats, but conv/metrics aggregation upstream does not. | duplicate-key scan: `('two_players', …, 'baseline') x 2` for PPO (15) and Gradient (3) |
| P4 | **da q=55 Round-4 data uncommitted:** the 5 `r4_h1_long` q=55 JSONs and the updated `results/different_ability/summary.csv` are untracked — regeneration would rest on uncommitted inputs. | `git status` |
| P5 | **All gradient JSONs are stale by construction:** the 8 `gradient_*_convergence.json` files were produced by the closed-form FD solver replaced in commits a4b5aae / 8050f57 / 4d2812b / e2f614b. Worse, gradient filenames carry **no tag or seed**, so a re-run silently **overwrites** them. | filenames; FIX_CHANGELOG §5, §10-12 |
| P6 | **Every PPO JSON on disk is interim:** 3P/dc/da runs were trained on closed-form rewards (fixed in commits 24a8e6a/e5e1050/77ca27f) and 3P used eps=0.05. Canonicalizing them makes regenerated tables *internally consistent*, not *final*. | FIX_CHANGELOG §1-3, §7 |

## 3. The canonicalization — what gets registered/relabeled

**No results file is moved, renamed, or deleted.** Selection changes happen in
`paper/generator` config/code only; demoted runs stay on disk and remain visible as
non-baseline ablations.

| Experiment | Canonical baseline after this plan | Mechanism | Demoted (stays on disk) |
|---|---|---|---|
| two_players Set 1 | `ppo_q{35,45,55}.0_seed{42-46}_convergence.json` (Round-2 warmup-fix) | **no change** (already `baseline`) | — |
| two_players Set 2 | `*_wh8_wl4_*` | **no change** (`weight_variant`) | — |
| three_players | `ppo_3p_q{35,55}.0_seed{42-46}_round3_baseline_convergence.json` | (C2) registry tag-precedence rule reclassifies them as ablation `round3_baseline`; (C1) override promotes that to baseline | pre-fix `ppo_3p_*_baseline_*` (10 files; note seed-42 q35 was overwritten Apr 14 — pre-fix provenance already broken, see AUDIT §5.5) |
| different_cost | `*_r4_dc_final_*` (q∈{35,55} × seeds 42-46) | (C1) override `r4_dc_final` → baseline | pre-fix `*_baseline_*`, `dc_baseline_extra`, `dc_diag`, `r4_h1_only`, `r4_h1h2` |
| different_ability | `*_r4_h1_long_*` (q∈{35,55} × seeds 42-46; q=55 needs the data commit D1) | (C1) override `r4_h1_long` → baseline | pre-fix `*_baseline_*`, `da_diag`, `r4_h1_only`, `r4_h1h2` |
| gradient (all 4 experiments) | **none — must be re-run** with the new MC-FD solvers | n/a (P5) | current `gradient_*` files: archive or add output tags **before** re-running, else silent overwrite |

## 4. Change set (each item = one future commit; NOT executed)

**C1 — config: populate BASELINE_OVERRIDES** (`paper/generator/config.py:145`):
```python
BASELINE_OVERRIDES: Dict[Tuple[str, float], str] = {
    ("three_players", 35.0): "round3_baseline",
    ("three_players", 55.0): "round3_baseline",
    ("different_cost", 35.0): "r4_dc_final",
    ("different_cost", 55.0): "r4_dc_final",
    ("different_ability", 35.0): "r4_h1_long",
    ("different_ability", 55.0): "r4_h1_long",
}
```

**C2 — registry: filename-tag precedence + duplicate-key guard** (`run_registry.py`):
- If the JSON/metadata `ablation_name` is `"baseline"` but the filename parse yields a more
  specific non-weight-variant tag (e.g. `round3_baseline` from `--output-tag`), keep the
  filename tag. This is the minimal rule that separates the P1 collision; it changes nothing
  for files whose filename and JSON agree (2P, dc, da, diagnostics).
- After discovery, assert no two runs share `(experiment, method, q, seed, ablation,
  weight_variant)`; on collision, emit a loud warning naming both file paths (and exclude the
  older file by mtime) instead of silently merging.

**C3 — extract/metrics: add `weight_variant` to run-level groupbys** in
`get_convergence_step`, `get_verified_convergence_step`, `compute_summary_metrics` (fixes P3).

**D1 — data commit:** `git add` the 5 untracked
`different_ability_ppo_q55.0_seed{42-46}_r4_h1_long_convergence.json` + the modified
`results/different_ability/summary.csv` (data-only commit, separate from C1-C3 per repo rules).

**V1 — read-only dry run (no artifacts):** after C1-C3+D1, print per `(experiment, q)` the
selected baseline file paths, seeds, and `stop_reason`. Acceptance: exactly 5 seeds per cell,
all post-fix tags, zero duplicate keys, every `stop_reason == "exploitability"`.

**E — regeneration (BLOCKED ON YOUR APPROVAL):** `python -m paper.generator make_all`, then an
artifact-only commit. Even then the tables are **interim** (P6): they become final only after
the post-env-fix re-runs.

## 5. Re-run wave (context for what comes after E)

- Proposed tag for the post-env-fix PPO batch: `--ablation-name r5_sampled` on all four
  runners (5 seeds × all q, eps=0.03, max_updates=1500, theory-align-v2 + per-scenario
  stopping flags from `docs/round3_round4_report.md`). When blessed, BASELINE_OVERRIDES flips
  every entry (including two_players) to `r5_sampled`.
- Gradient re-runs need an overwrite-protection decision first (P5): either archive the old
  `gradient_*` files (requires explicit confirmation — results dir) or add a tag/seed to
  gradient output filenames.
- Fig-7 ablation batch: suggest `--ablation-name no_stability_screen` /
  `no_exploit_verification` so the registry keeps them distinct from baseline.
- Out of scope here: `_archive_pre_warmup_fix/` (hyperparam-sensitivity sources) — separate
  decision; does not affect Tables 3/4.
