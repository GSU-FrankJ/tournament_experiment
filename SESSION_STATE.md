# SESSION_STATE.md — full project state for a fresh session (zero context loss)

Last updated: 2026-06-12 · **Branch: `fix/audit-remediation`** · HEAD at time of writing:
the cleanup-session docs commit following `9af59df` (`chore: re-render hyperparam_sensitivity
with corrected gate label`); run `git log --oneline -1` to confirm.

## What this whole effort did (one line per phase)

Read-only audit of the repo against the paper spec (`AUDIT_REPORT.md`) → fixed the core
spec violation (3P/dc/da trained on closed-form expected utilities; envs rewritten to SAMPLED
rank rewards with equivalence-in-expectation tests) → fixed the MC-FD baseline (sampled+CRN,
no symmetry projection, no e\*-dependent stopping), removed dormant e\*-in-loss paths, unified
eps_eq=0.03, fixed the all-NC table artifact (tables now report the method's OWN verified
stop) → registry canonicalization (tag-precedence, duplicate guard, weight-variant keys) →
**r5_sampled re-run wave** (160 runs, 0 failures: all scenarios + Set 2 + gradient baselines +
Fig-7 ablation arms, sampled training) → promoted r5 to canonical baseline and regenerated all
paper artifacts (final, not interim) → cleanup: retired the pooled `convergence_comparison`
table and two dormant plot paths; prepared (NOT launched) the sensitivity sweep.

## Adopted canonical state

- `BASELINE_OVERRIDES` (paper/generator/config.py): every (experiment, q) →
  `r5_sampled`, except het-ability → `("r5_sampled_std", "r5_sampled")` (PPO arm = STANDARD
  config — the sampled std-vs-v2 head-to-head REJECTED theory-align-v2 for da: equal-or-worse
  error with 3–14× seed variance; the v2 arm stays on disk as `r5_sampled_v2`).
- Honest r5 headline (all 5 seeds/cell, eps_eq=0.03, stop = the method's own
  stability+exploitability verification): **2P 4.12/1.54/2.74%** (q=35/45/55; Set 2
  6.33/1.57/3.49%), **3P 8.03/6.63%**, **het-cost 2.88/4.69%** (effort gaps 10.48/7.37 vs
  analytic 10.37/7.24), **da standard 5.25/3.26%**, **sampled MC-FD gradient 0.14–0.65%**
  everywhere with real ±std. Every PPO run has `stop_reason="exploitability"` except the
  Fig-7 no-exploit arm (max_updates by design).
- **The old sub-1% 3P/dc/da numbers are ABANDONED** — they were closed-form-training
  artifacts and are not reproducible under the spec. Do not quote them.
- Pipeline after cleanup: `python -m paper.generator make_all` → **9 figures + 4 tables**,
  green; `convergence_comparison` and the `exploitability_q25`/`beta_snapshots` plot paths
  are retired. The ONLY degenerate artifact is `hyperparam_sensitivity` (baseline-only data,
  pending the sweep below).

## THE ONE OPEN TODO: refill hyperparam_sensitivity

1. Launch the prepared sweep (owner go-ahead required; ~105 GPU runs ≈ 4–6 h on 8 GPUs):
   `bash run/r5_sensitivity_sweep.sh            # dry run: prints the full matrix`
   `bash run/r5_sensitivity_sweep.sh --launch   # starts 8 tmux workers (r5sens_gpu0..7)`
   IMPORTANT: the current figure consumes VERIFICATION-hyperparameter variants
   (`eps_001/eps_003/eps_010/eps_020`, `pat_01/pat_03/pat_10`) — NOT the old
   entropy/lr/clip grid (that belongs to the figure's pre-redesign era; available behind
   `--include-training-hparams` but not figure-consumed). Monitor:
   `results/r5_sensitivity/manifest.csv`; logs in `results/r5_sensitivity/logs/`.
2. When all 105 JSONs exist (`ls results/two_players/convergence/ppo_q*_{eps,pat}_*.json`):
   commit the results (data-only commit), then `python -m paper.generator make_all`.
3. Acceptance check: `paper/data/hyperparam_sensitivity.csv` must contain ablations
   {baseline, eps_001, eps_003, eps_010, eps_020, pat_01, pat_03, pat_10} × q∈{35,45,55} ×
   5 seeds; the figure must keep its exact 2×3 layout (ε row / patience row × q columns),
   fonts, colors, axes, legend placement — the ONLY visual delta vs the committed file
   (`9af59df`, which already carries the corrected "ε=0.03" baseline label) is the new data
   curves. Before/after compare against
   `git show 9af59df:paper/figures/hyperparam_sensitivity.png`. Commit artifacts separately;
   update PROVENANCE §4 item 1 (drop the UNVERIFIED flag).
   Known minor pipeline quirks (pre-existing, harmless): hyperparam_sensitivity is
   make_all-only (not in the per-figure PLOT_TYPES dispatch); global CLI flags must precede
   the subcommand (`python -m paper.generator --out-dir X make_all`).

## Still pending after that, in order

1. **Phase 2 of the paper-refill format verification:** before/after checklist for every
   file in paper/figures + paper/tables (8 figures + 4 tables are already r5-correct from
   commit `693e74c` + cleanup; sensitivity is the last refill). Compare each against its
   pre-r5 counterpart (`git show a82c92e:<path>`): same layout/axes/fonts/legends, only data
   changed.
2. **Repo archive/reorg:** move (NEVER delete) legacy results and DELETE-bucket files from
   `AUDIT_REPORT.md` §3 into `_archive/`, leaving a single current-data folder. PLAN FIRST,
   owner approves the move list before anything moves.
3. **Manuscript pass on main.tex** (owner-led; main.tex is NOT in this repo): §5.3 rewrite
   around the verify-but-8%-off 3P finding — the eps_eq utility-vs-effort mechanism note is
   written for this in `paper/PROVENANCE.md` §5; Tables 3/4 number swap from
   `paper/tables/final_summary.*`; da-standard justification (sampled evidence, see
   `docs/registry_canonicalization_plan.md` owner-decision block).
4. **Optional:** two-stage extension (env+config exist, runner never built; the two-stage
   config still encodes denominator-6 — reconcile before building; see AUDIT_REPORT §3).

## Key provenance pointers

- `paper/PROVENANCE.md` — every table cell/figure → r5 run files + seeds + launch command;
  UNVERIFIED list; §5 = the eps_eq/§5.3 mechanism note.
- `FIX_CHANGELOG.md` — §1–22, every change of the whole effort with commits and evidence.
- `docs/registry_canonicalization_plan.md` — canonicalization design + owner decisions
  (incl. the resolved da std-vs-v2 item).
- `AUDIT_REPORT.md` — the original audit (KEEP/FIX/DELETE/BUILD classification of every file).
- r5 data: `results/{two_players,three_players,different_cost,different_ability}/convergence/`
  `*r5_*` (committed in `d6a6e81`); wave manifest/queues/logs in `results/r5_sampled/`;
  Fig-7 collection in `results/ablation/r5_fig7/`.
- Repeatable registry acceptance check: `python3 tests/test_registry_canonicalization.py`
  (plus 6 env/gradient equivalence tests in `tests/`).

## Hard rules that carry over (non-negotiable)

1. **No fabricated or estimated numbers, ever** — every reported value traces to a run file.
2. **Archive by moving, not deleting** — results are precious and irreproducible; any move
   into `_archive/` is plan-first, owner-approved.
3. **Never overwrite gradient/result files** — gradient runners refuse without `--force`;
   PPO runs have no guard, so always use fresh `--ablation-name` tags (the sensitivity
   script's preflight refuses pre-existing targets).
4. **Figure formats must not change** — same scripts, layout, styling, fonts, colors, axes,
   legends; data swaps only. (Only exception ever made: factually wrong text, e.g. the
   ε=0.05 legend label, with owner-visible logging.)
5. **Don't touch main.tex without the owner's say-so** (it lives outside this repo anyway).
6. Repo conventions: tmux for anything long-running; conventional commits; one logical
   change per commit; never mix code changes with experiment reruns or artifact regen.
