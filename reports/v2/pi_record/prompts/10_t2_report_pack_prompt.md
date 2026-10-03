# Build the T=2 v2 report pack (P0–P6): find or generate every table, figure and number

This round builds the **materials** for a report on the T=2 v2 work, from the Phase 0 audit through the fresh-seed confirmation. Collect or generate every table, figure and headline number the report needs, with full provenance.
- **You do not write the report prose.**
- T=3 and the paper are out of scope.

**Where to work.** The canonical worktree `/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2`, on branch `v2-stagewise-pilots`. At the start, record HEAD as the pack's **base commit**.

**Scope.** §3 lists the items, grouped by report section. §5 gives PI-supplied content that is not in the repository. Build everything, report back (§6), then **STOP**.

---

## 0. Ground rules

### Read-only
- Do not modify, move or delete any existing file under `results/`, `reports/`, `protocols/`, `run/`, `utils/`, `agents/`, `envs/`, `tools/` or `tests/`.
- Write only to two new locations:
  - `reports/v2/t2_report/`, the pack;
  - `tools/v2/report/`, the scripts that build it.

### No training
Allowed computation:
- reading saved files, and recomputing summaries from per-run records;
- evaluating the closed form and `lr_at`;
- forward passes of saved weights, for example ê₁(0) from the weight exports;
- verifier evaluations of saved or analytic policies, **only** when an item cannot be built from saved arrays.

Log every forward pass and verifier evaluation in `reevaluations.csv`: the policy source, the tier, the commit, and the wall time.

Run everything single-threaded (`OMP/MKL/OPENBLAS_NUM_THREADS=1`) with the `.venv` Python. Check the load first, and do not kill jobs you did not start.

### Prefer what exists
- If a table or figure already exists (a CSV, a report table, a PNG), use it, and record its path and SHA-256.
- If a report table has an underlying CSV, cite the CSV. Cite the report text only when no data file exists, and mark it "source: report text".
- To restyle a figure, plot it from the same data, and confirm the plotted values match the original data file.
- Places to look:
  - `reports/v2/*.md` and `reports/v2/figures/`;
  - `results/v2_pilots/*/analysis/`, `results/v2_pilots/induced_band/`, `results/v2_pilots/phase2_regression/`;
  - `results/v2_T2_locked/` (`calibration/`, `consolidation/`, `rehearsal_analysis/`, `cusp_diagnostic/`, `v1_1/`, `rehearsal_v1_1_analysis/`, `confirmation_analysis/`, and the per-run folders);
  - `protocols/`.

### Provenance
Every value in every table, figure and key number must trace to a file. For each item, record:
- its source paths, with SHA-256;
- the script and commit that built it;
- any transformation applied.

No estimates. If a value cannot be determined, write `UNKNOWN` and explain why in `gaps.md`.

### Cross-check
Where a pack value also appears in an existing report, compare the two. List every mismatch in `consistency.md`, and do not edit the old report.

### Commit
Commit the pack and its scripts: CSV, MD, PNG/PDF, and scripts only. Do not commit large arrays or checkpoints. Do not push.

---

## 1. Pack layout

```
reports/v2/t2_report/
  README.md            the outline in §3, with each item's status and links
  manifest.csv         one row per item (§1.1)
  key_numbers.csv      the headline numbers (§4)
  data_dictionary.csv  every column used in a pack table: definition, units, normalization, tier, source function
  tables/              T##_<slug>.csv and T##_<slug>.md
  figures/             F##_<slug>.pdf, F##_<slug>.png (300 dpi), F##_<slug>_data.csv, F##_<slug>_caption.md
  data/                consolidated per-run tables D01–D09
  gaps.md              items not found and not generable, with reasons
  consistency.md       mismatches against existing reports
  reevaluations.csv    every forward pass or verifier evaluation performed
tools/v2/report/
  build_t2_report_pack.py   one entry point that rebuilds the whole pack from results/
  style.py                  the shared figure style
```

### 1.1 `manifest.csv`

Columns: `id, section, title, type (table/figure/data/number), priority (core/supp), status, files, sources (path:sha256; …), script, commit, notes`.

| status | meaning |
|---|---|
| `found` | an existing file, used as is (copied into the pack, or referenced if it is large) |
| `regenerated` | rebuilt from existing data for a consistent style; matches the original |
| `generated` | new, from saved data or allowed computation |
| `derived` | computed from other pack items |
| `missing` | could not be found or generated; explained in `gaps.md` |

---

## 2. Conventions

- **q** is always reported separately (50 and 60). Every per-run table lists its seeds explicitly.
- **Evaluated policy:** the deterministic Beta mean.
- **Tier:** final, unless stated otherwise. Label every dev-tier value.
- **Normalization:**
  - deviation metrics (η₂, Δ, Ĝmax_full, EXP_root, dReach, Δmax_all, dFull) are divided by ΔW;
  - recovery errors are fractions of e₁*(0) or e₂*(0), signed unless stated otherwise;
  - raw effort values are in effort units [0, 100], labelled "raw".
- **Final checkpoint of each study:**

  | Study | Final checkpoint |
  |---|---|
  | Pilot 1 | u400 |
  | Pilot 2, Pilot 3 | global u1000 |
  | Phase A extension | u400, u800, u1200, u1600 |
  | Pilot 4 §2a | u1600 |
  | Pilot 4 §2b | u2200 |
  | Locked runs | end of A (u1600) and end of B (u2200) |

- **Across-seed summaries:** the median and the IQR (25th–75th percentile, numpy linear interpolation), plus the min and max.
- **Paired statistics:** reuse the existing bootstrap results, and state their resamples and seed. Recompute only if missing: 10,000 resamples, with the seed stated.
- **Figures:**
  - one shared style in `style.py`: fixed colours per arm and per q across all figures (document them in the README), axis labels with units, legends that state n;
  - 7 in wide, with no text smaller than 8 pt at that size;
  - PDF plus a 300 dpi PNG;
  - each figure gets a data CSV and a factual caption (what is plotted, data source, n, tier), with no interpretation.

---

## 3. Items, by report section

- `[core]` items go in the main text, `[supp]` items in the appendices.
- **Part I** follows the PI's 0930 plan (§5.2). **Part II** covers the extensions beyond it, grouped by theme.

### Front matter

- **T01 [core] Study inventory.** One row per study:
  - Phase 0 audit;
  - Phase 1 verifier;
  - Phase 2 infrastructure and opening checks;
  - dReach reach-mask check;
  - Pilot 1;
  - Pilot 2, plus its §6 recompute;
  - Pilot 3;
  - Phase A extension;
  - Pilot 4 (§1a–§1d, §2a, §2b);
  - dirty-flag re-run;
  - v1.0 lock and calibration;
  - v1.0 rehearsal and Check 2;
  - cusp diagnostic;
  - v1.1 lock;
  - v1.1 re-rehearsal;
  - confirmation.

  Columns: report path, launch commit(s), date, q, seeds, arms, runs, phase budget(s), parents, results root.
- **T02 [core] Decision log.** One row per decision in §5.1, with:
  - the options compared and the chosen value;
  - the evidence (report and section);
  - the round, and the first commit at which it applied.

  If no repository file records a decision, write "PI instruction, not recorded in repo".
- **T03 [core] Plan vs execution.** Each 0930 plan item (§5.2) against what was actually done, with the reason and source for every difference.

### Part I — Baseline plan (0930)

**§2 Setting and baseline audit (P0)**
- **T04 [core] Parameters and derived constants**, per q: ΔW, W_L, W_H, k, effort bounds, shock model, e₁*(0), e₂*(0), r = ΔW/(kq²), the positive-effort support |d| < 2q, and the D₁ and D₂ ranges and node counts per tier.
- **F01 [core] Closed-form benchmark:** e₂*(d) on D₂ for both q, with e₁*(0) marked. Use the repository's own closed-form function.
- **T05 [core] Baseline audit findings.**
  - Every Phase 0 item (stop rules, joint-training semantics, the shared actor, start-state coverage, the evaluation convention, checkpoint completeness), with the legacy behaviour, the risk, and how v2 handles it.
  - Also the PI's 0929 issue list (§5.3), each with a status: addressed (say where), not applicable, or open.
- **T06 [supp] Resolved configuration** of the locked v1.1 protocol, per q: PPO, network and Beta parameterization, observation encoding, budgets, LR windows, verifier tiers and cadence, threads, RNG namespaces. Mark the values that differ from the legacy as-run configuration reported in Phase 0.

**§3 Method change (a): the full-domain MPE metric**
- **T07 [core] Metric definitions:**
  - EXP_root, dReach, Δmax_all, Ĝmax_full, dFull, η₂;
  - the recovery metrics: stage-1 relative error; stage-2 peak error, signed and location-free; RMSE over the positive region; tail mean and max; symmetry; σ;
  - the on-path rule.

  Columns: symbol, definition, state region, deviation type, aggregation, normalization, implementing `file:function`.
- **T08 [supp] Verifier tiers**, per q: state-grid range, step and node count; the action search; quadrature; interpolation; dev vs final; cadence.
- **T09 [supp] Invariant checks.** For each relation, the maximum violation in ΔW units, and the evaluations it was measured over (tests, plus per-call logs if they exist):
  - G₂ = Δ₂;
  - Δmax_all ≤ Ĝmax_full ≤ dFull;
  - EXP_root = G₁(0);
  - EXP_root ≤ dReach ≤ dFull.
- **T10 [supp] dReach reach-mask check** (168 evaluations).
- **T11 [supp] Benchmark consistency:** the Monte Carlo terminal win probability against the analytic CDF, as the maximum deviation over the MC standard error.
- **T12 [core] Calibration:**
  - the analytic equilibrium (the floor) and the zero-effort policy, both tiers, both q;
  - Phase 1's 2×-finer check;
  - agreement between the Phase 1 and lock-time calibrations;
  - the PI-side reference values (§5.4).
- **F02 [supp] Zero-effort deviation gains:** G₂(d) and Δ₂(d) on D₂, with the stage-1 value as a marker, final tier, both q. Use saved arrays if they exist; otherwise re-evaluate (allowed; log it).

**§4 Method change (b): stagewise learning with frozen continuation**
- **T13 [core] Flags:** each v2 flag, with its values, the legacy default, the study that decided it, and the locked value.
- **T14 [supp] Snapshot drift.**
  - For each frozen-mode study: the number of runs with drift exactly 0, and the max absolute drift.
  - For Pilot 2's joint arm: the drift on and off path.
- **T15 [supp] C7 regression** at every launch commit: commit, compare file, verdict, dirty flag.
- **T16 [supp] Test suite** at every launch commit: passed, xfailed, failed (name the known failure).
- **T17 [supp] RNG streams:**
  - each stream and what it drives;
  - where the learn and opp streams desynchronize in each study;
  - the v1.1 handling of the process-global RNGs.
- **T18 [core] Reproducibility ledger.** Every bit-identity check run so far, with its scope, n and result:
  - the Phase 2 branching test;
  - Pilot 3 stochastic vs Pilot 2 B2;
  - Pilot 4 §2a constant vs the extension;
  - the dirty-flag re-run;
  - the pre-lock insurance runs;
  - v1.0 Check 1 and Check 2;
  - v1.1 R1.

**§5 Pilots**
- **T19 [core] Pilot design summary** for Pilots 1–3: question, phase and budget, parents, arms (flag values), q × seeds, runs, launch commit, wall time per run, and the decision taken.

*Pilot 1: sampled vs expected*
- **T20 [core] Final medians and IQR**, per arm and q:
  - peak error: signed, absolute, and location-free (derive it from saved arrays if it was not recorded);
  - RMSE/e₂*(0); tail mean and max; symmetry;
  - η₂; Δ₂ on and off path; σ₂(0).
- **T21 [core] Paired differences** (expected − sampled): median, sign count, bootstrap CI.
- **F03 [core] Learning curves** from the weight exports every 25 updates: peak error, RMSE/e₂*(0), tail mean, η₂. Median and IQR per arm, per q.
- **F04 [core] ê₂(d) against e₂*(d) at u400:** the across-seed median with an IQR band, per arm and q. Add a lower panel with σ₂(d).
- **F05 [supp] Peak error against σ₂(0)/q** for each run, with Spearman ρ per arm.

*Pilot 2: A joint, B1 frozen with all-rows normalization, B2 frozen with stage-1-rows normalization*
- **T22 [core] Final medians**, per arm and q: stage-2 drift on and off path, peak error, tail mean, stage-1 error, Ĝmax_full with (t*, d*), EXP_root, dReach.
- **T23 [core] Paired comparisons** B1 − A, B2 − A and B2 − B1, for the same metrics.
- **T24 [supp] On-path definition and drift decomposition:** cell-mass weighted and unweighted.
- **T25 [supp] Location of Ĝmax_full:** counts of (t*, d*) per arm.
- **F06 [core] Learning curves.** The existing figure may be reused or regenerated.
- **F07 [core] Stage-2 mapping at the end of Phase B against the parent.** Panel 1: ê₂(d) for A and B2 (across-seed median). Panel 2: the change from the parent. Shade the on-path region.
- **F08 [supp] Advantage SD ratio**, B1 vs B2.

*Pilot 3: stochastic vs mean continuation*
- **T26 [core] Reproducibility:** the stochastic arm vs Pilot 2 B2.
- **T27 [core] Final medians** per arm and q, including the decomposition terms.
- **T28 [core] Paired differences** (mean − stochastic).
- **T29 [core] Stability:**
  - within-run SD and range over the last 5 exports;
  - across-seed SD and IQR;
  - σ₁(0).
- **F09 [core] Learning curves.**
- **F10 [supp] ê₁(0) trajectories** from the weight exports: every seed, per arm and q, with e₁*(0) marked.

**§6 Formal T=2 experiment: locked protocol and fresh-seed confirmation**
- **T30 [core] Locked pipeline (v1.1):** every setting of Phase A, the freeze, Phase B, and the evaluated candidate.
- **T31 [core] Criteria:**
  - G-A, G-F, G-N and S1, each with its definition, threshold, tier and source function;
  - run pass and the pass rule;
  - the seeds;
  - the bootstrap and Clopper–Pearson settings;
  - the crash rule.
- **F11 [supp] LR schedule:** actor and critic LR against global update (Phase A 1–1600, Phase B 1601–2200), from `lr_at` with the locked windows. Note its agreement with the logged LR.
- **T32 [core] Protocol history:**
  - the v1.0 gates;
  - the v1.0 rehearsal pass counts and failing criteria;
  - the v1.0 → v1.1 change log;
  - the v1.0 pass-probability recomputation.
- **T33 [supp] Timeline (UTC):** lock commits, LOCK records, rehearsal launches, the checks commit, and the confirmation launch and end.
- **T34 [supp] Re-rehearsal checks** R1–R6.
- **T35 [core] Confirmation verdict:** per q, the passes out of 20, the exact 95% CI and the rule; then the overall verdict.
- **T36 [supp] Per-run confirmation table** (40 rows).
- **T37 [core] Distributions** (min, p10, p25, median, p75, p90, max) of every gate metric and of dev − final.
- **T38 [core] S1 summary.**
- **T39 [core] Reported metrics**, per q (median, IQR, min, max):
  - peak error, signed and location-free, with the argmax d;
  - symmetry; tail max; Δ₂ on and off path; σ₂(0); the smoothed-game share;
  - ê₁(0); EXP_root; dReach; Δmax_all; dFull; counts of (t*, d*); σ₁(0);
  - the learning and inherited terms, and how many of their bands contain 0;
  - the drift test.
- **F12 [core] Gate metrics against thresholds.** Every run's value against its threshold, per q, for:
  - η₂;
  - RMSE/e₂*(0);
  - tail mean/e₂*(0);
  - Ĝmax_full;
  - |dev − final| for η₂ and for Ĝmax;
  - |stage-1 error|, with the S1 line.
- **F13 [core] ê₂(d) of all 20 runs per q** (thin lines) with their median and e₂*(d), plus a lower panel with σ₂(d).
  - Source: the end-of-A arrays (`gateA_*.npz`).
  - Stage 2 is frozen, so the end-of-B arrays must give the same mapping; verify this.
- **F14 [core] Stage-1 effort and its decomposition.** Panel 1: ê₁(0) per run against e₁*(0), with ±5% and ±10% bands. Panel 2: the learning and inherited terms per run, with their band intervals.
- **F15 [core] EXP_root/ΔW against the squared stage-1 error,** with the OLS fit per q.
- **F16 [core] Learning curves under the locked pipeline:** the confirmation runs, median and IQR per q.
  - Phase A: η₂, RMSE/e₂*(0), peak error, tail mean, σ₂(0).
  - Phase B: stage-1 error, Ĝmax_full, EXP_root.
  - Use the checkpoint CSVs if their updates align across runs. Otherwise, evaluate the weight exports on the dev tier, and log it.
- **F17 [supp] G_t(d) on D_t** for two runs per q: the run with the median Ĝmax, and the run with the max Ĝmax.
- **T40 [supp] Rehearsal vs confirmation:** distributions on the development seeds against the fresh seeds.
- **T41 [core] Targets vs results.** For each 0929 target (§5.4):
  - the locked criterion it became, or "reported only";
  - the confirmation result per q: median, max, and the number of runs that meet the original target.

### Part II — Extensions beyond the 0930 plan

**§7 Stage-2 accuracy beyond 400 updates**
- **T42 [core] Phase A extension:** the median, IQR, min and max of the stage-2 metrics at u400, u800, u1200 and u1600.
- **F18 [core] Extension learning curves**, u400–u1600.
- **T43 [core] Pilot 4 §2a** (LR decay over u1201–1600): final tables, paired differences (decay − constant), and the reproducibility check.
- **T44 [supp] Stage-2 tail averaging:** Pilot 4 §1c.2 against K = 1, and Pilot 4 §2a with K ∈ {1, 4, 8}.
- **F19 [core] Pilot 4 §2a learning curves.**

**§8 Anatomy of the stage-2 peak gap**
- **T45 [core] Smoothed-game share** of the d = 0 peak gap, wherever it was computed: Pilot 1 (both arms), the extension (u400–u1600), the v1.0 rehearsal, and the confirmation.
- **T46 [core] Representation floor:** the supervised fit (5 inits × 2 q), and the three-way comparison of the RL gap, the smoothing part, and the supervised floor.
- **T47 [core] Cusp diagnostic:**
  - the steps to |peak error| < 0.05, 0.03 and 0.01;
  - the RMSE at the 0.05 crossing;
  - the RL actor-step counts.
- **F20 [core] Three-way peak gap** per q (bars, or points with ranges).
- **F21 [core] Supervised-fit trajectories:** |peak error| and RMSE/e₂*(0) against optimizer steps (5 inits per q), with the RL step counts marked.

**§9 Stage-1 accuracy**
- **T48 [core] Induced-target method:**
  - the definition;
  - the calibration gate (the band contains e₁* when ê₂ = e₂*);
  - band-width and contiguity statistics;
  - sweep-range coverage;
  - the comparison with the superseded solver (Pilot 2 §6, Appendix S).
- **F22 [core] Δ₁(e; ê₂) against e** for one confirmation run per q: the run with the median |stage-1 error|. Mark the band, ẽ₁, ê₁ and e₁*. Source: `band_sweep.npz`.
- **T49 [core] Root stage game:** the numeric BR slope and own curvature, against the PI references.
- **T50 [core] Anatomy of the stage-1 fluctuation:** the ACF and the window regression.
- **F23 [supp] ACF plot**, if one does not already exist.
- **T51 [core] Pilot 4 §2b** (Phase B LR decay): final tables, paired differences, stability; and stage-1 tail averaging (Pilot 4 §1c.1 and §2b, K ∈ {1, 4, 8, 12}).
- **F24 [core] Pilot 4 §2b learning curves.**

**§10 From development distributions to pre-registered gates**
- **T52 [supp] Pilot 4 §6 distribution tables** for the end-of-phase candidates.
- **T53 [core] v1.0 rehearsal:** gates per run, and pass counts. This is the evidence for v1.1.

**§11 Protocol engineering and reproducibility**
- **T54 [supp] v1.0 Check 1 and Check 2.** For Check 1, the field table plus the explanation of the three process-global RNG states.
- **T55 [supp] Consolidation:** results-root checksums, the dirty-flag re-run, and the seed inventory.
- **T56 [core] v1.1 global-RNG hardening:**
  - the seeding and the reference point;
  - the draws before the reference point, with their call site;
  - the assertion results in all 60 runs;
  - the injection tests.

**§12 Known issues and open questions (T=2 only)**
- **T57 [core] Issue list.** For each issue, the evidence, status and impact:
  - the stage-2 peak bias;
  - stage-1 dispersion;
  - the q=60 final-tier floor of 3.5e−7 at (t=1, d=0);
  - learn/opp stream desynchronization;
  - non-contiguous bands, and early rows outside the sweep;
  - the known test failure;
  - Beta clip likelihood consistency: say whether any report addresses it, and if none does, say so;
  - misspecified-policy discrimination, which has been tested only on the zero-effort policy.

### Appendices
- **D01–D09 [supp] Consolidated per-run tables** in `data/`: Pilot 1, Pilot 2, Pilot 3, the extension, Pilot 4 §2a, Pilot 4 §2b, the v1.0 rehearsal, the v1.1 re-rehearsal, and the confirmation. Keep the original column names, and document them in `data_dictionary.csv`.
- **T58 [supp] Compute**, per study: runs, wall time per run (min/median/max), parallelism, `nproc` and load at launch, and total CPU-hours.
- **T59 [supp] Commands to reproduce,** collected from every report, in study order.

---

## 4. Key numbers (`key_numbers.csv`)

Columns: `id, description, value, unit, normalization, tier, q, source file, selector (row/column), computation`.

At least the following:
- **Confirmation (K01–K09):**
  - **K01** primary passes per q, with the exact 95% CI, and the overall verdict;
  - **K02** Ĝmax_full/ΔW: median and max per q;
  - **K03** η₂: median and max per q;
  - **K04** RMSE/e₂*(0): median and max per q;
  - **K05** tail mean, /e₂*(0) and raw: median and max per q;
  - **K06** peak error at d = 0: median and range per q, and the number of runs within 5%;
  - **K07** stage-1 error: mean signed error with its bootstrap CI, SD, S1 passes, and the number of runs within 5%, per q;
  - **K08** the max |dev − final| for η₂ and for Ĝmax_full;
  - **K09** the v1.0 outcome on the confirmation runs.
- **Pilots and extensions (K10–K16):**
  - **K10** Pilot 1: medians of η₂ and peak error per arm and q, and the sign counts;
  - **K11** Pilot 2: the joint arm's drift on and off path, and the peak error and tail mean per arm;
  - **K12** Pilot 3: paired differences of the stage-1 metrics, and the within-run SD;
  - **K13** Phase A extension: peak error at u800, u1200 and u1600, and the trends of the tail mean and σ₂(0);
  - **K14** Pilot 4: decay − constant effects (§2a on RMSE and tail mean; §2b on stage-1 dispersion);
  - **K15** smoothed-game share: medians across studies;
  - **K16** the supervised-fit floor of the peak error.
- **Verifier, reproducibility and compute (K17–K19):**
  - **K17** the calibration floor values, the zero-effort Ĝmax, and its agreement with the PI references;
  - **K18** the number of bit-identity checks passed, out of how many;
  - **K19** total runs and total CPU-hours over P0–P6.

---

## 5. PI-supplied content (not in the repository)

### 5.1 Decisions to log in T02
1. Base commit `657f54a` and the repository path (Phase 0).
2. q ∈ {50, 60}. Development seeds 10501–10510: three were proposed, then expanded to ten in Phase 2.
3. Stage-1 training settings:
   - stage-1 training is Phase B only;
   - the stage-1 opponent is the lagged copy, refreshed every 20 updates;
   - in frozen mode, advantages are normalized over stage-1 rows only;
   - joint mode is unchanged.
4. RNG alignment (A6): `expected` still draws the shocks; `mean` still draws the stage-2 actions and discards them.
5. On-path rule: the open interval, with exact cell masses (Phase 2).
6. Reward estimator `expected` (after Pilot 1).
7. Frozen variant B2 (after Pilot 2).
8. ẽ₁ by residual minimization on the final tier, with a band (after Pilot 2).
9. The Phase A extension runs in parallel with Pilot 3.
10. Continuation mode `mean` (after Pilot 3).
11. After Pilot 3 and the extension:
    - Phase A is fixed at 1600 updates, with an end-of-phase gate;
    - joint training and Phase C are dropped;
    - the sampler is not changed.
12. After Pilot 4: LR decay in both phases, evaluate the last iterate, no tail averaging.
13. Lock round: the v1.0 gates and pass rule; 20 fresh seeds; peak error reported only.
14. v1.1 round:
    - Check 1 accepted, with hardening;
    - the stage-1 criterion moved to S1;
    - G-N added;
    - seeds 20501–20520 confirmed;
    - automatic continuation to the confirmation.

### 5.2 The 0930 plan (the baseline of the report)
- **Goal:** improve T=2 accuracy in both recovery and verification, so that T=2 can serve as the algorithm-development and calibration environment; then move to T=3.
- **Method change (a):** the full-domain, grid-based MPE metric Ĝmax_full, keeping EXP_root, dReach, Δmax_all and dFull.
- **Method change (b):** stagewise backward learning with frozen continuation.
- **Pilot definitions:** Δ₂(d), G₂ = Δ₂, η₂.
- **Pilot 1:** terminal-only; sampled vs conditional expected reward.
  - Compare peak error, RMSE, tail effort and η₂.
  - Only the estimator changes.
- **Pilot 2:** from the same stage-2 checkpoint, joint vs frozen stage 2.
  - Compare stage-2 drift, stage-1 recovery, Ĝmax_full, dReach and EXP_root, and the frozen stage's output drift.
- **Pilot 3:** stochastic vs mean continuation, with stage 2 frozen; judged by stage-1 accuracy and stability.
- **If the pilots improve accuracy,** run the formal two-stage experiment.
- The 0929 plan specified 2 q × 3 paired seeds for the pilots.

### 5.3 The 0929 issue list (for T05)
- Phase A fixed-budget semantics.
- T2/T3 verifier warm-up differences.
- Legacy runner defaults.
- One actor shared by all stages.
- No explicit trainable-stage mask.
- Budget-forced continuation.
- KL only monitored.
- Likelihood of clipped Beta samples.
- dReach reach mask vs PMF support.
- Recovery used only post hoc.
- Ĝmax_full not a primary metric.
- Branching that loads weights only.

### 5.4 The 0929 validation targets and the PI-side references

**Targets:**
- stage-1 relative error ≤ 5%;
- stage-2 peak relative error ≤ 5%;
- positive-region normalized RMSE ≤ 5%;
- tail mean effort ≤ 2 (effort units);
- Ĝmax_full/ΔW ≤ 0.01;
- a formal dev–final refinement threshold;
- verifier calibration: the exact equilibrium sits at the numerical floor;
- discrimination: misspecified policies show clearly larger deviation;
- reliability on fresh held-out seeds.

**PI-side references:**
- zero-effort Ĝmax_full/ΔW: 0.2593 at q = 50 (stage 2, d ≈ −26) and 0.1955 at q = 60 (stage 2, d ≈ −23.5);
- root gains: 0.243 and 0.191;
- root-game BR slope: −0.961 and −0.309;
- E[V₂″]/(2k): 0.490 and 0.236.

---

## 6. Report back and STOP

When the pack is built:
1. Commit it (lightweight files only).
2. Report:
   - counts by status (found / regenerated / generated / derived / missing);
   - the `gaps.md` list;
   - the `consistency.md` mismatches;
   - the evaluations performed, and their cost;
   - the base commit and the pack's commit.
3. **STOP.** Do not write the report text, and do not start any new experiment.
