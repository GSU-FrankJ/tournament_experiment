<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- ACTIVE WORK (prepended 2026-07-24): MISSING-cell evaluations + final    -->
<!-- T-A/T-C (follow-up to the 2026-07-23 session below).                    -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

# SESSION_STATE — MISSING evaluations + finalized tables (2026-07-24)

**Branch:** `feat/multistage-phase0` · Post-hoc evaluation + MC-BR polish on
frozen profiles only; zero PPO training. **Locked conventions:** C1 = paper
error metric is `err_of_mean` = |ē − e*| (MAE kept on record as separate CSV
columns); C2 = 3P polished convention is the per-seed per-player aggregation
(24.75 / 15.84 from `polish_per_seed_all.json`).

## Tasks run (artifact → filled cell)

1. **2P q45 post-polish exploitability** — `tools/one_stage_ablation_q45.py`
   (new) mirrors phase 1's referee procedure; evaluated objects = the 5
   per-seed q45 polish landings + their cross-seed mean 35.0902 (phase 1's
   `e_fp` also averaged 2 probe controls, which don't exist at q45 — stated).
   → `results/one_stage_ablation/ablation_results_q45.{json,log}` (sibling
   file; committed `ablation_results.json` untouched).
   **EXP_det@polished-mean = 2.738e-5 abs** (per-seed 1.97–3.59e-5);
   EXP_det@raw-mean = 1.057e-5 (raw q45 sits closer to e* than the polish
   fixed point — mirrors err_of_mean 0.11 raw vs 0.26 polished); legacy-MC
   floor ≈ 4e-3 (cannot separate). Fills T-A full-arm q45 + T-C 2P q45.
2. **no_exploit arm terminal exploitability** —
   `tools/no_exploit_terminal_exploitability.py` (new) +
   **`BetaPolicyAgent` added to `tools/one_stage_mc_adapter.py`** (additive):
   reconstructs each run's stochastic terminal Beta(α,β) policy and runs the
   SHIPPED `eval_exploitability` (M=8192, CRN, 5.0/1.0/0.25) — the identical
   code path behind the verified arms' `final_exploit_max`. R=5 fresh seeds
   (100+7r), per-profile rep-mean, cell = mean±std across seeds.
   → `results/one_stage_ablation/no_exploit_terminal_exploitability.{json,log}`.
   **MC: q35 0.0109±0.0071, q45 0.0086±0.0055, q55 0.0044±0.0031** (abs);
   secondary referee EXP_det 5.5e-3 / 4.9e-3 / 1.9e-3 — 5×/465×/5× the
   baseline raw-mean EXP_det. Fills T-A row 4.
3. **Set-2 (8,4) MC-BR polish** — `tools/one_stage_polish_per_seed_set2.py`
   (new; reuses `_polish_row`/`POL` from the canonical per-seed polish tool;
   identical solver settings, start = final Beta mean, polish_seed = 4000+si)
   → `results/one_stage_ablation/polish_per_seed_set2.{json,log}` (sibling;
   `polish_per_seed_all.json` untouched), 15 rows, 1903 s tmux.
   **Landings: q35 47.045±0.032 (e* 47.6190, err 0.574), q45 36.751±0.042
   (37.0370, 0.286), q55 30.127±0.025 (30.3030, 0.177)** — same undershoot
   pattern as Set 1. `paper/figures/convergence_main.{png,pdf}` re-rendered
   (ONLY this figure; mtime-verified) — Set-2 row now carries ★ markers via
   variant-keyed `_polished_two_player_means()` in `paper/generator/plots.py`.
   `convergence_main_1x3` (Set-1 only) and all other figures untouched.
4. **3P per-seed (a)-leg under C2** —
   `tools/threep_postpolish_exploit_per_seed.py` (new): identical
   `exploitability_frozen_profile` call (M=200k, seeds 600000+cid*100+si) on
   each seed's polished per-player profile.
   → `results/one_stage_ablation/threep_postpolish_exploit_per_seed.{json,log}`.
   **q35 0.000816±0.000519, q55 0.000651±0.000212 — REPRODUCES the committed
   log's (a)EXP=0.0008/0.0007.** Key correction to the 2026-07-23 note: the
   committed (a)-leg was ALREADY measured per-seed on the C2 profiles
   (phase0_verify polishes per seed; identical inputs/seeds as
   polish_per_seed_all); the log's "24.68" is merely PLAYER 1's cross-seed
   mean (`labels=["P(sym)"]` prints only p=0) — there is no separate
   symmetric-projection object. T-C 3P exploitability cells therefore stay
   0.0008/0.0007, now with persisted per-seed values and ±std.

## Gray-line semantics REVERTED to verified stop (owner request, 2026-07-24)

Owner review flagged the first-pass gray line as misleading (q35: line at
update ~11 while the plotted curves are still far apart). Investigation
confirmed the data call was CORRECT but the first-pass definition misreads:
the plotted Agent-1/2 curves carry the asymmetric exploration warm-up bias
(`run_two_players.py:960-969`, 20 updates), while the exploitability check
evaluates the near-e*-initialized POLICY (policy_mean_effort[0]≈48.35;
update-8 check = 0.0183 < 0.03 legitimately passes). Line switched back to
the VERIFIED stop (mean stopped_at_update = where training stops and the raw
estimate is read out): Set-1 55/75/87, Set-2 51/71/97. First-pass values
remain computed + printed for reference. Both `convergence_main.{png,pdf}`
(2x3) and `convergence_main_1x3.{png,pdf}` re-rendered; sizes for layout
choice: 2x3 aspect ≈ 1.34:1, 1x3 ≈ 2.13:1 at \linewidth.

## F9 dotplot polish (owner request, 2026-07-24)

`equilibrium_recovery_dotplot.{png,pdf}` re-rendered: legend folded to two
rows (ncol=3, title pad 30→52 to clear it); seed markers de-emphasized —
semi-transparent (face alpha 0.35, edge alpha 0.45), s 70→55, zorder 5→3 so
the across-seed mean diamonds sit on top. Legend swatches match. Data
unchanged (Claim-B polished override).

## q45 "polish raises Final Error" — diagnosed NOT a bug (owner query, 2026-07-24)

T-A shows raw err_of_mean 0.108 < polished 0.263 at q45. Root cause = two
documented mechanisms colliding: (i) the raw 0.108 is a SIGN-CANCELLATION
artifact — per-seed signed errors are −0.49/+0.42/+0.66/+0.55/−0.60 (3 above,
2 below e*; per-seed MAE 0.544); (ii) the polish collapses all seeds to the
solver's sampled fixed point (sd 0.60→0.034, 18×), which carries the
systematic offset e_fp − e* ≈ −0.26 (monotone across q: −0.51/−0.26/−0.17,
exactly the Phase-0 decomposition err_pol ≈ fixed-point offset + ~0.02
solver noise). New artifact `results/one_stage_ablation/q45_polish_probe.json`
(+`.log`, tools/one_stage_polish_probe_q45.py): starts {raw mean, 0, 50} land
{35.0907, 35.0993, 35.1106}, spread 0.0199 ⇒ init-independent fixed point,
same as the q35/55 Phase-0 probe. Under per-seed MAE the polish still HALVES
the q45 error (0.544→0.263); only the err-of-mean column shows a regression,
because the raw mean is accidentally accurate. Consistent exploitability
picture: EXP_det@raw_mean 1.06e-5 < EXP_det@polish 2.74e-5, both ≪ eps_eq.

## CSV / README (convention C1)

- `results/one_stage_claimb_summary.csv`: 4 columns APPENDED by
  `tools/extend_claimb_summary_err_columns.py` (new): `raw_err_of_mean`,
  `raw_mae_across_seeds`, `polished_err_of_mean`, `polished_mae_across_seeds`.
  Existing cells byte-identical (CRLF line endings preserved; verified every
  original line is a byte-prefix of its extended line). New MAE column
  reproduces the historical `|ē−e*|` raw values (0.79/1.05/0.94/1.07/0.99),
  confirming the old column's mixed semantics.
- `results/README.md`: appended section documenting both conventions and that
  paper tables use `err_of_mean` (C1), plus the 3P C2 note.

## Final table state

- T-A: both previously-MISSING exploitability cells filled (full-arm q45 =
  2.7e-5 det; no_exploit row = MC 0.0109/0.0086/0.0044). No other cell changed.
- T-C: 2P q45 post-polish cell = 2.7e-5 det; 3P cells unchanged in value,
  footnote corrected (per-seed C2 measurement, ±std now on record).

## MISSING v2

- dc/da per-seed (a)-leg dispersion (cell means 0.0006/0.0002/0.0010/0.0005
  committed; per-seed values still unpersisted) — extend
  `tools/threep_postpolish_exploit_per_seed.py` to cell_ids 2–5.
- Set-2 post-polish independent exploitability at the new landings — referee
  run with Set-2 params (w=8/4, k=0.0006; the referee module's constants are
  Set-1, needs parameterization).
- (8,5.5) symmetric 2P cell — new training, out of scope.

## Commit note

Split per repo rules: (i) code (5 new tools + additive adapter class +
plots.py variant-keyed star map), (ii) new evaluation artifacts under
results/one_stage_ablation/, (iii) CSV/README extension, (iv) figure regen
(convergence_main + convergence_main.csv). plots.py still carries earlier
uncommitted edits (2026-07-20/23).

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- ACTIVE WORK (prepended 2026-07-23): Figure regeneration (F2/F6/F9) +    -->
<!-- table data extraction (T-A/T-B/T-C) for the one-stage pipeline.         -->
<!-- Sections below are unchanged prior state.                               -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

# SESSION_STATE — Figure regen + table extraction (one-stage, 2026-07-23)

**Branch:** `feat/multistage-phase0` · **Zero training, zero GPU.** Plotting +
data extraction only; nothing under `results/*/convergence/` touched. All
extracted numbers were adversarially re-verified by an independent 6-agent
pass against the raw artifacts (0 numeric refutations on T-A/T-C).

## Figures regenerated (overwritten in place, `paper/figures/`)

1. **`convergence_main.{png,pdf}`** (manuscript Fig. 2) — `plot_convergence_main`:
   - Gray vertical line now marks the **FIRST-PASS verification update** (first
     in-training exploitability check the raw profile passes, < ε_eq=0.03),
     mean across 5 seeds — previously it marked **streak completion**
     (`stopped_at_update` = 5th consecutive passing check). New helper
     `_first_pass_updates()`; 1-based convention matches `stopped_at_update`
     (verified: last passing index+1 == stopped_at_update in all 30 runs).
     First-pass vs streak (mean): Set 1 q35 11/55, q45 35/75, q55 47/87;
     Set 2 (8,4) 11/51, 31/71, 57/97.
   - New endpoint markers: ○ raw final estimate (cross-seed mean of final
     policy-mean efforts) on every panel; ★ MC-BR polished estimate
     (cross-seed mean of `polish_per_seed_all.json` landings: 44.947 / 35.090 /
     28.757) on the Set-1 row only — **no polish artifact exists for Set 2**,
     so no star there. Star x-offset so markers never overlap.
   - Legend: "Convergence update" → "Verification update"; added "Raw
     estimate" (○), "MC-BR polished estimate" (★).
   - Known cosmetic carry-over: the e*-value annotation still overlaps the
     trajectory tail in q=55 panels (pre-existing placement, unchanged).
2. **`convergence_main_1x3.{png,pdf}` (NEW FILE)** — 1×3 companion, Set-1 row
   only, sized like exploitability_dynamics (figsize 10×4);
   `plot_convergence_main_1x3()` + `tools/regen_convergence_main_1x3.py`.
   Does not overwrite the 2×3 figure or convergence_main.csv.
3. **`exploitability_dynamics.{png,pdf}`** (manuscript Fig. 6) — y-label →
   **"Raw profile exploitability"** (data source confirmed in code: in-loop
   `eval_exploitability` on the current self-play policy,
   `run/run_two_players.py:1385-1417` — the streak-stop quantity, NOT
   post-polish). Legend compressed to exactly 4 entries: "Mean Exploit.",
   "Tolerance $\varepsilon_{eq} = 0.03$", "Stability passed", "Verified";
   moved fully above the axes (panels no longer squeezed). Log scale/data
   unchanged. (No ±1 std band ever existed here — per-seed thin curves kept.)
4. **`equilibrium_recovery_dotplot.{png,pdf}`** (manuscript Fig. 8) — title →
   "Equilibrium Effort Recovery Across Tournament Variants and Noise Levels";
   y-label → "Effort Level"; seed markers enlarged (s 40→70), light face +
   black edge, zorder above the mean diamonds (diamonds 100→130 so both
   layers read); "Per-seed estimate" → "Seed-level estimates"; het-cost
   labels → "Low-cost agent"/"High-cost agent" (mapping verified: k1=4e-4 <
   k2=5.5e-4, e1*=38.03>e2*=27.66, learned effort1>effort2 in all 10 runs);
   het-cost theory lines → BLACK (solid=low-cost, dashed=high-cost) with new
   legend entry "Theory e* (low-cost)". Data unchanged (Claim-B polished
   per-seed override). Note: het-cost mean diamonds remain agent-colored
   (pre-existing) while the legend swatch is red.

**Retired figure `ablation_comparison.{png,pdf}` untouched** (verified mtime
2026-07-20; replaced by table T-A delivered in the session reply).

## ⚠️ As-built discovery (affects Table 2 / ONE_STAGE_ASBUILT.md)

The canonical `r5_sampled` PPO baselines for **2P (CLI default), 3P and dc
(explicit `--theory-align-v2` in manifest cmdlines)** trained under the
**theory-align-v2 override block** (`run/run_two_players.py:1764-1767,
1856-1892`): entropy coef **0** for the whole run (not 0.03→0.005), LR
**5e-5 → 2e-5** (not 3e-4→2e-4), clip **0.20 → 0.15** (not 0.50→0.35 nor
0.30→0.15), **1 PPO epoch/update** (not 6), target_kl 0.06, and the policy
net is **`ActorCriticMeanConc`** (sigmoid mean × softplus concentration,
α=mean·conc — not ActorCritic's softplus+1 α/β heads). Runtime signature
confirms: α+β ≈ 169 at update 1 (conc ramp start 100), impossible for
ActorCritic init. The v2 writes of γ/λ/value_coef/max_grad_norm into cfg are
**dead code** (PPOConfig never reads them) → effective γ=0.99, λ=0.95 (both
inert, single-step), value_coef=0.5, max_grad_norm=0.5. Only the **da**
baseline (`r5_sampled_std`) is the standard arm.
⇒ `docs/ONE_STAGE_ASBUILT.md` §1 ("canonical = ActorCritic softplus+1,
theory_align_v2=False") and its §Q4 lr/entropy/clip/epochs rows are **wrong
for the as-run baselines** — flagged, not edited. Manuscript Table 2's
"Theory-align-v2" column values for epochs (1), clip (0.2→0.15), entropy (0)
match as-built; its v2 claims γ=λ=1.0 and value-loss 1.0 do NOT (dead code).

## Code changes (uncommitted, on top of prior uncommitted plots.py edits)

- `paper/generator/plots.py`: `_first_pass_updates()`,
  `_polished_two_player_means()`, `plot_convergence_main` (line semantics,
  endpoint markers, legend, figsize param), `plot_convergence_main_1x3()`,
  F6 label/legend edits, F9 edits.
- `paper/generator/__main__.py`: fixed pre-existing crash — `cmd_plot` passed
  the output path positionally into `plot_convergence_main`'s
  `weight_variants` slot.
- `tools/regen_convergence_main_1x3.py` (new).
- Regenerated `paper/data/{convergence_main,exploitability_dynamics,equilibrium_recovery_dotplot}.csv`.
- COMMIT NOTE: split code edits vs figure/data regen (repo rule); plots.py
  still carries earlier uncommitted edits from 2026-07-20.

## Open issues / MISSING

- Post-polish independent exploitability for 2P **q45**: no artifact
  (phase0/phase1 referee ran q35/q55 only). Producing run: extend
  `tools/one_stage_ablation_phase1.py` / `tools/one_stage_referee.py` to q45.
- Final exploitability for the `no_exploit` ablation arm: never evaluated
  in-run (by design; `final_exploit_max` null in all 15 JSONs). Producing
  run: post-hoc referee/legacy-MC eval at the terminal efforts.
- Set 2 (wh8_wl4): no MC-BR polish artifact → no ★ markers, no polished row.
- 3P polished value has two committed conventions: per-seed per-player polish
  aggregation 24.75±0.02 / 15.84±0.01 (`polish_per_seed_all.json`, what F9
  plots) vs phase0_verify symmetric-projection 24.68 / 15.82 (claimb CSV,
  what the (a)-leg exploitability was measured on). Flagged, not reconciled.

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- ACTIVE WORK (prepended 2026-07-16): One-Stage Ablation (raw vs MC-BR    -->
<!-- polish), Phase 0 COMPLETE — STOPPED for owner review. Sections below    -->
<!-- are unchanged prior state.                                              -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

# SESSION_STATE — One-Stage Ablation: raw PPO vs MC-BR polish (Phase 0 done, 2026-07-16)

**Branch:** `feat/multistage-phase0` · **HEAD:** `d3eeea9` · **Zero training, zero GPU.**
Phase-0 note: [`docs/ONE_STAGE_ABLATION_PHASE0.md`](docs/ONE_STAGE_ABLATION_PHASE0.md).

## Phase 0 result: 1 design-critical assumption FALSIFIED, everything else confirms

1. **STOP-FLAG — `utils/mc_br_polish.py` is a damped FIXED-POINT solver, not a single BR step.**
   Canonical `POL` (`tools/phase0_verify.py:30`): η=0.4, M=150k, **always 320 rounds**
   (`min_rounds=999 > max_rounds=320` ⇒ early-stop never fires), Polyak avg over last 200,
   **local ±10 BR window**. Proven twice — by code and by experiment: polish from starts
   {raw, 0, 50} lands within **0.015 (q35) / 0.020 (q55)** of the same point ⇒ **init-independent**.
   Landings 44.94 (q35) / 28.78 (q55) reproduce the committed 44.95 / 28.76. ✅
   ⇒ The prompt's P4/P5 decomposition ("BR truncation + solver noise") is invalid, and P6/P7's
   control predictions are **falsified for the polish arm** (P7's "q35 c=50 → 8.42 worse" is FALSE:
   it lands at 44.93, error 0.52 = **better**). Damping neutralizes the q35 expansive regime:
   iteration slope `0.6+0.4(−1.8519) = −0.141`, |·|<1 ⇒ contractive.
   **Proposed replacement:** `err_pol = (e_fp − e*) + (e_pol − e_fp)` = sampled-fixed-point offset
   (**−0.51 q35 / −0.14 q55**) + solver noise (**≈0.015–0.02**). Nearly all residual error is the
   offset, not solver noise.
2. **STOP-FLAG — legacy MC estimator consumes a policy object, not an effort**
   (`run/run_two_players.py:266`; needs `agent.net/state_from_params/dist/dist.sample`).
   Needs a degenerate point-policy adapter; running it on the raw *agent* measures a different
   (stochastic-policy) quantity. `utils/mc_br_polish.exploitability_frozen_profile:211` is the
   no-adapter alternative (used for P8 here).
3. **STOP-FLAG — one-stage ε=0.03 is ABSOLUTE payoff units** (`run/run_two_players.py:1425` vs
   `:356`) = **0.00857 /ΔW**; two-stage's `epsilon_over_dw=0.03` is **normalized**. Same numeral,
   different units.

## Built + validated (new files only; nothing existing modified)

- `tools/one_stage_referee.py` — deterministic closed-form referee (zero MC). **21/21 tests pass**:
  BR(e*)==e* exactly, **EXP(e*)=0.000e+00**, two independent BR paths agree to **≤6.5e-11**,
  F_ξ==∫f_ξ. BR-map structure reproduced: slope_above **−1.8519** (q35, expansive) / −0.3568 (q55),
  `q_expansive=√(ΔW/4k)=39.886`.
- `tools/one_stage_ablation_phase0.py` — reconstruction + tripwire →
  `results/one_stage_ablation/phase0_tripwire.json`.
- `tools/one_stage_polish_probe.py` — init-independence probe →
  `results/one_stage_ablation/phase0_polish_probe.json`.

## Key numbers (all recomputed this session)

- ê_raw 5-seed mean: **43.5798** (q35, sd 1.115) / **29.6543** (q55, sd 0.681); bit-check vs
  `final.effort` exact to **7e-15**; matches committed CSV 43.58 / 29.65. ✅
- Tripwire P1/P2/P3 **agree**: EXP_det(raw)=1.172e-3 / 3.962e-4; EXP_det(pol)=8.49e-5 / 1.19e-5;
  ratio **13.81× / 33.17×**. P9 crossing confirmed (mechanism re-attributed to the fp offset).
- **Legacy-MC floor ≈ 6.7e-3 (q35) / 7.2e-3 (q55)**, rep-sd 1.6e-3 / 3.4e-3 — an *upward-biased
  level* (max over noisy grid), which **exceeds both arms' true EXP** ⇒ legacy MC cannot separate
  the arms (C4 premise confirmed). Explains committed `final_exploit_max` ≈ 0.004–0.010 = the floor.
- Per-seed polished values were **never persisted** ⇒ A2 needs a re-run: start = per-seed raw mean,
  `seed=4000+si`, si=0..4 over seeds 42..46 (`tools/phase0_verify.py:117`). ≈19 min CPU in tmux.

## NEXT — blocked on owner decisions (5 items in the note §7)

Re-specify P4–P7 + decomposition; legacy-MC choice (adapter vs frozen-profile); P8 level-vs-SE;
deliverable path (prose → `docs/` vs the prompt's `results/`); C1 sign-count framing given the
polished arm is ~constant across seeds. **No Phase-1 work started.**

---

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- ACTIVE WORK (prepended 2026-07-12): Two-Stage TEL-PPO Correctness Pass.  -->
<!-- The project-wide state from the audit-remediation effort follows below,  -->
<!-- unchanged. This section is scoped to the T=2 correctness pass only.      -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

# SESSION_STATE — Two-Stage TEL-PPO Correctness Pass (ACTIVE, 2026-07-12)

**Branch:** `feat/multistage-phase0` · **Discipline:** investigate → test →
minimal fix → stop at phase boundary. Phase-0 diagnosis (approved):
[`DIAGNOSIS.md`](DIAGNOSIS.md).

## Approved decisions (owner, 2026-07-12)

1. **Retrain persists full checkpoint + raw Beta (α,β).** Mean stays PRIMARY
   reported extraction; mean-vs-mode kept as a DIAGNOSTIC (not dropped).
2. **Broader certification scope accepted** — corrected semantics apply to
   T=2,3,4,5. Distinguish:
   - **re-certification only** (JSON already stores the learned ê_t(d) curve):
     **T=3, T=4, T=5** (`effort_curves` present) + the T=2 `densecurve` run.
   - **retrain required** (no policy object / no α,β saved): the **5-seed gated
     T=2** runs (predate `effort_curves`; only ê_1(0)+81-pt ê_2+5-pt probe).
     No run saved α,β or a `.pt`, so **mean-vs-mode (T8) needs a retrain for
     every horizon**.
3. **dReach UCB = deterministic discretization bound**, NOT a Monte-Carlo CI:
   nested score-gap grid refinement, `dReach_UCB = dReach_fine +
   |dReach_fine − dReach_coarse|`; certify iff `dReach_UCB/ΔW ≤ 0.03`.
4. This state section created at Phase-1 start (prepended, non-destructive).

## Patch / test sequence (Phase 1) — tests first: `tools/test_phase1_correctness.py`

| Task | Change | Status |
|---|---|---|
| **T1** | `multi_stage_metrics.onpath_expected_stage2_effort()` (GL quad over Triangular[−2q,2q]); `RecoveryMetrics.e2_onpath_expected`; runner prints vs 46.6667 + per-d CF | ✅ tested |
| **T2** | `dp_verifier.certify_refined()` deterministic nested-grid `dReach_UCB`; runner + gate certify on it | ✅ tested |
| **T7** | runner: select ckpt on **coarse** grid (101), certify on **fine** (201)+UCB; persist `.pt` + α,β dump | ✅ smoke |
| **T8** | `ppo_multi_stage.beta_mode_normalized()`, `beta_params()`, `effort_function(extraction=…)`, `save()/load()` | ✅ tested |
| **T9** | SOC verification via `validate_two_stage_params` stage-2 global scan; param condition `q_soc=√(ΔW/8k)` | ✅ tested |

### Evidence (2026-07-12)
- **Unit tests:** `tools/test_phase1_correctness.py` — **ALL PASS** (15 checks:
  T1 E[g2]=46.6667 & constant-policy identity; T2 UCB≥fine, deterministic, CF
  certifies / const-high fails; T8 mode(3,3)=0.5, (2,5)=0.2, α≤1→mean fallback;
  T9 q_soc=41.833, q=50 dev2≈5.5e-17, q=35 invalid).
- **Runner smoke** (T=2, 10 upd, CPU, 4s — pipeline check, NOT a real run;
  artifacts deleted): all wiring fires — `E[e2(d2)]` printed vs 46.6667; the
  `dReach_UCB/DW` gate; coarse-select + `.pt` persistence; MEAN/MODE line; JSON
  gains `provenance` / `beta_params` / `extraction_diagnostic` /
  `e2_onpath_expected` / `dreach_{coarse,fine,ucb}_over_dw`. Real-world guard
  check: at 10 upd stage-1 α≈0.71<1, so MODE correctly fell back to MEAN.

### Commits (Phase 1, branch `feat/multistage-phase0`)
- `26de10d` — test: add Phase-1 correctness tests (tests-first)
- `77aaf9e` — feat: dReach-UCB gate, on-path E[e2], mean/mode, ckpt persistence
- _(this docs commit)_ — docs: Phase-0 diagnosis + Phase-1 session state
- T7 full-independence follow-up (selection grid 51; certify 101/201) is folded
  into `26de10d`/`77aaf9e` above (grid-role constants + disjointness test).

### Files changed (Phase 1 core)
- `utils/multi_stage_metrics.py` — `onpath_expected_stage2_effort()` + field (additive)
- `utils/dp_verifier.py` — `certify_refined()` (additive)
- `agents/ppo_multi_stage.py` — `beta_mode_normalized()`, `beta_params()`,
  `effort_function(extraction=…)` (default mean, unchanged behavior), `save()/load()`
- `run/run_multi_stage.py` — coarse-select/fine-certify+UCB, `.pt`+α,β persistence,
  provenance, E[ê_2] + mean/mode reporting
- `tools/test_phase1_correctness.py` — new self-checking test (T1,T2,T8,T9)
- `DIAGNOSIS.md` — Phase-0 deliverable (unchanged this phase)

**Deferred to Phase 1b (reporting; coupled to Phase-2 regen):** T4 falsification
table (EXP+dReach), T5 figures (stage-1 panel, `E[ê_2]` annotation, T=2 Δ_t),
T6 rename of *existing* g→e* keys/labels. New T1/T2 outputs already use
e-notation keys; the existing-key rename is held so figures/tables migrate in one
coordinated step when Phase-2 regenerates the JSON schema.

## Stopping point — PHASE 2 CORE TASKS COMPLETE (stopped at Phase 2 boundary)

Phase 1 committed (`26de10d`/`77aaf9e`/`e7d278a`) + T7 full-independence folded in.
Phase 2 tasks 1–4 done; **Phase-2 artifacts NOT yet committed** (awaiting review,
mirroring the Phase-1 rhythm). Extended follow-ups (Phase 1b figures, T=3/4/5
retrain) NOT started — stopped at the boundary per instruction.

### Phase 2 — DONE (evidence)
1. **Retrain 5-seed T=2** (tag `p2cert`, commit `e7d278a`, 2000×512, CPU ~15min/seed):
   **5/5 certify**, dReach_UCB/DW mean 0.0064 (max 0.0072). E[ê_2(d_2)] mean 44.69
   (target 46.667); ê_1(0) 46.38 (0.6% low); ê_2(0) 63.51 (μ*(κ) smoothing). Full
   persistence (`.pt` + α,β + provenance + mean/mode). MEAN≈MODE (Δ~0.16%) → D6:
   keep mean.
2. **Re-certify T=3/4/5** (`tools/recertify_multistage.py`, extracted-policy from
   effort curves): 5/5, 5/5, 4/5 — verdicts UNCHANGED under the UCB gate. Fine grid
   reproduces original dReach exactly.
3. **Reporting/tables** made schema-backward-compatible (`make_multistage_tables.py`);
   regen on old data = byte-identical (verified).
4. **`docs/two_stage_report.md`** + **`results/two_stage_results.json`** written;
   every number from an actual run.

### Commits (Phase 2, branch `feat/multistage-phase0`)
- `9ca4f2d` — feat: extracted-policy re-certification tool (code)
- `41d2889` — fix: table generator backward-compatible with dReach-UCB schema (code)
- `462630c` — results: T=2 retrain 5 seeds (p2cert convergence JSONs)
- `bc0dbfa` — results: T=3-5 re-certification + aggregated two_stage_results.json
- _(this docs commit)_ — docs: corrected two-stage report + session state
- `.pt` checkpoints under `results/multi_stage/checkpoints/` are gitignored (not committed).

### NOT started (extended scope — needs approval)
- Phase 1b: figure regen (T5), falsification table (T4), g→e existing-key rename (T6).
- Optional T=3/4/5 retrain for full stochastic-policy re-eval + α,β at those horizons.

---

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

---

# PHASE 0 — DEFINITIVE CONSOLIDATION (consolidated 2026-07-01, vector2)

## §A. Provenance anchor（每个数字都溯源到这里）

- **Source of truth**: `results/phase0_verify_20260701_1941.log`（unbuffered，已落盘；exit 0）
- **Scripts / commit**: 分支 `fix/audit-remediation`；`utils/mc_br_polish.py` + `tools/phase0_verify.py` + `tools/phase0_decomposition.py` 提交于 `8df54d0`，2P do-no-harm 收窄于 `7e593b4`（restrict to Set 1, exclude `wh8_wl4`）
- **Inputs**: r5_sampled convergence JSON，3P/dc/da × q∈{35,55} × seeds 42–46 = 30 runs（全部已入库、磁盘实文件）。**da 用** `_std` **tag**（effort 交叉验证：q35 mean 43.993 ≈ 43.99、q55 mean 29.703 ≈ 29.70；`_v2` 偏离 ~0.94/0.57，非 Phase-0 采用）
- **Repro params**（已对照 `tools/phase0_verify.py` 常量 + `results/phase0_verify_20260701_1941.log` header 逐项核对，一致）: polish M=150k、up to 320 rounds、**vertex BR (bias_correct=on)**；independent (b) FOC leg M=1,000,000；(a) exploitability leg M=200,000；FD step δ=0.75；seeds 42–46
- **Thresholds（pre-registered gate 参数，固定于数据之前）**: τ_E=0.005，τ_g=0.001，τ_e=0.1

## §B. Pre-registered acceptance gate（RULE，此节不含任何 result 数字）

规则独立于数据，先记录、后套用。每格须过三条**独立** legs：

- **(a)** post-polish exploitability，fresh draws + constant efforts +独立 seed：`EXP < τ_E`
- **(b)** interior FOC，frozen polished profile 上 **fresh seed / different FD step / larger M**：`|FOC| < τ_g`（boundary → projected/KKT）
- **(c)** polish converged：per-seed 轨迹内 Polyak-window `drift < τ_e` **或**窗口 SE
  （std/√n_avg）`< τ_e`——OR 判据，两个量都是 within-trajectory（**非** cross-seed），
  只作 polish 收敛 sanity check，承重为 (a)+(b)。（修正 2026-07-02：旧表述"以
  cross-seed SE 为准"与 `tools/phase0_verify.py` 实现不符，已按代码改正）
- **(d)** payoff loss small = u_br − u_policy = exploit delta（复用）
- **(e)** abs + rel error（`utils/evaluation.py`）

**Non-circularity（承重约束）**: leg (b) **禁止**用 FOC-root-find 做 polish——否则 polished 点 by construction 就是 polish 自己 FOC=0 处，(b) 退化为 auto-pass。设计 = 零阶 quadratic-vertex polish + 一阶 fresh-seed/different-step/larger-M 的 (b) + 独立零阶 (a)。(a) 结构上无法被 polish 方法 game。

**Directional undershoot guard**: 不接受"所有 seed undershoot 且 sampled MC-FD 指上坡"的 grazing-pass。

**dc 特判**: 按 plan §7 判 payoff-loss + exploitability（weak-identification；mixed-sign raw error 是 weak-id 指纹），不单看 effort error。

## §C. VERDICT — 6/6 main cells PASS

**OVERALL: 6 main cells all PASS = True；max polished error-vs-e* = 0.319（来自 3P q35）**

| Cell   | e*            | polished (mean) | error                  | verdict  | (a) EXP / (b) \|FOC\| / (c) drift, SE |
| ------ | ------------- | ---------------- | ----------------------- | -------- | -------------------------------------- |
| 3P q35 | 25.00         | 24.68            | −0.32 (1.28%)           | **PASS** | 0.0008 / 0.00044 / 0.159, 0.028        |
| 3P q55 | 15.91         | 15.82            | −0.09 (0.57%)           | **PASS** | 0.0007 / 0.00047 / 0.105, 0.016        |
| dc q35 | 38.03 / 27.66 | 38.04 / 27.66    | +0.01 / +0.01 (~0.02%)  | **PASS** | 0.0006 / 0.00027 / 0.078, 0.031        |
| dc q55 | 26.54 / 19.30 | 26.56 / 19.30    | +0.02 (0.09%) / −0.00 (0.02%) | **PASS** | 0.0002 / 0.00025 / 0.052, 0.018 |
| da q35 | 46.43 / 46.43 | 46.45 / 46.45    | +0.02 / +0.02 (~0.05%)  | **PASS** | 0.0010 / 0.00036 / 0.227, 0.068        |
| da q55 | 30.37 / 30.37 | 30.36 / 30.37    | −0.01 / +0.00 (~0.02%)  | **PASS** | 0.0005 / 0.00024 / 0.127, 0.029        |

- 三条 legs 全部满足（每格 (a)<0.005、(b)<0.001、(c)<0.1，drift 超 0.1 的格 SE 均 <0.1）。
- 误差上界 3P q35 的 **1.28%** 是 **sampled-vs-analytic gap**（sampled 均衡本就在 analytic 25 下方），非 learning failure，不触发 Phase-1（见 §E-2）。

**2P do-no-harm（sanity check，不计入 6-cell 表；`7e593b4` 后限 Set 1）:**

| Cell   | raw-mean err | polished err | no-harm  |
| ------ | ------------ | ------------ | -------- |
| 2P q35 | 1.875        | 0.508        | **True** |
| 2P q55 | 0.793        | 0.168        | **True** |

polish 在两 cell 均改善（误差下降，在 cross-seed σ 内），无 regression、无 polish bug。

## §D. SUPERSEDED map（保留 provenance，**DO NOT DELETE**）

标注 `SUPERSEDED → <definitive replacement>`，附被取代原因：

- **3P q35 polished**: 24.81（首个一次性脚本 `tools/phase0_mc_br_polish.py`，历史中已无此文件）→ 24.75（`phase0_decomposition.py`，argmax BR）→ "24.94 ± 0.04 confirmed FOC=0 point"（root-find）→ **SUPERSEDED → 24.68**（DEFINITIVE，`phase0_verify.py` vertex BR + independent legs，§C）
- **3P q35 verdict**: decomposition 版 **FAIL**（leg (b) |FOC|=0.00131）→ **SUPERSEDED → PASS**（|FOC|=0.00044）。原因：decomposition 的 FAIL 是 (c) τ_e stop 在 (b) 满足前先触发的 stopping artifact + argmax BR 方差，非真实 best-response gap；definitive 的 vertex BR（降方差）+ independent (b) leg（M=1M、fresh seed）在 flat plateau 上给出 sub-threshold FOC。**注意反直觉点**：definitive 数字 24.68 比 decomposition 的 24.75 更**低**（离 e*=25 更远），但 verdict 从 FAIL 变 PASS——因为曲面够平，24.68 与 24.75 的 |FOC| 都在噪声地板附近，验收看的是 |FOC|<τ_g 而非离某个特定点多近。
- **da q35 polished**: 46.41/46.46（decomposition）→ **SUPERSEDED → 46.45/46.45**（DEFINITIVE）
- 其余 cell 同理：decomposition 值 → §C definitive 值。
- **首个 throwaway 结果整段**（3P q35→24.81 那批、attribution probe 的早期数字）：标 SUPERSEDED banner，勿与 §C 混淆。

## §E. Component disposition（最终）

| Comp                              | 处置                          | 依据                                                                                                                |
| ---------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| (1) MC reward avg (K)              | **DROPPED，K=1**              | 唯一触碰 training reward 者；not "tried and failed"，是不需要。K>1 会软化"single realized outcome"说法。            |
| (2) mode-conc + conditional ramp   | **DROPPED**（spec 存档，见 §G）| α+β≈25k–33k near-spike → width 前提失效；polish init-independent → retrain 动不了 polished 落点。但它是唯一区分 Claim A/B 的实验。 |
| (3) mode extraction                | **INERT**（保留为 diagnostic 列）| L1 mode Δ ≈ −0.00；3P 可 post-hoc（JSON 存 α,β），**dc/da 只存 effort → mode N/A**。                                 |
| (4) MC-BR polish                   | **LOAD-BEARING**              | 零 GPU、sampled-only、不需 policy net；§C 全部由它得出。                                                              |
| (5) exploitability + FOC stop      | **finalized**                 | exploit-streak stop 已 production；新增 frozen polished profile 上 fresh-seed 第二次 exploit。                       |

**最终 pipeline（Claim B 路线）**: post-hoc MC-BR polish 单独把 3P/dc/da undershoot 收到 verified sampled 均衡，零 GPU。1 & 2 DROPPED，3 INERT（诊断），4 LOAD-BEARING，5 finalized。

---

# PHASE 0 — SELF-CORRECTION HISTORY（**DO NOT RESURRECT**）

措辞：先前结论 X 被证据 Y 证伪，替换为 Z。任何未来 session 不得复活 X。

### 自我修正 #1 —— "argmax-BR 有 ~0.17 downward bias"

- **X（已证伪）**: 曾断言 argmax-of-noisy-payoff BR 在 flat cell 上系统性向下偏 ~0.17，polish-vs-rootfind 的 gap 由此 bias 造成。
- **Y（证据）**: debias smoke 几乎没动均值——24.821 → 24.811（预期 ~0.13，实测 ~0.01）。vertex debias 的真实作用是**降方差**（e=25 处 argmax 24.89 ± 0.37 → vertex 24.75 ± 0.27），不是去偏。
- **Z（替换）**: polish-vs-rootfind 的 gap **不是 argmax bias**，是 **flat-plateau 宽度**。vertex BR 保留是因为降方差有价值，不是因为它去偏。

### 自我修正 #2 —— "confirmed FOC=0 point = 24.94 ± 0.04"

- **X（已证伪）**: 曾断言 3P q35 的 sampled FOC=0 点精确在 24.94 ± 0.04。
- **Y（证据）**: ±0.04 只是 3-seed spread；实测 |FOC| < ~0.0003 横跨 [24.7, 25.0]，与 FOC estimator 自身噪声地板同量级。
- **Z（替换）**: 3P q35 的 sampled 均衡**只能钉到一个平台区间**（|FOC| 在该区间内不可区分于 0），不是一个 sharp 点。因此 §C 里 polished 24.68 过 (b) 与"24.94 是均衡点"并不矛盾——验收判据是 |FOC|<τ_g，不是逼近某个过度精确的点估计。

### 中途修正（附记，非上述两条之一）—— "polishing 帮不了 dc"

- **X**: 曾预测 dc 因 flat FOC / weak-id，polish 移不动 effort。
- **Y**: dc 干净 polish 到 e*（§C，err ~0.02–0.05%）。
- **Z**: MC-BR 是重平均**直接求解器**，能定位 flat optimum，即便单次采样奖励下的**学习** agent 不能。weak-id 讲 PPO 学习动态与经济解读，不是暴力求解器能否找到 e*。dc 仍按 §7 报告。

---

# OPEN DECISIONS（待 Frank 拍板；§C 已 6/6 PASS；决策 3 已有实验结果，见下）

- **决策 1（headline）Claim A vs B**：
  - **A** "PPO self-play **学到**均衡 effort" → polish 不能支撑（§C 的 attribution：polish 从任意 init 都到 e*，是 global solver）；须跑 §G 的 Component-2 retrain。
  - **B** "PPO 到达 basin；sampled MC-BR + exploitability **证明**均衡" → polish 够用，但论文须明确把最终数字归功 MC-BR，raw/polished 保持独立两列。**当前证据（6/6 PASS + attribution + Component-2 retrain 结果，见决策 3）指向 B。**
- **决策 2 审稿人预防**："post-hoc polish ≠ proof of convergence" —— 论文须主动回应（polish 是 global solver 会被抓）。
- **决策 3（仅当选 A）**：授权 §G 存档的 Component-2 retrain。**已授权并执行（2026-07-02），结果如下：**

## 决策 3 执行结果 —— Component-2 mode-conc retrain（3P q35，seeds 42–46）

**结论：Claim A 不被支持。这是一个干净的负结果，不是噪声。**

新增 `ActorCriticModeConc` 头（α,β≥1 的 mode-conc 参数化）+ exploitability-triggered
的 κ 斜坡（explore κ∈[1,20] → 触发条件 EXP_raw<0.05 连续 3 次 → κ 走 [20,50,100,200]，
每档 20 updates → κ=200 后交给正常 exploit-stop），代码在
`agents/ppo_three_players.py` (`ActorCriticModeConc`) + `run/run_three_players.py`
(`--mode-conc-ramp` 及相关 4 个 flag)，默认关闭，不碰 `theory_align_v2` 路径。

**raw PPO effort（无 polish）vs e*=25，5 seeds：**

| seed | raw effort | err | 备注 |
| --- | --- | --- | --- |
| 42 | 22.698 | −9.21% | ramp 完整跑完 |
| 43 | 22.989 | −8.04% | ramp 完整跑完，几乎与 r5 raw mean 完全一致 |
| 44 | 25.663 | +2.65% | ramp 完整跑完 |
| 45 | 20.865 | −16.54% | ramp 完整跑完 |
| 46 | 21.417 | −14.33% | **ramp 全程未触发**，跑满 1500 updates 预算，全程停在 explore（κ=20） |

**vs r5 baseline（旧 ramp，同样是 raw PPO effort）：**

| | mean | std | mean\|err\| | range |
| --- | --- | --- | --- | --- |
| r5（旧 ramp） | 22.993 | **0.255**（紧） | 8.03% | [22.75, 23.42] |
| Component-2 | 22.726 | **1.666**（6.5× 大） | 10.16% | [20.86, 25.66] |

三条独立证据都指向同一方向：(1) mean 没有更靠近 e*，反而略差；(2) 方差暴涨 6.5×，
说明 retrain 不是可靠收敛，是碰运气散布；(3) 20% 的 seed（46）在整个训练预算内
**从未触发**该机制。

**机制诊断（2026-07-02 依据 mode/mean 轨迹修订）**：`EXP_raw<0.05` 的触发条件在
payoff 平台很平（Finding B）的区域里，在偏离 e* 的位置就被满足——四个触发的 seed
均在 mode≈17.9–18.2（mean≈20.8–21.1）处触发，离 e*=25 还差约 7 units，触发位置跨
seed 几乎相同。但触发后 κ 斜坡**并未冻结**策略：~80-update 的 ramp 窗口内 mode 继续
朝 e* 移动 +2.7–7.2 units（s42 17.9→22.4、s43 18.2→22.7、s44 18.2→25.4 越过 e*、
s45 17.9→20.6）。真实失败模式 = 过早触发 + 定长窗口走不完剩余距离；各 seed 窗口内
走的距离不同 → 6.5× 方差在窗口内产生。这与 §45（Finding A）旧 ramp 的"lock 冻结
undershoot"**不是同一个失败模式**，共同点只在结果：PPO 自身都没有可靠到达 e*。
（修正记录：旧表述"κ 斜坡冻结当时的均值、与旧 ramp 同一失败模式"已被轨迹数据证伪，
禁止复活。）

**处置**：代码保留（不删除——这是一个有完整 provenance 的真实负结果），默认关闭
（`--mode-conc-ramp` 需显式开启），不影响任何 r5/`theory_align_v2` 复现性。
Raw JSON：`results/three_players/convergence/ppo_3p_q35.0_seed{42..46}_c2_mode_conc_convergence.json`。
详见 `docs/tasks/component2-mode-conc-retrain/STATE.md`。

**待 Frank 拍板**：Claim B 是否就此定为论文 headline？若仍想追 Claim A，当前这版
Component-2 设计已被数据证伪，需要一个根本不同的 retrain 设计（例如把触发条件换成
"effort 接近某个已知范围"而非纯 exploitability，或大幅拉长/放缓 ramp）——这超出本次
授权范围，需要重新决策。

<!-- §G Component-2 spec 见 Phase0_reconstruction.md §9 决策3（该文件已上传到仓库根目录，此指针现在有效）。 -->
