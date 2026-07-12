# DIAGNOSIS.md — Two-Stage TEL-PPO Correctness Pass (Phase 0, read-only)

**Date:** 2026-07-12 · **Branch:** `feat/multistage-phase0` · **Scope:** T=2 code paths only
**Status:** Phase 0 complete. **No repo code was modified.** Awaiting approval before Phase 1.

This is the Phase-0 deliverable: (a) pipeline map, (b) defect table D1–D6 with
file:line evidence, (c) verifier audit, (d) corrected re-measurement of the
existing checkpoint, (e) proposed Phase-1 patch plan. Where a "confirmed defect"
did not reproduce as stated, that is called out with evidence (per the repo's
disagreement rule).

---

## ⚠️ Blocking finding (read first): there is no saved policy checkpoint

`find . -name '*.pt' -o -name '*.pth'` returns only venv artifacts — **no policy
checkpoints are persisted**. The runner holds the best checkpoint in memory
(`best["state_dict"]`, `run/run_multi_stage.py:269`) and discards it at process
exit; only the convergence JSON is written. The JSON stores the **mean-extracted
effort curves**, not the Beta `(α, β)` (verified: no `alpha`/`beta` keys in the
JSON).

Consequences for this pass:
- **Phase 2 item 1 ("run corrected eval on the existing checkpoint FIRST") is not
  literally executable** — there is no checkpoint object to load. Corrected
  *scalar* re-measurement from the saved curves IS possible and is done below
  (§D). A full `dReach^UCB` re-certification from saved curves is feasible for
  T=2 (needs only `ê_1(0)` + the `ê_2(d)` curve, both saved), but re-running the
  *policy* (e.g. to compare extraction rules) requires a retrain.
- **D6 (mean vs mode) cannot be answered on the existing artifacts** — mode needs
  `(α, β)`, which was never saved. This requires either a retrain with checkpoint
  persistence or an added `(α,β)` dump. Flagged in §E/T8.

---

## (a) Pipeline map

| Component | File | Key entry points |
|---|---|---|
| Config + validity gate | `config/multi_stage_two_players.py` | `config`, `validate()` (q_crit) |
| Theory / closed forms | `utils/theory_multistage.py` | `g1_two_stage`, `g2_two_stage`, `F_xi`, `f_xi`, `eq_utility_two_stage` |
| Env (sampled rewards) | `envs/multi_stage_env.py` | `step_batch`, `obs_batch`, `sample_exploring_starts_batch` |
| Agent (Beta policy, PPO) | `agents/ppo_multi_stage.py` | `MultiStageActorCritic` (280+), `mean_effort` (471), `effort_function` (486), `update` (511+) |
| **Training entry point** | `run/run_multi_stage.py` | `main` (212); rollout (50), eval (100), **ckpt select (252, 265–271, 281–284)**, report print (373–392) |
| **DP BR verifier** | `utils/dp_verifier.py` | `verify` (218), `verify_grid_refinement` (365) |
| Recovery metrics + gate | `utils/multi_stage_metrics.py` | `recovery_metrics` (50), `evaluate_gate` (155) |
| **Figures** | `tools/make_multistage_figures.py` | `fig1_recovery` (85), `fig2_calibration` (133) |
| Tables | `tools/make_multistage_tables.py` | — |
| Gate scorer | `tools/evaluate_gate.py` | — |
| Data (no ckpts) | `results/multi_stage/convergence/*.json` | 5-seed gated T=2 + `densecurve` T=2 |

---

## (b) Defect table

| ID | Symptom | Root cause (file:line) | Reproduced? | Minimal fix | Blast radius |
|---|---|---|---|---|---|
| **D1** | On-path expected stage-2 effort `E[ê_2(d_2)]` (target 46.67) is never computed/printed; the stage-2 recovery check against 46.67 is not performed | `recovery_metrics` computes RE_1/MAE_2/RMSE_2/RPE_2/PL_2 but **not** `E[ê_2]` (`utils/multi_stage_metrics.py:77–107`); T=2 report prints probe values only (`run/run_multi_stage.py:377–382`). `onpath_summary` *does* compute `E[ê_2]` via μ^e (`run/run_multi_stage.py:178–209`) but its per-stage value is **only printed for T≥3** (`:385–391`) and is **absent from the existing T=2 JSONs** (they predate the field) | **YES** | Add `E[ê_2(d_2)]` by quadrature to the T=2 report + JSON; print beside 46.67. Can reuse `onpath_summary` (μ^e ≡ f for T=2 under symmetric stage-1) | `run/run_multi_stage.py`, `utils/multi_stage_metrics.py` — shared with T≥3 (additive only) |
| **D2** | Figures "incorrect and incomplete; don't show both stages" | `fig1_recovery` plots only the **stage-2** curve (`tools/make_multistage_figures.py:98–126`); stage-1 appears only as a text annotation (`:119`), not a comparison; `E[ê_2]` vs 46.67 not annotated; `fig2` bars use **EXP/ΔW**, not the gate metric dReach (`:154–172`) | **PARTIAL** — see note below | Add a stage-1 comparison panel; annotate `E[ê_2]`; add a dReach calibration variant; add a T=2 Δ_t(d) figure | `tools/make_multistage_figures.py` only (regenerable) |
| **D3** | "Code still certifies via `EXP^UCB` with Monte-Carlo SE / 1.96" | **Does not reproduce as stated.** The verifier is fully deterministic (quadrature `dp_verifier.py:95–114`; closed-form terminal `w_l + ΔW·F_xi(y)` `:266–268`). No `1.96`, no SE, no `np.random` in any eval/cert path. Certification is `delta_sum_reachable/ΔW ≤ 0.03` at a **single** grid (`dp_verifier.py:340–342`; gate `multi_stage_metrics.py:171`) | **NO (already retired)** — but a real *residual* gap remains: cert uses **single-grid dReach**, not `dReach^UCB`; and `dReach^UCB` is **never computed** (only `exp_ucb`, `run/run_multi_stage.py:318`, which is reported, not gating) | Compute `dReach_coarse`, `dReach_fine`, `dReach^UCB = dReach_fine + |fine−coarse|`; certify on `dReach^UCB/ΔW ≤ 0.03`; report both | `utils/dp_verifier.py`, `run/run_multi_stage.py`, `utils/multi_stage_metrics.py` — shared with T≥3; **re-certification of all existing runs required** |
| **D4** | Notation drift `g` → should be `e, e*, ê` | Figure annotation uses `$g_1$` (`tools/make_multistage_figures.py:119`); JSON keys `g1`,`g2` (`utils/multi_stage_metrics.py:46–47, 105–106`; `run/run_multi_stage.py:334`). Learned side already uses `ê` (`e_hat_*`) | **PARTIAL (user-facing yes; internals are benchmark `g`)** | Rename user-facing labels/keys to `e_star_CF`, keep internal `g*_two_stage` function names (low-risk boundary) | Figures + JSON key rename; downstream table/figure readers must track key rename |
| **D5** | Checkpoint selection peeks at the certification verifier | Best ckpt selected by `ev["delta_sum_reachable"]` at grid `d_grid_sizes[-1]=201` (`run/run_multi_stage.py:118, 252, 265–271`); **final certification uses the same verifier at the same grid 201** (`:293–302`). Selection metric ≡ certification metric | **YES** | Choose ONE pre-specified rule: (i) final checkpoint after fixed budget, or (ii) select on a **coarse** validation grid (e.g. 101), certify **exclusively** on fine (201). Add per-checkpoint dReach trajectory diagnostic | `run/run_multi_stage.py` — shared with T≥3 |
| **D6** | Open: is Beta **mode** closer to CF than **mean**? | Agent implements **mean only** (`agents/ppo_multi_stage.py:471–483, 486–509`); no mode path. No `(α,β)` saved → cannot evaluate mode post-hoc | **N/A — unanswerable on existing artifacts** | Add mode extraction (guard α>1,β>1, else fall back to mean); compare on a checkpoint that saves `(α,β)` | Needs a retrain or `(α,β)` dump; agent shared with T≥3 |

**D2 note (pushback):** the current figures are **incomplete and carry notation
drift**, but I do not find them flatly "incorrect": `fig1_recovery` correctly plots
`ê_2(d)` against `e*_{2,CF}(d)` with the CF peak at ~70 and correct triangular
shape (`:99`), and `fig2` correctly separates the learned policy from the bad-policy
suite (`:150–177`). The defensible defects are: (1) no stage-1 *comparison* panel,
(2) `E[ê_2]` vs 46.67 not annotated, (3) `$g_1$` notation, (4) calibration uses EXP
rather than the gate metric. I recommend keeping the correct parts and fixing the
four specific gaps rather than a rewrite.

---

## (c) Verifier audit (`utils/dp_verifier.py`)

- **Transition expectation:** deterministic quadrature, **not** Monte Carlo.
  `_quadrature_nodes` (`:95–114`) builds `n_quad=129` trapezoidal nodes on
  `[−2q, 2q]` weighted by the triangular density `f_xi` and renormalized to sum 1.
  Continuations use `_smoothed_continuation` = Σ w·interp(landing+ξ) (`:117–140`).
- **Terminal integration:** closed form `terminal_W(y) = w_l + ΔW·F_xi(y)`
  (`:266–268`) — never interpolates the step reward R near 0. ✔ matches ground truth.
- **Certification formula:** `certified = delta_sum_reachable/ΔW ≤ epsilon_over_dw`
  (`:340–342`), with `delta_sum_reachable = Σ_t max over the BR-reachable support of
  Δ_t(d)` (`:328–338`). Δ_t(d) is the one-step deviation gap against the *learned*
  continuation (`:307–315`) — correct construction.
- **Reachable set:** computed empirically by forward-propagating a unit mass at d=0
  through the BR drift (`_forward_distribution`, `:174–215`; `br_reach`, `:326`),
  then `max` of Δ_t over cells with mass > 1e-9. This is a **generalization** of the
  spec's closed-form T=2 interval `R_2 = [Δe−2q, Δe+2q]`; needs a check that the two
  agree for T=2 (flagged in T2 below).
- **Grids / interpolation:** score gap `d ∈ [−D_max, D_max]`,
  `D_max = T(ē+2q)+50` (`:259–261`); linear interp off-grid with constant-tail
  clamping (`np.interp`, `:139`). Runner passes `d_grid=201, e_grid=201, n_quad=129`
  (`run/run_multi_stage.py:118–121, 293–296`; note `verify`'s own default is
  `e_grid=401` — the runner overrides to 201 via `verifier.e_grid_size`).
- **UCB:** `verify_grid_refinement` (`:365–416`) runs grids `[51,101,201]` and
  Richardson-extrapolates **EXP** only. The runner forms
  `exp_ucb = exp_fine + |exp_fine − exp_coarse|` (`:318`) — **reported, not gating**.
  **No `dReach^UCB` is computed anywhere.** ← the real D3 gap.

**Verdict:** the retired MC/SE/`1.96`/`EXP^UCB`-MC rule is **absent** from the
current code; the verifier already meets the "deterministic quadrature + closed-form
terminal + dReach" spec. The one substantive deviation from the *new* rule is that
certification gates on **single-grid dReach**, not the refinement `dReach^UCB`.

---

## (d) Corrected re-measurement of the existing T=2 runs (MEAN extraction)

Method: `E[ê_2(d_2)] = ∫ ê_2(δ) f(δ) dδ` by Gauss–Legendre quadrature (48 nodes per
half of `[−2q, 2q]`, split at 0), with `ê_2` linearly interpolated from the saved
`recovery_metrics.e_hat_2` (81-pt) curve. Quadrature self-test on the CF curve
reproduces the analytic targets exactly (`E[g2]=46.6667`, `g2(0)=70.0000`).
Script: `scratchpad/remeasure_T2.py` (outside repo; no repo edits).

| run | ê_1(0) | ê_2(0) | E[ê_2(d_2)] |
|---|---|---|---|
| **TARGET (closed form)** | **46.67** | **70.00** | **46.67** |
| seed42 (gateT2) | 49.13 | 64.02 | 45.58 |
| seed43 (gateT2) | 45.61 | 64.38 | 45.15 |
| seed44 (gateT2) | 49.12 | 66.42 | 44.00 |
| seed45 (gateT2) | 46.95 | 62.78 | 45.35 |
| seed46 (gateT2) | 51.16 | 63.78 | 45.80 |
| **gated 5-seed mean** | **48.40** | **64.28** | **45.18** |
| gated 5-seed std | 1.92 | 1.20 | 0.63 |
| seed42 (densecurve) | 45.97 | 60.65 | 45.19 |
| **mode extraction** | **N/A** | **N/A** | **N/A** — no `(α,β)` saved (see blocking finding) |

**Reading:**
- `ê_2(0) = 64.3` must be compared to **70.0**, not 46.67 (the D1 trap). It sits
  ~8% below the myopic peak — the expected μ\*(κ) exploration smoothing.
- `E[ê_2(d_2)] = 45.2` vs target 46.67 — **~3.2% low; recovered.** This is the
  quantity D1 says is missing, and it shows the policy is on-path fine even though
  the peak is smoothed.
- `ê_1(0) = 48.4` vs 46.67 — ~3.7% high; recovered (direction is a slight
  *overshoot*, not undershoot).

**Bottom line: D1 was hiding a policy that is actually acceptable on stage-2 on-path
recovery.** The missing print, not the policy, is the defect.

---

## (e) Proposed Phase-1 patch plan (per T1–T9; deviations flagged)

- **T1 (D1) — report E[ê_2(d_2)].** Add GL quadrature over Triangular[−2q,2q]
  (≥32 nodes/half); print `ê_1(0)|46.6667`, `ê_2(0)|70.0`, `E[ê_2(d_2)]|46.6667`;
  add per-d CF column to the probe table. **Deviation to confirm:** `onpath_summary`
  already computes `E[ê_2]` via μ^e (≡ ∫ê·f for T=2). Recommend *reusing* it rather
  than adding a second quadrature path — decide which is canonical.
- **T2 (D3) — dReach^UCB gate.** Compute `dReach` at coarse+fine, form
  `dReach^UCB = dReach_fine + |fine−coarse|`, certify on `≤0.03`, report both
  `dReach_fine/ΔW` and `dReach^UCB/ΔW`. Keep EXP + its UCB as diagnostics.
  **Flag:** this changes the gate → **all existing T=2..5 runs must be
  re-certified**; margins may tighten (watch T=5 seed 46). **Also verify** the
  empirical reachable set equals the spec's `R_2=[Δe−2q, Δe+2q]` for T=2.
- **T3 — Experiment-1 metrics.** Already implemented and formula-matching
  (`multi_stage_metrics.py:77–100`: RE_1=AE_1/(1+e*_1), RPE_2=RMSE_2/(1+mean g2),
  PL_2=U(e*)−U(ê)). Action: document the exact stage-2 grid and any validity
  sub-region excluded (ties to T9).
- **T4 — calibration + falsification table.** The bad-policy suite exists in `fig2`
  but only as EXP bars. Add a **table** reporting BOTH EXP and dReach for
  {e*_CF, ê, const-lo, const-hi, one-stage-repeat, no-gap} and the CF error floor.
- **T5 (D2) — figures.** Add stage-1 comparison (bar/point ê_1(0) vs e*_{1,CF}(0));
  annotate `E[ê_2]` vs 46.67 on the stage-2 panel; add Δ_t(d) figure over the
  reachable region; convert `$g_1$` → e-notation.
- **T6 (D4) — notation.** Rename user-facing labels + JSON keys (`g1`→`e_star_1_CF`,
  etc.); leave internal `g*_two_stage` function names. Update the table/figure
  readers that consume those keys.
- **T7 (D5) — checkpoint rule.** Pick ONE and record it: default = final checkpoint;
  or select on **coarse** grid (101) and certify **only** on fine (201). Add the
  per-checkpoint dReach-vs-training trajectory diagnostic (isolated-dip flag).
- **T8 (D6) — mean vs mode.** Add mode extraction (α>1,β>1 guard, else mean).
  **Blocked on artifacts:** requires a checkpoint that saves `(α,β)` → a retrain (or
  an `(α,β)` dump added to the runner). This is a **Phase-2 retrain decision**, not a
  pure Phase-1 edit.
- **T9 — SOC / validity.** `validate_two_stage_params` already scans validity
  (`utils/theory_multistage.py`); add the numeric "global argmax of Q_2(d,·) == interior
  FOC" check over the eval region and emit the parameter inequality (in ΔW, k, q²) as
  a short paper note.

**Cross-cutting blast-radius note:** `utils/dp_verifier.py`,
`utils/multi_stage_metrics.py`, `agents/ppo_multi_stage.py`, and
`run/run_multi_stage.py` are all shared with the T≥3 paths. Every edit above is
either additive (T1, T4, T5) or changes certification semantics for **all** T (T2,
T7) — the latter require re-certifying T=3/4/5, not just T=2. **No one-stage code
is touched** (that path uses `agents/ppo_two_players_clean.py` etc., disjoint from
these modules).

---

## STOP — awaiting your review

No repo code was modified in Phase 0. On approval I will proceed to Phase 1 (fixes
T1–T9), flagging the two items that cannot be completed without a retrain (T8
mean-vs-mode; and any policy-level re-run), and the certification-semantics changes
(T2, T7) that will require re-certifying the existing T=2..5 runs in Phase 2.
