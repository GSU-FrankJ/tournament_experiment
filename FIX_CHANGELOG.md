# FIX_CHANGELOG — audit remediation (branch `fix/audit-remediation`)

Running log of spec-compliance fixes per `AUDIT_REPORT.md`. Decisions fixed up-front:
eps_eq = 0.03 everywhere · max PPO updates = 1500 · theory-align-v2 is the single canonical config.
No training re-runs in this session; no experimental number fabricated or hand-edited.

---

## 1. `fix: train three-player env on sampled rank rewards`

**Spec rationale:** training must observe only sampled tournament outcomes (one winner gets w_H,
others w_L, minus k·e²); closed-form win-prob/expected-utility must not touch the env step, reward,
or policy update.

- `envs/three_players_env.py` — `step()` now draws `eps ~ U(-q,q)`, ranks realized outputs,
  pays w_H to the winner / w_L to the two losers, subtracts k·e_i² (mirrors the 2-player env,
  including uniform random tie-break). Removed the closed-form `"expected"` reward mode, the
  `"hybrid"`/`"pairwise_binary"` modes, `_win_probs`, the MC shortcut machinery, and the
  reward-mode constructor knobs. Kept `expected_utility_gradient` as an explicitly
  EVAL/BASELINE-ONLY helper (used by the numerical gradient reference).
- `run/run_three_players.py` — dropped reward-mode/noise-scale plumbing and the corresponding
  CLI flags (`--reward-mode`, `--noise-scale`, `--binary-rewards`); **removed the COMA
  counterfactual baseline** (`_coma_baseline` + `--coma-k`), which subtracted closed-form
  `win_prob_three_players`-based expected utilities from realized training rewards when enabled —
  the same class of leakage, one flag away. `_stochastic_fd_gradients_3p` now plainly delegates
  to the env's closed-form gradient (baseline behavior unchanged this session; see §"Known
  leftovers").
- **Safeguard test:** `tests/test_three_players_env_sampled.py` — fixes effort profiles
  (25,25,25), (20,25,30), (5,50,95) at paper params (k=0.001, w=(6.5,3), q=35), averages
  120,000 sampled rewards, compares against the OLD env's closed-form EU (normalized
  `win_prob_three_players`, replicated in-test). **PASS** — max |sampled mean − closed form| =
  0.0068, tolerance max(6·SE, 0.02) ≈ 0.029. Committed only after this passed.

**Effect on results:** none yet — no runs re-executed. All existing 3P convergence JSONs were
produced by the old closed-form reward path and are now non-canonical pending re-runs.

---

## 2. `fix: train different-cost env on sampled rank rewards`

**Spec rationale:** same train/eval invariant — het-cost reward must be the realized
`w_H`/`w_L` by rank minus `k_i e_i²`, with `y_i = e_i + ε_i`, `ε ~ U(-q,q)`; the old `step()`
returned the closed-form `w_L + p·(w_H−w_L) − k_i e_i²` (exact triangular-CDF p).

- `envs/different_cost_env.py` — `step()` now samples noise, ranks realized outputs, pays
  w_H/w_L, subtracts per-player cost (mirrors `TwoPlayersEnv.step`, incl. uniform tie-break).
  Added a per-env RNG (constructed once per run; advances across steps). `expected_utility`
  kept verbatim but documented EVALUATION/BASELINE-ONLY (it is the FD-baseline oracle and the
  reference for the safeguard test).
- **Safeguard test:** `tests/test_different_cost_env_sampled.py` — profiles (38.03, 27.66)
  (the analytical equilibrium at q=35), (30,30), (60,15) at paper params
  (k1=0.0004, k2=0.00055, w=(8,5.5)); 120,000 sampled draws vs the OLD closed-form reward.
  **PASS** — max |diff| = 0.0053, tolerance ≈ 0.021. Committed only after this passed.

**Effect on results:** none yet; existing dc JSONs (incl. `r4_dc_final`) were produced under
closed-form rewards and are non-canonical pending re-runs.

---

## 3. `fix: train different-ability env on sampled rank rewards`

**Spec rationale:** same train/eval invariant, het-ability variant — performance is
`y_i = e_i + l_i + ε_i`, `ε ~ U(-q,q)`; reward is realized `w_H`/`w_L` by rank minus `k_i e_i²`.
The old `step()` returned closed-form `w_L + p₁(win)·(w_H−w_L) − k_i e_i²` (exact triangular CDF).

- `envs/different_ability_env.py` — `step()` now samples noise, ranks realized ability-shifted
  outputs, pays w_H/w_L, subtracts cost (uniform tie-break). Added per-env RNG (constructed once
  per run). `probability_win_player1` / `compute_utility` / `analyze_equilibrium` /
  `compute_gradients` kept verbatim but documented EVALUATION/BASELINE-ONLY (FD-baseline oracle
  + safeguard-test reference). `info["win_probabilities"]` replaced by sampled
  `noises`/`outputs`/`winner` (no consumer read the old key; the PPO loop discards info, and
  `tools/check_env_noise_determinism.py` expects the new keys).
- **Safeguard test:** `tests/test_different_ability_env_sampled.py` — profiles (46.43, 46.43)
  (analytical symmetric equilibrium at q=35), (40,50), (20,30) at paper params
  (l=(10,5), k=0.0005, w=(6.5,3)); 120,000 sampled draws vs the OLD closed-form reward.
  **PASS** — max |diff| = 0.0061, tolerance 0.03. Committed only after this passed.

**Effect on results:** none yet; existing da JSONs (incl. `r4_h1_long`) were produced under
closed-form rewards and are non-canonical pending re-runs.

---

## 4. `docs: state sampled-training invariant in CLAUDE.md and README`

**Spec rationale:** the repo docs codified the leakage violation as an "invariant", which would
steer future edits back into non-compliance.

Old wording replaced:
- `.claude/CLAUDE.md:48`: "**`envs/`** — Gym-like environments. Closed-form expected utilities
  (no stochastic sampling)"
- `.claude/CLAUDE.md:112` (Critical Invariants): "**Closed-form expected utilities** — no
  stochastic noise during rollouts"
- `README.md:91`: "Two-player environment uses closed-form expected utilities (no stochastic
  noise is sampled during rollouts)."

New invariant (both files): training uses sampled outcomes only — fresh uniform noise, realized
rank, w_H/w_L prizes minus cost; closed-form win-prob / expected-payoff / e* are
evaluation/baseline-only and never enter training rewards or policy updates.

---

## 5. `fix: make 2P gradient baseline true MC-FD with CRN, drop symmetry projection`

**Spec rationale (Appendix A):** the baseline is MC-FD gradient play — sampled performance with
common random numbers, central finite-difference step delta, projected gradient ascent,
simultaneous update, tolerance tau; NO closed-form win probability, no symmetry projection.

- `run/run_two_players.py`:
  - `gradient_descent_two_players` now calls `_stochastic_fd_gradients` (sampled payoffs; one
    shared noise+tie-break batch reused for all four perturbed evaluations = CRN) instead of
    `_closed_form_fd_gradients`, which is deleted.
  - Removed the periodic symmetry enforcement (`e1 = e2 = avg` every 50 steps) and the
    `symmetry_gap < symmetry_tol` term in the stop criterion; symmetry gap is now reported only.
    Stop = `grad_norm < tol AND max step change < tol` (tolerance tau). Projection via effort
    clipping and simultaneous updates unchanged.
  - Dropped now-dead `--grad-symmetry-enforce` / `--grad-symmetry-tol` CLI flags and the
    corresponding parameters.
  - The saved `gradient_mode = "stochastic_uniform"` label is now truthful (previously a
    mislabel over closed-form FD).
- **Verification:** py_compile + 300-iteration smoke (no files written): from init (30, 70) both
  efforts move toward e* = 45.45 under sampled gradients; values finite and projected.

**Effect on results:** existing `gradient_*_convergence.json` files were produced by the old
closed-form solver and no longer reflect this code path; baseline re-runs are required before the
gradient rows/figures are quoted (out of scope this session).

**Known leftover (intentional):** the 3P/dc/da gradient baselines still use the envs'
closed-form helpers (now explicitly EVAL/BASELINE-ONLY). Porting the sampled MC-FD pattern to
those runners is follow-up work; this session's target was the 2P implementation.

---

## 6. `refactor: remove dormant e*-in-loss paths from PPO agents`

**Spec rationale:** analytical e* must never appear in the policy update. Two dormant
(coefficient-0.0 everywhere) but one-flag-away paths existed:

1. `theory_align_v2_br_coef` — reconstructed `e* = w_gap/(4qk)` from the de-normalized state
   inside the minibatch loss and penalized `(mean_effort − e*)²`
   (`agents/ppo_two_players_clean.py` and `agents/ppo_three_players.py`).
2. 3P `br_reg_coef` — runner computed `br_target = e_star_three_players(q, w_h, w_l, k)` each
   update and passed it into `agent.update(br_target=…)` as a squared-error target.

Removed entirely:
- both agents: the `theory_align_v2_br_coef` config field, the `want_dist` br-term, and the
  e*-loss block; 3P agent additionally loses `br_reg_coef` field, the `br_target` parameter of
  `update()`, and the br-reg loss block.
- `run/run_two_players.py`: br_coef read/passthrough and the preset line.
- `run/run_three_players.py`: PPOConfig passthrough, br_reg plumbing/print, the
  `br_target = e_star_three_players(...)` block (update now `agent.update()`), preset lines,
  `--br-reg-coef`/`--br-reg-warmup` CLI flags, and the now-orphaned `local_best_response_3p`
  FOC-bisection helper (zero callers).

**Verification:** `grep -rn "br_coef|br_reg|br_target|local_best_response"` over all Python
directories → zero matches. py_compile clean. Smoke: both agents construct and run a full
store→update cycle under the canonical theory-align-v2 config (var_coef=5e-2) on CPU.

---

## 7. `fix: unify exploitability tolerance to eps_eq=0.03 across configs`

**Decision (fixed up-front):** eps_eq = 0.03 everywhere; max PPO updates = 1500.

- `config/one_stage_three_players.py`, `config/one_stage_different_cost.py`,
  `config/one_stage_different_ability.py`: `exploit_eps` 0.05 → **0.03** (2P config was already
  0.03; the canonical Round-4 dc/da runs already passed `--exploit-eps 0.03` on the CLI, so for
  dc/da this aligns defaults with the runs actually used).
- `run/run_different_cost.py` / `run_different_ability.py`: stale `--exploit-eps` help text
  ("default … 0.05") corrected to 0.03.
- `paper/generator/config.py`: diagnostic `CONVERGENCE_CONFIG.exploit_threshold` 0.05 → **0.03**
  so post-hoc diagnostics use the same tolerance as the runner gates.
- **max_updates:** verified `"max_updates": 1500` in all four configs and grep found no
  500-update reference anywhere — nothing to change (decision already satisfied by code).

**Note:** existing 3P round3 JSONs were stopped under eps=0.05 (their recorded measured
exploitability at stop was < 0.03, but the gate was 0.05); re-runs under the unified 0.03 gate
are part of the post-env-fix re-run batch anyway.

---

## 8. `fix: report steps-to-convergence from the method's own verification`

**Spec rationale:** convergence is defined by the method's verification module (stability screen
AND exploitability streak ⇒ stop; else budget ⇒ NC). The all-NC Tables 3/4 came from a second,
mis-specified post-hoc gate (|e−e*| < 0.5 for 20 consecutive logged updates, min 100 updates,
`paper/generator/config.py:CONVERGENCE_CONFIG`) that early-stopping runs can never satisfy.

- `paper/generator/extract.py`:
  - flat + nested loaders now carry the run-level verification verdict as columns
    (`stop_reason`, `stopped_at_update` — both already recorded in every convergence JSON).
  - new `get_verified_convergence_step(df)`: per run, `verified = (stop_reason ==
    "exploitability")`, `convergence_update = stopped_at_update` (PPO update index) when
    verified, NaN (→ "NC") when the run hit `max_updates`.
  - `get_convergence_step` (the effort-band detector) kept but re-documented as
    **DIAGNOSTIC-ONLY** with an explanation of why it must not be used for paper tables.
- `paper/generator/metrics.py` `compute_summary_metrics`: PPO runs now take
  `converged`/`convergence_step` from the recorded verification verdict; the effort-band
  detectors remain only as fallback for runs without a verdict (gradient baseline).
- `paper/generator/tables.py`: `final_summary` consumes `get_verified_convergence_step`;
  column renamed "Steps to Conv." → **"Conv. Update (verified)"** (unit = PPO updates, making
  the semantics change explicit); `convergence_comparison` caption now states the per-method
  semantics (gradient: effort-band iteration; TEL-PPO: verified-stop update).
- **Method stopping logic untouched** (runners unchanged in this commit).

**Read-only validation (no artifacts written):** loading all current JSONs through the modified
pipeline, every (experiment, q) baseline group is now verified 5/5 with finite mean
convergence_update — e.g. two_players q=35/45/55 → 53/67/93 updates; three_players 99/53;
dc 35/33; da 34/32 — where the old criterion produced all-NC. Values are read directly from each
run's recorded `stopped_at_update`; nothing fabricated.

**Deliberately NOT done here:** `paper/tables/*` were not regenerated. Regeneration must wait
for (a) registry canonicalization (pre-fix vs Round-3/4 run selection — audit known issue) and
(b) the post-env-fix re-runs; regenerating now would mix non-canonical runs.

---

## 9. `feat: wire Fig-7 ablation toggles into the 3P runner`

**Spec rationale:** Fig 7's component ablation requires the stability screen and the
exploitability verification to be independently disable-able.

- **Status check:** `run_two_players.py`, `run_different_cost.py`, `run_different_ability.py`
  already expose AND plumb `--disable-cheap-gate` / `--disable-exploitability` (verified via
  grep + `--help`). `run_three_players.py` DECLARED both flags but never read them — args were
  silently ignored.
- `run/run_three_players.py`: flags now flow `args → cfg → run_ppo`;
  `--disable-cheap-gate` forces the stability screen to pass (same semantics as 2P);
  `--disable-exploitability` suppresses all exploitability evals so the run goes to budget with
  `stop_reason="max_updates"`; both toggles are recorded in the output JSON's `exploit_config`
  for provenance. No behavior change when neither flag is passed.
- **Verification:** py_compile; introspection confirms run_ppo reads both toggles; `--help` on
  all four runners lists both flags. Nothing executed beyond CLI help.

---

## Session verification summary

- `tests/test_three_players_env_sampled.py` — PASS (max |diff| 0.0068, tol ≈ 0.029)
- `tests/test_different_cost_env_sampled.py` — PASS (max |diff| 0.0053, tol ≈ 0.021)
- `tests/test_different_ability_env_sampled.py` — PASS (max |diff| 0.0061, tol = 0.03)
- py_compile clean on every touched file; agents smoke-tested (construct + update) under the
  canonical theory-align-v2 config; 2P MC-FD solver smoke-tested (300 iters, no files written);
  verified-convergence extraction validated read-only against all existing JSONs (every
  baseline group 5/5 verified with finite conv updates vs all-NC before).
- No training runs executed; no experimental numbers created or edited; `results/` and
  `paper/{tables,figures,data}` artifacts untouched.

## Follow-ups intentionally left for later sessions

1. Re-run 3P/dc/da training (and 2P/all gradient baselines) under the fixed sampled-reward
   envs and unified eps=0.03 — all existing non-2P PPO results and all gradient results are
   non-canonical against the fixed code.
2. Port the sampled MC-FD baseline to the 3P/dc/da runners (2P implementation is the reference).
3. Registry canonicalization (select Round-3/4 runs as baseline, exclude pre-fix), then
   regenerate tables/figures via `python -m paper.generator make_all`.
4. Run the Fig-7 component-ablation batch (full / no-stability-screen / no-exploitability).
5. DELETE-bucket cleanup per AUDIT_REPORT.md §3 (explicitly out of scope this session).

---

# Follow-up session (post-review)

## 10. `fix: port true MC-FD baseline to the three-player gradient solver`

**Spec rationale (Appendix A):** the numerical reference is MC-FD gradient play — sampled
payoffs with common random numbers, central finite differences, projected simultaneous ascent,
tolerance tau, no closed-form win probability, no symmetry projection. The 3P solver still used
the env's closed-form analytic gradient plus a symmetry projection every 50 steps.

- `envs/three_players_env.py`: added `draw_noise_batch(batch_size)` → `(eps (n,3), tie_breaks)`
  mirroring `TwoPlayersEnv.draw_noise_batch` (CRN-friendly batches from the env RNG).
- `run/run_three_players.py`:
  - new `_batch_payoffs_uniform_3p`: vectorized sampled payoffs — rank realized
    `y_i = e_i + eps_i`, winner w_H / losers w_L, minus k·e_i²; exact ties (measure-zero)
    resolved with the provided tie-break draws.
  - `_stochastic_fd_gradients_3p` now computes central differences of those sampled payoffs
    under ONE shared CRN batch for all six perturbed evaluations (was: delegation to
    closed-form `env.expected_utility_gradient`, which remains EVAL-ONLY).
  - `gradient_descent_three_players`: uses the sampled gradient; symmetry enforcement and the
    symmetry stop-term removed (symmetry reported only); stop = grad_norm < tol AND max step
    change < tol; `--grad-symmetry-enforce`/`--grad-symmetry-tol` CLI flags dropped.
- **Safeguard test:** `tests/test_three_players_mcfd_gradient.py` — at (25,25,25) (=e*, q=35)
  and (20,25,30), the mean of 50 independent CRN-batch gradients (8192 samples each) must match
  the central difference of the closed-form EU at the same delta (valid because
  E[sampled payoff] = closed-form EU, proven by the env equivalence test).
  **PASS** — max |diff| = 0.00088, tolerance ≈ 0.003-0.004. Committed only after this passed.

---

## 11. `fix: port true MC-FD baseline to the het-cost gradient solver`

**Spec rationale:** same Appendix-A port; additionally the dc solver's stop rule contained
`max_gap < tol` — distance to the analytical (e1*, e2*) — i.e., an e*-dependent termination
criterion. The baseline must not condition its stopping on the answer; gaps stay logged
(evaluation-only).

- `envs/different_cost_env.py`: added `draw_noise_batch` (eps1, eps2, tie_breaks from the env
  RNG; exact mirror of `TwoPlayersEnv.draw_noise_batch`).
- `run/run_different_cost.py`: new `_batch_payoffs_uniform_dc` (sampled rank payoffs with
  per-player k_i); `_compute_gradients_different_cost` now does central differences of sampled
  payoffs under ONE shared CRN batch (was: closed-form `env.expected_utility` differences);
  `max_gap < tol` removed from the stop rule (now grad_norm < tol AND max step change < tol);
  no symmetry machinery existed (asymmetric equilibrium) and none was added.
- **Safeguard test:** `tests/test_different_cost_mcfd_gradient.py` — at (38.03, 27.66)
  (analytical equilibrium, q=35; gradients ≈ 0) and (30, 30): mean of 50 CRN-batch gradients
  (8192 samples each) vs closed-form FD at the same delta.
  **PASS** — max |diff| = 0.00038, tolerance ≈ 0.002. Committed only after this passed.

---

## 12. `fix: port true MC-FD baseline to the het-ability gradient solver`

**Spec rationale:** same Appendix-A port with ability-shifted outputs
`y_i = e_i + l_i + ε_i`; the da solver also had the e*-dependent `max_gap < tol` stop term
(removed; gaps logged for evaluation only).

- `envs/different_ability_env.py`: added `draw_noise_batch` (CRN batches from the env RNG).
- `run/run_different_ability.py`: new `_batch_payoffs_uniform_da` (sampled rank payoffs over
  ability-shifted outputs); `_compute_gradients_different_ability` now does central differences
  of sampled payoffs under ONE shared CRN batch (was: closed-form `env.compute_utility`
  differences) and gains the previously-missing `num_samples` parameter, plumbed through
  `gradient_descent_different_ability` → `run_gradient` → new `--grad-samples` CLI flag
  (default from config `gradient_num_samples`, matching dc); `max_gap < tol` removed from the
  stop rule.
- **Safeguard test:** `tests/test_different_ability_mcfd_gradient.py` — at (46.43, 46.43)
  (analytical symmetric equilibrium, q=35; gradients ≈ 0) and (40, 50): mean of 50 CRN-batch
  gradients (8192 samples each) vs closed-form FD at the same delta.
  **PASS** — max |diff| = 0.00041, tolerance ≈ 0.0036-0.0043. Committed only after this passed.

With commits 10-12, all four scenarios' numerical baselines are sampled MC-FD with CRN and
contain no closed-form win probability, no symmetry projection, and no e*-dependent stopping.
All existing `gradient_*_convergence.json` artifacts predate this and require re-runs before
being quoted.

---

## 13. `docs: registry canonicalization plan (PLAN ONLY — awaiting approval)`

**Deliverable:** `docs/registry_canonicalization_plan.md`. **Nothing executed** — `make_all`
NOT run, `BASELINE_OVERRIDES` NOT edited, no results files moved/committed.

New read-only findings sharpening the audit's registry issue:
- 28 duplicate run keys in the current registry: the 10 real 3P collisions (pre-fix and
  `round3_baseline` files BOTH carry JSON `ablation_name="baseline"` — the Round-3 batch used
  `--output-tag`, which only renames the file), whose trajectories the loader silently MERGES
  per seed; plus 18 Set 1/Set 2 pairs that merge because run-level groupbys omit
  `weight_variant`.
- dc/da Round-4 canonical runs are cleanly tagged in JSON (`r4_dc_final`, `r4_h1_long`) — a
  pure `BASELINE_OVERRIDES` fix; 3P additionally needs a narrow filename-tag precedence rule
  in the registry plus a duplicate-key guard.

Plan summary (full details + exact override dict in the plan doc):
C1 populate BASELINE_OVERRIDES (3P→round3_baseline, dc→r4_dc_final, da→r4_h1_long; 2P
unchanged) · C2 registry filename-tag precedence + duplicate-key guard · C3 add
weight_variant to run-level groupbys · D1 commit the 5 untracked da q=55 r4 JSONs +
summary.csv · V1 read-only dry-run acceptance check (5 seeds/cell, post-fix tags only, zero
duplicates, all stop_reason=exploitability) · E regeneration via make_all — **blocked on
owner approval**, and even then interim until the post-env-fix `r5_sampled` re-run wave
(gradient re-runs additionally need overwrite protection: gradient filenames carry no tag/seed).

---

## 14. Canonicalization executed C1→V1 (owner-approved); E still blocked

**Owner approval (2026-06-10):** 3P → `round3_baseline`, dc → `r4_dc_final`,
da → `r4_h1_long` — explicitly as TEMPORARY picks (all closed-form-trained legacy; the
`r5_sampled` sampled-training wave replaces them). Recorded decisions: (1) da's pick is
standard-config, NOT theory-align-v2 — the da v2-vs-standard question is an **open decision
for r5_sampled** (run both under sampled training and compare); (2) provenance footnotes
required at E: round3's gate was eps=0.05 (measured exploitability also passes 0.03), and
r4_dc_final has no committed launch script. Both are recorded as comments on
`BASELINE_OVERRIDES` and in the plan doc.

- `5999165` **C2** `fix: keep filename batch tags distinct in run registry, guard duplicates`
  — a "baseline" ablation_name from metadata/JSON no longer erases a more specific filename
  tag (fixes the 3P `--output-tag` collision at the source: round3 files now classify as
  `round3_baseline`); plus a duplicate-run-identity guard (warn with both paths, keep newer).
  Verified: 112 runs, 0 duplicate identities, 0 guard warnings, 10 round3 + 10 distinct
  pre-fix 3P runs.
- `58fcc26` **C3** `fix: key run-level aggregation on weight_variant` — weight_variant added
  to per-run groupbys in `get_convergence_step` / `get_verified_convergence_step` /
  `compute_summary_metrics` (carried through `SummaryMetrics`), and table cells filter
  weight_variant=baseline. Verified: 2P Set 1 / Set 2 now 15+15 separate rows.
- `d048418` **D1** `chore: add da q=55 Round-4 (r4_h1_long) results and summary rows` —
  data-only commit of the five untracked JSONs + summary.csv.
- `2044608` **C1+V1** `feat: canonicalize approved Round-3/4 runs as interim baseline` —
  BASELINE_OVERRIDES populated with the approved tags + provenance/caveat comments;
  `tests/test_registry_canonicalization.py` added as the repeatable V1 acceptance check.

**V1 acceptance result (read-only): PASS** — 0 duplicate identities across 112 runs; exactly
seeds {42..46} in every (experiment, q) baseline cell; every promoted row sourced from the
approved post-fix tag (verified against registry file paths); all stop_reason="exploitability"
with conv updates: 3P q35 [309,300,305,308,307], q55 [304,307,307,305,303]; dc q35
[300,307,304,307,302], q55 [306,301,301,308,302]; da q35/q55 all [1000]×5 (min_updates floor);
2P q35 [69,49,49,49,49], q45 [69,69,59,69,69], q55 [99,89,109,89,79]; Set 2 kept separate
(15 PPO + 3 gradient).

**NOT executed:** E (`python -m paper.generator make_all`) — awaiting explicit go-ahead.

---

## 15. `fix: seed/tag-qualify gradient result filenames, refuse overwrites`

**Rationale:** gradient result filenames carried no seed or tag, so any re-run silently
clobbered the previous file — the audit already caught one such silent overwrite
(`ppo_3p_q35.0_seed42_baseline_convergence.json`, clobbered by an untagged Round-3-era run).
This had to land BEFORE the r5_sampled re-run wave touches any gradient baseline.

- All four runners (`run_two_players.py`, `run_three_players.py`, `run_different_cost.py`,
  `run_different_ability.py`): gradient outputs now write to
  `<prefix>_gradient…_q{q}_seed{seed}[_{ablation}]_convergence.json` — the same
  qualification scheme as the PPO outputs (`--ablation-name` is now threaded into the 3P/dc/da
  gradient paths too, and the seed is also embedded in the JSON body for registry use).
- New `--force` flag on every runner; without it, writing onto an existing gradient file
  raises `FileExistsError` with a remediation hint instead of overwriting.
- `paper/generator/run_registry.py`: the 3-player pattern now accepts
  `(ppo|gradient)_3p_q…_seed…[_tag]` so the new seeded gradient filenames stay discoverable
  (dc/da/2P patterns already covered seeded gradient files; legacy un-seeded names still parse).

**Verification (no training):** py_compile on all five files; `_parse_filename` round-trips
seven new/legacy gradient filename forms incl. `r5_sampled` tags and the Set-2 weight variant;
`--force` visible in all four `--help`s; V1 acceptance test still PASS. Existing
`gradient_*` files on disk are untouched (legacy names remain readable; new runs simply write
new names).
