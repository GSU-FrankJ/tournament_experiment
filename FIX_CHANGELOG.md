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
