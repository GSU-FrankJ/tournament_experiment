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
