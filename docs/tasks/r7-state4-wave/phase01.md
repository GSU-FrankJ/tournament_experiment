# Phase 01 — implement 4-dim state, smoke, launch wave

## Code changes (all uncommitted at launch; snapshot in results/r7_state4/code_state.diff)

| File | Change |
|---|---|
| agents/ppo_two_players_clean.py | `state_dim` 3→4; `state_from_params` adds `l_gap=0.0`, 4th component `l_gap/10` |
| agents/ppo_three_players.py | same |
| run/run_two_players.py:683 | `state_dim=4` |
| run/run_three_players.py:663 | `state_dim=4` (was missed by the first grep — hardcoded like 2P/dc) |
| run/run_different_cost.py:559 | `state_dim=4` |
| run/run_different_ability.py | `state_dim=4`; rollout builds s1 (`l_gap=l1−l2`) and s2 (`l_gap=l2−l1`), acts and stores per player; per-update + final eval report `effort1`/`effort2`/mean; history gains `effort1`,`effort2`; `final` gains `effort1`,`effort2`; result gains `final_effort1/2`; p1_win now at (e1,e2) |
| utils/exploit_asymmetric.py | `_sample_policy_efforts(..., l_gap=0.0)`; `eval_exploitability_asymmetric` passes `l_gap_i = l_i − l_{−i}` per player (0 unless het-ability); MockAgent stub → 4-dim |
| tools/one_stage_mc_adapter.py | both adapter `state_from_params` accept `l_gap` and emit 4-dim (they feed the same estimator) |
| tools/diagnose_data_provenance.py, tools/audit_rollout_modes.py, tools/verify_rollout_modes.py | explicit `state_dim=3` → 4 so the diagnostics stay runnable |
| config/one_stage_two_players.py:110 | exploit `M` 8192 → 16384 (unified; comment updated) |

## Verification before launch

- `py_compile` on all edited files: OK.
- CPU sanity: state shape (1,4); l_gap ±5 → 4th comp ±0.5; per-player means
  differ at init (|Δ|≈0.20); store/update round-trip OK; asymmetric
  exploitability with l1=10,l2=5 runs; v2 head path OK; 3P agent OK.
- GPU smoke (tag `r7smoke*`, seed 999, GPUs 0–4, short budgets):
  2P (20 upd), 3P, dc, da std (30 upd), da v2 — all training, no crashes;
  da logs show per-player e1/e2. Smoke JSONs left in place under
  results/*/convergence/*r7smoke* (clearly tagged, not matched by any
  baseline glob).

## Wave

`run/r7_state4_wave.sh --launch` — 100 jobs, LPT order, 8 tmux workers
(`r7w_gpu0..7`), venv python, per-job logs + manifest in `results/r7_state4/`.
Cmdlines byte-identical to r5 templates except tags. ~182 GPU-h ≈ 24 h.
