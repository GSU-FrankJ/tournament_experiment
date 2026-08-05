# Task: r7-state4-wave

## Goal

Rerun the full one-stage baseline + Table-3 ablation matrix under the owner's
4-dim state (docs/Figures&Tables073026.docx red items 1–4):

    s_i = [ q/60,  k_i/1e-3,  (w_H − w_L)/10,  (l_i − l̄_{−i})/10 ]

with the training verifier unified at M=16384 (two-player was the only 8192).

## Scope boundaries

- DO NOT touch `results/**/*r5_*` — r5 is the 3-dim comparison arm for the
  owner's accuracy-degradation decision (red item 2).
- New tags only: `r7_state4`, `r7_state4_std`, `r7_state4_v2`,
  `r7_fig7_no_stability`, `r7_fig7_no_exploit`.
- het-cost keeps its TWO independent agents (owner asked for identical *input*
  architecture, not merged models).
- Gradient (MC-FD) baselines are NOT rerun — no network, state-independent.
- No paper-generator promotion in this task; comparison first.

## Key decisions

- Only het-ability's 4th dim is non-zero (±0.5). For 2P/3P/dc it is
  identically 0 → mathematically inert (no forward contribution, zero
  gradient); those groups are rerun anyway because fan-in 3→4 changes the
  init scale (±1/√3 → ±1/√4) and shifts the RNG stream.
- het-ability now feeds per-player states (s1: +Δl/10, s2: −Δl/10) to the
  shared agent: effort symmetry becomes a LEARNED outcome, not an
  architectural constraint. Decision metric afterwards: |e1−e2| vs cross-seed
  SD, and |err| vs the r5 3-dim arm. If accuracy degrades → keep 4-dim only
  for het-ability (owner's red item 2).
- The da v2 arm is included so the std-vs-v2 policy-head comparison stays
  same-generation under 4-dim.

## Key files

- `agents/ppo_two_players_clean.py`, `agents/ppo_three_players.py` —
  `PPOConfig.state_dim=4`, `state_from_params(..., l_gap=0.0)`
- `run/run_different_ability.py` — per-player rollout states, per-player
  transitions, per-player eval (`effort1`/`effort2` in history and `final`)
- `utils/exploit_asymmetric.py` — `_sample_policy_efforts(..., l_gap)`;
  gaps computed from l1/l2 inside `eval_exploitability_asymmetric`
- `config/one_stage_two_players.py` — exploit M 8192→16384
- `run/r7_state4_wave.sh` — 100-job wave (flock queue, 8 tmux workers)
