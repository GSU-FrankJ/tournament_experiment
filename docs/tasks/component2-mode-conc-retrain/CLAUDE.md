# Component-2 mode-conc retrain

## Goal
Test **Claim A** ("PPO self-play learning itself converges to equilibrium effort")
by giving the 3P PPO agent a mode-concentration policy head + an
exploitability-*triggered* concentration ramp, so that the agent's OWN raw output
(not post-hoc MC-BR polish) reaches ~e*. If the raw policy mean lands near
e*=25 for 3P q35, Claim A becomes defensible; if it still undershoots, the
evidence stays at Claim B (see SESSION_STATE.md "PHASE 0" + "OPEN DECISIONS").

## Why this is needed
Phase-0 6/6 PASS came from MC-BR polish, which is a **global solver** (reaches e*
from any init) — so it cannot attribute convergence to PPO learning. Raw PPO
undershoots: 3P q35 raw 22.99 vs e*=25 (-8%), da q35 raw 43.99 vs 46.43 (-5%).
Component-2 is the ONLY experiment that distinguishes "PPO learned it" (A) from
"MC-BR found it" (B). Authorized by owner 2026-07-01.

## Scope — what to touch
- `agents/ppo_three_players.py`: add `ActorCriticModeConc` head; add PPOConfig
  fields; add init branch selecting it under `mode_conc_ramp`.
- `run/run_three_players.py`: config plumbing + 4 CLI flags + ramp state machine
  in the training loop + mode diagnostic column.

## Scope — what NOT to touch
- Do NOT modify the `theory_align_v2` path — old r5 runs must stay reproducible.
  Component-2 is a NEW code path (§259: "根本不同的 regime，需新代码，不是 flag flip").
- Do NOT touch `paper/generator/config.py` theory params.
- Do NOT change dc/da runners (Component-2 is 3P-only per spec).
- Do NOT modify anything under `results/*/convergence/`.

## Locked decisions
- **Head** (§259): `s=sigmoid(mode_head)`, `κ=clamp(softplus(conc_head)·scale+κ_min, max=κ_max)`,
  `α=1+s·κ`, `β=1+(1−s)·κ`. The `+1` floors force α,β≥1 (interior mode).
- **Ramp**: Explore (κ∈[1,20], entropy-driven) → Trigger on `EXP_raw<0.05` for
  3 consecutive in-loop evals → Ramp κ∈[20,50,100,200], 20 updates/stage →
  hold at κ=200 until normal exploit stop (eps_eq=0.03, patience 5).
- **Stop gating**: the normal exploitability stop is SUPPRESSED until the ramp
  reaches its final stage (κ=200). Rationale (§255): stopping during explore at
  low κ would freeze the wide, undershooting policy we are trying to sharpen.
- **Reporting metric**: report **Beta mean** as the headline converged effort
  (CLAUDE.md hard invariant + cross-experiment consistency; owner decision
  2026-07-01). Store mode as a diagnostic column; at κ=200 mode≈mean.
- **CLI flags**: `--mode-conc-ramp`, `--kappa-schedule 20,50,100,200`,
  `--ramp-trigger-exp 0.05`, `--kappa-stage-hold 20`.
- **Scale**: 3P q35, seeds 42–46. Smoke first (1 seed, ~30 updates) to verify
  ramp fires / κ steps / mode extracts / no crash; then 5 seeds parallel in tmux,
  GPU-pinned, episodes=6,144,000 (full), K=1.

## Hard rules carried over
- tmux for any run >1min; no nohup/bare background.
- No fabricated numbers — every reported value traces to a run JSON.
- Confirm params with owner before the full 5-seed GPU launch.
