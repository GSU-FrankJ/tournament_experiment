# Phase 02: adaptive-batch κ-continuation — implementation + smoke + 1-seed pilot

## Objective
Implement the owner-approved (kill overruled 2026-07-02) adaptive-batch continuation
mode, smoke it, and run a 1-seed GPU pilot whose ONLY job is to measure whether the
large-batch ladder actually shrinks the diffusion band as 1/√B predicts.

## Design (frozen)
- **Head**: existing `ActorCriticModeConc` (unchanged).
- **Phases**: explore (κ∈[1,20] free, batch 4096) → ladder (κ pinned per stage) → done
  (normal exploit-stop takes over; stop suppressed until done).
- **Ladder**: κ = [20, 35, 60, 100, 200, 400]; per-stage batch =
  [16384, 16384, 16384, 65536, 65536, 65536] (early stages need no tight band —
  later stages fix earlier landing errors; the band only matters at the top).
- **Stage advance = kinematic convergence gate** (no payoff-gain trigger, no clock,
  no analytic e*): advance when |mean(mode over last W) − mean(mode over prior W)|
  < τ_conv with W=30, τ_conv=0.3, after a min hold of 40 updates; forced advance at
  250 updates (logged `forced_advance`, counts against the pilot).
- **Optimizer floors during ladder+done**: lr ≥ 1e-4, entropy_coef ≥ 0.003 (D2 showed
  KL healthy at c2's late-run values; floors only guard the late end).
- **Budget**: episodes ≈ explore (~0.2–0.9M) + ladder (~3×70×16384 + 3×70×65536
  ≈ 17M) + done tail ≈ **20M episodes** for the pilot (~3× c2; ~1–1.5 days wall,
  1 GPU).

## Pilot gate (pre-registered)
- **PASS**: diffusion band (std of mode over the converged window) at κ≥100 stages
  visibly < c2's band, and at κ=400 ≤ ~0.5; no stall; budget within 20M.
- **KILL**: band at κ≥100 still ≥ ~1.0 (1/√B scaling refuted in vivo), or
  forced_advance on ≥2 stages, or budget blowout. → report negative, stop.
- 5-seed launch only after pilot PASS + owner confirmation.

## Files to modify
- `agents/ppo_three_players.py`: PPOConfig field `kappa_continuation`; head-selection
  branch extended (`mode_conc_ramp OR kappa_continuation` → ModeConc head).
- `run/run_three_players.py`: config plumbing + CLI flags (`--kappa-continuation`,
  `--continuation-ladder`, `--continuation-batch`, gate/floor params) + continuation
  state machine + adaptive batch switch + stop gating + `cont_phase` diagnostic column.
- NOT touched: theory_align_v2 / r5 / c2 gain-trigger paths; results/ read-only.

## Verification
- CPU smoke (loose gates, tiny batches): explore→ladder stages→done transitions,
  batch switch visible, JSON well-formed, exit 0.
- GPU smoke: same on GPU.
- Pilot: tmux, GPU 0, seed 42, tag `c3_cont`, full params.
