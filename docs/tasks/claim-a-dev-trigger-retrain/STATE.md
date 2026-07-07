# Claim-A dev-distance-trigger retrain

Status: complete (Gate A resolved 2026-07-02: owner chose branch (ii) — authorize the
non-locking redesign as a new sub-task, accepting it fights the documented r5 stall.
Successor task: `docs/tasks/claim-a-nonlocking-continuation/`.)
Current phase: phase01 (Phase A — DONE; task closed at Gate A)

## What's done
- Task created 2026-07-02 after the corrected Component-2 diagnosis (dev doc §6.4).
- Design decision: replace the trigger OBSERVABLE (payoff-gain → BR-distance),
  not just its parameters. Rationale + kill conditions in CLAUDE.md.
- **Phase A complete** (`tools/claim_a_phase_a_screen.py`, ZERO GPU). Findings in
  `phase01_findings.md`, raw dump `phase01_screen.json`, log `phase01_run.log`.
  Results:
  - A1: deterministic-mean BR-distance IS a clean monotone trigger signal
    (6.5 near mode≈18 → 0.5 at e*=25); gain crosses 0.05 exactly at c≈18, which
    mechanically explains the old gain-trigger firing point.
  - A2 (decisive): at explore-κ (κ=20, ~9-unit spread) the BR collapses to ~22.5
    and distance is noisy 0.26–3.85, dipping <1.0 far from e*. A distance trigger
    vs the STOCHASTIC policy fires early too — must be defined vs the deterministic
    MEAN profile to be usable.
  - A3 (deeper): raising κ freezes the climb (velocity dies/reverses at κ=100–200,
    2/4 seeds stalled); r5 corroborates — raw PPO stalls at 22.99 on the full 6M
    budget with no κ lock. The 2-unit undershoot is a PPO-dynamics property, not a
    trigger/schedule one.
- **Gate A recommendation: lean STOP / adopt Claim B.** The trigger is fixable but
  A3 + r5 show a Component-2-style GPU retrain would most likely reproduce the stall.

## What's next (owner decision at Gate A)
- (i) STOP → finalize Claim B (default recommendation), OR
- (ii) authorize a NON-LOCKING redesign (keep κ low, distance-vs-mean as stop signal,
  climb under sustained entropy) as a new sub-task — must still fight the r5 stall.

## Blockers
- Gate A is an owner decision point (see phase01_findings.md "Gate A recommendation").
