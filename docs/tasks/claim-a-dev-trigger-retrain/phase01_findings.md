# Phase A findings — zero-GPU feasibility screen

Run 2026-07-02, vector2. Script: `tools/claim_a_phase_a_screen.py`; raw dump
`phase01_screen.json`; console log `phase01_run.log`. ZERO GPU, ZERO training.
3P q35 (k=0.001, w_H=6.5, w_L=3.0, q=35, n=3, bounds [0,100], e*=25).

## TL;DR — Gate A: **lean STOP** (do not launch the Component-2-style GPU retrain)

The trigger *can* be fixed, but the screen surfaced a deeper structural problem the
trigger swap does not solve. Recommend adopting Claim B unless owner explicitly wants
to spend GPU on the one remaining low-probability lever (a *non-locking* redesign,
which is materially different from Component-2 and needs its own authorization).

## A1 — Deterministic BR-distance IS a clean trigger signal (and explains the old failure)

Sweeping a symmetric deterministic center c (opponents fixed at c), payoff-gain and
best-response distance `|BR(c)−c|`:

| c | gain | BR | \|BR−c\| |
|---|---|---|---|
| 18 | 0.0497 | 24.50 | 6.50 |
| 20 | 0.0284 | 25.50 | 5.50 |
| 21 | 0.0181 | 25.00 | 4.00 |
| 22 | 0.0109 | 24.75 | 2.75 |
| 23 | 0.0061 | 25.50 | 2.50 |
| 25 | 0.0010 | 25.50 | 0.50 |

Two things:
1. **gain crosses 0.05 right at c≈18** — this is the mechanical reason the
   Component-2 gain trigger (`EXP_raw<0.05`) fired at mode≈18. Confirmed by construction.
2. **BR-distance is a strong, monotone signal**: 6.5 near mode≈18 → 0.5 at e*=25.
   Against deterministic opponents, distance HAS the discrimination in [18,25] that
   gain lacks. ✓ the trigger observable is fixable **in principle**.

## A2 — …but only in the high-κ regime; at explore-κ the signal collapses

The trigger must fire during **explore (κ=20)**, where opponents are NOT deterministic
— the ModeConc Beta at κ=20 has ~9 effort-units of spread. Reconstructing the BR against
opponents *sampled from the actual explore-κ policy* (the quantity the in-loop
`eval_exploitability_3p` really sees):

- BR-to-stochastic-opponents ≈ **22.5**, not the deterministic ~25 (a wide, low-centered
  opponent population doesn't require pushing to 25 to beat).
- Distance `|BR − policy_mean|` at the **historical gain-trigger points**: 1.69, 2.82,
  1.47, 3.07 (mean **2.27**).
- Worse, at non-trigger checkpoints (mode≈18–20, far from e*) the stochastic distance
  frequently **dips below 1.0**: 0.41 (s42), 0.26/0.90 (s45/s43), 0.89 (s44), 0.70 (s46).

**Consequence**: a distance trigger with τ_dist≈1.0 defined against the *stochastic
policy* would fire early too — the signal is noisy and collapses far from e*, the SAME
early-firing failure as gain. The **actionable fix**: define the trigger BR against the
**deterministic mean profile** `[mean,mean,mean]` (i.e. the A1 signal), not the
stochastic policy. That recovers distance ≈ 4–5 at the trajectory mean (~21) vs ≈ 0.5
at e*=25 — clean and monotone.

## A3 — the deeper problem: raising κ freezes the climb (trigger-independent)

Per-κ-stage mode velocity across the real c2 ramp segments (units/update):

| seed | κ=20 | κ=50 | κ=100 | κ=200 | resid@trigger |
|---|---|---|---|---|---|
| 42 | +0.229 | +0.048 | +0.006 | −0.002 | 7.11 |
| 43 | +0.068 | +0.065 | +0.080 | +0.060 | 7.03 |
| 44 | −0.066 | +0.148 | +0.127 | +0.076 | 6.21 |
| 45 | +0.079 | +0.083 | +0.011 | −0.103 | 6.76 |

- Velocity **dies (or reverses) as κ rises** — at κ=100–200 the mean is essentially
  frozen (s42 −0.002, s45 −0.103). High concentration = low entropy = no exploration
  left to move the mean. The ~7-unit residual never closes: 2/4 seeds barely moved,
  s45 reversed.
- Even a *perfect* trigger cannot beat this: the mechanism that Component-2 uses to
  "lock in" determinism (raise κ) is the same mechanism that removes the exploration
  needed to reach e*.
- **Independent corroboration from r5**: with the FULL 6M-step budget and NO κ lock
  (sustained exploration), raw PPO still stalls at mean **22.99** (8.03% under e*=25).
  The 2-unit undershoot is a property of the PPO *learning dynamics*, not of the
  trigger or the concentration schedule.

## Gate A recommendation

- The naive plan ("swap gain→distance trigger, lengthen the window") is **not
  sufficient**: A2 shows the distance signal must be redefined against the deterministic
  mean to be usable, and A3 shows the κ-ramp architecture self-defeats regardless of
  trigger quality — corroborated by the r5 stall at 23.
- **The one untested lever**: a *non-locking* redesign — keep κ low/exploratory (do NOT
  ramp to 200), use distance-against-the-mean purely as an acceptance/stop signal, and
  let the policy climb under sustained entropy. This is the only path A1–A3 leave open,
  but it must still overcome the r5 dynamics stall at ~23, which the same data says is
  unlikely. It is also a materially different architecture from Component-2 (it drops
  the concentration lock), so it needs its own owner authorization + design.
- **Default recommendation: STOP and adopt Claim B.** The zero-GPU screen has done its
  job — it indicates a ~2-day / 5-GPU Component-2-style retrain would most likely
  reproduce the stall, and it isolated the reason (PPO dynamics + κ-lock, not the
  trigger). Claim B (PPO reaches the basin; MC-BR + exploitability prove the equilibrium)
  remains the defensible central claim.

Owner decision required: (i) STOP → finalize Claim B; or (ii) authorize the non-locking
redesign as a new sub-task, accepting it fights the documented r5 stall.
