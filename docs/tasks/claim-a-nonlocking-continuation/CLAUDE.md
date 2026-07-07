# Claim-A non-locking κ-continuation retrain

## Goal
Third (and per pre-registration, final without new evidence) attempt at **Claim A**
for 3P q35: make raw PPO output land near e*=25 by replacing the "explore → lock"
architecture with **velocity-gated κ continuation** — the policy follows the moving
equilibrium of the exploration-smoothed game as κ rises, advancing κ only when it has
re-converged at the current κ. Authorized by owner 2026-07-02 (Gate A of
`docs/tasks/claim-a-dev-trigger-retrain/`, branch (ii)), explicitly accepting that it
fights the documented r5 stall at 22.99.

## Why continuation, not "stay wide" and not "lock harder"
Established by the predecessor task's Phase-A screen (zero GPU, all numbers in
`docs/tasks/claim-a-dev-trigger-retrain/phase01_findings.md`):
- With policy spread (explore κ), the best response against the SAMPLED opponent
  population is ~22.5, not 25 (A2) — so "keep κ low forever" converges to the smoothed
  game's equilibrium below e*, by construction. Literal non-locking is insufficient.
- Clock-driven κ ramps freeze the climb: mode velocity dies/reverses at κ=100–200 for
  2/4 seeds (A3); r5 corroborates (stall at 22.99 with full budget). Locking on a fixed
  schedule is refuted (Component-2 negative result).
- But 2/4 Component-2 seeds (43, 44) DID keep velocity +0.06..+0.15/update at every κ
  stage up to 200 — existence proof that tracking under rising κ is possible.
- Therefore: walk κ up a ladder, **gate each step on kinematic convergence** (policy
  stopped moving at current κ), never on a clock and never on a payoff-gain threshold
  (Component-2's trigger fired ~7 units early — gain has no signal on the plateau).

## Architecture (design principles; measured params frozen in phase01_findings.md)
- **Head**: existing `ActorCriticModeConc` (unchanged).
- **κ ladder**: κ ∈ {20, 35, 60, 100, 200, ...} — top κ chosen from the measured
  smoothed-equilibrium curve μ*(κ) such that μ*(κ_top) ≥ 24.75 (1% of e*).
- **Stage advance = kinematic convergence gate**: advance to the next κ when the
  policy mean has stopped moving at the current κ (drift-window detector, reusing the
  existing cheap-gate/drift machinery), with a minimum per-stage hold. NO payoff-gain
  trigger, NO fixed stage clock, NO analytic e* anywhere in-loop.
- **Optimizer floors**: lr and entropy_coef floors during the ladder (velocity death
  autopsy decides values) so the schedule cannot starve tracking.
- **Acceptance (end of ladder)**: existing exploitability stop + deterministic-mean
  BR-distance as a diagnostic; headline metric = Beta mean (CLAUDE.md invariant).

## Kill conditions (pre-registered, honour them)
- **Stall kill (per run)**: at any stage, if |Δmean| over the last 60 updates < 0.5
  while the remaining ladder implies ≥1 unit still to climb, and the stage has not
  converged within a max stage budget (~200 updates) → declare STALL, stop the run,
  count it as a negative outcome. Do not extend budgets to rescue it.
- **Gate C (carried over verbatim from the predecessor task)**: after the 5-seed q35
  run, triggered/completed seeds with cross-seed std > 1.0 or mean|err| > 4% ⇒ Claim A
  is dead in this parameterization; no further retrain iterations without new
  authorization. Success target: mean|err| ≤ 2% (i.e. raw mean ≥ 24.5) and std ≤ 0.5.
- **Design-analysis kill (phase01)**: if the measured μ*(κ) curve does not reach
  ≥ 24.75 at any implementable κ, or the autopsy shows velocity death with HEALTHY
  approx_kl (physics, not schedule — floors cannot fix it), recommend STOP before GPU.

## Scope — what to touch
- Phase01 (analysis): `tools/claim_a_continuation_design.py` + findings here. ZERO GPU.
- Phase02 (only if phase01 supports): additive continuation mode in
  `run/run_three_players.py` behind a new flag (e.g. `--kappa-continuation`), default
  off; reuse ModeConc head, drift gate, κ pinning.
- Phase03: 5-seed GPU run — **confirm params with owner before launch** (house rule).

## Scope — what NOT to touch
- `results/*/convergence/` read-only. r5 / theory_align_v2 / Component-2 gain-trigger
  paths stay byte-identical and reproducible.
- No analytic e* in any trigger/gate/reward/update (sampled-only invariant).
- `paper/generator/config.py` theory params untouched.

## Framing note (paper value either way)
If the run succeeds → Claim A in a stronger form: "PPO tracks the equilibrium of the
exploration-smoothed game; κ-continuation walks it to the deterministic equilibrium."
If it stall-kills → the μ*(κ) curve + stall evidence upgrade the Claim-B narrative:
raw PPO lands at the smoothed equilibrium (measured curve), and MC-BR + exploitability
bridge the final gap. Both outcomes are publishable; neither is wasted GPU.

## Hard rules carried over
- tmux for any run >1 min; no nohup/bare background.
- Every reported number traces to a JSON or logged script run.
- Split commits: analysis code vs results vs docs.
