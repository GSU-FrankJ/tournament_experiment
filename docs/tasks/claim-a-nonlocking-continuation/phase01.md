# Phase 01: continuation design analysis (zero GPU)

## Objective
Measure the two quantities the continuation design depends on, then freeze the design
or trigger the design-analysis kill condition (CLAUDE.md).

## Analyses
1. **Smoothed-equilibrium curve μ*(κ)** — for κ ∈ {20, 35, 60, 100, 200, 400} and a
   grid of policy means μ, compute the sampled best response b(μ, κ) of a deterministic
   candidate against 2 opponents drawn from the ModeConc Beta family
   (α+β=κ+2, mean=μ), CRN; the fixed point b(μ*)=μ* is the equilibrium of the
   κ-smoothed game. Also the deterministic (κ=∞) crossing via
   `exploitability_frozen_profile`. Outputs: μ*(κ) table; smallest κ_top with
   μ*(κ_top) ≥ 24.75; per-stage climb distances for the ladder.
   Validation targets: μ*(∞) ≈ 25 (Finding B); μ*(20) should rationalize the c2
   explore hover (~21) and/or A2's BR≈22.5.
2. **Velocity-death autopsy (c2 seeds 42–45)** — per κ-stage segment: mode velocity,
   mean approx_kl, mean batch_entropy. Discriminates:
   - approx_kl collapsed where velocity died → optimizer/schedule starvation →
     fixable with lr/entropy floors (design proceeds);
   - approx_kl healthy but velocity ~0 → gradient-SNR physics → floors won't help
     (recommend STOP before GPU).
3. **Ladder budget estimate** — combine 1+2: per-stage climb distance / healthy
   velocity → minimum stage budgets, total update budget, stall-kill thresholds.

## Files
- Add: `tools/claim_a_continuation_design.py` (analysis-only, sampled-only).
- Write: `phase01_findings.md`, `phase01_design.json`, `phase01_run.log` (this folder).
- Read-only: c2 + r5 convergence JSONs.

## Verification
Script exits 0; every number in findings traces to the JSON dump; kill conditions
applied as pre-registered.
