# Two-stage closed-form benchmark: independent audit (2026-07-09)

Audit of the derivation added to `docs/Experiments Plan_Multi-stage.md`
(main commit `8e9433e`) against independent re-derivation and numerical
integration. Verification code: `tools/verify_two_stage_benchmark.py`
(all checks PASS). Implementation: `utils/theory_multistage.py` (repo
convention `c(e) = k e^2`).

## Verdict

The derivation is substantially correct — g2(d), the stage-2 SOC/q_SOC,
g1, and the participation algebra all check out numerically. It contains
**one genuine math error (stage-1 SOC), one factor-2 bookkeeping conflict,
one leftover contradiction, and one gap (zero-effort deviation)**. None
change the headline numbers (g1 = 46.67, g2(0) = 70 at q = 50), but the
SOC error changes the validity region logic, and the validity region is
tighter than previously assumed: **q_crit = 41.83, so q = 35 and q = 40
are invalid**.

## Confirmed correct (numerically, to quadrature precision)

| Claim (plan doc, c=(k/2)e² convention) | Repo convention (c=ke²) | Check |
|---|---|---|
| g2(d) = (ΔW/k) f_ξ(d), even in d | g2(d) = ΔW f_ξ(d)/(2k) | global argmax on grid ✓ |
| g2(0) = ΔW/(2kq) | ΔW/(4kq) = 70 at q=50 | ✓ |
| q_SOC = √(ΔW/(4k)) | √(ΔW/(8k)) = 41.83 | tight: deviations appear just below ✓ |
| g1 = ΔW/(3kq) | ΔW/(6kq) = 46.67 at q=50 | global argmax on grid ✓ |
| E[g2(d₂)] on-path = g1 exactly | same | ✓ |
| E[cost] terms, 17/144 (→17/288) | U_eq = (W_H+W_L)/2 − 17ΔW²/(288kq²) | matches numerics to 4+ digits ✓ |
| q_B2, q_B1 effort-bound thresholds | 35.0 / 23.3 | ✓ |
| q_PC algebra | 28.75 at Ū=0 | ✓ (never binds here) |

## Finding 1 (math error): stage-1 SOC is NOT unconditional

The doc claims `E[V₂*''(ξ₁)] = −ΔW²/(16kq⁴)`, making the stage-1
curvature `−k − ΔW²/(16kq⁴) < 0` unconditionally. This drops a Dirac
term: V₂*(d) has a **convex kink at d = 0** (the −c(g2(d))² cost term is
a tent-shaped function of |d|), contributing `+ΔW²/(4kq³)·δ₀` to V₂*''.
The correct value is

    E[V₂*''(ξ₁)] = +ΔW²/(16kq⁴)   ( (k/2)e² convention )

so the stage-1 curvature is `−k + ΔW²/(16kq⁴)` — negative **iff q > q_SOC**.
The doc's conclusion ("stage-1 SOC holds whenever the final-stage region
is valid") survives, but only because the corrected condition coincides
exactly with q_SOC, not unconditionally.

Numerical confirmation (repo convention; `−2k ± ΔW²/(32kq⁴)`):

| q | numeric U₁''(g1) | corrected formula | doc formula | stage-1 global argmax |
|---|---|---|---|---|
| 38 | **+2.64e-4** | +2.68e-4 | −1.41e-3 | 0.0 (give-up), gain +0.229 |
| 40 | **+1.09e-4** | +1.12e-4 | −1.26e-3 | 8.9, gain +0.061 |
| 42 | −1.12e-5 | −9.03e-6 | −1.13e-3 | 55.6 = g1 ✓ |
| 50 | −2.92e-4 | −2.91e-4 | −8.51e-4 | 46.7 = g1 ✓ |

At q = 40 the symmetric candidate is a **local minimum** — the doc's
formula would have certified it as valid.

## Finding 2 (bookkeeping): ΔW = 2 example vs ΔW = 4 parameter table

The derivation's numeric example uses ΔW = 2 with c = (k/2)e²; the
parameter table says prize spread = 6 − 2 = 4; the model section (§1.4)
defines c = ke². The example lands on the same 46.7 only because the two
convention swaps cancel in the FOC. They do NOT cancel in:
- the EXP/ΔW normalized-exploitability threshold (2× difference),
- absolute payoff levels / participation numbers.

Resolution (canonical, in `config/multi_stage_two_players.py`):
**w_h = 6, w_l = 2 (ΔW = 4), c(e) = k e², k = 1/3500** — matching the repo
invariant and the parameter table. All plan formulas convert via k → 2k.

## Finding 3 (leftover): Experiment 1 still states a constant stage-2 effort

The Experiment-1 block still reads "symmetric expected equilibrium effort
in stage 2: e* = ΔW/(6qk)" — pre-derivation text. The stage-2 benchmark is
the FUNCTION g2(d); 46.7 is only its on-path expectation (which happens to
equal g1 exactly). The recovery test must compare ê₂(d) against g2(d)
over the state grid, not against a constant.

## Finding 4 (gap): participation ≠ zero-effort deviation

The doc's participation constraint compares U_eq to an outside option Ū.
The interior-validity question ("can a player gain by deviating to zero
effort while staying in the game?") is a different check, and local SOC
alone does not answer it. `validate_two_stage_params` therefore runs a
numerical global-deviation scan over the full effort grid (which covers
e = 0) for both stages. Result: unprofitable for all q > q_crit
(e.g. −0.295 payoff at q = 50), profitable below (+0.053 at q = 40).

## Consequences for the experiment grid

    q_crit = max(q_SOC=41.83, q_B2=35.0, q_B1=23.3, q_PC=28.7) = 41.83

- Valid: q ∈ {45, 50, 55} (canonical q_list).
- Invalid: q = 35 (one-stage habit) and q = 40 — SOC fails, closed form
  meaningless there. Usable only as separately-reported boundary cases.
- Every runner must call `config.multi_stage_two_players.validate()`
  before training (raises on violation).

## Errata to fold back into the Word source of the plan

1. §"Second-order condition for g1": replace `E[V₂*''] = −ΔW²/(16kq⁴)`
   with `+ΔW²/(16kq⁴)`; conclusion becomes "stage-1 SOC holds iff
   q > q_SOC" (same region, different logic).
2. Make the numeric example use ΔW = 4 with the k→2k conversion, or state
   W_H = 3, W_L = 1 explicitly; reconcile with the parameter table.
3. Rewrite the Experiment-1 closed-form block in terms of g1 and g2(d);
   delete the constant stage-2 line (keep E[g2] = g1 as a remark).
4. Add the zero-effort/global-deviation check to the validity-region
   section (analytic screens + numerical scan).
5. Resolve the mean/mode highlights: repo invariant is Beta **mean**.
