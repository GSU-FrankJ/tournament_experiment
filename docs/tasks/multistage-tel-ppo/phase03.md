# Phase 03: independent DP best-response verifier (COMPLETE 2026-07-09)

## Result

`utils/dp_verifier.py` + `tools/calibrate_verifier.py` (all checks PASS).
Calibration table (T=2, q=50, DW=4):

| policy | EXP | EXP/DW | dReach | dFull | onpathΔ | cert? |
|---|---|---|---|---|---|---|
| closed_form_CF | 0.0001 | 0.0000 | 0.0001 | 0.6222 | 0.0001 | **yes** |
| bad:const_low(5) | 0.810 | 0.203 | 1.327 | 1.368 | 0.957 | no |
| bad:const_high(100) | 3.714 | 0.929 | 3.945 | 5.714 | 2.344 | no |
| bad:one_stage_repeated | 0.889 | 0.222 | 1.576 | 2.800 | 0.500 | no |
| bad:no_gap_stage2 | 0.367 | 0.092 | 1.400 | 2.022 | 0.325 | no |
| bad:random_mean(50) | 0.121 | 0.030 | 0.718 | 1.429 | 0.108 | no |

Grid refinement (closed form): EXP 0.00024 -> 0.00014 -> 0.00012 across
M=51/101/201, Richardson 0.00012. T=3 machinery smoke-tested (3 stages,
bound holds, Δ_3~0 since the final stage is myopic).

## Certificate design decision (resolves blind-spot H6)

Three Δ aggregates are reported; the PRIMARY certificate is **dReach**
(Σ_t max over the BR-reachable support of Δ_t), NOT the full-grid
worst-case. Reasoning:
- Root EXP alone can over-certify because the grid DP best response is an
  approximation (underestimates BR).
- Full-grid worst-case Δ over-states for an on-path-only benchmark: at
  stage 1 only d=0 is reachable (d_1 ≡ 0), yet the full-grid max evaluates
  the closed form at d != 0 where it was never defined -> spurious 0.62.
- dReach restricts each stage's max to states reachable from d_1=0 under
  (deviator best-responds, opponent plays ê), which is exactly the PDL
  reachable set. It upper-bounds EXP (verified: dReach >= EXP for all
  policies) AND excludes the unreachable stage-1 states -> 0.0001 for the
  closed form. This is the honest conservative certificate.

`random_mean(50)` sits at EXP/DW=0.03 (raw threshold) but dReach/DW=0.18
rejects it — concrete evidence the conservative certificate is the right
gate, per the owner decision.

## Owner-mandated numerics: all implemented

- Closed-form terminal integration E_ξ[R(y+ξ)]=w_l+ΔW F_ξ(y) (no interp).
- Deterministic triangular quadrature for t<T continuations via the 1-D
  smoothed value W(y) (avoids the M*K*quad tensor).
- Parabolic polish on the BR effort argmax (mitigates ε_grid
  under-estimation of BR / over-certification).
- Grid refinement + h² Richardson.

## Verifier independence

`dp_verifier.py` imports only F_ξ/f_ξ from `theory_multistage`; it never
imports `envs/multi_stage_env` or the PPO agent. Its verdict is external.

## Original objective (below, retained)


## Objective

`utils/dp_verifier.py`: backward-induction best-response verifier for the
multi-stage game, plus its calibration on the closed form and a
falsification suite. This is the INDEPENDENT certifier — it must not import
the training env or the PPO agent; it reads only a policy callable
ê_t(d) and the game parameters.

## Math (plan section 4, owner decisions)

- Best-response value (opponent fixed at ê):
  V_t^BR(d) = max_e { -c(e) + E_ξ[V_{t+1}^BR(d + e - ê_t(-d) + ξ)] },
  V_{T+1}(d) = R(d).
- Learned-policy value (both play ê):
  V_t^ê(d) = -c(ê_t(d)) + E_ξ[V_{t+1}^ê(d + ê_t(d) - ê_t(-d) + ξ)].
- Exploitability: EXP = V_1^BR(0) - V_1^ê(0).
- One-step deviation gap (learned continuation):
  Δ_t(d) = max_e Q_t^ê(d,e) - Q_t^ê(d, ê_t(d)),
  Q_t^ê(d,e) = -c(e) + E_ξ[V_{t+1}^ê(d + e - ê_t(-d) + ξ)].

## Owner-mandated numerics

1. **Closed-form terminal integration.** E_ξ[R(y+ξ)] = w_l + ΔW·F_ξ(y)
   exactly (F_ξ from `utils.theory_multistage`). NEVER interpolate the
   step reward R near y=0 — that would put O(h) error at the most
   important state.
2. **Δ_t(d) is the PRIMARY certificate.** By the performance-difference
   lemma, EXP ≤ Σ_t max_d Δ_t(d) (a true upper bound on exploitability).
   Report this sum as the certificate; report root-state EXP alongside.
   Also report the on-path E_{d~ê}[Δ_t(d)].
3. **Deterministic quadrature** over the triangular ξ for t<T
   continuations; MC only as a fallback with CIs.
4. **Grid refinement + Richardson.** Run M ∈ {51,101,201}; report the EXP
   sequence and an h²-Richardson extrapolate. Certification honesty:
   watch that refinement does not REVEAL exploitability the coarse grid
   hid (coarse effort grid underestimates BR → over-certifies), so the
   effort search is fine + parabolic-polished.

## Calibration + falsification (plan 4.5, Experiments 3-4)

- EXP(e*_CF) ≈ 0 → establishes the numerical error floor.
- Bad policies (constant-low, constant-high, one-stage-repeated,
  no-gap-stage-2) → EXP ≫ error floor. Confirms discriminatory power
  BEFORE the verifier is trusted on T≥3.

## Deliverables

| Artifact | Status |
|---|---|
| `utils/dp_verifier.py` | done |
| `tools/calibrate_verifier.py` (calibration + falsification table) | done |

## Acceptance

- [x] EXP(e*_CF) within the error floor (target ≤ ~1e-2 in payoff units,
      ≪ ΔW).
- [x] Every bad policy scores EXP ≫ EXP(e*_CF).
- [x] Δ-sum upper bound ≥ EXP on every policy (sanity of the bound).
- [x] EXP stable under grid refinement (Richardson consistent).
