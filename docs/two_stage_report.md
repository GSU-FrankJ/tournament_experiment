# Two-Stage TEL-PPO — Corrected Certification Report

**Date:** 2026-07-12 · **Branch:** `feat/multistage-phase0` · **Code commit at run time:** `e7d278a`
**Machine-readable companion:** [`results/two_stage_results.json`](two_stage_results.json)

This report supersedes the certification numbers in the 2026-07-10 status report
for the two-stage (T=2) horizon. It presents the results of the correctness pass
(Phase 0 diagnosis → Phase 1 fixes → Phase 2 runs) with the corrected
measurement, certification, and reporting. **Every number here comes from an
actual run in this session** (retrain tag `p2cert`; re-certification tool output).

---

## 0. Executive summary

The two-stage pipeline had reporting/measurement defects (not a bad policy). After
fixing them and adopting a deterministic refinement-UCB certificate, we retrained
the five T=2 seeds and re-certified T=3/4/5:

- **T=2 retrain (tag `p2cert`, 5 seeds): 5/5 certify** under the deterministic
  `dReach_UCB/ΔW ≤ 0.03` gate. `dReach_UCB/ΔW` mean **0.0064** (max 0.0072).
- **On-path stage-2 recovery, now actually measured** (defect D1): `E[ê_2(d_2)]`
  mean **44.69** vs target **46.667** (~4.2% low). Stage-1 `ê_1(0)` mean **46.38**
  vs 46.667 (**0.6%** low). The stage-2 *peak* `ê_2(0)` mean **63.51** vs **70.0**
  (~9% low) is the expected μ\*(κ) exploration smoothing — and is correctly
  compared to 70, never to 46.67.
- **Mean vs mode (D6): resolved.** At convergence mode ≈ mean (E[ê_2] differs by
  **+0.07** on average, ~0.16%). Mode is **not** closer to the closed form; the
  Beta **mean** stays the primary reported extraction (repo invariant).
- **T=3/4/5 re-certification** (deterministic extracted-policy, from saved effort
  curves): **5/5, 5/5, 4/5** — verdicts **unchanged** by the stricter UCB gate.
- **Certificate is validated** (§4): it separates the learned/closed-form policies
  (dReach_UCB/ΔW ≈ 0.006 / 0.000) from a bad-policy suite (0.33–0.99) by 50–180×,
  and the benchmark is valid at q=50 > q_soc = 41.83 (§5).

---

## 1. What changed (methodology)

**Certificate (D3 → deterministic UCB).** Certification uses a fully deterministic
verifier (closed-form terminal `w_l + ΔW·F_ξ(y)`; triangular quadrature) — **no
Monte Carlo, no `1.96`, no standard errors**. The gate is a nested-grid
discretization bound:

```
dReach_UCB = dReach_fine + |dReach_fine − dReach_coarse|,   certify iff dReach_UCB/ΔW ≤ 0.03
```

with score-gap grids **coarse = 101, fine = 201**. This is a Richardson-style
`|fine−coarse|` numerical-error margin, not a confidence interval.

**Selection independence (D5/T7).** Best-checkpoint selection uses grid **51**,
**disjoint** from the two certification grids (101/201), so selection cannot peek
at the certifier — not even through the UCB's coarse term.

**On-path recovery (D1/T1).** `E[ê_2(d_2)] = ∫ ê_2(δ) f_ξ(δ) dδ` is computed by
Gauss–Legendre quadrature over the triangular shock (exact target 46.667), and
reported alongside `ê_1(0)|46.667`, `ê_2(0)|70.0`.

**Persistence (T7/T8).** Each run now persists the full policy checkpoint (`.pt`,
gitignored), the raw Beta `(α,β)` on the curve grid, provenance (commit, grids,
quadrature), and both mean- and mode-extraction diagnostics.

**Targets (exact).** `e*_{1,CF}(0) = ΔW/(6qk) = 46.667`; `E[e*_{2,CF}(d_2)] = 46.667`;
`e*_{2,CF}(0) = ΔW/(4qk) = 70.0`. Parameters: w_H=6, w_L=2 (ΔW=4), k=1/3500, q=50,
e∈[0,100]. Validity: q > q_crit = 41.833 (SOC), so q=50 is valid.

---

## 2. T=2 retrain results (tag `p2cert`, seeds 42–46)

Protocol: q=50, T=2, 2000 updates × 512 episodes, entropy 0.005, γ=λ=1, CPU
(~14–15 min/seed). Reported policy = Beta **mean**.

### 2.1 Certification (deterministic dReach-UCB gate)

| seed | EXP/ΔW | dReach_fine/ΔW | dReach_UCB/ΔW | certified | ckpt |
|------|--------|----------------|---------------|-----------|------|
| 42 | 0.0020 | 0.0056 | 0.0056 | **yes** | u1900 |
| 43 | 0.0012 | 0.0065 | 0.0065 | **yes** | u2000 |
| 44 | 0.0018 | 0.0072 | 0.0072 | **yes** | u1700 |
| 45 | 0.0017 | 0.0071 | 0.0071 | **yes** | u1500 |
| 46 | 0.0015 | 0.0054 | 0.0054 | **yes** | u1100 |
| **mean** | **0.0016** | **0.0064** | **0.0064** | **5/5** | — |

`dReach_UCB ≈ dReach_fine` at every seed (the `|fine−coarse|` margin is < 5e-5): the
T=2 certificate is grid-stable. All five clear 0.03 by ~4×.

### 2.2 Recovery vs closed form

| seed | ê_1(0) → 46.667 | ê_2(0) → 70.0 | E[ê_2(d_2)] → 46.667 | RE_1 | RPE_2^core |
|------|------|------|------|------|------|
| 42 | 46.75 | 63.0 | 45.59 | 0.002 | 0.074 |
| 43 | 47.57 | 61.5 | 43.11 | 0.019 | 0.097 |
| 44 | 44.65 | 67.2 | 47.78 | 0.042 | 0.070 |
| 45 | 45.35 | 62.1 | 43.57 | 0.028 | 0.064 |
| 46 | 47.55 | 63.7 | 43.40 | 0.019 | 0.069 |
| **mean** | **46.38** | **63.51** | **44.69** | **0.022** | **0.075** |

- **Stage 1 recovers** to 0.6% (RE_1 mean 2.2%, well under the 10% target and
  tighter than the old gated runs' 4.5%).
- **On-path stage-2 recovers** to ~4.2% (E[ê_2]=44.69 vs 46.667) — the quantity
  defect D1 had hidden.
- **Peak stage-2 undershoots** (63.51 vs 70.0, ~9%): the μ\*(κ) exploration
  smoothing. This is the value that must be read against 70, and it is.

### 2.3 Mean vs mode (D6 diagnostic)

At convergence the Beta is sharp and unimodal, so mode ≈ mean:
`ê_1(0)` and `ê_2(0)` agree to ≤ 0.2, and `E[ê_2]` differs by **+0.07** on average
(mean minus mode; ~0.16%). **Conclusion: mode is not closer to the closed form;
keep the Beta mean as the standard extraction** (consistent with the repo
invariant). Per-seed α,β are saved for post-hoc revisiting.

### 2.4 Stage-2 effort function ê_2(d) (seed 42, representative)

| d | −100 | −75 | −50 | −25 | 0 | +25 | +50 | +75 | +100 |
|---|------|-----|-----|-----|---|-----|-----|-----|------|
| **ê_2(d)** | 7.1 | 13.4 | 28.0 | 52.5 | **63.0** | 57.1 | 37.4 | 14.1 | 9.0 |
| e*_{2,CF}(d) | 0.0 | 17.5 | 35.0 | 52.5 | **70.0** | 52.5 | 35.0 | 17.5 | 0.0 |

The learned curve keeps the hump shape and tracks the closed form closely near the
core (`|d| ≤ 25`), with the μ\*(κ) smoothing rounding the peak (63 vs 70) and the
corners (tails ≈ 7–9 vs 0). A mild finite-sample **asymmetry** remains at the
shoulders (e.g. d=−50 → 28 vs d=+50 → 37); it partly cancels under the symmetric
on-path density, which is why `E[ê_2(d_2)]` still lands at 45.6 for this seed. The
asymmetry is a training artifact (the final-stage equilibrium is even), not a
certificate concern — the reachable-support certificate is unaffected.

### 2.5 Effect of the correction (before → after)

The independent grid-51 selection (D5 fix) plus a fresh run notably improves the
stage-1 recovery over the old gated T=2 runs:

| run | ê_1(0) mean | RE_1 mean |
|-----|-------------|-----------|
| old `gateT2` (grid-201 selection = certifier) | 48.40 | 0.045 |
| new `p2cert` (grid-51 selection, independent) | **46.38** | **0.022** |

The old runs overshot ê_1(0) by ~3.7%; the corrected runs land within 0.6% and
halve RE_1. (Certification verdict was 5/5 in both; the correction improves the
recovery estimate and removes the selection/certifier coupling.)

---

## 3. T=3/4/5 re-certification (deterministic, extracted-policy)

Applied the corrected `dReach_UCB` gate to the existing T=3/4/5 gated runs
**without retraining**, by rebuilding ê_t(d) from the saved (mean-extraction)
`effort_curves` and re-running the verifier. Tool:
[`tools/recertify_multistage.py`](../tools/recertify_multistage.py) → output in
[`results/multi_stage/recertification_T345.json`](multi_stage/recertification_T345.json).

> **Scope caveat (important).** This is an **extracted-policy** certification: the
> interpolated mean-effort curve is certified, **not** the trained stochastic
> network. The T=3/4/5 runs saved no `.pt`/`(α,β)` and their `effort_curves` cover
> only `|d| ≤ 4q` (states beyond use flat extrapolation). The fine-grid dReach
> reproduces the original per-seed certificate **exactly**, so the interpolant is
> faithful over the reachable support; a fully faithful network re-evaluation
> would require a retrain.

| T | certify (UCB gate) | dReach_UCB/ΔW mean | dReach_UCB/ΔW max | verdict vs original |
|---|--------------------|--------------------|-------------------|---------------------|
| 3 | **5/5** | 0.0101 | 0.0145 | unchanged (was 5/5) |
| 4 | **5/5** | 0.0161 | 0.0218 | unchanged (was 5/5) |
| 5 | **4/5** | 0.0207 | 0.0329 | unchanged (was 4/5; seed 46 lone non-cert, 0.0321→0.0329 under UCB) |

The UCB margin (+0.0004–0.0025) tightens the numbers but **flips no verdict**. T=5
is the first horizon where the conservative certificate binds (seed 46).

---

## 4. Verifier calibration & falsification (T=2)

The certificate is only meaningful if it separates the equilibrium from
non-equilibrium policies. Running the deterministic verifier on the closed form
(numerical error floor), the learned policy, and a bad-policy suite:

| policy | EXP/ΔW | dReach_UCB/ΔW | certified |
|--------|--------|---------------|-----------|
| closed form e*_CF (error floor) | 0.0000 | 0.0000 | yes |
| **TEL-PPO learned (seed 42)** | **0.0020** | **0.0056** | **yes** |
| constant low (e=5) | 0.2025 | 0.3324 | no |
| constant high (e=100) | 0.9286 | 0.9866 | no |
| one-stage effort repeated | 0.2221 | 0.3942 | no |
| gap-independent stage-2 (no-gap) | 0.0916 | 0.3500 | no |

The learned policy sits at the closed-form floor (`dReach_UCB/ΔW` 0.0056 vs 0.0000),
while every bad policy is **50–180×** above the 0.03 gate and fails. The
"no-gap stage-2" falsifier is the most instructive: ignoring the score gap at the
final stage is only mildly exploitable at the root (EXP/ΔW 0.09) but grossly
exploitable state-wise (dReach_UCB/ΔW 0.35) — exactly why the state-wise Δ
certificate, not root EXP, is the gate.

---

## 5. Second-order condition / validity (paper note)

The closed-form interior benchmark is a genuine equilibrium only where the stage
objective is globally concave. The binding threshold is

```
q_soc = sqrt(ΔW / (8k)) = 41.833   (for ΔW=4, k=1/3500)
```

Numerically verified: the global argmax of the stage-2 objective Q_2(d,·) over the
effort grid coincides with the interior FOC e*_{2,CF}(d) across the evaluation
region **iff q > q_soc**:

| q | stage-1 curvature | stage-2 global-dev gain | valid |
|---|-------------------|-------------------------|-------|
| 40 | +1.12e−4 | 1.88e−1 | **no** (SOC fails) |
| 45 | −1.45e−4 | 0.00 | yes |
| 50 | −2.91e−4 | 5.6e−17 | yes |
| 55 | −3.80e−4 | 0.00 | yes |

At q ≤ q_soc the symmetric candidate becomes a local minimum and a give-up
deviation is profitable (dev gain 0.19 at q=40); the sign of the stage-1 curvature
`−2k + ΔW²/(32kq⁴)` flips exactly at q_soc. **All runs use q=50 > q_soc**, so the
benchmark is valid. (Paper-usable statement: *the two-stage interior MPE exists and
is the global best response iff q > √(ΔW/8k); at the baseline this is q > 41.83,
satisfied by q=50.*)

---

## 6. Defect resolution (D1–D6)

| ID | Status |
|----|--------|
| **D1** on-path E[ê_2] never reported | **Fixed** — computed by quadrature and reported (44.69 vs 46.667); the policy was fine, the print was missing |
| **D2** figures incomplete/notation | **Deferred (Phase 1b)** — figure/table regen against the new schema; table code already backward-compatible |
| **D3** stale MC-SE certification | **Corrected** — verifier already deterministic; added the `dReach_UCB` refinement gate (no MC anywhere) |
| **D4** g→e notation | **Partial** — new outputs use e-notation keys; existing-key rename deferred to the coordinated Phase-1b regen |
| **D5** ckpt peeks at certifier | **Fixed** — selection grid 51 disjoint from certification grids 101/201 |
| **D6** mean vs mode | **Resolved** — mode ≈ mean at convergence (Δ ~0.16%); mean stays primary; α,β persisted |

---

## 7. Provenance & reproduction

- **Code commit at run time:** `e7d278a` (Phase-1 fixes on `feat/multistage-phase0`).
- **Grids:** selection 51 · certification coarse 101 / fine 201 · effort grid 201 ·
  quadrature 129 nodes · terminal integration closed-form F_ξ.
- **Certificate:** `dReach_UCB = dReach_fine + |dReach_fine − dReach_coarse|`, gate 0.03.
- **Artifacts:** `results/multi_stage/convergence/ms_T2_q50_seed{42..46}_p2cert_convergence.json`
  (+ `.pt` checkpoints, gitignored); `results/multi_stage/recertification_T345.json`;
  `results/two_stage_results.json`.

Reproduce:

```bash
# T=2 retrain (per seed; ~15 min CPU)
python run/run_multi_stage.py --q 50 --T 2 --seed <42..46> \
  --updates 2000 --episodes 512 --entropy-coef 0.005 --tag p2cert --device cpu

# T=3/4/5 extracted-policy re-certification (deterministic, seconds)
OMP_NUM_THREADS=4 python tools/recertify_multistage.py

# correctness unit tests (T1,T2,T7,T8,T9)
python tools/test_phase1_correctness.py
```

---

## 8. Status & what remains

- **Done:** corrected measurement + deterministic UCB certification; T=2 retrain
  (5/5) with full persistence; T=3/4/5 re-certification (verdicts unchanged);
  mean-vs-mode resolved; verifier calibration/falsification (§4) and SOC/validity
  (§5); this report + `two_stage_results.json`.
- **Deferred (Phase 1b, not started):** figure regeneration (stage-1 comparison
  panel, E[ê_2] annotation, T=2 Δ_t(d)); the falsification/SOC numbers are in §4–§5
  but not yet emitted as standalone paper `.tex` tables; and the existing-key g→e
  notation rename — all against the new JSON schema. Table code is already
  backward-compatible.
- **Optional:** retrain T=3/4/5 with the patched runner to obtain `.pt`/(α,β) for a
  full stochastic-policy re-evaluation and mean-vs-mode at those horizons (current
  T=3/4/5 certification is extracted-policy).
