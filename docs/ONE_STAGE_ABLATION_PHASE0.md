# One-Stage Ablation — Phase 0 Design-Confirmation Note (STOP for review)

**Date:** 2026-07-16 · **Server:** vector2 · **Branch:** `feat/multistage-phase0` · **HEAD:** `d3eeea9`
**Scope:** Phase 0 only. Zero training. No existing module modified. New files only:
`tools/one_stage_referee.py`, `tools/one_stage_ablation_phase0.py`, `tools/one_stage_polish_probe.py`;
data under `results/one_stage_ablation/`.

**Bottom line: one design-critical assumption is FALSIFIED (both by code and by experiment).
`utils/mc_br_polish.py` is a damped fixed-point solver, not a single BR step. P4/P5/P6/P7 and the
polished-arm error decomposition must be re-specified before Phase 1. Everything else confirms.**

---

## 1. Canonical inputs + reconstruction (Phase 0 item 1) — ✅ exact

10 JSONs: `results/two_players/convergence/ppo_q{35,55}.0_seed{42..46}_r5_sampled_convergence.json`
(Set 1; `wh8_wl4` excluded). Provenance: `run_id=20260610_194959`, `git_sha=444636040fa9`,
`ablation_name=r5_sampled`; all 10 `stop_reason="exploitability"`.

ê_raw reconstructed as `100·α/(α+β)` from the final `(alpha_mean, beta_mean)`:

| q | 5-seed mean ê_raw | sd | CSV target | max bit-check \|e_rec − final.effort\| |
|---|---|---|---|---|
| 35 | **43.5798** | 1.1154 | 43.58 ✅ | 7.11e-15 |
| 55 | **29.6543** | 0.6807 | 29.65 ✅ | 3.55e-15 |

Bit-check is exact to float roundoff; both cross-validate the committed Claim-B CSV.

## 2. STOP-FLAG #1 — the polish is NOT a single BR step (design-critical)

**Code** (`utils/mc_br_polish.py:171-207`, canonical config `tools/phase0_verify.py:30,116-117`):

```
POL = dict(eta=0.4, M=150_000, min_rounds=999, max_rounds=320, n_avg=200, tau_e=0.0, bias_correct=True)
```
- **Iterates**: damped *simultaneous* BR, `e ← (1−η)e + η·BR(e)`, η=0.4 (`:196`).
- **Always 320 rounds**: `min_rounds=999 > max_rounds=320` ⇒ the early-stop branch (`:198`) never
  fires; `tau_e=0.0` ⇒ `converged` is False by construction (a config artifact, not a failure).
- **Polyak-Ruppert average** of the last `n_avg=200` iterates is the returned value (`:203-207`).
- **BR is LOCAL**: coarse grid only within `±window=10.0` of the current effort, step 0.5, then
  fine ±1.0 @0.05, plus quadratic-vertex debias (`:122-159`).
- Clipping to `[0,100]` throughout.

**Experiment** (`results/one_stage_ablation/phase0_polish_probe.json`, canonical polish, seed 4000,
6 runs × 111s):

| q | start | landing | 1-step BR (referee) | e* | \|landing − BR₁\| | \|landing − e*\| |
|---|---|---|---|---|---|---|
| 35 | 43.5798 (raw) | 44.9407 | 44.7165 | 45.4545 | 0.224 | 0.514 |
| 35 | 0.0 | **44.9488** | 27.5591 | 45.4545 | **17.390** | 0.506 |
| 35 | 50.0 | **44.9335** | 37.0370 | 45.4545 | **7.897** | 0.521 |
| 55 | 29.6543 (raw) | 28.7851 | 28.6656 | 28.9256 | 0.119 | 0.141 |
| 55 | 0.0 | **28.7722** | 22.9030 | 28.9256 | **5.869** | 0.153 |
| 55 | 50.0 | **28.7917** | 21.4067 | 28.9256 | **7.385** | 0.134 |

**Landing spread across starts {raw, 0, 50}: 0.0153 (q35), 0.0195 (q55) ⇒ INIT-INDEPENDENT.**
The polish converges to a *sampled fixed point* — 44.94 (q35) / 28.78 (q55) — regardless of where it
starts. Cross-validation: the raw-start landings reproduce the committed 44.95 / 28.76. ✅

### Consequences (what breaks)

- **P4 is a valid number but mislabeled.** The referee's 1-step BR is 44.7165 / 28.6656 (matches P4),
  but that is *not what the polish computes*.
- **P5's decomposition is invalid.** `err_pol = (BR_exact(ê_raw) − e*) + (e_pol − BR_exact(ê_raw))`
  presumes one-step truncation. There is no truncation term: the polish runs to a fixed point.
  The +0.23/+0.09 gap is the distance between two unrelated objects, **not** "polish solver noise".
- **P6/P7 are falsified for the polish arm.** P7's headline — *q35 c=50 → 8.42, worse* — is **FALSE**:
  the polish lands at 44.93, error **0.52**, i.e. **better** than the start's 4.55. Same at q55
  (c=50 → 28.79, not 21.41). P6/P7's values are correct only as *referee* 1-step BRs (arm **A4**).
- **The q35 "expansive regime" does not destabilize the polish.** Damping neutralizes it: the
  iteration-map slope is `1−η+η·BR′ = 0.6 + 0.4·(−1.8519) = −0.1408`, |·| < 1 ⇒ contractive, even
  though |BR′| = 1.85 > 1. The expansive asymmetry is real in the **BR map** (A4/F1) but is not
  expected to appear in the control arms.

### Proposed re-specification (needs your call)

1. **Replace the decomposition** with one that matches the algorithm:
   `err_pol = (e_fp − e*) + (e_pol − e_fp)` = **sampled-fixed-point offset** + **solver/Polyak noise**.
   Measured this session: offset ≈ **−0.51** (q35) / **−0.14** (q55); solver noise ≈ **0.015–0.020**
   (spread across starts). I.e. essentially *all* residual polished error is the sampled-vs-analytic
   fixed-point offset, not solver noise — the opposite of P5's framing.
2. **Re-frame A3 controls** as an **init-independence test of the solver** (which they decisively are),
   and keep the BR landing map as the referee-side **A4** exhibit (P6/P7 live there, correctly).
3. **Note for C1:** because the polish is init-independent, the polished arm is ~constant across seeds
   (spread ≈0.02). Per-seed paired stats therefore reduce to "is raw |err| > the fixed-point |err|?"
   — the polished arm has ~zero seed variance **by construction**. Pre-visible consequence: at q35,
   seed 45 (raw |err| = 0.266) is *better* than the fixed point (0.51), so polish will make that seed
   **worse** — C1 would land at 4/5 (still passing its ≥4/5 bar) via a mechanism worth reporting.

## 3. STOP-FLAG #2 — legacy MC estimator consumes a policy, not an effort

`run/run_two_players.py:266 eval_exploitability(agent, ...)` touches exactly four attributes:
`agent.net.parameters()` (device only, `:201,290`), `agent.state_from_params(q,k,w_h,w_l)` (`:212`),
`agent.dist(state) → (dist,_)` (`:214`), `dist.sample((M,), generator=)` (`:216`). It samples the
Beta for **both** players and cannot evaluate a fixed deterministic effort.

- **Minimal adapter** (new file, shipped module untouched): a stub exposing those four attributes,
  whose `dist.sample((M,))` returns a constant tensor at the target normalized effort (a degenerate
  point policy). Exact for a deterministic profile.
- **Comparability caveat:** running the legacy estimator on the *actual raw agent* measures
  **stochastic-policy** exploitability — a different (larger) quantity than the referee's
  deterministic-profile EXP. For the C4 noise-floor exhibit to be apples-to-apples, both must be run
  on the **same deterministic profiles** via the adapter.
- **Alternative already available:** `utils/mc_br_polish.exploitability_frozen_profile:211` natively
  takes a deterministic profile (fresh draws, grid 0.25). **I used this for P8** — it is *not* the
  shipped in-training estimator (M=8192 **CRN**, coarse→fine). Please pick: (a) adapter + shipped
  `eval_exploitability` (faithful to "the shipped estimator"), or (b) `exploitability_frozen_profile`
  (no adapter, slightly different grid/noise handling). **My recommendation: (a)**, with (b) as a
  secondary row.

## 4. STOP-FLAG #3 — ε = 0.03 is ABSOLUTE, not /ΔW

`config/one_stage_two_players.py:108` sets `exploit_eps: 0.03`; `run/run_two_players.py:1425`
compares it to `exploitability_val` = `best_delta` = a **payoff difference** (`:343,356`).
⇒ **one-stage ε = 0.03 is absolute payoff units** = **0.008571 /ΔW**.
By contrast `config/multi_stage_two_players.py:84` sets `epsilon_over_dw: 0.03` — **normalized**.
**The same numeral 0.03 means different things in the two pipelines.** Where the new values sit:

| q | EXP_det(raw) abs | ×ε | EXP_det(pol) abs | ×ε |
|---|---|---|---|---|
| 35 | 1.172e-03 | 0.039 | 8.489e-05 | 0.0028 |
| 55 | 3.962e-04 | 0.013 | 1.195e-05 | 0.0004 |

Both arms sit **1.5–3 orders of magnitude below** the historical stop threshold.

## 5. Per-seed polished values do NOT exist (Phase 0 item 5)

`tools/phase0_verify.py:112-125` computes per-seed 2P polish but prints **only the mean**;
`results/phase0_verify_20260701_1941.log:37-38` carries only the aggregate (44.95 / 28.76);
`tools/make_one_stage_claimb_summary.py:22-24` states per-seed polished values were never persisted.
⇒ **A2 requires a re-run.** **Proposed seed policy (reproduces canonical exactly):** start =
per-seed raw mean, `seed = 4000 + si` for `si = 0..4` over sorted seeds 42..46, `POL` as above
(`tools/phase0_verify.py:117`). Cost: 10 runs × ~111 s ≈ **19 min** (CPU, tmux). A3 controls at
seed 4000 are **already done** (§2 table).

## 6. Referee built and validated (Phase 0 item 6)

`tools/one_stage_referee.py` — closed-form, zero Monte Carlo. **21/21 unit tests pass**:
F_ξ == ∫f_ξ (max err 5e-6), F_ξ(0)=0.5, endpoints, ∫f_ξ=1, **BR(e*)==e* exactly**,
**EXP(e*) = 0.000e+00** (analytic floor), EXP ≥ 0 on probes, and the two independent BR paths agree
to **≤ 6.5e-11** (analytic FOC+corners vs 20001-grid + parabolic refinement). EXP_UCB = EXP_fine +
|EXP_fine − EXP_coarse| (20001/5001); the discretization margin is **0.0e+00** at these grids, so
EXP_UCB == EXP_fine — the grids are already saturated.

**BR-map structure independently reproduced** (matches the prompt exactly):

| q | a | slope below | slope above | \|above\|>1 |
|---|---|---|---|---|
| 35 | 7.142857e-04 | +0.3937 | **−1.8519** | **True** (expansive) |
| 55 | 2.892562e-04 | +0.2082 | −0.3568 | False |

`q_expansive = √(ΔW/4k) = 39.886` ✅

### Tripwire verdicts (at 5-seed-mean profiles)

| # | Quantity | q=35 measured / pred | q=55 measured / pred | Verdict |
|---|---|---|---|---|
| P1 | EXP_det(raw) abs | 1.1721e-03 / 1.2e-3 | 3.9622e-04 / 3.9e-4 | **agree** |
| P2 | EXP_det(pol) abs | 8.4889e-05 / 8.5e-5 | 1.1945e-05 / 1.2e-5 | **agree** |
| P3 | ratio raw/pol | 13.81× / 14× | 33.17× / 33× | **agree** |
| P4 | 1-step BR(ê_raw) | 44.7164 / 44.72 | 28.6656 / 28.67 | agree (but **mislabeled**, §2) |
| P5 | e_pol − BR₁ | +0.2336 / +0.23 | +0.0944 / +0.09 | number agrees, **label invalid** (§2) |
| P6 | BR(0) | 27.5591 / 27.6 | 22.9030 / 22.9 | agree **as A4**; **falsified as a polish landing** |
| P7 | BR(50), err before→after | 37.0370, 4.55→8.42 worse / 37.0 | 21.4067, 21.07→7.52 / 21.4 | agree **as A4**; **falsified as a polish landing** |
| P8 | legacy-MC floor (M=8192, R=5) | level **6.66e-03** ± sd 1.64e-03 / "SE ≈5e-3" | level **7.15e-03** ± sd 3.36e-03 / "SE ≈4e-3" | see below |
| P9 | q55 crossing | — | raw 29.654 > e* 28.926 > e_pol 28.76 | **confirmed** |

**P8 — label ambiguity, not a material disagreement.** The MC reading is not zero-mean noise: EXP is a
**max over a grid of noisy estimates**, so it is **upward-biased**. I measured both a *level*
(6.7e-3 / 7.2e-3 — same order as the predicted 5e-3/4e-3) and a *rep-sd* (1.6e-3 / 3.4e-3). Please
confirm P8 meant the **level**. Either way the design's point stands, and stronger than predicted:
**the floor (~7e-3) exceeds BOTH arms' true EXP** (1.2e-3 raw, 8.5e-5 polished) ⇒ the legacy MC
cannot separate the arms, while the referee separates them by 14×/33×. This also explains the
committed runs' `final_exploit_max` ≈ 0.004–0.010 — those readings are the floor, not the policy.

**P9 mechanism caveat:** the crossing is confirmed, but attributing it to "negative BR slope" is
questionable — the polish lands at a **fixed point**, where the BR slope does not induce a systematic
crossing. The measured cause is the **sampled-fixed-point offset** (−0.14 at q55), which sits below
e* in *both* cells (−0.51 at q35, where there is no crossing because raw is below e*). Suggest
re-wording P9's mechanism.

---

## 7. Deviations / decisions I need from you

1. **[Design-critical] Re-specify P4–P7 + the decomposition** per §2 (offset + solver noise), and
   re-frame A3 as an init-independence test. **Blocks Phase 1.**
2. **[Design-critical] Legacy-MC choice**: adapter + shipped `eval_exploitability` (recommended) vs
   `exploitability_frozen_profile`. **Blocks Table B / gate C4.**
3. **P8 semantics**: level (~7e-3) vs SE (~2-3e-3)?
4. **Deliverable path conflict**: the prompt says `results/one_stage_ablation/ONE_STAGE_ABLATION.md`,
   but repo convention is prose → `docs/`, data → `results/`. **Proposal:** report at
   `docs/ONE_STAGE_ABLATION.md`, JSON + figures under `results/one_stage_ablation/`. Confirm.
5. **C1 reading** (§2.3): the polished arm is ~constant across seeds by construction; the paired
   per-seed statistics are near-degenerate. Confirm you still want the sign-count framing.

**Phase-1 cost if approved:** ≈19 min CPU (A2 10 polish runs, tmux) + referee/MC (seconds). No GPU.

**STOP — awaiting approval.**
