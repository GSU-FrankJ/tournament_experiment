# Pre-registration: T=3 verified equilibrium (FROZEN 2026-07-09)

Fixes the T=3 acceptance criteria and training protocol BEFORE the gated
run. T=3 is the plan's main contribution (section 5.3): there is NO
closed-form benchmark, so PPO computes candidate effort functions
e_hat_1(d), e_hat_2(d), e_hat_3(d) and the independent DP verifier is the
SOLE certifier. Authorized by the T=2 gate PASS (T=2 validated the
train->extract->verify->certify pipeline against the closed form).

## Why the same threshold transfers from T=2

The verifier is IDENTICAL to the one calibrated at T=2 (phase03): closed-form
terminal integration, deterministic quadrature, dReach reachable-set
certificate. At T=2 it was shown to (a) assign ~0 exploitability to the
closed-form equilibrium (error floor dReach/DW ~ 2.5e-5) and (b) assign
large exploitability to misspecified policies (falsification). That
discriminatory power is a property of the verifier, not of T, so the same
certificate threshold applies at T=3. Numerical validity at T=3 is checked
per-run by grid refinement (51/101/201) + Richardson.

## PRIMARY gate (hard; decides "T=3 certified, proceed to T=4/5")

- **G1 certification (per seed):** `dReach / DW <= 0.03` (frozen
  `GATE_DREACH_OVER_DW`, unchanged from T=2). dReach upper-bounds root
  exploitability, so it cannot over-certify.
- **G2 reproducibility:** `>= 4/5 seeds` certify (frozen
  `GATE_MIN_CERT_SEEDS_FRAC = 0.8`).

PASS => the T=3 learned effort profile is a certified epsilon-approximate
MPE, reproducibly across seeds; T=4/5 benchmark extensions authorized.

## Reported diagnostics (NOT gating; no closed form exists)

- Per-stage learned effort functions e_hat_t(d), t=1,2,3 (Figure 3).
- Best-response vs learned effort per stage (Figure 4).
- State-wise one-step deviation gaps Δ_t(d): worst-case and on-path
  (Figure 5), plus reachable-set per stage.
- Exploitability certificate: EXP, EXP/DW, EXP^UCB/DW (grid-refinement
  Richardson band), dReach/DW, dFull; cross-seed mean +/- std.
- Expected economic patterns (plan): effort high when the contest is close,
  lower when far ahead, possible discouragement far behind, later stages
  more gap-sensitive. Reported qualitatively from the curves.

## Training protocol (frozen)

| Item | Value |
|---|---|
| Parameters | w_h=6, w_l=2, k=1/3500, e_bar=100, q=50 |
| Horizon | T=3 |
| Seeds | 42, 43, 44, 45, 46 |
| Updates | 3000 (raised from T=2's 2000: sparser terminal credit over 3 stages) |
| Episodes / update | 512 |
| gamma, lambda | 1.0, 1.0 |
| Policy | Beta (mean/conc), state [t/T, d/(q sqrt t)] |
| lr / clip / entropy | 3e-4 / 0.2 / 0.005 |
| Epochs / minibatch | 10 / 256 |
| Exploring starts | on, es_on_path_fraction=0.5 (covers all t in 1..3) |
| Extraction | Beta MEAN |
| Checkpoint rule | lowest validation dReach (pre-specified) |
| Verifier | DP, d-grid 201, e-grid 401, closed-form terminal, dReach cert |
| Device | CPU (tiny net is CPU-bound; GPU adds transfer overhead) |

## Deferred (separate follow-up, not part of this gate)

- Curriculum ablation (T=1->2->3 warm start vs direct) — plan 5.4. Requires
  checkpoint-transfer infrastructure; the main T=3 result is direct-from-
  random training, which is the more conservative claim.
- Adversarial-RL best-response cross-check — plan 5.4 secondary robustness.

## Decision rule

`tools/evaluate_gate.py --glob ".../ms_T3_q50_seed*_gateT3_convergence.json"`.
Its PASS/FAIL verdict (5/5 or >=4/5 certify) is the record. RE_1/RPE_2
columns are NaN by design (no closed form) and do not gate.
