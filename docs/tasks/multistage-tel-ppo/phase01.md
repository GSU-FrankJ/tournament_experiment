# Phase 01: theory audit + config validation (COMPLETE 2026-07-09)

## Objective

Before any training minute: (1) independently verify the two-stage
closed-form derivation on main, (2) turn the validity region into
executable q_crit checks wired into a canonical config.

## Deliverables

| Artifact | Status |
|---|---|
| `docs/technical/two_stage_benchmark_audit.md` | done — 4 findings + errata |
| `utils/theory_multistage.py` | done |
| `config/multi_stage_two_players.py` | done — validate() raises on bad q |
| `tools/verify_two_stage_benchmark.py` | done — PASS |

## Key numbers (canonical convention c = k e^2, w_h=6, w_l=2, k=1/3500)

| Quantity | Formula | Value (q=50) |
|---|---|---|
| g1 (stage-1 effort at d=0) | ΔW/(6kq) | 46.67 |
| g2(d) (stage-2 function) | ΔW(2q−|d|)/(8kq²) on |d|≤2q | peak 70.00 |
| E[g2(d₂)] on-path | = g1 exactly | 46.67 |
| U_eq | (w_h+w_l)/2 − 17ΔW²/(288kq²) | 2.678 |
| q_SOC (binding) | √(ΔW/(8k)) | 41.83 |
| q_B2 / q_B1 / q_PC | ΔW/(4kē) / ΔW/(6kē) / PC(Ū=0) | 35.0 / 23.3 / 28.7 |
| **q_crit** | max of the four | **41.83** |

Stage-1 SOC correction: curvature = −2k + ΔW²/(32kq⁴) (kink term at d=0
flips the doc's sign); negative iff q > q_SOC. Numerically tight: at q=40
the symmetric candidate is a local MINIMUM (curv +1.1e-4, give-up
deviation gains +0.061).

## Acceptance

- [x] All analytic claims cross-checked numerically (PASS).
- [x] validate() rejects q=35/40, accepts q_list=[45,50,55].
- [x] Errata list delivered for the plan Word doc.
