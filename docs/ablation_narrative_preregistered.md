# Pre-registered narrative for the dual-endpoint ablation (written 2026-08-02, BEFORE results)

Registered before the r8 wave, the stage-A/B polish, and the MC-BR-only
baseline finish, so the interpretation is not chosen after seeing the numbers.

## The prediction

MC-BR polishing is a strong attractor: in every Table-4 cell to date the
polished cross-seed SD is 0.01–0.07 regardless of the raw starting point.
We therefore expect, in the dual-endpoint Table 3:

- **Polished column: all arms collapse together** (TEL-PPO ≈ w/o stability ≈
  w/o exploitability ≈ MC-BR-only, each ± ≤0.1). No arm separation.
- **Raw column: the arms separate.** This is where the ablation's evidence
  lives (landing bias, cross-seed SD, and the no-verification arm's
  cell-to-cell instability across generations: r5's bad cell was q45,
  r7's is q55).

## What each column then measures

- The **polished column measures the game** (the solver's answer, i.e. the
  instrument): damped BR iteration from any interior start converges to the
  unique interior NE. It is a property of MC-BR polishing, not of the learner.
- The **raw column measures the learner**: where independent PPO agents,
  training only on sampled own-play tournament outcomes, actually land — and
  whether the stopping was certified (exploitability ≤ 0.03, 5 consecutive
  independent checks) or hit the budget.

## What TEL-PPO is for, if polish alone also finds e*

The MC-BR-only baseline (`tools/mc_br_only_baseline.py`) will, we expect,
land on e* with SD comparable to the TEL-PPO polished rows. The paper's claim
must therefore NOT be "TEL-PPO is needed to locate e*". The defensible claims:

1. **TEL-PPO answers a learning question, polish a computation question.**
   The result is that *learning agents converge to near-equilibrium behavior*
   (raw column) and that this convergence is *certified online* by the
   verifier. MC-BR polish is the referee and refiner, not the phenomenon.
2. **Query models differ.** MC-BR needs an oracle for counterfactual
   deviations: payoffs at arbitrary chosen profiles, 150000 samples per grid
   point, at profiles nobody plays. PPO agents observe only realized outcomes
   of their own play. In a real tournament only the second channel exists.
3. **The policy object.** TEL-PPO produces a behavioral policy (a Beta
   distribution whose concentration the schedule controls); polish produces a
   point. Claims about learning dynamics, KL stability, and verified stopping
   attach to the policy, not the point.

## Pre-registered non-claims (do not write these)

- Do NOT claim TEL-PPO saves polishing compute: the canonical POL runs a
  fixed 320 rounds with no early stop (min_rounds=999 > max_rounds=320,
  tau_e=0), so polish cost is start-independent **by design**.
- Do NOT claim the polished column separates the training arms (unless it
  does — see branch B).

## Branch B (surprise outcome)

If the polished column DOES separate arms — e.g. polishing cannot recover the
no-verification arm's landings, or MC-BR-only lands off e* from some ladder
starts — that is a STRONGER result for TEL-PPO (a good start is then
necessary, not decorative), and the raw-column narrative above still holds
unchanged. Wording: "polishing recovers e* only from verified-quality
starts; from uninformed starts it fails in X/9 cells."

## Table 3 presentation plan (dual endpoint, five rows)

| Arm | raw endpoint | polished endpoint |
|---|---|---|
| TEL-PPO (full) | learning + certification evidence | instrument check |
| w/o stability screening | ↑ raw column carries the story | ↑ expected ≈ equal |
| w/o exploitability verification | ↑ | ↑ |
| MC-BR only (start 50, no training) | 50.00 (by construction) | expected ≈ e* |
| MC-BR only (ladder 10/30/70/90) | start value (by construction) | robustness appendix |

Data: `results/one_stage_ablation/polish_per_seed_r7.json` (stages A+B),
`results/one_stage_ablation/mc_br_only.json`, raw endpoints from the
`r7_state4` / `r7_fig7_*` / `r8_unified` convergence JSONs.
