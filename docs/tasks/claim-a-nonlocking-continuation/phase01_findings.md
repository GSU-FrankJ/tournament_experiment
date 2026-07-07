# Phase 01 findings — continuation design analysis

Run 2026-07-02, vector2. Script `tools/claim_a_continuation_design.py`; raw dump
`phase01_design.json`; log `phase01_run.log`. ZERO GPU. 3P q35, e*=25 (benchmark only).

## TL;DR — **pre-registered design-analysis kill condition (2) FIRED: recommend STOP before GPU**

The κ-continuation target path exists (D1), but the autopsy (D2) shows the velocity
deaths happened with a HEALTHY optimizer — the failure is gradient-SNR physics on the
flat plateau, which lr/entropy floors cannot fix. Per the pre-registered rule in
CLAUDE.md, the recommendation is STOP. The measured μ*(κ) curve is itself a valuable
paper artifact (it quantitatively explains the raw undershoot); details below.

## D1 — the smoothed-equilibrium curve μ*(κ) exists and explains everything so far

Fixed point of the sampled BR against opponents drawn from the ModeConc Beta family
(α+β=κ+2, mean=μ), 2 CRN replicate seeds per point (μ* noise roughly ±0.3–0.5, the
payoff plateau makes argmax-BR noisy):

| κ | 20 | 35 | 60 | 100 | 200 | 400 | ∞ (det) |
|---|---|---|---|---|---|---|---|
| μ*(κ) | 22.59 | 22.74 | 23.20 | 23.79 | 23.96 | 24.74 | 24.55¹ |

¹ det crossing is noise-limited on the flat plateau; Finding B pins the deterministic
sampled equilibrium at ~24.7–25.0 via FOC, which is the better estimate. μ*(400) >
μ*(det) is within MC noise.

Consequences:
- **The continuation target path is real and monotone** — that part of the design was
  sound: κ_top ≈ 400+ would be needed for a ≥24.5 (2%) raw target.
- **Component-2 was structurally capped**: its κ_top=200 put the smoothed target at
  ≈24.0 — even PERFECT tracking could not have satisfied Claim A (would land 4% under).
- **The r5 "stall" and c2 outcomes are rationalized**: r5's 22.99 sits in
  [μ*(20), μ*(60)] = [22.6, 23.2]; the c2 finals (mean 22.73, std 1.67) are consistent
  with a target μ*(200)≈24.0 plus a ±1.5–2 diffusion band (see D2). Raw PPO is not
  "failing" — it converges to (the diffusion neighbourhood of) the equilibrium of the
  exploration-smoothed game it actually plays.

## D2 — velocity death happens with a healthy optimizer → physics, not schedule (KILL)

Per κ-stage segments of the c2 ramp windows (velocity in units/update; approx_kl is
the discriminator — collapsed KL would mean optimizer starvation, fixable by floors):

| seed | seg | vel | approx_kl | verdict |
|---|---|---|---|---|
| 42 | κ=100 | +0.006 | 0.0068 | died, KL healthy |
| 42 | κ=200 | −0.002 | 0.0083 | died, KL healthy (highest KL of its run!) |
| 45 | κ=100 | +0.011 | 0.0073 | died, KL healthy |
| 45 | κ=200 | **−0.103** | 0.0068 | moved AWAY 2 units, KL healthy |
| 44 | κ=200 | +0.076 → mode 27.0 | 0.0090 | blew PAST both targets by ~3 units |
| 43 | all | +0.06..+0.08 | 0.006–0.007 | tracked (the lucky draw) |

Explore-tail KL 0.0037–0.0053 vs ramp KL 0.0055–0.0090 — the optimizer took *larger*
steps during the ramp than in explore. Nothing was starved.

Reading: within ~1.5 units of the smoothed target the payoff slope is below the batch
noise floor, so the mean/mode **diffuses** rather than climbs — down (s45), up past the
target (s44), or stalls (s42), direction random per seed. That is exactly the c2
cross-seed spread (std 1.67), and floors/lr/entropy tweaks cannot manufacture signal
that the sampled payoffs do not contain at batch size 4096.

**Pre-registered kill condition (2)** ("velocity death with HEALTHY approx_kl →
physics → recommend STOP before GPU") **is met.**

## D3 — moot, recorded for completeness

Ladder climb budget at p25 healthy velocity would be tiny (~36 updates of climb), but
that number presumes directed velocity, which D2 shows does not exist within ~1.5 units
of the target. The binding constraint is the diffusion band (±1.5–2), which already
violates Gate C (std ≤ 0.5, mean ≥ 24.5) in expectation.

## Gate recommendation (owner decision)

- **STOP before GPU** (pre-registered rule). Any κ-schedule member of this family —
  locking (Component-2), non-locking, or convergence-gated continuation — faces the
  same two walls: (i) κ ≤ 200 caps the target at ~24; (ii) within ~1.5 units of any
  target the signal is sub-noise ⇒ landing spread ≥ the Gate-C kill zone. Beating it
  would need a *variance-reduction* change to the learning signal itself (e.g. much
  larger batches, baselines/CRN inside PPO's reward path — the latter touches the
  sampled-training invariant), i.e. a different research question.
- **What the owner gets instead**: the μ*(κ) curve is a quantitative, sampled-only
  explanation of the raw undershoot — "PPO converges to the exploration-smoothed
  equilibrium; the deterministic-equilibrium gap is bridged by MC-BR + exploitability
  verification". Suggested paper artifact: μ*(κ) curve figure with r5/c2 raw outcomes
  overlaid. This upgrades Claim B from a concession to a principled two-stage method.
- Claim A verdict across all three attempts (r5 schedule, Component-2 lock, this
  design analysis): not reachable in this parameterization at batch 4096. Recorded to
  prevent future resurrection without new variance-reduction evidence.
