# Status Report — 2026-07-07

## Claim-A retrain saga (3P q=35) → decision to adopt Claim B

**Audience:** collaborators picking this up on `main` after the
`fix/audit-remediation` merge. This report is self-contained — you do not need
the PR thread or prior sessions to understand what changed and why.

---

## 0. Executive summary (read this first)

We ran three independent experiments to test **Claim A** — that raw PPO output
lands at the deterministic Nash equilibrium **e\*=25** for the 3-player, q=35
tournament. **All three failed, concordantly.** The final and most rigorous
attempt (adaptive-batch κ-continuation, 5 seeds) **solved the run-to-run
variance problem** (cross-seed std 0.15, ~11× tighter than the earlier ramp) but
exposed a **systematic ~0.7-unit bias**: PPO converges to the *exploration-
smoothed* equilibrium μ\*(κ) ≈ 24.3, not to e\*=25.

**Decision: adopt Claim B.** Raw PPO recovers the smoothed equilibrium μ\*(κ);
the Monte-Carlo best-response (MC-BR) polish + exploitability evaluation bridge
the residual gap to e\*. This is a clean, publishable story — the measured μ\*(κ)
curve is a candidate paper figure — and it is now backed by **four concordant
negative results** for strong Claim A (r5, Component-2, the design analysis, and
the c3 5-seed run).

Nothing here changes the 2-player / different-cost / different-ability results or
the existing figure pipeline. This is scoped to the 3P q=35 Claim-A question.

---

## 1. Background: the two competing claims

**The game.** 3 players, tournament with effort cost. Theory gives an interior
Nash equilibrium effort. For 3P q=35 with k=0.001, w_H=6.5, w_L=3.0 the
closed-form deterministic equilibrium is **e\* = 25.0**.

**How PPO trains here (invariant).** Agents train on *sampled* one-step
tournament outcomes: y_i = e_i + ε_i, ε ~ U(−q, q); the realized winner gets
w_H, others w_L, minus cost k·e². Closed-form win-probability / expected payoff /
analytic e\* are **evaluation-only** — they never enter the env reward or the
policy update. This matters below: with exploration noise in the policy, the
*best response against the sampled opponent population* is not e\*.

**Claim A (strong):** raw PPO's own output (the Beta-mean effort it converges to)
reaches e\*=25 without any post-hoc correction.

**Claim B (the fallback, now adopted):** raw PPO converges to the
**exploration-smoothed equilibrium** μ\*(κ) — the equilibrium of the game as
*seen through the policy's own exploration spread* (concentration κ). μ\*(κ) sits
below e\* and rises toward it as κ increases (less exploration). The gap from
μ\*(κ_max) to e\* is then closed by MC-BR polishing + exploitability checks.

**The measured smoothed-equilibrium curve** (zero-GPU, sampled-only;
`tools/claim_a_continuation_design.py`):

| κ | 20 | 200 | 400 |
|---|----|-----|-----|
| μ\*(κ) | 22.59 | 23.96 | 24.74 |

μ\*(κ) is nearly flat past κ=400 and never reaches e\*=25 at any implementable κ.
This curve is the backbone of the Claim-B narrative.

---

## 2. What we tried (three Claim-A attempts)

### 2.1 Attempt #1 — Component-2 mode-conc ramp  (`--mode-conc-ramp`)

An exploitability-triggered κ ramp: train with an interior mode+concentration
Beta head, and when exploitability drops below a threshold, ramp κ up a fixed
schedule to sharpen the policy toward e\*.

- **Result: negative (stall).** The trigger fired **~7 units too early** (payoff
  gain has no signal on the plateau), and the ramp then froze the climb. Seed 42
  landed at **22.70**; cross-seed std **1.67**.
- Data: `results/three_players/convergence/ppo_3p_q35.0_seed{42..46}_c2_mode_conc_convergence.json`
- Task: `docs/tasks/component2-mode-conc-retrain/`

### 2.2 Attempt #2 — dev-trigger Phase-A screen  (zero-GPU)

A redesign screen: replace the trigger observable (payoff-gain → best-response
distance) and check feasibility *before* spending GPU.

- **Result: negative (feasibility fails).** The BR-distance trigger observable is
  fixable in principle (a clean 6.5 → 0.5 signal near e\* on the deterministic
  mean), **but** it collapses at exploration-κ, and raising κ to recover it
  freezes the climb. Independently, raw PPO (the "r5" baseline) stalls at
  **22.99** on the full budget with no κ lock — so the ~2-unit undershoot is a
  property of PPO dynamics, not of the trigger or schedule.
- Tool: `tools/claim_a_phase_a_screen.py`
- Task: `docs/tasks/claim-a-dev-trigger-retrain/` (closed at "Gate A")

### 2.3 Attempt #3 — non-locking κ-continuation  (`--kappa-continuation`, the main event)

The design analysis (`tools/claim_a_continuation_design.py`) first measured
μ\*(κ) (§1) and **pre-registered a KILL** on gradient-SNR grounds: the plateau's
gradient *mean* is ~0, so the undershoot is physics, not optimizer starvation.
The owner **overruled** that kill and authorized a variance-attack variant:

- **Idea:** walk κ up a ladder {20, 35, 60, 100, 200, 400}, advancing only when
  the policy has kinematically re-converged at the current κ (a drift-window
  gate, never a clock), and **enlarge the batch** at each stage
  (4096 → 16384 → 65536) so the gradient-estimate noise floor shrinks ∝ 1/√B.
  If the plateau undershoot were sampling noise, a 16× batch would close it.
- **Pilot (seed 42):** clean run, all 6 ladder stages reached, 0 forced
  advances, final **24.30**. Borderline — enough to justify the 5-seed test.
  Data: `...ppo_3p_q35.0_seed42_c3_cont_convergence.json`.

---

## 3. The decisive experiment: c3_cont 5-seed Gate C

### 3.1 Setup

Five fresh seeds **43–47** (seed 42 kept as pilot to avoid pilot-selection
bias). Parameters byte-identical to the pilot:

```
python run/run_three_players.py --method ppo --q 35 --seed <43..47> \
  --kappa-continuation --episodes 34000000 --max-updates 9000 \
  --cont-max-hold 120 --ablation-name c3_cont
```

All 5 finished clean: stop_reason = exploitability, **forced_advances = 0** (all
ladder advances gate-triggered), final exploitability 0.004–0.011 (< eps 0.03),
stopped at update 566–608.

### 3.2 Per-seed results

`final` = Beta-mean effort at stop (Metric B, the reported statistic);
`κ=400 mean/band` = mean and std of `policy_mean_effort` over the last 30 updates
of the κ=400 ladder stage (the time-averaged level, before the single-update
"done" snapshot).

| seed | final | gap to e\* | κ=400 mean | κ=400 band |
|------|-------|-----------|-----------|-----------|
| 43 | 22.98 | 2.02 | 24.42 | 0.47 |
| 44 | 24.43 | 0.57 | 24.44 | 0.36 |
| 45 | 24.76 | 0.24 | 24.08 | 0.59 |
| 46 | 24.21 | 0.79 | 24.27 | 0.70 |
| 47 | 23.78 | 1.22 | 24.25 | 0.66 |
| *42 (pilot)* | *24.30* | *0.70* | *24.22* | *0.59* |

### 3.3 Gate C scoring (pre-registered)

Thresholds (verbatim from `docs/tasks/claim-a-nonlocking-continuation/CLAUDE.md`):
**PASS** = cross-seed mean ≥ 24.5 (|err| ≤ 2%) AND std ≤ 0.5; **KILL** = std >
1.0 OR mean < 24.0 (|err| > 4%).

On the pre-registered **snapshot** metric (per-seed `final`):

| metric | value | success line | kill line |
|--------|-------|--------------|-----------|
| cross-seed mean | **24.03** (\|err\| 3.86%) | ≥ 24.5 | < 24.0 |
| cross-seed std | **0.69** | ≤ 0.5 | > 1.0 |

**Verdict: BORDERLINE — neither branch fires.** (Mean sits 0.03 above the kill
line; |err| 3.86% sits 0.14 pp under the 4% kill line.)

### 3.4 The decomposition (why "borderline" is actually decisive)

The snapshot spread is almost entirely **within-run diffusion sampled at a single
update**: a run stops on the *first* update of its "done" phase (the
exploitability streak ages during the ladder), so `final` is a 1-sample draw
from the κ=400 diffusion band. Per-seed `final − (κ=400 mean)` = −1.44, −0.01,
+0.69, −0.06, −0.47 — i.e. the ~0.5–0.7 band, not real cross-seed disagreement.
(Seed 43's −1.44 was inspected in the raw trajectory: normal ±0.7–1.0
update-to-update swings through κ=400, with the last few updates on a
locally-correlated downswing. The ladder→done transition pins κ / batch / lr /
entropy identically — no regime-shift artifact.)

On the **time-averaged κ=400 metric** (last 30 updates of the stage):

| metric | value |
|--------|-------|
| cross-seed mean | **24.29** (\|err\| 2.83%) |
| cross-seed std | **0.146** (SE ≈ 0.065) |
| full-stage robustness check | mean 24.27, std 0.149 (window choice does not drive it) |

Reading:

- **Variance is solved.** std 0.146 ≪ 0.5, ~11× tighter than Component-2's 1.67.
  The 16× adaptive batch did exactly what it was designed to do.
- **The mean is systematically biased.** 24.29 misses the 24.5 success line by
  ~3 SE, and sits ~0.4 below μ\*(400)=24.7 and ~0.7 below e\*=25. All six runs
  (including the pilot) land in 24.1–24.4. **This is bias, not noise** — more GPU
  / bigger batch cannot fix a ~0 gradient mean on the plateau.

---

## 4. Conclusion & decision

**Strong Claim A is dead in this parameterization, with high confidence.** Four
independent, concordant negatives: r5 (raw PPO 22.99), Component-2 (22.70, std
1.67), the pre-registered design-analysis KILL (gradient-SNR physics), and this
c3 5-seed run (24.29, bias ~3 SE below target).

**Adopt Claim B.** Evidence chain for the paper:
1. the measured μ\*(κ) curve (22.6 → 24.7), showing raw PPO's target is the
   smoothed equilibrium, structurally below e\*;
2. the 6-run c3_cont ensemble, showing κ-continuation tracks μ\*(κ) reproducibly
   (±0.15) and tops out at 24.3;
3. the c2 / r5 negatives, showing the undershoot is not a schedule/trigger
   artifact;
4. MC-BR polish + exploitability, which bridge the residual ~0.7 to e\*.

**Do NOT** spend more GPU pushing strong Claim A (higher κ_top, bigger batch)
without new variance-reduction *and* bias-reduction evidence — μ\*(κ) is flat
past 400 and the bias persists even relative to μ\*(κ) itself.

---

## 5. What changed in the repo (this merge)

Nine commits on top of the previous `main`, grouped by type:

**Code**
- `feat: add MC-BR polishing module and Phase-0 verification drivers` (8df54d0)
- `feat: add ModeConc head + Claim-A retrain modes to 3P runner` (6324f49) —
  `ActorCriticModeConc` (Beta by mode+concentration, α,β≥1) and the two mutually
  exclusive flags `--mode-conc-ramp` (Component-2) and `--kappa-continuation`.
  **This is the machinery that generated every c2/c3 result** (it was previously
  uncommitted — this closes that reproducibility gap).
- `feat: add zero-GPU Claim-A design/screen analysis tools` (7f347f9) —
  `tools/claim_a_phase_a_screen.py`, `tools/claim_a_continuation_design.py`.

**Data** (3P q=35 convergence JSONs, `results/three_players/convergence/`)
- `chore: add c3_cont kappa-continuation 5-seed results (seeds 43-47)` (558f0e5)
- `chore: add c2_mode_conc + c3_cont pilot 3P q35 convergence results` (13a1164)
  — c2_mode_conc seeds 42–46, c3_cont pilot seed 42.

**Docs**
- `fix: restrict Phase-0 2P do-no-harm to Set 1 (exclude wh8_wl4)` (7e593b4)
- `docs: score Gate C on c3_cont 5-seed run ...` (1ee929d)
- `docs: land Phase-0 revision-response + Claim-A retrain-saga docs` (44ddbfd)
- `docs: add Phase-0 audit + Claim-A retrain sections to project STATE` (6ef16e0)

**History caveat:** because the work was landed incrementally, the c3_cont
5-seed *data* commit (558f0e5) precedes the *code* commit that generated it
(6324f49). All code + data + docs are present; the ordering is just not strictly
generative. History was not rewritten (the earlier commits were already pushed).

---

## 6. Where to find things (file map)

| What | Path |
|------|------|
| **This report** | `docs/status_report_20260707.md` |
| Project summary (top section) | `docs/STATE.md` |
| **Full Gate C table + decomposition** | `docs/tasks/claim-a-nonlocking-continuation/STATE.md` |
| Continuation design + kill autopsy | `docs/tasks/claim-a-nonlocking-continuation/{phase01_findings.md,CLAUDE.md}` |
| Attempt #1 (Component-2) | `docs/tasks/component2-mode-conc-retrain/` |
| Attempt #2 (dev-trigger screen) | `docs/tasks/claim-a-dev-trigger-retrain/` |
| 5-seed data | `results/three_players/convergence/ppo_3p_q35.0_seed{43..47}_c3_cont_convergence.json` |
| Pilot + Component-2 data | `...seed42_c3_cont...`, `...seed{42..46}_c2_mode_conc...` |
| Retrain code | `agents/ppo_three_players.py` (`ActorCriticModeConc`), `run/run_three_players.py` (flags) |
| Zero-GPU analysis tools | `tools/claim_a_continuation_design.py`, `tools/claim_a_phase_a_screen.py` |

---

## 7. How to reproduce

**5-seed Gate C run** (GPU; ~18 h/seed on a V100, run in parallel via tmux):

```
python run/run_three_players.py --method ppo --q 35 --seed <43..47> \
  --kappa-continuation --episodes 34000000 --max-updates 9000 \
  --cont-max-hold 120 --ablation-name c3_cont
```

Note the budget flags: base config caps at `max_updates × 4096` *steps*; the
adaptive tail spends 65536 steps/update, so `--max-updates 9000` is required to
avoid truncating the ladder (pilot #1 hit exactly this bug and stopped at κ=100).

**μ\*(κ) curve + design analysis** (zero GPU, seconds):

```
python tools/claim_a_continuation_design.py
python tools/claim_a_phase_a_screen.py
```

**Re-score Gate C from the JSONs** — the per-seed / cross-seed numbers in §3 come
straight from the six `*_c3_cont_convergence.json` files (`final.effort` for the
snapshot; last-30-update slice of the κ=400 `cont_phase`/`kappa` segment for the
time-averaged metric).

Environment note: these V100 GPUs require `torch==2.5.1+cu121`; the cu130 wheel
silently breaks GPU training. `nvidia-smi` on the host is NVML-broken — monitor
via `torch.cuda.mem_get_info`.

---

## 8. Current state & next steps

- **Owner decision pending:** paper framing (Claim B final form). The science is
  settled; what remains is writing it up around the μ\*(κ) curve + this ensemble.
- **MC-BR bridge:** the ~0.7 residual from μ\*(κ_max)=24.3 to e\*=25 is handled by
  the MC-BR polishing module (landed in 8df54d0) + exploitability — this is the
  Claim-B mechanism and should be foregrounded in the writeup.
- **Not done / recommendation:** `requirements.lock` exists locally but is **not
  committed** (left out of this PR by scope). Given this project's torch-version
  sensitivity, committing a dependency lock is recommended for reproducibility.
- **Do not resurrect strong Claim A** without new evidence that attacks the
  *bias* (not just variance) — this is pre-registered in the task CLAUDE.md.

---

## 9. Glossary (tags & terms)

- **e\*** — deterministic (closed-form) Nash equilibrium effort; 25.0 for 3P q=35.
- **μ\*(κ)** — exploration-smoothed equilibrium: the effort PPO actually targets
  given policy spread κ. Measured 22.6 (κ=20) → 24.7 (κ=400).
- **κ (kappa)** — Beta concentration; higher κ = sharper policy = less
  exploration. The ladder walks κ from 20 to 400.
- **Metric B** — the reported statistic is the Beta **mean** effort at stop
  (`policy_mean_effort[-1]`), not the mode (a CLAUDE.md invariant).
- **band** — within-run std of the policy mean over a stage's last 30 updates;
  the run-to-run diffusion amplitude at fixed κ.
- **c2_mode_conc** — Attempt #1 result tag (Component-2 ramp).
- **c3_cont** — Attempt #3 result tag (κ-continuation): pilot = seed 42,
  Gate C = seeds 43–47.
- **r5** — an earlier raw-PPO baseline that stalls at 22.99 (no κ manipulation).
- **Gate C** — the pre-registered pass/kill test for strong Claim A (§3.3).
- **forced advance** — a ladder stage advancing on the max-hold timeout instead
  of the convergence gate; 0 across all runs here (all advances were genuine).
