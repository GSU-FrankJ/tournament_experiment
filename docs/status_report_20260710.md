# Status Report — 2026-07-10

## Multi-stage TEL-PPO — build-out and verified-equilibrium results (T = 2..5)

**Audience:** collaborators picking up the `feat/multistage-phase0` branch. This
report is self-contained — you do not need the plan doc or prior sessions to
understand what was built, what was run, and what the results mean. It covers the
whole multi-stage session: the from-scratch pipeline (theory → env → verifier →
agent → runner) and the four pre-registered, gated GPU/CPU experiments.

---

## 0. Executive summary (read this first)

We extended TEL-PPO from one-shot to **multi-stage dynamic tournaments** (horizons
T = 2, 3, 4, 5) under the **Claim-B framing** inherited from the one-stage saga
(see `docs/status_report_20260707.md`): PPO self-play does **not** learn the
deterministic equilibrium e\*; it converges to the *exploration-smoothed*
equilibrium μ\*(κ). An **independent DP best-response verifier** then *certifies*
how close the learned effort functions are to an exact Markov-perfect equilibrium
(MPE) by upper-bounding exploitability with a state-wise one-step deviation
certificate.

**All four horizons were pre-registered and gated, and all four passed:**

| T | seeds certifying | certificate (dReach/ΔW, mean) | headline |
|---|---|---|---|
| 2 | **5/5** | 0.0063 | recovers the closed-form benchmark |
| 3 | **5/5** | 0.0097 | **main result: certified MPE, no closed form** |
| 4 | **5/5** | 0.0155 | benchmark extension |
| 5 | **4/5** | 0.0190 | certificate first binds (seed 46 at 0.0321) |

The gate is a **numerical certificate**, not effort-recovery: a policy passes when
its worst reachable one-step deviation gain is ≤ 3% of the prize spread ΔW. The
T=3 result is the plan's main contribution — **a numerically certified
ε-approximate MPE for a game with no closed-form solution**. Economics come out
clean and monotone: effort rises toward the final stage, is hump-shaped in the
score gap, and total expected effort increases with the horizon.

Nothing here touches the one-stage envs/agents/runners or the existing figure
pipeline. This is an entirely new, self-contained multi-stage stack.

---

## 1. Background: game, theory, and the Claim-B framing

**The game.** Two players contest a T-stage tournament. Each stage each player
picks effort e; the sampled per-stage output is `y_i = e_i + ε_i`, ε ~ U(−q, q),
and the running score gap `d` accumulates. Only the **terminal** score decides the
winner (prize w_H) vs loser (prize w_L); each stage costs `c(e) = k·e²`. State is
Markov: `(t, d)`. Canonical parameters (`config/multi_stage_two_players.py`, the
single source of truth):

```
w_h = 6, w_l = 2  (ΔW = 4),  k = 1/3500,  c(e) = k e²,  q ∈ {45, 50, 55}
γ = 1, λ = 1  (undiscounted, terminal reward)
```

**Validity constraint (from the Phase-01 audit).** The two-stage closed form is
only a valid interior equilibrium when the stage-1 second-order condition holds,
which requires **q > q_crit = 41.83**. `q = 35` and `q = 40` are invalid (SOC
fails) and `validate()` raises on them. **All runs use q = 50.**

**Claim-B framing (binding owner decision).** The one-stage saga produced four
concordant negatives against strong Claim A ("PPO learns e\*"); the adopted story
is that PPO reaches the smoothed equilibrium μ\*(κ), which the verifier then
certifies. Multi-stage makes this *worse* (a sparse terminal reward spreads credit
over T stages), so the narrative is deliberately **not** "ê ≈ e\* recovery" — it is
"PPO proposes a smoothed candidate ê_t(d); the independent verifier certifies it is
within ε of an exact MPE." This report never claims learned effort equals the
myopic optimum.

---

## 2. What was built (Phases 01–04, the pipeline)

Every component was cross-validated against theory through **independent code
paths** before any training minute was spent.

### 2.1 Phase 01 — theory audit + config validation
Independently audited the two-stage closed-form derivation. Verdict: substantially
correct, with **four corrections** folded into an errata doc:
a stage-1 SOC that was missing the V₂\* kink term (holds iff q > q_SOC, not
unconditionally), a factor-2 bookkeeping conflict (ΔW=2 example vs ΔW=4 table), a
leftover "constant stage-2 effort" contradiction, and an uncovered zero-effort
deviation. Products: `utils/theory_multistage.py` (closed forms g1, g2(d), V₂\*,
U_eq; the q_crit family; corrected stage-1 curvature; a numerical global-deviation
scan including e=0), the canonical config with `validate()`, and
`tools/verify_two_stage_benchmark.py` (all numerical checks PASS; validity flips
exactly at q_SOC, U_eq closed form matches numerics to 4+ digits).
Audit: `docs/technical/two_stage_benchmark_audit.md`.

### 2.2 Phase 02 — multi-stage env rewrite
`envs/multi_stage_env.py`: a **from-scratch** terminal-reward game with **sampled
rewards only** (intermediate = −k·e²; terminal realizes the winner from the
accumulated sampled gap), Markov state `(t, d)`, normalized observation
`[t/T, d/(q√t)]`, configurable T, and an **exploring-starts** reset API so off-path
states receive gradient signal. `tools/verify_multi_stage_env.py` PASS: payoff =
U_eq, E[stage-2 effort] = g1, win rate = F_ξ(d), transition moments — all matched
against theory. **Carry-forward flagged here:** the one-stage GAE would misbootstrap
if p0/p1 transitions interleave — fixed in Phase 04.

### 2.3 Phase 03 — independent DP best-response verifier
`utils/dp_verifier.py`: backward-induction best response (opponent frozen at ê),
with the **terminal winning probability integrated in closed form via F_ξ(d)** —
never by interpolating the step reward. Parabolic polish on the BR argmax, grid
refinement + Richardson extrapolation. The **primary certificate is dReach** — the
BR-reachable-support Δ-sum, which upper-bounds root exploitability and excludes
unreachable off-path states (fixing a spurious stage-1 term the full-grid worst
case shows). `tools/calibrate_verifier.py` PASS: the closed form sits at the noise
floor and certifies; **five deliberately bad policies all fail** (EXP ≫ floor,
not certified); dReach ≥ EXP everywhere; Richardson stable. Crucially the verifier
imports only F_ξ/f_ξ from theory — **never the env or the agent**, so its verdict
is genuinely external.

### 2.4 Phase 04 — trajectory-aware GAE, actor-critic, PPO, runner
- **Step 1 (GAE fix):** `agents/ppo_multi_stage.py` computes advantages
  **per-trajectory** (zero terminal bootstrap), so the interleaving misbootstrap
  cannot recur. `tools/test_multi_stage_gae.py` reproduces the old bug and confirms
  the fix (γ=λ=1 ≡ Monte Carlo; ordering-independent).
- **Step 2 (agent):** `MultiStageActorCritic` (Beta mean/concentration head,
  state_dim=2), `MultiStagePPO` with a clipped update and a verifier hook. No
  theory-alignment, no opponent-lag. CPU smoke PASS.
- **Step 3 (runner):** `run/run_multi_stage.py` — validates params (q_crit) before
  training, runs vectorized self-play with exploring starts (each player's episode
  is its own trajectory), evaluates the DP verifier periodically, keeps the
  best-dReach checkpoint, and writes a convergence JSON. `step_batch` matches the
  scalar step under common random numbers to 2e-7.
- **Step 4 (gate machinery):** `utils/multi_stage_metrics.py` (recovery metrics +
  gate logic), `tools/evaluate_gate.py`, and the frozen `preregistration_T2.md` —
  **thresholds committed before the gated run** (commit e73775c).

---

## 3. The gated experiments

**Gate rule (pre-registered, same across all T).** A seed **certifies** when its
best-checkpoint `dReach/ΔW ≤ 0.03` (worst reachable one-step deviation ≤ 3% of the
prize spread). A horizon **PASSES** when ≥ 80% of seeds certify (4/5). Per the
pipeline gate in the task CLAUDE.md, **T=2 had to pass before any T≥3 compute was
spent.** All runs: q = 50, seeds 42–46, entropy 0.005.

### 3.1 T=2 — closed-form recovery + verifier calibration (GATE PASS 5/5)
`2000 upd × 512 ep`, GPU, ~30 min/seed. **5/5 certify**, dReach/ΔW mean **0.0063**
(max 0.0078 ≪ 0.03), EXP/ΔW mean 0.0016. Recovery metrics also clear their targets:
RE₁ 0.045 (< 0.10), RPE₂ᶜᵒʳᵉ 0.052 (< 0.15). The recovered stage-2 curve is
near-symmetric (seed 42 e₂ = [7.4, 34.6, 64.0, 31.5, 10.1] vs closed form
[0, 35, 70, 35, 0]); the peak 64 vs 70 is residual μ\*(κ) smoothing. The
finite-sample stage-2 asymmetry seen in the CPU pilot did not survive the gated run.
Per pre-registration, this **PASS authorized the T=3 spend.**

### 3.2 T=3 — verified equilibrium, no closed form (GATE PASS 5/5) — MAIN RESULT
`3000 upd × 512 ep`, CPU, ~25 min/seed. **5/5 certify**, dReach/ΔW mean **0.0097**
(max 0.0144, std 0.0026), EXP/ΔW mean 0.0027.

| seed | EXP/ΔW | EXP^UCB/ΔW | dReach/ΔW | cert | ckpt |
|---|---|---|---|---|---|
| 42 | 0.0030 | 0.0031 | 0.0097 | yes | u1050 |
| 43 | 0.0035 | 0.0036 | 0.0144 | yes | u2100 |
| 44 | 0.0023 | 0.0024 | 0.0084 | yes | u900 |
| 45 | 0.0020 | 0.0021 | 0.0065 | yes | u1800 |
| 46 | 0.0026 | 0.0027 | 0.0096 | yes | u2550 |

This is a **numerically-certified ε-approximate MPE for a 3-stage tournament with
no closed-form benchmark** — the DP verifier is the sole certifier, and it is
independent of the training path. This is the plan's headline contribution.

### 3.3 T=4 / T=5 — benchmark extensions (T=4 PASS 5/5, T=5 PASS 4/5)
Pre-registered in `preregistration_T4_T5.md` (frozen 80ee48c before the runs); same
verifier/threshold, grid-stable (<0.3% EXP change 201→401). Budget `1000·T upd ×
512 ep`, CPU (~45 min T=4, ~75 min T=5).
- **T=4: PASS 5/5.** dReach/ΔW mean 0.0155 (max 0.0217), EXP/ΔW 0.0044.
- **T=5: PASS 4/5** (exactly the 80% line). dReach/ΔW mean 0.0190 (max 0.0321).
  Seed 46 is the lone non-certifier at 0.0321 — just over 0.03, while its EXP/ΔW is
  still tiny (0.0062). **T=5 is where the conservative reachable-Δ certificate first
  binds** — the anticipated "certification degrades at larger horizons." Reported
  with numerical caveats, not treated as a failure.

---

## 4. Results: certificate scaling and economics

### 4.1 Multi-stage summary (cross-seed mean; plan Table 4)

| T | certify | dReach/ΔW (max) | EXP/ΔW | total effort | e_hat_t(0) per stage |
|---|---|---|---|---|---|
| 2 | 5/5 | 0.0063 (0.0078) | 0.0016 | 93.3\* | ~[46.7, 46.7]\* (recovers CF) |
| 3 | 5/5 | 0.0097 (0.0144) | 0.0027 | 108.1 | [43.3, 50.9, 64.8] |
| 4 | 5/5 | 0.0155 (0.0217) | 0.0044 | 116.8 | [40.3, 42.6, 50.2, 64.0] |
| 5 | 4/5 | 0.0190 (0.0321) | 0.0051 | 128.0 | [38.9, 39.4, 41.9, 47.4, 59.8] |

\* T=2 total effort = 2·g1 (analytic recovered value); the T=2 run predates the
`effort_curves`/`onpath_summary` JSON fields.

### 4.2 Main questions (plan) — all answered YES
- **Q1 — effort rises toward the final stage:** YES at every T (e_hat_t(0)
  increasing in t within each row; e.g. T=3: 43.3 → 50.9 → 64.8).
- **Q3 — hump-shaped in the score gap:** YES at every stage/horizon (peak at d=0,
  low far ahead/behind; later stages more gap-sensitive).
- **Q4 — total expected effort increases with T:** YES, monotone
  93.3 → 108.1 → 116.8 → 128.0.
- **Leader/follower asymmetry** appears at intermediate stages (a player *behind*
  exerts less than one equally *ahead* — discouragement-when-behind / lead-defense);
  legitimate for T≥3, since only the final stage is symmetric by the myopic argument.
- **As T grows, early-stage effort falls** (T=5 stage-1 38.9 vs T=3 stage-1 43.3):
  more remaining stages ⇒ more chance to catch up ⇒ less early effort.
- **Final-stage effort ~60–64** across T (vs the myopic ~70), the same residual
  μ\*(κ) smoothing seen at T=2 — expected under Claim-B.

### 4.3 Falsification & robustness (Table 3)
The verifier cleanly separates equilibrium from non-equilibrium (T=2, EXP/ΔW):
closed form 0.000, TEL-PPO learned 0.0016, vs constant-low 0.203, constant-high
0.929, one-stage-repeated 0.222, no-gap-stage-2 0.092. Grid refinement is stable
(EXP 0.01337 → 0.01217 → 0.01219 across M = 51/101/201; Richardson 0.01220).
Cross-seed std of dReach/ΔW grows gently with T (0.0010 → 0.0026 → 0.0038 → 0.0066).

---

## 5. Figures & tables (plan section 6)

Generated by `tools/make_multistage_figures.py` and `tools/make_multistage_tables.py`
from the committed JSONs into `paper/multistage/` (PDFs + `.tex` committed; PNG
previews gitignored, regenerable).

| file | content | data |
|---|---|---|
| `F1_two_stage_recovery` | closed-form vs TEL-PPO stage-2 effort | T=2 |
| `F2_verifier_calibration` | EXP of closed-form / TEL-PPO / bad policies | T=2 |
| **`F3_three_stage_effort`** | learned e_hat_t(d), t=1,2,3 (**main figure**) | T=3 |
| `F4_three_stage_br_vs_learned` | best response vs learned per stage | T=3 |
| `F5_three_stage_deviation_gaps` | one-step deviation gaps Δ_t(d) | T=3 |

Tables: `table1` (T=2 recovery + exploitability), `table2` (T=3 per-seed
certificate), `table3` (grid/seed robustness + falsification), `table4` (T=2–5
summary).

**Two caveats baked into the artifacts** (`paper/multistage/README.md`):
1. **F1 uses a dense single-seed re-run** (`ms_T2_q50_seed42_densecurve`, same
   frozen protocol; dReach/ΔW = 0.0051, RE₁ = 0.015) because the 5-seed gated T=2
   runs predate the `effort_curves` field. The 5-seed recovery robustness lives in
   Table 1.
2. **Table 4's T=2 total effort/cost are analytic recovered values** (2·g1 and
   (W_H+W_L)/2 − U_eq); T=3/4/5 use the on-path summary from the saved curves.

---

## 6. Conclusion & decision

**The multi-stage pipeline is built and validated end to end.** From a from-scratch
env + independent DP verifier + trajectory-aware PPO, we produced **certified
ε-approximate MPE across four horizons** — including T=3, the first horizon with no
closed-form benchmark (the plan's main contribution). The certificate degrades
monotonically and predictably with T, binding for the first time at T=5. The
economics are clean, monotone, and match the plan's questions.

**Claim-B holds and is reinforced in the dynamic setting.** Learned effort is a
smoothed candidate (final-stage peak ~60–64 vs myopic ~70); the verifier certifies
that this smoothed policy is within ε of an exact MPE. We do **not** claim ê = e\*.

---

## 7. What changed in the repo (this branch)

18 commits on `feat/multistage-phase0` on top of `main`, grouped by type.

**Code**
- `feat: add two-stage benchmark theory module with q_crit config validation` (d38e08c)
- `feat: add multi-stage tournament env (terminal reward, sampled outcomes)` (33394f2)
- `feat: add independent DP best-response verifier with reachable-set certificate` (ddce466)
- `fix: add trajectory-aware GAE for multi-stage rollouts with unit test` (197d67e)
- `feat: add multi-stage Beta actor-critic and PPO update` (5d2cb86)
- `feat: add multi-stage training runner with vectorized self-play rollout` (3cd0c65)
- `feat: add pre-registered T=2 gate, recovery metrics, and gate aggregation` (e73775c)
- `feat: generalize multi-stage runner to T>=3 with per-stage curves` (b35005e)
- `feat: add on-path expected effort/cost summary to multi-stage runner` (054e42d)
- `feat: add multi-stage figure and table generators (plan section 6)` (04cac81)

**Data** (`results/multi_stage/convergence/`)
- `results: T=2 gated multi-stage run (5 seeds) — gate PASSED` (a7cce2e)
- `results: T=3 verified-equilibrium gated run (5 seeds) — gate PASSED` (de18377)
- `results: T=4 and T=5 benchmark-extension gated runs (5 seeds each)` (c98e371)
- `results: dense-curve T=2 re-run (seed 42) for Figure 1` (260828a)

**Docs / artifacts**
- task audit + pre-registrations + phase logs (c514baa, 1449b77, 80ee48c, 322136c,
  45ae0b2, 014af4c, 8cc571b, and the phase step-completion docs)
- `docs: add generated multi-stage figures (PDF) and tables (LaTeX)` (e1b9d12),
  `docs: mark multi-stage figures/tables delivered in phase05` (567dc6e),
  `docs: use dense curve for Figure 1 (two-stage recovery)` (d8a33c2)

**Pre-registration discipline:** every gate's thresholds were committed *before*
its gated run (T=2 in e73775c; T=3 in 1449b77; T=4/T=5 in 80ee48c).

---

## 8. Where to find things (file map)

| What | Path |
|------|------|
| **This report** | `docs/status_report_20260710.md` |
| Task summary (authoritative) | `docs/tasks/multistage-tel-ppo/STATE.md` |
| Task decisions / scope | `docs/tasks/multistage-tel-ppo/CLAUDE.md` |
| Phase logs | `docs/tasks/multistage-tel-ppo/phase0{1..5}.md` |
| Pre-registrations (frozen) | `.../preregistration_T2.md`, `_T3.md`, `_T4_T5.md` |
| Benchmark derivation audit | `docs/technical/two_stage_benchmark_audit.md` |
| Theory / q_crit / validation | `utils/theory_multistage.py` |
| Canonical config | `config/multi_stage_two_players.py` |
| Env | `envs/multi_stage_env.py` |
| **Independent DP verifier** | `utils/dp_verifier.py` |
| Agent (GAE + actor-critic + PPO) | `agents/ppo_multi_stage.py` |
| Runner | `run/run_multi_stage.py` |
| Metrics / gate | `utils/multi_stage_metrics.py`, `tools/evaluate_gate.py` |
| Verification tools | `tools/verify_two_stage_benchmark.py`, `verify_multi_stage_env.py`, `calibrate_verifier.py`, `test_multi_stage_gae.py`, `smoke_multi_stage_ppo.py` |
| Result JSONs | `results/multi_stage/convergence/ms_T{2..5}_q50_seed{42..46}_gate*_convergence.json` |
| Figures / tables | `paper/multistage/figures/`, `paper/multistage/tables/`, `paper/multistage/README.md` |

---

## 9. How to reproduce

**Gated runs** (q = 50, seeds 42–46; run in tmux):

```bash
# T=2 (GPU ~30 min/seed) — must PASS before T>=3 compute
python run/run_multi_stage.py --T 2 --q 50 --seed <42..46> \
  --updates 2000 --episodes 512 --entropy-coef 0.005 --tag gateT2

# T=3 (CPU ~25 min/seed) — main result, no closed form
python run/run_multi_stage.py --T 3 --q 50 --seed <42..46> \
  --updates 3000 --episodes 512 --entropy-coef 0.005 --tag gateT3

# T=4 / T=5 (CPU ~45 / ~75 min/seed) — budget 1000*T updates
python run/run_multi_stage.py --T <4|5> --q 50 --seed <42..46> \
  --updates <4000|5000> --episodes 512 --entropy-coef 0.005 --tag gate<T4|T5>

# Score a gate from the JSONs
python tools/evaluate_gate.py   # reads results/multi_stage/convergence/*

# Regenerate figures + tables
python tools/make_multistage_figures.py
python tools/make_multistage_tables.py
```

Flags verified against `run/run_multi_stage.py` argparse. The `--tag` value becomes
the JSON suffix (`ms_T{T}_q{q}_seed{seed}_{tag}_convergence.json`). Other exposed
knobs: `--eval-every` (default 100), `--lr` (3e-4), `--gae-lambda` (1.0),
`--es-on-path-fraction`, `--device`. Note `--entropy-coef` defaults to 0.01; the
gated runs used **0.005**.

**Perf note:** the tiny 64-hidden net is CPU-bound; `--device cuda` is no faster
than CPU (act_batch torch↔numpy round-trips dominate). T=3–5 were run on CPU. If a
V100 is used elsewhere, note the `torch==2.5.1+cu121` requirement (the cu130 wheel
silently breaks GPU training; see the memory note).

---

## 10. Current state & next steps

**Done:** Phases 01–05 complete. Full pipeline built, cross-validated, and gated;
T=2/3/4 PASS 5/5, T=5 PASS 4/5; figures 1–5 and tables 1–4 generated.

**Deferred (none started):**
- **Curriculum ablation** (T=1→2→3 checkpoint transfer) and **adversarial-RL BR
  cross-check** (plan 5.4) — need checkpoint-transfer plumbing / a second BR method.
- Optional: optimize the CPU-bound rollout; re-run T=2/T=3 to backfill
  `onpath_summary`/`effort_curves` so Table 4's T=2 row is uniform (it currently
  uses the analytic recovered total effort).

**Open items / caveats:**
- The plan Word doc still carries the 5 benchmark-derivation errata (in the audit
  doc's "Errata" section) — owner to fold back into the source.
- Commit `8e9433e` on main has a mismatched message ("Update print statement…") for
  a plan-doc edit — noted, history not rewritten.
- `requirements.lock` exists locally but is untracked (`??` in git status); given
  this project's torch sensitivity, committing a dependency lock is recommended.

---

## 11. Glossary

- **ê_t(d)** — the learned (Beta-mean) effort function at stage t as a function of
  the running score gap d. The candidate the verifier certifies.
- **MPE** — Markov-perfect equilibrium; the multi-stage analogue of Nash.
- **EXP** — exploitability: best-response payoff gain over the incumbent policy.
- **Δ_t(d)** — one-step deviation gap at state (t, d); the state-wise certificate.
  `EXP ≤ Σ_t max_d Δ_t(d)`.
- **dReach** — the primary certificate: the Δ-sum over the **BR-reachable** support
  only (excludes unreachable off-path states; upper-bounds root EXP). **dFull** is
  the full-grid worst case (loose, includes off-path); reported alongside.
- **ΔW** — prize spread w_H − w_L = 4; all certificates are normalized by it.
- **certify / gate** — a seed certifies when dReach/ΔW ≤ 0.03; a horizon PASSES
  when ≥ 4/5 seeds certify.
- **μ\*(κ)** — the exploration-smoothed equilibrium (κ = Beta concentration); the
  effort PPO actually targets, structurally below the myopic optimum. Inherited
  from the one-stage saga (`docs/status_report_20260707.md`).
- **q_crit = 41.83** — stage-1 SOC threshold; q ≤ q_crit is an invalid benchmark
  (all runs use q = 50).
- **RE₁ / RPE₂ᶜᵒʳᵉ** — T=2 recovery metrics (stage-1 relative effort error /
  stage-2 core relative profile error); only meaningful where a closed form exists.
- **exploring starts** — episodes reset from random (t, d) so off-path states get
  gradient signal, preserving the full-MPE claim.
</content>
</invoke>
