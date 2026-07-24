# ONE_STAGE_ASBUILT.md — As-Built Audit of the One-Stage TEL-PPO Pipeline

**Scope:** read-only audit of the *current* one-stage (two-player, single-shot)
TEL-PPO code against the published/drafted framework figure (components F1–F9),
plus a fresh mode-vs-mean measurement. No repo edits, no retraining. Every number
below comes from an actual measurement in this session or a repo file cited by
`path:line`; anything not measured is marked **MISSING**.

---

## 1. Provenance block

| Item | Value |
|---|---|
| Server | `vector2` (V100 / sm_70; torch 2.5.1+cu121 per `requirements.lock`) |
| Branch | `feat/multistage-phase0` |
| HEAD at audit **start** | `168884d` (`docs: move two-stage report from results/ to docs/`) |
| HEAD at audit **end** | `d3eeea9` (`fix: correct two-stage report relative links after move to docs/`) |
| Note | HEAD advanced 3 commits *during* this session — a concurrent/owner commit landed the one-stage Claim-B summary (`5841338`, `ea48f61`) + a link fix (`d3eeea9`). Read-only audit → analysis unaffected; the Claim-B CSV read below is the committed version. Still-untracked: `tools/mu_star_screen.py`, `docs/mu_star_screen.json`. |
| Canonical one-stage variant | **PPO + `ActorCritic` (softplus+1 head) + `TwoPlayersEnv` (sampled rewards)**, tag `r5_sampled`, config `config/one_stage_two_players.py`. Per `SESSION_STATE.md` ("Adopted canonical state"): every (experiment,q) → `r5_sampled`. |
| Canonical checkpoint (`.pt`) | **NONE EXISTS.** `run_two_players.py` never persists a policy network; `find results/two_players -name '*.pt'` → empty (only `multi_stage` saves `.pt`). The recoverable policy state is the `alpha_mean`/`beta_mean` time-series inside the convergence JSONs. |
| Part-B data source | `results/two_players/convergence/ppo_q{35,55}.0_seed{42..46}_r5_sampled_convergence.json` (10 files, committed `Jul 1`). Training provenance: 6,144,000 episodes budget, 4096 steps/update, stop = exploitability streak. |

### Variant enumeration (canonical vs. others)

| Variant | Where | Status |
|---|---|---|
| `ActorCritic` (softplus+1 α,β heads) | `agents/ppo_two_players_clean.py:27-59` | **CANONICAL** — used by all `r5_sampled` Set-1 runs. |
| `ActorCriticMeanConc` (mean·conc, `theory_align_v2`) | `agents/ppo_two_players_clean.py:62-111` | **EXPERIMENTAL / non-canonical.** Off by default (`theory_align_v2=False`). The `_v2` arm was **REJECTED** for het-ability (SESSION_STATE: equal-or-worse error, 3–14× seed variance); kept on disk as `r5_sampled_v2`, not a baseline. |
| `theory_align` v1 (conc-min rescale) | `agents/ppo_two_players_clean.py:53-56` | Dormant; `theory_align=False` by default. |
| `TwoPlayersEnv` (sampled rank rewards) | `envs/two_players_env.py:17-76` | **CANONICAL** — the only env `run_two_players.py` imports (`run_two_players.py:59`). |
| `OneStageEnv` (closed-form `probability_uniform` blend) | `envs/one_stage_env.py:4-72` | **LEGACY / DEPRECATED.** Trains on a hand-tuned closed-form win-prob (the exact "closed-form training" the audit abandoned). `grep OneStageEnv` across `run/ agents/ tools/ paper/` → **zero** importers. Superseded by the sampled rewrite. |

---

## 2. Side-by-side audit table (figure component → as-built → file:line → verdict)

| # | Figure says | As-built behavior | Evidence (`file:line`) | Verdict |
|---|---|---|---|---|
| **F1** | State `s = (w_H, w_L, k, q)` | State is a **3-vector, normalized, constant per experiment**: `[q/60, k/1e-3, (w_H−w_L)/10]`. Prize levels `w_H,w_L` are compressed to a single **gap** feature; there is no per-episode state variation. | `agents/ppo_two_players_clean.py:675-681` | **CHANGED** |
| **F2** | Two **independent** policy networks (Agent 1 / Agent 2) | **One shared `ActorCritic`.** Both players sample from the same `self.net`; both transitions are stored (pure self-play). A lagged `opponent_policy` copy exists but is **vestigial** — `selfplay` rollout never calls `act_opponent()` (opponent-lag config all set to 0/disabled). | `agents/ppo_two_players_clean.py:171,183,234-246`; `config/one_stage_two_players.py:31-39` | **CHANGED** |
| **F3** | Beta policy `π_θ(a\|s)` with heads α_θ(s), β_θ(s) | Exactly this: two linear heads → `α = softplus(·)+1`, `β = softplus(·)+1` → `Beta(α,β)`. | `agents/ppo_two_players_clean.py:34-43,58` | **MATCH** |
| **F4** | Affine map `e_i = a_i(e_max−e_min)+e_min` | `effort = low + a·(high−low)`, bounds `(0,100)`. | `agents/ppo_two_players_clean.py:231`; `config/one_stage_two_players.py:17` | **MATCH** |
| **F5** | `y_i = e_i + ε_i`; rank-based payoff | `ε_i ~ U(−q,q)`, `y_i=e_i+ε_i`, argmax winner → `w_H`, loser → `w_L`, minus `k·e_i²`. Single realized outcome (K=1). SAMPLED reward. | `envs/two_players_env.py:54-76` | **MATCH** |
| **F6** | PPO, clipped objective, **one-step** advantage | Clipped PPO surrogate present. Advantage is **GAE(γ=0.99, λ=0.95)**, but episodes are single-step bandit (`done=True` every step) so GAE **collapses to `r − V(s)`** — one-step in effect. | `agents/ppo_two_players_clean.py:342-358,498-503`; `envs/two_players_env.py:76` | **MATCH (in effect)** |
| **F7** | KL-window / drift threshold; "continue self-play" loop | `CheapGateTracker` computes `mean_kl_window / std_kl_window / drift_effort`, but these do **not** stop training — they **gate when the exploitability check runs**. Periodic eval (every 10 updates) bypasses the gate entirely. Under the canonical `relaxed` profile. | `run_two_players.py:162-194,1340-1376`; `config/one_stage_two_players.py:66-98` | **CHANGED (repurposed as a gate, not a stop criterion)** |
| **F8** | Exploitability / ε-equilibrium; else continue | Monte-Carlo best-response: `M=8192` CRN samples, coarse-to-fine grid; `exploitability = max_e′ E[u(e′,π)] − E[u(π,π)]`. Stop when `exploitability < 0.03` for `patience=5` consecutive evals. This is the **sole** stop. | `run_two_players.py:266-359,1385-1439`; `config/one_stage_two_players.py:107-118` | **MATCH (but MC; predates two-stage quadrature-UCB)** |
| **F9** | Output: effort profile + (optional) symmetry `\|ē₁−ē₂\|` | Reported `final.effort` = **Beta MEAN** of the final policy mapped to effort. `symmetry_gap = \|ē₁−ē₂\|` is computed but **logging-only, vacuous** (one shared net → gap is pure sampling noise). No mode extracted. | `run_two_players.py:1664-1684,1364-1368` | **MATCH (mean); symmetry diagnostic → vacuous** |

---

## 3. Answers Q1–Q8 (with evidence)

**Q1 (F2/F9) — Two nets or shared? Is |ē₁−ē₂| still meaningful?**
**Single shared policy.** One `ActorCritic` (`ppo_two_players_clean.py:171`); both players draw from it and both transitions are stored (docstring lines 10-12). The lagged opponent net is never used in `selfplay` (config `opponent_sync_interval=0`, etc., `one_stage_two_players.py:31-39`). The symmetry diagnostic **is still computed** (`run_two_players.py:1367`, reported as `Symmetry Gap` = 0.10/0.04 for q35/q55 in the Claim-B CSV) **but is vacuous** — the code itself says so: *"Both players share one policy network, so any p1/p2 gap is pure sampling variance — not a real divergence signal"* (`run_two_players.py:1364-1368`). It measures Monte-Carlo rollout noise between two draws of one policy, not architectural asymmetry.

**Q2 (F3) — How are α, β produced? Are α,β > 1 guaranteed?**
`α = softplus(head_α) + 1.0`, `β = softplus(head_β) + 1.0` (`ppo_two_players_clean.py:40-41`). softplus ∈ (0,∞) so the `+1.0` floor makes **α > 1 and β > 1 structurally guaranteed** for the canonical head. No explicit clamp on the canonical path (the `theory_align`/`v2` rescales at lines 53-56 are OFF by default). **⇒ the Beta mode `(α−1)/(α+β−2)` is well-defined and interior everywhere.** Measured α,β ∈ [65,136] (Part B) — nowhere near the α≤1 boundary. (The experimental `ActorCriticMeanConc` uses `α = mean·conc`, `β=(1−mean)·conc` with `conc_min≥1`, lines 100-104 — a different guarantee, not used canonically.)

**Q3 (F9) — Which deterministic effort is extracted/reported?**
The **Beta MEAN**: `dist.mean = α/(α+β)`, mapped by `low + mean·(high−low)`. Extraction point:
`run_two_players.py:1664-1668` → `compute_policy_mean_effort(α,β,low,high)` (`utils/rollout_stats.py:241`, mean = `α/(α+β)`) → `final.effort` (line 1682). Recomputed on the **final policy at stop** (no best-checkpoint selection; `stopped_at_update = update_idx`). **No mode, no argmax-grid extraction exists in the one-stage runner.**

**Q4 (F6) — One-step advantage? γ, entropy, lr, clip, epochs, batch (canonical config)?**
Advantage: GAE that collapses to one-step `r−V` (see F6). Canonical values from `config/one_stage_two_players.py`, corroborated by the JSON (`steps_per_update`, `episodes`, `exploit_config`, eval cadence all match the run):

| Param | Value | Source |
|---|---|---|
| γ (gamma) | 0.99 | `ppo_two_players_clean.py:116` (single-step ⇒ inert) |
| GAE λ | 0.95 | `:117` (single-step ⇒ inert) |
| entropy_coef | **0.03 → 0.005** (held to ~2/3 of updates, then annealed) | `one_stage_two_players.py:49-51`; schedule `run_two_players.py:729-739,898-902` |
| learning rate | **3e-4 → 2e-4** (annealed) | `:52-53` |
| clip ε | **0.50 → 0.35** (annealed) | `:54-55` |
| PPO epochs | **6** | `:44` |
| minibatch | **1024** | `:43` |
| steps/update (batch) | **4096** | `:42` (JSON: last step 196608 = 48×4096 ✓) |
| value_coef | 0.5 | `ppo_two_players_clean.py:120` |
| max_grad_norm | 0.5 | `:122` |
| hidden width | 128 | `run_two_players.py:684` |
| target_kl | 0.08 | `:56` |

**Q5 (F7) — Is KL-window/drift stability screening implemented & active? Thresholds?**
**Implemented and active, but as a *gate*, not a stop.** `CheapGateTracker` (window 20) computes `mean_kl_window, std_kl_window, drift_effort` (`run_two_players.py:162-194`). Canonical `relaxed` profile: `mean_kl ≤ 0.015`, `std_kl ≤ 0.012`, `drift_effort ≤ 8.0`, `patience_drift = 1` (`one_stage_two_players.py:92-98`). When the gate passes for `patience_drift` evals it *triggers an exploitability evaluation* (`:1376`), and a failed exploitability check resets the streak (`:1429-1430`). It never sets `stop_reason` by itself, and periodic eval every 10 updates fires regardless (`:1375,1383`). So the figure's "KL window ⇒ continue loop" is **not** the convergence criterion; it only schedules the F8 check.

**Q6 (F8) — How is exploitability computed? Consistent with two-stage conventions?**
**Monte-Carlo best-response, stochastic** (`eval_exploitability`, `run_two_players.py:266-359`): draw `M=8192` CRN noise samples `ε~U(−q,q)` and `M` policy-effort samples; scan a **coarse-to-fine grid** (step 5.0 → 1.0 within ±15 → 0.25 within ±3) for the best deviation; `exploitability = best_dev_payoff − u_selfplay`. Criterion `< 0.03`, patience 5 (`:1425-1431`). **Not consistent with the new two-stage conventions.** Two-stage uses deterministic Gauss-Legendre quadrature over the triangular ξ and a refinement-based `dReach_UCB` (`SESSION_STATE.md` Phase-1 T1/T2; `utils/dp_verifier.py:certify_refined`). One-stage **predates** those: it is sampled MC with CRN (deterministic *given the seed*, but no quadrature and no discretization UCB bound).

**Q7 (whole pipeline) — Everything after PPO training that the figure omits.**
Within `run_two_players.py`, **nothing** post-processes the effort: `grep polish|mc_br|refine|argmax` over the runner → empty; `final.effort` is the raw mean of the final policy (Q3). **The reported one-stage number is pure-PPO output.** The steps the figure omits are all *inside* the training loop or *outside* the runner:

1. **Cheap gate scheduling** (in-loop) — `run_two_players.py:1340-1376`; decides *when* F8 runs; does **not** modify effort.
2. **MC exploitability eval** (in-loop, every ≤10 updates) — `:1385-1412`; produces the certificate; does **not** modify effort.
3. **Final effort recomputation at stop** — `:1664-1684`; mean of the final policy; **this is the reported number**.
4. **External MC-BR polishing (NOT in this runner)** — `utils/mc_br_polish.py` + `tools/phase0_verify.py`. A separate, zero-GPU post-hoc global best-response solver that moves the raw landing to the certified sampled equilibrium. It **is** reported, as a **separate column** in `results/one_stage_claimb_summary.csv` (`Polished`): 2P q35 43.58 → **44.95**; q55 29.65 → **28.76**. Per SESSION_STATE Phase-0 §E, polish is **LOAD-BEARING** for the paper's Claim B; raw PPO and polished are kept as independent columns.

**Framing consequence:** the final *reported effort* is **PPO + post-processing depending on which column you cite** — raw-PPO-mean for the "TEL-PPO" convergence number, but the ≤1.3%-error "verified equilibrium" number is **PPO-to-basin + MC-BR polish** (Claim B). The paper must not present the polished number as raw PPO self-convergence.

**Q8 (ground truth) — Closed-form benchmark, formula, configured params, target.**
Formula in code: `e_star_two_players(q,w_H,w_L,k) = (w_H−w_L)/(4·q·k)` (`utils/theory.py:38`), the denominator-4 two-player one-stage equilibrium — consistent with `e*_CF = ΔW·f(0)/(2k) = ΔW/(4qk)` for `ε~U[−q,q]` (ξ triangular on [−2q,2q], `f(0)=1/(2q)`). **Configured params (`config/one_stage_two_players.py:8-13`): w_H=6.5, w_L=3.0, k=0.00055, q∈{35,45,55}, bounds [0,100].**

> ⚠️ The prompt's guessed params (W_H=6, W_L=2, k=1/3500, q=50 → target 70.0) are **NOT** what the repo uses. The actual one-stage targets are:
> - **q=35 → e* = 3.5/(4·35·0.00055) = 45.4545**
> - **q=55 → e* = 3.5/(4·55·0.00055) = 28.9256**
> (q=45 → 35.3535.) These match `config['effort']` (`:139`) and the Theory rows of the Claim-B CSV.

---

## 4. As-built pipeline in execution order

1. **Resolve state** `s = [q/60, k/1e-3, (w_H−w_L)/10]` — constant for the run (`ppo_two_players_clean.py:675-681`).
2. **Build one shared `ActorCritic`** (softplus+1 α,β heads + value head), Adam lr=3e-4 (`:171-172`, `run_two_players.py:679-711`).
3. **Rollout (self-play, K=1):** for each of 4096 steps, sample two actions from the same policy, map to efforts, `TwoPlayersEnv.step` returns **sampled** rank rewards `w_H/w_L − k e²`; store **both** transitions (`two_players_env.py:54-76`).
4. **PPO update:** GAE(0.99,0.95)→r−V; clipped surrogate; 6 epochs × minibatch 1024; entropy/lr/clip on annealing schedules; grad-clip 0.5 (`ppo_two_players_clean.py:361-672`).
5. **Log diagnostics:** α_mean, β_mean, policy-mean effort, sampled p1/p2 efforts, `symmetry_gap` (logging-only), KL, drift (`run_two_players.py:1124-1204`).
6. **Cheap gate (KL/drift):** update rolling window; if it passes `patience_drift` in a row → eligible to run F8 (`:1340-1376`).
7. **Exploitability (F8):** every ≤10 updates OR when gated, run MC grid best-response (M=8192, CRN); if `< 0.03` increment streak (`:1385-1428`).
8. **Stop:** when exploitability streak ≥ 5 → `stop_reason="exploitability"`, break (`:1431-1439`). Else exhaust the 1500-update / 6.144M-episode budget → `stop_reason="max_updates"`.
9. **Report:** recompute **Beta MEAN** effort of the final policy → `final.effort`; dump convergence JSON (α/β series, efforts, exploitability, stop info). **No `.pt` saved. No in-runner polishing.** (`:1660-1739`).
10. *(Separate, post-hoc, outside the runner)* **MC-BR polish** → `Polished` column of the Claim-B summary; verified by fresh-seed exploitability + FOC legs (Phase-0).

---

## 5. Part B — Mode-vs-mean measurement

**Method.** For each canonical run I read the **final** `(alpha_mean, beta_mean)` snapshot from the convergence JSON (batch-mean over a size-1 constant state ⇒ the exact network α,β), and compute
`e_mean = 100·α/(α+β)`, `e_mode = 100·(α−1)/(α+β−2)`, Beta per-draw SD `= 100·√(αβ / ((α+β)²(α+β+1)))`. Cross-check: `e_mean` reproduces the JSON `final.effort` and the committed Claim-B CSV raw column bit-for-bit (q35 agg 43.58, q55 agg 29.65). α,β > 1 everywhere ⇒ mode interior/well-defined (no boundary cases).

### 5a. Per-seed table

**q = 35 (e\* = 45.4545):**

| seed | α | β | α+β | per-draw SD | MEAN | \|err\| | rel% | MODE | \|err\| | rel% | mode−mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 79.80 | 104.26 | 184.1 | 3.64 | 43.357 | 2.098 | 4.62 | 43.284 | 2.171 | 4.78 | −0.073 |
| 43 | 102.58 | 135.97 | 238.5 | 3.20 | 43.000 | 2.455 | 5.40 | 42.941 | 2.514 | 5.53 | −0.059 |
| 44 | 94.56 | 130.75 | 225.3 | 3.28 | 41.968 | 3.486 | 7.67 | 41.896 | 3.558 | 7.83 | −0.072 |
| 45 | 102.55 | 124.39 | 226.9 | 3.30 | 45.188 | 0.266 | 0.59 | 45.146 | 0.309 | 0.68 | −0.043 |
| 46 | 98.75 | 123.73 | 222.5 | 3.32 | 44.386 | 1.069 | 2.35 | 44.335 | 1.119 | 2.46 | −0.051 |
| **agg** | | | **219.5** | **3.35** | **43.580** | **1.875 (4.12%)** | | **43.520** | **1.934 (4.26%)** | | **−0.060** |

**q = 55 (e\* = 28.9256):**

| seed | α | β | α+β | per-draw SD | MEAN | \|err\| | rel% | MODE | \|err\| | rel% | mode−mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 78.92 | 190.45 | 269.4 | 2.77 | 29.298 | 0.372 | 1.29 | 29.143 | 0.217 | 0.75 | −0.155 |
| 43 | 64.83 | 160.55 | 225.4 | 3.01 | 28.765 | 0.161 | 0.56 | 28.575 | 0.351 | 1.21 | −0.190 |
| 44 | 73.29 | 167.07 | 240.4 | 2.96 | 30.493 | 1.567 | 5.42 | 30.329 | 1.403 | 4.85 | −0.164 |
| 45 | 78.28 | 188.84 | 267.1 | 2.78 | 29.305 | 0.379 | 1.31 | 29.148 | 0.223 | 0.77 | −0.156 |
| 46 | 76.99 | 176.18 | 253.2 | 2.89 | 30.412 | 1.486 | 5.14 | 30.256 | 1.330 | 4.60 | −0.156 |
| **agg** | | | **251.1** | **2.88** | **29.654** | **0.729 (2.52%)** | | **29.490** | **0.565 (1.95%)** | | **−0.164** |

Per-agent / symmetry: game is symmetric with one shared net; `|ē₁−ē₂|` = 0.10 (q35) / 0.04 (q55) — pure sampling variance (Q1).

### 5b. Policy concentration

- **α+β ≈ 220 (q35) / 251 (q55)** — moderately concentrated, **not** a razor-thin spike. Per-draw effort SD ≈ **3.3 / 2.9 units** on the [0,100] range, i.e. the stochastic policy still has real width.
- Despite that width, **MEAN and MODE coincide to within 0.06 (q35) / 0.16 (q55) effort units** (≈0.14% / 0.55% of effort). Both are point functionals of the same (α,β); the tiny gap is the mild right-skew (β>α ⇒ mode < mean).
- **Scale check:** mode−mean gap (0.06–0.16 u) ≪ cross-seed std of the mean (**1.12 / 0.68 u**) ≪ mean-vs-e\* bias (**1.88 / 0.73 u**). The extraction choice is 10–30× smaller than either the seed noise or the systematic bias.

> Note on the "α+β ≈ 25k–33k near-spike" phrase in `SESSION_STATE.md`: that refers to the **experimental `ActorCriticMeanConc` mode-conc retrain** (3P, κ-ramp), *not* the canonical `r5_sampled` `ActorCritic`, which lands at α+β ≈ 220–250.

### 5c. Reconciliation with prior findings

- **SESSION_STATE §E comp (3) "mode extraction INERT; L1 mode Δ ≈ −0.00" (3P):** **AGREE.** My 2P mode−mean = −0.06…−0.16 u (≈0.1–0.5%) rounds to "≈0" at reporting precision. Same conclusion: mode ≈ mean.
- **SESSION_STATE Phase-2 D6 (two-stage T=2): "MEAN≈MODE (Δ~0.16%) → keep mean":** **AGREE** and same order of magnitude (my q35 0.14%, q55 0.55%). Supports the already-made two-stage choice.
- **New nuance vs. prior:** prior notes framed mode≈mean as "the spike is too narrow to matter." The canonical one-stage policy is **not** razor-narrow (SD≈3 u); mode≈mean holds anyway because the skew is mild. And critically, **mode does not systematically reduce e\* error** — it helps only when the policy *overshoots* (q55: mode 1.95% vs mean 2.52%) and *hurts* when it *undershoots* (q35: mode 4.26% vs mean 4.12%). It is not a principled correction toward e\*, only a fixed downward nudge.

### 5d. Recommendation

**Keep the Beta MEAN as the standard one-stage extraction, and carry MEAN into the two-stage pipeline** — and this follows from the numbers, not convention. (i) The mode-vs-mean gap (≤0.16 u, ≤0.55%) is an order of magnitude below both the cross-seed std (0.7–1.1 u) and the mean-vs-e\* bias (0.7–1.9 u), so the choice is numerically immaterial to any reported error. (ii) The mode is **not** a consistent step toward e\*: it improves q55 but worsens q35, because it is merely a fixed downward shift, not a bias correction — switching to mode would trade a 0.15% improvement at q55 for a 0.14% regression at q35 while adding an extraction inconsistency across experiments. (iii) MEAN is already the reported quantity, matches the CLAUDE.md hard invariant ("Beta mean for evaluation, not mode"), and matches the two-stage default (`ppo_multi_stage.effort_function` default = mean). The α,β>1 guarantee makes mode *available* as a diagnostic column, but nothing in the measurement justifies promoting it to the headline. **The real lever on one-stage accuracy is the raw-PPO undershoot/overshoot vs e\*, which is addressed by MC-BR polish (Claim B) — not by the mean↔mode switch.**

---

## 6. Figure change list (minimal edits to make the framework figure truthful)

1. **F1 — figure says** state `s=(w_H, w_L, k, q)` (four scalars) **→ reality is** a normalized 3-vector `s=[q/60, k/1e-3, (w_H−w_L)/10]`: prize *levels* are collapsed to the *gap* `w_H−w_L`, and the state is **constant** for a run (not a per-episode observation). *(`ppo_two_players_clean.py:675-681`)*
2. **F2 — figure says** two independent Agent-1 / Agent-2 policy networks **→ reality is** a **single shared** policy network sampled twice (pure self-play); the drawn "opponent network" box should be removed or marked vestigial/unused. *(`:171,234-246`)*
3. **F6 — figure says** "one-step advantage estimate" **→ reality is** GAE(γ=0.99, λ=0.95) that *reduces to* one-step `r−V` only because episodes are single-step bandit — label it "single-step bandit ⇒ advantage = r−V(s)" so the γ,λ machinery isn't implied to be active. *(`:342-358`, `two_players_env.py:76`)*
4. **F7 — figure says** the KL-window / drift threshold is a convergence gate in the "continue self-play" decision **→ reality is** it only **schedules when the exploitability check runs** (a cheap pre-filter); it never stops training, and a periodic timer runs F8 regardless. Redraw F7 as feeding *into* F8, not as a parallel stop. *(`run_two_players.py:1340-1383`)*
5. **F8 — figure says** ε-equilibrium check (generic) **→ reality is** a **Monte-Carlo** (M=8192, CRN) coarse-to-fine grid best-response with ε=0.03 and patience-5; note it is the **sole** stop, and that it **predates** the two-stage deterministic-quadrature + dReach-UCB verifier (flag if the multi-stage figure is meant to share this box — it should not). *(`:266-359,1425-1439`)*
6. **F9 (effort) — figure says** "learned effort profile" **→ reality is** the **Beta mean** of the *final* policy at stop (no checkpoint selection), and the reported "TEL-PPO" number is **raw PPO**; the near-e\* "verified equilibrium" number is a **separate post-hoc MC-BR-polished** column. Add a downstream "MC-BR polish (Claim B)" box that the figure currently omits entirely. *(`:1664-1684`; `utils/mc_br_polish.py`; `results/one_stage_claimb_summary.csv`)*
7. **F9 (symmetry) — figure says** symmetry diagnostic `|ē₁−ē₂|` in the symmetric case **→ reality is** it is computed but **vacuous** (one shared net ⇒ it measures sampling noise, ~0.04–0.10). Either drop it or annotate it as a sampling-variance sanity check, not an equilibrium-symmetry signal. *(`:1364-1368`)*

---

*Hard-rule compliance: every code claim carries a `file:line`; every number is from a file in the repo or the Part-B script run this session (`scratchpad/partB_mode_vs_mean.py`, which reproduces the committed Claim-B CSV raw column). No placeholders. No figure or code was modified in this pass.*
