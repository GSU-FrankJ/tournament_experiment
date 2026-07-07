# Claim-A non-locking κ-continuation retrain

Status: in-progress (owner OVERRULED the kill 2026-07-02, choosing the
adaptive-batch continuation variant: batch is enlarged during the κ ladder to
attack the D2 gradient-SNR physics directly — noise floor ∝ 1/√B, predicted
landing std 1.67 → ~0.4 at 16×. Pilot-gated: 1 seed first, measure the diffusion
band, then owner decides on 5 seeds.)
Current phase: phase02 (implementation + smoke + 1-seed pilot)

## What's done
- Task created 2026-07-02 under owner authorization (Gate A branch (ii) of
  `docs/tasks/claim-a-dev-trigger-retrain/`).
- Architecture + kill conditions pre-registered in CLAUDE.md.
- **Phase01 complete** (`tools/claim_a_continuation_design.py`, ZERO GPU). Findings in
  `phase01_findings.md`, dump `phase01_design.json`, log `phase01_run.log`:
  - D1: smoothed-equilibrium curve μ*(κ) measured: 22.59 (κ=20) → 23.96 (κ=200) →
    24.74 (κ=400); det ≈24.5–25 (Finding B FOC is the better det estimate).
    Component-2's κ_top=200 structurally capped its target at ~24 (4% under e*).
    r5 stall 22.99 ∈ [μ*(20), μ*(60)]; c2 outcomes = target 24.0 ± diffusion band.
  - D2 (KILL): every velocity death happened with HEALTHY approx_kl (0.0068–0.0083,
    larger than explore-phase KL) — gradient-SNR physics (diffusion within ~1.5 units
    of target at batch 4096), not optimizer starvation. Floors cannot fix it.
    Pre-registered kill condition (2) met → recommend STOP before GPU.
  - D3: ladder budget moot; binding constraint is the ±1.5–2 diffusion band, which
    violates Gate C (std ≤ 0.5, mean ≥ 24.5) in expectation.
- Recommendation: STOP; adopt Claim B upgraded by the μ*(κ) curve ("PPO converges to
  the exploration-smoothed equilibrium; MC-BR + exploitability bridge the gap").
  Claim A: not reachable in this parameterization at batch 4096 — three independent
  attempts (r5 schedule, Component-2 lock, this design analysis) now agree. DO NOT
  resurrect without new variance-reduction evidence.

## Phase02 (2026-07-02) — implemented, smoked, pilot LAUNCHED
- Owner overruled the kill, choosing the **adaptive-batch continuation** variant
  (attacks the D2 SNR physics: noise floor ∝ 1/√B; predicted landing std
  1.67 → ~0.4 at 16× ladder batch). Pilot-gated before any 5-seed spend.
- Implemented `--kappa-continuation` in `run/run_three_players.py` +
  `kappa_continuation` PPOConfig field / head-selection in
  `agents/ppo_three_players.py`. Design: explore (κ∈[1,20], batch 4096) →
  κ ladder [20,35,60,100,200,400] with per-stage batch [16384×3, 65536×3],
  minibatch scaled to keep grad-step count constant, kinematic gate
  (|Δmode|<0.3 over W=30, hold 60–250, forced-advance logged), lr/entropy pinned
  (3e-4 / 0.02) during ladder, stop suppressed until done, done-phase exploit eval
  every 3 updates. New JSON columns: cont_phase, batch_size; continuation_config
  block in output. Mutually exclusive with --mode-conc-ramp; all legacy paths
  untouched.
- CPU smoke + GPU smoke passed (transitions, batch switch, κ pinning, JSON schema;
  smoke JSONs moved out of results/ to scratchpad). NOTE: host NVML has a
  driver/library mismatch — nvidia-smi is broken, but torch CUDA verified working
  (V100 ×8, real matmul); monitor GPUs via torch.cuda.mem_get_info.
- **Pilot RUNNING**: tmux session `c3_pilot`, GPU 0, seed 42, tag `c3_cont`,
  episodes 20M (~1–1.5 days). Log:
  `results/three_players/logs/c3_cont_pilot_seed42.log`.

## Recheck (2026-07-02, ~update 392, stage 3 κ=100)
- Verified against LIVE pilot data, no code bugs found:
  - Adaptive batch confirmed WORKING: measured step delta/update = 4096 (explore) →
    16384 (stages 0-2) → 65536 (stage 3+). agent.cfg is ppo_cfg (same object), so the
    mutation reaches the rollout loop. Grad-step count held constant (minibatch scales).
  - GPU confirmed in use (torch mem_get_info; nvidia-smi still NVML-broken but irrelevant).
  - Stop correctly SUPPRESSED during ladder despite exploit_ok_streak already =13
    (ramp_allows_stop gate works); will stop promptly once done is reached (≥60 updates
    at κ=400 exist first, enough to measure the band).
- Two design PROPERTIES to flag to owner (not bugs):
  1. Mean is tracking the modest smoothed-equilibrium targets μ*(κ) (hovering 21-23.5
     through κ=60), NOT shooting to 25 — exactly as D1 predicts (μ* only reaches ~24.7
     at κ=400). Verdict hinges entirely on stages κ=200/400 + done.
  2. Budget tension: each stage advanced at exactly min_hold=60 (window-trend flattened
     fast → hover regime, not slow-climb). At 60/stage the total ≈ 18-19M < 20M budget,
     OK. IF a high-κ stage instead slow-climbs (the success case), 3×~150 upd ×65536 ≈
     29M would truncate at 20M. Evidence (0-2 flatten at 60; D2 velocity dies at high κ)
     says slow-climb is unlikely, so 20M stands; if a run hits max_updates mid-ladder,
     rerun with a larger budget rather than scoring it a stall.
- Scientific read: adaptive batch cuts gradient-estimate VARIANCE (→ tighter band) but
  not the plateau's tiny gradient MEAN (→ drive). Optimistic case: noise was masking a
  real small uphill signal → climb resumes. Pessimistic: plateau gradient is truly ~0 →
  band tightens but mean stays below target → KILL. Stages κ=200/400 decide.

## Recheck #2 (2026-07-03) — pilot #1 hit a BUDGET BUG; relaunched as pilot #2
- **BUG found**: pilot #1 stopped at update 417 / stage 3 (κ=100), stop_reason
  `max_updates`, never reaching κ=200/400. Root cause: the base config
  (`config/one_stage_three_players.py`) sets `max_updates: 1500`, which caps
  total_steps_target at 1500×4096 = **6,144,000 STEPS** — but the adaptive ladder
  spends 65536 steps/update in the high-κ tail, so 6.144M steps = only ~417
  updates and the ladder truncated. `--episodes 20000000` was silently overridden
  by the step cap. Pilot #1 is INVALID as a full test.
- **Fix**: relaunch with `--episodes 34000000 --max-updates 9000` (lifts the cap to
  36.9M ≥ 34M budget) + `--cont-max-hold 120` (guarantees the ladder reaches done
  within budget even if high stages force-advance: worst case ~30M < 34M).
- **Partial signal salvaged from truncated pilot #1** (stages 0-2 completed;
  preserved as `phase02_pilot_truncated_seed42.{json,log}`). Per-stage headline
  MEAN level vs μ*(κ) target, and band = std(mean, last 30 upd):
  | stage | batch | mean | μ* | band |
  |---|---|---|---|---|
  | explore κ≤20 | 4096 | 21.0 | — | 1.12 |
  | κ=20 | 16384 | 20.9 | 22.6 | 0.69 |
  | κ=35 | 16384 | 22.2 | 22.7 | 0.68 |
  | κ=60 | 16384 | 22.98 | 23.2 | 0.79 |
  | κ=100 (partial) | 65536 | 23.46 | 23.8 | 0.55 |
  | c2 ref κ=200 | 4096 | 24.1 | | 1.07 |
  BOTH gate metrics trend the right way but BORDERLINE: mean tracks μ*(κ) and is
  still climbing (23.46 at κ=100); band shrinks 1.07→0.72→0.55 with batch (only
  ~2× per 16×, weaker than ideal 1/√B ⇒ part of the band is non-sampling). The
  decisive κ=200/400 stages were exactly what the bug truncated.
- Verified no OTHER bug: adaptive batch reaches the rollout (step deltas), grad-step
  count constant, GPU in use, stop correctly suppressed during ladder.

## Pilot #2 COMPLETE (2026-07-04, ~18h, clean exit, 0 forced advances)
- Ladder ran all 6 stages to κ=400 (advances at ~60 upd each, all gate not forced);
  done stopped immediately (exploit streak 122, exploit=0.004). Final mean **24.30**.
- Per-stage headline mean vs μ*(κ) and band = std(mean, last 30 upd):
  | stage | batch | mean | μ* | band |
  |---|---|---|---|---|
  | explore κ≤20 | 4096 | 21.2 | — | 0.96 |
  | κ=20 | 16384 | 20.9 | 22.6 | 0.61 |
  | κ=35 | 16384 | 22.3 | 22.7 | 0.80 |
  | κ=60 | 16384 | 22.5 | 23.2 | 0.84 |
  | κ=100 | 65536 | 23.5 | 23.8 | 0.63 |
  | κ=200 | 65536 | 23.8 | 24.0 | 0.54 |
  | κ=400 | 65536 | 24.23 | 24.7 | 0.59 |
- vs baselines: c2 seed42 final 22.70 (band 1.07), r5 22.99. Adaptive batch gave
  **+1.6 mean and ~2× tighter band** — real, monotone improvement. But 16× batch
  bought only ~2× band reduction (not the ideal 4× of 1/√B) ⇒ ~half the band is
  policy diffusion, not sampling noise.
- **VERDICT: borderline — falls in the PASS/KILL gap.** Both Gate-C metrics MISS by
  small margins: mean 24.23–24.30 < 24.5 (by ~0.2–0.3); κ=400 band 0.59 > 0.5 (by
  ~0.1). Not a KILL (band ≪ 1.0, mean ≥ 24.0, 0 forced advances, clean run), not a
  clean PASS. Even best-case, PPO reaches 24.3 not 25 ⇒ at most a WEAK Claim A; the
  final ~0.7 gap still needs MC-BR polish (Claim B's bridge survives).
- Data: `results/three_players/convergence/ppo_3p_q35.0_seed42_c3_cont_convergence.json`.

## What's next — OWNER DECISION (pilot is borderline, not a clean gate)
- (a) 5-seed q35 run — the decisive Gate-C metric is CROSS-SEED std (≤0.5) + mean
  (≥24.5), which one seed cannot measure; per-seed band halving suggests the 5-seed
  spread may tighten well below c2's 1.67. ~5 GPU × ~18h. Recommended IF Claim A
  matters, eyes open that success = ~24.3, not 25.
- (b) Call it → adopt Claim B, now strengthened: even a 16× batch continuation tops
  out at 24.3, so the undershoot is fundamental and the MC-BR bridge is justified.
  The μ*(κ) curve + this pilot are the paper evidence.
- (c) Push κ_top higher (800/1600) or batch bigger + re-pilot — diminishing returns
  (D1 μ* nearly flat past 400; higher κ = less exploration = flatter gate).

## Phase03 LAUNCHED (2026-07-06): owner chose (a), 5 FRESH seeds 43–47
- Pilot κ=200/400 numbers re-verified straight from the JSON before the decision
  (segment-by-cont_phase recompute matched the table above; final 24.30, gap 0.70,
  stop_reason=exploitability, 0 forced advances).
- **Seed-set decision (owner-confirmed)**: Gate C is scored on 5 FRESH seeds 43–47
  ONLY. Pilot seed 42 is demoted to pilot/supplementary — including it would add
  selection bias (the 5-seed spend only happens when the pilot looks OK). Cost of
  cleanliness: +1 run (~18 GPU-h); wall time unchanged (parallel). For the paper's
  c2-vs-c3 comparison, the 42-overlap caveat must be noted (c2 used 42–46).
- Command per seed (params byte-identical to pilot #2; budget flags dodge the
  pilot-#1 step-cap bug: 9000×4096 = 36.9M ≥ 34M):
  `CUDA_VISIBLE_DEVICES=<0-4> python run/run_three_players.py --method ppo --q 35
   --seed <43-47> --kappa-continuation --episodes 34000000 --max-updates 9000
   --cont-max-hold 120 --ablation-name c3_cont`
- tmux sessions `c3_s43`–`c3_s47`, GPUs 0–4, launched 2026-07-06 19:59; all 5
  confirmed running (config banner correct, ~0.8 GB/GPU allocated). Logs:
  `results/three_players/logs/c3_cont_seed{43..47}.log`. ETA ~18h → 2026-07-07 pm.
- **Gate C scoring on completion (pre-registered, verbatim)**: over seeds 43–47,
  PASS = mean ≥ 24.5 AND cross-seed std ≤ 0.5; KILL = std > 1.0 OR mean|err| > 4%
  (mean < 24.0); between = borderline, owner call. Also record per-seed final mean,
  band, forced advances, stop_reason; any max_updates mid-ladder exit = rerun with
  bigger budget, not a stall.

## Phase03 COMPLETE (2026-07-07) — Gate C scored on seeds 43–47
All 5 seeds finished clean: stop_reason=exploitability, forced_advances=0 (verified
from continuation_config in each JSON), all ladder advances gate-triggered, final
exploit 0.004–0.011 (< eps 0.03), stopped at update 566–608. JSONs:
`results/three_players/convergence/ppo_3p_q35.0_seed{43..47}_c3_cont_convergence.json`.

Per-seed (final = policy_mean_effort[-1], Metric B — verified equal to the JSON
`final.effort` field; k400 = mean/std of last 30 updates of the κ=400 ladder stage):
| seed | final | gap | k400 mean | k400 band |
|---|---|---|---|---|
| 43 | 22.98 | 2.02 | 24.42 | 0.47 |
| 44 | 24.43 | 0.57 | 24.44 | 0.36 |
| 45 | 24.76 | 0.24 | 24.08 | 0.59 |
| 46 | 24.21 | 0.79 | 24.27 | 0.70 |
| 47 | 23.78 | 1.22 | 24.25 | 0.66 |
(pilot 42, excluded from gate per owner decision: final 24.30, k400 24.22/0.59 —
consistent with the 5 fresh seeds)

**Gate C (pre-registered metric = final snapshot; thresholds re-read verbatim from
this task's CLAUDE.md): BORDERLINE — no branch fired.**
- cross-seed mean 24.034 → |err| 3.86% (< 4% KILL line by 0.14pp; ≥ 2% success
  line missed)
- cross-seed std 0.688 (sample, n−1) (> 0.5 success line; < 1.0 KILL line)

**Decomposition (the scientific verdict, sharper than the gate):** the snapshot
spread is almost entirely WITHIN-RUN diffusion sampled at one update — done stops
on its first update (exploit streak ages during the ladder), so "final" is a
1-sample draw from the κ=400 band. Per-seed (final − k400 mean): −1.44, −0.01,
+0.69, −0.06, −0.47 ≈ band magnitude. Seed 43's −1.44 was inspected in the raw
tail: real ±0.7–1.0 update-to-update swings all through κ=400, last 4 updates a
locally-correlated downswing; the ladder→done transition code pins κ/batch/lr/
entropy identically (run_three_players.py:986-991) — no regime-shift artifact.
On the time-averaged metric (κ=400 stage, last 30 upd):
- cross-seed mean **24.29**, std **0.146** (SE ≈ 0.065), |err| 2.83%
- robustness: full-stage (not last-30) mean 24.27, std 0.149 — window choice
  doesn't drive the result
- ⇒ variance criterion PASSES decisively (0.146 ≪ 0.5; c2 was 1.67 ⇒ ~11×
  tighter); mean criterion (≥ 24.5) fails by 0.21 ≈ 3 SE. The undershoot is
  SYSTEMATIC, not noise. All 6 runs (incl. pilot) land 24.1–24.4, also ~0.4 below
  μ*(400)=24.7 — the ladder tracks μ*(κ) with a persistent lag at high κ.

**Read: adaptive batch SOLVED the variance problem; the residual failure is pure
bias. Strong Claim A (raw mean ≥ 24.5, |err| ≤ 2%) is dead in this
parameterization with high confidence — do not spend more GPU on it. This is the
decisive answer the 1-seed pilot could not give.**

## What's next — owner decision (recommendation: adopt Claim B, final form)
- Recommended: write Claim B with this as capstone evidence: PPO converges to the
  exploration-smoothed equilibrium μ*(κ); κ-continuation + 16× batch tracks μ*(κ)
  reproducibly (±0.15 across seeds) up to κ=400 and lands at 24.29 (2.8% under
  e*); MC-BR + exploitability bridge the final gap. μ*(κ) curve + 6-run c3
  ensemble + c2/r5 negatives = complete pre-registered story.
- NOT recommended: κ_top 800/1600 or larger batch (D1: μ* flattens past 400; the
  lag vs μ* itself also persists).
- Honesty note for the paper: the pre-registered snapshot metric returned
  BORDERLINE; report the time-averaged decomposition alongside it, not instead of
  it. Both metrics agree the mean criterion fails.

## Blockers
- None. Owner decision on paper framing (Claim B final form).
