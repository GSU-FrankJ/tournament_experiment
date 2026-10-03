# T=2 v2: PI-side record (plans, prompts, decisions)

This folder holds the PI-side half of the T=2 v2 work: the two plan documents, every prompt sent to the implementation agent, and the decision made at each review gate. The agent-side half (code, results, reports) is in the repository's commit history and `reports/v2/`.

**Scope:** P0 (code audit and regression) to P6 (fresh-seed confirmation), plus the report pack. T=3 and the paper are out of scope.

**Completeness note.** The planning conversation was long and was partly compacted, so this folder does **not** contain a verbatim transcript of the PI-side discussion. It contains:
- every prompt exactly as delivered (`prompts/`);
- the PI's decisions at each gate, recorded below from the session record.

Analysis messages written in Chinese during the discussion are summarized here only through their decisions.

## Contents

| Path | What it is |
|---|---|
| `plans/MultiStage_092826.docx` (+ `.md` conversion) | 0929 revision plan: full v2 design, P0–P9, validation targets |
| `plans/MultiStage_093026.docx` (+ `.md` conversion) | 0930 narrowed plan: the two method changes and three pilots (baseline of the T=2 report) |
| `prompts/01`–`10` | Prompts to the implementation agent, in delivery order |
| `analysis/zero_effort_check.py` | Independent PI-side check of the zero-effort gains and the root-game BR slope (no repo code) |
| `agent_messages/` | Two agent chat summaries that are not stored elsewhere in the repo |
| `SHA256SUMS` | Checksums of every file in this folder |

## Chronology and decisions

| # | Prompt | Round | PI decisions at the gate that followed |
|---|---|---|---|
| 01 | `01_pilots_prompt.md` | Phase 0 audit (read-only) | Keep base `657f54a` and repo path; q = 50, 60; seeds 10501–10503; stage-1 training is Phase B only; in frozen mode advantages over stage-1 rows only, joint mode unchanged; stage-1 opponent stays the lagged copy; RNG alignment (A6): `expected` still draws shocks, `mean` still draws and discards stage-2 actions; pytest only in `.venv` |
| 02 | `02_phase1_prompt.md` | Phase 1 verifier and calibration | Run all three groups; on-path probability strictly > 0 (later revised); both opening checks |
| 03 | `03_phase2_prompt.md` | Phase 2 infrastructure | On-path rule = open interval with exact cell masses; expand to 10 seeds (10501–10510); log RNG positions and check the dReach reach mask |
| 04 | `04_pilot1_prompt.md` | Pilot 1: sampled vs expected reward | Reward estimator = `expected` |
| 05 | `05_pilot2_prompt.md` | Pilot 2: joint vs frozen | Frozen variant B2 (`adv_norm_scope = stage1_rows`); ẽ₁ by residual minimization with an uncertainty band; run a Phase A extension in parallel |
| 06 | `06_pilot3_prompt.md` | Pilot 3 + Phase A extension | Continuation mode = `mean`; do a stabilization round first; Phase A fixed at 1600 updates with an end-of-phase gate; joint training and Phase C dropped; no sampler change |
| 07 | `07_pilot4_prompt.md` | Pilot 4: stabilization | LR decay in both phases, evaluate the last iterate; adopt the proposed gates; 20 fresh seeds per q with pass rule ≥ 18/20; peak error reported only, plus a cusp diagnostic |
| 08 | `08_lock_prompt.md` | Protocol lock v1.0 + dev-seed rehearsal | Stage-1 criterion becomes pre-registered secondary criterion S1; accept Check 1 with hardening of the process-global RNGs; add numerical refinement G-N (\|dev − final\| ≤ 0.001·ΔW) to run pass; continue to the confirmation automatically if all checks pass; seeds 20501–20520 |
| 09 | `09_v1_1_confirmation_prompt.md` | v1.1 lock, re-rehearsal, confirmation | Result: confirmation PASS, 20/20 at both q |
| 10 | `10_t2_report_pack_prompt.md` | T=2 report pack | Pack built (111 items) |

## Corrections made by the PI side during the work

These are recorded so that the prompts are read correctly:
- **On-path rule.** "Exact probability > 0 is enough" was wrong (quadrature-node PMF artefacts, Phase 2 check 1c). It was replaced by the open interval with exact cell masses.
- **Kink-representation hypothesis.** The remaining stage-2 peak error was attributed to the network smoothing the cusp. Pilot 4 §1d falsified this: the supervised fit reaches the target.
- **Stage-1 oscillation.** It was over-attributed to best-response cycling. Pilot 4 §1b showed it is noise-driven (partial adjustment, contraction factors 0.6–0.85).
- **Stage-1 10% gate.** It was set from Pilot 4 distributions with constant-LR Phase A parents. The locked pipeline's rehearsal showed a heavier tail, which led to v1.1.
- **v1.0 pass probability.** An early estimate of 16–46% was corrected to 0.14 (rate model) and 0.35 (normal model). The agent recomputed 0.1395 and 0.3494.
- **Check 1 wording.** "All RNG states" was broader than intended; three never-consumed process-global RNG states differed.
