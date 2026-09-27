# Prospective q=50, T=2 multiple random restarts
Registered UTC: 2026-09-24T14:06:36.104298+00:00

## Purpose and fixed experiment
Measure whether a budget of up to three independent full solver initializations obtains certified candidates more reliably than one. This is a new held-out evaluation, separate from the retrospective regrouping in RESTART_PROTOCOL_20260923.md.
Use the existing FINAL_T2_PROTOCOL_20260922.json and its unchanged runner, development verifier, locked final verifier, thresholds, fixed per-run caps and first-eligible checkpoint rule. Each candidate receives the same final verifier; a final rejection is a failed restart. No checkpoint substitution, threshold adjustment, warm start, or training resumed after final verification.
Economic configuration: T=2, q=50, w_h=6, w_l=2, k=1/3500, effort=[0,100].
Seeds: exactly 10201 through 10230, all30 independent and previously unused in searched experiment seed records.
Ten fixed triples in increasing seed order. k=1,2,3 use prefixes within each triple. These are ten replications of one economic configuration, not ten different configurations.

| Replicate | Seed1 | Seed2 | Seed3 |
|---|---|---|---|
| 1 | 10201 | 10202 | 10203 |
| 2 | 10204 | 10205 | 10206 |
| 3 | 10207 | 10208 | 10209 |
| 4 | 10210 | 10211 | 10212 |
| 5 | 10213 | 10214 | 10215 |
| 6 | 10216 | 10217 | 10218 |
| 7 | 10219 | 10220 | 10221 |
| 8 | 10222 | 10223 | 10224 |
| 9 | 10225 | 10226 | 10227 |
| 10 | 10228 | 10229 | 10230 |

## Measurement design and operational policy
All30 scheduled runs are executed to obtain unselected per-run rates and paired k1/k2/k3 comparisons, even when an earlier member succeeds. This fixed measurement design differs from stopping actual computation at first success.
The operational policy being evaluated is at most3 total runs (one initial attempt plus at most2 additional restarts), returning the first final_joint_pass. A development-only candidate does not stop operational restarts.
Report actual evaluation cost (all30), separately from reconstructed sequential first-certified-success cost, per k. Neither the prefix comparisons nor the reconstructed costs are extra independent data.
Each process starts from scratch with its own training seed/RNG namespaces; no state is shared. Run order and outcome do not alter assignments.

## Endpoints, denominators and uncertainty
1. Candidate-discovery rate: number of runs with a development-eligible candidate / all30 scheduled runs.
2. Conditional certification rate: candidates satisfying final_joint_pass / all discovered candidates. Zero candidates => N/A.
3. End-to-end success rate: joint-certified candidates / all30 scheduled runs.
4. Group success for fixed k1/k2/k3 prefixes: groups with >=1 joint-certified candidate /10.
Report all30 rows, all10 group rows, failures, first successful restart position and Wilson95 intervals. For no candidate, certification is not_applicable_no_candidate; terminal diagnostics cannot become candidates.
Report per-run computing cost and recovery diagnostics by first-success position where available. Recovery has no new acceptance threshold.
Infrastructure failures remain explicit and in scheduled denominators; they are not replaced by new seeds. An unresolved infrastructure failure prevents a definitive scientific decision.
Do not pool these outcomes with history for the primary comparison. The old 7/10 discovery and pooled26/30 discovery,24/30 end-to-end remain historical.
The illustrative calculation p=.7 gives k2=.91 and k3=.973 for discovery under independent identical p; these are not measured certification probabilities or guarantees.

## Prespecified decision and stopping
Complete the fixed30-run batch; do not stop for favorable intermediate outcomes or add seeds for unfavorable ones.
Retain restart as a useful two-stage engineering pilot if k3 joint-certified group success strictly exceeds k1 and at least one failed first attempt is rescued by position2 or3. Report the gain and resource cost. This is a descriptive engineering decision, not a significance test or a claim of improved per-run convergence.
If no measured gain, terminate the method after this batch and do not introduce it in three-stage.
If there is a gain, it may justify a separately preregistered three-stage pilot. T2 probabilities and verifier cannot certify T3, and this experiment does not launch T3 or override its existing readiness requirements.
No new hash scheme, contract, baseline or gate is introduced; existing locked verifier safeguards remain in force.

## Execution and reproducibility
Root: . on vector2 only.
Runner and Python paths are preserved in manifest.json. launch.py is copied unchanged from E1.
CPU,10 concurrent processes, one Torch/OMP/MKL/OPENBLAS thread each.
Per-run outputs retain full configs, logs, checkpoints, histories and final_eval.json.
