# Two-stage and three-stage tournament experiments

Start here for the September 2026 multi-stage experiments. The repository root README also describes a separate, older one-stage study; its economic parameters and success criteria must not be substituted for the experiments below.

This release connects **design → executable configuration → every run → verification → economic diagnostics**. It includes unsuccessful runs. Recorded outputs are evidence from completed experiments; fresh reproduction outputs belong in `scratch/`.

## Read in this order

1. [Model and numerical methods](METHODS.md).
2. [Two-stage cohorts and reproduction](two_stage/README.md).
3. [Three-stage pilot and formal results](three_stage/README.md).
4. [Exploratory diagnostics and stopped approaches](experiments/diagnostics/README.md).
5. [What is included, and why](PUBLICATION_SCOPE.md), [reproduction environment](REPRODUCIBILITY.md), and [publication validation](PUBLICATION_VALIDATION.md).

## Main results at a glance

“Candidate” means the training search satisfied its development stopping rule. “Certified” additionally requires the same locked final verifier, numerical refinement and concentration checks. A discovered candidate can fail certification.

| Study / role | q | Seeds | Candidate / all | Certified / all |
|---|---:|---|---:|---:|
| T2 confirmation of final protocol | 50 | 10101–10110 | 7/10 | 7/10 |
| T2 confirmation of final protocol | 60 | 10111–10120 | 10/10 | 10/10 |
| T2 additional q50 cohort (E1) | 50 | 10121–10140 | 19/20 | 17/20 |
| T2 prospective restart study, individual runs | 50 | 10201–10230 | 25/30 | 22/30 |
| T2 precision supplement to restart study | 50 | 10231–10240 | 4/10 | 3/10 |
| T3 implementation pilot | 50 | 10411–10413 | 0/3 | 0/3 |
| T3 implementation pilot | 60 | 10401–10403 | 0/3 | 0/3 |
| T3 formal | 50 | 11001–11020 | 0/20 | 0/20 |
| T3 formal | 60 | 11101–11120 | 0/20 | 0/20 |

The two T2 sets with 30 runs are different: confirmation+E1 gives 26/30 discovery and 24/30 certification; the independent restart study gives 25/30 and 22/30. Do not combine them under a single “30-run result.” The restart study preassigned groups before execution; its group-level success must be read from its own tables, with the run cost reported.

For T3, all 40 formal runs finished normally. All Phase B exits were budget-forced and all Phase C searches exhausted 1,800 updates. There were **no candidates**, so conditional certification is **N/A**, not a measured zero out of candidates. Discovery and end-to-end success were each 0/20 per q (Wilson 95% interval [0, 0.161]). Endpoints and historical minimum-deviation checkpoints are diagnostic policies, not certified equilibria.

## Code map

| Responsibility | File |
|---|---|
| Sampled tournament dynamics, domains and starts | [curriculum_env.py](../envs/curriculum_env.py) |
| Shared actor, critic and PPO update | [ppo_curriculum.py](../agents/ppo_curriculum.py) |
| Dynamic best response, one-step gains and reach propagation | [dp_br_verifier.py](../utils/dp_br_verifier.py) |
| Earlier T2 runner and shared collection/evaluation functions | [run_final_dp_br.py](../run/run_final_dp_br.py) |
| T2 final-protocol manifest runner | [run_final_dp_br_round3_dense.py](../run/run_final_dp_br_round3_dense.py) |
| T3 experiment implementation and tests | [T3 source directory](../experiments/three_stage_implementation_pilot_20260924/) |

Only paths and publication/reporting interfaces were adapted for this release. Original source experiments and large raw artifacts remain on the research server. See the scope document for the distinction between reproducible primary studies and archived exploratory evidence.
