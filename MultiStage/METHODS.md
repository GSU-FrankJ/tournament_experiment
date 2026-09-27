# Model and numerical methods

## Economic model

Two identical players act for T = 2 or T = 3 stages. At each stage each chooses effort in [0, 100], incurs cost k e² with k = 1/3500, and receives an independent uniform shock in [−q, q]. The larger final cumulative score wins 6, the loser receives 2, and a tie splits the prizes. Thus ΔW = 4. The experiments use q = 50 and q = 60.

The player's signed score gap evolves as d′ = d + e − e_opponent + ε − ε_opponent. An opponent observing the same state sees −d. Set B = 100 + 2q. The feasible stage-t domain is D_t = [−(t−1)B, (t−1)B], with D_1 = {0}. The observation is ((t−1)/(T−1), d/((t−1)B)), with the root gap component set to zero.

Rewards used for PPO training are **sampled realized outcomes**. Exact terminal probabilities and numerical best responses are used for offline/development verification, not substituted into training rewards.

## Shared learning implementation

The actor and critic are separate 2→64→64 networks with tanh hidden layers. The actor parameterizes a Beta action distribution by its mean and concentration, with concentration 100 + softplus(z). The mean head is initially zero. The normalized action maps to effort [0, 100].

The implemented PPO uses Adam (learning rate 3e−4), ten epochs per update, minibatches of 256 learner transitions, clipping 0.2, gradient clipping 0.5, zero entropy coefficient, and γ = λ = 1. Actor and critic optimizers are separate. Frozen opponent snapshots refresh every 20 global updates and on phase entry; optimizer state continues across phases. Independent random streams separate initialization, shocks, learner actions, opponent actions, starts/roles and minibatches.

These are method summaries. The published manifest and runner for a particular cohort remain the authority for its exact stopping rule and numerical settings.

## Backward curriculum and exploring starts

| Horizon | Phase A | Phase B | Phase C |
|---|---|---|---|
| Final T2 protocol | Terminal stage 2, 400-update cap; 512 stage-2 ES episodes/update | Full stages 1–2, at most 600 updates; 512 root episodes/update | Full stages 1–2, at most 1,000 updates; 256 root + 256 stage-2 ES episodes/update |
| T3 protocol | Terminal stage 3, fixed 400 updates; 512 stage-3 ES episodes/update | Continuation stages 2–3, at most 600 updates; 512 stage-2 ES episodes/update | Full stages 1–3, at most 1,800 updates; 256 root + 85 stage-2 ES + 171 stage-3 ES episodes/update |

ES means exploring starts. Gap bins have width 10; choose a bin uniformly, then draw uniformly within it. For T3, q50 has 40 bins in D2 and 80 in D3; q60 has 44 and 88. The C mixture approximately equalizes expected ES exposure per stage/bin. Subsequent transitions contribute additional state visits.

The retained T2 A transition rule still permits verifier-based advancement; the reported runs used all 400 updates. Thus its design description as fixed-budget pretraining should not be read as removal of that code path.

For T3, A is a fixed budget, B is checked every 25 updates with three consecutive eligible passes needed, and C freezes the first eligible checkpoint at cadence 25. B checks a **two-stage dynamic BR starting at stage 2**, not a one-step stage-2 action. T3 B requires max_D2(V2_BR − V2_mean)/ΔW ≤ 0.02 and concentration ≤ 0.04; C requires dReach/ΔW ≤ 0.01 and concentration ≤ 0.04. A budget-forced B exit is recorded explicitly.

## Verification quantities

Fix the checkpoint's deterministic mean policy ê_t(d) and the opponent's mean policy ê_t(−d). The verifier works backward from the terminal stage to the root:

- V_t_BR optimizes current and future actions against the fixed opponent.
- V_t_mean follows the checkpoint's mean policy for the player and opponent.
- EXP_root = V_1_BR(0) − V_1_mean(0).
- δ_t(d) is the best **one-step** deviation gain when subsequent own actions follow the mean policy. This differs from V_t_BR − V_t_mean.
- BR-reachable sets R_t propagate the intervals induced by the selected dynamic BR action, the mean opponent and the full bounded shock support.
- dReach = Σ_t max_(d∈R_t) δ_t(d).
- Δ_max_all = max_(t,d∈D_t) δ_t(d). It is not the sum of domain-wise maxima.

Reported payoff quantities divide by ΔW. The final-tier grid has state step 2, effort step 0.5 and 32 Gauss–Legendre nodes per half of the triangular difference-noise support. Development uses state step 4, effort step 1 and 16 nodes per half. Nonterminal continuation values use linear interpolation; out-of-domain landings are errors. Terminal expectations use the exact triangular CDF. The effort search includes the policy action and recomputed valid parabola-vertex candidates.

Final certification requires an actual discovered candidate, valid development and final evaluations, final dReach/ΔW ≤ 0.01, development-to-final differences in both dReach and EXP_root at most 0.002, and maximum normalized Beta standard deviation ≤ 0.04 on the dense grid (step 0.05). Earlier-protocol cohorts are identified separately; their search rules must not be inferred from the final protocol.

No-candidate endpoints are evaluated for diagnosis. A favorable historical minimum is not retrospectively promoted to a candidate. Numerical validity and refinement agreement alone do not establish certification.

## Economic outputs and uncertainty

Policy functions, root effort, expected effort by stage, state visitation, and leader/follower asymmetry describe the evaluated checkpoint. For uncertified T3 endpoints, these are **economic diagnostics of learned policies**, not equilibrium comparative statics.

Mean-policy and stochastic-policy simulations are distinct. Across-seed standard deviations, Monte Carlo standard errors, and bin counts have different meanings and are labeled separately. Self-play visitation is not BR reachability; a visit to a bin is not a visit to one exact continuous state.

Discovery rate uses all prescribed runs as its denominator. Conditional certification uses discovered candidates, and is N/A if none exist. End-to-end success uses all runs. Multiple-restart reliability uses preassigned groups and reports the computational cost; diagnostic seeds and post-hoc groupings are not held-out success-rate estimates.
