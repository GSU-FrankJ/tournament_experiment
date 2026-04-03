# 3-Player Algorithm Improvement

Status: **blocked — fundamental game theory issue discovered**
Current phase: phase01 (paused)

## Critical Finding (2026-03-28)

**The theoretical equilibrium e*=(w_H-w_L)/(4qk) is NOT a global Nash equilibrium
for the 3-player tournament at q=25, 35, 40.** It is only a local optimum of the
expected utility. The global best response is to shirk (e≈0), saving on cost while
accepting the loser's payoff w_L.

### Validity condition

The interior NE is globally valid iff:

    q >= q_crit = sqrt(N * w_gap / (16k))

| N | q_crit | q=25 | q=35 | q=40 | q=55 |
|---|--------|------|------|------|------|
| 2 | 33.07  | FAIL | pass | pass | pass |
| 3 | 40.50  | FAIL | FAIL | FAIL | pass |

At q=35 (N=3): the equilibrium cost (k*e*²=1.56) exceeds the equilibrium
gain (w_gap/3=1.17), so deviating to e≈0 gains +0.40 in utility.

### Implications

1. PPO's "failure" to converge at 3p q=25,35,40 is NOT a learning failure —
   the symmetric pure-strategy NE does not exist at these parameters.
2. The gradient solver finds the interior LOCAL equilibrium (e*=62.5) but
   misses the global deviation because it only follows local gradients.
3. At 3p q=55 (e*=39.8), the NE IS valid. PPO's gap of ~3.3 here is a
   genuine convergence issue (slow but correct direction).
4. At 2p q=25, the same issue exists — 2p also fails at q=25 (gap ~2.7).

### What was tried (phase01)

| Variant | Gap | Verdict |
|---------|-----|---------|
| pairwise_binary (no warmup) | 11.64 | Worse; early-stops on flat exploit landscape |
| hybrid ns=0.3 (no warmup) | 10.25 | Same failure mode |
| bigbatch 16384 spu (no warmup) | 12.41 | Same failure mode |
| COMA K=32 | killed | Moot — target NE doesn't exist |

All PPO variants converge to effort ~50-52 where the exploitability landscape
is nearly flat (exploit ~0.03-0.06), falsely triggering early stopping. This is
consistent with being near the boundary where the BR function jumps from
interior (~61) to corner (~0) solution.

## What's next

The task needs to pivot from "fix PPO convergence" to:

1. **Derive the participation constraint** q >= sqrt(N*w_gap/(16k)) formally
   and add it to the paper's theory section.
2. **Verify 3p q=55 convergence** — this is the only 3p case where NE is valid.
   Run with more episodes if current gap (~3.3) is too large.
3. **Decide paper strategy**: either (a) restrict 3p results to q≥55,
   (b) change game parameters (lower k or w_gap) so NE is valid at q=35,40,
   or (c) characterize the mixed-strategy equilibrium for q<q_crit.
