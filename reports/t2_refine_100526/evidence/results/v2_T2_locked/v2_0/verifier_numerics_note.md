# Verifier numerics item (check (ii), decision D3)

**Reported, not gated.** Source: `results/v2_T2_locked/v2_0/continuation_check_v2_0.json` (key `verifier_numerics`), written by `tests/test_v2_refine_continuation.py --write` (`verifier_numerics_item`). Parents: the end-of-A states of the v1.1 rehearsal, seed 10501, both q (`parents` in the record, with SHA-256).

**What is measured.** With the stage-1 opponent fixed at 40 (q = 50) / 45 (q = 60), the verifier's stage-1 value Q₁(0, e) + k e² is compared with the table Ṽ₂(e − ê₁(0)) on the verifier's effort grid; δ(e) is their difference (units of ΔW = 4). If δ were an error of the table, the stage-1 objective −k e² + Ṽ₂(e − ê₁(0)) + δ(e) (δ linearly interpolated onto a 0.001 effort grid) would have its maximiser at the verifier's own stage-1 optimum instead of the table's. The shift of that maximiser is the precision with which the verifier, on that tier, can locate the stage-1 optimum ẽ₁ that the training objective defines.

| verifier tier | q | max \|δ\| / ΔW | at effort | stage-1 optimum, table | optimum if δ were a table error | implied shift | shift / e₁* |
|---|---|---|---|---|---|---|---|
| final (state step 2, 32 GL nodes per half interval) | 50 | 3.229e-05 | 22 | 50.137 | 50.037 | -0.100 | 0.00214 |
| development (state step 4, 16 GL nodes per half interval) | 50 | 0.0001329 | 3 | 50.137 | 49.743 | -0.394 | 0.00844 |
| final (state step 2, 32 GL nodes per half interval) | 60 | 1.96e-05 | 21.5 | 36.862 | 36.729 | -0.133 | 0.00342 |
| development (state step 4, 16 GL nodes per half interval) | 60 | 8.511e-05 | 16 | 36.862 | 36.543 | -0.319 | 0.0082 |

**Reading.** On the standard final tier the gap is 3.229e-05·ΔW (q = 50) and 1.96e-05·ΔW (q = 60), and the stage-1 optimum it would imply differs from the table's by -0.100 and -0.133 effort units (0.00214·e₁* and 0.00342·e₁*). That is above the limit of 1e-3·e₁* that (ii-c) applies to the table's own convergence profile, whose measured shift is 2.14e-05·e₁* (q = 50) and 0·e₁* (q = 60). The gap sits on the verifier's side (its state-grid interpolation of V₂, step 2 on the final tier), falls by about 4× per halving of the state step, and is 4.982e-07·ΔW / 4.1e-07·ΔW on the refined configuration ((ii-b)). It therefore bounds the precision of the verifier's own ẽ₁ on the standard tier; it is not evidence about the table.
