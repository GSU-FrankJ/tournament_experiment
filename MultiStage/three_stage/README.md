# Three-stage results

The T3 curriculum runs all completed, but **none found a candidate or certified equilibrium**. Every policy curve and economic comparison here describes an uncertified diagnostic policy.

| Cohort | q50 | q60 | Interpretation |
|---|---|---|---|
| Formal | 0 candidates / 20 seeds | 0 candidates / 20 seeds | All Phase B exits budget-forced; all C searches exhausted 1,800 updates |
| Pilot | 0 candidates / 3 seeds | 0 candidates / 3 seeds | Implementation/debug evidence; not formal reliability |

The formal discovery and end-to-end rates are each 0/20 per q; their Wilson 95% interval is [0, 0.161125]. Conditional certification is N/A because there were no candidates.

- [Portable code, tests, replay, and artifact instructions](../../experiments/three_stage_implementation_pilot_20260924/README.md).
- [Full formal report](../../experiments/three_stage_implementation_pilot_20260924/reports/FORMAL_REPORT.md) and [formal per-seed tables](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/FORMAL_TABLES.md).
- [Pilot report](../../experiments/three_stage_implementation_pilot_20260924/reports/PILOT_REPORT.md).
- [Chinese formal experiment record](T3_FORMAL_EXPERIMENT_RECORD_20260926.md), [tables](T3_FORMAL_EXPERIMENT_RECORD_20260926_tables.md), and [data](T3_FORMAL_EXPERIMENT_RECORD_20260926.csv).
- [Implementation plan](T3_IMPLEMENTATION_PLAN_20260924.md), preserved as a historical protocol; execution commands in the portable README supersede its server-specific workflow.

The compact archive retains complete seed-level outcomes and mean/stochastic economics, both endpoint and minimum-development weights, and policy samples for all 46 seeds. Large raw policy tables, action-value arrays, optimizer files, and most update histories are excluded. The complete pilot q60 seed 10401 history is included for the original training regression test. This archive supports saved-weight verifier replay and new training runs, while full historical report regeneration requires the omitted raw artifacts.
