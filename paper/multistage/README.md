# Multi-stage paper artifacts

Figures and tables for the multi-stage tournament experiments (plan
section 6). Regenerated from the committed convergence JSONs in
`results/multi_stage/convergence/` by:

```bash
python tools/make_multistage_figures.py   # -> figures/*.pdf (+ .png previews)
python tools/make_multistage_tables.py    # -> tables/*.tex
```

PDF figures and `.tex` tables are committed; PNG previews are gitignored
(regenerable).

## Figures (plan section 6)

| file | content | data |
|---|---|---|
| `F1_two_stage_recovery` | closed-form vs TEL-PPO stage-2 effort | T=2 |
| `F2_verifier_calibration` | EXP of closed-form / TEL-PPO / bad policies | T=2 |
| `F3_three_stage_effort` | learned e_hat_t(d), t=1,2,3 (main result) | T=3 |
| `F4_three_stage_br_vs_learned` | best response vs learned per stage | T=3 |
| `F5_three_stage_deviation_gaps` | one-step deviation gaps Δ_t(d) | T=3 |

## Tables (plan section 6)

| file | content |
|---|---|
| `table1_two_stage_recovery` | T=2 recovery metrics + exploitability |
| `table2_three_stage_certificate` | T=3 exploitability certificate (per seed) |
| `table3_robustness` | grid refinement, seed robustness, falsification |
| `table4_multistage_summary` | T=2,3,4,5 benchmark comparison |

## Caveats

- **F1 is sparse** (5 probe points): the T=2 runs predate the
  `effort_curves` field. Re-run one T=2 seed with the current runner for a
  dense stage-2 curve.
- **Table 4 T=2 total effort/cost** are analytic recovered values (same
  reason). T=3/4/5 use the on-path summary from the saved curves.
