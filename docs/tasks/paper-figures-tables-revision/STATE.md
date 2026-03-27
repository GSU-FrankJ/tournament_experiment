# Paper Figures & Tables Revision

Status: in-progress
Current phase: complete

## Key decisions
- **q values**: {35, 40, 55} in all main figures and tables; q=25 only in Figure 6b
- **Tables 3–4**: q=25 rows replaced by q=35 (not appended)
- **q=25 non-convergence**: discussed separately in paper text, not shown in main results
- **Entropy**: NOT removed; baseline uses schedule 0.03→0.005; theory-align modes not used in paper results

## What's done
- Task folder created with CLAUDE.md, STATE.md, and 11 phase files
- Phase structure: 01 (config) → 02–09 (per-figure) → 10–11 (tables)
- PDF reviewed (10 pages), all current figures/tables assessed
- Confirmed all 20 paper baseline runs use entropy schedule (no theory-align-v2)
- **Phase 01 complete**: added SHADE_ALPHA, format_q(), CONV_VLINE_*, ABLATION_LABELS, ABLATION_LINEWIDTHS, updated WEIGHT_VARIANT_LABELS with k
- **Phase 02 complete**: Figure 1 restyled — q=35/40/55 columns, k in labels, lighter shading, convergence vline, deduplicated legend
  - **Open**: no wh8_wl4 data for q=35 (bottom-left panel empty); error box kept pending user decision

- **Phase 03 complete**: Figure 2 (effort_drift) restyled — SHADE_ALPHA for lighter shading, thicker threshold line (2.5), standardized legend entries, format_q() titles

- **Phase 04 complete**: Figure 3 (kl_dynamics) restyled — SHADE_ALPHA, thicker threshold (2.5), legend: "10–90% interval", "Median KL", "Reference threshold (0.0045)", format_q() titles

- **Phase 05 complete**: Figure 4 (distance_to_equilibrium) — new title "Convergence Error to the Analytical Equilibrium", y-axis "Equilibrium error |ē − e*|", SHADE_ALPHA, thinner conv vlines, thicker threshold, q=35 color added, format_q() labels

- **Phase 06 complete**: Figure 5 (beta_snapshots + beta_evolution) — auto-selects best seed (seed=42, |e-e*|=0.05), lighter shading (SHADE_ALPHA), format_q() titles in evolution

- **Phase 07 complete**: Figure 6a (exploitability_dynamics) — renamed labels: "Tolerance threshold", "Stability screening passed", "Approx. Nash verified", format_q() titles. New Figure 6b (exploitability_q25) for excluded low-noise case

- **Phase 08 complete**: Figure 7 (ablation_comparison) — ABLATION_LABELS/LINEWIDTHS applied, Theory line prominent (red, thick), SHADE_ALPHA, unified y-axis, x-axis formatter, format_q() titles

- **Phase 09 complete**: Figure 8 (equilibrium_recovery_dotplot) — new title, alternating grey backgrounds, Theory line #333333, labels "Per-seed estimate"/"Across-seed mean", smaller/lighter dots, closer scenario labels

- **Phase 10 complete**: Tables 1-2 — no ==?== placeholders found; q values updated to {35, 40, 55} in environment_config table
- **Phase 11 complete**: Tables 3-4 — q=25 rows excluded from final_summary (filtered to Q_VALUES only); q=35 rows now included. Fixed make_all data loader to not pre-filter by q_values (needed for exploitability_q25)

## What's next
- All 11 phases complete. `python -m paper.generator make_all` generates 11 figures + 5 tables

## Blockers
- (none)
