# Project state

Last updated: 2026-08-19

## Figure PDFs embed TrueType, not Type 3 (2026-08-19)

Owner's publisher check on the uploaded PDF: "Font DejaVuSans-Oblique is of
type Type 3". The oblique face comes from mathtext ($e^*$, $q$, $w_H$), and
matplotlib's default pdf.fonttype is 3.

- paper/generator/plots.py setup_matplotlib_style() now sets pdf.fonttype=42 and
  ps.fonttype=42, matching what tools/make_multistage_figures.py has always set.
  Applies to EVERY figure this generator produces, not just F2-F4.
- Embedding only, no layout change: the three regenerated PNGs are byte-identical
  to the previous ones, and only the .pdf files show as modified. Verified for
  convergence_main / kl_dynamics / distance_to_equilibrium too (PNG identical,
  PDF now clean).
- F2/F3/F4 regenerated: Type3 counts 3/3/4 -> 0, now Type0 + CIDFontType2
  (subsetted TrueType).
- STILL TYPE 3 -- not regenerated, deliberately out of scope. paper/figures/:
  ablation_comparison, beta_evolution, convergence_main, distance_to_equilibrium,
  effort_drift, hyperparam_sensitivity, kl_dynamics. ALL of
  overleaf_export/figures/ (stale copies; that main.tex still references
  convergence_main.pdf, not the 1x3). If any of these ship in the paper the
  check will fire again -- regenerating them is one make_all, BUT kl_dynamics and
  distance_to_equilibrium are known to render differently from their committed
  copies (pre-existing r5-era staleness, see the 2026-08-03 note), so that regen
  changes figure CONTENT and needs owner review, not just a font pass.

## F2/F3/F4 enlarged for print; F4 title and F2 row label dropped (2026-08-19)

Owner report: the three figures were too small to read in print. Text size and
two label removals only -- no data, no code path, no figure geometry changed.

- paper/generator/plots.py: PRINT_FONT_SCALE = 1.45 multiplies the FONT_SIZES
  rcParams (tick 10->14, axis label 12->17, panel title 14->20, legend 10->14)
  via a new font_scale arg on plot_convergence_main /
  plot_exploitability_dynamics / plot_equilibrium_recovery_dotplot. At 10in wide
  reproduced in a ~6.5in text block the old 10pt defaults landed near 6.5pt on
  paper.
- SIZE ONLY, NO ADDED WEIGHT. The first cut also bolded axis labels, panel
  titles and tick labels and scaled the spine/tick widths; owner: "字体过重,
  线条显得不突出" -- the chrome outweighed the curves. Reverted to default
  weights and default 0.8 line widths. Weights that were already bold before
  this change (F2 e* annotation, F4 scenario group labels) are untouched.
- F4 (dotplot) runs at DOTPLOT_FONT_SCALE = 1.30, not 1.45: nine "q=35" ticks
  plus four two-line scenario labels share one 10in axis, and at 14pt
  "Heterogeneous Cost" collides with "Heterogeneous Ability". Measured, not
  guessed -- 1.30 leaves 0.23in between ticks and 0.23in between group labels;
  1.35 is already negative. (The group labels are bold independently of this
  change, so lightening the tick weight did not raise the cap.)
- F4 in-figure title "Verified TEL-PPO Effort Estimates Relative to Analytical
  Benchmarks" REMOVED (owner: match F2/F3, which carry none; the LaTeX caption
  carries it).
- F2 rotated row label "$w_H=6.5, w_L=3.0, k=5.5e-4$" REMOVED (owner: it existed
  to tell the two weight-variant ROWS apart, and the 1x3 shows only one set,
  which the paper text states). Implemented as `col_idx == 0 and n_rows > 1`, so
  the 2x3 convergence_main keeps it.
- LEGENDS ALIGNED TO THE PLOT AREA (owner: "legend和图对齐"). All three legends
  were WIDER than the panels they sat above (F2 by 1.70in, F3 by 2.11in, F4 by
  0.84in) and anchored to the figure, so bbox_inches='tight' turned the overhang
  into the saved margin and the panels read as inset -- F2's legend alone pushed
  the left edge out 1.59in. New _axes_span / _fit_legend helpers anchor each
  legend to the panels' span and step the legend font down until it fits
  (floor 60% of the requested size). Result: F2/F3 legends 14pt -> 12pt, F4
  stays 13pt; all three now inset 0.09-0.33in inside the plot area. Legend
  spacing also tightened (_LEGEND_TIGHT: handlelength 1.6, handletextpad 0.5,
  columnspacing 1.0), which bought ~8% of the width before any font reduction.
  ncol is UNCHANGED everywhere -- F2's 3-column and F4's column-major layouts
  encode meaning (see the comments at each call site). F3 and F4 now build their
  legend AFTER tight_layout/subplots_adjust, since those move the axes the
  legend is anchored to. Saved widths dropped: F2 10.35->9.38in, F3
  10.84->9.58in, i.e. the panels themselves get more of the printed width.
- F2 x ticks thinned to every 50 updates when font_scale >= 1.3. Re-measured
  after the row label came off: 25-spacing would leave 0.011in between "100"
  and "125", so 50 stays.
- setup_matplotlib_style() now restates every rcParam _apply_font_scale can
  set, so print styling cannot leak into the next figure in the same process.
  plot_convergence_main at font_scale=1.0 skips the scaling AND the legend
  re-anchoring entirely, so the 2x3 convergence_main stays byte-identical.
- VERIFIED: pre-patch renders of all three reproduce the committed PNGs
  byte-for-byte (so styling is the only delta); post-patch convergence_main
  (2x3), kl_dynamics and distance_to_equilibrium are byte-identical to what the
  unpatched code produces (no leak). Dotplot still prints mean rel err 3.47% and
  its backing CSV regenerates unchanged -- same r9 raw landings as before.
- CAUTION: tools/regen_equilibrium_recovery_dotplot.py still calls the RETIRED
  load_polished_dotplot_final override and would silently swap F4 to polished
  values (two-player q35 mean 44.95 instead of the raw 43.5). Use
  plot_equilibrium_recovery_dotplot(df) / make_all instead. Script not touched.

## r9_cert001 wave LAUNCHED: eps=0.01 / M=65536 adopted as THE config (2026-08-03)

Owner decision after the sensitivity result: adopt the tightened certificate
for the whole matrix (combined, no eps-vs-M decomposition for now; mechanism
note: M-up alone provably does not improve accuracy -- r7 2P q35 showed M
8192->16384 worsened err 1.87->2.75 via earlier stops; M-up is the ENABLER
that keeps the verifier floor (~2.5e-3 at 65536) clear of the 0.01 gate).

- PRE-FLIGHT FIX: run_two_players.py _sample_policy_efforts had the same
  latent (M,1)x(M,) -> (M,M) broadcast as exploit_asymmetric (34 GB int64
  winners at M=65536). Fixed with an M>16384 flatten branch; legacy M<=16384
  path bit-identical. Unit test: 0.04 s/call at 65536 (50x faster than the
  quadratic 16384 path); runner smoke rc=0 with exploit_config recording
  eps=0.01/M=65536. 3P evaluator proven flat at 65536 by the sens wave;
  dc/da use the exploit_asymmetric CPU branch.
- WAVE: 65 runs, tags r9_cert001 / r9_fig7_no_stability
  (run/r9_cert001_wave.sh, results/r9_cert001/). ~29 GPU-h ~= 4 h on 8 V100s.
- REUSED (no rerun): 3P q55 + da q55 = r8_sens_eps001 (they ARE this config);
  no_exploit arm = r7_fig7_no_exploit (verifier disabled -> eps/M never enter
  training).
- COMPLETE (2026-08-03): 65/65 rc=0, ~4 h; ALL 65 verification-triggered.
  Referee re-run on true r9 profiles (first pass accidentally aggregated stale
  r8 profiles for 3P/dc/da -- sed-generated tool kept r8_unified globs;
  caught because old values reproduced verbatim; fixed with asserts):
  EVERY seed passes eps=0.01 independently (max 5.0e-3 TEL-PPO / 7.3e-3 incl
  no_stability); cells sit 1.5-6.4x above floor (was 1.4-26.5x). Matrix mean
  |err| 1.33 -> 0.91; worst cell 3.42 -> 2.29; former problem cells all fixed
  (3P q35 0.39, da q35 0.57, q55 pair 1.12/0.71); q45-type on-e* cells wobble
  within the narrower band. 2P S1 q35 SD 1.78 -> 0.34.
  BASELINE_OVERRIDES -> r9 (3P/da q55 = r8_sens_eps001); F2/F3/F4 + 1x3
  regenerated (dotplot mean rel err 3.47%); table2_unified_config.md updated
  (eps=0.01/M=65536, polish row retired). Deliverable:
  docs/tables34_r9_final.md + figures (sent). Referee artifact:
  results/one_stage_ablation/final_exploit_r9.json (n_calls incl 3P mask).

## Figures F2/F3/F4 regenerated on r7/r8 with new labels (2026-08-03)

Per docs/Figures&Tables080226.docx figure items:
- BASELINE_OVERRIDES promoted to the unified generation (paper/generator/
  config.py): two_players -> r7_state4, 3P/dc/da -> r8_unified. NOTE: gradient
  (MC-FD) runs still carry r5 tags and are NOT promoted (method-agnostic
  promote_preferred_ablations would mix PPO generations) -- figures/tables
  needing a gradient baseline require a method-aware promotion first.
- F2 convergence_main (+_1x3): MC-BR polished star REMOVED; the open circle
  (now "Verified TEL-PPO Estimate") sits ON the stopping line (x = mean
  stopped_at_update); vline renamed "Verification-triggered stopping update".
- F3 exploitability_dynamics: y-label "Estimated Exploitability"; legend
  "Stability pass" / "Verification-triggered stop".
- F4 equilibrium_recovery_dotplot: title "Verified TEL-PPO Effort Estimate
  Relative to Analytical Benchmarks"; "Seed-Level TEL-PPO outputs"; polished
  final_override RETIRED in plots.py generate_all + __main__ -- the figure now
  shows RAW verified landings (mean rel err 6.81%, honest raw picture; the
  3P/da q55 certificate-width gaps are visible, matching Table 4).
- Only these four figure files regenerated; ALL OTHER paper/figures/* remain
  r5-based and stale until a full make_all after the gradient-promotion
  decision. plots.py helpers load_polished_dotplot_final /
  _polished_two_player_means retained but unused by the canonical paths.

## Sensitivity wave (eps=0.01, M=65536) + quadratic-broadcast fix (2026-08-03)

Owner red item: 3P q55 & da q55 rerun at eps=0.01, M=65536 (tag
r8_sens_eps001; cmdlines = r8_unified + --exploit-eps 0.01 --exploit-M 65536).

- First launch: all 5 da jobs crashed at the first exploit call; 3P unaffected.
  ROOT CAUSE (latent since r5): utils/exploit_asymmetric.py samples policies
  as (M,1) while BR candidates are (M,), so _payoff_player broadcasts to an
  (M,M) matrix — ~1 GB/candidate at M=16384 (silently absorbed by 32 GB
  V100s in ALL r5/r7/r8 da+dc training verifier calls), 17 GB at M=65536 →
  OOM surfacing as an NVML assert on this broken-NVML host.
- FIX: eval_on_cpu branch for M>16384 in eval_exploitability_asymmetric +
  flat-(M,) CPU sampler — linear memory, 0.3 s/eval (~50x faster than the
  quadratic GPU path), deterministic; M<=16384 legacy path untouched
  bit-for-bit (r7/r8 reproducibility preserved).
- PAPER FOOTNOTE: in r5-r8, the da/dc TRAINING verifier's BR-candidate means
  were effectively over M^2 sample pairs while u_policy used M pairs
  (unbiased either way; asymmetric sample sizes; independent final referee
  in numpy unaffected).
- COMPLETE (2026-08-03): all 10 runs verification-triggered (streak=5).
  Certificate-width mechanism confirmed: err 3.22->1.12 (3P q55) and
  3.42->0.71 (da q55); independent referee (17.1->3.1)e-3 and (8.8->1.3)e-3
  (da lands AT the evaluator floor 0.85e-3); cost ~30-35 extra updates; SD
  tightens in both cells. Sensitivity section appended to
  docs/tables34_raw_r7r8.md (sent to owner).

## Independent final exploitability on r7/r8 raw profiles — COMPLETE (2026-08-03)

Owner review docs/Figures&Tables080226.docx removes MC-BR polishing from the
framework; owner fixed the generation to r7 (2P) + r8_unified (3P/dc/da) RAW.
New `tools/final_exploit_r7r8.py` evaluated all 75 reported raw profiles under
the independent referee (M=200,000, grid 0.25, seeds 700000+q*1000+si*7) ->
`results/one_stage_ablation/final_exploit_r7r8.json`; e* floor reused from
exploit_noise_floor.json.

Key numbers: every seed passes eps=0.03 under the referee (max 0.0218, 3P
q55); raw profiles sit 1.4x-26.5x above the floor (polished used to sit AT
it); the two wide-certificate cells (3P/da q55) read 17.1/8.8 e-3.
Deliverable sent: docs/tables34_raw_r7r8.md (new-format Tables 3 & 4, incl.
the Number-of-Exploitability-Calls column; 3P n_calls counted via the
exploitability_is_valid mask).

Remaining red items: sensitivity runs (3P q55 & da q55, eps=0.01 M=65536,
20 runs, NOT launched); figure regen per new labels (F2 no-star + circle on
the line, F3/F4 relabels); Table 1/2 text cells to fill in the docx.
## r8_unified + dual-endpoint polish + MC-BR-only — ALL COMPLETE (2026-08-03)

Owner directive (one configuration, no per-scenario selection) fully executed:

- **r8_unified wave** (30 runs, 0 failures): 3P/dc/da under the unified config
  (mean×conc head, entropy 0, no floor, M=16384, 4-dim state). ALL runs are
  true verification stops (streak=5). Finding: the eps=0.03 certificate is wide
  in effort space where payoffs are flat (band ~ sqrt(0.03/k)); q55 raw errs hit
  3.2-3.4 INSIDE the certificate band -- the old floors had been compensating.
  Realized config is constant everywhere (conc=100, var=0): every scenario
  verifies before the 200-update ramp, so the late schedule is inert.
- **Dual-endpoint polish** (stages A+B, 90 rows,
  results/one_stage_ablation/polish_per_seed_r7.json): pre-registered
  prediction CONFIRMED -- polished column collapses across all Table-3 arms
  (spread 0.001-0.012); q55 blowups polish to 0.07/0.00. Set-1 polished values
  match r5 to 0.01.
- **MC-BR-only baseline** (81 rows, mc_br_only.json): polish from ANY start
  (50 or ladder 10-90) reproduces the polished column exactly, incl. the
  asymmetric dc NE from a symmetric start. "Polished Err." = the instrument's
  own bias. TEL-PPO's evidence = raw column + certified stopping + policy
  object (see docs/ablation_narrative_preregistered.md, branch A).
- **Deliverables**: docs/r7_unified_2p_results.md (final Table 3 five-row
  dual-endpoint + Table 4 all-scenario, sent to owner),
  docs/table2_unified_config.md (unified Table 2 + realized values + data
  registry), docs/ablation_narrative_preregistered.md.
- **Open methodology decision for owner**: tighten eps uniformly (needs
  M~65536) vs uniform burn-in vs clean-rule+polish (option 3, now fully
  evidenced). Also pending: independent M=200000 final exploitability check
  for r7/r8; generator promotion; code commits (3-way split, snapshots in
  results/{r7_state4,r8_unified}/code_state.diff); r7smoke cleanup.

## r7_state4 wave: 4-dim state + unified verifier M — COMPLETE (2026-08-01)

Wave finished 2026-08-01 16:45 UTC: 23.1 h, 100/100 jobs, rc=0 everywhere.
**Red item 2 verdict: no systematic accuracy degradation under the 4-dim state**
(10 of 14 baseline cells improved; worst move +0.88 ≈ 1.1 SE at 2P S1 q35, likely
an M=16384 earlier-stop effect, not a state effect). da learned symmetry:
|e1−e2| ≈ cross-seed SD (1–2% of e*). Full per-cell table + mechanisms:
`docs/tasks/r7-state4-wave/STATE.md`. Polish + unified tables pending
(polish tool needs r7 output path first — it would overwrite the r5 rows).

Owner request (docs/Figures&Tables073026.docx red items 1/3/4): state becomes
`s_i = [q/60, k_i/1e-3, Δw/10, (l_i − l̄_{−i})/10]` for ALL scenarios; ablation
arms redone; training verifier unified at M=16384 (two-player was the only 8192,
`config/one_stage_two_players.py:110`).

- **Code**: `state_dim=4` in both agents + all four runners (3P's hardcoded
  `state_dim=3` at run_three_players.py:663 included); het-ability now feeds
  per-player states (l_gap ±5 → 4th comp ±0.5) to the shared agent, stores
  per-player transitions, and reports `effort1`/`effort2` in history/final —
  effort symmetry is now a LEARNED outcome, not architectural.
  `utils/exploit_asymmetric.py` samples each player's policy at its own l_gap
  (computed from l1/l2 internally; 0 for dc/symmetric). MC adapters + 3
  diagnostic tools updated to 4-dim. All uncommitted at launch; snapshot in
  `results/r7_state4/code_state.{txt,diff}`; suggested commit split in
  `docs/tasks/r7-state4-wave/STATE.md`.
- **Verification**: py_compile + CPU sanity + 5-path GPU smoke (tag `r7smoke*`,
  seed 999, files left in results/*/convergence/) all passed; da smoke shows
  per-player means (|e1−e2| ≈ 0.3–0.5 after 30 updates); 2P smoke confirms
  exploit_M=16384.
- **Wave**: 100 runs (2P Set1/Set2/no_stab/no_exploit ×15 each, 3P ×10, dc ×10,
  da std ×10, da v2 ×10; seeds 42–46), cmdlines byte-identical to r5 templates
  except tags `r7_state4[_std|_v2]` / `r7_fig7_no_{stability,exploit}`.
  ~182 GPU-h ≈ 24 h. tmux `r7w_gpu0..7`, flock queue; status in
  `results/r7_state4/manifest.csv`, logs in `results/r7_state4/logs/`.
  **r5 results untouched** — they are the 3-dim comparison arm for red item 2
  (fall back to 3-dim everywhere but het-ability if accuracy degrades).
- **Next after completion**: per-cell readout vs r5 (|mean−e*|, cross-seed SD,
  stop updates; da |e1−e2|); owner decision on red item 2; MC-BR polish for
  baselines AND both fig7 arms (dual-endpoint Table 3); rerun
  `tools/unified_exploitability_tables.py`. Task folder:
  `docs/tasks/r7-state4-wave/`.

## MISSING-cell evaluations + final T-A/T-C (2026-07-24)

Post-hoc follow-up (zero training): filled all three MISSING items from the
2026-07-23 session. (1) 2P q45 post-polish referee exploitability = 2.7e-5
(`results/one_stage_ablation/ablation_results_q45.json`); (2) no_exploit-arm
terminal exploitability via the shipped MC BR search on reconstructed
stochastic terminal policies = 0.0109/0.0086/0.0044 for q35/45/55
(`no_exploit_terminal_exploitability.json`; new `BetaPolicyAgent` adapter);
(3) Set-2 (8,4) MC-BR polish, 15 profiles = 47.045/36.751/30.127
(`polish_per_seed_set2.json`) with ★ markers added to `convergence_main` only.
(4) 3P per-seed (a)-leg re-measured under convention C2 — reproduces the
committed 0.0008/0.0007 exactly; the phase0 log's (a)-leg was already on the
per-seed profiles (its "24.68" is player-1's mean, not a separate projection).
Conventions locked: C1 err_of_mean for paper tables (4 columns appended to
`one_stage_claimb_summary.csv`, existing cells byte-identical, README note);
C2 per-seed player-mean 3P polish. Details: SESSION_STATE.md (2026-07-24).

## Figure regen (F2/F6/F9) + table extraction for the one-stage pipeline (2026-07-23)

**What was done.** Plotting + data-extraction session (zero training):
- `paper/figures/convergence_main.{png,pdf}` — gray line re-semantized to the
  FIRST-PASS verification update (was streak completion); new ○ raw / ★ MC-BR
  polished endpoint markers; legend renamed ("Verification update", "Raw
  estimate", "MC-BR polished estimate"). NEW file
  `paper/figures/convergence_main_1x3.{png,pdf}` (Set-1 row only) via
  `tools/regen_convergence_main_1x3.py`.
- `paper/figures/exploitability_dynamics.{png,pdf}` — y-label "Raw profile
  exploitability" (data source verified = in-training raw-profile MC
  exploitability), 4-entry compact legend.
- `paper/figures/equilibrium_recovery_dotplot.{png,pdf}` — new title/y-label,
  visible seed markers, "Low-cost/High-cost agent" labels (mapping verified),
  black het-cost theory lines.
- Tables T-A (ablation), T-B (as-built config), T-C (merged results) extracted
  from artifacts and delivered in the session reply; verified by a 6-agent
  adversarial pass.

**Key discovery.** The canonical r5_sampled PPO baselines (2P/3P/dc) trained
under the **theory-align-v2 override**: entropy 0, LR 5e-5→2e-5, clip
0.20→0.15, 1 epoch/update, `ActorCriticMeanConc` net. `docs/ONE_STAGE_ASBUILT.md`
§Q4's optimizer table is wrong for these runs (flagged there and in
SESSION_STATE.md; doc not edited). da baseline (`r5_sampled_std`) = standard arm.

**Known issues / next steps.** (i) 2P q45 post-polish exploitability MISSING —
extend the phase-1 referee to q45. (ii) `no_exploit` arm has no exploitability
artifact (by design) — post-hoc eval would fill it. (iii) plots.py +
__main__.py edits uncommitted; split code vs artifact commits per repo rules.
Full detail: SESSION_STATE.md (2026-07-23 section).

## Equilibrium-recovery dot plot (Figure 6): rebuilt from Claim-B polished values (2026-07-20)

**What was done.** Regenerated `paper/figures/equilibrium_recovery_dotplot.{png,pdf}`
(and backing `paper/data/equilibrium_recovery_dotplot.csv`) so the markers show the
Claim-B MC-BR **polished** efforts (which land on e*) instead of the previous **raw**
PPO landings (which sat below e*). The drawing style is unchanged.

- `tools/one_stage_polish_per_seed.py` — reproduces `phase0_verify.py`'s per-seed
  polish (same POL config, reading, start point, per-cell seed policy) and PERSISTS
  the per-seed polished efforts to `results/one_stage_ablation/polish_per_seed_all.json`
  (45 rows: 2P q35/45/55, 3P/dc/da q35/55, 5 seeds each). CPU-only, ~108 min.
  Cross-seed means reproduce `results/phase0_verify_20260701_1941.log` exactly
  (2P q35→44.95, 3P q35→24.68, dc q35→38.04/27.66, da q35→46.45, …). 2P q45 has no
  log reference (outside the Claim-B set) but uses identical machinery.
- `paper/generator/plots.py::plot_equilibrium_recovery_dotplot` — added a
  backward-compatible `final_override` param that swaps the data source only; the
  drawing logic is untouched.
- `tools/regen_equilibrium_recovery_dotplot.py` — thin wrapper for a one-off regen.
  Mean relative error on the new figure = 0.47%.

**Wired into the generator.** `plots.load_polished_dotplot_final()` reads
`polish_per_seed_all.json` and maps it to the plot's `final_override` columns.
`generate_all_figures` (→ `make_all`) and `python -m paper.generator plot
equilibrium_recovery_dotplot` now use it by default, so the canonical figure is the
polished one. If the polished JSON is absent, both fall back to the raw PPO landings
with a printed note (no crash). The regen tool now reuses the same loader.

## Ablation figure (Figure 7): full rebuild from Claim-B data (2026-07-20)

- Rewrote `plot_ablation_comparison` in `paper/generator/plots.py` and regenerated
  `paper/figures/ablation_comparison.{png,pdf}` + `paper/data/ablation_comparison.csv`.
  Implements the phase08 spec (`docs/tasks/paper-figures-tables-revision/phase08.md`)
  items a–h plus the owner's additions i (summary table) and j (no-polish comparison).
- **Root cause fixed**: the old figure rendered **only TEL-PPO** — the ablation arms
  live on disk as `r5_fig7_no_stability` / `r5_fig7_no_exploit`, which never matched the
  hardcoded filter keys `no_cheap_gate` / `no_exploitability`. Phase08 was marked
  "complete" but the arms were silently absent. New code remaps disk tags → canonical
  keys (`_ABL_DISK_TO_CANON`).
- **Design** (owner-approved draft): 3 stacked q-panels (q = 35/45/55) with a **broken
  x-axis** — left = convergence detail (0→0.55×10⁶), right = the non-terminating
  `no_exploit` tail (→6.25×10⁶). Per-seed traces + 95% CI bands, prominent red theory
  line (lw 3.0), TEL-PPO thick / ablations thin, unified y-axis, x in ×10⁶, titles
  without `.0`. TEL-PPO endpoint carries a **no-polish (○ raw) vs MC-BR-polished
  (★ Claim-B)** fork. Bottom **summary table**: terminal |ē−e*|, final exploitability,
  NC (non-convergence) rate, time-to-verification per arm.
- **q note**: data exists at q = 35/**45**/55 (not 40 as the phase08 text says — no q=40
  runs exist anywhere). Owner confirmed 45. The reshared reference image (q=25/40/55,
  effort ~50–100) is a *style* template from a different/older parameterization, not data.
- **Key numbers** (all from `results/`): terminal |ē−e*| TEL-PPO polished 0.51/0.11/0.17,
  no_stability 2.57/0.24/1.30, no_exploit 0.85/3.31/0.58; NC rate 0/0/100%;
  `no_exploit` runs to 1500 updates (never verifies) and at q=45 drifts *away* from e*
  (35.5→32.0). Polished landings from `results/one_stage_ablation/ablation_results.json`
  (q35 44.95, q55 28.76; q45 not run → ★ omitted, table shows raw).
- **Not committed** (awaiting review). NOTE per repo commit rules: this bundles a **code
  edit** (`plots.py`) + **figure/data regen** — split into two commits. Also `plots.py`
  already carried unrelated uncommitted effort_drift/kl_dynamics edits from prior work.

## Convergence-main figure: Claim-B refresh (2026-07-20)

- Regenerated `paper/figures/convergence_main.{png,pdf}` (2×3 grid, weight
  variant × q) from the 5-seed `r5_sampled` two-player baseline runs — the
  **Claim-B canonical baseline** (same runs behind
  `results/one_stage_claimb_summary.csv`). Style/layout otherwise unchanged.
- Changes to `plot_convergence_main` in `paper/generator/plots.py`:
  1. **Vertical dashed convergence line** now marks the *verified* convergence
     update (`get_verified_convergence_step` → mean `stopped_at_update` over the
     exploitability-verified seeds), i.e. the first update satisfying the
     criterion. Matches Claim-B "Conv. Update (verified)" exactly: two-player
     baseline q35→55, q55→87 (was the old `get_convergence_step` effort-δ path,
     which never fired here because `min_steps=100` > run length).
  2. **e\* value annotated** on each panel (red `$e^*=…$` at the right edge of
     the theory line).
  3. **X-axis unified** across all six panels and relabeled **Training Updates**
     (tick = step/4096). Range extends to the longest (high-noise q=55) baseline
     run so the extra updates high noise needs are visible; x_max restricted to
     the baseline ablation so long `eps_*/pat_*` sweep arms don't stretch it.
  4. **Legend** moved out of the top-left panel to a single figure-level legend
     at the top-right, outside the grid.
- Internal plotting stays in step space, so `paper/data/convergence_main.csv` is
  unchanged. Not committed (awaiting review). NOTE: code edit + figure regen —
  split into two commits if committing (per repo commit rules).

## KL-dynamics figure: unified x-axis (2026-07-20)

- Regenerated `paper/figures/kl_dynamics.{png,pdf}` ("KL Divergence across Noise
  Levels") with a **shared training-step x-axis** across the three q panels
  (all 0→442368, the global max; each curve still ends at its own data extent).
- Data source confirmed already correct: the figure draws from the 5-seed
  `r5_sampled` two-player baseline runs — the **Claim-B canonical baseline**, the
  same runs the claim-b summary (`results/one_stage_claimb_summary.csv`) is built
  from. `paper/data/kl_dynamics.csv` is **byte-identical** after regen (no data
  change) — only the axis limits changed.
- Code: `plot_kl_dynamics` in `paper/generator/plots.py` now computes
  `global_step_max` over positive-KL rows and applies `ax.set_xlim(0, global_step_max)`
  to every panel. All other style (colors, fonts, layout, y-axis, legend,
  threshold/median/band) unchanged. Regen via `python -m paper.generator plot kl_dynamics`.
- Not committed (awaiting review). NOTE: this bundles a code edit + a figure
  regen — split into two commits if committing (per repo commit rules).

## Multi-stage task launched: theory audit + q_crit config validation (2026-07-09)

- New task `docs/tasks/multistage-tel-ppo/` for the multi-stage plan
  (`docs/Experiments Plan_Multi-stage.md`), under **owner-decided Claim-B framing**
  (PPO = smoothed-candidate generator; DP verifier certifies; exploring starts with
  full approximate-MPE claim; Δ_t(d) one-step deviation gaps as the primary
  certificate; closed-form terminal integration via F_xi).
- **Phase 01 complete (zero GPU).** Audited the owner's two-stage closed-form
  derivation (main `8e9433e`): g2(d), q_SOC, g1, participation algebra all confirmed
  numerically. **One math error found and corrected**: stage-1 SOC misses the V2*
  kink term at d=0 — correct curvature is −2k + ΔW²/(32kq⁴) (repo convention),
  negative iff q > q_SOC, NOT unconditional. Numerically tight: at q=40 the
  symmetric candidate is a local minimum. Full audit + Word-doc errata:
  `docs/technical/two_stage_benchmark_audit.md`.
- **Validity region: q_crit = 41.83** (binding: SOC) for canonical parameters
  (w_h=6, w_l=2, k=1/3500, c=ke², ē=100). **q=35 and q=40 are INVALID.**
  Canonical q_list = [45, 50, 55]. Targets at q=50: g1=46.67, g2(0)=70.00,
  E[g2]=g1 exactly, U_eq=2.678.
- New: `utils/theory_multistage.py` (closed forms + validation scan incl.
  zero-effort deviation), `config/multi_stage_two_players.py` (canonical config,
  supersedes `config/two_stage_two_players.py` for this task; `validate()` raises
  on invalid q), `tools/verify_two_stage_benchmark.py` (all checks PASS).
- Next: phase02 env rewrite (`envs/two_stage_env.py` is a different game — per-stage
  prize flow, logit model, expected-value rewards, no gap state; reference only).
- NOTE: main commit `8e9433e` carries a mismatched message ("Update print
  statement…") for the plan-doc edit; noted, no history rewrite.

## Claim-A κ-continuation 5-seed run + Gate C (2026-07-07)

- Pilot #2 (seed 42) completed 2026-07-04: clean exit, all 6 ladder stages to
  κ=400, 0 forced advances, final 24.30 — borderline vs Gate C, owner chose the
  5-seed run. **Gate C scored on 5 FRESH seeds 43–47** (pilot 42 excluded to avoid
  pilot-selection bias; owner-confirmed); tmux `c3_s43`–`c3_s47`, GPUs 0–4, params
  byte-identical to pilot #2.
- **All 5 seeds clean** (exploitability stop, forced_advances=0, exploit
  0.004–0.011). **Gate C verdict: BORDERLINE — no branch fired.** Final-snapshot
  metric: cross-seed mean 24.034 (|err| 3.86%, KILL line is 4%), std 0.688
  (success ≤0.5, KILL >1.0).
- **Decomposition is decisive though**: snapshot spread ≈ within-run κ=400
  diffusion sampled at 1 update ("done" stops on its first update). Time-averaged
  (κ=400 stage, last 30 upd): cross-seed mean **24.29, std 0.146** (SE 0.065) —
  variance solved (~11× tighter than c2's 1.67), but mean misses the 24.5 success
  line by ~3 SE. **Undershoot is systematic bias, not noise; strong Claim A is
  dead in this parameterization.** All 6 runs land 24.1–24.4, ~0.4 below
  μ*(400)=24.7.
- Recommendation: adopt Claim B final form (PPO → smoothed equilibrium μ*(κ);
  continuation tracks it reproducibly; MC-BR bridges the last 0.7). No more GPU
  on Claim A without new variance-reduction evidence — this is now four
  concordant negatives (r5, c2, design analysis, c3 5-seed).
- Details + per-seed table: `docs/tasks/claim-a-nonlocking-continuation/STATE.md`.
  Data: `results/three_players/convergence/ppo_3p_q35.0_seed{42..47}_c3_cont_convergence.json`.

## Phase-0 response doc audit + corrections (2026-07-02)

- Audited `docs/phase0_response_to_revision_plan.md` against code, convergence JSONs,
  `results/phase0_verify_20260701_1941.log`, theory formulas, and git history. All §3/§6.3
  numbers, Finding A/B values, attribution finding, and provenance claims verified correct.
- **Corrected 4 issues found by the audit** (doc + upstream sources, no result files touched):
  1. §6.4 mechanism ("κ ramp freezes the mean at trigger") was contradicted by the stored
     mode/mean trajectories — mode moved +2.7..+7.2 units toward e* after trigger. Rewritten
     as "premature trigger + fixed-length ramp window too short; per-seed travel variance is
     the 6.5x std source". Same fix in `SESSION_STATE.md` and
     `docs/tasks/component2-mode-conc-retrain/STATE.md` (old wording RETRACTED, do not resurrect).
  2. Leg (c) description: implementation is `drift<0.1 OR within-trajectory SE<0.1`
     (`tools/phase0_verify.py:45`), not "cross-seed SE"; 4/6 cells passed via the SE branch.
     Fixed in response doc §3, `SESSION_STATE.md` §B, and the verify script docstring
     (comment only — criterion code untouched).
  3. §6.2 flag count: 4 companion flags (incl. `--kappa-schedule`), not 3.
  4. Finding A max|mean−mode|: 0.19–0.30 per run (at conc≈210–270), not flat 0.20.
- Known issue: leg (c) as implemented is a weak sanity check (SE branch nearly always passes);
  acceptance weight rests on legs (a)+(b). Documented, not changed.
- Next: owner decision on Claim B framing (response doc §7); if Claim A is still pursued,
  a redesigned retrain (longer/slower ramp or effort-proximity trigger) needs new authorization.

## Claim-A dev-trigger retrain — Phase A screen (2026-07-02)

- New task `docs/tasks/claim-a-dev-trigger-retrain/` for a redesigned Claim-A attempt
  (replace the trigger observable: payoff-gain → best-response distance).
- **Phase A (zero-GPU screen, `tools/claim_a_phase_a_screen.py`) complete.** Findings:
  the trigger observable is fixable (A1: deterministic-mean BR-distance is a clean
  6.5→0.5 signal near e*), BUT (A2) the signal collapses at explore-κ and must be
  defined vs the deterministic mean, and (A3) raising κ freezes the climb — corroborated
  by r5 (raw PPO stalls at 22.99 on the full 6M budget, no κ lock). The 2-unit
  undershoot is a PPO-dynamics property, not a trigger/schedule one.
- **Gate A recommendation: lean STOP / adopt Claim B.** A Component-2-style GPU retrain
  would most likely reproduce the stall. One non-locking redesign lever remains but
  fights the r5 stall; needs its own authorization. Details:
  `docs/tasks/claim-a-dev-trigger-retrain/phase01_findings.md`.

## Claim-A non-locking continuation — design analysis KILL (2026-07-02)

- Owner authorized the non-locking redesign (Gate A branch (ii)); new task
  `docs/tasks/claim-a-nonlocking-continuation/` created; predecessor task closed.
- Phase01 (`tools/claim_a_continuation_design.py`, zero GPU) measured the
  exploration-smoothed equilibrium curve **μ*(κ)**: 22.59 (κ=20) → 23.96 (κ=200) →
  24.74 (κ=400). Component-2's κ_top=200 structurally capped its target at ~24;
  r5's 22.99 "stall" sits in [μ*(20), μ*(60)] — raw PPO converges to the smoothed
  equilibrium, not a failure.
- **Pre-registered kill fired**: all velocity deaths in the c2 ramps occurred with
  HEALTHY approx_kl → gradient-SNR physics (diffusion within ~1.5 units of target at
  batch 4096); optimizer floors cannot fix it. Predicted landing spread ±1.5–2
  violates Gate C in expectation. **Recommendation: STOP before GPU; adopt Claim B
  upgraded by the μ*(κ) curve** (candidate paper figure). Claim A now has three
  concordant negative results; do not resurrect without variance-reduction evidence.
- Owner decision pending at the kill gate (accept STOP vs overrule).
- **UPDATE 2026-07-02**: owner OVERRULED the kill → adaptive-batch continuation variant
  (16× batch on the κ ladder to shrink the diffusion band ∝ 1/√B). Implemented as
  `--kappa-continuation` (additive, default off), CPU+GPU smoked, **1-seed pilot
  running** (tmux `c3_pilot`, GPU 0, tag `c3_cont`, 20M episodes, ~1–1.5 days).
  Pre-registered pilot gate before any 5-seed spend — see
  `docs/tasks/claim-a-nonlocking-continuation/{phase02.md,STATE.md}`.
  NOTE: host NVML broken (nvidia-smi unusable); torch CUDA verified fine — monitor
  via torch.cuda.mem_get_info.
- **RECHECK 2026-07-03**: pilot #1 hit a budget bug — base config `max_updates: 1500`
  caps the run at 1500×4096 = 6.144M STEPS, but the adaptive 65536-step tail updates
  exhausted that in only 417 updates, truncating the ladder at κ=100 (never reached
  κ=200/400). Fixed with `--episodes 34000000 --max-updates 9000 --cont-max-hold 120`;
  pilot #2 relaunched (tmux `c3_pilot2`). Salvaged partial band data (stages 0-2):
  band shrinks 1.07→0.72→0.55 as batch grows 4096→16384→65536, mean tracks μ*(κ) —
  both gate metrics borderline, decision rides on the truncated κ=200/400 stages.
  Truncated pilot #1 preserved at
  `docs/tasks/claim-a-nonlocking-continuation/phase02_pilot_truncated_seed42.*`.
  CAUTION for any adaptive-batch run here: `max_updates` caps by STEPS at base batch —
  always raise it when later updates use a larger batch.
- **PILOT #2 COMPLETE (2026-07-04)**: clean exit, all 6 ladder stages to κ=400,
  0 forced advances, final mean 24.30 (gap 0.70), κ=400 band 0.59. **Borderline** —
  misses Gate C (mean ≥24.5, band ≤0.5) by ~0.2–0.3 / ~0.1; not a kill (band ≪1.0,
  mean ≥24.0). 16× batch bought only ~2× band (half the band is policy diffusion).
- **PHASE03 LAUNCHED (2026-07-06)**: owner chose the 5-seed run. Gate C scored on
  5 FRESH seeds 43–47 (pilot 42 excluded → no pilot-selection bias; owner-confirmed).
  tmux `c3_s43`–`c3_s47`, GPUs 0–4, params byte-identical to pilot #2. ETA ~18h.
  Gate C: PASS mean ≥24.5 AND cross-seed std ≤0.5; KILL std >1.0 OR mean <24.0.

## Current status
- **Parameter overhaul**: All configs updated to match `docs/experiment_config_040726.md`
- **Concentration fix (major)**: `--override-conc-ramp-warmup 200` resolves q=45/55 convergence
  - Root cause: theory_align_v2 concentration ramp froze policy in ~20 updates, before effort reached e*
  - Fix: extend warmup from 20→200, giving policy time to descend to e* before concentration rises
  - Results (Round 2, 5 seeds, Metric B): q=35 rel 4.3%, q=45 rel 2.1%, q=55 rel 2.4%
  - Old entropy_end_0.002 files archived to `results/two_players/convergence/_archive_pre_warmup_fix/`
- **Figure pipeline**: All 12 figures regenerated with warmup=200 results
- **Pending**: q=35 seeds 44/45, q=45 seed 44 retrying (OOM from parallel run)
- Two-stage runner deferred to separate task

## Figure pipeline (2026-04-09)

### Step 0: Style gate — PASSED
- Updated `paper/generator/config.py`: Wong palette, IEEE/ACM font sizes (8/9/10pt),
  TrueType embedding (pdf.fonttype=42), horizontal-only grid
- Gate checks: no Type 3 fonts, PDF width 6.70" (within 6.75±0.05"), all Wong colors verified
- Output: `paper/generator/output/style_test/style_test.{pdf,png}`

### F1: Equilibrium Recovery (hero) — COMPLETE
- Patched `run_registry.py`: `BASELINE_ALIASES = {"entropy_end_0.002"}` (line ~283)
  so q=45/55 Set 1 runs classified as baseline
- All 5 experiment groups × all q values plotted (14 conditions total)
- Bootstrap percentile 95% CI (n=5, n_resamples=2000)
- All seed counts = 5, all CI widths < 5 effort units
- q=55 2P Set 1 gray band + "See Fig. 2" annotation
- Output: `paper/generator/output/figures/F1_equilibrium_recovery.{pdf,png}`

### F2: EU Landscape & Gradient Signal — COMPLETE
- 3 panels: (a) EU landscape, (b) symmetric gradient, (c) stall-point bars at e=36
- Theory values verified: e*, gradient at e*=0, gradient at 36 matches plan
- Output: `paper/generator/output/figures/F2_eu_landscape.{pdf,png}`

### F3: Training Diagnostics — COMPLETE
- 2×3 grid: KL (top) + exploitability (bottom) × q=35/45/55
- Output: `paper/generator/output/figures/F3_training_diagnostics.{pdf,png}`

### F4: Distance to Equilibrium — COMPLETE
- Single-column, log-scale |e-e*| with terminal gaps (Metric B): q=35: 1.96, q=45: 0.74, q=55: 0.68
- Output: `paper/generator/output/figures/F4_distance_to_equilibrium.{pdf,png}`

### F5: Ablation — BLOCKED (no data)

### Appendix (8 figures) — ALL COMPLETE
- FA_2p, FA_set2, FA_gvp, FA_3p, FA_het, FA_beta, FA_snap, FA_drift
- All in `paper/generator/output/figures/`

## Figure Manifest (Phase 3 complete)

| fig_id | filename | width | fonts | status |
|--------|----------|-------|-------|--------|
| F1 | F1_equilibrium_recovery.pdf | 6.70" | TT | done |
| F2 | F2_eu_landscape.pdf | 6.77" | TT | done |
| F3 | F3_training_diagnostics.pdf | 6.69" | TT | done |
| F4 | F4_distance_to_equilibrium.pdf | 3.20" | TT | done |
| FA_2p | FA_2p_convergence.pdf | 6.69" | TT | done |
| FA_3p | FA_3p_convergence.pdf | 3.20" | TT | done |
| FA_beta | FA_beta_evolution.pdf | 6.67" | TT | done |
| FA_drift | FA_drift_post_convergence.pdf | 6.69" | TT | done |
| FA_gvp | FA_gvp_comparison.pdf | 6.69" | TT | done |
| FA_het | FA_het_convergence.pdf | 6.69" | TT | done |
| FA_set2 | FA_set2_convergence.pdf | 6.69" | TT | done |
| FA_snap | FA_snap_beta_pdf.pdf | 6.70" | TT | done |

## Parameter overhaul (2026-04-08)

Updated all experiment configs to new parameters that satisfy SOC and participation constraints.

### Config changes applied

| Experiment | k | (w_h, w_l) | q_list | effort_range |
|---|---|---|---|---|
| 2P Set 1 | 0.00055 | (6.5, 3.0) | [35, 45, 55] | [0, 100] |
| 2P Set 2 | via CLI: --k 0.0006 --w_h 8 --w_l 4 | (8, 4) | [35, 45, 55] | [0, 100] |
| 3P | 0.001 | (6.5, 3.0) | [35, 55] | [0, 100] |
| Diff Cost | k1=0.0004, k2=0.00055 | (8, 5.5) | [35, 55] | [0, 100] |
| Diff Ability | 0.0005 | (6.5, 3.0) | [35, 55] | [0, 100] |
| Two Stage | 0.0004 | (6.5, 3.0) | [25, 40, 55] | stage1=[0,100], stage2=[0,100] |

### Theoretical equilibria (verified)

| Experiment | q=35 | q=45 | q=55 |
|---|---|---|---|
| 2P Set 1 | 45.45 | 35.35 | 28.93 |
| 2P Set 2 | 47.62 | 37.04 | 30.30 |
| 3P | 25.00 | — | 15.91 |
| Diff Cost | e1=38.03, e2=27.66 | — | e1=26.54, e2=19.30 |
| Diff Ability | 46.43 | — | 30.37 |

### Files modified
- `config/one_stage_two_players.py` — k, q, q_list, effort_range
- `config/one_stage_three_players.py` — k, q, q_list, effort_range
- `config/one_stage_different_cost.py` — w_h, w_l, q, q_list, effort_range
- `config/one_stage_different_ability.py` — k, q, q_list, effort_range
- `config/two_stage_two_players.py` — effort_bounds_stage2
- `paper/generator/config.py` — per-experiment THEORY_PARAMS, Q_VALUES, updated e_star defaults
- `paper/generator/extract.py` — per-experiment theory param lookup
- `paper/generator/metrics.py` — per-experiment theory param lookup
- `paper/generator/tables.py` — per-experiment q_values and theory params
- `paper/generator/__init__.py` — export new symbols

### Results deleted
- All convergence JSONs, logs, and summary CSVs from previous parameter runs

## Previous critical finding: interior NE validity (2026-03-28)

Still relevant. The new parameters were chosen to satisfy the participation constraint:
- q_crit(2P, k=0.00055) = sqrt(2*3.5/(16*0.00055)) = 28.2 → q=35 passes
- q_crit(3P, k=0.001) = sqrt(3*3.5/(16*0.001)) = 25.6 → q=35 passes
- All experiments now use q >= 35, satisfying both SOC and participation constraint

## Task status

| Task | Status | Notes |
|------|--------|-------|
| parameter-overhaul | **complete** | All configs match experiment_config_040726.md |
| two-stage-runner | deferred | Config and env exist, runner needs to be built |
| paper-figures-tables-revision | stale | Needs re-run after new experiment data |
| runner-refactor | deferred | Post-project cleanup |

## Phase 2 (Metric B Migration) — Complete

### Loader schema (two formats)

| format | experiments | policy_mean_effort | sample_effort_mean |
|--------|------------|-------------------|-------------------|
| flat | two_players, three_players | from JSON `policy_mean_effort` field (deterministic Beta mean) | `(agent1_effort + agent2_effort) / 2` (sample rollout averages) |
| nested | different_cost, different_ability | `(agent1_effort + agent2_effort) / 2` (agents store policy means directly) | NaN (not recorded by nested runners) |
| flat (gradient) | all (gradient method) | `(agent1_effort + agent2_effort) / 2` (deterministic, same as sample) | same as policy_mean_effort |

### Changes

- Renamed column `effort_mean` → `sample_effort_mean` (4 files, 48 call sites)
- Switched all paper-reporting metrics from sample_effort_mean to `policy_mean_effort`:
  - `extract.py`: effort_error (lines 479, 527), convergence step detection (line 608)
  - `metrics.py`: effort_series (line 391)
  - `tables.py`: final summary Mean/Std (lines 348-349, 368-369)
  - `plots.py`: best-seed selection (line 907), F4 distance curve (line 1331), dotplot (lines 1657, 1772), all trajectory plots (lines 1056, 1073, 1212, 1226, 1229)
- Fixed nested loader (extract.py): `policy_mean_effort` computed from agent policy means, `sample_effort_mean` set to NaN
- Fixed flat loader (extract.py): raises ValueError for missing `policy_mean_effort` in PPO runs; falls back to agent average for gradient runs
- All convergence trajectory plots now use `policy_mean_effort` (removes sample noise)

### Verification

- Schema verification passed (verify_loader_schema.py): 6/6 assertions
- Metric B sanity check passed (verify_metric_b.py): 6/6 assertions (flat gap=0.362, nested gap=2.233)

### Regenerated artifacts

- Tables: final_summary.tex, convergence_comparison.tex, summary_metrics.tex, ablation_results.tex, environment_config.tex
- Data: convergence_main.csv, equilibrium_recovery_dotplot.csv, distance_to_equilibrium.csv, + 6 others
- Figures: convergence_main.pdf, equilibrium_recovery_dotplot.pdf, distance_to_equilibrium.pdf, + 6 others

### Verified numbers

| q | decision doc | regenerated table | match? |
|---|-------------|-------------------|--------|
| 35 | 4.3% | 4.30% | yes |
| 45 | 2.1% | 2.08% | yes |
| 55 | 2.4% | 2.36% | yes |

### Known items NOT done this round

- `docs/metric_diagnosis.md` — intentionally preserves old-metric context
- `docs/round2_metric_decision.md` — intentionally preserves old-vs-new comparison

### Commit suggestion

`refactor: switch paper reporting to Metric B (policy_mean_effort[-1])`

## Conc-ramp warmup port: three_players (in progress)

Ported concentration ramp logic from `run_two_players.py:889-910` to `run_three_players.py`.
Added CLI flag `--override-conc-ramp-warmup`. Default stays at 20 (as in theory_align_v2 defaults block).

### Agent attribute mapping (two_players vs three_players)

| attribute | PPOTwoPlayersBandit | PPOThreePlayersBandit | match? |
|-----------|--------------------|-----------------------|--------|
| `agent.net.conc_min` | yes (ActorCriticMeanConc:80) | yes (ActorCriticMeanConc:100) | identical |
| `agent.net.conc_scale` | yes (:81) | yes (:101) | identical |
| `agent.net.conc_max` | yes (:82) | yes (:102) | identical |
| `agent.opponent_policy.conc_min` | yes (deepcopy of net) | yes (deepcopy of net) | identical |
| `agent.cfg.theory_align_v2_var_coef` | yes (PPOConfig:145-146) | yes (PPOConfig:167-168) | identical |

All attributes match 1:1. The ramp code is an exact copy.

### Not yet ported (pending results from 3p)

- `run_different_cost.py`: uses TWO agents (agent1, agent2) — ramp needs to apply to both. Deferred.
- `run_different_ability.py`: single shared agent, similar to two_players. Deferred.

## Next steps
1. Run gradient baselines for all experiment types with new parameters
2. Run PPO experiments (5 seeds each) for all q values
3. Two-player Set 2 via CLI flags: `--k 0.0006 --w_h 8 --w_l 4`
4. Regenerate paper artifacts after results are collected
5. Build two-stage runner (separate task)

## Figure fix: e* label placement in convergence_main (2026-07-29)

Review comment on `docs/Figures&Tables07272026.docx` (Figure 2): the `e* = 28.93`
label in the q=55 panel sat on top of the agent trajectories.

Cause: `plot_convergence_main` pinned the label to the right edge just above the
theory line. All panels share the longest run's x-limit, so that slot is only
empty for panels whose run ends well short of the right margin. q=55 IS the
longest run and is still descending toward e* at the margin, so its label landed
on the curve; q=35/45 end early enough that their right edge is blank.

Fix (`paper/generator/plots.py`): the annotation moved after the trajectories are
drawn and now picks its slot from the panel's curve extent — panels reaching
>= 85% of the global x-max put the label below the theory line at mid-panel
(the band under e* is free, since curves approach the equilibrium from above),
all others keep the original right-edge slot.

Regenerated in place: `convergence_main.{png,pdf}`, `convergence_main_1x3.{png,pdf}`.
Only the two q=55 panels changed; the other four are pixel-identical in layout.

## Figure fix: dotplot legend order (2026-07-29)

Review comment on the same doc: swap "Seed-level estimates" and "High-cost
agent" in the `equilibrium_recovery_dotplot` legend.

The handles were appended in construction order and matplotlib fills legend
columns top-to-bottom, so with ncol=3 the low-/high-cost pair was split across
both the row and column axes. `plot_equilibrium_recovery_dotplot` now sorts the
handles through an explicit `_legend_order` before `ax.legend`, giving

    row 1: Theory e*        | Across-seed mean | Seed-level estimates
    row 2: High-cost agent  | Low-cost agent

Regenerated in place: `equilibrium_recovery_dotplot.{png,pdf}` (via
`load_polished_dotplot_final()`, the same Claim-B polished override `make_all`
uses — not the raw-landing fallback). Legend only; no marker or datum moved.

### Known issues / next steps
1. Estimator unification (Tables 3 & 4) is the blocking item — see the review
   doc. `det`/`in`/`mc`/`a` are mixed within single columns; the ask is to move
   everything to independent MC with fresh draws
   (`utils/mc_br_polish.py:211 exploitability_frozen_profile`). Expect TEL-PPO's
   exploitability to rise from ~1e-5 into the MC noise floor (~1e-3), which will
   shrink or erase the "two orders of magnitude" claim.
2. Calibration check not yet run: feed the analytic e* to
   `exploitability_frozen_profile` (true EXP = 0) to measure the estimator's
   noise floor before rewriting the tables.
3. Remaining review comments unaddressed: Table 3 column split (Final Effort
   mean±SD vs separate Absolute Bias), Verification Update ± SD, Table 2
   metadata/entropy confirmations, Table 4 exploitability formatting as mean±SD.
   Both figure comments (q=55 e* label, dotplot legend order) are done.

## Estimator unification for Tables 3 & 4 (2026-07-29)

Acted on the review's request to drop the mixed det/in/mc/a reporting and use one
independent MC estimator everywhere.

New tools (both CPU-only, run in tmux):
- `tools/exploit_noise_floor.py` -> `results/one_stage_ablation/exploit_noise_floor.json`
  Evaluates `exploitability_frozen_profile` at the analytic e*, where true EXP is
  exactly 0, so the return value is the estimator's own error. Also sweeps M on
  the two-player q=35 cell.
- `tools/unified_exploitability_tables.py` -> `results/one_stage_ablation/unified_exploitability_tables.json`
  Recomputes every Table 3 and Table 4 row per seed under that one estimator
  (M=200000, grid 0.25, fresh seeds, CRN shared across the four Table-3 arms).

Clean tables written to `docs/tables34_unified.md`.

### Findings

- Noise floor is 2.8e-04 to 1.5e-03 across cells at M=200000, and decays like
  1/sqrt(M): M=800000 only reaches ~7.8e-04. The old TEL-PPO numbers
  (8.6e-05 / 2.8e-05 / 1.2e-05, deterministic referee) sit 5-70x BELOW the floor,
  so they were never comparable to the MC rows they were tabulated against.
- Under the unified estimator TEL-PPO reads 1.09e-03 / 1.02e-03 / 0.87e-03,
  i.e. 0.90x / 1.37x / 1.44x the floor. The "two orders of magnitude" claim does
  not survive; TEL-PPO is at the floor, which is a ceiling on what the instrument
  can resolve, not a measured gap.
- Arm ordering survives (monotone at q=35 and q=55) but separations are 2-8x with
  comparable per-seed SDs.
- The effort columns are estimator-independent and DO separate the arms:
  TEL-PPO's cross-seed SD is 0.02-0.03 vs 0.60-3.53 for every ablation (40-100x
  tighter). That is what the ablation actually establishes.
- Corrected a reporting error in the old Table 3: the "Final Effort" cell for
  `w/o stability screening` and `w/o exploitability verification` held the BIAS
  (2.57/0.24/1.30 and 0.85/3.31/0.58), not the effort. Actual efforts are
  42.88/35.11/30.22 and 44.60/32.05/29.51.
- Added Verification Update mean +/- SD (was mean only); the
  `w/o exploitability verification` arm has none - all 5 seeds hit the
  1500-update budget (`stop_reason = max_updates`).

### Next steps
1. Decide how the paper frames the ablation now that exploitability is at the
   floor - recommend leading with the effort-SD result.
2. Remaining review items: Table 2 metadata/entropy corrections (see
   `docs/table2_metadata_audit.md`), Table 2 row splits for het-ability.
