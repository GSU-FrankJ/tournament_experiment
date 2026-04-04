# Hyperparam Sensitivity Figure: Redesign Discussion

**Figure**: `paper/figures/hyperparam_sensitivity.png`
**Generator**: `paper/generator/plots.py:plot_hyperparam_sensitivity()`
**Current layout**: 2 rows (Clipping epsilon, Patience p) x 3 columns (q=35, 40, 55)

## Current figure description

- Top row: sweep over PPO clipping epsilon (0.01, 0.03, 0.10, 0.20 vs baseline 0.05)
- Bottom row: sweep over exploitability patience (1, 3, 10 vs baseline 5)
- Each panel: effort vs training steps, with per-seed thin traces + aggregate mean + CI band
- Baseline (black) runs to 5M+ steps; sweep variants are colored lines

## Problems

### P1: Extreme x-axis scale mismatch

Sweep runs are early-stopped by the exploitability mechanism. Their x-range is
tiny compared to the baseline:

| Variant     | q=40 max step | q=55 max step | Baseline max step |
|-------------|---------------|---------------|-------------------|
| pat_01 (p=1)   | 74k       | 156k          | 5.2M              |
| eps_020 (e=0.20) | 401k    | 401k          | 5.2M              |
| pat_10 (p=10)  | 893k      | 893k          | 6.1M              |
| eps_001 (e=0.01) | 1.1M    | 729k          | 6.1M              |

The sweep lines are compressed into the leftmost 5-20% of the panel, making them
nearly invisible. The reader cannot compare convergence behavior across variants.

### P2: q=35 has no sweep data

q=35 was added later; sensitivity sweeps were only run for q=25 (excluded), q=40,
q=55. The q=35 column shows only the baseline — visually empty and inconsistent.

### P3: The figure conflates two different messages

- **Clipping epsilon row**: Shows how different PPO clip ratios affect learning
  speed. The interesting signal is the *convergence trajectory shape*, which
  requires overlapping x-ranges to compare.
- **Patience row**: Shows how many consecutive exploitability passes are needed
  before early-stopping. The interesting signal is *where training stops*, i.e.,
  the total training length, not the trajectory shape.

These are fundamentally different stories but are forced into the same plot format.

### P4: No clear takeaway without reading the text

A reader glancing at the figure sees colored blobs in the first 1M and a long
black line trailing off. It is unclear what conclusion to draw — is shorter
better? Worse? The same? The figure does not visually emphasize the key result
(baseline parameters are near-optimal).

### P5: Legend placement

Legend only on the rightmost panel (q=55), partially occluding data. For a 6-panel
figure the reader has to look far right to decode colors in the left panels.

## Proposed solutions

### Option A: Log-scale x-axis

**Change**: `ax.set_xscale('log')` on all panels. X-axis ticks at 10k, 100k, 1M.

**Pros**: Minimal code change (1 line). Sweep lines spread across 10k-1M range;
baseline tail compressed. All data preserved.

**Cons**: Log-scale effort curves look unfamiliar in RL papers. Early training
(first 10k steps) may get over-emphasized. Tick labels need careful formatting.

**Effort**: ~10 lines of code.

### Option B: Truncated x-axis (unified cutoff)

**Change**: Set `ax.set_xlim(0, 1.5e6)` on all panels. Baseline extends beyond
the visible range but the portion shown is still informative.

**Pros**: Simple, familiar linear scale. All sweep lines fully visible. Readers
can compare trajectories directly.

**Cons**: Loses the baseline's long convergence tail (viewer doesn't see where
baseline actually stabilizes). Could add a text annotation "baseline continues
to 5M" or a small arrow at the right edge.

**Effort**: ~5 lines of code.

### Option C: Broken axis (split x-axis)

**Change**: Left 70% of each panel covers 0-1.5M; right 30% covers 1.5M-6M,
with diagonal break marks at the split.

**Pros**: Shows both the sweep detail region and the full baseline trajectory.
No information lost.

**Cons**: Complex to implement in matplotlib (requires two axes per panel,
12 total). Break marks can look cluttered in a 2x3 grid. Unconventional for
this type of figure.

**Effort**: ~80-100 lines of code.

### Option D: Inset zoom panels

**Change**: Keep current full x-range. Add a small inset axis in each panel
zoomed to 0-1.5M where sweep lines are visible.

**Pros**: Full baseline visible at a glance; inset provides sweep detail. Clean
separation of the two scales.

**Cons**: 6 insets in a 2x3 grid may feel cramped. The inset competes for
attention with the main plot. Hard to get sizing right.

**Effort**: ~40-50 lines of code.

### Option E: Replace with dot plot / bar chart

**Change**: Instead of convergence curves, plot each variant as a point or bar:
- x-axis: variant name (eps=0.01, ..., baseline, ..., p=10)
- y-axis (left): final effort at stopping (or mean effort over last 10% of training)
- y-axis (right) or color: training steps at stopping

This directly communicates "baseline is near-optimal: similar final effort,
reasonable training time."

**Pros**: Clearest takeaway at a glance. No x-scale mismatch. Naturally handles
the q=35 missing data (just show baseline dot). Compact — could be a single
row of panels instead of 2x3.

**Cons**: Loses the trajectory information (how effort evolves over training).
A reviewer might want to see the convergence dynamics.

**Effort**: ~60-80 lines (rewrite the function).

### Option F: Hybrid — curves (truncated) + summary bar

**Change**: Two-part figure.
- Left: 2x3 convergence curves with truncated x-axis (Option B) showing
  trajectory shape for the first 1.5M steps.
- Right: 1x2 bar chart (one per sweep type) showing total training steps
  by variant, which communicates the early-stopping message.

**Pros**: Trajectory detail + summary in one figure. Best of both worlds.

**Cons**: Complex layout. Wider figure. May feel like two separate figures
forced together.

**Effort**: ~100 lines of code.

### Option G: Normalize x-axis to fraction of training

**Change**: Convert x-axis from absolute steps to "fraction of max episodes"
(0.0 to 1.0). Each run's trajectory is stretched to fill the full x-range.

**Pros**: All lines span the full panel width. Easy to compare shapes.

**Cons**: Misleading — makes a 74k-step run look the same length as a 5M-step
run. Obscures the key difference (training efficiency). Not recommended.

### Option H: Two-column figure (drop q=35 or move to appendix)

**Change**: Only show q=40 and q=55 (where sweep data exists). Mention q=35 in
text. This gives more horizontal space per panel.

**Pros**: No empty panels. Larger, clearer sub-figures. Honest about data
availability.

**Cons**: Inconsistent with other figures that show all three q values.
Requires running q=35 sweeps anyway if a reviewer asks.

## Missing data: q=35 sweep experiments

Regardless of figure design, the q=35 sweep data needs to be generated.
This requires 21 training runs:

- 4 epsilon variants x 3 seeds = 12 runs
- 3 patience variants x 3 seeds = 9 runs

Each run uses `--episodes 2048000`. Expected runtime: ~10-30 min per run
(most will early-stop much sooner). Commands documented in
`docs/tasks/q35-all-experiments/phase05-2p-ablations.md` (ablations) and
a forthcoming phase06 (sweeps).

## Recommendation

**Option B (truncated x-axis)** or **Option E (dot plot)** depending on what
story the paper is telling:

- If the paper emphasizes *convergence dynamics* under different hyperparameters
  → **Option B**. Simple, familiar, shows trajectories. Add a one-line annotation
  "baseline continues to Xm steps" at right edge.

- If the paper emphasizes *robustness of baseline parameters* (the main PPO
  hyperparameters are near-optimal) → **Option E**. Clearest single-glance
  takeaway. Could keep a full convergence curve figure in the appendix.

- **Option F (hybrid)** if space permits and both messages matter equally.
