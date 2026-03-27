# Paper Figures & Tables Revision

## Goal

Revise all paper figures and tables to match updated revision notes: restyle plots (lighter shading, renamed legends, vertical convergence lines), rename labels to formal terminology, unify axes, fill in missing table parameters, and incorporate q=35 data alongside q=40 and q=55.

## Scope

- **Touch:** `paper/generator/plots.py`, `paper/generator/tables.py`, `paper/generator/config.py`, `paper/generator/extract.py`, `paper/generator/metrics.py`
- **May touch:** `paper/generator/__main__.py` (if new entry points needed)
- **Do NOT touch:** `agents/`, `envs/`, `run/`, `utils/`, `results/*/convergence/*.json`
- **Do NOT modify** theory parameters in `config.py` without explicit confirmation

## Key files

- `paper/generator/config.py` — Q_VALUES, styles, colors, thresholds
- `paper/generator/plots.py` — all figure generation functions
- `paper/generator/tables.py` — table generation (CSV + LaTeX)
- `paper/generator/extract.py` — data loading and aggregation
- `paper/generator/metrics.py` — metric computation

## Constraints

- All changes must be backward-compatible with `python -m paper.generator make_all`
- Figures must remain publication-quality (300 DPI, proper font sizes)
- Label renames are exact — use the specific text from revision notes
- q values displayed without `.0` suffix (e.g., `q = 35` not `q = 35.0`)
- **Primary q values: {35, 40, 55}** — q=25 does NOT appear in main figures or tables
- q=25 appears ONLY in Figure 6b (excluded low-noise case) and is discussed separately in text
- Tables 3–4: q=25 rows replaced by q=35 (not appended)
- Verify each figure visually after generation before moving to next phase

## Terminology mapping (revision notes)

| Old label | New label |
|:--|:--|
| Cheap gate | Stability screening passed |
| Nash convergence | Approx. Nash verified |
| Threshold | Tolerance threshold |
| Baseline (ablation) | TEL-PPO |
| No cheap gate | No stability screening |
| No exploitability | No exploitability verification |
| Per-seed | Per-seed estimate |
| Seed mean | Across-seed mean |
