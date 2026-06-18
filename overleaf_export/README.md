# Overleaf export — TEL-PPO paper

LaTeX conversion of
`Draft2_Exploring Equilibrium Efforts in Rank-Order Tournaments via Policy-Gradient Self-Play.docx`,
produced with `pandoc 3.8`.

## Files

```
main.tex            # the paper (self-contained preamble — compile this)
figures/            # all 9 figures referenced by main.tex
header.tex          # build-only: preamble extras (already inlined into main.tex)
_postprocess.py     # build-only: image-path / highlight / symbol fixups
```

Only `main.tex` and `figures/` are needed to compile. `header.tex` and
`_postprocess.py` are kept for reproducibility and are ignored by the compiler.

## Use in Overleaf

1. Upload `overleaf_export.zip` via **New Project → Upload Project**
   (or drag `main.tex` + the `figures/` folder into a blank project).
2. Set the main document to `main.tex`.
3. **Compiler:** default **pdfLaTeX** works. XeLaTeX / LuaLaTeX also compile.
4. Compile.

All packages used (graphicx, booktabs, longtable, amsmath, soul, newunicodechar,
hyperref, …) are part of Overleaf's standard TeX Live, so no extra setup is needed.

## Figure mapping

The 9 plots are taken from the repository's current pipeline renders
(`paper/figures/*.pdf`), per your choice. `image1` (the framework schematic) is
the exact PNG embedded in the docx.

| docx image | caption in doc | file used |
|-----------|----------------|-----------|
| image1.png | Fig 1 — TEL-PPO framework | `figures/fig1_framework.png` (from docx) |
| image2 | Fig 2 — Convergence under noise/prizes | `convergence_main.pdf` |
| image3 | Fig 3 — Effort drift | `effort_drift.pdf` |
| image4 | Fig 3 — Effort drift (v2) | `effort_drift.pdf` |
| image5 | Fig 4 — KL divergence | `kl_dynamics.pdf` |
| image6 | Fig 5 — Convergence error \|ē−e*\| | `distance_to_equilibrium.pdf` |
| image7 | Fig 6 — Exploitability | `exploitability_dynamics.pdf` |
| image8 | Fig 7 — Component ablation | `ablation_comparison.pdf` |
| image9 | Fig 8 — Equilibrium recovery | `equilibrium_recovery_dotplot.pdf` |
| image10 | Fig 6 — Policy-distribution evolution | `beta_evolution.pdf` |

## Known differences from the .docx (read me)

- **Figures are the repo's latest renders, not the bitmaps embedded in the docx.**
  Most match the captions, but some plots have changed since the draft was
  written. Confirmed difference: the docx's **policy-distribution** figure shows
  density snapshots at 10/50/90 % of training, whereas `beta_evolution.pdf` plots
  the α/β parameter trajectories. The two "Figure 3 (effort drift)" panels in the
  draft both map to the single `effort_drift.pdf`.
  To use the exact docx images instead, the embedded EMF files would need to be
  converted to PDF (LibreOffice/Inkscape) — ask and this can be redone.
- **Title** renders as the first large unnumbered heading, matching the Word
  styling. Section numbering is off because the Word headings were not numbered.
- **Word highlights** (yellow) are preserved via `soul`. One highlighted caption
  that wrapped a forced line break was split so it compiles.
- **A few symbols** (±, −, →, ×, ∈, ε, ē) are rendered via `newunicodechar`
  so the document also compiles under pdfLaTeX.
- **Draft annotations** present in the Word file ("(Baseline?)", "NC", "500/1500?")
  are kept verbatim — they are the author's marks, not conversion artifacts.

## Rebuild

```bash
pandoc "Draft2_...Self-Play.docx" -s -H overleaf_export/header.tex \
  --extract-media=overleaf_export/_media -t latex -o overleaf_export/main.tex
python3 overleaf_export/_postprocess.py      # rewrites image paths + fixups
# then copy paper/figures/*.pdf + image1.png into overleaf_export/figures/
```
