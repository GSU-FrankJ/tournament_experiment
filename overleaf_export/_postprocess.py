#!/usr/bin/env python3
"""Post-process pandoc output into a self-contained, Overleaf-ready main.tex."""
import re
import sys

TEX = "overleaf_export/main.tex"

# docx image basename -> (target file in figures/, includegraphics options)
IMG_MAP = {
    "image1.png":  ("fig1_framework.png",            "width=0.55\\linewidth"),
    "image2.emf":  ("convergence_main.pdf",          "width=\\linewidth"),
    "image3.emf":  ("effort_drift.pdf",              "width=\\linewidth"),
    "image4.emf":  ("effort_drift.pdf",              "width=\\linewidth"),
    "image5.emf":  ("kl_dynamics.pdf",               "width=\\linewidth"),
    "image6.emf":  ("distance_to_equilibrium.pdf",   "width=\\linewidth"),
    "image7.emf":  ("exploitability_dynamics.pdf",   "width=\\linewidth"),
    "image8.emf":  ("ablation_comparison.pdf",       "width=\\linewidth"),
    "image9.emf":  ("equilibrium_recovery_dotplot.pdf", "width=\\linewidth"),
    "image10.emf": ("beta_evolution.pdf",            "width=\\linewidth"),
}

with open(TEX, encoding="utf-8") as fh:
    src = fh.read()

# 1) Rewrite every \includegraphics that points at an extracted media file.
def repl_graphic(m):
    opts_full = m.group(0)
    path = m.group("path")
    base = path.rsplit("/", 1)[-1]
    if base not in IMG_MAP:
        print(f"WARN: unmapped image {base}", file=sys.stderr)
        return opts_full
    target, opts = IMG_MAP[base]
    return f"\\includegraphics[{opts}]{{figures/{target}}}"

n_before = src.count("\\includegraphics")
src = re.sub(
    r"\\includegraphics\[[^\]]*\]\{(?P<path>[^}]*?(?:image\d+\.(?:emf|png)))\}",
    repl_graphic,
    src,
)

# 2) Macron over e written as literal U+02C9 inside math -> proper \overline.
src = src.replace("{\\overset{ˉ}{e}}", "{\\overline{e}}")
assert "ˉ" not in src, "stray macron char remains"

# 3) The one highlight that wraps an explicit line break (\\): soul cannot span
#    a forced break, so close the highlight before \\ and reopen after it.
old_hl = ("\\hl{\\textbf{Figure 3.} \\textbf{Effort Drift Across Noise "
          "Levels.}\\\\")
new_hl = ("\\hl{\\textbf{Figure 3.} \\textbf{Effort Drift Across Noise "
          "Levels.}}\\\\\n\\hl{")
assert src.count(old_hl) == 1, f"expected 1 hl-with-break, found {src.count(old_hl)}"
src = src.replace(old_hl, new_hl)

with open(TEX, "w", encoding="utf-8") as fh:
    fh.write(src)

n_after = src.count("figures/")
print(f"includegraphics before={n_before}; figures/ refs now={n_after}")
print("post-process OK")
