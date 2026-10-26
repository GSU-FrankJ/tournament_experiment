"""Figure style of the P-FIG round: 7 in wide, every font at least 8 pt, a colour-blind-safe palette (Okabe-Ito),
vector PDF plus PNG at 300 dpi. Imported by the plot scripts ``F*.py`` (which read ``data/F*.csv`` only).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent                          # paper/multistage_rev
DATA_DIR = OUT / "data"
FIG_DIR = Path(os.environ.get("PFIG_FIG_DIR", str(OUT / "figures")))   # PFIG_FIG_DIR: write elsewhere (reviews)

WIDTH_IN = 7.0
MIN_FONT = 8.0

# Okabe-Ito palette (colour-blind safe)
BLACK, ORANGE, SKY, GREEN = "#000000", "#E69F00", "#56B4E9", "#009E73"
YELLOW, BLUE, VERMILION, PURPLE, GREY = "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#7F7F7F"

# semantic colours shared by all figures
C_THEORY = BLACK        # closed-form equilibrium
C_RUN = BLUE            # a learned run (thin lines / points)
C_MEDIAN = "#003A5C"    # median across runs (bold)
C_MARK = VERMILION      # the marked run (failed run / seed 80502 / seed 30510)
C_BR = ORANGE           # dynamic best-response effort
C_G = GREEN             # state-wise dynamic gain G_t(d)/dW
C_DELTA = PURPLE        # one-step deviation gain Delta_t(d)/dW
C_POS = SKY             # positive-effort region (shading)
C_TAILS = GREY          # theoretical zero-effort tails (shading)
C_REACH = YELLOW        # BR-reachable region (shading)


def apply() -> None:
    """Set the matplotlib rcParams (all font sizes 8 pt, fonts embedded as TrueType, no figure title).

    Mathtext sub- and superscripts are shrunk to 70 % by matplotlib (5.6 pt at 8 pt), which would break the
    8 pt minimum for symbols such as G_{max,full} and tick exponents; the shrink factor is therefore set to 1
    (scripts keep the 8 pt size and are only shifted).
    """
    import matplotlib._mathtext as _mt
    _mt.SHRINK_FACTOR = 1.0
    plt.rcParams.update({
        "font.size": MIN_FONT, "axes.labelsize": MIN_FONT, "axes.titlesize": MIN_FONT,
        "xtick.labelsize": MIN_FONT, "ytick.labelsize": MIN_FONT, "legend.fontsize": MIN_FONT,
        "figure.titlesize": MIN_FONT, "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    })


def new_figure(height_in: float, **kwargs):
    """A 7 in wide figure with constrained layout (so the saved width stays exactly 7 in)."""
    apply()
    return plt.figure(figsize=(WIDTH_IN, height_in), layout="constrained", **kwargs)


def check_fonts(fig) -> None:
    """Assert that every text object of the figure is at least 8 pt and that the figure has no title."""
    from matplotlib.text import Text
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text)
             if t.get_text().strip() and t.get_fontsize() < MIN_FONT - 1e-9]
    if small:
        raise AssertionError(f"text below {MIN_FONT} pt: {small[:5]}")
    if fig.get_suptitle():
        raise AssertionError("figures carry no titles")
    for ax in fig.axes:
        if ax.get_title() or ax.get_title(loc="left") or ax.get_title(loc="right"):
            raise AssertionError("figures carry no titles")
    w, _ = fig.get_size_inches()
    if abs(w - WIDTH_IN) > 1e-9:
        raise AssertionError(f"figure width {w} in, expected {WIDTH_IN}")


def pdf_min_font_size(path: Path) -> float:
    """Smallest font size (pt) set by a ``Tf`` operator in the content streams of a matplotlib PDF.

    Covers the glyph runs that ``check_fonts`` cannot see (mathtext sub- and superscripts).
    """
    import re
    import zlib
    raw = Path(path).read_bytes()
    sizes = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
        try:
            body = zlib.decompress(m.group(1))
        except zlib.error:
            continue
        sizes += [float(x) for x in re.findall(rb"/F\d+(?:-\d+)? ([0-9.]+) Tf", body)]
    return min(sizes) if sizes else float("nan")


def save(fig, stem: str) -> Iterable[Path]:
    """Write ``figures/{stem}.pdf`` (vector) and ``figures/{stem}.png`` (300 dpi); returns the two paths."""
    check_fonts(fig)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIG_DIR / f"{stem}.pdf", FIG_DIR / f"{stem}.png"]
    fig.savefig(paths[0], metadata={"CreationDate": None})
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    smallest = pdf_min_font_size(paths[0])
    if not smallest >= MIN_FONT - 1e-6:
        raise AssertionError(f"{paths[0].name}: a glyph run is set at {smallest} pt (< {MIN_FONT} pt)")
    return paths
