"""Shared figure style for the T=2 v2 report pack.

One style for every figure of the pack:

* 7 in wide (``FIG_W``), no text smaller than 8 pt at that size (``MIN_FONT_PT``; checked by
  :func:`assert_fonts` before every save);
* PDF (TrueType, ``pdf.fonttype = 42``) plus a 300 dpi PNG;
* fixed colours per arm (``ARM_COLORS``) and per q (``Q_COLORS``), the same in every figure.

Colour rules (``dataviz`` method): categorical hues come from the documented 8-slot palette, in a
fixed order that is never cycled; every figure with two or more series carries a legend that
states n; q is always also encoded by line style and marker (``Q_LINESTYLE``, ``Q_MARKER``), so no
figure relies on colour alone. Arm colours were chosen by enumerating the assignments of the 8
documented hues to the 8 arm labels and keeping the one with the largest worst-pair CVD
separation inside every group of arms that appear together (worst CVD dE >= 10 over the groups;
``validate_palette.py --pairs all`` on each group: lightness band, chroma floor, CVD and
normal-vision floors pass). Groups: {sampled, expected}, {A_joint, B1, B2},
{B2/stochastic, mean}, {constant, decay}. The documented palette has 8 hues, so q reuses the first
two slots (blue, orange); a figure colours either by arm or by q, never both, and q is always
repeated by line style and marker. Magenta, aqua and yellow are below 3:1 contrast on white:
every figure that uses them carries a legend, and lines are drawn >= 1.4 pt with markers.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_W = 7.0
MIN_FONT_PT = 8.0
DPI = 300

# documented categorical palette (light surface), fixed slot order
SLOT = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
        "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}

ARM_COLORS: Dict[str, str] = {
    "sampled": SLOT["blue"],
    "expected": SLOT["orange"],
    "expected_ext": SLOT["orange"],
    "A_joint": SLOT["magenta"],
    "B1_frozen_allnorm": SLOT["green"],
    "B2_frozen_s1norm": SLOT["violet"],
    "stochastic": SLOT["violet"],
    "mean": SLOT["aqua"],
    "B2_frozen_s1norm_mean": SLOT["aqua"],
    "constant": SLOT["yellow"],
    "B2_mean_constant": SLOT["yellow"],
    "decay": SLOT["red"],
    "B2_mean_decay": SLOT["red"],
}
ARM_LABELS: Dict[str, str] = {
    "sampled": "sampled", "expected": "expected", "expected_ext": "expected (extension)",
    "A_joint": "A joint", "B1_frozen_allnorm": "B1 frozen, all-rows norm.",
    "B2_frozen_s1norm": "B2 frozen, stage-1-rows norm.", "stochastic": "stochastic continuation",
    "mean": "mean continuation", "B2_frozen_s1norm_mean": "mean continuation",
    "constant": "constant LR", "B2_mean_constant": "constant LR",
    "decay": "LR decay", "B2_mean_decay": "LR decay",
}
Q_COLORS: Dict[int, str] = {50: SLOT["blue"], 60: SLOT["orange"]}
Q_LINESTYLE: Dict[int, str] = {50: "-", 60: "--"}
Q_MARKER: Dict[int, str] = {50: "o", 60: "s"}

# reference / theory marks (neutral inks, not categorical identity)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
REF = "#0b0b0b"          # closed-form / target curves
THRESH = "#9c1c1c"       # threshold lines (dashed)

LINE_W = 1.5
THIN_W = 0.7


def apply() -> None:
    """Install the pack's rcParams (idempotent)."""
    mpl.rcParams.update({
        "figure.figsize": (FIG_W, 3.2),
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.0,
        "figure.titlesize": 9.0,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": LINE_W,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 6,
        "mathtext.default": "regular",
        "axes.unicode_minus": True,
    })


def new_figure(nrows: int = 1, ncols: int = 1, height: float = 3.2, sharex: bool = False,
               sharey: bool = False, **kw):
    """Create a 7 in wide figure with the pack style.

    Args:
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        height: Figure height in inches (the width is fixed at 7 in).
        sharex: Share x axes.
        sharey: Share y axes.
        **kw: Passed to ``plt.subplots`` (``gridspec_kw``, ``constrained_layout`` ...).

    Returns:
        ``(fig, axes)`` as from ``plt.subplots`` (axes always an ndarray of shape (nrows, ncols)).
    """
    apply()
    kw.setdefault("constrained_layout", True)
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_W, height), sharex=sharex, sharey=sharey,
                             squeeze=False, **kw)
    return fig, axes


def label_n(label: str, n: int) -> str:
    """Legend text that states n, e.g. ``'expected (n=10)'``."""
    return f"{label} (n={int(n)})"


def arm_label(arm: str) -> str:
    """Display label of an arm."""
    return ARM_LABELS.get(arm, arm)


def arm_color(arm: str) -> str:
    """Fixed colour of an arm (raises KeyError for an unregistered arm)."""
    return ARM_COLORS[arm]


def q_color(q: int) -> str:
    """Fixed colour of a q value."""
    return Q_COLORS[int(q)]


def median_iqr(values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    """(median, p25, p75, min, max) with numpy linear interpolation."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan,) * 5
    return (float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)),
            float(v.min()), float(v.max()))


def band_plot(ax, x, med, lo, hi, color: str, label: Optional[str] = None, ls: str = "-",
              marker: Optional[str] = None, lw: float = LINE_W, alpha: float = 0.22, **kw):
    """Median line with an IQR band."""
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)
    return ax.plot(x, med, color=color, ls=ls, marker=marker, lw=lw, label=label, **kw)[0]


def legend(ax, **kw):
    """Legend with the pack defaults (no frame)."""
    kw.setdefault("loc", "best")
    return ax.legend(**kw)


def text_violations(fig) -> List[Tuple[str, float]]:
    """All visible Text artists smaller than ``MIN_FONT_PT``."""
    bad = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        fs = float(t.get_fontsize())
        if fs < MIN_FONT_PT - 1e-9:
            bad.append((t.get_text()[:40], fs))
    return bad


def assert_fonts(fig) -> None:
    """Raise if any visible text is smaller than 8 pt or the figure is not 7 in wide."""
    fig.canvas.draw()
    bad = text_violations(fig)
    if bad:
        raise ValueError(f"text below {MIN_FONT_PT} pt: {bad[:5]}")
    w, _ = fig.get_size_inches()
    if abs(w - FIG_W) > 1e-9:
        raise ValueError(f"figure width {w} in != {FIG_W} in")


def save(fig, stem: str) -> List[str]:
    """Write ``stem.pdf`` and ``stem.png`` (300 dpi) without volatile metadata.

    Args:
        fig: Matplotlib figure.
        stem: Path without extension.

    Returns:
        The two written paths.
    """
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    pdf, png = stem + ".pdf", stem + ".png"
    fig.savefig(pdf, format="pdf", metadata={"Creator": "tools/v2/report", "Producer": "matplotlib",
                                             "CreationDate": None, "ModDate": None,
                                             "Title": os.path.basename(stem)})
    fig.savefig(png, format="png", dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return [pdf, png]


def palette_doc() -> List[Tuple[str, str, str]]:
    """Rows (kind, key, hex) documenting every fixed colour, for the README."""
    rows: List[Tuple[str, str, str]] = []
    for k, v in ARM_COLORS.items():
        rows.append(("arm", k, v))
    for k, v in Q_COLORS.items():
        rows.append(("q", f"q={k}", v))
    rows += [("reference", "closed form / target", REF), ("reference", "threshold line (dashed)", THRESH)]
    return rows


def median_iqr_curves(ax, df, x: str, y: str, color: str, label: str, ls: str = "-", marker: Optional[str] = None,
                      lw: float = LINE_W, extra: Optional[dict] = None):
    """Plot the across-seed median with an IQR band of ``y`` against ``x`` and return what was plotted.

    Args:
        ax: Matplotlib axes.
        df: Long DataFrame with one row per (seed, x) holding ``x`` and ``y``.
        x: x column (e.g. the update).
        y: y column.
        color: Series colour (use ``ARM_COLORS`` / ``Q_COLORS``).
        label: Legend text (state n, e.g. via ``label_n``).
        ls: Line style.
        marker: Optional marker.
        lw: Line width.
        extra: Constant columns added to the returned rows (``{"q": 50, "arm": "expected", "panel": "..."}``).

    Returns:
        DataFrame ``[x, y_name, median, q25, q75, min, max, n]`` of the plotted values, plus ``extra``.
    """
    import pandas as pd
    rows = []
    for xv, g in df.groupby(x, sort=True):
        m, lo, hi, mn, mx = median_iqr(g[y].to_numpy(dtype=float))
        rows.append({x: xv, "metric": y, "median": m, "q25": lo, "q75": hi, "min": mn, "max": mx,
                     "n": int(np.isfinite(g[y].to_numpy(dtype=float)).sum())})
    out = pd.DataFrame(rows)
    if len(out):
        band_plot(ax, out[x].to_numpy(), out["median"].to_numpy(), out["q25"].to_numpy(), out["q75"].to_numpy(),
                  color=color, label=label, ls=ls, marker=marker, lw=lw)
    for k, v in (extra or {}).items():
        out[k] = v
    return out
