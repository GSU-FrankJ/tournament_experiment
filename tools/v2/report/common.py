"""Shared machinery of the T=2 v2 report pack: provenance, writers, manifest, cross-checks.

Every builder module (``sec_*.py``) creates one :class:`Pack`, registers its items with
``Pack.table / found_table / figure / data / numbers_item / gap / mismatch / reeval``, and calls
``Pack.save_fragment()``. ``finalize()`` merges the fragments of all modules into
``manifest.csv``, ``key_numbers.csv``, ``data_dictionary.csv``, ``gaps.md``, ``consistency.md``,
``reevaluations.csv`` and ``README.md``.

Paths: ``results/...`` is read from ``RESULTS_ROOT`` (``$T2_REPORT_RESULTS_ROOT`` or
``<repo>/results``; the large untracked arrays live in the canonical worktree's results
directory), every other repo-relative path from ``REPO``. Provenance always records the
repo-relative name plus the SHA-256 of the bytes actually read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import dictionary  # noqa: E402
import spec  # noqa: E402
import style  # noqa: E402

PACK_REL = "reports/v2/t2_report"
PACK_DIR = REPO / PACK_REL
BUILD_DIR = PACK_DIR / "_build"
RESULTS_ROOT = Path(os.environ.get("T2_REPORT_RESULTS_ROOT", str(REPO / "results")))
BASE_COMMIT = "cb0b541b3ba16c04a0fa5d1a465e9d94c635ed89"
SCRIPT_DIR_REL = "tools/v2/report"
MAX_COPY_BYTES = 400_000

STATUSES = ("found", "regenerated", "generated", "derived", "missing")


# ----------------------------------------------------------------------------------------------
# paths and hashing
# ----------------------------------------------------------------------------------------------

def abspath(rel: Union[str, Path]) -> Path:
    """Absolute path of a repo-relative path (``results/...`` resolves against RESULTS_ROOT)."""
    rel = str(rel)
    if os.path.isabs(rel):
        return Path(rel)
    if rel == "results" or rel.startswith("results/"):
        return RESULTS_ROOT / rel[len("results/"):] if rel != "results" else RESULTS_ROOT
    return REPO / rel


def relpath(p: Union[str, Path]) -> str:
    """Repo-relative name of an absolute path under REPO or RESULTS_ROOT (else the absolute path)."""
    p = Path(p).resolve()
    try:
        return "results/" + str(p.relative_to(RESULTS_ROOT.resolve()))
    except ValueError:
        pass
    try:
        return str(p.relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


_SHA: Dict[Tuple[str, int, int], str] = {}


def sha256_file(path: Union[str, Path]) -> str:
    """SHA-256 of a file (cached per process by path, size and mtime)."""
    p = Path(path)
    st = p.stat()
    key = (str(p), st.st_size, int(st.st_mtime_ns))
    if key not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _SHA[key] = h.hexdigest()
    return _SHA[key]


@dataclass(frozen=True)
class Source:
    """A file that an item was built from."""

    path: str
    sha256: str
    nbytes: int
    selector: str = ""

    def fmt(self) -> str:
        """``path:sha256`` (with the selector in brackets when given)."""
        return f"{self.path}:{self.sha256}" + (f"[{self.selector}]" if self.selector else "")


@dataclass
class SourceSet:
    """Many files hashed as one set (per-run files of a study)."""

    label: str
    files: List[Source]

    @property
    def set_sha256(self) -> str:
        """SHA-256 of the sorted ``path:sha256`` lines."""
        txt = "\n".join(sorted(f"{s.path}:{s.sha256}" for s in self.files))
        return hashlib.sha256(txt.encode()).hexdigest()


def src(rel: Union[str, Path], selector: str = "") -> Source:
    """Source record of one existing file.

    Args:
        rel: Repo-relative path (``results/...`` allowed).
        selector: Optional row/column/key selector, recorded in brackets.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = abspath(rel)
    if not p.is_file():
        raise FileNotFoundError(str(rel))
    return Source(relpath(p) if os.path.isabs(str(rel)) else str(rel), sha256_file(p), p.stat().st_size, selector)


def srcs(patterns: Union[str, Iterable[str]], label: str = "", expect: Optional[int] = None) -> SourceSet:
    """Source records of all files matching repo-relative glob patterns.

    Args:
        patterns: One glob or an iterable of globs (repo-relative; ``results/...`` allowed).
        label: Name of the set (defaults to the first pattern).
        expect: If given, the number of files that must match.

    Returns:
        The :class:`SourceSet`.
    """
    pats = [patterns] if isinstance(patterns, str) else list(patterns)
    files: List[Source] = []
    for pat in pats:
        base = abspath(pat.split("*")[0].rstrip("/") or ".")
        if "*" not in pat:
            files.append(src(pat))
            continue
        if pat.startswith("results/"):
            root, sub = RESULTS_ROOT, pat[len("results/"):]
        else:
            root, sub = REPO, pat
        for p in sorted(root.glob(sub)):
            if p.is_file():
                files.append(Source(relpath(p), sha256_file(p), p.stat().st_size))
    if expect is not None and len(files) != expect:
        raise ValueError(f"{pats}: expected {expect} files, found {len(files)}")
    return SourceSet(label or pats[0], files)


def script_ref(script: str) -> Tuple[str, str]:
    """``(path:function, sha256 of the module file)`` for ``'sec_x.py:build_t20'``."""
    mod = script.split(":")[0]
    p = HERE / mod
    sha = sha256_file(p) if p.is_file() else ""
    return f"{SCRIPT_DIR_REL}/{script}", sha


def git_head() -> str:
    """HEAD commit of the repository at build time (informational)."""
    try:
        import subprocess
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:  # pragma: no cover
        return ""


# ----------------------------------------------------------------------------------------------
# numbers and markdown
# ----------------------------------------------------------------------------------------------

def fmt_cell(v: Any, digits: int = 4) -> str:
    """Markdown cell text of a value (floats with ``digits`` significant digits)."""
    if v is None or v is pd.NA or v is pd.NaT:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "True" if bool(v) else "False"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(float(v)):
            return "" if math.isnan(float(v)) else ("inf" if v > 0 else "-inf")
        return f"{float(v):.{digits}g}"
    s = str(v)
    return s.replace("|", "\\|").replace("\n", " ")


def df_to_md(df: pd.DataFrame, digits: int = 4, max_rows: Optional[int] = None) -> str:
    """GitHub-flavoured markdown table of a DataFrame."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    n = len(df)
    rows = df if max_rows is None or n <= max_rows else df.head(max_rows)
    for rec in rows.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt_cell(v, digits) for v in rec) + " |")
    if max_rows is not None and n > max_rows:
        lines.append(f"| ... {n - max_rows} more rows in the CSV |" + " |" * (len(cols) - 1))
    return "\n".join(lines)


def num(x: Any) -> str:
    """Full-precision text of a number for key_numbers.csv."""
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    return str(x)


def median_iqr(values: Sequence[float]) -> Dict[str, float]:
    """Median, IQR (25th/75th percentile, numpy linear interpolation), min, max, n of the finite values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(median=np.nan, q25=np.nan, q75=np.nan, min=np.nan, max=np.nan, n=0)
    return dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                min=float(v.min()), max=float(v.max()), n=int(v.size))


def bootstrap_mean_ci(diff: Sequence[float], n_boot: int = 10000, seed: int = 20261001) -> Tuple[float, float]:
    """95% percentile bootstrap CI of the mean (same method as ``tools/v2/pilot4_common.paired_summary``)."""
    dv = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    bm = dv[rng.integers(0, dv.size, size=(n_boot, dv.size))].mean(axis=1)
    return float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


# ----------------------------------------------------------------------------------------------
# existing-report parsing (cross-checks)
# ----------------------------------------------------------------------------------------------

def _split_row(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells = re.split(r"(?<!\\)\|", s)
    return [c.strip().replace("\\|", "|").replace("`", "").strip() for c in cells]


def parse_md_tables(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Markdown tables of a file: ``[{heading, line, header, rows}]`` (cells as strings)."""
    p = abspath(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    out: List[Dict[str, Any]] = []
    heading, i = "", 0
    sep = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            heading = ln.lstrip("# ").strip()
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and sep.match(lines[i + 1]):
            header = _split_row(ln)
            rows, j = [], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(_split_row(lines[j]))
                j += 1
            out.append({"heading": heading, "line": i + 1, "header": header, "rows": rows})
            i = j
            continue
        i += 1
    return out


def parse_num(s: Any) -> Optional[float]:
    """Float value of a report cell (unicode minus, +, %, 1e-05), else ``None``."""
    if s is None:
        return None
    t = str(s).strip().replace("\u2212", "-").replace(",", "")
    t = t.replace("\u2009", "").rstrip("%").strip()
    if t in ("", "-", "n/a", "nan", "NaN", "\u2014"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _sig_digits(s: str) -> Tuple[int, float]:
    """(significant digits shown, value) of a numeric string."""
    t = s.strip().replace("\u2212", "-").lstrip("+-")
    mant = re.split(r"[eE]", t)[0].replace(".", "")
    mant = mant.lstrip("0")
    return max(len(mant), 1), float(s.strip().replace("\u2212", "-"))


def consistent(x: float, report_cell: str, rel_tol: float = 0.0) -> bool:
    """Whether ``x`` rounds to the report's displayed value at the report's own precision."""
    v = parse_num(report_cell)
    if v is None or x is None or not math.isfinite(x):
        return False
    if v == 0.0:
        return abs(x) <= 5e-17 or abs(x) < 1e-12
    sig, _ = _sig_digits(str(report_cell))
    exp10 = math.floor(math.log10(abs(v)))
    half = 0.5 * 10.0 ** (exp10 - sig + 1)
    return abs(x - v) <= half * 1.0001 + rel_tol * abs(v)


def summarize(df: pd.DataFrame, by: Sequence[str], cols: Sequence[str]) -> pd.DataFrame:
    """Across-seed summary: for each group and column the median, q25, q75, min, max and n.

    Percentiles use numpy linear interpolation (the pack convention). Output is long format with
    columns ``by + [metric, median, q25, q75, min, max, n]``.
    """
    rows = []
    for key, g in df.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        for c in cols:
            st = median_iqr(g[c].to_numpy(dtype=float))
            rows.append({**dict(zip(by, key)), "metric": c, **st})
    return pd.DataFrame(rows)


def max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    """Largest absolute difference of two equally long numeric sequences (NaN-aware, 0.0 if empty)."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    if x.size == 0:
        return 0.0
    both_nan = np.isnan(x) & np.isnan(y)
    d = np.where(both_nan, 0.0, np.abs(x - y))
    return float(np.nanmax(d))


# ----------------------------------------------------------------------------------------------
# the Pack
# ----------------------------------------------------------------------------------------------

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df.reset_index(drop=True)


class Pack:
    """Collects the items built by one module and writes their files into the pack directory."""

    def __init__(self, module: str, pack_dir: Path = PACK_DIR):
        self.module = module
        self.dir = Path(pack_dir)
        self.items: List[Dict[str, Any]] = []
        self.numbers: List[Dict[str, Any]] = []
        self.gaps: List[Dict[str, Any]] = []
        self.mismatches: List[Dict[str, Any]] = []
        self.crosschecks: List[Dict[str, Any]] = []
        self.reevals: List[Dict[str, Any]] = []
        self.docs_rows: List[Dict[str, Any]] = []
        for sub in ("tables", "figures", "data", "provenance", "_build/fragments"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)

    # ---- provenance helpers ---------------------------------------------------------------------
    def _src_field(self, item_id: str, sources: Sequence[Union[Source, SourceSet]]) -> str:
        parts: List[str] = []
        flat: List[Source] = []
        for s in sources:
            if isinstance(s, SourceSet):
                flat += s.files
                parts.append(f"set:{s.label}#n={len(s.files)}:{s.set_sha256}")
            else:
                flat.append(s)
                parts.append(s.fmt())
        if len(flat) > 12:
            prov = self.dir / "provenance" / f"{item_id}_sources.csv"
            with open(prov, "w", newline="") as fh:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(["path", "sha256", "bytes", "selector"])
                for s in sorted(flat, key=lambda z: z.path):
                    w.writerow([s.path, s.sha256, s.nbytes, s.selector])
            return ("see " + f"{PACK_REL}/provenance/{item_id}_sources.csv" + f" ({len(flat)} files); " +
                    "; ".join(p for p in parts if p.startswith("set:")))
        return "; ".join(parts)

    def _register(self, id: str, typ: str, title: str, priority: str, status: str, files: List[str],
                  sources: Sequence[Union[Source, SourceSet]], script: str, notes: str, tier: str = "") -> None:
        if status not in STATUSES:
            raise ValueError(f"{id}: bad status {status}")
        if id in {i["id"] for i in self.items}:
            raise ValueError(f"{id}: registered twice in module {self.module}")
        sref, ssha = script_ref(script)
        self.items.append({
            "id": id, "section": spec.SECTION_OF.get(id, ""), "title": title, "type": typ, "priority": priority,
            "status": status, "files": "; ".join(files), "sources": self._src_field(id, sources),
            "script": f"{sref} (sha256 {ssha})", "commit": BASE_COMMIT, "notes": notes, "tier": tier,
        })

    def _defaults(self, id: str, title: Optional[str], priority: Optional[str]) -> Tuple[str, str, str]:
        d = spec.ITEMS.get(id)
        if d is None:
            raise KeyError(f"unknown item id {id}")
        return title or d["title"], priority or d["priority"], d["type"]

    def _docs(self, id: str, df: pd.DataFrame, docs: Optional[Dict[str, Any]], tier: str) -> None:
        docs = docs or {}
        for col in df.columns:
            d = docs.get(col)
            if isinstance(d, str):
                d = {"definition": d}
            base = dictionary.lookup(col, tier)
            row = {"item": id, "column": col, "definition": "", "units": "", "normalization": "", "tier": "",
                   "source": ""}
            if base:
                row.update(base)
            if d:
                row.update({k: v for k, v in d.items() if k in row and v not in (None, "")})
            if not row["definition"]:
                row["definition"] = "UNDOCUMENTED"
            self.docs_rows.append(row)

    # ---- tables -------------------------------------------------------------------------------
    def table(self, id: str, df: pd.DataFrame, *, status: str, sources: Sequence[Union[Source, SourceSet]],
              script: str, notes: str = "", title: Optional[str] = None, priority: Optional[str] = None,
              docs: Optional[Dict[str, Any]] = None, tier: str = "", caption: str = "",
              md_digits: int = 4, md_max_rows: Optional[int] = None, slug: Optional[str] = None) -> pd.DataFrame:
        """Write ``tables/<id>_<slug>.csv`` and ``.md`` and register the item.

        Args:
            id: Item id (``T01`` ... ``T59``).
            df: The table (full precision; every column must be documented via ``docs`` or the dictionary).
            status: found / regenerated / generated / derived.
            sources: Source files (``src(...)`` / ``srcs(...)`` / pack items via ``pack_src``).
            script: ``'sec_x.py:build_fn'``.
            notes: Transformation applied (median/IQR over seeds, filtering, ...).
            title: Override of the default title.
            priority: Override of the default priority.
            docs: Column documentation overrides ``{col: str | dict(definition, units, normalization, tier, source)}``.
            tier: Tier label of the table (``final``, ``development``, ``final and development``, ``n/a``).
            caption: Factual description shown above the table in the .md file.
            md_digits: Significant digits in the .md file.
            md_max_rows: Row cap of the .md file (the CSV is always complete).
            slug: Override of the file-name slug.

        Returns:
            The written DataFrame.
        """
        title, priority, typ = self._defaults(id, title, priority)
        if typ != "table":
            raise ValueError(f"{id} is a {typ}")
        df = _clean_df(df)
        stem = f"{id}_{slug or spec.slug(title)}"
        csv_rel, md_rel = f"{PACK_REL}/tables/{stem}.csv", f"{PACK_REL}/tables/{stem}.md"
        df.to_csv(self.dir / "tables" / f"{stem}.csv", index=False, lineterminator="\n")
        head = [f"# {id}: {title}", "", f"- priority: {priority}; status: {status}" + (f"; tier: {tier}" if tier else ""),
                f"- sources: " + (", ".join(f"`{s.path}`" if isinstance(s, Source) else f"`{s.label}` ({len(s.files)} files)"
                                            for s in sources) or "none"),
                f"- built by: `{SCRIPT_DIR_REL}/{script}`; base commit `{BASE_COMMIT[:7]}`"]
        if notes:
            head.append(f"- transformation: {notes}")
        if caption:
            head += ["", caption]
        (self.dir / "tables" / f"{stem}.md").write_text("\n".join(head) + "\n\n" + df_to_md(df, md_digits, md_max_rows) + "\n",
                                                        encoding="utf-8")
        self._docs(id, df, docs, tier)
        self._register(id, "table", title, priority, status, [csv_rel, md_rel], sources, script, notes, tier)
        return df

    def found_table(self, id: str, rel_src: str, *, script: str, notes: str = "", title: Optional[str] = None,
                    priority: Optional[str] = None, docs: Optional[Dict[str, Any]] = None, tier: str = "",
                    caption: str = "", columns: Optional[Sequence[str]] = None, status: str = "found",
                    md_digits: int = 4, md_max_rows: Optional[int] = 60) -> pd.DataFrame:
        """Use an existing CSV as is: copy it (or reference it when large) and render a .md view.

        Args:
            id: Item id.
            rel_src: Repo-relative path of the existing CSV.
            columns: If given, the pack file keeps only these columns (then status should be ``derived``/``regenerated``).
            status: ``found`` for a byte-identical copy.

        Returns:
            The DataFrame read from the source.
        """
        title, priority, typ = self._defaults(id, title, priority)
        s = src(rel_src)
        df = pd.read_csv(abspath(rel_src), float_precision="round_trip")
        stem = f"{id}_{spec.slug(title)}"
        csv_rel, md_rel = f"{PACK_REL}/tables/{stem}.csv", f"{PACK_REL}/tables/{stem}.md"
        files = [md_rel]
        if columns is None and s.nbytes <= MAX_COPY_BYTES:
            shutil.copyfile(abspath(rel_src), self.dir / "tables" / f"{stem}.csv")
            files.insert(0, csv_rel)
            stat_note = "byte-identical copy"
        elif columns is not None:
            df = df[list(columns)]
            df.to_csv(self.dir / "tables" / f"{stem}.csv", index=False, lineterminator="\n")
            files.insert(0, csv_rel)
            stat_note = f"column subset ({len(columns)} columns) of the source"
        else:
            stat_note = f"referenced, not copied ({s.nbytes} bytes > {MAX_COPY_BYTES}); {len(df)} rows x {df.shape[1]} columns"
        head = [f"# {id}: {title}", "", f"- priority: {priority}; status: {status}" + (f"; tier: {tier}" if tier else ""),
                f"- source file: `{s.path}` (sha256 `{s.sha256[:12]}...`; {stat_note})",
                f"- built by: `{SCRIPT_DIR_REL}/{script}`; base commit `{BASE_COMMIT[:7]}`"]
        if notes:
            head.append(f"- transformation: {notes}")
        if caption:
            head += ["", caption]
        (self.dir / "tables" / f"{stem}.md").write_text("\n".join(head) + "\n\n" + df_to_md(df, md_digits, md_max_rows) + "\n",
                                                        encoding="utf-8")
        self._docs(id, df, docs, tier)
        self._register(id, "table", title, priority, status, files, [s], script, f"{stat_note}. {notes}".strip(), tier)
        return df

    def data(self, id: str, df: pd.DataFrame, *, status: str, sources: Sequence[Union[Source, SourceSet]],
             script: str, notes: str = "", title: Optional[str] = None, docs: Optional[Dict[str, Any]] = None,
             tier: str = "", slug: Optional[str] = None) -> pd.DataFrame:
        """Write a consolidated per-run table ``data/<id>_<slug>.csv`` (D01-D09) and register it."""
        title, priority, typ = self._defaults(id, title, None)
        if typ != "data":
            raise ValueError(f"{id} is a {typ}")
        df = _clean_df(df)
        stem = f"{id}_{slug or spec.slug(title)}"
        df.to_csv(self.dir / "data" / f"{stem}.csv", index=False, lineterminator="\n")
        self._docs(id, df, docs, tier)
        self._register(id, "data", title, priority, status, [f"{PACK_REL}/data/{stem}.csv"], sources, script, notes, tier)
        return df

    # ---- figures --------------------------------------------------------------------------------
    def figure(self, id: str, fig, data: pd.DataFrame, *, status: str, sources: Sequence[Union[Source, SourceSet]],
               script: str, caption: str, notes: str = "", title: Optional[str] = None,
               priority: Optional[str] = None, docs: Optional[Dict[str, Any]] = None, tier: str = "",
               slug: Optional[str] = None, checks: Optional[List[str]] = None) -> None:
        """Save ``figures/<id>_<slug>.pdf/.png/_data.csv/_caption.md`` and register the item.

        Args:
            id: Item id (``F01`` ... ``F24``).
            fig: Matplotlib figure made with ``style.new_figure`` (7 in wide; fonts >= 8 pt are asserted).
            data: Everything plotted, long format, one row per plotted value.
            status: found / regenerated / generated / derived.
            caption: Factual caption (what is plotted, data source, n, tier); no interpretation.
            checks: Statements of the checks of plotted values against the source data (recorded in notes).
        """
        title, priority, typ = self._defaults(id, title, priority)
        if typ != "figure":
            raise ValueError(f"{id} is a {typ}")
        style.assert_fonts(fig)
        stem = f"{id}_{slug or spec.slug(title)}"
        fdir = self.dir / "figures"
        style.save(fig, str(fdir / stem))
        data = _clean_df(data)
        data.to_csv(fdir / f"{stem}_data.csv", index=False, lineterminator="\n")
        (fdir / f"{stem}_caption.md").write_text(f"**{id}. {title}.** {caption.strip()}\n", encoding="utf-8")
        files = [f"{PACK_REL}/figures/{stem}.{e}" for e in ("pdf", "png")] + \
                [f"{PACK_REL}/figures/{stem}_data.csv", f"{PACK_REL}/figures/{stem}_caption.md"]
        self._docs(id, data, docs, tier)
        n = (notes + " " if notes else "") + ("Checks: " + "; ".join(checks) if checks else "")
        self._register(id, "figure", title, priority, status, files, sources, script, n.strip(), tier)

    # ---- numbers ----------------------------------------------------------------------------------------
    def numbers_item(self, kid: str, rows: List[Dict[str, Any]], *, status: str, script: str, notes: str = "",
                     title: Optional[str] = None) -> None:
        """Register key-number group ``kid`` (``K01``..``K19``).

        Each row: ``sub, description, value, unit, normalization, tier, q, source_file, selector, computation``.
        Sources are the distinct ``source_file`` paths of the rows.
        """
        title, priority, typ = self._defaults(kid, title, None)
        if typ != "number":
            raise ValueError(f"{kid} is a {typ}")
        files: Dict[str, Union[Source, SourceSet]] = {}
        for r in rows:
            sf = r["source_file"]
            if sf not in files:
                files[sf] = srcs(sf, label=sf) if "*" in sf else src(sf)
            self.numbers.append({
                "id": f"{kid}.{r['sub']}", "description": r["description"], "value": num(r["value"]),
                "unit": r.get("unit", ""), "normalization": r.get("normalization", ""), "tier": r.get("tier", ""),
                "q": r.get("q", ""), "source file": sf, "selector (row/column)": r.get("selector", ""),
                "computation": r.get("computation", "as is"),
            })
        self._register(kid, "number", title, priority, status, [f"{PACK_REL}/key_numbers.csv"], list(files.values()),
                       script, notes)

    # ---- gaps / consistency / re-evaluation ------------------------------------------------------------------
    def gap(self, id: str, reason: str, tried: str = "", title: Optional[str] = None,
            priority: Optional[str] = None, script: str = "common.py:gap") -> None:
        """Register an item as ``missing`` and record the reason in ``gaps.md``."""
        title, priority, typ = self._defaults(id, title, priority)
        self._register(id, typ, title, priority, "missing", [], [], script, f"MISSING: {reason}")
        self.gaps.append({"id": id, "title": title, "reason": reason, "tried": tried})

    def unknown_value(self, item: str, what: str, why: str) -> None:
        """Record a single ``UNKNOWN`` value inside a built item (listed in ``gaps.md``)."""
        self.gaps.append({"id": item, "title": what, "reason": "UNKNOWN: " + why, "tried": ""})

    def mismatch(self, item: str, quantity: str, pack_value: Any, report_path: str, report_value: Any,
                 comment: str = "") -> None:
        """Record a difference between a pack value and an existing report (for ``consistency.md``)."""
        self.mismatches.append({"item": item, "quantity": quantity, "pack_value": num(pack_value),
                                "report": report_path, "report_value": str(report_value), "comment": comment})

    def crosscheck(self, item: str, df: pd.DataFrame, report_path: str, *, header_has: Sequence[str],
                   key_map: Dict[str, str], value_map: Dict[str, str], heading_has: str = "",
                   tables: Optional[Sequence[int]] = None, label: str = "") -> Dict[str, Any]:
        """Compare report-table cells with the pack DataFrame at the report's displayed precision.

        Args:
            item: Pack item id the DataFrame belongs to.
            df: Pack values (full precision).
            report_path: Existing report (``reports/v2/*.md``).
            header_has: Column names that must all appear in the report table header.
            key_map: ``{report column: df column}`` identifying a row.
            value_map: ``{report column: df column}`` of the numeric values to compare.
            heading_has: Substring that must appear in the nearest heading above the table.
            tables: Optional indices (among matching tables) to use.
            label: Text for the summary line.

        Returns:
            ``{n_compared, n_mismatch, n_unmatched_rows, n_tables}``; mismatches are recorded.
        """
        reps = [t for t in parse_md_tables(report_path)
                if all(h in t["header"] for h in header_has) and heading_has.lower() in t["heading"].lower()]
        if tables is not None:
            reps = [reps[i] for i in tables]
        n_cmp = n_bad = n_unm = 0
        for t in reps:
            hdr = t["header"]
            for row in t["rows"]:
                if len(row) != len(hdr):
                    continue
                rec = dict(zip(hdr, row))
                mask = pd.Series(True, index=df.index)
                for rc, dc in key_map.items():
                    cell = rec[rc]
                    colv = df[dc]
                    kv = parse_num(cell)
                    if kv is not None and pd.api.types.is_numeric_dtype(colv):
                        mask &= (colv.astype(float) == kv)
                    else:
                        mask &= (colv.astype(str) == cell)
                hit = df[mask]
                if len(hit) != 1:
                    n_unm += 1
                    continue
                for rc, dc in value_map.items():
                    cell = rec.get(rc, "")
                    if parse_num(cell) is None:
                        continue
                    x = float(hit.iloc[0][dc])
                    n_cmp += 1
                    if not consistent(x, cell):
                        n_bad += 1
                        keys = ", ".join(f"{dc_}={hit.iloc[0][dc_]}" for dc_ in key_map.values())
                        self.mismatch(item, f"{dc} [{keys}]", x, f"{report_path} (line {t['line']}, '{t['heading']}')", cell)
        res = {"item": item, "report": report_path, "label": label or ",".join(value_map.values()),
               "n_tables": len(reps), "n_compared": n_cmp, "n_mismatch": n_bad, "n_unmatched_rows": n_unm}
        self.crosschecks.append(res)
        return res

    def manual_check(self, item: str, report: str, label: str, n_compared: int, n_mismatch: int, n_tables: int = 0) -> None:
        """Record a comparison with report prose or an irregular table (the caller records each mismatch with ``mismatch``)."""
        self.crosschecks.append({"item": item, "report": report, "label": label, "n_tables": n_tables, "n_compared": int(n_compared),
                                 "n_mismatch": int(n_mismatch), "n_unmatched_rows": 0})

    def reeval(self, item: str, policy_source: str, tier: str, wall_s: float, n_calls: int, purpose: str,
               kind: str = "forward pass", note: str = "") -> None:
        """Log a forward pass / verifier evaluation (``reevaluations.csv``)."""
        self.reevals.append({"item": item, "kind": kind, "policy_source": policy_source, "tier": tier,
                             "commit": BASE_COMMIT, "n_calls": int(n_calls), "wall_s": round(float(wall_s), 3),
                             "purpose": purpose, "note": note})

    # ---- reading back ---------------------------------------------------------------------------------------
    def pack_table(self, id: str) -> pd.DataFrame:
        """Read a pack table written earlier (by this or another module) from ``tables/``."""
        hits = sorted((self.dir / "tables").glob(f"{id}_*.csv"))
        if not hits:
            raise FileNotFoundError(f"pack table {id} has not been built")
        return pd.read_csv(hits[0], float_precision="round_trip")

    def pack_src(self, id: str) -> Source:
        """Source record of a pack file (derived items cite the pack tables they were computed from)."""
        for sub in ("tables", "data"):
            hits = sorted((self.dir / sub).glob(f"{id}_*.csv"))
            if hits:
                return Source(f"{PACK_REL}/{sub}/{hits[0].name}", sha256_file(hits[0]), hits[0].stat().st_size)
        raise FileNotFoundError(f"pack item {id} has not been built")

    # ---- fragments -------------------------------------------------------------------------------------------
    def save_fragment(self) -> Path:
        """Write this module's registry to ``_build/fragments/<module>.json``."""
        path = self.dir / "_build" / "fragments" / f"{self.module}.json"
        frag = {"module": self.module, "items": self.items, "numbers": self.numbers, "gaps": self.gaps,
                "mismatches": self.mismatches, "crosschecks": self.crosschecks, "reevals": self.reevals,
                "docs": self.docs_rows}
        path.write_text(json.dumps(frag, indent=1, default=str), encoding="utf-8")
        und = [(d["item"], d["column"]) for d in self.docs_rows if d["definition"] == "UNDOCUMENTED"]
        by_status: Dict[str, List[str]] = {}
        for it in self.items:
            by_status.setdefault(it["status"], []).append(it["id"])
        print(f"[{self.module}] items by status: " + "; ".join(f"{k}={','.join(v)}" for k, v in sorted(by_status.items())), flush=True)
        print(f"[{self.module}] key-number rows: {len(self.numbers)}; gaps/UNKNOWN: {len(self.gaps)}; cross-checks: "
              f"{sum(c['n_compared'] for c in self.crosschecks)} cells, {len(self.mismatches)} mismatches; "
              f"re-evaluation rows: {len(self.reevals)}", flush=True)
        if und:
            print(f"[{self.module}] UNDOCUMENTED columns ({len(und)}): {und}", flush=True)
        return path


# ----------------------------------------------------------------------------------------------
# finalize: merge fragments into the pack-level files
# ----------------------------------------------------------------------------------------------

def _load_fragments(pack_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for p in sorted((pack_dir / "_build" / "fragments").glob("*.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]], cols: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def finalize(pack_dir: Path = PACK_DIR) -> Dict[str, Any]:
    """Merge all module fragments and write manifest, key numbers, dictionary, gaps, consistency, README.

    Returns:
        A summary dict (counts by status, undocumented columns, unbuilt ids).
    """
    frags = _load_fragments(pack_dir)
    items: Dict[str, Dict[str, Any]] = {}
    for f in frags:
        for it in f["items"]:
            if it["id"] in items:
                raise ValueError(f"item {it['id']} built by two modules")
            items[it["id"]] = it
    gaps = [g for f in frags for g in f["gaps"]]
    # unbuilt ids become explicit 'missing' rows
    unbuilt = [i for i in spec.ORDER if i not in items]
    for i in unbuilt:
        d = spec.ITEMS[i]
        items[i] = {"id": i, "section": spec.SECTION_OF[i], "title": d["title"], "type": d["type"], "priority": d["priority"],
                    "status": "missing", "files": "", "sources": "", "script": "", "commit": BASE_COMMIT,
                    "notes": "MISSING: not built (no builder registered this item)", "tier": ""}
        gaps.append({"id": i, "title": d["title"], "reason": "not built (no builder registered this item)", "tried": ""})
    rows = [items[i] for i in spec.ORDER if i in items]
    _write_csv(pack_dir / "manifest.csv", rows,
               ["id", "section", "title", "type", "priority", "status", "files", "sources", "script", "commit", "notes"])

    nums = sorted((n for f in frags for n in f["numbers"]), key=lambda r: r["id"])
    _write_csv(pack_dir / "key_numbers.csv", nums,
               ["id", "description", "value", "unit", "normalization", "tier", "q", "source file", "selector (row/column)",
                "computation"])

    docs = [d for f in frags for d in f["docs"]]
    docs.sort(key=lambda r: (spec.ORDER_INDEX.get(r["item"], 999),))
    _write_csv(pack_dir / "data_dictionary.csv", docs, ["item", "column", "definition", "units", "normalization", "tier", "source"])
    undocumented = [(d["item"], d["column"]) for d in docs if d["definition"] == "UNDOCUMENTED"]

    # gaps.md
    gl = ["# Gaps: items not found and not generable, and UNKNOWN values", ""]
    if not gaps:
        gl.append("None.")
    for g in sorted(gaps, key=lambda g: spec.ORDER_INDEX.get(g["id"], 999)):
        gl += [f"## {g['id']}: {g['title']}", "", f"- reason: {g['reason']}"] + ([f"- tried: {g['tried']}"] if g.get("tried") else []) + [""]
    (pack_dir / "gaps.md").write_text("\n".join(gl) + "\n", encoding="utf-8")

    # consistency.md
    cl = ["# Consistency: pack values against the existing reports", "",
          "Comparison rule: each pack value is rounded to the number of significant digits that the report shows for the same "
          "cell; a difference beyond that rounding is a mismatch. The old reports are not edited.", ""]
    checks = [c for f in frags for c in f["crosschecks"]]
    mism = [m for f in frags for m in f["mismatches"]]
    cl += [f"- cells compared: {sum(c['n_compared'] for c in checks)}; mismatches: {len(mism)}", "",
           "## Checks performed", "", "| pack item | report | what | tables | cells compared | mismatches | unmatched report rows |",
           "|---|---|---|---|---|---|---|"]
    for c in sorted(checks, key=lambda c: (spec.ORDER_INDEX.get(c["item"], 999), c["report"])):
        cl.append(f"| {c['item']} | {c['report']} | {c['label']} | {c['n_tables']} | {c['n_compared']} | {c['n_mismatch']} | {c['n_unmatched_rows']} |")
    cl += ["", "## Mismatches", ""]
    if not mism:
        cl.append("None among the compared cells.")
    else:
        cl += ["| pack item | quantity | pack value | report | report value | comment |", "|---|---|---|---|---|---|"]
        for m in mism:
            cl.append(f"| {m['item']} | {m['quantity']} | {m['pack_value']} | {m['report']} | {m['report_value']} | {m['comment']} |")
    (pack_dir / "consistency.md").write_text("\n".join(cl) + "\n", encoding="utf-8")

    # reevaluations.csv
    re_rows = [r for f in frags for r in f["reevals"]]
    _write_csv(pack_dir / "reevaluations.csv", re_rows,
               ["item", "kind", "policy_source", "tier", "commit", "n_calls", "wall_s", "purpose", "note"])

    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    _write_readme(pack_dir, rows, counts, re_rows, len(mism), undocumented)
    return {"counts": counts, "undocumented": undocumented, "unbuilt": unbuilt, "n_mismatch": len(mism),
            "n_reeval_rows": len(re_rows), "n_items": len(rows)}


def _write_readme(pack_dir: Path, rows: List[Dict[str, Any]], counts: Dict[str, int], re_rows: List[Dict[str, Any]],
                  n_mismatch: int, undocumented: List[Tuple[str, str]]) -> None:
    by = {r["id"]: r for r in rows}
    L: List[str] = []
    A = L.append
    A("# T=2 v2 report pack (P0 to P6)")
    A("")
    A("Materials for a report on the T=2 v2 work, from the Phase 0 audit through the fresh-seed confirmation: every table, figure "
      "and headline number, with provenance. This directory contains no report prose; captions and notes describe what is shown "
      "and where it comes from.")
    A("")
    A("## Provenance")
    A("")
    A(f"- **Base commit of the pack:** `{BASE_COMMIT}` (HEAD of branch `v2-stagewise-pilots` in the canonical worktree "
      "`.claude/worktrees/pilot-4-stabilization-fb99a2` when the pack was started).")
    A("- **Pack commit:** the commit that adds this directory (`git log --diff-filter=A -- reports/v2/t2_report/README.md`).")
    A("- **Where it was built:** the pack and its scripts were written in the session worktree `.claude/worktrees/t2-v2-report-pack-f51a78` "
      "(branch `claude/t2-v2-report-pack-f51a78`, fast-forwarded to the base commit), not in the canonical worktree, because a hook blocks writes into "
      "other worktrees; the canonical worktree's `results/` is the read-only data root. To bring the pack onto `v2-stagewise-pilots`: from the canonical "
      "worktree, `git merge --ff-only claude/t2-v2-report-pack-f51a78` (or cherry-pick the pack commits).")
    A(f"- **Builder:** `{SCRIPT_DIR_REL}/build_t2_report_pack.py` rebuilds the whole pack from `results/`; the module and its "
      "SHA-256 that built each item are in `manifest.csv` (`script`). `commit` in the manifest is the base commit above.")
    A("- **Data root:** `results/` of the canonical worktree. The arrays, weight exports and full states are untracked there "
      "(size, gitignored `.pt`); the builder reads them read-only (`T2_REPORT_RESULTS_ROOT` or `--results-root`). Every source "
      "path in the pack is repo-relative with the SHA-256 of the bytes that were read.")
    A("- **Rebuild:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fjiang4/tournament_experiment/.venv/bin/python "
      f"{SCRIPT_DIR_REL}/build_t2_report_pack.py --results-root <canonical worktree>/results`")
    A("- No training was run. Forward passes and verifier evaluations that the pack needed are listed in "
      "`reevaluations.csv` (policy source, tier, commit, wall time).")
    A("")
    A("## Files")
    A("")
    A("| file | content |\n|---|---|")
    A("| `manifest.csv` | one row per item: id, section, title, type, priority, status, files, sources (path:sha256), script, commit, notes |")
    A("| `key_numbers.csv` | the headline numbers K01 to K19 with source file, selector and computation |")
    A("| `data_dictionary.csv` | every column of every pack table: definition, units, normalization, tier, source function |")
    A("| `tables/`, `figures/`, `data/` | the items; `provenance/` lists the files behind items with more than 12 sources |")
    A("| `gaps.md` | items not found and not generable, and UNKNOWN values, with reasons |")
    A("| `consistency.md` | comparisons with the existing reports and every mismatch |")
    A("| `reevaluations.csv` | every forward pass / verifier evaluation performed for the pack |")
    A("")
    A("## Status")
    A("")
    A("| status | meaning | items |\n|---|---|---|")
    meaning = {"found": "an existing file used as is (copied, or referenced when large)",
               "regenerated": "rebuilt from existing data for a consistent style; matches the original",
               "generated": "new, from saved data or allowed computation", "derived": "computed from other pack items",
               "missing": "could not be found or generated (see gaps.md)"}
    for s in STATUSES:
        A(f"| {s} | {meaning[s]} | {counts[s]} |")
    A(f"| **total** | | **{sum(counts.values())}** |")
    A("")
    A("## Conventions")
    A("")
    A("- **q** is reported separately (50 and 60). Per-run tables list their seeds explicitly. Development seeds: 10501 to 10510; "
      "fresh confirmation seeds: 20501 to 20520.")
    A("- **Evaluated policy:** the deterministic Beta mean. **Tier:** final unless stated otherwise; every development-tier value "
      "is labelled (`tier` column, `_dev` suffix, or the table header). The pilot learning curves are development-tier (re-evaluations "
      "of the 25-update weight exports); tier-independent metrics (recovery grid, direct policy queries) are marked as such in "
      "`data_dictionary.csv`.")
    A("- **Normalization:** deviation metrics (eta_2, Delta, Gmax_full, EXP_root, dReach, Delta_max_all, dFull) are divided by "
      "Delta W = 4; recovery errors are fractions of e_1*(0) (46.667 / 38.889) or e_2*(0) (70 / 58.333), signed unless stated; "
      "raw effort values are in effort units [0, 100] and labelled raw.")
    A("- **Final checkpoint of each study:** Pilot 1 u400; Pilots 2 and 3 global u1000; Phase A extension u400, u800, u1200, u1600; "
      "Pilot 4 section 2a u1600; section 2b u2200; locked runs end of A (u1600) and end of B (u2200).")
    A("- **Across-seed summaries:** median and IQR (25th to 75th percentile, numpy linear interpolation), plus min and max. "
      "**Paired statistics** reuse the existing bootstrap results (10,000 resamples, numpy seed 20261001; S1 uses seed 20261002).")
    A("- **Figures:** 7 in wide, no text below 8 pt, PDF (TrueType) plus 300 dpi PNG, each with `_data.csv` and `_caption.md`. "
      "Fixed colours (the same in every figure):")
    A("")
    A("| kind | key | colour |\n|---|---|---|")
    for kind, key, hexv in style.palette_doc():
        A(f"| {kind} | {key} | `{hexv}` |")
    A("")
    A("  A figure colours either by arm or by q, never both; q is also encoded by line style (50 solid, 60 dashed) and marker "
      "(circle, square). Palette: the documented 8-slot categorical palette, assignment chosen for the largest worst-pair CVD "
      "separation within each group of arms that co-occur (see `tools/v2/report/style.py`).")
    A("")
    A("## Outline")
    A("")
    for sec, ids in spec.OUTLINE:
        A(f"### {sec}")
        A("")
        A("| id | item | priority | status | files |\n|---|---|---|---|---|")
        for i in ids:
            r = by.get(i)
            if not r:
                continue
            links = []
            for f in [x for x in r["files"].split("; ") if x]:
                links.append(f"[{os.path.basename(f)}]({f[len(PACK_REL) + 1:]})")
            A(f"| {i} | {r['title']} | {r['priority']} | {r['status']} | {', '.join(links) if links else 'none'} |")
        A("")
    A("## Open points of the build")
    A("")
    A(f"- Mismatches against the existing reports: {n_mismatch} (`consistency.md`).")
    A(f"- Items with status `missing`: {counts['missing']} (`gaps.md`).")
    A(f"- Columns without a dictionary entry: {len(undocumented)}.")
    A(f"- Re-evaluation rows: {len(re_rows)} (`reevaluations.csv`).")
    A("")
    (pack_dir / "README.md").write_text("\n".join(L), encoding="utf-8")
