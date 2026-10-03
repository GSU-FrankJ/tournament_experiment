"""Pack items of the locked T=2 protocol (v1.0 and v1.1): definition, criteria, history, timeline,
re-rehearsal checks, confirmation verdict and per-run tables.

Items: T30, T31, F11, T32, T33, T34, T35, T36, T37, T38, T40, D07, D08, D09.

Every value is read from a file (protocol JSON, analysis CSV/JSON, per-run records, read-only git
commit metadata) or recomputed from per-run records with the repository's own functions
(``Run.lr_for`` / ``lr_at``, ``confirmation_analysis.clopper_pearson``,
``make_locked_protocol_v1_1.apply_changes`` / ``diff``, ``v1_0_pass_probability.p_both``). No forward
pass and no verifier evaluation is run.

CSV files are read with ``float_precision="round_trip"``: pandas 3.0's default parser drops the
17th significant digit, so a default read followed by a write would not reproduce the source values.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import common as C
import style
import studies as S

MOD = "sec_locked_a.py"
LK = "results/v2_T2_locked"
CA = f"{LK}/confirmation_analysis"
RA = f"{LK}/rehearsal_analysis"
RA11 = f"{LK}/rehearsal_v1_1_analysis"
P10 = "protocols/v2_T2_locked.json"
P11 = "protocols/v2_T2_locked_v1_1.json"
MD11 = "protocols/v2_T2_locked_v1_1.md"
LOCK = "protocols/LOCK"
REP10 = "reports/v2/protocol_lock_and_rehearsal.md"
REP11 = "reports/v2/protocol_v1_1_confirmation.md"
SUMMARY = "reports/v2/summary.md"
ENTRY = "run/run_v2_T2_locked.py"
STAGEWISE = "run/run_v2_stagewise.py"
LR_AT = "run/run_final_dp_br_round3_dense.py"
CONF_TOOL = "tools/v2/confirmation_analysis.py"
CHECK_TOOL = "tools/v2/v1_1_rehearsal_checks.py"
GEN11 = "tools/v2/make_locked_protocol_v1_1.py"
PASSPROB_TOOL = "tools/v2/v1_0_pass_probability.py"
V2M = "utils/v2_metrics.py"
QS = (50, 60)
WALL_KEYS = ("time_sec", "update_wall_sec", "elapsed_phase_wall_sec", "verifier_sec")
NUM = r"[0-9]+(?:\.[0-9]+)?"   # a number without a trailing sentence period


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def _rt(rel: str) -> pd.DataFrame:
    """CSV read with exact float round trip (``float_precision='round_trip'``)."""
    return pd.read_csv(C.abspath(rel), float_precision="round_trip")


def _js(rel: str) -> Any:
    """JSON file (repo-relative; ``results/...`` resolved against the results root)."""
    with open(C.abspath(rel), encoding="utf-8") as fh:
        return json.load(fh)


def _repo_on_path() -> None:
    """Make the repository root importable (``run.*``, ``utils.*``)."""
    if str(C.REPO) not in sys.path:
        sys.path.append(str(C.REPO))


def _tool(rel: str, name: str) -> ModuleType:
    """Import a repository script by path under a private module name (no sys.path change)."""
    spec = importlib.util.spec_from_file_location(name, str(C.REPO / rel))
    if spec is None or spec.loader is None:
        raise ImportError(rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _def_line(rel: str, name: str) -> str:
    """``'rel:name (line n)'`` of the first ``def name(`` in a repository file."""
    for i, ln in enumerate(C.abspath(rel).read_text(encoding="utf-8").splitlines(), 1):
        if re.match(rf"^\s*def {re.escape(name)}\(", ln):
            return f"{rel}:{name} (line {i})"
    raise KeyError(f"{name} is not defined in {rel}")


def _git_commit(sha: str) -> Tuple[str, str]:
    """(full hash, committer date in UTC ``YYYY-MM-DDTHH:MM:SS``) of a commit (read-only git log)."""
    out = subprocess.run(["git", "-C", str(C.REPO), "log", "-1", "--format=%H %cI", sha], capture_output=True,
                         text=True, check=True, timeout=30).stdout.split()
    dt = datetime.fromisoformat(out[1]).astimezone(timezone.utc)
    return out[0], dt.strftime("%Y-%m-%dT%H:%M:%S")


def _mtime_utc(rel: str) -> str:
    """File modification time in UTC (filesystem metadata)."""
    ts = C.abspath(rel).stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def _vtxt(v: Any) -> str:
    """A JSON value as text: strings as they are, everything else as a JSON literal."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _flatten(obj: Any, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """(JSON pointer, leaf) pairs; lists of scalars and empty objects stay whole, lists of objects are indexed."""
    if isinstance(obj, dict) and obj:
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}/{k}")
    elif isinstance(obj, list) and any(isinstance(x, (dict, list)) for x in obj):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}/{i}")
    else:
        yield prefix, obj


def _ybool(v: Any) -> str:
    """Report rendering of a cell value (booleans as yes/no, like the analysis scripts)."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if bool(v) else "no"
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)


def _tables(report: str, header_has: Sequence[str], heading_has: str = "") -> List[Dict[str, Any]]:
    """Markdown tables of a report whose header has all ``header_has`` and whose heading has ``heading_has``."""
    return [t for t in C.parse_md_tables(report)
            if all(h in t["header"] for h in header_has) and heading_has.lower() in t["heading"].lower()]


class _Cmp:
    """Manual comparison of pack values with report cells or prose, at the report's displayed precision."""

    def __init__(self, pack: C.Pack, item: str, report: str, label: str) -> None:
        self.pack, self.item, self.report, self.label = pack, item, report, label
        self.n, self.bad, self.unmatched, self.tables = 0, [], 0, 0

    def num(self, quantity: str, value: float, cell: str, where: str = "", comment: str = "") -> None:
        """Compare a number with a report cell (mismatch if it does not round to the cell)."""
        self.n += 1
        if not C.consistent(float(value), str(cell)):
            self.bad.append((quantity, value, cell, where, comment))

    def text(self, quantity: str, value: Any, cell: str, where: str = "", comment: str = "") -> None:
        """Compare a value rendered as text (yes/no for booleans) with a report cell."""
        self.n += 1
        if _ybool(value) != str(cell).strip():
            self.bad.append((quantity, _ybool(value), cell, where, comment))

    def flag(self, quantity: str, ok: bool, pack_value: Any, report_value: Any, where: str = "",
             comment: str = "") -> None:
        """Record the outcome of a comparison made by the caller."""
        self.n += 1
        if not ok:
            self.bad.append((quantity, pack_value, report_value, where, comment))

    def done(self) -> None:
        """Record the mismatches and one cross-check summary row."""
        for quantity, pv, rv, where, comment in self.bad:
            self.pack.mismatch(self.item, quantity, pv, f"{self.report}{(' (' + where + ')') if where else ''}", rv,
                               comment)
        self.pack.crosschecks.append({"item": self.item, "report": self.report, "label": self.label,
                                      "n_tables": self.tables, "n_compared": self.n, "n_mismatch": len(self.bad),
                                      "n_unmatched_rows": self.unmatched})


def _crosscheck_text(pack: C.Pack, item: str, df: pd.DataFrame, report: str, header_has: Sequence[str],
                     key_map: Dict[str, str], text_map: Dict[str, str], heading_has: str = "", label: str = "") -> None:
    """Compare text / boolean cells of report tables with the pack DataFrame (booleans as yes/no)."""
    cmp = _Cmp(pack, item, report, label or "text cells: " + ", ".join(text_map.values()))
    for t in _tables(report, header_has, heading_has):
        cmp.tables += 1
        hdr = t["header"]
        for row in t["rows"]:
            if len(row) != len(hdr):
                continue
            rec = dict(zip(hdr, row))
            mask = pd.Series(True, index=df.index)
            for rc, dc in key_map.items():
                kv = C.parse_num(rec[rc])
                mask &= (df[dc].astype(float) == kv) if kv is not None else (df[dc].astype(str) == rec[rc])
            hit = df[mask]
            if len(hit) != 1:
                cmp.unmatched += 1
                continue
            keys = ", ".join(f"{dc}={hit.iloc[0][dc]}" for dc in key_map.values())
            for rc, dc in text_map.items():
                if rc in rec:
                    cmp.text(f"{dc} [{keys}]", hit.iloc[0][dc], rec[rc], f"line {t['line']}, '{t['heading']}'")
    cmp.done()


def _numeric_cols(df: pd.DataFrame, exclude: Sequence[str] = ()) -> List[str]:
    """Numeric, non-boolean columns of a DataFrame."""
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def _grep_text(patterns: Sequence[str], needles: Sequence[str]) -> Tuple[List[str], int]:
    """Text files (json/txt/out/log/csv/md) matching repo-relative globs that contain any needle; (hits, n files)."""
    files = []
    for pat in patterns:
        root, sub = (C.RESULTS_ROOT, pat[len("results/"):]) if pat.startswith("results/") else (C.REPO, pat)
        files += sorted(C.relpath(p) for p in root.glob(sub)
                        if p.is_file() and p.suffix.lstrip(".") in ("json", "txt", "out", "log", "csv", "md"))
    hits = [f for f in files if any(nd in C.abspath(f).read_text(encoding="utf-8", errors="replace") for nd in needles)]
    return hits, len(files)


def _opt(col: pd.Series, kind: type) -> pd.Series:
    """Optional ints / bools as Python values with None for missing (object dtype; empty CSV and .md cells)."""
    return pd.Series([None if (v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA) else kind(v)
                      for v in col], index=col.index, dtype=object)


STAT_COLS = ("min", "p10", "p25", "median", "p75", "p90", "max", "q25", "q75", "mean", "sd")


def _docs_final(cols: Sequence[str], docs: Dict[str, Any], table_tier: str, default_tier: str = "n/a",
                stat_tier: str = "") -> Dict[str, Any]:
    """Column docs with an explicit tier for every column.

    Explicit ``docs`` tiers win. Otherwise: A_dmf_*/B_dmf_* -> dev - final (unless tier-independent); A_*/B_* -> final
    (end-of-phase values are final tier); dec_* -> final (the residual band uses the final tier); summary statistics ->
    ``stat_tier``; columns without any tier -> ``default_tier``; else the global dictionary's tier.
    """
    import dictionary as D
    out: Dict[str, Any] = {}
    for c in cols:
        d = docs.get(c)
        d = {"definition": d} if isinstance(d, str) else dict(d or {})
        if not d.get("tier"):
            base = D.lookup(c, table_tier) or {}
            bt = base.get("tier", "")
            if base and c.startswith(("A_dmf_", "B_dmf_")):
                d["tier"] = bt if bt == "tier-independent" else "dev - final"
            elif base and c.startswith(("A_", "B_")):
                d["tier"] = "final" if bt in (table_tier, "unspecified", "n/a", "") else bt
            elif c.startswith("dec_"):
                d["tier"] = "final (residual band on the final tier)"
            elif stat_tier and c in STAT_COLS:
                d["tier"] = stat_tier
            elif not bt:
                d["tier"] = default_tier
        if d:
            out[c] = d
    return out


def _table(pack: C.Pack, id: str, df: pd.DataFrame, *, docs: Dict[str, Any], tier: str, default_tier: str = "n/a",
           stat_tier: str = "", **kw: Any) -> pd.DataFrame:
    """``pack.table`` with every column's tier made explicit (``_docs_final``)."""
    return pack.table(id, df, docs=_docs_final(df.columns, docs, tier, default_tier, stat_tier), tier=tier, **kw)


def _found(pack: C.Pack, id: str, rel: str, *, docs: Dict[str, Any], tier: str, default_tier: str = "n/a",
           stat_tier: str = "", **kw: Any) -> pd.DataFrame:
    """``pack.found_table`` (byte-identical copy) with every column's tier made explicit."""
    cols = list(_rt(rel).columns)
    return pack.found_table(id, rel, docs=_docs_final(cols, docs, tier, default_tier, stat_tier), tier=tier, **kw)


def _figure(pack: C.Pack, id: str, fig: Any, data: pd.DataFrame, *, docs: Dict[str, Any], tier: str,
            default_tier: str = "n/a", **kw: Any) -> None:
    """``pack.figure`` with every data column's tier made explicit."""
    pack.figure(id, fig, data, docs=_docs_final(data.columns, docs, tier, default_tier), tier=tier, **kw)


def _status_utc(s: str) -> str:
    """``'YYYY-MM-DD HH:MM:SS'`` (status.json, local clock = UTC) as ``YYYY-MM-DDTHH:MM:SS``."""
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S")


# ----------------------------------------------------------------------------------------------
# column documentation
# ----------------------------------------------------------------------------------------------

_CA_SRC = CONF_TOOL + ":analyse"
DOC_PER_RUN: Dict[str, Any] = {
    "status": dict(definition="Run classification by the pre-registered analysis: completed (status done, exit code 0), "
                   "missing, failed_exception, incomplete, global_rng_violation", units="label",
                   source=CONF_TOOL + ":classify"),
    "exit_code": dict(definition="Exit code in the run's status.json (0 = success, 1 = pipeline exception, 5 = global-RNG "
                      "violation)", units="integer", source=CONF_TOOL + ":classify"),
    "status_info": dict(definition="Traceback tail (failed_exception) or violation list (global_rng_violation) as JSON "
                        "text; empty for completed runs", units="text", source=CONF_TOOL + ":classify"),
    "crashed_attempts": dict(definition="Infrastructure-crash attempts moved to <root>/crashed/q{q}/seed{s}_attempt*/ "
                             "under the crash rule; empty = none", units="paths", source=_CA_SRC),
    "eta_final": dict(definition="eta_2/DW at the end of Phase A, final tier, recomputed from gateA_final.npz as "
                      "max(v_t2_delta)/DW (G-A criterion, threshold 0.005)", units="Delta W (dimensionless)",
                      normalization="divided by Delta W = 4", tier="final", source=CONF_TOOL + ":recompute"),
    "eta_dev": dict(definition="eta_2/DW at the end of Phase A, development tier (gateA_development.npz)",
                    units="Delta W (dimensionless)", normalization="divided by Delta W = 4", tier="development",
                    source=CONF_TOOL + ":recompute"),
    "eta_dev_minus_final": dict(definition="eta_dev - eta_final, signed (G-N uses the absolute value, threshold 0.001)",
                                units="Delta W (dimensionless)", normalization="divided by Delta W = 4",
                                tier="dev - final", source=CONF_TOOL + ":recompute"),
    "rmse": dict(definition="RMSE of e_hat_2 - e2* over the recovery-grid nodes with |d| < 2q, divided by e2*(0), end of "
                 "Phase A (G-A criterion, threshold 0.05); from the recovery arrays of gateA_final.npz",
                 units="fraction of e_2*(0)", normalization="divided by e_2*(0)", tier="tier-independent",
                 source=CONF_TOOL + ":recompute"),
    "tail": dict(definition="Mean of e_hat_2 over the recovery-grid nodes with |d| >= 2q divided by e2*(0), end of Phase A "
                 "(G-A criterion, threshold 0.02); raw tail mean = tail x e2*(0) (70 at q=50, 58.333 at q=60)",
                 units="fraction of e_2*(0)", normalization="divided by e_2*(0)", tier="tier-independent",
                 source=CONF_TOOL + ":recompute"),
    "gmax_final": dict(definition="Gmax_full/DW at the end of Phase B, final tier: max over t of max(v_t{t}_G)/DW from "
                       "final_final.npz (G-F criterion, threshold 0.01)", units="Delta W (dimensionless)",
                       normalization="divided by Delta W = 4", tier="final", source=CONF_TOOL + ":recompute"),
    "gmax_dev": dict(definition="Gmax_full/DW at the end of Phase B, development tier (final_development.npz)",
                     units="Delta W (dimensionless)", normalization="divided by Delta W = 4", tier="development",
                     source=CONF_TOOL + ":recompute"),
    "gmax_dev_minus_final": dict(definition="gmax_dev - gmax_final, signed (G-N uses the absolute value, threshold 0.001)",
                                 units="Delta W (dimensionless)", normalization="divided by Delta W = 4",
                                 tier="dev - final", source=CONF_TOOL + ":recompute"),
    "s1": dict(definition="|e_hat_1(0) - e1*(0)| / e1*(0) at the end of Phase B, from v_t1_e_hat[0] of final_final.npz "
               "(S1, secondary criterion, threshold 0.10; not part of run pass)", units="fraction of e_1*(0)",
               normalization="divided by e_1*(0), absolute", tier="tier-independent (direct policy query)",
               source=CONF_TOOL + ":recompute"),
    "stage1_rel_err_signed": dict(definition="(e_hat_1(0) - e1*(0)) / e1*(0) at the end of Phase B (signed S1 value)",
                                  units="fraction of e_1*(0)", normalization="divided by e_1*(0), signed",
                                  tier="tier-independent (direct policy query)", source=CONF_TOOL + ":recompute"),
    "G-A_pass": dict(definition="G-A verdict recomputed from the arrays: eta_final <= 0.005 and rmse <= 0.05 and "
                     "tail <= 0.02 (inclusive, full float64)", units="bool", tier="final", source=CONF_TOOL + ":decide"),
    "G-F_pass": dict(definition="G-F verdict (v1.1): gmax_final <= 0.01", units="bool", tier="final",
                     source=CONF_TOOL + ":decide"),
    "G-N_pass": dict(definition="G-N verdict: |eta_dev - eta_final| <= 0.001 and |gmax_dev - gmax_final| <= 0.001",
                     units="bool", tier="development vs final", source=CONF_TOOL + ":decide"),
    "S1_pass": dict(definition="S1 verdict: s1 <= 0.10 (secondary; not part of run pass)", units="bool",
                    tier="tier-independent", source=CONF_TOOL + ":decide"),
    "eta_N_pass": dict(definition="First G-N criterion alone: |eta_dev_minus_final| <= 0.001", units="bool",
                       tier="development vs final", source=_CA_SRC),
    "gmax_N_pass": dict(definition="Second G-N criterion alone: |gmax_dev_minus_final| <= 0.001", units="bool",
                        tier="development vs final", source=_CA_SRC),
    "global_rng": dict(definition="gates.json global_rng.status of the process-global RNG assertions (ok / violation)",
                       units="label", source=_CA_SRC),
    "run_pass": dict(definition="Run pass under v1.1: G-A and G-F and G-N pass, status completed and exit code 0",
                     units="bool", source=_CA_SRC),
    "v1_0_G-F": dict(definition="v1.0 G-F verdict on the same run: gmax_final <= 0.01 and s1 <= 0.10 (reported only)",
                     units="bool", tier="final", source=CONF_TOOL + ":decide"),
    "v1_0_run_pass": dict(definition="v1.0 run pass on the same run: G-A and the v1.0 G-F (reported only; decides nothing)",
                          units="bool", tier="final", source=CONF_TOOL + ":decide"),
    "wall_sec": dict(definition="Wall-clock seconds of the whole run (v2_run_summary.json total_wall_sec)", units="seconds",
                     source=_CA_SRC),
    "outcome": dict(definition="Run outcome: pass, stage2_failure (G-A fails), fail_G-F, fail_G-N, global_rng_violation, "
                    "or the status of a run without gates", units="label", source=_CA_SRC),
}
DOC_REPORTED: Dict[str, Any] = {
    "A_smoothed_share_peak_gap_d0": dict(
        definition="End of Phase A: share of the d = 0 peak gap predicted by action-noise smoothing, (e2*(0) - e_pred(0)) / "
                   "(e2*(0) - e_hat_2(0)), e_pred from the closed form averaged over 400 equal-probability nodes of each "
                   "Beta (Pilot-1 method)", units="ratio", normalization="none", tier="tier-independent",
        source=ENTRY + ":smoothed_share"),
    "dec_band_contiguous": dict(
        definition="Stage-1 decomposition at the end of Phase B: whether the band {e: Delta_1(e) <= Delta_1,min + floor} is "
                   "one contiguous run of sweep nodes (if not, the outermost band points are used)", units="bool",
        normalization="none", tier="final", source=V2M + ":induced_band"),
    "drift_test_pass": dict(
        definition="Snapshot integrity at the end of the run: frozen stage-2 mapping unchanged since the freeze (max abs "
                   "diff 0 for mean, alpha, beta) and snapshot parameters bit-identical to the end-of-A actor "
                   "(drift_test.json pass)", units="bool", normalization="none", tier="tier-independent",
        source=ENTRY + ":run_pipeline"),
}
_LRA = "tools/v2/locked_rehearsal_analysis.py:gates_table"
DOC_GATES_V10: Dict[str, Any] = {
    "protocol_sha256": dict(definition="First 12 hex characters of the SHA-256 of the protocol JSON the run read "
                            "(manifest locked_protocol.sha256)", units="hash", source=_LRA),
    "G-A": dict(definition="v1.0 G-A verdict, final tier (eta_2 <= 0.005, RMSE/e2*(0) <= 0.05, tail mean/e2*(0) <= 0.02)",
                units="bool", tier="final", source=_LRA),
    "G-F": dict(definition="v1.0 G-F verdict, final tier (Gmax_full/DW <= 0.01 and |stage-1 rel. err| <= 0.10)",
                units="bool", tier="final", source=_LRA),
    "G-A_dev": dict(definition="v1.0 G-A verdict evaluated on the development tier (reported only)", units="bool",
                    tier="development", source=_LRA),
    "G-F_dev": dict(definition="v1.0 G-F verdict evaluated on the development tier (reported only)", units="bool",
                    tier="development", source=_LRA),
    "run_pass": dict(definition="v1.0 run pass = G-A and G-F (final tier)", units="bool", tier="final", source=_LRA),
    "outcome": dict(definition="v1.0 outcome: pass, stage2_failure (G-A fails) or fail_G-F", units="label", source=_LRA),
    "wall_sec": dict(definition="Wall-clock seconds of the whole run (v2_run_summary.json total_wall_sec)",
                     units="seconds", source=_LRA),
    "A_smoothed_e_pred_0": dict(definition="End of Phase A: e_pred(0), the closed-form stage-2 effort at d = 0 "
                                "averaged over "
                                "400 equal-probability nodes of each player's Beta (smoothed game)", units="effort units",
                                normalization="none", tier="tier-independent", source=ENTRY + ":smoothed_share"),
    "A_smoothed_e_learned_0": dict(definition="End of Phase A: e_hat_2(0) (Beta mean) used in the smoothed-game share",
                                   units="effort units [0, 100]", normalization="none", tier="tier-independent",
                                   source=ENTRY + ":smoothed_share"),
    "A_smoothed_share_peak_gap_d0": DOC_REPORTED["A_smoothed_share_peak_gap_d0"],
    "dec_band_contiguous": DOC_REPORTED["dec_band_contiguous"],
    "dec_sweep_lo": dict(definition="Stage-1 decomposition: lowest node of the residual sweep grid", units="effort units",
                         normalization="none", tier="final", source=V2M + ":induced_band"),
    "dec_sweep_hi": dict(definition="Stage-1 decomposition: highest node of the residual sweep grid", units="effort units",
                         normalization="none", tier="final", source=V2M + ":induced_band"),
    "drift_test_pass": DOC_REPORTED["drift_test_pass"],
}
for _m, _thr, _what in (("eta_T_over_dw", "0.005", "eta_2/DW"), ("stage2_rmse_pos_over_g2_0", "0.05", "RMSE/e2*(0)"),
                        ("stage2_tail_mean_over_g2_0", "0.02", "tail mean/e2*(0)"),
                        ("Gmax_full_over_dw", "0.01", "Gmax_full/DW"), ("stage1_rel_err_abs", "0.10", "|stage-1 rel. err|")):
    DOC_GATES_V10[f"{_m}_pass"] = dict(definition=f"v1.0 criterion {_what} <= {_thr} on the final-tier value", units="bool",
                                       tier="final", source=_LRA)
for _m, _what, _u, _n in (("stage2_rmse_pos_over_g2_0", "RMSE/e2*(0) over |d| < 2q", "fraction of e_2*(0)",
                           "divided by e_2*(0)"),
                          ("stage2_tail_mean_over_g2_0", "tail mean/e2*(0) over |d| >= 2q", "fraction of e_2*(0)",
                           "divided by e_2*(0)"),
                          ("stage1_rel_err_abs", "|e_hat_1(0) - e1*(0)| / e1*(0)", "fraction of e_1*(0)",
                           "divided by e_1*(0)")):
    for _suf, _which in (("_final", "final-tier evaluation call"), ("_dev", "development-tier evaluation call"),
                         ("_dev_minus_final", "development minus final")):
        DOC_GATES_V10[f"{_m}{_suf}"] = dict(definition=f"{_what}, {_which}; the metric uses the recovery grid / a direct "
                                            "policy query, so dev - final is 0 by construction", units=_u,
                                            normalization=_n, tier="tier-independent", source=_LRA)
DOC_PASS_COUNTS: Dict[str, Any] = {
    "n_expected": "Runs expected for the q (seeds in the analysed block)",
    "n_completed": "Runs classified completed (status done, exit code 0)",
    "n_missing": "Runs without a run directory or status.json",
    "n_failed_exception": "Runs whose status.json state is failed (pipeline exception)",
    "n_incomplete": "Runs with status.json present but not done or failed (e.g. killed)",
    "n_global_rng_violation": "Runs with a process-global RNG assertion violation (exit code 5; counted as failed)",
    "n_pass": "Runs with run_pass (G-A and G-F and G-N, completed, exit code 0)",
    "pass_rate": dict(definition="n_pass / n_expected", units="fraction"),
    "rule": dict(definition="Pass rule applied to the q: '>= 18 of 20' when 20 seeds are expected, otherwise 'not "
                 "applicable'", units="text"),
    "q_passes_rule": dict(definition="Whether n_pass >= 18 (only when the rule applies; empty otherwise)", units="bool"),
    "n_G-A_pass": "Runs passing G-A", "n_G-F_pass": "Runs passing G-F (v1.1: Gmax_full only)",
    "n_G-N_pass": "Runs passing G-N",
    "n_v1_0_run_pass": "Runs passing the v1.0 run rule (G-A and v1.0 G-F) on the same runs (reported only)",
}
DOC_PASS_COUNTS = {k: (dict(definition=v, units="count", source=CONF_TOOL + ":main") if isinstance(v, str)
                       else dict(v, source=CONF_TOOL + ":main")) for k, v in DOC_PASS_COUNTS.items()}
for _k, _t in (("n_pass", "final (G-N: development vs final)"), ("pass_rate", "final (G-N: development vs final)"),
               ("q_passes_rule", "final (G-N: development vs final)"), ("n_G-A_pass", "final"),
               ("n_G-F_pass", "final"), ("n_G-N_pass", "development vs final"), ("n_v1_0_run_pass", "final")):
    DOC_PASS_COUNTS[_k]["tier"] = _t
DOC_S1: Dict[str, Any] = {
    "S1_pass": dict(definition="Runs with |stage-1 rel. err| <= 0.10 (S1 passes)", units="count",
                    source=CONF_TOOL + ":main"),
    "mean_signed": dict(definition="Mean of the signed stage-1 relative error over the runs", units="fraction of e_1*(0)",
                        normalization="divided by e_1*(0), signed", tier="tier-independent", source=CONF_TOOL + ":main"),
    "boot95_lo": dict(definition="Lower end of the 95% percentile-bootstrap CI of mean_signed (10,000 resamples of the "
                      "runs; a fresh numpy.random.default_rng(20261002) for each q, q ascending)",
                      units="fraction of e_1*(0)", normalization="divided by e_1*(0), signed", source=CONF_TOOL + ":main"),
    "boot95_hi": dict(definition="Upper end of the 95% percentile-bootstrap CI of mean_signed", units="fraction of e_1*(0)",
                      normalization="divided by e_1*(0), signed", source=CONF_TOOL + ":main"),
    "median_signed": dict(definition="Median of the signed stage-1 relative error", units="fraction of e_1*(0)",
                          normalization="divided by e_1*(0), signed", source=CONF_TOOL + ":main"),
    "sd_signed": dict(definition="Sample SD (ddof = 1) of the signed stage-1 relative error", units="fraction of e_1*(0)",
                      normalization="divided by e_1*(0)", source=CONF_TOOL + ":main"),
    "bootstrap_resamples": dict(definition="Bootstrap resamples used for boot95_lo/hi", units="count",
                                source=P11 + ":/secondary/bootstrap"),
    "bootstrap_seed": dict(definition="numpy seed of the bootstrap generator", units="integer",
                           source=P11 + ":/secondary/bootstrap"),
}
for _k in ("S1_pass", "mean_signed", "boot95_lo", "boot95_hi", "median_signed", "sd_signed"):
    DOC_S1[_k]["tier"] = "tier-independent"


DOC_DIST: Dict[str, Any] = {
    "root": dict(definition="Results root the runs were read from", units="path", source=CONF_TOOL + ":distributions"),
    "metric": dict(definition="Per-run column summarised (names of per_run.csv: eta_final, eta_dev, eta_dev_minus_final, "
                   "rmse, tail, gmax_final, gmax_dev, gmax_dev_minus_final, s1, stage1_rel_err_signed)", units="text",
                   source=CONF_TOOL + ":distributions"),
}
for _st, _d in (("min", "minimum"), ("p10", "10th percentile"), ("p25", "25th percentile"), ("median", "median"),
                ("p75", "75th percentile"), ("p90", "90th percentile"), ("max", "maximum")):
    for _suf, _who in (("_root", "fresh-seed confirmation (results/v2_T2_locked/confirmation, seeds 20501-20520, n = 20 "
                        "per q)"), ("_compare_root", "v1.1 re-rehearsal on the development seeds (results/v2_T2_locked/"
                                    "rehearsal_v1_1, seeds 10501-10510, n = 10 per q)")):
        DOC_DIST[f"{_st}{_suf}"] = dict(definition=f"{_d} over the runs of the {_who} (pandas linear interpolation)",
                                        units="as the metric", source=CONF_TOOL + ":main (--compare-root)")


# ----------------------------------------------------------------------------------------------
# T30 locked pipeline
# ----------------------------------------------------------------------------------------------

def _role_pipeline(path: str, proto: Dict[str, Any]) -> str:
    """Phase / role label of a /pipeline JSON pointer (builder's classification)."""
    m = re.match(r"^/pipeline/lr_decay/(\d+)/", path)
    if m:
        return f"Phase {proto['pipeline']['lr_decay'][int(m.group(1))]['phase']} (LR decay window)"
    rules = [("/pipeline/phase_A/", "Phase A"), ("/pipeline/freeze", "freeze"), ("/pipeline/phase_B/", "Phase B"),
             ("/pipeline/flags", "flags (expected reward in A and B; the other three act in Phase B, see flags_note)"),
             ("/pipeline/lr_form", "Phase A and B (LR form)"), ("/pipeline/evaluated_candidate", "evaluated candidate"),
             ("/pipeline/outputs", "outputs"), ("/pipeline/", "pipeline")]
    for pre, lab in rules:
        if path.startswith(pre):
            return lab
    return "pipeline"


def _role_record(rel: str) -> str:
    """Phase / role label of a per-q record field (relative pointer below /records/<q>)."""
    exact = {
        "/protocol/phase_caps/A": "Phase A (budget, updates)", "/protocol/phase_caps/B": "Phase B (budget, updates)",
        "/protocol/phase_caps/C": "not run (Phase C; pipeline/not_used)",
        "/protocol/verifier_timeout/A": "Phase A (verifier cadence; would-have-fired only)",
        "/protocol/verifier_timeout/B": "Phase B (verifier cadence; would-have-fired only)",
        "/protocol/verifier_timeout/C": "not run (Phase C; pipeline/not_used)",
        "/protocol/phase_thr_over_dw/A": "Phase A (legacy phase rule; would-have-fired only)",
        "/protocol/phase_thr_over_dw/B": "Phase B (legacy phase rule; would-have-fired only)",
        "/protocol/phase_thr_over_dw/C": "not run (Phase C; pipeline/not_used)",
        "/protocol/phase_c_root": "not run (Phase C; pipeline/not_used)",
        "/protocol/phase_c_es": "not run (Phase C; pipeline/not_used)",
        "/protocol/episodes_per_update": "Phase A and B (episodes per update)",
        "/protocol/snapshot_every": "Phase A and B (lagged opponent refresh, global updates)",
        "/protocol/es_bin_width": "Phase A (exploring starts)", "/es_bins_stage2": "Phase A (exploring starts)",
        "/protocol/weights_every": "Phase A and B (weight exports, global updates)",
        "/protocol/recovery_step": "evaluated candidate (recovery grid step)",
        "/obs_encoding": "network input (Phase A and B)", "/action_mapping": "policy (Phase A and B)",
        "/action_sampling": "policy (Phase A and B)",
        "/mean_extraction": "evaluated candidate (Beta mean extraction)",
        "/lr_schedule/ab_lr": "Phase A and B (LR before the decay window, lr_at)",
        "/smoke_overrides": "record bookkeeping", "/group": "record bookkeeping",
        "/run": "set per run (run name)", "/seed": "set per run (seed)", "/output_dir": "set per run (output dir)",
        "/dw": "game", "/B": "game", "/domain_half_stage2": "game", "/T": "game", "/q": "game",
        "/protocol/w_h": "game (protocol copy)", "/protocol/w_l": "game (protocol copy)",
        "/protocol/k": "game (protocol copy)", "/protocol/e_min": "game (protocol copy)",
        "/protocol/e_max": "game (protocol copy)", "/device": "numerics", "/torch_threads": "numerics",
        "/lr_schedule/kind": "record LR schedule (replaced by the decay window in Run.lr_for)",
        "/lr_schedule/linear_denominator": "record LR schedule (replaced by the decay window in Run.lr_for)",
        "/protocol/k_stop": "not run (Phase C stop rule; pipeline/not_used)",
    }
    for k in ("final_thr_over_dw", "refine_thr_over_dw", "sensitivity_thr_over_dw", "dense_conc_step"):
        exact[f"/protocol/{k}"] = "not used by run/run_v2_T2_locked.py (final_evaluation of mode full only)"
    if rel in exact:
        return exact[rel]
    # prefixes, most specific first; every prefix ends with '/' or '_'
    pre = [("/game/", "game"), ("/ppo/", "PPO and network (Phase A and B)"),
           ("/protocol/rng_namespaces/", "RNG streams (Phase A and B)"),
           ("/protocol/direct_rollout_", "not used by run/run_v2_T2_locked.py (final_evaluation of mode full only)"),
           ("/protocol/", "verifier cadence and legacy phase rule (Phase A and B; fixed budget, would-have-fired only)"),
           ("/verifier/", "evaluated candidate (verifier tiers)"), ("/dtypes/", "numerics"),
           ("/versions/", "environment versions (checked by Run)"),
           ("/lr_schedule/c_", "record LR schedule, phase-C fields (replaced by the decay window in Run.lr_for)"),
           ("/lr_schedule/", "Phase A and B (LR application rules, checked by Run)")]
    for p, lab in pre:
        if rel.startswith(p):
            return lab
    return "record"


def build_t30(pack: C.Pack) -> None:
    """T30: every setting of Phase A, the freeze, Phase B and the evaluated candidate (protocol v1.1 JSON)."""
    proto = _js(P11)
    p10 = _js(P10)
    rows: List[Dict[str, Any]] = []

    def add(role: str, setting: str, v50: Any, v60: Any, src: str) -> None:
        rows.append({"role": role, "setting": setting, "value_q50": _vtxt(v50), "value_q60": _vtxt(v60),
                     "same_for_both_q": _vtxt(v50) == _vtxt(v60), "source": src})

    for k in ("version", "base_commit", "q_values"):
        add("protocol", f"/{k}", proto[k], proto[k], f"{P11}:/{k}")
    for path, v in _flatten(proto["pipeline"], "/pipeline"):
        add(_role_pipeline(path, proto), path, v, v, f"{P11}:{path}")
    for gate in ("G-A", "G-F"):
        for k in ("stage", "candidate", "tier"):
            v = proto["gates"][gate][k]
            add("evaluated candidate", f"/gates/{gate}/{k}", v, v, f"{P11}:/gates/{gate}/{k}")
    for path, v in _flatten(proto["stage1_decomposition"], "/stage1_decomposition"):
        add("evaluated candidate (stage-1 decomposition, end of B)", path, v, v, f"{P11}:{path}")
    for path, v in _flatten(proto["reported_not_gated"], "/reported_not_gated"):
        add("evaluated candidate (reported, not gated)", path, v, v, f"{P11}:{path}")
    add("evaluated candidate (reported, not gated)", "/peak_error", proto["peak_error"], proto["peak_error"],
        f"{P11}:/peak_error")
    f50 = dict(_flatten(proto["records"]["50"]))
    f60 = dict(_flatten(proto["records"]["60"]))
    if list(f50) != list(f60):
        raise ValueError("records of q=50 and q=60 have different fields")
    for rel in f50:
        add(_role_record(rel), f"/records/<q>{rel}", f50[rel], f60[rel], f"{P11}:/records/50{rel}; /records/60{rel}")
    for k, v in proto["threads_per_process"].items():
        add("numerics", f"/threads_per_process/{k}", v, v, f"{P11}:/threads_per_process/{k}")
    add("record bookkeeping", "/record_sources/<q>", proto["record_sources"]["50"], proto["record_sources"]["60"],
        f"{P11}:/record_sources")
    # hash of the locked protocol file against protocols/LOCK
    lock = _lock_records()
    rec11 = [r for r in lock if r.get("protocol") == P11][-1]
    sha_file = C.src(P11).sha256
    add("protocol", "SHA-256 of protocols/v2_T2_locked_v1_1.json (computed)", sha_file, sha_file,
        f"{P11} (file bytes); LOCK record protocol_sha256 = {rec11['protocol_sha256']} (equal: "
        f"{sha_file == rec11['protocol_sha256']})")
    df = pd.DataFrame(rows)
    # v1.0 identity of the pipeline and the records (the v1.1 report says they did not change)
    same_records = p10["records"] == proto["records"]
    pipe_diff = sorted(k for k in set(p10["pipeline"]) | set(proto["pipeline"])
                       if p10["pipeline"].get(k) != proto["pipeline"].get(k))
    notes = ("Compiled from the locked protocol JSON (source: protocol text, every value read from the JSON, nothing "
             "paraphrased): /pipeline flattened to JSON pointers (lists of scalars kept whole, lists of objects indexed), "
             "the G-A/G-F stage/candidate/tier, /stage1_decomposition, /reported_not_gated, /peak_error, every field of "
             "/records/50 and /records/60 aligned by pointer, /threads_per_process and /record_sources. 'role' is the "
             "builder's classification of each pointer (Phase A, freeze, Phase B, evaluated candidate, ...), read from the "
             "JSON structure and the code paths run/run_v2_stagewise.py:Run.lr_for (record LR fields replaced by the decay "
             "window), Run.run_phase (verifier cadence, legacy phase rule; k_stop only in Phase C) and "
             "run/run_final_dp_br.py:final_evaluation (direct rollout and final/refine/sensitivity thresholds, mode full "
             "only; not called by run/run_v2_T2_locked.py). Values are JSON literals "
             f"as text. Protocol v1.0 vs v1.1: records identical = {same_records}; /pipeline keys that differ = "
             f"{pipe_diff}.")
    _table(pack, "T30", df, status="generated",
               sources=[C.src(P11), C.src(P10), C.src(LOCK), C.src(STAGEWISE), C.src("run/run_final_dp_br.py")],
               script=f"{MOD}:build_t30", notes=notes, tier="n/a",
               docs={"role": "Phase or role of the setting (builder's classification of the JSON pointer)",
                     "setting": "JSON pointer of the setting in the protocol (per-q record fields as /records/<q>/...)",
                     "value_q50": dict(definition="Value at q = 50 as a JSON literal (text)", units="as the setting"),
                     "value_q60": dict(definition="Value at q = 60 as a JSON literal (text)", units="as the setting"),
                     "same_for_both_q": dict(definition="Whether the q = 50 and q = 60 values are identical", units="bool"),
                     "source": "File and JSON pointer(s) the value was read from"},
               caption="Locked v1.1 pipeline: one row per JSON setting; per-q record values side by side.")
    # cross-check: the v1.1 report states that the pipeline (except outputs/entry_point_reads) and the records did not change
    cmp = _Cmp(pack, "T30", REP11, "1.2 'Nothing else changed': records and pipeline v1.0 vs v1.1")
    cmp.flag("records v1.0 == v1.1", same_records, same_records, "identical (report 1.2)")
    cmp.flag("pipeline keys changed v1.0 -> v1.1", pipe_diff == ["entry_point_reads", "outputs"], pipe_diff,
             "only /pipeline/entry_point_reads and /pipeline/outputs (report 1.2)")
    cmp.done()


def _lock_records() -> List[Dict[str, Any]]:
    """All JSON records of protocols/LOCK (concatenated JSON objects)."""
    txt, dec, out, i = C.abspath(LOCK).read_text(encoding="utf-8"), json.JSONDecoder(), [], 0
    while True:
        while i < len(txt) and txt[i].isspace():
            i += 1
        if i >= len(txt):
            return out
        obj, i = dec.raw_decode(txt, i)
        out.append(obj)


# ----------------------------------------------------------------------------------------------
# T31 criteria
# ----------------------------------------------------------------------------------------------

def build_t31(pack: C.Pack) -> None:
    """T31: G-A, G-F, G-N, S1, run pass and pass rule, seeds, bootstrap / Clopper-Pearson settings, crash rule."""
    p = _js(P11)
    g, sec, conf = p["gates"], p["secondary"], p["confirmation"]
    verd = _def_line(ENTRY, "verdicts")
    rec, dec = _def_line(CONF_TOOL, "recompute"), _def_line(CONF_TOOL, "decide")
    cp, ana, mainf = _def_line(CONF_TOOL, "clopper_pearson"), _def_line(CONF_TOOL, "analyse"), _def_line(CONF_TOOL, "main")
    pipe = _def_line(ENTRY, "run_pipeline")
    ev, recm = _def_line(V2M, "evaluate"), _def_line(V2M, "recovery_metrics")
    metric_fn = {
        "eta_T_over_dw": f"{ev} scalar 'eta_T_over_dw' (= utils/dp_br_verifier.py:verify full_delta_max[2] / DW)",
        "stage2_rmse_pos_over_g2_0": f"{recm} 'stage2_rmse_pos_over_g2_0' (recovery grid, step 0.5)",
        "stage2_tail_mean_over_g2_0": f"{recm} 'stage2_tail_mean_over_g2_0' (recovery grid, step 0.5)",
        "Gmax_full_over_dw": f"{ev} scalar 'Gmax_full_over_dw'",
        "eta_T_over_dw_dev_minus_final_abs": f"|{ev} 'eta_T_over_dw' (development) - (final)|",
        "Gmax_full_over_dw_dev_minus_final_abs": f"|{ev} 'Gmax_full_over_dw' (development) - (final)|",
        "stage1_rel_err_abs": f"{recm} 'stage1_rel_err_signed', absolute value ({ev} 'stage1_rel_err_abs')",
    }
    decision = f"{verd} (entry point, writes gates.json); {rec} + {dec} (pre-registered recomputation from the NPZ arrays)"
    rows: List[Dict[str, Any]] = []

    def add(element: str, kind: str, src: str, definition: str, gate: str = "", metric: str = "", op: str = "",
            threshold: Optional[float] = None, tier: str = "", stage: str = "", candidate: str = "",
            part: str = "", mfn: str = "", dfn: str = "") -> None:
        rows.append({"element": element, "kind": kind, "gate": gate, "metric": metric, "op": op,
                     "threshold": threshold, "tier": tier, "stage": stage, "candidate": candidate,
                     "definition": definition, "part_of_run_pass": part, "source_json": src,
                     "metric_function": mfn, "decision_function": dfn})

    for gate in ("G-A", "G-F", "G-N"):
        blk = g[gate]
        for i, c in enumerate(blk["all_must_hold"]):
            add(f"{gate} criterion {i + 1}", "gate criterion", f"{P11}:/gates/{gate}/all_must_hold/{i}", c["definition"],
                gate, c["metric"], c["op"], float(c["threshold"]), blk["tier"], blk["stage"], blk["candidate"], "yes",
                metric_fn[c["metric"]], decision)
    s1 = sec["S1"]
    c = s1["criterion"]
    add("S1 criterion", "secondary criterion", f"{P11}:/secondary/S1/criterion", c["definition"], "S1", c["metric"],
        c["op"], float(c["threshold"]), "final-tier evaluation call (value is a direct policy query, tier-independent)",
        g["G-F"]["stage"], g["G-F"]["candidate"], "no (/secondary/S1/part_of_run_pass = false; pass_rule = null)",
        metric_fn[c["metric"]], decision)
    add("S1 reporting per run", "secondary criterion", f"{P11}:/secondary/S1/report_per_run", s1["report_per_run"], "S1")
    add("S1 reporting per q", "secondary criterion", f"{P11}:/secondary/S1/report_per_q", "; ".join(s1["report_per_q"]),
        "S1", dfn=mainf)
    add("comparison rule", "rule", f"{P11}:/gates/comparison", g["comparison"], dfn=f"{verd}; {dec}")
    add("tier rule", "rule", f"{P11}:/gates/dev_tier", g["dev_tier"])
    add("run pass", "rule", f"{P11}:/gates/run_outcome", g["run_outcome"], part="defines run pass",
        dfn=f"{verd} (run_pass = G-A and G-F and G-N); {pipe} (and no global-RNG violation); {ana} (run_pass = gates "
            "pass and status completed and exit code 0)")
    add("outcome labels", "rule", f"{ENTRY}:verdicts (source: code text)",
        "outcome = pass | stage2_failure (G-A fails) | fail_G-F | fail_G-N; global_rng_violation when a hardening "
        "assertion fails (gates.json outcome)", dfn=f"{verd}; {pipe}")
    hard = p["global_rng_hardening"]
    add("global-RNG assertion", "rule", f"{P11}:/global_rng_hardening/assertion; /on_violation",
        f"{hard['assertion']}; points: {', '.join(hard['assertion_points'])}; on violation: {hard['on_violation']}",
        part="a violation makes the run fail", dfn=pipe)
    add("v1.0 outcome (reported only)", "rule", f"{P11}:/v1_0_outcome_reported",
        f"{p['v1_0_outcome_reported']['definition']}; role: {p['v1_0_outcome_reported']['role']}", part="no",
        dfn=f"{verd}; {dec}")
    add("pass rule", "rule", f"{P11}:/confirmation/pass_rule", conf["pass_rule"],
        dfn=f"{mainf} (rule 18 of 20, applied only when 20 seeds are expected; overall = both q)")
    add("confirmation reporting", "rule", f"{P11}:/confirmation/reporting", conf["reporting"])
    blk = conf["seed_block"]
    add("confirmation seeds", "seeds", f"{P11}:/confirmation/seed_block; /n_seeds; /seeds_used_for_both_q; "
        "/seed_block_status",
        f"{blk[0]}-{blk[-1]} ({len(blk)} seeds; n_seeds = {conf['n_seeds']}; seeds_used_for_both_q = "
        f"{_vtxt(conf['seeds_used_for_both_q'])}; {conf['seed_block_status']}); q in {_vtxt(conf['q_values'])}")
    add("seed disjointness", "seeds", f"{P11}:/confirmation/disjointness_check", conf["disjointness_check"])
    dev = p["development_seeds"]
    add("development seeds (rehearsals)", "seeds", f"{P11}:/development_seeds",
        f"{dev[0]}-{dev[-1]} ({len(dev)} seeds; used by the v1.0 rehearsal and the v1.1 re-rehearsal)")
    bs = sec["bootstrap"]
    add("bootstrap", "statistics", f"{P11}:/secondary/bootstrap",
        f"resamples = {bs['resamples']}; seed = {bs['seed']}; rng = {bs['rng']}; method = {bs['method']}. "
        f"Implementation (source: code text): a fresh numpy.random.default_rng({bs['seed']}) for each q, q ascending; "
        "one statistic (mean signed stage-1 relative error); percentiles 2.5 and 97.5 of the resampled means",
        "S1", dfn=mainf)
    add("Clopper-Pearson", "statistics", f"{P11}:/secondary/binomial_ci",
        f"{sec['binomial_ci']}. Implementation (source: code text): alpha = 0.05; lo = beta.ppf(0.025, k, n - k + 1) "
        "(0 if k = 0); hi = beta.ppf(0.975, k + 1, n - k) (1 if k = n)", dfn=cp)
    add("crash rule", "rule", f"{P11}:/confirmation/crash_rule", conf["crash_rule"], dfn=f"{ana}; "
        f"{_def_line(CONF_TOOL, 'classify')} (classes completed / missing / failed_exception / incomplete / "
        "global_rng_violation; crashed attempts listed)")
    lock11 = [r for r in _lock_records() if r.get("protocol") == P11][-1]
    sha_tool = C.src(CONF_TOOL).sha256
    add("analysis script", "rule", f"{P11}:/confirmation/analysis_script; {LOCK} (v1.1 record)",
        f"{conf['analysis_script']}; SHA-256 at the base commit {sha_tool} equals the LOCK record "
        f"({lock11['analysis_script_sha256']}): {sha_tool == lock11['analysis_script_sha256']}")
    df = pd.DataFrame(rows)
    notes = ("Compiled from protocols/v2_T2_locked_v1_1.json (source: protocol text; thresholds read as numbers from the "
             "JSON) and the implementing functions located by name in the code at the base commit (line numbers computed "
             "by the builder). Text marked 'source: code text' is read from the code named in decision_function.")
    _table(pack, "T31", df, status="generated",
               sources=[C.src(P11), C.src(LOCK), C.src(ENTRY), C.src(CONF_TOOL), C.src(V2M), C.src(MD11)],
               script=f"{MOD}:build_t31", notes=notes, tier="n/a",
               docs={"element": "Criterion or rule element", "kind": "Kind of element (gate criterion, secondary "
                     "criterion, rule, seeds, statistics)", "gate": "Gate (G-A, G-F, G-N) or S1",
                     "metric": "Metric name as in the protocol JSON", "op": "Comparison operator (inclusive)",
                     "threshold": dict(definition="Threshold of the criterion as in the protocol JSON",
                                       units="Delta W (eta_2, Gmax_full, G-N) or fraction of e_2*(0) / e_1*(0)",
                                       normalization="as the metric"),
                     "tier": "Verifier tier the criterion is evaluated on",
                     "stage": "When the criterion is evaluated (end of Phase A / B)",
                     "candidate": "Policy the criterion is evaluated on",
                     "definition": "Definition text (protocol JSON, or code text where stated)",
                     "part_of_run_pass": "Whether the element decides run pass",
                     "source_json": "File and JSON pointer(s) of the element",
                     "metric_function": "file:function (line) that computes the metric",
                     "decision_function": "file:function (line) that applies the criterion or rule"},
               caption="Locked v1.1 criteria and rules; thresholds are inclusive and compared at full float64 precision.")
    # cross-check thresholds with the human-readable v1.1 protocol (gate table) and S1 text
    cmp = _Cmp(pack, "T31", MD11, "thresholds JSON vs protocol .md gate table and S1 text")
    tabs = _tables(MD11, ["Gate", "Criterion"], "Gates")
    cmp.tables = len(tabs)
    crit_rows = [r for t in tabs for r in t["rows"]]
    gn_seen = 0
    for r in df[df.kind == "gate criterion"].itertuples():
        thr = None
        if r.gate in ("G-A", "G-F"):
            for row in crit_rows:
                m = re.search(rf"{re.escape(r.metric)}\s*≤\s*([0-9]+(?:\.[0-9]+)?)", row[2])
                if m:
                    thr = m.group(1)
        else:
            gn = [row for row in crit_rows if "dev" in row[2] and "final" in row[2]]
            thr = re.search(r"≤\s*([0-9]+(?:\.[0-9]+)?)", gn[gn_seen][2]).group(1) if gn_seen < len(gn) else None
            gn_seen += 1
        if thr is None:
            cmp.unmatched += 1
            continue
        cmp.num(f"threshold {r.gate} {r.metric}", r.threshold, thr, "gate table, section 1")
    txt = C.abspath(MD11).read_text(encoding="utf-8")
    m = re.search(r"\*\*S1:\*\*[^\n]*≤\s*([0-9]+(?:\.[0-9]+)?)", txt)
    if m:
        cmp.num("threshold S1", float(df[df.gate.eq("S1") & df.kind.eq("secondary criterion")].threshold.dropna().iloc[0]),
                m.group(1), "section 2")
    else:
        cmp.unmatched += 1
    cmp.done()


# ----------------------------------------------------------------------------------------------
# F11 LR schedule
# ----------------------------------------------------------------------------------------------

def _schedule(L: ModuleType, Run: Any, proto: Dict[str, Any], q: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Per-update LR from ``Run.lr_for`` with the locked record schedule and decay windows (as Run.__init__ sets them).

    Returns the schedule (phase, local_update, update, lr) and the inputs (record schedule, windows, caps).
    """
    cfg = L.build_config(proto, q, S.SEEDS_CONF[0], "unused")
    ns = SimpleNamespace(sched=dict(cfg["record"]["lr_schedule"]),
                         lr_windows={w["phase"]: w for w in cfg["lr_decay"]})
    caps = cfg["record"]["protocol"]["phase_caps"]
    cap_a, cap_b = int(caps["A"]), int(caps["B"])
    rows = [("A", j, j, Run.lr_for(ns, "A", j)) for j in range(1, cap_a + 1)]
    rows += [("B", j, cap_a + j, Run.lr_for(ns, "B", j)) for j in range(1, cap_b + 1)]
    return (pd.DataFrame(rows, columns=["phase", "local_update", "update", "lr"]),
            {"sched": ns.sched, "windows": ns.lr_windows, "cap_a": cap_a, "cap_b": cap_b})


def build_f11(pack: C.Pack) -> None:
    """F11: actor and critic LR against the global update (Run.lr_for / lr_at), with the logged LR."""
    _repo_on_path()
    import run.run_v2_T2_locked as L  # noqa: E402
    from run.run_v2_stagewise import Run  # noqa: E402
    proto11 = L.load_protocol()                      # refuses on any hash mismatch (file, LOCK record)
    proto10 = _js(P10)
    sch, inp = _schedule(L, Run, proto11, 50)
    cap_a, cap_b, wa, wb = inp["cap_a"], inp["cap_b"], inp["windows"]["A"], inp["windows"]["B"]
    d_q = C.max_abs_diff(sch.lr, _schedule(L, Run, proto11, 60)[0].lr)
    d_v = max(C.max_abs_diff(sch.lr, _schedule(L, Run, proto10, q)[0].lr) for q in QS)
    want = {(p_, j): lr for p_, j, lr in zip(sch.phase, sch.local_update, sch.lr)}
    checks: List[str] = [f"computed schedule identical for q=50 and q=60 (max abs diff {d_q}) and for the v1.0 and v1.1 "
                         f"protocols (max abs diff {d_v})"]
    sources: List[Any] = [C.src(P11), C.src(P10), C.src(LOCK), C.src(ENTRY), C.src(STAGEWISE), C.src(LR_AT)]
    logged: Dict[str, Dict[int, List[Tuple[float, float]]]] = {}
    tot = {"runs": 0, "values": 0, "mismatch": 0, "max": 0.0}
    for study, n_exp in (("rehearsal", 20), ("rehearsal_v1_1", 20), ("confirmation", 40)):
        ss = C.srcs(f"{LK}/{study}/q*/seed*/train_history.json", label=f"{study} train_history.json", expect=n_exp)
        sources.append(ss)
        da = dc = 0.0
        n_mis = n_upd = n_bad_index = 0
        per_u: Dict[int, List[Tuple[float, float]]] = {}
        for s in ss.files:
            for x in _js(s.path)["history"]:
                w = want[(x["phase"], int(x["local"]))]
                n_bad_index += int(int(x["update"]) != (int(x["local"]) + (0 if x["phase"] == "A" else cap_a)))
                da, dc = max(da, abs(x["actor_lr"] - w)), max(dc, abs(x["critic_lr"] - w))
                n_mis += int(x["actor_lr"] != w) + int(x["critic_lr"] != w)
                n_upd += 1
                per_u.setdefault(int(x["update"]), []).append((float(x["actor_lr"]), float(x["critic_lr"])))
        logged[study] = per_u
        tot["runs"] += len(ss.files)
        tot["values"] += 2 * n_upd
        tot["mismatch"] += n_mis
        tot["max"] = max(tot["max"], da, dc)
        checks.append(f"{study}: logged actor_lr / critic_lr in train_history.json of {len(ss.files)} runs "
                      f"({n_upd} updates) vs computed: max abs diff actor {da}, critic {dc}; exact mismatches {n_mis} "
                      f"of {2 * n_upd}; update index inconsistent in {n_bad_index}")
    lrc = _rt(f"{RA}/lr_schedule_check.csv")
    sources.append(C.src(f"{RA}/lr_schedule_check.csv"))
    checks.append(f"rehearsal_analysis/lr_schedule_check.csv (v1.0 rehearsal): n_lr_mismatch = 0 in "
                  f"{int((lrc.n_lr_mismatch == 0).sum())}/{len(lrc)} runs, n_updates = {cap_a + cap_b} in "
                  f"{int((lrc.n_updates == cap_a + cap_b).sum())}/{len(lrc)}")
    gs = C.srcs(f"{LK}/confirmation/q*/seed*/gates.json", label="confirmation gates.json", expect=40)
    sources.append(gs)
    last_a, last_b = float(sch.lr[sch.phase.eq("A")].iloc[-1]), float(sch.lr[sch.phase.eq("B")].iloc[-1])
    n_last = sum(int(_js(s.path)["lr_last"]["A"] == last_a and _js(s.path)["lr_last"]["B"] == last_b) for s in gs.files)
    checks.append(f"end-of-phase LR lr(A,{cap_a}) = {last_a!r}, lr(B,{cap_b}) = {last_b!r} equal gates.json lr_last in "
                  f"{n_last}/{len(gs.files)} confirmation runs")
    # plotted logged markers: every 100th update, confirmation runs
    conf = logged["confirmation"]
    mk = []
    for u in range(100, cap_a + cap_b + 1, 100):
        a = np.array([x[0] for x in conf[u]])
        c_ = np.array([x[1] for x in conf[u]])
        mk.append({"update": u, "a_min": a.min(), "a_max": a.max(), "c_min": c_.min(), "c_max": c_.max(), "n": a.size})
    mk = pd.DataFrame(mk)
    n_conf = int(mk.n.max())
    checks.append(f"plotted logged markers (confirmation, every 100th update): min = max over the {n_conf} runs at every "
                  f"marker (actor: {bool((mk.a_min == mk.a_max).all())}, critic: {bool((mk.c_min == mk.c_max).all())})")
    # ---- figure
    fig, axes = style.new_figure(1, 1, height=3.1)
    ax = axes[0, 0]
    u = sch["update"].to_numpy()
    lr = sch.lr.to_numpy()
    b0 = cap_a + int(wb["local_first"]) - 0.5
    ax.axvspan(int(wa["local_first"]) - 0.5, cap_a + 0.5, color=style.GRID, alpha=0.9, lw=0,
               label=f"linear decay windows (A local {wa['local_first']}-{wa['local_last']}, "
                     f"B local {wb['local_first']}-{wb['local_last']})")
    ax.axvspan(b0, cap_a + int(wb["local_last"]) + 0.5, color=style.GRID, alpha=0.9, lw=0)
    ax.plot(u, lr * 1e4, color=style.REF, lw=1.8, label="actor LR, Run.lr_for")
    ax.plot(u, lr * 1e4, color=style.INK2, lw=1.1, ls=(0, (4, 3)), label="critic LR, Run.lr_for (same schedule)")
    ax.plot(mk["update"], mk.a_min * 1e4, ls="none", marker="o", mfc="none", mec=style.INK, ms=5,
            label=style.label_n("logged actor LR, confirmation runs", n_conf))
    ax.plot(mk["update"], mk.c_min * 1e4, ls="none", marker="+", color=style.INK2, ms=6,
            label=style.label_n("logged critic LR, confirmation runs", n_conf))
    ax.axvline(cap_a + 0.5, color=style.INK2, lw=0.8, ls=":")
    ytop = float(lr.max()) * 1e4
    ax.text(cap_a / 2, ytop * 1.14, f"Phase A (stage 2): u1-{cap_a}", ha="center", va="center")
    ax.text(cap_a + cap_b / 2, ytop * 1.14, f"Phase B (stage 1)\nu{cap_a + 1}-{cap_a + cap_b}", ha="center",
            va="center")
    ax.set_xlim(0, cap_a + cap_b + 10)
    ax.set_ylim(0, ytop * 1.235)
    ax.set_xlabel("global update u")
    ax.set_ylabel(r"learning rate ($\times 10^{-4}$)")
    style.legend(ax, loc="lower left")
    data = []
    for nm in ("actor_lr_computed", "critic_lr_computed"):
        for ph, j, uu, v in zip(sch.phase, sch.local_update, sch["update"], sch.lr):
            data.append({"series": nm, "update": int(uu), "phase": ph, "local_update": int(j), "lr": float(v),
                         "n_runs": np.nan, "lr_min_over_runs": np.nan, "lr_max_over_runs": np.nan})
    for nm, lo, hi in (("actor_lr_logged_confirmation", "a_min", "a_max"),
                       ("critic_lr_logged_confirmation", "c_min", "c_max")):
        for r in mk.itertuples():
            ph = "A" if r.update <= cap_a else "B"
            data.append({"series": nm, "update": int(r.update), "phase": ph,
                         "local_update": int(r.update if ph == "A" else r.update - cap_a), "lr": float(getattr(r, lo)),
                         "n_runs": int(r.n), "lr_min_over_runs": float(getattr(r, lo)),
                         "lr_max_over_runs": float(getattr(r, hi))})
    if tot["mismatch"] == 0:
        agree = (f"The logged actor and critic LR of all {tot['runs']} locked runs (v1.0 rehearsal, v1.1 re-rehearsal, "
                 f"confirmation; {tot['values']} logged values) equal the computed schedule at every update (max abs "
                 f"difference {tot['max']}).")
    else:
        agree = (f"The logged actor and critic LR of the {tot['runs']} locked runs differ from the computed schedule in "
                 f"{tot['mismatch']} of {tot['values']} logged values (max abs difference {tot['max']}).")
    caption = (f"Learning rate applied to the actor and to the critic before each update, against the global update u "
               f"(Phase A u1-{cap_a}, stage 2 only; Phase B u{cap_a + 1}-{cap_a + cap_b}, stage 1 only), computed with "
               "run/run_v2_stagewise.py:Run.lr_for, which calls run/run_final_dp_br_round3_dense.py:lr_at, from the locked "
               f"record schedule (ab_lr = {inp['sched']['ab_lr']}) and the two linear decay windows of "
               f"protocols/v2_T2_locked_v1_1.json (pipeline.lr_decay: Phase A local {wa['local_first']}-{wa['local_last']},"
               f" {wa['start_lr']} to {wa['end_lr']}; Phase B local {wb['local_first']}-{wb['local_last']}, "
               f"{wb['start_lr']} to {wb['end_lr']}). The same schedule applies to both optimizers and to both q (50 and "
               f"60). Markers: LR logged in train_history.json of the {n_conf} confirmation runs (q = 50 and 60, seeds "
               f"20501-20520), every 100th update. {agree} Axis in units of 1e-4. Tier: not applicable.")
    _figure(pack, "F11", fig, pd.DataFrame(data), status="generated", sources=sources, script=f"{MOD}:build_f11",
                caption=caption, tier="n/a", checks=checks,
                notes="Schedule computed by calling the repository's Run.lr_for on the locked record schedule and decay "
                      "windows exactly as Run.__init__ stores them (cfg from run/run_v2_T2_locked.py:build_config); no "
                      "re-implementation of lr_at.",
                docs={"series": "Plotted series (computed schedule, or LR logged by the confirmation runs)",
                      "update": "Global update index u (Phase B local j = u - 1600)",
                      "phase": "Training phase (A = stage 2 only, B = stage 1 only)",
                      "local_update": "Update index within the phase",
                      "lr": dict(definition="Learning rate before the update (computed, or the common logged value)",
                                 units="learning rate (Adam step size)"),
                      "n_runs": dict(definition="Runs whose logged value is summarised at the update", units="count"),
                      "lr_min_over_runs": dict(definition="Smallest logged value over the runs at the update",
                                               units="learning rate"),
                      "lr_max_over_runs": dict(definition="Largest logged value over the runs at the update",
                                               units="learning rate")})


# ----------------------------------------------------------------------------------------------
# T32 protocol history
# ----------------------------------------------------------------------------------------------

def build_t32(pack: C.Pack) -> None:
    """T32: v1.0 gates, v1.0 rehearsal pass counts and failing criteria, v1.0 -> v1.1 change log, pass probability."""
    p10, p11 = _js(P10), _js(P11)
    diff_rel = f"{LK}/v1_1/protocol_diff_v1_0_to_v1_1.json"
    prob_rel = f"{LK}/v1_1/v1_0_pass_probability.json"
    dif, prob = _js(diff_rel), _js(prob_rel)
    pc = _rt(f"{RA}/pass_counts.csv")
    gpr = _rt(f"{RA}/gates_per_run.csv")
    rows: List[Dict[str, Any]] = []

    def add(block: str, entry: str, source: str, **kw: Any) -> None:
        r = {"block": block, "entry": entry, "q": None, "seed": None, "metric": "", "op": "", "threshold": None,
             "tier": "", "value": None, "n": None, "reference_value": None, "detail": "", "decision": "",
             "evidence": "", "source": source}
        r.update(kw)
        rows.append(r)

    # (1) v1.0 gates
    b1 = "1 v1.0 gates"
    for gate in ("G-A", "G-F"):
        blk = p10["gates"][gate]
        for i, c in enumerate(blk["all_must_hold"]):
            add(b1, f"{gate} criterion {i + 1}", f"{P10}:/gates/{gate}/all_must_hold/{i}", metric=c["metric"], op=c["op"],
                threshold=float(c["threshold"]), tier=blk["tier"],
                detail=f"{blk['stage']}; candidate: {blk['candidate']}; {c['definition']}")
    for k in ("run_outcome", "dev_tier"):
        add(b1, k, f"{P10}:/gates/{k}", detail=p10["gates"][k])
    for k in ("pass_rule", "seed_block_status", "status"):
        add(b1, f"confirmation {k}", f"{P10}:/confirmation/{k}", detail=p10["confirmation"][k])
    # (2) v1.0 rehearsal pass counts and failing criteria
    b2 = "2 v1.0 rehearsal pass counts"
    for r in pc.itertuples(index=False):
        for col in ("G_A_pass", "G_F_pass", "run_pass", "G_A_pass_dev", "G_F_pass_dev"):
            add(b2, f"pass count {col}", f"{RA}/pass_counts.csv [q={r.q}, {col}]", q=int(r.q), metric=col,
                tier="development" if col.endswith("_dev") else "final", value=int(getattr(r, col)), n=int(r.n))
    b2f = "2 v1.0 rehearsal failing criteria"
    thr10 = {c["metric"]: float(c["threshold"]) for gate in ("G-A", "G-F") for c in p10["gates"][gate]["all_must_hold"]}
    fails = gpr[~gpr.run_pass.astype(bool)]
    for rd in fails.to_dict("records"):
        for m, thr in thr10.items():
            if not bool(rd[f"{m}_pass"]):
                add(b2f, f"q{rd['q']} s{rd['seed']}: {m}", f"{RA}/gates_per_run.csv [q={rd['q']}, seed={rd['seed']}]",
                    q=int(rd["q"]), seed=int(rd["seed"]), metric=m, op="<=", threshold=thr, tier="final",
                    value=float(rd[f"{m}_final"]),
                    detail=(f"outcome {rd['outcome']}; G-A {_ybool(rd['G-A'])}; dev-tier G-F {_ybool(rd['G-F_dev'])}; "
                            f"Gmax_full_over_dw_final = {rd['Gmax_full_over_dw_final']!r} at (t*, d*) = "
                            f"({int(rd['B_Gmax_full_t'])}, {rd['B_Gmax_full_d']:g}); dev value "
                            f"{rd[m + '_dev']!r}"))
    # (3) change log: machine diff (24 entries) and the 4 appended change_log entries
    b3 = "3 v1.0 -> v1.1 JSON diff"
    rep_rows = [dict(zip(t["header"], row)) for t in _tables(REP11, ["Op", "Path(s)", "Decision"]) for row in t["rows"]]

    def report_row(path: str) -> Optional[Dict[str, str]]:
        for rr in rep_rows:
            cell = rr["Path(s)"]
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(path)}(?![A-Za-z0-9_])", cell):
                return rr
        parent, last = path.rsplit("/", 1)
        for rr in rep_rows:
            cell = rr["Path(s)"]
            if parent + "/" in cell and re.search(rf"(?<![A-Za-z0-9_/]){re.escape(last)}(?![A-Za-z0-9_])", cell):
                return rr
        return None

    n_appended = len(p11["change_log"]) - len(p10["change_log"])
    for i, e in enumerate(dif):
        rr = report_row(e["path"])
        old = _vtxt(e["old"]) if "old" in e else ""
        new = _vtxt(e["new"]) if "new" in e else ""
        if e["path"] == "/change_log":
            det = (f"old: {len(e['old'])} entry; new: {len(e['new'])} entries (the {n_appended} appended v1.1 entries are "
                   "the rows of block '3 v1.1 change_log entries')")
        else:
            det = f"old: {old}" * bool(old) + ("; " if old and new else "") + f"new: {new}" * bool(new)
        add(b3, f"{e['op']} {e['path']}", f"{diff_rel} [{i}]", metric=e["op"], detail=det,
            decision=(rr["Decision"] + " (source: report text, protocol_v1_1_confirmation.md 1.2)") if rr else "UNKNOWN")
        if rr is None:
            pack.unknown_value("T32", f"decision of diff entry {e['path']}", "path not listed in the report table 1.2")
    b3c = "3 v1.1 change_log entries"
    for i, e in enumerate(p11["change_log"]):
        if e.get("version") != "1.1":
            continue
        add(b3c, f"change_log [{i}]", f"{P11}:/change_log/{i}", detail=f"change: {e['change']}; why: {e['why']}",
            decision=e["decided_by"], evidence="; ".join(e["evidence"]))
    # (4) v1.0 pass probability recomputation
    b4 = "4 v1.0 pass probability (recomputed)"
    pi = prob["pi_reference"]
    for k, ref, lab in (("P_both_rate_model", pi["p_rate"], "rate model (true rates = rehearsal pass rates)"),
                        ("P_both_normal_ddof1", pi["p_normal"], "normal model, fitted mean and SD (ddof = 1)"),
                        ("P_both_normal_PI_inputs", None, "normal model, the PI's rounded inputs"),
                        ("P_both_normal_ddof0", None, "normal model, SD ddof = 0")):
        add(b4, lab, f"{prob_rel}:/{k}", metric=k, value=float(prob[k]), reference_value=ref,
            detail="P(both q >= 18 of 20 pass) = prod over q of P(Binomial(20, p_q) >= 18); reference_value = PI-side "
                   "value" if ref is not None else "P(both q >= 18 of 20 pass)")
    refmap = {"rehearsal_pass_rate": "rate", "mean_signed": "mean", "sd_ddof1": "sd", "p90_abs_err_linear_interp": "p90_abs"}
    what = {"rehearsal_pass_rate": "rate-model input: v1.0 rehearsal run-pass rate of the q",
            "mean_signed": "normal-model input: mean signed stage-1 relative error of the rehearsal runs",
            "sd_ddof1": "normal-model input: SD (ddof = 1) of the signed stage-1 relative error",
            "sd_ddof0": "SD (ddof = 0) of the signed stage-1 relative error (variant)",
            "p90_abs_err_linear_interp": "90th percentile of |stage-1 rel. err| (linear interpolation; descriptive)",
            "per_run_pass_prob_normal_ddof1": "per-run P(|err| <= 0.10) under N(mean_signed, sd_ddof1)",
            "per_run_pass_prob_normal_ddof0": "per-run P(|err| <= 0.10) under N(mean_signed, sd_ddof0)",
            "per_run_pass_prob_normal_PI_inputs": "per-run P(|err| <= 0.10) under N(PI mean, PI SD)",
            "P_ge18_of_20_rate": "P(Binomial(20, rehearsal_pass_rate) >= 18)",
            "P_ge18_of_20_normal_ddof1": "P(Binomial(20, per_run_pass_prob_normal_ddof1) >= 18)"}
    for q in QS:
        for k, v in prob["per_q"][str(q)].items():
            if k == "n":
                continue
            ref = pi[refmap[k]][str(q)] if k in refmap else None
            add(b4, f"q={q} {k}", f"{prob_rel}:/per_q/{q}/{k}", q=q, metric=k, value=float(v), reference_value=ref,
                n=int(prob["per_q"][str(q)]["n"]),
                detail=what.get(k, "UNKNOWN") + " (source: code text, tools/v2/v1_0_pass_probability.py)")
            if k not in what:
                pack.unknown_value("T32", f"meaning of {k} in v1_0_pass_probability.json", "not described by the tool")
    df = pd.DataFrame(rows)
    for c in ("q", "seed", "n"):
        df[c] = _opt(df[c], int)
    # ---- recomputation checks (repository functions, base commit)
    gen = _tool(GEN11, "_sec_locked_a_gen11")
    regen = json.dumps(gen.apply_changes(p10), indent=1) + "\n"
    regen_ok = regen == C.abspath(P11).read_text(encoding="utf-8")
    diff_ok = gen.diff(p10, p11) == dif
    pp = _tool(PASSPROB_TOOL, "_sec_locked_a_passprob")
    from scipy.stats import norm  # noqa: E402
    rate, pn1 = {}, {}
    for q in QS:
        x = gpr[gpr.q == q].B_stage1_rel_err_signed.astype(float).to_numpy()
        rate[q] = float(gpr[gpr.q == q].run_pass.astype(bool).mean())
        m_, s_ = float(x.mean()), float(x.std(ddof=1))
        pn1[q] = float(norm.cdf(0.10, m_, s_) - norm.cdf(-0.10, m_, s_))
    d_rate = abs(pp.p_both(rate) - prob["P_both_rate_model"])
    d_norm = abs(pp.p_both(pn1) - prob["P_both_normal_ddof1"])
    notes = ("Blocks: (1) v1.0 gates from protocols/v2_T2_locked.json; (2) v1.0 rehearsal pass counts "
             "(rehearsal_analysis/pass_counts.csv, values as in the file) and every failed criterion of the failing runs "
             "(gates_per_run.csv, run_pass = False); (3) the 24 entries of results/v2_T2_locked/v1_1/protocol_diff_v1_0_"
             "to_v1_1.json (old/new as JSON literals; the decision column is the report's table 1.2, source: report text) "
             "and the change_log entries of version 1.1 of protocols/v2_T2_locked_v1_1.json; (4) "
             "results/v2_T2_locked/v1_1/v1_0_pass_probability.json with the PI-side values as reference_value. Checks: "
             f"make_locked_protocol_v1_1.apply_changes(v1.0) reproduces the v1.1 JSON byte for byte: {regen_ok}; "
             f"make_locked_protocol_v1_1.diff(v1.0, v1.1) equals the stored diff: {diff_ok}; pass probability recomputed "
             "from gates_per_run.csv with v1_0_pass_probability.p_both (rate model, normal ddof = 1): abs diff "
             f"{d_rate:.3g} and {d_norm:.3g}.")
    _table(pack, "T32", df, status="generated",
               sources=[C.src(P10), C.src(P11), C.src(diff_rel), C.src(prob_rel), C.src(f"{RA}/pass_counts.csv"),
                        C.src(f"{RA}/gates_per_run.csv"), C.src(REP11), C.src(GEN11), C.src(PASSPROB_TOOL)],
               script=f"{MOD}:build_t32", notes=notes, tier="final and development",
               docs={"block": "Block of the protocol history (1 gates, 2 rehearsal, 3 change log, 4 pass probability)",
                     "entry": "Row label within the block",
                     "metric": "Metric, count column, diff operation or probability name of the row",
                     "op": "Comparison operator of a criterion", "threshold": dict(
                         definition="Threshold of the v1.0 criterion", units="as the metric"),
                     "tier": "Tier of the value or criterion",
                     "value": dict(definition="Value of the row: count of passing runs (block 2 counts), the failing "
                                   "final-tier value (block 2 failures), or the probability / model input (block 4)",
                                   units="count, fraction of e_1*(0), or probability as stated by metric",
                                   tier="as the row's tier column (n/a for block 4)"),
                     "n": dict(definition="Number of runs behind the value", units="count"),
                     "reference_value": dict(definition="PI-side reference value of the quantity (block 4)",
                                             units="as value"),
                     "detail": "Definition, old/new values (JSON literals), or change and reason",
                     "decision": "Owner decision of the change (report table 1.2 for diff entries; decided_by of the "
                                 "change_log entry)",
                     "evidence": "Evidence cited by the change_log entry"},
               caption="Protocol history: v1.0 gates, v1.0 rehearsal outcome, the v1.0 to v1.1 change log, and the "
                       "recomputed v1.0 pass probability.")
    # ---- cross-checks
    pc_df = pc.copy()
    pack.crosscheck("T32", pc_df, REP10, header_has=["q", "n", "G_A_pass", "G_F_pass", "run_pass"], key_map={"q": "q"},
                    value_map={c: c for c in ("n", "G_A_pass", "G_F_pass", "run_pass", "G_A_pass_dev", "G_F_pass_dev")},
                    label="v1.0 rehearsal pass counts (4.3)")
    cmp = _Cmp(pack, "T32", SUMMARY, "v1.0 rehearsal G-A / G-F / run pass counts (summary table)")
    for t in _tables(SUMMARY, ["q", "G-A", "G-F", "run pass"], "Protocol lock"):
        cmp.tables += 1
        for row in t["rows"]:
            rec = dict(zip(t["header"], row))
            hit = pc[pc.q == int(rec["q"])]
            for rc, dc in (("G-A", "G_A_pass"), ("G-F", "G_F_pass"), ("run pass", "run_pass")):
                cmp.text(f"{dc} [q={rec['q']}]", f"{int(hit[dc].iloc[0])}/{int(hit.n.iloc[0])}", rec[rc],
                         f"line {t['line']}")
    cmp.done()
    txt10 = C.abspath(REP10).read_text(encoding="utf-8")
    cmp = _Cmp(pack, "T32", REP10, "4.3 prose: failing stage-1 errors and their Gmax")
    m = re.search(rf"\|stage-1 error\| > 0\.10\*\* \(({NUM}), ({NUM}), ({NUM})\)", txt10)
    fv = fails.sort_values(["q", "seed"]).stage1_rel_err_abs_final.tolist()
    if m and len(fv) == 3:
        for k in range(3):
            cmp.num(f"failing |stage-1 err| #{k + 1}", fv[k], m.group(k + 1), "4.3")
    else:
        cmp.unmatched += 1
    m = re.search(r"Ĝmax_full/ΔW is ([0-9]+(?:\.[0-9]+)?)–([0-9]+(?:\.[0-9]+)?), at \(t=1, d=0\)", txt10)
    if m:
        cmp.num("min Gmax_full_over_dw_final of the failing runs", fails.Gmax_full_over_dw_final.min(), m.group(1), "4.3")
        cmp.num("max Gmax_full_over_dw_final of the failing runs", fails.Gmax_full_over_dw_final.max(), m.group(2), "4.3")
        loc_ok = bool((fails.B_Gmax_full_t == 1).all() and (fails.B_Gmax_full_d == 0).all())
        cmp.flag("(t*, d*) of the failing runs", loc_ok, "(1, 0) in all" if loc_ok else "not all (1, 0)", "(t=1, d=0)")
    else:
        cmp.unmatched += 1
    cmp.done()
    txt11 = C.abspath(REP11).read_text(encoding="utf-8")
    cmp = _Cmp(pack, "T32", REP11, "1.2 diff (24 entries, ops, paths) and 1.3 pass-probability prose")
    cmp.flag("number of diff entries", len(dif) == 24, len(dif), "24 entries (1.2)")
    for e in dif:
        rr = report_row(e["path"])
        cmp.flag(f"diff entry {e['op']} {e['path']} listed with its op", rr is not None and e["op"] in rr["Op"],
                 e["op"], rr["Op"] if rr else "not listed", "1.2")
    for pat, val, q in ((r"rate model \*\*([0-9]+(?:\.[0-9]+)?)\*\*", prob["P_both_rate_model"], None),
                        (r"normal model \*\*([0-9]+(?:\.[0-9]+)?)\*\*", prob["P_both_normal_ddof1"], None),
                        (rf"rounded inputs the normal model gives ({NUM})", prob["P_both_normal_PI_inputs"], None),
                        (r"SD ddof=0 it gives ([0-9]+(?:\.[0-9]+)?)", prob["P_both_normal_ddof0"], None)):
        m = re.search(pat, txt11)
        if m:
            cmp.num(f"pass probability ({pat.split(' ')[0]} ...)", val, m.group(1), "1.3")
        else:
            cmp.unmatched += 1
    m = re.search(rf"fitted mean (−?{NUM}) / (−?{NUM}) and SD \(ddof=1\) ({NUM}) / ({NUM})", txt11)
    if m:
        for k, (key, q) in enumerate((("mean_signed", 50), ("mean_signed", 60), ("sd_ddof1", 50), ("sd_ddof1", 60))):
            cmp.num(f"{key} q={q}", prob["per_q"][str(q)][key], m.group(k + 1), "1.3")
    else:
        cmp.unmatched += 1
    m = re.search(r"90th percentile of \|error\| ([0-9]+(?:\.[0-9]+)?) / ([0-9]+(?:\.[0-9]+)?)", txt11)
    if m:
        for k, q in enumerate(QS):
            cmp.num(f"p90 |stage-1 err| q={q}", prob["per_q"][str(q)]["p90_abs_err_linear_interp"], m.group(k + 1), "1.3")
    else:
        cmp.unmatched += 1
    cmp.done()


# ----------------------------------------------------------------------------------------------
# T33 timeline
# ----------------------------------------------------------------------------------------------

def build_t33(pack: C.Pack) -> None:
    """T33: lock commits, LOCK records, rehearsal launches, checks commit, confirmation launch and end (UTC)."""
    rows: List[Dict[str, Any]] = []
    sources: List[Any] = []

    def commit(event: str, sha: str) -> None:
        full, t = _git_commit(sha)
        rows.append({"event": event, "commit": full, "time_utc": t, "time_kind": "commit time (committer date)",
                     "source": f"git log -1 --format=%cI {full} (read-only; commit objects are content-addressed)"})

    def stamp(event: str, t: str, kind: str, source: str, commit_: str = "") -> None:
        rows.append({"event": event, "commit": _git_commit(commit_)[0] if commit_ else "", "time_utc": t,
                     "time_kind": kind, "source": source})

    # clock check: the uptime clock in launch_env.txt equals launch_time_utc.txt (local clock = UTC)
    clk = []
    for study in ("confirmation", "rehearsal_v1_1"):
        lt = C.abspath(f"{LK}/{study}/launch_time_utc.txt").read_text().strip()
        env = C.abspath(f"{LK}/{study}/launch_env.txt").read_text()
        up = re.search(r"(\d\d:\d\d:\d\d) up", env).group(1)
        clk.append(f"{study}: uptime clock {up} vs launch_time_utc {lt[11:19]} (equal: {up == lt[11:19]})")
        sources += [C.src(f"{LK}/{study}/launch_time_utc.txt"), C.src(f"{LK}/{study}/launch_env.txt")]
    commit("v1.0 lock commit", "4bd2214")
    commit("LOCK record v1.0 (protocols/LOCK)", "7412c41")
    commit("v1.0 rehearsal launch commit (consolidation, dirty-flag re-run, C7 at 4bd2214)", "5b07293")
    load = C.abspath(f"{LK}/rehearsal_launch_load.txt").read_text()
    sources.append(C.src(f"{LK}/rehearsal_launch_load.txt"))
    st = {}
    for study, n in (("rehearsal", 20), ("rehearsal_v1_1", 20), ("confirmation", 40)):
        ss = C.srcs(f"{LK}/{study}/q*/seed*/status.json", label=f"{study} status.json", expect=n)
        sources.append(ss)
        recs = [(_js(s.path), s.path) for s in ss.files]
        st[study] = recs
    first = min(st["rehearsal"], key=lambda r: r[0]["start_time"])
    up = re.search(r"(\d\d:\d\d:\d\d) up", load).group(1)
    stamp("v1.0 rehearsal launch (uptime clock at launch)", f"{first[0]['start_time'][:10]}T{up}",
          "launch record (uptime clock; date from the runs' status.json)", f"{LK}/rehearsal_launch_load.txt", "5b07293")
    c2 = f"{LK}/locked_check2_phaseB/launch_20261002_025428.json"
    sources.append(C.src(c2))
    started = _js(c2)["started"]
    stamp("v1.0 Check 2 launch (pilot launcher, Phase B from the stitched u1600 state)",
          datetime.strptime(started, "%Y%m%d_%H%M%S").strftime("%Y-%m-%dT%H:%M:%S"), "launch record", f"{c2}:/started",
          "5b07293")
    for study, lab, sha in (("rehearsal", "v1.0 rehearsal", "5b07293"), ("rehearsal_v1_1", "v1.1 re-rehearsal", "95c000e"),
                            ("confirmation", "confirmation", "f6838ec")):
        recs = st[study]
        t0, t1 = min(r[0]["start_time"] for r in recs), max(r[0]["end_time"] for r in recs)
        first_runs = [r[1] for r in recs if r[0]["start_time"] == t0]
        last_runs = [r[1] for r in recs if r[0]["end_time"] == t1]
        n_ok = sum(int(r[0]["state"] == "done" and r[0].get("exit_code") == 0) for r in recs)
        if study == "rehearsal_v1_1":
            commit("v1.1 lock commit", "431474d")
            commit("LOCK record v1.1 (protocols/LOCK) and C7 at the lock", "95c000e")
            lt = C.abspath(f"{LK}/rehearsal_v1_1/launch_time_utc.txt").read_text().strip()
            stamp("v1.1 re-rehearsal launch", lt.rstrip("Z"), "launch record (UTC)",
                  f"{LK}/rehearsal_v1_1/launch_time_utc.txt",
                  "95c000e")
        if study == "confirmation":
            commit("re-rehearsal records and R1-R6 checks commit (confirmation launch commit)", "f6838ec")
            lt = C.abspath(f"{LK}/confirmation/launch_time_utc.txt").read_text().strip()
            stamp("confirmation launch", lt.rstrip("Z"), "launch record (UTC)", f"{LK}/confirmation/launch_time_utc.txt",
                  "f6838ec")
        stamp(f"{lab}: first run start", _status_utc(t0), "status.json start_time (local clock = UTC)",
              f"earliest start_time of {len(recs)} status.json files: " + ", ".join(first_runs), sha)
        stamp(f"{lab}: last run end ({n_ok}/{len(recs)} done with exit code 0)", _status_utc(t1),
              "status.json end_time (local clock = UTC)",
              f"latest end_time of {len(recs)} status.json files: " + ", ".join(last_runs), sha)
        lo = f"{LK}/{study}/launcher.out"
        sources.append(C.src(lo))
        stamp(f"{lab}: launcher log last written", _mtime_utc(lo), "file modification time (filesystem metadata)", lo, sha)
        if study == "rehearsal":
            commit("v1.0 lock-round analysis tools commit", "344154a")
            commit("v1.0 rehearsal, calibration and cusp records commit", "36060ca")
            commit("v1.0 lock and rehearsal report commit", "3098259")
            commit("D1 addendum (Check 1 accepted) appended to the v1.0 report", "7c13aee")
    vd = f"{CA}/verdict.json"
    sources.append(C.src(vd))
    stamp("confirmation analysis written (verdict.json)", _mtime_utc(vd), "file modification time (filesystem metadata)",
          vd, "f6838ec")
    commit("confirmation records and analysis commit", "0588705")
    commit("confirmation report commit", "cb0b541")
    df = pd.DataFrame(rows).sort_values(["time_utc", "event"], kind="stable").reset_index(drop=True)
    notes = ("Commit times: committer dates from read-only `git log -1 --format=%cI` (all +00:00). Launch times: "
             "launch_time_utc.txt (UTC), the uptime clock of rehearsal_launch_load.txt, and the Check-2 launch record. Run "
             "start/end: status.json start_time/end_time, written with the local clock; the local clock is UTC ("
             + "; ".join(clk) + "). Launcher-log and verdict times are file modification times (filesystem metadata, not "
             "file content). Sorted by time.")
    _table(pack, "T33", df, status="generated", sources=sources, script=f"{MOD}:build_t33", notes=notes, tier="n/a",
               docs={"event": "Event", "commit": "Full commit hash of the event (commit rows) or the code commit the "
                     "launch ran at", "time_utc": dict(definition="Time of the event in UTC (ISO 8601, seconds; "
                                                      "milliseconds for file times)", units="UTC time"),
                     "time_kind": "Where the time comes from (commit date, launch record, status.json, file time)",
                     "source": "File (or git command) the time was read from"},
               caption="Timeline of the lock rounds, rehearsals and the confirmation (UTC).")
    # cross-check with the v1.1 report's timeline table (1.5)
    cmp = _Cmp(pack, "T33", REP11, "1.5 timeline (UTC)")
    keymap = [("D1 addendum", lambda e: e.startswith("D1 addendum")),
              ("v1.1 lock commit", lambda e: e == "v1.1 lock commit"),
              ("LOCK v1.1 record", lambda e: e.startswith("LOCK record v1.1")),
              ("Re-rehearsal launch", lambda e: e == "v1.1 re-rehearsal launch"),
              ("Re-rehearsal records", lambda e: e.startswith("re-rehearsal records")),
              ("Confirmation launch", lambda e: e == "confirmation launch"),
              ("Confirmation finished", lambda e: e.startswith("confirmation: last run end"))]
    for t in _tables(REP11, ["Event", "Commit", "Time"], "Timeline"):
        cmp.tables += 1
        for row in t["rows"]:
            rec = dict(zip(t["header"], row))
            ev = rec["Event"].replace("*", "").strip()
            sel = [f for k, f in keymap if ev.startswith(k)]
            hit = df[df.event.map(sel[0])] if sel else df.iloc[0:0]
            if len(hit) != 1:
                cmp.unmatched += 1
                continue
            rep_t = rec["Time"].split(" ")[0]
            mine = hit.time_utc.iloc[0][:19]
            note = ""
            if ev.startswith("Confirmation finished") and mine != rep_t:
                lo = df[df.event.str.startswith("confirmation: launcher log")].time_utc.iloc[0]
                hits, n_files = _grep_text([f"{LK}/confirmation/*", f"{LK}/confirmation/q*/seed*/*", f"{CA}/*"],
                                           (rep_t[11:19], rep_t.replace("T", " ")))
                note = (f"pack time = latest status.json end_time ({hit.source.iloc[0]}); confirmation/launcher.out last "
                        f"modified {lo}; the report time {rep_t[11:19]} occurs in {len(hits)} of {n_files} text files "
                        "of results/v2_T2_locked/confirmation and confirmation_analysis"
                        + (": " + ", ".join(hits) if hits else ""))
            cmp.flag(f"time of '{ev}'", mine == rep_t, mine, rep_t, f"line {t['line']}", note)
            if rec["Commit"] not in ("—", "-", ""):
                cmp.flag(f"commit of '{ev}'", hit.commit.iloc[0].startswith(rec["Commit"]), hit.commit.iloc[0][:7],
                         rec["Commit"], f"line {t['line']}")
    cmp.done()


# ----------------------------------------------------------------------------------------------
# T34 re-rehearsal checks
# ----------------------------------------------------------------------------------------------

def _check_definitions() -> Dict[str, str]:
    """R1-R6 definitions from the docstring of tools/v2/v1_1_rehearsal_checks.py (source: code text)."""
    doc = ast.get_docstring(ast.parse(C.abspath(CHECK_TOOL).read_text(encoding="utf-8"))) or ""
    out: Dict[str, str] = {}
    cur = None
    for ln in doc.splitlines():
        m = re.match(r"^(R[1-6])\s{2,}(.*)$", ln)
        if m:
            cur = m.group(1)
            out[cur] = m.group(2).strip()
        elif cur and ln.startswith("    ") and ln.strip():
            out[cur] += " " + ln.strip()
        elif cur and not ln.strip():
            cur = None
    return out


def _r1_field_doc(field: str) -> str:
    """What an R1 field compares (from tools/v2/v1_1_rehearsal_checks.py:r1_one; source: code text)."""
    if ":" in field and field.startswith("state_end_"):
        f, k = field.split(":")
        what = {"rng_streams": "state['rng'] (numpy training streams)",
                "torch_generator": "state['torch_generator_state']"}.get(k, f"state['agent']['{k}']")
        return f"{f}: {what}, exact equality (tools/v2/v1_1_rehearsal_checks.py:teq)"
    if field == "weight_exports":
        return "weights/u*.npz: same file names, exactly 88 files, every array equal"
    if field.endswith(".npz"):
        return f"{field}: same array names, every array equal"
    if field.startswith("train_history:"):
        return (f"train_history.json['{field.split(':')[1]}'] without the wall-clock keys {', '.join(WALL_KEYS)}; equal "
                "JSON text (sorted keys)")
    if field.endswith(".csv"):
        return (f"{field} parsed by pandas (default float parser), wall-clock columns dropped, same columns and "
                "DataFrame.equals")
    if field.endswith(".json"):
        return f"{field}: equal JSON objects"
    return {"gate_metric_values": "v1.0 gates.json criteria (value_final, value_dev) vs v1.1 metric_values / "
                                  "dev_tier_values for eta_2, RMSE, tail, Gmax_full and |stage-1 error|, float equality",
            "reported_metrics": "gates.json['reported'] (end of A, end of B, drift test) as JSON text with sorted keys",
            "ALL": "all fields of the run identical"}.get(field, field)


def build_t34(pack: C.Pack) -> None:
    """T34: re-rehearsal checks R1-R6 with per-q counts, and the R1 fields with n identical."""
    ck_rel, det_rel = f"{LK}/rehearsal_v1_1_checks.json", f"{LK}/rehearsal_v1_1_R1_details.csv"
    ck, det = _js(ck_rel), _rt(det_rel)
    defs = _check_definitions()
    root11, root10 = f"{LK}/rehearsal_v1_1", f"{LK}/rehearsal"
    launcher = C.abspath(f"{root11}/launcher.out").read_text()
    ag = _rt(f"{RA11}/agreement.csv")
    sources: List[Any] = [C.src(ck_rel), C.src(det_rel), C.src(CHECK_TOOL), C.src(f"{root11}/launcher.out"),
                          C.src(f"{RA11}/agreement.csv"), C.src(f"{LK}/rehearsal_v1_1_pytest.txt")]
    per: Dict[str, Dict[int, int]] = {k: {50: 0, 60: 0} for k in ("R1", "R2", "R3", "R4", "R5")}
    for kind in ("gates.json", "status.json", "manifest.json"):
        sources.append(C.srcs(f"{root11}/q*/seed*/{kind}", label=f"rehearsal_v1_1 {kind}", expect=20))
    sha11 = ck["v1_1_protocol_sha256"]
    for q in QS:
        for s in S.SEEDS_DEV:
            d = f"{root11}/q{q}/seed{s}"
            g, st, man = _js(f"{d}/gates.json"), _js(f"{d}/status.json"), _js(f"{d}/manifest.json")
            per["R2"][q] += int(g["G-A"]["pass"] and g["G-F"]["pass"] and g["G-N"]["pass"])
            rc = re.search(rf"q{q} s{s} rc=(\d+)", launcher)
            per["R3"][q] += int(g["global_rng"]["status"] == "ok" and not g["global_rng"]["violations"]
                                and st.get("exit_code") == 0 and rc is not None and rc.group(1) == "0")
            per["R4"][q] += int(man["locked_protocol"]["sha256"] == sha11 and str(man["locked_protocol"]["version"]) == "1.1"
                                and man["git"]["commit"].startswith(ck["launch_commit"]) and man["clean_tree"] is True)
        per["R1"][q] = int(det[det.q == q].ALL.astype(bool).sum())
        per["R5"][q] = int(ag[ag.q == q].all_agree.astype(bool).sum())
    # exact-float re-check of the three CSV fields of R1 (R1 parsed them with pandas' default float parser)
    recheck: Dict[str, Dict[int, int]] = {f: {50: 0, 60: 0} for f in ("v2_updates.csv", "v2_checkpoints_A.csv",
                                                                      "v2_checkpoints_B.csv")}
    for f in recheck:
        sources.append(C.srcs([f"{root10}/q*/seed*/{f}", f"{root11}/q*/seed*/{f}"], label=f"rehearsal + re-rehearsal {f}",
                              expect=40))
        for q in QS:
            for s in S.SEEDS_DEV:
                a, b = _rt(f"{root10}/q{q}/seed{s}/{f}"), _rt(f"{root11}/q{q}/seed{s}/{f}")
                a = a.drop(columns=[c for c in a.columns if c in WALL_KEYS])
                b = b.drop(columns=[c for c in b.columns if c in WALL_KEYS])
                recheck[f][q] += int(list(a.columns) == list(b.columns) and a.equals(b))
    rows: List[Dict[str, Any]] = []
    nkey = {"R1": "n_identical", "R2": "n_pass", "R3": "n_ok", "R4": "n_ok", "R5": "n_agree"}
    for k in ("R1", "R2", "R3", "R4", "R5", "R6"):
        r = ck[k]
        detail = "; ".join(f"{kk} = {_vtxt(vv)}" for kk, vv in r.items() if kk not in ("pass", "n", nkey.get(k, "")))
        n_ok = r.get(nkey[k]) if k in nkey else None
        rows.append({"block": "check", "check": k, "field": "", "verdict": "pass" if r["pass"] else "fail",
                     "n": r.get("n"), "n_ok": n_ok, "n_ok_q50": per[k][50] if k in per else None,
                     "n_ok_q60": per[k][60] if k in per else None, "recheck_exact_float_n_identical": None,
                     "detail": detail, "definition": defs.get(k, "UNKNOWN"), "source": f"{ck_rel}:/{k}"})
        if k in per and n_ok is not None and per[k][50] + per[k][60] != n_ok:
            pack.mismatch("T34", f"{k} per-q recount", per[k][50] + per[k][60], ck_rel, n_ok,
                          "recount from the run files differs from the checks JSON")
    rows.append({"block": "check", "check": "ALL_PASS", "field": "", "verdict": "pass" if ck["ALL_PASS"] else "fail",
                 "n": None, "n_ok": None, "n_ok_q50": None, "n_ok_q60": None, "recheck_exact_float_n_identical": None,
                 "detail": f"launch_commit = {ck['launch_commit']}; v1_1_protocol_sha256 = {sha11}",
                 "definition": "R1 to R6 all pass", "source": f"{ck_rel}:/ALL_PASS"})
    for f in [c for c in det.columns if c not in ("q", "seed")]:
        col = det[f].astype(bool)
        rows.append({"block": "R1 field", "check": "R1", "field": f, "verdict": "pass" if col.all() else "fail",
                     "n": int(col.size), "n_ok": int(col.sum()), "n_ok_q50": int(col[det.q == 50].sum()),
                     "n_ok_q60": int(col[det.q == 60].sum()),
                     "recheck_exact_float_n_identical": (recheck[f][50] + recheck[f][60]) if f in recheck else None,
                     "detail": f"identical in {int(col.sum())} of {col.size} runs (q=50: {int(col[det.q == 50].sum())}/"
                               f"{int((det.q == 50).sum())}, q=60: {int(col[det.q == 60].sum())}/"
                               f"{int((det.q == 60).sum())})",
                     "definition": _r1_field_doc(f), "source": f"{det_rel} [column {f}]"})
    df = pd.DataFrame(rows)
    for c in ("n", "n_ok", "n_ok_q50", "n_ok_q60", "recheck_exact_float_n_identical"):
        df[c] = _opt(df[c], int)
    notes = ("Checks R1-R6 from results/v2_T2_locked/rehearsal_v1_1_checks.json; per-q counts recounted by the builder from "
             "the run files exactly as tools/v2/v1_1_rehearsal_checks.py does (R1: R1_details.csv ALL; R2: gates.json G-A, "
             "G-F and G-N pass; R3: global_rng ok, no violations, status exit code 0, launcher rc 0; R4: manifest protocol "
             "SHA-256, version 1.1, launch commit, clean_tree; R5: rehearsal_v1_1_analysis/agreement.csv all_agree); "
             "definitions from the checks tool's docstring (source: code text). R1 fields: the 39 columns of "
             "rehearsal_v1_1_R1_details.csv with n identical per q. recheck_exact_float_n_identical: the builder "
             "re-compared "
             "the three CSV fields (v1.0 rehearsal vs v1.1 re-rehearsal, wall-clock columns dropped) with exact float "
             "parsing (round_trip), because R1 used pandas' default parser, which drops the 17th significant digit.")
    _table(pack, "T34", df, status="generated", sources=sources, script=f"{MOD}:build_t34", notes=notes, tier="n/a",
               docs={"block": "check = one of R1-R6 (and ALL_PASS); R1 field = one compared field of R1",
                     "check": "Check id", "field": "R1 field (column of rehearsal_v1_1_R1_details.csv)",
                     "verdict": "pass / fail of the check, or of the field over all runs",
                     "n": dict(definition="Runs checked", units="count"),
                     "n_ok": dict(definition="Runs satisfying the check or with the field identical", units="count"),
                     "n_ok_q50": dict(definition="n_ok among the q = 50 runs (seeds 10501-10510)", units="count"),
                     "n_ok_q60": dict(definition="n_ok among the q = 60 runs (seeds 10501-10510)", units="count"),
                     "recheck_exact_float_n_identical": dict(
                         definition="Builder's re-check of the CSV field with exact float parsing: runs identical",
                         units="count"),
                     "detail": "Other fields of the check record (JSON literals), or the per-q identity counts",
                     "definition": "What the check or field compares (source: code text of tools/v2/v1_1_rehearsal_"
                                   "checks.py)"},
               caption="Re-rehearsal checks R1-R6 under protocol v1.1 (development seeds 10501-10510, both q).")
    # cross-check with the report's checks table (section 3)
    cmp = _Cmp(pack, "T34", REP11, "3 checks table (verdicts and numbers)")
    for t in _tables(REP11, ["Check", "Verdict", "Detail"], "Re-rehearsal"):
        cmp.tables += 1
        for row in t["rows"]:
            rec = dict(zip(t["header"], row))
            k = rec["Check"].split(" ")[0]
            hit = df[(df.block == "check") & (df.check == k)]
            if len(hit) != 1:
                cmp.unmatched += 1
                continue
            cmp.text(f"{k} verdict", hit.verdict.iloc[0], rec["Verdict"].replace("*", "").strip(), f"line {t['line']}")
            m = re.search(r"(\d+)/(\d+)", rec["Detail"])
            if m and k in per:
                cmp.flag(f"{k} n_ok/n", f"{hit.n_ok.iloc[0]}/{hit.n.iloc[0]}" == m.group(0),
                         f"{hit.n_ok.iloc[0]}/{hit.n.iloc[0]}", m.group(0), f"line {t['line']}")
            if k == "R6":
                m = re.search(r"(\d+) passed, (\d+) xfailed", rec["Detail"])
                mm = re.search(r"(\d+) passed, (\d+) xfailed", ck["R6"]["pytest_summary"])
                cmp.flag("R6 passed/xfailed", bool(m and mm and m.groups() == mm.groups()),
                         mm.group(0) if mm else "", m.group(0) if m else "", f"line {t['line']}")
            if k == "R5":
                cmp.num("R5 max abs value difference", ck["R5"]["max_abs_value_diff"], "0", f"line {t['line']}")
    cmp.done()


# ----------------------------------------------------------------------------------------------
# T35 verdict, T36 per-run table, T37 distributions, T38 S1, T40 rehearsal vs confirmation
# ----------------------------------------------------------------------------------------------

def build_t35(pack: C.Pack) -> None:
    """T35: per q passes out of 20, exact 95% CI and the rule; overall verdict."""
    pc, vd, pr = _rt(f"{CA}/pass_counts.csv"), _js(f"{CA}/verdict.json"), _rt(f"{CA}/per_run.csv")
    cat = _tool(CONF_TOOL, "_sec_locked_a_conf")
    rows = []
    max_d = 0.0
    for r in pc.to_dict("records"):
        g = pr[pr.q == r["q"]]
        k, n = int(g.run_pass.astype(bool).sum()), len(g)
        lo, hi = cat.clopper_pearson(k, n)
        max_d = max(max_d, abs(lo - r["cp95_lo"]), abs(hi - r["cp95_hi"]), abs(k - r["n_pass"]), abs(n - r["n_expected"]))
        rows.append({"scope": f"q={int(r['q'])}", **r, "verdict": "pass" if bool(r["q_passes_rule"]) else "fail",
                     "per_q": "", "seeds": f"{vd['seeds'][0]}-{vd['seeds'][1]}", "n_agreement_rows": None,
                     "n_all_agree": None, "root": vd["root"]})
    rows.append({"scope": "overall (both q)", "q": None, "rule": vd["rule"], "verdict": vd["overall"],
                 "per_q": json.dumps(vd["per_q"]), "seeds": f"{vd['seeds'][0]}-{vd['seeds'][1]}",
                 "n_agreement_rows": vd["n_agreement_rows"], "n_all_agree": vd["n_all_agree"], "root": vd["root"]})
    df = pd.DataFrame(rows)
    for c in ["q", "n_agreement_rows", "n_all_agree"] + [c for c in pc.columns if c.startswith("n_")]:
        df[c] = _opt(df[c], int)
    df["q_passes_rule"] = _opt(df["q_passes_rule"], bool)
    overall_ok = (vd["overall"] == "PASS") == bool(pc.q_passes_rule.astype(bool).all())
    notes = ("Per-q rows: results/v2_T2_locked/confirmation_analysis/pass_counts.csv, values as in the file; overall row: "
             "verdict.json (rule, per_q, overall, agreement counts). Check: n_pass, n_expected and the Clopper-Pearson "
             "bounds recounted from per_run.csv with tools/v2/confirmation_analysis.py:clopper_pearson (scipy beta "
             f"quantiles) equal the file (max abs diff {max_d}); overall = both q pass is consistent with per-q verdicts: "
             f"{overall_ok}.")
    _table(pack, "T35", df, status="regenerated",
               sources=[C.src(f"{CA}/pass_counts.csv"), C.src(f"{CA}/verdict.json"), C.src(f"{CA}/per_run.csv"),
                        C.src(CONF_TOOL)],
               script=f"{MOD}:build_t35", notes=notes, tier="final (G-N compares development and final)",
               docs={**DOC_PASS_COUNTS, "cp95_lo": dict(tier="final (G-N: development vs final)"),
                     "cp95_hi": dict(tier="final (G-N: development vs final)"),
                     "scope": "Row scope (one q, or the overall verdict over both q)",
                     "verdict": "Verdict of the scope: per q pass/fail of the >= 18 of 20 rule; overall PASS/FAIL (both q)",
                     "per_q": "verdict.json per_q (JSON text) on the overall row",
                     "seeds": "Seed block analysed (inclusive)",
                     "n_agreement_rows": dict(definition="Runs compared with gates.json by the analysis", units="count",
                                              source=CONF_TOOL + ":main"),
                     "n_all_agree": dict(definition="Runs whose recomputed values and verdicts all agree with gates.json",
                                         units="count", source=CONF_TOOL + ":main"),
                     "root": dict(definition="Results root of the confirmation runs", units="path")},
               caption="Confirmation verdict under protocol v1.1: per q passes out of 20 with the exact 95% Clopper-Pearson "
                       "CI and the rule, then the overall verdict.")
    pack.crosscheck("T35", pc, REP11, header_has=["q", "n_expected", "n_pass", "cp95_lo"], key_map={"q": "q"},
                    value_map={c: c for c in _numeric_cols(pc, ("q",))}, heading_has="4.1 Verdict",
                    label="4.1 pass counts")
    _crosscheck_text(pack, "T35", pc, REP11, ["q", "n_expected", "n_pass", "cp95_lo"], {"q": "q"},
                     {"rule": "rule", "q_passes_rule": "q_passes_rule"}, "4.1 Verdict")
    cmp = _Cmp(pack, "T35", REP11, "Verdict table (top) and 4.1 verdict row")
    for t in _tables(REP11, ["q", "primary passes", "rule"], "Verdict"):
        cmp.tables += 1
        for row in t["rows"]:
            rec = dict(zip(t["header"], row))
            hit = pc[pc.q == int(rec["q"])].iloc[0]
            m = re.match(r"\*?\*?(\d+) / (\d+)", rec["primary passes"])
            cmp.flag(f"primary passes [q={rec['q']}]", bool(m) and (int(m.group(1)), int(m.group(2))) ==
                     (int(hit.n_pass), int(hit.n_expected)), f"{int(hit.n_pass)} / {int(hit.n_expected)}",
                     rec["primary passes"], f"line {t['line']}")
            ci = re.findall(r"[0-9]+(?:\.[0-9]+)?", rec["exact 95% CI (Clopper–Pearson)"])
            cmp.num(f"cp95_lo [q={rec['q']}]", hit.cp95_lo, ci[0], f"line {t['line']}")
            cmp.num(f"cp95_hi [q={rec['q']}]", hit.cp95_hi, ci[1], f"line {t['line']}")
            cmp.text(f"q passes [q={rec['q']}]", bool(hit.q_passes_rule), rec["q passes"], f"line {t['line']}")
            cmp.flag(f"rule [q={rec['q']}]", rec["rule"].replace("≥", ">=") == hit.rule, hit.rule, rec["rule"],
                     f"line {t['line']}")
    for t in _tables(REP11, ["root", "seeds", "overall", "n_all_agree"], "4.1 Verdict"):
        cmp.tables += 1
        for row in t["rows"]:
            rec = dict(zip(t["header"], row))
            cmp.text("overall", vd["overall"], rec["overall"], f"line {t['line']}")
            cmp.num("n_agreement_rows", vd["n_agreement_rows"], rec["n_agreement_rows"], f"line {t['line']}")
            cmp.num("n_all_agree", vd["n_all_agree"], rec["n_all_agree"], f"line {t['line']}")
    cmp.done()
    cmp = _Cmp(pack, "T35", SUMMARY, "summary confirmation table: primary passes, CI, v1.0 outcome")
    for t in _tables(SUMMARY, ["q", "primary passes", "exact 95% CI", "v1.0 outcome"], "confirmation"):
        cmp.tables += 1
        for row in t["rows"]:
            rec = dict(zip(t["header"], row))
            hit = pc[pc.q == int(rec["q"])].iloc[0]
            cmp.text(f"primary passes [q={rec['q']}]", f"{int(hit.n_pass)}/{int(hit.n_expected)}", rec["primary passes"],
                     f"line {t['line']}")
            ci = re.findall(r"[0-9]+(?:\.[0-9]+)?", rec["exact 95% CI"])
            cmp.num(f"cp95_lo [q={rec['q']}]", hit.cp95_lo, ci[0], f"line {t['line']}")
            cmp.num(f"cp95_hi [q={rec['q']}]", hit.cp95_hi, ci[1], f"line {t['line']}")
            cmp.text(f"v1.0 outcome [q={rec['q']}]", f"{int(hit.n_v1_0_run_pass)}/{int(hit.n_expected)}",
                     rec["v1.0 outcome"], f"line {t['line']}")
    cmp.done()


def build_t36(pack: C.Pack) -> None:
    """T36: per-run confirmation table (40 rows), the analysis file as is."""
    rel = f"{CA}/per_run.csv"
    _found(pack, "T36", rel, script=f"{MOD}:build_t36", docs=DOC_PER_RUN, tier="final and development",
                     notes="Columns: G-A metrics eta_final (final tier; eta_dev = development tier), rmse and tail "
                           "(tier-independent); G-F gmax_final (final tier; gmax_dev = development tier); G-N from the "
                           "*_dev_minus_final columns; S1 s1 (|stage-1 rel. err|); v1_0_* = the v1.0 outcome (reported "
                           "only). Seeds 20501-20520 for both q.",
                     caption="Per-run confirmation table: 40 runs (q = 50 and 60, seeds 20501-20520), recomputed gate "
                             "values and verdicts of the pre-registered analysis.")
    df = _rt(rel)
    pack.crosscheck("T36", df, REP11, header_has=["q", "seed", "eta_final", "gmax_final"],
                    key_map={"q": "q", "seed": "seed"},
                    value_map={c: c for c in _numeric_cols(df, ("q", "seed"))}, heading_has="4.2",
                    label="4.2 per-run table (numbers)")
    _crosscheck_text(pack, "T36", df, REP11, ["q", "seed", "eta_final", "gmax_final"], {"q": "q", "seed": "seed"},
                     {c: c for c in ("status", "G-A_pass", "G-F_pass", "G-N_pass", "S1_pass", "eta_N_pass", "gmax_N_pass",
                                     "global_rng", "run_pass", "v1_0_G-F", "v1_0_run_pass", "outcome")}, "4.2",
                     "4.2 per-run table (verdicts and labels)")


def _quantiles(x: pd.Series) -> Dict[str, float]:
    """min, p10, p25, median, p75, p90, max with pandas linear interpolation (as the analysis script)."""
    x = x.dropna().astype(float)
    return {"n": int(x.size), "min": float(x.quantile(0.0)), "p10": float(x.quantile(0.10)),
            "p25": float(x.quantile(0.25)), "median": float(x.quantile(0.50)), "p75": float(x.quantile(0.75)),
            "p90": float(x.quantile(0.90)), "max": float(x.quantile(1.0))}


ROLE = {"eta_final": "G-A (final tier)", "eta_dev": "development tier (G-N input)",
        "eta_dev_minus_final": "dev - final, signed (G-N uses the absolute value)", "rmse": "G-A (tier-independent)",
        "tail": "G-A (tier-independent)", "gmax_final": "G-F (final tier)", "gmax_dev": "development tier (G-N input)",
        "gmax_dev_minus_final": "dev - final, signed (G-N uses the absolute value)", "s1": "S1 (secondary)",
        "stage1_rel_err_signed": "S1 signed value (reported)", "eta_dev_minus_final_abs": "G-N criterion 1",
        "gmax_dev_minus_final_abs": "G-N criterion 2",
        "rmse_dev_minus_final": "dev - final of a tier-independent gate metric",
        "tail_dev_minus_final": "dev - final of a tier-independent gate metric",
        "s1_dev_minus_final": "dev - final of a tier-independent criterion"}


def build_t37(pack: C.Pack) -> None:
    """T37: distributions of every gate metric and of dev - final (found rows plus |dev - final| and zero-by-design rows)."""
    dist, pr = _rt(f"{CA}/distributions.csv"), _rt(f"{CA}/per_run.csv")
    gs = C.srcs(f"{LK}/confirmation/q*/seed*/gates.json", label="confirmation gates.json", expect=40)
    extra = []
    gd = []
    for s in gs.files:
        g = _js(s.path)
        a, b = g["reported"]["end_of_A"]["dev_minus_final"], g["reported"]["end_of_B"]["dev_minus_final"]
        gd.append({"q": int(g["q"]), "seed": int(g["seed"]), "rmse_dev_minus_final": a["stage2_rmse_pos_over_g2_0"],
                   "tail_dev_minus_final": a["stage2_tail_mean_over_g2_0"], "s1_dev_minus_final": b["stage1_rel_err_abs"]})
    gd = pd.DataFrame(gd).sort_values(["q", "seed"]).reset_index(drop=True)
    # verification of the found rows against per_run.csv
    max_found = 0.0
    for r in dist.itertuples(index=False):
        st = _quantiles(pr[pr.q == r.q][r.metric])
        max_found = max(max_found, max(abs(st[k] - getattr(r, k)) for k in ("min", "p10", "p25", "median", "p75", "p90",
                                                                             "max")))
    for q in QS:
        g = pr[pr.q == q]
        for m, x, src in (("eta_dev_minus_final_abs", g.eta_dev_minus_final.abs(), "per_run.csv |eta_dev_minus_final|"),
                          ("gmax_dev_minus_final_abs", g.gmax_dev_minus_final.abs(), "per_run.csv |gmax_dev_minus_final|")):
            extra.append({"root": dist.root.iloc[0], "q": q, "metric": m, **_quantiles(x),
                          "row_source": f"generated: {src}"})
        h = gd[gd.q == q]
        for m in ("rmse_dev_minus_final", "tail_dev_minus_final", "s1_dev_minus_final"):
            extra.append({"root": dist.root.iloc[0], "q": q, "metric": m, **_quantiles(h[m]),
                          "row_source": "generated: gates.json reported.*.dev_minus_final"})
    base = dist.copy()
    base["row_source"] = "found: confirmation_analysis/distributions.csv"
    df = pd.concat([base, pd.DataFrame(extra)], ignore_index=True)
    df["q"] = df.q.astype(int)
    df = df.sort_values(["q"], kind="stable").reset_index(drop=True)
    df.insert(3, "gate_role", df.metric.map(ROLE))
    notes = ("Rows with row_source 'found' are confirmation_analysis/distributions.csv as is (values equal the file; "
             f"recomputed from per_run.csv with the same quantile method, max abs diff {max_found}). Added rows: the G-N "
             "gate metrics |eta_dev - eta_final| and |gmax_dev - gmax_final| from per_run.csv, and dev - final of the "
             "tier-independent gate metrics (RMSE, tail, S1) from gates.json reported.end_of_A / end_of_B "
             "dev_minus_final; quantiles with pandas linear interpolation (= numpy linear) as in the analysis script.")
    _table(pack, "T37", df, status="generated", sources=[C.src(f"{CA}/distributions.csv"), C.src(f"{CA}/per_run.csv"), gs],
               script=f"{MOD}:build_t37", notes=notes, tier="final and development",
               stat_tier="as the metric (final, development, dev - final or tier-independent; see gate_role)",
               docs={**DOC_DIST, "metric": dict(definition="Per-run quantity summarised: the per_run.csv column, or "
                                                "<metric>_abs = absolute value, or <metric>_dev_minus_final of a "
                                                "tier-independent metric", units="text"),
                     "gate_role": "Role of the quantity in the locked criteria",
                     "row_source": "found (row of distributions.csv) or generated (added by the builder; source named)",
                     "n": dict(definition="Runs summarised", units="count")},
               caption="Confirmation distributions (n = 20 runs per q) of every gate metric and of the dev - final "
                       "differences; metric units as in T36.")
    pack.crosscheck("T37", dist, REP11, header_has=["root", "q", "metric", "n", "min", "p10"],
                    key_map={"q": "q", "metric": "metric"},
                    value_map={c: c for c in ("n", "min", "p10", "p25", "median", "p75", "p90", "max")}, heading_has="4.4",
                    label="4.4 distributions")
    txt = C.abspath(REP11).read_text(encoding="utf-8")
    cmp = _Cmp(pack, "T37", REP11, "4.4 prose: largest |dev - final|")
    m = re.search(r"largest \|dev − final\| is ([0-9]+(?:\.[0-9]+)?)e−([0-9]+)", txt)
    mx = float(pd.concat([pr.eta_dev_minus_final.abs(), pr.gmax_dev_minus_final.abs()]).max())
    if m:
        cmp.num("max |dev - final| (eta_2 and Gmax_full, both q)", mx, f"{m.group(1)}e-{m.group(2)}", "4.4")
    else:
        cmp.unmatched += 1
    cmp.done()


def build_t38(pack: C.Pack) -> None:
    """T38: S1 summary per q (analysis file) plus the runs that fail S1."""
    s1, pr = _rt(f"{CA}/s1_summary.csv"), _rt(f"{CA}/per_run.csv")
    rows = []
    for r in s1.to_dict("records"):
        g = pr[(pr.q == r["q"]) & ~pr.S1_pass.astype(bool)]
        rows.append({"row_type": "per-q summary", **r, "n_S1_fail": len(g),
                     "S1_fail_seeds": "; ".join(str(int(s)) for s in g.seed)})
    for r in pr[~pr.S1_pass.astype(bool)].itertuples(index=False):
        rd = r._asdict()
        rows.append({"row_type": "S1 failure", "q": int(r.q), "seed": int(r.seed), "stage1_rel_err_signed":
                     float(r.stage1_rel_err_signed), "s1": float(r.s1), "gmax_final": float(r.gmax_final),
                     "run_pass": bool(r.run_pass), "v1_0_run_pass": bool(rd["v1_0_run_pass"])})
    df = pd.DataFrame(rows)
    for c in ("n", "S1_pass", "bootstrap_resamples", "bootstrap_seed", "n_S1_fail", "seed"):
        df[c] = _opt(df[c], int)
    for c in ("run_pass", "v1_0_run_pass"):
        df[c] = _opt(df[c], bool)
    # recomputation from per_run.csv (same method as the analysis script)
    cat = _tool(CONF_TOOL, "_sec_locked_a_conf38")
    dmax = 0.0
    for r in s1.itertuples(index=False):
        g = pr[(pr.q == r.q) & pr.status.isin(["completed", "global_rng_violation"])]
        x = g.stage1_rel_err_signed.astype(float).to_numpy()
        k = int(g.S1_pass.astype(bool).sum())
        lo, hi = cat.clopper_pearson(k, len(x))
        blo, bhi = C.bootstrap_mean_ci(x, int(r.bootstrap_resamples), int(r.bootstrap_seed))
        got = (k, lo, hi, float(np.mean(x)), blo, bhi, float(np.median(x)), float(np.std(x, ddof=1)))
        ref = (r.S1_pass, r.cp95_lo, r.cp95_hi, r.mean_signed, r.boot95_lo, r.boot95_hi, r.median_signed, r.sd_signed)
        dmax = max(dmax, max(abs(float(a) - float(b)) for a, b in zip(got, ref)))
    notes = ("Per-q rows: confirmation_analysis/s1_summary.csv as is (bootstrap 10,000 resamples, numpy seed 20261002, a "
             "fresh generator per q, q ascending; exact Clopper-Pearson CI), plus the number and seeds of the S1 failures; "
             "'S1 failure' rows: the failing runs from per_run.csv with their signed error, Gmax_full/DW (final tier) and "
             "run pass under v1.1 and v1.0. Check: S1 count, CP bounds, mean, bootstrap CI (common.bootstrap_mean_ci, same "
             f"method and seed), median and SD recomputed from per_run.csv equal the file (max abs diff {dmax}).")
    _table(pack, "T38", df, status="generated", sources=[C.src(f"{CA}/s1_summary.csv"), C.src(f"{CA}/per_run.csv"),
                                                        C.src(CONF_TOOL)],
               script=f"{MOD}:build_t38", notes=notes, tier="tier-independent (stage-1 error is a direct policy query)",
               docs={**DOC_S1, "cp95_lo": dict(tier="tier-independent"), "cp95_hi": dict(tier="tier-independent"),
                     "row_type": "per-q summary (s1_summary.csv) or S1 failure (one failing run)",
                     "n_S1_fail": dict(definition="Runs failing S1 (|stage-1 rel. err| > 0.10)", units="count"),
                     "S1_fail_seeds": "Seeds of the runs failing S1",
                     "stage1_rel_err_signed": DOC_PER_RUN["stage1_rel_err_signed"], "s1": DOC_PER_RUN["s1"],
                     "gmax_final": DOC_PER_RUN["gmax_final"], "run_pass": DOC_PER_RUN["run_pass"],
                     "v1_0_run_pass": DOC_PER_RUN["v1_0_run_pass"]},
               caption="S1 (secondary; no pass rule): per q pass count with exact 95% CI, mean signed stage-1 error with "
                       "its bootstrap 95% CI, median and SD; then the runs that fail S1.")
    pack.crosscheck("T38", s1, REP11, header_has=["q", "n", "S1_pass", "cp95_lo", "mean_signed"], key_map={"q": "q"},
                    value_map={c: c for c in _numeric_cols(s1, ("q",))}, heading_has="4.3", label="4.3 S1 table")
    cmp = _Cmp(pack, "T38", REP11, "4.3 formatted S1 table and prose")
    for t in _tables(REP11, ["q", "S1 passes", "exact 95% CI", "median", "SD"], "4.3"):
        cmp.tables += 1
        for row in t["rows"]:
            rec = dict(zip(t["header"], row))
            h = s1[s1.q == int(rec["q"])].iloc[0]
            m = re.match(r"(\d+) / (\d+)", rec["S1 passes"])
            ok = bool(m) and (int(m.group(1)), int(m.group(2))) == (int(h.S1_pass), int(h.n))
            cmp.flag(f"S1 passes [q={rec['q']}]", ok,
                     f"{int(h.S1_pass)} / {int(h.n)}", rec["S1 passes"], f"line {t['line']}")
            ci = re.findall(r"[0-9]+(?:\.[0-9]+)?", rec["exact 95% CI"])
            cmp.num(f"cp95_lo [q={rec['q']}]", h.cp95_lo, ci[0], f"line {t['line']}")
            cmp.num(f"cp95_hi [q={rec['q']}]", h.cp95_hi, ci[1], f"line {t['line']}")
            nums = re.findall(r"[+−-]?[0-9]+(?:\.[0-9]+)?", rec["mean signed stage-1 error [bootstrap 95% CI]"])
            for v, cell in zip((h.mean_signed, h.boot95_lo, h.boot95_hi), nums):
                cmp.num(f"mean/boot CI [q={rec['q']}]", v, cell, f"line {t['line']}")
            cmp.num(f"median_signed [q={rec['q']}]", h.median_signed, rec["median"], f"line {t['line']}")
            cmp.num(f"sd_signed [q={rec['q']}]", h.sd_signed, rec["SD"], f"line {t['line']}")
    txt = C.abspath(REP11).read_text(encoding="utf-8")
    m = re.search(rf"S1 failures are q=60 seed (\d+) \(([+−-]{NUM})\) and seed (\d+) \(([+−-]{NUM})\)", txt)
    fails = pr[~pr.S1_pass.astype(bool)].sort_values(["q", "seed"])
    if m and len(fails) == 2:
        for k in range(2):
            f = fails.iloc[k]
            cmp.flag(f"S1 failure #{k + 1} seed", int(f.seed) == int(m.group(2 * k + 1)) and int(f.q) == 60,
                     f"q{int(f.q)} s{int(f.seed)}", f"q60 s{m.group(2 * k + 1)}", "4.3")
            cmp.num(f"S1 failure #{k + 1} signed error", f.stage1_rel_err_signed, m.group(2 * k + 2), "4.3")
    else:
        cmp.unmatched += 1
    m = re.search(r"Their Ĝmax_full/ΔW is ([0-9]+(?:\.[0-9]+)?) and ([0-9]+(?:\.[0-9]+)?)", txt)
    if m and len(fails) == 2:
        cmp.num("S1 failure #1 gmax_final", fails.iloc[0].gmax_final, m.group(1), "4.3")
        cmp.num("S1 failure #2 gmax_final", fails.iloc[1].gmax_final, m.group(2), "4.3")
    else:
        cmp.unmatched += 1
    cmp.done()
    cmp = _Cmp(pack, "T38", SUMMARY, "summary confirmation table: S1 passes and mean signed error [CI]")
    for t in _tables(SUMMARY, ["q", "S1 passes", "mean signed stage-1 err [boot 95% CI]"], "confirmation"):
        cmp.tables += 1
        for row in t["rows"]:
            rec = dict(zip(t["header"], row))
            h = s1[s1.q == int(rec["q"])].iloc[0]
            cmp.text(f"S1 passes [q={rec['q']}]", f"{int(h.S1_pass)}/{int(h.n)}", rec["S1 passes"], f"line {t['line']}")
            nums = re.findall(r"[+−-]?[0-9]+(?:\.[0-9]+)?", rec["mean signed stage-1 err [boot 95% CI]"])
            for v, cell in zip((h.mean_signed, h.boot95_lo, h.boot95_hi), nums):
                cmp.num(f"mean/boot CI [q={rec['q']}]", v, cell, f"line {t['line']}")
    cmp.done()


def build_t40(pack: C.Pack) -> None:
    """T40: distributions on the development seeds (v1.1 re-rehearsal) against the fresh seeds (confirmation)."""
    rel = f"{CA}/compare_distributions.csv"
    cmpd, d11, dconf = _rt(rel), _rt(f"{RA11}/distributions.csv"), _rt(f"{CA}/distributions.csv")
    mx_c = mx_r = 0.0
    for r in cmpd.itertuples(index=False):
        a = d11[(d11.q == r.q) & (d11.metric == r.metric)].iloc[0]
        b = dconf[(dconf.q == r.q) & (dconf.metric == r.metric)].iloc[0]
        for st in ("min", "p10", "p25", "median", "p75", "p90", "max"):
            mx_c = max(mx_c, abs(getattr(r, f"{st}_compare_root") - a[st]))
            mx_r = max(mx_r, abs(getattr(r, f"{st}_root") - b[st]))
    _found(pack, "T40", rel, script=f"{MOD}:build_t40", docs=DOC_DIST, tier="final and development",
           default_tier="as the metric (final, development, dev - final or tier-independent; see metric)",
                     notes=("*_root = confirmation (seeds 20501-20520, n = 20 per q), *_compare_root = v1.1 re-rehearsal "
                            "(development seeds 10501-10510, n = 10 per q; training state bit-identical to the v1.0 "
                            "rehearsal, R1). Descriptive only. Check: *_compare_root equal rehearsal_v1_1_analysis/"
                            f"distributions.csv (max abs diff {mx_c}); *_root equal confirmation_analysis/distributions.csv "
                            f"(max abs diff {mx_r})."),
                     caption="Distributions of the gate metrics on the development seeds (v1.1 re-rehearsal) and on the "
                             "fresh confirmation seeds, side by side.")
    pack.crosscheck("T40", cmpd, REP11, header_has=["q", "metric", "max_compare_root", "max_root"],
                    key_map={"q": "q", "metric": "metric"}, value_map={c: c for c in _numeric_cols(cmpd, ("q",))},
                    heading_has="5.", label="5 rehearsal vs confirmation")


# ----------------------------------------------------------------------------------------------
# D07-D09 consolidated per-run tables
# ----------------------------------------------------------------------------------------------

def build_d07(pack: C.Pack) -> None:
    """D07: v1.0 rehearsal per-run table (gates_per_run.csv, 112 original columns)."""
    rel = f"{RA}/gates_per_run.csv"
    df = _rt(rel)
    identical = df.to_csv(index=False, lineterminator="\n").encode() == C.abspath(rel).read_bytes()
    pack.data("D07", df, status="found" if identical else "regenerated", sources=[C.src(rel)],
              script=f"{MOD}:build_d07", tier="final and development",
              docs=_docs_final(df.columns, DOC_GATES_V10, "final and development"),
              notes=(f"results/v2_T2_locked/rehearsal_analysis/gates_per_run.csv with its original column names; written "
                     f"file byte-identical to the source: {identical}. 20 runs (q = 50 and 60, seeds 10501-10510, commit "
                     "5b07293, protocol v1.0). Gate values *_final (final tier), *_dev (development tier), A_* end of "
                     "Phase A and B_* end of Phase B (final tier), dec_* stage-1 decomposition, A_dmf_/B_dmf_ dev - final."))
    for hh, vm, lab in ((["q", "seed", "eta_T_over_dw_final", "G-A"], None, "4.3 per-run gates"),
                        (["q", "seed", "A_stage2_peak_rel_err_signed"], None, "4.4 reported metrics, end of A"),
                        (["q", "seed", "B_e1_at_0", "dec_learning_rel"], None, "4.4 reported metrics, end of B")):
        cols = [c for c in _numeric_cols(df, ("q", "seed"))]
        pack.crosscheck("D07", df, REP10, header_has=hh, key_map={"q": "q", "seed": "seed"},
                        value_map={c: c for c in cols}, label=lab)
    _crosscheck_text(pack, "D07", df, REP10, ["q", "seed", "eta_T_over_dw_final", "G-A"], {"q": "q", "seed": "seed"},
                     {c: c for c in ("G-A", "G-F", "run_pass", "outcome", "G-A_dev", "G-F_dev")}, "",
                     "4.3 per-run verdicts")


def _merged(study_analysis: str) -> Tuple[pd.DataFrame, List[Any]]:
    """per_run.csv merged with reported_metrics.csv on (q, seed), original column names."""
    a, b = _rt(f"{study_analysis}/per_run.csv"), _rt(f"{study_analysis}/reported_metrics.csv")
    overlap = (set(a.columns) & set(b.columns)) - {"q", "seed"}
    if overlap:
        raise ValueError(f"per_run and reported_metrics share columns {overlap}")
    if a[["q", "seed"]].duplicated().any() or b[["q", "seed"]].duplicated().any():
        raise ValueError("duplicate (q, seed) keys")
    df = a.merge(b, on=["q", "seed"], how="left", validate="one_to_one")
    if df[b.columns.drop(["q", "seed"])].isna().all(axis=1).any():
        raise ValueError("per-run rows without reported metrics")
    return df, [C.src(f"{study_analysis}/per_run.csv"), C.src(f"{study_analysis}/reported_metrics.csv")]


def build_d08(pack: C.Pack) -> None:
    """D08: v1.1 re-rehearsal per-run table (per_run.csv + reported_metrics.csv)."""
    df, srcs_ = _merged(RA11)
    pack.data("D08", df, status="regenerated", sources=srcs_, script=f"{MOD}:build_d08", tier="final and development",
              docs=_docs_final(df.columns, {**DOC_PER_RUN, **DOC_REPORTED}, "final and development"),
              notes=("rehearsal_v1_1_analysis/per_run.csv joined one-to-one with reported_metrics.csv on (q, seed); "
                     "original column names (the files share only q and seed), values unchanged (exact float parsing). 20 "
                     "runs (q = 50 and 60, seeds 10501-10510, commit 95c000e, protocol v1.1)."))
    pack.crosscheck("D08", df, REP11, header_has=["q", "seed", "eta_final", "gmax_final"],
                    key_map={"q": "q", "seed": "seed"},
                    value_map={c: c for c in _numeric_cols(df, ("q", "seed"))}, heading_has="3. Re-rehearsal",
                    label="3 per-run table (re-rehearsal)")
    _crosscheck_text(pack, "D08", df, REP11, ["q", "seed", "eta_final", "gmax_final"], {"q": "q", "seed": "seed"},
                     {c: c for c in ("status", "G-A_pass", "G-F_pass", "G-N_pass", "S1_pass", "global_rng", "run_pass",
                                     "v1_0_G-F", "v1_0_run_pass", "outcome")}, "3. Re-rehearsal",
                     "3 per-run verdicts (re-rehearsal)")
    v10 = _rt(f"{RA}/gates_per_run.csv")
    m = df.merge(v10, on=["q", "seed"], suffixes=("", "_v10"))
    same = [C.max_abs_diff(m.eta_final, m.eta_T_over_dw_final), C.max_abs_diff(m.gmax_final, m.Gmax_full_over_dw_final),
            C.max_abs_diff(m.s1, m.stage1_rel_err_abs_final), C.max_abs_diff(m.B_e1_at_0, m.B_e1_at_0_v10)]
    cmp = _Cmp(pack, "D08", REP11, "3: re-rehearsal gate values equal the v1.0 rehearsal (R1)")
    cmp.flag("max abs diff eta / gmax / s1 / e1 vs v1.0 rehearsal", max(same) == 0.0, max(same), "0 (R1 identical)")
    cmp.flag("v1.0 outcome on the re-rehearsal runs (q50, q60)",
             (int(df[df.q == 50].v1_0_run_pass.sum()), int(df[df.q == 60].v1_0_run_pass.sum())) ==
             (int(v10[v10.q == 50].run_pass.sum()), int(v10[v10.q == 60].run_pass.sum())),
             f"{int(df[df.q == 50].v1_0_run_pass.sum())}/{int(df[df.q == 60].v1_0_run_pass.sum())}",
             "9/10 and 8/10 (matching the v1.0 rehearsal)")
    cmp.done()


def build_d09(pack: C.Pack) -> None:
    """D09: confirmation per-run table (per_run.csv + reported_metrics.csv)."""
    df, srcs_ = _merged(CA)
    pack.data("D09", df, status="regenerated", sources=srcs_, script=f"{MOD}:build_d09", tier="final and development",
              docs=_docs_final(df.columns, {**DOC_PER_RUN, **DOC_REPORTED}, "final and development"),
              notes=("confirmation_analysis/per_run.csv joined one-to-one with reported_metrics.csv on (q, seed); original "
                     "column names (the files share only q and seed), values unchanged (exact float parsing). 40 runs (q = "
                     "50 and 60, seeds 20501-20520, commit f6838ec, protocol v1.1)."))
    pack.crosscheck("D09", df, REP11, header_has=["q", "seed", "eta_final", "gmax_final"],
                    key_map={"q": "q", "seed": "seed"},
                    value_map={c: c for c in _numeric_cols(df, ("q", "seed")) if c in _rt(f"{CA}/per_run.csv").columns},
                    heading_has="4.2", label="4.2 per-run table")
    pack.crosscheck("D09", df, REP11, header_has=["q", "seed", "A_stage2_peak_rel_err_signed", "dec_learning_rel"],
                    key_map={"q": "q", "seed": "seed"},
                    value_map={c: c for c in _numeric_cols(df, ("q", "seed"))}, heading_has="4.5",
                    label="4.5 reported metrics")
    _crosscheck_text(pack, "D09", df, REP11, ["q", "seed", "A_stage2_peak_rel_err_signed", "dec_learning_rel"],
                     {"q": "q", "seed": "seed"},
                     {c: c for c in ("dec_learning_contains_0", "dec_inherited_contains_0", "dec_band_contiguous",
                                     "dec_e1_inside_sweep", "drift_test_pass")}, "4.5", "4.5 reported metrics (flags)")


# ----------------------------------------------------------------------------------------------
# entry
# ----------------------------------------------------------------------------------------------

def build() -> None:
    """Build every item of this module and save the fragment."""
    pack = C.Pack("sec_locked_a")
    build_t30(pack)
    build_t31(pack)
    build_f11(pack)
    build_t32(pack)
    build_t33(pack)
    build_t34(pack)
    build_t35(pack)
    build_t36(pack)
    build_t37(pack)
    build_t38(pack)
    build_t40(pack)
    build_d07(pack)
    build_d08(pack)
    build_d09(pack)
    pack.save_fragment()
