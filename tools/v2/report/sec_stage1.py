"""Pack items of Part II sections 9, 11 and 12: stage-1 accuracy, protocol engineering, issue list.

Items: T48 (induced-target method), F22 (Delta_1 against e), T49 (root stage game), T50 (anatomy
of the stage-1 fluctuation), F23 (ACF plot), T51 (Pilot 4 section 2b), F24 (Pilot 4 section 2b
learning curves), T54 (v1.0 Check 1 and Check 2), T55 (consolidation), T56 (v1.1 global-RNG
hardening), T57 (issue list).

Everything is read from saved files (CSV/JSON/NPZ records, saved full states, reports, code). No
forward pass and no verifier evaluation is run, so nothing is logged with ``pack.reeval``.
Comparisons with report prose (numbers that are not in a parseable report table) are recorded
with :class:`_Checks`, which uses the same rounding rule as ``Pack.crosscheck``.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import common as C
import style
import studies as S

if str(C.REPO) not in sys.path:
    sys.path.insert(0, str(C.REPO))

MOD = "sec_stage1.py"
QS = (50, 60)

IB = "results/v2_pilots/induced_band"
P2A = "results/v2_pilots/pilot2/analysis"
P3A = "results/v2_pilots/pilot3/analysis"
P4A = "results/v2_pilots/pilot4/analysis"
RG = f"{P4A}/root_game"
FL = f"{P4A}/fluctuation"
LK = "results/v2_T2_locked"
CA = f"{LK}/confirmation_analysis"
RA = f"{LK}/rehearsal_analysis"
CONS = f"{LK}/consolidation"
PROTO = "protocols/v2_T2_locked_v1_1.json"

RPT_P0 = "reports/v2/phase0_audit.md"
RPT_P1 = "reports/v2/phase1_verifier.md"
RPT_OC = "reports/v2/phase2_opening_checks.md"
RPT_P2 = "reports/v2/pilot2_freeze.md"
RPT_P3 = "reports/v2/pilot3_continuation_mode.md"
RPT_EXT = "reports/v2/phaseA_ext.md"
RPT_P4 = "reports/v2/pilot4_stabilization.md"
RPT_LOCK = "reports/v2/protocol_lock_and_rehearsal.md"
RPT_V11 = "reports/v2/protocol_v1_1_confirmation.md"
RPT_SUM = "reports/v2/summary.md"
RPT_DR = "reports/v2/dreach_reach_mask_check.md"

ARM_SHORT_P2 = {"A_joint": "A", "B1_frozen_allnorm": "B1", "B2_frozen_s1norm": "B2"}
ARM_P4B = {"B2_mean_constant": "constant", "B2_mean_decay": "decay"}

# column documentation shared by the long-format tables of this module
D_BLOCK = "Block of the table (one sub-part of the item); columns that do not apply to a block are empty"
D_QUANT = "Name of the quantity reported in the row"
D_VALUE = "Value of the quantity (number, flag or text); its unit is in `unit`"
D_UNIT = "Unit and normalization of `value`"
D_SRC = ("File(s) the value was read or computed from (repo-relative); 'source: report text' marks values "
         "copied from a report because no data file holds them")
D_Q_ALL = "q (50 or 60); 'all' = both q pooled; '50 and 60' = applies to both"


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------

def _js(rel: str) -> Any:
    """JSON file (repo-relative; results/ resolves against the results root)."""
    return S.read_json(rel)


def _csv(rel: str, **kw) -> pd.DataFrame:
    """CSV file (repo-relative)."""
    return S.read_csv(rel, **kw)


def _text(rel: str) -> str:
    """Text of a repo file."""
    return C.abspath(rel).read_text(encoding="utf-8")


def _dw() -> float:
    """Delta W = w_h - w_l from the locked protocol records (identical for both q)."""
    rec = _js(PROTO)["records"]
    vals = sorted({float(rec[k]["dw"]) for k in rec})
    if len(vals) != 1:
        raise ValueError(f"Delta W differs between q: {vals}")
    return vals[0]


def _g1() -> Dict[int, float]:
    """e_1*(0) per q (floors.json of the induced-band tool)."""
    return {int(k): float(v["g1"]) for k, v in _js(f"{IB}/floors.json").items()}


def _f(x: Any, sig: int = 4) -> str:
    """Compact text of a number for evidence strings (sig significant digits)."""
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "nan"
    if v == int(v) and abs(v) < 1e6:
        return str(int(v))
    return f"{v:.{sig}g}"


def _stat(values: Iterable[float]) -> Dict[str, float]:
    """median, q25, q75, min, max, n (numpy linear interpolation; pack convention)."""
    return C.median_iqr(list(values))


def _b(s: pd.Series) -> pd.Series:
    """Boolean view of a flag column stored as bool or as yes/no / True/False text (raises on other values)."""
    if s.dtype == bool:
        return s
    m = {"true": True, "yes": True, "1": True, "1.0": True, "false": False, "no": False, "0": False, "0.0": False}
    out = s.astype(str).str.strip().str.lower().map(m)
    if out.isna().any():
        raise ValueError(f"non-boolean values in flag column {s.name}: {sorted(set(s.astype(str)) - set(m))}")
    return out.astype(bool)


def _cell_ok(x: Any, cell: str) -> bool:
    """Whether x rounds to a report cell; '0.00'-style cells use their decimal precision."""
    v = C.parse_num(cell)
    if v is None or x is None:
        return False
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(xf):
        return False
    t = str(cell).strip().replace("−", "-").rstrip("%")
    if v == 0.0 and "." in t and "e" not in t.lower():
        dec = len(t.split(".")[1])
        return abs(xf) <= 0.5 * 10.0 ** (-dec) * 1.0001
    return C.consistent(xf, str(cell))


class _Checks:
    """Comparisons of pack values with numbers in report prose or in tables that ``crosscheck`` cannot parse.

    Mismatches go to ``pack.mismatch``; one summary record per instance is appended to the pack's
    cross-check list (same fields as ``Pack.crosscheck`` records).
    """

    def __init__(self, pack: C.Pack, item: str, report: str, label: str):
        self.pack, self.item, self.report, self.label = pack, item, report, label
        self.n = 0
        self.bad = 0

    def num(self, quantity: str, x: Any, cell: str, where: str, comment: str = "") -> bool:
        """Compare one number at the report's displayed precision."""
        self.n += 1
        ok = _cell_ok(x, cell)
        if not ok:
            self.bad += 1
            self.pack.mismatch(self.item, quantity, x, f"{self.report} ({where})", cell, comment)
        return ok

    def same(self, quantity: str, pack_value: Any, report_value: Any, where: str, comment: str = "") -> bool:
        """Compare two non-numeric values for equality (text, flags)."""
        self.n += 1
        ok = str(pack_value) == str(report_value)
        if not ok:
            self.bad += 1
            self.pack.mismatch(self.item, quantity, pack_value, f"{self.report} ({where})", report_value, comment)
        return ok

    def differ(self, quantity: str, pack_value: Any, report_value: Any, where: str, comment: str) -> None:
        """Record a known disagreement between data and a report statement."""
        self.n += 1
        self.bad += 1
        self.pack.mismatch(self.item, quantity, pack_value, f"{self.report} ({where})", report_value, comment)

    def close(self) -> None:
        """Append the summary record (call once)."""
        self.pack.crosschecks.append({"item": self.item, "report": self.report, "label": self.label,
                                      "n_tables": 0, "n_compared": self.n, "n_mismatch": self.bad,
                                      "n_unmatched_rows": 0})


def _md_tables(rel: str, header_has: Sequence[str], heading_has: str = "") -> List[Dict[str, Any]]:
    """Report tables whose header contains all names (rows with unescaped '|' in a cell are repaired)."""
    out = []
    for t in C.parse_md_tables(rel):
        if not all(h in t["header"] for h in header_has) or heading_has.lower() not in t["heading"].lower():
            continue
        n = len(t["header"])
        rows = []
        for r in t["rows"]:
            r = list(r)
            # '|x|' labels inside a cell split into ['', 'x', '']: merge them back
            while len(r) > n:
                for i in range(len(r) - 2):
                    if r[i] == "" and r[i + 2] == "":
                        r = r[:i] + ["|" + r[i + 1] + "|"] + r[i + 3:]
                        break
                else:
                    break
            rows.append(r)
        out.append({**t, "rows": rows})
    return out


def _parse_bracket(cell: str) -> Optional[Tuple[float, float, float]]:
    """'m [lo, hi]' -> (m, lo, hi) as floats; also '[lo, hi]' -> (nan, lo, hi)."""
    t = str(cell).replace("−", "-").replace("**", "").strip()
    m = re.match(r"^\s*([-+]?[\d.]+(?:e[-+]?\d+)?)?\s*\[\s*([-+]?[\d.]+(?:e[-+]?\d+)?)\s*,\s*"
                 r"([-+]?[\d.]+(?:e[-+]?\d+)?)\s*\]\s*$", t)
    if not m:
        return None
    return (float(m.group(1)) if m.group(1) else float("nan"), float(m.group(2)), float(m.group(3)))


def _bracket_strs(cell: str) -> Optional[Tuple[str, str, str]]:
    """The three number strings of 'm [lo, hi]' (to compare at their own precision)."""
    t = str(cell).replace("−", "-").replace("**", "").strip()
    m = re.match(r"^\s*([-+]?[\d.]+(?:e[-+]?\d+)?)?\s*\[\s*([-+]?[\d.]+(?:e[-+]?\d+)?)\s*,\s*"
                 r"([-+]?[\d.]+(?:e[-+]?\d+)?)\s*\]\s*$", t)
    if not m:
        return None
    return (m.group(1) or "", m.group(2), m.group(3))


def _union(frames: Sequence[pd.DataFrame], cols: Sequence[str]) -> pd.DataFrame:
    """Concatenate block frames into one table with the given column order (missing -> empty)."""
    if len(set(cols)) != len(cols):
        raise ValueError(f"duplicate column names: {sorted(c for c in set(cols) if list(cols).count(c) > 1)}")
    parts = []
    for f in frames:
        g = f.copy()
        for c in cols:
            if c not in g.columns:
                g[c] = None
        parts.append(g[list(cols)])
    out = pd.concat(parts, ignore_index=True)
    return out.astype(object).where(pd.notna(out), None)


def _locked_band_rows(root: str) -> Tuple[pd.DataFrame, C.SourceSet]:
    """induced_band.json of every run of a locked root, one row per (q, seed)."""
    ss = C.srcs(f"{LK}/{root}/q*/seed*/induced_band.json", label=f"{LK}/{root}/q*/seed*/induced_band.json")
    rows = []
    for s in ss.files:
        m = re.search(r"/q(\d+)/seed(\d+)/", s.path)
        j = _js(s.path)
        rows.append({"q": int(m.group(1)), "seed": int(m.group(2)), **j})
    return pd.DataFrame(rows).sort_values(["q", "seed"]).reset_index(drop=True), ss


def _band_sets(dw: float) -> Tuple[List[Tuple[str, pd.DataFrame]], List[Any]]:
    """The five sets of induced bands with a common schema, and their sources."""
    pb = _csv(f"{IB}/parent_bands.csv")
    jb = _csv(f"{IB}/joint_export_bands.csv")
    p4 = _csv(f"{IB}/parent4_bands.csv")
    reh, s_reh = _locked_band_rows("rehearsal")
    conf, s_conf = _locked_band_rows("confirmation")
    for d in (reh, conf):
        d["delta1_min_over_dw"] = d["delta1_min"] / dw
    sets = [("Pilot 1 u400 parents: frozen stage 2 of Pilot 2 B1/B2 and Pilot 3 (parent_bands.csv)", pb),
            ("Pilot 2 A_joint weight exports u425-u1000: live stage 2 (joint_export_bands.csv)", jb),
            ("Phase A extension u1600 parents: frozen stage 2 of Pilot 4 2b (parent4_bands.csv)", p4),
            ("v1.0 rehearsal, end of Phase B (rehearsal/q*/seed*/induced_band.json)", reh),
            ("v1.1 confirmation, end of Phase B (confirmation/q*/seed*/induced_band.json)", conf)]
    srcs = [C.src(f"{IB}/parent_bands.csv"), C.src(f"{IB}/joint_export_bands.csv"),
            C.src(f"{IB}/parent4_bands.csv"), s_reh, s_conf]
    return sets, srcs


def _band_stats(d: pd.DataFrame) -> Dict[str, Any]:
    """Width, contiguity and location statistics of a set of bands."""
    w = (d["band_hi"] - d["band_lo"]) / d["g1"]
    return {"n_bands": int(len(d)), "band_width_over_e1star_median": float(np.median(w)),
            "band_width_over_e1star_min": float(w.min()), "band_width_over_e1star_max": float(w.max()),
            "band_n_points_median": float(np.median(d["band_n_points"])),
            "n_noncontiguous": int((~_b(d["band_contiguous"])).sum()),
            "n_argmin_at_sweep_edge": int(_b(d["argmin_at_sweep_edge"]).sum()),
            "delta1_min_over_dw_max": float(d["delta1_min_over_dw"].max()),
            "e_tilde_min": float(d["e_tilde"].min()), "e_tilde_max": float(d["e_tilde"].max()),
            "sweep_lo_min": float(d["sweep_lo"].min()), "sweep_hi_max": float(d["sweep_hi"].max())}


BAND_UNITS = {"n_bands": "count", "band_width_over_e1star_median": "fraction of e_1*(0)",
              "band_width_over_e1star_min": "fraction of e_1*(0)", "band_width_over_e1star_max": "fraction of e_1*(0)",
              "band_n_points_median": "count (sweep points)", "n_noncontiguous": "count",
              "n_argmin_at_sweep_edge": "count", "delta1_min_over_dw_max": "Delta W (dimensionless)",
              "e_tilde_min": "effort units (raw)", "e_tilde_max": "effort units (raw)",
              "sweep_lo_min": "effort units (raw)", "sweep_hi_max": "effort units (raw)"}


def _pilot2_checkpoints() -> Tuple[pd.DataFrame, C.SourceSet]:
    """Training-time checkpoint rows of the 60 Pilot 2 runs (superseded-solver columns)."""
    ss = C.srcs("results/v2_pilots/pilot2/q*/seed*/*/v2_checkpoints.csv", label="results/v2_pilots/pilot2/q*/seed*/*/v2_checkpoints.csv",
                expect=60)
    cols = ["q", "seed", "arm", "update", "induced_n_sign_changes", "induced_e1", "induced_residual"]
    parts = [_csv(s.path)[cols] for s in ss.files]
    return pd.concat(parts, ignore_index=True), ss


# ----------------------------------------------------------------------------------------------
# T48 induced-target method
# ----------------------------------------------------------------------------------------------

def build_t48(pack: C.Pack) -> None:
    """Induced-target method: definition, calibration gate, band statistics, coverage, superseded solver."""
    dw, g1 = _dw(), _g1()
    fl = _js(f"{IB}/floors.json")
    cal = _csv(f"{IB}/calibration.csv")
    sets, set_srcs = _band_sets(dw)
    r1 = _csv(f"{LK}/rehearsal_v1_1_R1_details.csv")
    d2 = _csv(f"{P2A}/decomposition_residual_band.csv")
    d3 = _csv(f"{P3A}/decomposition_residual_band.csv")
    cand = _csv(f"{P4A}/candidates_all.csv")
    ft2 = _csv(f"{P2A}/final_table.csv")
    ck2, ck2_src = _pilot2_checkpoints()
    rows: List[Dict[str, Any]] = []

    def add(block: str, subset: str, q: Any, quantity: str, value: Any, unit: str, source: str) -> None:
        rows.append({"block": block, "subset": subset, "q": q, "quantity": quantity, "value": value,
                     "unit": unit, "source": source})

    # ---- 1 definition (compiled from code and report text)
    b1 = "1 definition"
    rt = f"{RPT_P2} section 6 'Method' (source: report text)"
    add(b1, "all bands", "50 and 60", "residual Delta_1(e; e_hat_2)",
        "stage-1 one-step root residual delta[0] returned by the verifier's verify() when both players' root action is "
        "set to e and the candidate's stage-2 mapping e_hat_2 is the continuation for both players; final tier (state "
        "step 2, effort step 0.5, GL 32 nodes per half); the verifier's own action search (effort grid, concave vertices, "
        "own action, dynamic-BR action), no extra refinement; Delta_1 >= 0 because the own action is a candidate",
        "value units (Delta W = %s for /DW)" % _f(dw), f"utils/v2_metrics.py:stage1_residual_sweep; {rt}")
    add(b1, "all bands", "50 and 60", "sweep grid",
        "e_1*(0) + k * 0.01 (k integer) covering the sweep range, clipped to [0, 100]; e_1*(0) is an exact sweep node",
        "effort units (raw)", "utils/v2_metrics.py:sweep_grid; tools/v2/induced_band.py (STEP = 0.01)")
    for subset, rng in (
            ("Pilot 1 u400 parents (frozen stage 2 of Pilot 2 B1/B2 and Pilot 3)", "[0.4 e_1*, 1.7 e_1*]"),
            ("Pilot 2 A_joint weight exports (live stage 2)", "[min(e_hat_1, e_1*) - 2, max(e_hat_1, e_1*) + 2]"),
            ("Phase A extension u1600 parents (frozen stage 2 of Pilot 4 2b)",
             "[min(0.4 e_1*, e1_min - 2), max(1.7 e_1*, e1_max + 2)], e1_min/e1_max over every checkpoint and export "
             "row of both 2b arms of the (q, seed)"),
            ("calibration (e_hat_2 = e_2*)", "[e_1* - 5, e_1* + 5]"),
            ("locked protocol runs (end of Phase B)", "[min(0.4 e_1*, e_hat_1 - 2), max(1.7 e_1*, e_hat_1 + 2)]")):
        add(b1, subset, "50 and 60", "sweep range", rng, "effort units (raw)",
            "tools/v2/induced_band.py (docstring, PARENT_RANGE, MARGIN); " +
            (f"{PROTO} stage1_decomposition.sweep_range" if subset.startswith("locked") else rt))
    add(b1, "all bands", "50 and 60", "point estimate e~_1",
        "sweep point with the smallest Delta_1 (numpy argmin: the first such point on ties)", "effort units (raw)",
        "utils/v2_metrics.py:induced_band")
    add(b1, "all bands", "50 and 60", "band",
        "{e in sweep: Delta_1(e) <= Delta_1,min + floor}; band_lo / band_hi = the outermost band points, used as the "
        "interval also when the band is not contiguous", "effort units (raw)", f"utils/v2_metrics.py:induced_band; {rt}")
    for q in QS:
        f = fl[str(q)]
        add(b1, "all bands", q, "floor = max(Delta_1(e_1*; e_2*) on the final tier, 1e-12 Delta W)", float(f["floor"]) / dw,
            "Delta W (dimensionless)", f"{IB}/floors.json")
        add(b1, "all bands", q, "Delta_1(e_1*; e_2*) on the final tier (calibration residual)",
            float(f["calibration_delta1"]) / dw, "Delta W (dimensionless)", f"{IB}/floors.json")
    add(b1, "all bands", "50 and 60", "learning term",
        "(e_hat_1 - e~_1) / e_1*, interval [(e_hat_1 - band_hi) / e_1*, (e_hat_1 - band_lo) / e_1*]",
        "fraction of e_1*(0)", "tools/v2/decomposition.py:rows_for")
    add(b1, "all bands", "50 and 60", "inherited term",
        "(e~_1 - e_1*) / e_1*, interval [(band_lo - e_1*) / e_1*, (band_hi - e_1*) / e_1*]", "fraction of e_1*(0)",
        "tools/v2/decomposition.py:rows_for")
    add(b1, "all bands", "50 and 60", "rows decomposed",
        "Pilot 2: every weight export u425-u1000 of all arms and every training-time checkpoint of B1/B2 (band of the "
        "parent, constant); A_joint training-time checkpoints are not covered (stage-2 weights exist only at the 25-update "
        "exports). Pilot 3: every checkpoint and export against the parent band. Pilot 4 2b: every export and tail "
        "candidate against the u1600 parent band. Locked runs: the end-of-B last iterate (induced_band.json)", "text",
        f"tools/v2/decomposition.py; {rt}; {RPT_P3} section 1; {RPT_P4} section 2b (source: report text)")
    add(b1, "superseded solver (Appendix S)", "50 and 60", "method",
        "fixed point of h(e) = BR(e) - e with BR = argmax of the verifier's interpolated stage-1 Q (dense grid search plus "
        "bounded scalar refinement; the opponent's own action excluded), bracketed on the effort grid and refined with "
        "Brent's method (xtol 1e-10), first root reported; evaluated at every training-time checkpoint on the development "
        "tier; residual |BR(e~_1) - e~_1| logged as induced_residual", "text",
        f"utils/v2_metrics.py:induced_stage1_target; {RPT_P2} section 1 and Appendix S (source: report text)")

    # ---- 2 calibration gate
    b2 = "2 calibration gate (e_hat_2 = e_2*)"
    for r in cal.itertuples(index=False):
        q = int(r.q)
        for qty, val, unit in (
                ("e~_1", r.e_tilde, "effort units (raw)"), ("band_lo", r.band_lo, "effort units (raw)"),
                ("band_hi", r.band_hi, "effort units (raw)"), ("band width / e_1*", r.band_width_over_e1star,
                                                              "fraction of e_1*(0)"),
                ("band points", int(r.band_n_points), "count"), ("band contiguous", bool(r.band_contiguous), "bool"),
                ("Delta_1,min / DW", float(r.delta1_min) / dw, "Delta W (dimensionless)"),
                ("argmin at sweep edge", bool(r.argmin_at_sweep_edge), "bool"),
                ("band contains e_1*", bool(r.contains_e1star), "bool"),
                ("e_1* is a sweep node", bool(r.e1star_is_sweep_node), "bool")):
            add(b2, "calibration sweep e_1* +- 5 (calibration.csv)", q, qty, val, unit, f"{IB}/calibration.csv")
        cal_d1 = float(fl[str(q)]["calibration_delta1"])
        by_constr = bool(r.e1star_is_sweep_node) and (cal_d1 - float(r.delta1_min) <= float(r.floor))
        add(b2, "calibration sweep e_1* +- 5 (calibration.csv)", q,
            "containment guaranteed by construction (e_1* is a sweep node, Delta_1 >= 0, floor >= Delta_1(e_1*; e_2*))",
            by_constr, "bool", f"{IB}/calibration.csv; {IB}/floors.json")

    # ---- 3 band width and contiguity
    b3 = "3 band width and contiguity"
    for (name, d), s in zip(sets, set_srcs):
        src_txt = s.path if isinstance(s, C.Source) else s.label
        for q in (50, 60, "all"):
            dd = d if q == "all" else d[d["q"] == q]
            for k, v in _band_stats(dd).items():
                add(b3, name, q, k, v, BAND_UNITS[k], src_txt)
    n_r1 = int(_b(r1["induced_band.json"]).sum())
    add(b3, "v1.1 re-rehearsal, end of Phase B", "all", "runs whose induced_band.json and band_sweep.npz equal the v1.0 "
        "rehearsal (R1)", f"{n_r1}/{len(r1)}", "count", f"{LK}/rehearsal_v1_1_R1_details.csv")

    # ---- 4 sweep-range coverage
    b4 = "4 sweep-range coverage"
    cov = [("Pilot 2 decomposition rows (all checkpoints and exports)", d2, f"{P2A}/decomposition_residual_band.csv"),
           ("Pilot 3 decomposition rows (all checkpoints and exports)", d3, f"{P3A}/decomposition_residual_band.csv")]
    for name, d, src in cov:
        out = d[~_b(d["e1_inside_sweep"])].sort_values(["q", "seed", "arm", "update"])
        add(b4, name, "all", "rows", int(len(d)), "count", src)
        add(b4, name, "all", "rows with e_hat_1(0) outside the sweep", int(len(out)), "count", src)
        add(b4, name, "all", "outside rows at the final checkpoint (u1000)", int((out["update"] == 1000).sum()),
            "count", src)
        add(b4, name, "all", "sources of the outside rows (weights = 25-update export, checkpoint = training-time verifier "
            "checkpoint)", "; ".join(f"{k}: {v}" for k, v in out["source"].value_counts().sort_index().items()),
            "count", src)
        for r in out.itertuples(index=False):
            add(b4, f"{name}: q{r.q} seed {r.seed} {r.arm} u{r.update} ({r.source})", int(r.q),
                "e_hat_1(0) / e_1*(0) of an outside row (sweep upper end 1.7 e_1*)", float(r.e1_at_0) / g1[int(r.q)],
                "fraction of e_1*(0)", src)
    o2 = d2[~_b(d2["e1_inside_sweep"])]
    o3 = d3[~_b(d3["e1_inside_sweep"])]
    key = ["q", "seed", "update", "source", "e1_at_0"]
    dup = o3[o3["arm"] == "B2_frozen_s1norm"].merge(o2[o2["arm"] == "B2_frozen_s1norm"][key], on=key)
    add(b4, "Pilot 3 outside rows identical to Pilot 2 B2 outside rows (stochastic arm = Pilot 2 B2, bit-identical)",
        "all", "rows", int(len(dup)), "count",
        f"{P2A}/decomposition_residual_band.csv; {P3A}/decomposition_residual_band.csv")
    c2b = cand[cand["family"] == "2b"]
    add(b4, "Pilot 4 2b candidates (exports u1625-u2200 and tail candidates)", "all", "rows", int(len(c2b)), "count",
        f"{P4A}/candidates_all.csv")
    add(b4, "Pilot 4 2b candidates (exports u1625-u2200 and tail candidates)", "all",
        "rows with e_hat_1(0) outside the sweep", int((~_b(c2b["e1_inside_sweep"])).sum()), "count",
        f"{P4A}/candidates_all.csv")
    for name, d in sets[3:]:
        add(b4, name, "all", "runs with e_hat_1(0) outside the sweep",
            f"{int((~_b(d['e1_inside_sweep'])).sum())}/{len(d)}", "count", name.split("(")[-1].rstrip(")"))

    # ---- 5 comparison with the superseded solver
    b5 = "5 comparison with the superseded solver (u1000, /e_1*)"
    fin = d2[(d2["source"] == "weights") & (d2["update"] == 1000)].copy()
    fin["abs_l"] = fin["learning_rel"].abs()
    fin["abs_i"] = fin["inherited_rel"].abs()
    comp = []
    for (q, arm), g in fin.groupby(["q", "arm"], sort=True):
        f2 = ft2[(ft2["q"] == q) & (ft2["arm"] == arm)]
        rec = {"q": int(q), "arm": ARM_SHORT_P2[arm], "arm_full": arm,
               "revised_median_abs_learning": float(np.median(g["abs_l"])),
               "revised_median_inherited": float(np.median(g["inherited_rel"])),
               "revised_median_abs_inherited": float(np.median(g["abs_i"])),
               "superseded_median_abs_learning": float(np.median(f2["stage1_learning_err_rel_abs"])),
               "superseded_median_inherited": float(np.median(f2["stage1_inherited_err_rel"])),
               "superseded_median_abs_inherited": float(np.median(f2["stage1_inherited_err_rel_abs"])),
               "revised_learning_band_contains_0": int(_b(g["learning_contains_0"]).sum()),
               "revised_inherited_band_contains_0": int(_b(g["inherited_contains_0"]).sum()),
               "n_runs": int(len(g))}
        comp.append(rec)
        for k, v in rec.items():
            if k in ("q", "arm", "arm_full"):
                continue
            unit = "count" if k.startswith("n_") or k.endswith("contains_0") else "fraction of e_1*(0)"
            src = (f"{P2A}/final_table.csv (development tier, last checkpoint u1000)" if k.startswith("superseded")
                   else f"{P2A}/decomposition_residual_band.csv (source == weights, update == 1000; final tier)")
            add(b5, f"Pilot 2 arm {arm}", int(q), k, v, unit, src)
    ck2 = ck2.copy()
    ck2["grp"] = np.where(ck2["arm"] == "A_joint", "A_joint", "B1_frozen_allnorm + B2_frozen_s1norm")
    ck_src = "results/v2_pilots/pilot2/q*/seed*/*/v2_checkpoints.csv (column induced_residual; development tier)"
    for (q, grp), g in ck2.groupby(["q", "grp"], sort=True):
        r = g["induced_residual"].astype(float)
        sub = f"superseded solver residual, Pilot 2 {grp} training-time checkpoints"
        add(b5, sub, int(q), "checkpoints", int(len(g)), "count", ck_src)
        add(b5, sub, int(q), "fraction with |BR(e~_1) - e~_1| > 1e-3", float((r > 1e-3).mean()), "fraction", ck_src)
        add(b5, sub, int(q), "median |BR(e~_1) - e~_1|", float(r.median()), "effort units (raw)", ck_src)
        add(b5, sub, int(q), "max |BR(e~_1) - e~_1|", float(r.max()), "effort units (raw)", ck_src)
    rel = ck2["induced_residual"].astype(float) / ck2["q"].map(g1)
    add(b5, "superseded solver residual, all 1,260 Pilot 2 checkpoints", "all", "max |BR(e~_1) - e~_1| / e_1*",
        float(rel.max()), "fraction of e_1*(0)", ck_src)
    add(b5, "superseded solver residual, all 1,260 Pilot 2 checkpoints", "all",
        "checkpoints with two sign changes of h (first root reported)",
        f"{int((ck2['induced_n_sign_changes'] >= 2).sum())}/{len(ck2)}", "count", ck_src)
    add(b5, "revised band widths, Pilot 2 A_joint exports", "all", "max band width / e_1*",
        float(((sets[1][1]["band_hi"] - sets[1][1]["band_lo"]) / sets[1][1]["g1"]).max()), "fraction of e_1*(0)",
        f"{IB}/joint_export_bands.csv")
    # Appendix S.1 (in-session script, no data file)
    s1t = _md_tables(RPT_P2, ["q", "Tier", "relative"], heading_has="S.1")
    if s1t:
        hdr = s1t[0]["header"]
        for r in s1t[0]["rows"]:
            rec = dict(zip(hdr, r))
            for col in hdr[2:]:
                v = C.parse_num(rec[col])
                add("5 comparison: superseded solver calibration (Appendix S.1)", f"tier {rec['Tier']}", int(rec["q"]),
                    col, v if v is not None else rec[col], "as in the report column",
                    f"{RPT_P2} Appendix S.1 (source: report text; in-session script, no data file)")
    else:  # pragma: no cover
        pack.unknown_value("T48", "Appendix S.1 calibration of the superseded solver", "table not found in the report")

    df = pd.DataFrame(rows)
    sources = [C.src(f"{IB}/floors.json"), C.src(f"{IB}/calibration.csv"), *set_srcs,
               C.src(f"{LK}/rehearsal_v1_1_R1_details.csv"), C.src(f"{P2A}/decomposition_residual_band.csv"),
               C.src(f"{P3A}/decomposition_residual_band.csv"), C.src(f"{P4A}/candidates_all.csv"),
               C.src(f"{P2A}/final_table.csv"), ck2_src, C.src(RPT_P2), C.src(RPT_P3), C.src(RPT_P4),
               C.src("tools/v2/induced_band.py"), C.src("tools/v2/decomposition.py"), C.src("utils/v2_metrics.py"),
               C.src(PROTO)]
    pack.table("T48", df, status="generated", sources=sources, script=f"{MOD}:build_t48", tier="final and development",
               notes=("Long format (block, subset, q, quantity, value). Block 1 compiled from code and report text (source: "
                      "report text where marked); blocks 2-4 computed from the band CSVs, the per-run induced_band.json "
                      "and the decomposition rows; block 5 medians at u1000 over 10 seeds per (q, arm) (numpy median); "
                      "superseded-solver residuals from the 60 Pilot 2 v2_checkpoints.csv; Appendix S.1 copied from the "
                      "report (source: report text). Band statistics: width = (band_hi - band_lo)/e_1*; medians numpy. "
                      "Residual-band values are final tier; superseded-solver values are development tier."),
               caption="Induced stage-1 target by residual minimisation (decision D2): definition, calibration gate, "
                       "band statistics, sweep coverage and comparison with the superseded bracketing + Brent solver.",
               docs={"block": D_BLOCK, "subset": "Set of bands / rows / runs the row refers to", "q": D_Q_ALL,
                     "quantity": D_QUANT, "value": D_VALUE, "unit": D_UNIT, "source": D_SRC})
    _t48_checks(pack, cal, sets, d2, d3, fin, comp, ck2, rel, cand, g1)


def _t48_checks(pack: C.Pack, cal: pd.DataFrame, sets, d2: pd.DataFrame, d3: pd.DataFrame, fin: pd.DataFrame,
                comp: List[Dict[str, Any]], ck2: pd.DataFrame, rel: pd.Series, cand: pd.DataFrame,
                g1: Dict[int, float]) -> None:
    """Cross-checks of T48 against pilot2_freeze.md section 6, pilot3/pilot4/lock reports and summary.md."""
    pack.crosscheck("T48", cal, RPT_P2, header_has=["q", "Width / e₁*"], key_map={"q": "q"},
                    value_map={"Width / e₁*": "band_width_over_e1star"}, heading_has="Calibration gate",
                    label="calibration gate band width")
    # summary per (q, arm) at u1000
    sm = []
    for (q, arm), g in fin.groupby(["q", "arm"], sort=True):
        sm.append({"q": int(q), "arm": ARM_SHORT_P2[arm], "median total": float(np.median(g["total_rel"])),
                   "median learning": float(np.median(g["learning_rel"])),
                   "median abs learning": float(np.median(g["abs_l"])),
                   "lc0": int(_b(g["learning_contains_0"]).sum()),
                   "median inherited": float(np.median(g["inherited_rel"])),
                   "median abs inherited": float(np.median(g["abs_i"])),
                   "ic0": int(_b(g["inherited_contains_0"]).sum())})
    sm = pd.DataFrame(sm)
    pack.crosscheck("T48", sm, RPT_P2, header_has=["q", "arm", "median total", "median abs learning"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={"median total": "median total", "median learning": "median learning",
                               "median abs learning": "median abs learning",
                               "learning band contains 0 (of 10)": "lc0", "median inherited": "median inherited",
                               "median abs inherited": "median abs inherited",
                               "inherited band contains 0 (of 10)": "ic0"},
                    label="revised decomposition medians per (q, arm) at u1000")
    allrows = []
    for arm, g in d2.groupby("arm", sort=True):
        allrows.append({"arm": arm, "rows": int(len(g)), "lc": float(_b(g["learning_contains_0"]).mean()),
                        "ic": float(_b(g["inherited_contains_0"]).mean()),
                        "out": int((~_b(g["e1_inside_sweep"])).sum())})
    pack.crosscheck("T48", pd.DataFrame(allrows), RPT_P2, header_has=["arm", "rows", "e1_outside_sweep"],
                    key_map={"arm": "arm"}, value_map={"rows": "rows", "learning_band_contains0_frac": "lc",
                                                       "inherited_band_contains0_frac": "ic", "e1_outside_sweep": "out"},
                    label="all decomposition rows per arm (Pilot 2)")
    cp = pd.DataFrame(comp)
    pack.crosscheck("T48", cp, RPT_P2, header_has=["q", "arm", "superseded median abs learning"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={"median abs learning": "revised_median_abs_learning",
                               "median inherited": "revised_median_inherited",
                               "median abs inherited": "revised_median_abs_inherited",
                               "superseded median abs learning": "superseded_median_abs_learning",
                               "superseded median inherited": "superseded_median_inherited",
                               "superseded median abs inherited": "superseded_median_abs_inherited"},
                    label="revised vs superseded medians at u1000")
    pr = fin.copy()
    pr["arm"] = pr["arm"].map(ARM_SHORT_P2)
    pack.crosscheck("T48", pr, RPT_P2, header_has=["q", "seed", "arm", "total_rel", "learning_rel", "inherited_rel"],
                    key_map={"q": "q", "seed": "seed", "arm": "arm"},
                    value_map={"total_rel": "total_rel", "learning_rel": "learning_rel", "inherited_rel": "inherited_rel"},
                    heading_has="Final checkpoint (update 1000", label="per-run decomposition at u1000 (60 runs)")

    ck = _Checks(pack, "T48", RPT_P2, "band diagnostics and superseded-solver residual (prose)")
    for name_idx, label in ((0, "Parents"), (1, "A_joint weight exports")):
        st = _band_stats(sets[name_idx][1])
        where = f"section 6 'Band diagnostics', {label}"
        txt = _text(RPT_P2)
        line = next(ln for ln in txt.splitlines() if ln.startswith(f"- **{label}"))
        nums = re.findall(r"n=(\d+); band width/e1\* median ([\d.]+), max ([\d.]+); non-contiguous (\d+); argmin at "
                          r"sweep edge (\d+); max Delta1_min/DW ([\d.e-]+)", line)[0]
        for qty, x, cell in (("n bands", st["n_bands"], nums[0]), ("median width/e1*", st["band_width_over_e1star_median"],
                                                                     nums[1]),
                             ("max width/e1*", st["band_width_over_e1star_max"], nums[2]),
                             ("non-contiguous", st["n_noncontiguous"], nums[3]),
                             ("argmin at sweep edge", st["n_argmin_at_sweep_edge"], nums[4]),
                             ("max Delta1_min/DW", st["delta1_min_over_dw_max"], nums[5])):
            ck.num(f"{label}: {qty}", x, cell, where)
    for q, grp, frac_cell, med_cell, max_cell in ((60, "A_joint", "100", "0.19", "0.29"),
                                                   (60, "B1_frozen_allnorm + B2_frozen_s1norm", "90", "0.11", None),
                                                   (50, "A_joint", "1", None, None),
                                                   (50, "B1_frozen_allnorm + B2_frozen_s1norm", "10", None, None)):
        r = ck2[(ck2["q"] == q) & (ck2["grp"] == grp)]["induced_residual"].astype(float)
        where = "section 3 anomaly 2"
        ck.num(f"q{q} {grp}: % checkpoints with residual > 1e-3", 100.0 * float((r > 1e-3).mean()), frac_cell, where)
        if med_cell:
            ck.num(f"q{q} {grp}: median residual", float(r.median()), med_cell, where)
        if max_cell:
            ck.num(f"q{q} {grp}: max residual", float(r.max()), max_cell, where)
    ck.num("max residual as % of e1*", 100.0 * float(rel.max()), "0.74", "section 3 anomaly 2")
    ck.num("checkpoints with two sign changes", int((ck2["induced_n_sign_changes"] >= 2).sum()), "84",
           "section 3 anomaly 2")
    ck.num("checkpoints in total", len(ck2), "1260", "section 3 anomaly 2 ('84 of 1,260')")
    ck.num("max joint band width as % of e1*",
           100.0 * float(((sets[1][1]["band_hi"] - sets[1][1]["band_lo"]) / sets[1][1]["g1"]).max()), "0.81",
           "section 6 'Does any Pilot 2 statement change?'")
    o2 = d2[~_b(d2["e1_inside_sweep"])]
    ck.num("Pilot 2 rows outside the sweep", len(o2), "7", "section 6, after the all-rows table")
    srcs = sorted(o2["source"].unique())
    if srcs != ["checkpoint", "weights"]:
        ck.differ("sources of the 7 out-of-sweep rows", "; ".join(f"{s}: {int((o2['source'] == s).sum())}" for s in srcs),
                  "early B1/B2 exports and checkpoints", "section 6, after the all-rows table",
                  "all 7 rows are 25-update weight exports at u425/u450 (column source == weights); no training-time "
                  "checkpoint row lies outside the sweep")
    ck.close()

    ck3 = _Checks(pack, "T48", RPT_P3, "Pilot 3 sweep coverage and non-contiguous parent bands (prose)")
    o3 = d3[~_b(d3["e1_inside_sweep"])]
    ck3.num("Pilot 3 rows outside the sweep", len(o3), "6", "section 8 anomaly 1")
    ck3.num("Pilot 3 decomposition rows", len(d3), "1800", "section 8 anomaly 1 ('6 of 1,800')")
    ck3.num("max e_hat_1/e1* among outside rows", float((o3["e1_at_0"] / o3["q"].map(g1)).max()), "1.92",
            "section 8 anomaly 1")
    pb = sets[0][1]
    nonc = sorted(int(s) for s in pb[(pb["q"] == 50) & (~_b(pb["band_contiguous"]))]["seed"])
    ck3.same("non-contiguous q=50 parent bands (seeds)", nonc, [10505, 10508, 10509, 10510], "section 8 anomaly 2")
    ck3.close()

    ck4 = _Checks(pack, "T48", RPT_P4, "Pilot 4 2b parent bands (prose)")
    p4 = sets[2][1]
    c2b = cand[cand["family"] == "2b"]
    for q, lo_c, hi_c in ((50, "46.12", "47.30"), (60, "38.45", "39.03")):
        g = p4[p4["q"] == q]
        ck4.num(f"q{q} min e~1", float(g["e_tilde"].min()), lo_c, "section 2b, induced-target bands")
        ck4.num(f"q{q} max e~1", float(g["e_tilde"].max()), hi_c, "section 2b, induced-target bands")
    ck4.num("q50 parent4 bands not contiguous", int((~_b(p4[p4["q"] == 50]["band_contiguous"])).sum()), "7",
            "section 2b, induced-target bands")
    ck4.num("2b export rows outside the sweep", int((~_b(c2b["e1_inside_sweep"])).sum()), "0",
            "section 2b, induced-target bands")
    ck4.num("2b rows checked for sweep coverage", len(c2b), "1080", "section 2b ('0 of 1,080 export rows')")
    for q, cell in ((50, "1.834"), (60, "1.959")):
        g = p4[p4["q"] == q]
        ck4.num(f"q{q} max sweep_hi / e1*", float(g["sweep_hi"].max()) / g1[q], cell, "section 2b, ranges used")
    tail = c2b[(c2b["kind"] == "tail") & (c2b["K"] == 1)]
    for q, cell in ((50, "-0.0030"), (60, "-0.0033")):
        ck4.num(f"q{q} median inherited term (2b, K=1)", float(np.median(tail[tail["q"] == q]["inherited_rel"])), cell,
                "section 2b, induced-target bands")
    ck4.close()

    ckl = _Checks(pack, "T48", RPT_LOCK, "v1.0 rehearsal bands (prose)")
    reh = sets[3][1]
    ckl.num("q50 rehearsal bands not contiguous", int((~_b(reh[reh["q"] == 50]["band_contiguous"])).sum()), "5",
            "section 4.4 notes")
    ckl.num("rehearsal runs with e_hat_1 inside the sweep", int(_b(reh["e1_inside_sweep"]).sum()), "20",
            "section 4.4 notes")
    ckl.close()

    cks = _Checks(pack, "T48", RPT_SUM, "open question 5 (prose)")
    jb = sets[1][1]
    cks.num("non-contiguous A_joint export bands", int((~_b(jb["band_contiguous"])).sum()), "167",
            "open question 5")
    cks.num("A_joint export bands", len(jb), "480", "open question 5")
    cks.num("early rows outside the parent sweep (Pilot 2 + Pilot 3)", len(o2) + len(o3), "13", "open question 5")
    cks.close()


# ----------------------------------------------------------------------------------------------
# F22 Delta_1(e; e_hat_2) against e
# ----------------------------------------------------------------------------------------------

def _median_run(pr: pd.DataFrame, q: int) -> Tuple[int, float, pd.DataFrame]:
    """Seed of the run with the median |stage-1 error| (n even: the two middle runs tie -> lower seed)."""
    g = pr[pr["q"] == q].sort_values(["s1", "seed"]).reset_index(drop=True)
    n = len(g)
    med = float(np.median(g["s1"]))
    if n % 2 == 1:
        mids = g.iloc[[n // 2]]
    else:
        mids = g.iloc[[n // 2 - 1, n // 2]]
    return int(mids["seed"].min()), med, mids


def build_f22(pack: C.Pack) -> None:
    """Delta_1/DW against the root effort for the median-|stage-1 error| confirmation run per q."""
    from utils.v2_metrics import induced_band

    dw = _dw()
    pr = _csv(f"{CA}/per_run.csv")
    rm = _csv(f"{CA}/reported_metrics.csv")
    fl = _js(f"{IB}/floors.json")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = style.new_figure(3, 2, height=8.9)
    handles: List[Any] = []
    data, checks, sel_txt, srcs = [], [], [], [C.src(f"{CA}/per_run.csv"), C.src(f"{CA}/reported_metrics.csv"),
                                               C.src(f"{IB}/floors.json")]
    max_d_plot, max_d_band, max_d_rm = 0.0, 0.0, 0.0
    for j, q in enumerate(QS):
        seed, med, mids = _median_run(pr, q)
        rd = f"{LK}/confirmation/q{q}/seed{seed}"
        srcs += [C.src(f"{rd}/band_sweep.npz"), C.src(f"{rd}/induced_band.json")]
        z = np.load(C.abspath(f"{rd}/band_sweep.npz"))
        E, D = np.asarray(z["e_sweep"], float), np.asarray(z["delta1"], float)
        ib = _js(f"{rd}/induced_band.json")
        rec = rm[(rm["q"] == q) & (rm["seed"] == seed)].iloc[0]
        b = induced_band(E, D, float(ib["floor"]))
        max_d_band = max(max_d_band, max(abs(b[k] - ib[k]) for k in ("e_tilde", "band_lo", "band_hi", "delta1_min")),
                         float(b["band_n_points"] != ib["band_n_points"]),
                         float(b["band_contiguous"] != ib["band_contiguous"]))
        max_d_rm = max(max_d_rm, abs(ib["e_tilde"] - rec["dec_e_tilde"]), abs(ib["band_lo"] - rec["dec_band_lo"]),
                       abs(ib["band_hi"] - rec["dec_band_hi"]), abs(ib["e1_at_0"] - rec["B_e1_at_0"]),
                       abs(ib["g1"] - float(fl[str(q)]["g1"])))
        y = D / dw
        thr = (float(ib["delta1_min"]) + float(ib["floor"])) / dw
        e1s, et, eh, blo, bhi = float(ib["g1"]), float(ib["e_tilde"]), float(ib["e1_at_0"]), float(ib["band_lo"]), \
            float(ib["band_hi"])
        inb = D <= float(ib["delta1_min"]) + float(ib["floor"])
        col, ls, mk = style.Q_COLORS[q], style.Q_LINESTYLE[q], style.Q_MARKER[q]
        lo_z = min(e1s, et, eh, blo) - 1.0
        hi_z = max(e1s, et, eh, bhi) + 1.0
        lo_b, hi_b = blo - 0.12, bhi + 0.12
        for i, (lo, hi) in enumerate(((E[0], E[-1]), (lo_z, hi_z), (lo_b, hi_b))):
            ax = axes[i, j]
            w = (E >= lo) & (E <= hi)
            if i < 2:
                ax.plot(E[w], y[w], color=col, ls=ls, lw=style.LINE_W)
            else:
                ax.plot(E[w], y[w], color=col, ls=ls, lw=style.THIN_W)
                ax.plot(E[w & inb], y[w & inb], ls="none", marker=mk, ms=3.5, color=col)
                ax.plot(E[w & ~inb], y[w & ~inb], ls="none", marker=mk, ms=3.5, mfc="white", mec=col)
                ax.axhline(thr, color=style.THRESH, ls="--", lw=1.0)
                ax.text(0.5 * (blo + bhi), 0.9 * max(float(y[w].max()), thr) * 1.15,
                        f"threshold {thr:.3g} ΔW", ha="center", va="top", fontsize=8, color=style.THRESH,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.0))
            ax.axvspan(blo, bhi, color=style.INK2, alpha=0.18, lw=0)
            if lo <= e1s <= hi:
                ax.axvline(e1s, color=style.REF, ls="-", lw=1.0)
            if lo <= et <= hi:
                ax.axvline(et, color=style.INK2, ls="--", lw=1.0)
            if lo <= eh <= hi:
                ax.axvline(eh, color=style.REF, ls="-.", lw=1.0)
            ax.set_xlim(lo, hi)
            if i == 1:
                top = float(y[w].max()) * 1.08
                ax.set_ylim(bottom=-0.03 * top, top=top)
            if i == 2:
                top = max(float(y[w].max()), thr) * 1.15
                ax.set_ylim(bottom=-0.05 * top, top=top)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            ax.set_xlabel("root effort e (effort units, raw)")
            ax.set_ylabel("Δ₁(e; ê₂) / ΔW")
            if i == 0:
                ax.set_title(f"q = {q}: seed {seed}, full sweep")
            elif i == 1:
                ax.set_title(f"q = {q}: window around e₁*, ẽ₁, ê₁")
            else:
                ax.set_title(f"q = {q}: band detail")
        handles.append(Line2D([], [], color=col, ls=ls, lw=style.LINE_W,
                              label=style.label_n(f"Δ₁/ΔW, q={q}, seed {seed}", 1)))
        max_d_plot = max(max_d_plot, C.max_abs_diff(y * dw, D))
        for e_, v_, b_ in zip(E, y, inb):
            data.append({"q": q, "seed": seed, "series": "delta1_over_dw", "e": float(e_), "delta1_over_dw": float(v_),
                         "in_band": bool(b_)})
        for name, v in (("e1_star", e1s), ("e_tilde", et), ("e_hat_1", eh), ("band_lo", blo), ("band_hi", bhi)):
            data.append({"q": q, "seed": seed, "series": name, "e": v, "delta1_over_dw": None, "in_band": None})
        data.append({"q": q, "seed": seed, "series": "band_threshold", "e": None, "delta1_over_dw": thr, "in_band": None})
        others = ", ".join(f"seed {int(r.seed)} (|err| {r.s1:.5f})" for r in mids.itertuples(index=False))
        sel_txt.append(f"q={q}: median |stage-1 error| {med:.5f}; middle runs {others}; seed {seed} taken")
    handles += [Line2D([], [], color=style.REF, ls="-", lw=1.0, label="e₁*(0)"),
                Line2D([], [], color=style.INK2, ls="--", lw=1.0, label="ẽ₁ (argmin of Δ₁)"),
                Line2D([], [], color=style.REF, ls="-.", lw=1.0, label="ê₁(0) of the run"),
                Patch(facecolor=style.INK2, alpha=0.18, label="band [band_lo, band_hi]"),
                Line2D([], [], color=style.INK2, marker="o", ls="none", ms=4, label="sweep node in the band"),
                Line2D([], [], color=style.INK2, marker="o", mfc="white", ls="none", ms=4,
                       label="sweep node outside the band"),
                Line2D([], [], color=style.THRESH, ls="--", lw=1.0, label="band threshold (Δ₁,min + floor)/ΔW")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=8, handlelength=1.8)
    checks += [f"band recomputed from band_sweep.npz with utils.v2_metrics.induced_band and the floor of "
               f"induced_band.json equals induced_band.json (e_tilde, band_lo, band_hi, delta1_min, band points, "
               f"contiguity; max abs diff {max_d_band:g})",
               f"induced_band.json e_tilde/band_lo/band_hi/e1_at_0 equal reported_metrics.csv dec_e_tilde/dec_band_lo/"
               f"dec_band_hi/B_e1_at_0 and g1 equals floors.json (max abs diff {max_d_rm:g})",
               f"plotted Delta_1/DW = delta1 / DW from the npz (max abs diff {max_d_plot:g})"]
    cap = ("Δ₁(e; ê₂)/ΔW, the verifier's root residual with both players' root action set to e and the "
           "run's frozen stage-2 mapping as continuation (final tier), against the root effort e (raw effort units), "
           "for one confirmation run per q (left q=50, right q=60), from the run's band_sweep.npz (sweep step 0.01 "
           "anchored at e₁*(0); range [0.4 e₁*, 1.7 e₁*]). Run selection: the run with the median "
           "|stage-1 error| (per_run.csv column s1) among the 20 runs of the q; n = 20 is even, so the two middle runs "
           "are equidistant from the median and the lower seed is taken (" + "; ".join(sel_txt) + "). Top row: full "
           "sweep; middle row: window e₁*, ẽ₁, ê₁ ± 1; bottom row: band ± 0.12 with the "
           "sweep nodes (filled = in the band) and the band threshold (Δ₁,min + floor)/ΔW (dashed). Marks: "
           "e₁*(0) (solid), ẽ₁ = argmin (dashed), ê₁(0) (dash-dot), band [band_lo, band_hi] "
           "(shaded). Final tier; n = 1 run per column.")
    docs = {"q": "q of the confirmation run", "seed": "Seed of the selected confirmation run",
            "series": "delta1_over_dw = sweep point of the curve; e1_star / e_tilde / e_hat_1 / band_lo / band_hi = "
                      "vertical marks (value in e); band_threshold = horizontal threshold (value in delta1_over_dw)",
            "e": {"definition": "Root effort e of the sweep point, or position of the vertical mark",
                  "units": "effort units (raw)"},
            "delta1_over_dw": {"definition": "Delta_1(e; e_hat_2) / Delta W at the sweep point (or the band threshold "
                                             "(Delta_1,min + floor)/Delta W)", "units": "Delta W (dimensionless)",
                               "normalization": "divided by Delta W = 4", "tier": "final",
                               "source": "band_sweep.npz (utils/v2_metrics.py:stage1_residual_sweep)"},
            "in_band": "Whether the sweep point belongs to the band {e: Delta_1 <= Delta_1,min + floor}"}
    pack.figure("F22", fig, pd.DataFrame(data), status="generated", sources=srcs, script=f"{MOD}:build_f22",
                caption=cap, tier="final", checks=checks, docs=docs,
                notes="Plotted directly from band_sweep.npz; marks from induced_band.json (equal to reported_metrics.csv).")


# ----------------------------------------------------------------------------------------------
# T49 root stage game
# ----------------------------------------------------------------------------------------------

OC_COLS = ["e1_star", "BR_at_e1star", "BR_minus_e1star", "grid_argmax", "own_curv_2a", "own_curv_over_2k",
           "ev2pp_over_2k", "slope_implied_by_curv", "ref_ev2pp_over_2k", "ref_own_curv_over_2k"]
BS_COLS = ["BR_plus", "BR_minus", "slope", "ref_slope"]


def _t49_summary(oc: pd.DataFrame, bs: pd.DataFrame) -> pd.DataFrame:
    """Range of every root-game quantity over fit widths (and h) per q and tier, with the reference."""
    rows = []
    for (q, tier), g in bs.groupby(["q", "tier"], sort=True):
        go = oc[(oc["q"] == q) & (oc["tier"] == tier)]
        ref_s = float(g["ref_slope"].iloc[0])
        items = [("dBR/de_opp, all fit widths W and steps h", g["slope"], ref_s, "PI reference (ref_slope)"),
                 ("dBR/de_opp, h <= 1", g[g["h"] <= 1.0]["slope"], ref_s, "PI reference (ref_slope)"),
                 ("dBR/de_opp, h = 1, W = 2", g[(g["h"] == 1.0) & (g["fit_half_width"] == 2.0)]["slope"], ref_s,
                  "PI reference (ref_slope)"),
                 ("own curvature 2a/(2k)", go["own_curv_over_2k"], float(go["ref_own_curv_over_2k"].iloc[0]),
                  "ref_own_curv_over_2k = ref E[V2'']/(2k) - 1"),
                 ("E[V2'']/(2k) = 1 + 2a/(2k)", go["ev2pp_over_2k"], float(go["ref_ev2pp_over_2k"].iloc[0]),
                  "PI reference (ref_ev2pp_over_2k)"),
                 ("slope implied by the curvature, -r/(1-r) with r = E[V2'']/(2k)", go["slope_implied_by_curv"],
                  ref_s, "PI reference slope (ref_slope)"),
                 ("BR(e1*) - e1* (fixed-point check)", go["BR_minus_e1star"], 0.0, "0 at the fixed point e1*")]
        for name, vals, ref, rsrc in items:
            v = vals.astype(float).to_numpy()
            rows.append({"block": "summary", "q": int(q), "tier": tier, "quantity": name, "n": int(v.size),
                         "min": float(v.min()), "max": float(v.max()), "median": float(np.median(v)), "reference": ref,
                         "reference_in_range": bool(v.min() <= ref <= v.max()), "reference_source": rsrc})
    return pd.DataFrame(rows)


def build_t49(pack: C.Pack) -> None:
    """Root stage game: numeric BR slope and own curvature against the PI references (Pilot 4 section 1a)."""
    oc = _csv(f"{RG}/own_curvature.csv")
    bs = _csv(f"{RG}/br_slope.csv")
    a = oc.copy()
    a.insert(0, "block", "per fit: own curvature (own_curvature.csv)")
    b = bs.copy()
    b.insert(0, "block", "per fit: BR slope (br_slope.csv)")
    sm = _t49_summary(oc, bs)
    cols = ["block", "q", "tier", "fit_half_width", "fit_points", "h"] + OC_COLS + BS_COLS + \
           ["quantity", "n", "min", "max", "median", "reference", "reference_in_range", "reference_source"]
    df = _union([sm, a, b], cols)
    tdoc = "Verifier tier of the fit: final (state 2, effort 0.5, GL 32) or fine (state 0.5, effort 0.25, GL 64; diagnostic)"
    src = "tools/v2/pilot4_root_game.py"
    docs = {
        "block": D_BLOCK, "q": "q (50 or 60)", "tier": tdoc,
        "fit_half_width": {"definition": "Half-width W of the least-squares quadratic fit of Q_1 around the grid argmax",
                           "units": "effort units", "source": src},
        "fit_points": {"definition": "Effort-grid nodes in the fit window", "units": "count", "source": src},
        "h": {"definition": "Finite-difference step of the slope (BR(e1*+h) - BR(e1*-h))/(2h)", "units": "effort units",
              "source": src},
        "e1_star": {"definition": "e_1*(0), the closed-form stage-1 effort", "units": "effort units (raw)"},
        "BR_at_e1star": {"definition": "Best response BR(e_opp = e1*): vertex of the quadratic fit of the verifier's Q_1",
                         "units": "effort units (raw)", "source": src},
        "BR_minus_e1star": {"definition": "BR(e1*) - e1* (0 at the fixed point)", "units": "effort units", "source": src},
        "grid_argmax": {"definition": "Grid argmax of Q_1(0, e | e_opp = e1*) on the tier's effort grid",
                        "units": "effort units (raw)", "source": src},
        "own_curv_2a": {"definition": "Own curvature d2Q_1/de^2 = 2a of the quadratic fit at e_opp = e1*",
                        "units": "value units per effort unit^2", "source": src},
        "own_curv_over_2k": {"definition": "Own curvature 2a / (2k)", "units": "dimensionless", "source": src},
        "ev2pp_over_2k": {"definition": "Implied E[V_2'']/(2k) = 1 + 2a/(2k)", "units": "dimensionless", "source": src},
        "slope_implied_by_curv": {"definition": "BR slope implied by the curvature, -r/(1-r) with r = E[V_2'']/(2k)",
                                  "units": "dimensionless", "source": src},
        "ref_ev2pp_over_2k": {"definition": "PI-side reference E[V_2'']/(2k) (0.490 at q=50, 0.236 at q=60)",
                              "units": "dimensionless", "source": "PI reference (request section 5.4); own_curvature.csv"},
        "ref_own_curv_over_2k": {"definition": "Reference own curvature /2k = reference E[V_2'']/(2k) - 1",
                                 "units": "dimensionless", "source": "own_curvature.csv"},
        "BR_plus": {"definition": "BR(e1* + h)", "units": "effort units (raw)", "source": src},
        "BR_minus": {"definition": "BR(e1* - h)", "units": "effort units (raw)", "source": src},
        "slope": {"definition": "Numeric root-game BR slope (BR(e1*+h) - BR(e1*-h))/(2h)", "units": "dimensionless",
                  "source": src},
        "ref_slope": {"definition": "PI-side reference BR slope (-0.961 at q=50, -0.309 at q=60)",
                      "units": "dimensionless", "source": "PI reference (request section 5.4); br_slope.csv"},
        "quantity": D_QUANT,
        "n": "Number of fits (W, h combinations) behind the summary row",
        "min": "Minimum over the fits of the summary row", "max": "Maximum over the fits of the summary row",
        "median": "Median over the fits of the summary row (numpy)",
        "reference": "Reference value of the quantity (PI reference, or 0 for the fixed-point check)",
        "reference_in_range": "Whether min <= reference <= max",
        "reference_source": "Where the reference comes from"}
    pack.table("T49", df, status="generated", sources=[C.src(f"{RG}/own_curvature.csv"), C.src(f"{RG}/br_slope.csv"),
                                                        C.src(src), C.src(RPT_P4)],
               script=f"{MOD}:build_t49", tier="final",
               notes=("Blocks 'per fit' are verbatim copies of own_curvature.csv (12 rows) and br_slope.csv (60 rows); "
                      "block 'summary' gives, per q and tier, n/min/max/median over the fit widths W (and steps h for the "
                      "slope) and whether the reference lies in [min, max]. The references in the CSVs equal the PI "
                      "references of the request (section 5.4): BR slope -0.961/-0.309, E[V2'']/(2k) 0.490/0.236. "
                      "Tier: final (requested) and fine (diagnostic finer tier), labelled in column tier."),
               caption="Root stage game at e_opp = e1*: numeric BR slope and own curvature from the verifier's stage-1 Q "
                       "with the closed-form continuation, against the PI references.", docs=docs)
    # cross-checks against pilot4 section 1a and summary.md
    pack.crosscheck("T49", oc, RPT_P4, header_has=["q", "tier", "fit_half_width", "own_curv_over_2k", "ev2pp_over_2k"],
                    key_map={"q": "q", "tier": "tier", "fit_half_width": "fit_half_width"},
                    value_map={"fit_points": "fit_points", "BR_minus_e1star": "BR_minus_e1star",
                               "own_curv_over_2k": "own_curv_over_2k", "ev2pp_over_2k": "ev2pp_over_2k",
                               "ref_ev2pp_over_2k": "ref_ev2pp_over_2k", "slope_implied_by_curv": "slope_implied_by_curv"},
                    label="own curvature per fit (section 1a)")
    piv = bs.pivot_table(index=["q", "tier", "fit_half_width"], columns="h", values="slope").reset_index()
    piv.columns = [c if not isinstance(c, float) else f"h={c:g}" for c in piv.columns]
    piv["reference"] = piv["q"].map(bs.groupby("q")["ref_slope"].first())
    pack.crosscheck("T49", piv, RPT_P4, header_has=["q", "tier", "fit_half_width", "h=0.25", "reference"],
                    key_map={"q": "q", "tier": "tier", "fit_half_width": "fit_half_width"},
                    value_map={"h=0.25": "h=0.25", "h=0.5": "h=0.5", "h=1": "h=1", "h=2": "h=2", "h=4": "h=4",
                               "reference": "reference"}, label="BR slope per fit and h (section 1a)")
    ck = _Checks(pack, "T49", RPT_P4, "section 1a comparison with the reference values (ranges)")
    t = _md_tables(RPT_P4, ["q", "quantity", "reference", "fine tier"])
    rng = {(r["q"], r["tier"], r["quantity"]): r for r in sm.to_dict("records")}
    qmap = {"E[V₂″]/(2k)": "E[V2'']/(2k) = 1 + 2a/(2k)", "own curvature /2k": "own curvature 2a/(2k)",
            "dBR/de_opp": "dBR/de_opp, all fit widths W and steps h"}
    for tb in t:
        for r in tb["rows"]:
            rec = dict(zip(tb["header"], r))
            qq, name = int(rec["q"]), qmap.get(rec["quantity"])
            if name is None:
                continue
            ck.num(f"q{qq} {name}: reference", rng[(qq, "final", name)]["reference"], rec["reference"],
                   "section 1a comparison table")
            for tier, col in (("final", tb["header"][3]), ("fine", "fine tier")):
                cell = rec[col]
                main = cell.split("(")[0]
                vals = re.findall(r"[−\-]?\d+\.\d+", main)
                st = rng[(qq, tier, name)]
                for v_str in vals:
                    x = min((st["min"], st["max"]), key=lambda v: abs(v - C.parse_num(v_str)))
                    ck.num(f"q{qq} {name} {tier}: range end", x, v_str, f"section 1a comparison table ({tier})",
                           comment="pack value = the end of the [min, max] range over the fits closest to the report "
                                   "number (root_game CSVs)")
                if "h ≤ 1" in cell:
                    st1 = rng[(qq, tier, "dBR/de_opp, h <= 1")]
                    for v_str in re.findall(r"[−\-]?\d+\.\d+", cell.split("h ≤ 1:")[1]):
                        x = min((st1["min"], st1["max"]), key=lambda v: abs(v - C.parse_num(v_str)))
                        ck.num(f"q{qq} dBR/de_opp h<=1 {tier}: range end", x, v_str,
                               f"section 1a comparison table ({tier}, h <= 1)")
    # prose: "q=60. The numeric values bracket the reference on both tiers."
    out = []
    for tier in ("final", "fine"):
        for name in ("E[V2'']/(2k) = 1 + 2a/(2k)", "own curvature 2a/(2k)", "dBR/de_opp, all fit widths W and steps h"):
            st = rng[(60, tier, name)]
            if not st["reference_in_range"]:
                out.append(f"{name.split(' =')[0]} {tier} [{st['min']:.4f}, {st['max']:.4f}] vs {st['reference']:g}")
    if out:
        ck.differ("q=60 numeric values vs reference: ranges that do not contain the reference", "; ".join(out),
                  "The numeric values bracket the reference on both tiers", "section 1a, bullet q=60",
                  "only the BR slope ranges contain the reference -0.309; the curvature ranges (E[V2'']/(2k) and own "
                  "curvature /2k) exclude 0.236 / -0.764 on both tiers (they are within 0.0053 of it on the fine tier)")
    fine60 = oc[(oc["q"] == 60) & (oc["tier"] == "fine")]
    ck.num("q=60 fine tier: max |E[V2'']/(2k) - 0.236|",
           float((fine60["ev2pp_over_2k"] - fine60["ref_ev2pp_over_2k"]).abs().max()), "0.005",
           "section 1a, bullet q=60 ('curvature is within 0.005')")
    bs60 = bs[(bs["q"] == 60) & (bs["tier"] == "fine")]
    mx = float((bs60["slope"] - bs60["ref_slope"]).abs().max())
    if mx > 0.02:
        ck.differ("q=60 fine tier: max |slope - reference|", mx, "within 0.02", "section 1a, bullet q=60", "")
    else:
        ck.n += 1
    mbr = float(oc["BR_minus_e1star"].abs().max())
    if mbr > 0.05:
        ck.differ("max |BR(e1*) - e1*| over all fits", mbr, "within +-0.05", "section 1a, fixed point", "")
    else:
        ck.n += 1
    ck.close()
    cks = _Checks(pack, "T49", RPT_SUM, "Pilot 4 bullet 1a (slope ranges)")
    for qq, lo_c, hi_c, ref_c in ((50, "-0.85", "-1.06", "-0.961"), (60, "-0.26", "-0.33", "-0.309")):
        st = rng[(qq, "final", "dBR/de_opp, all fit widths W and steps h")]
        cks.num(f"q{qq} slope range end (smallest |slope|)", st["max"], lo_c, "Pilot 4 bullet 1a")
        cks.num(f"q{qq} slope range end (largest |slope|)", st["min"], hi_c, "Pilot 4 bullet 1a")
        cks.num(f"q{qq} reference slope", st["reference"], ref_c, "Pilot 4 bullet 1a")
    cks.close()


# ----------------------------------------------------------------------------------------------
# T50 anatomy of the stage-1 fluctuation, F23 ACF plot
# ----------------------------------------------------------------------------------------------

def _acf_check_tables(pack: C.Pack, acf20: pd.DataFrame, acf25: pd.DataFrame) -> Dict[str, str]:
    """Compare the five ACF tables of pilot4 section 1b (cells 'median [q25, q75]') with the CSVs.

    The four every-20 tables carry no segment/kind label in the report; each is matched to the
    (segment_start, kind) combination whose values it reproduces.
    """
    ck = _Checks(pack, "T50", RPT_P4, "section 1b ACF tables (median [q25, q75], 2 decimals)")
    combos = [(s, k) for s in (650, 700) for k in ("centred", "about_e1star")]
    tabs = _md_tables(RPT_P4, ["q", "arm", "lag 20", "lag 160"], heading_has="1b")
    mapping: Dict[str, str] = {}

    def cells_of(tb, frame, seg, kind, lag_cols):
        out = []
        for r in tb["rows"]:
            rec = dict(zip(tb["header"], r))
            sseg = int(rec["segment_start"]) if "segment_start" in rec else seg
            for lc in lag_cols:
                lag = int(lc.split()[1])
                hit = frame[(frame["segment_start"] == sseg) & (frame["kind"] == kind) & (frame["q"] == int(rec["q"]))
                            & (frame["arm"] == rec["arm"]) & (frame["lag_updates"] == lag)]
                out.append((rec, lc, lag, hit, rec[lc]))
        return out

    def score(tb, frame, seg, kind, lag_cols):
        n_ok = 0
        for rec, lc, lag, hit, cell in cells_of(tb, frame, seg, kind, lag_cols):
            s = _bracket_strs(cell)
            if s is None or len(hit) != 1:
                continue
            h = hit.iloc[0]
            n_ok += sum(_cell_ok(float(h[c]), v) for c, v in zip(("median", "q25", "q75"), s))
        return n_ok

    for i, tb in enumerate(tabs):
        lag_cols = [c for c in tb["header"] if c.startswith("lag ")]
        best = max(combos, key=lambda sk: score(tb, acf20, sk[0], sk[1], lag_cols))
        mapping[f"every-20 table {i + 1} (report line {tb['line']})"] = f"segment u>={best[0]}, {best[1]}"
        for rec, lc, lag, hit, cell in cells_of(tb, acf20, best[0], best[1], lag_cols):
            s = _bracket_strs(cell)
            h = hit.iloc[0]
            for c, v in zip(("median", "q25", "q75"), s):
                ck.num(f"ACF {c}, u>={best[0]} {best[1]}, q{rec['q']} {rec['arm']} lag {lag}", float(h[c]), v,
                       f"line {tb['line']}")
    t25 = _md_tables(RPT_P4, ["segment_start", "q", "arm", "lag 25", "lag 100"], heading_has="1b")
    for tb in t25:
        lag_cols = [c for c in tb["header"] if c.startswith("lag ")]
        best_kind = max(("centred", "about_e1star"), key=lambda k: score(tb, acf25, 0, k, lag_cols))
        mapping[f"exports-25 table (report line {tb['line']})"] = f"both segments, {best_kind}"
        for rec, lc, lag, hit, cell in cells_of(tb, acf25, 0, best_kind, lag_cols):
            s = _bracket_strs(cell)
            h = hit.iloc[0]
            for c, v in zip(("median", "q25", "q75"), s):
                ck.num(f"ACF exports {c}, u>={rec['segment_start']} {best_kind}, q{rec['q']} {rec['arm']} lag {lag}",
                       float(h[c]), v, f"line {tb['line']}")
    ck.close()
    return mapping


def build_t50(pack: C.Pack) -> None:
    """ACF of the stage-1 fluctuation and the window regression (Pilot 4 section 1b, Pilot 3 runs)."""
    acf20 = _csv(f"{FL}/acf_summary_every20.csv")
    acf25 = _csv(f"{FL}/acf_summary_exports25.csv")
    pr20 = _csv(f"{FL}/acf_per_run_every20.csv")
    wr = _csv(f"{FL}/window_regression.csv")
    ser = _csv(f"{FL}/stability_series_e1.csv")
    ex = _csv(f"{FL}/export_series_e1.csv")
    win = _csv(f"{FL}/windows.csv")
    bs = _csv(f"{RG}/br_slope.csv")
    # recompute the every-20 summary from the per-run ACFs (numpy linear percentiles)
    rec = pr20.groupby(["segment_start", "kind", "q", "arm", "lag_updates"], sort=True)["acf"].agg(
        median=lambda s: float(np.median(s)), q25=lambda s: float(np.percentile(s, 25)),
        q75=lambda s: float(np.percentile(s, 75)), n_runs="size").reset_index()
    mm = acf20.merge(rec, on=["segment_start", "kind", "q", "arm", "lag_updates"], suffixes=("", "_re"))
    d_rec = max(C.max_abs_diff(mm[c], mm[c + "_re"]) for c in ("median", "q25", "q75"))
    rows_rec = []
    steps = sorted(set(np.diff(sorted(ser["update"].unique()))))
    n_pts = ser.groupby(["q", "seed", "arm"]).size()
    seg_n = pr20.groupby("segment_start")["n"].unique()
    for qty, val, unit in (
            ("series", "e_hat_1(0) (Beta mean) from the stability log of train_history.json, every 20 updates; no "
                       "per-update record of the root policy mean exists (history mean_effort_by_stage is the batch mean "
                       "of sampled efforts) (source: report text)", "text"),
            ("runs", int(len(n_pts)), "count (Pilot 3: 2 arms x 2 q x 10 seeds)"),
            ("points per run", "; ".join(str(v) for v in sorted(n_pts.unique())), "count"),
            ("first and last update", f"u{int(ser['update'].min())} .. u{int(ser['update'].max())}", "global update"),
            ("spacing", "; ".join(str(int(s)) for s in steps), "updates"),
            ("ACF segment u>=650: points per run", "; ".join(str(int(v)) for v in seg_n.loc[650]), "count (u660..u1000)"),
            ("ACF segment u>=700: points per run", "; ".join(str(int(v)) for v in seg_n.loc[700]), "count (u700..u1000)"),
            ("ACF definitions", "per run, x = e_hat_1(0) - e1* on the segment; 'centred' = sample ACF about the segment "
                                "mean (denominator n c0); 'about_e1star' = same with x not demeaned; summary = median and "
                                "q25/q75 over the 10 runs per (q, arm)", "text"),
            ("secondary series", f"25-update weight exports u{int(ex['update'].min())}..u{int(ex['update'].max())} "
                                 f"({int(ex.groupby(['q', 'seed', 'arm']).size().iloc[0])} per run), lags 25-100",
             "text"),
            ("window regression", "y = e_hat_1(u+20) - e_hat_1(u) on x = e_hat_1(u) - e1* (opponent refreshed to the actor "
                                  "at u), OLS with intercept pooled over runs per (q, arm) and per q; 95% CI by a cluster "
                                  "bootstrap over runs (10,000 resamples, numpy seed 20261001)", "text"),
            ("windows (all segments)", int(len(win)), "count"),
            ("recomputed every-20 summary from acf_per_run_every20.csv (numpy percentiles) vs acf_summary_every20.csv: "
             "max abs diff", d_rec, "ACF (dimensionless)")):
        rows_rec.append({"block": "record and definitions", "quantity": qty, "value": val, "unit": unit,
                         "source": f"{FL}/stability_series_e1.csv; tools/v2/pilot4_fluctuation.py; {RPT_P4} section 1b"})
    a20 = acf20.copy()
    a20.insert(0, "block", "ACF every 20 updates (acf_summary_every20.csv)")
    a25 = acf25.copy()
    a25.insert(0, "block", "ACF of the 25-update exports, secondary (acf_summary_exports25.csv)")
    w = wr.copy()
    w.insert(0, "block", "window regression (window_regression.csv)")
    comp = []
    for q in QS:
        p = wr[(wr["segment_start"] == 650) & (wr["q"] == q) & (wr["arm"] == "both")].iloc[0]
        fine = bs[(bs["q"] == q) & (bs["tier"] == "fine") & (bs["h"] == 1.0) & (bs["fit_half_width"] == 2.0)]["slope"]
        fin = bs[(bs["q"] == q) & (bs["tier"] == "final")]["slope"]
        for qty, val, lo, hi, src in (
                ("pooled window slope (u>=650, both arms)", p["slope"], p["boot_ci95_lo"], p["boot_ci95_hi"],
                 f"{FL}/window_regression.csv"),
                ("per-run median window slope (u>=650, both arms)", p["per_run_slope_median"], None, None,
                 f"{FL}/window_regression.csv"),
                ("root-game BR slope, PI reference", float(bs[bs["q"] == q]["ref_slope"].iloc[0]), None, None,
                 f"{RG}/br_slope.csv (ref_slope)"),
                ("root-game BR slope, fine tier, h = 1, W = 2", float(fine.iloc[0]), None, None, f"{RG}/br_slope.csv"),
                ("root-game BR slope, final tier, min over W and h", float(fin.min()), None, None, f"{RG}/br_slope.csv"),
                ("root-game BR slope, final tier, max over W and h", float(fin.max()), None, None, f"{RG}/br_slope.csv")):
            comp.append({"block": "comparison with the root-game BR slope", "q": q, "quantity": qty, "value": float(val),
                         "boot_ci95_lo": lo, "boot_ci95_hi": hi, "unit": "dimensionless", "source": src})
    cols = ["block", "segment_start", "kind", "q", "arm", "lag_updates", "median", "q25", "q75", "n_runs", "n_windows",
            "slope", "boot_ci95_lo", "boot_ci95_hi", "per_run_slope_median", "per_run_slope_q25", "per_run_slope_q75",
            "quantity", "value", "unit", "source"]
    df = _union([pd.DataFrame(rows_rec), a20, a25, w, pd.DataFrame(comp)], cols)
    src_fl = "tools/v2/pilot4_fluctuation.py"
    docs = {
        "block": D_BLOCK,
        "segment_start": {"definition": "Start of the segment: ACF on u >= 650 (u660..u1000, 18 points per run) or "
                                        "u >= 700 (u700..u1000, 16 points); window regression on windows starting at u >= "
                                        "segment_start", "units": "global update", "source": src_fl},
        "kind": {"definition": "ACF version: centred = sample ACF of x = e_hat_1(0) - e1* about the segment mean; "
                               "about_e1star = same with x not demeaned (deviations about e1*)", "source": src_fl},
        "q": "q (50 or 60)",
        "arm": "Pilot 3 arm: stochastic (B2_frozen_s1norm) or mean (B2_frozen_s1norm_mean); 'both' = arms pooled",
        "lag_updates": {"definition": "ACF lag", "units": "updates (20 per step for the stability log, 25 for exports)"},
        "median": {"definition": "Median over the runs of the per-run ACF at the lag", "units": "dimensionless",
                   "tier": "tier-independent (direct policy query)", "source": src_fl},
        "q25": {"definition": "25th percentile over runs of the per-run ACF (linear interpolation)",
                "units": "dimensionless", "source": src_fl},
        "q75": {"definition": "75th percentile over runs of the per-run ACF (linear interpolation)",
                "units": "dimensionless", "source": src_fl},
        "n_runs": "Number of runs (seeds, or seed x arm units for 'both') behind the row",
        "n_windows": "Number of 20-update windows in the pooled regression",
        "slope": {"definition": "OLS slope (with intercept) of y = e_hat_1(u+20) - e_hat_1(u) on x = e_hat_1(u) - e1*, "
                                "windows pooled over runs", "units": "dimensionless", "source": src_fl},
        "boot_ci95_lo": {"definition": "Lower end of the 95% cluster-bootstrap CI of the pooled slope (resampling runs; "
                                       "10,000 resamples, numpy seed 20261001)", "units": "dimensionless",
                         "source": src_fl},
        "boot_ci95_hi": {"definition": "Upper end of the 95% cluster-bootstrap CI of the pooled slope",
                         "units": "dimensionless", "source": src_fl},
        "per_run_slope_median": {"definition": "Median over runs of the per-run OLS slope", "units": "dimensionless"},
        "per_run_slope_q25": {"definition": "25th percentile over runs of the per-run OLS slope", "units": "dimensionless"},
        "per_run_slope_q75": {"definition": "75th percentile over runs of the per-run OLS slope", "units": "dimensionless"},
        "quantity": D_QUANT, "value": D_VALUE, "unit": D_UNIT, "source": D_SRC}
    srcs = [C.src(f"{FL}/acf_summary_every20.csv"), C.src(f"{FL}/acf_summary_exports25.csv"),
            C.src(f"{FL}/acf_per_run_every20.csv"), C.src(f"{FL}/window_regression.csv"),
            C.src(f"{FL}/stability_series_e1.csv"), C.src(f"{FL}/export_series_e1.csv"), C.src(f"{FL}/windows.csv"),
            C.src(f"{RG}/br_slope.csv"), C.src(src_fl), C.src(RPT_P4)]
    mapping = _acf_check_tables(pack, acf20, acf25)
    pack.table("T50", df, status="generated", sources=srcs, script=f"{MOD}:build_t50", tier="tier-independent",
               notes=("Blocks: record and definitions (from the series CSVs and the tool docstring; text marked source: "
                      "report text); verbatim copies of acf_summary_every20.csv (both versions, both segments, lags 20-160), "
                      "acf_summary_exports25.csv and window_regression.csv; comparison of the pooled/per-run window slopes "
                      "with the root-game BR slope (br_slope.csv). ACF summaries re-derived from acf_per_run_every20.csv "
                      f"with numpy percentiles: max abs diff {d_rec:g}. The four every-20 ACF tables of pilot4 section 1b "
                      "carry no segment/kind label; matched by value: " +
                      "; ".join(f"{k} = {v}" for k, v in mapping.items()) + ". Values are tier-independent (Beta-mean "
                      "policy queries)."),
               caption="Anatomy of the stage-1 fluctuation in the Pilot 3 runs (both arms): ACF of e_hat_1(0) - e1* from "
                       "the 20-update stability log, the 25-update export ACF, and the 20-update window regression.",
               docs=docs)
    pack.crosscheck("T50", wr, RPT_P4, header_has=["segment_start", "q", "arm", "n_windows", "slope"],
                    key_map={"segment_start": "segment_start", "q": "q", "arm": "arm"},
                    value_map={c: c for c in ("n_windows", "n_runs", "slope", "boot_ci95_lo", "boot_ci95_hi",
                                              "per_run_slope_median", "per_run_slope_q25", "per_run_slope_q75")},
                    label="window regression (section 1b)")
    _t50_prose(pack, acf20, wr, bs)


def _t50_prose(pack: C.Pack, acf20: pd.DataFrame, wr: pd.DataFrame, bs: pd.DataFrame) -> None:
    """Checks of the section 1b comparison table and summary bullets, and of summary.md bullet 1b."""
    ck = _Checks(pack, "T50", RPT_P4, "section 1b comparison table and summary bullets")
    for tb in _md_tables(RPT_P4, ["q", "per-run median slope"], heading_has="1b"):
        for r in tb["rows"]:
            rec = dict(zip(tb["header"], r))
            q = int(rec["q"])
            p = wr[(wr["segment_start"] == 650) & (wr["q"] == q) & (wr["arm"] == "both")].iloc[0]
            s = _bracket_strs(rec[tb["header"][1]])
            for c, v in zip(("slope", "boot_ci95_lo", "boot_ci95_hi"), s):
                ck.num(f"q{q} pooled window {c}", float(p[c]), v, "section 1b comparison table")
            ck.num(f"q{q} per-run median slope", float(p["per_run_slope_median"]), rec["per-run median slope"],
                   "section 1b comparison table")
            ref_c, fine_c = [x.strip() for x in rec[tb["header"][3]].split("/")]
            ck.num(f"q{q} BR slope reference", float(bs[bs["q"] == q]["ref_slope"].iloc[0]), ref_c,
                   "section 1b comparison table")
            fine = bs[(bs["q"] == q) & (bs["tier"] == "fine") & (bs["h"] == 1.0) & (bs["fit_half_width"] == 2.0)]
            ck.num(f"q{q} BR slope fine tier h=1 (W=2)", float(fine["slope"].iloc[0]), fine_c,
                   "section 1b comparison table (W not stated; W = 2 reproduces the value)")
    c650 = acf20[(acf20["segment_start"] == 650) & (acf20["kind"] == "centred")]
    l20 = c650[c650["lag_updates"] == 20]["median"]
    l40 = c650[c650["lag_updates"] == 40]["median"]
    ck.num("median lag-20 ACF, lowest (centred, u>=650)", float(l20.min()), "0.48", "section 1b summary")
    ck.num("median lag-20 ACF, highest", float(l20.max()), "0.60", "section 1b summary")
    ck.num("median lag-40 ACF, lowest", float(l40.min()), "0.23", "section 1b summary")
    ck.num("median lag-40 ACF, highest", float(l40.max()), "0.30", "section 1b summary")
    late = c650[c650["lag_updates"].between(100, 160)]
    lo_l, hi_l = float(late["median"].min()), float(late["median"].max())
    ck.num("median ACF at lags 100-160, most negative", lo_l, "-0.25", "section 1b summary")
    if hi_l >= 0:
        r_ = late.loc[late["median"].idxmax()]
        ck.differ("median ACF at lags 100-160 (centred, u>=650), largest value", hi_l,
                  "negative at lags 100-160: medians -0.06 to -0.25", "section 1b summary",
                  f"the largest median at lags 100-160 is +{hi_l:.4f} (q={int(r_['q'])} {r_['arm']}, lag "
                  f"{int(r_['lag_updates'])}), not negative; the other medians lie in [{lo_l:.3f}, "
                  f"{float(late[late['median'] < 0]['median'].max()):.3f}]")
    else:
        ck.num("median ACF at lags 100-160, least negative", hi_l, "-0.06", "section 1b summary")
    ck.close()
    cks = _Checks(pack, "T50", RPT_SUM, "Pilot 4 bullet 1b")
    cks.num("median lag-20 ACF, lowest", float(l20.min()), "0.48", "Pilot 4 bullet 1b")
    cks.num("median lag-20 ACF, highest", float(l20.max()), "0.60", "Pilot 4 bullet 1b")
    for q, cell in ((50, "-0.15"), (60, "-0.20")):
        p = wr[(wr["segment_start"] == 650) & (wr["q"] == q) & (wr["arm"] == "both")].iloc[0]
        cks.num(f"q{q} pooled window slope", float(p["slope"]), cell, "Pilot 4 bullet 1b")
    if hi_l >= 0:
        cks.differ("median ACF at lags 100-160 (centred, u>=650), largest value", hi_l, "negative at 100-160",
                   "Pilot 4 bullet 1b", "one median at lags 100-160 is positive (see the pilot4 section 1b record)")
    cks.close()


def build_f23(pack: C.Pack) -> None:
    """ACF of e_hat_1(0) - e1* against lag (median and IQR over runs), per q and Pilot-3 arm."""
    acf20 = _csv(f"{FL}/acf_summary_every20.csv")
    pr20 = _csv(f"{FL}/acf_per_run_every20.csv")
    seg = 650
    fig, axes = style.new_figure(2, 2, height=5.6, sharex=True, sharey="row")
    rows, dmax, dmax_re = [], 0.0, 0.0
    arm_mk = {"mean": "o", "stochastic": "^"}
    for i, kind in enumerate(("centred", "about_e1star")):
        for j, q in enumerate(QS):
            ax = axes[i, j]
            for arm in ("stochastic", "mean"):
                g = acf20[(acf20["segment_start"] == seg) & (acf20["kind"] == kind) & (acf20["q"] == q)
                          & (acf20["arm"] == arm)].sort_values("lag_updates")
                n = int(g["n_runs"].iloc[0])
                style.band_plot(ax, g["lag_updates"].to_numpy(), g["median"].to_numpy(), g["q25"].to_numpy(),
                                g["q75"].to_numpy(), color=style.ARM_COLORS[arm], marker=arm_mk[arm],
                                label=style.label_n(style.arm_label(arm), n))
                p = pr20[(pr20["segment_start"] == seg) & (pr20["kind"] == kind) & (pr20["q"] == q)
                         & (pr20["arm"] == arm)]
                re_ = p.groupby("lag_updates")["acf"].agg(lambda s: float(np.median(s))).reindex(g["lag_updates"])
                dmax_re = max(dmax_re, C.max_abs_diff(re_.to_numpy(), g["median"].to_numpy()))
                for r in g.itertuples(index=False):
                    rows.append({"segment_start": seg, "kind": kind, "q": q, "arm": arm, "lag_updates": int(r.lag_updates),
                                 "median": float(r.median), "q25": float(r.q25), "q75": float(r.q75),
                                 "n_runs": int(r.n_runs)})
            ax.axhline(0.0, color=style.REF, lw=0.8)
            ax.set_title(f"q = {q}, ACF {'centred on the segment mean' if kind == 'centred' else 'about e1* (not demeaned)'}")
            if i == 1:
                ax.set_xlabel("lag (updates)")
            if j == 0:
                ax.set_ylabel("ACF of ê₁(0) − e₁*")
            ax.set_xticks([20, 40, 60, 80, 100, 120, 140, 160])
            if i == 0 and j == 0:
                ax.legend(loc="upper right")
    data = pd.DataFrame(rows)
    chk = data.merge(acf20, on=["segment_start", "kind", "q", "arm", "lag_updates"], suffixes=("", "_src"))
    dmax = max(C.max_abs_diff(chk[c], chk[c + "_src"]) for c in ("median", "q25", "q75"))
    cap = ("Autocorrelation of x = ê₁(0) − e₁* against lag (updates) in the Pilot 3 runs: median over 10 "
           "runs per arm (line) and IQR (band), stochastic and mean continuation arms, segment u ≥ 650 "
           "(u660–u1000, 18 points per run from the 20-update stability log). Top row: ACF centred on the segment "
           "mean; bottom row: ACF about e₁* (x not demeaned). Columns: q = 50 and q = 60. Source: "
           "results/v2_pilots/pilot4/analysis/fluctuation/acf_summary_every20.csv. Tier-independent (Beta-mean policy "
           "queries); n = 10 runs per arm and q.")
    pack.figure("F23", fig, data, status="generated",
                sources=[C.src(f"{FL}/acf_summary_every20.csv"), C.src(f"{FL}/acf_per_run_every20.csv")],
                script=f"{MOD}:build_f23", caption=cap, tier="tier-independent",
                checks=[f"plotted median/q25/q75 equal acf_summary_every20.csv (max abs diff {dmax:g})",
                        f"plotted medians equal numpy medians of acf_per_run_every20.csv (max abs diff {dmax_re:g})"],
                notes="No ACF figure exists in reports/v2/figures/ (checked); generated from the existing summary CSV.",
                docs={"segment_start": "Segment start: ACF on u >= 650 (u660..u1000)",
                      "kind": "ACF version: centred (about the segment mean) or about_e1star (not demeaned)",
                      "q": "q (50 or 60)", "arm": "Pilot 3 arm: stochastic or mean continuation",
                      "lag_updates": {"definition": "ACF lag", "units": "updates"},
                      "median": {"definition": "Median over runs of the per-run ACF", "units": "dimensionless"},
                      "q25": {"definition": "25th percentile over runs of the per-run ACF", "units": "dimensionless"},
                      "q75": {"definition": "75th percentile over runs of the per-run ACF", "units": "dimensionless"},
                      "n_runs": "Number of runs behind the median"})


# ----------------------------------------------------------------------------------------------
# T51 Pilot 4 section 2b, F24 learning curves
# ----------------------------------------------------------------------------------------------

UNIT_REL_S = "fraction of e_1*(0), signed"
UNIT_REL = "fraction of e_1*(0)"
UNIT_DW = "Delta W (dimensionless)"
UNIT_RAW = "effort units (raw)"
TI = "tier-independent"

T51_FINAL = [  # (per-run column, metric name in the table, tier, unit)
    ("e1_cand", "e1_cand", TI, UNIT_RAW),
    ("stage1_rel_err_signed", "stage1_rel_err_signed", TI, UNIT_REL_S),
    ("stage1_rel_err_abs", "stage1_rel_err_abs", TI, UNIT_REL),
    ("learning_rel", "learning_rel", TI + " (final-tier band)", UNIT_REL_S),
    ("learning_rel_abs", "learning_rel_abs", TI + " (final-tier band)", UNIT_REL),
    ("inherited_rel", "inherited_rel", TI + " (final-tier band)", UNIT_REL_S),
    ("sigma_effort_at_0_t1", "sigma_effort_at_0_t1", TI, UNIT_RAW),
    ("within_run_sd_e1_last5", "within_run_sd_e1_last5", TI, UNIT_RAW),
    ("within_run_range_e1_last5", "within_run_range_e1_last5", TI, UNIT_RAW),
    ("Gmax_full_over_dw", "Gmax_full_over_dw_dev", "development", UNIT_DW),
    ("final_tier__Gmax_full_over_dw", "Gmax_full_over_dw_final", "final", UNIT_DW),
    ("EXP_root_over_dw", "EXP_root_over_dw_dev", "development", UNIT_DW),
    ("final_tier__EXP_root_over_dw", "EXP_root_over_dw_final", "final", UNIT_DW),
    ("dReach_over_dw", "dReach_over_dw_dev", "development", UNIT_DW),
    ("final_tier__dReach_over_dw", "dReach_over_dw_final", "final", UNIT_DW),
    ("Deltamax_all_over_dw", "Deltamax_all_over_dw_dev", "development", UNIT_DW),
    ("final_tier__Deltamax_all_over_dw", "Deltamax_all_over_dw_final", "final", UNIT_DW),
    ("dFull_over_dw", "dFull_over_dw_dev", "development", UNIT_DW),
    ("final_tier__dFull_over_dw", "dFull_over_dw_final", "final", UNIT_DW),
    ("kl_final_epoch", "kl_final_epoch", "n/a (training record)", "nats"),
    ("clip_frac", "clip_frac", "n/a (training record)", "fraction"),
    ("kl_median", "kl_median", "n/a (training record)", "nats"),
    ("clip_median", "clip_median", "n/a (training record)", "fraction"),
    ("phase_wall_sec", "phase_wall_sec", "n/a (training record)", "seconds"),
    ("would_fire_update", "would_fire_update", "n/a (training record)", "global update"),
]
TAIL_METRICS = [("e1_cand", TI, UNIT_RAW), ("stage1_rel_err_signed", TI, UNIT_REL_S), ("stage1_rel_err_abs", TI, UNIT_REL),
                ("learning_rel", TI, UNIT_REL_S), ("learning_rel_abs", TI, UNIT_REL), ("inherited_rel", TI, UNIT_REL_S),
                ("Gmax_full_over_dw", "development", UNIT_DW), ("EXP_root_over_dw", "development", UNIT_DW),
                ("dReach_over_dw", "development", UNIT_DW)]
PAIRED_COLS = ["n_pairs", "mean", "n_neg", "n_pos", "n_zero", "boot_ci95_lo", "boot_ci95_hi", "n_better", "better_if"]


def _metric_unit(m: str) -> str:
    """Unit of a Pilot 4 metric name (paired rows)."""
    if m.startswith(("stage1_rel_err", "learning_rel", "inherited_rel")):
        return UNIT_REL if m.endswith("abs") else UNIT_REL_S
    if m.endswith("_over_dw"):
        return UNIT_DW
    if m.startswith("kl_"):
        return "nats"
    if m.startswith("clip"):
        return "fraction"
    if m.endswith("wall_sec"):
        return "seconds"
    if m.startswith("within_run"):
        return UNIT_RAW
    return "as the metric"


def _k_of(label: str) -> Optional[int]:
    """K of a paired-summary arm label such as 'decay K=4'."""
    m = re.search(r"K=(\d+)", str(label))
    return int(m.group(1)) if m else None


def _p4b_per_run() -> pd.DataFrame:
    """Per-run Pilot 4 2b frame at u2200 (K = 1): candidates (dev tier), run records, final-tier scalars."""
    cand = _csv(f"{P4A}/candidates_all.csv")
    rr = _csv(f"{P4A}/run_records.csv")
    t = cand[(cand["family"] == "2b") & (cand["kind"] == "tail") & (cand["K"] == 1)].copy()
    r = rr[rr["family"] == "2b"][["q", "seed", "arm", "kl_final_epoch", "clip_frac", "kl_median", "clip_median",
                                  "phase_wall_sec", "would_fire_update", "within_run_sd_e1_last5",
                                  "within_run_range_e1_last5", "n_last5"]]
    ftc = S.final_tier_columns("pilot4_B")
    ftc["arm"] = ftc["arm"].map(ARM_P4B)
    keep = ["q", "seed", "arm", "run_dir"] + [c for c in ftc.columns if c.startswith("final_tier__")]
    per = t.merge(r, on=["q", "seed", "arm"], how="inner").merge(ftc[keep], on=["q", "seed", "arm"], how="inner")
    if len(per) != 40:
        raise ValueError(f"Pilot 4 2b per-run frame has {len(per)} rows, expected 40")
    return per.sort_values(["q", "arm", "seed"]).reset_index(drop=True)


def build_t51(pack: C.Pack) -> None:
    """Pilot 4 section 2b: final table at u2200, paired decay - constant, stability, stage-1 tail averaging."""
    cand = _csv(f"{P4A}/candidates_all.csv")
    rr = _csv(f"{P4A}/run_records.csv")
    ps = _csv(f"{P4A}/paired_summary.csv")
    psr = _csv(f"{P4A}/paired_summary_run_records.csv")
    st = _csv(f"{P4A}/stability_2b.csv")
    per = _p4b_per_run()
    blocks: List[pd.DataFrame] = []
    # 1 final table at u2200
    rows = []
    for (q, arm), g in per.groupby(["q", "arm"], sort=True):
        for col, name, tier, unit in T51_FINAL:
            rows.append({"block": "1 final checkpoint u2200 (K = 1, last iterate)", "family": "2b", "q": int(q),
                         "arm": arm, "K": 1, "metric": name, "tier": tier, "unit": unit,
                         **_stat(g[col].astype(float))})
        for tier, tcol, dcol in (("development", "Gmax_full_t", "Gmax_full_d"),
                                 ("final", "final_tier__Gmax_full_t", "final_tier__Gmax_full_d")):
            locs = g[[tcol, dcol]].astype(float)
            rows.append({"block": "1 final checkpoint u2200 (K = 1, last iterate)", "family": "2b", "q": int(q),
                         "arm": arm, "K": 1, "metric": f"Gmax_full location (t*, d*) counts, {tier} tier", "tier": tier,
                         "unit": "count (d* in effort units)",
                         "value": "; ".join(f"t*={int(t_)}, d*={_f(d_)}: {int(n_)}" for (t_, d_), n_ in
                                            locs.value_counts().sort_index().items()), "n": int(len(g))})
    blocks.append(pd.DataFrame(rows))
    # 2 run records
    rows = []
    r2 = rr[rr["family"] == "2b"]
    for (q, arm), g in r2.groupby(["q", "arm"], sort=True):
        dirty = sorted(int(s) for s in g[_b(g["dirty"])]["seed"])
        for name, val, unit in (
                ("runs", int(len(g)), "count"), ("seeds", " ".join(str(int(s)) for s in sorted(g["seed"])), "seed"),
                ("launch commit(s)", " ".join(sorted(g["commit"].astype(str).unique())), "hash"),
                ("runs with dirty = true", f"{len(dirty)} (seeds {' '.join(map(str, dirty)) or 'none'})", "count"),
                ("actor LR at the first update", " ".join(_f(v) for v in sorted(g["actor_lr_first"].unique())),
                 "learning rate"),
                ("actor LR at the last update", " ".join(_f(v) for v in sorted(g["actor_lr_last"].unique())),
                 "learning rate"),
                ("critic LR at the last update", " ".join(_f(v) for v in sorted(g["critic_lr_last"].unique())),
                 "learning rate"),
                ("runs where the existing Phase-B stop rule would have fired", int(g["would_fire_update"].notna().sum()),
                 "count"),
                ("drift_test passes", int(_b(g["drift_test_pass"]).sum()), "count"),
                ("max snapshot drift (mean, alpha, beta)", float(g["snapshot_drift_max"].max()), "Beta parameter units")):
            rows.append({"block": "2 run records", "family": "2b", "q": int(q), "arm": arm, "metric": name,
                         "tier": "n/a (training record)", "unit": unit, "value": val})
        for col, unit in (("adv_s1_std_median", "advantage units"), ("adv_used_std_median", "advantage units")):
            rows.append({"block": "2 run records", "family": "2b", "q": int(q), "arm": arm, "metric": col,
                         "tier": "n/a (training record)", "unit": unit, **_stat(g[col].astype(float))})
    blocks.append(pd.DataFrame(rows))
    # 3 paired decay - constant
    p = ps[(ps["family"] == "2b") & (ps["comparison"] == "decay_minus_constant")].copy()
    p["K"] = p["a"].map(_k_of)
    p["block"] = "3 paired decay - constant per (q, seed)"
    pr = psr[psr["family"] == "2b"].copy()
    pr["block"] = "3 paired decay - constant per (q, seed), run records"
    pr["a"], pr["b"] = "decay", "constant"
    p["arm"], pr["arm"] = "decay - constant", "decay - constant"
    for d in (p, pr):
        d["unit"] = d["metric"].map(_metric_unit)
        d["tier"] = d["metric"].map(lambda m: "development" if m in ("Gmax_full_over_dw", "EXP_root_over_dw",
                                                                     "dReach_over_dw") else
                                    ("n/a (training record)" if m in ("kl_final_epoch", "clip_frac", "phase_wall_sec")
                                     else TI))
    blocks += [p, pr]
    # 4 stability
    rows = []
    for r in st.itertuples(index=False):
        for col in ("across_seed_sd_final_e1", "across_seed_iqr_final_e1", "median_sigma1_0", "median_within_run_sd_last5",
                    "median_within_run_range_last5"):
            rows.append({"block": "4 stability (stability_2b.csv)", "family": "2b", "q": int(r.q), "arm": r.arm,
                         "K": 1, "metric": col, "tier": TI, "unit": UNIT_RAW, "value": float(getattr(r, col))})
    blocks.append(pd.DataFrame(rows))
    # 5 stage-1 tail averaging: candidates and paired K - K=1
    rows = []
    tail = cand[(cand["kind"] == "tail") & (cand["family"].isin(["p3", "2b"])) & (cand["K"].isin([1, 4, 8, 12]))]
    for (fam, q, arm, K), g in tail.groupby(["family", "q", "arm", "K"], sort=True):
        upd = int(g["update"].iloc[0])
        for col, tier, unit in TAIL_METRICS:
            rows.append({"block": "5 stage-1 tail averaging: candidates", "family": fam, "q": int(q), "arm": arm,
                         "K": int(K), "metric": col, "tier": tier, "unit": unit,
                         "value": f"exports u{int(g['first_export'].iloc[0])}..u{upd}", **_stat(g[col].astype(float))})
        loc = g[["Gmax_full_t", "Gmax_full_d"]].astype(float)
        rows.append({"block": "5 stage-1 tail averaging: candidates", "family": fam, "q": int(q), "arm": arm, "K": int(K),
                     "metric": "Gmax_full location (t*, d*) counts, development tier", "tier": "development",
                     "unit": "count (d* in effort units)", "n": int(len(g)),
                     "value": "; ".join(f"t*={int(t_)}, d*={_f(d_)}: {int(n_)}" for (t_, d_), n_ in
                                        loc.value_counts().sort_index().items())})
    blocks.append(pd.DataFrame(rows))
    pk = ps[(ps["family"].isin(["p3", "2b"])) & (ps["comparison"] == "K_vs_K1")].copy()
    pk["K"] = pk["a"].map(_k_of)
    pk["arm"] = pk["a"].map(lambda s: str(s).split()[0])
    pk["block"] = "6 stage-1 tail averaging: paired K - (K = 1) per (q, seed)"
    pk["unit"] = pk["metric"].map(_metric_unit)
    pk["tier"] = pk["metric"].map(lambda m: "development" if m in ("Gmax_full_over_dw", "EXP_root_over_dw",
                                                                   "dReach_over_dw") else TI)
    blocks.append(pk)
    cols = ["block", "family", "q", "arm", "K", "metric", "tier", "unit", "n", "median", "q25", "q75", "min", "max",
            "value", "comparison", "a", "b"] + PAIRED_COLS
    df = _union(blocks, cols)
    # paired-row min/max/median columns come from the paired files; summary rows from _stat
    sources = [C.src(f"{P4A}/candidates_all.csv"), C.src(f"{P4A}/run_records.csv"), C.src(f"{P4A}/paired_summary.csv"),
               C.src(f"{P4A}/paired_summary_run_records.csv"), C.src(f"{P4A}/stability_2b.csv"),
               C.srcs("results/v2_pilots/pilot4_B/q*/seed*/*/final_v2.json",
                      label="results/v2_pilots/pilot4_B/q*/seed*/*/final_v2.json", expect=40),
               C.src("tools/v2/pilot4_analysis.py"), C.src(RPT_P4)]
    docs = {
        "block": D_BLOCK,
        "family": "Analysis family: 2b = Pilot 4 section 2b (Phase B LR decay, parents u1600), p3 = Pilot 3 (parents u400)",
        "q": "q (50 or 60)",
        "arm": "Arm: constant / decay (2b: B2_mean_constant / B2_mean_decay), stochastic / mean (Pilot 3)",
        "K": "Number of last weight exports averaged pointwise (K = 1 = last iterate); empty for run-record rows",
        "metric": ("Metric name (column of candidates_all.csv / run_records.csv / stability_2b.csv; suffix _dev = "
                   "development tier from candidates_all.csv, _final = final tier from final_v2.json['final'])"),
        "tier": "Verifier tier of the metric (development, final, tier-independent, or n/a for training records)",
        "unit": D_UNIT,
        "n": "Number of runs behind a summary row",
        "median": "Summary rows: median over the 10 seeds; paired rows: median of the 10 paired differences",
        "q25": "25th percentile over seeds (numpy linear interpolation)",
        "q75": "75th percentile over seeds (numpy linear interpolation)",
        "min": "Summary rows: minimum over seeds; paired rows: minimum paired difference",
        "max": "Summary rows: maximum over seeds; paired rows: maximum paired difference",
        "value": "Text or single value of the row (counts, locations, run records, stability statistics, export window)",
        "comparison": "Paired comparison: decay_minus_constant (decay - constant, same q and seed) or K_vs_K1 (K - (K=1), "
                      "same arm, q and seed)",
        "a": "First member of the paired difference (arm and K)",
        "b": "Second member of the paired difference (arm and K)",
        "n_pairs": "Number of (q, seed) pairs",
        "mean": "Mean paired difference",
        "n_neg": "Pairs with a negative difference", "n_pos": "Pairs with a positive difference",
        "n_zero": "Pairs with a zero difference",
        "boot_ci95_lo": ("Lower end of the 95% percentile-bootstrap CI of the mean paired difference (10,000 resamples, "
                         "numpy default_rng(20261001), one generator per comparison block; tools/v2/pilot4_analysis.py)"),
        "boot_ci95_hi": "Upper end of the 95% percentile-bootstrap CI of the mean paired difference",
        "n_better": "Pairs in the better direction (better_if); empty when the metric has no preferred direction",
        "better_if": "Direction counted as better"}
    pack.table("T51", df, status="generated", sources=sources, script=f"{MOD}:build_t51", tier="final and development",
               notes=("Pilot 4 section 2b runs: q in {50, 60} x seeds 10501-10510 x arms constant/decay (40 runs; parents "
                      "Phase A extension state_u01600.pt; stage 2 frozen at u1600; Phase B u1601-u2200). Block 1: median, "
                      "IQR, min, max over the 10 seeds of candidates_all.csv (family 2b, kind tail, K = 1 = the u2200 "
                      "export, development tier) merged with run_records.csv and the final-tier scalars of each run's "
                      "final_v2.json (suffix _final). Blocks 3 and 6 are the existing paired summaries (bootstrap 10,000 "
                      "resamples, numpy seed 20261001) copied as is; block 4 is stability_2b.csv; block 5 summarises the "
                      "tail-averaged candidates of Pilot 3 (u1000, K = 12 covers u725-u1000) and 2b (u2200, K = 12 covers "
                      "u1925-u2200). Tail-averaged candidates exist on the development tier only."),
               caption="Pilot 4 section 2b (Phase B LR decay 3e-4 to 3e-5 vs constant): final medians at u2200, paired "
                       "differences, stability, and stage-1 tail averaging (Pilot 4 sections 1c.1 and 2b).", docs=docs)
    _t51_checks(pack, per, cand, rr, ps, psr, st)


def _t51_checks(pack: C.Pack, per: pd.DataFrame, cand: pd.DataFrame, rr: pd.DataFrame, ps: pd.DataFrame,
                psr: pd.DataFrame, st: pd.DataFrame) -> None:
    """Cross-checks of T51 against pilot4_stabilization.md (sections 1c.1, 2a/2b, 7) and summary.md."""
    pack.crosscheck("T51", st, RPT_P4, header_has=["q", "arm", "across_seed_sd_final_e1", "median_within_run_sd_last5"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={c: c for c in st.columns if c not in ("q", "arm")}, label="2b stability table")
    tail = cand[(cand["kind"] == "tail") & (cand["family"].isin(["p3", "2b"])) & (cand["K"].isin([1, 4, 8, 12]))]
    mcols = ["e1_cand", "stage1_rel_err_signed", "stage1_rel_err_abs", "learning_rel", "inherited_rel",
             "Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw"]
    med = tail.groupby(["q", "arm", "K"], sort=True)[mcols].median().reset_index()
    pack.crosscheck("T51", med, RPT_P4, header_has=["q", "arm", "K", "e1_cand", "stage1_rel_err_signed"],
                    key_map={"q": "q", "arm": "arm", "K": "K"}, value_map={c: c for c in mcols},
                    label="tail-averaged candidates, medians (sections 1c.1 and 2b)")
    vcols = ["e1_cand", "stage1_rel_err_signed", "learning_rel", "inherited_rel", "sigma_effort_at_0_t1",
             "within_run_sd_e1_last5", "within_run_range_e1_last5", "Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d",
             "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw", "kl_final_epoch",
             "clip_frac", "phase_wall_sec", "would_fire_update"]
    pack.crosscheck("T51", per, RPT_P4, header_has=["q", "seed", "arm", "e1_cand", "within_run_sd_e1_last5"],
                    key_map={"q": "q", "seed": "seed", "arm": "arm"}, value_map={c: c for c in vcols},
                    label="2b per-run table at u2200 (40 runs)")
    rec = []
    for (q, arm), g in rr[rr["family"] == "2b"].groupby(["q", "arm"], sort=True):
        wf = g["would_fire_update"].astype(float)
        rec.append({"q": int(q), "arm": arm, "n": int(len(g)), "lr_first": float(g["actor_lr_first"].median()),
                    "lr_last": float(g["actor_lr_last"].median()), "n_would_fire": int(wf.notna().sum()),
                    "wf_median": float(wf.median()), "wf_min": float(wf.min()), "wf_max": float(wf.max()),
                    "wall_median": float(g["phase_wall_sec"].median()), "kl_median": float(g["kl_median"].median()),
                    "clip_median": float(g["clip_median"].median()),
                    "adv_s1_std": float(g["adv_s1_std_median"].median()),
                    "adv_used_std": float(g["adv_used_std_median"].median())})
    rec = pd.DataFrame(rec)
    pack.crosscheck("T51", rec, RPT_P4, header_has=["q", "arm", "n", "commit", "wf_median", "adv_used_std"],
                    key_map={"q": "q", "arm": "arm"}, heading_has="2b. Phase B decay",
                    value_map={c: c for c in rec.columns if c not in ("q", "arm")}, label="2b run records table")
    pack.crosscheck("T51", psr, RPT_P4, header_has=["family", "q", "metric", "median", "n_better", "n_neg"],
                    key_map={"family": "family", "q": "q", "metric": "metric"},
                    value_map={c: c for c in ("median", "n_better", "n_neg", "n_pos", "boot_ci95_lo", "boot_ci95_hi")},
                    label="run-record paired differences (families 2a and 2b)")
    loc = []
    for (fam, q, arm, K), g in tail.groupby(["family", "q", "arm", "K"], sort=True):
        loc.append({"family": fam, "q": int(q), "arm": arm, "K": int(K), "t1": int((g["Gmax_full_t"] == 1).sum()),
                    "t2": int((g["Gmax_full_t"] == 2).sum()),
                    "dvals": " ".join(_f(v) for v in sorted(g["Gmax_full_d"].astype(float).unique()))})
    loc = pd.DataFrame(loc)
    pack.crosscheck("T51", loc, RPT_P4, header_has=["family", "q", "arm", "K", "t*=1 count", "t*=2 count"],
                    key_map={"family": "family", "q": "q", "arm": "arm", "K": "K"},
                    value_map={"t*=1 count": "t1", "t*=2 count": "t2"}, label="location of Gmax_full (counts)")
    ck = _Checks(pack, "T51", RPT_P4, "2b and 1c.1 paired tables and prose (sections 1c.1, 2b, 7)")
    for tb in _md_tables(RPT_P4, ["t*=1 count", "d* values"]):
        for r in tb["rows"]:
            rec_ = dict(zip(tb["header"], r))
            hit = loc[(loc["family"] == rec_["family"]) & (loc["q"] == int(rec_["q"])) & (loc["arm"] == rec_["arm"])
                      & (loc["K"] == int(rec_["K"]))]
            if len(hit) == 1:
                ck.same(f"d* values {rec_['family']} q{rec_['q']} {rec_['arm']} K={rec_['K']}", hit.iloc[0]["dvals"],
                        rec_["d* values"], "section 2b, location of Gmax_full")
    labmap = {"|stage-1 err|": ("ps", "stage1_rel_err_abs"), "|learning term|": ("ps", "learning_rel_abs"),
              "EXP_root/ΔW": ("ps", "EXP_root_over_dw"), "dReach/ΔW": ("ps", "dReach_over_dw"),
              "within-run SD last5": ("psr", "within_run_sd_e1_last5"),
              "within-run range last5": ("psr", "within_run_range_e1_last5")}
    for tb in _md_tables(RPT_P4, ["q", "K", "metric", "median", "better", "CI95"], heading_has="2b"):
        for r in tb["rows"]:
            rec_ = dict(zip(tb["header"], r))
            src, met = labmap.get(rec_["metric"], (None, None))
            if src is None:
                continue
            q, K = int(rec_["q"]), int(rec_["K"])
            if src == "ps":
                h = ps[(ps["family"] == "2b") & (ps["comparison"] == "decay_minus_constant") & (ps["q"] == q)
                       & (ps["metric"] == met) & (ps["a"] == f"decay K={K}")]
            else:
                h = psr[(psr["family"] == "2b") & (psr["q"] == q) & (psr["metric"] == met)]
            h = h.iloc[0]
            lo, hi = _bracket_strs(rec_["CI95"])[1:]
            where = f"section 2b paired decay - constant (line {tb['line']})"
            ck.num(f"q{q} K={K} {met}: median", float(h["median"]), rec_["median"], where)
            ck.num(f"q{q} K={K} {met}: better", float(h["n_better"]), rec_["better"], where)
            ck.num(f"q{q} K={K} {met}: CI lo", float(h["boot_ci95_lo"]), lo, where)
            ck.num(f"q{q} K={K} {met}: CI hi", float(h["boot_ci95_hi"]), hi, where)
    for tb in _md_tables(RPT_P4, ["q", "arm", "K", "Δ abs stage-1 err (median)"]):
        for r in tb["rows"]:
            q, arm, K = int(r[0]), r[1], int(r[2])
            for met, i0 in (("stage1_rel_err_abs", 3), ("EXP_root_over_dw", 6)):
                h = ps[(ps["family"] == "p3") & (ps["comparison"] == "K_vs_K1") & (ps["q"] == q)
                       & (ps["metric"] == met) & (ps["a"] == f"{arm} K={K}")].iloc[0]
                lo, hi = _bracket_strs(r[i0 + 2])[1:]
                where = f"section 1c.1 paired K - (K=1) (line {tb['line']})"
                ck.num(f"p3 q{q} {arm} K={K} {met}: median", float(h["median"]), r[i0], where)
                ck.num(f"p3 q{q} {arm} K={K} {met}: better", float(h["n_better"]), r[i0 + 1], where)
                ck.num(f"p3 q{q} {arm} K={K} {met}: CI lo", float(h["boot_ci95_lo"]), lo, where)
                ck.num(f"p3 q{q} {arm} K={K} {met}: CI hi", float(h["boot_ci95_hi"]), hi, where)
    txt = _text(RPT_P4)
    m = re.search(r"Signed stage-1 error, decay − constant, K = 1: q=50 median ([+−\-\d.]+) \((\d+)−/(\d+)\+, CI "
                  r"\[([−\-\d.]+), ([−\-\d.]+)\]\); q=60 ([+−\-\d.]+) \((\d+)−/(\d+)\+, CI \[([−\-\d.]+), ([−\-\d.]+)\]\)",
                  txt)
    for q, gi in ((50, 1), (60, 6)):
        h = ps[(ps["family"] == "2b") & (ps["comparison"] == "decay_minus_constant") & (ps["q"] == q)
               & (ps["metric"] == "stage1_rel_err_signed") & (ps["a"] == "decay K=1")].iloc[0]
        where = "section 2b, signed stage-1 error sentence"
        for col, cell in (("median", m.group(gi)), ("n_neg", m.group(gi + 1)), ("n_pos", m.group(gi + 2)),
                          ("boot_ci95_lo", m.group(gi + 3)), ("boot_ci95_hi", m.group(gi + 4))):
            ck.num(f"q{q} signed stage-1 error decay - constant K=1: {col}", float(h[col]), cell, where)
    for arm, line_start in (("constant", "- constant arm:"), ("decay", "- decay arm:")):
        line = next(ln for ln in txt.splitlines() if ln.startswith(line_start))
        for q in QS:
            seg = line.split(f"q={q}")[1]
            vals = re.findall(r"[+−\-]\d+\.\d+", seg)[:3]
            for K, cell in zip((4, 8, 12), vals):
                h = ps[(ps["family"] == "2b") & (ps["comparison"] == "K_vs_K1") & (ps["q"] == q)
                       & (ps["metric"] == "stage1_rel_err_abs") & (ps["a"] == f"{arm} K={K}")].iloc[0]
                ck.num(f"2b {arm} q{q} K={K} - K=1 |stage-1 err| median", float(h["median"]), cell,
                       "section 2b, K - (K=1) bullets")
                if not (h["boot_ci95_lo"] <= 0.0 <= h["boot_ci95_hi"]):
                    ck.differ(f"2b {arm} q{q} K={K}: CI contains 0", False, "All CIs include 0",
                              "section 2b, K - (K=1) bullets", "")
    def stat(arm: str, q: int, col: str) -> float:
        return float(per[(per["arm"] == arm) & (per["q"] == q)][col].median())
    stq = st.set_index(["q", "arm"])
    for q, sd_d, sd_c, ac_d, ac_c, ab_d, ab_c, sg_c in ((50, "1.02", "1.75", "2.49", "3.33", "0.043", "0.063", "+0.049"),
                                                        (60, "0.99", "1.37", "1.70", "3.31", "0.034", "0.055", "+0.036")):
        w7 = "section 7, bullet 2b"
        ck.num(f"q{q} decay median within-run SD last5", stq.loc[(q, "decay"), "median_within_run_sd_last5"], sd_d, w7)
        ck.num(f"q{q} constant median within-run SD last5", stq.loc[(q, "constant"), "median_within_run_sd_last5"], sd_c, w7)
        ck.num(f"q{q} decay across-seed SD final e1", stq.loc[(q, "decay"), "across_seed_sd_final_e1"], ac_d, w7)
        ck.num(f"q{q} constant across-seed SD final e1", stq.loc[(q, "constant"), "across_seed_sd_final_e1"], ac_c, w7)
        ck.num(f"q{q} decay median |stage-1 err| K=1", stat("decay", q, "stage1_rel_err_abs"), ab_d, w7)
        ck.num(f"q{q} constant median |stage-1 err| K=1", stat("constant", q, "stage1_rel_err_abs"), ab_c, w7)
        ck.num(f"q{q} constant median signed stage-1 err K=1", stat("constant", q, "stage1_rel_err_signed"), sg_c, w7)
    t12 = cand[(cand["family"] == "2b") & (cand["kind"] == "tail") & (cand["K"] == 12) & (cand["arm"] == "constant")]
    for q, cell in ((50, "+0.058"), (60, "+0.065")):
        ck.num(f"q{q} constant median signed stage-1 err K=12",
               float(t12[t12["q"] == q]["stage1_rel_err_signed"].median()), cell, "section 7, bullet 2b")
    clip = rr[rr["family"] == "2b"].groupby(["q", "arm"])["clip_median"].median()
    for q, cd, cc in ((50, "0.051", "0.087"), (60, "0.059", "0.099")):
        ck.num(f"q{q} decay clip median", float(clip.loc[(q, "decay")]), cd, "section 2b, advantage statistics")
        ck.num(f"q{q} constant clip median", float(clip.loc[(q, "constant")]), cc, "section 2b, advantage statistics")
    for q in QS:
        h = psr[(psr["family"] == "2b") & (psr["q"] == q) & (psr["metric"] == "clip_frac")].iloc[0]
        ck.num(f"q{q} pairs with lower last-update clip fraction (decay)", float(h["n_neg"]), "9",
               "section 2b, advantage statistics")
    wall = rr[rr["family"] == "2b"].groupby(["q", "arm"])["phase_wall_sec"].median()
    ck.num("2b Phase B wall, smallest median per (q, arm)", float(wall.min()), "145", "section 2b, wall clock")
    ck.num("2b Phase B wall, largest median per (q, arm)", float(wall.max()), "156", "section 2b, wall clock")
    ck.close()
    cks = _Checks(pack, "T51", RPT_SUM, "Pilot 4 bullet 2b table")
    for q, sd, acs, ab, nlow in ((50, ("1.02", "1.75"), ("2.49", "3.33"), ("0.043", "0.063"), "7"),
                                 (60, ("0.99", "1.37"), ("1.70", "3.31"), ("0.034", "0.055"), "6")):
        cks.num(f"q{q} within-run SD last5 decay", stq.loc[(q, "decay"), "median_within_run_sd_last5"], sd[0], "2b table")
        cks.num(f"q{q} within-run SD last5 constant", stq.loc[(q, "constant"), "median_within_run_sd_last5"], sd[1],
                "2b table")
        cks.num(f"q{q} across-seed SD decay", stq.loc[(q, "decay"), "across_seed_sd_final_e1"], acs[0], "2b table")
        cks.num(f"q{q} across-seed SD constant", stq.loc[(q, "constant"), "across_seed_sd_final_e1"], acs[1], "2b table")
        cks.num(f"q{q} |stage-1 err| K=1 decay", stat("decay", q, "stage1_rel_err_abs"), ab[0], "2b table")
        cks.num(f"q{q} |stage-1 err| K=1 constant", stat("constant", q, "stage1_rel_err_abs"), ab[1], "2b table")
        h = ps[(ps["family"] == "2b") & (ps["comparison"] == "decay_minus_constant") & (ps["q"] == q)
               & (ps["metric"] == "EXP_root_over_dw") & (ps["a"] == "decay K=1")].iloc[0]
        cks.num(f"q{q} EXP_root lower with decay (pairs)", float(h["n_better"]), nlow, "2b table")
    cks.close()


def build_f24(pack: C.Pack) -> None:
    """Pilot 4 2b learning curves (u1625-u2200), median and IQR constant vs decay per q (dev tier)."""
    cand = _csv(f"{P4A}/candidates_all.csv")
    c = cand[(cand["family"] == "2b") & (cand["K"] == 1)]
    panels = [("stage1_rel_err_signed", "stage-1 error (signed, /e₁*)", True),
              ("learning_rel", "learning term (/e₁*)", True),
              ("sigma_effort_at_0_t1", "σ₁(0) (effort units)", False),
              ("Gmax_full_over_dw", "Ĝmax_full / ΔW (dev tier)", False),
              ("EXP_root_over_dw", "EXP_root / ΔW (dev tier)", False)]
    fig, axes = style.new_figure(len(panels), 2, height=9.4, sharex=True)
    out, d_pd = [], 0.0
    arm_mk = {"constant": "o", "decay": "D"}
    for j, q in enumerate(QS):
        for i, (m, ylab, zero) in enumerate(panels):
            ax = axes[i, j]
            for arm in ("constant", "decay"):
                g = c[(c["q"] == q) & (c["arm"] == arm)]
                n = int(g["seed"].nunique())
                res = style.median_iqr_curves(ax, g, "update", m, style.ARM_COLORS[arm],
                                              style.label_n(style.arm_label(arm), n), marker=arm_mk[arm],
                                              extra={"q": q, "arm": arm})
                gb = g.groupby("update")[m]
                d_pd = max(d_pd, C.max_abs_diff(res["median"], gb.median().to_numpy()),
                           C.max_abs_diff(res["q25"], gb.quantile(0.25).to_numpy()),
                           C.max_abs_diff(res["q75"], gb.quantile(0.75).to_numpy()))
                out.append(res)
            if zero:
                ax.axhline(0.0, color=style.REF, lw=0.8)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(f"q = {q}")
            if i == len(panels) - 1:
                ax.set_xlabel("global update")
            if i == 0 and j == 1:
                ax.legend(loc="upper right")
    for ax in axes.ravel():
        for ln in ax.get_lines():
            if ln.get_marker() not in ("None", None, ""):
                ln.set_markersize(3.0)
    data = pd.concat(out, ignore_index=True)
    fin = data[data["update"] == 2200]
    per = _p4b_per_run()
    d_fin = 0.0
    for r in fin.itertuples(index=False):
        g = per[(per["q"] == r.q) & (per["arm"] == r.arm)][r.metric]
        d_fin = max(d_fin, abs(float(np.median(g)) - float(r.median)))
    n_upd = sorted(data["update"].unique())
    nmin = int(data["n"].min())
    cap = ("Pilot 4 section 2b (Phase B u1601–u2200, stage 2 frozen at the u1600 parent): median (line) and IQR "
           "(band) over 10 seeds per arm of the weight exports every 25 updates (u1625–u2175) and the u2200 last "
           "iterate, constant LR vs LR decay 3e-4→3e-5, per q (columns). Panels: signed stage-1 error and learning "
           "term (fractions of e₁*(0); final-tier band), σ₁(0) (effort units), Ĝmax_full/ΔW and "
           "EXP_root/ΔW (development tier). Source: results/v2_pilots/pilot4/analysis/candidates_all.csv, family 2b, "
           f"K = 1 ({len(n_upd)} updates). Development tier for the verifier metrics; recovery metrics tier-independent; "
           f"n = {nmin} runs per arm, update and q.")
    pack.figure("F24", fig, data, status="regenerated", sources=[C.src(f"{P4A}/candidates_all.csv"),
                                                                 C.src("reports/v2/figures/pilot4/curves_2b_q50.png"),
                                                                 C.src("reports/v2/figures/pilot4/curves_2b_q60.png"),
                                                                 C.src("tools/v2/pilot4_analysis.py")],
                script=f"{MOD}:build_f24", caption=cap, tier="development",
                checks=[f"plotted median/q25/q75 equal the pandas groupby median/quantile(0.25/0.75) used by "
                        f"tools/v2/pilot4_analysis.py:plots for the original curves_2b_q50/q60.png (max abs diff {d_pd:g})",
                        f"plotted u2200 medians equal the T51 block-1 medians of the same 40 runs (max abs diff {d_fin:g})",
                        f"every plotted point has n = {nmin} runs"],
                notes=("Restyled from the same rows as reports/v2/figures/pilot4/curves_2b_q50.png and curves_2b_q60.png "
                       "(candidates_all.csv, family 2b, K = 1, kind curve + tail); one figure for both q."),
                docs={"update": "Global update of the export (u2200 = last iterate)", "metric": "Plotted metric (column "
                      "of candidates_all.csv)", "median": "Median over the 10 seeds", "q25": "25th percentile over seeds "
                      "(numpy linear)", "q75": "75th percentile over seeds (numpy linear)", "min": "Minimum over seeds",
                      "max": "Maximum over seeds", "n": "Number of runs at the point", "q": "q (50 or 60)",
                      "arm": "Arm: constant LR or LR decay"})


# ----------------------------------------------------------------------------------------------
# T54 v1.0 Check 1 and Check 2
# ----------------------------------------------------------------------------------------------

GLOBAL_RNGS = ("torch_global", "numpy_global", "python_random")


def _sha(b: bytes) -> str:
    """SHA-256 hex digest."""
    return hashlib.sha256(b).hexdigest()


def _saved_global_states(rel: str) -> Dict[str, Any]:
    """Digests of the three process-global RNG states stored in a full-state file, plus counters.

    The digest definition is that of ``run/run_v2_T2_locked.py`` (``_state_bytes`` / ``digests``).
    """
    import torch

    s = torch.load(C.abspath(rel), weights_only=False, map_location="cpu")
    kind, key, pos, has_gauss, cached = s["numpy_global_rng_state"]
    return {"torch_global": _sha(s["torch_global_rng_state"].numpy().tobytes()),
            "numpy_global": _sha(repr((kind, pos, has_gauss, float(cached))).encode() + np.asarray(key).tobytes()),
            "python_random": _sha(repr(s["python_random_state"]).encode()),
            "numpy_pos": int(pos), "global_u": int(s["counters"]["global_u"]),
            "snapshot_refreshes": int(s["counters"]["snapshot_refreshes"])}


def build_t54(pack: C.Pack) -> None:
    """v1.0 Check 1 (field table, process-global RNG explanation) and Check 2 (field table), decision D1."""
    c1 = _csv(f"{RA}/check1_phaseA_vs_stitched.csv")
    c2 = _csv(f"{RA}/check2_phaseB_vs_launcher.csv")
    proto = _js(PROTO)
    rows: List[Dict[str, Any]] = []
    rep_fields = set()
    for tb in _md_tables(RPT_LOCK, ["field", "identical (of 20)"]):
        rep_fields |= {r[0] for r in tb["rows"]}
    bcols = [c for c in c1.columns if c1[c].dtype == bool]
    for col in bcols:
        for q in ("all", 50, 60):
            g = c1 if q == "all" else c1[c1["q"] == q]
            rows.append({"block": "1 Check 1 field table (rehearsal Phase A vs stitched development path)", "field": col,
                         "q": q, "n_runs": int(len(g)), "n_identical": int(_b(g[col]).sum()),
                         "value": "in the report's field table" if col in rep_fields else
                         "not in the report's field table",
                         "source": f"{RA}/check1_phaseA_vs_stitched.csv"})
    for col in ("global_u", "stitched_global_u", "n_exports_A", "snapshot_refreshes_rehearsal",
                "snapshot_refreshes_stitched"):
        vc = c1[col].value_counts().sort_index()
        rows.append({"block": "1 Check 1 field table (rehearsal Phase A vs stitched development path)", "field": col,
                     "q": "all", "n_runs": int(len(c1)),
                     "value": "; ".join(f"{int(k)} in {int(v)}/{len(c1)}" for k, v in vc.items()),
                     "source": f"{RA}/check1_phaseA_vs_stitched.csv"})
    # process-global RNG states from the saved full states (100 files)
    reh_a = C.srcs(f"{LK}/rehearsal/q*/seed*/state_end_A.pt", label=f"{LK}/rehearsal/q*/seed*/state_end_A.pt",
                   expect=20)
    reh_b = C.srcs(f"{LK}/rehearsal/q*/seed*/state_end_B.pt", label=f"{LK}/rehearsal/q*/seed*/state_end_B.pt",
                   expect=20)
    st400 = C.srcs("results/v2_pilots/pilot1/q*/seed*/expected/state_end_A.pt",
                   label="results/v2_pilots/pilot1/q*/seed*/expected/state_end_A.pt (stitched path u400)", expect=20)
    st1200 = C.srcs("results/v2_pilots/phaseA_ext/q*/seed*/expected_ext/state_u01200.pt",
                    label="results/v2_pilots/phaseA_ext/q*/seed*/expected_ext/state_u01200.pt (stitched path u1200)",
                    expect=20)
    st1600 = C.srcs("results/v2_pilots/pilot4_A/q*/seed*/decay/state_u01600.pt",
                    label="results/v2_pilots/pilot4_A/q*/seed*/decay/state_u01600.pt (stitched path u1600)", expect=20)

    def key(path: str) -> Tuple[int, int]:
        m = re.search(r"/q(\d+)/seed(\d+)/", path)
        return int(m.group(1)), int(m.group(2))

    states = {name: {key(s.path): _saved_global_states(s.path) for s in ss.files}
              for name, ss in (("reh_a", reh_a), ("reh_b", reh_b), ("s400", st400), ("s1200", st1200),
                               ("s1600", st1600))}
    runs = sorted(states["reh_a"])
    blk = "2 Check 1: the three process-global RNG states"
    stsrc = "saved full states (torch_global_rng_state, numpy_global_rng_state, python_random_state)"
    expl = {"torch_global": "torch global generator (torch.manual_seed / torch.get_rng_state)",
            "numpy_global": "numpy legacy global RandomState (np.random.seed / np.random.get_state)",
            "python_random": "Python random module (random.seed / random.getstate)"}
    for rng in GLOBAL_RNGS:
        col = {"torch_global": "torch_global_rng_identical", "numpy_global": "numpy_global_rng_identical",
               "python_random": "python_random_identical"}[rng]
        n_ab = sum(states["reh_a"][k][rng] == states["reh_b"][k][rng] for k in runs)
        n_st = sum(states["s400"][k][rng] == states["s1200"][k][rng] == states["s1600"][k][rng] for k in runs)
        n_distinct = len({states["reh_a"][k][rng] for k in runs})
        n_vs = sum(states["reh_a"][k][rng] == states["s1600"][k][rng] for k in runs)
        for field, n_id, val, src in (
                ("what it is", None, expl[rng], "run/run_v2_T2_locked.py (global_states); source: report text"),
                ("seeded by the v1.0 runner", None, "no: a fresh process initialises it from OS entropy; the full state "
                 "records it and restore copies it (source: report text)", f"{RPT_LOCK} section 4.1"),
                ("rehearsal end of A vs stitched u1600 (Check 1 field)", int(_b(c1[col]).sum()),
                 f"identical in {int(_b(c1[col]).sum())}/{len(c1)}", f"{RA}/check1_phaseA_vs_stitched.csv"),
                ("recomputed from the saved states: rehearsal end of A vs stitched u1600", n_vs,
                 f"identical in {n_vs}/{len(runs)}", stsrc),
                ("rehearsal: end of A vs end of B of the same run (600 Phase B updates)", n_ab,
                 f"identical in {n_ab}/{len(runs)}", stsrc),
                ("rehearsal: distinct states across the 20 processes (end of A)", None, f"{n_distinct} distinct", stsrc),
                ("stitched path: Pilot 1 u400 = extension u1200 = 2a decay u1600", n_st,
                 f"identical in {n_st}/{len(runs)} (restored and never advanced over 1,200 updates)", stsrc)):
            rows.append({"block": blk, "field": f"{rng}: {field}", "q": "all", "n_runs": len(runs) if n_id is not None
                         else None, "n_identical": n_id, "value": val, "source": src})
    pos = sorted({v["numpy_pos"] for d in states.values() for v in d.values()})
    rows.append({"block": blk, "field": "numpy_global: MT19937 position in all 100 saved states", "q": "all",
                 "n_runs": 100, "value": " ".join(map(str, pos)), "source": stsrc})
    rows.append({"block": blk, "field": "consumed by training?", "q": "all",
                 "value": "no evidence of consumption: the states do not advance within a run (rows above) while every "
                          "training stream, the torch generator, all 64 exports and the u1600 stage-2 metrics are identical "
                          "(field table); if training drew from them, exports and streams would diverge (source: report "
                          "text, section 4.1)", "source": f"{RPT_LOCK} section 4.1; {RA}/check1_phaseA_vs_stitched.csv"})
    rows.append({"block": blk, "field": "torch.initial_seed() in one fresh process", "q": "all",
                 "value": "17167615931904387274 (entropy-derived; source: report text)", "source": f"{RPT_LOCK} section 4.1"})
    # snapshot-refresh counter
    blk3 = "3 Check 1: snapshot-refresh counter (field not in the report's table)"
    for name, label in (("reh_a", "rehearsal end of A (u1600)"), ("s400", "stitched: Pilot 1 end (u400)"),
                        ("s1200", "stitched: extension u1200"), ("s1600", "stitched: 2a decay u1600")):
        vc = pd.Series([v["snapshot_refreshes"] for v in states[name].values()]).value_counts().sort_index()
        rows.append({"block": blk3, "field": f"snapshot refreshes, {label}", "q": "all", "n_runs": len(states[name]),
                     "value": "; ".join(f"{int(k)} in {int(v)}/{len(states[name])}" for k, v in vc.items()),
                     "source": stsrc})
    ln = _line_of("run/run_v2_stagewise.py", r"agent\.refresh_snapshot\(\)", after=r"def run_phase")
    rows.append({"block": blk3, "field": "why the counts differ by 2", "q": "all",
                 "value": f"run_phase refreshes the lagged opponent at every phase entry (run/run_v2_stagewise.py:{ln}); "
                          "the stitched path entered Phase A three times (u0, u400, u1200), the rehearsal once; the opponent "
                          "tensors are identical in 20/20 (field opponent_identical)",
                 "source": f"run/run_v2_stagewise.py; {RA}/check1_phaseA_vs_stitched.csv"})
    # Check 2
    blk4 = "4 Check 2 field table (rehearsal Phase B vs the pilot launcher from the stitched u1600 state)"
    for col in [c for c in c2.columns if c not in ("q", "seed")]:
        vals = "; ".join(f"q{int(r.q)} s{int(r.seed)}: {getattr(r, col)}" for r in c2.itertuples(index=False))
        n_id = int(_b(c2[col]).sum()) if c2[col].dtype == bool else None
        rows.append({"block": blk4, "field": col, "q": "all", "n_runs": int(len(c2)), "n_identical": n_id, "value": vals,
                     "source": f"{RA}/check2_phaseB_vs_launcher.csv"})
    # decision D1
    blk5 = "5 owner decision D1 (v1.1 round) and the training-relevant state"
    trs = proto["training_relevant_state"]
    rows.append({"block": blk5, "field": "decision", "q": "all",
                 "value": "Check 1 accepted: training state bit-identical in 20/20; the three process-global RNG states were "
                          "never consumed by training (diagnostics only) (source: report text)",
                 "source": f"{RPT_LOCK} addendum (2026-10-02); {RPT_V11} section 1.3"})
    for i, item in enumerate(trs["compared"], 1):
        rows.append({"block": blk5, "field": f"compared {i}", "q": "all", "value": item,
                     "source": f"{PROTO} training_relevant_state.compared"})
    for i, item in enumerate(trs["excluded"], 1):
        rows.append({"block": blk5, "field": f"excluded {i}", "q": "all", "value": item,
                     "source": f"{PROTO} training_relevant_state.excluded"})
    rows.append({"block": blk5, "field": "decided", "q": "all", "value": trs["decided"],
                 "source": f"{PROTO} training_relevant_state.decided"})
    df = pd.DataFrame(rows)
    df = _union([df], ["block", "field", "q", "n_runs", "n_identical", "value", "source"])
    srcs = [C.src(f"{RA}/check1_phaseA_vs_stitched.csv"), C.src(f"{RA}/check2_phaseB_vs_launcher.csv"),
            reh_a, reh_b, st400, st1200, st1600, C.src(PROTO), C.src(RPT_LOCK), C.src(RPT_V11),
            C.src("run/run_v2_stagewise.py"), C.src("run/run_v2_T2_locked.py")]
    pack.table("T54", df, status="generated", sources=srcs, script=f"{MOD}:build_t54", tier="n/a",
               notes=("Check 1 field counts from check1_phaseA_vs_stitched.csv (20 runs, per q and pooled). The explanation "
                      "of the three process-global RNG states is verified on the saved full states: SHA-256 of each state "
                      "(digest definition of run/run_v2_T2_locked.py) compared within runs (rehearsal end of A vs end of B), "
                      "across runs, and along the stitched path (Pilot 1 u400, extension u1200, 2a decay u1600); text "
                      "marked source: report text. Check 2 from check2_phaseB_vs_launcher.csv (2 runs). Decision D1 and the "
                      "training-relevant state from the v1.1 protocol JSON and the v1.0 report addendum."),
               caption="v1.0 rehearsal Check 1 (Phase A vs the stitched development path) and Check 2 (Phase B code path), "
                       "with the explanation of the three process-global RNG states and decision D1.",
               docs={"block": D_BLOCK, "field": "Compared field, or the statement the row documents",
                     "q": D_Q_ALL, "n_runs": "Number of runs (or saved states) compared",
                     "n_identical": "Number of runs in which the field is bit-identical",
                     "value": "Values / statement of the row", "source": D_SRC})
    sub = df[(df["block"].str.startswith("1 ")) & (df["q"] == "all") & df["n_identical"].notna()][["field", "n_identical"]]
    pack.crosscheck("T54", sub.astype({"n_identical": float}), RPT_LOCK, header_has=["field", "identical (of 20)"],
                    key_map={"field": "field"}, value_map={"identical (of 20)": "n_identical"},
                    label="Check 1 field table")
    ck = _Checks(pack, "T54", RPT_LOCK, "Check 2 field table (yes/no) and section 4.1 prose")
    for tb in _md_tables(RPT_LOCK, ["field", "q50 s10503", "q60 s10503"]):
        for r in tb["rows"]:
            f, v50, v60 = r[0], r[1], r[2]
            if f not in c2.columns or f in ("q", "seed"):
                continue
            for q, cell in ((50, v50), (60, v60)):
                v = c2[c2["q"] == q].iloc[0][f]
                if isinstance(v, (bool, np.bool_)):
                    ck.same(f"Check 2 q{q} {f}", "yes" if bool(v) else "no", cell, "section 4.2 table")
                else:
                    ck.num(f"Check 2 q{q} {f}", float(v), cell, "section 4.2 table")
    for rng in GLOBAL_RNGS:
        n_ab = sum(states["reh_a"][k][rng] == states["reh_b"][k][rng] for k in runs)
        n_st = sum(states["s400"][k][rng] == states["s1200"][k][rng] == states["s1600"][k][rng] for k in runs)
        ck.num(f"{rng}: identical at end of A and end of B (rehearsal)", n_ab, "20", "section 4.1 (all runs)")
        ck.num(f"{rng}: identical at u400, u1200, u1600 (stitched)", n_st, "20", "section 4.1 (all runs)")
    ck.num("numpy legacy MT19937 position", pos[0] if len(pos) == 1 else float("nan"), "623", "section 4.1")
    ck.close()


def _line_of(rel: str, pattern: str, after: Optional[str] = None) -> int:
    """1-based line number of the first line matching ``pattern`` (after the first match of ``after``)."""
    lines = _text(rel).splitlines()
    start = 0
    if after:
        start = next(i for i, ln in enumerate(lines) if re.search(after, ln))
    return next(i + 1 for i, ln in enumerate(lines) if i >= start and re.search(pattern, ln))


# ----------------------------------------------------------------------------------------------
# T55 consolidation
# ----------------------------------------------------------------------------------------------

def build_t55(pack: C.Pack) -> None:
    """Consolidation: results-root checksums, the dirty-flag re-run, the seed inventory."""
    ck_csv = _csv(f"{CONS}/results_root_checksums.csv")
    dr = _js(f"{CONS}/dirty_rerun_compare.json")
    si = _csv(f"{CONS}/seed_inventory.csv")
    rr = _csv(f"{P4A}/run_records.csv")
    m_orig = _js("results/v2_pilots/pilot4_B/q60/seed10507/B2_mean_constant/manifest.json")
    m_new = _js("results/v2_pilots/pilot4_B_rerun_clean/q60/seed10507/B2_mean_constant/manifest.json")
    proto = _js(PROTO)
    rows: List[Dict[str, Any]] = []

    def add(block: str, quantity: str, value: Any, unit: str, source: str) -> None:
        rows.append({"block": block, "quantity": quantity, "value": value, "unit": unit, "source": source})

    b1 = "1 results-root checksums (original results/v2_pilots vs the canonical copy)"
    s1 = f"{CONS}/results_root_checksums.csv"
    idn = _b(ck_csv["identical"])
    inc = _b(ck_csv["in_canonical"])
    par = _b(ck_csv["pilot4_parent"])
    add(b1, "files in the original copy", int(len(ck_csv)), "count", s1)
    add(b1, "present in the canonical copy", int(inc.sum()), "count", s1)
    add(b1, "byte-identical (SHA-256)", int(idn.sum()), "count", s1)
    add(b1, "missing in the canonical copy", int((~inc).sum()), "count", s1)
    add(b1, "present but differing", int((inc & ~idn).sum()), "count", s1)
    add(b1, "Pilot 4 parent files (phaseA_ext state_u01200.pt / state_u01600.pt)", int(par.sum()), "count", s1)
    add(b1, "Pilot 4 parent files byte-identical", int((par & idn).sum()), "count", s1)
    add(b1, "files only in the canonical copy at comparison time", "4,027 (source: report text; the CSV lists only the "
        "original copy's files, the tool printed this count)", "count", f"{RPT_LOCK} section 1.2")
    add(b1, "tool", "tools/v2/compare_results_roots.py (SHA-256 of every file of the original root vs the same path "
        "under the canonical root)", "text", "tools/v2/compare_results_roots.py")
    b2 = "2 dirty-flag re-run (Pilot 4 2b q=60 seed 10507 B2_mean_constant)"
    s2 = f"{CONS}/dirty_rerun_compare.json"
    dirty = rr[(rr["family"] == "2b") & _b(rr["dirty"])].sort_values(["q", "seed", "arm"])
    add(b2, "runs launched with dirty = true (Pilot 4 2b)", f"{len(dirty)}: " +
        "; ".join(f"q{int(r.q)} s{int(r.seed)} {r.arm}" for r in dirty.itertuples(index=False)), "count",
        f"{P4A}/run_records.csv")
    add(b2, "original run: commit, dirty", f"{m_orig['git']['short']}, {m_orig['git']['dirty']}", "text",
        "results/v2_pilots/pilot4_B/q60/seed10507/B2_mean_constant/manifest.json")
    add(b2, "clean re-run: commit, dirty", f"{m_new['git']['short']}, {m_new['git']['dirty']}", "text",
        "results/v2_pilots/pilot4_B_rerun_clean/q60/seed10507/B2_mean_constant/manifest.json")
    for k, v in dr.items():
        add(b2, k, v, "flag" if isinstance(v, bool) else ("count" if isinstance(v, int) else "text"), s2)
    b3 = "3 seed inventory (proposed fresh block 20501-20520)"
    s3 = f"{CONS}/seed_inventory.csv"
    blk = proto["confirmation"]["seed_block"]
    lo, hi = int(min(blk)), int(max(blk))
    seeds = si["seed"].astype("int64")
    below = seeds[seeds < lo]
    above = seeds[seeds > hi]
    add(b3, "seed block", f"{lo}-{hi}", "seed", f"{PROTO} confirmation.seed_block")
    add(b3, "distinct recorded seed values", int(seeds.nunique()), "count", s3)
    add(b3, "recorded seeds inside the block (collisions)", int(seeds.between(lo, hi).sum()), "count", s3)
    add(b3, "recorded seeds within +-1000 of the block", int(seeds.between(lo - 1000, hi + 1000).sum()), "count", s3)
    add(b3, "nearest recorded seed below the block", int(below.max()), "seed", s3)
    add(b3, "nearest recorded seed above the block", int(above.min()), "seed", s3)
    add(b3, "smallest distance from the block", int(min(lo - below.max(), above.min() - hi)), "seed values", s3)
    add(b3, "smallest and largest recorded seed", f"{int(seeds.min())}, {int(seeds.max())}", "seed", s3)
    add(b3, "files scanned", "31,167 JSON and 3,070 CSV files in this repository (all worktrees), "
        "/home/fjiang4/tournament_experiment_upload_20260927, /home/fjiang4/TEL_PPO and the Phase 0 audit (source: "
        "report text; tools/v2/seed_inventory.py output)", "text", f"{RPT_LOCK} section 2.1")
    df = pd.DataFrame(rows)
    df = _union([df], ["block", "quantity", "value", "unit", "source"])
    srcs = [C.src(s1), C.src(s2), C.src(s3), C.src(f"{P4A}/run_records.csv"),
            C.src("results/v2_pilots/pilot4_B/q60/seed10507/B2_mean_constant/manifest.json"),
            C.src("results/v2_pilots/pilot4_B_rerun_clean/q60/seed10507/B2_mean_constant/manifest.json"),
            C.src(PROTO), C.src("tools/v2/compare_results_roots.py"), C.src("tools/v2/seed_inventory.py"),
            C.src(RPT_LOCK)]
    pack.table("T55", df, status="generated", sources=srcs, script=f"{MOD}:build_t55", tier="n/a",
               notes=("Counts computed from results_root_checksums.csv (10,034 rows) and seed_inventory.csv (263 rows); every "
                      "field of dirty_rerun_compare.json copied; the canonical-only file count and the scan scope of the "
                      "seed inventory are not in any data file (source: report text)."),
               caption="Consolidation before the lock: one development results root (checksums), the clean re-run that "
                       "resolves the dirty-flag runs, and the seed inventory behind the fresh seed block.",
               docs={"block": D_BLOCK, "quantity": D_QUANT, "value": D_VALUE, "unit": D_UNIT, "source": D_SRC})
    ck = _Checks(pack, "T55", RPT_LOCK, "section 1.2, 1.3 and 2.1 prose")
    ck.num("files of the original copy", len(ck_csv), "10034", "section 1.2")
    ck.num("byte-identical files", int(idn.sum()), "10034", "section 1.2")
    ck.num("missing files", int((~inc).sum()), "0", "section 1.2")
    ck.num("Pilot 4 parent files", int(par.sum()), "40", "section 1.2")
    ck.num("Pilot 4 parent files identical", int((par & idn).sum()), "40", "section 1.2")
    ck.num("dirty re-run: updates compared", int(dr["n_updates"]), "600", "section 1.3")
    ck.num("dirty re-run: weight exports compared", int(dr["n_weight_exports"]), "24", "section 1.3")
    ck.same("dirty re-run: ALL_IDENTICAL", bool(dr["ALL_IDENTICAL"]), True, "section 1.3 ('bit-identical')")
    ck.num("distinct recorded seed values", int(seeds.nunique()), "263", "section 2.1")
    ck.num("collisions with the block", int(seeds.between(lo, hi).sum()), "0", "section 2.1")
    ck.num("recorded seeds within +-1000 of the block", int(seeds.between(lo - 1000, hi + 1000).sum()), "0", "section 2.1")
    ck.num("Pilot 4 2b runs with dirty = true", len(dirty), "8", f"{RPT_P4} header table / section 8.1")
    ck.close()


# ----------------------------------------------------------------------------------------------
# T56 v1.1 global-RNG hardening
# ----------------------------------------------------------------------------------------------

def _count_tests(rel: str) -> Dict[str, Optional[int]]:
    """Collected test cases per test function of a pytest file (product of parametrize list lengths).

    Parametrize values must be literal lists/tuples or module-level names bound to one; otherwise
    the count of that function is None.
    """
    tree = ast.parse(_text(rel))
    consts = {t.id: len(node.value.elts) for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.value, (ast.List, ast.Tuple)) for t in node.targets if isinstance(t, ast.Name)}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            n: Optional[int] = 1
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "parametrize" and len(d.args) >= 2:
                    vals = d.args[1]
                    if isinstance(vals, (ast.List, ast.Tuple)) and n is not None:
                        n *= len(vals.elts)
                    elif isinstance(vals, ast.Name) and vals.id in consts and n is not None:
                        n *= consts[vals.id]
                    else:
                        n = None
            out[node.name] = n
    return out


def build_t56(pack: C.Pack) -> None:
    """v1.1 hardening of the process-global RNGs: seeding, reference, draws before it, assertions, tests."""
    proto = _js(PROTO)
    h = proto["global_rng_hardening"]
    trace = _js(f"{LK}/v1_1/global_rng_trace.json")
    rows: List[Dict[str, Any]] = []

    def add(block: str, quantity: str, value: Any, source: str, **kw: Any) -> None:
        rows.append({"block": block, "quantity": quantity, "value": value, "source": source, **kw})

    b1 = "1 seeding and the reference point"
    ln_seed = _line_of("run/run_v2_T2_locked.py", r"^\s+seed_globals\(seed\)", after=r"def run_pipeline")
    ln_run = _line_of("run/run_v2_T2_locked.py", r"run = Run\(cfg, out_dir\)", after=r"def run_pipeline")
    ln_ref = _line_of("run/run_v2_T2_locked.py", r"reference_before_first_A_update", after=r"def hook")
    ln_refresh = _line_of("run/run_v2_stagewise.py", r"agent\.refresh_snapshot\(\)", after=r"def run_phase")
    ln_hook = _line_of("run/run_v2_stagewise.py", r"self\.phase_start_hook\(phase\)", after=r"def run_phase")
    ln_loop = _line_of("run/run_v2_stagewise.py", r"while local < cap", after=r"def run_phase")
    add(b1, "seeding (protocol)", h["seeding"], f"{PROTO} global_rng_hardening.seeding")
    add(b1, "seeding (code)", f"run/run_v2_T2_locked.py:{ln_seed} seed_globals(seed) (torch.manual_seed, np.random.seed, "
        f"random.seed) is the first statement of run_pipeline after reading the seed, before Run is constructed "
        f"(line {ln_run})", "run/run_v2_T2_locked.py")
    add(b1, "reference point (protocol)", h["reference_state"], f"{PROTO} global_rng_hardening.reference_state")
    add(b1, "reference point (code)", f"Run.phase_start_hook('A') (run/run_v2_T2_locked.py:{ln_ref}) is called at "
        f"run/run_v2_stagewise.py:{ln_hook} inside run_phase, after the phase-entry snapshot refresh (line {ln_refresh}) and "
        f"immediately before the update loop (line {ln_loop})", "run/run_v2_T2_locked.py; run/run_v2_stagewise.py")
    add(b1, "assertion points (protocol)", ", ".join(h["assertion_points"]), f"{PROTO} global_rng_hardening.assertion_points")
    add(b1, "on violation (protocol)", h["on_violation"], f"{PROTO} global_rng_hardening.on_violation")
    add(b1, "digests (protocol)", h["digests"], f"{PROTO} global_rng_hardening.digests")
    # per-run records
    per = []
    sets = {}
    for study in ("rehearsal_v1_1", "confirmation"):
        ss = C.srcs(f"{LK}/{study}/q*/seed*/manifest.json", label=f"{LK}/{study}/q*/seed*/manifest.json")
        gs = C.srcs(f"{LK}/{study}/q*/seed*/gates.json", label=f"{LK}/{study}/q*/seed*/gates.json")
        sets[study] = (ss, gs)
        for s_m, s_g in zip(ss.files, gs.files):
            m, g = _js(s_m.path), _js(s_g.path)
            gr = m["global_rng"]
            ref = gr["reference_before_first_A_update"]["digests"]
            pts = list(g["global_rng"]["assertion_points"])
            per.append({
                "block": "4 assertions per run", "study": study, "q": int(m["q"]), "seed": int(m["seed"]),
                "global_rng_status": g["global_rng"]["status"], "n_assertion_points": len(pts),
                "n_violations": len(g["global_rng"]["violations"]),
                "seeded_with_run_seed": all(int(v) == int(m["seed"]) for v in gr["seeding"]["seeds"].values()),
                "torch_global_drawn_in_construction": gr["seeding"]["digests"]["torch_global"] !=
                gr["after_run_construction"]["digests"]["torch_global"],
                "numpy_global_drawn_in_construction": gr["seeding"]["digests"]["numpy_global"] !=
                gr["after_run_construction"]["digests"]["numpy_global"],
                "python_random_drawn_in_construction": gr["seeding"]["digests"]["python_random"] !=
                gr["after_run_construction"]["digests"]["python_random"],
                "any_draw_between_construction_and_reference": gr["after_run_construction"]["digests"] != ref,
                "assertion_digests_equal_reference": all(gr[p]["digests"] == ref for p in pts),
                "start_of_B_digest_equals_reference": gr["before_first_B_update"]["digests"] == ref,
                "manifest_global_rng_status": m["global_rng_status"]["status"],
                "ref_digest_torch_global": ref["torch_global"][:16], "ref_digest_numpy_global": ref["numpy_global"][:16],
                "ref_digest_python_random": ref["python_random"][:16]})
    per_df = pd.DataFrame(per)
    n = len(per_df)
    b2 = "2 draws between seeding and the reference point"
    for rng in GLOBAL_RNGS:
        k = int(per_df[f"{rng}_drawn_in_construction"].sum())
        add(b2, f"{rng}: drawn during Run construction (seeding digest != after_run_construction digest)", f"{k}/{n} runs",
            "manifest.json global_rng of the 20 + 40 runs")
    add(b2, "any RNG drawn between Run construction and the reference", f"{int(per_df['any_draw_between_construction_and_reference'].sum())}/{n} runs",
        "manifest.json global_rng of the 20 + 40 runs")
    for st in trace["steps"]:
        changed = [r for r in GLOBAL_RNGS if st.get(f"{r}_changed")]
        add(b2, f"trace (q={trace['q']}, seed {trace['seed']}): {st['step']}", "changes: " + (", ".join(changed) or "none"),
            f"{LK}/v1_1/global_rng_trace.json")
    add(b2, "call site of the torch-global draw", "PyTorch's default initialiser torch.nn.Linear.__init__ -> reset_parameters "
        "(kaiming-uniform weights, uniform bias) for each layer of BetaActor and Critic (agents/ppo_curriculum.py, built via "
        "CurriculumPPOv2 in Run.__init__); the weights are then re-initialised from the explicit torch generator, so "
        "training does not depend on this draw (source: report text)", f"{RPT_V11} section 2")
    b4 = "3 assertion summary"
    for qty, val in (("runs", n), ("global_rng status ok (gates.json)", int((per_df["global_rng_status"] == "ok").sum())),
                     ("manifest global_rng_status ok", int((per_df["manifest_global_rng_status"] == "ok").sum())),
                     ("assertion points per run", " ".join(map(str, sorted(per_df["n_assertion_points"].unique())))),
                     ("violations (total)", int(per_df["n_violations"].sum())),
                     ("runs whose 4 assertion digests equal the reference", int(per_df["assertion_digests_equal_reference"].sum())),
                     ("runs whose start-of-B digest equals the reference",
                      int(per_df["start_of_B_digest_equals_reference"].sum())),
                     ("runs seeded with the run seed (all three RNGs)", int(per_df["seeded_with_run_seed"].sum()))):
        add(b4, qty, val, "gates.json / manifest.json of the 20 v1.1 re-rehearsal + 40 confirmation runs")
    # injection tests
    b5 = "5 injection tests (tests/test_v2_locked.py)"
    counts = _count_tests("tests/test_v2_locked.py")
    if any(v is None for v in counts.values()):
        raise ValueError("tests/test_v2_locked.py: a parametrisation could not be counted")
    tl = _text(f"{LK}/rehearsal_v1_1_pytest.txt")
    summ = re.search(r"(\d+) failed, (\d+) passed, (\d+) xfailed", tl)
    failed_tests = re.findall(r"^FAILED (\S+)", tl, re.M)
    xf_n = _count_tests("tests/test_v2_verifier.py")["test_node_pmf_support_mismatch_documented"]
    ln_inj = _line_of("tests/test_v2_locked.py", r"def test_injected_global_draw_detected")
    src = re.search(r'parametrize\("rng_name", \[(.*?)\]\)', _text("tests/test_v2_locked.py")).group(1)
    names = re.findall(r'"(\w+)"', src)
    for nm in names:
        add(b5, f"test_injected_global_draw_detected[{nm}]",
            f"one draw from {nm} injected in the first call of smoothed_share (runs between the end of A and the after-G-A "
            f"check) in a reduced-budget in-process pipeline; asserts: status 'violation', run_pass False, outcome "
            f"'global_rng_violation', violations == {{{nm}}} at point 'after_G-A', exit code RNG_VIOLATION_EXIT = 5, status "
            f"done and state_end_B.pt written (the run finishes); result: passed", f"tests/test_v2_locked.py:{ln_inj}; "
            f"{LK}/rehearsal_v1_1_pytest.txt")
    add(b5, "test cases in tests/test_v2_locked.py (parametrisations counted)", int(sum(counts.values())),
        "tests/test_v2_locked.py (AST)")
    add(b5, "full suite at 95c000e (R6)", f"{summ.group(2)} passed, {summ.group(3)} xfailed, {summ.group(1)} failed",
        f"{LK}/rehearsal_v1_1_pytest.txt")
    add(b5, "failed tests at 95c000e", "; ".join(failed_tests), f"{LK}/rehearsal_v1_1_pytest.txt")
    add(b5, "xfailed tests", f"{xf_n} = test_node_pmf_support_mismatch_documented x 2 q (strict xfail, tests/test_v2_verifier.py); "
        "no other xfail marker in tests/", "tests/test_v2_verifier.py")
    add(b5, "how the injection result is known", "the -q log lists no per-test names; the only failure is the known "
        "registry test and the only xfails are the strict-xfail verifier test, so every test_v2_locked.py case, including "
        "the three injection cases, passed", f"{LK}/rehearsal_v1_1_pytest.txt")
    cols = ["block", "quantity", "value", "source", "study", "q", "seed", "global_rng_status", "n_assertion_points",
            "n_violations", "seeded_with_run_seed", "torch_global_drawn_in_construction",
            "numpy_global_drawn_in_construction", "python_random_drawn_in_construction",
            "any_draw_between_construction_and_reference", "assertion_digests_equal_reference",
            "start_of_B_digest_equals_reference", "manifest_global_rng_status", "ref_digest_torch_global",
            "ref_digest_numpy_global", "ref_digest_python_random"]
    per_df["source"] = per_df.apply(lambda r: f"{LK}/{r['study']}/q{r['q']}/seed{r['seed']}/manifest.json + gates.json",
                                    axis=1)
    df = _union([pd.DataFrame(rows), per_df], cols)
    df["_o"] = df["block"].map(lambda b: int(str(b)[0]))
    df = df.sort_values(["_o"], kind="stable").drop(columns="_o").reset_index(drop=True)
    srcs = [C.src(PROTO), C.src(f"{LK}/v1_1/global_rng_trace.json"), *[s for pair in sets.values() for s in pair],
            C.src(f"{LK}/rehearsal_v1_1_pytest.txt"), C.src("tests/test_v2_locked.py"), C.src("tests/test_v2_verifier.py"),
            C.src("run/run_v2_T2_locked.py"), C.src("run/run_v2_stagewise.py"), C.src(RPT_V11)]
    dig = "First 16 hex characters of the SHA-256 digest of the {} state at the reference point (full digest in manifest.json global_rng)"
    pack.table("T56", df, status="generated", sources=srcs, script=f"{MOD}:build_t56", tier="n/a",
               notes=("Blocks 1-2, 4-5: protocol text, code locations (line numbers found in the files at the base commit), "
                      "the trace JSON (one run, q=50 seed 10501), manifest digests of all 60 v1.1 runs, the test file and "
                      "the R6 pytest log; text marked source: report text. Block 3: one row per run (20 re-rehearsal seeds "
                      "10501-10510 and 40 confirmation seeds 20501-20520, both q) from manifest.json global_rng and "
                      "gates.json global_rng."),
               caption="v1.1 hardening of the three process-global RNGs (torch global, numpy legacy global, Python random): "
                       "seeding, reference point, draws before it, assertion results in all 60 runs, injection tests.",
               docs={"block": D_BLOCK, "quantity": D_QUANT, "value": D_VALUE, "source": D_SRC,
                     "study": "Locked study of the run: rehearsal_v1_1 (development seeds) or confirmation (fresh seeds)",
                     "q": "q (50 or 60)", "seed": "Run seed",
                     "global_rng_status": "gates.json global_rng.status (ok / violation)",
                     "n_assertion_points": "Number of assertion points listed in gates.json global_rng.assertion_points",
                     "n_violations": "Number of recorded violations (gates.json global_rng.violations)",
                     "seeded_with_run_seed": "Whether manifest global_rng.seeding.seeds equal the run seed for all three RNGs",
                     "torch_global_drawn_in_construction": "Whether the torch-global digest changed between seeding and "
                                                           "after Run construction",
                     "numpy_global_drawn_in_construction": "Whether the numpy-legacy digest changed between seeding and after "
                                                           "Run construction",
                     "python_random_drawn_in_construction": "Whether the Python-random digest changed between seeding and "
                                                            "after Run construction",
                     "any_draw_between_construction_and_reference": "Whether any digest changed between Run construction "
                                                                    "and the reference point",
                     "assertion_digests_equal_reference": "Whether all three digests equal the reference at every assertion "
                                                          "point",
                     "start_of_B_digest_equals_reference": "Whether the digests recorded before the first Phase B update equal "
                                                           "the reference",
                     "manifest_global_rng_status": "manifest.json global_rng_status.status",
                     "ref_digest_torch_global": dig.format("torch global"),
                     "ref_digest_numpy_global": dig.format("numpy legacy global"),
                     "ref_digest_python_random": dig.format("Python random")})
    ck = _Checks(pack, "T56", RPT_V11, "section 2 table and section 1.4 test counts")
    ck.num("runs with torch global drawn during Run construction", int(per_df["torch_global_drawn_in_construction"].sum()),
           "60", "section 2 table ('yes, in 60/60 runs')")
    for rng in ("numpy_global", "python_random"):
        ck.num(f"runs with {rng} drawn during Run construction", int(per_df[f"{rng}_drawn_in_construction"].sum()), "0",
               "section 2 table ('no (60/60)')")
    ck.num("runs with a draw between construction and the reference",
           int(per_df["any_draw_between_construction_and_reference"].sum()), "0", "section 2 table")
    ck.num("runs whose assertion digests equal the reference", int(per_df["assertion_digests_equal_reference"].sum()), "60",
           "section 2 'Assertions'")
    ck.num("test cases in tests/test_v2_locked.py", int(sum(counts.values())), "26", "section 1.4")
    ck.num("passed at 95c000e", int(summ.group(2)), "103", "section 1.4")
    ck.num("xfailed at 95c000e", int(summ.group(3)), "2", "section 1.4")
    ck.num("failed at 95c000e", int(summ.group(1)), "1", "section 1.4")
    ck.close()


# ----------------------------------------------------------------------------------------------
# T57 issue list
# ----------------------------------------------------------------------------------------------

CLIP_TOPICS = [("log-prob / likelihood", r"log.?prob|likelihood", re.I),
               ("action clamp configuration", r"action clamp|action_clamp", re.I),
               ("Beta mean clamp", r"μ = clamp|mu.?clamp|μ clamp|clamp\(sigmoid|clamp is not binding", re.I),
               ("verifier residuals reported without clamping", r"without clamping|never clamped", re.I),
               ("gradient-norm clipping", r"grad", re.I),
               ("closed-form g2 clip", r"g2|g₂|benchmark clip", re.I),
               ("PPO ratio clip", r"clipped surrogate|clip 0\.2|clip range|clip_eps|PPO clipping", re.I),
               ("PPO clip-fraction statistic", r"clip_frac|clip fraction|clip_median|clip median|KL|\bclip\b", 0)]


def _grep_lines(paths: Sequence[str], pattern: str, flags: int = re.I) -> List[Tuple[str, int, str]]:
    """(path, line number, line) of every line matching ``pattern``."""
    out = []
    for p in paths:
        for i, ln in enumerate(_text(p).splitlines(), 1):
            if re.search(pattern, ln, flags):
                out.append((p, i, ln.strip()))
    return out


def _rng_desync() -> List[Dict[str, Any]]:
    """First divergence of the five streams per arm pair, per study (rng_divergence.csv files)."""
    specs = [("Pilot 1 (sampled vs expected, from initialisation)", "results/v2_pilots/pilot1/analysis/rng_divergence.csv",
              None, 0),
             ("Pilot 2 (arm pairs B1-A, B2-A, B2-B1; branch u400)", f"{P2A}/rng_divergence.csv", None, 400),
             ("Pilot 3 (stochastic vs mean; branch u400)", f"{P3A}/rng_divergence.csv", None, 400),
             ("Pilot 4 2a (constant vs decay; branch u1200)", f"{P4A}/rng_divergence.csv", "2a", 1200),
             ("Pilot 4 2b (constant vs decay; branch u1600)", f"{P4A}/rng_divergence.csv", "2b", 1600)]
    out = []
    for name, rel, fam, branch in specs:
        d = _csv(rel)
        if fam:
            d = d[d["family"] == fam]
        rec = {"study": name, "source": rel, "pairs": int(len(d))}
        for s in ("env", "start", "minibatch", "learn", "opp"):
            v = d[s].astype(str)
            rec[f"{s}_never"] = int((v == "never").sum())
            num = pd.to_numeric(v[v != "never"]) - branch
            rec[f"{s}_after"] = num.to_numpy(dtype=float)
        first = np.minimum(pd.to_numeric(d["learn"].astype(str).replace("never", np.nan)),
                           pd.to_numeric(d["opp"].astype(str).replace("never", np.nan))) - branch
        rec["first_desync"] = first.to_numpy(dtype=float)
        out.append(rec)
    return out


def build_t57(pack: C.Pack) -> None:
    """Issue list (T=2): evidence recomputed from the data, status, impact, where it is addressed."""
    dw, g1 = _dw(), _g1()
    proto = _js(PROTO)
    thr_gf = next(c["threshold"] for c in proto["gates"]["G-F"]["all_must_hold"] if c["metric"] == "Gmax_full_over_dw")
    thr_s1 = float(proto["secondary"]["S1"]["criterion"]["threshold"])
    ext = _csv("results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv")
    rm = _csv(f"{CA}/reported_metrics.csv")
    pr = _csv(f"{CA}/per_run.csv")
    s1s = _csv(f"{CA}/s1_summary.csv")
    tw = _csv(f"{P4A}/repr_floor/three_way_peak_gap.csv")
    cusp = _csv(f"{LK}/cusp_diagnostic/thresholds_summary.csv")
    steps = _js(f"{LK}/cusp_diagnostic/rl_actor_steps.json")
    gp = _csv(f"{RA}/gates_per_run.csv")
    st2b = _csv(f"{P4A}/stability_2b.csv")
    psr = _csv(f"{P4A}/paired_summary_run_records.csv")
    cal = _csv(f"{LK}/calibration/calibration_locked.csv")
    cal1 = _csv("results/v2_pilots/phase1/calibration/calibration.csv")
    egm = _csv("results/v2_pilots/phase2_opening/effort_grid_membership.csv")
    fl = _js(f"{IB}/floors.json")
    sets, set_srcs = _band_sets(dw)
    d2 = _csv(f"{P2A}/decomposition_residual_band.csv")
    d3 = _csv(f"{P3A}/decomposition_residual_band.csv")
    cand = _csv(f"{P4A}/candidates_all.csv")
    probe = _csv("results/v2_pilots/phase1/invariants_probe.csv")
    dr = _js(f"{CONS}/dirty_rerun_compare.json")
    c1 = _csv(f"{RA}/check1_phaseA_vs_stitched.csv")
    fits = _csv(f"{P4A}/repr_floor/fits.csv")
    dmc = _csv("results/v2_pilots/dreach_mask_check/dreach_mask_check.csv")
    ft2 = _csv(f"{P2A}/final_table.csv")
    tlog = _text(f"{LK}/rehearsal_v1_1_pytest.txt")
    reports = sorted(p.replace(str(C.REPO) + "/", "") for p in glob.glob(str(C.REPO / "reports/v2/*.md")))
    docs_md = sorted(p.replace(str(C.REPO) + "/", "") for p in glob.glob(str(C.REPO / "docs/**/*.md"), recursive=True))
    proto_files = ["protocols/v2_T2_locked.md", "protocols/v2_T2_locked_v1_1.md", "protocols/v2_T2_locked.json", PROTO]
    rows: List[Dict[str, Any]] = []

    def add(no: int, issue: str, group: str, status: str, evidence: str, impact: str, addressed: str, src: str) -> None:
        rows.append({"issue_no": no, "issue": issue, "group": group, "status": status, "evidence": evidence,
                     "impact": impact, "addressed_in": addressed, "sources": src})

    REQ = "required by the request (T57)"
    ADD = "additional (from the reports)"
    # ---- 1 stage-2 peak bias
    e = []
    for q in QS:
        x = ext[(ext["q"] == q) & (ext["metric"] == "stage2_peak_rel_err_signed")].set_index("update")["median"]
        e.append(f"q={q}: {', '.join(f'u{u} {_f(x.loc[u], 3)}' for u in (400, 800, 1200, 1600))}")
    ev1 = "Phase A extension, median signed peak error (10 seeds): " + "; ".join(e) + ". "
    conf_pk, n05 = [], {}
    for q in QS:
        g = rm[rm["q"] == q]
        v = g["A_stage2_peak_rel_err_signed"].astype(float)
        n05[q] = int((v.abs() <= 0.05).sum())
        conf_pk.append(f"q={q}: median {_f(v.median(), 3)}, range [{_f(v.min(), 3)}, {_f(v.max(), 3)}], location-free "
                       f"median {_f(g['A_stage2_peak_locfree_rel_err'].median(), 3)}, smoothed-game share median "
                       f"{_f(g['A_smoothed_share_peak_gap_d0'].median(), 3)}")
    ev1 += "Confirmation, end of A (20 runs per q): " + "; ".join(conf_pk) + ". "
    tw_t = []
    for q in QS:
        t = tw[tw["q"] == q].set_index("quantity")
        tw_t.append(f"q={q}: RL {_f(t.loc['rl_peak_gap_d0', 'median'], 4)} vs smoothing-predicted "
                    f"{_f(t.loc['smoothing_predicted_gap_d0', 'median'], 4)} vs supervised-fit floor "
                    f"{_f(t.loc['supervised_floor_gap_d0', 'median'], 3)}")
    ev1 += "Three-way d = 0 gap at u1600 (medians, effort units): " + "; ".join(tw_t) + ". "
    cu = cusp.set_index(["q", "quantity"])
    ev1 += ("Cusp diagnostic: the exact-target supervised fit first has |peak error| < 0.05 after a median of " +
            " / ".join(f"{_f(cu.loc[(q, 'step_abs_peak_lt_0.05'), 'median'])} (q={q})" for q in QS) +
            " full-batch steps, with RMSE/e2*(0) " + " / ".join(_f(cu.loc[(q, 'rmse_at_peak_lt_0.05'), 'median'], 3)
                                                     for q in QS) +
            f" at that step; RL Phase A has {steps['50']['actor_steps_all_A']} actor steps "
            f"({steps['50']['actor_steps_last_400']} in the last 400 updates).")
    add(1, "stage-2 peak bias (e_hat_2(0) below e_2*(0))", REQ,
        "open (the part not predicted by action-noise smoothing is unexplained); reported, not gated",
        ev1, f"the 0929 target |stage-2 peak error| <= 0.05 is met in {n05[50]}/20 (q=50) and {n05[60]}/20 (q=60) "
             f"confirmation runs; the locked protocol reports the peak error and does not gate it ({PROTO} peak_error: "
             f"'{proto['peak_error']}')",
        f"{RPT_EXT} section 3; {RPT_P4} sections 1c.2, 1d and 7; {RPT_LOCK} section 5; {RPT_V11} section 4.5; "
        f"{RPT_SUM} open question 1",
        f"phaseA_ext/analysis/table_400_800_1200_1600.csv; {CA}/reported_metrics.csv; {P4A}/repr_floor/three_way_peak_gap.csv;"
        f" {LK}/cusp_diagnostic/thresholds_summary.csv; {LK}/cusp_diagnostic/rl_actor_steps.json")
    # ---- 2 stage-1 dispersion
    s1q = s1s.set_index("q")
    fails = pr[~_b(pr["S1_pass"])]
    reh_f = gp[gp["stage1_rel_err_abs_final"] > thr_s1]
    lc0 = {q: int((~_b(rm[rm["q"] == q]["dec_learning_contains_0"])).sum()) for q in QS}
    n5 = {q: int((pr[pr["q"] == q]["s1"] <= 0.05).sum()) for q in QS}
    sdq = st2b.set_index(["q", "arm"])
    ev2 = ("Confirmation (20 runs per q): SD of the signed stage-1 error " +
           " / ".join(f"{_f(s1q.loc[q, 'sd_signed'], 3)} (q={q})" for q in QS) + "; mean signed " +
           " / ".join(f"{_f(s1q.loc[q, 'mean_signed'], 2)} [{_f(s1q.loc[q, 'boot95_lo'], 2)}, {_f(s1q.loc[q, 'boot95_hi'], 2)}]"
                      for q in QS) +
           f" (bootstrap {int(s1q.loc[50, 'bootstrap_resamples'])} resamples, seed {int(s1q.loc[50, 'bootstrap_seed'])}); "
           f"S1 (|err| <= {_f(thr_s1)}) passes " + " / ".join(f"{int(s1q.loc[q, 'S1_pass'])}/{int(s1q.loc[q, 'n'])}"
                                                               for q in QS) +
           "; S1 failures: " + "; ".join(f"q={int(r.q)} seed {int(r.seed)} {_f(r.stage1_rel_err_signed, 4)}"
                                         for r in fails.itertuples(index=False)) +
           f". v1.0 rehearsal: |err| > {_f(thr_s1)} in {len(reh_f)}/20 runs (" +
           "; ".join(f"q={int(r.q)} seed {int(r.seed)} {_f(r.stage1_rel_err_abs_final, 3)}" for r in reh_f.itertuples(index=False)) +
           "). Learning term dominates: learning band excludes 0 in " +
           " / ".join(f"{lc0[q]}/20 (q={q})" for q in QS) + " confirmation runs; median |learning| " +
           " / ".join(_f(rm[rm['q'] == q]['dec_learning_rel'].abs().median(), 3) for q in QS) + " vs median |inherited| " +
           " / ".join(_f(rm[rm['q'] == q]['dec_inherited_rel'].abs().median(), 3) for q in QS) +
           ". Pilot 4 2b median within-run SD of e_hat_1(0) over the last 5 exports, constant vs decay: " +
           "; ".join(f"q={q} {_f(sdq.loc[(q, 'constant'), 'median_within_run_sd_last5'], 3)} vs "
                     f"{_f(sdq.loc[(q, 'decay'), 'median_within_run_sd_last5'], 3)} effort units" for q in QS) + ".")
    add(2, "stage-1 dispersion (run-to-run spread of e_hat_1(0))", REQ,
        "accepted for the verdict (stage-1 criterion moved from G-F to the secondary S1 in v1.1, decision D2/D3); open as "
        "an accuracy question", ev2,
        f"S1 fails in {int(s1q.loc[60, 'n'] - s1q.loc[60, 'S1_pass'])}/20 runs at q=60 and "
        f"{int(s1q.loc[50, 'n'] - s1q.loc[50, 'S1_pass'])}/20 at q=50 without affecting the primary verdict; the 0929 target "
        f"|stage-1 error| <= 0.05 is met in {n5[50]}/20 (q=50) and {n5[60]}/20 (q=60) confirmation runs",
        f"{RPT_P3} section 9; {RPT_P4} sections 1b, 2b, 7; {RPT_LOCK} section 4.3; {RPT_V11} sections 1.3 and 4.3; "
        f"{RPT_SUM} open question 2",
        f"{CA}/s1_summary.csv; {CA}/per_run.csv; {CA}/reported_metrics.csv; {RA}/gates_per_run.csv; {P4A}/stability_2b.csv")
    # ---- 3 q=60 final-tier floor
    a60 = cal[(cal["q"] == 60) & (cal["policy"] == "analytic_eq")].set_index("tier")
    a50 = cal[(cal["q"] == 50) & (cal["policy"] == "analytic_eq")]
    p1 = cal1[(cal1["q"] == 60) & (cal1["policy"] == "analytic_eq")].set_index("verifier_tier")
    ev3 = (f"Analytic equilibrium, Gmax_full/DW at lock time: q=60 final tier {_f(a60.loc['final', 'Gmax_full_over_dw'], 4)} "
           f"at (t={int(a60.loc['final', 'Gmax_full_t'])}, d={_f(a60.loc['final', 'Gmax_full_d'])}), development tier "
           f"{_f(a60.loc['development', 'Gmax_full_over_dw'], 3)}; q=50 max over both tiers "
           f"{_f(a50['Gmax_full_over_dw'].max(), 3)}. Phase 1: q=60 dev_2x {_f(p1.loc['dev_2x', 'Gmax_full_over_dw'], 4)}, "
           f"final {_f(p1.loc['final', 'Gmax_full_over_dw'], 4)}, both at (t=1, d=0). e_1* is off the effort grid at both q "
           "(distance to the nearest node: " + "; ".join(f"q={int(r.q)} {r.tier} {_f(r.dist_to_nearest, 3)}"
                                                           for r in egm.itertuples(index=False)) +
           f"). The same floor sets the q=60 band floor of the induced-target method: Delta_1(e1*; e2*)/DW = "
           f"{_f(float(fl['60']['calibration_delta1']) / dw, 4)} (q=50: {_f(float(fl['50']['calibration_delta1']) / dw)}, "
           f"floor 1e-12 DW).")
    add(3, "q=60 final-tier calibration floor at (t=1, d=0)", REQ,
        "unexplained (open): the cause is not established; e_1* off the effort grid does not distinguish q=60 from q=50 "
        "(source: report text)", ev3,
        f"{_f(float(a60.loc['final', 'Gmax_full_over_dw']) / thr_gf, 2)} of the G-F threshold {_f(thr_gf)} DW; it widens the "
        "q=60 induced bands (band floor 3.5e-7 DW instead of 1e-12 DW)",
        f"{RPT_P1} sections 4.2 and 6 (item 3); {RPT_OC} section 1b; {RPT_LOCK} section 3; {RPT_SUM} open question 6",
        f"{LK}/calibration/calibration_locked.csv; results/v2_pilots/phase1/calibration/calibration.csv; "
        f"results/v2_pilots/phase2_opening/effort_grid_membership.csv; {IB}/floors.json")
    # ---- 4 learn/opp stream desynchronisation
    parts = []
    for rec in _rng_desync():
        never = ", ".join(f"{s} {rec[f'{s}_never']}/{rec['pairs']}" for s in ("env", "start", "minibatch"))
        lr_, op_ = rec["learn_after"], rec["opp_after"]
        parts.append(f"{rec['study']}: never diverge: {never}; learn first differs in {rec['pairs'] - rec['learn_never']}/"
                     f"{rec['pairs']} pairs, {_f(np.min(lr_))}-{_f(np.max(lr_))} updates after the branch (median "
                     f"{_f(np.median(lr_))}); opp in {rec['pairs'] - rec['opp_never']}/{rec['pairs']}, "
                     f"{_f(np.min(op_))}-{_f(np.max(op_))} (median {_f(np.median(op_))})")
    add(4, "learn/opp action-stream desynchronisation between paired arms", REQ,
        "accepted (decision D4: no sampler change)", "; ".join(parts) + ". Cause (source: report text): numpy's "
        "rejection-based Beta sampler consumes a parameter-dependent number of draws, so the learn/opp streams drift apart "
        "once the arms' policies differ.",
        "paired comparisons control initialisation, shocks, starts and minibatches but not the action noise; an inverse-CDF "
        "sampler would align the streams but break C7 bit-exactness against the existing runner (source: report text)",
        f"{RPT_P2} section 3 item 4; {RPT_P3} section 8 item 3; {RPT_P4} section 2b; {RPT_SUM} open question 4",
        "results/v2_pilots/{pilot1,pilot2,pilot3,pilot4}/analysis/rng_divergence.csv")
    # ---- 5 non-contiguous bands and early rows outside the sweep
    nc = []
    for name, d in sets:
        n50 = int((~_b(d[d["q"] == 50]["band_contiguous"])).sum())
        n60 = int((~_b(d[d["q"] == 60]["band_contiguous"])).sum())
        nc.append(f"{name.split(':')[0].split(' (')[0]} q=50 {n50}/{int((d['q'] == 50).sum())}, q=60 {n60}/"
                  f"{int((d['q'] == 60).sum())}")
    o2 = d2[~_b(d2["e1_inside_sweep"])]
    o3 = d3[~_b(d3["e1_inside_sweep"])]
    key = ["q", "seed", "update", "source", "e1_at_0"]
    dup = o3[o3["arm"] == "B2_frozen_s1norm"].merge(o2[o2["arm"] == "B2_frozen_s1norm"][key], on=key)
    c2b = cand[cand["family"] == "2b"]
    out_locked = sum(int((~_b(d["e1_inside_sweep"])).sum()) for _, d in sets[3:])
    n_locked = sum(len(d) for _, d in sets[3:])
    o_all = pd.concat([o2, o3])
    ev5 = ("Non-contiguous bands: " + "; ".join(nc) + " (floors: 1e-12 DW at q=50, "
           f"{_f(float(fl['60']['floor']) / dw, 4)} DW at q=60). Rows with e_hat_1(0) outside the sweep: Pilot 2 "
           f"{len(o2)}/{len(d2)}, Pilot 3 {len(o3)}/{len(d3)} ({len(dup)} of them the stochastic-arm rows that equal Pilot 2 "
           f"B2 rows), all 25-update exports at u{int(o_all['update'].min())}-u{int(o_all['update'].max())} with e_hat_1 up to "
           f"{_f((o_all['e1_at_0'] / o_all['q'].map(g1)).max(), 3)} e1* (sweep upper end 1.7 e1*); Pilot 4 2b "
           f"{int((~_b(c2b['e1_inside_sweep'])).sum())}/{len(c2b)} (adaptive range); locked runs {out_locked}/{n_locked}.")
    add(5, "non-contiguous induced bands; early rows outside the sweep", REQ,
        "accepted (the interval uses the outermost band points, so it is conservative; e~_1 and its band do not depend on "
        "e_hat_1, so out-of-sweep rows stay defined) (source: report text)", ev5,
        "intervals are conservative where the band is not contiguous; no final-checkpoint row lies outside the sweep; the "
        "request that the sweep cover the current e_hat_1 is not met for the early rows listed (source: report text)",
        f"{RPT_P2} section 6; {RPT_P3} section 8 items 1-2; {RPT_P4} sections 2b and 8 item 5; {RPT_LOCK} section 4.4; "
        f"{RPT_SUM} open question 5",
        f"{IB}/*_bands.csv; {LK}/{{rehearsal,confirmation}}/q*/seed*/induced_band.json; {P2A}/decomposition_residual_band.csv; "
        f"{P3A}/decomposition_residual_band.csv; {P4A}/candidates_all.csv")
    # ---- 6 known test failure
    summ = re.search(r"(\d+) failed, (\d+) passed, (\d+) xfailed", tlog)
    failed = re.findall(r"^FAILED (\S+)", tlog, re.M)
    msg = re.search(r"AssertionError: (expected 15 Set-2 gradient runs, got \d+)", tlog)
    add(6, "known test failure: tests/test_registry_canonicalization.py", REQ,
        "open (pre-existing, left unchanged; recorded as a known issue of the locked protocol)",
        f"R6 suite at 95c000e: {summ.group(2)} passed, {summ.group(3)} xfailed, {summ.group(1)} failed; failed: "
        f"{'; '.join(failed)} ('{msg.group(1) if msg else 'UNKNOWN'}'); {PROTO} known_issues: '{proto['known_issues'][0]}'. "
        "The same single failure is reported at c92ee74 (77 passed), 4bd2214 (87 passed) and 431474d (103 passed) "
        "(source: report text).",
        "none on v2 per the reports: the test checks the paper results registry against data on disk and fails "
        "identically on main (source: report text)",
        f"{RPT_P4} header and section 8 item 6; {RPT_LOCK} section 1.4; {RPT_V11} sections 1.4 and 6 item 6",
        f"{LK}/rehearsal_v1_1_pytest.txt; {PROTO}")
    # ---- 7 Beta clip likelihood consistency
    hits = _grep_lines(reports, r"clip|clamp|log.?prob|likelihood")
    topics: Dict[str, List[str]] = {}
    for p, i, ln in hits:
        topic = next((t for t, pat, fl_ in CLIP_TOPICS if re.search(pat, ln, fl_)), "other")
        topics.setdefault(topic, []).append(f"{p.split('/')[-1]}:{i}")
    n_lik = len(_grep_lines(reports, r"log.?prob|likelihood"))
    doc_lik = {p: len(_grep_lines([p], r"log.?prob|likelihood")) for p in docs_md}
    doc_lik = {p: n for p, n in doc_lik.items() if n}
    prot_lik = {p: len(_grep_lines([p], r"log.?prob|likelihood")) for p in proto_files}
    ln_clip = _line_of("agents/ppo_curriculum.py", r"return np\.clip\(a, c, 1\.0 - c\)\.astype\(np\.float32\)")
    ln_lp = _line_of("agents/ppo_curriculum.py", r"return dist\.log_prob\(torch\.as_tensor\(actions\)\)")
    ln_upd = _line_of("agents/ppo_curriculum_v2.py", r"logp = dist\.log_prob\(ac\[rows\]\)")
    rec_txt = proto["records"]["50"]["action_sampling"]
    topic_txt = "; ".join(f"{t} {len(v)}" + (f" ({', '.join(v)})" if t not in ("PPO clip-fraction statistic",) else "")
                          for t, v in sorted(topics.items(), key=lambda kv: kv[0]))
    ev7 = (f"Mechanism (code): Beta draws are clipped to [1e-6, 1 - 1e-6] and cast to float32 "
           f"(agents/ppo_curriculum.py:{ln_clip}); the rollout log-prob is the Beta density at the stored clipped value "
           f"(agents/ppo_curriculum.py:{ln_lp}) and so is the update's new log-prob (agents/ppo_curriculum_v2.py:{ln_upd}); "
           f"the locked protocol records it as '{rec_txt}'. Grep of the {len(reports)} reports in reports/v2/: "
           f"'log.?prob|likelihood' -> {n_lik} lines; 'clip|clamp' lines by topic: {topic_txt}. Outside reports/v2: "
           f"protocols {', '.join(f'{p.split(chr(47))[-1]} {n}' for p, n in prot_lik.items())} matches of "
           f"'log.?prob|likelihood' (the action_sampling record only); docs/**/*.md ({len(docs_md)} files): "
           f"{', '.join(f'{p} {n}' for p, n in doc_lik.items()) or 'none'} (three-player and one-stage code, not T=2 v2).")
    add(7, "likelihood consistency of clipped Beta samples (0929 issue list)", REQ,
        "not addressed: no report discusses it (open)", ev7,
        "UNKNOWN: no run log records how often a Beta draw hits the clamp, and no report quantifies the effect on the PPO "
        "ratio or the policy gradient",
        "none (no report addresses it); phase0_audit.md section 3 (line 183) only lists the action clamp in the "
        "configuration",
        f"reports/v2/*.md ({len(reports)} files); docs/**/*.md ({len(docs_md)} files); protocols/v2_T2_locked*.{{md,json}}; "
        "agents/ppo_curriculum.py; agents/ppo_curriculum_v2.py")
    pack.unknown_value("T57", "impact of the clipped-Beta likelihood issue (issue 7)",
                       "no run log records the number of Beta draws that hit the [1e-6, 1-1e-6] clamp (v2_updates.csv and "
                       "train_history.json fields checked) and no report analyses it")
    # ---- 8 misspecified-policy discrimination
    tv = _text("tests/test_v2_verifier.py")
    disc = re.search(r"def test_analytic_floor_and_zero_policy_discriminates.*?(?=\n\n\n|\n@)", tv, re.S).group(0)
    asserts = [a.strip() for a in re.findall(r"assert (.+)", disc)]
    cands = re.findall(r'^\s+"(\w+)":', re.search(r"def _candidates\(spec\):.*?\n    \}", tv, re.S).group(0), re.M)
    z = cal[(cal["policy"] == "zero") & (cal["tier"] == "final")].set_index("q")
    ev8 = (f"Discrimination test: tests/test_v2_verifier.py::test_analytic_floor_and_zero_policy_discriminates asserts "
           f"{' and '.join(asserts)} (analytic vs zero effort, both q). Invariant test test_invariants uses {len(cands)} "
           f"candidates ({', '.join(cands)}) and asserts invariant residuals only. Calibration tables contain only the "
           f"policies {sorted(cal1['policy'].unique())} (Phase 1) and {sorted(cal['policy'].unique())} (lock). The Phase 1 "
           f"probe (invariants_probe.csv: {len(probe)} evaluations = {probe['candidate'].nunique()} synthetic candidates x "
           f"{probe['q'].nunique()} q x {probe['verifier_tier'].nunique()} tiers) is analysed only for invariant residuals "
           f"(phase1_verifier.md section 4.1). Zero-effort Gmax_full/DW (final tier): " +
           " / ".join(f"{_f(z.loc[q, 'Gmax_full_over_dw'], 4)} (q={q})" for q in QS) +
           f", against the analytic floor <= {_f(float(a60.loc['final', 'Gmax_full_over_dw']), 4)}.")
    add(8, "misspecified-policy discrimination tested only on the zero-effort policy", REQ, "open", ev8,
        "the 0929 target 'misspecified policies show clearly larger deviation' rests on one misspecified policy; the "
        "verifier's response to small or local misspecification has no test or reported measurement",
        f"{RPT_P1} sections 3, 4.1, 4.2; {RPT_LOCK} section 3",
        "tests/test_v2_verifier.py; results/v2_pilots/phase1/calibration/calibration.csv; "
        f"{LK}/calibration/calibration_locked.csv; results/v2_pilots/phase1/invariants_probe.csv")
    # ---- additional rows
    rr = _csv(f"{P4A}/run_records.csv")
    nd = int(_b(rr[rr["family"] == "2b"]["dirty"]).sum())
    add(9, "dirty flag in Pilot 4 2b manifests", ADD, "resolved",
        f"{nd} of 40 Pilot 4 2b manifests carry dirty = true (q=60 seeds 10507-10510, both arms; untracked analysis files at "
        f"launch); the clean re-run of q=60 seed 10507 B2_mean_constant (commit 5d50a9d, dirty = false) is bit-identical "
        f"(ALL_IDENTICAL = {dr['ALL_IDENTICAL']}: {dr['n_updates']} updates, {dr['n_weight_exports']} exports, end state)",
        "none on the results; one of the 8 runs was re-run, the other 7 rely on the same code argument (source: report text)",
        f"{RPT_P4} section 8 item 1; {RPT_LOCK} section 1.3; {RPT_SUM} open question 9",
        f"{P4A}/run_records.csv; {CONS}/dirty_rerun_compare.json")
    n_tr = int(c1[["actor_identical", "critic_identical", "opponent_identical", "opt_actor_identical",
                   "opt_critic_identical", "rng_minibatch_identical", "rng_streams_identical",
                   "torch_generator_identical", "exports_A_identical", "stage2_metrics_u1600_identical"]].all(axis=1).sum())
    n_gl = int(c1[["torch_global_rng_identical", "numpy_global_rng_identical", "python_random_identical"]].any(axis=1).sum())
    add(10, "v1.0 Check 1 fails on its literal criterion", ADD,
        "accepted (owner decision D1); hardened in v1.1",
        f"training state identical in {n_tr}/20 runs; the three process-global RNG states identical in {n_gl}/20; they do "
        "not advance within a run or along the stitched path (T54); v1.1 seeds and asserts them: status ok in 60/60 runs "
        "(T56). The snapshot-refresh counter also differs (82 vs 84 in 20/20; two extra phase-entry refreshes of the "
        "stitched path), a field the report's table does not list (T54)",
        "none on training (all training-relevant fields identical); the counter difference is not discussed in any report",
        f"{RPT_LOCK} section 4.1 and addendum; {RPT_V11} section 2",
        f"{RA}/check1_phaseA_vs_stitched.csv; saved full states (T54); manifests and gates (T56)")
    add(11, "supervised-fit representation floor is an upper bound", ADD, "open (accepted as an upper bound)",
        f"all {len(fits)} fits stopped at the step cap ({sorted(fits['stop'].unique())}, steps "
        f"{sorted(int(s) for s in fits['steps'].unique())}); the plateau rule never fired and the loss was still falling "
        "(source: report text)",
        "the reported floor (<= 0.06% of e2*(0) at the peak) bounds the representation error from above only",
        f"{RPT_P4} sections 1d and 8 item 2; {RPT_LOCK} section 5", f"{P4A}/repr_floor/fits.csv")
    ck2, _ = _pilot2_checkpoints()
    r60a = ck2[(ck2["q"] == 60) & (ck2["arm"] == "A_joint")]["induced_residual"]
    add(12, "superseded bracketing + Brent induced-target solver ill-conditioned at q=60", ADD,
        "resolved (replaced by residual minimisation on the final tier, decision D2, commit cd760fd)",
        f"|BR(e~_1) - e~_1| > 1e-3 effort units in {_f(100 * float((r60a > 1e-3).mean()), 3)}% of the q=60 A_joint "
        f"checkpoints (median {_f(r60a.median(), 3)}); {int((ck2['induced_n_sign_changes'] >= 2).sum())}/{len(ck2)} "
        "checkpoints with two sign changes (T48 block 5)",
        "superseded values remain only in Pilot 2 Appendix S and the Pilot 2 checkpoint CSVs",
        f"{RPT_P2} section 3 items 1-3, section 6, Appendix S", "results/v2_pilots/pilot2/q*/seed*/*/v2_checkpoints.csv")
    xd = dmc["dreach_official_minus_refmask_over_dw"].astype(float)
    add(13, "dReach reach mask: closed vs open support boundary", ADD, "accepted (no change to dReach)",
        f"official minus reference-mask dReach > 0 in {int((xd > 0).sum())}/{len(dmc)} evaluations, by at most "
        f"{_f(xd.max(), 4)} DW; never negative ({int((xd < 0).sum())})",
        "below the 0.01 DW threshold but not negligible relative to it (source: report text)",
        f"{RPT_DR}", "results/v2_pilots/dreach_mask_check/dreach_mask_check.csv")
    med = ft2.groupby(["q", "arm"])[["stage2_peak_rel_err_signed", "stage2_tail_mean"]].median()
    add(14, "joint training improves the stage-2 peak at the cost of the tail; interaction with a longer Phase A untested",
        ADD, "open (joint training dropped from v2, decision D3)",
        "Pilot 2 medians at u1000, A_joint vs B2: " + "; ".join(
            f"q={q} peak {_f(med.loc[(q, 'A_joint'), 'stage2_peak_rel_err_signed'], 3)} vs "
            f"{_f(med.loc[(q, 'B2_frozen_s1norm'), 'stage2_peak_rel_err_signed'], 3)}, tail mean "
            f"{_f(med.loc[(q, 'A_joint'), 'stage2_tail_mean'], 3)} vs {_f(med.loc[(q, 'B2_frozen_s1norm'), 'stage2_tail_mean'], 3)} "
            "effort units" for q in QS),
        "not pursued under the locked protocol (frozen stage 2)", f"{RPT_P2} section 4; {RPT_SUM} open question 3",
        f"{P2A}/final_table.csv")
    add(15, "Phase C and T=3", ADD, "out of scope (not run)", "Phase C dropped from v2 (decision D3); T=3 not started "
        "(source: report text)", "none on the T=2 results", f"{RPT_SUM} open question 7; {RPT_P4} header (decisions)",
        RPT_SUM)
    df = pd.DataFrame(rows)
    srcs = [C.src(PROTO), C.src("results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv"),
            C.src(f"{CA}/reported_metrics.csv"), C.src(f"{CA}/per_run.csv"), C.src(f"{CA}/s1_summary.csv"),
            C.src(f"{P4A}/repr_floor/three_way_peak_gap.csv"), C.src(f"{LK}/cusp_diagnostic/thresholds_summary.csv"),
            C.src(f"{LK}/cusp_diagnostic/rl_actor_steps.json"), C.src(f"{RA}/gates_per_run.csv"),
            C.src(f"{P4A}/stability_2b.csv"), C.src(f"{P4A}/paired_summary_run_records.csv"),
            C.src(f"{LK}/calibration/calibration_locked.csv"), C.src("results/v2_pilots/phase1/calibration/calibration.csv"),
            C.src("results/v2_pilots/phase2_opening/effort_grid_membership.csv"), C.src(f"{IB}/floors.json"), *set_srcs,
            C.src(f"{P2A}/decomposition_residual_band.csv"), C.src(f"{P3A}/decomposition_residual_band.csv"),
            C.src(f"{P4A}/candidates_all.csv"), C.src("results/v2_pilots/phase1/invariants_probe.csv"),
            C.src(f"{CONS}/dirty_rerun_compare.json"), C.src(f"{RA}/check1_phaseA_vs_stitched.csv"),
            C.src(f"{P4A}/repr_floor/fits.csv"), C.src("results/v2_pilots/dreach_mask_check/dreach_mask_check.csv"),
            C.src(f"{P2A}/final_table.csv"), C.src(f"{P4A}/run_records.csv"), C.src(f"{LK}/rehearsal_v1_1_pytest.txt"),
            C.src("results/v2_pilots/pilot1/analysis/rng_divergence.csv"), C.src(f"{P2A}/rng_divergence.csv"),
            C.src(f"{P3A}/rng_divergence.csv"), C.src(f"{P4A}/rng_divergence.csv"),
            C.src("tests/test_v2_verifier.py"), C.src("agents/ppo_curriculum.py"), C.src("agents/ppo_curriculum_v2.py"),
            *[C.src(p) for p in reports], *[C.src(p) for p in docs_md], *[C.src(p) for p in proto_files[:-1]]]
    pack.table("T57", df, status="generated", sources=srcs, script=f"{MOD}:build_t57", tier="n/a",
               notes=("One row per issue: the 8 issues named in the request (group 'required') and 7 further items found in "
                      "the reports (group 'additional'). Every number in `evidence` and `impact` is recomputed by the builder "
                      "from the files in `sources`; statements copied from reports are marked 'source: report text'. The "
                      "grep evidence of issue 7 is computed over every reports/v2/*.md, docs/**/*.md and the v2 protocol "
                      "files at build time. Tiers: calibration values final/development as stated; recovery metrics "
                      "tier-independent."),
               caption="Known issues and open questions of the T=2 v2 work: evidence, status and impact.",
               docs={"issue_no": "Issue number (1-8: the issues listed in the request; 9-15: additional items)",
                     "issue": "Short name of the issue", "group": "required by the request, or additional",
                     "status": "open / resolved / accepted / unexplained / not addressed / out of scope, with the reason",
                     "evidence": "Evidence with numbers recomputed from the source files (report text marked)",
                     "impact": "Effect on the results or the targets (UNKNOWN when no data or analysis exists)",
                     "addressed_in": "Report sections that discuss the issue", "sources": "Files the evidence comes from"})
    _t57_checks(pack, rm, pr, s1s, ext, tw, cusp, steps, gp, cal, fl, dw)


def _t57_checks(pack: C.Pack, rm: pd.DataFrame, pr: pd.DataFrame, s1s: pd.DataFrame, ext: pd.DataFrame,
                tw: pd.DataFrame, cusp: pd.DataFrame, steps: Dict[str, Any], gp: pd.DataFrame, cal: pd.DataFrame,
                fl: Dict[str, Any], dw: float) -> None:
    """Cross-checks of the T57 evidence numbers against the reports."""
    ck = _Checks(pack, "T57", RPT_V11, "S1 summary and S1 failures (section 4.3)")
    s1q = s1s.set_index("q")
    for q, sd_c, mean_c in ((50, "0.0494", "-0.0051"), (60, "0.0625", "+0.0077")):
        ck.num(f"q{q} SD of the signed stage-1 error", float(s1q.loc[q, "sd_signed"]), sd_c, "section 4.3")
        ck.num(f"q{q} mean signed stage-1 error", float(s1q.loc[q, "mean_signed"]), mean_c, "section 4.3")
    fails = pr[~_b(pr["S1_pass"])].sort_values("seed")
    ck.same("S1 failures (q, seed)", [(int(r.q), int(r.seed)) for r in fails.itertuples(index=False)],
            [(60, 20510), (60, 20515)], "section 4.3")
    for r, cell in zip(fails.itertuples(index=False), ("+0.153", "-0.113")):
        ck.num(f"S1 failure q{int(r.q)} seed {int(r.seed)}", float(r.stage1_rel_err_signed), cell, "section 4.3")
    ck.close()
    ckp = _Checks(pack, "T57", RPT_P4, "peak bias evidence (sections 1c, 1d)")
    for q, cell in ((50, "-0.06835"), (60, "-0.06003")):
        x = ext[(ext["q"] == q) & (ext["metric"] == "stage2_peak_rel_err_signed") & (ext["update"] == 1600)]["median"]
        ckp.num(f"q{q} extension u1600 median signed peak error", float(x.iloc[0]), cell, "section 1c consistency check")
    for q, rl_c, sm_c, fl_c in ((50, "4.784", "2.159", "0.00133"), (60, "3.502", "1.581", "0.0348")):
        t = tw[tw["q"] == q].set_index("quantity")
        ckp.num(f"q{q} RL peak gap", float(t.loc["rl_peak_gap_d0", "median"]), rl_c, "section 1d three-way table")
        ckp.num(f"q{q} smoothing-predicted gap", float(t.loc["smoothing_predicted_gap_d0", "median"]), sm_c,
                "section 1d three-way table")
        ckp.num(f"q{q} supervised floor gap", float(t.loc["supervised_floor_gap_d0", "median"]), fl_c,
                "section 1d three-way table")
    ckp.close()
    ckl = _Checks(pack, "T57", RPT_LOCK, "stage-1 failures, calibration floor, cusp steps (sections 3, 4.3, 5)")
    reh_f = gp[gp["stage1_rel_err_abs_final"] > 0.10].sort_values(["q", "seed"])
    for r, cell in zip(reh_f.itertuples(index=False), ("0.103", "0.126", "0.113")):
        ckl.num(f"v1.0 rehearsal |stage-1 err| q{int(r.q)} seed {int(r.seed)}", float(r.stage1_rel_err_abs_final), cell,
                "section 4.3")
    a60 = cal[(cal["q"] == 60) & (cal["policy"] == "analytic_eq") & (cal["tier"] == "final")].iloc[0]
    ckl.num("q60 analytic final-tier floor", float(a60["Gmax_full_over_dw"]), "3.51e-7", "section 3")
    ckl.num("q60 band floor (calibration Delta_1 / DW)", float(fl["60"]["calibration_delta1"]) / dw, "3.51e-7",
            "section 3 (same floor)")
    cu = cusp.set_index(["q", "quantity"])
    for q, cell in ((50, "8500"), (60, "9500")):
        ckl.num(f"q{q} steps to |peak| < 0.05 (median)", float(cu.loc[(q, "step_abs_peak_lt_0.05"), "median"]), cell,
                "section 5")
    ckl.num("RL actor steps in Phase A", float(steps["50"]["actor_steps_all_A"]), "32000", "section 5")
    ckl.num("RL actor steps in the last 400 updates", float(steps["50"]["actor_steps_last_400"]), "8000", "section 5")
    ckl.close()
    ckr = _Checks(pack, "T57", RPT_P2, "learn/opp desynchronisation ranges (prose)")
    by = {r["study"].split(" (")[0]: r for r in _rng_desync()}
    fd = by["Pilot 2"]["first_desync"]
    ckr.num("Pilot 2: earliest learn/opp divergence after u400", float(np.nanmin(fd)), "3", "section 3 item 4")
    late = max(float(np.nanmax(by["Pilot 2"]["learn_after"])), float(np.nanmax(by["Pilot 2"]["opp_after"])))
    ckr.num("Pilot 2: latest first divergence of learn or opp after u400", late, "164", "section 3 item 4")
    ckr.close()
    ck3 = _Checks(pack, "T57", RPT_P3, "learn/opp desynchronisation ranges (prose)")
    fd = by["Pilot 3"]["first_desync"]
    ck3.num("Pilot 3: earliest learn/opp divergence after u400", float(np.nanmin(fd)), "4", "section 8 item 3")
    late = max(float(np.nanmax(by["Pilot 3"]["learn_after"])), float(np.nanmax(by["Pilot 3"]["opp_after"])))
    ck3.num("Pilot 3: latest first divergence of learn or opp after u400", late, "134", "section 8 item 3")
    ck3.close()
    ck4 = _Checks(pack, "T57", RPT_P4, "learn/opp desynchronisation in 2a (prose)")
    fd = by["Pilot 4 2a"]["first_desync"]
    ck4.num("2a: earliest learn/opp divergence after u1200", float(np.nanmin(fd)), "4", "section 2b RNG divergence note")
    late = max(float(np.nanmax(by["Pilot 4 2a"]["learn_after"])), float(np.nanmax(by["Pilot 4 2a"]["opp_after"])))
    ck4.num("2a: latest first divergence of learn or opp after u1200", late, "40", "section 2b RNG divergence note")
    ck4.close()


# ----------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------

def build() -> None:
    """Build every item of this module."""
    pack = C.Pack("sec_stage1")
    build_t48(pack)
    build_f22(pack)
    build_t49(pack)
    build_t50(pack)
    build_f23(pack)
    build_t51(pack)
    build_f24(pack)
    build_t54(pack)
    build_t55(pack)
    build_t56(pack)
    build_t57(pack)
    pack.save_fragment()
