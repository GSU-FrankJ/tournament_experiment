"""Part I, sections 2 and 3 of the T=2 v2 report pack: setting, baseline audit, metric
definitions, verifier tiers, invariant checks, dReach mask check, benchmark consistency and
calibration.

Items: T04, F01, T05, T06, T07, T08, T09, T10, T11, T12, F02.

No training, no forward pass and no verifier evaluation: the repository's closed-form functions
and grid constructors are evaluated, saved CSV / JSON / NPZ records are read, and the invariant
residuals of the locked gate evaluations are recomputed from their saved arrays. Report values
are compared at the report's own precision, with ``Pack.crosscheck`` where the report table is
plain and with :class:`XCheck` where cells carry bold marks, ``a / b`` pairs or prose.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import common as C
import studies as S
import style

if str(C.REPO) not in sys.path:
    sys.path.append(str(C.REPO))

from envs.curriculum_env import GameSpec  # noqa: E402
from utils.dp_br_verifier import (DEV_CONFIG, FINAL_CONFIG, VerifierConfig, effort_grid,
                                  # noqa: E402
                                  gl_shock_rule, stage_grid)
from utils.theory_multistage import f_xi, g1_two_stage, g2_two_stage, q_crit  # noqa: E402
from utils.v2_metrics import symmetric_grid  # noqa: E402

MOD = "sec_p0p1.py"
QS = (50, 60)
PROTO = "protocols/v2_T2_locked_v1_1.json"
P1 = "results/v2_pilots/phase1"
CAL_P1 = f"{P1}/calibration/calibration.csv"
CAL_LK = "results/v2_T2_locked/calibration/calibration_locked.csv"
CONF = "results/v2_T2_locked/confirmation"
LEGACY = {50: "experiments/two_stage_q50_precision_20260924/manifest.json",
          60: "experiments/two_stage_confirmation_T2_20260922/manifest.json"}
LEGACY_RUN50 = "tel_q50_s10231"   # Phase 0 section 2: q=50 record; q=60: first q=60 record
DEV_2X = VerifierConfig("dev_2x", state_step=DEV_CONFIG.state_step / 2,
                        effort_step=DEV_CONFIG.effort_step / 2, gl_half=DEV_CONFIG.gl_half)
TIERS = (DEV_CONFIG, DEV_2X, FINAL_CONFIG)
R_P0 = "reports/v2/phase0_audit.md"
R_P1 = "reports/v2/phase1_verifier.md"
R_OPEN = "reports/v2/phase2_opening_checks.md"
R_INFRA = "reports/v2/phase2_infra.md"
R_DR = "reports/v2/dreach_reach_mask_check.md"
R_LOCK = "reports/v2/protocol_lock_and_rehearsal.md"
R_SUM = "reports/v2/summary.md"


# ----------------------------------------------------------------------------------------------
# small readers and formatting
# ----------------------------------------------------------------------------------------------

_JSON: Dict[str, Any] = {}


def _json(rel: str) -> Any:
    """Parsed JSON file (cached)."""
    if rel not in _JSON:
        _JSON[rel] = json.loads(C.abspath(rel).read_text(encoding="utf-8"))
    return _JSON[rel]


def _csv(rel: str) -> pd.DataFrame:
    """CSV with round-trip float parsing (pandas' default C parser can be off by one ulp)."""
    return S.read_csv(rel, float_precision="round_trip")


def _proto() -> Dict[str, Any]:
    """The locked v1.1 protocol JSON."""
    return _json(PROTO)


def _spec(q: int) -> GameSpec:
    """Game of the locked protocol record for ``q``."""
    return GameSpec(**_proto()["records"][str(q)]["game"])


def _legacy_record(q: int) -> Tuple[Dict[str, Any], str]:
    """Archived as-run record of the latest legacy T=2 runs (Phase 0 section 2) and its run name."""
    runs = _json(LEGACY[q])["runs"]
    if q == 50:
        rec = [r for r in runs if r["run"] == LEGACY_RUN50][0]
    else:
        rec = [r for r in runs if float(r["q"]) == 60.0][0]
    return rec, rec["run"]


def _txt(v: Any) -> str:
    """Text form of a configuration value (JSON for containers, repr for floats)."""
    if isinstance(v, bool) or v is None or isinstance(v, (list, dict)):
        return json.dumps(v)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Nested dict -> {'a.b.c': value}; empty dicts are kept as values."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _norm(text: str) -> str:
    """Report text with unicode minus, thin spaces, bold marks and backticks normalized."""
    return (text.replace("\u2212", "-").replace("\u2009", " ").replace("\u202f", " ")
            .replace("**", "").replace("`", ""))


_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _nums(cell: str) -> List[str]:
    """Numeric substrings of a report cell (after :func:`_norm`)."""
    return re.findall(_NUM, _norm(cell).replace(",", ""))


# ----------------------------------------------------------------------------------------------
# code locations (line numbers read from the files at build time)
# ----------------------------------------------------------------------------------------------

_SRC: Dict[str, List[str]] = {}


def _lines(rel: str) -> List[str]:
    if rel not in _SRC:
        _SRC[rel] = C.abspath(rel).read_text(encoding="utf-8").splitlines()
    return _SRC[rel]


def _span(rel: str, name: str) -> Tuple[int, int]:
    """(first, last) line of function ``name`` or method ``Class.name`` in a repo file."""
    tree = ast.parse("\n".join(_lines(rel)))
    parts = name.split(".")
    nodes: Sequence[ast.AST] = tree.body
    found: Optional[ast.AST] = None
    for p in parts:
        found = None
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == p:
                found = n
                break
        if found is None:
            raise KeyError(f"{rel}: {name} not found")
        nodes = found.body
    return found.lineno, found.end_lineno


def _line(rel: str, pattern: str, within: Optional[str] = None) -> int:
    """Line number of the first line containing ``pattern`` (inside ``within`` if given)."""
    lo, hi = _span(rel, within) if within else (1, len(_lines(rel)))
    for i in range(lo, hi + 1):
        if pattern in _lines(rel)[i - 1]:
            return i
    raise KeyError(f"{rel}: '{pattern}' not found in {within or 'file'}")


def _assign(rel: str, name: str) -> Tuple[Any, int, int]:
    """(literal value, first line, last line) of the top-level assignment ``name = <literal>`` in a repo file."""
    for n in ast.parse("\n".join(_lines(rel))).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value), n.lineno, n.end_lineno
    raise KeyError(f"{rel}: assignment {name} not found")


def loc(rel: str, name: str, *patterns: str) -> str:
    """``file:function (Lfirst-last; Lx: pattern ...)`` with line numbers from the file itself."""
    a, b = _span(rel, name)
    extra = [f"L{_line(rel, p, name)}" for p in patterns]
    return f"{rel}:{name} (L{a}-{b}" + ("; " + ", ".join(extra) if extra else "") + ")"


# ----------------------------------------------------------------------------------------------
# cross-checks of values that Pack.crosscheck cannot parse
# ----------------------------------------------------------------------------------------------

class XCheck:
    """Comparisons of pack values with report cells or prose, at the report's displayed precision.

    Every comparison is counted; a difference beyond the report's rounding is recorded with
    ``pack.mismatch``; ``done()`` adds one summary row to the pack's cross-check list (the same
    record ``Pack.crosscheck`` writes). Report snippets that cannot be located count as unmatched.
    """

    def __init__(self, pack: C.Pack, item: str, report: str, label: str):
        self.pack, self.item, self.report, self.label = pack, item, report, label
        self.text = _norm(C.abspath(report).read_text(encoding="utf-8"))
        self.n_cmp = self.n_bad = self.n_unm = 0
        self.tables: set = set()

    def cmp(self, quantity: str, value: float, report_str: str, where: str = "",
            comment: str = "") -> bool:
        """Equality at the report's precision."""
        self.n_cmp += 1
        ok = C.consistent(float(value), report_str)
        if not ok:
            self.n_bad += 1
            self.pack.mismatch(self.item, quantity, value,
                               f"{self.report} ({where})" if where else self.report,
                               report_str, comment)
        return ok

    def bound(self, quantity: str, value: float, report_str: str, where: str = "") -> bool:
        """``value <= report bound`` (the report states an upper bound)."""
        self.n_cmp += 1
        b = C.parse_num(report_str)
        ok = b is not None and float(value) <= b * (1 + 1e-12) + 1e-300
        if not ok:
            self.n_bad += 1
            self.pack.mismatch(self.item, quantity + " (upper bound)", value,
                               f"{self.report} ({where})" if where else self.report, f"<= {report_str}")
        return ok

    def same(self, quantity: str, value: str, report_str: str, where: str = "") -> bool:
        """Exact text equality (labels, version strings)."""
        self.n_cmp += 1
        ok = str(value).strip() == str(report_str).strip()
        if not ok:
            self.n_bad += 1
            self.pack.mismatch(self.item, quantity, value,
                               f"{self.report} ({where})" if where else self.report,
                               report_str)
        return ok

    def find(self, pattern: str) -> Optional[re.Match]:
        """First regex match in the normalized report text (counted as unmatched if absent)."""
        m = re.search(pattern, self.text, flags=re.S)
        if m is None:
            self.n_unm += 1
            print(f"[{MOD}] WARNING {self.item}: pattern not found "
                  f"in {self.report}: {pattern!r}", flush=True)
        return m

    def findall(self, pattern: str) -> List[Any]:
        """All regex matches (unmatched if none)."""
        ms = re.findall(pattern, self.text, flags=re.S)
        if not ms:
            self.n_unm += 1
            print(f"[{MOD}] WARNING {self.item}: pattern not found "
                  f"in {self.report}: {pattern!r}", flush=True)
        return ms

    def tables_with(self, header_has: Sequence[str], heading_has: str = "") -> List[Dict[str, Any]]:
        """Markdown tables of the report whose header contains every string of ``header_has``."""
        out = []
        for t in C.parse_md_tables(self.report):
            hdr = [_norm(h) for h in t["header"]]
            if all(any(h0 in h for h in hdr) for h0 in header_has) and heading_has.lower() in t["heading"].lower():
                out.append({**t, "header": hdr, "rows": [[_norm(c) for c in r] for r in t["rows"]]})
                self.tables.add(t["line"])
        if not out:
            self.n_unm += 1
            print(f"[{MOD}] WARNING {self.item}: no table "
                  f"with {header_has} in {self.report}", flush=True)
        return out

    def unmatched(self, what: str) -> None:
        """Count a report row that could not be matched to a pack value."""
        self.n_unm += 1
        print(f"[{MOD}] WARNING {self.item}: unmatched report row "
              f"in {self.report}: {what}", flush=True)

    def done(self) -> Dict[str, Any]:
        """Register the summary of this check with the pack."""
        res = {"item": self.item, "report": self.report, "label": self.label,
               "n_tables": len(self.tables),
               "n_compared": self.n_cmp, "n_mismatch": self.n_bad, "n_unmatched_rows": self.n_unm}
        self.pack.crosschecks.append(res)
        return res


# ----------------------------------------------------------------------------------------------
# T04 parameters and derived constants
# ----------------------------------------------------------------------------------------------

def _t04_values(q: int) -> Dict[str, Any]:
    """Every T04 quantity for one q, computed with the repository's own functions."""
    rec = _proto()["records"][str(q)]
    sp = _spec(q)
    g1 = g1_two_stage(sp.q, sp.w_h, sp.w_l, sp.k)
    g20 = float(g2_two_stage(np.zeros(1), sp.q, sp.w_h, sp.w_l, sp.k, sp.e_max)[0])
    rg = symmetric_grid(sp.domain_half(2), float(rec["protocol"]["recovery_step"]))
    two_q = 2.0 * float(sp.q)
    out = {
        "w_h": sp.w_h, "w_l": sp.w_l, "dw": sp.dw, "k": sp.k, "e_min": sp.e_min, "e_max": sp.e_max, "T": sp.T,
        "q": sp.q,
        "shock": (f"eps_i,t ~ U(-{q}, {q}) i.i.d. per player and stage; xi = eps_i - eps_j ~ "
                  f"Triangular(-{2 * q}, "
                  f"{2 * q}) with density f_xi(x) = (2q - |x|)/(4q^2); d_(t+1) = d_t + e_i - e_j + "
                  f"xi_t"),
        "f0": float(f_xi(np.zeros(1), sp.q)[0]), "B": sp.B, "e1": g1, "e2": g20,
        "r": sp.dw / (sp.k * float(sp.q) ** 2),
        "support": f"(-{two_q:g}, {two_q:g})", "two_q": two_q,
        "qcrit": q_crit(sp.w_h, sp.w_l, sp.k, sp.e_max),
        "D1": "{0}", "D1_n": {c.name: int(stage_grid(1, sp.B, c.state_step).size) for c in TIERS},
        "D2": f"[-{sp.domain_half(2):g}, {sp.domain_half(2):g}]",
        "D2_n": {c.name: int(stage_grid(2, sp.B, c.state_step).size) for c in TIERS},
        "rec_n": int(rg.size), "rec_pos": int((np.abs(rg) < two_q).sum()), "rec_tail": int((np.abs(rg) >= two_q).sum()),
        "rec_step": float(rec["protocol"]["recovery_step"]),
    }
    # the closed form vanishes exactly at |d| = 2q and is positive inside (checked, not assumed)
    g2_edge = g2_two_stage(np.array([-two_q, two_q]), sp.q, sp.w_h, sp.w_l, sp.k, sp.e_max)
    inside = g2_two_stage(rg[np.abs(rg) < two_q], sp.q, sp.w_h, sp.w_l, sp.k, sp.e_max)
    out["support_checked"] = bool(np.all(g2_edge == 0.0) and np.all(inside > 0.0))
    return out


def build_t04(pack: C.Pack) -> None:
    """T04: parameters and derived constants per q (repository functions, cross-read with records)."""
    proto = _proto()
    V = {q: _t04_values(q) for q in QS}
    cal = _csv(CAL_P1)
    man = {q: _json(f"{CONF}/q{q}/seed20501/manifest.json") for q in QS}
    rows: List[Dict[str, Any]] = []

    def add(quantity: str, symbol: str, key: str, unit: str, definition: str, computed_with: str,
            rec: Optional[Dict[int, Any]] = None, recorded_in: str = "", sub: Optional[str] = None) -> None:
        v = {q: (V[q][key][sub] if sub else V[q][key]) for q in QS}
        r: Dict[str, Any] = {"quantity": quantity, "symbol": symbol, "q50": v[50], "q60": v[60],
                             "unit": unit,
                             "definition": definition, "computed_with": computed_with,
                             "recorded_q50": rec[50] if rec else np.nan, "recorded_q60": rec[60] if rec else np.nan,
                             "recorded_in": recorded_in}
        def same(a: Any, b: Any) -> bool:
            return a == b if isinstance(a, str) or isinstance(b, str) else float(a) == float(b)
        r["matches_record"] = (bool(all(same(rec[q], v[q]) for q in QS)) if rec else np.nan)
        rows.append(r)

    recs = {q: proto["records"][str(q)] for q in QS}
    gsrc = "protocols/v2_T2_locked_v1_1.json records[q].game"
    add("Winner prize", "W_H", "w_h", "payoff units", "prize of the player with the positive final "
                                                      "gap",
        "GameSpec.w_h", {q: recs[q]["game"]["w_h"] for q in QS}, gsrc + ".w_h")
    add("Loser prize", "W_L", "w_l", "payoff units", "prize of the player with the negative final "
                                                     "gap",
        "GameSpec.w_l", {q: recs[q]["game"]["w_l"] for q in QS}, gsrc + ".w_l")
    add("Prize spread", "DW = W_H - W_L", "dw", "payoff units",
        "normalizer of every deviation metric",
        "envs/curriculum_env.py:GameSpec.dw", {q: recs[q]["dw"] for q in QS},
        "protocols/v2_T2_locked_v1_1.json records[q].dw")
    add("Effort cost coefficient", "k", "k", "payoff units per effort unit squared",
        "stage cost c(e) = k e^2",
        "GameSpec.k", {q: recs[q]["game"]["k"] for q in QS}, gsrc + ".k")
    add("Effort lower bound", "e_min", "e_min", "effort units", "lower end of the effort interval",
        "GameSpec.e_min", {q: recs[q]["game"]["e_min"] for q in QS}, gsrc + ".e_min")
    add("Effort upper bound", "e_max", "e_max", "effort units", "upper end of the effort interval",
        "GameSpec.e_max", {q: recs[q]["game"]["e_max"] for q in QS}, gsrc + ".e_max")
    add("Horizon", "T", "T", "stages", "number of effort stages", "GameSpec.T",
        {q: recs[q]["game"]["T"] for q in QS}, gsrc + ".T")
    add("Shock half-width", "q", "q", "effort units", "eps ~ U(-q, q)", "GameSpec.q",
        {q: recs[q]["game"]["q"] for q in QS}, gsrc + ".q")
    add("Shock model", "eps, xi", "shock", "", "per-player uniform action shocks; the gap moves by "
                                               "the shock "
        "difference xi", "envs/curriculum_env.py:step_gap; utils/theory_multistage.py:f_xi")
    add("Density of xi at 0", "f_xi(0) = 1/(2q)", "f0", "1/effort units",
        "peak of the triangular density",
        "utils/theory_multistage.py:f_xi")
    add("Maximal one-stage gap change", "B = (e_max - e_min) + 2q", "B", "effort units",
        "half-width of D_2", "envs/curriculum_env.py:GameSpec.B", {q: recs[q]["B"] for q in QS},
        "protocols/v2_T2_locked_v1_1.json records[q].B")
    add("Stage-1 closed-form effort", "e1*(0) = DW/(6kq)", "e1", "effort units",
        "benchmark stage-1 effort at d = 0",
        "utils/theory_multistage.py:g1_two_stage",
        {q: float(cal[(cal.q == q) & (cal.policy == "analytic_eq") & (cal.verifier_tier == "final")].g1.iloc[0])
         for q in QS}, f"{CAL_P1}: g1 (analytic_eq, final)")
    add("Stage-2 closed-form effort at d = 0", "e2*(0) = DW f_xi(0)/(2k)", "e2", "effort units",
        "peak of e2*(d) = clip(DW f_xi(d)/(2k), 0, "
        "e_max)", "utils/theory_multistage.py:g2_two_stage",
        {q: float(cal[(cal.q == q) & (cal.policy == "analytic_eq") & (cal.verifier_tier == "final")].g2_at_0.iloc[0])
         for q in QS}, f"{CAL_P1}: g2_at_0 (analytic_eq, final)")
    add("r", "r = DW/(k q^2)", "r", "dimensionless", "prize spread over the cost of a q-sized "
                                                     "effort",
        "computed from GameSpec fields")
    add("Positive-effort support of e2*(d)", "|d| < 2q", "support", "effort units (gap d)",
        "open interval on which e2*(d) > 0 (e2* = 0 at |d| = 2q; checked on the recovery grid)",
        "utils/theory_multistage.py:g2_two_stage")
    add("Closed-form validity threshold", "q_crit", "qcrit", "effort units",
        "max(q_soc, stage-2 and stage-1 effort bounds, participation); the closed form is valid "
        "for q > q_crit",
        "utils/theory_multistage.py:q_crit")
    add("Stage-1 state domain", "D_1", "D1", "effort units (gap d)", "root only",
        "utils/dp_br_verifier.py:stage_grid")
    for c in TIERS:
        add(f"D_1 nodes, {c.name} tier", "|D_1|", "D1_n", "nodes", "stage-1 grid (every tier)",
            "utils/dp_br_verifier.py:stage_grid",
            {q: man[q]["grids"][c.name]["stage_grid_points"]["1"] for q in QS} if c.name != "dev_2x" else None,
            f"{CONF}/q*/seed20501/manifest.json "
            f"grids.{c.name}.stage_grid_points.1" if c.name != "dev_2x" else "",
            sub=c.name)
    add("Stage-2 state domain", "D_2 = [-B, B]", "D2", "effort units (gap d)",
        "feasible stage-2 gaps from the root", "envs/curriculum_env.py:GameSpec.domain_half",
        {q: f"[-{recs[q]['domain_half_stage2']:g}, {recs[q]['domain_half_stage2']:g}]" for q in QS},
        "protocols/v2_T2_locked_v1_1.json records[q].domain_half_stage2")
    for c in TIERS:
        if c.name == "dev_2x":
            sel = cal[(cal.policy == "analytic_eq") & (cal.verifier_tier == "dev_2x")]
            rec = {q: int(sel[sel.q == q].DeltaT_over_dw_n_on.iloc[0] + sel[sel.q == q].DeltaT_over_dw_n_off.iloc[0])
                   for q in QS}
            rin = f"{CAL_P1}: DeltaT_over_dw_n_on + DeltaT_over_dw_n_off (analytic_eq, dev_2x)"
            used = "Phase 1 calibration only"
        else:
            rec = {q: man[q]["grids"][c.name]["stage_grid_points"]["2"] for q in QS}
            rin = f"{CONF}/q*/seed20501/manifest.json grids.{c.name}.stage_grid_points.2"
            used = "training-time verifier calls" if c.name == "development" else "gates and end-of-run evaluation"
        add(f"D_2 nodes, {c.name} tier (state step {c.state_step:g})", "|D_2|", "D2_n", "nodes",
            f"symmetric grid with 0 and both endpoints "
            f"({used})", "utils/dp_br_verifier.py:stage_grid", rec, rin,
            sub=c.name)
    sel = cal[(cal.policy == "analytic_eq") & (cal.verifier_tier == "final")]
    add(f"Recovery grid nodes on D_2 (step {V[50]['rec_step']:g}; "
        f"tier-independent)", "|D_2 recovery|", "rec_n", "nodes",
        "grid of the closed-form recovery metrics (0 is an exact "
        "node)", "utils/v2_metrics.py:symmetric_grid",
        {q: int(sel[sel.q == q].recovery_n_pos.iloc[0] + sel[sel.q == q].recovery_n_tail.iloc[0]) for q in QS},
        f"{CAL_P1}: recovery_n_pos + recovery_n_tail")
    add("Recovery grid nodes with |d| < "
        "2q", "n_pos", "rec_pos", "nodes", "nodes of the RMSE region",
        "utils/v2_metrics.py:recovery_metrics", {q: int(sel[sel.q == q].recovery_n_pos.iloc[0]) for q in QS},
        f"{CAL_P1}: recovery_n_pos")
    add("Recovery grid nodes with |d| >= "
        "2q", "n_tail", "rec_tail", "nodes", "nodes of the tail region",
        "utils/v2_metrics.py:recovery_metrics", {q: int(sel[sel.q == q].recovery_n_tail.iloc[0]) for q in QS},
        f"{CAL_P1}: recovery_n_tail")
    df = pd.DataFrame(rows)
    assert all(V[q]["support_checked"] for q in QS), "e2* support check failed"
    assert all(V[q]["qcrit"] < q for q in QS)

    srcs = [C.src("utils/theory_multistage.py"), C.src("envs/curriculum_env.py"),
            C.src("utils/dp_br_verifier.py"),
            C.src("utils/v2_metrics.py"), C.src(PROTO), C.src(CAL_P1)] + \
        [C.src(f"{CONF}/q{q}/seed20501/manifest.json", "grids") for q in QS]
    docs = {
        "quantity": "Name of the parameter or derived constant",
        "symbol": "Symbol and defining formula",
        "q50": dict(definition="Value at q = 50 (numbers in the unit of the row; text for sets and "
                               "models)",
                    units="see column unit", tier="tier-independent unless the row names a tier"),
        "q60": dict(definition="Value at q = 60 (numbers in the unit of the row; text for sets and "
                               "models)",
                    units="see column unit", tier="tier-independent unless the row names a tier"),
        "unit": "Unit of the row's values",
        "definition": "What the quantity is",
        "computed_with": "Repository function (or field) that produced the q50 / q60 values",
        "recorded_q50": "The same quantity as recorded in a run or protocol file (recorded_in), q "
                        "= 50",
        "recorded_q60": "The same quantity as recorded in a run or protocol file (recorded_in), q "
                        "= 60",
        "recorded_in": "File and key of the recorded value",
        "matches_record": "Whether the computed and the recorded values are equal at both q "
                          "(blank: no record)",
    }
    pack.table("T04", df, status="generated", sources=srcs, script=f"{MOD}:build_t04", docs=docs,
               tier="n/a",
               notes=("Game parameters from the locked protocol records; every derived constant "
                      "computed with the "
                      "repository's functions (g1_two_stage, g2_two_stage, f_xi, q_crit, "
                      "GameSpec.B/dw/domain_half, "
                      "stage_grid, symmetric_grid; dev_2x tier = state step 2, effort step 0.5, GL "
                      "16 as in "
                      "tools/v2/common.py); recorded_* columns cross-read the same numbers from "
                      "the protocol, from the "
                      "Phase 1 calibration rows and from the confirmation manifests (grids "
                      "block)."),
               caption=("Parameters and derived constants of the T=2 game, per q. Node counts come "
                        "from the grid "
                        "constructors (no typed numbers). dev_2x is the Phase 1 2x-finer check "
                        "tier."))

    # ---- cross-checks with the reports
    xc = XCheck(pack, "T04", "protocols/v2_T2_locked.md", "game table: T, B, e1*(0), e2*(0)")
    for t in xc.tables_with(["q = 50", "q = 60"], "Game"):
        for r in t["rows"]:
            lab = r[0]
            key = ("T" if lab == "T" else "B" if lab.startswith("B =") else "e1" if lab.startswith("e\u2081")
                   else "e2" if lab.startswith("e\u2082") else None)
            if key is None:
                continue
            for j, q in ((1, 50), (2, 60)):
                xc.cmp(f"{key} (q={q})", V[q][key], _nums(r[j])[0], f"line {t['line']}, row {lab}")
    xc.done()
    xc = XCheck(pack, "T04", R_P0, "e1*, e2*(0), D_2, grid sizes (sections 1.2, 1.9, 2)")
    m = xc.find(r"g2\(0\) = (" + _NUM + r") and (" + _NUM + r") respectively")
    if m:
        xc.cmp("e2*(0) (q=50)", V[50]["e2"], m.group(1), "section 1.9")
        xc.cmp("e2*(0) (q=60)", V[60]["e2"], m.group(2), "section 1.9")
    m = xc.find(r"giving (" + _NUM + r") \(q=50\) and (" + _NUM + r") \(q=60\)")
    if m:
        xc.cmp("e1*(0) (q=50)", V[50]["e1"], m.group(1), "section 1.9")
        xc.cmp("e1*(0) (q=60)", V[60]["e1"], m.group(2), "section 1.9")
    m = xc.find(r"D_2 = \[-(\d+), (\d+)\] at q=50 and \[-(\d+), (\d+)\] at q=60")
    if m:
        for j, q in ((2, 50), (4, 60)):
            xc.cmp(f"D_2 half-width (q={q})", _spec(q).domain_half(2), m.group(j), "section 1.2")
    m = xc.find(r"\| B, D_2 half-width \| (\d+) / (\d+) \|")
    if m:
        xc.cmp("B (q=50)", V[50]["B"], m.group(1), "section 2 table")
        xc.cmp("B (q=60)", V[60]["B"], m.group(2), "section 2 table")
    for tier, mm in zip(("development", "final"), xc.findall(r"D_2 grid (\d+) / (\d+) points; "
                                                             r"effort grid (\d+)\)")):
        xc.cmp(f"D_2 nodes {tier} (q=50)", V[50]["D2_n"][tier], mm[0], "section 2 table")
        xc.cmp(f"D_2 nodes {tier} (q=60)", V[60]["D2_n"][tier], mm[1], "section 2 table")
    xc.done()
    xc = XCheck(pack, "T04", R_P1, "B, grid sizes (section 2.2), recovery-region node counts "
                                   "(section 4.3)")
    m = xc.find(r"B = 100 \+ 2q \((\d+) at q=50, (\d+) at q=60\)")
    if m:
        xc.cmp("B (q=50)", V[50]["B"], m.group(1), "section 2.2")
        xc.cmp("B (q=60)", V[60]["B"], m.group(2), "section 2.2")
    m = xc.find(r"development tier \((\d+) / (\d+) points\), the 2\S* tier \((\d+) / (\d+)\), the "
                r"final tier "
                r"\((\d+) / (\d+)\) and the 0\.5-step recovery grid \((\d+) / (\d+)\)")
    if m:
        vals = [V[50]["D2_n"]["development"], V[60]["D2_n"]["development"], V[50]["D2_n"]["dev_2x"],
                V[60]["D2_n"]["dev_2x"], V[50]["D2_n"]["final"], V[60]["D2_n"]["final"], V[50]["rec_n"], V[60]["rec_n"]]
        names = ["dev q50", "dev q60", "dev_2x q50", "dev_2x q60", "final q50", "final q60",
                 "recovery q50",
                 "recovery q60"]
        for i, (nm, v) in enumerate(zip(names, vals)):
            xc.cmp(f"grid nodes {nm}", v, m.group(i + 1), "section 2.2")
    m = xc.find(r"\| Stage-2 RMSE over \\?\|d\\?\| < 2q, raw / . e.\*\(0\) \| [^|]*\((\d+) nodes\)")
    if m:
        xc.cmp("recovery nodes |d| < 2q (q=50)", V[50]["rec_pos"], m.group(1), "section 4.3")
    m = xc.find(r"\\?\|d\\?\| >= 2q, (\d+) nodes\)|\\?\|d\\?\| ≥ 2q, (\d+) nodes\)")
    if m:
        xc.cmp("recovery nodes |d| >= 2q (q=50)", V[50]["rec_tail"], m.group(1) or m.group(2),
               "section 4.3")
    xc.done()


# ----------------------------------------------------------------------------------------------
# F01 closed-form benchmark
# ----------------------------------------------------------------------------------------------

def build_f01(pack: C.Pack) -> None:
    """F01: e2*(d) on D_2 for both q (g2_two_stage) with e1*(0) marked at d = 0."""
    fig, axes = style.new_figure(1, 1, height=3.3)
    ax = axes[0, 0]
    rows: List[Dict[str, Any]] = []
    checks: List[str] = []
    cal = _csv(CAL_P1)
    for q in QS:
        sp = _spec(q)
        step = float(_proto()["records"][str(q)]["protocol"]["recovery_step"])
        d = symmetric_grid(sp.domain_half(2), step)
        e2 = g2_two_stage(d, sp.q, sp.w_h, sp.w_l, sp.k, sp.e_max)
        e1 = g1_two_stage(sp.q, sp.w_h, sp.w_l, sp.k)
        col, ls, mk = style.Q_COLORS[q], style.Q_LINESTYLE[q], style.Q_MARKER[q]
        ax.plot(d, e2, color=col, ls=ls, lw=style.LINE_W,
                label=f"q={q}: $e_2^*(d)$ on $D_2$=[{-sp.domain_half(2):g}, {sp.domain_half(2):g}] "
                      f"({d.size} nodes)")
        ax.plot([0.0], [e1], ls="none", marker=mk, ms=6, mfc=col, mec=style.INK, mew=0.8,
                label=f"q={q}: $e_1^*(0)$ = {e1:.3f} (stage 1, d = 0)")
        rows += [{"series": "e2_star", "q": q, "d": float(a), "effort": float(b)} for a,
                 b in zip(d, e2)]
        rows.append({"series": "e1_star_0", "q": q, "d": 0.0, "effort": float(e1)})
        # checks against saved arrays and the Phase 1 calibration rows
        z = np.load(C.abspath(f"{P1}/calibration/npz/q{q}_analytic_eq_final.npz"))
        dmax = C.max_abs_diff(z["recovery_g2"],
                              e2) if z["recovery_d_grid"].shape == d.shape else float("nan")
        same_grid = bool(np.array_equal(z["recovery_d_grid"], d))
        r = cal[(cal.q == q) & (cal.policy == "analytic_eq") & (cal.verifier_tier == "final")].iloc[0]
        checks.append(f"q={q}: plotted e2*(d) equals recovery_g2 "
                      f"of {P1}/calibration/npz/q{q}_analytic_eq_final.npz "
                      f"(same grid: {same_grid}; max abs diff {dmax:g})")
        checks.append(f"q={q}: plotted e2*(0) - calibration.csv g2_at_0 "
                      f"= {float(e2[d == 0.0][0]) - r.g2_at_0:g}; "
                      f"e1*(0) - g1 = {e1 - r.g1:g}")
    ax.set_xlabel("gap d at the start of stage 2 (effort units)")
    ax.set_ylabel("closed-form effort (effort units)")
    ax.set_xlim(-235, 235)
    ax.set_ylim(-2, 80)
    style.legend(ax, loc="upper left", fontsize=8)
    data = pd.DataFrame(rows)
    srcs = [C.src("utils/theory_multistage.py"), C.src("utils/v2_metrics.py"), C.src(PROTO),
            C.src(CAL_P1)] + \
        [C.src(f"{P1}/calibration/npz/q{q}_analytic_eq_final.npz",
               "recovery_d_grid, recovery_g2") for q in QS]
    docs = {"series": "e2_star = closed-form stage-2 effort e2*(d) on the recovery grid; e1_star_0 "
                      "= closed-form "
                      "stage-1 effort e1*(0) plotted at d = 0",
            "q": None, "d": dict(definition="Gap d at the start of stage 2 (0 for the stage-1 "
                                            "marker)",
                                 units="effort units (gap d)", tier="tier-independent"),
            "effort": dict(definition="Closed-form effort (g2_two_stage or "
                                      "g1_two_stage)", units="effort units, raw",
                           tier="tier-independent")}
    docs = {k: v for k, v in docs.items() if v is not None}
    pack.figure("F01", fig, data, status="generated", sources=srcs, script=f"{MOD}:build_f01",
                docs=docs,
                tier="tier-independent",
                caption=("Closed-form stage-2 equilibrium effort e2*(d) = clip(DW f_xi(d)/(2k), 0, "
                         "100) on D_2 = [-B, B] "
                         "for q = 50 (solid, B = 200) and q = 60 (dashed, B = 220), evaluated with "
                         "utils/theory_multistage.g2_two_stage on the 0.5-step recovery grid; "
                         "markers: closed-form "
                         "stage-1 effort e1*(0) = DW/(6kq) (g1_two_stage) at d = 0. Parameters W_H "
                         "= 6, W_L = 2, "
                         "k = 1/3500 (locked protocol records). Closed form; no data, no tier."),
                checks=checks)


# ----------------------------------------------------------------------------------------------
# shared evidence (cached scans used by several items)
# ----------------------------------------------------------------------------------------------

INV_COLS = [  # (relation label, residual column, kind)
    ("G_2 = Delta_2 (whole stage-2 grid)", "inv_GT_eq_DeltaT_absdiff_over_dw", "equality"),
    ("Delta_t(d) <= G_t(d), pointwise", "inv_Delta_le_G_pointwise_excess_over_dw", "inequality"),
    ("Delta_max_all <= Gmax_full", "inv_Deltamax_le_Gmax_excess_over_dw", "inequality"),
    ("Gmax_full <= dFull", "inv_Gmax_le_dFull_excess_over_dw", "inequality"),
    ("EXP_root = G_1(0)", "inv_EXP_eq_G1_absdiff_over_dw", "equality"),
    ("EXP_root <= dReach", "inv_EXP_le_dReach_excess_over_dw", "inequality"),
    ("dReach <= dFull", "inv_dReach_le_dFull_excess_over_dw", "inequality"),
    ("(companion) EXP_root <= dReach over the BR-PMF "
     "support", "inv_EXP_le_dReachPMF_excess_over_dw", "inequality"),
]
_CACHE: Dict[str, Any] = {}


def _percall_logs() -> Tuple[pd.DataFrame, C.SourceSet]:
    """Every training-time verifier call: rows of all v2_checkpoints*.csv under results/ (development tier)."""
    if "percall" not in _CACHE:
        ss = C.srcs(["results/v2_pilots/**/v2_checkpoints*.csv",
                     "results/v2_T2_locked/**/v2_checkpoints*.csv"],
                    label="results/{v2_pilots,v2_T2_locked}/**/v2_checkpoints*.csv")
        parts = []
        keep = [c for _, c, _ in INV_COLS] + ["inv_Delta_le_G_pointwise_at_t",
                                              "inv_Delta_le_G_pointwise_at_d",
                                               "inv_GT_eq_DeltaT_at_d", "verifier_tier", "valid", "q", "update"]
        for s in ss.files:
            d = _csv(s.path)
            d = d[[c for c in keep if c in d.columns]].copy()
            d["file"] = s.path
            d["study"] = s.path.split("/")[2]
            parts.append(d)
        _CACHE["percall"] = (pd.concat(parts, ignore_index=True), ss)
    return _CACHE["percall"]


def _evidence() -> Dict[str, Any]:
    """Counts quoted in T05 (read from the analysis CSVs, never typed)."""
    if "evidence" in _CACHE:
        return _CACHE["evidence"]
    ev: Dict[str, Any] = {}
    rel = {"p3": "results/v2_pilots/pilot3/analysis/reproducibility_vs_pilot2_B2.csv",
           "2a": "results/v2_pilots/pilot4/analysis/repro_2a_constant_vs_ext.csv",
           "vr": "results/v2_pilots/pilot4/analysis/verifier_consumes_no_rng.csv",
           "dm": "results/v2_pilots/dreach_mask_check/dreach_mask_check.csv"}
    for k in ("p3", "2a"):
        d = _csv(rel[k])
        idc = [c for c in d.columns if c.endswith("_identical")]
        ev[f"n_{k}"], ev[f"N_{k}"] = int(d[idc].all(axis=1).sum()), len(d)
    d = _csv(rel["vr"])
    ev["n_vr"], ev["N_vr"] = int(d["rng_states_unchanged_after_6_verifier_calls"].sum()), len(d)
    d = _csv(rel["dm"])
    diff = d["dreach_official_minus_refmask_over_dw"]
    ev.update(dm_n=len(d), dm_holes=int(d["n_holes"].sum()), dm_nonzero=int((diff != 0).sum()),
              dm_max=float(diff.max()), dm_neg=int((diff < 0).sum()), dm_extra_nodes=int(d["n_extra"].sum()))
    reps = sorted((C.REPO / "reports" / "v2").glob("*.md")) + sorted((C.REPO / "protocols").glob("*.md"))
    ev["n_reports"] = len(reps)
    ev["report_files"] = [str(p.relative_to(C.REPO)) for p in reps]
    hits = []
    for p in reps:
        t = p.read_text(encoding="utf-8")
        if re.search(r"likelihood|log-?prob", t, flags=re.I):
            hits.append(p.name)
    ev["lik_hits"] = hits
    t3 = []
    for p in reps:
        for ln in p.read_text(encoding="utf-8").splitlines():
            if re.search(r"warm-?up", ln, flags=re.I) and re.search(r"T=3|T3\b", ln):
                t3.append(p.name)
    ev["t3_hits"] = sorted(set(t3))
    pc, _ = _percall_logs()
    ev["n_calls"], ev["n_call_files"] = len(pc), int(pc["file"].nunique())
    ev["sources"] = [C.src(v) for v in rel.values()]
    _CACHE["evidence"] = ev
    return ev


# ----------------------------------------------------------------------------------------------
# T05 baseline audit findings
# ----------------------------------------------------------------------------------------------

def build_t05(pack: C.Pack) -> None:
    """T05: Phase 0 audit items (legacy behaviour, risk, v2 handling) and the PI's 0929 issue list."""
    pr = _proto()
    P50 = pr["records"]["50"]["protocol"]
    ga = {c["metric"]: c["threshold"] for c in pr["gates"]["G-A"]["all_must_hold"]}
    gf = {c["metric"]: c["threshold"] for c in pr["gates"]["G-F"]["all_must_hold"]}
    gn = {c["metric"]: c["threshold"] for c in pr["gates"]["G-N"]["all_must_hold"]}
    bins = {q: pr["records"][str(q)]["es_bins_stage2"] for q in QS}
    ev = _evidence()
    RS = "run/run_v2_stagewise.py"
    AG2 = "agents/ppo_curriculum_v2.py"
    RO = "run/v2_rollout.py"
    LK = "run/run_v2_T2_locked.py"
    AG = "agents/ppo_curriculum.py"
    VM = "utils/v2_metrics.py"
    repro = (f"{ev['n_p3']}/{ev['N_p3']} Pilot 3 stochastic runs bit-identical to Pilot 2 B2 "
             f"(reproducibility_vs_pilot2_B2.csv); {ev['n_2a']}/{ev['N_2a']} Pilot 4 section 2a "
             f"constant runs "
             f"bit-identical to the Phase A extension (repro_2a_constant_vs_ext.csv)")
    A, B_ = "a: Phase 0 audit item", "b: PI 0929 issue list"
    rows = [
        dict(block=A, item="Stop rules (Phase 0 section 3)",
             legacy_behaviour=("Phase caps A 400 / B 600 / C 1000 (exit 'budget_exhausted'); A and "
                               "B advance after "
                               "k_phase = 3 consecutive eligible development-tier verifier calls "
                               "(valid, criterion <= "
                               "threshold, concentration <= 0.04), dropping the unused budget; C "
                               "stops at the first "
                               "eligible call (k_stop = 1) and that iterate becomes the "
                               "checkpoint; calls are triggered "
                               "by warm-up (local 100), stability, timeout or phase end. In the 80 "
                               "latest runs the A rule "
                               "never fired (A always ran 400 updates)."),
             risk=("Phase length and the evaluated iterate are decided by development-tier "
                   "verifier calls at the "
                   "cadence, so compute and the selected checkpoint vary between runs, and the "
                   "checkpoint is selected "
                   "on a verifier criterion that is also reported."),
             v2_handling=("fixed_budget = true: no early exit; the update at which the legacy rule "
                          "would have fired is "
                          "recorded (v2_run_summary.json would_have_fired). Locked v1.1: Phase A "
                          f"{pr['pipeline']['phase_A']['updates']} and Phase "
                          f"B {pr['pipeline']['phase_B']['updates']} "
                          "updates, Phase C not used, evaluated candidate = last iterate, "
                          "pass/fail decided by "
                          "end-of-phase gates (G-A and G-F on the final tier, G-N development vs "
                          "final)."),
             status="addressed",
             where_v2=(f"{loc(RS, 'Run.run_phase', 'if not self.fixed:')}; {PRO_KEYS('pipeline.fixed_budget, phase_A, phase_B, not_used, evaluated_candidate; gates')}; "
                       f"{R_INFRA} section 1 point 6"),
             legacy_source=f"{R_P0} section 3 (run/run_final_dp_br_round3_dense.py lines 406, "
                           f"484-557 as cited there)"),
        dict(block=A, item="Joint-training semantics (Phase 0 section 4, Q2)",
             legacy_behaviour=("In Phases B and C the learner transitions of every visited stage "
                               "enter the clipped "
                               "surrogate (no stage mask) and the advantages are normalized over "
                               "all rows of both stages; "
                               "stage-2 actions are stochastic for both players (learner: live "
                               "actor, rng_learn; "
                               "opponent: lagged snapshot, rng_opp)."),
             risk=("Stage-1 training keeps moving the stage-2 mapping that defines the stage-1 "
                   "target, and the stage-2 "
                   "advantages set the scale of the stage-1 advantages (Phase 0 Q2)."),
             v2_handling=("Phase B with stage2_update_mode = frozen: a deep-copied stage-2 "
                          "snapshot (no grad, eval mode, "
                          "in no optimizer) plays both players' stage-2 actions and is the "
                          "candidate's stage 2; "
                          "adv_norm_scope = stage1_rows (policy loss and advantage normalization "
                          "over stage-1 rows, "
                          "same minibatch permutation); continuation_action_mode = mean (stage-2 "
                          "draws made and "
                          "discarded). Joint training is not used in the locked protocol (it was "
                          "the Pilot 2 A arm)."),
             status="addressed",
             where_v2=(f"{loc(AG2, 'CurriculumPPOv2.update')}; {loc(AG2, 'CurriculumPPOv2.freeze_stage2_snapshot')}; "
                       f"{loc(RO, 'collect_batch_v2')}; tests/test_v2_infra.py "
                       f"(test_gradient_isolation, "
                       "test_norm_scope, test_snapshot_immutable); reports/v2/pilot2_freeze.md; "
                       f"reports/v2/pilot3_continuation_mode.md; {PRO_KEYS('pipeline.flags')}"),
             legacy_source=f"{R_P0} section 4 and Q2"),
        dict(block=A, item="Shared actor (Phase 0 section 1.4)",
             legacy_behaviour=("One actor network (2 -> 64 tanh -> 64 tanh -> 2, Beta "
                               "mean/concentration) serves both "
                               "stages; the stage enters only through the input tau = (t-1)/(T-1); "
                               "no per-stage heads; "
                               "separate critic and Adam optimizers; rollouts against a lagged "
                               "opponent copy refreshed "
                               "every 20 global updates and at phase entry."),
             risk="Gradient steps on the rows of one stage also change the other stage's mapping "
                  "through the shared parameters.",
             v2_handling=("Architecture unchanged (one actor). In Phase B the stage-2 mapping used "
                          "for both players' "
                          "stage-2 actions and by the evaluated candidate is the frozen snapshot, "
                          "so later changes of the "
                          "live network's stage-2 output enter neither rollouts nor evaluation; "
                          "the live drift is logged "
                          "for information (stage2_drift_live_* columns); drift test: max |change| "
                          "of the snapshot's "
                          "stage-2 mean, alpha and beta since the freeze must be 0 "
                          "(drift_test.json)."),
             status="addressed (by freezing; the actor is still shared)",
             where_v2=(f"{loc(RS, 'Run.policy_fns')}; {loc(RS, 'Run.drift_scalars')}; "
                       f"{loc(LK, 'run_pipeline', 'drift_test.json')}; {R_INFRA} section 1 point "
                       f"4"),
             legacy_source=f"{R_P0} section 1.4"),
        dict(block=A, item="Start-state coverage of the stage-2-only phase (Phase 0 section 5)",
             legacy_behaviour=(f"StartSampler.balanced: one of {bins[50]} (q=50) "
                               f"/ {bins[60]} (q=60) equal-width bins of "
                               f"width {P50['es_bin_width']:g} on D_2 uniformly, then uniform "
                               f"within the bin; learner role "
                               "a fair coin (the learner sees +d or -d); support equal to D_2."),
             risk=("No coverage gap found; the start density is uniform in d while the root-path "
                   "stage-2 distribution "
                   "is concentrated near the centre (described, not judged, in Phase 0)."),
             v2_handling=(f"Unchanged: {pr['pipeline']['phase_A']['starts']}; "
                          f"es_bins_stage2 {bins[50]} / {bins[60]} "
                          "checked against the sampler at start-up."),
             status="unchanged (no gap found)",
             where_v2=(f"{PRO_KEYS('pipeline.phase_A.starts; records[q].es_bins_stage2')}; "
                       f"{loc(RS, 'Run.__init__', 'self.sampler.n_bins(2)')}; {loc(RS, 'Run.run_phase', 'self.sampler.balanced')}"),
             legacy_source=f"{R_P0} section 5"),
        dict(block=A, item="Evaluation convention (Phase 0 section 6)",
             legacy_behaviour=("Evaluated policy = Beta mean e_hat_t(d) = e_min + (e_max - e_min) "
                               "alpha/(alpha+beta) in "
                               "float64 from float32 alpha, beta, observation built as in "
                               "training; the opponent is the "
                               "same function at -d."),
             risk="None identified in Phase 0.",
             v2_handling=(f"Unchanged (record mean_extraction: "
                          f"'{pr['records']['50']['mean_extraction']}'); in frozen mode "
                          "the candidate is (live actor at t = 1, frozen snapshot at t = 2); "
                          "locked candidate: "
                          f"'{pr['pipeline']['evaluated_candidate']}'."),
             status="unchanged (convention kept)",
             where_v2=(f"{PRO_KEYS('records[q].mean_extraction; pipeline.evaluated_candidate')}; "
                       f"{loc('run/run_final_dp_br.py', 'make_policy_fns')}; {loc(RS, 'Run.policy_fns')}"),
             legacy_source=f"{R_P0} section 6"),
        dict(block=A, item="Checkpoint completeness (Phase 0 section 7)",
             legacy_behaviour=("checkpoint.pt (actor, critic, opponent snapshot, both Adam states, "
                               "cfg) is written once, at "
                               "the C stopping point; missing: the five numpy stream states, torch "
                               "and python RNG states, "
                               "the counters (global/local update, eligible, stability, last call, "
                               "snapshot phase); no "
                               "end-of-A full state; weights/u*.npz (every 25 updates) hold actor "
                               "and critic weights only."),
             risk=("No existing run can serve as an exact branching parent (Phase 0 section 7, "
                   "'Consequence'); weight "
                   "exports alone cannot reproduce a continuation."),
             v2_handling=("Full-state checkpoints state_end_<phase>.pt (optional "
                          "state_u<update>.pt): actor, critic, lagged "
                          "opponent, frozen snapshot, both Adam states, snapshot_refreshes, the "
                          "env/learn/opp/start "
                          "streams and the minibatch stream, the torch generator, the torch/numpy "
                          "global and python "
                          "random states, counters, snapshot log, schedule positions; a Phase B "
                          "parent whose SHA-256 "
                          "differs is refused; exact branching tested "
                          "(test_continuous_equals_branched, "
                          f"test_branch_reproducibility) and observed: {repro}."),
             status="addressed",
             where_v2=(f"{loc(RS, 'Run.full_state')}; {loc(RS, 'Run.restore')}; {loc(RS, 'main', 'parent sha256')}; "
                       f"{loc(AG2, 'CurriculumPPOv2.full_state')}; "
                       f"tests/test_v2_infra.py; {R_INFRA} section 1 point 3"),
             legacy_source=f"{R_P0} section 7"),
        # ---- block b: the PI's 0929 issue list (SPEC 5.3)
        dict(block=B_, item="Phase A fixed-budget semantics",
             legacy_behaviour="Phase A cap 400 with a k_phase = 3 early-exit rule that never fired "
                              "in the 80 latest runs.",
             risk="Phase A length depends on verifier timing; de facto fixed at 400 without being "
                  "declared fixed.",
             v2_handling=("fixed_budget flag (no early exit; would-have-fired recorded); locked "
                          "Phase A = "
                          f"{pr['pipeline']['phase_A']['updates']} updates with the end-of-phase "
                          f"gate G-A on the final tier "
                          f"(eta_2/DW <= {ga['eta_T_over_dw']:g}, RMSE/e2*(0) "
                          f"<= {ga['stage2_rmse_pos_over_g2_0']:g}, "
                          f"tail mean/e2*(0) <= {ga['stage2_tail_mean_over_g2_0']:g})."),
             status="addressed",
             where_v2=(f"{loc(RS, 'Run.run_phase', 'if not self.fixed:')}; {PRO_KEYS('pipeline.phase_A; gates.G-A')}; "
                       "reports/v2/phaseA_ext.md (decision D2)"),
             legacy_source=f"{R_P0} section 3"),
        dict(block=B_, item="T2/T3 verifier warm-up differences",
             legacy_behaviour=(f"T=2 cadence of the archived record: warm-up call at "
                               f"local {P50['warmup']}, stability "
                               f"checks every {P50['stability_every']}, timeouts "
                               f"A {P50['verifier_timeout']['A']} / "
                               f"B {P50['verifier_timeout']['B']} / "
                               f"C {P50['verifier_timeout']['C']}."),
             risk="Verifier timing differs between horizons, so phase rules fire at different "
                  "points.",
             v2_handling=("T=3 is out of scope of the T=2 v2 work; T=2 keeps the archived cadence. "
                          "Under the fixed budget "
                          "training-time calls only record (no early exit) and the verifier "
                          "consumes no RNG "
                          f"({ev['n_vr']}/{ev['N_vr']} Pilot 4 runs: RNG states unchanged after 6 "
                          f"calls), so the cadence "
                          f"cannot change training. Files of reports/v2/*.md and protocols/*.md "
                          f"with a line mentioning "
                          f"T=3 together with warm-up: {len(ev['t3_hits'])} of {ev['n_reports']}."),
             status="not applicable (T=3 out of scope)",
             where_v2=(f"{PRO_KEYS('records[q].protocol.warmup, stability_every, '
                                   'verifier_timeout')}; "
                       "results/v2_pilots/pilot4/analysis/verifier_consumes_no_rng.csv; "
                       "tests/test_v2_verifier.py::test_evaluate_consumes_no_rng"),
             legacy_source=f"{R_P0} section 2 (verifier cadence row)"),
        dict(block=B_, item="Legacy runner defaults",
             legacy_behaviour=("run/run_final_dp_br.py code defaults differ from the as-run "
                               "manifest (verifier_timeout "
                               "100 vs A 100 / B 25 / C 25; k_stop 5 vs 1; phase_thr C 0.015 vs "
                               "0.01; weights_every "
                               "absent vs 25); they matter only for the round-1 main()."),
             risk="A run started with code defaults would silently differ from the as-run "
                  "protocol.",
             v2_handling=("Every v2 run embeds the archived as-run record with strict key checks "
                          "(verifier tiers must "
                          "equal DEV_CONFIG / FINAL_CONFIG); the locked entry point reads only "
                          f"{PROTO} (SHA-256 checked against the code and protocols/LOCK) and "
                          f"refuses any argument other "
                          "than --q, --seed, --out-dir."),
             status="addressed",
             where_v2=(f"{loc(RS, 'validate_config')}; {loc(RS, 'Run.__init__', 'strict_keys(P')}; "
                       f"{loc(LK, 'load_protocol')}; {loc(LK, 'parse_args')}; {loc(LK, 'build_config')}; "
                       f"{R_INFRA} section 1 points 1-2"),
             legacy_source=f"{R_P0} section 2 (config vs code-default table)"),
        dict(block=B_, item="One actor shared by all stages",
             legacy_behaviour="One actor for both stages; stage only an input feature.",
             risk="Cross-stage interference through shared parameters.",
             v2_handling="Actor still shared; the stage-2 mapping is frozen for Phase B (see block "
                         "a, 'Shared actor').",
             status="addressed (by freezing; the actor is still shared)",
             where_v2=f"{loc(RS, 'Run.policy_fns')}; {loc(AG2, 'CurriculumPPOv2.freeze_stage2_snapshot')}",
             legacy_source=f"{R_P0} section 1.4"),
        dict(block=B_, item="No explicit trainable-stage mask",
             legacy_behaviour="All rows of all visited stages enter the policy loss; there is no "
                              "stage mask anywhere.",
             risk="Stage 2 cannot be held fixed while stage 1 trains.",
             v2_handling=("CurriculumPPOv2.update(policy_mask, norm_mask): actor loss over the "
                          "policy rows only (other "
                          "rows never enter the graph), normalization over the norm_mask rows, "
                          "unchanged minibatch "
                          "permutation; every row carries a stage label; joint mode calls the "
                          "original update "
                          "(bit-identical)."),
             status="addressed",
             where_v2=(f"{loc(AG2, 'CurriculumPPOv2.update', 'rows = idx[pm[idx]]')}; {loc(RO, 'collect_batch_v2', 'stage = np.concatenate')}; "
                       "tests/test_v2_infra.py (test_gradient_isolation, "
                       "test_b2_update_ignores_stage2_advantages_end_to_end, "
                       "test_joint_update_bit_identical_to_original)"),
             legacy_source=f"{R_P0} sections 1.5 and 4"),
        dict(block=B_, item="Budget-forced continuation",
             legacy_behaviour=("A phase that reaches its cap without verifier passes continues "
                               "into the next phase "
                               "('budget_exhausted'); in the 80 latest runs every A ended by "
                               "budget and continued to B."),
             risk="Later stages train on top of an unverified earlier stage without a record of "
                  "the failure.",
             v2_handling=(f"Phase B always runs (fixed budget), but: "
                          f"'{pr['gates']['run_outcome']}'."),
             status="addressed",
             where_v2=f"{PRO_KEYS('gates.run_outcome')}; {loc(LK, 'run_pipeline')}; {loc(LK, 'verdicts')}",
             legacy_source=f"{R_P0} section 3"),
        dict(block=B_, item="KL only monitored",
             legacy_behaviour=("Approximate KL mean((r-1) - log r) computed after every epoch and "
                               "logged; no KL penalty "
                               f"or early stop; KL enters only the stability trigger of verifier "
                               f"calls (kl_thr "
                               f"{P50['kl_thr']:g})."),
             risk="Large policy steps are not limited beyond the PPO clip.",
             v2_handling=("Unchanged: KL is still only monitored (kl_final_epoch per update in "
                          "v2_updates.csv). The "
                          "locked LR decay windows lowered KL and clip fraction in Pilot 4 "
                          "(reports/v2/"
                          "pilot4_stabilization.md, advantage statistics, KL and clip)."),
             status="open (unchanged)",
             where_v2=f"{loc(AG, 'CurriculumPPO.update', 'kl_epochs.append')}; {loc(RS, 'Run.run_phase', 'kl_final_epoch')}",
             legacy_source=f"{R_P0} sections 1.5 and 3"),
        dict(block=B_, item="Likelihood of clipped Beta samples",
             legacy_behaviour=(f"Actions sampled with numpy Beta in float64, clipped to [1e-6, 1 - "
                               f"1e-6] and stored as "
                               f"float32; old and new log-probabilities are Beta densities at the "
                               f"stored (clipped) action "
                               f"(record action_sampling: '{pr['records']['50']['action_sampling']}')."),
             risk="The density of the clipped value is used where the sample may be a clipped "
                  "point mass.",
             v2_handling=("Unchanged in Phase A and for the stage-1 rows of Phase B (stage-2 rows "
                          "of Phase B do not enter "
                          "the policy loss; in mean mode their stored action is the clipped Beta "
                          "mean). No report "
                          "addresses the likelihood consistency of clipped samples: files of "
                          "reports/v2/*.md and "
                          f"protocols/*.md mentioning 'likelihood' or "
                          f"'log-prob': {len(ev['lik_hits'])} of {ev['n_reports']}."),
             status="open (no report addresses it)",
             where_v2=(f"{loc(AG, 'CurriculumPPO.sample_actions')}; {loc(AG, 'CurriculumPPO.log_prob')}; "
                       f"{loc(RO, 'collect_batch_v2', 'act_L = np.clip(m_L')}"),
             legacy_source=f"{R_P0} section 2 (Beta row)"),
        dict(block=B_, item="dReach reach mask vs PMF support",
             legacy_behaviour=("R_2 = grid nodes covered by the closed interval [a^BR_1(0) - "
                               "e_hat_1(0) - 2q, "
                               "a^BR_1(0) - e_hat_1(0) + 2q] intersected with D_2; the GL-node PMF "
                               "of the BR chain is "
                               "recorded separately (PDL residual)."),
             risk="The reach mask could differ from the support of the stage-2 state law (holes or "
                  "extra nodes).",
             v2_handling=(f"Checked, dReach unchanged: over {ev['dm_n']} evaluations the official "
                          f"mask has "
                          f"{ev['dm_holes']} holes and {ev['dm_extra_nodes']} extra nodes, all on "
                          f"the closed boundary; "
                          f"dReach differs from the open-support reference "
                          f"in {ev['dm_nonzero']} evaluation(s), by at most "
                          f"{ev['dm_max']:.4g} DW ({ev['dm_neg']} negative differences). The GL "
                          f"node-PMF support "
                          "artefacts (+-2q included, interior holes; phase2_opening_checks.md "
                          "section 1c) led to the v2 "
                          "on-path rule (open interval, exact cell masses)."),
             status="addressed (checked; dReach unchanged)",
             where_v2=(f"{R_DR}; results/v2_pilots/dreach_mask_check/dreach_mask_check.csv (T10); "
                       f"{loc(VM, 'onpath_mask')}; {loc(VM, 'cell_masses')}"),
             legacy_source=f"{R_P0} section 1.8; {R_P1} section 2.3"),
        dict(block=B_, item="Recovery used only post hoc",
             legacy_behaviour="Recovery metrics against the closed form computed only in the final "
                              "evaluation.",
             risk="Stage-2 accuracy is not tracked during training and decides nothing.",
             v2_handling=("utils.v2_metrics.evaluate computes the recovery metrics at every "
                          "verifier call "
                          f"({ev['n_calls']} training-time calls logged "
                          f"in {ev['n_call_files']} v2_checkpoints*.csv "
                          "files) and at the gates; G-A gates on RMSE/e2*(0) and tail mean/e2*(0); "
                          "the closed form "
                          "never enters training."),
             status="addressed",
             where_v2=(f"{loc(VM, 'evaluate', 'rsc, rarr = recovery_metrics(')}; {loc(RS, 'Run.verify_candidate')}; "
                       f"{PRO_KEYS('gates.G-A')}"),
             legacy_source=f"{R_P0} section 1.9"),
        dict(block=B_, item="Gmax_full not a primary metric",
             legacy_behaviour=("v_br and v_mean are full-domain arrays, but max_t max_d (v_br - "
                               "v_mean) is never "
                               "aggregated (Phase 0 section 1.8)."),
             risk="Deviation gains off the BR-reachable set are not measured.",
             v2_handling=(f"Gmax_full (with t*, d*) computed by utils.v2_metrics.evaluate since "
                          f"Phase 1; gate G-F "
                          f"Gmax_full/DW <= {gf['Gmax_full_over_dw']:g} on the final tier; G-N "
                          f"|dev - final| <= "
                          f"{gn['Gmax_full_over_dw_dev_minus_final_abs']:g} DW."),
             status="addressed",
             where_v2=f"{loc(VM, 'evaluate', 'Gmax_full_over_dw')}; {PRO_KEYS('gates.G-F, gates.G-N')}",
             legacy_source=f"{R_P0} section 1.8"),
        dict(block=B_, item="Branching that loads weights only",
             legacy_behaviour=("CurriculumPPO.load_weights restores actor, critic and opponent "
                               "weights only; no RNG "
                               "states, optimizer restore or counters (Phase 0 section 7)."),
             risk="A branch is not the continuation of its parent; paired arms are not exactly "
                  "comparable.",
             v2_handling=f"Full-state restore of weights, optimizers, every RNG stream and the "
                         f"counters; parent SHA-256 check; {repro}.",
             status="addressed",
             where_v2=f"{loc(RS, 'Run.restore')}; {loc(RS, 'main', 'parent sha256')}; "
                      f"tests/test_v2_infra.py::test_continuous_equals_branched",
             legacy_source=f"{R_P0} section 7; {loc(AG, 'CurriculumPPO.load_weights')}"),
    ]
    df = pd.DataFrame(rows)
    srcs = [C.src(p) for p in (R_P0, R_P1, R_OPEN, R_INFRA, R_DR, "reports/v2/pilot2_freeze.md",
                               "reports/v2/pilot3_continuation_mode.md", "reports/v2/phaseA_ext.md",
                               "reports/v2/pilot4_stabilization.md", PROTO, RS, AG2, RO, LK, AG, VM,
                               "run/run_final_dp_br.py", "tests/test_v2_infra.py", "tests/test_v2_verifier.py")]
    srcs += ev["sources"] + [C.srcs(ev["report_files"], label="reports/v2/*.md and protocols/*.md "
                                                              "(keyword searches)")]
    docs = {"block": "a = Phase 0 audit item (phase0_audit.md); b = item of the PI's 0929 issue "
                     "list (SPEC 5.3)",
            "item": "Audit item or issue",
            "legacy_behaviour": "Behaviour of the legacy T=2 pipeline (source: report text of "
                                "legacy_source)",
            "risk": "Risk of the legacy behaviour, stated at the mechanism level (not a measured "
                    "effect)",
            "v2_handling": "How the v2 code and the locked v1.1 protocol handle it (verified in "
                           "the code cited in where_v2)",
            "status": "addressed / unchanged / not applicable / open",
            "where_v2": "Code (file:function with line numbers read from the file), protocol keys, "
                        "tests and reports",
            "legacy_source": "Report section (and the legacy code lines it cites)"}
    pack.table("T05", df, status="generated", sources=srcs, script=f"{MOD}:build_t05", docs=docs,
               tier="n/a",
               notes=("source: report text (phase0_audit.md and later reports) plus code lines; "
                      "every v2 statement "
                      "checked in the cited code; counts and thresholds inserted from the protocol "
                      "JSON and the "
                      "analysis CSVs; keyword searches run at build time over reports/v2/*.md and "
                      "protocols/*.md."),
               caption=("Block a: the six Phase 0 audit items named in the request. Block b: the "
                        "PI's 0929 issue list "
                        "(SPEC 5.3), each with status addressed / not applicable / open and where "
                        "it is handled."))


def PRO_KEYS(keys: str) -> str:
    """Citation of protocol JSON keys."""
    return f"{PROTO} {keys}"


# ----------------------------------------------------------------------------------------------
# T06 resolved configuration of the locked v1.1 protocol
# ----------------------------------------------------------------------------------------------

def _t06_group(key: str) -> str:
    """Report group of a flattened protocol-record key."""
    if key.startswith("game.") or key in ("q", "T", "dw", "B", "domain_half_stage2"):
        return "game"
    if key.startswith("ppo."):
        return "PPO"
    if key.startswith("protocol."):
        k = key.split(".", 1)[1]
        if k in ("w_h", "w_l", "k", "e_min", "e_max"):
            return "game (protocol copy)"
        if k.startswith("phase_caps") or k in ("episodes_per_update", "phase_c_root", "phase_c_es"):
            return "budgets and phases"
        if k in ("warmup", "stability_every", "drift_thr", "kl_thr", "stability_consecutive") or \
                k.startswith("verifier_timeout"):
            return "verifier cadence"
        if k in ("k_phase", "k_stop", "conc_thr") or k.startswith("phase_thr_over_dw"):
            return "legacy phase rules (would-have-fired only)"
        if k == "snapshot_every":
            return "opponent"
        if k == "es_bin_width":
            return "starts"
        if k == "recovery_step":
            return "recovery grid"
        if k.startswith("rng_namespaces"):
            return "RNG namespaces"
        if k == "weights_every":
            return "outputs"
        return "legacy final-evaluation settings"
    if key in ("obs_encoding", "action_mapping", "mean_extraction", "action_sampling"):
        return "observation, action and evaluation convention"
    if key == "es_bins_stage2":
        return "starts"
    if key.startswith("verifier."):
        return "verifier tiers"
    if key.startswith("dtypes."):
        return "precision"
    if key in ("device", "torch_threads"):
        return "threads and device"
    if key.startswith("versions."):
        return "versions"
    if key.startswith("lr_schedule."):
        return "LR schedule (record)"
    return "other"


def build_t06(pack: C.Pack) -> None:
    """T06: every setting of the locked v1.1 protocol per q, marked where it differs from the legacy as-run record."""
    pr = _proto()
    pl = pr["pipeline"]
    recs = {q: pr["records"][str(q)] for q in QS}
    legs = {q: _legacy_record(q) for q in QS}
    flat = {q: _flatten(recs[q]) for q in QS}
    lflat = {q: _flatten(legs[q][0]) for q in QS}
    man_rel = {q: [f"{CONF}/q{q}/seed{s}/manifest.json" for s in S.SEEDS_CONF] for q in QS}
    mans = {q: [_json(r) for r in man_rel[q]] for q in QS}
    cflat = {q: [_flatten(m["input_config"]["record"]) for m in mans[q]] for q in QS}
    per_run = {"run", "seed", "output_dir"}
    keys = [k for k in flat[50] if k not in per_run]
    assert set(keys) == set(k for k in flat[60] if k not in per_run)
    _, o0, o1 = _assign("run/run_v2_stagewise.py", "OVERRIDE_KEYS")
    v2_code = [ln for i, ln in enumerate(_lines("run/run_v2_stagewise.py"),
                                         1) if not o0 <= i <= o1] + \
        _lines("run/run_v2_T2_locked.py")
    lr_win = {w["phase"]: w for w in pl["lr_decay"]}
    rows: List[Dict[str, Any]] = []
    for key in keys:
        grp = _t06_group(key)
        v = {q: flat[q][key] for q in QS}
        lv = {q: lflat[q].get(key, "absent") for q in QS}
        note = ""
        k = key.split(".", 1)[1] if key.startswith("protocol.") else key
        if grp == "legacy final-evaluation settings":
            used = [ln for ln in v2_code if f'"{k}"' in ln]
            if not used:
                note = ("not read by run/run_v2_stagewise.py or run/run_v2_T2_locked.py (required "
                        "by the strict record "
                        "check; read only by the legacy final evaluation, mode full)")
        elif grp.startswith("legacy phase rules"):
            note = "fixed budget: the rule only sets would_have_fired (v2_run_summary.json); no early exit"
        elif key in ("protocol.phase_caps.C", "protocol.verifier_timeout.C",
                     "protocol.phase_c_root",
                     "protocol.phase_c_es"):
            note = "Phase C is not run (pipeline.not_used)"
        elif key == "protocol.phase_caps.A":
            note = f"pipeline.phase_A.updates = {pl['phase_A']['updates']}; lr_decay window A ends at the cap"
        elif grp == "LR schedule (record)":
            note = ("the record keeps the legacy constant schedule; phases A and B use lr_at's A/B "
                    "branch (ab_lr) "
                    "outside the pipeline LR windows (rows 'LR, Phase A/B' of group pipeline)")
        elif key == "group":
            note = "label carried over from the archived record"
        rows.append({
            "group": grp, "setting": key, "v1_1_q50": _txt(v[50]), "v1_1_q60": _txt(v[60]),
            "legacy_q50": _txt(lv[50]), "legacy_q60": _txt(lv[60]),
            "differs_from_legacy": bool(any(v[q] != lv[q] for q in QS)),
            "confirmation_runs_q50": int(sum(cf.get(key) == v[50] for cf in cflat[50])),
            "confirmation_runs_q60": int(sum(cf.get(key) == v[60] for cf in cflat[60])),
            "source_v1_1": f"{PROTO} records[q].{key}",
            "source_legacy": f"{LEGACY[50]} run {legs[50][1]} / {LEGACY[60]} run {legs[60][1]}: {key}",
            "note": note})
    # ---- pipeline-level settings (not in the per-q record)
    dflags = _assign("run/run_v2_stagewise.py", "DEFAULT_FLAGS")[0]
    leg_man = _json(LEGACY[50])
    conf_top = {q: [(m["mode"], m["fixed_budget"], m["flags"],
                     m["resolved_config"]["lr_decay"]) for m in mans[q]]
                for q in QS}

    def prow(setting: str, v11: Any, legacy: Any, differs: bool, src: str, lsrc: str,
             note: str = "",
             conf: Optional[Callable[[Tuple], bool]] = None) -> None:
        rows.append({"group": "pipeline (v1.1)", "setting": setting, "v1_1_q50": _txt(v11),
                     "v1_1_q60": _txt(v11),
                     "legacy_q50": _txt(legacy), "legacy_q60": _txt(legacy), "differs_from_legacy": differs,
                     "confirmation_runs_q50": int(sum(conf(c) for c in conf_top[50])) if conf else np.nan,
                     "confirmation_runs_q60": int(sum(conf(c) for c in conf_top[60])) if conf else np.nan,
                     "source_v1_1": src, "source_legacy": lsrc, "note": note})

    prow("entry point", pl["entry_point"], leg_man["runner"], True, f"{PROTO} pipeline.entry_point",
         f"{LEGACY[50]} runner")
    prow("mode", f"{pl['mode']}: Phase A -> G-A -> freeze -> Phase B -> G-F, G-N (one process)",
         "Phase A -> Phase B -> Phase C with the legacy phase "
         "rules", True, f"{PROTO} pipeline.mode",
         f"{R_P0} sections 1.6 and 3", conf=lambda c: c[0] == pl["mode"])
    prow("fixed_budget", pl["fixed_budget"], False, True, f"{PROTO} pipeline.fixed_budget",
         f"{R_P0} section 3",
         conf=lambda c: c[1] is True)
    for fk, fv in pl["flags"].items():
        prow(f"flags.{fk}", fv, dflags[fk], fv != dflags[fk], f"{PROTO} pipeline.flags.{fk}",
             "run/run_v2_stagewise.py DEFAULT_FLAGS (reproduce the legacy runner bit-exactly, C7)",
             note=pl["flags_note"] if fk != "reward_mode" else "",
             conf=lambda c, fk=fk, fv=fv: c[2][fk] == fv)
    prow("Phase A updates", pl["phase_A"]["updates"], lflat[50]["protocol.phase_caps.A"], True,
         f"{PROTO} pipeline.phase_A.updates", f"{LEGACY[50]} protocol.phase_caps.A",
         note="legacy: cap with the k_phase early-exit rule, which never fired in the 80 latest "
              "runs")
    prow("Phase A active stages", pl["phase_A"]["active_stages"], [2], False,
         f"{PROTO} pipeline.phase_A.active_stages",
         f"{R_P0} section 1.6")
    prow("Phase A starts", pl["phase_A"]["starts"], "bin-balanced exploring starts on D_2 "
                                                    "(es_bin_width 10)", False,
         f"{PROTO} pipeline.phase_A.starts", f"{R_P0} section 5")
    prow("freeze", pl["freeze"], "none (stage 2 keeps training in Phases B and "
                                 "C)", True, f"{PROTO} pipeline.freeze",
         f"{R_P0} section 4")
    prow("Phase B updates", pl["phase_B"]["updates"], lflat[50]["protocol.phase_caps.B"], False,
         f"{PROTO} pipeline.phase_B.updates", f"{LEGACY[50]} protocol.phase_caps.B",
         note="legacy: cap with the k_phase early-exit rule on EXP_root; v1.1: fixed")
    prow("Phase B active stages", pl["phase_B"]["active_stages"], [1, 2], True,
         f"{PROTO} pipeline.phase_B.active_stages",
         f"{R_P0} sections 1.6 and 4")
    prow("Phase B starts", pl["phase_B"]["starts"], "root (t=1, d=0)", False,
         f"{PROTO} pipeline.phase_B.starts",
         f"{R_P0} section 1.6")
    prow("Phase B opponent", pl["phase_B"]["opponent"],
         "lagged copy of the actor, refreshed every 20 global updates and at phase entry", False,
         f"{PROTO} pipeline.phase_B.opponent", f"{R_P0} section 1.4")
    prow("LR, Phase A", pl["phase_A"]["lr"], f"constant {lflat[50]['lr_schedule.ab_lr']:g}", True,
         f"{PROTO} pipeline.phase_A.lr, pipeline.lr_decay[A]", f"{LEGACY[50]} lr_schedule (kind "
                                                               f"constant)",
         note=f"window {lr_win['A']['local_first']}-{lr_win['A']['local_last']}: {lr_win['A']['start_lr']:g} -> "
              f"{lr_win['A']['end_lr']:g}", conf=lambda c: any(w == lr_win["A"] for w in c[3]))
    prow("LR, Phase B", pl["phase_B"]["lr"], f"constant {lflat[50]['lr_schedule.ab_lr']:g}", True,
         f"{PROTO} pipeline.phase_B.lr, pipeline.lr_decay[B]", f"{LEGACY[50]} lr_schedule (kind "
                                                               f"constant)",
         note=f"window {lr_win['B']['local_first']}-{lr_win['B']['local_last']}: {lr_win['B']['start_lr']:g} -> "
              f"{lr_win['B']['end_lr']:g}", conf=lambda c: any(w == lr_win["B"] for w in c[3]))
    prow("LR form", pl["lr_form"], "lr_at kind constant: ab_lr in A and B, c_start_lr = c_end_lr "
                                   "in C", True,
         f"{PROTO} pipeline.lr_form", "run/run_final_dp_br_round3_dense.py:lr_at with the record "
                                      "lr_schedule")
    prow("Phase C", "not used",
         (f"{lflat[50]['protocol.phase_c_root']} root "
          f"+ {lflat[50]['protocol.phase_c_es']} stage-2 exploring starts, "
          f"cap {lflat[50]['protocol.phase_caps.C']}, stop at the first eligible call (k_stop "
          f"{lflat[50]['protocol.k_stop']})"), True, f"{PROTO} pipeline.not_used", f"{LEGACY[50]} protocol; {R_P0} section 3")
    prow("evaluated candidate", pl["evaluated_candidate"],
         "checkpoint at the C stopping point (first eligible C call) or at C budget exhaustion; no "
         "best-checkpoint restore",
         True, f"{PROTO} pipeline.evaluated_candidate", f"{R_P0} sections 1.7 and 3")
    prow("tail averaging", "not used", "not used", False, f"{PROTO} pipeline.not_used",
         f"{R_P0} (no averaging in the legacy runner)")
    prow("process-global RNGs", pr["global_rng_hardening"]["seeding"],
         "not seeded; the torch generator is seeded from the init stream and used only for the "
         "weight init",
         True, f"{PROTO} global_rng_hardening.seeding",
         f"{R_P0} section 2 (RNG row); {PROTO} change_log (v1.1: 'three unseeded, never-consumed "
         f"process-global RNG "
         "states differed between processes')")
    prow("global-RNG assertion points", pr["global_rng_hardening"]["assertion_points"], "none",
         True,
         f"{PROTO} global_rng_hardening.assertion_points", f"{R_P0} section 7")
    prow("threads per process", pr["threads_per_process"], leg_man["threads_per_process"],
         pr["threads_per_process"] != leg_man["threads_per_process"], f"{PROTO} threads_per_process",
         f"{LEGACY[50]} threads_per_process")
    # ---- network and Beta parameterization (code; identical in the legacy runner)
    AG = "agents/ppo_curriculum.py"
    code_rows = [
        ("actor network", "2 -> 64 tanh -> 64 tanh -> 2 (z_mu, z_c)", loc(AG, "BetaActor")),
        ("Beta parameterization", "mu = clamp(sigmoid(z_mu), mu_clamp, 1 - mu_clamp); c = c_min + "
                                  "softplus(z_c); "
                                  "alpha = mu c, beta = (1 - mu) c", loc(AG, "BetaActor.forward")),
        ("critic network", "separate 2 -> 64 tanh -> 64 tanh -> 1; no shared "
                           "trunk", loc(AG, "Critic")),
        ("initialization", "hidden layers orthogonal gain sqrt(2), bias 0; actor head weight 0, "
                           "bias 0 (initial c = "
                           "c_min + ln 2, mean 0.5); critic head orthogonal gain 1; torch "
                           "generator seeded from the init "
                           "stream", loc(AG, "BetaActor.__init__")),
        ("optimizers and clipping", "separate Adam per network; separate global grad-norm clip per "
                                    "network",
         loc(AG, "CurriculumPPO.__init__")),
        ("KL diagnostic", "mean((r - 1) - log r) over the buffer after every epoch; monitored, not "
                          "used in the loss",
         loc(AG, "CurriculumPPO.update", "kl_epochs.append")),
    ]
    for setting, txt, where in code_rows:
        rows.append({"group": "network and Beta parameterization "
                              "(code)", "setting": setting, "v1_1_q50": txt,
                     "v1_1_q60": txt, "legacy_q50": txt, "legacy_q60": txt, "differs_from_legacy": False,
                     "confirmation_runs_q50": np.nan, "confirmation_runs_q60": np.nan, "source_v1_1": where,
                     "source_legacy": f"{where} (file unchanged since base "
                                      f"657f54a; {R_P0} section 1.4)",
                     "note": "values of hidden, c_min, mu_clamp, action_clamp: rows ppo.*"})
    rows.append({"group": "network and Beta parameterization "
                          "(code)", "setting": "advantage normalization",
                 "v1_1_q50": "(adv - mean)/(population SD + adv_norm_eps); Phase A all rows; Phase "
                             "B stage-1 rows only",
                 "v1_1_q60": "(adv - mean)/(population SD + adv_norm_eps); Phase A all rows; Phase "
                             "B stage-1 rows only",
                 "legacy_q50": "(adv - mean)/(population SD + adv_norm_eps) over all rows of all "
                               "stages",
                 "legacy_q60": "(adv - mean)/(population SD + adv_norm_eps) over all rows of all "
                               "stages",
                 "differs_from_legacy": True, "confirmation_runs_q50": np.nan, "confirmation_runs_q60": np.nan,
                 "source_v1_1": f"{loc('agents/ppo_curriculum_v2.py', 'CurriculumPPOv2.update')}; {PROTO} pipeline.phase_B.adv_norm_scope",
                 "source_legacy": f"{loc(AG, 'CurriculumPPO.update', 'adv_std = adv_raw.std(unbiased=False)')}; {R_P0} section 1.5",
                 "note": ""})
    df = pd.DataFrame(rows)
    order = ["pipeline (v1.1)", "game", "game (protocol copy)", "budgets and phases", "starts",
             "opponent",
             "LR schedule (record)", "PPO", "network and Beta parameterization (code)",
             "observation, action and evaluation convention", "verifier tiers", "verifier cadence",
             "legacy phase rules (would-have-fired "
             "only)", "recovery grid", "RNG namespaces", "precision",
             "threads and device", "versions", "outputs", "legacy final-evaluation settings", "other"]
    df["_o"] = df["group"].map({g: i for i, g in enumerate(order)})
    df = df.sort_values(["_o"], kind="stable").drop(columns="_o").reset_index(drop=True)
    for c in ("confirmation_runs_q50", "confirmation_runs_q60"):
        df[c] = pd.Series([None if pd.isna(x) else int(x) for x in df[c]], dtype=object)
    srcs = [C.src(PROTO), C.src(LEGACY[50], f"runs[{legs[50][1]}]"),
            C.src(LEGACY[60], f"runs[{legs[60][1]}]"),
            C.src("run/run_v2_stagewise.py"), C.src("run/run_v2_T2_locked.py"), C.src(AG),
            C.src("agents/ppo_curriculum_v2.py"), C.src(R_P0),
            C.srcs([r for q in QS for r in man_rel[q]], label=f"{CONF}/q*/seed*/manifest.json",
                   expect=40)]
    docs = {"group": "Setting group", "setting": "Setting (flattened key of the protocol record, "
                                                 "or pipeline/code item)",
            "v1_1_q50": "Value in the locked v1.1 protocol at q = 50 (text; JSON for lists)",
            "v1_1_q60": "Value in the locked v1.1 protocol at q = 60",
            "legacy_q50": "Legacy as-run value at q = 50 (archived record tabulated in Phase 0 "
                          "section 2, or the legacy "
                          "behaviour named in source_legacy)",
            "legacy_q60": "Legacy as-run value at q = 60",
            "differs_from_legacy": "True where the v1.1 value differs from the legacy as-run value "
                                   "at either q",
            "confirmation_runs_q50": dict(definition="Number of the 20 q = 50 confirmation runs "
                                                     "whose manifest records "
                                                     "this value (blank: not recorded per "
                                                     "run)", units="runs (of 20)"),
            "confirmation_runs_q60": dict(definition="Number of the 20 q = 60 confirmation runs "
                                                     "whose manifest records "
                                                     "this value (blank: not recorded per "
                                                     "run)", units="runs (of 20)"),
            "source_v1_1": "Protocol key or code location of the v1.1 value",
            "source_legacy": "File / key (or report section) of the legacy value",
            "note": "Remark on how the setting acts in the locked pipeline"}
    pack.table("T06", df, status="generated", sources=srcs, script=f"{MOD}:build_t06", docs=docs,
               tier="n/a",
               notes=("Protocol record flattened key by key (per-run keys run, seed and output_dir "
                      "omitted); legacy values "
                      "read from the archived as-run records that Phase 0 section 2 tabulates "
                      "(q=50 "
                      f"{legs[50][1]}, q=60 {legs[60][1]}); pipeline-level and code rows compiled "
                      f"from the protocol "
                      "pipeline block, phase0_audit.md (source: report text) and the agent code; "
                      "confirmation_runs_* counts the 20 confirmation manifests per q "
                      "(input_config.record, mode, "
                      "fixed_budget, flags, resolved lr_decay) that carry the same value."),
               caption=("Resolved configuration of the locked v1.1 protocol per q; "
                        "differs_from_legacy marks the values "
                        "that differ from the legacy as-run configuration of Phase 0 section 2."))
    # ---- cross-check of the legacy column with the Phase 0 section 2 table
    xc = XCheck(pack, "T06", R_P0, "legacy as-run values (section 2 table)")
    L = lflat[50]
    m = xc.find(r"\| Phase caps \| A (\d+), B (\d+), C (\d+) \|")
    if m:
        for i, ph in enumerate("ABC"):
            xc.cmp(f"legacy phase_caps.{ph}", L[f"protocol.phase_caps.{ph}"], m.group(i + 1),
                   "section 2")
    m = xc.find(r"\| Episodes per update \| (\d+) \|")
    if m:
        xc.cmp("legacy episodes_per_update", L["protocol.episodes_per_update"], m.group(1),
               "section 2")
    m = xc.find(r"\| Exploring-start bins \| width (\d+) . (\d+) / (\d+) bins")
    if m:
        xc.cmp("legacy es_bin_width", L["protocol.es_bin_width"], m.group(1), "section 2")
        xc.cmp("legacy es_bins_stage2 (q=50)", lflat[50]["es_bins_stage2"], m.group(2), "section 2")
        xc.cmp("legacy es_bins_stage2 (q=60)", lflat[60]["es_bins_stage2"], m.group(3), "section 2")
    m = xc.find(r"\| PPO \| lr (" + _NUM + r") constant \(actor and critic\), Adam "
                                           r"\((" + _NUM + r"), (" + _NUM +
                r"), (" + _NUM + r")\), wd (\d+), clip (" + _NUM + r"), value coef (" + _NUM + r"), grad-norm (" +
                _NUM + r"), (\d+) epochs, minibatch (\d+)")
    if m:
        for i, k in enumerate(["ppo.lr", "ppo.adam_betas.0", "ppo.adam_betas.1", "ppo.adam_eps",
                               "ppo.weight_decay",
                               "ppo.clip_eps", "ppo.value_coef", "ppo.max_grad_norm", "ppo.epochs", "ppo.minibatch"]):
            val = L["ppo.adam_betas"][int(k[-1])] if k.startswith("ppo.adam_betas") else L[k]
            xc.cmp(f"legacy {k}", val, m.group(i + 1), "section 2")
    m = xc.find(r"\| Stop / advance \| k_phase (\d+) \(A, B\), k_stop (\d+) \(C\); thresholds A "
                r"(" + _NUM +
                r") .*?, B (" + _NUM + r") .*?, C (" + _NUM + r") .*?; concentration max std/range "
                                                              r". (" + _NUM + r")")
    if m:
        for i, k in enumerate(["protocol.k_phase", "protocol.k_stop",
                               "protocol.phase_thr_over_dw.A",
                               "protocol.phase_thr_over_dw.B", "protocol.phase_thr_over_dw.C", "protocol.conc_thr"]):
            xc.cmp(f"legacy {k}", L[k], m.group(i + 1), "section 2")
    m = xc.find(r"\| Opponent snapshot \| refresh every (\d+) global updates")
    if m:
        xc.cmp("legacy snapshot_every", L["protocol.snapshot_every"], m.group(1), "section 2")
    m = xc.find(r"\| Weights export \| every (\d+) global updates")
    if m:
        xc.cmp("legacy weights_every", L["protocol.weights_every"], m.group(1), "section 2")
    m = xc.find(r"\| Versions \(enforced\) \| Python (\d+\.\d+\.\d+), torch (\S+), numpy "
                r"(\d+\.\d+\.\d+)")
    if m:
        for i, k in enumerate(["python", "torch", "numpy"]):
            xc.same(f"legacy versions.{k}", L[f"versions.{k}"], m.group(i + 1), "section 2")
    m = xc.find(r"namespaces init (\d+), env_noise (\d+), learner_action (\d+), opponent_action "
                r"(\d+), "
                r"starts_roles (\d+), minibatch (\d+)")
    if m:
        for i, k in enumerate(["init", "env_noise", "learner_action", "opponent_action",
                               "starts_roles", "minibatch"]):
            xc.cmp(f"legacy rng_namespaces.{k}", L[f"protocol.rng_namespaces.{k}"], m.group(i + 1),
                   "section 2")
    xc.done()
    xc = XCheck(pack, "T06", "protocols/v2_T2_locked.md", "v1 protocol document: budgets and LR "
                                                          "windows (section 2)")
    m = xc.find(r"Phase A: stage 2 only, (\d+) updates")
    if m:
        xc.cmp("Phase A updates", pl["phase_A"]["updates"], m.group(1), "section 2")
    m = xc.find(r"Phase B: stage 1 only, (\d+) updates")
    if m:
        xc.cmp("Phase B updates", pl["phase_B"]["updates"], m.group(1), "section 2")
    m = xc.find(r"constant for local updates (\d+)[\u2013-](\d+), then linear "
                r"(" + _NUM + r") . (" + _NUM +
                r") over local updates (\d+)[\u2013-](\d+)")
    if m:
        w = lr_win["A"]
        for i, val in enumerate([1, w["local_first"] - 1, w["start_lr"], w["end_lr"],
                                 w["local_first"], w["local_last"]]):
            xc.cmp(f"LR window A field {i}", val, m.group(i + 1), "section 2")
    xc.done()


# ----------------------------------------------------------------------------------------------
# T07 metric definitions
# ----------------------------------------------------------------------------------------------

def build_t07(pack: C.Pack) -> None:
    """T07: definition, region, deviation type, aggregation, normalization and implementing code of every metric."""
    DV, VM = "utils/dp_br_verifier.py", "utils/v2_metrics.py"
    P4C, LK = "tools/v2/pilot4_common.py", "run/run_v2_T2_locked.py"
    op = _csv("results/v2_pilots/phase2_opening/onpath_sets.csv")
    n_drift0, n_drift = int((op["stage1_drift"] == 0.0).sum()), len(op)
    rec = {q: _t04_values(q) for q in QS}
    nn = f"{rec[50]['rec_pos']} / {rec[60]['rec_pos']} nodes (q = 50 / 60)"
    nt = f"{rec[50]['rec_tail']} / {rec[60]['rec_tail']} nodes"
    rows = [
        dict(symbol="Delta_t(d)", definition=("one-step deviation gap: max over {effort grid E; "
                                              "valid concave-parabola "
                                              "vertices of Q^mean; e_hat_t(d); a^BR_t(d)} of "
                                              "Q^mean_t(d, e) - V^e_hat_t(d), "
                                              "deviating at t only and continuing with e_hat (ties "
                                              "to the smaller effort)"),
             state_region="every node of D_t", deviation_type="one-step (deviate at t, then follow "
                                                              "e_hat)",
             aggregation="per node", normalization="/DW where reported",
             implementing=loc(DV, "verify",
                              "delta = vdev - v_mean"), column_name="(arrays v_t{t}_delta)"),
        dict(symbol="G_t(d)", definition=("full dynamic deviation gain V^BR_t(d) - V^e_hat_t(d); "
                                          "V^BR re-optimizes at t "
                                          "and at every later stage against the opponent "
                                          "e_hat_t(-d); V^e_hat follows e_hat"),
             state_region="every node of D_t", deviation_type="dynamic (multi-stage best response)",
             aggregation="per node", normalization="/DW where reported",
             implementing=f"{loc(VM, 'evaluate', 'G = {t: s.v_br - s.v_mean')}; V^BR "
                          f"in {loc(DV, 'verify', 'v_br, a_br, a_br_src = _select')}",
             column_name="(arrays v_t{t}_G)"),
        dict(symbol="EXP_root", definition="V^BR_1(0) - V^e_hat_1(0) = G_1(0): gain of the best "
                                           "dynamic deviation from the root",
             state_region="root (t = 1, d = 0)", deviation_type="dynamic", aggregation="single value",
             normalization="/DW", implementing=loc(DV, "verify",
                                                   "exp_root = float("), column_name="EXP_root_over_dw"),
        dict(symbol="dReach", definition=("sum_t max_{d in R_t} Delta_t(d); R_1 = {0}; R_2 = grid "
                                          "nodes covered by the closed "
                                          "interval [a^BR_1(0) - e_hat_1(0) - 2q, a^BR_1(0) - "
                                          "e_hat_1(0) + 2q] "
                                          "intersected with D_2 (tolerance 1e-9)"),
             state_region="BR-reachable nodes R_t (support of the state law on the root BR path)",
             deviation_type="one-step", aggregation="sum over stages of the per-stage "
                                                    "maximum", normalization="/DW",
             implementing=loc(DV, "verify", "stages[t + 1].reach[:] = mask",
                              "dreach = float(sum(reach_dm.values()))"),
             column_name="dReach_over_dw"),
        dict(symbol="Delta_max_all", definition="max_t max_{d in D_t} Delta_t(d)",
             state_region="every node of D_1 and D_2",
             deviation_type="one-step", aggregation="maximum over stages and nodes", normalization="/DW",
             implementing=loc(DV, "verify", "delta_max_all = float(max(full_dm.values()))"),
             column_name="Deltamax_all_over_dw"),
        dict(symbol="Gmax_full (t*, d*)", definition="max_t max_{d in D_t} G_t(d), with the stage "
                                                     "t* and node d* attaining it",
             state_region="full domain: D_1 = {0} and every node of D_2 = [-B, "
                          "B]", deviation_type="dynamic",
             aggregation="maximum over stages and nodes", normalization="/DW",
             implementing=loc(VM, "evaluate", "for t in sorted(G):",
                              '"Gmax_full_over_dw": gmax / dw'),
             column_name="Gmax_full_over_dw, Gmax_full_t, Gmax_full_d"),
        dict(symbol="dFull", definition="sum_t max_{d in D_t} Delta_t(d)",
             state_region="every node of D_1 and D_2",
             deviation_type="one-step", aggregation="sum over stages of the per-stage "
                                                    "maximum", normalization="/DW",
             implementing=loc(DV, "verify",
                              "dfull = float(sum(full_dm.values()))"), column_name="dFull_over_dw"),
        dict(symbol="eta_2", definition=("max_{d in D_2} Delta_2(d); at the last stage Delta_2 = "
                                         "G_2 node by node, so "
                                         "eta_2 = max_{d in D_2} G_2(d)"),
             state_region="every node of D_2 (full stage-2 "
                          "grid)", deviation_type="one-step (= dynamic at t = T)",
             aggregation="maximum over nodes", normalization="/DW",
             implementing=loc(VM, "evaluate", '"eta_T_over_dw": res.full_delta_max[spec.T] / dw'),
             column_name="eta_T_over_dw"),
        dict(symbol="stage-1 relative error", definition=("(e_hat_1(0) - e1*(0))/e1*(0), signed; "
                                                          "|.| is the S1 criterion "
                                                          "(v1.0: part of G-F)"),
             state_region="root (t = 1, d = 0)", deviation_type="none (recovery against the closed "
                                                                "form)",
             aggregation="single value", normalization="fraction of e1*(0) = DW/(6kq)",
             implementing=(loc(VM, "recovery_metrics", "stage1_rel_err_signed") + "; abs: "
                           + loc(VM, "evaluate", 'sc["stage1_rel_err_abs"]')),
             column_name="stage1_rel_err_signed, stage1_rel_err_abs"),
        dict(symbol="stage-2 peak error at d = 0",
             definition="(e_hat_2(0) - e2*(0))/e2*(0), signed; absolute value also reported",
             state_region="node d = 0 of the recovery grid (exact "
                          "node)", deviation_type="none (recovery)",
             aggregation="single value", normalization="fraction of e2*(0) = DW f_xi(0)/(2k)",
             implementing=loc(VM, "recovery_metrics", '"stage2_peak_rel_err_signed"'),
             column_name="stage2_peak_rel_err_signed, stage2_peak_rel_err_abs"),
        dict(symbol="location-free peak error (argmax "
                    "d)", definition=("(max_d e_hat_2(d) - e2*(0))/e2*(0) over the "
                                                                       "recovery grid, with the gap d of the maximum"),
             state_region=f"recovery grid on D_2 ({rec[50]['rec_n']} / {rec[60]['rec_n']} nodes)",
             deviation_type="none (recovery)", aggregation="maximum over nodes",
             normalization="fraction of e2*(0), signed",
             implementing=f"{loc(P4C, 'location_free')}; locked "
                          f"pipeline: {loc(LK, 'stage2_extra')}",
             column_name="stage2_peak_locfree_rel_err, stage2_peak_locfree_argmax_d"),
        dict(symbol="RMSE over the positive region",
             definition="sqrt(mean over recovery nodes with |d| < 2q of (e_hat_2(d) - e2*(d))^2)",
             state_region=f"recovery nodes with |d| < 2q ({nn})", deviation_type="none (recovery)",
             aggregation="root mean square", normalization="raw (effort units) and /e2*(0) (G-A "
                                                           "criterion)",
             implementing=loc(VM, "recovery_metrics", "pos = np.abs(D) < 2.0 * q",
                              "rmse = float(np.sqrt("),
             column_name="stage2_rmse_pos, stage2_rmse_pos_over_g2_0"),
        dict(symbol="tail mean", definition="mean of e_hat_2(d) over recovery nodes with |d| >= 2q "
                                            "(where e2* = 0)",
             state_region=f"recovery nodes with |d| >= 2q ({nt})", deviation_type="none (recovery)",
             aggregation="mean", normalization="raw (effort units) and /e2*(0) (G-A criterion)",
             implementing=loc(VM, "recovery_metrics", '"stage2_tail_mean": float(e2[tail].mean())'),
             column_name="stage2_tail_mean, stage2_tail_mean_over_g2_0"),
        dict(symbol="tail max", definition="max of e_hat_2(d) over recovery nodes with |d| >= 2q, "
                                           "with its gap d",
             state_region=f"recovery nodes with |d| >= 2q ({nt})", deviation_type="none (recovery)",
             aggregation="maximum", normalization="raw (effort units) and /e2*(0)",
             implementing=loc(VM, "recovery_metrics", "jt = int(np.argmax(np.where(tail, e2, "
                                                      "-np.inf)))"),
             column_name="stage2_tail_max, stage2_tail_max_over_g2_0, stage2_tail_argmax_d"),
        dict(symbol="symmetry error", definition="max_d |e_hat_2(d) - e_hat_2(-d)| over the "
                                                 "recovery grid, with |d| at the maximum",
             state_region="recovery grid on D_2", deviation_type="none (recovery)", aggregation="maximum",
             normalization="raw (effort units) and /e2*(0)",
             implementing=loc(VM, "recovery_metrics", "sym = np.abs(e2 - e2[::-1])"),
             column_name="stage2_sym_err_max, stage2_sym_err_max_over_g2_0, "
                         "stage2_sym_err_argmax_d"),
        dict(symbol="sigma_t(d)", definition=("SD of the effort under the Beta action at (t, d): "
                                              "(e_max - e_min) "
                                              "sqrt(alpha beta / ((alpha + beta)^2 (alpha + beta + "
                                              "1))); reported as "
                                              "sigma_1(0), sigma_2(0) and the mean of sigma_2(d) "
                                              "over |d| < 2q"),
             state_region="(t=1, d=0), (t=2, d=0) and the verifier-grid nodes of D_2 with |d| < 2q",
             deviation_type="none (spread of the stochastic "
                            "policy)", aggregation="point values; mean over nodes",
             normalization="effort units [0, 100], not normalized",
             implementing=f"{loc(DV, 'beta_std_norm')}; {loc(VM, 'evaluate', 'sigma_effort_at_0_t', 'sigma2_effort_mean_pos')}",
             column_name="sigma_effort_at_0_t1, sigma_effort_at_0_t2, sigma2_effort_mean_pos"),
        dict(symbol="on-path rule", definition=("stage-2 on-path set {d in D_2 : |d - drift| < 2q} "
                                                "(open interval), "
                                                "drift = e_hat_1(0) - e_hat_1(-0) (exactly 0 in "
                                                f"{n_drift0}/{n_drift} opening-check cases); "
                                                f"weights = exact cell masses "
                                                "F_xi(b_i+1 - drift) - F_xi(b_i - drift), cells "
                                                "bounded by node midpoints, "
                                                "renormalized over the on-path cells; nodes at "
                                                "exactly 2q are off-path. "
                                                "Decision of Phase 2 (phase2_opening_checks.md "
                                                "section 1c, options b and c); "
                                                "supersedes the node-based GL PMF rule of "
                                                "phase1_verifier.md section 2.4, "
                                                "whose statement on the nodes at +-2q is corrected "
                                                "in "
                                                "phase2_opening_checks.md section 1c"),
             state_region="nodes of D_2", deviation_type="applies to Delta_2 (one-step) and to the "
                                                         "stage-2 drift",
             aggregation="on-path max (argmax d), unweighted mean and cell-mass-weighted mean; "
                         "off-path max and unweighted mean",
             normalization="Delta_2 parts /DW; drift parts in effort units",
             implementing=(f"{loc(VM, 'onpath_mask')}; {loc(VM, 'cell_masses')}; {loc(VM, 'onoff_split')}; "
                           f"{loc(VM, 'evaluate', 'on_T = onpath_mask(', 'split = onoff_split(')}"),
             column_name="DeltaT_over_dw_on_*, DeltaT_over_dw_off_*, stage2_drift_*_on_*/off_*"),
    ]
    df = pd.DataFrame(rows)
    docs = {"symbol": "Metric symbol", "definition": "Definition as implemented",
            "state_region": "States over which the metric is taken",
            "deviation_type": "Kind of deviation measured (one-step against the e_hat "
                              "continuation, dynamic best response, or none)",
            "aggregation": "How node values are combined", "normalization": "Normalization and units",
            "implementing": "Implementing file:function with its line span and the defining lines "
                            "(read from the file at build time)",
            "column_name": "Column name(s) of the metric in the per-run records and pack tables"}
    srcs = [C.src(p) for p in (DV, VM, P4C, LK, R_P1, R_OPEN,
                               "reports/v2/pilot1_reward_estimator.md")] + \
        [C.src("results/v2_pilots/phase2_opening/onpath_sets.csv", "stage1_drift")]
    pack.table("T07", df, status="generated", sources=srcs, script=f"{MOD}:build_t07", docs=docs,
               tier="n/a",
               notes=("source: code (definitions transcribed from the implementing functions; line "
                      "numbers located by "
                      "ast and text search at build time) and report text (on-path decision); "
                      "drift count from "
                      "onpath_sets.csv; node counts from symmetric_grid."),
               caption="Metric definitions with the implementing code. Deviation metrics are "
                       "divided by DW = 4; recovery errors by e1*(0) or e2*(0).")
    xc = XCheck(pack, "T07", R_OPEN, "stage-1 drift exactly 0 (section 1c)")
    m = xc.find(r"exactly 0 in all (\d+) cases")
    if m:
        xc.cmp("cases with stage-1 drift exactly 0", n_drift0, m.group(1), "section 1c")
        xc.cmp("cases checked", n_drift, m.group(1), "section 1c")
    xc.done()


# ----------------------------------------------------------------------------------------------
# T08 verifier tiers
# ----------------------------------------------------------------------------------------------

def build_t08(pack: C.Pack) -> None:
    """T08: state and effort grids, action search, quadrature, interpolation, role and cadence of each tier, per q."""
    DV = "utils/dp_br_verifier.py"
    pr = _proto()
    rows: List[Dict[str, Any]] = []
    search = ("effort grid E + valid concave-parabola vertices (strictly concave grid triple, Q "
              "recomputed at the vertex) "
              "+ the candidate's own mean action; the Delta search adds a^BR; ties to the smaller "
              "effort "
              f"[{loc(DV, 'verify', 'eff_c = np.concatenate', 'eff_d = np.concatenate')}; {loc(DV, '_select')}; "
              f"{loc(DV, '_vertex_candidates')}]")
    interp = ("stage 1: np.interp of the stage-2 value array on the D_2 grid at the GL landing "
              "points; a landing point "
              "outside D_2 (beyond 1e-9 max(1, B)) raises DomainError, no extrapolation; stage 2 "
              "(terminal): closed form "
              f"w_L + DW F_xi(y), no interpolation "
              f"[{loc(DV, 'verify', 'acc += w * np.interp(land, g, v_next)')}]")
    for q in QS:
        sp = _spec(q)
        P = pr["records"][str(q)]["protocol"]
        role = {
            "development": ("training-time verifier calls of every v2 run (phase-local cadence: "
                            "warm-up call at local "
                            f"{P['warmup']}; stability checks "
                            f"every {P['stability_every']} local updates, a call after "
                            f"{P['stability_consecutive']} consecutive stable checks (drift "
                            f"<= {P['drift_thr']:g} of the "
                            f"effort range and KL <= {P['kl_thr']:g}); "
                            f"timeout {P['verifier_timeout']['A']} (A) / "
                            f"{P['verifier_timeout']['B']} (B) local updates; a phase-end call); "
                            f"dev side of G-N at the "
                            "end of A (eta_2) and B (Gmax_full); every gate and reported metric is "
                            "also reported on it"),
            "dev_2x": "Phase 1 only: calibration (calibration.csv) and the invariant probe "
                      "(invariants_probe.csv); isolates the state and effort steps (GL unchanged)",
            "final": ("gates G-A (end of Phase A) and G-F (end of Phase B) and the final side of "
                      "G-N; stage-1 residual-band "
                      "sweep (step 0.01); end-of-run evaluation of every pilot run "
                      "(final_v2.json); Phase 1 calibration"),
        }
        for c in TIERS:
            g2 = stage_grid(2, sp.B, c.state_step)
            E = effort_grid(sp.e_min, sp.e_max, c.effort_step)
            nodes, w = gl_shock_rule(float(sp.q), c.gl_half)
            rows.append({
                "q": q, "tier": c.name, "D1_nodes": int(stage_grid(1, sp.B, c.state_step).size),
                "D2_range": f"[{g2[0]:g}, {g2[-1]:g}]", "state_step": c.state_step, "D2_nodes": int(g2.size),
                "effort_range": f"[{E[0]:g}, {E[-1]:g}]", "effort_step": c.effort_step, "effort_nodes": int(E.size),
                "gl_nodes_per_half": c.gl_half, "gl_nodes_total": int(nodes.size),
                "gl_outermost_node": float(nodes.max()), "gl_weight_sum": float(w.sum()),
                "quadrature": ("Gauss-Legendre on [-2q, 0] and [0, 2q], weights times the "
                               "triangular density f_xi "
                               f"(nodes strictly inside (-2q, 2q)) [{loc(DV, 'gl_shock_rule')}]"),
                "action_search": search, "interpolation": interp,
                "tolerances": f"grid_tol {c.grid_tol:g}, mass_tol {c.mass_tol:g}, PDL residual "
                              f"<= {c.pdl_tol_over_dw:g} DW",
                "role_and_cadence": role[c.name]})
        rg = symmetric_grid(sp.domain_half(2), float(P["recovery_step"]))
        rows.append({"q": q, "tier": "recovery grid (tier-independent)", "D1_nodes": np.nan,
                     "D2_range": f"[{rg[0]:g}, {rg[-1]:g}]", "state_step": float(P["recovery_step"]),
                     "D2_nodes": int(rg.size), "effort_range": "", "effort_step": np.nan, "effort_nodes": np.nan,
                     "gl_nodes_per_half": np.nan, "gl_nodes_total": np.nan, "gl_outermost_node": np.nan,
                     "gl_weight_sum": np.nan, "quadrature": "none (direct policy queries)", "action_search": "none",
                     "interpolation": "none (the policy is queried at every "
                                      "node)", "tolerances": "0 must be an exact node",
                     "role_and_cadence": ("closed-form recovery metrics at every verifier call and "
                                          "at every evaluation "
                                          f"[{loc('utils/v2_metrics.py', 'recovery_metrics')}]")})
    df = pd.DataFrame(rows)
    for c in (DEV_CONFIG, FINAL_CONFIG):   # the record's tiers are the code constants (checked by the runner too)
        rec_t = pr["records"]["50"]["verifier"][c.name]
        assert (rec_t["state_step"], rec_t["effort_step"], rec_t["gl_half"]) == (c.state_step,
                                                                                 c.effort_step, c.gl_half)
    docs = {"q": None, "tier": "Verifier tier (development, dev_2x, final) or the recovery grid",
            "D1_nodes": dict(definition="Nodes of the stage-1 grid D_1 = {0}", units="nodes"),
            "D2_range": dict(definition="Range of the stage-2 grid D_2 = [-B, "
                                        "B]", units="effort units (gap d)"),
            "state_step": dict(definition="State-grid spacing on D_2",
                               units="effort units (gap d)"),
            "D2_nodes": dict(definition="Nodes of the stage-2 grid (stage_grid / "
                                        "symmetric_grid)", units="nodes"),
            "effort_range": dict(definition="Range of the action-search effort "
                                            "grid", units="effort units"),
            "effort_step": dict(definition="Effort-grid spacing", units="effort units"),
            "effort_nodes": dict(definition="Nodes of the effort grid (effort_grid)",
                                 units="nodes"),
            "gl_nodes_per_half": dict(definition="Gauss-Legendre nodes per half of the "
                                                 "shock-difference support",
                                      units="nodes"),
            "gl_nodes_total": dict(definition="Total quadrature nodes (two halves)", units="nodes"),
            "gl_outermost_node": dict(definition="Largest quadrature node (strictly inside "
                                                 "2q)", units="effort units"),
            "gl_weight_sum": dict(definition="Sum of the quadrature weights (validity requires "
                                             "|sum - 1| <= 1e-12)",
                                  units="probability"),
            "quadrature": "Expectation over the shock difference "
                          "xi", "action_search": "Candidate actions of the BR and Delta searches",
            "interpolation": "Continuation values between grid nodes",
            "tolerances": "Numerical tolerances of the tier (VerifierConfig)",
            "role_and_cadence": "Where the tier is used (development vs final) and when it is "
                                "called"}
    docs = {k: v for k, v in docs.items() if v is not None}
    srcs = [C.src(DV), C.src("utils/v2_metrics.py"), C.src(PROTO), C.src("tools/v2/common.py"),
            C.src(R_P1),
            C.src("run/run_v2_stagewise.py"), C.src("run/run_v2_T2_locked.py")]
    pack.table("T08", df, status="generated", sources=srcs, script=f"{MOD}:build_t08", docs=docs,
               tier="final and development",
               notes=("Counts from stage_grid / effort_grid / gl_shock_rule / symmetric_grid; tier "
                      "parameters from "
                      "utils/dp_br_verifier.py DEV_CONFIG / FINAL_CONFIG (equal to the protocol "
                      "records, asserted) and "
                      "tools/v2/common.py DEV_2X; cadence from the protocol record; role text "
                      "compiled from the code "
                      "and reports (source: report text)."),
               caption="Verifier tiers per q; dev_2x was used only in Phase 1; the recovery grid "
                       "is tier-independent.")
    xc = XCheck(pack, "T08", R_P0, "tier grids (section 2 table)")
    for tier, mm in zip(("development", "final"), xc.findall(
            r"state step (" + _NUM + r"), effort step (" + _NUM + r"), GL (\d+)/half \(D_2 grid (\d+) / (\d+) points; "
            r"effort grid (\d+)\)")):
        sub = df[df.tier == tier]
        xc.cmp(f"{tier} state step", float(sub.state_step.iloc[0]), mm[0], "section 2")
        xc.cmp(f"{tier} effort step", float(sub.effort_step.iloc[0]), mm[1], "section 2")
        xc.cmp(f"{tier} GL per half", float(sub.gl_nodes_per_half.iloc[0]), mm[2], "section 2")
        xc.cmp(f"{tier} D_2 nodes q=50", float(sub[sub.q == 50].D2_nodes.iloc[0]), mm[3],
               "section 2")
        xc.cmp(f"{tier} D_2 nodes q=60", float(sub[sub.q == 60].D2_nodes.iloc[0]), mm[4],
               "section 2")
        xc.cmp(f"{tier} effort nodes", float(sub.effort_nodes.iloc[0]), mm[5], "section 2")
    xc.done()
    xc = XCheck(pack, "T08", R_OPEN, "outermost GL node (section 1c)")
    m = xc.find(r"the outermost is at about .(" + _NUM + r") for q = 50")
    if m:
        v = float(df[(df.q == 50) & (df.tier == "development")].gl_outermost_node.iloc[0])
        xc.cmp("outermost GL node, 16 per half, q=50", v, m.group(1), "section 1c")
    xc.done()


# ----------------------------------------------------------------------------------------------
# T09 invariant checks
# ----------------------------------------------------------------------------------------------

def _inv_from_npz(rel: str, dw: float) -> Dict[str, Any]:
    """Invariant residuals and aggregates of one saved evaluation (``evaluate`` arrays), same float ops as
    ``utils.v2_metrics._invariants`` / ``utils.dp_br_verifier.verify``."""
    z = np.load(C.abspath(rel))
    G = {t: z[f"v_t{t}_G"] for t in (2, 1)}          # verify fills stages from t = T down to 1
    D = {t: z[f"v_t{t}_delta"] for t in (2, 1)}
    g2 = z["v_t2_d_grid"]
    out: Dict[str, Any] = {}
    diff = np.abs(G[2] - D[2])
    j = int(np.argmax(diff))
    out["inv_GT_eq_DeltaT_absdiff_over_dw"], out["inv_GT_eq_DeltaT_at_d"] = float(diff[j] / dw), float(g2[j])
    best = (-np.inf, None, None)
    for t in (2, 1):
        ex = D[t] - G[t]
        i = int(np.argmax(ex))
        if ex[i] > best[0]:
            best = (float(ex[i]), t, float(z[f"v_t{t}_d_grid"][i]))
    out["inv_Delta_le_G_pointwise_excess_over_dw"] = best[0] / dw
    out["inv_Delta_le_G_pointwise_at_t"], out["inv_Delta_le_G_pointwise_at_d"] = best[1], best[2]
    full_dm = {t: float(D[t].max()) for t in (2, 1)}
    reach_dm = {t: float(D[t][z[f"v_t{t}_reach"].astype(bool)].max()) for t in (2, 1)}
    sup_dm = {t: float(D[t][z[f"v_t{t}_pmf"] > 0.0].max()) for t in (2, 1)}
    gmax = max(float(G[t].max()) for t in (2, 1))
    dfull, dreach, dsup = sum(full_dm.values()), sum(reach_dm.values()), sum(sup_dm.values())
    dmax_all = max(full_dm.values())
    exp_root = float(z["v_t1_v_br"][0] - z["v_t1_v_mean"][0])
    out.update({
        "inv_Deltamax_le_Gmax_excess_over_dw": (dmax_all - gmax) / dw,
        "inv_Gmax_le_dFull_excess_over_dw": (gmax - dfull) / dw,
        "inv_EXP_eq_G1_absdiff_over_dw": abs(exp_root - float(G[1][0])) / dw,
        "inv_EXP_le_dReach_excess_over_dw": (exp_root - dreach) / dw,
        "inv_dReach_le_dFull_excess_over_dw": (dreach - dfull) / dw,
        "inv_EXP_le_dReachPMF_excess_over_dw": (exp_root - dsup) / dw,
        "Gmax_full_over_dw": gmax / dw, "EXP_root_over_dw": exp_root / dw, "dReach_over_dw": dreach / dw,
        "Deltamax_all_over_dw": dmax_all / dw, "dFull_over_dw": dfull / dw, "eta_T_over_dw": full_dm[2] / dw})
    return out


def _locked_gate_evals() -> Tuple[pd.DataFrame, C.SourceSet, float, int, C.SourceSet]:
    """Invariants recomputed from gateA_*.npz / final_*.npz of every locked run, with the gates.json check."""
    if "locked" in _CACHE:
        return _CACHE["locked"]
    dw = float(_proto()["records"]["50"]["dw"])
    ss = C.srcs([f"results/v2_T2_locked/{s}/q*/seed*/{p}_{t}.npz" for s in ("rehearsal",
                                                                            "rehearsal_v1_1", "confirmation")
                 for p in ("gateA", "final") for t in ("final", "development")],
                label="results/v2_T2_locked/{rehearsal,rehearsal_v1_1,confirmation}/q*/seed*/{gateA,final}_*.npz",
                expect=320)
    rows, maxdiff, n_cmp = [], 0.0, 0
    for s in ss.files:
        r = _inv_from_npz(s.path, dw)
        rd = s.path.rsplit("/", 1)[0]
        stem = s.path.rsplit("/", 1)[1][:-4]
        point, tier = ("end_of_A" if stem.startswith("gateA") else "end_of_B"), stem.split("_", 1)[1]
        gates = _json(f"{rd}/gates.json")
        rep = gates["reported"][point][tier]
        for k in ("Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw",
                  "Deltamax_all_over_dw", "dFull_over_dw",
                  "eta_T_over_dw"):
            if k in rep:
                maxdiff = max(maxdiff, abs(float(rep[k]) - r[k]))
                n_cmp += 1
        r.update(file=s.path, verifier_tier=tier, valid=bool(rep.get("valid")),
                 source=f"{rd}/gates.json")
        rows.append(r)
    gsrc = C.srcs(sorted({r["source"] for r in rows}), label="gates.json of the locked runs (valid "
                                                             "flags, reported values)")
    _CACHE["locked"] = (pd.DataFrame(rows), ss, maxdiff, n_cmp, gsrc)
    return _CACHE["locked"]


def _final_v2_evals() -> Tuple[pd.DataFrame, C.SourceSet]:
    """End-of-run evaluations (development and final tier) of every pilot-style run (final_v2.json)."""
    ss = C.srcs(["results/v2_pilots/**/final_v2.json", "results/v2_T2_locked/**/final_v2.json"],
                label="results/{v2_pilots,v2_T2_locked}/**/final_v2.json")
    rows = []
    for s in ss.files:
        j = _json(s.path)
        for tier in ("development", "final"):
            sc = j.get(tier, {})
            if "error" in sc or not sc:
                rows.append({"file": s.path, "verifier_tier": tier, "valid": False,
                             "error": sc.get("error", "missing")})
                continue
            rows.append({"file": s.path, "verifier_tier": tier, "valid": bool(sc.get("valid")),
                         **{k: v for k, v in sc.items() if k.startswith("inv_")}})
    return pd.DataFrame(rows), ss


def _test_counts() -> Dict[str, Any]:
    """Evaluations of tests/test_v2_verifier.py::test_invariants (QS x TIERS x _candidates) and its tolerance."""
    tf = "tests/test_v2_verifier.py"
    tree = ast.parse("\n".join(_lines(tf)))
    nq = nc = None
    tol = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id == "QS":
                nq = len(n.value.elts)
            if n.targets[0].id == "ROUND":
                tol = ast.literal_eval(n.value)
        if isinstance(n, ast.FunctionDef) and n.name == "_candidates":
            ret = [s for s in ast.walk(n) if isinstance(s, ast.Return)][0]
            nc = len(ret.value.keys)
    tiers_node = [n for n in ast.parse("\n".join(_lines("tools/v2/common.py"))).body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0],
                                                              ast.Name) and n.targets[0].id == "TIERS"][0]
    nt = len(tiers_node.value.elts)
    return {"n_q": nq, "n_tiers": nt, "n_cand": nc, "n": nq * nt * nc, "tol": tol}


def build_t09(pack: C.Pack) -> None:
    """T09: maximum invariant residual (DW units) over every evaluation set that logged it, per relation."""
    probe = _csv(f"{P1}/invariants_probe.csv")
    ckpt = _csv(f"{P1}/invariants_checkpoints.csv")
    cal = _csv(CAL_P1)
    pc, pc_ss = _percall_logs()
    fv, fv_ss = _final_v2_evals()
    lk, lk_ss, lk_maxdiff, lk_ncmp, lk_gsrc = _locked_gate_evals()
    tc = _test_counts()

    def lab_probe(r: pd.Series) -> str:
        return f"q={r.q:g}, candidate {r.candidate}, tier {r.verifier_tier}"

    def lab_ckpt(r: pd.Series) -> str:
        return f"{r.cohort}/{r.run} (q={r.q:g}), tier {r.verifier_tier}"

    def lab_cal(r: pd.Series) -> str:
        return f"q={r.q:g}, {r.policy}, tier {r.verifier_tier}"

    def lab_file(r: pd.Series) -> str:
        return f"{r['file']}" + (f" u{int(r['update'])}" if "update" in r.index and pd.notna(r["update"]) else "") + \
            (f", tier {r['verifier_tier']}" if "verifier_tier" in r.index else "")

    sets = [
        ("Phase 1 invariant probe (900 synthetic candidates; "
         "results/v2_pilots/phase1/invariants_probe.csv)", probe, lab_probe),
        ("Phase 1 legacy checkpoints (80 x 3 tiers; "
         "results/v2_pilots/phase1/invariants_checkpoints.csv)", ckpt, lab_ckpt),
        ("Phase 1 calibration (analytic and zero policy; "
         "results/v2_pilots/phase1/calibration/calibration.csv)", cal, lab_cal),
        ("training-time verifier calls of every v2 run (all v2_checkpoints*.csv)", pc, lab_file),
        ("end-of-run evaluations of the pilot-style runs (all final_v2.json)", fv, lab_file),
        ("locked gate evaluations, recomputed from gateA_*.npz and final_*.npz", lk, lab_file),
    ]
    rows: List[Dict[str, Any]] = []
    allmax: Dict[str, Tuple[float, str, int, int, int]] = {}
    for rel_label, col, kind in INV_COLS:
        rows.append({"relation": rel_label, "residual_column": col, "kind": kind,
                     "evaluation_set": "tests/test_v2_verifier.py::test_invariants",
                     "tiers": "development, dev_2x, final", "n_evaluations": tc["n"], "n_valid": tc["n"],
                     "n_residual_positive": "UNKNOWN", "max_residual_over_dw": "UNKNOWN",
                     "location_of_max": "", "note": (f"{tc['n_q']} q x {tc['n_tiers']} tiers "
                                                     f"x {tc['n_cand']} candidates; "
                                                     f"asserts residual <= {tc['tol']:g} DW and "
                                                     f"valid; values not logged; "
                                                     "passed in every reported suite run")})
        n_all = nv_all = npos_all = 0
        best = (-np.inf, "")
        for name, df, lab in sets:
            if col not in df.columns:
                continue
            v = df[col].astype(float)
            j = int(np.nanargmax(v.to_numpy()))
            r = df.iloc[j]
            where = lab(r)
            if col == "inv_Delta_le_G_pointwise_excess_over_dw" and "inv_Delta_le_G_pointwise_at_t" in df.columns:
                where += f", at t={r['inv_Delta_le_G_pointwise_at_t']:g}, d={r['inv_Delta_le_G_pointwise_at_d']:g}"
            if col == "inv_GT_eq_DeltaT_absdiff_over_dw" and "inv_GT_eq_DeltaT_at_d" in df.columns:
                where += f", at d={r['inv_GT_eq_DeltaT_at_d']:g}"
            n, nv, npos = int(v.notna().sum()), int(df["valid"].astype(bool).sum()), int((v > 0).sum())
            tiers = ", ".join(sorted(df["verifier_tier"].astype(str).unique()))
            rows.append({"relation": rel_label, "residual_column": col, "kind": kind,
                         "evaluation_set": name,
                         "tiers": tiers, "n_evaluations": n, "n_valid": nv, "n_residual_positive": npos,
                         "max_residual_over_dw": float(v.iloc[j]), "location_of_max": where, "note": ""})
            n_all, nv_all, npos_all = n_all + n, nv_all + nv, npos_all + npos
            if float(v.iloc[j]) > best[0]:
                best = (float(v.iloc[j]), f"{name.split(' (')[0]}: {where}")
        rows.append({"relation": rel_label, "residual_column": col, "kind": kind,
                     "evaluation_set": "ALL logged evaluations (the six sets "
                                       "above)", "tiers": "development, dev_2x, final",
                     "n_evaluations": n_all, "n_valid": nv_all, "n_residual_positive": npos_all,
                     "max_residual_over_dw": best[0], "location_of_max": best[1],
                     "note": ("excess (lhs - rhs)/DW, positive = violated" if kind == "inequality"
                              else "max |lhs - rhs|/DW")})
        allmax[col] = (best[0], best[1], n_all, nv_all, npos_all)
    df = pd.DataFrame(rows)
    pack.unknown_value("T09", "test_invariants: maximum residual and number of positive residuals",
                       f"tests/test_v2_verifier.py::test_invariants asserts every residual "
                       f"<= {tc['tol']:g} DW on "
                       f"{tc['n']} evaluations but writes no values; no file records them (the "
                       f"logged sets of T09 cover "
                       "the same relations)")
    docs = {"relation": "Invariant relation (SPEC T09) or the companion / pointwise form logged by "
                        "utils.v2_metrics._invariants",
            "residual_column": "Residual column written by utils/v2_metrics.py:_invariants on "
                               "every evaluate() call",
            "kind": "equality (max |lhs - rhs|) or inequality (excess lhs - rhs; positive = "
                    "violated)",
            "evaluation_set": "Set of verifier evaluations over which the residual was measured",
            "tiers": "Verifier tiers present in the set",
            "n_evaluations": dict(definition="Evaluations in the set with a logged "
                                             "residual", units="evaluations"),
            "n_valid": dict(definition="Evaluations of the set with valid = "
                                       "True", units="evaluations"),
            "n_residual_positive": dict(definition="Evaluations with residual > 0 (UNKNOWN: not "
                                                   "logged)", units="evaluations"),
            "max_residual_over_dw": dict(definition="Largest residual over the set (raw, never "
                                                    "clamped; UNKNOWN: not logged)",
                                         units="Delta W (dimensionless)", normalization="divided by Delta W = 4",
                                         tier="tiers of the set"),
            "location_of_max": "Evaluation (and stage t / node d for the pointwise and G_2 = "
                               "Delta_2 rows) of the largest residual",
            "note": "Remark"}
    srcs = [C.src(f"{P1}/invariants_probe.csv"), C.src(f"{P1}/invariants_checkpoints.csv"),
            C.src(CAL_P1), pc_ss, fv_ss,
            lk_ss, lk_gsrc, C.src("tests/test_v2_verifier.py"), C.src("tools/v2/common.py"), C.src("utils/v2_metrics.py"),
            C.src(R_P1)]
    by_study = pc.groupby("study").size()
    pack.table("T09", df, status="generated", sources=srcs, script=f"{MOD}:build_t09", docs=docs,
               tier="final and development",
               notes=("Residual columns inv_* as logged by utils.v2_metrics.evaluate (Phase 1 "
                      "CSVs, per-call "
                      "v2_checkpoints*.csv, final_v2.json); for the locked runs the same residuals "
                      "are recomputed from the "
                      "saved gate arrays (v_t*_G, v_t*_delta, v_t*_reach, v_t*_pmf, "
                      "v_t1_v_br/v_mean) with the float "
                      f"operations of verify/_invariants, and the recomputed aggregates equal "
                      f"gates.json ({lk_ncmp} values, "
                      f"max abs diff {lk_maxdiff:g}). Per-call rows by study: "
                      + ", ".join(f"{k} {v}" for k, v in by_study.items()) + "."),
               caption=("Maximum invariant residual in DW units per relation and evaluation set; "
                        "the ALL row of each "
                        "relation is the maximum over every logged evaluation. Equalities: max "
                        "|lhs - rhs|/DW; "
                        "inequalities: excess (lhs - rhs)/DW, positive = violated."))
    # ---- cross-checks with phase1_verifier.md section 4.1
    xc = XCheck(pack, "T09", R_P1, "invariant residuals (section 4.1 tables and prose)")
    tokens = [("G\u2082 = \u0394\u2082", 0), ("pointwise", 1), ("\u0394max_all \u2264", 2),
              ("\u011cmax_full \u2264 dFull", 3),
              ("EXP_root = G\u2081(0)", 4), ("EXP_root \u2264 dReach", 5), ("dReach \u2264 dFull",
                                                                            6), ("companion", 7)]
    S1 = {name: (d, lab) for name, d, lab in sets[:3]}
    names = [s[0] for s in sets[:3]]
    for t in xc.tables_with(["Relation", "Max over 900"]):
        for r in t["rows"]:
            idx = next((i for tok, i in tokens if (r[0].startswith(tok) if i == 5 else tok in r[0])), None)
            if idx is None:
                xc.unmatched(r[0])
                continue
            col = INV_COLS[idx][1]
            for ci, nm in ((1, names[0]), (3, names[1]), (4, names[2])):
                d = S1[nm][0]
                nums = _nums(r[ci])
                if not nums:
                    continue
                comment = ""
                if not C.consistent(float(d[col].max()), nums[0]):
                    for tier_, sub in d.groupby("verifier_tier"):
                        if C.consistent(float(sub[col].max()), nums[0]):
                            comment = (f"the report value equals the maximum over "
                                       f"the {tier_}-tier rows only ({len(sub)} "
                                       f"rows); the maximum over all {len(d)} rows "
                                       f"is {float(d[col].max()):g}, attained in "
                                       f"{int((d[col] == d[col].max()).sum())} rows "
                                       f"({', '.join(sorted(d.loc[d[col] == d[col].max(), 'verifier_tier'].unique()))} tier)")
                            break
                xc.cmp(f"max {col} over {nm.split(' (')[0]}", float(d[col].max()), nums[0],
                       f"line {t['line']}",
                       comment=comment)
                mm = re.search(r"\((\d+) of (\d+) rows > 0\)", r[ci])
                if mm:
                    xc.cmp(f"rows > 0 of {col}, {nm.split(' (')[0]}", int((d[col] > 0).sum()),
                           mm.group(1), f"line {t['line']}")
                    xc.cmp(f"rows of {nm.split(' (')[0]}", len(d), mm.group(2), f"line {t['line']}")
                if "no row > 0" in r[ci]:
                    xc.cmp(f"rows > 0 of {col}, {nm.split(' (')[0]}", int((d[col] > 0).sum()), "0",
                           f"line {t['line']}")
            if idx == 1:   # location of the pointwise maximum (compared only if the maximum is unique)
                d = probe
                mx = d[col].max()
                hit = d[d[col] == mx]
                mm = re.search(r"t=(\d+), d=(" + _NUM + r") \(q=(\d+), candidate (\d+), (\w+) "
                                                        r"tier", r[2])
                if mm and len(hit) == 1:
                    h = hit.iloc[0]
                    for nm_, val, s_ in (("t", h["inv_Delta_le_G_pointwise_at_t"], mm.group(1)),
                                         ("d", h["inv_Delta_le_G_pointwise_at_d"], mm.group(2)),
                                         ("q", h["q"], mm.group(3)), ("candidate", h["candidate"],
                                                                      mm.group(4))):
                        xc.cmp(f"location of the pointwise maximum: {nm_}", float(val), s_,
                               f"line {t['line']}")
                    xc.same("location of the pointwise maximum: "
                            "tier", h["verifier_tier"], mm.group(5), f"line {t['line']}")
                elif mm:
                    xc.unmatched(f"pointwise maximum location: {len(hit)} tied rows")
    for t in xc.tables_with(["Tier", "Max excess"]):
        for r in t["rows"]:
            tier = r[0].split(" ")[0]
            sub = probe[probe.verifier_tier == tier]["inv_Delta_le_G_pointwise_excess_over_dw"]
            if sub.empty:
                xc.unmatched(r[0])
                continue
            xc.cmp(f"probe pointwise max, tier {tier}", float(sub.max()), _nums(r[1])[0],
                   f"line {t['line']}")
            xc.cmp(f"probe pointwise rows > 0, tier {tier}", int((sub > 0).sum()), _nums(r[2])[0],
                   f"line {t['line']}")
    m = xc.find(r"raised 0 times in ([\d,]+) verifier evaluations: all of them have valid=True\. "
                r"The count is (\d+) in "
                r"P1/invariants_probe\.csv, (\d+) in P1/invariants_checkpoints\.csv and (\d+) in "
                r"P1/calibration/calibration\.csv")
    if m:
        tot = len(probe) + len(ckpt) + len(cal)
        xc.cmp("Phase 1 evaluations", tot, m.group(1).replace(",", ""), "section 2.2")
        xc.cmp("Phase 1 valid evaluations",
               int(probe.valid.sum() + ckpt.valid.sum() + cal.valid.sum()),
               m.group(1).replace(",", ""), "section 2.2")
        for i, d in ((2, probe), (3, ckpt), (4, cal)):
            xc.cmp("rows of a Phase 1 evaluation set", len(d), m.group(i), "section 2.2")
    m = xc.find(r"Across ([\d,]+) evaluations the largest excess is (" + _NUM + r")")
    if m:
        p1max = max(float(d["inv_Delta_le_G_pointwise_excess_over_dw"].max()) for d in (probe, ckpt, cal))
        xc.cmp("largest pointwise excess over the Phase 1 "
               "evaluations", p1max, m.group(2), "section 6")
    xc.done()


# ----------------------------------------------------------------------------------------------
# T10 dReach reach-mask check
# ----------------------------------------------------------------------------------------------

def build_t10(pack: C.Pack) -> None:
    """T10: the 168-evaluation dReach mask check (existing CSV used as is) with a summary block."""
    rel = "results/v2_pilots/dreach_mask_check/dreach_mask_check.csv"
    d = _csv(rel)
    ck = ~d["candidate"].isin(["analytic", "zero"])
    summ = []
    for (q, tier), g in d.groupby(["q", "tier"], sort=True):
        summ.append({"q": int(q), "tier": tier, "evaluations": len(g),
                     "valid": int(g["valid"].sum()),
                     "total_holes": int(g["n_holes"].sum()), "with_0_extras": int((g["n_extra"] == 0).sum()),
                     "with_1_extra": int((g["n_extra"] == 1).sum()), "with_2_extras": int((g["n_extra"] == 2).sum()),
                     "checkpoint_evaluations": int(ck[g.index].sum()),
                     "checkpoints_with_2_extras": int(((g["n_extra"] == 2) & ck[g.index]).sum()),
                     "dreach_diff_nonzero": int((g["dreach_official_minus_refmask_over_dw"] != 0).sum()),
                     "dreach_diff_max_over_dw": float(g["dreach_official_minus_refmask_over_dw"].max())})
    summ = pd.DataFrame(summ)
    diff = d["dreach_official_minus_refmask_over_dw"]
    ex = d[diff != 0]
    dist = sorted({x for s in d["extra_dist_to_support_edge"].dropna() for x in str(s).split()})
    same_ck = d[(d["candidate"].isin(ex["candidate"])) & (d["tier"] == "final")]
    lines = [C.df_to_md(summ, 4), "",
             f"- evaluations: {len(d)} ({int(d['valid'].sum())} valid); candidates: analytic, zero "
             f"and "
             f"{int(ck.sum() // 2)} legacy checkpoints, each on 2 tiers",
             f"- extra nodes in total: {int(d['n_extra'].sum())}; distinct distances to the "
             f"support edge: {', '.join(dist)}",
             f"- (official - reference-mask) dReach/DW: exactly 0 "
             f"in {int((diff == 0).sum())} evaluations; "
             f"nonzero in {len(ex)}; negative in {int((diff < 0).sum())}"]
    for _, r in ex.iterrows():
        lines.append(f"- nonzero case: {r['candidate']} ({r['tier']} tier): "
                     f"difference {r['dreach_official_minus_refmask_over_dw']:.4g}; "
                     f"drift_BR {r['drift_BR']:.6f}; max Delta_2/DW over "
                     f"R_2 {r['max_delta2_R2_over_dw']:.6f} vs over the "
                     f"reference support {r['max_delta2_ref_over_dw']:.6f}; official "
                     f"dReach/DW {r['dreach_official_over_dw']:.6f} "
                     f"vs reference-mask {r['dreach_refmask_over_dw']:.6f}")
    for _, r in same_ck.iterrows():
        lines.append(f"- the same checkpoint on the final tier: "
                     f"drift_BR {r['drift_BR']:.6f}, {int(r['n_extra'])} extra "
                     f"nodes, difference {r['dreach_official_minus_refmask_over_dw']:g}")
    docs = {
        "candidate": "Evaluated candidate: analytic equilibrium, zero policy, or a legacy final "
                     "checkpoint (<cohort>/<run>)",
        "q": None, "tier": None, "valid": None,
        "n_grid": dict(definition="Nodes of the stage-2 grid of the tier", units="nodes"),
        "a_br_1": dict(definition="Root best-response action a^BR_1(0) of the "
                                  "verifier", units="effort units"),
        "e_hat_1": dict(definition="Candidate root action e_hat_1(0) (= "
                                   "e_hat_1(-0))", units="effort units, raw"),
        "drift_BR": dict(definition="Centre of the BR reach interval, a^BR_1(0) - "
                                    "e_hat_1(0)", units="effort units (gap d)"),
        "n_R2": dict(definition="Nodes of the official reach mask R_2 (closed interval, tolerance "
                                "1e-9)", units="nodes"),
        "n_ref": dict(definition="Nodes of the reference support {d : |d - drift_BR| < 2q} "
                                 "(open)", units="nodes"),
        "n_holes": dict(definition="Reference-support nodes missing from R_2", units="nodes"),
        "n_extra": dict(definition="R_2 nodes outside the reference support", units="nodes"),
        "holes_d": dict(definition="Gaps d of the holes (space separated; empty if "
                                   "none)", units="effort units (gap d)"),
        "extra_d": dict(definition="Gaps d of the extra nodes (space "
                                   "separated)", units="effort units (gap d)"),
        "extra_dist_to_support_edge": dict(definition="Distance of each extra node to |d - "
                                                      "drift_BR| = 2q",
                                           units="effort units (gap d)"),
        "dreach_official_over_dw": dict(definition="Official dReach (mask R_2), utils/dp_br_verifier.py:verify",
                                        units="Delta W (dimensionless)", normalization="divided by Delta W"),
        "dreach_refmask_over_dw": dict(definition="Delta_1(0) + max of Delta_2 over the reference "
                                                  "support (comparison only)",
                                       units="Delta W (dimensionless)", normalization="divided by Delta W"),
        "dreach_official_minus_refmask_over_dw": dict(definition="Official minus reference-mask dReach",
                                                      units="Delta W (dimensionless)", normalization="divided by Delta W"),
        "max_delta2_R2_over_dw": dict(definition="max of Delta_2 over R_2",
                                      units="Delta W (dimensionless)",
                                      normalization="divided by Delta W"),
        "max_delta2_ref_over_dw": dict(definition="max of Delta_2 over the reference "
                                                  "support", units="Delta W (dimensionless)",
                                       normalization="divided by Delta W"),
    }
    docs = {k: v for k, v in docs.items() if v is not None}
    pack.found_table("T10", rel, script=f"{MOD}:build_t10", docs=docs, tier="final and development",
                     notes="summary block in the .md computed from the same CSV (groupby q, tier); "
                           "tier per row in column tier",
                     caption="Summary (computed from the CSV below):\n\n" + "\n".join(lines))
    # ---- cross-check with dreach_reach_mask_check.md
    xc = XCheck(pack, "T10", R_DR, "mask comparison table and dReach effect")
    for t in xc.tables_with(["q", "Tier", "Evaluations", "Total holes"]):
        for r in t["rows"]:
            q, tier = int(_nums(r[0])[0]), r[1]
            s = summ[(summ.q == q) & (summ.tier == tier)]
            if s.empty:
                xc.unmatched(f"{q} {tier}")
                continue
            s = s.iloc[0]
            xc.cmp(f"evaluations q={q} {tier}", s.evaluations, _nums(r[2])[0], f"line {t['line']}")
            xc.cmp(f"total holes q={q} {tier}", s.total_holes, _nums(r[3])[0], f"line {t['line']}")
            e = _nums(r[4])
            for k, nm in enumerate(("with_0_extras", "with_1_extra", "with_2_extras")):
                xc.cmp(f"{nm} q={q} {tier}", s[nm], e[k], f"line {t['line']}")
    m = xc.find(r"\(official - reference-mask\)/ΔW, (\d+) of (\d+) evaluations \| exactly 0")
    if m:
        xc.cmp("evaluations with zero dReach difference", int((diff == 0).sum()), m.group(1),
               "effect table")
        xc.cmp("evaluations", len(d), m.group(2), "effect table")
    m = xc.find(r"the one exception \| (" + _NUM + r") \|")
    if m and len(ex) == 1:
        xc.cmp("the nonzero dReach difference",
               float(ex.iloc[0]["dreach_official_minus_refmask_over_dw"]), m.group(1),
               "effect table")
    m = xc.find(r"max Δ₂/ΔW is (" + _NUM + r") over R₂ but (" + _NUM + r") over the reference support")
    if m and len(ex) == 1:
        xc.cmp("max Delta_2 over R_2 (exception)", float(ex.iloc[0]["max_delta2_R2_over_dw"]),
               m.group(1), "prose")
        xc.cmp("max Delta_2 over the reference "
               "(exception)", float(ex.iloc[0]["max_delta2_ref_over_dw"]), m.group(2), "prose")
    m = xc.find(r"Official dReach/ΔW is (" + _NUM + r"); the reference-mask value is "
                                                    r"(" + _NUM + r")")
    if m and len(ex) == 1:
        xc.cmp("official dReach (exception)", float(ex.iloc[0]["dreach_official_over_dw"]),
               m.group(1), "prose")
        xc.cmp("reference-mask dReach (exception)", float(ex.iloc[0]["dreach_refmask_over_dw"]),
               m.group(2), "prose")
    m = xc.find(r"same checkpoint has drift_BR = (" + _NUM + r")")
    if m and len(same_ck) == 1:
        xc.cmp("drift_BR of the exception on the final "
               "tier", float(same_ck.iloc[0]["drift_BR"]), m.group(1), "prose")
    m = xc.find(r"All (\d+) extra nodes are at distance")
    if m:
        xc.cmp("extra nodes in total", int(d["n_extra"].sum()), m.group(1), "prose")
    m = xc.find(r"the 80 checkpoints alone account for (\d+) holes and (\d+) two-extra cases")
    if m:
        xc.cmp("holes among the checkpoints", int(d.loc[ck, "n_holes"].sum()), m.group(1), "prose")
        xc.cmp("two-extra cases among the checkpoints", int(((d["n_extra"] == 2) & ck).sum()),
               m.group(2), "prose")
    xc.done()


# ----------------------------------------------------------------------------------------------
# T11 benchmark consistency
# ----------------------------------------------------------------------------------------------

def build_t11(pack: C.Pack) -> None:
    """T11: Monte Carlo terminal win probability against the analytic CDF (max deviation over the MC SE), per q."""
    rel, rel2 = f"{P1}/benchmark_consistency.csv", f"{P1}/benchmark_recheck.csv"
    b, rc = _csv(rel), _csv(rel2)
    rows: List[Dict[str, Any]] = []

    def add(q: int, block: str, quantity: str, value: float, detail: str, computation: str) -> None:
        rows.append({"q": q, "block": block, "quantity": quantity, "value": value, "detail": detail,
                     "computation": computation})

    BW, BR, BE = "MC terminal win probability", "recheck of the max-|z| cell", "MC expected terminal reward"
    for q, g in b.groupby("q", sort=True):
        q = int(q)
        dn, en = sorted(g["d"].unique()), sorted(g["e_i"].unique())
        add(q, BW, "grid cells", len(g), f"d: {len(dn)} nodes on [{dn[0]:g}, {dn[-1]:g}] x e_i, "
                                         f"e_j in "
            f"{{{', '.join(f'{e:g}' for e in en)}}}", "rows of benchmark_consistency.csv")
        add(q, BW, "draws per cell and player", float(g["n"].iloc[0]),
            "both players use the same draws", "column n")
        intr = (g["F_i"] > 0) & (g["F_i"] < 1)
        add(q, BW, "interior cells (0 < F < 1)", int(intr.sum()), "player i",
            "count of 0 < F_i < 1")
        add(q, BW, "z mean over interior cells", float(g.loc[intr, "z_i"].mean()),
            "player i; z = (p_hat - F)/SE, "
            "SE = sqrt(F(1 - F)/n)", "mean of z_i")
        add(q, BW, "z SD over interior cells", float(g.loc[intr, "z_i"].std(ddof=1)),
            "player i; ddof = 1", "SD of z_i")
        zi, zj = g["z_i"].abs(), g["z_j"].abs()
        who = "j" if zj.max() > zi.max() else "i"
        r = g.loc[(zj if who == "j" else zi).idxmax()]
        add(q, BW, "max |z| = max deviation over the MC standard "
                   "error", float(max(zi.max(), zj.max())),
            f"cell d={r['d']:g}, e_i={r['e_i']:g}, "
            f"e_j={r['e_j']:g}, player {who}", "max over cells and players of |z|")
        add(q, BW, "max |p_hat - F| over interior cells",
            float(max(g.loc[intr, "diff_i"].abs().max(),
                                                                      g.loc[intr, "diff_j"].abs().max())),
            "both players", "max |diff|")
        add(q, BW, "cells with F in {0, 1}", int((~intr).sum()), "player i", "count")
        add(q, BW, "max |p_hat - F| over cells with F in {0, "
                   "1}", float(max(g.loc[~intr, "diff_i"].abs().max(),
                                                                              g.loc[~intr, "diff_j"].abs().max())),
            "both players", "max |diff|")
        add(q, BW, "max |z_i + z_j|", float((g["z_i"] + g["z_j"]).abs().max()),
            "player j's outcome is the complement of player i's on the same draws (z_j = -z_i): "
            "not independent tests",
            "max over cells")
        rr = rc[rc["q"] == q].iloc[0]
        add(q, BR, "F at the recheck cell", float(rr["F"]), f"own gap {rr['cell_d_own']:g}, "
                                                            f"e_own {rr['cell_e_own']:g}, "
            f"e_opp {rr['cell_e_opp']:g} (the max-|z| cell in the player's own "
            f"perspective)", "column F")
        add(q, BR, "replicates", float(rr["reps"]), f"{int(rr['n_per_rep'])} draws each (fresh "
                                                    f"streams)", "column reps")
        for k, nm in (("z_mean", "z mean"), ("z_sd", "z SD (ddof = 1)"), ("z_mean_over_se",
                                                                          "z mean over its SE"),
                      ("max_abs_z", "max |z|")):
            add(q, BR, nm, float(rr[k]), "40 replicates", f"column {k}")
        ri, rj = g["r_z_i"].abs(), g["r_z_j"].abs()
        whor = "j" if rj.max() > ri.max() else "i"
        r = g.loc[(rj if whor == "j" else ri).idxmax()]
        add(q, BE, "max |z| of the mean reward over cells with a random "
                   "reward", float(max(ri.max(), rj.max())),
            f"cell d={r['d']:g}, e_i={r['e_i']:g}, "
            f"e_j={r['e_j']:g}, player {whor}; r_bar = w_L + DW F - k e^2",
            "max |r_z|")
        add(q, BE, "cells with a constant reward", int(g["r_const_i"].astype(bool).sum()),
            "player i (z undefined, NaN)",
            "count of r_const_i")
        add(q, BE, "max |r_hat - r_bar| over constant-reward cells",
            float(max(g.loc[g["r_const_i"].astype(bool), "r_diff_i"].abs().max(),
                      g.loc[g["r_const_j"].astype(bool),
                            "r_diff_j"].abs().max())), "both players", "max |r_diff|")
    df = pd.DataFrame(rows)
    docs = {"q": None, "block": "Check: terminal win probability (benchmark_consistency.csv), "
                                "recheck of the max-|z| "
                                "cell (benchmark_recheck.csv), or expected terminal reward",
            "quantity": "Summary quantity", "value": dict(definition="Value of the quantity (counts, z-scores, "
                                                                     "probabilities or reward differences as named)",
                                                          units="as named in quantity", tier="tier-independent"),
            "detail": "Location (cell) or definition detail", "computation": "Source column and operation"}
    docs = {k: v for k, v in docs.items() if v is not None}
    pack.table("T11", df, status="generated", sources=[C.src(rel), C.src(rel2),
                                                       C.src("tools/v2/benchmark_consistency.py")],
               script=f"{MOD}:build_t11", docs=docs, tier="tier-independent",
               notes=("Per q from the per-cell CSV (756 cells, 200,000 draws per cell, environment "
                      "code path) and the "
                      "40-replicate recheck CSV; z SD with ddof = 1; maxima over both players."),
               caption=("Monte Carlo of the environment's terminal step against the closed-form "
                        "CDF F_xi. The maximum "
                        "deviation over the MC standard error is max |z|; the two players are not "
                        "independent tests."))
    xc = XCheck(pack, "T11", R_P1, "benchmark consistency (section 4.5 table and prose)")
    for t in xc.tables_with(["Interior cells", "max"]):
        for r in t["rows"]:
            q = int(_nums(r[0])[0])
            s = df[df.q == q].set_index("quantity")["value"]
            xc.cmp(f"interior cells q={q}", s["interior cells (0 < F < 1)"], _nums(r[1])[0],
                   f"line {t['line']}")
            zz = _nums(r[2])
            xc.cmp(f"z mean q={q}", s["z mean over interior cells"], zz[0], f"line {t['line']}")
            xc.cmp(f"z SD q={q}", s["z SD over interior cells"], zz[1], f"line {t['line']}")
            mz = _nums(r[3])
            xc.cmp(f"max |z| q={q}", s["max |z| = max deviation over the MC standard "
                                       "error"], mz[0], f"line {t['line']}")
            det = df[(df.q == q) & (df.quantity.str.startswith("max |z| ="))].detail.iloc[0]
            mm = re.search(r"d=(" + _NUM + r"), e_i=(" + _NUM + r"), e_j=(" + _NUM + r"), player (\w)", r[3])
            md = re.search(r"d=(" + _NUM + r"), e_i=(" + _NUM + r"), e_j=(" + _NUM + r"), player (\w)", det)
            if mm and md:
                for k in range(3):
                    xc.cmp(f"max |z| cell coordinate {k} q={q}", float(md.group(k + 1)),
                           mm.group(k + 1), f"line {t['line']}")
                xc.same(f"max |z| player q={q}", md.group(4), mm.group(4), f"line {t['line']}")
            xc.cmp(f"max |p_hat - F| q={q}", s["max |p_hat - F| over interior cells"],
                   _nums(r[4])[0], f"line {t['line']}")
            xc.cmp(f"max |p_hat - F| (F in 0,1) q={q}", s["max |p_hat - F| over cells with F in "
                                                          "{0, 1}"], _nums(r[5])[0],
                   f"line {t['line']}")
    for q in QS:
        m = xc.find(r"q=" + str(q) + r": z mean (" + _NUM + r"), sd (" + _NUM + r"), max \\?\|z\\?\| (" + _NUM +
                    r")\. The mean is (" + _NUM + r") standard errors from 0")
        if m:
            s = df[(df.q == q) & (df.block == BR)].set_index("quantity")["value"]
            xc.cmp(f"recheck z mean q={q}", s["z mean"], m.group(1), "section 4.5")
            xc.cmp(f"recheck z SD q={q}", s["z SD (ddof = 1)"], m.group(2), "section 4.5")
            xc.cmp(f"recheck max |z| q={q}", s["max |z|"], m.group(3), "section 4.5")
            xc.cmp(f"recheck z mean over SE q={q}", s["z mean over its SE"], m.group(4),
                   "section 4.5")
    m = xc.find(r"Interior cells: max \\?\|z\\?\| "
                r"(" + _NUM + r") at q=50, the same cell as above, and (" + _NUM + r") at q=60")
    if m:
        for i, q in ((1, 50), (2, 60)):
            v = df[(df.q == q) & (df.block == BE) & (df.quantity.str.startswith("max |z|"))].value.iloc[0]
            xc.cmp(f"expected-reward max |z| q={q}", v, m.group(i), "section 4.5")
    m = xc.find(r"Cells where the reward is constant: max \\?\|r̂ − r̄\\?\| = "
                r"(" + _NUM + r")|reward is constant: max [^=]*= (" + _NUM + r")")
    if m:
        v = df[(df.block == BE) & (df.quantity.str.startswith("max |r_hat"))].value.max()
        xc.cmp("max |r_hat - r_bar| over constant cells (both "
               "q)", v, m.group(1) or m.group(2), "section 4.5")
    xc.done()


# ----------------------------------------------------------------------------------------------
# T12 calibration
# ----------------------------------------------------------------------------------------------

T12_METRICS = ["Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "G_max_t1_over_dw",
               "G_max_t2_over_dw", "eta_T_over_dw",
               "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw"]
T12_DEV = ["Gmax_full_over_dw", "G_max_t1_over_dw", "G_max_t2_over_dw", "eta_T_over_dw",
           "EXP_root_over_dw",
           "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw"]


def build_t12(pack: C.Pack) -> None:
    """T12: verifier calibration: floor and zero policy (both tiers, both q), 2x-finer check, Phase 1 vs lock, PI references."""
    p1, lk = _csv(CAL_P1), _csv(CAL_LK)
    bs = _csv("results/v2_pilots/pilot4/analysis/root_game/br_slope.csv")
    oc = _csv("results/v2_pilots/pilot4/analysis/root_game/own_curvature.csv")
    rows: List[Dict[str, Any]] = []
    B1, B2_, B3, B4 = ("1 floor (analytic equilibrium) and zero-effort policy, Phase "
                       "1", "2 Phase 1 2x-finer check",
                       "3 agreement of the lock-time calibration with Phase "
                       "1", "4 PI-side references (SPEC 5.4)")

    def base(block: str, kind: str, q: int, pol: str, tier: str, src: str) -> Dict[str, Any]:
        return {"block": block, "row_kind": kind, "q": q, "policy": pol, "tier": tier,
                "source": src}

    for _, r in p1.sort_values(["q", "policy", "verifier_tier"], key=lambda s: s.map(
            {"development": 0, "dev_2x": 1,
             "final": 2}) if s.name == "verifier_tier" else s).iterrows():
        rows.append({**base(B1, "value", int(r.q), r.policy, r.verifier_tier, CAL_P1),
                     **{m: r[m] for m in T12_METRICS}, "pdl_residual_over_dw": r.pdl_residual_over_dw,
                     "valid": bool(r.valid)})
    for q in QS:
        for pol in ("analytic_eq", "zero"):
            s = {t: p1[(p1.q == q) & (p1.policy == pol) & (p1.verifier_tier == t)].iloc[0]
                 for t in ("development", "dev_2x", "final")}
            for a, b_ in (("dev_2x", "development"), ("final", "dev_2x")):
                rows.append({**base(B2_, "difference", q, pol, f"{a} - {b_}", CAL_P1),
                             **{m: float(s[a][m]) - float(s[b_][m]) for m in T12_DEV},
                             "Gmax_full_t": float(s[a]["Gmax_full_t"]) - float(s[b_]["Gmax_full_t"]),
                             "Gmax_full_d": float(s[a]["Gmax_full_d"]) - float(s[b_]["Gmax_full_d"]),
                             "pdl_residual_over_dw": float(s[a]["pdl_residual_over_dw"]) - float(s[b_]["pdl_residual_over_dw"])})
    internal = 0.0
    n_exact = n_exact_nonzero = n_lossy = 0
    rec_max = 0.0
    for _, r in lk.sort_values(["q", "policy", "tier"], ascending=[True, True, False]).iterrows():
        q, pol, tier = int(r.q), r.policy, r.tier
        rows.append({**base(B3, "value (lock, commit 5b07293)", q, pol, tier, CAL_LK),
                     **{m: r[m] for m in T12_METRICS}, "valid": bool(r.valid)})
        ph = p1[(p1.q == q) & (p1.policy == pol) & (p1.verifier_tier == tier)].iloc[0]
        exact = {m: float(r[m]) - float(ph[m]) for m in T12_METRICS}      # both CSVs parsed round-trip
        recd = {m: float(r[f"diff_vs_phase1_{m}"]) for m in T12_METRICS}  # as written by tools/v2/locked_calibration.py
        n_exact += len(exact)
        n_exact_nonzero += sum(v != 0.0 for v in exact.values())
        n_lossy += sum(float(r[f"phase1_{m}"]) != float(ph[m]) for m in T12_METRICS)
        rec_max = max(rec_max, max(abs(v) for v in recd.values()))
        rows.append({**base(B3, "difference (lock - Phase 1), exact", q, pol, tier,
                            f"{CAL_LK}; {CAL_P1}"), **exact})
        rows.append({**base(B3, "difference as recorded in calibration_locked.csv "
                                "(diff_vs_phase1_*)", q, pol, tier,
                            f"{CAL_LK} diff_vs_phase1_*"), **recd})
    for q in QS:
        z = lk[(lk.q == q) & (lk.policy == "zero")]
        ref = z.iloc[0]
        slope_ref = bs[bs.q == q]["ref_slope"].unique()
        ev2_ref = oc[oc.q == q]["ref_ev2pp_over_2k"].unique()
        assert len(slope_ref) == 1 and len(ev2_ref) == 1
        rows.append({**base(B4, "PI reference", q, "zero", "none (reference value)",
                            f"{CAL_LK} pi_ref_*; root_game ref columns"),
                     "Gmax_full_over_dw": ref.pi_ref_Gmax, "Gmax_full_t": ref.pi_ref_t, "Gmax_full_d": ref.pi_ref_d,
                     "EXP_root_over_dw": ref.pi_ref_root_gain, "br_slope": float(slope_ref[0]),
                     "ev2pp_over_2k": float(ev2_ref[0])})
        for tier in ("final", "development"):
            r = z[z.tier == tier].iloc[0]
            rows.append({**base(B4, "difference (lock value - reference)", q, "zero", tier,
                                f"{CAL_LK} diff_vs_pi_*"),
                         "Gmax_full_over_dw": float(r.diff_vs_pi_Gmax),
                         "Gmax_full_t": float(r.Gmax_full_t) - float(r.pi_ref_t),
                         "Gmax_full_d": float(r.Gmax_full_d) - float(r.pi_ref_d),
                         "EXP_root_over_dw": float(r.diff_vs_pi_root_gain)})
            internal = max(internal, abs(float(r.Gmax_full_over_dw) - float(r.pi_ref_Gmax) - float(r.diff_vs_pi_Gmax)),
                           abs(float(r.EXP_root_over_dw) - float(r.pi_ref_root_gain) - float(r.diff_vs_pi_root_gain)))
        for kind, fn in (("pack minimum over the root-game "
                          "fits", min), ("pack maximum over the root-game fits", max)):
            rows.append({**base(B4, kind, q, "analytic_eq", "final and fine",
                                "results/v2_pilots/pilot4/analysis/root_game/{br_slope,own_curvature}.csv"),
                         "br_slope": float(fn(bs[bs.q == q]["slope"])), "ev2pp_over_2k": float(fn(oc[oc.q == q]["ev2pp_over_2k"]))})
    df = pd.DataFrame(rows)
    cols = ["block", "row_kind", "q", "policy", "tier"] + T12_METRICS + ["pdl_residual_over_dw",
                                                                         "valid", "br_slope",
                                                                         "ev2pp_over_2k", "source"]
    df = df[cols]
    an = df[(df.block == B1) & (df.policy == "analytic_eq")]
    floor_max = float(an[T12_DEV].abs().to_numpy().max())
    lk_diff = df[df.row_kind == "difference (lock - Phase 1), exact"]
    lk_rec = df[df.row_kind.str.startswith("difference as recorded")]
    agree_max = float(lk_diff[T12_DEV].abs().to_numpy().max())
    argmax_same = bool((lk_diff[["Gmax_full_t", "Gmax_full_d"]].abs().to_numpy() == 0).all())
    artefact = (f"tools/v2/locked_calibration.py read calibration.csv with pandas' default float "
                f"parser, which returned "
                f"{n_lossy} of {n_exact} Phase 1 values one ulp away from their stored text (the "
                f"phase1_* columns of "
                "calibration_locked.csv); parsed round-trip, the lock-time and Phase 1 values are "
                "identical "
                f"({n_exact_nonzero} nonzero exact differences out of {n_exact})")
    docs = {"block": "Sub-part of T12: 1 floor and zero policy (Phase 1, three tiers); 2 2x-finer "
                     "check; 3 lock-time "
                     "calibration vs Phase 1; 4 PI-side references",
            "row_kind": "value, difference (named), PI reference, or the pack's range over the "
                        "root-game fits",
            "q": None, "policy": "Candidate: analytic_eq = closed-form equilibrium (e1*, e2*); "
                                 "zero = e_hat = 0 at every stage",
            "tier": "Verifier tier of the row (development, dev_2x, final), or the pair of a "
                    "difference",
            "pdl_residual_over_dw": dict(tier="per row (column tier)"), "valid": None,
            **{m: dict(tier="per row (column tier: development, dev_2x or "
                            "final)") for m in T12_METRICS},
            "br_slope": dict(definition="Root-game best-response slope dBR/de at e1*: PI "
                                        "reference, or the minimum / maximum of "
                                        "the numeric slopes over the Pilot 4 fits (br_slope.csv: "
                                        "tiers final and fine, "
                                        "fit half-widths 1, 2, 4, steps h)", units="dimensionless",
                             tier="final and fine (Pilot 4 root-game "
                                  "analysis)", source="tools/v2/pilot4_root_game.py"),
            "ev2pp_over_2k": dict(definition="E[V_2''] / (2k) at e1*: PI reference, or the minimum "
                                             "/ maximum over the Pilot 4 "
                                             "curvature fits (own_curvature.csv)", units="dimensionless",
                                  tier="final and fine (Pilot 4 root-game "
                                       "analysis)", source="tools/v2/pilot4_root_game.py"),
            "source": "File(s) the row's values were read from"}
    docs = {k: v for k, v in docs.items() if v is not None}
    srcs = [C.src(CAL_P1), C.src(CAL_LK),
            C.src("results/v2_pilots/pilot4/analysis/root_game/br_slope.csv"),
            C.src("results/v2_pilots/pilot4/analysis/root_game/own_curvature.csv"), C.src(R_P1), C.src(R_LOCK)]
    pack.table("T12", df, status="generated", sources=srcs, script=f"{MOD}:build_t12", docs=docs,
               tier="final and development",
               notes=(f"Values from the two calibration CSVs, parsed round-trip; block 2 "
                      f"differences computed from "
                      f"calibration.csv; block 3 'exact' differences recomputed from the stored "
                      f"values: max |difference| "
                      f"{agree_max:g} DW, argmax (t*, d*) "
                      f"identical: {argmax_same}; the recorded diff_vs_phase1_* columns "
                      f"(max |.| {rec_max:.3g}) are a parsing "
                      f"artefact: {artefact}. Block 4 differences are the lock tool's "
                      f"diff_vs_pi_* columns (equal to the recomputation "
                      f"within {internal:.2g}); analytic floor max over every "
                      f"metric, tier and q {floor_max:.4g} DW; PI references from "
                      f"calibration_locked.csv pi_ref_* and the root-game CSVs' ref "
                      "columns (equal to SPEC 5.4); root-game rows give the range of the Pilot 4 "
                      "fits (full table in T49)."),
               caption=("Calibration of the verifier. Block 1: analytic equilibrium (floor) and "
                        "zero-effort policy, both q, "
                        "tiers development / dev_2x / final (Phase 1). Block 2: the 2x-finer check "
                        "(differences between "
                        "tiers). Block 3: the lock-time calibration (locked code) and its "
                        "difference from Phase 1. Block 4: "
                        "PI-side references and the differences of the lock-time zero-policy "
                        "values from them. All gains /DW."))
    # ---- cross-checks
    xc = XCheck(pack, "T12", R_P1, "calibration table and prose (section 4.2)")
    polmap = {"analytic": "analytic_eq", "\u00ea \u2261 0": "zero"}
    for t in xc.tables_with(["Candidate", "Tier", "EXP_root"]):
        hdr = t["header"]
        ci = {h: i for i, h in enumerate(hdr)}
        for r in t["rows"]:
            q, pol, tier = int(_nums(r[0])[0]), polmap.get(r[1]), r[2]
            sub = p1[(p1.q == q) & (p1.policy == pol) & (p1.verifier_tier == tier)]
            if sub.empty:
                xc.unmatched(" | ".join(r[:3]))
                continue
            s = sub.iloc[0]
            for h, m in (("\u011cmax_full", "Gmax_full_over_dw"), ("\u03b7\u2082", "eta_T_over_dw"),
                         ("EXP_root", "EXP_root_over_dw"), ("dReach", "dReach_over_dw"),
                         ("\u0394max_all", "Deltamax_all_over_dw"), ("dFull", "dFull_over_dw")):
                if h in ci:
                    xc.cmp(f"{m} q={q} {pol} {tier}", s[m], _nums(r[ci[h]])[0], f"line {t['line']}")
            td = _nums(r[ci["(t*, d*)"]])
            xc.cmp(f"Gmax_full_t q={q} {pol} {tier}", s.Gmax_full_t, td[0], f"line {t['line']}")
            xc.cmp(f"Gmax_full_d q={q} {pol} {tier}", s.Gmax_full_d, td[1], f"line {t['line']}")
    b2 = df[df.block == B2_].set_index(["q", "policy", "tier"])

    def why(q: int, pol: str, mt: str) -> str:
        s2 = p1[(p1.q == q) & (p1.policy == pol)].set_index("verifier_tier")[mt]
        return f"{CAL_P1} {mt}: dev_2x {float(s2['dev_2x'])!r} - development {float(s2['development'])!r}"
    m = xc.find(r"Analytic: 0 at q=50; \+(" + _NUM + r") in Ĝmax_full, EXP_root and dReach at "
                                                     r"q=60\.")
    if m:
        xc.cmp("dev_2x - dev, analytic q=50, max "
               "|diff|", float(b2.loc[(50, "analytic_eq", "dev_2x - development"),
                                      T12_DEV].abs().max()), "0")
        for mt in ("Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw"):
            xc.cmp(f"dev_2x - dev {mt}, analytic q=60", b2.loc[(60, "analytic_eq",
                                                                "dev_2x - development"), mt], m.group(1))
    m = xc.find(r"Zero effort, q=50: Ĝmax_full \+(" + _NUM + r"), dReach \+(" + _NUM + r")\.")
    if m:
        xc.cmp("dev_2x - dev Gmax_full, zero q=50", b2.loc[(50, "zero", "dev_2x - development"),
                                                           "Gmax_full_over_dw"],
               m.group(1), "section 4.2", why(50, "zero", "Gmax_full_over_dw"))
        xc.cmp("dev_2x - dev dReach, zero q=50", b2.loc[(50, "zero", "dev_2x - development"),
                                                        "dReach_over_dw"], m.group(2),
               "section 4.2", why(50, "zero", "dReach_over_dw"))
    m = xc.find(r"Zero effort, q=60: Ĝmax_full (" + _NUM + r"), dReach \+(" + _NUM + r")\.")
    if m:
        xc.cmp("dev_2x - dev Gmax_full, zero q=60", b2.loc[(60, "zero", "dev_2x - development"),
                                                           "Gmax_full_over_dw"],
               m.group(1), "section 4.2", why(60, "zero", "Gmax_full_over_dw"))
        xc.cmp("dev_2x - dev dReach, zero q=60", b2.loc[(60, "zero", "dev_2x - development"),
                                                        "dReach_over_dw"], m.group(2),
               "section 4.2", why(60, "zero", "dReach_over_dw"))
    m = xc.find(r"the floor is . (" + _NUM + r")·ΔW in every metric")
    if m:
        xc.bound("analytic floor, max over metrics, tiers and "
                 "q", floor_max, m.group(1), "section 4.2")
    m = xc.find(r"Max \\?\|pdl_residual_over_dw\\?\| over the 12 calibration rows is "
                r"(" + _NUM + r")")
    if m:
        xc.cmp("max |pdl residual| over the Phase 1 calibration "
               "rows", float(p1.pdl_residual_over_dw.abs().max()), m.group(1))
    xc.done()
    # lock report tables: plain cells -> Pack.crosscheck on frames built from the T12 values
    b3 = df[(df.block == B3) & (df.row_kind.str.startswith("value"))].copy()
    pack.crosscheck("T12", b3, R_LOCK, header_has=["q", "policy", "tier", "Gmax_full_over_dw",
                                                   "EXP_root_over_dw"],
                    key_map={"q": "q", "policy": "policy", "tier": "tier"},
                    value_map={m: m for m in T12_METRICS if m not in ("G_max_t2_over_dw",)},
                    heading_has="Calibration at the lock", label="lock-time calibration values")
    pi = []
    for q in QS:
        for tier in ("final", "development"):
            r = lk[(lk.q == q) & (lk.policy == "zero") & (lk.tier == tier)].iloc[0]
            pi.append({"q": q, "tier": tier, "Gmax_full_over_dw": r.Gmax_full_over_dw,
                       "pi_ref_Gmax": r.pi_ref_Gmax,
                       "diff_vs_pi_Gmax": r.diff_vs_pi_Gmax, "Gmax_full_t": r.Gmax_full_t,
                       "pi_ref_t": r.pi_ref_t, "Gmax_full_d": r.Gmax_full_d, "pi_ref_d": r.pi_ref_d,
                       "EXP_root_over_dw": r.EXP_root_over_dw, "pi_ref_root_gain": r.pi_ref_root_gain,
                       "diff_vs_pi_root_gain": r.diff_vs_pi_root_gain})
    pi = pd.DataFrame(pi)
    pack.crosscheck("T12", pi, R_LOCK, header_has=["q", "tier", "pi_ref_Gmax", "diff_vs_pi_Gmax"],
                    key_map={"q": "q", "tier": "tier"}, value_map={c: c for c in pi.columns if c not in ("q", "tier")},
                    heading_has="Calibration at the lock", label="zero policy vs PI references")
    dd = []
    for _, r in lk_rec.iterrows():
        dd.append({"q": r.q, "policy": r.policy, "tier": r.tier,
                   "max_abs_diff_vs_phase1": float(np.abs(r[T12_METRICS].astype(float)).max()),
                   **{f"diff_vs_phase1_{m}": r[m] for m in ("Gmax_full_over_dw",
                                                            "EXP_root_over_dw", "dReach_over_dw",
                                                            "Gmax_full_t", "Gmax_full_d")}})
    dd = pd.DataFrame(dd)
    pack.crosscheck("T12", dd, R_LOCK, header_has=["q", "policy", "tier", "max_abs_diff_vs_phase1"],
                    key_map={"q": "q", "policy": "policy", "tier": "tier"},
                    value_map={c: c for c in dd.columns if c not in ("q", "policy", "tier")},
                    heading_has="Calibration at the lock",
                    label="lock - Phase 1 differences as recorded (diff_vs_phase1_* columns; "
                          "rendering check)")
    xc = XCheck(pack, "T12", R_LOCK, "lock - Phase 1 differences, exact (both CSVs parsed "
                                     "round-trip)")
    for t in xc.tables_with(["q", "policy", "tier", "max_abs_diff_vs_phase1"],
                            "Calibration at the lock"):
        hdr = t["header"]
        for r in t["rows"]:
            q, pol, tier = int(_nums(r[0])[0]), r[1], r[2]
            sub = lk_diff[(lk_diff.q == q) & (lk_diff.policy == pol) & (lk_diff.tier == tier)]
            if sub.empty:
                xc.unmatched(" | ".join(r[:3]))
                continue
            for ci, h in enumerate(hdr[3:], start=3):
                m = "max_abs" if h == "max_abs_diff_vs_phase1" else h[len("diff_vs_phase1_"):]
                v = float(np.abs(sub.iloc[0][T12_METRICS].astype(float)).max()) if m == "max_abs" else float(sub.iloc[0][m])
                xc.cmp(f"{h} q={q} {pol} {tier} (exact)", v, _nums(r[ci])[0], f"line {t['line']}",
                       comment=artefact)
    xc.done()
    tt = lk[["q", "policy", "tier", "Gmax_full_t", "phase1_Gmax_full_t", "Gmax_full_d",
             "phase1_Gmax_full_d"]].copy()
    pack.crosscheck("T12", tt, R_LOCK, header_has=["q", "policy", "tier", "phase1_Gmax_full_t"],
                    key_map={"q": "q", "policy": "policy", "tier": "tier"},
                    value_map={c: c for c in tt.columns if c not in ("q", "policy", "tier")},
                    heading_has="Calibration at the lock", label="argmax (t*, d*) vs Phase 1")
    xc = XCheck(pack, "T12", R_LOCK, "calibration prose (section 3)")
    zf = {q: lk[(lk.q == q) & (lk.policy == "zero") & (lk.tier == "final")].iloc[0] for q in QS}
    m = xc.find(r"Ĝmax_full/ΔW = (" + _NUM + r") \(q=50\) and (" + _NUM + r") \(q=60\), at stage 2")
    if m:
        xc.cmp("zero final Gmax q=50", zf[50].Gmax_full_over_dw, m.group(1))
        xc.cmp("zero final Gmax q=60", zf[60].Gmax_full_over_dw, m.group(2))
    m = xc.find(r"root dynamic gains \(EXP_root\) "
                r"(" + _NUM + r") and (" + _NUM + r"), vs (" + _NUM + r") and (" + _NUM + r")")
    if m:
        xc.cmp("zero final EXP_root q=50", zf[50].EXP_root_over_dw, m.group(1))
        xc.cmp("zero final EXP_root q=60", zf[60].EXP_root_over_dw, m.group(2))
        xc.cmp("PI root gain q=50", zf[50].pi_ref_root_gain, m.group(3))
        xc.cmp("PI root gain q=60", zf[60].pi_ref_root_gain, m.group(4))
    m = xc.find(r"Differences: . (" + _NUM + r") in Ĝmax and . (" + _NUM + r") in the root gain")
    if m:
        xc.cmp("max |zero final Gmax - PI|",
               max(abs(float(zf[q].Gmax_full_over_dw) - float(zf[q].pi_ref_Gmax)) for q in QS),
               m.group(1))
        xc.cmp("max |zero final EXP_root - PI root gain|",
               max(abs(float(zf[q].EXP_root_over_dw) - float(zf[q].pi_ref_root_gain)) for q in QS), m.group(2))
    m = xc.find(r"Every metric agrees with Phase 1 to . (" + _NUM + r")")
    if m:
        xc.cmp("max |lock - Phase 1| over metrics, as recorded "
               "(diff_vs_phase1_*)", rec_max, m.group(1))
        xc.cmp("max |lock - Phase 1| over metrics, exact", agree_max, m.group(1),
               comment="the report attributes the differences to floating-point summation "
                       "order; " + artefact)
    m = xc.find(r"The analytic floor is . (" + _NUM + r") everywhere except q=60 on the final "
                                                      r"tier, where it is (" + _NUM + r")")
    if m:
        lka = df[(df.block == B3) & (df.policy == "analytic_eq") & df.row_kind.str.startswith("value")]
        rest = lka[~((lka.q == 60) & (lka.tier == "final"))][T12_DEV].abs().to_numpy().max()
        q60f = lka[(lka.q == 60) & (lka.tier == "final")][T12_DEV].abs().to_numpy().max()
        xc.cmp("analytic floor except q=60 final (lock)", float(rest), m.group(1))
        xc.cmp("analytic floor q=60 final (lock)", float(q60f), m.group(2))
    xc.done()
    xc = XCheck(pack, "T12", "reports/v2/pilot4_stabilization.md",
                "root-game references and the range of the fits (section 1a)")
    rng = r"(" + _NUM + r")\s*(?:\u2013|to)\s*(" + _NUM + r")"
    for t in xc.tables_with(["q", "quantity", "reference", "fine tier"]):
        for r in t["rows"]:
            q = int(_nums(r[0])[0])
            col = "ev2pp_over_2k" if r[1].startswith("E[V") else "br_slope" if r[1].startswith("dBR") else None
            if col is None:
                continue
            b4 = df[(df.block == B4) & (df.q == q)].set_index("row_kind")[col]
            xc.cmp(f"PI reference {col} q={q}", b4["PI reference"], _nums(r[2])[0],
                   f"line {t['line']}")
            ends = [x for c in (r[3], r[4]) for x in re.match(r"\s*" + rng, c).groups()]
            lo_s, hi_s = min(ends, key=float), max(ends, key=float)
            xc.cmp(f"minimum of the root-game fits {col} q={q}",
                   b4["pack minimum over the root-game fits"], lo_s,
                   f"line {t['line']} (lower end over the final and fine ranges)")
            xc.cmp(f"maximum of the root-game fits {col} q={q}",
                   b4["pack maximum over the root-game fits"], hi_s,
                   f"line {t['line']} (upper end over the final and fine ranges)")
    xc.done()
    xc = XCheck(pack, "T12", R_SUM, "zero-policy Gmax at the lock; lock vs Phase 1 bound")
    m = xc.find(r"Calibration at the lock matches Phase 1 to . (" + _NUM + r")")
    if m:
        xc.bound("max |lock - Phase 1| (exact)", agree_max, m.group(1))
    m = xc.find(r"Zero policy: (" + _NUM + r") / (" + _NUM + r") at stage 2")
    if m:
        xc.cmp("zero final Gmax q=50", zf[50].Gmax_full_over_dw, m.group(1))
        xc.cmp("zero final Gmax q=60", zf[60].Gmax_full_over_dw, m.group(2))
    xc.done()


# ----------------------------------------------------------------------------------------------
# F02 zero-effort deviation gains
# ----------------------------------------------------------------------------------------------

def build_f02(pack: C.Pack) -> None:
    """F02: G_2(d)/DW and Delta_2(d)/DW of the zero policy on the final-tier D_2, with G_1(0)/DW, both q (saved arrays)."""
    fig, axes = style.new_figure(1, 2, height=4.3, sharey=True)
    cal, lk = _csv(CAL_P1), _csv(CAL_LK)
    rows: List[Dict[str, Any]] = []
    checks: List[str] = []
    srcs = [C.src(CAL_P1), C.src(CAL_LK)]
    for k, q in enumerate(QS):
        rel = f"{P1}/calibration/npz/q{q}_zero_final.npz"
        srcs.append(C.src(rel, "v_t2_d_grid, v_t2_G, v_t2_delta, v_t1_G"))
        z = np.load(C.abspath(rel))
        dw = float(_spec(q).dw)
        d, G2, D2, G1 = z["v_t2_d_grid"], z["v_t2_G"] / dw, z["v_t2_delta"] / dw, float(z["v_t1_G"][0]) / dw
        ax = axes[0, k]
        col, mk = style.Q_COLORS[q], style.Q_MARKER[q]
        ax.plot(d, G2, color=col, lw=2.6, alpha=0.55, label=f"$G_2(d)/\\Delta W$ ({d.size} nodes)")
        ax.plot(d, D2, color=style.INK, lw=0.9, ls="--", label="$\\Delta_2(d)/\\Delta W$")
        ax.plot([0.0], [G1], ls="none", marker=mk, ms=6.5, mfc=col, mec=style.INK, mew=0.8,
                label=f"stage 1: $G_1(0)/\\Delta W$ = {G1:.4f}")
        j = int(np.argmax(G2))
        ax.plot([d[j]], [G2[j]], ls="none", marker="v", ms=7, mfc="white", mec=col, mew=1.4,
                label=f"max $G_2$ = {G2[j]:.5f} at d = {d[j]:g}".replace("-", "\u2212"))
        r = lk[(lk.q == q) & (lk.policy == "zero") & (lk.tier == "final")].iloc[0]
        ax.plot([r.pi_ref_d], [r.pi_ref_Gmax], ls="none", marker="x", ms=7, mew=1.4,
                color=style.REF,
                label=f"PI reference {r.pi_ref_Gmax:g} at d "
                      f"\u2248 {r.pi_ref_d:g}".replace("-", "\u2212"))
        ax.set_title(f"q = {q}, zero-effort policy, final tier")
        ax.set_xlabel("gap d at the start of stage 2 (effort units)")
        if k == 0:
            ax.set_ylabel("deviation gain (units of $\\Delta W$)")
        ax.set_ylim(-0.01, 0.30)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=8)
        rows += [{"q": q, "series": "G2_over_dw", "d": float(a), "value": float(b)} for a,
                 b in zip(d, G2)]
        rows += [{"q": q, "series": "Delta2_over_dw", "d": float(a), "value": float(b)} for a,
                 b in zip(d, D2)]
        rows += [{"q": q, "series": "G1_at_0_over_dw", "d": 0.0, "value": G1},
                 {"q": q, "series": "G2_max_over_dw", "d": float(d[j]), "value": float(G2[j])},
                 {"q": q, "series": "pi_reference_Gmax_over_dw", "d": float(r.pi_ref_d),
                  "value": float(r.pi_ref_Gmax)}]
        c1 = cal[(cal.q == q) & (cal.policy == "zero") & (cal.verifier_tier == "final")].iloc[0]
        checks.append(f"q={q}: plotted max G_2/DW - calibration.csv Gmax_full_over_dw "
                      f"= {float(G2[j]) - c1.Gmax_full_over_dw:g} "
                      f"at d = {d[j]:g} (Gmax_full_d {c1.Gmax_full_d:g}, "
                      f"Gmax_full_t {int(c1.Gmax_full_t)}); "
                      f"G_1(0)/DW - EXP_root_over_dw = {G1 - c1.EXP_root_over_dw:g}; max |G_2 - "
                      f"Delta_2|/DW = "
                      f"{C.max_abs_diff(G2, D2):g}; nodes {d.size}")
    data = pd.DataFrame(rows)
    docs = {"q": None, "series": ("G2_over_dw / Delta2_over_dw = stage-2 deviation gain / one-step "
                                  "gap of the zero policy "
                                  "on the final-tier D_2 grid; G1_at_0_over_dw = stage-1 gain "
                                  "G_1(0) (= EXP_root); "
                                  "G2_max_over_dw = maximum of G_2 (marker); "
                                  "pi_reference_Gmax_over_dw = PI-side "
                                  "reference (SPEC 5.4, calibration_locked.csv pi_ref_Gmax / "
                                  "pi_ref_d)"),
            "d": dict(definition="Gap d at the start of stage 2 (0 for the stage-1 "
                                 "value)", units="effort units (gap d)"),
            "value": dict(definition="Deviation gain divided by Delta "
                                     "W", units="Delta W (dimensionless)",
                          normalization="divided by Delta W = 4", tier="final")}
    docs = {k: v for k, v in docs.items() if v is not None}
    pack.figure("F02", fig, data, status="generated", sources=srcs, script=f"{MOD}:build_f02",
                docs=docs, tier="final",
                caption=("Zero-effort policy (e_hat = 0 at both stages): stage-2 deviation gain "
                         "G_2(d)/DW (thick line) and "
                         "one-step gap Delta_2(d)/DW (dashed; equal to G_2 at every node) on the "
                         "final-tier D_2 grid "
                         "(state step 2; 201 / 221 nodes), stage-1 gain G_1(0)/DW = EXP_root/DW at "
                         "d = 0 (filled marker), "
                         "maximum of G_2 (triangle) and the PI-side reference (cross). Source: "
                         "saved Phase 1 calibration "
                         "arrays results/v2_pilots/phase1/calibration/npz/q{50,60}_zero_final.npz "
                         "(no re-evaluation); "
                         "final tier."),
                checks=checks)


# ----------------------------------------------------------------------------------------------
# entry
# ----------------------------------------------------------------------------------------------

def build() -> None:
    """Build every item of this module and save the fragment."""
    t0 = time.time()
    pack = C.Pack("sec_p0p1")
    for fn in (build_t04, build_f01, build_t05, build_t06, build_t07, build_t08, build_t09,
               build_t10, build_t11,
               build_t12, build_f02):
        fn(pack)
    pack.save_fragment()
    print(f"[sec_p0p1] module run time {time.time() - t0:.1f} s", flush=True)
