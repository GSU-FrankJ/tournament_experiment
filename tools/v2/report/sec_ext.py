"""Pack items of Part II, sections 7, 8 and 10, of the T=2 v2 report pack.

Section 7 (stage-2 accuracy beyond 400 updates): T42, F18, T43, T44, F19.
Section 8 (anatomy of the stage-2 peak gap): T45, T46, T47, F20, F21.
Section 10 (development distributions and the v1.0 rehearsal): T52, T53.
Appendix per-run tables: D04 (Phase A extension), D05 (Pilot 4 section 2a), D06 (Pilot 4 section 2b).

Everything is read from ``results/`` (via ``common.abspath``). The only computation beyond summaries is
the final-tier verifier evaluation of the extension's weight exports at u800 and u1200 (no saved
final-tier value exists there), plus the same evaluation at u400 and u1600 as a check against the
saved final-tier values; both are logged with ``Pack.reeval``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import common as C
import dictionary
import studies as S
import style
import sec_ext_util as U

SCRIPT = "sec_ext.py"
QS = (50, 60)
SEEDS = list(range(10501, 10511))
MARKS = (400, 800, 1200, 1600)

P1 = "results/v2_pilots/pilot1"
PX = "results/v2_pilots/phaseA_ext"
PXA = PX + "/analysis"
P4 = "results/v2_pilots/pilot4/analysis"
RF = P4 + "/repr_floor"
LK = "results/v2_T2_locked"
CUSP = LK + "/cusp_diagnostic"
RA = LK + "/rehearsal_analysis"

REP_EXT = "reports/v2/phaseA_ext.md"
REP_P4 = "reports/v2/pilot4_stabilization.md"
REP_P1 = "reports/v2/pilot1_reward_estimator.md"
REP_LOCK = "reports/v2/protocol_lock_and_rehearsal.md"
REP_CONF = "reports/v2/protocol_v1_1_confirmation.md"
REP_SUM = "reports/v2/summary.md"

BOOT_NOTE = "bootstrap: 95% percentile CI of the mean paired difference, 10,000 resamples, numpy seed 20261001"
# stage-2 metrics whose value depends on the verifier tier (the rest use the recovery grid / direct queries)
FT_KEYS_A = ("eta_T_over_dw", "DeltaT_over_dw_on_max", "DeltaT_over_dw_on_mean_cellmass_weighted",
             "DeltaT_over_dw_off_max", "sigma2_effort_mean_pos")
FT_KEYS_B = ("Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "EXP_root_over_dw", "dReach_over_dw",
             "Deltamax_all_over_dw", "dFull_over_dw", "eta_T_over_dw", "DeltaT_over_dw_on_max",
             "DeltaT_over_dw_off_max", "G_max_t1_over_dw", "G_max_t2_over_dw")
STAGE2_13 = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_peak_locfree_rel_err",
             "stage2_peak_locfree_rel_err_abs", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean", "stage2_tail_max",
             "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "stage2_sym_err_max", "eta_T_over_dw",
             "DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max")
T42_ORDER = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_peak_locfree_rel_err",
             "stage2_peak_locfree_argmax_d", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean",
             "stage2_tail_mean_over_g2_0", "stage2_tail_max", "stage2_tail_max_over_g2_0", "stage2_sym_err_max",
             "eta_T_over_dw", "DeltaT_over_dw_on_max", "DeltaT_over_dw_on_mean_cellmass_weighted",
             "DeltaT_over_dw_off_max", "sigma_effort_at_0_t2", "sigma2_effort_mean_pos", "e_pred_0", "e_learned_0",
             "share_peak_gap_explained")
TIER_ORDER = {"development": 0, "tier-independent": 0, "final": 1}

_CACHE: Dict[str, Any] = {}


# ----------------------------------------------------------------------------------------------
# shared readers
# ----------------------------------------------------------------------------------------------

def _cached(key: str, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def curves() -> pd.DataFrame:
    """Extension learning curves (dev tier), every 25 updates u400-u1600."""
    return _cached("curves", lambda: S.read_csv(f"{PXA}/curves_weights_every25.csv"))


def cand() -> pd.DataFrame:
    """Pilot 4 candidates (curve and tail rows of the families ext, 2a, 2b, p3)."""
    return _cached("cand", lambda: S.read_csv(f"{P4}/candidates_all.csv"))


def paired() -> pd.DataFrame:
    """Pilot 4 paired summaries with the arm and K of both sides parsed from 'a' / 'b'."""
    def load() -> pd.DataFrame:
        p = S.read_csv(f"{P4}/paired_summary.csv")
        for side in ("a", "b"):
            parts = p[side].astype(str).str.extract(r"^(\S+) K=(\d+)$")
            p[f"{side}_arm"], p[f"{side}_K"] = parts[0], parts[1].astype(int)
        return p
    return _cached("paired", load)


def run_records() -> pd.DataFrame:
    """Pilot 4 run records; ``lr_decay`` keeps its JSON text 'null' (constant arms) instead of NaN."""
    def load() -> pd.DataFrame:
        rr = S.read_csv(f"{P4}/run_records.csv")
        rr["lr_decay"] = rr["lr_decay"].fillna("null")
        return rr
    return _cached("run_records", load).copy()


def tier_of(metric: str) -> str:
    """'development' for tier-dependent verifier metrics, else 'tier-independent'."""
    ent = dictionary.E.get(metric)
    dep = (ent[3] is True) if ent else metric.startswith("DeltaT_")
    return "development" if dep else "tier-independent"


def _band(ax, g: pd.DataFrame, x: str, y: str, color: str, label: str, ls: str = "-",
          marker: Optional[str] = None, markevery: Optional[int] = None) -> pd.DataFrame:
    """Plot the across-seed median and IQR band of ``y`` against ``x``; return the plotted statistics."""
    rows = []
    for xv, gg in g.groupby(x, sort=True):
        rows.append({x: xv, "metric": y, **C.median_iqr(gg[y].to_numpy(dtype=float))})
    out = pd.DataFrame(rows)
    style.band_plot(ax, out[x].to_numpy(float), out["median"].to_numpy(float), out["q25"].to_numpy(float),
                    out["q75"].to_numpy(float), color=color, label=label, ls=ls, marker=marker, markevery=markevery)
    return out


def _fig_legend(fig, axes, loc: str, ncols: int = 2, first: str = "") -> None:
    """One figure-level legend (outside the panels) from the labelled artists of all axes, first label wins;
    labels starting with ``first`` are listed first."""
    seen: Dict[str, Any] = {}
    for ax in np.ravel(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    labs = sorted(seen, key=lambda s: 0 if (first and s.startswith(first)) else 1)
    fig.legend([seen[s] for s in labs], labs, loc=loc, ncols=ncols)


def _summ(df: pd.DataFrame, by: Sequence[str], cols: Sequence[str], **const) -> pd.DataFrame:
    """``C.summarize`` plus constant columns."""
    out = C.summarize(df, list(by), list(cols))
    for k, v in const.items():
        out[k] = v
    return out


# ----------------------------------------------------------------------------------------------
# per-run extension table at u400 / u800 / u1200 / u1600 (D04; summarised in T42)
# ----------------------------------------------------------------------------------------------

def _rec_npz(q: int, s: int, u: int) -> str:
    """Saved arrays with the recovery grid of the extension policy at checkpoint u (u400 = Pilot-1 parent)."""
    if u == 400:
        return f"{P1}/q{q}/seed{s}/expected/final_development.npz"
    return f"{PX}/q{q}/seed{s}/expected_ext/checkpoints/u{u:05d}.npz"


def _export(q: int, s: int, u: int) -> str:
    """Weight export of the extension policy at u (u400 = the Pilot-1 parent export)."""
    if u == 400:
        return f"{P1}/q{q}/seed{s}/expected/weights/u00400.npz"
    return f"{PX}/q{q}/seed{s}/expected_ext/weights/u{u:05d}.npz"


def ext_marks(pack: C.Pack) -> Tuple[pd.DataFrame, Dict[str, Any], List[Any]]:
    """Per-run rows of the extension at the four checkpoints, with location-free and final-tier columns.

    Returns:
        (frame, info with check results, sources).
    """
    if "ext_marks" in _CACHE:
        return _CACHE["ext_marks"]
    cv = curves()
    m = cv[cv["update"].isin(MARKS)].sort_values(["q", "seed", "update"]).reset_index(drop=True)
    info: Dict[str, Any] = {}
    # location-free peak error from the saved recovery arrays (tier-independent)
    lf, d0 = [], []
    for r in m.itertuples(index=False):
        q, s, u = int(r.q), int(r.seed), int(r.update)
        z = np.load(C.abspath(_rec_npz(q, s, u)))
        D, e2, g2 = z["recovery_d_grid"], z["recovery_e2"], z["recovery_g2"]
        j0 = int(np.flatnonzero(D == 0.0)[0])
        g20, j = float(g2[j0]), int(np.argmax(e2))
        lf.append({"stage2_peak_locfree_rel_err": (float(e2[j]) - g20) / g20,
                   "stage2_peak_locfree_argmax_d": float(D[j]), "stage2_max_e2": float(e2[j])})
        d0.append((float(e2[j0]) - g20) / g20)
    m = pd.concat([m, pd.DataFrame(lf)], axis=1)
    info["peak0_arrays_vs_curves"] = U.max_abs(d0, m["stage2_peak_rel_err_signed"])
    c = cand()
    ext1 = c[(c.family == "ext") & (c.K == 1)].set_index(["q", "seed"])
    m1600 = m[m["update"] == 1600].set_index(["q", "seed"])
    info["locfree_u1600_vs_candidates"] = U.max_abs(m1600["stage2_peak_locfree_rel_err"],
                                                    ext1.loc[m1600.index, "stage2_peak_locfree_rel_err"])
    # final tier: saved at u400 (Pilot-1 final_v2.json) and u1600 (extension final_v2.json); re-evaluated at
    # u800 and u1200; the re-evaluation is also run at u400 and u1600 to check it against the saved values
    saved: Dict[Tuple[int, int, int], Tuple[Dict[str, Any], str]] = {}
    for q in QS:
        for s in SEEDS:
            f400 = f"{P1}/q{q}/seed{s}/expected/final_v2.json"
            f1600 = f"{PX}/q{q}/seed{s}/expected_ext/final_v2.json"
            saved[(q, s, 400)] = (S.read_json(f400)["final"], f"saved: {f400}[final]")
            saved[(q, s, 1600)] = (S.read_json(f1600)["final"], f"saved: {f1600}[final]")
    reev: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    with U.Timer() as t_new:
        for q in QS:
            for s in SEEDS:
                for u in (800, 1200):
                    reev[(q, s, u)] = U.final_tier_scalars(_export(q, s, u), q)
    with U.Timer() as t_chk:
        diffs = []
        for q in QS:
            for s in SEEDS:
                for u in (400, 1600):
                    sc = U.final_tier_scalars(_export(q, s, u), q)
                    diffs += [abs(float(sc[k]) - float(saved[(q, s, u)][0][k])) for k in FT_KEYS_A]
    info["reeval_vs_saved_max_abs"] = float(max(diffs))
    pack.reeval("T42", f"{PX}/q*/seed*/expected_ext/weights/u00800.npz and u01200.npz (20 runs)", "final",
                t_new.wall, 40, "final-tier stage-2 metrics of the extension at u800 and u1200 (no saved final-tier "
                "evaluation exists there); used in T42 and D04", kind="verifier evaluation",
                note="utils.v2_metrics.evaluate with FINAL_CONFIG on the Beta mean of the exported actor")
    pack.reeval("T42", f"{P1}/q*/seed*/expected/weights/u00400.npz and {PX}/q*/seed*/expected_ext/weights/u01600.npz",
                "final", t_chk.wall, 40, "check: the same evaluation at u400 and u1600 reproduces the saved final-tier "
                f"values of final_v2.json (max abs diff {info['reeval_vs_saved_max_abs']:.3g} over "
                f"{len(diffs)} values)", kind="verifier evaluation")
    ft_rows = []
    for r in m.itertuples(index=False):
        key = (int(r.q), int(r.seed), int(r.update))
        if key in saved:
            sc, src = saved[key]
        else:
            sc, src = reev[key], f"re-evaluated: {_export(*key)}"
        ft_rows.append({**{f"final_tier__{k}": float(sc[k]) for k in FT_KEYS_A}, "final_tier_source": src})
    m = pd.concat([m, pd.DataFrame(ft_rows)], axis=1)
    runs = S.read_csv(f"{PXA}/runs.csv")
    m = m.merge(runs, on=["q", "seed"], how="left", validate="many_to_one")
    m.insert(2, "arm", "expected_ext")
    m.insert(3, "run_dir", [f"{PX}/q{q}/seed{s}/expected_ext" for q, s in zip(m["q"], m["seed"])])
    srcs = [C.src(f"{PXA}/curves_weights_every25.csv"), C.src(f"{PXA}/runs.csv"),
            C.srcs([f"{P1}/q*/seed*/expected/final_development.npz"] +
                   [f"{PX}/q*/seed*/expected_ext/checkpoints/u{u:05d}.npz" for u in (800, 1200, 1600)],
                   "recovery arrays at u400 (Pilot-1 parent) and u800/u1200/u1600", expect=80),
            C.srcs([f"{P1}/q*/seed*/expected/final_v2.json", f"{PX}/q*/seed*/expected_ext/final_v2.json"],
                   "final-tier values saved at u400 (Pilot 1) and u1600 (extension)", expect=40),
            C.srcs([f"{P1}/q*/seed*/expected/weights/u00400.npz"] +
                   [f"{PX}/q*/seed*/expected_ext/weights/u{u:05d}.npz" for u in (800, 1200, 1600)],
                   "weight exports re-evaluated on the final tier", expect=80),
            C.src(f"{P4}/candidates_all.csv")]
    _CACHE["ext_marks"] = (m, info, srcs)
    return _CACHE["ext_marks"]


# ----------------------------------------------------------------------------------------------
# Section 7
# ----------------------------------------------------------------------------------------------

def build_t42(pack: C.Pack) -> None:
    """T42: medians, IQR, min and max of the stage-2 metrics of the extension at u400-u1600 (both tiers)."""
    d04, info, srcs = ext_marks(pack)
    tab_rel = f"{PXA}/table_400_800_1200_1600.csv"
    tab = S.read_csv(tab_rel)
    rec = C.summarize(d04, ["q", "update"], list(dict.fromkeys(tab["metric"])))
    mm = tab.merge(rec, on=["q", "update", "metric"], suffixes=("", "_re"), validate="one_to_one")
    stats = ("median", "q25", "q75", "min", "max")
    d_found = max(U.max_abs(mm[s], mm[s + "_re"]) for s in stats)
    n_ok = bool((mm["n"] == mm["n_re"]).all())
    found = tab.assign(tier=tab["metric"].map(tier_of), source=tab_rel)
    lf = _summ(d04, ["q", "update"], ["stage2_peak_locfree_rel_err", "stage2_peak_locfree_argmax_d"],
               tier="tier-independent", source="recovery arrays (D04): Pilot-1 final_development.npz at u400, "
               "extension checkpoints/u*.npz at u800-u1600")
    ft = _summ(d04, ["q", "update"], [f"final_tier__{k}" for k in FT_KEYS_A], tier="final",
               source="D04 final_tier__* columns: final_v2.json[final] at u400 (Pilot 1) and u1600; "
               "re-evaluated weight exports at u800 and u1200")
    ft["metric"] = ft["metric"].str.replace("final_tier__", "", regex=False)
    t = pd.concat([found, lf, ft], ignore_index=True)
    order = {m: i for i, m in enumerate(T42_ORDER)}
    t["_o"] = t["metric"].map(order)
    t["_t"] = t["tier"].map(TIER_ORDER)
    t = t.sort_values(["q", "_o", "_t", "update"]).drop(columns=["_o", "_t"]).reset_index(drop=True)
    t = t[["q", "update", "metric", "tier", "median", "q25", "q75", "min", "max", "n", "source"]]
    notes = (f"Rows with tier development / tier-independent and source {tab_rel} are that file's rows verbatim "
             f"(recomputed from the per-run values of curves_weights_every25.csv: max abs diff {d_found:.3g}, n equal: "
             f"{n_ok}). Added rows: location-free peak error and its argmax d from the saved recovery arrays (d = 0 value "
             f"of the arrays equals the curves' signed peak error, max abs diff {info['peak0_arrays_vs_curves']:.3g}; u1600 "
             f"location-free equals candidates_all.csv ext K=1, max abs diff {info['locfree_u1600_vs_candidates']:.3g}); "
             "final-tier rows of the five tier-dependent stage-2 metrics (u400: Pilot-1 final_v2.json, the same policy as "
             "the u400 parent export; u1600: extension final_v2.json; u800/u1200: final-tier verifier evaluation of the "
             "weight exports, logged in reevaluations.csv; the same evaluation reproduces the saved u400/u1600 values, "
             f"max abs diff {info['reeval_vs_saved_max_abs']:.3g}). Statistics over 10 seeds per q (10501-10510), numpy "
             "linear percentiles. The u400 row is the Pilot-1 expected parent (seeds are the same runs). Development tier "
             "= state step 4, effort step 1, GL 16; final = 2, 0.5, 32.")
    tdf = pack.table("T42", t, status="generated", sources=[C.src(tab_rel)] + srcs, script=f"{SCRIPT}:build_t42",
                     notes=notes, tier="final and development", docs=U.docs_for(t.columns, "development"),
                     caption="Long format: one row per (q, checkpoint u, metric, tier). Raw effort units for tail, "
                             "symmetry, sigma and e_pred/e_learned; *_over_g2_0 and peak errors are fractions of "
                             "e2*(0); eta_2 and Delta_2 are /Delta W.")
    _check_t42_report(pack, tdf)
    _check_ext_prose(pack, tdf)


def _check_t42_report(pack: C.Pack, t: pd.DataFrame) -> None:
    """Compare T42 with phaseA_ext.md section 2.1 (cells 'median [q25, q75] (min-max)')."""
    ck = U.Checker(pack, "T42", REP_EXT, "section 2.1: every 'median [IQR] (min-max)' cell, per q, metric, update")
    base = t[t["tier"] != "final"]
    for tb in C.parse_md_tables(REP_EXT):
        if tb["header"][:1] != ["metric"] or not tb["heading"].startswith("q = "):
            continue
        q = int(tb["heading"].split("=")[1])
        ck.tables.add(tb["line"])
        for i, row in enumerate(tb["rows"]):
            for j, u in enumerate(tb["header"][1:], start=1):
                cells = U.parse_mid(row[j]) if j < len(row) else None
                hit = base[(base["q"] == q) & (base["update"] == int(u)) & (base["metric"] == row[0])]
                if cells is None or len(hit) != 1:
                    ck.miss()
                    continue
                for st, txt in cells.items():
                    ck.num(f"{row[0]} {st} [q={q}, u{u}]", float(hit.iloc[0][st]), txt, f"line {tb['line'] + 2 + i}")
    ck.close()
    # the added location-free rows at u1600 against pilot4_stabilization.md section 1c.2 (extension, K = 1)
    ck = U.Checker(pack, "T42", REP_P4, "added rows: u1600 location-free peak-error medians vs section 1c.2 (ext, K=1)")
    lf = t[(t["metric"] == "stage2_peak_locfree_rel_err") & (t["update"] == 1600)].set_index("q")["median"]
    for tb in U.find_tables(REP_P4, ["q", "arm", "K", "stage2_peak_rel_err_signed"], "1c.2"):
        ck.tables.add(tb["line"])
        for i, row in enumerate(tb["rows"]):
            rec = dict(zip(tb["header"], row))
            if rec["arm"] == "ext" and rec["K"] == "1":
                ck.num(f"stage2_peak_locfree_rel_err median [q={rec['q']}, u1600]", float(lf.loc[int(rec["q"])]),
                       rec["stage2_peak_locfree_rel_err"], f"line {tb['line'] + 2 + i}")
    ck.close()


def _check_ext_prose(pack: C.Pack, t: pd.DataFrame) -> None:
    """Numbers quoted in the prose of phaseA_ext.md sections 2.3 and 3 against the data."""
    ck = U.Checker(pack, "T42", REP_EXT, "section 3 prose (plateau description) and section 2.3 stop-rule text")
    med = t[t["tier"] != "final"].set_index(["q", "metric", "update"])["median"]
    quotes = {
        ("stage2_peak_rel_err_signed", 50): ["−0.129", "−0.079", "−0.060", "−0.068"],
        ("stage2_peak_rel_err_signed", 60): ["−0.099", "−0.058", "−0.054", "−0.060"],
        ("share_peak_gap_explained", 50): ["0.33", "0.48", "0.58", "0.51"],
        ("share_peak_gap_explained", 60): ["0.38", "0.58", "0.51", "0.40"],
    }
    for (mt, q), cells in quotes.items():
        for u, cell in zip(MARKS, cells):
            ck.num(f"median {mt} [q={q}, u{u}]", float(med[(q, mt, u)]), cell, "section 3")
    ends = {("stage2_rmse_pos_over_g2_0", 50): ("0.044", "0.026"), ("stage2_rmse_pos_over_g2_0", 60): ("0.033", "0.024"),
            ("stage2_tail_mean", 50): ("1.48", "0.52"), ("stage2_tail_mean", 60): ("1.34", "0.53"),
            ("stage2_tail_max", 50): ("4.76", "2.22"), ("stage2_tail_max", 60): ("3.57", "1.90"),
            ("eta_T_over_dw", 50): ("0.0046", "0.0015"), ("eta_T_over_dw", 60): ("0.0018", "0.0009"),
            ("sigma_effort_at_0_t2", 50): ("3.9", "2.7"), ("sigma_effort_at_0_t2", 60): ("4.0", "2.9")}
    for (mt, q), (a, b) in ends.items():
        ck.num(f"median {mt} [q={q}, u400]", float(med[(q, mt, 400)]), a, "section 3")
        ck.num(f"median {mt} [q={q}, u1600]", float(med[(q, mt, 1600)]), b, "section 3")
    cv = curves()
    u16 = cv[(cv["update"] == 1600) & (cv.q == 60)]
    above = u16[u16["e_learned_0"] > U.g2_star_0(60)]
    ck.claim("q=60 u1600 runs with e_learned(0) > e2*(0)", len(above) == 1, len(above), "one q=60 seed", "section 3")
    if len(above) == 1:
        ck.num("signed peak error of that run", float(above.iloc[0]["stage2_peak_rel_err_signed"]), "+0.0035", "section 3")
        ck.num("share of that run", float(above.iloc[0]["share_peak_gap_explained"]), "−7.2", "section 3")
    # per-update medians (25-update curve) quoted as ranges
    pm = cv.groupby(["q", "update"])

    def where(s: pd.Series, fn: str) -> str:
        q_, u_ = getattr(s, fn)()
        return f"q={q_} u{u_}"

    sym = pm["stage2_sym_err_max"].median()
    lo, hi = float(sym.min()), float(sym.max())
    ck.claim("range of the per-update median symmetry error u400-u1600 (both q)", 2.35 <= lo and hi < 3.55,
             f"{lo:.4g} ({where(sym, 'idxmin')}) to {hi:.4g} ({where(sym, 'idxmax')})",
             "the median stays between 2.4 and 3.5 effort units", "section 3",
             "holds at the four checkpoints u400/u800/u1200/u1600 (2.39-3.47), not along the 25-update curve")
    pk = pm["stage2_peak_rel_err_signed"].median()
    late = pk[pk.index.get_level_values("update") >= 800]
    lo, hi = float(late.min()), float(late.max())
    ck.claim("range of the per-update median signed peak error u800-u1600 (both q)", -0.095 <= lo and hi < -0.035,
             f"{lo:.4g} ({where(late, 'idxmin')}) to {hi:.4g} ({where(late, 'idxmax')})",
             "fluctuates between about -0.04 and -0.09", "section 3")
    runs = S.read_csv(f"{PXA}/runs.csv")
    wf = runs["would_have_fired_A"].map(lambda x: json.loads(x)["global_update"])
    ck.claim("stop rule would have fired (runs) and global-update range", len(wf.dropna()) == 20 and wf.min() == 700
             and wf.max() == 1100, f"{len(wf.dropna())} runs, u{int(wf.min())}-u{int(wf.max())}",
             "all 20 continuations, at global updates 700-1100", "section 2.3")
    ck.close()


def build_d04(pack: C.Pack) -> None:
    """D04: per-run table of the extension at u400, u800, u1200, u1600 (80 rows)."""
    d04, info, srcs = ext_marks(pack)
    notes = ("One row per (q, seed, checkpoint u in {400, 800, 1200, 1600}). Columns q ... share_peak_gap_explained are "
             "the original columns of curves_weights_every25.csv (development tier for eta_2 / Delta_2 / sigma2_mean; "
             "u400 is the Pilot-1 expected parent export); commit ... full_states are the original run-level columns of "
             "runs.csv (repeated on the four rows of a run); stage2_peak_locfree_* and stage2_max_e2 are computed from the "
             "saved recovery arrays (tools/v2/pilot4_common.py:location_free; u400: Pilot-1 final_development.npz, "
             "u800-u1600: checkpoints/u*.npz); final_tier__* are the final-tier values (saved final_v2.json at u400 and "
             "u1600; re-evaluated weight exports at u800 and u1200, logged in reevaluations.csv; re-evaluation reproduces "
             f"the saved values, max abs diff {info['reeval_vs_saved_max_abs']:.3g}).")
    df = pack.data("D04", d04, status="generated", sources=srcs, script=f"{SCRIPT}:build_d04", notes=notes,
                   tier="final and development", docs=U.docs_for(d04.columns, "development"))
    # cross-check: section 2.3 run-record summary of phaseA_ext.md
    ck = U.Checker(pack, "D04", REP_EXT, "section 2.3 run records (summary of the run-level columns)")
    runs = df.drop_duplicates(["q", "seed"])
    for tb in U.find_tables(REP_EXT, ["q", "n", "commit", "dirty", "n_would_fire"]):
        ck.tables.add(tb["line"])
        for i, row in enumerate(tb["rows"]):
            rec = dict(zip(tb["header"], row))
            g = runs[runs.q == int(rec["q"])]
            wf = g["would_have_fired_A"].map(lambda x: json.loads(x)["global_update"])
            vals = {"n": len(g), "n_would_fire": int(wf.notna().sum()), "wf_min": wf.min(), "wf_median": wf.median(),
                    "wf_max": wf.max(), "wall_min": g["phase_A_ext_wall_sec"].min(),
                    "wall_median": g["phase_A_ext_wall_sec"].median(), "wall_max": g["phase_A_ext_wall_sec"].max(),
                    "kl_median": g["kl_median"].median(), "clip_median": g["clip_median"].median()}
            where = f"line {tb['line'] + 2 + i}"
            for k, v in vals.items():
                ck.num(f"{k} [q={rec['q']}]", float(v), rec[k], where)
            ck.text(f"commit [q={rec['q']}]", ",".join(sorted(set(g["commit"]))), rec["commit"], where)
            ck.text(f"dirty [q={rec['q']}]", ",".join(sorted(set(str(x) for x in g["dirty"]))), rec["dirty"], where)
    ck.close()


F18_PANELS = (("stage2_peak_rel_err_signed", "peak error at d = 0, signed", "fraction of e₂*(0)"),
              ("stage2_rmse_pos_over_g2_0", "RMSE over |d| < 2q / e₂*(0)", "fraction of e₂*(0)"),
              ("stage2_tail_mean", "tail mean (|d| ≥ 2q)", "effort units (raw)"),
              ("stage2_tail_max", "tail max (|d| ≥ 2q)", "effort units (raw)"),
              ("stage2_sym_err_max", "symmetry error", "effort units"),
              ("eta_T_over_dw", "η₂ (development tier)", "/ΔW"),
              ("DeltaT_over_dw_off_max", "off-path max Δ₂ (development tier)", "/ΔW"),
              ("sigma_effort_at_0_t2", "σ₂(0)", "effort units"))


def build_f18(pack: C.Pack) -> None:
    """F18: extension learning curves u400-u1600, median and IQR per q (restyled from the same CSV)."""
    cv = curves()
    tab_rel = f"{PXA}/table_400_800_1200_1600.csv"
    tab = S.read_csv(tab_rel)
    fig, axes = style.new_figure(4, 2, height=8.4, sharex=True)
    parts = []
    for ax, (mt, title, unit) in zip(axes.ravel(), F18_PANELS):
        for q in QS:
            g = cv[cv.q == q]
            out = _band(ax, g, "update", mt, style.Q_COLORS[q], style.label_n(f"q = {q}", g.seed.nunique()),
                        ls=style.Q_LINESTYLE[q], marker=style.Q_MARKER[q], markevery=8)
            parts.append(out.assign(q=q, panel=mt))
        if mt == "stage2_peak_rel_err_signed":
            ax.axhline(0.0, color=style.REF, lw=0.8)
            parts.append(pd.DataFrame([{"panel": mt, "series": "zero line", "y_value": 0.0}]))
        ax.set_title(title)
        ax.set_ylabel(unit)
    for ax in axes[-1]:
        ax.set_xlabel("global update (Phase A)")
    _fig_legend(fig, axes, "outside upper center")
    data = pd.concat(parts, ignore_index=True)
    data["series"] = data["series"].fillna("median and IQR over seeds")
    data = data[["panel", "series", "q", "update", "metric", "median", "q25", "q75", "min", "max", "n", "y_value"]]
    pl = data[data["update"].isin(MARKS)].astype({"q": int, "update": int})
    pl = pl.merge(tab, on=["q", "update", "metric"], suffixes=("", "_tab"), validate="one_to_one")
    d = max(U.max_abs(pl[s], pl[s + "_tab"]) for s in ("median", "q25", "q75", "min", "max"))
    checks = [f"plotted median/q25/q75/min/max at u400/u800/u1200/u1600 equal {tab_rel} "
              f"(max abs diff {d:.3g} over {len(pl) * 5} values)",
              "same data and panels as reports/v2/figures/phaseA_ext/curves_q50.png and curves_q60.png (one panel per "
              "metric with both q instead of one figure per q)"]
    cap = ("Phase A extension (stage 2 only, continued from u400 to u1600): across-seed median (line) and IQR (band) "
           "of eight stage-2 metrics at every weight export, 25 updates apart; u400 is the Pilot-1 expected parent "
           "export. q = 50 blue solid circles, q = 60 orange dashed squares (markers every 200 updates); n = 10 seeds "
           "per q (10501-10510). eta_2 and off-path Delta_2 are development tier (state step 4, effort step 1, GL 16); "
           "the other panels use the recovery grid or a direct policy query (tier-independent). Source: "
           f"{PXA}/curves_weights_every25.csv.")
    pack.figure("F18", fig, U.intify(data), status="regenerated",
                sources=[C.src(f"{PXA}/curves_weights_every25.csv"), C.src(tab_rel),
                         C.src("reports/v2/figures/phaseA_ext/curves_q50.png"),
                         C.src("reports/v2/figures/phaseA_ext/curves_q60.png")],
                script=f"{SCRIPT}:build_f18", caption=cap, checks=checks, tier="development",
                docs=U.docs_for(data.columns, "development"))


T43_METRICS = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_peak_locfree_rel_err",
               "stage2_peak_locfree_argmax_d", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean", "stage2_tail_max",
               "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "stage2_sym_err_max", "eta_T_over_dw",
               "DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max", "DeltaT_over_dw_on_mean_cellmass_weighted",
               "sigma_effort_at_0_t2", "e2_at_0")
PAIRED_COLS = ("n_pairs", "mean", "median", "min", "max", "n_neg", "n_pos", "n_zero", "boot_ci95_lo", "boot_ci95_hi",
               "n_better", "better_if")


def _p2a_runs() -> pd.DataFrame:
    """2a last iterates at u1600 (candidates_all tail K=1) with their final-tier values (final_v2.json)."""
    def load() -> pd.DataFrame:
        c = cand()
        t = c[(c.family == "2a") & (c.kind == "tail") & (c.K == 1)].copy()
        ft = S.final_tier_columns("pilot4_A")
        keep = ["q", "seed", "arm", "run_dir"] + [f"final_tier__{k}" for k in FT_KEYS_A]
        return t.merge(ft[keep], on=["q", "seed", "arm"], how="left", validate="one_to_one")
    return _cached("p2a_runs", load)


def build_t43(pack: C.Pack) -> None:
    """T43: Pilot 4 section 2a (LR decay over u1201-u1600): final table, paired differences, reproducibility."""
    per = _p2a_runs()
    blocks: List[pd.DataFrame] = []
    rows = []
    for (q, arm), g in per.groupby(["q", "arm"]):
        for mt in T43_METRICS:
            rows.append({"block": "final_u1600", "q": q, "arm": arm, "K": 1, "metric": mt, "tier": tier_of(mt),
                         **C.median_iqr(g[mt]), "source": f"{P4}/candidates_all.csv (family 2a, tail, K=1)"})
        for k in FT_KEYS_A:
            rows.append({"block": "final_u1600", "q": q, "arm": arm, "K": 1, "metric": k, "tier": "final",
                         **C.median_iqr(g[f"final_tier__{k}"]), "source": "final_v2.json[final] of the 2a runs"})
    blocks.append(pd.DataFrame(rows))
    p = paired()
    pd2 = p[(p.comparison == "decay_minus_constant") & (p.family == "2a")]
    blocks.append(pd.DataFrame({"block": "paired_decay_minus_constant", "q": pd2["q"], "arm": "decay - constant",
                                "K": pd2["a_K"], "metric": pd2["metric"], "tier": pd2["metric"].map(tier_of),
                                **{c: pd2[c] for c in PAIRED_COLS}, "source": f"{P4}/paired_summary.csv"}))
    # final-tier paired differences (not in any existing file): recomputed, bootstrap as the existing ones
    rows = []
    for q in QS:
        A = per[(per.q == q) & (per.arm == "decay")].set_index("seed").loc[SEEDS]
        B = per[(per.q == q) & (per.arm == "constant")].set_index("seed").loc[SEEDS]
        for k in ("eta_T_over_dw", "DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max"):
            dv = (A[f"final_tier__{k}"] - B[f"final_tier__{k}"]).to_numpy(float)
            lo, hi = C.bootstrap_mean_ci(dv, 10000, 20261001)
            rows.append({"block": "paired_decay_minus_constant", "q": q, "arm": "decay - constant", "K": 1, "metric": k,
                         "tier": "final", "n_pairs": dv.size, "mean": dv.mean(), "median": float(np.median(dv)),
                         "min": dv.min(), "max": dv.max(), "n_neg": int((dv < 0).sum()), "n_pos": int((dv > 0).sum()),
                         "n_zero": int((dv == 0).sum()), "boot_ci95_lo": lo, "boot_ci95_hi": hi,
                         "n_better": int((dv < 0).sum()), "better_if": "diff < 0",
                         "source": "recomputed here from final_v2.json[final] (C.bootstrap_mean_ci, seed 20261001)"})
    blocks.append(pd.DataFrame(rows))
    pr = S.read_csv(f"{P4}/paired_summary_run_records.csv")
    pr = pr[pr.family == "2a"]
    blocks.append(pd.DataFrame({"block": "paired_run_records", "q": pr["q"], "arm": "decay - constant", "K": np.nan,
                                "metric": pr["metric"], "tier": "n/a", **{c: pr[c] for c in PAIRED_COLS},
                                "source": f"{P4}/paired_summary_run_records.csv"}))
    blocks.append(_t43_run_records())
    blocks.append(_t43_repro())
    t = U.union_frame(blocks, ["block", "q", "arm", "K", "metric", "tier", "median", "q25", "q75", "min", "max", "n"])
    notes = ("Blocks: final_u1600 = last iterate at u1600 per arm and q, median/IQR/min/max over seeds 10501-10510 "
             "(development tier from candidates_all.csv, the dev re-evaluation of the u1600 export; final tier from "
             "final_v2.json['final'] of each run); paired_decay_minus_constant = decay - constant per (q, seed) from "
             f"paired_summary.csv (K = 1, 4, 8; {BOOT_NOTE}) plus final-tier rows (K = 1) recomputed here with "
             "common.bootstrap_mean_ci (same resamples and seed, fresh generator per metric); paired_run_records = "
             "run-level paired differences (paired_summary_run_records.csv); run_records = per (q, arm) summary of "
             "run_records.csv (value = distinct values / counts; median/q25/q75/min/max over runs); reproducibility = "
             "repro_2a_constant_vs_ext.csv and verifier_consumes_no_rng.csv counts (value = runs with True), plus the "
             "check that the development-tier curves of the constant arm equal the extension's at u1225-u1600.")
    tdf = pack.table("T43", t, status="generated",
                     sources=[C.src(f"{P4}/candidates_all.csv"), C.src(f"{P4}/paired_summary.csv"),
                              C.src(f"{P4}/paired_summary_run_records.csv"), C.src(f"{P4}/run_records.csv"),
                              C.src(f"{P4}/repro_2a_constant_vs_ext.csv"), C.src(f"{P4}/verifier_consumes_no_rng.csv"),
                              C.src(f"{PXA}/curves_weights_every25.csv"),
                              C.srcs(f"{S.STUDIES['pilot4_A']['root']}/q*/seed*/*/final_v2.json", "2a final_v2.json",
                                     expect=40)],
                     script=f"{SCRIPT}:build_t43", notes=notes, tier="final and development",
                     docs=U.docs_for(t.columns, "development", {
                         "value": "Text or count of the row: distinct values (commit, dirty, verifier updates), number "
                                  "of runs with True (reproducibility checks), or the max abs difference (identity check)",
                         "arm": "Arm (constant LR / LR decay 3e-4 -> 3e-5 over u1201-u1600), or 'decay - constant' for "
                                "paired rows",
                         "source": "File the row was read from or computed from"}),
                     caption="Pilot 4 section 2a: Phase A continued from the extension's u1200 state to u1600 with a "
                             "constant LR or a linear decay 3e-4 -> 3e-5. See notes for the blocks.")
    _check_t43(pack, tdf)


def _t43_run_records() -> pd.DataFrame:
    """Per (q, arm) summary of the 2a run records."""
    rr = run_records()
    rr = rr[rr.family == "2a"]
    rows = []
    src = f"{P4}/run_records.csv (family 2a)"
    for (q, arm), g in rr.groupby(["q", "arm"]):
        base = {"block": "run_records", "q": q, "arm": arm, "tier": "n/a", "source": src}
        rows.append({**base, "metric": "commit", "value": ",".join(sorted(set(g["commit"]))), "n": len(g)})
        rows.append({**base, "metric": "dirty", "value": ",".join(sorted(set(str(x) for x in g["dirty"]))), "n": len(g)})
        rows.append({**base, "metric": "n_would_fire", "value": str(int(g["would_fire_update"].notna().sum())),
                     "n": len(g)})
        for mt in ("actor_lr_first", "actor_lr_last", "would_fire_update", "phase_wall_sec", "kl_median", "clip_median",
                   "adv_used_std_median"):
            rows.append({**base, "metric": mt, **C.median_iqr(g[mt])})
    return pd.DataFrame(rows)


def _t43_repro() -> pd.DataFrame:
    """Reproducibility block: constant arm vs extension (bit identity) and verifier RNG consumption."""
    rp = S.read_csv(f"{P4}/repro_2a_constant_vs_ext.csv")
    vr = S.read_csv(f"{P4}/verifier_consumes_no_rng.csv")
    rows = []
    for q in QS:
        g, v = rp[rp.q == q], vr[vr.q == q]
        base = {"block": "reproducibility", "q": q, "arm": "constant vs extension", "tier": "n/a",
                "source": f"{P4}/repro_2a_constant_vs_ext.csv"}
        for col in ("weight_exports_identical", "final_weights_identical_to_ext_u1600", "state_u1600_agent_identical",
                    "state_u1600_rng_identical", "history_1201_1600_identical", "rng_positions_every_update_identical",
                    "verifier_cadence_differs"):
            rows.append({**base, "metric": col, "n": len(g), "value": str(int(g[col].astype(bool).sum()))})
        rows.append({**base, "metric": "n_exports_compared", "n": len(g),
                     "value": ",".join(str(x) for x in sorted(set(g["n_exports_compared"])))})
        rows.append({**base, "metric": "verifier_updates_constant (= extension)", "n": len(g),
                     "value": ",".join(sorted(set(g["verifier_updates_constant"])))})
        rows.append({"block": "reproducibility", "q": q, "arm": "constant (restored parents)", "tier": "n/a",
                     "metric": "rng_states_unchanged_after_6_verifier_calls", "n": len(v),
                     "value": str(int(v["rng_states_unchanged_after_6_verifier_calls"].astype(bool).sum())),
                     "source": f"{P4}/verifier_consumes_no_rng.csv"})
    # development-tier curves of the constant arm against the extension's (same runs, u1225-u1600)
    c = cand()
    a = c[(c.family == "2a") & (c.arm == "constant") & (c.K == 1)]
    mets = ("stage2_peak_rel_err_signed", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean", "stage2_tail_max",
            "stage2_sym_err_max", "eta_T_over_dw", "DeltaT_over_dw_off_max", "sigma_effort_at_0_t2")
    mm = a.merge(curves(), on=["q", "seed", "update"], suffixes=("", "_ext"), validate="one_to_one")
    for q in QS:
        g = mm[mm.q == q]
        d = max(U.max_abs(g[m], g[m + "_ext"]) for m in mets)
        rows.append({"block": "reproducibility", "q": q, "arm": "constant vs extension", "tier": "development",
                     "metric": "max abs diff of 8 dev-tier curve metrics, u1225-u1600 (" + ", ".join(mets) + ")",
                     "n": len(g), "value": f"{d:.3g}",
                     "source": f"{P4}/candidates_all.csv vs {PXA}/curves_weights_every25.csv"})
    _CACHE["const_vs_ext_rows"] = len(mm)
    return pd.DataFrame(rows)


_P2A_LABELS = {"|peak err|": "stage2_peak_rel_err_abs", "peak err (signed)": "stage2_peak_rel_err_signed",
               "RMSE/e₂*(0)": "stage2_rmse_pos_over_g2_0", "tail mean": "stage2_tail_mean",
               "tail max": "stage2_tail_max", "symmetry err": "stage2_sym_err_max", "η₂": "eta_T_over_dw",
               "Δ₂ off-path max": "DeltaT_over_dw_off_max"}


def _check_t43(pack: C.Pack, t: pd.DataFrame) -> None:
    """Cross-check T43 with pilot4_stabilization.md section 2a."""
    pr = t[t["block"] == "paired_decay_minus_constant"]
    pr = pr[(pr["K"] == 1) & (pr["tier"] != "final")].set_index(["q", "metric"])
    ck = U.Checker(pack, "T43", REP_P4, "section 2a paired decay - constant (K = 1): median, better / sign counts, CI95")
    for tb in U.find_tables(REP_P4, ["q", "metric", "median", "better", "CI95"], "2a."):
        ck.tables.add(tb["line"])
        for ln, cells in U.raw_rows(REP_P4, tb):
            q, label = int(cells[0]), "|".join(cells[1:-3]).strip()
            med, better, ci = cells[-3:]
            mt = _P2A_LABELS.get(label)
            if mt is None or (q, mt) not in pr.index:
                ck.miss()
                continue
            r = pr.loc[(q, mt)]
            w = f"line {ln}"
            ck.num(f"{mt} median [q={q}]", float(r["median"]), med, w)
            sc = re.match(r"^\((\d+)[−-]/(\d+)\+\)$", better.strip())
            if sc:
                ck.num(f"{mt} n_neg [q={q}]", float(r["n_neg"]), sc.group(1), w)
                ck.num(f"{mt} n_pos [q={q}]", float(r["n_pos"]), sc.group(2), w)
            else:
                ck.num(f"{mt} n_better [q={q}]", float(r["n_better"]), better, w)
            lohi = U.parse_ci(ci)
            if lohi:
                ck.num(f"{mt} boot_ci95_lo [q={q}]", float(r["boot_ci95_lo"]), lohi[0], w)
                ck.num(f"{mt} boot_ci95_hi [q={q}]", float(r["boot_ci95_hi"]), lohi[1], w)
    ck.close()
    rr = t[t["block"] == "paired_run_records"].copy()
    rr["family"] = "2a"
    pack.crosscheck("T43", rr, REP_P4, header_has=["family", "q", "metric", "median", "n_better", "boot_ci95_lo"],
                    key_map={"family": "family", "q": "q", "metric": "metric"},
                    value_map={c: c for c in ("median", "n_better", "n_neg", "n_pos", "boot_ci95_lo", "boot_ci95_hi")},
                    label="section 2a run-level paired differences (the 10 unmatched family-2b rows of that table belong "
                          "to T51)")
    # run-records summary table of section 2a
    rec = t[t["block"] == "run_records"]
    wide = []
    for (q, arm), g in rec.groupby(["q", "arm"]):
        gi = g.set_index("metric")
        wide.append({"q": q, "arm": arm, "n": gi.loc["commit", "n"], "lr_first": gi.loc["actor_lr_first", "median"],
                     "lr_last": gi.loc["actor_lr_last", "median"], "n_would_fire": float(gi.loc["n_would_fire", "value"]),
                     "wf_median": gi.loc["would_fire_update", "median"], "wf_min": gi.loc["would_fire_update", "min"],
                     "wf_max": gi.loc["would_fire_update", "max"], "wall_median": gi.loc["phase_wall_sec", "median"],
                     "kl_median": gi.loc["kl_median", "median"], "clip_median": gi.loc["clip_median", "median"],
                     "adv_used_std": gi.loc["adv_used_std_median", "median"]})
    pack.crosscheck("T43", pd.DataFrame(wide), REP_P4, header_has=["q", "arm", "n", "commit", "lr_first", "wf_median"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={c: c for c in ("n", "lr_first", "lr_last", "n_would_fire", "wf_median", "wf_min", "wf_max",
                                              "wall_median", "kl_median", "clip_median", "adv_used_std")},
                    heading_has="2a.", label="section 2a run-records summary")
    # reproducibility prose
    ck = U.Checker(pack, "T43", REP_P4, "section 2a reproducibility statements (20/20 counts, verifier updates)")
    rp = t[t["block"] == "reproducibility"]
    for mt in ("weight_exports_identical", "final_weights_identical_to_ext_u1600", "state_u1600_agent_identical",
               "state_u1600_rng_identical", "history_1201_1600_identical", "rng_positions_every_update_identical"):
        n_true = int(rp.loc[rp["metric"] == mt, "value"].astype(int).sum())
        ck.claim(f"runs with {mt}", n_true == 20, n_true, "20/20 bit-identical", "section 2a")
    vu = set(rp.loc[rp["metric"].str.startswith("verifier_updates"), "value"])
    ck.claim("verifier-call updates of the constant arm", vu == {"1300 1400 1500 1600"}, ";".join(sorted(vu)),
             "u1300, 1400, 1500 and 1600 in every run", "section 2a")
    n_rng = int(rp.loc[rp["metric"] == "rng_states_unchanged_after_6_verifier_calls", "value"].astype(int).sum())
    ck.claim("runs with RNG states unchanged after 6 verifier calls", n_rng == 20, n_rng, "unchanged in 20/20",
             "section 2a")
    ck.close()


def build_t44(pack: C.Pack) -> None:
    """T44: stage-2 tail averaging (extension K in {1, 4, 8, 16}; 2a K in {1, 4, 8})."""
    c = cand()
    t = c[(c.kind == "tail") & c.family.isin(["ext", "2a"])]
    rows = []
    for (fam, arm, K, q), g in t.groupby(["family", "arm", "K", "q"]):
        for mt in STAGE2_13:
            rows.append({"block": "candidates", "family": fam, "arm": arm, "K": K, "q": q, "metric": mt,
                         "tier": tier_of(mt), "quantity": "candidate value", **C.median_iqr(g[mt])})
    cblk = pd.DataFrame(rows)
    fam_order = {"ext": 0, "2a": 1}
    cblk["_f"] = cblk["family"].map(fam_order)
    cblk = cblk.sort_values(["_f", "arm", "q", "K"], kind="stable").drop(columns="_f")
    p = paired()
    pk = p[(p.comparison == "K_vs_K1") & p.family.isin(["ext", "2a"])].copy()
    pk["_f"] = pk["family"].map(fam_order)
    pk = pk.sort_values(["_f", "a_arm", "q", "a_K"], kind="stable")
    pblk = pd.DataFrame({"block": "paired_K_vs_K1", "family": pk["family"], "arm": pk["a_arm"], "K": pk["a_K"],
                         "q": pk["q"], "metric": pk["metric"], "tier": pk["metric"].map(tier_of),
                         "quantity": "paired difference (K) - (K = 1)", **{c_: pk[c_] for c_ in PAIRED_COLS}})
    tt = U.union_frame([cblk, pblk], ["block", "family", "arm", "K", "q", "metric", "tier", "quantity"])
    # identity: 2a constant K equals extension K (the runs are bit-identical over u1201-u1600)
    e = t[t.family == "ext"].set_index(["q", "seed", "K"])
    k2 = t[(t.family == "2a") & (t.arm == "constant")].set_index(["q", "seed", "K"])
    d_id = max(U.max_abs(k2[m], e.loc[k2.index, m]) for m in STAGE2_13)
    notes = ("Blocks: candidates = median/IQR/min/max over seeds 10501-10510 of the tail-averaged candidates "
             "(candidate_K = pointwise average of the last K weight-export mappings, K = 1 the last iterate; extension "
             "at u1600 with K = 16 covering u1225-u1600; 2a arms at u1600); paired_K_vs_K1 = paired difference "
             f"(K) - (K = 1) per (q, seed) from paired_summary.csv ({BOOT_NOTE}; n_better counts pairs with "
             "K < K=1 for smaller-is-better metrics). Development tier for eta_2 / Delta_2 (state step 4, effort step 1, "
             "GL 16); recovery metrics tier-independent; sigma is not defined for K > 1 (the average is not a Beta). "
             "The 2a constant arm is bit-identical to the extension over u1201-u1600, so its candidates equal the "
             f"extension's for K = 1, 4, 8 (max abs diff {d_id:.3g} over the 13 metrics, 60 per-run rows).")
    tdf = pack.table("T44", tt, status="generated",
                     sources=[C.src(f"{P4}/candidates_all.csv"), C.src(f"{P4}/paired_summary.csv")],
                     script=f"{SCRIPT}:build_t44", notes=notes, tier="development",
                     docs=U.docs_for(tt.columns, "development", {
                         "median": "Median over the 10 seeds of the row's quantity (candidate value or paired difference)",
                         "min": "Minimum over the 10 seeds of the row's quantity",
                         "max": "Maximum over the 10 seeds of the row's quantity",
                         "arm": "Arm: ext = Phase A extension; constant / decay = Pilot 4 section 2a arms"}),
                     caption="Pilot 4 sections 1c.2 (extension, u1600) and 2a (u1600): tail-averaged stage-2 candidates.")
    # cross-check: candidate medians (sections 1c.2 and 2a tables) and the 1c.2 paired table
    med = cblk.pivot_table(index=["family", "arm", "K", "q"], columns="metric", values="median").reset_index()
    pack.crosscheck("T44", med, REP_P4, header_has=["q", "arm", "K", "stage2_peak_rel_err_signed",
                                                   "stage2_peak_locfree_rel_err", "DeltaT_over_dw_off_max"],
                    key_map={"q": "q", "arm": "arm", "K": "K"}, value_map={m: m for m in STAGE2_13 if m in med.columns},
                    label="sections 1c.2 and 2a candidate medians per q, arm, K")
    _check_t44_paired(pack, tdf)


def _check_t44_paired(pack: C.Pack, t: pd.DataFrame) -> None:
    """Section 1c.2 paired K - (K=1) table: 5 metric groups (median, better, CI95) per row, parsed by position."""
    groups = ("stage2_rmse_pos_over_g2_0", "stage2_sym_err_max", "stage2_tail_mean", "eta_T_over_dw",
              "stage2_peak_rel_err_abs")
    pk = t[(t["block"] == "paired_K_vs_K1") & (t["family"] == "ext")].set_index(["q", "K", "metric"])
    ck = U.Checker(pack, "T44", REP_P4, "section 1c.2 paired K - (K=1), extension: median, better, CI95 of 5 metrics")
    for tb in U.find_tables(REP_P4, ["q", "K", "Δ RMSE/e₂*(0)"], "1c.2"):
        ck.tables.add(tb["line"])
        for ln, cells in U.raw_rows(REP_P4, tb):
            if len(cells) != 17:
                ck.miss()
                continue
            q, K = int(cells[0]), int(cells[1])
            for gi, mt in enumerate(groups):
                med, better, ci = cells[2 + 3 * gi: 5 + 3 * gi]
                if (q, K, mt) not in pk.index:
                    ck.miss()
                    continue
                r = pk.loc[(q, K, mt)]
                w = f"line {ln}"
                ck.num(f"{mt} median [q={q}, K={K}]", float(r["median"]), med, w)
                ck.num(f"{mt} n_better [q={q}, K={K}]", float(r["n_better"]), better, w)
                lohi = U.parse_ci(ci)
                if lohi:
                    ck.num(f"{mt} boot_ci95_lo [q={q}, K={K}]", float(r["boot_ci95_lo"]), lohi[0], w)
                    ck.num(f"{mt} boot_ci95_hi [q={q}, K={K}]", float(r["boot_ci95_hi"]), lohi[1], w)
    ck.close()


F19_PANELS = (("stage2_peak_rel_err_signed", "peak error d = 0", "/e₂*(0), signed"),
              ("stage2_peak_locfree_rel_err", "location-free peak error", "/e₂*(0), signed"),
              ("stage2_rmse_pos_over_g2_0", "RMSE |d| < 2q", "/e₂*(0)"),
              ("stage2_tail_mean", "tail mean", "effort (raw)"),
              ("stage2_tail_max", "tail max", "effort (raw)"),
              ("stage2_sym_err_max", "symmetry error", "effort units"),
              ("eta_T_over_dw", "η₂ (dev tier)", "/ΔW"),
              ("DeltaT_over_dw_off_max", "off-path max Δ₂ (dev)", "/ΔW"),
              ("sigma_effort_at_0_t2", "σ₂(0)", "effort units"))
ARM_MARK = {"constant": "o", "decay": "^"}


def build_f19(pack: C.Pack) -> None:
    """F19: Pilot 4 section 2a learning curves (u1225-u1600), median and IQR per arm, q by panel block."""
    c = cand()
    g2a = c[(c.family == "2a") & (c.K == 1)]
    fig, axes = style.new_figure(6, 3, height=10.4, sharex=True)
    parts = []
    for bi, q in enumerate(QS):
        for k, (mt, title, unit) in enumerate(F19_PANELS):
            ax = axes[3 * bi + k // 3, k % 3]
            for arm in ("constant", "decay"):
                g = g2a[(g2a.q == q) & (g2a.arm == arm)]
                out = _band(ax, g, "update", mt, style.ARM_COLORS[arm],
                            style.label_n(style.arm_label(arm), g.seed.nunique()), marker=ARM_MARK[arm], markevery=3)
                parts.append(out.assign(q=q, arm=arm, panel=mt))
            if "signed" in unit:
                ax.axhline(0.0, color=style.REF, lw=0.8)
                parts.append(pd.DataFrame([{"q": q, "panel": mt, "series": "zero line", "y_value": 0.0}]))
            ax.set_title(f"q = {q}: {title}")
            ax.set_ylabel(unit)
    for ax in axes.ravel():
        ax.set_xticks([1300, 1400, 1500, 1600])
    for ax in axes[-1]:
        ax.set_xlabel("global update")
    _fig_legend(fig, axes, "outside upper center")
    data = pd.concat(parts, ignore_index=True)
    data["series"] = data["series"].fillna("median and IQR over seeds")
    data = data[["panel", "series", "q", "arm", "update", "metric", "median", "q25", "q75", "min", "max", "n", "y_value"]]
    # check 1: u1600 median / IQR against gate_distribution_tables.csv (phase A, K = 1)
    gd = S.read_csv(f"{P4}/gate_distribution_tables.csv")
    gd = gd[(gd.phase == "A") & (gd.K == 1)]
    pl = data[(data["update"] == 1600)].astype({"q": int})
    pl = pl.merge(gd, on=["q", "arm", "metric"], suffixes=("", "_gd"), validate="one_to_one")
    d1 = max(U.max_abs(pl["median"], pl["median_gd"]), U.max_abs(pl["q25"], pl["p25"]), U.max_abs(pl["q75"], pl["p75"]))
    # check 2: the constant arm equals the extension curves (bit-identical runs)
    mets = [m for m, _, _ in F19_PANELS if m in curves().columns]
    mm = g2a[g2a.arm == "constant"].merge(curves(), on=["q", "seed", "update"], suffixes=("", "_ext"))
    d2 = max(U.max_abs(mm[m], mm[m + "_ext"]) for m in mets)
    checks = [f"plotted u1600 median/q25/q75 equal {P4}/gate_distribution_tables.csv (phase A, K=1; metrics present "
              f"there: {pl['metric'].nunique()}; max abs diff {d1:.3g})",
              f"constant-arm per-run values equal the extension curves at u1225-u1600 (max abs diff {d2:.3g} over "
              f"{len(mm)} runs x updates and {len(mets)} metrics)",
              "panels of reports/v2/figures/pilot4/curves_2a_q50.png / _q60.png plus off-path Delta_2; same data"]
    cap = ("Pilot 4 section 2a: Phase A continued from the extension's u1200 state to u1600 with a constant LR (yellow, "
           "circles) or a linear LR decay 3e-4 -> 3e-5 over u1201-u1600 (red, triangles). Across-seed median (line) "
           "and IQR (band) of nine stage-2 metrics at every weight export u1225-u1600 (25 updates apart; u1600 is the "
           "last iterate; markers every third export), q = 50 in the top three rows and q = 60 in the bottom three; "
           "n = 10 seeds per arm and q "
           "(10501-10510). The constant arm is bit-identical to the Phase A extension over u1201-u1600, so its curves "
           "repeat F18 over u1225-u1600. eta_2 and Delta_2 development tier; the other panels tier-independent. Source: "
           f"{P4}/candidates_all.csv (family 2a, K = 1, kind curve and tail).")
    pack.figure("F19", fig, U.intify(data), status="regenerated",
                sources=[C.src(f"{P4}/candidates_all.csv"), C.src(f"{P4}/gate_distribution_tables.csv"),
                         C.src(f"{PXA}/curves_weights_every25.csv"),
                         C.src("reports/v2/figures/pilot4/curves_2a_q50.png"),
                         C.src("reports/v2/figures/pilot4/curves_2a_q60.png")],
                script=f"{SCRIPT}:build_f19", caption=cap, checks=checks, tier="development",
                docs=U.docs_for(data.columns, "development"))


# ----------------------------------------------------------------------------------------------
# Section 8
# ----------------------------------------------------------------------------------------------

def _share_runs() -> Tuple[pd.DataFrame, Dict[str, Any], List[Any]]:
    """Per-run smoothed-game values of every study that computed them."""
    if "share_runs" in _CACHE:
        return _CACHE["share_runs"]
    info: Dict[str, Any] = {}
    parts = []
    p1 = S.read_csv(f"{P1}/analysis/smoothed_game/per_run.csv")
    parts.append(pd.DataFrame({"study": "pilot1", "arm": p1["arm"], "q": p1["q"], "seed": p1["seed"], "update": 400,
                               "e_pred_0": p1["e_pred_0"], "e_learned_0": p1["e_learned_0"],
                               "share": p1["share_peak_gap_explained"], "n_nodes": p1["n_nodes_per_beta"],
                               "source": f"{P1}/analysis/smoothed_game/per_run.csv"}))
    info["p1_estar_vs_closed_form"] = max(abs(float(e) - U.g2_star_0(int(q))) for q, e in zip(p1["q"], p1["e_star_0"]))
    cv = curves()
    mk = cv[cv["update"].isin(MARKS)]
    parts.append(pd.DataFrame({"study": "phaseA_ext", "arm": "expected_ext", "q": mk["q"], "seed": mk["seed"],
                               "update": mk["update"], "e_pred_0": mk["e_pred_0"], "e_learned_0": mk["e_learned_0"],
                               "share": mk["share_peak_gap_explained"], "n_nodes": 400,
                               "source": f"{PXA}/curves_weights_every25.csv"}))
    g = S.read_csv(f"{RA}/gates_per_run.csv")
    parts.append(pd.DataFrame({"study": "rehearsal", "arm": "locked v1.0", "q": g["q"], "seed": g["seed"],
                               "update": 1600, "e_pred_0": g["A_smoothed_e_pred_0"],
                               "e_learned_0": g["A_smoothed_e_learned_0"], "share": g["A_smoothed_share_peak_gap_d0"],
                               "n_nodes": 400, "source": f"{RA}/gates_per_run.csv"}))
    srcs = [C.src(f"{P1}/analysis/smoothed_game/per_run.csv"), C.src(f"{PXA}/curves_weights_every25.csv"),
            C.src(f"{RA}/gates_per_run.csv")]
    for st, arm in (("rehearsal_v1_1", "locked v1.1"), ("confirmation", "locked v1.1")):
        rm = S.read_csv(f"{LK}/{st}_analysis/reported_metrics.csv").set_index(["q", "seed"])
        rows, dmax = [], 0.0
        for q, s, _, rd in S.iter_runs(st):
            sm = S.read_json(f"{rd}/gates.json")["reported"]["end_of_A"]["smoothed_game"]
            dmax = max(dmax, abs(float(sm["smoothed_share_peak_gap_d0"]) - float(rm.loc[(q, s), "A_smoothed_share_peak_gap_d0"])))
            rows.append({"study": st, "arm": arm, "q": q, "seed": s, "update": 1600,
                         "e_pred_0": float(sm["smoothed_e_pred_0"]), "e_learned_0": float(sm["smoothed_e_learned_0"]),
                         "share": float(sm["smoothed_share_peak_gap_d0"]), "n_nodes": 400,
                         "source": f"{S.STUDIES[st]['root']}/q*/seed*/gates.json[reported][end_of_A][smoothed_game]"})
        parts.append(pd.DataFrame(rows))
        info[f"{st}_gates_vs_csv"] = dmax
        n = len(S.STUDIES[st]["seeds"]) * 2
        srcs += [C.srcs(f"{S.STUDIES[st]['root']}/q*/seed*/gates.json", f"{st} gates.json", expect=n),
                 C.src(f"{LK}/{st}_analysis/reported_metrics.csv")]
    runs = pd.concat(parts, ignore_index=True)
    # identities and the node count of every implementation
    e400 = runs[(runs.study == "phaseA_ext") & (runs["update"] == 400)].set_index(["q", "seed"])
    p1e = runs[(runs.study == "pilot1") & (runs.arm == "expected")].set_index(["q", "seed"]).loc[e400.index]
    info["ext_u400_vs_pilot1"] = max(U.max_abs(e400[k], p1e[k]) for k in ("e_pred_0", "e_learned_0", "share"))
    r11 = runs[runs.study == "rehearsal_v1_1"].set_index(["q", "seed"])
    d10 = 0.0
    for q, s, _, rd in S.iter_runs("rehearsal"):  # v1.0 gates.json against v1.1 gates.json (JSON to JSON)
        sm = S.read_json(f"{rd}/gates.json")["reported"]["end_of_A"]["smoothed_game"]
        for k, col in (("smoothed_e_pred_0", "e_pred_0"), ("smoothed_e_learned_0", "e_learned_0"),
                       ("smoothed_share_peak_gap_d0", "share")):
            d10 = max(d10, abs(float(sm[k]) - float(r11.loc[(q, s), col])))
    info["v11_vs_v10_rehearsal"] = d10
    srcs.append(C.srcs(f"{S.STUDIES['rehearsal']['root']}/q*/seed*/gates.json", "rehearsal (v1.0) gates.json", expect=20))
    rl = S.read_csv(f"{RF}/rl_u1600.csv").set_index(["q", "seed"])
    e16 = runs[(runs.study == "phaseA_ext") & (runs["update"] == 1600)].set_index(["q", "seed"]).loc[rl.index]
    info["pilot4_1d_vs_ext_u1600"] = max(U.max_abs(rl["e_pred_0"], e16["e_pred_0"]),
                                         U.max_abs(rl["e_learned_0_curves"], e16["e_learned_0"]),
                                         U.max_abs(rl["e2_at_0"], e16["e_learned_0"]))
    code = {"run/run_v2_T2_locked.py": r"^SMOOTH_NODES = (\d+)", "tools/v2/phaseA_ext_analysis.py": r"^N_NODES = (\d+)",
            "tools/v2/pilot1_smoothed_game.py": r"--n-nodes\", type=int, default=(\d+)"}
    nodes = {}
    for rel, pat in code.items():
        mm = re.search(pat, C.abspath(rel).read_text(encoding="utf-8"), flags=re.M)
        nodes[rel] = int(mm.group(1)) if mm else None
        srcs.append(C.src(rel))
    info["nodes"] = nodes
    info["p1_nodes"] = sorted(set(int(x) for x in p1["n_nodes_per_beta"]))
    srcs.append(C.src(f"{RF}/rl_u1600.csv"))
    _CACHE["share_runs"] = (runs, info, srcs)
    return _CACHE["share_runs"]


STUDY_ORDER = {"pilot1": 0, "phaseA_ext": 1, "rehearsal": 2, "rehearsal_v1_1": 3, "confirmation": 4}


def build_t45(pack: C.Pack) -> None:
    """T45: smoothed-game share of the d = 0 peak gap in every study that computed it."""
    runs, info, srcs = _share_runs()
    rows = []
    for (st, arm, q, u), g in runs.groupby(["study", "arm", "q", "update"]):
        e2s = U.g2_star_0(int(q))
        sh = g["share"].to_numpy(float)
        stt = C.median_iqr(sh)
        out = g[(g["share"] > 1) | (g["share"] < 0)].sort_values("seed")
        gp, gl = C.median_iqr(e2s - g["e_pred_0"])["median"], C.median_iqr(e2s - g["e_learned_0"])["median"]
        rows.append({"study": st, "arm": arm, "q": q, "update": u,
                     "checkpoint": f"end of Phase A (u{u})" if st in ("rehearsal", "rehearsal_v1_1", "confirmation")
                     else f"u{u}", "seeds": U.fmt_seeds(g["seed"]), "n": stt["n"], "e2_star_0": e2s,
                     "share_median": stt["median"], "share_q25": stt["q25"], "share_q75": stt["q75"],
                     "share_min": stt["min"], "share_max": stt["max"], "n_share_gt_1": int((sh > 1).sum()),
                     "n_share_lt_0": int((sh < 0).sum()),
                     "runs_share_outside_0_1": "; ".join(f"s{int(s)} ({v:.4g})" for s, v in zip(out["seed"], out["share"])),
                     "e_pred_0_median": C.median_iqr(g["e_pred_0"])["median"],
                     "e_learned_0_median": C.median_iqr(g["e_learned_0"])["median"], "gap_pred_median": gp,
                     "gap_learned_median": gl, "ratio_of_median_gaps": gp / gl,
                     "n_nodes_per_beta": int(g["n_nodes"].iloc[0]), "source": g["source"].iloc[0]})
    t = pd.DataFrame(rows)
    t["_s"] = t["study"].map(STUDY_ORDER)
    t = t.sort_values(["_s", "arm", "q", "update"]).drop(columns="_s").reset_index(drop=True)
    nodes = info["nodes"]
    notes = ("share = (e2*(0) - e_pred(0)) / (e2*(0) - e_learned(0)) of the d = 0 peak gap, e_pred(0) the smoothed-game "
             "prediction (location-shift approximation with the policy's own mean-centred Beta action noise, equal-"
             "probability quadrature). The ratio is ill-conditioned when e_learned(0) approaches e2*(0): a share > 1 "
             "means the learned gap is smaller than the predicted smoothing gap, a share < 0 that e_learned(0) is above "
             "e2*(0); such runs are counted and listed, and the median and ratio_of_median_gaps (= gap_pred_median / "
             "gap_learned_median) are reported instead of means. Node count: SMOOTH_NODES = "
             f"{nodes['run/run_v2_T2_locked.py']} in run/run_v2_T2_locked.py (same value at the lock commits 4bd2214 and "
             f"431474d), N_NODES = {nodes['tools/v2/phaseA_ext_analysis.py']} in tools/v2/phaseA_ext_analysis.py, "
             f"--n-nodes default {nodes['tools/v2/pilot1_smoothed_game.py']} in tools/v2/pilot1_smoothed_game.py and "
             f"n_nodes_per_beta {info['p1_nodes']} in Pilot 1 per_run.csv. Checks: the extension's u400 rows are the "
             f"Pilot-1 expected parents (max abs diff {info['ext_u400_vs_pilot1']:.3g}); the v1.1 re-rehearsal equals the "
             f"v1.0 rehearsal (R1; max abs diff {info['v11_vs_v10_rehearsal']:.3g}); Pilot 4 section 1d (rl_u1600.csv) "
             f"uses the extension's u1600 values, not a separate computation (max abs diff "
             f"{info['pilot4_1d_vs_ext_u1600']:.3g}), so it has no row here; gates.json shares equal reported_metrics.csv "
             f"(max abs diff {max(info['rehearsal_v1_1_gates_vs_csv'], info['confirmation_gates_vs_csv']):.3g}); Pilot 1 "
             f"e_star_0 equals the closed form (max abs diff {info['p1_estar_vs_closed_form']:.3g}). Tier-independent "
             "(direct query of the Beta parameters at d = 0).")
    tdf = pack.table("T45", t, status="generated", sources=srcs, script=f"{SCRIPT}:build_t45", notes=notes,
                     tier="tier-independent", docs=U.docs_for(t.columns, "tier-independent", {
                         "arm": "Arm of the study (Pilot 1 sampled / expected; extension expected_ext; locked protocol "
                                "v1.0 or v1.1)", "update": "Global update of the stage-2 policy (u1600 = end of Phase A)",
                         "n": "Runs behind the row", "source": "File(s) the per-run values were read from"}),
                     caption="Share of the d = 0 stage-2 peak gap predicted by action-noise smoothing, per study, arm, "
                             "q and checkpoint; effort values raw (effort units).")
    _check_t45(pack, tdf, runs)


def _check_t45(pack: C.Pack, t: pd.DataFrame, runs: pd.DataFrame) -> None:
    """Cross-check T45 and its per-run inputs with the reports that show the share."""
    # Pilot 1 section 7 summary table
    ck = U.Checker(pack, "T45", REP_P1, "section 7 summary: median e_pred(0), e_learned(0), share median [min, max]")
    p1 = t[t.study == "pilot1"].set_index(["q", "arm"])
    for tb in U.find_tables(REP_P1, ["q", "arm"], "Summary"):
        ck.tables.add(tb["line"])
        hdr = tb["header"]
        ip = [i for i, h in enumerate(hdr) if h.startswith("median ê_pred")]
        il = [i for i, h in enumerate(hdr) if h.startswith("median ê_learned")]
        ish = [i for i, h in enumerate(hdr) if h.startswith("share of peak gap")]
        for i, row in enumerate(tb["rows"]):
            key = (int(row[0]), row[1])
            if key not in p1.index or not (ip and il and ish):
                ck.miss()
                continue
            r, w = p1.loc[key], f"line {tb['line'] + 2 + i}"
            ck.num(f"e_pred_0 median [q={key[0]}, {key[1]}]", float(r["e_pred_0_median"]), row[ip[0]], w)
            ck.num(f"e_learned_0 median [q={key[0]}, {key[1]}]", float(r["e_learned_0_median"]), row[il[0]], w)
            mci = U.parse_med_ci(row[ish[0]])
            if mci:
                for stt, txt in zip(("share_median", "share_min", "share_max"), mci):
                    ck.num(f"{stt} [q={key[0]}, {key[1]}]", float(r[stt]), txt, w)
    ck.close()
    per1 = runs[runs.study == "pilot1"]
    pack.crosscheck("T45", per1, REP_P1, header_has=["q", "seed", "arm", "ê_learned(0)", "ê_pred(0)"],
                    key_map={"q": "q", "seed": "seed", "arm": "arm"},
                    value_map={"ê_learned(0)": "e_learned_0", "ê_pred(0)": "e_pred_0",
                               "share of peak gap explained": "share"},
                    label="per-run inputs: Pilot 1 section 7 per-run table")
    pack.crosscheck("T45", t[t.study == "pilot1"], REP_SUM, header_has=["q", "arm", "share_peak_gap_explained"],
                    key_map={"q": "q", "arm": "arm"}, value_map={"share_peak_gap_explained": "share_median"},
                    label="summary.md Pilot 1 share medians")
    # extension rows against phaseA_ext.md section 2.1
    ck = U.Checker(pack, "T45", REP_EXT, "section 2.1 share / e_pred / e_learned cells (extension rows)")
    ex = t[t.study == "phaseA_ext"].set_index(["q", "update"])
    colmap = {"share_peak_gap_explained": {"median": "share_median", "q25": "share_q25", "q75": "share_q75",
                                           "min": "share_min", "max": "share_max"},
              "e_pred_0": {"median": "e_pred_0_median"}, "e_learned_0": {"median": "e_learned_0_median"}}
    for tb in C.parse_md_tables(REP_EXT):
        if tb["header"][:1] != ["metric"] or not tb["heading"].startswith("q = "):
            continue
        q = int(tb["heading"].split("=")[1])
        ck.tables.add(tb["line"])
        for i, row in enumerate(tb["rows"]):
            if row[0] not in colmap:
                continue
            for j, u in enumerate(tb["header"][1:], start=1):
                cells = U.parse_mid(row[j])
                for st_, col in colmap[row[0]].items():
                    ck.num(f"{row[0]} {st_} [q={q}, u{u}]", float(ex.loc[(q, int(u)), col]), cells[st_],
                           f"line {tb['line'] + 2 + i}")
    ck.close()
    # locked studies: per-run shares in the lock and confirmation reports, and the lock-report prose
    for st, rep, hh, lab in (("rehearsal", REP_LOCK, "4.4", "lock report 4.4 per-run shares (v1.0 rehearsal)"),
                             ("confirmation", REP_CONF, "4.5", "confirmation report 4.5 per-run shares")):
        df = runs[runs.study == st]
        pack.crosscheck("T45", df, rep, header_has=["q", "seed", "A_smoothed_share_peak_gap_d0"],
                        key_map={"q": "q", "seed": "seed"}, value_map={"A_smoothed_share_peak_gap_d0": "share"},
                        heading_has=hh, label="per-run inputs: " + lab)
    ck = U.Checker(pack, "T45", REP_LOCK, "section 4.4 notes: median share per q, runs with share > 1")
    rr = t[t.study == "rehearsal"].set_index("q")
    ck.num("median share, v1.0 rehearsal [q=50]", float(rr.loc[50, "share_median"]), "0.52", "section 4.4 notes")
    ck.num("median share, v1.0 rehearsal [q=60]", float(rr.loc[60, "share_median"]), "0.55", "section 4.4 notes")
    n1 = int(rr["n_share_gt_1"].sum())
    ck.claim("v1.0 rehearsal runs with share > 1", n1 == 2, n1, "Two runs have a share > 1", "section 4.4 notes")
    ck.close()


T46_FIT_COLS = ("q", "init_seed", "steps", "stop", "final_loss_mse", "best_loss_mse", "best_loss_step",
                "final_over_best_loss", "loss_at_step_280000", "best_loss_up_to_step_280000",
                "best_loss_rel_drop_last_20000", "stage2_peak_rel_err_signed", "stage2_peak_locfree_rel_err",
                "stage2_peak_locfree_argmax_d", "e2_at_0", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean",
                "stage2_tail_max", "stage2_sym_err_max", "eta_T_over_dw", "Gmax_full_over_dw", "max_abs_resid",
                "max_abs_resid_at_d", "tail_min_fit", "tail_max_fit", "rmse_fit_grid", "n_grid")
T46_SUM = ("stage2_peak_rel_err_signed", "stage2_peak_locfree_rel_err", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean",
           "stage2_tail_max", "stage2_sym_err_max", "eta_T_over_dw", "Gmax_full_over_dw")


def _fits() -> pd.DataFrame:
    """fits.csv plus loss-log columns (NPZ loss logs, every 1,000 steps): logged and running-best loss near the cap."""
    def load() -> pd.DataFrame:
        f = S.read_csv(f"{RF}/fits.csv")
        new: Dict[str, List[float]] = {k: [] for k in ("loss_at_step_280000", "best_loss_up_to_step_280000",
                                                        "best_loss_rel_drop_last_20000", "best_loss_step",
                                                        "_last_logged", "_best_logged", "_logged_drop_last_20000",
                                                        "_max_over_final_last_20000")}
        for q, s in zip(f["q"], f["init_seed"]):
            lg = np.load(C.abspath(f"{RF}/fit_q{q}_init{s}.npz"))["loss_log"]
            st, lv = lg[:, 0].astype(int), lg[:, 1].astype(float)
            b280, b = float(lv[st <= 280000].min()), float(lv.min())
            l280 = float(lv[st == 280000][0])
            new["loss_at_step_280000"].append(l280)
            new["best_loss_up_to_step_280000"].append(b280)
            new["best_loss_rel_drop_last_20000"].append((b280 - b) / b280)
            new["best_loss_step"].append(int(st[int(np.argmin(lv))]))
            new["_last_logged"].append(float(lv[-1]))
            new["_best_logged"].append(b)
            new["_logged_drop_last_20000"].append((l280 - float(lv[-1])) / l280)
            new["_max_over_final_last_20000"].append(float(lv[st >= 280000].max()) / float(lv[-1]))
        for k, v in new.items():
            f[k] = v
        f["final_over_best_loss"] = f["final_loss_mse"] / f["best_loss_mse"]
        return f
    return _cached("fits", load)


def build_t46(pack: C.Pack) -> None:
    """T46: representation floor: supervised fit (5 inits x 2 q), RL at u1600, three-way peak-gap comparison."""
    f = _fits()
    rl = S.read_csv(f"{RF}/rl_u1600.csv")
    tw = S.read_csv(f"{RF}/three_way_peak_gap.csv")
    blocks = [f[list(T46_FIT_COLS)].assign(block="supervised_fit", tier="development (eta_2, Gmax_full); "
                                                                       "other columns tier-independent")]
    blocks.append(_summ(f, ["q"], T46_SUM, block="supervised_fit_summary", tier="development (eta_2, Gmax_full); "
                        "other metrics tier-independent").rename(columns={"metric": "quantity"}))
    blocks.append(_summ(rl, ["q"], T46_SUM[:-1] + ("e2_at_0",), block="rl_u1600_summary",
                        tier="development (eta_2); other metrics tier-independent").rename(columns={"metric": "quantity"}))
    twb = tw.assign(block="three_way", tier="tier-independent")
    ratio = []
    for q in QS:
        tq = tw[tw.q == q].set_index("quantity")
        ratio.append({"block": "three_way", "q": q, "quantity": "smoothing_over_rl_ratio_of_medians", "n": 10,
                      "median": tq.loc["smoothing_predicted_gap_d0", "median"] / tq.loc["rl_peak_gap_d0", "median"],
                      "tier": "tier-independent"})
    blocks += [twb, pd.DataFrame(ratio)]
    t = U.union_frame(blocks, ["block", "q", "init_seed", "quantity", "n", "median", "min", "max",
                               "median_over_e2star0", "tier"])
    # recompute the three-way gaps from the per-run values
    re3 = []
    for q in QS:
        r, fi = rl[rl.q == q], f[f.q == q]
        g20 = float(r["g2_at_0"].iloc[0])
        for name, x in (("rl_peak_gap_d0", g20 - r["e2_at_0"]), ("smoothing_predicted_gap_d0", g20 - r["e_pred_0"]),
                        ("supervised_floor_gap_d0", g20 - fi["e2_at_0"]),
                        ("rl_locfree_rel_err", r["stage2_peak_locfree_rel_err"]),
                        ("floor_locfree_rel_err", fi["stage2_peak_locfree_rel_err"])):
            st = C.median_iqr(x)
            re3.append({"q": q, "quantity": name, "median_re": st["median"], "min_re": st["min"], "max_re": st["max"]})
    mm = tw.merge(pd.DataFrame(re3), on=["q", "quantity"])
    d3 = max(U.max_abs(mm[s], mm[s + "_re"]) for s in ("median", "min", "max"))
    d_last = max(U.max_abs(f["final_loss_mse"], f["_last_logged"]), U.max_abs(f["best_loss_mse"], f["_best_logged"]))
    nstop = int((f["stop"] == "max_steps").sum())
    bd = f["best_loss_rel_drop_last_20000"]
    n_hi = int((f["final_over_best_loss"] > 1.01).sum())
    notes = (f"Blocks: supervised_fit = the 10 fits of fits.csv (Pilot 4 section 1d; BetaActor fitted to e2*(d) by full-"
             f"batch Adam, lr 1e-3, uniform least squares on the development D_2 grid). {nstop} of 10 fits stopped at the "
             "300,000-step cap (stop = max_steps) with the best logged loss still falling (best_loss_rel_drop_last_20000 "
             f"from the NPZ loss logs: {bd.min():.3g} to {bd.max():.3g}), so every fit value is an UPPER BOUND on the floor "
             "for this optimizer budget, not a converged floor. The logged loss oscillates strongly (inside the last "
             f"20,000 steps it reaches up to {f['_max_over_final_last_20000'].max():.3g} x the last logged value); the fit "
             "metrics are those of the network at step 300,000, whose loss exceeds the best logged loss in "
             f"{n_hi} fits (final_over_best_loss up to {f['final_over_best_loss'].max():.3g}). Loss-log columns agree with "
             f"fits.csv (last / best logged loss vs final_loss_mse / best_loss_mse: max abs diff {d_last:.3g}). "
             "supervised_fit_summary / rl_u1600_summary = median, min, max "
             "over the 5 inits / 10 RL seeds (extension last iterate at u1600) of the metric named in quantity. "
             "three_way = three_way_peak_gap.csv (d = 0 peak gap e2*(0) - e_hat_2(0) in effort units and /e2*(0); "
             "location-free errors as fractions), recomputed here from rl_u1600.csv and fits.csv (max abs diff "
             f"{d3:.3g}), plus the ratio of the median smoothing-predicted gap to the median RL gap. The smoothing "
             "prediction at u1600 is the extension's (Pilot-1 method, 400 nodes per Beta). eta_2 / Gmax_full of the "
             "fits and the RL rows are development tier.")
    tdf = pack.table("T46", t, status="generated",
                     sources=[C.src(f"{RF}/fits.csv"), C.src(f"{RF}/rl_u1600.csv"), C.src(f"{RF}/three_way_peak_gap.csv"),
                              C.srcs(f"{RF}/fit_q*_init*.npz", "supervised-fit loss logs", expect=10)],
                     script=f"{SCRIPT}:build_t46", notes=notes, tier="development",
                     docs=U.docs_for(t.columns, "development", {
                         "median": "Median over the inits / seeds of the row's quantity",
                         "min": "Minimum over the inits / seeds of the row's quantity",
                         "max": "Maximum over the inits / seeds of the row's quantity",
                         "n": "Inits (fits) or seeds (RL) behind the row",
                         "tier": "Tier of the tier-dependent columns of the row"}),
                     caption="Pilot 4 section 1d: representation floor of the stage-2 actor (diagnostic only).")
    _check_t46(pack, tdf, f)


def _check_t46(pack: C.Pack, t: pd.DataFrame, f: pd.DataFrame) -> None:
    """Cross-check T46 with pilot4_stabilization.md section 1d (tables and prose)."""
    fits = t[t["block"] == "supervised_fit"]
    vals = [c for c in T46_FIT_COLS if c not in ("q", "init_seed", "stop", "best_loss_mse", "best_loss_step",
                                                  "final_over_best_loss", "loss_at_step_280000",
                                                  "best_loss_up_to_step_280000", "best_loss_rel_drop_last_20000",
                                                  "e2_at_0", "rmse_fit_grid", "n_grid")]
    pack.crosscheck("T46", fits, REP_P4, header_has=["q", "init_seed", "steps", "stop", "final_loss_mse"],
                    key_map={"q": "q", "init_seed": "init_seed"}, value_map={c: c for c in vals}, heading_has="1d.",
                    label="section 1d per-init fit table")
    ck = U.Checker(pack, "T46", REP_P4, "section 1d median tables (fits, RL u1600) and the three-way summary cells")
    for block, need_gmax in (("supervised_fit_summary", True), ("rl_u1600_summary", False)):
        sub = t[t["block"] == block].set_index(["q", "quantity"])
        for tb in U.find_tables(REP_P4, ["q", "stage2_peak_rel_err_signed", "stage2_peak_locfree_rel_err"], "1d."):
            if ("Gmax_full_over_dw" in tb["header"]) != need_gmax:
                continue
            ck.tables.add(tb["line"])
            for i, row in enumerate(tb["rows"]):
                q = int(row[0])
                for h, cell in zip(tb["header"][1:], row[1:]):
                    if (q, h) in sub.index:
                        ck.num(f"{block} median {h} [q={q}]", float(sub.loc[(q, h), "median"]), cell,
                               f"line {tb['line'] + 2 + i}")
                    else:
                        ck.miss()
    tw = t[t["block"] == "three_way"].set_index(["q", "quantity"])
    for tb in U.find_tables(REP_P4, ["q", "RL peak gap"], "1d."):
        ck.tables.add(tb["line"])
        for i, row in enumerate(tb["rows"]):
            q = int(row[0])
            for name, cell in zip(("rl_peak_gap_d0", "smoothing_predicted_gap_d0", "supervised_floor_gap_d0"), row[1:]):
                pp = U.parse_lead_paren(cell)
                if not pp:
                    ck.miss()
                    continue
                w = f"line {tb['line'] + 2 + i}"
                ck.num(f"{name} median [q={q}]", float(tw.loc[(q, name), "median"]), pp[0], w)
                ck.num(f"{name} median/e2*(0) [q={q}]", float(tw.loc[(q, name), "median_over_e2star0"]), pp[1], w)
    ck.close()
    pack.crosscheck("T46", t[t["block"] == "three_way"], REP_P4,
                    header_has=["q", "quantity", "n", "median", "min", "max", "median_over_e2star0"],
                    key_map={"q": "q", "quantity": "quantity"},
                    value_map={c: c for c in ("n", "median", "min", "max", "median_over_e2star0")},
                    heading_has="1d.", label="section 1d three-way table")
    # prose of section 1d
    ck = U.Checker(pack, "T46", REP_P4, "section 1d prose (loss still falling, fitted tails, clamp, residual locations)")
    i0 = f[f.init_seed == 0].set_index("q")
    for q, a, b in ((50, "1.38e-3", "1.07e-3"), (60, "8.7e-5", "6.8e-5")):
        ck.num(f"init 0 loss at step 280,000 [q={q}]", float(i0.loc[q, "loss_at_step_280000"]), a, "section 1d")
        ck.num(f"init 0 loss at step 300,000 [q={q}]", float(i0.loc[q, "final_loss_mse"]), b, "section 1d")
    a0, b0 = (float(i0.loc[q, "_logged_drop_last_20000"]) for q in QS)
    ck.claim("init 0 logged-loss decrease over the last 20,000 steps", 0.19 <= min(a0, b0) and max(a0, b0) <= 0.25,
             f"q=50 {a0:.3g}, q=60 {b0:.3g}", "about 22% (init 0)", "section 1d")
    bd = f["best_loss_rel_drop_last_20000"]
    by_q = {q: bd[f.q == q] for q in QS}
    ck.claim("decrease of the running best loss over the last 20,000 steps (all 10 fits)",
             bool(((bd > 0.17) & (bd < 0.27)).all()),
             "; ".join(f"q={q}: {v.min():.3g} to {v.max():.3g}" for q, v in by_q.items()),
             "The loss was still falling, by about 22% over the last 20,000 steps (init 0: ...)", "section 1d",
             "about 22% is init 0's logged loss at 280k vs 300k; the logged loss oscillates (spikes up to "
             f"{f['_max_over_final_last_20000'].max():.3g} x the last value within the last 20,000 steps); the best logged "
             "loss fell in every fit (still falling), by the amounts given here; final loss above the best logged loss in "
             f"{int((f['final_over_best_loss'] > 1.01).sum())} fits (up to {f['final_over_best_loss'].max():.3g} x)")
    lo = float(f["tail_min_fit"].min())
    ck.num("smallest fitted tail mean over the 10 fits (lower end of 'fitted means are 0.001-0.22')", lo, "0.001",
           "section 1d, 'Can the head reach the target?'")
    n_clamp = int((f["tail_min_fit"] < 1.0001e-4).sum())
    ck.claim("fits whose smallest fitted tail mean equals the head floor 100*1e-6 = 1e-4", n_clamp == 0, n_clamp,
             "the clamp is not binding", "section 1d",
             f"{n_clamp} fits (q=50 inits " + ",".join(str(int(s)) for s in f[f['tail_min_fit'] < 1.0001e-4]['init_seed'])
             + ") reach the clamp floor 1e-4 at some tail node; the effect is at most 1e-4 effort units")
    edge = {50: 100.0, 60: 120.0}
    off = f[[abs(abs(d) - edge[q]) > 8 for q, d in zip(f["q"], f["max_abs_resid_at_d"])]]
    ck.claim("largest residual at or near the domain edge (d = +-100 / +-120)", len(off) == 0,
             "; ".join(f"q={q} init {s}: d={d:g}" for q, s, d in zip(off["q"], off["init_seed"], off["max_abs_resid_at_d"])),
             "The largest residuals sit at or near the domain edges, d = ±100 (q=50) and ±120 (q=60)", "section 1d")
    fm = t[t["block"] == "supervised_fit_summary"].set_index(["q", "quantity"])
    apk = f["stage2_peak_rel_err_signed"].abs()
    meds = [abs(float(fm.loc[(q, "stage2_peak_rel_err_signed"), "median"])) for q in QS]
    ck.claim("max over the 10 fits of |peak error at d = 0|", float(apk.max()) <= 0.0006 * 1.0001,
             f"{float(apk.max()):.4g} (per-q medians {meds[0]:.3g}, {meds[1]:.3g})",
             "represent e2* to within 0.06% at the peak", "section 7 (1d)",
             f"per-q medians <= 0.0006: {all(m <= 0.0006 * 1.0001 for m in meds)}; inits above 0.0006: "
             f"{int((apk > 0.0006 * 1.0001).sum())} of 10")
    rm = f["stage2_rmse_pos_over_g2_0"]
    m50 = float(fm.loc[(50, "stage2_rmse_pos_over_g2_0"), "median"])
    ck.claim("max over the 10 fits of RMSE/e2*(0)", float(rm.max()) <= 0.004, f"{float(rm.max()):.4g}",
             "<= 0.004 RMSE/e2*(0)", "section 7 (1d)",
             f"q=50 median {m50:.4g} (rounds to 0.004); inits above 0.004: {int((rm > 0.004).sum())} of 10")
    ck.close()


def build_t47(pack: C.Pack) -> None:
    """T47: cusp diagnostic: steps to |peak error| < 0.05 / 0.03 / 0.01, RMSE at the 0.05 crossing, RL actor steps."""
    summ = S.read_csv(f"{CUSP}/thresholds_summary.csv")
    per = S.read_csv(f"{CUSP}/thresholds_per_init.csv")
    rl = S.read_json(f"{CUSP}/rl_actor_steps.json")
    ident = S.read_csv(f"{CUSP}/identity_vs_pilot4.csv")
    lrc = S.read_csv(f"{RA}/lr_schedule_check.csv")
    cv = S.read_csv(f"{CUSP}/curves_every500.csv")
    # recompute the first crossings from the logged curves
    re_rows = []
    for (q, s), g in cv.sort_values("step").groupby(["q", "init_seed"]):
        r = {"q": q, "init_seed": s}
        for thr in ("0.05", "0.03", "0.01"):
            hit = g[g["peak_rel_err_signed"].abs() < float(thr)]
            r[f"step_abs_peak_lt_{thr}"] = float(hit["step"].iloc[0]) if len(hit) else np.nan
            if thr == "0.05" and len(hit):
                r["rmse_at_peak_lt_0.05"] = float(hit["rmse_pos_over_g2_0"].iloc[0])
                r["locfree_at_peak_lt_0.05"] = float(hit["peak_locfree_rel_err"].iloc[0])
            hl = g[g["peak_locfree_rel_err"].abs() < float(thr)]
            r[f"step_abs_locfree_lt_{thr}"] = float(hl["step"].iloc[0]) if len(hl) else np.nan
        re_rows.append(r)
    rec = pd.DataFrame(re_rows).merge(per, on=["q", "init_seed"], suffixes=("_re", ""))
    qcols = [c for c in per.columns if c not in ("q", "init_seed")]
    d_re = max(U.max_abs(rec[c], rec[c + "_re"]) for c in qcols)
    blocks = [summ.assign(block="first_crossing_summary", source=f"{CUSP}/thresholds_summary.csv")]
    pm = per.melt(id_vars=["q", "init_seed"], var_name="quantity", value_name="value")
    blocks.append(pm.assign(block="first_crossing_per_init", source=f"{CUSP}/thresholds_per_init.csv"))
    rows = []
    for qs, d in rl.items():
        for k, v in d.items():
            rows.append({"block": "rl_actor_steps", "q": int(qs), "quantity": k, "value": v,
                         "source": f"{CUSP}/rl_actor_steps.json (from protocols/v2_T2_locked.json)"})
    for q in QS:
        g = lrc[lrc.q == q]
        for col, want in (("actor_minibatch_steps_A", 32000), ("actor_minibatch_steps_A_last400", 8000)):
            rows.append({"block": "rl_actor_steps_check", "q": q, "quantity": f"rehearsal runs with {col} = {want}",
                         "value": int((g[col] == want).sum()), "n": len(g), "source": f"{RA}/lr_schedule_check.csv"})
    blocks.append(pd.DataFrame(rows))
    im = ident.melt(id_vars=["q", "init_seed"], var_name="quantity", value_name="value")
    blocks.append(im.assign(block="identity_vs_pilot4", source=f"{CUSP}/identity_vs_pilot4.csv"))
    t = U.union_frame(blocks, ["block", "q", "init_seed", "quantity", "value", "n_reached", "median", "min", "max", "n"])
    t["value"] = t["value"].astype(object)
    notes = ("Supervised fit of Pilot 4 section 1d re-run with logging every 500 Adam steps on the recovery grid "
             "(tools/v2/cusp_diagnostic.py). Blocks: first_crossing_summary = thresholds_summary.csv (median/min/max over "
             "5 inits of the first logged step with |peak error at d = 0| < 0.05 / 0.03 / 0.01, the location-free "
             "variants, and RMSE/e2*(0) and the location-free error at the 0.05 crossing); first_crossing_per_init = "
             f"thresholds_per_init.csv in long format (recomputed here from curves_every500.csv: max abs diff {d_re:.3g}); "
             "rl_actor_steps = RL actor (minibatch) steps of the locked Phase A computed from the protocol (512 stage-2 "
             "rows per update, minibatch 256, 10 epochs); rl_actor_steps_check = v1.0 rehearsal runs whose histories "
             "give the same counts; identity_vs_pilot4 = float32 loss logs identical to the Pilot 4 fits at every common "
             "step. Tier-independent (recovery grid).")
    tdf = pack.table("T47", t, status="generated",
                     sources=[C.src(f"{CUSP}/thresholds_summary.csv"), C.src(f"{CUSP}/thresholds_per_init.csv"),
                              C.src(f"{CUSP}/rl_actor_steps.json"), C.src(f"{CUSP}/identity_vs_pilot4.csv"),
                              C.src(f"{CUSP}/curves_every500.csv"), C.src(f"{RA}/lr_schedule_check.csv")],
                     script=f"{SCRIPT}:build_t47", notes=notes, tier="tier-independent",
                     docs=U.docs_for(t.columns, "tier-independent", {
                         "value": "Value of the quantity (steps, fraction of e2*(0), count, or bool)",
                         "median": "Median over the 5 inits", "min": "Minimum over the 5 inits",
                         "max": "Maximum over the 5 inits", "n": "Runs checked (rl_actor_steps_check)",
                         "quantity": "Quantity of the row (column name of the source file)",
                         "source": "File the row was read from"}),
                     caption="Cusp diagnostic (no RL runs): how fast the supervised fit of the stage-2 actor reaches "
                             "the d = 0 peak, against the RL actor-step counts.")
    pack.crosscheck("T47", tdf[tdf["block"] == "first_crossing_summary"], REP_LOCK,
                    header_has=["q", "quantity", "n_reached", "median", "min", "max"],
                    key_map={"q": "q", "quantity": "quantity"},
                    value_map={c: c for c in ("n_reached", "median", "min", "max")}, heading_has="5.",
                    label="section 5 thresholds_summary table")
    ck = U.Checker(pack, "T47", REP_LOCK, "section 5 'median [range]' table, RL actor-step text, identity count")
    sm = summ.set_index(["q", "quantity"])
    cols = (("step_abs_peak_lt_0.05", True), ("step_abs_peak_lt_0.03", True), ("step_abs_peak_lt_0.01", True),
            ("rmse_at_peak_lt_0.05", True), ("locfree_at_peak_lt_0.05", False))
    for tb in U.find_tables(REP_LOCK, ["q", "< 0.05", "< 0.03"], "5."):
        ck.tables.add(tb["line"])
        for i, row in enumerate(tb["rows"]):
            q, w = int(row[0]), f"line {tb['line'] + 2 + i}"
            for (qty, rng), cell in zip(cols, row[1:]):
                r = sm.loc[(q, qty)]
                if rng:
                    mr = U.parse_med_range(cell)
                    if not mr:
                        ck.miss()
                        continue
                    for st_, txt in zip(("median", "min", "max"), mr):
                        ck.num(f"{qty} {st_} [q={q}]", float(r[st_]), txt, w)
                else:
                    ck.num(f"{qty} median [q={q}]", float(r["median"]), cell, w)
    for q in QS:
        d = rl[str(q)]
        for k, cell in (("rows_per_update", "512"), ("minibatch", "256"), ("epochs", "10"),
                        ("actor_steps_per_update", "20"), ("actor_steps_all_A", "32,000"),
                        ("actor_steps_last_400", "8,000")):
            ck.num(f"{k} [q={q}]", float(d[k]), cell, "section 5, RL Phase A actor steps")
    n_id = int(ident["identical_to_pilot4"].astype(bool).sum())
    ck.claim("fits with loss logs identical to Pilot 4", n_id == 10, n_id, "10/10 identical", "section 5 / 6.5")
    ck.close()


def build_f20(pack: C.Pack) -> None:
    """F20: three-way d = 0 peak gap per q (RL at u1600, smoothing-predicted, supervised floor), median and range."""
    rl = S.read_csv(f"{RF}/rl_u1600.csv")
    f = S.read_csv(f"{RF}/fits.csv")
    tw = S.read_csv(f"{RF}/three_way_peak_gap.csv")
    qty = (("rl_peak_gap_d0", "RL policy, u1600\n(n = 10 seeds)"),
           ("smoothing_predicted_gap_d0", "smoothing-predicted\n(n = 10 seeds)"),
           ("supervised_floor_gap_d0", "supervised fit\n(n = 5 inits)"))
    vals = []
    for q in QS:
        r, fi = rl[rl.q == q], f[f.q == q]
        g20 = float(r["g2_at_0"].iloc[0])
        for name, ids, x in (("rl_peak_gap_d0", r["seed"], g20 - r["e2_at_0"]),
                             ("smoothing_predicted_gap_d0", r["seed"], g20 - r["e_pred_0"]),
                             ("supervised_floor_gap_d0", fi["init_seed"], g20 - fi["e2_at_0"])):
            for i_, v in zip(ids, x):
                vals.append({"q": q, "quantity": name, "stat": "run", "init_or_seed": int(i_),
                             "value_effort_units": float(v), "value_over_e2star0": float(v) / g20})
            st = C.median_iqr(x)
            for k in ("median", "min", "max"):
                vals.append({"q": q, "quantity": name, "stat": k, "value_effort_units": st[k],
                             "value_over_e2star0": st[k] / g20})
    data = pd.DataFrame(vals)
    fig, axes = style.new_figure(1, 2, height=3.3, gridspec_kw={"width_ratios": [2.3, 1]})
    off = {50: -0.13, 60: 0.13}
    for pi, (ax, names) in enumerate(((axes[0, 0], [n for n, _ in qty]), (axes[0, 1], ["supervised_floor_gap_d0"]))):
        for xi, name in enumerate(names):
            for q in QS:
                d = data[(data.q == q) & (data.quantity == name)]
                runs_ = d[d.stat == "run"]["value_effort_units"].to_numpy(float)
                med = float(d[d.stat == "median"]["value_effort_units"].iloc[0])
                lo, hi = (float(d[d.stat == k]["value_effort_units"].iloc[0]) for k in ("min", "max"))
                x0 = xi + off[q]
                ax.plot(np.full(runs_.size, x0 + 0.06), runs_, ls="none", marker=style.Q_MARKER[q], ms=2.5,
                        color=style.Q_COLORS[q], alpha=0.45)
                ax.errorbar([x0], [med], yerr=[[med - lo], [hi - med]], fmt=style.Q_MARKER[q], ms=6,
                            color=style.Q_COLORS[q], capsize=3, lw=1.4,
                            label=(f"q = {q} (n = 10 seeds; fit n = 5 inits)" if (pi == 0 and xi == 0) else None))
        ax.axhline(0.0, color=style.REF, lw=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([lab for n, lab in qty if n in names])
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("d = 0 peak gap e₂*(0) − ê₂(0)\n(effort units)")
    axes[0, 1].set_ylabel("effort units")
    axes[0, 0].set_title("all three (linear scale)")
    axes[0, 1].set_title("supervised fit only (zoom)")
    axes[0, 0].legend(loc="upper right")
    st = data[data.stat != "run"].pivot_table(index=["q", "quantity"], columns="stat", values="value_effort_units")
    mm = tw.set_index(["q", "quantity"]).loc[st.index]
    d = max(U.max_abs(st[k], mm[k]) for k in ("median", "min", "max"))
    checks = [f"plotted medians/min/max equal {RF}/three_way_peak_gap.csv (max abs diff {d:.3g} over 18 values)",
              "per-run dots: rl_u1600.csv (g2_at_0 - e2_at_0, g2_at_0 - e_pred_0) and fits.csv (g2_at_0 - e2_at_0)"]
    cap = ("Three-way comparison of the d = 0 stage-2 peak gap e2*(0) - e_hat_2(0), effort units, per q (q = 50 blue "
           "circles, q = 60 orange squares): the RL stage-2 policies of the Phase A extension at u1600 (last iterate, "
           "n = 10 seeds), the gap predicted by action-noise smoothing for the same policies (e2*(0) - e_pred(0), "
           "Pilot-1 method, 400 nodes per Beta; n = 10) and the supervised-fit floor (n = 5 inits; all fits stopped at "
           "the 300,000-step cap, so upper bounds). Large marker: median; bar: min to max; small markers: individual "
           "runs / inits. Right panel: the supervised fit on its own linear scale. Negative values: e_hat_2(0) above "
           "e2*(0). Tier-independent. Sources: "
           f"{RF}/rl_u1600.csv, fits.csv, three_way_peak_gap.csv.")
    pack.figure("F20", fig, U.intify(data), status="generated",
                sources=[C.src(f"{RF}/rl_u1600.csv"), C.src(f"{RF}/fits.csv"), C.src(f"{RF}/three_way_peak_gap.csv")],
                script=f"{SCRIPT}:build_f20", caption=cap, checks=checks, tier="tier-independent",
                docs=U.docs_for(data.columns, "tier-independent",
                                {"quantity": "rl_peak_gap_d0 / smoothing_predicted_gap_d0 / supervised_floor_gap_d0"}))


def build_f21(pack: C.Pack) -> None:
    """F21: supervised-fit trajectories |peak error| and RMSE/e2*(0) against Adam steps, RL step counts marked."""
    cv = S.read_csv(f"{CUSP}/curves_every500.csv").sort_values(["q", "init_seed", "step"])
    rl = S.read_json(f"{CUSP}/rl_actor_steps.json")
    per = S.read_csv(f"{CUSP}/thresholds_per_init.csv")
    cv["abs_peak_rel_err"] = cv["peak_rel_err_signed"].abs()
    fig, axes = style.new_figure(2, 2, height=5.6, sharex=True, sharey="row")
    steps_last, steps_all = int(rl["50"]["actor_steps_last_400"]), int(rl["50"]["actor_steps_all_A"])
    assert all(int(rl[str(q)]["actor_steps_last_400"]) == steps_last for q in QS)
    extra = []
    for ci, q in enumerate(QS):
        for ri, (col, lab) in enumerate((("abs_peak_rel_err", "|peak error at d = 0|\n(fraction of e₂*(0))"),
                                         ("rmse_pos_over_g2_0", "RMSE over |d| < 2q\n(fraction of e₂*(0))"))):
            ax = axes[ri, ci]
            g = cv[cv.q == q]
            for k, (s, gg) in enumerate(g.groupby("init_seed")):
                ax.plot(gg["step"], gg[col], color=style.Q_COLORS[q], lw=style.THIN_W + 0.2, alpha=0.85,
                        label=(style.label_n(f"supervised fit, q = {q}, one line per init", 5) if k == 0 else None))
            ax.set_xscale("log")
            ax.set_yscale("log")
            if col == "abs_peak_rel_err":
                ax.set_ylim(1e-5, 1.5)
            for xv, ls, lab2 in ((steps_last, ":", f"RL actor steps, last 400 updates ({steps_last:,})"),
                                 (steps_all, "-.", f"RL actor steps, all of Phase A ({steps_all:,})")):
                ax.axvline(xv, color=style.REF, ls=ls, lw=1.0, label=lab2)
                extra.append({"q": q, "panel": col, "series": lab2, "x_value": float(xv)})
            if col == "abs_peak_rel_err":
                for thr in (0.05, 0.03, 0.01):
                    ax.axhline(thr, color=style.THRESH, ls="--", lw=0.8,
                               label=("|peak error| = 0.05, 0.03, 0.01" if thr == 0.05 else None))
                    extra.append({"q": q, "panel": col, "series": "threshold", "y_value": thr})
            if ci == 0:
                ax.set_ylabel(lab)
            if ri == 0:
                ax.set_title(f"q = {q}")
            if ri == 1:
                ax.set_xlabel("Adam steps of the supervised fit (log scale)")
    _fig_legend(fig, axes, "outside lower center", first="supervised fit")
    n_clip = int((cv["abs_peak_rel_err"] < 1e-5).sum())
    data = pd.concat([cv[["q", "init_seed", "step", "abs_peak_rel_err", "rmse_pos_over_g2_0", "peak_rel_err_signed"]]
                      .assign(series="supervised fit (one line per init)"), pd.DataFrame(extra)], ignore_index=True)
    rec = []
    for (q, s), gg in cv.groupby(["q", "init_seed"]):
        h = gg[gg["abs_peak_rel_err"] < 0.05]
        rec.append({"q": q, "init_seed": s, "re": float(h["step"].iloc[0])})
    rr = pd.DataFrame(rec).merge(per, on=["q", "init_seed"])
    d = U.max_abs(rr["re"], rr["step_abs_peak_lt_0.05"])
    checks = [f"first logged step with |peak error| < 0.05 read from the plotted curves equals {CUSP}/thresholds_per_init.csv "
              f"(max abs diff {d:.3g}, 10 inits)",
              f"RL step marks from {CUSP}/rl_actor_steps.json ({steps_last} and {steps_all}, both q)",
              f"top-row y axis clipped at 1e-5: {n_clip} of {len(cv)} logged |peak error| values lie below (all in the "
              "data CSV)"]
    cap = ("Supervised fit of the stage-2 actor to e2*(d) (Pilot 4 section 1d set-up, re-run with logging): |peak error "
           "at d = 0| (top) and RMSE over |d| < 2q divided by e2*(0) (bottom) on the recovery grid against Adam steps, "
           "both axes logarithmic, logged every 500 steps up to the 300,000-step cap; 5 inits per q (thin lines; left "
           "q = 50, right q = 60). Vertical lines: RL actor steps of the locked Phase A (20 per update: 8,000 over the "
           "last 400 updates, dotted; 32,000 over all 1,600 updates, dash-dot). Dashed horizontal lines (top): "
           f"|peak error| = 0.05, 0.03, 0.01. The top-row y axis is cut at 1e-5 ({n_clip} of {len(cv)} logged values lie "
           "below it, where the signed error passes through 0; all values are in the data CSV). Tier-independent. "
           f"Sources: {CUSP}/curves_every500.csv and rl_actor_steps.json.")
    pack.figure("F21", fig, U.intify(data), status="generated",
                sources=[C.src(f"{CUSP}/curves_every500.csv"), C.src(f"{CUSP}/rl_actor_steps.json"),
                         C.src(f"{CUSP}/thresholds_per_init.csv")],
                script=f"{SCRIPT}:build_f21", caption=cap, checks=checks, tier="tier-independent",
                docs=U.docs_for(data.columns, "tier-independent"))


# ----------------------------------------------------------------------------------------------
# Section 10
# ----------------------------------------------------------------------------------------------

def build_t52(pack: C.Pack) -> None:
    """T52: Pilot 4 section 6 distribution tables of the end-of-phase candidates (existing CSV as is)."""
    rel = f"{P4}/gate_distribution_tables.csv"
    df = pack.found_table("T52", rel, script=f"{SCRIPT}:build_t52",
                          notes="Quantiles over seeds 10501-10510 (pandas quantile = numpy linear) of each end-of-phase "
                                "candidate: phase A = Pilot 4 section 2a at u1600 (constant, decay) x K in {1, 4, 8}; "
                                "phase B = section 2b at u2200 x K in {1, 4, 8, 12} (stage 2 frozen at the u1600 parent, "
                                "so its stage-2 rows repeat across arms and K). Development tier for eta_2, Gmax_full, "
                                "EXP_root; recovery metrics tier-independent.",
                          tier="development", md_max_rows=None,
                          docs=U.docs_for(["phase", "arm", "K"], "development", {
                              "phase": "End of phase of the candidate: A = Pilot 4 section 2a at u1600 (stage 1 untrained, "
                                       "full-policy metrics omitted); B = section 2b at u2200",
                              "arm": "constant = constant LR; decay = linear LR decay 3e-4 -> 3e-5 over the phase "
                                     "(2b arms B2_mean_constant / B2_mean_decay)",
                              "n": "Seeds behind the row (10)"}),
                          caption="Pilot 4 section 6: distribution tables for setting gates (no threshold proposals).")
    for i, ph in enumerate(("A", "B")):
        pack.crosscheck("T52", df[df.phase == ph], REP_P4, header_has=["arm", "K", "q", "metric", "min", "p10", "p90"],
                        key_map={"arm": "arm", "K": "K", "q": "q", "metric": "metric"},
                        value_map={c: c for c in ("min", "p10", "p25", "median", "p75", "p90", "max")},
                        heading_has="6. Distribution tables", tables=[i], label=f"section 6, phase {ph} table")


T53_COLS = ("q", "seed", "eta_T_over_dw_final", "eta_T_over_dw_dev", "stage2_rmse_pos_over_g2_0_final",
            "stage2_tail_mean_over_g2_0_final", "G-A", "Gmax_full_over_dw_final", "Gmax_full_over_dw_dev",
            "stage1_rel_err_abs_final", "G-F", "run_pass", "outcome", "G-A_dev", "G-F_dev")
T53_CRIT = (("eta_T_over_dw_pass", "eta_T_over_dw_final", "eta_T_over_dw"),
            ("stage2_rmse_pos_over_g2_0_pass", "stage2_rmse_pos_over_g2_0_final", "stage2_rmse_pos_over_g2_0"),
            ("stage2_tail_mean_over_g2_0_pass", "stage2_tail_mean_over_g2_0_final", "stage2_tail_mean_over_g2_0"),
            ("Gmax_full_over_dw_pass", "Gmax_full_over_dw_final", "Gmax_full_over_dw"),
            ("stage1_rel_err_abs_pass", "stage1_rel_err_abs_final", "stage1_rel_err_abs"))


def build_t53(pack: C.Pack) -> None:
    """T53: v1.0 rehearsal: gates per run (20 rows), pass counts, failing criteria, thresholds."""
    g = S.read_csv(f"{RA}/gates_per_run.csv")
    pc = S.read_csv(f"{RA}/pass_counts.csv")
    proto = S.read_json("protocols/v2_T2_locked.json")
    thr = {c["metric"]: (gate, c["op"], float(c["threshold"])) for gate in ("G-A", "G-F")
           for c in proto["gates"][gate]["all_must_hold"]}
    per = g[list(T53_COLS)].copy()
    for pcol, vcol, _ in T53_CRIT:
        per[pcol] = g[pcol]
    fails = []
    for _, r in g.iterrows():
        fl = [f"{m} = {r[v]:.4g} (needs {thr[m][1]} {thr[m][2]:g})" for p_, v, m in T53_CRIT if not bool(r[p_])]
        fails.append("; ".join(fl) or "none")
    per["failing_criteria"] = fails
    for c in ("B_stage1_rel_err_signed", "B_Gmax_full_t", "B_Gmax_full_d", "commit", "clean_tree", "protocol_sha256",
              "wall_sec"):
        per[c] = g[c]
    blocks = [per.assign(block="gates_per_run")]
    cnt = pc.rename(columns={"run_pass": "n_run_pass"})
    for p_, _, m in T53_CRIT:
        cnt[f"n_fail_{m}"] = [int((~g.loc[g.q == q, p_].astype(bool)).sum()) for q in cnt["q"]]
    blocks.append(cnt.assign(block="pass_counts"))
    blocks.append(pd.DataFrame([{"block": "thresholds", "gate": gate, "metric": m, "op": op, "threshold": v}
                                for m, (gate, op, v) in thr.items()]))
    t = U.union_frame(blocks, ["block", "q", "seed"])
    # checks: counts recomputed from the per-run verdicts; protocol hash
    rc = g.groupby("q").agg(G_A_pass=("G-A", "sum"), G_F_pass=("G-F", "sum"), run_pass=("run_pass", "sum"),
                            G_A_pass_dev=("G-A_dev", "sum"), G_F_pass_dev=("G-F_dev", "sum")).reset_index()
    cnt_ok = bool((rc.set_index("q")[["G_A_pass", "G_F_pass", "run_pass", "G_A_pass_dev", "G_F_pass_dev"]] ==
                   pc.set_index("q")[["G_A_pass", "G_F_pass", "run_pass", "G_A_pass_dev", "G_F_pass_dev"]]).all().all())
    psha = C.src("protocols/v2_T2_locked.json").sha256
    sha_ok = bool((g["protocol_sha256"].astype(str).map(lambda s: psha.startswith(s))).all())
    same_tier = bool(((g["G-A"] == g["G-A_dev"]) & (g["G-F"] == g["G-F_dev"])).all())
    notes = ("Blocks: gates_per_run = the 20 v1.0 rehearsal runs (seeds 10501-10510 per q; commit 5b07293, clean tree, "
             "protocol protocols/v2_T2_locked.json) with the gate values (G-A at the end of Phase A, G-F at the end of "
             "Phase B; final tier, dev-tier verdicts in G-A_dev / G-F_dev), the per-criterion pass flags and the failing "
             "criteria; pass_counts = pass_counts.csv plus failures per criterion (counted from the per-run flags; the "
             f"counts recomputed from the per-run verdicts equal pass_counts.csv: {cnt_ok}); thresholds = the v1.0 gates "
             f"of the protocol JSON (its SHA-256 starts with the runs' protocol_sha256: {sha_ok}). Dev and final tier give "
             f"the same verdict in every run: {same_tier}. Run pass = G-A and G-F (v1.0 rule); the three failures fail "
             "only |stage-1 error| <= 0.10, the criterion that v1.1 moved to the secondary S1.")
    tdf = pack.table("T53", t, status="generated",
                     sources=[C.src(f"{RA}/gates_per_run.csv"), C.src(f"{RA}/pass_counts.csv"),
                              C.src("protocols/v2_T2_locked.json")],
                     script=f"{SCRIPT}:build_t53", notes=notes, tier="final and development",
                     docs=U.docs_for(t.columns, "final", {
                         "metric": "Metric of the gate criterion (thresholds block)",
                         "n": "Runs per q", "wall_sec": "Wall-clock seconds of the whole locked run"}),
                     caption="v1.0 dress rehearsal on the development seeds: gates per run and pass counts.")
    _check_t53(pack, tdf, g)


def _check_t53(pack: C.Pack, t: pd.DataFrame, g: pd.DataFrame) -> None:
    """Cross-check T53 with protocol_lock_and_rehearsal.md section 4.3 (tables, booleans, prose)."""
    per = t[t["block"] == "gates_per_run"]
    numc = ("eta_T_over_dw_final", "eta_T_over_dw_dev", "stage2_rmse_pos_over_g2_0_final",
            "stage2_tail_mean_over_g2_0_final", "Gmax_full_over_dw_final", "Gmax_full_over_dw_dev",
            "stage1_rel_err_abs_final")
    pack.crosscheck("T53", per, REP_LOCK, header_has=["q", "seed", "eta_T_over_dw_final", "G-A", "G-F", "run_pass"],
                    key_map={"q": "q", "seed": "seed"}, value_map={c: c for c in numc}, heading_has="4.3",
                    label="section 4.3 per-run gate values")
    pack.crosscheck("T53", t[t["block"] == "pass_counts"], REP_LOCK,
                    header_has=["q", "n", "G_A_pass", "G_F_pass", "run_pass"], key_map={"q": "q"},
                    value_map={"n": "n", "G_A_pass": "G_A_pass", "G_F_pass": "G_F_pass", "run_pass": "n_run_pass",
                               "G_A_pass_dev": "G_A_pass_dev", "G_F_pass_dev": "G_F_pass_dev"},
                    heading_has="4.3", label="section 4.3 pass counts")
    ck = U.Checker(pack, "T53", REP_LOCK, "section 4.3 yes/no verdicts and outcome per run; prose on the 3 failures")
    pi = per.set_index(["q", "seed"])
    for tb in U.find_tables(REP_LOCK, ["q", "seed", "eta_T_over_dw_final"], "4.3"):
        ck.tables.add(tb["line"])
        for i, row in enumerate(tb["rows"]):
            rec = dict(zip(tb["header"], row))
            key = (int(rec["q"]), int(rec["seed"]))
            w = f"line {tb['line'] + 2 + i}"
            for c in ("G-A", "G-F", "run_pass", "G-A_dev", "G-F_dev"):
                ck.boolean(f"{c} [q={key[0]}, s{key[1]}]", bool(pi.loc[key, c]), rec[c], w)
            ck.text(f"outcome [q={key[0]}, s{key[1]}]", pi.loc[key, "outcome"], rec["outcome"], w)
    nga = int(g["G-A"].astype(bool).sum())
    ck.claim("runs passing G-A", nga == 20, nga, "G-A passes in 20/20", "section 4.3")
    fl = g[~g["run_pass"].astype(bool)].sort_values(["q", "seed"])
    who = ", ".join(f"q={q} s{s}" for q, s in zip(fl["q"], fl["seed"]))
    ck.claim("failing runs", who == "q=50 s10507, q=60 s10505, q=60 s10510", who,
             "q=50 seed 10507; q=60 seeds 10505 and 10510", "section 4.3")
    only_s1 = bool((~fl["stage1_rel_err_abs_pass"].astype(bool)).all() and fl["Gmax_full_over_dw_pass"].astype(bool).all()
                   and fl["G-A"].astype(bool).all())
    ck.claim("all failures fail only |stage-1 error| > 0.10", only_s1, only_s1, "All three fail on |stage-1 error| > 0.10",
             "section 4.3")
    for v, cell in zip(fl["stage1_rel_err_abs_final"], ("0.103", "0.126", "0.113")):
        ck.num("|stage-1 error| of a failing run", float(v), cell, "section 4.3")
    gm = fl["Gmax_full_over_dw_final"]
    ck.num("min Gmax_full/dW of the failing runs", float(gm.min()), "0.0021", "section 4.3")
    ck.num("max Gmax_full/dW of the failing runs", float(gm.max()), "0.0023", "section 4.3")
    loc = sorted(set(zip(fl["B_Gmax_full_t"].astype(int), fl["B_Gmax_full_d"].astype(float))))
    ck.claim("location (t*, d*) of Gmax_full in the failing runs", loc == [(1, 0.0)], str(loc), "at (t=1, d=0)",
             "section 4.3")
    same = bool(((g["G-A"] == g["G-A_dev"]) & (g["G-F"] == g["G-F_dev"])).all())
    ck.claim("dev tier gives the same verdict as the final tier", same, same, "in every run", "section 4.3")
    ck.close()


# ----------------------------------------------------------------------------------------------
# per-run tables of Pilot 4
# ----------------------------------------------------------------------------------------------

def build_d05(pack: C.Pack) -> None:
    """D05: per-run table of Pilot 4 section 2a at u1600 (last iterate, K = 1) with the run records (40 rows)."""
    per = _p2a_runs()
    rr = run_records()
    rr = rr[rr.family == "2a"]
    misattributed = ["n_last5", "within_run_sd_e1_last5", "within_run_range_e1_last5"]
    rb = run_records()
    rb = rb[rb.family == "2b"].set_index(["q", "seed", "arm"])
    ra = rr.set_index(["q", "seed", "arm"])
    d_mis = max(U.max_abs(ra[c], rb.loc[ra.index, c]) for c in misattributed)
    rr = rr.drop(columns=misattributed + ["family"])
    rr = rr.loc[:, rr.notna().any()]
    per = per.loc[:, per.notna().any()]
    df = per.merge(rr, on=["q", "seed", "arm"], how="left", validate="one_to_one")
    first = ["family", "q", "seed", "arm", "run_dir", "K", "update", "kind"]
    df = df[first + [c for c in df.columns if c not in first]].sort_values(["q", "seed", "arm"]).reset_index(drop=True)
    dropped_c = sorted(set(cand().columns) - set(per.columns))
    notes = ("One row per 2a run (q x seed 10501-10510 x arm constant / decay) at u1600: the last-iterate candidate "
             "(candidates_all.csv family 2a, kind tail, K = 1; development-tier re-evaluation of the u1600 export; "
             "original column names) merged with run_records.csv (family 2a) and the final-tier values of "
             "final_v2.json['final'] (final_tier__*). Dropped: candidates_all columns that are empty for 2a (stage 1 is "
             f"untrained in Phase A: {', '.join(dropped_c)}), empty run-record columns, and the run-record columns "
             f"{', '.join(misattributed)}: in run_records.csv the 2a rows carry the 2b values (tools/v2/pilot4_analysis.py "
             "merges them on (q, seed, arm) without the family; 2a vs 2b max abs diff "
             f"{d_mis:.3g}). sigma_effort_at_0_t1 is the untrained stage-1 output of the shared actor.")
    out = pack.data("D05", df, status="generated",
                    sources=[C.src(f"{P4}/candidates_all.csv"), C.src(f"{P4}/run_records.csv"),
                             C.srcs(f"{S.STUDIES['pilot4_A']['root']}/q*/seed*/*/final_v2.json", "2a final_v2.json",
                                    expect=40)],
                    script=f"{SCRIPT}:build_d05", notes=notes, tier="final and development",
                    docs=U.docs_for(df.columns, "development", {
                        "sigma_effort_at_0_t1": "sigma_1(0) of the shared actor's stage-1 output; stage 1 is untrained "
                                                "in Phase A, so this is not evidence about stage 1"}))
    cols = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_peak_locfree_rel_err",
            "stage2_rmse_pos_over_g2_0", "stage2_tail_mean", "stage2_tail_max", "stage2_tail_mean_over_g2_0",
            "stage2_tail_max_over_g2_0", "stage2_sym_err_max", "eta_T_over_dw", "DeltaT_over_dw_on_max",
            "DeltaT_over_dw_off_max", "sigma_effort_at_0_t2")
    pack.crosscheck("D05", out, REP_P4, header_has=["q", "seed", "arm", "stage2_peak_rel_err_signed", "sigma_effort_at_0_t2"],
                    key_map={"q": "q", "seed": "seed", "arm": "arm"}, value_map={c: c for c in cols},
                    heading_has="2a.", label="section 2a per-run final metrics (u1600)")


def build_d06(pack: C.Pack) -> None:
    """D06: per-run table of Pilot 4 section 2b at u2200 (last iterate, K = 1) with the run records (40 rows)."""
    c = cand()
    t = c[(c.family == "2b") & (c.kind == "tail") & (c.K == 1)].copy()
    t = t.loc[:, t.notna().any()]
    ft = S.final_tier_columns("pilot4_B")
    ft["arm"] = ft["arm"].map({"B2_mean_constant": "constant", "B2_mean_decay": "decay"})
    keep = ["q", "seed", "arm", "run_dir"] + [f"final_tier__{k}" for k in FT_KEYS_B]
    rr = run_records()
    rr = rr[rr.family == "2b"].drop(columns=["family"])
    df = t.merge(ft[keep], on=["q", "seed", "arm"], how="left", validate="one_to_one")
    df = df.merge(rr, on=["q", "seed", "arm"], how="left", validate="one_to_one")
    first = ["family", "q", "seed", "arm", "run_dir", "K", "update", "kind"]
    df = df[first + [x for x in df.columns if x not in first]].sort_values(["q", "seed", "arm"]).reset_index(drop=True)
    dr = S.read_json(f"{LK}/consolidation/dirty_rerun_compare.json")
    n_dirty = int(df["dirty"].astype(bool).sum())
    notes = ("One row per 2b run (q x seed 10501-10510 x arm constant = B2_mean_constant / decay = B2_mean_decay) at "
             "u2200: the last-iterate candidate (candidates_all.csv family 2b, kind tail, K = 1: live stage 1 at u2200 "
             "with the frozen u1600 stage-2 parent; development tier; band decomposition against parent4_bands.csv; "
             "empty columns dropped), the final-tier values of final_v2.json['final'] (final_tier__*), and "
             f"run_records.csv (family 2b). dirty = True in {n_dirty} runs (q = 60, seeds 10507-10510, both arms; untracked "
             "analysis scripts at launch); the clean re-run of q = 60 seed 10507 B2_mean_constant is bit-identical "
             f"(dirty_rerun_compare.json ALL_IDENTICAL = {dr.get('ALL_IDENTICAL')}).")
    out = pack.data("D06", df, status="generated",
                    sources=[C.src(f"{P4}/candidates_all.csv"), C.src(f"{P4}/run_records.csv"),
                             C.src(f"{LK}/consolidation/dirty_rerun_compare.json"),
                             C.srcs(f"{S.STUDIES['pilot4_B']['root']}/q*/seed*/*/final_v2.json", "2b final_v2.json",
                                    expect=40)],
                    script=f"{SCRIPT}:build_d06", notes=notes, tier="final and development",
                    docs=U.docs_for(df.columns, "development"))
    cols = ("e1_cand", "stage1_rel_err_signed", "learning_rel", "inherited_rel", "sigma_effort_at_0_t1",
            "within_run_sd_e1_last5", "within_run_range_e1_last5", "Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d",
            "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw", "kl_final_epoch",
            "clip_frac", "phase_wall_sec", "would_fire_update")
    pack.crosscheck("D06", out, REP_P4, header_has=["q", "seed", "arm", "e1_cand", "learning_band"],
                    key_map={"q": "q", "seed": "seed", "arm": "arm"}, value_map={x: x for x in cols},
                    heading_has="2b.", label="section 2b per-run table (u2200)")


# ----------------------------------------------------------------------------------------------

def build() -> None:
    """Build every item of the module (deterministic; replaces the module's previous outputs)."""
    _CACHE.clear()
    pack = C.Pack("sec_ext")
    build_t42(pack)
    build_f18(pack)
    build_t43(pack)
    build_t44(pack)
    build_f19(pack)
    build_t45(pack)
    build_t46(pack)
    build_t47(pack)
    build_f20(pack)
    build_f21(pack)
    build_t52(pack)
    build_t53(pack)
    build_d04(pack)
    build_d05(pack)
    build_d06(pack)
    pack.save_fragment()
