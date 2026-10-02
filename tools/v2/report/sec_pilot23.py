"""Pack items of Part I section 5, Pilots 2 and 3 (module ``sec_pilot23``).

* Pilot 2: joint stage 2 (``A_joint``) vs frozen stage 2 with all-rows (``B1_frozen_allnorm``) or
  stage-1-rows (``B2_frozen_s1norm``) advantage normalization; Phase B, global u401-u1000, from the
  Pilot 1 ``expected`` end-of-Phase-A states (u400).
* Pilot 3: frozen B2 with stochastic (``B2_frozen_s1norm``) vs mean (``B2_frozen_s1norm_mean``)
  stage-2 continuation; same parents and budget.

Items: T22-T25, F06-F08 (Pilot 2); T26-T29, F09, F10 (Pilot 3); D02, D03 (per-run tables).

No training and no re-evaluation: every value is read from saved records. ``final_table.csv`` holds
the last training-time checkpoint (u1000, development tier); ``final_v2.json['final']`` the same
policy on the final tier; ``final_development.npz`` the tier-independent recovery arrays;
``decomposition_residual_band.csv`` the revised stage-1 decomposition (residual band, final tier).
"""

from __future__ import annotations

import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import common as C
import dictionary as DICT
import studies as S
import style

sys.path.insert(0, str(C.REPO))
from utils.theory_multistage import g2_two_stage  # noqa: E402  (closed form, numpy only)

MOD = "sec_pilot23.py"
P1 = "results/v2_pilots/pilot1"
P2 = "results/v2_pilots/pilot2"
P2A = f"{P2}/analysis"
P3 = "results/v2_pilots/pilot3"
P3A = f"{P3}/analysis"
REP2 = "reports/v2/pilot2_freeze.md"
REP3 = "reports/v2/pilot3_continuation_mode.md"
SUMM = "reports/v2/summary.md"
QS = (50, 60)
SEEDS = tuple(range(10501, 10511))
SEEDS_TXT = "10501-10510"
ARMS2 = ("A_joint", "B1_frozen_allnorm", "B2_frozen_s1norm")
SHORT2 = {"A_joint": "A", "B1_frozen_allnorm": "B1", "B2_frozen_s1norm": "B2"}
COMPARE2 = (("B1_frozen_allnorm", "A_joint"), ("B2_frozen_s1norm", "A_joint"),
            ("B2_frozen_s1norm", "B1_frozen_allnorm"))
ARMS3 = ("B2_frozen_s1norm", "B2_frozen_s1norm_mean")
SHORT3 = {"B2_frozen_s1norm": "stochastic", "B2_frozen_s1norm_mean": "mean"}
LABEL3 = {"B2_frozen_s1norm": "stochastic", "B2_frozen_s1norm_mean": "mean"}   # style label keys
MARK = {"A_joint": "o", "B1_frozen_allnorm": "s", "B2_frozen_s1norm": "^",
        "B2_frozen_s1norm_mean": "o"}
N_BOOT, BOOT_SEED = 10000, 20261001
LAST5 = (900, 925, 950, 975, 1000)
DEV, FIN, IND, NA = "development", "final", "tier-independent", "n/a"
TRAIN = "none (training statistic)"   # table cell label; a bare "n/a" reads back as NaN
FT = "final_tier__"
SUPERSEDED2 = ("stage1_learning_err_rel", "stage1_learning_err_rel_abs", "stage1_inherited_err_rel",
               "stage1_inherited_err_rel_abs")
DEV_METRICS = ("Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw",
               "dFull_over_dw", "eta_T_over_dw", "DeltaT_over_dw_on_max",
               "DeltaT_over_dw_on_mean_cellmass_weighted", "DeltaT_over_dw_off_max", "Gmax_full_t",
               "Gmax_full_d") + SUPERSEDED2
TIER_TXT = ("development = state step 4, effort step 1, GL 16 per half; final = state step 2, "
            "effort step 0.5, GL 32 per half; tier-independent = recovery grid (step 0.5) or "
            "direct policy query")
DECOMP_TXT = ("revised stage-1 decomposition (residual-band method, final tier; pilot2_freeze.md "
              "section 6), rows source == weights, update == 1000 of "
              "decomposition_residual_band.csv")


def _sc(fn: str) -> str:
    """Script reference of a builder function."""
    return f"{MOD}:{fn}"


def _csv(rel: str, **kw: Any) -> pd.DataFrame:
    """CSV under the repo, parsed with exact float round-tripping (pandas 3 default drops digits)."""
    return S.read_csv(rel, float_precision="round_trip", **kw)


# ----------------------------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------------------------

def _clean_own_outputs(pack: C.Pack) -> None:
    """Remove stale files of this module's items (exact id prefixes) so a rebuild replaces them."""
    ids = ("T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29", "F06", "F07", "F08", "F09",
           "F10", "D02", "D03")
    for sub in ("tables", "figures", "data", "provenance"):
        for p in sorted((pack.dir / sub).glob("*")):
            if p.is_file() and any(p.name.startswith(i + "_") for i in ids):
                p.unlink()


def _lookup_docs(cols: Sequence[str], tier: str) -> Dict[str, Dict[str, str]]:
    """Global-dictionary entries of the covered columns (``tier`` = table tier)."""
    out = {}
    for c in cols:
        d = DICT.lookup(c, tier)
        if d:
            out[c] = dict(d)
    return out


def _final_tier_docs(cols: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Docs of ``final_tier__*`` columns built from the dictionary entry of the inner name."""
    out: Dict[str, Dict[str, str]] = {}
    for c in cols:
        if not c.startswith(FT):
            continue
        inner = c[len(FT):]
        base = DICT.lookup(inner, FIN) or {}
        dfn = base.get("definition", "")
        if inner in ("DeltaT_over_dw_on_mass", "DeltaT_over_dw_off_mass"):
            side = "on" if "_on_" in inner else "off"
            dfn = (f"Sum over the {side}-path stage-2 nodes of the normalized exact cell masses "
                   "(on path = open set |d - drift| < 2q; nodes at exactly 2q are off path)")
        out[c] = dict(definition="Final tier (state step 2, effort step 0.5, GL 32 per half), "
                                 "last checkpoint (global u1000), from "
                                 "final_v2.json['final']: " + dfn,
                      units=base.get("units", ""), normalization=base.get("normalization", ""),
                      tier=FIN,
                      source=base.get("source", "")
                      + "; read with tools/v2/report/studies.py:final_tier_columns")
    return out


def _stats(values: Sequence[float]) -> Dict[str, float]:
    """Median, q25, q75, min, max, n (numpy linear interpolation)."""
    return C.median_iqr(values)


def _summary_rows(df: pd.DataFrame, by: Sequence[str],
                  specs: Sequence[Tuple[str, str, str, str]]) -> List[Dict]:
    """Across-seed summaries; specs are ``(metric, column, tier, kind)``, kind 'num'/'count'."""
    rows: List[Dict] = []
    for key, g in df.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        for name, col, tier, kind in specs:
            base = {**dict(zip(by, key)), "metric": name, "tier": tier}
            if kind == "count":
                v = g[col].astype(bool).to_numpy()
                rows.append({**base, "median": np.nan, "q25": np.nan, "q75": np.nan, "min": np.nan,
                             "max": np.nan,
                             "n": int(v.size), "n_true": int(v.sum())})
            else:
                rows.append({**base, **_stats(g[col].to_numpy(dtype=float)), "n_true": np.nan})
    return rows


def _argmax_counts(t: Sequence[int], d: Sequence[float]) -> str:
    """'(t=2, d=-4) x9; (t=1, d=0) x1' sorted by count, then t, then d."""
    pairs: Dict[Tuple[int, float], int] = {}
    for ti, di in zip(t, d):
        k = (int(ti), float(di))
        pairs[k] = pairs.get(k, 0) + 1
    items = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return "; ".join(f"(t={k[0]}, d={k[1]:g}) x{n}" for k, n in items)


def _paired_row(dv: np.ndarray, rng: np.random.Generator, lower: bool) -> Dict[str, Any]:
    """Paired statistics as in tools/v2/pilot2_analysis.py:paired (percentile bootstrap)."""
    dv = np.asarray(dv, dtype=float)
    bm = dv[rng.integers(0, dv.size, size=(N_BOOT, dv.size))].mean(axis=1)
    return {"n_pairs": int(dv.size), "mean": float(dv.mean()), "sd": float(dv.std(ddof=1)),
            "median": float(np.median(dv)), "min": float(dv.min()), "max": float(dv.max()),
            "n_neg": int((dv < 0).sum()), "n_pos": int((dv > 0).sum()),
            "n_zero": int((dv == 0).sum()),
            "boot_ci95_lo": float(np.percentile(bm, 2.5)),
            "boot_ci95_hi": float(np.percentile(bm, 97.5)),
            "_lower": lower}


def _pair_diff(df: pd.DataFrame, q: int, x: str, y: str, col: str) -> np.ndarray:
    """x - y per seed (sorted seeds) for one q."""
    g = df[df["q"] == q]
    a = g[g["arm"] == x].set_index("seed")[col]
    b = g[g["arm"] == y].set_index("seed")[col]
    seeds = sorted(set(a.index) & set(b.index))
    return np.array([float(a[s]) - float(b[s]) for s in seeds], dtype=float)


def _manual_check(pack: C.Pack, item: str, report: str, label: str, n_cmp: int, n_bad: int) -> None:
    """Record a hand-made comparison in the module's cross-check summary."""
    pack.crosschecks.append({"item": item, "report": report,
                             "label": label + " (manual comparison)", "n_tables": 1,
                             "n_compared": int(n_cmp), "n_mismatch": int(n_bad),
                             "n_unmatched_rows": 0})


def _ci_cell_ok(lo: float, hi: float, cell: str) -> Optional[bool]:
    """Whether a cell '[a, b]' matches (lo, hi) at the shown precision (None if unparseable)."""
    m = re.match(r"^\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\]\s*$", str(cell))
    if not m:
        return None
    return C.consistent(lo, m.group(1)) and C.consistent(hi, m.group(2))


def _band_cell_ok(lo: float, hi: float, cell: str) -> Optional[bool]:
    """Whether a report band cell '[0.0182, 0.0233]' equals (lo, hi) formatted with 4 decimals."""
    m = re.match(r"^\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\]\s*$", str(cell))
    if not m:
        return None
    fmt = lambda v: f"{v:.4f}".replace("-0.0000", "0.0000")  # noqa: E731
    return fmt(lo) == m.group(1).strip().replace("-0.0000", "0.0000") and \
        fmt(hi) == m.group(2).strip().replace("-0.0000", "0.0000")


def _report_table(report: str, heading_has: str, header_has: Sequence[str]) -> Dict[str, Any]:
    """The single report table under a heading whose header contains the given names."""
    hits = [t for t in C.parse_md_tables(report) if heading_has.lower() in t["heading"].lower()
            and all(h in t["header"] for h in header_has)]
    if len(hits) != 1:
        raise ValueError(f"{report}: {len(hits)} tables for heading '{heading_has}' / "
                         f"{list(header_has)}")
    return hits[0]


def _check_paired_extras(pack: C.Pack, item: str, df: pd.DataFrame, report: str, heading_has: str,
                         better_col: str, ci_col: str = "CI95") -> None:
    """Compare the CI95 and 'better' cells of a paired report table with the pack rows."""
    t = _report_table(report, heading_has, ["label", ci_col])
    hdr = t["header"]
    n_cmp = n_bad = 0
    for row in t["rows"]:
        if len(row) != len(hdr):
            continue
        rec = dict(zip(hdr, row))
        hit = df[df["label"] == rec["label"]]
        if len(hit) != 1:
            continue
        h = hit.iloc[0]
        ok = _ci_cell_ok(float(h["boot_ci95_lo"]), float(h["boot_ci95_hi"]), rec[ci_col])
        if ok is not None:
            n_cmp += 1
            if not ok:
                n_bad += 1
                pack.mismatch(item, f"bootstrap CI95 [{heading_has}; {rec['label']}]",
                              f"[{h['boot_ci95_lo']!r}, {h['boot_ci95_hi']!r}]",
                              f"{report} (line {t['line']})", rec[ci_col])
        cell = rec.get(better_col, "")
        m1 = re.match(r"^(\d+)/(\d+)$", cell.strip())
        m2 = re.match(r"^n/a \((\d+)\D+(\d+)\D+(\d+)\D*\)$", cell.strip())
        if m1:
            n_cmp += 1
            val = h[_first_better_col(df)]
            if pd.isna(val) or int(val) != int(m1.group(1)):
                n_bad += 1
                pack.mismatch(item, f"better count [{heading_has}; {rec['label']}]", val,
                              f"{report} (line {t['line']})", cell)
        elif m2:
            n_cmp += 1
            got = (int(h["n_neg"]), int(h["n_pos"]), int(h["n_zero"]))
            exp = tuple(int(m2.group(i)) for i in (1, 2, 3))
            if got != exp:
                n_bad += 1
                pack.mismatch(item, f"sign counts [{heading_has}; {rec['label']}]", str(got),
                              f"{report} (line {t['line']})", cell)
    _manual_check(pack, item, report, f"CI95 and better/sign counts, '{heading_has}'", n_cmp, n_bad)


def _first_better_col(df: pd.DataFrame) -> str:
    """Name of the 'number better' column of a paired table."""
    return "n_better_first" if "n_better_first" in df.columns else "n_mean_better"


# ----------------------------------------------------------------------------------------------
# data context (read once)
# ----------------------------------------------------------------------------------------------

def _rec_arrays(rel: str) -> Dict[str, np.ndarray]:
    """Recovery-grid arrays (tier-independent) of a final_*.npz."""
    with np.load(C.abspath(rel)) as z:
        return {k: np.asarray(z[k], dtype=float) for k in ("recovery_d_grid", "recovery_e2",
                                                           "recovery_g2")}


def _locfree(arr: Dict[str, np.ndarray], g20: float) -> Dict[str, float]:
    """Location-free peak error (same formula as tools/v2/pilot4_common.py:location_free)."""
    D, e2 = arr["recovery_d_grid"], arr["recovery_e2"]
    j = int(np.argmax(e2))
    err = (float(e2[j]) - g20) / g20
    return {"stage2_peak_locfree_rel_err": err, "stage2_peak_locfree_rel_err_abs": abs(err),
            "stage2_peak_locfree_argmax_d": float(D[j]), "stage2_max_e2": float(e2[j])}


def _context() -> Dict[str, Any]:
    """Read every per-run record and analysis file used by the items of this module."""
    ctx: Dict[str, Any] = {}
    ctx["ft2"] = _csv(f"{P2A}/final_table.csv", dtype={"commit": str, "parent_sha256": str})
    ctx["ft3"] = _csv(f"{P3A}/final_table.csv", dtype={"commit": str})
    ctx["fv2"] = {(q, s, a): S.final_v2(rd) for q, s, a, rd in S.iter_runs("pilot2")}
    ctx["fv3"] = {(q, s, a): S.final_v2(rd) for q, s, a, rd in S.iter_runs("pilot3")}
    ctx["fin2"] = S.final_tier_columns("pilot2")
    ctx["fin3"] = S.final_tier_columns("pilot3")
    ctx["dec2"] = _csv(f"{P2A}/decomposition_residual_band.csv")
    ctx["dec3"] = _csv(f"{P3A}/decomposition_residual_band.csv")
    ctx["cur2"] = _csv(f"{P2A}/curves_weights_every25.csv")
    ctx["cur3"] = _csv(f"{P3A}/curves_weights_every25.csv")
    rec: Dict[Tuple[int, int, str], Dict[str, np.ndarray]] = {}
    for q, s, a, rd in S.iter_runs("pilot2"):
        rec[(q, s, a)] = _rec_arrays(f"{rd}/final_development.npz")
    for q in QS:
        for s in SEEDS:
            rec[(q, s, "parent")] = _rec_arrays(f"{P1}/q{q}/seed{s}/expected/final_development.npz")
    ctx["rec"] = rec
    cks = []
    for q, s, a, rd in S.iter_runs("pilot2"):
        ck = _csv(f"{rd}/v2_checkpoints.csv")
        cks.append(ck[["q", "seed", "arm", "update", "stage1_drift", "Gmax_full_t", "Gmax_full_d",
                       "stage2_drift_cand_maxabs"]])
    ctx["ck2"] = pd.concat(cks, ignore_index=True)
    ctx["dt2"] = {(q, s, a): S.read_json(f"{rd}/drift_test.json")
                  for q, s, a, rd in S.iter_runs("pilot2")}
    ctx["rc2"] = {(q, s, a): S.read_json(f"{rd}/run_config.json")
                  for q, s, a, rd in S.iter_runs("pilot2")}
    ctx["rc3"] = {(q, s, a): S.read_json(f"{rd}/run_config.json")
                  for q, s, a, rd in S.iter_runs("pilot3")}
    # provenance sets (hashed once; the hash cache is per process)
    ctx["src_fv2"] = C.srcs(f"{P2}/q*/seed*/*/final_v2.json", "Pilot 2 final_v2.json (60 runs)",
                            expect=60)
    ctx["src_fv3"] = C.srcs(f"{P3}/q*/seed*/*/final_v2.json", "Pilot 3 final_v2.json (40 runs)",
                            expect=40)
    ctx["src_npz2"] = C.srcs(f"{P2}/q*/seed*/*/final_development.npz",
                             "Pilot 2 final_development.npz (60 runs)", expect=60)
    ctx["src_npz1"] = C.srcs(f"{P1}/q*/seed*/expected/final_development.npz",
                             "Pilot 1 expected final_development.npz (20 parents, u400)", expect=20)
    ctx["src_ck2"] = C.srcs(f"{P2}/q*/seed*/*/v2_checkpoints.csv",
                            "Pilot 2 v2_checkpoints.csv (60 runs)", expect=60)
    ctx["src_dt2"] = C.srcs(f"{P2}/q*/seed*/*/drift_test.json", "Pilot 2 drift_test.json (60 runs)",
                            expect=60)
    ctx["src_rc2"] = C.srcs(f"{P2}/q*/seed*/*/run_config.json", "Pilot 2 run_config.json (60 runs)",
                            expect=60)
    ctx["src_rc3"] = C.srcs(f"{P3}/q*/seed*/*/run_config.json", "Pilot 3 run_config.json (40 runs)",
                            expect=40)
    ctx["src_par"] = C.srcs(f"{P1}/q*/seed*/expected/state_end_A.pt", "Pilot 1 expected "
                                                                      "state_end_A.pt (20 parents)",
                            expect=20)
    ctx["p2"] = _p2_runs(ctx)
    ctx["p3"] = _p3_runs(ctx)
    return ctx


def _dec_final(dec: pd.DataFrame) -> pd.DataFrame:
    """Decomposition rows of the u1000 weight export, one per (q, seed, arm)."""
    w = dec[(dec["source"] == "weights") & (dec["update"] == 1000)].copy()
    if w.duplicated(["q", "seed", "arm"]).any():
        raise ValueError("duplicate u1000 decomposition rows")
    return w


def _p2_runs(ctx: Dict[str, Any]) -> pd.DataFrame:
    """Per-run Pilot 2 table (D02): final_table.csv columns + appended documented columns."""
    ft = ctx["ft2"].copy()
    keys = ["q", "seed", "arm"]
    lf, dr = [], []
    for r in ft[keys].itertuples(index=False):
        fv = ctx["fv2"][(r.q, r.seed, r.arm)]
        lf.append({"q": r.q, "seed": r.seed, "arm": r.arm,
                   **_locfree(ctx["rec"][(r.q, r.seed, r.arm)],
                              float(fv["development"]["g2_at_0"]))})
        dvp = fv["drift_vs_parent"]
        dr.append({"q": r.q, "seed": r.seed, "arm": r.arm,
                   **{k: float(dvp[k]) for k in ("stage2_drift_cand_on_mean_unweighted",
                                                 "stage2_drift_cand_maxabs",
                                                 "stage2_drift_live_on_max",
                                                 "stage2_drift_live_on_mean_cellmass_weighted",
                                                 "stage2_drift_live_on_mean_unweighted",
                                                 "stage2_drift_live_off_max",
                                                 "stage2_drift_live_off_mean_unweighted")}})
    df = ft.merge(pd.DataFrame(lf), on=keys, how="left", validate="one_to_one")
    df = df.merge(pd.DataFrame(dr), on=keys, how="left", validate="one_to_one")
    fin = ctx["fin2"].drop(columns=["run_dir", FT + "stage1_learning_err_rel",
                                    FT + "stage1_inherited_err_rel"])
    df = df.merge(fin, on=keys, how="left", validate="one_to_one")
    w = _dec_final(ctx["dec2"])
    cols = ["e_tilde", "band_lo", "band_hi", "learning_rel", "learning_rel_lo", "learning_rel_hi",
            "learning_contains_0", "inherited_rel", "inherited_rel_lo", "inherited_rel_hi",
            "inherited_contains_0", "total_rel", "e1_inside_sweep"]
    w = w[keys + cols].rename(columns={c: "dec_" + c for c in cols})
    w["dec_learning_rel_abs"] = w["dec_learning_rel"].abs()
    w["dec_inherited_rel_abs"] = w["dec_inherited_rel"].abs()
    df = df.merge(w, on=keys, how="left", validate="one_to_one")
    must = ["dec_learning_rel", FT + "Gmax_full_over_dw", "stage2_peak_locfree_rel_err"]
    if df[must].isna().any().any():
        raise ValueError("Pilot 2 per-run table has missing appended values")
    return df


def _p3_runs(ctx: Dict[str, Any]) -> pd.DataFrame:
    """Per-run Pilot 3 table (D03): final_table.csv columns + appended documented columns."""
    ft = ctx["ft3"].copy()
    keys = ["q", "seed", "arm"]
    df = ft.merge(ctx["fin3"].drop(columns=["run_dir"]), on=keys, how="left", validate="one_to_one")
    w = _dec_final(ctx["dec3"])
    cols = ["e_tilde", "band_lo", "band_hi", "total_rel", "e1_inside_sweep"]
    w = w[keys + cols].rename(columns={c: "dec_" + c for c in cols})
    df = df.merge(w, on=keys, how="left", validate="one_to_one")
    df["inherited_rel_abs"] = df["inherited_rel"].abs()
    if df[[FT + "Gmax_full_over_dw", "dec_e_tilde"]].isna().any().any():
        raise ValueError("Pilot 3 per-run table has missing appended values")
    return df


# ----------------------------------------------------------------------------------------------
# column documentation shared by the items
# ----------------------------------------------------------------------------------------------

SUMMARY_DOCS = {
    "arm_short": "Short arm label. Pilot 2: A = A_joint (stage 2 trained jointly in Phase B), B1 "
                 "= B1_frozen_allnorm (stage 2 frozen, advantages normalized over all rows), B2 = "
                 "B2_frozen_s1norm (stage 2 frozen, advantages normalized over stage-1 rows). "
                 "Pilot 3: stochastic = B2_frozen_s1norm, mean = B2_frozen_s1norm_mean "
                 "(continuation action mode)",
    "metric": "Per-run column that is summarized (definitions in data_dictionary.csv under D02 / "
              "D03; dec_* = revised residual-band decomposition at u1000)",
    "tier": dict(definition="Verifier tier of the metric: " + TIER_TXT
                 + "; the stage-2 drift is measured on the development-tier D_2 grid",
                 units="label"),
    "n_true": dict(definition="Boolean metrics only (band contains 0): number of runs where the "
                              "flag is True, out of n",
                   units="count"),
    "argmax_t_d_counts": dict(definition="Gmax_full rows only: the locations (t*, d*) at which "
                                         "Gmax_full is attained, with the number of runs at each, "
                                         "on the row's tier",
                              units="stage, effort units"),
}


def _orig_docs_p2() -> Dict[str, Any]:
    """Docs of the original Pilot 2 final_table.csv columns missing from the global dictionary."""
    return {
        "reward_mode": dict(definition="Terminal reward estimator of the run (expected = "
                                       "conditional expected terminal reward, chosen after "
                                       "Pilot 1)",
                            units="label", tier=NA, source="run manifest"),
        "parent_sha256": dict(definition="First 12 hex digits of the SHA-256 of the restored "
                                         "parent full state (Pilot 1 expected, state_end_A.pt, "
                                         "global u400); the full hash in run_config.json equals "
                                         "the hash of results/v2_pilots/pilot1/q*/seed*/expected/"
                                         "state_end_A.pt",
                              units="hash", tier=NA, source="run manifest (parent_sha256)"),
        "n_verifier_checkpoints": dict(definition="Number of training-time verifier checkpoints "
                                                  "in v2_checkpoints.csv (global u500 to u1000 "
                                                  "every 25 updates)",
                                       units="count", tier=NA,
                                       source="tools/v2/pilot2_analysis.py:final_table"),
        "phase_B_wall_sec": dict(definition="Phase B wall-clock time of the run "
                                            "(v2_run_summary.json phase_timing.B)",
                                 units="seconds", tier=NA, source="run/run_v2_stagewise.py"),
        "would_have_fired_B": dict(definition="JSON record of the first update at which the "
                                              "legacy Phase-B stop rule (k_phase = 3 consecutive "
                                              "eligible calls, EXP_root/dW <= 0.02, concentration "
                                              "<= 0.04) would have fired; empty = never. The "
                                              "fixed budget means the rule never stopped a run",
                                   units="JSON", tier=DEV,
                                   source="run/run_v2_stagewise.py (would_fire)"),
        "snapshot_drift_mean": dict(definition="Frozen arms: max over the dev D_2 grid of "
                                               "|snapshot stage-2 Beta mean at u1000 - at freeze "
                                               "time| (drift_test.json); joint arm: the same for "
                                               "the live network (its stage-2 output moves)",
                                    units="effort units",
                                    tier=DEV, source="run/run_v2_stagewise.py (drift_test.json)"),
        "snapshot_drift_alpha": dict(definition="As snapshot_drift_mean, for the Beta alpha "
                                                "parameter",
                                     units="Beta parameter units", tier=DEV,
                                     source="drift_test.json"),
        "snapshot_drift_beta": dict(definition="As snapshot_drift_mean, for the Beta beta "
                                               "parameter",
                                    units="Beta parameter units", tier=DEV,
                                    source="drift_test.json"),
        "drift_test_pass": dict(definition="C5 output-drift test of the frozen snapshot (zero "
                                           "drift of mean/alpha/beta, tensors bit-identical to "
                                           "the parent actor, no gradient, in no optimizer, eval "
                                           "mode); 'n/a (joint)' for A_joint", units="bool / label",
                                tier=NA,
                                source="drift_test.json"),
        "live_stage2_drift_maxabs": dict(definition="max over the dev D_2 grid of |live network "
                                                    "stage-2 Beta mean - parent stage-2 mean| at "
                                                    "u1000 (in the frozen arms the live stage-2 "
                                                    "output is not used for any action)",
                                         units="effort units", tier=DEV,
                                         source="v2_checkpoints.csv (stage2_drift_live_maxabs)"),
        "induced_e1": dict(definition="SUPERSEDED (pilot2_freeze.md Appendix S): induced stage-1 "
                                      "target from the bracketing + Brent solver on the "
                                      "development tier",
                           units="effort units", tier=DEV,
                           source="utils/v2_metrics.py:induced_stage1_target (legacy)"),
        "induced_residual": dict(definition="SUPERSEDED: |BR(e~1) - e~1| of the superseded solver",
                                 units="effort units", tier=DEV,
                                 source="utils/v2_metrics.py (legacy)"),
        "stage1_learning_err_rel": dict(definition="SUPERSEDED solver (Appendix S): learning term "
                                                   "(e1 - e~1)/e1* on the development tier; the "
                                                   "revised term is dec_learning_rel",
                                        units="fraction of e_1*(0)", tier=DEV,
                                        source="utils/v2_metrics.py (legacy)"),
        "stage1_learning_err_rel_abs": dict(definition="SUPERSEDED solver: "
                                                       "|stage1_learning_err_rel|",
                                            units="fraction of e_1*(0)", tier=DEV,
                                            source="tools/v2/pilot2_analysis.py"),
        "stage1_inherited_err_rel": dict(definition="SUPERSEDED solver (Appendix S): inherited "
                                                    "term (e~1 - e1*)/e1* on the development "
                                                    "tier; the revised term is dec_inherited_rel",
                                         units="fraction of e_1*(0)", tier=DEV,
                                         source="utils/v2_metrics.py (legacy)"),
        "stage1_inherited_err_rel_abs": dict(definition="SUPERSEDED solver: "
                                                        "|stage1_inherited_err_rel|",
                                             units="fraction of e_1*(0)", tier=DEV,
                                             source="tools/v2/pilot2_analysis.py"),
    }


def _appended_docs() -> Dict[str, Any]:
    """Docs of the columns appended by this module (both per-run tables)."""
    d = {
        "stage2_peak_locfree_rel_err_abs": dict(
            definition="|stage2_peak_locfree_rel_err|", units="fraction of e_2*(0)",
            normalization="divided by e_2*(0)", tier=IND,
            source="tools/v2/pilot4_common.py:location_free (abs)"),
        "inherited_rel_abs": dict(definition="|inherited_rel| (revised residual-band inherited "
                                             "term, absolute)",
                                  units="fraction of e_1*(0)", normalization="divided by e_1*(0)",
                                  tier=FIN,
                                  source="tools/v2/decomposition.py (abs taken by the pack)"),
        "dec_learning_rel_abs": dict(definition="|dec_learning_rel|", units="fraction of e_1*(0)",
                                     normalization="divided by e_1*(0)", tier=FIN,
                                     source="tools/v2/decomposition.py"),
        "dec_inherited_rel_abs": dict(definition="|dec_inherited_rel|", units="fraction of e_1*(0)",
                                      normalization="divided by e_1*(0)", tier=FIN,
                                      source="tools/v2/decomposition.py"),
    }
    for c in ("e_tilde", "band_lo", "band_hi", "learning_rel", "learning_rel_lo", "learning_rel_hi",
              "learning_contains_0", "inherited_rel", "inherited_rel_lo", "inherited_rel_hi",
              "inherited_contains_0", "total_rel", "e1_inside_sweep"):
        base = DICT.lookup(c) or {}
        d["dec_" + c] = dict(definition="Revised stage-1 decomposition at the u1000 weight export "
                                        "(residual band on the final tier; "
                                        "decomposition_residual_band.csv, source = weights): "
                                        + base.get("definition", ""), units=base.get("units", ""),
                             normalization=base.get("normalization", ""), tier=FIN,
                             source=base.get("source", ""))
    return d


# ----------------------------------------------------------------------------------------------
# T22: Pilot 2 final medians
# ----------------------------------------------------------------------------------------------

T22_SPECS = [
    ("stage2_drift_cand_on_max", "stage2_drift_cand_on_max", DEV, "num"),
    ("stage2_drift_cand_on_mean_cellmass_weighted", "stage2_drift_cand_on_mean_cellmass_weighted",
     DEV, "num"),
    ("stage2_drift_cand_off_max", "stage2_drift_cand_off_max", DEV, "num"),
    ("stage2_drift_cand_off_mean_unweighted", "stage2_drift_cand_off_mean_unweighted", DEV, "num"),
    ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_signed", IND, "num"),
    ("stage2_peak_rel_err_abs", "stage2_peak_rel_err_abs", IND, "num"),
    ("stage2_peak_locfree_rel_err", "stage2_peak_locfree_rel_err", IND, "num"),
    ("stage2_peak_locfree_rel_err_abs", "stage2_peak_locfree_rel_err_abs", IND, "num"),
    ("stage2_peak_locfree_argmax_d", "stage2_peak_locfree_argmax_d", IND, "num"),
    ("stage2_tail_mean", "stage2_tail_mean", IND, "num"),
    ("stage2_tail_mean_over_g2_0", "stage2_tail_mean_over_g2_0", IND, "num"),
    ("stage1_rel_err_signed", "stage1_rel_err_signed", IND, "num"),
    ("stage1_rel_err_abs", "stage1_rel_err_abs", IND, "num"),
    ("Gmax_full_over_dw", "Gmax_full_over_dw", DEV, "num"),
    ("Gmax_full_over_dw", FT + "Gmax_full_over_dw", FIN, "num"),
    ("EXP_root_over_dw", "EXP_root_over_dw", DEV, "num"),
    ("EXP_root_over_dw", FT + "EXP_root_over_dw", FIN, "num"),
    ("dReach_over_dw", "dReach_over_dw", DEV, "num"),
    ("dReach_over_dw", FT + "dReach_over_dw", FIN, "num"),
    ("dec_learning_rel", "dec_learning_rel", FIN, "num"),
    ("dec_learning_rel_abs", "dec_learning_rel_abs", FIN, "num"),
    ("dec_inherited_rel", "dec_inherited_rel", FIN, "num"),
    ("dec_inherited_rel_abs", "dec_inherited_rel_abs", FIN, "num"),
    ("dec_learning_contains_0", "dec_learning_contains_0", FIN, "count"),
    ("dec_inherited_contains_0", "dec_inherited_contains_0", FIN, "count"),
]


def _with_argmax(t: pd.DataFrame, p: pd.DataFrame, arms_short: Dict[str, str]) -> pd.DataFrame:
    """Add the (t*, d*) counts to the Gmax_full rows and the short arm label."""
    t = t.copy()
    t.insert(2, "arm_short", t["arm"].map(arms_short))
    vals = []
    for r in t.itertuples(index=False):
        if r.metric != "Gmax_full_over_dw":
            vals.append("")
            continue
        g = p[(p["q"] == r.q) & (p["arm"] == r.arm)]
        tc, dc = (("Gmax_full_t", "Gmax_full_d") if r.tier == DEV
                  else (FT + "Gmax_full_t", FT + "Gmax_full_d"))
        vals.append(_argmax_counts(g[tc], g[dc]))
    t["argmax_t_d_counts"] = vals
    return t


def build_t22(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """T22: Pilot 2 final medians per arm and q, both tiers for tier-dependent metrics."""
    p2 = ctx["p2"]
    t = _with_argmax(pd.DataFrame(_summary_rows(p2, ["q", "arm"], T22_SPECS)), p2, SHORT2)
    fv = ctx["fv2"]
    dev_cols = ("Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw", "stage1_rel_err_signed",
                "stage2_peak_rel_err_signed")
    dev_eq = max(abs(float(r[c]) - float(fv[(r["q"], r["seed"], r["arm"])]["development"][c]))
                 for _, r in p2.iterrows() for c in dev_cols)
    drift_cols = ("stage2_drift_cand_on_max", "stage2_drift_cand_off_max")
    drift_eq = max(abs(float(r[c]) - float(fv[(r["q"], r["seed"], r["arm"])]["drift_vs_parent"][c]))
                   for _, r in p2.iterrows() for c in drift_cols)
    n_out = int((~ctx["dec2"]["e1_inside_sweep"].astype(bool)).sum())
    notes = ("Median, q25, q75 (numpy linear interpolation), min, max and n over the 10 seeds "
             f"({SEEDS_TXT}) per (q, arm) at global u1000. Development-tier values: "
             "results/v2_pilots/pilot2/analysis/final_table.csv (last training-time checkpoint; "
             f"equal to final_v2.json['development'] within {dev_eq:.2g}); final-tier values: "
             "final_v2.json['final'] (studies.final_tier_columns). Stage-2 drift = |e_hat_2 - "
             "e_hat_2^parent| of the candidate's stage-2 mapping on the dev D_2 grid (A: live "
             "network; B1/B2: frozen snapshot), effort units (final_table.csv equals "
             f"final_v2.json['drift_vs_parent'] within {drift_eq:.2g}). Location-free peak error "
             "from the recovery arrays of final_development.npz (formula of "
             f"tools/v2/pilot4_common.py:location_free). dec_* = {DECOMP_TXT}; the superseded "
             "stage1_learning_err_rel / stage1_inherited_err_rel of final_table.csv are not used. "
             f"{n_out} early decomposition rows (u425/u450, B1/B2, q=60) have e_hat_1 outside the "
             "parent sweep; no u1000 row is affected. (t*, d*) counts per location are also in "
             "T25.")
    sources = [C.src(f"{P2A}/final_table.csv"), ctx["src_fv2"], ctx["src_npz2"],
               C.src(f"{P2A}/decomposition_residual_band.csv")]
    pack.table("T22", t, status="generated", sources=sources, script=_sc("build_t22"), notes=notes,
               docs=SUMMARY_DOCS, tier="final and development",
               caption="Pilot 2 at global u1000 (end of Phase B), per q and arm, n = 10 seeds "
                       f"({SEEDS_TXT}). Drift and tail mean in effort units (raw); errors relative "
                       "to e_1*(0) or e_2*(0); Gmax_full, EXP_root, dReach divided by Delta W = "
                       "4. Column tier names the verifier tier of each row.")
    # cross-checks against the existing reports (development-tier medians)
    med = (p2.assign(arm_s=p2["arm"].map(SHORT2)).groupby(["q", "arm_s"])
           .median(numeric_only=True).reset_index())
    med = med.rename(columns={"arm_s": "arm"})
    cols = ["stage2_peak_rel_err_signed", "stage2_tail_mean", "stage2_drift_cand_on_max",
            "stage2_drift_cand_off_max", "stage1_rel_err_signed", "Gmax_full_over_dw",
            "EXP_root_over_dw", "dReach_over_dw"]
    pack.crosscheck("T22", med, REP2,
                    header_has=["q", "arm", "stage2_drift_cand_on_max", "eta_T_over_dw"],
                    key_map={"q": "q", "arm": "arm"}, value_map={c: c for c in cols},
                    heading_has="2.1 Final checkpoint", label="Pilot 2 medians (section 2.1)")
    pack.crosscheck("T22", med, SUMM,
                    header_has=["q", "arm", "stage2_drift_cand_on_max", "Gmax_full_over_dw"],
                    key_map={"q": "q", "arm": "arm"}, value_map={c: c for c in cols},
                    heading_has="Pilot 2", label="summary.md Pilot 2 medians")
    cnt = p2.assign(arm_s=p2["arm"].map(SHORT2)).groupby(["q", "arm_s"]).agg(
        n_l=("dec_learning_contains_0", "sum"),
        n_i=("dec_inherited_contains_0", "sum")).reset_index()
    dm = med.merge(cnt.rename(columns={"arm_s": "arm"}), on=["q", "arm"])
    dm["n_l"] = dm["n_l"].astype(float)
    dm["n_i"] = dm["n_i"].astype(float)
    pack.crosscheck("T22", dm, REP2, header_has=["median total", "median abs learning"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={"median total": "stage1_rel_err_signed",
                               "median learning": "dec_learning_rel",
                               "median abs learning": "dec_learning_rel_abs",
                               "learning band contains 0 (of 10)": "n_l",
                               "median inherited": "dec_inherited_rel",
                               "median abs inherited": "dec_inherited_rel_abs",
                               "inherited band contains 0 (of 10)": "n_i"},
                    heading_has="Summary per (q, arm)",
                    label="revised decomposition medians (section 6)")
    pack.crosscheck("T22", dm, REP2, header_has=["superseded median abs learning"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={"median abs learning": "dec_learning_rel_abs",
                               "median inherited": "dec_inherited_rel",
                               "median abs inherited": "dec_inherited_rel_abs"},
                    heading_has="Comparison with the superseded solver",
                    label="revised medians vs superseded (section 6)")
    pack.crosscheck("T22", dm, SUMM, header_has=["median_abs_learning", "inherited_band_contains0"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={"median_abs_learning": "dec_learning_rel_abs",
                               "median_inherited": "dec_inherited_rel",
                               "inherited_band_contains0": "n_i"},
                    heading_has="Pilot 2", label="summary.md revised decomposition")
    out = ctx["dec2"][~ctx["dec2"]["e1_inside_sweep"].astype(bool)]
    if len(out) and set(out["source"]) == {"weights"}:
        pack.mismatch("T22", "kind of the Pilot 2 decomposition rows with e_hat_1 outside the "
                             "parent sweep",
                      f"{len(out)} rows, all weight exports (updates "
                      f"{sorted(int(u) for u in set(out['update']))}; arms "
                      f"{sorted(set(out['arm']))}; q {sorted(int(v) for v in set(out['q']))})",
                      f"{REP2} (section 6, below 'Summary per (q, arm)')",
                      "The 7 out-of-sweep rows are early B1/B2 exports and checkpoints",
                      "count agrees; none of the rows is a training-time checkpoint row (those "
                      "start at u500)")


# ----------------------------------------------------------------------------------------------
# T23: Pilot 2 paired comparisons
# ----------------------------------------------------------------------------------------------

def _tier_p2(m: str) -> str:
    """Tier label of a Pilot 2 paired metric."""
    if m.startswith(FT) or m.startswith("dec_"):
        return FIN
    if m.startswith("stage2_drift_") or m in DEV_METRICS:
        return DEV
    if m in ("kl_final_epoch", "clip_frac", "phase_B_wall_sec"):
        return TRAIN
    return IND


def build_t23(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """T23: paired comparisons B1 - A, B2 - A, B2 - B1 (existing summary + appended rows)."""
    rel = f"{P2A}/paired_summary.csv"
    ps = _csv(rel)
    p2, ft2 = ctx["p2"], ctx["ft2"]
    metrics = list(dict.fromkeys(ps["metric"]))
    # verify the existing summary: point statistics and the sequential bootstrap (pilot2_analysis)
    rng = np.random.default_rng(BOOT_SEED)
    rep = []
    for q in QS:
        for x, y in COMPARE2:
            for m in metrics:
                rep.append({"q": q, "comparison": f"{SHORT2[x]}-{SHORT2[y]}", "metric": m,
                            **_paired_row(_pair_diff(ft2, q, x, y, m), rng, True)})
    rp = ps.merge(pd.DataFrame(rep), on=["q", "comparison", "metric"], suffixes=("", "_re"),
                  validate="one_to_one")
    d_pt = max(C.max_abs_diff(rp[c], rp[c + "_re"]) for c in ("mean", "sd", "median", "min", "max",
                                                              "n_neg", "n_pos", "n_zero"))
    d_ci = max(C.max_abs_diff(rp[c], rp[c + "_re"]) for c in ("boot_ci95_lo", "boot_ci95_hi"))
    ps["tier"] = [_tier_p2(m) for m in ps["metric"]]
    ps["source"] = rel
    ps["note"] = ["superseded solver (pilot2_freeze.md Appendix S); revised terms in the dec_* rows"
                  if m in SUPERSEDED2 else "" for m in ps["metric"]]
    # new rows: final tier, location-free peak error, revised decomposition
    extra = {
        FT + "Gmax_full_over_dw": ("Gmax_full/DW (final tier)", True),
        FT + "EXP_root_over_dw": ("EXP_root/DW (final tier)", True),
        FT + "dReach_over_dw": ("dReach/DW (final tier)", True),
        "stage2_peak_locfree_rel_err": ("stage-2 location-free peak rel. err (signed)", False),
        "stage2_peak_locfree_rel_err_abs": ("stage-2 abs location-free peak rel. err", True),
    }
    src_fresh = ("recomputed from the per-run values (D02); bootstrap: fresh numpy "
                 "default_rng(20261001) per row, 10,000 resamples of the 10 pairs")
    dec_map = [("abs_learning", "dec_learning_rel_abs", "revised abs learning term / e1*", True),
               ("abs_inherited", "dec_inherited_rel_abs", "revised abs inherited term / e1*", True),
               ("learning_rel", "dec_learning_rel", "revised learning term (e1 - e~1)/e1* (signed)",
                False),
               ("inherited_rel", "dec_inherited_rel",
                "revised inherited term (e~1 - e1*)/e1* (signed)", False)]
    rng_dec = np.random.default_rng(BOOT_SEED)   # one stream, row order of the existing pairs file
    dec_rows: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for q in QS:
        for x, y in COMPARE2:
            comp = f"{SHORT2[x]}-{SHORT2[y]}"
            for _, col, _, low in dec_map:
                dec_rows[(q, comp, col)] = _paired_row(_pair_diff(p2, q, x, y, col), rng_dec, low)
    blocks = []
    for q in QS:
        for x, y in COMPARE2:
            comp = f"{SHORT2[x]}-{SHORT2[y]}"
            blocks.append(ps[(ps["q"] == q) & (ps["comparison"] == comp)])
            new = []
            for col, (label, low) in extra.items():
                r = _paired_row(_pair_diff(p2, q, x, y, col), np.random.default_rng(BOOT_SEED), low)
                new.append({"q": q, "comparison": comp, "metric": col, "label": label, **r,
                            "source": src_fresh, "note": ""})
            for _, col, label, low in dec_map:
                new.append({"q": q, "comparison": comp, "metric": col, "label": label,
                            **dec_rows[(q, comp, col)],
                            "source": "recomputed from decomposition_residual_band.csv (u1000 "
                                      "weight exports); bootstrap: one numpy "
                                      "default_rng(20261001) stream in the row order of "
                                      "decomposition_residual_band_paired.csv (reproduces its "
                                      "CI95)",
                            "note": DECOMP_TXT})
            nd = pd.DataFrame(new)
            nd["better"] = [f"{SHORT2[x]} better if diff < 0" if lw else "no preferred direction"
                            for lw in nd["_lower"]]
            nd["n_better_first"] = [float(n) if lw else np.nan
                                    for n, lw in zip(nd["n_neg"], nd["_lower"])]
            nd["tier"] = [_tier_p2(m) for m in nd["metric"]]
            blocks.append(nd.drop(columns=["_lower"])[list(ps.columns)])
    t = pd.concat(blocks, ignore_index=True)
    # existing revised-decomposition pairs file: equality at its precision
    dp = _csv(f"{P2A}/decomposition_residual_band_paired.csv")
    name = {a: c for a, c, _, _ in dec_map}
    n_cmp = n_bad = 0
    for r in dp.itertuples(index=False):
        mine = dec_rows[(int(r.q), r.comparison, name[r.metric])]
        for got, exp in ((mine["mean"], r.mean), (mine["median"], r.median)):
            n_cmp += 1
            if abs(got - exp) > 1e-12:
                n_bad += 1
        ok = _ci_cell_ok(mine["boot_ci95_lo"], mine["boot_ci95_hi"], r.CI95)
        n_cmp += 1
        n_bad += 0 if ok else 1
    notes = (f"Rows with source = {rel} are verbatim (all 31 metrics of the existing analysis; "
             "development tier for verifier metrics, dev D_2 grid for drift); recomputing them "
             f"from final_table.csv gives max abs diff {d_pt:.2g} (point statistics) and "
             f"{d_ci:.2g} (CI ends, single numpy default_rng(20261001) stream in the order of "
             "tools/v2/pilot2_analysis.py:paired). Appended per comparison: final-tier Gmax_full, "
             "EXP_root, dReach and the location-free peak error (not in the existing analysis; "
             "bootstrap with a fresh default_rng(20261001) per row, 10,000 resamples) and the "
             "revised residual-band decomposition terms (their recomputation matches "
             f"decomposition_residual_band_paired.csv in {n_cmp - n_bad}/{n_cmp} cells: mean and "
             "median to 1e-12, CI95 at its 4 displayed digits). Difference = first arm - second "
             "arm per seed; better = number of pairs with diff < 0 for metrics where smaller is "
             "better. The location (t*, d*) of Gmax_full is categorical and is compared through "
             "the counts in T25, not as a paired difference.")
    sources = [C.src(rel), C.src(f"{P2A}/final_table.csv"),
               C.src(f"{P2A}/decomposition_residual_band_paired.csv"),
               C.src(f"{P2A}/decomposition_residual_band.csv"), ctx["src_fv2"], ctx["src_npz2"]]
    docs = {"n_better_first": dict(definition="For metrics where smaller is better: number of "
                                              "pairs with first arm - second arm < 0 (first arm "
                                              "better); empty for signed metrics", units="count"),
            "tier": dict(definition="Tier of the metric: " + TIER_TXT + "; " + TRAIN
                         + " = KL, clip fraction, wall time", units="label"),
            "source": dict(definition="File the row is copied from, or how it was recomputed "
                                      "(with the bootstrap stream)",
                           units="text"),
            "note": dict(definition="superseded-solver flag or decomposition provenance",
                         units="text")}
    pack.table("T23", t, status="generated", sources=sources, script=_sc("build_t23"), notes=notes,
               docs=docs, tier="final and development",
               caption="Pilot 2 paired differences per (q, seed), first arm minus second arm, n = "
                       f"10 pairs (seeds {SEEDS_TXT}) per q and comparison; CI = 95% percentile "
                       "bootstrap of the mean difference.")
    _manual_check(pack, "T23", f"{P2A}/decomposition_residual_band_paired.csv",
                  "recomputed revised-decomposition pairs vs existing file", n_cmp, n_bad)
    for q in QS:
        for x, y in COMPARE2:
            comp = f"{SHORT2[x]}-{SHORT2[y]}"
            head = f"q = {q}, {SHORT2[x]} − {SHORT2[y]}"
            sub = t[(t["q"] == q) & (t["comparison"] == comp) & (t["source"] == rel)]
            pack.crosscheck("T23", sub, REP2, header_has=["label", "mean", "sd", "CI95"],
                            key_map={"label": "label"},
                            value_map={"mean": "mean", "sd": "sd", "median": "median", "min": "min",
                                       "max": "max"},
                            heading_has=head, label=f"paired table {head}")
            _check_paired_extras(pack, "T23", sub, REP2, head, "better (first arm)")
    chk = pd.DataFrame([{"q": q, "comparison": c, "metric": a,
                         "mean": dec_rows[(q, c, col)]["mean"],
                         "median": dec_rows[(q, c, col)]["median"],
                         "n<0": float(dec_rows[(q, c, col)]["n_neg"]),
                         "n>0": float(dec_rows[(q, c, col)]["n_pos"]),
                         "n=0": float(dec_rows[(q, c, col)]["n_zero"])}
                        for (q, c, col) in dec_rows
                        for a in [k for k, v in name.items() if v == col]])
    pack.crosscheck("T23", chk, REP2, header_has=["q", "comparison", "metric", "n<0", "CI95"],
                    key_map={"q": "q", "comparison": "comparison", "metric": "metric"},
                    value_map={"mean": "mean", "median": "median", "n<0": "n<0", "n>0": "n>0",
                               "n=0": "n=0"},
                    heading_has="Paired comparisons of the revised terms",
                    label="revised-term pairs (section 6)")
    tab = _report_table(REP2, "Paired comparisons of the revised terms", ["CI95", "n<0"])
    n_cmp = n_bad = 0
    for row in tab["rows"]:
        rec = dict(zip(tab["header"], row))
        mine = dec_rows[(int(rec["q"]), rec["comparison"], name[rec["metric"]])]
        n_cmp += 1
        if not _ci_cell_ok(mine["boot_ci95_lo"], mine["boot_ci95_hi"], rec["CI95"]):
            n_bad += 1
            pack.mismatch("T23", f"CI95 revised pair q={rec['q']} {rec['comparison']} "
                                 f"{rec['metric']}",
                          f"[{mine['boot_ci95_lo']!r}, {mine['boot_ci95_hi']!r}]",
                          f"{REP2} (line {tab['line']})", rec["CI95"])
    _manual_check(pack, "T23", REP2, "CI95 of the revised-term pairs (section 6)", n_cmp, n_bad)


# ----------------------------------------------------------------------------------------------
# T24: on-path definition and drift decomposition
# ----------------------------------------------------------------------------------------------

RULE_DEF = {
    "n_grid": "number of stage-2 grid nodes of the tier (D_2 = [-(100 + 2q), 100 + 2q])",
    "DeltaT_over_dw_n_on": "number of on-path stage-2 nodes: d in D_2 with |d - drift| < 2q (open "
                           "interval; nodes at exactly |d - drift| = 2q are off path); drift = "
                           "e_hat_1(0) - e_hat_1(-0)",
    "DeltaT_over_dw_n_off": "number of off-path stage-2 nodes (complement of the on-path set)",
    "DeltaT_over_dw_on_mass": "sum over on-path nodes of the normalized exact cell masses m_i = "
                              "F_xi(b_{i+1} - drift) - F_xi(b_i - drift), xi ~ Triangular(-2q, "
                              "2q), b = midpoints between nodes, outer bounds = domain edges; m "
                              "normalized by its total",
    "DeltaT_over_dw_off_mass": "sum over off-path nodes of the normalized cell masses (the cells "
                               "of the two nodes at exactly |d| = 2q carry this mass)",
    "cellmass_captured_total": "raw total of the cell masses before normalization (1 = the grid "
                               "captures the whole law)",
    "stage1_drift": "root drift e_hat_1(0) - e_hat_1(-0) at the final evaluation (u1000), effort "
                    "units",
    "stage1_drift_checkpoints": "root drift at every training-time checkpoint (u500-u1000, 21 per "
                                "run), effort units",
}
DRIFT_STATS = [("on_max", "max over on-path nodes of |e_hat_2(d) - e_hat_2^parent(d)|"),
               ("on_mean_cellmass_weighted", "on-path mean of |e_hat_2 - e_hat_2^parent| weighted "
                                             "by the normalized cell masses (renormalized over "
                                             "the on-path cells)"),
               ("on_mean_unweighted", "unweighted mean of |e_hat_2 - e_hat_2^parent| over the "
                                      "on-path nodes"),
               ("off_max", "max over off-path nodes of |e_hat_2 - e_hat_2^parent|"),
               ("off_mean_unweighted", "unweighted mean of |e_hat_2 - e_hat_2^parent| over the "
                                       "off-path nodes"),
               ("maxabs", "max over all D_2 nodes of |e_hat_2 - e_hat_2^parent|")]


def build_t24(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """T24: on-path rule (node counts, cell masses, drift) and the stage-2 drift decomposition."""
    rows: List[Dict[str, Any]] = []
    fv, ck = ctx["fv2"], ctx["ck2"]
    for q in QS:
        for tier in (DEV, FIN):
            vals: Dict[str, List[float]] = {k: [] for k in RULE_DEF
                                            if k != "stage1_drift_checkpoints"}
            for (qq, s, a), rec in fv.items():
                if qq != q:
                    continue
                sc = rec[tier]
                vals["n_grid"].append(sc["DeltaT_over_dw_n_on"] + sc["DeltaT_over_dw_n_off"])
                for k in vals:
                    if k != "n_grid":
                        vals[k].append(float(sc[k]))
            for k, v in vals.items():
                rows.append({"block": "on-path rule", "q": q, "arm": "all", "network": "",
                             "tier": tier, "metric": k,
                             "definition": RULE_DEF[k], **_stats(v)})
        cq = ck[ck["q"] == q]
        rows.append({"block": "on-path rule", "q": q, "arm": "all", "network": "", "tier": DEV,
                     "metric": "stage1_drift_checkpoints",
                     "definition": RULE_DEF["stage1_drift_checkpoints"],
                     **_stats(cq["stage1_drift"].to_numpy(dtype=float))})
    live_eq_cand_A = 0.0
    for q in QS:
        for arm in ARMS2:
            for net, key in (("candidate", "cand"), ("live network", "live")):
                for stat, dfn in DRIFT_STATS:
                    col = f"stage2_drift_{key}_{stat}"
                    v = [float(fv[(q, s, arm)]["drift_vs_parent"][col]) for s in SEEDS]
                    if arm == "A_joint" and key == "live":
                        cc = f"stage2_drift_cand_{stat}"
                        cand = [float(fv[(q, s, arm)]["drift_vs_parent"][cc]) for s in SEEDS]
                        live_eq_cand_A = max(live_eq_cand_A, C.max_abs_diff(v, cand))
                    who = (" (candidate: A = live network, B1/B2 = frozen snapshot)"
                           if key == "cand" else " (live network's own stage-2 output)")
                    rows.append({"block": "drift decomposition", "q": q, "arm": arm, "network": net,
                                 "tier": DEV, "metric": col, "definition": dfn + who, **_stats(v)})
    t = pd.DataFrame(rows)
    # checks: live drift of the frozen arms (final_table / drift_test), report section 2.6 medians
    p2 = ctx["p2"]
    keys_a = list(p2[["q", "seed", "arm"]].itertuples(index=False, name=None))
    fv_live = [fv[k]["drift_vs_parent"]["stage2_drift_live_maxabs"] for k in keys_a]
    live_ft = C.max_abs_diff(p2["live_stage2_drift_maxabs"], fv_live)
    fro = p2[p2["arm"] != "A_joint"]
    keys_f = list(fro[["q", "seed", "arm"]].itertuples(index=False, name=None))
    dt_live = [ctx["dt2"][k]["live_network_stage2_mean_maxabs_drift"] for k in keys_f]
    live_dt = C.max_abs_diff(fro["live_stage2_drift_maxabs"], dt_live)
    n_ck = len(ck)
    s1max = float(ck["stage1_drift"].abs().max())
    candmax = float(ck[ck["arm"] != "A_joint"]["stage2_drift_cand_maxabs"].abs().max())
    notes = ("Rule block: values of all 30 Pilot 2 runs per q (arms pooled) at u1000 from "
             "final_v2.json[tier] (functions utils/v2_metrics.py:onpath_mask, cell_masses, "
             f"onoff_split), plus the root drift in all {n_ck} training-time checkpoint rows "
             f"(v2_checkpoints.csv; max |drift| = {s1max:g}). Drift block: "
             "final_v2.json['drift_vs_parent'] (run/run_v2_stagewise.py:drift_scalars) on the "
             "development-tier D_2 grid, median/q25/q75/min/max over the 10 seeds; candidate = "
             "the evaluated stage-2 mapping (A: live network; B1/B2: frozen snapshot); live "
             "network = the network's own stage-2 output (not used for actions in B1/B2). A: live "
             f"= candidate (max abs diff {live_eq_cand_A:g}). Frozen candidate drift in every "
             f"B1/B2 checkpoint row: max {candmax:g}. live maxabs equals final_table.csv "
             f"live_stage2_drift_maxabs (max abs diff {live_ft:.2g}) and drift_test.json "
             f"live_network_stage2_mean_maxabs_drift (max abs diff {live_dt:.2g}).")
    docs = {"block": dict(definition="Part of the table: 'on-path rule' (definition and its node "
                                     "counts / cell masses) or 'drift decomposition' (stage-2 "
                                     "drift against the parent)",
                          units="label"),
            "network": dict(definition="Whose stage-2 output the drift is measured on: candidate "
                                       "(the evaluated policy's stage 2: A = live network, B1/B2 "
                                       "= frozen snapshot) or live network", units="label"),
            "definition": dict(definition="Definition of the row's quantity", units="text"),
            "tier": dict(definition="Verifier tier / grid of the row: " + TIER_TXT, units="label"),
            "metric": dict(definition="Quantity summarized (final_v2.json key; stage2_drift_* in "
                                      "effort units, raw)",
                           units="text")}
    pack.table("T24", t, status="generated",
               sources=[ctx["src_fv2"], ctx["src_ck2"], C.src(f"{P2A}/final_table.csv"),
                        ctx["src_dt2"], C.src("utils/v2_metrics.py")],
               script=_sc("build_t24"), notes=notes, docs=docs, tier="final and development",
               caption="On-path rule of the stage-2 split and the stage-2 drift decomposition, "
                       f"Pilot 2 at u1000, n = 10 seeds ({SEEDS_TXT}) per (q, arm) and n = 30 runs "
                       "per q in the rule block. Drift in effort units (raw), cell-mass weighted "
                       "and unweighted means.")
    lv = (fro.assign(arm=fro["arm"].map(SHORT2)).groupby(["q", "arm"])
          .agg(live=("live_stage2_drift_maxabs", "median")).reset_index())
    pack.crosscheck("T24", lv, REP2, header_has=["live_net_stage2_drift_median"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={"live_net_stage2_drift_median": "live"},
                    heading_has="Snapshot integrity",
                    label="median live-network stage-2 drift of the frozen arms (section 2.6)")


# ----------------------------------------------------------------------------------------------
# T25: location of Gmax_full
# ----------------------------------------------------------------------------------------------

def _loc_class(t: Sequence[int], d: Sequence[float], q: int) -> List[str]:
    """stage1 / stage2_onpath (|d*| < 2q) / stage2_offpath, as in tools/v2/pilot2_analysis.py."""
    return ["stage1" if int(ti) == 1
            else ("stage2_onpath" if abs(float(di)) < 2 * q else "stage2_offpath")
            for ti, di in zip(t, d)]


def build_t25(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """T25: location classes of Gmax_full (existing counts, final tier) and exact (t*, d*) pairs."""
    p2, ck = ctx["p2"], ctx["ck2"]
    locs = ("stage1", "stage2_offpath", "stage2_onpath")
    rows: List[Dict[str, Any]] = []
    found = {"final checkpoint u1000": f"{P2A}/gmax_location_final.csv",
             "21 training-time checkpoints u500-u1000": f"{P2A}/gmax_location_all_checkpoints.csv"}
    recomputed = {}
    p2c = p2.assign(cls=[_loc_class([t], [d], q)[0] for t, d, q in zip(p2["Gmax_full_t"],
                                                                       p2["Gmax_full_d"], p2["q"])])
    ckc = ck.assign(cls=[_loc_class([t], [d], q)[0] for t, d, q in zip(ck["Gmax_full_t"],
                                                                       ck["Gmax_full_d"], ck["q"])])
    recomputed["final checkpoint u1000"] = p2c.groupby(["q", "arm", "cls"]).size()
    recomputed["21 training-time checkpoints u500-u1000"] = ckc.groupby(["q", "arm", "cls"]).size()
    max_diff = 0
    for scope, rel in found.items():
        g = _csv(rel)
        for r in g.itertuples(index=False):
            n = int(sum(getattr(r, l) for l in locs))
            for l in locs:
                cnt = int(getattr(r, l))
                re_cnt = int(recomputed[scope].get((r.q, r.arm, l), 0))
                max_diff = max(max_diff, abs(cnt - re_cnt))
                rows.append({"block": "location class", "q": r.q, "arm": r.arm, "tier": DEV,
                             "scope": scope,
                             "location": l, "t_star": np.nan, "d_star": np.nan, "count": cnt,
                             "n": n, "source": rel})
    fin_cls = [_loc_class([t], [d], q)[0] for t, d, q in zip(p2[FT + "Gmax_full_t"],
                                                             p2[FT + "Gmax_full_d"], p2["q"])]
    pf = p2.assign(cls=fin_cls)
    for q in QS:
        for arm in ARMS2:
            g = pf[(pf["q"] == q) & (pf["arm"] == arm)]
            for l in locs:
                rows.append({"block": "location class", "q": q, "arm": arm, "tier": FIN,
                             "scope": "final checkpoint u1000",
                             "location": l, "t_star": np.nan, "d_star": np.nan,
                             "count": int((g["cls"] == l).sum()), "n": len(g),
                             "source": "final_v2.json['final'] (Gmax_full_t, Gmax_full_d)"})
    for tier, tc, dc, src in ((DEV, "Gmax_full_t", "Gmax_full_d", f"{P2A}/final_table.csv"),
                              (FIN, FT + "Gmax_full_t", FT + "Gmax_full_d",
                               "final_v2.json['final']")):
        for q in QS:
            for arm in ARMS2:
                g = p2[(p2["q"] == q) & (p2["arm"] == arm)]
                cnt = g.groupby([tc, dc]).size().reset_index(name="n_pair").sort_values(
                    ["n_pair", tc, dc], ascending=[False, True, True])
                for t_, d_, n_ in cnt[[tc, dc, "n_pair"]].itertuples(index=False, name=None):
                    rows.append({"block": "(t*, d*) pair", "q": q, "arm": arm, "tier": tier,
                                 "scope": "final checkpoint u1000",
                                 "location": _loc_class([t_], [d_], q)[0],
                                 "t_star": int(t_), "d_star": float(d_), "count": int(n_),
                                 "n": len(g), "source": src})
    t = pd.DataFrame(rows)
    t.insert(3, "arm_short", t["arm"].map(SHORT2))
    dif = ((p2["Gmax_full_t"] != p2[FT + "Gmax_full_t"])
           | (p2["Gmax_full_d"] != p2[FT + "Gmax_full_d"]))
    n_t = int((p2["Gmax_full_t"] != p2[FT + "Gmax_full_t"]).sum())
    dd = float((p2["Gmax_full_d"] - p2[FT + "Gmax_full_d"]).abs().max())
    n_cls = int((p2c["cls"].to_numpy() != pf["cls"].to_numpy()).sum())
    notes = ("Location classes: stage1 (t* = 1, always d* = 0), stage2_onpath (t* = 2, |d*| < 2q; "
             "the root drift is 0 in every checkpoint), stage2_offpath (t* = 2, |d*| >= 2q). Rows "
             "with a gmax_location_*.csv source are the existing counts (development tier); "
             "recounting them from final_table.csv and the 60 v2_checkpoints.csv files gives max "
             f"abs diff {max_diff}. Final-tier classes and all exact (t*, d*) pairs are generated "
             "from final_table.csv (development tier) and final_v2.json['final'] (final tier; d* "
             f"on the 2-step grid). The (t*, d*) of the two tiers differ in {int(dif.sum())} of 60 "
             f"runs: t* in {n_t}, d* by at most {dd:g} effort units; the location class differs in "
             f"{n_cls} run(s).")
    docs = {"block": dict(definition="Part of the table: 'location class' counts or exact "
                                     "'(t*, d*) pair' counts", units="label"),
            "scope": dict(definition="Which checkpoints are counted: the final checkpoint (u1000) "
                                     "of each run, or all 21 training-time checkpoints "
                                     "(u500-u1000 every 25 updates) of each run", units="text"),
            "location": dict(definition="stage1 (t* = 1), stage2_onpath (t* = 2 and |d*| < 2q) or "
                                        "stage2_offpath (t* = 2 and |d*| >= 2q)", units="label"),
            "t_star": dict(definition="Stage t* at which Gmax_full is attained (pair rows)",
                           units="stage index"),
            "d_star": dict(definition="Gap d* at which Gmax_full is attained (pair rows)",
                           units="effort units (gap d)"),
            "count": dict(definition="Number of runs (or checkpoints, see scope) in the row's "
                                     "class or pair", units="count"),
            "n": dict(definition="Number of runs (or checkpoints) of the (q, arm, tier, scope) "
                                 "group", units="count"),
            "tier": dict(definition="Verifier tier of the evaluation: " + TIER_TXT, units="label"),
            "arm_short": SUMMARY_DOCS["arm_short"]}
    pack.table("T25", t, status="generated",
               sources=[C.src(rel) for rel in found.values()]
               + [C.src(f"{P2A}/final_table.csv"), ctx["src_fv2"], ctx["src_ck2"]],
               script=_sc("build_t25"), notes=notes, docs=docs, tier="final and development",
               caption="Where Gmax_full is attained in Pilot 2: counts per (q, arm), n = 10 runs "
                       f"(seeds {SEEDS_TXT}) at u1000 or 210 training-time checkpoints per (q, "
                       "arm).")
    for k, scope in enumerate(found):
        g = _csv(found[scope])
        pack.crosscheck("T25", g, REP2,
                        header_has=["q", "arm", "stage1", "stage2_offpath", "stage2_onpath"],
                        key_map={"q": "q", "arm": "arm"}, value_map={l: l for l in locs},
                        heading_has="Location of", tables=[k],
                        label=f"location counts, {scope} (section 2.4)")


# ----------------------------------------------------------------------------------------------
# F06: Pilot 2 learning curves
# ----------------------------------------------------------------------------------------------

def _curve_grid(panels: List[Tuple[str, pd.DataFrame, str, str]], arms: Sequence[str],
                arm_label: Dict[str, str], height: float,
                zero_rows: Sequence[str]) -> Tuple[Any, pd.DataFrame]:
    """Median + IQR curves; rows = panels ``(metric, frame, column, ylabel)``, columns = q."""
    fig, axes = style.new_figure(len(panels), 2, height=height, sharex=True)
    out = []
    for i, (name, frame, col, ylab) in enumerate(panels):
        for j, q in enumerate(QS):
            ax = axes[i, j]
            for arm in arms:
                g = frame[(frame["q"] == q) & (frame["arm"] == arm)][["seed", "update", col]]
                n = g["seed"].nunique()
                res = style.median_iqr_curves(ax, g, "update", col, style.ARM_COLORS[arm],
                                              style.label_n(arm_label[arm], n), marker=MARK[arm],
                                              extra={"q": q, "arm": arm, "panel": name})
                for ln in ax.get_lines()[-1:]:
                    ln.set_markersize(2.6)
                out.append(res)
            if name in zero_rows:
                ax.axhline(0.0, color=style.REF, lw=0.6, zorder=0)
            if i == 0:
                ax.set_title(f"q = {q}")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == len(panels) - 1:
                ax.set_xlabel("global update")
            ax.set_xlim(415, 1010)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=len(arms), frameon=False)
    data = pd.concat(out, ignore_index=True)
    data = data.rename(columns={"metric": "source_column"})
    return fig, data[["q", "arm", "panel", "source_column", "update", "median", "q25", "q75",
                      "min", "max", "n"]]


def build_f06(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """F06: Pilot 2 learning curves (exports every 25 updates), median and IQR per arm and q."""
    cur, dec = ctx["cur2"], ctx["dec2"]
    w = dec[dec["source"] == "weights"]
    panels = [("stage-2 drift on path, max", cur, "drift_on_max",
               "stage-2 drift,\non-path max (effort)"),
              ("stage-2 drift off path, max", cur, "drift_off_max",
               "stage-2 drift,\noff-path max (effort)"),
              ("stage-1 relative error", cur, "stage1_rel_err_signed",
               "stage-1 rel. error\n(signed, /e1*)"),
              ("Gmax_full", cur, "Gmax_full_over_dw", "Gmax_full / ΔW\n(dev tier)"),
              ("EXP_root", cur, "EXP_root_over_dw", "EXP_root / ΔW\n(dev tier)"),
              ("dReach", cur, "dReach_over_dw", "dReach / ΔW\n(dev tier)"),
              ("learning term (revised)", w, "learning_rel",
               "learning term\n" + r"$(\hat e_1-\tilde e_1)/e_1^*$"),
              ("inherited term (revised)", w, "inherited_rel",
               "inherited term\n" + r"$(\tilde e_1-e_1^*)/e_1^*$")]
    labels = {a: SHORT2[a] + {"A_joint": " joint", "B1_frozen_allnorm": " frozen, all-rows norm.",
                              "B2_frozen_s1norm": " frozen, stage-1-rows norm."}[a] for a in ARMS2}
    zero_rows = ("stage-1 relative error", "learning term (revised)", "inherited term (revised)")
    fig, data = _curve_grid(panels, ARMS2, labels, 12.6, zero_rows)
    # checks: u1000 medians against final_table.csv / the revised decomposition rows
    ft = ctx["p2"]
    pairs = {"drift_on_max": "stage2_drift_cand_on_max",
             "drift_off_max": "stage2_drift_cand_off_max",
             "stage1_rel_err_signed": "stage1_rel_err_signed",
             "Gmax_full_over_dw": "Gmax_full_over_dw",
             "EXP_root_over_dw": "EXP_root_over_dw", "dReach_over_dw": "dReach_over_dw",
             "learning_rel": "dec_learning_rel", "inherited_rel": "dec_inherited_rel"}
    checks = []
    for c_col, f_col in pairs.items():
        sel = (data["source_column"] == c_col) & (data["update"] == 1000)
        a = data[sel].sort_values(["q", "arm"])["median"]
        b = ft.groupby(["q", "arm"])[f_col].median().sort_index()
        checks.append(f"u1000 median of {c_col} vs final_table/D02 {f_col}: max abs diff "
                      f"{C.max_abs_diff(a, b):.2g}")
    n_out = int((~w["e1_inside_sweep"].astype(bool)).sum())
    checks.append(f"{n_out} plotted decomposition rows (u425/u450, q=60, B1/B2) have e_hat_1 "
                  "outside the parent sweep (their terms are defined: e~1 does not depend on "
                  "e_hat_1)")
    caption = ("Pilot 2 learning curves: median (line) and IQR (band) over n = 10 seeds "
               f"({SEEDS_TXT}) per arm, at the weight exports every 25 updates (global "
               "u425-u1000), q = 50 (left) and q = 60 (right). Rows: stage-2 drift |e_hat_2 - "
               "e_hat_2^parent| of the candidate's stage-2 mapping on the development-tier D_2 "
               "grid, on-path max and off-path max (effort units, raw; A = live network, B1/B2 = "
               "frozen snapshot); stage-1 relative error (signed, /e1*(0), tier-independent); "
               "Gmax_full, EXP_root and dReach (/Delta W, development tier: state step 4, effort "
               "step 1, GL 16 per half); learning and inherited terms of the revised "
               "residual-band decomposition (/e1*(0), final tier). Sources: "
               "results/v2_pilots/pilot2/analysis/curves_weights_every25.csv and "
               "decomposition_residual_band.csv (rows source = weights). B1 and B2 coincide in "
               "the drift rows (both 0) and in the inherited-term row (same frozen stage 2). "
               "Reference line at 0 in the signed rows. The existing "
               "reports/v2/figures/pilot2/curves_q50.png and curves_q60.png plot the same first "
               "six rows; their last two panels show the superseded solver's terms.")
    pack.figure("F06", fig, data, status="regenerated",
                sources=[C.src(f"{P2A}/curves_weights_every25.csv"),
                         C.src(f"{P2A}/decomposition_residual_band.csv"),
                         C.src(f"{P2A}/final_table.csv"),
                         C.src("reports/v2/figures/pilot2/curves_q50.png"),
                         C.src("reports/v2/figures/pilot2/curves_q60.png")],
                script=_sc("build_f06"), caption=caption, tier="final and development",
                checks=checks,
                notes="Re-plotted from the data of the existing figure (rows 1-6); rows 7-8 use "
                      "the revised decomposition (decomposition_residual_band.csv) instead of the "
                      "superseded terms of the existing figure.",
                docs=_curve_docs())


def _curve_docs() -> Dict[str, Any]:
    """Docs of the learning-curve data columns."""
    return {"panel": dict(definition="Figure row (quantity plotted)", units="label"),
            "source_column": dict(definition="Column of the source CSV that is summarized",
                                  units="text"),
            "update": dict(definition="Global update of the weight export (every 25 updates, "
                                      "u425-u1000)", units="updates"),
            "median": dict(definition="Across-seed median at the update (plotted line)",
                           units="as the quantity"),
            "q25": dict(definition="Across-seed 25th percentile (lower edge of the plotted band)",
                        units="as the quantity"),
            "q75": dict(definition="Across-seed 75th percentile (upper edge of the plotted band)",
                        units="as the quantity"),
            "min": dict(definition="Across-seed minimum (not plotted)", units="as the quantity"),
            "max": dict(definition="Across-seed maximum (not plotted)", units="as the quantity"),
            "n": dict(definition="Number of seeds at the update", units="count")}


# ----------------------------------------------------------------------------------------------
# F07: stage-2 mapping at the end of Phase B against the parent
# ----------------------------------------------------------------------------------------------

def build_f07(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """F07: e_hat_2(d) at u1000 (A, B2) against the parent (Pilot 1 expected, u400) and e2*(d)."""
    rec = ctx["rec"]
    pidx = ctx["p2"].set_index(["q", "seed", "arm"])
    fig, axes = style.new_figure(2, 2, height=6.0, sharex="col")
    rows: List[pd.DataFrame] = []
    b2_zero = 0.0
    g2_check = 0.0
    e2_at0 = 0.0
    for j, q in enumerate(QS):
        D = rec[(q, SEEDS[0], "parent")]["recovery_d_grid"]
        par = np.vstack([rec[(q, s, "parent")]["recovery_e2"] for s in SEEDS])
        g2 = rec[(q, SEEDS[0], "parent")]["recovery_g2"]
        game = ctx["rc2"][(q, SEEDS[0], "A_joint")]["record"]["game"]
        g2_ref = g2_two_stage(D, q, game["w_h"], game["w_l"], game["k"], game["e_max"])
        g2_check = max(g2_check, C.max_abs_diff(g2, g2_ref))
        for s in SEEDS:
            for a in ARMS2:
                if not np.array_equal(rec[(q, s, a)]["recovery_d_grid"], D):
                    raise ValueError("recovery grids differ")
        ch = {a: np.vstack([rec[(q, s, a)]["recovery_e2"] for s in SEEDS])
              for a in ("A_joint", "B2_frozen_s1norm")}
        b2_zero = max(b2_zero, float(np.max(np.abs(ch["B2_frozen_s1norm"] - par))),
                      float(np.max(np.abs(np.vstack([rec[(q, s, "B1_frozen_allnorm")]["recovery_e2"]
                                                     for s in SEEDS]) - par))))
        i0 = int(np.argmin(np.abs(D)))
        for a in ARMS2:
            for s in SEEDS:
                g20 = float(ctx["fv2"][(q, s, a)]["development"]["g2_at_0"])
                pk = (rec[(q, s, a)]["recovery_e2"][i0] - g20) / g20
                e2_at0 = max(e2_at0,
                             abs(pk - float(pidx.loc[(q, s, a), "stage2_peak_rel_err_signed"])))
        two_q = 2.0 * q
        ax0, ax1 = axes[0, j], axes[1, j]
        for ax in (ax0, ax1):
            ax.axvspan(-two_q, two_q, color=style.GRID, alpha=0.6, lw=0, zorder=0,
                       label="on-path region |d| < 2q" if (ax is ax0) else None)
        ax0.plot(D, g2, color=style.REF, lw=1.0, label="e2*(d), closed form", zorder=4)
        pm = np.median(par, axis=0)
        ax0.plot(D, pm, color=style.INK2, lw=3.2, alpha=0.35,
                 label=style.label_n("parent (Pilot 1 expected, u400), median", 10),
                 zorder=2)
        rows.append(pd.DataFrame({"q": q, "panel": "e_hat_2 at u1000", "series": "e2_star", "d": D,
                                  "value": g2}))
        rows.append(pd.DataFrame({"q": q, "panel": "e_hat_2 at u1000", "series": "parent_median",
                                  "d": D, "value": pm}))
        for a, lw in (("B2_frozen_s1norm", 1.2), ("A_joint", 1.5)):
            m = np.median(ch[a], axis=0)
            ax0.plot(D, m, color=style.ARM_COLORS[a], lw=lw, marker=MARK[a], markevery=60,
                     markersize=3, label=style.label_n(f"{SHORT2[a]}, u1000, median", 10), zorder=3)
            rows.append(pd.DataFrame({"q": q, "panel": "e_hat_2 at u1000",
                                      "series": f"{SHORT2[a]}_median", "d": D, "value": m}))
            dlt = ch[a] - par
            md = np.median(dlt, axis=0)
            lo, hi = np.percentile(dlt, 25, axis=0), np.percentile(dlt, 75, axis=0)
            style.band_plot(ax1, D, md, lo, hi, color=style.ARM_COLORS[a], lw=lw, marker=MARK[a],
                            markevery=60, markersize=3, zorder=3)
            for nm, v in (("median", md), ("q25", lo), ("q75", hi)):
                rows.append(pd.DataFrame({"q": q, "panel": "change from parent",
                                          "series": f"{SHORT2[a]}_{nm}", "d": D, "value": v}))
        ax1.axhline(0.0, color=style.REF, lw=0.6, zorder=1)
        ax0.set_title(f"q = {q}")
        ax1.set_xlabel("gap d at stage 2 (effort units)")
        ax0.set_xlim(D[0], D[-1])
    axes[0, 0].set_ylabel(r"$\hat e_2(d)$" + " (effort units, raw)")
    axes[1, 0].set_ylabel(r"$\hat e_2(d)-\hat e_2^{parent}(d)$" + "\n(effort units)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3, frameon=False)
    data = pd.concat(rows, ignore_index=True)
    checks = ["B2 (and B1) e_hat_2 equals the parent's on all recovery nodes of all 20 pairs: max "
              f"abs diff {b2_zero:g}",
              "recovery_g2 equals utils/theory_multistage.py:g2_two_stage with the run's game "
              f"constants: max abs diff {g2_check:.2g}",
              "(e_hat_2(0) - e2*(0))/e2*(0) from the plotted arrays equals final_table.csv "
              f"stage2_peak_rel_err_signed: max abs diff {e2_at0:.2g}",
              "parent states: SHA-256 of "
              "results/v2_pilots/pilot1/q*/seed*/expected/state_end_A.pt equals the parent_sha256 "
              "recorded in all 60 Pilot 2 run_config.json files"]
    _check_parent_hashes(ctx, "rc2")
    caption = ("Pilot 2 stage-2 mapping at the end of Phase B (global u1000) against its parent. "
               "Top: across-seed median of e_hat_2(d) (Beta mean, recovery grid step 0.5 on D_2; "
               f"n = 10 seeds, {SEEDS_TXT}) for A (joint) and B2 (frozen, stage-1-rows "
               "normalization), the parent's median (Pilot 1 expected at u400, the restored "
               "Phase-A state) and the closed form e2*(d). Bottom: e_hat_2^child(d) - "
               "e_hat_2^parent(d) per seed, median (line) and IQR (band). Shaded: on-path region "
               "|d| < 2q (root drift 0). The B2 curve equals the parent's (frozen stage 2; "
               "difference exactly 0). Source: final_development.npz (recovery_d_grid, "
               "recovery_e2, recovery_g2) of the Pilot 2 runs and of "
               "results/v2_pilots/pilot1/q*/seed*/expected; tier-independent.")
    docs = {"panel": dict(definition="Figure row: 'e_hat_2 at u1000' (top) or 'change from "
                                     "parent' (bottom)", units="label"),
            "series": dict(definition="Plotted series: e2_star (closed form), parent_median, "
                                      "A_median / B2_median (top); A_/B2_ median, q25, q75 of the "
                                      "child - parent difference over seeds (bottom)",
                           units="label"),
            "d": dict(definition="Stage-2 gap d (recovery grid, step 0.5)",
                      units="effort units (gap d)"),
            "value": dict(definition="Plotted value (effort units, raw; bottom row: difference "
                                     "child - parent)",
                          units="effort units", tier=IND)}
    pack.figure("F07", fig, data, status="generated",
                sources=[ctx["src_npz2"], ctx["src_npz1"], ctx["src_fv2"],
                         C.src(f"{P2A}/final_table.csv"),
                         ctx["src_rc2"], ctx["src_par"], C.src("utils/theory_multistage.py")],
                script=_sc("build_f07"), caption=caption, tier="tier-independent", checks=checks,
                docs=docs)


def _check_parent_hashes(ctx: Dict[str, Any], key: str) -> None:
    """Raise unless each child run_config parent_sha256 equals the hash of the parent state."""
    par = {s.path: s.sha256 for s in ctx["src_par"].files}
    for (q, s, a), rc in ctx[key].items():
        p = f"{P1}/q{q}/seed{s}/expected/state_end_A.pt"
        if par.get(p) != rc["parent_sha256"]:
            raise ValueError(f"parent hash mismatch for {q}/{s}/{a}")


# ----------------------------------------------------------------------------------------------
# F08: advantage SD ratio
# ----------------------------------------------------------------------------------------------

def build_f08(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """F08: SD(adv, stage-1 rows) / SD(adv, all rows) per update, B1 vs B2 (median and IQR)."""
    rel = f"{P2A}/per_update_adv_stats.csv"
    up = _csv(rel, usecols=["q", "arm", "seed", "update", "sd_ratio_s1_over_all"])
    summ = _csv(f"{P2A}/adv_sd_ratio_summary.csv")
    fig, axes = style.new_figure(2, 2, height=5.0)
    out = []
    for j, q in enumerate(QS):
        for i, (u0, title) in enumerate(((401, "all updates (u401-u1000)"), (402, "u402-u1000"))):
            ax = axes[i, j]
            for arm in ARMS2:
                g = up[(up["q"] == q) & (up["arm"] == arm) & (up["update"] >= u0)]
                if arm == "A_joint":
                    st = [dict(update=u, metric="sd_ratio_s1_over_all",
                               **dict(zip(("median", "q25", "q75", "min", "max"),
                                          style.median_iqr(x["sd_ratio_s1_over_all"]))),
                               n=len(x)) for u, x in g.groupby("update")]
                    res = pd.DataFrame(st)
                    res["q"], res["arm"], res["plotted"] = q, arm, False
                else:
                    scope = "all-rows" if arm.startswith("B1") else "stage-1-rows"
                    lab = style.label_n(f"{SHORT2[arm]} frozen, {scope} norm.", g["seed"].nunique())
                    res = style.median_iqr_curves(ax, g, "update", "sd_ratio_s1_over_all",
                                                  style.ARM_COLORS[arm], lab,
                                                  lw=1.2 if arm.startswith("B1") else 1.0,
                                                  extra={"q": q, "arm": arm})
                    res["plotted"] = True
                res["view"] = title
                out.append(res)
            ax.set_title(f"q = {q}, {title}")
            if j == 0:
                ax.set_ylabel("SD(adv, stage-1 rows) /\nSD(adv, all rows)")
            if i == 1:
                ax.set_xlabel("global update")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False)
    data = pd.concat(out, ignore_index=True)[["q", "arm", "view", "plotted", "update", "median",
                                              "q25", "q75", "min", "max", "n"]]
    # checks against the existing summary (pandas describe over 6000 updates x seeds per (q, arm))
    mine = up.groupby(["q", "arm"])["sd_ratio_s1_over_all"].agg(
        count="count", mean="mean", std="std", min="min",
        p25=lambda x: float(np.percentile(x, 25)), p50="median",
        p75=lambda x: float(np.percentile(x, 75)),
        max="max").reset_index()
    sm = summ.merge(mine, on=["q", "arm"])
    col_pairs = (("count_x", "count_y"), ("mean_x", "mean_y"), ("std_x", "std_y"),
                 ("min_x", "min_y"), ("25%", "p25"), ("50%", "p50"), ("75%", "p75"),
                 ("max_x", "max_y"))
    dmax = max(C.max_abs_diff(sm[a], sm[b]) for a, b in col_pairs)
    u401 = up[up["update"] == 401].groupby("q")["sd_ratio_s1_over_all"].median()
    after = up[up["update"] >= 402].groupby("q")["sd_ratio_s1_over_all"].min()
    checks = ["per-(q, arm) count/mean/std/min/quartiles/max over all updates recomputed from "
              f"{rel} equal adv_sd_ratio_summary.csv: max abs diff {dmax:.2g}",
              f"median over runs at u401: q50 {u401[50]:.4f}, q60 {u401[60]:.4f} (identical in the "
              f"three arms); minimum from u402 on: q50 {after[50]:.4f}, q60 {after[60]:.4f}"]
    caption = ("Ratio of the advantage SD over stage-1 rows to the advantage SD over all rows at "
               "every Phase B update (global u401-u1000), Pilot 2 arms B1 (frozen, all-rows "
               "normalization) and B2 (frozen, stage-1-rows normalization): median (line) and IQR "
               f"(band) over n = 10 seeds ({SEEDS_TXT}), q = 50 (left) and q = 60 (right). Top "
               "row: all updates; bottom row: from u402 on (y range of the IQR). Source: "
               "results/v2_pilots/pilot2/analysis/per_update_adv_stats.csv (C3 statistics of "
               "v2_updates.csv); the A_joint values are in the data file (plotted = False). Tier "
               "n/a (training statistics).")
    docs = {"view": dict(definition="Figure row: all updates (top) or u402-u1000 (bottom)",
                         units="label"),
            "plotted": dict(definition="Whether the series is drawn (A_joint is listed, not drawn)",
                            units="bool"),
            "update": dict(definition="Global Phase B update", units="updates"),
            "median": dict(definition="Across-seed median of the SD ratio at the update",
                           units="ratio"),
            "q25": dict(definition="Across-seed 25th percentile (band lower edge)", units="ratio"),
            "q75": dict(definition="Across-seed 75th percentile (band upper edge)", units="ratio"),
            "min": dict(definition="Across-seed minimum (not drawn)", units="ratio"),
            "max": dict(definition="Across-seed maximum (not drawn)", units="ratio"),
            "n": dict(definition="Number of seeds", units="count")}
    pack.figure("F08", fig, data, status="regenerated",
                sources=[C.src(rel), C.src(f"{P2A}/adv_sd_ratio_summary.csv"),
                         C.src("reports/v2/figures/pilot2/adv_sd_ratio.png")],
                script=_sc("build_f08"), caption=caption, tier="n/a", checks=checks, docs=docs)
    chk = mine.rename(columns={"p25": "25%", "p50": "50%", "p75": "75%"})
    chk["count"] = chk["count"].astype(float)
    pack.crosscheck("F08", chk, REP2, header_has=["q", "arm", "count", "mean", "std", "25%"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={c: c for c in ("count", "mean", "std", "min", "25%", "50%", "75%",
                                              "max")},
                    heading_has="Normalization scope", label="SD-ratio summary (section 2.5)")
    pack.mismatch("F08", "SD ratio at the first Phase B update u401 (median over the 10 runs per "
                         "q, identical in all arms)",
                  f"q50 {float(u401[50])!r}; q60 {float(u401[60])!r}",
                  f"{REP2} (section 4, 'Normalization scope')",
                  "the SD ratio is about √2 in every arm and at every update",
                  f"prose claim; true from u402 on (min q50 {after[50]:.4f}, q60 {after[60]:.4f}); "
                  "at u401 the ratio is 0.90-1.32 in every run")


# ----------------------------------------------------------------------------------------------
# T26: Pilot 3 reproducibility
# ----------------------------------------------------------------------------------------------

def build_t26(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """T26: Pilot 3 stochastic arm vs Pilot 2 B2 (bit identity) and the inherited-term identity."""
    rel, rel2 = f"{P3A}/reproducibility_vs_pilot2_B2.csv", f"{P3A}/inherited_identity.csv"
    rp = _csv(rel)
    ih = _csv(rel2)
    m = rp["run"].str.extract(r"^q(\d+)/seed(\d+)/(.+)$")
    rp.insert(1, "q", m[0].astype(int))
    rp.insert(2, "seed", m[1].astype(int))
    rp.insert(3, "arm", m[2])
    t = rp.merge(ih, on=["q", "seed"], how="left", validate="one_to_one")
    bool_cols = [c for c in t.columns if c.endswith("_identical")]
    n_all = int(t[bool_cols].astype(bool).all(axis=1).sum())
    _check_parent_hashes(ctx, "rc3")
    same_cfg = ("flags", "q", "seed", "mode", "fixed_budget", "parent_sha256")
    for q in QS:
        for s in SEEDS:
            a, b = ctx["rc3"][(q, s, "B2_frozen_s1norm")], ctx["rc2"][(q, s, "B2_frozen_s1norm")]
            if any(a[k] != b[k] for k in same_cfg):
                raise ValueError(f"Pilot 3 stochastic and Pilot 2 B2 configs differ for q={q} "
                                 f"seed={s}")
    # independent read: Pilot 3 stochastic analysis rows equal Pilot 2 B2 rows
    c3, c2 = ctx["cur3"], ctx["cur2"]
    a = c3[c3["arm"] == "B2_frozen_s1norm"].set_index(["q", "seed", "update"]).sort_index()
    b = c2[c2["arm"] == "B2_frozen_s1norm"].set_index(["q", "seed", "update"]).sort_index()
    cur_cols = ("stage1_rel_err_signed", "Gmax_full_over_dw", "EXP_root_over_dw")
    cd = max(C.max_abs_diff(a[c], b.loc[a.index, c]) for c in cur_cols)
    f3 = ctx["p3"][ctx["p3"]["arm"] == "B2_frozen_s1norm"].set_index(["q", "seed"]).sort_index()
    f2 = ctx["p2"][ctx["p2"]["arm"] == "B2_frozen_s1norm"].set_index(["q", "seed"]).sort_index()
    fin_cols = ("e1_at_0", "stage1_rel_err_signed", "Gmax_full_over_dw", "EXP_root_over_dw",
                "dReach_over_dw", FT + "Gmax_full_over_dw", FT + "EXP_root_over_dw")
    fd = max(C.max_abs_diff(f3[c], f2.loc[f3.index, c]) for c in fin_cols)
    notes = (f"n = 20 runs (q in {{50, 60}} x seeds {SEEDS_TXT}); {n_all}/20 identical in every "
             "compared field. Compared (tools/v2/pilot3_analysis.py:reproducibility, Pilot 3 at "
             "cd760fd vs Pilot 2 at 1791687): train_history.json history (600 updates, exact "
             "equality), verifier_calls (exact, time_sec removed), stability log, "
             "checkpoint_weights.npz (final weights), every weight export weights/u*.npz (24 per "
             "run), state_end_B.pt actor, critic, lagged opponent and frozen snapshot tensors. "
             "The column name end_state_actor_critic_opt_identical says 'opt', but the code "
             "compares the actor, critic, opponent and frozen tensors only, not the optimizer "
             "states. Joined from inherited_identity.csv: frozen snapshot tensors bit-identical "
             "between the stochastic and mean arms of each (q, seed), and the inherited term's "
             "values in both arms. Pack check: the stochastic rows of pilot3 "
             f"curves_weights_every25.csv equal the Pilot 2 B2 rows (max abs diff {cd:g} over 24 "
             "exports x 20 runs x 3 metrics); the u1000 values in final_table.csv / final_v2.json "
             f"equal Pilot 2 B2 (max abs diff {fd:g}); the parent_sha256 in all 40 Pilot 3 "
             "run_config.json files equals the SHA-256 of "
             "results/v2_pilots/pilot1/q*/seed*/expected/state_end_A.pt; run_config.json flags, "
             "q, seed, mode, fixed_budget and parent_sha256 are equal for each compared pair.")
    docs = {
        "run": dict(definition="Run directory relative to results/v2_pilots/pilot3 (and to pilot2 "
                               "for the compared run)",
                    units="path"),
        "history_identical": dict(definition="train_history.json 'history' (per-update training "
                                             "records) exactly equal",
                                  units="bool"),
        "verifier_calls_identical": dict(definition="train_history.json 'verifier_calls' equal "
                                                    "after removing time_sec",
                                         units="bool"),
        "stability_identical": dict(definition="train_history.json 'stability' log exactly equal",
                                    units="bool"),
        "final_weights_identical": dict(definition="checkpoint_weights.npz: every array equal "
                                                   "(final actor and critic)",
                                        units="bool"),
        "weight_exports_identical": dict(definition="same number of weights/u*.npz exports and "
                                                    "every array of every export equal",
                                         units="bool"),
        "end_state_actor_critic_opt_identical": dict(definition="state_end_B.pt: actor, critic, "
                                                                "lagged opponent and frozen "
                                                                "snapshot tensors equal (despite "
                                                                "the name, optimizer states are "
                                                                "not compared)",
                                                     units="bool"),
        "n_updates": dict(definition="Number of Phase B updates in the compared history",
                          units="updates"),
        "snapshots_bit_identical": dict(definition="Frozen snapshot tensors bit-identical between "
                                                   "the stochastic and mean arms of the (q, seed) "
                                                   "(inherited_identity.csv)",
                                        units="bool"),
        "inherited_rel_values": dict(definition="Distinct inherited-term values (revised "
                                                "decomposition, all rows) per arm for the (q, "
                                                "seed), as written in inherited_identity.csv",
                                     units="fraction of e_1*(0)", tier=FIN),
        "arm": dict(definition="Pilot 3 arm compared (B2_frozen_s1norm = stochastic continuation)",
                    units="label"),
    }
    pack.table("T26", t, status="regenerated",
               sources=[C.src(rel), C.src(rel2), C.src(f"{P3A}/curves_weights_every25.csv"),
                        C.src(f"{P2A}/curves_weights_every25.csv"), C.src(f"{P3A}/final_table.csv"),
                        C.src(f"{P2A}/final_table.csv"), ctx["src_rc3"], ctx["src_par"]],
               script=_sc("build_t26"), notes=notes, docs=docs, tier="n/a",
               caption="Pilot 3 stochastic arm against Pilot 2 B2 (same parent, same flags, same "
                       "seeds), one row per run; values of reproducibility_vs_pilot2_B2.csv "
                       "joined with inherited_identity.csv.")
    # report section 2 tables: booleans and n_updates
    n_cmp = n_bad = 0
    for k, keycols in ((0, ["run"]), (1, ["q", "seed"])):
        tab = [x for x in C.parse_md_tables(REP3)
               if x["heading"].startswith("2. Reproducibility")][k]
        for row in tab["rows"]:
            rec = dict(zip(tab["header"], row))
            sel = t
            for kc in keycols:
                sel = sel[sel[kc].astype(str) == rec[kc]]
            if len(sel) != 1:
                continue
            for c in tab["header"]:
                if c in keycols or c not in t.columns:
                    continue
                n_cmp += 1
                if str(sel.iloc[0][c]) != rec[c]:
                    n_bad += 1
                    pack.mismatch("T26", f"{c} [{rec[keycols[0]]}]", sel.iloc[0][c],
                                  f"{REP3} (line {tab['line']})", rec[c])
    _manual_check(pack, "T26", REP3, "reproducibility and snapshot-identity tables (section 2)",
                  n_cmp, n_bad)


# ----------------------------------------------------------------------------------------------
# T27: Pilot 3 final medians
# ----------------------------------------------------------------------------------------------

T27_SPECS = [
    ("e1_at_0", "e1_at_0", IND, "num"),
    ("stage1_rel_err_signed", "stage1_rel_err_signed", IND, "num"),
    ("stage1_rel_err_abs", "stage1_rel_err_abs", IND, "num"),
    ("learning_rel", "learning_rel", FIN, "num"),
    ("learning_rel_abs", "learning_rel_abs", FIN, "num"),
    ("inherited_rel", "inherited_rel", FIN, "num"),
    ("inherited_rel_abs", "inherited_rel_abs", FIN, "num"),
    ("learning_contains_0", "learning_contains_0", FIN, "count"),
    ("inherited_contains_0", "inherited_contains_0", FIN, "count"),
    ("within_run_sd_e1_last5", "within_run_sd_e1_last5", IND, "num"),
    ("within_run_range_e1_last5", "within_run_range_e1_last5", IND, "num"),
    ("sigma_effort_at_0_t1", "sigma_effort_at_0_t1", IND, "num"),
    ("Gmax_full_over_dw", "Gmax_full_over_dw", DEV, "num"),
    ("Gmax_full_over_dw", FT + "Gmax_full_over_dw", FIN, "num"),
    ("EXP_root_over_dw", "EXP_root_over_dw", DEV, "num"),
    ("EXP_root_over_dw", FT + "EXP_root_over_dw", FIN, "num"),
    ("dReach_over_dw", "dReach_over_dw", DEV, "num"),
    ("dReach_over_dw", FT + "dReach_over_dw", FIN, "num"),
    ("Deltamax_all_over_dw", "Deltamax_all_over_dw", DEV, "num"),
    ("Deltamax_all_over_dw", FT + "Deltamax_all_over_dw", FIN, "num"),
    ("dFull_over_dw", "dFull_over_dw", DEV, "num"),
    ("dFull_over_dw", FT + "dFull_over_dw", FIN, "num"),
]


def build_t27(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """T27: Pilot 3 final medians per arm and q, including the decomposition terms, both tiers."""
    p3 = ctx["p3"]
    t = _with_argmax(pd.DataFrame(_summary_rows(p3, ["q", "arm"], T27_SPECS)), p3, SHORT3)
    fv = ctx["fv3"]
    dev_cols = ("Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw",
                "dFull_over_dw", "stage1_rel_err_signed")
    dev_eq = max(abs(float(r[c]) - float(fv[(r["q"], r["seed"], r["arm"])]["development"][c]))
                 for _, r in p3.iterrows() for c in dev_cols)
    w = _dec_final(ctx["dec3"]).set_index(["q", "seed", "arm"])
    pi = p3.set_index(["q", "seed", "arm"])
    dec_cols = ("e1_at_0", "learning_rel", "learning_rel_lo", "learning_rel_hi", "inherited_rel")
    dec_eq = max(C.max_abs_diff(pi[c], w.loc[pi.index, c]) for c in dec_cols)
    n_last = sorted(set(p3["n_last5"]))
    n_excl = int((~p3["learning_contains_0"].astype(bool)).sum())
    notes = ("Median, q25, q75 (numpy linear interpolation), min, max and n over the 10 seeds "
             f"({SEEDS_TXT}) per (q, arm) at global u1000. Development-tier values from "
             "results/v2_pilots/pilot3/analysis/final_table.csv (last training-time checkpoint; "
             f"equal to final_v2.json['development'] within {dev_eq:.2g}); final-tier values from "
             "final_v2.json['final']. learning/inherited terms: residual-band decomposition with "
             "the parent band (final tier); final_table.csv equals the u1000 weight-export rows "
             f"of decomposition_residual_band.csv (max abs diff {dec_eq:g}). Within-run SD (ddof = "
             "1) and range of e_hat_1(0) over the weight exports u900, 925, 950, 975, 1000 "
             f"(n_last5 = {n_last}). Learning band excludes 0 in {n_excl} of 40 final rows.")
    sources = [C.src(f"{P3A}/final_table.csv"), ctx["src_fv3"],
               C.src(f"{P3A}/decomposition_residual_band.csv")]
    pack.table("T27", t, status="generated", sources=sources, script=_sc("build_t27"), notes=notes,
               docs=SUMMARY_DOCS, tier="final and development",
               caption="Pilot 3 at global u1000, per q and arm (stochastic vs mean continuation), "
                       f"n = 10 seeds ({SEEDS_TXT}). e_hat_1(0), within-run SD/range and "
                       "sigma_1(0) in effort units (raw); errors and decomposition terms relative "
                       "to e_1*(0); deviation metrics divided by Delta W = 4.")
    med = (p3.assign(arm_s=p3["arm"].map(SHORT3)).groupby(["q", "arm_s"])
           .median(numeric_only=True).reset_index())
    med = med.rename(columns={"arm_s": "arm"})
    cols = ["stage1_rel_err_signed", "stage1_rel_err_abs", "learning_rel", "inherited_rel",
            "sigma_effort_at_0_t1", "Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw"]
    pack.crosscheck("T27", med, REP3,
                    header_has=["q", "arm", "stage1_rel_err_signed", "learning_rel"],
                    key_map={"q": "q", "arm": "arm"}, value_map={c: c for c in cols},
                    heading_has="3. Final checkpoint", label="Pilot 3 medians (section 3)")
    pack.crosscheck("T27", med, SUMM,
                    header_has=["stage1_rel_err", "abs_stage1_err", "within_run_sd_last5"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={"stage1_rel_err": "stage1_rel_err_signed",
                               "abs_stage1_err": "stage1_rel_err_abs",
                               "learning": "learning_rel", "inherited": "inherited_rel",
                               "within_run_sd_last5": "within_run_sd_e1_last5",
                               "EXP": "EXP_root_over_dw",
                               "dReach": "dReach_over_dw"},
                    heading_has="Pilot 3", label="summary.md Pilot 3 medians")
    if n_excl != 39:
        pack.mismatch("T27", "final rows whose learning band excludes 0", n_excl,
                      f"{REP3} (section 9)", "39 of 40")


# ----------------------------------------------------------------------------------------------
# T28: Pilot 3 paired differences
# ----------------------------------------------------------------------------------------------

def _tier_p3(m: str) -> str:
    """Tier label of a Pilot 3 paired metric."""
    if m.startswith(FT) or m in ("learning_rel", "learning_rel_abs", "inherited_rel"):
        return FIN
    if m in DEV_METRICS:
        return DEV
    if m in ("kl_final_epoch", "clip_frac", "phase_B_wall_sec"):
        return TRAIN
    return IND


def build_t28(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """T28: paired differences mean - stochastic (existing summary + final-tier verifier rows)."""
    rel = f"{P3A}/paired_summary.csv"
    ps = _csv(rel)
    p3 = ctx["p3"]
    x, y = ARMS3[1], ARMS3[0]
    metrics = list(dict.fromkeys(ps["metric"]))
    rng = np.random.default_rng(BOOT_SEED)
    rep = [{"q": q, "metric": m,
            **_paired_row(_pair_diff(p3, q, x, y, m), rng, True)} for q in QS for m in metrics]
    rp = ps.merge(pd.DataFrame(rep), on=["q", "metric"], suffixes=("", "_re"),
                  validate="one_to_one")
    d_pt = max(C.max_abs_diff(rp[c], rp[c + "_re"]) for c in ("mean", "sd", "median", "min", "max",
                                                              "n_neg", "n_pos", "n_zero"))
    d_ci = max(C.max_abs_diff(rp[c], rp[c + "_re"]) for c in ("boot_ci95_lo", "boot_ci95_hi"))
    ps["tier"] = [_tier_p3(m) for m in ps["metric"]]
    ps["source"] = rel
    extra = [(FT + "Gmax_full_over_dw", "Gmax_full/DW (final tier)"),
             (FT + "EXP_root_over_dw", "EXP_root/DW (final tier)"),
             (FT + "dReach_over_dw", "dReach/DW (final tier)"),
             (FT + "Deltamax_all_over_dw", "Delta_max_all/DW (final tier)"),
             (FT + "dFull_over_dw", "dFull/DW (final tier)")]
    blocks = []
    for q in QS:
        blocks.append(ps[ps["q"] == q])
        new = []
        for col, label in extra:
            r = _paired_row(_pair_diff(p3, q, x, y, col), np.random.default_rng(BOOT_SEED), True)
            new.append({"q": q, "metric": col, "label": label, **r,
                        "source": "recomputed from the per-run values (D03); bootstrap: fresh "
                                  "numpy default_rng(20261001) per row, 10,000 resamples of the "
                                  "10 pairs"})
        nd = pd.DataFrame(new)
        nd["better"] = "mean better if diff < 0"
        nd["n_mean_better"] = nd["n_neg"].astype(float)
        nd["tier"] = FIN
        blocks.append(nd.drop(columns=["_lower"])[list(ps.columns)])
    t = pd.concat(blocks, ignore_index=True)
    notes = (f"Rows with source = {rel} are verbatim (16 metrics; development tier for the "
             "verifier metrics); recomputing them from final_table.csv gives max abs diff "
             f"{d_pt:.2g} (point statistics) and {d_ci:.2g} (CI ends, single numpy "
             "default_rng(20261001) stream in the order of tools/v2/pilot3_analysis.py:paired). "
             "Appended per q: the five verifier metrics on the final tier (not in the existing "
             "analysis; bootstrap with a fresh default_rng(20261001) per row, 10,000 resamples). "
             "Difference = mean - stochastic per seed; better = number of pairs with diff < 0 "
             "(mean better) for metrics where smaller is better.")
    docs = {"n_mean_better": dict(definition="For metrics where smaller is better: number of "
                                             "pairs with mean - stochastic < 0 (mean better); "
                                             "empty for signed metrics",
                                  units="count"),
            "tier": dict(definition="Tier of the metric: " + TIER_TXT + "; " + TRAIN
                         + " = KL, clip fraction, wall time", units="label"),
            "source": dict(definition="File the row is copied from, or how it was recomputed",
                           units="text")}
    pack.table("T28", t, status="generated",
               sources=[C.src(rel), C.src(f"{P3A}/final_table.csv"), ctx["src_fv3"]],
               script=_sc("build_t28"), notes=notes, docs=docs, tier="final and development",
               caption="Pilot 3 paired differences mean - stochastic per (q, seed), n = 10 pairs "
                       f"(seeds {SEEDS_TXT}) per q; CI = 95% percentile bootstrap of the mean "
                       "difference.")
    for q, head in ((50, "4.1 q = 50"), (60, "4.2 q = 60")):
        sub = t[(t["q"] == q) & (t["source"] == rel)]
        pack.crosscheck("T28", sub, REP3, header_has=["label", "mean", "sd", "CI95"],
                        key_map={"label": "label"},
                        value_map={"mean": "mean", "sd": "sd", "median": "median", "min": "min",
                                   "max": "max"},
                        heading_has=head, label=f"paired table {head}")
        _check_paired_extras(pack, "T28", sub, REP3, head, "better")
    sub = t[t["source"] == rel].copy()
    pack.crosscheck("T28", sub, SUMM, header_has=["q", "metric", "n_mean_better", "boot_ci95_lo"],
                    key_map={"q": "q", "metric": "metric"},
                    value_map={"median": "median", "n_mean_better": "n_mean_better",
                               "boot_ci95_lo": "boot_ci95_lo",
                               "boot_ci95_hi": "boot_ci95_hi"},
                    heading_has="Pilot 3", label="summary.md Pilot 3 paired differences")


# ----------------------------------------------------------------------------------------------
# T29: Pilot 3 stability
# ----------------------------------------------------------------------------------------------

def build_t29(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """T29: within-run SD/range (last 5 exports), across-seed SD/IQR of e_hat_1(0), sigma_1(0)."""
    p3 = ctx["p3"]
    specs = [("within_run_sd_e1_last5", IND), ("within_run_range_e1_last5", IND), ("e1_at_0", IND),
             ("sigma_effort_at_0_t1", IND)]
    rows = []
    for (q, arm), g in p3.groupby(["q", "arm"], sort=True):
        for col, tier in specs:
            v = g[col].to_numpy(dtype=float)
            st = _stats(v)
            rows.append({"q": q, "arm": arm, "arm_short": SHORT3[arm], "metric": col, "tier": tier,
                         **st,
                         "sd": float(np.std(v, ddof=1)), "iqr": st["q75"] - st["q25"]})
    t = pd.DataFrame(rows)
    stab = _csv(f"{P3A}/stability.csv")
    w = t.pivot_table(index=["q", "arm"], columns="metric", values=["median", "sd", "iqr"])
    mine = pd.DataFrame({"across_seed_sd_final_e1": w[("sd", "e1_at_0")],
                         "across_seed_iqr_final_e1": w[("iqr", "e1_at_0")],
                         "median_within_run_sd_last5": w[("median", "within_run_sd_e1_last5")],
                         "median_within_run_range_last5":
                             w[("median", "within_run_range_e1_last5")],
                         "median_sigma1_0": w[("median", "sigma_effort_at_0_t1")]}).reset_index()
    sm = stab.merge(mine, on=["q", "arm"], suffixes=("", "_re"))
    dmax = max(C.max_abs_diff(sm[c], sm[c + "_re"]) for c in mine.columns if c not in ("q", "arm"))
    # within-run SD recomputed from the curves file
    cur = ctx["cur3"]
    l5 = cur[cur["update"].isin(LAST5)].groupby(["q", "seed", "arm"])["e1_at_0"]
    re_sd = l5.std(ddof=1)
    re_rg = l5.max() - l5.min()
    pi = p3.set_index(["q", "seed", "arm"])
    d_sd = C.max_abs_diff(pi["within_run_sd_e1_last5"], re_sd.loc[pi.index])
    d_rg = C.max_abs_diff(pi["within_run_range_e1_last5"], re_rg.loc[pi.index])
    notes = (f"Per (q, arm) over the 10 seeds ({SEEDS_TXT}): median, q25, q75 (numpy linear "
             "interpolation), min, max, n, sd (ddof = 1) and iqr = q75 - q25 of: within-run SD "
             "(ddof = 1) and range of e_hat_1(0) over the 5 weight exports u900, 925, 950, 975, "
             "1000 (final_table.csv within_run_*; recomputed from curves_weights_every25.csv "
             f"e1_at_0: max abs diff {d_sd:.2g} / {d_rg:.2g}); the final e_hat_1(0) (u1000; its sd "
             "and iqr are the across-seed SD and IQR); sigma_1(0) (u1000). The cells (e1_at_0, "
             "sd), (e1_at_0, iqr), (within_run_sd_e1_last5, median), (within_run_range_e1_last5, "
             "median), (sigma_effort_at_0_t1, median) reproduce "
             f"results/v2_pilots/pilot3/analysis/stability.csv: max abs diff {dmax:.2g}. Note: "
             "stability.csv computes the IQR with pandas quantile (linear), the same as numpy "
             "linear interpolation.")
    docs = {"sd": dict(definition="Across-seed sample SD (ddof = 1) of the metric",
                       units="effort units"),
            "iqr": dict(definition="Across-seed interquartile range q75 - q25 (numpy linear "
                                   "interpolation)",
                        units="effort units"),
            "arm_short": SUMMARY_DOCS["arm_short"], "metric": SUMMARY_DOCS["metric"],
            "tier": SUMMARY_DOCS["tier"]}
    pack.table("T29", t, status="generated",
               sources=[C.src(f"{P3A}/final_table.csv"), C.src(f"{P3A}/stability.csv"),
                        C.src(f"{P3A}/curves_weights_every25.csv")],
               script=_sc("build_t29"), notes=notes, docs=docs, tier="tier-independent",
               caption=f"Pilot 3 stage-1 stability per q and arm, n = 10 seeds ({SEEDS_TXT}); all "
                       "values in effort units (raw).")
    pack.crosscheck("T29", mine.assign(arm=mine["arm"].map(SHORT3)), REP3,
                    header_has=["across_seed_sd_final_e1", "median_sigma1_0"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={c: c for c in mine.columns if c not in ("q", "arm")},
                    heading_has="6. Stability",
                    label="stability table (section 6)")
    pack.crosscheck("T29", mine.assign(arm=mine["arm"].map(SHORT3)), SUMM,
                    header_has=["across_seed_sd_final_e1", "median_sigma1_0"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={c: c for c in mine.columns if c not in ("q", "arm")},
                    heading_has="Pilot 3",
                    label="summary.md stability")


# ----------------------------------------------------------------------------------------------
# F09: Pilot 3 learning curves
# ----------------------------------------------------------------------------------------------

def build_f09(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """F09: Pilot 3 learning curves, median and IQR per arm and q."""
    cur = ctx["cur3"]
    panels = [("stage-1 relative error", cur, "stage1_rel_err_signed",
               "stage-1 rel. error\n(signed, /e1*)"),
              ("learning term", cur, "learning_rel",
               "learning term\n" + r"$(\hat e_1-\tilde e_1)/e_1^*$"),
              ("sigma_1(0)", cur, "sigma_effort_at_0_t1", "σ₁(0)\n(effort units)"),
              ("Gmax_full", cur, "Gmax_full_over_dw", "Gmax_full / ΔW\n(dev tier)"),
              ("EXP_root", cur, "EXP_root_over_dw", "EXP_root / ΔW\n(dev tier)")]
    labels = {a: style.arm_label(LABEL3[a]) for a in ARMS3}
    fig, data = _curve_grid(panels, ARMS3, labels, 8.6, ("stage-1 relative error", "learning term"))
    p3 = ctx["p3"]
    checks = []
    for col in ("stage1_rel_err_signed", "learning_rel", "sigma_effort_at_0_t1",
                "Gmax_full_over_dw", "EXP_root_over_dw"):
        sel = (data["source_column"] == col) & (data["update"] == 1000)
        a = data[sel].sort_values(["q", "arm"])["median"]
        b = p3.groupby(["q", "arm"])[col].median().sort_index()
        checks.append(f"u1000 median of {col} vs final_table.csv: max abs diff "
                      f"{C.max_abs_diff(a, b):.2g}")
    caption = ("Pilot 3 learning curves: median (line) and IQR (band) over n = 10 seeds "
               f"({SEEDS_TXT}) per arm (stochastic vs mean continuation, frozen B2 stage 2), at "
               "the weight exports every 25 updates (global u425-u1000), q = 50 (left) and q = 60 "
               "(right). Rows: stage-1 relative error (signed, /e1*(0), tier-independent); "
               "learning term of the residual-band decomposition (/e1*(0), parent band, final "
               "tier); sigma_1(0) (SD of the stage-1 Beta action at d = 0, effort units); "
               "Gmax_full and EXP_root (/Delta W, development tier: state step 4, effort step 1, "
               "GL 16 per half). Source: "
               "results/v2_pilots/pilot3/analysis/curves_weights_every25.csv. The stochastic "
               "arm's runs are bit-identical to Pilot 2 B2 (T26). Reference line at 0 in the "
               "signed rows.")
    pack.figure("F09", fig, data, status="regenerated",
                sources=[C.src(f"{P3A}/curves_weights_every25.csv"),
                         C.src(f"{P3A}/final_table.csv"),
                         C.src("reports/v2/figures/pilot3/curves_q50.png"),
                         C.src("reports/v2/figures/pilot3/curves_q60.png")],
                script=_sc("build_f09"), caption=caption, tier="final and development",
                checks=checks,
                notes="Re-plotted in the pack style from the data of "
                      "reports/v2/figures/pilot3/curves_q50.png and curves_q60.png (same five "
                      "quantities).", docs=_curve_docs())


# ----------------------------------------------------------------------------------------------
# F10: e_hat_1(0) trajectories
# ----------------------------------------------------------------------------------------------

def build_f10(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """F10: e_hat_1(0) of every seed at the weight exports, per arm and q, with e1*(0)."""
    cur = ctx["cur3"]
    fig, axes = style.new_figure(2, 2, height=5.9, sharex=True, sharey="col")
    rows = []
    g1 = {q: sorted({float(ctx["fv3"][(q, s, a)]["development"]["g1"])
                     for s in SEEDS for a in ARMS3}) for q in QS}
    for q in QS:
        if len(g1[q]) != 1:
            raise ValueError("e1* differs across runs")
    handles, labels = [], []
    for i, arm in enumerate(ARMS3):
        for j, q in enumerate(QS):
            ax = axes[i, j]
            g = cur[(cur["q"] == q) & (cur["arm"] == arm)]
            for k, (s, gs) in enumerate(g.groupby("seed", sort=True)):
                gs = gs.sort_values("update")
                ln = ax.plot(gs["update"], gs["e1_at_0"], color=style.ARM_COLORS[arm], lw=1.0,
                             marker=MARK[arm], markersize=1.8, alpha=0.9)[0]
                if j == 0 and k == 0:
                    handles.append(ln)
                    lab = f"{style.arm_label(LABEL3[arm])}, one line per seed"
                    labels.append(style.label_n(lab, g["seed"].nunique()))
                rows.append(gs[["q", "seed", "arm", "update", "e1_at_0"]]
                            .assign(series="e_hat_1(0)"))
            e1s = g1[q][0]
            ref = ax.axhline(e1s, color=style.REF, lw=1.0, ls="-", zorder=5)
            if i == 0 and j == 0:
                ref_handle = ref
            rows.append(pd.DataFrame({"q": [q], "seed": [np.nan], "arm": [arm], "update": [np.nan],
                                      "e1_at_0": [e1s],
                                      "series": ["e1_star"]}))
            if i == 0:
                ax.set_title(f"q = {q}")
            if j == 0:
                ax.set_ylabel(r"$\hat e_1(0)$" + f" (effort, raw)\n{SHORT3[arm]}")
            if i == 1:
                ax.set_xlabel("global update")
    handles.append(ref_handle)
    labels.append(f"e1*(0), closed form ({g1[50][0]:.3f} at q = 50, {g1[60][0]:.3f} at q = 60)")
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    data = pd.concat(rows, ignore_index=True)
    fin = ctx["p3"].set_index(["q", "seed", "arm"])["e1_at_0"].sort_index()
    sel = (data["series"] == "e_hat_1(0)") & (data["update"] == 1000)
    u1 = data[sel].set_index(["q", "seed", "arm"])["e1_at_0"]
    checks = ["plotted e_hat_1(0) at u1000 equals final_table.csv e1_at_0 (40 runs): max abs diff "
              f"{C.max_abs_diff(u1.sort_index(), fin.loc[u1.sort_index().index]):.2g}",
              "e1*(0) = g1 of final_v2.json (identical in all 40 runs per q)"]
    caption = ("Pilot 3 stage-1 effort at the root, e_hat_1(0) (Beta mean, effort units, raw), at "
               "the weight exports every 25 updates (global u425-u1000): one line per seed (n = "
               f"10 seeds, {SEEDS_TXT}) for the stochastic (top) and mean (bottom) continuation "
               "arms, q = 50 (left) and q = 60 (right), with the closed-form e1*(0) = Delta W / "
               "(6kq). The u400 starting point (untrained stage 1 of the Pilot 1 parent) is not "
               "shown. Source: results/v2_pilots/pilot3/analysis/curves_weights_every25.csv "
               "(column e1_at_0); tier-independent.")
    docs = {"series": dict(definition="Plotted series: e_hat_1(0) of one seed, or e1_star "
                                      "(closed-form reference line)",
                           units="label"),
            "e1_at_0": dict(definition="Plotted value: e_hat_1(0) of the seed at the update, or "
                                       "e1*(0) for the e1_star row",
                            units="effort units [0, 100]", tier=IND),
            "update": dict(definition="Global update of the weight export (empty for the "
                                      "reference row)", units="updates"),
            "seed": dict(definition="Run seed (empty for the reference row)", units="integer")}
    pack.figure("F10", fig, data, status="generated",
                sources=[C.src(f"{P3A}/curves_weights_every25.csv"),
                         C.src(f"{P3A}/final_table.csv"), ctx["src_fv3"]],
                script=_sc("build_f10"), caption=caption, tier="tier-independent", checks=checks,
                docs=docs)


# ----------------------------------------------------------------------------------------------
# D02 / D03: per-run tables
# ----------------------------------------------------------------------------------------------

def build_d02(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """D02: Pilot 2 per-run table (final_table.csv columns + documented appended columns)."""
    p2 = ctx["p2"]
    orig = list(ctx["ft2"].columns)
    docs: Dict[str, Any] = {}
    docs.update(_lookup_docs(orig, DEV))
    docs.update(_orig_docs_p2())
    docs.update(_appended_docs())
    docs.update(_final_tier_docs(p2.columns))
    for c in p2.columns:
        if c.startswith("stage2_drift_"):
            base = DICT.lookup(c, DEV) or {}
            docs[c] = dict(base, units="effort units [0, 100]",
                           definition=base.get("definition", "") + "; dev D_2 grid, u1000")
    notes = ("One row per run (q, seed, arm; seeds explicit). Columns 1-51: "
             "results/v2_pilots/pilot2/analysis/final_table.csv unchanged (last training-time "
             "checkpoint u1000, development tier; stage1_learning_err_rel / "
             "stage1_inherited_err_rel and induced_* are the SUPERSEDED solver). Appended: "
             "stage2_peak_locfree_* and stage2_max_e2 (final_development.npz recovery arrays); "
             "stage2_drift_* statistics not in final_table.csv "
             "(final_v2.json['drift_vs_parent']); final_tier__* (final_v2.json['final'] via "
             "studies.final_tier_columns; the superseded solver's final-tier terms are left out); "
             "dec_* (revised residual-band decomposition at the u1000 weight export, "
             "decomposition_residual_band.csv).")
    pack.data("D02", p2, status="generated",
              sources=[C.src(f"{P2A}/final_table.csv"), ctx["src_fv2"], ctx["src_npz2"],
                       C.src(f"{P2A}/decomposition_residual_band.csv")],
              script=_sc("build_d02"), notes=notes, docs=docs, tier="final and development")
    _crosscheck_perrun_p2(pack, p2)


def _crosscheck_perrun_p2(pack: C.Pack, p2: pd.DataFrame) -> None:
    """Compare D02 with the per-run tables of pilot2_freeze.md (sections 2.1 and 6)."""
    chk = p2.assign(arm=p2["arm"].map(SHORT2))
    tab = _report_table(REP2, "2.1 Final checkpoint", ["seed", "wall s"])
    mine = ["stage2_drift_cand_on_max", "stage2_drift_cand_on_mean_cellmass_weighted",
            "stage2_drift_cand_off_max", "stage2_peak_rel_err_signed", "stage2_rmse_pos_over_g2_0",
            "stage2_tail_mean", "stage1_rel_err_signed", "sigma_effort_at_0_t1",
            "stage1_learning_err_rel", "stage1_inherited_err_rel", "induced_residual",
            "Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "EXP_root_over_dw", "dReach_over_dw",
            "Deltamax_all_over_dw", "dFull_over_dw", "eta_T_over_dw", "phase_B_wall_sec"]
    hdr = tab["header"]
    if len(hdr) != 3 + len(mine):
        raise ValueError("unexpected header of the per-run table in pilot2_freeze.md section 2.1")
    pack.crosscheck("D02", chk, REP2, header_has=["q", "seed", "arm", "wall s"],
                    key_map={"q": "q", "seed": "seed", "arm": "arm"},
                    value_map=dict(zip(hdr[3:], mine)), heading_has="2.1 Final checkpoint",
                    label="per-run table (section 2.1)")
    pack.crosscheck("D02", chk, REP2, header_has=["q", "seed", "arm", "total_rel", "learning band"],
                    key_map={"q": "q", "seed": "seed", "arm": "arm"},
                    value_map={"total_rel": "dec_total_rel", "learning_rel": "dec_learning_rel",
                               "inherited_rel": "dec_inherited_rel"},
                    heading_has="Final checkpoint (update 1000, weight export)",
                    label="revised decomposition per run (section 6)")
    tab = _report_table(REP2, "Final checkpoint (update 1000, weight export)",
                        ["learning band", "inherited band"])
    n_cmp = n_bad = 0
    ci = chk.set_index(["q", "seed", "arm"])
    for row in tab["rows"]:
        rec = dict(zip(tab["header"], row))
        r = ci.loc[(int(rec["q"]), int(rec["seed"]), rec["arm"])]
        for cell, lo, hi in ((rec["learning band"], r["dec_learning_rel_lo"],
                              r["dec_learning_rel_hi"]),
                             (rec["inherited band"], r["dec_inherited_rel_lo"],
                              r["dec_inherited_rel_hi"])):
            n_cmp += 1
            if not _band_cell_ok(float(lo), float(hi), cell):
                n_bad += 1
                pack.mismatch("D02", f"band [q={rec['q']}, seed={rec['seed']}, arm={rec['arm']}]",
                              f"[{lo!r}, {hi!r}]",
                              f"{REP2} (line {tab['line']})", cell)
        for c, mc in (("learning_contains_0", "dec_learning_contains_0"),
                      ("inherited_contains_0", "dec_inherited_contains_0")):
            n_cmp += 1
            if str(bool(r[mc])) != rec[c]:
                n_bad += 1
                pack.mismatch("D02", f"{c} [q={rec['q']}, seed={rec['seed']}, arm={rec['arm']}]",
                              bool(r[mc]),
                              f"{REP2} (line {tab['line']})", rec[c])
    _manual_check(pack, "D02", REP2, "bands and contains-0 flags per run (section 6)", n_cmp, n_bad)


def build_d03(pack: C.Pack, ctx: Dict[str, Any]) -> None:
    """D03: Pilot 3 per-run table (final_table.csv columns + documented appended columns)."""
    p3 = ctx["p3"]
    orig = list(ctx["ft3"].columns)
    docs: Dict[str, Any] = {}
    docs.update(_lookup_docs(orig, DEV))
    for c in ("learning_rel", "learning_rel_lo", "learning_rel_hi", "learning_contains_0",
              "inherited_rel", "inherited_rel_lo", "inherited_rel_hi", "inherited_contains_0"):
        base = DICT.lookup(c) or {}
        docs[c] = dict(base, tier=FIN,
                       definition="Residual-band decomposition at the u1000 weight export with "
                                  "the parent band (final tier): " + base.get("definition", ""))
    docs.update({
        "continuation_action_mode": dict(definition="Stage-2 continuation action mode in Phase B: "
                                                    "stochastic = both players sample the frozen "
                                                    "stage-2 Beta; mean = both play the frozen "
                                                    "Beta mean (the draws are still made and "
                                                    "discarded, A6)",
                                         units="label", tier=NA,
                                         source="run manifest"),
        "reward_mode": _orig_docs_p2()["reward_mode"],
        "phase_B_wall_sec": _orig_docs_p2()["phase_B_wall_sec"],
        "would_have_fired_B": _orig_docs_p2()["would_have_fired_B"],
        "snapshot_drift_max": dict(definition="max over the Beta mean, alpha and beta of the max "
                                              "over the dev D_2 grid of |frozen snapshot at u1000 "
                                              "- at freeze time| (drift_test.json)",
                                   units="effort units / Beta parameter units", tier=DEV,
                                   source="drift_test.json"),
        "drift_test_pass": _orig_docs_p2()["drift_test_pass"],
        "within_run_sd_e1_last5": dict(definition="Within-run SD (ddof = 1) of e_hat_1(0) over "
                                                  "the weight exports u900, 925, 950, 975, 1000",
                                       units="effort units [0, 100]", tier=IND,
                                       source="tools/v2/pilot3_analysis.py:final_table"),
        "within_run_range_e1_last5": dict(definition="Within-run range (max - min) of e_hat_1(0) "
                                                     "over the same 5 exports",
                                          units="effort units [0, 100]", tier=IND,
                                          source="tools/v2/pilot3_analysis.py:final_table"),
        "n_last5": dict(definition="Number of exports used for the within-run statistics",
                        units="count", tier=NA,
                        source="tools/v2/pilot3_analysis.py:final_table"),
        "learning_rel_abs": dict(definition="|learning_rel|", units="fraction of e_1*(0)",
                                 normalization="divided by e_1*(0)",
                                 tier=FIN, source="tools/v2/pilot3_analysis.py:final_table"),
    })
    docs.update(_appended_docs())
    docs.update(_final_tier_docs(p3.columns))
    notes = ("One row per run (q, seed, arm; seeds explicit). Columns 1-36: "
             "results/v2_pilots/pilot3/analysis/final_table.csv unchanged (last training-time "
             "checkpoint u1000, development tier for verifier metrics; decomposition terms from "
             "the residual band, final tier). Appended: final_tier__* (final_v2.json['final'] via "
             "studies.final_tier_columns); dec_e_tilde, dec_band_lo, dec_band_hi, dec_total_rel, "
             "dec_e1_inside_sweep (u1000 weight-export row of decomposition_residual_band.csv); "
             "inherited_rel_abs = |inherited_rel|.")
    pack.data("D03", p3, status="generated",
              sources=[C.src(f"{P3A}/final_table.csv"), ctx["src_fv3"],
                       C.src(f"{P3A}/decomposition_residual_band.csv")],
              script=_sc("build_d03"), notes=notes, docs=docs, tier="final and development")
    chk = p3.assign(arm=p3["arm"].map(SHORT3))
    tab = _report_table(REP3, "3. Final checkpoint", ["seed", "wall s"])
    hdr = tab["header"]
    mine = {3: "e1_at_0", 4: "stage1_rel_err_signed", 5: "learning_rel", 7: "inherited_rel",
            9: "sigma_effort_at_0_t1", 10: "within_run_sd_e1_last5",
            11: "within_run_range_e1_last5", 12: "Gmax_full_over_dw", 13: "Gmax_full_t",
            14: "Gmax_full_d", 15: "EXP_root_over_dw", 16: "dReach_over_dw",
            17: "Deltamax_all_over_dw", 18: "dFull_over_dw", 19: "kl_final_epoch", 20: "clip_frac",
            21: "phase_B_wall_sec"}
    pack.crosscheck("D03", chk, REP3, header_has=["q", "seed", "arm", "wall s"],
                    key_map={"q": "q", "seed": "seed", "arm": "arm"},
                    value_map={hdr[i]: c for i, c in mine.items()},
                    heading_has="3. Final checkpoint", label="per-run table (section 3)")
    n_cmp = n_bad = 0
    ci = chk.set_index(["q", "seed", "arm"])
    for row in tab["rows"]:
        rec = dict(zip(hdr, row))
        r = ci.loc[(int(rec["q"]), int(rec["seed"]), rec["arm"])]
        for cell, lo, hi in ((rec["learning band"], r["learning_rel_lo"], r["learning_rel_hi"]),
                             (rec["inherited band"], r["inherited_rel_lo"], r["inherited_rel_hi"])):
            n_cmp += 1
            if not _band_cell_ok(float(lo), float(hi), cell):
                n_bad += 1
                pack.mismatch("D03", f"band [q={rec['q']}, seed={rec['seed']}, arm={rec['arm']}]",
                              f"[{lo!r}, {hi!r}]",
                              f"{REP3} (line {tab['line']})", cell)
    _manual_check(pack, "D03", REP3, "learning and inherited bands per run (section 3)", n_cmp,
                  n_bad)


# ----------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------

def build() -> None:
    """Build T22-T29, F06-F10, D02 and D03 from scratch."""
    pack = C.Pack("sec_pilot23")
    _clean_own_outputs(pack)
    ctx = _context()
    build_d02(pack, ctx)
    build_d03(pack, ctx)
    build_t22(pack, ctx)
    build_t23(pack, ctx)
    build_t24(pack, ctx)
    build_t25(pack, ctx)
    build_f06(pack, ctx)
    build_f07(pack, ctx)
    build_f08(pack, ctx)
    build_t26(pack, ctx)
    build_t27(pack, ctx)
    build_t28(pack, ctx)
    build_t29(pack, ctx)
    build_f09(pack, ctx)
    build_f10(pack, ctx)
    pack.save_fragment()
