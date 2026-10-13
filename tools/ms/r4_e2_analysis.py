#!/usr/bin/env python3
"""MS-R4 pre-registered analysis of E2, does the development DP-BR help PPO training (prompt D5, D6 and section 3.3).

Arms: ``fb_a0`` (1, fixed stratified sampling, alpha 0), ``fb_r`` (2, residual-guided sampling inside the strata, alpha 0.5),
``fb_rg`` (3, residual + deviation-gain sampling inside the strata, alpha 0.5) and ``fb_rx`` (descriptive: the existing
cross-stratum sampler, alpha 0.5), each q in {50, 60} x the 20 development seeds 10511-10530. Everything but the sampling
feedback is identical (t1 actor, terminal stage 1600 updates, stage 1 600 updates, no stop rule, no polishing, the same checks).

Primary statistic (D6): the area ``A = (1 / 1575) x trapezoid`` of the FINAL-tier ``eta_2 / DW`` over the stage-2 weight exports
u25 ... u1600 (64 fixed checkpoints, evaluated offline by ``tools/ms/r4_curves.py``). The reduction is
``rho_X = 1 - mean(A_X) / mean(A_fb_a0)`` (means over the 20 seeds) with the paired percentile bootstrap interval (the seeds are
resampled jointly and the ratio of means is recomputed). X meets the target iff, at BOTH q,

  (E2-a) rho_X >= 0.20 and the lower end of its 95 % interval is > 0;
  (E2-b) no (q, seed) that passes all four gates under ``fb_a0`` fails any gate under X;
  (E2-c) Gmax_full/DW at the end of stage 1 (final tier): the upper end of the 95 % interval of the mean paired difference
         X - ``fb_a0`` is <= +0.001 (the G-N tolerance as the non-inferiority margin).

Decision order: ``fb_r`` first; if it meets the target the statement rests on it and ``fb_rg`` is descriptive; otherwise
``fb_rg`` is evaluated; if neither meets it the development DP-BR stays a verifier only and no training improvement is claimed.
97.5 % intervals are reported next to the 95 % ones (descriptive: two arms had a chance at one statement). A part that needs all
20 pairs is ``False`` with ``incomplete`` when a planned run or curve is missing. The thresholds, the bootstrap size and seed
(20261014) come from ``reports/ms/r4/prereg_criteria.json`` (``--criteria``); a FRESH ``default_rng(seed)`` per (comparison, q,
statistic).

Descriptive tables: ``fb_rx`` against ``fb_a0`` with the same statistics; the secondary statistics (mean log10 eta, the area over
local 200-1600 / 1400, the first export with eta <= 0.005, eta at u1600, the area of the development-tier online values on the same
64 updates, |peak error|, gap, RMSE_pos, tail mean at the freeze, the stage-1 error, the realised strata shares, the cost); the
sampling weights per check; the strata shares per update.

Outputs under ``--out`` (an existing non-empty directory is refused): ``per_run.csv``, ``criterion_e2.csv``, ``secondary_e2.csv``,
``fb_rx_comparison.csv``, ``learning_curves_e2.csv``, ``sampling_weights.csv``, ``strata_shares_by_update.csv``,
``sampling_weights_bins_seed_mean.csv`` (seed-mean per-bin weights, maps and probabilities, long form; the per-bin arrays of every
run are in its ``ms_binmaps_stage2.npz``), ``cost_e2.csv``, ``predictions_e2.csv``, ``analysis_info.json``, ``summary.txt``, ``figures/*.png``. Exit code 0 when every
planned run is done, 3 otherwise, 2 on an argument error.

Usage:
    python tools/ms/r4_e2_analysis.py --pilot-root results/ms_r4/pilot --curves-root results/ms_r4/curves \
        --criteria reports/ms/r4/prereg_criteria.json --out results/ms_r4/analysis/e2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import r1_analysis as R1  # noqa: E402
import r4_common as C  # noqa: E402

ETA = "final_eta_T_over_dw"
ARM_COLORS = {"fb_a0": "tab:blue", "fb_r": "tab:orange", "fb_rg": "tab:green", "fb_rx": "tab:red"}
#: secondary per-run statistics: (name, lower is better / None, kind) -- "area" kinds also get the reduction rho
SECONDARY: List[Tuple[str, Optional[bool], str]] = [
    ("A_late", True, "area"), ("A_dev_online", True, "area"), ("log10_mean", True, "diff"), ("first_le_level", True, "diff"),
    ("eta_u1600", True, "diff"), ("peak_abs", True, "diff"), ("gap", True, "diff"), ("rmse_pos", True, "diff"),
    ("tail_mean", True, "diff"), ("s1_abs", True, "diff"), ("gmax_final", True, "diff"), ("share_near", None, "diff"),
    ("share_mid", None, "diff"), ("wall_total_sec", True, "diff"), ("dev_verifier_share_stage2", True, "diff")]


# ---------------------------------------------------------------------------------------------- curve-derived statistics
def run_curve(curves: pd.DataFrame, arm: str, q: int, seed: int, crit: Mapping[str, Any]) -> Optional[pd.DataFrame]:
    """The 64-point final-tier curve of one run (sorted by update), or ``None`` if it is not complete and finite."""
    g = curves[(curves["arm"] == arm) & (curves["q"] == q) & (curves["seed"] == seed)].sort_values("update")
    a = crit["e2"]["area"]
    want = list(range(int(a["first_update"]), int(a["last_update"]) + 1, int(a["step"])))
    g = g[g["update"].isin(want)]                      # the fixed checkpoints; further exports of the curve file are ignored
    if list(g["update"].astype(int)) != want or not np.isfinite(g[ETA].astype(float)).all():
        return None
    return g


def curve_stats(g: Optional[pd.DataFrame], crit: Mapping[str, Any]) -> Dict[str, float]:
    """A, A_late, mean log10 eta, the first export with eta <= level (NaN if never), eta at the last export."""
    nan = float("nan")
    out = {"A": nan, "A_late": nan, "log10_mean": nan, "first_le_level": nan, "eta_u1600": nan, "n_final_invalid": nan}
    if g is None:
        return out
    if "final_valid" in g.columns:           # descriptive: a point whose verifier result is not valid still enters A (flagged here)
        out["n_final_invalid"] = float((~g["final_valid"].astype(bool)).sum())
    a, lw = crit["e2"]["area"], crit["e2"]["late_window"]
    u = g["update"].astype(float).to_numpy()
    y = g[ETA].astype(float).to_numpy()
    out["A"] = C.trapezoid(u, y) / float(a["divisor"])
    m = u >= float(lw["first_update"])
    out["A_late"] = C.trapezoid(u[m], y[m]) / float(lw["divisor"])
    out["log10_mean"] = float(np.mean(np.log10(np.maximum(y, 1e-300))))
    hit = np.flatnonzero(y <= float(crit["e2"]["eta_level"]))
    out["first_le_level"] = float(u[hit[0]]) if hit.size else nan
    out["eta_u1600"] = float(y[-1])
    return out


def online_area(run_dir: Path, crit: Mapping[str, Any]) -> float:
    """Area of the development-tier online values (``Delta`` of ``ms_checks_stage2.csv``) on the same 64 updates, / 1575."""
    p = Path(run_dir) / "ms_checks_stage2.csv"
    if not p.exists():
        return float("nan")
    ck = pd.read_csv(p)
    a = crit["e2"]["area"]
    want = list(range(int(a["first_update"]), int(a["last_update"]) + 1, int(a["step"])))
    ck = ck[ck["local"].isin(want)].sort_values("local")
    if list(ck["local"].astype(int)) != want:
        return float("nan")
    return C.trapezoid(ck["local"].astype(float), ck["Delta"].astype(float)) / float(a["divisor"])


def add_curve_columns(df: pd.DataFrame, curves: pd.DataFrame, crit: Mapping[str, Any]) -> pd.DataFrame:
    """Append A, the secondary curve statistics, the online area and the realised near / middle shares to the per-run table."""
    rows = []
    for r in df.itertuples(index=False):
        g = run_curve(curves, r.arm, int(r.q), int(r.seed), crit) if str(r.status) == "done" else None
        st = curve_stats(g, crit)
        st["A_dev_online"] = online_area(Path(r.run_dir), crit) if str(r.status) == "done" else float("nan")
        st["share_near"] = float(getattr(r, "t2_share_near_all", np.nan))
        st["share_mid"] = float(getattr(r, "t2_share_mid_all", np.nan))
        st["curve_complete"] = g is not None
        rows.append(st)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ---------------------------------------------------------------------------------------------- criterion (pure)
def _pairs(df: pd.DataFrame, x: str, ref: str, q: int, col: str) -> Tuple[List[int], np.ndarray, np.ndarray]:
    a = C.sel(df, x, q).set_index("seed")
    b = C.sel(df, ref, q).set_index("seed")
    seeds = [s for s in sorted(set(a.index) & set(b.index))
             if np.isfinite(float(a.loc[s, col])) and np.isfinite(float(b.loc[s, col]))]
    return seeds, np.array([float(a.loc[s, col]) for s in seeds]), np.array([float(b.loc[s, col]) for s in seeds])


def gate_violations(df: pd.DataFrame, x: str, ref: str, q: int) -> List[int]:
    """Seeds that pass all four gates under ``ref`` and fail any gate under ``x`` (a missing or failed ``x`` run fails)."""
    a = C.sel(df, x, q).set_index("seed")
    b = C.sel(df, ref, q).set_index("seed")
    out = []
    for s in sorted(b.index):
        ref_pass = C.truthy(b.loc[s, "gates_all_pass"])
        x_pass = bool(s in a.index and C.truthy(a.loc[s, "gates_all_pass"]))
        if ref_pass and not x_pass:
            out.append(int(s))
    return out


def criterion_e2(df: pd.DataFrame, crit: Mapping[str, Any], x_arms: Optional[Sequence[str]] = None,
                 role: str = "decision") -> pd.DataFrame:
    """The E2 criterion rows: one per (X, q) and one ``q = both`` row per X; the decision is added by :func:`decide`.

    Columns: ``arm, role, q, n_expected, n_pairs, mean_A_x, mean_A_ref, rho, ci95_lo, ci95_hi, ci975_lo, ci975_hi, a_pass,
    n_gate_violations, violating_seeds, b_pass, mean_gmax_diff, gmax_ci_lo, gmax_ci_hi, c_pass, met_q, status``.
    """
    e2 = crit["e2"]
    ref = e2["arms"]["reference"]
    xs = list(x_arms if x_arms is not None else e2["arms"]["x"])
    n_exp = int(crit["seeds"]["n_per_q"])
    n_boot, seed = int(crit["bootstrap"]["n_resamples"]), int(crit["bootstrap"]["seed_e2"])
    p95, p975 = tuple(crit["bootstrap"]["pct95"]), tuple(crit["bootstrap"]["pct975"])
    rows: List[Dict[str, Any]] = []
    for x in xs:
        per_q = []
        for q in crit["qs"]:
            seeds, ax_, ar_ = _pairs(df, x, ref, q, "A")
            red = C.boot_reduction(ax_, ar_, n_boot, seed, (p95, p975)) if len(seeds) else \
                {"rho": float("nan"), "intervals": [(float("nan"),) * 2] * 2}
            (lo95, hi95), (lo975, hi975) = red["intervals"]
            a_pass = bool(len(seeds) == n_exp and red["rho"] >= float(e2["reduction_min"])
                          and lo95 > float(e2["reduction_ci_lower_above"]))
            viol = gate_violations(df, x, ref, q)
            b_pass = not viol
            gseeds, gx, gr = _pairs(df, x, ref, q, "gmax_final")
            glo, ghi = C.boot_mean_ci(gx - gr, n_boot, seed, p95) if len(gseeds) else (float("nan"),) * 2
            c_pass = bool(len(gseeds) == n_exp and ghi <= float(e2["gmax_noninferiority_margin"]))
            complete = len(seeds) == n_exp and len(gseeds) == n_exp
            per_q.append({"arm": x, "role": role, "q": q, "n_expected": n_exp, "n_pairs": len(seeds),
                          "mean_A_x": float(ax_.mean()) if len(seeds) else np.nan,
                          "mean_A_ref": float(ar_.mean()) if len(seeds) else np.nan, "rho": red["rho"],
                          "ci95_lo": lo95, "ci95_hi": hi95, "ci975_lo": lo975, "ci975_hi": hi975, "a_pass": a_pass,
                          "n_gate_violations": len(viol), "violating_seeds": ";".join(map(str, viol)), "b_pass": b_pass,
                          "mean_gmax_diff": float((gx - gr).mean()) if len(gseeds) else np.nan, "gmax_ci_lo": glo,
                          "gmax_ci_hi": ghi, "c_pass": c_pass, "met_q": bool(a_pass and b_pass and c_pass),
                          "status": "ok" if complete else "incomplete"})
        rows.extend(per_q)
        met = bool(len(per_q) == len(crit["qs"]) and all(r["met_q"] for r in per_q))
        rows.append({"arm": x, "role": role, "q": "both", "met_q": met,
                     "status": "MEETS THE TARGET" if met else "does not meet the target"})
    return pd.DataFrame(rows)


def decide(tab: pd.DataFrame, crit: Mapping[str, Any]) -> Dict[str, Any]:
    """The decision by the pre-registered order (``fb_r`` first, then ``fb_rg``)."""
    met = {r.arm: bool(r.met_q) for r in tab.itertuples(index=False) if str(r.q) == "both" and r.role == "decision"}
    for arm in crit["e2"]["decision_order"]:
        if met.get(arm):
            other = [a for a in crit["e2"]["decision_order"] if a != arm]
            return {"statement_rests_on": arm, "descriptive": other + list(crit["e2"]["arms"]["descriptive"]),
                    "text": f"development-DP-BR sampling feedback lowers the area under the stage-2 eta_2/DW curve at this "
                            f"budget (the statement rests on {arm})"}
    return {"statement_rests_on": None, "descriptive": list(crit["e2"]["arms"]["x"]) + list(crit["e2"]["arms"]["descriptive"]),
            "text": "neither fb_r nor fb_rg meets the target: the development DP-BR is kept as a verifier only and no "
                    "training improvement is claimed"}


def secondary_table(df: pd.DataFrame, crit: Mapping[str, Any], arms: Sequence[str]) -> pd.DataFrame:
    """Descriptive paired statistics of :data:`SECONDARY` (arm - fb_a0) and, for the area kinds, the reduction rho."""
    ref = crit["e2"]["arms"]["reference"]
    n_boot, seed = int(crit["bootstrap"]["n_resamples"]), int(crit["bootstrap"]["seed_e2"])
    rows = []
    for arm in arms:
        for q in crit["qs"]:
            for name, lower, kind in SECONDARY:
                if name not in df.columns:
                    continue
                seeds, x, r = _pairs(df, arm, ref, q, name)
                st = C.paired_stats(x, r, n_boot, seed)
                row = {"arm": arm, "q": q, "statistic": name, "direction": "lower is better" if lower else "none",
                       "n_expected": int(crit["seeds"]["n_per_q"]), **st}
                if kind == "area" and len(seeds):
                    red = C.boot_reduction(x, r, n_boot, seed)
                    row.update(rho=red["rho"], rho_ci_lo=red["intervals"][0][0], rho_ci_hi=red["intervals"][0][1])
                if name == "first_le_level":
                    a = C.sel(df, arm, q)
                    row["n_never_reached"] = int(a[name].isna().sum())
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- sampling weights, shares
def _digest(p: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(p, dtype="<f8").tobytes()).hexdigest()[:16]


def sampling_weights_table(df: pd.DataFrame, arms: Sequence[str]
                           ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[Tuple[str, int], np.ndarray]]:
    """Per check: the stratum masses of the probabilities in force from the next update, the total variation distance to the
    alpha-0 probabilities, the maximum within-stratum weight ratios, the mass of the EMA maps; and the seed-mean heat-map arrays.

    ``probs_next`` is used where recorded, otherwise the probabilities are reconstructed from ``probs_first_update`` /
    ``probs_table``. Returns ``(table, {f"{arm}_q{q}_{rho_bar|gain_bar|probs_next}": seed-mean array of checks x bins},
    {(arm, q): the check updates})``.
    """
    from envs.curriculum_env import GameSpec, StartSampler
    rows: List[Dict[str, Any]] = []
    heat: Dict[str, List[np.ndarray]] = {}
    updates: Dict[Tuple[str, int], np.ndarray] = {}
    for r in df.itertuples(index=False):
        if r.arm not in arms or str(r.status) != "done":
            continue
        d = Path(r.run_dir)
        try:
            cfg = json.load(open(d / "run_config.json"))
            g = cfg["record"]["game"]
            spec = GameSpec(**{k: g[k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")})
            sw = cfg["start_weights"]
            sp = StartSampler(spec, float(cfg["record"]["protocol"]["es_bin_width"]))
            lab = sp.stratum_labels(2, float(sw["near_tie_half_width"]))
            p0 = sp.stratified_bin_probs(2, float(sw["lambda_P"]), float(sw["near_tie_half_width"]), alpha=0.0)
            z = np.load(d / "ms_binmaps_stage2.npz")
        except Exception:  # noqa: BLE001 - an unreadable run has no rows
            continue
        last_update = int(z["update"][-1])
        pn_run: List[np.ndarray] = []
        for i, u in enumerate(z["update"]):
            if "probs_next" in z.files:
                p = z["probs_next"][i]
            elif int(u) >= last_update:
                p = np.full(p0.size, np.nan)
            else:
                idx = int(np.searchsorted(z["probs_first_update"], int(u) + 1, side="right")) - 1
                p = z["probs_table"][idx]
            pn_run.append(np.asarray(p, dtype=float))
            ok = np.isfinite(p).all()
            row: Dict[str, Any] = {"arm": r.arm, "q": r.q, "seed": r.seed, "update": int(u), "local": int(z["local"][i]),
                                   "p_digest_next": _digest(p) if ok else ""}
            if ok:
                row.update(mass_tail=float(p[lab == 0].sum()), mass_near=float(p[lab == 1].sum()), mass_mid=float(p[lab == 2].sum()),
                           tv_from_alpha0=float(0.5 * np.abs(p - p0).sum()),
                           max_ratio_near=float(p[lab == 1].max() / p[lab == 1].mean()),
                           max_ratio_mid=float(p[lab == 2].max() / p[lab == 2].mean()))
            rb = z["rho_bar"][i]
            row["rho_bar_max"] = float(np.nanmax(rb)) if np.isfinite(rb).any() else float("nan")
            if "gain_bar" in z.files:
                gb = z["gain_bar"][i]
                row["gain_bar_max"] = float(np.nanmax(gb)) if np.isfinite(gb).any() else float("nan")
            rows.append(row)
        heat.setdefault(f"{r.arm}_q{r.q}_probs_next", []).append(np.stack(pn_run))
        updates[(r.arm, int(r.q))] = np.asarray(z["update"])
        heat.setdefault(f"{r.arm}_q{r.q}_rho_bar", []).append(z["rho_bar"])
        if "gain_bar" in z.files:
            heat.setdefault(f"{r.arm}_q{r.q}_gain_bar", []).append(z["gain_bar"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # the tail bins are NaN in every run: an all-NaN column stays NaN
        mean_heat = {k: np.nanmean(np.stack(v), axis=0) for k, v in heat.items() if v and np.stack(v).shape[0]}
    return pd.DataFrame(rows), mean_heat, updates


HEAT_KEY = re.compile(r"^(?P<arm>.+)_q(?P<q>\d+)_(?P<kind>rho_bar|gain_bar|probs_next)$")


def heat_long_table(heat: Mapping[str, np.ndarray], updates: Mapping[Tuple[str, int], np.ndarray]) -> pd.DataFrame:
    """Seed-mean per-bin values of the maps and of the probabilities in force from the next update, in long form (arm, q, kind,
    check update, bin, bin centre, stratum, value); NaN where a map is undefined (the tail bins) or there is no next update."""
    from envs.curriculum_env import GameSpec, StartSampler
    rows = []
    for key, arr in sorted(heat.items()):
        m = HEAT_KEY.match(key)
        if not m:
            continue
        arm, q, kind = m.group("arm"), int(m.group("q")), m.group("kind")
        spec = GameSpec(w_h=1.0, w_l=0.0, k=1.0, q=float(q), T=2, e_min=0.0, e_max=100.0)       # the strata depend on q only
        sp = StartSampler(spec, 10.0)
        edges = sp.bin_edges(2)
        centres = 0.5 * (edges[:-1] + edges[1:])
        lab = np.array(["tail", "near", "mid"])[sp.stratum_labels(2, 20.0)]
        for i, u in enumerate(updates[(arm, q)]):
            for b in range(arr.shape[1]):
                rows.append({"arm": arm, "q": q, "kind": kind, "update": int(u), "bin": b, "bin_centre": float(centres[b]),
                             "stratum": lab[b], "seed_mean": float(arr[i, b])})
    return pd.DataFrame(rows, columns=["arm", "q", "kind", "update", "bin", "bin_centre", "stratum", "seed_mean"])


def strata_shares_table(df: pd.DataFrame, arms: Sequence[str]) -> pd.DataFrame:
    """Seed-mean realised start shares (tail / near-tie / middle) per arm, q and terminal-stage update."""
    acc: Dict[Tuple[str, int], List[pd.DataFrame]] = {}
    for r in df.itertuples(index=False):
        if r.arm not in arms or str(r.status) != "done":
            continue
        p = Path(r.run_dir) / "ms_updates.csv"
        if not p.exists():
            continue
        u = pd.read_csv(p, usecols=["stage", "local", "n_start_tail", "n_start_near", "n_start_mid"])
        u = u[u["stage"] == 2]
        n = u[["n_start_tail", "n_start_near", "n_start_mid"]].sum(axis=1)
        acc.setdefault((r.arm, int(r.q)), []).append(pd.DataFrame({
            "local": u["local"], "share_tail": u["n_start_tail"] / n, "share_near": u["n_start_near"] / n,
            "share_mid": u["n_start_mid"] / n}))
    rows = []
    for (arm, q), lst in acc.items():
        cat = pd.concat(lst).groupby("local").mean().reset_index()
        cat.insert(0, "q", q)
        cat.insert(0, "arm", arm)
        cat["n_runs"] = len(lst)
        rows.append(cat)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["wall_total_sec", "wall_stage2_sec", "wall_stage1_sec", "stage2_train_rollout_sec", "stage2_train_update_sec",
            "stage2_dev_verifier_sec", "stage1_dev_verifier_sec", "dev_verifier_share_stage2"]
    rows = []
    for (arm, q), g in df.groupby(["arm", "q"], sort=False):
        for c in cols:
            v = g[c].astype(float)
            v = v[np.isfinite(v)]
            rows.append({"arm": arm, "q": q, "quantity": c, "n": len(v), "median": float(v.median()) if len(v) else np.nan,
                         "min": float(v.min()) if len(v) else np.nan, "max": float(v.max()) if len(v) else np.nan,
                         "mean": float(v.mean()) if len(v) else np.nan})
    return pd.DataFrame(rows)


def curve_summary(curves: pd.DataFrame, arms: Sequence[str]) -> pd.DataFrame:
    rows = []
    for (arm, q, u), g in curves[curves["arm"].isin(arms)].groupby(["arm", "q", "update"], sort=True):
        v = g[ETA].astype(float).to_numpy()
        rows.append({"arm": arm, "q": q, "update": u, "n": len(v), "eta_median": float(np.median(v)),
                     "eta_q25": float(np.percentile(v, 25)), "eta_q75": float(np.percentile(v, 75)),
                     "eta_mean": float(v.mean())})
    return pd.DataFrame(rows)


def predictions_table(crit_tab: pd.DataFrame, sec: pd.DataFrame, df: pd.DataFrame, crit: Mapping[str, Any]) -> pd.DataFrame:
    """D7: E2-P1 .. E2-P3 against the outcome (expectations, not tests)."""
    rows = []
    rx = crit_tab[(crit_tab["arm"] == "fb_rx") & (crit_tab["q"].astype(str) == "60")]
    if len(rx):
        rho = float(rx["rho"].iloc[0])
        rows.append({"prediction": "E2-P1", "statement": "fb_rx has a larger area than fb_a0 at q = 60", "observed_rho_fb_rx_q60": rho,
                     "outcome": "larger area (rho < 0)" if rho < 0 else "not larger (rho >= 0)"})
    for arm in crit["e2"]["arms"]["x"]:
        for q in crit["qs"]:
            r = crit_tab[(crit_tab["arm"] == arm) & (crit_tab["q"].astype(str) == str(q))]
            s = sec[(sec["arm"] == arm) & (sec["q"] == q) & (sec["statistic"] == "A_late")]
            if len(r):
                rows.append({"prediction": "E2-P2", "arm": arm, "q": q,
                             "statement": "fb_r falls short of a 20 % reduction; any gain shows more in the area over local 200-1600",
                             "rho_full": float(r["rho"].iloc[0]), "rho_late": float(s["rho"].iloc[0]) if len(s) and "rho" in s else np.nan,
                             "short_of_20pct": bool(float(r["rho"].iloc[0]) < float(crit["e2"]["reduction_min"]))})
    sh = df.groupby("arm")["dev_verifier_share_stage2"].max()
    for arm, v in sh.items():
        rows.append({"prediction": "E2-P3", "arm": arm, "statement": "the development checks take under 1 % of the stage-2 wall time",
                     "max_dev_verifier_share_stage2": float(v), "outcome": "under 1 %" if v < 0.01 else "1 % or more"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- figures
def _plt() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _footer(fig: Any, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=6, color="gray", ha="left", va="bottom")


def fig_curves(path: Path, cs: pd.DataFrame, crit: Mapping[str, Any], log: bool, footer: str) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(crit["qs"]), figsize=(10, 4), sharey=True)
    for ax, q in zip(np.atleast_1d(axes), crit["qs"]):
        for arm, g in cs[cs["q"] == q].groupby("arm"):
            ax.plot(g["update"], g["eta_median"], color=ARM_COLORS.get(arm, "k"), label=arm)
            ax.fill_between(g["update"], g["eta_q25"], g["eta_q75"], color=ARM_COLORS.get(arm, "k"), alpha=0.15)
        ax.axhline(float(crit["e2"]["eta_level"]), color="gray", ls=":", lw=0.8)
        if log:
            ax.set_yscale("log")
        ax.set_title(f"q = {q}")
        ax.set_xlabel("terminal-stage update")
    np.atleast_1d(axes)[0].set_ylabel("eta_2 / DW (final tier, offline)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_area_paired(path: Path, df: pd.DataFrame, crit: Mapping[str, Any], footer: str) -> None:
    plt = _plt()
    ref = crit["e2"]["arms"]["reference"]
    xs = list(crit["e2"]["arms"]["x"]) + list(crit["e2"]["arms"]["descriptive"])
    fig, axes = plt.subplots(1, len(crit["qs"]), figsize=(10, 4.5))
    for ax, q in zip(np.atleast_1d(axes), crit["qs"]):
        for x in xs:
            seeds, ax_, ar_ = _pairs(df, x, ref, q, "A")
            ax.scatter(ar_, ax_, s=16, color=ARM_COLORS.get(x, "k"), label=x)
        lim = [ax.get_xlim()[0], ax.get_xlim()[1]]
        ax.plot(lim, lim, "k-", lw=0.8)
        ax.plot(lim, [v * (1.0 - float(crit["e2"]["reduction_min"])) for v in lim], "k--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"A of {ref} (per seed)")
        ax.set_title(f"q = {q}  (solid: no change; dashed: 20 % reduction)")
    np.atleast_1d(axes)[0].set_ylabel("A of the arm (same seed)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_heat(path: Path, heat: Mapping[str, np.ndarray], key: str, crit: Mapping[str, Any], footer: str) -> bool:
    items = [(k, v) for k, v in sorted(heat.items()) if k.endswith(key)]
    if not items:
        return False
    plt = _plt()
    n = len(items)
    cols = len(crit["qs"])
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(5 * cols, 3.2 * ((n + cols - 1) // cols)), squeeze=False)
    for ax, (k, v) in zip(axes.ravel(), items):
        m = ~np.isnan(v).all(axis=0)
        im = ax.imshow(v[:, m].T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(k.replace("_" + key, "") + " " + key, fontsize=8)
        ax.set_xlabel("check index (every 25 updates)")
        ax.set_ylabel("non-tail bin")
        fig.colorbar(im, ax=ax)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def fig_shares(path: Path, sh: pd.DataFrame, crit: Mapping[str, Any], footer: str) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(crit["qs"]), figsize=(10, 4), sharey=True)
    for ax, q in zip(np.atleast_1d(axes), crit["qs"]):
        for arm, g in sh[sh["q"] == q].groupby("arm"):
            w = g.sort_values("local")
            ax.plot(w["local"], w["share_near"].rolling(25, min_periods=1).mean(), color=ARM_COLORS.get(arm, "k"), label=f"{arm} near-tie")
            ax.plot(w["local"], w["share_mid"].rolling(25, min_periods=1).mean(), color=ARM_COLORS.get(arm, "k"), ls="--")
        ax.set_title(f"q = {q} (solid near-tie, dashed middle; 25-update means)")
        ax.set_xlabel("terminal-stage update")
    np.atleast_1d(axes)[0].set_ylabel("realised start share")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- driver
def summary_text(tab: pd.DataFrame, dec: Mapping[str, Any], roots: Mapping[str, str]) -> str:
    lines = ["MS-R4 E2 analysis (tools/ms/r4_e2_analysis.py)", f"roots: {json.dumps(dict(roots))}", ""]
    for r in tab.itertuples(index=False):
        if str(r.q) == "both":
            lines.append(f"{r.arm} ({r.role}): {r.status}")
        else:
            lines.append(f"{r.arm} q={r.q}: rho = {r.rho:+.4f} 95% [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}] 97.5% [{r.ci975_lo:+.4f}, "
                         f"{r.ci975_hi:+.4f}] (a: {r.a_pass}); gate violations {int(r.n_gate_violations)} (b: {r.b_pass}); Gmax diff "
                         f"{r.mean_gmax_diff:+.5f} upper {r.gmax_ci_hi:+.5f} (c: {r.c_pass}); {r.status}")
    lines += ["", "DECISION (order fb_r, then fb_rg): " + dec["text"]]
    return "\n".join(lines) + "\n"


def run_analysis(pilot_root: Path, curves_root: Path, criteria_path: Path, out: Path,
                 protocol_path: Path = ROOT / "protocols" / "v2_T2_locked_v2_0.json") -> Dict[str, Any]:
    """The whole E2 analysis; returns the info dict (also written to ``analysis_info.json``)."""
    import ms_configs as mc
    crit, crit_sha = C.load_criteria(criteria_path)
    th, proto_sha = R1.load_thresholds(protocol_path)
    proto = mc.load_protocol(protocol_path)
    e2 = crit["e2"]
    ref = e2["arms"]["reference"]
    arms = [ref] + list(e2["arms"]["x"]) + list(e2["arms"]["descriptive"])
    seeds = list(range(int(crit["seeds"]["pilot"][0]), int(crit["seeds"]["pilot"][1]) + 1))
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    curves = pd.read_csv(Path(curves_root) / "curves_all.csv")
    df = C.build_per_run(pilot_root, arms, crit["qs"], seeds, th, proto, {a: "E2" for a in arms})
    df = add_curve_columns(df, curves, crit)
    df = df.copy()
    df.insert(0, "roots_id", R1.roots_id({"pilot": str(pilot_root), "curves": str(curves_root)}))
    df.to_csv(out / "per_run.csv", index=False)
    tab = pd.concat([criterion_e2(df, crit), criterion_e2(df, crit, e2["arms"]["descriptive"], "descriptive")], ignore_index=True)
    dec = decide(tab, crit)
    # the decision by the order of D6 is the last row of criterion_e2.csv (role "decision_rule"; the decision itself reads the
    # "decision" rows only)
    verdict = pd.DataFrame([{"arm": "DECISION", "role": "decision_rule", "q": "both", "met_q": dec["statement_rests_on"] is not None,
                             "status": dec["text"]}])
    pd.concat([tab, verdict], ignore_index=True).to_csv(out / "criterion_e2.csv", index=False)
    sec = secondary_table(df, crit, arms[1:])
    sec.to_csv(out / "secondary_e2.csv", index=False)
    tab[tab["role"] == "descriptive"].to_csv(out / "fb_rx_comparison.csv", index=False)
    cs = curve_summary(curves, arms)
    cs.to_csv(out / "learning_curves_e2.csv", index=False)
    sw, heat, upd = sampling_weights_table(df, arms)
    sw.to_csv(out / "sampling_weights.csv", index=False)
    heat_long_table(heat, upd).to_csv(out / "sampling_weights_bins_seed_mean.csv", index=False)
    np.savez(out / "heatmaps_seed_mean.npz", **heat)
    sh = strata_shares_table(df, arms)
    sh.to_csv(out / "strata_shares_by_update.csv", index=False)
    cost_table(df).to_csv(out / "cost_e2.csv", index=False)
    predictions_table(tab, sec, df, crit).to_csv(out / "predictions_e2.csv", index=False)
    footer = f"pilot root: {pilot_root}; criteria sha256 {crit_sha[:12]}"
    figs = []
    fig_curves(out / "figures" / "eta_curves_log.png", cs, crit, True, footer)
    fig_curves(out / "figures" / "eta_curves_linear.png", cs, crit, False, footer)
    fig_area_paired(out / "figures" / "area_paired.png", df, crit, footer)
    figs += ["eta_curves_log.png", "eta_curves_linear.png", "area_paired.png"]
    if fig_heat(out / "figures" / "rho_bar_heatmaps.png", heat, "rho_bar", crit, footer):
        figs.append("rho_bar_heatmaps.png")
    if fig_heat(out / "figures" / "gain_bar_heatmaps.png", heat, "gain_bar", crit, footer):
        figs.append("gain_bar_heatmaps.png")
    if len(sh):
        fig_shares(out / "figures" / "strata_shares.png", sh, crit, footer)
        figs.append("strata_shares.png")
    roots = {"pilot": str(pilot_root), "curves": str(curves_root), "criteria": str(criteria_path)}
    (out / "summary.txt").write_text(summary_text(tab, dec, roots))
    info = {"tool": "tools/ms/r4_e2_analysis.py", "roots": roots, "criteria_sha256": crit_sha, "protocol_sha256": proto_sha,
            "arms": arms, "qs": list(crit["qs"]), "seeds": seeds, "n_planned": len(df),
            "n_not_done": int((df["status"].astype(str) != "done").sum()), "decision": dec, "figures": figs}
    with open(out / "analysis_info.json", "w") as f:
        json.dump(info, f, indent=1)
    return info


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("--pilot-root", required=True)
    p.add_argument("--curves-root", required=True)
    p.add_argument("--criteria", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error(f"{out} exists and is not empty")
    info = run_analysis(Path(a.pilot_root), Path(a.curves_root), Path(a.criteria), out)
    print(json.dumps({k: info[k] for k in ("n_planned", "n_not_done")}, indent=1))
    print(info["decision"]["text"])
    return 0 if info["n_not_done"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
