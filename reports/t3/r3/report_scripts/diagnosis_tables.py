#!/usr/bin/env python3
"""Tables of ``reports/t3/r3/01_diagnosis.md`` and ``02_proposal.md``: every table is generated here from the CSVs of
``results/t3_r3/diag/`` (and the tracked T3-R1 / T3-R2 tables). Usage: ``python reports/t3/r3/report_scripts/diagnosis_tables.py``
writes ``reports/t3/r3/tables/<name>.md`` and prints the names; ``build_report.py`` substitutes them into the templates."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

ROOT = Path(__file__).resolve().parents[4]
DIAG = ROOT / "results" / "t3_r3" / "diag"
OUT = ROOT / "reports" / "t3" / "r3" / "tables"
sys.path.insert(0, str(ROOT / "tools" / "t3"))
sys.path.insert(0, str(ROOT))

FAIL_KEYS = [("t3_r1", 50, 50504), ("t3_r1", 50, 50517), ("t3_r1", 60, 50514), ("t3_r1", 60, 50517), ("t3_r2", 50, 60501),
             ("t3_r2", 50, 60506), ("t3_r2", 50, 60513), ("t3_r2", 50, 60515), ("t3_r2", 60, 60516), ("t3_r2", 60, 60518)]


def fmt(v: Any, nd: int = 4) -> str:
    """Cell text: NaN -> '-', floats with 4 significant digits (scientific below 1e-3 / above 1e5)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v == 0.0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.3g}"
        return f"{v:.{nd}g}"
    return str(v)


def md(rows: Sequence[Dict[str, Any]], cols: Optional[Sequence[str]] = None, heads: Optional[Sequence[str]] = None,
       nd: int = 4) -> str:
    """Markdown table of dict rows."""
    cols = list(cols or rows[0].keys())
    heads = list(heads or cols)
    out = ["| " + " | ".join(heads) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c), nd) for c in cols) + " |")
    return "\n".join(out)


def rd(name: str) -> pd.DataFrame:
    """A CSV of the diagnosis output directory."""
    return pd.read_csv(DIAG / name)


def label(r: str, q: int, s: int) -> str:
    """Run label ``T3-R1 q50 s50504``."""
    return f"{'T3-R1' if r == 't3_r1' else 'T3-R2'} q{q} s{s}"


def cp(k: int, n: int) -> str:
    """Exact (Clopper-Pearson) 95 % interval as text."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return f"[{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------------------------- DG1
def t_inventory() -> str:
    inv = rd("dg1_inventory.csv")
    v20 = rd("dg1_inventory_v20.csv")
    tr = [c for c in inv.columns if c.startswith("tracked_")]
    un = [c for c in inv.columns if c.startswith("untracked_")]
    rows = []
    for rnd, g in inv.groupby("round"):
        rows.append({"records": f"{'T3-R1' if rnd == 't3_r1' else 'T3-R2'} `bw` runs ({len(g)})",
                     "tracked files present (of 8 kinds, all runs)": f"{int(g[tr].sum().min())}-{int(g[tr].sum().max())} of {len(g)} per kind",
                     "untracked files present (of 8 kinds)": f"{int(g[un].sum().min())}-{int(g[un].sum().max())} of {len(g)} per kind",
                     "stage-3 exports `u00025`-`u01600`": f"{int(g['n_stage3_exports'].sum())} of {len(g) * 64}",
                     "exports in total": f"{int(g['n_exports'].min())}-{int(g['n_exports'].max())} per run",
                     "`ms_binmaps_stage3.npz` `probs` all NaN": f"{int(g['binmaps3_probs_all_nan'].sum())} of {len(g)}"})
    rows.append({"records": f"v2.0 confirmation runs ({len(v20)})", "tracked files present (of 8 kinds, all runs)": f"`v2_updates.csv` {int(v20['tracked_v2_updates.csv'].sum())} of {len(v20)}",
                 "untracked files present (of 8 kinds)": f"`v2_updates.csv` {int(v20['untracked_v2_updates.csv'].sum())} of {len(v20)}",
                 "stage-3 exports `u00025`-`u01600`": "-", "exports in total": f"{int(v20['n_exports'].min())}-{int(v20['n_exports'].max())} per run",
                 "`ms_binmaps_stage3.npz` `probs` all NaN": "-"})
    return md(rows)


# ---------------------------------------------------------------------------------- DG2
def t_formation() -> str:
    d = rd("dg2_runs.csv")
    rows = []
    for q in (50, 60):
        g = d[d.q == q]
        f = g.dropna(subset=["formation_export"])
        rows.append({"q": q, "runs": len(g), "formed (F exceeds the threshold)": f"{len(f)} of {len(g)}",
                     "threshold (half the median final F of the passing runs, effort units)": float(g.formation_threshold.iloc[0]),
                     "formation export, min / median / max": f"{int(f.formation_export.min())} / {int(f.formation_export.median())} / {int(f.formation_export.max())}",
                     "formed by u250": f"{int((f.formation_export <= 250).sum())}", "formed after u250": f"{int((f.formation_export > 250).sum())}",
                     "not formed": ";".join(str(int(s)) for s in g[g.formation_export.isna()].seed)})
    return md(rows)


def t_final_shape() -> str:
    d = rd("dg2_runs.csv")
    rows = []
    for q in (50, 60):
        for nm, sel in (("passing", d[(d.q == q) & d["pass"]]), ("failing, formed", d[(d.q == q) & ~d["pass"] & d.formation_export.notna()]),
                        ("failing, not formed (60518)", d[(d.q == q) & ~d["pass"] & d.formation_export.isna()])):
            if len(sel) == 0:
                continue
            rows.append({"q": q, "group": nm, "runs": len(sel), "F(u01600), median": sel.F_final.median(), "e_3(0), median": sel.e0_final.median(),
                         "in-band range, median": sel.in_range_final.median(), "out-of-band mean effort, median": sel.out_mean_final.median(),
                         "share of out nodes with alpha < 1, median": sel.out_alpha_lt1_final.median(),
                         "|d mu / d input 2| in band, median": sel.sens_in_mean_final.median(), "layer-2 saturation in band, median": sel.sat2_in_final.median()})
    return md(rows)


def t_traj() -> str:
    ex = rd("dg2_exports.csv")
    ups = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 575, 800, 1200, 1600]
    f = ex[(ex.q == 60) & (ex.seed == 60518)].set_index("update")
    g = ex[(ex.q == 60) & (ex.seed == 60501) & (ex["round"] == "t3_r2")].set_index("update")
    fe = rd("dg2_runs.csv")
    fe = int(fe[(fe.q == 60) & (fe.seed == 60501) & (fe["round"] == "t3_r2")].formation_export.iloc[0])
    p = rd("dg6_s3_runs.csv")
    ok = p[(p.q == 60) & (p["pass"])]
    passing = ex.merge(ok[["round", "q", "seed"]], on=["round", "q", "seed"])
    med = passing.groupby("update")[["e0", "in_range", "sens_in_mean", "sat2_in"]].median()
    rows = []
    for u in ups:
        rows.append({"update": u, "60518 e(0)": f.loc[u, "e0"], "60518 in-band range": f.loc[u, "in_range"], "60518 sens": f.loc[u, "sens_in_mean"],
                     "60518 sat2": f.loc[u, "sat2_in"], f"60501 (formation export {fe}) e(0)": g.loc[u, "e0"], "60501 sens": g.loc[u, "sens_in_mean"],
                     "60501 sat2": g.loc[u, "sat2_in"], "passing q60 median e(0)": med.loc[u, "e0"], "median sens": med.loc[u, "sens_in_mean"]})
    return md(rows)


def t_bottleneck() -> str:
    ex = rd("dg2_exports.csv").rename(columns={"update": "u"})
    rr = rd("dg2_runs.csv")
    rows = []
    for (r, q, s), g in ex.groupby(["round", "q", "seed"]):
        g = g.sort_values("u").set_index("u")
        e = g.loc[100:300]
        rows.append({"round": r, "q": q, "seed": s, "min_sens": float(e.sens_in_mean.min()), "min_sens_u": int(e.sens_in_mean.idxmin()),
                     "sens_u150": float(g.loc[150, "sens_in_mean"]), "sens_u100": float(g.loc[100, "sens_in_mean"]),
                     "sens_u125": float(g.loc[125, "sens_in_mean"]), "min_e0_u100_300": float(e.e0.min())})
    d = pd.DataFrame(rows).merge(rr[["round", "q", "seed", "formation_export", "pass"]], on=["round", "q", "seed"])
    f = d.dropna(subset=["formation_export"])
    r150 = spearmanr(f.sens_u150, f.formation_export)
    rmin = spearmanr(f.min_sens, f.formation_export)
    r125, r100 = spearmanr(f.sens_u125, f.formation_export), spearmanr(f.sens_u100, f.formation_export)
    head = (f"Spearman correlation with the formation export over the {len(f)} runs that formed: sensitivity at u150 {r150[0]:.3f} (p = {r150[1]:.1e}), "
            f"at u125 {r125[0]:.3f}, at u100 {r100[0]:.3f} (the formation exports are 150 or later, so these two are measured before any formation); "
            f"minimum sensitivity over u100-u300 {rmin[0]:.3f} (p = {rmin[1]:.1e}). Over all {len(d)} runs the minimum of e(0) over u100-u300 is "
            f"{d.min_e0_u100_300.min():.1f}-{d.min_e0_u100_300.max():.1f} (median {d.min_e0_u100_300.median():.1f}) and the minimum sensitivity {d.min_sens.min():.1e}-{d.min_sens.max():.1e} "
            f"(median {d.min_sens.median():.1e}).\n\n")
    top = d.sort_values("min_sens").head(6)
    rows2 = [{"run": label(x["round"], x.q, x.seed), "minimum of |d mu / d input 2| over u100-u300": x.min_sens, "at update": x.min_sens_u,
              "formation export": x.formation_export, "pass": bool(x["pass"])} for _, x in top.iterrows()]
    late = f[f.formation_export >= 325]
    rest = f[f.formation_export < 325]
    a, b, c_, d_ = int((~late["pass"]).sum()), int(late["pass"].sum()), int((~rest["pass"]).sum()), int(rest["pass"].sum())
    p = fisher_exact([[a, b], [c_, d_]])[1]
    tail = (f"\n\nFormation export >= 325 (descriptive): {a} of {len(late)} such runs fail, against {c_} of {len(rest)} with an earlier formation "
            f"(Fisher exact p = {p:.3f}; 60518 excluded, it never formed).")
    return head + md(rows2) + tail


# ---------------------------------------------------------------------------------- DG3
def t_dg3_checks() -> str:
    ck = rd("dg3_checks.csv")
    s3 = rd("dg6_s3_runs.csv")
    m = ck.merge(s3[["round", "q", "seed", "S3", "pass"]], on=["round", "q", "seed"])
    ref = m[(m.q == 60) & m["pass"] & (m["round"] == "t3_r2")]
    one = m[(m.q == 60) & (m.seed == 60518)].iloc[0]
    cols = [("Delta_L25", "Delta, local 25"), ("Delta_L50", "Delta, local 50"), ("Delta_L100", "Delta, local 100"), ("Delta_L200", "Delta, local 200"),
            ("Delta_L400", "Delta, local 400"), ("Delta_L800", "Delta, local 800"), ("Delta_L1600", "Delta, local 1600"),
            ("s_L100", "s (best-response effort at d = 0), local 100"), ("s_L400", "s, local 400"), ("s_L1600", "s, local 1600"),
            ("R_L100", "R, local 100"), ("R_L400", "R, local 400"), ("R_L1600", "R, local 1600"), ("R_tail_L100", "R_tail, local 100"),
            ("R_tail_L400", "R_tail, local 400"), ("R_tail_L1600", "R_tail, local 1600"), ("C_L100", "C, local 100"), ("C_L400", "C, local 400"),
            ("C_L1600", "C, local 1600"), ("first_local_Delta_lt_0.05", "first check with Delta < 0.05 (local)"),
            ("n_checks_Delta_in_0.12_0.15", "checks with Delta in [0.12, 0.15] (of 64)")]
    rows = [{"quantity": h, "60518": one[c], f"{len(ref)} passing q60 runs of T3-R2: min": ref[c].min(), "median": ref[c].median(), "max": ref[c].max()}
            for c, h in cols]
    return md(rows)


def t_dg3_signal() -> str:
    w = rd("dg3_windows.csv")
    s3 = rd("dg6_s3_runs.csv")
    w = w.merge(s3[["round", "q", "seed", "pass"]], on=["round", "q", "seed"])
    hw = rd("dg3_history_windows.csv").merge(s3[["round", "q", "seed", "pass"]], on=["round", "q", "seed"])
    rows = []
    for a, b in ((1, 100), (101, 400), (401, 800), (1201, 1600)):
        sw = w[(w["first"] == a)]
        sh = hw[(hw["first"] == a)]
        ref = sw[(sw.q == 60) & sw["pass"] & (sw["round"] == "t3_r2")]
        refh = sh[(sh.q == 60) & sh["pass"] & (sh["round"] == "t3_r2")]
        one = sw[(sw.q == 60) & (sw.seed == 60518)].iloc[0]
        oneh = sh[(sh.q == 60) & (sh.seed == 60518)].iloc[0]
        for c, hdr, src_o, src_r in (("n_start_tail", "starts in the tail stratum (sum over the window)", one, ref),
                                      ("n_start_near", "starts in the near-tie stratum", one, ref), ("n_start_mid", "starts in the middle stratum", one, ref),
                                      ("mean_effort", "mean effort", one, ref), ("kl_final_epoch", "kl_final_epoch", one, ref),
                                      ("clip_frac", "clip_frac", one, ref), ("adv_raw_std", "adv_raw_std", one, ref),
                                      ("value_loss", "value_loss", one, ref), ("grad_norm_actor_mean", "actor grad norm (pre-clip, mean)", oneh, refh),
                                      ("grad_norm_critic_mean", "critic grad norm (mean)", oneh, refh),
                                      ("entropy_post_update_effort_scale", "entropy (effort scale, after the update)", oneh, refh),
                                      ("conc_mean", "concentration alpha + beta (mean over the buffer)", oneh, refh)):
            rows.append({"local updates": f"{a}-{b}", "quantity": hdr, "60518": src_o[c], "passing q60 (T3-R2): min": src_r[c].min(),
                         "median": src_r[c].median(), "max": src_r[c].max()})
    return md(rows)


def t_dg3_all() -> str:
    s3 = rd("dg6_s3_runs.csv")
    w = rd("dg3_windows.csv").merge(s3[["round", "q", "seed", "class", "pass"]], on=["round", "q", "seed"])
    hw = rd("dg3_history_windows.csv").merge(s3[["round", "q", "seed", "class"]], on=["round", "q", "seed"])
    ex = rd("dg2_exports.csv")
    ex = ex[ex["update"] == 1600].merge(s3[["round", "q", "seed", "class"]], on=["round", "q", "seed"])
    rows = []
    for q in (50, 60):
        for cls in ("regime", "mid", "low"):
            a = w[(w["first"] == 1201) & (w.q == q) & (w["class"] == cls)]
            b = hw[(hw["first"] == 1201) & (hw.q == q) & (hw["class"] == cls)]
            c = ex[(ex.q == q) & (ex["class"] == cls)]
            rows.append({"q": q, "S3 class": cls, "runs": len(a), "failures": int((~a["pass"]).sum()), "kl_final_epoch": a.kl_final_epoch.median(),
                         "clip_frac": a.clip_frac.median(), "actor grad norm, mean": b.grad_norm_actor_mean.median(),
                         "actor grad norm, max": b.grad_norm_actor_max.median(), "entropy (effort scale)": b.entropy_post_update_effort_scale.median(),
                         "concentration (mean)": b.conc_mean.median(), "min alpha on the D_3 grid at u01600": c.alpha_min.median()})
    return md(rows)


# ---------------------------------------------------------------------------------- DG4
def t_dg4_profile() -> str:
    p = rd("dg4_profiles.csv")
    s3 = rd("dg6_s3_runs.csv")
    rows = []
    for (q, s), g in p.groupby(["q", "seed"], sort=False):
        c = s3[(s3.q == q) & (s3.seed == s) & (s3["round"] == "t3_r2")].iloc[0]
        r = {"run": f"q{q} s{s}", "S3": c.S3, "pass": bool(c["pass"])}
        for u in (100, 400, 800, 1600):
            x = g[g["update"] == u].iloc[0]
            r[f"u{u}: max Delta (d*; e, BR)"] = f"{x.Delta_max_over_dw:.4f} ({x.Delta_argmax_d:g}; {x.e_hat_at_argmax:.1f}, {x.a_dev_at_argmax:.1f})"
        rows.append(r)
    return md(rows)


def t_dg4_curves() -> str:
    c = rd("dg4_curves.csv")
    rows = []
    for (q, s, u) in ((60, 60518, 1600), (60, 60518, 100), (50, 60501, 1600), (50, 60515, 1600), (60, 60516, 1600)):
        sub = c[(c.q == q) & (c.seed == s) & (c["update"] == u) & (c.label.str.startswith("node")) & (c.d.isin([0.0, -4.0, -6.0, -36.0, -20.0, 40.0]))]
        for _, x in sub.iterrows():
            rows.append({"run": f"q{q} s{s}", "export": f"u{u:05d}", "d": x.d, "learned effort": x.e_hat, "slope dQ/de there": x.slope_at_e_hat,
                         "one-step best response": x.a_dev, "Delta/DW": x.Delta_over_dw, "local maxima of Q_3(d, .) (efforts)": x.local_maxima_efforts})
    return md(rows)


def t_dg4_counts() -> str:
    c = rd("dg4_curves.csv")
    return (f"Over the {len(c)} curves Q_3(d, .) of DG4 (14 runs x 4 exports x up to 17 gaps), {int((c.n_local_maxima == 1).sum())} have exactly one local maximum "
            f"on the effort grid of step 0.01 ({int((c.n_local_maxima != 1).sum())} have another number).")


# ---------------------------------------------------------------------------------- DG5
def t_dg5_identity() -> str:
    i = rd("dg5_identity.csv")
    rows = [{"q": int(x.q), "seed": int(x.seed), "64 exports": "identical" if x.exports_identical else "DIFFERENT", "stage-3 rows of ms_updates.csv": "identical" if x.updates_identical else "DIFFERENT",
             "ms_checks_stage3.csv": "identical" if x.checks_identical else "DIFFERENT", "all identical": bool(x.all_identical),
             "wall time (s)": x.wall_sec} for _, x in i.iterrows()]
    rec = pd.read_csv(DIAG.parent / "replays" / "replay_identity.csv")
    n_ok = int(rec.all_identical.sum())
    return md(rows) + f"\n\nFreeze arrays `freeze_stage3_{{final,development}}.npz`, rollout clamp counters against the replay's stored-action counts and global-RNG assertions: {n_ok} of {len(rec)} replays bit-identical in all items; global-RNG violations in total {int(rec.rng_violations.sum())}."


def t_dg5_shares() -> str:
    m = rd("dg5_group_medians.csv")
    names = {"failure": "failures (6)", "pass_regime": "passing, clamp regime (4)", "pass_low": "passing, not in the regime (4)"}
    rows = []
    for _, x in m.iterrows():
        rows.append({"group": names[x.group], "updates": f"{int(x['first'])}-{int(x['last'])}", "sample share out of band": x.sample_share_out_median,
                     "gradient share out of band": x.grad_share_out_pooled_median, "sample share, draws at the lower clamp": x.sample_share_clamp_lo_median,
                     "gradient share, draws at the lower clamp": x.grad_share_clamp_lo_pooled_median, "gradient share in band": x.grad_share_in_pooled_median,
                     "|g| total": x.gnorm_total_median, "|g_in|": x.gnorm_in_median, "|g_out|": x.gnorm_out_median})
    return md(rows)


def t_dg5_60518() -> str:
    w = rd("dg5_windows.csv")
    one = w[(w.q == 60) & (w.seed == 60518)]
    rows = [{"updates": f"{int(x['first'])}-{int(x['last'])}", "|g|": x.gnorm_total, "|g_in|": x.gnorm_in, "|g_out|": x.gnorm_out,
             "cos(g_in, g_out)": x.cos_in_out_mean, "share in": x.grad_share_in_pooled, "share out": x.grad_share_out_pooled,
             "sensitivity": x.sens_in_mean, "layer-1 saturation": x.sat1_in, "layer-2 saturation": x.sat2_in} for _, x in one.iterrows()]
    other = w[(w.group == "failure") & ~((w.q == 60) & (w.seed == 60518))]
    og = other.groupby("first")[["gnorm_total", "gnorm_in", "gnorm_out", "cos_in_out_mean", "sens_in_mean", "sat2_in"]].median()
    rows2 = [{"updates": f"{int(a)}-{int(b)}", "|g|": og.loc[a, "gnorm_total"], "|g_in|": og.loc[a, "gnorm_in"], "|g_out|": og.loc[a, "gnorm_out"],
              "cos(g_in, g_out)": og.loc[a, "cos_in_out_mean"], "sensitivity": og.loc[a, "sens_in_mean"], "layer-2 saturation": og.loc[a, "sat2_in"]}
             for a, b in ((1, 100), (101, 400), (401, 800), (801, 1200), (1201, 1600))]
    return md(rows) + "\n\nMedians of the five other failures (formed, in the clamp regime):\n\n" + md(rows2)


def t_dg5_coverage() -> str:
    b = rd("dg5_bins.csv")
    rows = []
    for (q, s), g in b.groupby(["q", "seed"], sort=False):
        w = g[g["first"] == 1201]
        inb = w[w.in_band]
        out = w[~w.in_band]
        rows.append({"run": f"q{q} s{s}", "in-band share of starts": inb.starts_share.sum(), "bins": len(w), "in-band bins": len(inb),
                     "min / max bin share of starts": f"{w.starts_share.min():.4f} / {w.starts_share.max():.4f}",
                     "in-band share of the gradient (pooled)": inb.grad_share_pooled.sum(), "gradient share of the 6 bins with |d| <= 60": w[w.bin_centre.abs() <= 50].grad_share_pooled.sum()})
    return md(rows)


# ---------------------------------------------------------------------------------- DG6
def t_s3_counts() -> str:
    s3 = rd("dg6_s3_runs.csv")
    rows = []
    for scope, sel in (("all 80", s3), ("q = 50", s3[s3.q == 50]), ("q = 60", s3[s3.q == 60])):
        r = {"scope": scope, "runs": len(sel), "S3 > 0.2": int((sel.S3 > 0.2).sum()), "S3 > 0.4": int((sel.S3 > 0.4).sum()),
             "S3 < 0.05": int((sel.S3 < 0.05).sum()), "failures in S3 > 0.2": int(((sel.S3 > 0.2) & ~sel["pass"]).sum()),
             "failures in S3 <= 0.2": int(((sel.S3 <= 0.2) & ~sel["pass"]).sum()), "runs with a stage-2 stretch in S3 > 0.2": int(((sel.S3 > 0.2) & sel.stretch_listed).sum()),
             "runs with a stage-2 stretch in S3 <= 0.2": int(((sel.S3 <= 0.2) & sel.stretch_listed).sum())}
        rows.append(r)
    sub = s3
    a, b = int(((sub.S3 > 0.2) & ~sub["pass"]).sum()), int(((sub.S3 > 0.2) & sub["pass"]).sum())
    c, d = int(((sub.S3 <= 0.2) & ~sub["pass"]).sum()), int(((sub.S3 <= 0.2) & sub["pass"]).sum())
    p = fisher_exact([[a, b], [c, d]])[1]
    a2, b2 = int(((sub.S3 > 0.2) & sub.stretch_listed).sum()), int(((sub.S3 > 0.2) & ~sub.stretch_listed).sum())
    c2, d2 = int(((sub.S3 <= 0.2) & sub.stretch_listed).sum()), int(((sub.S3 <= 0.2) & ~sub.stretch_listed).sum())
    p2 = fisher_exact([[a2, b2], [c2, d2]])[1]
    return md(rows) + (f"\n\nFailure rate {a} of {a + b} ({cp(a, a + b)}) against {c} of {c + d} ({cp(c, c + d)}), Fisher exact p = {p:.4f}. "
                       f"Stage-2 stretch {a2} of {a2 + b2} ({cp(a2, a2 + b2)}) against {c2} of {c2 + d2} ({cp(c2, c2 + d2)}), Fisher exact p = {p2:.1e}.")


def t_s3_groups() -> str:
    g = rd("dg6_groups.csv")
    return md(g.to_dict("records"), ["scope", "group", "runs", "failures", "with_stretch", "median_eta_3", "median_peak_T_signed",
                                    "median_rmse_pos_T_over_eT0", "median_Gmax_full"],
              ["scope", "group", "runs", "failures", "with a stage-2 stretch", "median eta_3", "median peak_T_signed", "median rmse_pos_T/e_T(0)*", "median Gmax_full"])


def t_s3_rel() -> str:
    r = rd("dg6_relations.csv")
    a = r[~r.scope.str.startswith("timing")]
    rows = [{"scope": x.scope, "quantity": x.quantity, "n": int(x.n), "Spearman correlation with S3": x.spearman_with_S3} for _, x in a.iterrows()]
    t = r[r.scope.str.startswith("timing")]
    rows2 = [{"scope": x.scope, "quantity (effort units)": x.quantity, "regime runs, median": x.regime_median, "other runs, median": x.other_median,
              "Mann-Whitney p": x.mannwhitney_p} for _, x in t.iterrows()]
    return md(rows) + "\n\nTiming (reference update = the run's own regime onset, or the median onset of the regime runs of its q):\n\n" + md(rows2)


def t_t2_analogue() -> str:
    t = rd("dg6_t2_analogue.csv")
    rows = []
    for q in (50, 60):
        g = t[t.q == q]
        rows.append({"q": q, "runs": len(g), "max": g.S2_analogue.max(), "median": g.S2_analogue.median(), "below 0.005": int((g.S2_analogue < 0.005).sum()),
                     "above 0.05": int((g.S2_analogue > 0.05).sum())})
    rows.append({"q": "both", "runs": len(t), "max": t.S2_analogue.max(), "median": t.S2_analogue.median(), "below 0.005": int((t.S2_analogue < 0.005).sum()),
                 "above 0.05": int((t.S2_analogue > 0.05).sum())})
    return md(rows)


def t_s3_failures() -> str:
    s3 = rd("dg6_s3_runs.csv")
    rows = []
    for r, q, s in FAIL_KEYS:
        x = s3[(s3["round"] == r) & (s3.q == q) & (s3.seed == s)].iloc[0]
        rows.append({"run": label(r, q, s), "S3": x.S3, "regime onset (local update)": x.onset_local, "e_3(0) at the onset": x.e0_at_onset,
                     "e_3(0) at u1600": x.e0_u1600, "stage-2 stretch": bool(x.stretch_listed)})
    return md(rows)


# ---------------------------------------------------------------------------------- DG8
def t_entry() -> str:
    e = rd("dg8_phase_entry.csv")
    rows = []
    for cls in ("regime", "mid", "low"):
        g = e[e["class"] == cls]
        rows.append({"S3 class": cls, "runs": len(g), "with a final stage-2 stretch": int(g.has_final_stretch.sum()),
                     "with >= 1 in-band node below 1 at the phase entry": int((g.entry_nodes_lt1 > 0).sum()),
                     "median in-band nodes below 1 at entry": g.entry_nodes_lt1.median(), "max": g.entry_nodes_lt1.max(),
                     "median entry effort over the band": g.entry_median_effort_in_band.median()})
    s = e[e.has_final_stretch]
    rows2 = [{"run": label(x["round"], x.q, x.seed), "S3": x.S3, "final stretch nodes": x.final_stretch_nodes, "of which below 1 at entry": x.final_stretch_nodes_lt1_at_entry,
              "median entry effort at the stretch nodes": x.final_stretch_entry_median_effort} for _, x in s.iterrows()]
    return md(rows) + "\n\nThe ten runs with a final stretch:\n\n" + md(rows2)


# ---------------------------------------------------------------------------------- DG7
def t_classes() -> str:
    d = rd("dg7_failures.csv")
    rows = []
    for _, x in d.iterrows():
        rows.append({"run": label(x["round"], x.q, x.seed), "failing": x.failing, "eta_3": x.eta_3, "rmse_pos": x.rmse_pos, "Gmax_full": x.Gmax_full,
                     "S3": x.S3, "F(u01600)": x.F_final, "formation export": x.formation_export,
                     "eta_3 > 0.005 at n of the last 16 exports": x["eta3_last16_n_above_0.005"], "rank of the final eta_3 (1 = largest)": x.eta3_final_rank_from_top,
                     "dev Delta, local 1600": x.dev_Delta_L1600, "stage-2 stretch nodes": x.stage2_final_stretch_nodes})
    return md(rows)


# ---------------------------------------------------------------------------------- retro
def t_retro() -> str:
    t = rd("retro_triggers.csv")
    rows = []
    for _, x in t.iterrows():
        rows.append({"trigger at local update 1600": x.trigger, "fires on": int(x.n_fired), "of which failed": int(x.fired_and_failed), "failing runs not caught": int(x.failing_missed),
                     "extra updates": int(x.extra_updates), "extra cost (share of the 80 x 3800 updates)": x.extra_cost_fraction})
    return md(rows)


def t_retro_runs() -> str:
    t = rd("retro_triggers.csv")
    rows = []
    for _, x in t.iterrows():
        rows.append({"trigger": x.trigger, "fired runs": x.fired_runs if isinstance(x.fired_runs, str) else "-", "failing runs not caught": x.missed_runs if isinstance(x.missed_runs, str) else "-"})
    return md(rows[:1] + rows[2:3] + rows[4:5] + rows[6:7])


def t_retro_margins() -> str:
    c = rd("retro_checks_local1600.csv")
    o = c[c.label != "r2_q60_s60518"]
    top = c.sort_values("Delta", ascending=False).head(14)
    rows = [{"run": x.label, "pass": bool(x["pass"]), "Delta": x.Delta, "R": x.R, "R_tail": x.R_tail, "C": x.C} for _, x in top.iterrows()]
    return md(rows) + (f"\n\nLargest values among the other 79 runs (without 60518): R {o.R.max():.4f} ({o.loc[o.R.idxmax(), 'label']}), "
                       f"R_tail {o.R_tail.max():.4f} ({o.loc[o.R_tail.idxmax(), 'label']}); 60518: R {c[c.label == 'r2_q60_s60518'].R.iloc[0]:.4f}, "
                       f"R_tail {c[c.label == 'r2_q60_s60518'].R_tail.iloc[0]:.4f}.")

def t_mu_clamp() -> str:
    d = rd("dg2_runs.csv").merge(rd("dg6_s3_runs.csv")[["round", "q", "seed", "S3", "class"]], on=["round", "q", "seed"])
    rows = []
    for q in (50, 60):
        for cls in ("regime", "mid", "low"):
            g = d[(d.q == q) & (d["class"] == cls)]
            rows.append({"q": q, "S3 class": cls, "runs": len(g), "runs with a tail node at the mean clamp at some export": int(g.first_export_tail_mu_at_clamp.notna().sum()),
                         "smallest Beta mean on the D_3 grid at u01600, median": g.mu_min_final.median(),
                         "share of out-of-band nodes at the mean clamp, median": g.out_mu_at_clamp_final.median(),
                         "share of out-of-band nodes with mean < 1e-4, median": g.out_mu_lt_1e4_final.median()})
    return md(rows)


def t_t2_failure() -> str:
    t = rd("dg6_t2_analogue.csv")
    x = t[(t.q == 50) & (t.seed == 30510)].iloc[0]
    r50 = t[t.q == 50].S2_analogue.rank(ascending=False)
    return (f"The one failed run of the 40 v2.0 confirmation runs, q = 50 seed 30510, has the T = 2 analogue {x.S2_analogue:.4f}, the largest of the 40 "
            f"(second largest {t.S2_analogue.sort_values(ascending=False).iloc[1]:.4f}).")

def t_dg4_agree() -> str:
    p = rd("dg4_profiles.csv")
    s3 = rd("dg6_s3_runs.csv")
    rows = []
    for (q, s), g in p.groupby(["q", "seed"], sort=False):
        x = g[g["update"] == 1600].iloc[0]
        st = s3[(s3.q == q) & (s3.seed == s) & (s3["round"] == "t3_r2")].iloc[0]
        rows.append({"run": f"q{q} s{s}", "stored eta_3 (final-tier verifier)": st.eta_3, "calculator max Delta_3 at u01600": x.Delta_max_over_dw,
                     "difference": x.Delta_max_over_dw - st.eta_3})
    mx = max(abs(r["difference"]) for r in rows)
    return md(rows) + f"\n\nLargest absolute difference over the 14 runs: {mx:.2e} (units of DW)."


TABLES: Dict[str, Callable[[], str]] = {
    "inventory": t_inventory, "formation": t_formation, "final_shape": t_final_shape, "traj": t_traj, "bottleneck": t_bottleneck,
    "dg3_checks": t_dg3_checks, "dg3_signal": t_dg3_signal, "dg3_all": t_dg3_all, "dg4_profile": t_dg4_profile, "dg4_curves": t_dg4_curves,
    "dg4_counts": t_dg4_counts, "dg5_identity": t_dg5_identity, "dg5_shares": t_dg5_shares, "dg5_60518": t_dg5_60518,
    "dg5_coverage": t_dg5_coverage, "s3_counts": t_s3_counts, "s3_groups": t_s3_groups, "s3_rel": t_s3_rel, "t2_analogue": t_t2_analogue,
    "s3_failures": t_s3_failures, "entry": t_entry, "classes": t_classes, "retro": t_retro, "retro_runs": t_retro_runs,
    "retro_margins": t_retro_margins, "mu_clamp": t_mu_clamp, "dg4_agree": t_dg4_agree, "t2_failure": t_t2_failure}


def main() -> int:
    """Write all tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in TABLES.items():
        (OUT / f"{name}.md").write_text(fn() + "\n")
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
