#!/usr/bin/env python3
"""Pilot 4: render the report tables (Markdown) from the analysis CSVs.

Output: results/v2_pilots/pilot4/analysis/report_tables.md (one section per report table; each
table is preceded by its source path).

Usage: python tools/v2/pilot4_tables.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "results" / "v2_pilots" / "pilot4" / "analysis"
OUT = A / "report_tables.md"


def fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "—"
        if v == 0:
            return "0"
        return f"{v:.4g}"
    return str(v)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def miqr(x: pd.Series) -> str:
    return f"{x.median():.4g} [{x.quantile(.25):.4g}, {x.quantile(.75):.4g}]"


def main() -> int:
    s = []

    def sec(title, src, df):
        s.append(f"### {title}\n\nSource: `{src}`\n\n{md(df)}\n")

    rel = lambda p: str(p.relative_to(ROOT))  # noqa: E731
    # 1a
    cv = pd.read_csv(A / "root_game" / "own_curvature.csv")
    sec("1a own curvature", rel(A / "root_game" / "own_curvature.csv"),
        cv[["q", "tier", "fit_half_width", "fit_points", "BR_minus_e1star", "own_curv_over_2k", "ev2pp_over_2k",
            "ref_ev2pp_over_2k", "slope_implied_by_curv"]])
    sl = pd.read_csv(A / "root_game" / "br_slope.csv")
    t = sl.pivot_table(index=["q", "tier", "fit_half_width"], columns="h", values="slope").reset_index()
    t.columns = [c if not isinstance(c, float) else f"h={c:g}" for c in t.columns]
    t["reference"] = t.q.map({50: -0.961, 60: -0.309})
    sec("1a BR slope by h", rel(A / "root_game" / "br_slope.csv"), t)
    # 1b
    f = A / "fluctuation"
    a = pd.read_csv(f / "acf_summary_every20.csv")
    for st in (650, 700):
        for kind in ("centred", "about_e1star"):
            t = a[(a.segment_start == st) & (a.kind == kind) & (a.lag_updates <= 160)].copy()
            t["v"] = [f"{m:.2f} [{l:.2f}, {h:.2f}]" for m, l, h in zip(t["median"], t.q25, t.q75)]
            p = t.pivot_table(index=["q", "arm"], columns="lag_updates", values="v", aggfunc="first").reset_index()
            p.columns = [c if isinstance(c, str) else f"lag {c}" for c in p.columns]
            sec(f"1b ACF every-20 record, segment u>={st}, {kind} (median [IQR] over 10 seeds)", rel(f / "acf_summary_every20.csv"), p)
    b = pd.read_csv(f / "acf_summary_exports25.csv")
    t = b[b.kind == "centred"].copy()
    t["v"] = [f"{m:.2f} [{l:.2f}, {h:.2f}]" for m, l, h in zip(t["median"], t.q25, t.q75)]
    p = t.pivot_table(index=["segment_start", "q", "arm"], columns="lag_updates", values="v", aggfunc="first").reset_index()
    p.columns = [c if isinstance(c, str) else f"lag {c}" for c in p.columns]
    sec("1b ACF of the 25-update exports, centred", rel(f / "acf_summary_exports25.csv"), p)
    sec("1b window regression", rel(f / "window_regression.csv"), pd.read_csv(f / "window_regression.csv"))
    # 1c / 2a / 2b candidates
    c = pd.read_csv(A / "candidates_all.csv")
    tail = c[c.kind == "tail"]
    s2 = ["stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_peak_locfree_rel_err", "stage2_rmse_pos_over_g2_0",
          "stage2_tail_mean", "stage2_tail_max", "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "stage2_sym_err_max",
          "eta_T_over_dw", "DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max"]
    s1 = ["e1_cand", "stage1_rel_err_signed", "stage1_rel_err_abs", "learning_rel", "inherited_rel", "Gmax_full_over_dw",
          "EXP_root_over_dw", "dReach_over_dw"]
    for fam, mets, title in (("ext", s2, "1c Phase A extension u1600, stage-2 candidates (medians over 10 seeds)"),
                             ("p3", s1, "1c Pilot 3 u1000, stage-1 candidates (medians over 10 seeds)"),
                             ("2a", s2, "2a end-of-phase candidates u1600 (medians over 10 seeds)"),
                             ("2b", s1, "2b end-of-phase candidates u2200 (medians over 10 seeds)")):
        g = tail[tail.family == fam].groupby(["q", "arm", "K"])[mets].median().reset_index()
        sec(title, rel(A / "candidates_all.csv"), g)
    # Gmax location (t*, d*) for stage-1 families
    loc = tail[tail.family.isin(["p3", "2b"])].groupby(["family", "q", "arm", "K"]).apply(
        lambda g: pd.Series({"t*=1 count": int((g.Gmax_full_t == 1).sum()), "t*=2 count": int((g.Gmax_full_t == 2).sum()),
                             "d* values": " ".join(f"{x:g}" for x in sorted(g.Gmax_full_d.unique()))})).reset_index()
    sec("1c/2b location (t*, d*) of Gmax_full", rel(A / "candidates_all.csv"), loc)
    ps = pd.read_csv(A / "paired_summary.csv")
    keep = ["family", "q", "a", "b", "metric", "median", "n_better", "n_neg", "n_pos", "n_zero", "boot_ci95_lo", "boot_ci95_hi"]
    sec("Paired K vs K=1", rel(A / "paired_summary.csv"), ps[ps.comparison == "K_vs_K1"][keep])
    sec("Paired decay - constant (candidates)", rel(A / "paired_summary.csv"), ps[ps.comparison == "decay_minus_constant"][keep])
    pr = pd.read_csv(A / "paired_summary_run_records.csv")
    sec("Paired decay - constant (run records)", rel(A / "paired_summary_run_records.csv"),
        pr[["family", "q", "metric", "median", "n_better", "n_neg", "n_pos", "boot_ci95_lo", "boot_ci95_hi"]])
    # per-run final tables
    rr = pd.read_csv(A / "run_records.csv")
    for fam in ("2a", "2b"):
        r = rr[rr.family == fam]
        g = r.groupby(["q", "arm"]).agg(n=("seed", "size"), commit=("commit", lambda x: ",".join(sorted(set(x)))),
                                        dirty=("dirty", lambda x: ",".join(sorted(set(map(str, x))))),
                                        lr_first=("actor_lr_first", "first"), lr_last=("actor_lr_last", "median"),
                                        n_would_fire=("would_fire_update", lambda x: int(x.notna().sum())),
                                        wf_median=("would_fire_update", "median"), wf_min=("would_fire_update", "min"),
                                        wf_max=("would_fire_update", "max"), wall_median=("phase_wall_sec", "median"),
                                        kl_median=("kl_median", "median"), clip_median=("clip_median", "median"),
                                        adv_s1_std=("adv_s1_std_median", "median"), adv_used_std=("adv_used_std_median", "median")).reset_index()
        sec(f"{fam} run records", rel(A / "run_records.csv"), g)
    fin2b = tail[(tail.family == "2b") & (tail.K == 1)].merge(
        rr[rr.family == "2b"][["q", "seed", "arm", "within_run_sd_e1_last5", "within_run_range_e1_last5", "kl_final_epoch",
                               "clip_frac", "phase_wall_sec", "would_fire_update"]], on=["q", "seed", "arm"])
    fin2b["learning_band"] = [f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(fin2b.learning_rel_lo, fin2b.learning_rel_hi)]
    fin2b["inherited_band"] = [f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(fin2b.inherited_rel_lo, fin2b.inherited_rel_hi)]
    sec("2b final checkpoint u2200, all runs", rel(A / "candidates_all.csv") + " + run_records.csv",
        fin2b[["q", "seed", "arm", "e1_cand", "stage1_rel_err_signed", "learning_rel", "learning_band", "inherited_rel", "inherited_band",
               "sigma_effort_at_0_t1", "within_run_sd_e1_last5", "within_run_range_e1_last5", "Gmax_full_over_dw", "Gmax_full_t",
               "Gmax_full_d", "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw", "kl_final_epoch",
               "clip_frac", "phase_wall_sec", "would_fire_update"]].sort_values(["q", "seed", "arm"]))
    fin2a = tail[(tail.family == "2a") & (tail.K == 1)]
    sec("2a final u1600, all runs", rel(A / "candidates_all.csv"),
        fin2a[["q", "seed", "arm"] + s2 + ["sigma_effort_at_0_t2"]].sort_values(["q", "seed", "arm"]))
    sec("2b stability", rel(A / "stability_2b.csv"), pd.read_csv(A / "stability_2b.csv"))
    rd = pd.read_csv(A / "rng_divergence.csv")
    out = []
    for (fam, q), g in rd.groupby(["family", "q"]):
        row = {"family": fam, "q": q}
        for st in ("env", "learn", "opp", "start", "minibatch"):
            v = pd.to_numeric(g[st], errors="coerce")
            row[st] = "never (10/10)" if v.isna().all() else f"{int(v.notna().sum())}/10; min {v.min():.0f}, median {v.median():.0f}, max {v.max():.0f}"
        out.append(row)
    sec("RNG divergence decay vs constant (global update)", rel(A / "rng_divergence.csv"), pd.DataFrame(out))
    # 1d
    rf = A / "repr_floor"
    fits = pd.read_csv(rf / "fits.csv")
    sec("1d supervised fits (each init)", rel(rf / "fits.csv"),
        fits[["q", "init_seed", "steps", "stop", "final_loss_mse", "stage2_peak_rel_err_signed", "stage2_peak_locfree_rel_err",
              "stage2_peak_locfree_argmax_d", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean", "stage2_tail_max",
              "stage2_sym_err_max", "eta_T_over_dw", "Gmax_full_over_dw", "max_abs_resid", "max_abs_resid_at_d",
              "tail_min_fit", "tail_max_fit"]])
    med = fits.groupby("q")[["stage2_peak_rel_err_signed", "stage2_peak_locfree_rel_err", "stage2_rmse_pos_over_g2_0",
                             "stage2_tail_mean", "stage2_tail_max", "stage2_sym_err_max", "eta_T_over_dw", "Gmax_full_over_dw"]].median().reset_index()
    sec("1d supervised fits (median over 5 inits)", rel(rf / "fits.csv"), med)
    rl = pd.read_csv(rf / "rl_u1600.csv")
    rlm = rl.groupby("q")[["stage2_peak_rel_err_signed", "stage2_peak_locfree_rel_err", "stage2_rmse_pos_over_g2_0",
                           "stage2_tail_mean", "stage2_tail_max", "stage2_sym_err_max", "eta_T_over_dw"]].median().reset_index()
    sec("1d RL u1600 (median over 10 seeds), same metrics", rel(rf / "rl_u1600.csv"), rlm)
    sec("1d three-way peak gap at d = 0 (effort units)", rel(rf / "three_way_peak_gap.csv"), pd.read_csv(rf / "three_way_peak_gap.csv"))
    # gate distributions
    gd = pd.read_csv(A / "gate_distribution_tables.csv")
    for ph in ("A", "B"):
        sec(f"Distribution tables, Phase {ph}", rel(A / "gate_distribution_tables.csv"),
            gd[gd.phase == ph].drop(columns=["phase", "n"]).sort_values(["metric", "q", "arm", "K"]))
    OUT.write_text("\n".join(s))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
