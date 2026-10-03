#!/usr/bin/env python3
"""Pilot 4 sections 1c, 2a, 2b and the gate-distribution tables (descriptive).

Inputs
  1c  results/v2_pilots/pilot3/q*/seed*/{B2_frozen_s1norm, B2_frozen_s1norm_mean}/weights (stage 1;
      stage 2 frozen = Pilot-1 parent actor), results/v2_pilots/phaseA_ext/.../weights (stage 2)
  2a  results/v2_pilots/pilot4_A/q*/seed*/{constant, decay}/   (u1201..u1600, parent state_u01200)
  2b  results/v2_pilots/pilot4_B/q*/seed*/{B2_mean_constant, B2_mean_decay}/ (u1601..u2200,
      stage 2 frozen = phaseA_ext actor at u1600)
  bands  results/v2_pilots/induced_band/{parent_bands.csv (Pilot 3), parent4_bands.csv (2b)}
Every metric is ``utils.v2_metrics.evaluate`` on the development tier (Beta mean), as in Pilots 1-3
and the extension. candidate_K = pointwise average of the last K weight exports' deterministic
mappings for each TRAINED stage (tools/v2/pilot4_common.py); frozen stages unchanged.
Bootstrap: percentile CI of the mean paired difference, 10000 resamples, numpy seed 20261001.

Outputs: results/v2_pilots/pilot4/analysis/*.csv, reports/v2/figures/pilot4/*.png

Usage: python tools/v2/pilot4_analysis.py [--workers 60]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from common import spec_for  # noqa: E402
from decomposition import rows_for  # noqa: E402
from pilot2_analysis import _actor_fns  # noqa: E402
from pilot4_common import BOOT_SEED, average, compose, eval_row, last_k, mean_fn, paired_summary  # noqa: E402
from rng_divergence import first_divergence  # noqa: E402

V2 = ROOT / "results" / "v2_pilots"
P1, P3, PX, P4A, P4B = V2 / "pilot1", V2 / "pilot3", V2 / "phaseA_ext", V2 / "pilot4_A", V2 / "pilot4_B"
BANDS = V2 / "induced_band"
OUT = V2 / "pilot4" / "analysis"
FIG = ROOT / "reports" / "v2" / "figures" / "pilot4"
SEEDS = range(10501, 10511)
QS = (50, 60)

# (family, run-dir pattern, arms, K values, end update, trained stage)
FAMILIES = {
    "ext": (PX, ("expected_ext",), (1, 4, 8, 16), 1600, 2),
    "2a": (P4A, ("constant", "decay"), (1, 4, 8), 1600, 2),
    "p3": (P3, ("B2_frozen_s1norm", "B2_frozen_s1norm_mean"), (1, 4, 8, 12), 1000, 1),
    "2b": (P4B, ("B2_mean_constant", "B2_mean_decay"), (1, 4, 8, 12), 2200, 1),
}
SHORT = {"expected_ext": "ext", "constant": "constant", "decay": "decay", "B2_frozen_s1norm": "stochastic",
         "B2_frozen_s1norm_mean": "mean", "B2_mean_constant": "constant", "B2_mean_decay": "decay"}
LOWER_BETTER = {  # metric -> True if smaller is better, None if no preferred direction
    "stage2_peak_rel_err_signed": None, "stage2_peak_rel_err_abs": True, "stage2_peak_locfree_rel_err": None,
    "stage2_peak_locfree_rel_err_abs": True, "stage2_rmse_pos_over_g2_0": True, "stage2_tail_mean": True,
    "stage2_tail_max": True, "stage2_tail_mean_over_g2_0": True, "stage2_tail_max_over_g2_0": True,
    "stage2_sym_err_max": True, "eta_T_over_dw": True, "DeltaT_over_dw_on_max": True,
    "DeltaT_over_dw_on_mean_cellmass_weighted": True, "DeltaT_over_dw_off_max": True,
    "stage1_rel_err_signed": None, "stage1_rel_err_abs": True, "learning_rel": None, "learning_rel_abs": True,
    "inherited_rel": None, "Gmax_full_over_dw": True, "EXP_root_over_dw": True, "dReach_over_dw": True,
    "Deltamax_all_over_dw": True, "dFull_over_dw": True, "sigma_effort_at_0_t1": None, "sigma_effort_at_0_t2": None,
    "within_run_sd_e1_last5": True, "within_run_range_e1_last5": True, "kl_final_epoch": None, "clip_frac": None,
    "phase_wall_sec": True,
}
STAGE2_REPORT = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_peak_locfree_rel_err",
                 "stage2_peak_locfree_rel_err_abs", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean", "stage2_tail_max",
                 "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0", "stage2_sym_err_max", "eta_T_over_dw",
                 "DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max")
STAGE1_REPORT = ("stage1_rel_err_signed", "stage1_rel_err_abs", "learning_rel", "learning_rel_abs", "inherited_rel",
                 "Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw")
GATE_METRICS = ("eta_T_over_dw", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0",
                "stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_peak_locfree_rel_err",
                "stage1_rel_err_abs", "Gmax_full_over_dw", "EXP_root_over_dw")


def exports(run_dir: Path, after: int):
    return [f for f in sorted(glob.glob(str(run_dir / "weights" / "u*.npz"))) if int(os.path.basename(f)[1:6]) > after]


def frozen_stage2(family: str, q: int, seed: int) -> str:
    """Actor of the frozen stage 2 (the parent) for stage-1 families."""
    if family == "p3":
        return str(P1 / f"q{q}" / f"seed{seed}" / "expected" / "state_end_A.pt")
    return str(PX / f"q{q}" / f"seed{seed}" / "expected_ext" / "state_u01600.pt")


def _job(job):
    family, q, seed, arm, K, files, u_end, kind = job
    spec = spec_for(q)
    trained = FAMILIES[family][4]
    pols = [mean_fn(f, spec) for f in files]
    cand = average(pols)
    beta_fn = None
    if K == 1:
        mb = _actor_fns(files[0], spec)[1]
        beta_fn = mb
    if trained == 1:
        par_m, par_b = _actor_fns(frozen_stage2(family, q, seed), spec)
        cand = compose(cand, par_m)
        if beta_fn is not None:
            live_b = beta_fn
            beta_fn = lambda t, d: par_b(t, d) if t == 2 else live_b(t, d)  # noqa: E731
    row = eval_row(cand, spec, beta_fn=beta_fn, full=(trained == 1))
    row.update({"family": family, "q": q, "seed": seed, "arm": SHORT[arm], "K": K, "update": u_end, "kind": kind,
                "first_export": int(os.path.basename(files[0])[1:6]), "n_exports": len(files)})
    if trained == 1:
        row["e1_cand"] = float(cand(1, np.zeros(1))[0])
    return row


def jobs():
    out = []
    for fam, (root, arms, Ks, end, trained) in FAMILIES.items():
        after = {"ext": 400, "2a": 1200, "p3": 400, "2b": 1600}[fam]
        for q in QS:
            for seed in SEEDS:
                for arm in arms:
                    ex = exports(root / f"q{q}" / f"seed{seed}" / arm, after)
                    if int(os.path.basename(ex[-1])[1:6]) != end:
                        raise RuntimeError(f"{fam} q{q} s{seed} {arm}: last export {ex[-1]}")
                    for K in Ks:
                        out.append((fam, q, seed, arm, K, last_k(ex, K), end, "tail"))
                    if fam in ("2a", "2b"):          # learning curves: every export (K = 1)
                        for f in ex[:-1]:
                            out.append((fam, q, seed, arm, 1, [f], int(os.path.basename(f)[1:6]), "curve"))
    return out


def add_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    pb = pd.read_csv(BANDS / "parent_bands.csv").set_index(["q", "seed"])
    p4 = pd.read_csv(BANDS / "parent4_bands.csv").set_index(["q", "seed"])
    rows = []
    for r in df.itertuples():
        if r.family not in ("p3", "2b"):
            rows.append({})
            continue
        band = (pb if r.family == "p3" else p4).loc[(r.q, r.seed)]
        rows.append(rows_for(float(r.e1_cand), band, float(band["g1"])))
    dec = pd.DataFrame(rows, index=df.index)
    for c in dec.columns:
        df[c if c != "e1_at_0" else "e1_band_input"] = dec[c]
    df["learning_rel_abs"] = df["learning_rel"].abs()
    df["stage2_peak_locfree_rel_err_abs"] = df["stage2_peak_locfree_rel_err"].abs()
    return df


def paired_vs(df: pd.DataFrame, by_fam, a_sel, b_sel, metrics, label) -> pd.DataFrame:
    """Paired (a - b) per (q, seed) for every metric; a_sel / b_sel are (arm, K)."""
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    t = df[(df.family == by_fam) & (df.kind == "tail")]
    for q in QS:
        A = t[(t.q == q) & (t.arm == a_sel[0]) & (t.K == a_sel[1])].set_index("seed")
        B = t[(t.q == q) & (t.arm == b_sel[0]) & (t.K == b_sel[1])].set_index("seed")
        for m in metrics:
            if m not in A.columns or A[m].isna().all():
                continue
            dv = np.array([A.loc[s, m] - B.loc[s, m] for s in SEEDS], float)
            out.append({"comparison": label, "family": by_fam, "q": q, "a": f"{a_sel[0]} K={a_sel[1]}",
                        "b": f"{b_sel[0]} K={b_sel[1]}", "metric": m, **paired_summary(dv, LOWER_BETTER.get(m), rng)})
    return pd.DataFrame(out)


def run_records() -> pd.DataFrame:
    rows = []
    for fam, root, arms in (("2a", P4A, ("constant", "decay")), ("2b", P4B, ("B2_mean_constant", "B2_mean_decay"))):
        ph = "A" if fam == "2a" else "B"
        for q in QS:
            for seed in SEEDS:
                for arm in arms:
                    d = root / f"q{q}" / f"seed{seed}" / arm
                    man = json.load(open(d / "manifest.json"))
                    rs = json.load(open(d / "v2_run_summary.json"))
                    up = pd.read_csv(d / "v2_updates.csv")
                    hist = json.load(open(d / "train_history.json"))["history"]
                    r = {"family": fam, "q": q, "seed": seed, "arm": SHORT[arm], "commit": man["git"]["short"],
                         "dirty": man["git"]["dirty"], "parent": os.path.relpath(man["parent_checkpoint"], ROOT),
                         "lr_decay": json.dumps(man["input_config"].get("lr_decay")),
                         "actor_lr_first": hist[0]["actor_lr"], "actor_lr_last": hist[-1]["actor_lr"],
                         "critic_lr_last": hist[-1]["critic_lr"], "n_updates": len(hist),
                         "would_have_fired": json.dumps(rs["would_have_fired"][ph]),
                         "would_fire_update": (rs["would_have_fired"][ph] or {}).get("global_update"),
                         "phase_wall_sec": rs["phase_timing"][ph]["wall_sec"],
                         "kl_final_epoch": float(up.kl_final_epoch.iloc[-1]), "clip_frac": float(up.clip_frac.iloc[-1]),
                         "kl_median": float(up.kl_final_epoch.median()), "clip_median": float(up.clip_frac.median())}
                    for c in ("adv_all_mean", "adv_all_std", "adv_s1_mean", "adv_s1_std", "adv_used_std"):
                        r[f"{c}_median"] = float(up[c].median())
                    if fam == "2b":
                        dt = json.load(open(d / "drift_test.json"))
                        r["drift_test_pass"] = dt.get("pass")
                        r["snapshot_drift_max"] = max(dt["max_abs_diff_vs_freeze_time"].values())
                        last = pd.read_csv(d / "v2_checkpoints.csv").iloc[-1]
                        for k in ("e1_at_0", "Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw", "sigma_effort_at_0_t1"):
                            r[f"ckpt_{k}"] = float(last[k])
                        r["ckpt_update"] = int(last["update"])
                    rows.append(r)
    return pd.DataFrame(rows)


def quantile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t = df[(df.kind == "tail") & (df.family.isin(["2a", "2b"]))]
    for (fam, arm, K, q), g in t.groupby(["family", "arm", "K", "q"]):
        phase = "A" if fam == "2a" else "B"
        for m in GATE_METRICS:
            if phase == "A" and m in ("stage1_rel_err_abs", "Gmax_full_over_dw", "EXP_root_over_dw"):
                continue          # stage 1 untrained in Phase A: not defined
            x = g[m].astype(float)
            rows.append({"phase": phase, "arm": arm, "K": K, "q": q, "metric": m, "n": int(x.size), "min": x.min(),
                         "p10": x.quantile(.10), "p25": x.quantile(.25), "median": x.median(), "p75": x.quantile(.75),
                         "p90": x.quantile(.90), "max": x.max()})
    return pd.DataFrame(rows)


def plots(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    col = {"constant": "#1f77b4", "decay": "#d62728"}
    spec = {"2a": [("stage2_peak_rel_err_signed", "peak rel. err (signed)"), ("stage2_peak_locfree_rel_err", "loc.-free peak err"),
                   ("stage2_rmse_pos_over_g2_0", "RMSE / e2*(0)"), ("stage2_tail_mean", "tail mean"),
                   ("stage2_tail_max", "tail max"), ("stage2_sym_err_max", "symmetry err"), ("eta_T_over_dw", "eta_2"),
                   ("sigma_effort_at_0_t2", "sigma_2(0)")],
            "2b": [("stage1_rel_err_signed", "stage-1 rel. err (signed)"), ("learning_rel", "learning term / e1*"),
                   ("sigma_effort_at_0_t1", "sigma_1(0)"), ("Gmax_full_over_dw", "Gmax_full / DW"),
                   ("EXP_root_over_dw", "EXP_root / DW")]}
    for fam, panels in spec.items():
        c = df[(df.family == fam) & (df.K == 1)]
        for q in QS:
            n = len(panels)
            fig, axes = plt.subplots(2 if n > 5 else 1, 4 if n > 5 else n, figsize=(17, 7) if n > 5 else (20, 3.6))
            for ax, (m, lab) in zip(np.ravel(axes), panels):
                for arm in ("constant", "decay"):
                    g = c[(c.q == q) & (c.arm == arm)].groupby("update")[m]
                    med = g.median()
                    ax.plot(med.index, med.values, color=col[arm], lw=1.5, label=arm)
                    ax.fill_between(med.index, g.quantile(.25).values, g.quantile(.75).values, color=col[arm], alpha=.18, lw=0)
                ax.set_title(lab, fontsize=9)
                ax.set_xlabel("global update")
                if "signed" in m or m in ("learning_rel", "stage2_peak_locfree_rel_err"):
                    ax.axhline(0, color="0.5", lw=.8)
            np.ravel(axes)[0].legend(frameon=False, fontsize=8)
            ph = "Phase A u1200-u1600" if fam == "2a" else "Phase B u1600-u2200"
            fig.suptitle(f"Pilot 4 {fam} ({ph}), q={q}: median and IQR over 10 seeds (weights every 25 updates, dev tier)",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(FIG / f"curves_{fam}_q{q}.png", dpi=120)
            plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=60)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    js = jobs()
    with Pool(a.workers) as pool:
        df = pd.DataFrame(pool.map(_job, js, chunksize=1))
    df = add_decomposition(df)
    df = df.sort_values(["family", "q", "seed", "arm", "kind", "K", "update"]).reset_index(drop=True)
    df.to_csv(OUT / "candidates_all.csv", index=False)
    # coverage of the 2b bands
    t2b = df[df.family == "2b"]
    print("2b rows outside sweep:", int((~t2b.e1_inside_sweep.astype(bool)).sum()), "of", len(t2b))
    # ---- paired comparisons
    pc = []
    for fam, arms, Ks, mets in (("ext", ("ext",), (4, 8, 16), STAGE2_REPORT), ("p3", ("stochastic", "mean"), (4, 8, 12), STAGE1_REPORT),
                                ("2a", ("constant", "decay"), (4, 8), STAGE2_REPORT), ("2b", ("constant", "decay"), (4, 8, 12), STAGE1_REPORT)):
        for arm in arms:
            for K in Ks:
                pc.append(paired_vs(df, fam, (arm, K), (arm, 1), mets, "K_vs_K1"))
    for fam, Ks, mets in (("2a", (1, 4, 8), STAGE2_REPORT), ("2b", (1, 4, 8, 12), STAGE1_REPORT)):
        for K in Ks:
            pc.append(paired_vs(df, fam, ("decay", K), ("constant", K), mets, "decay_minus_constant"))
    pcs = pd.concat(pc, ignore_index=True)
    pcs.to_csv(OUT / "paired_summary.csv", index=False)
    # ---- 2b within-run / across-seed stability (exports u2100..u2200) and run records
    rec = run_records()
    st = []
    curves = df[(df.family == "2b") & (df.K == 1)]
    for (q, seed, arm), g in curves.groupby(["q", "seed", "arm"]):
        last5 = g[g["update"] >= 2100].sort_values("update")
        st.append({"q": q, "seed": seed, "arm": arm, "n_last5": len(last5),
                   "within_run_sd_e1_last5": float(last5.e1_cand.std(ddof=1)),
                   "within_run_range_e1_last5": float(last5.e1_cand.max() - last5.e1_cand.min())})
    st = pd.DataFrame(st)
    rec = rec.merge(st, on=["q", "seed", "arm"], how="left")
    rec.to_csv(OUT / "run_records.csv", index=False)
    rng = np.random.default_rng(BOOT_SEED)
    rp = []
    for fam in ("2a", "2b"):
        r_ = rec[rec.family == fam]
        mets = ["kl_final_epoch", "clip_frac", "phase_wall_sec"] + (["within_run_sd_e1_last5", "within_run_range_e1_last5"] if fam == "2b" else [])
        for q in QS:
            A = r_[(r_.q == q) & (r_.arm == "decay")].set_index("seed")
            B = r_[(r_.q == q) & (r_.arm == "constant")].set_index("seed")
            for m in mets:
                dv = np.array([A.loc[s, m] - B.loc[s, m] for s in SEEDS], float)
                rp.append({"comparison": "decay_minus_constant", "family": fam, "q": q, "metric": m,
                           **paired_summary(dv, LOWER_BETTER.get(m), rng)})
    pd.DataFrame(rp).to_csv(OUT / "paired_summary_run_records.csv", index=False)
    fin = df[(df.family == "2b") & (df.K == 1) & (df.kind == "tail")]
    stab = fin.groupby(["q", "arm"]).agg(across_seed_sd_final_e1=("e1_cand", lambda x: x.std(ddof=1)),
                                         across_seed_iqr_final_e1=("e1_cand", lambda x: x.quantile(.75) - x.quantile(.25)),
                                         median_sigma1_0=("sigma_effort_at_0_t1", "median")).reset_index()
    stab = stab.merge(st.groupby(["q", "arm"]).agg(median_within_run_sd_last5=("within_run_sd_e1_last5", "median"),
                                                   median_within_run_range_last5=("within_run_range_e1_last5", "median")).reset_index(),
                      on=["q", "arm"])
    stab.to_csv(OUT / "stability_2b.csv", index=False)
    rd = [{"family": fam, "q": q, "seed": s, **first_divergence(str(root / f"q{q}" / f"seed{s}" / a0), str(root / f"q{q}" / f"seed{s}" / a1))}
          for fam, root, a0, a1 in (("2a", P4A, "constant", "decay"), ("2b", P4B, "B2_mean_constant", "B2_mean_decay"))
          for q in QS for s in SEEDS]
    pd.DataFrame(rd).to_csv(OUT / "rng_divergence.csv", index=False)
    quantile_table(df).to_csv(OUT / "gate_distribution_tables.csv", index=False)
    plots(df)
    print("done:", len(df), "candidate evaluations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
