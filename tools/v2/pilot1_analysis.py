#!/usr/bin/env python3
"""Pilot 1 analysis (descriptive): final table, paired differences, curves, scatter, RNG.

Inputs : results/v2_pilots/pilot1/q<q>/seed<seed>/<arm>/ (v2_checkpoints.csv, v2_updates.csv,
         v2_run_summary.json, weights/u*.npz, manifest.json)
Outputs: results/v2_pilots/pilot1/analysis/*.csv|json and reports/v2/figures/pilot1/*.png

Learning curves re-evaluate the actor weights exported every 25 updates (weights/u*.npz) with
utils.v2_metrics.evaluate on the development tier (pure evaluation; the training-time verifier
checkpoints fall at run-specific updates because stability-triggered calls move them).
Bootstrap: percentile CI of the mean paired difference, 10000 resamples, numpy seed 20261001.

Usage: python tools/v2/pilot1_analysis.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from common import spec_for  # noqa: E402
from agents.ppo_curriculum import CurriculumPPO, PPOConfig  # noqa: E402
from run.run_final_dp_br import make_policy_fns  # noqa: E402
from rng_divergence import first_divergence  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG  # noqa: E402
from utils.v2_metrics import evaluate  # noqa: E402

P1 = ROOT / "results" / "v2_pilots" / "pilot1"
OUT = P1 / "analysis"
FIG = ROOT / "reports" / "v2" / "figures" / "pilot1"
ARMS = ("sampled", "expected")
N_BOOT, BOOT_SEED = 10000, 20261001

# metric column -> (label, favour rule: "lower" = smaller is better for expected-sampled < 0)
METRICS = {
    "stage2_peak_rel_err_signed": ("peak rel. err. at d=0 (signed)", None),
    "stage2_peak_rel_err_abs": ("|peak rel. err.| at d=0", "lower"),
    "stage2_rmse_pos": ("RMSE |d|<2q (effort)", "lower"),
    "stage2_rmse_pos_over_g2_0": ("RMSE / e2*(0)", "lower"),
    "stage2_tail_mean": ("tail mean effort", "lower"),
    "stage2_tail_mean_over_g2_0": ("tail mean / e2*(0)", "lower"),
    "stage2_tail_max": ("tail max effort", "lower"),
    "stage2_tail_max_over_g2_0": ("tail max / e2*(0)", "lower"),
    "stage2_sym_err_max": ("symmetry err. max (effort)", "lower"),
    "stage2_sym_err_max_over_g2_0": ("symmetry err. / e2*(0)", "lower"),
    "eta_T_over_dw": ("eta_2 = max Delta_2 / DW", "lower"),
    "DeltaT_over_dw_on_max": ("Delta_2/DW on-path max", "lower"),
    "DeltaT_over_dw_on_mean_cellmass_weighted": ("Delta_2/DW on-path cell-mass mean", "lower"),
    "DeltaT_over_dw_off_max": ("Delta_2/DW off-path max", "lower"),
    "sigma_effort_at_0_t2": ("sigma_2(0) (effort)", None),
    "sigma2_effort_mean_pos": ("mean sigma_2 over |d|<2q", None),
    "kl_final_epoch": ("KL (final epoch, last update)", None),
    "clip_frac": ("clip fraction (last update)", None),
    "phase_A_wall_sec": ("Phase A wall-clock (s)", "lower"),
    "EXP_root_over_dw": ("EXP_root/DW [stage1_untrained]", "lower"),
    "dReach_over_dw": ("dReach/DW [stage1_untrained]", "lower"),
    "Gmax_full_over_dw": ("Gmax_full/DW [stage1_untrained]", "lower"),
}


def run_dirs():
    for d in sorted(glob.glob(str(P1 / "q*" / "seed*" / "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "status.json")):
            yield d


def final_table() -> pd.DataFrame:
    rows = []
    for d in run_dirs():
        man = json.load(open(os.path.join(d, "manifest.json")))
        summ = json.load(open(os.path.join(d, "v2_run_summary.json")))
        ck = pd.read_csv(os.path.join(d, "v2_checkpoints.csv"))
        last = ck.iloc[-1]
        if int(last["update"]) != 400:
            raise RuntimeError(f"{d}: last verifier checkpoint at {last['update']}, not 400")
        r = {"q": int(man["q"]), "seed": man["seed"], "arm": man["arm"], "run_dir": os.path.relpath(d, ROOT),
             "commit": man["git"]["short"], "dirty": man["git"]["dirty"], "update": int(last["update"]),
             "stage1_status": last["stage1_status"], "stage1_drift": last["stage1_drift"],
             "phase_A_wall_sec": summ["phase_timing"]["A"]["wall_sec"],
             "would_have_fired_A": json.dumps(summ["would_have_fired"]["A"]),
             "n_verifier_checkpoints": len(ck)}
        for m in METRICS:
            if m != "phase_A_wall_sec":
                r[m] = float(last[m])
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["q", "seed", "arm"]).reset_index(drop=True)


def paired(df: pd.DataFrame):
    rng = np.random.default_rng(BOOT_SEED)
    diffs, summ = [], []
    for q, g in df.groupby("q"):
        e = g[g.arm == "expected"].set_index("seed")
        s = g[g.arm == "sampled"].set_index("seed")
        seeds = sorted(set(e.index) & set(s.index))
        for m, (label, fav) in METRICS.items():
            dvec = np.array([e.loc[sd, m] - s.loc[sd, m] for sd in seeds], dtype=float)
            for sd, v in zip(seeds, dvec):
                diffs.append({"q": q, "seed": sd, "metric": m, "expected_minus_sampled": v})
            idx = rng.integers(0, len(dvec), size=(N_BOOT, len(dvec)))
            bm = dvec[idx].mean(axis=1)
            summ.append({"q": q, "metric": m, "label": label, "n_pairs": len(dvec),
                         "mean": dvec.mean(), "sd": dvec.std(ddof=1), "median": float(np.median(dvec)),
                         "min": dvec.min(), "max": dvec.max(),
                         "favour_rule": "expected favoured if diff < 0" if fav == "lower" else "no preferred direction",
                         "n_favour_expected": int((dvec < 0).sum()) if fav == "lower" else None,
                         "n_diff_negative": int((dvec < 0).sum()), "n_diff_positive": int((dvec > 0).sum()),
                         "n_diff_zero": int((dvec == 0).sum()),
                         "boot_ci95_lo": float(np.percentile(bm, 2.5)), "boot_ci95_hi": float(np.percentile(bm, 97.5))})
    return pd.DataFrame(diffs), pd.DataFrame(summ)


def load_actor_policy(npz: str, spec):
    w = np.load(npz)
    agent = CurriculumPPO(PPOConfig(), torch.Generator().manual_seed(0), np.random.default_rng(0))
    sd = {k[len("actor."):]: torch.as_tensor(w[k]) for k in w.files if k.startswith("actor.")}
    agent.actor.load_state_dict(sd)
    return make_policy_fns(agent, spec)


def curves() -> pd.DataFrame:
    rows = []
    keys = ["stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_rmse_pos_over_g2_0",
            "stage2_tail_mean", "eta_T_over_dw", "sigma_effort_at_0_t2"]
    for d in run_dirs():
        man = json.load(open(os.path.join(d, "manifest.json")))
        spec = spec_for(man["q"])
        for f in sorted(glob.glob(os.path.join(d, "weights", "u*.npz"))):
            u = int(os.path.basename(f)[1:6])
            mf, bf = load_actor_policy(f, spec)
            sc = evaluate(mf, spec, DEV_CONFIG, beta_fn=bf).scalars
            rows.append({"q": int(man["q"]), "seed": man["seed"], "arm": man["arm"], "update": u,
                         **{k: sc[k] for k in keys}})
    return pd.DataFrame(rows)


def plots(cv: pd.DataFrame, df_ck: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    os.makedirs(FIG, exist_ok=True)
    colors = {"sampled": "#1f77b4", "expected": "#d62728"}
    panels = [("stage2_peak_rel_err_signed", "peak rel. err. at d=0 (signed)"),
              ("stage2_rmse_pos_over_g2_0", "RMSE(|d|<2q) / e2*(0)"),
              ("stage2_tail_mean", "tail mean effort"),
              ("eta_T_over_dw", "eta_2 = max Delta_2 / DW"),
              ("sigma_effort_at_0_t2", "sigma_2(0) (effort units)")]
    for q in sorted(cv.q.unique()):
        fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.3))
        for ax, (m, lab) in zip(axes, panels):
            for arm in ARMS:
                g = cv[(cv.q == q) & (cv.arm == arm)].groupby("update")[m]
                med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
                ax.plot(med.index, med.values, color=colors[arm], lw=1.6, label=arm)
                ax.fill_between(med.index, lo.values, hi.values, color=colors[arm], alpha=0.2, lw=0)
            ax.set_title(lab, fontsize=9)
            ax.set_xlabel("update")
            if m == "stage2_peak_rel_err_signed":
                ax.axhline(0, color="0.5", lw=0.8)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Pilot 1, q={q}: median and IQR across 10 seeds (weights every 25 updates, dev tier)", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"curves_q{q}.png", dpi=130)
        plt.close(fig)
    sp = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, q in zip(axes, sorted(df_ck.q.unique())):
        g = df_ck[df_ck.q == q]
        for arm in ARMS:
            h = g[g.arm == arm]
            ax.scatter(h.sigma_effort_at_0_t2 / q, h.stage2_peak_rel_err_signed, s=10, alpha=0.6,
                       color=colors[arm], label=arm)
            r = spearmanr(h.sigma_effort_at_0_t2 / q, h.stage2_peak_rel_err_signed)
            sp.append({"q": q, "arm": arm, "n_points": len(h), "spearman_rho": r.statistic, "p_value": r.pvalue})
        r = spearmanr(g.sigma_effort_at_0_t2 / q, g.stage2_peak_rel_err_signed)
        sp.append({"q": q, "arm": "pooled", "n_points": len(g), "spearman_rho": r.statistic, "p_value": r.pvalue})
        ax.set_xlabel("sigma_2(0) / q")
        ax.set_ylabel("peak rel. err. at d=0 (signed)")
        ax.set_title(f"q={q}: all verifier checkpoints, all runs", fontsize=10)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "scatter_peakerr_vs_sigma.png", dpi=130)
    plt.close(fig)
    return pd.DataFrame(sp)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    df = final_table()
    df.to_csv(OUT / "final_table.csv", index=False)
    diffs, summ = paired(df)
    diffs.to_csv(OUT / "paired_differences.csv", index=False)
    summ.to_csv(OUT / "paired_summary.csv", index=False)
    cks = []
    for d in run_dirs():
        man = json.load(open(os.path.join(d, "manifest.json")))
        c = pd.read_csv(os.path.join(d, "v2_checkpoints.csv"))
        c["q"], c["arm"], c["seed"] = int(man["q"]), man["arm"], man["seed"]
        cks.append(c)
    df_ck = pd.concat(cks, ignore_index=True)
    cv = curves()
    cv.to_csv(OUT / "curves_weights_every25.csv", index=False)
    # consistency: weight export at u400 must reproduce the run's own final checkpoint row
    chk = cv[cv["update"] == 400].merge(df[["q", "seed", "arm", "stage2_peak_rel_err_signed", "eta_T_over_dw"]],
                                     on=["q", "seed", "arm"], suffixes=("_w", "_ck"))
    cons = {"max_abs_diff_peak": float((chk.stage2_peak_rel_err_signed_w - chk.stage2_peak_rel_err_signed_ck).abs().max()),
            "max_abs_diff_eta": float((chk.eta_T_over_dw_w - chk.eta_T_over_dw_ck).abs().max()), "n": len(chk)}
    sp = plots(cv, df_ck)
    sp.to_csv(OUT / "spearman_peakerr_vs_sigma.csv", index=False)
    rng = []
    for q in (50, 60):
        for seed in range(10501, 10511):
            a, b = (str(P1 / f"q{q}" / f"seed{seed}" / arm) for arm in ARMS)
            rng.append({"q": q, "seed": seed, **first_divergence(a, b)})
    pd.DataFrame(rng).to_csv(OUT / "rng_divergence.csv", index=False)
    json.dump({"weights_u400_vs_final_checkpoint": cons, "n_checkpoint_rows": len(df_ck),
               "stage1_drift_all_zero": bool((df_ck.stage1_drift == 0).all()),
               "cellmass_total_range": [float(df_ck.cellmass_captured_total.min()), float(df_ck.cellmass_captured_total.max())],
               "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "type": "percentile, mean of paired differences"}},
              open(OUT / "analysis_meta.json", "w"), indent=1)
    print("done", cons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
