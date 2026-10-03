#!/usr/bin/env python3
"""Pilot 2 analysis (descriptive): joint (A) vs frozen_allnorm (B1) vs frozen_s1norm (B2).

Inputs : results/v2_pilots/pilot2/q<q>/seed<seed>/<arm>/ and the Pilot-1 parents.
Outputs: results/v2_pilots/pilot2/analysis/*.csv|json, reports/v2/figures/pilot2/*.png

Curves re-evaluate the weight exports every 25 updates (u425..u1000) on the development tier
with the run's candidate: live actor at t=1; at t=2 the live actor (A) or the frozen snapshot,
which is the parent's actor (B1, B2; the snapshot equality is checked by each run's
drift_test.json). Drift is measured against the parent's stage-2 mapping on the dev D_2 grid.
Bootstrap: percentile CI of the mean paired difference, 10000 resamples, numpy seed 20261001.

Usage: python tools/v2/pilot2_analysis.py [--workers 16]
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
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from common import spec_for  # noqa: E402
from agents.ppo_curriculum import CurriculumPPO, PPOConfig  # noqa: E402
from run.run_final_dp_br import dense_grid, make_policy_fns  # noqa: E402
from rng_divergence import first_divergence  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG  # noqa: E402
from utils.v2_metrics import evaluate, onoff_split  # noqa: E402

P2 = ROOT / "results" / "v2_pilots" / "pilot2"
P1 = ROOT / "results" / "v2_pilots" / "pilot1"
OUT = P2 / "analysis"
FIG = ROOT / "reports" / "v2" / "figures" / "pilot2"
ARMS = ("A_joint", "B1_frozen_allnorm", "B2_frozen_s1norm")
SHORT = {"A_joint": "A", "B1_frozen_allnorm": "B1", "B2_frozen_s1norm": "B2"}
COMPARE = (("B1_frozen_allnorm", "A_joint"), ("B2_frozen_s1norm", "A_joint"), ("B2_frozen_s1norm", "B1_frozen_allnorm"))
N_BOOT, BOOT_SEED = 10000, 20261001

METRICS = {   # column -> (label, "lower" if smaller is better else None)
    "stage2_drift_cand_on_max": ("stage-2 drift on-path max |d e2|", "lower"),
    "stage2_drift_cand_on_mean_cellmass_weighted": ("stage-2 drift on-path cell-mass mean", "lower"),
    "stage2_drift_cand_off_max": ("stage-2 drift off-path max", "lower"),
    "stage2_drift_cand_off_mean_unweighted": ("stage-2 drift off-path mean", "lower"),
    "stage2_peak_rel_err_signed": ("stage-2 peak rel. err (signed)", None),
    "stage2_peak_rel_err_abs": ("stage-2 abs peak rel. err", "lower"),
    "stage2_rmse_pos_over_g2_0": ("stage-2 RMSE / e2*(0)", "lower"),
    "stage2_tail_mean": ("stage-2 tail mean", "lower"),
    "stage2_tail_mean_over_g2_0": ("stage-2 tail mean / e2*(0)", "lower"),
    "stage2_tail_max": ("stage-2 tail max", "lower"),
    "stage2_tail_max_over_g2_0": ("stage-2 tail max / e2*(0)", "lower"),
    "stage2_sym_err_max": ("stage-2 symmetry err", "lower"),
    "stage1_rel_err_signed": ("stage-1 rel. err (signed)", None),
    "stage1_rel_err_abs": ("stage-1 abs rel. err", "lower"),
    "sigma_effort_at_0_t1": ("sigma_1(0)", None),
    "stage1_learning_err_rel": ("stage-1 learning err (e1 - e~1)/e1* (signed)", None),
    "stage1_learning_err_rel_abs": ("abs stage-1 learning err / e1*", "lower"),
    "stage1_inherited_err_rel": ("inherited err (e~1 - e1*)/e1* (signed)", None),
    "stage1_inherited_err_rel_abs": ("abs inherited err / e1*", "lower"),
    "Gmax_full_over_dw": ("Gmax_full/DW", "lower"),
    "EXP_root_over_dw": ("EXP_root/DW", "lower"),
    "dReach_over_dw": ("dReach/DW", "lower"),
    "Deltamax_all_over_dw": ("Delta_max_all/DW", "lower"),
    "dFull_over_dw": ("dFull/DW", "lower"),
    "eta_T_over_dw": ("eta_2", "lower"),
    "DeltaT_over_dw_on_max": ("Delta_2/DW on-path max", "lower"),
    "DeltaT_over_dw_on_mean_cellmass_weighted": ("Delta_2/DW on-path cell-mass mean", "lower"),
    "DeltaT_over_dw_off_max": ("Delta_2/DW off-path max", "lower"),
    "kl_final_epoch": ("KL (last update)", None),
    "clip_frac": ("clip fraction (last update)", None),
    "phase_B_wall_sec": ("Phase B wall (s)", "lower"),
}


def run_dirs():
    for d in sorted(glob.glob(str(P2 / "q*" / "seed*" / "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "status.json")):
            yield d


def final_table() -> pd.DataFrame:
    rows = []
    for d in run_dirs():
        man = json.load(open(os.path.join(d, "manifest.json")))
        summ = json.load(open(os.path.join(d, "v2_run_summary.json")))
        dt = json.load(open(os.path.join(d, "drift_test.json")))
        ck = pd.read_csv(os.path.join(d, "v2_checkpoints.csv"))
        last = ck.iloc[-1]
        if int(last["update"]) != 1000:
            raise RuntimeError(f"{d}: last checkpoint {last['update']} != 1000")
        r = {"q": int(man["q"]), "seed": man["seed"], "arm": man["arm"], "run_dir": os.path.relpath(d, ROOT),
             "commit": man["git"]["short"], "dirty": man["git"]["dirty"], "reward_mode": man["reward_mode"],
             "parent_sha256": man["parent_sha256"][:12], "n_verifier_checkpoints": len(ck),
             "phase_B_wall_sec": summ["phase_timing"]["B"]["wall_sec"],
             "would_have_fired_B": json.dumps(summ["would_have_fired"]["B"]),
             "Gmax_full_t": int(last["Gmax_full_t"]), "Gmax_full_d": float(last["Gmax_full_d"]),
             "snapshot_drift_mean": dt["max_abs_diff_vs_freeze_time"]["mean"],
             "snapshot_drift_alpha": dt["max_abs_diff_vs_freeze_time"]["alpha"],
             "snapshot_drift_beta": dt["max_abs_diff_vs_freeze_time"]["beta"],
             "drift_test_pass": dt.get("pass", "n/a (joint)"),
             "live_stage2_drift_maxabs": float(last["stage2_drift_live_maxabs"]),
             "induced_e1": float(last["induced_e1"]), "e1_at_0": float(last["e1_at_0"]),
             "induced_residual": float(last["induced_residual"])}
        for m in METRICS:
            if m == "phase_B_wall_sec":
                continue
            if m.endswith("_rel_abs") and m not in last:
                r[m] = abs(float(last[m[:-4]]))
            else:
                r[m] = float(last[m])
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["q", "seed", "arm"]).reset_index(drop=True)


def paired(df: pd.DataFrame):
    rng = np.random.default_rng(BOOT_SEED)
    diffs, summ = [], []
    for q, g in df.groupby("q"):
        by = {a: g[g.arm == a].set_index("seed") for a in ARMS}
        for x, y in COMPARE:
            seeds = sorted(set(by[x].index) & set(by[y].index))
            for m, (label, fav) in METRICS.items():
                dv = np.array([by[x].loc[s, m] - by[y].loc[s, m] for s in seeds], dtype=float)
                for s, v in zip(seeds, dv):
                    diffs.append({"q": q, "comparison": f"{SHORT[x]}-{SHORT[y]}", "seed": s, "metric": m, "diff": v})
                bm = dv[rng.integers(0, len(dv), size=(N_BOOT, len(dv)))].mean(axis=1)
                summ.append({"q": q, "comparison": f"{SHORT[x]}-{SHORT[y]}", "metric": m, "label": label,
                             "n_pairs": len(dv), "mean": dv.mean(), "sd": dv.std(ddof=1), "median": float(np.median(dv)),
                             "min": dv.min(), "max": dv.max(),
                             "better": f"{SHORT[x]} better if diff < 0" if fav == "lower" else "no preferred direction",
                             "n_better_first": int((dv < 0).sum()) if fav == "lower" else None,
                             "n_neg": int((dv < 0).sum()), "n_pos": int((dv > 0).sum()), "n_zero": int((dv == 0).sum()),
                             "boot_ci95_lo": float(np.percentile(bm, 2.5)), "boot_ci95_hi": float(np.percentile(bm, 97.5))})
    return pd.DataFrame(diffs), pd.DataFrame(summ)


def _actor_fns(npz_or_state, spec):
    agent = CurriculumPPO(PPOConfig(), torch.Generator().manual_seed(0), np.random.default_rng(0))
    if npz_or_state.endswith(".npz"):
        w = np.load(npz_or_state)
        sd = {k[len("actor."):]: torch.as_tensor(w[k]) for k in w.files if k.startswith("actor.")}
    else:
        sd = torch.load(npz_or_state, weights_only=False)["agent"]["actor"]
    agent.actor.load_state_dict(sd)
    return make_policy_fns(agent, spec)


def _curve_job(args):
    d, f = args
    man = json.load(open(os.path.join(d, "manifest.json")))
    spec = spec_for(man["q"])
    live_m, live_b = _actor_fns(f, spec)
    par_m, par_b = _actor_fns(man["parent_checkpoint"], spec)
    frozen = man["stage2_update_mode"] == "frozen"
    mf = (lambda t, x: par_m(t, x) if t == 2 else live_m(t, x)) if frozen else live_m
    bf = (lambda t, x: par_b(t, x) if t == 2 else live_b(t, x)) if frozen else live_b
    ev = evaluate(mf, spec, DEV_CONFIG, beta_fn=bf)
    s = ev.scalars
    g = dense_grid(spec.domain_half(2), DEV_CONFIG.state_step)
    dd = np.abs(np.asarray(mf(2, g), float) - np.asarray(par_m(2, g), float))
    sp = onoff_split(dd, ev.arrays["v_t2_onpath"], ev.arrays["v_t2_cell_mass"], g)
    keep = ("stage1_rel_err_signed", "Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw",
            "stage1_learning_err_rel", "stage1_inherited_err_rel", "induced_residual", "Gmax_full_t")
    return {"q": int(man["q"]), "seed": man["seed"], "arm": man["arm"], "update": int(os.path.basename(f)[1:6]),
            **{k: s[k] for k in keep}, "drift_on_max": sp["on_max"], "drift_off_max": sp["off_max"],
            "drift_on_wmean": sp["on_mean_cellmass_weighted"]}


def curves(workers: int) -> pd.DataFrame:
    jobs = [(d, f) for d in run_dirs() for f in sorted(glob.glob(os.path.join(d, "weights", "u*.npz")))
            if int(os.path.basename(f)[1:6]) > 400]
    with Pool(workers) as pool:
        rows = pool.map(_curve_job, jobs, chunksize=4)
    return pd.DataFrame(rows)


def plots(cv: pd.DataFrame, upd: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    col = {"A_joint": "#1f77b4", "B1_frozen_allnorm": "#d62728", "B2_frozen_s1norm": "#2ca02c"}
    panels = [("drift_on_max", "stage-2 drift, on-path max"), ("drift_off_max", "stage-2 drift, off-path max"),
              ("stage1_rel_err_signed", "stage-1 rel. err (signed)"), ("Gmax_full_over_dw", "Gmax_full / DW"),
              ("EXP_root_over_dw", "EXP_root / DW"), ("dReach_over_dw", "dReach / DW"),
              ("stage1_learning_err_rel", "learning err (e1-e~1)/e1*"), ("stage1_inherited_err_rel", "inherited err (e~1-e1*)/e1*")]
    for q in sorted(cv.q.unique()):
        fig, axes = plt.subplots(2, 4, figsize=(17, 7))
        for ax, (m, lab) in zip(axes.ravel(), panels):
            for arm in ARMS:
                g = cv[(cv.q == q) & (cv.arm == arm)].groupby("update")[m]
                med = g.median()
                ax.plot(med.index, med.values, color=col[arm], lw=1.5, label=SHORT[arm])
                ax.fill_between(med.index, g.quantile(.25).values, g.quantile(.75).values, color=col[arm], alpha=.18, lw=0)
            ax.set_title(lab, fontsize=9)
            ax.set_xlabel("global update")
            if "signed" in m or "err_rel" in m:
                ax.axhline(0, color="0.5", lw=.8)
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Pilot 2, q={q}: median and IQR over 10 seeds (weights every 25 updates, dev tier)", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"curves_q{q}.png", dpi=120)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, q in zip(axes, sorted(upd.q.unique())):
        for arm in ARMS:
            g = upd[(upd.q == q) & (upd.arm == arm)].groupby("update")["sd_ratio_s1_over_all"]
            med = g.median()
            ax.plot(med.index, med.values, color=col[arm], lw=1.2, label=SHORT[arm])
            ax.fill_between(med.index, g.quantile(.25).values, g.quantile(.75).values, color=col[arm], alpha=.18, lw=0)
        ax.set_title(f"q={q}: SD(adv, stage-1 rows) / SD(adv, all rows)", fontsize=9)
        ax.set_xlabel("global update")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "adv_sd_ratio.png", dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = final_table()
    df.to_csv(OUT / "final_table.csv", index=False)
    diffs, summ = paired(df)
    diffs.to_csv(OUT / "paired_differences.csv", index=False)
    summ.to_csv(OUT / "paired_summary.csv", index=False)
    cks = []
    ups = []
    for d in run_dirs():
        man = json.load(open(os.path.join(d, "manifest.json")))
        c = pd.read_csv(os.path.join(d, "v2_checkpoints.csv"))
        c["q"], c["arm"], c["seed"] = int(man["q"]), man["arm"], man["seed"]
        cks.append(c)
        u = pd.read_csv(os.path.join(d, "v2_updates.csv"))
        u["q"], u["arm"], u["seed"] = int(man["q"]), man["arm"], man["seed"]
        u["sd_ratio_s1_over_all"] = u.adv_s1_std / u.adv_all_std
        ups.append(u[["q", "arm", "seed", "update", "adv_all_mean", "adv_all_std", "adv_s1_mean", "adv_s1_std",
                      "adv_used_std", "sd_ratio_s1_over_all", "kl_final_epoch", "clip_frac", "update_wall_sec"]])
    ck = pd.concat(cks, ignore_index=True)
    upd = pd.concat(ups, ignore_index=True)
    upd.to_csv(OUT / "per_update_adv_stats.csv", index=False)
    upd.groupby(["q", "arm"]).sd_ratio_s1_over_all.describe(percentiles=[.25, .5, .75]).to_csv(OUT / "adv_sd_ratio_summary.csv")
    ck["gmax_loc"] = np.where(ck.Gmax_full_t == 1, "stage1",
                              np.where(ck.Gmax_full_d.abs() < 2 * ck.q, "stage2_onpath", "stage2_offpath"))
    loc_all = ck.groupby(["q", "arm", "gmax_loc"]).size().unstack(fill_value=0)
    loc_all.to_csv(OUT / "gmax_location_all_checkpoints.csv")
    fl = df.assign(gmax_loc=np.where(df.Gmax_full_t == 1, "stage1",
                                     np.where(df.Gmax_full_d.abs() < 2 * df.q, "stage2_onpath", "stage2_offpath")))
    fl.groupby(["q", "arm", "gmax_loc"]).size().unstack(fill_value=0).to_csv(OUT / "gmax_location_final.csv")
    cv = curves(a.workers)
    cv.to_csv(OUT / "curves_weights_every25.csv", index=False)
    chk = cv[cv["update"] == 1000].merge(df[["q", "seed", "arm", "stage1_rel_err_signed", "Gmax_full_over_dw"]],
                                         on=["q", "seed", "arm"], suffixes=("_w", "_ck"))
    cons = {"max_abs_diff_stage1_err": float((chk.stage1_rel_err_signed_w - chk.stage1_rel_err_signed_ck).abs().max()),
            "max_abs_diff_gmax": float((chk.Gmax_full_over_dw_w - chk.Gmax_full_over_dw_ck).abs().max()), "n": len(chk)}
    plots(cv, upd)
    rng = []
    for q in sorted(df.q.unique()):
        for seed in sorted(df.seed.unique()):
            for x, y in COMPARE:
                rng.append({"q": q, "seed": seed, "pair": f"{SHORT[x]}-{SHORT[y]}",
                            **first_divergence(str(P2 / f"q{q}" / f"seed{seed}" / x), str(P2 / f"q{q}" / f"seed{seed}" / y))})
    pd.DataFrame(rng).to_csv(OUT / "rng_divergence.csv", index=False)
    json.dump({"weights_u1000_vs_final_checkpoint": cons, "n_checkpoint_rows": len(ck),
               "stage1_drift_all_zero": bool((ck.stage1_drift == 0).all()),
               "frozen_cand_drift_all_zero": bool((ck[ck.arm != "A_joint"].stage2_drift_cand_maxabs == 0).all()),
               "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED}}, open(OUT / "analysis_meta.json", "w"), indent=1)
    print("done", cons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
