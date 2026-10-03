#!/usr/bin/env python3
"""Pilot 3 analysis (descriptive): frozen B2, continuation 'stochastic' vs 'mean'.

Inputs : results/v2_pilots/pilot3/q<q>/seed<seed>/{B2_frozen_s1norm, B2_frozen_s1norm_mean}/,
         results/v2_pilots/pilot3/analysis/decomposition_residual_band.csv (tools/v2/decomposition.py),
         results/v2_pilots/pilot2/... (reproducibility check of the stochastic arm).
Outputs: results/v2_pilots/pilot3/analysis/*, reports/v2/figures/pilot3/*.png
Bootstrap: percentile CI of the mean paired difference (mean - stochastic), 10000 resamples,
numpy seed 20261001. Curves: weight exports every 25 updates (dev tier, Beta mean, frozen
stage 2 = parent actor).

Usage: python tools/v2/pilot3_analysis.py [--workers 16]
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
from induced_band import actor_policy  # noqa: E402
from pilot2_analysis import _actor_fns  # noqa: E402
from rng_divergence import first_divergence  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG  # noqa: E402
from utils.v2_metrics import evaluate  # noqa: E402

P3 = ROOT / "results" / "v2_pilots" / "pilot3"
P2 = ROOT / "results" / "v2_pilots" / "pilot2"
OUT = P3 / "analysis"
FIG = ROOT / "reports" / "v2" / "figures" / "pilot3"
ARMS = ("B2_frozen_s1norm", "B2_frozen_s1norm_mean")
SHORT = {"B2_frozen_s1norm": "stochastic", "B2_frozen_s1norm_mean": "mean"}
N_BOOT, BOOT_SEED = 10000, 20261001
LAST5 = (900, 925, 950, 975, 1000)          # local Phase-B updates 500..600
METRICS = {
    "stage1_rel_err_signed": ("stage-1 rel. err (signed)", None),
    "stage1_rel_err_abs": ("stage-1 abs rel. err", "lower"),
    "learning_rel": ("learning term (e1 - e~1)/e1* (signed)", None),
    "learning_rel_abs": ("abs learning term / e1*", "lower"),
    "inherited_rel": ("inherited term (e~1 - e1*)/e1*", None),
    "sigma_effort_at_0_t1": ("sigma_1(0)", None),
    "Gmax_full_over_dw": ("Gmax_full/DW", "lower"),
    "EXP_root_over_dw": ("EXP_root/DW", "lower"),
    "dReach_over_dw": ("dReach/DW", "lower"),
    "Deltamax_all_over_dw": ("Delta_max_all/DW", "lower"),
    "dFull_over_dw": ("dFull/DW", "lower"),
    "within_run_sd_e1_last5": ("within-run SD of e1(0), last 5 exports", "lower"),
    "within_run_range_e1_last5": ("within-run range of e1(0), last 5 exports", "lower"),
    "kl_final_epoch": ("KL (last update)", None),
    "clip_frac": ("clip fraction (last update)", None),
    "phase_B_wall_sec": ("Phase B wall (s)", "lower"),
}


def run_dirs():
    for d in sorted(glob.glob(str(P3 / "q*" / "seed*" / "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "status.json")):
            yield d


def reproducibility() -> pd.DataFrame:
    rows = []
    for d in sorted(glob.glob(str(P3 / "q*" / "seed*" / "B2_frozen_s1norm"))):
        rel = os.path.relpath(d, P3)
        d2 = str(P2 / rel)
        h3, h2 = (json.load(open(os.path.join(x, "train_history.json"))) for x in (d, d2))
        strip = lambda vs: [{k: v for k, v in x.items() if k != "time_sec"} for x in vs]  # noqa: E731
        w3, w2 = (np.load(os.path.join(x, "checkpoint_weights.npz")) for x in (d, d2))
        e3 = sorted(glob.glob(os.path.join(d, "weights", "u*.npz")))
        e2 = sorted(glob.glob(os.path.join(d2, "weights", "u*.npz")))
        exp_eq = len(e3) == len(e2) and all(
            all(np.array_equal(np.load(a)[k], np.load(b)[k]) for k in np.load(a).files) for a, b in zip(e3, e2))
        s3 = torch.load(os.path.join(d, "state_end_B.pt"), weights_only=False)["agent"]
        s2 = torch.load(os.path.join(d2, "state_end_B.pt"), weights_only=False)["agent"]
        rows.append({"run": rel, "history_identical": h3["history"] == h2["history"],
                     "verifier_calls_identical": json.dumps(strip(h3["verifier_calls"]), sort_keys=True, default=str)
                     == json.dumps(strip(h2["verifier_calls"]), sort_keys=True, default=str),
                     "stability_identical": h3["stability"] == h2["stability"],
                     "final_weights_identical": all(np.array_equal(w3[k], w2[k]) for k in w3.files),
                     "weight_exports_identical": bool(exp_eq),
                     "end_state_actor_critic_opt_identical": all(torch.equal(s3[p][k], s2[p][k]) for p in ("actor", "critic", "opponent", "frozen") for k in s3[p]),
                     "n_updates": len(h3["history"])})
    return pd.DataFrame(rows)


def final_table(dec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in run_dirs():
        man = json.load(open(os.path.join(d, "manifest.json")))
        summ = json.load(open(os.path.join(d, "v2_run_summary.json")))
        dt = json.load(open(os.path.join(d, "drift_test.json")))
        last = pd.read_csv(os.path.join(d, "v2_checkpoints.csv")).iloc[-1]
        if int(last["update"]) != 1000:
            raise RuntimeError(f"{d}: last checkpoint {last['update']}")
        q, seed, arm = int(man["q"]), man["seed"], man["arm"]
        dw = dec[(dec.q == q) & (dec.seed == seed) & (dec.arm == arm) & (dec.source == "weights")]
        fin = dw[dw["update"] == 1000].iloc[0]
        last5 = dw[dw["update"].isin(LAST5)].e1_at_0
        r = {"q": q, "seed": seed, "arm": arm, "commit": man["git"]["short"], "dirty": man["git"]["dirty"],
             "continuation_action_mode": man["continuation_action_mode"], "reward_mode": man["reward_mode"],
             "phase_B_wall_sec": summ["phase_timing"]["B"]["wall_sec"],
             "would_have_fired_B": json.dumps(summ["would_have_fired"]["B"]),
             "snapshot_drift_max": max(dt["max_abs_diff_vs_freeze_time"].values()), "drift_test_pass": dt.get("pass"),
             "Gmax_full_t": int(last["Gmax_full_t"]), "Gmax_full_d": float(last["Gmax_full_d"]),
             "e1_at_0": float(last["e1_at_0"]), "learning_rel": float(fin.learning_rel),
             "learning_rel_lo": float(fin.learning_rel_lo), "learning_rel_hi": float(fin.learning_rel_hi),
             "learning_contains_0": bool(fin.learning_contains_0), "inherited_rel": float(fin.inherited_rel),
             "inherited_rel_lo": float(fin.inherited_rel_lo), "inherited_rel_hi": float(fin.inherited_rel_hi),
             "inherited_contains_0": bool(fin.inherited_contains_0),
             "within_run_sd_e1_last5": float(last5.std(ddof=1)), "within_run_range_e1_last5": float(last5.max() - last5.min()),
             "n_last5": int(last5.size)}
        r["learning_rel_abs"] = abs(r["learning_rel"])
        for m in METRICS:
            if m not in r:
                r[m] = float(last[m])
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["q", "seed", "arm"]).reset_index(drop=True)


def paired(df):
    rng = np.random.default_rng(BOOT_SEED)
    out, diffs = [], []
    for q, g in df.groupby("q"):
        s_ = g[g.arm == ARMS[0]].set_index("seed")
        m_ = g[g.arm == ARMS[1]].set_index("seed")
        seeds = sorted(set(s_.index) & set(m_.index))
        for m, (lab, fav) in METRICS.items():
            dv = np.array([m_.loc[x, m] - s_.loc[x, m] for x in seeds], float)
            for x, v in zip(seeds, dv):
                diffs.append({"q": q, "seed": x, "metric": m, "mean_minus_stochastic": v})
            bm = dv[rng.integers(0, len(dv), size=(N_BOOT, len(dv)))].mean(axis=1)
            out.append({"q": q, "metric": m, "label": lab, "n_pairs": len(dv), "mean": dv.mean(), "sd": dv.std(ddof=1),
                        "median": float(np.median(dv)), "min": dv.min(), "max": dv.max(),
                        "better": "mean better if diff < 0" if fav else "no preferred direction",
                        "n_mean_better": int((dv < 0).sum()) if fav else None, "n_neg": int((dv < 0).sum()),
                        "n_pos": int((dv > 0).sum()), "n_zero": int((dv == 0).sum()),
                        "boot_ci95_lo": float(np.percentile(bm, 2.5)), "boot_ci95_hi": float(np.percentile(bm, 97.5))})
    return pd.DataFrame(diffs), pd.DataFrame(out)


def _curve_job(args):
    d, f = args
    man = json.load(open(os.path.join(d, "manifest.json")))
    spec = spec_for(man["q"])
    live_m, live_b = _actor_fns(f, spec)
    par_m, par_b = _actor_fns(man["parent_checkpoint"], spec)
    s = evaluate(lambda t, x: par_m(t, x) if t == 2 else live_m(t, x), spec, DEV_CONFIG,
                 beta_fn=lambda t, x: par_b(t, x) if t == 2 else live_b(t, x)).scalars
    return {"q": int(man["q"]), "seed": man["seed"], "arm": man["arm"], "update": int(os.path.basename(f)[1:6]),
            **{k: s[k] for k in ("stage1_rel_err_signed", "sigma_effort_at_0_t1", "Gmax_full_over_dw", "EXP_root_over_dw",
                                 "e1_at_0")}}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rep = reproducibility()
    rep.to_csv(OUT / "reproducibility_vs_pilot2_B2.csv", index=False)
    dec = pd.read_csv(OUT / "decomposition_residual_band.csv")
    df = final_table(dec)
    df.to_csv(OUT / "final_table.csv", index=False)
    diffs, summ = paired(df)
    diffs.to_csv(OUT / "paired_differences.csv", index=False)
    summ.to_csv(OUT / "paired_summary.csv", index=False)
    stab = df.groupby(["q", "arm"]).agg(
        across_seed_sd_final_e1=("e1_at_0", lambda x: x.std(ddof=1)),
        across_seed_iqr_final_e1=("e1_at_0", lambda x: x.quantile(.75) - x.quantile(.25)),
        median_within_run_sd_last5=("within_run_sd_e1_last5", "median"),
        median_within_run_range_last5=("within_run_range_e1_last5", "median"),
        median_sigma1_0=("sigma_effort_at_0_t1", "median")).reset_index()
    stab.to_csv(OUT / "stability.csv", index=False)
    # inherited term identical within pairs: same parent snapshot tensors in both arms
    inh = []
    for q in (50, 60):
        for seed in range(10501, 10511):
            sa, sb = (torch.load(str(P3 / f"q{q}" / f"seed{seed}" / arm / "state_end_B.pt"), weights_only=False)["agent"]["frozen"]
                      for arm in ARMS)
            pa = dec[(dec.q == q) & (dec.seed == seed)].groupby("arm").inherited_rel.unique()
            inh.append({"q": q, "seed": seed, "snapshots_bit_identical": all(torch.equal(sa[k], sb[k]) for k in sa),
                        "inherited_rel_values": "; ".join(f"{k}:{list(v)}" for k, v in pa.items())})
    pd.DataFrame(inh).to_csv(OUT / "inherited_identity.csv", index=False)
    jobs = [(d, f) for d in run_dirs() for f in sorted(glob.glob(os.path.join(d, "weights", "u*.npz")))
            if int(os.path.basename(f)[1:6]) > 400]
    with Pool(a.workers) as pool:
        cv = pd.DataFrame(pool.map(_curve_job, jobs, chunksize=4))
    lt = dec[dec.source == "weights"][["q", "seed", "arm", "update", "learning_rel"]]
    cv = cv.merge(lt, on=["q", "seed", "arm", "update"], how="left")
    cv.to_csv(OUT / "curves_weights_every25.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {ARMS[0]: "#2ca02c", ARMS[1]: "#9467bd"}
    panels = [("stage1_rel_err_signed", "stage-1 rel. err (signed)"), ("learning_rel", "learning term (e1-e~1)/e1*"),
              ("sigma_effort_at_0_t1", "sigma_1(0)"), ("Gmax_full_over_dw", "Gmax_full / DW"), ("EXP_root_over_dw", "EXP_root / DW")]
    for q in (50, 60):
        fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
        for ax, (m, lab) in zip(axes, panels):
            for arm in ARMS:
                g = cv[(cv.q == q) & (cv.arm == arm)].groupby("update")[m]
                med = g.median()
                ax.plot(med.index, med.values, color=col[arm], lw=1.5, label=SHORT[arm])
                ax.fill_between(med.index, g.quantile(.25).values, g.quantile(.75).values, color=col[arm], alpha=.18, lw=0)
            ax.set_title(lab, fontsize=9)
            ax.set_xlabel("global update")
            if "signed" in m or m == "learning_rel":
                ax.axhline(0, color="0.5", lw=.8)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Pilot 3, q={q}: median and IQR over 10 seeds (weights every 25 updates, dev tier)", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"curves_q{q}.png", dpi=120)
        plt.close(fig)
    rd = [{"q": q, "seed": s, **first_divergence(str(P3 / f"q{q}" / f"seed{s}" / ARMS[0]), str(P3 / f"q{q}" / f"seed{s}" / ARMS[1]))}
          for q in (50, 60) for s in range(10501, 10511)]
    pd.DataFrame(rd).to_csv(OUT / "rng_divergence.csv", index=False)
    u = []
    for d in run_dirs():
        man = json.load(open(os.path.join(d, "manifest.json")))
        x = pd.read_csv(os.path.join(d, "v2_updates.csv"))
        x["q"], x["seed"], x["arm"] = int(man["q"]), man["seed"], man["arm"]
        u.append(x[["q", "seed", "arm", "update", "adv_all_mean", "adv_all_std", "adv_s1_mean", "adv_s1_std", "adv_used_std",
                    "kl_final_epoch", "clip_frac"]])
    pd.concat(u).groupby(["q", "arm"]).median(numeric_only=True).drop(columns=["seed", "update"]).to_csv(OUT / "adv_stats_median.csv")
    print("done; reproducibility all identical:", bool(rep.drop(columns=["run", "n_updates"]).all().all()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
