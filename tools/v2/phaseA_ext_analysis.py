#!/usr/bin/env python3
"""Phase A extension analysis (descriptive): stage-2-only training continued from u400 to u1600.

Inputs : results/v2_pilots/phaseA_ext/q<q>/seed<seed>/expected_ext/ (weights every 25 updates,
         v2_checkpoints.csv, v2_updates.csv, v2_run_summary.json) and the Pilot-1 parent export
         results/v2_pilots/pilot1/q<q>/seed<seed>/expected/weights/u00400.npz.
Outputs: results/v2_pilots/phaseA_ext/analysis/*, reports/v2/figures/phaseA_ext/*.png
Metrics are evaluate() on the development tier (Beta mean). The smoothed-game prediction uses
tools/v2/pilot1_smoothed_game.py's quadrature (400 equal-probability nodes per Beta) on the
development D_2 grid at u400 / 800 / 1200 / 1600.

Usage: python tools/v2/phaseA_ext_analysis.py [--workers 16]
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
from pilot1_smoothed_game import centred_nodes  # noqa: E402
from pilot2_analysis import _actor_fns  # noqa: E402
from run.run_final_dp_br import dense_grid  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG  # noqa: E402
from utils.theory_multistage import f_xi, g2_two_stage  # noqa: E402
from utils.v2_metrics import evaluate  # noqa: E402

PX = ROOT / "results" / "v2_pilots" / "phaseA_ext"
P1 = ROOT / "results" / "v2_pilots" / "pilot1"
OUT = PX / "analysis"
FIG = ROOT / "reports" / "v2" / "figures" / "phaseA_ext"
KEEP = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean",
        "stage2_tail_mean_over_g2_0", "stage2_tail_max", "stage2_tail_max_over_g2_0", "stage2_sym_err_max",
        "eta_T_over_dw", "DeltaT_over_dw_on_max", "DeltaT_over_dw_on_mean_cellmass_weighted", "DeltaT_over_dw_off_max",
        "sigma_effort_at_0_t2", "sigma2_effort_mean_pos")
MARKS = (400, 800, 1200, 1600)
N_NODES = 400


def _job(args):
    q, seed, u, f = args
    spec = spec_for(q)
    mf, bf = _actor_fns(f, spec)
    s = evaluate(mf, spec, DEV_CONFIG, beta_fn=bf).scalars
    row = {"q": q, "seed": seed, "update": u, **{k: s[k] for k in KEEP}}
    if u in MARKS:
        G = dense_grid(spec.domain_half(2), DEV_CONFIG.state_step)
        al, be = bf(2, G)
        eh = np.asarray(mf(2, G), float)
        z0 = int(np.nonzero(G == 0.0)[0][0])
        ai = centred_nodes(al[z0], be[z0], N_NODES, spec.e_range)
        pred0 = spec.dw / (2 * spec.k) * f_xi(ai[:, None] - ai[None, :], q).mean()
        g20 = float(g2_two_stage(np.array([0.0]), q, spec.w_h, spec.w_l, spec.k, spec.e_max)[0])
        row.update(e_pred_0=float(pred0), e_learned_0=float(eh[z0]),
                   share_peak_gap_explained=float((g20 - pred0) / (g20 - eh[z0])))
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    jobs, summ = [], []
    for d in sorted(glob.glob(str(PX / "q*" / "seed*" / "expected_ext"))):
        man = json.load(open(os.path.join(d, "manifest.json")))
        q, seed = int(man["q"]), man["seed"]
        jobs.append((q, seed, 400, str(P1 / f"q{q}" / f"seed{seed}" / "expected" / "weights" / "u00400.npz")))
        for f in sorted(glob.glob(os.path.join(d, "weights", "u*.npz"))):
            jobs.append((q, seed, int(os.path.basename(f)[1:6]), f))
        rs = json.load(open(os.path.join(d, "v2_run_summary.json")))
        up = pd.read_csv(os.path.join(d, "v2_updates.csv"))
        summ.append({"q": q, "seed": seed, "commit": man["git"]["short"], "dirty": man["git"]["dirty"],
                     "would_have_fired_A": json.dumps(rs["would_have_fired"]["A"]),
                     "phase_A_ext_wall_sec": rs["phase_timing"]["A"]["wall_sec"], "updates": rs["phase_timing"]["A"]["updates"],
                     "kl_median": float(up.kl_final_epoch.median()), "clip_median": float(up.clip_frac.median()),
                     "full_states": " ".join(sorted(os.path.basename(x) for x in glob.glob(os.path.join(d, "state_*.pt"))))})
    pd.DataFrame(summ).to_csv(OUT / "runs.csv", index=False)
    with Pool(a.workers) as pool:
        cv = pd.DataFrame(pool.map(_job, jobs, chunksize=4))
    cv.to_csv(OUT / "curves_weights_every25.csv", index=False)
    marks = cv[cv["update"].isin(MARKS)]
    cols = list(KEEP) + ["e_pred_0", "e_learned_0", "share_peak_gap_explained"]
    tab = []
    for (q, u), g in marks.groupby(["q", "update"]):
        for c in cols:
            x = g[c]
            tab.append({"q": q, "update": u, "metric": c, "median": x.median(), "q25": x.quantile(.25),
                        "q75": x.quantile(.75), "min": x.min(), "max": x.max(), "n": int(x.size)})
    pd.DataFrame(tab).to_csv(OUT / "table_400_800_1200_1600.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("stage2_peak_rel_err_signed", "peak rel. err (signed)"), ("stage2_rmse_pos_over_g2_0", "RMSE / e2*(0)"),
              ("stage2_tail_mean", "tail mean"), ("stage2_tail_max", "tail max"), ("stage2_sym_err_max", "symmetry err"),
              ("eta_T_over_dw", "eta_2"), ("DeltaT_over_dw_off_max", "Delta_2 off-path max / DW"),
              ("sigma_effort_at_0_t2", "sigma_2(0)")]
    for q in (50, 60):
        fig, axes = plt.subplots(2, 4, figsize=(17, 7))
        for ax, (m, lab) in zip(axes.ravel(), panels):
            g = cv[cv.q == q].groupby("update")[m]
            med = g.median()
            ax.plot(med.index, med.values, color="#d62728", lw=1.5)
            ax.fill_between(med.index, g.quantile(.25).values, g.quantile(.75).values, color="#d62728", alpha=.2, lw=0)
            ax.set_title(lab, fontsize=9)
            ax.set_xlabel("global update (phase A)")
            if "signed" in m:
                ax.axhline(0, color="0.5", lw=.8)
        fig.suptitle(f"Phase A extension, q={q}: median and IQR over 10 seeds (weights every 25 updates, dev tier)", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"curves_q{q}.png", dpi=120)
        plt.close(fig)
    print("done", len(cv), "evaluations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
