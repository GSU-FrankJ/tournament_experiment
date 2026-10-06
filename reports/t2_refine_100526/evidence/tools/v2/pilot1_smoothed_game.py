#!/usr/bin/env python3
"""Pilot-1 side analysis: smoothed-game (stochastic-policy FOC) prediction of the stage-2 policy.

e_pred(d) = (DW / 2k) E[ f_xi(d + a_i - a_j) ], a_i = effort(d) - mean(d) under the learned Beta at
d, a_j = effort(-d) - mean(-d) under the learned Beta at -d (each centred on its own mean).
alpha(d), beta(d) are the saved final-checkpoint arrays on the development D_2 grid
(final_development.npz: v_t2_alpha, v_t2_beta, v_t2_e_hat, v_t2_d_grid). The grid is symmetric
about 0 and contains -d for every d, so no interpolation is needed (checked).
Quadrature: N equal-probability nodes per Beta (midpoint quantiles (i + 1/2)/N, weight 1/N),
tensor product over the two Betas.

Usage: python tools/v2/pilot1_smoothed_game.py [--n-nodes 400]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from common import spec_for  # noqa: E402
from utils.theory_multistage import f_xi, g2_two_stage  # noqa: E402

P1 = ROOT / "results" / "v2_pilots" / "pilot1"
OUT = P1 / "analysis" / "smoothed_game"
FIG = ROOT / "reports" / "v2" / "figures" / "pilot1"


def centred_nodes(a: float, b: float, n: int, e_range: float) -> np.ndarray:
    """Effort-unit, mean-centred equal-probability nodes of Beta(a, b)."""
    u = (np.arange(n) + 0.5) / n
    x = beta_dist.ppf(u, a, b)
    return e_range * (x - a / (a + b))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-nodes", type=int, default=400)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows, curves = [], []
    for d in sorted(glob.glob(str(P1 / "q*" / "seed*" / "*"))):
        if not os.path.exists(os.path.join(d, "final_development.npz")):
            continue
        man = json.load(open(os.path.join(d, "manifest.json")))
        q = float(man["q"])
        spec = spec_for(q)
        z = np.load(os.path.join(d, "final_development.npz"))
        G, al, be, eh = z["v_t2_d_grid"], z["v_t2_alpha"], z["v_t2_beta"], z["v_t2_e_hat"]
        if not np.array_equal(G, -G[::-1]):
            raise RuntimeError("grid not symmetric")
        pred = np.empty(G.size)
        for i in range(G.size):
            j = G.size - 1 - i                   # index of -d
            ai = centred_nodes(al[i], be[i], a.n_nodes, spec.e_range)
            aj = centred_nodes(al[j], be[j], a.n_nodes, spec.e_range)
            pred[i] = spec.dw / (2 * spec.k) * f_xi(G[i] + ai[:, None] - aj[None, :], q).mean()
        g2 = g2_two_stage(G, q, spec.w_h, spec.w_l, spec.k, spec.e_max)
        pos = np.abs(G) < 2 * q
        z0 = int(np.nonzero(G == 0.0)[0][0])
        rows.append({"q": int(q), "seed": man["seed"], "arm": man["arm"],
                     "rmse_learned_minus_pred_pos": float(np.sqrt(np.mean((eh[pos] - pred[pos]) ** 2))),
                     "rmse_learned_minus_estar_pos": float(np.sqrt(np.mean((eh[pos] - g2[pos]) ** 2))),
                     "e_star_0": float(g2[z0]), "e_pred_0": float(pred[z0]), "e_learned_0": float(eh[z0]),
                     "share_peak_gap_explained": float((g2[z0] - pred[z0]) / (g2[z0] - eh[z0])),
                     "n_nodes_per_beta": a.n_nodes})
        for i in range(G.size):
            curves.append({"q": int(q), "seed": man["seed"], "arm": man["arm"], "d": G[i], "e_star": g2[i],
                           "e_pred": pred[i], "e_learned": eh[i]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_run.csv", index=False)
    cv = pd.DataFrame(curves)
    cv.to_csv(OUT / "curves.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for q in (50, 60):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for ax, arm in zip(axes, ("expected", "sampled")):
            g = cv[(cv.q == q) & (cv.arm == arm)].groupby("d")
            med = g.median(numeric_only=True)
            ax.plot(med.index, med.e_star, color="0.3", ls="--", lw=1.4, label="e2*")
            ax.plot(med.index, med.e_pred, color="C2", lw=1.4, label="e_pred (median over seeds)")
            ax.plot(med.index, med.e_learned, color="C0", lw=1.4, label="e_learned (median over seeds)")
            ax.set_title(f"q={q}, arm={arm}", fontsize=10)
            ax.set_xlabel("d")
        axes[0].set_ylabel("stage-2 effort")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f"smoothed_game_q{q}.png", dpi=130)
        plt.close(fig)
    print(df.groupby(["q", "arm"])[["rmse_learned_minus_pred_pos", "rmse_learned_minus_estar_pos",
                                     "share_peak_gap_explained", "e_pred_0", "e_learned_0"]].median())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
