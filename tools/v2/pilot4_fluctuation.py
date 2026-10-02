#!/usr/bin/env python3
"""Pilot 4 section 1b: anatomy of the stage-1 fluctuation of e_hat_1(0) in the Pilot 3 runs.

Record used: the stability log in train_history.json (``stability[*].e_hat['1']``), the root
policy mean (Beta mean) every 20 updates at global updates 420, 440, ..., 1000. No per-update
record of the root policy mean exists (``history[*].mean_effort_by_stage`` is the batch mean of
SAMPLED efforts). The verifier calls (every 25 updates from u500) and the weight exports (every 25)
are sparser. The stability samples fall exactly on the opponent-refresh updates (global multiples
of 20): after update u the lagged opponent is refreshed to the actor at u, so the opponent for
updates u+1..u+20 has root mean e_hat_1(u), the same value the stability log records at u.

ACF: per run, x_t = e_hat_1(u_t) - e1* on the segment u_t >= START (650 -> 660..1000, n = 18; and
700 -> 700..1000, n = 16). Two versions: 'centred' (standard sample ACF about the segment mean,
denominator n * var) and 'about_e1star' (same with x not demeaned, i.e. deviations about e1*).
Also the ACF of the 25-update weight exports at lags 25, 50, 75, 100 (secondary).

Window regression: for every refresh update u with START <= u and u + 20 <= 1000,
y = e_hat_1(u + 20) - e_hat_1(u) (learner change over the window), x = e_opp - e1* = e_hat_1(u) - e1*.
OLS slope with intercept pooled over seeds per (q, arm) and per q (arms pooled); 95% CI by a
cluster bootstrap over seeds (10000 resamples, numpy seed 20261001). Only the window endpoints
enter, so per-update data would give the same regression.

Usage: python tools/v2/pilot4_fluctuation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from common import analytic_policy, spec_for  # noqa: E402
from induced_band import actor_policy  # noqa: E402

P3 = ROOT / "results" / "v2_pilots" / "pilot3"
OUT = ROOT / "results" / "v2_pilots" / "pilot4" / "analysis" / "fluctuation"
ARMS = {"B2_frozen_s1norm": "stochastic", "B2_frozen_s1norm_mean": "mean"}
STARTS = (650, 700)
N_BOOT, BOOT_SEED = 10000, 20261001
MAX_LAG = 8


def acf(x: np.ndarray, lags, centred: bool) -> np.ndarray:
    """Sample ACF at integer lags (biased estimator, denominator n * c0)."""
    x = np.asarray(x, float)
    y = x - x.mean() if centred else x
    c0 = np.dot(y, y)
    return np.array([np.dot(y[:-L], y[L:]) / c0 if L < y.size else np.nan for L in lags])


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope with intercept."""
    xc = x - x.mean()
    return float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))


def load_series():
    rows, exp = [], []
    for q in (50, 60):
        g1 = float(analytic_policy(spec_for(q))(1, np.zeros(1))[0])
        for seed in range(10501, 10511):
            for arm, short in ARMS.items():
                d = P3 / f"q{q}" / f"seed{seed}" / arm
                h = json.load(open(d / "train_history.json"))
                for s in h["stability"]:
                    if s["phase"] != "B":
                        continue
                    rows.append({"q": q, "seed": seed, "arm": short, "update": int(s["update"]),
                                 "e1": float(s["e_hat"]["1"][0]), "e1_star": g1})
                for f in sorted(d.glob("weights/u*.npz")):
                    u = int(f.name[1:6])
                    if u > 400:
                        exp.append({"q": q, "seed": seed, "arm": short, "update": u, "e1_star": g1,
                                    "e1": float(actor_policy(str(f), spec_for(q))(1, np.zeros(1))[0])})
    return pd.DataFrame(rows), pd.DataFrame(exp)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    st, ex = load_series()
    st.to_csv(OUT / "stability_series_e1.csv", index=False)
    ex.to_csv(OUT / "export_series_e1.csv", index=False)
    step = int(np.diff(sorted(st["update"].unique())).min())
    acf_rows, acf25 = [], []
    for start in STARTS:
        for (q, seed, arm), g in st.groupby(["q", "seed", "arm"]):
            g = g[g["update"] >= start].sort_values("update")
            x = (g.e1 - g.e1_star).to_numpy()
            if not np.all(np.diff(g["update"]) == step):
                raise RuntimeError("irregular stability record")
            lags = range(1, MAX_LAG + 1)
            for kind in ("centred", "about_e1star"):
                for L, v in zip(lags, acf(x, lags, kind == "centred")):
                    acf_rows.append({"segment_start": start, "q": q, "seed": seed, "arm": arm, "kind": kind,
                                     "lag_updates": L * step, "acf": v, "n": x.size})
        for (q, seed, arm), g in ex.groupby(["q", "seed", "arm"]):
            g = g[g["update"] >= start].sort_values("update")
            x = (g.e1 - g.e1_star).to_numpy()
            for kind in ("centred", "about_e1star"):
                for L, v in zip(range(1, 5), acf(x, range(1, 5), kind == "centred")):
                    acf25.append({"segment_start": start, "q": q, "seed": seed, "arm": arm, "kind": kind,
                                  "lag_updates": 25 * L, "acf": v, "n": x.size})
    a = pd.DataFrame(acf_rows)
    a.to_csv(OUT / "acf_per_run_every20.csv", index=False)
    a25 = pd.DataFrame(acf25)
    a25.to_csv(OUT / "acf_per_run_exports25.csv", index=False)
    for df, name in ((a, "acf_summary_every20.csv"), (a25, "acf_summary_exports25.csv")):
        df.groupby(["segment_start", "kind", "q", "arm", "lag_updates"]).acf.agg(
            median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75), n_runs="size"
        ).reset_index().to_csv(OUT / name, index=False)
    # ---- window regression
    win = []
    for (q, seed, arm), g in st.groupby(["q", "seed", "arm"]):
        g = g.sort_values("update").set_index("update")
        for u in g.index:
            if u + step in g.index:
                win.append({"q": q, "seed": seed, "arm": arm, "u_start": u, "x_opp_dev": g.e1[u] - g.e1_star[u],
                            "y_learner_change": g.e1[u + step] - g.e1[u]})
    w = pd.DataFrame(win)
    w.to_csv(OUT / "windows.csv", index=False)
    rng = np.random.default_rng(BOOT_SEED)
    reg = []
    for start in STARTS:
        ws = w[w.u_start >= start]
        groups = [((q, arm), g) for (q, arm), g in ws.groupby(["q", "arm"])] + \
                 [((q, "both"), g) for q, g in ws.groupby("q")]
        for (q, arm), g in groups:
            units = sorted({(s, a_) for s, a_ in zip(g.seed, g.arm)})
            by = {u_: g[(g.seed == u_[0]) & (g.arm == u_[1])] for u_ in units}
            b = []
            for _ in range(N_BOOT):
                pick = [units[i] for i in rng.integers(0, len(units), len(units))]
                gg = pd.concat([by[p] for p in pick])
                b.append(ols_slope(gg.x_opp_dev.to_numpy(), gg.y_learner_change.to_numpy()))
            per_run = [ols_slope(by[u_].x_opp_dev.to_numpy(), by[u_].y_learner_change.to_numpy()) for u_ in units]
            reg.append({"segment_start": start, "q": q, "arm": arm, "n_windows": len(g), "n_runs": len(units),
                        "slope": ols_slope(g.x_opp_dev.to_numpy(), g.y_learner_change.to_numpy()),
                        "boot_ci95_lo": float(np.percentile(b, 2.5)), "boot_ci95_hi": float(np.percentile(b, 97.5)),
                        "per_run_slope_median": float(np.median(per_run)),
                        "per_run_slope_q25": float(np.percentile(per_run, 25)),
                        "per_run_slope_q75": float(np.percentile(per_run, 75))})
    r = pd.DataFrame(reg)
    r.to_csv(OUT / "window_regression.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.to_string())
    print(pd.read_csv(OUT / "acf_summary_every20.csv").query("kind=='centred' and segment_start==650").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
