#!/usr/bin/env python3
"""Cusp diagnostic (descriptive; no RL runs): how fast the stage-2 actor fits the d = 0 peak.

Re-runs the Pilot 4 section 1d supervised fit with the identical setup (tools/v2/pilot4_repr_floor.py:
same BetaActor / head / init seeds 0..4, development D_2 grid, uniform least squares on the Beta mean,
full-batch Adam lr 1e-3, 300k-step cap, same plateau rule) and logs every 500 optimizer steps, on the
recovery grid (step 0.5, as utils.v2_metrics.recovery_metrics):
  signed peak error (e_hat_2(0) - e2*(0)) / e2*(0); location-free peak error
  (max_d e_hat_2(d) - e2*(0)) / e2*(0); RMSE over |d| < 2q divided by e2*(0).
The logging forward passes run under no_grad and do not touch the optimizer, so the trajectory is
the Pilot 4 one (checked: logged float32 losses vs the Pilot 4 loss logs fit_q*_init*.npz, every
common step). ``--identity-only`` re-runs that check from the saved curves without refitting.

Never a training target, never an initialization, never part of the method.

Usage: python tools/v2/cusp_diagnostic.py [--workers 10]
"""

from __future__ import annotations

import argparse
import json
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
from pilot4_repr_floor import INITS, LOG_EVERY, LR, MAX_STEPS, PATIENCE, REL_TOL  # noqa: E402
from agents.ppo_curriculum import BetaActor, PPOConfig  # noqa: E402
from run.run_final_dp_br import dense_grid  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG  # noqa: E402
from utils.theory_multistage import g2_two_stage  # noqa: E402
from utils.v2_metrics import symmetric_grid, zero_index  # noqa: E402

OUT = ROOT / "results" / "v2_T2_locked" / "cusp_diagnostic"
P4FITS = ROOT / "results" / "v2_pilots" / "pilot4" / "analysis" / "repr_floor" / "fits.csv"
METRIC_EVERY = 500
THRESH = (0.05, 0.03, 0.01)
PROTOCOL = ROOT / "protocols" / "v2_T2_locked.json"


def fit(job):
    q, s = job
    torch.set_num_threads(1)
    spec = spec_for(q)
    cfg = PPOConfig()
    D = dense_grid(spec.domain_half(2), DEV_CONFIG.state_step)
    y = torch.as_tensor(g2_two_stage(D, q, spec.w_h, spec.w_l, spec.k, spec.e_max), dtype=torch.float32)
    x = torch.as_tensor(spec.encode_obs(2, D))
    R = symmetric_grid(spec.domain_half(2), 0.5)
    xr = torch.as_tensor(spec.encode_obs(2, R))
    g2r = g2_two_stage(R, q, spec.w_h, spec.w_l, spec.k, spec.e_max)
    z = zero_index(R)
    g20 = float(g2r[z])
    pos = np.abs(R) < 2.0 * q
    net = BetaActor(cfg.hidden, cfg.c_min, cfg.mu_clamp, torch.Generator().manual_seed(s))
    opt = torch.optim.Adam(net.parameters(), lr=LR, betas=cfg.adam_betas, eps=cfg.adam_eps)
    best, best_step, log, curve = np.inf, 0, [], []
    step = 0
    while step < MAX_STEPS:
        step += 1
        a, b = net(x)
        loss = torch.mean((spec.e_min + spec.e_range * a / (a + b) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % METRIC_EVERY == 0:
            with torch.no_grad():
                ar, br = net(xr)
            ar, br = ar.numpy().astype(float), br.numpy().astype(float)
            e2 = spec.e_min + spec.e_range * ar / (ar + br)
            j = int(np.argmax(e2))
            curve.append({"q": q, "init_seed": s, "step": step, "loss_mse": float(loss),
                          "peak_rel_err_signed": (float(e2[z]) - g20) / g20,
                          "peak_locfree_rel_err": (float(e2[j]) - g20) / g20, "peak_locfree_argmax_d": float(R[j]),
                          "rmse_pos_over_g2_0": float(np.sqrt(np.mean((e2[pos] - g2r[pos]) ** 2))) / g20})
        if step % LOG_EVERY == 0:
            lv = float(loss)
            log.append((step, lv))
            if lv < best * (1 - REL_TOL):
                best, best_step = lv, step
            elif step - best_step >= PATIENCE:
                break
    return curve, {"q": q, "init_seed": s, "steps": step, "final_loss_mse": log[-1][1],
                   "stop": "plateau" if step < MAX_STEPS else "max_steps"}


def identity_check(cv: pd.DataFrame) -> pd.DataFrame:
    """Logged losses (float32 values) vs the Pilot 4 loss logs (exact, NPZ) at every common step."""
    rows = []
    for (q, s), c in cv.groupby(["q", "init_seed"]):
        lg = np.load(P4FITS.parent / f"fit_q{q}_init{s}.npz")["loss_log"]
        mine = dict(zip(c.step.astype(int), c.loss_mse))
        common = [(int(st), lv) for st, lv in lg if int(st) in mine]
        same = [np.float32(lv) == np.float32(mine[st]) for st, lv in common]
        rows.append({"q": q, "init_seed": s, "n_common_steps": len(common), "n_identical_float32": int(sum(same)),
                     "identical_to_pilot4": bool(all(same))})
    return pd.DataFrame(rows)


def first_below(c: pd.DataFrame, col: str, thr: float):
    hit = c[c[col].abs() < thr]
    return (int(hit.step.iloc[0]), hit.iloc[0]) if len(hit) else (None, None)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--identity-only", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.identity_only:
        r = identity_check(pd.read_csv(OUT / "curves_every500.csv"))
        r.to_csv(OUT / "identity_vs_pilot4.csv", index=False)
        print(r.to_string())
        return 0
    with Pool(a.workers) as pool:
        res = pool.map(fit, [(q, s) for q in (50, 60) for s in INITS], chunksize=1)
    cv = pd.DataFrame([r for c, _ in res for r in c])
    fins = pd.DataFrame([f for _, f in res])
    cv.to_csv(OUT / "curves_every500.csv", index=False)
    fins.to_csv(OUT / "fits_final.csv", index=False)
    identity_check(cv).to_csv(OUT / "identity_vs_pilot4.csv", index=False)
    rows = []
    for (q, s), c in cv.groupby(["q", "init_seed"]):
        c = c.sort_values("step")
        r = {"q": q, "init_seed": s}
        for thr in THRESH:
            st, row = first_below(c, "peak_rel_err_signed", thr)
            r[f"step_abs_peak_lt_{thr}"] = st
            if thr == 0.05:
                r["rmse_at_peak_lt_0.05"] = None if row is None else float(row.rmse_pos_over_g2_0)
                r["locfree_at_peak_lt_0.05"] = None if row is None else float(row.peak_locfree_rel_err)
            st2, _ = first_below(c, "peak_locfree_rel_err", thr)
            r[f"step_abs_locfree_lt_{thr}"] = st2
        rows.append(r)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "thresholds_per_init.csv", index=False)
    summ = []
    for q, g in per.groupby("q"):
        for c in per.columns:
            if c in ("q", "init_seed"):
                continue
            v = pd.to_numeric(g[c], errors="coerce")
            summ.append({"q": q, "quantity": c, "n_reached": int(v.notna().sum()), "median": v.median(), "min": v.min(), "max": v.max()})
    pd.DataFrame(summ).to_csv(OUT / "thresholds_summary.csv", index=False)
    proto = json.load(open(PROTOCOL))
    rl = {}
    for q in (50, 60):
        rec = proto["records"][str(q)]
        n_ep, mb, ep = rec["protocol"]["episodes_per_update"], rec["ppo"]["minibatch"], rec["ppo"]["epochs"]
        rows_per_update = n_ep                      # Phase A: one stage-2 transition per episode
        mbs = int(np.ceil(rows_per_update / mb))
        A = rec["protocol"]["phase_caps"]["A"]
        rl[q] = {"phase_A_updates": A, "rows_per_update": rows_per_update, "minibatch": mb, "epochs": ep,
                 "minibatches_per_update": mbs, "actor_steps_per_update": ep * mbs, "actor_steps_all_A": A * ep * mbs,
                 "actor_steps_last_400": 400 * ep * mbs}
    json.dump(rl, open(OUT / "rl_actor_steps.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(fins.to_string())
    print(pd.DataFrame(summ).to_string())
    print(json.dumps(rl, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
