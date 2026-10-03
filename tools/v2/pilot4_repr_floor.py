#!/usr/bin/env python3
"""Pilot 4 section 1d: representation floor of the stage-2 actor (diagnostic only).

Never a training target, never an initialization, never part of the method.

Fit: the training actor class ``agents.ppo_curriculum.BetaActor`` with the protocol config
(2 -> 64 tanh -> 64 tanh -> 2; hidden layers orthogonal gain sqrt(2), bias 0; output head zero;
mu = clamp(sigmoid(z_mu), 1e-6, 1 - 1e-6), c = 100 + softplus(z_c), alpha = mu c, beta = (1 - mu) c),
input ``GameSpec.encode_obs(2, d)`` (float32), float32 throughout. The Beta mean
e_min + range * alpha / (alpha + beta) is fitted to e2*(d) by least squares with uniform weights over
the development D_2 grid (state step 4). Full-batch Adam, lr 1e-3, betas (0.9, 0.999), eps 1e-8, no
gradient clipping. Plateau rule: the loss is recorded every 1,000 steps; stop when the best loss has
not improved by at least 0.1% (relative) over the last 20,000 steps, or at 300,000 steps.
5 initializations per q: torch.Generator().manual_seed(s), s in 0..4 (orthogonal init draws).

Evaluation: complete candidate (stage 1 = e1*(0), stage 2 = fitted mapping) on the development
tier via ``utils.v2_metrics.evaluate``; recovery metrics on the recovery grid (step 0.5).
Comparison: RL stage-2 policies of the Phase A extension at u1600 (weights/u01600.npz) with the
smoothed-game prediction e_pred(0) at u1600 from results/v2_pilots/phaseA_ext/analysis/
curves_weights_every25.csv (Pilot-1 method).

Usage: python tools/v2/pilot4_repr_floor.py [--workers 10]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from common import analytic_policy, spec_for  # noqa: E402
from pilot4_common import compose, eval_row, mean_fn  # noqa: E402
from agents.ppo_curriculum import BetaActor, CurriculumPPO, PPOConfig  # noqa: E402
from run.run_final_dp_br import dense_grid, make_policy_fns  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG  # noqa: E402
from utils.theory_multistage import g2_two_stage  # noqa: E402

OUT = ROOT / "results" / "v2_pilots" / "pilot4" / "analysis" / "repr_floor"
PX = ROOT / "results" / "v2_pilots" / "phaseA_ext"
LR, LOG_EVERY, PATIENCE, REL_TOL, MAX_STEPS = 1e-3, 1000, 20000, 1e-3, 300000
INITS = range(5)


def fit(job):
    q, s = job
    torch.set_num_threads(1)
    spec = spec_for(q)
    cfg = PPOConfig()
    D = dense_grid(spec.domain_half(2), DEV_CONFIG.state_step)
    y = torch.as_tensor(g2_two_stage(D, q, spec.w_h, spec.w_l, spec.k, spec.e_max), dtype=torch.float32)
    x = torch.as_tensor(spec.encode_obs(2, D))
    net = BetaActor(cfg.hidden, cfg.c_min, cfg.mu_clamp, torch.Generator().manual_seed(s))
    opt = torch.optim.Adam(net.parameters(), lr=LR, betas=cfg.adam_betas, eps=cfg.adam_eps)
    best, best_step, log = np.inf, 0, []
    step = 0
    while step < MAX_STEPS:
        step += 1
        a, b = net(x)
        loss = torch.mean((spec.e_min + spec.e_range * a / (a + b) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % LOG_EVERY == 0:
            lv = float(loss)
            log.append((step, lv))
            if lv < best * (1 - REL_TOL):
                best, best_step = lv, step
            elif step - best_step >= PATIENCE:
                break
    agent = CurriculumPPO(cfg, torch.Generator().manual_seed(0), np.random.default_rng(0))
    agent.actor.load_state_dict(net.state_dict())
    mf = make_policy_fns(agent, spec)[0]
    g1 = float(analytic_policy(spec)(1, np.zeros(1))[0])
    row = eval_row(compose(lambda t, d: np.full(np.asarray(d).shape, g1), mf), spec, full=True)
    e_fit = np.asarray(mf(2, D), float)
    res = e_fit - y.numpy().astype(float)
    tail = np.abs(D) >= 2 * q
    j = int(np.argmax(np.abs(res)))
    row.update({"q": q, "init_seed": s, "steps": step, "final_loss_mse": log[-1][1], "best_loss_mse": best,
                "stop": "plateau" if step < MAX_STEPS else "max_steps", "rmse_fit_grid": float(np.sqrt(np.mean(res ** 2))),
                "max_abs_resid": float(np.abs(res).max()), "max_abs_resid_at_d": float(D[j]),
                "tail_min_fit": float(e_fit[tail].min()), "tail_max_fit": float(e_fit[tail].max()),
                "n_grid": int(D.size)})
    np.savez(OUT / f"fit_q{q}_init{s}.npz", d=D, e_fit=e_fit, e_star=y.numpy(), loss_log=np.array(log))
    torch.save(net.state_dict(), OUT / f"fit_q{q}_init{s}.pt")
    return row


def rl_rows():
    """RL stage-2 policies at u1600 (Phase A extension): same metric set."""
    cv = pd.read_csv(PX / "analysis" / "curves_weights_every25.csv")
    rows = []
    for q in (50, 60):
        spec = spec_for(q)
        for seed in range(10501, 10511):
            mf = mean_fn(str(PX / f"q{q}" / f"seed{seed}" / "expected_ext" / "weights" / "u01600.npz"), spec)
            r = eval_row(mf, spec)
            c = cv[(cv.q == q) & (cv.seed == seed) & (cv["update"] == 1600)].iloc[0]
            r.update({"q": q, "seed": seed, "e_pred_0": float(c.e_pred_0), "e_learned_0_curves": float(c.e_learned_0)})
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        fits = pd.DataFrame(pool.map(fit, [(q, s) for q in (50, 60) for s in INITS], chunksize=1))
    fits.to_csv(OUT / "fits.csv", index=False)
    rl = rl_rows()
    rl.to_csv(OUT / "rl_u1600.csv", index=False)
    three = []
    for q in (50, 60):
        f, r = fits[fits.q == q], rl[rl.q == q]
        g20 = float(r.g2_at_0.iloc[0])
        rl_gap = g20 - r.e2_at_0
        smooth = g20 - r.e_pred_0
        floor = g20 - f.e2_at_0
        for name, x in (("rl_peak_gap_d0", rl_gap), ("smoothing_predicted_gap_d0", smooth), ("supervised_floor_gap_d0", floor),
                        ("rl_locfree_rel_err", r.stage2_peak_locfree_rel_err), ("floor_locfree_rel_err", f.stage2_peak_locfree_rel_err)):
            three.append({"q": q, "quantity": name, "n": int(x.size), "median": float(x.median()), "min": float(x.min()),
                          "max": float(x.max()), "median_over_e2star0": float(x.median()) / g20 if "gap" in name else None})
    pd.DataFrame(three).to_csv(OUT / "three_way_peak_gap.csv", index=False)
    pd.set_option("display.width", 250)
    print(fits[["q", "init_seed", "steps", "stop", "final_loss_mse", "stage2_peak_rel_err_signed", "stage2_peak_locfree_rel_err",
                "stage2_peak_locfree_argmax_d", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean", "stage2_tail_max",
                "stage2_sym_err_max", "eta_T_over_dw", "Gmax_full_over_dw", "max_abs_resid", "max_abs_resid_at_d"]].to_string())
    print(pd.DataFrame(three).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
