#!/usr/bin/env python3
"""Claim-A non-locking kappa-continuation — Phase 01 design analysis. ZERO GPU.

Three sampled-only analyses (no training, no analytic e* in any in-loop quantity;
e*=25 appears ONLY as a reporting benchmark):

  1. Smoothed-equilibrium curve mu*(kappa): for each kappa, sweep opponent-policy
     mean mu; compute the sampled best response b(mu, kappa) of a deterministic
     candidate vs 2 opponents drawn from the ModeConc Beta family
     (alpha+beta = kappa+2, mean = mu), CRN. The fixed point b(mu*) = mu* is the
     equilibrium of the kappa-smoothed game. Also the deterministic (kappa=inf)
     crossing via exploitability_frozen_profile. This is the moving target the
     continuation ladder must track, and it fixes kappa_top.

  2. Velocity-death autopsy on the Component-2 ramp segments (seeds 42-45):
     per kappa stage, mode velocity vs mean approx_kl vs mean batch_entropy.
     Distinguishes optimizer starvation (KL collapsed -> floors can fix) from
     gradient-SNR physics (KL healthy, velocity ~0 -> floors cannot fix).

  3. Ladder budget estimate combining 1 + 2.

Outputs a printed report + JSON dump into the task folder.

3P q35: k=0.001, w_H=6.5, w_L=3.0, q=35, n=3, bounds [0,100], e*=25 (benchmark only).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mc_br_polish import exploitability_frozen_profile

# ------------------------------- game params --------------------------------
K = 0.001
W_H = 6.5
W_L = 3.0
Q = 35.0
N = 3
L = np.zeros(N)
KVEC = np.full(N, K)
BOUNDS = (0.0, 100.0)
E_STAR_BENCH = 25.0  # reporting benchmark ONLY — never enters any BR/gate below
C2_GLOB = "results/three_players/convergence/ppo_3p_q35.0_seed{seed}_c2_mode_conc_convergence.json"
SEEDS = [42, 43, 44, 45, 46]
TASK_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "docs", "tasks", "claim-a-nonlocking-continuation"))

KAPPAS = [20.0, 35.0, 60.0, 100.0, 200.0, 400.0]
MU_GRID = np.arange(18.0, 26.5 + 1e-9, 0.5)
BR_SEEDS = [11, 12]           # replicate CRN seeds per (kappa, mu) point
M_BR = 100_000


def _beta_from_mean_kappa(mu_effort: float, kappa: float) -> Tuple[float, float]:
    """ModeConc family (alpha+beta = kappa+2) parameterized by its MEAN in effort units."""
    lo, hi = BOUNDS
    m = (mu_effort - lo) / (hi - lo)
    alpha = m * (kappa + 2.0)
    beta = (kappa + 2.0) - alpha
    if not (alpha > 1.0 and beta > 1.0):
        raise ValueError(f"mean {mu_effort} not representable at kappa {kappa} "
                         f"(alpha={alpha:.3f}, beta={beta:.3f}; need both > 1)")
    return alpha, beta


def br_vs_smoothed(mu: float, kappa: float, *, M: int = M_BR, seed: int = 0) -> float:
    """Sampled BR of a deterministic candidate vs 2 opponents ~ Beta(mean=mu, kappa).

    Coarse-to-fine grid, CRN across candidates. Sampled-only: realized argmax winner,
    w_H/w_L prizes, quadratic cost. No closed form anywhere.
    """
    lo, hi = BOUNDS
    alpha, beta = _beta_from_mean_kappa(mu, kappa)
    rng = np.random.default_rng(seed)
    opp = lo + (hi - lo) * rng.beta(alpha, beta, size=(M, N - 1))   # (M, 2)
    eps = rng.uniform(-Q, Q, size=(M, N))                           # (M, 3)
    others_max = (opp + eps[:, 1:]).max(axis=1)                     # (M,)

    def _payoff(cands: np.ndarray) -> np.ndarray:
        yi = cands[:, None] + eps[:, 0][None, :]                    # (G, M)
        win = yi > others_max[None, :]
        prize = np.where(win, W_H, W_L)
        return prize.mean(axis=1) - K * (cands ** 2)

    coarse = np.arange(5.0, 45.0 + 1e-9, 0.5)
    c = coarse[int(_payoff(coarse).argmax())]
    fine = np.arange(max(lo, c - 1.5), min(hi, c + 1.5) + 1e-9, 0.1)
    return float(fine[int(_payoff(fine).argmax())])


# ------------------------------ Analysis 1 ----------------------------------
def _root_from_curve(mus: np.ndarray, gs: np.ndarray) -> Optional[float]:
    """Root of g(mu) = b(mu) - mu via local linear fit around the sign change."""
    sign = np.sign(gs)
    idx = None
    for i in range(1, len(mus)):
        if sign[i - 1] > 0 and sign[i] <= 0:
            idx = i
            break
    if idx is None:
        return None
    lo, hi = max(0, idx - 3), min(len(mus), idx + 3)
    slope, intercept = np.polyfit(mus[lo:hi], gs[lo:hi], 1)
    if abs(slope) < 1e-9:
        return None
    return float(-intercept / slope)


def analysis_mu_star() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for kappa in KAPPAS:
        rows = []
        for mu in MU_GRID:
            brs = [br_vs_smoothed(float(mu), kappa,
                                  seed=int(1e6 + kappa * 1000 + mu * 10 + s))
                   for s in BR_SEEDS]
            b = float(np.mean(brs))
            rows.append({"mu": float(mu), "br": round(b, 3),
                         "g": round(b - float(mu), 3)})
        mus = np.array([r["mu"] for r in rows])
        gs = np.array([r["g"] for r in rows])
        mu_star = _root_from_curve(mus, gs)
        out[f"kappa{kappa:g}"] = {"kappa": kappa, "curve": rows,
                                  "mu_star": (round(mu_star, 2)
                                              if mu_star is not None else None)}
    # deterministic (kappa = inf) crossing, reusing the certified frozen-profile BR
    rows = []
    for mu in np.arange(22.0, 26.5 + 1e-9, 0.5):
        brs = []
        for s in (21, 22):
            _, br_arr = exploitability_frozen_profile(
                np.full(N, float(mu)), L, KVEC, W_H, W_L, Q, BOUNDS,
                M=200_000, grid_step=0.25, seed=880_000 + int(mu * 10) + s)
            brs.append(float(br_arr[0]))
        b = float(np.mean(brs))
        rows.append({"mu": float(mu), "br": round(b, 3), "g": round(b - float(mu), 3)})
    mus = np.array([r["mu"] for r in rows])
    gs = np.array([r["g"] for r in rows])
    mu_star = _root_from_curve(mus, gs)
    out["kappa_inf"] = {"kappa": None, "curve": rows,
                        "mu_star": round(mu_star, 2) if mu_star is not None else None}
    return out


# ------------------------------ Analysis 2 ----------------------------------
def _load_c2(seed: int) -> Optional[dict]:
    fs = glob.glob(C2_GLOB.format(seed=seed))
    return json.load(open(fs[0])) if fs else None


def _segments(kappa: List[float], trig: int, end: int) -> List[Tuple[int, int, float]]:
    """(start, end, kappa_value) segments of the ramp window [trig, end]."""
    segs = []
    seg_start = trig
    for i in range(trig + 1, end + 1):
        if kappa[i] != kappa[i - 1] or i == end:
            segs.append((seg_start, i, float(kappa[seg_start])))
            seg_start = i
    return segs


def analysis_autopsy() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for seed in SEEDS:
        d = _load_c2(seed)
        if d is None:
            continue
        mode = d["mode_effort"]
        kappa = d["kappa"]
        phases = d["ramp_phase"]
        kl = d.get("approx_kl", [])
        ent = d.get("batch_entropy", [])
        n = len(mode)
        trig = done = None
        for i in range(1, n):
            if phases[i - 1] == "explore" and phases[i] == "ramping" and trig is None:
                trig = i
            if phases[i - 1] == "ramping" and phases[i] == "done" and done is None:
                done = i
        recs = []

        def _seg_stats(a: int, b: int, kval, label: str) -> dict:
            du = max(1, b - a)
            vel = (mode[b] - mode[a]) / du
            klw = [x for x in kl[a:b] if x is not None and np.isfinite(x)]
            enw = [x for x in ent[a:b] if x is not None and np.isfinite(x)]
            return {"label": label, "kappa": kval, "start": a, "end": b,
                    "updates": b - a, "vel_per_update": round(vel, 4),
                    "mean_approx_kl": round(float(np.mean(klw)), 5) if klw else None,
                    "mean_batch_entropy": (round(float(np.mean(enw)), 4)
                                           if enw else None)}

        if trig is not None:
            a0 = max(0, trig - 200)
            recs.append(_seg_stats(a0, trig, float(kappa[trig - 1]), "explore_tail"))
            end = done if done is not None else n - 1
            for (a, b, kv) in _segments(kappa, trig, end):
                recs.append(_seg_stats(a, b, kv, f"ramp_k{kv:g}"))
            if done is not None and n - 1 > done:
                recs.append(_seg_stats(done, n - 1, float(kappa[done]), "done_tail"))
        else:
            recs.append(_seg_stats(max(0, n - 1 - 300), n - 1,
                                   float(kappa[-1]), "explore_last300"))
        out[f"seed{seed}"] = {"trigger": trig, "done": done, "n": n, "segments": recs}
    return out


# ------------------------------ Analysis 3 ----------------------------------
def analysis_ladder(mu_star: Dict[str, dict], autopsy: Dict[str, dict]) -> dict:
    """Ladder targets + per-stage climb + budget under a conservative velocity."""
    targets = []
    for kappa in KAPPAS:
        ms = mu_star[f"kappa{kappa:g}"]["mu_star"]
        targets.append({"kappa": kappa, "mu_star": ms})
    det = mu_star["kappa_inf"]["mu_star"]
    # healthy velocity = median of POSITIVE ramp-stage velocities across seeds
    vels = [s["vel_per_update"] for sv in autopsy.values()
            for s in sv["segments"]
            if s["label"].startswith("ramp_k") and s["vel_per_update"] > 0]
    v_healthy = float(np.median(vels)) if vels else float("nan")
    v_consv = float(np.percentile(vels, 25)) if vels else float("nan")
    climbs, budget = [], 0.0
    prev = None
    for t in targets:
        if t["mu_star"] is None:
            continue
        if prev is not None:
            dist = t["mu_star"] - prev
            need = dist / v_consv if (np.isfinite(v_consv) and v_consv > 0) else None
            climbs.append({"kappa": t["kappa"], "climb": round(dist, 2),
                           "updates_at_p25_vel": (round(need, 0)
                                                  if need is not None else None)})
            if need is not None and need > 0:
                budget += need
        prev = t["mu_star"]
    return {"targets": targets, "mu_star_deterministic": det,
            "vel_healthy_median": round(v_healthy, 4),
            "vel_conservative_p25": round(v_consv, 4),
            "per_stage_climbs": climbs,
            "ladder_budget_updates_p25": round(budget, 0)}


# --------------------------------- main -------------------------------------
def main() -> None:
    print("=" * 94)
    print("CLAIM-A kappa-continuation — PHASE 01 design analysis (3P q35). ZERO GPU.")
    print("=" * 94)

    mu_star = analysis_mu_star()
    print("\n[D1] Smoothed-equilibrium curve mu*(kappa)  (fixed point of sampled BR "
          "vs Beta(mean=mu, kappa) opponents)")
    for key in [f"kappa{k:g}" for k in KAPPAS] + ["kappa_inf"]:
        blk = mu_star[key]
        klabel = "inf(det)" if blk["kappa"] is None else f"{blk['kappa']:g}"
        print(f"  kappa={klabel:>8}: mu* = {blk['mu_star']}")
    print("  curves (mu: br):")
    for key in [f"kappa{k:g}" for k in KAPPAS] + ["kappa_inf"]:
        blk = mu_star[key]
        klabel = "inf" if blk["kappa"] is None else f"{blk['kappa']:g}"
        pts = "  ".join(f"{r['mu']:.1f}:{r['br']:.1f}" for r in blk["curve"][::2])
        print(f"    k={klabel:>5}: {pts}")

    autopsy = analysis_autopsy()
    print("\n[D2] Velocity-death autopsy (c2 ramp segments): vel vs approx_kl vs entropy")
    for sk, sv in autopsy.items():
        print(f"  {sk}: trigger@{sv['trigger']} done@{sv['done']}")
        for s in sv["segments"]:
            print(f"     {s['label']:>15} kappa={s['kappa']:6.1f} upd={s['updates']:4d} "
                  f"vel={s['vel_per_update']:+.4f} kl={s['mean_approx_kl']} "
                  f"ent={s['mean_batch_entropy']}")

    ladder = analysis_ladder(mu_star, autopsy)
    print("\n[D3] Ladder design numbers")
    print(f"  targets: " + "  ".join(f"k{t['kappa']:g}->{t['mu_star']}"
                                     for t in ladder["targets"]))
    print(f"  deterministic mu* = {ladder['mu_star_deterministic']} "
          f"(benchmark e* = {E_STAR_BENCH})")
    print(f"  healthy vel (median of positive ramp-stage vels) = "
          f"{ladder['vel_healthy_median']}/upd; conservative p25 = "
          f"{ladder['vel_conservative_p25']}/upd")
    for c in ladder["per_stage_climbs"]:
        print(f"    to k={c['kappa']:g}: climb {c['climb']} units "
              f"-> ~{c['updates_at_p25_vel']} upd at p25 vel")
    print(f"  ladder budget (p25 vel, excl. stage-0 convergence): "
          f"~{ladder['ladder_budget_updates_p25']} updates")

    dump = {"params": {"k": K, "w_h": W_H, "w_l": W_L, "q": Q, "n": N,
                       "bounds": list(BOUNDS), "e_star_benchmark": E_STAR_BENCH,
                       "kappas": KAPPAS, "M_br": M_BR, "br_seeds": BR_SEEDS},
            "D1_mu_star": mu_star, "D2_autopsy": autopsy, "D3_ladder": ladder}
    out_path = os.path.join(TASK_DIR, "phase01_design.json")
    with open(out_path, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\n[dump] {out_path}")
    print("=" * 94)


if __name__ == "__main__":
    main()
