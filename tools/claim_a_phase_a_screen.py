#!/usr/bin/env python3
"""Claim-A dev-distance-trigger — Phase A zero-GPU feasibility screen.

Three sampled-only analyses (NO GPU, NO training, NO analytic e* in any trigger):

  1. Deterministic BR curve (high-kappa regime, signal-in-principle):
     sweep a symmetric deterministic center c in [14, 28]; report payoff-gain(c)
     and best-response distance |BR(c) - c|. Tests whether BR-distance is steep
     near e*=25 while gain is flat.

  2. Explore-kappa stochastic backtest (decisive):
     at each Component-2 seed's stored (mode, kappa) trajectory, reconstruct the
     best response against opponents SAMPLED from the actual explore-kappa Beta
     policy, and read the BR-distance at the historical gain-trigger point
     (mode ~ 18) and along the trajectory. Tests whether a distance trigger would
     ALSO fire early (same failure) or correctly wait near e*.

  3. Ramp window calibration:
     from the c2 ramp segments, measure mode velocity (units/update) per kappa
     stage and estimate the stage_hold needed to cover the residual distance.

Outputs a printed report + a JSON dump. Read-only on results/*/convergence/.

3P q35 params: k=0.001, w_H=6.5, w_L=3.0, q=35, l=[0,0,0], n=3, bounds=[0,100],
e*=25.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mc_br_polish import exploitability_frozen_profile, sampled_payoff_player

# ------------------------------- game params --------------------------------
K = 0.001
W_H = 6.5
W_L = 3.0
Q = 35.0
N = 3
L = np.zeros(N)
KVEC = np.full(N, K)
BOUNDS = (0.0, 100.0)
E_STAR = 25.0
C2_GLOB = "results/three_players/convergence/ppo_3p_q35.0_seed{seed}_c2_mode_conc_convergence.json"
SEEDS = [42, 43, 44, 45, 46]


def _beta_params_from_mode_kappa(mode_effort: float, kappa: float) -> Tuple[float, float]:
    """ActorCriticModeConc inverse: mode (effort units) + kappa -> (alpha, beta)."""
    lo, hi = BOUNDS
    s = (mode_effort - lo) / (hi - lo)
    s = min(max(s, 1e-6), 1 - 1e-6)
    alpha = 1.0 + s * kappa
    beta = 1.0 + (1.0 - s) * kappa
    return alpha, beta


def _beta_mean_effort(alpha: float, beta: float) -> float:
    lo, hi = BOUNDS
    return lo + (hi - lo) * alpha / (alpha + beta)


def stochastic_br(
    mode_effort: float,
    kappa: float,
    *,
    M: int = 120_000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Best response of one player vs 2 opponents SAMPLED from the explore-kappa
    Beta policy Beta(alpha, beta) (CRN over noise + opponent draws).

    Returns (BR_effort, policy_mean_effort). Coarse-to-fine grid over [5, 45].
    """
    lo, hi = BOUNDS
    alpha, beta = _beta_params_from_mode_kappa(mode_effort, kappa)
    mean_eff = _beta_mean_effort(alpha, beta)
    rng = np.random.default_rng(seed)
    # opponents (2) drawn from the policy Beta, scaled to effort units; own noise + opp noise
    opp = lo + (hi - lo) * rng.beta(alpha, beta, size=(M, N - 1))     # (M, 2)
    eps = rng.uniform(-Q, Q, size=(M, N))                            # (M, 3): [own, opp1, opp2]
    y_opp = opp + eps[:, 1:]                                          # (M, 2)
    others_max = y_opp.max(axis=1)                                   # (M,)

    def _payoff(cands: np.ndarray) -> np.ndarray:
        yi = cands[:, None] + eps[:, 0][None, :]                     # (G, M)
        win = yi > others_max[None, :]
        prize = np.where(win, W_H, W_L)
        return prize.mean(axis=1) - K * (cands ** 2)

    coarse = np.arange(5.0, 45.0 + 1e-9, 0.5)
    vc = _payoff(coarse)
    c = coarse[int(vc.argmax())]
    fine = np.arange(max(lo, c - 1.5), min(hi, c + 1.5) + 1e-9, 0.1)
    vf = _payoff(fine)
    br = float(fine[int(vf.argmax())])
    return br, float(mean_eff)


# ------------------------------ Analysis 1 ----------------------------------
def analysis_deterministic_curve() -> List[Dict[str, float]]:
    """gain(c) and |BR(c)-c| for symmetric deterministic center c in [14,28]."""
    rows = []
    for c in np.arange(14.0, 28.0 + 1e-9, 0.5):
        prof = np.full(N, float(c))
        gain, brs = exploitability_frozen_profile(
            prof, L, KVEC, W_H, W_L, Q, BOUNDS, M=200_000, grid_step=0.25,
            seed=770_000 + int(round(c * 10)),
        )
        br = float(brs[0])
        rows.append({"c": float(c), "gain": float(gain),
                     "br": br, "dist": abs(br - float(c))})
    return rows


# ------------------------------ Analysis 2 ----------------------------------
def _load_c2(seed: int) -> Optional[dict]:
    fs = glob.glob(C2_GLOB.format(seed=seed))
    if not fs:
        return None
    return json.load(open(fs[0]))


def _phase_transitions(phases: List[str]) -> Dict[str, Optional[int]]:
    trig = done = None
    for i in range(1, len(phases)):
        if phases[i - 1] == "explore" and phases[i] == "ramping" and trig is None:
            trig = i
        if phases[i - 1] == "ramping" and phases[i] == "done" and done is None:
            done = i
    return {"trigger": trig, "done": done}


def analysis_stochastic_backtest() -> Dict[str, dict]:
    """Reconstruct explore-kappa BR-distance at the historical trigger + checkpoints."""
    out: Dict[str, dict] = {}
    for seed in SEEDS:
        d = _load_c2(seed)
        if d is None:
            continue
        mode = d["mode_effort"]
        kappa = d["kappa"]
        phases = d["ramp_phase"]
        tr = _phase_transitions(phases)
        n = len(mode)
        # checkpoints: trigger (if any) + every 150 updates through the explore phase
        trig = tr["trigger"]
        cps = sorted(set(
            ([trig] if trig is not None else [])
            + list(range(150, (trig if trig is not None else n), 150))
        ))
        recs = []
        for t in cps:
            if t >= n:
                continue
            # window-average mode over 5 updates for stability
            w0 = max(0, t - 4)
            m_t = float(np.mean(mode[w0:t + 1]))
            k_t = float(kappa[t])
            br, mean_eff = stochastic_br(m_t, k_t, M=120_000, seed=1000 + seed * 7 + t)
            recs.append({
                "update": int(t), "is_trigger": bool(trig is not None and t == trig),
                "kappa": k_t, "mode": round(m_t, 3), "policy_mean": round(mean_eff, 3),
                "br": round(br, 3), "dist_br_mean": round(abs(br - mean_eff), 3),
            })
        out[f"seed{seed}"] = {"trigger_update": trig, "done_update": tr["done"],
                              "n_updates": n, "checkpoints": recs}
    return out


# ------------------------------ Analysis 3 ----------------------------------
def analysis_window_calibration() -> Dict[str, dict]:
    """Per-kappa-stage mode velocity and implied stage_hold from c2 ramp segments."""
    out: Dict[str, dict] = {}
    for seed in SEEDS:
        d = _load_c2(seed)
        if d is None:
            continue
        mode = d["mode_effort"]
        kappa = d["kappa"]
        phases = d["ramp_phase"]
        tr = _phase_transitions(phases)
        trig, done = tr["trigger"], tr["done"]
        if trig is None:
            out[f"seed{seed}"] = {"triggered": False}
            continue
        end = done if done is not None else len(mode) - 1
        # per-stage (distinct kappa value) velocity across the ramp window
        stages: List[Dict[str, float]] = []
        seg_start = trig
        for i in range(trig + 1, end + 1):
            if kappa[i] != kappa[i - 1] or i == end:
                seg_end = i
                du = seg_end - seg_start
                if du >= 1:
                    dv = (mode[seg_end] - mode[seg_start]) / du
                    stages.append({"kappa": float(kappa[seg_start]),
                                   "updates": int(du),
                                   "vel_per_update": round(dv, 4),
                                   "mode_start": round(mode[seg_start], 3),
                                   "mode_end": round(mode[seg_end], 3)})
                seg_start = i
        resid_at_trigger = E_STAR - float(np.mean(mode[max(0, trig - 4):trig + 1]))
        # dominant (fastest) velocity = the lowest-kappa stage; implied hold to
        # cover the residual at that velocity (upper bound on useful window)
        vmax = max((s["vel_per_update"] for s in stages), default=0.0)
        implied_hold = (resid_at_trigger / vmax) if vmax > 1e-6 else float("inf")
        out[f"seed{seed}"] = {
            "triggered": True, "trigger_update": trig, "done_update": done,
            "resid_dist_at_trigger": round(resid_at_trigger, 3),
            "stages": stages, "max_vel_per_update": round(vmax, 4),
            "implied_hold_to_close_resid": (round(implied_hold, 1)
                                            if np.isfinite(implied_hold) else None),
        }
    return out


# --------------------------------- main -------------------------------------
def main() -> None:
    print("=" * 92)
    print("CLAIM-A dev-distance-trigger — PHASE A screen (3P q35, e*=25). ZERO GPU.")
    print("=" * 92)

    det = analysis_deterministic_curve()
    print("\n[A1] Deterministic BR curve (symmetric center c; high-kappa regime)")
    print("  c      gain      BR      |BR-c|")
    for r in det:
        mark = "  <- e*" if abs(r["c"] - E_STAR) < 1e-6 else ""
        print(f"  {r['c']:5.1f}  {r['gain']:.5f}  {r['br']:6.2f}  {r['dist']:6.2f}{mark}")
    # signal summary: gain spread vs distance slope in [18, 25]
    seg = [r for r in det if 18.0 <= r["c"] <= 25.0]
    gain_rng = (min(x["gain"] for x in seg), max(x["gain"] for x in seg))
    dist_at_18 = next(r["dist"] for r in det if abs(r["c"] - 18.0) < 1e-6)
    dist_at_25 = next(r["dist"] for r in det if abs(r["c"] - 25.0) < 1e-6)
    print(f"  [A1 signal] over c in [18,25]: gain range={gain_rng[0]:.4f}..{gain_rng[1]:.4f}"
          f" (span {gain_rng[1]-gain_rng[0]:.4f}); |BR-c| {dist_at_18:.2f}(@18) -> "
          f"{dist_at_25:.2f}(@25)")

    stoch = analysis_stochastic_backtest()
    print("\n[A2] Explore-kappa stochastic backtest (BR vs SAMPLED opponents at kappa=20)")
    for sk, sv in stoch.items():
        print(f"  {sk}: trigger@{sv['trigger_update']} done@{sv['done_update']} "
              f"n={sv['n_updates']}")
        for r in sv["checkpoints"]:
            tag = " TRIGGER" if r["is_trigger"] else ""
            print(f"     upd={r['update']:5d} kappa={r['kappa']:5.1f} mode={r['mode']:6.2f} "
                  f"mean={r['policy_mean']:6.2f} BR={r['br']:6.2f} "
                  f"|BR-mean|={r['dist_br_mean']:5.2f}{tag}")
    trig_dists = [rr["dist_br_mean"] for sv in stoch.values()
                  for rr in sv["checkpoints"] if rr["is_trigger"]]
    if trig_dists:
        print(f"  [A2 signal] BR-distance AT the historical gain-trigger point: "
              f"{[f'{x:.2f}' for x in trig_dists]} "
              f"(mean {np.mean(trig_dists):.2f}) — a distance trigger with "
              f"tau_dist~1.0 would NOT fire here iff these are >> 1.")

    win = analysis_window_calibration()
    print("\n[A3] Ramp window calibration (per-kappa-stage mode velocity, c2 ramp segments)")
    holds = []
    for sk, sv in win.items():
        if not sv.get("triggered"):
            print(f"  {sk}: NOT triggered (no ramp segment)")
            continue
        print(f"  {sk}: resid@trigger={sv['resid_dist_at_trigger']:.2f} "
              f"max_vel={sv['max_vel_per_update']:.4f}/upd "
              f"implied_hold_to_close_resid={sv['implied_hold_to_close_resid']}")
        for st in sv["stages"]:
            print(f"     kappa={st['kappa']:6.1f} updates={st['updates']:3d} "
                  f"vel={st['vel_per_update']:+.4f}/upd "
                  f"mode {st['mode_start']:.2f}->{st['mode_end']:.2f}")
        if sv["implied_hold_to_close_resid"] is not None:
            holds.append(sv["implied_hold_to_close_resid"])
    if holds:
        print(f"  [A3 signal] implied stage_hold to close residual at fastest stage: "
              f"{[f'{h:.0f}' for h in holds]} (max {max(holds):.0f})")

    dump = {"params": {"k": K, "w_h": W_H, "w_l": W_L, "q": Q, "e_star": E_STAR,
                       "bounds": list(BOUNDS)},
            "A1_deterministic_curve": det,
            "A2_stochastic_backtest": stoch,
            "A3_window_calibration": win}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "docs", "tasks", "claim-a-dev-trigger-retrain",
                            "phase01_screen.json")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\n[dump] {out_path}")
    print("=" * 92)


if __name__ == "__main__":
    main()
