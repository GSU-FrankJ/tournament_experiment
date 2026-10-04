#!/usr/bin/env python3
"""Read-only diagnostic of ONE failed run of the v2.0 locked confirmation (T=2 tournament, PPO).

Default target: q=50, seed 30510 (gate G-A failed: eta_2/DW = 0.0058 > 0.005; stage-2 peak error
-0.1458 relative). The tool only READS the run directories and writes under --out.

What it does (see report_draft.md written to --out):
  1. Peak trajectory: e2_hat(0) and the location-free peak (max over the recovery grid of the stage-2
     Beta mean) from EVERY weight export of Phase A (updates 25..1600), target vs the 10-90% band and
     median of the other seeds of the same q (and the 20 q=60 runs).
  2. End-of-A profile: e2_hat(d)-e2*(d), Delta_2(d) with the eta_2 argmax, on/off-path split,
     symmetry error, sigma_2(d), target vs the two other seeds nearest to the median peak error.
  3. Optimisation: KL, clip fraction, actor grad norm, advantage SD, concentration at d=0, dev-tier eta_2
     at every verifier call, the 'would_have_fired' record.
  4. Sampling: cumulative visitation of the stage-2 bins (peak-set share) and the D1 clamp counts.
  5. Decomposition of the d=0 peak gap: RL gap, smoothing-predicted gap, supervised floor.
  6. The PRE-REGISTERED hypothesis rules H1-H4, applied literally (function ``apply_rules``; the rule
     text is in ``RULE_TEXT``). Every verdict word of the report (sections 0 and 7) is produced from the
     computed booleans (numbers.json keys ``rules.*``; tables/hypothesis_rules.csv). The rules are applied
     to the target and, for reference, to each of the 20 q=50 runs (target against the other 19) and to
     each of the 20 q=60 runs (against the other 19 q=60 runs). Alternative readings the author
     considered are in a separate subsection 'Post hoc readings (not pre-registered)'.

Definitions are the repo's (utils/v2_metrics.py recovery_metrics, run/run_v2_T2_locked.py
smoothed_share / stage2_extra, tools/v2/confirmation_analysis_v2_0.py). The network forward pass is
re-implemented (agents/ppo_curriculum.py mean_effort_numpy) and checked against the arrays the runs
saved. The closed form (g2) is used for EVALUATION metrics only. The optional --repo flag imports
utils.dp_br_verifier (read-only, bytecode writing disabled) to recompute the dev-tier eta_2 at every
weight export; without it that one series is skipped.

Writes only under --out (report_draft.md, numbers.json, tables/) and --fig-dir (default <out>/figures).
The report refers to figures as figures/figN.png (it is meant to sit next to the figure directory).

Usage:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python diag_seed30510.py \
      --out <dir> [--fig-dir <dir>] [--run-root <.../results/v2_T2_locked/confirmation_v2_0>] \
      [--repo <worktree>] [--floor-dir <.../v2_pilots/pilot4/analysis/repr_floor>]
Defaults: --run-root and --floor-dir in the v2-t2-refine worktree; --repo = the worktree this file is in
(parents[2]) if it has utils/dp_br_verifier.py, else the v2-t2-refine worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True   # never leave .pyc files next to read-only code

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.titlesize"] = 9.5
matplotlib.rcParams["axes.labelsize"] = 9
matplotlib.rcParams["legend.fontsize"] = 8
import matplotlib.pyplot as plt

EXPORT_US = list(range(25, 1601, 25))          # Phase-A weight exports (64)
LR_DECAY_FIRST = 1201                          # Phase-A LR decay window: updates 1201..1600
PRE_WIN = (900, 1200)                          # late constant-LR window (exports)
DEC_WIN = (1225, 1600)                         # LR-decay window (exports)
LATE_WIN = (900, 1600)
BAND_LO, BAND_HI = 10.0, 90.0
SMOOTH_NODES = 400                             # run/run_v2_T2_locked.py SMOOTH_NODES
PEAK_HALF = 20.0                               # peak set: bins intersecting (-20, 20)
BIN_W = 10.0
C_ORANGE, C_BLUE, C_AQUA, C_VIOLET = "#eb6834", "#2a78d6", "#1baf7a", "#4a3aa7"
C_BAND, C_GREY, C_INK, C_INK2 = "#b7d3f6", "#9a9a96", "#0b0b0b", "#52514e"
DEFAULT_REFINE = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine")
DEFAULT_RUN_ROOT = DEFAULT_REFINE / "results" / "v2_T2_locked" / "confirmation_v2_0"
DEFAULT_FLOOR_DIR = DEFAULT_REFINE / "results" / "v2_pilots" / "pilot4" / "analysis" / "repr_floor"

# Pre-registered hypothesis rules (written before any diagnostic output existed; applied LITERALLY).
# S(u) = (e2_hat(0; u) - e2*(0)) / e2*(0)  (signed peak error at d=0, weight export u = 25, ..., 1600)
# L(u) = (max_d e2_hat(d; u) - e2*(0)) / e2*(0)  (location-free peak error, recovery grid;
#        tools/v2/pilot4_common.location_free). Pack = the 19 other runs of the same q block.
# p10_pack(.,u), p90_pack(.,u): numpy.percentile (linear) over the pack at the same export.
# out_S(u) := S_target(u) < p10_pack(S, u).  u_leave := the first export from which out_S holds at that
# export and at every later export up to 1600 (undefined if out_S(1600) is false). Resolution 25 updates.
RULE_TEXT = {
    "H1": "peak never rose to the pack's level: max over ALL exports u of S_target(u) < p10_pack(S, 1600)",
    "H2": "rose, then decayed in the LR-decay window: not H1; out_S(1600); u_leave >= 1225",
    "H3": "late excursion after a stable plateau: not H1; out_S(1600); u_leave <= 1200; and the target is NOT out_S at "
          ">= 16 consecutive exports ending at u_leave - 25",
    "H4": "plateau at the pack's level of peak HEIGHT with an unusually rounded cusp: out_S(1600); L_target(u) >= "
          "p10_pack(L, u) at every export u in 1225..1600; and (L - S)_target(1600) > p90_pack(L - S, 1600)",
    "NONE": "if out_S(1600), not H1, u_leave <= 1200 and no 400-update (16-export) plateau: 'none of H1-H4 (early or "
            "unstable departure)'",
}
PLATEAU_EXPORTS = 16                           # 400 updates / 25
H2_FIRST_U = 1225
H3_LAST_LEAVE_U = 1200
H4_WIN = (1225, 1600)


# ------------------------------------------------------------------------------------------ helpers
def read_json(p: Path) -> dict:
    """Load a JSON file."""
    with open(p) as f:
        return json.load(f)


def sha256(p: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


class Numbers:
    """Registry of every number quoted in the report (value + where it came from)."""

    def __init__(self) -> None:
        self.values, self.sources = {}, {}

    def put(self, key: str, value, source: str):
        """Store a number (numpy scalars converted) and its source; returns the value."""
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        self.values[key] = value
        self.sources[key] = source
        return value

    def dump(self, path: Path) -> None:
        """Write numbers.json."""
        with open(path, "w") as f:
            json.dump({"values": self.values, "sources": self.sources}, f, indent=1, default=str)


def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Mean of the last ``w`` entries ending at each index (NaN before w entries exist)."""
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def band(M: np.ndarray, others: np.ndarray):
    """Median, 10th and 90th percentile (linear) over the rows ``others`` of M (seeds x exports)."""
    X = M[others]
    return np.median(X, 0), np.percentile(X, BAND_LO, 0), np.percentile(X, BAND_HI, 0)


def rank_lowest(M: np.ndarray, i: int) -> np.ndarray:
    """Rank of row i at each column among all rows (1 = lowest value)."""
    return (M < M[i][None, :]).sum(0) + 1


def trailing_true(flag: np.ndarray) -> int:
    """Length of the final run of True in a boolean vector."""
    n = 0
    for v in flag[::-1]:
        if not v:
            break
        n += 1
    return n


def verdict_word(value) -> str:
    """'supported' / 'not supported' / 'undetermined' from a rule boolean (None = an input is missing)."""
    if value is None:
        return "undetermined"
    return "supported" if value else "not supported"


def apply_rules(us, s_t, l_t, s_pack, l_pack) -> dict:
    """The pre-registered H1-H4 rules, applied literally (see RULE_TEXT), with all intermediate quantities.

    Args:
        us: Export updates (25, 50, ..., 1600).
        s_t: Target signed peak error S(u) at every export.
        l_t: Target location-free peak error L(u) at every export.
        s_pack: Pack S values, shape (n_pack, n_exports).
        l_pack: Pack L values, shape (n_pack, n_exports).

    Returns:
        Dict with the intermediate quantities, the rule booleans (None when an input is missing), the
        verdict words and the label. Nothing is tuned: thresholds are the literal ones of RULE_TEXT.
    """
    us = np.asarray(us)
    res = {"inputs_ok": True}
    arrs = [np.asarray(x, dtype=float) for x in (s_t, l_t, s_pack, l_pack)]
    n = len(us)
    ok = (arrs[0].shape == (n,) and arrs[1].shape == (n,) and arrs[2].ndim == 2 and arrs[2].shape[1] == n
          and arrs[3].shape == arrs[2].shape and arrs[2].shape[0] >= 2 and 1600 in list(us)
          and all(np.all(np.isfinite(x)) for x in arrs))
    if not ok:
        res.update(inputs_ok=False, H1=None, H2=None, H3=None, H4=None, out_S_1600=None, u_leave=None, plateau_len=None,
                   S_max_all=None, u_of_S_max=None, p10_S_1600=None, n_pack=None, n_out_exports=None, n_in_band_exports=None,
                   L_ge_p10_all_1225_1600=None, n_L_below_p10_1225_1600=None, min_L_margin_1225_1600=None,
                   cusp_rounding_1600=None, p90_cusp_1600=None, cusp_exceeds_p90=None, S_1600=None, L_1600=None,
                   label="undetermined (an input is missing)")
        for h in ("H1", "H2", "H3", "H4"):
            res["verdict_" + h] = "undetermined"
        return res
    s_t, l_t, s_pack, l_pack = arrs
    i1600 = int(np.nonzero(us == 1600)[0][0])
    p10_s = np.percentile(s_pack, 10, axis=0)
    p10_l = np.percentile(l_pack, 10, axis=0)
    p90_cusp = np.percentile(l_pack - s_pack, 90, axis=0)
    out_s = s_t < p10_s
    out1600 = bool(out_s[i1600])
    j_max = int(np.argmax(s_t))
    h1 = bool(float(s_t.max()) < float(p10_s[i1600]))
    if out1600:
        n_trail = trailing_true(out_s)
        i_leave = n - n_trail
        u_leave = int(us[i_leave])
        k, run = i_leave - 1, 0          # exports ending at u_leave - 25, counted backwards
        while k >= 0 and not out_s[k]:
            run += 1
            k -= 1
        plateau = run
    else:
        u_leave, plateau = None, None
    h2 = bool((not h1) and out1600 and u_leave >= H2_FIRST_U) if out1600 else False
    h3 = bool((not h1) and out1600 and u_leave <= H3_LAST_LEAVE_U and plateau >= PLATEAU_EXPORTS) if out1600 else False
    win = (us >= H4_WIN[0]) & (us <= H4_WIN[1])
    l_ge = bool(np.all(l_t[win] >= p10_l[win]))
    cusp_t = float(l_t[i1600] - s_t[i1600])
    cusp_ex = bool(cusp_t > float(p90_cusp[i1600]))
    h4 = bool(out1600 and l_ge and cusp_ex)
    sup = [h for h, v in (("H1", h1), ("H2", h2), ("H3", h3), ("H4", h4)) if v]
    if sup:
        label = "supported: " + ", ".join(sup)
    elif out1600 and (not h1) and u_leave <= H3_LAST_LEAVE_U and plateau < PLATEAU_EXPORTS:
        label = "none of H1-H4 (early or unstable departure)"
    elif not out1600:
        label = "none of H1-H4 (not out of the pack at u=1600)"
    else:
        label = "none of H1-H4 (other)"
    res.update({"H1": h1, "H2": h2, "H3": h3, "H4": h4, "out_S_1600": out1600, "u_leave": u_leave, "plateau_len": plateau,
                "S_max_all": float(s_t.max()), "u_of_S_max": int(us[j_max]), "p10_S_1600": float(p10_s[i1600]), "n_pack": int(s_pack.shape[0]),
                "n_out_exports": int(out_s.sum()), "n_in_band_exports": int((~out_s).sum()),
                "L_ge_p10_all_1225_1600": l_ge, "n_L_below_p10_1225_1600": int((l_t[win] < p10_l[win]).sum()),
                "n_exports_1225_1600": int(win.sum()), "min_L_margin_1225_1600": float((l_t[win] - p10_l[win]).min()),
                "cusp_rounding_1600": cusp_t, "p90_cusp_1600": float(p90_cusp[i1600]), "cusp_exceeds_p90": cusp_ex,
                "S_1600": float(s_t[i1600]), "L_1600": float(l_t[i1600]), "label": label,
                "out_S_series": [bool(v) for v in out_s]})
    for h in ("H1", "H2", "H3", "H4"):
        res["verdict_" + h] = verdict_word(res[h])
    return res


def first_run(flag: np.ndarray, length: int) -> int:
    """Index where the first run of >= ``length`` consecutive True starts (-1 if none)."""
    n = 0
    for i, v in enumerate(flag):
        n = n + 1 if v else 0
        if n >= length:
            return i - length + 1
    return -1


# ------------------------------------------------------------------------------------------ model
def actor_ab(W: dict, obs: np.ndarray):
    """Float32 actor forward pass (agents/ppo_curriculum.py mean_effort_numpy), returns (alpha, beta)."""
    x = np.asarray(obs, dtype=np.float32)
    h = np.tanh(x @ W["actor.l1.weight"].T + W["actor.l1.bias"])
    h = np.tanh(h @ W["actor.l2.weight"].T + W["actor.l2.bias"])
    z = h @ W["actor.out.weight"].T + W["actor.out.bias"]
    mu = np.clip(1.0 / (1.0 + np.exp(-z[:, 0])), 1e-6, 1.0 - 1e-6).astype(np.float32)
    zc = z[:, 1].astype(np.float32)
    c = (np.float32(100.0) + np.logaddexp(np.float32(0.0), zc)).astype(np.float32)
    return (mu * c).astype(np.float32), ((np.float32(1.0) - mu) * c).astype(np.float32)


def obs_stage(t: int, d: np.ndarray, B: float) -> np.ndarray:
    """GameSpec.encode_obs for T=2: [(t-1)/(T-1), d/((t-1)B)] (float64 -> float32)."""
    d = np.asarray(d, float).reshape(-1)
    out = np.empty((d.size, 2), np.float32)
    out[:, 0] = np.float32(0.0 if t <= 1 else 1.0)
    out[:, 1] = (np.zeros_like(d) if t <= 1 else d / ((t - 1) * B)).astype(np.float32)
    return out


def f_xi(x, q):
    """Density of the stage shock difference, Triangular(-2q, 2q) (utils/theory_multistage.py)."""
    x = np.asarray(x, float)
    return np.where(np.abs(x) <= 2.0 * q, (2.0 * q - np.abs(x)) / (4.0 * q * q), 0.0)


def g2_closed(d, spec) -> np.ndarray:
    """Closed-form stage-2 equilibrium effort DW f_xi(d)/(2k), clipped to [0, e_bar] (EVALUATION only)."""
    return np.clip(spec["dw"] * f_xi(d, spec["q"]) / (2.0 * spec["k"]), 0.0, 100.0)


def recovery_grid(q: float) -> np.ndarray:
    """Recovery grid on D_2 = [-B, B], step 0.5, exact 0 node (utils/v2_metrics.symmetric_grid)."""
    B = 100.0 + 2.0 * q
    n_half = int(np.ceil(B / 0.5 - 1e-9))
    return np.linspace(-B, B, 2 * n_half + 1)


def smoothed_pred0(a0: float, b0: float, spec) -> float:
    """Smoothed-game prediction of e2(0) from the policy's own d=0 noise (run_v2_T2_locked.smoothed_share)."""
    from scipy.stats import beta as beta_dist
    u = (np.arange(SMOOTH_NODES) + 0.5) / SMOOTH_NODES
    x = 100.0 * (beta_dist.ppf(u, a0, b0) - a0 / (a0 + b0))
    return float(spec["dw"] / (2.0 * spec["k"]) * f_xi(x[:, None] - x[None, :], spec["q"]).mean())


def stage2_metrics(W: dict, spec: dict, D: np.ndarray, z0: int, g2: np.ndarray) -> dict:
    """All per-export stage-2 metrics of one weight file (definitions of utils/v2_metrics.recovery_metrics)."""
    q, B, g20 = spec["q"], spec["B"], spec["g20"]
    a, b = actor_ab(W, obs_stage(2, D, B))
    a64, b64 = a.astype(float), b.astype(float)
    e = 100.0 * a64 / (a64 + b64)
    pos = np.abs(D) < 2.0 * q
    tail = ~pos
    j = int(np.argmax(e))
    sym = np.abs(e - e[::-1])
    c0 = a64[z0] + b64[z0]
    sig0 = 100.0 * float(np.sqrt(a64[z0] * b64[z0] / (c0 * c0 * (c0 + 1.0))))
    pred0 = smoothed_pred0(a64[z0], b64[z0], spec)
    m = {"e2_0": float(e[z0]), "peak_rel_err": float((e[z0] - g20) / g20),
         "locfree_rel_err": float((e[j] - g20) / g20), "locfree_argmax_d": float(D[j]),
         "rmse_pos_over_g20": float(np.sqrt(np.mean((e[pos] - g2[pos]) ** 2)) / g20),
         "tail_mean_over_g20": float(e[tail].mean() / g20), "tail_max": float(e[tail].max()),
         "sym_err_max": float(sym.max()), "alpha0": float(a64[z0]), "beta0": float(b64[z0]),
         "conc0": float(c0), "sigma0": sig0, "smooth_pred0": pred0,
         "smooth_share": float((g20 - pred0) / (g20 - e[z0])) if g20 != e[z0] else float("nan")}
    for dd in (10.0, 20.0, 40.0):   # shoulder: mean of e(+dd), e(-dd) against e*(dd)
        ip, im = int(np.argmin(np.abs(D - dd))), int(np.argmin(np.abs(D + dd)))
        gd = float(g2[ip])
        m[f"shoulder{int(dd)}_rel_err"] = float((0.5 * (e[ip] + e[im]) - gd) / gd)
    a1, b1 = actor_ab(W, obs_stage(1, np.zeros(1), B))
    m["e1_0"] = float(100.0 * float(a1[0]) / (float(a1[0]) + float(b1[0])))
    return m, e


# ------------------------------------------------------------------------------------------ loading
def load_run(root: Path, q: int, seed: int, verify_fn) -> SimpleNamespace:
    """Everything one run directory holds that the diagnostic needs (read-only)."""
    d = root / f"q{q}" / f"seed{seed}"
    r = SimpleNamespace(q=q, seed=seed, dir=d)
    r.status = read_json(d / "status.json")
    r.cfg = read_json(d / "run_config.json")
    g = r.cfg["record"]["game"]
    q_ = float(g["q"])
    r.spec = {"w_h": float(g["w_h"]), "w_l": float(g["w_l"]), "k": float(g["k"]), "q": q_,
              "dw": float(g["w_h"]) - float(g["w_l"]), "B": 100.0 + 2.0 * q_}
    r.spec["g20"] = float(g2_closed(np.zeros(1), r.spec)[0])
    r.gates = read_json(d / "gates.json")
    r.summ = read_json(d / "v2_run_summary.json")
    r.ckpt = pd.read_csv(d / "v2_checkpoints_A.csv")
    upd = pd.read_csv(d / "v2_updates.csv")
    r.upd = upd[upd["phase"] == "A"].reset_index(drop=True)
    th = read_json(d / "train_history.json")
    hist = pd.DataFrame([h for h in th["history"] if h["phase"] == "A"])
    cb = np.array(hist["conc_buffer_min_mean_max"].tolist(), float)
    hist["conc_buf_min"], hist["conc_buf_mean"], hist["conc_buf_max"] = cb[:, 0], cb[:, 1], cb[:, 2]
    hist["eff_stage2_batch_mean"] = [h["mean_effort_by_stage"]["2"] for h in th["history"] if h["phase"] == "A"]
    r.hist = hist
    r.vis = [{"update": c["update"], "counts": np.asarray(c["visitation_cumulative_phase"]["stage2_direct_es"], float),
              "criterion": c["criterion_value_over_dw"], "eligible": c["eligible"]}
             for c in th["verifier_calls"] if c["phase"] == "A"]
    r.stability_A = [s for s in th["stability"] if s["phase"] == "A"]
    D = recovery_grid(q_)
    z0 = int(np.nonzero(D == 0.0)[0][0])
    g2 = g2_closed(D, r.spec)
    rows, e_at = [], {}
    for u in EXPORT_US:
        z = np.load(d / "weights" / f"u{u:05d}.npz")
        W = {k: z[k] for k in z.files}
        m, e = stage2_metrics(W, r.spec, D, z0, g2)
        m.update({"q": q, "seed": seed, "u": u})
        if verify_fn is not None:
            m.update(verify_fn(W, r.spec))
        rows.append(m)
        if u in (400, 800, 1200, 1600):
            e_at[u] = e
    r.exports = pd.DataFrame(rows)
    r.e_at = e_at
    r.D, r.z0, r.g2 = D, z0, g2
    r.gA = np.load(d / "gateA_final.npz")
    return r


def make_verify_fn(repo: Path):
    """Closure W, spec -> dev-tier eta_2 etc. via utils.dp_br_verifier (read-only import), or None."""
    sys.path.insert(0, str(repo))
    try:
        from utils.dp_br_verifier import DEV_CONFIG, verify
    except Exception as exc:   # pragma: no cover
        print(f"[warn] cannot import verifier from {repo}: {exc}; eta_2 per export skipped")
        return None

    def fn(W: dict, spec: dict) -> dict:
        def mean_fn(t, d):
            a, b = actor_ab(W, obs_stage(t, d, spec["B"]))
            return 100.0 * a.astype(float) / (a.astype(float) + b.astype(float))
        res = verify(mean_fn, w_h=spec["w_h"], w_l=spec["w_l"], k=spec["k"], q=spec["q"], T=2,
                     e_min=0.0, e_max=100.0, cfg=DEV_CONFIG)
        s2, s1 = res.stages[2], res.stages[1]
        drift = float(s1.e_hat[0] - s1.e_opp[0])
        dl = s2.delta / res.dw
        on = np.abs(s2.d_grid - drift) < 2.0 * spec["q"]
        jon = int(np.argmax(np.where(on, dl, -np.inf)))
        joff = int(np.argmax(np.where(~on, dl, -np.inf)))
        return {"v_eta2": float(res.full_delta_max[2] / res.dw), "v_on_max": float(dl[jon]),
                "v_on_argmax_d": float(s2.d_grid[jon]), "v_off_max": float(dl[joff]),
                "v_off_argmax_d": float(s2.d_grid[joff]), "v_valid": bool(res.valid)}
    return fn


# ------------------------------------------------------------------------------------------ analysis
def matrix(runs: list, col: str) -> np.ndarray:
    """Seeds x exports matrix of an exports column, rows in the order of ``runs``."""
    return np.stack([r.exports[col].to_numpy(float) for r in runs])


def exit_stats(z: np.ndarray, p10: np.ndarray, us: np.ndarray) -> dict:
    """Band-exit descriptors of one series against a lower band edge (rules stated in the report)."""
    below = z < p10
    i_trail = trailing_true(below)
    i_run8 = first_run(below, 8)
    in_band = np.nonzero(~below)[0]
    late = (us >= LATE_WIN[0]) & (us <= LATE_WIN[1])
    return {"n_exports": int(len(us)), "n_below": int(below.sum()), "frac_below": float(below.mean()),
            "first_below_u": int(us[np.argmax(below)]) if below.any() else None,
            "last_in_band_u": int(us[in_band.max()]) if in_band.size else None,
            "sustained_exit_u": int(us[len(us) - i_trail]) if i_trail > 0 else None,
            "run8_start_u": int(us[i_run8]) if i_run8 >= 0 else None,
            "n_in_band": int((~below).sum()), "in_band_us": [int(u) for u in us[~below]],
            "frac_below_late": float(below[late].mean()),
            "frac_below_const_400_1200": float(below[(us >= 400) & (us <= 1200)].mean()),
            "frac_below_decay": float(below[(us >= DEC_WIN[0]) & (us <= DEC_WIN[1])].mean())}


def per_seed_stats(runs: list, col: str = "peak_rel_err") -> pd.DataFrame:
    """Per-seed trajectory statistics (late level, windows, trend, volatility, time to reach -0.10)."""
    us = np.array(EXPORT_US)
    M = matrix(runs, col)
    rows = []
    for i, r in enumerate(runs):
        z = M[i]
        oth = np.array([j for j in range(len(runs)) if j != i])
        _, p10, _ = band(M, oth)
        late = (us >= LATE_WIN[0]) & (us <= LATE_WIN[1])
        pre = (us >= PRE_WIN[0]) & (us <= PRE_WIN[1])
        dec = (us >= DEC_WIN[0]) & (us <= DEC_WIN[1])
        slope = float(np.polyfit(us[late], z[late], 1)[0] * 100.0)
        ok = np.nonzero(z >= -0.10)[0]
        rows.append({"q": r.q, "seed": r.seed, "late_mean": float(z[late].mean()), "late_median": float(np.median(z[late])),
                     "late_max": float(z[late].max()), "late_min": float(z[late].min()),
                     "late_sd": float(z[late].std(ddof=1)), "pre_mean": float(z[pre].mean()), "dec_mean": float(z[dec].mean()),
                     "dec_minus_pre": float(z[dec].mean() - z[pre].mean()), "late_slope_per100": slope,
                     "first_u_ge_m010": int(us[ok[0]]) if ok.size else None,
                     "loo_frac_below_p10_late": float((z < p10)[late].mean())})
    return pd.DataFrame(rows)


def contrast_seeds(runs50: list, target: int, n: int) -> list:
    """The n other seeds whose end-of-A signed peak error is nearest to the median of the others."""
    vals = {r.seed: r.gates["reported"]["end_of_A"]["final"]["stage2_peak_rel_err_signed"] for r in runs50 if r.seed != target}
    med = float(np.median(list(vals.values())))
    order = sorted(vals, key=lambda s: (abs(vals[s] - med), s))
    return order[:n], med


def window_table(runs: list, series: list, width: int = 100) -> pd.DataFrame:
    """Window means (width updates) of per-update series, per run, long format."""
    rows = []
    for r in runs:
        for name, col, frame in series:
            x = (r.upd if frame == "upd" else r.hist)[col].to_numpy(float)
            for k in range(len(x) // width):
                rows.append({"q": r.q, "seed": r.seed, "series": name, "w_first": k * width + 1,
                             "w_last": (k + 1) * width, "mean": float(x[k * width:(k + 1) * width].mean())})
    return pd.DataFrame(rows)


def peak_bins(q: float) -> tuple:
    """Bin edges of D_2 (width 10) and the indices of the peak-set bins (interval intersects (-20, 20))."""
    half = 100.0 + 2.0 * q
    nb = int(np.ceil(2.0 * half / BIN_W - 1e-9))
    edges = np.linspace(-half, half, nb + 1)
    ids = np.nonzero((edges[:-1] < PEAK_HALF) & (edges[1:] > -PEAK_HALF))[0]
    return edges, ids


# ------------------------------------------------------------------------------------------ figures
def style(ax, grid=True):
    """Light axes: no top/right spines, recessive grid."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors=C_INK2, length=3)
    if grid:
        ax.grid(True, color="#e6e5e1", lw=0.6)
        ax.set_axisbelow(True)


def save(fig, figdir: Path, name: str):
    """Write pdf (fonts type 42) and png."""
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def traj_panel(ax, us, M, tgt_i, others, label_tgt, zoom=None, ylab=None, extra=None, others_label="other seeds", zero=True):
    """Band + median of ``others`` rows, thin lines for them, the target series on top."""
    med, p10, p90 = band(M, others)
    for j in others:
        ax.plot(us, M[j], color=C_GREY, lw=0.5, alpha=0.35, zorder=1)
    ax.fill_between(us, p10, p90, color=C_BAND, alpha=0.75, lw=0, zorder=2, label=f"{int(BAND_LO)}-{int(BAND_HI)}% band, {others_label}")
    ax.plot(us, med, color=C_BLUE, lw=1.8, zorder=3, label=f"median, {others_label}")
    if extra is not None:
        ax.plot(us, extra[0], color=C_INK2, lw=1.2, ls="--", zorder=3, label=extra[1])
    if tgt_i is not None:
        ax.plot(us, M[tgt_i], color=C_ORANGE, lw=1.8, marker="o", ms=2.6, zorder=4, label=label_tgt)
    ax.axvline(LR_DECAY_FIRST - 0.5, color=C_INK2, lw=0.8, ls=":")
    if zero:
        ax.axhline(0.0, color="#8a8984", lw=0.6)
    if zoom:
        ax.set_xlim(zoom[0], 1610)
        ax.set_ylim(zoom[1], zoom[2])
    else:
        ax.set_xlim(0, 1625)
    ax.set_xlabel("update (Phase A)")
    if ylab:
        ax.set_ylabel(ylab)
    style(ax)


def build_report(N: Numbers, a, root: Path, ana: Path, out: Path, figdir: Path, have_verifier: bool) -> str:
    """report_draft.md: every number is read from the Numbers registry (numbers.json)."""
    V = N.values
    TQ, TS = a.q, a.seed
    rd = f"{root}/q{TQ}/seed{TS}"

    def f(key, spec=".4f"):
        v = V[key]
        if v is not None and spec.endswith("d") and isinstance(v, float):
            v = int(round(v))
        return "n/a" if v is None else format(v, spec)

    def sg(key, spec="+.4f"):
        return f(key, spec)

    def table(header, rows):
        out_ = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        out_ += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        return "\n".join(out_)

    def stat_row(label, base, spec=".4f", signed=False):
        s = "+" + spec if signed else spec
        return [label, f(f"{base}.target", s), f(f"{base}.others_median", s), f"{f(f'{base}.others_p10', s)} .. {f(f'{base}.others_p90', s)}",
                f"{f(f'{base}.others_min', s)} .. {f(f'{base}.others_max', s)}", f(f"{base}.rank_lowest_of_20", "d")]

    def inr(base):
        """'inside' / 'outside' the range of the other seeds' values (min..max) for a registered target value."""
        t_, lo_, hi_ = V[f"{base}.target"], V[f"{base}.others_min"], V[f"{base}.others_max"]
        return "inside" if lo_ <= t_ <= hi_ else "outside"

    def le_max(t_key, max_key):
        """True when the target value does not exceed the others' maximum."""
        return V[t_key] <= V[max_key]

    seeds_prof = V["contrast.seeds"]
    c1, c2 = seeds_prof[1], seeds_prof[2]
    L = []
    P = L.append
    P(f"# Diagnostic draft: q={TQ} seed {TS}, v2.0 confirmation (read-only, descriptive)")
    P("")
    P("Draft for the PI. Produced by `diag_seed30510.py` from files that were only read. No run was repeated, no threshold or "
      "protocol value was changed, no training was started. Every number below is stored with its source in `numbers.json`; "
      "tables carry a source line. Association is not cause: where this draft compares seeds it says so.")
    P("")
    P("## 0. Summary")
    P("")
    P(f"- Failure (reproduced from the files): eta_2/DW = {f('owner.eta2_over_dw', '.5f')} against the G-A threshold {f('owner.eta2_threshold', '.3f')}; "
      f"stage-2 peak error {sg('owner.peak_err_signed')}; eta_2 maximum on-path ({f('owner.on_max', '.5f')}) against off-path ({f('owner.off_max', '.6f')}); "
      f"smoothed-game share {f('owner.smoothed_share', '.3f')}. Match with the four numbers quoted in the request (to the digits quoted): "
      f"{'yes, all four' if V['owner.match.all'] else 'NO, at least one differs (section 1)'}.")
    P(f"- Peak error of seed {TS}: {sg('end.peak_rel_err.target')}. The other 19 q={TQ} seeds: median {sg('end.peak_rel_err.others_median')}, "
      f"10th percentile {sg('end.peak_rel_err.others_p10')}, 90th percentile {sg('end.peak_rel_err.others_p90')} (range {sg('end.peak_rel_err.others_min')} .. {sg('end.peak_rel_err.others_max')}). "
      f"Rank of seed {TS} from the lowest: {f('end.peak_rel_err.rank_lowest_of_20', 'd')} of 20 q={TQ} runs, {f('end.pooled40.peak_err_target_rank_lowest_of_40', 'd')} of {f('end.n_runs_pooled', 'd')} runs; the lowest of the other 39 is "
      f"q={f('end.pooled40.next_lowest_q', 'd')} seed {f('end.pooled40.next_lowest_seed', 'd')} at {sg('end.pooled40.next_lowest_peak_err')} (its G-A verdict: {'pass' if V['end.pooled40.next_lowest_G_A_pass'] else 'fail'}).")
    P(f"- Pre-registered departure update: u_leave = {f('rules.target.u_leave', 'd')} (the first export from which S_target(u) < p10_pack(S, u) holds at that export and at every later one; resolution 25 updates). "
      f"The LR-decay window starts at update {LR_DECAY_FIRST}; u_leave >= {H2_FIRST_U}: {V['rules.target.u_leave'] is not None and V['rules.target.u_leave'] >= H2_FIRST_U}. "
      f"Descriptive facts on the same series: last in-band export {f('traj.e2_0.last_in_band_u', 'd')}; below the band at {f('traj.e2_0.n_below', 'd')} of {f('traj.e2_0.n_exports', 'd')} exports in total.")
    P(f"- Pre-registered rules H1-H4 applied literally (section 7; verdict words are produced from computed booleans, `numbers.json` keys `rules.target.*`): "
      f"H1 {V['rules.target.verdict_H1']}, H2 {V['rules.target.verdict_H2']}, H3 {V['rules.target.verdict_H3']}, H4 {V['rules.target.verdict_H4']}. Classification label: **{V['rules.target.label']}**. "
      f"Inputs: max over all exports of S = {sg('rules.target.S_max_all')} at u={f('rules.target.u_of_S_max', 'd')} against p10_pack(S, 1600) = {sg('rules.target.p10_S_1600')}; out_S(1600) = {V['rules.target.out_S_1600']}; "
      f"u_leave = {f('rules.target.u_leave', 'd')}; consecutive exports not out of the band ending at u_leave - 25: {f('rules.target.plateau_len', 'd')} (the H3 plateau condition needs {PLATEAU_EXPORTS}).")
    P(f"- Early signature (descriptive): the concentration alpha+beta at d=0 leaves the other seeds' band at update {f('optexit.conc0.sustained_exit_u', 'd')} and never returns; "
      f"the actor gradient norm from update {f('optexit.grad_norm_actor_mean.sustained_exit_u', 'd')}, sigma_2(0) from {f('optexit.sigma0.sustained_exit_u', 'd')}. "
      f"Whether this is a cause or a co-symptom of the low peak cannot be decided from these files (section 8).")
    P("")
    P("### Six key numbers")
    P("")
    P(table(["#", "quantity", "value"], [
        [1, f"peak error of seed {TS} (signed, relative)", sg("end.peak_rel_err.target")],
        [2, "median of the other 19 q=50 seeds", sg("end.peak_rel_err.others_median")],
        [3, "10th percentile of the other 19", sg("end.peak_rel_err.others_p10")],
        [4, "90th percentile of the other 19", sg("end.peak_rel_err.others_p90")],
        [5, "u_leave: update from which 30510 is out of the pack's band (S < p10_pack(S, u)) at every later export (pre-registered definition)", f("rules.target.u_leave", "d")],
        [6, "pre-registered rules H1 / H2 / H3 / H4 and label", f"{V['rules.target.verdict_H1']} / {V['rules.target.verdict_H2']} / {V['rules.target.verdict_H3']} / {V['rules.target.verdict_H4']}; {V['rules.target.label']}"]]))
    P("")
    P("## Source legend")
    P("")
    P(f"All run files are under `{root}/q{{50,60}}/seed{{30501..30520}}/`.")
    P("")
    P(table(["tag", "file(s)", "content used"], [
        ["[G]", "`gates.json`", "`reported.end_of_A` (final-tier and development-tier metrics, smoothed game), `metric_values`, `G-A`"],
        ["[W]", "`weights/u00025.npz` .. `u01600.npz`", "actor weights every 25 updates (64 exports in Phase A); the stage-2 mean, alpha, beta are recomputed with the repo's float32 forward pass"],
        ["[A]", "`gateA_final.npz`", "end-of-A recovery grid (e2_hat, e2*) and final-tier verifier arrays (Delta_2, on-path mask, sigma_2, alpha, beta)"],
        ["[C]", "`v2_checkpoints_A.csv`", f"development-tier verifier calls in Phase A ({f('meta.n_calls_A_per_run_min', 'd')} per run; runs with another count: {V['meta.n_runs_with_extra_call']})"],
        ["[U]", "`v2_updates.csv`", "per-update KL, clip fraction, advantage SD, losses, D1 clamp counts (Phase-A rows)"],
        ["[H]", "`train_history.json`", "per-update actor grad norms, entropy, buffer concentration; `verifier_calls[A].visitation_cumulative_phase.stage2_direct_es`"],
        ["[S]", "`v2_run_summary.json`", "`would_have_fired`"],
        ["[R]", f"`{ana}/reported_metrics.csv`, `per_run.csv`", "the confirmation analysis tables (cross-check of the owner's numbers)"],
        ["[F]", "`results/v2_pilots/pilot4/analysis/repr_floor/fits.csv`, `three_way_peak_gap.csv` (v2-t2-refine worktree)", "supervised-fit floor of the actor (5 inits, pilot 4 section 1d) and pilot-4 reference gaps"],
        ["[V]", "this tool + `utils/dp_br_verifier.py` (read-only import)", "development-tier eta_2 recomputed on every weight export"]]))
    P("")
    P("## 1. Methods and reproduction of the owner's numbers")
    P("")
    P("**Methods.** For every one of the 40 runs (q in {50, 60} x seeds 30501-30520) the 64 Phase-A weight exports were read and the stage-2 actor "
      "re-evaluated in float32 exactly as `agents/ppo_curriculum.mean_effort_numpy` does (input [1, d/B], B = 100 + 2q; mean = 100 alpha/(alpha+beta)) on the recovery grid "
      f"(step 0.5, {4 * (100 + 2 * a.q) + 1} points for q={a.q}, {4 * (100 + 2 * a.q_other) + 1} for q={a.q_other}) as in `utils/v2_metrics.recovery_metrics`. e2_hat(0) is the Beta mean at d=0. The location-free peak is the maximum of the "
      "mean over that grid, as `run/run_v2_T2_locked.stage2_extra` defines it (relative error (max - e2*(0))/e2*(0)). The closed form e2*(d) = DW f_xi(d)/(2k) "
      f"(DW={f('meta.dw', '.0f')}, k={f('meta.k', '.6f')}, e2*(0)={f('meta.g2_at_0_q50', '.0f')} at q={a.q}; e2*(0)={f('meta.g2_at_0_q60', '.2f')} at q={a.q_other}) "
      "is used for evaluation only. The smoothed-game prediction is the repo's (`smoothed_share`: 400 Beta quantile nodes at d=0). End-of-A arrays come from the runs' own `gateA_final.npz`. "
      f"{'The development-tier eta_2 was recomputed on every export by importing the repo verifier read-only (bytecode writing disabled); ' if have_verifier else ''}"
      "nothing was written inside the repository or any results directory. "
      "'Others' means the 19 other q=50 seeds; the band is the 10th-90th percentile (numpy linear) of their values at the same update. 'Below the band' means below the 10th percentile "
      "(the errors are negative, so below = a larger error). Ratios and percentages written in the text are computed from the listed registry values. Percentiles of 19 values are coarse: any single seed is below the band at roughly 10-15% of exports by construction, and exports of a run are autocorrelated, "
      "so run-level comparisons below use the other seeds' own leave-one-out frequencies.")
    P("")
    P("**Reproduction.** Sources: [G], [W], [A], [R], [V].")
    P("")
    rep_rows = [
        ["peak error signed (final tier)", "-0.1458", sg("owner.peak_err_signed", "+.6f"), sg("repro.peak_err_u1600_mine", "+.6f"), f"{V['repro.peak_err_absdiff_vs_gates']:.1e}"],
        ["eta_2 on-path max, final tier [G] / dev tier [G] / dev tier recomputed [V]", "0.0058", f"{f('owner.on_max', '.7f')} / {f('owner.on_max_dev', '.7f')}",
         f("repro.on_max_u1600_dev_mine", ".7f") if have_verifier else "n/a", ""],
        ["eta_2 off-path max, final tier [G] / dev tier [G] / dev tier recomputed [V]", "0.00082", f"{f('owner.off_max', '.7f')} / {f('owner.off_max_dev', '.7f')}",
         f("repro.off_max_u1600_dev_mine", ".7f") if have_verifier else "n/a", ""],
        ["smoothed-game share of the d=0 gap", "0.279", f("owner.smoothed_share", ".6f"), f("repro.smooth_share_u1600_mine", ".6f"), f"{abs(V['owner.smoothed_share'] - V['repro.smooth_share_u1600_mine']):.1e}"],
        ["e2_hat(0) (effort units)", "n/a", f("owner.e2_at_0", ".5f"), f("repro.e2_0_u1600_mine", ".5f"), f"{abs(V['owner.e2_at_0'] - V['repro.e2_0_u1600_mine']):.1e}"],
        ["location-free peak error", "n/a", sg("owner.locfree_err", "+.6f"), sg("repro.locfree_u1600_mine", "+.6f"), f"{abs(V['owner.locfree_err'] - V['repro.locfree_u1600_mine']):.1e}"],
        ["sigma_2(0) (effort units)", "n/a", f("owner.sigma0", ".5f"), f("repro.sigma0_u1600_mine", ".5f"), f"{abs(V['owner.sigma0'] - V['repro.sigma0_u1600_mine']):.1e}"]]
    P("(The column 'owner's message' is quoted from the request, not read from a file.)")
    P("")
    P(table(["quantity", "owner's message", "gates.json [G]", "recomputed here", "abs diff"], rep_rows))
    P("")
    P(f"The same comparison over all 40 runs: largest |difference| in the signed peak error {V['repro.block_max_absdiff_peak_err']:.1e}, in the smoothed share "
      f"{V['repro.block_max_absdiff_smooth_share']:.1e}, in sigma_2(0) {V['repro.block_max_absdiff_sigma0']:.1e}; the recovery-grid mean from the u1600 weights differs from the run's own "
      f"`recovery_e2` by at most {V['repro.recovery_e2_maxabsdiff_u1600']:.1e} effort units (float32 rounding of numpy against torch)."
      + (f" The recomputed development-tier eta_2 matches `v2_checkpoints_A.csv` at every Phase-A call of every run to within {V['repro.eta2_dev_max_absdiff_vs_csv']:.1e}." if have_verifier else ""))
    P(f"Match with the four numbers quoted in the request (file value rounded to the digits quoted): peak error {V['owner.match.peak_err']}, on-path maximum {V['owner.match.eta_on']}, "
      f"off-path maximum {V['owner.match.eta_off']}, smoothed share {V['owner.match.share']}; all four: {V['owner.match.all']}. Final-tier and development-tier on-path maxima identical in the files: "
      f"{V['owner.on_max'] == V['owner.on_max_dev']}; off-path maxima final {V['owner.off_max']:.7f}, development {V['owner.off_max_dev']:.7f}"
      + (f"; the development-tier on-path maximum recomputed here equals the stored one to {abs(V['repro.on_max_u1600_dev_mine'] - V['owner.on_max_dev']):.1e}." if have_verifier else "."))
    P("")
    # ---- 2
    P("## 2. Peak trajectory (figures/fig1_peak_trajectory_q50.png, figures/fig2_peak_trajectory_q60_reference.png, figures/fig3b_profile_evolution_near_peak.png)")
    P("")
    P("### 2.1 Definitions used in this section")
    P("")
    P("For every export u in {25, ..., 1600}: z(u) = relative error of e2_hat(0) = S(u) (and, separately, of the location-free peak = L(u)) of seed 30510; p10(u), median(u), p90(u) over the 19 other q=50 seeds. "
      "Below the band: z(u) < p10(u) (the pre-registered out_S(u) for S). Reported: (a) first export below the band; (b) the first export of a run of at least 8 consecutive below-band exports (200 updates; not pre-registered); "
      "(c) the last in-band export; (d) u_leave, the pre-registered departure update: the earliest export u0 such that z(u) < p10(u) at every export u0, ..., 1600. "
      "In-band exports are listed so that the reader can apply another rule. The pre-registered rules H1-H4 are applied in section 7.")
    P("")
    P("### 2.2 Snapshot of e2_hat(0) (relative error)  [W]")
    P("")
    rows = []
    for u in (100, 250, 325, 400, 600, 800, 825, 850, 900, 1200, 1600):
        k = f"traj.e2_0.at_u{u}"
        rows.append([u, sg(k + ".target"), sg(k + ".others_median"), f"{sg(k + '.others_p10')} .. {sg(k + '.others_p90')}", f(k + ".rank", "d")])
    P(table(["update", f"seed {TS}", "others median", "others p10 .. p90", "rank of 30510 among 20 (1 = lowest)"], rows))
    P("")
    P("### 2.3 Band-exit statistics  [W]")
    P("")
    rows = []
    for lab, k in (("e2_hat(0)", "e2_0"), ("location-free peak", "locfree")):
        rows.append([lab, f"{f(f'traj.{k}.n_below', 'd')} / {f(f'traj.{k}.n_exports', 'd')}", f(f"traj.{k}.first_below_u", "d"), f(f"traj.{k}.run8_start_u", "d"),
                     f(f"traj.{k}.last_in_band_u", "d"), f(f"traj.{k}.sustained_exit_u", "d"), f"{f(f'traj.{k}.n_in_band_u400plus', 'd')} / {f(f'traj.{k}.n_exports_u400plus', 'd')}",
                     f"{f(f'traj.{k}.n_below_u900plus', 'd')} / {f(f'traj.{k}.n_exports_u900plus', 'd')}", f"{f(f'traj.{k}.n_rank1_late', 'd')} / {f(f'traj.{k}.n_exports_u900plus', 'd')}"])
    P(table(["series", "exports below band", "first below", "start of first run of >= 8", "last in band", "sustained departure", "in band at u >= 400",
             "below band at u >= 900", "rank 1 (lowest of 20) at u >= 900"], rows))
    P("")
    P(f"In-band exports of the e2_hat(0) series: {V['traj.e2_0.in_band_us']}. Location-free series: {V['traj.locfree.in_band_us']}.")
    P("")
    P("### 2.4 Trajectory facts (descriptive; any wording that compares values is the author's reading, not pre-registered)")
    P("")
    P(f"- Early values. At update 100: {sg('traj.e2_0.at_u100.target')} against a median of {sg('traj.e2_0.at_u100.others_median')} (p10 {sg('traj.e2_0.at_u100.others_p10')}, rank {f('traj.e2_0.at_u100.rank', 'd')} of 20 from the lowest). "
      f"At update 250: {sg('traj.e2_0.at_u250.target')} (rank {f('traj.e2_0.at_u250.rank', 'd')}; median {sg('traj.e2_0.at_u250.others_median')}); at 325: {sg('traj.e2_0.at_u325.target')} (rank {f('traj.e2_0.at_u325.rank', 'd')}).")
    P(f"- Later values. Pack median {sg('traj.e2_0.at_u400.others_median')} at 400, {sg('traj.e2_0.at_u600.others_median')} at 600, {sg('early.pack_median_median_u900plus')} as the median over exports from 900 on; "
      f"seed {TS}: {sg('traj.e2_0.at_u400.target')} at 400, {sg('traj.e2_0.at_u600.target')} at 600, {sg('traj.e2_0.at_u900.target')} at 900, {sg('traj.e2_0.at_u1200.target')} at 1200, {sg('traj.e2_0.at_u1600.target')} at 1600.")
    P(f"- First export at or above -0.10: update {f('seedstat.t_ge_m010.target', 'd')} for 30510, against a median of {f('seedstat.t_ge_m010.others_median', '.0f')} for the other seeds "
      f"(p10 {f('seedstat.t_ge_m010.others_p10', '.0f')}, p90 {f('seedstat.t_ge_m010.others_p90', '.0f')}, latest {f('seedstat.t_ge_m010.others_max', '.0f')}; all {f('seedstat.t_ge_m010.n_others_reaching', 'd')} others reach it). "
      f"That export (u=825) has S = {sg('traj.e2_0.at_u825.target')} (pack median {sg('traj.e2_0.at_u825.others_median')}, rank {f('traj.e2_0.at_u825.rank', 'd')}), between {sg('traj.e2_0.at_u800.target')} (u=800) and {sg('traj.e2_0.at_u850.target')} (u=850)."
      + (f" At that export the dev-tier eta_2/DW is {f('eta_exp.target_at_u825', '.5f')} (the others' median at u=825: {f('eta_exp.others_median_at_u825', '.5f')}); the largest eta_2/DW of 30510 from u=400 on is at u={f('eta_exp.target_argmax_u_u400plus', 'd')} ({f('eta_exp.target_max_u400plus', '.5f')})."
         if have_verifier else ""))
    P(f"- Late level (updates 900-1600): mean {sg('seedstat.late_mean.target')} for 30510 against a median of {sg('seedstat.late_mean.others_median')} for the other seeds' means "
      f"(range {sg('seedstat.late_mean.others_min')} .. {sg('seedstat.late_mean.others_max')}): {f('seedstat.n_others_late_mean_lt_target', 'd')} of the 19 others have a late mean at or below it. "
      f"Its best late export ({sg('early.target_best_late_val')} at u={f('early.target_best_late_u', 'd')}) is at the pooled 10th percentile of the others' late exports ({sg('pool.late_p10')}); only {f('pool.n_target_late_ge_pool_p10', 'd')} of its {f('traj.e2_0.n_exports_u900plus', 'd')} late exports reaches it.")
    P(f"- Leave-one-out comparison: the fraction of late exports below a seed's own leave-one-out p10 is {f('seedstat.loo_frac_below_p10_late.target', '.2f')} for 30510; for the other seeds the median is "
      f"{f('seedstat.loo_frac_below_p10_late.others_median', '.2f')} and the maximum {f('seedstat.loo_frac_below_p10_late.others_max', '.2f')}; {f('seedstat.n_others_loo_frac_below_p10_late_ge_0p5', 'd')} of them reach 0.5.")
    P(f"- Volatility: SD of the late series {f('seedstat.late_sd.target', '.4f')} (others' median {f('seedstat.late_sd.others_median', '.4f')}; target {inr('seedstat.late_sd')} the others' range); SD of consecutive-export changes "
      f"{f('hyp.step_sd_u900plus.target', '.4f')} (others' median {f('hyp.step_sd_u900plus.others_median', '.4f')}, maximum {f('hyp.step_sd_u900plus.others_max', '.4f')}; not above the others' maximum: {le_max('hyp.step_sd_u900plus.target', 'hyp.step_sd_u900plus.others_max')}); largest single step "
      f"{f('hyp.target_max_abs_step_per25_u900plus', '.4f')} (others' median of the largest step {f('hyp.others_max_abs_step_per25_u900plus_median', '.4f')}, maximum {f('hyp.others_max_abs_step_per25_u900plus_max', '.4f')}; not above the others' maximum: "
      f"{le_max('hyp.target_max_abs_step_per25_u900plus', 'hyp.others_max_abs_step_per25_u900plus_max')}).")
    P(f"- LR-decay window: mean(1225-1600) minus mean(900-1200) is {sg('seedstat.dec_minus_pre.target')} for 30510 (rank {f('seedstat.dec_minus_pre.rank_lowest_of_20', 'd')} of 20 from the lowest; {inr('seedstat.dec_minus_pre')} the others' range); "
      f"the other seeds: median {sg('seedstat.dec_minus_pre.others_median')}, p10 {sg('seedstat.dec_minus_pre.others_p10')}, p90 {sg('seedstat.dec_minus_pre.others_p90')}. OLS slope of the late series: "
      f"{sg('seedstat.late_slope_per100.target', '+.4f')} per 100 updates (others' median {sg('seedstat.late_slope_per100.others_median', '+.4f')}).")
    P(f"- q=60 pack (figures/fig2_peak_trajectory_q60_reference.png; 20 runs): median {sg('traj60.e2_0.at_u1600.median')}, p10 {sg('traj60.e2_0.at_u1600.p10')}, p90 {sg('traj60.e2_0.at_u1600.p90')} at u=1600; "
      f"30510 ({sg('traj.e2_0.at_u1600.target')}) is below that p10: {V['traj.e2_0.at_u1600.target'] < V['traj60.e2_0.at_u1600.p10']}.")
    P("")
    P("Per-seed trajectory statistics (source: `tables/tab_per_seed_trajectory_stats.csv`, from [W]):")
    P("")
    rows = []
    for lab, k, spec in (("late mean of e2_hat(0) error (u 900-1600)", "late_mean", "+.4f"), ("late maximum", "late_max", "+.4f"), ("late SD", "late_sd", ".4f"),
                         ("mean(1225-1600) - mean(900-1200)", "dec_minus_pre", "+.4f"), ("OLS slope per 100 updates (u 900-1600)", "late_slope_per100", "+.4f"),
                         ("fraction of late exports below own leave-one-out p10", "loo_frac_below_p10_late", ".3f")):
        rows.append(stat_row(lab, f"seedstat.{k}", spec))
    P(table(["statistic", f"seed {TS}", "others median", "others p10 .. p90", "others min .. max", "rank of 30510 (1 = lowest)"], rows))
    P("")
    P("### 2.5 Shape of the peak  [W]")
    P("")
    P(table(["quantity at u=1600", f"seed {TS}", "others median", "others p10 .. p90"], [
        ["e2_hat(0) error", sg("traj.e2_0.at_u1600.target"), sg("traj.e2_0.at_u1600.others_median"), f"{sg('traj.e2_0.at_u1600.others_p10')} .. {sg('traj.e2_0.at_u1600.others_p90')}"],
        ["error at |d|=10 (mean of +-10, relative to e2*(10))", sg("shoulder.shoulder10_rel_err.u1600.target"), sg("shoulder.shoulder10_rel_err.u1600.others_median"), f"{sg('shoulder.shoulder10_rel_err.u1600.others_p10')} .. {sg('shoulder.shoulder10_rel_err.u1600.others_p90')}"],
        ["error at |d|=20", sg("shoulder.shoulder20_rel_err.u1600.target"), sg("shoulder.shoulder20_rel_err.u1600.others_median"), f"{sg('shoulder.shoulder20_rel_err.u1600.others_p10')} .. {sg('shoulder.shoulder20_rel_err.u1600.others_p90')}"],
        ["error at |d|=40", sg("shoulder.shoulder40_rel_err.u1600.target"), sg("shoulder.shoulder40_rel_err.u1600.others_median"), f"{sg('shoulder.shoulder40_rel_err.u1600.others_p10')} .. {sg('shoulder.shoulder40_rel_err.u1600.others_p90')}"],
        ["cusp depth: error at 0 minus error at |d|=10", sg("shoulder.cusp_minus_shoulder10.target"), sg("shoulder.cusp_minus_shoulder10.others_median"), f"{sg('shoulder.cusp_minus_shoulder10.others_p10')} .. {sg('shoulder.cusp_minus_shoulder10.others_p90')}"]]))
    P("")
    P(f"The location-free peak of 30510 is {sg('owner.locfree_err')} at d={f('owner.locfree_argmax_d', '.1f')}; L - S at u=1600 is {sg('rules.target.cusp_rounding_1600')}. "
      f"The error at |d|=10 is {sg('shoulder.shoulder10_rel_err.u1600.target')} (others' median {sg('shoulder.shoulder10_rel_err.u1600.others_median')}) and at |d|=40 {sg('shoulder.shoulder40_rel_err.u1600.target')} (others' median {sg('shoulder.shoulder40_rel_err.u1600.others_median')}) "
      f"(figures/fig3b_profile_evolution_near_peak.png). The depth of the cusp (error at d=0 minus error at |d|=10) is {sg('shoulder.cusp_minus_shoulder10.target')} (others' median {sg('shoulder.cusp_minus_shoulder10.others_median')}, "
      f"p10 {sg('shoulder.cusp_minus_shoulder10.others_p10')}). e2_hat(0) in effort units at 400/800/1200/1600: "
      f"{f('profev.u400.target_e2_0', '.1f')} / {f('profev.u800.target_e2_0', '.1f')} / {f('profev.u1200.target_e2_0', '.1f')} / {f('profev.u1600.target_e2_0', '.1f')} for 30510, against medians of "
      f"{f('profev.u400.others_median_e2_0', '.1f')} / {f('profev.u800.others_median_e2_0', '.1f')} / {f('profev.u1200.others_median_e2_0', '.1f')} / {f('profev.u1600.others_median_e2_0', '.1f')} (e2*(0) = {f('meta.g2_at_0_q50', '.0f')}).")
    P("")
    # ---- 3
    P("## 3. End-of-A profile (figures/fig3_endA_profile.png)")
    P("")
    P(f"Contrast seeds: the two other q=50 seeds nearest to the median signed peak error of the others ({sg('contrast.median_peak_err_others')}): {c1} and {c2}. Sources: [A], [G].")
    P("")
    keys = [("peak error (signed)", "peak_rel_err", "+.4f"), ("location-free peak error", "locfree_rel_err", "+.4f"), ("location-free argmax d", "locfree_argmax_d", ".1f"),
            ("e2_hat(0)", "e2_0", ".3f"), ("eta_2/DW (final tier)", "eta2_over_dw", ".6f"), ("argmax d* of Delta_2", "eta2_argmax_d", ".0f"), ("on-path max Delta_2/DW", "on_max", ".6f"),
            ("off-path max Delta_2/DW", "off_max", ".6f"), ("argmax d (off-path)", "off_argmax_d", ".0f"), ("symmetry error max", "sym_err_max", ".3f"), ("|d| of the symmetry maximum", "sym_err_argmax_abs_d", ".1f"),
            ("sigma_2(0)", "sigma2_at_0", ".3f"), ("mean sigma_2 over |d|<2q", "sigma2_mean_pos", ".3f"), ("tail max (|d|>=2q)", "tail_max", ".3f"), ("tail argmax d", "tail_argmax_d", ".0f"),
            ("RMSE on |d|<2q / e2*(0)", "rmse_pos_over_g20", ".4f"), ("tail mean / e2*(0)", "tail_mean_over_g20", ".5f"), ("error at +-10 (effort)", "err_at_pm10", "+.3f"), ("error at +-20 (effort)", "err_at_pm20", "+.3f"),
            ("fraction of on-path nodes with e2_hat below e2*", "frac_pos_nodes_below_target", ".3f")]
    P(table(["quantity", f"seed {TS}", f"seed {c1}", f"seed {c2}"], [[lab, *[format(V[f"prof.s{s}.{k}"], spec) for s in (TS, c1, c2)]] for lab, k, spec in keys]))
    P("")
    P("All 20 q=50 runs (source: `tables/tab_endA_scalars_all_runs.csv`, from [G] and [A]):")
    P("")
    P(table(["quantity", f"seed {TS}", "others median", "others p10 .. p90", "others min .. max", "rank (1 = lowest)"], [
        stat_row("eta_2/DW (= on-path max)", "end.eta2_over_dw", ".5f"), stat_row("off-path max Delta_2/DW", "end.off_max", ".6f"), stat_row("sigma_2(0)", "end.sigma0", ".3f"),
        stat_row("mean sigma_2 over |d|<2q", "end.sigma2_mean_pos", ".3f"), stat_row("RMSE on |d|<2q / e2*(0)", "end.rmse_over_g20", ".4f"), stat_row("tail max", "end.tail_max", ".3f"),
        stat_row("symmetry error max", "end.sym_err_max", ".3f"), stat_row("tail mean / e2*(0)", "end.tail_mean_over_g20", ".5f")]))
    P("")
    P(f"- On/off-path split: the maximum of Delta_2 is on-path at d*={f('prof.s%d.eta2_argmax_d' % TS, '.0f')} ({f('owner.on_max', '.5f')}); the off-path maximum is {f('owner.off_max', '.6f')} at d={f('prof.s%d.off_argmax_d' % TS, '.0f')}. "
      f"Ranks from the lowest of the 20 q=50 runs: on-path {f('end.on_max.rank_lowest_of_20', 'd')}, off-path {f('end.off_max.rank_lowest_of_20', 'd')} (20 = largest). Others: on-path max {f('end.on_max.others_median', '.5f')} median, {f('end.on_max.others_max', '.5f')} maximum; "
      f"off-path {f('end.off_max.others_median', '.6f')} median, {f('end.off_max.others_max', '.6f')} maximum. On-path value above the G-A threshold {f('owner.eta2_threshold', '.3f')}: {V['owner.on_max'] > V['owner.eta2_threshold']}; off-path value above it: {V['owner.off_max'] > V['owner.eta2_threshold']}.")
    P(f"- Location of d*: the on-path argmax is within |d|<=20 for {f('end.n_others_on_argmax_abs_le20', 'd')} of the 19 other q=50 seeds (median |d*| {f('end.on_argmax_abs_d.others_median', '.0f')}) and for {f('end.n_q60_on_argmax_abs_le20', 'd')} of 20 q=60 runs; for seed {TS}: {f('end.on_argmax_d.target', '.0f')}. "
      f"The on-path maximum of seed {TS} is {f('end.on_max.target', '.5f')} against a maximum of {f('end.on_max.others_max', '.5f')} for the others. "
      f"At the development-tier calls from u=400 on, the on-path argmax of 30510 is within |d|<=20 at {f('calls.target.n_on_argmax_abs_le20_u400plus', 'd')} of {f('calls.target.n_calls_u400plus', 'd')} calls (the others: {[(u, d) for u, d in V['calls.target.on_argmax_us_u400plus'] if abs(d) > 20]}), against a fraction of {f('calls.others.frac_on_argmax_abs_le20_u400plus', '.2f')} of the other seeds' calls.")
    P(f"- Policy noise: sigma_2(0) = {f('end.sigma0.target', '.3f')} (rank {f('end.sigma0.rank_lowest_of_20', 'd')} of 20 from the lowest; others {f('end.sigma0.others_median', '.3f')} median, {f('end.sigma0.others_max', '.3f')} maximum). "
      f"Mean sigma_2 over |d|<2q: {f('end.sigma2_mean_pos.target', '.3f')} against {f('end.sigma2_mean_pos.others_median', '.3f')} (others' median; rank {f('end.sigma2_mean_pos.rank_lowest_of_20', 'd')}); see panel (e).")
    P(f"- Symmetry error: rank {f('end.sym_err_max.rank_lowest_of_20', 'd')} of 20 from the lowest; {f('end.sym_err_max.target', '.3f')} against a median of {f('end.sym_err_max.others_median', '.3f')}. "
      f"RMSE {f('end.rmse_over_g20.target', '.4f')} (rank {f('end.rmse_over_g20.rank_lowest_of_20', 'd')}) and tail max {f('end.tail_max.target', '.3f')} (rank {f('end.tail_max.rank_lowest_of_20', 'd')}); gate G-A(RMSE) {'pass' if V['gates.G_A_rmse_pass'] else 'fail'} "
      f"(threshold {f('meta.G_A_rmse_threshold', '.2f')}), gate G-A(tail) {'pass' if V['gates.G_A_tail_pass'] else 'fail'} (threshold {f('meta.G_A_tail_threshold', '.2f')}; tail mean/e2*(0) = {f('end.tail_mean_over_g20.target', '.5f')}).")
    P("")
    # ---- 4
    P("## 4. Optimisation (figures/fig4_optimisation_q50.png, figures/fig5_eta2_calls.png)")
    P("")
    P("### 4.1 Window means (sources: [U] for KL, clip fraction, advantage SD, losses; [H] for grad norms; [W] for concentration and sigma_2(0))")
    P("")
    rows = []
    for lab, k, spec in (("KL after the 10 epochs", "kl", ".5f"), ("clip fraction", "clip_frac", ".4f"), ("actor grad norm, mean pre-clip (clip level 0.5)", "grad_norm_actor_mean", ".2f"),
                         ("advantage SD (raw)", "adv_sd", ".4f"), ("critic value loss", "value_loss", ".5f")):
        for lo, hi in ((1, 100), (101, 400), (401, 900), (901, 1200), (1201, 1600)):
            b = f"opt.{k}.w{lo}_{hi}"
            rows.append([lab, f"{lo}-{hi}", f(b + ".target", spec), f(b + ".others_median", spec), f"{f(b + '.others_p10', spec)} .. {f(b + '.others_p90', spec)}", f(b + ".rank_lowest_of_20", "d")])
    for lab, k, spec in (("concentration alpha+beta at d=0 (mean of exports)", "conc0", ".1f"), ("sigma_2(0) (mean of exports)", "sigma0", ".3f")):
        for lo, hi in ((1, 400), (900, 1600)):
            b = f"opt.{k}.u{lo}_{hi}"
            rows.append([lab, f"{lo}-{hi}", f(b + ".target", spec), f(b + ".others_median", spec), f"{f(b + '.others_p10', spec)} .. {f(b + '.others_p90', spec)}", f(b + ".rank_lowest_of_20", "d")])
    P(table(["quantity", "updates", f"seed {TS}", "others median", "others p10 .. p90", "rank (1 = lowest)"], rows))
    P("")
    P(f"The actor gradient is clipped to norm {f('opt.max_grad_norm', '.1f')} ({f('opt.n_minibatch_steps_per_update', 'd')} minibatch steps per update); the others' pre-clip window means are of the order of 3-4 (10th percentile {f('opt.grad_norm_actor_mean.w901_1200.others_p10', '.2f')} in the 901-1200 window), "
      f"i.e. {V['opt.grad_norm_actor_mean.w901_1200.others_p10'] / V['opt.max_grad_norm']:.1f} times the clip level; the pre-clip norm of 30510 is {V['opt.grad_norm_actor_mean.w901_1200.target'] / V['opt.max_grad_norm']:.1f} times it. Clipping bounds the step, so a larger pre-clip norm is not by itself a larger step. Ranks (1 = lowest of the 20, 20 = highest) of 30510 in the 1-100 window: "
      + ", ".join(f"{lab} {f(f'opt.{k}.w1_100.rank_lowest_of_20', 'd')}" for lab, k in (("KL", "kl"), ("clip fraction", "clip_frac"), ("actor grad norm", "grad_norm_actor_mean"), ("advantage SD", "adv_sd"), ("critic loss", "value_loss")))
      + " (no excess in this window); in the 101-400 window: "
      + ", ".join(f"{lab} {f(f'opt.{k}.w101_400.rank_lowest_of_20', 'd')}" for lab, k in (("KL", "kl"), ("clip fraction", "clip_frac"), ("actor grad norm", "grad_norm_actor_mean"), ("advantage SD", "adv_sd"), ("critic loss", "value_loss")))
      + ". The separation starts right after the first 100 updates.")
    P("")
    P("### 4.2 Where the optimisation footprint leaves the others' band (same rules as 2.1; the 'beyond' side is above the 90th percentile, or below the 10th for the concentration)  [W], [U], [H]")
    P("")
    rows = []
    for lab, k, side in (("concentration alpha+beta at d=0", "conc0", "below p10"), ("actor grad norm (mean, pre-clip)", "grad_norm_actor_mean", "above p90"), ("sigma_2(0)", "sigma0", "above p90"),
                         ("advantage SD", "adv_sd", "above p90"), ("critic value loss", "value_loss", "above p90"), ("clip fraction", "clip_frac", "above p90"), ("KL", "kl", "above p90")):
        rows.append([lab, side, f(f"optexit.{k}.first_below_u", "d"), f(f"optexit.{k}.run8_start_u", "d"), f(f"optexit.{k}.sustained_exit_u", "d"), f(f"optexit.{k}.last_in_band_u", "d"),
                     f(f"optexit.{k}.frac_below", ".2f"), f(f"optexit.{k}.frac_below_late", ".2f")])
    P(table(["series", "beyond-band side", "first export beyond", "start of first run of >= 8", "sustained departure", "last in band", "fraction beyond (all exports)", "fraction beyond (u >= 900)"], rows))
    P("")
    P(f"Spearman correlation across the 19 other seeds between the late level of the peak error and each of these (mean over updates >= 900, n=19): concentration at d=0 {sg('corr.late_peak_vs_conc0.others19', '+.2f')}, sigma_2(0) {sg('corr.late_peak_vs_sigma0.others19', '+.2f')}, "
      f"actor grad norm {sg('corr.late_peak_vs_grad_norm_actor_mean.others19', '+.2f')}, KL {sg('corr.late_peak_vs_kl.others19', '+.2f')}, advantage SD {sg('corr.late_peak_vs_adv_sd.others19', '+.2f')}, critic loss {sg('corr.late_peak_vs_value_loss.others19', '+.2f')} "
      f"(with 30510 included, n=20: {sg('corr.late_peak_vs_conc0.all20', '+.2f')}, {sg('corr.late_peak_vs_sigma0.all20', '+.2f')}, {sg('corr.late_peak_vs_grad_norm_actor_mean.all20', '+.2f')}). Source: `tables/tab_late_means_per_seed_q50.csv`. "
      f"Largest absolute coefficient among the 19 others: {max(abs(V[f'corr.late_peak_vs_{n_}.others19']) for n_ in ('kl', 'clip_frac', 'grad_norm_actor_mean', 'adv_sd', 'value_loss', 'conc0', 'sigma0')):.2f}. Descriptive only; association is not cause.")
    P("")
    P("### 4.3 Dev-tier eta_2/DW at every Phase-A verifier call  [C]")
    P("")
    rows = []
    for u in range(100, 1601, 100):
        k = f"calls.eta2.u{u}"
        rows.append([u, f(k + ".target", ".5f"), f(k + ".others_median", ".5f"), f"{f(k + '.others_min', '.5f')} .. {f(k + '.others_max', '.5f')}", f(k + ".rank_lowest_of_20", "d")])
    P(table(["update (call)", f"seed {TS}", "others median", "others min .. max", "rank (1 = lowest, 20 = highest eta_2)"], rows))
    P("")
    P(f"- 30510 is above the others' median at {f('calls.target.n_calls_above_others_median', 'd')} of {f('calls.target.n_calls_total', 'd')} calls, the highest of the 20 at {f('calls.target.n_calls_highest_of_20', 'd')}, and above every other seed at {f('calls.target.n_calls_above_others_max', 'd')}. "
      f"From u=400 on, its value is {f('calls.target.eta2_min_u400plus', '.5f')} at the lowest and {f('calls.target.eta2_max_u400plus', '.5f')} at the highest; it is above 0.005 at {f('calls.target.n_calls_gt_0p005_u400plus', 'd')} of {f('calls.target.n_calls_u400plus', 'd')} calls, "
      f"with the on-path maximum above the off-path maximum at every one (flag: {V['calls.target.on_gt_off_all_calls_u400plus']}).")
    P(f"- For the other seeds, {f('calls.others.frac_calls_gt_0p005_u400plus', '.3f')} of their calls from u=400 on exceed 0.005 (largest value {f('calls.others.max_eta2_u400plus', '.5f')}); {f('calls.others.n_runs_with_any_call_gt_0p005_u400plus', 'd')} of the 19 had at least one such call. "
      f"At the last call (u=1600) the others' values are {f('calls.others.eta2_u1600_median', '.5f')} median and {f('calls.others.eta2_u1600_max', '.5f')} maximum (all below 0.005: {V['calls.others.eta2_u1600_max'] < 0.005}).")
    P(f"- Calls of 30510 at u=1300 ({f('calls.eta2.u1300.target', '.5f')}), 1400 ({f('calls.eta2.u1400.target', '.5f')}) and 1500 ({f('calls.eta2.u1500.target', '.5f')}) against the 0.005 threshold (all below: "
      f"{all(V[f'calls.eta2.u{u_}.target'] < 0.005 for u_ in (1300, 1400, 1500))}); the end-of-A value {f('calls.eta2.u1600.target', '.5f')} is above it: {V['calls.eta2.u1600.target'] > 0.005}.")
    if have_verifier:
        P(f"- Recomputed at every export from u=400 on [V]: 30510 has eta_2/DW between {f('eta_exp.target_min_u400plus', '.5f')} and {f('eta_exp.target_max_u400plus', '.5f')}; from u=900 on it is above 0.005 at "
          f"{f('eta_exp.target_n_gt_0p005_u900plus', 'd')} of {f('eta_exp.target_n_u900plus', 'd')} exports (median {f('eta_exp.target_median_u900plus', '.5f')}); the others' pooled exports from u=900 on: median {f('eta_exp.others_median_u900plus', '.5f')}, 90th percentile {f('eta_exp.others_p90_u900plus', '.5f')}, "
          f"99th percentile {f('eta_exp.others_p99_u900plus', '.5f')}, maximum {f('eta_exp.others_max_u900plus', '.5f')}; {f('eta_exp.others_frac_gt_0p005_u900plus', '.3f')} of them exceed 0.005. 30510 is above the others' 90th percentile at {f('eta_exp.target_frac_above_p90_u900plus', '.2f')} of the exports from u=900 on. "
          f"Fraction of 30510's exports from u=900 on above 0.005: {V['eta_exp.target_n_gt_0p005_u900plus'] / V['eta_exp.target_n_u900plus']:.2f}; ratio of the target's median to the others' pooled median over the same exports: "
          f"{V['eta_exp.target_median_u900plus'] / V['eta_exp.others_median_u900plus']:.1f}; ratio of the end-of-A values (target over the others' median): {V['end.eta2_over_dw.target'] / V['end.eta2_over_dw.others_median']:.1f}.")
    P(f"- `would_have_fired` ([S], fixed-budget run): update {f('wf.target_A_global_update', 'd')} for 30510; the other seeds: median {f('wf.others_A_median', '.0f')}, range {f('wf.others_A_min', '.0f')} .. {f('wf.others_A_max', '.0f')}. "
      f"The Phase-A rule fires after {f('wf.k_phase', 'd')} consecutive eligible calls; a call is eligible when eta_2/DW <= {f('wf.phase_A_rule_threshold_over_dw', '.2f')} (not the G-A threshold 0.005) and the maximum over the D2 grid of the normalised policy std is <= {f('elig.conc_thr_normalised', '.2f')} "
      f"(= {100 * V['elig.conc_thr_normalised']:.1f} effort units). For 30510 eta_2/DW is <= 0.02 from the call at u={f('elig.target_first_call_eta_le_0p02', 'd')} (others: median {f('elig.others_first_call_eta_le_0p02_median', '.0f')}, latest {f('elig.others_first_call_eta_le_0p02_max', 'd')}), "
      f"but the maximum std is above the threshold at calls up to u={f('elig.target_last_call_conc_above_thr', 'd')} (others: last such call, median {f('elig.others_last_call_conc_above_thr_median', '.0f')}, latest {f('elig.others_last_call_conc_above_thr_max', 'd')}); its first eligible call is u={f('elig.target_first_eligible_call_u', 'd')} "
      f"(others: median {f('elig.others_first_eligible_call_median', '.0f')}, range {f('elig.others_first_eligible_call_min', 'd')} .. {f('elig.others_first_eligible_call_max', 'd')}). Of its {f('elig.target_n_calls_not_eligible', 'd')} non-eligible calls, "
      f"{f('elig.target_n_calls_eta_le_0p02_but_not_eligible', 'd')} have eta_2/DW <= 0.02 and {f('elig.target_n_calls_not_eligible_with_conc_over_thr', 'd')} have the std above the threshold; "
      f"non-eligible calls with eta_2/DW <= 0.02 and the std at or below the threshold: {f('elig.target_n_noneligible_eta_ok_noise_ok', 'd')} (figures/fig5_eta2_calls.png panel d). The G-A verdict was {'pass' if V['wf.target_G_A_pass'] else 'fail'}.")
    P("")
    # ---- 5
    P("## 5. Sampling (figures/fig6_sampling.png)")
    P("")
    P(f"Phase A draws exploring starts bin-balanced on D_2 (bin chosen uniformly, then uniform inside the bin; bins of width 10: {f('vis.n_bins_q50', 'd')} bins for q=50). "
      f"The stored visitation counts the learner's starting-gap bins ({f('vis.peak_share.n_rows', 'd')} learner rows over 1600 updates). Peak set: bins {V['vis.peak_bins_q50']} (intervals meeting (-20, 20)). Sources: [H] "
      f"`verifier_calls[A].visitation_cumulative_phase.stage2_direct_es`; `tables/tab_visitation_peak_share_by_call.csv`.")
    P("")
    P(table(["quantity", f"seed {TS}", "others"], [
        ["peak-set share at u=1600", f("vis.peak_share.target", ".5f"), f"median {f('vis.peak_share.others_median', '.5f')}, range {f('vis.peak_share.others_min', '.5f')} .. {f('vis.peak_share.others_max', '.5f')}"],
        ["design value (4 of 40 bins)", f("vis.peak_share.expected", ".5f"), ""],
        ["binomial z-score of the share", sg("vis.peak_share.z_target", "+.2f"), f"range {sg('vis.peak_share.others_z_min', '+.2f')} .. {sg('vis.peak_share.others_z_max', '+.2f')}; SD of the 20 q=50 z-scores {f('vis.z_sd_q50', '.2f')}, of the 20 q=60 z-scores {f('vis.z_sd_q60', '.2f')}"],
        ["smallest / largest bin share", f"{f('vis.target_min_bin_share', '.4f')} / {f('vis.target_max_bin_share', '.4f')}", "design 0.0250"]]))
    P("")
    P(f"Largest |binomial z| of the cumulative peak-set share of seed {TS} over the Phase-A calls: {f('vis.target_absz_max_over_calls', '.2f')} (within +-2: {V['vis.target_absz_max_over_calls'] <= 2.0}); "
      f"at u=1600 the 19 other q=50 runs have z between {sg('vis.peak_share.others_z_min', '+.2f')} and {sg('vis.peak_share.others_z_max', '+.2f')}.")
    P("")
    P("D1 clamp counts, Phase A, summed over updates 1-1600 (raw Beta draws below 1e-6 or above 1-1e-6 before the clip; source [U] `d1_*` columns):")
    P("")
    P(table(["count", f"seed {TS}", "other 19 q=50: median", "other 19: maximum", "rank of 30510 (1 = lowest)"], [
        ["learner stage-2 rows", f("clamp.d1_L_s2_n_sum.target", "d"), f("vis.peak_share.n_rows", "d"), f("clamp.d1_L_s2_n_sum.others_max", "d"), ""],
        ["learner draws below 1e-6", f("clamp.d1_L_s2_lo_sum.target", "d"), f("clamp.d1_L_s2_lo_sum.others_median", ".0f"), f("clamp.d1_L_s2_lo_sum.others_max", "d"), f("clamp.d1_L_s2_lo_sum.rank_lowest_of_20", "d")],
        ["learner draws above 1-1e-6", f("clamp.d1_L_s2_hi_sum.target", "d"), "0", f("clamp.d1_L_s2_hi_sum.others_max", "d"), ""],
        ["  of which inside |d|<2q (lo + hi)", f"{f('clamp.d1_L_s2_in_lo_sum.target', 'd')} + {f('clamp.d1_L_s2_in_hi_sum.target', 'd')}", "0", f"{f('clamp.d1_L_s2_in_lo_sum.others_max', 'd')}", ""],
        ["opponent draws below 1e-6", f("clamp.d1_O_s2_lo_sum.target", "d"), "", f("clamp.d1_O_s2_lo_sum.others_max", "d"), ""],
        ["learner rows with alpha < 1", f("clamp.d1_pol_n_alpha_lt1_sum.target", "d"), f("clamp.d1_pol_n_alpha_lt1_sum.others_median", ".0f"), f("clamp.d1_pol_n_alpha_lt1_sum.others_max", "d"), f("clamp.d1_pol_n_alpha_lt1_sum.rank_lowest_of_20", "d")],
        ["learner rows with beta < 1", f("clamp.d1_pol_n_beta_lt1_sum.target", "d"), "0", f("clamp.d1_pol_n_beta_lt1_sum.others_max", "d"), ""],
        ["smallest alpha seen", f("clamp.pol_alpha_min_target", ".4f"), "", f"(others' smallest: {f('clamp.pol_alpha_min_others_min', '.4f')})", ""]]))
    P("")
    P(f"All clamp hits of 30510 are lower-clamp hits at stage-2 states outside the support (|d| >= 2q, where e2* = 0 and the learned mean is small: tail mean {V['end.tail_mean_over_g20.target'] * V['meta.g2_at_0_q50']:.2f} effort units, alpha < 1). Over all 40 runs: {f('clamp.all40_inside_support_total', 'd')} hits inside |d| < 2q and {f('clamp.all40_total_hi', 'd')} at the upper clamp (all-run total of lo+hi counts {f('clamp.all40_total_lo_hi', 'd')}, "
      f"of which q=60 maximum per run {f('clamp.q60.d1_L_s2_lo_sum.max', 'd')}). The count of 30510 ({f('clamp.d1_L_s2_lo_sum.target', 'd')}, {100 * V['clamp.lo_frac_of_learner_rows.target']:.1f}% of its learner rows, {100 * V['clamp.lo_frac_of_out_rows.target']:.1f}% of its out-of-support rows) is "
      f"{f('clamp.d1_L_s2_lo_sum.target_over_others_median', '.0f')} times the median of the others and {V['clamp.d1_L_s2_lo_sum.target'] / V['clamp.d1_L_s2_lo_sum.others_max']:.1f} times the next largest ({f('clamp.d1_L_s2_lo_sum.others_max', 'd')}); "
      f"{f('clamp.n_q50_others_with_lo_gt_20000', 'd')} other q=50 seed exceeds 20,000. Its share of rows with alpha<1 is {100 * V['clamp.alpha_lt1_frac_of_rows.target']:.1f}% against a median of {100 * V['clamp.alpha_lt1_frac_of_rows.others_median']:.1f}%. "
      f"The counts are produced by the policy's draws at the stored starting states, whose bin distribution is the same design in every run (above). Spearman correlation of the number of clamp hits with the peak error "
      f"{sg('clamp.spearman_peak_err_vs_lo_sum.q50_others', '+.2f')} among the 19 others, {sg('clamp.spearman_peak_err_vs_lo_sum.pooled40', '+.2f')} over all 40 runs (descriptive).")
    P("")
    # ---- 6
    P("## 6. Three-way decomposition of the d=0 peak gap (figures/fig7_decomposition.png)")
    P("")
    P("Definitions (the repo's: `run/run_v2_T2_locked.smoothed_share`, pilot 4 section 1d). RL gap = e2*(0) - e2_hat(0) (effort units). Smoothing-predicted gap = e2*(0) - e_pred(0), where e_pred(0) = DW/(2k) E[f_xi(x_i - x_j)] "
      "and x_i, x_j are independent draws of the policy's own d=0 noise (Beta quantile nodes centred on its mean), i.e. what a best responder would play against the policy's own randomness. "
      "Remainder = e_pred(0) - e2_hat(0) = RL gap - smoothing-predicted gap (the 'smoothing-free' part). Share = smoothing-predicted gap / RL gap (`smoothed_share_peak_gap_d0`). "
      "Supervised floor = e2*(0) - e2_fit(0) for the same actor class fitted to e2* by least squares (pilot 4 section 1d, 5 inits, 300,000 steps; the plateau rule never fired, so these values are upper bounds of the floor; "
      "this is not a fit of this seed). Sources: [G] (`reported.end_of_A`), [F].")
    P("")
    P(table(["component (effort units)", f"seed {TS}", "others median", "others p10 .. p90", "others min .. max", "rank (1 = lowest)"], [
        stat_row("RL gap  e2*(0) - e2_hat(0)", "end.rl_gap", ".3f"), stat_row("smoothing-predicted gap", "end.smooth_gap", ".3f"), stat_row("remainder (smoothing-free)", "end.remainder_gap", ".3f"),
        stat_row("share = smoothing gap / RL gap", "end.smooth_share", ".3f"), stat_row("e_pred(0)", "end.smooth_pred0", ".3f")]))
    P("")
    if "floor.gap_d0_median" not in V:
        P("- (supervised floor not read: --floor-dir was not given)")
    else:
      P(f"- Supervised floor (same actor class, fitted to e2*): median {f('floor.gap_d0_median', '.5f')} effort units over {f('floor.n_fits', 'd')} initialisations (range {f('floor.gap_d0_min', '.4f')} .. {f('floor.gap_d0_max', '.4f')}); "
      f"the RL gap of 30510 is about {V['dec.rl_gap_over_floor_median']:.0f} times that median. The pilot-4 reference (Phase-A extension, 10 other seeds at u1600, [F]): RL gap median {f('floor.pilot4.rl_peak_gap_d0.median', '.3f')}, smoothing-predicted {f('floor.pilot4.smoothing_predicted_gap_d0.median', '.3f')}. "
      f"Largest absolute deviation of the five floor fits at d=0: {f('floor.gap_d0_absmax', '.4f')} effort units.")
    P(f"- RL gap of 30510 is {f('dec.target_rl_gap_over_others_median_rl_gap', '.2f')} times the others' median ({f('end.rl_gap.others_max', '.2f')} maximum). Its smoothing-predicted part is {f('end.smooth_gap.target', '.3f')}, larger than any other seed's ({f('end.smooth_gap.others_max', '.3f')} maximum); the prediction is a function of sigma_2(0), which is the largest; "
      f"it covers {f('end.smooth_share.target', '.3f')} of the gap against a median of {f('end.smooth_share.others_median', '.3f')} for the others (rank {f('end.smooth_share.rank_lowest_of_20', 'd')} from the lowest). The remainder, {f('end.remainder_gap.target', '.3f')}, is {f('dec.target_remainder_over_others_median_remainder', '.1f')} times the others' median ({f('end.remainder_gap.others_median', '.3f')}; maximum {f('end.remainder_gap.others_max', '.3f')}).")
    P(f"- q=60 for scale (20 runs): RL gap median {f('dec60.rl_gap.median', '.3f')} (p10 {f('dec60.rl_gap.p10', '.3f')}, p90 {f('dec60.rl_gap.p90', '.3f')}, max {f('dec60.rl_gap.max', '.3f')}); remainder median {f('dec60.remainder_gap.median', '.3f')} (max {f('dec60.remainder_gap.max', '.3f')}); share median {f('dec60.smooth_share.median', '.3f')}.")
    P(f"- Across the 19 other q=50 seeds, OLS of RL gap on smoothing-predicted gap: slope {f('dec.fit_others.slope', '.2f')}, intercept {f('dec.fit_others.intercept', '.2f')}; the line predicts {f('dec.fit_others.pred_for_target', '.2f')} for 30510, observed {f('end.rl_gap.target', '.2f')} "
      f"(residual {sg('dec.fit_others.resid_target', '+.2f')} = {f('dec.fit_others.resid_target_over_sd', '.1f')} residual SDs; descriptive, 19 points, an extrapolation in the smoothing-gap axis). Panel (b) shows the same.")
    P("")
    # ---- 7
    def hd(b):
        """'holds' / 'does not hold' for a computed boolean (post hoc readings)."""
        return "holds" if b else "does not hold"

    def rv(k):
        return V["rules.target." + k]

    ul = rv("u_leave")
    cond_h2_leave = bool(ul is not None and ul >= H2_FIRST_U)
    cond_h3_leave = bool(ul is not None and ul <= H3_LAST_LEAVE_U)
    pl = rv("plateau_len")
    cond_h3_plateau = bool(pl is not None and pl >= PLATEAU_EXPORTS)
    cond_none = bool(rv("out_S_1600") and (not rv("H1")) and cond_h3_leave and pl is not None and pl < PLATEAU_EXPORTS)
    P("## 7. Pre-registered hypothesis rules H1-H4, applied literally")
    P("")
    P("The rules below were written before any diagnostic output existed (pre-registration draft, mtime 07:03 UTC, as communicated to the author; that file is not read by this tool) and are applied literally by "
      "`apply_rules()`. Nothing was changed after the data were seen, including where a literal rule is awkward. Every verdict word in this section and in section 0 is produced from the computed booleans "
      "(`numbers.json` keys `rules.target.*`; `tables/hypothesis_rules.csv`).")
    P("")
    P("Notation. S(u) = (e2_hat(0; u) - e2*(0)) / e2*(0): signed peak error at d=0 from the weight export of update u (u = 25, ..., 1600). L(u) = (max_d e2_hat(d; u) - e2*(0)) / e2*(0) on the recovery grid "
      "(location-free peak error, `pilot4_common.location_free`). Pack = the 19 other q=50 runs of confirmation_v2_0. p10_pack(., u), p90_pack(., u): 10th / 90th percentile (numpy.percentile, linear) over the pack "
      "at the same export. out_S(u) := S_30510(u) < p10_pack(S, u). u_leave := the first export from which out_S holds at that export and at every later export up to 1600 (undefined if out_S(1600) is false). "
      "Resolution 25 updates.")
    P("")
    for h_ in ("H1", "H2", "H3", "H4", "NONE"):
        P(f"- **{h_}**: {RULE_TEXT[h_]}")
    P("")
    P("Verdict per hypothesis: 'supported' when the rule holds, 'not supported' when it does not, 'undetermined' if an input is missing. The rules are not exclusive by construction.")
    P("")
    P(table(["hypothesis", "computed inputs", "verdict"], [
        ["H1", f"max over all {f('meta.n_exports', 'd')} exports of S = {sg('rules.target.S_max_all')} (at u={f('rules.target.u_of_S_max', 'd')}); p10_pack(S, 1600) = {sg('rules.target.p10_S_1600')}; "
               f"max < p10: {rv('H1')}", rv("verdict_H1")],
        ["H2", f"not H1: {not rv('H1')}; out_S(1600): {rv('out_S_1600')}; u_leave = {f('rules.target.u_leave', 'd')}, u_leave >= {H2_FIRST_U}: {cond_h2_leave}", rv("verdict_H2")],
        ["H3", f"not H1: {not rv('H1')}; out_S(1600): {rv('out_S_1600')}; u_leave = {f('rules.target.u_leave', 'd')}, u_leave <= {H3_LAST_LEAVE_U}: {cond_h3_leave}; "
               f"consecutive exports not out_S ending at u_leave - 25: {f('rules.target.plateau_len', 'd')}, >= {PLATEAU_EXPORTS}: {cond_h3_plateau}", rv("verdict_H3")],
        ["H4", f"out_S(1600): {rv('out_S_1600')}; L >= p10_pack(L, u) at every export {H4_WIN[0]}..{H4_WIN[1]}: {rv('L_ge_p10_all_1225_1600')} "
               f"({f('rules.target.n_L_below_p10_1225_1600', 'd')} of {f('rules.target.n_exports_1225_1600', 'd')} exports below; smallest margin L - p10_pack(L) = {sg('rules.target.min_L_margin_1225_1600')}); "
               f"(L - S)(1600) = {sg('rules.target.cusp_rounding_1600', '+.5f')} against p90_pack(L - S, 1600) = {sg('rules.target.p90_cusp_1600', '+.5f')}, exceeds: {rv('cusp_exceeds_p90')}", rv("verdict_H4")]]))
    P("")
    P(f"**Label (from the rules): {rv('label')}.** The 'none' rule: out_S(1600) {rv('out_S_1600')}, not H1 {not rv('H1')}, u_leave <= {H3_LAST_LEAVE_U} {cond_h3_leave}, "
      f"no {PLATEAU_EXPORTS}-export (400-update) plateau {pl is not None and pl < PLATEAU_EXPORTS}: all four hold: {cond_none}.")
    P("")
    P(f"Awkward-looking features of the literal rules, reported as they come out (not changed): H1 compares the maximum over ALL exports, early transient included, with p10_pack(S, 1600); "
      f"for seed {TS} the margin max S - p10_pack(S, 1600) is {format(V['rules.target.S_max_all'] - V['rules.target.p10_S_1600'], '+.4f')} (H1 needs it to be negative), "
      f"the maximum being at u={f('rules.target.u_of_S_max', 'd')}. H3 counts the exports not out_S that end at u_leave - 25 = {(ul - 25) if ul is not None else 'n/a'}; the run is {f('rules.target.plateau_len', 'd')} export(s) long. "
      f"Consistency check: u_leave equals the 'sustained departure' of section 2.3: {V['rules.target.u_leave_equals_sustained_exit']}.")
    P("")
    P("### 7.1 The same rules applied to the other runs (how they behave on runs that passed)")
    P("")
    P(f"Each of the 20 q={TQ} runs taken as the target against the other 19 q={TQ} runs (the seed {TS} row repeats the target above), and each of the 20 q={a.q_other} runs against the other 19 q={a.q_other} runs. "
      f"Source: `tables/hypothesis_rules.csv` (`numbers.json` keys `rules.table.*`). Columns: G-A = the run's own G-A verdict; S = S(1600); p10 = p10_pack(S, 1600); out = out_S(1600); max S (u) = H1's statistic; "
      f"plateau = consecutive exports not out_S ending at u_leave - 25.")
    P("")
    for blk_q in (TQ, a.q_other):
        lab_txt = "; ".join(f"{k_}: {n_}" for k_, n_ in V[f"rules.table.q{blk_q}.label_counts"].items())
        P(f"**q={blk_q} block.** Labels ({lab_txt}). Runs with out_S(1600): {f(f'rules.table.q{blk_q}.n_out_S_1600', 'd')} of {f(f'rules.table.q{blk_q}.n_runs', 'd')}; "
          f"supported counts H1 / H2 / H3 / H4: {f(f'rules.table.q{blk_q}.n_H1_supported', 'd')} / {f(f'rules.table.q{blk_q}.n_H2_supported', 'd')} / {f(f'rules.table.q{blk_q}.n_H3_supported', 'd')} / {f(f'rules.table.q{blk_q}.n_H4_supported', 'd')}.")
        P("")
        rows_b = []
        for s_ in range(a.seeds[0], a.seeds[1] + 1):
            tk = f"rules.table.q{blk_q}.s{s_}"
            rows_b.append([s_, "pass" if V[tk + ".G_A_pass"] else "fail", sg(tk + ".S_1600"), sg(tk + ".p10_S_1600"), V[tk + ".out_S_1600"], f(tk + ".u_leave", "d"),
                           f"{sg(tk + '.S_max_all')} ({f(tk + '.u_of_S_max', 'd')})", f(tk + ".plateau_len", "d"), V[tk + ".verdict_H1"], V[tk + ".verdict_H2"], V[tk + ".verdict_H3"],
                           V[tk + ".verdict_H4"], V[tk + ".label"]])
        P(table(["seed", "G-A", "S", "p10", "out", "u_leave", "max S (u)", "plateau", "H1", "H2", "H3", "H4", "label"], rows_b))
        P("")
    P("### 7.2 Post hoc readings (not pre-registered)")
    P("")
    P("These are the author's own readings, written after the data were seen. They are not the pre-registered rules and carry no verdict; each line gives a rule, its computed inputs and whether it holds.")
    P("")
    P(f"- H1, strict from u >= 400 (S below p10_pack at every export u >= 400): {hd(V['hyp.H1_strict_all_below_u400plus'])}; in band at {f('traj.e2_0.n_in_band_u400plus', 'd')} of {f('traj.e2_0.n_exports_u400plus', 'd')} exports "
      f"({[u for u in V['traj.e2_0.in_band_us'] if u >= 400]}).")
    P(f"- H1, plateau reading (S below p10_pack at every export u >= 900): {hd(V['hyp.H1_late_all_below'])} ({f('traj.e2_0.n_below_u900plus', 'd')} of {f('traj.e2_0.n_exports_u900plus', 'd')}); late mean {sg('seedstat.late_mean.target')} "
      f"against the others' median {sg('seedstat.late_mean.others_median')}; first export with S >= -0.10 at u={f('seedstat.t_ge_m010.target', 'd')} (others' median {f('seedstat.t_ge_m010.others_median', '.0f')}).")
    P(f"- H2-like, in band at u=1200 and out of band at u=1600: {hd(V['hyp.H2_in_band_at_1200_and_out_by_1600'])} (S(1200) = {sg('traj.e2_0.at_u1200.target')} against p10 {sg('traj.e2_0.at_u1200.others_p10')}; "
      f"fraction of exports 900-1200 in band {f('hyp.H2_frac_in_band_900_1200', '.2f')}; last in-band export {f('hyp.H2_last_in_band_u', 'd')}, i.e. {LR_DECAY_FIRST - V['hyp.H2_last_in_band_u']} updates before the decay window). "
      f"Decay-window change mean(1225-1600) - mean(900-1200) = {sg('seedstat.dec_minus_pre.target')}, {inr('seedstat.dec_minus_pre')} the others' range ({sg('seedstat.dec_minus_pre.others_min')} .. {sg('seedstat.dec_minus_pre.others_max')}).")
    P(f"- H3-like, at least half of the exports 900-1200 in band followed by a drop out of band: {hd(V['hyp.H3_in_band_plateau_then_drop'])} (fraction in band {f('hyp.H2_frac_in_band_900_1200', '.2f')}); "
      f"largest single step {f('hyp.target_max_abs_step_per25_u900plus', '.4f')} (others' maximum {f('hyp.others_max_abs_step_per25_u900plus_max', '.4f')}; not above it: {le_max('hyp.target_max_abs_step_per25_u900plus', 'hyp.others_max_abs_step_per25_u900plus_max')}); "
      f"late slope {sg('seedstat.late_slope_per100.target', '+.4f')} per 100 updates (others' median {sg('seedstat.late_slope_per100.others_median', '+.4f')}).")
    l_in = V["end.locfree_rel_err.target"] >= V["end.locfree_rel_err.others_p10"]
    sh_in = V["shoulder.shoulder10_rel_err.u1600.target"] >= V["shoulder.shoulder10_rel_err.u1600.others_p10"]
    cusp_out = not (V["shoulder.cusp_minus_shoulder10.others_p10"] <= V["shoulder.cusp_minus_shoulder10.target"] <= V["shoulder.cusp_minus_shoulder10.others_p90"])
    rem_gt = V["end.remainder_gap.target"] > V["end.remainder_gap.others_max"]
    P(f"- H4-like shape readings at u=1600: location-free peak at or above the others' p10 ({sg('end.locfree_rel_err.target')} against {sg('end.locfree_rel_err.others_p10')}): {hd(l_in)}; "
      f"error at |d|=10 at or above the others' p10 ({sg('shoulder.shoulder10_rel_err.u1600.target')} against {sg('shoulder.shoulder10_rel_err.u1600.others_p10')}): {hd(sh_in)}; "
      f"cusp depth (error at 0 minus error at |d|=10) outside the others' p10..p90 ({sg('shoulder.cusp_minus_shoulder10.target')} against {sg('shoulder.cusp_minus_shoulder10.others_p10')} .. {sg('shoulder.cusp_minus_shoulder10.others_p90')}): {hd(cusp_out)}; "
      f"smoothing-free remainder of the d=0 gap above the others' maximum ({f('end.remainder_gap.target', '.2f')} against {f('end.remainder_gap.others_max', '.2f')}): {hd(rem_gt)}.")
    P(f"- Timing of the optimisation footprint (descriptive): sustained departure of the concentration at update {f('optexit.conc0.sustained_exit_u', 'd')}, actor grad norm {f('optexit.grad_norm_actor_mean.sustained_exit_u', 'd')}, "
      f"sigma_2(0) {f('optexit.sigma0.sustained_exit_u', 'd')}, advantage SD and critic loss {f('optexit.adv_sd.sustained_exit_u', 'd')}; these dates say when series leave their bands, not which one drives which.")
    P("")
    P("## 8. What cannot be distinguished, and which experiment would")
    P("")
    P(f"1. Cause or co-symptom of the early concentration/noise divergence. The footprint series leave their bands for good at updates {f('optexit.conc0.sustained_exit_u', 'd')}-{f('optexit.adv_sd.sustained_exit_u', 'd')}; the pre-registered u_leave of the peak is {f('rules.target.u_leave', 'd')}. "
      f"The smoothing prediction accounts for {f('end.smooth_share.target', '.0%')} of the gap; among the other 19 seeds the Spearman coefficients between the late peak error and these quantities are at most {max(abs(V[f'corr.late_peak_vs_{n_}.others19']) for n_ in ('kl', 'clip_frac', 'grad_norm_actor_mean', 'adv_sd', 'value_loss', 'conc0', 'sigma0')):.2f} in absolute value. "
      "The files contain no manipulation that separates 'a lower concentration produced the low peak' from 'both follow from an earlier state of the network'. An experiment that would: re-run seed 30510 (the pipeline is seeded, so the same streams are intended to reproduce it; not verified here) with a full-state checkpoint at update 100 "
      "(`full_state_at`), then branch from it with the minibatch and sampling streams reseeded (same network state, different noise): if every branch ends at the same plateau, the plateau is fixed by the state at update ~100; "
      "if only some do, it is noise-driven. A second branch with the concentration manipulated at that state (the repo has a `conc_anneal` mechanism for phase A continuation) would test the concentration route directly. Not run.")
    P(f"2. Whether the plateau is permanent. Over updates 900-1600 the late slope is {sg('seedstat.late_slope_per100.target', '+.4f')} per 100 updates (others {sg('seedstat.late_slope_per100.others_median', '+.4f')}). "
      "`state_end_A.pt` of the run exists, so a Phase-A continuation from it (mode `phase_A_continue`) would show whether it moves; not run.")
    P(f"3. How much of the G-A failure is the end-iterate draw. The per-export eta_2 of 30510 is above 0.005 at {f('eta_exp.target_n_gt_0p005_u900plus', 'd') if have_verifier else 'n/a'} of {f('eta_exp.target_n_u900plus', 'd') if have_verifier else 'n/a'} exports from u=900 on; "
      f"the ratio of its median to the others' pooled median is {V['eta_exp.target_median_u900plus'] / V['eta_exp.others_median_u900plus'] if have_verifier else float('nan'):.1f} (over the same exports). "
      "A tail-averaged candidate (pilot 4 section 1c) would be the experiment, and it is outside this diagnostic.")
    P("4. The clamp counts. A conjecture, untested: rows whose raw draw is clipped at 1e-6 have a log-density of the stored action that is large, and the gradient of that log-density with respect to alpha is of order |ln 1e-6| = 13.8 per such row; "
      "30510 has more of them, and its pre-clip actor gradient norm is larger. The files show only the co-occurrence (section 5 lists the counts and their Spearman correlation with the peak error). "
      "Re-running with the clip level moved would test it; not run.")
    P("5. The export at u=825 is a single point at a 25-update resolution; the data cannot say whether it is a transient of the policy mean or a move that was reversed.")
    P("")
    P("## 9. Not reproduced, inconsistencies, caveats")
    P("")
    P(f"- Reproduction of the four numbers quoted in the request: {'all four match to the digits quoted' if V['owner.match.all'] else 'AT LEAST ONE DOES NOT MATCH (section 1)'}.")
    P("- The location-free peak error recomputed from the weights differs from `gates.json` in the 7th digit (float32 rounding: numpy against torch); the `recovery_e2` arrays agree to 3e-5 effort units.")
    ns = {}
    for q_, s_, u_, r_ in V["meta.nonstandard_calls"]:
        ns.setdefault((q_, s_), []).append(f"{u_} ({r_})")
    P("- The weight exports are every 25 updates; any statement about an update that is not a multiple of 25 is not possible from the exports. Verifier calls are every 100 updates (dev tier) with these exceptions "
      f"(runs whose Phase-A calls are off that grid or not labelled timeout/warmup, as update (reason)): "
      + "; ".join(f"q={k_[0]} seed {k_[1]}: {', '.join(v_)}" for k_, v_ in ns.items()) + ". For q=60 seed 30505 a stability-triggered call at u=540 shifts the later calls to u=640, 740, ...; "
      "those calls have no weight export and are left out of the check of the recomputed eta_2 against the CSV (they remain in `tables/tab_eta2_verifier_calls_all_runs.csv`). No q=50 call used in the tables above is affected "
      "except that seed 30517's call at u=1600 is labelled `stability` (the same update).")
    P("- The supervised floor and the pilot-4 reference values are read from the pilot-4 tables [F]; they were not recomputed here and do not belong to these seeds. The floor fits stopped at the step cap and are upper bounds.")
    P("- 'Others' are 19 values: percentiles are coarse, and band membership at a single export carries little information; run-level statements use leave-one-out frequencies and ranks. Exports of a run are autocorrelated, so no binomial tail probability is quoted.")
    if have_verifier:
        P(f"- Provenance of the verifier: `utils/dp_br_verifier.py` (sha256 `{V['meta.sha256_repo_dp_br_verifier'][:16]}...`) and `utils/theory_multistage.py` (`{V['meta.sha256_repo_theory_multistage'][:16]}...`) were imported read-only from the worktree given with `--repo`, "
          f"not from the commit that produced the runs ({V['meta.run_commit'][:7]}); the agreement with the stored CSV to {V['repro.eta2_dev_max_absdiff_vs_csv']:.1e} at every call shows that it behaves as the code that was used.")
    P("- The per-export eta_2 (section 4.3, last bullet) is the development tier (state step 4); the gate uses the final tier. For 30510 at u=1600 the two on-path values are identical.")
    P("- The mean over buffer states of the entropy (`entropy_post_update_effort_scale`) and of the concentration (`conc_buf_mean`) are computed over a state-uniform buffer that is half out-of-support; they are in `tables/tab_opt_window100_all_runs.csv` but are not interpreted here.")
    P("")
    P("## 10. Files")
    P("")
    P(f"Figures (pdf with fonts type 42, and png): `{figdir}`; the report refers to them as figures/figN.png and is meant to sit next to that directory. Everything else under `{out}`: `report_draft.md`, `numbers.json` (values and sources), "
      "`tables/hypothesis_rules.csv` (pre-registered rules for the target and the 40 runs), `tables/*.csv` (every plotted series: `tab_peak_trajectory_band_q50.csv`, `tab_peak_trajectory_band_q60.csv`, `tab_exports_all_runs.csv`, "
      "`tab_endA_profile_*.csv`, `tab_opt_trailing25_band_q50.csv`, `tab_opt_window100_all_runs.csv`, `tab_opt_target_per_update.csv`, `tab_eta2_verifier_calls_all_runs.csv`, `tab_eta2_every_export_q50.csv`, `tab_visitation_*.csv`, "
      "`tab_d1_clamp_counts_phaseA_sum.csv`, `tab_decomposition_*.csv`, `tab_per_seed_trajectory_stats.csv`, `tab_late_means_per_seed_q50.csv`, `tab_profile_evolution_near_peak.csv`, `tab_would_have_fired_q50.csv`).")
    P("")
    P(f"Run commit of the data: `{V['meta.run_commit']}` (clean tree: {V['meta.run_clean_tree']}). Tool sha256 `{V['meta.tool_sha256'][:16]}...`. Command:")
    P("")
    P("```")
    P("OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python diag_seed30510.py --out <dir> --fig-dir <dir>/figures \\")
    P("    [--run-root <.../confirmation_v2_0>] [--repo <worktree>] [--floor-dir <.../v2_pilots/pilot4/analysis/repr_floor>]")
    P("```")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT), help="default: the v2-t2-refine worktree's confirmation_v2_0")
    ap.add_argument("--out", required=True, help="report_draft.md, numbers.json and tables/ are written here")
    ap.add_argument("--fig-dir", default=None, help="figure directory (default <out>/figures)")
    ap.add_argument("--q", type=int, default=50)
    ap.add_argument("--seed", type=int, default=30510)
    ap.add_argument("--q-other", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs=2, default=[30501, 30520])
    ap.add_argument("--analysis-root", default=None, help="default: <run-root>_analysis")
    ap.add_argument("--floor-dir", default=None, help="pilot4 repr_floor directory (default: the v2-t2-refine one if it exists)")
    ap.add_argument("--repo", default=None, help="worktree to import utils.dp_br_verifier from, read-only (default: the worktree "
                    "this file is in if it has utils/dp_br_verifier.py, else the v2-t2-refine worktree)")
    ap.add_argument("--no-verifier", action="store_true", help="skip the per-export eta_2 recomputation")
    ap.add_argument("--contrast-n", type=int, default=2)
    a = ap.parse_args()

    root, out = Path(a.run_root).resolve(), Path(a.out).resolve()
    here_repo = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else None
    if a.repo:
        repo = Path(a.repo).resolve()
    elif here_repo is not None and (here_repo / "utils" / "dp_br_verifier.py").exists():
        repo = here_repo
    else:
        repo = DEFAULT_REFINE
    a.repo = None if a.no_verifier else str(repo)
    if a.floor_dir is None and DEFAULT_FLOOR_DIR.exists():
        a.floor_dir = str(DEFAULT_FLOOR_DIR)
    figdir, tabdir = (Path(a.fig_dir).resolve() if a.fig_dir else out / "figures"), out / "tables"
    for target_dir in (out, figdir):          # never write into the run directories that are being read
        if target_dir == root or root in target_dir.parents or target_dir in root.parents:
            raise SystemExit(f"refusing to write at or around the run root {root}")
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    ana = Path(a.analysis_root) if a.analysis_root else Path(str(root) + "_analysis")
    N = Numbers()
    t_start = time.time()
    seeds = list(range(a.seeds[0], a.seeds[1] + 1))
    TQ, TS = a.q, a.seed
    verify_fn = make_verify_fn(Path(a.repo)) if a.repo else None

    # ---- load
    runs = {}
    for q in (a.q, a.q_other):
        for s in seeds:
            runs[(q, s)] = load_run(root, q, s, verify_fn)
            print(f"loaded q{q} seed{s}", flush=True)
    R50 = [runs[(TQ, s)] for s in seeds]
    R60 = [runs[(a.q_other, s)] for s in seeds]
    i_t = seeds.index(TS)
    oth50 = np.array([i for i, s in enumerate(seeds) if s != TS])
    all60 = np.arange(len(seeds))
    tgt = runs[(TQ, TS)]
    us = np.array(EXPORT_US)
    src_w = f"{root}/q*/seed*/weights/u*.npz (re-implemented actor, see Methods)"

    # ---- provenance
    N.put("meta.run_commit", tgt.gates["commit"], f"{tgt.dir}/gates.json commit (code that produced the runs)")
    N.put("meta.run_clean_tree", tgt.gates["clean_tree"], f"{tgt.dir}/gates.json")
    N.put("meta.sha256_gates_target", sha256(tgt.dir / "gates.json"), f"{tgt.dir}/gates.json")
    N.put("meta.sha256_weights_u1600_target", sha256(tgt.dir / "weights" / "u01600.npz"), f"{tgt.dir}/weights/u01600.npz")
    N.put("meta.sha256_gateA_final_target", sha256(tgt.dir / "gateA_final.npz"), f"{tgt.dir}/gateA_final.npz")
    N.put("meta.tool_sha256", sha256(Path(__file__).resolve()), "this script")
    if a.repo:
        for rel in ("utils/dp_br_verifier.py", "utils/theory_multistage.py"):
            N.put(f"meta.sha256_repo_{Path(rel).stem}", sha256(Path(a.repo) / rel), f"{a.repo}/{rel} (imported read-only for the per-export eta_2)")
    # ---- 0. reproduction of the owner's numbers and of the saved arrays
    rep = pd.read_csv(ana / "reported_metrics.csv")
    gr = tgt.gates["reported"]["end_of_A"]
    fin = gr["final"]
    N.put("owner.peak_err_signed", fin["stage2_peak_rel_err_signed"], f"{tgt.dir}/gates.json reported.end_of_A.final")
    N.put("owner.peak_err_signed_csv", float(rep[(rep.q == TQ) & (rep.seed == TS)]["A_stage2_peak_rel_err_signed"].iloc[0]),
          f"{ana}/reported_metrics.csv")
    N.put("owner.eta2_over_dw", tgt.gates["metric_values"]["eta_final"], f"{tgt.dir}/gates.json metric_values.eta_final")
    N.put("owner.eta2_threshold", tgt.gates["G-A"]["criteria"][0]["threshold"], f"{tgt.dir}/gates.json G-A.criteria[0]")
    N.put("owner.on_max", fin["DeltaT_over_dw_on_max"], f"{tgt.dir}/gates.json reported.end_of_A.final")
    N.put("owner.off_max", fin["DeltaT_over_dw_off_max"], f"{tgt.dir}/gates.json reported.end_of_A.final")
    N.put("owner.smoothed_share", gr["smoothed_game"]["smoothed_share_peak_gap_d0"], f"{tgt.dir}/gates.json reported.end_of_A.smoothed_game")
    N.put("owner.smoothed_pred0", gr["smoothed_game"]["smoothed_e_pred_0"], f"{tgt.dir}/gates.json reported.end_of_A.smoothed_game")
    N.put("owner.e2_at_0", fin["e2_at_0"], f"{tgt.dir}/gates.json reported.end_of_A.final")
    N.put("owner.g2_at_0", fin["g2_at_0"], f"{tgt.dir}/gates.json reported.end_of_A.final")
    N.put("owner.locfree_err", fin["stage2_peak_locfree_rel_err"], f"{tgt.dir}/gates.json reported.end_of_A.final")
    N.put("owner.locfree_argmax_d", fin["stage2_peak_locfree_argmax_d"], f"{tgt.dir}/gates.json reported.end_of_A.final")
    N.put("owner.sym_err_max", fin["stage2_sym_err_max"], f"{tgt.dir}/gates.json reported.end_of_A.final")
    N.put("owner.sigma0", fin["sigma_effort_at_0_t2"], f"{tgt.dir}/gates.json reported.end_of_A.final")
    N.put("owner.tail_max", fin["stage2_tail_max"], f"{tgt.dir}/gates.json reported.end_of_A.final")
    N.put("owner.rmse", tgt.gates["metric_values"]["rmse"], f"{tgt.dir}/gates.json metric_values")
    N.put("owner.tail_mean_over_g20", tgt.gates["metric_values"]["tail"], f"{tgt.dir}/gates.json metric_values")
    N.put("owner.outcome", tgt.gates["outcome"], f"{tgt.dir}/gates.json")
    quoted_ = {"peak_err": (-0.1458, 4, "owner.peak_err_signed"), "eta_on": (0.0058, 4, "owner.on_max"),
               "eta_off": (0.00082, 5, "owner.off_max"), "share": (0.279, 3, "owner.smoothed_share")}
    for nm_, (qv_, dg_, key_) in quoted_.items():
        N.put(f"owner.quoted.{nm_}", qv_, "number quoted in the request (not read from a file)")
        N.put(f"owner.match.{nm_}", bool(abs(round(N.values[key_], dg_) - qv_) < 1e-12), f"file value {key_} rounded to {dg_} digits equals the quoted value")
    N.put("owner.match.all", bool(all(N.values[f"owner.match.{nm_}"] for nm_ in quoted_)), "all four quoted numbers match")
    N.put("gates.G_A_eta_pass", bool(tgt.gates["G-A"]["criteria"][0]["pass"]), f"{tgt.dir}/gates.json G-A.criteria[0].pass")
    N.put("gates.G_A_rmse_pass", bool(tgt.gates["G-A"]["criteria"][1]["pass"]), f"{tgt.dir}/gates.json G-A.criteria[1].pass")
    N.put("gates.G_A_tail_pass", bool(tgt.gates["G-A"]["criteria"][2]["pass"]), f"{tgt.dir}/gates.json G-A.criteria[2].pass")
    ex_t = tgt.exports.set_index("u")
    last = ex_t.loc[1600]
    N.put("repro.e2_0_u1600_mine", float(last["e2_0"]), src_w)
    N.put("repro.peak_err_u1600_mine", float(last["peak_rel_err"]), src_w)
    N.put("repro.peak_err_absdiff_vs_gates", abs(float(last["peak_rel_err"]) - fin["stage2_peak_rel_err_signed"]), "mine vs gates.json")
    N.put("repro.locfree_u1600_mine", float(last["locfree_rel_err"]), src_w)
    N.put("repro.sigma0_u1600_mine", float(last["sigma0"]), src_w)
    N.put("repro.smooth_pred0_u1600_mine", float(last["smooth_pred0"]), src_w)
    N.put("repro.smooth_share_u1600_mine", float(last["smooth_share"]), src_w)
    zw = np.load(tgt.dir / "weights" / "u01600.npz")
    a_, b_ = actor_ab({k: zw[k] for k in zw.files}, obs_stage(2, tgt.D, tgt.spec["B"]))
    e_mine = 100.0 * a_.astype(float) / (a_.astype(float) + b_.astype(float))
    if verify_fn is not None:
        N.put("repro.on_max_u1600_dev_mine", float(last["v_on_max"]), "dev-tier verifier recomputation on u01600.npz (state step 4)")
        N.put("repro.off_max_u1600_dev_mine", float(last["v_off_max"]), "same")
        N.put("repro.on_argmax_u1600_dev_mine", float(last["v_on_argmax_d"]), "same")
        N.put("owner.on_max_dev", tgt.gates["reported"]["end_of_A"]["development"]["DeltaT_over_dw_on_max"], f"{tgt.dir}/gates.json reported.end_of_A.development")
        N.put("owner.off_max_dev", tgt.gates["reported"]["end_of_A"]["development"]["DeltaT_over_dw_off_max"], f"{tgt.dir}/gates.json reported.end_of_A.development")
    N.put("repro.recovery_e2_maxabsdiff_u1600", float(np.abs(e_mine - tgt.gA["recovery_e2"]).max()),
          f"mine (u01600.npz) vs {tgt.dir}/gateA_final.npz recovery_e2 (effort units)")
    # full-block reproduction of the reported end-of-A peak error and smoothed share from weights u1600
    d_pe, d_sh, d_sig = [], [], []
    for r in list(runs.values()):
        e = r.exports.set_index("u").loc[1600]
        fr = r.gates["reported"]["end_of_A"]
        d_pe.append(abs(float(e["peak_rel_err"]) - fr["final"]["stage2_peak_rel_err_signed"]))
        d_sh.append(abs(float(e["smooth_share"]) - fr["smoothed_game"]["smoothed_share_peak_gap_d0"]))
        d_sig.append(abs(float(e["sigma0"]) - fr["final"]["sigma_effort_at_0_t2"]))
    N.put("repro.block_max_absdiff_peak_err", max(d_pe), "mine (u01600 weights) vs gates.json, all 40 runs")
    N.put("repro.block_max_absdiff_smooth_share", max(d_sh), "mine (u01600 weights) vs gates.json, all 40 runs")
    N.put("repro.block_max_absdiff_sigma0", max(d_sig), "mine (u01600 weights) vs gates.json, all 40 runs")
    if verify_fn is not None:
        dd = []
        for r in runs.values():
            ex = r.exports.set_index("u")
            for _, c in r.ckpt.iterrows():
                if int(c["update"]) in ex.index:   # the one off-grid call (q60 seed 30505, u=540) has no export
                    dd.append(abs(float(ex.loc[int(c["update"]), "v_eta2"]) - float(c["eta_T_over_dw"])))
        N.put("repro.eta2_dev_max_absdiff_vs_csv", max(dd), "verifier recomputation vs v2_checkpoints_A.csv eta_T_over_dw, all runs, all Phase-A calls")
    N.put("meta.n_exports", len(EXPORT_US), "weights u00025..u01600 (every 25)")
    N.put("meta.g2_at_0_q50", tgt.spec["g20"], f"closed form DW/(4kq) = {tgt.spec['dw']}/(4*{tgt.spec['k']}*{tgt.spec['q']})")
    N.put("meta.dw", tgt.spec["dw"], f"{tgt.dir}/run_config.json")
    N.put("meta.k", tgt.spec["k"], f"{tgt.dir}/run_config.json")
    N.put("meta.g2_at_0_q60", R60[0].spec["g20"], f"{R60[0].dir}/run_config.json + closed form")

    # ---- 1. peak trajectory
    ex_all = pd.concat([r.exports for r in runs.values()], ignore_index=True)
    ex_all.to_csv(tabdir / "tab_exports_all_runs.csv", index=False)
    M0, ML = matrix(R50, "peak_rel_err"), matrix(R50, "locfree_rel_err")
    M0_60, ML_60 = matrix(R60, "peak_rel_err"), matrix(R60, "locfree_rel_err")
    out_rows = []
    for name, M in (("peak_rel_err", M0), ("locfree_rel_err", ML)):
        med, p10, p90 = band(M, oth50)
        rk = rank_lowest(M, i_t)
        for k, u in enumerate(us):
            out_rows.append({"q": TQ, "metric": name, "u": int(u), "target": M[i_t, k], "others_median": med[k],
                             "others_p10": p10[k], "others_p90": p90[k], "target_below_p10": bool(M[i_t, k] < p10[k]),
                             "target_rank_lowest_of_20": int(rk[k])})
    pd.DataFrame(out_rows).to_csv(tabdir / "tab_peak_trajectory_band_q50.csv", index=False)
    out60 = []
    for name, M in (("peak_rel_err", M0_60), ("locfree_rel_err", ML_60)):
        med, p10, p90 = band(M, all60)
        for k, u in enumerate(us):
            out60.append({"q": a.q_other, "metric": name, "u": int(u), "median": med[k], "p10": p10[k], "p90": p90[k],
                          "target_q50_s30510": (M0 if name == "peak_rel_err" else ML)[i_t, k]})
    pd.DataFrame(out60).to_csv(tabdir / "tab_peak_trajectory_band_q60.csv", index=False)

    ex0 = {}
    for name, M in (("e2_0", M0), ("locfree", ML)):
        med, p10, p90 = band(M, oth50)
        ex0[name] = exit_stats(M[i_t], p10, us)
        for kk, v in ex0[name].items():
            N.put(f"traj.{name}.{kk}", v, f"{src_w}; band = {int(BAND_LO)}/{int(BAND_HI)}th percentile of the 19 other q={TQ} seeds at the same update")
        rk = rank_lowest(M, i_t)
        N.put(f"traj.{name}.n_rank1", int((rk == 1).sum()), "rank 1 = lowest of the 20 q=50 runs at that export")
        N.put(f"traj.{name}.n_rank1_late", int((rk[(us >= LATE_WIN[0])] == 1).sum()), "exports u>=900")
        N.put(f"traj.{name}.n_rank_le2_late", int((rk[(us >= LATE_WIN[0])] <= 2).sum()), "exports u>=900")
        N.put(f"traj.{name}.median_rank_late", float(np.median(rk[us >= LATE_WIN[0]])), "exports u>=900")
        N.put(f"traj.{name}.n_in_band_u400plus", int(sum(1 for u_ in ex0[name]["in_band_us"] if u_ >= 400)), "exports u>=400 at or above the others' p10")
        N.put(f"traj.{name}.n_exports_u400plus", int((us >= 400).sum()), "")
        N.put(f"traj.{name}.n_exports_u900plus", int((us >= 900).sum()), "")
        N.put(f"traj.{name}.n_below_u900plus", int((M[i_t][us >= 900] < band(M, oth50)[1][us >= 900]).sum()), "exports u>=900 below the others' p10")
        for u in (100, 250, 325, 400, 600, 800, 825, 850, 900, 1200, 1600):
            k = EXPORT_US.index(u)
            N.put(f"traj.{name}.at_u{u}.rank", int(rk[k]), "rank of the target among the 20 q=50 runs at this export (1 = lowest)")
            N.put(f"traj.{name}.at_u{u}.target", M[i_t, k], src_w)
            N.put(f"traj.{name}.at_u{u}.others_median", med[k], src_w)
            N.put(f"traj.{name}.at_u{u}.others_p10", p10[k], src_w)
            N.put(f"traj.{name}.at_u{u}.others_p90", p90[k], src_w)
    for name, M in (("e2_0", M0_60), ("locfree", ML_60)):
        med, p10, p90 = band(M, all60)
        for u in (900, 1200, 1600):
            k = EXPORT_US.index(u)
            N.put(f"traj60.{name}.at_u{u}.median", med[k], src_w)
            N.put(f"traj60.{name}.at_u{u}.p10", p10[k], src_w)
            N.put(f"traj60.{name}.at_u{u}.p90", p90[k], src_w)
    # ---- PRE-REGISTERED RULES H1-H4 (literal). Target, then every q=50 run against the other 19, then every q=60 run
    # against the other 19 q=60 runs (how the rules behave on runs that passed).
    def rules_for(ms, ml, i):
        others_ = [j for j in range(ms.shape[0]) if j != i]
        return apply_rules(EXPORT_US, ms[i], ml[i], ms[others_], ml[others_])
    RT = rules_for(M0, ML, i_t)
    src_r = f"apply_rules() on the weight exports {src_w}; pack = the {len(oth50)} other q={TQ} runs; rule text: RULE_TEXT"
    for kk, vv in RT.items():
        N.put(f"rules.target.{kk}", vv, src_r)
    N.put("rules.target.u_leave_equals_sustained_exit", bool(RT["u_leave"] == N.values["traj.e2_0.sustained_exit_u"]),
          "consistency check: u_leave of the rules equals the 'sustained departure' of section 2.3")
    N.put("rules.n_pack", int(RT["n_pack"]), src_r)
    rule_rows = []
    for blk_q, ms, ml, runs_blk in ((TQ, M0, ML, R50), (a.q_other, M0_60, ML_60, R60)):
        for i_r, r_ in enumerate(runs_blk):
            rr = rules_for(ms, ml, i_r)
            rule_rows.append({"block_q": blk_q, "seed": r_.seed, "is_target": bool(blk_q == TQ and r_.seed == TS), "n_pack": rr["n_pack"],
                              "G_A_pass": bool(r_.gates["G-A"]["pass"]), "run_outcome": r_.gates["outcome"],
                              "S_1600": rr["S_1600"], "p10_pack_S_1600": rr["p10_S_1600"], "S_max_all_exports": rr["S_max_all"],
                              "u_of_S_max": rr["u_of_S_max"], "out_S_1600": rr["out_S_1600"], "u_leave": rr["u_leave"],
                              "plateau_exports_before_u_leave": rr["plateau_len"], "n_exports_out_S": rr["n_out_exports"],
                              "L_ge_p10_all_1225_1600": rr["L_ge_p10_all_1225_1600"], "n_L_below_p10_1225_1600": rr["n_L_below_p10_1225_1600"],
                              "cusp_rounding_L_minus_S_1600": rr["cusp_rounding_1600"], "p90_pack_cusp_1600": rr["p90_cusp_1600"],
                              "cusp_exceeds_p90": rr["cusp_exceeds_p90"], "H1": rr["verdict_H1"], "H2": rr["verdict_H2"],
                              "H3": rr["verdict_H3"], "H4": rr["verdict_H4"], "label": rr["label"]})
            tk = f"rules.table.q{blk_q}.s{r_.seed}"
            for kk in ("S_1600", "p10_S_1600", "S_max_all", "u_of_S_max", "out_S_1600", "u_leave", "plateau_len", "n_out_exports",
                       "L_ge_p10_all_1225_1600", "n_L_below_p10_1225_1600", "cusp_rounding_1600", "p90_cusp_1600", "cusp_exceeds_p90",
                       "verdict_H1", "verdict_H2", "verdict_H3", "verdict_H4", "label"):
                N.put(f"{tk}.{kk}", rr[kk], f"apply_rules() for this run against the other 19 runs of its block ({tabdir}/hypothesis_rules.csv)")
            N.put(f"{tk}.G_A_pass", bool(r_.gates["G-A"]["pass"]), f"{r_.dir}/gates.json G-A.pass")
    RULE_DF = pd.DataFrame(rule_rows)
    RULE_DF.to_csv(tabdir / "hypothesis_rules.csv", index=False)
    for blk_q in (TQ, a.q_other):
        sub = RULE_DF[RULE_DF.block_q == blk_q]
        N.put(f"rules.table.q{blk_q}.n_runs", int(len(sub)), f"{tabdir}/hypothesis_rules.csv")
        N.put(f"rules.table.q{blk_q}.label_counts", {k_: int(v_) for k_, v_ in sub["label"].value_counts().items()}, f"{tabdir}/hypothesis_rules.csv")
        for h_ in ("H1", "H2", "H3", "H4"):
            N.put(f"rules.table.q{blk_q}.n_{h_}_supported", int((sub[h_] == "supported").sum()), f"{tabdir}/hypothesis_rules.csv")
        N.put(f"rules.table.q{blk_q}.n_out_S_1600", int(sub["out_S_1600"].sum()), f"{tabdir}/hypothesis_rules.csv")
    # per-seed statistics (leave-one-out) -> is the target distinct from the pack
    ps50 = per_seed_stats(R50)
    ps60 = per_seed_stats(R60)
    pd.concat([ps50, ps60]).to_csv(tabdir / "tab_per_seed_trajectory_stats.csv", index=False)
    pt = ps50[ps50.seed == TS].iloc[0]
    po = ps50[ps50.seed != TS]
    for col in ("late_mean", "late_max", "late_sd", "dec_minus_pre", "late_slope_per100", "loo_frac_below_p10_late"):
        N.put(f"seedstat.{col}.target", float(pt[col]), f"{tabdir}/tab_per_seed_trajectory_stats.csv")
        N.put(f"seedstat.{col}.others_median", float(po[col].median()), "same, 19 other q=50 seeds")
        N.put(f"seedstat.{col}.others_p10", float(po[col].quantile(0.1)), "same")
        N.put(f"seedstat.{col}.others_p90", float(po[col].quantile(0.9)), "same")
        N.put(f"seedstat.{col}.others_min", float(po[col].min()), "same")
        N.put(f"seedstat.{col}.others_max", float(po[col].max()), "same")
        N.put(f"seedstat.{col}.rank_lowest_of_20", int((ps50[col] < pt[col]).sum() + 1), "1 = lowest value")
    N.put("seedstat.n_others_loo_frac_below_p10_late_ge_0p5", int((po["loo_frac_below_p10_late"] >= 0.5).sum()), "other seeds below own leave-one-out p10 in >= half of exports u>=900")
    N.put("seedstat.n_others_late_mean_lt_target", int((po["late_mean"] <= pt["late_mean"]).sum()), "others with a late mean at or below the target's")
    for nm, col in (("t_ge_m010", "first_u_ge_m010"),):
        N.put(f"seedstat.{nm}.target", pt[col], f"{tabdir}/tab_per_seed_trajectory_stats.csv")
        vals = po[col].dropna().astype(float)
        N.put(f"seedstat.{nm}.others_median", float(vals.median()), "same, 19 other q=50 seeds")
        N.put(f"seedstat.{nm}.others_p10", float(vals.quantile(0.1)), "same")
        N.put(f"seedstat.{nm}.others_p90", float(vals.quantile(0.9)), "same")
        N.put(f"seedstat.{nm}.others_max", float(vals.max()), "same")
        N.put(f"seedstat.{nm}.n_others_reaching", int(len(vals)), "same")
    # pack late level distribution (all exports 900..1600 pooled over the 19 others) and the target's best exports
    late = (us >= LATE_WIN[0])
    pool = M0[oth50][:, late].ravel()
    N.put("pool.late_p10", float(np.percentile(pool, 10)), "19 other q=50 seeds x exports u>=900 pooled")
    N.put("pool.late_median", float(np.median(pool)), "same")
    N.put("pool.late_p90", float(np.percentile(pool, 90)), "same")
    N.put("pool.late_min", float(pool.min()), "same")
    N.put("pool.frac_pool_below_target_late_mean", float((pool < pt["late_mean"]).mean()), "share of pooled other-seed exports below the target's late mean")
    N.put("pool.n_target_late_ge_pool_p10", int((M0[i_t, late] >= np.percentile(pool, 10)).sum()), "target exports u>=900 at or above the pooled p10")
    # LR-decay response of the pack vs target
    N.put("lr.pack_dec_minus_pre_median", float(po["dec_minus_pre"].median()), "others median of mean(1225..1600) - mean(900..1200)")

    # ---- 2. end-of-A profile
    contrast, med_peak = contrast_seeds(R50, TS, a.contrast_n)
    N.put("contrast.median_peak_err_others", med_peak, f"{ana}/reported_metrics.csv (19 others, A_stage2_peak_rel_err_signed)")
    prof_seeds = [TS] + contrast
    N.put("contrast.seeds", prof_seeds, "target + the others nearest to the median signed peak error")
    prof_rows, delta_rows, scal_rows = [], [], []
    for s in prof_seeds:
        r = runs[(TQ, s)]
        z = r.gA
        D, e2, g2 = z["recovery_d_grid"], z["recovery_e2"], z["recovery_g2"]
        for d_, e_, g_ in zip(D, e2, g2):
            prof_rows.append({"seed": s, "d": d_, "e2_hat": e_, "e2_star": g_, "err": e_ - g_})
        dG = z["v_t2_d_grid"]
        dl = z["v_t2_delta"] / r.spec["dw"]
        on = z["v_t2_onpath"].astype(bool)
        for i in range(dG.size):
            delta_rows.append({"seed": s, "d": dG[i], "delta2_over_dw": dl[i], "onpath": bool(on[i]), "sigma2_effort": z["v_t2_sigma_effort"][i],
                               "alpha": z["v_t2_alpha"][i], "beta": z["v_t2_beta"][i], "cell_mass": z["v_t2_cell_mass"][i]})
        jg = int(np.argmax(dl))
        jon = int(np.argmax(np.where(on, dl, -np.inf)))
        joff = int(np.argmax(np.where(~on, dl, -np.inf)))
        pos = np.abs(D) < 2.0 * r.spec["q"]
        j_t = int(np.argmax(np.where(~pos, e2, -np.inf)))
        sym = np.abs(e2 - e2[::-1])
        js = int(np.argmax(sym))
        zero = int(np.nonzero(D == 0.0)[0][0])
        zg = int(np.nonzero(dG == 0.0)[0][0])
        sc = {"seed": s, "peak_rel_err": float((e2[zero] - g2[zero]) / g2[zero]), "e2_0": float(e2[zero]),
              "locfree_rel_err": float((e2.max() - g2[zero]) / g2[zero]), "locfree_argmax_d": float(D[int(np.argmax(e2))]),
              "eta2_over_dw": float(dl.max()), "eta2_argmax_d": float(dG[jg]), "on_max": float(dl[jon]), "on_argmax_d": float(dG[jon]),
              "off_max": float(dl[joff]), "off_argmax_d": float(dG[joff]), "sym_err_max": float(sym[js]),
              "sym_err_argmax_abs_d": float(abs(D[js])), "sigma2_at_0": float(z["v_t2_sigma_effort"][zg]),
              "sigma2_mean_pos": float(z["v_t2_sigma_effort"][np.abs(dG) < 2.0 * r.spec["q"]].mean()),
              "tail_max": float(e2[j_t]), "tail_argmax_d": float(D[j_t]),
              "rmse_pos_over_g20": float(np.sqrt(np.mean((e2[pos] - g2[pos]) ** 2)) / g2[zero]),
              "tail_mean_over_g20": float(e2[~pos].mean() / g2[zero]),
              "err_mean_pos": float(np.mean(e2[pos] - g2[pos])), "err_mean_abs_pos": float(np.mean(np.abs(e2[pos] - g2[pos]))),
              "err_at_pm10": float(0.5 * ((e2[np.argmin(np.abs(D - 10))] - g2[np.argmin(np.abs(D - 10))]) + (e2[np.argmin(np.abs(D + 10))] - g2[np.argmin(np.abs(D + 10))]))),
              "err_at_pm20": float(0.5 * ((e2[np.argmin(np.abs(D - 20))] - g2[np.argmin(np.abs(D - 20))]) + (e2[np.argmin(np.abs(D + 20))] - g2[np.argmin(np.abs(D + 20))]))),
              "frac_pos_nodes_below_target": float(np.mean((e2[pos] - g2[pos]) < 0.0))}
        scal_rows.append(sc)
        for kk, v in sc.items():
            if kk != "seed":
                N.put(f"prof.s{s}.{kk}", v, f"{r.dir}/gateA_final.npz (recovery + final-tier verifier arrays)")
    pd.DataFrame(prof_rows).to_csv(tabdir / "tab_endA_profile_recovery_grid.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(tabdir / "tab_endA_profile_verifier_grid.csv", index=False)
    pd.DataFrame(scal_rows).to_csv(tabdir / "tab_endA_profile_scalars.csv", index=False)

    # end-of-A scalars of all 40 runs (reported by the runs themselves) + ranks of the target
    allsc = []
    for r in runs.values():
        fr = r.gates["reported"]["end_of_A"]
        f_, sm = fr["final"], fr["smoothed_game"]
        zA = r.gA
        dlA = zA["v_t2_delta"] / r.spec["dw"]
        onA = zA["v_t2_onpath"].astype(bool)
        jA, jonA = int(np.argmax(dlA)), int(np.argmax(np.where(onA, dlA, -np.inf)))
        allsc.append({"q": r.q, "seed": r.seed, "eta_argmax_d": float(zA["v_t2_d_grid"][jA]),
                      "on_argmax_d": float(zA["v_t2_d_grid"][jonA]), "g2_at_0": f_["g2_at_0"],
                      "sigma2_mean_pos": float(zA["v_t2_sigma_effort"][np.abs(zA["v_t2_d_grid"]) < 2.0 * r.spec["q"]].mean()), "e2_at_0": f_["e2_at_0"], "peak_rel_err": f_["stage2_peak_rel_err_signed"],
                      "locfree_rel_err": f_["stage2_peak_locfree_rel_err"], "locfree_argmax_d": f_["stage2_peak_locfree_argmax_d"],
                      "sym_err_max": f_["stage2_sym_err_max"], "tail_max": f_["stage2_tail_max"], "eta2_over_dw": f_["eta_T_over_dw"],
                      "on_max": f_["DeltaT_over_dw_on_max"], "off_max": f_["DeltaT_over_dw_off_max"], "sigma0": f_["sigma_effort_at_0_t2"],
                      "rmse_over_g20": f_["stage2_rmse_pos_over_g2_0"], "tail_mean_over_g20": f_["stage2_tail_mean_over_g2_0"],
                      "smooth_pred0": sm["smoothed_e_pred_0"], "smooth_share": sm["smoothed_share_peak_gap_d0"],
                      "run_pass": bool(r.gates["run_pass"]), "outcome": r.gates["outcome"], "G_A_pass": bool(r.gates["G-A"]["pass"])})
    ALL = pd.DataFrame(allsc)
    ALL["rl_gap"] = ALL["g2_at_0"] - ALL["e2_at_0"]
    ALL["smooth_gap"] = ALL["g2_at_0"] - ALL["smooth_pred0"]
    ALL["remainder_gap"] = ALL["smooth_pred0"] - ALL["e2_at_0"]
    ALL.to_csv(tabdir / "tab_endA_scalars_all_runs.csv", index=False)
    A50 = ALL[ALL.q == TQ]
    At = A50[A50.seed == TS].iloc[0]
    Ao = A50[A50.seed != TS]
    for col in ("peak_rel_err", "locfree_rel_err", "eta2_over_dw", "on_max", "off_max", "sigma0", "sym_err_max", "tail_max",
                "rmse_over_g20", "tail_mean_over_g20", "smooth_share", "rl_gap", "smooth_gap", "remainder_gap", "smooth_pred0",
                "sigma2_mean_pos"):
        N.put(f"end.{col}.target", float(At[col]), f"{tabdir}/tab_endA_scalars_all_runs.csv (gates.json reported.end_of_A)")
        N.put(f"end.{col}.others_median", float(Ao[col].median()), "19 other q=50 seeds")
        N.put(f"end.{col}.others_p10", float(Ao[col].quantile(0.1)), "same")
        N.put(f"end.{col}.others_p90", float(Ao[col].quantile(0.9)), "same")
        N.put(f"end.{col}.others_min", float(Ao[col].min()), "same")
        N.put(f"end.{col}.others_max", float(Ao[col].max()), "same")
        N.put(f"end.{col}.rank_lowest_of_20", int((A50[col] < At[col]).sum() + 1), "1 = lowest")
    N.put("end.n_runs_pass_G_A_q50_others", int(Ao["G_A_pass"].sum()), "gates.json G-A.pass")
    others_all = ALL[~((ALL.q == TQ) & (ALL.seed == TS))]
    N.put("end.pooled40.peak_err_target_rank_lowest_of_40", int((ALL["peak_rel_err"] < At["peak_rel_err"]).sum() + 1), "all 40 runs")
    nxt = others_all.sort_values("peak_rel_err").iloc[0]
    N.put("end.pooled40.next_lowest_peak_err", float(nxt["peak_rel_err"]), "all other 39 runs")
    N.put("end.pooled40.next_lowest_q", int(nxt["q"]), "")
    N.put("end.pooled40.next_lowest_seed", int(nxt["seed"]), "")
    N.put("end.pooled40.next_lowest_eta2", float(nxt["eta2_over_dw"]), "")
    N.put("end.pooled40.next_lowest_G_A_pass", bool(nxt["G_A_pass"]), "G-A verdict of that run (gates.json)")
    from scipy.stats import spearmanr
    rho, pv = spearmanr(ALL["peak_rel_err"], ALL["on_max"])
    N.put("end.pooled40.spearman_peak_err_vs_on_max", float(rho), "40 runs, tab_endA_scalars_all_runs.csv (descriptive)")
    rho50, _ = spearmanr(A50["peak_rel_err"], A50["on_max"])
    N.put("end.q50.spearman_peak_err_vs_on_max", float(rho50), "20 q=50 runs")
    N.put("end.n_runs_pooled", int(len(ALL)), "")
    N.put("end.on_argmax_d.target", float(At["on_argmax_d"]), "final-tier verifier grid (gateA_final.npz v_t2_delta)")
    N.put("end.on_argmax_abs_d.others_median", float(Ao["on_argmax_d"].abs().median()), "19 other q=50 seeds")
    N.put("end.on_argmax_abs_d.others_min", float(Ao["on_argmax_d"].abs().min()), "")
    N.put("end.on_argmax_abs_d.others_max", float(Ao["on_argmax_d"].abs().max()), "")
    N.put("end.n_others_on_argmax_abs_le20", int((Ao["on_argmax_d"].abs() <= 20).sum()), "other q=50 seeds with the on-path eta_2 argmax within |d|<=20")
    A60_ = ALL[ALL.q == a.q_other]
    N.put("end.n_q60_on_argmax_abs_le20", int((A60_["on_argmax_d"].abs() <= 20).sum()), "q=60 runs with the on-path eta_2 argmax within |d|<=20")
    N.put("end.q60_on_argmax_abs_median", float(A60_["on_argmax_d"].abs().median()), "")

    # ---- 3. optimisation
    series_upd = [("kl", "kl_final_epoch", "upd"), ("clip_frac", "clip_frac", "upd"), ("adv_sd", "adv_all_std", "upd"),
                  ("policy_loss", "policy_loss", "upd"), ("value_loss", "value_loss", "upd"),
                  ("grad_norm_actor_mean", "grad_norm_actor_mean", "hist"), ("grad_norm_actor_max", "grad_norm_actor_max", "hist"),
                  ("grad_norm_critic_mean", "grad_norm_critic_mean", "hist"), ("entropy_effort_scale", "entropy_post_update_effort_scale", "hist"),
                  ("conc_buf_mean", "conc_buf_mean", "hist"), ("eff_stage2_batch_mean", "eff_stage2_batch_mean", "hist"),
                  ("mean_episode_return", "mean_episode_return", "hist")]
    wt = window_table(list(runs.values()), series_upd, 100)
    wt.to_csv(tabdir / "tab_opt_window100_all_runs.csv", index=False)
    opt_export = {}
    opt_rows = []
    for name, col, frame in series_upd:
        Mx = np.stack([trailing_mean((r.upd if frame == "upd" else r.hist)[col].to_numpy(float), 25)[np.array(EXPORT_US) - 1] for r in R50])
        opt_export[name] = Mx
        med, p10, p90 = band(Mx, oth50)
        rk = rank_lowest(Mx, i_t)
        for k, u in enumerate(us):
            opt_rows.append({"series": name, "u": int(u), "target": Mx[i_t, k], "others_median": med[k], "others_p10": p10[k],
                             "others_p90": p90[k], "target_rank_lowest_of_20": int(rk[k])})
    for name, col in (("conc0", "conc0"), ("sigma0", "sigma0")):
        Mx = matrix(R50, col)
        opt_export[name] = Mx
        med, p10, p90 = band(Mx, oth50)
        rk = rank_lowest(Mx, i_t)
        for k, u in enumerate(us):
            opt_rows.append({"series": name, "u": int(u), "target": Mx[i_t, k], "others_median": med[k], "others_p10": p10[k],
                             "others_p90": p90[k], "target_rank_lowest_of_20": int(rk[k])})
    pd.DataFrame(opt_rows).to_csv(tabdir / "tab_opt_trailing25_band_q50.csv", index=False)
    tgt.upd.merge(tgt.hist, on="update", suffixes=("", "_h")).to_csv(tabdir / "tab_opt_target_per_update.csv", index=False)
    wlate = [(901, 1200), (1201, 1600), (1, 100), (101, 400), (1, 400), (401, 900)]
    for name, col, frame in series_upd:
        w = wt[(wt.series == name) & (wt.q == TQ)]
        for lo, hi in wlate:
            sel = w[(w.w_first >= lo) & (w.w_last <= hi)].groupby("seed")["mean"].mean()
            N.put(f"opt.{name}.w{lo}_{hi}.target", float(sel.loc[TS]), f"{tabdir}/tab_opt_window100_all_runs.csv")
            o = sel.drop(TS)
            N.put(f"opt.{name}.w{lo}_{hi}.others_median", float(o.median()), "19 other q=50 seeds")
            N.put(f"opt.{name}.w{lo}_{hi}.others_p10", float(o.quantile(0.1)), "")
            N.put(f"opt.{name}.w{lo}_{hi}.others_p90", float(o.quantile(0.9)), "")
            N.put(f"opt.{name}.w{lo}_{hi}.rank_lowest_of_20", int((sel < sel.loc[TS]).sum() + 1), "1 = lowest")
    for name in ("conc0", "sigma0"):
        Mx = opt_export[name]
        for lo, hi in ((900, 1600), (1, 400)):
            m = (us >= lo) & (us <= hi)
            v = Mx[:, m].mean(1)
            N.put(f"opt.{name}.u{lo}_{hi}.target", float(v[i_t]), src_w)
            N.put(f"opt.{name}.u{lo}_{hi}.others_median", float(np.median(v[oth50])), "19 other q=50 seeds")
            N.put(f"opt.{name}.u{lo}_{hi}.others_p10", float(np.percentile(v[oth50], 10)), "")
            N.put(f"opt.{name}.u{lo}_{hi}.others_p90", float(np.percentile(v[oth50], 90)), "")
            N.put(f"opt.{name}.u{lo}_{hi}.rank_lowest_of_20", int((v < v[i_t]).sum() + 1), "1 = lowest")
    # when does the optimisation footprint leave the band of the others (same rule as for the peak)
    sides = {"kl": "above", "clip_frac": "above", "grad_norm_actor_mean": "above", "adv_sd": "above", "value_loss": "above",
             "conc0": "below", "sigma0": "above"}
    for nm, side in sides.items():
        med, p10, p90 = band(opt_export[nm], oth50)
        z = opt_export[nm][i_t]
        ee = exit_stats(z, p10, us) if side == "below" else exit_stats(-z, -p90, us)
        for kk in ("frac_below", "first_below_u", "sustained_exit_u", "run8_start_u", "frac_below_late", "last_in_band_u", "n_in_band"):
            N.put(f"optexit.{nm}.{kk}", ee[kk], f"{tabdir}/tab_opt_trailing25_band_q50.csv; beyond-band side = {side} the {int(BAND_HI) if side == 'above' else int(BAND_LO)}th percentile of the 19 others")
    # cross-seed association between the late level of the peak error and the optimisation / clamp footprint (descriptive)
    lw = wt[(wt.q == TQ) & (wt.w_first >= 901)].groupby(["seed", "series"])["mean"].mean().unstack()
    lw["conc0"] = [float(R50[i].exports.loc[R50[i].exports.u >= 900, "conc0"].mean()) for i in range(len(R50))]
    lw["sigma0"] = [float(R50[i].exports.loc[R50[i].exports.u >= 900, "sigma0"].mean()) for i in range(len(R50))]
    lw["peak_late_mean"] = [float(R50[i].exports.loc[R50[i].exports.u >= 900, "peak_rel_err"].mean()) for i in range(len(R50))]
    lw.to_csv(tabdir / "tab_late_means_per_seed_q50.csv")
    from scipy.stats import spearmanr as _sp
    for nm in ("kl", "clip_frac", "grad_norm_actor_mean", "adv_sd", "value_loss", "conc0", "sigma0", "entropy_effort_scale", "conc_buf_mean"):
        rho_all = float(_sp(lw["peak_late_mean"], lw[nm])[0])
        rho_oth = float(_sp(lw.drop(TS)["peak_late_mean"], lw.drop(TS)[nm])[0])
        N.put(f"corr.late_peak_vs_{nm}.all20", rho_all, f"Spearman across the 20 q=50 seeds of the mean over u>=900 ({tabdir}/tab_late_means_per_seed_q50.csv); descriptive, not causal")
        N.put(f"corr.late_peak_vs_{nm}.others19", rho_oth, "same without the target")
    # fraction of the actor-gradient norms above the clip level (mean pre-clip norm vs 0.5)
    N.put("opt.max_grad_norm", float(tgt.cfg["record"]["ppo"]["max_grad_norm"]), f"{tgt.dir}/run_config.json")
    N.put("opt.n_minibatch_steps_per_update", int(tgt.hist["n_minibatch_steps"].iloc[0]), f"{tgt.dir}/train_history.json history[0]")

    # ---- 3b. verifier calls (dev tier) and would_have_fired
    ck_rows = []
    for r in list(runs.values()):
        c = r.ckpt
        for _, x in c.iterrows():
            ck_rows.append({"q": r.q, "seed": r.seed, "u": int(x["update"]), "reason": x["reason"], "eta2_over_dw": x["eta_T_over_dw"],
                            "on_max": x["DeltaT_over_dw_on_max"], "on_argmax_d": x["DeltaT_over_dw_on_argmax_d"],
                            "off_max": x["DeltaT_over_dw_off_max"], "off_argmax_d": x["DeltaT_over_dw_off_argmax_d"],
                            "peak_rel_err": x["stage2_peak_rel_err_signed"], "eligible": bool(x["eligible"]),
                            "conc_max_std_norm": x["conc_max_std_norm"], "phase_criterion_value_over_dw": x["phase_criterion_value_over_dw"],
                            "consecutive_eligible": int(x["consecutive_eligible"]), "valid": bool(x["valid"])})
    CK = pd.DataFrame(ck_rows)
    CK.to_csv(tabdir / "tab_eta2_verifier_calls_all_runs.csv", index=False)
    calls = sorted(CK[CK.q == TQ]["u"].unique())
    Ce = CK[CK.q == TQ].pivot(index="seed", columns="u", values="eta2_over_dw")
    Ce_on = CK[CK.q == TQ].pivot(index="seed", columns="u", values="on_max")
    for u in calls:
        v = Ce[u]
        N.put(f"calls.eta2.u{u}.target", float(v.loc[TS]), f"{root}/q{TQ}/seed*/v2_checkpoints_A.csv eta_T_over_dw (dev tier)")
        N.put(f"calls.eta2.u{u}.others_median", float(v.drop(TS).median()), "19 other q=50 seeds")
        N.put(f"calls.eta2.u{u}.others_max", float(v.drop(TS).max()), "")
        N.put(f"calls.eta2.u{u}.others_min", float(v.drop(TS).min()), "")
        N.put(f"calls.eta2.u{u}.rank_lowest_of_20", int((v < v.loc[TS]).sum() + 1), "1 = lowest")
    tc = CK[(CK.q == TQ) & (CK.seed == TS) & (CK.u >= 400)]
    N.put("calls.target.eta2_min_u400plus", float(tc["eta2_over_dw"].min()), "target, calls u>=400")
    N.put("calls.target.eta2_max_u400plus", float(tc["eta2_over_dw"].max()), "target, calls u>=400")
    N.put("calls.target.n_calls_gt_0p005_u400plus", int((tc["eta2_over_dw"] > 0.005).sum()), "target, calls u>=400 (dev tier)")
    N.put("calls.target.n_calls_u400plus", int(len(tc)), "")
    oc = CK[(CK.q == TQ) & (CK.seed != TS) & (CK.u >= 400)]
    N.put("calls.others.frac_calls_gt_0p005_u400plus", float((oc["eta2_over_dw"] > 0.005).mean()), "19 others, calls u>=400")
    N.put("calls.others.max_eta2_u400plus", float(oc["eta2_over_dw"].max()), "")
    N.put("calls.others.n_runs_with_any_call_gt_0p005_u400plus", int(oc[oc["eta2_over_dw"] > 0.005]["seed"].nunique()), "")
    seeds_any = sorted(oc[oc["eta2_over_dw"] > 0.005]["seed"].unique().tolist())
    N.put("calls.others.seeds_with_any_call_gt_0p005_u400plus", seeds_any, "")
    ocl = CK[(CK.q == TQ) & (CK.seed != TS) & (CK.u == 1600)]
    N.put("calls.others.eta2_u1600_max", float(ocl["eta2_over_dw"].max()), "19 others, call at u=1600")
    N.put("calls.others.eta2_u1600_median", float(ocl["eta2_over_dw"].median()), "")
    N.put("calls.target.on_gt_off_all_calls_u400plus", bool((tc["on_max"] > tc["off_max"]).all()), "target on-path max > off-path max at every call u>=400")
    N.put("calls.target.n_calls_total", int(len(calls)), "common verifier calls (u=100..1600 step 100)")
    ncalls = CK.groupby(["q", "seed"]).size()
    N.put("meta.n_calls_A_per_run_min", int(ncalls.min()), f"{root}/q*/seed*/v2_checkpoints_A.csv row counts")
    N.put("meta.n_calls_A_per_run_max", int(ncalls.max()), "same")
    N.put("meta.n_runs_with_extra_call", [[int(q_), int(s_)] for (q_, s_), n_ in ncalls.items() if n_ != ncalls.min()], "runs with a non-standard number of Phase-A calls")
    N.put("calls.target.n_calls_highest_of_20", int(sum(1 for u in calls if (Ce[u] < Ce[u].loc[TS]).sum() == len(seeds) - 1)), "calls at which the target has the largest eta_2 of the 20 q=50 runs")
    N.put("calls.target.n_calls_above_others_median", int(sum(1 for u in calls if Ce[u].loc[TS] > Ce[u].drop(TS).median())), "")
    N.put("calls.target.n_calls_above_others_max", int(sum(1 for u in calls if Ce[u].loc[TS] > Ce[u].drop(TS).max())), "")
    N.put("calls.target.n_on_argmax_abs_le20_u400plus", int((tc["on_argmax_d"].abs() <= 20).sum()), "target calls u>=400 whose on-path eta_2 argmax lies within |d|<=20 (dev grid)")
    N.put("calls.target.on_argmax_us_u400plus", [[int(u), float(d)] for u, d in zip(tc["u"], tc["on_argmax_d"])], "target (u, on-path argmax d) at every call u>=400")
    N.put("calls.others.frac_on_argmax_abs_le20_u400plus", float((oc["on_argmax_d"].abs() <= 20).mean()), "19 others, calls u>=400")
    N.put("calls.others.n_runs_any_on_argmax_abs_le20_u400plus", int(oc[oc["on_argmax_d"].abs() <= 20]["seed"].nunique()), "")
    ocl6 = CK[(CK.q == a.q_other) & (CK.u >= 400)]
    N.put("calls.q60.frac_on_argmax_abs_le20_u400plus", float((ocl6["on_argmax_d"].abs() <= 20).mean()), "20 q=60 runs, calls u>=400")
    wf = {}
    for r in R50:
        wh = r.summ["would_have_fired"]["A"]
        wf[r.seed] = None if wh is None else int(wh["global_update"])
    N.put("wf.target_A_global_update", wf[TS], f"{tgt.dir}/v2_run_summary.json would_have_fired.A")
    wf_o = [v for s, v in wf.items() if s != TS and v is not None]
    N.put("wf.others_A_median", float(np.median(wf_o)), f"{root}/q{TQ}/seed*/v2_run_summary.json")
    N.put("wf.others_A_min", float(np.min(wf_o)), "")
    N.put("wf.others_A_max", float(np.max(wf_o)), "")
    N.put("wf.others_A_n_none", int(sum(v is None for s, v in wf.items() if s != TS)), "")
    N.put("wf.target_G_A_pass", bool(tgt.gates["G-A"]["pass"]), f"{tgt.dir}/gates.json")
    N.put("wf.phase_A_rule_threshold_over_dw", float(tgt.cfg["record"]["protocol"]["phase_thr_over_dw"]["A"]), f"{tgt.dir}/run_config.json (phase_thr_over_dw.A)")
    N.put("wf.k_phase", int(tgt.cfg["record"]["protocol"]["k_phase"]), f"{tgt.dir}/run_config.json")
    N.put("wf.target_exit_reason_A", tgt.summ["phases_done"], "")
    pd.DataFrame([{"seed": s, "would_have_fired_A": v} for s, v in wf.items()]).to_csv(tabdir / "tab_would_have_fired_q50.csv", index=False)
    # what the Phase-A eligibility test is made of: eta_2/DW <= 0.02 AND max normalised std <= conc_thr, 3 calls in a row
    elig_thr = float(tgt.cfg["record"]["protocol"]["conc_thr"])
    N.put("elig.conc_thr_normalised", elig_thr, f"{tgt.dir}/run_config.json record.protocol.conc_thr (x 100 = effort units)")
    C50 = CK[CK.q == TQ]
    fe = C50[C50["eligible"]].groupby("seed")["u"].min()
    N.put("elig.target_first_eligible_call_u", int(fe.loc[TS]), "first Phase-A call with eligible=True (v2_checkpoints_A.csv)")
    N.put("elig.others_first_eligible_call_median", float(fe.drop(TS).median()), "19 others")
    N.put("elig.others_first_eligible_call_min", int(fe.drop(TS).min()), "")
    N.put("elig.others_first_eligible_call_max", int(fe.drop(TS).max()), "")
    ct2 = C50[C50.seed == TS]
    N.put("elig.target_n_calls_eta_le_0p02_but_not_eligible", int(((ct2["phase_criterion_value_over_dw"] <= 0.02) & (~ct2["eligible"])).sum()), "target calls with eta_2/DW <= 0.02 that are not eligible")
    N.put("elig.target_n_calls_not_eligible_with_conc_over_thr", int(((~ct2["eligible"]) & (ct2["conc_max_std_norm"] > elig_thr)).sum()), "target calls not eligible with max normalised std > threshold")
    N.put("elig.target_n_calls_not_eligible", int((~ct2["eligible"]).sum()), "")
    N.put("elig.target_n_noneligible_eta_ok_noise_ok", int(((~ct2["eligible"]) & (ct2["phase_criterion_value_over_dw"] <= 0.02) & (ct2["conc_max_std_norm"] <= elig_thr)).sum()),
          "target non-eligible calls with eta_2/DW <= 0.02 and max normalised std <= conc_thr")
    def last_above(df):
        """Update of the last call at which the max normalised std is above the threshold (None if never)."""
        d_ = df[df["conc_max_std_norm"] > elig_thr]
        return int(d_["u"].max()) if len(d_) else None
    la = {s: last_above(C50[C50.seed == s]) for s in seeds}
    N.put("elig.target_last_call_conc_above_thr", la[TS], "last Phase-A call at which the max over the D2 grid of the normalised std exceeds conc_thr")
    lo_ = [v for s, v in la.items() if s != TS and v is not None]
    N.put("elig.others_last_call_conc_above_thr_median", float(np.median(lo_)), "19 others")
    N.put("elig.others_last_call_conc_above_thr_max", int(max(lo_)), "")
    N.put("elig.target_first_call_eta_le_0p02", int(ct2[ct2["phase_criterion_value_over_dw"] <= 0.02]["u"].min()), "first call with eta_2/DW <= 0.02 (br_pass)")
    fb = C50[C50["phase_criterion_value_over_dw"] <= 0.02].groupby("seed")["u"].min()
    N.put("elig.others_first_call_eta_le_0p02_median", float(fb.drop(TS).median()), "19 others")
    N.put("elig.others_first_call_eta_le_0p02_max", int(fb.drop(TS).max()), "")
    for u in range(100, 1601, 100):
        v = C50[C50.u == u].set_index("seed")["conc_max_std_norm"] * 100.0
        N.put(f"elig.conc_std.u{u}.target", float(v.loc[TS]), "max over the D2 dev grid of the normalised std x 100 (effort units), call at this update")
        N.put(f"elig.conc_std.u{u}.others_median", float(v.drop(TS).median()), "")
        N.put(f"elig.conc_std.u{u}.others_max", float(v.drop(TS).max()), "")
    N.put("meta.G_A_rmse_threshold", float(tgt.gates["G-A"]["criteria"][1]["threshold"]), f"{tgt.dir}/gates.json G-A.criteria[1]")
    N.put("meta.G_A_tail_threshold", float(tgt.gates["G-A"]["criteria"][2]["threshold"]), f"{tgt.dir}/gates.json G-A.criteria[2]")
    off_grid = CK[(CK.u % 100 != 0) | (~CK.reason.isin(["timeout", "warmup_forced"]))]
    N.put("meta.nonstandard_calls", [[int(r_.q), int(r_.seed), int(r_.u), str(r_.reason)] for r_ in off_grid.itertuples()], "Phase-A calls that are not on the u=100k grid or whose reason is not timeout/warmup_forced")
    # per-export verifier eta_2 (optional)
    if verify_fn is not None:
        Mv = matrix(R50, "v_eta2")
        Mv_on = matrix(R50, "v_on_max")
        med, p10, p90 = band(Mv, oth50)
        pd.DataFrame({"u": us, "target": Mv[i_t], "target_on_max": Mv_on[i_t], "target_off_max": matrix(R50, "v_off_max")[i_t],
                      "others_median": med, "others_p10": p10, "others_p90": p90,
                      "target_rank_lowest_of_20": rank_lowest(Mv, i_t)}).to_csv(tabdir / "tab_eta2_every_export_q50.csv", index=False)
        late = us >= 400
        N.put("eta_exp.target_min_u400plus", float(Mv[i_t, late].min()), "dev-tier eta_2/DW recomputed at each export u>=400 (verifier import)")
        N.put("eta_exp.target_max_u400plus", float(Mv[i_t, late].max()), "same")
        N.put("eta_exp.target_argmax_u_u400plus", int(us[late][np.argmax(Mv[i_t, late])]), "export at which the target's eta_2 (u>=400) is largest")
        k825 = EXPORT_US.index(825)
        N.put("eta_exp.target_at_u825", float(Mv[i_t, k825]), "dev-tier eta_2/DW at the export u=825 (the excursion of e2_hat(0) to the pack's median)")
        N.put("eta_exp.others_median_at_u825", float(np.median(Mv[oth50, k825])), "19 others at the same export")
        N.put("eta_exp.target_median_u900plus", float(np.median(Mv[i_t, us >= 900])), "same")
        N.put("eta_exp.others_median_u900plus", float(np.median(Mv[oth50][:, us >= 900])), "pooled 19 others, exports u>=900")
        N.put("eta_exp.others_p90_u900plus", float(np.percentile(Mv[oth50][:, us >= 900], 90)), "same")
        N.put("eta_exp.others_p99_u900plus", float(np.percentile(Mv[oth50][:, us >= 900], 99)), "same")
        N.put("eta_exp.others_max_u900plus", float(Mv[oth50][:, us >= 900].max()), "same")
        N.put("eta_exp.target_n_gt_0p005_u900plus", int((Mv[i_t, us >= 900] > 0.005).sum()), "same")
        N.put("eta_exp.target_n_u900plus", int((us >= 900).sum()), "")
        N.put("eta_exp.others_frac_gt_0p005_u900plus", float((Mv[oth50][:, us >= 900] > 0.005).mean()), "pooled 19 others")
        N.put("eta_exp.target_frac_below_p10_u900plus", float((Mv[i_t] < p10)[us >= 900].mean()), "target below the others' p10 at exports u>=900")
        N.put("eta_exp.target_frac_above_p90_u900plus", float((Mv[i_t] > p90)[us >= 900].mean()), "target above the others' p90 at exports u>=900")
        N.put("eta_exp.target_on_gt_off_frac", float((Mv_on[i_t, late] > matrix(R50, "v_off_max")[i_t, late]).mean()), "")
        # relation of the per-export eta_2 to the per-export peak error within the pack (Spearman, pooled u>=900)
        xs, ys = M0[:, us >= 900].ravel(), Mv[:, us >= 900].ravel()
        N.put("eta_exp.spearman_peakerr_vs_eta2_pooled_u900plus", float(spearmanr(xs, ys)[0]), "20 q=50 runs x exports u>=900 (descriptive)")

    # ---- 4. sampling
    edges, ids = peak_bins(float(TQ))
    vis_rows, clamp_rows = [], []
    for r in list(runs.values()):
        e_, ids_ = peak_bins(r.spec["q"])
        nb = len(e_) - 1
        p_exp = len(ids_) / nb
        for v in r.vis:
            tot = v["counts"].sum()
            sh = v["counts"][ids_].sum() / tot
            sd = np.sqrt(p_exp * (1 - p_exp) / tot)
            vis_rows.append({"q": r.q, "seed": r.seed, "u": v["update"], "n_rows": int(tot), "peak_share": sh, "expected": p_exp,
                             "z_binomial": (sh - p_exp) / sd, "min_bin_share": float(v["counts"].min() / tot), "max_bin_share": float(v["counts"].max() / tot)})
        u_ = r.upd
        row = {"q": r.q, "seed": r.seed}
        for col in u_.columns:
            if col.startswith("d1_") and col != "d1_pol_alpha_min" and col != "d1_pol_beta_min":
                row[col + "_sum"] = int(u_[col].sum())
        row["d1_pol_alpha_min_overall"] = float(u_["d1_pol_alpha_min"].min())
        row["d1_pol_beta_min_overall"] = float(u_["d1_pol_beta_min"].min())
        row["n_updates"] = int(len(u_))
        clamp_rows.append(row)
    VIS = pd.DataFrame(vis_rows)
    VIS.to_csv(tabdir / "tab_visitation_peak_share_by_call.csv", index=False)
    CL = pd.DataFrame(clamp_rows)
    CL.to_csv(tabdir / "tab_d1_clamp_counts_phaseA_sum.csv", index=False)
    cnt_rows = []
    for r in runs.values():
        v = r.vis[-1]
        cnt_rows.append({"q": r.q, "seed": r.seed, "u": v["update"], **{f"bin{i}": int(c) for i, c in enumerate(v["counts"])}})
    pd.DataFrame(cnt_rows).to_csv(tabdir / "tab_visitation_bin_counts_u1600.csv", index=False)
    Vf = VIS[VIS.u == 1600]
    Vt = Vf[(Vf.q == TQ) & (Vf.seed == TS)].iloc[0]
    Vo = Vf[(Vf.q == TQ) & (Vf.seed != TS)]
    N.put("vis.peak_bins_q50", [int(i) for i in ids], "bin indices intersecting (-20,20); edges = linspace(-200,200,41)")
    N.put("vis.n_bins_q50", int(len(edges) - 1), "")
    N.put("vis.peak_share.target", float(Vt["peak_share"]), f"{tabdir}/tab_visitation_peak_share_by_call.csv (train_history.json verifier_calls[A].visitation_cumulative_phase.stage2_direct_es)")
    N.put("vis.peak_share.expected", float(Vt["expected"]), "4 bins / 40 bins (balanced exploring starts: bin uniform, then uniform inside)")
    N.put("vis.peak_share.n_rows", int(Vt["n_rows"]), "")
    N.put("vis.peak_share.z_target", float(Vt["z_binomial"]), "(share - 0.1) / sqrt(0.1*0.9/n_rows)")
    N.put("vis.peak_share.others_median", float(Vo["peak_share"].median()), "19 others at u=1600")
    N.put("vis.peak_share.others_min", float(Vo["peak_share"].min()), "")
    N.put("vis.peak_share.others_max", float(Vo["peak_share"].max()), "")
    N.put("vis.peak_share.others_z_min", float(Vo["z_binomial"].min()), "")
    N.put("vis.peak_share.others_z_max", float(Vo["z_binomial"].max()), "")
    N.put("vis.peak_share.rank_lowest_of_20", int((Vf[Vf.q == TQ]["peak_share"] < Vt["peak_share"]).sum() + 1), "")
    for q in (TQ, a.q_other):
        zz = Vf[Vf.q == q]["z_binomial"]
        N.put(f"vis.z_sd_q{q}", float(zz.std(ddof=1)), "SD of the 20 binomial z-scores at u=1600 (about 1 if only sampling noise)")
        N.put(f"vis.z_absmax_q{q}", float(zz.abs().max()), "")
    N.put("vis.target_absz_max_over_calls", float(VIS[(VIS.q == TQ) & (VIS.seed == TS)]["z_binomial"].abs().max()), "largest |binomial z| of the cumulative peak share over the 16 Phase-A calls")
    N.put("vis.target_min_bin_share", float(Vt["min_bin_share"]), "")
    N.put("vis.target_max_bin_share", float(Vt["max_bin_share"]), "")
    ct = CL[(CL.q == TQ) & (CL.seed == TS)].iloc[0]
    co = CL[(CL.q == TQ) & (CL.seed != TS)]
    for col in ("d1_L_s2_n_sum", "d1_L_s2_lo_sum", "d1_L_s2_hi_sum", "d1_O_s2_lo_sum", "d1_O_s2_hi_sum", "d1_L_s2_in_n_sum",
                "d1_L_s2_in_lo_sum", "d1_L_s2_in_hi_sum", "d1_L_s2_out_lo_sum", "d1_L_s2_out_hi_sum", "d1_pol_n_alpha_lt1_sum",
                "d1_pol_n_beta_lt1_sum"):
        N.put(f"clamp.{col}.target", int(ct[col]), f"{tabdir}/tab_d1_clamp_counts_phaseA_sum.csv (v2_updates.csv Phase-A rows)")
        N.put(f"clamp.{col}.others_max", int(co[col].max()), "19 others")
        N.put(f"clamp.{col}.others_sum", int(co[col].sum()), "19 others")
    C50 = CL[CL.q == TQ]
    for col in ("d1_L_s2_lo_sum", "d1_O_s2_lo_sum", "d1_pol_n_alpha_lt1_sum"):
        N.put(f"clamp.{col}.others_median", float(co[col].median()), "19 others")
        N.put(f"clamp.{col}.others_min", int(co[col].min()), "19 others")
        N.put(f"clamp.{col}.rank_lowest_of_20", int((C50[col] < ct[col]).sum() + 1), "1 = lowest")
        N.put(f"clamp.{col}.target_over_others_median", float(ct[col] / co[col].median()), "")
    N.put("clamp.lo_frac_of_learner_rows.target", float(ct["d1_L_s2_lo_sum"] / ct["d1_L_s2_n_sum"]), "")
    N.put("clamp.lo_frac_of_out_rows.target", float(ct["d1_L_s2_out_lo_sum"] / (ct["d1_L_s2_n_sum"] - ct["d1_L_s2_in_n_sum"])), "")
    N.put("clamp.alpha_lt1_frac_of_rows.target", float(ct["d1_pol_n_alpha_lt1_sum"] / ct["d1_L_s2_n_sum"]), "")
    N.put("clamp.alpha_lt1_frac_of_rows.others_median", float((co["d1_pol_n_alpha_lt1_sum"] / co["d1_L_s2_n_sum"]).median()), "")
    N.put("clamp.n_q50_others_with_lo_gt_20000", int((co["d1_L_s2_lo_sum"] > 20000).sum()), "")
    C60 = CL[CL.q == a.q_other]
    N.put("clamp.q60.d1_L_s2_lo_sum.max", int(C60["d1_L_s2_lo_sum"].max()), "20 q=60 runs")
    N.put("clamp.q60.d1_L_s2_lo_sum.median", float(C60["d1_L_s2_lo_sum"].median()), "20 q=60 runs")
    pk = ALL.set_index(["q", "seed"])["peak_rel_err"]
    cl_ = CL.set_index(["q", "seed"])["d1_L_s2_lo_sum"]
    N.put("clamp.spearman_peak_err_vs_lo_sum.pooled40", float(spearmanr(pk.loc[cl_.index], cl_)[0]), "40 runs (descriptive, not causal)")
    k50 = [(TQ, s) for s in seeds]
    N.put("clamp.spearman_peak_err_vs_lo_sum.q50", float(spearmanr(pk.loc[k50], cl_.loc[k50])[0]), "20 q=50 runs")
    k50o = [(TQ, s) for s in seeds if s != TS]
    N.put("clamp.spearman_peak_err_vs_lo_sum.q50_others", float(spearmanr(pk.loc[k50o], cl_.loc[k50o])[0]), "19 other q=50 runs")
    N.put("clamp.pol_alpha_min_target", float(ct["d1_pol_alpha_min_overall"]), "min over Phase A of d1_pol_alpha_min")
    N.put("clamp.pol_alpha_min_others_min", float(co["d1_pol_alpha_min_overall"].min()), "")
    N.put("clamp.pol_beta_min_target", float(ct["d1_pol_beta_min_overall"]), "")
    N.put("clamp.pol_beta_min_others_min", float(co["d1_pol_beta_min_overall"].min()), "")
    N.put("clamp.all40_total_lo_hi", int(CL[[c for c in CL.columns if c.endswith("_lo_sum") or c.endswith("_hi_sum")]].to_numpy().sum()), "sum of every d1 lo/hi count, all 40 runs, Phase A")
    N.put("clamp.all40_total_hi", int(CL[[c for c in CL.columns if c.endswith("_hi_sum")]].to_numpy().sum()), "sum of every upper-clamp count (learner, opponent, stage 1 and 2), all 40 runs")
    N.put("clamp.all40_inside_support_total", int(CL[["d1_L_s2_in_lo_sum", "d1_L_s2_in_hi_sum"]].to_numpy().sum()), "learner stage-2 clamp hits at |d|<2q, all 40 runs")

    # ---- 5. decomposition (RL gap, smoothing-predicted gap, supervised floor)
    floor = None
    if a.floor_dir:
        fd = Path(a.floor_dir)
        fits = pd.read_csv(fd / "fits.csv")
        f50 = fits[fits.q == TQ]
        floor = (f50["g2_at_0"] - f50["e2_at_0"]).to_numpy(float)
        N.put("floor.gap_d0_median", float(np.median(floor)), f"{fd}/fits.csv (g2_at_0 - e2_at_0, 5 inits, q={TQ})")
        N.put("floor.gap_d0_min", float(floor.min()), "same")
        N.put("floor.gap_d0_max", float(floor.max()), "same")
        N.put("floor.gap_d0_absmax", float(np.abs(floor).max()), "same")
        N.put("floor.n_fits", int(len(floor)), "same")
        N.put("floor.fit_steps_all_cap", bool((f50["stop"] == "max_steps").all()), "same (plateau rule never fired: upper bounds on the floor)")
        tw = pd.read_csv(fd / "three_way_peak_gap.csv")
        for _, x in tw[tw.q == TQ].iterrows():
            N.put(f"floor.pilot4.{x['quantity']}.median", float(x["median"]), f"{fd}/three_way_peak_gap.csv (pilot-4 Phase-A extension u1600, 10 seeds)")
            N.put(f"floor.pilot4.{x['quantity']}.min", float(x["min"]), "same")
            N.put(f"floor.pilot4.{x['quantity']}.max", float(x["max"]), "same")
    dec = A50.copy()
    dec["floor_median"] = float(np.median(floor)) if floor is not None else np.nan
    dec.to_csv(tabdir / "tab_decomposition_q50.csv", index=False)
    ALL.to_csv(tabdir / "tab_decomposition_all_runs.csv", index=False)
    N.put("dec.remainder_over_gap.target", float(At["remainder_gap"] / At["rl_gap"]), "= 1 - smoothed share")
    N.put("dec.remainder_over_rl_gap.others_median", float((Ao["remainder_gap"] / Ao["rl_gap"]).median()), "")
    N.put("dec.target_remainder_over_others_median_remainder", float(At["remainder_gap"] / Ao["remainder_gap"].median()), "")
    N.put("dec.target_smooth_gap_over_others_median_smooth_gap", float(At["smooth_gap"] / Ao["smooth_gap"].median()), "")
    N.put("dec.target_rl_gap_over_others_median_rl_gap", float(At["rl_gap"] / Ao["rl_gap"].median()), "")
    N.put("dec.rl_gap_over_g20.target", float(At["rl_gap"] / At["g2_at_0"]), "")
    if floor is not None:
        N.put("dec.rl_gap_over_floor_median", float(At["rl_gap"] / abs(np.median(floor))), "ratio, floor median 5 inits (an upper bound on the floor)")
    A60 = ALL[ALL.q == a.q_other]
    for col in ("rl_gap", "smooth_gap", "remainder_gap", "smooth_share"):
        N.put(f"dec60.{col}.median", float(A60[col].median()), "20 q=60 runs")
        N.put(f"dec60.{col}.p10", float(A60[col].quantile(0.1)), "")
        N.put(f"dec60.{col}.p90", float(A60[col].quantile(0.9)), "")
        N.put(f"dec60.{col}.max", float(A60[col].max()), "")
    # across-seed regression of the RL gap on the smoothing-predicted gap (descriptive; the 19 others)
    xs, ys = Ao["smooth_gap"].to_numpy(), Ao["rl_gap"].to_numpy()
    sl, ic = np.polyfit(xs, ys, 1)
    N.put("dec.fit_others.slope", float(sl), "OLS rl_gap ~ smooth_gap, 19 other q=50 seeds (descriptive)")
    N.put("dec.fit_others.intercept", float(ic), "")
    N.put("dec.fit_others.pred_for_target", float(sl * At["smooth_gap"] + ic), "")
    N.put("dec.fit_others.resid_target", float(At["rl_gap"] - (sl * At["smooth_gap"] + ic)), "")
    res_o = ys - (sl * xs + ic)
    N.put("dec.fit_others.resid_sd", float(res_o.std(ddof=2)), "residual SD of the fit on the 19 others (ddof=2)")
    N.put("dec.fit_others.resid_target_over_sd", float((At["rl_gap"] - (sl * At["smooth_gap"] + ic)) / res_o.std(ddof=2)), "")

    # ---- 6. hypotheses (operational rules, applied to e2_hat(0))
    e0 = ex0["e2_0"]
    pe = M0[i_t]
    med_, p10_, p90_ = band(M0, oth50)
    H = {}
    # H1: late-phase form: below the p10 edge at every export from 900 on; strict form: below at every export after the initial transient (u>=400)
    H["H1_late_all_below"] = bool(e0["frac_below_late"] == 1.0)
    H["H1_strict_all_below_u400plus"] = bool(np.all(pe[us >= 400] < p10_[us >= 400]))
    H["H2_in_band_at_1200_and_out_by_1600"] = bool((pe[us == 1200][0] >= p10_[us == 1200][0]) and (pe[us == 1600][0] < p10_[us == 1600][0]))
    H["H2_frac_in_band_900_1200"] = float(np.mean(pe[(us >= 900) & (us <= 1200)] >= p10_[(us >= 900) & (us <= 1200)]))
    H["H2_last_in_band_u"] = e0["last_in_band_u"]
    H["H3_in_band_plateau_then_drop"] = bool(H["H2_frac_in_band_900_1200"] >= 0.5 and e0["frac_below_decay"] > 0.5)
    for kk, v in H.items():
        N.put(f"hyp.{kk}", v, "rule applied to e2_hat(0) trajectory, see report")
    jump = np.abs(np.diff(pe))
    N.put("hyp.target_max_abs_step_per25_u900plus", float(jump[us[1:] >= 900].max()), "max |e2_hat(0) rel. err change between consecutive exports|, u>=900")
    pj = np.array([np.abs(np.diff(M0[i]))[us[1:] >= 900].max() for i in oth50])
    N.put("hyp.others_max_abs_step_per25_u900plus_median", float(np.median(pj)), "median over the 19 others of their max step, u>=900")
    N.put("hyp.others_max_abs_step_per25_u900plus_max", float(pj.max()), "")
    sd25 = np.array([np.diff(M0[i])[us[1:] >= 900].std(ddof=1) for i in range(len(seeds))])
    N.put("hyp.step_sd_u900plus.target", float(sd25[i_t]), "SD of consecutive-export changes, u>=900")
    N.put("hyp.step_sd_u900plus.others_median", float(np.median(sd25[oth50])), "")
    N.put("hyp.step_sd_u900plus.others_max", float(np.max(sd25[oth50])), "")
    # shoulder (is the whole peak region low, or only the cusp?)
    for nm in ("shoulder10_rel_err", "shoulder20_rel_err", "shoulder40_rel_err"):
        Ms = matrix(R50, nm)
        v = Ms[:, us == 1600][:, 0]
        N.put(f"shoulder.{nm}.u1600.target", float(v[i_t]), src_w)
        N.put(f"shoulder.{nm}.u1600.others_median", float(np.median(v[oth50])), "")
        N.put(f"shoulder.{nm}.u1600.others_p10", float(np.percentile(v[oth50], 10)), "")
        N.put(f"shoulder.{nm}.u1600.others_p90", float(np.percentile(v[oth50], 90)), "")
    N.put("shoulder.cusp_minus_shoulder10.target", float(last["peak_rel_err"] - last["shoulder10_rel_err"]), "e2_hat(0) rel err minus mean(e2_hat(+-10)) rel err, u1600")
    cs = matrix(R50, "peak_rel_err")[:, us == 1600][:, 0] - matrix(R50, "shoulder10_rel_err")[:, us == 1600][:, 0]
    N.put("shoulder.cusp_minus_shoulder10.others_median", float(np.median(cs[oth50])), "")
    N.put("shoulder.cusp_minus_shoulder10.others_p10", float(np.percentile(cs[oth50], 10)), "")
    N.put("shoulder.cusp_minus_shoulder10.others_p90", float(np.percentile(cs[oth50], 90)), "")
    N.put("shoulder.cusp_minus_shoulder10.rank_lowest_of_20", int((cs < cs[i_t]).sum() + 1), "")
    # mean level shifts: mean peak error u>=900 ; target's first ascent (u<=325): did it track?
    early = (us >= 25) & (us <= 325)
    N.put("early.target_mean_dev_from_median_u25_325", float((pe[early] - med_[early]).mean()), "mean over exports 25..325 of target - others median")
    N.put("early.target_u250_rank", int(rank_lowest(M0, i_t)[us == 250][0]), "")
    N.put("early.target_u325_rank", int(rank_lowest(M0, i_t)[us == 325][0]), "")
    N.put("early.target_best_post400_u", int(us[(us >= 400)][np.argmax(pe[us >= 400])]), "export (u>=400) with the highest target e2_hat(0)")
    N.put("early.target_best_post400_val", float(pe[us >= 400].max()), "")
    N.put("early.target_best_post400_others_median_at_same_u", float(med_[us == int(us[(us >= 400)][np.argmax(pe[us >= 400])])][0]), "")
    N.put("early.target_best_late_val", float(pe[us >= 900].max()), "best target export u>=900")
    N.put("early.target_best_late_u", int(us[us >= 900][np.argmax(pe[us >= 900])]), "")
    N.put("early.target_post900_median", float(np.median(pe[us >= 900])), "")
    N.put("early.pack_p10_median_u900plus", float(np.median(p10_[us >= 900])), "median over exports u>=900 of the others' p10")
    N.put("early.pack_median_median_u900plus", float(np.median(med_[us >= 900])), "")
    N.put("early.pack_p90_median_u900plus", float(np.median(p90_[us >= 900])), "")

    # ---- figures
    figs = []
    # Fig 1: peak trajectory q=50
    fig, axs = plt.subplots(3, 2, figsize=(11.5, 10), gridspec_kw={"height_ratios": [1.1, 1.1, 0.7]})
    for c, (M, ttl) in enumerate(((M0, r"$\hat e_2(0)$"), (ML, r"location-free peak $\max_d \hat e_2(d)$"))):
        traj_panel(axs[0, c], us, M, i_t, oth50, f"q={TQ} seed {TS}", ylab=f"signed relative error vs $e_2^*(0)$ = {tgt.spec['g20']:.0f}",
                   others_label=f"other {len(oth50)} q={TQ} seeds")
        axs[0, c].set_ylim(-0.85, 0.05)
        axs[0, c].set_title(f"(a{c + 1}) full Phase A: {ttl}")
        traj_panel(axs[1, c], us, M, i_t, oth50, f"q={TQ} seed {TS}", zoom=(250, -0.27, 0.03), ylab="same, zoom",
                   others_label=f"other {len(oth50)} q={TQ} seeds")
        axs[1, c].set_title(f"(b{c + 1}) zoom, update >= 250")
        rk = rank_lowest(M, i_t)
        axs[2, c].step(us, rk, where="mid", color=C_ORANGE, lw=1.6)
        axs[2, c].axhline(1.0, color="#8a8984", lw=0.6)
        axs[2, c].axvline(LR_DECAY_FIRST - 0.5, color=C_INK2, lw=0.8, ls=":")
        axs[2, c].set_ylim(20.8, 0.4)
        axs[2, c].set_xlim(0, 1625)
        axs[2, c].set_ylabel(f"rank of seed {TS} among 20\n(1 = lowest)")
        axs[2, c].set_xlabel("update (Phase A)")
        axs[2, c].set_title(f"(c{c + 1}) rank")
        style(axs[2, c])
    exs = ex0["e2_0"]
    for ax in axs[0:2, 0]:
        if exs["sustained_exit_u"]:
            ax.axvline(exs["sustained_exit_u"] - 12.5, color=C_ORANGE, lw=0.9, ls="--")
    axs[1, 0].text(exs["sustained_exit_u"] - 20 if exs["sustained_exit_u"] else 900, 0.012, f"never back in band\nfrom u={exs['sustained_exit_u']}",
                   color=C_ORANGE, ha="right", va="top", fontsize=7.5)
    axs[1, 0].text(LR_DECAY_FIRST + 10, -0.262, "LR decay\n1201-1600", color=C_INK2, fontsize=7.5, va="bottom")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, figdir, "fig1_peak_trajectory_q50")
    # Fig 2: peak trajectory vs q=60 pack
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.2))
    for c, (M6, M5, ttl) in enumerate(((M0_60, M0, r"$\hat e_2(0)$"), (ML_60, ML, r"location-free peak"))):
        med50 = np.median(M5[oth50], 0)
        traj_panel(axs[0, c], us, M6, None, all60, "", ylab="signed relative error vs $e_2^*(0)$", others_label=f"all {len(all60)} q={a.q_other} runs")
        axs[0, c].plot(us, M5[i_t], color=C_ORANGE, lw=1.8, marker="o", ms=2.6, zorder=4, label=f"q={TQ} seed {TS}")
        axs[0, c].plot(us, med50, color=C_INK2, lw=1.2, ls="--", zorder=3, label=f"median of the other q={TQ} seeds")
        axs[0, c].set_ylim(-0.85, 0.05)
        axs[0, c].set_title(f"(a{c + 1}) full Phase A: {ttl}")
        traj_panel(axs[1, c], us, M6, None, all60, "", zoom=(250, -0.27, 0.03), ylab="same, zoom", others_label=f"all {len(all60)} q={a.q_other} runs")
        axs[1, c].plot(us, M5[i_t], color=C_ORANGE, lw=1.8, marker="o", ms=2.6, zorder=4)
        axs[1, c].plot(us, med50, color=C_INK2, lw=1.2, ls="--", zorder=3)
        axs[1, c].set_title(f"(b{c + 1}) zoom, update >= 250")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, figdir, "fig2_peak_trajectory_q60_reference")
    # Fig 3: end-of-A profile
    cols = {TS: C_ORANGE, contrast[0]: C_AQUA, contrast[1]: C_VIOLET} if len(contrast) >= 2 else {TS: C_ORANGE}
    lss = {TS: "-", contrast[0]: "--", contrast[1]: ":"} if len(contrast) >= 2 else {TS: "-"}
    fig, axs = plt.subplots(3, 2, figsize=(11.5, 11))
    PR = pd.read_csv(tabdir / "tab_endA_profile_recovery_grid.csv")
    DL = pd.read_csv(tabdir / "tab_endA_profile_verifier_grid.csv")
    q_ = tgt.spec["q"]
    gstar = PR[PR.seed == TS]
    axs[0, 0].plot(gstar["d"], gstar["e2_star"], color=C_INK, lw=1.2, ls="-", label=r"closed form $e_2^*(d)$")
    for s in cols:
        p = PR[PR.seed == s]
        axs[0, 0].plot(p["d"], p["e2_hat"], color=cols[s], lw=1.6, ls=lss[s], label=f"seed {s}: peak err {float(N.values[f'prof.s{s}.peak_rel_err']):+.3f}")
        axs[0, 1].plot(p["d"], p["err"], color=cols[s], lw=1.2, ls=lss[s], label=f"seed {s}")
        axs[1, 0].plot(p["d"], p["err"], color=cols[s], lw=1.4, ls=lss[s])
    axs[0, 0].set_xlim(-2.2 * q_, 2.2 * q_)
    axs[0, 0].set_xlabel("stage-2 gap d")
    axs[0, 0].set_ylabel("effort")
    axs[0, 0].set_title(r"(a) $\hat e_2(d)$ against $e_2^*(d)$, end of Phase A (u1600)")
    axs[0, 0].legend(frameon=False, loc="upper right")
    axs[0, 1].axhline(0, color="#8a8984", lw=0.6)
    axs[0, 1].set_xlabel("stage-2 gap d")
    axs[0, 1].set_ylabel(r"$\hat e_2(d) - e_2^*(d)$  (effort)")
    axs[0, 1].set_title(r"(b) error on the whole recovery grid D$_2$")
    axs[1, 0].axhline(0, color="#8a8984", lw=0.6)
    axs[1, 0].set_xlim(-45, 45)
    axs[1, 0].set_ylim(-18, 6)
    axs[1, 0].set_xlabel("stage-2 gap d")
    axs[1, 0].set_ylabel(r"$\hat e_2(d) - e_2^*(d)$  (effort)")
    axs[1, 0].set_title("(c) error near the cusp (|d| <= 45)")
    for s in cols:
        p = DL[DL.seed == s]
        j = int(np.argmax(p["delta2_over_dw"].to_numpy()))
        axs[1, 1].plot(p["d"], p["delta2_over_dw"], color=cols[s], lw=1.4, ls=lss[s], marker="o", ms=2.2,
                       label=f"seed {s}: max {p['delta2_over_dw'].iloc[j]:.4f} at d* = {p['d'].iloc[j]:g}")
        axs[1, 1].plot([p["d"].iloc[j]], [p["delta2_over_dw"].iloc[j]], marker="v", color=cols[s], ms=7, mec="white", mew=0.8, zorder=5)
    axs[1, 1].legend(frameon=False, loc="center left", fontsize=7.5)
    axs[1, 1].axvspan(-2.2 * q_, -2 * q_, color="#ececE8", lw=0)
    axs[1, 1].axvspan(2 * q_, 2.2 * q_, color="#ececE8", lw=0)
    axs[1, 1].axhline(float(N.values["owner.eta2_threshold"]), color=C_INK2, lw=0.9, ls="--")
    axs[1, 1].text(2.2 * q_ - 3, float(N.values["owner.eta2_threshold"]) + 0.0001, "G-A threshold 0.005", ha="right", va="bottom", fontsize=7.5, color=C_INK2)
    axs[1, 1].set_xlim(-2.2 * q_, 2.2 * q_)
    axs[1, 1].set_xlabel("stage-2 gap d (shaded: off-path, |d| >= 2q)")
    axs[1, 1].set_ylabel(r"$\Delta_2(d)/\Delta W$ (final-tier verifier grid)")
    axs[1, 1].set_title(r"(d) one-step deviation gain $\Delta_2(d)$; marker = argmax $d^*$")
    for s in cols:
        p = DL[DL.seed == s]
        axs[2, 0].plot(p["d"], p["sigma2_effort"], color=cols[s], lw=1.4, ls=lss[s])
        sym = PR[PR.seed == s]
        dpos = sym["d"].to_numpy()
        e = sym["e2_hat"].to_numpy()
        m = dpos >= 0
        axs[2, 1].plot(dpos[m], np.abs(e[m] - e[::-1][m]), color=cols[s], lw=1.1, ls=lss[s], label=f"seed {s}: max {float(N.values[f'prof.s{s}.sym_err_max']):.2f}")
    axs[2, 0].set_xlim(-2.2 * q_, 2.2 * q_)
    axs[2, 0].set_xlabel("stage-2 gap d")
    axs[2, 0].set_ylabel(r"$\sigma_2(d)$ (std of effort, effort units)")
    axs[2, 0].set_title(r"(e) policy noise $\sigma_2(d)$; at d=0 target: " + f"{float(N.values[f'prof.s{TS}.sigma2_at_0']):.2f}")
    axs[2, 1].set_xlabel("|d|")
    axs[2, 1].set_ylabel(r"$|\hat e_2(d) - \hat e_2(-d)|$  (effort)")
    axs[2, 1].set_title("(f) symmetry error")
    axs[2, 1].legend(frameon=False, loc="upper right")
    for ax in axs.ravel():
        style(ax)
    fig.tight_layout()
    save(fig, figdir, "fig3_endA_profile")
    # Fig 3b: profile evolution near the peak, target vs pack median at u = 400, 800, 1200, 1600
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    dgrid = tgt.D
    sel = np.abs(dgrid) <= 40
    for ax, u in zip(axs, (400, 800, 1200, 1600)):
        eo = np.stack([runs[(TQ, s)].e_at[u] for s in seeds if s != TS])
        lo, hi = np.percentile(eo, [BAND_LO, BAND_HI], 0)
        ax.fill_between(dgrid[sel], lo[sel], hi[sel], color=C_BAND, alpha=0.75, lw=0, label="10-90% band, other seeds")
        ax.plot(dgrid[sel], np.median(eo, 0)[sel], color=C_BLUE, lw=1.6, label="median, other seeds")
        ax.plot(dgrid[sel], tgt.e_at[u][sel], color=C_ORANGE, lw=1.8, label=f"seed {TS}")
        ax.plot(dgrid[sel], tgt.g2[sel], color=C_INK, lw=1.0, ls="-", label=r"$e_2^*(d)$")
        ax.set_title(f"u = {u}")
        ax.set_xlabel("stage-2 gap d")
        style(ax)
    axs[0].set_ylabel(r"$\hat e_2(d)$ (effort)")
    axs[0].legend(frameon=False, loc="lower center", fontsize=7)
    fig.tight_layout()
    save(fig, figdir, "fig3b_profile_evolution_near_peak")
    prof_ev = []
    for u in (400, 800, 1200, 1600):
        eo = np.stack([runs[(TQ, s)].e_at[u] for s in seeds if s != TS])
        lo, hi = np.percentile(eo, [BAND_LO, BAND_HI], 0)
        for i in np.nonzero(sel)[0]:
            prof_ev.append({"u": u, "d": dgrid[i], "target": tgt.e_at[u][i], "others_median": float(np.median(eo[:, i])), "others_p10": lo[i],
                            "others_p90": hi[i], "e2_star": tgt.g2[i]})
    pd.DataFrame(prof_ev).to_csv(tabdir / "tab_profile_evolution_near_peak.csv", index=False)
    pe_df = pd.DataFrame(prof_ev)
    for u in (400, 800, 1200, 1600):
        r0 = pe_df[(pe_df.u == u) & (pe_df.d == 0.0)].iloc[0]
        N.put(f"profev.u{u}.target_e2_0", float(r0["target"]), f"{src_w} (effort units)")
        N.put(f"profev.u{u}.others_median_e2_0", float(r0["others_median"]), "19 other q=50 seeds")
        N.put(f"profev.u{u}.others_p10_e2_0", float(r0["others_p10"]), "")
        N.put(f"profev.u{u}.others_p90_e2_0", float(r0["others_p90"]), "")
    # Fig 4: optimisation
    panels = [("kl", "KL after the 10 epochs (kl_final_epoch)"), ("clip_frac", "clip fraction"),
              ("grad_norm_actor_mean", "actor grad norm, pre-clip mean (clip level 0.5)"), ("adv_sd", "advantage SD (raw, before normalisation)"),
              ("conc0", r"concentration $\alpha+\beta$ at d = 0"), ("sigma0", r"$\sigma_2(0)$ (effort units)")]
    fig, axs = plt.subplots(3, 2, figsize=(11.5, 9.5), sharex=True)
    for ax, (nm, ttl) in zip(axs.ravel(), panels):
        Mx = opt_export[nm]
        traj_panel(ax, us, Mx, i_t, oth50, f"q={TQ} seed {TS}", others_label=f"other {len(oth50)} q={TQ} seeds", zero=False)
        ax.set_title(ttl)
        ax.set_ylabel("")
        if nm == "kl":
            ax.set_yscale("log")
    for ax in axs[-1]:
        ax.set_xlabel("update (25-update trailing mean; concentration and sigma: weight export)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, figdir, "fig4_optimisation_q50")
    # Fig 5: eta_2
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.8))
    axs = axs.ravel()
    ax = axs[0]
    for s in seeds:
        if s == TS:
            continue
        c_ = CK[(CK.q == TQ) & (CK.seed == s)]
        ax.plot(c_["u"], c_["eta2_over_dw"], color=C_GREY, lw=0.5, alpha=0.4)
    Cm = Ce.drop(TS)
    ax.fill_between(Cm.columns, Cm.quantile(0.1), Cm.quantile(0.9), color=C_BAND, alpha=0.75, lw=0, label="10-90% band, other seeds")
    ax.plot(Cm.columns, Cm.median(), color=C_BLUE, lw=1.8, label="median, other seeds")
    ct_ = CK[(CK.q == TQ) & (CK.seed == TS)]
    ax.plot(ct_["u"], ct_["eta2_over_dw"], color=C_ORANGE, lw=1.8, marker="o", ms=3.5, label=f"seed {TS}")
    ax.axhline(0.005, color=C_INK2, lw=0.9, ls="--")
    ax.text(1590, 0.0053, "G-A threshold 0.005", ha="right", fontsize=7.5, color=C_INK2)
    ax.axhline(0.02, color=C_INK2, lw=0.9, ls=":")
    ax.text(1590, 0.0215, "Phase-A eligibility 0.02", ha="right", fontsize=7.5, color=C_INK2)
    ax.axvline(wf[TS] - 0, color=C_ORANGE, lw=0.8, ls=":")
    ax.text(wf[TS] + 15, 0.00055, f"would-have-fired\n(k_phase), u={wf[TS]}", color=C_ORANGE, fontsize=7.5, va="bottom")
    ax.set_ylim(0.0005, 0.2)
    ax.set_yscale("log")
    ax.set_xlabel("update (verifier call)")
    ax.set_ylabel(r"dev-tier $\eta_2/\Delta W$")
    ax.set_title(r"(a) $\eta_2/\Delta W$ at every Phase-A verifier call")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    style(ax)
    ax = axs[1]
    ax.plot(ct_["u"], ct_["on_max"], color=C_ORANGE, lw=1.8, marker="o", ms=3.5, label="on-path max, target")
    ax.plot(ct_["u"], ct_["off_max"], color=C_ORANGE, lw=1.4, ls="--", marker="s", ms=3, label="off-path max, target")
    cc = CK[(CK.q == TQ) & (CK.seed != TS)]
    gb = cc.groupby("u")
    ax.fill_between(gb["on_max"].median().index, gb["on_max"].quantile(0.1), gb["on_max"].quantile(0.9), color=C_BAND, alpha=0.75, lw=0)
    ax.plot(gb["on_max"].median().index, gb["on_max"].median(), color=C_BLUE, lw=1.6, label="on-path max, others median (10-90% band)")
    ax.plot(gb["off_max"].median().index, gb["off_max"].median(), color=C_BLUE, lw=1.2, ls="--", label="off-path max, others median")
    ax.axhline(0.005, color=C_INK2, lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xlim(300, 1620)
    ax.set_xlabel("update (verifier call)")
    ax.set_title(r"(b) on-path vs off-path max of $\Delta_2/\Delta W$ (dev tier)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    style(ax)
    if verify_fn is not None:
        ax = axs[2]
        Mv = matrix(R50, "v_eta2")
        traj_panel(ax, us, Mv, i_t, oth50, f"q={TQ} seed {TS}", zoom=(300, 0.0, 0.0145), others_label=f"other {len(oth50)} q={TQ} seeds", zero=False)
        ax.axhline(0.005, color=C_INK2, lw=0.9, ls="--")
        ax.set_title(r"(c) $\eta_2/\Delta W$ recomputed at every export (dev tier)")
        ax.set_ylabel(r"dev-tier $\eta_2/\Delta W$")
    else:
        axs[2].axis("off")
    ax = axs[3]
    for s in seeds:
        if s == TS:
            continue
        c_ = CK[(CK.q == TQ) & (CK.seed == s)]
        ax.plot(c_["u"], 100.0 * c_["conc_max_std_norm"], color=C_GREY, lw=0.5, alpha=0.4)
    Cs = CK[(CK.q == TQ) & (CK.seed != TS)].pivot(index="seed", columns="u", values="conc_max_std_norm") * 100.0
    ax.fill_between(Cs.columns, Cs.quantile(0.1), Cs.quantile(0.9), color=C_BAND, alpha=0.75, lw=0, label="10-90% band, other seeds")
    ax.plot(Cs.columns, Cs.median(), color=C_BLUE, lw=1.8, label="median, other seeds")
    ax.plot(ct_["u"], 100.0 * ct_["conc_max_std_norm"], color=C_ORANGE, lw=1.8, marker="o", ms=3.5, label=f"seed {TS}")
    ax.axhline(100.0 * elig_thr, color=C_INK2, lw=0.9, ls="--")
    ax.text(1590, 100.0 * elig_thr + 0.08, f"eligibility threshold {100.0 * elig_thr:.1f} effort units", ha="right", va="bottom", fontsize=7.5, color=C_INK2)
    ax.set_xlabel("update (verifier call)")
    ax.set_ylabel("max over D2 grid of std(effort), effort units")
    ax.set_title("(d) max policy noise over the grid, entering the Phase-A eligibility test")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    style(ax)
    fig.tight_layout()
    save(fig, figdir, "fig5_eta2_calls")
    # Fig 6: sampling
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axs[0]
    for q, xc in ((TQ, 0), (a.q_other, 1)):
        v = Vf[Vf.q == q]
        jit = (np.arange(len(v)) - len(v) / 2) * 0.012
        ax.scatter(xc + jit, v["peak_share"], s=20, color=C_BLUE, alpha=0.8, zorder=3)
        pe_ = float(v["expected"].iloc[0])
        sd_ = np.sqrt(pe_ * (1 - pe_) / float(v["n_rows"].iloc[0]))
        ax.fill_between([xc - 0.3, xc + 0.3], pe_ - 2 * sd_, pe_ + 2 * sd_, color=C_BAND, alpha=0.8, lw=0, zorder=1)
        ax.plot([xc - 0.3, xc + 0.3], [pe_, pe_], color=C_INK2, lw=1.0, zorder=2)
        if q == TQ:
            vt = v[v.seed == TS]
            ax.scatter(xc + jit[list(v.seed).index(TS)], vt["peak_share"], s=70, color=C_ORANGE, zorder=5, edgecolor="white", linewidth=0.8, label=f"seed {TS}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"q={TQ}\n(4/40 bins)", f"q={a.q_other}\n(4/44 bins)"])
    ax.set_ylabel("peak-set share of the cumulative Phase-A visits")
    ax.set_title("(a) share of visits in bins meeting (-20, 20), u=1600\n(band: +-2 binomial SD around the design value)")
    ax.legend(frameon=False)
    style(ax)
    ax = axs[1]
    for s in seeds:
        v = VIS[(VIS.q == TQ) & (VIS.seed == s)]
        if s == TS:
            continue
        ax.plot(v["u"], v["z_binomial"], color=C_GREY, lw=0.6, alpha=0.5)
    v = VIS[(VIS.q == TQ) & (VIS.seed == TS)]
    ax.plot(v["u"], v["z_binomial"], color=C_ORANGE, lw=1.8, marker="o", ms=3.5, label=f"seed {TS}")
    ax.axhline(0, color=C_INK2, lw=0.8)
    ax.axhspan(-2, 2, color=C_BAND, alpha=0.4, lw=0)
    ax.set_xlabel("update (verifier call)")
    ax.set_ylabel("binomial z-score of the cumulative peak share")
    ax.set_title("(b) cumulative peak-set share over Phase A (grey: other q=50)")
    ax.legend(frameon=False)
    style(ax)
    ax = axs[2]
    ax.axis("off")
    txt = ("D1 clamp counts, Phase A (updates 1..1600), summed\n(raw Beta draws below 1e-6 or above 1-1e-6)\n\n"
           f"learner stage-2 rows   : {int(ct['d1_L_s2_n_sum']):,}\n"
           f"  lo / hi (all)        : {int(ct['d1_L_s2_lo_sum'])} / {int(ct['d1_L_s2_hi_sum'])}\n"
           f"  inside |d|<2q lo/hi  : {int(ct['d1_L_s2_in_lo_sum'])} / {int(ct['d1_L_s2_in_hi_sum'])}\n"
           f"  outside lo/hi        : {int(ct['d1_L_s2_out_lo_sum'])} / {int(ct['d1_L_s2_out_hi_sum'])}\n"
           f"opponent stage-2 lo/hi : {int(ct['d1_O_s2_lo_sum'])} / {int(ct['d1_O_s2_hi_sum'])}\n"
           f"rows with alpha<1 / beta<1 : {int(ct['d1_pol_n_alpha_lt1_sum'])} / {int(ct['d1_pol_n_beta_lt1_sum'])}\n"
           f"min alpha / min beta   : {float(ct['d1_pol_alpha_min_overall']):.3f} / {float(ct['d1_pol_beta_min_overall']):.1f}\n\n"
           f"19 other q=50 seeds: lo+hi total = {int(co['d1_L_s2_lo_sum'].sum() + co['d1_L_s2_hi_sum'].sum() + co['d1_O_s2_lo_sum'].sum() + co['d1_O_s2_hi_sum'].sum())};\n"
           f"all 40 runs: every lo/hi count summed = {int(N.values['clamp.all40_total_lo_hi'])}")
    ax.text(0.0, 1.0, txt, va="top", ha="left", family="monospace", fontsize=8.5, color=C_INK)
    ax.set_title("(c) D1 clamp counts, seed " + str(TS) + " (table)")
    fig.tight_layout()
    save(fig, figdir, "fig6_sampling")
    # Fig 7: decomposition
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    A50s = A50.sort_values("rl_gap").reset_index(drop=True)
    xx = np.arange(len(A50s))
    ax.bar(xx, A50s["smooth_gap"], color=C_BLUE, width=0.7, label="smoothing-predicted gap  e*(0) - e_pred(0)")
    ax.bar(xx, A50s["remainder_gap"], bottom=A50s["smooth_gap"], color=C_GREY, width=0.7, label="remainder  e_pred(0) - e_hat(0)")
    ax.set_xticks(xx)
    ax.set_xticklabels([str(s) for s in A50s["seed"]], rotation=90, fontsize=7)
    for i, s in enumerate(A50s["seed"]):
        if s == TS:
            ax.get_xticklabels()[i].set_color(C_ORANGE)
            ax.get_xticklabels()[i].set_fontweight("bold")
            ax.annotate(f"seed {TS}", (i, A50s["rl_gap"].iloc[i]), textcoords="offset points", xytext=(-4, 4), ha="right", color=C_ORANGE, fontsize=8.5)
    if floor is not None:
        ax.axhline(float(np.median(floor)), color=C_INK, lw=1.2, ls="--", label=f"supervised-fit floor (median of 5 inits) = {np.median(floor):.4f}, on the axis")
    ax.set_ylabel("d = 0 peak gap  e*(0) - e_hat(0)  (effort units)")
    ax.set_title(f"(a) three-way decomposition of the d=0 gap, q={TQ}, 20 runs sorted by RL gap")
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    ax = axs[1]
    for q, col, mk in ((a.q_other, C_AQUA, "^"), (TQ, C_BLUE, "o")):
        A_ = ALL[ALL.q == q]
        A_ = A_[~((A_.q == TQ) & (A_.seed == TS))]
        ax.scatter(A_["smooth_gap"], A_["rl_gap"], s=22, color=col, marker=mk, alpha=0.85, label=f"q={q} runs", zorder=3)
    ax.scatter([At["smooth_gap"]], [At["rl_gap"]], s=80, color=C_ORANGE, edgecolor="white", linewidth=0.8, zorder=5, label=f"q={TQ} seed {TS}")
    lim = float(max(ALL["rl_gap"].max(), ALL["smooth_gap"].max())) * 1.05
    ax.plot([0, lim], [0, lim], color=C_INK2, lw=0.9, ls="--")
    ax.text(lim * 0.31, lim * 0.36, "RL gap = smoothing gap", rotation=45, fontsize=7.5, color=C_INK2)
    ax.set_xlim(0, lim * 0.45)
    ax.set_ylim(0, lim)
    ax.set_xlabel("smoothing-predicted gap  e*(0) - e_pred(0)")
    ax.set_ylabel("RL gap  e*(0) - e_hat(0)")
    ax.set_title("(b) RL gap against the smoothing-predicted gap, all runs")
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    save(fig, figdir, "fig7_decomposition")
    # Fig 8: per-seed late level against trend, all 20 q=50 seeds
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axs[0]
    for _, x in ps50.iterrows():
        isT = int(x["seed"]) == TS
        ax.scatter(x["dec_minus_pre"], x["late_mean"], s=70 if isT else 22, color=C_ORANGE if isT else C_BLUE, edgecolor="white" if isT else "none", zorder=4 if isT else 3)
    ax.set_xlabel("mean(u 1225..1600) - mean(u 900..1200) of e_hat(0) relative error")
    ax.set_ylabel("mean of e_hat(0) relative error, u 900..1600")
    ax.axvline(0, color=C_INK2, lw=0.7)
    ax.set_title(f"(a) late level against LR-decay response, q={TQ} seeds (orange: {TS})")
    style(ax)
    ax = axs[1]
    for _, x in ps50.iterrows():
        isT = int(x["seed"]) == TS
        v = x["first_u_ge_m010"]
        v = 1650 if v is None or (isinstance(v, float) and np.isnan(v)) else v
        ax.scatter(v, x["late_mean"], s=70 if isT else 22, color=C_ORANGE if isT else C_BLUE, edgecolor="white" if isT else "none", zorder=4 if isT else 3)
    ax.set_xlabel("first export with e_hat(0) relative error >= -0.10")
    ax.set_ylabel("mean of e_hat(0) relative error, u 900..1600")
    ax.set_title("(b) late level against speed of the ascent")
    style(ax)
    fig.tight_layout()
    save(fig, figdir, "fig8_per_seed_level_trend")

    N.put("meta.elapsed_sec", time.time() - t_start, "wall time of this tool")
    N.dump(out / "numbers.json")
    (out / "report_draft.md").write_text(build_report(N, a, root, ana, out, figdir, verify_fn is not None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
