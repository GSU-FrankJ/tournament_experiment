"""Pilot 4 shared helpers: actor loading, tail-averaged candidates, stage-2 / stage-1 metric rows.

candidate_K is the pointwise average of the K deterministic (Beta-mean) mappings
mu_k(t, d) = e_min + range * alpha_k / (alpha_k + beta_k). Every network is evaluated at the exact
query point d, so the average is exact at every d the verifier (or the recovery grid) asks for; no
interpolation is involved, the same as for any single-network policy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from pilot2_analysis import _actor_fns  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG  # noqa: E402
from utils.v2_metrics import evaluate  # noqa: E402

Policy = Callable[[int, np.ndarray], np.ndarray]
N_BOOT, BOOT_SEED = 10000, 20261001
STAGE2_KEYS = ("stage2_peak_rel_err_signed", "stage2_peak_rel_err_abs", "stage2_rmse_pos_over_g2_0",
               "stage2_tail_mean", "stage2_tail_max", "stage2_tail_mean_over_g2_0", "stage2_tail_max_over_g2_0",
               "stage2_sym_err_max", "eta_T_over_dw", "DeltaT_over_dw_on_max", "DeltaT_over_dw_on_mean_cellmass_weighted",
               "DeltaT_over_dw_off_max", "DeltaT_over_dw_off_mean_unweighted", "e2_at_0", "g2_at_0")
FULL_KEYS = ("e1_at_0", "g1", "stage1_rel_err_signed", "stage1_rel_err_abs", "Gmax_full_over_dw", "Gmax_full_t",
             "Gmax_full_d", "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw")


def mean_fn(path: str, spec) -> Policy:
    """Beta-mean policy of the actor stored in a weight-export NPZ or a v2 full-state checkpoint."""
    return _actor_fns(path, spec)[0]


def average(policies: Sequence[Policy]) -> Policy:
    """Pointwise average of deterministic mappings (exact at every query point)."""
    pols = list(policies)

    def pol(t: int, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        return np.mean([np.asarray(p(t, d), dtype=float) for p in pols], axis=0)
    return pol


def compose(stage1: Policy, stage2: Policy) -> Policy:
    """Candidate with ``stage1`` at t = 1 and ``stage2`` at t = 2."""
    return lambda t, d: stage2(t, d) if t == 2 else stage1(t, d)


def location_free(arrays: Dict[str, np.ndarray], g20: float) -> Dict[str, float]:
    """(max_d e_hat_2(d) - e2*(0)) / e2*(0) on the recovery grid, with the argmax location."""
    D, e2 = arrays["recovery_d_grid"], arrays["recovery_e2"]
    j = int(np.argmax(e2))
    return {"stage2_peak_locfree_rel_err": (float(e2[j]) - g20) / g20, "stage2_peak_locfree_argmax_d": float(D[j]),
            "stage2_max_e2": float(e2[j])}


def eval_row(policy: Policy, spec, beta_fn=None, cfg=DEV_CONFIG, full: bool = False) -> Dict[str, float]:
    """evaluate() on the dev tier; stage-2 metrics (+ full-policy metrics if ``full``)."""
    ev = evaluate(policy, spec, cfg, beta_fn=beta_fn)
    s = ev.scalars
    row = {k: s[k] for k in STAGE2_KEYS}
    row.update(location_free(ev.arrays, float(s["g2_at_0"])))
    if full:
        row.update({k: s[k] for k in FULL_KEYS})
    for k in ("sigma_effort_at_0_t1", "sigma_effort_at_0_t2"):
        if k in s:
            row[k] = s[k]
    return row


def paired_summary(diff: np.ndarray, lower_better: Optional[bool], rng: np.random.Generator) -> Dict[str, object]:
    """Median, sign count and 95% percentile bootstrap CI of the mean paired difference."""
    dv = np.asarray(diff, float)
    bm = dv[rng.integers(0, dv.size, size=(N_BOOT, dv.size))].mean(axis=1)
    out = {"n_pairs": int(dv.size), "mean": float(dv.mean()), "median": float(np.median(dv)),
           "min": float(dv.min()), "max": float(dv.max()), "n_neg": int((dv < 0).sum()), "n_pos": int((dv > 0).sum()),
           "n_zero": int((dv == 0).sum()), "boot_ci95_lo": float(np.percentile(bm, 2.5)),
           "boot_ci95_hi": float(np.percentile(bm, 97.5))}
    out["n_better"] = int((dv < 0).sum()) if lower_better else None
    out["better_if"] = "diff < 0" if lower_better else "no preferred direction"
    return out


def last_k(paths: List[str], k: int) -> List[str]:
    """The last ``k`` of a sorted list of export paths."""
    if len(paths) < k:
        raise ValueError(f"only {len(paths)} exports for K={k}")
    return paths[-k:]
