"""Metrics of round T3-R1 (D6): the in-band / out-of-band split of the deviation gains, the terminal-stage
recovery against the closed-form tent for any horizon, the gates, and the per-tier metric dictionary of one
candidate.

Evaluation and reporting only. The closed-form tent ``e_T*(d)`` (``utils.theory_multistage.g2_two_stage``) is read
here and by nothing that trains. The verifier arrays come from ``utils.v2_metrics.evaluate`` (pure).

Column convention of :func:`tier_metrics`: the names below, per tier; the per-run table carries the final-tier
columns unprefixed and the development-tier columns with the prefix ``dev_``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.curriculum_env import GameSpec  # noqa: E402
from utils.theory_multistage import g1_two_stage, g2_two_stage  # noqa: E402
from utils.v2_metrics import V2Eval, symmetric_grid, zero_index  # noqa: E402

CRITERIA_PATH = ROOT / "reports" / "t3" / "r1" / "prereg_criteria.json"
BAND_TOL = 1e-9
NAN = float("nan")


def load_criteria(path: Path = CRITERIA_PATH) -> Tuple[Dict[str, Any], str]:
    """The pre-registered parameters and the SHA-256 of the file.

    Args:
        path: ``reports/t3/r1/prereg_criteria.json``.

    Returns:
        ``(criteria, sha256)``.
    """
    raw = Path(path).read_bytes()
    return json.loads(raw.decode()), hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------------- band split
def in_band_mask(d_grid: np.ndarray, q: float, t: int, tol: float = BAND_TOL) -> np.ndarray:
    """Nodes of the stage-``t`` grid with ``|d| < 2 q (t - 1)``; a node at exactly the edge is out of band.

    ``tol`` keeps a node that sits on the edge up to float error out of band.
    """
    return np.abs(np.asarray(d_grid, dtype=float)) < 2.0 * float(q) * (int(t) - 1) - tol


def masked_max(values: np.ndarray, mask: np.ndarray, d_grid: np.ndarray, dw: float) -> Tuple[float, float, int]:
    """``(max / dw, argmax d, n nodes)`` of ``values`` over the masked nodes (NaN, NaN, 0 if the mask is empty)."""
    v = np.asarray(values, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return NAN, NAN, 0
    j = int(np.argmax(np.where(m, v, -np.inf)))
    return float(v[j] / dw), float(np.asarray(d_grid, dtype=float)[j]), int(m.sum())


def band_columns(d_grid: np.ndarray, delta: np.ndarray, gain: np.ndarray, q: float, t: int,
                 dw: float) -> Dict[str, float]:
    """In-band and out-of-band maxima of the one-step gain ``Delta_t`` and the dynamic gain ``G_t`` at stage ``t >= 2``.

    Returns:
        ``Delta_in_t{t}``, ``Delta_out_t{t}``, ``G_in_t{t}``, ``G_out_t{t}`` (divided by ``dw``), their argmax
        gaps (``*_argmax_d_t{t}``) and the node counts ``n_in_t{t}``, ``n_out_t{t}``.
    """
    inb = in_band_mask(d_grid, q, t)
    out: Dict[str, float] = {}
    for name, arr in (("Delta", delta), ("G", gain)):
        for side, m in (("in", inb), ("out", ~inb)):
            val, arg, n = masked_max(arr, m, d_grid, dw)
            out[f"{name}_{side}_t{t}"] = val
            out[f"{name}_{side}_argmax_d_t{t}"] = arg
            out[f"n_{side}_t{t}"] = n
    return out


# ---------------------------------------------------------------------------------- terminal-stage recovery
def terminal_recovery(policy: Callable[[int, np.ndarray], np.ndarray], spec: GameSpec, step: float
                      ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Recovery of the last stage against the closed-form tent, for any horizon.

    The grid is the symmetric grid of ``D_T`` with spacing ``step`` (0.5); ``pos`` is ``|d| < 2q``, ``tail`` the
    rest. At T = 2 every value equals ``utils.v2_metrics.recovery_metrics`` bit for bit (same expressions).

    Returns:
        ``(scalars, arrays)`` with ``rmse_pos_T_over_eT0``, ``tail_mean_T_over_eT0``, ``tail_max_T_over_eT0``,
        ``peak_T_signed``, ``peak_T_abs``, ``eT_at_0``, ``eT_star_at_0`` and the grid arrays.
    """
    T, q = int(spec.T), float(spec.q)
    D = symmetric_grid(spec.domain_half(T), step)
    z = zero_index(D)
    eT = np.asarray(policy(T, D), dtype=float)
    g = g2_two_stage(D, q, spec.w_h, spec.w_l, spec.k, spec.e_max)
    g0 = float(g[z])
    pos = np.abs(D) < 2.0 * q
    tail = ~pos
    err_pos = eT[pos] - g[pos]
    rmse = float(np.sqrt(np.mean(err_pos ** 2)))
    jt = int(np.argmax(np.where(tail, eT, -np.inf)))
    sc = {
        "eT_at_0": float(eT[z]), "eT_star_at_0": g0,
        "peak_T_signed": (float(eT[z]) - g0) / g0, "peak_T_abs": abs(float(eT[z]) - g0) / g0,
        "rmse_pos_T": rmse, "rmse_pos_T_over_eT0": rmse / g0,
        "tail_mean_T": float(eT[tail].mean()), "tail_mean_T_over_eT0": float(eT[tail].mean()) / g0,
        "tail_max_T": float(eT[jt]), "tail_max_T_over_eT0": float(eT[jt]) / g0,
        "tail_argmax_d_T": float(D[jt]),
        "recovery_grid_step": float(step), "recovery_n_pos": int(pos.sum()), "recovery_n_tail": int(tail.sum())}
    return sc, {"recovery_d_grid": D, "recovery_eT": eT, "recovery_gT": g}


# ---------------------------------------------------------------------------------- gates
def gate_columns(fin: Mapping[str, Any], dev: Mapping[str, Any], T: int, criteria: Mapping[str, Any]
                 ) -> Dict[str, Any]:
    """The gate values and verdicts of one run (v2.0's thresholds; every ``<=`` inclusive in float64).

    G-A (final tier): ``eta_T_over_dw``, ``rmse_pos_T_over_eT0``, ``tail_mean_T_over_eT0``. G-F (final):
    ``Gmax_full_over_dw``. G-N: development minus final of ``eta_T_over_dw`` and of ``Gmax_full_over_dw``.
    G-S (T = 2 only): ``stage1_rel_err_abs``. Run pass: G-A and G-F and G-N (and G-S at T = 2).

    Args:
        fin: Final-tier metric dictionary of :func:`tier_metrics`.
        dev: Development-tier metric dictionary.
        T: Horizon.
        criteria: ``prereg_criteria.json``.

    Returns:
        ``gate_G_A, gate_G_F, gate_G_N, gate_G_S`` (None at T = 3), ``gates_pass`` and the sub-verdicts.
    """
    g = criteria["gates"]
    ga, gf, gn = g["G-A"], g["G-F"], g["G-N"]
    ga_eta = bool(float(fin["eta_T_over_dw"]) <= ga["eta_T_over_dw"])
    ga_rmse = bool(float(fin["rmse_pos_T_over_eT0"]) <= ga["rmse_pos_T_over_eT0"])
    ga_tail = bool(float(fin["tail_mean_T_over_eT0"]) <= ga["tail_mean_T_over_eT0"])
    g_f = bool(float(fin["Gmax_full_over_dw"]) <= gf["Gmax_full_over_dw"])
    gn_eta = bool(abs(float(dev["eta_T_over_dw"]) - float(fin["eta_T_over_dw"]))
                  <= gn["eta_T_over_dw_dev_minus_final_abs"])
    gn_gmax = bool(abs(float(dev["Gmax_full_over_dw"]) - float(fin["Gmax_full_over_dw"]))
                   <= gn["Gmax_full_over_dw_dev_minus_final_abs"])
    out: Dict[str, Any] = {
        "ga_eta": ga_eta, "ga_rmse": ga_rmse, "ga_tail": ga_tail, "gn_eta": gn_eta, "gn_gmax": gn_gmax,
        "gate_G_A": ga_eta and ga_rmse and ga_tail, "gate_G_F": g_f, "gate_G_N": gn_eta and gn_gmax,
        "gate_G_S": None, "gs_s1": None}
    ok = out["gate_G_A"] and out["gate_G_F"] and out["gate_G_N"]
    if int(T) == g["G-S"]["applies_at_horizon"]:
        gs = bool(float(fin["stage1_rel_err_abs"]) <= g["G-S"]["stage1_rel_err_abs"])
        out["gate_G_S"] = gs
        out["gs_s1"] = gs
        ok = ok and gs
    out["gates_pass"] = bool(ok)
    return out


# ---------------------------------------------------------------------------------- one tier
def tier_metrics(ev: V2Eval, mean_fn: Callable[[int, np.ndarray], np.ndarray], spec: GameSpec,
                 recovery_step: float = 0.5) -> Dict[str, Any]:
    """Every D6 quantity of one candidate on one verifier tier (final or development).

    Args:
        ev: ``utils.v2_metrics.evaluate`` of the candidate on the tier.
        mean_fn: The candidate's mean-effort function ``(t, d) -> effort`` (the one that was evaluated).
        spec: Game parameters (T = 2 or 3).
        recovery_step: Spacing of the terminal-stage recovery grid.

    Returns:
        Flat dict: validity, ``Gmax_full_over_dw`` (+ argmax stage and gap), ``EXP_root_over_dw``,
        ``dReach_over_dw``, ``Deltamax_all_over_dw``, ``dFull_over_dw``, ``eta_T_over_dw``, per stage the
        ``G_max_t{t}_over_dw`` / ``Delta_max_t{t}_over_dw`` with argmax gaps, the band split at t >= 2, the
        terminal-stage recovery, ``e1_at_0`` (and at T = 2 the stage-1 error), and the maximum normalised
        Beta standard deviation per stage on the tier grid.
    """
    sc = ev.scalars
    T, dw = int(spec.T), float(spec.dw)
    out: Dict[str, Any] = {"valid": bool(sc["valid"]), "invalid_reasons": str(sc["invalid_reasons"])}
    for k in ("Gmax_full_over_dw", "Gmax_full_t", "Gmax_full_d", "EXP_root_over_dw", "dReach_over_dw",
              "Deltamax_all_over_dw", "dFull_over_dw", "eta_T_over_dw"):
        out[k] = sc[k]
    for t in range(1, T + 1):
        for k in (f"G_max_t{t}_over_dw", f"G_argmax_d_t{t}", f"Delta_max_t{t}_over_dw", f"Delta_argmax_d_t{t}"):
            out[k] = sc[k]
        arr = ev.arrays
        if f"v_t{t}_std_norm" in arr:
            out[f"conc_max_std_norm_t{t}"] = float(np.nanmax(arr[f"v_t{t}_std_norm"]))
        if t >= 2:
            out.update(band_columns(arr[f"v_t{t}_d_grid"], arr[f"v_t{t}_delta"], arr[f"v_t{t}_G"], spec.q, t, dw))
    rsc, _ = terminal_recovery(mean_fn, spec, recovery_step)
    out.update(rsc)
    out["e1_at_0"] = float(np.asarray(mean_fn(1, np.zeros(1)), dtype=float)[0])
    if T == 2:
        g1 = g1_two_stage(spec.q, spec.w_h, spec.w_l, spec.k)
        out["g1"] = g1
        out["stage1_rel_err_signed"] = (out["e1_at_0"] - g1) / g1
        out["stage1_rel_err_abs"] = abs(out["stage1_rel_err_signed"])
    return out


def run_columns(fin: Mapping[str, Any], dev: Mapping[str, Any], T: int, criteria: Mapping[str, Any]
                ) -> Dict[str, Any]:
    """The per-run row body: final-tier columns unprefixed, development-tier columns prefixed ``dev_``, and the
    gate columns."""
    row: Dict[str, Any] = dict(fin)
    row.update({f"dev_{k}": v for k, v in dev.items()})
    row.update(gate_columns(fin, dev, T, criteria))
    return row


def terminal_recovery_matches_evaluate(ev: V2Eval, rec: Mapping[str, float]) -> bool:
    """T = 2 check used by the tests: the recovery fields equal ``evaluate``'s stage-2 fields bit for bit."""
    pairs = (("rmse_pos_T_over_eT0", "stage2_rmse_pos_over_g2_0"), ("tail_mean_T_over_eT0", "stage2_tail_mean_over_g2_0"),
             ("tail_max_T_over_eT0", "stage2_tail_max_over_g2_0"), ("peak_T_signed", "stage2_peak_rel_err_signed"),
             ("peak_T_abs", "stage2_peak_rel_err_abs"), ("eT_at_0", "e2_at_0"), ("eT_star_at_0", "g2_at_0"))
    return all(float(rec[a]) == float(ev.scalars[b]) for a, b in pairs)
