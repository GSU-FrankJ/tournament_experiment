"""Shared helpers of the MS-R4 analysis tools (``r4_e1_analysis.py``, ``r4_e2_analysis.py``, ``r4_preamble.py``).

  * the pre-registered numeric parameters (``reports/ms/r4/prereg_criteria.json``): :func:`load_criteria`;
  * statistics: percentile bootstrap of a mean and of the ratio-of-means reduction with a fresh generator per call,
    the exact Clopper-Pearson interval, the trapezoidal area;
  * the per-run table: :func:`per_run_row` / :func:`build_per_run` (``r1_analysis.extract_ms_run`` + the MS-R2 / MS-R3
    column functions + the MS-R4 columns: the runner's own gate verdicts, the metric values of ``gates.json``, the cost per
    phase and category, the bounds of the mean map).

Everything here is evaluation / reporting; the closed-form columns are the ones the earlier analysis tools already carry.
The blind recomputation tools (``r4_blind_e1.py``, ``r4_blind_e2.py``) do NOT import this module.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import r1_analysis as R1  # noqa: E402
import r2_analysis as R2  # noqa: E402
import r3_analysis as R3  # noqa: E402

GATE_NAMES = ("G-A", "G-F", "G-N", "G-S")
TIE_BOUND_TOL = 1e-3          # effort units: the tie mean within this of a bound of the mean map is "at the bound"
R4_COLS = (["experiment", "peak_abs", "peak_signed", "gate_G_A", "gate_G_F", "gate_G_N", "gate_G_S", "gates_all_pass",
            "nonzero_exit", "eta_final", "eta_dev", "gmax_final", "gmax_dev", "s1_abs", "rmse_pos", "tail_mean",
            "tie_at_lower_bound", "tie_at_upper_bound", "tie_dist_lower", "tie_dist_upper", "wall_total_sec",
            "wall_stage2_sec", "wall_stage1_sec", "cpu_stage2_sec", "stage2_train_rollout_sec", "stage2_train_update_sec",
            "stage2_dev_verifier_sec", "stage1_train_rollout_sec", "stage1_train_update_sec", "stage1_dev_verifier_sec",
            "dev_verifier_share_stage2", "n_updates_stage2", "actor_variant_cfg", "focus_scope", "focus_map"])
STR_R4 = {"experiment", "actor_variant_cfg", "focus_scope", "focus_map"}


# ---------------------------------------------------------------------------------------------- criteria file
def load_criteria(path: Path) -> Tuple[Dict[str, Any], str]:
    """The pre-registered parameters of D6 and the SHA-256 of the file."""
    raw = Path(path).read_bytes()
    return json.loads(raw.decode()), hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------------------------- statistics
def boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """``rng.integers(0, n, size=(n_boot, n))`` of a FRESH ``numpy.random.default_rng(seed)`` (the convention of D6)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def boot_mean_ci(x: Sequence[float], n_boot: int, seed: int, pct: Tuple[float, float] = (2.5, 97.5)) -> Tuple[float, float]:
    """Percentile interval of the mean of ``x`` over ``n_boot`` resamples of its entries (fresh generator per call)."""
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    m = a[boot_indices(a.size, n_boot, seed)].mean(axis=1)
    return float(np.percentile(m, pct[0])), float(np.percentile(m, pct[1]))


def boot_reduction(a_x: Sequence[float], a_ref: Sequence[float], n_boot: int, seed: int,
                   pcts: Sequence[Tuple[float, float]] = ((2.5, 97.5),)) -> Dict[str, Any]:
    """``rho = 1 - mean(A_x) / mean(A_ref)`` and its paired percentile bootstrap intervals (the seeds are resampled
    jointly for the two arms and the ratio of the means is recomputed on every resample).

    Args:
        a_x: Per-seed areas of the arm.
        a_ref: Per-seed areas of the reference arm, in the same seed order.
        n_boot: Number of resamples.
        seed: Seed of the fresh generator.
        pcts: Percentile pairs.

    Returns:
        ``{"rho": float, "intervals": [(lo, hi), ...]}``.
    """
    x, r = np.asarray(a_x, dtype=float), np.asarray(a_ref, dtype=float)
    if x.shape != r.shape or x.size == 0:
        return {"rho": float("nan"), "intervals": [(float("nan"), float("nan")) for _ in pcts]}
    idx = boot_indices(x.size, n_boot, seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_b = 1.0 - x[idx].mean(axis=1) / r[idx].mean(axis=1)
    return {"rho": float(1.0 - x.mean() / r.mean()),
            "intervals": [(float(np.percentile(rho_b, lo)), float(np.percentile(rho_b, hi))) for lo, hi in pcts]}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided ``1 - alpha`` interval of a binomial proportion ``k / n``."""
    from scipy.stats import beta
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def trapezoid(x: Sequence[float], y: Sequence[float]) -> float:
    """Trapezoidal area of ``y`` over ``x`` (NaN if any value is not finite)."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xa.size < 2 or not (np.isfinite(xa).all() and np.isfinite(ya).all()):
        return float("nan")
    return float(np.sum((xa[1:] - xa[:-1]) * (ya[1:] + ya[:-1]) * 0.5))


def paired_stats(a: Sequence[float], b: Sequence[float], n_boot: int, seed: int) -> Dict[str, Any]:
    """Mean paired difference ``a - b`` with its 95 % percentile interval, over the pairs where both are finite."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = (x - y)[ok]
    lo, hi = boot_mean_ci(d, n_boot, seed)
    return {"n_pairs": int(d.size), "mean_a": float(x[ok].mean()) if d.size else float("nan"),
            "mean_b": float(y[ok].mean()) if d.size else float("nan"),
            "mean_diff": float(d.mean()) if d.size else float("nan"),
            "median_diff": float(np.median(d)) if d.size else float("nan"),
            "ci_lo": lo, "ci_hi": hi, "n_neg": int((d < 0).sum()), "n_pos": int((d > 0).sum())}


# ---------------------------------------------------------------------------------------------- per-run table
def _g(d: Mapping[str, Any], *keys: str) -> Any:
    return R1._get(d, *keys)


def _gate_pass(gates: Mapping[str, Any], name: str) -> Any:
    v = _g(gates, name, "pass")
    return bool(v) if v is not None else float("nan")


def cost_columns(d: Path) -> Dict[str, Any]:
    """Wall time per phase and category from ``ms_run_summary.json`` / ``status.json`` (NaN where absent)."""
    nan = float("nan")
    out = {k: nan for k in ("wall_total_sec", "wall_stage2_sec", "wall_stage1_sec", "cpu_stage2_sec",
                            "stage2_train_rollout_sec", "stage2_train_update_sec", "stage2_dev_verifier_sec",
                            "stage1_train_rollout_sec", "stage1_train_update_sec", "stage1_dev_verifier_sec",
                            "dev_verifier_share_stage2")}
    an: List[str] = []
    summ = R1._jload(d / "ms_run_summary.json", an, False)
    st = R1._jload(d / "status.json", an, False)
    out["wall_total_sec"] = R1._f(st.get("total_wall_sec"))
    pt = summ.get("phase_timing") or {}
    for t in (2, 1):
        ph = pt.get(f"stage{t}") or {}
        out[f"wall_stage{t}_sec"] = R1._f(ph.get("wall_sec"))
        cat = ph.get("by_category_sec") or {}
        for c in ("train_rollout", "train_update", "dev_verifier"):
            out[f"stage{t}_{c}_sec"] = R1._f(cat.get(c))
        if t == 2:
            out["cpu_stage2_sec"] = R1._f(ph.get("process_cpu_sec"))
    w2, v2 = out["wall_stage2_sec"], out["stage2_dev_verifier_sec"]
    out["dev_verifier_share_stage2"] = v2 / w2 if np.isfinite(w2) and w2 > 0 and np.isfinite(v2) else nan
    return out


def per_run_row(d: Path, arm: str, q: int, seed: int, th: R1.Thresholds, proto: Mapping[str, Any],
                experiment: str = "") -> Dict[str, Any]:
    """One per-run row: the ``r1_analysis`` / ``r2_analysis`` / ``r3_analysis`` columns plus :data:`R4_COLS`.

    The gate verdicts are the runner's own (``gates.json``); a run whose directory is missing or whose status is not
    ``done`` has ``gates_all_pass`` False and NaN metrics, and counts as a failed run (``nonzero_exit`` True when the status is
    not ``done`` with exit code 0).
    """
    nan = float("nan")
    r, _, _ = R1.extract_ms_run(d, arm, q, seed, th, "pilot")
    R2.add_r2_columns(r, True)
    R3.add_r3_columns(r, proto)
    an: List[str] = []
    gates = R1._jload(d / "gates.json", an, False)
    cfg = R1._jload(d / "run_config.json", an, False)
    mv = gates.get("metric_values") or {}
    r["experiment"] = experiment
    r["peak_abs"] = R1._f(r.get("stage2_peak_rel_err_abs"))
    r["peak_signed"] = R1._f(r.get("stage2_peak_rel_err_signed"))
    for g in GATE_NAMES:
        r["gate_" + g.replace("-", "_")] = _gate_pass(gates, g)
    done = r.get("status") == "done"
    gp = [r["gate_" + g.replace("-", "_")] for g in GATE_NAMES]
    r["gates_all_pass"] = bool(done and all(isinstance(x, bool) and x for x in gp)
                               and (gates.get("global_rng") or {}).get("status") == "ok")
    r["nonzero_exit"] = bool(not (done and R1._f(r.get("exit_code")) == 0.0))
    for k_out, k_in in (("eta_final", "eta_final"), ("eta_dev", "eta_dev"), ("gmax_final", "gmax_final"),
                        ("gmax_dev", "gmax_dev"), ("s1_abs", "s1"), ("rmse_pos", "rmse"), ("tail_mean", "tail")):
        r[k_out] = R1._f(mv.get(k_in))
    e0 = R1._f(r.get("e2_at_0"))
    m = float((cfg.get("record") or {}).get("ppo", {}).get("mu_clamp", 1e-6)) if cfg else 1e-6
    emax = float((cfg.get("record") or {}).get("game", {}).get("e_max", 100.0)) if cfg else 100.0
    r["tie_dist_lower"] = e0 - emax * m if np.isfinite(e0) else nan
    r["tie_dist_upper"] = emax * (1.0 - m) - e0 if np.isfinite(e0) else nan
    r["tie_at_lower_bound"] = bool(np.isfinite(e0) and r["tie_dist_lower"] <= TIE_BOUND_TOL)
    r["tie_at_upper_bound"] = bool(np.isfinite(e0) and r["tie_dist_upper"] <= TIE_BOUND_TOL)
    r.update(cost_columns(d))
    r["n_updates_stage2"] = R1._f(((gates.get("rule") or {}).get("2") or {}).get("total_updates"))   # updates, not checks (t2_n_checks)
    r["actor_variant_cfg"] = str(cfg.get("actor_variant", "t1")) if cfg else ""
    r["focus_scope"] = str(cfg.get("focus_scope", "")) if cfg else ""
    r["focus_map"] = str(cfg.get("focus_map", "")) if cfg else ""
    return r


def per_run_columns() -> List[str]:
    """Column order of the per-run table: ``r3_analysis.COLUMNS`` then the MS-R4 columns."""
    return list(R3.COLUMNS) + [c for c in R4_COLS if c not in R3.COLUMNS]


def build_per_run(pilot_root: Path, arms: Sequence[str], qs: Sequence[int], seeds: Sequence[int], th: R1.Thresholds,
                  proto: Mapping[str, Any], experiment_of: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    """The per-run table of the given arms (one row per planned run, in arm / q / seed order; a missing run is a row with
    status ``missing``)."""
    cols = per_run_columns()
    rows = []
    for arm in arms:
        for q in qs:
            for s in seeds:
                d = Path(pilot_root) / f"q{q}" / f"seed{s}" / arm
                rows.append(per_run_row(d, arm, q, s, th, proto, (experiment_of or {}).get(arm, "")))
    df = pd.DataFrame(rows, columns=cols)
    string_cols = set(R3.STR_COLS) | STR_R4
    bool_cols = set(R3.BOOL_COLS) | {"gate_G_A", "gate_G_F", "gate_G_N", "gate_G_S", "gates_all_pass", "nonzero_exit",
                                      "tie_at_lower_bound", "tie_at_upper_bound"}
    for c in cols:
        if c not in string_cols and c not in bool_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def truthy(v: Any) -> bool:
    """Bool of a value read back from a CSV (NaN, empty and 'False' are False)."""
    return R1._truthy(v)


def sel(df: pd.DataFrame, arm: str, q: int) -> pd.DataFrame:
    """Rows of one (arm, q), sorted by seed."""
    return df[(df["arm"] == arm) & (df["q"] == q)].sort_values("seed")


__all__ = ["GATE_NAMES", "R4_COLS", "load_criteria", "boot_indices", "boot_mean_ci", "boot_reduction", "clopper_pearson",
           "trapezoid", "paired_stats", "cost_columns", "per_run_row", "per_run_columns", "build_per_run", "truthy", "sel"]
