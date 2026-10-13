#!/usr/bin/env python3
"""MS-R4 pre-registered analysis of E1, the stage-2 tie bias (prompt D5, D6 and section 3.3).

Arms: ``t1_st_s16`` (control), ``relu_st_s16`` (control) and ``lrsm_st_s16`` (the candidate: LeakyReLU hidden units and a
smooth, unclamped mean map), each q in {50, 60} x the 20 development seeds 10511-10530. Runs live at
``<pilot-root>/q<q>/seed<seed>/<arm>/``; the offline learning curves at ``<curves-root>`` (``tools/ms/r4_curves.py``).

Primary criterion (D6; ``criterion_e1.csv``; all parts at BOTH q; descriptive in the sense that no gate and no selection
follows from it, the decision is the PI's):

  (E1-a) the mean |peak error| of ``lrsm_st_s16`` over its 20 runs <= 0.04;
  (E1-b) the 95 % interval of the mean paired difference of |peak error| (``lrsm_st_s16`` - ``t1_st_s16``) lies below 0;
  (E1-c) none of the 40 ``lrsm_st_s16`` runs fails G-A, G-F, G-N or G-S (the runner's own verdicts in ``gates.json``) and
         none exits nonzero.

A part that needs all 20 runs (or all 20 pairs) is reported ``False`` with ``incomplete`` in the status column when a planned
run is missing or failed. The thresholds, the bootstrap size and the seed come from ``reports/ms/r4/prereg_criteria.json``
(``--criteria``). Bootstrap: ``rng.integers(0, n, size=(10000, n))`` of a FRESH ``numpy.random.default_rng(20261013)`` per
(comparison, q, statistic); percentile intervals. The effect is the effect of the combined change; no part of it is attributed
to the LeakyReLU or to the mean map.

Descriptive tables: the same paired statistics for ``lrsm - relu`` and ``relu - t1``, the gate failures per arm and q with
exact Clopper-Pearson intervals, the paired changes of the decomposition and stage-1 metrics, the diagnostics of every failed
run (trajectories of e_hat_2(0), R0, RMSE_pos/e2*(0) and the symmetry error, whether the tie mean sits at a bound of the mean
map, the share of development nodes with a positive pre-activation per hidden unit at the freeze), the cost.

Outputs under ``--out`` (an existing non-empty directory is refused): ``per_run.csv``, ``criterion_e1.csv``, ``paired_e1.csv``,
``paired_seed_level_e1.csv``, ``failure_rates.csv``, ``failures.csv``, ``failure_trajectories.csv``, ``failure_units.csv``,
``cost_e1.csv``, ``strata.csv``, ``strata_summary.csv``, ``learning_curves_e1.csv``, ``tie_profile_e1.csv``, ``predictions_e1.csv``, ``analysis_info.json``,
``summary.txt``, ``figures/*.png``. Exit code 0 when every planned run is done, 3 otherwise, 2 on an argument error.

Usage:
    python tools/ms/r4_e1_analysis.py --pilot-root results/ms_r4/pilot --curves-root results/ms_r4/curves \
        --criteria reports/ms/r4/prereg_criteria.json --out results/ms_r4/analysis/e1
"""

from __future__ import annotations

import argparse
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
import r4_common as C  # noqa: E402
import r4_configs as rc  # noqa: E402

#: metrics of the paired descriptive tables: (column, lower is better / None)
PAIRED_METRICS: List[Tuple[str, Optional[bool]]] = [
    ("peak_abs", True), ("peak_signed", None), ("gap", True), ("smoothing", True), ("remainder", None),
    ("sigma_2_0", True), ("stage2_rmse_pos_over_g2_0", True), ("stage2_tail_mean_over_g2_0", True),
    ("eta_T_over_dw", True), ("t2_R0_final", True), ("w_eff", True), ("stage1_rel_err_abs", True),
    ("Gmax_full_over_dw", True)]
#: the E1-P1 reference: relu_st_s16 - t1_st_s16 on seeds 10501-10510 (results/ms_r3/analysis/criterion.csv, row relu_st_s16)
PRIOR_RELU_MINUS_T1 = {50: (-0.0104, -0.0162, -0.0047), 60: (-0.0206, -0.0309, -0.0068)}
TIE_HALF_WIDTH = 30.0
ARM_COLORS = {"t1_st_s16": "tab:blue", "relu_st_s16": "tab:red", "lrsm_st_s16": "tab:green"}


# ---------------------------------------------------------------------------------------------- criterion (pure)
def _complete(df: pd.DataFrame, arm: str, q: int) -> pd.DataFrame:
    s = C.sel(df, arm, q)
    return s[s["status"].astype(str).eq("done")]


def failed_run_mask(rows: pd.DataFrame) -> pd.Series:
    """Runs that exit nonzero (status not done with exit 0) or fail a gate or the process-global RNG assertion."""
    nz = rows["nonzero_exit"].map(C.truthy)
    ok = rows["gates_all_pass"].map(C.truthy)
    return nz | ~ok


def criterion_e1(df: pd.DataFrame, crit: Mapping[str, Any]) -> pd.DataFrame:
    """The E1 criterion table: one row per q and a final row ``q = both`` with the verdict.

    Args:
        df: Per-run table (columns ``arm, q, seed, status, peak_abs, nonzero_exit, gates_all_pass, gate_G_*``).
        crit: The parsed ``prereg_criteria.json``.

    Returns:
        Columns ``q, n_expected, n_runs_done, mean_abs_peak_lrsm, a_threshold, a_pass, n_pairs, mean_diff_vs_t1,
        ci_lo, ci_hi, b_pass, n_gate_fail, n_G_A_fail, n_G_F_fail, n_G_N_fail, n_G_S_fail, n_nonzero_exit, c_pass,
        met_q, status``; the ``both`` row holds ``met`` (all parts at both q).
    """
    e1 = crit["e1"]
    cand, ctrl = e1["arms"]["candidate"], e1["arms"]["control"]
    n_exp = int(crit["seeds"]["n_per_q"])
    n_boot, seed = int(crit["bootstrap"]["n_resamples"]), int(crit["bootstrap"]["seed_e1"])
    rows: List[Dict[str, Any]] = []
    for q in crit["qs"]:
        a = C.sel(df, cand, q)
        t = C.sel(df, ctrl, q)
        done_a = a[a["status"].astype(str).eq("done")]
        peak = done_a["peak_abs"].astype(float)
        both = done_a.merge(t[t["status"].astype(str).eq("done")][["seed", "peak_abs"]], on="seed", suffixes=("", "_ctrl"))
        diff = (both["peak_abs"] - both["peak_abs_ctrl"]).to_numpy(float)
        complete_a = len(done_a) == n_exp and bool(np.isfinite(peak).all())
        mean_a = float(peak.mean()) if len(peak) else float("nan")
        a_pass = bool(complete_a and mean_a <= float(e1["peak_abs_mean_max"]))
        lo, hi = C.boot_mean_ci(diff, n_boot, seed, tuple(crit["bootstrap"]["pct95"])) if len(diff) else (float("nan"),) * 2
        b_pass = bool(len(diff) == n_exp and np.isfinite(diff).all() and hi < float(e1["peak_diff_ci_upper_below"]))
        nz = int(a["nonzero_exit"].map(C.truthy).sum()) + max(0, n_exp - len(a))   # a missing run counts as a failed run
        gate_fail = {g: int(sum(1 for v in a["gate_" + g.replace("-", "_")] if not (isinstance(v, (bool, np.bool_)) and v)))
                     + max(0, n_exp - len(a)) for g in C.GATE_NAMES}
        any_fail = int(failed_run_mask(a).sum()) + max(0, n_exp - len(a))
        c_pass = bool(len(a) == n_exp and any_fail == 0 and nz == 0)
        status = "ok" if (complete_a and len(diff) == n_exp) else "incomplete"
        rows.append({"q": q, "n_expected": n_exp, "n_runs_done": len(done_a), "mean_abs_peak_lrsm": mean_a,
                     "a_threshold": float(e1["peak_abs_mean_max"]), "a_pass": a_pass, "n_pairs": len(diff),
                     "mean_diff_vs_t1": float(diff.mean()) if len(diff) else float("nan"), "ci_lo": lo, "ci_hi": hi,
                     "b_pass": b_pass, "n_gate_fail": any_fail, **{f"n_{g.replace('-', '_')}_fail": gate_fail[g] for g in C.GATE_NAMES},
                     "n_nonzero_exit": nz, "c_pass": c_pass, "met_q": bool(a_pass and b_pass and c_pass), "status": status})
    out = pd.DataFrame(rows)
    met = bool(len(out) == len(crit["qs"]) and out["met_q"].all())
    verdict = {c: "" for c in out.columns}
    verdict.update(q="both", met_q=met, status="MET" if met else "NOT MET (the line stops; no other slope, map or variant follows)")
    return pd.concat([out, pd.DataFrame([verdict])], ignore_index=True)


def paired_table(df: pd.DataFrame, crit: Mapping[str, Any], pairs: Sequence[Tuple[str, str]]
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Paired descriptive statistics ``arm - baseline`` of :data:`PAIRED_METRICS` per q, and the seed-level values."""
    n_boot, seed = int(crit["bootstrap"]["n_resamples"]), int(crit["bootstrap"]["seed_e1"])
    rows, longs = [], []
    for arm, base in pairs:
        for q in crit["qs"]:
            a = _complete(df, arm, q).set_index("seed")
            b = _complete(df, base, q).set_index("seed")
            common = sorted(set(a.index) & set(b.index))
            for m, lower in PAIRED_METRICS:
                if m not in a.columns:
                    continue
                x = a.loc[common, m].astype(float).to_numpy()
                y = b.loc[common, m].astype(float).to_numpy()
                st = C.paired_stats(x, y, n_boot, seed)
                rows.append({"comparison": f"{arm} - {base}", "arm": arm, "baseline": base, "q": q, "metric": m,
                             "direction": "lower is better" if lower else "none", "n_expected": int(crit["seeds"]["n_per_q"]),
                             **st, "ci_contains_0": bool(st["ci_lo"] <= 0.0 <= st["ci_hi"]) if st["n_pairs"] else False,
                             "ci_hi_below_0": bool(st["n_pairs"] and st["ci_hi"] < 0.0)})
                for s_, xv, yv in zip(common, x, y):
                    longs.append({"comparison": f"{arm} - {base}", "q": q, "metric": m, "seed": s_, "arm_value": xv,
                                  "base_value": yv, "diff": xv - yv})
    return pd.DataFrame(rows), pd.DataFrame(longs)


def failure_rate_table(df: pd.DataFrame, crit: Mapping[str, Any], arms: Sequence[str]) -> pd.DataFrame:
    """Runs that fail a gate or exit nonzero per arm and q (and pooled over q), with exact Clopper-Pearson 95 % intervals."""
    n_exp = int(crit["seeds"]["n_per_q"])
    rows = []
    for arm in arms:
        pooled_k = pooled_n = 0
        for q in list(crit["qs"]) + ["both"]:
            if q == "both":
                k, n = pooled_k, pooled_n
            else:
                s = C.sel(df, arm, q)
                k = int(failed_run_mask(s).sum()) + max(0, n_exp - len(s))
                n = n_exp
                pooled_k += k
                pooled_n += n
            lo, hi = C.clopper_pearson(k, n)
            rows.append({"arm": arm, "q": q, "n_runs": n, "n_fail": k, "rate": k / n if n else float("nan"),
                         "cp95_lo": lo, "cp95_hi": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- diagnostics
def failure_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per failed run: which gates failed, the exit state and the tie-mean bound flags."""
    keep = ["arm", "q", "seed", "status", "status_info", "exit_code", "nonzero_exit", "gate_G_A", "gate_G_F", "gate_G_N",
            "gate_G_S", "peak_abs", "peak_signed", "e2_at_0", "g2_at_0", "tie_dist_lower", "tie_dist_upper",
            "tie_at_lower_bound", "tie_at_upper_bound", "stage2_rmse_pos_over_g2_0", "eta_T_over_dw", "t2_R0_final", "run_dir"]
    f = df[failed_run_mask(df)]
    return f[[c for c in keep if c in f.columns]].reset_index(drop=True)


def failure_trajectories(fail: pd.DataFrame, curves: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Trajectories (every 25 local updates) of e_hat_2(0), R0, RMSE_pos/e2*(0) (online development-tier checks of
    ``ms_checks_stage2.csv``) and the symmetry error (final tier, offline curve) of every failed run."""
    cols = ["arm", "q", "seed", "local", "e2_at_0", "R0", "stage2_rmse_pos_over_g2_0", "stage2_peak_rel_err_abs", "sym_err_max"]
    out: List[pd.DataFrame] = []
    for r in fail.itertuples(index=False):
        p = Path(r.run_dir) / "ms_checks_stage2.csv"
        if not p.exists():
            continue
        ck = pd.read_csv(p)
        t = pd.DataFrame({"arm": r.arm, "q": r.q, "seed": r.seed, "local": ck["local"],
                          "e2_at_0": ck.get("e2_at_0"), "R0": ck.get("R0", np.nan),
                          "stage2_rmse_pos_over_g2_0": ck.get("stage2_rmse_pos_over_g2_0"),
                          "stage2_peak_rel_err_abs": ck.get("stage2_peak_rel_err_abs")})
        t["sym_err_max"] = np.nan
        if curves is not None and len(curves):
            cv = curves[(curves["arm"] == r.arm) & (curves["q"] == r.q) & (curves["seed"] == r.seed)].set_index("local")
            if len(cv):
                t["sym_err_max"] = t["local"].map(cv["final_stage2_sym_err_max"])
        out.append(t)
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def unit_activity(arrays: Mapping[str, np.ndarray], spec: Any, state_step: float = 4.0) -> Dict[str, np.ndarray]:
    """Share of the development nodes of D_2 with a positive pre-activation per hidden unit, first and second layer
    (post hoc, descriptive; the export's own variant: ReLU or LeakyReLU units).

    Raises:
        ValueError: A variant without piecewise-linear hidden units (tanh).
    """
    from run.run_final_dp_br import dense_grid
    variant = str(np.asarray(arrays["actor_variant"])) if "actor_variant" in arrays else "t1"
    if variant not in ("relu", "lrsm"):
        raise ValueError(f"unit activity is defined for relu / lrsm exports; got {variant!r}")
    x = spec.encode_obs(2, dense_grid(spec.domain_half(2), state_step)).astype(np.float32)
    z1 = x @ arrays["actor.l1.weight"].T + arrays["actor.l1.bias"]
    h1 = np.maximum(z1, np.float32(0.0)) if variant == "relu" else np.where(z1 > 0, z1, z1 * np.float32(0.01)).astype(np.float32)
    z2 = h1 @ arrays["actor.l2.weight"].T + arrays["actor.l2.bias"]
    return {"layer1": (z1 > 0).mean(axis=0), "layer2": (z2 > 0).mean(axis=0), "variant": np.asarray(variant)}


def failure_units_table(fail: pd.DataFrame, pilot_root: Path) -> pd.DataFrame:
    """``unit_activity`` of the last terminal-stage export of every failed relu / lrsm run (one row per unit)."""
    from envs.curriculum_env import GameSpec
    cols = ["arm", "q", "seed", "layer", "unit", "share_positive"]
    rows = []
    for r in fail.itertuples(index=False):
        if not str(r.arm).startswith(("relu", "lrsm")):
            continue
        d = Path(r.run_dir)
        try:
            cfg = json.load(open(d / "run_config.json"))
            g = cfg["record"]["game"]
            spec = GameSpec(**{k: g[k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")})
            budget = int(cfg["pipeline"]["budgets"]["2"])
            with np.load(d / "weights" / f"u{budget:05d}.npz") as z:
                arrays = {k: z[k] for k in z.files}
            act = unit_activity(arrays, spec)
        except Exception:  # noqa: BLE001 - a failed run may lack its exports
            continue
        for layer in ("layer1", "layer2"):
            for u, v in enumerate(act[layer]):
                rows.append({"arm": r.arm, "q": r.q, "seed": r.seed, "layer": layer, "unit": u, "share_positive": float(v)})
    return pd.DataFrame(rows, columns=cols)


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm and q cost summary (median and range of the wall times by phase and category)."""
    cols = ["wall_total_sec", "wall_stage2_sec", "wall_stage1_sec", "stage2_train_rollout_sec", "stage2_train_update_sec",
            "stage2_dev_verifier_sec", "stage1_train_rollout_sec", "stage1_train_update_sec", "stage1_dev_verifier_sec",
            "dev_verifier_share_stage2"]
    rows = []
    for (arm, q), g in df.groupby(["arm", "q"], sort=False):
        for c in cols:
            v = g[c].astype(float)
            v = v[np.isfinite(v)]
            rows.append({"arm": arm, "q": q, "quantity": c, "n": len(v), "median": float(v.median()) if len(v) else np.nan,
                         "min": float(v.min()) if len(v) else np.nan, "max": float(v.max()) if len(v) else np.nan,
                         "mean": float(v.mean()) if len(v) else np.nan})
    return pd.DataFrame(rows)


def curve_summary(curves: pd.DataFrame, arms: Sequence[str]) -> pd.DataFrame:
    """Median and inter-quartile band over the seeds of the final-tier curves, per arm, q and update."""
    cols = {"eta": "final_eta_T_over_dw", "peak_abs": "final_stage2_peak_rel_err_abs", "e2_at_0": "final_e2_at_0"}
    rows = []
    for (arm, q, u), g in curves[curves["arm"].isin(arms)].groupby(["arm", "q", "update"], sort=True):
        row: Dict[str, Any] = {"arm": arm, "q": q, "update": u, "n": len(g)}
        for k, c in cols.items():
            v = g[c].astype(float).to_numpy()
            row.update({f"{k}_median": float(np.median(v)), f"{k}_q25": float(np.percentile(v, 25)),
                        f"{k}_q75": float(np.percentile(v, 75))})
        rows.append(row)
    return pd.DataFrame(rows)


def tie_profile(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int], half: float = TIE_HALF_WIDTH) -> pd.DataFrame:
    """e_hat_2(d) for |d| <= ``half`` from the final-tier recovery arrays of the freeze (seed median and quartiles) and
    the closed-form e2*(d) (evaluation only)."""
    from utils.theory_multistage import g2_two_stage
    rows = []
    for arm in arms:
        for q in qs:
            s = _complete(df, arm, q)
            arrs = []
            grid = None
            for r in s.itertuples(index=False):
                f = Path(r.run_dir) / "freeze_stage2_final.npz"
                if not f.exists():
                    continue
                with np.load(f) as z:
                    d, e = z["recovery_d_grid"], z["recovery_e2"]
                m = np.abs(d) <= half
                grid = d[m]
                arrs.append(e[m])
            if not arrs:
                continue
            a = np.stack(arrs)
            cfg = json.load(open(Path(s.iloc[0]["run_dir"]) / "run_config.json"))["record"]["game"]
            star = g2_two_stage(grid, float(cfg["q"]), float(cfg["w_h"]), float(cfg["w_l"]), float(cfg["k"]), float(cfg["e_max"]))
            for j, d in enumerate(grid):
                rows.append({"arm": arm, "q": q, "d": float(d), "n": a.shape[0], "median": float(np.median(a[:, j])),
                             "q25": float(np.percentile(a[:, j], 25)), "q75": float(np.percentile(a[:, j], 75)),
                             "closed_form": float(star[j])})
    return pd.DataFrame(rows)


def predictions_table(crit_tab: pd.DataFrame, paired: pd.DataFrame, rates: pd.DataFrame, crit: Mapping[str, Any]) -> pd.DataFrame:
    """D7: E1-P1 and E1-P2 against the outcome (expectations, not tests)."""
    rows = []
    for q in crit["qs"]:
        r = paired[(paired["comparison"] == "lrsm_st_s16 - t1_st_s16") & (paired["q"] == q) & (paired["metric"] == "peak_abs")]
        rr = paired[(paired["comparison"] == "relu_st_s16 - t1_st_s16") & (paired["q"] == q) & (paired["metric"] == "peak_abs")]
        m, lo, hi = PRIOR_RELU_MINUS_T1[q]
        rows.append({"prediction": "E1-P1", "q": q,
                     "statement": "lrsm has a lower mean |peak error| than t1, about as relu had on 10501-10510",
                     "prior_relu_minus_t1": f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]",
                     "observed_lrsm_minus_t1": float(r["mean_diff"].iloc[0]) if len(r) else np.nan,
                     "observed_lrsm_ci": f"[{r['ci_lo'].iloc[0]:+.4f}, {r['ci_hi'].iloc[0]:+.4f}]" if len(r) else "",
                     "observed_relu_minus_t1": float(rr["mean_diff"].iloc[0]) if len(rr) else np.nan,
                     "outcome": "lrsm lower than t1" if len(r) and r["mean_diff"].iloc[0] < 0 else "lrsm not lower"})
    for q in crit["qs"]:
        kr = rates[(rates["arm"] == "relu_st_s16") & (rates["q"] == q)]
        kl = rates[(rates["arm"] == "lrsm_st_s16") & (rates["q"] == q)]
        if len(kr) and len(kl):
            nr, nl = int(kr["n_fail"].iloc[0]), int(kl["n_fail"].iloc[0])
            rows.append({"prediction": "E1-P2", "q": q, "statement": "relu fails at least one gate at q = 50 and lrsm fails fewer",
                         "observed_relu_failures": nr, "observed_lrsm_failures": nl,
                         "outcome": ("relu fails >= 1 and lrsm fewer" if (nr >= 1 and nl < nr) else
                                     "relu fails >= 1, lrsm not fewer" if nr >= 1 else "relu fails none")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- figures
def _plt() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _footer(fig: Any, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=6, color="gray", ha="left", va="bottom")


def fig_peak_by_arm(path: Path, df: pd.DataFrame, crit: Mapping[str, Any], footer: str) -> None:
    plt = _plt()
    arms = [crit["e1"]["arms"]["control"], crit["e1"]["arms"]["second_control"], crit["e1"]["arms"]["candidate"]]
    fig, axes = plt.subplots(1, len(crit["qs"]), figsize=(9, 4), sharey=True)
    for ax, q in zip(np.atleast_1d(axes), crit["qs"]):
        for i, arm in enumerate(arms):
            s = _complete(df, arm, q)
            ax.scatter(np.full(len(s), i) + np.linspace(-0.15, 0.15, max(len(s), 1))[:len(s)], s["peak_abs"], s=18,
                       color=ARM_COLORS.get(arm, "k"), label=arm)
            if len(s):
                ax.hlines(s["peak_abs"].mean(), i - 0.3, i + 0.3, color="k", lw=1.5)
        ax.axhline(float(crit["e1"]["peak_abs_mean_max"]), color="gray", ls="--", lw=1)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a.replace("_st_s16", "") for a in arms])
        ax.set_title(f"q = {q}")
        ax.set_xlabel("arm (bar = mean; dashed = 0.04)")
    np.atleast_1d(axes)[0].set_ylabel("|peak error| of the frozen terminal stage (final tier)")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_paired(path: Path, longs: pd.DataFrame, paired: pd.DataFrame, crit: Mapping[str, Any], footer: str) -> None:
    plt = _plt()
    comps = list(dict.fromkeys(paired["comparison"]))
    fig, axes = plt.subplots(1, len(crit["qs"]), figsize=(9, 4), sharey=True)
    for ax, q in zip(np.atleast_1d(axes), crit["qs"]):
        for i, c in enumerate(comps):
            s = longs[(longs["comparison"] == c) & (longs["q"] == q) & (longs["metric"] == "peak_abs")]
            ax.scatter(np.full(len(s), i) + np.linspace(-0.15, 0.15, max(len(s), 1))[:len(s)], s["diff"], s=14, alpha=0.7)
            p = paired[(paired["comparison"] == c) & (paired["q"] == q) & (paired["metric"] == "peak_abs")]
            if len(p):
                ax.errorbar([i], [p["mean_diff"].iloc[0]], yerr=[[p["mean_diff"].iloc[0] - p["ci_lo"].iloc[0]],
                                                                 [p["ci_hi"].iloc[0] - p["mean_diff"].iloc[0]]],
                            color="k", capsize=5, marker="o")
        ax.axhline(0.0, color="gray", lw=1)
        ax.set_xticks(range(len(comps)))
        ax.set_xticklabels([c.replace("_st_s16", "") for c in comps], rotation=12, fontsize=8)
        ax.set_title(f"q = {q}")
    np.atleast_1d(axes)[0].set_ylabel("paired difference of |peak error| (mean with 95 % interval)")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_tie_profile(path: Path, tp: pd.DataFrame, crit: Mapping[str, Any], footer: str) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(crit["qs"]), figsize=(10, 4))
    for ax, q in zip(np.atleast_1d(axes), crit["qs"]):
        for arm, g in tp[tp["q"] == q].groupby("arm"):
            ax.plot(g["d"], g["median"], color=ARM_COLORS.get(arm, "k"), label=arm)
            ax.fill_between(g["d"], g["q25"], g["q75"], color=ARM_COLORS.get(arm, "k"), alpha=0.15)
        g0 = tp[tp["q"] == q]
        if len(g0):
            g1 = g0[g0["arm"] == g0["arm"].iloc[0]]
            ax.plot(g1["d"], g1["closed_form"], "k--", lw=1, label="e2*(d) (closed form)")
        ax.set_title(f"q = {q}")
        ax.set_xlabel("d")
    np.atleast_1d(axes)[0].set_ylabel("effort e_hat_2(d)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_learning(path: Path, cs: pd.DataFrame, crit: Mapping[str, Any], footer: str) -> None:
    plt = _plt()
    fig, axes = plt.subplots(2, len(crit["qs"]), figsize=(10, 7), sharex=True, squeeze=False)
    for j, q in enumerate(crit["qs"]):
        for i, (k, lab) in enumerate((("eta", "eta_2 / DW (final tier)"), ("peak_abs", "|peak error| (final tier)"))):
            ax = axes[i][j]
            for arm, g in cs[cs["q"] == q].groupby("arm"):
                ax.plot(g["update"], g[f"{k}_median"], color=ARM_COLORS.get(arm, "k"), label=arm)
                ax.fill_between(g["update"], g[f"{k}_q25"], g[f"{k}_q75"], color=ARM_COLORS.get(arm, "k"), alpha=0.15)
            ax.set_yscale("log")
            ax.set_title(f"q = {q}")
            if j == 0:
                ax.set_ylabel(lab)
            ax.axvline(2001, color="gray", lw=0.6, ls=":")
            ax.axvline(2200, color="gray", lw=0.6, ls=":")
            if i == 1:
                ax.set_xlabel("terminal-stage update")
    axes[0][0].legend(fontsize=7)
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_failed(path: Path, traj: pd.DataFrame, footer: str) -> bool:
    if traj.empty:
        return False
    plt = _plt()
    keys = list(traj[["arm", "q", "seed"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for k in keys:
        g = traj[(traj["arm"] == k[0]) & (traj["q"] == k[1]) & (traj["seed"] == k[2])]
        for ax, c in zip(axes, ("e2_at_0", "R0", "stage2_rmse_pos_over_g2_0", "sym_err_max")):
            ax.plot(g["local"], g[c], label=f"{k[0]} q{k[1]} s{k[2]}")
            ax.set_ylabel(c)
    axes[0].legend(fontsize=6)
    axes[-1].set_xlabel("terminal-stage update")
    _footer(fig, footer)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------------------------- driver
def summary_text(crit_tab: pd.DataFrame, rates: pd.DataFrame, roots: Mapping[str, str]) -> str:
    lines = ["MS-R4 E1 analysis (tools/ms/r4_e1_analysis.py)", f"roots: {json.dumps(dict(roots))}", ""]
    for r in crit_tab.itertuples(index=False):
        if r.q == "both":
            lines.append(f"VERDICT: {r.status}")
            continue
        lines.append(f"q = {r.q}: mean |peak| (lrsm) = {r.mean_abs_peak_lrsm:.5f} (a: {r.a_pass}); paired lrsm - t1 = "
                     f"{r.mean_diff_vs_t1:+.5f} [{r.ci_lo:+.5f}, {r.ci_hi:+.5f}] (b: {r.b_pass}); runs failing a gate or "
                     f"exiting nonzero = {r.n_gate_fail} of {r.n_expected} (c: {r.c_pass}); status {r.status}")
    lines.append("")
    for r in rates.itertuples(index=False):
        lines.append(f"{r.arm} q={r.q}: {r.n_fail} of {r.n_runs} runs fail (CP95 [{r.cp95_lo:.3f}, {r.cp95_hi:.3f}])")
    lines += ["", "The effect is that of the combined change (LeakyReLU hidden units and the smooth unclamped mean map); no part is "
              "attributed to either. Zero failures in 40 runs bound the per-run failure rate by 8.8 % (CP95 upper limit); a candidate "
              "with relu_st_s16's observed q = 50 rate (1 of 10) passes all 20 q = 50 runs with probability 0.9^20 = 0.12: passing "
              "E1 screens out relu-sized failure rates, it does not establish robustness."]
    return "\n".join(lines) + "\n"


def run_analysis(pilot_root: Path, curves_root: Optional[Path], criteria_path: Path, out: Path,
                 protocol_path: Path = ROOT / "protocols" / "v2_T2_locked_v2_0.json") -> Dict[str, Any]:
    """The whole E1 analysis; returns the info dict (also written to ``analysis_info.json``)."""
    import ms_configs as mc
    crit, crit_sha = C.load_criteria(criteria_path)
    th, proto_sha = R1.load_thresholds(protocol_path)
    proto = mc.load_protocol(protocol_path)
    e1 = crit["e1"]
    arms = [e1["arms"]["control"], e1["arms"]["second_control"], e1["arms"]["candidate"]]
    seeds = list(range(int(crit["seeds"]["pilot"][0]), int(crit["seeds"]["pilot"][1]) + 1))
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    df = C.build_per_run(pilot_root, arms, crit["qs"], seeds, th, proto, {a: "E1" for a in arms})
    df = df.copy()
    df.insert(0, "roots_id", R1.roots_id({"pilot": str(pilot_root), "curves": str(curves_root or "")}))
    df.to_csv(out / "per_run.csv", index=False)
    crit_tab = criterion_e1(df, crit)
    crit_tab.to_csv(out / "criterion_e1.csv", index=False)
    cand, ctrl, rel = e1["arms"]["candidate"], e1["arms"]["control"], e1["arms"]["second_control"]
    paired, longs = paired_table(df, crit, [(cand, ctrl), (cand, rel), (rel, ctrl)])
    paired.to_csv(out / "paired_e1.csv", index=False)
    longs.to_csv(out / "paired_seed_level_e1.csv", index=False)
    rates = failure_rate_table(df, crit, arms)
    rates.to_csv(out / "failure_rates.csv", index=False)
    fail = failure_rows(df)
    fail.to_csv(out / "failures.csv", index=False)
    curves = None
    if curves_root is not None and (Path(curves_root) / "curves_all.csv").exists():
        curves = pd.read_csv(Path(curves_root) / "curves_all.csv")
    traj = failure_trajectories(fail, curves)
    traj.to_csv(out / "failure_trajectories.csv", index=False)
    failure_units_table(fail, pilot_root).to_csv(out / "failure_units.csv", index=False)
    cost_table(df).to_csv(out / "cost_e1.csv", index=False)
    strata = R1.strata_table(df)                                       # D5: the per-stratum errors at the freeze (both tiers)
    strata.to_csv(out / "strata.csv", index=False)
    R1.strata_summary_table(strata).to_csv(out / "strata_summary.csv", index=False)
    preds = predictions_table(crit_tab, paired, rates, crit)
    preds.to_csv(out / "predictions_e1.csv", index=False)
    footer = f"pilot root: {pilot_root}; criteria sha256 {crit_sha[:12]}"
    figs = []
    fig_peak_by_arm(out / "figures" / "peak_abs_by_arm.png", df, crit, footer)
    fig_paired(out / "figures" / "paired_differences.png", longs, paired, crit, footer)
    figs += ["peak_abs_by_arm.png", "paired_differences.png"]
    tp = tie_profile(df, arms, crit["qs"])
    tp.to_csv(out / "tie_profile_e1.csv", index=False)
    if len(tp):
        fig_tie_profile(out / "figures" / "tie_profile.png", tp, crit, footer)
        figs.append("tie_profile.png")
    if curves is not None and len(curves):
        cs = curve_summary(curves, arms)
        cs.to_csv(out / "learning_curves_e1.csv", index=False)
        fig_learning(out / "figures" / "learning_curves.png", cs, crit, footer)
        figs.append("learning_curves.png")
    if fig_failed(out / "figures" / "failed_runs.png", traj, footer):
        figs.append("failed_runs.png")
    roots = {"pilot": str(pilot_root), "curves": str(curves_root or ""), "criteria": str(criteria_path)}
    (out / "summary.txt").write_text(summary_text(crit_tab, rates, roots))
    n_missing = int((df["status"].astype(str) != "done").sum())
    info = {"tool": "tools/ms/r4_e1_analysis.py", "roots": roots, "criteria_sha256": crit_sha, "protocol_sha256": proto_sha,
            "arms": arms, "qs": list(crit["qs"]), "seeds": seeds, "n_planned": len(df), "n_not_done": n_missing,
            "figures": figs, "verdict": str(crit_tab["status"].iloc[-1])}
    with open(out / "analysis_info.json", "w") as f:
        json.dump(info, f, indent=1)
    return info


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("--pilot-root", required=True)
    p.add_argument("--curves-root", default=None)
    p.add_argument("--criteria", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error(f"{out} exists and is not empty")
    info = run_analysis(Path(a.pilot_root), Path(a.curves_root) if a.curves_root else None, Path(a.criteria), out)
    print(json.dumps({k: info[k] for k in ("n_planned", "n_not_done", "verdict")}, indent=1))
    return 0 if info["n_not_done"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
