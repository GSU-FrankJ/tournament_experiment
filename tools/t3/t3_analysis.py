#!/usr/bin/env python3
"""Analysis of round T3-R1 (D7, prompt section 3.3): the pre-registered paired comparison, the secondary statistics,
the gate counts, the diagnosis of every ``bw`` failure, best of run, coverage, cost, the learning curves and the figures.

Inputs: ``per_run.csv`` and ``best_of_run.csv`` of ``tools/t3/t3_eval.py``, the run directories (cost, coverage, curves),
``reports/t3/r1/prereg_criteria.json`` (read by this tool; the blind tool ``t3_blind.py`` follows it independently).
The statistics are those of ``tools/ms/r4_common.py`` (``paired_stats``, ``boot_mean_ci``, ``boot_indices``,
``clopper_pearson``), whose definition is D7's: ``numpy.random.default_rng(S).integers(0, n, size=(10000, n))`` created afresh
for every (hypothesis, q, statistic), percentile intervals of the resampled means (``numpy.percentile``, linear method).
Failed runs: every planned run is a row; its metrics are missing and its pair is dropped from the paired statistics.

Usage:
    python tools/t3/t3_analysis.py --root results/t3_r1 --eval-dir results/t3_r1/eval --out-dir results/t3_r1/analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

import r4_common as R4  # noqa: E402
import t3_configs as tc  # noqa: E402
import t3_metrics as tm  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402

QS = tc.QS
ARM_ORDER = ("bw", "jt", "jt_es", "v20", "jt2")
HORIZON = {"bw": 3, "jt": 3, "jt_es": 3, "v20": 2, "jt2": 2}
BOOL_COLS = ("gate_G_A", "gate_G_F", "gate_G_N", "gate_G_S", "gates_pass", "ga_eta", "ga_rmse", "ga_tail", "gn_eta",
             "gn_gmax", "gs_s1", "valid", "dev_valid", "global_rng_ok")


# ---------------------------------------------------------------------------------- loading
def _truthy(v: Any) -> bool:
    return v is True or (isinstance(v, str) and v == "True") or (isinstance(v, (int, float, np.bool_)) and not isinstance(v, bool)
                                                                 and not (isinstance(v, float) and math.isnan(v)) and bool(v))


def load_per_run(path: Path) -> pd.DataFrame:
    """The per-run table of ``t3_eval.py``: one row per planned run; the boolean columns as Python bools (False where
    missing), ``exit_ok`` and ``passed`` (all gates and a clean exit and the global-RNG assertion) added."""
    df = pd.read_csv(path)
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].map(_truthy)
    df = df.copy()
    df["exit_ok"] = (df["status"] == "done") & (df["exit_code"] == 0)
    df["passed"] = df["exit_ok"] & df["gates_pass"] & df["global_rng_ok"]
    return df.copy()


def planned_seeds(arm: str) -> Tuple[int, ...]:
    """The planned seeds of an arm: 50501-50520 (T = 3 arms) or 30501-30520 (``v20``, ``jt2``)."""
    return tc.WAVE_SEEDS["t3"] if HORIZON[arm] == 3 else tc.WAVE_SEEDS["t2j"]


def series(df: pd.DataFrame, arm: str, q: int, stat: str) -> np.ndarray:
    """The values of ``stat`` for the planned seeds of (arm, q), in seed order; NaN where the run is missing, failed or
    has no value."""
    sub = df[(df["arm"] == arm) & (df["q"] == q)].set_index("seed")
    out = []
    for s in planned_seeds(arm):
        v = float("nan")
        if s in sub.index and bool(sub.loc[s, "exit_ok"]) and stat in sub.columns:
            x = sub.loc[s, stat]
            v = float(x) if pd.notna(x) else float("nan")
        out.append(v)
    return np.array(out)


# ---------------------------------------------------------------------------------- paired statistics
def paired_row(x: np.ndarray, y: np.ndarray, S: int, n_boot: int, pct95: Sequence[float], pct9875: Sequence[float]
               ) -> Dict[str, Any]:
    """Paired statistics of a - b over the complete pairs: the mean with its 95 % and 98.75 % percentile intervals, the
    median with its 95 % interval (the same bootstrap indices), the sign count and the dropped pairs."""
    st = R4.paired_stats(x, y, n_boot, S)
    ok = np.isfinite(x) & np.isfinite(y)
    d = (np.asarray(x, float) - np.asarray(y, float))[ok]
    lo95, hi95 = R4.boot_mean_ci(d, n_boot, S, tuple(pct95))          # the percentiles of the criteria file
    lo98, hi98 = R4.boot_mean_ci(d, n_boot, S, tuple(pct9875))
    if d.size:
        med = np.median(d[R4.boot_indices(d.size, n_boot, S)], axis=1)
        mlo, mhi = float(np.percentile(med, pct95[0])), float(np.percentile(med, pct95[1]))
    else:
        mlo = mhi = float("nan")
    return {"n_pairs": st["n_pairs"], "dropped_pairs": int(len(x) - st["n_pairs"]), "mean_a": st["mean_a"],
            "mean_b": st["mean_b"], "mean_diff": st["mean_diff"], "ci95_lo": lo95, "ci95_hi": hi95,
            "ci98_75_lo": lo98, "ci98_75_hi": hi98, "median_diff": st["median_diff"], "median_ci95_lo": mlo,
            "median_ci95_hi": mhi, "n_neg": st["n_neg"], "n_pos": st["n_pos"]}


def criterion_table(df: pd.DataFrame, crit: Mapping[str, Any]) -> pd.DataFrame:
    """H1, H2, H3 and H1-T2 per q on the primary statistic (final-tier ``Gmax_full_over_dw``): one row per
    (hypothesis, q) with the intervals, the verdict at that q (upper end of the 95 % interval strictly below 0) and the
    hypothesis verdict (both q)."""
    stat = crit["primary"]["statistic"]
    n_boot = int(crit["statistics"]["n_boot"])
    p95, p9875 = crit["statistics"]["percentiles"]["ci95"], crit["statistics"]["percentiles"]["ci98_75"]
    rows: List[Dict[str, Any]] = []
    for h, spec in crit["hypotheses"].items():
        S = int(crit["statistics"]["bootstrap_seed"][str(spec["horizon"])])
        for q in QS:
            r = paired_row(series(df, spec["a"], q, stat), series(df, spec["b"], q, stat), S, n_boot, p95, p9875)
            r.update({"hypothesis": h, "horizon": spec["horizon"], "a": spec["a"], "b": spec["b"], "q": q,
                      "statistic": stat, "bootstrap_seed": S, "ci_hi_below_0": bool(r["ci95_hi"] < 0.0)})
            rows.append(r)
    out = pd.DataFrame(rows)
    verdict = out.groupby("hypothesis")["ci_hi_below_0"].all().to_dict()
    fails = out[~out["ci_hi_below_0"]].groupby("hypothesis")["q"].apply(lambda s: ";".join(str(int(v)) for v in s)).to_dict()
    out["hypothesis_supported"] = out["hypothesis"].map(verdict)
    out["fails_at_q"] = out["hypothesis"].map(lambda h: fails.get(h, ""))
    first = ["hypothesis", "horizon", "a", "b", "q", "statistic", "n_pairs", "dropped_pairs", "mean_a", "mean_b", "mean_diff",
             "ci95_lo", "ci95_hi", "ci_hi_below_0", "hypothesis_supported", "fails_at_q", "ci98_75_lo", "ci98_75_hi",
             "median_diff", "median_ci95_lo", "median_ci95_hi", "n_neg", "n_pos", "bootstrap_seed"]
    return out[first + [c for c in out.columns if c not in first]]


def secondary_stats(horizon: int) -> List[str]:
    """The secondary statistics of D7 that exist at a horizon (column names of the per-run table)."""
    T = int(horizon)
    cols = ["EXP_root_over_dw", "dReach_over_dw", "rmse_pos_T_over_eT0", "tail_mean_T_over_eT0", "peak_T_abs", "e1_at_0"]
    for t in range(1, T + 1):
        cols += [f"Delta_max_t{t}_over_dw", f"G_max_t{t}_over_dw"]
    for t in range(2, T + 1):
        cols += [f"Delta_in_t{t}", f"Delta_out_t{t}", f"G_in_t{t}", f"G_out_t{t}"]
    if T == 2:
        cols.append("stage1_rel_err_abs")
    return cols


def secondary_table(df: pd.DataFrame, crit: Mapping[str, Any]) -> pd.DataFrame:
    """The same paired statistics for each of the four pairs and every secondary statistic (descriptive)."""
    n_boot = int(crit["statistics"]["n_boot"])
    p95, p9875 = crit["statistics"]["percentiles"]["ci95"], crit["statistics"]["percentiles"]["ci98_75"]
    rows: List[Dict[str, Any]] = []
    for h, spec in crit["hypotheses"].items():
        S = int(crit["statistics"]["bootstrap_seed"][str(spec["horizon"])])
        for stat in secondary_stats(spec["horizon"]):
            for q in QS:
                r = paired_row(series(df, spec["a"], q, stat), series(df, spec["b"], q, stat), S, n_boot, p95, p9875)
                r.update({"hypothesis": h, "a": spec["a"], "b": spec["b"], "q": q, "statistic": stat})
                rows.append(r)
    out = pd.DataFrame(rows)
    first = ["hypothesis", "a", "b", "q", "statistic"]
    return out[first + [c for c in out.columns if c not in first]]


# ---------------------------------------------------------------------------------- gate counts
def gate_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, q): planned runs, runs with a clean exit, passes of G-A, G-F, G-N, (G-S at T = 2) and of the run pass, each
    with the exact Clopper-Pearson 95 % interval of k / planned."""
    rows: List[Dict[str, Any]] = []
    for arm in ARM_ORDER:
        for q in QS:
            sub = df[(df["arm"] == arm) & (df["q"] == q)]
            if sub.empty:
                continue
            n = len(planned_seeds(arm))
            r: Dict[str, Any] = {"arm": arm, "q": q, "horizon": HORIZON[arm], "n_planned": n,
                                 "n_exit_ok": int(sub["exit_ok"].sum())}
            ok = sub[sub["exit_ok"]]
            gates = [("G-A", "gate_G_A"), ("G-F", "gate_G_F"), ("G-N", "gate_G_N")]
            if HORIZON[arm] == 2:
                gates.append(("G-S", "gate_G_S"))
            for name, col in gates:
                k = int(ok[col].sum()) if col in ok.columns else 0
                lo, hi = R4.clopper_pearson(k, n)
                r.update({f"{name}_pass": k, f"{name}_lo": lo, f"{name}_hi": hi})
            k = int(sub["passed"].sum())
            lo, hi = R4.clopper_pearson(k, n)
            r.update({"run_pass": k, "run_pass_lo": lo, "run_pass_hi": hi})
            rows.append(r)
    return pd.DataFrame(rows)


def reading_rule(counts: pd.DataFrame, crit: Mapping[str, Any]) -> Dict[str, Any]:
    """D7's reading of ``bw`` at T = 3: at least 18 of 20 runs pass G-A and G-F and G-N at both q."""
    need = int(crit["reading_rule"]["min_passes_of_20_at_each_q"])
    bw = counts[counts["arm"] == "bw"].set_index("q")
    per_q = {int(q): int(bw.loc[q, "run_pass"]) for q in bw.index}
    return {"min_passes": need, "passes_by_q": per_q, "candidate_for_a_lock": bool(len(per_q) == len(QS)
                                                                                   and all(v >= need for v in per_q.values()))}


# ---------------------------------------------------------------------------------- bw diagnosis
def bw_gate_table(df: pd.DataFrame, crit: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """The T = 3 gate table of ``bw`` (one row per planned run) with the per-gate, per-stage diagnosis of every failure:
    which criteria fail, the values, the stage and gap of the largest dynamic gain and whether it lies in band. The
    thresholds quoted in the text are those of the criteria file."""
    if crit is None:
        crit, _ = tm.load_criteria()
    g = crit["gates"]
    th_eta, th_rmse, th_tail = (g["G-A"][k] for k in ("eta_T_over_dw", "rmse_pos_T_over_eT0", "tail_mean_T_over_eT0"))
    th_f = g["G-F"]["Gmax_full_over_dw"]
    th_n = g["G-N"]["eta_T_over_dw_dev_minus_final_abs"], g["G-N"]["Gmax_full_over_dw_dev_minus_final_abs"]
    sub = df[df["arm"] == "bw"].sort_values(["q", "seed"])
    rows = []
    for _, r in sub.iterrows():
        row: Dict[str, Any] = {"q": int(r["q"]), "seed": int(r["seed"]), "exit_ok": bool(r["exit_ok"]), "passed": bool(r["passed"])}
        if not r["exit_ok"] or pd.isna(r["Gmax_full_over_dw"]):
            code = "none" if pd.isna(r["exit_code"]) else int(r["exit_code"])
            err = r["eval_error"] if "eval_error" in r.index and isinstance(r["eval_error"], str) and r["eval_error"] else ""
            row["diagnosis"] = f"no result (status {r['status']}, exit code {code}{', evaluation error: ' + err if err else ''})"
            rows.append(row)
            continue
        for c in ("gate_G_A", "gate_G_F", "gate_G_N"):
            row[c] = bool(r[c])
        row.update({c: r[c] for c in ("eta_T_over_dw", "rmse_pos_T_over_eT0", "tail_mean_T_over_eT0", "Gmax_full_over_dw",
                                      "Gmax_full_t", "Gmax_full_d", "dev_eta_T_over_dw", "dev_Gmax_full_over_dw", "peak_T_signed")})
        for t in (1, 2, 3):
            row[f"G_max_t{t}_over_dw"] = r[f"G_max_t{t}_over_dw"]
            row[f"Delta_max_t{t}_over_dw"] = r[f"Delta_max_t{t}_over_dw"]
        for t in (2, 3):
            for c in ("G_in", "G_out", "Delta_in", "Delta_out"):
                row[f"{c}_t{t}"] = r[f"{c}_t{t}"]
        why = []
        if not r["ga_eta"]:
            why.append(f"G-A eta_3 {r['eta_T_over_dw']:.5f} > {th_eta:g}")
        if not r["ga_rmse"]:
            why.append(f"G-A rmse_pos {r['rmse_pos_T_over_eT0']:.4f} > {th_rmse:g}")
        if not r["ga_tail"]:
            why.append(f"G-A tail_mean {r['tail_mean_T_over_eT0']:.4f} > {th_tail:g}")
        if not r["gate_G_F"]:
            t = int(r["Gmax_full_t"])
            side = "out of band" if t >= 2 and r[f"G_out_t{t}"] >= r[f"G_in_t{t}"] else "in band"
            if t == 1:
                side = "root"
            why.append(f"G-F Gmax {r['Gmax_full_over_dw']:.5f} > {th_f:g} at stage {t}, d = {r['Gmax_full_d']:g} ({side})")
        if not r["gn_eta"]:
            why.append(f"G-N |eta_3(dev) - eta_3(final)| = {abs(r['dev_eta_T_over_dw'] - r['eta_T_over_dw']):.5f} > {th_n[0]:g}")
        if not r["gn_gmax"]:
            why.append(f"G-N |Gmax(dev) - Gmax(final)| = {abs(r['dev_Gmax_full_over_dw'] - r['Gmax_full_over_dw']):.5f} > {th_n[1]:g}")
        row["diagnosis"] = "; ".join(why)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- best of run
def best_of_run_table(df: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Per (joint arm, q): last iterate against the selected candidate (selected with the development verifier; not a
    pre-specified candidate): means and medians of ``Gmax_full_over_dw`` and ``EXP_root_over_dw``, the mean selected
    update and the number of runs whose last iterate was selected."""
    rows = []
    for arm in ("jt", "jt_es", "jt2"):
        for q in QS:
            last = df[(df["arm"] == arm) & (df["q"] == q) & df["exit_ok"]].set_index("seed")
            b = best[(best["arm"] == arm) & (best["q"] == q) & (best["status"] == "done")].set_index("seed")
            common = sorted(set(last.index) & set(b.index))
            if not common:
                continue
            r: Dict[str, Any] = {"arm": arm, "q": q, "n": len(common), "label": "selected with the development verifier; "
                                                                                "not a pre-specified candidate"}
            for stat in ("Gmax_full_over_dw", "EXP_root_over_dw"):
                lv = last.loc[common, stat].astype(float).to_numpy()
                bv = b.loc[common, stat].astype(float).to_numpy()
                r.update({f"{stat}_last_mean": float(np.nanmean(lv)), f"{stat}_best_mean": float(np.nanmean(bv)),
                          f"{stat}_last_median": float(np.nanmedian(lv)), f"{stat}_best_median": float(np.nanmedian(bv))})
            r["best_update_mean"] = float(b.loc[common, "best_update"].astype(float).mean())
            r["n_last_iterate_selected"] = int((b.loc[common, "best_source"] == "last_iterate").sum())
            r["n_gates_pass_best"] = int(b.loc[common, "gates_pass"].map(_truthy).sum()) if "gates_pass" in b.columns else 0
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- coverage and cost
def coverage_arrays(root: Path, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Coverage of the joint arms from ``visitation.npz`` (rows per stage and bin, by start stage, and the share of each
    stage's rows in band) and of ``bw`` from the design (expected direct starts per bin; realised stratum counts of
    ``ms_updates.csv``). Returns the table (one row per run and stage) and the per-bin arrays (mean over seeds)."""
    rows: List[Dict[str, Any]] = []
    acc: Dict[str, List[np.ndarray]] = {}
    for _, r in df[df["exit_ok"] & df["arm"].isin(["jt", "jt_es", "jt2"])].iterrows():
        d = Path(r["run_dir"])
        v = np.load(d / "visitation.npz")
        T = int(r["T"])
        for t in range(2, T + 1):
            edges = v[f"stage{t}_bin_edges"]
            mid = 0.5 * (edges[:-1] + edges[1:])
            inb = np.abs(mid) < 2.0 * r["q"] * (t - 1)
            srcs = {k[len(f"stage{t}_"):]: v[k] for k in v.files if k.startswith(f"stage{t}_") and "edges" not in k}
            tot = sum(srcs.values())
            row = {"arm": r["arm"], "q": int(r["q"]), "seed": int(r["seed"]), "stage": t, "rows": int(tot.sum()),
                   "n_bins": int(tot.size), "bins_visited": int((tot > 0).sum()),
                   "share_in_band": float(tot[inb].sum() / tot.sum()),
                   "mean_rows_per_bin_in_band": float(tot[inb].mean()), "mean_rows_per_bin_out_of_band": float(tot[~inb].mean())}
            for k, a in srcs.items():
                row[f"rows_{k}"] = int(a.sum())
            rows.append(row)
            for k, a in srcs.items():
                acc.setdefault(f"{r['arm']}_q{int(r['q'])}_stage{t}_{k}", []).append(a.astype(float))
    for _, r in df[df["exit_ok"] & (df["arm"] == "bw")].iterrows():
        d = Path(r["run_dir"])
        cfg = json.load(open(d / "run_config.json"))
        spec = GameSpec(**cfg["record"]["game"])
        smp = StartSampler(spec, float(cfg["record"]["protocol"]["es_bin_width"]))
        upd = pd.read_csv(d / "ms_updates.csv")
        n_ep = int(cfg["record"]["protocol"]["episodes_per_update"])
        for t in (2, 3):
            nb = smp.n_bins(t)
            sub = upd[upd["stage"] == t]
            edges = smp.bin_edges(t)
            mid = 0.5 * (edges[:-1] + edges[1:])
            inb = np.abs(mid) < 2.0 * r["q"] * (t - 1)
            exp_per_bin = len(sub) * n_ep / nb
            rows.append({"arm": "bw", "q": int(r["q"]), "seed": int(r["seed"]), "stage": t, "rows": len(sub) * n_ep,
                         "n_bins": nb, "bins_visited": nb, "share_in_band": float(inb.mean()),
                         "mean_rows_per_bin_in_band": exp_per_bin, "mean_rows_per_bin_out_of_band": exp_per_bin,
                         "rows_direct_design": len(sub) * n_ep,
                         "n_start_tail": int(sub["n_start_tail"].sum()), "n_start_near": int(sub["n_start_near"].sum()),
                         "n_start_mid": int(sub["n_start_mid"].sum())})
            acc.setdefault(f"bw_q{int(r['q'])}_stage{t}_direct_design", []).append(np.full(nb, exp_per_bin))
    arrays = {k: np.mean(v, axis=0) for k, v in acc.items()}
    return pd.DataFrame(rows), arrays


def cost_row(r: Mapping[str, Any]) -> Dict[str, Any]:
    """Updates, policy rows by stage, minibatch steps, wall and CPU time per phase / category of one run (from its
    summary files)."""
    d = Path(r["run_dir"])
    out: Dict[str, Any] = {"arm": r["arm"], "q": int(r["q"]), "seed": int(r["seed"])}
    nan = float("nan")
    if r["arm"] == "bw":
        s = json.load(open(d / "ms_run_summary.json"))
        g = json.load(open(d / "gates.json"))
        st = json.load(open(d / "status.json"))
        cfg = json.load(open(d / "run_config.json"))
        n_ep = int(cfg["record"]["protocol"]["episodes_per_update"])
        out.update({"updates": s["final_global_update"], "policy_rows_total": s["total_transitions"],
                    "minibatch_steps": g["budget_totals"]["minibatch_steps"], "wall_sec": st["total_wall_sec"]})
        for t in (3, 2, 1):
            ph = s["phase_timing"][f"stage{t}"]
            out.update({f"rows_t{t}": ph["updates"] * n_ep, f"wall_stage{t}_sec": ph["wall_sec"],
                        f"cpu_stage{t}_sec": ph["process_cpu_sec"]})
            for c, v in ph["by_category_sec"].items():
                out[f"stage{t}_{c}_sec"] = v
        out["cpu_sec"] = sum(s["phase_timing"][f"stage{t}"]["process_cpu_sec"] for t in (3, 2, 1))
        for c, v in s["costs"]["seconds"].items():
            out[f"{c}_sec"] = v
    elif r["arm"] == "v20":
        s = json.load(open(d / "v2_run_summary.json"))
        rows = pd.read_csv(d / "v2_updates.csv")
        steps = int((rows["n_epochs_run"] * np.ceil(rows["n_rows"] / 256.0)).sum())
        out.update({"updates": s["final_global_update"], "minibatch_steps": steps,
                    "policy_rows_total": int(rows["n_policy_rows"].sum()), "wall_sec": s["total_wall_sec"]})
        for ph, t in (("A", 2), ("B", 1)):
            sub = rows[rows["phase"] == ph]
            out[f"rows_t{t}"] = int(sub["n_policy_rows"].sum())
            out[f"wall_stage{t}_sec"] = s["phase_timing"][ph]["wall_sec"]
        for c, v in s["costs"]["seconds"].items():
            out[f"{c}_sec"] = v
    else:
        s = json.load(open(d / "joint_run_summary.json"))
        out.update({"updates": s["updates"], "policy_rows_total": s["policy_rows_total"], "minibatch_steps": s["minibatch_steps"],
                    "wall_sec": s["wall_sec"], "cpu_sec": s["process_cpu_sec"], "peak_rss_kb": s["peak_rss_kb"]})
        for t, v in s["policy_rows_by_stage"].items():
            out[f"rows_t{t}"] = v
        for c, v in s["costs_wall_sec"].items():
            out[f"{c}_sec"] = v
        for c, v in s["costs_cpu_sec"].items():
            out[f"{c}_cpu_sec"] = v
    return out


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    """One cost row per completed run."""
    rows = [cost_row(r) for _, r in df[df["exit_ok"]].iterrows()]
    return pd.DataFrame(rows)


def budget_check(cost: pd.DataFrame) -> pd.DataFrame:
    """Per arm: the number of completed runs and how many have exactly the updates, policy rows (per stage and in total) and
    minibatch steps of D5 (``t3_configs.expected_budget``; for ``v20`` from each run's ``v2_updates.csv``)."""
    rows = []
    for arm in ARM_ORDER:
        sub = cost[cost["arm"] == arm]
        if sub.empty:
            continue
        exp = tc.expected_budget(arm)
        ok_stage = np.ones(len(sub), dtype=bool)
        for t, n in exp["rows_by_stage"].items():
            ok_stage &= (sub[f"rows_t{t}"].to_numpy() == n)
        rows.append({"arm": arm, "n_runs": len(sub), "updates_expected": exp["updates"],
                     "updates_equal": int((sub["updates"] == exp["updates"]).sum()),
                     "rows_total_expected": exp["rows_total"], "rows_total_equal": int((sub["policy_rows_total"] == exp["rows_total"]).sum()),
                     "rows_by_stage_equal": int(ok_stage.sum()),
                     "minibatch_steps_expected": exp["minibatch_steps"],
                     "minibatch_steps_equal": int((sub["minibatch_steps"] == exp["minibatch_steps"]).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- learning curves
def joint_curve(run_dir: Path) -> pd.DataFrame:
    """The development checks of a joint run against cumulative policy rows (all stages)."""
    c = pd.read_csv(run_dir / "joint_checks.csv")
    rows_cols = [k for k in c.columns if k.startswith("rows_t")]
    c["cum_rows"] = c[rows_cols].sum(axis=1)
    return c


def bw_curve(run_dir: Path) -> pd.DataFrame:
    """The per-phase checks of a ``bw`` run (each stage's own ``Delta``) against cumulative policy rows."""
    cfg = json.load(open(run_dir / "run_config.json"))
    n_ep = int(cfg["record"]["protocol"]["episodes_per_update"])
    parts = [pd.read_csv(run_dir / f"ms_checks_stage{t}.csv") for t in (3, 2, 1)]
    c = pd.concat(parts, ignore_index=True)
    c["cum_rows"] = c["update"] * n_ep
    return c


# ---------------------------------------------------------------------------------- figures
def _plt() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


COLORS = {"bw": "#1b9e77", "jt": "#d95f02", "jt_es": "#7570b3", "v20": "#1b9e77", "jt2": "#d95f02"}
ARMS_BY_T = {3: ("bw", "jt", "jt_es"), 2: ("v20", "jt2")}
Q_MARK = {50: "o", 60: "s"}


def save_figure(fig: Any, path: Path) -> None:
    """Write a figure (dpi 130) and close it."""
    fig.savefig(path, dpi=130)
    _plt().close(fig)


def _arrays_of(df: pd.DataFrame, arrays_dir: Path, arm: str, q: int) -> List[Any]:
    """The per-run figure arrays (``eval/arrays/{run}.npz``) of the completed runs of (arm, q)."""
    out = []
    for _, r in df[(df["arm"] == arm) & (df["q"] == q) & df["exit_ok"]].sort_values("seed").iterrows():
        p = Path(arrays_dir) / f"{r['run']}.npz"
        if p.exists():
            out.append(np.load(p))
    return out


def _game(q: int) -> Dict[str, Any]:
    return tc.load_protocol()["records"][str(q)]["game"]


def figure_effort(df: pd.DataFrame, arrays_dir: Path, q: int) -> Any:
    """e_t(d) per arm and stage at one q: the seed median with an inter-quartile band (stage 1: the median with the
    inter-quartile bars at d = 0). Rows: T = 3 arms (``bw``, ``jt``, ``jt_es``), T = 2 arms (``v20``, ``jt2``). The closed
    form e_T*(d) is added at the terminal stage of every row, and g1 at stage 1 of the T = 2 row."""
    from utils.theory_multistage import g1_two_stage
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), squeeze=False)
    g = _game(q)
    for row, T in enumerate((3, 2)):
        for t in range(1, 4):
            ax = axes[row, t - 1]
            if t > T:
                ax.axis("off")
                continue
            ref = None
            for arm in ARMS_BY_T[T]:
                zs = _arrays_of(df, arrays_dir, arm, q)
                if not zs:
                    continue
                ref = zs[0]
                d = zs[0][f"fin_t{t}_d_grid"]
                E = np.array([z[f"fin_t{t}_e_hat"] for z in zs])
                med, lo, hi = np.median(E, axis=0), np.percentile(E, 25, axis=0), np.percentile(E, 75, axis=0)
                if t == 1:
                    ax.errorbar([0.0], med, yerr=[med - lo, hi - med], fmt="o", color=COLORS[arm], capsize=4, label=arm)
                else:
                    ax.plot(d, med, color=COLORS[arm], label=arm)
                    ax.fill_between(d, lo, hi, color=COLORS[arm], alpha=0.2)
            if ref is not None and t == T:
                ax.plot(ref["recovery_d_grid"], ref["recovery_gT"], "k--", lw=1.0, label="closed form e_T*(d)")
            if T == 2 and t == 1:
                ax.axhline(g1_two_stage(q, g["w_h"], g["w_l"], g["k"]), color="k", ls="--", lw=1.0, label="g1")
            ax.set_title(f"T = {T}, stage {t}, q = {q}")
            ax.set_xlabel("d")
            ax.set_ylabel("effort")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def figure_gains(df: pd.DataFrame, arrays_dir: Path, q: int) -> Any:
    """Delta_t(d) / dW per arm and stage at one q: the seed median (solid) and the seed maximum (dotted); the band edges
    |d| = 2 q (t - 1) are marked at t >= 2."""
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), squeeze=False)
    dw = float(_game(q)["w_h"] - _game(q)["w_l"])
    for row, T in enumerate((3, 2)):
        for t in range(1, 4):
            ax = axes[row, t - 1]
            if t > T:
                ax.axis("off")
                continue
            for arm in ARMS_BY_T[T]:
                zs = _arrays_of(df, arrays_dir, arm, q)
                if not zs:
                    continue
                d = zs[0][f"fin_t{t}_d_grid"]
                D = np.array([z[f"fin_t{t}_delta"] for z in zs]) / dw
                if t == 1:
                    ax.plot([0.0], np.median(D, axis=0), "o", color=COLORS[arm], label=f"{arm} median")
                    ax.plot([0.0], D.max(axis=0), "x", color=COLORS[arm], label=f"{arm} max")
                else:
                    ax.plot(d, np.median(D, axis=0), color=COLORS[arm], label=f"{arm} median")
                    ax.plot(d, D.max(axis=0), color=COLORS[arm], ls=":", label=f"{arm} max")
            if t >= 2:
                for s in (-1, 1):
                    ax.axvline(s * 2 * q * (t - 1), color="0.4", lw=0.8, ls="--")
            ax.set_title(f"T = {T}, stage {t}, q = {q}")
            ax.set_xlabel("d")
            ax.set_ylabel("Delta_t / dW")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


PAIRED_STATS = ("Gmax_full_over_dw", "EXP_root_over_dw", "G_in_t2", "G_out_t2", "Delta_in_t2", "Delta_out_t2",
                "G_in_t3", "G_out_t3", "Delta_in_t3", "Delta_out_t3")


def paired_forest_rows(df: pd.DataFrame, crit: Mapping[str, Any], stats: Sequence[str] = PAIRED_STATS
                       ) -> pd.DataFrame:
    """Paired statistics (mean difference with the 95 % and 98.75 % intervals) of the figure: every statistic that exists at
    the horizon of a hypothesis, per hypothesis and q."""
    n_boot = int(crit["statistics"]["n_boot"])
    p95, p9875 = crit["statistics"]["percentiles"]["ci95"], crit["statistics"]["percentiles"]["ci98_75"]
    rows = []
    for stat in stats:
        for h, spec in crit["hypotheses"].items():
            if stat not in ["Gmax_full_over_dw", "EXP_root_over_dw"] + secondary_stats(spec["horizon"]):
                continue
            S = int(crit["statistics"]["bootstrap_seed"][str(spec["horizon"])])
            for q in QS:
                r = paired_row(series(df, spec["a"], q, stat), series(df, spec["b"], q, stat), S, n_boot, p95, p9875)
                r.update({"statistic": stat, "hypothesis": h, "a": spec["a"], "b": spec["b"], "q": q})
                rows.append(r)
    return pd.DataFrame(rows)


def figure_paired(df: pd.DataFrame, crit: Mapping[str, Any]) -> Any:
    """Paired differences a - b with their 95 % (thick) and 98.75 % (thin) intervals: ``Gmax_full``, ``EXP_root`` and the
    in-band and out-of-band maxima of the dynamic gain G_t and the one-step gain Delta_t per stage, for the four pairs and
    both q."""
    plt = _plt()
    rows = paired_forest_rows(df, crit)
    stats = [s for s in PAIRED_STATS if s in set(rows["statistic"])]
    ncol = 5
    nrow = int(np.ceil(len(stats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, stat in zip(axes.flat, stats):
        ax.axis("on")
        sub = rows[rows["statistic"] == stat].reset_index(drop=True)
        for i, r in sub.iterrows():
            ax.plot([r["ci98_75_lo"], r["ci98_75_hi"]], [i, i], color="0.7", lw=1)
            ax.plot([r["ci95_lo"], r["ci95_hi"]], [i, i], color=COLORS.get(r["a"], "C0"), lw=3)
            ax.plot([r["mean_diff"]], [i], Q_MARK[int(r["q"])], color="k", ms=4)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f"{r['hypothesis']} q={int(r['q'])}" for _, r in sub.iterrows()], fontsize=7)
        ax.set_title(stat, fontsize=9)
    fig.tight_layout()
    return fig


def figure_curves(df: pd.DataFrame, q: int, T: int = 3) -> Any:
    """Learning curves against cumulative policy rows (seed median with an inter-quartile band): the joint arms' development
    checks (``Gmax_full``, ``EXP_root``, eta_t = ``Delta_max_t`` and ``G_max_t`` per stage) and, for ``bw``, each phase's own
    ``Delta`` at the checks of that phase (drawn in the eta_t panel of the stage)."""
    plt = _plt()
    cols = ["Gmax_full_over_dw", "EXP_root_over_dw"] + [f"Delta_max_t{t}_over_dw" for t in range(1, T + 1)] + \
        [f"G_max_t{t}_over_dw" for t in range(1, T + 1)]
    ncol = 4 if T == 3 else 3
    fig, axes = plt.subplots(2, ncol, figsize=(4.2 * ncol, 7), squeeze=False)
    arms = ("jt", "jt_es") if T == 3 else ("jt2",)
    for ax, col in zip(axes.flat, cols):
        for arm in arms:
            sub = df[(df["arm"] == arm) & (df["q"] == q) & df["exit_ok"]]
            cs = [joint_curve(Path(r["run_dir"])) for _, r in sub.iterrows()]
            if not cs:
                continue
            x = cs[0]["cum_rows"].to_numpy()
            Y = np.array([c[col].to_numpy() for c in cs])
            ax.plot(x, np.maximum(np.median(Y, axis=0), 1e-7), color=COLORS[arm], label=arm)
            ax.fill_between(x, np.maximum(np.percentile(Y, 25, axis=0), 1e-7), np.maximum(np.percentile(Y, 75, axis=0), 1e-7),
                            color=COLORS[arm], alpha=0.2)
        if T == 3 and col.startswith("Delta_max_t"):
            t = int(col[len("Delta_max_t")])
            sub = df[(df["arm"] == "bw") & (df["q"] == q) & df["exit_ok"]]
            cs = [bw_curve(Path(r["run_dir"])) for _, r in sub.iterrows()]
            cs = [c[c["stage"] == t] for c in cs]
            if cs:
                x = cs[0]["cum_rows"].to_numpy()
                Y = np.array([c["Delta"].to_numpy() for c in cs])
                ax.plot(x, np.maximum(np.median(Y, axis=0), 1e-7), color=COLORS["bw"], label=f"bw (phase of stage {t})")
                ax.fill_between(x, np.maximum(np.percentile(Y, 25, axis=0), 1e-7), np.maximum(np.percentile(Y, 75, axis=0), 1e-7),
                                color=COLORS["bw"], alpha=0.2)
        ax.set_yscale("log")
        ax.set_xlabel("cumulative policy rows")
        ax.set_title(f"{col} (q = {q})", fontsize=9)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=7)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def figure_visitation(arrays: Mapping[str, np.ndarray], q: int) -> Any:
    """Visitation per stage (log rows per bin, mean over seeds): the joint arms (all start stages added; ``jt_es`` also by
    source) against ``bw``'s design coverage; panels: T = 3 stages 2 and 3, T = 2 stage 2 (``jt2``); the band edges
    |d| = 2 q (t - 1) are marked."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    B = 100.0 + 2.0 * q
    panels = [(3, 2, ("jt", "jt_es")), (3, 3, ("jt", "jt_es")), (2, 2, ("jt2",))]
    for ax, (T, t, arms) in zip(axes[0], panels):
        half = (t - 1) * B
        for arm in arms:
            parts = {k[len(f"{arm}_q{q}_stage{t}_"):]: v for k, v in arrays.items() if k.startswith(f"{arm}_q{q}_stage{t}_")}
            if not parts:
                continue
            tot = np.sum(list(parts.values()), axis=0)
            x = np.linspace(-half, half, tot.size + 1)
            x = 0.5 * (x[:-1] + x[1:])
            ax.plot(x, np.maximum(tot, 0.5), color=COLORS[arm], label=f"{arm} (all sources)")
            if arm == "jt_es":
                for src, v in parts.items():
                    ax.plot(x, np.maximum(v, 0.5), color=COLORS[arm], ls=":", lw=0.8, label=f"{arm}: {src}")
        d = arrays.get(f"bw_q{q}_stage{t}_direct_design")
        if d is not None and T == 3:
            x = np.linspace(-half, half, d.size + 1)
            ax.plot(0.5 * (x[:-1] + x[1:]), d, color=COLORS["bw"], label="bw design (direct starts)")
        for s in (-1, 1):
            ax.axvline(s * 2 * q * (t - 1), color="0.4", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"T = {T}, stage {t}, q = {q}")
        ax.set_xlabel("d")
        ax.set_ylabel("rows per bin")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------- driver
def run_analysis(root: Path, eval_dir: Path, out_dir: Path, crit: Mapping[str, Any], figures: bool = True) -> Dict[str, Any]:
    """All tables (and figures) of section 3.3; returns the headline numbers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_per_run(eval_dir / "per_run.csv")
    df.to_csv(out_dir / "per_run_table.csv", index=False)
    crit_df = criterion_table(df, crit)
    crit_df.to_csv(out_dir / "criterion.csv", index=False)
    secondary_table(df, crit).to_csv(out_dir / "secondary.csv", index=False)
    counts = gate_counts(df)
    counts.to_csv(out_dir / "gate_counts.csv", index=False)
    bw_gate_table(df, crit).to_csv(out_dir / "bw_t3_gates.csv", index=False)
    best_path = eval_dir / "best_of_run.csv"
    if best_path.exists():
        best_of_run_table(df, pd.read_csv(best_path)).to_csv(out_dir / "best_of_run.csv", index=False)
    cov, arrays = coverage_arrays(root, df)
    cov.to_csv(out_dir / "coverage.csv", index=False)
    np.savez(out_dir / "coverage_arrays.npz", **arrays)
    cost = cost_table(df)
    cost.to_csv(out_dir / "cost.csv", index=False)
    bud = budget_check(cost)
    bud.to_csv(out_dir / "budget_check.csv", index=False)
    if figures:
        fdir = out_dir / "figures"
        fdir.mkdir(exist_ok=True)
        adir = eval_dir / "arrays"
        for q in QS:
            save_figure(figure_effort(df, adir, q), fdir / f"effort_functions_q{q}.png")
            save_figure(figure_gains(df, adir, q), fdir / f"deviation_gains_q{q}.png")
            save_figure(figure_curves(df, q, 3), fdir / f"learning_curves_t3_q{q}.png")
            save_figure(figure_curves(df, q, 2), fdir / f"learning_curves_t2_q{q}.png")
            save_figure(figure_visitation(arrays, q), fdir / f"visitation_q{q}.png")
        save_figure(figure_paired(df, crit), fdir / "paired_differences.png")
    rr = reading_rule(counts, crit)
    head = {"reading_rule": rr, "hypotheses": {h: bool(g["hypothesis_supported"].iloc[0]) for h, g in crit_df.groupby("hypothesis")},
            "budget_check_all_equal": bool(((bud["updates_equal"] == bud["n_runs"]) & (bud["rows_total_equal"] == bud["n_runs"])
                                            & (bud["rows_by_stage_equal"] == bud["n_runs"])
                                            & (bud["minibatch_steps_equal"] == bud["n_runs"])).all())}
    with open(out_dir / "headline.json", "w") as f:
        json.dump(head, f, indent=1)
    return head


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="T3-R1 analysis", allow_abbrev=False)
    ap.add_argument("--root", default=str(tc.DEFAULT_ROOT))
    ap.add_argument("--eval-dir", default=str(tc.DEFAULT_ROOT / "eval"))
    ap.add_argument("--out-dir", default=str(tc.DEFAULT_ROOT / "analysis"))
    ap.add_argument("--criteria", default=str(tm.CRITERIA_PATH))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    crit, sha = tm.load_criteria(Path(a.criteria))
    print(f"criteria {a.criteria} sha256 {sha}", flush=True)
    head = run_analysis(Path(a.root), Path(a.eval_dir), Path(a.out_dir), crit, not a.no_figures)
    print(json.dumps(head, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
