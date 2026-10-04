#!/usr/bin/env python3
"""R2b pre-registered analysis of the two pilot waves (reports 03, 04, 05; reuses the R1 tools).

Reads (read-only): the R2b wave runs ``<root>/waveA|waveP/q*/seed*/<arm>``, the R1 comparator runs
``<r1-root>/parents_A/q*/seed*`` (wave-A baseline, called ``A_base`` here) and
``<r1-root>/stage2/q*/seed*/{A_ctrl200,A_detmean}`` (wave-P comparators), and the rehearsal parent
candidates ``<ref-root>/q*/seed*/gates.json`` (``parent_u1600``). Every comparison root is an
explicit argument and is recorded in the output; a missing run is reported as such, never skipped.

The statistics are those of ``tools/v2/refine_analysis.py`` (pre-registration section 6 of R2b): final
tier for gate metrics, differences ``arm - comparator`` paired by (q, seed), 10,000 percentile
bootstrap resamples with ``numpy.random.default_rng(20261004)`` and one fresh generator per
(q, statistic) in table order, the primary metric ``stage2_peak_rel_err_abs`` (absolute signed
peak error, improvement = negative difference), criterion (a) and (b) reported separately.

Usage:
    python tools/v2/refine_r2b_analysis.py all --root results/v2_refine_r2b \
        --r1-root <v2-t2-refine>/results/v2_refine --ref-root <canonical>/results/v2_T2_locked/rehearsal_v1_1 \
        --out results/v2_refine_r2b/analysis --reports reports/v2/refine_r2b --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import refine_analysis as R  # noqa: E402  (reuses the R1 extraction, pairing, bootstrap, criterion)
import launch_refine as L  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R.BOOT_SEED = 20261004                        # D3 of the R2b prompt
QS: Tuple[int, ...] = (50, 60)
SEEDS: Tuple[int, ...] = tuple(range(10501, 10511))
ARMS_A: List[str] = list(L.R2B_WAVEA_ARMS)    # A_peak25, A_peak50, A_censored
ARMS_P: List[str] = list(L.R2B_WAVEP_ARMS)    # P20_lr3e-5, P20_lr3e-4, A_ctrl200_lr3e-4
BASE_A = "A_base"                             # R1 parents_A (the wave-A baseline)
R1_PAIR = ["A_ctrl200", "A_detmean"]          # R1 comparators of wave P
PARENT = "parent_u1600"
PATHWISE = ("A_detmean", "P20_lr3e-5", "P20_lr3e-4")
MATCHED_CONTROL = {"P20_lr3e-5": "A_ctrl200", "P20_lr3e-4": "A_ctrl200_lr3e-4"}
PRIMARY = R.PRIMARY["stage2"]
PEAK_HALF_WIDTH = 20.0
TAIL_PEAK, TAIL_ETA = 0.05, 0.004             # |peak error| <= 0.05 ; eta_2/DW margin of G-A

# comparisons of each wave: (arm, comparator, label); labels "vs baseline" and "ablation vs matched control"
# carry the pre-registered criterion, every other label is reported without a verdict
COMPS_A: List[Tuple[str, str, str]] = [(a, BASE_A, "vs baseline") for a in ARMS_A]
COMPS_P: List[Tuple[str, str, str]] = [
    ("P20_lr3e-5", "A_ctrl200", "ablation vs matched control"),
    ("P20_lr3e-4", "A_ctrl200_lr3e-4", "ablation vs matched control"),
    ("A_ctrl200_lr3e-4", PARENT, "vs baseline"),
    ("P20_lr3e-5", PARENT, "vs parent u1600 candidate"),
    ("P20_lr3e-4", PARENT, "vs parent u1600 candidate"),
    ("A_ctrl200", PARENT, "vs parent u1600 candidate"),
    ("A_detmean", PARENT, "vs parent u1600 candidate"),
    ("P20_lr3e-5", "A_detmean", "vs R1 A_detmean (one step per update)"),
    ("P20_lr3e-4", "A_detmean", "vs R1 A_detmean (one step per update)"),
    ("P20_lr3e-5", "A_ctrl200_lr3e-4", "vs PPO control at the other LR"),
    ("P20_lr3e-4", "A_ctrl200", "vs PPO control at the other LR"),
    ("A_ctrl200_lr3e-4", "A_ctrl200", "vs R1 A_ctrl200 (LR 3e-5)"),
]


# --------------------------------------------------------------------------- run directories
@dataclass(frozen=True)
class R2bCtx(R.Ctx):
    """Maps (arm) to the run directory of the R2b waves and of the R1 comparators."""

    r2b_root: str = ""
    r1_root: str = ""

    def run_dir(self, wave: str, q: int, seed: int, arm: str) -> Path:
        """Run directory of ``arm`` at (q, seed); ``wave`` is the R1 wave label and is ignored."""
        if arm == BASE_A:
            return Path(self.r1_root) / "parents_A" / f"q{q}" / f"seed{seed}"
        if arm in ARMS_A:
            return Path(self.r2b_root) / "waveA" / f"q{q}" / f"seed{seed}" / arm
        if arm in ARMS_P:
            return Path(self.r2b_root) / "waveP" / f"q{q}" / f"seed{seed}" / arm
        return Path(self.r1_root) / "stage2" / f"q{q}" / f"seed{seed}" / arm


def _phase_state(arm: str) -> Tuple[str, str]:
    """(phase letter, end-of-phase state file) of an arm of this round or of R1's method-5 pair."""
    return ("P", "state_end_P.pt") if arm in PATHWISE else ("A", "state_end_A.pt")


R.stage2_phase_state = _phase_state            # extract_stage2 looks the function up at call time


# --------------------------------------------------------------------------- per-run extras
def _read_json(p: Path) -> Any:
    with open(p) as f:
        return json.load(f)


def d1_extras(d: Path) -> Dict[str, Any]:
    """Raw-draw clamp counts of the learner at stage 2 (``d1_L_s2_*`` of ``v2_updates.csv``)."""
    up = pd.read_csv(d / "v2_updates.csv")
    out: Dict[str, Any] = {}
    if "d1_L_s2_lo" not in up:
        return out
    n = (up["d1_L_s2_lo"] + up["d1_L_s2_hi"]).astype(int)
    out["d1_flagged_L_s2"] = int(n.sum())
    out["d1_first_flagged_update"] = int(up.loc[n > 0, "update"].iloc[0]) if (n > 0).any() else None
    out["d1_L_s2_rows"] = int(up["d1_L_s2_n"].sum())
    for tag in ("in", "out"):
        if f"d1_L_s2_{tag}_lo" in up:
            out[f"d1_flagged_L_s2_{tag}"] = int((up[f"d1_L_s2_{tag}_lo"] + up[f"d1_L_s2_{tag}_hi"]).sum())
    if "n_censored_rows" in up:
        c = up["n_censored_rows"].fillna(0).astype(int)
        out["censored_rows_total"] = int(c.sum())
        out["censored_first_update"] = int(up.loc[c > 0, "update"].iloc[0]) if (c > 0).any() else None
    return out


def visitation_extras(d: Path, spec: Any, bin_width: float) -> Dict[str, Any]:
    """Share of the learner's stage-2 starts in the peak set (cumulative over the phase, last verifier call)."""
    th = _read_json(d / "train_history.json")
    last = None
    for c in th.get("verifier_calls", []):
        vis = (c.get("visitation_cumulative_phase") or {}).get("stage2_direct_es")
        if vis is not None:
            last = np.asarray(vis, dtype=float)
    if last is None:
        return {}
    mask = R.StartSampler(spec, bin_width).peak_set(spec.T, PEAK_HALF_WIDTH)
    if mask.size != last.size:
        return {"peak_visit_note": f"bin count {last.size} != peak mask {mask.size}"}
    return {"peak_visit_share": float(last[mask].sum() / last.sum()), "peak_visit_rows": int(last.sum()),
            "peak_visit_design_share": float(mask.sum() / mask.size)}


def tail_extras(d: Path, spec: Any, q: int) -> Dict[str, Any]:
    """Stage-2 profile on |d| >= 2q (recovery grid) and the off-path Delta_2 (verifier grid, final tier)."""
    out: Dict[str, Any] = {}
    with np.load(d / "final_final.npz") as z:
        dg, e2, g2 = z["recovery_d_grid"], z["recovery_e2"], z["recovery_g2"]
        g20 = float(g2[int(np.nonzero(dg == 0.0)[0][0])])
        m = np.abs(dg) >= 2 * q
        out["tail2q_mean_e2hat"] = float(e2[m].mean())
        out["tail2q_mean_abs_err_over_g2_0"] = float(np.abs(e2[m] - g2[m]).mean() / g20)
        out["tail2q_max_abs_err_over_g2_0"] = float(np.abs(e2[m] - g2[m]).max() / g20)
        dw = float(spec.w_h - spec.w_l)
        on = np.asarray(z["v_t2_onpath"]).astype(bool)
        delta = np.asarray(z["v_t2_delta"], dtype=float)
        if (~on).any():
            out["offpath_delta2_max_over_dw"] = float(np.max(delta[~on]) / dw)
            out["offpath_delta2_mean_over_dw"] = float(np.mean(delta[~on]) / dw)
        if on.any():
            out["onpath_delta2_max_over_dw"] = float(np.max(delta[on]) / dw)
    return out


def pathwise_extras(d: Path, anomalies: List[str]) -> Dict[str, Any]:
    """Phase-P loss / FOC trajectory summary and step counts (``v2_updates.csv``; the P20 arms also log n_steps)."""
    out: Dict[str, Any] = {}
    lg = R.detmean_logged(d)
    w = R.LOGGED_WINDOW
    out["loss_first20_mean"] = float(lg["loss"].iloc[:w].mean())
    out["loss_last20_mean"] = float(lg["loss"].iloc[-w:].mean())
    out["foc_logged_mean_last20"] = float(lg["foc_abs_mean"].iloc[-w:].mean())
    out["foc_logged_max_last20"] = float(lg["foc_abs_max"].iloc[-w:].max())
    out["gn_pre_clip_last20_mean"] = float(lg["grad_norm_pre_clip"].iloc[-w:].mean())
    up = pd.read_csv(d / "v2_updates.csv")
    out["n_steps_total"] = int(up["n_steps"].sum()) if "n_steps" in up else int(len(up))
    if "grad_norm_pre_clip_max" in up:
        out["gn_pre_clip_max"] = float(up["grad_norm_pre_clip_max"].max())
    pc = _read_json(d / "phaseP_checks.json")
    out["phaseP_checks_all_true"] = bool(all(pc.values()))
    if not all(pc.values()):
        anomalies.append(f"phaseP_checks.json has a false entry: {pc}")
    return out


def run_extras(ctx: R2bCtx, row: Dict[str, Any]) -> Dict[str, Any]:
    """Extra columns of one complete run (empty for a run that is not complete)."""
    if not row.get("complete") or R._truthy(row.get("reference")):      # the reference rows carry NaN, not False
        return {}
    arm, q, seed = row["arm"], int(row["q"]), int(row["seed"])
    d = ctx.run_dir("stage2", q, seed, arm)
    spec = R.spec_for_q(ctx, q)
    bw = float(R._protocol(ctx.protocol)["records"][str(q)]["protocol"]["es_bin_width"])
    anomalies: List[str] = []
    out: Dict[str, Any] = {}
    for fn, args in ((d1_extras, (d,)), (visitation_extras, (d, spec, bw)), (tail_extras, (d, spec, q))):
        try:
            out.update(fn(*args))
        except Exception as exc:  # noqa: BLE001 - reported in the table
            anomalies.append(f"{fn.__name__} unavailable: {type(exc).__name__}: {exc}")
    if arm in PATHWISE:
        try:
            out.update(pathwise_extras(d, anomalies))
        except Exception as exc:  # noqa: BLE001
            anomalies.append(f"pathwise_extras unavailable: {type(exc).__name__}: {exc}")
    if arm in ARMS_P:                                         # offline fixed-grid objective at the parent and exports
        try:
            man = _read_json(d / "manifest.json")
            traj = R.pathwise_trajectory(ctx, q, seed, arm, spec, d, man.get("parent_checkpoint"))
            out["J_offline_start"], out["J_offline_end"] = traj[0]["J"], traj[-1]["J"]
            out["foc_offline_end_mean"], out["foc_offline_end_max"] = traj[-1]["foc_mean"], traj[-1]["foc_max"]
            out["_traj"] = traj
        except Exception as exc:  # noqa: BLE001
            anomalies.append(f"offline objective unavailable: {type(exc).__name__}: {exc}")
    if arm in ARMS_P:                                         # RNG divergence against the matched PPO control
        ref = {"P20_lr3e-5": "A_ctrl200", "P20_lr3e-4": "A_ctrl200_lr3e-4", "A_ctrl200_lr3e-4": "A_ctrl200"}[arm]
        out.update(R.rng_columns(d, ctx.run_dir("stage2", q, seed, ref), False))
    if anomalies:
        out["anomalies_r2b"] = "; ".join(anomalies)
    return out


# --------------------------------------------------------------------------- extraction and tables
def extract(ctx: R2bCtx, qs: Sequence[int], seeds: Sequence[int], workers: int
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-run table of every arm of both waves and the R1 comparators, plus the offline trajectories."""
    arms = [BASE_A] + ARMS_A + R1_PAIR + ARMS_P
    df, traj1 = R.extract_wave(ctx, "stage2", qs, seeds, arms, workers=workers)
    extra_rows, traj = [], list(traj1.to_dict("records")) if len(traj1) else []
    for row in df.to_dict("records"):
        ex = run_extras(ctx, row)
        t = ex.pop("_traj", [])
        if row["arm"] in ARMS_P:                              # R1's pair brings its own trajectories
            traj += t
        extra_rows.append(ex)
    df = df.reset_index(drop=True)
    ex = pd.DataFrame(extra_rows, index=df.index)
    for c in ex.columns:                                      # an extra overrides the R1 column of the same name
        if c in df.columns:
            df[c] = df[c].astype(object)
            m = ex[c].notna()
            df.loc[m, c] = ex.loc[m, c].astype(object)
            try:                                              # numeric columns stay numeric (rng_div_* mix strings and ints)
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
        else:
            df[c] = ex[c]
    df = R.prepare(df)
    df["tail_peak_ok"] = df[PRIMARY].astype(float) <= TAIL_PEAK
    df["tail_eta_over_margin"] = df["eta_T_over_dw"].astype(float) > TAIL_ETA
    return df, pd.DataFrame(traj)


def tail_table(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int]) -> pd.DataFrame:
    """Per arm and q: max |peak error|, runs with |peak| <= 0.05, runs with eta_2/DW > 0.004, G-A passes."""
    rows = []
    for arm in arms:
        for q in qs:
            g = df[(df.arm == arm) & (df.q == q) & df.complete]
            peak = R._num(g[PRIMARY])
            rows.append({"arm": arm, "q": q, "n_complete": int(len(g)),
                         "max_abs_peak_error": float(peak.max()) if len(g) else float("nan"),
                         "n_abs_peak_le_0.05": int((peak <= TAIL_PEAK).sum()),
                         "n_eta_over_0.004": int((R._num(g["eta_T_over_dw"]) > TAIL_ETA).sum()),
                         "n_gate_pass": int(sum(1 for v in g["gate_pass"] if R._truthy(v))),
                         "mean_abs_peak_error": float(peak.mean()) if len(g) else float("nan"),
                         "max_tail_mean_over_g2_0": float(R._num(g["stage2_tail_mean_over_g2_0"]).max()) if len(g)
                         else float("nan"),
                         "n_tail_mean_over_0.02": int((R._num(g["stage2_tail_mean_over_g2_0"]) > 0.02).sum())})
    return pd.DataFrame(rows)


def seed_table(df: pd.DataFrame, comps: Sequence[Tuple[str, str, str]], qs: Sequence[int],
               seeds: Sequence[int]) -> pd.DataFrame:
    """Seed-level primary values (arm, comparator, difference) for every comparison, with run flags."""
    rows = []
    for arm, base, label in comps:
        for q in qs:
            sv = R.paired_seed_values(df, arm, base, q, PRIMARY, seeds)
            for r in sv.itertuples(index=False):
                a = df[(df.arm == arm) & (df.q == q) & (df.seed == r.seed)].iloc[0]
                row = {"comparison": label, "arm": arm, "comparator": base, "q": q, "seed": r.seed,
                       "arm_abs_peak": r.arm_value, "comparator_abs_peak": r.base_value, "diff": r.diff,
                       "improved": r.diff < 0, "tied": r.diff == 0}
                for c in ("d1_flagged_L_s2", "d1_first_flagged_update", "censored_rows_total",
                          "censored_first_update"):
                    if c in a.index:
                        row[c] = a[c]
                rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class Result:
    """All tables of the analysis."""

    df: pd.DataFrame
    traj: pd.DataFrame
    tabs: Dict[str, pd.DataFrame]
    comps_all: List[Tuple[str, str, str]]


def compute(args: argparse.Namespace) -> Result:
    """Extract every planned run and compute the pre-registered tables (CSVs written under ``args.out``)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qs, seeds = tuple(args.qs), tuple(args.seeds)
    ctx = R2bCtx(root=str(args.root), ref_root=str(args.ref_root), r2b_root=str(args.root),
                 r1_root=str(args.r1_root))
    df, traj = extract(ctx, qs, seeds, args.workers)
    comps = COMPS_A + COMPS_P
    paired, seed_level = R.paired_tables(df, "stage2", comps, qs, seeds)
    arms_all = [BASE_A] + ARMS_A + R1_PAIR + ARMS_P
    tabs: Dict[str, pd.DataFrame] = {
        "per_run": df.drop(columns=[c for c in df.columns if c == "_traj"], errors="ignore"),
        "paired": paired, "paired_seed_level": seed_level, "seed_table": seed_table(df, comps, qs, seeds),
        "criterion": R.criterion_table(df, "stage2", paired, comps, qs, seeds),
        "tail": tail_table(df, arms_all + [PARENT], qs),
        "arm_summary": R.arm_summary(df, "stage2", arms_all + [PARENT], qs),
        "gate_counts": R.gate_counts_table(df, "stage2", arms_all, qs),
        "optimisation": R.opt_table(df, "stage2", arms_all, qs),
        "rng_divergence_A": R.rng_table(df, "stage2", ARMS_A + [BASE_A], BASE_A, qs),
        "cost_A": R.cost_table(df, "stage2", [BASE_A] + ARMS_A, BASE_A),
        "cost_P": R.cost_table(df, "stage2", R1_PAIR + ARMS_P, "A_ctrl200"),
        "dispersion_A": R.dispersion_table(df, "stage2", [BASE_A] + ARMS_A, BASE_A, qs, seeds),
        "dispersion_P": R.dispersion_table(df, "stage2", ARMS_P, "A_ctrl200", qs, seeds,
                                           extra_pairs=[(a, b) for a, b, _ in COMPS_P[:2]]),
        "completeness": completeness(df, arms_all, qs, seeds, Path(args.root)),
        "manifest_commits": R.manifest_commit_table(df),
        "trajectory_per_run": traj,
    }
    cp = tabs["cost_P"]
    same_lr = {"P20_lr3e-5": "A_ctrl200", "P20_lr3e-4": "A_ctrl200_lr3e-4", "A_detmean": "A_ctrl200"}
    wall = dict(zip(cp["arm"], cp["mean_phase_wall_sec"]))
    cp["wall_ratio_vs_same_lr_ppo_control"] = [
        (wall[a] / wall[same_lr[a]] if a in same_lr and wall.get(same_lr[a]) else float("nan")) for a in cp["arm"]]
    tabs["locfree_argmax_per_run"], tabs["locfree_argmax_summary"] = locfree_tables(df, arms_all + [PARENT], qs)
    for name, t in tabs.items():
        t.to_csv(out / f"{name}.csv", index=False)
    return Result(df, traj, tabs, comps)


def locfree_tables(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per run and per (arm, q): the location-free peak error and the d at which the stage-2 mean is maximal."""
    g = df[df.arm.isin(arms) & df.complete]
    per = g[["arm", "q", "seed", "stage2_peak_locfree_rel_err", "stage2_peak_locfree_argmax_d"]].copy()
    rows = []
    for arm in arms:
        for q in qs:
            x = R._num(g[(g.arm == arm) & (g.q == q)]["stage2_peak_locfree_argmax_d"])
            rows.append({"arm": arm, "q": q, "n": int(x.notna().sum()),
                         "median_abs_argmax_d": float(x.abs().median()) if x.notna().any() else float("nan"),
                         "min_argmax_d": float(x.min()) if x.notna().any() else float("nan"),
                         "max_argmax_d": float(x.max()) if x.notna().any() else float("nan"),
                         "n_argmax_at_0": int((x == 0).sum())})
    return per, pd.DataFrame(rows)


def completeness(df: pd.DataFrame, arms: Sequence[str], qs: Sequence[int], seeds: Sequence[int],
                 root: Path) -> pd.DataFrame:
    """Runs done / failed / incomplete / missing per arm and q (R1's table for wave A and wave P together)."""
    rows = []
    for arm in arms:
        sub = "waveA" if arm in ARMS_A else ("waveP" if arm in ARMS_P else "R1")
        t = R.completeness_table(df, [arm], qs, seeds, root, sub if sub != "R1" else "stage2")
        t.insert(1, "source", sub)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- reports
SECONDARY = [("stage2_peak_rel_err_signed", "signed peak error"),
             ("stage2_peak_locfree_rel_err_abs", "|location-free peak error|"),
             ("stage2_rmse_pos_over_g2_0", "RMSE_pos / e2*(0)"), ("stage2_tail_mean_over_g2_0", "tail mean / e2*(0)"),
             ("stage2_tail_max_over_g2_0", "tail max / e2*(0)"), ("eta_T_over_dw", "eta_2 / DW (final)"),
             ("eta_dev_minus_final_abs", "|dev - final| of eta_2 (G-N part)"),
             ("stage2_sym_err_max", "symmetry error"), ("sigma_effort_at_0_t2", "sigma_2(0)"),
             ("smoothed_share_peak_gap_d0", "smoothed-game share of the d = 0 gap")]


def comp_rows(paired: pd.DataFrame, comps: Sequence[Tuple[str, str, str]], metric: str) -> pd.DataFrame:
    """Display rows of one metric: one row per (comparison, q) with the paired statistics and the CIs."""
    rows = []
    for arm, base, label in comps:
        for q in QS:
            r = paired[(paired.arm == arm) & (paired.baseline == base) & (paired.q == q)
                       & (paired.metric == metric) & (paired.comparison == label)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"comparison": label, "arm": arm, "comparator": base, "q": q, "n": int(r["n_pairs"]),
                         "mean diff": r["mean"], "95% CI mean": R.ci_str(r["ci_mean_lo"], r["ci_mean_hi"]),
                         "median diff": r["median"], "95% CI median": R.ci_str(r["ci_median_lo"], r["ci_median_hi"]),
                         "n_better": _int(r["n_better"]), "n_zero": _int(r["n_zero"])})
    return pd.DataFrame(rows)


def _int(v: Any) -> Any:
    """Integer count from a table cell that pandas may have turned into a float (None for an undirected metric)."""
    return int(v) if v is not None and pd.notna(v) else None


def crit_display(crit: pd.DataFrame) -> pd.DataFrame:
    """Criterion table, parts (a) and (b) separately."""
    cols = ["arm", "baseline", "comparison"]
    for q in QS:
        cols += [f"mean_q{q}", f"ci_mean_lo_q{q}", f"ci_mean_hi_q{q}", f"a_q{q}"]
    cols += ["a_met", "n_base_pass", "b_status", "b_violations", "b_pending", "overall"]
    return crit[[c for c in cols if c in crit.columns]].rename(columns={"baseline": "comparator"})


def wave_header(doc: R.Doc, title: str, intro: str, args: argparse.Namespace, sub: str) -> None:
    """Title, scope paragraph and the roots of the analysis."""
    doc.h(1, title)
    doc.p(intro)
    doc.p(f"Generated by `tools/v2/refine_r2b_analysis.py {sub}`; deterministic given the same inputs. Every number is read "
          "from a CSV that is cited under its table. Roots (explicit arguments, recorded here): R2b runs "
          f"`{R._rel(args.root)}`; R1 comparator runs `{args.r1_root}` (`parents_A` = the wave-A baseline `A_base`, "
          f"`stage2/<arm>` = `A_ctrl200`, `A_detmean`); rehearsal parent candidates `{args.ref_root}` (`{PARENT}`).")


def provenance_block(doc: R.Doc, res: Result, args: argparse.Namespace, arms: Sequence[str], source: str) -> None:
    """Completeness, manifest commits and the launch-check summary."""
    doc.h(3, "Provenance and completeness")
    comp = res.tabs["completeness"]
    comp = comp[comp.arm.isin(arms)]
    doc.table(comp, f"{source}_completeness",
              disp=comp[["arm", "source", "q", "planned", "done", "failed", "incomplete", "missing", "not_done_seeds"]])
    doc.p("Commit and dirty flag recorded in `manifest.json` of the analysed runs (all arms, both waves and the R1 "
          "comparators):")
    doc.table(res.tabs["manifest_commits"], f"{source}_manifest_commits")
    lc = Path(args.root) / "launch_checks.json"
    if lc.exists():
        j = _read_json(lc)
        doc.p(f"Launch checks (`tools/v2/r2b_launch_checks.py`, `{R._rel(lc)}`): {j['n_ok']} of {j['n']} runs pass every "
              "check (manifest dirty flag false and commit within results/ and reports/ of the code commit; config "
              "differs from the on-disk R1 comparator config in exactly the pre-registered keys; the manifest's four R2b "
              f"keys equal the arm table; wave P: parent SHA-256 equals `parents.csv`); ALL = {j['ALL']}.")
    else:
        doc.p("Launch checks: `launch_checks.json` is not present.")


def tail_block(doc: R.Doc, res: Result, arms: Sequence[str], name: str) -> None:
    """The tail statistics table."""
    doc.h(3, "Tail statistics (next to the criterion)")
    t = res.tabs["tail"]
    doc.p("Per arm and q over the complete runs: the maximum of |signed peak error| over the 10 seeds, the number of "
          f"runs with |peak error| <= {TAIL_PEAK}, the number with eta_2/DW > {TAIL_ETA} (the G-A margin; the G-A limit is "
          "0.005) and the number that pass G-A with its G-N part.")
    doc.table(t[t.arm.isin(arms)], name)


def optimisation_block(doc: R.Doc, res: Result, arms: Sequence[str], cost_key: str, name: str) -> None:
    """Optimisation diagnostics and cost tables."""
    doc.h(3, "Optimisation diagnostics and cost per run")
    o = res.tabs["optimisation"]
    doc.p("Median over the complete runs of the per-run diagnostics (KL and clip fraction exist for PPO runs only; "
          "`n_minibatch_steps_total` is the number of optimiser steps of the phase. The column `n_actor_steps_total` of the R1 "
          "extraction is left out: it counts only updates that took the masked update path, which is not every update of "
          "a PPO run with flagged rows).")
    doc.table(o[o.arm.isin(arms)].drop(columns=["n_actor_steps_total", "median_n_actor_steps_total"], errors="ignore"),
              f"{name}_optimisation")
    doc.table(res.tabs[cost_key].drop(columns=["mean_n_actor_steps_total"], errors="ignore").dropna(axis=1, how="all"),
              f"{name}_cost",
              note="mean over the complete runs of both q; wall time is machine-load dependent (wave A and wave P ran together, "
                   "up to 40 processes, load average 6 to 43; the R1 runs on 2026-10-03)")


CRIT: Dict[str, Tuple[str, str]] = {a: (b, lab) for a, b, lab in COMPS_A + COMPS_P if lab in R.CRITERION_LABELS}


def method_block(doc: R.Doc) -> None:
    """Statistics used by every table of the report (pre-registration section 6)."""
    doc.p("Statistics (`01_preregistration.md` section 6): differences are arm - comparator, paired by (q, seed); `n_better` "
          "counts strictly improving seeds and `n_zero` exact ties; the 95% CIs are percentile bootstrap intervals of the mean "
          "and of the median of the 10 paired differences (10,000 resamples with replacement, "
          "`numpy.random.default_rng(20261004)`, one fresh generator per (q, statistic) in table order, so all intervals of the "
          "same sample size use the same resample indices); the primary metric is the absolute signed peak error at d = 0 "
          "(final tier); a run is complete iff `status.json` says done with exit code 0 and `final_v2.json` holds a "
          "final-tier evaluation.")


def arm_definition(arm: str) -> str:
    """Definition of an arm from the launcher tables (``tools/v2/launch_refine.py``)."""
    t = L.R2B_WAVEA_ARMS.get(arm) or L.R2B_WAVEP_ARMS.get(arm) or {}
    parts = [str(t.get("definition", "")).rstrip(".") + "."]
    if t.get("r2b"):
        parts.append("Non-default R2b keys: " + ", ".join(f"`{k}` = `{json.dumps(v)}`" for k, v in t["r2b"].items()) + ".")
    if t.get("lr"):
        parts.append(f"LR window(s) (start, end, local first, local last): `{t['lr']}`.")
    return " ".join(p for p in parts if p)


def other_metrics_table(paired: pd.DataFrame, arm: str, base: str, label: str) -> pd.DataFrame:
    """Compact table: every non-primary metric, mean paired difference [95% CI of the mean] and n_better per q."""
    rows = []
    for m, lab in SECONDARY:
        row: Dict[str, Any] = {"metric": lab}
        for q in QS:
            r = paired[(paired.arm == arm) & (paired.baseline == base) & (paired.q == q) & (paired.metric == m)
                       & (paired.comparison == label)]
            if r.empty:
                continue
            r = r.iloc[0]
            nb = _int(r["n_better"])
            row[f"q{q} mean diff [95% CI]"] = f"{R._g(r['mean'])} {R.ci_str(r['ci_mean_lo'], r['ci_mean_hi'])}" + \
                (f", {nb}/{int(r['n_pairs'])} better" if nb is not None else "")
        rows.append(row)
    return pd.DataFrame(rows)


def arm_notable(df: pd.DataFrame, arm: str, base: str) -> List[str]:
    """Computed notable events of one arm: gate failures with their components, positive peak errors, ties, no-flag runs."""
    out: List[str] = []
    g = df[(df.arm == arm) & df.complete]
    for r in g.itertuples():
        if not R._truthy(r.gate_pass):
            out.append(f"q{int(r.q)}/{int(r.seed)} fails G-A or its G-N part: {failing_components(df, arm, int(r.q), int(r.seed))}")
    for r in g.itertuples():
        if float(r.stage2_peak_rel_err_signed) > 0:
            out.append(f"q{int(r.q)}/{int(r.seed)} has a POSITIVE signed peak error ({R._g(float(r.stage2_peak_rel_err_signed))}): "
                       "for this run the difference of |peak error| is not the negative of the difference of the signed error")
    for r in g.itertuples():
        share, gap = getattr(r, "smoothed_share_peak_gap_d0", float("nan")), getattr(r, "observed_gap_d0", float("nan"))
        if pd.notna(share) and pd.notna(gap) and (float(gap) <= 0 or abs(float(share)) > 5):
            out.append(f"q{int(r.q)}/{int(r.seed)}: the smoothed-game share ({R._g(float(share))}) is not meaningful, the "
                       f"observed d = 0 gap is {R._g(float(gap))} effort units (a near-zero or negative denominator drives the "
                       "mean and CI of that metric)")
    b = df[(df.arm == base) & df.complete].set_index(["q", "seed"])
    ties = [f"q{int(r.q)}/{int(r.seed)}" for r in g.itertuples()
            if (int(r.q), int(r.seed)) in b.index and float(b.loc[(int(r.q), int(r.seed)), PRIMARY]) == float(getattr(r, PRIMARY))]
    if ties:
        out.append("exact ties with the comparator on the primary metric: " + ", ".join(ties))
    if arm == "A_censored" and "censored_rows_total" in g:
        nf = [f"q{int(r.q)}/{int(r.seed)}" for r in g.itertuples() if not (float(r.censored_rows_total or 0) > 0)]
        if nf:
            out.append("runs that never had a flagged row (identical to the baseline by construction): " + ", ".join(nf))
    return out


def arm_section(doc: R.Doc, res: Result, arm: str, comps: Sequence[Tuple[str, str, str]], wave: str,
                fig_items: Sequence[Tuple[str, Sequence[str]]], fig_dir: Path, rep_dir: Path) -> None:
    """One section of a pilot report: definition, paired tables, criterion, tails, other metrics, specifics, figure, anomalies."""
    paired, crit, df = res.tabs["paired"], res.tabs["criterion"], res.df
    base, label = CRIT[arm]
    doc.h(3, f"`{arm}`")
    doc.p(arm_definition(arm) + f" Comparator of the pre-registered criterion: `{base}` (label `{label}`).")
    doc.p("Primary metric, paired differences arm - comparator (negative = better):")
    doc.table(comp_rows(paired, [c for c in comps if c[0] == arm], PRIMARY), f"{wave}_{arm}_primary")
    doc.p("Criterion, parts (a) and (b) separately:")
    doc.table(crit_display(crit[crit.arm == arm]), f"{wave}_{arm}_criterion")
    t = res.tabs["tail"]
    doc.p("Tail statistics of the arm and of its criterion comparator:")
    doc.table(t[t.arm.isin([arm, base])], f"{wave}_{arm}_tail")
    doc.p("Other metrics against the criterion comparator (mean difference arm - comparator with the 95% CI of the mean; "
          "`k/10 better` counts seeds with a smaller value, an improvement for the error, tail and gate metrics; the signed "
          "peak error, sigma_2(0) and the smoothed-game share have no direction and show no count; because the signed peak "
          "errors are negative, a POSITIVE difference of the signed error is an improvement. The smoothed-game share is a "
          "ratio whose denominator is the observed d = 0 gap, so a run with a near-zero or negative gap, listed under the "
          "notable events if there is one, dominates its mean and CI):")
    doc.table(other_metrics_table(paired, arm, base, label), f"{wave}_{arm}_metrics")
    o = res.tabs["optimisation"]
    doc.p("Optimisation diagnostics (median over the runs):")
    doc.table(o[o.arm == arm].drop(columns=["n_actor_steps_total", "median_n_actor_steps_total"], errors="ignore"),
              f"{wave}_{arm}_optimisation")
    if wave == "waveA":
        cols = ["arm", "q", "peak_visit_share", "peak_visit_design_share", "d1_flagged_L_s2", "d1_flagged_L_s2_in",
                "censored_rows_total", "tail2q_mean_e2hat", "tail2q_mean_abs_err_over_g2_0", "offpath_delta2_max_over_dw",
                "onpath_delta2_max_over_dw"]
        sp = df[df.arm.isin([BASE_A, arm]) & df.complete][[c for c in cols + ["seed"] if c in df.columns]]
        gm = (sp.groupby(["arm", "q"]).mean(numeric_only=True).reset_index().drop(columns=["seed"], errors="ignore")
              .dropna(axis=1, how="all"))
        doc.p("Start sampling, clamp counts and stage-2 profile (mean over the 10 seeds; baseline for comparison):")
        doc.table(gm, f"{wave}_{arm}_specifics")
    elif arm in PATHWISE:
        cols = ["arm", "q", "seed", "n_steps_total", "loss_first20_mean", "loss_last20_mean", "foc_logged_mean_last20",
                "gn_pre_clip_last20_mean", "J_offline_start", "J_offline_end", "foc_offline_end_mean", "foc_offline_end_max"]
        pw = df[(df.arm == arm) & df.complete][[c for c in cols if c in df.columns]]
        gm = pw.groupby(["arm", "q"]).mean(numeric_only=True).reset_index().drop(columns=["seed"], errors="ignore")
        doc.p("Pathwise loss, FOC residual, steps and the offline objective (mean over the 10 seeds):")
        doc.table(gm, f"{wave}_{arm}_pathwise")
    if fig_items:
        doc.figures(list(fig_items), fig_dir, rep_dir)
    notable = arm_notable(df, arm, base)
    doc.p("Notable events of this arm (computed): " + ("; ".join(notable) if notable else "none."))


def arm_index(res: Result, arms: Sequence[str]) -> pd.DataFrame:
    """One row per arm of a wave: comparator, primary difference per q with CI, criterion (a) / (b), tails."""
    paired, crit, tail = res.tabs["paired"], res.tabs["criterion"], res.tabs["tail"]
    rows = []
    for arm in arms:
        base, label = CRIT[arm]
        row: Dict[str, Any] = {"arm": arm, "comparator": base}
        for q in QS:
            r = paired[(paired.arm == arm) & (paired.baseline == base) & (paired.q == q) & (paired.metric == PRIMARY)
                       & (paired.comparison == label)]
            if len(r):
                r = r.iloc[0]
                row[f"q{q} mean diff [95% CI]"] = f"{R._g(r['mean'])} {R.ci_str(r['ci_mean_lo'], r['ci_mean_hi'])}"
                row[f"q{q} better/ties of {int(r['n_pairs'])}"] = f"{_int(r['n_better'])}/{_int(r['n_zero'])}"
        c = crit[(crit.arm == arm) & (crit.baseline == base)]
        if len(c):
            c = c.iloc[0]
            row["(a) both q"] = bool(c["a_met"])
            row["(b)"] = c["b_status"] + (f" ({violation_text(res.df, arm, c['b_violations'])})" if c["b_violations"] else "")
        for q in QS:
            t = tail[(tail.arm == arm) & (tail.q == q)]
            if len(t):
                t = t.iloc[0]
                row[f"q{q} max|peak|, n<=0.05, n eta>0.004"] = (
                    f"{R._g(t['max_abs_peak_error'])}, {t['n_abs_peak_le_0.05']}, {t['n_eta_over_0.004']}")
        rows.append(row)
    return pd.DataFrame(rows)


def anomalies_block(doc: R.Doc, res: Result, arms: Sequence[str]) -> None:
    """Anomalies of the wave: the tool's own column plus computed notable events."""
    df = res.df
    sub = df[df.arm.isin(list(arms) + [PARENT])]
    n_an = int(sub["anomalies"].fillna("").astype(str).str.len().gt(0).sum()) if "anomalies" in sub else 0
    n_an2 = int(sub["anomalies_r2b"].fillna("").astype(str).str.len().gt(0).sum()) if "anomalies_r2b" in sub else 0
    n_done = int((sub["status"] == "done").sum())
    n_ref = int((sub["status"] == "reference").sum())
    doc.p(f"The extraction recorded anomalies in {n_an} (`anomalies` column of `per_run.csv`) and {n_an2} (`anomalies_r2b`, "
          f"a column that exists only when it is non-empty) of the {len(sub)} rows of the arms of this report plus the "
          f"`{PARENT}` rows ({n_done} runs with status done, every one exited 0, and {n_ref} `{PARENT}` reference rows read "
          "from the rehearsal `gates.json`). Notable events computed from the tables, per arm:")
    for arm in arms:
        if arm == BASE_A:
            doc.lines.append("- `A_base`: the R1 `parents_A` baseline; its end-of-A state equals the rehearsal's in 20/20 "
                             "(`parents_A_checks.json`, recorded in `parents.csv`).")
            continue
        base = CRIT[arm][0] if arm in CRIT else ("A_ctrl200" if arm == "A_detmean" else PARENT)
        ev = arm_notable(df, arm, base)
        doc.lines.append(f"- `{arm}`: " + ("; ".join(ev) if ev else "none."))
    doc.lines.append("")


def commands_block(doc: R.Doc, args: argparse.Namespace, wave: str) -> None:
    """Launch records (argv, workers, nproc, load, disk), the check command and the analysis command."""
    sub = "waveA" if wave == "waveA" else "waveP"
    recs = sorted((Path(args.root) / sub).glob("launch_*.json"))
    rows = []
    for p in recs:
        j = _read_json(p)
        runs = j.get("runs", [])
        rows.append({"record": R._rel(p), "argv": " ".join(j.get("argv", [])), "workers": j.get("workers"),
                     "nproc": j.get("nproc"), "loadavg at start": str([round(x, 2) for x in j.get("loadavg_at_start", [])]),
                     "loadavg at end": str([round(x, 2) for x in j.get("loadavg_at_end", [])]),
                     "free disk (TB)": round(j.get("disk_free_bytes", 0) / 1e12, 2), "head": str(j.get("head", ""))[:7],
                     "code commit": str(j.get("code_commit_resolved", ""))[:7], "state": j.get("state"),
                     "runs finished / return code 0": f"{len(runs)} / {sum(1 for r in runs if r.get('returncode') == 0)}"})
    doc.p("Launch record of this wave (`tools/v2/launch_refine.py`; the record stores `nproc`, load average and free disk before "
          "the launch and `git diff --stat <code commit> HEAD`; the dry-run records `dryrun_*.json` in the same directory "
          "validated every config before the launch):")
    doc.table(pd.DataFrame(rows), f"{wave}_launch_record")
    lc = Path(args.root) / "launch_checks.json"
    if lc.exists():
        j = _read_json(lc)
        doc.p("Post-launch checks: `python tools/v2/r2b_launch_checks.py --code-commit " + str(j["code_commit"]) +
              " --r1-root " + str(j["r1_root"]) + " --out " + R._rel(lc) + "`.")
    doc.p("Analysis: `python tools/v2/refine_r2b_analysis.py all --root results/v2_refine_r2b --r1-root " + args.r1_root +
          " --ref-root " + args.ref_root + " --out results/v2_refine_r2b/analysis --reports reports/v2/refine_r2b --workers 8` "
          "(single-threaded workers; `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`).")


def locfree_block(doc: R.Doc, res: Result, arms: Sequence[str], name: str) -> None:
    """The location-free peak error and its argmax d per arm and q."""
    t = res.tabs["locfree_argmax_summary"]
    doc.p("Location-free peak error (the maximum of the stage-2 mean over the recovery grid relative to e2*(0)) and where "
          "that maximum lies: per arm and q the median |argmax d|, its range and the number of runs whose maximum is exactly at "
          "d = 0 (the per-run values are in `locfree_argmax_per_run.csv`):")
    doc.table(t[t.arm.isin(arms)], name)


def fig_wavep_offline(fig_dir: Path, traj: pd.DataFrame) -> List[str]:
    """Offline fixed-grid objective J and FOC residual at the weight exports (mean over seeds), per arm and q."""
    plt = R._mpl()
    arms = ["A_ctrl200", "A_ctrl200_lr3e-4", "A_detmean", "P20_lr3e-5", "P20_lr3e-4"]
    fig, axes = plt.subplots(2, len(QS), figsize=(4.4 * len(QS), 6.0), squeeze=False)
    for j, q in enumerate(QS):
        for i, arm in enumerate(arms):
            g = traj[(traj.arm == arm) & (traj.q == q)].groupby("update")[["J", "foc_mean"]].mean().reset_index()
            if g.empty:
                continue
            kw = dict(marker=R.ARM_STYLE["marker"][i], color=R.ARM_STYLE["color"][i], ms=4, label=arm)
            axes[0][j].plot(g["update"], g["J"], **kw)
            axes[1][j].plot(g["update"], g["foc_mean"], **kw)
        axes[0][j].set_title(f"q = {q}: offline J (descriptive level)")
        axes[1][j].set_title(f"q = {q}: offline FOC residual (mean |dR/de|; lower is better)")
        axes[1][j].set_xlabel("global update (1600 = the parent state)")
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("Wave P: fixed-grid objective and FOC residual at the weight exports (mean over 10 seeds)", fontsize=9)
    fig.tight_layout()
    return R.save_fig(fig, fig_dir / "waveP_offline_trajectory")


def logged_trajectory(args: argparse.Namespace) -> pd.DataFrame:
    """Rolling-mean (20 updates) logged loss and FOC residual of the pathwise arms, mean over seeds, per arm and q."""
    rows = []
    for arm in ("A_detmean", "P20_lr3e-5", "P20_lr3e-4"):
        for q in QS:
            frames = []
            for s in SEEDS:
                d = (Path(args.r1_root) / "stage2" if arm == "A_detmean" else Path(args.root) / "waveP") / f"q{q}" / f"seed{s}" / arm
                lg = R.detmean_logged(d)
                frames.append(lg.set_index("local")[["loss_roll_mean", "foc_abs_mean_roll_mean", "e0"]])
            m = sum(frames) / len(frames)
            for local, r in m.iterrows():
                rows.append({"arm": arm, "q": q, "local": int(local), "loss_roll_mean": float(r["loss_roll_mean"]),
                             "foc_abs_mean_roll_mean": float(r["foc_abs_mean_roll_mean"]), "e0": float(r["e0"])})
    return pd.DataFrame(rows)


def fig_wavep_logged(fig_dir: Path, lt: pd.DataFrame) -> List[str]:
    """Logged loss and FOC residual (rolling mean over 20 updates, mean over seeds) of the pathwise arms."""
    plt = R._mpl()
    arms = ["A_detmean", "P20_lr3e-5", "P20_lr3e-4"]
    fig, axes = plt.subplots(2, len(QS), figsize=(4.4 * len(QS), 6.0), squeeze=False)
    for j, q in enumerate(QS):
        for i, arm in enumerate(arms):
            g = lt[(lt.arm == arm) & (lt.q == q)]
            kw = dict(color=R.ARM_STYLE["color"][i + 2], label=arm)
            axes[0][j].plot(g["local"], g["loss_roll_mean"], **kw)
            axes[1][j].plot(g["local"], g["foc_abs_mean_roll_mean"], **kw)
        axes[0][j].set_title(f"q = {q}: logged loss (rolling mean 20)")
        axes[1][j].set_title(f"q = {q}: logged FOC residual (rolling mean 20)")
        axes[1][j].set_xlabel("local update")
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("Wave P: logged loss and FOC residual (P20: mean of the pre-step losses / after the last step; A_detmean: "
                 "pre-step: not one time basis)", fontsize=8)
    fig.tight_layout()
    return R.save_fig(fig, fig_dir / "waveP_logged_trajectory")


def arm_figures(res: Result, comps: Sequence[Tuple[str, str, str]], fig_dir: Path, stem: str, title: str
                ) -> Dict[str, List[Tuple[str, List[str]]]]:
    """Paired-dot figure per arm (against its criterion comparator): {arm: [(caption, files)]}."""
    paired, seed_level = res.tabs["paired"], res.tabs["paired_seed_level"]
    out: Dict[str, List[Tuple[str, List[str]]]] = {}
    for arm, base, label in comps:
        if label not in R.CRITERION_LABELS:
            continue
        sl = seed_level[(seed_level.arm == arm) & (seed_level.baseline == base) & (seed_level.metric == PRIMARY)
                        & (seed_level.comparison == label)]
        sm = paired[(paired.arm == arm) & (paired.baseline == base) & (paired.metric == PRIMARY)
                    & (paired.comparison == label)]
        files = R.fig_paired_dots(fig_dir / f"{stem}_{arm}_vs_{base}", sl, sm, QS, f"{title}: {arm} - {base}",
                                  "paired difference of |peak error|")
        out.setdefault(arm, []).append((f"{arm} minus {base}: seed-level paired differences of |peak error| (negative = better)",
                                        files))
    return out


def cross_arm_tables(doc: R.Doc, res: Result, comps: Sequence[Tuple[str, str, str]], arms_tail: Sequence[str], wave: str,
                     crit_arms: Sequence[str], base_text: str) -> None:
    """The cross-arm tables (every comparison of the wave in one table per quantity)."""
    paired, crit = res.tabs["paired"], res.tabs["criterion"]
    doc.h(3, "Primary metric, every comparison")
    doc.table(comp_rows(paired, comps, PRIMARY), f"{wave}_primary", note="`paired.csv`")
    doc.h(3, "Criterion, every arm")
    doc.table(crit_display(crit[crit.arm.isin(crit_arms)]), f"{wave}_criterion", note="`criterion.csv`")
    tail_block(doc, res, arms_tail, f"{wave}_tail")
    doc.h(3, f"Other metrics ({base_text})")
    for m, lab in SECONDARY:
        sub = comp_rows(paired, comps, m)
        if len(sub):
            doc.p(f"**{lab}** (`{m}`)")
            doc.table(sub, f"{wave}_metric_{m}")
    doc.h(3, "Seed-level values")
    st = res.tabs["seed_table"]
    doc.table(st[st.arm.isin(crit_arms + (R1_PAIR if wave == "waveP" else []))].dropna(axis=1, how="all"),
              f"{wave}_seed_level")


def write_wave_a(res: Result, args: argparse.Namespace) -> Path:
    """``03_pilot_waveA.md``."""
    doc = R.Doc(Path(args.out))
    fig_dir, rep_dir = Path(args.figures), Path(args.reports)
    wave_header(doc, "R2b pilot wave A: peak-focused starts and censored likelihood (full Phase A from scratch)",
                "Candidate: the stage-2 last iterate at global update 1600. Baseline: the R1 `parents_A` runs (`A_base`; "
                "end-of-A state bit-identical to `rehearsal_v1_1` in 20/20, `results/v2_refine_r2b/parents.csv`). Arms: "
                "`A_peak25`, `A_peak50` (start_weights peak_focused, half-width 20, share 0.25 / 0.50) and `A_censored` "
                "(clamp_likelihood = censored); one change each, paired by (q, seed). Definitions, criterion and statistics: "
                "`01_preregistration.md` section 6.", args, "all")
    doc.h(2, "1. Checks: provenance, completeness, method")
    provenance_block(doc, res, args, [BASE_A] + ARMS_A, "waveA")
    method_block(doc)
    doc.h(2, "2. Arms at a glance")
    doc.p("One row per arm against the `A_base` baseline; the sections of section 3 give every table of one arm.")
    doc.table(arm_index(res, ARMS_A), "waveA_arm_index")
    doc.h(2, "3. One section per arm")
    figs = arm_figures(res, COMPS_A, fig_dir, "waveA", "Wave A")
    for arm in ARMS_A:
        arm_section(doc, res, arm, COMPS_A, "waveA", figs.get(arm, []), fig_dir, rep_dir)
    doc.h(2, "4. Cross-arm tables")
    cross_arm_tables(doc, res, COMPS_A, [BASE_A] + ARMS_A, "waveA", ARMS_A, "paired, arm - A_base")
    doc.h(3, "Location-free peak and its argmax d")
    locfree_block(doc, res, [BASE_A] + ARMS_A, "waveA_locfree_argmax")
    doc.h(3, "Wave-A specifics (per run)")
    df = res.df
    cols = ["arm", "q", "seed", "peak_visit_share", "peak_visit_design_share", "d1_flagged_L_s2", "d1_flagged_L_s2_in",
            "d1_flagged_L_s2_out", "d1_first_flagged_update", "censored_rows_total", "censored_first_update",
            "tail2q_mean_e2hat", "tail2q_mean_abs_err_over_g2_0", "tail2q_max_abs_err_over_g2_0",
            "offpath_delta2_max_over_dw", "offpath_delta2_mean_over_dw", "onpath_delta2_max_over_dw"]
    spec = df[df.arm.isin([BASE_A] + ARMS_A) & df.complete][[c for c in cols if c in df.columns]]
    doc.p("Per run: the share of the learner's stage-2 starts in the peak set (cumulative over the phase, from the last "
          "verifier call; the design share is the bin-balanced one, 4/40 and 4/44), the D1 clamp counts of the learner at "
          "stage 2 (total, inside and outside the support |d| < 2q), the censored-row counts of `A_censored`, the stage-2 "
          "profile on |d| >= 2q and the off-path Delta_2 / DW (final-tier verifier grid).")
    doc.table(spec, "waveA_specifics")
    g = spec.groupby(["arm", "q"]).mean(numeric_only=True).reset_index().drop(columns=["seed"], errors="ignore")
    doc.p("Mean over the 10 seeds:")
    doc.table(g, "waveA_specifics_mean")
    doc.h(3, "RNG divergence from the baseline")
    doc.table(res.tabs["rng_divergence_A"], "waveA_rng",
              note="`peak_focused` draws its bins with `rng.random`, `balanced` with `rng.integers`: the `start` stream differs "
                   "from the baseline from the first update by construction; for `A_censored` the `learn` and `opp` streams "
                   "first differ a few updates after the first update whose buffer holds a flagged row (their position "
                   "depends on the policy's draws; `d1_first_flagged_update` is in `waveA_specifics.csv`)")
    optimisation_block(doc, res, [BASE_A] + ARMS_A, "cost_A", "waveA")
    doc.h(2, "5. Anomalies")
    anomalies_block(doc, res, [BASE_A] + ARMS_A)
    doc.h(2, "6. Commands")
    commands_block(doc, args, "waveA")
    path = rep_dir / "03_pilot_waveA.md"
    path.write_text(doc.text())
    return path


def write_wave_p(res: Result, args: argparse.Namespace) -> Path:
    """``04_pilot_waveP.md``."""
    doc = R.Doc(Path(args.out))
    fig_dir, rep_dir = Path(args.figures), Path(args.reports)
    wave_header(doc, "R2b pilot wave P: pathwise terminal fine-tuning with a matched optimiser budget",
                "Candidate: the stage-2 last iterate after the 200 updates from the `rehearsal_v1_1` end-of-A state "
                "(parent SHA-256 in `parents.csv`). Arms: `P20_lr3e-5`, `P20_lr3e-4` (phase P, E = 10 epochs x M = 256 "
                "minibatches over 512 rows = 20 exact-gradient steps per update) and the new PPO control `A_ctrl200_lr3e-4`; "
                "comparators reused from R1: `A_ctrl200` (PPO, LR 3e-5), `A_detmean` (one exact step per update, LR 3e-5) "
                "and the parent candidate. Each pathwise arm is matched to the PPO control with the same LR "
                "(`P20_lr3e-5` with `A_ctrl200`, `P20_lr3e-4` with `A_ctrl200_lr3e-4`). Mechanism 3 is model-based "
                "(ablation and diagnostic only).", args, "all")
    doc.h(2, "1. Checks: provenance, completeness, method")
    provenance_block(doc, res, args, R1_PAIR + ARMS_P, "waveP")
    method_block(doc)
    doc.h(2, "2. Arms at a glance")
    doc.p("One row per arm against its criterion comparator (pathwise arms: the PPO control with the same LR; the new "
          "control: the parent u1600 candidate); the sections of section 3 give every table of one arm.")
    doc.table(arm_index(res, ARMS_P), "waveP_arm_index")
    doc.h(2, "3. One section per arm")
    figs = arm_figures(res, COMPS_P, fig_dir, "waveP", "Wave P")
    for arm in ARMS_P:
        arm_section(doc, res, arm, COMPS_P, "waveP", figs.get(arm, []), fig_dir, rep_dir)
    doc.h(2, "4. Cross-arm tables")
    cross_arm_tables(doc, res, COMPS_P, [PARENT] + R1_PAIR + ARMS_P, "waveP", ARMS_P, "paired")
    doc.h(3, "Location-free peak and its argmax d")
    locfree_block(doc, res, [PARENT] + R1_PAIR + ARMS_P, "waveP_locfree_argmax")
    doc.h(3, "Pathwise loss, first-order-condition residual and steps")
    df = res.df
    cols = ["arm", "q", "seed", "n_steps_total", "n_minibatch_steps_total", "loss_first20_mean", "loss_last20_mean",
            "foc_logged_mean_last20", "foc_logged_max_last20", "gn_pre_clip_last20_mean", "gn_pre_clip_max",
            "J_offline_start", "J_offline_end", "foc_offline_end_mean", "foc_offline_end_max", "phaseP_checks_all_true"]
    pw = df[df.arm.isin(["A_detmean", "P20_lr3e-5", "P20_lr3e-4"]) & df.complete][[c for c in cols if c in df.columns]]
    doc.p("The logged `loss` / FOC residual / e(0) of `pathwise_update` (P20 arms) are the mean of the pre-step minibatch "
          "losses and the values AFTER the last step on all 512 rows; R1's `A_detmean` logs them at the pre-step "
          "parameters of its single full-batch step, so the logged curves are not on one time basis (a one-update offset "
          "for e(0) and the FOC residual). The end-of-phase comparison therefore uses the OFFLINE fixed-grid quantities "
          "evaluated from the saved weights with the same function for every arm (`refine_analysis.offline_objective`): "
          "`J_offline_*` is the mean expected payoff on the bin centres under symmetric self-play, reported as a descriptive "
          "level (the pathwise update ascends the expected payoff of one player against a lagged opponent, not this "
          "quantity); `foc_offline_*` is the first-order-condition residual, lower is better.")
    doc.table(pw, "waveP_pathwise")
    gm = pw.groupby(["arm", "q"]).mean(numeric_only=True).reset_index().drop(columns=["seed"], errors="ignore")
    doc.p("Mean over the 10 seeds:")
    doc.table(gm, "waveP_pathwise_mean")
    tr = (res.traj.groupby(["arm", "q", "update"]).agg(J=("J", "mean"), foc_mean=("foc_mean", "mean"),
                                                        foc_max=("foc_max", "max")).reset_index())
    doc.p("Offline objective and FOC residual at every weight export (update 1600 = the parent state; mean over seeds; "
          "FOC max = maximum over seeds):")
    doc.table(tr, "waveP_offline_trajectory")
    lt = logged_trajectory(args)
    lt.to_csv(Path(args.out) / "waveP_logged_trajectory.csv", index=False)
    doc.p("Figure of the offline objective J and the offline FOC residual at the weight exports for all five wave-P-type "
          "arms (the table above is its data):")
    doc.figures([("Wave P: offline objective and FOC residual at the weight exports, all arms",
                  fig_wavep_offline(fig_dir, res.traj))], fig_dir, rep_dir)
    doc.p("Logged loss and FOC residual per update of the pathwise arms (rolling mean over 20 updates, mean over seeds; the "
          "time bases of `pathwise_update` and R1's `pathwise_step` differ, see above):")
    doc.figures([("Wave P: logged loss and FOC residual of the pathwise arms (time bases differ, see text)",
                  fig_wavep_logged(fig_dir, lt))], fig_dir, rep_dir)
    doc.p(f"Source: `{R._rel(Path(args.out) / 'waveP_logged_trajectory.csv')}`.")
    doc.h(3, "RNG divergence from the matched PPO control")
    rng = res.df[res.df.arm.isin(ARMS_P) & res.df.complete][
        ["arm", "q", "seed"] + [c for c in res.df.columns if c.startswith("rng_div_")]]
    doc.table(rng, "waveP_rng", note="first update at which each stream differs from the matched PPO control run "
              "(`P20_lr3e-5`: `A_ctrl200`; `P20_lr3e-4`: `A_ctrl200_lr3e-4`; `A_ctrl200_lr3e-4`: `A_ctrl200`); "
              "the minibatch stream is expected to stay aligned for the pathwise arms by construction")
    optimisation_block(doc, res, R1_PAIR + ARMS_P, "cost_P", "waveP")
    doc.h(2, "5. Anomalies")
    anomalies_block(doc, res, R1_PAIR + ARMS_P)
    doc.h(2, "6. Commands")
    commands_block(doc, args, "waveP")
    path = rep_dir / "04_pilot_waveP.md"
    path.write_text(doc.text())
    return path


MECH = {"A_peak25": ("1 peak-focused starts", "PPO-internal; eligible for v2.1 if the criterion is met", "A"),
        "A_peak50": ("1 peak-focused starts", "PPO-internal; eligible for v2.1 if the criterion is met", "A"),
        "A_censored": ("2 censored likelihood", "PPO-internal; eligible for v2.1 if the criterion is met", "A"),
        "P20_lr3e-5": ("3 pathwise, 20 steps/update", "model-based; ablation and diagnostic only (owner decides)", "P"),
        "P20_lr3e-4": ("3 pathwise, 20 steps/update", "model-based; ablation and diagnostic only (owner decides)", "P")}


def failing_components(df: pd.DataFrame, arm: str, q: int, seed: int) -> str:
    """Which parts of G-A and its G-N part the run fails (names with the run's value), from the per-run table."""
    r = df[(df.arm == arm) & (df.q == q) & (df.seed == seed)]
    if r.empty:
        return "run missing"
    r = r.iloc[0]
    out = []
    for flag, name, col, lim in (("G_A_eta_pass", "eta_2/DW", "eta_T_over_dw", 0.005),
                                 ("G_A_rmse_pass", "RMSE_pos", "stage2_rmse_pos_over_g2_0", 0.05),
                                 ("G_A_tail_pass", "tail mean", "stage2_tail_mean_over_g2_0", 0.02)):
        if str(r.get(flag)) == "False":
            out.append(f"{name} {R._g(float(r[col]))} > {lim}")
    if str(r.get("G_N_eta")) == "False":
        out.append(f"|dev - final| of eta_2 {R._g(float(r['eta_dev_minus_final_abs']))} > 0.001")
    return ", ".join(out) if out else "none (gate computed from the run's own flags)"


def violation_text(df: pd.DataFrame, arm: str, violations: str) -> str:
    """``q60/10503 [tail mean 0.02038 > 0.02]`` for every violating run of part (b)."""
    parts = []
    for tok in violations.split():
        q, s = tok.split("/")
        parts.append(f"{tok} [{failing_components(df, arm, int(q[1:]), int(s))}]")
    return "; ".join(parts)


def mech_rows(res: Result) -> pd.DataFrame:
    """One row per arm of the three mechanisms: primary paired difference per q with CI, criterion, tails, cost."""
    paired, crit, tail = res.tabs["paired"], res.tabs["criterion"], res.tabs["tail"]
    cost = {"A": res.tabs["cost_A"], "P": res.tabs["cost_P"]}
    rows = []
    for arm, (mname, elig, w) in MECH.items():
        base = BASE_A if w == "A" else MATCHED_CONTROL[arm]
        row: Dict[str, Any] = {"mechanism": mname, "arm": arm, "comparator": base}
        for q in QS:
            r = paired[(paired.arm == arm) & (paired.baseline == base) & (paired.q == q) & (paired.metric == PRIMARY)
                       & paired.comparison.isin(R.CRITERION_LABELS)]
            if len(r):
                r = r.iloc[0]
                row[f"q{q} mean diff [95% CI]"] = f"{R._g(r['mean'])} {R.ci_str(r['ci_mean_lo'], r['ci_mean_hi'])}"
                row[f"q{q} n_better/n_zero"] = f"{_int(r['n_better'])}/{_int(r['n_zero'])} of {int(r['n_pairs'])}"
        c = crit[(crit.arm == arm) & (crit.baseline == base)]
        if len(c):
            c = c.iloc[0]
            row["(a) both q"] = bool(c["a_met"])
            row["(b)"] = c["b_status"] + (f" ({violation_text(res.df, arm, c['b_violations'])})"
                                          if c["b_violations"] else "")
        for who, key in ((arm, "arm"), (base, "comparator")):
            for q in QS:
                t = tail[(tail.arm == who) & (tail.q == q)]
                if len(t):
                    t = t.iloc[0]
                    row[f"q{q} {key}: max|peak|, n<=0.05, n eta>0.004"] = (
                        f"{R._g(t['max_abs_peak_error'])}, {t['n_abs_peak_le_0.05']}, {t['n_eta_over_0.004']}")
        cc = cost[w][cost[w].arm == arm]
        cb = cost[w][cost[w].arm == base]
        if len(cc):
            row["mean phase wall sec/run"] = float(cc["mean_phase_wall_sec"].iloc[0])
            row["mean optimiser steps/run"] = cc["mean_n_minibatch_steps_total"].iloc[0]
        if len(cb):
            row["comparator: wall sec/run, optimiser steps/run"] = (
                f"{R._g(float(cb['mean_phase_wall_sec'].iloc[0]))}, {R._g(float(cb['mean_n_minibatch_steps_total'].iloc[0]))}")
        row["protocol eligibility (D2)"] = elig
        rows.append(row)
    return pd.DataFrame(rows)


def observations(res: Result) -> List[str]:
    """Computed sentences for the 'observations' section (descriptive; every number from the tables)."""
    paired, crit = res.tabs["paired"], res.tabs["criterion"]
    out: List[str] = []
    for arm, base, label in COMPS_A + COMPS_P[:3]:
        for q in QS:
            r = paired[(paired.arm == arm) & (paired.baseline == base) & (paired.q == q)
                       & (paired.metric == PRIMARY) & (paired.comparison == label)]
            if r.empty:
                continue
            r = r.iloc[0]
            side = ("excludes 0 on the improving side" if r["ci_mean_hi"] < 0 else
                    ("excludes 0 on the worsening side" if r["ci_mean_lo"] > 0 else "contains 0"))
            out.append(f"`{arm}` minus `{base}`, q = {q}: mean difference of |peak error| {R._g(r['mean'])} "
                       f"{R.ci_str(r['ci_mean_lo'], r['ci_mean_hi'])} ({side}); {_int(r['n_better'])} of {int(r['n_pairs'])} "
                       f"seeds improve, {_int(r['n_zero'])} tie, {_int(r['n_pos'])} worsen.")
    for _, c in crit.iterrows():
        out.append(f"`{c['arm']}` vs `{c['baseline']}` ({c['comparison']}): part (a) "
                   f"{'met at both q' if c['a_met'] else 'not met'} (a_q50 = {c.get('a_q50')}, a_q60 = {c.get('a_q60')}); "
                   f"part (b) {c['b_status']}"
                   + (f" (violations: {violation_text(res.df, c['arm'], c['b_violations'])})" if c["b_violations"] else "")
                   + f"; overall {c['overall']}.")
    df = res.df
    for arm in [BASE_A] + ARMS_A:
        for q in QS:
            g = df[(df.arm == arm) & (df.q == q) & df.complete]
            if len(g) and "peak_visit_share" in g:
                out.append(f"`{arm}`, q = {q} (mean over {len(g)} seeds): share of the learner's stage-2 starts in the peak set "
                           f"{R._g(g['peak_visit_share'].mean())} (the bin-balanced design share is "
                           f"{R._g(g['peak_visit_design_share'].mean())}); mean e2-hat on |d| >= 2q "
                           f"{R._g(g['tail2q_mean_e2hat'].mean())} effort units; tail mean / e2*(0) "
                           f"{R._g(R._num(g['stage2_tail_mean_over_g2_0']).mean())}; flagged learner stage-2 raw draws per run "
                           f"{R._g(g['d1_flagged_L_s2'].mean())}.")
    if len(res.traj):
        tr = res.traj.groupby(["arm", "q", "update"])[["J", "foc_mean"]].mean().reset_index()
        for q in QS:
            parts = []
            for arm in R1_PAIR + ARMS_P:
                g = tr[(tr.arm == arm) & (tr.q == q)].sort_values("update")
                if len(g):
                    parts.append(f"`{arm}` {R._g(g['foc_mean'].iloc[0])} -> {R._g(g['foc_mean'].iloc[-1])}")
            out.append(f"Wave P, q = {q}: offline FOC residual (mean |dR/de| on the bin centres, mean over 10 seeds) from the "
                       "parent state to the end of the phase: " + "; ".join(parts) + " (`waveP_offline_trajectory.csv`).")
    cs = df[(df.arm == "A_censored") & df.complete]
    if len(cs) and "censored_rows_total" in cs:
        tot = cs["censored_rows_total"].fillna(0)
        out.append(f"`A_censored`: {int((tot == 0).sum())} of {len(cs)} runs never had a flagged row, so they equal their "
                   f"baseline exactly; the others had {int(tot[tot > 0].min()) if (tot > 0).any() else 0} to {int(tot.max())} "
                   "flagged rows (`waveA_specifics.csv`).")
    return out


def write_decision(res: Result, args: argparse.Namespace) -> Path:
    """``05_decision_inputs.md``: one row per mechanism arm, then the labelled observations."""
    doc = R.Doc(Path(args.out))
    rep_dir = Path(args.reports)
    doc.h(1, "R2b decision inputs")
    doc.p("Generated by `tools/v2/refine_r2b_analysis.py decision`. One row per arm of the three mechanisms; all numbers are "
          "read from the tables of `03_pilot_waveA.md` and `04_pilot_waveP.md`. The criterion is descriptive "
          "(pre-registration section 6): (a) the 95% bootstrap CI of the mean paired difference of |peak error| excludes 0 on "
          "the improving side at both q; (b) no run that passed G-A and its G-N part under the comparator fails it under the "
          "arm. The comparator is the baseline `parents_A` (wave A) or the PPO control with the same LR (wave P). Tail "
          "columns: max |peak error| over the 10 seeds, number of runs with |peak error| <= 0.05, number with eta_2/DW > "
          "0.004. Cost per run is the mean wall time of the phase and the optimiser steps of the arm and of its comparator; "
          "wall time depends on the machine load (wave A and wave P ran together, up to 40 processes, load average 6 to 43; "
          "the R1 comparators ran on 2026-10-03), so only the step counts are comparable across arms. Protocol "
          "eligibility is the statement of D2 of the round; this report decides nothing.")
    method_block(doc)
    doc.table(mech_rows(res), "decision_inputs")
    doc.h(2, "Observations (labelled: descriptive, computed from the tables above)")
    doc.p("These sentences restate the tables; they are not a recommendation, and a CI that contains 0 with 10 seeds does "
          "not show that a mechanism has no effect.")
    for s in observations(res):
        doc.lines.append(f"- {s}")
    doc.lines.append("")
    path = rep_dir / "05_decision_inputs.md"
    path.write_text(doc.text())
    return path


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    p = argparse.ArgumentParser(description="R2b pre-registered analysis")
    p.add_argument("cmd", choices=["all", "waveA", "waveP", "decision"])
    p.add_argument("--root", default=str(L.R2B_ROOT))
    p.add_argument("--r1-root", required=True)
    p.add_argument("--ref-root", required=True)
    p.add_argument("--out", default=str(L.R2B_ROOT / "analysis"))
    p.add_argument("--reports", default=str(HERE.parents[2] / "reports" / "v2" / "refine_r2b"))
    p.add_argument("--figures", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--qs", type=int, nargs="+", default=list(QS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the analysis and write the CSVs and the reports 03, 04, 05."""
    a = build_parser().parse_args(argv)
    a.figures = a.figures or str(Path(a.reports) / "figures")
    for k in ("root", "r1_root", "ref_root", "out", "reports", "figures"):
        setattr(a, k, str(Path(getattr(a, k)).resolve()))
    Path(a.reports).mkdir(parents=True, exist_ok=True)
    res = compute(a)
    if a.cmd in ("all", "waveA"):
        print(f"wrote {write_wave_a(res, a)}", flush=True)
    if a.cmd in ("all", "waveP"):
        print(f"wrote {write_wave_p(res, a)}", flush=True)
    if a.cmd in ("all", "decision"):
        print(f"wrote {write_decision(res, a)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
