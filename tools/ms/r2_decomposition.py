#!/usr/bin/env python3
"""MS-R2 premise check (prompt section 2.4): the d = 0 gap of the terminal stage splits into a noise-smoothing part
and a remainder, reproduced from the MS-R1 per-run table.

    gap       = e2*(0) - e_hat_2(0)
    smoothing = e2*(0) - e_sigma(0)        e_sigma(0) = ``smoothed_e_pred_0`` (the tie first-order condition of the
                                           game both players play with the learned noise, `run/run_v2_T2_locked.py:
                                           smoothed_share`)
    remainder = e_sigma(0) - e_hat_2(0)

and, for Gaussian noise, smoothing = e2*(0) * sigma_2(0) / (sqrt(pi) * q) (derivation in ``reports/ms/r2/
01_decomposition.md``). Inputs: ``results/ms_r1/analysis/per_run.csv`` of ``origin/ms-r1`` (columns ``g2_at_0``,
``e2_at_0``, ``smoothed_e_pred_0``, ``sigma_effort_at_0_t2``; rows ``role = ms_arm``). Optionally the freeze arrays of
the runs (``--pilot-root``, ``--base-root``) give the learned Beta parameters at d = 0, from which e_sigma(0) is
recomputed independently and the skewness and excess kurtosis of the noise are reported.

Outputs (``--out``): ``decomposition_per_run.csv``, ``decomposition_table.csv`` (arm x q means),
``decomposition_paired.csv`` (paired differences with percentile bootstrap intervals), ``decomposition_checks.json``;
``--markdown`` prints the tables. Exit code 3 if the ratio check fails (any run outside [0.99, 1.01]) or the table is
not reproduced to its second decimal (the stop condition of section 2.4); 0 otherwise.

Usage:
    python tools/ms/r2_decomposition.py --per-run results/ms_r1/analysis/per_run.csv \
        --pilot-root <MS-R1 pilot root> --base-root <MS-R1 base root> --out results/ms_r2/decomposition --markdown
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.ms_noise import beta_noise_moments, decompose_gap, smoothed_tie_prediction  # noqa: E402

N_BOOT = 10000
BOOT_SEED = 20261007
QS: Tuple[int, ...] = (50, 60)
SEEDS: Tuple[int, ...] = tuple(range(10501, 10511))
SAMPLER_ARMS: Tuple[str, ...] = ("MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5")
RATIO_LIMITS = (0.99, 1.01)

#: The table of the PI's preamble (effort units, mean over the 10 seeds): (arm, q) -> sigma, gap, smoothing, remainder.
PREAMBLE: Dict[Tuple[str, int], Tuple[float, float, float, float]] = {
    ("MS_base", 50): (2.92, 4.59, 2.30, 2.29), ("MS_base2400", 50): (2.53, 3.71, 2.00, 1.71),
    ("MS_s35a5", 50): (2.50, 2.40, 1.97, 0.43), ("MS_base", 60): (2.99, 3.11, 1.64, 1.47),
    ("MS_base2400", 60): (2.59, 2.62, 1.42, 1.20), ("MS_s35a5", 60): (2.57, 2.54, 1.41, 1.13)}
TABLE_ARMS = ("MS_base", "MS_base2400", "MS_s35a5")


# ------------------------------------------------------------------------------------------------ the formulas
# (utils/ms_noise.py: smoothed_tie_prediction, beta_noise_moments, decompose_gap, the Gaussian formula)
decompose = decompose_gap


# ------------------------------------------------------------------------------------------------ statistics
def boot_ci(x: Sequence[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Tuple[float, float]:
    """95 % percentile bootstrap interval of the mean with a fresh ``default_rng(seed)`` (one per call)."""
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------------------------------------ the analysis
def load_per_run(path: Path) -> pd.DataFrame:
    """The MS arms of ``per_run.csv`` with the decomposition columns added (done runs only)."""
    df = pd.read_csv(path, float_precision="round_trip")
    df = df[(df["role"] == "ms_arm") & (df["status"] == "done")].copy()
    rows = []
    for r in df.itertuples():
        d = decompose(float(r.g2_at_0), float(r.smoothed_e_pred_0), float(r.e2_at_0),
                      float(r.sigma_effort_at_0_t2), float(r.q))
        rows.append(d)
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["smoothing_rel"] = out["smoothing"] / out["g2_at_0"]
    out["remainder_rel"] = out["remainder"] / out["g2_at_0"]
    out["gap_rel"] = out["gap"] / out["g2_at_0"]
    return out


def arm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Means over the seeds per (arm, q): sigma_2(0), gap, smoothing part, remainder (effort units)."""
    g = df.groupby(["arm", "q"], sort=False)
    t = g.agg(n=("seed", "count"), sigma=("sigma_effort_at_0_t2", "mean"), gap=("gap", "mean"),
              smoothing=("smoothing", "mean"), remainder=("remainder", "mean"),
              remainder_sd=("remainder", lambda s: s.std(ddof=1)),
              remainder_median=("remainder", "median")).reset_index()
    return t


def paired(df: pd.DataFrame, arm: str, base: str, q: int, col: str) -> Dict[str, Any]:
    """Paired (by seed) difference arm - base of ``col`` with the percentile bootstrap interval of its mean."""
    a = df[(df.arm == arm) & (df.q == q)].set_index("seed")[col]
    b = df[(df.arm == base) & (df.q == q)].set_index("seed")[col]
    seeds = [s for s in SEEDS if s in a.index and s in b.index]
    d = np.array([a.loc[s] - b.loc[s] for s in seeds], dtype=float)
    lo, hi = boot_ci(d)
    return {"arm": arm, "baseline": base, "q": q, "quantity": col, "n": len(seeds), "mean": float(d.mean()),
            "ci_lo": lo, "ci_hi": hi, "contains_0": bool(lo <= 0.0 <= hi)}


def paired_table(df: pd.DataFrame) -> pd.DataFrame:
    """Budget (MS_base2400 - MS_base) and sampler (arm - MS_base2400) differences of sigma, smoothing, remainder, gap."""
    rows: List[Dict[str, Any]] = []
    for q in QS:
        for col in ("sigma_effort_at_0_t2", "smoothing", "remainder", "gap"):
            rows.append({"comparison": "budget", **paired(df, "MS_base2400", "MS_base", q, col)})
    for q in QS:
        for arm in SAMPLER_ARMS:
            for col in ("smoothing", "remainder", "gap"):
                rows.append({"comparison": "sampler", **paired(df, arm, "MS_base2400", q, col)})
    return pd.DataFrame(rows)


def recompute_from_arrays(df: pd.DataFrame, pilot_root: Optional[Path], base_root: Optional[Path]) -> pd.DataFrame:
    """e_sigma(0) and the Beta noise moments from the freeze arrays (alpha, beta at the node d = 0)."""
    rows = []
    for r in df.itertuples():
        root = base_root if r.arm == "MS_base" else pilot_root
        if root is None:
            continue
        d = Path(root) / ("q%d" % r.q) / ("seed%d" % r.seed) / r.arm
        z = np.load(d / "freeze_stage2_final.npz")
        cfg = json.loads((d / "run_config.json").read_text())
        g = cfg["record"]["game"]
        i0 = int(np.argmin(np.abs(z["v_t2_d_grid"])))
        a, b = float(z["v_t2_alpha"][i0]), float(z["v_t2_beta"][i0])
        e_range = float(g["e_max"]) - float(g["e_min"])
        e_sig, sig = smoothed_tie_prediction(a, b, float(g["w_h"]) - float(g["w_l"]), float(g["k"]), float(g["q"]), e_range)
        skew, exk = beta_noise_moments(a, b)
        rows.append({"arm": r.arm, "q": r.q, "seed": r.seed, "alpha0": a, "beta0": b, "d_node": float(z["v_t2_d_grid"][i0]),
                     "e_sigma_recomputed": e_sig, "sigma_recomputed": sig, "skewness": skew, "excess_kurtosis": exk,
                     "diff_e_sigma": e_sig - float(r.smoothed_e_pred_0),
                     "diff_sigma": sig - float(r.sigma_effort_at_0_t2)})
    return pd.DataFrame(rows)


def run_checks(df: pd.DataFrame, table: pd.DataFrame, pair: pd.DataFrame, rec: pd.DataFrame) -> Dict[str, Any]:
    """The premise checks of section 2.4 and the numbers quoted in the preamble."""
    out: Dict[str, Any] = {"n_runs": int(len(df))}
    out["ratio_min"], out["ratio_max"] = float(df["ratio"].min()), float(df["ratio"].max())
    bad = df[(df["ratio"] < RATIO_LIMITS[0]) | (df["ratio"] > RATIO_LIMITS[1])]
    out["ratio_outside_limits"] = [f"{r.arm} q{r.q} s{r.seed}" for r in bad.itertuples()]
    out["ratio_check_pass"] = bool(len(bad) == 0 and len(df) == 140)
    diffs = []
    for (arm, q), (sg, gap, sm, rem) in PREAMBLE.items():
        t = table[(table.arm == arm) & (table.q == q)].iloc[0]
        mine = (float(t.sigma), float(t.gap), float(t.smoothing), float(t.remainder))
        diffs.append({"arm": arm, "q": q, "preamble": [sg, gap, sm, rem], "reproduced": [round(x, 4) for x in mine],
                      "rounded_2dp": [round(x, 2) for x in mine],
                      "match_2dp": bool(all(abs(round(m, 2) - p) < 1e-9 for m, p in zip(mine, (sg, gap, sm, rem))))})
    out["table"] = diffs
    out["table_reproduced"] = bool(all(d["match_2dp"] for d in diffs))
    out["learned_peak_below_closed_form"] = int((df["e2_at_0"] < df["g2_at_0"]).sum())
    sa = df[(df.arm == "MS_s35a5") & (df.q == 50)]
    out["median_remainder_MS_s35a5_q50"] = float(sa["remainder"].median())
    out["remainder_seed_sd_range"] = [float(table["remainder_sd"].min()), float(table["remainder_sd"].max())]
    s = pair[pair.comparison == "sampler"]
    sm = s[s.quantity == "smoothing"]
    out["sampler_smoothing_max_abs"] = float(sm["mean"].abs().max())
    for q in QS:
        r = s[(s.quantity == "remainder") & (s.q == q)]
        out["sampler_remainder_range_q%d" % q] = [float(r["mean"].min()), float(r["mean"].max())]
        out["sampler_remainder_all_ci_contain_0_q%d" % q] = bool(r["contains_0"].all())
    b = pair[pair.comparison == "budget"]
    for q in QS:
        for col in ("sigma_effort_at_0_t2", "smoothing", "remainder", "gap"):
            r = b[(b.q == q) & (b.quantity == col)].iloc[0]
            out["budget_%s_q%d" % (col, q)] = [float(r["mean"]), float(r["ci_lo"]), float(r["ci_hi"])]
    if len(rec):
        out["recomputed_runs"] = int(len(rec))
        out["recomputed_e_sigma_max_abs_diff"] = float(rec["diff_e_sigma"].abs().max())
        out["recomputed_sigma_max_abs_diff"] = float(rec["diff_sigma"].abs().max())
        out["skewness_range"] = [float(rec["skewness"].min()), float(rec["skewness"].max())]
        out["excess_kurtosis_range"] = [float(rec["excess_kurtosis"].min()), float(rec["excess_kurtosis"].max())]
        out["d_node_max_abs"] = float(rec["d_node"].abs().max())
    out["stop_condition"] = bool(not out["ratio_check_pass"] or not out["table_reproduced"])
    return out


def markdown(table: pd.DataFrame, pair: pd.DataFrame, checks: Dict[str, Any]) -> str:
    """The tables of ``reports/ms/r2/01_decomposition.md`` (generated, not typed)."""
    lines = ["| q | arm | sigma_2(0) | gap | smoothing part | remainder |", "|---|---|---|---|---|---|"]
    for q in QS:
        for arm in TABLE_ARMS:
            t = table[(table.arm == arm) & (table.q == q)].iloc[0]
            lines.append(f"| {q} | `{arm}` | {t.sigma:.2f} | {t.gap:.2f} | {t.smoothing:.2f} | {t.remainder:.2f} |")
    lines.append("")
    lines.append("| comparison | arm - baseline | q | quantity | mean [95 % CI] | contains 0 |")
    lines.append("|---|---|---|---|---|---|")
    for r in pair.itertuples():
        lines.append(f"| {r.comparison} | `{r.arm}` - `{r.baseline}` | {r.q} | {r.quantity} | {r.mean:+.3f} "
                     f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {'yes' if r.contains_0 else 'no'} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--per-run", required=True)
    p.add_argument("--pilot-root", default=None, help="MS-R1 pilot root (q*/seed*/<arm>), for the Beta recomputation")
    p.add_argument("--base-root", default=None, help="MS-R1 base-wave root (q*/seed*/MS_base)")
    p.add_argument("--out", required=True)
    p.add_argument("--markdown", action="store_true")
    a = p.parse_args(argv)
    df = load_per_run(Path(a.per_run))
    table = arm_table(df)
    pair = paired_table(df)
    rec = recompute_from_arrays(df, Path(a.pilot_root) if a.pilot_root else None,
                                Path(a.base_root) if a.base_root else None)
    checks = run_checks(df, table, pair, rec)
    checks["inputs"] = {"per_run": str(Path(a.per_run).resolve()), "pilot_root": a.pilot_root, "base_root": a.base_root,
                        "bootstrap": f"{N_BOOT} resamples, fresh numpy.random.default_rng({BOOT_SEED}) per call"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "decomposition_per_run.csv", index=False)
    table.to_csv(out / "decomposition_table.csv", index=False)
    pair.to_csv(out / "decomposition_paired.csv", index=False)
    if len(rec):
        rec.to_csv(out / "decomposition_beta_recompute.csv", index=False)
    (out / "decomposition_checks.json").write_text(json.dumps(checks, indent=1))
    if a.markdown:
        print(markdown(table, pair, checks))
    print(json.dumps({k: v for k, v in checks.items() if k not in ("table", "inputs")}, indent=1), file=sys.stderr)
    return 3 if checks["stop_condition"] else 0


if __name__ == "__main__":
    sys.exit(main())
