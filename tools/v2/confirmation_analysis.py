#!/usr/bin/env python3
"""Pre-registered analysis of locked-protocol (v1.1) runs: verdicts, pass rule, S1, distributions.

Locked with protocol v1.1 and used UNCHANGED on the v1.1 re-rehearsal and on the confirmation.

For every expected (q, seed) under --root/q{q}/seed{s}/ the run is classified:
  missing              no run directory or no status.json
  failed_exception     status.json state 'failed' (pipeline exception; traceback head reported)
  incomplete           status.json present but not done/failed (e.g. killed) - counted as not passing
  global_rng_violation gates.json global_rng.status == 'violation' (exit code 5) - counted as failed
  completed            status done, exit code 0
Infrastructure-crash attempts moved to --root/crashed/q{q}/seed{s}_attempt*/ are listed.

Verdicts are RECOMPUTED from the saved evaluation arrays, independently of the entry point's code:
  eta_2/DW     = max(v_t2_delta) / DW                         (gateA_{final,development}.npz)
  RMSE/e2*(0)  = sqrt(mean((recovery_e2 - recovery_g2)^2 over |d| < 2q)) / recovery_g2(d=0)   (gateA_final.npz)
  tail/e2*(0)  = mean(recovery_e2 over |d| >= 2q) / recovery_g2(d=0)                          (gateA_final.npz)
  Gmax/DW      = max over t of max(v_t{t}_G) / DW              (final_{final,development}.npz)
  S1           = |v_t1_e_hat[0] - e1*| / e1*                   (final_final.npz; e1* closed form, evaluation only)
with the thresholds of the protocol JSON (inclusive <=, full float64 precision):
  G-A: eta_final, RMSE, tail;  G-F: Gmax_final;  G-N: |eta_dev - eta_final|, |Gmax_dev - Gmax_final|;
  run pass = G-A and G-F and G-N and global-RNG status ok and exit code 0;
  S1 (reported only); v1.0 outcome = G-A and (Gmax_final <= 0.01 and S1) (reported only).
Each recomputed value and verdict is compared with the run's gates.json (agreement table).

Per q: pass count vs the >= 18 of 20 rule (applied only when 20 seeds are expected), exact
Clopper-Pearson 95% CI of the pass rate; overall verdict = both q pass. S1: pass count with
Clopper-Pearson CI; mean signed stage-1 error with a 95% percentile bootstrap CI (10000 resamples,
a fresh numpy.random.default_rng(protocol seed) for each q, q ascending); median and SD (ddof=1).
Distributions (min, p10, p25, median, p75, p90, max) of every gate metric (final, dev, dev - final);
every reported metric of the v1.0 report section 4.4 incl. the stage-1 decomposition with bands;
EXP_root/DW vs the stage-1 relative error per q with a least-squares fit EXP_root/DW = a + b*err^2.
With --compare-root, side-by-side distributions of the two roots (descriptive).

Outputs in --out: per_run.csv, agreement.csv, pass_counts.csv, verdict.json, s1_summary.csv,
distributions.csv, reported_metrics.csv, exp_vs_s1.csv, exp_vs_s1_fit.csv, [compare_distributions.csv],
tables.md.

Usage:
  python tools/v2/confirmation_analysis.py --root results/v2_T2_locked/confirmation --out <dir>
  python tools/v2/confirmation_analysis.py --root results/v2_T2_locked/rehearsal_v1_1 --seeds 10501 10510 --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols" / "v2_T2_locked_v1_1.json"
QUANT = (("min", 0.0), ("p10", 0.10), ("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90), ("max", 1.0))
GATE_METRICS = ("eta_final", "eta_dev", "eta_dev_minus_final", "rmse", "tail", "gmax_final", "gmax_dev",
                "gmax_dev_minus_final", "s1", "stage1_rel_err_signed")


def g1_closed_form(spec: dict) -> float:
    """e1*(0) of the T=2 game (closed form, evaluation only; utils.theory_multistage.g1_two_stage)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.theory_multistage import g1_two_stage
    return float(g1_two_stage(spec["q"], spec["w_h"], spec["w_l"], spec["k"]))


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact two-sided (1 - alpha) CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def recompute(d: Path, game: dict) -> dict:
    """Gate-metric values recomputed from the evaluation NPZs of one run."""
    dw, q = float(game["w_h"]) - float(game["w_l"]), float(game["q"])
    out = {}
    for tier, key in (("final", "eta_final"), ("development", "eta_dev")):
        z = np.load(d / f"gateA_{tier}.npz")
        out[key] = float(z["v_t2_delta"].max()) / dw
    z = np.load(d / "gateA_final.npz")
    D, e2, g2 = z["recovery_d_grid"], z["recovery_e2"], z["recovery_g2"]
    g20 = float(g2[np.nonzero(D == 0.0)[0][0]])
    pos = np.abs(D) < 2.0 * q
    out["rmse"] = float(np.sqrt(np.mean((e2[pos] - g2[pos]) ** 2))) / g20
    out["tail"] = float(e2[~pos].mean()) / g20
    for tier, key in (("final", "gmax_final"), ("development", "gmax_dev")):
        z = np.load(d / f"final_{tier}.npz")
        out[key] = max(float(z[k].max()) for k in z.files if k.startswith("v_t") and k.endswith("_G")) / dw
    z = np.load(d / "final_final.npz")
    e1 = float(z["v_t1_e_hat"][0])
    g1 = g1_closed_form(game)
    out["stage1_rel_err_signed"] = (e1 - g1) / g1
    out["s1"] = abs(out["stage1_rel_err_signed"])
    out["e1_at_0"] = e1
    out["eta_dev_minus_final"] = out["eta_dev"] - out["eta_final"]
    out["gmax_dev_minus_final"] = out["gmax_dev"] - out["gmax_final"]
    return out


def decide(v: dict, proto: dict) -> dict:
    """Verdicts from values (independent implementation of protocol v1.1 section 3)."""
    g = proto["gates"]
    th = {c["metric"]: float(c["threshold"]) for blk in ("G-A", "G-F", "G-N") for c in g[blk]["all_must_hold"]}
    t1 = float(proto["secondary"]["S1"]["criterion"]["threshold"])
    ga = (v["eta_final"] <= th["eta_T_over_dw"] and v["rmse"] <= th["stage2_rmse_pos_over_g2_0"]
          and v["tail"] <= th["stage2_tail_mean_over_g2_0"])
    gf = v["gmax_final"] <= th["Gmax_full_over_dw"]
    gn = (abs(v["eta_dev"] - v["eta_final"]) <= th["eta_T_over_dw_dev_minus_final_abs"]
          and abs(v["gmax_dev"] - v["gmax_final"]) <= th["Gmax_full_over_dw_dev_minus_final_abs"])
    s1 = v["s1"] <= t1
    return {"G-A": bool(ga), "G-F": bool(gf), "G-N": bool(gn), "gates_pass": bool(ga and gf and gn), "S1": bool(s1),
            "G-F_v1_0": bool(gf and s1), "run_pass_v1_0": bool(ga and gf and s1)}


def classify(d: Path):
    st_path = d / "status.json"
    if not st_path.exists():
        return "missing", None, None
    st = json.load(open(st_path))
    if st.get("state") == "failed":
        return "failed_exception", st.get("exit_code"), (st.get("traceback") or "").strip().splitlines()[-1:]
    if st.get("state") != "done":
        return "incomplete", st.get("exit_code"), None
    g = json.load(open(d / "gates.json"))
    if g.get("global_rng", {}).get("status") != "ok":
        return "global_rng_violation", st.get("exit_code"), g.get("global_rng", {}).get("violations")
    return "completed", st.get("exit_code"), None


def analyse(root: Path, proto: dict, seeds) -> tuple:
    rows, agree, rep = [], [], []
    for q in proto["q_values"]:
        game = proto["records"][str(q)]["game"]
        for s in seeds:
            d = root / f"q{q}" / f"seed{s}"
            status, code, info = classify(d)
            crashed = sorted(os.path.relpath(p, root) for p in glob.glob(str(root / "crashed" / f"q{q}" / f"seed{s}_attempt*")))
            r = {"q": q, "seed": s, "status": status, "exit_code": code, "status_info": json.dumps(info) if info else "",
                 "crashed_attempts": " ".join(crashed)}
            if status in ("completed", "global_rng_violation"):
                g = json.load(open(d / "gates.json"))
                summ = json.load(open(d / "v2_run_summary.json"))
                v = recompute(d, game)
                dec = decide(v, proto)
                r.update({k: v[k] for k in GATE_METRICS})
                r.update({f"{k}_pass": dec[k] for k in ("G-A", "G-F", "G-N", "S1")})
                r["eta_N_pass"] = abs(v["eta_dev_minus_final"]) <= float(proto["gates"]["G-N"]["all_must_hold"][0]["threshold"])
                r["gmax_N_pass"] = abs(v["gmax_dev_minus_final"]) <= float(proto["gates"]["G-N"]["all_must_hold"][1]["threshold"])
                r["global_rng"] = g["global_rng"]["status"]
                r["run_pass"] = bool(dec["gates_pass"] and status == "completed" and code == 0)
                r["v1_0_G-F"] = dec["G-F_v1_0"]
                r["v1_0_run_pass"] = dec["run_pass_v1_0"]
                r["wall_sec"] = summ.get("total_wall_sec")
                r["outcome"] = ("global_rng_violation" if status != "completed" else "pass" if r["run_pass"]
                                else "stage2_failure" if not dec["G-A"] else "fail_G-F" if not dec["G-F"] else "fail_G-N")
                mv = g["metric_values"]
                a = {"q": q, "seed": s}
                for k in ("eta_final", "eta_dev", "rmse", "tail", "gmax_final", "gmax_dev", "s1"):
                    a[f"absdiff_{k}"] = abs(float(mv[k]) - v[k])
                a.update({"G-A": dec["G-A"] == g["G-A"]["pass"], "G-F": dec["G-F"] == g["G-F"]["pass"],
                          "G-N": dec["G-N"] == g["G-N"]["pass"], "S1": dec["S1"] == g["S1"]["pass"],
                          "v1_0_G-F": dec["G-F_v1_0"] == g["v1_0_outcome"]["G-F_v1_0"],
                          "v1_0_run_pass": dec["run_pass_v1_0"] == g["v1_0_outcome"]["run_pass_v1_0"],
                          "run_pass": r["run_pass"] == g["run_pass"], "outcome": r["outcome"] == g["outcome"]})
                a["all_agree"] = all(a[k] for k in ("G-A", "G-F", "G-N", "S1", "v1_0_G-F", "v1_0_run_pass", "run_pass", "outcome"))
                agree.append(a)
                ra, rb = g["reported"]["end_of_A"], g["reported"]["end_of_B"]
                x = {"q": q, "seed": s}
                for k in ("stage2_peak_rel_err_signed", "stage2_peak_locfree_rel_err", "stage2_peak_locfree_argmax_d",
                          "stage2_sym_err_max", "stage2_tail_max", "DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max",
                          "sigma_effort_at_0_t2"):
                    x[f"A_{k}"] = ra["final"].get(k)
                x["A_smoothed_share_peak_gap_d0"] = ra["smoothed_game"]["smoothed_share_peak_gap_d0"]
                for k in ("e1_at_0", "stage1_rel_err_signed", "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw",
                          "dFull_over_dw", "Gmax_full_t", "Gmax_full_d", "sigma_effort_at_0_t1"):
                    x[f"B_{k}"] = rb["final"].get(k)
                dc = rb["decomposition"]
                for k in ("e_tilde", "band_lo", "band_hi", "learning_rel", "learning_rel_lo", "learning_rel_hi", "inherited_rel",
                          "inherited_rel_lo", "inherited_rel_hi", "learning_contains_0", "inherited_contains_0", "band_contiguous",
                          "e1_inside_sweep"):
                    x[f"dec_{k}"] = dc[k]
                x["drift_test_pass"] = g["reported"]["drift_test_pass"]
                rep.append(x)
            else:
                r.update({"run_pass": False, "outcome": status})
            rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(agree), pd.DataFrame(rep)


def distributions(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    out = []
    for q, g in df.groupby("q"):
        for m in GATE_METRICS:
            if m not in g:
                continue
            x = g[m].dropna().astype(float)
            r = {"root": label, "q": q, "metric": m, "n": int(x.size)}
            r.update({n_: (float(x.quantile(p)) if x.size else float("nan")) for n_, p in QUANT})
            out.append(r)
    return pd.DataFrame(out)


def fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return "—"
        if float(v).is_integer() and abs(v) < 1e7:
            return str(int(v))
        return f"{v:.4g}"
    return "" if v is None else str(v)


def md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--protocol", default=str(PROTOCOL))
    p.add_argument("--seeds", type=int, nargs=2, help="first last (inclusive); default: the protocol's confirmation block")
    p.add_argument("--compare-root", help="second root (same layout) for side-by-side distributions")
    p.add_argument("--compare-seeds", type=int, nargs=2)
    a = p.parse_args()
    proto = json.load(open(a.protocol))
    seeds = list(range(a.seeds[0], a.seeds[1] + 1)) if a.seeds else list(proto["confirmation"]["seed_block"])
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, agree, rep = analyse(root, proto, seeds)
    runs.to_csv(out / "per_run.csv", index=False)
    agree.to_csv(out / "agreement.csv", index=False)
    rep.to_csv(out / "reported_metrics.csv", index=False)
    # ---- pass rule
    pc, q_pass = [], {}
    rule_k, rule_n = 18, 20
    for q in proto["q_values"]:
        g = runs[runs.q == q]
        n, k = len(g), int(g.run_pass.sum())
        lo, hi = clopper_pearson(k, n)
        applicable = n == rule_n
        q_pass[q] = bool(applicable and k >= rule_k)
        pc.append({"q": q, "n_expected": n, "n_completed": int((g.status == "completed").sum()),
                   "n_missing": int((g.status == "missing").sum()), "n_failed_exception": int((g.status == "failed_exception").sum()),
                   "n_incomplete": int((g.status == "incomplete").sum()),
                   "n_global_rng_violation": int((g.status == "global_rng_violation").sum()),
                   "n_pass": k, "pass_rate": k / n if n else float("nan"), "cp95_lo": lo, "cp95_hi": hi,
                   "rule": f">= {rule_k} of {rule_n}" if applicable else f"not applicable (n = {n} != {rule_n})",
                   "q_passes_rule": q_pass[q] if applicable else None,
                   "n_G-A_pass": int(g.get("G-A_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                   "n_G-F_pass": int(g.get("G-F_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                   "n_G-N_pass": int(g.get("G-N_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                   "n_v1_0_run_pass": int(g.get("v1_0_run_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())})
    pcd = pd.DataFrame(pc)
    pcd.to_csv(out / "pass_counts.csv", index=False)
    all_applicable = all(len(runs[runs.q == q]) == rule_n for q in proto["q_values"])
    verdict = {"root": str(root), "seeds": [seeds[0], seeds[-1]], "rule": f"each q >= {rule_k} of {rule_n} runs pass; both q",
               "per_q": {str(q): q_pass[q] for q in q_pass},
               "overall": ("PASS" if all(q_pass.values()) else "FAIL") if all_applicable else "not applicable (rule needs 20 seeds per q)",
               "n_agreement_rows": len(agree), "n_all_agree": int(agree.all_agree.sum()) if len(agree) else 0}
    json.dump(verdict, open(out / "verdict.json", "w"), indent=1)
    # ---- S1
    s1, bs = [], proto["secondary"]["bootstrap"]
    for q in proto["q_values"]:
        g = runs[(runs.q == q) & runs.status.isin(["completed", "global_rng_violation"])]
        x = g.stage1_rel_err_signed.astype(float).to_numpy()
        k = int(g.S1_pass.astype(bool).sum())
        lo, hi = clopper_pearson(k, len(x))
        rng = np.random.default_rng(int(bs["seed"]))
        bm = x[rng.integers(0, len(x), size=(int(bs["resamples"]), len(x)))].mean(axis=1) if len(x) else np.array([np.nan])
        s1.append({"q": q, "n": len(x), "S1_pass": k, "cp95_lo": lo, "cp95_hi": hi, "mean_signed": float(np.mean(x)) if len(x) else np.nan,
                   "boot95_lo": float(np.percentile(bm, 2.5)), "boot95_hi": float(np.percentile(bm, 97.5)),
                   "median_signed": float(np.median(x)) if len(x) else np.nan,
                   "sd_signed": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                   "bootstrap_resamples": int(bs["resamples"]), "bootstrap_seed": int(bs["seed"])})
    s1d = pd.DataFrame(s1)
    s1d.to_csv(out / "s1_summary.csv", index=False)
    done = runs[runs.status.isin(["completed", "global_rng_violation"])]
    dist = distributions(done, str(root))
    dist.to_csv(out / "distributions.csv", index=False)
    # ---- EXP_root vs stage-1 error
    ev, fits = [], []
    for q in proto["q_values"]:
        r_ = rep[rep.q == q] if len(rep) else rep
        if not len(r_):
            continue
        x = r_.B_stage1_rel_err_signed.astype(float).to_numpy()
        y = r_.B_EXP_root_over_dw.astype(float).to_numpy()
        for s_, xx, yy in zip(r_.seed, x, y):
            ev.append({"q": q, "seed": s_, "stage1_rel_err_signed": xx, "stage1_rel_err_sq": xx * xx, "EXP_root_over_dw": yy})
        A = np.column_stack([np.ones_like(x), x * x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef
        r2 = 1.0 - float(res @ res) / float(((y - y.mean()) ** 2).sum()) if len(y) > 1 else float("nan")
        fits.append({"q": q, "n": len(x), "intercept": float(coef[0]), "slope_on_err_sq": float(coef[1]), "r2": r2,
                     "model": "EXP_root/DW = a + b * (stage-1 rel. err)^2, ordinary least squares"})
    pd.DataFrame(ev).to_csv(out / "exp_vs_s1.csv", index=False)
    pd.DataFrame(fits).to_csv(out / "exp_vs_s1_fit.csv", index=False)
    sections = [("verdict", pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in verdict.items()}])),
                ("pass counts", pcd), ("per-run verdicts", runs), ("agreement with gates.json", agree), ("S1", s1d),
                ("distributions", dist), ("reported metrics", rep), ("EXP_root vs stage-1 error", pd.DataFrame(ev)),
                ("EXP_root vs err^2 fit", pd.DataFrame(fits))]
    if a.compare_root:
        cseeds = list(range(a.compare_seeds[0], a.compare_seeds[1] + 1)) if a.compare_seeds else seeds
        c_runs, _, _ = analyse(Path(a.compare_root), proto, cseeds)
        c_done = c_runs[c_runs.status.isin(["completed", "global_rng_violation"])]
        cmp_ = pd.concat([distributions(done, "root"), distributions(c_done, "compare_root")])
        cmp_ = cmp_.pivot_table(index=["q", "metric"], columns="root", values=[n_ for n_, _ in QUANT]).reset_index()
        cmp_.columns = [c if isinstance(c, str) else (c[0] if not c[1] else f"{c[0]}_{c[1]}") for c in cmp_.columns]
        cmp_.to_csv(out / "compare_distributions.csv", index=False)
        sections.append((f"side by side: root={root} vs compare_root={a.compare_root}", cmp_))
    (out / "tables.md").write_text("\n".join(f"### {t}\n\n{md(df)}\n" for t, df in sections))
    print(json.dumps(verdict, indent=1))
    print(pcd.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
