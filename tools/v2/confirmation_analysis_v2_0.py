#!/usr/bin/env python3
"""Pre-registered analysis of locked-protocol (v2.0) runs: verdicts, pass rule, S1/G-S, distributions.

Locked with protocol v2.0 and used UNCHANGED on the v2.0 re-rehearsal and on the confirmation. (The
v1.1 script, tools/v2/confirmation_analysis.py, stays untouched and is the analysis of protocol v1.1.)

For every expected (q, seed) under --root/q{q}/seed{s}/ the run is classified:
  missing              no run directory or no status.json
  failed_exception     status.json state 'failed' (pipeline exception; traceback head reported)
  incomplete           status.json present but not done/failed (e.g. killed) - counted as not passing
  global_rng_violation gates.json global_rng.status == 'violation' (exit code 5) - counted as failed
  completed            status done, exit code 0
Infrastructure-crash attempts moved to --root/crashed/q{q}/seed{s}_attempt*/ are listed.

Verdicts are RECOMPUTED from the saved evaluation arrays, independently of the entry point's code
(both tiers where a development value exists):
  eta_2/DW     = max(v_t2_delta) / DW                         (gateA_{final,development}.npz)
  RMSE/e2*(0)  = sqrt(mean((recovery_e2 - recovery_g2)^2 over |d| < 2q)) / recovery_g2(d=0)   (gateA_*.npz)
  tail/e2*(0)  = mean(recovery_e2 over |d| >= 2q) / recovery_g2(d=0)                          (gateA_*.npz)
  Gmax/DW      = max over t of max(v_t{t}_G) / DW              (final_{final,development}.npz)
  stage-1 err  = |v_t1_e_hat[0] - e1*| / e1*                   (final_{final,development}.npz; e1* closed form, evaluation only)
with the thresholds of the protocol JSON (inclusive <=, full float64 precision):
  G-A: eta_final, RMSE, tail;  G-F: Gmax_final;  G-N: |eta_dev - eta_final|, |Gmax_dev - Gmax_final|;
  G-S: stage-1 error (final tier) <= 0.05;
  run pass = G-A and G-F and G-N and G-S and global-RNG status ok and exit code 0;
  reported only: S1 at 0.10, the v1.1 outcome (G-A and G-F and G-N), the v1.0 outcome (G-A and Gmax_final <= 0.01 and S1 at 0.10).
Each recomputed value and verdict is compared with the run's gates.json (agreement table); also checked:
protocol version and SHA-256, and that continuation_table.npz on disk has the SHA-256 recorded in gates.json and
manifest.json and that the recorded table rule is the protocol's rule. clean_tree and the commit are recorded per run
(per_run.csv) and counted in verdict.json (they are not part of any verdict).

Per q: pass count vs the >= 18 of 20 rule (applied only when 20 seeds are expected), exact
Clopper-Pearson 95% CI of the pass rate; overall verdict = both q pass. Stage-1 error: pass counts at the G-S
threshold (0.05) and at S1 (0.10) with Clopper-Pearson CIs; mean signed stage-1 error with a 95% percentile
bootstrap CI (10000 resamples, a fresh numpy.random.default_rng(protocol seed) for each q, q ascending); median
and SD (ddof=1). Distributions (min, p10, p25, median, p75, p90, max) of every gate metric (final, dev, dev - final);
every reported metric of the v1.1 report section 4.4 incl. the stage-1 decomposition with bands; EXP_root/DW vs the
stage-1 relative error per q with a least-squares fit EXP_root/DW = a + b*err^2 (descriptive).

--rehearsal-safety: the D4 safety condition on the development-seed rehearsal (>= 19 of 20 runs under G-S and, per q,
the normal-model probability that >= 18 of 20 fresh runs pass G-S >= 0.90; the numbers are read from the protocol).
--paired-root: runs of the same (q, seed) in another root (e.g. rehearsal_v1_1), paired differences of every gate
metric and the per-gate pass/fail flips. --compare-root: side-by-side distributions of another root (unpaired;
descriptive). A paired or compare root must hold the evaluation NPZs (the in-repo light copies of the v1.1 roots do
not; use the canonical results worktree, e.g. .claude/worktrees/pilot-4-stabilization-fb99a2/results/v2_T2_locked/...).

Outputs in --out: per_run.csv, agreement.csv, pass_counts.csv, verdict.json, s1_summary.csv, distributions.csv,
reported_metrics.csv, exp_vs_s1.csv, exp_vs_s1_fit.csv, [rehearsal_safety.json], [paired_per_run.csv,
paired_summary.csv, paired_gate_flips.csv], [compare_distributions.csv], tables.md.

Usage:
  python tools/v2/confirmation_analysis_v2_0.py --root results/v2_T2_locked/confirmation_v2_0 --out <dir>
  python tools/v2/confirmation_analysis_v2_0.py --root results/v2_T2_locked/rehearsal_v2_0 --seeds 10501 10510 \
      --rehearsal-safety --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols" / "v2_T2_locked_v2_0.json"
QUANT = (("min", 0.0), ("p10", 0.10), ("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90), ("max", 1.0))
GATE_METRICS = ("eta_final", "eta_dev", "eta_dev_minus_final", "rmse", "rmse_dev", "rmse_dev_minus_final",
                "tail", "tail_dev", "tail_dev_minus_final", "gmax_final", "gmax_dev", "gmax_dev_minus_final",
                "s1", "s1_dev", "s1_dev_minus_final", "stage1_rel_err_signed")
PAIRED_METRICS = ("eta_final", "rmse", "tail", "gmax_final", "s1", "stage1_rel_err_signed", "e1_at_0",
                  "eta_dev_minus_final", "gmax_dev_minus_final")
GATE_FLAGS = ("G-A_pass", "G-F_pass", "G-N_pass", "G-S_pass", "S1_pass", "v1_1_run_pass", "run_pass")
DONE = ("completed", "global_rng_violation")


def g1_closed_form(spec: dict) -> float:
    """e1*(0) of the T=2 game (closed form, evaluation only; utils.theory_multistage.g1_two_stage)."""
    sys.path.insert(0, str(ROOT))
    from utils.theory_multistage import g1_two_stage
    return float(g1_two_stage(spec["q"], spec["w_h"], spec["w_l"], spec["k"]))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact two-sided (1 - alpha) CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recompute(d: Path, game: dict) -> dict:
    """Gate-metric values recomputed from the evaluation NPZs of one run (final and development tier)."""
    dw, q = float(game["w_h"]) - float(game["w_l"]), float(game["q"])
    g1 = g1_closed_form(game)
    out: dict = {}
    for tier, sfx in (("final", ""), ("development", "_dev")):
        z = np.load(d / f"gateA_{tier}.npz")
        out["eta_final" if not sfx else "eta_dev"] = float(z["v_t2_delta"].max()) / dw
        D, e2, g2 = z["recovery_d_grid"], z["recovery_e2"], z["recovery_g2"]
        g20 = float(g2[np.nonzero(D == 0.0)[0][0]])
        pos = np.abs(D) < 2.0 * q
        out[f"rmse{sfx}"] = float(np.sqrt(np.mean((e2[pos] - g2[pos]) ** 2))) / g20
        out[f"tail{sfx}"] = float(e2[~pos].mean()) / g20
        z = np.load(d / f"final_{tier}.npz")
        out["gmax_final" if not sfx else "gmax_dev"] = max(float(z[k].max()) for k in z.files if k.startswith("v_t") and k.endswith("_G")) / dw
        e1 = float(z["v_t1_e_hat"][0])
        out[f"s1{sfx}"] = abs((e1 - g1) / g1)
        if not sfx:
            out["stage1_rel_err_signed"] = (e1 - g1) / g1
            out["e1_at_0"] = e1
        else:
            out["stage1_rel_err_signed_dev"] = (e1 - g1) / g1
    for k in ("eta", "rmse", "tail", "gmax", "s1"):
        fin = {"eta": "eta_final", "gmax": "gmax_final"}.get(k, k)
        dev = {"eta": "eta_dev", "gmax": "gmax_dev"}.get(k, f"{k}_dev")
        out[f"{k}_dev_minus_final"] = out[dev] - out[fin]
    return out


def thresholds(proto: dict) -> dict:
    """Gate thresholds of the protocol JSON by metric name, plus the S1 threshold."""
    g = proto["gates"]
    th = {c["metric"]: float(c["threshold"]) for blk in ("G-A", "G-F", "G-N", "G-S") for c in g[blk]["all_must_hold"]}
    th["S1"] = float(proto["secondary"]["S1"]["criterion"]["threshold"])
    return th


def decide(v: dict, proto: dict) -> dict:
    """Verdicts from values (independent implementation of protocol v2.0 section 1)."""
    th = thresholds(proto)
    ga_parts = {"eta": v["eta_final"] <= th["eta_T_over_dw"], "rmse": v["rmse"] <= th["stage2_rmse_pos_over_g2_0"],
                "tail": v["tail"] <= th["stage2_tail_mean_over_g2_0"]}
    ga = all(ga_parts.values())
    gf = v["gmax_final"] <= th["Gmax_full_over_dw"]
    n_eta = abs(v["eta_dev"] - v["eta_final"]) <= th["eta_T_over_dw_dev_minus_final_abs"]
    n_gmax = abs(v["gmax_dev"] - v["gmax_final"]) <= th["Gmax_full_over_dw_dev_minus_final_abs"]
    gn = n_eta and n_gmax
    gs = v["s1"] <= th["stage1_rel_err_abs"]
    s1 = v["s1"] <= th["S1"]
    return {"G-A": bool(ga), "G-A_eta": bool(ga_parts["eta"]), "G-A_rmse": bool(ga_parts["rmse"]), "G-A_tail": bool(ga_parts["tail"]),
            "G-F": bool(gf), "G-N": bool(gn), "G-N_eta": bool(n_eta), "G-N_gmax": bool(n_gmax), "G-S": bool(gs),
            "gates_pass": bool(ga and gf and gn and gs), "S1": bool(s1), "v1_1_run_pass": bool(ga and gf and gn),
            "G-F_v1_0": bool(gf and s1), "run_pass_v1_0": bool(ga and gf and s1)}


def classify(d: Path):
    """(status, exit code, info) of one run directory."""
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


def outcome_of(dec: dict, status: str) -> str:
    """Outcome label (same order as the entry point)."""
    if status != "completed":
        return "global_rng_violation" if status == "global_rng_violation" else status
    return ("pass" if dec["gates_pass"] else "stage2_failure" if not dec["G-A"] else "fail_G-F" if not dec["G-F"]
            else "fail_G-N" if not dec["G-N"] else "fail_G-S")


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Columns the later tables read exist even when no run completed (NaN / False)."""
    for c in GATE_METRICS + ("e1_at_0",):
        if c not in df:
            df[c] = np.nan
    for c in GATE_FLAGS + ("G-A_eta_pass", "G-A_rmse_pass", "G-A_tail_pass", "G-N_eta_pass", "G-N_gmax_pass", "v1_0_run_pass"):
        if c not in df:
            df[c] = False
    return df


def _rule_ok(rec_gates: dict, rec_manifest: dict, proto: dict, q: int) -> bool:
    """The table rule recorded in gates.json and manifest.json agree with each other and with the protocol's rule."""
    if not rec_gates or rec_gates != rec_manifest:
        return False
    r = proto["pipeline"]["continuation_table"]["rule"]
    return bool(rec_gates.get("panel_width") == r["panel_width"] and rec_gates.get("nodes_per_panel") == r["nodes_per_panel"]
                and rec_gates.get("step_requested") == r["y_grid"]["step"] and rec_gates.get("n_y") == r["y_grid"]["n_nodes"]
                and rec_gates.get("n_panels") == r["n_panels"][str(q)] and rec_gates.get("n_nodes") == r["n_nodes"][str(q)]
                and rec_gates.get("stage") == proto["pipeline"]["continuation_table"]["stage"])


def analyse(root: Path, proto: dict, seeds, v20: bool = True) -> tuple:
    """Per-run table, agreement table and reported-metrics table for every expected (q, seed)."""
    rows, agree, rep = [], [], []
    proto_sha = sha256_file(PROTOCOL) if v20 else None
    for q in proto["q_values"]:
        game = proto["records"][str(q)]["game"]
        for s in seeds:
            d = root / f"q{q}" / f"seed{s}"
            status, code, info = classify(d)
            crashed = sorted(os.path.relpath(p, root) for p in glob.glob(str(root / "crashed" / f"q{q}" / f"seed{s}_attempt*")))
            r = {"q": q, "seed": s, "status": status, "exit_code": code, "status_info": json.dumps(info) if info else "",
                 "crashed_attempts": " ".join(crashed)}
            if status in DONE:
                g = json.load(open(d / "gates.json"))
                summ = json.load(open(d / "v2_run_summary.json"))
                v = recompute(d, game)
                dec = decide(v, proto)
                r.update({k: v[k] for k in GATE_METRICS})
                r["e1_at_0"] = v["e1_at_0"]
                r.update({f"{k}_pass": dec[k] for k in ("G-A", "G-A_eta", "G-A_rmse", "G-A_tail", "G-F", "G-N", "G-N_eta", "G-N_gmax", "G-S", "S1")})
                r["global_rng"] = g["global_rng"]["status"]
                r["run_pass"] = bool(dec["gates_pass"] and status == "completed" and code == 0)
                r["v1_1_run_pass"] = dec["v1_1_run_pass"]
                r["v1_0_G-F"] = dec["G-F_v1_0"]
                r["v1_0_run_pass"] = dec["run_pass_v1_0"]
                r["wall_sec"] = summ.get("total_wall_sec")
                r["outcome"] = outcome_of(dec, status)
                tb = (g.get("continuation_table") or {})
                r["table_sha256"] = tb.get("npz_sha256")
                r["table_build_sec"] = tb.get("build_seconds")
                mv = g["metric_values"]
                a = {"q": q, "seed": s}
                for k in ("eta_final", "eta_dev", "rmse", "tail", "gmax_final", "gmax_dev", "s1"):
                    a[f"absdiff_{k}"] = abs(float(mv[k]) - v[k])
                dv = g["dev_tier_values"]
                a["absdiff_rmse_dev"] = abs(float(dv["G-A"]["stage2_rmse_pos_over_g2_0"]) - v["rmse_dev"])
                a["absdiff_tail_dev"] = abs(float(dv["G-A"]["stage2_tail_mean_over_g2_0"]) - v["tail_dev"])
                a["absdiff_s1_dev"] = abs(float(dv["S1"]["stage1_rel_err_abs"]) - v["s1_dev"])
                chk = {"G-A": dec["G-A"] == g["G-A"]["pass"], "G-F": dec["G-F"] == g["G-F"]["pass"], "G-N": dec["G-N"] == g["G-N"]["pass"],
                       "S1": dec["S1"] == g["S1"]["pass"],
                       "v1_0_G-F": dec["G-F_v1_0"] == g["v1_0_outcome"]["G-F_v1_0"],
                       "v1_0_run_pass": dec["run_pass_v1_0"] == g["v1_0_outcome"]["run_pass_v1_0"]}
                if v20:
                    npz = d / "continuation_table.npz"
                    sha_disk = sha256_file(npz) if npz.exists() else None
                    man = json.load(open(d / "manifest.json"))
                    r["table_sha256_disk"] = sha_disk
                    r["clean_tree"] = bool(g.get("clean_tree") is True and man.get("clean_tree") is True)
                    r["commit"] = g.get("commit")
                    chk.update({"G-S": dec["G-S"] == g["G-S"]["pass"], "v1_1_run_pass": dec["v1_1_run_pass"] == g["v1_1_outcome"]["run_pass_v1_1"],
                                "run_pass": r["run_pass"] == g["run_pass"], "outcome": r["outcome"] == g["outcome"],
                                "protocol_version": g.get("protocol_version") == proto["version"] and man.get("protocol_version") == proto["version"],
                                "protocol_sha256": g.get("protocol_sha256") == proto_sha and man["locked_protocol"]["sha256"] == proto_sha,
                                "table_sha256": bool(sha_disk is not None and sha_disk == tb.get("npz_sha256")
                                                     and sha_disk == (man.get("continuation_table") or {}).get("npz_sha256")),
                                "table_rule": bool(_rule_ok(tb.get("rule"), (man.get("continuation_table") or {}).get("rule"), proto, q))})
                a.update(chk)
                a["all_agree"] = all(chk.values())
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
                r.update({"run_pass": False, "v1_1_run_pass": False, "outcome": status})
            rows.append(r)
    return ensure_columns(pd.DataFrame(rows)), pd.DataFrame(agree), pd.DataFrame(rep)


def distributions(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """min, p10, p25, median, p75, p90, max of every gate metric per q."""
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


def normal_model(x: np.ndarray, thr: float, n: int = 20, k: int = 18) -> dict:
    """Normal-model pass probability: signed error ~ N(mean, SD ddof=1); run passes iff |error| <= thr;
    P(Binomial(n, p) >= k) for n fresh runs."""
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    p = float(norm.cdf(thr, m, s) - norm.cdf(-thr, m, s))
    return {"n": int(x.size), "mean_signed": m, "sd_ddof1": s, "per_run_pass_prob": p, "P_ge_k_of_n": float(binom.sf(k - 1, n, p))}


def rehearsal_safety(runs: pd.DataFrame, proto: dict, seeds) -> dict:
    """Decision D4 safety condition on the development-seed rehearsal."""
    cond = proto["confirmation"]["rehearsal"]["safety_condition"]
    thr = thresholds(proto)["stage1_rel_err_abs"]
    need_k, need_p = int(cond["G-S_pass_at_least_of_20"]), float(cond["normal_model_probability_at_least_per_q"])
    n_expected = len(proto["q_values"]) * len(list(seeds))
    done = runs[runs.status.isin(DONE)]
    k_all = int(done["G-S_pass"].astype(bool).sum()) if len(done) else 0
    per_q = {}
    for q in proto["q_values"]:
        d = done[done.q == q]
        x = d.stage1_rel_err_signed.astype(float).to_numpy()
        nan = float("nan")
        nm = (normal_model(x, thr) if len(x) > 1 else
              {"n": int(len(x)), "mean_signed": nan, "sd_ddof1": nan, "per_run_pass_prob": nan, "P_ge_k_of_n": nan})
        per_q[str(q)] = {"n_runs": int(len(d)), "n_G-S_pass": int(d["G-S_pass"].astype(bool).sum()), **nm,
                         "meets_probability": bool(nm["P_ge_k_of_n"] >= need_p)}
    ok = bool(len(done) == n_expected and n_expected == 20 and k_all >= need_k and all(v["meets_probability"] for v in per_q.values()))
    return {"condition": cond, "G-S_threshold": thr, "n_expected": n_expected, "n_completed": int(len(done)),
            "pooled_n_G-S_pass": k_all, "pooled_needed": need_k, "per_q": per_q, "pass": ok}


def fmt(v) -> str:
    """Compact cell text."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return "—"
        if float(v).is_integer() and abs(v) < 1e7:
            return str(int(v))
        return f"{v:.6g}"
    return "" if v is None else str(v)


def md(df: pd.DataFrame) -> str:
    """Markdown table of a DataFrame."""
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def paired_tables(new: pd.DataFrame, ref: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Paired (q, seed) differences new - ref of every gate metric, their per-q summary and the pass/fail flips (descriptive)."""
    a, b = new[new.status.isin(DONE)], ref[ref.status.isin(DONE)]
    m = a.merge(b, on=["q", "seed"], suffixes=("_new", "_ref"))
    if m.empty:
        return (pd.DataFrame(columns=["q", "seed"]), pd.DataFrame(columns=["q", "metric", "n_pairs"]),
                pd.DataFrame(columns=["q", "flag", "n_pairs"]))
    rows = []
    for _, r in m.iterrows():
        row = {"q": int(r["q"]), "seed": int(r["seed"])}
        for k in PAIRED_METRICS:
            row[f"{k}_ref"], row[f"{k}_new"] = float(r[f"{k}_ref"]), float(r[f"{k}_new"])
            row[f"{k}_diff"] = row[f"{k}_new"] - row[f"{k}_ref"]
        rows.append(row)
    per_run = pd.DataFrame(rows)
    summ = []
    for q, g in per_run.groupby("q"):
        for k in PAIRED_METRICS:
            d = g[f"{k}_diff"].to_numpy()
            summ.append({"q": int(q), "metric": k, "n_pairs": int(len(d)), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                         "n_new_lower": int((d < 0).sum()), "n_new_higher": int((d > 0).sum()), "n_equal": int((d == 0).sum()),
                         "mean_ref": float(g[f"{k}_ref"].mean()), "mean_new": float(g[f"{k}_new"].mean()),
                         "sd_ref_ddof1": float(g[f"{k}_ref"].std(ddof=1)) if len(g) > 1 else float("nan"),
                         "sd_new_ddof1": float(g[f"{k}_new"].std(ddof=1)) if len(g) > 1 else float("nan")})
    flips = []
    for q, g in m.groupby("q"):
        for k in GATE_FLAGS:
            r_, n_ = g[f"{k}_ref"].astype(bool), g[f"{k}_new"].astype(bool)
            flips.append({"q": int(q), "flag": k, "n_pairs": int(len(g)), "n_ref_pass": int(r_.sum()), "n_new_pass": int(n_.sum()),
                          "n_ref_pass_new_fail": int((r_ & ~n_).sum()), "n_ref_fail_new_pass": int((~r_ & n_).sum())})
    return per_run, pd.DataFrame(summ), pd.DataFrame(flips)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--protocol", default=str(PROTOCOL))
    p.add_argument("--seeds", type=int, nargs=2, help="first last (inclusive); default: the protocol's confirmation block")
    p.add_argument("--rehearsal-safety", action="store_true", help="evaluate the D4 safety condition (run on the rehearsal root)")
    p.add_argument("--paired-root", help="root of the same (q, seed) runs under another protocol, for paired differences")
    p.add_argument("--paired-label", default="paired_root")
    p.add_argument("--compare-root", help="second root (same layout) for side-by-side distributions (unpaired)")
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

        def cnt(col: str) -> int:
            return int(g.get(col, pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        pc.append({"q": q, "n_expected": n, "n_completed": int((g.status == "completed").sum()),
                   "n_missing": int((g.status == "missing").sum()), "n_failed_exception": int((g.status == "failed_exception").sum()),
                   "n_incomplete": int((g.status == "incomplete").sum()),
                   "n_global_rng_violation": int((g.status == "global_rng_violation").sum()),
                   "n_pass": k, "pass_rate": k / n if n else float("nan"), "cp95_lo": lo, "cp95_hi": hi,
                   "rule": f">= {rule_k} of {rule_n}" if applicable else f"not applicable (n = {n} != {rule_n})",
                   "q_passes_rule": q_pass[q] if applicable else None,
                   "n_G-A_pass": cnt("G-A_pass"), "n_G-F_pass": cnt("G-F_pass"), "n_G-N_pass": cnt("G-N_pass"), "n_G-S_pass": cnt("G-S_pass"),
                   "n_S1_0.10_pass": cnt("S1_pass"), "n_v1_1_run_pass": cnt("v1_1_run_pass"), "n_v1_0_run_pass": cnt("v1_0_run_pass")})
    pcd = pd.DataFrame(pc)
    pcd.to_csv(out / "pass_counts.csv", index=False)
    all_applicable = all(len(runs[runs.q == q]) == rule_n for q in proto["q_values"])
    verdict = {"root": str(root), "seeds": [seeds[0], seeds[-1]], "rule": f"each q >= {rule_k} of {rule_n} runs pass; both q",
               "per_q": {str(q): q_pass[q] for q in q_pass},
               "overall": ("PASS" if all(q_pass.values()) else "FAIL") if all_applicable else "not applicable (rule needs 20 seeds per q)",
               "n_agreement_rows": len(agree), "n_all_agree": int(agree.all_agree.sum()) if len(agree) else 0,
               "n_expected": int(len(runs)), "n_completed": int((runs.status == "completed").sum()),
               "n_missing": int((runs.status == "missing").sum()), "n_failed_exception": int((runs.status == "failed_exception").sum()),
               "n_incomplete": int((runs.status == "incomplete").sum()),
               "n_global_rng_violation": int((runs.status == "global_rng_violation").sum()),
               "n_clean_tree": int(runs["clean_tree"].fillna(False).astype(bool).sum()) if "clean_tree" in runs else 0,
               "commits": sorted({c for c in runs["commit"].dropna().unique()}) if "commit" in runs else []}
    # ---- stage-1 error: G-S threshold and S1 threshold
    th = thresholds(proto)
    s1, bs = [], proto["secondary"]["bootstrap"]
    for q in proto["q_values"]:
        g = runs[(runs.q == q) & runs.status.isin(DONE)]
        x = g.stage1_rel_err_signed.astype(float).to_numpy()
        rng = np.random.default_rng(int(bs["seed"]))
        bm = x[rng.integers(0, len(x), size=(int(bs["resamples"]), len(x)))].mean(axis=1) if len(x) else np.array([np.nan])
        for label, thr in (("G-S", th["stage1_rel_err_abs"]), ("S1", th["S1"])):
            k = int((np.abs(x) <= thr).sum())
            lo, hi = clopper_pearson(k, len(x))
            s1.append({"q": q, "criterion": label, "threshold": thr, "n": len(x), "n_pass": k, "cp95_lo": lo, "cp95_hi": hi,
                       "mean_signed": float(np.mean(x)) if len(x) else np.nan,
                       "boot95_lo": float(np.percentile(bm, 2.5)), "boot95_hi": float(np.percentile(bm, 97.5)),
                       "median_signed": float(np.median(x)) if len(x) else np.nan,
                       "sd_signed": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                       "bootstrap_resamples": int(bs["resamples"]), "bootstrap_seed": int(bs["seed"])})
    s1d = pd.DataFrame(s1)
    s1d.to_csv(out / "s1_summary.csv", index=False)
    done = runs[runs.status.isin(DONE)]
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
    sections = [("pass counts", pcd), ("per-run verdicts", runs), ("agreement with gates.json", agree),
                ("stage-1 error: G-S (0.05) and S1 (0.10)", s1d), ("distributions", dist), ("reported metrics", rep),
                ("EXP_root vs stage-1 error", pd.DataFrame(ev)), ("EXP_root vs err^2 fit", pd.DataFrame(fits))]
    if a.rehearsal_safety:
        rs = rehearsal_safety(runs, proto, seeds)
        json.dump(rs, open(out / "rehearsal_safety.json", "w"), indent=1)
        verdict["rehearsal_safety"] = {"pass": rs["pass"], "pooled_n_G-S_pass": rs["pooled_n_G-S_pass"],
                                       "per_q_P": {q: v["P_ge_k_of_n"] for q, v in rs["per_q"].items()}}
        sections.insert(0, ("rehearsal safety condition (D4)", pd.DataFrame(
            [{"q": q, **{k: v[k] for k in ("n_runs", "n_G-S_pass", "mean_signed", "sd_ddof1", "per_run_pass_prob", "P_ge_k_of_n", "meets_probability")}}
             for q, v in rs["per_q"].items()])))
    if a.paired_root:
        ref_runs, _, _ = analyse(Path(a.paired_root), proto, seeds, v20=False)
        pr, ps, pf = paired_tables(runs, ref_runs)
        pr.to_csv(out / "paired_per_run.csv", index=False)
        ps.to_csv(out / "paired_summary.csv", index=False)
        pf.to_csv(out / "paired_gate_flips.csv", index=False)
        verdict["paired_root"] = {"root": a.paired_root, "label": a.paired_label, "n_pairs": int(len(pr))}
        if not len(pr):
            print(f"WARNING: no (q, seed) of {a.paired_root} completed with NPZs: the paired tables are empty")
        sections.append((f"paired by (q, seed): this root - {a.paired_label}", ps))
        sections.append((f"pass/fail flips by (q, seed): this root vs {a.paired_label}", pf))
    if a.compare_root:
        cseeds = list(range(a.compare_seeds[0], a.compare_seeds[1] + 1)) if a.compare_seeds else seeds
        c_runs, _, _ = analyse(Path(a.compare_root), proto, cseeds, v20=False)
        c_done = c_runs[c_runs.status.isin(DONE)]
        verdict["compare_root"] = {"root": a.compare_root, "n_completed": int(len(c_done))}
        if not len(c_done):
            print(f"WARNING: no run of {a.compare_root} completed with NPZs: the side-by-side table has one side only")
        cmp_ = pd.concat([distributions(done, "root"), distributions(c_done, "compare_root")])
        cmp_ = cmp_.pivot_table(index=["q", "metric"], columns="root", values=["n"] + [n_ for n_, _ in QUANT]).reset_index()
        cmp_.columns = [c if isinstance(c, str) else (c[0] if not c[1] else f"{c[0]}_{c[1]}") for c in cmp_.columns]
        cmp_.to_csv(out / "compare_distributions.csv", index=False)
        sections.append((f"side by side (unpaired): root={root} vs compare_root={a.compare_root}", cmp_))
    json.dump(verdict, open(out / "verdict.json", "w"), indent=1)
    sections.insert(0, ("verdict", pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in verdict.items()}])))
    (out / "tables.md").write_text("\n".join(f"### {t}\n\n{md(df)}\n" for t, df in sections))
    print(json.dumps(verdict, indent=1))
    print(pcd.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
