"""Extraction for F2 (T = 2 verifier calibration, final tier): writes ``data/F2.csv``.

One evaluation function is used for every policy that is evaluated here:
``utils.v2_metrics.evaluate(policy, spec, utils.dp_br_verifier.FINAL_CONFIG).scalars`` (the call of
``tools/v2/locked_calibration.py``).  The policies are implemented in this file from
``utils.theory_multistage.g1_two_stage`` / ``g2_two_stage``.

Step 1 (stop rule): the closed-form equilibrium and the zero-effort policy are re-evaluated at both q and
compared with the stored final-tier rows of ``results/v2_T2_locked/calibration/calibration_locked.csv``
(<= 1e-12 in units of DeltaW; Gmax_full_t, Gmax_full_d and valid must be equal).  Any miss stops the script
before ``data/F2.csv`` is written.
Step 2: the three misspecified policies M1-M3 are new diagnostic evaluations (valid must be True).
The frozen v2.0 policies are the 40 ``v20`` rows of ``results/t3_r1/eval/per_run.csv`` (stored values; seed
30510 at q = 50 is the only failing run; pass counts 19/20 and 20/20 are checked).

Usage: .venv/bin/python paper/multistage_rev/scripts/extract_F2.py   (any cwd)
"""

from __future__ import annotations

import csv
import importlib.util
import json
from typing import Callable, Dict, List, Optional, Tuple

import _common as C

import numpy as np

from envs.curriculum_env import GameSpec
from utils.dp_br_verifier import FINAL_CONFIG
from utils.theory_multistage import g1_two_stage, g2_two_stage
from utils.v2_metrics import evaluate

Policy = Callable[[int, np.ndarray], np.ndarray]

FLOOR = 1e-8                       # values below this are drawn at the floor
TOL = 1e-12                        # stop rule of Step 1, in units of DeltaW
FAILED_SEED, FAILED_Q = 30510, 50
PROTOCOL = C.ROOT / "protocols" / "v2_T2_locked_v2_0.json"
OUT_CSV = C.OUT / "data" / "F2.csv"
CALIB_REL = "results/v2_T2_locked/calibration/calibration_locked.csv"
PER_RUN_REL = "results/t3_r1/eval/per_run.csv"
NEW_SRC = "new diagnostic evaluation by scripts/extract_F2.py"

FLOAT_COLS = ("Gmax_full_over_dw", "G_max_t1_over_dw", "G_max_t2_over_dw", "eta_T_over_dw",
              "EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw")
EXACT_COLS = ("Gmax_full_t", "Gmax_full_d", "valid")
# plotted quantity -> column of the verifier scalars / of the stored CSVs
METRICS = (("eta_2_over_dw", "eta_T_over_dw"), ("Gmax_full_over_dw", "Gmax_full_over_dw"),
           ("EXP_root_over_dw", "EXP_root_over_dw"))
PT_EQ, PT_ZERO, PT_V20 = "closed-form equilibrium", "zero effort", "frozen v2.0 policies"
PT_M1 = "M1: 0.8 x closed-form equilibrium"
PT_M2 = "M2: 1.2 x closed-form equilibrium"
PT_M3 = "M3: stage 2 constant at g2(0)"
ORDER = (PT_EQ, PT_ZERO, PT_V20, PT_M1, PT_M2, PT_M3)


def stop(msg: str) -> None:
    """Print a STOP message and leave without writing any output."""
    print("STOP: " + msg)
    raise SystemExit(2)


def make_spec(q: int) -> GameSpec:
    """The as-run T = 2 game for one q (w_H = 6, w_L = 2, k = 1/3500, effort range [0, 100])."""
    return GameSpec(w_h=6.0, w_l=2.0, k=1.0 / 3500.0, q=float(q), T=2, e_min=0.0, e_max=100.0)


def eq_policy(spec: GameSpec, scale: float = 1.0) -> Policy:
    """Closed-form equilibrium (stage 1 constant g1, stage 2 g2_two_stage), times ``scale``, clipped."""
    g1 = g1_two_stage(spec.q, spec.w_h, spec.w_l, spec.k)

    def pol(t: int, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        if t == 1:
            e = np.full(d.shape, g1)
        else:
            e = g2_two_stage(d, spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max)
        return np.clip(scale * e, spec.e_min, spec.e_max)
    return pol


def zero_policy(spec: GameSpec) -> Policy:
    """Zero effort at every stage and state."""
    def pol(t: int, d: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(d, dtype=float).shape)
    return pol


def m3_policy(spec: GameSpec) -> Policy:
    """Stage 2 constant at g2(0) for every d, stage 1 at g1, clipped to the effort range."""
    g1 = g1_two_stage(spec.q, spec.w_h, spec.w_l, spec.k)
    g20 = float(g2_two_stage(0.0, spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max))

    def pol(t: int, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        return np.clip(np.full(d.shape, g1 if t == 1 else g20), spec.e_min, spec.e_max)
    return pol


def evaluate_final(policy: Policy, spec: GameSpec) -> Dict[str, object]:
    """The one evaluation function of F2: final-tier verifier scalars of a policy."""
    return evaluate(policy, spec, FINAL_CONFIG).scalars


def load_common_for_comparison():
    """Import tools/v2/common.py by path (read-only; used only to compare constants and policies)."""
    path = C.ROOT / "tools" / "v2" / "common.py"
    spec = importlib.util.spec_from_file_location("v2_common_for_comparison", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_constants(checks: List[str]) -> None:
    """The game constants, the final-tier settings and the policies equal tools/v2/common.py's."""
    cm = load_common_for_comparison()
    if not (cm.W_H == 6.0 and cm.W_L == 2.0 and cm.K == 1.0 / 3500.0):
        stop(f"tools/v2/common.py constants W_H={cm.W_H} W_L={cm.W_L} K={cm.K} differ from 6, 2, 1/3500")
    for q in C.QS:
        if cm.spec_for(q) != make_spec(q):
            stop(f"spec_for({q}) of tools/v2/common.py differs from the spec used here")
    if (FINAL_CONFIG.name, FINAL_CONFIG.state_step, FINAL_CONFIG.effort_step, FINAL_CONFIG.gl_half) != (
            "final", 2.0, 0.5, 32):
        stop("FINAL_CONFIG is not (state step 2, effort step 0.5, 32 nodes per half)")
    checks.append("constants: W_H, W_L, K and spec_for(q) of tools/v2/common.py equal the values used here "
                  "(q = 50, 60); FINAL_CONFIG = (state step 2.0, effort step 0.5, 32 nodes per half)")
    grid = np.arange(-400.0, 400.0 + 0.25, 0.25)
    for q in C.QS:
        spec = make_spec(q)
        for name, mine, theirs in (("closed-form equilibrium", eq_policy(spec), cm.analytic_policy(spec)),
                                   ("zero effort", zero_policy(spec), cm.zero_policy)):
            for t in (1, 2):
                if not np.array_equal(np.asarray(mine(t, grid)), np.asarray(theirs(t, grid))):
                    stop(f"{name} policy differs from tools/v2/common.py at stage {t}, q={q}")
    checks.append("policies: own closed-form equilibrium and zero-effort callables are bit-equal to "
                  "tools/v2/common.py's on d in [-400, 400] step 0.25, stages 1 and 2, q = 50, 60 "
                  "(comparison only; common.py's policies are not used for any value)")


def step1(checks: List[str]) -> Tuple[Dict[Tuple[int, str], Dict[str, object]], Dict[Tuple[int, str], float]]:
    """Re-evaluate analytic_eq and zero at both q and compare with the stored final-tier rows."""
    stored = C.load_per_run(C.CALIB)
    stored = stored[stored["tier"] == "final"]
    rows: Dict[Tuple[int, str], Dict[str, object]] = {}
    diffs: Dict[Tuple[int, str], float] = {}
    for q in C.QS:
        spec = make_spec(q)
        for name, pol in (("analytic_eq", eq_policy(spec)), ("zero", zero_policy(spec))):
            sel = stored[(stored["q"] == q) & (stored["policy"] == name)]
            if len(sel) != 1:
                stop(f"{len(sel)} stored final-tier rows for q={q}, policy={name}")
            st = sel.iloc[0]
            sc = evaluate_final(pol, spec)
            d = {c: abs(float(sc[c]) - float(st[c])) for c in FLOAT_COLS}
            bit = all(float(sc[c]) == float(st[c]) for c in FLOAT_COLS)
            exact = {c: (sc[c] == st[c]) for c in EXACT_COLS}
            worst = max(d.values())
            checks.append(f"step 1 q={q} {name}: max |re-evaluation - stored| over the 8 DeltaW columns = "
                          f"{worst!r}; bit-equal on all 8 = {bit}; Gmax_full_t/Gmax_full_d/valid equal = "
                          f"{exact['Gmax_full_t']}/{exact['Gmax_full_d']}/{exact['valid']}")
            if worst > TOL or not all(exact.values()):
                stop(f"re-evaluation of {name} at q={q} differs from the stored record: max diff {worst!r}, "
                     f"exact columns {exact}")
            rows[(q, name)] = {"stored": st, "reeval": sc}
            diffs[(q, name)] = worst
    return rows, diffs


def step2(checks: List[str]) -> Dict[Tuple[int, str], Dict[str, object]]:
    """The three misspecified policies at both q (new diagnostic evaluations)."""
    out: Dict[Tuple[int, str], Dict[str, object]] = {}
    for q in C.QS:
        spec = make_spec(q)
        for name, pol in ((PT_M1, eq_policy(spec, 0.8)), (PT_M2, eq_policy(spec, 1.2)), (PT_M3, m3_policy(spec))):
            sc = evaluate_final(pol, spec)
            if sc["valid"] is not True:
                stop(f"{name} at q={q}: valid = {sc['valid']!r} ({sc['invalid_reasons']})")
            out[(q, name)] = sc
            checks.append(f"step 2 q={q} {name[:2]}: valid = True; eta_2/DW = {float(sc['eta_T_over_dw'])!r}, "
                          f"Gmax_full/DW = {float(sc['Gmax_full_over_dw'])!r} (t={sc['Gmax_full_t']}, "
                          f"d={sc['Gmax_full_d']}), EXP_root/DW = {float(sc['EXP_root_over_dw'])!r}")
    return out


def v20_rows(checks: List[str]):
    """The 40 v20 rows of per_run.csv, with the pass counts of the v2.0 gates checked."""
    df = C.load_per_run(C.PER_RUN_R1)
    v = df[df["arm"] == "v20"].sort_values(["q", "seed"]).reset_index(drop=True)
    gates = C.read_json(PROTOCOL)["gates"]
    thr = {m["metric"]: m["threshold"] for g in ("G-A", "G-F", "G-N", "G-S") for m in gates[g]["all_must_hold"]}
    counts: Dict[int, int] = {}
    for q in C.QS:
        vq = v[v["q"] == q]
        if list(vq["seed"]) != list(range(30501, 30521)):
            stop(f"v20 rows at q={q} are not seeds 30501-30520")
        n_pass = 0
        for _, r in vq.iterrows():
            g_a = (r["eta_T_over_dw"] <= thr["eta_T_over_dw"]
                   and r["rmse_pos_T_over_eT0"] <= thr["stage2_rmse_pos_over_g2_0"]
                   and r["tail_mean_T_over_eT0"] <= thr["stage2_tail_mean_over_g2_0"])
            g_f = r["Gmax_full_over_dw"] <= thr["Gmax_full_over_dw"]
            g_n = (abs(r["dev_eta_T_over_dw"] - r["eta_T_over_dw"]) <= thr["eta_T_over_dw_dev_minus_final_abs"]
                   and abs(r["dev_Gmax_full_over_dw"] - r["Gmax_full_over_dw"])
                   <= thr["Gmax_full_over_dw_dev_minus_final_abs"])
            g_s = r["stage1_rel_err_abs"] <= thr["stage1_rel_err_abs"]
            passed = bool(g_a and g_f and g_n and g_s)
            stored = C.read_json(C.v20_run_dir(q, int(r["seed"])) / "gates.json")
            if passed != bool(stored["run_pass"]):
                stop(f"v20 q={q} seed {int(r['seed'])}: pass from per_run.csv ({passed}) differs from "
                     f"gates.json run_pass ({stored['run_pass']})")
            if not (r["valid"] and stored["metric_values"]["eta_final"] == r["eta_T_over_dw"]
                    and stored["metric_values"]["gmax_final"] == r["Gmax_full_over_dw"]):
                stop(f"v20 q={q} seed {int(r['seed'])}: per_run.csv differs from gates.json metric_values")
            n_pass += passed
            if (q, int(r["seed"])) == (FAILED_Q, FAILED_SEED):
                failing = [g for g, ok in (("G-A", g_a), ("G-F", g_f), ("G-N", g_n), ("G-S", g_s)) if not ok]
                checks.append(f"v20 q=50 seed 30510 failing gate(s) {failing}: eta_2/DW = "
                              f"{float(r['eta_T_over_dw'])!r} > {thr['eta_T_over_dw']}")
        counts[q] = n_pass
    if counts != {50: 19, 60: 20}:
        stop(f"v2.0 pass counts {counts}, expected 19/20 at q=50 and 20/20 at q=60")
    checks.append("v20 pass counts: q=50 19/20, q=60 20/20 (run pass recomputed from per_run.csv with the "
                  "protocol thresholds and equal to gates.json run_pass for all 40 runs; eta_T and Gmax_full of "
                  "per_run.csv bit-equal to gates.json metric_values eta_final/gmax_final for all 40 runs)")
    return v


def make_row(q, policy_type: str, kind: str, metric: str, value: float, seed: Optional[int], source: str,
             valid: Optional[bool], reeval_abs_diff: Optional[float] = None,
             reeval_value: Optional[float] = None) -> Dict[str, str]:
    """One long-format row of data/F2.csv (floats as round-trip text)."""
    at_floor = bool(float(value) < FLOOR)
    plotted = FLOOR if at_floor else float(value)
    marked = bool(seed == FAILED_SEED and q == FAILED_Q)
    return {
        "q": str(q), "policy_type": policy_type,
        "x_order": str(ORDER.index(policy_type)) if policy_type in ORDER else "",
        "kind": kind, "metric": metric, "seed": "" if seed is None else str(seed),
        "value": C.fmt(value), "plotted_value": C.fmt(plotted), "at_floor": str(at_floor),
        "marked": str(marked), "valid": "" if valid is None else str(bool(valid)), "source": source,
        "reeval_abs_diff": "" if reeval_abs_diff is None else C.fmt(reeval_abs_diff),
        "reeval_value": "" if reeval_value is None else C.fmt(reeval_value),
    }


def main() -> int:
    """Run Step 1 and Step 2, assemble the rows and write data/F2.csv."""
    C.single_thread()
    checks: List[str] = []
    check_constants(checks)
    stored_eval, diffs = step1(checks)
    new_eval = step2(checks)
    v20 = v20_rows(checks)

    rows: List[Dict[str, str]] = []
    for q in C.QS:
        for pt, name in ((PT_EQ, "analytic_eq"), (PT_ZERO, "zero")):
            st, re_ = stored_eval[(q, name)]["stored"], stored_eval[(q, name)]["reeval"]
            for metric, col in METRICS:
                rows.append(make_row(
                    q, pt, "stored record", metric, float(st[col]), None,
                    f"{CALIB_REL} column {col} (policy {name}, tier final, q={q})", bool(st["valid"]),
                    reeval_abs_diff=diffs[(q, name)], reeval_value=float(re_[col])))
        vq = v20[v20["q"] == q]
        for _, r in vq.iterrows():
            for metric, col in METRICS:
                rows.append(make_row(q, PT_V20, "stored record", metric, float(r[col]), int(r["seed"]),
                                     f"{PER_RUN_REL} column {col} (arm v20, q={q}, seed {int(r['seed'])})",
                                     bool(r["valid"])))
        for pt in (PT_M1, PT_M2, PT_M3):
            sc = new_eval[(q, pt)]
            for metric, col in METRICS:
                rows.append(make_row(q, pt, "new diagnostic evaluation", metric, float(sc[col]), None,
                                     f"{NEW_SRC} (evaluate(policy, spec, FINAL_CONFIG).scalars['{col}'], "
                                     f"q={q})", bool(sc["valid"])))
    gate_src = "protocols/v2_T2_locked_v2_0.json gates.%s.all_must_hold[metric=%s].threshold"
    gates = C.read_json(PROTOCOL)["gates"]
    eta_thr = [m["threshold"] for m in gates["G-A"]["all_must_hold"] if m["metric"] == "eta_T_over_dw"][0]
    g_thr = [m["threshold"] for m in gates["G-F"]["all_must_hold"] if m["metric"] == "Gmax_full_over_dw"][0]
    for q in C.QS:
        rows.append(make_row(q, "", "gate limit", "eta_2_over_dw", eta_thr, None,
                             gate_src % ("G-A", "eta_T_over_dw"), None))
        rows.append(make_row(q, "", "gate limit", "Gmax_full_over_dw", g_thr, None,
                             gate_src % ("G-F", "Gmax_full_over_dw"), None))
        rows.append(make_row(q, "", "plot floor", "", FLOOR, None,
                             "scripts/extract_F2.py FLOOR (values below it are drawn at it)", None))
    # the floor and the gate limits are constants, not data: they must not be flagged as floor values
    for r in rows:
        if r["kind"] in ("gate limit", "plot floor"):
            r["at_floor"], r["plotted_value"] = "False", r["value"]
    # all sources of the numbers must be non-empty
    if any(not r["source"] for r in rows):
        stop("a row without a source")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["q", "policy_type", "x_order", "kind", "metric", "seed", "value", "plotted_value", "at_floor",
            "marked", "valid", "source", "reeval_abs_diff", "reeval_value"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    n_floor = sum(r["at_floor"] == "True" for r in rows)
    checks.append(f"data/F2.csv: {len(rows)} rows ({n_floor} at the floor), commit {C.git_commit()}")
    for c in checks:
        print("CHECK", c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
