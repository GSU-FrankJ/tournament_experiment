"""T2 - T = 2 recovery and confirmation (table builder of the P-FIG round).

Reads the 40 rows with ``arm == 'v20'`` of ``results/t3_r1/eval/per_run.csv`` (the frozen v2.0 policies,
final-tier verifier, 20 runs per q) and the gate limits of ``reports/t3/r1/prereg_criteria.json``
(cross-checked against ``protocols/v2_T2_locked_v2_0.json``), and writes ``tables/T2.csv`` (one value per
row) and ``tables/T2.md`` (its readable rendering). Nothing is evaluated or trained; the output is
deterministic.

Usage:
    python paper/multistage_rev/scripts/T2.py [--crosscheck]

``--crosscheck`` additionally compares every used per-run value and verdict with the ``gates.json`` of the
run directory (read-only, untracked root ``V20_ROOT``); it writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from _common import OUT, PER_RUN_R1, QS, ROOT, V20_ROOT, load_per_run, read_json

ARM = "v20"
FAILED_Q, FAILED_SEED = 50, 30510
EXPECTED_PASS = {50: 19, 60: 20}          # stop-and-report if the stored verdicts differ
N_PER_Q = 20
SEEDS = "seeds 30501-30520"

PER = str(PER_RUN_R1.relative_to(ROOT))
PRE = "reports/t3/r1/prereg_criteria.json"
PROT = "protocols/v2_T2_locked_v2_0.json"
COLUMNS = ["table", "row_group", "row", "column_or_q", "statistic", "value", "source"]

# (stored column, CSV label, markdown label, protocol metric name or None, (gate, prereg key) or None)
QUANTITIES: List[Tuple[str, str, str, Any, Any]] = [
    ("stage1_rel_err_signed", "Stage-1 error, signed", "Stage-1 error, signed", None, None),
    ("stage1_rel_err_abs", "Stage-1 error, absolute", "Stage-1 error, absolute",
     "stage1_rel_err_abs", ("G-S", "stage1_rel_err_abs")),
    ("rmse_pos_T_over_eT0", "Stage-2 RMSE", "Stage-2 RMSE", "stage2_rmse_pos_over_g2_0",
     ("G-A", "rmse_pos_T_over_eT0")),
    ("tail_mean_T_over_eT0", "Tail effort", "Tail effort", "stage2_tail_mean_over_g2_0",
     ("G-A", "tail_mean_T_over_eT0")),
    ("peak_T_signed", "Tie-point error, signed", "Tie-point error, signed", None, None),
    ("peak_T_abs", "Tie-point error, absolute", "Tie-point error, absolute", None, None),
    ("eta_T_over_dw", "eta_2/DeltaW", "η_2/ΔW", "eta_T_over_dw", ("G-A", "eta_T_over_dw")),
    ("Gmax_full_over_dw", "G_max,full/DeltaW", "G_max,full/ΔW", "Gmax_full_over_dw",
     ("G-F", "Gmax_full_over_dw")),
]
LABEL = {c: (lc, lm) for c, lc, lm, _, _ in QUANTITIES}

# G-N: development-minus-final differences (stored columns of the final / development tier)
GN = [("eta_T_over_dw_dev_minus_final_abs", "eta_T_over_dw", "dev_eta_T_over_dw", "gn_eta"),
      ("Gmax_full_over_dw_dev_minus_final_abs", "Gmax_full_over_dw", "dev_Gmax_full_over_dw", "gn_gmax")]


def fnum(x: Any) -> str:
    """Round-trip text of a float, plain text of an int or str."""
    if isinstance(x, (bool, np.bool_)):
        return str(int(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    return str(x)


def g4(x: Any) -> str:
    """Markdown rendering: counts as integers, floats with 4 significant digits."""
    if isinstance(x, (bool, np.bool_, int, np.integer)):
        return str(int(x))
    return f"{float(x):.4g}"


def load_inputs() -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """The 40 v20 rows (round-trip floats), the prereg gates and the v2.0 protocol."""
    df = load_per_run(PER_RUN_R1)
    v = df[df["arm"] == ARM].copy()
    pre = read_json(ROOT / PRE)
    prot = read_json(ROOT / PROT)
    return v, pre, prot


def check_inputs(v: Any) -> List[str]:
    """Stop-and-report conditions on the 40 rows; returns the log of the checks that held."""
    log: List[str] = []
    for q in QS:
        n = int((v["q"] == q).sum())
        if n != N_PER_Q:
            raise SystemExit(f"STOP: {n} v20 rows at q={q}, expected {N_PER_Q}")
        seeds = sorted(int(s) for s in v.loc[v["q"] == q, "seed"])
        if seeds != list(range(30501, 30521)):
            raise SystemExit(f"STOP: v20 seeds at q={q} are {seeds}")
    if not (v["status"].eq("done").all() and v["exit_code"].eq(0).all() and v["global_rng_ok"].all()
            and v["valid"].all()):
        raise SystemExit("STOP: a v20 row is not done / exit 0 / global_rng_ok / valid")
    log.append("40 v20 rows: 20 per q, seeds 30501-30520, status done, exit_code 0, global_rng_ok, valid")
    for q in QS:
        n_pass = int(v.loc[v["q"] == q, "gates_pass"].astype(bool).sum())
        if n_pass != EXPECTED_PASS[q]:
            raise SystemExit(f"STOP: pass count at q={q} is {n_pass}/20, expected {EXPECTED_PASS[q]}/20")
        log.append(f"pass count q={q}: {n_pass}/20")
    return log


def check_gates(pre: Dict[str, Any], prot: Dict[str, Any]) -> List[str]:
    """prereg_criteria.json 'gates' against protocols/v2_T2_locked_v2_0.json 'gates' (equal thresholds)."""
    pg, qg = pre["gates"], prot["gates"]
    thr = {}
    for gate, spec in qg.items():
        if isinstance(spec, dict) and "all_must_hold" in spec:
            for c in spec["all_must_hold"]:
                if c["op"] != "<=":
                    raise SystemExit(f"STOP: protocol operator {c['op']}")
                thr[(gate, c["metric"])] = c["threshold"]
    want = {("G-A", "eta_T_over_dw"): pg["G-A"]["eta_T_over_dw"],
            ("G-A", "stage2_rmse_pos_over_g2_0"): pg["G-A"]["rmse_pos_T_over_eT0"],
            ("G-A", "stage2_tail_mean_over_g2_0"): pg["G-A"]["tail_mean_T_over_eT0"],
            ("G-F", "Gmax_full_over_dw"): pg["G-F"]["Gmax_full_over_dw"],
            ("G-N", "eta_T_over_dw_dev_minus_final_abs"): pg["G-N"]["eta_T_over_dw_dev_minus_final_abs"],
            ("G-N", "Gmax_full_over_dw_dev_minus_final_abs"):
                pg["G-N"]["Gmax_full_over_dw_dev_minus_final_abs"],
            ("G-S", "stage1_rel_err_abs"): pg["G-S"]["stage1_rel_err_abs"]}
    if thr != want:
        raise SystemExit(f"STOP: prereg gates {want} differ from protocol gates {thr}")
    if pg["run_pass"]["2"] != ["G-A", "G-F", "G-N", "G-S"] or pg["G-S"]["applies_at_horizon"] != 2:
        raise SystemExit("STOP: run_pass at T = 2 is not G-A, G-F, G-N, G-S")
    if "stage2_peak_rel_err_signed" not in prot["reported_not_gated"]["stage2"]:
        raise SystemExit("STOP: the protocol does not list the tie-point error as not gated")
    return ["7 thresholds of prereg_criteria.json gates equal the thresholds of protocols/"
            "v2_T2_locked_v2_0.json gates (all '<=' inclusive); run_pass at T = 2 is G-A, G-F, G-N, G-S"]


def build_rows(v: Any, pre: Dict[str, Any], prot: Dict[str, Any]) -> List[Dict[str, str]]:
    """All rows of the table, one value per row."""
    rows: List[Dict[str, str]] = []

    def add(group: str, row: str, col: str, stat: str, value: Any, source: str) -> None:
        rows.append({"table": "T2", "row_group": group, "row": row, "column_or_q": col,
                     "statistic": stat, "value": fnum(value), "source": source})

    # ---- A. confirmation, per q
    grp = "confirmation of the frozen v2.0 policies, per q"
    for q in QS:
        sub = v[v["q"] == q]
        where = f"{PER}, rows arm == 'v20' and q == {q} (20 rows, {SEEDS})"
        add(grp, "Runs passing all gates (gates_pass)", f"q={q}", "pass count",
            int(sub["gates_pass"].astype(bool).sum()),
            f"{where}, column gates_pass; count of True over the 20 rows")
        add(grp, "Runs passing all gates (gates_pass)", f"q={q}", "runs evaluated", len(sub),
            f"{where}; number of rows")
    for col, lc, _, _, _ in QUANTITIES:
        for q in QS:
            x = v.loc[v["q"] == q, col].to_numpy(dtype=float)
            for stat, fn, how in (("median", np.median, "numpy.median; even count: mean of the two "
                                   "middle values"), ("min", np.min, "numpy.min"),
                                  ("max", np.max, "numpy.max")):
                add(grp, f"{lc} ({col})", f"q={q}", stat, float(fn(x)),
                    f"{stat} over the 20 rows of arm v20, q={q} ({SEEDS}), column {col} of {PER} ({how})")

    # ---- B. the failed run
    f = v[(v["q"] == FAILED_Q) & (v["seed"] == FAILED_SEED)]
    if len(f) != 1 or bool(f["gates_pass"].iloc[0]):
        raise SystemExit("STOP: the failed run q=50, seed 30510 is not a single failing v20 row")
    r = f.iloc[0]
    qcol = f"q={FAILED_Q}"
    where = f"{PER}, row arm == 'v20', q == {FAILED_Q}, seed == {FAILED_SEED}"
    g1 = "failed run (q=50, seed 30510): quantities"
    for col, lc, _, _, _ in QUANTITIES:
        add(g1, f"{lc} ({col})", qcol, "value (single run)", float(r[col]), f"{where}, column {col}")
    g2 = "failed run (q=50, seed 30510): gates"
    for name, col in (("G-A", "gate_G_A"), ("G-F", "gate_G_F"), ("G-N", "gate_G_N"), ("G-S", "gate_G_S"),
                      ("Run pass, all four gates", "gates_pass")):
        add(g2, f"{name} ({col})", qcol, "stored verdict (1 = holds, 0 = fails)", int(bool(r[col])),
            f"{where}, column {col}")
    pg = pre["gates"]
    subs = [("G-A", "eta_2/DeltaW <= limit", "ga_eta", "eta_T_over_dw", float(r["eta_T_over_dw"]),
             pg["G-A"]["eta_T_over_dw"], f"column eta_T_over_dw; limit {PRE} gates.G-A.eta_T_over_dw"),
            ("G-A", "Stage-2 RMSE <= limit", "ga_rmse", "rmse_pos_T_over_eT0",
             float(r["rmse_pos_T_over_eT0"]), pg["G-A"]["rmse_pos_T_over_eT0"],
             f"column rmse_pos_T_over_eT0; limit {PRE} gates.G-A.rmse_pos_T_over_eT0"),
            ("G-A", "Tail effort <= limit", "ga_tail", "tail_mean_T_over_eT0",
             float(r["tail_mean_T_over_eT0"]), pg["G-A"]["tail_mean_T_over_eT0"],
             f"column tail_mean_T_over_eT0; limit {PRE} gates.G-A.tail_mean_T_over_eT0"),
            ("G-F", "G_max,full/DeltaW <= limit", "gate_G_F", "Gmax_full_over_dw",
             float(r["Gmax_full_over_dw"]), pg["G-F"]["Gmax_full_over_dw"],
             f"column Gmax_full_over_dw; limit {PRE} gates.G-F.Gmax_full_over_dw")]
    for key, fin, dev, flag in GN:
        val = abs(float(r[dev]) - float(r[fin]))
        lab = "eta_2/DeltaW" if fin.startswith("eta") else "G_max,full/DeltaW"
        label = f"abs({lab} (development) - {lab} (final)) <= limit"
        subs.append(("G-N", label, flag, key, val, pg["G-N"][key],
                     f"derived: abs({dev} - {fin}); limit {PRE} gates.G-N.{key}"))
    subs.append(("G-S", "Stage-1 error, absolute <= limit", "gs_s1", "stage1_rel_err_abs",
                 float(r["stage1_rel_err_abs"]), pg["G-S"]["stage1_rel_err_abs"],
                 f"column stage1_rel_err_abs; limit {PRE} gates.G-S.stage1_rel_err_abs"))
    failing: List[str] = []
    for gate, label, flag, key, val, lim, src in subs:
        holds = bool(r[flag])
        if holds != bool(val <= lim):
            raise SystemExit(f"STOP: stored verdict {flag} disagrees with value {val!r} <= {lim!r}")
        if not holds:
            failing.append(f"{gate}: {key}")
        row = f"{gate} sub-condition: {label}"
        add(g2, row, qcol, f"stored verdict ({flag}; 1 = holds, 0 = fails)", int(holds),
            f"{where}, column {flag}")
        add(g2, row, qcol, "value", val, f"{where}, {src.split(';')[0]}")
        add(g2, row, qcol, "limit (<=, inclusive)", lim, src.split("; ")[1].replace("limit ", ""))
    add(g2, "Failing gate and sub-condition", qcol, "name", "; ".join(failing),
        f"derived: the sub-condition columns ga_eta, ga_rmse, ga_tail, gate_G_F, gn_eta, gn_gmax, gs_s1 of {where} "
        "that are False")

    # ---- C. gate limits
    g3 = "gate limits"
    prot_g = prot["gates"]
    for col, lc, _, pmetric, pre_key in QUANTITIES:
        row = f"{lc} ({col})"
        if pre_key is None:
            if col == "stage1_rel_err_signed":
                src = (f"{PRE} gates: no key for {col} (the G-S limit is on stage1_rel_err_abs); "
                       f"{PROT} gates: none")
            else:
                src = (f"{PRE} gates: no key for {col}; {PROT} key peak_error: "
                       f"'{prot['peak_error']}', and key reported_not_gated.stage2 lists "
                       "stage2_peak_rel_err_signed")
            add(g3, row, "not gated", "limit (<=, inclusive)", "no limit", src)
        else:
            gate, key = pre_key
            src = (f"{PRE} gates.{gate}.{key}; equal to {PROT} gates.{gate}.all_must_hold "
                   f"[metric == {pmetric}].threshold")
            add(g3, row, gate, "limit (<=, inclusive)", pre["gates"][gate][key], src)
    for key, fin, dev, _ in GN:
        lab = "eta_2/DeltaW" if fin.startswith("eta") else "G_max,full/DeltaW"
        add(g3, f"abs({lab} (development) - {lab} (final)) ({key})", "G-N", "limit (<=, inclusive)",
            pre["gates"]["G-N"][key],
            f"{PRE} gates.G-N.{key}; equal to {PROT} gates.G-N.all_must_hold [metric == {key}].threshold")
    return rows


def write_csv(rows: List[Dict[str, str]], path: Any) -> None:
    """Write the table CSV (UTF-8, LF, header row)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def render_md(rows: List[Dict[str, str]]) -> str:
    """Readable rendering of the rows (median [min, max]; 4 significant digits; sources as footnotes)."""
    def val(group: str, row: str, col: str, stat: str) -> str:
        m = [x["value"] for x in rows if (x["row_group"], x["row"], x["column_or_q"], x["statistic"])
             == (group, row, col, stat)]
        if len(m) != 1:
            raise SystemExit(f"STOP: {len(m)} rows for {(group, row, col, stat)}")
        return m[0]

    def num(s: str) -> str:
        return g4(int(s)) if s.lstrip("-").isdigit() else g4(float(s))

    ga, gf = "confirmation of the frozen v2.0 policies, per q", None
    L: List[str] = []
    L.append("# T2. T = 2 recovery and confirmation\n")
    L.append("Frozen v2.0 policies at T = 2, final-tier verifier: the confirmation runs stored as arm `v20` (20 per q) in "
             "`results/t3_r1/eval/per_run.csv`. Every number below is a row of `T2.csv`; signed errors keep "
             "their sign. Notation: e_1(0) and e_2(d) are the learned mean efforts at stages 1 and 2, g1 and "
             "g_2(d) the closed-form equilibrium; the tie point is d = 0, the positive-effort region is "
             "|d| < 2q and the theoretical zero-effort tails are |d| >= 2q (stage 2).\n")
    L.append("## A. Confirmation, per q: median [min, max] over the 20 runs\n")
    L.append("| Quantity | Stored column | q = 50 | q = 60 | Source |")
    L.append("|---|---|---|---|---|")
    pr = "Runs passing all gates (gates_pass)"
    cells = [f"{num(val(ga, pr, f'q={q}', 'pass count'))} of {num(val(ga, pr, f'q={q}', 'runs evaluated'))}"
             for q in QS]
    L.append(f"| Runs passing all gates (G-A, G-F, G-N, G-S) | `gates_pass` | {cells[0]} | {cells[1]} | [1] |")
    desc = {"stage1_rel_err_signed": "(e_1(0) - g1)/g1",
            "stage1_rel_err_abs": "abs((e_1(0) - g1)/g1)",
            "rmse_pos_T_over_eT0": "RMSE of e_2 to g_2 over the positive-effort region / g_2(0)",
            "tail_mean_T_over_eT0": "mean of e_2 over the theoretical zero-effort tails / g_2(0)",
            "peak_T_signed": "(e_2(0) - g_2(0))/g_2(0)", "peak_T_abs": "abs((e_2(0) - g_2(0))/g_2(0))",
            "eta_T_over_dw": "largest one-step deviation gain over D_2, / ΔW",
            "Gmax_full_over_dw": "largest state-wise dynamic gain over all stages and states, / ΔW"}
    for col, lc, lm, _, _ in QUANTITIES:
        row = f"{lc} ({col})"
        cs = []
        for q in QS:
            m, lo, hi = (num(val(ga, row, f"q={q}", s)) for s in ("median", "min", "max"))
            cs.append(f"{m} [{lo}, {hi}]")
        L.append(f"| {lm}: {desc[col]} | `{col}` | {cs[0]} | {cs[1]} | [1] |")
    L.append("")
    L.append("## B. The failed run (q = 50, seed 30510)\n")
    g1 = "failed run (q=50, seed 30510): quantities"
    g2 = "failed run (q=50, seed 30510): gates"
    L.append("| Quantity | Stored column | Value | Source |")
    L.append("|---|---|---|---|")
    for col, lc, lm, _, _ in QUANTITIES:
        L.append(f"| {lm} | `{col}` | {num(val(g1, f'{lc} ({col})', 'q=50', 'value (single run)'))} | [2] |")
    L.append("")
    names = [("G-A", "gate_G_A"), ("G-F", "gate_G_F"), ("G-N", "gate_G_N"), ("G-S", "gate_G_S"),
             ("Run pass, all four gates", "gates_pass")]
    verd = {n: ("holds" if val(g2, f"{n if n.startswith('Run') else n} ({c})", "q=50",
                              "stored verdict (1 = holds, 0 = fails)") == "1" else "fails")
            for n, c in names}
    L.append("Gate verdicts: " + "; ".join(f"{n} {verd[n]}" for n, _ in names) + " (stored columns "
             + ", ".join(f"`{c}`" for _, c in names) + ") [2].\n")
    L.append("| Gate | Sub-condition | Stored verdict column | Value | Limit (<=) | Verdict | Source |")
    L.append("|---|---|---|---|---|---|---|")
    sub_rows = []
    for x in rows:
        if x["row_group"] == g2 and x["row"].split(" sub-condition: ")[0] in ("G-A", "G-F", "G-N", "G-S") \
                and " sub-condition: " in x["row"] and x["statistic"] == "value":
            sub_rows.append(x["row"])
    flags = {"eta_2/DeltaW <= limit": "ga_eta", "Stage-2 RMSE <= limit": "ga_rmse",
             "Tail effort <= limit": "ga_tail", "G_max,full/DeltaW <= limit": "gate_G_F",
             "Stage-1 error, absolute <= limit": "gs_s1"}
    for row in sub_rows:
        gate, lab = row.split(" sub-condition: ")
        flag = flags.get(lab) or ("gn_eta" if "eta_2" in lab else "gn_gmax")
        lab_md = (lab.replace(" <= limit", "").replace("DeltaW", "ΔW").replace("eta_2", "η_2")
                  .replace("G_max,full", "G_max,full"))
        v_ = num(val(g2, row, "q=50", "value"))
        lim = num(val(g2, row, "q=50", "limit (<=, inclusive)"))
        ok = val(g2, row, "q=50", f"stored verdict ({flag}; 1 = holds, 0 = fails)") == "1"
        src = "[2], [3], [4]" if gate == "G-N" else "[2], [3]"
        L.append(f"| {gate} | {lab_md} | `{flag}` | {v_} | {lim} | {'holds' if ok else 'fails'} | {src} |")
    fail = val(g2, "Failing gate and sub-condition", "q=50", "name")
    L.append("")
    L.append(f"Failing gate and sub-condition, read from the stored verdict columns: **{fail}**.\n")
    L.append("## C. Gate limits (all `<=`, inclusive, full float64 precision)\n")
    L.append("| Quantity | Stored column | Gate | Limit | Source |")
    L.append("|---|---|---|---|---|")
    g3 = "gate limits"
    for x in rows:
        if x["row_group"] == g3:
            lab, col = x["row"].rsplit(" (", 1)
            col = col.rstrip(")")
            lab_md = (lab.replace("eta_2/DeltaW", "η_2/ΔW").replace("G_max,full/DeltaW", "G_max,full/ΔW")
                      .replace("DeltaW", "ΔW"))
            if lab.startswith("abs("):
                lab_md = "Development-minus-final difference, " + (
                    "η_2/ΔW" if "eta_2" in lab else "G_max,full/ΔW")
            lim = x["value"] if x["value"] == "no limit" else num(x["value"])
            gate = x["column_or_q"]
            src = "[5]" if lim != "no limit" else "[6]"
            L.append(f"| {lab_md} | `{col}` | {gate} | {lim} | {src} |")
    L.append("")
    L.append("## Sources\n")
    L.append(f"1. `{PER}`: the 20 rows with `arm == 'v20'` and `q == 50` ({SEEDS}) and the 20 rows with "
             "`q == 60`, column as named. Pass count: number of `True` in `gates_pass`. Median, min, max: "
             "numpy over the 20 rows (median of the even count = mean of the two middle values).")
    L.append(f"2. `{PER}`: the single row with `arm == 'v20'`, `q == 50`, `seed == 30510`, column as named; "
             "gate verdicts from the columns `gate_G_A`, `gate_G_F`, `gate_G_N`, `gate_G_S`, `gates_pass`, "
             "`ga_eta`, `ga_rmse`, `ga_tail`, `gn_eta`, `gn_gmax`, `gs_s1`.")
    L.append(f"3. Limits of the sub-conditions: `{PRE}` key `gates` (see C).")
    L.append("4. G-N value: derived as abs(`dev_eta_T_over_dw` - `eta_T_over_dw`) and "
             "abs(`dev_Gmax_full_over_dw` - `Gmax_full_over_dw`) of the same row (development-tier and "
             "final-tier columns).")
    L.append(f"5. `{PRE}` key `gates` (G-A, G-F, G-N, G-S); each threshold equals the one in "
             f"`{PROT}` key `gates` (metric names there: `stage2_rmse_pos_over_g2_0` for "
             "`rmse_pos_T_over_eT0`, `stage2_tail_mean_over_g2_0` for `tail_mean_T_over_eT0`).")
    L.append(f"6. No limit: the quantity has no key in `gates` of `{PRE}` or `{PROT}`; the tie-point error "
             f"is listed under `reported_not_gated.stage2` and `peak_error` ('not gated') of `{PROT}`; the "
             "G-S limit applies to the absolute stage-1 error only.")
    L.append("")
    return "\n".join(L)


def crosscheck(v: Any) -> int:
    """Compare every used per-run value and verdict with gates.json of the run (read-only); returns #mismatch."""
    where = {"stage1_rel_err_signed": ("end_of_B", "stage1_rel_err_signed"),
             "stage1_rel_err_abs": ("end_of_B", "stage1_rel_err_abs"),
             "rmse_pos_T_over_eT0": ("end_of_A", "stage2_rmse_pos_over_g2_0"),
             "tail_mean_T_over_eT0": ("end_of_A", "stage2_tail_mean_over_g2_0"),
             "peak_T_signed": ("end_of_A", "stage2_peak_rel_err_signed"),
             "peak_T_abs": ("end_of_A", "stage2_peak_rel_err_abs"),
             "eta_T_over_dw": ("end_of_A", "eta_T_over_dw"),
             "Gmax_full_over_dw": ("end_of_B", "Gmax_full_over_dw")}
    n = bad = 0
    for _, r in v.iterrows():
        g = json.loads((V20_ROOT / f"q{int(r['q'])}" / f"seed{int(r['seed'])}" / "gates.json").read_text())
        for col, (ph, key) in where.items():
            n += 1
            if float(r[col]) != g["reported"][ph]["final"][key]:
                bad += 1
                print("MISMATCH", r["q"], r["seed"], col)
        for col, gk in (("gate_G_A", "G-A"), ("gate_G_F", "G-F"), ("gate_G_N", "G-N"), ("gate_G_S", "G-S")):
            n += 1
            bad += bool(r[col]) != bool(g[gk]["pass"])
        n += 1
        bad += bool(r["gates_pass"]) != bool(g["gates_pass"])
    print(f"crosscheck against gates.json of the 40 run directories: {n} comparisons, {bad} mismatches")
    return bad


def main() -> int:
    """Build tables/T2.csv and tables/T2.md."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--crosscheck", action="store_true", help="also compare with the runs' gates.json")
    args = ap.parse_args()
    v, pre, prot = load_inputs()
    for line in check_inputs(v) + check_gates(pre, prot):
        print("check:", line)
    if args.crosscheck and crosscheck(v):
        return 1
    rows = build_rows(v, pre, prot)
    out = OUT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "T2.csv")
    (out / "T2.md").write_text(render_md(rows), encoding="utf-8")
    print(f"wrote {out / 'T2.csv'} ({len(rows)} rows) and {out / 'T2.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
