#!/usr/bin/env python
"""T1 builder (round P-FIG): experimental protocols and gates, one column each for T = 2 v2.0, T3-1.0 and T3-1.1.

Reads the protocol JSON files, ``reports/t3/r1/prereg_criteria.json``, the run configs / manifests / summaries of the runs
of each protocol (untracked roots, read-only), the stored analysis CSVs, and a few code lines (quoted and asserted).
Writes ``tables/T1.csv`` (one value per row: table, row_group, row, column_or_q, statistic, value, source) and
``tables/T1.md`` (the readable rendering, produced FROM the CSV). Deterministic; evaluates and trains nothing. Every
consistency check is an assertion: on a failed check nothing is written (a STOP for the round).

Run:  python paper/multistage_rev/scripts/T1.py     (any cwd)
"""

from __future__ import annotations

import _common as C  # noqa: E402  (pins the thread env vars; adds the repo to sys.path)

import copy
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

C.single_thread()

import t3_configs as tc  # noqa: E402  (tools/t3; only its config_sha256 is used)
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402

ROOT = C.ROOT
OUT = C.OUT
WT = C.WT
QS = C.QS
CHECKS: List[Tuple[str, str]] = []          # (what, result) - printed at the end and kept in the log


def check(what: str, cond: bool, detail: str = "") -> None:
    """Record a consistency check; a failed check aborts the script before anything is written."""
    CHECKS.append((what, ("OK " + detail).strip() if cond else "FAILED " + detail))
    if not cond:
        for w, r in CHECKS:
            print(f"{r[:6]:6s} {w} {r[3:]}", file=sys.stderr)
        raise SystemExit(f"STOP: check failed: {what} {detail}")


def rel(p: Path) -> str:
    """Path relative to the repository (tracked) or to the main checkout (``.claude/worktrees/...``, untracked)."""
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return ".claude/worktrees/" + str(p.relative_to(WT))


def rj(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def code_line(path: str, needle: str, nth: int = 1) -> Tuple[int, str]:
    """Line number (1-based) and text of the ``nth`` line of a repo file containing ``needle`` (asserted to exist)."""
    lines = (ROOT / path).read_text().splitlines()
    hits = [i for i, l in enumerate(lines, 1) if needle in l]
    check(f"code line {path}: {needle!r}", len(hits) >= nth, f"(found {len(hits)})")
    return hits[nth - 1], lines[hits[nth - 1] - 1].strip()


def code_par(path: str, first_needle: str, last_needle: str) -> Tuple[int, int, str]:
    """(first line, last line, text) of the block of a repo file from the line containing ``first_needle`` to the first line
    (from there on) containing ``last_needle``; non-empty lines joined, white space collapsed (asserted to exist)."""
    lines = (ROOT / path).read_text().splitlines()
    i = [k for k, l in enumerate(lines) if first_needle in l]
    check(f"code block start {path}: {first_needle!r}", len(i) >= 1)
    j = [k for k in range(i[0], len(lines)) if last_needle in lines[k]]
    check(f"code block end {path}: {last_needle!r}", len(j) >= 1)
    block = [l.strip() for l in lines[i[0]:j[0] + 1] if l.strip()]
    block[-1] = block[-1][:block[-1].index(last_needle) + len(last_needle)]      # cut the last line after the end marker
    return i[0] + 1, j[0] + 1, re.sub(r"\s+", " ", " ".join(block))


# ----------------------------------------------------------------------------------------------- sources
P20_REL, P30_REL, P31_REL = ("protocols/v2_T2_locked_v2_0.json", "protocols/v2_T3_locked_v1_0.json",
                             "protocols/v2_T3_locked_v1_1.json")
CRIT_REL, PARAMS_REL = "reports/t3/r1/prereg_criteria.json", "reports/t3/r1/params_bw_t3.json"
P20, P30, P31 = (rj(ROOT / P20_REL), rj(ROOT / P30_REL), rj(ROOT / P31_REL))
CRIT, PARAMS = rj(ROOT / CRIT_REL), rj(ROOT / PARAMS_REL)

SEEDS = {"v20": list(range(30501, 30521)), "t30": list(range(50501, 50521)), "t31": list(range(80501, 80541))}
CFG_PATH = {"v20": lambda q, s: C.v20_run_dir(q, s) / "run_config.json",
            "t30": lambda q, s: C.r1_run_dir(q, s, "bw") / "run_config.json",
            "t31": lambda q, s: C.r4_run_dir(q, s) / "run_config.json"}
CFG_PATTERN = {  # for the source strings
    "v20": rel(C.v20_run_dir(50, 30501)).replace("q50", "q{q}").replace("seed30501", "seed{seed}") + "/run_config.json",
    "t30": rel(C.r1_run_dir(50, 50501, "bw")).replace("q50", "q{q}").replace("seed50501", "seed{seed}") + "/run_config.json",
    "t31": rel(C.r4_run_dir(50, 80501)).replace("q50", "q{q}").replace("seed80501", "seed{seed}") + "/run_config.json"}
MAN_PATH = {k: (lambda q, s, f=f: f(q, s).parent / "manifest.json") for k, f in CFG_PATH.items()}
CFGS = {col: {(q, s): rj(CFG_PATH[col](q, s)) for q in QS for s in SEEDS[col]} for col in CFG_PATH}
MANS = {col: {(q, s): rj(MAN_PATH[col](q, s)) for q in QS for s in SEEDS[col]} for col in CFG_PATH}
NRUN = {col: len(v) for col, v in CFGS.items()}
check("run configs read", NRUN == {"v20": 40, "t30": 40, "t31": 80}, str(NRUN))

COLS = ("v20", "t30", "t31")
COLNAME = {"v20": "T = 2 v2.0", "t30": "T3-1.0", "t31": "T3-1.1"}
PROTO = {"v20": (P20, P20_REL), "t30": (P30, P30_REL), "t31": (P31, P31_REL)}
GATES_P = {"v20": P20["gates"], "t30": P30["gates"], "t31": P31["gates"]}


def psrc(col: str, key: str) -> str:
    """Source string: protocol JSON file + key path."""
    return f"{PROTO[col][1]} :: {key}"


def csrc(col: str, key: str, note: str = "all configs of that q agree") -> str:
    """Source string: run_config.json pattern + key path (+ how many configs were read)."""
    return f"{CFG_PATTERN[col]} :: {key} ({NRUN[col]} configs read; {note})"


def fsrc(path: Path, what: str = "") -> str:
    return rel(path) + (f" :: {what}" if what else "")


# ----------------------------------------------------------------------------------------------- the rows
ROWS: List[Dict[str, Any]] = []
GROUP_ORDER: List[str] = []


def add(group: str, row: str, col: str, stat: str, value: Any, source: str) -> None:
    """Append one value (one row of the CSV). ``value`` is an int, a float or a string."""
    assert col in COLS and source and group, (group, row, col)
    assert not isinstance(value, (bool, np.bool_)), (group, row, stat)
    if isinstance(value, (np.integer,)):
        value = int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if group not in GROUP_ORDER:
        GROUP_ORDER.append(group)
    ROWS.append({"table": "T1", "row_group": group, "row": row, "column_or_q": COLNAME[col], "statistic": stat,
                 "value": value, "source": source})


def agree(col: str, getter, what: str, per_q: bool = True) -> Dict[int, Any]:
    """The value of ``getter(cfg)`` that every config of that q holds (asserted); returns {q: value}."""
    out: Dict[int, Any] = {}
    for q in QS:
        vals = [json.dumps(getter(CFGS[col][(q, s)]), sort_keys=True) for s in SEEDS[col]]
        check(f"{COLNAME[col]} q={q}: {what} identical in {len(vals)} configs", len(set(vals)) == 1)
        out[q] = json.loads(vals[0])
    return out


# ----------------------------------------------------------------------------------------------- checks on the runs read
def drop_run_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Config without the fields that differ per run (run name, seed, output directory)."""
    c = copy.deepcopy(cfg)
    for k in ("run", "seed"):
        c.pop(k, None)
    for k in ("run", "seed", "output_dir"):
        c["record"].pop(k, None)
    return c


# v2.0: the 20 configs of a q agree on every field except the run fields
for q in QS:
    js = {json.dumps(drop_run_fields(CFGS["v20"][(q, s)]), sort_keys=True) for s in SEEDS["v20"]}
    check(f"T = 2 v2.0 q={q}: run_config.json identical in the 20 runs except run/seed/output_dir", len(js) == 1)


def normalise_r1(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The lock's normalisation (tools/t3/t3c_launch.normalise): run fields reset to T3-R1's reference run."""
    c = copy.deepcopy(cfg)
    name = f"t3r1_q{int(c['q'])}_s50501_bw"
    c["seed"] = 50501
    c["record"]["seed"] = 50501
    c["run"] = name
    c["record"]["run"] = name
    c["record"]["output_dir"] = "REFERENCE"
    c["pilot"] = "t3_r1"
    return c


for q in QS:
    h30 = {tc.config_sha256(normalise_r1(CFGS["t30"][(q, s)])) for s in SEEDS["t30"]}
    check(f"T3-1.0 q={q}: the 20 T3-R1 bw configs normalise to reference_config_sha256 of protocols/v2_T3_locked_v1_0.json",
          h30 == {P30["reference_config_sha256"][str(q)]})
    h31 = {tc.config_sha256(normalise_r1(CFGS["t31"][(q, s)])) for s in SEEDS["t31"]}
    check(f"T3-1.1 q={q}: the 40 T3-R4 configs normalise to reference_config_sha256 of protocols/v2_T3_locked_v1_1.json",
          h31 == {P31["reference_config_sha256"][str(q)]})

# protocol record == run config on the quoted game / verifier fields
for col in COLS:
    P = PROTO[col][0]
    for q in QS:
        rec_p, rec_c = P["records"][str(q)], CFGS[col][(q, SEEDS[col][0])]["record"]
        gp, gc = dict(rec_p["game"]), dict(rec_c["game"])
        if col != "v20":                                   # the T3 JSON keeps the v2.0 record (T = 2); the config sets T = 3
            check(f"{COLNAME[col]} q={q}: protocol record game.T is 2 (v2.0 record), config game.T is 3", gp["T"] == 2 and gc["T"] == 3)
            gp["T"] = gc["T"]
        check(f"{COLNAME[col]} q={q}: protocol record game == run_config record.game", gp == gc)
        check(f"{COLNAME[col]} q={q}: protocol record dw, B, verifier == run_config",
              rec_p["dw"] == rec_c["dw"] and rec_p["B"] == rec_c["B"] and rec_p["verifier"] == rec_c["verifier"])
        check(f"{COLNAME[col]} q={q}: protocol record protocol.episodes_per_update, es_bin_width == run_config",
              rec_p["protocol"]["episodes_per_update"] == rec_c["protocol"]["episodes_per_update"]
              and rec_p["protocol"]["es_bin_width"] == rec_c["protocol"]["es_bin_width"])

# the files read at the base commit are the locked ones (SHA-256 recorded in the protocols)
def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


check("SHA-256 of protocols/v2_T2_locked_v2_0.json == protocol_sha256 of T3-1.0 and T3-1.1", sha(P20_REL) == P30["protocol_sha256"] == P31["protocol_sha256"])
check("SHA-256 of params_bw_t3.json and prereg_criteria.json == params_sha256 and criteria_sha256 of both T3 protocols",
      sha(PARAMS_REL) == P30["params_sha256"] == P31["params_sha256"] and sha(CRIT_REL) == P30["criteria_sha256"] == P31["criteria_sha256"])
check("SHA-256 of protocols/v2_T3_locked_v1_0.json == builds_on.protocol_sha256 of T3-1.1", sha(P30_REL) == P31["builds_on"]["protocol_sha256"])
check("tools/t3/t3_eval.py, t3_metrics.py and run/ms_retry.py, run/run_ms_stagewise.py at the base commit match the tool_sha256 of the locks",
      all(sha(f) == P31["tool_sha256"][f] for f in ("tools/t3/t3_eval.py", "tools/t3/t3_metrics.py", "run/ms_retry.py", "run/run_ms_stagewise.py"))
      and all(sha(f) == P30["tool_sha256"][f] for f in ("tools/t3/t3_eval.py", "tools/t3/t3_metrics.py")))

# verifier tiers: protocol == code constants
from utils.dp_br_verifier import DEV_CONFIG, FINAL_CONFIG  # noqa: E402
for q in QS:
    v = P20["records"][str(q)]["verifier"]
    check(f"q={q}: verifier tiers of the v2.0 record == FINAL_CONFIG / DEV_CONFIG of utils/dp_br_verifier.py",
          all(v[t][k] == getattr(cfg, k) for t, cfg in (("final", FINAL_CONFIG), ("development", DEV_CONFIG))
              for k in ("state_step", "effort_step", "gl_half")))

# gates: T3-1.0 == T3-1.1 == prereg_criteria.json; v2.0 thresholds equal the T3 thresholds
check("gates of T3-1.0 == gates of T3-1.1 == gates of prereg_criteria.json", P30["gates"] == P31["gates"] == CRIT["gates"])
g20 = {m["metric"]: m["threshold"] for gate in ("G-A", "G-F", "G-N", "G-S") for m in P20["gates"][gate]["all_must_hold"]}
gc3 = CRIT["gates"]
check("v2.0 thresholds == prereg_criteria.json thresholds",
      g20["eta_T_over_dw"] == gc3["G-A"]["eta_T_over_dw"] and g20["stage2_rmse_pos_over_g2_0"] == gc3["G-A"]["rmse_pos_T_over_eT0"]
      and g20["stage2_tail_mean_over_g2_0"] == gc3["G-A"]["tail_mean_T_over_eT0"]
      and g20["Gmax_full_over_dw"] == gc3["G-F"]["Gmax_full_over_dw"]
      and g20["eta_T_over_dw_dev_minus_final_abs"] == gc3["G-N"]["eta_T_over_dw_dev_minus_final_abs"]
      and g20["Gmax_full_over_dw_dev_minus_final_abs"] == gc3["G-N"]["Gmax_full_over_dw_dev_minus_final_abs"]
      and g20["stage1_rel_err_abs"] == gc3["G-S"]["stage1_rel_err_abs"])
K_STORED = P20["records"]["50"]["game"]["k"]
check("stored k == 1.0/3500.0 exactly, both q, all three protocols",
      all(PROTO[c][0]["records"][str(q)]["game"]["k"] == 1.0 / 3500.0 for c in COLS for q in QS))

# manifests: commit and clean tree
COMMIT: Dict[str, str] = {}
for col in COLS:
    cs = {(m["commit"], m["clean_tree"]) for m in MANS[col].values()}
    check(f"{COLNAME[col]}: manifest.json commit identical and clean_tree true in all {NRUN[col]} runs", len(cs) == 1 and next(iter(cs))[1] is True,
          str(cs))
    COMMIT[col] = next(iter(cs))[0]

import subprocess  # noqa: E402


def ln_at(col: str, path: str, needle: str, nth: int = 1) -> int:
    """Line number of ``needle`` in ``path`` as it was at the commit recorded in the manifests of the runs of ``col``."""
    txt = subprocess.run(["git", "-C", str(ROOT), "show", f"{COMMIT[col]}:{path}"], capture_output=True, text=True)
    check(f"git show {COMMIT[col][:7]}:{path} readable", txt.returncode == 0)
    hits = [i for i, l in enumerate(txt.stdout.splitlines(), 1) if needle in l]
    check(f"{path} at {COMMIT[col][:7]} contains {needle!r}", len(hits) >= nth)
    return hits[nth - 1]


MSW = "run/run_ms_stagewise.py"
LN = {k: {c: ln_at(c, MSW, n) for c in ("t30", "t31")} for k, n in (
    ("root", "d0 = np.zeros(n_ep)"), ("bal", "d0 = self.sampler.balanced(t, n_ep, rng_start)"), ("t1", "if t == 1:"),
    ("ms710", 'reward_mode="expected"'), ("cts", "CONT_TABLE_STEP = 0.05"), ("dv", "ev, err = self.verify_candidate(self.dev_cfg)"),
    ("snap", "snap = copy.deepcopy(self.agent.actor)"),
    ("fz_def", "def freeze_stage("), ("fz_loop", "for t in range(T, 0, -1):"), ("fz_call", "run.freeze_stage(t, rec, check)"),
    ("fz_build", "tbl = build_table(snap, spec, t - 1, self.tables.get(t), step=CONT_TABLE_STEP)"),
    ("fz_store", "self.tables[t - 1] = tbl"),
    ("fz_rng", 'raise RuntimeError("a training RNG stream moved during the continuation-table build")'),
    ("tb_use", "table_next = self.tables.get(t) if t < T else None"))}


def msw(col: str, key: str) -> str:
    """Citation of a line of run/run_ms_stagewise.py as it was at the run commit of the column."""
    return f"{MSW}:{LN[key][col]} (file at the run commit {COMMIT[col][:7]})"


# pass counts recomputed from the per-run CSVs with the thresholds of prereg_criteria.json
PR1 = C.load_per_run(C.PER_RUN_R1)
PR2 = C.load_per_run(C.ROOT / "results/t3_r2/confirmation/eval/per_run.csv")
PR3 = C.load_per_run(C.ROOT / "results/t3_r3/dev/eval/per_run.csv")
PR4 = C.load_per_run(C.PER_RUN_R4)
GA, GF, GN, GS = (CRIT["gates"][k] for k in ("G-A", "G-F", "G-N", "G-S"))


def run_ok(df: pd.DataFrame, T: int) -> pd.DataFrame:
    """Per-run gate parts and run pass, recomputed with the thresholds of prereg_criteria.json (inclusive <=)."""
    out = pd.DataFrame(index=df.index)
    out["ga_eta"] = df.eta_T_over_dw <= GA["eta_T_over_dw"]
    out["ga_rmse"] = df.rmse_pos_T_over_eT0 <= GA["rmse_pos_T_over_eT0"]
    out["ga_tail"] = df.tail_mean_T_over_eT0 <= GA["tail_mean_T_over_eT0"]
    out["gf"] = df.Gmax_full_over_dw <= GF["Gmax_full_over_dw"]
    out["gn_eta"] = (df.dev_eta_T_over_dw - df.eta_T_over_dw).abs() <= GN["eta_T_over_dw_dev_minus_final_abs"]
    out["gn_gmax"] = (df.dev_Gmax_full_over_dw - df.Gmax_full_over_dw).abs() <= GN["Gmax_full_over_dw_dev_minus_final_abs"]
    out["gs"] = (df.stage1_rel_err_abs <= GS["stage1_rel_err_abs"]) if T == GS["applies_at_horizon"] else True
    infra = (df.status == "done") & (df.exit_code == 0) & (df.global_rng_ok == True) & (df.valid == True) & (df.dev_valid == True)  # noqa: E712
    out["ok"] = infra & out.ga_eta & out.ga_rmse & out.ga_tail & out.gf & out.gn_eta & out.gn_gmax & out.gs
    return out


def failing_parts(df: pd.DataFrame, T: int) -> Dict[Tuple[int, int], str]:
    """{(q, seed): failing gate parts as text} for the runs that do not pass (stored per-run values)."""
    r = run_ok(df, T)
    out: Dict[Tuple[int, int], str] = {}
    for i in df.index[~r.ok]:
        parts = []
        if not r.ga_eta[i]:
            parts.append("G-A (eta_T_over_dw)")
        if not r.ga_rmse[i]:
            parts.append("G-A (rmse_pos_T_over_eT0)")
        if not r.ga_tail[i]:
            parts.append("G-A (tail_mean_T_over_eT0)")
        if not r.gf[i]:
            parts.append("G-F (Gmax_full_over_dw)")
        if not r.gn_eta[i]:
            parts.append("G-N (eta_T_over_dw)")
        if not r.gn_gmax[i]:
            parts.append("G-N (Gmax_full_over_dw)")
        if not r.gs[i]:
            parts.append("G-S (stage1_rel_err_abs)")
        out[(int(df.q[i]), int(df.seed[i]))] = "; ".join(parts) if parts else "run did not complete / invalid evaluation"
    return out


def counts(df: pd.DataFrame, T: int) -> Dict[int, int]:
    r = run_ok(df, T)
    return {q: int(r.ok[df.q == q].sum()) for q in QS}


V20 = PR1[PR1.arm == "v20"].reset_index(drop=True)
BW1 = PR1[PR1.arm == "bw"].reset_index(drop=True)
cnt = {"v20_conf": counts(V20, 2), "bw_r1": counts(BW1, 3), "bw_r2": counts(PR2, 3), "bwA_r4": counts(PR4, 3)}
check("v2.0 confirmation pass counts recomputed from results/t3_r1/eval/per_run.csv (arm v20) are 19/20 and 20/20",
      cnt["v20_conf"] == {50: 19, 60: 20}, str(cnt["v20_conf"]))
check("T3-R1 bw pass counts recomputed (arm bw) are 18/20 and 18/20", cnt["bw_r1"] == {50: 18, 60: 18}, str(cnt["bw_r1"]))
check("T3-R2 confirmation (T3-1.0) pass counts recomputed are 16/20 and 18/20 (34/40)", cnt["bw_r2"] == {50: 16, 60: 18}, str(cnt["bw_r2"]))
check("T3-R4 confirmation (T3-1.1) pass counts recomputed are 40/40 and 39/40 (79/80)", cnt["bwA_r4"] == {50: 40, 60: 39}, str(cnt["bwA_r4"]))
for arm in ("bw", "bwA", "bwAB"):
    cnt[f"r3_{arm}"] = counts(PR3[PR3.arm == arm].reset_index(drop=True), 3)
check("T3-R3 development counts recomputed: bwA 20+19, bw 18+19, bwAB 18+20",
      (cnt["r3_bwA"], cnt["r3_bw"], cnt["r3_bwAB"]) == ({50: 20, 60: 19}, {50: 18, 60: 19}, {50: 18, 60: 20}), str((cnt["r3_bwA"], cnt["r3_bw"], cnt["r3_bwAB"])))
# the stored run-pass flags agree with the recomputation (R1 table has gates_pass)
check("stored gates_pass == recomputed run pass (R1 v20 and bw rows, R2, R4)",
      all(bool((run_ok(d, T).ok == d.gates_pass).all()) for d, T in ((V20, 2), (BW1, 3), (PR2, 3), (PR4, 3))))

# the seeds of every block
check("per-run CSV seeds: v20 30501-30520 and bw 50501-50520 (R1), 60501-60520 (R2), 70501-70520 (R3), 80501-80540 (R4), both q",
      all(sorted(d[d.q == q].seed) == list(range(a, b + 1)) for d, a, b in
          ((V20, 30501, 30520), (BW1, 50501, 50520), (PR2, 60501, 60520), (PR3[PR3.arm == "bwA"], 70501, 70520), (PR4, 80501, 80540))
          for q in QS))


# ----------------------------------------------------------------------------------------------- budgets (stored CSVs + run files)
COST1 = C.load_per_run(C.ROOT / "results/t3_r1/analysis/cost.csv")
CV, CB = COST1[COST1.arm == "v20"], COST1[COST1.arm == "bw"]
NEP = 512
check("episodes_per_update is 512 in the protocol records of all three protocols and in all configs",
      all(PROTO[c][0]["records"][str(q)]["protocol"]["episodes_per_update"] == NEP for c in COLS for q in QS)
      and all(cfg["record"]["protocol"]["episodes_per_update"] == NEP for c in COLS for cfg in CFGS[c].values()))
BUD = {"v20": {"updates": {"2": 1600, "1": 600}}, "t30": {"updates": {"3": 1600, "2": 1600, "1": 600}}}
check("v2.0 budgets: protocol phase_A.updates 1600 / phase_B.updates 600; cost.csv (arm v20, 40 runs): updates 2200, rows_t2 819200, rows_t1 307200, total 1126400",
      P20["pipeline"]["phase_A"]["updates"] == 1600 and P20["pipeline"]["phase_B"]["updates"] == 600
      and set(CV.updates) == {2200} and set(CV.rows_t2) == {1600 * NEP} and set(CV.rows_t1) == {600 * NEP}
      and set(CV.policy_rows_total) == {2200 * NEP} and len(CV) == 40)
check("T3-1.0 budgets: protocol settings.budgets 1600/1600/600; cost.csv (arm bw, 40 runs): updates 3800, rows_t3 819200, rows_t2 819200, rows_t1 307200, total 1945600",
      P30["settings"]["budgets"] == {"3": 1600, "2": 1600, "1": 600} and set(CB.updates) == {3800}
      and set(CB.rows_t3) == {1600 * NEP} and set(CB.rows_t2) == {1600 * NEP} and set(CB.rows_t1) == {600 * NEP}
      and set(CB.policy_rows_total) == {3800 * NEP} and len(CB) == 40)
check("T3-1.1 budgets: settings.budgets 1600/1600/600 as T3-1.0; budget_of_a_fired_run 5400 updates / 2764800 rows = 5400*512",
      P31["settings"]["budgets"] == {"3": 1600, "2": 1600, "1": 600}
      and P31["settings"]["retry"]["budget_of_a_fired_run"]["updates"] == 5400
      and P31["settings"]["retry"]["budget_of_a_fired_run"]["policy_rows"] == 5400 * NEP == 2764800)

# the run files themselves: v2.0 (v2_updates.csv), T3-R1 bw (ms_run_summary.json), T3-R4 (ms_run_summary.json + retry.json)
for q in QS:
    for s in SEEDS["v20"]:
        u = pd.read_csv(C.v20_run_dir(q, s) / "v2_updates.csv", usecols=["phase", "n_policy_rows"])
        ok = (u[u.phase == "A"].n_policy_rows == NEP).all() and len(u[u.phase == "A"]) == 1600 \
            and (u[u.phase == "B"].n_policy_rows == NEP).all() and len(u[u.phase == "B"]) == 600
        check(f"v2.0 q={q} seed {s}: v2_updates.csv has 1600 phase-A and 600 phase-B updates of 512 policy rows", bool(ok))
    for s in SEEDS["t30"]:
        d = C.r1_run_dir(q, s, "bw")
        sm = rj(d / "ms_run_summary.json")
        pt = sm["phase_timing"]
        ok = (pt["stage3"]["updates"], pt["stage2"]["updates"], pt["stage1"]["updates"]) == (1600, 1600, 600) \
            and (pt["stage3"]["global_exit"], pt["stage2"]["global_exit"], pt["stage1"]["global_exit"]) == (1600, 3200, 3800) \
            and sm["total_transitions"] == 3800 * NEP and sm["final_global_update"] == 3800
        check(f"T3-R1 bw q={q} seed {s}: ms_run_summary.json 1600/1600/600 updates ending at global updates 1600/3200/3800, 1945600 rows", bool(ok))
FIRED_Q, FIRED_SEEDS = [], []
for q in QS:
    for s in SEEDS["t31"]:
        d = C.r4_run_dir(q, s)
        sm = rj(d / "ms_run_summary.json")
        pt = sm["phase_timing"]
        ok = (pt["stage3"]["updates"], pt["stage2"]["updates"], pt["stage1"]["updates"]) == (1600, 1600, 600) \
            and (pt["stage3"]["global_exit"], pt["stage2"]["global_exit"], pt["stage1"]["global_exit"]) == (1600, 3200, 3800) \
            and sm["total_transitions"] == 3800 * NEP
        check(f"T3-R4 q={q} seed {s}: frozen-path ms_run_summary.json 1600/1600/600 updates, global updates 1600/3200/3800", bool(ok))
        rp = d / "retry.json"
        if rp.exists():
            rt = rj(rp)
            check(f"T3-R4 q={q} seed {s}: retry.json fired, frozen attempt 2, attempt 1 has 1600 updates, totals 5400 updates / 2764800 rows",
                  rt["fired"] is True and rt["frozen_attempt"] == 2 and rt["attempt1"]["counts"]["updates"] == 1600
                  and rt["attempt1"]["counts"]["transitions"] == 1600 * NEP and rt["totals"]["updates"] == 5400
                  and rt["totals"]["transitions"] == 5400 * NEP and rt["config"] == P31["settings"]["stage3_retry"])
            FIRED_Q.append(q)
            FIRED_SEEDS.append(s)
        else:
            check(f"T3-R4 q={q} seed {s}: no retry.json, so no retry fired (3800 updates)", sm["final_global_update"] == 3800)
check("T3-R4: the retry fired in 5 runs, all at q = 50 (seeds 80501, 80515, 80526, 80532, 80535)",
      sorted(zip(FIRED_Q, FIRED_SEEDS)) == [(50, 80501), (50, 80515), (50, 80526), (50, 80532), (50, 80535)])
A4 = C.load_per_run(C.ROOT / "results/t3_r4/confirmation/analysis/per_run.csv")
fired_a4 = sorted(zip(A4[A4.fired == True].q, A4[A4.fired == True].seed))  # noqa: E712
check("results/t3_r4/confirmation/analysis/per_run.csv (column fired) lists the same 5 runs; total_updates 5400 / 3800, total_transitions 2764800 / 1945600",
      fired_a4 == sorted(zip(FIRED_Q, FIRED_SEEDS))
      and set(A4[A4.fired == True].total_updates) == {5400} and set(A4[A4.fired == False].total_updates) == {3800}  # noqa: E712
      and set(A4[A4.fired == True].total_transitions) == {2764800} and set(A4[A4.fired == False].total_transitions) == {1945600})

# the evaluated policy: v2.0 exports; bw snapshots == exports (bit for bit)
import torch  # noqa: E402
torch.set_num_threads(1)


def arrays_equal(sd: Dict[str, Any], npz_path: Path) -> bool:
    z = np.load(npz_path, allow_pickle=True)
    return all(("actor." + k) in z and np.array_equal(z["actor." + k], v.detach().cpu().numpy()) for k, v in sd.items())


for q in QS:
    for s in SEEDS["v20"]:
        d = C.v20_run_dir(q, s)
        check(f"v2.0 q={q} seed {s}: weights/u01600.npz and weights/u02200.npz exist",
              (d / "weights" / "u01600.npz").is_file() and (d / "weights" / "u02200.npz").is_file())
for col, rd in (("t30", lambda q, s: C.r1_run_dir(q, s, "bw")), ("t31", C.r4_run_dir)):
    for q in QS:
        for s in SEEDS[col]:
            d = rd(q, s)
            st = torch.load(d / "state_end_stage1.pt", map_location="cpu", weights_only=False)
            ok = sorted(st["frozen_stages"]) == ["2", "3"] and arrays_equal(st["frozen_stages"]["3"], d / "weights" / "u01600.npz") \
                and arrays_equal(st["frozen_stages"]["2"], d / "weights" / "u03200.npz") \
                and arrays_equal(st["agent"]["actor"], d / "weights" / "u03800.npz")
            check(f"{COLNAME[col]} q={q} seed {s}: state_end_stage1.pt frozen_stages['3'] == weights/u01600, ['2'] == u03200, agent.actor == u03800 (bit for bit)", bool(ok))


# ================================================================================================ ROWS
def gsrc(col: str, field: str) -> str:
    """Source of a game field: protocol record (both q) + run configs."""
    return psrc(col, f"records.50.game.{field} and records.60.game.{field} (equal)") + "; " + csrc(col, f"record.game.{field}")


GA_ = "A model parameters"
ln_eps, txt_eps = code_line("envs/curriculum_env.py", "eps ~ U(-q, q) i.i.d. per player and stage")
ln_cost, txt_cost = code_line("envs/curriculum_env.py", "stage reward r_t = -k e_t^2")
ln_B, txt_B = code_line("envs/curriculum_env.py", "Maximal one-stage gap change")
ln_D, txt_D = code_line("envs/curriculum_env.py", "Half-width (t-1)B of the feasible gap domain D_t")
ln_xi, txt_xi = code_line("utils/ms_continuation.py", "z = eps_L - eps_O,  density")
m_xi = re.search(r"z = eps_L - eps_O with density .*? on \[-2q, 2q\]", P20["pipeline"]["continuation_table"]["definition"])
check("v2.0 continuation_table.definition contains the shock statement", m_xi is not None)
ENV = "envs/curriculum_env.py"

for col in COLS:
    P = PROTO[col][0]
    T_val = P["records"]["50"]["game"]["T"] if col == "v20" else P["settings"]["horizon"]
    add(GA_, "horizon T (number of stages)", col, "value", T_val,
        (psrc(col, "records.50.game.T") if col == "v20" else psrc(col, "settings.horizon")) + "; " + csrc(col, "record.game.T"))
    gm = P["records"]["50"]["game"]
    add(GA_, "cost parameter k", col, "value", float(gm["k"]), gsrc(col, "k"))
    add(GA_, "cost parameter k", col, "value as a fraction", "1/3500",
        "derived: stored k equals 1.0/3500.0 exactly (checked in scripts/T1.py, both q); k from " + gsrc(col, "k"))
    add(GA_, "effort cost per stage", col, "stage cost (quoted)", txt_cost.strip(), f"{ENV}:{ln_cost} (module docstring); k from " + gsrc(col, "k"))
    add(GA_, "prize w_H", col, "value", int(gm["w_h"]), gsrc(col, "w_h"))
    add(GA_, "prize w_L", col, "value", int(gm["w_l"]), gsrc(col, "w_l"))
    add(GA_, "prize spread ΔW = w_H - w_L", col, "value", int(P["records"]["50"]["dw"]),
        psrc(col, "records.50.dw and records.60.dw (equal)") + "; " + csrc(col, "record.dw"))
    check(f"{COLNAME[col]}: dw == w_h - w_l", P["records"]["50"]["dw"] == gm["w_h"] - gm["w_l"])
    add(GA_, "effort range", col, "lower bound e_min", int(gm["e_min"]), gsrc(col, "e_min"))
    add(GA_, "effort range", col, "upper bound e_max", int(gm["e_max"]), gsrc(col, "e_max"))
    add(GA_, "shock on a player's score (per player and stage)", col, "distribution (quoted)", txt_eps,
        f"{ENV}:{ln_eps} (module docstring)")
    if col == "v20":
        add(GA_, "shock on the score gap (difference of the two player shocks)", col, "distribution (quoted)", m_xi.group(0),
            psrc(col, "pipeline.continuation_table.definition") + " (substring)")
    else:
        add(GA_, "shock on the score gap (difference of the two player shocks)", col, "distribution (quoted)",
            re.sub(r"\s+", " ", re.search(r"z = eps_L - eps_O,\s+density .*? on \[-2q, 2q\]", txt_xi).group(0)),
            f"utils/ms_continuation.py:{ln_xi} (module docstring, white space collapsed); the T3 pipeline builds its tables with this density")
    for q in QS:
        add(GA_, "shock on the score gap (difference of the two player shocks)", col, f"support half-width 2q at q = {q}", 2 * q,
            "derived: 2*q with q from " + (psrc(col, "q_values") if col == "v20" else psrc(col, "confirmation.qs")))
    add(GA_, "B = (e_max - e_min) + 2q", col, "formula (quoted)", txt_B.strip('" '), f"{ENV}:{ln_B} (GameSpec.B docstring)")
    for q in QS:
        add(GA_, "B = (e_max - e_min) + 2q", col, f"B at q = {q}", int(P["records"][str(q)]["B"]),
            psrc(col, f"records.{q}.B") + "; " + csrc(col, "record.B"))
        check(f"{COLNAME[col]} q={q}: protocol B == e_max - e_min + 2q", P["records"][str(q)]["B"] == 100 + 2 * q)
    add(GA_, "state domain D_t = [-(t-1)B, (t-1)B]", col, "formula (quoted)", txt_D.strip('" '), f"{ENV}:{ln_D} (GameSpec.domain_half docstring)")
    for q in QS:
        spec = GameSpec(**CFGS[col][(q, SEEDS[col][0])]["record"]["game"])
        if q == QS[0]:
            check(f"{COLNAME[col]}: GameSpec.domain_half(1) == 0 (D_1 is the single state d = 0)", spec.domain_half(1) == 0.0)
            add(GA_, "state domain D_t = [-(t-1)B, (t-1)B]", col, "D_1", "single state d = 0 (half-width 0)",
                f"derived: GameSpec.domain_half(1) = 0 ({ENV}:{ln_D}), spec from " + csrc(col, "record.game"))
        for t in (2, 3):
            if t > spec.T:
                add(GA_, "state domain D_t = [-(t-1)B, (t-1)B]", col, f"D_{t} half-width at q = {q}", f"not applicable (T = {spec.T} has no stage {t})",
                    psrc(col, "records.50.game.T") if col == "v20" else psrc(col, "settings.horizon"))
                continue
            hw = float(spec.domain_half(t))
            src = f"derived: (t-1)*B = GameSpec.domain_half({t}) ({ENV}:{ln_D}), spec from " + csrc(col, "record.game") + " and " + csrc(col, "record.B", "all configs of that q agree")
            if col == "v20" and t == 2:
                check(f"v2.0 q={q}: domain_half(2) == protocol domain_half_stage2", hw == P20["records"][str(q)]["domain_half_stage2"])
                src += "; " + psrc(col, f"records.{q}.domain_half_stage2")
            add(GA_, "state domain D_t = [-(t-1)B, (t-1)B]", col, f"D_{t} half-width at q = {q}", int(hw), src)


# ---------------------------------------------------------------------------------------------- B q values and seed blocks
GB = "B q values and seed blocks"
R = C.ROOT
GC1 = C.load_per_run(R / "results/t3_r1/analysis/gate_counts.csv")
GC1_REL = "results/t3_r1/analysis/gate_counts.csv"
REH20 = C.load_per_run(R / "results/v2_T2_locked/rehearsal_v2_0_analysis/pass_counts.csv")
CONF20 = C.load_per_run(R / "results/v2_T2_locked/confirmation_v2_0_analysis/pass_counts.csv")
CONF20_V = rj(R / "results/v2_T2_locked/confirmation_v2_0_analysis/verdict.json")
PASS20_PROB = rj(R / "results/v2_T2_locked/v2_0/v2_0_pass_probability.json")
REH20_SAFE = rj(R / "results/v2_T2_locked/rehearsal_v2_0_analysis/rehearsal_safety.json")
CONF30 = C.load_per_run(R / "results/t3_r2/confirmation/analysis/pass_counts.csv")
CONF30_V = rj(R / "results/t3_r2/confirmation/analysis/verdict.json")
CNT3 = C.load_per_run(R / "results/t3_r3/dev/analysis/counts.csv")
HEAD3 = rj(R / "results/t3_r3/dev/analysis/headline.json")
CNT4 = C.load_per_run(R / "results/t3_r4/confirmation/analysis/counts.csv")
HEAD4 = rj(R / "results/t3_r4/confirmation/analysis/headline.json")
FAIL4 = C.load_per_run(R / "results/t3_r4/confirmation/analysis/failures.csv")
STAGE1_PR = pd.read_csv(R / "results/v2_refine/analysis/stage1_per_run.csv")

# ---- numbers read from the stored CSVs / JSON, cross-checked with the recomputation above
n20 = {q: int(CONF20[CONF20.q == q].n_pass.iloc[0]) for q in QS}
check("confirmation_v2_0_analysis/pass_counts.csv n_pass == recomputed 19, 20; n_expected 20", n20 == cnt["v20_conf"] and set(CONF20.n_expected) == {20})
r20 = {q: int(REH20[REH20.q == q].n_pass.iloc[0]) for q in QS}
check("rehearsal_v2_0_analysis/pass_counts.csv: 10 of 10 pass at each q", r20 == {50: 10, 60: 10} and set(REH20.n_expected) == {10})
g30 = {q: GC1[(GC1.arm == "bw") & (GC1.q == q)].iloc[0] for q in QS}
check("results/t3_r1/analysis/gate_counts.csv: bw run_pass 18 and 18 of 20 (== recomputed)", {q: int(g30[q].run_pass) for q in QS} == cnt["bw_r1"] and all(int(g30[q].n_planned) == 20 for q in QS))
p30 = CONF30.iloc[0]
check("results/t3_r2/confirmation/analysis/pass_counts.csv: 16, 18, total 34 of 40 (== recomputed)",
      (int(p30.pass_q50), int(p30.pass_q60), int(p30.pass_total), int(p30.n_planned_total)) == (16, 18, 34, 40) and {50: 16, 60: 18} == cnt["bw_r2"])
c3 = {a: {q: int(CNT3[(CNT3.arm == a) & (CNT3.q.astype(str) == str(q))]["pass"].iloc[0]) for q in QS} for a in ("bw", "bwA", "bwAB")}
check("results/t3_r3/dev/analysis/counts.csv == recomputed counts of the three arms", all(c3[a] == cnt[f"r3_{a}"] for a in c3))
c4 = {q: int(CNT4[CNT4.q.astype(str) == str(q)]["pass"].iloc[0]) for q in QS}
c4all = int(CNT4[CNT4.q.astype(str) == "both"]["pass"].iloc[0])
check("results/t3_r4/confirmation/analysis/counts.csv: 40, 39, total 79 of 80 (== recomputed)", c4 == cnt["bwA_r4"] == {50: 40, 60: 39} and c4all == 79 and set(CNT4.runs) == {40, 80})
check("headline.json of T3-R4: 79 of 80 pass, first_attempt_pass 74, fired 5, fired_runs_that_pass 5",
      (HEAD4["counts"]["pass_total"], HEAD4["trigger"]["first_attempt_pass"], HEAD4["trigger"]["fired"], HEAD4["trigger"]["fired_runs_that_pass"]) == (79, 74, 5, 5))
FAIL4_PARTS = failing_parts(PR4, 3)
FIRST_ATTEMPT = 80 - len(set(FAIL4_PARTS) | set(zip(FIRED_Q, FIRED_SEEDS)))
check("first-attempt count recomputed: 80 runs minus (runs that failed or fired) = 74", FIRST_ATTEMPT == 74 == HEAD4["trigger"]["first_attempt_pass"],
      f"(failed {sorted(FAIL4_PARTS)}, fired {sorted(zip(FIRED_Q, FIRED_SEEDS))})")

# ---- rows
for col in COLS:
    P = PROTO[col][0]
    qv = P["q_values"] if col == "v20" else P["confirmation"]["qs"]
    add(GB, "q values", col, "values (all blocks of the protocol)", ", ".join(str(x) for x in qv), psrc(col, "q_values" if col == "v20" else "confirmation.qs"))

# role of each block in the terms of the spec (development, comparison, confirmation block) --------------------------------
GRL = "role of the block (spec terms)"
GBS = "development or comparison block: seeds"
SPEC_REL = "paper/multistage_rev/pi_record/30_paper_figures_prompt.md"
ln_sp1, _ = code_line(SPEC_REL, "| 50501-50520 | T3-R1 |")
ln_sp3, _ = code_line(SPEC_REL, "| 70501-70520 | T3-R3 | development |")
ln_spF5, _ = code_line(SPEC_REL, "**F5. Method comparison** (T3-R1, paired).")
ln_spT4, _ = code_line(SPEC_REL, "**The T3-R1 comparison,** per q:")
ln_r1w, txt_r1w = code_line("reports/t3/r1/02_preregistration.md", "Wave `t3` (T = 3): seeds 50501-50520 at both q, arms `bw`, `jt`, `jt_es`")
check("T3-R1 pre-registration: wave t3 has seeds 50501-50520, arms bw, jt, jt_es, 'paired by (q, seed)'", "paired by (q, seed)" in txt_r1w)
ln_r3t, _ = code_line("reports/t3/r3/04_preregistration.md", "# T3-R3, P2: pre-registration of the development wave")
ln_r3s, _ = code_line("reports/t3/r3/04_preregistration.md", "**Seeds.** 70501-70520 at q = 50 and q = 60, the same seeds in every arm")
rat30 = P30["confirmation"]["rule"]["rationale"]
rnd30 = P30["confirmation"]["reported_not_deciding"]
i30 = [i_ for i_, s_ in enumerate(rnd30) if "T3-R1's 40 development runs" in s_]
check("protocols/v2_T3_locked_v1_0.json calls the T3-R1 runs its development runs: confirmation.rule.rationale and confirmation.reported_not_deciding[3]",
      "T3-R1's development pass rate" in rat30 and i30 == [3])
rnd31 = P31["confirmation"]["reported_not_deciding"]
i31 = [i_ for i_, s_ in enumerate(rnd31) if "bwA's 40 development runs of T3-R3" in s_]
check("protocols/v2_T3_locked_v1_1.json calls the T3-R3 runs of bwA development runs: confirmation.reported_not_deciding[5]", i31 == [5])
check("round of the confirmation of T3-1.0 is t3_r2 and of T3-1.1 is t3_r4 (protocol key round)", P30["round"] == "t3_r2" and P31["round"] == "t3_r4")
sb30_, sb31_ = P30["confirmation"]["seed_block"], P31["confirmation"]["seed_block"]
cmp_a, cmp_b = int(BW1.seed.min()), int(BW1.seed.max())
dev3_a, dev3_b = int(PR3[PR3.arm == "bwA"].seed.min()), int(PR3[PR3.arm == "bwA"].seed.max())
ds_ = P20["development_seeds"]
add(GB, GRL, "v20", "development or comparison block", f"development block, seeds {ds_[0]}-{ds_[-1]}, as recorded in its protocol (key development_seeds)",
    psrc("v20", "development_seeds") + "; " + psrc("v20", "confirmation.rehearsal.seeds") + f" (quoted: {P20['confirmation']['rehearsal']['seeds']})")
add(GB, GRL, "v20", "confirmation block", f"confirmation block, seeds {P20['confirmation']['seed_block'][0]}-{P20['confirmation']['seed_block'][-1]} (the v2.0 confirmation)",
    psrc("v20", "confirmation.seed_block") + "; " + psrc("v20", "confirmation.seed_block_status") + f" (quoted: {P20['confirmation']['seed_block_status']})")
add(GB, GRL, "t30", "development or comparison block",
    f"comparison block of round T3-R1, seeds {cmp_a}-{cmp_b}: bw, jt and jt_es on the same seeds, paired by (q, seed)",
    f"{SPEC_REL}:{ln_sp1} (seed-block table, row 50501-50520: T3-R1, bw, jt, jt_es (paired)), :{ln_spF5} (F5 'Method comparison (T3-R1, paired)') and :{ln_spT4} ('The T3-R1 comparison'); "
    f"reports/t3/r1/02_preregistration.md:{ln_r1w} (wave t3: seeds 50501-50520, arms bw, jt, jt_es, 'paired by (q, seed)'); "
    + fsrc(C.PER_RUN_R1, "arm == bw, column seed (both q); operation: min and max") + "; " + GC1_REL + " (column arm, horizon == 3)")
add(GB, GRL, "t30", "wording of the protocol file (quoted)",
    f"the protocol file itself calls these runs its development runs: confirmation.rule.rationale = \"{rat30}\"; confirmation.reported_not_deciding[3] = \"{rnd30[3]}\"",
    psrc("t30", "confirmation.rule.rationale") + "; " + psrc("t30", "confirmation.reported_not_deciding[3]"))
add(GB, GRL, "t30", "confirmation block", f"confirmation block of round T3-R2, seeds {sb30_[0]}-{sb30_[1]} (the confirmation of T3-1.0)",
    psrc("t30", "confirmation.seed_block") + "; " + psrc("t30", "round") + f" (= {P30['round']})")
add(GB, GRL, "t31", "development or comparison block",
    f"development block of round T3-R3, seeds {dev3_a}-{dev3_b}: bw, bwA and bwAB on the same seeds",
    f"{SPEC_REL}:{ln_sp3} (seed-block table, row 70501-70520: T3-R3, development); reports/t3/r3/04_preregistration.md:{ln_r3t} (title: 'P2: pre-registration of the development wave') "
    f"and :{ln_r3s} (seeds 70501-70520, the same seeds in every arm); " + fsrc(R / "results/t3_r3/dev/eval/per_run.csv", "arm == bwA, column seed (both q); operation: min and max")
    + "; " + fsrc(R / "results/t3_r3/dev/analysis/counts.csv", "column arm"))
add(GB, GRL, "t31", "wording of the protocol file (quoted)",
    "the protocol file names the runs of bwA on these seeds development runs: confirmation.reported_not_deciding[5] (substring) = \"bwA's 40 development runs of T3-R3\"",
    psrc("t31", "confirmation.reported_not_deciding[5] (substring)"))
add(GB, GRL, "t31", "confirmation block", f"confirmation block of round T3-R4, seeds {sb31_[0]}-{sb31_[1]} (the confirmation of T3-1.1)",
    psrc("t31", "confirmation.seed_block") + "; " + psrc("t31", "round") + f" (= {P31['round']})")

# development or comparison block ---------------------------------------------------------------------------------------------
ds = P20["development_seeds"]
check("v2.0 development_seeds are 10501-10510", ds == list(range(10501, 10511)))
add(GB, GBS, "v20", "first seed", ds[0], psrc("v20", "development_seeds"))
add(GB, GBS, "v20", "last seed", ds[-1], psrc("v20", "development_seeds"))
add(GB, GBS, "v20", "runs per q (same seeds at both q)", len(ds),
    psrc("v20", "development_seeds") + " (length); " + psrc("v20", "confirmation.rehearsal.q_values"))
add(GB, GBS, "v20", "what ran", "rehearsal_v2_0 (the locked pipeline) and the R1 stage-1 pilot (8 arms incl. B_base and B_expcont)",
    psrc("v20", "confirmation.rehearsal.root") + "; " + fsrc(R / "results/v2_refine/analysis/stage1_per_run.csv", "columns arm, seed (8 arms, seeds 10501-10510, both q)"))
check("stage1_per_run.csv: 8 arms x 2 q x 10 seeds 10501-10510, B_expcont among them",
      STAGE1_PR.arm.nunique() == 8 and "B_expcont" in set(STAGE1_PR.arm) and sorted(STAGE1_PR.seed.unique()) == ds and len(STAGE1_PR) == 160)
for col, df, a, b, desc, src in (
        ("t30", BW1, 50501, 50520, "T3-R1: bw, jt, jt_es (paired, same seeds)", fsrc(C.PER_RUN_R1, "arm == bw, column seed (both q)")),
        ("t31", PR3[PR3.arm == "bwA"], 70501, 70520, "T3-R3: bw, bwA, bwAB (same seeds)", fsrc(R / "results/t3_r3/dev/eval/per_run.csv", "arm == bwA, column seed (both q)"))):
    check(f"{COLNAME[col]} development or comparison block seeds {a}-{b}, 20 per q", all(sorted(df[df.q == q].seed) == list(range(a, b + 1)) for q in QS))
    add(GB, GBS, col, "first seed", a, src + "; operation: min over the 20 rows of each q")
    add(GB, GBS, col, "last seed", b, src + "; operation: max over the 20 rows of each q")
    add(GB, GBS, col, "runs per q (same seeds at both q)", 20, src + "; operation: count of rows per q")
    add(GB, GBS, col, "what ran", desc,
        (GC1_REL + " (column arm, horizon == 3)" if col == "t30" else fsrc(R / "results/t3_r3/dev/analysis/counts.csv", "column arm")))
check("gate_counts.csv horizon-3 arms are bw, jt, jt_es; counts.csv arms are bw, bwA, bwAB",
      set(GC1[GC1.horizon == 3].arm) == {"bw", "jt", "jt_es"} and set(CNT3.arm) == {"bw", "bwA", "bwAB"})

GBP = "development or comparison block: run-pass count (the protocol's method)"
for q in QS:
    add(GB, GBP, "v20", f"runs passing, q = {q}", r20[q], fsrc(R / "results/v2_T2_locked/rehearsal_v2_0_analysis/pass_counts.csv", f"column n_pass, row q = {q}"))
    add(GB, GBP, "v20", f"runs planned, q = {q}", int(REH20[REH20.q == q].n_expected.iloc[0]), fsrc(R / "results/v2_T2_locked/rehearsal_v2_0_analysis/pass_counts.csv", f"column n_expected, row q = {q}"))
    add(GB, GBP, "t30", f"runs passing, q = {q}", int(g30[q].run_pass), fsrc(R / GC1_REL, f"column run_pass, row arm = bw, q = {q}"))
    add(GB, GBP, "t30", f"runs planned, q = {q}", int(g30[q].n_planned), fsrc(R / GC1_REL, f"column n_planned, row arm = bw, q = {q}"))
    add(GB, GBP, "t31", f"runs passing, q = {q}", c3["bwA"][q], fsrc(R / "results/t3_r3/dev/analysis/counts.csv", f"column pass, row arm = bwA, q = {q}"))
    add(GB, GBP, "t31", f"runs planned, q = {q}", int(CNT3[(CNT3.arm == "bwA") & (CNT3.q.astype(str) == str(q))].runs.iloc[0]), fsrc(R / "results/t3_r3/dev/analysis/counts.csv", f"column runs, row arm = bwA, q = {q}"))
add(GB, GBP, "v20", "runs passing, both q", sum(r20.values()), "derived: sum of the two q counts above")
add(GB, GBP, "v20", "runs planned, both q", 20, "derived: sum of the two q counts above")
add(GB, GBP, "t30", "runs passing, both q", sum(int(g30[q].run_pass) for q in QS), "derived: sum of the two q counts above")
add(GB, GBP, "t30", "runs planned, both q", 40, "derived: sum of the two q counts above")
add(GB, GBP, "t31", "runs passing, both q", sum(c3["bwA"].values()),
    fsrc(R / "results/t3_r3/dev/analysis/counts.csv", "column pass, rows arm = bwA, q = 50 and 60; operation: sum"))
add(GB, GBP, "t31", "runs planned, both q", 40, "derived: sum of the two q counts above")
check("T3-R3 counts.csv row 'both' for bwA is 39 of 40", int(CNT3[(CNT3.arm == "bwA") & (CNT3.q == "both")]["pass"].iloc[0]) == 39 == sum(c3["bwA"].values()))

# other arms run on the same development seeds
GBO = "development or comparison block: other arms on the same seeds"
add(GB, GBO, "v20", "arm B_expcont (adopted in v2.0): runs with stage1_rel_err_abs <= 0.05, q = 50",
    int(PASS20_PROB["per_q"]["50"]["n_within_threshold"]), fsrc(R / "results/v2_T2_locked/v2_0/v2_0_pass_probability.json", "per_q.50.n_within_threshold"))
add(GB, GBO, "v20", "arm B_expcont (adopted in v2.0): runs planned, q = 50", int(PASS20_PROB["per_q"]["50"]["n"]), fsrc(R / "results/v2_T2_locked/v2_0/v2_0_pass_probability.json", "per_q.50.n"))
add(GB, GBO, "v20", "arm B_expcont (adopted in v2.0): runs with stage1_rel_err_abs <= 0.05, q = 60",
    int(PASS20_PROB["per_q"]["60"]["n_within_threshold"]), fsrc(R / "results/v2_T2_locked/v2_0/v2_0_pass_probability.json", "per_q.60.n_within_threshold"))
add(GB, GBO, "v20", "arm B_expcont (adopted in v2.0): runs planned, q = 60", int(PASS20_PROB["per_q"]["60"]["n"]), fsrc(R / "results/v2_T2_locked/v2_0/v2_0_pass_probability.json", "per_q.60.n"))
check("v2_0_pass_probability.json: arm B_expcont, source stage1_per_run.csv, 10 of 10 within the G-S threshold at both q",
      PASS20_PROB["arm"] == "B_expcont" and PASS20_PROB["threshold"] == 0.05 and all(PASS20_PROB["per_q"][str(q)]["n_within_threshold"] == 10 == PASS20_PROB["per_q"][str(q)]["n"] for q in QS))
for arm in ("jt", "jt_es"):
    for q in QS:
        r = GC1[(GC1.arm == arm) & (GC1.q == q)].iloc[0]
        add(GB, GBO, "t30", f"arm {arm}: runs passing, q = {q}", int(r.run_pass), fsrc(R / GC1_REL, f"column run_pass, row arm = {arm}, q = {q}"))
        add(GB, GBO, "t30", f"arm {arm}: runs planned, q = {q}", int(r.n_planned), fsrc(R / GC1_REL, f"column n_planned, row arm = {arm}, q = {q}"))
for arm in ("bw", "bwAB"):
    for q in QS:
        r = CNT3[(CNT3.arm == arm) & (CNT3.q.astype(str) == str(q))].iloc[0]
        add(GB, GBO, "t31", f"arm {arm}: runs passing, q = {q}", c3[arm][q], fsrc(R / "results/t3_r3/dev/analysis/counts.csv", f"column pass, row arm = {arm}, q = {q}"))
        add(GB, GBO, "t31", f"arm {arm}: runs planned, q = {q}", int(r.runs), fsrc(R / "results/t3_r3/dev/analysis/counts.csv", f"column runs, row arm = {arm}, q = {q}"))

# development-block reading rule -------------------------------------------------------------------------------------
GBR = "development or comparison block: rule applied to its outcome"
sc = P20["confirmation"]["rehearsal"]["safety_condition"]
add(GB, GBR, "v20", "minimum G-S passes, pooled over both q (of 20 runs)", int(sc["G-S_pass_at_least_of_20"]), psrc("v20", "confirmation.rehearsal.safety_condition.G-S_pass_at_least_of_20"))
add(GB, GBR, "v20", "minimum normal-model probability per q", float(sc["normal_model_probability_at_least_per_q"]), psrc("v20", "confirmation.rehearsal.safety_condition.normal_model_probability_at_least_per_q"))
add(GB, GBR, "v20", "observed G-S passes, pooled over both q (of 20 runs)", int(REH20_SAFE["pooled_n_G-S_pass"]),
    fsrc(R / "results/v2_T2_locked/rehearsal_v2_0_analysis/rehearsal_safety.json", "pooled_n_G-S_pass"))
for q in QS:
    add(GB, GBR, "v20", f"observed normal-model probability, q = {q}", float(REH20_SAFE["per_q"][str(q)]["P_ge_k_of_n"]),
        fsrc(R / "results/v2_T2_locked/rehearsal_v2_0_analysis/rehearsal_safety.json", f"per_q.{q}.P_ge_k_of_n"))
rr = CRIT["reading_rule"]
check("prereg_criteria.json reading_rule: arm bw, horizon 3, min_passes_of_20_at_each_q 18", rr["arm"] == "bw" and rr["horizon"] == 3 and rr["min_passes_of_20_at_each_q"] == 18)
add(GB, GBR, "t30", "minimum passes of 20 at each q (arm bw)", int(rr["min_passes_of_20_at_each_q"]), fsrc(R / CRIT_REL, "reading_rule.min_passes_of_20_at_each_q"))
add(GB, GBR, "t30", "rule met by the block", "yes" if all(cnt["bw_r1"][q] >= rr["min_passes_of_20_at_each_q"] for q in QS) else "no",
    "derived: bw run_pass counts of " + GC1_REL + " (18 and 18) against reading_rule.min_passes_of_20_at_each_q of " + CRIT_REL)
ln_tg, txt_tg = code_line("reports/t3/r3/04_preregistration.md", "A candidate arm reaches it iff at least")
m_tg = re.search(r"at least (\d+) of its (\d+) runs pass and at least (\d+) of (\d+) at each q", txt_tg)
check("T3-R3 pre-registration states the target 'at least 38 of its 40 runs pass and at least 18 of 20 at each q'", m_tg is not None and m_tg.groups() == ("38", "40", "18", "20"))
add(GB, GBR, "t31", "target: minimum passes of 40 (arm bwA)", int(m_tg.group(1)), f"reports/t3/r3/04_preregistration.md:{ln_tg} (section 6, 'The target (the PI's)')")
add(GB, GBR, "t31", "target: minimum passes of 20 at each q (arm bwA)", int(m_tg.group(3)), f"reports/t3/r3/04_preregistration.md:{ln_tg} (section 6, 'The target (the PI's)')")
add(GB, GBR, "t31", "target reached by arm bwA", "yes" if HEAD3["targets"]["bwA"]["reaches_target"] else "no", fsrc(R / "results/t3_r3/dev/analysis/headline.json", "targets.bwA.reaches_target"))
check("T3-R3 headline.json targets.bwA reaches_target true, bw false", HEAD3["targets"]["bwA"]["reaches_target"] is True and HEAD3["targets"]["bw"]["reaches_target"] is False)

# confirmation block --------------------------------------------------------------------------------------------------
GBC = "confirmation block: seeds"
sb20 = P20["confirmation"]["seed_block"]
check("v2.0 confirmation.seed_block is 30501-30520", sb20 == list(range(30501, 30521)) and P20["confirmation"]["n_seeds"] == 20 and P20["confirmation"]["seeds_used_for_both_q"] is True)
add(GB, GBC, "v20", "first seed", sb20[0], psrc("v20", "confirmation.seed_block"))
add(GB, GBC, "v20", "last seed", sb20[-1], psrc("v20", "confirmation.seed_block"))
add(GB, GBC, "v20", "runs per q (same seeds at both q)", int(P20["confirmation"]["n_seeds"]), psrc("v20", "confirmation.n_seeds") + "; " + psrc("v20", "confirmation.seeds_used_for_both_q"))
add(GB, GBC, "v20", "runs in total", 2 * int(P20["confirmation"]["n_seeds"]), "derived: 2 q values x confirmation.n_seeds of " + P20_REL)
for col in ("t30", "t31"):
    P = PROTO[col][0]
    sb = P["confirmation"]["seed_block"]
    check(f"{COLNAME[col]} confirmation.seed_block is [first, last] with same_seeds_at_both_q",
          sb == ([60501, 60520] if col == "t30" else [80501, 80540]) and P["confirmation"]["same_seeds_at_both_q"] is True)
    add(GB, GBC, col, "first seed", sb[0], psrc(col, "confirmation.seed_block[0]"))
    add(GB, GBC, col, "last seed", sb[1], psrc(col, "confirmation.seed_block[1]"))
    add(GB, GBC, col, "runs per q (same seeds at both q)", (sb[1] - sb[0] + 1), "derived: seed_block[1] - seed_block[0] + 1 of " + PROTO[col][1] + " (consecutive seeds; same_seeds_at_both_q = true)")
    add(GB, GBC, col, "runs in total", int(P["confirmation"]["n_runs"]), psrc(col, "confirmation.n_runs"))
    check(f"{COLNAME[col]} confirmation.n_runs == 2 x runs per q", P["confirmation"]["n_runs"] == 2 * (sb[1] - sb[0] + 1))

GBN = "confirmation block: run-pass count"
for q in QS:
    add(GB, GBN, "v20", f"runs passing, q = {q}", n20[q], fsrc(R / "results/v2_T2_locked/confirmation_v2_0_analysis/pass_counts.csv", f"column n_pass, row q = {q}"))
    add(GB, GBN, "v20", f"runs planned, q = {q}", int(CONF20[CONF20.q == q].n_expected.iloc[0]), fsrc(R / "results/v2_T2_locked/confirmation_v2_0_analysis/pass_counts.csv", f"column n_expected, row q = {q}"))
    add(GB, GBN, "t30", f"runs passing, q = {q}", int(p30[f"pass_q{q}"]), fsrc(R / "results/t3_r2/confirmation/analysis/pass_counts.csv", f"column pass_q{q}"))
    add(GB, GBN, "t30", f"runs planned, q = {q}", int(p30[f"n_planned_q{q}"]), fsrc(R / "results/t3_r2/confirmation/analysis/pass_counts.csv", f"column n_planned_q{q}"))
    add(GB, GBN, "t31", f"runs passing, q = {q}", c4[q], fsrc(R / "results/t3_r4/confirmation/analysis/counts.csv", f"column pass, row q = {q}"))
    add(GB, GBN, "t31", f"runs planned, q = {q}", int(CNT4[CNT4.q.astype(str) == str(q)].runs.iloc[0]), fsrc(R / "results/t3_r4/confirmation/analysis/counts.csv", f"column runs, row q = {q}"))
add(GB, GBN, "v20", "runs passing, both q", sum(n20.values()), "derived: sum of the two q counts above (" + fsrc(R / "results/v2_T2_locked/confirmation_v2_0_analysis/pass_counts.csv", "column n_pass") + ")")
add(GB, GBN, "v20", "runs planned, both q", 40, "derived: sum of the two q counts above")
add(GB, GBN, "t30", "runs passing, both q", int(p30.pass_total), fsrc(R / "results/t3_r2/confirmation/analysis/pass_counts.csv", "column pass_total"))
add(GB, GBN, "t30", "runs planned, both q", int(p30.n_planned_total), fsrc(R / "results/t3_r2/confirmation/analysis/pass_counts.csv", "column n_planned_total"))
add(GB, GBN, "t31", "runs passing, both q", c4all, fsrc(R / "results/t3_r4/confirmation/analysis/counts.csv", "column pass, row q = both"))
add(GB, GBN, "t31", "runs planned, both q", int(CNT4[CNT4.q.astype(str) == "both"].runs.iloc[0]), fsrc(R / "results/t3_r4/confirmation/analysis/counts.csv", "column runs, row q = both"))

GBU = "confirmation block: pass rule of the protocol"
m_r20 = re.search(r"at least (\d+) of (\d+) runs pass", P20["confirmation"]["pass_rule"])
check("v2.0 confirmation.pass_rule: 'for each q, at least 18 of 20 runs pass'", m_r20 is not None and m_r20.groups() == ("18", "20"))
add(GB, GBU, "v20", "minimum passes per q (of 20 runs)", int(m_r20.group(1)), psrc("v20", "confirmation.pass_rule") + " (number read from the text)")
add(GB, GBU, "v20", "rule text (quoted)", P20["confirmation"]["pass_rule"], psrc("v20", "confirmation.pass_rule"))
rl30, rl31 = P30["confirmation"]["rule"], P31["confirmation"]["rule"]
add(GB, GBU, "t30", "minimum passes in total (of 40 runs)", int(rl30["min_total_passes"]), psrc("t30", "confirmation.rule.min_total_passes"))
add(GB, GBU, "t30", "minimum passes per q (of 20 runs)", int(rl30["min_passes_per_q"]), psrc("t30", "confirmation.rule.min_passes_per_q"))
add(GB, GBU, "t30", "rule text (quoted)", rl30["text"], psrc("t30", "confirmation.rule.text"))
add(GB, GBU, "t31", "maximum failures in total (of 80 runs)", int(rl31["max_failures_total"]), psrc("t31", "confirmation.rule.max_failures_total"))
add(GB, GBU, "t31", "maximum failures per q (of 40 runs)", int(rl31["max_failures_per_q"]), psrc("t31", "confirmation.rule.max_failures_per_q"))
add(GB, GBU, "t31", "rule text (quoted)", rl31["text"], psrc("t31", "confirmation.rule.text"))

GBV = "confirmation block: verdict"
add(GB, GBV, "v20", "verdict", CONF20_V["overall"], fsrc(R / "results/v2_T2_locked/confirmation_v2_0_analysis/verdict.json", "overall"))
add(GB, GBV, "v20", "rule (quoted)", CONF20_V["rule"], fsrc(R / "results/v2_T2_locked/confirmation_v2_0_analysis/verdict.json", "rule"))
add(GB, GBV, "t30", "verdict", CONF30_V["verdict"]["verdict"], fsrc(R / "results/t3_r2/confirmation/analysis/verdict.json", "verdict.verdict"))
add(GB, GBV, "t30", "reason (quoted)", CONF30_V["verdict"]["reason"], fsrc(R / "results/t3_r2/confirmation/analysis/verdict.json", "verdict.reason"))
add(GB, GBV, "t31", "verdict", HEAD4["verdict"]["verdict"], fsrc(R / "results/t3_r4/confirmation/analysis/headline.json", "verdict.verdict"))
add(GB, GBV, "t31", "reason (quoted)", HEAD4["verdict"]["reason"], fsrc(R / "results/t3_r4/confirmation/analysis/headline.json", "verdict.reason"))
check("verdicts: v2.0 PASS, T3-1.0 fail, T3-1.1 pass", (CONF20_V["overall"], CONF30_V["verdict"]["verdict"], HEAD4["verdict"]["verdict"]) == ("PASS", "fail", "pass"))
check("T3-R2 and T3-R4 infra_events are empty (no run replaced)",
      rj(R / "results/t3_r2/confirmation/infra_events.json")["events"] == [] and rj(R / "results/t3_r4/confirmation/infra_events.json")["events"] == [])

GBF = "confirmation block: failed runs and failing gates"
FP = {"v20": failing_parts(V20, 2), "t30": failing_parts(PR2, 3), "t31": FAIL4_PARTS}
check("failed runs: v2.0 only (q=50, seed 30510); T3-1.0 six runs; T3-1.1 only (q=60, seed 80502)",
      list(FP["v20"]) == [(50, 30510)] and len(FP["t30"]) == 6 and list(FP["t31"]) == [(60, 80502)], str({k: list(v) for k, v in FP.items()}))
check("T3-R4 failures.csv: seed 80502 fails gate_G_F;gate_G_N at stage 2",
      list(FAIL4.run) == ["t3r4_q60_s80502_bwA"] and FAIL4.failing_gates.iloc[0] == "gate_G_F;gate_G_N" and float(FAIL4.Gmax_full_stage.iloc[0]) == 2.0
      and FP["t31"][(60, 80502)] == "G-F (Gmax_full_over_dw); G-N (Gmax_full_over_dw)")
F2 = pd.read_csv(R / "results/t3_r2/confirmation/analysis/failures.csv")
F2A = F2[F2.failing.str.startswith("all failing parts:")]
stored_parts = {(int(r.q), int(r.seed)): [x.strip() for x in r.failing.split(":", 1)[1].split(",")] for r in F2A.itertuples()}
mapped = {(q_, s_): [t_.replace(" (", " ").replace(")", "").replace("eta_T_over_dw", "eta_3").replace("rmse_pos_T_over_eT0", "rmse_pos").replace("tail_mean_T_over_eT0", "tail_mean").replace("Gmax_full_over_dw", "Gmax_full")
                     for t_ in v_.split("; ")] for (q_, s_), v_ in FP["t30"].items()}
check("T3-R2 failing runs and failing gate parts recomputed == results/t3_r2/confirmation/analysis/failures.csv ('all failing parts')", mapped == stored_parts, str((mapped, stored_parts))[:400])
PRSRC = {"v20": (C.PER_RUN_R1, "arm == v20"), "t30": (R / "results/t3_r2/confirmation/eval/per_run.csv", "all rows"), "t31": (C.PER_RUN_R4, "all rows")}
for col in COLS:
    for (q, s), parts in FP[col].items():
        add(GB, GBF, col, f"failed run: q = {q}, seed {s}", parts,
            fsrc(PRSRC[col][0], f"{PRSRC[col][1]}; failing gate parts recomputed from the stored columns eta_T_over_dw, rmse_pos_T_over_eT0, tail_mean_T_over_eT0, "
                                f"Gmax_full_over_dw, dev_* and (T = 2) stage1_rel_err_abs against the thresholds of {CRIT_REL}"))

GBA = "first-attempt count (verifier-gated restart counted as a failed first attempt)"
for col in ("v20", "t30"):
    add(GB, GBA, col, "value", "not applicable (no restart rule)", psrc(col, "settings") if col == "t30" else psrc(col, "pipeline") + " (no stage3_retry key; the restart is defined for T = 3 only: run/ms_retry.py validate)")
add(GB, GBA, "t31", "runs passing on the first attempt", FIRST_ATTEMPT, fsrc(R / "results/t3_r4/confirmation/analysis/headline.json", "trigger.first_attempt_pass"))
add(GB, GBA, "t31", "runs planned", int(HEAD4["trigger"]["n"]), fsrc(R / "results/t3_r4/confirmation/analysis/headline.json", "trigger.n"))
add(GB, GBA, "t31", "runs in which the restart fired, q = 50", sum(1 for q_ in FIRED_Q if q_ == 50), fsrc(R / "results/t3_r4/confirmation/analysis/fired.csv", "column q; operation: count of rows with q = 50"))
add(GB, GBA, "t31", "runs in which the restart fired, q = 60", sum(1 for q_ in FIRED_Q if q_ == 60), fsrc(R / "results/t3_r4/confirmation/analysis/fired.csv", "column q; operation: count of rows with q = 60"))
add(GB, GBA, "t31", "runs passing after the restart (final count, same confirmation block)", c4all, fsrc(R / "results/t3_r4/confirmation/analysis/headline.json", "counts.pass_total"))
FIRED_CSV = C.load_per_run(R / "results/t3_r4/confirmation/analysis/fired.csv")
check("fired.csv lists the 5 fired runs, all q = 50, all passing", sorted(zip(FIRED_CSV.q, FIRED_CSV.seed)) == sorted(zip(FIRED_Q, FIRED_SEEDS)) and set(FIRED_CSV.q) == {50}
      and bool(FIRED_CSV.passed.all()))

# ---- the round of every block row: the statistic of each row of group B ends with the round its numbers belong to
DEV_FAMILY = {GBS: "seeds", GBP: "pass", GBO: "arms", GBR: "rule"}
TAG_DEV = {"v20": {"seeds": "T = 2 round R1 and v2.0 rehearsal", "pass": "T = 2 v2.0 rehearsal", "arms": "T = 2 round R1", "rule": "T = 2 v2.0 rehearsal"},
           "t30": {k_: "round T3-R1" for k_ in DEV_FAMILY.values()}, "t31": {k_: "round T3-R3" for k_ in DEV_FAMILY.values()}}
TAG_CONF = {"v20": "T = 2 v2.0 confirmation", "t30": "round T3-R2", "t31": "round T3-R4"}
COLKEY = {v_: k_ for k_, v_ in COLNAME.items()}
n_tagged = 0
for r_ in ROWS:
    if r_["row_group"] != GB or r_["row"] in ("q values", GRL):
        continue
    ck_ = COLKEY[r_["column_or_q"]]
    if r_["row"] in DEV_FAMILY:
        tag_ = TAG_DEV[ck_][DEV_FAMILY[r_["row"]]]
    else:
        check(f"group B row {r_['row']!r} is a confirmation-block row or the first-attempt row", r_["row"].startswith("confirmation block:") or r_["row"] == GBA)
        tag_ = TAG_CONF[ck_]
    r_["statistic"] = f"{r_['statistic']} [{tag_}]"
    n_tagged += 1
check("every row of group B except 'q values' and the role row carries the round of its numbers in brackets", n_tagged > 0 and all(
    r_["statistic"].endswith("]") for r_ in ROWS if r_["row_group"] == GB and r_["row"] not in ("q values", GRL)))


# ---------------------------------------------------------------------------------------------- C training
GCt = "C training"
PH_A, PH_B = P20["pipeline"]["phase_A"], P20["pipeline"]["phase_B"]
check("v2.0 phases: A trains stage 2 alone, B trains stage 1 alone", PH_A["active_stages"] == [2] and PH_B["active_stages"] == [1])
COST4_REL = "results/t3_r4/confirmation/analysis/per_run.csv"
CV_REL, CB_REL = "results/t3_r1/analysis/cost.csv", "results/t3_r1/analysis/cost.csv"


def stage_rows(col: str, stage: int) -> Optional[Tuple[int, str, int, str]]:
    """(updates, source, policy rows, source) of a stage, or None if the protocol has no such stage."""
    if col == "v20":
        if stage == 3:
            return None
        ph, key = (PH_A, "phase_A") if stage == 2 else (PH_B, "phase_B")
        ur = psrc(col, f"pipeline.{key}.updates (active_stages = {ph['active_stages']})")
        cr = fsrc(R / CV_REL, f"column rows_t{stage}, arm == v20 (40 rows, all equal)")
        return int(ph["updates"]), ur, int(CV[f"rows_t{stage}"].iloc[0]), cr
    P = PROTO[col][0]
    ur = psrc(col, f"settings.budgets.{stage}")
    if col == "t30":
        cr = fsrc(R / CB_REL, f"column rows_t{stage}, arm == bw (40 rows, all equal)")
        return int(P["settings"]["budgets"][str(stage)]), ur, int(CB[f"rows_t{stage}"].iloc[0]), cr
    ub = psrc(col, f"settings.budgets.{stage}")
    cr = (f"derived: updates x policy rows per update. Updates: {ub} (the row 'updates per stage'), confirmed per stage by ms_run_summary.json phase_timing.stage{stage}.updates "
          f"in the 80 runs ({rel(C.r4_run_dir(50, 80501)).replace('q50', 'q{q}').replace('seed80501', 'seed{seed}')}/ms_run_summary.json). "
          f"Policy rows per update: {NEP} = record.protocol.episodes_per_update of the 80 run configs (the row 'policy rows per update'). "
          f"The column total_transitions of {COST4_REL} confirms only the per-run sum over the three stages (rows with fired == False: 1945600 = 819200 + 819200 + 307200), not any single stage")
    return int(P["settings"]["budgets"][str(stage)]), ur, int(P["settings"]["budgets"][str(stage)]) * NEP, cr


for col in COLS:
    for stage in (3, 2, 1):
        r = stage_rows(col, stage)
        if r is None:
            add(GCt, "updates per stage", col, f"stage {stage}", "not applicable (T = 2 has stages 1 and 2)", psrc(col, "records.50.game.T"))
            add(GCt, "policy rows per stage", col, f"stage {stage}", "not applicable (T = 2 has stages 1 and 2)", psrc(col, "records.50.game.T"))
            continue
        add(GCt, "updates per stage", col, f"stage {stage}", r[0], r[1])
        add(GCt, "policy rows per stage", col, f"stage {stage}", r[2], r[3])
        check(f"{COLNAME[col]} stage {stage}: policy rows == updates x 512", r[2] == r[0] * NEP)
    add(GCt, "episodes per update", col, "value", NEP, psrc(col, "records.50.protocol.episodes_per_update and records.60.protocol.episodes_per_update (equal)") + "; " + csrc(col, "record.protocol.episodes_per_update"))
    add(GCt, "policy rows per update", col, "value", NEP,
        {"v20": fsrc(R / "results/t3_r1/analysis/cost.csv", "rows_t2 / updates of phase A, rows_t1 / updates of phase B; and v2_updates.csv column n_policy_rows (every update) of the 40 runs"),
         "t30": fsrc(R / CB_REL, "rows_t3, rows_t2 / 1600 and rows_t1 / 600 (arm bw); and ms_run_summary.json total_transitions / final_global_update of the 40 runs"),
         "t31": f"derived: updates x episodes per update (one learner row per episode in a one-stage phase), as in {COST4_REL} (total_transitions / total_updates of the 80 runs)"}[col])
# totals
for col in COLS:
    if col == "v20":
        add(GCt, "total updates per run", col, "value", int(CV.updates.iloc[0]), fsrc(R / CV_REL, "column updates, arm == v20 (40 rows, all equal)"))
        add(GCt, "total policy rows per run", col, "value", int(CV.policy_rows_total.iloc[0]), fsrc(R / CV_REL, "column policy_rows_total, arm == v20 (40 rows, all equal)"))
    elif col == "t30":
        add(GCt, "total updates per run", col, "value", int(CB.updates.iloc[0]), fsrc(R / CB_REL, "column updates, arm == bw (40 rows, all equal)"))
        add(GCt, "total policy rows per run", col, "value", int(CB.policy_rows_total.iloc[0]), fsrc(R / CB_REL, "column policy_rows_total, arm == bw (40 rows, all equal)"))
    else:
        nf = A4[A4.fired == False]  # noqa: E712
        ff = A4[A4.fired == True]   # noqa: E712
        add(GCt, "total updates per run", col, "run without a stage-3 restart", int(nf.total_updates.iloc[0]), fsrc(R / COST4_REL, "column total_updates, rows with fired == False (75 rows, all equal)"))
        add(GCt, "total policy rows per run", col, "run without a stage-3 restart", int(nf.total_transitions.iloc[0]), fsrc(R / COST4_REL, "column total_transitions, rows with fired == False (75 rows, all equal)"))
        add(GCt, "total updates per run", col, "run with a stage-3 restart", int(ff.total_updates.iloc[0]), fsrc(R / COST4_REL, "column total_updates, rows with fired == True (5 rows, all equal)") + "; " + psrc(col, "settings.retry.budget_of_a_fired_run.updates"))
        add(GCt, "total policy rows per run", col, "run with a stage-3 restart", int(ff.total_transitions.iloc[0]), fsrc(R / COST4_REL, "column total_transitions, rows with fired == True (5 rows, all equal)") + "; " + psrc(col, "settings.retry.budget_of_a_fired_run.policy_rows"))
        check("T3-1.1 run without a restart: 819200 + 819200 + 307200 == total_transitions of per_run.csv (the only per-run confirmation of the stage values)",
              1600 * NEP + 1600 * NEP + 600 * NEP == int(nf.total_transitions.iloc[0]) == 1945600)
        s3 = 2 * 1600 * NEP
        add(GCt, "policy rows per stage", col, "stage 3 in a run with a stage-3 restart (two attempts)", s3,
            "derived: attempt 1 (retry.json attempt1.counts.transitions) + attempt 2 (ms_run_summary.json phase_timing.stage3.updates x 512) of the 5 fired runs; "
            + f"{rel(C.r4_run_dir(50, 80501)).replace('q50', 'q{q}').replace('seed80501', 'seed{seed}')}/retry.json")
        check("T3-1.1 fired run: stage 3 rows 1638400 + 819200 + 307200 == 2764800 total", s3 + 1600 * NEP + 600 * NEP == int(ff.total_transitions.iloc[0]))
check("total updates/rows consistent with the stage budgets", int(CV.updates.iloc[0]) == 1600 + 600 and int(CB.updates.iloc[0]) == 1600 + 1600 + 600)

# learning-rate schedule
for col in COLS:
    if col == "v20":
        dec = {d["phase"]: d for d in P20["pipeline"]["lr_decay"]}
        starts, ends = {dec[k]["start_lr"] for k in dec}, {dec[k]["end_lr"] for k in dec}
        check("v2.0 lr: start 0.0003 / end 3e-05 in both phases", starts == {0.0003} and ends == {3e-05})
        add(GCt, "learning rate", col, "start value", float(next(iter(starts))), psrc(col, "pipeline.lr_decay[*].start_lr (phases A and B)"))
        add(GCt, "learning rate", col, "end value", float(next(iter(ends))), psrc(col, "pipeline.lr_decay[*].end_lr (phases A and B)"))
        add(GCt, "learning rate", col, "stage 3 decay window (local updates)", "not applicable (T = 2 has stages 1 and 2)", psrc(col, "records.50.game.T"))
        add(GCt, "learning rate", col, "stage 2 decay window (local updates)", f"{dec['A']['local_first']}-{dec['A']['local_last']}", psrc(col, "pipeline.lr_decay[phase A].local_first and .local_last"))
        add(GCt, "learning rate", col, "stage 1 decay window (local updates)", f"{dec['B']['local_first']}-{dec['B']['local_last']}", psrc(col, "pipeline.lr_decay[phase B].local_first and .local_last"))
        add(GCt, "learning rate", col, "schedule in phase A (quoted)", PH_A["lr"], psrc(col, "pipeline.phase_A.lr"))
        add(GCt, "learning rate", col, "schedule in phase B (quoted)", PH_B["lr"], psrc(col, "pipeline.phase_B.lr"))
        add(GCt, "learning rate", col, "form (quoted)", P20["pipeline"]["lr_form"], psrc(col, "pipeline.lr_form"))
    else:
        P = PROTO[col][0]
        lw = P["settings"]["lr_windows"]
        check(f"{COLNAME[col]} lr windows: one linear window per stage, 0.0003 -> 3e-05",
              all(len(lw[k]) == 1 and lw[k][0]["start"] == 0.0003 and lw[k][0]["end"] == 3e-05 for k in lw) and P["settings"]["lr"]["start"] == 0.0003 and P["settings"]["lr"]["end"] == 3e-05)
        add(GCt, "learning rate", col, "start value", float(P["settings"]["lr"]["start"]), psrc(col, "settings.lr.start"))
        add(GCt, "learning rate", col, "end value", float(P["settings"]["lr"]["end"]), psrc(col, "settings.lr.end"))
        for st in ("3", "2", "1"):
            add(GCt, "learning rate", col, f"stage {st} decay window (local updates)", f"{lw[st][0]['first']}-{lw[st][0]['last']}", psrc(col, f"settings.lr_windows.{st}[0].first and .last"))
        add(GCt, "learning rate", col, "schedule (quoted)", P["settings"]["lr"]["text"], psrc(col, "settings.lr.text"))
        sched = agree(col, lambda c: c["record"]["lr_schedule"], "record.lr_schedule")[50]
        add(GCt, "learning rate", col, "applied to (from the run config)",
            f"actor_and_critic = {str(sched['actor_and_critic']).lower()}; apply_before_each_update = {str(sched['apply_before_each_update']).lower()}; "
            f"constant_within_update = {str(sched['constant_within_update']).lower()}; preserve_adam_state = {str(sched['preserve_adam_state']).lower()}",
            csrc(col, "record.lr_schedule.{actor_and_critic, apply_before_each_update, constant_within_update, preserve_adam_state}"))

# start design
ln_root, txt_root = code_line("run/run_ms_stagewise.py", "d0 = np.zeros(n_ep)")
ln_bal, txt_bal = code_line("run/run_ms_stagewise.py", "d0 = self.sampler.balanced(t, n_ep, rng_start)")
ln_t1, txt_t1 = code_line("run/run_ms_stagewise.py", "if t == 1:", nth=1)
check("run_ms_stagewise.py lines: 'if t == 1:' directly above 'd0 = np.zeros(n_ep)'", ln_t1 == ln_root - 1 and ln_bal > ln_root)
ln_ss, txt_ss = code_line("envs/curriculum_env.py", "State-balanced exploring starts on D_t")
txt_ss = txt_ss.strip('" ')
ln_bd, txt_bd = code_line("envs/curriculum_env.py", "Draw n gaps: pick a bin uniformly, then uniform within the bin.")
txt_bd = txt_bd.strip('" ')
ln_nb, _ = code_line("envs/curriculum_env.py", "def n_bins(self, t: int)")
for col in COLS:
    if col == "v20":
        add(GCt, "start design", col, "stage 3", "not applicable (T = 2 has stages 1 and 2)", psrc(col, "records.50.game.T"))
        add(GCt, "start design", col, "stage 2 (phase A)", PH_A["starts"], psrc(col, "pipeline.phase_A.starts"))
        add(GCt, "start design", col, "stage 1 (phase B)", PH_B["starts"], psrc(col, "pipeline.phase_B.starts"))
        add(GCt, "start design", col, "bin width on D_2", int(P20["records"]["50"]["protocol"]["es_bin_width"]), psrc(col, "records.50.protocol.es_bin_width and records.60.protocol.es_bin_width (equal)"))
        for q in QS:
            add(GCt, "start design", col, f"bins on D_2 at q = {q}", int(P20["records"][str(q)]["es_bins_stage2"]), psrc(col, f"records.{q}.es_bins_stage2") + "; " + csrc(col, "record.es_bins_stage2"))
            sampler = StartSampler(GameSpec(**CFGS[col][(q, SEEDS[col][0])]["record"]["game"]), 10.0)
            check(f"v2.0 q={q}: StartSampler(bin width 10).n_bins(2) == es_bins_stage2", sampler.n_bins(2) == P20["records"][str(q)]["es_bins_stage2"])
    else:
        P = PROTO[col][0]
        sw = P["settings"]["start_weights"]
        check(f"{COLNAME[col]}: start_weights.scheme == bin_balanced in the protocol and in all configs", sw == {"scheme": "bin_balanced"}
              and all(cfg["start_weights"] == sw for cfg in CFGS[col].values()))
        add(GCt, "start design", col, "scheme (protocol key)", sw["scheme"], psrc(col, "settings.start_weights.scheme") + "; " + csrc(col, "start_weights.scheme"))
        add(GCt, "start design", col, "stage 3", f"bin_balanced starts on D_3 via StartSampler.balanced: '{txt_bd}'",
            f"{msw(col, 'bal')} (`{txt_bal}` for t >= 2, scheme bin_balanced); envs/curriculum_env.py:{ln_bd} (StartSampler.balanced docstring)")
        add(GCt, "start design", col, "stage 2", f"bin_balanced starts on D_2 via StartSampler.balanced: '{txt_bd}'",
            f"{msw(col, 'bal')} (`{txt_bal}` for t >= 2, scheme bin_balanced); envs/curriculum_env.py:{ln_bd} (StartSampler.balanced docstring)")
        add(GCt, "start design", col, "stage 1", "root: d = 0 (code: `d0 = np.zeros(n_ep)` under `if t == 1:`)",
            f"{MSW}:{LN['t1'][col]}-{LN['root'][col]} (file at the run commit {COMMIT[col][:7]}; `{txt_t1}` / `{txt_root}`)")
        add(GCt, "start design", col, "bin width on D_t", int(P["records"]["50"]["protocol"]["es_bin_width"]),
            psrc(col, "records.50.protocol.es_bin_width and records.60.protocol.es_bin_width (equal)") + "; " + csrc(col, "record.protocol.es_bin_width"))
        for q in QS:
            sampler = StartSampler(GameSpec(**CFGS[col][(q, SEEDS[col][0])]["record"]["game"]), 10.0)
            for t in (3, 2):
                add(GCt, "start design", col, f"bins on D_{t} at q = {q}", int(sampler.n_bins(t)),
                    f"derived: StartSampler.n_bins({t}) = ceil(2 (t-1) B / bin width) (envs/curriculum_env.py:{ln_nb}), game and es_bin_width from " + csrc(col, "record.game and record.protocol.es_bin_width"))


# ---------------------------------------------------------------------------------------------- D continuation-value calculation
GD = "D continuation-value calculation"
CT20 = P20["pipeline"]["continuation_table"]
RULE20 = CT20["rule"]
ln_re, ln_re2, txt_re = code_par("run/v2_rollout.py", 'terminal shocks are still drawn', "Non-terminal stage")
m_re = re.search(r"the terminal reward is the conditional expectation .*?k e_own\^2\.", txt_re)
check("run/v2_rollout.py docstring sentence on the expected terminal reward found", m_re is not None)
txt_re = m_re.group(0)
ln_ms710, txt_ms710 = code_line("run/run_ms_stagewise.py", 'reward_mode="expected"')
ln_msr, ln_msr2, txt_msr = code_par("run/ms_rollout.py", "In the phase of stage ``t < T`` the learner rolls out stage ``t`` only", "lookup).")
ln_vd, txt_vd = code_line("utils/ms_continuation.py", "V~_{t+1}(y) = E_z[ g_{t+1}(y + z) ]")
ln_gT, txt_gT = code_line("utils/ms_continuation.py", "g_T(d)       = w_l + DW F_xi")
ln_gs, txt_gs = code_line("utils/ms_continuation.py", "g_s(d)       = -k e_hat_s(d)^2 + V~_{s+1}")
ln_rule, ln_rule2, txt_rule = code_par("utils/ms_continuation.py", "Integration rule = the locked v2.0 rule", "never interpolated.")
check("ms_continuation rule paragraph states 6 nodes per panel, width 1, step 0.05, float64, linear lookup",
      all(t in txt_rule for t in ("6 nodes per panel", "panels of width 1", "y-grid step 0.05", "float64, linear lookup")))
ln_defs, _ = code_line("utils/v2_continuation.py", "DEFAULT_STEP = 0.05")
check("utils/v2_continuation.py defaults: step 0.05, panel width 1.0, 6 nodes per panel",
      (code_line("utils/v2_continuation.py", "DEFAULT_PANEL_WIDTH = 1.0")[0], code_line("utils/v2_continuation.py", "DEFAULT_NODES_PER_PANEL = 6")[0]) == (ln_defs + 1, ln_defs + 2))
ln_cts, _ = code_line("run/run_ms_stagewise.py", "CONT_TABLE_STEP = 0.05")
ln_ygrid, _ = code_line("utils/ms_continuation.py", "half = spec.domain_half(t) + spec.e_range")

# the manifests' continuation_tables records (T3): identical in all runs of a protocol and q except build_seconds
def table_meta(m: Dict[str, Any], k: str) -> Dict[str, Any]:
    d = dict(m["continuation_tables"][k]["meta"])
    d.pop("build_seconds")
    return d


TM = {}
for col in ("t30", "t31"):
    for q in QS:
        ms = [json.dumps({k: table_meta(MANS[col][(q, s)], k) for k in ("1", "2")}, sort_keys=True) for s in SEEDS[col]]
        check(f"{COLNAME[col]} q={q}: manifest.json continuation_tables meta identical in all {len(ms)} runs (except build_seconds)", len(set(ms)) == 1)
        TM[(col, q)] = json.loads(ms[0])
for q in QS:
    check(f"q={q}: T3-1.0 and T3-1.1 continuation-table meta identical", TM[("t30", q)] == TM[("t31", q)])


def git_text(col: str, path: str) -> str:
    """Text of ``path`` at the commit recorded in the manifests of the runs of ``col``."""
    out = subprocess.run(["git", "-C", str(ROOT), "show", f"{COMMIT[col]}:{path}"], capture_output=True, text=True)
    check(f"git show {COMMIT[col][:7]}:{path} readable", out.returncode == 0)
    return out.stdout


for col in ("t30", "t31"):
    msw_txt = git_text(col, MSW).splitlines()
    check(f"{COLNAME[col]}: run/run_ms_stagewise.py at the run commit assigns self.tables once at construction and fills it once, in freeze_stage (`self.tables[t - 1] = tbl`)",
          sum(1 for l in msw_txt if re.search(r"self\.tables(\s*:[^=]*)?\s*=", l)) == 1 and sum(1 for l in msw_txt if re.search(r"self\.tables\[[^\]]*\]\s*=[^=]", l)) == 1
          and LN["fz_def"][col] < LN["snap"][col] < LN["fz_build"][col] < LN["fz_rng"][col] < LN["fz_store"][col])
    check(f"{COLNAME[col]}: the build is guarded by `if t >= 2:` two lines above it and follows the deep copy of the actor in freeze_stage",
          msw_txt[LN["fz_build"][col] - 3].strip() == "if t >= 2:" and LN["snap"][col] < LN["fz_build"][col] and LN["fz_call"][col] > LN["fz_loop"][col])
    for f_ in ("utils/ms_continuation.py", "utils/v2_continuation.py"):
        check(f"{COLNAME[col]}: {f_} at the run commit is identical to the file at HEAD (the cited line numbers hold)", git_text(col, f_) == (ROOT / f_).read_text())
    check(f"{COLNAME[col]}: rng_unchanged_by_build is true (training, torch_global, numpy_global, python_random) for both tables in all {NRUN[col]} manifests",
          all(m_["continuation_tables"][k_]["rng_unchanged_by_build"] == {"training": True, "torch_global": True, "numpy_global": True, "python_random": True}
              for m_ in MANS[col].values() for k_ in ("1", "2")))
    check(f"{COLNAME[col]}: weight_sum of both tables is 1.0 to 1e-12 in all {NRUN[col]} manifests",
          all(abs(m_["continuation_tables"][k_]["meta"]["weight_sum"] - 1.0) <= 1e-12 for m_ in MANS[col].values() for k_ in ("1", "2")))
ln_rt31 = ln_at("t31", MSW, "if t == T and run.retry_cfg is not None:")
ln_sq, _ = code_line("utils/ms_continuation.py", "nodes, weights, n_panels = shock_quadrature(spec.q, panel_width, nodes_per_panel)")
ln_wsum, _ = code_line("utils/ms_continuation.py", '"n_nodes": int(nodes.size), "weight_sum": float(weights.sum()),')
ln_w1, ln_w2, txt_w = code_par("utils/v2_continuation.py", "Composite Gauss-Legendre rule for", "weights include the density at the node.")
txt_w = txt_w.lstrip('"')                      # the docstring's opening quotes belong to the code, not to the quoted text
ln_wf, txt_wf = code_line("utils/v2_continuation.py", "weights = base * (2.0 * q - np.abs(nodes)) / (4.0 * q * q)")
# v2.0: the manifest records the rule of the table that was built
V20T = {}
for q in QS:
    rs = [json.dumps({k: v for k, v in MANS["v20"][(q, s)]["continuation_table"]["rule"].items()}, sort_keys=True) for s in SEEDS["v20"]]
    check(f"v2.0 q={q}: manifest.json continuation_table.rule identical in the 20 runs", len(set(rs)) == 1)
    V20T[q] = json.loads(rs[0])
    check(f"v2.0 q={q}: protocol rule n_panels / n_nodes == the manifest's", RULE20["n_panels"][str(q)] == V20T[q]["n_panels"] and RULE20["n_nodes"][str(q)] == V20T[q]["n_nodes"]
          and RULE20["panel_width"] == V20T[q]["panel_width"] and RULE20["nodes_per_panel"] == V20T[q]["nodes_per_panel"] and V20T[q]["n_y"] == RULE20["y_grid"]["n_nodes"])

def man_src(col: str, key: str) -> str:
    pat = CFG_PATTERN[col].replace("run_config.json", "manifest.json")
    return f"{pat} :: {key} ({NRUN[col]} manifests read; identical in all runs of that q except build_seconds)"


for col in COLS:
    if col == "v20":
        add(GD, "terminal-stage training reward", col, "flag (protocol key)", P20["pipeline"]["flags"]["reward_mode"], psrc(col, "pipeline.flags.reward_mode"))
        add(GD, "terminal-stage training reward", col, "meaning (quoted from code)", txt_re,
            f"run/v2_rollout.py:{ln_re}-{ln_re2} (module docstring, flag reward_mode=\"expected\"; white space collapsed)")
        add(GD, "stages whose training return uses the continuation table", col, "where (quoted)", CT20["applies_to"], psrc(col, "pipeline.continuation_table.applies_to"))
        add(GD, "continuation table", col, "definition (quoted)", CT20["definition"], psrc(col, "pipeline.continuation_table.definition"))
        add(GD, "continuation table", col, "tables built", "one table V2~, built once at Phase-B entry from the frozen stage-2 snapshot", psrc(col, "pipeline.continuation_table.built") + " (quoted: " + CT20["built"] + ")")
        add(GD, "integration rule", col, "z-integral (quoted)", RULE20["integration"], psrc(col, "pipeline.continuation_table.rule.integration"))
        add(GD, "integration rule", col, "nodes per panel", int(RULE20["nodes_per_panel"]), psrc(col, "pipeline.continuation_table.rule.nodes_per_panel") + "; " + fsrc(MAN_PATH[col](50, 30501), "continuation_table.rule.nodes_per_panel"))
        add(GD, "integration rule", col, "panel width", float(RULE20["panel_width"]), psrc(col, "pipeline.continuation_table.rule.panel_width") + "; " + fsrc(MAN_PATH[col](50, 30501), "continuation_table.rule.panel_width"))
        add(GD, "integration rule", col, "panel alignment (quoted)", RULE20["panel_alignment"], psrc(col, "pipeline.continuation_table.rule.panel_alignment"))
        for q in QS:
            add(GD, "integration rule", col, f"panels at q = {q}", int(RULE20["n_panels"][str(q)]), psrc(col, f"pipeline.continuation_table.rule.n_panels.{q}") + "; " + man_src(col, "continuation_table.rule.n_panels").replace("identical in all runs of that q except build_seconds", "identical in the 20 runs of that q"))
            add(GD, "integration rule", col, f"quadrature nodes at q = {q}", int(RULE20["n_nodes"][str(q)]), psrc(col, f"pipeline.continuation_table.rule.n_nodes.{q}") + "; " + man_src(col, "continuation_table.rule.n_nodes").replace("identical in all runs of that q except build_seconds", "identical in the 20 runs of that q"))
        add(GD, "integration rule", col, "y-grid step", float(RULE20["y_grid"]["step"]), psrc(col, "pipeline.continuation_table.rule.y_grid.step"))
        add(GD, "integration rule", col, "y-grid range (quoted)", RULE20["y_grid"]["range"], psrc(col, "pipeline.continuation_table.rule.y_grid.range"))
        for q in QS:
            add(GD, "integration rule", col, f"y-grid nodes at q = {q}", int(V20T[q]["n_y"]), psrc(col, "pipeline.continuation_table.rule.y_grid.n_nodes") + "; " + man_src(col, "continuation_table.rule.n_y").replace("identical in all runs of that q except build_seconds", "identical in the 20 runs of that q"))
        add(GD, "integration rule", col, "lookup (quoted)", RULE20["y_grid"]["lookup"] if "lookup" in RULE20["y_grid"] else RULE20["lookup"], psrc(col, "pipeline.continuation_table.rule.lookup"))
        add(GD, "integration rule", col, "actor evaluation (quoted)", RULE20["actor_evaluation"], psrc(col, "pipeline.continuation_table.rule.actor_evaluation"))
        add(GD, "integration rule", col, "arithmetic", RULE20["dtype"], psrc(col, "pipeline.continuation_table.rule.dtype"))
        add(GD, "integration rule", col, "module", CT20["module"], psrc(col, "pipeline.continuation_table.module"))
    else:
        P = PROTO[col][0]
        add(GD, "terminal-stage training reward", col, "flag (code)", "reward_mode=\"expected\" in the phase of stage T", f"{msw(col, 'ms710')} (`{txt_ms710}` in the `if t == T:` branch of run_stage)")
        add(GD, "terminal-stage training reward", col, "meaning (quoted from code)", txt_re, f"run/v2_rollout.py:{ln_re}-{ln_re2} (module docstring, flag reward_mode=\"expected\"; white space collapsed)")
        add(GD, "stages whose training return uses the continuation table", col, "where (quoted)", txt_msr, f"run/ms_rollout.py:{ln_msr}-{ln_msr2} (module docstring, white space collapsed)")
        add(GD, "continuation table", col, "definition (quoted)", " ; ".join(re.sub(r"\s+", " ", t.strip()) for t in (txt_vd, txt_gT, txt_gs)), f"utils/ms_continuation.py:{ln_vd}, :{ln_gT}, :{ln_gs} (module docstring, white space collapsed)")
        t2, t1 = TM[(col, 50)]["2"], TM[(col, 50)]["1"]
        add(GD, "continuation table", col, "table used by the phase of stage 2", f"V~_{t2['stage']} from the frozen stage-{t2['stage']} policy (nested = {str(t2['nested']).lower()})", man_src(col, "continuation_tables.2.meta.{stage, phase_stage, nested}"))
        add(GD, "continuation table", col, "table used by the phase of stage 1", f"V~_{t1['stage']} from the frozen stage-{t1['stage']} policy (nested = {str(t1['nested']).lower()}: its g uses the lookup of the table above)", man_src(col, "continuation_tables.1.meta.{stage, phase_stage, nested}"))
        tb_ = ("two tables, each built once and then held fixed: V~_3 (used by the phase of stage 2) at the end of the stage-3 phase and V~_2 (used by the phase of stage 1) at the end of the stage-2 phase, "
               "each inside freeze_stage(t) from a deep copy of the actor frozen at that point (V~_2 integrates the frozen stage-2 actor and looks up V~_3); "
               "no table is built at the end of the stage-1 phase; the build draws no random number (a moved training stream raises an error; rng_unchanged_by_build is true for every table of every run)")
        if col == "t31":
            tb_ += "; in a run whose stage-3 restart fired, freeze_stage(3) runs after the second attempt, so V~_3 is built from the second attempt's actor"
        tb_src = (f"{MSW}:{LN['fz_loop'][col]}-{LN['fz_call'][col]} (loop over t = T, ..., 1: run_stage(t), then freeze_stage(t)); :{LN['fz_def'][col]} (def freeze_stage); "
                  f":{LN['snap'][col]} (`snap = copy.deepcopy(self.agent.actor)`); :{LN['fz_build'][col]} (`tbl = build_table(snap, spec, t - 1, self.tables.get(t), step=CONT_TABLE_STEP)`, under `if t >= 2:` two lines above); "
                  f":{LN['fz_rng'][col]} (RuntimeError if a training stream moved); :{LN['fz_store'][col]} (`self.tables[t - 1] = tbl`, the only fill of self.tables); "
                  f":{LN['tb_use'][col]} (the phase of stage t reads self.tables.get(t)) (file at the run commit {COMMIT[col][:7]})")
        if col == "t31":
            tb_src += f"; {MSW}:{ln_rt31}-{LN['fz_call'][col]} (`if t == T and run.retry_cfg is not None:` ... second `rec = run.run_stage(t)`, then `run.freeze_stage(t, rec, check)`; same commit)"
        tb_src += "; " + man_src(col, "continuation_tables.{1,2}.{file, meta.stage, meta.phase_stage, rng_unchanged_by_build}")
        add(GD, "continuation table", col, "tables built", tb_, tb_src)
        add(GD, "integration rule", col, "rule (quoted)", txt_rule, f"utils/ms_continuation.py:{ln_rule}-{ln_rule2} (module docstring, white space collapsed)")
        add(GD, "integration rule", col, "z-integral (quoted)", t2["rule"], man_src(col, "continuation_tables.2.meta.rule"))
        add(GD, "integration rule", col, "weights (quoted)", txt_w,
            f"utils/v2_continuation.py:{ln_w1}-{ln_w2} (shock_quadrature docstring, white space collapsed; density in the code: `{txt_wf}`, :{ln_wf}); called by utils/ms_continuation.py:{ln_sq}")
        add(GD, "integration rule", col, "sum of the quadrature weights", float(t2["weight_sum"]),
            man_src(col, "continuation_tables.{1,2}.meta.weight_sum") + f"; utils/ms_continuation.py:{ln_wsum} (meta weight_sum = weights.sum())")
        add(GD, "integration rule", col, "nodes per panel", int(t2["nodes_per_panel"]), man_src(col, "continuation_tables.{1,2}.meta.nodes_per_panel") + f"; defaults utils/v2_continuation.py:{ln_defs + 2}")
        add(GD, "integration rule", col, "panel width", float(t2["panel_width"]), man_src(col, "continuation_tables.{1,2}.meta.panel_width") + f"; defaults utils/v2_continuation.py:{ln_defs + 1}")
        for q in QS:
            m2, m1 = TM[(col, q)]["2"], TM[(col, q)]["1"]
            check(f"{COLNAME[col]} q={q}: both tables use the same panel rule", {k: m2[k] for k in ("n_panels", "n_nodes", "panel_width", "nodes_per_panel")} == {k: m1[k] for k in ("n_panels", "n_nodes", "panel_width", "nodes_per_panel")}
                  and m2["n_panels"] == RULE20["n_panels"][str(q)] and m2["n_nodes"] == RULE20["n_nodes"][str(q)])
            add(GD, "integration rule", col, f"panels at q = {q}", int(m2["n_panels"]), man_src(col, "continuation_tables.{1,2}.meta.n_panels"))
            add(GD, "integration rule", col, f"quadrature nodes at q = {q}", int(m2["n_nodes"]), man_src(col, "continuation_tables.{1,2}.meta.n_nodes"))
        add(GD, "integration rule", col, "y-grid step", float(t2["step_requested"]), man_src(col, "continuation_tables.{1,2}.meta.step_requested") + f"; {msw(col, 'cts')} (CONT_TABLE_STEP)")
        for q in QS:
            m2, m1 = TM[(col, q)]["2"], TM[(col, q)]["1"]
            add(GD, "integration rule", col, f"y-grid range, table of the stage-2 phase, q = {q}", f"[-{m2['y_half']:g}, {m2['y_half']:g}]", man_src(col, "continuation_tables.2.meta.y_half") + f"; utils/ms_continuation.py:{ln_ygrid} (half-width domain_half(t) + e_range)")
            add(GD, "integration rule", col, f"y-grid nodes, table of the stage-2 phase, q = {q}", int(m2["n_y"]), man_src(col, "continuation_tables.2.meta.n_y"))
            add(GD, "integration rule", col, f"y-grid range, table of the stage-1 phase, q = {q}", f"[-{m1['y_half']:g}, {m1['y_half']:g}]", man_src(col, "continuation_tables.1.meta.y_half") + f"; utils/ms_continuation.py:{ln_ygrid}")
            add(GD, "integration rule", col, f"y-grid nodes, table of the stage-1 phase, q = {q}", int(m1["n_y"]), man_src(col, "continuation_tables.1.meta.n_y"))
        add(GD, "integration rule", col, "arithmetic and lookup", "float64, linear lookup", f"utils/ms_continuation.py:{ln_rule}-{ln_rule2} (substring 'float64, linear lookup' of the rule paragraph)")


# ---------------------------------------------------------------------------------------------- E retry rule
GE = "E retry rule"
RT = P31["settings"]["retry"]
SR = P31["settings"]["stage3_retry"]
ln_dv, txt_dv = code_line("run/run_ms_stagewise.py", "ev, err = self.verify_candidate(self.dev_cfg)")
ln_Dl, txt_Dl = code_line("utils/ms_residual.py", "Delta_t   = full_delta_max[t] / dw")
ln_tr, txt_tr = code_line("run/ms_retry.py", "if delta > float(delta_gt):")
check("no stage3_retry key in the T3-1.0 protocol settings, in the v2.0 protocol, or in any T3-R1 / v2.0 run_config.json",
      "stage3_retry" not in P30["settings"] and "stage3_retry" not in json.dumps(P20) and all("stage3_retry" not in cfg for c in ("v20", "t30") for cfg in CFGS[c].values()))
check("stage3_retry == {delta_gt 0.005, max_retries 1} in the T3-1.1 protocol and in all 80 T3-R4 configs",
      SR == {"delta_gt": 0.005, "max_retries": 1} and all(cfg["stage3_retry"] == SR for cfg in CFGS["t31"].values()))
add(GE, "restart rule", "v20", "value", "none", psrc("v20", "pipeline (no restart key)") + "; " + csrc("v20", "(no stage3_retry key)") + "; run/ms_retry.py validate: the restart is defined for T = 3 only")
add(GE, "restart rule", "t30", "value", "none", psrc("t30", "settings (no stage3_retry key)") + "; " + csrc("t30", "(no stage3_retry key)"))
add(GE, "restart rule", "t31", "config key", f"stage3_retry = {json.dumps(SR)}", psrc("t31", "settings.stage3_retry") + "; " + csrc("t31", "stage3_retry"))
add(GE, "restart rule", "t31", "difference from T3-1.0 (quoted)", P31["builds_on"]["difference"], psrc("t31", "builds_on.difference"))
add(GE, "quantity measured", "t31", "Delta (quoted definition)", re.sub(r"\s+", " ", txt_Dl.strip()), f"utils/ms_residual.py:{ln_Dl} (module docstring, white space collapsed)")
add(GE, "quantity measured", "t31", "Delta of the stage-3 check at local update 1600 vs dev_eta_T_over_dw", "equal bit for bit in all 80 runs (Delta is the development-tier η_3/ΔW)",
    CFG_PATTERN["t31"].replace("run_config.json", "ms_checks_stage3.csv") + " :: column Delta, row local == 1600, compared with " + rel(C.PER_RUN_R4) + " :: column dev_eta_T_over_dw (80 runs)")
add(GE, "quantity measured", "t31", "tier of the check", "development tier", f"{msw('t31', 'dv')} (`{txt_dv}` in the development check of the phase); " + psrc("t31", "paper_wording") + " (development-tier one-step deviation gain)")
add(GE, "quantity measured", "t31", "in the paper wording", P31["paper_wording"], psrc("t31", "paper_wording"))
add(GE, "when it is read", "t31", "trigger (quoted)", RT["trigger"], psrc("t31", "settings.retry.trigger"))
add(GE, "when it is read", "t31", "threshold delta_gt (Delta must exceed it)", float(SR["delta_gt"]), psrc("t31", "settings.stage3_retry.delta_gt") + f"; run/ms_retry.py:{ln_tr}")
add(GE, "what happens", "t31", "retry (quoted)", RT["retry"], psrc("t31", "settings.retry.retry"))
add(GE, "what happens", "t31", "maximum number of retries", int(SR["max_retries"]), psrc("t31", "settings.stage3_retry.max_retries"))
m_fz = re.search(r"the second attempt is frozen whatever its own check says", RT["retry"])
m_bw = re.search(r"a run whose trigger does not fire is bit-identical to bw", RT["records"])
check("settings.retry texts contain the statements on the frozen attempt and on bit-identity with bw", m_fz is not None and m_bw is not None)
add(GE, "which attempt is frozen", "t31", "fired run (quoted)", m_fz.group(0), psrc("t31", "settings.retry.retry (substring)") + "; retry.json frozen_attempt = 2 in the 5 fired runs (checked)")
add(GE, "which attempt is frozen", "t31", "run whose trigger does not fire (quoted)", m_bw.group(0), psrc("t31", "settings.retry.records (substring)"))
add(GE, "records kept", "t31", "records (quoted)", RT["records"], psrc("t31", "settings.retry.records"))
bf = RT["budget_of_a_fired_run"]
add(GE, "cost of a fired run", "t31", "updates", int(bf["updates"]), psrc("t31", "settings.retry.budget_of_a_fired_run.updates"))
add(GE, "cost of a fired run", "t31", "policy rows", int(bf["policy_rows"]), psrc("t31", "settings.retry.budget_of_a_fired_run.policy_rows"))
add(GE, "cost of a fired run", "t31", "minibatch steps", int(bf["minibatch_steps"]), psrc("t31", "settings.retry.budget_of_a_fired_run.minibatch_steps"))
add(GE, "cost of a fired run", "t31", "development checks", int(bf["development_checks"]), psrc("t31", "settings.retry.budget_of_a_fired_run.development_checks"))
for col in ("v20", "t30"):
    for stat in ("quantity measured", "when it is read", "what happens", "which attempt is frozen", "records kept", "cost of a fired run"):
        add(GE, stat, col, "value", "not applicable (no restart rule)", {"v20": psrc("v20", "pipeline (no restart key)"), "t30": psrc("t30", "settings (no stage3_retry key)")}[col])
# for the paper wording rows of v20/t30 the first-listed columns above already carry the statement; keep one row per statement and column

# ---------------------------------------------------------------------------------------------- F evaluated mean-effort policy
GF_ = "F evaluated mean-effort policy"
ln_v20e, txt_v20e = code_line("tools/t3/t3_eval.py", "V20_EXPORTS = {2:")
ln_lbw, _ = code_line("tools/t3/t3_eval.py", "def load_bw(")
ln_ld1, _ = code_line("tools/t3/t3_eval.py", 'st = torch.load(run_dir / "state_end_stage1.pt"')
ln_ld2, _ = code_line("tools/t3/t3_eval.py", 'for s, sd in st["frozen_stages"].items()')
ln_ld3, _ = code_line("tools/t3/t3_eval.py", 'nets[1] = net_from_state_dict(st["agent"]["actor"], cfg)')
check("tools/t3/t3_eval.py: load_bw reads state_end_stage1.pt, st['frozen_stages'] and st['agent']['actor'] on three consecutive lines inside the function",
      ln_ld1 == ln_lbw + 2 and ln_ld2 == ln_ld1 + 1 and ln_ld3 == ln_ld2 + 1)
ln_lbw_doc, txt_lbw_doc = code_line("tools/t3/t3_eval.py", "Composite of ``bw``: frozen stage 3, frozen stage 2, live actor at stage 1")
ln_snap, txt_snap = code_line("run/run_ms_stagewise.py", "snap = copy.deepcopy(self.agent.actor)")
ln_me, _ = code_line("utils/v2_continuation.py", "def frozen_mean_effort")
ma = P20["records"]["50"]["mean_extraction"]
check("mean_extraction text identical in the three protocols", ma == P30["records"]["50"]["mean_extraction"] == P31["records"]["50"]["mean_extraction"] == P20["records"]["60"]["mean_extraction"])
for col in COLS:
    add(GF_, "mean effort", col, "definition (quoted)", ma, psrc(col, "records.50.mean_extraction and records.60.mean_extraction (equal)"))
add(GF_, "candidate", "v20", "evaluated candidate (quoted)", P20["pipeline"]["evaluated_candidate"], psrc("v20", "pipeline.evaluated_candidate"))
ln_l20, txt_l20 = code_line("tools/t3/t3_eval.py", "v2.0's evaluated candidate: stage 2 from")
add(GF_, "composition (quoted)", "v20", "load_v20", txt_l20.strip('" '), f"tools/t3/t3_eval.py:{ln_l20} (docstring of load_v20)")
add(GF_, "stage 3", "v20", "policy", "not applicable (T = 2 has stages 1 and 2)", psrc("v20", "records.50.game.T"))
add(GF_, "stage 2", "v20", "policy", "export weights/u01600.npz (global update 1600 = end of Phase A, the frozen stage-2 snapshot)",
    f"tools/t3/t3_eval.py:{ln_v20e} (`{txt_v20e}`); " + psrc("v20", "gates.G-A.candidate") + f" (quoted: {P20['gates']['G-A']['candidate']}); exports present in all 40 runs (checked)")
add(GF_, "stage 1", "v20", "policy", "export weights/u02200.npz (global update 2200 = end of Phase B, the live stage-1 actor)",
    f"tools/t3/t3_eval.py:{ln_v20e} (`{txt_v20e}`); " + psrc("v20", "gates.G-F.candidate") + f" (quoted: {P20['gates']['G-F']['candidate']}); exports present in all 40 runs (checked)")
check("v2.0 global updates: phase A ends at 1600, phase B at 2200", PH_A["updates"] == 1600 and PH_A["updates"] + PH_B["updates"] == 2200 and set(CV.updates) == {2200})
for col in ("t30", "t31"):
    P = PROTO[col][0]
    add(GF_, "candidate", col, "evaluated policy (quoted)", P["settings"]["evaluated_policy"], psrc(col, "settings.evaluated_policy"))
    add(GF_, "composition (quoted)", col, "load_bw", txt_lbw_doc.strip('" '), f"tools/t3/t3_eval.py:{ln_lbw_doc} (docstring of load_bw, which reads state_end_stage1.pt keys frozen_stages and agent)")
    extra = ""
    add(GF_, "stage 3", col, "policy", "frozen stage-3 snapshot taken at the end of the stage-3 phase (global update 1600); equals export weights/u01600.npz bit for bit" + extra,
        f"{msw(col, 'snap')} (`{txt_snap}`); tools/t3/t3_eval.py:{ln_ld1}-{ln_ld2} (load_bw: torch.load of state_end_stage1.pt, then the networks of st['frozen_stages']); ms_run_summary.json phase_timing.stage3.global_exit = 1600 and bit-for-bit equality with weights/u01600.npz checked in all {NRUN[col]} runs of " + CFG_PATTERN[col].replace("run_config.json", ""))
    add(GF_, "stage 2", col, "policy", "frozen stage-2 snapshot taken at the end of the stage-2 phase (global update 3200); equals export weights/u03200.npz bit for bit",
        f"{msw(col, 'snap')}; tools/t3/t3_eval.py:{ln_ld1}-{ln_ld2} (load_bw: torch.load of state_end_stage1.pt, then the networks of st['frozen_stages']); ms_run_summary.json phase_timing.stage2.global_exit = 3200 and equality with weights/u03200.npz checked in all {NRUN[col]} runs")
    add(GF_, "stage 1", col, "policy", "live stage-1 actor at the end of the stage-1 phase (global update 3800); equals export weights/u03800.npz bit for bit",
        f"tools/t3/t3_eval.py:{ln_ld1} and :{ln_ld3} (load_bw: torch.load of state_end_stage1.pt, then st['agent']['actor'] as the stage-1 network); ms_run_summary.json phase_timing.stage1.global_exit = 3800 and equality with weights/u03800.npz checked in all {NRUN[col]} runs")
add(GF_, "stage 3", "t31", "if the restart fired", "the frozen stage-3 policy is the second attempt's; the first attempt's artefacts are in attempt1/ (checked in the 5 fired runs: attempt1/weights/u01600.npz differs from the frozen stage-3 snapshot)",
    psrc("t31", "settings.retry.retry and settings.retry.records") + "; run/ms_retry.py; retry.json frozen_attempt = 2 in the 5 fired runs")

# ---------------------------------------------------------------------------------------------- G verifier tiers
GG = "G verifier tiers"
ln_fc, txt_fc = code_line("utils/dp_br_verifier.py", "FINAL_CONFIG = VerifierConfig(")
ln_dc, txt_dc = code_line("utils/dp_br_verifier.py", "DEV_CONFIG = VerifierConfig(")
for col in COLS:
    v = PROTO[col][0]["records"]["50"]["verifier"]
    check(f"{COLNAME[col]}: verifier record identical at q=50 and q=60", PROTO[col][0]["records"]["50"]["verifier"] == PROTO[col][0]["records"]["60"]["verifier"])
    for tier, label in (("final", "final tier"), ("development", "development tier")):
        add(GG, label, col, "state step", int(v[tier]["state_step"]), psrc(col, f"records.50.verifier.{tier}.state_step and records.60... (equal)") + "; " + csrc(col, f"record.verifier.{tier}.state_step") + f"; utils/dp_br_verifier.py:{ln_fc if tier == 'final' else ln_dc}")
        add(GG, label, col, "effort step", float(v[tier]["effort_step"]), psrc(col, f"records.50.verifier.{tier}.effort_step and records.60... (equal)") + "; " + csrc(col, f"record.verifier.{tier}.effort_step") + f"; utils/dp_br_verifier.py:{ln_fc if tier == 'final' else ln_dc}")
        add(GG, label, col, "Gauss-Legendre nodes per half of the shock support", int(v[tier]["gl_half"]), psrc(col, f"records.50.verifier.{tier}.gl_half and records.60... (equal)") + "; " + csrc(col, f"record.verifier.{tier}.gl_half") + f"; utils/dp_br_verifier.py:{ln_fc if tier == 'final' else ln_dc}")
    if col == "v20":
        for g_ in ("G-A", "G-F", "G-N", "G-S"):
            add(GG, f"tier used by {g_}", col, "tier", P20["gates"][g_]["tier"], psrc(col, f"gates.{g_}.tier"))
        add(GG, "tier statement (quoted)", col, "text", P20["gates"]["dev_tier"], psrc(col, "gates.dev_tier"))
        tiers_seen = set()
        for q_ in QS:
            for s_ in SEEDS["v20"]:
                for ph_ in ("A", "B"):
                    tiers_seen |= set(pd.read_csv(C.v20_run_dir(q_, s_) / f"v2_checkpoints_{ph_}.csv", usecols=["verifier_tier"]).verifier_tier)
        check("v2.0: every verifier call recorded in v2_checkpoints_A.csv / v2_checkpoints_B.csv of the 40 runs is on the development tier", tiers_seen == {"development"}, str(tiers_seen))
        check("v2.0 protocol: fixed_budget true", P20["pipeline"]["fixed_budget"] is True)
        add(GG, "other use of the development tier", col, "development-tier checks during training",
            "recorded at checkpoints (v2_checkpoints_A.csv and v2_checkpoints_B.csv, column verifier_tier = development); fixed_budget = true: 'no early exit'",
            psrc(col, "pipeline.fixed_budget and pipeline.outputs (v2_checkpoints_A.csv, v2_checkpoints_B.csv: per verifier call)") + f"; run/run_v2_stagewise.py:{ln_at('v20', 'run/run_v2_stagewise.py', 'fixed_budget   no early exit')} (file at the run commit {COMMIT['v20'][:7]}; docstring: 'fixed_budget   no early exit')" + "; column verifier_tier of v2_checkpoints_{A,B}.csv in " + CFG_PATTERN[col].replace("run_config.json", "") + " (all rows of the 40 runs are development)")
    else:
        P = PROTO[col][0]
        TS = "G-A and G-F on the final tier; G-N compares development and final tier"
        check(f"{COLNAME[col]}: gates.tier is the sentence '{TS}'", P["gates"]["tier"] == TS)
        add(GG, "tier used by G-A", col, "tier", "final", psrc(col, "gates.tier") + " (read from the sentence 'G-A and G-F on the final tier; ...')")
        add(GG, "tier used by G-F", col, "tier", "final", psrc(col, "gates.tier") + " (read from the sentence 'G-A and G-F on the final tier; ...')")
        add(GG, "tier used by G-N", col, "tier", "development vs final", psrc(col, "gates.tier") + " (read from the sentence '...; G-N compares development and final tier')")
        add(GG, "tier used by G-S", col, "tier", f"not applied at T = 3 (applies at horizon {P['gates']['G-S']['applies_at_horizon']} only)", psrc(col, "gates.G-S.applies_at_horizon"))
        add(GG, "tier statement (quoted)", col, "text", P["gates"]["tier"], psrc(col, "gates.tier"))
        add(GG, "other use of the development tier", col, "development check during training", f"every {P['settings']['check_every']} local updates, recorded (stage rule disabled: enabled = {str(P['settings']['rule']['enabled']).lower()})",
            psrc(col, "settings.check_every and settings.rule.enabled") + f"; {msw(col, 'dv')} (`{txt_dv}`)")
        if col == "t31":
            add(GG, "other use of the development tier", col, "restart trigger", "the trigger reads the development-tier check at local update 1600 of the stage-3 phase", psrc(col, "settings.retry.trigger") + "; " + psrc(col, "paper_wording"))
        else:
            add(GG, "other use of the development tier", col, "restart trigger", "not applicable (no restart rule)", psrc(col, "settings (no stage3_retry key)"))
    if col == "v20":
        add(GG, "other use of the development tier", col, "restart trigger", "not applicable (no restart rule)", psrc(col, "pipeline (no restart key)"))

# the quantity the restart reads: the stage-3 check's Delta at local update 1600 is the development-tier eta_3 / DeltaW
bad_ = []
for r_ in PR4.itertuples():
    ck_ = pd.read_csv(C.r4_run_dir(int(r_.q), int(r_.seed)) / "ms_checks_stage3.csv", float_precision="round_trip")
    row_ = ck_[ck_.local == 1600]
    if len(row_) != 1 or float(row_.Delta.iloc[0]) != float(r_.dev_eta_T_over_dw):
        bad_.append((int(r_.q), int(r_.seed)))
check("T3-R4: Delta of the stage-3 check at local update 1600 (final attempt) == dev_eta_T_over_dw of per_run.csv, bit for bit, all 80 runs", not bad_, str(bad_[:3]))
FIRED_SET = set(zip(FIRED_Q, FIRED_SEEDS))
check("T3-R4 trigger consistency: the 5 fired runs have first-attempt Delta > delta_gt (fired.csv trigger_Delta), the 75 others have dev_eta_T_over_dw <= delta_gt",
      bool((FIRED_CSV.trigger_Delta > SR["delta_gt"]).all())
      and all(float(r_.dev_eta_T_over_dw) <= SR["delta_gt"] for r_ in PR4.itertuples() if (int(r_.q), int(r_.seed)) not in FIRED_SET))

# the fired runs: attempt 1's stage-3 policy is not the frozen one
for s in FIRED_SEEDS:
    d = C.r4_run_dir(50, s)
    st = torch.load(d / "state_end_stage1.pt", map_location="cpu", weights_only=False)
    check(f"T3-R4 q=50 seed {s} (fired): attempt1/weights/u01600.npz exists and differs from the frozen stage-3 snapshot (attempt 2 is frozen)",
          (d / "attempt1" / "weights" / "u01600.npz").is_file() and not arrays_equal(st["frozen_stages"]["3"], d / "attempt1" / "weights" / "u01600.npz")
          and rj(d / "retry.json")["frozen_attempt"] == 2)

# ---------------------------------------------------------------------------------------------- H pass criteria as applied
GH = "H pass criteria as applied"
ln_gcA, txt_gcA = code_line("tools/t3/t3_metrics.py", 'ga_eta = bool(float(fin["eta_T_over_dw"])')
ln_gcS, txt_gcS = code_line("tools/t3/t3_metrics.py", 'if int(T) == g["G-S"]["applies_at_horizon"]:')
ln_gcOK, txt_gcOK = code_line("tools/t3/t3_metrics.py", 'ok = out["gate_G_A"] and out["gate_G_F"] and out["gate_G_N"]')
ln_gcDoc, txt_gcDoc = code_line("tools/t3/t3_metrics.py", "G-S (T = 2 only): ``stage1_rel_err_abs``. Run pass: G-A and G-F and G-N (and G-S at T = 2).")
V = {m["metric"]: m for g in ("G-A", "G-F", "G-N", "G-S") for m in P20["gates"][g]["all_must_hold"]}
GV = {"G-A": ("eta_T_over_dw", "rmse_pos_T_over_eT0", "tail_mean_T_over_eT0"), "G-F": ("Gmax_full_over_dw",),
      "G-N": ("eta_T_over_dw_dev_minus_final_abs", "Gmax_full_over_dw_dev_minus_final_abs"), "G-S": ("stage1_rel_err_abs",)}
V20NAME = {"rmse_pos_T_over_eT0": "stage2_rmse_pos_over_g2_0", "tail_mean_T_over_eT0": "stage2_tail_mean_over_g2_0"}
LABEL = {
    "eta_T_over_dw": "G-A: η_T/ΔW (largest one-step deviation gain at the terminal stage / ΔW), stored name eta_T_over_dw",
    "rmse_pos_T_over_eT0": "G-A: terminal-stage positive-region RMSE / e_T*(0), stored name rmse_pos_T_over_eT0",
    "tail_mean_T_over_eT0": "G-A: mean effort in the theoretical zero-effort tails / e_T*(0), stored name tail_mean_T_over_eT0",
    "Gmax_full_over_dw": "G-F: full-domain dynamic best-response gain G_max,full/ΔW, stored name Gmax_full_over_dw",
    "eta_T_over_dw_dev_minus_final_abs": "G-N: |η_T/ΔW(development) - η_T/ΔW(final)|, stored name eta_T_over_dw_dev_minus_final_abs",
    "Gmax_full_over_dw_dev_minus_final_abs": "G-N: |G_max,full/ΔW(development) - G_max,full/ΔW(final)|, stored name Gmax_full_over_dw_dev_minus_final_abs",
    "stage1_rel_err_abs": "G-S: |e_1(0) - e_1*(0)| / e_1*(0), stored name stage1_rel_err_abs"}
for g, metrics in GV.items():
    for m in metrics:
        # v2.0 column
        v20_name = V20NAME.get(m, m)
        mm = V[v20_name]
        check(f"v2.0 gate metric {v20_name}: op '<='", mm["op"] == "<=")
        add(GH, LABEL[m], "v20", "threshold (<=, inclusive)", float(mm["threshold"]),
            psrc("v20", f"gates.{g}.all_must_hold[metric == {v20_name}].threshold") + "; " + (f"evaluator: {CRIT_REL} :: gates.{g}.{m}" if m in CRIT["gates"][g] else ""))
        # T3 columns
        for col in ("t30", "t31"):
            if g == "G-S":
                add(GH, LABEL[m], col, "threshold (<=, inclusive)", f"not applied at T = 3 (applies at horizon {PROTO[col][0]['gates']['G-S']['applies_at_horizon']} only)",
                    psrc(col, "gates.G-S.applies_at_horizon and gates.run_pass.3"))
            else:
                add(GH, LABEL[m], col, "threshold (<=, inclusive)", float(PROTO[col][0]["gates"][g][m]), psrc(col, f"gates.{g}.{m}") + f"; {CRIT_REL} :: gates.{g}.{m}")
# gates in the run-pass definition
rp20 = "G-A, G-F, G-N, G-S"
check("v2.0 run_outcome text: 'pass iff G-A, G-F, G-N and G-S all pass'", "pass iff G-A, G-F, G-N and G-S all pass" in P20["gates"]["run_outcome"])
add(GH, "gates in the run-pass definition", "v20", "gates", rp20, psrc("v20", "gates.run_outcome") + " (quoted: 'pass iff G-A, G-F, G-N and G-S all pass')")
for col in ("t30", "t31"):
    rp = PROTO[col][0]["gates"]["run_pass"]
    add(GH, "gates in the run-pass definition", col, "gates", ", ".join(rp["3"]), psrc(col, "gates.run_pass.3") + f"; tools/t3/t3_metrics.py:{ln_gcDoc}-{ln_gcOK}")
    check(f"{COLNAME[col]}: gates.run_pass['3'] == G-A, G-F, G-N and ['2'] adds G-S", rp["3"] == ["G-A", "G-F", "G-N"] and rp["2"] == ["G-A", "G-F", "G-N", "G-S"])
add(GH, "gates in the run-pass definition", "v20", "gate applied by the evaluator at horizon 2 (same as the protocol)", "G-A, G-F, G-N, G-S",
    psrc("t30", "gates.run_pass.2") + f"; tools/t3/t3_metrics.py:{ln_gcS} (G-S when T == applies_at_horizon); this evaluator produced the v20 rows of {rel(C.PER_RUN_R1)}")
add(GH, "run-pass definition (quoted)", "v20", "text", P20["gates"]["run_outcome"], psrc("v20", "gates.run_outcome"))
add(GH, "run-pass definition (quoted)", "t30", "text", P30["confirmation"]["run_pass"], psrc("t30", "confirmation.run_pass"))
add(GH, "run-pass definition (quoted)", "t31", "text", P31["confirmation"]["run_pass"], psrc("t31", "confirmation.run_pass"))
for col in COLS:
    cmpt = PROTO[col][0]["gates"]["comparison"]
    add(GH, "comparison rule (quoted)", col, "text", cmpt, psrc(col, "gates.comparison"))
for col in ("v20", "t30", "t31"):
    add(GH, "recovery grid", col, "grid step on D_T (recovery_step)", float(PROTO[col][0]["records"]["50"]["protocol"]["recovery_step"]),
        psrc(col, "records.50.protocol.recovery_step and records.60.protocol.recovery_step (equal)") + "; " + csrc(col, "record.protocol.recovery_step"))
for col in ("t30", "t31"):
    add(GH, "recovery grid", col, "positive-effort region and theoretical zero-effort tails (quoted)", f"pos: {CRIT['recovery']['pos']}; tail: {CRIT['recovery']['tail']}", CRIT_REL + " :: recovery.pos and recovery.tail")
GA20 = P20["gates"]["G-A"]["all_must_hold"]
check("v2.0 gates.G-A.all_must_hold[1] (stage2_rmse_pos_over_g2_0) states the positive region |d| < 2q and all_must_hold[2] (stage2_tail_mean_over_g2_0) the tail |d| >= 2q",
      GA20[1]["metric"] == "stage2_rmse_pos_over_g2_0" and "|d| < 2q" in GA20[1]["definition"]
      and GA20[2]["metric"] == "stage2_tail_mean_over_g2_0" and "|d| >= 2q" in GA20[2]["definition"])
add(GH, "recovery grid", "v20", "positive-effort region and theoretical zero-effort tails (quoted)", f"pos: {CRIT['recovery']['pos']}; tail: {CRIT['recovery']['tail']}",
    CRIT_REL + " :: recovery.pos and recovery.tail (the evaluator that produced the v20 rows); " + psrc("v20", "gates.G-A.all_must_hold[1].definition") + " (positive region |d| < 2q); "
    + psrc("v20", "gates.G-A.all_must_hold[2].definition") + " (tail |d| >= 2q)")

# ================================================================================================ write the CSV and the MD
def csv_value(v: Any) -> str:
    if isinstance(v, bool):
        raise AssertionError("no booleans in the table")
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    return str(v)


COLUMNS = ["table", "row_group", "row", "column_or_q", "statistic", "value", "source"]
# a row must be unique in (row_group, row, column_or_q, statistic)
keys = [(r["row_group"], r["row"], r["column_or_q"], r["statistic"]) for r in ROWS]
dups = {k for k in keys if keys.count(k) > 1}
check("every (row_group, row, column, statistic) occurs once", not dups, str(sorted(dups))[:300])
check("every row has a non-empty source and a value", all(r["source"].strip() and str(r["value"]).strip() != "" for r in ROWS))
check("CSV value floats are Python floats that round-trip", all(float(repr(r["value"])) == r["value"] for r in ROWS if isinstance(r["value"], float)))

OUT_T = OUT / "tables"
OUT_T.mkdir(parents=True, exist_ok=True)
CSV_PATH, MD_PATH = OUT_T / "T1.csv", OUT_T / "T1.md"
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(COLUMNS)
    for r in ROWS:
        w.writerow([r[c] if c != "value" else csv_value(r[c]) for c in COLUMNS])


# ---- the Markdown rendering: produced FROM the CSV just written
T = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
check("T1.csv round trip: same number of rows and the float values round-trip exactly",
      len(T) == len(ROWS) and all(float(t) == r["value"] for t, r in zip(T.value, ROWS) if isinstance(r["value"], float)))
INT_RE = re.compile(r"-?\d+")
FLT_RE = re.compile(r"-?\d+\.\d*(?:e[+-]?\d+)?|-?\d+e[+-]?\d+")


def disp(v: str) -> str:
    """Display of a CSV value: integers as integers, floats to four significant digits, text unchanged."""
    if INT_RE.fullmatch(v):
        return str(int(v))
    if FLT_RE.fullmatch(v):
        return f"{float(v):.4g}"
    return v


TAGS = {".claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0": "V20_ROOT",
        ".claude/worktrees/t3-bw-vs-jt-51e472/results/t3_r1": "R1_ROOT",
        ".claude/worktrees/t3-r4-lock-confirm/results/t3_r4/confirmation": "R4_ROOT"}


def esc(t: str) -> str:
    return t.replace("|", "\\|").replace("\n", " ")


def short(t: str) -> str:
    """Source text for the MD: the three long run roots are replaced by their names in scripts/_common.py."""
    for k, v in TAGS.items():
        t = t.replace(k, v)
    return t


COLORDER = [COLNAME[c] for c in COLS]
md: List[str] = []
md.append("# T1 Experimental protocols and gates\n")
md.append("Readable rendering of `tables/T1.csv` (one value per CSV row: table, row_group, row, column_or_q, statistic, value, source). "
          "Every number below is a value of the CSV; integers are shown as integers and floats to four significant digits "
          "(the CSV holds them at full precision). One column per protocol: T = 2 v2.0 (`protocols/v2_T2_locked_v2_0`), "
          "T3-1.0 (`protocols/v2_T3_locked_v1_0`, the settings of round T3-R1's `bw`) and T3-1.1 (`protocols/v2_T3_locked_v1_1`, `bw` plus the "
          "verifier-gated restart of stage 3). 'not applicable' marks a quantity the protocol does not have; 'quoted' marks text copied "
          "from the source named in the sources table of the group. In the sources tables the three untracked run roots are written as "
          "V20_ROOT = `.claude/worktrees/v2-t2-refine/results/v2_T2_locked/confirmation_v2_0`, R1_ROOT = `.claude/worktrees/t3-bw-vs-jt-51e472/results/t3_r1` and "
          "R4_ROOT = `.claude/worktrees/t3-r4-lock-confirm/results/t3_r4/confirmation` (relative to the main checkout, as defined in `scripts/_common.py`; read only); "
          "the CSV holds the full paths. Counts of runs and seeds are of the stored records; nothing here was trained or re-run. "
          "In group B each block is given its role in the terms of the spec (development, comparison or confirmation block) in the row 'role of the block (spec terms)', "
          "and the label of every other statistic of the group, except the three 'q values' rows, ends, in brackets, with the round its numbers belong to.\n")
for grp in GROUP_ORDER:
    sub = T[T.row_group == grp]
    md.append(f"\n## {esc(grp)}\n")
    md.append("| Row | " + " | ".join(COLORDER) + " |")
    md.append("|---|" + "---|" * len(COLORDER))
    row_names = list(dict.fromkeys(sub.row))
    for rn in row_names:
        cells = []
        for cn in COLORDER:
            ss = sub[(sub.row == rn) & (sub.column_or_q == cn)]
            if len(ss) == 0:
                cells.append("")
                continue
            parts = [(esc(disp(v)) if st == "value" else f"{esc(st)}: {esc(disp(v))}") for st, v in zip(ss.statistic, ss.value)]
            cells.append("<br>".join(parts))
        md.append(f"| {esc(rn)} | " + " | ".join(cells) + " |")
    md.append(f"\n**Sources of the group {esc(grp)}**\n")
    md.append("| Row | Column | Statistic(s) <- source (file :: key or line; derived values name the operation) |")
    md.append("|---|---|---|")
    for rn in row_names:
        for cn in COLORDER:
            ss = sub[(sub.row == rn) & (sub.column_or_q == cn)]
            if len(ss) == 0:
                continue
            by_src: Dict[str, List[str]] = {}
            for st, sr in zip(ss.statistic, ss.source):
                by_src.setdefault(sr, []).append(st)
            lines_ = []
            for sr, sts in by_src.items():
                label = "value" if sts == ["value"] else "; ".join(sts)
                lines_.append(f"{esc(label)} <- {esc(short(sr))}")
            md.append(f"| {esc(rn)} | {esc(cn)} | " + " <br> ".join(lines_) + " |")
MD_TEXT = "\n".join(md) + "\n"

# no number in the MD that is not in the CSV
TOK = re.compile(r"\d+(?:\.\d+)?(?:e[+-]?\d+)?")
allowed = set()
for r in T.itertuples():
    for fld in (r.row_group, r.row, r.column_or_q, r.statistic, r.source, r.value):
        allowed |= set(TOK.findall(fld))
    allowed |= set(TOK.findall(disp(r.value)))
extra = sorted(set(TOK.findall(MD_TEXT)) - allowed)
check("every number token of T1.md occurs in T1.csv (a field or the four-digit display of a value)", not extra, str(extra))
MD_PATH.write_text(MD_TEXT, encoding="utf-8")

# ---- report
print(f"T1: {len(ROWS)} rows -> {CSV_PATH.relative_to(ROOT)} and {MD_PATH.relative_to(ROOT)}")
print(f"checks run: {len(CHECKS)}; all passed")
if "--checks" in sys.argv:
    for w, r_ in CHECKS:
        print(f"  {r_[:2]} {w} {r_[2:]}")
print("commits recorded in the manifests of the runs read:", {COLNAME[c]: COMMIT[c] for c in COLS})
