#!/usr/bin/env python
"""T4 - method comparison and cost (round P-FIG, paper/multistage_rev).

Reads stored records only (no training, no evaluation of a policy) and writes ``tables/T4.csv`` (one value per
row: table, row_group, row, column_or_q, statistic, value, source) and ``tables/T4.md`` (the readable
rendering of the same rows). Deterministic: running it twice gives byte-identical files.

Parts:
  1. T3-R1 comparison of bw, jt, jt_es (T = 3, seeds 50501-50520): pass counts, paired differences of
     G_max,full/DeltaW with the pre-registered 95 % intervals.
  2. Per arm: budget, runtime, network structure, state-start design.
  3. T3-1.1 retry cost (T3-R4 confirmation, seeds 80501-80540): fired runs, updates and policy rows per run,
     first-attempt count next to the final count.

The untracked run roots (``_common.R1_ROOT``, ``_common.R4_ROOT``) are read only.

Usage:
    /home/fjiang4/tournament_experiment/.venv/bin/python paper/multistage_rev/scripts/T4.py
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402  (sets the single-thread environment)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TABLE = "T4"
OUT_CSV = C.OUT / "tables" / "T4.csv"
OUT_MD = C.OUT / "tables" / "T4.md"
ARMS = ("bw", "jt", "jt_es")
SEEDS_R1 = tuple(range(50501, 50521))
SEEDS_R4 = tuple(range(80501, 80541))
REPO_TOP = C.WT.parents[1]                       # /home/fjiang4/tournament_experiment (the repository)
R1_REL = str(C.R1_ROOT.relative_to(REPO_TOP))    # untracked, read-only
R4_REL = str(C.R4_ROOT.relative_to(REPO_TOP))    # untracked, read-only
RUN_COMMIT_R1 = "a848114eef501d9ef6bab0fcd7c71a3989d32d3c"  # commit in the manifest.json of the 120 T3-R1 runs

# tracked sources (repo-relative)
F_GATE = "results/t3_r1/analysis/gate_counts.csv"
F_CRIT = "results/t3_r1/analysis/criterion.csv"
F_BUDGET = "results/t3_r1/analysis/budget_check.csv"
F_COST1 = "results/t3_r1/analysis/cost.csv"
F_PREREG = "reports/t3/r1/prereg_criteria.json"
F_HASHES = "reports/t3/r1/run_config_hashes.csv"
F_COST4 = "results/t3_r4/confirmation/analysis/cost.csv"
F_FIRED = "results/t3_r4/confirmation/analysis/fired.csv"
F_COUNTS = "results/t3_r4/confirmation/analysis/counts.csv"
F_HEAD = "results/t3_r4/confirmation/analysis/headline.json"
F_PER4 = "results/t3_r4/confirmation/analysis/per_run.csv"
F_PROTO11 = "protocols/v2_T3_locked_v1_1.json"

# footnotes of the markdown (id -> text); every row carries the ids of its sources
FOOT: Dict[str, str] = {
    "gc": f"{F_GATE}: columns n_planned, run_pass, G-A_pass, G-F_pass, G-N_pass (rows horizon=3)",
    "cr": f"{F_CRIT}: columns mean_diff, ci95_lo, ci95_hi, n_pairs, ci_hi_below_0, hypothesis_supported, "
          f"bootstrap_seed (rows horizon=3)",
    "pr": f"{F_PREREG}: gates, statistics (n_boot, percentiles, interval_method, bootstrap_seed), hypotheses",
    "bu": f"{F_BUDGET}: columns n_runs, updates_expected, updates_equal, rows_total_expected, rows_total_equal, "
          f"rows_by_stage_equal, minibatch_steps_expected, minibatch_steps_equal",
    "c1": f"{F_COST1}: columns wall_sec, rows_t1, rows_t2, rows_t3, updates, policy_rows_total, minibatch_steps",
    "cf": f"{R1_REL}/runs/q{{50,60}}/seed{{50501..50520}}/{{bw,jt,jt_es}}/run_config.json (untracked, read-only; "
          f"SHA-256 of every file equals run_config_sha256 in {F_HASHES}): keys record.ppo, record.protocol, "
          f"record.game, record.obs_encoding, record.action_mapping, record.action_sampling, record.mean_extraction, "
          f"pipeline, start_weights, joint",
    "cd": "agents/ppo_curriculum.py: BetaActor.__init__, BetaActor.forward (variant t1) and Critic; run/run_ms_stagewise.py "
          "(module docstring, MSRun.__init__, MSRun.run_stage, MSRun.freeze_stage); run/run_joint.py (module docstring, "
          "draw_starts, policy_rows_per_update); tools/t3/t3_eval.py (load_bw, load_joint); "
          "tools/t3/t3_configs.py (JOINT_BUDGETS, BW_BUDGETS, expected_budget)",
    "st": f"{R1_REL}/runs/q{{50,60}}/seed{{50501..50520}}/{{bw,jt,jt_es}}/state_end_stage1.pt (bw) or state_end.pt "
          f"(jt, jt_es) and weights/u*.npz (untracked, read-only): stored actor and critic weights of the "
          f"evaluated policy",
    "mf": f"{R1_REL}/runs/q{{50,60}}/seed{{50501..50520}}/{{jt,jt_es}}/manifest.json: key policy_rows_per_update "
          f"(a dict with one entry per stage; its entries sum to the policy rows per update; the dict is identical "
          f"in all 40 files of the arm)",
    "la": f"{R1_REL}/launch_t3_t2j_20261017_010337.json and {R4_REL}/launch_20261022_071148.json: keys workers, "
          f"nproc; key threads_per_process (torch 1, OMP 1, MKL 1, OPENBLAS 1) of the run_config.json of the 120 "
          f"runs of bw, jt, jt_es ({R1_REL}/runs/q{{50,60}}/seed{{50501..50520}}/{{bw,jt,jt_es}}/) and of the 80 runs "
          f"of bwA ({R4_REL}/q{{50,60}}/seed{{80501..80540}}/bwA/), identical in all of them",
    "c4": f"{F_COST4}: columns runs, fired, fired_share, mean_updates, mean_policy_rows, median_wall_sec, "
          f"mean_wall_sec_fired, mean_wall_sec_not_fired, ratio_updates_to_not_fired, "
          f"ratio_policy_rows_to_not_fired",
    "fi": f"{F_FIRED}: column run (the fired runs)",
    "cn": f"{F_COUNTS}: columns runs, pass",
    "hd": f"{F_HEAD}: keys counts.pass_total, trigger.fired, trigger.n, trigger.runs, trigger.first_attempt_pass, "
          f"trigger.fired_runs_that_pass, n_rows",
    "p4": f"{F_PER4}: columns run, q, seed, fired, passed, total_updates, total_transitions, wall_sec",
    "pt": f"{F_PROTO11}: settings.stage3_retry",
    "c4cfg": f"{R4_REL}/q{{50,60}}/seed{{80501..80540}}/bwA/run_config.json (untracked, read-only): keys "
             f"stage3_retry, pipeline.budgets",
    "t4": "tools/t3/t3r4_analysis.py, line 344: first_pass is the number of runs with passed and not fired",
    "rt": "run/ms_retry.py (module docstring: trigger, at most one retry, attempt 2 frozen) and utils/ms_residual.py "
          "(Delta_t = full_delta_max[t] / dw of the development check)",
}


# row groups (shared by the CSV rows and the Markdown rendering)
G_PASS = "T3-R1 pass counts (T = 3, seeds 50501-50520, 20 runs per arm and q)"
G_DIFF = "T3-R1 paired differences of the full-domain dynamic best-response gain G_max,full/ΔW (a - b; same seed within q)"
G_METHOD = "T3-R1 interval method"
G_LIMITS = "gate limits behind the pass counts (T = 3: run pass = G-A, G-F and G-N)"
G_BUDGET = "per-arm budget (T3-R1, 40 runs per arm)"
G_RUNTIME = "per-arm runtime (T3-R1)"
G_LAUNCH = "launch of the runs (context of the wall times)"
G_NET = "per-arm network structure (evaluated policy)"
G_START = "per-arm state-start design (T3-R1)"
G_RESTART = "T3-1.1 restart rule (stage-3 retry)"
G_COUNTS = "T3-1.1 confirmation block: final count and first-attempt count"
G_FIRED = "T3-1.1 fired runs"
G_COST = "T3-1.1 cost per run"

ROWS: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------------- helpers
def py(x: Any) -> Any:
    """A numpy scalar as the matching python scalar."""
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def vtext(x: Any) -> str:
    """CSV text of a value: floats by repr (full precision), ints and bools plainly, strings as they are."""
    x = py(x)
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return repr(x)
    return str(x)


def add(group: str, row: str, col: str, stat: str, value: Any, source: str, fn: Sequence[str]) -> None:
    """Append one row (one value) of the table."""
    ROWS.append({"table": TABLE, "row_group": group, "row": row, "column_or_q": col, "statistic": stat,
                 "value": vtext(value), "source": source, "_raw": py(value), "_fn": tuple(fn)})


def one(df: pd.DataFrame, **cond: Any) -> pd.Series:
    """The single row of ``df`` that satisfies all conditions."""
    m = pd.Series(True, index=df.index)
    for k, v in cond.items():
        m &= df[k] == v
    sel = df[m]
    if len(sel) != 1:
        raise RuntimeError(f"{len(sel)} rows for {cond}")
    return sel.iloc[0]


def unique(values: Sequence[Any], what: str) -> Any:
    """The common value of a sequence (error if the values differ)."""
    s = {json.dumps(py(v)) for v in values}
    if len(s) != 1:
        raise RuntimeError(f"{what}: values differ: {sorted(s)}")
    return py(values[0])


def stop(msg: str) -> None:
    """Stop-and-report (the checks of the P-FIG brief)."""
    raise SystemExit(f"STOP: {msg}")


def get(d: Any, path: str) -> Any:
    """Value at a dotted key path of nested dicts."""
    for k in path.split("."):
        d = d[k]
    return d


# ---------------------------------------------------------------------------------- stored run configs
def read_r1_configs() -> Dict[str, List[Dict[str, Any]]]:
    """run_config.json of all 40 runs per arm of T3-R1; each file's SHA-256 must equal the tracked record."""
    h = pd.read_csv(C.ROOT / F_HASHES)
    h = h[h["wave"] == "t3"].set_index(["arm", "q", "seed"])["run_config_sha256"]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for arm in ARMS:
        out[arm] = []
        for q in C.QS:
            for s in SEEDS_R1:
                p = C.r1_run_dir(q, s, arm) / "run_config.json"
                if hashlib.sha256(p.read_bytes()).hexdigest() != h[(arm, q, s)]:
                    stop(f"{p}: SHA-256 differs from run_config_sha256 of {F_HASHES}")
                out[arm].append(C.read_json(p))
                if C.read_json(p.parent / "manifest.json")["commit"] != RUN_COMMIT_R1:
                    stop(f"{p.parent / 'manifest.json'}: commit differs from {RUN_COMMIT_R1}")
    return out


def stored_networks() -> Dict[str, Dict[str, Any]]:
    """Structure of the evaluated policy read from the stored weights of all 40 runs per arm."""
    import torch
    C.single_thread()
    res: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        layers, inputs, outputs, widths, critic, n_distinct, variant = [], [], [], [], [], [], []
        for q in C.QS:
            for s in SEEDS_R1:
                d = C.r1_run_dir(q, s, arm)
                if arm == "bw":
                    st = torch.load(d / "state_end_stage1.pt", map_location="cpu", weights_only=False)
                    if sorted(st["frozen_stages"]) != ["2", "3"]:
                        stop(f"{d}: frozen_stages of state_end_stage1.pt are {sorted(st['frozen_stages'])}, "
                             f"not ['2', '3']")
                    nets = {"3": st["frozen_stages"]["3"], "2": st["frozen_stages"]["2"],
                            "1": st["agent"]["actor"]}
                else:
                    st = torch.load(d / "state_end.pt", map_location="cpu", weights_only=False)
                    nets = {"all": st["agent"]["actor"]}
                variant.append(("actor_variant" in st) or ("actor_variant" in st["agent"]))
                last = sorted(glob.glob(str(d / "weights" / "u*.npz")))[-1]
                variant.append("actor_variant" in np.load(last).files)
                for sd in nets.values():
                    names = sorted({k.split(".")[0] for k in sd if k.endswith(".weight")})
                    layers.append(len(names) - 1)
                    inputs.append(int(sd["l1.weight"].shape[1]))
                    widths.append((int(sd["l1.weight"].shape[0]), int(sd["l2.weight"].shape[0]),
                                   int(sd["l2.weight"].shape[1])))
                    outputs.append(int(sd["out.weight"].shape[0]))
                cr = st["agent"]["critic"]
                critic.append((int(cr["l1.weight"].shape[1]), int(cr["l1.weight"].shape[0]),
                               int(cr["l2.weight"].shape[0]), int(cr["out.weight"].shape[0])))
                ks = list(nets)
                distinct = 1 if arm != "bw" else len(
                    {i for i in range(3)
                     if all(not all(torch.equal(nets[ks[i]][k], nets[ks[j]][k]) for k in nets[ks[i]])
                            for j in range(i))})
                n_distinct.append(distinct)
        res[arm] = {"hidden_layers": unique(layers, f"{arm} hidden layers"),
                    "inputs": unique(inputs, f"{arm} inputs"), "outputs": unique(outputs, f"{arm} outputs"),
                    "widths": unique([list(w) for w in widths], f"{arm} widths"),
                    "critic": unique([list(c) for c in critic], f"{arm} critic"),
                    "n_distinct": unique(n_distinct, f"{arm} distinct networks"),
                    "variant_entry": unique(variant, f"{arm} actor_variant entry")}
    return res


# ---------------------------------------------------------------------------------- part 1
def part1() -> None:
    """T3-R1 pass counts and paired differences."""
    gc = C.load_per_run(C.ROOT / F_GATE)
    cr = C.load_per_run(C.ROOT / F_CRIT)
    prereg = C.read_json(C.ROOT / F_PREREG)

    g = G_PASS
    for arm in ARMS:
        for q in C.QS:
            r = one(gc, arm=arm, q=q, horizon=3)
            sel = f"row arm={arm}, q={q}, horizon=3"
            for stat, col in (("runs planned (n_planned)", "n_planned"),
                              ("runs passing the run-pass rule (run_pass)", "run_pass"),
                              ("runs passing G-A (G-A_pass)", "G-A_pass"),
                              ("runs passing G-F (G-F_pass)", "G-F_pass"),
                              ("runs passing G-N (G-N_pass)", "G-N_pass")):
                add(g, arm, f"q={q}", stat, int(r[col]), f"{F_GATE}, column {col} ({sel})", ["gc"])

    g = G_DIFF
    S = prereg["statistics"]["bootstrap_seed"]["3"]
    for h, a, b in (("H1", "bw", "jt"), ("H2", "bw", "jt_es"), ("H3", "jt_es", "jt")):
        pre = prereg["hypotheses"][h]
        if (pre["a"], pre["b"], pre["horizon"]) != (a, b, 3):
            stop(f"{h} in {F_PREREG} is not {a} - {b} at horizon 3")
        supp = []
        for q in C.QS:
            r = one(cr, hypothesis=h, horizon=3, q=q)
            if (r["a"], r["b"], r["statistic"]) != (a, b, "Gmax_full_over_dw") or int(r["bootstrap_seed"]) != S:
                stop(f"{F_CRIT}: row {h} q={q} differs from the pre-registration")
            supp.append(bool(r["hypothesis_supported"]))
            sel = f"row hypothesis={h}, horizon=3, q={q}"
            row = f"{h}: {a} - {b}"
            for stat, col in (("mean paired difference (mean_diff)", "mean_diff"),
                              ("95 % interval, lower end (ci95_lo)", "ci95_lo"),
                              ("95 % interval, upper end (ci95_hi)", "ci95_hi"),
                              ("paired seeds (n_pairs)", "n_pairs"),
                              ("upper end below 0 (ci_hi_below_0)", "ci_hi_below_0")):
                v = int(r[col]) if col == "n_pairs" else (bool(r[col]) if col == "ci_hi_below_0" else float(r[col]))
                add(g, row, f"q={q}", stat, v, f"{F_CRIT}, column {col} ({sel})", ["cr"])
        add(g, f"{h}: {a} - {b}", "q=50 and q=60", "hypothesis supported (hypothesis_supported)",
            unique(supp, f"{h} hypothesis_supported"),
            f"{F_CRIT}, column hypothesis_supported (rows hypothesis={h}, horizon=3, q=50 and q=60, equal)", ["cr"])

    g = G_METHOD
    st = prereg["statistics"]
    add(g, "all hypotheses", "H1, H2, H3", "bootstrap_seed", S,
        f"{F_PREREG}, key statistics.bootstrap_seed.3; equal to column bootstrap_seed of {F_CRIT} in all six "
        f"rows horizon=3", ["pr", "cr"])
    for q in C.QS:
        for h in ("H1", "H2", "H3"):
            if int(one(cr, hypothesis=h, horizon=3, q=q)["bootstrap_seed"]) != S:
                stop(f"{F_CRIT}: bootstrap_seed of {h} q={q} differs")
    add(g, "all hypotheses", "H1, H2, H3", "bootstrap resamples (n_boot)", int(st["n_boot"]),
        f"{F_PREREG}, key statistics.n_boot", ["pr"])
    add(g, "all hypotheses", "H1, H2, H3", "percentiles of the 95 % interval",
        "; ".join(repr(float(x)) for x in st["percentiles"]["ci95"]),
        f"{F_PREREG}, key statistics.percentiles.ci95 (numpy.percentile, linear method, of the resampled means)",
        ["pr"])

    g = G_LIMITS
    gt = prereg["gates"]
    if gt["run_pass"]["3"] != ["G-A", "G-F", "G-N"]:
        stop(f"{F_PREREG}: run_pass at horizon 3 is not G-A, G-F, G-N")
    for gate, key, stat in (("G-A", "eta_T_over_dw", "limit on eta_T_over_dw (<=)"),
                            ("G-A", "rmse_pos_T_over_eT0", "limit on rmse_pos_T_over_eT0 (<=)"),
                            ("G-A", "tail_mean_T_over_eT0", "limit on tail_mean_T_over_eT0 (<=)"),
                            ("G-F", "Gmax_full_over_dw", "limit on Gmax_full_over_dw (<=)"),
                            ("G-N", "eta_T_over_dw_dev_minus_final_abs",
                             "limit on |eta_T_over_dw, development tier - final tier| (<=)"),
                            ("G-N", "Gmax_full_over_dw_dev_minus_final_abs",
                             "limit on |Gmax_full_over_dw, development tier - final tier| (<=)")):
        add(g, gate, "T = 3", stat, float(gt[gate][key]), f"{F_PREREG}, key gates.{gate}.{key}", ["pr"])


# ---------------------------------------------------------------------------------- part 2
def part2(cfgs: Dict[str, List[Dict[str, Any]]]) -> None:
    """Per-arm budget, runtime, network structure, state-start design."""
    bu = C.load_per_run(C.ROOT / F_BUDGET)
    c1 = C.load_per_run(C.ROOT / F_COST1)

    g = G_BUDGET
    for arm in ARMS:
        r = one(bu, arm=arm)
        s = c1[c1["arm"] == arm]
        if len(s) != 40:
            stop(f"{F_COST1}: {len(s)} rows for arm {arm}")
        upd, rows = int(r["updates_expected"]), int(r["rows_total_expected"])
        # the stored per-run values equal the expected ones in all 40 runs
        if not ((s["updates"] == upd).all() and (s["policy_rows_total"] == rows).all()
                and (s["minibatch_steps"] == int(r["minibatch_steps_expected"])).all()):
            stop(f"{F_COST1}: stored updates / policy rows / minibatch steps of arm {arm} differ from "
                 f"{F_BUDGET}")
        if rows % upd:
            stop(f"arm {arm}: rows_total_expected is not a multiple of updates_expected")
        ep = unique([get(c, "record.protocol.episodes_per_update") for c in cfgs[arm]], f"{arm} episodes")
        cf_src = f"{R1_REL}/runs/q{{50,60}}/seed{{50501..50520}}/{arm}/run_config.json"
        col = "all 40 runs"
        for stat, c_ in (("runs (n_runs)", "n_runs"),
                         ("updates per run (updates_expected)", "updates_expected"),
                         ("runs whose stored updates equal updates_expected (updates_equal)", "updates_equal"),
                         ("policy rows per run (rows_total_expected)", "rows_total_expected"),
                         ("runs whose stored policy rows equal rows_total_expected (rows_total_equal)",
                          "rows_total_equal"),
                         ("runs whose stored policy rows by stage equal the expected rows (rows_by_stage_equal)",
                          "rows_by_stage_equal"),
                         ("minibatch steps per run (minibatch_steps_expected)", "minibatch_steps_expected"),
                         ("runs whose stored minibatch steps equal minibatch_steps_expected "
                          "(minibatch_steps_equal)", "minibatch_steps_equal")):
            add(g, arm, col, stat, int(r[c_]), f"{F_BUDGET}, column {c_} (row arm={arm})", ["bu"])
        for t in (3, 2, 1):
            v = unique(list(s[f"rows_t{t}"]), f"{arm} rows_t{t}")
            add(g, arm, col, f"policy rows at stage {t} per run (rows_t{t})", int(v),
                f"{F_COST1}, column rows_t{t} (identical in all 40 rows of arm {arm})", ["c1"])
        add(g, arm, col, "episodes per update", int(ep),
            f"{cf_src}, key record.protocol.episodes_per_update (identical in all 40 files)", ["cf"])
        mf_src = ""
        if arm != "bw":
            pru = unique([C.read_json(C.r1_run_dir(q, sd, arm) / "manifest.json")["policy_rows_per_update"]
                          for q in C.QS for sd in SEEDS_R1], f"{arm} manifest policy_rows_per_update")
            if sum(pru.values()) != rows // upd:
                stop(f"arm {arm}: manifest policy_rows_per_update sums to {sum(pru.values())}, not {rows // upd}")
            mf_src = (f"; equals the sum of the stage entries of the dict policy_rows_per_update in each manifest.json "
                      f"({R1_REL}/runs/q{{50,60}}/seed{{50501..50520}}/{arm}/manifest.json), a dict that is identical "
                      f"in all 40 files of the arm ({', '.join(f'stage {k}: {v}' for k, v in pru.items())}); it is "
                      f"not a sum over the files")
        add(g, arm, col, "policy rows per update (rows_total_expected / updates_expected)", rows // upd,
            f"derived: rows_total_expected / updates_expected of {F_BUDGET}, row arm={arm}" + mf_src,
            ["bu"] if arm == "bw" else ["bu", "mf"])

    g = G_RUNTIME
    for arm in ARMS:
        s = c1[c1["arm"] == arm]
        for col, sub in (("q=50", s[s["q"] == 50]), ("q=60", s[s["q"] == 60]), ("both q", s)):
            if len(sub) != (40 if col == "both q" else 20):
                stop(f"{F_COST1}: {len(sub)} rows for arm {arm}, {col}")
            add(g, arm, col, "median wall time per run, seconds (wall_sec)", float(np.median(sub["wall_sec"])),
                f"median over the {len(sub)} rows of arm {arm}"
                f"{'' if col == 'both q' else ', ' + col}, column wall_sec of {F_COST1} (numpy.median)", ["c1"])

    g = G_LAUNCH
    la1 = C.read_json(C.R1_ROOT / "launch_t3_t2j_20261017_010337.json")
    la4 = C.read_json(C.R4_ROOT / "launch_20261022_071148.json")
    thr = {"torch": 1, "OMP": 1, "MKL": 1, "OPENBLAS": 1}
    cfg_r1 = [c for arm in ARMS for c in cfgs[arm]]
    cfg_r4 = [C.read_json(C.r4_run_dir(q, sd) / "run_config.json") for q in C.QS for sd in SEEDS_R4]
    for what, cl in (("T3-R1", cfg_r1), ("T3-R4", cfg_r4)):
        if unique([c["threads_per_process"] for c in cl], f"{what} threads_per_process") != thr:
            stop(f"{what}: threads_per_process of the run_config.json files is not {thr}")
    thr_txt = ", ".join(f"{k}: {v}" for k, v in thr.items())
    for name, la, rel, cl, cfrel in (
            ("T3-R1 (bw, jt, jt_es)", la1, f"{R1_REL}/launch_t3_t2j_20261017_010337.json", cfg_r1,
             f"{R1_REL}/runs/q{{50,60}}/seed{{50501..50520}}/{{bw,jt,jt_es}}/run_config.json"),
            ("T3-R4 (bwA)", la4, f"{R4_REL}/launch_20261022_071148.json", cfg_r4,
             f"{R4_REL}/q{{50,60}}/seed{{80501..80540}}/bwA/run_config.json")):
        add(g, name, "all runs", "worker processes in the launch, each single-threaded (workers)", int(la["workers"]),
            f"{rel}, key workers; single-threaded: key threads_per_process ({thr_txt}) in each of the {len(cl)} "
            f"run_config.json of the runs ({cfrel}), identical in all of them", ["la"])
        add(g, name, "all runs", "CPU cores of the machine (nproc)", int(la["nproc"]), f"{rel}, key nproc", ["la"])

    net = stored_networks()
    g = G_NET
    col = "all 40 runs"
    for arm in ARMS:
        n = net[arm]
        cs = cfgs[arm]

        def pp(k: str) -> Any:
            """Value of record.ppo.<k>, identical in the 40 configs of the arm."""
            return unique([get(c, f"record.ppo.{k}") for c in cs], f"{arm} ppo.{k}")

        cfsrc = f"{R1_REL}/runs/q{{50,60}}/seed{{50501..50520}}/{arm}/run_config.json"
        stsrc = f"{R1_REL}/runs/q{{50,60}}/seed{{50501..50520}}/{arm}/" + \
            ("state_end_stage1.pt" if arm == "bw" else "state_end.pt")
        if (n["hidden_layers"] != 2 or n["inputs"] != 2 or n["outputs"] != 2 or n["variant_entry"]
                or any("actor_variant" in c for c in cs)):
            stop(f"arm {arm}: stored network is not 2 inputs - 2 hidden layers - 2 outputs with the locked actor")
        hid = pp("hidden")
        if n["widths"] != [hid, hid, hid]:
            stop(f"arm {arm}: stored layer widths {n['widths']} differ from record.ppo.hidden {hid}")
        if n["critic"] != [2, hid, hid, 1]:
            stop(f"arm {arm}: stored critic shape {n['critic']}")
        add(g, arm, col, "hidden layers of the actor", n["hidden_layers"],
            f"count of weight matrices minus one in the stored actor state_dict(s) ({stsrc}, keys l1, l2, out; "
            f"identical in all 40 runs{', for each of the 3 stage networks' if arm == 'bw' else ''})", ["st", "cd"])
        add(g, arm, col, "width of each hidden layer", int(hid),
            f"{cfsrc}, key record.ppo.hidden (identical in all 40 files); equals the stored weight shapes of {stsrc}",
            ["cf", "st"])
        add(g, arm, col, "activation of the hidden layers", "tanh",
            f"agents/ppo_curriculum.py, BetaActor.forward with variant t1 (torch.tanh on both hidden layers); no "
            f"actor_variant key in the 40 run_config.json files, in {stsrc} or in the last weights/u*.npz of any run",
            ["cd", "cf", "st"])
        obs = unique([c["record"]["obs_encoding"] for c in cs], f"{arm} obs_encoding")
        if obs != "tau=(t-1)/(T-1); dtilde_1=0; dtilde_t=d/((t-1)B); float64 -> float32":
            stop(f"arm {arm}: obs_encoding {obs!r}")
        add(g, arm, col, "inputs of the actor", "2: tau_t = (t-1)/(T-1) and d/((t-1)B) (0 at stage 1)",
            f"{cfsrc}, key record.obs_encoding (identical in all 40 files); input width 2 from the stored weights",
            ["cf", "st"])
        add(g, arm, col, "outputs of the actor", "2 linear outputs (z_mu, z_c)",
            f"agents/ppo_curriculum.py, BetaActor.__init__ line 81: `self.out = nn.Linear(hidden, 2)`; output width 2 from {stsrc}", ["cd", "st"])
        add(g, arm, col, "Beta mean parameter mu", "clamp(sigmoid(z_mu), mu_clamp, 1 - mu_clamp)",
            "agents/ppo_curriculum.py, BetaActor.forward", ["cd"])
        add(g, arm, col, "mu_clamp", pp("mu_clamp"), f"{cfsrc}, key record.ppo.mu_clamp (identical in all 40 files)",
            ["cf"])
        add(g, arm, col, "Beta concentration c", "c_min + softplus(z_c)", "agents/ppo_curriculum.py, BetaActor.forward",
            ["cd"])
        add(g, arm, col, "c_min", pp("c_min"), f"{cfsrc}, key record.ppo.c_min (identical in all 40 files)", ["cf"])
        add(g, arm, col, "Beta parameters", "alpha = mu c, beta = (1 - mu) c",
            "agents/ppo_curriculum.py, BetaActor.forward line 112: `return mu * c, (1.0 - mu) * c`", ["cd"])
        amap = unique([c["record"]["action_mapping"] + "|" + c["record"]["action_sampling"] for c in cs],
                      f"{arm} action mapping")
        if not amap.startswith("e = e_min + a (e_max - e_min); a ~ Beta(mu c, (1-mu) c)|"):
            stop(f"arm {arm}: action_mapping {amap!r}")
        add(g, arm, col, "action and effort",
            "a ~ Beta(alpha, beta) clipped to [action_clamp, 1 - action_clamp]; e = e_min + a (e_max - e_min)",
            f"{cfsrc}, keys record.action_mapping, record.action_sampling (identical in all 40 files)", ["cf"])
        add(g, arm, col, "action_clamp", pp("action_clamp"),
            f"{cfsrc}, key record.ppo.action_clamp (identical in all 40 files)", ["cf"])
        add(g, arm, col, "e_min and e_max",
            "{}; {}".format(*[unique([get(c, f"record.game.{k}") for c in cs], f"{arm} {k}") for k in ("e_min", "e_max")]),
            f"{cfsrc}, keys record.game.e_min, record.game.e_max (identical in all 40 files)", ["cf"])
        add(g, arm, col, "mean effort of the evaluated policy", unique([c["record"]["mean_extraction"] for c in cs],
                                                                      f"{arm} mean_extraction"),
            f"{cfsrc}, key record.mean_extraction (identical in all 40 files)", ["cf"])
        add(g, arm, col, "critic (training only)",
            f"separate network {n['critic'][0]} -> {hid} tanh -> {hid} tanh -> {n['critic'][3]}, no shared trunk",
            f"agents/ppo_curriculum.py, Critic; shapes from the stored critic state_dict in {stsrc}", ["cd", "st"])
        if arm == "bw":
            add(g, arm, col, "do the stages share one network", "no: one network per stage",
                "tools/t3/t3_eval.py, load_bw (stage 3 and stage 2 from state_end_stage1.pt keys frozen_stages, "
                "stage 1 from keys agent.actor)", ["cd", "st"])
            add(g, arm, col, "distinct networks in the evaluated policy", n["n_distinct"],
                f"{stsrc}: the three actor state_dicts (frozen stage 3, frozen stage 2, live actor at stage 1) are "
                f"pairwise different in all 40 runs", ["st"])
            add(g, arm, col, "how the stage networks arise",
                "one actor/critic pair is trained through the phases of stages 3, 2, 1 without re-initialisation; a "
                "copy of the actor is frozen at the end of each phase (the code also freezes a copy at the end of the "
                "stage-1 phase, which the evaluated policy does not use); the evaluated policy takes stages 3 and 2 "
                "from the frozen copies and stage 1 from the live actor stored in state_end_stage1.pt "
                "(tools/t3/t3_eval.load_bw)",
                f"run/run_ms_stagewise.py at the run commit {RUN_COMMIT_R1} (commit in manifest.json of the 40 bw "
                "runs): MSRun.__init__ line 419 (one CurriculumPPOv2 agent); MSRun.freeze_stage lines 854 and 858 "
                "(snap = copy.deepcopy(self.agent.actor) and self.frozen_nets[t] = snap), called for t = 3, 2, 1 in "
                "the loop of lines 947-949 without a condition on t; state_end_stage1.pt is written at line 824, "
                "before the copy of the stage-1 phase is taken, so it holds frozen_stages for stages 2 and 3 only "
                "(checked in all 40 bw runs); the same statements are at lines 444, 885, 889, 980-989 and 855 of the "
                "file at HEAD; tools/t3/t3_eval.py, load_bw lines 96-104 (stages 3 and 2 from frozen_stages, stage 1 "
                "from agent.actor of state_end_stage1.pt)", ["cd", "st"])
        else:
            add(g, arm, col, "do the stages share one network", "yes: one network for all three stages (the stage "
                "enters through the input tau_t)",
                "tools/t3/t3_eval.py, load_joint ({t: net for every stage t}, from state_end.pt key agent.actor)",
                ["cd", "st"])
            add(g, arm, col, "distinct networks in the evaluated policy", n["n_distinct"],
                "tools/t3/t3_eval.py, load_joint (one BetaActor used at every stage)", ["cd"])
            add(g, arm, col, "how the stage networks arise",
                "one actor/critic pair is trained on whole-game episodes; all stages are updated together in every "
                "update", "run/run_joint.py (module docstring: one network for all stages, every episode "
                "simulated from its start stage to stage T, all stages updated together)", ["cd"])

    g = G_START
    for arm in ARMS:
        cs = cfgs[arm]
        ep = int(unique([get(c, "record.protocol.episodes_per_update") for c in cs], f"{arm} episodes"))
        cfsrc = f"{R1_REL}/runs/q{{50,60}}/seed{{50501..50520}}/{arm}/run_config.json"
        bw_w = int(unique([c["record"]["protocol"]["es_bin_width"] for c in cs], f"{arm} es_bin_width"))
        if arm == "bw":
            if unique([c["start_weights"] for c in cs], "bw start_weights") != {"scheme": "bin_balanced"}:
                stop("bw start_weights is not bin_balanced")
            pl = unique([c["pipeline"] for c in cs], "bw pipeline")
            if pl["phases"] != [3, 2, 1]:
                stop("bw phases are not 3, 2, 1")
            for ph in (3, 2, 1):
                row = f"{arm}, phase of stage {ph}"
                for st, lab in ((1, "root (stage 1, d = 0)"), (2, "stage 2"), (3, "stage 3")):
                    n_st = ep if st == ph else 0
                    add(g, row, lab, "episodes started per update", n_st,
                        f"run/run_ms_stagewise.py, MSRun.run_stage: all {ep} episodes of an update start at the "
                        f"stage of the phase (d0 = 0 at stage 1, StartSampler.balanced on D_t at stages 2 and 3); "
                        f"episodes_per_update from {cfsrc}", ["cd", "cf"])
                    add(g, row, lab, "share of the episodes of an update", n_st / ep,
                        f"derived: episodes started / episodes_per_update ({cfsrc}, key "
                        f"record.protocol.episodes_per_update)", ["cd", "cf"])
                add(g, row, "phase", "updates in the phase", int(pl["budgets"][str(ph)]),
                    f"{cfsrc}, key pipeline.budgets.{ph} (identical in all 40 files)", ["cf"])
            add(g, arm, "stages 2 and 3", "distribution of the start score gap d",
                f"bin-balanced: a bin of width es_bin_width ({bw_w}) of D_t is chosen uniformly, then d is uniform in "
                f"the bin; start_weights.scheme = bin_balanced",
                f"{cfsrc}, keys start_weights.scheme, record.protocol.es_bin_width; envs/curriculum_env.py, "
                f"StartSampler.balanced", ["cf", "cd"])
            add(g, arm, "stage 1", "distribution of the start score gap d", "root: d = 0 for every episode",
                f"run/run_ms_stagewise.py, MSRun.run_stage, lines 659-660 at the run commit {RUN_COMMIT_R1} (lines "
                "687-688 at HEAD): `if t == 1:` then `d0 = np.zeros(n_ep)`", ["cd"])
        else:
            jt = unique([c["joint"] for c in cs], f"{arm} joint")
            starts = jt["starts"]
            if sum(starts.values()) != ep:
                stop(f"{arm}: starts {starts} do not sum to {ep}")
            for st, lab, key in ((1, "root (stage 1, d = 0)", "root"), (2, "stage 2", "stage2"),
                                 (3, "stage 3", "stage3")):
                add(g, arm, lab, "episodes started per update", int(starts[key]),
                    f"{cfsrc}, key joint.starts.{key} (identical in all 40 files)", ["cf"])
                add(g, arm, lab, "share of the episodes of an update", int(starts[key]) / ep,
                    f"derived: joint.starts.{key} / record.protocol.episodes_per_update ({cfsrc})", ["cf"])
            add(g, arm, "all start stages", "updates with this start mix", int(jt["updates"]),
                f"{cfsrc}, key joint.updates (identical in all 40 files)", ["cf"])
            if arm == "jt":
                add(g, arm, "stages 2 and 3", "distribution of the start score gap d", "no exploring starts (joint.starts.stage2 = "
                    "joint.starts.stage3 = 0)", f"{cfsrc}, keys joint.starts.stage2, joint.starts.stage3", ["cf"])
                add(g, arm, "stage 1", "distribution of the start score gap d", "root: d = 0 for every episode",
                    "run/run_joint.py, draw_starts, line 348 (root starts): `parts = [np.zeros(n_root)]`", ["cd"])
            else:
                add(g, arm, "stages 2 and 3", "distribution of the start score gap d",
                    f"bin-balanced exploring starts: a bin of width es_bin_width ({bw_w}) of D_t is chosen "
                    f"uniformly, then d is uniform in the bin",
                    f"run/run_joint.py, draw_starts, lines 350 and 352: `parts.append(self.sampler.balanced(2, n_2, rng))` "
                    f"and `parts.append(self.sampler.balanced(3, n_3, rng))`; "
                    f"{cfsrc}, key record.protocol.es_bin_width", ["cd", "cf"])
                add(g, arm, "stage 1", "distribution of the start score gap d", "root: d = 0 for the root starts",
                    "run/run_joint.py, draw_starts, line 348 (root starts): `parts = [np.zeros(n_root)]`", ["cd"])
            add(g, arm, "all start stages", "episode played",
                "whole-game episode from its start stage to stage 3; one policy row at every stage t >= start stage",
                "run/run_joint.py, policy_rows_per_update (an episode started at stage s contributes one policy row "
                "to every stage t >= s)", ["cd"])


# ---------------------------------------------------------------------------------- part 3
def part3() -> None:
    """T3-1.1 retry cost (T3-R4 confirmation)."""
    cost = C.load_per_run(C.ROOT / F_COST4)
    fired = C.load_per_run(C.ROOT / F_FIRED)
    counts = C.load_per_run(C.ROOT / F_COUNTS)
    head = C.read_json(C.ROOT / F_HEAD)
    per = C.load_per_run(C.ROOT / F_PER4)
    proto = C.read_json(C.ROOT / F_PROTO11)

    # the facts of the P-FIG brief (stop-and-report)
    cq = {k: counts[counts["q"].astype(str) == k].iloc[0] for k in ("50", "60", "both")}
    want = {"50": (40, 40), "60": (40, 39), "both": (80, 79)}
    for k, (n, p) in want.items():
        if (int(cq[k]["runs"]), int(cq[k]["pass"])) != (n, p):
            stop(f"{F_COUNTS}: q={k} has {cq[k]['pass']} of {cq[k]['runs']}, expected {p} of {n}")
    if head["counts"]["pass_total"] != 79 or head["trigger"]["first_attempt_pass"] != 74 or head["n_rows"] != 80:
        stop(f"{F_HEAD}: pass_total {head['counts']['pass_total']}, first_attempt_pass "
             f"{head['trigger']['first_attempt_pass']}, n_rows {head['n_rows']} (expected 79, 74, 80)")
    cs = {k: cost[cost["q"].astype(str) == k].iloc[0] for k in ("50", "60", "both")}
    if (int(cs["50"]["fired"]), int(cs["60"]["fired"]), int(cs["both"]["fired"])) != (5, 0, 5):
        stop(f"{F_COST4}: fired per q is not 5, 0, 5")
    names = list(fired["run"])
    if names != head["trigger"]["runs"] or names != list(per[per["fired"]]["run"]):
        stop("fired runs differ between fired.csv, headline.json trigger.runs and analysis/per_run.csv")
    if len(per) != 80 or {int(q): sorted(per[per["q"] == q]["seed"]) for q in C.QS} != {q: list(SEEDS_R4) for q in C.QS}:
        stop(f"{F_PER4}: the 80 rows are not seeds 80501-80540 at q=50 and q=60")

    cfg4 = [C.read_json(C.r4_run_dir(q, s) / "run_config.json") for q in C.QS for s in SEEDS_R4]
    retry = unique([c["stage3_retry"] for c in cfg4], "R4 stage3_retry")
    b3 = int(unique([c["pipeline"]["budgets"]["3"] for c in cfg4], "R4 budgets.3"))
    if retry != proto["settings"]["stage3_retry"]:
        stop(f"stage3_retry of the R4 run configs differs from {F_PROTO11}")
    if int(unique([sum(c["pipeline"]["budgets"].values()) for c in cfg4], "R4 budget sum")) != 3800:
        stop("sum of the stage budgets of the R4 run configs is not 3800")

    g = G_RESTART
    add(g, "stage3_retry", "T3-1.1", "delta_gt (limit on eta_3/ΔW of the development check at the last update of the stage-3 phase)",
        float(retry["delta_gt"]), f"{F_PROTO11}, key settings.stage3_retry.delta_gt; equal to key stage3_retry.delta_gt "
        f"in all 80 run_config.json of the confirmation; trigger defined in run/ms_retry.py", ["pt", "c4cfg", "rt"])
    add(g, "stage3_retry", "T3-1.1", "max_retries", int(retry["max_retries"]),
        f"{F_PROTO11}, key settings.stage3_retry.max_retries; equal in all 80 run configs", ["pt", "c4cfg"])
    add(g, "stage3_retry", "T3-1.1", "updates of the stage-3 phase (budget of one attempt)", b3,
        f"{R4_REL}/q{{50,60}}/seed{{80501..80540}}/bwA/run_config.json, key pipeline.budgets.3 (identical in all 80 "
        f"files)", ["c4cfg"])

    g = G_COUNTS
    blk = "seeds 80501-80540 at q=50 and q=60 (80 runs)"
    for k, lab in (("50", "q=50"), ("60", "q=60"), ("both", "both q")):
        add(g, "final count (after the restart)", lab, "runs (runs)", int(cq[k]["runs"]),
            f"{F_COUNTS}, column runs (row q={k})", ["cn"])
        add(g, "final count (after the restart)", lab, "runs passing (pass)", int(cq[k]["pass"]),
            f"{F_COUNTS}, column pass (row q={k})", ["cn"])
    add(g, "final count (after the restart)", "both q", "runs passing (counts.pass_total)",
        int(head["counts"]["pass_total"]), f"{F_HEAD}, key counts.pass_total", ["hd"])
    add(g, "first-attempt count", "both q", "runs (trigger.n)", int(head["trigger"]["n"]),
        f"{F_HEAD}, key trigger.n", ["hd"])
    add(g, "first-attempt count", "both q", "runs passing without a restart (trigger.first_attempt_pass)",
        int(head["trigger"]["first_attempt_pass"]), f"{F_HEAD}, key trigger.first_attempt_pass", ["hd"])
    for q in C.QS:
        s = per[per["q"] == q]
        add(g, "first-attempt count", f"q={q}", "runs passing without a restart (derived)",
            int((s["passed"] & ~s["fired"]).sum()),
            f"derived: number of rows with passed and not fired among the {len(s)} rows q={q} of {F_PER4}", ["p4", "t4"])
        add(g, "first-attempt count", f"q={q}", "runs at this q (n)", len(s), f"{F_PER4}, rows q={q}", ["p4"])
    if int(sum((per[per["q"] == q]["passed"] & ~per[per["q"] == q]["fired"]).sum() for q in C.QS)) != 74:
        stop(f"{F_PER4}: passed and not fired is not 74 of 80")
    add(g, "first-attempt count", "both q", "definition",
        "runs that passed and did not fire the restart; a fired run counts as a failure of its first attempt "
        "(convention)", 'tools/t3/t3r4_analysis.py, line 344: first_pass = sum(1 for r in rows if r["passed"] and not r.get("fired"))',
        ["t4"])
    add(g, "first-attempt count", "both q", "fired runs that pass (trigger.fired_runs_that_pass)",
        int(head["trigger"]["fired_runs_that_pass"]), f"{F_HEAD}, key trigger.fired_runs_that_pass", ["hd"])
    add(g, "block of both counts", "both q", "the final count and the first-attempt count refer to the same runs",
        True, f"{blk}: n_rows and trigger.n of {F_HEAD} are 80; seeds from {F_PER4} columns q, seed (80501 to 80540 at "
        f"each q); both counts come from the one file headline.json", ["hd", "p4"])
    for q in C.QS:
        s = per[per["q"] == q]["seed"]
        add(g, "block of both counts", f"q={q}", "first seed", int(s.min()), f"{F_PER4}, column seed (min, rows q={q})",
            ["p4"])
        add(g, "block of both counts", f"q={q}", "last seed", int(s.max()), f"{F_PER4}, column seed (max, rows q={q})",
            ["p4"])

    g = G_FIRED
    for k, lab in (("50", "q=50"), ("60", "q=60"), ("both", "both q")):
        add(g, "fired runs", lab, "fired runs (fired)", int(cs[k]["fired"]), f"{F_COST4}, column fired (row q={k})",
            ["c4"])
        add(g, "fired runs", lab, "runs (runs)", int(cs[k]["runs"]), f"{F_COST4}, column runs (row q={k})", ["c4"])
        add(g, "fired runs", lab, "share of runs fired (fired_share)", float(cs[k]["fired_share"]),
            f"{F_COST4}, column fired_share (row q={k})", ["c4"])
    add(g, "fired runs", "both q", "fired runs that pass (trigger.fired_runs_that_pass)",
        int(head["trigger"]["fired_runs_that_pass"]), f"{F_HEAD}, key trigger.fired_runs_that_pass", ["hd"])
    for nm in names:
        r = one(per, run=nm)
        q = int(r["q"])
        pre = f"{F_PER4}, column %s (row run={nm})"
        add(g, nm, f"q={q}", "seed", int(r["seed"]), pre % "seed", ["p4", "fi"])
        add(g, nm, f"q={q}", "run passed (passed)", bool(r["passed"]), pre % "passed", ["p4"])
        add(g, nm, f"q={q}", "total updates, both attempts (total_updates)", int(r["total_updates"]),
            pre % "total_updates", ["p4"])
        add(g, nm, f"q={q}", "total policy rows, both attempts (total_transitions)", int(r["total_transitions"]),
            pre % "total_transitions", ["p4"])
        add(g, nm, f"q={q}", "wall time, seconds (wall_sec)", float(r["wall_sec"]), pre % "wall_sec", ["p4"])

    g = G_COST
    nf = per[~per["fired"]]
    u0 = int(unique(list(nf["total_updates"]), "non-fired total_updates"))
    r0 = int(unique(list(nf["total_transitions"]), "non-fired total_transitions"))
    fr = per[per["fired"]]
    u1 = int(unique(list(fr["total_updates"]), "fired total_updates"))
    r1 = int(unique(list(fr["total_transitions"]), "fired total_transitions"))
    if u0 != 3800 or r0 != 1945600 or len(nf) != 75:
        stop(f"{F_PER4}: a run without a restart has {u0} updates and {r0} policy rows over {len(nf)} runs")
    if u1 - u0 != b3:
        stop(f"{F_PER4}: a fired run has {u1 - u0} more updates than a run without a restart, not {b3}")
    for k, lab in (("50", "q=50"), ("60", "q=60"), ("both", "both q")):
        add(g, "mean per run", lab, "mean updates per run (mean_updates)", float(cs[k]["mean_updates"]),
            f"{F_COST4}, column mean_updates (row q={k})", ["c4"])
        add(g, "mean per run", lab, "mean policy rows per run (mean_policy_rows)", float(cs[k]["mean_policy_rows"]),
            f"{F_COST4}, column mean_policy_rows (row q={k})", ["c4"])
        add(g, "mean per run", lab, "ratio of mean updates to the updates of a run without a restart "
            "(ratio_updates_to_not_fired)", float(cs[k]["ratio_updates_to_not_fired"]),
            f"{F_COST4}, column ratio_updates_to_not_fired (row q={k})", ["c4"])
        add(g, "mean per run", lab, "ratio of mean policy rows to the policy rows of a run without a restart "
            "(ratio_policy_rows_to_not_fired)", float(cs[k]["ratio_policy_rows_to_not_fired"]),
            f"{F_COST4}, column ratio_policy_rows_to_not_fired (row q={k})", ["c4"])
        add(g, "wall time per run", lab, "median wall time per run, seconds (median_wall_sec)",
            float(cs[k]["median_wall_sec"]), f"{F_COST4}, column median_wall_sec (row q={k})", ["c4"])
        if k != "60":
            add(g, "wall time per run", lab, "mean wall time of the fired runs, seconds (mean_wall_sec_fired)",
                float(cs[k]["mean_wall_sec_fired"]), f"{F_COST4}, column mean_wall_sec_fired (row q={k})", ["c4"])
        add(g, "wall time per run", lab, "mean wall time of the runs without a restart, seconds "
            "(mean_wall_sec_not_fired)", float(cs[k]["mean_wall_sec_not_fired"]),
            f"{F_COST4}, column mean_wall_sec_not_fired (row q={k})", ["c4"])
    add(g, "run without a restart", "all 75 runs", "updates per run (total_updates)", u0,
        f"{F_PER4}, column total_updates (rows fired=False; identical in all 75 rows); equals the sum of "
        f"pipeline.budgets in the 80 run_config.json", ["p4", "c4cfg"])
    add(g, "run without a restart", "all 75 runs", "policy rows per run (total_transitions)", r0,
        f"{F_PER4}, column total_transitions (rows fired=False; identical in all 75 rows)", ["p4"])
    add(g, "fired run", "all 5 fired runs", "updates per run, both attempts (total_updates)", u1,
        f"{F_PER4}, column total_updates (rows fired=True; identical in all 5 rows)", ["p4"])
    add(g, "fired run", "all 5 fired runs", "policy rows per run, both attempts (total_transitions)", r1,
        f"{F_PER4}, column total_transitions (rows fired=True; identical in all 5 rows)", ["p4"])
    add(g, "fired run", "all 5 fired runs", "extra updates of a fired run relative to a run without a restart", u1 - u0,
        f"derived: total_updates of a fired run minus total_updates of a run without a restart ({F_PER4}); equals the "
        f"updates of the stage-3 phase, key pipeline.budgets.3 of the run configs", ["p4", "c4cfg"])
    add(g, "fired run", "all 5 fired runs", "updates of a fired run / updates of a run without a restart", u1 / u0,
        f"derived: total_updates (rows fired=True) / total_updates (rows fired=False) of {F_PER4}", ["p4"])
    add(g, "fired run", "all 5 fired runs", "policy rows of a fired run / policy rows of a run without a restart",
        r1 / r0, f"derived: total_transitions (rows fired=True) / total_transitions (rows fired=False) of {F_PER4}",
        ["p4"])


# ---------------------------------------------------------------------------------- output
COLS = ["table", "row_group", "row", "column_or_q", "statistic", "value", "source"]


def write_csv() -> None:
    """tables/T4.csv"""
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLS)
        for r in ROWS:
            w.writerow([r[c] for c in COLS])


# ---------------------------------------------------------------------------------- markdown
class View:
    """Lookup of the rows by (row_group, row, column_or_q, statistic substring); remembers the footnotes used."""

    def __init__(self) -> None:
        self.order = list(FOOT)
        self.used: List[str] = []

    def raw(self, group: str, row: str, col: str, stat: str) -> Any:
        """The value of the one row that matches (error if none or several)."""
        m = [r for r in ROWS if r["row_group"] == group and r["row"] == row and r["column_or_q"] == col
             and stat in r["statistic"]]
        if len(m) != 1:
            raise KeyError((group, row, col, stat, len(m)))
        self.used.extend(m[0]["_fn"])
        return m[0]["_raw"]

    def cite(self, *ids: str) -> str:
        """Footnote numbers of the given footnote ids, e.g. ``[12,15]``."""
        return "[" + ",".join(str(self.order.index(i) + 1) for i in sorted(ids, key=self.order.index)) + "]"

    def src(self) -> str:
        """Footnote numbers used since the last call, e.g. ``[1,3]``."""
        ids = sorted({self.order.index(i) + 1 for i in self.used})
        self.used = []
        return "[" + ",".join(str(i) for i in ids) + "]"


def mf(x: Any) -> str:
    """Markdown text of a value: counts as integers, other floats rounded to 4 significant digits."""
    if isinstance(x, (bool, int)):
        return str(x)
    if isinstance(x, float):
        return str(int(x)) if x.is_integer() and abs(x) < 1e15 else f"{x:.4g}"
    return str(x)


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """A pipe table (pipes inside cells are escaped)."""
    esc = lambda c: str(c).replace("|", "\\|")  # noqa: E731
    out = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return out + [""]


def write_md() -> None:
    """tables/T4.md: the readable rendering of the rows of T4.csv (every number comes from ROWS)."""
    V = View()
    L: List[str] = []
    L += ["# T4. Method comparison and cost", "",
          "Data table. Every number below is a row of `T4.csv` (one value per row, with its source); this file "
          "only renders those rows. Floats are rounded for reading, the CSV has the full precision. Bracketed "
          "numbers refer to the sources listed at the end. bw is the arm of T3-1.0; T3-1.1 is bw plus the "
          "verifier-gated restart of stage 3.", ""]

    # ---- 1. T3-R1 comparison
    L += ["## 1. T3-R1 comparison of bw, jt and jt_es (T = 3)", "", "### 1.1 Pass counts per q", ""]
    rows = []
    for arm in ARMS:
        for q in C.QS:
            c = f"q={q}"
            n = V.raw(G_PASS, arm, c, "(n_planned)")
            rows.append([arm, str(q), f"{mf(V.raw(G_PASS, arm, c, '(run_pass)'))} of {mf(n)}",
                         mf(V.raw(G_PASS, arm, c, "(G-A_pass)")), mf(V.raw(G_PASS, arm, c, "(G-F_pass)")),
                         mf(V.raw(G_PASS, arm, c, "(G-N_pass)")), V.src()])
    L += md_table(["arm", "q", "run_pass (runs)", "G-A_pass", "G-F_pass", "G-N_pass", "Src"], rows)
    L += ["The run-pass rule at T = 3 is G-A, G-F and G-N (gate limits in the last table of this section). G-A, G-F and G-N counts "
          "are the number of runs that pass that gate.", ""]

    L += ["### 1.2 Mean paired differences of the full-domain dynamic best-response gain G_max,full/ΔW", ""]
    rows = []
    for h, a, b in (("H1", "bw", "jt"), ("H2", "bw", "jt_es"), ("H3", "jt_es", "jt")):
        hr = f"{h}: {a} - {b}"
        for q in C.QS:
            c = f"q={q}"
            rows.append([hr, str(q), mf(V.raw(G_DIFF, hr, c, "(mean_diff)")),
                         f"[{mf(V.raw(G_DIFF, hr, c, '(ci95_lo)'))}, {mf(V.raw(G_DIFF, hr, c, '(ci95_hi)'))}]",
                         mf(V.raw(G_DIFF, hr, c, "(n_pairs)")), mf(V.raw(G_DIFF, hr, c, "(ci_hi_below_0)")),
                         mf(V.raw(G_DIFF, hr, "q=50 and q=60", "(hypothesis_supported)")), V.src()])
    L += md_table(["hypothesis (a - b)", "q", "mean difference", "95 % interval", "paired seeds",
                   "upper end below 0", "hypothesis supported (both q)", "Src"], rows)

    L += ["### 1.3 Interval method", ""]
    nb = V.raw(G_METHOD, "all hypotheses", "H1, H2, H3", "(n_boot)")
    sd = V.raw(G_METHOD, "all hypotheses", "H1, H2, H3", "bootstrap_seed")
    pc = V.raw(G_METHOD, "all hypotheses", "H1, H2, H3", "percentiles")
    L += [f"Percentile bootstrap of the mean paired difference: {mf(nb)} resamples of the seed pairs, "
          f"bootstrap seed {mf(sd)}, interval = the {pc.replace('; ', ' and ')} percentiles of the resampled "
          f"means. A hypothesis is supported when the upper end is below 0 at both q. {V.src()}", ""]

    L += ["### 1.4 Gate limits behind the pass counts", ""]
    rows = []
    for gate, stat in (("G-A", "eta_T_over_dw"), ("G-A", "rmse_pos_T_over_eT0"), ("G-A", "tail_mean_T_over_eT0"),
                       ("G-F", "Gmax_full_over_dw"), ("G-N", "|eta_T_over_dw"), ("G-N", "|Gmax_full_over_dw")):
        rows.append([gate, stat.lstrip("|") + (" (development tier minus final tier, absolute value)"
                                                if gate == "G-N" else ""),
                     "<= " + mf(V.raw(G_LIMITS, gate, "T = 3", stat)), V.src()])
    L += md_table(["gate", "quantity", "gate limit", "Src"], rows)

    # ---- 2. per arm
    L += ["## 2. Per arm", "", "### 2.1 Policy rows and updates (budget_check over the 40 runs of each arm)", ""]
    items = [("runs", "(n_runs)"), ("updates per run", "(updates_expected)"),
             ("policy rows per run", "(rows_total_expected)"),
             ("policy rows at stage 3 per run", "(rows_t3)"), ("policy rows at stage 2 per run", "(rows_t2)"),
             ("policy rows at stage 1 per run", "(rows_t1)"), ("episodes per update", "episodes per update"),
             ("policy rows per update", "policy rows per update"),
             ("minibatch steps per run", "(minibatch_steps_expected)"),
             ("runs with the stored updates equal to the expected", "(updates_equal)"),
             ("runs with the stored policy rows equal to the expected", "(rows_total_equal)"),
             ("runs with the stored policy rows by stage equal to the expected", "(rows_by_stage_equal)"),
             ("runs with the stored minibatch steps equal to the expected", "(minibatch_steps_equal)")]
    rows = []
    for lab, key in items:
        rows.append([lab] + [mf(V.raw(G_BUDGET, arm, "all 40 runs", key)) for arm in ARMS] + [V.src()])
    L += md_table(["item"] + list(ARMS) + ["Src"], rows)
    L += ["An update of bw trains one stage on episodes of that stage (one policy row per episode). An update of "
          "jt or jt_es plays whole-game episodes, which give a policy row at every stage from the start stage on, "
          "so the numbers of updates differ between the arms while the total policy rows are nearly equal.", ""]

    L += ["### 2.2 Runtime: median wall time per run (seconds)", ""]
    rows = []
    for arm in ARMS:
        rows.append([arm] + [mf(V.raw(G_RUNTIME, arm, c, "(wall_sec)")) for c in ("q=50", "q=60", "both q")]
                    + [V.src()])
    L += md_table(["arm", "q=50", "q=60", "both q", "Src"], rows)
    w1 = V.raw(G_LAUNCH, "T3-R1 (bw, jt, jt_es)", "all runs", "(workers)")
    w4 = V.raw(G_LAUNCH, "T3-R4 (bwA)", "all runs", "(workers)")
    n1 = V.raw(G_LAUNCH, "T3-R1 (bw, jt, jt_es)", "all runs", "(nproc)")
    n4 = V.raw(G_LAUNCH, "T3-R4 (bwA)", "all runs", "(nproc)")
    L += [f"Context: the runs of a launch ran concurrently, T3-R1 with {mf(w1)} single-threaded worker processes "
          f"on {mf(n1)} CPU cores and T3-R4 with {mf(w4)} on {mf(n4)}. {V.src()}", ""]

    L += ["### 2.3 Network structure of the evaluated policy", ""]
    net_items = ["hidden layers of the actor", "width of each hidden layer", "activation of the hidden layers",
                 "inputs of the actor", "outputs of the actor", "Beta mean parameter mu", "mu_clamp",
                 "Beta concentration c", "c_min", "Beta parameters", "action and effort", "action_clamp",
                 "e_min and e_max", "mean effort of the evaluated policy", "critic (training only)",
                 "do the stages share one network", "distinct networks in the evaluated policy",
                 "how the stage networks arise"]
    same, differ = [], []
    for it in net_items:
        vals = [V.raw(G_NET, arm, "all 40 runs", it) for arm in ARMS]
        if len({str(v) for v in vals}) == 1:
            same.append([it, mf(vals[0]), V.src()])
        else:
            differ.append([it] + [mf(v) for v in vals] + [V.src()])
    L += ["Identical in the three arms:", ""] + md_table(["item", "value (bw, jt, jt_es)", "Src"], same)
    L += ["Different between the arms:", ""] + md_table(["item"] + list(ARMS) + ["Src"], differ)

    L += ["### 2.4 State-start design (episodes started per update, with the share of the episodes of an update)", ""]
    rows = []
    stage_cols = ("root (stage 1, d = 0)", "stage 2", "stage 3")
    for lab, arm, upd_key, upd_col in (
            ("bw, phase of stage 3", "bw", "updates in the phase", "phase"),
            ("bw, phase of stage 2", "bw", "updates in the phase", "phase"),
            ("bw, phase of stage 1", "bw", "updates in the phase", "phase"),
            ("jt", "jt", "updates with this start mix", "all start stages"),
            ("jt_es", "jt_es", "updates with this start mix", "all start stages")):
        row_key = lab if arm == "bw" else arm
        cells = [f"{mf(V.raw(G_START, row_key, c, 'episodes started'))} "
                 f"({mf(V.raw(G_START, row_key, c, 'share of the episodes'))})" for c in stage_cols]
        rows.append([lab] + cells + [mf(V.raw(G_START, row_key, upd_col, upd_key)), V.src()])
    L += md_table(["arm / phase", "root starts (share)", "stage 2 starts (share)", "stage 3 starts (share)",
                   "updates", "Src"], rows)
    for arm in ARMS:
        s1 = V.raw(G_START, arm, "stage 1", "distribution of the start score gap d")
        s23 = V.raw(G_START, arm, "stages 2 and 3", "distribution of the start score gap d")
        L += [f"- {arm}: stage 1: {s1}. Stages 2 and 3: {s23}. {V.src()}"]
        if arm != "bw":
            L += [f"  Episodes: {V.raw(G_START, arm, 'all start stages', 'episode played')}. {V.src()}"]
    L += [""]

    # ---- 3. T3-1.1 retry cost
    L += ["## 3. T3-1.1 retry cost (T3-R4 confirmation)", "", "### 3.1 Restart rule", ""]
    dg = V.raw(G_RESTART, "stage3_retry", "T3-1.1", "delta_gt")
    mr = V.raw(G_RESTART, "stage3_retry", "T3-1.1", "max_retries")
    b3 = V.raw(G_RESTART, "stage3_retry", "T3-1.1", "updates of the stage-3 phase")
    L += [f"stage3_retry: delta_gt {mf(dg)}, max_retries {mf(mr)}; one attempt of the stage-3 phase is {mf(b3)} "
          f"updates. {V.src()}", ""]

    L += ["### 3.2 Final count and first-attempt count (one confirmation block)", ""]
    rows = []
    for lab, grow in (("final count (after the restart)", "final count (after the restart)"),
                      ("first-attempt count", "first-attempt count")):
        cells = []
        for c in ("q=50", "q=60", "both q"):
            if grow.startswith("final"):
                k = V.raw(G_COUNTS, grow, c, "(pass)")
                n = V.raw(G_COUNTS, grow, c, "(runs)")
            elif c == "both q":
                k = V.raw(G_COUNTS, grow, c, "(trigger.first_attempt_pass)")
                n = V.raw(G_COUNTS, grow, c, "(trigger.n)")
            else:
                k = V.raw(G_COUNTS, grow, c, "runs passing without a restart (derived)")
                n = V.raw(G_COUNTS, grow, c, "runs at this q")
            cells.append(f"{mf(k)} of {mf(n)}")
        rows.append([lab] + cells + [V.src()])
    L += md_table(["count", "q=50", "q=60", "both q", "Src"], rows)
    s0 = V.raw(G_COUNTS, "block of both counts", "q=50", "first seed")
    s1 = V.raw(G_COUNTS, "block of both counts", "q=50", "last seed")
    L += [f"Both counts are for the same confirmation block: seeds {mf(s0)} to {mf(s1)} at q=50 and at q=60, "
          f"{mf(V.raw(G_COUNTS, 'first-attempt count', 'both q', '(trigger.n)'))} runs; the file headline.json holds "
          f"both. {V.src()}",
          f"The first-attempt count is the number of runs that passed and did not fire the restart; a fired run "
          f"counts as a failure of its first attempt (convention). {V.cite('t4')}",
          f"All {mf(V.raw(G_COUNTS, 'first-attempt count', 'both q', 'fired runs that pass'))} fired runs pass after "
          f"the restart. {V.src()}", ""]

    L += ["### 3.3 Fired runs", ""]
    rows = []
    for c in ("q=50", "q=60", "both q"):
        rows.append([c, f"{mf(V.raw(G_FIRED, 'fired runs', c, '(fired)'))} of "
                        f"{mf(V.raw(G_FIRED, 'fired runs', c, '(runs)'))}",
                     mf(V.raw(G_FIRED, "fired runs", c, "(fired_share)")), V.src()])
    L += md_table(["q", "fired runs", "share fired", "Src"], rows)
    fired_names = [r["row"] for r in ROWS if r["row_group"] == G_FIRED and r["statistic"].startswith("seed")]
    rows = []
    for nm in fired_names:
        c = [r["column_or_q"] for r in ROWS if r["row_group"] == G_FIRED and r["row"] == nm][0]
        rows.append([nm, mf(V.raw(G_FIRED, nm, c, "seed")), mf(V.raw(G_FIRED, nm, c, "(passed)")),
                     mf(V.raw(G_FIRED, nm, c, "(total_updates)")), mf(V.raw(G_FIRED, nm, c, "(total_transitions)")),
                     mf(V.raw(G_FIRED, nm, c, "(wall_sec)")), V.src()])
    L += md_table(["fired run", "seed", "run passed", "updates (both attempts)", "policy rows (both attempts)",
                   "wall time (s)", "Src"], rows)

    L += ["### 3.4 Cost per run", ""]
    stats = [("mean updates per run", "mean per run", "(mean_updates)"),
             ("mean policy rows per run", "mean per run", "(mean_policy_rows)"),
             ("ratio of mean updates to a run without a restart", "mean per run", "(ratio_updates_to_not_fired)"),
             ("ratio of mean policy rows to a run without a restart", "mean per run",
              "(ratio_policy_rows_to_not_fired)"),
             ("median wall time per run (s)", "wall time per run", "(median_wall_sec)"),
             ("mean wall time of the fired runs (s)", "wall time per run", "(mean_wall_sec_fired)"),
             ("mean wall time of the runs without a restart (s)", "wall time per run", "(mean_wall_sec_not_fired)")]
    rows = []
    for lab, grow, key in stats:
        cells = []
        for c in ("q=50", "q=60", "both q"):
            try:
                cells.append(mf(V.raw(G_COST, grow, c, key)))
            except KeyError:
                cells.append("none fired")
        rows.append([lab] + cells + [V.src()])
    L += md_table(["statistic", "q=50", "q=60", "both q", "Src"], rows)
    rows = []
    for lab, key_nf, key_f in (("updates per run", "updates per run (total_updates)",
                                 "updates per run, both attempts (total_updates)"),
                                ("policy rows per run", "policy rows per run (total_transitions)",
                                 "policy rows per run, both attempts (total_transitions)")):
        nf_ = V.raw(G_COST, "run without a restart", "all 75 runs", key_nf)
        f_ = V.raw(G_COST, "fired run", "all 5 fired runs", key_f)
        rows.append([lab, mf(nf_), mf(f_), V.src()])
    rows.append(["extra updates of a fired run", "none", mf(V.raw(G_COST, "fired run", "all 5 fired runs",
                                                                 "extra updates")), V.src()])
    rows.append(["updates of a fired run / updates of a run without a restart", "reference",
                 mf(V.raw(G_COST, "fired run", "all 5 fired runs", "updates of a fired run /")), V.src()])
    rows.append(["policy rows of a fired run / policy rows of a run without a restart", "reference",
                 mf(V.raw(G_COST, "fired run", "all 5 fired runs", "policy rows of a fired run /")), V.src()])
    L += md_table(["item", "run without a restart (all runs of that kind)", "fired run (all fired runs)", "Src"], rows)
    L += [f"A fired run repeats the stage-3 phase once, so it costs {mf(b3)} more updates than a run without a "
          f"restart. {V.cite('p4', 'c4cfg')}", ""]

    # ---- sources
    L += ["## Sources", ""]
    for i, key in enumerate(FOOT, 1):
        L += [f"{i}. {FOOT[key]}"]
    L += ["", "Untracked run roots are read only. Per-row sources (file, column, selection, operation) are in the "
          "`source` column of `T4.csv`.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    """Build all rows and write the CSV and the Markdown."""
    cfgs = read_r1_configs()
    part1()
    part2(cfgs)
    part3()
    write_csv()
    write_md()
    print(f"wrote {OUT_CSV} ({len(ROWS)} rows) and {OUT_MD}")


if __name__ == "__main__":
    main()
