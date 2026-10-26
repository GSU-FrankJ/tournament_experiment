"""Table T3 of the P-FIG round: the T = 3 fresh-seed confirmation (protocol T3-1.1, arm bwA).

Reads only stored records (nothing is trained or evaluated) and writes ``tables/T3.csv`` (one value per
row: table, row_group, row, column_or_q, statistic, value, source) and ``tables/T3.md`` (the readable
rendering of the same rows, with a source legend). Deterministic. Every cross-check is exact (``==``);
if one fails, nothing is written and the process exits with status 2.

Records read (repo-relative):
  results/t3_r4/confirmation/eval/per_run.csv
  results/t3_r4/confirmation/analysis/{counts,cost,fired,failures}.csv, headline.json
  protocols/v2_T3_locked_v1_1.json, reports/t3/r1/prereg_criteria.json
and, read-only, the run directories of the untracked T3-R4 root (retry.json, status.json) for the
cross-checks of the retry counts and of the wall times (skipped, with a notice, if that root is absent).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

REL = lambda p: str(Path(p).relative_to(C.ROOT))  # noqa: E731
CONF = C.ROOT / "results" / "t3_r4" / "confirmation"
PR = REL(C.PER_RUN_R4)
CNT, COST = REL(CONF / "analysis" / "counts.csv"), REL(CONF / "analysis" / "cost.csv")
FIRED, FAIL = REL(CONF / "analysis" / "fired.csv"), REL(CONF / "analysis" / "failures.csv")
HEAD = REL(CONF / "analysis" / "headline.json")
PROT_P = C.ROOT / "protocols" / "v2_T3_locked_v1_1.json"
CRIT_P = C.ROOT / "reports" / "t3" / "r1" / "prereg_criteria.json"
PROT, CRIT = REL(PROT_P), REL(CRIT_P)
TABLES = C.OUT / "tables"

CHECKS: List[Tuple[str, bool, str]] = []


def check(what: str, ok: bool, detail: str = "") -> None:
    """Record one exact check (printed at the end; any failure stops before anything is written)."""
    CHECKS.append((what, bool(ok), detail))


# ---------------------------------------------------------------------------------- row store
ROWS: List[Dict[str, str]] = []
KIND: Dict[Tuple[str, str, str, str], str] = {}   # row key -> source kind (md legend letter)


def vtxt(v: Any) -> str:
    """Value text: ints as integers, floats at full round-trip precision."""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return C.fmt(float(v))


def add(group: str, row: str, col: Any, stat: str, value: Any, source: str, kind: str) -> None:
    """Append one row (one value) of the table."""
    ROWS.append({"table": "T3", "row_group": group, "row": row, "column_or_q": str(col),
                 "statistic": stat, "value": vtxt(value), "source": source})
    KIND[(group, row, str(col), stat)] = kind


def get(group: str, row: str, col: Any, stat: str) -> str:
    """The stored CSV text of one row's value (the md is rendered from these texts only)."""
    hit = [r["value"] for r in ROWS if (r["row_group"], r["row"], r["column_or_q"], r["statistic"])
           == (group, row, str(col), stat)]
    assert len(hit) == 1, (group, row, col, stat, hit)
    return hit[0]


# ---------------------------------------------------------------------------------- load
def load() -> Dict[str, Any]:
    """Read every stored record used by the table."""
    d: Dict[str, Any] = {}
    d["pr"] = C.load_per_run(C.PER_RUN_R4)
    rt = {"float_precision": "round_trip"}
    d["counts"] = pd.read_csv(CONF / "analysis" / "counts.csv", dtype={"q": str}, **rt)
    d["cost"] = pd.read_csv(CONF / "analysis" / "cost.csv", dtype={"q": str}, **rt)
    d["fired"] = pd.read_csv(CONF / "analysis" / "fired.csv", **rt)
    d["fail"] = pd.read_csv(CONF / "analysis" / "failures.csv", **rt)
    d["head"] = C.read_json(CONF / "analysis" / "headline.json")
    d["prot"] = C.read_json(PROT_P)
    d["crit"] = C.read_json(CRIT_P)
    return d


# ---------------------------------------------------------------------------------- checks on the inputs
def check_inputs(d: Dict[str, Any]) -> Dict[str, Any]:
    """Exact checks of the stored records against the facts of the task; returns the gate limits."""
    pr, prot, crit, head = d["pr"], d["prot"], d["crit"], d["head"]
    check("per_run.csv: 80 rows, 40 per q, arm bwA, T = 3",
          len(pr) == 80 and (pr.groupby("q").size() == 40).all() and set(pr["arm"]) == {"bwA"}
          and set(pr["T"]) == {3} and set(pr["q"]) == {50, 60})
    seeds = set(range(80501, 80541))
    check("per_run.csv: seeds 80501-80540 at both q, each once",
          all(set(pr.loc[pr.q == q, "seed"]) == seeds and pr.loc[pr.q == q, "seed"].is_unique
              for q in C.QS))
    check("protocol confirmation.seed_block == [80501, 80540], arm bwA, n_runs 80",
          prot["confirmation"]["seed_block"] == [80501, 80540] and prot["confirmation"]["arm"] == "bwA"
          and prot["confirmation"]["n_runs"] == 80)
    check("every run: status done, exit_code 0, global_rng_ok, valid, dev_valid",
          bool((pr["status"] == "done").all() and (pr["exit_code"] == 0).all()
               and pr["global_rng_ok"].all() and pr["valid"].all() and pr["dev_valid"].all()))
    # gate limits: protocol == prereg_criteria.json
    check("protocol gates == reports/t3/r1/prereg_criteria.json gates (all keys, all values)",
          prot["gates"] == crit["gates"])
    g = prot["gates"]
    lim = {"eta": g["G-A"]["eta_T_over_dw"], "rmse": g["G-A"]["rmse_pos_T_over_eT0"],
           "tail": g["G-A"]["tail_mean_T_over_eT0"], "gmax": g["G-F"]["Gmax_full_over_dw"],
           "n_eta": g["G-N"]["eta_T_over_dw_dev_minus_final_abs"],
           "n_gmax": g["G-N"]["Gmax_full_over_dw_dev_minus_final_abs"]}
    check("run_pass at T = 3 is G-A, G-F, G-N", g["run_pass"]["3"] == ["G-A", "G-F", "G-N"])
    # the gate columns follow from the stored values and the limits (<= inclusive)
    ga = ((pr.eta_T_over_dw <= lim["eta"]) & (pr.rmse_pos_T_over_eT0 <= lim["rmse"])
          & (pr.tail_mean_T_over_eT0 <= lim["tail"]))
    gf = pr.Gmax_full_over_dw <= lim["gmax"]
    gn_e = (pr.dev_eta_T_over_dw - pr.eta_T_over_dw).abs() <= lim["n_eta"]
    gn_g = (pr.dev_Gmax_full_over_dw - pr.Gmax_full_over_dw).abs() <= lim["n_gmax"]
    check("80 rows: ga_eta, ga_rmse, ga_tail flags == value <= limit",
          bool((pr.ga_eta == (pr.eta_T_over_dw <= lim["eta"])).all()
               and (pr.ga_rmse == (pr.rmse_pos_T_over_eT0 <= lim["rmse"])).all()
               and (pr.ga_tail == (pr.tail_mean_T_over_eT0 <= lim["tail"])).all()))
    check("80 rows: gate_G_A, gate_G_F == recomputed from the stored values and limits",
          bool((pr.gate_G_A == ga).all() and (pr.gate_G_F == gf).all()))
    check("80 rows: gn_eta, gn_gmax, gate_G_N == recomputed |dev - final| <= limit",
          bool((pr.gn_eta == gn_e).all() and (pr.gn_gmax == gn_g).all()
               and (pr.gate_G_N == (gn_e & gn_g)).all()))
    check("80 rows: gates_pass == gate_G_A and gate_G_F and gate_G_N",
          bool((pr.gates_pass == (pr.gate_G_A & pr.gate_G_F & pr.gate_G_N)).all()))
    return lim


# ---------------------------------------------------------------------------------- the table rows
QUANT = [
    ("rmse_pos_T_over_eT0",
     "rmse_pos_T_over_eT0 (RMSE of the mean effort in the positive-effort region at stage 3, "
     "relative to the closed-form equilibrium at the tie point)"),
    ("tail_mean_T_over_eT0",
     "tail_mean_T_over_eT0 (mean effort in the theoretical zero-effort tails at stage 3, "
     "relative to the closed-form equilibrium at the tie point)"),
    ("peak_T_signed",
     "peak_T_signed (signed tie-point error of the stage-3 mean effort, relative to the "
     "closed-form equilibrium at d = 0)"),
    ("peak_T_abs",
     "peak_T_abs (absolute tie-point error of the stage-3 mean effort, relative to the "
     "closed-form equilibrium at d = 0)"),
    ("eta_T_over_dw", "eta_T_over_dw (η_3/ΔW)"),
    ("EXP_root_over_dw", "EXP_root_over_dw (root exploitability)"),
]
G_PASS = "pass counts and pre-registered failure limits"
G_STAT = "per-q statistics over the 40 runs (median, minimum, maximum)"
G_GMAX = "full-domain dynamic best-response gain G_max,full/ΔW (final tier)"
G_RETRY = "stage-3 restart (retry) counts"
G_COST = "cost per run"
G_FAIL = "seed 80502 (q = 60), the sole failure"
RUN_ROW = "T3-1.1 confirmation, arm bwA, seeds 80501-80540"


def build_rows(d: Dict[str, Any], lim: Dict[str, float]) -> None:
    """Compute every table value from the stored records (with its cross-checks) and add the rows."""
    pr, prot, head, cnt = d["pr"], d["prot"], d["head"], d["counts"].set_index("q")
    cost, fired, fail = d["cost"].set_index("q"), d["fired"], d["fail"]
    qsel = {q: pr[pr.q == q] for q in C.QS}
    qsel["both"] = pr

    # ---- pass counts, failure limits, verdict
    n = {k: len(v) for k, v in qsel.items()}
    npass = {k: int(v["gates_pass"].sum()) for k, v in qsel.items()}
    nfail = {k: n[k] - npass[k] for k in n}
    check("pass counts 40/40 (q=50), 39/40 (q=60), 79/80 overall (STOP rule)",
          (npass[50], n[50], npass[60], n[60], npass["both"], n["both"]) == (40, 40, 39, 40, 79, 80),
          f"{npass[50]}/{n[50]}, {npass[60]}/{n[60]}, {npass['both']}/{n['both']}")
    check("counts.csv pass/runs == recomputed (50, 60, both)",
          all(int(cnt.loc[str(k), "pass"]) == npass[k] and int(cnt.loc[str(k), "runs"]) == n[k]
              for k in (50, 60, "both")))
    hc = head["counts"]
    check("headline.json counts == recomputed",
          (hc["pass_q50"], hc["n_q50"], hc["pass_q60"], hc["n_q60"], hc["pass_total"], hc["n_total"],
           hc["fail_q50"], hc["fail_q60"], hc["fail_total"])
          == (npass[50], n[50], npass[60], n[60], npass["both"], n["both"], nfail[50], nfail[60],
              nfail["both"]))
    rule = prot["confirmation"]["rule"]
    lim_q, lim_t = rule["max_failures_per_q"], rule["max_failures_total"]
    check("failure limits: protocol 4 per q and 6 of 80 (== headline.json verdict.rule)",
          (lim_q, lim_t, rule["runs_per_q"], rule["n_runs"]) == (4, 6, 40, 80)
          and head["verdict"]["rule"] == {"max_failures_total": 6, "max_failures_per_q": 4})
    within = {50: nfail[50] <= lim_q, 60: nfail[60] <= lim_q, "both": nfail["both"] <= lim_t}
    verdict = int(all(within.values()))
    check("verdict from the rule == headline.json verdict.verdict ('pass')",
          verdict == 1 and head["verdict"]["verdict"] == "pass"
          and head["verdict"]["q50_condition"] is True and head["verdict"]["q60_condition"] is True
          and head["verdict"]["total_condition"] is True)
    check("first-attempt count (not tabulated in T3): headline 74 == 79 passing - 5 fired passing",
          head["trigger"]["first_attempt_pass"] == 74
          and npass["both"] - int(fired["passed"].sum()) == 74,
          f"headline {head['trigger']['first_attempt_pass']}, recomputed "
          f"{npass['both'] - int(fired['passed'].sum())}")

    src_run = lambda k: (f"count of the rows with q={k} in {PR}" if k != "both"   # noqa: E731
                         else f"count of all rows in {PR}")
    for k in (50, 60, "both"):
        sel = "all 80 rows" if k == "both" else f"the 40 rows with q={k}"
        add(G_PASS, RUN_ROW, k, "runs", n[k], src_run(k), "A")
        add(G_PASS, RUN_ROW, k, "runs passing (G-A, G-F and G-N)", npass[k],
            f"sum of column gates_pass over {sel} of {PR}; equals column pass of {CNT} (row q={k})", "A")
        add(G_PASS, RUN_ROW, k, "runs failing", nfail[k],
            f"runs minus runs passing (derived); equals {HEAD} key counts.fail_"
            + ("total" if k == "both" else f"q{k}"), "A")
        add(G_PASS, RUN_ROW, k, "pre-registered failure limit (maximum failing runs)",
            lim_t if k == "both" else lim_q,
            f"{PROT} key confirmation.rule.max_failures_" + ("total" if k == "both" else "per_q"), "B")
        add(G_PASS, RUN_ROW, k, "failures within the limit (1 = yes, 0 = no)", int(within[k]),
            "1 if runs failing <= failure limit (derived); equals "
            + f"{HEAD} key verdict.{'total' if k == 'both' else 'q' + str(k)}_condition", "B")
    add(G_PASS, "confirmation verdict", "both", "verdict (1 = pass, 0 = fail)", verdict,
        f"{PROT} key confirmation.rule applied to the failing counts (all three conditions hold); "
        f"equals {HEAD} key verdict.verdict (= pass)", "B")

    # ---- per-q statistics
    for col, label in QUANT:
        for q in C.QS:
            x = qsel[q][col].to_numpy(dtype=float)
            med, lo, hi = float(np.median(x)), float(np.min(x)), float(np.max(x))
            if col != "peak_T_abs":
                check(f"median of {col}, q={q}: numpy == counts.csv median_{col}",
                      med == float(cnt.loc[str(q), f"median_{col}"]))
            base = f"{{}} over the {len(x)} rows with q={q} (arm bwA), column {col} of {PR}"
            add(G_STAT, label, q, "median", med,
                base.format("median (numpy: mean of the two middle values)")
                + (f"; equals column median_{col} of {CNT}" if col != "peak_T_abs" else ""), "C")
            add(G_STAT, label, q, "minimum", lo, base.format("minimum"), "C")
            add(G_STAT, label, q, "maximum", hi, base.format("maximum"), "C")
    for col in ("eta_T_over_dw", "rmse_pos_T_over_eT0", "tail_mean_T_over_eT0", "peak_T_signed",
                "Gmax_full_over_dw", "EXP_root_over_dw"):
        check(f"median of {col}, both q: numpy == counts.csv median_{col}",
              float(np.median(pr[col].to_numpy(dtype=float))) == float(cnt.loc["both", f"median_{col}"]))

    # ---- G_max,full: median and worst
    for q in C.QS:
        s = qsel[q]
        x = s["Gmax_full_over_dw"].to_numpy(dtype=float)
        med, worst = float(np.median(x)), float(np.max(x))
        iw = int(np.argmax(x))
        wrow = s.iloc[iw]
        check(f"q={q}: unique maximum of Gmax_full_over_dw", int((x == worst).sum()) == 1)
        check(f"q={q}: median Gmax_full == counts.csv median_Gmax_full_over_dw",
              med == float(cnt.loc[str(q), "median_Gmax_full_over_dw"]))
        check(f"q={q}: worst Gmax_full == counts.csv worst_Gmax_full",
              worst == float(cnt.loc[str(q), "worst_Gmax_full"]))
        check(f"q={q}: worst run == counts.csv worst_Gmax_full_run",
              str(wrow["run"]) == str(cnt.loc[str(q), "worst_Gmax_full_run"]), str(wrow["run"]))
        pf = f"over the {len(x)} rows with q={q} (arm bwA), column Gmax_full_over_dw of {PR}"
        row = "G_max,full/ΔW (Gmax_full_over_dw)"
        add(G_GMAX, row, q, "median", med,
            f"median {pf} (numpy); equals column median_Gmax_full_over_dw of {CNT}", "D")
        add(G_GMAX, row, q, "worst (maximum)", worst,
            f"maximum {pf}; equals column worst_Gmax_full of {CNT}", "D")
        add(G_GMAX, row, q, "seed of the worst run", int(wrow["seed"]),
            f"column seed of the row attaining the maximum {pf}; run name equals column "
            f"worst_Gmax_full_run of {CNT}", "D")

    # ---- retry counts
    cfired = {k: int(cost.loc[str(k), "fired"]) for k in (50, 60, "both")}
    ffired = {q: int((fired.q == q).sum()) for q in C.QS}
    check("retry counts: cost.csv fired == (5, 0, 5)", (cfired[50], cfired[60], cfired["both"]) == (5, 0, 5))
    check("retry counts: fired.csv rows per q == cost.csv fired",
          (ffired[50], ffired[60], len(fired)) == (cfired[50], cfired[60], cfired["both"]))
    check("retry counts: headline.json trigger.fired == 5 and trigger.runs == fired.csv runs",
          head["trigger"]["fired"] == 5 and head["trigger"]["n"] == 80
          and sorted(head["trigger"]["runs"]) == sorted(fired["run"]))
    check("retry counts: all fired runs are at q=50 and pass",
          set(fired.q) == {50} and bool(fired["passed"].all()))
    for k in (50, 60, "both"):
        add(G_RETRY, "runs whose stage-3 restart fired", k, "fired runs", cfired[k],
            f"column fired of {COST} (row q={k}); equals the number of rows of {FIRED}"
            + ("" if k == "both" else f" with q={k}")
            + (f", {HEAD} key trigger.fired," if k == "both" else "")
            + " and the number of run directories holding a retry.json (untracked root "
            "t3-r4-lock-confirm/results/t3_r4/confirmation/q{q}/seed{seed}/bwA/, read-only)", "E")
        add(G_RETRY, "runs whose stage-3 restart fired", k, "fired share of the runs",
            float(cost.loc[str(k), "fired_share"]), f"column fired_share of {COST} (row q={k})", "E")

    # ---- cost per run (as stored)
    base_up = sum(prot["settings"]["budgets"].values())
    base_rows = base_up * prot["settings"]["episodes_per_update"]
    check("run without a restart: 3800 updates and 1,945,600 policy rows (protocol budgets, 512 episodes)",
          (base_up, base_rows) == (3800, 1945600), f"{base_up}, {base_rows}")
    check("cost.csv q=60 (no fired run): mean_updates 3800, mean_policy_rows 1945600",
          (float(cost.loc["60", "mean_updates"]), float(cost.loc["60", "mean_policy_rows"]))
          == (3800.0, 1945600.0))
    add(G_COST, "updates of a run without a restart", "all", "reference", base_up,
        f"sum of the values of {PROT} key settings.budgets (1600 + 1600 + 600); equals mean_updates "
        f"of {COST} (row q=60, no fired run)", "F")
    add(G_COST, "policy rows of a run without a restart", "all", "reference", base_rows,
        f"{PROT} key settings.episodes_per_update (512) times the 3800 updates (derived); equals "
        f"mean_policy_rows of {COST} (row q=60)", "F")
    for k in (50, 60, "both"):
        r = cost.loc[str(k)]
        mu, mr = float(r["mean_updates"]), float(r["mean_policy_rows"])
        check(f"cost.csv q={k}: ratio columns == mean / (3800, 1945600) exactly",
              float(r["ratio_updates_to_not_fired"]) == mu / 3800.0
              and float(r["ratio_policy_rows_to_not_fired"]) == mr / (3800 * 512))
        add(G_COST, "updates per run", k, "mean over the runs", mu,
            f"column mean_updates of {COST} (row q={k})", "F")
        add(G_COST, "policy rows per run", k, "mean over the runs", mr,
            f"column mean_policy_rows of {COST} (row q={k})", "F")
        add(G_COST, "ratio of the mean updates to the 3800 of a run without a restart", k, "ratio",
            float(r["ratio_updates_to_not_fired"]),
            f"column ratio_updates_to_not_fired of {COST} (row q={k})", "F")
        add(G_COST, "ratio of the mean policy rows to the 1945600 of a run without a restart", k,
            "ratio", float(r["ratio_policy_rows_to_not_fired"]),
            f"column ratio_policy_rows_to_not_fired of {COST} (row q={k})", "F")
        add(G_COST, "wall time per run (seconds)", k, "median over the runs",
            float(r["median_wall_sec"]),
            f"column median_wall_sec of {COST} (row q={k}); median over the runs of total_wall_sec of "
            "each run's status.json, the whole run including any restarted attempt", "F")

    # ---- seed 80502
    f = pr[(pr.q == 60) & (pr.seed == 80502)]
    assert len(f) == 1
    f = f.iloc[0]
    check("the only failing run is q=60 seed 80502",
          list(pr.loc[~pr.gates_pass, ["q", "seed"]].itertuples(index=False, name=None)) == [(60, 80502)])
    check("80502: gate_G_A True, gate_G_F False, gate_G_N False (gn_eta True, gn_gmax False)",
          (bool(f.gate_G_A), bool(f.gate_G_F), bool(f.gate_G_N), bool(f.gn_eta), bool(f.gn_gmax))
          == (True, False, False, True, False))
    check("80502: Gmax_full_t == 2 and G_max_t2_over_dw == Gmax_full_over_dw (stage 2 holds the max)",
          int(f.Gmax_full_t) == 2 and float(f.G_max_t2_over_dw) == float(f.Gmax_full_over_dw)
          and float(f.G_max_t1_over_dw) < float(f.Gmax_full_over_dw)
          and float(f.G_max_t3_over_dw) < float(f.Gmax_full_over_dw))
    fr = fail[(fail.q == 60) & (fail.seed == 80502)]
    check("failures.csv: exactly one row (q=60, seed 80502), gates gate_G_F;gate_G_N", len(fr) == 1
          and fr.iloc[0]["failing_gates"] == "gate_G_F;gate_G_N" and len(fail) == 1)
    fr = fr.iloc[0]
    d_gm = abs(float(f.dev_Gmax_full_over_dw) - float(f.Gmax_full_over_dw))
    d_eta = abs(float(f.dev_eta_T_over_dw) - float(f.eta_T_over_dw))
    exp_txt = (f"Gmax_full = {float(f.Gmax_full_over_dw):.6g} (limit 0.01) at stage {int(f.Gmax_full_t)}; "
               f"|dev - final| of Gmax_full = {d_gm:.6g} (limit 0.001) at stage {int(f.Gmax_full_t)}")
    check("failures.csv failing_quantities text == text rebuilt from per_run.csv",
          fr["failing_quantities"] == exp_txt, fr["failing_quantities"])
    check("failures.csv eta_3, rmse_pos, tail_mean, Gmax_full, Gmax_full_stage == per_run.csv",
          (float(fr.eta_3), float(fr.rmse_pos), float(fr.tail_mean), float(fr.Gmax_full),
           float(fr.Gmax_full_stage))
          == (float(f.eta_T_over_dw), float(f.rmse_pos_T_over_eT0), float(f.tail_mean_T_over_eT0),
              float(f.Gmax_full_over_dw), float(f.Gmax_full_t)))
    check("headline.json failures == [t3r4_q60_s80502_bwA]",
          [x["run"] for x in head["failures"]] == ["t3r4_q60_s80502_bwA"]
          and head["failures"][0]["failing_gates"] == "gate_G_F;gate_G_N")
    check("80502 not fired (fired.csv has no row for it)", not (fired.seed == 80502).any())

    sp = f"{PR} (row q=60, seed 80502)"
    for gate, col in (("G-A", "gate_G_A"), ("G-F", "gate_G_F"), ("G-N", "gate_G_N")):
        add(G_FAIL, f"gate {gate}", 60, "gate result (1 = pass, 0 = fail)", int(bool(f[col])),
            f"column {col} of {sp}; failing gates also listed in column failing_gates of {FAIL}", "G")
    qrows = [
        ("G-A", "eta_T_over_dw (η_3/ΔW)", float(f.eta_T_over_dw), lim["eta"],
         f"column eta_T_over_dw of {sp}", f"{PROT} key gates.G-A.eta_T_over_dw"),
        ("G-A", "rmse_pos_T_over_eT0", float(f.rmse_pos_T_over_eT0), lim["rmse"],
         f"column rmse_pos_T_over_eT0 of {sp}", f"{PROT} key gates.G-A.rmse_pos_T_over_eT0"),
        ("G-A", "tail_mean_T_over_eT0", float(f.tail_mean_T_over_eT0), lim["tail"],
         f"column tail_mean_T_over_eT0 of {sp}", f"{PROT} key gates.G-A.tail_mean_T_over_eT0"),
        ("G-F", "G_max,full/ΔW (Gmax_full_over_dw)", float(f.Gmax_full_over_dw), lim["gmax"],
         f"column Gmax_full_over_dw of {sp}", f"{PROT} key gates.G-F.Gmax_full_over_dw"),
        ("G-N", "|development - final| of η_3/ΔW", d_eta, lim["n_eta"],
         f"|dev_eta_T_over_dw - eta_T_over_dw| of {sp} (derived)",
         f"{PROT} key gates.G-N.eta_T_over_dw_dev_minus_final_abs"),
        ("G-N", "|development - final| of G_max,full/ΔW", d_gm, lim["n_gmax"],
         f"|dev_Gmax_full_over_dw - Gmax_full_over_dw| of {sp} (derived); equals the value "
         f"in column failing_quantities of {FAIL}",
         f"{PROT} key gates.G-N.Gmax_full_over_dw_dev_minus_final_abs"),
    ]
    for gate, name, val, limit, vsrc, lsrc in qrows:
        add(G_FAIL, f"{gate}: {name}", 60, "value", val, vsrc, "G")
        add(G_FAIL, f"{gate}: {name}", 60, "limit (<= inclusive)", limit, lsrc, "G")
    add(G_FAIL, "G-N input: development-tier G_max,full/ΔW", 60, "value",
        float(f.dev_Gmax_full_over_dw), f"column dev_Gmax_full_over_dw of {sp}", "G")
    add(G_FAIL, "G-N input: development-tier η_3/ΔW", 60, "value",
        float(f.dev_eta_T_over_dw), f"column dev_eta_T_over_dw of {sp}", "G")
    add(G_FAIL, "stage of G_max,full", 60, "stage (Gmax_full_t)", int(f.Gmax_full_t),
        f"column Gmax_full_t of {sp}; equals column Gmax_full_stage of {FAIL}", "G")
    add(G_FAIL, "location of G_max,full", 60, "score gap d (Gmax_full_d)", float(f.Gmax_full_d),
        f"column Gmax_full_d of {sp}", "G")


# ---------------------------------------------------------------------------------- run-directory cross-checks
def check_run_dirs(d: Dict[str, Any]) -> None:
    """Read-only cross-checks against the untracked T3-R4 run directories (retry.json, status.json)."""
    if not C.R4_ROOT.is_dir():
        print("NOTICE: untracked root C.R4_ROOT not found; run-directory cross-checks skipped")
        return
    pr, fired, cost = d["pr"], d["fired"], d["cost"].set_index("q")
    fired_dirs: List[Tuple[int, int]] = []
    upd: Dict[Any, List[int]] = {50: [], 60: []}
    rows: Dict[Any, List[int]] = {50: [], 60: []}
    wall: Dict[Any, List[float]] = {50: [], 60: []}
    for q, seed in pr[["q", "seed"]].itertuples(index=False, name=None):
        rd = C.r4_run_dir(int(q), int(seed))
        rj = rd / "retry.json"
        if rj.is_file():
            j = C.read_json(rj)
            fired_dirs.append((int(q), int(seed)))
            assert j["fired"] is True and j["frozen_attempt"] == 2
            upd[q].append(int(j["totals"]["updates"]))
            rows[q].append(int(j["totals"]["transitions"]))
        else:
            upd[q].append(3800)
            rows[q].append(3800 * 512)
        wall[q].append(float(C.read_json(rd / "status.json")["total_wall_sec"]))
    check("run dirs: retry.json (fired true) in exactly the 5 runs of fired.csv",
          sorted(fired_dirs) == sorted(zip(fired.q.astype(int), fired.seed.astype(int)))
          and len(fired_dirs) == 5, str(sorted(fired_dirs)))
    for q in C.QS:
        check(f"run dirs q={q}: mean updates / policy rows (retry.json totals, else 3800 / 1945600) "
              "== cost.csv", float(np.mean(upd[q])) == float(cost.loc[str(q), "mean_updates"])
              and float(np.mean(rows[q])) == float(cost.loc[str(q), "mean_policy_rows"]))
        check(f"run dirs q={q}: median of status.json total_wall_sec == cost.csv median_wall_sec",
              float(np.median(wall[q])) == float(cost.loc[str(q), "median_wall_sec"]))
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    span = []
    for q, seed in pr[["q", "seed"]].itertuples(index=False, name=None):
        st = C.read_json(C.r4_run_dir(int(q), int(seed)) / "status.json")
        span.append(abs((datetime.strptime(st["end_time"], fmt) - datetime.strptime(st["start_time"], fmt))
                        .total_seconds() - float(st["total_wall_sec"])))
    check("run dirs: status.json total_wall_sec == end_time - start_time (within 2 s) for all 80 runs",
          max(span) <= 2.0, f"max deviation {max(span):.3f} s")
    allw = wall[50] + wall[60]
    check("run dirs both q: median of status.json total_wall_sec == cost.csv median_wall_sec",
          float(np.median(allw)) == float(cost.loc["both", "median_wall_sec"]))
    check("run dirs both q: mean updates / policy rows == cost.csv",
          float(np.mean(upd[50] + upd[60])) == float(cost.loc["both", "mean_updates"])
          and float(np.mean(rows[50] + rows[60])) == float(cost.loc["both", "mean_policy_rows"]))


# ---------------------------------------------------------------------------------- md rendering
def f4(v: str, kind: str = "float") -> str:
    """4 significant digits for floats (integers as integers); input is a CSV value text."""
    x = float(v)
    if kind == "int":
        assert x == int(x), v
        return str(int(x))
    if x == 0.0:
        return "0"
    mant, ex = f"{x:.3e}".split("e")          # 4 significant digits, trailing zeros kept
    e, sign, dig = int(ex), "-" if mant.startswith("-") else "", mant.lstrip("-").replace(".", "")
    if e < -4 or e > 5:
        return f"{x:.3e}"
    if e >= 0:
        ip, fp = (dig[:e + 1] + "0" * max(0, e + 1 - 4)), dig[e + 1:]
        return sign + ip + ("." + fp if fp else "")
    return sign + "0." + "0" * (-e - 1) + dig


def md_esc(text: str) -> str:
    """A cell text for a Markdown table: literal pipes are escaped (``\\|``)."""
    return text.replace("|", "\\|")


def mmm(group: str, row: str, q: Any) -> str:
    """'median [min, max]' of one per-q quantity from the stored rows."""
    return (f"{f4(get(group, row, q, 'median'))} [{f4(get(group, row, q, 'minimum'))}, "
            f"{f4(get(group, row, q, 'maximum'))}]")


LEGEND = {
    "A": f"`{PR}` (80 rows, arm bwA, one row per run): the run counts and `sum` of column `gates_pass` "
         f"(`gate_G_A` and `gate_G_F` and `gate_G_N`) per q; equal to `{CNT}` columns `runs`, `pass` "
         f"and to `{HEAD}` `counts`.",
    "B": f"`{PROT}` key `confirmation.rule` (`max_failures_per_q` = limit per q, `max_failures_total` "
         f"= limit of 80), applied to the failing counts; equal to `{HEAD}` `verdict`.",
    "C": f"`{PR}`, the column named in the first column, median / minimum / maximum over the 40 rows "
         f"of the q (numpy; the median of 40 values is the mean of the two middle values; signed "
         f"errors keep their sign); the medians equal the `median_*` columns of `{CNT}` exactly "
         f"(no `median_peak_T_abs` column is stored).",
    "D": f"`{PR}` column `Gmax_full_over_dw`: median and maximum over the 40 rows of the q (equal to "
         f"`{CNT}` columns `median_Gmax_full_over_dw`, `worst_Gmax_full`, `worst_Gmax_full_run`); "
         f"the seed is column `seed` of the row of the maximum.",
    "E": f"`{COST}` columns `fired`, `fired_share`; `fired` equals the number of rows of `{FIRED}` (with "
         f"the q), `{HEAD}` `trigger.fired` (both q) and the number of run directories (untracked root "
         f"`t3-r4-lock-confirm/results/t3_r4/confirmation/q{{q}}/seed{{seed}}/bwA/`, read-only) "
         f"holding a `retry.json`.",
    "F": f"`{COST}` columns `mean_updates`, `mean_policy_rows`, `ratio_updates_to_not_fired`, "
         f"`ratio_policy_rows_to_not_fired`, `median_wall_sec` (median over the runs of "
         f"`total_wall_sec` of each run's `status.json`, the whole run including a restarted "
         f"attempt); the two reference values are the sums of `{PROT}` `settings.budgets` and "
         f"512 `settings.episodes_per_update` times that sum.",
    "G": f"`{PR}` (row q = 60, seed 80502): the gate columns `gate_G_A`, `gate_G_F`, `gate_G_N`, the "
         f"values and the development-tier columns `dev_*`; the limits are the keys of `{PROT}` "
         f"`gates` (equal to `{CRIT}` `gates`); the failing gates are listed in both `{FAIL}` (column "
         f"`failing_gates`) and `{HEAD}` (key `failures`, field `failing_gates`) and agree; the failing "
         f"quantities (column `failing_quantities`) and the stage (column `Gmax_full_stage`) are only in "
         f"`{FAIL}`, where they equal the values of this table.",
}


def render_md() -> str:
    """The readable rendering of the rows (values from the stored row texts, 4 significant digits)."""
    L: List[str] = []
    L.append("# T3. T = 3 fresh-seed confirmation (protocol T3-1.1, arm bwA)\n")
    L.append("Every value below is a row of `T3.csv` (one value per row, full precision, with its "
             "source); this file shows the same values at 4 significant digits (counts and "
             "updates / policy rows as integers). The letters in the source column are explained "
             "under Sources. Medians of 40 values are the mean of the two middle values. Signed "
             "errors keep their sign. The gate limits are inclusive (<=).\n")

    # 1. pass counts
    L.append("## Pass counts and pre-registered failure limits\n")
    L.append(f"{RUN_ROW}. A run passes iff G-A, G-F and G-N all hold.\n")
    L.append("| q | runs | runs passing | runs failing | failure limit (maximum failing runs) "
             "| within the limit (1 = yes) | source |")
    L.append("|---|---|---|---|---|---|---|")
    for k, name in ((50, "50"), (60, "60"), ("both", "both q")):
        g = lambda s: get(G_PASS, RUN_ROW, k, s)   # noqa: E731
        L.append(f"| {name} | {f4(g('runs'), 'int')} | "
                 f"{f4(g('runs passing (G-A, G-F and G-N)'), 'int')} | {f4(g('runs failing'), 'int')} | "
                 f"{f4(g('pre-registered failure limit (maximum failing runs)'), 'int')} | "
                 f"{f4(g('failures within the limit (1 = yes, 0 = no)'), 'int')} | A, B |")
    v = get(G_PASS, "confirmation verdict", "both", "verdict (1 = pass, 0 = fail)")
    L.append(f"\nConfirmation verdict: **{'pass' if v == '1' else 'fail'}** (verdict value {v}; "
             "pass iff the failures are within the limit at each q and in total) [B].\n")

    # 2. per-q statistics
    L.append("## Terminal-stage recovery and exploitability, per q\n")
    L.append("Median [minimum, maximum] over the 40 runs of the q [C].\n")
    L.append("| quantity | q = 50 | q = 60 |")
    L.append("|---|---|---|")
    for col, label in QUANT:
        L.append(f"| {label} | {mmm(G_STAT, label, 50)} | {mmm(G_STAT, label, 60)} |")
    L.append("")

    # 3. Gmax
    L.append("## Full-domain dynamic best-response gain G_max,full/ΔW, per q\n")
    row = "G_max,full/ΔW (Gmax_full_over_dw)"
    L.append("| statistic | q = 50 | q = 60 |")
    L.append("|---|---|---|")
    for s in ("median", "worst (maximum)"):
        L.append(f"| {s} | {f4(get(G_GMAX, row, 50, s))} | {f4(get(G_GMAX, row, 60, s))} |")
    L.append(f"| seed of the worst run | {f4(get(G_GMAX, row, 50, 'seed of the worst run'), 'int')} | "
             f"{f4(get(G_GMAX, row, 60, 'seed of the worst run'), 'int')} |")
    L.append("\nSource [D]. The gate limit of G-F is in the seed 80502 block below.\n")

    # 4. retry + cost
    L.append("## Stage-3 restart (retry) counts and cost per run\n")
    ref_u = f4(get(G_COST, "updates of a run without a restart", "all", "reference"), "int")
    ref_r = f4(get(G_COST, "policy rows of a run without a restart", "all", "reference"), "int")
    L.append(f"A run without a restart has {ref_u} updates and {ref_r} policy rows [F].\n")
    L.append("| quantity | q = 50 | q = 60 | both q | source |")
    L.append("|---|---|---|---|---|")

    def trio(group: str, row: str, stat: str, kind: str, src: str, label: str) -> None:
        cells = " | ".join(f4(get(group, row, k, stat), kind) for k in (50, 60, "both"))
        L.append(f"| {label} | {cells} | {src} |")

    trio(G_RETRY, "runs whose stage-3 restart fired", "fired runs", "int", "E", "runs whose restart fired")
    trio(G_RETRY, "runs whose stage-3 restart fired", "fired share of the runs", "float", "E",
         "fired share of the runs")
    trio(G_COST, "updates per run", "mean over the runs", "int", "F", "mean updates per run")
    trio(G_COST, "policy rows per run", "mean over the runs", "int", "F", "mean policy rows per run")
    trio(G_COST, "ratio of the mean updates to the 3800 of a run without a restart", "ratio", "float",
         "F", f"ratio of the mean updates to the {ref_u} of a run without a restart")
    trio(G_COST, "ratio of the mean policy rows to the 1945600 of a run without a restart", "ratio",
         "float", "F", f"ratio of the mean policy rows to the {ref_r} of a run without a restart")
    trio(G_COST, "wall time per run (seconds)", "median over the runs", "float", "F",
         "median wall time per run (seconds, column median_wall_sec)")
    L.append("")

    # 5. seed 80502
    L.append("## Seed 80502 (q = 60), the sole failure\n")
    L.append("| gate | result (1 = pass, 0 = fail) | source |")
    L.append("|---|---|---|")
    for gate in ("G-A", "G-F", "G-N"):
        r = get(G_FAIL, f"gate {gate}", 60, "gate result (1 = pass, 0 = fail)")
        L.append(f"| {gate} | {f4(r, 'int')} | G |")
    L.append("\nFailing gates: " + ", ".join(g for g in ("G-A", "G-F", "G-N")
             if get(G_FAIL, f"gate {g}", 60, "gate result (1 = pass, 0 = fail)") == "0") + ".\n")
    L.append("| gate quantity | value | limit (<=) | source |")
    L.append("|---|---|---|---|")
    for r in ROWS:
        if r["row_group"] == G_FAIL and r["statistic"] == "value" and r["row"][:4] in ("G-A:", "G-F:", "G-N:"):
            lim_v = get(G_FAIL, r["row"], 60, "limit (<= inclusive)")
            L.append(f"| {md_esc(r['row'])} | {f4(r['value'])} | {f4(lim_v)} | G |")
    L.append("")
    L.append("| item | value | source |")
    L.append("|---|---|---|")
    for rw in ("G-N input: development-tier G_max,full/ΔW", "G-N input: development-tier η_3/ΔW"):
        L.append(f"| {rw} | {f4(get(G_FAIL, rw, 60, 'value'))} | G |")
    L.append(f"| stage of G_max,full (Gmax_full_t) | "
             f"{f4(get(G_FAIL, 'stage of G_max,full', 60, 'stage (Gmax_full_t)'), 'int')} | G |")
    L.append(f"| location of G_max,full, score gap d (Gmax_full_d) | "
             f"{f4(get(G_FAIL, 'location of G_max,full', 60, 'score gap d (Gmax_full_d)'), 'float')} | G |")
    L.append("")

    # sources
    L.append("## Sources\n")
    for k in sorted(LEGEND):
        L.append(f"- **{k}** {LEGEND[k]}")
    L.append("")
    return "\n".join(L)


def main() -> None:
    """Build the rows, run every check, and (only if all checks pass) write T3.csv and T3.md."""
    d = load()
    lim = check_inputs(d)
    build_rows(d, lim)
    check_run_dirs(d)
    bad = [c for c in CHECKS if not c[1]]
    for what, ok, det in CHECKS:
        print(("OK   " if ok else "FAIL ") + what + (f"  [{det}]" if det else ""))
    if bad:
        print(f"{len(bad)} check(s) failed: nothing written")
        raise SystemExit(2)
    TABLES.mkdir(parents=True, exist_ok=True)
    cols = ["table", "row_group", "row", "column_or_q", "statistic", "value", "source"]
    with open(TABLES / "T3.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(ROWS)
    (TABLES / "T3.md").write_text(render_md(), encoding="utf-8")
    print(f"wrote {len(ROWS)} rows to tables/T3.csv and tables/T3.md; {len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
