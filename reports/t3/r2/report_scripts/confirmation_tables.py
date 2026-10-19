#!/usr/bin/env python3
"""Tables, a figure and the inline numbers of ``reports/t3/r2/04_confirmation.md``, generated from ``results/t3_r2/confirmation/``.

Reads the launch record, ``launch_checks.json``, ``infra_events.json``, ``eval/per_run.csv``, ``analysis/*`` (``t3c_analysis.py``), the blind
recomputation, the run directories (``status.json``, ``ms_run_summary.json``) and, for the predictions, the exports of the 40 runs (the
last-16-export eta_3 of A8 and the A7 formation, with the functions of ``tools/t3/t3r2_audit.py``). Writes the Markdown fragments to
``reports/t3/r2/tables/c*.md``, the figure to ``figures/`` and ``tables/confirmation_numbers.json``. Nothing here is a statistic of its own: it
selects, orders, formats and computes maxima, medians and counts.

Usage:
    python reports/t3/r2/report_scripts/confirmation_tables.py [--workers 20]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import REP, RES, ROOT, md, num, write  # noqa: E402

import t3r2_audit as au  # noqa: E402

CONF = RES / "confirmation"
ANA = CONF / "analysis"
FIG = REP / "figures"
N: Dict[str, Any] = {}
QS = (50, 60)
SEEDS = list(range(60501, 60521))
UNTRACKED_T3R1 = Path(os.environ.get("T3R1_UNTRACKED_ROOT", "/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-vs-jt-51e472/results/t3_r1"))


def sci(x: float) -> str:
    """Two-digit scientific format."""
    return f"{float(x):.2e}"


def settings() -> None:
    import t3c_launch as cl

    launch = json.loads(sorted(CONF.glob("launch_2*.json"))[-1].read_text())
    rec = cl.t3_record()
    pre = (CONF / "launch_preconditions.txt").read_text().split("--- ")
    stat = next(x for x in pre if x.startswith("git diff --stat")).split("\n", 1)[1].strip()
    rows = [("HEAD at the launch (the commit that holds the LOCK record)", f"`{launch['head']}`"),
            ("lock commit (in the LOCK record)", f"`{rec['lock_commit']}`"),
            ("lock JSON SHA-256 (LOCK record)", f"`{rec['protocol_sha256']}`"), ("lock MD SHA-256 (LOCK record)", f"`{rec['protocol_md_sha256']}`"),
            ("code commit recorded by the launcher", f"`{launch['code_commit']}`"),
            ("`git diff --stat be2b7e9f HEAD -- agents envs utils run tools/ms tools/v2` at the launch", "empty" if not stat else f"`{stat}`"),
            ("`git status --porcelain -- . ':(exclude)results'` at the launch (the clean-tree definition)", "empty"),
            ("`git status --porcelain` at the launch (recorded by the launcher in `launch_*.json`)",
             f"{len(launch['status_porcelain'])} entries, " + ("all of them untracked files under `results/` (the audit's re-run weights and arrays, `a7_emat.npz`, and `results/t3_r2/confirmation/` itself)"
                                                               if all(x.startswith("?? results/") for x in launch["status_porcelain"]) else "NOT all under results/")),
            ("workers (single-threaded)", launch["workers"]), ("`nproc`", launch["nproc"]),
            ("load average at the start", ", ".join(f"{x:.2f}" for x in launch["loadavg_at_start"])),
            ("free disk at the start", f"{launch['disk_free_bytes'] / 1e9:.0f} GB of {launch['disk_total_bytes'] / 1e9:.0f} GB"),
            ("parameters file SHA-256", f"`{launch['params_sha256']}`"), ("planned / finished", f"{launch['n_planned']} / {len(launch['runs'])}"),
            ("final state of the launch", launch["state"])]
    write("c01_settings", md(["item", "value"], rows))
    N["launch_head"], N["launch_state"], N["lock_commit"] = launch["head"], launch["state"], rec["lock_commit"]
    N["nproc"], N["load_start"] = launch["nproc"], launch["loadavg_at_start"][0]
    N["load_end"] = launch.get("loadavg_at_end", [float("nan")])[0]
    wall = [r["wall_sec"] for r in launch["runs"] if "wall_sec" in r]
    N["launch_wall_max_sec"], N["launch_wall_median_sec"], N["launch_wall_min_sec"] = max(wall), float(np.median(wall)), min(wall)
    N["n_runs_exit0"] = sum(1 for r in launch["runs"] if r.get("returncode") == 0)


def checks() -> None:
    c = json.loads((CONF / "launch_checks.json").read_text())
    names = {"exit_ok": "exit code 0 / status done", "files_complete": "complete files", "manifest_commit_ok": "manifest at the launch commit",
             "clean_tree": "clean tree (manifest flag)", "global_rng_ok": "process-global RNG assertions ok", "config_equals_plan": "config equals the plan (relative hash)",
             "config_production_ok": "config equals the lock (production values)", "budget_ok": "budgets (updates, policy rows, minibatch steps)",
             "lr_ok": "learning rate at every update", "refresh_ok": "snapshot refreshes (phase entries, multiples of 20)", "exports_ok": "152 exports u00025...u03800",
             "checks_ok": "64 / 64 / 24 checks per stage", "init_equals_plan": "C-INIT: recorded init digest = planned", "replay_ok": "C-REPLAY: freeze arrays reproduced bit for bit"}
    rows = [(names.get(k, k), v["pass"], v["fail"], v["not_evaluated"]) for k, v in c["verdicts"].items()]
    rows.append(("`infra_events.json` present", "yes" if c["infra_events_present"] else "NO", "", ""))
    write("c02_checks", md(["check", "runs passing", "failing", "not evaluated"], rows))
    N["checks_all_pass"] = bool(all(v["fail"] == 0 and v["not_evaluated"] == 0 for v in c["verdicts"].values()) and c["infra_events_present"])
    N["checks_n"] = int(c["n_runs"])
    ev = c["infra_events"]
    N["infra_events"], N["infra_stopped"] = len(ev["events"]), len(ev["stopped_without_verdict"])


def per_seed() -> None:
    t = pd.read_csv(ANA / "per_run_table.csv", float_precision="round_trip")
    for q in QS:
        rows = []
        for r in t[t.q == q].itertuples():
            rows.append((int(r.seed), "pass" if r.passed else f"**FAIL** ({r.fail_reason})", num(r.eta_T_over_dw, 5), num(r.rmse_pos_T_over_eT0, 4), num(r.tail_mean_T_over_eT0, 4),
                         num(r.Gmax_full_over_dw, 5), f"{int(r.Gmax_full_t)}, {r.Gmax_full_d:+.0f}", num(r.gn_eta_diff, 5), num(r.gn_gmax_diff, 5), num(r.EXP_root_over_dw, 5)))
        write(f"c03_per_seed_q{q}", md(["seed", "run", "eta_3/dW", "RMSE_pos/e_3(0)*", "tail mean/e_3(0)*", "Gmax_full/dW", "its stage, gap d",
                                      "abs(eta_3 dev - final)", "abs(Gmax dev - final)", "EXP_root/dW"], rows))


def verdict() -> None:
    v = json.loads((ANA / "verdict.json").read_text())
    c = v["counts"]
    rows = []
    for q in QS:
        rows.append((q, f"{c[f'pass_q{q}']} of {c[f'n_counted_q{q}']}", f"[{c[f'ci_lo_q{q}']:.3f}, {c[f'ci_hi_q{q}']:.3f}]", f">= 17", "yes" if v["verdict"][f"q{q}_condition"] else "NO"))
    rows.append(("both", f"{c['pass_total']} of {c['n_counted_total']}", f"[{c['ci_lo_total']:.3f}, {c['ci_hi_total']:.3f}]", ">= 36", "yes" if v["verdict"]["total_condition"] else "NO"))
    write("c04_verdict", md(["q", "runs passing", "exact Clopper-Pearson 95 % interval", "rule", "met"], rows))
    bl = json.loads((ANA / "blind" / "blind_verdict.json").read_text())
    N["verdict"], N["blind_verdict"] = v["verdict"]["verdict"], bl["verdict"]
    N["pass_q50"], N["pass_q60"], N["pass_total"] = c["pass_q50"], c["pass_q60"], c["pass_total"]
    from scipy.stats import binom
    pm = binom.pmf(np.arange(21), 20, 0.875)
    N["oc_pass_875"] = float(sum(pm[a] * pm[b] for a in range(17, 21) for b in range(17, 21) if a + b >= 36))
    N["oc_fail_875"] = 1.0 - N["oc_pass_875"]
    N["blind_agrees"] = bool(bl["verdict"] == v["verdict"]["verdict"] and all(int(bl[k]) == int(c[k]) for k in ("pass_q50", "pass_q60", "pass_total")))
    N["reason"] = v["verdict"]["reason"]
    N["ci_total"] = f"[{c['ci_lo_total']:.3f}, {c['ci_hi_total']:.3f}]"


def failures() -> None:
    f = pd.read_csv(ANA / "failures.csv") if (ANA / "failures.csv").is_file() and (ANA / "failures.csv").stat().st_size > 2 else pd.DataFrame()
    if f.empty:
        write("c05_failures", "no failing run\n")
        N["n_failing"] = 0
        return
    cols = [c for c in f.columns]
    write("c05_failures", md(cols, [[("" if (isinstance(x, float) and np.isnan(x)) else (num(x, 5) if isinstance(x, float) else x)) for x in r] for r in f.itertuples(index=False)]))
    N["n_failing"] = int(f[["q", "seed"]].drop_duplicates().shape[0])


def failure_history() -> None:
    """The stage-3 phase of every failing run: the value of the failing quantity, the shape of the policy, the development-tier Delta of the
    phase at fixed local updates, and the position of the evaluated iterate in its own last-16-export range (``a8`` columns need ``predictions``)."""
    t = pd.read_csv(ANA / "per_run_table.csv", float_precision="round_trip")
    ev = pd.read_csv(CONF / "eval" / "per_run.csv", float_precision="round_trip").set_index(["q", "seed"])
    locs = (25, 100, 200, 400, 800, 1200, 1400, 1600)
    rows, rows2 = [], []
    for r in t[~t.passed].itertuples():
        e = ev.loc[(r.q, r.seed)]
        h = pd.read_csv(CONF / f"q{r.q}" / f"seed{r.seed}" / "bw" / "ms_checks_stage3.csv").set_index("local")
        rows.append((int(r.q), int(r.seed), r.fail_reason.replace("gate_", "").replace("_", "-"), num(r.eta_T_over_dw, 5), num(r.rmse_pos_T_over_eT0, 4),
                     num(r.tail_mean_T_over_eT0, 4), num(r.Gmax_full_over_dw, 5), f"{num(e.eT_at_0, 2)} (e*(0) = {num(e.eT_star_at_0, 2)})",
                     num(e.peak_T_signed, 3), num(e.tail_max_T, 2), num(e.G_max_t2_over_dw, 5)))
        rows2.append((int(r.q), int(r.seed), *[num(float(h.loc[k, "Delta"]), 5) for k in locs], f"{float(h['Delta'].min()):.5f} (local {int(h['Delta'].idxmin())})"))
    write("c12_failure_values", md(["q", "seed", "failing gates", "eta_3/dW", "RMSE_pos/e_3(0)*", "tail mean/e_3(0)*", "Gmax_full/dW", "learned e_3(0)", "(e_3(0) - e*(0))/e*(0)",
                                    "largest effort at |d| >= 2q", "G_max,2/dW (stage 2)"], rows))
    write("c13_failure_history", md(["q", "seed", *[f"Delta at local {k}" for k in locs], "smallest Delta of the 64 checks"], rows2))


def stretches() -> None:
    s = pd.read_csv(ANA / "stretches.csv")
    rows = [(int(r.q), int(r.seed), "pass" if r.passed else "**fail**", int(r.nodes), int(r.nodes_in_band), r.stretches if isinstance(r.stretches, str) else "-") for r in s.itertuples()]
    write("c06_stretches", md(["q", "seed", "run", "in-band stage-2 nodes with learned effort < 1", "in-band nodes", "stretches (d)"], rows))
    N["n_with_stretch"] = int((s.nodes > 0).sum())
    N["n_with_stretch_q50"], N["n_with_stretch_q60"] = int(((s.nodes > 0) & (s.q == 50)).sum()), int(((s.nodes > 0) & (s.q == 60)).sum())
    N["stretch_fail_overlap"] = int(((s.nodes > 0) & (~s.passed)).sum())
    N["nostretch_fail"] = int(((s.nodes == 0) & (~s.passed)).sum())


def comparison() -> None:
    g = pd.read_csv(ANA / "gate_values_vs_t3r1.csv")
    names = {"eta_T_over_dw": "eta_3/dW", "rmse_pos_T_over_eT0": "RMSE_pos/e_3(0)*", "tail_mean_T_over_eT0": "tail mean/e_3(0)*", "Gmax_full_over_dw": "Gmax_full/dW",
             "gn_eta_diff": "abs(eta_3 dev - final)", "gn_gmax_diff": "abs(Gmax dev - final)"}
    rows = []
    for q in QS:
        for k, nm in names.items():
            a = g[(g.set == "t3r2_confirmation") & (g.q == q) & (g.quantity == k)].iloc[0]
            b = g[(g.set == "t3r1_bw") & (g.q == q) & (g.quantity == k)].iloc[0]
            rows.append((q, nm, num(a.limit, 4), num(a["median"], 5), num(a["max"], 5), int(a.n_above_limit), num(b["median"], 5), num(b["max"], 5), int(b.n_above_limit)))
    write("c07_gate_values", md(["q", "quantity", "limit", "confirmation median", "max", "runs above the limit", "T3-R1 `bw` median", "max", "runs above the limit"], rows))
    p = pd.read_csv(ANA / "pooled_descriptive.csv")
    write("c08_pooled", md(["set", "q", "runs", "passes", "rate", "exact 95 % interval"],
                           [(r.set, r.q, int(r.n), int(r.passes), f"{r.rate:.3f}", f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]") for r in p.itertuples()]))
    pool = p[(p.set == "pooled_80_descriptive") & (p.q == "all")].iloc[0]
    N["pooled_passes"], N["pooled_n"], N["pooled_ci"] = int(pool.passes), int(pool.n), f"[{pool.ci_lo:.3f}, {pool.ci_hi:.3f}]"
    t = pd.read_csv(ANA / "per_run_table.csv", float_precision="round_trip")
    old = pd.read_csv(ROOT / "results" / "t3_r1" / "eval" / "per_run.csv", float_precision="round_trip")
    old = old[old.arm == "bw"]
    means_rows = [(q, num(t[t.q == q].Gmax_full_over_dw.mean(), 5), num(old[old.q == q].Gmax_full_over_dw.mean(), 5), num(t[t.q == q].EXP_root_over_dw.mean(), 5),
             num(old[old.q == q].EXP_root_over_dw.mean(), 5)) for q in QS]
    ev = pd.read_csv(CONF / "eval" / "per_run.csv", float_precision="round_trip")
    rows = []
    for label, d in (("confirmation", ev), ("T3-R1 `bw`", old)):
        for q in QS:
            x = d[d.q == q]
            rows.append((label, q, len(x), int((x.Gmax_full_t == 3).sum()), int((x.Gmax_full_t == 2).sum()), int((x.Gmax_full_t == 1).sum()),
                         num(x.G_max_t2_over_dw.median(), 5), num(x.G_max_t2_over_dw.max(), 5), num(x.G_max_t3_over_dw.median(), 5), num(x.G_max_t3_over_dw.max(), 5)))
    write("c15_stage_of_max", md(["set", "q", "runs", "Gmax_full at stage 3", "at stage 2", "at stage 1", "median G_max,2/dW", "max", "median G_max,3/dW", "max"], rows))
    r518 = ev[(ev.q == 60) & (ev.seed == 60518)].iloc[0]
    N["d518_delta2"], N["d518_delta2_d"], N["d518_g2"] = float(r518.Delta_max_t2_over_dw), float(r518.Delta_argmax_d_t2), float(r518.G_max_t2_over_dw)
    N["conf_stage2_max_runs"] = int((ev.Gmax_full_t == 2).sum())
    N["t3r1_stage2_max_runs"] = int((old.Gmax_full_t == 2).sum())
    N["conf_gmax2_max_excl_60518"] = float(ev[~((ev.q == 60) & (ev.seed == 60518))].G_max_t2_over_dw.max())
    write("c09_means", md(["q", "mean Gmax_full/dW, confirmation", "T3-R1 `bw`", "mean EXP_root/dW, confirmation", "T3-R1 `bw`"], means_rows))


def cost() -> None:
    rows, walls = [], []
    for q in QS:
        for s in SEEDS:
            d = CONF / f"q{q}" / f"seed{s}" / "bw"
            st = json.loads((d / "status.json").read_text())
            summ = json.loads((d / "ms_run_summary.json").read_text()) if (d / "ms_run_summary.json").is_file() else {}
            walls.append((q, s, float(st.get("total_wall_sec", np.nan)), summ.get("final_global_update")))
    w = pd.DataFrame(walls, columns=["q", "seed", "wall", "updates"])
    t1 = pd.read_csv(ROOT / "results" / "t3_r1" / "analysis" / "cost.csv")
    t1 = t1[t1.arm == "bw"]
    rows = [("confirmation (40 workers)", len(w), f"{w.wall.min():.0f}", f"{w.wall.median():.0f}", f"{w.wall.max():.0f}", f"{w.wall.sum() / 3600:.2f}"),
            ("T3-R1 `bw` (wave of 160 runs at 40 workers)", len(t1), f"{t1.wall_sec.min():.0f}", f"{t1.wall_sec.median():.0f}", f"{t1.wall_sec.max():.0f}", f"{t1.wall_sec.sum() / 3600:.2f}")]
    write("c10_cost", md(["set", "runs", "wall s, min", "median", "max", "sum of the runs' wall time, hours"], rows))
    N["wall_median"], N["wall_max"], N["wall_min"] = float(w.wall.median()), float(w.wall.max()), float(w.wall.min())
    N["updates_all_3800"] = "yes, 40 of 40 (`final_global_update` of `ms_run_summary.json`)" if bool((w.updates == 3800).all()) else "NO"


def predictions(workers: int) -> None:
    """AP2 and AP3 applied to the confirmation runs: the last-16-export eta_3 of the 40 runs and the A7 formation of their stretches."""
    jobs = [{"arm": "bw", "q": q, "seed": s, "T": 3, "run": f"t3r2_q{q}_s{s}_bw", "run_dir": str(CONF / f"q{q}" / f"seed{s}" / "bw"), "kind": "bw"}
            for q in QS for s in SEEDS]
    items = [(j, u) for j in jobs for u in au.LAST16]
    res = pd.DataFrame(au.parallel(au.a8_one, items, workers))
    res.to_csv(ANA / "a8_last16_confirmation.csv", index=False)
    s = res.groupby(["q", "seed"]).agg(mn=("eta_3_over_dw", "min"), med=("eta_3_over_dw", "median"), mx=("eta_3_over_dw", "max"))
    last = res[res["update"] == 1600].set_index(["q", "seed"]).eta_3_over_dw
    t = pd.read_csv(ANA / "per_run_table.csv", float_precision="round_trip").set_index(["q", "seed"])
    rows = []
    for q in QS:
        sq = s.loc[q]
        rows.append((q, 20, int((sq.mx > 0.005).sum()), int((sq.mn > 0.005).sum()), int((last.loc[q] > 0.005).sum()), f"{sq.med.median():.5f}"))
    write("c11_eta3_variability", md(["q", "runs", "runs with eta_3/dW > 0.005 at one or more of the last 16 exports", "at all 16 exports", "at the evaluated iterate (u01600)",
                                      "median over runs of the per-run median"], rows))
    for q in QS:
        sq = s.loc[q]
        N[f"eta_any_q{q}"], N[f"eta_all_q{q}"], N[f"eta_final_q{q}"] = int((sq.mx > 0.005).sum()), int((sq.mn > 0.005).sum()), int((last.loc[q] > 0.005).sum())
    N["a8_consistent"] = float((last - t.eta_T_over_dw).abs().max())
    rows = []
    for r in t[~t.passed].reset_index().itertuples():
        x = s.loc[(r.q, r.seed)]
        rows.append((int(r.q), int(r.seed), num(x.mn, 5), num(x.med, 5), num(x.mx, 5), num(float(last.loc[(r.q, r.seed)]), 5),
                     int((res[(res.q == r.q) & (res.seed == r.seed)].eta_3_over_dw > 0.005).sum())))
    write("c14_failure_variability", md(["q", "seed", "min eta_3/dW over the last 16 exports", "median", "max", "at the evaluated iterate (u01600)", "exports above 0.005"], rows))
    # a figure of eta_3 against the export, the failing runs highlighted
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, q in zip(axes, QS):
        for sd in SEEDS:
            tt = res[(res.q == q) & (res.seed == sd)].sort_values("update")
            hot = not bool(t.loc[(q, sd)].passed)
            ax.plot(tt["update"], tt.eta_3_over_dw, color="C3" if hot else "0.7", lw=1.6 if hot else 0.8, zorder=3 if hot else 1)
        ax.axhline(0.005, color="k", ls="--", lw=0.8)
        ax.set_title(f"confirmation, q = {q}")
        ax.set_xlabel("stage-3 export (update)")
    axes[0].set_ylabel("eta_3 / dW (final tier)")
    axes[0].set_ylim(0, 0.03)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "f03_confirmation_eta3.png", dpi=110)
    plt.close(fig)


def main() -> int:
    """Write every fragment and the numbers file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    for fn in (settings, checks, per_seed, verdict, failures, failure_history, stretches, comparison, cost):
        fn()
    predictions(a.workers)
    (REP / "tables" / "confirmation_numbers.json").write_text(json.dumps(N, indent=1, default=str))
    print(json.dumps(N, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
