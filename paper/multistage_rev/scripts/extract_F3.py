#!/usr/bin/env python3
"""Extraction for F3 (T = 3 effort policies, T3-1.1 confirmation, arm bwA, 40 runs per q) -> data/F3.csv.

Evaluation only (nothing is trained, no run file is written). For each of the 80 runs (q in 50, 60; seeds
80501..80540) the stored policy is loaded with ``tools/t3/t3_eval.load_bw`` (for a run whose stage-3 restart fired
this is the frozen second attempt) and evaluated once at the final tier (``evaluate_tier(cand, FINAL_CONFIG)``).

Checks (any failure is a STOP: nothing is written, exit code 2):
  * results/t3_r4/confirmation/eval/per_run.csv: gates_pass 40/40 at q = 50, 39/40 at q = 60, 79/80, the only
    failure being (q = 60, seed 80502);
  * recomputed Gmax_full_over_dw, eta_T_over_dw, EXP_root_over_dw, Gmax_full_t, Gmax_full_d are bit-equal (==) to
    the stored row of every run;
  * e_1(0) == e1_at_0 and the stage-3 mean effort at d = 0 == eT_at_0 (bit-equal) for every run;
  * candidate_pmf(ev.res, q) is np.array_equal to ev.pmf_cand, and every stage's pmf sums to 1 within 1e-9.

Output rows (long format; columns panel, q, series, seed, d, value):
  stage1_effort       e_1(0) per run (series run / failed_run, d blank) and its median over the 40 runs;
  stage2_effort       mean effort on D_2 at step 1 per run, and the median over the 40 runs at every d;
  stage3_effort       the same on D_3 at step 1;
  closed_form_equilibrium  g2_two_stage on the stage-3 grid with the spec parameters;
  stage{2,3}_visitation    candidate_pmf of the final-tier result (mass per verifier node, state step 2): median,
                           10 % and 90 % percentile (np.percentile, linear) over the 40 runs at every node.
The failed run (q = 60, seed 80502 ONLY) is series ``failed_run``; it is inside every median / percentile.

Usage:  .venv/bin/python paper/multistage_rev/scripts/extract_F3.py
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import _common as C

import numpy as np

SEEDS = tuple(range(80501, 80541))
FAILED = (60, 80502)
WORKERS = 8
STAT_COLS = ("Gmax_full_over_dw", "eta_T_over_dw", "EXP_root_over_dw", "Gmax_full_t", "Gmax_full_d")
OUT_CSV = C.OUT / "data" / "F3.csv"


def eval_run(job: Tuple[int, int]) -> Dict[str, Any]:
    """Load and evaluate one run (final tier); return the scalars, the mean-effort arrays and the pmfs."""
    C.single_thread()
    import t3_eval as te
    from replay_dev_rule import candidate_fns
    from utils.dp_br_verifier import FINAL_CONFIG
    from utils.theory_multistage import g2_two_stage
    from utils.v2_metrics import candidate_pmf, symmetric_grid, zero_index

    q, seed = job
    run_dir = C.r4_run_dir(q, seed)
    cfg = C.read_json(run_dir / "run_config.json")
    cand = te.load_bw(run_dir, cfg)
    ev = te.evaluate_tier(cand, FINAL_CONFIG)
    spec = cand.spec
    mean_fn, _ = candidate_fns(spec, cand.nets)
    d3 = symmetric_grid(spec.domain_half(3), 1.0)
    d2 = symmetric_grid(spec.domain_half(2), 1.0)
    e3 = np.asarray(mean_fn(3, d3), dtype=float)
    e2 = np.asarray(mean_fn(2, d2), dtype=float)
    e1 = float(np.asarray(mean_fn(1, np.zeros(1)), dtype=float)[0])
    g3 = g2_two_stage(d3, float(spec.q), spec.w_h, spec.w_l, spec.k, spec.e_max)
    cp = candidate_pmf(ev.res, spec.q)
    retry_p = run_dir / "retry.json"
    retry = C.read_json(retry_p) if retry_p.exists() else {}
    manifest = C.read_json(run_dir / "manifest.json")
    return {
        "q": q, "seed": seed, "spec": (spec.w_h, spec.w_l, spec.k, spec.q, spec.T, spec.e_min, spec.e_max),
        "scal": {k: ev.scalars[k] for k in STAT_COLS},
        "e1": e1, "e3_at_0": float(e3[zero_index(d3)]), "g3_at_0": float(g3[zero_index(d3)]),
        "d2": d2, "e2": e2, "d3": d3, "e3": e3, "g3": g3,
        "vd2": np.asarray(ev.res.stages[2].d_grid, dtype=float), "vd3": np.asarray(ev.res.stages[3].d_grid, dtype=float),
        "pmf1": np.asarray(ev.pmf_cand[1], dtype=float), "pmf2": np.asarray(ev.pmf_cand[2], dtype=float),
        "pmf3": np.asarray(ev.pmf_cand[3], dtype=float),
        "pmf_equal": bool(all(np.array_equal(cp[t], ev.pmf_cand[t]) for t in (1, 2, 3))),
        "retry_fired": bool(retry.get("fired", False)), "frozen_attempt": retry.get("frozen_attempt"),
        "commit": manifest.get("commit"), "dirty": manifest.get("git", {}).get("dirty"),
    }


def main() -> int:
    """Run the checks, evaluate the 80 runs and write data/F3.csv."""
    stops: List[str] = []
    log: List[str] = []
    pr = C.load_per_run(C.PER_RUN_R4)

    # ---- pass counts from the stored per-run record
    cnt = {q: (int(pr[pr.q == q].gates_pass.astype(bool).sum()), int((pr.q == q).sum())) for q in C.QS}
    log.append(f"per_run.csv gates_pass: q50 {cnt[50][0]}/{cnt[50][1]}, q60 {cnt[60][0]}/{cnt[60][1]}, "
               f"overall {cnt[50][0] + cnt[60][0]}/{cnt[50][1] + cnt[60][1]}")
    if cnt[50] != (40, 40) or cnt[60] != (39, 40):
        stops.append(f"pass counts {cnt} differ from q50 40/40, q60 39/40")
    fails = [(int(r.q), int(r.seed)) for r in pr.itertuples() if not bool(r.gates_pass)]
    log.append(f"failed runs in per_run.csv: {fails}")
    if fails != [FAILED]:
        stops.append(f"failures {fails} differ from [{FAILED}]")
    pairs = sorted((int(a), int(b)) for a, b in zip(pr.q, pr.seed))
    if pairs != sorted((q, s) for q in C.QS for s in SEEDS) or set(pr.arm) != {"bwA"}:
        stops.append("per_run.csv does not hold exactly arm bwA, q in (50, 60), seeds 80501..80540")
    if stops:
        print("STOP:\n  " + "\n  ".join(stops))
        return 2

    # ---- evaluate the 80 runs (<= 8 single-threaded worker processes)
    jobs = [(q, s) for q in C.QS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn")) as ex:
        res_list = list(ex.map(eval_run, jobs))
    R: Dict[Tuple[int, int], Dict[str, Any]] = {(r["q"], r["seed"]): r for r in res_list}

    # ---- per-run bit-equality checks against the stored record
    n_scal = n_e1 = n_e3 = n_g3 = n_pmf = n_sum = 0
    for (q, s), r in sorted(R.items()):
        row = pr[(pr.q == q) & (pr.seed == s)].iloc[0]
        spec = r["spec"]
        if spec != (C.W_H, C.W_L, C.K, float(q), 3, 0.0, 100.0):
            stops.append(f"q{q} seed{s}: spec {spec} differs from k=1/3500, w_h=6, w_l=2, T=3, [0, 100]")
        bad = [k for k in STAT_COLS if not (r["scal"][k] == row[k])]
        if bad:
            stops.append(f"q{q} seed{s}: recomputed {[(k, r['scal'][k], row[k]) for k in bad]} not bit-equal to per_run.csv")
        else:
            n_scal += 1
        if r["e1"] == row["e1_at_0"]:
            n_e1 += 1
        else:
            stops.append(f"q{q} seed{s}: e_1(0)={r['e1']!r} != e1_at_0={row['e1_at_0']!r}")
        if r["e3_at_0"] == row["eT_at_0"]:
            n_e3 += 1
        else:
            stops.append(f"q{q} seed{s}: stage-3 effort at d=0 {r['e3_at_0']!r} != eT_at_0 {row['eT_at_0']!r}")
        if r["g3_at_0"] == row["eT_star_at_0"]:
            n_g3 += 1
        else:
            stops.append(f"q{q} seed{s}: g2_two_stage(0)={r['g3_at_0']!r} != eT_star_at_0={row['eT_star_at_0']!r}")
        if r["pmf_equal"]:
            n_pmf += 1
        else:
            stops.append(f"q{q} seed{s}: candidate_pmf != ev.pmf_cand")
        sums = [float(r[k].sum()) for k in ("pmf1", "pmf2", "pmf3")]
        if all(abs(x - 1.0) <= 1e-9 for x in sums):
            n_sum += 1
        else:
            stops.append(f"q{q} seed{s}: pmf sums {sums}")
    log.append(f"{n_scal}/80 runs: Gmax_full_over_dw, eta_T_over_dw, EXP_root_over_dw, Gmax_full_t, Gmax_full_d bit-equal")
    log.append(f"{n_e1}/80 runs: e_1(0) == e1_at_0 (bit-equal)")
    log.append(f"{n_e3}/80 runs: stage-3 mean effort at d=0 on the step-1 grid == eT_at_0 (bit-equal)")
    log.append(f"{n_g3}/80 runs: g2_two_stage(0) == eT_star_at_0 (bit-equal)")
    log.append(f"{n_pmf}/80 runs: candidate_pmf == ev.pmf_cand (np.array_equal), stages 1-3")
    log.append(f"{n_sum}/80 runs: every stage's pmf sums to 1 within 1e-9 "
               f"(max |sum-1| over stages 2, 3: {max(abs(float(r[k].sum()) - 1.0) for r in R.values() for k in ('pmf2', 'pmf3')):.3e})")
    gm = {q: [R[(q, s)]['scal']['Gmax_full_over_dw'] for s in SEEDS] for q in C.QS}
    for q in C.QS:
        log.append(f"q{q}: Gmax_full_over_dw min/median/max = {min(gm[q])!r} / {float(np.median(gm[q]))!r} / {max(gm[q])!r}")
    fired = {q: sorted(s for s in SEEDS if R[(q, s)]['retry_fired']) for q in C.QS}
    log.append(f"stage-3 restart fired (retry.json): q50 {len(fired[50])} runs {fired[50]}, q60 {len(fired[60])} runs {fired[60]}")
    if len(fired[50]) != 5 or fired[60]:
        stops.append(f"restart fired in {len(fired[50])} runs at q50 / {len(fired[60])} at q60, expected 5 / 0")
    commits = sorted({(str(r["commit"]), str(r["dirty"])) for r in R.values()})
    log.append(f"run manifests: (commit, dirty) = {commits}")
    for (q, s), r in sorted(R.items()):  # all runs share the grids
        if not (np.array_equal(r["d2"], R[(q, SEEDS[0])]["d2"]) and np.array_equal(r["d3"], R[(q, SEEDS[0])]["d3"])
                and np.array_equal(r["vd2"], R[(q, SEEDS[0])]["vd2"]) and np.array_equal(r["vd3"], R[(q, SEEDS[0])]["vd3"])
                and np.array_equal(r["g3"], R[(q, SEEDS[0])]["g3"])):
            stops.append(f"q{q} seed{s}: grids differ between runs")
    if stops:
        print("\n".join(log))
        print("STOP:\n  " + "\n  ".join(stops))
        return 2

    # ---- long-format rows
    rows: List[List[str]] = []

    def add(panel: str, q: int, series: str, seed: Any, d: Any, value: float) -> None:
        rows.append([panel, str(q), series, "" if seed is None else str(seed), "" if d is None else C.fmt(d), C.fmt(value)])

    for q in C.QS:
        run_series = lambda s: "failed_run" if (q, s) == FAILED else "run"   # noqa: E731
        g = R[(q, SEEDS[0])]
        # stage 1
        for s in SEEDS:
            add("stage1_effort", q, run_series(s), s, None, R[(q, s)]["e1"])
        add("stage1_effort", q, "median", None, None, float(np.median([R[(q, s)]["e1"] for s in SEEDS])))
        # stages 2 and 3 mean effort
        for panel, dk, ek in (("stage2_effort", "d2", "e2"), ("stage3_effort", "d3", "e3")):
            for s in SEEDS:
                for dv, ev_ in zip(R[(q, s)][dk], R[(q, s)][ek]):
                    add(panel, q, run_series(s), s, dv, ev_)
            med = np.median(np.stack([R[(q, s)][ek] for s in SEEDS]), axis=0)
            for dv, m in zip(g[dk], med):
                add(panel, q, "median", None, dv, m)
        # closed-form equilibrium at stage 3
        for dv, gv in zip(g["d3"], g["g3"]):
            add("closed_form_equilibrium", q, "closed_form_equilibrium", None, dv, gv)
        # state-visitation distribution
        for panel, dk, pk in (("stage2_visitation", "vd2", "pmf2"), ("stage3_visitation", "vd3", "pmf3")):
            st = np.stack([R[(q, s)][pk] for s in SEEDS])
            med = np.median(st, axis=0)
            p10, p90 = np.percentile(st, [10, 90], axis=0)
            for name, arr in (("median", med), ("p10", p10), ("p90", p90)):
                for dv, v in zip(g[dk], arr):
                    add(panel, q, name, None, dv, v)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["panel", "q", "series", "seed", "d", "value"])
        w.writerows(rows)
    log.append(f"wrote {OUT_CSV} ({len(rows)} rows)")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
