"""F4 extraction: T = 3 best-response diagnostics of three T3-1.1 confirmation runs (final verifier tier).

Runs (arm ``bwA`` of ``results/t3_r4/confirmation/eval/per_run.csv``; rule fixed by the round lead):
  * per q, the PASSING run (``gates_pass`` True) at the median rank of the passing runs sorted by
    ``(Gmax_full_over_dw, seed)``: q = 60 has 39 passing runs (unique median run); q = 50 has 40 (the two
    middle-ranked runs are equally close to the median value; the one with the LOWER SEED is chosen);
  * the failure q = 60, seed 80502.

Each run is re-evaluated (no training, no random number) with ``tools/t3/t3_eval.load_bw`` and
``t3_eval.evaluate_tier(cand, FINAL_CONFIG)``. STOP rule: ``Gmax_full_over_dw``, ``eta_T_over_dw``,
``EXP_root_over_dw``, ``Gmax_full_t``, ``Gmax_full_d`` must equal the stored per-run values bit for bit; the
maximum over stages of ``G_t / dW`` from the arrays must equal ``Gmax_full_over_dw`` bit for bit and sit at
``(Gmax_full_t, Gmax_full_d)``; the pass counts must be 40/40, 39/40, 79/80 (and 74/80 on the first attempt).
Any failed check stops the script before ``data/F4.csv`` is written.

Output: ``paper/multistage_rev/data/F4.csv`` (long format; columns ``run_label, q, seed, stage, d, quantity,
value``; quantities ``e_hat``, ``a_br`` (efforts), ``G_over_dw``, ``delta_over_dw`` (gains divided by
dW = spec.dw), ``reach`` (0/1) and ``gmax_full_marker`` (written ONLY for the failed run q = 60, seed
80502, since the spec asks for the G_max,full marker of that seed alone: stage = Gmax_full_t,
d = Gmax_full_d, value = Gmax_full_over_dw)). Floats are written with ``repr`` (full precision).

Usage:  /home/fjiang4/tournament_experiment/.venv/bin/python paper/multistage_rev/scripts/extract_F4.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import _common as C  # sets the thread env vars before numpy / torch are imported

C.single_thread()

import numpy as np  # noqa: E402
import t3_eval  # noqa: E402
from utils.dp_br_verifier import FINAL_CONFIG  # noqa: E402

OUT_CSV = C.OUT / "data" / "F4.csv"
FAIL_Q, FAIL_SEED = 60, 80502
EXPECTED_PASS = {50: 40, 60: 39}
EXPECTED_TOTAL_PASS, EXPECTED_FIRST_ATTEMPT = 79, 74
STORED_KEYS = ("Gmax_full_over_dw", "eta_T_over_dw", "EXP_root_over_dw", "Gmax_full_t", "Gmax_full_d")

CHECKS: List[Tuple[str, bool, str]] = []


def check(what: str, ok: bool, detail: str = "") -> bool:
    """Record and print one check."""
    CHECKS.append((what, bool(ok), detail))
    print(f"[{'ok' if ok else 'FAIL'}] {what}" + (f"  {detail}" if detail else ""), flush=True)
    return bool(ok)


def stop(msg: str) -> None:
    """STOP-and-report: print the failed checks and leave without writing any output."""
    print(f"\nSTOP: {msg}", file=sys.stderr)
    for what, ok, detail in CHECKS:
        if not ok:
            print(f"  FAILED: {what}  {detail}", file=sys.stderr)
    sys.exit(2)


def select_runs(per) -> List[Dict[str, Any]]:
    """The three runs of F4 with the selection record (printed)."""
    runs: List[Dict[str, Any]] = []
    for q in C.QS:
        p = per[(per.q == q) & (per.gates_pass == True)].sort_values(["Gmax_full_over_dw", "seed"])  # noqa: E712
        n = len(p)
        print(f"q={q}: {n} passing runs of {int((per.q == q).sum())}")
        if n % 2 == 1:
            row = p.iloc[n // 2]
            med = float(row.Gmax_full_over_dw)
            print(f"  odd count: unique median-rank run = rank {n // 2 + 1} of {n}: seed {int(row.seed)}, "
                  f"Gmax_full_over_dw = {med!r}")
            runs.append({"label": f"q{q}_median_passing", "q": q, "seed": int(row.seed)})
        else:
            lo, hi = p.iloc[n // 2 - 1], p.iloc[n // 2]
            med = 0.5 * (float(lo.Gmax_full_over_dw) + float(hi.Gmax_full_over_dw))
            chosen = lo if int(lo.seed) < int(hi.seed) else hi
            other = hi if chosen is lo else lo
            print(f"  even count: median value = {med!r} (mean of the two middle values)")
            print(f"  middle-ranked runs: rank {n // 2} seed {int(lo.seed)} Gmax_full_over_dw = {float(lo.Gmax_full_over_dw)!r} "
                  f"(t={int(lo.Gmax_full_t)}, d={float(lo.Gmax_full_d)}); rank {n // 2 + 1} seed {int(hi.seed)} "
                  f"Gmax_full_over_dw = {float(hi.Gmax_full_over_dw)!r} (t={int(hi.Gmax_full_t)}, d={float(hi.Gmax_full_d)})")
            print(f"  chosen (lower seed): seed {int(chosen.seed)}; the alternative reading (the lower-ranked of the "
                  f"two) would pick seed {int(lo.seed)}"
                  + ("" if int(lo.seed) == int(chosen.seed) else f" instead of seed {int(chosen.seed)} (different run)"))
            runs.append({"label": f"q{q}_median_passing", "q": q, "seed": int(chosen.seed)})
        runs[-1]["n_passing"] = n
        runs[-1]["median_value"] = med
    runs.append({"label": f"q{FAIL_Q}_failed", "q": FAIL_Q, "seed": FAIL_SEED})
    return runs


def main() -> int:
    """Select, re-evaluate, check, write ``data/F4.csv``."""
    per = C.load_per_run(C.PER_RUN_R4)
    check("per_run.csv has 80 rows, arm bwA only, T = 3, seeds 80501-80540 at both q",
          len(per) == 80 and set(per.arm) == {"bwA"} and set(per["T"]) == {3}
          and all(sorted(per[per.q == q].seed) == list(range(80501, 80541)) for q in C.QS))
    pass_q = {q: int(per[per.q == q].gates_pass.sum()) for q in C.QS}
    total = int(per.gates_pass.sum())
    check("pass count q=50 is 40/40", pass_q[50] == 40, f"got {pass_q[50]}/40")
    check("pass count q=60 is 39/40", pass_q[60] == 39, f"got {pass_q[60]}/40")
    check("pass count overall is 79/80", total == EXPECTED_TOTAL_PASS, f"got {total}/80")
    fails = per[~per.gates_pass]
    check("the only failure is q=60, seed 80502",
          len(fails) == 1 and int(fails.iloc[0].q) == FAIL_Q and int(fails.iloc[0].seed) == FAIL_SEED)
    headline = C.read_json(C.ROOT / "results" / "t3_r4" / "confirmation" / "analysis" / "headline.json")
    check("first-attempt pass count is 74/80 (analysis/headline.json trigger.first_attempt_pass)",
          int(headline["trigger"]["first_attempt_pass"]) == EXPECTED_FIRST_ATTEMPT
          and int(headline["trigger"]["n"]) == 80, f"got {headline['trigger']['first_attempt_pass']}/{headline['trigger']['n']}")
    fired_runs = set(headline["trigger"]["runs"])
    if not all(ok for _, ok, _ in CHECKS):
        stop("pass counts do not match the facts of the task")

    runs = select_runs(per)
    rows: List[List[Any]] = []
    for r in runs:
        q, seed, label = r["q"], r["seed"], r["label"]
        pr = per[(per.q == q) & (per.seed == seed)]
        assert len(pr) == 1
        pr = pr.iloc[0]
        run_dir = C.r4_run_dir(q, seed)
        blank_err = isinstance(pr.eval_error, float) and math.isnan(pr.eval_error) or pr.eval_error == ""
        check(f"[{label}] per_run row is {pr.run}: status done, exit code 0, valid, no eval_error",
              str(pr.status) == "done" and int(pr.exit_code) == 0 and bool(pr.valid) and blank_err)
        man = C.read_json(run_dir / "manifest.json")
        gates = C.read_json(run_dir / "gates.json")
        print(f"[{label}] run {pr.run} (q={q}, seed={seed}); run_dir {run_dir}\n"
              f"  manifest.json commit {man['commit']} (clean_tree {man['clean_tree']}, base {man['base_commit']}); "
              f"gates.json commit {gates['commit']}; stage-3 restart fired: {pr.run in fired_runs}; "
              f"per_run gates_pass = {bool(pr.gates_pass)}")
        cfg = t3_eval.read_json(run_dir / "run_config.json")
        cand = t3_eval.load_bw(run_dir, cfg)
        ev = t3_eval.evaluate_tier(cand, FINAL_CONFIG)
        spec = cand.spec
        dw = float(spec.dw)
        check(f"[{label}] spec: T=3, q={q}, dw=4, verifier valid, spec.dw == res.dw",
              int(spec.T) == 3 and int(spec.q) == q and dw == 4.0 and bool(ev.scalars["valid"]) and dw == float(ev.res.dw))
        for key in STORED_KEYS:
            new, old = ev.scalars[key], pr[key]
            check(f"[{label}] {key} bit-equal to per_run.csv", float(new) == float(old),
                  f"recomputed {float(new)!r} stored {float(old)!r}")
        # maximum over stages of G_t/dw from the arrays, and its location
        best_val, best_t, best_j = -np.inf, None, None
        for t in (1, 2, 3):
            g = ev.arrays[f"v_t{t}_G"] / dw
            j = int(np.argmax(g))
            if float(g[j]) > best_val:
                best_val, best_t, best_j = float(g[j]), t, j
        best_d = float(ev.arrays[f"v_t{best_t}_d_grid"][best_j])
        check(f"[{label}] max over stages of G_t/dw (arrays) == Gmax_full_over_dw (per_run.csv), bit for bit",
              best_val == float(pr.Gmax_full_over_dw), f"arrays {best_val!r} stored {float(pr.Gmax_full_over_dw)!r}")
        check(f"[{label}] location of that maximum is (Gmax_full_t, Gmax_full_d) of per_run.csv",
              best_t == int(pr.Gmax_full_t) and best_d == float(pr.Gmax_full_d),
              f"arrays (t={best_t}, d={best_d}) stored (t={int(pr.Gmax_full_t)}, d={float(pr.Gmax_full_d)})")
        eta = float((ev.arrays["v_t3_delta"] / dw).max())
        check(f"[{label}] max of Delta_3/dw (arrays) == eta_T_over_dw, bit for bit", eta == float(pr.eta_T_over_dw),
              f"arrays {eta!r} stored {float(pr.eta_T_over_dw)!r}")
        exp_root = float((ev.arrays["v_t1_G"] / dw)[0])
        check(f"[{label}] G_1(0)/dw (arrays) == EXP_root_over_dw, bit for bit", exp_root == float(pr.EXP_root_over_dw),
              f"arrays {exp_root!r} stored {float(pr.EXP_root_over_dw)!r}")
        g3_eq_d3 = bool(np.array_equal(ev.arrays["v_t3_G"], ev.arrays["v_t3_delta"]))
        print(f"  (G_3 == Delta_3 bit for bit at every stage-3 state: {g3_eq_d3}; max abs diff / dw = "
              f"{float(np.max(np.abs(ev.arrays['v_t3_G'] - ev.arrays['v_t3_delta'])) / dw)!r})")
        if label == f"q{FAIL_Q}_failed":
            check("[failed] G_max,full of seed 80502 is at stage 2, d = -14, value 0.0586...",
                  int(pr.Gmax_full_t) == 2 and float(pr.Gmax_full_d) == -14.0
                  and abs(float(pr.Gmax_full_over_dw) - 0.0586) < 5e-5, f"(t={int(pr.Gmax_full_t)}, d={float(pr.Gmax_full_d)}, "
                  f"value={float(pr.Gmax_full_over_dw)!r})")
        # rows
        for t in (1, 2, 3):
            d = np.asarray(ev.arrays[f"v_t{t}_d_grid"], dtype=float)
            cols = {"e_hat": np.asarray(ev.arrays[f"v_t{t}_e_hat"], dtype=float),
                    "a_br": np.asarray(ev.arrays[f"v_t{t}_a_br"], dtype=float),
                    "G_over_dw": np.asarray(ev.arrays[f"v_t{t}_G"], dtype=float) / dw,
                    "delta_over_dw": np.asarray(ev.arrays[f"v_t{t}_delta"], dtype=float) / dw,
                    "reach": np.asarray(ev.arrays[f"v_t{t}_reach"], dtype=bool)}
            for name, arr in cols.items():
                for dj, vj in zip(d, arr):
                    rows.append([label, q, seed, t, C.fmt(dj), name, str(int(vj)) if name == "reach" else C.fmt(vj)])
        if label == f"q{FAIL_Q}_failed":   # the spec marks G_max,full of seed 80502 only
            rows.append([label, q, seed, int(pr.Gmax_full_t), C.fmt(pr.Gmax_full_d), "gmax_full_marker",
                         C.fmt(pr.Gmax_full_over_dw)])
    if not all(ok for _, ok, _ in CHECKS):
        stop("a recomputed value is not bit-equal to the stored per-run value, or a count differs")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["run_label", "q", "seed", "stage", "d", "quantity", "value"])
        w.writerows(rows)
    print(f"\n{len(CHECKS)} checks, all ok; {len(rows)} rows -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
