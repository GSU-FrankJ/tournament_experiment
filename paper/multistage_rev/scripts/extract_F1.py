"""F1 extraction (T = 2 analytical recovery): learned mean effort of the 40 frozen v2.0 confirmation runs.

For q in (50, 60) and seeds 30501..30520 (run_dir = ``_common.v20_run_dir``) the stored networks are loaded with
``tools/t3/t3_eval.load_v20`` (stage 2 from export u01600, stage 1 from u02200) and the mean effort with
``tools/ms/replay_dev_rule.candidate_fns``:
  * stage 2: ``e_2(d)`` on the whole D_2 at step 0.5 (``utils.v2_metrics.symmetric_grid``);
  * stage 1: ``e_1(0)``;
  * closed-form equilibrium: ``g2_two_stage`` on the same grid and ``g1_two_stage``.
Every run is checked bit for bit against the ``arm == 'v20'`` rows of ``results/t3_r1/eval/per_run.csv`` (match on q
and seed). Any mismatch aborts BEFORE ``data/F1.csv`` is written. Evaluation only, nothing is trained.

Usage (any cwd):  .venv/bin/python paper/multistage_rev/scripts/extract_F1.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402  (sets the single-thread environment before numpy/torch are used)

import numpy as np  # noqa: E402

C.single_thread()

import t3_eval  # noqa: E402
import t3_metrics  # noqa: E402
from replay_dev_rule import candidate_fns  # noqa: E402
from utils.theory_multistage import g1_two_stage, g2_two_stage  # noqa: E402
from utils.v2_metrics import symmetric_grid  # noqa: E402

SEEDS = tuple(range(30501, 30521))
EXPECTED_PASS = {50: 19, 60: 20}
EXPECTED_FAILED = (50, 30510)
OUT_CSV = C.OUT / "data" / "F1.csv"
STORED_VS_RECOMPUTED = ("e1_at_0", "eT_at_0", "stage1_rel_err_signed", "rmse_pos_T_over_eT0",
                        "tail_mean_T_over_eT0", "peak_T_signed")


class Stop(Exception):
    """A check of the exactness / stop rule failed."""


def recompute_recovery(e2: np.ndarray, D: np.ndarray, q: float, spec: Any) -> Dict[str, float]:
    """The terminal-recovery scalars from ``e_2`` on the step-0.5 grid, with the expressions of
    ``tools/t3/t3_metrics.terminal_recovery`` (positive-effort region ``|d| < 2q``)."""
    z = np.flatnonzero(D == 0.0)
    if z.size != 1:
        raise Stop(f"grid has {z.size} nodes at d = 0")
    z = int(z[0])
    g = g2_two_stage(D, q, spec.w_h, spec.w_l, spec.k, spec.e_max)
    g0 = float(g[z])
    pos = np.abs(D) < 2.0 * q
    tail = ~pos
    rmse = float(np.sqrt(np.mean((e2[pos] - g[pos]) ** 2)))
    return {"eT_at_0": float(e2[z]), "eT_star_at_0": g0,
            "peak_T_signed": (float(e2[z]) - g0) / g0, "peak_T_abs": abs(float(e2[z]) - g0) / g0,
            "rmse_pos_T_over_eT0": rmse / g0, "tail_mean_T_over_eT0": float(e2[tail].mean()) / g0}


def main() -> None:
    """Evaluate the 40 runs, run every exactness check, then write ``data/F1.csv``."""
    stored = C.load_per_run(C.PER_RUN_R1)
    v20 = stored[stored["arm"] == "v20"]
    if len(v20) != 40:
        raise Stop(f"{len(v20)} v20 rows in per_run.csv, expected 40")
    counts = {q: (int(v20[v20.q == q].gates_pass.astype(bool).sum()), int((v20.q == q).sum())) for q in C.QS}
    failed = sorted((int(r.q), int(r.seed)) for r in v20.itertuples() if not bool(r.gates_pass))
    print("v20 pass counts (per_run.csv, arm v20, gates_pass):", counts, "failed runs:", failed)
    if counts != {50: (19, 20), 60: (20, 20)} or failed != [EXPECTED_FAILED]:
        raise Stop(f"pass counts {counts} / failed {failed} differ from 19/20, 20/20 / (50, 30510)")

    rows: List[Tuple[str, int, str, Any, Any, float]] = []   # panel, q, series, seed, d, value
    n_cmp = {k: 0 for k in STORED_VS_RECOMPUTED}
    n_extra = {"peak_T_abs": 0, "eT_star_at_0": 0, "terminal_recovery_all": 0, "g1_col": 0}
    for q in C.QS:
        D_ref = None
        for seed in SEEDS:
            rd = C.v20_run_dir(q, seed)
            cfg = C.read_json(rd / "run_config.json")
            cand = t3_eval.load_v20(rd, cfg)
            spec = cand.spec
            if (int(spec.T), int(spec.q), spec.w_h, spec.w_l, spec.k, spec.e_max) != (2, q, C.W_H, C.W_L, C.K, 100):
                raise Stop(f"q={q} seed={seed}: unexpected game spec {spec}")
            if float(cand.recovery_step) != 0.5:
                raise Stop(f"q={q} seed={seed}: recovery_step {cand.recovery_step}")
            mean_fn, _ = candidate_fns(spec, cand.nets)
            D = symmetric_grid(spec.domain_half(2), 0.5)
            if D_ref is None:
                D_ref = D
            elif not np.array_equal(D, D_ref):
                raise Stop(f"q={q}: grid differs between runs")
            e2 = np.asarray(mean_fn(2, D), dtype=float)
            e1 = float(np.asarray(mean_fn(1, np.zeros(1)), dtype=float)[0])
            g1 = g1_two_stage(spec.q, spec.w_h, spec.w_l, spec.k)

            row = v20[(v20.q == q) & (v20.seed == seed)]
            if len(row) != 1:
                raise Stop(f"q={q} seed={seed}: {len(row)} v20 rows in per_run.csv")
            r = row.iloc[0]
            if not str(r["run_dir"]).rstrip("/").endswith(f"q{q}/seed{seed}"):
                raise Stop(f"q={q} seed={seed}: per_run run_dir {r['run_dir']}")
            rec = recompute_recovery(e2, D, float(spec.q), spec)
            mine = {"e1_at_0": e1, "eT_at_0": rec["eT_at_0"], "stage1_rel_err_signed": (e1 - g1) / g1,
                    "rmse_pos_T_over_eT0": rec["rmse_pos_T_over_eT0"],
                    "tail_mean_T_over_eT0": rec["tail_mean_T_over_eT0"], "peak_T_signed": rec["peak_T_signed"]}
            for k in STORED_VS_RECOMPUTED:
                if not (float(r[k]) == mine[k]):
                    raise Stop(f"q={q} seed={seed}: {k} stored {float(r[k])!r} != recomputed {mine[k]!r}")
                n_cmp[k] += 1
            # additional cross-checks (same exactness rule)
            for k in ("peak_T_abs", "eT_star_at_0"):
                if not (float(r[k]) == rec[k]):
                    raise Stop(f"q={q} seed={seed}: {k} stored {float(r[k])!r} != recomputed {rec[k]!r}")
                n_extra[k] += 1
            sc, _ = t3_metrics.terminal_recovery(mean_fn, spec, 0.5)
            for k in ("eT_at_0", "eT_star_at_0", "peak_T_signed", "peak_T_abs", "rmse_pos_T_over_eT0",
                      "tail_mean_T_over_eT0"):
                if not (float(sc[k]) == rec[k]):
                    raise Stop(f"q={q} seed={seed}: terminal_recovery {k} {sc[k]!r} != mine {rec[k]!r}")
            n_extra["terminal_recovery_all"] += 1
            if "g1" in v20.columns and not np.isnan(float(r["g1"])):
                if float(r["g1"]) != g1:
                    raise Stop(f"q={q} seed={seed}: g1 stored {float(r['g1'])!r} != {g1!r}")
                n_extra["g1_col"] += 1

            is_failed = (q, seed) == EXPECTED_FAILED
            if bool(r["gates_pass"]) == is_failed:
                raise Stop(f"q={q} seed={seed}: gates_pass {r['gates_pass']} inconsistent with failed={is_failed}")
            series = "failed_run" if is_failed else "run"
            for d, v in zip(D, e2):
                rows.append(("stage2", q, series, seed, float(d), float(v)))
            rows.append(("stage1", q, series, seed, None, e1))
        g2 = g2_two_stage(D_ref, float(q), C.W_H, C.W_L, C.K, 100)
        for d, v in zip(D_ref, g2):
            rows.append(("stage2", q, "closed_form_equilibrium", None, float(d), float(v)))
        rows.append(("stage1", q, "closed_form_equilibrium", None, None, float(g1_two_stage(q, C.W_H, C.W_L, C.K))))

    print("bit-equal comparisons against per_run.csv (runs per quantity):", n_cmp)
    print("additional bit-equal comparisons:", n_extra)
    if any(v != 40 for v in n_cmp.values()):
        raise Stop(f"not all 40 runs compared: {n_cmp}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["panel", "q", "series", "seed", "d", "value"])
        for panel, q, series, seed, d, value in rows:
            w.writerow([panel, q, series, "" if seed is None else seed, "" if d is None else repr(d), repr(value)])
    tmp.replace(OUT_CSV)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    try:
        main()
    except Stop as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)
