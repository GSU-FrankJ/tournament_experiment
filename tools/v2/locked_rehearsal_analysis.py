#!/usr/bin/env python3
"""Dress-rehearsal analysis of the locked v2 T=2 pipeline on the development seeds (descriptive).

Check 1  end-of-Phase-A full state of every rehearsal run vs the stitched development state
         (Pilot 1 expected u400 -> Phase A extension u1200 -> Pilot 4 2a decay u1600):
         actor, critic, opponent, both Adam states, minibatch RNG, the numpy streams, the torch
         generator and the torch / numpy / python global RNG states (state_end_A.pt vs
         pilot4_A/.../decay/state_u01600.pt); every weight export u25..u1600 against the export of
         the stitched segment that produced it; stage-2 metrics of the u1600 verifier call
         (v2_checkpoints_A.csv vs pilot4_A decay v2_checkpoints.csv).
Check 2  Phase B of the rehearsal vs the pilot-launcher B2_mean_decay run from the stitched u1600
         state (results/v2_T2_locked/locked_check2_phaseB/, one seed per q): history, stability,
         verifier calls, exports, final weights, RNG positions, end state.
Gates    per-run table (final and dev tier), pass counts per q, distributions (min, 10/25/50/75/90%,
         max) and dev - final differences; per-update LR vs the locked schedule.

Usage: python tools/v2/locked_rehearsal_analysis.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from compare_full_runs import compare, teq  # noqa: E402
from run.run_final_dp_br_round3_dense import lr_at  # noqa: E402

LK = ROOT / "results" / "v2_T2_locked"
RH = LK / "rehearsal"
C2 = LK / "locked_check2_phaseB"
V2 = ROOT / "results" / "v2_pilots"
OUT = LK / "rehearsal_analysis"
QS, SEEDS = (50, 60), range(10501, 10511)
STAGE2_COLS = ("eta_T_over_dw", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean_over_g2_0", "stage2_peak_rel_err_signed",
               "stage2_sym_err_max", "stage2_tail_max", "DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max",
               "sigma_effort_at_0_t2", "e2_at_0")


def stitched_export(q: int, seed: int, u: int) -> Path:
    if u <= 400:
        return V2 / "pilot1" / f"q{q}" / f"seed{seed}" / "expected" / "weights" / f"u{u:05d}.npz"
    if u <= 1200:
        return V2 / "phaseA_ext" / f"q{q}" / f"seed{seed}" / "expected_ext" / "weights" / f"u{u:05d}.npz"
    return V2 / "pilot4_A" / f"q{q}" / f"seed{seed}" / "decay" / "weights" / f"u{u:05d}.npz"


def check1() -> pd.DataFrame:
    rows = []
    for q in QS:
        for s in SEEDS:
            d = RH / f"q{q}" / f"seed{s}"
            a = torch.load(d / "state_end_A.pt", weights_only=False)
            b = torch.load(V2 / "pilot4_A" / f"q{q}" / f"seed{s}" / "decay" / "state_u01600.pt", weights_only=False)
            r = {"q": q, "seed": s, "global_u": a["counters"]["global_u"], "stitched_global_u": b["counters"]["global_u"]}
            for k in ("actor", "critic", "opponent", "opt_actor", "opt_critic", "rng_minibatch"):
                r[f"{k}_identical"] = teq(a["agent"][k], b["agent"][k])
            r["rng_streams_identical"] = teq(a["rng"], b["rng"])
            r["torch_generator_identical"] = teq(a["torch_generator_state"], b["torch_generator_state"])
            r["torch_global_rng_identical"] = teq(a["torch_global_rng_state"], b["torch_global_rng_state"])
            r["numpy_global_rng_identical"] = teq(a["numpy_global_rng_state"], b["numpy_global_rng_state"])
            r["python_random_identical"] = teq(a["python_random_state"], b["python_random_state"])
            r["snapshot_refreshes_rehearsal"] = a["agent"]["snapshot_refreshes"]
            r["snapshot_refreshes_stitched"] = b["agent"]["snapshot_refreshes"]
            ex = sorted(glob.glob(str(d / "weights" / "u*.npz")))
            exA = [f for f in ex if int(os.path.basename(f)[1:6]) <= 1600]
            same = [all(np.array_equal(np.load(f)[k], np.load(stitched_export(q, s, int(os.path.basename(f)[1:6])))[k])
                        for k in np.load(f).files) for f in exA]
            r["n_exports_A"] = len(exA)
            r["exports_A_identical"] = bool(all(same) and len(exA) == 64)
            ca = pd.read_csv(d / "v2_checkpoints_A.csv").set_index("update").loc[1600]
            cb = pd.read_csv(V2 / "pilot4_A" / f"q{q}" / f"seed{s}" / "decay" / "v2_checkpoints.csv").set_index("update").loc[1600]
            r["stage2_metrics_u1600_identical"] = bool(all(float(ca[c]) == float(cb[c]) for c in STAGE2_COLS))
            r["ALL_IDENTICAL"] = all(v for k, v in r.items() if k.endswith("_identical"))
            rows.append(r)
    return pd.DataFrame(rows)


def check2() -> pd.DataFrame:
    rows = []
    for d in sorted(glob.glob(str(C2 / "q*" / "seed*" / "B2_mean_decay"))):
        man = json.load(open(os.path.join(d, "manifest.json")))
        q, s = int(man["q"]), int(man["seed"])
        rd = RH / f"q{q}" / f"seed{s}"
        ha = json.load(open(rd / "train_history.json"))
        hb = json.load(open(os.path.join(d, "train_history.json")))
        strip = lambda xs: [{k: v for k, v in x.items() if k not in ("time_sec", "update_wall_sec")} for x in xs]  # noqa: E731
        hA = [x for x in ha["history"] if x["phase"] == "B"]
        r = {"q": q, "seed": s, "parent": os.path.relpath(man["parent_checkpoint"], ROOT),
             "history_B_identical": strip(hA) == strip(hb["history"]),
             "stability_B_identical": [x for x in ha["stability"] if x["phase"] == "B"] == hb["stability"],
             "verifier_calls_B_identical": json.dumps(strip([x for x in ha["verifier_calls"] if x["phase"] == "B"]), sort_keys=True, default=str)
             == json.dumps(strip(hb["verifier_calls"]), sort_keys=True, default=str)}
        exB = [f for f in sorted(glob.glob(str(rd / "weights" / "u*.npz"))) if int(os.path.basename(f)[1:6]) > 1600]
        eb = sorted(glob.glob(os.path.join(d, "weights", "u*.npz")))
        r["n_exports_B"] = len(exB)
        r["exports_B_identical"] = [os.path.basename(x) for x in exB] == [os.path.basename(x) for x in eb] and all(
            all(np.array_equal(np.load(x)[k], np.load(y)[k]) for k in np.load(x).files) for x, y in zip(exB, eb))
        wa, wb = np.load(rd / "checkpoint_weights.npz"), np.load(os.path.join(d, "checkpoint_weights.npz"))
        r["final_weights_identical"] = all(np.array_equal(wa[k], wb[k]) for k in wa.files)
        ua = pd.read_csv(rd / "v2_updates.csv")
        ua = ua[ua.phase == "B"].reset_index(drop=True)
        ub = pd.read_csv(os.path.join(d, "v2_updates.csv"))
        rc = ["update"] + [c for c in ub.columns if c.startswith("rngpos_")]
        r["rng_positions_B_identical"] = ua[rc].equals(ub[rc])
        sa = torch.load(rd / "state_end_B.pt", weights_only=False)
        sb = torch.load(os.path.join(d, "state_end_B.pt"), weights_only=False)
        for k in ("actor", "critic", "opponent", "frozen", "opt_actor", "opt_critic", "rng_minibatch"):
            r[f"end_state_{k}_identical"] = teq(sa["agent"][k], sb["agent"][k])
        r["end_state_rng_identical"] = teq(sa["rng"], sb["rng"]) and teq(sa["torch_generator_state"], sb["torch_generator_state"])
        ca = pd.read_csv(rd / "v2_checkpoints_B.csv")
        cb = pd.read_csv(os.path.join(d, "v2_checkpoints.csv"))
        cols = ["update", "e1_at_0", "Gmax_full_over_dw", "EXP_root_over_dw", "dReach_over_dw", "stage1_rel_err_signed"]
        r["checkpoint_metrics_B_identical"] = ca[cols].equals(cb[cols])
        r["ALL_IDENTICAL"] = all(v for k, v in r.items() if k.endswith("_identical"))
        rows.append(r)
    return pd.DataFrame(rows)


def gates_table():
    rows, lr_rows = [], []
    proto = json.load(open(ROOT / "protocols" / "v2_T2_locked.json"))
    win = {w["phase"]: w for w in proto["pipeline"]["lr_decay"]}
    for q in QS:
        for s in SEEDS:
            d = RH / f"q{q}" / f"seed{s}"
            g = json.load(open(d / "gates.json"))
            man = json.load(open(d / "manifest.json"))
            summ = json.load(open(d / "v2_run_summary.json"))
            r = {"q": q, "seed": s, "commit": man["git"]["short"], "clean_tree": man["clean_tree"],
                 "protocol_sha256": man["locked_protocol"]["sha256"][:12], "G-A": g["G-A"]["pass"], "G-F": g["G-F"]["pass"],
                 "G-A_dev": g["G-A"]["pass_dev_tier"], "G-F_dev": g["G-F"]["pass_dev_tier"], "run_pass": g["run_pass"],
                 "outcome": g["outcome"], "wall_sec": summ["total_wall_sec"]}
            for gate in ("G-A", "G-F"):
                for c in g[gate]["criteria"]:
                    r[f"{c['metric']}_final"] = c["value_final"]
                    r[f"{c['metric']}_dev"] = c["value_dev"]
                    r[f"{c['metric']}_dev_minus_final"] = c["dev_minus_final"]
                    r[f"{c['metric']}_pass"] = c["pass_final"]
            ra, rb = g["reported"]["end_of_A"], g["reported"]["end_of_B"]
            for k, v in ra["final"].items():
                r[f"A_{k}"] = v
            for k, v in ra["smoothed_game"].items():
                r[f"A_{k}"] = v
            for k, v in rb["final"].items():
                r[f"B_{k}"] = v
            dec = rb["decomposition"]
            for k in ("e_tilde", "band_lo", "band_hi", "learning_rel", "learning_rel_lo", "learning_rel_hi", "inherited_rel",
                      "inherited_rel_lo", "inherited_rel_hi", "learning_contains_0", "inherited_contains_0", "band_contiguous",
                      "e1_inside_sweep", "sweep_lo", "sweep_hi"):
                r[f"dec_{k}"] = dec[k]
            r["drift_test_pass"] = g["reported"]["drift_test_pass"]
            for k, v in ra["dev_minus_final"].items():
                r[f"A_dmf_{k}"] = v
            for k, v in rb["dev_minus_final"].items():
                r[f"B_dmf_{k}"] = v
            rows.append(r)
            h = json.load(open(d / "train_history.json"))["history"]
            sched = json.load(open(d / "run_config.json"))["record"]["lr_schedule"]
            bad = 0
            for x in h:
                w = win[x["phase"]]
                if x["local"] < w["local_first"]:
                    want = lr_at(sched, x["phase"], x["local"])
                else:
                    lin = dict(sched, kind="linear", c_start_lr=w["start_lr"], c_end_lr=w["end_lr"], c_local_first=w["local_first"],
                               linear_denominator=w["local_last"] - w["local_first"])
                    want = lr_at(lin, "C", x["local"])
                bad += int(x["actor_lr"] != want or x["critic_lr"] != want)
            lr_rows.append({"q": q, "seed": s, "n_updates": len(h), "n_A": sum(x["phase"] == "A" for x in h),
                            "n_B": sum(x["phase"] == "B" for x in h), "n_lr_mismatch": bad,
                            "actor_minibatch_steps_A": sum(x["n_minibatch_steps"] for x in h if x["phase"] == "A"),
                            "actor_minibatch_steps_A_last400": sum(x["n_minibatch_steps"] for x in h if x["phase"] == "A" and x["local"] > 1200)})
    return pd.DataFrame(rows), pd.DataFrame(lr_rows)


def distributions(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.endswith(("_final", "_dev", "_dev_minus_final"))]
    out = []
    for q, g in df.groupby("q"):
        for c in cols:
            x = g[c].astype(float)
            out.append({"q": q, "metric": c, "min": x.min(), "p10": x.quantile(.1), "p25": x.quantile(.25), "median": x.median(),
                        "p75": x.quantile(.75), "p90": x.quantile(.9), "max": x.max()})
    return pd.DataFrame(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    c1 = check1()
    c1.to_csv(OUT / "check1_phaseA_vs_stitched.csv", index=False)
    print("CHECK 1 all identical:", bool(c1.ALL_IDENTICAL.all()), f"({int(c1.ALL_IDENTICAL.sum())}/{len(c1)})")
    c2 = check2()
    c2.to_csv(OUT / "check2_phaseB_vs_launcher.csv", index=False)
    print("CHECK 2 all identical:", bool(c2.ALL_IDENTICAL.all()), f"({int(c2.ALL_IDENTICAL.sum())}/{len(c2)})")
    gt, lr = gates_table()
    gt.to_csv(OUT / "gates_per_run.csv", index=False)
    lr.to_csv(OUT / "lr_schedule_check.csv", index=False)
    distributions(gt).to_csv(OUT / "gate_distributions.csv", index=False)
    pc = gt.groupby("q").agg(n=("seed", "size"), G_A_pass=("G-A", "sum"), G_F_pass=("G-F", "sum"), run_pass=("run_pass", "sum"),
                             G_A_pass_dev=("G-A_dev", "sum"), G_F_pass_dev=("G-F_dev", "sum")).reset_index()
    pc.to_csv(OUT / "pass_counts.csv", index=False)
    print(pc.to_string())
    print("LR mismatches:", int(lr.n_lr_mismatch.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
