"""Registry of the T=2 v2 studies and readers of their per-run records.

Layout of a run directory (pilot studies): ``<root>/q<q>/seed<seed>/<arm>/``; locked studies are
flat: ``<root>/q<q>/seed<seed>/``. All paths here are repo-relative (``results/...``); use
``common.abspath`` / ``common.src`` to resolve and hash them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

import common as C

QS = (50, 60)
SEEDS_DEV = list(range(10501, 10511))
SEEDS_CONF = list(range(20501, 20521))

# arm label (directory) -> short display name used in tables and figures
ARM_SHORT = {
    "sampled": "sampled", "expected": "expected", "expected_ext": "expected_ext",
    "A_joint": "A", "B1_frozen_allnorm": "B1", "B2_frozen_s1norm": "B2",
    "B2_frozen_s1norm_mean": "B2+mean", "constant": "constant", "decay": "decay",
    "B2_mean_constant": "B2_mean_constant", "B2_mean_decay": "B2_mean_decay",
}

PILOT = "results/v2_pilots"
LOCKED = "results/v2_T2_locked"

STUDIES: Dict[str, Dict[str, Any]] = {
    "pilot1": dict(
        title="Pilot 1: terminal reward estimator (sampled vs expected)", root=f"{PILOT}/pilot1",
        arms=["sampled", "expected"], seeds=SEEDS_DEV, qs=QS, report="reports/v2/pilot1_reward_estimator.md",
        launch_commit="89cd600", phase="A", budget="Phase A 400 updates (u1-u400), fixed budget", final_update=400,
        parents="none (from initialization)", analysis=f"{PILOT}/pilot1/analysis", kind="pilot"),
    "pilot2": dict(
        title="Pilot 2: joint vs frozen stage 2 in Phase B", root=f"{PILOT}/pilot2",
        arms=["A_joint", "B1_frozen_allnorm", "B2_frozen_s1norm"], seeds=SEEDS_DEV, qs=QS,
        report="reports/v2/pilot2_freeze.md", launch_commit="1791687", phase="B",
        budget="Phase B 600 updates (global u401-u1000), fixed budget", final_update=1000,
        parents=f"{PILOT}/pilot1/q*/seed*/expected/state_end_A.pt (u400)", analysis=f"{PILOT}/pilot2/analysis", kind="pilot"),
    "pilot3": dict(
        title="Pilot 3: continuation action mode, stochastic vs mean (frozen B2)", root=f"{PILOT}/pilot3",
        arms=["B2_frozen_s1norm", "B2_frozen_s1norm_mean"], seeds=SEEDS_DEV, qs=QS,
        report="reports/v2/pilot3_continuation_mode.md", launch_commit="cd760fd", phase="B",
        budget="Phase B 600 updates (global u401-u1000), fixed budget", final_update=1000,
        parents=f"{PILOT}/pilot1/q*/seed*/expected/state_end_A.pt (u400)", analysis=f"{PILOT}/pilot3/analysis", kind="pilot"),
    "phaseA_ext": dict(
        title="Phase A extension (stage 2 only, u400 to u1600)", root=f"{PILOT}/phaseA_ext", arms=["expected_ext"],
        seeds=SEEDS_DEV, qs=QS, report="reports/v2/phaseA_ext.md", launch_commit="cd760fd", phase="Acont",
        budget="Phase A continued by 1,200 updates (global u401-u1600), fixed budget", final_update=1600,
        parents=f"{PILOT}/pilot1/q*/seed*/expected/state_end_A.pt (u400)", analysis=f"{PILOT}/phaseA_ext/analysis", kind="pilot"),
    "pilot4_A": dict(
        title="Pilot 4 section 2a: Phase A end-of-phase LR decay", root=f"{PILOT}/pilot4_A", arms=["constant", "decay"],
        seeds=SEEDS_DEV, qs=QS, report="reports/v2/pilot4_stabilization.md", launch_commit="c92ee74", phase="Acont",
        budget="Phase A continued by 400 updates (global u1201-u1600); decay arm 3e-4 to 3e-5", final_update=1600,
        parents=f"{PILOT}/phaseA_ext/q*/seed*/expected_ext/state_u01200.pt (u1200)", analysis=f"{PILOT}/pilot4/analysis",
        kind="pilot"),
    "pilot4_B": dict(
        title="Pilot 4 section 2b: Phase B LR decay (B2 + mean)", root=f"{PILOT}/pilot4_B",
        arms=["B2_mean_constant", "B2_mean_decay"], seeds=SEEDS_DEV, qs=QS, report="reports/v2/pilot4_stabilization.md",
        launch_commit="c92ee74", phase="B", budget="Phase B 600 updates (global u1601-u2200); decay arm 3e-4 to 3e-5",
        final_update=2200, parents=f"{PILOT}/phaseA_ext/q*/seed*/expected_ext/state_u01600.pt (u1600)",
        analysis=f"{PILOT}/pilot4/analysis", kind="pilot"),
    "pilot4_B_rerun_clean": dict(
        title="Dirty-flag re-run: Pilot 4 section 2b q=60 seed 10507 B2_mean_constant", root=f"{PILOT}/pilot4_B_rerun_clean",
        arms=["B2_mean_constant"], seeds=[10507], qs=(60,), report="reports/v2/protocol_lock_and_rehearsal.md",
        launch_commit="5d50a9d", phase="B", budget="Phase B 600 updates (global u1601-u2200)", final_update=2200,
        parents=f"{PILOT}/phaseA_ext/q60/seed10507/expected_ext/state_u01600.pt", analysis="", kind="pilot"),
    "rehearsal": dict(
        title="v1.0 dress rehearsal (development seeds)", root=f"{LOCKED}/rehearsal", arms=["locked"], seeds=SEEDS_DEV, qs=QS,
        report="reports/v2/protocol_lock_and_rehearsal.md", launch_commit="5b07293", phase="locked",
        budget="Phase A 1600 + Phase B 600 updates, single process", final_update=2200, parents="none (from initialization)",
        analysis=f"{LOCKED}/rehearsal_analysis", kind="locked"),
    "locked_check2_phaseB": dict(
        title="v1.0 Check 2: Phase B code path from the stitched u1600 state", root=f"{LOCKED}/locked_check2_phaseB",
        arms=["B2_mean_decay"], seeds=[10503], qs=QS, report="reports/v2/protocol_lock_and_rehearsal.md",
        launch_commit="5b07293", phase="B", budget="Phase B 600 updates (global u1601-u2200)", final_update=2200,
        parents=f"{PILOT}/pilot4_A/q*/seed10503/decay/state_u01600.pt", analysis=f"{LOCKED}/rehearsal_analysis", kind="pilot"),
    "rehearsal_v1_1": dict(
        title="v1.1 re-rehearsal (development seeds)", root=f"{LOCKED}/rehearsal_v1_1", arms=["locked"], seeds=SEEDS_DEV, qs=QS,
        report="reports/v2/protocol_v1_1_confirmation.md", launch_commit="95c000e", phase="locked",
        budget="Phase A 1600 + Phase B 600 updates, single process", final_update=2200, parents="none (from initialization)",
        analysis=f"{LOCKED}/rehearsal_v1_1_analysis", kind="locked"),
    "confirmation": dict(
        title="v1.1 fresh-seed confirmation", root=f"{LOCKED}/confirmation", arms=["locked"], seeds=SEEDS_CONF, qs=QS,
        report="reports/v2/protocol_v1_1_confirmation.md", launch_commit="f6838ec", phase="locked",
        budget="Phase A 1600 + Phase B 600 updates, single process", final_update=2200, parents="none (from initialization)",
        analysis=f"{LOCKED}/confirmation_analysis", kind="locked"),
}


def run_dir(study: str, q: int, seed: int, arm: Optional[str] = None) -> str:
    """Repo-relative run directory."""
    s = STUDIES[study]
    base = f"{s['root']}/q{q}/seed{seed}"
    return base if s["kind"] == "locked" else f"{base}/{arm}"


def iter_runs(study: str) -> Iterator[Tuple[int, int, str, str]]:
    """``(q, seed, arm, run_dir)`` over the planned runs of a study (q, then seed, then arm)."""
    s = STUDIES[study]
    for q in s["qs"]:
        for seed in s["seeds"]:
            for arm in s["arms"]:
                yield q, seed, arm, run_dir(study, q, seed, arm)


def read_json(rel: str) -> Any:
    """JSON file under the repo (``results/...`` resolved against the results root)."""
    with open(C.abspath(rel), encoding="utf-8") as fh:
        return json.load(fh)


def read_csv(rel: str, **kw) -> pd.DataFrame:
    """CSV file under the repo."""
    kw.setdefault("float_precision", "round_trip")   # pandas 3 default drops the 17th significant digit
    return pd.read_csv(C.abspath(rel), **kw)


def manifest(rd: str) -> Dict[str, Any]:
    """``manifest.json`` of a run directory."""
    return read_json(f"{rd}/manifest.json")


def status(rd: str) -> Dict[str, Any]:
    """``status.json`` of a run directory (state, start/end time, wall seconds, exit code)."""
    return read_json(f"{rd}/status.json")


def final_v2(rd: str) -> Dict[str, Any]:
    """``final_v2.json`` of a pilot run: scalars of the last checkpoint on both tiers."""
    return read_json(f"{rd}/final_v2.json")


def checkpoints(rd: str) -> pd.DataFrame:
    """Training-time verifier checkpoint rows (``v2_checkpoints.csv``; locked runs: A and B files concatenated)."""
    p = C.abspath(f"{rd}/v2_checkpoints.csv")
    if p.is_file():
        return pd.read_csv(p, float_precision="round_trip")
    parts = []
    for ph in ("A", "B"):
        pp = C.abspath(f"{rd}/v2_checkpoints_{ph}.csv")
        if pp.is_file():
            parts.append(pd.read_csv(pp, float_precision="round_trip"))
    return pd.concat(parts, ignore_index=True)


def updates(rd: str) -> pd.DataFrame:
    """Per-update training log (``v2_updates.csv``)."""
    return pd.read_csv(C.abspath(f"{rd}/v2_updates.csv"), float_precision="round_trip")


def tier_scalars(rd: str, tier: str) -> Dict[str, Any]:
    """Scalars of the final checkpoint of a pilot run on ``tier`` ('development' or 'final')."""
    return final_v2(rd)[tier]


def final_tier_columns(study: str, arms: Optional[List[str]] = None) -> pd.DataFrame:
    """Final-tier scalars of every run of a pilot study, one row per (q, seed, arm).

    Reads ``final_v2.json['final']`` (the last checkpoint evaluated on the final tier) and keeps the
    tier-dependent verifier metrics (Gmax_full, EXP_root, dReach, Delta_max_all, dFull, eta_2, Delta_2 on/off
    split, G_max_t*, Delta_max_t*, invariants, sigma, ...) as ``final_tier__<name>``. Recovery metrics are tier
    independent and are not repeated. The development-tier counterparts are in the existing
    ``final_table.csv`` of the study (they come from the last training-time checkpoint).

    Args:
        study: Pilot study key (``pilot1`` ... ``pilot4_B``).
        arms: Optional subset of arms.

    Returns:
        DataFrame with ``q, seed, arm, run_dir`` and the ``final_tier__*`` columns.
    """
    import dictionary as D
    rows = []
    for q, seed, arm, rd in iter_runs(study):
        if arms and arm not in arms:
            continue
        fin = final_v2(rd)["final"]
        rec = {"q": q, "seed": seed, "arm": arm, "run_dir": rd}
        for k, v in fin.items():
            ent = D.E.get(k)
            tier_dep = (ent[3] is True) if ent else (k.startswith("inv_") or k.startswith("DeltaT_over_dw_"))
            if tier_dep and k not in ("verifier_tier", "state_step", "effort_step", "gl_half", "valid", "invalid_reasons"):
                rec[f"final_tier__{k}"] = v
        rows.append(rec)
    return pd.DataFrame(rows)
