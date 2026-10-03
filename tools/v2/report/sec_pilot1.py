"""Pack items of Pilot 1 (sampled vs expected terminal reward) and the design summary of Pilots 1-3.

Items (request section 3, "Pilots"):

* T19 design summary of Pilots 1-3: counts, flags, commits, parents and wall times from the run
  records; question and decision text from the reports ("source: report text");
* T20 Pilot 1 final (u400) medians and IQR per arm and q, both verifier tiers for the
  tier-dependent metrics, location-free peak error derived from the saved recovery arrays;
* T21 Pilot 1 paired differences expected - sampled: the existing bootstrap rows
  (``paired_summary.csv``) plus location-free and final-tier rows computed with the same method;
* F03 Pilot 1 learning curves from the 25-update weight exports (existing re-evaluation CSV);
* F04 Pilot 1 stage-2 mapping against the closed form at u400, with sigma_2(d) below;
* F05 Pilot 1 peak error against sigma_2(0)/q at every training-time verifier checkpoint;
* D01 Pilot 1 per-run table (original ``final_table.csv`` columns plus appended columns).

Only saved files are read: no forward pass and no verifier evaluation is performed.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as C
import dictionary as D
import studies as S
import style

if str(C.REPO) not in sys.path:
    sys.path.append(str(C.REPO))
from utils.theory_multistage import g2_two_stage  # noqa: E402

MOD = "sec_pilot1.py"
P1 = "results/v2_pilots/pilot1"
A1 = f"{P1}/analysis"
FINAL_TABLE = f"{A1}/final_table.csv"
PAIRED = f"{A1}/paired_summary.csv"
CURVES = f"{A1}/curves_weights_every25.csv"
SPEARMAN = f"{A1}/spearman_peakerr_vs_sigma.csv"
SMOOTHED = f"{A1}/smoothed_game/per_run.csv"
EXT_TABLE = "results/v2_pilots/phaseA_ext/analysis/table_400_800_1200_1600.csv"
META = f"{A1}/analysis_meta.json"
REP1 = "reports/v2/pilot1_reward_estimator.md"
REP2 = "reports/v2/pilot2_freeze.md"
REP3 = "reports/v2/pilot3_continuation_mode.md"
REP4 = "reports/v2/pilot4_stabilization.md"
SUMMARY = "reports/v2/summary.md"
FIG_ORIG = "reports/v2/figures/pilot1"
ARMS = ("sampled", "expected")
QS = (50, 60)
N_BOOT, BOOT_SEED = 10000, 20261001
TI, DEV, FIN = "tier-independent", "development", "final"
NA_TIER = "n/a (not a verifier quantity)"
NPZ_KEYS = ("recovery_d_grid", "recovery_e2", "recovery_g2", "v_t2_d_grid", "v_t2_sigma_effort",
            "v_t2_e_hat")
TIER_DEF = ("development = state step 4, effort step 1, GL 16 per half (the u400 training-time "
            "checkpoint row of v2_checkpoints.csv, collected in final_table.csv); final = state "
            "step 2, effort step 0.5, GL 32 per half (final_v2.json['final'], the same u400 policy "
            "re-evaluated at the end of the run); tier-independent = 0.5-step recovery grid or "
            "direct policy query")


# ----------------------------------------------------------------------------------------------
# readers and small helpers
# ----------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _manifest(rd: str) -> Dict[str, Any]:
    """Cached ``manifest.json`` of a run directory."""
    return S.manifest(rd)


@lru_cache(maxsize=None)
def _final_v2(rd: str) -> Dict[str, Any]:
    """Cached ``final_v2.json`` of a run directory."""
    return S.final_v2(rd)


@lru_cache(maxsize=None)
def _npz(rd: str, tier: str) -> Dict[str, np.ndarray]:
    """Arrays of ``final_<tier>.npz`` needed here (recovery grid, stage-2 mean and sigma)."""
    with np.load(C.abspath(f"{rd}/final_{tier}.npz")) as z:
        return {k: np.asarray(z[k]) for k in NPZ_KEYS}


def _read_final_table() -> pd.DataFrame:
    """``final_table.csv`` with the literal ``null`` strings kept (column would_have_fired_A)."""
    return pd.read_csv(C.abspath(FINAL_TABLE), keep_default_na=False, na_values=[""], float_precision="round_trip")


def _seeds_text(seeds: Iterable[int]) -> str:
    """Compact text of a seed list ('10501-10510' when contiguous)."""
    s = sorted(int(x) for x in seeds)
    if s and s == list(range(s[0], s[-1] + 1)):
        return f"{s[0]}-{s[-1]}"
    return ", ".join(str(x) for x in s)


def _record_check(pack: C.Pack, item: str, report: str, label: str, n: int, bad: int) -> None:
    """List a prose/text check with the module's cross-checks (``consistency.md``)."""
    pack.crosschecks.append({"item": item, "report": report, "label": label, "n_tables": 0,
                             "n_compared": n, "n_mismatch": bad, "n_unmatched_rows": 0})


def _prose(pack: C.Pack, item: str, report: str, label: str,
           checks: Sequence[Tuple[str, float, str]]) -> int:
    """Compare pack values with numbers quoted in report prose at the report's precision.

    Mismatches are recorded with ``pack.mismatch``; the check itself is listed with the other
    cross-checks of the module (``consistency.md``, 'Checks performed').

    Args:
        pack: The module's pack.
        item: Pack item the values belong to.
        report: Report path.
        label: What was compared (shown in consistency.md).
        checks: ``(quantity, pack value, number as printed in the report)`` triples.

    Returns:
        The number of mismatches.
    """
    bad = 0
    for quantity, value, text in checks:
        if not C.consistent(float(value), text):
            bad += 1
            pack.mismatch(item, quantity, value, report, text, "number quoted in the report prose")
    _record_check(pack, item, report, "prose: " + label, len(checks), bad)
    return bad


def _claims(pack: C.Pack, item: str, report: str, label: str,
            claims: Sequence[Tuple[str, str, str, bool]]) -> int:
    """Record text claims of a report (commit hashes, flags) checked against the run records.

    Args:
        pack: The module's pack.
        item: Pack item the values belong to.
        report: Report path.
        label: What was compared (shown in consistency.md).
        claims: ``(quantity, pack value, report text, agrees)`` tuples.

    Returns:
        The number of disagreements (each recorded with ``pack.mismatch``).
    """
    bad = 0
    for quantity, value, text, ok in claims:
        if not ok:
            bad += 1
            pack.mismatch(item, quantity, value, report, text, "text claim")
    _record_check(pack, item, report, "text: " + label, len(claims), bad)
    return bad


def _p1_sources() -> Dict[str, C.SourceSet]:
    """Source sets of the per-run files of Pilot 1 (40 runs each)."""
    pat = f"{P1}/q*/seed*/*/"
    names = ("final_v2.json", "final_development.npz", "final_final.npz", "v2_checkpoints.csv",
             "manifest.json")
    return {name: C.srcs(pat + name, label=f"pilot1 per-run {name}", expect=40) for name in names}


# ----------------------------------------------------------------------------------------------
# per-run values shared by T20, T21, F04 and D01
# ----------------------------------------------------------------------------------------------

def _run_frame() -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Per-run location-free peak error (recovery arrays) and the final-tier scalars.

    Returns:
        ``(frame, checks)``: one row per (q, seed, arm) with the location-free columns and the
        ``final_tier__*`` columns of ``studies.final_tier_columns``; ``checks`` holds max abs
        diffs: recovery arrays of final_development.npz vs final_final.npz, e2*(0) of
        final_v2.json vs the recovery grid, the repository closed form vs ``recovery_g2``,
        sigma_2(d) of the final-tier grid vs the development-tier grid at the common nodes, and
        ``recovery_e2`` vs the verifier's ``v_t2_e_hat`` at the development D2 nodes.
    """
    chk = dict(npz_dev_vs_final=0.0, g20_json_vs_grid=0.0, closed_form_vs_grid=0.0,
               sigma_final_vs_dev=0.0, recovery_vs_ehat=0.0)

    def upd(key: str, value: float) -> None:
        chk[key] = max(chk[key], float(value))

    rows = []
    for q, seed, arm, rd in S.iter_runs("pilot1"):
        zd, zf = _npz(rd, "development"), _npz(rd, "final")
        for k in ("recovery_d_grid", "recovery_e2", "recovery_g2"):
            upd("npz_dev_vs_final", C.max_abs_diff(zd[k], zf[k]))
        game = _manifest(rd)["resolved_config"]["game"]
        grid, e2 = zd["recovery_d_grid"], zd["recovery_e2"]
        g2 = g2_two_stage(grid, q, game["w_h"], game["w_l"], game["k"], game["e_max"])
        upd("closed_form_vs_grid", C.max_abs_diff(g2, zd["recovery_g2"]))
        g20 = float(_final_v2(rd)["development"]["g2_at_0"])
        upd("g20_json_vs_grid", abs(g20 - float(zd["recovery_g2"][grid == 0.0][0])))
        dd, fd = zd["v_t2_d_grid"], zf["v_t2_d_grid"]
        idx = np.searchsorted(fd, dd)
        if not np.array_equal(fd[idx], dd):
            raise RuntimeError(f"{rd}: development D2 grid is not a subset of the final D2 grid")
        upd("sigma_final_vs_dev",
            C.max_abs_diff(zf["v_t2_sigma_effort"][idx], zd["v_t2_sigma_effort"]))
        ir = np.searchsorted(grid, dd)
        if not np.array_equal(grid[ir], dd):
            raise RuntimeError(f"{rd}: development D2 grid is not a subset of the recovery grid")
        upd("recovery_vs_ehat", C.max_abs_diff(e2[ir], zd["v_t2_e_hat"]))
        j = int(np.argmax(e2))  # first maximum, as tools/v2/pilot4_common.py:location_free
        rows.append({"q": q, "seed": seed, "arm": arm,
                     "stage2_peak_locfree_rel_err": (float(e2[j]) - g20) / g20,
                     "stage2_peak_locfree_argmax_d": float(grid[j]), "stage2_max_e2": float(e2[j])})
    lf = pd.DataFrame(rows)
    fin = S.final_tier_columns("pilot1").drop(columns=["run_dir"])
    return lf.merge(fin, on=["q", "seed", "arm"], how="inner", validate="one_to_one"), chk


# ----------------------------------------------------------------------------------------------
# T19: design summary of Pilots 1-3
# ----------------------------------------------------------------------------------------------

PILOT_TEXT: Dict[str, Dict[str, str]] = {
    "pilot1": dict(
        name="Pilot 1",
        question=("Terminal reward estimator in Phase A (stage 1 untrained): reward_mode sampled "
                  "vs expected (conditional expected terminal reward), paired by seed (same "
                  "SeedSequence streams and initial network). 0930 plan: only the estimator "
                  "changes; compare peak error, RMSE, tail effort and eta_2."),
        question_source=("reports/v2/pilot1_reward_estimator.md (title; section 1 'The pilot', "
                         "Design); PI 0930 plan (request section 5.2)"),
        decision=("Reward estimator = expected. The Pilot 1 report itself does not choose ('The "
                  "estimator is not chosen here.')."),
        decision_source=("reports/v2/summary.md, 'Decisions taken so far' ('reward estimator = "
                         "expected (after Pilot 1)'); reports/v2/pilot2_freeze.md header ('Reward: "
                         "expected in all 60 runs')"),
    ),
    "pilot2": dict(
        name="Pilot 2",
        question=("Joint vs frozen stage 2 during Phase B, every arm restoring the same stage-2 "
                  "parent: A joint; B1 frozen with advantage normalization over all rows; B2 "
                  "frozen with normalization over the stage-1 rows. 0930 plan: compare stage-2 "
                  "drift, stage-1 recovery, Gmax_full, dReach and EXP_root, and the frozen stage's "
                  "output drift."),
        question_source=("reports/v2/pilot2_freeze.md (title; arms line; section 1.3 Design and "
                         "launch); PI 0930 plan (request section 5.2)"),
        decision=("Frozen variant = B2 (stage2_update_mode = frozen, adv_norm_scope = stage1_rows) "
                  "for Pilot 3; induced stage-1 target e~1 by residual minimization on the final "
                  "tier, with a band (decision D2), replacing the bracketing + Brent solver."),
        decision_source=("reports/v2/pilot2_freeze.md, end of section 5 ('Pilot 2 closed. Your "
                         "decisions were B2 for Pilot 3 and residual minimization for e~1 (section "
                         "6)') and section 6; reports/v2/summary.md, 'Decisions taken so far'"),
    ),
    "pilot3": dict(
        name="Pilot 3",
        question=("Continuation action mode in Phase B with stage 2 frozen (B2): stochastic "
                  "(stage-2 actions sampled from the frozen Beta) vs mean (both players play the "
                  "frozen Beta mean; the stage-2 draws are still made and discarded, A6). 0930 "
                  "plan: judged by stage-1 accuracy and stability."),
        question_source=("reports/v2/pilot3_continuation_mode.md (title; sections 0 and 1, "
                         "Design); PI 0930 plan (request section 5.2)"),
        decision="Continuation action mode = mean (decision D1, 2026-10-02, before Pilot 4).",
        decision_source=("reports/v2/summary.md, 'Decisions taken so far' ('(2026-10-02, before "
                         "Pilot 4) D1 continuation mode = mean'); "
                         "reports/v2/pilot4_stabilization.md header ('Decisions applied: D1 "
                         "continuation mean')"),
    ),
}

T19_DOCS: Dict[str, Any] = {
    "pilot": "Pilot name", "study": "Study key in tools/v2/report/studies.py",
    "report": "Report of the pilot (repo-relative path)",
    "question": ("Question of the pilot (source: report text; the 0930-plan comparison list is "
                 "PI-supplied, request section 5.2)"),
    "question_source": "Report sections (and plan) the question text was compiled from",
    "phase_budget": ("Training phase, number of updates, global update range and fixed_budget flag "
                     "(from v2_run_summary.json phase_timing and manifest.json fixed_budget; "
                     "identical in every run)"),
    "final_checkpoint": "Global update of the last checkpoint (status.json final_global_update)",
    "parents": "Parent states the runs restore (manifest.json parent_checkpoint)",
    "parent_check": ("Manifests whose parent_sha256 equals the SHA-256 of the parent file read "
                     "here; and the verdict of tools/v2/verify_parents.py "
                     "(pilot2_parents_check.json)"),
    "arms": "Arm labels (run directory names)",
    "arm_flags": ("Values of the flags that differ between the arms, per arm (manifest.json flags; "
                  "identical in every run of an arm)"),
    "varied_flags": "Flags that differ between the arms",
    "common_flags": "Flags with the same value in every arm",
    "q_values": "q values of the study",
    "seeds": "Seeds of the study (every (q, arm) uses all of them)",
    "design": "Number of q values x seeds x arms",
    "runs": "Runs with status done, exit code 0 and the expected final update (status.json)",
    "runs_planned": "Runs planned by the launcher (launch_*.json planned)",
    "runs_returncode0": "Launcher runs with returncode 0 (launch_*.json runs)",
    "launch_commit": "Code commit of every run (manifest.json git.short; identical in every run)",
    "launch_commit_full": "Full hash of the launch commit (manifest.json git.commit)",
    "dirty_runs": "Runs whose manifest has git.dirty = true",
    "launch_workers": ("Parallel worker processes of the launcher (launch_*.json workers); one "
                       "single-threaded process per run"),
    "wall_launcher_s_min": dict(definition=("Minimum over runs of the launcher wall time per run "
                                            "(launch_*.json runs[*].wall_sec; includes process "
                                            "start-up)"), units="seconds"),
    "wall_launcher_s_median": dict(definition="Median over runs of the launcher wall time per run",
                                   units="seconds"),
    "wall_launcher_s_max": dict(definition="Maximum over runs of the launcher wall time per run",
                                units="seconds"),
    "wall_total_s_min": dict(definition=("Minimum over runs of the run's own total wall time "
                                         "(status.json total_wall_sec)"), units="seconds"),
    "wall_total_s_median": dict(definition="Median over runs of status.json total_wall_sec",
                                units="seconds"),
    "wall_total_s_max": dict(definition="Maximum over runs of status.json total_wall_sec",
                             units="seconds"),
    "wall_phase_s_min": dict(definition=("Minimum over runs of the trained phase's wall time "
                                         "(v2_run_summary.json phase_timing.<phase>.wall_sec)"),
                             units="seconds"),
    "wall_phase_s_median": dict(definition="Median over runs of the phase wall time",
                                units="seconds"),
    "wall_phase_s_max": dict(definition="Maximum over runs of the phase wall time",
                             units="seconds"),
    "decision": "Decision taken after the pilot (source: report text)",
    "decision_source": "Report sections that record the decision",
    "decision_applied": ("Run records showing the decision applied in the next study (counted from "
                         "manifests)"),
}


def _study_runs(study: str) -> pd.DataFrame:
    """One row per planned run of a pilot study: manifest, status and phase-timing fields."""
    s = S.STUDIES[study]
    rows = []
    for q, seed, arm, rd in S.iter_runs(study):
        man, st = _manifest(rd), S.status(rd)
        pt = S.read_json(f"{rd}/v2_run_summary.json")["phase_timing"][s["phase"]]
        rows.append({"q": q, "seed": seed, "arm": arm, "run_dir": rd, "commit": man["git"]["short"],
                     "commit_full": man["git"]["commit"], "dirty": bool(man["git"]["dirty"]),
                     "flags": json.dumps(man["flags"], sort_keys=True),
                     "fixed_budget": bool(man["fixed_budget"]),
                     "parent_checkpoint": man.get("parent_checkpoint"),
                     "parent_sha256": man.get("parent_sha256"), "state": st["state"],
                     "exit_code": st["exit_code"], "final_u": int(st["final_global_update"]),
                     "wall_total": float(st["total_wall_sec"]), "wall_phase": float(pt["wall_sec"]),
                     "ph_updates": int(pt["updates"]), "ph_entry": int(pt["global_entry"]),
                     "ph_exit": int(pt["global_exit"])})
    return pd.DataFrame(rows)


def _launch(study: str) -> Tuple[str, Dict[str, Any]]:
    """The single ``launch_*.json`` of a pilot root: (repo-relative path, content)."""
    files = sorted(C.abspath(S.STUDIES[study]["root"]).glob("launch_*.json"))
    if len(files) != 1:
        raise RuntimeError(f"{study}: expected one launch record, found {len(files)}")
    return C.relpath(files[0]), json.loads(files[0].read_text(encoding="utf-8"))


def _launcher_walls(study: str, runs: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Launcher wall seconds per run (``launch_*.json runs[*].wall_sec``), matched 1:1 to runs."""
    _, L = _launch(study)
    by_key = {"/".join(Path(r["out"]).parts[-3:]): r for r in L["runs"]}
    keys = [f"q{q}/seed{s}/{a}" for q, s, a in zip(runs.q, runs.seed, runs.arm)]
    if sorted(by_key) != sorted(keys):
        raise RuntimeError(f"{study}: launcher runs do not match the run directories")
    walls = np.array([float(by_key[k]["wall_sec"]) for k in keys])
    rcs = [int(by_key[k]["returncode"]) for k in keys]
    return walls, {"n_rc0": sum(1 for r in rcs if r == 0), "workers": int(L["workers"]),
                   "planned": len(L["planned"])}


def _flag_text(runs: pd.DataFrame, arms: Sequence[str]) -> Tuple[str, str, str]:
    """(per-arm values of the varied flags, varied flag names, common flags) from the manifests."""
    per_arm: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        fl = runs.loc[runs.arm == arm, "flags"].unique()
        if len(fl) != 1:
            raise RuntimeError(f"arm {arm}: flags differ between runs: {fl}")
        per_arm[arm] = json.loads(fl[0])
    keys = sorted(per_arm[arms[0]])
    varied = [k for k in keys if len({json.dumps(per_arm[a][k]) for a in arms}) > 1]
    common = [k for k in keys if k not in varied]
    arm_txt = "; ".join(f"{a}: " + ", ".join(f"{k}={per_arm[a][k]}" for k in varied) for a in arms)
    common_txt = ", ".join(f"{k}={per_arm[arms[0]][k]}" for k in common)
    return arm_txt, ", ".join(varied), common_txt


def _flag_count(study: str, cond: Dict[str, str]) -> Tuple[int, int]:
    """(runs whose manifest flags match ``cond``, runs) of a study."""
    n = hit = 0
    for _q, _s, _a, rd in S.iter_runs(study):
        fl = _manifest(rd)["flags"]
        n += 1
        hit += int(all(fl.get(k) == v for k, v in cond.items()))
    return hit, n


def _t19_row(study: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """One T19 row (and the per-run records it was computed from)."""
    s, txt = S.STUDIES[study], PILOT_TEXT[study]
    runs = _study_runs(study)
    lw, linfo = _launcher_walls(study, runs)
    ok = (runs.state == "done") & (runs.exit_code == 0) & (runs.final_u == s["final_update"])
    if runs.commit.nunique() != 1 or runs.ph_updates.nunique() != 1 or runs.ph_entry.nunique() != 1:
        raise RuntimeError(f"{study}: launch commit or phase timing differs between runs")
    upd, entry, exit_ = (int(runs[c].iloc[0]) for c in ("ph_updates", "ph_entry", "ph_exit"))
    fb = "true" if runs.fixed_budget.all() else "mixed"
    if study == "pilot1":
        n_null = int(runs.parent_checkpoint.isna().sum())
        parents = (f"none: parent_checkpoint = null in {n_null}/{len(runs)} manifests (from "
                   "initialization)")
        pcheck = "not applicable (no parent states)"
    else:
        n_ok = 0
        for q, seed, sha in zip(runs.q, runs.seed, runs.parent_sha256):
            p = f"{P1}/q{q}/seed{seed}/expected/state_end_A.pt"
            n_ok += int(C.sha256_file(C.abspath(p)) == sha)
        chk = S.read_json("results/v2_pilots/pilot2_parents_check.json")
        parents = (f"the 20 Pilot 1 expected end-of-Phase-A full states "
                   f"{P1}/q<q>/seed<seed>/expected/state_end_A.pt (global u400), one per (q, "
                   f"seed), restored by each of the {len(s['arms'])} arms")
        pcheck = (f"parent_sha256 equals the SHA-256 of the parent file in {n_ok}/{len(runs)} "
                  f"manifests; pilot2_parents_check.json: all_ok = {chk['all_ok']}, n = {chk['n']}")
    arm_txt, varied, common = _flag_text(runs, s["arms"])
    if study == "pilot1":
        h2, n2 = _flag_count("pilot2", {"reward_mode": "expected"})
        h3, n3 = _flag_count("pilot3", {"reward_mode": "expected"})
        applied = f"reward_mode = expected in {h2}/{n2} Pilot 2 and {h3}/{n3} Pilot 3 manifests"
    elif study == "pilot2":
        cond = {"stage2_update_mode": "frozen", "adv_norm_scope": "stage1_rows"}
        h, n = _flag_count("pilot3", cond)
        applied = (f"stage2_update_mode = frozen and adv_norm_scope = stage1_rows in {h}/{n} "
                   "Pilot 3 manifests; e~1 of Pilot 3 from the parent residual bands "
                   "(results/v2_pilots/induced_band/parent_bands.csv; source: report text)")
    else:
        h, n = _flag_count("pilot4_B", {"continuation_action_mode": "mean"})
        applied = (f"continuation_action_mode = mean in {h}/{n} Pilot 4 section 2b manifests "
                   "(results/v2_pilots/pilot4_B)")
    row = {
        "pilot": txt["name"], "study": study, "report": s["report"], "question": txt["question"],
        "question_source": txt["question_source"],
        "phase_budget": (f"Phase {s['phase']}: {upd} updates, global u{entry + 1}-u{exit_} "
                         f"(fixed_budget = {fb})"),
        "final_checkpoint": f"u{int(runs.final_u.max())}", "parents": parents,
        "parent_check": pcheck,
        "arms": ", ".join(s["arms"]), "arm_flags": arm_txt, "varied_flags": varied,
        "common_flags": common, "q_values": ", ".join(str(q) for q in s["qs"]),
        "seeds": _seeds_text(s["seeds"]),
        "design": f"{len(s['qs'])} q x {len(s['seeds'])} seeds x {len(s['arms'])} arms",
        "runs": int(ok.sum()), "runs_planned": linfo["planned"], "runs_returncode0": linfo["n_rc0"],
        "launch_commit": runs.commit.iloc[0], "launch_commit_full": runs.commit_full.iloc[0],
        "dirty_runs": int(runs.dirty.sum()), "launch_workers": linfo["workers"],
        "wall_launcher_s_min": float(lw.min()), "wall_launcher_s_median": float(np.median(lw)),
        "wall_launcher_s_max": float(lw.max()),
        "wall_total_s_min": float(runs.wall_total.min()),
        "wall_total_s_median": float(runs.wall_total.median()),
        "wall_total_s_max": float(runs.wall_total.max()),
        "wall_phase_s_min": float(runs.wall_phase.min()),
        "wall_phase_s_median": float(runs.wall_phase.median()),
        "wall_phase_s_max": float(runs.wall_phase.max()),
        "decision": txt["decision"], "decision_source": txt["decision_source"],
        "decision_applied": applied,
    }
    return row, runs


def _t19_checks(pack: C.Pack, df: pd.DataFrame, runs_by: Dict[str, pd.DataFrame]) -> None:
    """Cross-check T19 against the report prose, report headers and per-arm wall tables."""
    r1, r2, r3 = df.iloc[0], df.iloc[1], df.iloc[2]
    _prose(pack, "T19", REP1, "Pilot 1 launcher and Phase A wall per run (min / median / max), "
           "runs, workers", [
               ("Pilot 1 launcher wall min [s]", r1.wall_launcher_s_min, "47.5"),
               ("Pilot 1 launcher wall median [s]", r1.wall_launcher_s_median, "50.6"),
               ("Pilot 1 launcher wall max [s]", r1.wall_launcher_s_max, "68.1"),
               ("Pilot 1 Phase A wall min [s]", r1.wall_phase_s_min, "41.34"),
               ("Pilot 1 Phase A wall median [s]", r1.wall_phase_s_median, "45.03"),
               ("Pilot 1 Phase A wall max [s]", r1.wall_phase_s_max, "62.72"),
               ("Pilot 1 runs with returncode 0", r1.runs_returncode0, "40"),
               ("Pilot 1 workers", r1.launch_workers, "40")])
    _prose(pack, "T19", REP2, "Pilot 2 launcher wall range per run, runs, workers", [
        ("Pilot 2 launcher wall min [s]", r2.wall_launcher_s_min, "189.1"),
        ("Pilot 2 launcher wall max [s]", r2.wall_launcher_s_max, "215.4"),
        ("Pilot 2 runs with returncode 0", r2.runs_returncode0, "60"),
        ("Pilot 2 workers", r2.launch_workers, "60")])
    _prose(pack, "T19", REP3, "Pilot 3 launcher wall range per run, runs, workers", [
        ("Pilot 3 launcher wall min [s]", r3.wall_launcher_s_min, "193.3"),
        ("Pilot 3 launcher wall max [s]", r3.wall_launcher_s_max, "199.9"),
        ("Pilot 3 runs with returncode 0", r3.runs_returncode0, "40"),
        ("Pilot 3 workers", r3.launch_workers, "40")])
    hdr = {1: (REP1, "Code commit for every Pilot 1 run"),
           2: (REP2, "Launch commit for all 60 runs"),
           3: (REP3, "Launch commit for all 40 runs")}
    for k, r in zip((1, 2, 3), (r1, r2, r3)):
        rep, key = hdr[k]
        cell = next((row[1] for t in C.parse_md_tables(rep) if t["header"] == ["Item", "Value"]
                     for row in t["rows"] if row and row[0].replace("*", "").strip() == key), "")
        _claims(pack, "T19", rep, f"Pilot {k} header: launch commit and dirty flag of every run", [
            (f"Pilot {k} launch commit", r.launch_commit, cell, r.launch_commit in cell),
            (f"Pilot {k} runs with dirty = true", str(r.dirty_runs), cell,
             r.dirty_runs == 0 and "dirty = false" in cell)])
    inv = {row[0]: row for t in C.parse_md_tables(SUMMARY)
           if t["header"][:4] == ["Study", "Report", "Launch commit", "Runs"] for row in t["rows"]}
    sclaims = []
    for k, r in zip((1, 2, 3), (r1, r2, r3)):
        row = next((v for kk, v in inv.items() if kk.startswith(f"Pilot {k}:")), None)
        sclaims.append((f"Pilot {k} launch commit (study table)", r.launch_commit,
                        row[2] if row else "", bool(row) and row[2].split()[0] == r.launch_commit))
        sclaims.append((f"Pilot {k} runs (study table)", str(r.runs), row[3] if row else "",
                        bool(row) and C.consistent(float(r.runs), row[3])))
    _claims(pack, "T19", SUMMARY, "study table: launch commit and runs of Pilots 1-3", sclaims)
    p2 = runs_by["pilot2"]
    w2 = (p2.assign(arm_s=p2.arm.map(S.ARM_SHORT)).groupby(["q", "arm_s"]).wall_phase
          .agg(["min", "median", "max"]).reset_index())
    pack.crosscheck("T19", w2, REP2, header_has=["q", "arm", "wall_min", "wall_median", "wall_max"],
                    key_map={"q": "q", "arm": "arm_s"},
                    value_map={"wall_min": "min", "wall_median": "median", "wall_max": "max"},
                    label="Pilot 2 Phase B wall per (q, arm) from v2_run_summary.json "
                          "(section 2.7)")
    p3 = runs_by["pilot3"]
    short3 = {"B2_frozen_s1norm": "stochastic", "B2_frozen_s1norm_mean": "mean"}
    w3 = (p3.assign(arm_s=p3.arm.map(short3)).groupby(["q", "arm_s"]).wall_phase.median()
          .reset_index())
    pack.crosscheck("T19", w3, REP3, header_has=["q", "arm", "wall_median", "drift_test_pass"],
                    key_map={"q": "q", "arm": "arm_s"}, value_map={"wall_median": "wall_phase"},
                    label="Pilot 3 Phase B wall median per (q, arm) from v2_run_summary.json "
                          "(section 7)")


def build_t19(pack: C.Pack) -> pd.DataFrame:
    """T19: design summary of Pilots 1-3 (one row per pilot)."""
    script = f"{MOD}:build_t19"
    rows, runs_by = [], {}
    for study in ("pilot1", "pilot2", "pilot3"):
        row, runs_by[study] = _t19_row(study)
        rows.append(row)
    df = pd.DataFrame(rows)
    _t19_checks(pack, df, runs_by)

    sources: List[Any] = []
    for study in ("pilot1", "pilot2", "pilot3"):
        root = S.STUDIES[study]["root"]
        n = len(S.STUDIES[study]["arms"]) * 20
        for name in ("manifest.json", "status.json", "v2_run_summary.json"):
            sources.append(C.srcs(f"{root}/q*/seed*/*/{name}", label=f"{study} per-run {name}",
                                  expect=n))
        sources.append(C.src(_launch(study)[0]))
    sources += [C.srcs(f"{P1}/q*/seed*/expected/state_end_A.pt",
                       label="Pilot 1 expected parents (SHA-256 check)", expect=20),
                C.src("results/v2_pilots/pilot2_parents_check.json"),
                C.srcs("results/v2_pilots/pilot4_B/q*/seed*/*/manifest.json",
                       label="pilot4_B manifests (flags)", expect=40)]
    sources += [C.src(r, selector="question/decision text (source: report text); cross-check")
                for r in (REP1, REP2, REP3, REP4, SUMMARY)]
    notes = ("One row per pilot. Counts, phase/budget, parents (with SHA-256 check of the 20 "
             "parent files), arm flags, launch commit, dirty flags, workers and wall times are "
             "computed from the run records (manifest.json, status.json, v2_run_summary.json, "
             "launch_*.json; wall = min/median/max over the runs of the pilot). Question and "
             "decision: source: report text (reports/v2/pilot1/2/3 reports, "
             "pilot4_stabilization.md header, summary.md), plus the PI's 0930 plan for the "
             "comparison list. Report prose checked: launcher/phase walls, run counts, workers; "
             "per-arm phase wall tables of Pilot 2 (section 2.7) and Pilot 3 (section 7).")
    pack.table("T19", df, status="generated", sources=sources, script=script, notes=notes,
               docs=T19_DOCS, tier="n/a",
               caption="Design of Pilots 1-3 as recorded in the run records; text columns compiled "
                       "from the reports.")
    return df


# ----------------------------------------------------------------------------------------------
# T20: Pilot 1 final medians and IQR
# ----------------------------------------------------------------------------------------------

# metric, label, units, tier-dependent
T20_METRICS: List[Tuple[str, str, str, bool]] = [
    ("stage2_peak_rel_err_signed", "peak error at d=0, signed: (e2hat(0) - e2*(0))/e2*(0)",
     "fraction of e2*(0)", False),
    ("stage2_peak_rel_err_abs", "peak error at d=0, absolute", "fraction of e2*(0)", False),
    ("stage2_peak_locfree_rel_err", "location-free peak error (max_d e2hat(d) - e2*(0))/e2*(0), "
     "signed (derived from the saved recovery arrays)", "fraction of e2*(0)", False),
    ("stage2_peak_locfree_argmax_d", "gap d of max_d e2hat(d) on the recovery grid (location of "
     "the location-free peak)", "effort units (gap d)", False),
    ("stage2_rmse_pos", "RMSE of e2hat - e2* over |d| < 2q, raw", "effort units", False),
    ("stage2_rmse_pos_over_g2_0", "RMSE of e2hat - e2* over |d| < 2q / e2*(0)",
     "fraction of e2*(0)",
     False),
    ("stage2_tail_mean", "tail mean of e2hat over |d| >= 2q, raw", "effort units [0, 100]", False),
    ("stage2_tail_mean_over_g2_0", "tail mean / e2*(0)", "fraction of e2*(0)", False),
    ("stage2_tail_max", "tail max of e2hat over |d| >= 2q, raw", "effort units [0, 100]", False),
    ("stage2_tail_max_over_g2_0", "tail max / e2*(0)", "fraction of e2*(0)", False),
    ("stage2_sym_err_max", "symmetry error max_d |e2hat(d) - e2hat(-d)|, raw", "effort units",
     False),
    ("stage2_sym_err_max_over_g2_0", "symmetry error / e2*(0)", "fraction of e2*(0)", False),
    ("eta_T_over_dw", "eta_2 = max over D2 of Delta_2(d)", "Delta W", True),
    ("DeltaT_over_dw_on_max", "Delta_2 on path (|d - drift| < 2q), max", "Delta W", True),
    ("DeltaT_over_dw_on_mean_cellmass_weighted", "Delta_2 on path, exact cell-mass weighted mean",
     "Delta W", True),
    ("DeltaT_over_dw_off_max", "Delta_2 off path, max", "Delta W", True),
    ("sigma_effort_at_0_t2", "sigma_2(0): SD of the stage-2 Beta action at d = 0",
     "effort units [0, 100]", False),
]
SRC_CK = "final_table.csv (u400 training-time checkpoint)"


def _t20_long(ft: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    """Per-run long frame (q, seed, arm, metric, tier, value, source) of the T20 metrics."""
    m = ft.merge(rf, on=["q", "seed", "arm"], how="inner", validate="one_to_one")
    if len(m) != 40:
        raise RuntimeError(f"T20: expected 40 runs, found {len(m)}")
    parts = []
    for metric, _label, _units, tdep in T20_METRICS:
        base = m[["q", "seed", "arm"]].copy()
        if metric.startswith("stage2_peak_locfree"):
            src = "final_development.npz (recovery_d_grid, recovery_e2) + final_v2.json g2_at_0"
            parts.append(base.assign(metric=metric, tier=TI, value=m[metric].astype(float),
                                     source=src))
        elif tdep:
            parts.append(base.assign(metric=metric, tier=DEV, value=m[metric].astype(float),
                                     source=SRC_CK))
            parts.append(base.assign(metric=metric, tier=FIN,
                                     value=m[f"final_tier__{metric}"].astype(float),
                                     source="final_v2.json['final']"))
        else:
            parts.append(base.assign(metric=metric, tier=TI, value=m[metric].astype(float),
                                     source=SRC_CK))
    return pd.concat(parts, ignore_index=True)


def _summary(long: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Median, q25, q75, min, max and n of ``value`` per group (numpy linear interpolation)."""
    rows = []
    for key, g in long.groupby(list(by), sort=False):
        rows.append({**dict(zip(by, key)), **C.median_iqr(g["value"].to_numpy(dtype=float))})
    return pd.DataFrame(rows)


def _tail_argmax_text(rf: pd.DataFrame) -> str:
    """Factual note on runs whose location-free maximum lies in the tail region |d| >= 2q."""
    tail = rf[rf.stage2_peak_locfree_argmax_d.abs() >= 2 * rf.q].sort_values(["q", "seed", "arm"])
    if tail.empty:
        return "No run has its location-free maximum in the tail region |d| >= 2q."
    items = "; ".join(f"q={r.q} seed {r.seed} {r.arm}: argmax d = "
                      f"{r.stage2_peak_locfree_argmax_d:g}, max e2hat = {r.stage2_max_e2:.4g}"
                      for r in tail.itertuples())
    return (f"Runs whose location-free maximum lies in the tail region |d| >= 2q (there the "
            f"location-free error is a tail value, not a peak value): {items}.")


def _t20_checks(ft: pd.DataFrame) -> Dict[str, float]:
    """Max abs diffs: dev-tier values of final_table.csv vs final_v2.json['development'], and
    tier-independent values of final_table.csv vs final_v2.json['final']."""
    dev_d = fin_d = 0.0
    for _, r in ft.iterrows():
        fv = _final_v2(str(r.run_dir))
        for metric, _l, _u, tdep in T20_METRICS:
            if metric.startswith("stage2_peak_locfree"):
                continue
            dev_d = max(dev_d, abs(float(r[metric]) - float(fv["development"][metric])))
            if not tdep:
                fin_d = max(fin_d, abs(float(r[metric]) - float(fv["final"][metric])))
    return {"dev_csv_vs_json": dev_d, "ti_csv_vs_final_json": fin_d}


def _t20_crosschecks(pack: C.Pack, summ: pd.DataFrame) -> None:
    """T20 against summary.md (Pilot 1 medians) and the pilot 1 report prose (section 5)."""
    wide = (summ[summ.tier.isin([TI, DEV])]
            .pivot_table(index=["q", "arm"], columns="metric", values="median").reset_index())
    cols = ["stage2_peak_rel_err_signed", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean",
            "eta_T_over_dw", "sigma_effort_at_0_t2"]
    pack.crosscheck("T20", wide, SUMMARY, header_has=["q", "arm"] + cols,
                    key_map={"q": "q", "arm": "arm"}, value_map={c: c for c in cols},
                    heading_has="Pilot 1", label="Pilot 1 final medians (eta_2 development tier)")
    med = summ.set_index(["q", "arm", "metric", "tier"])["median"]
    pk, tm = "stage2_peak_rel_err_signed", "stage2_tail_mean"
    _prose(pack, "T20", REP1, "section 5: median signed peak error and median tail mean per arm "
           "and q", [
               ("median signed peak error, q=50 sampled", med[(50, "sampled", pk, TI)], "-0.40"),
               ("median signed peak error, q=60 sampled", med[(60, "sampled", pk, TI)], "-0.47"),
               ("median signed peak error, q=50 expected", med[(50, "expected", pk, TI)], "-0.13"),
               ("median signed peak error, q=60 expected", med[(60, "expected", pk, TI)], "-0.10"),
               ("median tail mean, q=50 sampled", med[(50, "sampled", tm, TI)], "7.4"),
               ("median tail mean, q=60 sampled", med[(60, "sampled", tm, TI)], "10.0"),
               ("median tail mean, q=50 expected", med[(50, "expected", tm, TI)], "1.5"),
               ("median tail mean, q=60 expected", med[(60, "expected", tm, TI)], "1.3")])


def build_t20(pack: C.Pack, ft: pd.DataFrame, rf: pd.DataFrame,
              chk: Dict[str, float]) -> pd.DataFrame:
    """T20: Pilot 1 final (u400) medians and IQR per arm and q, both tiers where tier-dependent."""
    script = f"{MOD}:build_t20"
    if not ((ft["update"] == 400).all() and (ft["n_verifier_checkpoints"] == 4).all()):
        raise RuntimeError("final_table.csv: not every row is the u400 checkpoint of a "
                           "4-checkpoint run")
    summ = _summary(_t20_long(ft, rf), ["q", "arm", "metric", "tier", "source"])
    order_m = {m[0]: i for i, m in enumerate(T20_METRICS)}
    order_t = {TI: 0, DEV: 1, FIN: 2}
    summ = summ.assign(_q=summ.q, _a=summ.arm.map({a: i for i, a in enumerate(ARMS)}),
                       _m=summ.metric.map(order_m), _t=summ.tier.map(order_t))
    summ = (summ.sort_values(["_q", "_a", "_m", "_t"]).drop(columns=["_q", "_a", "_m", "_t"])
            .reset_index(drop=True))
    lab = {m: (lb, u) for m, lb, u, _ in T20_METRICS}
    summ.insert(3, "label", summ.metric.map(lambda x: lab[x][0]))
    summ.insert(4, "units", summ.metric.map(lambda x: lab[x][1]))
    summ["seeds"] = _seeds_text(S.SEEDS_DEV)
    summ = summ[["q", "arm", "metric", "label", "units", "tier", "median", "q25", "q75", "min",
                 "max", "n", "seeds", "source"]]
    tchk = _t20_checks(ft)
    # independent record of the expected arm at u400 (the Phase A extension starts from these runs)
    ext = pd.read_csv(C.abspath(EXT_TABLE), float_precision="round_trip")
    ext = ext[ext["update"] == 400]
    mx = summ[(summ.arm == "expected") & summ.tier.isin([TI, DEV])].merge(
        ext, on=["q", "metric"], suffixes=("", "_ext"), validate="one_to_one")
    d_ext = max(C.max_abs_diff(mx[c], mx[f"{c}_ext"])
                for c in ("median", "q25", "q75", "min", "max", "n"))
    _t20_crosschecks(pack, summ)

    notes = (f"Across-seed median, q25, q75 (numpy linear interpolation), min, max over the 10 "
             f"seeds per (q, arm) of the u400 values. Tiers: {TIER_DEF}. The existing "
             f"final_table.csv is the development-tier (training-time) checkpoint; final-tier "
             f"values exist only for the last checkpoint (final_v2.json['final']). Location-free "
             f"peak error was not recorded in Pilot 1: derived here from final_development.npz "
             f"recovery_d_grid/recovery_e2 with e2*(0) = final_v2.json g2_at_0 "
             f"(tools/v2/pilot4_common.py:location_free rule, first argmax). Checks: recovery "
             f"arrays of final_development.npz and final_final.npz identical in 40/40 runs (max "
             f"abs diff {chk['npz_dev_vs_final']:.3g}); g2_at_0 vs recovery_g2 at d=0 max abs diff "
             f"{chk['g20_json_vs_grid']:.3g}; development values of final_table.csv vs "
             f"final_v2.json['development'] max abs diff {tchk['dev_csv_vs_json']:.3g}; "
             f"tier-independent values of final_table.csv vs final_v2.json['final'] max abs diff "
             f"{tchk['ti_csv_vs_final_json']:.3g}; the {len(mx)} expected-arm rows shared with "
             f"phaseA_ext/analysis/table_400_800_1200_1600.csv (update 400, the same runs) agree "
             f"(max abs diff over median, q25, q75, min, max, n: {d_ext:.3g}). "
             f"{_tail_argmax_text(rf)} stage1_status = stage1_untrained in all 40 runs (no stage-1 "
             f"metric is shown here).")
    docs = {
        "label": "Human-readable definition of the metric",
        "units": "Units of the metric value (raw effort, fraction of e2*(0), Delta W or gap d)",
        "tier": dict(definition="Tier of the value: " + TIER_DEF),
        "seeds": "Seeds summarized in the row",
        "source": "File the per-run values were read from",
        "median": "Median over the 10 seeds of the per-run u400 values",
    }
    srcs = _p1_sources()
    sources = [C.src(FINAL_TABLE), srcs["final_v2.json"], srcs["final_development.npz"],
               srcs["final_final.npz"], srcs["manifest.json"],
               C.src("utils/theory_multistage.py", selector="g2_two_stage check"),
               C.src(EXT_TABLE, selector="update 400 rows (check)"),
               C.src(SUMMARY, selector="cross-check"), C.src(REP1, selector="cross-check")]
    pack.table("T20", summ, status="generated", sources=sources, script=script, notes=notes,
               docs=docs, tier="final and development",
               caption="Pilot 1 at u400 (Phase A end), per q and arm, n = 10 seeds (10501-10510). "
                       "Tier-dependent metrics appear twice (development, final).")
    return summ


# ----------------------------------------------------------------------------------------------
# T21: paired differences expected - sampled
# ----------------------------------------------------------------------------------------------

T21_FINAL = ["eta_T_over_dw", "DeltaT_over_dw_on_max", "DeltaT_over_dw_on_mean_cellmass_weighted",
             "DeltaT_over_dw_off_max", "sigma2_effort_mean_pos", "EXP_root_over_dw",
             "dReach_over_dw",
             "Gmax_full_over_dw"]
LOWER = "expected favoured if diff < 0"


def _pair_diff(df: pd.DataFrame, q: int, col: str) -> np.ndarray:
    """expected - sampled per seed (sorted seeds) for one q and one per-run column."""
    e = df[(df.q == q) & (df.arm == "expected")].set_index("seed")[col]
    s = df[(df.q == q) & (df.arm == "sampled")].set_index("seed")[col]
    seeds = sorted(set(e.index) & set(s.index))
    if len(seeds) != 10:
        raise RuntimeError(f"paired difference q={q} {col}: {len(seeds)} pairs")
    return np.array([float(e[x]) - float(s[x]) for x in seeds])


def _tier_of(metric: str) -> str:
    """Tier label of a final_table.csv column (development when tier-dependent)."""
    ent = D.E.get(metric)
    if ent is None or ent[3] is None:
        return NA_TIER
    return DEV if ent[3] else TI


def _paired_row(dv: np.ndarray, q: int, metric: str, label: str, lower_better: bool, tier: str,
                source: str) -> Dict[str, Any]:
    """One paired-summary row (statistics of paired_summary.csv; bootstrap, fresh generator)."""
    lo, hi = C.bootstrap_mean_ci(dv, N_BOOT, BOOT_SEED)
    return {"q": q, "metric": metric, "label": label, "n_pairs": int(dv.size),
            "mean": float(dv.mean()),
            "sd": float(dv.std(ddof=1)), "median": float(np.median(dv)), "min": float(dv.min()),
            "max": float(dv.max()),
            "favour_rule": LOWER if lower_better else "no preferred direction",
            "n_favour_expected": float((dv < 0).sum()) if lower_better else np.nan,
            "n_diff_negative": int((dv < 0).sum()), "n_diff_positive": int((dv > 0).sum()),
            "n_diff_zero": int((dv == 0).sum()), "boot_ci95_lo": lo, "boot_ci95_hi": hi,
            "tier": tier,
            "source": source}


def _t21_reproduce(ps: pd.DataFrame, ft: pd.DataFrame) -> float:
    """Recompute paired_summary.csv from final_table.csv with the original generator sequence.

    ``tools/v2/pilot1_analysis.py:paired`` used one ``default_rng(20261001)`` over the rows in file
    order. Returns the max abs diff over the statistics.
    """
    rng = np.random.default_rng(BOOT_SEED)
    rec = []
    for _, r in ps.iterrows():
        dv = _pair_diff(ft, int(r.q), str(r.metric))
        bm = dv[rng.integers(0, dv.size, size=(N_BOOT, dv.size))].mean(axis=1)
        rec.append([dv.mean(), dv.std(ddof=1), np.median(dv), dv.min(), dv.max(), (dv < 0).sum(),
                    (dv > 0).sum(), (dv == 0).sum(), np.percentile(bm, 2.5),
                    np.percentile(bm, 97.5)])
    cols = ["mean", "sd", "median", "min", "max", "n_diff_negative", "n_diff_positive",
            "n_diff_zero", "boot_ci95_lo", "boot_ci95_hi"]
    arr = np.array(rec)
    return max(C.max_abs_diff(arr[:, i], ps[c].to_numpy(dtype=float)) for i, c in enumerate(cols))


T21_DOCS: Dict[str, Any] = {
    "mean": "Mean over the 10 seeds of the paired difference expected - sampled (same q and seed)",
    "sd": "Sample SD (ddof = 1) of the 10 paired differences",
    "median": "Median of the 10 paired differences expected - sampled",
    "min": "Minimum of the 10 paired differences", "max": "Maximum of the 10 paired differences",
    "label": ("Human-readable name of the metric (as in paired_summary.csv; new rows: the "
              "location-free peak error, and '[final tier]' for the final-tier rows)"),
    "favour_rule": ("Which sign favours expected ('expected favoured if diff < 0' for metrics "
                    "where smaller is better; 'no preferred direction' otherwise)"),
    "n_favour_expected": dict(definition=("Pairs with expected - sampled < 0, for metrics where "
                                          "smaller is better (empty otherwise)"),
                              units="count (out of n_pairs)"),
    "n_diff_negative": dict(definition="Pairs with expected - sampled < 0", units="count"),
    "n_diff_positive": dict(definition="Pairs with expected - sampled > 0", units="count"),
    "n_diff_zero": dict(definition="Pairs with expected - sampled = 0", units="count"),
    "boot_ci95_lo": ("Lower end of the 95% percentile-bootstrap CI of the mean paired difference "
                     "(10,000 resamples of the 10 pairs, numpy seed 20261001)"),
    "boot_ci95_hi": "Upper end of the 95% percentile-bootstrap CI of the mean paired difference",
    "tier": dict(definition=("Tier of the metric: " + TIER_DEF + "; 'n/a (not a verifier "
                             "quantity)' for KL, clip fraction and wall time")),
    "source": "File the per-run values / statistics come from",
}


def build_t21(pack: C.Pack, ft: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    """T21: paired differences expected - sampled (existing rows plus new rows)."""
    script = f"{MOD}:build_t21"
    ps = pd.read_csv(C.abspath(PAIRED), float_precision="round_trip")
    repro = _t21_reproduce(ps, ft)
    old = ps.assign(tier=ps.metric.map(_tier_of), source="paired_summary.csv")
    # new rows (same method; fresh generator per row): the location-free peak error (not recorded in
    # Pilot 1, derived from the recovery arrays) and the final tier of the tier-dependent metrics
    fin = ft[["q", "seed", "arm"]].merge(rf, on=["q", "seed", "arm"], validate="one_to_one")
    fin["stage2_peak_locfree_rel_err_abs"] = fin.stage2_peak_locfree_rel_err.abs()
    src_lf = "final_development.npz recovery arrays (derived)"
    new = []
    for q in QS:
        lf, lfa = "stage2_peak_locfree_rel_err", "stage2_peak_locfree_rel_err_abs"
        new.append(_paired_row(_pair_diff(fin, q, lf), q, lf,
                               "location-free peak rel. err. (signed)", False, TI, src_lf))
        new.append(_paired_row(_pair_diff(fin, q, lfa), q, lfa, "|location-free peak rel. err.|",
                               True,
                               TI, src_lf))
        for metric in T21_FINAL:
            ref = ps[(ps.q == q) & (ps.metric == metric)].iloc[0]
            ftm = f"final_tier__{metric}"
            new.append(_paired_row(_pair_diff(fin, q, ftm), q, ftm,
                                   f"{ref.label} [final tier]", ref.favour_rule == LOWER, FIN,
                                   "final_v2.json['final']"))
    df = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
    df = df.sort_values("q", kind="stable").reset_index(drop=True)

    # cross-checks: pilot1 report section 3.2 (label key, per q) and summary.md (metric key)
    # the report words one label differently
    lab_map = {"RMSE |d|<2q (effort)": "RMSE over |d|<2q (effort)"}
    for q in QS:
        sub = old[old.q == q].assign(report_label=lambda x: x.label.replace(lab_map))
        pack.crosscheck("T21", sub, REP1,
                        header_has=["label", "mean", "sd", "median", "min", "max", "favour",
                                    "CI95"],
                        heading_has=f"q = {q}", key_map={"label": "report_label"},
                        value_map={c: c for c in ("mean", "sd", "median", "min", "max")},
                        label=f"q={q} paired differences: mean, sd, median, min, max (section 3.2)")
        _t21_favour_ci(pack, sub, q)
    vals = ("median", "n_favour_expected", "boot_ci95_lo", "boot_ci95_hi")
    pack.crosscheck("T21", old, SUMMARY, header_has=["q", "metric"] + list(vals),
                    heading_has="Pilot 1",
                    key_map={"q": "q", "metric": "metric"}, value_map={c: c for c in vals},
                    label="Pilot 1 paired differences in summary.md")
    _t21_prose(pack, old)

    notes = ("Rows with source paired_summary.csv: its 44 rows, values unchanged (expected - "
             "sampled per (q, seed) at u400; median, sign counts, 95% percentile bootstrap CI of "
             "the mean, 10,000 resamples of the 10 pairs, numpy seed 20261001, one generator over "
             "the rows in file order); their tier-dependent metrics are development tier. "
             "Recomputing them from "
             f"final_table.csv with that generator sequence reproduces the file (max abs diff "
             f"{repro:.3g}). New rows, the same statistics with common.bootstrap_mean_ci (10,000 "
             "resamples, numpy seed 20261001, a fresh generator per row): the location-free peak "
             "error, signed and absolute (tier-independent, derived from final_development.npz "
             "recovery arrays), and the final tier of the tier-dependent metrics (metric prefix "
             "final_tier__, final_v2.json['final']). n_favour_expected counts pairs with diff < 0 "
             "for metrics where smaller is better (empty for metrics without a preferred "
             "direction). EXP_root, dReach and Gmax_full are stage1_untrained (initial stage-1 "
             "network).")
    srcs = _p1_sources()
    sources = [C.src(PAIRED), C.src(FINAL_TABLE, selector="recomputation check"),
               srcs["final_v2.json"], srcs["final_development.npz"],
               C.src(META, selector="bootstrap settings"),
               C.src(REP1, selector="cross-check"), C.src(SUMMARY, selector="cross-check")]
    pack.table("T21", df, status="generated", sources=sources, script=script, notes=notes,
               docs=T21_DOCS, tier="final and development",
               caption="Paired differences expected - sampled at u400, per q (n = 10 pairs, seeds "
                       "10501-10510). Column tier labels each row.")
    return df


def _t21_favour_ci(pack: C.Pack, sub: pd.DataFrame, q: int) -> None:
    """Compare the 'favour' and 'CI95' cells of the report's section 3.2 table (q) with the CSV."""
    tabs = [t for t in C.parse_md_tables(REP1)
            if t["heading"] == f"q = {q}" and "CI95" in t["header"]]
    n = bad = unm = 0
    for t in tabs:
        h, where = t["header"], f"{REP1} (line {t['line']})"
        for row in t["rows"]:
            rec = dict(zip(h, row))
            hit = sub[sub.report_label == rec["label"]]
            if len(hit) != 1:
                unm += 1
                continue
            r = hit.iloc[0]
            fav = rec["favour"].replace("−", "-")
            if fav.startswith("n/a"):
                want = f"n/a ({int(r.n_diff_negative)}-/{int(r.n_diff_positive)}+)"
            else:
                want = f"{int(r.n_favour_expected)}/{int(r.n_pairs)}"
            n += 1
            if fav != want:
                bad += 1
                pack.mismatch("T21", f"favour [q={q}, {r.metric}]", want, where, rec["favour"])
            lo_s, hi_s = [x.strip() for x in rec["CI95"].strip("[]").split(",")]
            for val, txt, nm in ((r.boot_ci95_lo, lo_s, "boot_ci95_lo"),
                                 (r.boot_ci95_hi, hi_s, "boot_ci95_hi")):
                n += 1
                if not C.consistent(float(val), txt):
                    bad += 1
                    pack.mismatch("T21", f"{nm} [q={q}, {r.metric}]", val, where, txt)
    pack.crosschecks.append({"item": "T21", "report": REP1,
                             "label": f"q={q} favour counts and CI95 (section 3.2)",
                             "n_tables": len(tabs), "n_compared": n, "n_mismatch": bad,
                             "n_unmatched_rows": unm})


def _t21_prose(pack: C.Pack, ps: pd.DataFrame) -> None:
    """Check the sign-count statements of the pilot 1 report (section 5) against the paired rows."""
    def row(q: int, m: str) -> pd.Series:
        return ps[(ps.q == q) & (ps.metric == m)].iloc[0]

    rec_metrics = ["stage2_peak_rel_err_abs", "stage2_rmse_pos", "stage2_rmse_pos_over_g2_0",
                   "stage2_tail_mean", "stage2_tail_max", "eta_T_over_dw", "DeltaT_over_dw_on_max",
                   "DeltaT_over_dw_off_max"]
    n_all = sum(int(row(q, m).n_favour_expected == 10 and row(q, m).boot_ci95_hi < 0)
                for q in QS for m in rec_metrics)
    checks = [("pairs favouring expected with CI excluding 0, |peak|/RMSE/tail mean/tail max/eta_2/"
               "Delta_2 on and off max, both q (count of 16 metric-q cells meeting 10/10)", n_all,
               "16")]
    for q, want in ((50, "9"), (60, "7")):
        checks.append((f"sigma_2(0) lower with expected, q={q}",
                       row(q, "sigma_effort_at_0_t2").n_diff_negative, want))
        checks.append((f"mean sigma_2 over |d|<2q lower with expected, q={q}",
                       row(q, "sigma2_effort_mean_pos").n_diff_negative, "9"))
    full = ("EXP_root_over_dw", "dReach_over_dw", "Gmax_full_over_dw")
    pos = [int(row(q, m).n_diff_positive) for q in QS for m in full]
    checks += [("min pairs with EXP_root/dReach/Gmax_full higher with expected", min(pos), "8"),
               ("max pairs with EXP_root/dReach/Gmax_full higher with expected", max(pos), "10")]
    _prose(pack, "T21", REP1, "section 5: sign counts of the paired differences", checks)


# ----------------------------------------------------------------------------------------------
# F03: learning curves
# ----------------------------------------------------------------------------------------------

def _arm_patches(n_by_arm: Dict[str, int]) -> List[Any]:
    """Legend handles for the arms as colour patches (q-neutral), labels stating n."""
    from matplotlib.patches import Patch
    return [Patch(facecolor=style.arm_color(a), edgecolor="none",
                  label=style.label_n(style.arm_label(a), n_by_arm[a])) for a in ARMS]


F03_PANELS = [("stage2_peak_rel_err_signed", "peak error at d=0\n(signed, fraction of e₂*(0))"),
              ("stage2_rmse_pos_over_g2_0", "RMSE over |d|<2q\n(fraction of e₂*(0))"),
              ("stage2_tail_mean", "tail mean effort, |d|≥2q\n(raw, effort units)"),
              ("eta_T_over_dw", "η₂ / ΔW\n(development tier)")]


def _f03_checks(cv: pd.DataFrame, ft: pd.DataFrame, data: pd.DataFrame) -> List[str]:
    """Checks of the plotted F03 values against the source CSV and final_table.csv."""
    mets = [m for m, _ in F03_PANELS]
    u4 = cv[cv["update"] == 400].merge(ft, on=["q", "seed", "arm"], suffixes=("_w", "_ck"),
                                       validate="one_to_one")
    d_run = max(C.max_abs_diff(u4[f"{m}_w"], u4[f"{m}_ck"]) for m in mets)
    med_ft = ft.groupby(["q", "arm"])[mets].median()
    d_med = 0.0
    for (q, arm, metric), g in data[data["update"] == 400].groupby(["q", "arm", "panel"]):
        d_med = max(d_med, abs(float(g["median"].iloc[0]) - float(med_ft.loc[(q, arm), metric])))
    grp = cv.groupby(["q", "arm", "update"])[mets]
    d_pd = 0.0
    stats = (("median", grp.median()), ("q25", grp.quantile(0.25)), ("q75", grp.quantile(0.75)))
    for stat, fn in stats:
        for metric in mets:
            sel = data[data.panel == metric].set_index(["q", "arm", "update"])[stat]
            ref = fn[metric].reindex(sel.index).to_numpy()
            d_pd = max(d_pd, C.max_abs_diff(sel.to_numpy(), ref))
    return [f"per-run values at u400 equal the u400 checkpoint rows of final_table.csv (max abs "
            f"diff {d_run:.3g})",
            f"plotted medians at u400 equal final_table.csv medians (max abs diff {d_med:.3g})",
            f"plotted median/q25/q75 equal the pandas median/quantile of "
            f"tools/v2/pilot1_analysis.py:plots on the same CSV (max abs diff {d_pd:.3g})",
            "10 seeds at every (q, arm, update), exports u25-u400 every 25 updates",
            "compared visually with reports/v2/figures/pilot1/curves_q50.png and curves_q60.png "
            "(same median and IQR curves; the original also has a sigma_2(0) panel, not requested "
            "here)"]


def build_f03(pack: C.Pack, ft: pd.DataFrame) -> None:
    """F03: Pilot 1 learning curves (median and IQR per arm and q) from the 25-update exports."""
    script = f"{MOD}:build_f03"
    cv = pd.read_csv(C.abspath(CURVES), float_precision="round_trip")
    cnt = cv.groupby(["q", "arm", "update"]).seed.nunique()
    if not ((cnt == 10).all() and sorted(cv["update"].unique()) == list(range(25, 401, 25))):
        raise RuntimeError("curves_weights_every25.csv: not 10 seeds at every export u25..u400")
    fig, axes = style.new_figure(len(F03_PANELS), 2, height=7.6, sharex=True)
    parts = []
    for j, q in enumerate(QS):
        for i, (metric, ylab) in enumerate(F03_PANELS):
            ax = axes[i, j]
            if metric == "stage2_peak_rel_err_signed":
                ax.axhline(0.0, color=style.REF, lw=0.8)
            for arm in ARMS:
                g = cv[(cv.q == q) & (cv.arm == arm)]
                lab = style.label_n(style.arm_label(arm), int(g.seed.nunique()))
                parts.append(style.median_iqr_curves(
                    ax, g, "update", metric, style.arm_color(arm), lab, ls=style.Q_LINESTYLE[q],
                    marker=style.Q_MARKER[q], extra={"q": q, "arm": arm, "panel": metric}))
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(f"q = {q}")
            if i == len(F03_PANELS) - 1:
                ax.set_xlabel("update (weight export every 25 updates)")
    fig.legend(handles=_arm_patches(cv.groupby("arm").seed.nunique().to_dict()),
               loc="outside upper center", ncol=2)
    data = pd.concat(parts, ignore_index=True)
    checks = _f03_checks(cv, ft, data)
    caption = ("Across-seed median (line) and IQR (band, 25th to 75th percentile, numpy linear "
               "interpolation) of the stage-2 peak error at d=0 (signed, fraction of e2*(0)), RMSE "
               "over |d|<2q (fraction of e2*(0)), tail mean effort over |d|>=2q (raw effort units) "
               "and eta_2/Delta W against the update, at the weight exports every 25 updates (u25 "
               "to u400), per arm (colour) and q (column); n = 10 seeds (10501-10510) per arm and "
               "q. Data: results/v2_pilots/pilot1/analysis/curves_weights_every25.csv "
               "(re-evaluation of weights/u*.npz with utils/v2_metrics.evaluate on the development "
               "tier). eta_2 is development tier; the recovery metrics are tier-independent. Black "
               "horizontal line: zero error. q = 60 curves are dashed with square markers (pack "
               "convention: q is also encoded by line style and marker).")
    docs = {"panel": "Metric plotted in the panel (column of curves_weights_every25.csv)",
            "update": "Global update of the weight export (u25 ... u400)",
            "median": "Median over the 10 seeds at this export",
            "n": "Seeds with a finite value at this export"}
    sources = [C.src(CURVES), C.src(FINAL_TABLE, selector="u400 check"),
               C.src(f"{FIG_ORIG}/curves_q50.png", selector="visual comparison"),
               C.src(f"{FIG_ORIG}/curves_q60.png", selector="visual comparison")]
    pack.figure("F03", fig, data, status="regenerated", script=script, caption=caption,
                checks=checks, docs=docs, tier="development", sources=sources,
                notes="Re-plotted from the existing re-evaluation CSV (no forward pass); 4 metrics "
                      "x 2 q panels.")


# ----------------------------------------------------------------------------------------------
# F04: stage-2 mapping at u400
# ----------------------------------------------------------------------------------------------

def _band_rows(grid: np.ndarray, mat: np.ndarray) -> Dict[str, np.ndarray]:
    """Across-seed median, q25, q75, min, max and n per grid node (rows of ``mat`` are seeds)."""
    return {"d": grid, "median": np.median(mat, axis=0), "q25": np.percentile(mat, 25, axis=0),
            "q75": np.percentile(mat, 75, axis=0), "min": mat.min(axis=0), "max": mat.max(axis=0),
            "n": np.full(grid.size, mat.shape[0])}


def _stack(rds: Sequence[str], tier: str, grid_key: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
    """(common grid, seeds x nodes matrix) of one saved array over runs (grids identical)."""
    grids = [_npz(rd, tier)[grid_key] for rd in rds]
    if any(not np.array_equal(grids[0], g) for g in grids):
        raise RuntimeError(f"{grid_key} differs between runs")
    return grids[0], np.vstack([_npz(rd, tier)[key] for rd in rds])


F04_DOCS: Dict[str, Any] = {
    "series": "Plotted series: arm label, or closed_form (e2*(d))",
    "panel": ("e2_hat = across-seed band of e2hat(d); e2_star = closed form; sigma2 = band of "
              "sigma_2(d)"),
    "grid": "Grid of the d values: recovery grid (step 0.5) or final-tier stage-2 grid (step 2)",
    "d": dict(definition="Gap d at the start of stage 2 (grid node)", units="effort units (gap d)"),
    "median": "Median over the 10 seeds at the node (effort units)",
    "q25": "25th percentile over seeds at the node (numpy linear interpolation)",
    "q75": "75th percentile over seeds at the node (numpy linear interpolation)",
    "min": "Minimum over seeds at the node", "max": "Maximum over seeds at the node",
    "n": "Seeds at the node",
    "value": dict(definition="Closed-form e2*(d) (rows with series closed_form)",
                  units="effort units"),
}


def build_f04(pack: C.Pack, ft: pd.DataFrame, chk: Dict[str, float]) -> None:
    """F04: across-seed median and IQR of e2hat(d) at u400 against e2*(d), with sigma_2(d) below."""
    from matplotlib.lines import Line2D
    script = f"{MOD}:build_f04"
    fig, axes = style.new_figure(2, 2, height=5.6, sharex="col",
                                 gridspec_kw={"height_ratios": [1.7, 1.0]})
    parts, d_e0, d_s0, d_cf = [], 0.0, 0.0, 0.0
    for j, q in enumerate(QS):
        ax, axs = axes[0, j], axes[1, j]
        for arm in ARMS:
            rds = [rd for qq, _s, a, rd in S.iter_runs("pilot1") if qq == q and a == arm]
            col, ls = style.arm_color(arm), style.Q_LINESTYLE[q]
            grid, e2 = _stack(rds, "development", "recovery_d_grid", "recovery_e2")
            b = _band_rows(grid, e2)
            style.band_plot(ax, b["d"], b["median"], b["q25"], b["q75"], color=col, lw=1.3, ls=ls,
                            label=style.label_n(style.arm_label(arm), len(rds)))
            parts.append(pd.DataFrame({**b, "q": q, "series": arm, "panel": "e2_hat",
                                       "grid": "recovery (step 0.5)"}))
            e2_0 = np.median([_final_v2(rd)["final"]["e2_at_0"] for rd in rds])
            d_e0 = max(d_e0, abs(float(b["median"][b["d"] == 0.0][0]) - float(e2_0)))
            sgrid, sig = _stack(rds, "final", "v_t2_d_grid", "v_t2_sigma_effort")
            bs = _band_rows(sgrid, sig)
            style.band_plot(axs, bs["d"], bs["median"], bs["q25"], bs["q75"], color=col, lw=1.3,
                            ls=ls)
            parts.append(pd.DataFrame({**bs, "q": q, "series": arm, "panel": "sigma2",
                                       "grid": "final-tier D2 (step 2)"}))
            s0 = ft[(ft.q == q) & (ft.arm == arm)].sigma_effort_at_0_t2.median()
            d_s0 = max(d_s0, abs(float(bs["median"][bs["d"] == 0.0][0]) - float(s0)))
        rd0 = S.run_dir("pilot1", q, S.SEEDS_DEV[0], "expected")
        game = _manifest(rd0)["resolved_config"]["game"]
        grid = _npz(rd0, "development")["recovery_d_grid"]
        cf = g2_two_stage(grid, q, game["w_h"], game["w_l"], game["k"], game["e_max"])
        d_cf = max(d_cf, C.max_abs_diff(cf, _npz(rd0, "development")["recovery_g2"]))
        ax.plot(grid, cf, color=style.REF, lw=0.9, ls="-", label="closed form e₂*(d)")
        parts.append(pd.DataFrame({"d": grid, "value": cf, "q": q, "series": "closed_form",
                                   "panel": "e2_star", "grid": "recovery (step 0.5)"}))
        ax.set_title(f"q = {q}")
        axs.set_xlabel("gap d at the start of stage 2 (effort units)")
        if j == 0:
            ax.set_ylabel("stage-2 effort ê₂(d)\n(raw, effort units)")
            axs.set_ylabel("σ₂(d)\n(effort units)")
    n_arm = {a: int(ft[ft.arm == a].seed.nunique()) for a in ARMS}
    ref = Line2D([], [], color=style.REF, lw=0.9, label="closed form e₂*(d)")
    fig.legend(handles=_arm_patches(n_arm) + [ref], loc="outside upper center", ncol=3)
    cols = ["q", "series", "panel", "grid", "d", "median", "q25", "q75", "min", "max", "n", "value"]
    data = pd.concat(parts, ignore_index=True)[cols]
    checks = [f"recovery arrays (recovery_d_grid, recovery_e2, recovery_g2) of "
              f"final_development.npz and final_final.npz identical in 40/40 runs (max abs diff "
              f"{chk['npz_dev_vs_final']:.3g})",
              f"closed form utils/theory_multistage.py:g2_two_stage equals recovery_g2 (max abs "
              f"diff {max(d_cf, chk['closed_form_vs_grid']):.3g})",
              f"recovery_e2 at the development D2 nodes equals the verifier's Beta mean v_t2_e_hat "
              f"in 40/40 runs (max abs diff {chk['recovery_vs_ehat']:.3g})",
              f"median e2hat at d=0 equals the median of final_v2.json e2_at_0 (max abs diff "
              f"{d_e0:.3g})",
              f"sigma_2(d) on the final-tier D2 grid equals the development-tier array at the "
              f"common nodes (max abs diff {chk['sigma_final_vs_dev']:.3g}); median sigma_2(0) "
              f"equals the final_table.csv median (max abs diff {d_s0:.3g})"]
    caption = ("Upper row: across-seed median (line) and IQR (band) of the stage-2 Beta-mean "
               "effort e2hat(d) at u400 on the 0.5-step recovery grid over D2 = [-B, B] (B = 200 "
               "at q=50, 220 at q=60), per arm, with the closed form e2*(d) (thin black line, "
               "utils/theory_multistage.py:g2_two_stage); q=60 arm lines are dashed (pack "
               "convention). Lower row: across-seed median and IQR of sigma_2(d), the SD of the "
               "stage-2 Beta action in effort units, on the final-tier D2 grid (step 2). n = 10 "
               "seeds (10501-10510) per arm and q. Data: final_development.npz (recovery_d_grid, "
               "recovery_e2) and final_final.npz (v_t2_d_grid, v_t2_sigma_effort) of each Pilot 1 "
               "run. Both quantities are direct policy queries (tier-independent).")
    srcs = _p1_sources()
    sources = [srcs["final_development.npz"], srcs["final_final.npz"], srcs["final_v2.json"],
               srcs["manifest.json"], C.src(FINAL_TABLE, selector="sigma_2(0) check"),
               C.src("utils/theory_multistage.py", selector="g2_two_stage")]
    pack.figure("F04", fig, data, status="generated", script=script, caption=caption, checks=checks,
                docs=F04_DOCS, tier="tier-independent", sources=sources)


# ----------------------------------------------------------------------------------------------
# F05: peak error against sigma_2(0)/q
# ----------------------------------------------------------------------------------------------

def _checkpoints() -> pd.DataFrame:
    """The 160 training-time verifier checkpoint rows of Pilot 1 (selected columns)."""
    parts = []
    for q, seed, arm, rd in S.iter_runs("pilot1"):
        c = pd.read_csv(C.abspath(f"{rd}/v2_checkpoints.csv"), float_precision="round_trip")
        c = c[["update", "reason", "stage2_peak_rel_err_signed", "sigma_effort_at_0_t2"]].copy()
        parts.append(c.assign(q=q, seed=seed, arm=arm))
    ck = pd.concat(parts, ignore_index=True)
    ck["sigma_over_q"] = ck.sigma_effort_at_0_t2 / ck.q
    return ck


def _spearman(ck: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho and p of peak error vs sigma_2(0)/q per q and arm (and pooled).

    Over all checkpoints (the existing statistic) and over the u400 checkpoints only.
    """
    rows = []
    for subset, sel in (("all checkpoints", ck), ("u400 only", ck[ck["update"] == 400])):
        for q in QS:
            g = sel[sel.q == q]
            for arm in list(ARMS) + ["pooled"]:
                h = g if arm == "pooled" else g[g.arm == arm]
                r = spearmanr(h.sigma_over_q, h.stage2_peak_rel_err_signed)
                rows.append({"kind": "spearman", "subset": subset, "q": q, "arm": arm,
                             "n_points": len(h), "spearman_rho": float(r.statistic),
                             "p_value": float(r.pvalue)})
    return pd.DataFrame(rows)


def _minus(x: float) -> str:
    """Three-decimal text with a typographic minus sign."""
    return f"{x:.3f}".replace("-", "−")


F05_DOCS: Dict[str, Any] = {
    "kind": ("Row kind: checkpoint = one plotted point; spearman = a Spearman statistic in the "
             "legend"),
    "sigma_over_q": dict(definition="sigma_2(0) divided by q", units="dimensionless"),
    "subset": ("Points behind the Spearman row: all checkpoints (4 per run) or the u400 "
               "checkpoints only"),
    "n_points": dict(definition="Points behind the Spearman statistic", units="count"),
    "spearman_rho": dict(definition=("Spearman rank correlation of stage2_peak_rel_err_signed and "
                                     "sigma_2(0)/q (scipy.stats.spearmanr)"),
                         units="dimensionless"),
    "p_value": dict(definition="Two-sided p-value of scipy.stats.spearmanr (descriptive)",
                    units="probability"),
    "update": "Global update of the training-time verifier checkpoint",
}


def build_f05(pack: C.Pack, ck: pd.DataFrame) -> None:
    """F05: peak error at d=0 against sigma_2(0)/q at all 160 checkpoints, Spearman rho per arm."""
    script = f"{MOD}:build_f05"
    sp = _spearman(ck)
    old = pd.read_csv(C.abspath(SPEARMAN), float_precision="round_trip")
    allc = sp[sp.subset == "all checkpoints"].merge(old, on=["q", "arm"], suffixes=("", "_csv"),
                                                    validate="one_to_one")
    d_rho = C.max_abs_diff(allc.spearman_rho, allc.spearman_rho_csv)
    d_p = C.max_abs_diff(allc.p_value, allc.p_value_csv)
    d_n = C.max_abs_diff(allc.n_points, allc.n_points_csv)
    vals = ("n_points", "spearman_rho", "p_value")
    pack.crosscheck("F05", allc, REP1, header_has=["q", "arm"] + list(vals),
                    key_map={"q": "q", "arm": "arm"}, value_map={c: c for c in vals},
                    label="Spearman rho of peak error vs sigma_2(0)/q (section 3.4 table)")
    rho = sp.set_index(["subset", "q", "arm"]).spearman_rho
    ac = "all checkpoints"
    _prose(pack, "F05", REP1, "section 5: Spearman rho within arms", [
        ("rho sampled q=50 (all checkpoints)", rho[(ac, 50, "sampled")], "0.87"),
        ("rho sampled q=60 (all checkpoints)", rho[(ac, 60, "sampled")], "0.93"),
        ("rho expected q=50 (all checkpoints)", rho[(ac, 50, "expected")], "-0.45"),
        ("rho expected q=60 (all checkpoints)", rho[(ac, 60, "expected")], "-0.18")])

    fig, axes = style.new_figure(1, 2, height=4.7)
    for j, q in enumerate(QS):
        ax = axes[0, j]
        ax.axhline(0.0, color=style.REF, lw=0.8)
        for arm in ARMS:
            h = ck[(ck.q == q) & (ck.arm == arm)]
            early, last = h[h["update"] < 400], h[h["update"] == 400]
            col, mk = style.arm_color(arm), style.Q_MARKER[q]
            ax.scatter(early.sigma_over_q, early.stage2_peak_rel_err_signed, s=16, color=col,
                       alpha=0.6, linewidths=0, marker=mk)
            lab = (f"{style.arm_label(arm)}: ρ = {_minus(rho[(ac, q, arm)])}, all checkpoints "
                   f"(n = {len(h)})\nρ = {_minus(rho[('u400 only', q, arm)])}, u400 only (dark "
                   f"edge, n = {len(last)})")
            ax.scatter(last.sigma_over_q, last.stage2_peak_rel_err_signed, s=24, color=col,
                       edgecolors=style.INK, linewidths=0.8, label=lab, marker=mk)
        ax.set_title(f"q = {q}")
        ax.set_xlabel("σ₂(0) / q")
        if j == 0:
            ax.set_ylabel("peak error at d=0\n(signed, fraction of e₂*(0))")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), handletextpad=0.3,
                  borderaxespad=0.0)
    data = pd.concat([ck.assign(kind="checkpoint").drop(columns=["reason"]), sp], ignore_index=True)
    data = data[["kind", "q", "arm", "seed", "update", "sigma_effort_at_0_t2", "sigma_over_q",
                 "stage2_peak_rel_err_signed", "subset", "n_points", "spearman_rho", "p_value"]]
    checks = ["160 checkpoint rows (4 per run, u100/u200/u300/u400 in every run), the same rows as "
              "the original figure (tools/v2/pilot1_analysis.py:plots)",
              f"Spearman rho over all checkpoints recomputed with scipy.stats.spearmanr equals "
              f"spearman_peakerr_vs_sigma.csv (max abs diff rho {d_rho:.3g}, p {d_p:.3g}, n "
              f"{d_n:.3g})",
              "compared visually with reports/v2/figures/pilot1/scatter_peakerr_vs_sigma.png (same "
              "points)"]
    caption = ("Stage-2 peak error at d=0 (signed, fraction of e2*(0)) against sigma_2(0)/q at "
               "every training-time verifier checkpoint of every Pilot 1 run (u100, u200, u300, "
               "u400; 4 per run, 40 points per arm and q, 160 in total), colour by arm, one panel "
               "per q. Points with a dark edge are the u400 (final) checkpoint of each run. "
               "Legend: Spearman rho per arm over all checkpoints (n = 40) and over the u400 "
               "checkpoints only (n = 10 runs). Data: v2_checkpoints.csv of each run (columns "
               "stage2_peak_rel_err_signed, sigma_effort_at_0_t2); rho also in "
               "results/v2_pilots/pilot1/analysis/spearman_peakerr_vs_sigma.csv. Both metrics are "
               "tier-independent. Black horizontal line: zero error. Markers: circles at q = 50, "
               "squares at q = 60 (pack convention).")
    srcs = _p1_sources()
    sources = [srcs["v2_checkpoints.csv"], C.src(SPEARMAN),
               C.src(f"{FIG_ORIG}/scatter_peakerr_vs_sigma.png", selector="visual comparison"),
               C.src(REP1, selector="cross-check")]
    pack.figure("F05", fig, data, status="regenerated", script=script, caption=caption,
                checks=checks, docs=F05_DOCS, tier="tier-independent", sources=sources,
                notes="Re-plotted from the per-run checkpoint CSVs (the same 160 rows as the "
                      "original figure); the Spearman rho over the u400 checkpoints only (n = 10 "
                      "per arm) is new.")


def _checkpoint_claim(pack: C.Pack, ck: pd.DataFrame) -> None:
    """Record the report's stated reason for not using the checkpoints in the curves (section 3.3)
    if the checkpoint data contradict it."""
    pat = ck.groupby(["q", "seed", "arm"])["update"].apply(tuple)
    common = pat.value_counts()
    if len(common) == 1:
        ups = ", ".join(f"u{u}" for u in common.index[0])
        reasons = ", ".join(sorted(ck.reason.astype(str).unique()))
        pack.mismatch("F03", "updates of the training-time verifier checkpoints",
                      f"{ups} in {int(common.iloc[0])}/40 runs (reasons: {reasons})",
                      f"{REP1} (section 3.3, 'Deviation')",
                      "'Those fall at run-specific updates, because stability-triggered calls move "
                      "them, so they cannot be aligned across seeds.'",
                      "text claim: in Pilot 1 the 4 checkpoints are at the same updates in every "
                      "run (only 4 points per run, so the 25-update exports remain the finer "
                      "series)")


# ----------------------------------------------------------------------------------------------
# D01: per-run table
# ----------------------------------------------------------------------------------------------

SMOOTHED_DOCS: Dict[str, Tuple[str, str]] = {
    "rmse_learned_minus_pred_pos": (
        "RMSE over |d| < 2q (development-tier D2 nodes, step 4) of e_learned(d) - e_pred(d); "
        "e_pred(d) = (Delta W / 2k) E[f_xi(d + a_i - a_j)] is the smoothed-game (location-shift) "
        "prediction from the learned u400 stage-2 Beta actions", "effort units"),
    "rmse_learned_minus_estar_pos": (
        "RMSE over |d| < 2q (development-tier D2 nodes) of e_learned(d) - e2*(d)", "effort units"),
    "e_star_0": ("e2*(0), closed-form stage-2 effort at d = 0", "effort units"),
    "e_pred_0": ("Smoothed-game prediction e_pred(0)", "effort units"),
    "e_learned_0": ("e_learned(0): stage-2 Beta-mean effort at d = 0 (v_t2_e_hat)",
                    "effort units [0, 100]"),
    "share_peak_gap_explained": (
        "(e2*(0) - e_pred(0)) / (e2*(0) - e_learned(0)): share of the d=0 peak gap predicted by "
        "the policy's own action noise", "fraction"),
    "n_nodes_per_beta": ("Equal-probability quadrature nodes per Beta (tensor product over the two "
                         "Betas)", "count"),
}
D01_REPORT_COLS = {
    "peak rel (signed)": "stage2_peak_rel_err_signed", "RMSE": "stage2_rmse_pos",
    "RMSE/e2*(0)": "stage2_rmse_pos_over_g2_0", "tail mean": "stage2_tail_mean",
    "tail max": "stage2_tail_max", "sym": "stage2_sym_err_max", "η₂": "eta_T_over_dw",
    "Δ₂ on wmean": "DeltaT_over_dw_on_mean_cellmass_weighted",
    "Δ₂ off max": "DeltaT_over_dw_off_max", "σ₂(0)": "sigma_effort_at_0_t2",
    "σ̄₂(|d|<2q)": "sigma2_effort_mean_pos", "KL": "kl_final_epoch", "clip": "clip_frac",
    "wall s": "phase_A_wall_sec", "EXP*": "EXP_root_over_dw", "dReach*": "dReach_over_dw",
    "Ĝmax*": "Gmax_full_over_dw"}
D01_SMOOTHED_COLS = {
    "RMSE(learned − pred)": "smoothed_rmse_learned_minus_pred_pos",
    "RMSE(learned − e2*)": "smoothed_rmse_learned_minus_estar_pos",
    "ê_learned(0)": "smoothed_e_learned_0", "ê_pred(0)": "smoothed_e_pred_0",
    "e2*(0)": "smoothed_e_star_0",
    "share of peak gap explained": "smoothed_share_peak_gap_explained"}
LF_COLS = ["stage2_peak_locfree_rel_err", "stage2_peak_locfree_argmax_d", "stage2_max_e2"]


def _d01_checks(pack: C.Pack, df: pd.DataFrame) -> None:
    """D01 against the report's per-run tables (sections 3.1 and 7), summary.md and prose."""
    key = {"q": "q", "seed": "seed", "arm": "arm"}
    pack.crosscheck("D01", df, REP1, header_has=["q", "seed", "arm"] + list(D01_REPORT_COLS),
                    key_map=key, value_map=D01_REPORT_COLS, heading_has="3.1",
                    label="per-run final-checkpoint table (section 3.1)")
    pack.crosscheck("D01", df, REP1, header_has=["q", "seed", "arm"] + list(D01_SMOOTHED_COLS),
                    key_map=key, value_map=D01_SMOOTHED_COLS, heading_has="Per run",
                    label="per-run smoothed-game table (section 7)")
    med_exp = df.groupby(["q", "arm"]).EXP_root_over_dw.median().reset_index()
    pack.crosscheck("D01", med_exp, SUMMARY,
                    header_has=["q", "arm", "stage2_peak_rel_err_signed", "EXP_root_over_dw"],
                    key_map={"q": "q", "arm": "arm"},
                    value_map={"EXP_root_over_dw": "EXP_root_over_dw"},
                    heading_has="Pilot 1",
                    label="median EXP_root/dW (stage1_untrained, development tier) in summary.md")
    on_eq = int((df.eta_T_over_dw == df.DeltaT_over_dw_on_max).sum())
    exc = df[df.eta_T_over_dw != df.DeltaT_over_dw_on_max]
    checks = [("runs where the on-path max of Delta_2 equals eta_2 (development tier)", on_eq,
               "39")]
    if len(exc) == 1:
        e = exc.iloc[0]
        who = f"q{e.q} s{e.seed} {e.arm}"
        checks += [(f"eta_2 of the exception ({who})", e.eta_T_over_dw, "0.001511"),
                   (f"on-path max of the exception ({who})", e.DeltaT_over_dw_on_max, "0.001451")]
    checks += [("runs with stage1_status = stage1_untrained",
                int((df.stage1_status == "stage1_untrained").sum()), "40"),
               ("runs with would_have_fired_A = null",
                int((df.would_have_fired_A.astype(str) == "null").sum()), "40"),
               ("runs with 4 training-time verifier checkpoints",
                int((df.n_verifier_checkpoints == 4).sum()), "40")]
    _prose(pack, "D01", REP1, "sections 1 and 3.1: on-path max vs eta_2, stage1_status, stop rule, "
           "checkpoints", checks)


def _d01_docs(ft: pd.DataFrame, fin_cols: Sequence[str]) -> Dict[str, Any]:
    """Column documentation of D01 (overrides and additions to the global dictionary)."""
    docs: Dict[str, Any] = {
        "phase_A_wall_sec": dict(definition=("Phase A wall-clock time of the run "
                                             "(v2_run_summary.json phase_timing.A.wall_sec)"),
                                 units="seconds", tier="n/a"),
        "would_have_fired_A": dict(definition=("Record of the existing Phase A stop rule "
                                               "(v2_run_summary.json would_have_fired.A, JSON); "
                                               "null = it would not have fired within the budget"),
                                   units="JSON text", tier="n/a"),
        "n_verifier_checkpoints": dict(definition=("Number of training-time verifier checkpoints "
                                                   "of the run (rows of v2_checkpoints.csv)"),
                                       units="count", tier="n/a"),
        "update": "Global update of the final checkpoint (u400)",
    }
    for col in ft.columns:
        base = D.lookup(col, DEV)
        if base and base["tier"] == DEV and col not in docs:
            docs[col] = {"tier": "development (u400 training-time checkpoint)"}
    for col in fin_cols:
        base = D.lookup(col[len("final_tier__"):], FIN) or {}
        docs[col] = dict(definition=("Final tier (state step 2, effort step 0.5, GL 32 per half) "
                                     "evaluation of the u400 policy, final_v2.json['final']: "
                                     + base.get("definition", col)),
                         units=base.get("units", ""), normalization=base.get("normalization", ""),
                         tier="final",
                         source=("final_v2.json['final'] ("
                                 + base.get("source", "utils/v2_metrics.py:evaluate") + ")"))
    for k, (definition, units) in SMOOTHED_DOCS.items():
        docs[f"smoothed_{k}"] = dict(definition=("Smoothed-game side analysis "
                                                 "(analysis/smoothed_game/per_run.csv): "
                                                 + definition),
                                     units=units, normalization="none",
                                     tier="development-tier D2 grid (no verifier)",
                                     source="tools/v2/pilot1_smoothed_game.py")
    for c in LF_COLS:
        docs[c] = {"source": ("derived here from final_development.npz recovery_d_grid / "
                              "recovery_e2 (tools/v2/pilot4_common.py:location_free rule)")}
    return docs


def build_d01(pack: C.Pack, ft: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    """D01: the 40 rows of final_table.csv plus location-free, final-tier and smoothed columns."""
    script = f"{MOD}:build_d01"
    sm = pd.read_csv(C.abspath(SMOOTHED), float_precision="round_trip")
    keys = ("q", "seed", "arm")
    sm = sm.rename(columns={c: f"smoothed_{c}" for c in sm.columns if c not in keys})
    fin_cols = [c for c in rf.columns if c.startswith("final_tier__")]
    df = (ft.merge(rf[["q", "seed", "arm"] + LF_COLS + fin_cols], on=["q", "seed", "arm"],
                   how="left",
                   validate="one_to_one")
          .merge(sm, on=["q", "seed", "arm"], how="left", validate="one_to_one"))
    if len(df) != 40 or df[LF_COLS + fin_cols + list(sm.columns)].isna().any().any():
        raise RuntimeError("D01: missing appended values")
    _d01_checks(pack, df)
    srcs = _p1_sources()
    notes = ("The 40 rows of results/v2_pilots/pilot1/analysis/final_table.csv with their original "
             "column names and values (development-tier u400 checkpoint; seeds in column seed), "
             "plus appended columns: location-free peak error, its argmax d and max e2hat (derived "
             "from final_development.npz), the final-tier scalars of the same u400 policy "
             "(final_tier__*, final_v2.json['final'], via studies.final_tier_columns) and the "
             "smoothed-game columns of analysis/smoothed_game/per_run.csv (prefix smoothed_). "
             "Joined on (q, seed, arm). EXP_root, dReach and Gmax_full are stage1_untrained "
             "(column stage1_status). " + _tail_argmax_text(rf))
    sources = [C.src(FINAL_TABLE), srcs["final_v2.json"], srcs["final_development.npz"],
               C.src(SMOOTHED), C.src(REP1, selector="cross-check"),
               C.src(SUMMARY, selector="cross-check")]
    pack.data("D01", df, status="generated", script=script, notes=notes,
              docs=_d01_docs(ft, fin_cols), tier="final and development", sources=sources)
    return df


# ----------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------

def build() -> None:
    """Build T19, T20, T21, F03, F04, F05 and D01 into the pack."""
    pack = C.Pack("sec_pilot1")
    ft = _read_final_table()
    rf, chk = _run_frame()
    ck = _checkpoints()
    build_t19(pack)
    build_t20(pack, ft, rf, chk)
    build_t21(pack, ft, rf)
    build_f03(pack, ft)
    _checkpoint_claim(pack, ck)
    build_f04(pack, ft, chk)
    build_f05(pack, ck)
    build_d01(pack, ft, rf)
    pack.save_fragment()
