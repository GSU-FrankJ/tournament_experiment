#!/usr/bin/env python3
"""Build evidence/ and figures/ of reports/t2_status_100826 from git objects.

Every copied file is read with ``git show <source commit>:<path>`` (the last commit that
touched the path at SRC_COMMIT), so the pack depends on recorded commits, not on the working
tree. Files of 1 MiB or more, and untracked files, are listed as referenced and not copied.
Items already in the 100526 pack are cited in place (status ``cited in t2_refine_100526``).

Usage (from anywhere inside the repository)::

    python reports/t2_status_100826/report_scripts/build_pack.py          # (re)build in place
    python reports/t2_status_100826/report_scripts/build_pack.py --check  # rebuild in a temp dir
                                                                          # and compare byte for byte
"""
import argparse
import csv
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
REPO = Path(subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                                    text=True).strip())
PACK_REL = PACK.relative_to(REPO).as_posix()
SRC_COMMIT = "be4fd2021e5aee52625a994b3566dc18726e0586"   # origin/ms-r3
MAX_COPY = 1024 * 1024
T2R_MANIFEST = "reports/t2_refine_100526/evidence/manifest.csv"
T2R_EVIDENCE = "reports/t2_refine_100526/evidence/"
FIELDS = ["item_id", "title", "round", "source_branch", "source_commit", "source_path", "sha256",
          "size_bytes", "status", "copy_path", "conditions", "used_in"]

sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------------------------------------
# Registry. (id, title, source path, conditions)
# ---------------------------------------------------------------------------------------------
DEV = "development seeds 10501-10510, q in {50, 60}, n = 10 per (arm, q)"
M1: List[Tuple[str, str, str, str]] = [
    ("M1-01", "MS-R1 summary report", "reports/ms/r1/summary.md", ""),
    ("M1-02", "MS-R1 decision inputs", "reports/ms/r1/05_decision_inputs.md", DEV),
    ("M1-03", "MS-R1 pilot report", "reports/ms/r1/04_pilot.md", DEV + "; 120 pilot runs"),
    ("M1-04", "MS-R1 pre-registration with Addendum 1 (G1 decision)", "reports/ms/r1/02_preregistration.md", ""),
    ("M1-05", "MS-R1 calibration of the rule on the v2.0 exports", "reports/ms/r1/01_calibration.md", "v2.0 weight exports, development seeds"),
    ("M1-06", "MS-R1 checks (C-R4, C-MS1, tests, C-MS2)", "reports/ms/r1/03_checks.md", ""),
    ("M1-07", "MS-R1 pre-registered parameter file", "reports/ms/r1/prereg_parameters.json", ""),
    ("M1-08", "MS-R1 per-run table (arms and comparators)", "results/ms_r1/analysis/per_run.csv", DEV + "; 140 MS-arm runs + 40 comparator rows"),
    ("M1-09", "MS-R1 primary criterion (arms vs parents_A)", "results/ms_r1/analysis/criterion.csv", "bootstrap 10,000 resamples, paired by (q, seed)"),
    ("M1-10", "MS-R1 secondary criterion (arms vs MS_base2400)", "results/ms_r1/analysis/criterion_vs_MS_base2400.csv", "budget-matched control"),
    ("M1-11", "MS-R1 decision inputs table", "results/ms_r1/analysis/decision_inputs.csv", ""),
    ("M1-12", "MS-R1 budget table (updates, episodes, wall time)", "results/ms_r1/analysis/budget.csv", ""),
    ("M1-13", "MS-R1 stop-rule summary", "results/ms_r1/analysis/rule.csv", "rho_2 = 0.05"),
    ("M1-14", "MS-R1 R0 table", "results/ms_r1/analysis/r0.csv", ""),
    ("M1-15", "MS-R1 analysis summary text", "results/ms_r1/analysis/summary.txt", ""),
    ("M1-16", "MS-R1 stage-1 table", "results/ms_r1/analysis/stage1.csv", ""),
    ("M1-17", "MS-R1 tail table", "results/ms_r1/analysis/tail.csv", ""),
    ("M1-18", "MS-R1 pilot launch record", "results/ms_r1/pilot/launch_20261007_055045.json", "120 runs, 40 workers"),
    ("M1-19", "MS-R1 pilot launch checks (incl. C-MS2)", "results/ms_r1/pilot/launch_checks.json", ""),
    ("M1-20", "MS-R1 C-R4: unchanged v2.0 entry point reproduces rehearsal_v2_0", "results/ms_r1/v20_reproduction_checks.json", "20 runs"),
    ("M1-21", "MS-R1 base-wave launch record", "results/ms_r1/base/launch_20261007_030049.json", "20 base runs"),
    ("M1-22", "MS-R1 C-MS1 against parents_A", "results/ms_r1/base_checks_parents_A.json", "20 runs"),
    ("M1-23", "MS-R1 C-MS1 against rehearsal_v2_0", "results/ms_r1/base_checks_rehearsal_v2_0.json", "20 runs"),
    ("M1-24", "MS-R1 blind recomputation of the criterion tables", "results/ms_r1/analysis/blind_recomputation.txt", ""),
    ("M1-25", "MS-R1 analysis info (roots, seeds)", "results/ms_r1/analysis/analysis_info.json", ""),
    ("M1-26", "MS-R1 housekeeping record", "reports/ms/r1/00_housekeeping.md", ""),
]
M2: List[Tuple[str, str, str, str]] = [
    ("M2-01", "MS-R2 summary report", "reports/ms/r2/summary.md", ""),
    ("M2-02", "MS-R2 decision inputs", "reports/ms/r2/05_decision_inputs.md", DEV),
    ("M2-03", "MS-R2 pilot report", "reports/ms/r2/04_pilot.md", DEV + "; 120 pilot runs"),
    ("M2-04", "MS-R2 pre-registration", "reports/ms/r2/02_preregistration.md", ""),
    ("M2-05", "MS-R2 decomposition premise check on MS-R1", "reports/ms/r2/01_decomposition.md", ""),
    ("M2-06", "MS-R2 stop-candidate calibration on MS-R1 (D6)", "reports/ms/r2/01b_stop_candidates.md", ""),
    ("M2-07", "MS-R2 checks", "reports/ms/r2/03_checks.md", ""),
    ("M2-08", "MS-R2 per-run table", "results/ms_r2/analysis/per_run.csv", DEV + "; 120 MS-arm runs + 80 comparator rows"),
    ("M2-09", "MS-R2 primary criterion (NL arms vs same-sampler s = 1)", "results/ms_r2/analysis/criterion.csv", "bootstrap 10,000 resamples, paired by (q, seed)"),
    ("M2-10", "MS-R2 arms vs parents_A with MS-R1's criterion", "results/ms_r2/analysis/criterion_vs_parents_A.csv", "descriptive; confounds budget, sampler, scale"),
    ("M2-11", "MS-R2 transmission of the smoothing reduction to the gap", "results/ms_r2/analysis/transmission.csv", ""),
    ("M2-12", "MS-R2 sampler-by-scale interaction", "results/ms_r2/analysis/interaction.csv", ""),
    ("M2-13", "MS-R2 gate counts", "results/ms_r2/analysis/gates.csv", ""),
    ("M2-14", "MS-R2 budget table", "results/ms_r2/analysis/budget.csv", ""),
    ("M2-15", "MS-R2 predictions against outcome", "results/ms_r2/analysis/predictions.csv", ""),
    ("M2-16", "MS-R2 analysis summary text", "results/ms_r2/analysis/summary.txt", ""),
    ("M2-17", "MS-R2 pilot launch record", "results/ms_r2/pilot/launch_20261007_094801.json", "120 runs, 40 workers"),
    ("M2-18", "MS-R2 pilot launch checks (C-NL, C-MS3, C-MS4)", "results/ms_r2/pilot/launch_checks.json", ""),
    ("M2-19", "MS-R2 blind recomputation", "results/ms_r2/analysis/blind_recomputation.txt", ""),
    ("M2-20", "MS-R2 analysis info", "results/ms_r2/analysis/analysis_info.json", ""),
    ("M2-21", "MS-R2 fact-check ledger", "reports/ms/r2/pi_record/01_factcheck.md", ""),
    ("M2-22", "MS-R2 stage-1 table", "results/ms_r2/analysis/stage1.csv", ""),
]
M3: List[Tuple[str, str, str, str]] = [
    ("M3-01", "MS-R3 summary report", "reports/ms/r3/summary.md", ""),
    ("M3-02", "MS-R3 decision inputs", "reports/ms/r3/05_decision_inputs.md", DEV),
    ("M3-03", "MS-R3 pilot report", "reports/ms/r3/04_pilot.md", DEV + "; 240 pilot runs"),
    ("M3-04", "MS-R3 pre-registration", "reports/ms/r3/02_preregistration.md", ""),
    ("M3-05", "MS-R3 supervised screen and premise check", "reports/ms/r3/01_supervised_screen.md", "offline fit to the closed form"),
    ("M3-06", "MS-R3 RL-actor diagnostics on MS-R1/R2 exports", "reports/ms/r3/01b_rl_actor_diagnostics.md", ""),
    ("M3-07", "MS-R3 checks (tests, review, C-R6)", "reports/ms/r3/03_checks.md", ""),
    ("M3-08", "MS-R3 per-run table", "results/ms_r3/analysis/per_run.csv", DEV + "; 240 MS-arm runs + 120 comparator rows"),
    ("M3-09", "MS-R3 primary criterion (relu, t10 vs t1)", "results/ms_r3/analysis/criterion.csv", "bootstrap 10,000 resamples, paired by (q, seed)"),
    ("M3-10", "MS-R3 arms vs parents_A with MS-R1's criterion", "results/ms_r3/analysis/criterion_vs_parents_A.csv", "descriptive; confounds budget, sampler, landing"),
    ("M3-11", "MS-R3 transmission of the smoothing reduction to the gap", "results/ms_r3/analysis/transmission.csv", ""),
    ("M3-12", "MS-R3 quadrature check (additive vs quadrature)", "results/ms_r3/analysis/quadrature_check.csv", "12 cells"),
    ("M3-13", "MS-R3 Spearman of R0 with |peak error|", "results/ms_r3/analysis/r0_spearman.csv", ""),
    ("M3-14", "MS-R3 relu hidden-unit activity", "results/ms_r3/analysis/relu_units.csv", "post hoc"),
    ("M3-15", "MS-R3 trajectory by arm (local updates 1800-2800)", "results/ms_r3/analysis/trajectory_by_arm.csv", ""),
    ("M3-16", "MS-R3 gate counts", "results/ms_r3/analysis/gates.csv", ""),
    ("M3-17", "MS-R3 budget table", "results/ms_r3/analysis/budget.csv", ""),
    ("M3-18", "MS-R3 predictions against outcome", "results/ms_r3/analysis/predictions.csv", ""),
    ("M3-19", "MS-R3 analysis summary text", "results/ms_r3/analysis/summary.txt", ""),
    ("M3-20", "MS-R3 pilot launch record", "results/ms_r3/pilot/launch_20261008_002423.json", "240 runs, 40 workers"),
    ("M3-21", "MS-R3 pilot launch checks (C-INIT, C-NL, C-MS5)", "results/ms_r3/pilot/launch_checks.json", ""),
    ("M3-22", "MS-R3 C-R6: unchanged v2.0 entry point reproduces rehearsal_v2_0", "results/ms_r3/v20_reproduction_checks.json", "20 runs"),
    ("M3-23", "MS-R3 premise check (supervised fit)", "results/ms_r3/supervised_screen/premise_check.json", "56,000 steps, bin-balanced"),
    ("M3-24", "MS-R3 supervised screen, medians per cell", "results/ms_r3/supervised_screen/summary_median.csv", ""),
    ("M3-25", "MS-R3 blind recomputation", "results/ms_r3/analysis/blind_recomputation.txt", ""),
    ("M3-26", "MS-R3 analysis info", "results/ms_r3/analysis/analysis_info.json", ""),
    ("M3-27", "MS-R3 fact-check ledger", "reports/ms/r3/pi_record/01_factcheck.md", ""),
    ("M3-28", "MS-R3 RL-actor diagnostics summary", "results/ms_r3/rl_actor_diagnostics/summary.txt", ""),
    ("M3-29", "MS-R3 noise-landing table", "results/ms_r3/analysis/noise_landing.csv", ""),
    ("M3-30", "MS-R3 stage-1 paired against t1", "results/ms_r3/analysis/stage1_vs_t1.csv", "descriptive"),
    ("M3-31", "MS-R3 secondary paired changes against t1", "results/ms_r3/analysis/paired_secondary.csv", ""),
    ("M3-32", "PI sandbox script (not evidence)", "reports/ms/r3/pi_record/sandbox_fit_tip.py", "PI side, 3 seeds"),
    ("M3-33", "PI sandbox output (not evidence)", "reports/ms/r3/pi_record/sandbox_fit_tip_results.jsonl", "PI side, 3 seeds"),
    ("M3-34", "MS-R3 stage-1 table", "results/ms_r3/analysis/stage1.csv", ""),
    ("M3-35", "MS-R3 housekeeping record", "reports/ms/r3/00_housekeeping.md", ""),
]
PI: List[Tuple[str, str, str, str]] = [
    ("PI-01", "PI prompt 17: MS-R1 (verbatim)", "reports/ms/r1/pi_record/17_ms_r1_prompt.md", ""),
    ("PI-02", "PI reply at gate G1 (verbatim)", "reports/ms/r1/pi_record/18_g1_reply.md", ""),
    ("PI-03", "PI prompt 19: MS-R2 (verbatim)", "reports/ms/r2/pi_record/19_ms_r2_prompt.md", ""),
    ("PI-04", "PI prompt 20: MS-R3 (verbatim)", "reports/ms/r3/pi_record/20_ms_r3_prompt.md", ""),
]
LOCAL_PI = [
    ("PI-05", "PI prompt 21 (this round) with Appendix A, the PI's plan note", "pi_record/21_t2_status_pack_prompt.md",
     "Appendix A; the prompt is the only record of the plan note (Multistage100526.docx is not in the repository)"),
    ("PI-06", "PI prompt 21, Appendix B: PI-side reading after MS-R3 (input, not evidence)", "pi_record/21_t2_status_pack_prompt.md",
     "Appendix B; every number in it is an input to verify"),
]
# figures that are copies: (id, title, source path, round, branch, conditions)
FIG_COPIES = [
    ("FIG-01", "MS-R3 learned tie profiles, every run and the seed median", "results/ms_r3/analysis/figures/tie_profile_runs.png", "MS-R3", "ms-r3", DEV + "; 12 arms, terminal freeze"),
    ("FIG-02", "MS-R3 decomposition along the run, bin-balanced starts", "results/ms_r3/analysis/figures/trajectory_decomposition_bb.png", "MS-R3", "ms-r3", DEV + "; local updates 1800-2800"),
    ("FIG-03", "MS-R3 decomposition along the run, stratified starts", "results/ms_r3/analysis/figures/trajectory_decomposition_st.png", "MS-R3", "ms-r3", DEV + "; local updates 1800-2800"),
    ("FIG-04", "MS-R3 paired |peak error| differences against t1", "results/ms_r3/analysis/figures/paired_abs_peak_vs_t1.png", "MS-R3", "ms-r3", DEV + "; primary criterion, part (a)"),
    ("FIG-05", "MS-R3 noise-landing paired differences (s = 16 vs s = 1)", "results/ms_r3/analysis/figures/paired_noise_landing.png", "MS-R3", "ms-r3", DEV),
    ("FIG-06", "MS-R3 first-layer d-weights of the RL actors", "results/ms_r3/analysis/figures/first_layer_d_weights.png", "MS-R3", "ms-r3", DEV),
    ("FIG-07", "MS-R2 decomposition along the run", "results/ms_r2/analysis/figures/trajectory_decomposition.png", "MS-R2", "ms-r2", DEV + "; six arms"),
    ("FIG-08", "MS-R2 remainder change against smoothing change", "results/ms_r2/analysis/figures/scatter_remainder_vs_smoothing_change.png", "MS-R2", "ms-r2", DEV),
    ("FIG-09", "MS-R1 calibration: R and Delta against the peak error", "reports/ms/r1/figures/cal_fig1_scatter_R_Delta_vs_peak.png", "MS-R1", "ms-r1", "v2.0 weight exports (calibration)"),
    ("FIG-10", "100526 pack: end-of-Phase-A profile (seed 30510)", "reports/t2_refine_100526/figures/FG-13_fig3_endA_profile.png", "R2b", "t2-refine-pack", "T2R:FG-13; v2.0 confirmation seeds"),
    ("FIG-11", "100526 pack: stage-2 peak trajectory, q = 50", "reports/t2_refine_100526/figures/FG-12_fig1_peak_trajectory_q50.png", "R2b", "t2-refine-pack", "T2R:FG-12"),
]
FIG_GENERATED = [
    ("FIG-12", "F1: |peak error| per run, every arm of MS-R1..MS-R3 and the v2.0 confirmation", "FIG-12_F1_abs_peak_per_run.png"),
    ("FIG-13", "F2: d = 0 gap split into smoothing part and remainder, per arm", "FIG-13_F2_gap_decomposition.png"),
]
# cited in place from the 100526 pack: T2R id -> (title, round, conditions)
T2R_USED = {
    "PL-01": "Locked protocol v2.0 (JSON)", "PL-02": "Locked protocol v2.0 (Markdown)",
    "PL-03": "protocols/LOCK: lock records", "CF-01": "v2.0 confirmation verdict",
    "CF-02": "v2.0 confirmation pass counts", "CF-03": "v2.0 confirmation per-run table",
    "CF-04": "v2.0 confirmation stage-1 summary", "CF-13": "gates.json of the failed run q=50 seed 30510",
    "CF-19": "v2.0 re-rehearsal per-run table", "R1-06": "R1 stage-2 criterion per arm",
    "R1-37": "R1 stage-2 per-run table", "R2B-02": "R2b criterion parts (a)/(b)",
    "R2B-03": "R2b per-run table", "R2B-04": "R2b tail statistics", "R2B-13": "R2b arm summary",
    "R2B-17": "Seed-30510 diagnostic: decomposition q = 50 (with the least-squares floor)",
    "R2B-18": "Seed-30510 diagnostic: decomposition, all 40 confirmation runs",
    "R2C-01": "R2c per-run table", "R2C-02": "R2c criterion parts (a)/(b)", "R2C-03": "R2c selection record",
    "R2C-04": "R2c tail statistics", "RR-01": "R1 summary report", "RR-03": "v2.0 lock/confirmation report",
    "RR-04": "R2b summary report", "RR-06": "R2c summary report", "RR-11": "R2b seed-30510 diagnostic report",
}
T2R_REPORT = ("T2R:100526report", "100526 summary report (sections 0, 3, 6; not an item of that pack)",
              "reports/t2_refine_100526/100526report.md")
T2R_ROUND = {"PL": "v2.0", "CF": "v2.0", "R1": "R1", "R2B": "R2b", "R2C": "R2c", "RR": "R1..R2c"}


def sh(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(REPO)] + list(args))


def git_blob(path: str, commit: str = SRC_COMMIT) -> Optional[bytes]:
    try:
        return sh("show", "%s:%s" % (commit, path))
    except subprocess.CalledProcessError:
        return None


def last_commit(path: str) -> str:
    return sh("log", "-1", "--format=%H", SRC_COMMIT, "--", path).decode().strip()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def norm_path(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def read_t2r_manifest() -> Dict[str, Dict[str, str]]:
    raw = git_blob(T2R_MANIFEST)
    if raw is None:
        raise SystemExit("STOP: 100526 manifest not found at %s" % SRC_COMMIT)
    return {r["item_id"]: r for r in csv.DictReader(io.StringIO(raw.decode()))}


def build(out: Path, pack_for_local: Path, used_in: Dict[str, str]) -> List[Dict[str, str]]:
    """Write out/evidence, out/figures and return the manifest rows."""
    rows: List[Dict[str, str]] = []
    ev = out / "evidence"
    figd = out / "figures"
    ev.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)

    def row(**kw: str) -> None:
        r = {k: "" for k in FIELDS}
        r.update(kw)
        r["used_in"] = used_in.get(r["item_id"], "")
        rows.append(r)

    for lst, rnd, br in ((M1, "MS-R1", "ms-r1"), (M2, "MS-R2", "ms-r2"), (M3, "MS-R3", "ms-r3"),
                         (PI, "PI record", "ms-r1/r2/r3")):
        for iid, title, path, cond in lst:
            b = git_blob(path)
            if b is None:
                raise SystemExit("STOP: %s (%s) is not tracked at %s" % (iid, path, SRC_COMMIT))
            lc = last_commit(path)
            r_br = br
            if iid.startswith("PI-"):
                r_br = {"PI-01": "ms-r1", "PI-02": "ms-r1", "PI-03": "ms-r2", "PI-04": "ms-r3"}[iid]
            if len(b) < MAX_COPY:
                dest = ev / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b)
                row(item_id=iid, title=title, round=rnd, source_branch=r_br, source_commit=lc,
                    source_path=path, sha256=sha(b), size_bytes=str(len(b)), status="copied",
                    copy_path="evidence/" + path, conditions=cond)
            else:
                row(item_id=iid, title=title, round=rnd, source_branch=r_br, source_commit=lc,
                    source_path=path, sha256=sha(b), size_bytes=str(len(b)),
                    status="referenced (tracked, large)", copy_path="", conditions=cond)
    for iid, title, rel, cond in LOCAL_PI:
        b = (pack_for_local / rel).read_bytes()
        row(item_id=iid, title=title, round="PI record", source_branch="t2-status-pack",
            source_commit="(this branch)", source_path="%s/%s" % (PACK_REL, rel), sha256=sha(b),
            size_bytes=str(len(b)), status="own record (not copied)", copy_path="", conditions=cond)

    t2r = read_t2r_manifest()
    for tid, title in T2R_USED.items():
        m = t2r[tid]
        copy_rel = T2R_EVIDENCE + m["source_path"]
        b = git_blob(copy_rel)
        if b is None or sha(b) != m["sha256"]:
            raise SystemExit("STOP: T2R:%s copy in the 100526 pack does not match its manifest" % tid)
        row(item_id="T2R:" + tid, title=title, round=T2R_ROUND[tid.split("-")[0]],
            source_branch="t2-refine-pack", source_commit=m["source_commit"],
            source_path=m["source_path"], sha256=m["sha256"], size_bytes=m["size_bytes"],
            status="cited in t2_refine_100526", copy_path=copy_rel,
            conditions="item %s of the 100526 pack" % tid)
    iid, title, path = T2R_REPORT
    b = git_blob(path)
    row(item_id=iid, title=title, round="R1..R2c", source_branch="t2-refine-pack",
        source_commit=last_commit(path), source_path=path, sha256=sha(b), size_bytes=str(len(b)),
        status="cited in t2_refine_100526", copy_path=path, conditions="folder report")

    for iid, title, path, rnd, br, cond in FIG_COPIES:
        path = norm_path(path)
        b = git_blob(path)
        if b is None:
            raise SystemExit("STOP: figure %s (%s) is not tracked" % (iid, path))
        name = "%s_%s" % (iid, Path(path).name)
        (figd / name).write_bytes(b)
        row(item_id=iid, title=title, round=rnd, source_branch=br, source_commit=last_commit(path),
            source_path=path, sha256=sha(b), size_bytes=str(len(b)), status="copied (figure)",
            copy_path="figures/" + name, conditions=cond)

    # generated figures and tables (they read the copies just written)
    import figures as figmod   # noqa: E402
    import tables as tabmod    # noqa: E402
    figmod.generate(out, figd)
    for iid, title, name in FIG_GENERATED:
        b = (figd / name).read_bytes()
        row(item_id=iid, title=title, round="MS-R1..R3, v2.0", source_branch="t2-status-pack",
            source_commit="(this branch)", source_path="%s/report_scripts/figures.py" % PACK_REL,
            sha256=sha(b), size_bytes=str(len(b)), status="generated (figure)",
            copy_path="figures/" + name,
            conditions="drawn only from evidence copies (this pack and the 100526 pack)")
    for tid, title, text, inputs in tabmod.render_all(out):
        b = text.encode()
        row(item_id=tid, title=title, round="derived", source_branch="t2-status-pack",
            source_commit="(this branch)",
            source_path="%s/report_scripts/tables.py" % PACK_REL, sha256=sha(b),
            size_bytes=str(len(b)), status="generated (table text)", copy_path="",
            conditions="inputs: " + inputs)
    return rows


def manifest_text(rows: List[Dict[str, str]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def sums_text(d: Path, skip: str = "SHA256SUMS") -> str:
    lines = []
    for p in sorted(x for x in d.rglob("*") if x.is_file() and x.name != skip):
        lines.append("%s  %s" % (sha(p.read_bytes()), p.relative_to(d).as_posix()))
    return "\n".join(lines) + "\n"


def used_in_map() -> Dict[str, str]:
    """Map item id -> report sections that cite it (read from report.md, if it exists)."""
    rep = PACK / "report.md"
    out: Dict[str, List[str]] = {}
    if not rep.exists():
        return {}
    sec = "-"
    for line in rep.read_text().splitlines():
        m = re.match(r"^#{2,3}\s+(\d+(?:\.\d+)?)[.\s]", line)
        if m:
            sec = m.group(1)
        for blk in re.findall(r"\[((?:M[123]-|PI-|FIG-|TBL-|T2R:)[^\]]*)\]", line):
            for iid in re.findall(r"(M[123]-\d\d|PI-\d\d|FIG-\d\d|TBL-[\w]+|T2R:[A-Za-z0-9\-]+)", blk):
                out.setdefault(iid, [])
                if sec not in out[iid]:
                    out[iid].append(sec)
    return {k: ", ".join(v) for k, v in out.items()}


def write_all(out: Path, pack_for_local: Path) -> None:
    rows = build(out, pack_for_local, used_in_map())
    (out / "evidence" / "manifest.csv").write_text(manifest_text(rows))
    (out / "evidence" / "SHA256SUMS").write_text(sums_text(out / "evidence"))
    (out / "figures" / "SHA256SUMS").write_text(sums_text(out / "figures"))


def tree(d: Path) -> Dict[str, str]:
    return {p.relative_to(d).as_posix(): sha(p.read_bytes()) for p in sorted(d.rglob("*")) if p.is_file()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="rebuild in a temp dir and compare byte for byte")
    a = ap.parse_args()
    if a.check:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            write_all(tmp, PACK)
            bad = 0
            for sub in ("evidence", "figures"):
                want, have = tree(tmp / sub), tree(PACK / sub)
                for k in sorted(set(want) | set(have)):
                    if want.get(k) != have.get(k):
                        bad += 1
                        print("DIFF %s/%s: rebuilt=%s in-place=%s" % (sub, k, want.get(k, "-")[:12], have.get(k, "-")[:12]))
            print("build_pack --check: %s (%d differences)" % ("PASS" if not bad else "FAIL", bad))
            return 1 if bad else 0
    # in-place rebuild: wipe only the two generated folders
    for sub in ("evidence", "figures"):
        if (PACK / sub).exists():
            shutil.rmtree(PACK / sub)
    write_all(PACK, PACK)
    n = sum(1 for _ in (PACK / "evidence").rglob("*") if _.is_file())
    print("built %s: %d evidence files, %d figure files" % (PACK_REL, n, sum(1 for _ in (PACK / "figures").glob("*"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
