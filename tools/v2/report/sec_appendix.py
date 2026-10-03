"""Appendix tables T58 (compute) and T59 (commands) and the key numbers K01-K19.

Key numbers are computed directly from the source files (never from rounded report text); each row
names its source file, selector and computation.
"""

from __future__ import annotations

import glob
import json
import re
import statistics
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import common as C
import studies as S

MOD = "sec_appendix.py"
RPT = "reports/v2"
CONF = "results/v2_T2_locked/confirmation_analysis"
PIL = "results/v2_pilots"
LKD = "results/v2_T2_locked"
E2STAR0 = {50: 70.0, 60: 58.33333333333333}   # checked against g2_at_0 in the per-run records below
DW = 4.0


def _r(sub: str, desc: str, value: Any, unit: str, norm: str, tier: str, q: Any, sf: str, sel: str, comp: str = "as is") -> Dict[str, Any]:
    return {"sub": sub, "description": desc, "value": value, "unit": unit, "normalization": norm, "tier": tier, "q": q,
            "source_file": sf, "selector": sel, "computation": comp}


# ---------------------------------------------------------------------------------------------
# T58 compute
# ---------------------------------------------------------------------------------------------

def _cpu_and_wall(study: str) -> Dict[str, Any]:
    """Process CPU seconds (sum of phase_timing[*].process_cpu_sec) and wall seconds per run of a study."""
    cpu: List[float] = []
    wall: List[float] = []
    n_missing_cpu = 0
    files = []
    for q, seed, arm, rd in S.iter_runs(study):
        try:
            st = S.status(rd)
            wall.append(float(st["total_wall_sec"]))
            files.append(C.src(f"{rd}/status.json"))
        except FileNotFoundError:
            continue
        try:
            sm = S.read_json(f"{rd}/v2_run_summary.json")
            files.append(C.src(f"{rd}/v2_run_summary.json"))
            cpu.append(sum(float(v["process_cpu_sec"]) for v in sm["phase_timing"].values()))
        except (FileNotFoundError, KeyError):
            n_missing_cpu += 1
    return {"cpu": cpu, "wall": wall, "n_missing_cpu": n_missing_cpu, "files": files}


LAUNCH = {
    "pilot1": f"{PIL}/pilot1/launch_20261001_185352.json", "pilot2": f"{PIL}/pilot2/launch_20261001_225739.json",
    "pilot3": f"{PIL}/pilot3/launch_20261001_233012.json", "phaseA_ext": f"{PIL}/phaseA_ext/launch_20261001_233012.json",
    "pilot4_A": f"{PIL}/pilot4_A/launch_20261002_012525.json", "pilot4_B": f"{PIL}/pilot4_B/launch_20261002_012525.json",
    "pilot4_B_rerun_clean": f"{PIL}/pilot4_B_rerun_clean/launch_20261002_024319.json",
    "locked_check2_phaseB": f"{LKD}/locked_check2_phaseB/launch_20261002_025428.json",
}
# locked studies were launched with `xargs -P N` (commands in the reports); environment files
LOCKED_ENV = {
    "rehearsal": dict(workers=20, env=f"{LKD}/rehearsal_launch_load.txt", note="xargs -P 20 (protocol_lock_and_rehearsal.md section 7)"),
    "rehearsal_v1_1": dict(workers=20, env=f"{LKD}/rehearsal_v1_1/launch_env.txt", note="xargs -P 20 (protocol_v1_1_confirmation.md section 7)"),
    "confirmation": dict(workers=40, env=f"{LKD}/confirmation/launch_env.txt", note="xargs -P 40 (protocol_v1_1_confirmation.md section 7)"),
}


def _loadavg(text: str) -> Optional[List[float]]:
    m = re.search(r"load average:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", text)
    return [float(m.group(i)) for i in (1, 2, 3)] if m else None


def build_t58(pack: C.Pack) -> pd.DataFrame:
    """T58: runs, wall time per run, parallelism, nproc and load at launch, total CPU-hours, per study."""
    rows: List[Dict[str, Any]] = []
    srcs: List[Any] = []
    for study in ["pilot1", "pilot2", "pilot3", "phaseA_ext", "pilot4_A", "pilot4_B", "pilot4_B_rerun_clean", "rehearsal",
                  "locked_check2_phaseB", "rehearsal_v1_1", "confirmation"]:
        reg = S.STUDIES[study]
        cw = _cpu_and_wall(study)
        srcs += cw["files"]
        rec: Dict[str, Any] = {"study": reg["title"], "study_key": study, "runs": len(cw["wall"])}
        w = np.asarray(cw["wall"])
        rec.update({"run_wall_s_min": float(w.min()), "run_wall_s_median": float(np.median(w)), "run_wall_s_max": float(w.max())})
        if study in LAUNCH:
            lj = S.read_json(LAUNCH[study])
            srcs.append(C.src(LAUNCH[study]))
            lw = np.asarray([r["wall_sec"] for r in lj["runs"]])
            rec.update({"parallelism": int(lj["workers"]), "nproc": int(lj["nproc"]), "load_at_launch_1_5_15min": " / ".join(f"{x:.2f}" for x in lj["loadavg_at_start"]),
                        "load_at_end_1_5_15min": " / ".join(f"{x:.2f}" for x in lj.get("loadavg_at_end", [])),
                        "launcher_wall_s_min_median_max": f"{lw.min():.1f} / {np.median(lw):.1f} / {lw.max():.1f}",
                        "launch_record": LAUNCH[study], "parallelism_basis": "launch record (workers)"})
        else:
            le = LOCKED_ENV[study]
            txt = C.abspath(le["env"]).read_text()
            srcs.append(C.src(le["env"]))
            la = _loadavg(txt)
            m = re.search(r"nproc:?\s*(\d+)", txt) or re.match(r"\s*(\d+)\s*\n", txt)
            nproc = int(m.group(1)) if m else 64   # v1.0 rehearsal: environment file holds uptime only; nproc = 64 from the report text
            rec.update({"parallelism": le["workers"], "nproc": nproc,
                        "load_at_launch_1_5_15min": " / ".join(f"{x:.2f}" for x in la) if la else "UNKNOWN", "load_at_end_1_5_15min": "not recorded",
                        "launcher_wall_s_min_median_max": "not recorded", "launch_record": le["env"], "parallelism_basis": le["note"] + ("" if m else "; nproc = 64 from protocol_lock_and_rehearsal.md section 4 (source: report text)")})
        cpu = np.asarray(cw["cpu"])
        rec.update({"cpu_s_per_run_median": float(np.median(cpu)) if cpu.size else "UNKNOWN",
                    "total_cpu_hours": float(cpu.sum() / 3600.0) if cpu.size else "UNKNOWN",
                    "n_runs_with_cpu_record": int(cpu.size),
                    "total_wall_hours": float(w.sum() / 3600.0),
                    "notes": "CPU = sum over phases of phase_timing.process_cpu_sec in v2_run_summary.json (final-tier evaluation, band sweep and "
                             "output writing after the last phase are not in it); wall = status.json total_wall_sec (single-threaded processes: wall >= CPU)"})
        rows.append(rec)
    df = pd.DataFrame(rows)
    tot = {"study": "all listed studies", "study_key": "total", "runs": int(df["runs"].sum()),
           "total_cpu_hours": float(df["total_cpu_hours"].astype(float).sum()), "total_wall_hours": float(df["total_wall_hours"].sum()),
           "notes": "sum over the rows above"}
    df = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
    docs = {c: {"definition": d, "units": u, "normalization": "none", "tier": "n/a", "source": "status.json, v2_run_summary.json, launch records"}
            for c, d, u in [
                ("study", "Study", ""), ("study_key", "Key of the study in the registry", ""), ("runs", "Training runs with a status.json", "runs"),
                ("run_wall_s_min", "Minimum total wall time of a run (status.json total_wall_sec)", "seconds"),
                ("run_wall_s_median", "Median total wall time of a run", "seconds"), ("run_wall_s_max", "Maximum total wall time of a run", "seconds"),
                ("parallelism", "Concurrent single-threaded processes of the launch", "processes"),
                ("nproc", "Logical CPUs of the machine (nproc)", "count"),
                ("load_at_launch_1_5_15min", "Load average (1, 5, 15 min) when the launch started", "load"),
                ("load_at_end_1_5_15min", "Load average at the end of the launch (launch record)", "load"),
                ("launcher_wall_s_min_median_max", "Launcher wall time per run, min / median / max (launch record)", "seconds"),
                ("launch_record", "File with the launch record / environment", "path"),
                ("parallelism_basis", "Where the parallelism comes from", "text"),
                ("cpu_s_per_run_median", "Median process CPU seconds per run (sum of phase_timing process_cpu_sec)", "seconds"),
                ("total_cpu_hours", "Total process CPU hours over the runs (phase_timing process_cpu_sec)", "hours"),
                ("n_runs_with_cpu_record", "Runs that have a phase_timing CPU record", "runs"),
                ("total_wall_hours", "Sum of the runs' total wall times", "hours"), ("notes", "Definitions and caveats", "text")]}
    pack.table("T58", df, status="generated", sources=srcs, script=f"{MOD}:build_t58", tier="n/a", docs=docs,
               notes="per-run times and CPU from status.json and v2_run_summary.json; parallelism/nproc/load from launch_*.json (pilots) or the launch "
                     "environment files and the xargs commands in the reports (locked runs)")
    return df


# ---------------------------------------------------------------------------------------------
# T59 commands
# ---------------------------------------------------------------------------------------------

REPORT_ORDER = [
    ("Phase 0 audit", "phase0_audit.md"), ("Phase 1 verifier", "phase1_verifier.md"),
    ("Phase 2 opening checks", "phase2_opening_checks.md"), ("Phase 2 infrastructure", "phase2_infra.md"),
    ("dReach reach-mask check", "dreach_reach_mask_check.md"), ("Pilot 1", "pilot1_reward_estimator.md"),
    ("Pilot 2", "pilot2_freeze.md"), ("Pilot 3", "pilot3_continuation_mode.md"), ("Phase A extension", "phaseA_ext.md"),
    ("Pilot 4", "pilot4_stabilization.md"), ("v1.0 lock, rehearsal, cusp", "protocol_lock_and_rehearsal.md"),
    ("v1.1 lock, re-rehearsal, confirmation", "protocol_v1_1_confirmation.md"),
]


def _command_blocks(path: str) -> List[Dict[str, str]]:
    """Fenced code blocks that sit under a heading containing 'reproduce' (any level)."""
    lines = C.abspath(path).read_text(encoding="utf-8").splitlines()
    out: List[Dict[str, str]] = []
    head, in_sec, sec_level, i = "", False, 0, 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^(#+)\s+(.*)$", ln)
        if m:
            level = len(m.group(1))
            if in_sec and level <= sec_level:
                in_sec = False
            if re.search(r"reproduc", m.group(2), re.I) and "command" in m.group(2).lower() or m.group(2).strip().lower() == "reproduce":
                in_sec, sec_level, head = True, level, m.group(2).strip()
        if in_sec and ln.strip().startswith("```"):
            j = i + 1
            body = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                body.append(lines[j])
                j += 1
            out.append({"section": head, "command": "\n".join(body).strip(), "line": str(i + 1)})
            i = j
        i += 1
    return out


def build_t59(pack: C.Pack) -> None:
    """T59: the commands to reproduce, collected from every report, in study order."""
    rows: List[Dict[str, Any]] = []
    srcs = []
    n = 0
    for study, rep in REPORT_ORDER:
        path = f"{RPT}/{rep}"
        srcs.append(C.src(path))
        blocks = _command_blocks(path)
        if not blocks:
            rows.append({"order": n + 1, "study": study, "command": "UNKNOWN: this report lists no commands", "report": path, "section": "",
                         "line": "", "note": "see the launch record argv in the study's results root where one exists (T58 lists the records)"})
            pack.unknown_value("T59", f"commands to reproduce: {study}",
                               f"{path} has no fenced command block under a 'reproduce' heading (the row says so; no command is inferred)")
            n += 1
            continue
        for b in blocks:
            n += 1
            rows.append({"order": n, "study": study, "command": b["command"], "report": path, "section": b["section"], "line": int(b["line"]),
                         "note": ""})
    # studies whose report has no command block: the launch record's argv is a file-backed command
    extra = []
    for study, key in [("Pilot 3", "pilot3")]:
        lj = S.read_json(LAUNCH[key])
        srcs.append(C.src(LAUNCH[key]))
        extra.append({"order": 0, "study": study, "command": " ".join(lj["argv"]), "report": LAUNCH[key], "section": "launch record argv",
                      "line": "", "note": "source: launch record (not a report); the report names tmux session v2_pilot3 with 40 workers"})
    df = pd.DataFrame(rows)
    ex = pd.DataFrame(extra)
    for _, r in ex.iterrows():
        idx = df.index[(df["study"] == r["study"])][-1]
        df = pd.concat([df.iloc[:idx + 1], pd.DataFrame([r]), df.iloc[idx + 1:]], ignore_index=True)
    df["order"] = np.arange(1, len(df) + 1)
    docs = {"order": ("Order of the command (study order, then position in the report)", "count"), "study": ("Study the command belongs to", ""),
            "command": ("Command as written in the report (fenced block), or the launch record argv", "shell"),
            "report": ("Report (or launch record) the command was collected from", "path"), "section": ("Heading of the section", "text"),
            "line": ("Line of the fenced block in the report", "line number"), "note": ("Remark", "text")}
    pack.table("T59", df, status="generated", sources=srcs, script=f"{MOD}:build_t59", tier="n/a",
               docs={k: {"definition": d, "units": u, "normalization": "none", "tier": "n/a", "source": "reports/v2/*.md (source: report text)"} for k, (d, u) in docs.items()},
               notes="fenced code blocks under the 'Commands to reproduce' headings of the reports (source: report text), in study order; Pilot 3 has no "
                     "command section, so its launch-record argv is added and marked")


# ---------------------------------------------------------------------------------------------
# key numbers
# ---------------------------------------------------------------------------------------------

def _med_max(x: pd.Series) -> Dict[str, float]:
    v = x.to_numpy(dtype=float)
    return {"median": float(np.median(v)), "max": float(v.max()), "min": float(v.min()), "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75))}


def build_k_confirmation(pack: C.Pack) -> None:
    pc = S.read_csv(f"{CONF}/pass_counts.csv")
    pr = S.read_csv(f"{CONF}/per_run.csv")
    rm = S.read_csv(f"{CONF}/reported_metrics.csv")
    s1 = S.read_csv(f"{CONF}/s1_summary.csv")
    vd = S.read_json(f"{CONF}/verdict.json")
    dist = S.read_csv(f"{CONF}/distributions.csv")
    PCF, PRF, RMF, S1F, VDF, DSF = (f"{CONF}/pass_counts.csv", f"{CONF}/per_run.csv", f"{CONF}/reported_metrics.csv",
                                    f"{CONF}/s1_summary.csv", f"{CONF}/verdict.json", f"{CONF}/distributions.csv")
    qs = (50, 60)
    # K01
    rows = []
    for q in qs:
        r = pc[pc.q == q].iloc[0]
        sel = f"q={q}"
        rows += [_r(f"q{q}.n_pass", f"Primary passes (G-A and G-F and G-N) of 20 runs, q={q}", int(r["n_pass"]), "runs of 20", "none", "final", q, PCF, f"{sel} / n_pass"),
                 _r(f"q{q}.n_expected", f"Runs expected, q={q}", int(r["n_expected"]), "runs", "none", "final", q, PCF, f"{sel} / n_expected"),
                 _r(f"q{q}.cp95_lo", f"Exact (Clopper-Pearson) 95% CI of the pass rate, lower end, q={q}", float(r["cp95_lo"]), "probability", "none", "final", q, PCF, f"{sel} / cp95_lo"),
                 _r(f"q{q}.cp95_hi", f"Exact 95% CI of the pass rate, upper end, q={q}", float(r["cp95_hi"]), "probability", "none", "final", q, PCF, f"{sel} / cp95_hi"),
                 _r(f"q{q}.rule", f"Pass rule, q={q}", str(r["rule"]), "text", "none", "final", q, PCF, f"{sel} / rule"),
                 _r(f"q{q}.q_passes_rule", f"q={q} passes the rule", str(r["q_passes_rule"]), "yes/no", "none", "final", q, PCF, f"{sel} / q_passes_rule")]
    rows += [_r("overall.verdict", "Overall confirmation verdict (both q must pass)", str(vd["overall"]), "PASS/FAIL", "none", "final", "50, 60", VDF, "overall"),
             _r("overall.rule", "Rule of the verdict", str(vd["rule"]), "text", "none", "final", "50, 60", VDF, "rule")]
    pack.numbers_item("K01", rows, status="found", script=f"{MOD}:build_k_confirmation")

    # K02-K05 distribution numbers from per_run.csv (final tier unless stated)
    def mm_rows(kid: str, desc: str, col: str, unit: str, norm: str, tier: str, scale: float = 1.0, abs_: bool = False):
        out = []
        for q in qs:
            x = pr.loc[pr.q == q, col].abs() if abs_ else pr.loc[pr.q == q, col]
            x = x * (E2STAR0[q] if scale == -1 else 1.0)
            st = _med_max(x)
            comp_extra = f"; times e2*(0) = {E2STAR0[q]:.6g}" if scale == -1 else ""
            for k in ("median", "max"):
                out.append(_r(f"q{q}.{k}", f"{desc}, {k} over the 20 seeds, q={q}", st[k], unit, norm, tier, q, PRF, f"q={q} rows / {col}",
                              f"{k} over 20 rows (numpy){' of |value|' if abs_ else ''}{comp_extra}"))
        return out

    pack.numbers_item("K02", mm_rows("K02", "Gmax_full/dW", "gmax_final", "dW", "divided by Delta W", "final"), status="derived", script=f"{MOD}:build_k_confirmation")
    pack.numbers_item("K03", mm_rows("K03", "eta_2/dW", "eta_final", "dW", "divided by Delta W", "final"), status="derived", script=f"{MOD}:build_k_confirmation")
    pack.numbers_item("K04", mm_rows("K04", "RMSE over |d| < 2q divided by e2*(0)", "rmse", "fraction of e2*(0)", "divided by e2*(0)", "tier-independent"),
                      status="derived", script=f"{MOD}:build_k_confirmation")
    rows = mm_rows("K05", "Stage-2 tail mean divided by e2*(0)", "tail", "fraction of e2*(0)", "divided by e2*(0)", "tier-independent")
    rows += [dict(r, sub=r["sub"].replace("q", "raw.q", 1), description=r["description"].replace("divided by e2*(0)", "raw (effort units)"),
                  unit="effort units [0, 100]", normalization="raw", value=r["value"] * E2STAR0[int(r["q"])],
                  computation=r["computation"] + "; times e2*(0) = %.6g (raw tail mean = normalized x e2*(0))" % E2STAR0[int(r["q"])])
             for r in mm_rows("K05", "Stage-2 tail mean", "tail", "effort units", "raw", "tier-independent")]
    pack.numbers_item("K05", rows, status="derived", script=f"{MOD}:build_k_confirmation")

    # K06 peak error at d=0
    rows = []
    for q in qs:
        x = rm.loc[rm.q == q, "A_stage2_peak_rel_err_signed"].to_numpy(dtype=float)
        st = {"median": np.median(x), "min": x.min(), "max": x.max()}
        sf, sel = RMF, f"q={q} rows / A_stage2_peak_rel_err_signed"
        for k in ("median", "min", "max"):
            rows.append(_r(f"q{q}.{k}", f"Stage-2 peak error at d = 0 (signed), {k} over 20 seeds, q={q}", float(st[k]), "fraction of e2*(0)", "divided by e2*(0), signed",
                           "tier-independent", q, sf, sel, f"{k} over 20 rows (numpy)"))
        rows.append(_r(f"q{q}.n_within_5pct", f"Runs with |peak error at d = 0| <= 0.05, q={q}", int((np.abs(x) <= 0.05).sum()), "runs of 20", "none", "tier-independent", q, sf, sel,
                       "count of |value| <= 0.05"))
    pack.numbers_item("K06", rows, status="derived", script=f"{MOD}:build_k_confirmation")

    # K07 stage-1 error
    rows = []
    for q in qs:
        r = s1[s1.q == q].iloc[0]
        sel = f"q={q}"
        x = pr.loc[pr.q == q, "s1"].to_numpy(dtype=float)
        rows += [
            _r(f"q{q}.mean_signed", f"Mean signed stage-1 relative error, q={q}", float(r["mean_signed"]), "fraction of e1*(0)", "divided by e1*(0), signed", "tier-independent", q, S1F, f"{sel} / mean_signed"),
            _r(f"q{q}.boot95_lo", f"95% percentile bootstrap CI of the mean signed error, lower end, q={q}", float(r["boot95_lo"]), "fraction of e1*(0)", "divided by e1*(0)", "tier-independent", q, S1F,
               f"{sel} / boot95_lo", f"as is (10,000 resamples, numpy seed {int(r['bootstrap_seed'])})"),
            _r(f"q{q}.boot95_hi", f"95% percentile bootstrap CI of the mean signed error, upper end, q={q}", float(r["boot95_hi"]), "fraction of e1*(0)", "divided by e1*(0)", "tier-independent", q, S1F,
               f"{sel} / boot95_hi", f"as is (10,000 resamples, numpy seed {int(r['bootstrap_seed'])})"),
            _r(f"q{q}.sd_signed", f"SD (ddof = 1) of the signed stage-1 error, q={q}", float(r["sd_signed"]), "fraction of e1*(0)", "divided by e1*(0)", "tier-independent", q, S1F, f"{sel} / sd_signed"),
            _r(f"q{q}.median_signed", f"Median signed stage-1 error, q={q}", float(r["median_signed"]), "fraction of e1*(0)", "divided by e1*(0)", "tier-independent", q, S1F, f"{sel} / median_signed"),
            _r(f"q{q}.S1_pass", f"S1 passes (|stage-1 error| <= 0.10) of 20, q={q}", int(r["S1_pass"]), "runs of 20", "none", "tier-independent", q, S1F, f"{sel} / S1_pass"),
            _r(f"q{q}.n_within_5pct", f"Runs with |stage-1 error| <= 0.05, q={q}", int((x <= 0.05).sum()), "runs of 20", "none", "tier-independent", q, PRF, f"q={q} rows / s1",
               "count of s1 <= 0.05"),
        ]
    pack.numbers_item("K07", rows, status="derived", script=f"{MOD}:build_k_confirmation")

    # K08 max |dev - final|
    rows = []
    for q in qs:
        for col, nm in (("eta_dev_minus_final", "eta_2"), ("gmax_dev_minus_final", "Gmax_full")):
            x = np.abs(pr.loc[pr.q == q, col].to_numpy(dtype=float))
            rows.append(_r(f"q{q}.{nm}", f"max |dev - final| of {nm}/dW over 20 seeds, q={q}", float(x.max()), "dW", "divided by Delta W", "development minus final", q, PRF,
                           f"q={q} rows / {col}", "max of |value| (G-N threshold 0.001)"))
    for col, nm in (("eta_dev_minus_final", "eta_2"), ("gmax_dev_minus_final", "Gmax_full")):
        rows.append(_r(f"all.{nm}", f"max |dev - final| of {nm}/dW over all 40 runs", float(np.abs(pr[col].to_numpy(dtype=float)).max()), "dW", "divided by Delta W", "development minus final", "50, 60", PRF,
                       f"all rows / {col}", "max of |value| over both q"))
    pack.numbers_item("K08", rows, status="derived", script=f"{MOD}:build_k_confirmation")

    # K09 v1.0 outcome
    rows = []
    for q in qs:
        r = pc[pc.q == q].iloc[0]
        rows.append(_r(f"q{q}.n_v1_0_run_pass", f"Runs passing the v1.0 outcome (G-A and v1.0 G-F with the stage-1 criterion), q={q}", int(r["n_v1_0_run_pass"]), "runs of 20", "none", "final", q, PCF,
                       f"q={q} / n_v1_0_run_pass", "as is (reported only; decides nothing)"))
        fail = pr[(pr.q == q) & (~pr["v1_0_run_pass"].astype(bool))]
        rows.append(_r(f"q{q}.v1_0_failing_seeds", f"Seeds failing the v1.0 outcome, q={q}", ", ".join(str(s) for s in fail["seed"]) or "none", "seeds", "none", "final", q, PRF,
                       f"q={q} rows with v1_0_run_pass = False / seed", "list"))
    pack.numbers_item("K09", rows, status="derived", script=f"{MOD}:build_k_confirmation")

    # cross-check with the v1.1 report distribution table (medians and max of the gate metrics)
    chk = []
    for q in qs:
        for m in ("eta_final", "rmse", "tail", "gmax_final"):
            d = dist[(dist.q == q) & (dist.metric == m)].iloc[0]
            col = {"eta_final": "eta_final", "rmse": "rmse", "tail": "tail", "gmax_final": "gmax_final"}[m]
            st = _med_max(pr.loc[pr.q == q, col])
            chk.append({"q": q, "metric": m, "median": st["median"], "max": st["max"]})
            assert abs(st["median"] - float(d["median"])) < 1e-12 and abs(st["max"] - float(d["max"])) < 1e-12, (q, m)
    pack.crosscheck("K02", pd.DataFrame(chk), f"{RPT}/protocol_v1_1_confirmation.md", header_has=["q", "metric", "median", "max"], key_map={"q": "q", "metric": "metric"},
                    value_map={"median": "median", "max": "max"}, heading_has="Distributions", label="medians and maxima of eta_final, rmse, tail, gmax_final (K02-K05)")


def build_k_pilots(pack: C.Pack) -> None:
    # K10 Pilot 1
    ft = S.read_csv(f"{PIL}/pilot1/analysis/final_table.csv")
    ps = S.read_csv(f"{PIL}/pilot1/analysis/paired_summary.csv")
    FT, PSF = f"{PIL}/pilot1/analysis/final_table.csv", f"{PIL}/pilot1/analysis/paired_summary.csv"
    fin = S.final_tier_columns("pilot1")
    rows = []
    for q in (50, 60):
        for arm in ("sampled", "expected"):
            g = ft[(ft.q == q) & (ft.arm == arm)]
            gf = fin[(fin.q == q) & (fin.arm == arm)]
            rows += [
                _r(f"q{q}.{arm}.eta2_median_dev", f"Pilot 1 median eta_2/dW (development tier, u400), arm {arm}, q={q}", float(np.median(g.eta_T_over_dw)), "dW", "divided by Delta W", "development", q, FT,
                   f"q={q}, arm={arm} / eta_T_over_dw", "median over 10 seeds"),
                _r(f"q{q}.{arm}.eta2_median_final", f"Pilot 1 median eta_2/dW (final tier, u400), arm {arm}, q={q}", float(np.median(gf.final_tier__eta_T_over_dw)), "dW", "divided by Delta W", "final", q,
                   f"{PIL}/pilot1/q*/seed*/*/final_v2.json", f"q={q}, arm={arm} / final.eta_T_over_dw", "median over 10 seeds"),
                _r(f"q{q}.{arm}.peak_signed_median", f"Pilot 1 median signed peak error at d = 0 (u400), arm {arm}, q={q}", float(np.median(g.stage2_peak_rel_err_signed)), "fraction of e2*(0)",
                   "divided by e2*(0), signed", "tier-independent", q, FT, f"q={q}, arm={arm} / stage2_peak_rel_err_signed", "median over 10 seeds")]
        for met, nm in (("eta_T_over_dw", "eta_2"), ("stage2_peak_rel_err_abs", "|peak error|")):
            r = ps[(ps.q == q) & (ps.metric == met)].iloc[0]
            rows += [_r(f"q{q}.{nm}.n_favour_expected", f"Pilot 1 pairs (of 10) with expected better (smaller {nm}), q={q}", int(r["n_favour_expected"]), "pairs of 10", "none", "development" if nm == "eta_2" else "tier-independent", q, PSF,
                        f"q={q}, metric={met} / n_favour_expected"),
                     _r(f"q{q}.{nm}.median_diff", f"Pilot 1 median paired difference expected - sampled of {nm}, q={q}", float(r["median"]), "dW" if nm == "eta_2" else "fraction of e2*(0)", "as the metric",
                        "development" if nm == "eta_2" else "tier-independent", q, PSF, f"q={q}, metric={met} / median", "as is (bootstrap 10,000 resamples, numpy seed 20261001)")]
    pack.numbers_item("K10", rows, status="derived", script=f"{MOD}:build_k_pilots")

    # K11 Pilot 2
    f2 = S.read_csv(f"{PIL}/pilot2/analysis/final_table.csv")
    F2 = f"{PIL}/pilot2/analysis/final_table.csv"
    rows = []
    for q in (50, 60):
        for arm, short in (("A_joint", "A"), ("B1_frozen_allnorm", "B1"), ("B2_frozen_s1norm", "B2")):
            g = f2[(f2.q == q) & (f2.arm == arm)]
            for col, nm, unit, norm, tier in (("stage2_drift_cand_on_max", "on-path max stage-2 drift", "effort units", "none", "development"),
                                              ("stage2_drift_cand_off_max", "off-path max stage-2 drift", "effort units", "none", "development"),
                                              ("stage2_peak_rel_err_signed", "signed peak error at d = 0", "fraction of e2*(0)", "divided by e2*(0), signed", "tier-independent"),
                                              ("stage2_tail_mean", "stage-2 tail mean (raw)", "effort units [0, 100]", "raw", "tier-independent")):
                if arm != "A_joint" and "drift" in col:
                    pass  # frozen arms: 0 by construction, kept to show it
                rows.append(_r(f"q{q}.{short}.{col}", f"Pilot 2 median {nm} (u1000), arm {short}, q={q}", float(np.median(g[col])), unit, norm, tier, q, F2, f"q={q}, arm={arm} / {col}", "median over 10 seeds"))
    pack.numbers_item("K11", rows, status="derived", script=f"{MOD}:build_k_pilots")

    # K12 Pilot 3
    p3 = S.read_csv(f"{PIL}/pilot3/analysis/paired_summary.csv")
    st3 = S.read_csv(f"{PIL}/pilot3/analysis/stability.csv")
    P3, ST3 = f"{PIL}/pilot3/analysis/paired_summary.csv", f"{PIL}/pilot3/analysis/stability.csv"
    rows = []
    for q in (50, 60):
        for met in ("stage1_rel_err_abs", "learning_rel_abs", "EXP_root_over_dw", "dReach_over_dw", "within_run_sd_e1_last5"):
            r = p3[(p3.q == q) & (p3.metric == met)].iloc[0]
            unit = "effort units" if "within_run" in met else ("dW" if met.endswith("over_dw") else "fraction of e1*(0)")
            for k, col in (("median", "median"), ("ci_lo", "boot_ci95_lo"), ("ci_hi", "boot_ci95_hi"), ("n_mean_better", "n_mean_better")):
                rows.append(_r(f"q{q}.{met}.{k}", f"Pilot 3 paired difference mean - stochastic of {met}: {col}, q={q}", float(r[col]) if k != "n_mean_better" else int(r[col]), unit if k != "n_mean_better" else "pairs of 10",
                               "as the metric", "development", q, P3, f"q={q}, metric={met} / {col}", "as is (bootstrap 10,000 resamples, numpy seed 20261001)"))
        for arm, adir in (("stochastic", "B2_frozen_s1norm"), ("mean", "B2_frozen_s1norm_mean")):
            r = st3[(st3.q == q) & (st3.arm == adir)].iloc[0]
            rows.append(_r(f"q{q}.{arm}.median_within_run_sd_last5", f"Pilot 3 median within-run SD of e_hat_1(0) over the last 5 exports, arm {arm}, q={q}", float(r["median_within_run_sd_last5"]), "effort units",
                           "raw", "development", q, ST3, f"q={q}, arm={adir} / median_within_run_sd_last5"))
    pack.numbers_item("K12", rows, status="found", script=f"{MOD}:build_k_pilots")

    # K13 Phase A extension
    tb = S.read_csv(f"{PIL}/phaseA_ext/analysis/table_400_800_1200_1600.csv")
    TB = f"{PIL}/phaseA_ext/analysis/table_400_800_1200_1600.csv"
    rows = []
    for q in (50, 60):
        for met, unit, norm in (("stage2_peak_rel_err_signed", "fraction of e2*(0)", "divided by e2*(0), signed"), ("stage2_tail_mean", "effort units [0, 100]", "raw"),
                                ("sigma_effort_at_0_t2", "effort units [0, 100]", "raw")):
            sub = tb[(tb.q == q) & (tb.metric == met)].set_index("update")
            ups = (800, 1200, 1600) if met == "stage2_peak_rel_err_signed" else (400, 800, 1200, 1600)
            for u in ups:
                rows.append(_r(f"q{q}.{met}.u{u}", f"Phase A extension median {met} at u{u}, q={q}", float(sub.loc[u, "median"]), unit, norm, "tier-independent", q, TB,
                               f"q={q}, update={u}, metric={met} / median", "as is (median over 10 seeds)"))
            if met != "stage2_peak_rel_err_signed":
                rows.append(_r(f"q{q}.{met}.change_u400_to_u1600", f"Phase A extension change of the median {met} from u400 to u1600, q={q}", float(sub.loc[1600, "median"] - sub.loc[400, "median"]), unit, norm,
                               "tier-independent", q, TB, f"q={q}, metric={met} / median at update 1600 minus update 400", "difference of two medians"))
                med = [float(sub.loc[u, "median"]) for u in (400, 800, 1200, 1600)]
                rows.append(_r(f"q{q}.{met}.monotone_decreasing", f"Phase A extension: the median {met} decreases at every step u400 < u800 < u1200 < u1600, q={q}", bool(all(med[i] > med[i + 1] for i in range(3))),
                               "bool", "none", "tier-independent", q, TB, f"q={q}, metric={met} / median at the four updates", "all consecutive differences negative"))
    pack.numbers_item("K13", rows, status="derived", script=f"{MOD}:build_k_pilots")

    # K14 Pilot 4
    P4 = f"{PIL}/pilot4/analysis"
    pp = S.read_csv(f"{P4}/paired_summary.csv")
    pr_ = S.read_csv(f"{P4}/paired_summary_run_records.csv")
    sb = S.read_csv(f"{P4}/stability_2b.csv")
    rows = []
    for q in (50, 60):
        for met, unit in (("stage2_rmse_pos_over_g2_0", "fraction of e2*(0)"), ("stage2_tail_mean", "effort units [0, 100]")):
            r = pp[(pp.family == "2a") & (pp.comparison == "decay_minus_constant") & (pp.q == q) & (pp.metric == met) & (pp.a == "decay K=1")].iloc[0]
            for k, col in (("median", "median"), ("ci_lo", "boot_ci95_lo"), ("ci_hi", "boot_ci95_hi"), ("n_better", "n_better")):
                rows.append(_r(f"2a.q{q}.{met}.{k}", f"Pilot 4 section 2a decay - constant, last iterate (K=1), {met}: {col}, q={q}", float(r[col]) if k != "n_better" else int(r[col]),
                               unit if k != "n_better" else "pairs of 10", "as the metric", "tier-independent", q, f"{P4}/paired_summary.csv",
                               f"family=2a, comparison=decay_minus_constant, q={q}, metric={met}, a='decay K=1' / {col}", "as is (bootstrap 10,000 resamples, numpy seed 20261001)"))
        for met in ("within_run_sd_e1_last5", "within_run_range_e1_last5"):
            r = pr_[(pr_.family == "2b") & (pr_.q == q) & (pr_.metric == met)].iloc[0]
            for k, col in (("median", "median"), ("ci_lo", "boot_ci95_lo"), ("ci_hi", "boot_ci95_hi"), ("n_better", "n_better")):
                rows.append(_r(f"2b.q{q}.{met}.{k}", f"Pilot 4 section 2b decay - constant, {met}: {col}, q={q}", float(r[col]) if k != "n_better" else int(r[col]),
                               "effort units" if k != "n_better" else "pairs of 10", "raw", "development", q, f"{P4}/paired_summary_run_records.csv",
                               f"family=2b, comparison=decay_minus_constant, q={q}, metric={met} / {col}", "as is (bootstrap 10,000 resamples, numpy seed 20261001)"))
        for arm in ("constant", "decay"):
            r = sb[(sb.q == q) & (sb.arm == arm)].iloc[0]
            rows.append(_r(f"2b.q{q}.{arm}.across_seed_sd_final_e1", f"Pilot 4 section 2b across-seed SD of the final e_hat_1(0), arm {arm}, q={q}", float(r["across_seed_sd_final_e1"]), "effort units", "raw",
                           "development", q, f"{P4}/stability_2b.csv", f"q={q}, arm={arm} / across_seed_sd_final_e1"))
            rows.append(_r(f"2b.q{q}.{arm}.median_within_run_sd_last5", f"Pilot 4 section 2b median within-run SD over the last 5 exports, arm {arm}, q={q}", float(r["median_within_run_sd_last5"]), "effort units",
                           "raw", "development", q, f"{P4}/stability_2b.csv", f"q={q}, arm={arm} / median_within_run_sd_last5"))
    pack.numbers_item("K14", rows, status="found", script=f"{MOD}:build_k_pilots")


def build_k_misc(pack: C.Pack) -> None:
    # K15 smoothed-game share across studies
    rows = []
    sg = S.read_csv(f"{PIL}/pilot1/analysis/smoothed_game/per_run.csv")
    SGF = f"{PIL}/pilot1/analysis/smoothed_game/per_run.csv"
    for q in (50, 60):
        for arm in ("sampled", "expected"):
            x = sg[(sg.q == q) & (sg.arm == arm)].share_peak_gap_explained
            rows.append(_r(f"pilot1.q{q}.{arm}", f"Smoothed-game share of the d = 0 peak gap, Pilot 1 u400, arm {arm}, q={q}: median over 10 seeds", float(np.median(x)), "fraction of the gap", "none", "tier-independent",
                           q, SGF, f"q={q}, arm={arm} / share_peak_gap_explained", "median over 10 seeds"))
    tb = S.read_csv(f"{PIL}/phaseA_ext/analysis/table_400_800_1200_1600.csv")
    TB = f"{PIL}/phaseA_ext/analysis/table_400_800_1200_1600.csv"
    for q in (50, 60):
        for u in (400, 800, 1200, 1600):
            v = tb[(tb.q == q) & (tb["update"] == u) & (tb.metric == "share_peak_gap_explained")].iloc[0]["median"]
            rows.append(_r(f"ext.q{q}.u{u}", f"Smoothed-game share, Phase A extension u{u}, q={q}: median over 10 seeds", float(v), "fraction of the gap", "none", "tier-independent", q, TB,
                           f"q={q}, update={u}, metric=share_peak_gap_explained / median", "as is"))
    gp = S.read_csv(f"{LKD}/rehearsal_analysis/gates_per_run.csv")
    for q in (50, 60):
        x = gp[gp.q == q]["A_smoothed_share_peak_gap_d0"]
        rows.append(_r(f"rehearsal_v1_0.q{q}", f"Smoothed-game share, v1.0 rehearsal end of A (u1600), q={q}: median over 10 seeds", float(np.median(x)), "fraction of the gap", "none", "tier-independent", q,
                       f"{LKD}/rehearsal_analysis/gates_per_run.csv", f"q={q} rows / A_smoothed_share_peak_gap_d0", "median over 10 seeds"))
    for nm, d in (("rehearsal_v1_1", "rehearsal_v1_1_analysis"), ("confirmation", "confirmation_analysis")):
        rm = S.read_csv(f"{LKD}/{d}/reported_metrics.csv")
        for q in (50, 60):
            x = rm[rm.q == q]["A_smoothed_share_peak_gap_d0"]
            rows.append(_r(f"{nm}.q{q}", f"Smoothed-game share, {nm.replace('_', ' ')} end of A (u1600), q={q}: median over {len(x)} seeds", float(np.median(x)), "fraction of the gap", "none", "tier-independent", q,
                           f"{LKD}/{d}/reported_metrics.csv", f"q={q} rows / A_smoothed_share_peak_gap_d0", f"median over {len(x)} rows"))
    pack.numbers_item("K15", rows, status="derived", script=f"{MOD}:build_k_misc")

    # K16 supervised-fit floor
    fits = S.read_csv(f"{PIL}/pilot4/analysis/repr_floor/fits.csv")
    tw = S.read_csv(f"{PIL}/pilot4/analysis/repr_floor/three_way_peak_gap.csv")
    FF, TW = f"{PIL}/pilot4/analysis/repr_floor/fits.csv", f"{PIL}/pilot4/analysis/repr_floor/three_way_peak_gap.csv"
    rows = []
    for q in (50, 60):
        g = fits[fits.q == q]
        for col, nm in (("stage2_peak_rel_err_signed", "signed peak error at d = 0"), ("stage2_peak_locfree_rel_err", "location-free peak error")):
            x = g[col].to_numpy(dtype=float)
            rows += [_r(f"q{q}.{col}.median", f"Supervised-fit floor, {nm}, median over 5 inits, q={q}", float(np.median(x)), "fraction of e2*(0)", "divided by e2*(0), signed", "tier-independent", q, FF,
                        f"q={q} rows / {col}", "median over 5 fits (all 10 fits stopped at the 300,000-step cap: upper bounds)"),
                     _r(f"q{q}.{col}.max_abs", f"Supervised-fit floor, {nm}, max |value| over 5 inits, q={q}", float(np.abs(x).max()), "fraction of e2*(0)", "divided by e2*(0)", "tier-independent", q, FF,
                        f"q={q} rows / {col}", "max of |value| over 5 fits")]
        r = tw[(tw.q == q) & (tw.quantity == "supervised_floor_gap_d0")].iloc[0]
        rows += [_r(f"q{q}.floor_gap_d0.median", f"Supervised-fit floor of the d = 0 peak gap, median over 5 inits, q={q}", float(r["median"]), "effort units", "raw", "tier-independent", q, TW,
                    f"q={q}, quantity=supervised_floor_gap_d0 / median"),
                 _r(f"q{q}.floor_gap_d0.median_over_e2star0", f"Supervised-fit floor of the d = 0 peak gap divided by e2*(0), median, q={q}", float(r["median_over_e2star0"]), "fraction of e2*(0)",
                    "divided by e2*(0)", "tier-independent", q, TW, f"q={q}, quantity=supervised_floor_gap_d0 / median_over_e2star0")]
    pack.numbers_item("K16", rows, status="derived", script=f"{MOD}:build_k_misc")

    # K17 calibration
    cal = S.read_csv(f"{LKD}/calibration/calibration_locked.csv")
    CF = f"{LKD}/calibration/calibration_locked.csv"
    rows = []
    for q in (50, 60):
        for tier in ("final", "development"):
            a = cal[(cal.q == q) & (cal.policy == "analytic_eq") & (cal.tier == tier)].iloc[0]
            z = cal[(cal.q == q) & (cal.policy == "zero") & (cal.tier == tier)].iloc[0]
            sa, sz = f"q={q}, policy=analytic_eq, tier={tier}", f"q={q}, policy=zero, tier={tier}"
            rows += [_r(f"q{q}.{tier}.analytic_floor_gmax", f"Calibration floor: Gmax_full/dW of the analytic equilibrium, {tier} tier, q={q}", float(a["Gmax_full_over_dw"]), "dW", "divided by Delta W", tier, q, CF,
                        f"{sa} / Gmax_full_over_dw"),
                     _r(f"q{q}.{tier}.analytic_floor_t_d", f"Location (t*, d*) of the analytic-equilibrium floor, {tier} tier, q={q}", f"({int(a['Gmax_full_t'])}, {a['Gmax_full_d']:g})", "stage, gap", "none", tier, q, CF,
                        f"{sa} / Gmax_full_t, Gmax_full_d"),
                     _r(f"q{q}.{tier}.zero_gmax", f"Zero-effort policy Gmax_full/dW, {tier} tier, q={q}", float(z["Gmax_full_over_dw"]), "dW", "divided by Delta W", tier, q, CF, f"{sz} / Gmax_full_over_dw"),
                     _r(f"q{q}.{tier}.zero_gmax_t_d", f"Zero-effort Gmax_full location (t*, d*), {tier} tier, q={q}", f"({int(z['Gmax_full_t'])}, {z['Gmax_full_d']:g})", "stage, gap", "none", tier, q, CF,
                        f"{sz} / Gmax_full_t, Gmax_full_d"),
                     _r(f"q{q}.{tier}.zero_gmax_diff_vs_PI", f"Zero-effort Gmax_full/dW minus the PI-side reference ({z['pi_ref_Gmax']:g}), {tier} tier, q={q}", float(z["diff_vs_pi_Gmax"]), "dW", "divided by Delta W", tier,
                        q, CF, f"{sz} / diff_vs_pi_Gmax"),
                     _r(f"q{q}.{tier}.zero_root_gain", f"Zero-effort root gain EXP_root/dW, {tier} tier, q={q}", float(z["EXP_root_over_dw"]), "dW", "divided by Delta W", tier, q, CF, f"{sz} / EXP_root_over_dw"),
                     _r(f"q{q}.{tier}.zero_root_gain_diff_vs_PI", f"Zero-effort root gain minus the PI-side reference ({z['pi_ref_root_gain']:g}), {tier} tier, q={q}", float(z["diff_vs_pi_root_gain"]), "dW", "divided by Delta W",
                        tier, q, CF, f"{sz} / diff_vs_pi_root_gain")]
    pack.numbers_item("K17", rows, status="derived", script=f"{MOD}:build_k_misc")


def build_k19(pack: C.Pack, t58: pd.DataFrame) -> None:
    """K19: total runs and total CPU-hours, from the T58 table (and the files behind it)."""
    tot = t58[t58.study_key == "total"].iloc[0]
    t58_csv = pack.pack_src("T58")
    rows = [
        _r("total_training_runs", "Total training runs recorded in the listed studies (Pilots 1-4, dirty-flag re-run, v1.0 rehearsal, Check 2, v1.1 re-rehearsal, confirmation)", int(tot["runs"]), "runs", "none", "n/a",
           "50, 60", t58_csv.path, "row study_key=total / runs", "sum of the per-study run counts"),
        _r("total_cpu_hours", "Total process CPU hours of those runs (sum of phase_timing process_cpu_sec)", float(tot["total_cpu_hours"]), "hours", "raw", "n/a", "50, 60", t58_csv.path,
           "row study_key=total / total_cpu_hours", "sum over runs; excludes final-tier evaluation/band sweep after the last phase"),
        _r("total_wall_hours", "Total of the per-run wall times (single-threaded processes, wall >= CPU)", float(tot["total_wall_hours"]), "hours", "raw", "n/a", "50, 60", t58_csv.path,
           "row study_key=total / total_wall_hours", "sum over runs of status.json total_wall_sec"),
    ]
    pack.numbers_item("K19", rows, status="derived", script=f"{MOD}:build_k19",
                      notes="Phase 0 to Phase 2 flow-check runs (regression and smoke, about 10 runs of at most 120 updates) and the analysis computations (verifier "
                            "evaluations, supervised fits, benchmark Monte Carlo) are not counted: their CPU time is not recorded per run")


def _row_checks(pack: C.Pack, item: str, report: str, label: str, checks: List[Dict[str, Any]]) -> None:
    """Compare numbers in free-text report lines with pack values at the report's precision.

    Each check: ``{"pattern": regex with one capture group per number, "values": [pack values], "what": text}``.
    """
    text = C.abspath(report).read_text(encoding="utf-8").replace("\u2212", "-")
    n_cmp = n_bad = 0
    for ck in checks:
        m = re.search(ck["pattern"], text)
        if not m:
            pack.mismatch(item, ck["what"], ck["values"], f"{report}", "pattern not found", "report text could not be matched")
            n_bad += 1
            continue
        for g, v in zip(m.groups(), ck["values"]):
            n_cmp += 1
            if not C.consistent(float(v), g):
                n_bad += 1
                pack.mismatch(item, ck["what"], v, report, g)
    pack.crosschecks.append({"item": item, "report": report, "label": label, "n_tables": 0, "n_compared": n_cmp, "n_mismatch": n_bad,
                             "n_unmatched_rows": 0})


def build_k_crosschecks(pack: C.Pack) -> None:
    """Compare the key-number sources with the numbers printed in summary.md and the reports."""
    SUM, V11 = f"{RPT}/summary.md", f"{RPT}/protocol_v1_1_confirmation.md"
    # Pilot 1 medians (summary.md) and paired summary
    ft = S.read_csv(f"{PIL}/pilot1/analysis/final_table.csv")
    cols = ["stage2_peak_rel_err_signed", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean", "eta_T_over_dw", "sigma_effort_at_0_t2", "EXP_root_over_dw"]
    d1 = ft.groupby(["q", "arm"])[cols].median().reset_index()
    pack.crosscheck("K10", d1, SUM, header_has=["q", "arm", "eta_T_over_dw", "EXP_root_over_dw"], key_map={"q": "q", "arm": "arm"}, value_map={c: c for c in cols},
                    heading_has="Pilot 1", label="Pilot 1 final medians (summary.md)")
    ps = S.read_csv(f"{PIL}/pilot1/analysis/paired_summary.csv")
    pack.crosscheck("K10", ps, SUM, header_has=["q", "metric", "n_favour_expected"], key_map={"q": "q", "metric": "metric"},
                    value_map={"median": "median", "n_favour_expected": "n_favour_expected", "boot_ci95_lo": "boot_ci95_lo", "boot_ci95_hi": "boot_ci95_hi"},
                    heading_has="Pilot 1", label="Pilot 1 paired differences (summary.md)")
    # Pilot 2 medians
    f2 = S.read_csv(f"{PIL}/pilot2/analysis/final_table.csv")
    f2["arm_short"] = f2["arm"].map({"A_joint": "A", "B1_frozen_allnorm": "B1", "B2_frozen_s1norm": "B2"})
    c2 = ["stage2_drift_cand_on_max", "stage2_drift_cand_off_max", "stage2_peak_rel_err_signed", "stage2_tail_mean", "stage1_rel_err_signed", "Gmax_full_over_dw",
          "EXP_root_over_dw", "dReach_over_dw"]
    d2 = f2.groupby(["q", "arm_short"])[c2].median().reset_index()
    pack.crosscheck("K11", d2, SUM, header_has=["q", "arm", "stage2_drift_cand_on_max", "dReach_over_dw"], key_map={"q": "q", "arm": "arm_short"}, value_map={c: c for c in c2},
                    heading_has="Pilot 2", label="Pilot 2 final medians (summary.md)")
    # Pilot 3 stability and paired differences
    st3 = S.read_csv(f"{PIL}/pilot3/analysis/stability.csv")
    st3["arm"] = st3["arm"].map({"B2_frozen_s1norm": "stochastic", "B2_frozen_s1norm_mean": "mean"})
    pack.crosscheck("K12", st3, SUM, header_has=["q", "arm", "across_seed_sd_final_e1", "median_within_run_sd_last5"], key_map={"q": "q", "arm": "arm"},
                    value_map={c: c for c in ("across_seed_sd_final_e1", "across_seed_iqr_final_e1", "median_within_run_sd_last5", "median_within_run_range_last5", "median_sigma1_0")},
                    heading_has="Pilot 3", label="Pilot 3 stability (summary.md)")
    p3 = S.read_csv(f"{PIL}/pilot3/analysis/paired_summary.csv")
    pack.crosscheck("K12", p3, SUM, header_has=["q", "metric", "n_mean_better"], key_map={"q": "q", "metric": "metric"},
                    value_map={"median": "median", "n_mean_better": "n_mean_better", "boot_ci95_lo": "boot_ci95_lo", "boot_ci95_hi": "boot_ci95_hi"},
                    heading_has="Pilot 3", label="Pilot 3 paired differences (summary.md)")
    # Phase A extension medians (wide table in summary.md)
    tb = S.read_csv(f"{PIL}/phaseA_ext/analysis/table_400_800_1200_1600.csv")
    wide = tb.pivot_table(index=["metric", "q"], columns="update", values="median").reset_index()
    wide.columns = [str(c) for c in wide.columns]
    pack.crosscheck("K13", wide, SUM, header_has=["metric", "q", "400", "1600"], key_map={"metric": "metric", "q": "q"},
                    value_map={u: u for u in ("400", "800", "1200", "1600")}, heading_has="Phase A extension", label="extension medians u400-u1600 (summary.md)")
    # calibration (lock report)
    cal = S.read_csv(f"{LKD}/calibration/calibration_locked.csv")
    z = cal[cal.policy == "zero"]
    pack.crosscheck("K17", z, f"{RPT}/protocol_lock_and_rehearsal.md", header_has=["q", "tier", "pi_ref_Gmax", "diff_vs_pi_Gmax"], key_map={"q": "q", "tier": "tier"},
                    value_map={c: c for c in ("Gmax_full_over_dw", "pi_ref_Gmax", "diff_vs_pi_Gmax", "EXP_root_over_dw", "pi_ref_root_gain", "diff_vs_pi_root_gain")},
                    label="zero-effort calibration vs PI reference (lock report)")
    # supervised-fit floor medians (Pilot 4 report, 1d)
    fits = S.read_csv(f"{PIL}/pilot4/analysis/repr_floor/fits.csv")
    fc = ["stage2_peak_rel_err_signed", "stage2_peak_locfree_rel_err", "stage2_rmse_pos_over_g2_0", "stage2_tail_mean", "stage2_tail_max", "stage2_sym_err_max", "eta_T_over_dw",
          "Gmax_full_over_dw"]
    d16 = fits.groupby("q")[fc].median().reset_index()
    pack.crosscheck("K16", d16, f"{RPT}/pilot4_stabilization.md", header_has=["q", "stage2_peak_locfree_rel_err", "Gmax_full_over_dw"], key_map={"q": "q"},
                    value_map={c: c for c in fc}, heading_has="1d", tables=[1], label="supervised-fit medians (pilot4 report 1d)")
    # confirmation: free-text rows of the verdict and S1 tables
    pc = S.read_csv(f"{CONF}/pass_counts.csv")
    s1 = S.read_csv(f"{CONF}/s1_summary.csv")
    rows = []
    for q in (50, 60):
        r = pc[pc.q == q].iloc[0]
        rows.append({"pattern": rf"\| {q} \| \*\*(\d+) / 20\*\* \| [^|]*\| \[([\d.]+), ([\d.]+)\] \|", "values": [r["n_pass"], r["cp95_lo"], r["cp95_hi"]],
                     "what": f"K01 q={q}: passes, exact CI lo, hi (verdict table)"})
        t = s1[s1.q == q].iloc[0]
        rows.append({"pattern": rf"\| {q} \| (\d+) / 20 \| \[([\d.]+), ([\d.]+)\] \| ([+\-\d.]+) \[([-\d.]+), ([-\d.]+)\] \| ([+\-\d.]+) \| ([\d.]+) \|",
                     "values": [t["S1_pass"], t["cp95_lo"], t["cp95_hi"], t["mean_signed"], t["boot95_lo"], t["boot95_hi"], t["median_signed"], t["sd_signed"]],
                     "what": f"K07 q={q}: S1 passes, exact CI, mean signed [boot CI], median, SD (S1 table)"})
    _row_checks(pack, "K01", V11, "verdict and S1 table rows of the v1.1 report (K01, K07)", rows)


def build() -> None:
    pack = C.Pack("sec_appendix")
    t58 = build_t58(pack)
    build_t59(pack)
    build_k_confirmation(pack)
    build_k_pilots(pack)
    build_k_misc(pack)
    build_k19(pack, t58)
    build_k_crosschecks(pack)
    # K18 depends on the reproducibility ledger T18 (module sec_method); written by build_k18 when T18 exists
    try:
        import sec_appendix_k18 as k18
        k18.build_k18(pack)
    except ModuleNotFoundError:
        print("[sec_appendix] K18 skipped: sec_appendix_k18.py not present", flush=True)
    pack.save_fragment()
