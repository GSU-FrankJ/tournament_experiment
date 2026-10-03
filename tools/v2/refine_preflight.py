#!/usr/bin/env python3
"""P0 pre-flight data for the T=2 accuracy-refinement round R1 (read-only on all inputs).

Subcommands (all outputs are deterministic functions of the inputs):

  parents   results/v2_refine/parents.csv: path, size and SHA-256 of ``state_end_A.pt`` /
            ``state_end_B.pt`` of the 20 v1.1 rehearsal runs, plus the manifest protocol hash and
            commit. Exits with status 2 if a parent is missing or is not a v1.1 run from the launch
            commit (the files are never regenerated).
  seeds     runs ``tools/v2/seed_inventory.py`` for the blocks 10501-10510 and 30501-30520 (same
            extra roots as the lock-round inventory) and writes the per-seed source listing.
  baseline  per-run table and per-q medians of the v1.1 rehearsal (paired baseline of this round)
            and the per-q medians of the 20-seed confirmation.
  kl        distributions of the final-epoch KL and the counterfactual target-KL stopping
            statistics from the rehearsal ``train_history.json`` files.
  all       the four above, in that order.

Usage:
  python tools/v2/refine_preflight.py all [--can DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "v2"))

import seed_inventory  # noqa: E402

CAN = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2")
EXTRA_ROOTS = ("/home/fjiang4/tournament_experiment_upload_20260927", "/home/fjiang4/TEL_PPO")
QS = (50, 60)
SEEDS = tuple(range(10501, 10511))
PROTOCOL_SHA = "21d85983f2a2bebc0998e99fcf665c2c729f060996c529fa3162b3d62222a40f"
COMMIT_PREFIX = "95c000e"
TARGETS = (0.005, 0.01)
N_EPOCHS = 10
BLOCK_DEV = (10501, 10510)
BLOCK_RESERVED = (30501, 30520)
BLOCK_CONFIRM = (20501, 20520)


# ------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    """Write dict rows to CSV (LF line ends, python float repr)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into a list of string dicts."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def fmt(x: Any, sig: int = 4) -> str:
    """Compact number formatting for markdown tables."""
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{float(x):.{sig}g}"


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]], sig: int = 4) -> str:
    """Markdown table with compact number formatting."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(c, sig) for c in r) + " |")
    return "\n".join(out)


def run_dir(can: Path, q: int, seed: int) -> Path:
    """Directory of one rehearsal_v1_1 run in the canonical results worktree."""
    return can / "results" / "v2_T2_locked" / "rehearsal_v1_1" / f"q{q}" / f"seed{seed}"


# ------------------------------------------------------------------------------------------------
# parents
# ------------------------------------------------------------------------------------------------

PARENT_FIELDS = (
    "q", "seed", "run_dir", "run_status",
    "state_end_A_path", "state_end_A_bytes", "state_end_A_sha256",
    "state_end_B_path", "state_end_B_bytes", "state_end_B_sha256",
    "protocol_sha256", "protocol_version", "commit", "commit_short", "git_dirty", "clean_tree",
    "protocol_sha_ok", "commit_ok", "v11_ok",
)


def cmd_parents(can: Path, out_csv: Path, out_dir: Path) -> int:
    """Write the parents inventory; return 2 if any parent is missing or not a v1.1 launch run.

    Besides the full CSV, a compact markdown table (hash prefixes) is written to
    ``out_dir/parents_table.md`` for the pre-flight report.
    """
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for q in QS:
        for s in SEEDS:
            d = run_dir(can, q, s)
            row: Dict[str, Any] = {"q": q, "seed": s, "run_dir": str(d)}
            st = d / "status.json"
            row["run_status"] = json.load(open(st)).get("state", "") if st.is_file() else "missing"
            for tag in ("A", "B"):
                p = d / f"state_end_{tag}.pt"
                row[f"state_end_{tag}_path"] = str(p)
                if p.is_file():
                    row[f"state_end_{tag}_bytes"] = p.stat().st_size
                    row[f"state_end_{tag}_sha256"] = sha256_file(p)
                else:
                    problems.append(f"missing {p}")
            mp = d / "manifest.json"
            if mp.is_file():
                m = json.load(open(mp))
                row["protocol_sha256"] = m["locked_protocol"]["sha256"]
                row["protocol_version"] = m.get("protocol_version", "")
                row["commit"] = m["commit"]
                row["commit_short"] = m["git"]["short"]
                row["git_dirty"] = m["git"]["dirty"]
                row["clean_tree"] = m.get("clean_tree", "")
                row["protocol_sha_ok"] = row["protocol_sha256"] == PROTOCOL_SHA
                row["commit_ok"] = str(row["commit"]).startswith(COMMIT_PREFIX) and (
                    m["git"]["commit"] == row["commit"])
            else:
                problems.append(f"missing {mp}")
                row["protocol_sha_ok"] = row["commit_ok"] = False
            have_all = all(f"state_end_{t}_sha256" in row for t in ("A", "B"))
            row["v11_ok"] = bool(row["protocol_sha_ok"] and row["commit_ok"] and have_all)
            if not row["v11_ok"]:
                problems.append(f"q{q} seed{s}: not a v1.1 launch-commit run with both states")
            rows.append(row)
    write_csv(out_csv, rows, PARENT_FIELDS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parents_table.md").write_text(
        "<!-- generated by tools/v2/refine_preflight.py parents; full hashes in "
        "results/v2_refine/parents.csv -->\n\n" + md_table(
            ["q", "seed", "state_end_A.pt bytes", "state_end_A.pt SHA-256 (first 16)",
             "state_end_B.pt bytes", "state_end_B.pt SHA-256 (first 16)",
             "protocol hash = v1.1", "commit 95c000e", "git dirty"],
            [[r["q"], r["seed"], r.get("state_end_A_bytes", "missing"),
              str(r.get("state_end_A_sha256", "missing"))[:16],
              r.get("state_end_B_bytes", "missing"),
              str(r.get("state_end_B_sha256", "missing"))[:16],
              r["protocol_sha_ok"], r["commit_ok"], r.get("git_dirty", "")] for r in rows])
        + "\n")
    n_ok = sum(1 for r in rows if r["v11_ok"])
    print(f"parents: {len(rows)} rows, {n_ok} with v11_ok -> {out_csv}")
    for p in problems:
        print("  PROBLEM:", p)
    return 0 if not problems else 2


# ------------------------------------------------------------------------------------------------
# seeds
# ------------------------------------------------------------------------------------------------

def _tree_and_rel(path: str) -> Tuple[str, str]:
    """Split an absolute path into (checkout name, path relative to that checkout)."""
    tree = str(seed_inventory.TREE)
    m = re.match(re.escape(tree) + r"/\.claude/worktrees/([^/]+)/(.*)$", path)
    if m:
        return m.group(1), m.group(2)
    if path.startswith(tree + "/"):
        return "main_checkout", path[len(tree) + 1:]
    for extra in EXTRA_ROOTS:
        if path.startswith(extra + "/"):
            return Path(extra).name, path[len(extra) + 1:]
    return "other", path


def _category(rel: str) -> str:
    """Coarse location of a source file: three leading components under results/, else two."""
    parts = rel.split("/")
    return "/".join(parts[:3]) if parts[0] == "results" else "/".join(parts[:2])


def cmd_seeds(out_dir: Path) -> int:
    """Run the stock seed inventory for both blocks and write the per-seed source listing."""
    py = sys.executable
    tool = str(ROOT / "tools" / "v2" / "seed_inventory.py")
    summary: Dict[str, Any] = {"extra_roots": list(EXTRA_ROOTS), "blocks": {}}
    stock: Dict[str, Tuple[str, str]] = {}      # name -> (csv text, stdout)
    with tempfile.TemporaryDirectory(prefix="refine_seed_inv_") as tmp:
        # nothing is written into the scanned tree until every scan has finished, so that no
        # output of this command can be counted as a source by one of its own scans
        for lo, hi in (BLOCK_DEV, BLOCK_RESERVED):
            name = f"seed_inventory_{lo}_{hi}"
            tmp_csv = Path(tmp) / f"{name}.csv"
            cmd = [py, tool, "--block", str(lo), str(hi), "--out", str(tmp_csv),
                   "--extra-root", *EXTRA_ROOTS]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
            if res.returncode not in (0, 1):
                print(res.stdout, res.stderr, sep="\n")
                raise RuntimeError(f"seed_inventory failed (rc={res.returncode}) for {lo}-{hi}")
            stock[name] = (tmp_csv.read_text(), res.stdout)
            summary["blocks"][f"{lo}-{hi}"] = {
                "command": " ".join(cmd), "returncode": res.returncode,
                "meaning": "0 = no collision, 1 = at least one seed of the block is recorded"}
            lines = res.stdout.strip().splitlines()
            print(f"{lo}-{hi}: rc={res.returncode}\n  {lines[0]}\n  {lines[1]}")

    # one more in-process scan with the same roots for the per-seed source listing
    found, n_json, n_csv = seed_inventory.scan([str(seed_inventory.TREE), *EXTRA_ROOTS])
    audit = ROOT / "reports" / "v2" / "phase0_audit.md"
    for line in open(audit):
        if "seed" in line.lower():
            for tok in seed_inventory.INT_RE.findall(line):
                found[int(tok)].add((str(audit), "phase0_audit_line"))
    summary["detail_scan"] = {"n_json": n_json, "n_csv": n_csv,
                              "n_distinct_seed_values": len(found)}
    own = str(out_dir.resolve()) + "/"          # outputs of an earlier run are not sources
    proto_path = ROOT / "protocols" / "v2_T2_locked_v1_1.json"
    proto_dev = json.load(open(proto_path))["development_seeds"]
    summary["development_seeds_in_protocol"] = {
        "file": str(proto_path.relative_to(ROOT)), "key": "development_seeds",
        "seeds": proto_dev, "equals_10501_to_10510": proto_dev == list(SEEDS)}
    print(f"protocol development_seeds == 10501..10510: {proto_dev == list(SEEDS)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (csv_text, stdout) in stock.items():
        (out_dir / f"{name}.csv").write_text(csv_text)
        (out_dir / f"{name}.out").write_text(stdout)
    for tag, (lo, hi) in (("dev", BLOCK_DEV), ("reserved", BLOCK_RESERVED),
                          ("confirmation_info", BLOCK_CONFIRM)):
        grouped: Dict[Tuple[int, str, str], List[str]] = defaultdict(list)
        for s in range(lo, hi + 1):
            for path, key in found.get(s, ()):
                if path.startswith(own):
                    continue
                tree, rel = _tree_and_rel(path)
                grouped[(s, rel, key)].append(tree)
        rows = []
        for (s, rel, key), trees in sorted(grouped.items()):
            uniq = sorted(set(trees))
            rows.append({"seed": s, "rel_path": rel, "category": _category(rel), "key": key,
                         "n_checkouts": len(uniq),
                         "checkouts": ";".join(uniq[:4]) + (";..." if len(uniq) > 4 else "")})
        write_csv(out_dir / f"seed_sources_{lo}_{hi}.csv", rows,
                  ("seed", "rel_path", "category", "key", "n_checkouts", "checkouts"))
        per_seed = {s: sum(1 for k in grouped if k[0] == s) for s in range(lo, hi + 1)}
        cats: Dict[str, int] = defaultdict(int)
        for (s, rel, key) in grouped:
            cats[_category(rel)] += 1
        write_csv(out_dir / f"seed_source_categories_{lo}_{hi}.csv",
                  [{"category": c, "n_seed_path_key": n} for c, n in sorted(cats.items())],
                  ("category", "n_seed_path_key"))
        summary[f"sources_{lo}_{hi}"] = {
            "role": tag, "n_distinct_rel_paths_per_seed": per_seed,
            "n_seeds_with_any_source": sum(1 for v in per_seed.values() if v),
            "n_distinct_seed_path_key": len(grouped)}
        print(f"{tag} {lo}-{hi}: {len(grouped)} distinct (seed, path, key); "
              f"seeds with any source: {summary[f'sources_{lo}_{hi}']['n_seeds_with_any_source']}")
    (out_dir / "seed_inventory_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n")
    # PROMPT.md section 1 item 5: report (exit 2) if the reserved block is not collision-free or
    # the protocol's development seeds are not 10501-10510; the files above are still written
    reserved_rc = summary["blocks"][f"{BLOCK_RESERVED[0]}-{BLOCK_RESERVED[1]}"]["returncode"]
    if reserved_rc != 0 or proto_dev != list(SEEDS):
        print("PROBLEM: reserved block 30501-30520 collides, or development seeds differ")
        return 2
    return 0


# ------------------------------------------------------------------------------------------------
# baseline
# ------------------------------------------------------------------------------------------------

BASELINE_FIELDS = (
    "q", "seed", "outcome", "run_pass", "G_A_pass", "G_F_pass", "G_N_pass", "S1_pass",
    "stage1_rel_err_signed", "stage1_rel_err_abs", "e1_at_0", "g1",
    "peak_rel_err_signed", "peak_rel_err_locfree", "peak_locfree_argmax_d",
    "rmse_pos_over_e2star0", "tail_mean_over_e2star0", "tail_mean", "tail_max",
    "tail_max_over_e2star0",
    "eta2_final", "eta2_dev", "eta2_dev_minus_final",
    "gmax_final", "gmax_final_t", "gmax_final_d", "gmax_dev", "gmax_dev_t", "gmax_dev_d",
    "gmax_dev_minus_final",
    "exp_root", "dreach", "deltamax_all", "dfull",
    "sigma_effort_at_0_t2", "sigma_effort_at_0_t1", "e2_at_0", "g2_at_0",
    "sym_err_max", "smoothed_share_peak_gap_d0", "wall_sec_total",
)
BOOL_FIELDS = ("run_pass", "G_A_pass", "G_F_pass", "G_N_pass", "S1_pass")

# rows of the medians tables: (label, per-run column, confirmation source file, source column)
MEDIAN_ROWS = (
    ("stage-1 signed relative error", "stage1_rel_err_signed", "per_run.csv",
     "stage1_rel_err_signed"),
    ("stage-1 absolute relative error (S1)", "stage1_rel_err_abs", "per_run.csv", "s1"),
    ("stage-2 peak error, signed (d = 0)", "peak_rel_err_signed", "reported_metrics.csv",
     "A_stage2_peak_rel_err_signed"),
    ("stage-2 peak error, location-free", "peak_rel_err_locfree", "reported_metrics.csv",
     "A_stage2_peak_locfree_rel_err"),
    ("RMSE_pos / e2*(0)", "rmse_pos_over_e2star0", "per_run.csv", "rmse"),
    ("tail mean / e2*(0)", "tail_mean_over_e2star0", "per_run.csv", "tail"),
    ("tail max (effort units)", "tail_max", "reported_metrics.csv", "A_stage2_tail_max"),
    ("eta_2 / DW, final tier", "eta2_final", "per_run.csv", "eta_final"),
    ("eta_2 / DW, development tier", "eta2_dev", "per_run.csv", "eta_dev"),
    ("eta_2 dev - final", "eta2_dev_minus_final", "per_run.csv", "eta_dev_minus_final"),
    ("Gmax_full / DW, final tier", "gmax_final", "per_run.csv", "gmax_final"),
    ("Gmax_full / DW, development tier", "gmax_dev", "per_run.csv", "gmax_dev"),
    ("Gmax_full dev - final", "gmax_dev_minus_final", "per_run.csv", "gmax_dev_minus_final"),
    ("EXP_root / DW (final tier)", "exp_root", "reported_metrics.csv", "B_EXP_root_over_dw"),
    ("dReach / DW (final tier)", "dreach", "reported_metrics.csv", "B_dReach_over_dw"),
    ("Deltamax_all / DW (final tier)", "deltamax_all", "reported_metrics.csv",
     "B_Deltamax_all_over_dw"),
    ("dFull / DW (final tier)", "dfull", "reported_metrics.csv", "B_dFull_over_dw"),
    ("sigma_effort at d = 0, stage 2", "sigma_effort_at_0_t2", "reported_metrics.csv",
     "A_sigma_effort_at_0_t2"),
    ("e1_hat(0)", "e1_at_0", "reported_metrics.csv", "B_e1_at_0"),
)


def baseline_row(can: Path, q: int, seed: int) -> Dict[str, Any]:
    """One rehearsal_v1_1 run: values from ``gates.json`` and ``v2_run_summary.json``."""
    d = run_dir(can, q, seed)
    g = json.load(open(d / "gates.json"))
    ra, rb = g["reported"]["end_of_A"], g["reported"]["end_of_B"]
    fa, fb, db = ra["final"], rb["final"], rb["development"]
    mv = g["metric_values"]
    summ = json.load(open(d / "v2_run_summary.json"))
    return {
        "q": q, "seed": seed, "outcome": g["outcome"], "run_pass": g["run_pass"],
        "G_A_pass": g["G-A"]["pass"], "G_F_pass": g["G-F"]["pass"], "G_N_pass": g["G-N"]["pass"],
        "S1_pass": g["S1"]["pass"],
        "stage1_rel_err_signed": fb["stage1_rel_err_signed"],
        "stage1_rel_err_abs": fb["stage1_rel_err_abs"], "e1_at_0": fb["e1_at_0"], "g1": fb["g1"],
        "peak_rel_err_signed": fa["stage2_peak_rel_err_signed"],
        "peak_rel_err_locfree": fa["stage2_peak_locfree_rel_err"],
        "peak_locfree_argmax_d": fa["stage2_peak_locfree_argmax_d"],
        "rmse_pos_over_e2star0": fa["stage2_rmse_pos_over_g2_0"],
        "tail_mean_over_e2star0": fa["stage2_tail_mean_over_g2_0"],
        "tail_mean": fa["stage2_tail_mean"], "tail_max": fa["stage2_tail_max"],
        "tail_max_over_e2star0": fa["stage2_tail_max_over_g2_0"],
        "eta2_final": mv["eta_final"], "eta2_dev": mv["eta_dev"],
        "eta2_dev_minus_final": mv["eta_dev"] - mv["eta_final"],
        "gmax_final": mv["gmax_final"], "gmax_final_t": fb["Gmax_full_t"],
        "gmax_final_d": fb["Gmax_full_d"], "gmax_dev": mv["gmax_dev"],
        "gmax_dev_t": db["Gmax_full_t"], "gmax_dev_d": db["Gmax_full_d"],
        "gmax_dev_minus_final": mv["gmax_dev"] - mv["gmax_final"],
        "exp_root": fb["EXP_root_over_dw"], "dreach": fb["dReach_over_dw"],
        "deltamax_all": fb["Deltamax_all_over_dw"], "dfull": fb["dFull_over_dw"],
        "sigma_effort_at_0_t2": fa["sigma_effort_at_0_t2"],
        "sigma_effort_at_0_t1": fb["sigma_effort_at_0_t1"],
        "e2_at_0": fa["e2_at_0"], "g2_at_0": fa["g2_at_0"],
        "sym_err_max": fa["stage2_sym_err_max"],
        "smoothed_share_peak_gap_d0": ra["smoothed_game"]["smoothed_share_peak_gap_d0"],
        "wall_sec_total": summ.get("total_wall_sec"),
    }


def _crosscheck(rows: List[Dict[str, Any]], an_dir: Path) -> Dict[str, Any]:
    """Compare the gates.json values with the analysis CSVs of the existing analysis tools.

    This is a transcription check: those tools also take their values from ``gates.json``, so
    agreement shows the numbers were copied correctly, not that they were recomputed.
    """
    per_run = {(int(r["q"]), int(r["seed"])): r for r in read_csv(an_dir / "per_run.csv")}
    rep = {(int(r["q"]), int(r["seed"])): r for r in read_csv(an_dir / "reported_metrics.csv")}
    pairs = (
        ("eta2_final", per_run, "eta_final"), ("eta2_dev", per_run, "eta_dev"),
        ("rmse_pos_over_e2star0", per_run, "rmse"), ("tail_mean_over_e2star0", per_run, "tail"),
        ("gmax_final", per_run, "gmax_final"), ("gmax_dev", per_run, "gmax_dev"),
        ("stage1_rel_err_abs", per_run, "s1"),
        ("stage1_rel_err_signed", per_run, "stage1_rel_err_signed"),
        ("peak_rel_err_signed", rep, "A_stage2_peak_rel_err_signed"),
        ("peak_rel_err_locfree", rep, "A_stage2_peak_locfree_rel_err"),
        ("tail_max", rep, "A_stage2_tail_max"), ("exp_root", rep, "B_EXP_root_over_dw"),
        ("dreach", rep, "B_dReach_over_dw"), ("deltamax_all", rep, "B_Deltamax_all_over_dw"),
        ("dfull", rep, "B_dFull_over_dw"), ("e1_at_0", rep, "B_e1_at_0"),
        ("sigma_effort_at_0_t2", rep, "A_sigma_effort_at_0_t2"),
    )
    maxdiff = {}
    for col, src, scol in pairs:
        maxdiff[f"{col} vs {scol}"] = max(
            abs(float(r[col]) - float(src[(r["q"], r["seed"])][scol])) for r in rows)
    outcome_equal = all(r["outcome"] == per_run[(r["q"], r["seed"])]["outcome"] for r in rows)
    return {"n_rows": len(rows), "max_abs_diff": maxdiff, "outcome_all_equal": outcome_equal,
            "analysis_dir": str(an_dir)}


def _median_table(rows: List[Dict[str, Any]], cols: Sequence[str]) -> List[Dict[str, Any]]:
    """Per-q n, min, median, max of each numeric column."""
    out = []
    for q in QS:
        sub = [r for r in rows if int(r["q"]) == q]
        for c in cols:
            v = np.array([float(r[c]) for r in sub])
            out.append({"q": q, "metric": c, "n": len(sub), "min": float(v.min()),
                        "median": float(np.median(v)), "max": float(v.max())})
    return out


def cmd_baseline(can: Path, out_dir: Path) -> int:
    """Per-run rehearsal baseline, per-q medians, confirmation medians, cross-checks."""
    lock = can / "results" / "v2_T2_locked"
    rows = [baseline_row(can, q, s) for q in QS for s in SEEDS]
    write_csv(out_dir / "baseline_rehearsal_v1_1_per_run.csv", rows, BASELINE_FIELDS)
    cols = [c for _, c, _, _ in MEDIAN_ROWS]
    med = _median_table(rows, cols)
    write_csv(out_dir / "baseline_rehearsal_v1_1_per_q_median.csv", med,
              ("q", "metric", "n", "min", "median", "max"))
    outcomes = sorted({r["outcome"] for r in rows})
    counts = []
    for q in QS:
        sub = [r for r in rows if r["q"] == q]
        c: Dict[str, Any] = {"q": q, "n": len(sub)}
        for f in BOOL_FIELDS:
            c[f"n_{f}"] = sum(1 for r in sub if r[f])
        for o in outcomes:
            c[f"outcome_{o}"] = sum(1 for r in sub if r["outcome"] == o)
        counts.append(c)
    count_fields = ["q", "n"] + [f"n_{f}" for f in BOOL_FIELDS] + [f"outcome_{o}" for o in outcomes]
    write_csv(out_dir / "baseline_rehearsal_v1_1_gate_counts.csv", counts, count_fields)
    xc = _crosscheck(rows, lock / "rehearsal_v1_1_analysis")
    (out_dir / "baseline_rehearsal_v1_1_crosscheck.json").write_text(
        json.dumps(xc, indent=1, sort_keys=True) + "\n")

    # confirmation medians (20 seeds per q) from the analysis CSVs
    ca = lock / "confirmation_analysis"
    src = {"per_run.csv": read_csv(ca / "per_run.csv"),
           "reported_metrics.csv": read_csv(ca / "reported_metrics.csv")}
    dist = {(int(r["q"]), r["metric"]): float(r["median"])
            for r in read_csv(ca / "distributions.csv")}
    conf, max_dist_diff = [], 0.0
    for label, col, fname, scol in MEDIAN_ROWS:
        for q in QS:
            v = np.array([float(r[scol]) for r in src[fname] if int(r["q"]) == q])
            m = float(np.median(v))
            if (q, scol) in dist:
                max_dist_diff = max(max_dist_diff, abs(m - dist[(q, scol)]))
            conf.append({"q": q, "metric": label, "n": len(v), "min": float(v.min()),
                         "median": m, "max": float(v.max()),
                         "source_file": f"confirmation_analysis/{fname}", "source_column": scol})
    write_csv(out_dir / "baseline_confirmation_medians.csv", conf,
              ("q", "metric", "n", "min", "median", "max", "source_file", "source_column"))
    # runs meeting the 0929 target |stage-1 relative error| <= 0.05 (S1 itself is <= 0.10)
    s1_rows = []
    for q in QS:
        reh = [float(r["stage1_rel_err_abs"]) for r in rows if int(r["q"]) == q]
        con = [float(r["s1"]) for r in src["per_run.csv"] if int(r["q"]) == q]
        for name, vals in (("rehearsal_v1_1", reh), ("confirmation", con)):
            s1_rows.append({"set": name, "q": q, "n": len(vals),
                            "n_le_0.05": sum(1 for v in vals if v <= 0.05),
                            "n_le_0.10": sum(1 for v in vals if v <= 0.10),
                            "max_abs_err": max(vals)})
    write_csv(out_dir / "baseline_s1_target_counts.csv", s1_rows,
              ("set", "q", "n", "n_le_0.05", "n_le_0.10", "max_abs_err"))
    (out_dir / "baseline_confirmation_medians_check.json").write_text(json.dumps(
        {"max_abs_diff_vs_distributions_csv_median": max_dist_diff,
         "distributions_csv": str(ca / "distributions.csv")}, indent=1) + "\n")

    # markdown fragments for the report
    lab = {c: l for l, c, _, _ in MEDIAN_ROWS}
    byq = {(r["q"], r["metric"]): r for r in med}

    def cell(r: Dict[str, Any]) -> str:
        return f"{fmt(r['median'])} ({fmt(r['min'])}, {fmt(r['max'])})"

    t_reh = md_table(
        ["metric (column of the per-run CSV)", "q=50 median (min, max)", "q=60 median (min, max)"],
        [[f"{lab[c]} (`{c}`)"] + [cell(byq[(q, c)]) for q in QS] for c in cols])
    cbq = {(r["q"], r["metric"]): r for r in conf}
    t_conf = md_table(
        ["metric", "source file", "source column", "q=50 median (min, max)",
         "q=60 median (min, max)"],
        [[lb, f"`{f}`", f"`{sc}`"] + [cell(cbq[(q, lb)]) for q in QS]
         for lb, _, f, sc in MEDIAN_ROWS])
    gc_head = ["q", "n runs"] + [f.replace("_pass", "") for f in BOOL_FIELDS] + [
        f"outcome {o}" for o in outcomes]
    gc_rows = [[c["q"], c["n"]] + [c[f"n_{f}"] for f in BOOL_FIELDS]
               + [c[f"outcome_{o}"] for o in outcomes] for c in counts]
    t_run = md_table(
        ["q", "seed", "outcome", "S1", "stage-1 signed err", "peak err (signed)",
         "RMSE_pos/e2*(0)", "tail mean/e2*(0)", "eta_2/DW (final)", "Gmax/DW (final)",
         "Gmax (t*, d*)", "wall s"],
        [[r["q"], r["seed"], r["outcome"], r["S1_pass"], r["stage1_rel_err_signed"],
          r["peak_rel_err_signed"], r["rmse_pos_over_e2star0"], r["tail_mean_over_e2star0"],
          r["eta2_final"], r["gmax_final"],
          f"({int(r['gmax_final_t'])}, {fmt(r['gmax_final_d'])})", r["wall_sec_total"]]
         for r in rows], sig=4)
    (out_dir / "baseline_tables.md").write_text(
        "<!-- generated by tools/v2/refine_preflight.py baseline -->\n\n"
        "#### Rehearsal v1.1, per-q medians (n = 10 per q)\n\n" + t_reh + "\n\n"
        "#### Rehearsal v1.1, number of runs passing, per q\n\n"
        + md_table(gc_head, gc_rows) + "\n\n"
        "#### Stage-1 error against the 0.05 target and the S1 threshold 0.10\n\n"
        + md_table(["set", "q", "n runs", "|err| <= 0.05", "|err| <= 0.10", "max |err|"],
                   [[r["set"], r["q"], r["n"], r["n_le_0.05"], r["n_le_0.10"], r["max_abs_err"]]
                    for r in s1_rows]) + "\n\n"
        "#### Rehearsal v1.1, per run (the paired baseline of this round)\n\n" + t_run + "\n\n"
        "#### Confirmation, per-q medians (n = 20 per q)\n\n" + t_conf + "\n")
    print(f"baseline: {len(rows)} per-run rows -> {out_dir}")
    print("  max |gates.json - analysis CSV|:", max(xc["max_abs_diff"].values()),
          "; outcome equal:", xc["outcome_all_equal"])
    print("  max |median - distributions.csv median| (confirmation):", max_dist_diff)
    return 0


# ------------------------------------------------------------------------------------------------
# kl
# ------------------------------------------------------------------------------------------------

KL_GROUPS = (("A", "A", 1, 10 ** 9), ("B", "B", 1, 10 ** 9),
             ("A local 1-1200", "A", 1, 1200), ("A local 1201-1600", "A", 1201, 1600))


def load_kl(can: Path, q: int) -> List[Dict[str, Any]]:
    """KL arrays of the 10 rehearsal runs of one q: phase, local, final-epoch and per-epoch KL."""
    runs = []
    for s in SEEDS:
        h = json.load(open(run_dir(can, q, s) / "train_history.json"))["history"]
        phase = np.array([x["phase"] for x in h])
        local = np.array([x["local"] for x in h], dtype=int)
        kl_fin = np.array([x["kl_final_epoch"] for x in h], dtype=float)
        lens = {len(x["kl_epochs"]) for x in h}
        if lens != {N_EPOCHS}:
            raise RuntimeError(f"q{q} seed{s}: kl_epochs lengths {sorted(lens)} != {N_EPOCHS}")
        kl_ep = np.array([x["kl_epochs"] for x in h], dtype=float)
        if not np.all(np.isfinite(kl_ep)) or not np.array_equal(kl_ep[:, -1], kl_fin):
            raise RuntimeError(f"q{q} seed{s}: non-finite KL or kl_final_epoch != kl_epochs[-1]")
        runs.append({"seed": s, "phase": phase, "local": local, "kl_final": kl_fin,
                     "kl_epochs": kl_ep})
    return runs


def _select(runs: List[Dict[str, Any]], phase: str, lo: int, hi: int) -> List[np.ndarray]:
    """Per-run row masks of one phase and local-update window."""
    return [(r["phase"] == phase) & (r["local"] >= lo) & (r["local"] <= hi) for r in runs]


def _stop_epoch(kl_ep: np.ndarray, target: float) -> np.ndarray:
    """1-based first epoch whose KL exceeds ``target``; 0 where no epoch does."""
    over = kl_ep > target
    return np.where(over.any(axis=1), over.argmax(axis=1) + 1, 0)


def kl_tables(can: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Final-epoch KL distribution rows and target-KL stopping rows for every group and q."""
    dist_rows: List[Dict[str, Any]] = []
    stop_rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"n_epochs_logged": N_EPOCHS, "runs": {}}
    for q in QS:
        runs = load_kl(can, q)
        meta["runs"][str(q)] = [r["seed"] for r in runs]
        for gname, phase, lo, hi in KL_GROUPS:
            sel = _select(runs, phase, lo, hi)
            kl_fin = np.concatenate([r["kl_final"][m] for r, m in zip(runs, sel)])
            kl_ep = np.concatenate([r["kl_epochs"][m] for r, m in zip(runs, sel)])
            p10, med, p90, p99 = np.percentile(kl_fin, [10, 50, 90, 99])
            dist_rows.append({"group": gname, "q": q, "n_runs": len(runs),
                              "n_updates": int(kl_fin.size), "min": float(kl_fin.min()),
                              "p10": float(p10), "median": float(med), "p90": float(p90),
                              "p99": float(p99), "max": float(kl_fin.max()),
                              "mean": float(kl_fin.mean())})
            for target in TARGETS:
                se = _stop_epoch(kl_ep, target)
                n_run = np.where(se > 0, se, N_EPOCHS)
                stopped = se > 0
                per_run_frac = [float((_stop_epoch(r["kl_epochs"][m], target) > 0).mean())
                                for r, m in zip(runs, sel)]
                row: Dict[str, Any] = {
                    "group": gname, "q": q, "target_kl": target, "n_updates": int(se.size),
                    "n_stopped": int(stopped.sum()), "frac_stopped": float(stopped.mean()),
                    "n_truncated": int(((se > 0) & (se < N_EPOCHS)).sum()),
                    "frac_truncated": float(((se > 0) & (se < N_EPOCHS)).mean()),
                    "frac_stopped_per_run_min": min(per_run_frac),
                    "frac_stopped_per_run_max": max(per_run_frac),
                    "stop_epoch_median_among_stopped":
                        float(np.median(se[stopped])) if stopped.any() else float("nan"),
                    "stop_epoch_p90_among_stopped":
                        float(np.percentile(se[stopped], 90)) if stopped.any() else float("nan"),
                    "epochs_run_median": float(np.median(n_run)),
                    "epochs_run_p90": float(np.percentile(n_run, 90)),
                    "epochs_run_mean": float(n_run.mean()),
                }
                for e in range(1, N_EPOCHS + 1):
                    row[f"n_stop_at_epoch_{e}"] = int((se == e).sum())
                row["n_never_stopped"] = int((se == 0).sum())
                stop_rows.append(row)
    return dist_rows, stop_rows, meta


def _kl_md(dist_rows: List[Dict[str, Any]], stop_rows: List[Dict[str, Any]]) -> str:
    """Markdown tables for the KL appendix."""
    def dist_tab(groups: Sequence[str]) -> str:
        return md_table(
            ["phase", "q", "updates", "min", "p10", "median", "p90", "p99", "max", "mean"],
            [[r["group"], r["q"], r["n_updates"], r["min"], r["p10"], r["median"], r["p90"],
              r["p99"], r["max"], r["mean"]] for r in dist_rows if r["group"] in groups], sig=3)

    def stop_tab(groups: Sequence[str]) -> str:
        rows = []
        for r in stop_rows:
            if r["group"] not in groups:
                continue
            lo, hi = r["frac_stopped_per_run_min"], r["frac_stopped_per_run_max"]
            rows.append([
                r["group"], r["q"], r["target_kl"], r["n_updates"], r["n_stopped"],
                f"{100 * r['frac_stopped']:.2f}%", f"{100 * r['frac_truncated']:.2f}%",
                f"{100 * lo:.1f}% to {100 * hi:.1f}%",
                f"{fmt(r['stop_epoch_median_among_stopped'])} / "
                f"{fmt(r['stop_epoch_p90_among_stopped'])}",
                f"{fmt(r['epochs_run_median'])} / {fmt(r['epochs_run_p90'])} / "
                f"{fmt(r['epochs_run_mean'], 3)}"])
        return md_table(
            ["phase", "q", "target", "updates", "stopped (n)", "stopped (share)",
             "of which fewer than 10 epochs run (share)", "stopped share per run, min to max", "stop epoch median / p90 (stopped only)",
             "epochs run median / p90 / mean"], rows, sig=3)

    def epoch_tab(groups: Sequence[str]) -> str:
        head = ["phase", "q", "target"] + [f"ep {e}" for e in range(1, N_EPOCHS + 1)] + ["none"]
        rows = []
        for r in stop_rows:
            if r["group"] in groups:
                shares = [100 * r[f"n_stop_at_epoch_{e}"] / r["n_updates"]
                          for e in range(1, N_EPOCHS + 1)] + [
                    100 * r["n_never_stopped"] / r["n_updates"]]
                rows.append([r["group"], r["q"], r["target_kl"]] + [f"{v:.2f}" for v in shares])
        return md_table(head, rows, sig=3)

    main, supp = ("A", "B"), ("A local 1-1200", "A local 1201-1600")
    return "\n\n".join([
        "#### (a) Final-epoch KL (`kl_final_epoch`), all updates of the 10 seeds\n\n"
        + dist_tab(main),
        "#### (b) Counterfactual stopping at each target\n\n" + stop_tab(main),
        "#### (c) Stopping epoch, share of all updates (%)\n\n"
        "Epoch index is 1-based; 'none' = no epoch exceeded the target (10 epochs run).\n\n"
        + epoch_tab(main),
        "#### (d) Supplementary: phase A split by learning-rate window\n\n"
        "Local 1-1200 has constant LR 3e-4; local 1201-1600 is the linear decay 3e-4 to 3e-5, "
        "the window the phase-A continuation arms run in.\n\n"
        + dist_tab(supp) + "\n\n" + stop_tab(supp) + "\n\n" + epoch_tab(supp),
    ])


def cmd_kl(can: Path, out_dir: Path) -> int:
    """KL distribution and counterfactual target-KL tables (CSV and markdown)."""
    dist_rows, stop_rows, meta = kl_tables(can)
    write_csv(out_dir / "kl_final_epoch_distribution.csv", dist_rows,
              ("group", "q", "n_runs", "n_updates", "min", "p10", "median", "p90", "p99", "max",
               "mean"))
    stop_fields = (["group", "q", "target_kl", "n_updates", "n_stopped", "frac_stopped",
                    "n_truncated", "frac_truncated",
                    "frac_stopped_per_run_min", "frac_stopped_per_run_max",
                    "stop_epoch_median_among_stopped", "stop_epoch_p90_among_stopped",
                    "epochs_run_median", "epochs_run_p90", "epochs_run_mean"]
                   + [f"n_stop_at_epoch_{e}" for e in range(1, N_EPOCHS + 1)]
                   + ["n_never_stopped"])
    write_csv(out_dir / "kl_target_stopping.csv", stop_rows, stop_fields)
    (out_dir / "kl_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    header = (
        "### Appendix: KL statistics of the locked v1.1 baseline (descriptive)\n\n"
        "Source: `history[*]` of `train_history.json` of the 20 `rehearsal_v1_1` runs "
        "(q in {50, 60} x seeds 10501-10510), canonical worktree "
        "`results/v2_T2_locked/rehearsal_v1_1/q*/seed*/`. Every update logs `kl_epochs`, the "
        "whole-buffer KL over the policy rows after each of the 10 PPO epochs, "
        "mean((r - 1) - log r) with r = pi_new / pi_old, and `kl_final_epoch` = `kl_epochs[-1]`. "
        "Phase A has 1600 updates and phase B 600 per run, so each table row pools 16000 (A) or "
        "6000 (B) updates per q. Percentiles use linear interpolation (`numpy.percentile`).\n\n"
        "Stopping rule of the target-KL arms (D5): after an epoch, if the KL exceeds the target "
        "(strict >) no further epoch runs; the epoch that exceeded the target has run. The "
        "number of epochs run is therefore the 1-based index of the first epoch with KL > target, "
        "or 10 if none exceeds it. The tables apply that rule to the baseline `kl_epochs`. "
        "They are exact for each baseline update taken in isolation (up to and including the "
        "first stop the update is identical to the baseline), but counterfactual for the run as "
        "a whole: after the first stop the trajectory of a target-KL run differs from the "
        "baseline, so later updates would see different buffers and policies. They are not a "
        "prediction of the share of stopped updates in a target-KL run. 'Stopped' means that some "
        "epoch's KL exceeded the target; an exceedance at epoch 10 truncates nothing (10 epochs "
        "run either way), so the column 'fewer than 10 epochs run' (stop epoch 1 to 9) is the share "
        "of updates a target-KL run would actually shorten.\n\n"
        "Script: `python tools/v2/refine_preflight.py kl`; CSVs: "
        "`results/v2_refine/preflight/kl_final_epoch_distribution.csv`, "
        "`results/v2_refine/preflight/kl_target_stopping.csv`.\n\n")
    (out_dir / "kl_appendix.md").write_text(header + _kl_md(dist_rows, stop_rows) + "\n")
    for r in stop_rows:
        if r["group"] in ("A", "B"):
            print(f"  {r['group']} q={r['q']} target={r['target_kl']}: "
                  f"stopped {r['frac_stopped']:.4f}, epochs run median {r['epochs_run_median']}")
    print(f"kl: wrote tables to {out_dir}")
    return 0


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------

def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("cmd", choices=("parents", "seeds", "baseline", "kl", "all"))
    p.add_argument("--can", type=Path, default=CAN, help="canonical results worktree (read-only)")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "v2_refine" / "preflight")
    p.add_argument("--parents-csv", type=Path,
                   default=ROOT / "results" / "v2_refine" / "parents.csv")
    p.add_argument("--skip-seeds", action="store_true", help="with 'all': skip the slow seed scan")
    a = p.parse_args()
    rc = 0
    if a.cmd in ("parents", "all"):
        rc = max(rc, cmd_parents(a.can, a.parents_csv, a.out))
        if rc:
            return rc
    if a.cmd == "seeds" or (a.cmd == "all" and not a.skip_seeds):
        rc = max(rc, cmd_seeds(a.out))
    if a.cmd in ("baseline", "all"):
        rc = max(rc, cmd_baseline(a.can, a.out))
    if a.cmd in ("kl", "all"):
        rc = max(rc, cmd_kl(a.can, a.out))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
