#!/usr/bin/env python3
"""R2c wave S: check the launched runs against what the round pre-registered (read-only).

For every planned run of ``<root>/waveS/q{50,60}/seed{10501..10510}/<arm>`` (4 arms x 20 = 80):

  C1  ``manifest.json`` records ``dirty: false`` and a commit whose difference from the code commit
      touches only ``results/`` and ``reports/v2/refine_r2c/``;
  C2  the config the run read (``manifest.json`` ``input_config``) differs from the ON-DISK R1
      ``parents_A/q*/seed*/run_config.json`` (the four R2b keys added at their defaults) in exactly
      ``start_weights`` (``tools/v2/launch_refine.py`` ``R2C_EXPECTED_DIFFS``);
  C3  the manifest's ``start_weights`` equals the arm table's value including ``local_first``, and
      ``clamp_likelihood == 'density'``, ``pathwise_epochs == 1``, ``pathwise_minibatch is None``;
  C4  ``status.json``: ``state == done``, ``exit_code == 0``, ``final_global_update == 1600``;
  C5  the D2 prefix identities against R1's parents_A of the same (q, seed). The runs are bin-balanced
      until ``local_first``, so by construction
        A_peak50_late400: ``state_u01200.pt`` (everything but the three process-global RNG states) and
                          every ``weights/uNNNNN.npz`` with NNNNN <= 1200 are bit-identical;
        A_peak50_late800: every ``weights/uNNNNN.npz`` with NNNNN <= 800 is bit-identical;
      and (sanity, required) some later export differs. For the other arms C5 is descriptive only: the
      first differing export. For the two late arms the rows of ``v2_updates.csv`` up to the bound are
      also compared on all non-wall-clock columns (descriptive).

A missing file is a failed check, never a skipped one. Exit code 0 only if every check of every run
passed. Usage:

    python tools/v2/r2c_launch_checks.py --code-commit <sha> --out results/v2_refine_r2c/launch_checks.json
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "v2"))

import cr1_compare as C  # noqa: E402
import launch_refine as L  # noqa: E402

R1_DEFAULT = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine")
ALLOWED_AFTER_CODE = ("results/", "reports/v2/refine_r2c/")
WAVE = "r2c_waveS"
WAVE_DIRNAME = "waveS"
CHECKS = ("C1", "C2", "C3", "C4", "C5")
PHASE_A_UPDATES = 1600
STATE_AT = 1200                                   # the only update with a full-state file
PREFIX_BOUNDS = {"A_peak50_late400": 1200, "A_peak50_late800": 800}
GLOBAL_RNG_KEYS = ("torch_global_rng_state", "numpy_global_rng_state", "python_random_state")
WALL_COLUMNS = set(C.WALL_KEYS)
CLEAN_KEYS = {"clamp_likelihood": "density", "pathwise_epochs": 1, "pathwise_minibatch": None}


# --------------------------------------------------------------------------- C1: commits
def paths_outside(paths: Sequence[str], allowed: Sequence[str]) -> List[str]:
    """The paths that do not start with any allowed prefix."""
    return [p for p in paths if not p.startswith(tuple(allowed))]


@functools.lru_cache(maxsize=None)
def _diff_names(code_commit: str, commit: str) -> Optional[Tuple[str, ...]]:
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", code_commit, commit],
                       capture_output=True, text=True)
    return tuple(r.stdout.split()) if r.returncode == 0 else None


def commit_touches_only_allowed(code_commit: str, commit: str,
                                allowed: Sequence[str] = ALLOWED_AFTER_CODE) -> Optional[List[str]]:
    """Files changed between the code commit and ``commit`` outside the allowed prefixes (None if git fails)."""
    names = _diff_names(code_commit, commit)
    return None if names is None else paths_outside(names, allowed)


def check_manifest_commit(man: Dict[str, Any], code_commit: str, allowed: Sequence[str]) -> List[str]:
    """C1 problems of one manifest."""
    git = man.get("git") if isinstance(man.get("git"), dict) else {}
    problems: List[str] = []
    if git.get("dirty") is not False:
        problems.append(f"manifest git.dirty is {git.get('dirty')!r}")
    c = git.get("commit")
    out = commit_touches_only_allowed(code_commit, c, allowed) if c else None
    if out is None:
        problems.append(f"cannot compare manifest commit {c} with the code commit {code_commit}")
    elif out:
        problems.append(f"files outside {list(allowed)} changed since the code commit: {out[:5]}")
    return problems


# --------------------------------------------------------------------------- C2, C3: config
def config_diff_problems(input_config: Dict[str, Any], ref_cfg: Dict[str, Any],
                         expected: frozenset) -> Tuple[List[str], List[str]]:
    """(problems, differing keys) of the run's config against the R1 config (R2b keys added at defaults)."""
    ref = copy.deepcopy(ref_cfg)
    for k, v in L.R2B_DEFAULTS.items():
        ref.setdefault(k, copy.deepcopy(v))                       # R1 configs predate the four R2b keys
    diff = L.r2b_arm_diff(input_config, ref)
    if diff != expected:
        return [f"config differs from the R1 config in {sorted(diff)}, expected {sorted(expected)}"], sorted(diff)
    return [], sorted(diff)


def manifest_value_problems(man: Dict[str, Any], want_start_weights: Any) -> List[str]:
    """C3 problems of one manifest."""
    problems: List[str] = []
    if man.get("start_weights") != want_start_weights:
        problems.append(f"manifest start_weights = {man.get('start_weights')!r}, arm table says "
                        f"{want_start_weights!r}")
    ic = man.get("input_config")
    if isinstance(ic, dict) and ic.get("start_weights") != want_start_weights:      # the config the run read
        problems.append(f"manifest input_config.start_weights = {ic.get('start_weights')!r}, arm table says "
                        f"{want_start_weights!r}")
    for k, want in CLEAN_KEYS.items():
        if k not in man or man[k] != want:
            problems.append(f"manifest {k} = {man.get(k, '<absent>')!r}, expected {want!r}")
    return problems


# --------------------------------------------------------------------------- C4: exit status
def status_problems(status: Dict[str, Any]) -> List[str]:
    """C4 problems of one status.json."""
    problems: List[str] = []
    if status.get("state") != "done":
        problems.append(f"status state is {status.get('state')!r}, not 'done'")
    if status.get("exit_code") != 0:
        problems.append(f"status exit_code is {status.get('exit_code')!r}, not 0")
    if status.get("final_global_update") != PHASE_A_UPDATES:
        problems.append(f"status final_global_update is {status.get('final_global_update')!r}, "
                        f"not {PHASE_A_UPDATES}")
    return problems


# --------------------------------------------------------------------------- C5: prefix identities
def export_updates(d: Path) -> Dict[int, Path]:
    """``weights/uNNNNN.npz`` of a run directory, keyed by the update."""
    out: Dict[int, Path] = {}
    for p in sorted((Path(d) / "weights").glob("u*.npz")):
        m = re.fullmatch(r"u(\d+)\.npz", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def compare_exports(ref_dir: Path, new_dir: Path, bound: int) -> Dict[str, Any]:
    """Weight exports of the new run against the reference, bit for bit (dtype and shape included).

    Up to ``bound`` the two sets of exports must be equal and every array identical. Beyond it, the
    exports are read in order until the first one that differs.

    Returns:
        ``prefix_ok``, ``n_prefix`` (reference exports <= bound), ``missing`` / ``extra`` (updates
        <= bound only in one directory), ``differing`` (updates <= bound with a first difference) and
        ``first_diff_update`` (first differing export over both directories, any update, or None).
    """
    ea, eb = export_updates(ref_dir), export_updates(new_dir)
    ra, rb = {u for u in ea if u <= bound}, {u for u in eb if u <= bound}
    missing, extra = sorted(ra - rb), sorted(rb - ra)
    differing: List[Dict[str, Any]] = []
    first_diff: Optional[int] = None
    for u in sorted(set(ea) & set(eb)):
        if u > bound and first_diff is not None:
            break
        d = C.first_diff(C.load_npz(ea[u]), C.load_npz(eb[u]), ea[u].name)
        if d is None:
            continue
        first_diff = u if first_diff is None else first_diff
        if u <= bound:
            differing.append({"update": u, **d.as_dict()})
    return {"prefix_ok": bool(ra) and not (missing or extra or differing), "n_prefix": len(ra),
            "missing": missing, "extra": extra, "differing": differing[:3],
            "n_differing_in_prefix": len(differing), "first_diff_update": first_diff}


def compare_states(ref_pt: Path, new_pt: Path) -> Dict[str, Any]:
    """Training-relevant state of two full-state checkpoints, bit for bit.

    Compared: the agent (actor, critic, opponent, frozen snapshot, both Adam states, the minibatch
    stream, refresh count, concentration scales), the four numpy streams, the torch generator state,
    the counters, the snapshot log and everything else in the file except the three process-global RNG
    states, which ``cr1_compare`` excludes too and which are reported separately (``global_rng_identical``).
    """
    a, b = C.load_state(ref_pt), C.load_state(new_pt)
    fields: Dict[str, Optional[C.Diff]] = {}
    for k in sorted(set(a["agent"]) | set(b["agent"])):
        if k != "conc_scale":
            fields[f"agent.{k}"] = C.first_diff(a["agent"].get(k), b["agent"].get(k), k)
    fields["agent.conc_scale"] = C.first_diff(C.norm_conc(a["agent"].get("conc_scale")),
                                              C.norm_conc(b["agent"].get("conc_scale")), "conc_scale")
    rest_a = {k: v for k, v in a.items() if k not in ("agent",) + GLOBAL_RNG_KEYS}
    rest_b = {k: v for k, v in b.items() if k not in ("agent",) + GLOBAL_RNG_KEYS}
    for k in sorted(set(rest_a) | set(rest_b), key=str):
        fields[k] = C.first_diff(C.V1.strip(rest_a.get(k)), C.V1.strip(rest_b.get(k)), k)
    bad = {k: d for k, d in fields.items() if d is not None}
    glob = C.first_diff({k: a.get(k) for k in GLOBAL_RNG_KEYS}, {k: b.get(k) for k in GLOBAL_RNG_KEYS}, "")
    first = next(iter(bad.items()), None)
    return {"identical": not bad, "n_fields": len(fields), "differing_fields": sorted(bad),
            "first_difference": None if first is None else {"field": first[0], **first[1].as_dict()},
            "global_rng_identical": glob is None}


def compare_update_rows(ref_csv: Path, new_csv: Path, bound: int) -> Dict[str, Any]:
    """Rows 1..bound of two ``v2_updates.csv`` on the common non-wall-clock columns, as text."""
    ra, rb = C.read_csv_text(ref_csv), C.read_csv_text(new_csv)
    cols = [c for c in ra.columns if c in rb.columns and c not in WALL_COLUMNS]
    ra = ra[ra["update"].astype(int) <= bound].set_index("update")[[c for c in cols if c != "update"]]
    rb = rb[rb["update"].astype(int) <= bound].set_index("update")[[c for c in cols if c != "update"]]
    common = ra.index.intersection(rb.index)
    same = (ra.loc[common] == rb.loc[common]).all(axis=1)
    return {"n_rows_ref": len(ra), "n_rows_new": len(rb), "n_identical": int(same.sum()),
            "n_columns": len(cols) - 1, "columns_only_in_one": sorted(set(ra.columns) ^ set(rb.columns))}


def prefix_check(ref_dir: Path, new_dir: Path, bound: int, with_state: bool, with_csv: bool = True) -> Dict[str, Any]:
    """C5 of one run against its reference run: ``ok``, ``problems`` and the descriptive numbers."""
    ref_dir, new_dir = Path(ref_dir), Path(new_dir)
    problems: List[str] = []
    row: Dict[str, Any] = {"bound": bound}
    ex = compare_exports(ref_dir, new_dir, bound)
    row["exports"] = ex
    if ex["n_prefix"] == 0:
        problems.append(f"no reference weight export at or below update {bound} in {ref_dir}")
    if ex["missing"]:
        problems.append(f"exports <= {bound} missing from the run: {ex['missing'][:5]}")
    if ex["extra"]:
        problems.append(f"exports <= {bound} only in the run: {ex['extra'][:5]}")
    if ex["differing"]:
        problems.append(f"{ex['n_differing_in_prefix']} exports <= {bound} differ, first {ex['differing'][0]}")
    if ex["first_diff_update"] is None:
        problems.append("no weight export differs from the reference (the arm must diverge after the bound)")
    elif ex["first_diff_update"] <= bound:
        problems.append(f"first differing export is update {ex['first_diff_update']} <= {bound}")
    if with_state:
        name = f"state_u{STATE_AT:05d}.pt"
        try:
            st = compare_states(ref_dir / name, new_dir / name)
        except FileNotFoundError as exc:
            st = {"identical": False, "first_difference": {"field": "<missing file>", "path": str(exc)}}
        row["state"] = st
        if not st["identical"]:
            problems.append(f"{name} differs from the reference: {st['first_difference']}")
    if with_csv:
        try:
            row["v2_updates"] = compare_update_rows(ref_dir / "v2_updates.csv", new_dir / "v2_updates.csv", bound)
        except FileNotFoundError as exc:
            row["v2_updates"] = {"error": str(exc)}
    row.update(ok=not problems, problems=problems)
    return row


def descriptive_first_diff(ref_dir: Path, new_dir: Path) -> Dict[str, Any]:
    """C5 of an arm without a pre-registered prefix: the first differing export (never a failure)."""
    try:
        ex = compare_exports(ref_dir, new_dir, 0)
        return {"first_diff_update": ex["first_diff_update"], "ok": True, "problems": []}
    except Exception as exc:  # noqa: BLE001 - reported in the row
        return {"first_diff_update": None, "ok": True, "problems": [],
                "note": f"not computable: {type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- one run
def _load(p: Path) -> Any:
    with open(p) as f:
        return json.load(f)


def _load_checked(p: Path, what: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(JSON object, None) of a run file, or (None, problem) when it is missing, unreadable or not an object.

    A run killed while it rewrites its manifest leaves a half-written file: that is a failed check of that
    run, not a crash of the whole tool (which would leave the other runs without a recorded verdict).
    """
    try:
        obj = _load(p)
    except FileNotFoundError:
        return None, f"{what} missing"
    except (OSError, ValueError) as exc:
        return None, f"{what} unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(obj, dict):
        return None, f"{what} is not a JSON object"
    return obj, None


def arm_tables(arm: str) -> Tuple[Dict[str, Any], frozenset]:
    """(``r2b`` settings, expected config differences) of an arm: wave S, or R2b wave A for validation runs."""
    if arm in L.R2C_WAVE_S_ARMS:
        return L.R2C_WAVE_S_ARMS[arm]["r2b"], L.R2C_EXPECTED_DIFFS[WAVE][arm][1]
    return L.R2B_WAVEA_ARMS[arm].get("r2b", {}), L.R2B_EXPECTED_DIFFS["r2b_waveA"][arm][1]


def check_run(arm: str, q: int, seed: int, root: Path, r1: Path, code_commit: str,
              allowed: Sequence[str] = ALLOWED_AFTER_CODE, wave_dirname: str = WAVE_DIRNAME,
              prefix_bounds: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """All checks of one run. ``checks[Cn]`` has ``ok`` and ``problems``; ``ok`` is the conjunction."""
    bounds = PREFIX_BOUNDS if prefix_bounds is None else prefix_bounds
    d = Path(root) / wave_dirname / f"q{q}" / f"seed{seed}" / arm
    ref_dir = Path(r1) / "parents_A" / f"q{q}" / f"seed{seed}"
    row: Dict[str, Any] = {"arm": arm, "q": q, "seed": seed, "run_dir": str(d), "checks": {}}
    chk = row["checks"]
    if not d.is_dir():
        for c in CHECKS:
            chk[c] = {"ok": False, "problems": [f"run directory missing: {d}"]}
        row["ok"] = False
        return row
    table, expected = arm_tables(arm)
    man, man_problem = _load_checked(d / "manifest.json", "manifest.json")
    if man is None:
        for c in ("C1", "C2", "C3"):
            chk[c] = {"ok": False, "problems": [man_problem]}
    else:
        git = man.get("git") if isinstance(man.get("git"), dict) else {}
        p1 = check_manifest_commit(man, code_commit, allowed)
        chk["C1"] = {"ok": not p1, "problems": p1, "manifest_commit": git.get("commit"),
                     "manifest_dirty": git.get("dirty")}
        ref_path = ref_dir / "run_config.json"
        ref_cfg, ref_problem = _load_checked(ref_path, f"comparator config {ref_path}")
        if not isinstance(man.get("input_config"), dict):
            p2, diff = ["manifest has no input_config"], []
        elif ref_cfg is None:
            p2, diff = [ref_problem], []
        else:
            p2, diff = config_diff_problems(man["input_config"], ref_cfg, expected)
        chk["C2"] = {"ok": not p2, "problems": p2, "config_diff_vs_r1": diff}
        p3 = manifest_value_problems(man, table.get("start_weights", L.R2B_DEFAULTS["start_weights"]))
        chk["C3"] = {"ok": not p3, "problems": p3, "start_weights": man.get("start_weights")}
    status, status_problem = _load_checked(d / "status.json", "status.json")
    p4 = [status_problem] if status is None else status_problems(status)
    chk["C4"] = {"ok": not p4, "problems": p4}
    if arm in bounds:
        chk["C5"] = prefix_check(ref_dir, d, bounds[arm], with_state=bounds[arm] == STATE_AT)
        chk["C5"]["required"] = True
    else:
        chk["C5"] = {"required": False, **descriptive_first_diff(ref_dir, d)}
    row["ok"] = all(v["ok"] for v in chk.values())
    return row


def _check_star(args: Tuple[Any, ...]) -> Dict[str, Any]:
    return check_run(*args)


def _init_worker() -> None:
    torch.set_num_threads(1)


# --------------------------------------------------------------------------- summary and table
def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts per check (over the runs where the check is required) and ``all_pass``."""
    counts = {}
    for c in CHECKS:
        rel = [r["checks"][c] for r in rows if c != "C5" or r["checks"][c].get("required", True)]
        counts[c] = {"n": len(rel), "n_pass": sum(1 for x in rel if x["ok"])}
    return {"n_runs": len(rows), "n_runs_ok": sum(1 for r in rows if r["ok"]), "checks": counts,
            "all_pass": bool(rows) and all(r["ok"] for r in rows)}


def format_table(rows: List[Dict[str, Any]]) -> str:
    """Human-readable table: one line per run, then the problems of the failed checks."""
    def mark(r: Dict[str, Any], c: str) -> str:
        x = r["checks"][c]
        return "ok" if x["ok"] and x.get("required", True) else ("--" if x["ok"] else "FAIL")

    lines = [f"{'q':>3} {'seed':>6} {'arm':<18} " + " ".join(f"{c:>4}" for c in CHECKS)
             + f" {'first_diff_export':>18} {'csv_rows_same':>14}"]
    for r in rows:
        c5 = r["checks"]["C5"]
        fd = c5.get("first_diff_update", (c5.get("exports") or {}).get("first_diff_update"))
        csvr = c5.get("v2_updates") or {}
        csv_txt = f"{csvr['n_identical']}/{csvr['n_rows_ref']}" if "n_identical" in csvr else ""
        lines.append(f"{r['q']:>3} {r['seed']:>6} {r['arm']:<18} "
                     + " ".join(f"{mark(r, c):>4}" for c in CHECKS)
                     + f" {'' if fd is None else fd:>18} {csv_txt:>14}")
    for r in rows:
        for c, x in r["checks"].items():
            for p in x["problems"]:
                lines.append(f"  q{r['q']} seed{r['seed']} {r['arm']} {c}: {p}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI; exit code 0 iff every check of every run passes."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code-commit", required=True)
    ap.add_argument("--root", default=str(L.R2C_ROOT))
    ap.add_argument("--r1-root", default=str(R1_DEFAULT), help="R1 results root with parents_A/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allowed-after-code", action="append", default=None,
                    help=f"allowed path prefix of files changed after the code commit (repeatable; "
                         f"default {list(ALLOWED_AFTER_CODE)})")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--wave-dirname", default=WAVE_DIRNAME, help=argparse.SUPPRESS)
    ap.add_argument("--arms", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--prefix-bound", action="append", default=None, metavar="ARM=UPDATE",
                    help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    allowed = tuple(a.allowed_after_code or ALLOWED_AFTER_CODE)
    arms = a.arms.split(",") if a.arms else list(L.R2C_WAVE_S_ARMS)
    bounds = dict(PREFIX_BOUNDS)
    if a.prefix_bound:
        bounds = {k: int(v) for k, v in (s.split("=") for s in a.prefix_bound)}
    jobs = [(arm, q, s, Path(a.root), Path(a.r1_root), a.code_commit, allowed, a.wave_dirname, bounds)
            for q in L.DEFAULT_QS for s in L.DEFAULT_SEEDS for arm in arms]
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
            rows = list(ex.map(_check_star, jobs))
    else:
        torch.set_num_threads(1)
        rows = [_check_star(j) for j in jobs]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary = {"tool": "tools/v2/r2c_launch_checks.py", "code_commit": a.code_commit, "head": head,
               "roots": {"root": a.root, "r1_root": a.r1_root, "wave_dirname": a.wave_dirname},
               "allowed_after_code": list(allowed), "arms": arms, "prefix_bounds": bounds,
               **summarize(rows)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "runs": rows}, f, indent=1)
    print(format_table(rows))
    print(f"launch checks: {summary['n_runs_ok']}/{summary['n_runs']} runs pass; per check "
          + ", ".join(f"{c} {v['n_pass']}/{v['n']}" for c, v in summary["checks"].items())
          + f"; all_pass={summary['all_pass']}")
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
