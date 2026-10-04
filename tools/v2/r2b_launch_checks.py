#!/usr/bin/env python3
"""R2b section 4.3: check the launched pilot waves against what the round pre-registered (read-only).

For every planned run of ``results/v2_refine_r2b/waveA`` and ``waveP``:

  * the run's own ``manifest.json`` records ``dirty: false`` and a commit whose difference from the code
    commit touches only ``results/`` and ``reports/v2/refine_r2b/``;
  * the config the run read (``manifest.json`` ``input_config``) differs from the comparator's ON-DISK R1
    ``run_config.json`` (wave A: ``parents_A``; wave P: ``A_detmean`` for the pathwise arms, ``A_ctrl200``
    for the new control; the four R2b keys added at their defaults) in exactly the pre-registered keys
    (``tools/v2/launch_refine.py`` ``R2B_EXPECTED_DIFFS``): wave A = the locked Phase-A settings apart from
    the one change;
  * the manifest's four R2b keys equal the arm table's values;
  * wave P: the manifest's ``parent_sha256`` and the SHA-256 of the parent file equal the value of
    ``results/v2_refine_r2b/parents.csv``.

A missing file is a failed check, never a skipped one. Usage:

    python tools/v2/r2b_launch_checks.py --code-commit <sha> --out results/v2_refine_r2b/launch_checks.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "v2"))

import launch_refine as L  # noqa: E402

R1_DEFAULT = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine")
ALLOWED_AFTER_CODE = ("results/", "reports/v2/refine_r2b/")
R2B_KEYS = tuple(L.R2B_DEFAULTS)


def sha256_file(p: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(p: Path) -> Any:
    """JSON file; FileNotFoundError when missing."""
    with open(p) as f:
        return json.load(f)


def commit_touches_only_allowed(code_commit: str, commit: str) -> Optional[List[str]]:
    """Files changed between the code commit and ``commit`` outside the allowed prefixes (None if git fails)."""
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", code_commit, commit],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return [f for f in r.stdout.split() if not f.startswith(ALLOWED_AFTER_CODE)]


def check_run(wave: str, arm: str, q: int, seed: int, root: Path, r1: Path, parents: Dict[Any, Dict[str, str]],
              code_commit: str, commit_cache: Dict[str, Optional[List[str]]]) -> Dict[str, Any]:
    """All checks of one run; ``ok`` is the conjunction, ``problems`` lists every failed check."""
    sub = "waveA" if wave == "r2b_waveA" else "waveP"
    d = root / sub / f"q{q}" / f"seed{seed}" / arm
    problems: List[str] = []
    row: Dict[str, Any] = {"wave": wave, "arm": arm, "q": q, "seed": seed, "run_dir": str(d)}
    try:
        man = load(d / "manifest.json")
    except FileNotFoundError:
        row.update(ok=False, problems=["manifest.json missing"])
        return row
    git = man.get("git") or {}
    row["manifest_commit"], row["manifest_dirty"] = git.get("commit"), git.get("dirty")
    if git.get("dirty") is not False:
        problems.append(f"manifest git.dirty is {git.get('dirty')!r}")
    c = git.get("commit")
    if c not in commit_cache:
        commit_cache[c] = commit_touches_only_allowed(code_commit, c) if c else None
    if commit_cache[c] is None:
        problems.append(f"cannot compare manifest commit {c} with the code commit {code_commit}")
    elif commit_cache[c]:
        problems.append(f"files outside results/ and reports/v2/refine_r2b/ changed since the code commit: "
                        f"{commit_cache[c][:5]}")
    ref_name = L.R2B_EXPECTED_DIFFS[wave][arm][0]
    ref_path = (r1 / "parents_A" / f"q{q}" / f"seed{seed}" / "run_config.json" if wave == "r2b_waveA"
                else r1 / "stage2" / f"q{q}" / f"seed{seed}" / ref_name / "run_config.json")
    try:
        ref = load(ref_path)
        for k, v in L.R2B_DEFAULTS.items():
            ref.setdefault(k, copy.deepcopy(v))               # R1 configs predate the four R2b keys
        diff = frozenset(L.flatten_diff(man["input_config"], ref)) - L.IDENTITY_KEYS_R2B
        row["config_diff_vs_r1"] = sorted(diff)
        want = L.R2B_EXPECTED_DIFFS[wave][arm][1]
        if diff != want:
            problems.append(f"config differs from {ref_path} in {sorted(diff)}, expected {sorted(want)}")
    except FileNotFoundError:
        problems.append(f"comparator config missing: {ref_path}")
    table = (L.R2B_WAVEA_ARMS if wave == "r2b_waveA" else L.R2B_WAVEP_ARMS)[arm].get("r2b", {})
    for k in R2B_KEYS:
        want_v = table.get(k, L.R2B_DEFAULTS[k])
        if man.get(k) != want_v:
            problems.append(f"manifest {k} = {man.get(k)!r}, arm table says {want_v!r}")
    if wave == "r2b_waveP":
        p = parents[(q, seed)]
        row["parent_sha256_manifest"], row["parent_sha256_csv"] = man.get("parent_sha256"), p["waveP_parent_sha256"]
        if man.get("parent_sha256") != p["waveP_parent_sha256"]:
            problems.append("manifest parent_sha256 differs from parents.csv")
        if man.get("parent_checkpoint") != p["waveP_parent_path"]:
            problems.append("manifest parent_checkpoint differs from parents.csv")
        elif sha256_file(Path(p["waveP_parent_path"])) != p["waveP_parent_sha256"]:
            problems.append("the parent file's SHA-256 differs from parents.csv")
    row.update(ok=not problems, problems=problems)
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI; exit code 0 iff every run passes every check."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code-commit", required=True)
    ap.add_argument("--root", default=str(L.R2B_ROOT))
    ap.add_argument("--r1-root", default=str(R1_DEFAULT), help="R1 results root with parents_A/ and stage2/")
    ap.add_argument("--parents", default=str(L.R2B_ROOT / "parents.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.parents) as f:
        parents = {(int(r["q"]), int(r["seed"])): r for r in csv.DictReader(f)}
    rows: List[Dict[str, Any]] = []
    cache: Dict[str, Optional[List[str]]] = {}
    for wave, arms in (("r2b_waveA", L.R2B_WAVEA_ARMS), ("r2b_waveP", L.R2B_WAVEP_ARMS)):
        for q in L.DEFAULT_QS:
            for s in L.DEFAULT_SEEDS:
                for arm in arms:
                    rows.append(check_run(wave, arm, q, s, Path(a.root), Path(a.r1_root), parents,
                                          a.code_commit, cache))
    n_ok = sum(1 for r in rows if r["ok"])
    summary = {"tool": "tools/v2/r2b_launch_checks.py", "code_commit": a.code_commit, "root": a.root,
               "r1_root": a.r1_root, "parents": a.parents, "n": len(rows), "n_ok": n_ok,
               "ALL": n_ok == len(rows), "runs": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"launch checks {n_ok}/{len(rows)} ALL={summary['ALL']}")
    for r in rows:
        if not r["ok"]:
            print(f"  {r['wave']} q{r['q']} seed{r['seed']} {r['arm']}: {r['problems']}")
    return 0 if summary["ALL"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
