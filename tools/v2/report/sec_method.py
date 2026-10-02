"""Pack items T13-T18: Part I section 4, method change (b) (stagewise learning with frozen continuation).

T13 flags, T14 snapshot drift, T15 C7 regression per launch commit, T16 test suite per launch commit,
T17 RNG streams, T18 reproducibility ledger. Everything is read from saved files (run manifests,
drift tests, checkpoint and update logs, comparison outputs, analysis CSVs, protocol JSONs), from the
code at the base commit (constants parsed with ``ast``), from read-only ``git`` queries (``git diff
--name-only`` / ``git grep`` between fixed commits) or, where no data file exists, from the report
text (cited as "source: report text" with file and line). No training, no forward pass, no verifier
call is performed.
"""

from __future__ import annotations

import ast
import csv
import os
import re
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import common as C
import studies as S

MOD = "sec_method.py"
RUNNER = "run/run_v2_stagewise.py"
LOCKED_RUNNER = "run/run_v2_T2_locked.py"
LAUNCHER = "tools/v2/launch_pilot.py"
PROTO10 = "protocols/v2_T2_locked.json"
PROTO11 = "protocols/v2_T2_locked_v1_1.json"
REG2 = "results/v2_pilots/phase2_regression"
REG1 = "results/v2_pilots/phase1_regression"
C7_REFERENCE = f"{REG1}/before/FINAL_A400_B25_C25/tel_q50_s10501"
LK = "results/v2_T2_locked"
P4A = "results/v2_pilots/pilot4/analysis"
SMOKE_B = "results/v2_pilots/_smoke/smoke_B"
STREAMS = ("env", "learn", "opp", "start", "minibatch")
TRAIN_PATHS = ("agents", "envs", "run", "utils", "protocols", LAUNCHER)

# report files (repo-relative)
R_P0 = "reports/v2/phase0_audit.md"
R_P1 = "reports/v2/phase1_verifier.md"
R_P2 = "reports/v2/phase2_infra.md"
R_PI1 = "reports/v2/pilot1_reward_estimator.md"
R_PI2 = "reports/v2/pilot2_freeze.md"
R_PI3 = "reports/v2/pilot3_continuation_mode.md"
R_EXT = "reports/v2/phaseA_ext.md"
R_PI4 = "reports/v2/pilot4_stabilization.md"
R_LOCK = "reports/v2/protocol_lock_and_rehearsal.md"
R_V11 = "reports/v2/protocol_v1_1_confirmation.md"
R_SUM = "reports/v2/summary.md"

# short study labels, chronological order of launch
LABEL = {
    "smoke_B": "Phase 2 smoke (flow check)",
    "pilot1": "Pilot 1",
    "pilot2": "Pilot 2",
    "pilot3": "Pilot 3",
    "phaseA_ext": "Phase A ext.",
    "pilot4_A": "Pilot 4 2a",
    "pilot4_B": "Pilot 4 2b",
    "pilot4_B_rerun_clean": "dirty-flag re-run",
    "rehearsal": "v1.0 rehearsal",
    "locked_check2_phaseB": "v1.0 Check 2",
    "rehearsal_v1_1": "v1.1 re-rehearsal",
    "confirmation": "confirmation",
}
ORDER = [
    "pilot1",
    "pilot2",
    "pilot3",
    "phaseA_ext",
    "pilot4_A",
    "pilot4_B",
    "pilot4_B_rerun_clean",
    "rehearsal",
    "locked_check2_phaseB",
    "rehearsal_v1_1",
    "confirmation",
]
PHASE_B_STUDIES = (
    "pilot2",
    "pilot3",
    "pilot4_B",
    "pilot4_B_rerun_clean",
    "locked_check2_phaseB",
    "rehearsal",
    "rehearsal_v1_1",
    "confirmation",
)

_CACHE: Dict[str, Any] = {}


# ------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------


def _json(rel: str) -> Any:
    """JSON file (cached)."""
    if rel not in _CACHE:
        _CACHE[rel] = S.read_json(rel)
    return _CACHE[rel]


def _text(rel: str) -> str:
    """Text file under the repo or the results root."""
    return C.abspath(rel).read_text(encoding="utf-8")


def _line(rel: str, needle: str) -> int:
    """1-based number of the first line of a text file containing ``needle`` (raises if absent)."""
    for i, ln in enumerate(_text(rel).splitlines(), 1):
        if needle in ln:
            return i
    raise ValueError(f"{rel}: text not found: {needle!r}")


def _line_text(rel: str, needle: str) -> str:
    """The first line of a text file that contains ``needle``."""
    return _text(rel).splitlines()[_line(rel, needle) - 1]


def _cite(rel: str, needle: str) -> str:
    """``path line N`` reference of a report statement."""
    return f"{rel} line {_line(rel, needle)}"


def _seeds(seeds: Iterable[int]) -> str:
    """Explicit seed list in range notation (``10501-10510``; gaps separated by ``;``)."""
    s = sorted({int(x) for x in seeds})
    out: List[str] = []
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[j] + 1:
            j += 1
        out.append(str(s[i]) if i == j else f"{s[i]}-{s[j]}")
        i = j + 1
    return ";".join(out)


def _qs(df: pd.DataFrame, seedcol: str = "seed") -> str:
    """'q 50, 60 x seeds 10501-10510' from a frame with q and seed columns."""
    qs = ", ".join(str(int(x)) for x in sorted(set(df["q"])))
    return f"q {qs} x seeds {_seeds(df[seedcol])}"


def _code_constants(rel: str, names: Sequence[str]) -> Dict[str, Any]:
    """Literal module-level constants of a Python file (parsed with ``ast``, not imported)."""
    tree = ast.parse(_text(rel))
    out: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = ast.literal_eval(node.value)
    missing = [n for n in names if n not in out]
    if missing:
        raise KeyError(f"{rel}: constants not found {missing}")
    return out


def _git(args: Sequence[str]) -> Optional[str]:
    """Output of a read-only git command in the repository (``None`` on failure)."""
    try:
        r = subprocess.run(
            ["git", "-C", str(C.REPO)] + list(args), capture_output=True, text=True, timeout=60
        )
    except Exception:  # pragma: no cover
        return None
    return r.stdout if r.returncode == 0 else None


def _git_changed(a: str, b: str, paths: Sequence[str]) -> Optional[List[str]]:
    """Files changed between two commits under ``paths`` (read-only ``git diff --name-only``)."""
    out = _git(["diff", "--name-only", a, b, "--"] + list(paths))
    return None if out is None else sorted(x for x in out.splitlines() if x.strip())


def _git_full(c: str) -> str:
    """Full hash of a commit (read-only ``git rev-parse``), or UNKNOWN."""
    out = _git(["rev-parse", c])
    return out.strip() if out else "UNKNOWN"


def _names(files: Optional[List[str]]) -> str:
    """'none', 'UNKNOWN' (git failed) or the comma-joined file list."""
    return "UNKNOWN" if files is None else ("none" if not files else ", ".join(files))


def _xfail_tests(commit: str) -> List[str]:
    """Names of the xfail-marked test functions under tests/ at a commit (read-only ``git grep``)."""
    out = _git(["grep", "-n", "-A6", "pytest.mark.xfail", commit, "--", "tests/"]) or ""
    names = re.findall(r"def (test_\w+)\(", out)
    return sorted(set(names))


def _manual(
    pack: C.Pack, item: str, report: str, label: str, pairs: Sequence[Tuple[Any, ...]]
) -> None:
    """Compare pack values with report text values and record the result like ``Pack.crosscheck``.

    Args:
        pack: The pack.
        item: Pack item id.
        report: Report path (with line reference if useful).
        label: What was compared.
        pairs: ``(quantity, pack_value, report_value_text, ok[, comment])``; ``ok=None`` compares numerically
            at the report's displayed precision (``common.consistent``).
    """
    bad = 0
    for pr in pairs:
        qty, pv, rv, ok = pr[:4]
        comment = pr[4] if len(pr) > 4 else "manual comparison with report text"
        if ok is None:
            ok = C.consistent(float(pv), str(rv))
        if not ok:
            bad += 1
            pack.mismatch(item, qty, pv, report, rv, comment)
    pack.crosschecks.append(
        {
            "item": item,
            "report": report,
            "label": label,
            "n_tables": 0,
            "n_compared": len(pairs),
            "n_mismatch": bad,
            "n_unmatched_rows": 0,
        }
    )


def _intcol(s: pd.Series) -> pd.Series:
    """Integer column with empty cells: Python ints / None in an object column (CSV '10' or '', md '' for None)."""
    return pd.Series(
        [
            None if (v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA) else int(v)
            for v in s
        ],
        index=s.index,
        dtype=object,
    )


def _stats(v: Sequence[float]) -> Dict[str, float]:
    """median, q25, q75, min, max of finite values (numpy linear interpolation)."""
    st = C.median_iqr(v)
    return {k: st[k] for k in ("median", "q25", "q75", "min", "max")}


def _runs(study: str) -> List[Tuple[int, int, str, str]]:
    """``(q, seed, arm, run_dir)`` of a study (``smoke_B``: the Phase 2 smoke Phase-B runs)."""
    if study == "smoke_B":
        return [
            (50, 10501, a, f"{SMOKE_B}/q50/seed10501/{a}")
            for a in ("A_joint", "B1_frozen_allnorm", "B2_frozen_s1norm", "B1_frozen_allnorm_mean")
        ]
    return list(S.iter_runs(study))


def _manifest(rd: str) -> Dict[str, Any]:
    """``manifest.json`` of a run directory (cached)."""
    return _json(f"{rd}/manifest.json")


def _launch_commits(study: str) -> str:
    """Distinct ``git.short`` values of a study's run manifests."""
    return ",".join(sorted({_manifest(rd)["git"]["short"] for _, _, _, rd in _runs(study)}))


# ------------------------------------------------------------------------------------------------
# T15: C7 regression at every launch commit
# ------------------------------------------------------------------------------------------------


def _parse_c7(rel: str) -> Dict[str, Any]:
    """Fields of a ``tools/v2/compare_runs.py`` output file."""
    txt = _text(rel)
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    out: Dict[str, Any] = {"verdict": lines[-1]}
    m = re.search(r"train_history\.json: (\d+) differing leaves; excluded keys: (\[.*?\])", txt)
    out["train_history_leaves_differ"] = int(m.group(1))
    out["train_history_excluded"] = m.group(2)
    m = re.search(r"final_eval\.json: (\d+) differing leaves; excluded keys: (\[.*?\])", txt)
    out["final_eval_leaves_differ"] = int(m.group(1))
    out["final_eval_excluded_n"] = len(ast.literal_eval(m.group(2)))
    n_arr = n_diff = 0
    parts = []
    for f in (
        "arrays.npz",
        "checkpoint_weights.npz",
        "phase_A_exit_arrays.npz",
        "phase_B_exit_arrays.npz",
    ):
        m = re.search(re.escape(f) + r": (\d+) arrays, (\d+) differ", txt)
        n_arr += int(m.group(1))
        n_diff += int(m.group(2))
        parts.append(f"{f} {m.group(1)} arrays/{m.group(2)} differ")
    m = re.search(r"checkpoint\.pt: (\d+) differing tensors", txt)
    out.update(
        npz_arrays_compared=n_arr,
        npz_arrays_differ=n_diff,
        checkpoint_tensors_differ=int(m.group(1)),
    )
    out["compared_fields"] = (
        "train_history.json (all leaves except time_sec): "
        f"{out['train_history_leaves_differ']} differ; final_eval.json (all leaves except "
        f"{out['final_eval_excluded_n']} timing/path keys): {out['final_eval_leaves_differ']} "
        f"differ; " + "; ".join(parts) + "; checkpoint.pt actor/critic/opponent/Adam tensors: "
        f"{out['checkpoint_tensors_differ']} differ"
    )
    return out


_SKIP_KEYS = {
    "pid",
    "host",
    "cmd",
    "output_dir",
    "path",
    "weights_npz",
    "config_ref",
    "git_commit",
    "start_time",
    "end_time",
    "smoke_root",
}


def _json_diffs(a: Any, b: Any) -> int:
    """Number of differing leaves of two JSON trees (same exclusions as tools/v2/compare_runs.py)."""
    if isinstance(a, dict) and isinstance(b, dict):
        n = 0
        for k in set(a) | set(b):
            if k in _SKIP_KEYS or k == "seconds" or k.endswith("_sec") or "wall" in k or "cpu" in k:
                continue
            n += 1 if (k not in a or k not in b) else _json_diffs(a[k], b[k])
        return n
    if isinstance(a, list) and isinstance(b, list):
        return 1 if len(a) != len(b) else sum(_json_diffs(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return 0
    return int(a != b)


def _recheck_run_pair(da: str, db: str) -> Dict[str, int]:
    """Pack re-check of two runner output dirs: NPZ arrays and the two JSON files, bit for bit."""
    n_arr = n_bad = 0
    for f in (
        "arrays.npz",
        "checkpoint_weights.npz",
        "phase_A_exit_arrays.npz",
        "phase_B_exit_arrays.npz",
    ):
        with np.load(C.abspath(f"{da}/{f}")) as A, np.load(C.abspath(f"{db}/{f}")) as B:
            keys = sorted(set(A.files) | set(B.files))
            n_arr += len(keys)
            n_bad += sum(
                1
                for k in keys
                if k not in A.files
                or k not in B.files
                or not np.array_equal(A[k], B[k], equal_nan=A[k].dtype.kind in "fc")
            )
    n_json = sum(
        _json_diffs(_json(f"{da}/{f}"), _json(f"{db}/{f}"))
        for f in ("train_history.json", "final_eval.json")
    )
    return {"arrays": n_arr, "arrays_differ": n_bad, "json_leaves_differ": n_json}


# launch commits in order: (short commit, C7 run dir name or None, role / studies)
LAUNCH = [
    (
        "8f2840e",
        "v2_full",
        "Phase 2 code commit: C7 against the existing runner; Phase 2 smoke runs (6 flow checks)",
        ["smoke"],
    ),
    ("89cd600", "v2_full_89cd600", "Pilot 1 launch (40 runs)", ["pilot1"]),
    ("1791687", "v2_full_1791687", "Pilot 2 launch (60 runs)", ["pilot2"]),
    (
        "cd760fd",
        "v2_full_cd760fd",
        "Pilot 3 (40 runs) and Phase A extension (20 runs) launch; Pilot 2 section 6 "
        "recompute (analysis only)",
        ["pilot3", "phaseA_ext"],
    ),
    ("c92ee74", "v2_full_c92ee74", "Pilot 4 launch (2a and 2b, 80 runs)", ["pilot4_A", "pilot4_B"]),
    ("5d50a9d", None, "dirty-flag re-run launch (1 run)", ["pilot4_B_rerun_clean"]),
    ("4bd2214", "v2_full_4bd2214", "v1.0 lock commit (no runs launched at this commit)", []),
    (
        "5b07293",
        None,
        "v1.0 rehearsal (20 runs) and Check 2 (2 runs) launch",
        ["rehearsal", "locked_check2_phaseB"],
    ),
    ("431474d", "v2_full_431474d", "v1.1 lock commit (no runs launched at this commit)", []),
    (
        "95c000e",
        "v2_full_95c000e",
        "v1.1 re-rehearsal launch (20 runs); LOCK v1.1 record",
        ["rehearsal_v1_1"],
    ),
    ("f6838ec", None, "confirmation launch (40 runs)", ["confirmation"]),
]
# for commits without a C7 run: the C7 commit whose training code they share
NEAREST_C7 = {"5d50a9d": "c92ee74", "5b07293": "4bd2214", "f6838ec": "95c000e"}
C7_REPORT = {  # commit -> (report, needle of the verdict statement, needle of the dirty statement or None)
    "8f2840e": (
        R_P2,
        "**Overall: IDENTICAL, bit-exact on CPU.**",
        'manifest.json` → `"dirty": false`',
    ),
    "89cd600": (
        R_PI1,
        "C7 regression against the existing runner: **IDENTICAL**",
        "C7 regression against the existing runner: **IDENTICAL**",
    ),
    "1791687": (
        R_PI2,
        "C7 against the existing runner: **IDENTICAL**",
        "C7 against the existing runner: **IDENTICAL**",
    ),
    "cd760fd": (R_EXT, "**C7** is still bit-exact at `cd760fd`", None),
    "c92ee74": (R_PI4, "| C7 regression on `c92ee74` |", "| C7 regression on `c92ee74` |"),
    "4bd2214": (R_LOCK, "| C7 at the lock commit |", "| C7 at the lock commit |"),
    "431474d": (
        R_V11,
        "- **C7:** bit-exact at the lock commit `431474d`",
        "- **C7:** bit-exact at the lock commit",
    ),
    "95c000e": (R_V11, "| R6 tests + C7 at `95c000e` |", "| R6 tests + C7 at `95c000e` |"),
}


def _smoke_runs() -> List[str]:
    """Run dirs of the six Phase 2 smoke runs."""
    a = [f"results/v2_pilots/_smoke/smoke_A/q50/seed10501/{x}" for x in ("expected", "sampled")]
    return a + [rd for _, _, _, rd in _runs("smoke_B")]


def _launched_runs(keys: Sequence[str]) -> List[str]:
    """Run dirs of the studies launched at one commit."""
    out: List[str] = []
    for k in keys:
        out += _smoke_runs() if k == "smoke" else [rd for _, _, _, rd in _runs(k)]
    return out


def build_t15(pack: C.Pack) -> pd.DataFrame:
    """T15: C7 regression at every launch commit (compare file, verdict, dirty flag), with a pack re-check."""
    rows, sources, checks = [], [], []
    sources += [
        C.src(f"{C7_REFERENCE}/{f}")
        for f in (
            "arrays.npz",
            "checkpoint_weights.npz",
            "phase_A_exit_arrays.npz",
            "phase_B_exit_arrays.npz",
            "train_history.json",
            "final_eval.json",
        )
    ]
    run_srcs: List[str] = []
    for commit, rdname, role, keys in LAUNCH:
        runs = _launched_runs(keys)
        run_srcs += [f"{rd}/manifest.json" for rd in runs]
        n_dirty = sum(1 for rd in runs if _manifest(rd)["git"]["dirty"] is True)
        launch_commits = sorted({_manifest(rd)["git"]["short"] for rd in runs})
        rec: Dict[str, Any] = {
            "commit": commit,
            "commit_full": _git_full(commit),
            "studies_launched": role,
            "runs_launched": len(runs),
            "runs_launched_dirty": n_dirty,
            "runs_launched_commit": ",".join(launch_commits) if runs else "",
        }
        if rdname is None:
            base = NEAREST_C7[commit]
            ch = _git_changed(base, commit, TRAIN_PATHS)
            allch = _git_changed(base, commit, [".", ":(exclude)results"])
            rec.update(
                compare_file="none",
                reference_run="",
                regression_run_dir="",
                manifest_commit="",
                dirty=np.nan,
                regression_mode="",
                compared_fields="",
                train_history_leaves_differ=np.nan,
                final_eval_leaves_differ=np.nan,
                npz_arrays_compared=np.nan,
                npz_arrays_differ=np.nan,
                checkpoint_tensors_differ=np.nan,
                verdict="not run (no compare file at this commit)",
                pack_recheck_arrays=np.nan,
                pack_recheck_arrays_differ=np.nan,
                pack_recheck_json_leaves_differ=np.nan,
                training_code_changes=(
                    f"vs {base}: {_names(ch)} (paths {', '.join(TRAIN_PATHS)}); "
                    f"number of files changed outside results/: "
                    f"{'UNKNOWN' if allch is None else len(allch)}"
                ),
                report_statement="",
            )
            if ch is None:
                pack.unknown_value(
                    "T15", f"training-code changes {base}..{commit}", "git diff failed"
                )
        else:
            cmp_rel = (
                f"{REG2}/compare_vs_existing_runner.txt"
                if commit == "8f2840e"
                else f"{REG2}/{rdname}.compare.txt"
            )
            rd = f"{REG2}/{rdname}"
            sources += [C.src(cmp_rel), C.src(f"{rd}/manifest.json")]
            for f in (
                "arrays.npz",
                "checkpoint_weights.npz",
                "phase_A_exit_arrays.npz",
                "phase_B_exit_arrays.npz",
                "train_history.json",
                "final_eval.json",
            ):
                sources.append(C.src(f"{rd}/{f}"))
            p = _parse_c7(cmp_rel)
            man = _manifest(rd)
            chk = _recheck_run_pair(C7_REFERENCE, rd)
            fl = man["flags"]
            mode = (
                f"mode {man['mode']}, fixed_budget {str(man['fixed_budget']).lower()}, flags "
                + "/".join(
                    fl[k]
                    for k in (
                        "reward_mode",
                        "stage2_update_mode",
                        "adv_norm_scope",
                        "continuation_action_mode",
                    )
                )
            )
            rep, needle, dneedle = C7_REPORT[commit]
            vline = _line_text(rep, needle)
            rec.update(
                compare_file=cmp_rel,
                reference_run=C7_REFERENCE,
                regression_run_dir=rd,
                manifest_commit=man["git"]["short"],
                dirty=bool(man["git"]["dirty"]),
                regression_mode=mode,
                compared_fields=p["compared_fields"],
                train_history_leaves_differ=p["train_history_leaves_differ"],
                final_eval_leaves_differ=p["final_eval_leaves_differ"],
                npz_arrays_compared=p["npz_arrays_compared"],
                npz_arrays_differ=p["npz_arrays_differ"],
                checkpoint_tensors_differ=p["checkpoint_tensors_differ"],
                verdict=p["verdict"],
                pack_recheck_arrays=chk["arrays"],
                pack_recheck_arrays_differ=chk["arrays_differ"],
                pack_recheck_json_leaves_differ=chk["json_leaves_differ"],
                training_code_changes="",
                report_statement=f"{_cite(rep, needle)}: " + vline.strip()[:240],
            )
            rep_identical = ("IDENTICAL" in vline) or ("bit-exact" in vline)
            checks.append(
                (
                    f"C7 verdict at {commit}",
                    p["verdict"],
                    f"{_cite(rep, needle)}: "
                    f"{'IDENTICAL/bit-exact' if rep_identical else vline.strip()[:80]}",
                    rep_identical == (p["verdict"] == "IDENTICAL"),
                )
            )
            if dneedle is not None:
                dline = _line_text(rep, dneedle)
                rep_false = ("dirty: false" in dline) or ('"dirty": false' in dline)
                checks.append(
                    (
                        f"C7 manifest git.dirty at {commit}",
                        str(man["git"]["dirty"]).lower(),
                        f"{_cite(rep, dneedle)}: {'dirty: false' if rep_false else dline.strip()[:80]}",
                        rep_false == (man["git"]["dirty"] is False),
                    )
                )
        rows.append(rec)
    df = pd.DataFrame(rows)
    cols = [
        "commit",
        "commit_full",
        "studies_launched",
        "runs_launched",
        "runs_launched_commit",
        "runs_launched_dirty",
        "compare_file",
        "reference_run",
        "regression_run_dir",
        "manifest_commit",
        "dirty",
        "regression_mode",
        "compared_fields",
        "train_history_leaves_differ",
        "final_eval_leaves_differ",
        "npz_arrays_compared",
        "npz_arrays_differ",
        "checkpoint_tensors_differ",
        "verdict",
        "pack_recheck_arrays",
        "pack_recheck_arrays_differ",
        "pack_recheck_json_leaves_differ",
        "training_code_changes",
        "report_statement",
    ]
    df = df[cols]
    for c in (
        "runs_launched",
        "runs_launched_dirty",
        "train_history_leaves_differ",
        "final_eval_leaves_differ",
        "npz_arrays_compared",
        "npz_arrays_differ",
        "checkpoint_tensors_differ",
        "pack_recheck_arrays",
        "pack_recheck_arrays_differ",
        "pack_recheck_json_leaves_differ",
    ):
        df[c] = _intcol(df[c])
    _manual(
        pack,
        "T15",
        "reports/v2 (C7 statements of each report)",
        "C7 verdict and dirty flag per commit",
        [(q, v, r, ok) for q, v, r, ok in checks],
    )
    mchecks = []
    for keys, commit, rep, needle, n_exp, d_exp in [
        (["smoke"], "8f2840e", R_P2, "| sampled (A) | done | 8f2840e / false |", None, 0),
        (
            ["pilot1"],
            "89cd600",
            R_PI1,
            "(40/40 manifests: `git.short = 89cd600`, `dirty = false`)",
            40,
            0,
        ),
        (
            ["pilot2"],
            "1791687",
            R_PI2,
            "(all 60 manifests: `git.short = 1791687`, `dirty = false`",
            60,
            0,
        ),
        (
            ["pilot3"],
            "cd760fd",
            R_PI3,
            "| **Launch commit for all 40 runs** | **`cd760fd`** (all manifests `dirty = false`",
            40,
            0,
        ),
        (
            ["phaseA_ext"],
            "cd760fd",
            R_EXT,
            "| **Launch commit for all 20 runs** | **`cd760fd`** (all manifests",
            20,
            0,
        ),
        (
            ["pilot4_A", "pilot4_B"],
            "c92ee74",
            R_PI4,
            "| Dirty flag | `false` in 72 runs; `true` in 8 runs",
            80,
            8,
        ),
        (
            ["pilot4_B_rerun_clean"],
            "5d50a9d",
            R_LOCK,
            "on a clean tree (`5d50a9d`, manifest `dirty: false`)",
            1,
            0,
        ),
        (
            ["rehearsal"],
            "5b07293",
            R_LOCK,
            "- All 20 manifests: commit `5b07293`, `clean_tree: true`",
            20,
            0,
        ),
        (["locked_check2_phaseB"], "5b07293", R_LOCK, "(commit `5b07293`, clean)", 2, 0),
        (["rehearsal_v1_1"], "95c000e", R_V11, "| R4 manifests | **pass** | 20/20:", 20, 0),
        (["confirmation"], "f6838ec", R_V11, "All 40 manifests show v1.1", 40, 0),
    ]:
        runs = _launched_runs(keys)
        n_c = sum(1 for rd in runs if _manifest(rd)["git"]["short"] == commit)
        n_d = sum(1 for rd in runs if _manifest(rd)["git"]["dirty"] is True)
        if n_exp is None:  # the smoke checklist lists one row per run
            n_exp = sum(1 for ln in _text(rep).splitlines() if "| done | 8f2840e / false |" in ln)
        ref = f"{_cite(rep, needle)}"
        mchecks += [
            (
                f"{'+'.join(keys)}: runs whose manifest commit is {commit}",
                n_c,
                f"{n_exp} ({ref})",
                n_c == n_exp,
            ),
            (f"{'+'.join(keys)}: runs with git.dirty true", n_d, f"{d_exp} ({ref})", n_d == d_exp),
        ]
    p4d = sorted(
        (q, s_, a) for q, s_, a, rd in _runs("pilot4_B") if _manifest(rd)["git"]["dirty"] is True
    )
    want = sorted(
        (60, s_, a) for s_ in range(10507, 10511) for a in ("B2_mean_constant", "B2_mean_decay")
    )
    mchecks.append(
        (
            "Pilot 4 dirty runs = 2b q=60 seeds 10507-10510, both arms",
            str(p4d),
            "2b, q=60, seeds 10507-10510, both arms",
            p4d == want,
        )
    )
    locked_clean = [
        rd for k in ("rehearsal", "rehearsal_v1_1", "confirmation") for _, _, _, rd in _runs(k)
    ]
    n_ct = sum(1 for rd in locked_clean if _manifest(rd).get("clean_tree") is True)
    mchecks.append(
        (
            "locked runs with manifest clean_tree true",
            n_ct,
            f"{len(locked_clean)} (all rehearsal, re-rehearsal and confirmation manifests)",
            n_ct == len(locked_clean),
        )
    )
    _manual(
        pack,
        "T15",
        "reports/v2 (launch commit and dirty statements)",
        "manifest commit and dirty flag of the launched runs vs report statements",
        mchecks,
    )
    sources.append(
        C.srcs(
            sorted(set(run_srcs)), label="manifest.json of every run launched at the listed commits"
        )
    )
    sources += [C.src(r) for r in sorted({v[0] for v in C7_REPORT.values()})]
    _CACHE["t15"] = df
    reg_m = _manifest(f"{REG2}/v2_full")
    docs = {
        "commit": "Launch or lock commit (short hash); chronological order",
        "commit_full": "Full commit hash (read-only git rev-parse)",
        "studies_launched": "What was launched at this commit (source: report text and the study registry)",
        "runs_launched": "Number of training runs whose manifest records this launch (0 for lock-only commits)",
        "runs_launched_commit": "Distinct git.short values in the manifests of those runs (check of the launch commit)",
        "runs_launched_dirty": "Runs launched at this commit whose manifest has git.dirty = true (untracked or modified "
        "files outside results/ at launch)",
        "compare_file": "C7 output of tools/v2/compare_runs.py (v2 runner in mode full vs the existing runner); 'none' "
        "when no C7 run exists for this commit",
        "reference_run": "Existing-runner output (Phase 1 'before' run, commit 1ad3805): the C7 reference named in "
        "reports/v2/phase2_infra.md section 3 and the reference of the pack re-check (the later "
        "compare files do not name their reference)",
        "regression_run_dir": "Output directory of the v2 runner's C7 run at this commit",
        "manifest_commit": "git.short recorded in the C7 run's manifest.json (equals the commit column when present)",
        "dirty": {
            "definition": "git.dirty of the C7 run's manifest (tracked changes or untracked files outside "
            "results/ at launch); empty when no C7 run exists",
            "units": "bool",
        },
        "regression_mode": "Mode, budget semantics and the four flags of the C7 run (from its manifest)",
        "compared_fields": "What the compare file reports as compared (parsed from the file)",
        "train_history_leaves_differ": {
            "definition": "Differing leaves of train_history.json (excluding time_sec), "
            "parsed from the compare file",
            "units": "count",
        },
        "final_eval_leaves_differ": {
            "definition": "Differing leaves of final_eval.json (excluding timing and path "
            "keys), parsed from the compare file",
            "units": "count",
        },
        "npz_arrays_compared": {
            "definition": "Arrays compared in arrays.npz, checkpoint_weights.npz and the two "
            "phase-exit NPZs (parsed from the compare file)",
            "units": "count",
        },
        "npz_arrays_differ": {
            "definition": "Arrays that differ among those (parsed)",
            "units": "count",
        },
        "checkpoint_tensors_differ": {
            "definition": "Differing tensors of checkpoint.pt (actor, critic, opponent, Adam "
            "states), parsed from the compare file",
            "units": "count",
        },
        "verdict": "Last line of the compare file (IDENTICAL = 0 differences), or 'not run'",
        "pack_recheck_arrays": {
            "definition": "Pack re-check: arrays of the same four NPZ files compared bit for bit "
            "between the C7 run and the reference run",
            "units": "count",
        },
        "pack_recheck_arrays_differ": {
            "definition": "Pack re-check: arrays that differ",
            "units": "count",
        },
        "pack_recheck_json_leaves_differ": {
            "definition": "Pack re-check: differing leaves of train_history.json and "
            "final_eval.json with the exclusions of "
            "tools/v2/compare_runs.py",
            "units": "count",
        },
        "training_code_changes": "For commits without a C7 run: files changed relative to the nearest C7 commit under "
        "the training and launch paths (read-only git diff --name-only)",
        "report_statement": "The C7 statement of the report that covers this commit (source: report text)",
    }
    notes = (
        "One row per launch or lock commit. Compare files parsed; verdict = last line. Dirty flag from the C7 run's "
        "manifest.json (git.dirty). Pack re-check: the four NPZ files and the two JSON files of each C7 run were "
        "compared again with the existing-runner reference run (checkpoint.pt was not re-read). Commits without a "
        "C7 run: training-code identity with the nearest C7 commit by read-only git diff. Launched-run counts and "
        "dirty counts from the run manifests."
    )
    pack.table(
        "T15",
        df,
        status="generated",
        sources=sources,
        script=f"{MOD}:build_t15",
        notes=notes,
        docs=docs,
        tier="n/a",
        caption="C7 = the v2 runner in regression mode (mode full, default flags, fixed_budget "
        f"false) reproduces the existing runner's outputs bit for bit (q={reg_m['q']}, seed "
        f"{reg_m['seed']}, phase caps "
        + "/".join(str(reg_m["resolved_protocol"]["phase_caps"][p_]) for p_ in "ABC")
        + " from the regression run config).",
    )
    return df


# ------------------------------------------------------------------------------------------------
# T16: test suite at every launch commit
# ------------------------------------------------------------------------------------------------

PYTEST_LOG = f"{LK}/rehearsal_v1_1_pytest.txt"
R16_CHECKS = f"{LK}/rehearsal_v1_1_checks.json"
KNOWN_FAIL = "tests/test_registry_canonicalization.py::test_registry_canonicalization"
# commit, role, scope (source: report text), report, needle (None: no report states a result)
T16_ROWS = [
    (
        "b2bfec0",
        "Phase 1 code commit (verifier metrics; B7 regression)",
        "tests/test_v2_verifier.py",
        R_P1,
        "Result: **21 passed in 1.76 s.**",
    ),
    (
        "8f2840e",
        "Phase 2 code commit (smoke runs, C7)",
        "tests/test_v2_verifier.py tests/test_v2_infra.py",
        R_P2,
        "Result at `8f2840e`: **48 passed, 2 xfailed, 67.1 s.**",
    ),
    (
        "89cd600",
        "Pilot 1 launch",
        "tests/test_v2_verifier.py + tests/test_v2_infra.py",
        R_PI1,
        "Tests: `53 passed, 2 xfailed`",
    ),
    ("1791687", "Pilot 2 launch", "not stated ('Suite')", R_PI2, "Suite: `55 passed, 2 xfailed`."),
    ("cd760fd", "Pilot 3 and Phase A extension launch", "", None, None),
    ("c92ee74", "Pilot 4 launch", "pytest tests/ (full suite)", R_PI4, "| Tests on `c92ee74` |"),
    ("5d50a9d", "dirty-flag re-run launch", "", None, None),
    (
        "4bd2214",
        "v1.0 lock commit",
        "full suite on the lock content (pytest tests/)",
        R_LOCK,
        "Full suite on the lock content: **87 passed, 2 xfailed, 1 failed**",
    ),
    ("5b07293", "v1.0 rehearsal and Check 2 launch", "", None, None),
    (
        "431474d",
        "v1.1 lock commit",
        "full suite at the lock content (pytest tests/)",
        R_V11,
        "at the lock content: **103 passed, 2 xfailed, 1 failed**",
    ),
    (
        "95c000e",
        "v1.1 re-rehearsal launch (check R6)",
        "pytest tests/ -q (full suite)",
        PYTEST_LOG,
        "passed",
    ),
    ("f6838ec", "confirmation launch", "", None, None),
]
T16_PREV = {"cd760fd": "1791687", "5d50a9d": "c92ee74", "5b07293": "4bd2214", "f6838ec": "95c000e"}
# report line that names the failing test (rows with failures)
T16_FAIL_NEEDLE = {
    "c92ee74": "| Tests on `c92ee74` |",
    "4bd2214": "Full suite on the lock content",
    "431474d": "The single failure is the known",
}


def _counts(line: str) -> Dict[str, Any]:
    """passed / xfailed / failed counts and wall seconds of a pytest summary line."""
    g = lambda pat: int(re.search(pat, line).group(1)) if re.search(pat, line) else 0  # noqa: E731
    w = re.search(r"(?:in |, )([\d.]+) ?s\b", line)
    return {
        "passed": g(r"(\d+) passed"),
        "xfailed": g(r"(\d+) xfailed"),
        "failed": g(r"(?<![x\w])(\d+) failed"),
        "wall_s": float(w.group(1)) if w else np.nan,
    }


def build_t16(pack: C.Pack) -> pd.DataFrame:
    """T16: test-suite results per launch commit (report text and the R6 pytest log); UNKNOWN where not reported."""
    log = _text(PYTEST_LOG)
    summary = (
        [ln for ln in log.splitlines() if re.search(r"\d+ passed", ln)][-1].strip("= ").strip()
    )
    failed_names = re.findall(r"^FAILED (\S+)", log, flags=re.M)
    msg = re.search(r"^E\s+(AssertionError: .*)$", log, flags=re.M).group(1).strip()
    r6 = _json(R16_CHECKS)["R6"]
    rows = []
    checks = []
    for commit, role, scope, rep, needle in T16_ROWS:
        xf = _xfail_tests(commit)
        rec: Dict[str, Any] = {"commit": commit, "role": role, "test_scope": scope}
        if rep is None:
            prev = T16_PREV[commit]
            tch = _git_changed(prev, commit, ["tests"])
            cch = _git_changed(prev, commit, ["agents", "envs", "run", "utils"])
            rec.update(
                passed="UNKNOWN",
                xfailed="UNKNOWN",
                failed="UNKNOWN",
                failed_tests="UNKNOWN",
                wall_s=np.nan,
                source_kind="none",
                source="",
                note=(
                    f"No report or log states a test-suite result at {commit}; the suite may not be re-run at "
                    f"old commits. Relative to {prev}: tests/ changes "
                    f"{_names(tch)}; training-code changes (agents, envs, run, utils) "
                    f"{_names(cch)} (read-only git diff)."
                ),
            )
            pack.unknown_value(
                "T16",
                f"test-suite result at {commit} ({role})",
                "no report or log states it; re-running the suite at old commits is not allowed",
            )
        else:
            line = summary if rep == PYTEST_LOG else _line_text(rep, needle)
            c = _counts(line)
            src = (
                f"{PYTEST_LOG} (summary line) and {R16_CHECKS} R6"
                if rep == PYTEST_LOG
                else f"{_cite(rep, needle)}"
            )
            if c["failed"] > 0:
                fnames = failed_names if rep == PYTEST_LOG else [KNOWN_FAIL]
                ftxt = (
                    "; ".join(fnames)
                    + f" ({msg}; pre-existing, fails identically on main; checks the paper "
                    "registry against data on disk, not v2 code)"
                )
            else:
                ftxt = "none"
            if c["failed"] > 0 and rep != PYTEST_LOG:
                fl_needle = T16_FAIL_NEEDLE[commit]
                checks.append(
                    (
                        f"failing test named in the report at {commit}",
                        KNOWN_FAIL.split("::")[1],
                        _cite(rep, fl_needle),
                        "test_registry_canonicalization" in _line_text(rep, fl_needle),
                    )
                )
                if fl_needle != needle:
                    src += f"; failing test named at {_cite(rep, fl_needle)}"
            rec.update(
                passed=c["passed"],
                xfailed=c["xfailed"],
                failed=c["failed"],
                failed_tests=ftxt,
                wall_s=c["wall_s"],
                source_kind="pytest log" if rep == PYTEST_LOG else "report text",
                source=src,
                note="",
            )
            if rep == PYTEST_LOG:
                r6line = _line_text(R_V11, "| R6 tests + C7 at `95c000e` |")
                rc = _counts(r6line)
                same = _line_text(
                    R_V11, "at the rehearsal launch commit `95c000e` (R6): the same result"
                )
                rl = _counts(_line_text(R_V11, "at the lock content: **103 passed"))
                checks += [
                    (
                        "R6 pytest passed (log vs report R6 row)",
                        c["passed"],
                        str(rc["passed"]),
                        None,
                    ),
                    (
                        "R6 pytest xfailed (log vs report R6 row)",
                        c["xfailed"],
                        str(rc["xfailed"]),
                        None,
                    ),
                    (
                        "R6 pytest failed (log vs report: 'the same result' as the lock content)",
                        c["failed"],
                        str(rl["failed"]),
                        None if "the same result" in same else False,
                    ),
                    (
                        "R6 failure is the known test (log vs report R6 row)",
                        ";".join(failed_names),
                        "only the known failure",
                        "only the known failure" in r6line and failed_names == [KNOWN_FAIL],
                    ),
                    (
                        "R6 JSON summary equals the log summary",
                        summary,
                        r6["pytest_summary"],
                        r6["pytest_summary"] == summary,
                    ),
                    (
                        "R6 JSON failed test equals the log",
                        ";".join(failed_names),
                        ";".join(r6["pytest_failed"]),
                        failed_names == r6["pytest_failed"],
                    ),
                ]
        rec["xfail_marked_tests_at_commit"] = (
            (", ".join(xf) + " (parametrized over q = 50, 60)") if xf else "none"
        )
        rows.append(rec)
    df = pd.DataFrame(rows)[
        [
            "commit",
            "role",
            "test_scope",
            "passed",
            "xfailed",
            "failed",
            "failed_tests",
            "xfail_marked_tests_at_commit",
            "wall_s",
            "source_kind",
            "source",
            "note",
        ]
    ]
    _manual(
        pack,
        "T16",
        f"{R_PI4}, {R_LOCK}, {R_V11} and {PYTEST_LOG}",
        "known failing test named in each report; R6 pytest counts: log, checks JSON and report",
        checks,
    )
    sources = [C.src(r) for r in (R_P1, R_P2, R_PI1, R_PI2, R_PI4, R_LOCK, R_V11)] + [
        C.src(PYTEST_LOG),
        C.src(R16_CHECKS),
    ]
    docs = {
        "commit": "Launch, lock or code commit (short hash)",
        "role": "What the commit is (launch of which study, lock, code commit)",
        "test_scope": "Test files or command of the reported run (source: report text); empty when no result is reported",
        "passed": {
            "definition": "Tests passed in the reported run; UNKNOWN when no report or log states a result",
            "units": "count",
        },
        "xfailed": {
            "definition": "Tests that failed as expected (strict xfail) in the reported run; UNKNOWN when not "
            "reported",
            "units": "count",
        },
        "failed": {
            "definition": "Tests that failed in the reported run; UNKNOWN when not reported",
            "units": "count",
        },
        "failed_tests": "Name and message of each failing test (the known pre-existing failure); message from the R6 "
        "pytest log",
        "xfail_marked_tests_at_commit": "Test functions carrying pytest.mark.xfail under tests/ at the commit (read-only "
        "git grep); each is parametrized over the two q values, hence 2 xfailed",
        "wall_s": {
            "definition": "Wall time of the test run where the report or log states it",
            "units": "seconds",
        },
        "source_kind": "report text / pytest log / none",
        "source": "File and line of the stated result",
    }
    notes = (
        "source: report text (counts parsed from the cited report lines) for every row except 95c000e, whose counts "
        "come from the saved pytest log (and agree with the R6 entry of rehearsal_v1_1_checks.json and with the "
        "report). The test scope differs by commit: Phase 1 ran tests/test_v2_verifier.py only; Phase 2 to Pilot 2 "
        "the two v2 test files; from Pilot 4 the full tests/ directory, which contains the pre-existing failing "
        "test_registry_canonicalization. failed = 0 where the reported pytest summary lists no failure. The "
        "failure message is quoted from the R6 log (the Pilot 4 and lock reports quote the same message). "
        "UNKNOWN: commits whose result no report states (not re-run, by rule)."
    )
    pack.table(
        "T16",
        df,
        status="generated",
        sources=sources,
        script=f"{MOD}:build_t16",
        notes=notes,
        docs=docs,
        tier="n/a",
    )
    return df


# ------------------------------------------------------------------------------------------------
# T13: flags
# ------------------------------------------------------------------------------------------------


def _fmt_lr_decay(dec: Any) -> str:
    """Readable form of an lr_decay config value."""
    if dec is None:
        return "none"
    wins = dec if isinstance(dec, list) else [dec]
    return " + ".join(
        f"{w['phase']} local {w['local_first']}-{w['local_last']}: {w['start_lr']:g}->{w['end_lr']:g}"
        for w in wins
    )


def _flag_value(m: Dict[str, Any], flag: str) -> str:
    """Value of a flag / control in a run manifest, as text."""
    if flag in ("reward_mode", "stage2_update_mode", "adv_norm_scope", "continuation_action_mode"):
        return str(m["flags"][flag])
    if flag == "fixed_budget":
        return str(bool(m["fixed_budget"])).lower()
    if flag == "mode":
        return str(m["mode"])
    if flag == "lr_decay":
        return _fmt_lr_decay(m["resolved_config"].get("lr_decay"))
    if flag == "phase_caps":
        caps = m["resolved_protocol"]["phase_caps"]
        if "MODE_PHASES" not in _CACHE:
            _CACHE["MODE_PHASES"] = _code_constants(RUNNER, ["MODE_PHASES"])["MODE_PHASES"]
        return ", ".join(f"{p} {caps[p]}" for p in _CACHE["MODE_PHASES"][m["mode"]])
    if flag == "full_state_at":
        v = m.get("input_config", {}).get("full_state_at")
        return "none" if not v else ",".join(str(int(x)) for x in v)
    if flag == "parent_file":
        p = m.get("parent_checkpoint")
        return "none" if not p else os.path.basename(p)
    raise KeyError(flag)


STAGE1_FLAGS = ("stage2_update_mode", "adv_norm_scope", "continuation_action_mode")
PHASE_A_ONLY = ("pilot1", "phaseA_ext", "pilot4_A")


def _values_by_study(flag: str) -> Tuple[str, Dict[str, set]]:
    """Per study (chronological) the values of a flag by arm, from the run manifests."""
    parts, per_study = [], {}
    for st in ORDER:
        by_arm: Dict[str, set] = {}
        for q, seed, arm, rd in _runs(st):
            by_arm.setdefault(arm, set()).add(_flag_value(_manifest(rd), flag))
        vals = set().union(*by_arm.values())
        per_study[st] = vals
        tag = " [phase A only]" if (st in PHASE_A_ONLY and flag in STAGE1_FLAGS) else ""
        if len(vals) == 1:
            parts.append(f"{LABEL[st]}{tag}: {next(iter(vals))}")
        else:
            parts.append(
                f"{LABEL[st]}{tag}: "
                + "; ".join(f"{a}={'|'.join(sorted(v))}" for a, v in by_arm.items())
            )
    return " || ".join(parts), per_study


def build_t13(pack: C.Pack) -> pd.DataFrame:
    """T13: v2 flags and pipeline controls: values, legacy default, deciding study, locked value."""
    k = _code_constants(
        RUNNER,
        ["FLAG_KEYS", "DEFAULT_FLAGS", "MODES", "LOCKED_FLAGS", "LR_DECAY_KEYS", "OVERRIDE_KEYS"],
    )
    roll = _code_constants("run/v2_rollout.py", ["REWARD_MODES", "CONT_MODES"])
    la = _code_constants(LAUNCHER, ["ACONT_FULL_STATE_AT", "DECAY_ARMS", "LINEAR_END_LR"])
    rtxt = _text(RUNNER)
    s2 = ast.literal_eval(re.search(r'fl\["stage2_update_mode"\] not in (\(.*?\))', rtxt).group(1))
    an = ast.literal_eval(re.search(r'fl\["adv_norm_scope"\] not in (\(.*?\))', rtxt).group(1))
    p10, p11 = _json(PROTO10), _json(PROTO11)
    ev = p11["evidence"]
    rec50 = p11["records"]["50"]
    p1m = _manifest(S.run_dir("pilot1", 50, 10501, "sampled"))
    legacy_caps = p1m["input_config"]["record"]["protocol"]["phase_caps"]
    lrs = rec50["lr_schedule"]

    def locked(p: Dict[str, Any], flag: str) -> str:
        pl = p["pipeline"]
        if flag in k["FLAG_KEYS"]:
            return str(pl["flags"][flag])
        if flag == "fixed_budget":
            return str(bool(pl["fixed_budget"])).lower()
        if flag == "mode":
            return str(pl["mode"])
        if flag == "lr_decay":
            return _fmt_lr_decay(pl["lr_decay"])
        if flag == "phase_caps":
            caps = {q: p["records"][q]["protocol"]["phase_caps"] for q in ("50", "60")}
            if caps["50"] != caps["60"]:
                raise ValueError("locked phase caps differ by q")
            return (
                ", ".join(f"{ph} {caps['50'][ph]}" for ph in "AB")
                + " (C cap unused: Phase C not run)"
            )
        return ""

    specs = [
        dict(
            flag="reward_mode",
            kind="v2 run flag (config 'flags')",
            values="|".join(roll["REWARD_MODES"]),
            meaning="terminal reward of the rollout: sampled = realized prize minus cost; expected = conditional "
            "expectation w_L + DW F_xi(d + e_own - e_opp) - k e_own^2 given the stage-2 state and both executed "
            "efforts (shocks still drawn, A6); non-terminal rewards unchanged",
            legacy=k["DEFAULT_FLAGS"]["reward_mode"],
            decided=f"Pilot 1 (evidence '{'reward_mode=expected'}': {ev['reward_mode=expected']})",
            constraints="any mode; mode locked requires expected; mode full requires the default (sampled)",
            src="run/v2_rollout.py (module docstring, expected_terminal_reward); "
            + RUNNER
            + ":validate_config",
        ),
        dict(
            flag="stage2_update_mode",
            kind="v2 run flag (config 'flags')",
            values="|".join(s2),
            meaning="joint = the stage-2 policy keeps training in Phase B (shared actor, all rows in the policy loss); "
            "frozen = a deep copy of the actor at the Phase B start plays both players' stage-2 actions, the "
            "policy loss uses stage-1 rows only (exactly zero gradient from stage-2 rows)",
            legacy=k["DEFAULT_FLAGS"]["stage2_update_mode"],
            decided=f"Pilot 2 (evidence 'frozen B2 (adv_norm_scope=stage1_rows)': "
            f"{ev['frozen B2 (adv_norm_scope=stage1_rows)']}; 'no joint training, no Phase C': "
            f"{ev['no joint training, no Phase C']})",
            constraints="frozen only in mode phase_B or locked (in locked it takes effect in Phase B); joint requires "
            "adv_norm_scope=all_rows and continuation_action_mode=stochastic",
            src="agents/ppo_curriculum_v2.py (masked update, freeze_stage2_snapshot); run/v2_rollout.py; "
            + RUNNER
            + ":validate_config",
        ),
        dict(
            flag="adv_norm_scope",
            kind="v2 run flag (config 'flags')",
            values="|".join(an),
            meaning="rows over which the advantage mean and population SD are computed in the frozen-mode update: "
            "all_rows (B1) or stage-1 rows only (B2); the statistics are applied to every row; the minibatch "
            "partition is unchanged",
            legacy=k["DEFAULT_FLAGS"]["adv_norm_scope"],
            decided=f"Pilot 2, arm B2 (evidence 'frozen B2 (adv_norm_scope=stage1_rows)': "
            f"{ev['frozen B2 (adv_norm_scope=stage1_rows)']})",
            constraints="stage1_rows requires stage2_update_mode=frozen; modes phase_A / phase_A_continue require "
            "all_rows (no stage-1 rows)",
            src="agents/ppo_curriculum_v2.py:CurriculumPPOv2.update; "
            + RUNNER
            + ":validate_config",
        ),
        dict(
            flag="continuation_action_mode",
            kind="v2 run flag (config 'flags')",
            values="|".join(roll["CONT_MODES"]),
            meaning="stage-2 actions in Phase B rollouts: stochastic = draws from the frozen Beta; mean = both players "
            "execute the frozen Beta mean (the draws are still made and discarded, A6)",
            legacy=k["DEFAULT_FLAGS"]["continuation_action_mode"],
            decided=f"Pilot 3 (evidence 'continuation_action_mode=mean': {ev['continuation_action_mode=mean']})",
            constraints="mean requires stage2_update_mode=frozen; in mode locked, Phase A (no frozen actor) uses "
            "stochastic",
            src="run/v2_rollout.py (module docstring); "
            + RUNNER
            + ":validate_config and run_phase",
        ),
        dict(
            flag="fixed_budget",
            kind="run-config field",
            values="true|false",
            meaning="true = no early phase exit: every phase runs to its cap, and the update at which the existing "
            "phase rule (k_phase consecutive eligible verifier calls) would have fired is recorded; false = "
            "the existing early-exit rules apply",
            legacy="false (existing stop rules; mode full requires false)",
            decided="Phase 0 audit Q1 proposal and Phase 2 design: every pilot run uses a fixed budget "
            f"({_cite(R_P0, 'with a fixed budget of 600, and skip C')}; {_cite(R_P2, '6. **Fixed budget.**')}); "
            "locked in the protocol pipeline",
            constraints="bool; mode full requires false; mode locked requires true",
            src=RUNNER
            + ":validate_config and run_phase; "
            + LAUNCHER
            + " (docstring: every pilot run uses "
            "fixed_budget=true)",
        ),
        dict(
            flag="mode",
            kind="run-config field",
            values="|".join(k["MODES"]),
            meaning="full = A -> B -> C with the existing rules (regression mode, C7); phase_A = Phase A only, then a "
            "full-state checkpoint; phase_B = Phase B restored from a full-state parent; phase_A_continue = "
            "restore an end-of-A parent and continue Phase A (added cd760fd); locked = Phase A, freeze, Phase B "
            "in one process (added 4bd2214)",
            legacy="full",
            decided=f"Phase A extension and Pilots 2-4 (evidence 'Phase A 1600 updates': {ev['Phase A 1600 updates']}; "
            f"'no joint training, no Phase C': {ev['no joint training, no Phase C']})",
            constraints="phase_B and phase_A_continue need parent_checkpoint and parent_sha256; locked requires the "
            "locked flags and fixed_budget true; full requires the default flags and fixed_budget false",
            src=RUNNER + " (module docstring, MODES, validate_config, execute); " + LOCKED_RUNNER,
        ),
        dict(
            flag="phase_caps",
            kind="run-config field (record protocol.phase_caps, budget_overrides)",
            values="updates per phase (A, B, C)",
            meaning="budget of each phase in updates; with fixed_budget=true every phase runs exactly its cap",
            legacy=", ".join(f"{p} {legacy_caps[p]}" for p in "ABC")
            + " (embedded as-run record; early exit allowed)",
            decided=f"Phase A extension (evidence 'Phase A 1600 updates': {ev['Phase A 1600 updates']}); Phase B 600 "
            f"from the record (Pilots 2-4)",
            constraints="budget_overrides keys limited to "
            + ", ".join(k["OVERRIDE_KEYS"])
            + "; lr_decay.local_last "
            "must equal the phase cap",
            src=RUNNER
            + ":validate_config and Run.__init__; "
            + LAUNCHER
            + ":build_config (ACONT_CAPS)",
        ),
        dict(
            flag="lr_decay",
            kind="run-config field",
            values="none | window {"
            + ", ".join(k["LR_DECAY_KEYS"])
            + "} | list of windows (at most one per phase)",
            meaning="inside the window both optimizers follow the existing linear form lr_at: lr(j) = start + (end - "
            "start)(j - local_first)/(local_last - local_first), applied before each update, Adam state "
            "kept; before local_first the record's schedule applies (added c92ee74; list form 4bd2214)",
            legacy=f"none: the record's lr_schedule (kind {lrs['kind']}, ab_lr {lrs['ab_lr']:g}) in every phase",
            decided=f"Pilot 4 (evidence 'Phase A end decay u1201-1600': {ev['Phase A end decay u1201-1600']}; "
            f"'Phase B decay': {ev['Phase B decay']})",
            constraints="refused in mode full; phase must be run by the mode; 1 <= local_first < local_last = phase cap; "
            "start_lr = ab_lr; launcher decay arms "
            + ", ".join(la["DECAY_ARMS"])
            + f" end at {la['LINEAR_END_LR']:g}",
            src=RUNNER + ":validate_config and Run.lr_for; " + LAUNCHER,
        ),
        dict(
            flag="full_state_at",
            kind="run-config field",
            values="list of global updates (default none)",
            meaning="global updates at which an extra full-state checkpoint state_u<update>.pt is written (actor, "
            "critic, opponent, frozen snapshot, both Adam states, all RNG streams, counters); phase-end states "
            "state_end_<phase>.pt are always written outside mode full (added cd760fd)",
            legacy="none (the legacy runner writes no full-state checkpoint; checkpoint.pt at the C stop has no RNG "
            f"states, {_cite(R_P0, '| numpy RNG states: env_noise, learner_action')})",
            decided="Phase A extension design (mid-run parents u800/u1200/u1600, launcher ACONT_FULL_STATE_AT "
            f"{la['ACONT_FULL_STATE_AT']}); used as the Pilot 4 parents ({_cite(R_PI4, '`--parent-file`')})",
            constraints="must be a list",
            src=RUNNER + ":validate_config and run_phase; " + LAUNCHER,
        ),
        dict(
            flag="parent_file",
            kind="launcher option --parent-file (tools/v2/launch_pilot.py)",
            values="file name inside the parent run directory (default state_end_A.pt)",
            meaning="which full-state file of the parent run a Phase B / Phase A continuation branch restores "
            "(added c92ee74 for the mid-run Pilot 4 parents)",
            legacy="state_end_A.pt (default; the legacy runner has no branching from saved states)",
            decided=f"Pilot 4 ({_cite(R_PI4, '`--parent-file`')})",
            constraints="phase B and Acont launches need --parent-pilot, --parent-arm and --reward-mode",
            src=LAUNCHER + ":main",
        ),
    ]
    rows, checks = [], []
    for sp in specs:
        flag = sp["flag"]
        vbs, per = _values_by_study(flag)
        lv10, lv11 = locked(p10, flag), locked(p11, flag)
        if flag == "full_state_at":
            lv10 = (
                "none ("
                + ", ".join(sorted(per["rehearsal"]))
                + " in the v1.0 rehearsal manifests; build_config)"
            )
            lv11 = (
                "none ("
                + ", ".join(sorted(per["confirmation"] | per["rehearsal_v1_1"]))
                + " in the v1.1 manifests)"
            )
        if flag == "parent_file":
            lv10 = "none (single process; parent_checkpoint null in the v1.0 rehearsal manifests)"
            lv11 = "none (single process; parent_checkpoint null in the v1.1 manifests)"
        # first study (chronological, applicable studies) in which every run uses the v1.1 locked value
        target = {"full_state_at": "none", "parent_file": "none"}.get(flag, lv11.split(" (")[0])
        if flag == "phase_caps":
            target = "A 1600, B 600"
        appl = [s for s in ORDER if (s in PHASE_B_STUDIES or flag not in STAGE1_FLAGS)]
        if flag in ("full_state_at", "parent_file"):
            default = {"none", "state_end_A.pt"}
            used = next((s for s in ORDER if per[s] - default), None)
            first_txt = (
                "n/a: not used by the locked pipeline; first study using a non-default value: "
                + (f"{LABEL[used]} ({_launch_commits(used)})" if used else "none")
            )
        else:
            first = next((s for s in appl if per[s] == {target}), None)
            first_txt = f"{LABEL[first]} ({_launch_commits(first)})" if first else "none"
        if flag in k["FLAG_KEYS"]:
            checks.append(
                (
                    f"locked value of {flag}: protocol v1.1 JSON vs LOCKED_FLAGS in {RUNNER}",
                    lv11,
                    k["LOCKED_FLAGS"][flag],
                    lv11 == k["LOCKED_FLAGS"][flag],
                )
            )
            checks.append(
                (f"locked value of {flag}: v1.0 vs v1.1 protocol JSON", lv10, lv11, lv10 == lv11)
            )
            for st in ("rehearsal", "rehearsal_v1_1", "confirmation"):
                checks.append(
                    (
                        f"{flag} in every {LABEL[st]} manifest equals the protocol",
                        "|".join(sorted(per[st])),
                        lv11,
                        per[st] == {lv11},
                    )
                )
        rows.append(
            {
                "flag": flag,
                "kind": sp["kind"],
                "values": sp["values"],
                "meaning": sp["meaning"],
                "legacy_default": sp["legacy"],
                "values_by_study": vbs,
                "decided_by": sp["decided"],
                "first_study_all_runs_locked_value": first_txt,
                "locked_v1_0": lv10,
                "locked_v1_1": lv11,
                "constraints": sp["constraints"],
                "source": sp["src"],
            }
        )
    df = pd.DataFrame(rows)
    _manual(
        pack,
        "T13",
        f"{PROTO11} (pipeline) vs {RUNNER} and run manifests",
        "locked flags: protocol, code and manifests agree",
        checks,
    )
    man_src = [f"{rd}/manifest.json" for st in ORDER for _, _, _, rd in _runs(st)]
    sources = [
        C.src(RUNNER),
        C.src("run/v2_rollout.py"),
        C.src(LAUNCHER),
        C.src(LOCKED_RUNNER),
        C.src(PROTO10),
        C.src(PROTO11),
        C.src("agents/ppo_curriculum_v2.py"),
        C.src(R_P0),
        C.src(R_P2),
        C.src(R_PI4),
        C.srcs(
            man_src,
            label="manifest.json of every run of Pilots 1-4, the extension, the dirty-flag re-run, "
            "the v1.0 rehearsal and Check 2, the v1.1 re-rehearsal and the confirmation",
        ),
    ]
    docs = {
        "flag": "Flag or pipeline control (run-config key, or launcher option)",
        "kind": "Where the control lives",
        "values": "Allowed values (parsed from the code constants and validate_config)",
        "meaning": "What each value does (source: code docstrings)",
        "legacy_default": "Behaviour of the existing (legacy) runner, i.e. the value with which the v2 runner reproduces "
        "it bit for bit (C7); DEFAULT_FLAGS of run/run_v2_stagewise.py for the four flags",
        "values_by_study": "Values actually recorded in the run manifests, per study in launch order (all arms share "
        "the value unless arms are listed as arm=value); '[phase A only]' marks studies without "
        "stage-1 training, where only the phase-A values are valid",
        "decided_by": "Study (report) that decided the locked value: the protocol's own evidence entry "
        "(protocols/v2_T2_locked_v1_1.json 'evidence') or the cited report line",
        "first_study_all_runs_locked_value": "First study, in launch order, in which every run uses the v1.1 locked value "
        "(stage-1 flags: studies with a Phase B only), with its launch commit from "
        "the manifests; for controls the locked pipeline does not use, the first "
        "study that used a non-default value",
        "locked_v1_0": "Value in protocols/v2_T2_locked.json (v1.0, lock 4bd2214)",
        "locked_v1_1": "Value in protocols/v2_T2_locked_v1_1.json (v1.1, lock 431474d)",
        "constraints": "Allowed combinations enforced by run/run_v2_stagewise.py:validate_config (and Run.__init__)",
        "source": "Code locations",
    }
    notes = (
        "source: code text (constants parsed with ast from run/run_v2_stagewise.py, run/v2_rollout.py, "
        "tools/v2/launch_pilot.py; tuples of validate_config by regex), protocol JSONs (pipeline, records, evidence), "
        "report text for the decisions not recorded in the protocol (cited with line), and every run manifest for "
        "the values used per study. 'Locked' values: v1.0 and v1.1 JSONs (identical for every row)."
    )
    pack.table(
        "T13",
        df,
        status="generated",
        sources=sources,
        script=f"{MOD}:build_t13",
        notes=notes,
        docs=docs,
        tier="n/a",
    )
    return df


# ------------------------------------------------------------------------------------------------
# T14: snapshot drift
# ------------------------------------------------------------------------------------------------

# frozen-mode studies in launch order: (study key, frozen arms)
FROZEN = [
    ("smoke_B", ["B1_frozen_allnorm", "B2_frozen_s1norm", "B1_frozen_allnorm_mean"]),
    ("pilot2", ["B1_frozen_allnorm", "B2_frozen_s1norm"]),
    ("pilot3", ["B2_frozen_s1norm", "B2_frozen_s1norm_mean"]),
    ("pilot4_B", ["B2_mean_constant", "B2_mean_decay"]),
    ("pilot4_B_rerun_clean", ["B2_mean_constant"]),
    ("rehearsal", ["locked"]),
    ("locked_check2_phaseB", ["B2_mean_decay"]),
    ("rehearsal_v1_1", ["locked"]),
    ("confirmation", ["locked"]),
]
QDEF = {
    "drift_test_pass": (
        "runs whose drift_test.json has pass = true (snapshot mean, alpha, beta on the dev D_2 grid "
        "equal at freeze time and at the end of Phase B, snapshot tensors bit-identical to the parent / "
        "end-of-A actor; pilot runs also: no grad, in no optimizer)",
        "count",
    ),
    "snapshot_drift_mean": (
        "max over the dev D_2 grid of |e_hat_2 snapshot(end of B) - e_hat_2 snapshot(freeze time)| "
        "(drift_test.json max_abs_diff_vs_freeze_time.mean); n_meeting = runs with exactly 0; "
        "median..max over runs",
        "effort units [0, 100]",
    ),
    "snapshot_drift_alpha": (
        "as above for the Beta alpha parameter (max_abs_diff_vs_freeze_time.alpha)",
        "Beta parameter units",
    ),
    "snapshot_drift_beta": (
        "as above for the Beta beta parameter (max_abs_diff_vs_freeze_time.beta)",
        "Beta parameter units",
    ),
    "cand_drift_all_checkpoints": (
        "candidate stage-2 drift stage2_drift_cand_maxabs = max over the dev D_2 grid of "
        "|e_hat_2 candidate - e_hat_2 parent| at every Phase B training-time verifier "
        "checkpoint (v2_checkpoints[_B].csv); n_meeting = rows exactly 0, n_of = rows; "
        "median..max over rows",
        "effort units [0, 100]",
    ),
    "live_net_drift_end": (
        "information only: drift of the live network's own stage-2 output (not used for any action, "
        "no stage-2 policy gradient) at the last Phase B checkpoint (stage2_drift_live_maxabs); "
        "median..max over runs",
        "effort units [0, 100]",
    ),
}
JOINT_Q = {
    "stage2_drift_cand_on_max": "max of the candidate's stage-2 drift over the on-path nodes (|d - drift| < 2q), u1000",
    "stage2_drift_cand_on_mean_cellmass_weighted": "exact-cell-mass weighted mean of the drift over the on-path nodes, "
    "u1000",
    "stage2_drift_cand_off_max": "max of the drift over the off-path nodes, u1000",
    "stage2_drift_cand_off_mean_unweighted": "unweighted mean of the drift over the off-path nodes, u1000",
    "full_grid_drift_mean": "max over the whole dev D_2 grid of |e_hat_2 - e_hat_2 parent| at the end of Phase B "
    "(drift_test.json max_abs_diff_vs_freeze_time.mean)",
    "full_grid_drift_alpha": "as above for Beta alpha (drift_test.json)",
    "full_grid_drift_beta": "as above for Beta beta (drift_test.json)",
    "cand_drift_all_checkpoints": QDEF["cand_drift_all_checkpoints"][0],
}


def _ckpt_rel(study: str, rd: str) -> str:
    """Phase B checkpoint CSV of a run."""
    return (
        f"{rd}/v2_checkpoints_B.csv"
        if study in ("rehearsal", "rehearsal_v1_1", "confirmation")
        else f"{rd}/v2_checkpoints.csv"
    )


def _phase_b_ckpts(study: str, rd: str) -> pd.DataFrame:
    """Phase B rows of a run's training-time checkpoint CSV (cached)."""
    rel = _ckpt_rel(study, rd)
    if rel not in _CACHE:
        df = S.read_csv(rel)
        _CACHE[rel] = df[df["phase"] == "B"].reset_index(drop=True)
    return _CACHE[rel]


def _frozen_run_records() -> pd.DataFrame:
    """One row per frozen-mode run: drift test, checkpoint drift and live drift."""
    rows = []
    for st, arms in FROZEN:
        for q, seed, arm, rd in _runs(st):
            if arm not in arms:
                continue
            dt = _json(f"{rd}/drift_test.json")
            mx = dt["max_abs_diff_vs_freeze_time"]
            ck = _phase_b_ckpts(st, rd)
            bitid = dt.get(
                "snapshot_params_bit_identical_to_parent_actor",
                dt.get("snapshot_params_bit_identical_to_end_of_A_actor"),
            )
            rows.append(
                {
                    "study": st,
                    "arm": arm,
                    "q": q,
                    "seed": seed,
                    "run_dir": rd,
                    "commit": _manifest(rd)["git"]["short"],
                    "pass": bool(dt["pass"]),
                    "bit_identical": bool(bitid),
                    "mean": float(mx["mean"]),
                    "alpha": float(mx["alpha"]),
                    "beta": float(mx["beta"]),
                    "n_ckpt": len(ck),
                    "n_ckpt_zero": int((ck["stage2_drift_cand_maxabs"] == 0.0).sum()),
                    "cand_values": ck["stage2_drift_cand_maxabs"].to_numpy(dtype=float),
                    "live_end_ckpt": float(ck["stage2_drift_live_maxabs"].iloc[-1]),
                    "live_end_update": int(ck["update"].iloc[-1]),
                    "live_drift_test": float(
                        dt.get("live_network_stage2_mean_maxabs_drift", np.nan)
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_t14(pack: C.Pack) -> pd.DataFrame:
    """T14: snapshot drift of every frozen-mode study; Pilot 2 joint arm drift on and off path."""
    fr = _frozen_run_records()
    _CACHE["t14_runs"] = fr
    rows: List[Dict[str, Any]] = []
    base = lambda st, arm, q, g: {
        "block": "frozen snapshot integrity",
        "study": LABEL[st],
        "arm": arm,
        "q": q,  # noqa
        "seeds": _seeds(g["seed"]),
        "launch_commit": ",".join(sorted(set(g["commit"]))),
        "n_runs": len(g),
    }
    for st, arms in FROZEN:
        for arm in arms:
            for q in sorted(fr[(fr.study == st) & (fr.arm == arm)].q.unique()):
                g = fr[(fr.study == st) & (fr.arm == arm) & (fr.q == q)]
                b = base(st, arm, int(q), g)
                src = f"{g.run_dir.iloc[0].rsplit('/q', 1)[0]}/q{int(q)}/seed*/" + (
                    "" if arm == "locked" else f"{arm}/"
                )
                for qty, n_meet, n_of, vals, files in [
                    ("drift_test_pass", int(g["pass"].sum()), len(g), None, "drift_test.json"),
                    (
                        "snapshot_drift_mean",
                        int((g["mean"] == 0.0).sum()),
                        len(g),
                        g["mean"],
                        "drift_test.json",
                    ),
                    (
                        "snapshot_drift_alpha",
                        int((g["alpha"] == 0.0).sum()),
                        len(g),
                        g["alpha"],
                        "drift_test.json",
                    ),
                    (
                        "snapshot_drift_beta",
                        int((g["beta"] == 0.0).sum()),
                        len(g),
                        g["beta"],
                        "drift_test.json",
                    ),
                    (
                        "cand_drift_all_checkpoints",
                        int(g["n_ckpt_zero"].sum()),
                        int(g["n_ckpt"].sum()),
                        np.concatenate(g["cand_values"].to_list()),
                        "v2_checkpoints" + ("_B" if arm == "locked" else "") + ".csv",
                    ),
                    (
                        "live_net_drift_end",
                        np.nan,
                        np.nan,
                        g["live_end_ckpt"],
                        "v2_checkpoints"
                        + ("_B" if arm == "locked" else "")
                        + ".csv (last Phase B row)",
                    ),
                ]:
                    rec = dict(b, quantity=qty, unit=QDEF[qty][1], n_meeting=n_meet, n_of=n_of)
                    rec.update(
                        _stats(vals)
                        if vals is not None
                        else {k: np.nan for k in ("median", "q25", "q75", "min", "max")}
                    )
                    rec["source"] = src + files
                    rows.append(rec)
    # Pilot 2 joint arm
    ft_rel = "results/v2_pilots/pilot2/analysis/final_table.csv"
    ft = S.read_csv(ft_rel)
    ja = ft[ft.arm == "A_joint"].sort_values(["q", "seed"]).reset_index(drop=True)
    jrec = []
    for q, seed, arm, rd in _runs("pilot2"):
        if arm != "A_joint":
            continue
        dt = _json(f"{rd}/drift_test.json")
        ck = _phase_b_ckpts("pilot2", rd)
        last = ck.iloc[-1]
        jrec.append(
            {
                "q": q,
                "seed": seed,
                "commit": _manifest(rd)["git"]["short"],
                "mode": dt["mode"],
                "full_grid_drift_mean": float(dt["max_abs_diff_vs_freeze_time"]["mean"]),
                "full_grid_drift_alpha": float(dt["max_abs_diff_vs_freeze_time"]["alpha"]),
                "full_grid_drift_beta": float(dt["max_abs_diff_vs_freeze_time"]["beta"]),
                "last_update": int(last["update"]),
                "n_ckpt": len(ck),
                "n_ckpt_zero": int((ck["stage2_drift_cand_maxabs"] == 0.0).sum()),
                "cand_values": ck["stage2_drift_cand_maxabs"].to_numpy(dtype=float),
                **{c: float(last[c]) for c in list(JOINT_Q)[:4]},
            }
        )
    jr = pd.DataFrame(jrec)
    # consistency of final_table.csv with the per-run checkpoint rows (last row, u1000)
    jchk = []
    for c in list(JOINT_Q)[:4]:
        jchk.append(C.max_abs_diff(ja[c].to_numpy(float), jr[c].to_numpy(float)))
    # final_table.csv and the checkpoint CSVs are written by different scripts (texts differ in the 16th-17th digit);
    # with exact parsing they agree to a few ulps (differences of order 1e-16 on values of 0.15 to 7.1), so the tolerance is 1e-12 absolute
    if max(jchk) > 1e-12 or set(jr["last_update"]) != {1000} or set(jr["mode"]) != {"joint"}:
        raise ValueError(f"Pilot 2 joint drift: final_table.csv vs checkpoints differ {jchk}")
    for q in (50, 60):
        g = ja[ja.q == q]
        gj = jr[jr.q == q]
        b = {
            "block": "Pilot 2 joint arm drift (candidate = live network)",
            "study": LABEL["pilot2"],
            "arm": "A_joint",
            "q": q,
            "seeds": _seeds(g["seed"]),
            "launch_commit": ",".join(sorted(set(gj["commit"]))),
            "n_runs": len(g),
        }
        for qty, defn in JOINT_Q.items():
            if qty in ft.columns:
                vals, n_meet, n_of, src = (
                    g[qty],
                    int((g[qty] == 0.0).sum()),
                    len(g),
                    ft_rel + " (arm A_joint)",
                )
            elif qty == "cand_drift_all_checkpoints":
                vals = np.concatenate(gj["cand_values"].to_list())
                n_meet, n_of = int(gj["n_ckpt_zero"].sum()), int(gj["n_ckpt"].sum())
                src = f"results/v2_pilots/pilot2/q{q}/seed*/A_joint/v2_checkpoints.csv"
            else:
                vals, n_meet, n_of = gj[qty], int((gj[qty] == 0.0).sum()), len(gj)
                src = f"results/v2_pilots/pilot2/q{q}/seed*/A_joint/drift_test.json"
            unit = (
                "Beta parameter units"
                if qty.endswith(("alpha", "beta"))
                else "effort units [0, 100]"
            )
            rows.append(
                dict(
                    b,
                    quantity=qty,
                    unit=unit,
                    n_meeting=n_meet,
                    n_of=n_of,
                    **_stats(vals),
                    source=src,
                )
            )
    df = pd.DataFrame(rows)
    df["n_meeting"] = _intcol(df["n_meeting"])
    df["n_of"] = _intcol(df["n_of"])
    # ---- cross-checks with the reports
    p2 = fr[fr.study == "pilot2"]
    short = {"B1_frozen_allnorm": "B1", "B2_frozen_s1norm": "B2"}
    rep26 = pd.DataFrame(
        [
            {
                "q": int(q),
                "arm": short[a],
                "n": len(g),
                "drift_test_pass": int(g["pass"].sum()),
                "max_mean": g["mean"].max(),
                "max_alpha": g["alpha"].max(),
                "max_beta": g["beta"].max(),
                "live_net_stage2_drift_median": float(np.median(g["live_drift_test"])),
            }
            for (q, a), g in p2.groupby(["q", "arm"])
        ]
    )
    pack.crosscheck(
        "T14",
        rep26,
        R_PI2,
        header_has=["q", "arm", "drift_test_pass", "max_mean"],
        key_map={"q": "q", "arm": "arm"},
        value_map={
            "n": "n",
            "drift_test_pass": "drift_test_pass",
            "max_mean": "max_mean",
            "max_alpha": "max_alpha",
            "max_beta": "max_beta",
            "live_net_stage2_drift_median": "live_net_stage2_drift_median",
        },
        heading_has="Snapshot integrity",
        label="Pilot 2 snapshot integrity per arm (section 2.6)",
    )
    med = (
        ft.groupby(["q", "arm"])[["stage2_drift_cand_on_max", "stage2_drift_cand_off_max"]]
        .median()
        .reset_index()
    )
    med["arm"] = med["arm"].map(
        {"A_joint": "A", "B1_frozen_allnorm": "B1", "B2_frozen_s1norm": "B2"}
    )
    for rep, hd in ((R_PI2, "2.1 Final checkpoint"), (R_SUM, "Pilot 2")):
        pack.crosscheck(
            "T14",
            med,
            rep,
            header_has=["q", "arm", "stage2_drift_cand_on_max", "stage2_drift_cand_off_max"],
            key_map={"q": "q", "arm": "arm"},
            value_map={
                "stage2_drift_cand_on_max": "stage2_drift_cand_on_max",
                "stage2_drift_cand_off_max": "stage2_drift_cand_off_max",
            },
            heading_has=hd,
            label="Pilot 2 median candidate drift on/off path per arm",
        )
    p3 = fr[fr.study == "pilot3"]
    p3t = pd.DataFrame(
        [
            {
                "q": int(q),
                "arm": ("mean" if a.endswith("_mean") else "stochastic"),
                "snapshot_drift_max": max(g["mean"].max(), g["alpha"].max(), g["beta"].max()),
                "drift_test_pass": int(g["pass"].sum()),
            }
            for (q, a), g in p3.groupby(["q", "arm"])
        ]
    )
    pack.crosscheck(
        "T14",
        p3t,
        R_PI3,
        header_has=["q", "arm", "snapshot_drift_max", "drift_test_pass"],
        key_map={"q": "q", "arm": "arm"},
        value_map={
            "snapshot_drift_max": "snapshot_drift_max",
            "drift_test_pass": "drift_test_pass",
        },
        heading_has="Snapshot integrity",
        label="Pilot 3 snapshot drift and drift_test per arm (section 7)",
    )

    def zero_all(g: pd.DataFrame) -> int:
        return int(((g["mean"] == 0) & (g["alpha"] == 0) & (g["beta"] == 0)).sum())

    p4 = fr[fr.study == "pilot4_B"]
    reh = fr[fr.study == "rehearsal"]
    conf = fr[fr.study == "confirmation"]
    jm = ja.groupby("q")[["stage2_drift_cand_on_max", "stage2_drift_cand_off_max"]].median()
    ptxt = _line_text(R_PI2, "median on-path max drift")
    otxt = _line_text(R_PI2, "off-path 2.1 and 1.6")
    num = r"(\d+(?:\.\d+)?)"
    m_on = re.search(
        r"median on-path max drift " + num + r" \(q=50\) and " + num + r" \(q=60\)", ptxt
    )
    m_off = re.search(r"off-path " + num + " and " + num, otxt)
    conf_rep = [t for t in C.parse_md_tables(R_V11) if "drift_test_pass" in t["header"]][0]
    ci = conf_rep["header"].index("drift_test_pass")
    n_yes = sum(1 for r in conf_rep["rows"] if r[ci] == "yes")
    _manual(
        pack,
        "T14",
        R_PI2,
        "Pilot 2 snapshot drift and joint-arm drift (prose)",
        [
            (
                "Pilot 2: frozen runs with output drift exactly 0",
                zero_all(p2),
                f"40 ({_cite(R_PI2, 'exactly 0 in all 40 frozen')})",
                zero_all(p2) == 40 == len(p2),
            ),
            (
                "Pilot 2 A: median on-path max drift q=50",
                float(jm.loc[50, "stage2_drift_cand_on_max"]),
                m_on.group(1),
                None,
            ),
            (
                "Pilot 2 A: median on-path max drift q=60",
                float(jm.loc[60, "stage2_drift_cand_on_max"]),
                m_on.group(2),
                None,
            ),
            (
                "Pilot 2 A: median off-path max drift q=50",
                float(jm.loc[50, "stage2_drift_cand_off_max"]),
                m_off.group(1),
                None,
            ),
            (
                "Pilot 2 A: median off-path max drift q=60",
                float(jm.loc[60, "stage2_drift_cand_off_max"]),
                m_off.group(2),
                None,
            ),
        ],
    )
    _manual(
        pack,
        "T14",
        R_PI3,
        "Pilot 3 snapshot drift (prose)",
        [
            (
                "Pilot 3: runs with drift exactly 0",
                zero_all(p3),
                f"40 ({_cite(R_PI3, 'drift is exactly 0 in all 40 runs')})",
                zero_all(p3) == 40 == len(p3),
            ),
            (
                "Pilot 3: drift_test passes",
                int(p3["pass"].sum()),
                "40/40",
                int(p3["pass"].sum()) == 40,
            ),
        ],
    )
    _manual(
        pack,
        "T14",
        R_PI4,
        "Pilot 4 2b snapshot drift (prose)",
        [
            (
                "Pilot 4 2b: drift_test passes",
                int(p4["pass"].sum()),
                f"40/40 ({_cite(R_PI4, 'Snapshot integrity: `drift_test` passes in 40/40')})",
                int(p4["pass"].sum()) == 40,
            ),
            (
                "Pilot 4 2b: mean/alpha/beta drift exactly 0",
                zero_all(p4),
                "40 (exactly 0)",
                zero_all(p4) == 40,
            ),
        ],
    )
    _manual(
        pack,
        "T14",
        R_LOCK,
        "v1.0 rehearsal drift test (prose)",
        [
            (
                "v1.0 rehearsal: drift test passes",
                int(reh["pass"].sum()),
                f"20/20 ({_cite(R_LOCK, 'The snapshot drift test passes in 20/20')})",
                int(reh["pass"].sum()) == 20,
            )
        ],
    )
    _manual(
        pack,
        "T14",
        R_V11,
        "confirmation drift_test_pass column (reported-metrics table)",
        [
            (
                "confirmation: drift_test_pass = yes in the reported-metrics table",
                int(conf["pass"].sum()),
                f"{n_yes} rows 'yes' (table at line {conf_rep['line']})",
                int(conf["pass"].sum()) == n_yes,
            )
        ],
    )
    live_eq = fr.dropna(subset=["live_drift_test"])
    n_live_eq = int((live_eq["live_drift_test"] == live_eq["live_end_ckpt"]).sum())
    live_mad = C.max_abs_diff(
        live_eq["live_drift_test"].to_numpy(float), live_eq["live_end_ckpt"].to_numpy(float)
    )
    sources = [
        C.srcs(
            [f"{rd}/drift_test.json" for rd in fr.run_dir],
            label="drift_test.json of every frozen-mode run",
        ),
        C.srcs(
            sorted({_ckpt_rel(s, rd) for s, rd in zip(fr.study, fr.run_dir)}),
            label="Phase B checkpoint CSV of every frozen-mode run",
        ),
        C.srcs(
            [f"{rd}/manifest.json" for rd in fr.run_dir],
            label="manifest.json of every frozen-mode run",
        ),
        C.src(ft_rel),
        C.srcs(
            [
                f"{S.run_dir('pilot2', q, s, 'A_joint')}/{f}"
                for q in (50, 60)
                for s in S.SEEDS_DEV
                for f in ("drift_test.json", "v2_checkpoints.csv")
            ],
            label="Pilot 2 A_joint drift_test.json and v2_checkpoints.csv",
        ),
        C.src(R_PI2),
        C.src(R_PI3),
        C.src(R_PI4),
        C.src(R_LOCK),
        C.src(R_V11),
        C.src(R_SUM),
    ]
    docs = {
        "block": "frozen snapshot integrity (every frozen-mode study) or Pilot 2 joint arm drift",
        "study": "Study (launch order); Phase 2 smoke = reduced-budget flow check at 8f2840e, not a study",
        "arm": "Arm directory label (locked = the locked pipeline)",
        "seeds": "Seeds of the runs behind the row (range notation)",
        "launch_commit": "git.short of the runs' manifests",
        "n_runs": {"definition": "Runs behind the row", "units": "count"},
        "quantity": "Quantity of the row. "
        + " ".join(f"{k}: {v[0]}." for k, v in QDEF.items())
        + " Joint arm: "
        + " ".join(f"{k}: {v}." for k, v in JOINT_Q.items() if k != "cand_drift_all_checkpoints"),
        "unit": "Unit of median..max",
        "n_meeting": {
            "definition": "Count meeting the condition named in the definition (exactly 0 / pass)",
            "units": "count",
        },
        "n_of": {
            "definition": "Count out of which n_meeting is taken (runs, or checkpoint rows)",
            "units": "count",
        },
        "median": "Median over runs (or over checkpoint rows for cand_drift_all_checkpoints)",
        "q25": "25th percentile (numpy linear interpolation), same population as median",
        "q75": "75th percentile (numpy linear interpolation), same population as median",
        "min": "Minimum, same population as median",
        "max": "Maximum (the max absolute drift), same population as median",
        "source": "Per-run files read (glob over seeds) or analysis CSV",
    }
    notes = (
        f"Frozen-mode studies: every run's drift_test.json (end of Phase B vs freeze time, dev D_2 grid) and every "
        f"Phase B training-time checkpoint row (v2_checkpoints.csv; locked runs v2_checkpoints_B.csv). Live-network "
        f"drift: last Phase B checkpoint row (the final update in every run); drift_test.json "
        f"live_network_stage2_mean_maxabs_drift (recorded by the {len(live_eq)} pilot-launcher runs) agrees with it "
        f"to max abs diff {live_mad:.2g} (bit-equal in {n_live_eq}). Pilot 2 joint arm: final_table.csv (u1000; equal to "
        f"the last checkpoint row of every run, max abs diff 0) and drift_test.json; on-path = open rule "
        f"|d - drift| < 2q with exact cell masses. Drift grid = development D_2 grid (state step 4). Summaries over "
        f"runs: median, q25, q75 (numpy linear), min, max."
    )
    pack.table(
        "T14",
        df,
        status="generated",
        sources=sources,
        script=f"{MOD}:build_t14",
        notes=notes,
        docs=docs,
        tier="development",
        caption="Drift is measured against the stage-2 mapping at the start of Phase B (the parent / end of Phase "
        "A) on the development D_2 grid. In frozen arms the evaluated candidate uses the snapshot at "
        "stage 2, so its drift is 0 by construction; the live network's own stage-2 output is shown for "
        "information. Quantities: "
        + "; ".join(f"**{k}** = {v[0]}" for k, v in QDEF.items())
        + ". Pilot 2 joint arm: "
        + "; ".join(
            f"**{k}** = {v}" for k, v in JOINT_Q.items() if k != "cand_drift_all_checkpoints"
        )
        + ".",
    )
    return df


# ------------------------------------------------------------------------------------------------
# T17: RNG streams
# ------------------------------------------------------------------------------------------------


def _rngpos(rd: str) -> Dict[int, Dict[str, str]]:
    """Per-update RNG stream positions (``rngpos_*`` of v2_updates.csv), keyed by global update (cached)."""
    rel = f"{rd}/v2_updates.csv"
    if rel not in _CACHE:
        with open(C.abspath(rel), newline="") as fh:
            _CACHE[rel] = {
                int(r["update"]): {s: r[f"rngpos_{s}"] for s in STREAMS} for r in csv.DictReader(fh)
            }
    return _CACHE[rel]


def _first_div(da: str, db: str, updates: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    """First update at which each stream's logged position differs between two runs ('never' if none)."""
    a, b = _rngpos(da), _rngpos(db)
    us = sorted(set(a) & set(b)) if updates is None else sorted(updates)
    if updates is None and sorted(a) != sorted(b):
        raise ValueError(f"different update sequences: {da} vs {db}")
    out: Dict[str, Any] = {"n_updates_compared": len(us)}
    for s in STREAMS:
        out[s] = next((u for u in us if a[u][s] != b[u][s]), "never")
    return out


# paired-arm comparisons of the pilots: (study label, family/pair label, run study, arm_a, arm_b, branch update,
#                                       existing analysis CSV, CSV filter)
PAIRS = [
    (
        "Pilot 1",
        "expected vs sampled",
        "pilot1",
        "sampled",
        "expected",
        0,
        "results/v2_pilots/pilot1/analysis/rng_divergence.csv",
        {},
    ),
    (
        "Pilot 2",
        "B1-A",
        "pilot2",
        "B1_frozen_allnorm",
        "A_joint",
        400,
        "results/v2_pilots/pilot2/analysis/rng_divergence.csv",
        {"pair": "B1-A"},
    ),
    (
        "Pilot 2",
        "B2-A",
        "pilot2",
        "B2_frozen_s1norm",
        "A_joint",
        400,
        "results/v2_pilots/pilot2/analysis/rng_divergence.csv",
        {"pair": "B2-A"},
    ),
    (
        "Pilot 2",
        "B2-B1",
        "pilot2",
        "B2_frozen_s1norm",
        "B1_frozen_allnorm",
        400,
        "results/v2_pilots/pilot2/analysis/rng_divergence.csv",
        {"pair": "B2-B1"},
    ),
    (
        "Pilot 3",
        "mean vs stochastic",
        "pilot3",
        "B2_frozen_s1norm",
        "B2_frozen_s1norm_mean",
        400,
        "results/v2_pilots/pilot3/analysis/rng_divergence.csv",
        {},
    ),
    (
        "Pilot 4 2a",
        "decay vs constant",
        "pilot4_A",
        "constant",
        "decay",
        1200,
        f"{P4A}/rng_divergence.csv",
        {"family": "2a"},
    ),
    (
        "Pilot 4 2b",
        "decay vs constant",
        "pilot4_B",
        "B2_mean_constant",
        "B2_mean_decay",
        1600,
        f"{P4A}/rng_divergence.csv",
        {"family": "2b"},
    ),
]


def _pair_records() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Per (pair, q, seed): first divergence from the existing CSVs, re-computed from v2_updates.csv."""
    rows, n_cmp, n_bad, bad = [], 0, 0, []
    for lab, pair, st, a, b, br, csv_rel, filt in PAIRS:
        ex = S.read_csv(csv_rel, dtype=str)
        for k, v in filt.items():
            ex = ex[ex[k] == v]
        for q in S.QS:
            for seed in S.SEEDS_DEV:
                rec = _first_div(S.run_dir(st, q, seed, a), S.run_dir(st, q, seed, b))
                # branch point = the update before the first logged update (both arms start there)
                first_u = min(_rngpos(S.run_dir(st, q, seed, a)))
                if first_u - 1 != br:
                    raise ValueError(
                        f"{st} q={q} seed={seed}: first logged update {first_u}, expected {br + 1}"
                    )
                er = ex[(ex.q == str(q)) & (ex.seed == str(seed))]
                if len(er) != 1:
                    raise ValueError(f"{csv_rel}: no unique row for q={q} seed={seed} {filt}")
                er = er.iloc[0]
                for s in STREAMS + ("n_updates_compared",):
                    n_cmp += 1
                    if str(rec[s]) != str(er[s]):
                        n_bad += 1
                        bad.append((lab, pair, q, seed, s, rec[s], er[s]))
                rows.append(
                    {
                        "study": lab,
                        "comparison": pair,
                        "q": q,
                        "seed": seed,
                        "branch_update": br,
                        **rec,
                        "csv": csv_rel,
                    }
                )
    return pd.DataFrame(rows), {"n_compared": n_cmp, "n_mismatch": n_bad, "mismatches": bad}


def _parse_rng_cell(cell: str) -> Optional[Tuple[int, int, int, int, int]]:
    """'10/10; first update min 405, median 434, max 475' -> (n_div, n, min, median, max); None for 'never'."""
    m = re.search(
        r"(\d+)/(\d+); (?:first update )?min (\d+(?:\.\d+)?), median (\d+(?:\.\d+)?), max (\d+(?:\.\d+)?)",
        cell,
    )
    if not m:
        return None
    return tuple(float(x) for x in m.groups())  # type: ignore[return-value]


def build_t17(pack: C.Pack) -> pd.DataFrame:
    """T17: RNG streams (what each drives), where learn/opp desynchronize per study, v1.1 global-RNG handling."""
    rows: List[Dict[str, Any]] = []
    ns11 = {q: _json(PROTO11)["records"][q]["protocol"]["rng_namespaces"] for q in ("50", "60")}
    ns10 = {q: _json(PROTO10)["records"][q]["protocol"]["rng_namespaces"] for q in ("50", "60")}
    legacy_ns = ast.literal_eval(
        re.search(
            r'"rng_namespaces": (\{.*?\})', _text("run/run_final_dp_br.py"), flags=re.S
        ).group(1)
    )
    if not (ns11["50"] == ns11["60"] == ns10["50"] == ns10["60"] == legacy_ns):
        raise ValueError("RNG namespaces differ between protocol versions / q / legacy code")
    ns = ns11["50"]
    drives = {
        "init": (
            "torch.Generator seeded with SeedSequence([seed, q, 0]).generate_state(1)[0]; used only for the "
            "weight initialisation (orthogonal init of the actor and critic hidden layers and the critic output; "
            "the actor output head is zero-initialised)",
            "once, at construction",
            "run/run_v2_stagewise.py:Run.__init__; agents/ppo_curriculum.py:BetaActor, Critic",
        ),
        "env_noise": (
            "the two action shocks eps_0, eps_1 ~ U(-q, q) of every active episode at every stage; with "
            "reward_mode=expected they are still drawn (rule A6) and only move d after the last stage",
            "fixed number of draws per update (2 per transition)",
            "run/v2_rollout.py:collect_batch_v2 (rng_env.uniform)",
        ),
        "learner_action": (
            "the learner's Beta action draws at every stage (numpy Generator.beta in float64, clamped, "
            "stored float32); with continuation_action_mode=mean the stage-2 draws are made and "
            "discarded (rule A6)",
            "parameter-dependent: Generator.beta is rejection-based (gamma "
            "variates), so the number of underlying draws depends on the Beta parameters",
            "agents/ppo_curriculum.py:CurriculumPPO.sample_actions; run/v2_rollout.py; "
            "tests/test_v2_infra.py:test_numpy_beta_consumption_depends_on_parameters",
        ),
        "opponent_action": (
            "the opponent's Beta action draws: the lagged copy (refreshed every 20 updates) at stage 1, "
            "and at stage 2 the lagged copy (joint) or the frozen snapshot (frozen); discarded at stage "
            "2 with continuation_action_mode=mean",
            "parameter-dependent (as learner_action)",
            "agents/ppo_curriculum.py:sample_actions; run/v2_rollout.py:collect_batch_v2",
        ),
        "starts_roles": (
            "start states (Phase A: bin-balanced exploring starts on D_2, StartSampler.balanced draws a "
            "bin index and a uniform position; Phase B: root starts, no draw; Phase C: also the shuffle "
            "of root and exploring starts) and the learner role (0/1) of every episode",
            "fixed number of draws per update",
            "run/run_v2_stagewise.py:Run.run_phase; envs/curriculum_env.py:StartSampler.balanced",
        ),
        "minibatch": (
            "the PPO minibatch partition: one permutation of all buffer rows per epoch (also in the masked "
            "frozen-mode update, so consumption is unchanged)",
            "fixed per update (epochs x one permutation of the fixed number of rows of the phase)",
            "agents/ppo_curriculum.py:CurriculumPPO.update; agents/ppo_curriculum_v2.py:CurriculumPPOv2.update",
        ),
    }
    short = {
        "init": "init",
        "env_noise": "env",
        "learner_action": "learn",
        "opponent_action": "opp",
        "starts_roles": "start",
        "minibatch": "minibatch",
    }
    for name, nsv in ns.items():
        d, cons, src = drives[name]
        rows.append(
            {
                "block": "1 stream",
                "stream": name if short[name] == name else f"{name} ({short[name]})",
                "namespace": str(nsv),
                "what_it_drives": d,
                "consumption_or_handling": cons,
                "source": src + "; " + PROTO11 + " records.<q>.protocol.rng_namespaces",
            }
        )
    rows.append(
        {
            "block": "1 stream",
            "stream": "direct-rollout streams (legacy evaluation)",
            "namespace": "SeedSequence([direct_rollout_seed_base 9005000, q, seed, rep, player, t])",
            "what_it_drives": "shocks of the legacy sampled self-play payoff evaluation (direct_rollout in "
            "final_evaluation); run only in mode full (the C7 regression run), not by the v2 "
            "pilots or the locked pipeline",
            "consumption_or_handling": "separate streams; consume nothing from the training streams",
            "source": "run/run_final_dp_br.py:direct_rollout; run/run_v2_stagewise.py:execute (mode full only)",
        }
    )
    # ---- block 2: desynchronization per study
    pr, chk = _pair_records()
    _CACHE["t17_pairs"] = pr
    for (lab, pair), g0 in pr.groupby(["study", "comparison"], sort=False):
        for q in S.QS:
            g = g0[g0.q == q]
            base = {
                "block": "2 desynchronization",
                "study": lab,
                "comparison": pair,
                "q": q,
                "branch_update": int(g.branch_update.iloc[0]),
                "n_pairs": len(g),
                "source": f"{g.csv.iloc[0]} (re-computed from v2_updates.csv rngpos_* of both arms: equal)",
            }
            for s in ("learn", "opp"):
                fu = [int(x) for x in g[s] if x != "never"]
                rec = dict(base, stream=s, n_pairs_diverged=len(fu))
                if fu:
                    st_ = C.median_iqr(fu)
                    ab = [u - rec["branch_update"] for u in fu]
                    sa = C.median_iqr(ab)
                    rec.update(
                        first_update_min=st_["min"],
                        first_update_median=st_["median"],
                        first_update_max=st_["max"],
                        after_branch_min=sa["min"],
                        after_branch_median=sa["median"],
                        after_branch_max=sa["max"],
                    )
                rows.append(rec)
            other = sum(
                1
                for _, r in g.iterrows()
                if any(r[s] != "never" for s in ("env", "start", "minibatch"))
            )
            rows.append(dict(base, stream="env, start, minibatch", n_pairs_diverged=other))
    # same-configuration comparisons (no desynchronization expected)
    same = []
    for q in S.QS:
        recs = [
            _first_div(
                S.run_dir("pilot3", q, s, "B2_frozen_s1norm"),
                S.run_dir("pilot2", q, s, "B2_frozen_s1norm"),
            )
            for s in S.SEEDS_DEV
        ]
        nd = sum(1 for r in recs if any(r[k] != "never" for k in STREAMS))
        same.append(
            {
                "study": "Pilot 3 vs Pilot 2",
                "comparison": "Pilot 3 stochastic arm vs Pilot 2 B2 (same flags, same parent)",
                "q": q,
                "branch_update": 400,
                "n_pairs": len(recs),
                "n_pairs_diverged": nd,
                "source": "results/v2_pilots/pilot{2,3}/q*/seed*/B2_frozen_s1norm/v2_updates.csv (computed here)",
            }
        )
    _CACHE["t17_same_p3p2"] = (
        sum(r["n_pairs_diverged"] for r in same),
        sum(r["n_pairs"] for r in same),
    )
    ali = _json("results/v2_pilots/_smoke/rng_alignment.json")
    for pair_name, res in ali.items():
        nd = sum(1 for ck in res.values() for ok in ck.values() if not ok)
        same.append(
            {
                "study": "Phase 2 smoke (flow check)",
                "comparison": f"{pair_name}: stream positions after one batch and after 20 updates",
                "q": 50,
                "branch_update": 0 if "from init" in pair_name else 40,
                "n_pairs": 1,
                "n_pairs_diverged": int(nd > 0),
                "source": "results/v2_pilots/_smoke/rng_alignment.json (tools/v2/rng_alignment.py)",
            }
        )
    rp = S.read_csv(f"{P4A}/repro_2a_constant_vs_ext.csv")
    for q in S.QS:
        g = rp[rp.q == q]
        same.append(
            {
                "study": "Pilot 4 2a vs Phase A ext.",
                "comparison": "constant arm vs the extension over "
                "u1201-u1600 (same flags, same state)",
                "q": q,
                "branch_update": 1200,
                "n_pairs": len(g),
                "n_pairs_diverged": int(
                    (~g["rng_positions_every_update_identical"].astype(bool)).sum()
                ),
                "source": f"{P4A}/repro_2a_constant_vs_ext.csv (rng_positions_every_update_identical)",
            }
        )
    dr = _json(f"{LK}/consolidation/dirty_rerun_compare.json")
    same.append(
        {
            "study": "dirty-flag re-run",
            "comparison": "clean re-run vs original (q=60 seed 10507 B2_mean_constant)",
            "q": 60,
            "branch_update": 1600,
            "n_pairs": 1,
            "n_pairs_diverged": 0 if dr["rng_positions_identical"] else 1,
            "source": f"{LK}/consolidation/dirty_rerun_compare.json (rng_positions_identical)",
        }
    )
    c2 = S.read_csv(f"{LK}/rehearsal_analysis/check2_phaseB_vs_launcher.csv")
    for q in S.QS:
        g = c2[c2.q == q]
        same.append(
            {
                "study": "v1.0 Check 2",
                "comparison": "pilot launcher Phase B vs the rehearsal's Phase B (seed 10503)",
                "q": q,
                "branch_update": 1600,
                "n_pairs": len(g),
                "n_pairs_diverged": int((~g["rng_positions_B_identical"].astype(bool)).sum()),
                "source": f"{LK}/rehearsal_analysis/check2_phaseB_vs_launcher.csv (rng_positions_B_identical)",
            }
        )
    r1 = S.read_csv(f"{LK}/rehearsal_v1_1_R1_details.csv")
    for q in S.QS:
        g = r1[r1.q == q]
        same.append(
            {
                "study": "v1.1 re-rehearsal vs v1.0 rehearsal",
                "comparison": "R1: same seeds, v1.1 vs v1.0 "
                "entry point (per-update log incl. rngpos_*)",
                "q": q,
                "branch_update": 0,
                "n_pairs": len(g),
                "n_pairs_diverged": int((~g["v2_updates.csv"].astype(bool)).sum()),
                "source": f"{LK}/rehearsal_v1_1_R1_details.csv (v2_updates.csv)",
            }
        )
    for r in same:
        tag = (
            "all five (20-update horizon)" if r["study"].startswith("Phase 2 smoke") else "all five"
        )
        rows.append(dict(block="2 desynchronization", stream=tag, **r))
    for st, txt in (
        ("phaseA_ext", "single arm (no paired comparison)"),
        ("rehearsal", "single configuration per (q, seed); no paired arms"),
        ("rehearsal_v1_1", "single configuration per (q, seed); see the R1 rows"),
        ("confirmation", "single configuration per (q, seed), fresh seeds; no paired arms"),
    ):
        rows.append(
            {
                "block": "2 desynchronization",
                "stream": "n/a",
                "study": LABEL[st],
                "comparison": txt,
                "source": "study design (" + S.STUDIES[st]["report"] + ")",
            }
        )
    # ---- block 3: process-global RNGs
    v11 = [rd for st in ("rehearsal_v1_1", "confirmation") for _, _, _, rd in _runs(st)]
    pts = ("end_of_A", "after_G-A", "end_of_B", "end_of_run")
    c1 = S.read_csv(f"{LK}/rehearsal_analysis/check1_phaseA_vs_stitched.csv")
    trace = _json(f"{LK}/v1_1/global_rng_trace.json")
    gstat = [_json(f"{rd}/gates.json")["global_rng"]["status"] for rd in v11]
    names = {
        "torch_global": ("torch global RNG", "torch_global_rng_identical"),
        "numpy_global": ("numpy legacy global RNG", "numpy_global_rng_identical"),
        "python_random": ("Python random", "python_random_identical"),
    }
    for key, (nm, c1col) in names.items():
        seeded = drawn = between = asserted = b_ok = 0
        for rd in v11:
            g = _manifest(rd)["global_rng"]
            seed = int(_manifest(rd)["seed"])
            seeded += int(g["seeding"]["seeds"][key] == seed)
            drawn += int(
                g["seeding"]["digests"][key] != g["after_run_construction"]["digests"][key]
            )
            ref = g["reference_before_first_A_update"]["digests"][key]
            between += int(g["after_run_construction"]["digests"][key] != ref)
            asserted += int(all(g[p]["digests"][key] == ref for p in pts))
            b_ok += int(g["before_first_B_update"]["digests"][key] == ref)
        steps = [s["step"] for s in trace["steps"] if s[f"{key}_changed"]]
        drives_txt = (
            "drawn only by PyTorch's default nn.Linear initialisation while Run is constructed (the weights "
            "are then re-initialised from the explicit init generator); never by training"
            if steps
            else "drawn by nothing in construction or training"
        )
        n_c1 = int(c1[c1col].astype(bool).sum())
        rows.append(
            {
                "block": "3 process-global RNG",
                "stream": nm,
                "namespace": "process-global (not a SeedSequence stream)",
                "what_it_drives": drives_txt + (f" (trace: {'; '.join(steps)})" if steps else ""),
                "consumption_or_handling": (
                    f"v1.0: not seeded (entropy at process start), saved in every full state and restored on "
                    f"branching, never consumed; Check 1 identical in {n_c1}/{len(c1)} (literal failure). v1.1: "
                    f"seeded with the run seed at the start of run_pipeline; reference = state immediately before "
                    f"the first Phase A update (Run.phase_start_hook); equality asserted at {', '.join(pts)} "
                    f"(also logged before the first Phase B update); a violation is recorded in gates.json and the "
                    f"run exits with code 5; SHA-256 digests in manifest.json"
                ),
                "evidence": (
                    f"v1.1 runs (20 re-rehearsal + 40 confirmation): seeded with the run seed "
                    f"{seeded}/{len(v11)}; changed during Run construction {drawn}/{len(v11)}; changed "
                    f"between construction and the reference {between}/{len(v11)}; equal to the reference "
                    f"at all {len(pts)} assertion points {asserted}/{len(v11)}; equal before the first "
                    f"Phase B update {b_ok}/{len(v11)}; gates.json global_rng.status ok "
                    f"{sum(1 for x in gstat if x == 'ok')}/{len(v11)}"
                ),
                "source": f"{PROTO11} global_rng_hardening; {LOCKED_RUNNER}:seed_globals, run_pipeline; "
                f"{LK}/v1_1/global_rng_trace.json; v1.1 manifest.json global_rng and gates.json; "
                f"{LK}/rehearsal_analysis/check1_phaseA_vs_stitched.csv",
            }
        )
    df = pd.DataFrame(rows)
    cols = [
        "block",
        "stream",
        "namespace",
        "what_it_drives",
        "consumption_or_handling",
        "study",
        "comparison",
        "q",
        "branch_update",
        "n_pairs",
        "n_pairs_diverged",
        "first_update_min",
        "first_update_median",
        "first_update_max",
        "after_branch_min",
        "after_branch_median",
        "after_branch_max",
        "evidence",
        "source",
    ]
    df = df.reindex(columns=cols)
    for c in ("q", "branch_update", "n_pairs", "n_pairs_diverged"):
        df[c] = _intcol(df[c])
    # ---- cross-checks
    pack.crosschecks.append(
        {
            "item": "T17",
            "report": "results/v2_pilots/*/analysis/rng_divergence.csv",
            "label": "first divergence per stream re-computed from v2_updates.csv vs the existing CSVs",
            "n_tables": 0,
            "n_compared": chk["n_compared"],
            "n_mismatch": chk["n_mismatch"],
            "n_unmatched_rows": 0,
        }
    )
    for lab, pair, q, seed, s, mine, theirs in chk["mismatches"]:
        pack.mismatch(
            "T17",
            f"first divergence {lab} {pair} q={q} seed={seed} {s}",
            mine,
            "existing rng_divergence.csv",
            theirs,
            "data vs data",
        )
    p1 = pr[pr.study == "Pilot 1"].copy()
    for s in ("learn", "opp"):
        p1[s] = pd.to_numeric(p1[s], errors="coerce")
    pack.crosscheck(
        "T17",
        p1,
        R_PI1,
        header_has=["q", "seed", "env", "learn", "opp", "start", "minibatch"],
        key_map={"q": "q", "seed": "seed"},
        value_map={"learn": "learn", "opp": "opp", "n_updates_compared": "n_updates_compared"},
        heading_has="RNG alignment per pair",
        label="Pilot 1 first divergence per pair (section 3.5)",
    )
    t = [t for t in C.parse_md_tables(R_PI1) if "RNG alignment per pair" in t["heading"]][0]
    hdr = t["header"]
    nev = [
        (f"Pilot 1 q={r[0]} seed={r[1]} {s}", "never", r[hdr.index(s)], r[hdr.index(s)] == "never")
        for r in t["rows"]
        for s in ("env", "start", "minibatch")
    ]
    mine_never = [
        str(p1[(p1.q == int(r[0])) & (p1.seed == int(r[1]))].iloc[0][s])
        for r in t["rows"]
        for s in ("env", "start", "minibatch")
    ]
    nev = [(a, m, r, ok and m == "never") for (a, _, r, ok), m in zip(nev, mine_never)]
    _manual(
        pack,
        "T17",
        f"{R_PI1} line {t['line']}",
        "Pilot 1 'never' cells (env, start, minibatch)",
        nev,
    )
    summ = df[(df.block == "2 desynchronization") & df.stream.isin(["learn", "opp"])]

    def srow(study: str, comp: str, q: int, s: str) -> pd.Series:
        return summ[
            (summ.study == study) & (summ.comparison == comp) & (summ.q == q) & (summ.stream == s)
        ].iloc[0]

    # Pilot 1 prose: learn 11-69 (median 33) / 13-63 (32); opp 22-73 (43) / 22-69 (36)
    pairs = []
    for s in ("learn", "opp"):
        ln = _line_text(R_PI1, f"- **{s}:** diverges in every pair")
        m2 = re.findall(r"(\d+)–(\d+) \(median (\d+)\) at q=(\d+)", ln)
        for lo, hi, me, q in m2:
            r = srow("Pilot 1", "expected vs sampled", int(q), s)
            pairs += [
                (f"Pilot 1 {s} q={q} min", r.first_update_min, lo, None),
                (f"Pilot 1 {s} q={q} max", r.first_update_max, hi, None),
                (f"Pilot 1 {s} q={q} median", r.first_update_median, me, None),
            ]
    _manual(
        pack,
        "T17",
        _cite(R_PI1, "- **learn:** diverges in every pair"),
        "Pilot 1 divergence ranges (prose)",
        pairs,
    )
    # tables with cells 'n/n; [first update] min a, median b, max c' (Pilot 2, Pilot 3, Pilot 4)
    for rep, heading, keyf in (
        (R_PI2, "2.8 RNG stream divergence", None),
        (R_PI3, "RNG stream divergence", None),
        (R_PI4, "", "family"),
    ):
        tabs = [
            t
            for t in C.parse_md_tables(rep)
            if "learn" in t["header"]
            and "opp" in t["header"]
            and (heading.lower() in t["heading"].lower())
            and (keyf is None or keyf in t["header"])
        ]
        pairs = []
        for t in tabs:
            hdr = t["header"]
            for r in t["rows"]:
                rec = dict(zip(hdr, r))
                q = int(rec["q"])
                if rep == R_PI2:
                    study, comp = "Pilot 2", rec["pair"]
                elif rep == R_PI3:
                    study, comp = "Pilot 3", "mean vs stochastic"
                else:
                    study, comp = f"Pilot 4 {rec['family']}", "decay vs constant"
                for s in ("learn", "opp"):
                    pc = _parse_rng_cell(rec[s])
                    if pc is None:
                        continue
                    sr = srow(study, comp, q, s)
                    pairs += [
                        (
                            f"{study} {comp} q={q} {s} n diverged",
                            sr.n_pairs_diverged,
                            str(int(pc[0])),
                            None,
                        ),
                        (
                            f"{study} {comp} q={q} {s} min",
                            sr.first_update_min,
                            str(int(pc[2])),
                            None,
                        ),
                        (
                            f"{study} {comp} q={q} {s} median",
                            sr.first_update_median,
                            rec[s].split("median ")[1].split(",")[0],
                            None,
                        ),
                        (
                            f"{study} {comp} q={q} {s} max",
                            sr.first_update_max,
                            str(int(pc[4])),
                            None,
                        ),
                    ]
                for s in ("env", "start", "minibatch"):
                    ok = rec[s].startswith("never")
                    orow = df[
                        (df.block == "2 desynchronization")
                        & (df.study == study)
                        & (df.comparison == comp)
                        & (df.q == q)
                        & (df.stream == "env, start, minibatch")
                    ].iloc[0]
                    pairs.append(
                        (
                            f"{study} {comp} q={q} {s} never",
                            int(orow.n_pairs_diverged),
                            rec[s],
                            ok and int(orow.n_pairs_diverged) == 0,
                        )
                    )
        _manual(
            pack,
            "T17",
            f"{rep} (table at line {tabs[0]['line']})",
            "first-divergence summary cells",
            pairs,
        )
    # prose offsets after the branch point
    off = df[(df.block == "2 desynchronization") & df.stream.isin(["learn", "opp"])]
    pr_pairs = []
    for rep, needle, study, comps in (
        (R_PI2, "within 3–164 updates of the branch point", "Pilot 2", ("B1-A", "B2-A", "B2-B1")),
        (R_PI3, "desynchronize between arms 4–134 updates", "Pilot 3", ("mean vs stochastic",)),
        (R_PI4, "within 4–40 updates of u1200", "Pilot 4 2a", ("decay vs constant",)),
    ):
        ln = _line_text(rep, needle)
        lo, hi = re.search(r"(\d+)–(\d+) updates", ln).groups()
        g = off[(off.study == study) & off.comparison.isin(comps)]
        pr_pairs += [
            (
                f"{study} learn/opp first divergence, updates after branch: min",
                g.after_branch_min.min(),
                lo,
                None,
            ),
            (
                f"{study} learn/opp first divergence, updates after branch: max",
                g.after_branch_max.max(),
                hi,
                None,
            ),
        ]
    _manual(
        pack,
        "T17",
        f"{R_PI2}, {R_PI3}, {R_PI4} (anomaly prose)",
        "desynchronization offsets after branching",
        pr_pairs,
    )
    sl = _line_text(R_SUM, "**RNG pairing.**")
    lim = re.search(r"within tens to about (\d+) updates", sl).group(1)
    allmax = float(off.after_branch_max.max())
    who = off.loc[off.after_branch_max.idxmax()]
    _manual(
        pack,
        "T17",
        _cite(R_SUM, "**RNG pairing.**"),
        "learn/opp desynchronization horizon over all pilots (prose)",
        [
            (
                f"max over all pilot pairs of the first learn/opp divergence, updates after branching (largest: {who.study} "
                f"q={int(who.q)} {who.stream})",
                allmax,
                f"about {lim} ('within tens to about {lim} updates')",
                allmax <= 1.25 * float(lim),
                "statement dates from commit 8b35066 (before Pilot 4) and is unchanged in the current summary; the Pilot 4 "
                "2b pairs first diverge up to this many updates after the u1600 branch point (pilots 1-3 and 2a: at most "
                f"{int(off[~off.study.eq('Pilot 4 2b')].after_branch_max.max())})",
            )
        ],
    )
    sources = [
        C.src(PROTO10),
        C.src(PROTO11),
        C.src("run/run_final_dp_br.py"),
        C.src(RUNNER),
        C.src("run/v2_rollout.py"),
        C.src("agents/ppo_curriculum.py"),
        C.src("agents/ppo_curriculum_v2.py"),
        C.src("envs/curriculum_env.py"),
        C.src(LOCKED_RUNNER),
    ]
    sources += [C.src(r) for r in sorted({p[6] for p in PAIRS})]
    sources.append(
        C.srcs(
            sorted(
                {
                    f"{S.run_dir(p[2], q, s, a)}/v2_updates.csv"
                    for p in PAIRS
                    for q in S.QS
                    for s in S.SEEDS_DEV
                    for a in (p[3], p[4])
                }
            ),
            label="v2_updates.csv of every paired run",
        )
    )
    sources += [
        C.src("results/v2_pilots/_smoke/rng_alignment.json"),
        C.src(f"{P4A}/repro_2a_constant_vs_ext.csv"),
        C.src(f"{LK}/consolidation/dirty_rerun_compare.json"),
        C.src(f"{LK}/rehearsal_analysis/check2_phaseB_vs_launcher.csv"),
        C.src(f"{LK}/rehearsal_v1_1_R1_details.csv"),
        C.src(f"{LK}/rehearsal_analysis/check1_phaseA_vs_stitched.csv"),
        C.src(f"{LK}/v1_1/global_rng_trace.json"),
        C.srcs(
            [f"{rd}/{f}" for rd in v11 for f in ("manifest.json", "gates.json")],
            label="manifest.json and gates.json of the 60 v1.1 runs",
        ),
        C.src(R_P0),
        C.src(R_P2),
        C.src(R_PI1),
        C.src(R_PI2),
        C.src(R_PI3),
        C.src(R_PI4),
        C.src(R_LOCK),
        C.src(R_V11),
    ]
    docs = {
        "block": "1 stream = the SeedSequence streams and what they drive; 2 desynchronization = where the learn / opp "
        "streams of paired runs first differ; 3 process-global RNG = torch global, numpy legacy global, Python "
        "random",
        "stream": "Stream name (short name in v2_updates.csv rngpos_* in brackets), or the stream(s) of the row",
        "namespace": "Namespace of numpy SeedSequence([seed, q, namespace]) (protocol rng_namespaces)",
        "what_it_drives": "What the stream is consumed by (source: code text)",
        "consumption_or_handling": "Block 1: whether the number of draws per update is fixed or depends on the policy "
        "parameters; block 3: v1.0 and v1.1 handling",
        "study": "Study (block 2)",
        "comparison": "Compared runs (block 2): arm pair with different settings (full horizon), a "
        "same-configuration comparison, or a Phase 2 smoke arm pair (20-update horizon)",
        "branch_update": {
            "definition": "Global update at which the compared runs start from the same state "
            "(0 = same initialisation)",
            "units": "updates",
        },
        "n_pairs": {"definition": "Compared run pairs (one per seed)", "units": "count"},
        "n_pairs_diverged": {
            "definition": "Pairs in which the stream position differs at some update (for 'env, "
            "start, minibatch': in any of the three; 'all five': in any stream)",
            "units": "count",
        },
        "first_update_min": {
            "definition": "Smallest global update at which the stream first differs (over diverged "
            "pairs)",
            "units": "updates",
        },
        "first_update_median": {
            "definition": "Median of the first differing global update",
            "units": "updates",
        },
        "first_update_max": {
            "definition": "Largest first differing global update",
            "units": "updates",
        },
        "after_branch_min": {
            "definition": "first_update_min minus branch_update",
            "units": "updates",
        },
        "after_branch_median": {
            "definition": "first_update_median minus branch_update",
            "units": "updates",
        },
        "after_branch_max": {
            "definition": "first_update_max minus branch_update",
            "units": "updates",
        },
        "evidence": "Block 3: counts over the 60 v1.1 runs from manifest.json global_rng digests and gates.json",
        "source": "Files and code locations",
    }
    notes = (
        "Block 1 from the protocol records (namespaces identical in v1.0, v1.1, both q and the legacy runner "
        "defaults) and the code (source: code text). Block 2: first differing update per stream from the existing "
        f"rng_divergence.csv files; every entry re-computed from the two runs' v2_updates.csv rngpos_* columns "
        f"({chk['n_compared']} cells, {chk['n_mismatch']} differences); summaries over the seeds (median: numpy). "
        "Also listed: same-configuration comparisons (positions identical at every compared update) and the Phase 2 "
        "smoke alignment table (differing arms, aligned within 20 updates). "
        "The cause of the learn/opp desynchronization is stated as in the reports: numpy's rejection-based Beta "
        "sampler consumes a parameter-dependent number of draws once the arms' policies differ. Block 3 from the "
        "60 v1.1 manifests (global_rng digests), gates.json, the trace JSON and Check 1."
    )
    pack.table(
        "T17",
        df,
        status="generated",
        sources=sources,
        script=f"{MOD}:build_t17",
        notes=notes,
        docs=docs,
        tier="n/a",
    )
    return df


# ------------------------------------------------------------------------------------------------
# T18: reproducibility ledger
# ------------------------------------------------------------------------------------------------


def _bool_count(df: pd.DataFrame, cols: Sequence[str]) -> int:
    """Rows in which every listed boolean column is True."""
    return int(
        df[list(cols)].astype(str).apply(lambda c: c.str.lower() == "true").all(axis=1).sum()
    )


def build_t18(pack: C.Pack) -> pd.DataFrame:
    """T18: every bit-identity check run so far (scope, unit, n, result, source)."""
    rows: List[Dict[str, Any]] = []
    srcs: List[Any] = []
    checks: List[Tuple[str, Any, str, Optional[bool]]] = []

    def add(**kw: Any) -> None:
        kw.setdefault("note", "")
        rows.append(kw)

    # R01 Phase 1 B7
    b7 = _parse_c7(f"{REG1}/compare.txt")
    rb7 = _recheck_run_pair(
        f"{REG1}/before/FINAL_A400_B25_C25/tel_q50_s10501",
        f"{REG1}/after/FINAL_A400_B25_C25/tel_q50_s10501",
    )
    srcs += [C.src(f"{REG1}/compare.txt"), C.src(f"{REG1}/manifest.json")]
    add(
        scope="q=50, seed 10501; existing runner (run/run_final_dp_br_round3_dense.py), smoke caps A/B/C 40/40/40, warm-up 10, stability every 5, timeout 10, direct rollout 2000 x 1 (source: report text)",
        check="Phase 1 B7: existing runner before vs after the Phase 1 code",
        study="Phase 1 verifier",
        check_class="code-path regression (existing runner)",
        in_spec_list=False,
        commit="before: 1ad3805; after: with the Phase 1 code present (code commit b2bfec0; HEAD of the after run not "
        "stated)",
        compared=b7["compared_fields"] + "; stdout logs (report)",
        unit="run pair (q=50, seed 10501, smoke caps 40/40/40)",
        n_units=1,
        n_identical=int(b7["verdict"] == "IDENTICAL"),
        result="pass" if b7["verdict"] == "IDENTICAL" else "fail",
        source_kind="data file",
        source=f"{REG1}/compare.txt",
        report_section=f"{_cite(R_P1, '## 5. B7')}",
        note=f"pack re-check of the four NPZ files and the two JSON files: {rb7['arrays']} arrays, "
        f"{rb7['arrays_differ']} differ; {rb7['json_leaves_differ']} JSON leaves differ",
    )
    # R02 C7 at 8 commits
    t15 = _CACHE["t15"]
    c7 = t15[t15.compare_file != "none"]
    add(
        scope="q=50, seed 10501; v2 runner in mode full on the Phase 1 regression record, smoke caps A/B/C 40/40/40; one regression run per commit",
        check="C7 regression: v2 runner in mode full vs the existing runner",
        study="Phase 2 and every later code commit",
        check_class="code-path regression (v2 runner vs existing runner)",
        in_spec_list=False,
        commit=", ".join(c7.commit),
        compared="train_history.json, final_eval.json, arrays.npz (90), "
        "checkpoint_weights.npz (12), phase_A/B_exit_arrays.npz (41 + 41), checkpoint.pt tensors (see T15)",
        unit="commit (one regression run each, q=50, seed 10501, smoke caps 40/40/40)",
        n_units=len(c7),
        n_identical=int((c7.verdict == "IDENTICAL").sum()),
        result="pass" if (c7.verdict == "IDENTICAL").all() else "fail",
        source_kind="data file",
        source=f"{REG2}/compare_vs_existing_runner.txt and {REG2}/v2_full_<commit>.compare.txt (pack item T15)",
        report_section=f"{_cite(R_P2, '## 3. C7 regression')} and the C7 line of each later report",
        note=f"regression manifests with dirty = false: {int((c7.dirty == False).sum())}/{len(c7)}; pack re-check: "  # noqa: E712
        f"{int(c7.pack_recheck_arrays.sum())} arrays and the two JSON files, "
        f"{int(c7.pack_recheck_arrays_differ.sum()) + int(c7.pack_recheck_json_leaves_differ.sum())} differences",
    )
    # Phase 2 unit tests (report text at 8f2840e)
    t2 = "tests/test_v2_infra.py"
    n_b = int(_code_constants(t2, ["N_B"])["N_B"])
    srcs += [C.src(t2), C.src("tests/test_v2_verifier.py"), C.src(R_P2), C.src(R_P1)]
    p2_res = _line_text(R_P2, "Result at `8f2840e`")
    later = (
        "89cd600 (0 failed), c92ee74, 4bd2214, 431474d and 95c000e (only failure in those full-suite runs: the "
        "known test_registry_canonicalization; T16)"
    )
    later_a = (
        "c92ee74, 4bd2214, 431474d and 95c000e (only failure in those full-suite runs: the known "
        "test_registry_canonicalization; T16)"
    )
    for name, n, comp, row_needle, spec, cls in [
        (
            "test_continuous_equals_branched",
            1,
            "A then B in one process vs A, save full state, restore, B (joint, 20 B "
            "updates): B history, B verifier calls (except time_sec), final actor tensors, every RNG stream",
            "**C1** full-state restore",
            True,
            "state restore / branching (unit test)",
        ),
        (
            "test_branch_reproducibility",
            3,
            "two branches per arm (A_joint, B1, B2) from one parent, 20 B updates: "
            "train_history (history, verifier calls), v2_updates.csv, v2_checkpoints.csv, final weights",
            "branch reproducibility",
            True,
            "state restore / branching (unit test)",
        ),
        (
            "test_snapshot_immutable",
            2,
            "B1, B2 after 20 updates: snapshot tensors vs parent actor, stage-2 mean/alpha/beta "
            "at freeze vs end (dev D_2), candidate drift 0 at every checkpoint",
            "snapshot immutability",
            False,
            "artifact identity (frozen snapshot, unit test)",
        ),
        (
            "test_joint_update_bit_identical_to_original",
            1,
            "v2 joint update vs CurriculumPPO.update on the same batch "
            "and minibatch-RNG state: diagnostics, actor and critic tensors",
            "joint unchanged",
            False,
            "code-path regression (unit test)",
        ),
        (
            "test_gradient_isolation + test_b2_update_ignores_stage2_advantages_end_to_end",
            3,
            "B1, B2: parameter "
            "gradients bit-identical when stage-2 advantages are perturbed; B2: whole update bit-identical",
            "gradient isolation",
            False,
            "update isolation (unit test)",
        ),
    ]:
        add(
            scope="unit test on a throwaway parent (q=50, seed 10501, 6 Phase A updates; fixture "
            "docstring); "
            + (
                f"{n_b} Phase B updates per branch (N_B)"
                if ("branch" in name or "snapshot" in name)
                else "one Phase B batch"
            ),
            check=f"Phase 2 unit test {name}",
            study="Phase 2 infrastructure",
            check_class=cls,
            in_spec_list=spec,
            commit="8f2840e",
            compared=comp,
            unit="test case",
            n_units=n,
            n_identical=n,
            result="pass",
            source_kind="test code + report text",
            source=f"{t2}::{name.split(' + ')[0]}; {_cite(R_P2, row_needle)}",
            report_section=_cite(R_P2, "## 2. Test results"),
            note=f"suite result at 8f2840e: {p2_res.strip()}; also passing at {later}",
        )
    add(
        scope="unit test on the same throwaway parent; two continuations of 6 Phase A updates (u7-u12), full states at u9 and u12",
        check="Phase A continuation unit test test_phase_a_continue_reproducible_and_branchable",
        study="Phase A extension code (cd760fd)",
        check_class="state restore / branching (unit test)",
        in_spec_list=False,
        commit="cd760fd (test added)",
        compared="two phase_A_continue runs from one parent: training history; mid-run "
        "full states are valid Phase B parents",
        unit="test case",
        n_units=1,
        n_identical=1,
        result="pass",
        source_kind="test code + report text (T16)",
        source=f"{t2}::test_phase_a_continue_reproducible_and_branchable",
        report_section=_cite(R_PI4, "| Tests on `c92ee74` |"),
        note=f"no report states a result at cd760fd (T16: UNKNOWN); passing at {later_a}",
    )
    ali = _json("results/v2_pilots/_smoke/rng_alignment.json")
    n_al = sum(1 for v in ali.values() for ck in v.values() for ok in ck.values() if ok)
    n_alt = sum(1 for v in ali.values() for ck in v.values() for _ in ck.values())
    srcs.append(C.src("results/v2_pilots/_smoke/rng_alignment.json"))
    add(
        scope="smoke parent q=50, seed 10501 (40 Phase A updates); 5 arm pairs; after one batch and after 20 updates",
        check="Phase 2 RNG alignment of paired arms (tools/v2/rng_alignment.py and tests)",
        study="Phase 2 infrastructure",
        check_class="RNG state identity across arms",
        in_spec_list=False,
        commit="8f2840e",
        compared="positions of the 5 streams (env, learn, opp, start, minibatch) after one batch and after 20 updates, "
        "5 arm pairs (smoke parent q=50 seed 10501); unit tests test_rng_alignment_after_one_batch, "
        "test_rng_alignment_after_updates_fixed_count_streams, test_expected_mode_preserves_rng_after_one_batch",
        unit="(pair, checkpoint, stream)",
        n_units=n_alt,
        n_identical=n_al,
        result="pass" if n_al == n_alt else "fail",
        source_kind="data file + report text",
        source="results/v2_pilots/_smoke/rng_alignment.json",
        report_section=_cite(R_P2, "### 2.1 RNG alignment table"),
        note="over full pilot horizons the learn/opp streams of differing arms desynchronize (T17); same-configuration "
        "comparisons stay aligned",
    )
    add(
        scope="unit test: one evaluate call (tests/test_v2_verifier.py)",
        check="Verifier consumes no RNG: unit test test_evaluate_consumes_no_rng",
        study="Phase 1 verifier",
        check_class="RNG non-consumption (unit test)",
        in_spec_list=False,
        commit="b2bfec0",
        compared="numpy Generator states, numpy global state and torch RNG state before vs after utils.v2_metrics.evaluate",
        unit="test case",
        n_units=1,
        n_identical=1,
        result="pass",
        source_kind="test code + report text",
        source="tests/test_v2_verifier.py::test_evaluate_consumes_no_rng",
        report_section=_cite(R_P1, "| `test_evaluate_consumes_no_rng` |"),
        note=f"suite result: {_line_text(R_P1, 'Result: **21 passed').strip()}",
    )
    # R: Pilot 2 parents
    pc = _json("results/v2_pilots/pilot2_parents_check.json")
    keys = sorted({k for p_ in pc["parents"] for k in p_["restore_checks"]})
    n_ok = sum(1 for p_ in pc["parents"] if p_["ok"] and all(p_["restore_checks"].values()))
    srcs.append(C.src("results/v2_pilots/pilot2_parents_check.json"))
    add(
        scope=f"{_qs(pd.DataFrame(pc['parents']))}; Pilot 1 expected end-of-Phase-A full states (u400)",
        check="Pilot 2 parent verification (full-state restore of the Pilot 1 expected parents)",
        study="Pilot 2",
        check_class="state restore / branching",
        in_spec_list=False,
        commit="1791687 (tools/v2/verify_parents.py)",
        compared="restored vs saved: " + ", ".join(keys),
        unit="parent state (q, seed)",
        n_units=int(pc["n"]),
        n_identical=n_ok,
        result="pass" if n_ok == pc["n"] and pc["all_ok"] else "fail",
        source_kind="data file",
        source="results/v2_pilots/pilot2_parents_check.json",
        report_section=_cite(R_PI2, "### 1.2 Parent verification"),
    )
    # R: Pilot 3 vs Pilot 2 B2
    rp3 = S.read_csv("results/v2_pilots/pilot3/analysis/reproducibility_vs_pilot2_B2.csv")
    rp3_qs = pd.DataFrame(
        {
            "q": rp3["run"].str.extract(r"q(\d+)/")[0].astype(int),
            "seed": rp3["run"].str.extract(r"seed(\d+)/")[0].astype(int),
        }
    )
    c3 = [c for c in rp3.columns if c.endswith("_identical")]
    n3 = _bool_count(rp3, c3)
    pr = _CACHE.get("t17_same_p3p2", None)
    srcs.append(C.src("results/v2_pilots/pilot3/analysis/reproducibility_vs_pilot2_B2.csv"))
    add(
        scope=f"{_qs(rp3_qs)}; Phase B u401-u1000",
        check="Pilot 3 stochastic arm vs Pilot 2 B2 (same flags, same parent, different code commit)",
        study="Pilot 3",
        check_class="training reproduction (re-run)",
        in_spec_list=True,
        commit="cd760fd vs 1791687",
        compared=", ".join(c3)
        + f" (n_updates {', '.join(str(x) for x in sorted(rp3.n_updates.unique()))})",
        unit="run pair (q, seed)",
        n_units=len(rp3),
        n_identical=n3,
        result="pass" if n3 == len(rp3) else "fail",
        source_kind="data file",
        source="results/v2_pilots/pilot3/analysis/reproducibility_vs_pilot2_B2.csv",
        report_section=_cite(R_PI3, "## 2. Reproducibility check"),
        note=(
            f"pack: per-update RNG positions of all five streams identical in {pr[1] - pr[0]}/{pr[1]} pairs "
            f"(v2_updates.csv, T17)"
            if pr
            else ""
        ),
    )
    ii = S.read_csv("results/v2_pilots/pilot3/analysis/inherited_identity.csv")
    srcs.append(C.src("results/v2_pilots/pilot3/analysis/inherited_identity.csv"))
    add(
        scope=f"{_qs(ii)}; frozen snapshot of each pair (taken at u400)",
        check="Pilot 3 frozen snapshots identical between the two arms",
        study="Pilot 3",
        check_class="artifact identity",
        in_spec_list=False,
        commit="cd760fd",
        compared="frozen stage-2 snapshot tensors of the stochastic and mean arms of each (q, seed); the inherited term "
        "takes one value in both arms",
        unit="run pair (q, seed)",
        n_units=len(ii),
        n_identical=_bool_count(ii, ["snapshots_bit_identical"]),
        result="pass" if _bool_count(ii, ["snapshots_bit_identical"]) == len(ii) else "fail",
        source_kind="data file",
        source="results/v2_pilots/pilot3/analysis/inherited_identity.csv",
        report_section=_cite(R_PI3, "**Inherited-term identity within pairs.**"),
    )
    # R: Pilot 4 2a vs extension, verifier RNG
    r2a = S.read_csv(f"{P4A}/repro_2a_constant_vs_ext.csv")
    c2a = [
        c for c in r2a.columns if c.endswith("_identical") or c.endswith("_identical_to_ext_u1600")
    ]
    n2a = _bool_count(r2a, c2a)
    srcs.append(C.src(f"{P4A}/repro_2a_constant_vs_ext.csv"))
    add(
        scope=f"{_qs(r2a)}; u1201-u1600",
        check="Pilot 4 2a constant arm vs the Phase A extension over u1201-u1600",
        study="Pilot 4 2a",
        check_class="training reproduction (re-run)",
        in_spec_list=True,
        commit="c92ee74 vs cd760fd",
        compared=", ".join(c2a) + f" ({int(r2a.n_exports_compared.iloc[0])} weight exports, "
        f"{int(r2a.n_updates.iloc[0])} updates)",
        unit="run pair (q, seed)",
        n_units=len(r2a),
        n_identical=n2a,
        result="pass" if n2a == len(r2a) else "fail",
        source_kind="data file",
        source=f"{P4A}/repro_2a_constant_vs_ext.csv",
        report_section=_cite(R_PI4, "**Reproducibility check**"),
        note=f"verifier-call updates coincide ({r2a.verifier_updates_ext.iloc[0]}); verifier_cadence_differs true in "
        f"{int(r2a.verifier_cadence_differs.astype(bool).sum())}/{len(r2a)}",
    )
    vr = S.read_csv(f"{P4A}/verifier_consumes_no_rng.csv")
    nv = _bool_count(vr, ["rng_states_unchanged_after_6_verifier_calls"])
    srcs.append(C.src(f"{P4A}/verifier_consumes_no_rng.csv"))
    add(
        scope=f"{_qs(vr)}; the restored u1200 parents of Pilot 4 2a",
        check="Verifier consumes no RNG (direct check on the restored Pilot 4 parents)",
        study="Pilot 4 2a",
        check_class="RNG non-consumption",
        in_spec_list=False,
        commit="c92ee74 (tools/v2/pilot4_repro.py --verifier-rng)",
        compared="numpy env/learn/opp/start, minibatch stream, torch generator, torch/numpy/python global states before "
        "vs after six verifier calls (dev + final tier, three times)",
        unit="restored parent (q, seed)",
        n_units=len(vr),
        n_identical=nv,
        result="pass" if nv == len(vr) else "fail",
        source_kind="data file",
        source=f"{P4A}/verifier_consumes_no_rng.csv",
        report_section=_cite(R_PI4, "A **direct check** was run instead"),
    )
    # R: dirty re-run
    dr = _json(f"{LK}/consolidation/dirty_rerun_compare.json")
    dk = [k for k, v in dr.items() if isinstance(v, bool) and k != "ALL_IDENTICAL"]
    srcs.append(C.src(f"{LK}/consolidation/dirty_rerun_compare.json"))
    add(
        scope="q=60, seed 10507, B2_mean_constant; Phase B u1601-u2200",
        check="Dirty-flag re-run: clean re-run vs the dirty-flagged Pilot 4 2b run (q=60, seed 10507, B2_mean_constant)",
        study="dirty-flag re-run",
        check_class="training reproduction (re-run)",
        in_spec_list=True,
        commit="5d50a9d vs c92ee74 (dirty)",
        compared=", ".join(dk) + f" ({dr['n_updates']} updates, "
        f"{dr['n_weight_exports']} weight exports)",
        unit="run pair",
        n_units=1,
        n_identical=int(bool(dr["ALL_IDENTICAL"])),
        result="pass" if dr["ALL_IDENTICAL"] else "fail",
        source_kind="data file",
        source=f"{LK}/consolidation/dirty_rerun_compare.json",
        report_section=_cite(R_LOCK, "### 1.3 Dirty-flag re-run"),
    )
    checks += [
        ("dirty re-run updates", dr["n_updates"], "600", None),
        ("dirty re-run weight exports", dr["n_weight_exports"], "24", None),
    ]
    # R: results-root checksums
    rr = S.read_csv(f"{LK}/consolidation/results_root_checksums.csv")
    nid = int(rr["identical"].astype(bool).sum())
    npar = int(rr["pilot4_parent"].astype(bool).sum())
    npar_id = int((rr["pilot4_parent"].astype(bool) & rr["identical"].astype(bool)).sum())
    srcs.append(C.src(f"{LK}/consolidation/results_root_checksums.csv"))
    add(
        scope=f"{len(rr)} files of results/v2_pilots in the original v2 worktree vs the canonical worktree",
        check="Results-root consolidation: every file of the original v2_pilots copy vs the canonical copy",
        study="consolidation (v1.0 lock round)",
        check_class="artifact identity",
        in_spec_list=False,
        commit="5d50a9d (tools/v2/compare_results_roots.py)",
        compared="SHA-256 of each file",
        unit="file",
        n_units=len(rr),
        n_identical=nid,
        result="pass" if nid == len(rr) else "fail",
        source_kind="data file",
        source=f"{LK}/consolidation/results_root_checksums.csv",
        report_section=_cite(R_LOCK, "### 1.2 One development"),
        note=f"includes the {npar} Pilot 4 parent files ({npar_id} identical)",
    )
    m = re.search(
        r"All ([\d,]+) files", _line_text(R_LOCK, "files of the original copy are byte-identical")
    )
    checks.append(("results-root files identical", nid, m.group(1).replace(",", ""), None))
    # R: v1.0 pre-lock insurance (report text)
    ins10 = _line_text(R_LOCK, "**Pre-lock check (not a rehearsal run).**")
    add(
        scope="q=50, seed 10501; locked Phase A u1-u1600 (source: report text)",
        check="v1.0 pre-lock insurance: locked Phase A (q=50, seed 10501, in-process, scratch) vs the stitched development "
        "state at u1600",
        study="v1.0 lock",
        check_class="training reproduction (re-run)",
        in_spec_list=True,
        commit="UNKNOWN (lock content before the lock commit 4bd2214; tree not recorded)",
        compared="actor, critic, opponent, both Adam states, all five RNG streams, torch generator at u1600 (as stated); "
        "weight exports and process-global RNG states: not stated",
        unit="run pair",
        n_units=1,
        n_identical=1,
        result="pass",
        source_kind="report text (no data file: scratch run not kept)",
        source=_cite(R_LOCK, ins10[:60]),
        report_section=_cite(R_LOCK, "### 2.3 Tests"),
        note="source: report text: " + ins10.strip()[:300],
    )
    pack.unknown_value(
        "T18",
        "commit / tree of the v1.0 pre-lock insurance run",
        "run in scratch before the lock commit and not kept; the report does not record the tree",
    )
    # R: Check 1, Check 2
    c1 = S.read_csv(f"{LK}/rehearsal_analysis/check1_phaseA_vs_stitched.csv")
    train_cols = [
        "actor_identical",
        "critic_identical",
        "opponent_identical",
        "opt_actor_identical",
        "opt_critic_identical",
        "rng_minibatch_identical",
        "rng_streams_identical",
        "torch_generator_identical",
        "exports_A_identical",
        "stage2_metrics_u1600_identical",
    ]
    glob_cols = [
        "torch_global_rng_identical",
        "numpy_global_rng_identical",
        "python_random_identical",
    ]
    n_train = _bool_count(c1, train_cols)
    n_all = _bool_count(c1, ["ALL_IDENTICAL"])
    sr = sorted(set(zip(c1.snapshot_refreshes_rehearsal, c1.snapshot_refreshes_stitched)))
    srcs.append(C.src(f"{LK}/rehearsal_analysis/check1_phaseA_vs_stitched.csv"))
    add(
        scope=f"{_qs(c1)}; Phase A u1-u1600",
        check="v1.0 Check 1: rehearsal Phase A vs the stitched development path (Pilot 1 u400 -> extension u1200 -> "
        "2a decay u1600)",
        study="v1.0 rehearsal",
        check_class="training reproduction (re-run)",
        in_spec_list=True,
        commit="5b07293 vs 89cd600/cd760fd/c92ee74",
        compared=", ".join(train_cols + glob_cols)
        + f" ({int(c1.n_exports_A.iloc[0])} weight exports)",
        unit="run (q, seed)",
        n_units=len(c1),
        n_identical=n_all,
        result="fail (literal criterion); accepted by owner decision D1",
        source_kind="data file",
        source=f"{LK}/rehearsal_analysis/check1_phaseA_vs_stitched.csv",
        report_section=f"{_cite(R_LOCK, '### 4.1 Check 1')}; addendum {_cite(R_LOCK, '## Addendum (2026-10-02)')}",
        note=(
            f"training-relevant fields identical in {n_train}/{len(c1)}; the three process-global RNG states identical "
            f"in {_bool_count(c1, glob_cols)}/{len(c1)} (never consumed by training); snapshot_refreshes counter "
            f"(rehearsal, stitched) = {sr} in every run (not one of the compared fields; the stitched path re-entered "
            f"Phase A twice)"
        ),
    )
    c1df = pd.DataFrame([{"field": c, "n": _bool_count(c1, [c])} for c in train_cols + glob_cols])
    pack.crosscheck(
        "T18",
        c1df,
        R_LOCK,
        header_has=["field", "identical (of 20)"],
        key_map={"field": "field"},
        value_map={"identical (of 20)": "n"},
        heading_has="Check 1",
        label="Check 1 field counts",
    )
    c2 = S.read_csv(f"{LK}/rehearsal_analysis/check2_phaseB_vs_launcher.csv")
    c2c = [c for c in c2.columns if c.endswith("_identical") or c == "ALL_IDENTICAL"]
    n2 = _bool_count(c2, ["ALL_IDENTICAL"])
    srcs.append(C.src(f"{LK}/rehearsal_analysis/check2_phaseB_vs_launcher.csv"))
    add(
        scope=f"{_qs(c2)}; Phase B u1601-u2200",
        check="v1.0 Check 2: pilot-launcher Phase B (B2_mean_decay from the stitched u1600 state) vs the rehearsal's "
        "Phase B",
        study="v1.0 rehearsal",
        check_class="training reproduction (re-run)",
        in_spec_list=True,
        commit="5b07293",
        compared=", ".join(c for c in c2c if c != "ALL_IDENTICAL")
        + f" ({int(c2.n_exports_B.iloc[0])} weight exports)",
        unit="run pair (q, seed 10503)",
        n_units=len(c2),
        n_identical=n2,
        result="pass" if n2 == len(c2) else "fail",
        source_kind="data file",
        source=f"{LK}/rehearsal_analysis/check2_phaseB_vs_launcher.csv",
        report_section=_cite(R_LOCK, "### 4.2 Check 2"),
    )
    ct = [
        t
        for t in C.parse_md_tables(R_LOCK)
        if t["header"][:1] == ["field"] and "q50 s10503" in t["header"]
    ][0]
    for r in ct["rows"]:
        if r[0] in c2.columns and r[0].endswith("_identical"):
            for j, q in ((1, 50), (2, 60)):
                v = bool(c2[c2.q == q].iloc[0][r[0]])
                checks.append((f"Check 2 {r[0]} q={q}", v, r[j], (r[j] == "yes") == v))
    # R: cusp identity
    cu = S.read_csv(f"{LK}/cusp_diagnostic/identity_vs_pilot4.csv")
    ncu = _bool_count(cu, ["identical_to_pilot4"])
    srcs.append(C.src(f"{LK}/cusp_diagnostic/identity_vs_pilot4.csv"))
    add(
        scope=f"{_qs(cu, 'init_seed')} (init seeds); full-batch supervised fits, losses compared at the common logged steps",
        check="Cusp diagnostic supervised fits vs the Pilot 4 1d fits",
        study="cusp diagnostic",
        check_class="training reproduction (re-run of the supervised fit)",
        in_spec_list=False,
        commit="344154a (tools/v2/cusp_diagnostic.py)",
        compared=f"float32 loss at every common step ({int(cu.n_common_steps.min())} "
        "steps per fit) against Pilot 4's NPZ loss logs",
        unit="fit (q, init seed)",
        n_units=len(cu),
        n_identical=ncu,
        result="pass" if ncu == len(cu) else "fail",
        source_kind="data file",
        source=f"{LK}/cusp_diagnostic/identity_vs_pilot4.csv",
        report_section=_cite(R_LOCK, "**Identity with Pilot 4.**"),
        note=f"{int((cu.n_identical_float32 == cu.n_common_steps).sum())}/{len(cu)} fits identical at all common steps",
    )
    # R: v1.1 insurance (report text)
    ins11 = _line_text(R_V11, "**Pre-lock insurance (not a rehearsal run).**")
    add(
        scope="q=50, seed 10501; full v1.1 pipeline u1-u2200 (source: report text)",
        check="v1.1 pre-lock insurance: one full-budget v1.1 run (q=50, seed 10501, scratch) vs the v1.0 rehearsal run",
        study="v1.1 lock",
        check_class="training reproduction (re-run)",
        in_spec_list=True,
        commit="UNKNOWN (v1.1 content before the lock commit 431474d; tree not recorded)",
        compared="all training-relevant state (D1 definition) and the global-RNG assertions (as stated); field list "
        "and counts not stated",
        unit="run pair",
        n_units=1,
        n_identical=1,
        result="pass",
        source_kind="report text (no data file: scratch run not kept)",
        source=_cite(R_V11, ins11[:60]),
        report_section=_cite(R_V11, "### 1.4 Tests and C7"),
        note="source: report text: " + ins11.strip()[:300],
    )
    pack.unknown_value(
        "T18",
        "commit / tree and compared-field list of the v1.1 pre-lock insurance run",
        "run in scratch before the lock commit and not kept; the report states only 'all training-relevant "
        "state matched'",
    )
    # R: R1
    r1 = S.read_csv(f"{LK}/rehearsal_v1_1_R1_details.csv")
    r1c = [c for c in r1.columns if c not in ("q", "seed", "ALL")]
    nr1 = _bool_count(r1, ["ALL"])
    rj = _json(R16_CHECKS)
    srcs += [C.src(f"{LK}/rehearsal_v1_1_R1_details.csv"), C.src(R16_CHECKS)]
    add(
        scope=f"{_qs(r1)}; full pipeline u1-u2200",
        check="v1.1 R1: re-rehearsal (v1.1 entry point) vs the v1.0 rehearsal, same seeds",
        study="v1.1 re-rehearsal",
        check_class="training reproduction (re-run)",
        in_spec_list=True,
        commit="95c000e vs 5b07293",
        compared=f"{len(r1c)} fields: " + ", ".join(r1c),
        unit="run (q, seed)",
        n_units=len(r1),
        n_identical=nr1,
        result="pass" if (nr1 == len(r1) and rj["R1"]["pass"]) else "fail",
        source_kind="data file",
        source=f"{LK}/rehearsal_v1_1_R1_details.csv; {R16_CHECKS} (R1)",
        report_section=_cite(R_V11, "| R1 training-"),
    )
    checks.append(
        ("R1 n_identical: details CSV vs checks JSON", nr1, str(rj["R1"]["n_identical"]), None)
    )
    # R: global-RNG assertions (v1.1)
    v11 = [rd for st in ("rehearsal_v1_1", "confirmation") for _, _, _, rd in _runs(st)]
    pts = ("end_of_A", "after_G-A", "end_of_B", "end_of_run")
    n_as = 0
    for rd in v11:
        g = _manifest(rd)["global_rng"]
        ref = g["reference_before_first_A_update"]["digests"]
        n_as += int(all(g[p_]["digests"] == ref for p_ in pts))
    srcs.append(
        C.srcs([f"{rd}/manifest.json" for rd in v11], label="manifest.json of the 60 v1.1 runs")
    )
    add(
        scope=f"{len(v11)} runs: "
        + "; ".join(
            f"{LABEL[st_]} "
            + _qs(pd.DataFrame([{"q": q_, "seed": s_} for q_, s_, _, _ in _runs(st_)]))
            for st_ in ("rehearsal_v1_1", "confirmation")
        ),
        check="v1.1 global-RNG assertions: torch global, numpy legacy global and Python random states equal the reference "
        "(before the first Phase A update)",
        study="v1.1 re-rehearsal and confirmation",
        check_class="RNG state identity (in-run assertion)",
        in_spec_list=False,
        commit="95c000e (20 runs), f6838ec (40)",
        compared="SHA-256 digests of the three states at " + ", ".join(pts),
        unit="run",
        n_units=len(v11),
        n_identical=n_as,
        result="pass" if n_as == len(v11) else "fail",
        source_kind="data file",
        source="results/v2_T2_locked/{rehearsal_v1_1,confirmation}/q*/seed*/manifest.json (global_rng)",
        report_section=_cite(R_V11, "## 2. Hardening of the process-global RNGs"),
        note="see T17 block 3",
    )
    # R: R5 and confirmation agreement
    for rel, lab, sec_needle in (
        (
            f"{LK}/rehearsal_v1_1_analysis/agreement.csv",
            "v1.1 R5: pre-registered analysis script vs every run's gates.json (re-rehearsal)",
            "| R5 analysis script agrees",
        ),
        (
            f"{LK}/confirmation_analysis/agreement.csv",
            "Confirmation: pre-registered analysis script vs every run's gates.json",
            "The script's recomputed values and verdicts",
        ),
    ):
        ag = S.read_csv(rel)
        vcols = [c for c in ag.columns if c.startswith("absdiff_")]
        mx = float(ag[vcols].to_numpy(dtype=float).max())
        na = _bool_count(ag, ["all_agree"])
        srcs.append(C.src(rel))
        add(
            scope=f"{_qs(ag)}",
            check=lab,
            study="v1.1 re-rehearsal" if "rehearsal" in rel else "confirmation",
            check_class="analysis recomputation (exact)",
            in_spec_list=False,
            commit="95c000e" if "rehearsal" in rel else "f6838ec",
            compared="metric values "
            + ", ".join(vcols)
            + "; verdicts G-A, G-F, G-N, S1, v1.0 outcome, run pass, outcome",
            unit="run (q, seed)",
            n_units=len(ag),
            n_identical=na,
            result="pass" if (na == len(ag) and mx == 0.0) else "fail",
            source_kind="data file",
            source=rel,
            report_section=_cite(R_V11, sec_needle),
            note=f"max absolute value difference {mx:g}",
        )
    # R: snapshot integrity (T14)
    fr = _CACHE["t14_runs"]
    nfp = int(fr["pass"].sum())
    add(
        scope=f"{len(fr)} frozen-mode runs: "
        + "; ".join(f"{LABEL[s_]} {int((fr.study == s_).sum())}" for s_ in dict.fromkeys(fr.study)),
        check="C5 snapshot integrity (drift_test.json) in every frozen-mode run",
        study="Phase 2 smoke, Pilots 2-4 2b, "
        "dirty-flag re-run, v1.0 rehearsal and Check 2, v1.1 re-rehearsal, confirmation",
        check_class="artifact identity (frozen snapshot)",
        in_spec_list=False,
        commit=", ".join(sorted(set(fr.commit))),
        compared="snapshot tensors vs the parent / end-of-A actor; stage-2 "
        "mean, alpha, beta on the dev D_2 grid at freeze time vs end of Phase B",
        unit="run",
        n_units=len(fr),
        n_identical=nfp,
        result="pass" if nfp == len(fr) else "fail",
        source_kind="data file",
        source="drift_test.json of every frozen-mode run (pack item T14)",
        report_section=f"{_cite(R_PI2, '### 2.6 Snapshot integrity')} and later reports",
        note="see T14",
    )
    df = pd.DataFrame(rows)
    df.insert(0, "check_id", [f"R{i:02d}" for i in range(1, len(df) + 1)])
    cols = [
        "check_id",
        "check",
        "study",
        "scope",
        "check_class",
        "in_spec_list",
        "commit",
        "compared",
        "unit",
        "n_units",
        "n_identical",
        "result",
        "source_kind",
        "source",
        "report_section",
        "note",
    ]
    df = df[cols]
    for c in ("n_units", "n_identical"):
        df[c] = _intcol(df[c])
    _manual(pack, "T18", f"{R_LOCK}, {R_V11}", "ledger counts vs report text", checks)
    # R1-R6 table of the v1.1 report
    rt = [t for t in C.parse_md_tables(R_V11) if t["header"] == ["Check", "Verdict", "Detail"]][0]
    rr_pairs = []
    for r in rt["rows"]:
        key = r[0].split()[0]
        jv = rj.get(key, {}).get("pass")
        rr_pairs.append(
            (
                f"{key} verdict (checks JSON vs report)",
                "pass" if jv else "fail",
                r[1],
                (("pass" in r[1]) == bool(jv)),
            )
        )
    _manual(
        pack,
        "T18",
        f"{R_V11} line {rt['line']}",
        "R1-R6 verdicts vs rehearsal_v1_1_checks.json",
        rr_pairs,
    )
    srcs += [
        C.src(R_PI2),
        C.src(R_PI3),
        C.src(R_PI4),
        C.src(R_LOCK),
        C.src(R_V11),
        pack.pack_src("T14"),
        pack.pack_src("T15"),
    ]
    docs = {
        "check_id": "Row id of the ledger (one row = one named check)",
        "check": "Name of the bit-identity check",
        "study": "Study or phase in which the check was run",
        "scope": "What the check covers: q values and seeds (from the source file where it has them), update range, "
        "or the test fixture",
        "check_class": "training reproduction (re-run gives bit-identical state and logs) / state restore or branching / "
        "code-path regression / artifact identity / RNG state identity / RNG non-consumption / analysis "
        "recomputation (exact)",
        "in_spec_list": {
            "definition": "Whether the check is one of the checks named in the T18 request (Phase 2 "
            "branching test, Pilot 3 vs Pilot 2 B2, Pilot 4 2a vs extension, dirty-flag re-run, "
            "pre-lock insurance runs, v1.0 Check 1 and Check 2, v1.1 R1)",
            "units": "bool",
        },
        "commit": "Commit(s) at which the compared runs or the test ran",
        "compared": "Fields / files / states compared (from the source file's columns or keys, or the report text)",
        "unit": "What one comparison unit is (run pair, run, commit, test case, file, ...)",
        "n_units": {"definition": "Number of comparison units in the check", "units": "count"},
        "n_identical": {
            "definition": "Units in which every compared field was identical (literal criterion)",
            "units": "count",
        },
        "result": "pass (all units identical) or fail; Check 1 fails on its literal criterion",
        "source_kind": "data file / test code + report text / report text",
        "source": "Source file(s); report text cited with line",
        "report_section": "Report section (file and line) that presents the check",
        "note": "Counts of sub-criteria, pack re-checks, and statements quoted from the report",
    }
    notes = (
        "One row per named check; n_units / n_identical give the run-level (or test-case, commit, file) counts. "
        "Counts recomputed from the data files where they exist (booleans counted, compare files parsed, JSON "
        "keys); report-text rows (pre-lock insurance runs, unit-test results) are marked 'report text' and quote "
        "the report. Pack re-checks: B7 and the 8 C7 runs re-compared bit for bit (NPZ arrays, JSON leaves). "
        "Tolerance-based agreements (Pilot 1 u400 export vs checkpoint <= 1e-16; Pilot 2 u1000 re-evaluation <= "
        "9.9e-17) and document byte-identity checks (LOCK record prefix, v1.0 protocol files, report prefix) are "
        "not bit-identity checks of runs and are not listed."
    )
    pack.table(
        "T18",
        df,
        status="generated",
        sources=srcs,
        script=f"{MOD}:build_t18",
        notes=notes,
        docs=docs,
        tier="n/a",
    )
    return df


# ------------------------------------------------------------------------------------------------
# entry
# ------------------------------------------------------------------------------------------------


def build() -> None:
    """Build T13-T18 and save the module fragment."""
    pack = C.Pack("sec_method")
    build_t13(pack)
    build_t14(pack)
    build_t15(pack)
    build_t16(pack)
    build_t17(pack)
    build_t18(pack)
    pack.save_fragment()
