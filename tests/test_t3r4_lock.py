"""Tests of ``tools/t3/t3r4_make_lock.py``: determinism, contents against the prompt, the operating characteristic, the list of files."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Set

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools" / "ms", ROOT / "tools" / "t3", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
for _k in ("OMP", "MKL", "OPENBLAS"):
    os.environ[f"{_k}_NUM_THREADS"] = "1"

import t3_configs as tc  # noqa: E402
import t3r4_launch as lt  # noqa: E402
import t3r4_make_lock as ml  # noqa: E402

PAPER = ("a verifier-gated restart of the terminal stage: when the development-tier one-step deviation gain at the end of the stage-3 phase "
         "exceeds the gate's limit 0.005, stage 3 is re-initialised once and retrained, and the second attempt is kept; its cost is part of "
         "the method's cost.")


@pytest.fixture(scope="module")
def drafts(tmp_path_factory):
    a, b = tmp_path_factory.mktemp("a"), tmp_path_factory.mktemp("b")
    assert ml.main(["--out", str(a)]) == 0 and ml.main(["--out", str(b)]) == 0
    return a, b


def test_two_calls_give_byte_identical_files(drafts):
    a, b = drafts
    for name in ("v2_T3_locked_v1_1.json", "v2_T3_locked_v1_1.md"):
        assert (a / name).read_bytes() == (b / name).read_bytes() and (a / name).stat().st_size > 1000


def test_the_files_hold_no_timestamp_no_commit_and_no_absolute_path(drafts):
    for name in ("v2_T3_locked_v1_1.json", "v2_T3_locked_v1_1.md"):
        text = (drafts[0] / name).read_text()
        assert "/home/" not in text and str(ROOT) not in text
        assert not re.search(r"\d{4}-\d{2}-\d{2}", text) and not re.search(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", text)


def test_the_lock_holds_what_the_prompt_asks_for(drafts):
    lock = json.loads((drafts[0] / "v2_T3_locked_v1_1.json").read_text())
    md = (drafts[0] / "v2_T3_locked_v1_1.md").read_text()
    c, s = lock["confirmation"], lock["settings"]
    assert lock["protocol_version"] == "T3-1.1" and lock["paper_wording"] == PAPER and f"> {PAPER}" in md
    assert s["stage3_retry"] == {"delta_gt": 0.005, "max_retries": 1} and s["budgets"] == {"3": 1600, "2": 1600, "1": 600}
    assert c["seed_block"] == [80501, 80540] and c["qs"] == [50, 60] and c["n_runs"] == 80 and c["arm"] == "bwA" and c["same_seeds_at_both_q"]
    assert c["rule"]["max_failures_total"] == 6 and c["rule"]["max_failures_per_q"] == 4 and c["rule"]["runs_per_q"] == 40
    oc = c["operating_characteristic"]["pass_probability"]
    assert [round(oc[k], 3) for k in ("0.03", "0.05", "0.1", "0.15")] == [0.982, 0.865, 0.271, 0.030]
    fc = c["failure_counting"]
    assert fc["infrastructure_failure"] and "second" not in fc["reporting"] and "stops without a verdict" in fc["rerun_after_infrastructure_failure"]
    assert "never a re-run" in fc["scheme_steps"] and "any other nonzero exit" in fc["other_failures"]
    assert lock["gates"]["G-A"]["eta_T_over_dw"] == 0.005 and lock["gates"]["G-F"]["Gmax_full_over_dw"] == 0.01
    assert lock["builds_on"]["protocol_sha256"] == hashlib.sha256((ROOT / "protocols" / "v2_T3_locked_v1_0.json").read_bytes()).hexdigest()
    proto, params = tc.load_protocol(), tc.load_bw_params()
    assert lock["reference_config_sha256"] == {str(q): lt.reference_config_sha256(proto, params, q) for q in lt.QS}
    for f in ("run/ms_retry.py", "run/run_ms_stagewise.py", "tools/t3/t3r4_launch.py", "tools/t3/t3r4_make_lock.py", "tools/t3/t3r4_checks.py",
              "tools/t3/t3r4_eval.py", "tools/t3/t3r4_analysis.py", "tools/t3/t3r4_blind.py", "tools/t3/t3r4_post_launch.sh"):
        assert lock["tool_sha256"][f] == hashlib.sha256((ROOT / f).read_bytes()).hexdigest(), f


@pytest.mark.parametrize("f,expected", [(0.03, 0.98182), (0.05, 0.86476), (0.10, 0.27079), (0.15, 0.02989), (0.0, 1.0), (1.0, 0.0)])
def test_the_operating_characteristic_matches_the_prereg_values(f, expected):
    assert ml.pass_probability(f) == pytest.approx(expected, abs=2e-5)


def test_the_operating_characteristic_by_enumeration():
    """A direct enumeration over (failures at q = 50, failures at q = 60) agrees with the closed form for an asymmetric check."""
    from math import comb
    f = 0.07
    pmf = [comb(40, k) * f ** k * (1 - f) ** (40 - k) for k in range(41)]
    want = sum(pmf[a] * pmf[b] for a in range(41) for b in range(41) if a <= 4 and b <= 4 and a + b <= 6)
    assert ml.pass_probability(f) == pytest.approx(want, rel=1e-12)


def _local_imports(path: Path) -> Set[str]:
    tree = ast.parse(path.read_text())
    found: Set[str] = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for n in names:
            for pre in ("tools/t3/", "tools/ms/", "tools/v2/", ""):
                p = pre + n.replace(".", "/") + ".py"
                if (ROOT / p).is_file() and (p.startswith("tools/") or p.startswith("run/")):
                    found.add(p)
    return found


def test_the_lock_lists_every_tool_and_run_module_that_the_t3r4_tools_import():
    listed = set(ml.tool_files())
    todo = [p for p in sorted(listed) if p.startswith("tools/t3/t3r4_") and p.endswith(".py")]
    seen: Set[str] = set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        todo += sorted(_local_imports(ROOT / p) - seen)
    # tools/ms, tools/v2 and run/ are covered by the launcher's guard (the diff since the base commit over them is empty); tools/t3 is not
    missing = sorted(p for p in seen - listed if p.startswith("tools/t3/"))
    assert not missing, f"imported by the t3r4 tools (directly or not) but not listed in the lock: {missing}"
    assert all(p.startswith(tuple(f"{d}/" for d in lt.CODE_DIFF_PATHS)) for p in seen - listed), sorted(seen - listed)
    assert all((ROOT / f).is_file() for f in listed)
