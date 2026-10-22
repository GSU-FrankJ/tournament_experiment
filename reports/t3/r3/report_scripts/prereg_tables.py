#!/usr/bin/env python3
"""Tables of ``reports/t3/r3/04_preregistration.md``: the arms, the config differences, the plan, the seed inventories and C-LEGACY.
Reads ``reports/t3/r3/dev_plan.csv`` (``tools/t3/t3r3_launch.py --plan-csv``), ``results/t3_r3/preflight/`` and
``results/t3_r3/legacy/``. Writes ``reports/t3/r3/tables/prereg_*.md``."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "t3"))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
from diagnosis_tables import OUT, md  # noqa: E402

PLAN = ROOT / "reports" / "t3" / "r3" / "dev_plan.csv"
PRE = ROOT / "results" / "t3_r3" / "preflight"
LEGACY = ROOT / "results" / "t3_r3" / "legacy"


def t_arms() -> str:
    rows = [
        {"arm": "`bw`", "role": "control (locked T3-1.0 settings); not eligible", "scheme keys": "none", "runs": "40 (+ none diagnostic)",
         "order in the selection": "-"},
        {"arm": "`bwA`", "role": "candidate A", "scheme keys": '`stage3_retry = {"delta_gt": 0.005, "max_retries": 1}`',
         "runs": "40 + 1 diagnostic", "order in the selection": "1"},
        {"arm": "`bwAB`", "role": "candidate A and B", "scheme keys": '`stage3_retry` as above and `terminal_clamp_likelihood = "censored"`',
         "runs": "40 + 1 diagnostic", "order in the selection": "2"}]
    return md(rows)


def t_cfgdiff() -> str:
    import t3_configs as tc
    import t3r3_launch as lt
    params = tc.load_bw_params()
    proto = tc.load_protocol()
    base = lt.build_arm_config(proto, params, "bw", 50, 70501, "OUT/bw")
    rows = []
    for arm in lt.CANDIDATES:
        cfg = lt.build_arm_config(proto, params, arm, 50, 70501, f"OUT/{arm}")
        rows.append({"arm": f"`{arm}` against `bw`", "differing config paths (q = 50, seed 70501; the same for every seed, tested)":
                     ", ".join(f"`{k}`" for k in tc.config_diff(base, cfg))})
    return md(rows)


def t_plan() -> str:
    if not PLAN.is_file():
        return "(the plan file does not exist yet)"
    rows = []
    for r in csv.DictReader(open(PLAN)):
        rows.append({"run": f"`{r['run']}`", "kind": r["kind"], "config SHA-256 (16)": r["run_config_sha256"][:16],
                     "init digest, attempt 1 (16)": r["init_state_sha256"][:16],
                     "init digest, attempt 2 (16)": r["attempt2_init_state_sha256"][:16] or "-"})
    return md(rows)


def t_seeds() -> str:
    rows = []
    for tag, role in (("70501_70520", "development block"), ("80501_80540", "reserved for the later confirmation")):
        out = (PRE / f"seed_inventory_{tag}.out.txt").read_text()
        coll = [l for l in out.splitlines() if "collisions" in l][0]
        scan = [l for l in out.splitlines() if l.startswith("scanned")][0]
        hits = sum(1 for _ in csv.DictReader(open(PRE / f"seed_hits_{tag}.csv")))
        rows.append({"block": tag.replace("_", "-"), "role": role, "scan": scan, "result": coll, "hit rows (`seed_hits.py`)": hits})
    return md(rows)


def t_legacy() -> str:
    files = sorted(LEGACY.glob("compare_*.json"))
    if not files:
        return "(C-LEGACY has not been run yet)"
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        rel = lambda p: p[p.index("results/"):] if "results/" in p else p          # noqa: E731
        rows.append({"run": d["run"], "re-run": rel(d["new"]), "original (T3-R2 untracked root)": rel(d["orig"]),
                     "code commit of the re-run": d["commit"][:8],
                     "npz files compared": d["n_npz"], "all identical": d["ok"], "wall time (s)": d.get("wall_sec")})
    return md(rows)


TABLES = {"prereg_arms": t_arms, "prereg_cfgdiff": t_cfgdiff, "prereg_plan": t_plan, "prereg_seeds": t_seeds, "prereg_legacy": t_legacy}


def main() -> int:
    """Write the tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in TABLES.items():
        (OUT / f"{name}.md").write_text(fn() + "\n")
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
