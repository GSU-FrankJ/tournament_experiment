#!/usr/bin/env python3
"""Tables of ``reports/t3/r4/01_lock.md``: the settings and the rule of the lock draft, the operating characteristic recomputed here
independently of ``tools/t3/t3r4_make_lock.py``, the plan, the config differences against T3-R3's ``bwA`` configs, the seed inventory and
C-REPRO. Reads ``reports/t3/r4/lock_draft/``, ``reports/t3/r4/confirmation_plan.csv``, ``results/t3_r4/preflight/`` and
``results/t3_r4/repro/``. Writes ``reports/t3/r4/tables/lock_*.md``."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, md  # noqa: E402

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))
sys.path.insert(0, str(ROOT / "tools" / "t3"))

DRAFT = ROOT / "reports" / "t3" / "r4" / "lock_draft" / "v2_T3_locked_v1_1.json"
PLAN = ROOT / "reports" / "t3" / "r4" / "confirmation_plan.csv"
PRE = ROOT / "results" / "t3_r4" / "preflight"
REPRO = ROOT / "results" / "t3_r4" / "repro"


def lock() -> Dict[str, Any]:
    """The lock draft (JSON)."""
    return json.loads(DRAFT.read_text())


def t_settings() -> str:
    L = lock()
    s, c = L["settings"], L["confirmation"]
    rows = [
        {"item": "records", "value": f"the v2.0 record of each q (`{L['protocol_file']}`; the record itself has T = 2, the run configs set `game.T` = 3)"},
        {"item": "parameters file", "value": f"`{L['params_file']}`"},
        {"item": "phases, budgets (updates)", "value": f"{s['phases']}, {json.dumps(s['budgets'])}"},
        {"item": "learning rate", "value": s["lr"]["text"]},
        {"item": "episodes per update, starts", "value": f"{s['episodes_per_update']}, {json.dumps(s['start_weights'])}"},
        {"item": "rule, init digest, clamp likelihood", "value": f"disabled, {s['init_digest']}, `{s['clamp_likelihood']}`"},
        {"item": "checks, snapshots, exports", "value": f"every {s['check_every']}, {s['snapshot_every']}, {s['weights_every']} updates"},
        {"item": "restart (the only change against T3-1.0)", "value": f"`stage3_retry = {json.dumps(s['stage3_retry'])}`"},
        {"item": "evaluated policy", "value": s["evaluated_policy"]},
        {"item": "seed block, runs", "value": f"{c['seed_block'][0]}-{c['seed_block'][1]} at q in {c['qs']}: {c['n_runs']} runs, arm `{c['arm']}`"},
        {"item": "run names, directories", "value": f"`{c['run_name']}`, `{c['run_dir']}`"},
        {"item": "fields set per run", "value": ", ".join(f"`{f}`" for f in c["config_fields_set_per_run"])}]
    return md(rows)


def t_oc() -> str:
    """The operating characteristic recomputed with ``scipy.stats.binom`` (not with ``t3r4_make_lock.pass_probability``), set against the
    values in the lock."""
    L = lock()
    given = L["confirmation"]["operating_characteristic"]["pass_probability"]
    rows = []
    for f_txt, p_lock in given.items():
        f = float(f_txt)
        pmf = binom.pmf(range(5), 40, f)
        p = float(sum(pmf[a] * pmf[b] for a in range(5) for b in range(5) if a + b <= 6))
        rows.append({"per-run failure probability f (both q)": f, "P(at most 6 of 80 and at most 4 of 40 per q fail), recomputed": f"{p:.6f}",
                     "in the lock": f"{p_lock:.6f}", "agree to 1e-6": abs(p - p_lock) < 1e-6 + 5e-7})
    return md(rows)


def t_plan() -> str:
    if not PLAN.is_file():
        return "(the plan file does not exist yet)"
    rows = []
    for r in csv.DictReader(open(PLAN)):
        rows.append({"run": f"`{r['run']}`", "config SHA-256 (16)": r["run_config_sha256"][:16], "init digest, attempt 1 (16)": r["init_state_sha256"][:16],
                     "init digest, attempt 2 (16)": r["attempt2_init_state_sha256"][:16]})
    return md(rows)


def t_cfgdiff() -> str:
    import t3_configs as tc
    import t3r3_launch as r3
    import t3r4_launch as lt
    proto, params = tc.load_protocol(), tc.load_bw_params()
    rows = []
    for q in lt.QS:
        old = r3.build_arm_config(proto, params, "bwA", q, 70501, "OUT/old")
        new = lt.build_confirmation_config(proto, params, q, 80501, "OUT/new")
        rows.append({"q": q, "T3-R3 `bwA` config (seed 70501) against the T3-R4 config (seed 80501): differing config paths":
                     ", ".join(f"`{k}`" for k in tc.config_diff(old, new))})
    return md(rows)


def t_seeds() -> str:
    out = (PRE / "seed_inventory_80501_80540.out.txt").read_text()
    scan = next(line for line in out.splitlines() if line.startswith("scanned"))
    coll = next(line for line in out.splitlines() if "collisions" in line)
    hits = sum(1 for _ in csv.DictReader(open(PRE / "seed_hits_80501_80540.csv")))
    return md([{"block": "80501-80540", "scan": scan, "result": coll, "hit rows (`seed_hits.py`)": hits}])


def t_repro() -> str:
    files = sorted(REPRO.glob("compare_*.json"))
    if not files:
        return "(C-REPRO has not been run yet)"
    rows: List[Dict[str, Any]] = []
    for f in files:
        d = json.loads(f.read_text())
        rel = lambda p: p[p.index("results/"):] if "results/" in p else p          # noqa: E731
        a1 = d.get("attempt1")
        rows.append({"run": d["run"], "trigger fired in the original": d["fired_in_original"], "re-run": rel(d["new"]), "original (T3-R3 untracked root)": rel(d["orig"]),
                     "code commit of the re-run": d["commit"][:8], "`compare_runs.py bw` exit code": d.get("compare_runs_returncode"),
                     "npz files compared": d.get("n_npz"), "same set of files": d.get("file_sets_equal"),
                     "stage-2 and stage-3 end states (bitwise)": all(d.get("states_equal", {"x": False}).values()),
                     "gates, rule log, drift test, summary, manifest, train history": "identical" if not any(d.get("json_diffs", {"x": ["?"]}).values()) else "DIFFERENT",
                     "`attempt1/` and `retry.json`": ("identical" if (a1 and a1["ok"] and not d["retry_json_diff"]) else "DIFFERENT") if d["fired_in_original"] else "-",
                     "all identical": d["ok"], "wall time (s)": d.get("wall_sec")})
    return md(rows)


TABLES = {"lock_settings": t_settings, "lock_oc": t_oc, "lock_plan": t_plan, "lock_cfgdiff": t_cfgdiff, "lock_seeds": t_seeds, "lock_repro": t_repro}


def main() -> int:
    """Write the tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in TABLES.items():
        (OUT / f"{name}.md").write_text(fn() + "\n")
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
