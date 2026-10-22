#!/usr/bin/env python3
"""Tables of ``reports/t3/r4/03_confirmation.md`` (and ``summary.md``): every table is generated here from
``results/t3_r4/confirmation/launch_checks.json``, ``infra_events.json``, ``eval/per_run.csv`` and ``analysis/*`` (written by
``tools/t3/t3r4_checks.py``, ``t3r4_eval.py``, ``t3r4_analysis.py``, ``t3r4_blind.py``). Writes ``reports/t3/r4/tables/conf_*.md``.

Environment (for a rehearsal on other data): ``T3R4_CONF`` (the confirmation root, default ``results/t3_r4/confirmation``) and
``T3R4_TABLES_OUT`` (the output directory, default ``reports/t3/r4/tables``)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT as DEFAULT_OUT, ROOT, md  # noqa: E402

CONF = Path(os.environ.get("T3R4_CONF", ROOT / "results" / "t3_r4" / "confirmation"))
OUT = Path(os.environ.get("T3R4_TABLES_OUT", DEFAULT_OUT))
AN = CONF / "analysis"


def rd(name: str) -> pd.DataFrame:
    """A CSV of the analysis directory."""
    return pd.read_csv(AN / name)


def head() -> Dict[str, Any]:
    """``headline.json``."""
    return json.loads((AN / "headline.json").read_text())


# ---------------------------------------------------------------------------------- checks
def t_checks() -> str:
    c = json.loads((CONF / "launch_checks.json").read_text())
    rows = [{"check": v, "pass": x["pass"], "fail": x["fail"], "not evaluated": x["not_evaluated"]} for v, x in c["verdicts"].items()]
    led = c["infra_events"]
    txt = (f"\n\nRuns checked: {c['n_runs']}; trigger fired: {c['fired']}; trigger would fire (recomputed from the check file it read): {c['would_fire']}; "
           f"check errors: {len(c['errors'])}. Infrastructure ledger present: {c['infra_events_present']}; events: "
           f"{len(led['events']) if led else 'n/a'}; stopped without verdict: {led['stopped_without_verdict'] if led else 'n/a'}.")
    return md(rows) + txt


# ---------------------------------------------------------------------------------- per seed
def t_perseed(q: int) -> str:
    d = pd.read_csv(AN / "per_run.csv")
    rows = []
    for _, r in d[d.q == q].sort_values("seed").iterrows():
        tag = "pass" if r.passed else f"FAIL ({r.failing_gates if isinstance(r.failing_gates, str) and r.failing_gates else r.fail_reason})"
        if r.fired:
            tag += " [fired]"
        rows.append({"seed": int(r.seed), "verdict": tag, "eta_3": r.eta_T_over_dw, "rmse_pos/e*(0)": r.rmse_pos_T_over_eT0,
                     "tail_mean/e*(0)": r.tail_mean_T_over_eT0, "Gmax_full": r.Gmax_full_over_dw, "S3 (frozen)": r.S3_frozen,
                     "stage-2 stretch nodes": int(r.stretch_nodes), "wall (s)": r.wall_sec})
    return md(rows)


# ---------------------------------------------------------------------------------- verdict
def t_verdict() -> str:
    h = head()
    c, v = h["counts"], h["verdict"]
    rows = []
    for q, lim in ((50, 4), (60, 4), ("both", 6)):
        key = "total" if q == "both" else f"q{q}"
        rows.append({"q": q, "runs": c[f"n_{key}"], "pass": c[f"pass_{key}"], "fail": c[f"fail_{key}"], "limit on failures": lim,
                     "exact 95 % interval of the pass rate": f"[{c[f'ci_lo_{key}']:.3f}, {c[f'ci_hi_{key}']:.3f}]",
                     "condition met": v["total_condition"] if q == "both" else v[f"q{q}_condition"]})
    b = json.loads((AN / "blind.json").read_text())
    txt = (f"\n\n**Verdict under the rule: {v['verdict'].upper()}.** {v['reason']}. Blind recomputation (`tools/t3/t3r4_blind.py`): verdict "
           f"{b['verdict']}, failures {b['fail_q50']} + {b['fail_q60']} = {b['fail_total']}; agrees with the analysis: {b.get('agrees_with_headline')}; "
           f"differences: {b.get('differences')}.")
    return md(rows) + txt


def t_values() -> str:
    d = rd("counts.csv")
    rows = [{"q": x.q, "runs": int(x.runs), "pass": int(x["pass"]), "worst Gmax_full (run)": f"{x.worst_Gmax_full:.5f} ({x.worst_Gmax_full_run})",
             "median eta_3": x.median_eta_T_over_dw, "median rmse_pos/e*(0)": x.median_rmse_pos_T_over_eT0,
             "median tail_mean/e*(0)": x.median_tail_mean_T_over_eT0, "median peak_T_signed": x.median_peak_T_signed,
             "median Gmax_full": x.median_Gmax_full_over_dw, "median EXP_root": x.median_EXP_root_over_dw} for _, x in d.iterrows()]
    return md(rows)


# ---------------------------------------------------------------------------------- trigger
def t_trigger() -> str:
    t = head()["trigger"]
    d = rd("fired.csv")
    txt = (f"Trigger fired in {t['fired']} of {t['n']} runs (rate {t['rate']:.4f}, exact 95 % interval [{t['ci_lo']:.4f}, {t['ci_hi']:.4f}]); "
           f"{t['fired_runs_that_pass']} of the {t['fired']} fired runs pass on their frozen second attempt. **First-attempt outcome** (every fired run "
           f"counted as a failure of its first attempt): {t['first_attempt_pass']} of {t['n']} pass (rate {t['first_attempt_rate']:.4f}).")
    if len(d) == 0:
        return txt + "\n\nNo trigger fired."
    rows = [{"run": f"`{r.run}`", "q": r.q, "trigger Delta": r.trigger_Delta, "s": r.s, "C": r.C, "reason": r.reasons, "S3 of attempt 1": r.S3_first,
             "S3 of the frozen attempt 2": r.S3_frozen, "attempt-2 final Delta": r.attempt2_final_Delta,
             "outcome": "pass" if r.passed else f"FAIL ({r.fail_reason})", "eta_3": r.eta_3, "Gmax_full": r.Gmax_full} for _, r in d.iterrows()]
    return txt + "\n\n" + md(rows)


# ---------------------------------------------------------------------------------- failures
def t_failures() -> str:
    f = rd("failures.csv")
    if len(f) == 0:
        return "No run failed."
    l16 = rd("last16_summary.csv") if (AN / "last16_summary.csv").is_file() and (AN / "last16_summary.csv").stat().st_size > 5 else pd.DataFrame()
    rows = []
    for _, r in f.iterrows():
        mine = l16[l16.run == r.run] if len(l16) else pd.DataFrame()
        ev = "; ".join(f"{m.quantity} {m.final_value:.5f} (limit {m.limit}): rank {int(m.final_rank_from_top)} of 16 from the top, {int(m.n_above_limit)} of 16 above"
                       for _, m in mine.iterrows() if m.failing)
        rows.append({"run": f"`{r.run}`", "failing gates": r.failing_gates, "failing quantities (stage)": r.failing_quantities, "S3 (first / frozen)":
                     f"{r.S3_first:.4g} / {r.S3_frozen:.4g}", "fired": r.fired, "stage-2 stretch nodes": int(r.stretch_nodes),
                     "F(u01600) (threshold)": f"{r.F_u1600:.4g} ({r.formation_threshold:.4g})", "last 16 exports": ev or "-",
                     "class (suggested by rule)": r.class_suggested})
    return md(rows)


# ---------------------------------------------------------------------------------- cost and comparisons
def t_cost() -> str:
    d = rd("cost.csv")
    rows = [{"q": r.q, "runs": int(r.runs), "fired": int(r.fired), "fired share": r.fired_share, "mean updates": r.mean_updates,
             "mean policy rows": r.mean_policy_rows, "mean minibatch steps": r.mean_minibatch_steps, "mean dev checks": r.mean_dev_checks,
             "ratio updates (to a run that does not fire)": r.ratio_updates_to_not_fired, "total wall (s)": r.total_wall_sec,
             "mean wall (s)": r.mean_wall_sec, "median wall (s)": r.median_wall_sec, "mean wall, fired (s)": r.mean_wall_sec_fired,
             "mean wall, not fired (s)": r.mean_wall_sec_not_fired} for _, r in d.iterrows()]
    return md(rows)


def t_comparisons() -> str:
    d = rd("comparisons.csv")
    rows = [{"set": r.set, "q": r.q, "runs": int(r.n), "passes": int(r.passes), "rate": r.rate, "exact 95 % interval": f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"}
            for _, r in d.iterrows()]
    return md(rows)


TABLES = {"conf_checks": t_checks, "conf_perseed_q50": lambda: t_perseed(50), "conf_perseed_q60": lambda: t_perseed(60), "conf_verdict": t_verdict,
          "conf_values": t_values, "conf_trigger": t_trigger, "conf_failures": t_failures, "conf_cost": t_cost, "conf_comparisons": t_comparisons}


def main() -> int:
    """Write the tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in TABLES.items():
        (OUT / f"{name}.md").write_text(fn() + "\n")
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
