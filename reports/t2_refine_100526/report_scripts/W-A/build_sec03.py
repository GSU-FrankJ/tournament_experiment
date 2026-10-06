"""Build SCRATCH/sec03.md and SCRATCH/sec03_ledger.csv from the evidence pack (read-only).

Every number in the text goes through N() / CI(), which formats it (4 significant digits unless
a format is given) and writes one ledger row (statement_id, section, text, value, item_id,
locator).  Nothing is typed by hand except identifiers (seeds, commits, dates) that are quoted
from a named source in the prose.
"""
import csv
import json
import math
import re
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/t2-refine-pack/"
         "reports/t2_refine_100526/evidence")
SCR = Path("/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-"
           "r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/scratchpad/"
           "report_parts")
SCR.mkdir(parents=True, exist_ok=True)

# ---- pack paths -------------------------------------------------------------------------
P = {
    "PL-01": "protocols/v2_T2_locked_v2_0.json",
    "PL-04": "results/v2_T2_locked/v2_0/protocol_diff_v1_1_to_v2_0.json",
    "PL-05": "results/v2_T2_locked/v2_0/continuation_check_v2_0.json",
    "PL-07": "results/v2_refine/continuation_check.json",
    "PL-09": "results/v2_T2_locked/v2_0/v2_0_pass_probability.json",
    "CF-01": "results/v2_T2_locked/confirmation_v2_0_analysis/verdict.json",
    "CF-02": "results/v2_T2_locked/confirmation_v2_0_analysis/pass_counts.csv",
    "CF-03": "results/v2_T2_locked/confirmation_v2_0_analysis/per_run.csv",
    "CF-04": "results/v2_T2_locked/confirmation_v2_0_analysis/s1_summary.csv",
    "CF-08": "results/v2_T2_locked/rehearsal_v2_0_checks.json",
    "CF-10": "results/v2_T2_locked/confirmation_analysis/per_run.csv",
    "CF-11": "results/v2_T2_locked/confirmation_analysis/s1_summary.csv",
    "CF-13": "results/v2_T2_locked/confirmation_v2_0/q50/seed30510/gates.json",
    "CF-14": "results/v2_T2_locked/confirmation_analysis/verdict.json",
    "CF-15": "results/v2_T2_locked/confirmation_analysis/pass_counts.csv",
    "CF-18": "results/v2_T2_locked/confirmation_v2_0_analysis_vs_v1_1/compare_distributions.csv",
    "CF-19": "results/v2_T2_locked/rehearsal_v2_0_analysis/per_run.csv",
    "CF-20": "results/v2_T2_locked/v2_0/seed_inventory.csv",
    "R1-02": "results/v2_refine/analysis/stage1_criterion.csv",
    "R1-03": "results/v2_refine/analysis/stage1_dispersion.csv",
    "R1-04": "results/v2_refine/analysis/stage1_adv_ratio.csv",
    "R1-09": "results/v2_refine/v11_reproduction_checks.json",
    "R1-18": "results/v2_refine/analysis/stage1_paired.csv",
    "R1-22": "results/v2_refine/analysis/stage1_cost.csv",
    "R1-24": "results/v2_refine/analysis/stage1_gate_counts.csv",
}


def J(i):
    return json.load(open(E / P[i]))


def C(i):
    return pd.read_csv(E / P[i])


# ---- ledger machinery -----------------------------------------------------------------------
LED = []
SEC = ["3"]


def reg(text, value, item, loc):
    LED.append(dict(statement_id="S3-%03d" % (len(LED) + 1), section=SEC[0], text=text,
                    value=value, item_id=item, locator=loc))
    return text


def N(value, item, loc, fmt=".4g", text=None):
    """Format value, register it, return the text."""
    if text is None:
        text = format(value, fmt)
    return reg(text, value, item, "%s :: %s" % (P.get(item.split(";")[0].split("+")[0].strip(), ""), loc))


def CI(lo, hi, item, loc_lo, loc_hi, fmt=".4g"):
    return "[%s, %s]" % (N(lo, item, loc_lo, fmt), N(hi, item, loc_hi, fmt))


def computed(text, value, items, how):
    """A value computed here from pack items."""
    return reg(text, value, items, "computed here: " + how)


def report_only(text, value, loc):
    """A number that only exists in a report's prose."""
    return reg(text, value, "RR-03", "(report only) " + loc)


# ---- load data ---------------------------------------------------------------------------------
pl01 = J("PL-01")
pl04 = J("PL-04")
pl05 = J("PL-05")
pl07 = J("PL-07")
pl09 = J("PL-09")
cf01 = J("CF-01")
cf08 = J("CF-08")
cf13 = J("CF-13")
cf14 = J("CF-14")
cf02 = C("CF-02")
cf03 = C("CF-03")
cf04 = C("CF-04")
cf10 = C("CF-10")
cf11 = C("CF-11")
cf15 = C("CF-15")
cf18 = C("CF-18")
cf19 = C("CF-19")
cf20 = C("CF-20")
r102 = C("R1-02")
r103 = C("R1-03")
r104 = C("R1-04")
r118 = C("R1-18")
r122 = C("R1-22")
r124 = C("R1-24")
r109 = J("R1-09")

out = []          # markdown lines
w = out.append


def row(cells):
    return "| " + " | ".join(cells) + " |"


# =================================================================================================
w("## 3. Protocol v1.1 → v2.0 and its fresh-seed confirmation")
w("")
SEC[0] = "3 intro"
w("Protocol v2.0 differs from the locked v1.1 in three ways: one pipeline change (section 3.1, "
  "with check (ii) of its table in section 3.2), one added gate, G-S (section 3.3), and a new "
  "confirmation seed block (section 3.4). The locked file is `protocols/v2_T2_locked_v2_0.json` "
  "[PL-01]. The PI decided every change after the R1 round and before any confirmation seed was "
  "run (`decided_by` of each v2.0 entry in the protocol change log [PL-01]). The confirmation on "
  "fresh seeds passed the pre-registered rule (section 3.5). Binding decision for this "
  "publication (PI publication prompt, D1): the four rounds are closed as reported, R2c selected "
  "no arm, so there is no v2.1, and the locked T=2 solver is v2.0 (`protocols/v2_T2_locked_v2_0.json`, "
  "tags `t2-v2-lock-v2.0`, `t2-v2-confirmation-v2.0`, `t2-v2-main-v2.0`).")
w("")
w("Symbols used in this section. `q` is the noise half-width (runs at q = 50 and q = 60). "
  "`DW` = w_H - w_L. `e1*` is the closed-form stage-1 equilibrium effort, used in evaluation only. "
  "`eta_2` is the stage-2 one-step deviation gap of the verifier, so `eta_2/DW` is the G-A "
  "accuracy metric. `ê1(0)` is the learned stage-1 effort at state 0 (Beta mean).")
w("")

# ---------------------------------------------------------------------------------------------
SEC[0] = "3.1"
w("### 3.1 The one pipeline change")
w("")
ct = pl01["pipeline"]["continuation_table"]
rule = ct["rule"]
w("In Phase B the stage-1 rows use a table value for the continuation instead of the sampled "
  "stage-2 outcome (`continuation_value_mode = %s` [PL-01 `pipeline.flags`, "
  "`pipeline.phase_B`]). Phase A and every other pipeline setting are unchanged in the protocol; "
  "the paths that differ are listed below." %
  pl01["pipeline"]["phase_B"]["continuation_value_mode"])
w("")
w("**Definition** [PL-01 `pipeline.continuation_table.applies_to`, `.definition`]:")
w("")
w("- Stage-1 rows only. Stage-1 return = r_1 + Ṽ2(y), with y = e_1 - e_1^opp (both efforts as "
  "executed, float64); stage-1 advantage = that return - V(s_1). Stage-2 rows, their critic "
  "targets and every random draw are unchanged (shocks are still drawn); the opponent's "
  "stage-1 action stays sampled.")
w("- Ṽ2(y) = E_z[g_2(y + z)], with g_2(d) = w_l + DW F_ξ(d + ê2(d) - ê2(-d)) - k ê2(d)². Here "
  "ê2 is the frozen stage-2 Beta mean (the float path of `continuation_action_mode=mean`), "
  "F_ξ is the function named so in the protocol, and z = ε_L - ε_O has the density "
  "(2q - |z|)/(4q²) on [-2q, 2q].")
w("- Built once at Phase-B entry from the frozen stage-2 snapshot (`utils.v2_continuation."
  "build_continuation_table`), single-threaded, with no RNG draw: the five training streams, "
  "the torch generator and the three process-global RNGs are compared between Phase-B entry "
  "and the first Phase-B update, and a moved training stream aborts the run [PL-01 "
  "`pipeline.continuation_table.built`, `.rng`]. Over the 40 confirmation runs the build took "
  "%s to %s s [CF-03]." % (
      computed("%.4g" % cf03.table_build_sec.min(), float(cf03.table_build_sec.min()), "CF-03",
               "min of table_build_sec over 40 runs"),
      computed("%.4g" % cf03.table_build_sec.max(), float(cf03.table_build_sec.max()), "CF-03",
               "max of table_build_sec over 40 runs")))
w("")
w("**Fixed integration rule** [PL-01 `pipeline.continuation_table.rule`]. The entry point "
  "refuses a protocol whose rule differs from the code's `LOCKED_TABLE_RULE`, and checks the "
  "table it builds against the rule at the first Phase-B update "
  "[PL-01 `.entry_point_refuses_other_values`].")
w("")
w(row(["parameter", "q = 50", "q = 60", "value / note"]))
w(row(["---", "---", "---", "---"]))
loc = "pipeline.continuation_table.rule"
w(row(["integration in z", "", "",
       "composite Gauss-Legendre against the triangular density; weights include the density "
       "and sum to 1 to rounding"]))
w(row(["panel width", "", "", N(rule["panel_width"], "PL-01", loc + ".panel_width", ".1f")]))
w(row(["nodes per panel", "", "", N(rule["nodes_per_panel"], "PL-01", loc + ".nodes_per_panel", "d")]))
w(row(["panel alignment", "", "",
       "aligned at z = 0 and z = ±2q: each half interval is split into ceil(2q / panel width) "
       "equal panels"]))
w(row(["panels", N(rule["n_panels"]["50"], "PL-01", loc + ".n_panels.50", "d"),
       N(rule["n_panels"]["60"], "PL-01", loc + ".n_panels.60", "d"), ""]))
w(row(["quadrature nodes", N(rule["n_nodes"]["50"], "PL-01", loc + ".n_nodes.50", "d"),
       N(rule["n_nodes"]["60"], "PL-01", loc + ".n_nodes.60", "d"), ""]))
w(row(["actor evaluation", "", "", "the frozen actor is evaluated directly at d = y + z; g_2 is never interpolated"]))
w(row(["y grid", "", "",
       "symmetric on [-100, 100] (e_range = e_max - e_min = %s), contains 0; step %s; %s nodes" % (
           N(100, "PL-01", loc + ".y_grid.range (e_range = e_max - e_min = 100)", "d"),
           N(rule["y_grid"]["step"], "PL-01", loc + ".y_grid.step", ".2f"),
           N(rule["y_grid"]["n_nodes"], "PL-01", loc + ".y_grid.n_nodes", "d"))]))
w(row(["dtype, lookup", "", "",
       "%s; linear interpolation; refuses y outside the grid by more than edge_tolerance × "
       "max(1, grid span), edge_tolerance = %s" % (
           rule["dtype"], N(rule["edge_tolerance"], "PL-01", loc + ".edge_tolerance", text="1e-9"))]))
w("")
w("Source: PL-01 (`protocols/v2_T2_locked_v2_0.json`, key `pipeline.continuation_table.rule`); the "
  "Markdown copy PL-02 states the same rule in its section 4. The entry point compares panel width, "
  "nodes per panel, y step, stage, dtype, edge tolerance and chunk size; the descriptive fields of "
  "the JSON rule (panel counts, wording) are not machine-checked [PL-02 section 4].")
w("")

# --- machine diff -------------------------------------------------------------------------------
GROUPS = OrderedDict([
    ("identity (version, name, date, schema, supersedes)",
     lambda p: p in ("/date", "/name", "/schema", "/version") or p.startswith("/supersedes/")),
    ("expected continuation",
     lambda p: p.startswith("/pipeline/") or p == "/training_relevant_state/compared"),
    ("gate G-S", lambda p: p.startswith("/gates/") or p == "/v1_1_outcome_reported"),
    ("confirmation design", lambda p: p.startswith("/confirmation/")),
    ("known issues", lambda p: p == "/known_issues"),
    ("change log and evidence", lambda p: p in ("/change_log", "/evidence/v2.0 changes")),
])
by_group = OrderedDict((g, []) for g in GROUPS)
for e in pl04:
    hit = [g for g, f in GROUPS.items() if f(e["path"])]
    assert len(hit) == 1, (e["path"], hit)
    by_group[hit[0]].append(e)
ops = Counter(e["op"] for e in pl04)
assert sum(len(v) for v in by_group.values()) == len(pl04)
assert not [e for e in pl04 if e["path"].startswith(("/gates/G-A", "/gates/G-F", "/gates/G-N", "/pipeline/phase_A", "/records", "/q_values"))]

w("**What the machine diff lists.** The diff from the v1.1 JSON to the v2.0 JSON has %s entries "
  "[PL-04]: %s `changed` and %s `added`, none removed. Nothing under `gates.G-A`, `gates.G-F`, "
  "`gates.G-N`, `pipeline.phase_A`, `records` or `q_values` appears in it (checked here). The grouping below is computed "
  "here from the paths; the round report groups the same 30 paths as 8 / 7 / 3 / 9 / 3 "
  "(identity; expected continuation; G-S; confirmation; change log with evidence and known "
  "issues) [RR-03 section 1.2]." % (
      computed(str(len(pl04)), len(pl04), "PL-04", "number of entries in the diff list"),
      computed(str(ops["changed"]), ops["changed"], "PL-04", "count of op == changed"),
      computed(str(ops["added"]), ops["added"], "PL-04", "count of op == added")))
assert ops["removed"] == 0
w("")
w(row(["group", "changed", "added", "paths"]))
w(row(["---", "---", "---", "---"]))
for g, es in by_group.items():
    c = Counter(e["op"] for e in es)
    paths = "; ".join("`%s`%s" % (e["path"], " (added)" if e["op"] == "added" else "") for e in es)
    w(row([g,
           computed(str(c["changed"]), c["changed"], "PL-04", "count op==changed in group '%s'" % g),
           computed(str(c["added"]), c["added"], "PL-04", "count op==added in group '%s'" % g),
           paths]))
w(row(["**total**", computed(str(ops["changed"]), ops["changed"], "PL-04", "total changed"),
       computed(str(ops["added"]), ops["added"], "PL-04", "total added"), ""]))
w("")
w("Source: PL-04 (`results/v2_T2_locked/v2_0/protocol_diff_v1_1_to_v2_0.json`, fields `op` and "
  "`path` of each entry); the group assignment is done by path prefix in "
  "`W-A/build_sec03.py`.")
w("")
w("Phase A is unchanged in outcome: in the v2.0 re-rehearsal the Phase-A state equals "
  "`rehearsal_v1_1` in %s of %s runs in every compared field [CF-08 `R1`]. The pipeline code "
  "itself changed at commit `32a8c21` (R1 flags; every new key at its default behaves as "
  "before); the unchanged v1.1 entry point reproduces `rehearsal_v1_1` in %s of %s runs on the "
  "changed code (C-R1) [PL-01 change log, v2.0 entry 2; R1-09]." % (
      N(cf08["R1"]["n_identical"], "CF-08", "R1.n_identical", "d"),
      N(cf08["R1"]["n"], "CF-08", "R1.n", "d"),
      N(r109["n_identical"], "R1-09", "n_identical", "d"), N(r109["n"], "R1-09", "n", "d")))
w("")
w("The full R1 evidence behind the change is in section 3.7 and section 4.6.")
w("")

# ---------------------------------------------------------------------------------------------
SEC[0] = "3.2"
w("### 3.2 Check (ii) of the continuation table")
w("")
w("The literal R1 criterion for check (ii) was not met in R1. v2.0 replaced it with three tests "
  "(ii-a), (ii-b), (ii-c), all met at both q; the literal criterion was replaced, not satisfied.")
w("")
fin = pl07["table_vs_verifier"]["final"]
dev = pl07["table_vs_verifier"]["development"]
conv = pl07["convergence"]
w("**The pre-registered criterion.** The table is compared with the verifier's stage-1 value "
  "Q1(0, e) + k e² at every node of the verifier's effort grid; the literal criterion is a "
  "maximum absolute difference of at most %s DW on the verifier's standard final tier (state "
  "step %s, %s Gauss-Legendre nodes per half interval) [RR-08 section 4 item 6; PL-06; PL-07]. "
  "The comparison uses the end-of-A states of the v1.1 rehearsal, seed %s, at both q, with the "
  "stage-1 opponent mean fixed at %s (q = 50) and %s (q = 60) [PL-05 `parents`, `check_ii`]; it "
  "is a check on these two states, not on every run." % (
      N(pl05["ii_b"]["q50"]["threshold_over_dw"], "PL-05", "ii_b.q50.threshold_over_dw", text="1e-6"),
      N(fin["q50"]["state_step"], "PL-07", "table_vs_verifier.final.q50.state_step", ".1f"),
      N(fin["q50"]["gl_half"], "PL-07", "table_vs_verifier.final.q50.gl_half", "d"),
      N(pl05["parents"]["q50"]["seed"], "PL-05", "parents.q50.seed", "d"),
      N(pl05["parents"]["q50"]["e1_hat_fixed_opponent"], "PL-05", "parents.q50.e1_hat_fixed_opponent", ".0f"),
      N(pl05["parents"]["q60"]["e1_hat_fixed_opponent"], "PL-05", "parents.q60.e1_hat_fixed_opponent", ".0f")))
w("")
w("**What R1 measured** (maxima over the effort grid, in units of DW):")
w("")
w("- Standard final tier: %s DW at q = 50 and %s DW at q = 60, so the literal criterion is "
  "**not met** [PL-07 `table_vs_verifier.final.q*.max_abs_diff_over_dw`, `meets_spec_tolerance` "
  "= false; R1-32]. Development tier: %s and %s DW [PL-07 `table_vs_verifier.development`]." % (
      N(fin["q50"]["max_abs_diff_over_dw"], "PL-07", "table_vs_verifier.final.q50.max_abs_diff_over_dw"),
      N(fin["q60"]["max_abs_diff_over_dw"], "PL-07", "table_vs_verifier.final.q60.max_abs_diff_over_dw"),
      N(dev["q50"]["max_abs_diff_over_dw"], "PL-07", "table_vs_verifier.development.q50.max_abs_diff_over_dw"),
      N(dev["q60"]["max_abs_diff_over_dw"], "PL-07", "table_vs_verifier.development.q60.max_abs_diff_over_dw")))
sweep50 = pl07["diagnosis"]["verifier_grid_sweep_max_abs_diff_over_dw"]["q50"]["state_step_0.25"]["gl_half_64"]
w("- Refined verifier (state step %s, %s Gauss-Legendre nodes): %s DW at q = 50 and %s DW at q = 60, "
  "within %s DW [PL-05 `ii_b`; the q = 50 value equals the R1 sweep entry %s, PL-07 "
  "`diagnosis.verifier_grid_sweep_max_abs_diff_over_dw`]." % (
      N(pl05["ii_b"]["q50"]["verifier"]["state_step"], "PL-05", "ii_b.q50.verifier.state_step", ".2f"),
      N(pl05["ii_b"]["q50"]["verifier"]["gl_half"], "PL-05", "ii_b.q50.verifier.gl_half", "d"),
      N(pl05["ii_b"]["q50"]["max_abs_diff_over_dw"], "PL-05", "ii_b.q50.max_abs_diff_over_dw"),
      N(pl05["ii_b"]["q60"]["max_abs_diff_over_dw"], "PL-05", "ii_b.q60.max_abs_diff_over_dw"),
      N(pl05["ii_b"]["q50"]["threshold_over_dw"], "PL-05", "ii_b.q50.threshold_over_dw", text="1e-6"),
      N(sweep50, "PL-07", "diagnosis.verifier_grid_sweep_max_abs_diff_over_dw.q50.state_step_0.25.gl_half_64")))
assert abs(sweep50 - pl05["ii_b"]["q50"]["max_abs_diff_over_dw"]) < 1e-15
w("- Table self-convergence on the parent actors: halving the panel width and doubling the nodes "
  "from the default rule changes the table by at most %s DW (q = 50) and %s DW (q = 60); from "
  "the coarser rule (panel width 2, 3 nodes) to the default the change is at most %s and %s DW "
  "[PL-07 `convergence.q50_parent`, `convergence.q60_parent`]." % (
      N(conv["q50_parent"][1]["max_abs_change_over_dw"], "PL-07", "convergence.q50_parent[1].max_abs_change_over_dw"),
      N(conv["q60_parent"][1]["max_abs_change_over_dw"], "PL-07", "convergence.q60_parent[1].max_abs_change_over_dw"),
      N(conv["q50_parent"][0]["max_abs_change_over_dw"], "PL-07", "convergence.q50_parent[0].max_abs_change_over_dw"),
      N(conv["q60_parent"][0]["max_abs_change_over_dw"], "PL-07", "convergence.q60_parent[0].max_abs_change_over_dw")))
w("- The verifier's stage-2 values equal the table's g_2 at the verifier's own nodes (maximum "
  "difference %s DW), and the standard-tier gap falls by about 4× per halving of the verifier's "
  "state step [PL-07 `table_vs_verifier.final.q50.stage2_values_vs_g_max_over_dw`; RR-08 "
  "section 4 item 6; PL-06]." % N(fin["q50"]["stage2_values_vs_g_max_over_dw"], "PL-07",
                                   "table_vs_verifier.final.q50.stage2_values_vs_g_max_over_dw"))
w("")
w("**PI decision in R1.** The PI was asked and decided to keep method 6 in the round and to record "
  "both results; the literal test stayed in the suite as a strict expected failure and no "
  "tolerance was loosened [RR-08 section 4 item 6 (`reports/v2/refine/01_preregistration.md`)].")
w("")
ii = pl05
w("**How v2.0 settled it** (decision D3 of the v2.0 round [PL-05 `decision`; PL-01 change log, "
  "v2.0 entry 3]). The literal criterion is replaced by three tests, recorded by "
  "`tests/test_v2_refine_continuation.py --write` [PL-05]:")
w("")
w(row(["test", "specification", "q = 50", "q = 60", "pass"]))
w(row(["---", "---", "---", "---", "---"]))
w(row(["literal R1 criterion (superseded)",
       "max \\|Ṽ2 - verifier\\| ≤ 1e-6 DW, standard final tier",
       N(fin["q50"]["max_abs_diff_over_dw"], "PL-07", "table_vs_verifier.final.q50.max_abs_diff_over_dw") + " DW",
       N(fin["q60"]["max_abs_diff_over_dw"], "PL-07", "table_vs_verifier.final.q60.max_abs_diff_over_dw") + " DW",
       "no (R1)"]))
w(row(["(ii-a) self-convergence",
       "panel width 1 → 0.5, nodes 6 → 12: max change ≤ %s DW" %
       N(ii["ii_a"]["q50"]["threshold_over_dw"], "PL-05", "ii_a.q50.threshold_over_dw", text="1e-8"),
       N(ii["ii_a"]["q50"]["max_abs_change_over_dw"], "PL-05", "ii_a.q50.max_abs_change_over_dw") + " DW",
       N(ii["ii_a"]["q60"]["max_abs_change_over_dw"], "PL-05", "ii_a.q60.max_abs_change_over_dw") + " DW",
       "yes" if ii["ii_a"]["q50"]["pass"] and ii["ii_a"]["q60"]["pass"] else "NO"]))
w(row(["(ii-b) refined-verifier agreement",
       "state step 0.25, 64 GL nodes: max over effort grid ≤ %s DW" %
       N(ii["ii_b"]["q50"]["threshold_over_dw"], "PL-05", "ii_b.q50.threshold_over_dw", text="1e-6"),
       N(ii["ii_b"]["q50"]["max_abs_diff_over_dw"], "PL-05", "ii_b.q50.max_abs_diff_over_dw") + " DW",
       N(ii["ii_b"]["q60"]["max_abs_diff_over_dw"], "PL-05", "ii_b.q60.max_abs_diff_over_dw") + " DW",
       "yes" if ii["ii_b"]["q50"]["pass"] and ii["ii_b"]["q60"]["pass"] else "NO"]))
fo50 = ii["ii_c"]["q50"]["fixed_opponent"]
fo60 = ii["ii_c"]["q60"]["fixed_opponent"]
w(row(["(ii-c) training-side sensitivity",
       "adding (finest rule - default rule) to the table shifts argmax_e[-k e² + Ṽ2(e - ê1(0))] "
       "(0.001 grid) by ≤ %s e1*" %
       N(ii["ii_c"]["q50"]["threshold_over_e1_star"], "PL-05", "ii_c.q50.threshold_over_e1_star", text="1e-3"),
       "shift %s effort units, abs. shift / e1* = %s" % (
           N(fo50["shift"], "PL-05", "ii_c.q50.fixed_opponent.shift", ".3f"),
           N(fo50["shift_over_e1_star"], "PL-05", "ii_c.q50.fixed_opponent.shift_over_e1_star")),
       "shift %s effort units, abs. shift / e1* = %s" % (
           N(fo60["shift"], "PL-05", "ii_c.q60.fixed_opponent.shift", ".0f"),
           N(fo60["shift_over_e1_star"], "PL-05", "ii_c.q60.fixed_opponent.shift_over_e1_star", ".0f")),
       "yes" if ii["ii_c"]["q50"]["pass"] and ii["ii_c"]["q60"]["pass"] else "NO"]))
w("")
w("Source: PL-05 (`results/v2_T2_locked/v2_0/continuation_check_v2_0.json`, keys `ii_a`, `ii_b`, "
  "`ii_c`, field `max_abs_change_over_dw`, `max_abs_diff_over_dw`, `fixed_opponent.shift`, "
  "`fixed_opponent.shift_over_e1_star`, `pass`); first row PL-07 (`results/v2_refine/"
  "continuation_check.json`, `table_vs_verifier.final`). (ii-c) is also evaluated with the "
  "stage-1 opponent at e1*; the shift is %s at both q and the test passes there too "
  "[PL-05 `ii_c.q*.opponent_at_e1_star`]. `all_pass` = %s [PL-05]." % (
      N(ii["ii_c"]["q50"]["opponent_at_e1_star"]["shift"], "PL-05", "ii_c.q50.opponent_at_e1_star.shift", ".0f"),
      str(ii["all_pass"]).lower()))
assert ii["ii_c"]["q60"]["opponent_at_e1_star"]["shift"] == 0 and ii["ii_c"]["q50"]["opponent_at_e1_star"]["pass"]
w("")
vn = ii["verifier_numerics"]
w("**Verifier numerics item (reported, not gated).** On the standard final tier the verifier's "
  "stage-1 value differs from the table by %s DW (q = 50) and %s DW (q = 60). If that gap were a "
  "table error it would move the stage-1 optimum by %s and %s e1*, which is above the %s e1* "
  "limit that (ii-c) applies to the table's own convergence profile [PL-06; PL-05 "
  "`verifier_numerics`]. The report reads the gap as lying on the verifier's side (its "
  "state-grid interpolation of V2, step 2 on the final tier): it bounds the precision of the "
  "verifier's own ẽ1 on that tier and is not evidence about the table [PL-06 'Reading']. It is "
  "listed as a known issue of the protocol [PL-04 `/known_issues`]." % (
      N(vn["q50"]["final"]["max_abs_gap_over_dw"], "PL-05", "verifier_numerics.q50.final.max_abs_gap_over_dw"),
      N(vn["q60"]["final"]["max_abs_gap_over_dw"], "PL-05", "verifier_numerics.q60.final.max_abs_gap_over_dw"),
      N(vn["q50"]["final"]["implied_shift_over_e1_star"], "PL-05", "verifier_numerics.q50.final.implied_shift_over_e1_star"),
      N(vn["q60"]["final"]["implied_shift_over_e1_star"], "PL-05", "verifier_numerics.q60.final.implied_shift_over_e1_star"),
      N(ii["ii_c"]["q50"]["threshold_over_e1_star"], "PL-05", "ii_c.q50.threshold_over_e1_star", text="1e-3")))
w("")
w(row(["verifier tier", "q", "max \\|δ\\| / DW", "implied shift of the stage-1 optimum", "shift / e1*"]))
w(row(["---", "---", "---", "---", "---"]))
for tier, tl in (("final", "final (state step 2, 32 GL nodes per half interval)"),
                 ("development", "development (state step 4, 16 GL nodes per half interval)")):
    for q in ("50", "60"):
        d = vn["q" + q][tier]
        lc = "verifier_numerics.q%s.%s." % (q, tier)
        w(row([tl, q, N(d["max_abs_gap_over_dw"], "PL-05", lc + "max_abs_gap_over_dw"),
               N(d["implied_shift"], "PL-05", lc + "implied_shift", ".3f"),
               N(d["implied_shift_over_e1_star"], "PL-05", lc + "implied_shift_over_e1_star")]))
w("")
w("Source: PL-05 (`continuation_check_v2_0.json`, key `verifier_numerics`, fields "
  "`max_abs_gap_over_dw`, `implied_shift`, `implied_shift_over_e1_star`; every entry has `gated` = false).")
w("")
w("Limitation stated in the round report: (ii-a) to (ii-c) refer to the table's node values; "
  "none of them covers the interpolation error of the step-0.05 linear lookup between table "
  "nodes (pre-lock review finding T7, not changed) [RR-03 section 2 and section 6, item 3]. "
  "The module docstring of `utils/v2_continuation.py` still calls check (ii) open; the "
  "protocol text and PL-05 supersede it [PL-04 `/known_issues`].")
w("")

# ---------------------------------------------------------------------------------------------
SEC[0] = "3.3"
w("### 3.3 Gate G-S and its admission rule")
w("")
w("G-S requires the stage-1 last iterate to be within 5% of the closed form; it was admitted "
  "on development-seed evidence before any confirmation seed was run, and the pre-registered "
  "safety condition on the re-rehearsal was met at both q.")
w("")
gs = pl01["gates"]["G-S"]["all_must_hold"][0]
w("**Definition** [PL-01 `gates.G-S`, `gates.comparison`, `gates.run_outcome`]. At the end of "
  "Phase B, on the final tier, the candidate is the stage-1 last iterate (Beta mean of the live "
  "actor at t = 1, d = 0, global update u2200). G-S holds iff |ê1(0) - e1*(0)| / e1*(0) ≤ %s "
  "(the comparison is inclusive, in float64; e1* enters the evaluation only). G-S is the v1.1 "
  "secondary criterion S1 with its threshold set to 0.05, the 5%% stage-1 target of the 0929 plan "
  "[PL-01 `gates.G-S`; PL-04, old `/change_log`, first entry]. Run pass = G-A and "
  "G-F and G-N and G-S. S1 at %s (the v1.1 secondary criterion), the v1.1 outcome (G-A, G-F, "
  "G-N) and the v1.0 outcome are reported only and decide nothing." % (
      N(gs["threshold"], "PL-01", "gates.G-S.all_must_hold[0].threshold", ".2f"),
      N(pl01["secondary"]["S1"]["criterion"]["threshold"],
        "PL-01", "secondary.S1.criterion.threshold", ".2f")))
w("")
w("**Threshold evidence before any confirmation seed** [PL-09; the same numbers are quoted in "
  "the protocol change log, v2.0 entry 4, PL-01]. The record takes arm `B_expcont` (R1 stage-1 "
  "wave) on the development seeds 10501-10510, per q (n = 10), and fits the signed stage-1 "
  "error as N(mean, SD) with SD at ddof = 1. G-A, G-F and G-N are treated as passing, a run "
  "passes G-S iff |error| ≤ 0.05, and the probability is P(Binomial(20, p_q) ≥ 18) "
  "[PL-01 `confirmation.rehearsal.safety_condition.normal_model`].")
w("")
w(row(["q", "runs within 0.05", "max \\|error\\|", "mean signed error", "SD (ddof = 1)",
       "per-run pass probability p_q", "P(≥ 18 of 20 pass G-S)"]))
w(row(["---"] * 7))
for q in ("50", "60"):
    d = pl09["per_q"][q]
    lc = "per_q.%s." % q
    w(row([q, "%s / %s" % (N(d["n_within_threshold"], "PL-09", lc + "n_within_threshold", "d"), N(d["n"], "PL-09", lc + "n", "d")),
           N(d["max_abs_err"], "PL-09", lc + "max_abs_err"),
           N(d["mean_signed"], "PL-09", lc + "mean_signed"),
           N(d["sd_ddof1"], "PL-09", lc + "sd_ddof1"),
           N(d["per_run_pass_prob_normal_ddof1"], "PL-09", lc + "per_run_pass_prob_normal_ddof1"),
           N(d["P_ge18_of_20_normal_ddof1"], "PL-09", lc + "P_ge18_of_20_normal_ddof1")]))
w("")
w("Source: PL-09 (`results/v2_T2_locked/v2_0/v2_0_pass_probability.json`, key `per_q`, fields "
  "`n_within_threshold`, `n`, `max_abs_err`, `mean_signed`, `sd_ddof1`, "
  "`per_run_pass_prob_normal_ddof1`, `P_ge18_of_20_normal_ddof1`).")
w("")
prod = pl09["per_q"]["50"]["P_ge18_of_20_normal_ddof1"] * pl09["per_q"]["60"]["P_ge18_of_20_normal_ddof1"]
assert abs(prod - pl09["P_both_normal_ddof1"]) < 1e-12
w("The record also gives %s for both q together, which equals the product of the two per-q "
  "probabilities (checked here) [PL-09 `P_both_normal_ddof1`]." % N(pl09["P_both_normal_ddof1"], "PL-09", "P_both_normal_ddof1"))
w("")
sc = pl01["confirmation"]["rehearsal"]["safety_condition"]
r3b = cf08["R3"]["R3b"]
gsr = cf19[cf19.status == "completed"].groupby("q")["G-S_pass"].sum()
w("**Pre-registered safety condition on the re-rehearsal** [PL-01 `confirmation.rehearsal."
  "safety_condition`]: at least %s of 20 re-rehearsal runs pass G-S, and the recomputed "
  "normal-model probability is at least %s at each q. **Result:** %s of the %s re-rehearsal "
  "runs (development seeds, both q) pass G-S (%s per q, computed from `G-S_pass` of CF-19); "
  "the recomputed probabilities are %s (q = 50) and %s (q = 60); check R3b is `pass` = %s "
  "[CF-08 `R3.R3b`; CF-19]. The probabilities equal those of the pre-confirmation record, as "
  "they must, because the re-rehearsal reproduces the pilot arm bit for bit (check R2, "
  "section 3.4)." % (
      N(sc["G-S_pass_at_least_of_20"], "PL-01", "confirmation.rehearsal.safety_condition.G-S_pass_at_least_of_20", "d"),
      N(sc["normal_model_probability_at_least_per_q"], "PL-01", "confirmation.rehearsal.safety_condition.normal_model_probability_at_least_per_q", ".2f"),
      N(r3b["pooled_n_G-S_pass"], "CF-08", "R3.R3b.pooled_n_G-S_pass", "d"),
      computed("20", int(len(cf19)), "CF-19", "number of rows of per_run.csv"),
      computed("%d and %d" % (gsr[50], gsr[60]), "%d,%d" % (gsr[50], gsr[60]), "CF-19", "sum of G-S_pass by q"),
      N(r3b["per_q"]["50"]["P_ge_k_of_n"], "CF-08", "R3.R3b.per_q.50.P_ge_k_of_n"),
      N(r3b["per_q"]["60"]["P_ge_k_of_n"], "CF-08", "R3.R3b.per_q.60.P_ge_k_of_n"),
      str(r3b["pass"]).lower()))
assert len(cf19) == 20 and gsr[50] == 10 and gsr[60] == 10
w("")
w("Observation (not a recorded judgement): the probability record rests on the development "
  "seeds, which are also the seeds on which `B_expcont` was selected in R1 (section 3.7). It "
  "is therefore not an independent estimate; the fresh confirmation block is the independent "
  "test of the threshold.")
w("")
w("For context, the stage-1 criterion at 0.10 had been moved from G-F to the secondary criterion "
  "S1 in v1.1, after all 3 v1.0 rehearsal failures were on it [PL-04, old `/change_log`, v1.1 "
  "entry 1].")
w("")

# ---------------------------------------------------------------------------------------------
SEC[0] = "3.4"
w("### 3.4 Seed block and re-rehearsal")
w("")
w("The confirmation block is the 20 seeds 30501-30520, used for both q (40 runs), disjoint from "
  "every recorded seed value, and the re-rehearsal checks R1-R7 held, R7 only through the "
  "owner decision D-R7.")
w("")
cfm = pl01["confirmation"]
w("**Design** [PL-01 `confirmation`]. Seed block %s-%s for both q (status: \"%s\"); v1.1 had used "
  "%s-%s [PL-04 `/confirmation/seed_block`]. Pass rule: for each q at least %s of %s runs pass "
  "(G-A and G-F and G-N and G-S); the confirmation passes iff both q pass; each pass rate carries an "
  "exact Clopper-Pearson 95%% CI. Every run is reported, failures included; nothing changes "
  "after the lock. Crash rule: an infrastructure kill is re-run once; any other nonzero exit "
  "(pipeline exception, global-RNG violation) counts as a failed run. Analysis script: "
  "`%s`." % (
      N(cfm["seed_block"][0], "PL-01", "confirmation.seed_block[0]", "d"),
      N(cfm["seed_block"][-1], "PL-01", "confirmation.seed_block[-1]", "d"),
      cfm["seed_block_status"],
      N(pl04[[e["path"] for e in pl04].index("/confirmation/seed_block")]["old"][0], "PL-04", "/confirmation/seed_block old[0]", "d"),
      N(pl04[[e["path"] for e in pl04].index("/confirmation/seed_block")]["old"][-1], "PL-04", "/confirmation/seed_block old[-1]", "d"),
      N(18, "PL-01", "confirmation.pass_rule ('at least 18 of 20')", "d"),
      N(cfm["n_seeds"], "PL-01", "confirmation.n_seeds", "d"),
      cfm["analysis_script"]))
w("")
blk = cf20[(cf20.seed >= 30501) & (cf20.seed <= 30520)]
w("**Disjointness.** The seed inventory scans every JSON/CSV seed record of the tournament tree "
  "[PL-01 `confirmation.disjointness_check`]. The scan run at the lock, with the two files that "
  "declare the block excluded, lists %s distinct seed values and %s of them in %s-%s (computed "
  "here from the `seed` column of CF-20). The pre-lock scan, taken before the block appeared "
  "in any JSON/CSV seed record, also found 0 collisions among %s values [PL-01 "
  "`confirmation.disjointness_check`]. Without the exclusion the scan counts exactly the 20 "
  "seeds that the protocol itself declares and nothing else (stock scan: 20 collisions, single "
  "source `protocols/v2_T2_locked_v2_0.json`, key `seed_block`) [RR-03 section 6, item 4 "
  "(report only; the stock scan file is not in the pack)]." % (
      computed(str(len(cf20)), int(len(cf20)), "CF-20", "number of rows (distinct seeds) in seed_inventory.csv"),
      computed(str(len(blk)), int(len(blk)), "CF-20", "rows with 30501 <= seed <= 30520"),
      "30501", "30520",
      N(305, "PL-01", "confirmation.disjointness_check ('0 collisions among 305 recorded seed values')", "d")))
assert len(cf20) == 305 and len(blk) == 0
w("")
w("**Re-rehearsal** [PL-04 `/confirmation/rehearsal`; RR-03 section 3]. 20 runs, q in {50, 60} × "
  "development seeds 10501-10510, through the v2.0 entry point from scratch, before the "
  "confirmation. Checks R1-R7 and their outcomes as recorded in `rehearsal_v2_0_checks.json` "
  "[CF-08] (check descriptions from RR-03 section 3):")
w("")
w(row(["check", "what it tests", "outcome", "detail from CF-08"]))
w(row(["---"] * 4))
w(row(["R1", "Phase A equals `rehearsal_v1_1`", "pass" if cf08["R1"]["pass"] else "FAIL",
       "%s / %s identical in every compared field" % (N(cf08["R1"]["n_identical"], "CF-08", "R1.n_identical", "d"), N(cf08["R1"]["n"], "CF-08", "R1.n", "d"))]))
w(row(["R2", "Phase B equals the R1 pilot arm `B_expcont`, including the table", "pass" if cf08["R2"]["pass"] else "FAIL",
       "%s / %s identical" % (N(cf08["R2"]["n_identical"], "CF-08", "R2.n_identical", "d"), N(cf08["R2"]["n"], "CF-08", "R2.n", "d"))]))
w(row(["R3", "G-A, G-F, G-N, G-S on every run", "pass" if cf08["R3"]["pass"] else "FAIL",
       "%s / %s runs pass" % (N(cf08["R3"]["n_pass"], "CF-08", "R3.n_pass", "d"), N(cf08["R3"]["n"], "CF-08", "R3.n", "d"))]))
w(row(["R3b", "G-S safety condition (section 3.3)", "pass" if r3b["pass"] else "FAIL",
       "pooled %s G-S passes (needed %s); P = %s (q = 50), %s (q = 60)" % (
           N(r3b["pooled_n_G-S_pass"], "CF-08", "R3.R3b.pooled_n_G-S_pass", "d"),
           N(r3b["needed"], "CF-08", "R3.R3b.needed", "d"),
           N(r3b["per_q"]["50"]["P_ge_k_of_n"], "CF-08", "R3.R3b.per_q.50.P_ge_k_of_n"),
           N(r3b["per_q"]["60"]["P_ge_k_of_n"], "CF-08", "R3.R3b.per_q.60.P_ge_k_of_n"))]))
w(row(["R4", "global RNGs, return codes", "pass" if cf08["R4"]["pass"] else "FAIL",
       "%s / %s ok" % (N(cf08["R4"]["n_ok"], "CF-08", "R4.n_ok", "d"), N(cf08["R4"]["n"], "CF-08", "R4.n", "d"))]))
w(row(["R5", "manifests (v2.0 hash and version, launch commit, clean tree, table SHA-256)",
       "pass" if cf08["R5"]["pass"] else "FAIL",
       "%s / %s ok" % (N(cf08["R5"]["n_ok"], "CF-08", "R5.n_ok", "d"), N(cf08["R5"]["n"], "CF-08", "R5.n", "d"))]))
w(row(["R6", "analysis script agrees with `gates.json`", "pass" if cf08["R6"]["pass"] else "FAIL",
       "%s / %s agree, max abs value difference %s" % (
           N(cf08["R6"]["n_agree"], "CF-08", "R6.n_agree", "d"), N(cf08["R6"]["n"], "CF-08", "R6.n", "d"),
           N(cf08["R6"]["max_abs_value_diff"], "CF-08", "R6.max_abs_value_diff", ".0f"))]))
m = re.match(r"(\d+) failed, (\d+) passed, (\d+) xfailed", cf08["R7"]["pytest_summary"])
w(row(["R7", "full test suite and C7 (the v2 runner at default flags equals the pre-refactor reference bit for bit)",
       "**tool: %s; accepted as met (D-R7)**" % str(cf08["R7"]["pass"]).lower(),
       "pytest: %s failed (the known `test_registry_canonicalization`), %s passed, %s xfailed; `c7_tool` = %s; "
       "`c7_canonical_reference` = %s" % (
           N(int(m.group(1)), "CF-08", "R7.pytest_summary", "d"), N(int(m.group(2)), "CF-08", "R7.pytest_passed / pytest_summary", "d"),
           N(int(m.group(3)), "CF-08", "R7.pytest_summary", "d"), str(cf08["R7"]["c7_tool"]).lower(),
           cf08["R7"]["c7_canonical_reference"])]))
w("")
w("Source: CF-08 (`results/v2_T2_locked/rehearsal_v2_0_checks.json`, keys `R1`-`R7`, `R3.R3b`); "
  "`ALL_PASS` = %s as the tool wrote it, from R7 alone." % str(cf08["ALL_PASS"]).lower())
assert cf08["R7"]["pytest_known_failure_only"] is True and cf08["R7"]["pytest_passed"] == int(m.group(2))
w("")
w("**D-R7.** The tool's own R7 returned false. Pytest was fine (only the known registry failure). "
  "The C7 comparison crashed with `FileNotFoundError` because the reference run's "
  "`checkpoint.pt` (a gitignored `*.pt` file) is not in this worktree; the six other compared "
  "files had compared with 0 differences [CF-08 `R7.c7_tool_error`]. Against the canonical "
  "reference, C7 ends IDENTICAL including `checkpoint.pt` (0 differing tensors), for two "
  "candidates at the launch commit `f2d616c` [CF-09; RR-03 section 3]. The owner decision D-R7 "
  "(2026-10-04) accepted R7 as met on the canonical-reference comparison: the literal failure "
  "is a reference-path defect of the checks tool, not of the pipeline (precedent: Check 1 of "
  "the v1.0 lock round); the tool was not edited before the confirmation [CF-08 "
  "`R7.owner_decision`]. `R7.pass` and `ALL_PASS` stay false in the record. The pre-registered "
  "launch condition was \"R1-R7 all pass\" [PL-01 `confirmation.status`]; it was therefore "
  "met through D-R7, not as the tool computed it. After the confirmation the tool was fixed "
  "(commit `3fedaa2`, reference path only) and R7 re-run: pass = true [RR-03 section 6, "
  "deviation 1 (report only; the record is not in the pack)]. The deviation is listed in section 7.")
w("")
w("**Timeline** (all from RR-03 section 1.5, report only): v2.0 lock commit `1d6d4d0` at "
  "2026-10-03T23:35:28+00:00; re-rehearsal launched 2026-10-03T23:37:12Z at `f2d616c`; "
  "confirmation launched 2026-10-04T04:24:47Z at `d2e377d` (a results-only commit on top of the "
  "LOCK record) and finished 2026-10-04T04:37:16Z. All 40 manifests show the v2.0 protocol hash "
  "`6ca0a6f9...`, commit `d2e377d` and `clean_tree: true` [CF-01 `n_clean_tree` = %s, "
  "`commits`; CF-13 `protocol_sha256`]." % N(cf01["n_clean_tree"], "CF-01", "n_clean_tree", "d"))
w("")

# ---------------------------------------------------------------------------------------------
SEC[0] = "3.5"
w("### 3.5 Confirmation verdict")
w("")
w("**The confirmation passed under the pre-registered rule.** CF-01 records `overall` = \"%s\" "
  "for the rule \"%s\" (per q: q = 50 %s, q = 60 %s) [CF-01]." % (
      cf01["overall"], cf01["rule"], str(cf01["per_q"]["50"]).lower(), str(cf01["per_q"]["60"]).lower()))
w("")
w(row(["q", "runs", "passes (G-A ∧ G-F ∧ G-N ∧ G-S)", "pass rate", "exact 95% CI", "G-A", "G-F", "G-N", "G-S passes"]))
w(row(["---"] * 9))
for _, r in cf02.iterrows():
    q = int(r.q)
    lc = "row q=%d, column " % q
    w(row([str(q), N(int(r.n_completed), "CF-02", lc + "n_completed", "d"),
           "**%s / %s**" % (N(int(r.n_pass), "CF-02", lc + "n_pass", "d"), N(int(r.n_expected), "CF-02", lc + "n_expected", "d")),
           N(r.pass_rate, "CF-02", lc + "pass_rate", ".4f"),
           "[%s, %s]" % (N(r.cp95_lo, "CF-02", lc + "cp95_lo", ".4f"), N(r.cp95_hi, "CF-02", lc + "cp95_hi", ".4f")),
           N(int(r["n_G-A_pass"]), "CF-02", lc + "n_G-A_pass", "d"),
           N(int(r["n_G-F_pass"]), "CF-02", lc + "n_G-F_pass", "d"),
           N(int(r["n_G-N_pass"]), "CF-02", lc + "n_G-N_pass", "d"),
           "%s (CF-02) / %s (CF-04)" % (
               N(int(r["n_G-S_pass"]), "CF-02", lc + "n_G-S_pass", "d"),
               N(int(cf04[(cf04.q == q) & (cf04.criterion == "G-S")].n_pass.iloc[0]), "CF-04", "row q=%d criterion=G-S, column n_pass" % q, "d"))]))
w("")
w("Source: CF-02 (`results/v2_T2_locked/confirmation_v2_0_analysis/pass_counts.csv`, columns "
  "`n_completed`, `n_pass`, `n_expected`, `pass_rate`, `cp95_lo`, `cp95_hi`, `n_G-A_pass`, "
  "`n_G-F_pass`, `n_G-N_pass`, `n_G-S_pass`); CF-04 (`s1_summary.csv`, rows `criterion` = G-S, "
  "column `n_pass`). Rule: >= 18 of 20 at each q, both q.")
w("")
r50 = cf02[cf02.q == 50].iloc[0]
r60 = cf02[cf02.q == 60].iloc[0]
w("All %s runs completed with exit code 0, with %s global-RNG violations, %s exceptions and %s "
  "missing or incomplete runs; the analysis script's recomputed values agree with every run's "
  "`gates.json` in %s of %s runs [CF-01]. No run was re-run. Criteria that are reported and "
  "decide nothing [CF-02]: S1 at 0.10 passes in %s / 20 (q = 50) and %s / 20 (q = 60); the v1.1 "
  "outcome (G-A, G-F, G-N) in %s / 20 and %s / 20; the v1.0 outcome in %s / 20 and %s / 20. With "
  "n = 20 per q the exact intervals are wide." % (
      N(cf01["n_completed"], "CF-01", "n_completed", "d"), N(cf01["n_global_rng_violation"], "CF-01", "n_global_rng_violation", "d"),
      N(cf01["n_failed_exception"], "CF-01", "n_failed_exception", "d"),
      N(cf01["n_missing"] + cf01["n_incomplete"], "CF-01", "n_missing + n_incomplete", "d"),
      N(cf01["n_all_agree"], "CF-01", "n_all_agree", "d"), N(cf01["n_agreement_rows"], "CF-01", "n_agreement_rows", "d"),
      N(int(r50["n_S1_0.10_pass"]), "CF-02", "row q=50, n_S1_0.10_pass", "d"), N(int(r60["n_S1_0.10_pass"]), "CF-02", "row q=60, n_S1_0.10_pass", "d"),
      N(int(r50["n_v1_1_run_pass"]), "CF-02", "row q=50, n_v1_1_run_pass", "d"), N(int(r60["n_v1_1_run_pass"]), "CF-02", "row q=60, n_v1_1_run_pass", "d"),
      N(int(r50["n_v1_0_run_pass"]), "CF-02", "row q=50, n_v1_0_run_pass", "d"), N(int(r60["n_v1_0_run_pass"]), "CF-02", "row q=60, n_v1_0_run_pass", "d")))
w("")
fr = cf03[(cf03.q == 50) & (cf03.seed == 30510)].iloc[0]
assert (cf03.run_pass == False).sum() == 1 and fr.outcome == cf13["outcome"]
ga = {c["metric"]: c for c in cf13["G-A"]["criteria"]}
assert abs(fr.eta_final - ga["eta_T_over_dw"]["value"]) < 1e-15
pk = cf13["reported"]["end_of_A"]["final"]["stage2_peak_rel_err_signed"]
w("**The one failed run: q = 50, seed 30510.** Its `outcome` is `%s` [CF-03; CF-13]. The failing gate is "
  "G-A, through its eta_2 part: eta_2/DW = %s against the threshold %s (`G-A_eta_pass` = %s); "
  "the other two G-A parts pass (RMSE_pos/e2*(0) = %s against %s; tail mean/e2*(0) = %s against "
  "%s). G-F, G-N and G-S pass (G-S value %s against %s) [CF-13 `G-A.criteria`, `G-F`, `G-N`, "
  "`G-S`; CF-03 row q = 50, seed 30510]. The gates.json records the v1.1 outcome and the v1.0 "
  "outcome of this run as failed as well (G-A is a component of both) [CF-13 `v1_1_outcome`, "
  "`v1_0_outcome`]. The signed stage-2 peak error at the end of Phase A is %s [CF-13 "
  "`reported.end_of_A.final.stage2_peak_rel_err_signed`]." % (
      cf13["outcome"],
      N(ga["eta_T_over_dw"]["value"], "CF-13", "G-A.criteria[eta_T_over_dw].value"),
      N(ga["eta_T_over_dw"]["threshold"], "CF-13", "G-A.criteria[eta_T_over_dw].threshold", ".3f"),
      str(ga["eta_T_over_dw"]["pass"]).lower(),
      N(ga["stage2_rmse_pos_over_g2_0"]["value"], "CF-13", "G-A.criteria[stage2_rmse_pos_over_g2_0].value"),
      N(ga["stage2_rmse_pos_over_g2_0"]["threshold"], "CF-13", "G-A.criteria[stage2_rmse_pos_over_g2_0].threshold", ".2f"),
      N(ga["stage2_tail_mean_over_g2_0"]["value"], "CF-13", "G-A.criteria[stage2_tail_mean_over_g2_0].value"),
      N(ga["stage2_tail_mean_over_g2_0"]["threshold"], "CF-13", "G-A.criteria[stage2_tail_mean_over_g2_0].threshold", ".2f"),
      N(cf13["G-S"]["criteria"][0]["value"], "CF-13", "G-S.criteria[0].value"),
      N(cf13["G-S"]["criteria"][0]["threshold"], "CF-13", "G-S.criteria[0].threshold", ".2f"),
      N(pk, "CF-13", "reported.end_of_A.final.stage2_peak_rel_err_signed")))
w("")
w("What the round reports say about its nature: a stage-2 failure in Phase A (the gate G-A is "
  "evaluated at the end of Phase A), which v2.0 leaves unchanged; the run's Phase B was still "
  "executed and reported, and the run counts as failed; it was not re-run, because only an "
  "infrastructure kill is re-run [RR-03 Verdict section and section 6 item 7; PL-01 `gates.run_outcome`]. "
  "The later seed-30510 diagnostic is in section 6.")
w("")

# ---------------------------------------------------------------------------------------------
SEC[0] = "3.6"
w("### 3.6 Stage-1 error before and after")
w("")


def block(df, q):
    d = df[(df.q == q) & (df.status == "completed")]
    assert np.allclose(d.s1.values, np.abs(d.stage1_rel_err_signed.values), atol=1e-15)
    return d


stat = {}
for tag, df in (("v1.1", cf10), ("v2.0", cf03)):
    for q in (50, 60):
        d = block(df, q)
        stat[(tag, q)] = dict(n=len(d), med=float(np.median(d.s1)), mx=float(d.s1.max()),
                              w05=int((d.s1 <= 0.05).sum()), w10=int((d.s1 <= 0.10).sum()),
                              sd_check=float(np.std(d.stage1_rel_err_signed, ddof=1)))
# cross-check against CF-18 (median/max of s1 are tabulated there)
for q in (50, 60):
    c = cf18[(cf18.q == q) & (cf18.metric == "s1")].iloc[0]
    assert abs(stat[("v2.0", q)]["med"] - c.median_root) < 1e-15 and abs(stat[("v1.1", q)]["med"] - c.median_compare_root) < 1e-15
    assert abs(stat[("v2.0", q)]["mx"] - c.max_root) < 1e-15 and abs(stat[("v1.1", q)]["mx"] - c.max_compare_root) < 1e-15
for tag, s1t in (("v1.1", cf11), ("v2.0", cf04)):
    for q in (50, 60):
        row_ = s1t[s1t.q == q].iloc[0]
        assert abs(row_.sd_signed - stat[(tag, q)]["sd_check"]) < 1e-12
        stat[(tag, q)]["sd"] = float(row_.sd_signed)

w("On fresh seeds, the stage-1 error is smaller and less spread under v2.0 than under v1.1: the "
  "median |error| is %s (q = 50) and %s (q = 60) under v2.0 against %s and %s under v1.1, and the "
  "largest v2.0 error is %s (q = 50) and %s (q = 60). The two blocks use different seeds (v1.1 "
  "20501-20520, v2.0 30501-30520), so they are unpaired and no test is made." % (
      computed("%.4g" % stat[("v2.0", 50)]["med"], stat[("v2.0", 50)]["med"], "CF-03", "median of s1, q=50, completed runs"),
      computed("%.4g" % stat[("v2.0", 60)]["med"], stat[("v2.0", 60)]["med"], "CF-03", "median of s1, q=60, completed runs"),
      computed("%.4g" % stat[("v1.1", 50)]["med"], stat[("v1.1", 50)]["med"], "CF-10", "median of s1, q=50, completed runs"),
      computed("%.4g" % stat[("v1.1", 60)]["med"], stat[("v1.1", 60)]["med"], "CF-10", "median of s1, q=60, completed runs"),
      computed("%.4g" % stat[("v2.0", 50)]["mx"], stat[("v2.0", 50)]["mx"], "CF-03", "max of s1, q=50"),
      computed("%.4g" % stat[("v2.0", 60)]["mx"], stat[("v2.0", 60)]["mx"], "CF-03", "max of s1, q=60")))
w("")
w("`|error|` = |ê1(0) - e1*| / e1* of the final-tier end-of-B last iterate; the signed error "
  "keeps the sign. Figure: `figures/FG-01_stage1_error_v1_1_vs_v2_0.png` (per-run |error| of "
  "both protocols, with the 0.05 and 0.10 lines).")
w("")
w(row(["q", "protocol", "seeds", "n", "median \\|error\\|", "max \\|error\\|", "SD of signed error",
       "within 0.05", "within 0.10", "mean signed error [bootstrap 95% CI]"]))
w(row(["---"] * 10))
for q in (50, 60):
    for tag, src, srcs, sd_it, sd_t in (("v1.1", "CF-10", "CF-11", "CF-11", cf11), ("v2.0", "CF-03", "CF-04", "CF-04", cf04)):
        s = stat[(tag, q)]
        t = sd_t[sd_t.q == q].iloc[0]
        lc = "row q=%d, column " % q
        if tag == "v2.0":
            t = sd_t[(sd_t.q == q) & (sd_t.criterion == "G-S")].iloc[0]
            lc = "row q=%d criterion=G-S, column " % q
        w(row([str(q), tag, "20501-20520" if tag == "v1.1" else "30501-30520",
               computed(str(s["n"]), s["n"], src, "count of completed runs, q=%d" % q),
               computed("%.4g" % s["med"], s["med"], src, "median of s1, q=%d" % q),
               computed("%.4g" % s["mx"], s["mx"], src, "max of s1, q=%d" % q),
               N(s["sd"], sd_it, lc + "sd_signed"),
               computed("%d / %d" % (s["w05"], s["n"]), s["w05"], src, "count s1 <= 0.05, q=%d" % q),
               computed("%d / %d" % (s["w10"], s["n"]), s["w10"], src, "count s1 <= 0.10, q=%d" % q),
               "%s %s" % (N(t.mean_signed, sd_it, lc + "mean_signed"),
                          CI(t.boot95_lo, t.boot95_hi, sd_it, lc + "boot95_lo", lc + "boot95_hi"))]))
w("")
w("Source: medians, maxima and counts are computed here from the `s1` column of the per-run "
  "tables over completed runs (v1.1: CF-10; v2.0: CF-03); they agree with the tabulated medians "
  "and maxima of CF-18 (checked in the script). SD of the signed error = `sd_signed` of the s1 "
  "summary tables (CF-11 for v1.1; CF-04 for v2.0, row `criterion` = G-S; ddof = 1, checked "
  "against the per-run values). Mean signed error and its interval: `mean_signed`, `boot95_lo`, "
  "`boot95_hi` of the same tables (bootstrap of the mean, %s resamples, seed %s, a fresh "
  "generator per q). The \"within 0.05\" column is a counterfactual for v1.1, which had no G-S "
  "gate; it is shown for comparison only." % (
      N(int(cf04.bootstrap_resamples.iloc[0]), "CF-04", "bootstrap_resamples", "d"),
      N(int(cf04.bootstrap_seed.iloc[0]), "CF-04", "bootstrap_seed", "d")))
assert int(cf11.bootstrap_seed.iloc[0]) == int(cf04.bootstrap_seed.iloc[0])
w("")
cil = {}
for tag, t_ in (("v1.1", cf11), ("v2.0", cf04[cf04.criterion == "G-S"])):
    for q in (50, 60):
        t = t_[t_.q == q].iloc[0]
        cil[(tag, q)] = (t.boot95_lo, t.boot95_hi)
        assert t.boot95_lo < 0 < t.boot95_hi
wd = {k: v[1] - v[0] for k, v in cil.items()}
w("**What the intervals show.** All four bootstrap intervals of the mean signed error contain 0 "
  "(table above), so with n = 20 per block no mean offset from the closed form is detected in "
  "either protocol. The v2.0 intervals are narrower: their widths are %s (q = 50) and %s "
  "(q = 60) against %s and %s for v1.1 (computed here from the interval ends). The SD of the "
  "signed error under v2.0 is %s times (q = 50) and %s times (q = 60) the v1.1 value (computed "
  "here from `sd_signed`; unpaired, no interval)." % (
      computed("%.4g" % wd[("v2.0", 50)], wd[("v2.0", 50)], "CF-04", "boot95_hi - boot95_lo, q=50, G-S"),
      computed("%.4g" % wd[("v2.0", 60)], wd[("v2.0", 60)], "CF-04", "boot95_hi - boot95_lo, q=60, G-S"),
      computed("%.4g" % wd[("v1.1", 50)], wd[("v1.1", 50)], "CF-11", "boot95_hi - boot95_lo, q=50"),
      computed("%.4g" % wd[("v1.1", 60)], wd[("v1.1", 60)], "CF-11", "boot95_hi - boot95_lo, q=60"),
      computed("%.4g" % (stat[("v2.0", 50)]["sd"] / stat[("v1.1", 50)]["sd"]), stat[("v2.0", 50)]["sd"] / stat[("v1.1", 50)]["sd"], "CF-04; CF-11", "sd_signed v2.0 / sd_signed v1.1, q=50"),
      computed("%.4g" % (stat[("v2.0", 60)]["sd"] / stat[("v1.1", 60)]["sd"]), stat[("v2.0", 60)]["sd"] / stat[("v1.1", 60)]["sd"], "CF-04; CF-11", "sd_signed v2.0 / sd_signed v1.1, q=60")))
w("")
w("**The v1.1 verdict, for context.** The v1.1 confirmation (seeds 20501-20520) was `overall` = "
  "\"%s\" under its own rule [CF-14], with %s / 20 (q = 50) and %s / 20 (q = 60) runs passing "
  "G-A, G-F and G-N [CF-15 `n_pass`]. Under v1.1, S1 at 0.10 was passed by %s / 20 and %s / 20 "
  "runs [CF-11 `S1_pass`; the two failing runs are the two q = 60 runs above 0.10], and the v1.0 outcome by %s / 20 and %s / 20 [CF-15 "
  "`n_v1_0_run_pass`]. The S1 summary of v1.1 is a secondary criterion without a pass rule." % (
      cf14["overall"],
      N(int(cf15[cf15.q == 50].n_pass.iloc[0]), "CF-15", "row q=50, n_pass", "d"),
      N(int(cf15[cf15.q == 60].n_pass.iloc[0]), "CF-15", "row q=60, n_pass", "d"),
      N(int(cf11[cf11.q == 50].S1_pass.iloc[0]), "CF-11", "row q=50, S1_pass", "d"),
      N(int(cf11[cf11.q == 60].S1_pass.iloc[0]), "CF-11", "row q=60, S1_pass", "d"),
      N(int(cf15[cf15.q == 50].n_v1_0_run_pass.iloc[0]), "CF-15", "row q=50, n_v1_0_run_pass", "d"),
      N(int(cf15[cf15.q == 60].n_v1_0_run_pass.iloc[0]), "CF-15", "row q=60, n_v1_0_run_pass", "d")))
bad60 = block(cf10, 60)
assert (bad60.s1 > 0.10).sum() == 2
w("")

# ---------------------------------------------------------------------------------------------
SEC[0] = "3.7"
w("### 3.7 R1 evidence for the change: dispersion and advantage ratios")
w("")
cr = r102[r102.arm == "B_expcont"].iloc[0]
met_arms = r102[r102.a_met == True].arm.tolist()
assert met_arms == ["B_expcont"]
w("In R1, `B_expcont` (method 6) is the only one of the %s stage-1 arms whose pre-registered "
  "criterion is `%s`: the 95%% percentile bootstrap interval (10,000 resamples, seed 20261003) "
  "of the mean paired difference arm - `B_base` in |stage-1 error| lies below 0 at both q "
  "[RR-08 section 5] (part (a) %s), and no run that passed G-F under `B_base` fails it under the arm (part (b) "
  "`%s`) [R1-02]. The criterion is descriptive, not a gate. The paired difference is over the "
  "%s development seeds (10501-10510) per q." % (
      computed(str(len(r102)), int(len(r102)), "R1-02", "number of arm rows in stage1_criterion.csv"),
      cr.overall, str(bool(cr.a_met)).lower(), cr.b_status,
      N(int(cr.n_pairs_q50), "R1-02", "row B_expcont, n_pairs_q50", "d")))
w("")
w(row(["q", "paired diff. of \\|error\\|, mean [95% CI]", "seeds better", "SD of signed error, arm", "SD, base",
       "dispersion ratio (arm / base) [95% CI]", "mean per-pair ratio of stage-1 advantage SD", "runs within 0.05, base → arm"]))
w(row(["---"] * 8))
for q in (50, 60):
    qs = "q%d" % q
    pw = r118[(r118.arm == "B_expcont") & (r118.q == q) & (r118.metric == "stage1_rel_err_abs")].iloc[0]
    dp = r103[(r103.arm == "B_expcont") & (r103.q == q) & (r103.metric == "stage1_rel_err_signed")].iloc[0]
    ad = r104[(r104.arm == "B_expcont") & (r104.q == q)].iloc[0]
    gb = r124[(r124.arm == "B_base") & (r124.q == q)].iloc[0]
    ga_ = r124[(r124.arm == "B_expcont") & (r124.q == q)].iloc[0]
    w(row([str(q),
           "%s %s" % (N(cr["mean_" + qs], "R1-02", "row B_expcont, column mean_" + qs),
                      CI(cr["ci_mean_lo_" + qs], cr["ci_mean_hi_" + qs], "R1-02", "row B_expcont, column ci_mean_lo_" + qs, "row B_expcont, column ci_mean_hi_" + qs)),
           "%s / %s" % (N(int(pw.n_better), "R1-18", "row B_expcont q=%d stage1_rel_err_abs, n_better" % q, "d"), N(int(pw.n_pairs), "R1-18", "row B_expcont q=%d stage1_rel_err_abs, n_pairs" % q, "d")),
           N(dp.sd_arm, "R1-03", "row B_expcont q=%d stage1_rel_err_signed, sd_arm" % q),
           N(dp.sd_base, "R1-03", "row B_expcont q=%d stage1_rel_err_signed, sd_base" % q),
           "%s %s" % (N(dp.ratio, "R1-03", "row B_expcont q=%d stage1_rel_err_signed, ratio" % q),
                      CI(dp.ci_lo, dp.ci_hi, "R1-03", "ci_lo (same row)", "ci_hi (same row)")),
           N(ad.mean_ratio, "R1-04", "row B_expcont q=%d, mean_ratio" % q),
           "%s → %s of %s" % (N(int(gb.n_target0929_pass), "R1-24", "row B_base q=%d, n_target0929_pass" % q, "d"),
                              N(int(ga_.n_target0929_pass), "R1-24", "row B_expcont q=%d, n_target0929_pass" % q, "d"),
                              N(int(ga_.n_complete), "R1-24", "row B_expcont q=%d, n_complete" % q, "d"))]))
w("")
w("Source: R1-02 (`results/v2_refine/analysis/stage1_criterion.csv`, row `B_expcont`, columns "
  "`mean_q50`, `ci_mean_lo_q50`, `ci_mean_hi_q50` and the q60 columns; `a_met`, `b_status`, "
  "`overall`); R1-18 (`stage1_paired.csv`, `n_better`, `n_pairs`, metric `stage1_rel_err_abs`); "
  "R1-03 (`stage1_dispersion.csv`, metric `stage1_rel_err_signed`, columns `sd_arm`, `sd_base`, "
  "`ratio`, `ci_lo`, `ci_hi`; paired resampling, same bootstrap seed 20261003); R1-04 "
  "(`stage1_adv_ratio.csv`, `mean_ratio`); R1-24 (`stage1_gate_counts.csv`, "
  "`n_target0929_pass`).")
w("")
sg = {q: r118[(r118.arm == "B_expcont") & (r118.q == q) & (r118.metric == "stage1_rel_err_signed")].iloc[0] for q in (50, 60)}
cs = r122[r122.arm == "B_expcont"].iloc[0]
w("The improvement is a reduction of spread, not a shift of the mean: the paired difference of the "
  "signed error is %s %s at q = 50 and %s %s at q = 60, intervals that contain 0 [R1-18 `mean`, "
  "`ci_mean_lo`, `ci_mean_hi`, metric `stage1_rel_err_signed`]. The mean per-pair ratio of the "
  "stage-1 advantage SD (arm / baseline) is %s and %s [R1-04], and the Phase-B wall time is %s "
  "times the baseline's, including the table build [R1-22 `phase_wall_ratio_vs_base`]. The v2.0 "
  "change log records exactly these figures as the reason for the change [PL-01, v2.0 entry 1]; "
  "the chain of evidence, with the other five methods and the D1 and D2 diagnostics, is in "
  "section 4.6." % (
      N(sg[50]["mean"], "R1-18", "row B_expcont q=50 stage1_rel_err_signed, mean"),
      CI(sg[50].ci_mean_lo, sg[50].ci_mean_hi, "R1-18", "row B_expcont q=50 stage1_rel_err_signed, ci_mean_lo", "ci_mean_hi"),
      N(sg[60]["mean"], "R1-18", "row B_expcont q=60 stage1_rel_err_signed, mean"),
      CI(sg[60].ci_mean_lo, sg[60].ci_mean_hi, "R1-18", "row B_expcont q=60 stage1_rel_err_signed, ci_mean_lo", "ci_mean_hi"),
      N(r104[(r104.arm == "B_expcont") & (r104.q == 50)].mean_ratio.iloc[0], "R1-04", "row B_expcont q=50, mean_ratio"),
      N(r104[(r104.arm == "B_expcont") & (r104.q == 60)].mean_ratio.iloc[0], "R1-04", "row B_expcont q=60, mean_ratio"),
      N(cs.phase_wall_ratio_vs_base, "R1-22", "row B_expcont, phase_wall_ratio_vs_base")))
for q in (50, 60):
    assert sg[q].ci_mean_lo < 0 < sg[q].ci_mean_hi
w("")

# ---- supplementary ledger rows: small numbers and identifiers quoted in prose ----------------------
SEC[0] = "supplementary"
R3 = "RR-03"
report_only("8 / 7 / 3 / 9 / 3", "8,7,3,9,3", "section 1.2 table 'Entries' column (group sizes)")
report_only("20 collisions", 20, "section 6 item 4 'Without the exclusion ... 20 collisions' (V20/seed_inventory_stock.out, not in the pack)")
report_only("20 seeds that the protocol itself declares", 20, "section 6 item 4")
report_only("10,000 resamples, seed 20261003", "10000;20261003", "RR-08 section 5 'Bootstrap (D5)' (reports/v2/refine/01_preregistration.md)")
report_only("2026-10-03T23:35:28+00:00", "2026-10-03T23:35:28+00:00", "section 1.5 timeline, v2.0 lock commit 1d6d4d0")
report_only("2026-10-03T23:37:12Z", "2026-10-03T23:37:12Z", "section 1.5 timeline, re-rehearsal launch")
report_only("2026-10-04T04:24:47Z", "2026-10-04T04:24:47Z", "section 1.5 timeline, confirmation launch at d2e377d")
report_only("2026-10-04T04:37:16Z", "2026-10-04T04:37:16Z", "section 1.5 timeline, confirmation finished")
report_only("(2026-10-04)", "2026-10-04", "section 6 deviation 1 'Owner decision D-R7 (2026-10-04)' and CF-08 R7.owner_decision")
reg("all 3 v1.0 rehearsal failures", 3, "PL-04", "/change_log old, entry version 1.1 'stage-1 criterion ... moved from G-F to S1', field why")
reg("(40 runs)", int(cf01["n_expected"]), "CF-01", "n_expected (also PL-01 change log, v2.0 entry 5, '40 runs')")
reg("%s-%s" % ("10501", "10510"), "10501-10510", "PL-01", "confirmation.rehearsal.seeds 'development_seeds (10501-10510)'")
reg("0.001 grid", ii["ii_c"]["q50"]["grid_step"], "PL-05", "ii_c.q50.grid_step")
reg("state step 4, 16 GL nodes", "4.0;16", "PL-05", "verifier_numerics.q50.development.state_step / gl_half")
reg("state step 2, 32 GL nodes", "2.0;32", "PL-05", "verifier_numerics.q50.final.state_step / gl_half")
reg("panel width 2, 3 nodes", "2.0;3", "PL-07", "convergence.q50_parent[0].from_panel_width / from_nodes_per_panel")
reg("panel width 1 → 0.5, nodes 6 → 12", "1.0->0.5;6->12", "PL-05", "ii_a.q50.from_rule / to_rule")
reg("state step 0.25, 64 GL nodes", "0.25;64", "PL-05", "ii_b.q50.verifier.state_step / gl_half")
reg("0.05 and 0.10 lines", "0.05;0.10", "FG-01", "figures/FG-01_stage1_error_v1_1_vs_v2_0.png (reference lines G-S 0.05, S1 0.10)")

# ---- write ------------------------------------------------------------------------------------
(SCR / "sec03.md").write_text("\n".join(out) + "\n")
with open(SCR / "sec03_ledger.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=["statement_id", "section", "text", "value", "item_id", "locator"])
    wr.writeheader()
    wr.writerows(LED)
print("wrote", len(out), "lines;", len(LED), "ledger rows")
