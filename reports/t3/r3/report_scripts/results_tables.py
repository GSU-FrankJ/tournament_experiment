#!/usr/bin/env python3
"""Tables of ``reports/t3/r3/05_development.md`` (and ``summary.md``): every table is generated here from
``results/t3_r3/dev/launch_checks_absroot.json`` (the record of the post-launch checks; ``launch_checks.json`` is the first call, with a
relative ``--root``), ``results/t3_r3/dev/verify_clean_tree/*`` and ``results/t3_r3/dev/analysis/*`` (``tools/t3/t3r3_analysis.py``,
``t3r3_blind.py``, ``t3r3_last16.py``). Writes ``reports/t3/r3/tables/res_*.md``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnosis_tables import OUT, fmt, md  # noqa: E402

DEV = ROOT / "results" / "t3_r3" / "dev"
AN = DEV / "analysis"
VCT = DEV / "verify_clean_tree"
ARMS = ("bw", "bwA", "bwAB")


def rd(name: str) -> pd.DataFrame:
    """A CSV of the analysis directory."""
    return pd.read_csv(AN / name)


def head() -> Dict[str, Any]:
    """``headline.json``."""
    return json.loads((AN / "headline.json").read_text())


# ---------------------------------------------------------------------------------- checks
def _pair_label(arm: str) -> str:
    return "pair_ok (binding)" if arm != "bwAB" else "pair_ok (not a check)"


def t_checks() -> str:
    c = json.loads((DEV / "launch_checks_absroot.json").read_text())
    c1 = json.loads((DEV / "launch_checks.json").read_text())
    rows = []
    for v, x in c["verdicts"].items():
        x1 = c1["verdicts"][v]
        rows.append({"check": v if v != "pair_ok" else "pair_ok (binding for `bwA` and the diagnostic runs only)",
                     "pass": x["pass"], "fail": x["fail"], "not evaluated": x["not_evaluated"],
                     "first call (relative `--root`): pass / fail": f"{x1['pass']} / {x1['fail']}"})
    runs = c["runs"]
    arms = []
    for a in ("bw", "bwA", "bwAB"):
        sub = [r for r in runs if r["arm"] == a]
        fails = {v: sum(1 for r in sub if r.get(v) is False) for v in c["verdicts"] if v != "pair_ok"}
        fails = {k: n for k, n in fails.items() if n}
        dev = [r for r in sub if r["run_kind"] == "dev"]
        n_ok = sum(1 for r in dev if r.get("pair_ok") is True)
        n_diag = sum(1 for r in sub if r["run_kind"] == "diagnostic" and r.get("pair_ok") is True)
        pair = ("-" if a == "bw" else
                f"{n_ok} of {len(dev)} development runs as required (identical over the whole run where the trigger did not fire, same stage 3 up to the check where it fired); "
                f"{n_diag} of 1 diagnostic run equal to T3-R2's original up to the check" if a == "bwA" else
                f"{n_ok} of {len(dev)} development runs identical to `bw` (not a check: `bwAB` differs from `bw` by construction); "
                f"{n_diag} of 1 diagnostic run equal to T3-R2's original up to the check")
        arms.append({"arm": a, "runs": len(sub), "trigger fired": sum(1 for r in sub if r["fired"]),
                     "trigger would fire (recomputed)": sum(1 for r in sub if r["would_fire"]),
                     "failed checks (other than `pair_ok`)": ", ".join(f"{k}: {n}" for k, n in fails.items()) or "none", "C-PAIR against `bw`": pair})
    ci = c["cross_arm_init"]
    txt = (f"Infrastructure ledger present: {c['infra_events_present']}; events: {len(c['infra_events']['events']) if c['infra_events'] else 'n/a'}; "
           f"stopped without verdict: {c['infra_events']['stopped_without_verdict'] if c['infra_events'] else 'n/a'}. Cross-arm C-INIT: "
           f"attempt-1 digest differs across arms in {len(ci['attempt1_digest_differs_across_arms'])} (q, seed); attempt-2 digests of `bwA` and "
           f"`bwAB` differ in {len(ci['attempt2_digest_differs_bwA_bwAB'])}.")
    return md(rows) + "\n\n" + md(arms) + "\n\n" + txt


def t_boundary() -> str:
    b = json.loads((VCT / "boundary.json").read_text())
    bc = pd.read_csv(VCT / "boundary.csv")
    rows = []
    for label, flag, key in (("a clean tree", False, "clean_by_arm"), ("a dirty tree", True, "dirty_by_arm")):
        sub = bc[bc.manifest_dirty == flag]
        rows.append({"manifests record": label, "runs": b["n_clean" if not flag else "n_dirty"], "`bw`": b[key]["bw"], "`bwA`": b[key]["bwA"], "`bwAB`": b[key]["bwAB"],
                     "first start (UTC)": sub.start_utc.min(), "last start (UTC)": sub.start_utc.max()})
    txt = (f"\n\n`reports/t3/r3/report_scripts/results_tables.py` was created at {b['results_tables_py_birth_utc']} (birth time recorded in `boundary.json`); the wave was launched at "
           f"02:20:02 UTC. Runs whose recorded flag disagrees with 'started after the file was created': {len(b['runs_whose_flag_disagrees_with_the_boundary'])}. "
           f"All {b['n_runs']} manifests carry the commit `{b['commits'][0]}`; the diagnostic runs are all in the clean group "
           f"({b['dirty_by_kind']['diagnostic']} dirty). Table: `results/t3_r3/dev/verify_clean_tree/boundary.csv`.")
    return md(rows) + txt


def t_pairs() -> str:
    b = json.loads((VCT / "boundary.json").read_text())
    return (f"Pairs of `bwA` and `bw` of the same seed: {b['bwA_bw_pairs']}. Pairs in which the two runs record different flags: {b['pairs_flags_differ']}. "
            f"Of the {b['bwA_bw_pairs']} pairs, {b['pairs_bit_identical_full_total']} are bit-identical over the whole run (the trigger did not fire) and "
            f"{b['pairs_attempt1_identical_total']} have the same stage 3 up to the check (it fired); bit-identical with different flags: "
            f"{b['pairs_bit_identical_full_and_flags_differ']}; same stage 3 with different flags: {b['pairs_attempt1_identical_and_flags_differ']}. "
            f"Table: `results/t3_r3/dev/verify_clean_tree/pairs_across_boundary.csv`.")


def t_rerun() -> str:
    final = json.loads((VCT / "rerun_results_final.json").read_text())
    first = {r["run"]: r for r in json.loads((VCT / "rerun_results_first_comparison.json").read_text())}
    rows = []
    for r in final:
        cr = r["compare_runs_stdout"][0].split(" npz")[0]
        a1 = r.get("attempt1")
        rows.append({"run": f"`{r['run']}`", "why this run": r["why"].replace("ADDITION: ", "(addition) "), "flag, original": "dirty" if not r["orig_manifest_clean_tree"] else "clean",
                     "flag, re-run": "clean" if r["new_manifest_clean_tree"] else "dirty", "`compare_runs.py bw` exit code": r["compare_runs_returncode"],
                     "npz files identical": cr, "same file set": r["file_sets_equal"],
                     "`attempt1/` (64 exports, CSVs, bin maps, state) and `retry.json`": ("identical" if (a1 and a1["ok"] and not r["retry_json_diff"]) else "-") if r["fired"] else "no `attempt1/`",
                     "gates / rule log / drift / summary JSON": "identical" if not any(r["json_diffs"].values()) else "DIFFERENT",
                     "re-run wall time (s, driver clock)": round(first[r["run"]]["wall_sec"])})
    return md(rows)


# ---------------------------------------------------------------------------------- counts and target
def t_counts() -> str:
    d = rd("counts.csv")
    rows = []
    for _, x in d.iterrows():
        rows.append({"arm": x.arm, "q": x.q, "runs": int(x.runs), "pass": int(x["pass"]), "exact 95 % interval": f"[{x.ci_lo:.3f}, {x.ci_hi:.3f}]",
                     "worst Gmax_full (run)": f"{x.worst_Gmax_full:.5f} ({x.worst_Gmax_full_run})",
                     "median eta_3": x.median_eta_T_over_dw, "median rmse_pos/e*(0)": x.median_rmse_pos_T_over_eT0,
                     "median tail_mean/e*(0)": x.median_tail_mean_T_over_eT0, "median peak_T_signed": x.median_peak_T_signed,
                     "median Gmax_full": x.median_Gmax_full_over_dw, "median EXP_root": x.median_EXP_root_over_dw})
    return md(rows)


def t_target() -> str:
    h = head()
    rows = []
    for a in ARMS:
        t, c = h["targets"][a], h["counts"][a]
        rows.append({"arm": a, "eligible": "no (control)" if a == "bw" else "yes", "pass q50": f"{t['pass_q50']} of 20", "pass q60": f"{t['pass_q60']} of 20",
                     "pass total": f"{t['pass_total']} of 40", "total >= 38": t["total_condition"], "q50 >= 18": t["q50_condition"],
                     "q60 >= 18": t["q60_condition"], "reaches the target": t["reaches_target"]})
    sel = h["selection"]
    return md(rows) + f"\n\n**Selection:** {sel['reason']} (proposed: `{sel['proposed']}`; both arms reach the target: {sel['both_reach_target']})."


def t_blind() -> str:
    b = json.loads((AN / "blind.json").read_text())
    rows = [{"arm": a, "pass q50": v["pass_q50"], "pass q60": v["pass_q60"], "pass total": v["pass_total"], "reaches the target": v["reaches_target"]}
            for a, v in b["arms"].items()]
    return md(rows) + f"\n\nBlind recomputation (`tools/t3/t3r3_blind.py`): proposed `{b['proposed']}`; agrees with the analysis: {b.get('agrees_with_headline')}; differences: {b.get('differences')}."


# ---------------------------------------------------------------------------------- per seed
def t_perseed(q: int) -> str:
    d = pd.read_csv(AN / "per_run.csv")
    rows = []
    for s in range(70501, 70521):
        row: Dict[str, Any] = {"seed": s}
        for a in ARMS:
            r = d[(d.arm == a) & (d.q == q) & (d.seed == s) & (d.run_kind == "dev")].iloc[0]
            tag = "pass" if r.passed else f"FAIL ({r.failing_gates if isinstance(r.failing_gates, str) and r.failing_gates else r.fail_reason})"
            if a != "bw" and r.fired:
                tag += " [fired]"
            elif a == "bw" and r.would_fire:
                tag += " [would fire]"
            row[a] = tag
            row[f"{a} eta_3"] = r.eta_T_over_dw
            row[f"{a} Gmax"] = r.Gmax_full_over_dw
        rows.append(row)
    return md(rows)


# ---------------------------------------------------------------------------------- comparisons
def t_paired() -> str:
    d = rd("paired.csv")
    rows = []
    for _, x in d.iterrows():
        rows.append({"comparison": f"{x.a} - {x.b}", "statistic": x.stat, "q": x.q, "pairs": int(x.n_pairs), "mean a": x.mean_a, "mean b": x.mean_b,
                     "mean difference": x.mean_diff, "paired bootstrap 95 % interval": f"[{x.ci_lo:.5f}, {x.ci_hi:.5f}]",
                     "seeds with a < b / a > b": f"{int(x.n_neg)} / {int(x.n_pos)}"})
    x = rd("crosstab.csv")
    from scipy.stats import binomtest
    rows2 = [{"arm": r.arm, "reference": r.ref, "both pass": int(r.both_pass), "reference only": int(r.ref_only), "arm only": int(r.arm_only),
              "both fail": int(r.both_fail),
              "exact McNemar p (two-sided)": binomtest(min(int(r.ref_only), int(r.arm_only)), int(r.ref_only) + int(r.arm_only), 0.5).pvalue
              if int(r.ref_only) + int(r.arm_only) else 1.0} for _, r in x.iterrows()]
    return md(rows) + "\n\nPass / fail cross-tables over the 40 seeds:\n\n" + md(rows2)


def label(r: pd.Series, rank: float) -> str:
    """The D2 working class of a failing run, by rule (the report confirms it by hand): (K) in the stage-3 clamp regime (S3 of the frozen attempt
    above 0.2); (F) development ``Delta`` of the frozen attempt at local 1600 of at least 0.1 outside the regime; (V) otherwise, when the failing
    value is the largest of the last 16 exports of the phase (rank 1); (U) anything else."""
    d_final = r.a2_final_Delta if r.fired else r.trig_Delta
    if r.S3_frozen > 0.2:
        return "K"
    if d_final == d_final and d_final >= 0.1:
        return "F"
    return "V" if rank == 1 else "U"


def t_failures() -> str:
    d = pd.read_csv(AN / "per_run.csv")
    l16 = pd.read_csv(AN / "last16_summary.csv")
    rows = []
    for _, r in d[~d.passed].iterrows():
        mine = l16[l16.run == r.run]
        rank = float(mine.final_rank_from_top.min()) if len(mine) else float("nan")
        ev = "; ".join(f"{m.quantity.replace('_over_dw', '').replace('_T_over_eT0', '')} {m.final_value:.5f} > {m.limit}: rank {int(m.final_rank_from_top)} of 16 from the top, "
                       f"{int(m.n_above_limit)} of 16 above the limit" for _, m in mine.iterrows())
        rows.append({"run": f"`{r.run}`", "failing gates": r.failing_gates if isinstance(r.failing_gates, str) and r.failing_gates else r.fail_reason,
                     "eta_3": r.eta_T_over_dw, "rmse_pos/e*(0)": r.rmse_pos_T_over_eT0, "tail_mean/e*(0)": r.tail_mean_T_over_eT0,
                     "Gmax_full": r.Gmax_full_over_dw, "S3 (first attempt)": r.S3_first, "S3 (frozen)": r.S3_frozen, "trigger fired": r.fired,
                     "stage-2 stretch nodes": int(r.stretch_nodes), "dev Delta at local 1600 (frozen attempt)": r.a2_final_Delta if r.fired else r.trig_Delta,
                     "last 16 exports (failing quantities)": ev, "class": label(r, rank)})
    return md(rows) if rows else "No run failed."


def t_fired() -> str:
    d = rd("fired.csv")
    if len(d) == 0:
        return "No trigger fired."
    pr = pd.read_csv(AN / "per_run.csv").set_index("run")
    rows = [{"run": f"`{r.run}`", "kind": r.run_kind, "trigger Delta": r.trigger_Delta, "s": r.trigger_s, "C": r.trigger_C, "reason": r.reasons,
             "S3 of attempt 1": pr.loc[r.run, "S3_first"], "S3 of the frozen attempt 2": pr.loc[r.run, "S3_frozen"],
             "attempt-2 final Delta": r.attempt2_final_Delta, "outcome": "pass" if r.passed else f"FAIL ({r.fail_reason})", "eta_3": r.eta_3,
             "Gmax_full": r.Gmax_full} for _, r in d.iterrows()]
    return md(rows)


def t_firstattempt() -> str:
    d = rd("first_attempt.csv")
    h = head()["first_attempt"]
    rows = [{"arm": r.arm, "first-attempt pass (fired = failure)": int(r.first_attempt_pass), "pass of the run": int(r.pass_final), "trigger fired": int(r.fired)}
            for _, r in d.iterrows()]
    txt = (f"\n\nInvariant `bwA` = `bw` per seed under the same convention: mismatches {len(h['mismatch_bwA_vs_bw'])}; stop: {h['stop']}. "
           f"Raw pass count of `bw`: {h['bw_raw_pass_count']}; `bw` runs that would fire and nevertheless pass the gates: {h['bw_fired_runs_that_pass_in_bw']}.")
    return md(rows) + txt


def t_causal() -> str:
    a = rd("causal_b.csv")
    rows = [{"arm": r.arm, "q": r.q, "runs": int(r.runs), "runs with S3 > 0.2 (first attempt)": int(r["S3_first_gt_0.2"]),
             "median S3 (first)": r.median_S3_first, "runs with a tail node at the mean clamp (u01600)": int(r.runs_with_tail_at_mean_clamp),
             "median share of tail nodes at the mean clamp": r.median_first_out_mu_at_clamp, "median minimum alpha": r.median_first_alpha_min,
             "median dev Delta, local 1600": r.median_first_trigger_Delta_local1600, "fire rate": f"{int(r.fired)} of {int(r.runs)}",
             "runs with a stage-2 stretch": int(r.stretch_runs_final)} for _, r in a.iterrows()]
    b = rd("b_pairs.csv")
    rows2 = [{"pair": f"{r.arm_a} vs {r.arm_b}", "q": r.q, "regime runs a / b": f"{int(r.regime_a)} / {int(r.regime_b)}",
              "McNemar: a only / b only": f"{int(r.n_x_only)} / {int(r.n_y_only)}", "exact p": r.p_exact, "seeds with no trigger in either arm": int(r.n_neither_fired),
              "eta_3 mean difference [95 %]": f"{r.eta_T_over_dw_mean_diff:.5f} [{r.eta_T_over_dw_ci_lo:.5f}, {r.eta_T_over_dw_ci_hi:.5f}]",
              "peak_T_signed mean difference [95 %]": f"{r.peak_T_signed_mean_diff:.4f} [{r.peak_T_signed_ci_lo:.4f}, {r.peak_T_signed_ci_hi:.4f}]"} for _, r in b.iterrows()]
    return md(rows) + "\n\nPaired by seed, on first attempts:\n\n" + md(rows2)


def t_s3() -> str:
    d = pd.read_csv(AN / "per_run.csv")
    d = d[d.run_kind == "dev"]
    rows = []
    for a in ARMS:
        for q in (50, 60, "both"):
            x = d[(d.arm == a) & ((d.q == q) if q != "both" else True)]
            rows.append({"arm": a, "q": q, "runs": len(x), "S3 > 0.2, first attempt": int((x.S3_first > 0.2).sum()), "S3 > 0.2, frozen attempt": int((x.S3_frozen > 0.2).sum()),
                         "median S3, frozen": float(x.S3_frozen.median()), "runs with a stage-2 stretch": int(x.has_stretch.sum()),
                         "stretch nodes, total": int(x.stretch_nodes.sum()), "passing runs with S3 > 0.2 (frozen)": int(((x.S3_frozen > 0.2) & x.passed).sum()),
                         "failing runs with S3 > 0.2 (frozen)": int(((x.S3_frozen > 0.2) & ~x.passed).sum())})
    return md(rows)


def t_predictions() -> str:
    from scipy.stats import beta as _beta
    d = pd.read_csv(AN / "per_run.csv")
    d = d[d.run_kind == "dev"]

    def cp(k: int, n: int) -> str:
        lo = 0.0 if k == 0 else _beta.ppf(0.025, k, n - k + 1)
        hi = 1.0 if k == n else _beta.ppf(0.975, k + 1, n - k)
        return f"{k} of {n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]"
    bw, ba, bab = (d[d.arm == a] for a in ARMS)
    chk = json.loads((DEV / "launch_checks_absroot.json").read_text())
    pairs = [r for r in chk["runs"] if r["arm"] == "bwA" and r["run_kind"] == "dev"]
    full = sum(1 for r in pairs if r["pair_ok"] and not r["fired"])
    nonfired = sum(1 for r in pairs if not r["fired"])
    diag = pd.read_csv(AN / "diagnostic.csv")
    rows = [
        {"prediction (`04_preregistration.md` section 11)": "A fires on about 9 % of the runs (7 of 80 in the recorded runs)",
         "observed": f"`bw` would fire in {cp(int(bw.would_fire.sum()), len(bw))}; `bwA` fired in {cp(int(ba.fired.sum()), len(ba))}; `bwAB` fired in {cp(int(bab.fired.sum()), len(bab))} (exact 95 % intervals)"},
        {"prediction (`04_preregistration.md` section 11)": "`bwA` and `bw` are bit-identical wherever A does not fire",
         "observed": f"{full} of {nonfired} non-fired pairs are bit-identical over the whole run; the {len(pairs) - nonfired} fired pairs have the same stage 3 up to the check"},
        {"prediction (`04_preregistration.md` section 11)": "the diagnostic run of seed 60518 fires A in both arms",
         "observed": f"fired in {int(diag.fired.sum())} of {len(diag)} diagnostic runs (`bwA`, `bwAB`); Delta at local 1600 = {diag.trigger_Delta.iloc[0]:.5f}; "
                     f"attempt 1 equals T3-R2's original: {bool(diag.attempt1_equals_T3R2_original.all())}; both second attempts pass: {bool(diag.passed.all())}"},
        {"prediction (`04_preregistration.md` section 11)": "B is not expected to change the fire rate if the clamp regime is only a marker",
         "observed": f"fire rate of `bwAB` {int(bab.fired.sum())} of {len(bab)} against `bwA` {int(ba.fired.sum())} of {len(ba)}; the seeds that fired are not the same ones "
                     f"(see the table of fired triggers); no test at these counts"}]
    return md(rows)


def t_diag() -> str:
    d = rd("diagnostic.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append({"arm": r.arm, "A fired": r.fired, "trigger Delta / s / C": f"{r.trigger_Delta:.5f} / {r.trigger_s:.2f} / {r.trigger_C:.4f}",
                     "attempt 1 equals T3-R2's original (C-PAIR)": r.attempt1_equals_T3R2_original, "S3 (attempt 1)": r.S3_first,
                     "F(u01600) of the frozen attempt": r.F_u1600, "formation export (threshold 26.78)": r.formation_export,
                     "sensitivity at u01600": r.sens_u1600, "minimum sensitivity u100-u300": r.sens_min_u100_300,
                     "passes": r.passed, "eta_3": r.eta_3, "rmse_pos/e*(0)": r.rmse_pos, "Gmax_full": r.Gmax_full, "fail reason": r.fail_reason})
    return md(rows)


def t_cost() -> str:
    d = rd("cost.csv")
    rows = [{"arm": r.arm, "runs": int(r.runs), "mean updates": r.mean_updates, "mean policy rows": r.mean_policy_rows, "mean minibatch steps": r.mean_minibatch_steps,
             "mean dev checks": r.mean_dev_checks, "total wall (s)": r.total_wall_sec, "mean wall (s)": r.mean_wall_sec, "median wall (s)": r.median_wall_sec,
             "mean verifier (s)": r.mean_dev_verifier_sec, "ratio updates": r.ratio_mean_updates, "ratio policy rows": r.ratio_mean_policy_rows,
             "ratio wall (total)": r.ratio_total_wall_sec} for _, r in d.iterrows()]
    return md(rows)


def t_cost_identical() -> str:
    d = pd.read_csv(AN / "per_run.csv")
    d = d[d.run_kind == "dev"].set_index(["arm", "q", "seed"])
    keys = sorted({(q, s) for (_, q, s) in d.index})
    nf_a = [k for k in keys if not d.loc[("bwA",) + k, "fired"]]
    nf_ab = [k for k in keys if not d.loc[("bwAB",) + k, "fired"] and not d.loc[("bwA",) + k, "fired"]]          # neither candidate fired
    fired_a = [k for k in keys if d.loc[("bwA",) + k, "fired"]]

    def mean(arm: str, ks: List[Any]) -> float:
        return float(sum(d.loc[(arm,) + k, "wall_sec"] for k in ks) / len(ks))
    rows = [
        {"comparison": "`bwA` against `bw`, the seeds where `bwA` did not fire (bit-identical computations)", "seeds": len(nf_a), "mean wall a (s)": mean("bwA", nf_a),
         "mean wall b (s)": mean("bw", nf_a), "ratio a / b": mean("bwA", nf_a) / mean("bw", nf_a)},
        {"comparison": "`bwA` against `bw`, the seeds where `bwA` fired (stage 3 trained twice)", "seeds": len(fired_a), "mean wall a (s)": mean("bwA", fired_a),
         "mean wall b (s)": mean("bw", fired_a), "ratio a / b": mean("bwA", fired_a) / mean("bw", fired_a)},
        {"comparison": "`bwAB` against `bw`, the seeds where neither `bwA` nor `bwAB` fired", "seeds": len(nf_ab), "mean wall a (s)": mean("bwAB", nf_ab),
         "mean wall b (s)": mean("bw", nf_ab), "ratio a / b": mean("bwAB", nf_ab) / mean("bw", nf_ab)},
        {"comparison": "`bwAB` against `bwA`, the seeds where neither fired", "seeds": len(nf_ab), "mean wall a (s)": mean("bwAB", nf_ab),
         "mean wall b (s)": mean("bwA", nf_ab), "ratio a / b": mean("bwAB", nf_ab) / mean("bwA", nf_ab)}]
    return md(rows)


TABLES = {"res_pairs": t_pairs, "res_cost_identical": t_cost_identical, "res_checks": t_checks, "res_boundary": t_boundary, "res_rerun": t_rerun, "res_s3": t_s3, "res_predictions": t_predictions, "res_counts": t_counts, "res_target": t_target, "res_blind": t_blind,
          "res_perseed_q50": lambda: t_perseed(50), "res_perseed_q60": lambda: t_perseed(60), "res_paired": t_paired,
          "res_failures": t_failures, "res_fired": t_fired, "res_firstattempt": t_firstattempt, "res_causal": t_causal, "res_diag": t_diag,
          "res_cost": t_cost}


def main() -> int:
    """Write the tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in TABLES.items():
        (OUT / f"{name}.md").write_text(fn() + "\n")
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
