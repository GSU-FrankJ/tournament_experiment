#!/usr/bin/env python3
"""R2c pre-registered analysis of the pilot wave S and the mechanical selection rule (reports 02, 03).

Wave S = four start-distribution arms (``launch_refine.R2C_WAVE_S_ARMS``), each a full Phase A from
scratch (mode ``phase_A``, 1600 updates) on the development seeds, paired by (q, seed) against R1's
``parents_A`` runs (called ``A_base`` in every table, as in the R2b tool). The statistics are those of
``tools/v2/refine_r2b_analysis.py`` (itself the R1 tool ``refine_analysis.py``): final tier for the gate
metrics, differences ``arm - A_base`` paired by (q, seed), 10,000 percentile bootstrap resamples, one fresh
``numpy.random.default_rng(<boot seed>)`` per (q, statistic) in table order, the primary metric
``stage2_peak_rel_err_abs`` (absolute signed peak error of the end-of-A last iterate, final tier;
improvement = negative difference). The boot seed is a CLI value (default 20261005) that overrides the
R2b module's import-time value before any statistic is computed and is recorded in the CSVs, in
``selection.json`` and in the reports.

Per arm (as in R1 / R2b): criterion (a) the 95% bootstrap CI of the mean paired difference of the primary
metric excludes 0 in the improving direction at BOTH q; (b) no run that passed G-A and its G-N part under
the baseline fails it under the arm. Selection rule (the owner's text, quoted verbatim in
``03_selection.md`` and ``selection.json``): among the arms meeting (a) and (b), the largest total number of
runs with |peak error| <= 0.05 over both q; tie -> the smaller mean |peak error| over both q; further tie
-> the smaller change from the locked sampler (later ``local_first``, then smaller share). The rule is the
pure function :func:`select_arm`; it needs no result file.

Reads (read only): the wave runs ``<wave-dir>/q*/seed*/<arm>`` (``--wave-dir``, default ``<root>/waveS``),
the baseline ``<r1-root>/parents_A/q*/seed*``, the rehearsal parent candidates ``<ref-root>/q*/seed*/gates.json``
(pseudo-arm ``parent_u1600``), and, as descriptive reference rows that are never eligible, the R2b arms
``A_peak25`` and ``A_peak50`` at ``<r2b-root>/waveA/q*/seed*/<arm>`` (named ``R2b_A_peak25`` /
``R2b_A_peak50``; skipped with a note if ``--r2b-root`` does not exist). Optional files:
``<root>/launch_checks.json`` and ``<root>/v20_reproduction_checks.json`` (printed first in the report) and
``<wave-dir>/launch_*.json``.

Outputs: CSVs and ``selection.json`` under ``--out`` (default ``<root>/analysis``), the reports
``02_pilot_waveS.md`` and ``03_selection.md`` and the figures under ``--reports`` (``figures/``). The tool
is deterministic given the same inputs: no timestamps are written (the launch records' own timestamps are
copied from the records into the provenance table).

Exit code: 0 when every arm is complete (a selection of "no arm" is a result, not an error); 3 when any arm
has a missing, failed or incomplete run (reported as INCOMPLETE on stderr, in the reports and in
``selection.json``; such an arm cannot be selected).

Usage:
    python tools/v2/refine_r2c_analysis.py --root results/v2_refine_r2c --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")      # matplotlib: fixed PDF creation date (deterministic figures)

import refine_analysis as R  # noqa: E402  (extraction, pairing, bootstrap, criterion, Doc writer)

_BOOT_SEED_BEFORE_B = R.BOOT_SEED
import refine_r2b_analysis as B  # noqa: E402  (R2b extras and display helpers; sets R.BOOT_SEED = 20261004 at import)

R.BOOT_SEED = _BOOT_SEED_BEFORE_B       # undo that import side effect; compute() sets the CLI value before any statistic
import launch_refine as L  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_BOOT_SEED = 20261005
QS: Tuple[int, ...] = B.QS
SEEDS: Tuple[int, ...] = B.SEEDS
BASE = B.BASE_A                                   # "A_base" = R1 parents_A
PARENT = B.PARENT                                 # rehearsal parent candidate (pseudo-arm)
PRIMARY = B.PRIMARY                               # stage2_peak_rel_err_abs
TAIL_PEAK = B.TAIL_PEAK                           # 0.05, inclusive
CRIT_LABEL = "vs baseline"
REF_PREFIX = "R2b_"
REF_ARMS: Tuple[str, ...] = ("A_peak25", "A_peak50")
REF_LABEL = "R2b reference (descriptive, not eligible)"
R1_ROOT = "/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine/results/v2_refine"
R2B_REF_ROOT = "/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-r2b/results/v2_refine_r2b"
FIGURE_TITLE = "Wave S"
N_BOOT = R.N_BOOT
SEED_LEVEL_COLS = ["wave", "comparison", "arm", "baseline", "q", "metric", "seed", "arm_value", "base_value", "diff"]

RULE_TEXT = (
    "Criterion per arm (as in R1/R2b): (a) the 95% percentile bootstrap CI of the mean paired difference of the "
    "absolute signed peak error against parents_A excludes 0 in the improving direction at BOTH q; (b) no run that "
    "passed G-A and its G-N part under the baseline fails it under the arm. Bootstrap: 10,000 resamples, "
    "numpy.random.default_rng(20261005), one fresh generator per (q, statistic) in table order.\n\n"
    "Selection rule. Among the arms meeting (a) and (b): the largest total number of runs with |peak error| <= 0.05 "
    "over both q; tie -> the smaller mean |peak error| over both q; further tie -> the smaller change from the "
    "locked sampler (later local_first, then smaller share). The selected arm's start_weights is the one pipeline "
    "change of v2.1. No arm meets (a) and (b): write the pilot reports and stop. Do not relax the rule, do not add "
    "arms.")
INTERPRETATION = [
    "|peak error| is the primary metric `stage2_peak_rel_err_abs` of the end-of-A last iterate (final tier).",
    "\"Runs with |peak error| <= 0.05\" is the count, over the 20 runs of the arm (both q), with primary <= 0.05 "
    "(inclusive: a run with exactly 0.05 counts).",
    "\"Mean |peak error| over both q\" is the mean of the primary metric over all 20 runs of the arm.",
    "An arm is complete iff all 20 planned runs are done and all 10 paired differences per q are finite; an arm that is "
    "not complete cannot be selected (it is recorded, never skipped). Arms are compared on the same footing.",
    "Eligible = (a) met at both q and (b) holds (b_status = holds: no violation and nothing pending) and complete.",
    "The tie-break key of the last step is (-local_first, share) ascending, local_first read from the arm's "
    "start_weights (default 1) and share from its peak_share. If the first two arms of the ranking agree in all four "
    "key components the rule gives no answer: nothing is selected and the tie is reported (the arm name is never used "
    "to decide).",
    "R2b's A_peak25 and A_peak50 appear in the cross-arm tables as reference rows only; they are never inputs of "
    "the selection.",
]
SORT_FIELDS = ("n_abs_peak_le_0.05_total", "mean_abs_peak_error", "local_first", "share")
SORT_DESCRIPTION = ("sort key (ascending) = (-number of runs with |peak error| <= 0.05 over both q, mean |peak error| "
                    "over both q, -local_first, share)")


# --------------------------------------------------------------------------- the selection rule (pure)
def count_le(values: Any, thr: float = TAIL_PEAK) -> int:
    """Number of values <= ``thr`` (inclusive; NaN never counts)."""
    return int((np.asarray(values, dtype=float) <= thr).sum())


def ineligible_reasons(t: Mapping[str, Any]) -> List[str]:
    """Why an arm is not eligible for the selection (empty list = eligible).

    Args:
        t: Per-arm inputs; uses ``complete``, ``a_met``, ``b_status`` and, if present, ``not_done_runs``,
            ``a`` (per-q dicts with ``met``) and ``b_violations``.

    Returns:
        Human-readable reasons, one per failed condition.
    """
    reasons: List[str] = []
    if not t["complete"]:
        nd = list(t.get("not_done_runs") or [])
        shown = ", ".join(nd) if len(nd) <= 4 else ", ".join(nd[:3]) + f", ... ({len(nd)} runs not done in all)"
        reasons.append("incomplete: " + (shown if nd else "fewer than all paired finite runs"))
    if not t["a_met"]:
        qs = [q for q, v in (t.get("a") or {}).items() if not v.get("met")]
        reasons.append("(a) not met" + (f" at {', '.join(qs)}" if qs else " at both q"))
    if t["b_status"] != "holds":
        viol = list(t.get("b_violations") or [])
        reasons.append(f"(b) {t['b_status']}" + (f" ({', '.join(viol)})" if viol else ""))
    return reasons


def sort_key(t: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    """Ascending sort key of the rule: (-n runs <= 0.05, mean |peak|, -local_first, share)."""
    return (-float(t["n_abs_peak_le_0.05_total"]), float(t["mean_abs_peak_error"]),
            -float(t["local_first"]), float(t["share"]))


def select_arm(table: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Apply the pre-registered selection rule mechanically to a per-arm table.

    Args:
        table: One mapping per arm with the keys ``arm``, ``complete`` (bool), ``a_met`` (bool),
            ``b_status`` (``holds`` / ``violated`` / ``incomplete``), ``n_abs_peak_le_0.05_total`` (int),
            ``mean_abs_peak_error`` (float), ``local_first`` (int) and ``share`` (float); the optional keys
            ``not_done_runs``, ``a``, ``b_violations`` only improve the reason strings. The result does not
            depend on the order of ``table``.

    Returns:
        Dict with ``selected`` (arm name or None), ``outcome`` (``selected`` / ``none_eligible`` /
        ``unresolved_tie``), ``decided_by``, ``reason``, ``ranking`` (eligible arms, best first, with the sort
        key values) and ``ineligible`` ({arm: reasons}, sorted by arm).
    """
    rows = [dict(t) for t in table]
    names = [r["arm"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate arms in the selection table: {names}")
    reasons = {r["arm"]: ineligible_reasons(r) for r in rows}
    elig = sorted((r for r in rows if not reasons[r["arm"]]), key=lambda r: (sort_key(r), r["arm"]))
    ranking = [{"rank": i + 1, "arm": r["arm"], "n_abs_peak_le_0.05_total": int(r["n_abs_peak_le_0.05_total"]),
                "mean_abs_peak_error": float(r["mean_abs_peak_error"]), "local_first": int(r["local_first"]),
                "share": float(r["share"]), "sort_key": list(sort_key(r))} for i, r in enumerate(elig)]
    ineligible = {a: reasons[a] for a in sorted(reasons) if reasons[a]}
    out: Dict[str, Any] = {"selected": None, "outcome": "none_eligible", "decided_by": None, "reason": "",
                           "ranking": ranking, "ineligible": ineligible}
    if not ranking:
        why = "; ".join(f"{a}: {' and '.join(rs)}" for a, rs in ineligible.items()) or "no arm in the table"
        out["reason"] = f"No arm meets both parts of the criterion with a complete set of runs. {why}."
        return out
    labels = ("the number of runs with |peak error| <= 0.05 over both q (larger wins)",
              "the mean |peak error| over both q (smaller wins)",
              "local_first (later wins: the smaller change from the locked sampler)",
              "the share (smaller wins: the smaller change from the locked sampler)")
    if len(ranking) == 1:
        decided = "the only eligible arm"
    else:
        k0, k1 = ranking[0]["sort_key"], ranking[1]["sort_key"]
        diff = [i for i in range(4) if k0[i] != k1[i]]
        if not diff:
            out["outcome"] = "unresolved_tie"
            out["reason"] = (f"The first two eligible arms ({ranking[0]['arm']}, {ranking[1]['arm']}) agree in all four "
                             "components of the rule's key; the rule gives no answer and nothing is selected.")
            return out
        decided = f"decided by {labels[diff[0]]}: {ranking[0]['arm']} vs {ranking[1]['arm']}"
    out["selected"], out["outcome"], out["decided_by"] = ranking[0]["arm"], "selected", decided
    top = ranking[0]
    out["reason"] = (f"{top['arm']} meets (a) at both q and (b), is complete, and ranks first among {len(ranking)} "
                     f"eligible arm(s) ({decided}); n runs with |peak error| <= 0.05 = {top['n_abs_peak_le_0.05_total']}, "
                     f"mean |peak error| = {top['mean_abs_peak_error']:.6g}, local_first = {top['local_first']}, "
                     f"share = {top['share']:g}.")
    return out


# --------------------------------------------------------------------------- arm metadata
def arm_meta(arm: str) -> Dict[str, Any]:
    """Definition, ``start_weights``, ``local_first`` (default 1) and ``share`` of an arm from the launcher tables.

    Raises:
        SystemExit: if the arm is in neither ``R2C_WAVE_S_ARMS`` nor ``R2B_WAVEA_ARMS``.
    """
    t = L.R2C_WAVE_S_ARMS.get(arm) or L.R2B_WAVEA_ARMS.get(arm)
    if t is None:
        raise SystemExit(f"unknown arm {arm!r}: not in R2C_WAVE_S_ARMS {list(L.R2C_WAVE_S_ARMS)} or "
                         f"R2B_WAVEA_ARMS {list(L.R2B_WAVEA_ARMS)}")
    sw = (t.get("r2b") or {}).get("start_weights") or None
    return {"definition": str(t.get("definition", "")), "start_weights": sw,
            "local_first": int(sw.get("local_first", 1)) if sw else 1,
            "share": float(sw["peak_share"]) if sw else 0.0}


# --------------------------------------------------------------------------- run directories and extraction
@dataclass(frozen=True)
class R2cCtx(R.Ctx):
    """Maps an arm name to its run directory (wave dir, R1 ``parents_A``, R2b reference arms)."""

    wave_dir: str = ""
    r1_root: str = ""
    r2b_root: str = ""

    def run_dir(self, wave: str, q: int, seed: int, arm: str) -> Path:
        """Run directory of ``arm`` at (q, seed); ``wave`` is the R1 wave label and is ignored."""
        if arm == BASE:
            return Path(self.r1_root) / "parents_A" / f"q{q}" / f"seed{seed}"
        if arm.startswith(REF_PREFIX):
            return Path(self.r2b_root) / "waveA" / f"q{q}" / f"seed{seed}" / arm[len(REF_PREFIX):]
        return Path(self.wave_dir) / f"q{q}" / f"seed{seed}" / arm


def extract(ctx: R2cCtx, arms: Sequence[str], qs: Sequence[int], seeds: Sequence[int], workers: int
            ) -> pd.DataFrame:
    """Per-run table of the baseline, the arms (and reference arms) and the parent pseudo-arm, R2b extras included."""
    df, _ = R.extract_wave(ctx, "stage2", qs, seeds, [BASE] + list(arms), workers=workers)
    extra_rows = [B.run_extras(ctx, row) for row in df.to_dict("records")]
    df = df.reset_index(drop=True)
    ex = pd.DataFrame(extra_rows, index=df.index)
    for c in ex.columns:                                      # an extra overrides the R1 column of the same name
        if c in df.columns:
            df[c] = df[c].astype(object)
            m = ex[c].notna()
            df.loc[m, c] = ex.loc[m, c].astype(object)
            try:                                              # numeric columns stay numeric
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
        else:
            df[c] = ex[c]
    df = R.prepare(df)
    df["tail_peak_ok"] = df[PRIMARY].astype(float) <= TAIL_PEAK
    df["tail_eta_over_margin"] = df["eta_T_over_dw"].astype(float) > B.TAIL_ETA
    return df


# --------------------------------------------------------------------------- selection inputs
def selection_inputs(df: pd.DataFrame, crit: pd.DataFrame, arms: Sequence[str], qs: Sequence[int],
                     seeds: Sequence[int], meta: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per-arm inputs of :func:`select_arm` with the criterion details (CIs, violations) for the record.

    Args:
        df: Per-run table (columns arm, q, seed, status, complete, the primary metric).
        crit: Criterion table of ``refine_analysis.criterion_table`` (rows ``vs baseline`` per arm).
        arms: Arms of the selection (not the baseline and not the reference arms).
        qs: q values.
        seeds: Development seeds.
        meta: ``arm_meta`` of each arm (``local_first``, ``share``).
    """
    out: List[Dict[str, Any]] = []
    for arm in arms:
        c = crit[(crit.arm == arm) & (crit.comparison == CRIT_LABEL)]
        c = c.iloc[0] if len(c) else None
        g = df[(df.arm == arm) & df.complete]
        not_done = [f"q{int(r.q)}/{int(r.seed)} ({r.status})"
                    for r in df[(df.arm == arm) & ~df.complete].sort_values(["q", "seed"]).itertuples()]
        planned = {(q, s) for q in qs for s in seeds}
        have = set(zip(df[df.arm == arm]["q"].astype(int), df[df.arm == arm]["seed"].astype(int)))
        not_done += [f"q{q}/{s} (missing)" for q, s in sorted(planned - have)]
        t: Dict[str, Any] = {"arm": arm}
        t["complete"] = bool(c is not None and not not_done and bool(c["a_complete"]))
        t["not_done_runs"] = not_done
        t["a"] = {f"q{q}": {"n_pairs": int(c[f"n_pairs_q{q}"]), "mean": float(c[f"mean_q{q}"]),
                            "ci_lo": float(c[f"ci_mean_lo_q{q}"]), "ci_hi": float(c[f"ci_mean_hi_q{q}"]),
                            "met": bool(c[f"a_q{q}"])} if c is not None else
                  {"n_pairs": 0, "mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "met": False}
                  for q in qs}
        t["a_met"] = bool(c is not None and bool(c["a_met"]))
        t["b_status"] = str(c["b_status"]) if c is not None else "incomplete"
        t["b_violations"] = str(c["b_violations"]).split() if c is not None and isinstance(c["b_violations"], str) else []
        t["b_pending"] = str(c["b_pending"]).split() if c is not None and isinstance(c["b_pending"], str) else []
        vals = {q: R._num(g[g.q == q].sort_values("seed")[PRIMARY]).dropna() for q in qs}
        t["n_runs_complete"] = {f"q{q}": int(len(vals[q])) for q in qs}
        t["n_abs_peak_le_0.05"] = {f"q{q}": count_le(vals[q]) for q in qs}
        t["n_abs_peak_le_0.05_total"] = int(sum(t["n_abs_peak_le_0.05"].values()))
        allv = np.concatenate([vals[q].to_numpy() for q in qs]) if qs else np.array([])
        t["mean_abs_peak_error"] = float(allv.mean()) if allv.size else float("nan")
        t["local_first"] = int(meta[arm]["local_first"])
        t["share"] = float(meta[arm]["share"])
        t["ineligible_reasons"] = ineligible_reasons(t)
        t["eligible"] = not t["ineligible_reasons"]
        out.append(t)
    return out


def inputs_frame(inputs: Sequence[Mapping[str, Any]], boot_seed: int, qs: Sequence[int]) -> pd.DataFrame:
    """Flat table of the selection inputs (the CSV of ``selection.json``'s ``arms``)."""
    rows = []
    for t in inputs:
        r: Dict[str, Any] = {"arm": t["arm"], "local_first": t["local_first"], "share": t["share"],
                             "complete": t["complete"]}
        for q in qs:
            a = t["a"][f"q{q}"]
            r.update({f"a_mean_q{q}": a["mean"], f"a_ci_lo_q{q}": a["ci_lo"], f"a_ci_hi_q{q}": a["ci_hi"],
                      f"a_q{q}": a["met"]})
        r.update({"a_met": t["a_met"], "b_status": t["b_status"], "b_violations": " ".join(t["b_violations"])})
        for q in qs:
            r[f"n_abs_peak_le_0.05_q{q}"] = t["n_abs_peak_le_0.05"][f"q{q}"]
        r.update({"n_abs_peak_le_0.05_total": t["n_abs_peak_le_0.05_total"],
                  "mean_abs_peak_error": t["mean_abs_peak_error"], "eligible": t["eligible"],
                  "ineligible_reasons": "; ".join(t["ineligible_reasons"]), "boot_seed": boot_seed})
        rows.append(r)
    return pd.DataFrame(rows)


def json_clean(o: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, NaN / inf to null."""
    if isinstance(o, dict):
        return {str(k): json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def selection_record(inputs: Sequence[Mapping[str, Any]], sel: Mapping[str, Any], args: argparse.Namespace,
                     meta: Mapping[str, Mapping[str, Any]], ref_used: Sequence[str]) -> Dict[str, Any]:
    """The content of ``selection.json``."""
    incomplete = [t["arm"] for t in inputs if not t["complete"]]
    return json_clean({
        "rule": RULE_TEXT, "interpretation": INTERPRETATION, "sort_key_description": SORT_DESCRIPTION,
        "boot_seed": int(args.boot_seed), "n_boot": N_BOOT,
        "roots": {"root": args.root, "wave_dir": args.wave_dir, "r1_root": args.r1_root, "ref_root": args.ref_root,
                  "r2b_root": args.r2b_root if args.r2b_root and Path(args.r2b_root).exists() else None,
                  "out": args.out, "reports": args.reports},
        "arms": list(inputs),
        "ranking": sel["ranking"], "ineligible": sel["ineligible"],
        "selected": sel["selected"],
        "selected_start_weights": meta[sel["selected"]]["start_weights"] if sel["selected"] else None,
        "outcome": sel["outcome"], "decided_by": sel["decided_by"], "reason": sel["reason"],
        "all_arms_complete": not incomplete, "incomplete_arms": incomplete,
        "reference_arms_excluded_from_selection": list(ref_used)})


# --------------------------------------------------------------------------- tables
def completeness(df: pd.DataFrame, args: argparse.Namespace, arms: Sequence[str], refs: Sequence[str]
                 ) -> pd.DataFrame:
    """Runs done / failed / incomplete / missing per arm and q (baseline and reference arms from their own roots)."""
    wave_root, wave_name = str(Path(args.wave_dir).parent), Path(args.wave_dir).name
    rows = []
    for arm in [BASE] + list(arms) + list(refs):
        if arm == BASE:
            src, root, wave = "R1 parents_A", args.r1_root, "parents_A"
        elif arm in refs:
            src, root, wave = "R2b waveA (reference)", args.r2b_root, "waveA"
        else:
            src, root, wave = "this round", wave_root, wave_name
        t = R.completeness_table(df, [arm], QS, SEEDS, Path(root), wave)
        t.insert(1, "source", src)
        if src != "this round":
            t["set_aside_originals"] = float("nan")
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


@dataclass
class Result:
    """All tables of the analysis."""

    df: pd.DataFrame
    tabs: Dict[str, pd.DataFrame]
    comps: List[Tuple[str, str, str]]
    arms: List[str]
    refs: List[str]
    meta: Dict[str, Dict[str, Any]]
    inputs: List[Dict[str, Any]]
    selection: Dict[str, Any]


def compute(args: argparse.Namespace) -> Result:
    """Extract every planned run, compute the pre-registered tables and the selection, write CSV / JSON."""
    R.BOOT_SEED = int(args.boot_seed)      # before any statistic (the R2b module sets 20261004 at import)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = list(args.arms)
    refs = [REF_PREFIX + a for a in REF_ARMS] if args.r2b_root and Path(args.r2b_root).is_dir() else []
    meta = {a: arm_meta(a) for a in arms}
    meta.update({REF_PREFIX + a: arm_meta(a) for a in REF_ARMS})
    ctx = R2cCtx(root=args.root, ref_root=args.ref_root, wave_dir=args.wave_dir, r1_root=args.r1_root,
                 r2b_root=args.r2b_root or "")
    df = extract(ctx, arms + refs, QS, SEEDS, args.workers)
    comps = [(a, BASE, CRIT_LABEL) for a in arms] + [(r, BASE, REF_LABEL) for r in refs]
    paired, seed_level = R.paired_tables(df, "stage2", comps, QS, SEEDS)
    seed_level = seed_level.reindex(columns=SEED_LEVEL_COLS)      # no column at all when no arm run is complete
    crit = R.criterion_table(df, "stage2", paired, comps, QS, SEEDS)
    for t in (paired, crit):
        t["boot_seed"] = int(args.boot_seed)
        t["n_boot"] = N_BOOT
    arms_all = [BASE] + arms + refs
    inputs = selection_inputs(df, crit, arms, QS, SEEDS, meta)
    sel = select_arm(inputs)
    tabs: Dict[str, pd.DataFrame] = {
        "per_run": df, "paired": paired, "paired_seed_level": seed_level, "seed_table": B.seed_table(df, comps, QS, SEEDS),
        "criterion": crit, "tail": B.tail_table(df, arms_all + [PARENT], QS),
        "arm_summary": R.arm_summary(df, "stage2", arms_all + [PARENT], QS),
        "gate_counts": R.gate_counts_table(df, "stage2", arms_all, QS),
        "optimisation": R.opt_table(df, "stage2", arms_all, QS),
        "cost": R.cost_table(df, "stage2", arms_all, BASE),
        "completeness": completeness(df, args, arms, refs),
        "manifest_commits": R.manifest_commit_table(df),
        "selection_inputs": inputs_frame(inputs, int(args.boot_seed), QS),
        "ranking": pd.DataFrame([{**{k: v for k, v in r.items() if k != "sort_key"},
                                  "sort_key": " ".join(f"{x:.12g}" for x in r["sort_key"]),
                                  "boot_seed": int(args.boot_seed)} for r in sel["ranking"]]),
    }
    tabs["locfree_argmax_per_run"], tabs["locfree_argmax_summary"] = B.locfree_tables(df, arms_all + [PARENT], QS)
    for name, t in tabs.items():
        t.to_csv(out / f"{name}.csv", index=False)
    record = selection_record(inputs, sel, args, meta, refs)
    with open(out / "selection.json", "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    info = {"boot_seed": int(args.boot_seed), "n_boot": N_BOOT, "arms": arms, "reference_arms": refs,
            "roots": record["roots"], "qs": list(QS), "seeds": list(SEEDS)}
    with open(out / "analysis_info.json", "w") as f:
        json.dump(info, f, indent=1)
        f.write("\n")
    return Result(df, tabs, comps, arms, refs, meta, inputs, sel)


# --------------------------------------------------------------------------- report helpers
def _seeded(t: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Display copy of a bootstrap table with the boot seed recorded in the CSV."""
    t = t.copy()
    if len(t):
        t["boot_seed"] = int(args.boot_seed)
    return t


def _read_json(p: Path) -> Any:
    with open(p) as f:
        return json.load(f)


def launch_rows(wave_dir: Path) -> pd.DataFrame:
    """One row per launch record ``<wave-dir>/launch_*.json`` (and a count of the dry-run records)."""
    rows = []
    for p in sorted(wave_dir.glob("launch_*.json")):
        j = _read_json(p)
        runs = j.get("runs", [])
        rows.append({"record": R._rel(p), "started": j.get("started"), "argv": " ".join(j.get("argv", [])),
                     "workers": j.get("workers"), "nproc": j.get("nproc"),
                     "loadavg_at_start": str([round(x, 2) for x in j.get("loadavg_at_start", [])]),
                     "free disk (TB)": round(j.get("disk_free_bytes", 0) / 1e12, 2), "head": str(j.get("head", ""))[:7],
                     "code_commit": str(j.get("code_commit", "")),
                     "code_commit_resolved": str(j.get("code_commit_resolved", ""))[:7], "state": j.get("state"),
                     "n_planned": j.get("n_planned"),
                     "finished / return code 0": f"{len(runs)} / {sum(1 for r in runs if r.get('returncode') == 0)}"})
    return pd.DataFrame(rows)


def checks_section(doc: R.Doc, args: argparse.Namespace) -> None:
    """The launch checks and the C-R3 reproduction check, printed before any analysis."""
    doc.h(2, "1. Checks")
    for fname, label in (("launch_checks.json", "Launch checks (`tools/v2/r2c_launch_checks.py`)"),
                         ("v20_reproduction_checks.json", "C-R3: the unchanged v2.0 entry point against the reference")):
        p = Path(args.root) / fname
        if not p.exists():
            doc.p(f"**{label}**: `{R._rel(p)}` is not present.")
            continue
        j = _read_json(p)
        doc.p(f"**{label}**: `{R._rel(p)}`.")
        if isinstance(j, dict) and "summary" in j:
            doc.lines += ["```json", json.dumps(j["summary"], indent=1), "```", ""]
        elif isinstance(j, dict):
            doc.p("No top-level `summary`; top-level keys: " + ", ".join(f"`{k}`" for k in j) + ".")
            scal = [{"key": k, "value": v} for k, v in j.items() if isinstance(v, (str, int, float, bool)) or v is None]
            if scal:
                doc.lines += [R.md_table(pd.DataFrame(scal)), ""]
        else:
            doc.lines += ["```json", json.dumps(j, indent=1)[:2000], "```", ""]


def provenance_head(doc: R.Doc, res: Result, args: argparse.Namespace) -> None:
    """Roots, boot seed and the launch records."""
    doc.h(2, "2. Provenance")
    doc.lines += [f"- wave runs: `{args.wave_dir}` (`--wave-dir`)", f"- result root: `{args.root}`",
                  f"- baseline `{BASE}` = R1 `parents_A`: `{args.r1_root}`", f"- rehearsal parent candidates: `{args.ref_root}`",
                  "- R2b reference rows (descriptive): " + (f"`{args.r2b_root}`" if res.refs else
                                                           "skipped (`--r2b-root` not given or does not exist)"),
                  f"- bootstrap: {N_BOOT} resamples, `numpy.random.default_rng({args.boot_seed})` per (q, statistic)", ""]
    doc.p("This report contains no generation time (the tool is deterministic given its inputs); the `started` column "
          "below is copied from the launch records.")
    lr = launch_rows(Path(args.wave_dir))
    if len(lr):
        doc.p("Launch records of this wave (`tools/v2/launch_refine.py`):")
        doc.table(lr, "waveS_launch_record")
    else:
        doc.p(f"No launch record `launch_*.json` found under `{args.wave_dir}`.")


def completeness_method(doc: R.Doc, res: Result, args: argparse.Namespace) -> None:
    """Manifest commits, completeness and the statistics used by every table."""
    doc.h(2, "3. Completeness, manifests, method")
    doc.p("Commit and dirty flag recorded in `manifest.json` of the analysed runs (all arms, the baseline and the "
          "reference arms):")
    doc.table(res.tabs["manifest_commits"], "manifest_commits")
    doc.p("Completeness (a run is complete iff `status.json` says done with exit code 0 and `final_v2.json` holds a "
          "final-tier evaluation; every planned run is listed):")
    comp = res.tabs["completeness"]
    doc.table(comp, "completeness",
              disp=comp[["arm", "source", "q", "planned", "done", "failed", "incomplete", "missing", "not_done_seeds"]])
    doc.p("Statistics (`refine_r2b_analysis.py`, pre-registration section 6 of R2b): differences are arm - A_base, paired by "
          "(q, seed); `n_better` counts strictly improving seeds and `n_zero` exact ties; the 95% CIs are percentile "
          f"bootstrap intervals of the mean and of the median of the 10 paired differences ({N_BOOT} resamples with "
          f"replacement, `numpy.random.default_rng({args.boot_seed})`, one fresh generator per (q, statistic) in table "
          "order); the primary metric is the absolute signed peak error at d = 0 of the final-tier end-of-A last "
          "iterate (improvement = negative difference). Interpretation used by the selection (also in `03_selection.md`): "
          "|peak error| = that primary metric; runs with |peak error| <= 0.05 are counted inclusively over the 20 runs "
          "of an arm; the mean |peak error| is the mean over those 20 runs; an arm with any missing, failed or incomplete "
          "run cannot be selected.")


def definition_text(arm: str, meta: Mapping[str, Any]) -> str:
    """Definition and start_weights of an arm."""
    sw = json.dumps(meta["start_weights"]) if meta["start_weights"] else "none (bin-balanced)"
    return (f"{meta['definition']}. `start_weights` = `{sw}`; local_first (default 1) = {meta['local_first']}; "
            f"share = {meta['share']:g}.")


def arm_section(doc: R.Doc, res: Result, arm: str, args: argparse.Namespace, figs: Sequence[Tuple[str, List[str]]],
                fig_dir: Path, rep_dir: Path) -> None:
    """One section of the report: definition, paired tables, criterion, tails, other metrics, optimisation, figure."""
    paired, crit, df = res.tabs["paired"], res.tabs["criterion"], res.df
    comps = [c for c in res.comps if c[0] == arm]
    doc.h(3, f"`{arm}`")
    doc.p(definition_text(arm, res.meta[arm]) + f" Comparator of the criterion: `{BASE}` (label `{CRIT_LABEL}`).")
    doc.p("Primary metric, paired differences arm - A_base (negative = better):")
    doc.table(_seeded(B.comp_rows(paired, comps, PRIMARY), args), f"waveS_{arm}_primary")
    doc.p("Criterion, parts (a) and (b) separately:")
    doc.table(_seeded(B.crit_display(crit[crit.arm == arm]), args), f"waveS_{arm}_criterion")
    inp = next(t for t in res.inputs if t["arm"] == arm)
    doc.p(f"Selection inputs of this arm: complete = {inp['complete']}; runs with |peak error| <= {TAIL_PEAK}: "
          + " + ".join(f"{inp['n_abs_peak_le_0.05'][f'q{q}']} (q{q})" for q in QS)
          + f" = {inp['n_abs_peak_le_0.05_total']}; mean |peak error| over both q = {R._g(inp['mean_abs_peak_error'])}; "
          f"eligible = {inp['eligible']}" + (f" (not eligible because: {'; '.join(inp['ineligible_reasons'])})"
                                           if inp["ineligible_reasons"] else "") + ".")
    if inp["b_violations"]:
        doc.p("Part (b) violations (runs that passed G-A and its G-N part under the baseline and fail under the arm): "
              + B.violation_text(df, arm, " ".join(inp["b_violations"])) + ".")
    t = res.tabs["tail"]
    doc.p("Tail statistics of the arm and of the baseline (max |peak error|, runs with |peak error| <= 0.05, runs with "
          "eta_2/DW > 0.004, G-A passes):")
    doc.table(t[t.arm.isin([arm, BASE])], f"waveS_{arm}_tail")
    doc.p("Other metrics against the baseline (mean difference arm - A_base with the 95% CI of the mean; `k/10 better` "
          "counts seeds with a smaller value; the signed peak error, sigma_2(0) and the smoothed-game share have no "
          "direction; because the signed peak errors are negative, a POSITIVE difference of the signed error is an "
          "improvement):")
    doc.table(B.other_metrics_table(paired, arm, BASE, CRIT_LABEL), f"waveS_{arm}_metrics")
    o = res.tabs["optimisation"]
    doc.p("Optimisation diagnostics (median over the runs; KL, clip fraction, gradient norms, steps, wall time):")
    doc.table(o[o.arm == arm].drop(columns=["n_actor_steps_total", "median_n_actor_steps_total"], errors="ignore"),
              f"waveS_{arm}_optimisation")
    cols = ["arm", "q", "peak_visit_share", "peak_visit_design_share", "d1_flagged_L_s2", "d1_flagged_L_s2_in",
            "tail2q_mean_e2hat", "tail2q_mean_abs_err_over_g2_0", "offpath_delta2_max_over_dw", "onpath_delta2_max_over_dw"]
    sp = df[df.arm.isin([BASE, arm]) & df.complete][[c for c in cols + ["seed"] if c in df.columns]]
    gm = (sp.groupby(["arm", "q"]).mean(numeric_only=True).reset_index().drop(columns=["seed"], errors="ignore")
          .dropna(axis=1, how="all"))
    doc.p("Peak-set visitation share of the learner's stage-2 starts (cumulative over the whole phase, from the last "
          "verifier call; the design share is the bin-balanced one) with the D1 clamp counts and the stage-2 profile on "
          "|d| >= 2q (mean over the seeds; the baseline for comparison). For a late arm the whole-phase share mixes the "
          "bin-balanced and the peak-focused updates:")
    doc.table(gm, f"waveS_{arm}_specifics")
    if figs:
        doc.figures(list(figs), fig_dir, rep_dir)
    ev = B.arm_notable(df, arm, BASE)
    doc.p("Notable events of this arm (computed): " + ("; ".join(ev) if ev else "none."))


def arm_index(res: Result) -> pd.DataFrame:
    """One row per arm: definition keys, primary difference per q with CI, criterion (a) / (b), tails."""
    paired, crit, tail = res.tabs["paired"], res.tabs["criterion"], res.tabs["tail"]
    rows = []
    for arm in res.arms:
        row: Dict[str, Any] = {"arm": arm, "local_first": res.meta[arm]["local_first"], "share": res.meta[arm]["share"]}
        for q in QS:
            r = paired[(paired.arm == arm) & (paired.baseline == BASE) & (paired.q == q) & (paired.metric == PRIMARY)
                       & (paired.comparison == CRIT_LABEL)]
            if len(r):
                r = r.iloc[0]
                row[f"q{q} mean diff [95% CI]"] = f"{R._g(r['mean'])} {R.ci_str(r['ci_mean_lo'], r['ci_mean_hi'])}"
                row[f"q{q} better/ties of {int(r['n_pairs'])}"] = f"{B._int(r['n_better'])}/{B._int(r['n_zero'])}"
        c = crit[(crit.arm == arm) & (crit.baseline == BASE)]
        if len(c):
            c = c.iloc[0]
            row["(a) both q"] = bool(c["a_met"])
            row["(b)"] = c["b_status"] + (f" ({B.violation_text(res.df, arm, c['b_violations'])})" if c["b_violations"]
                                          else "")
        for q in QS:
            t = tail[(tail.arm == arm) & (tail.q == q)]
            if len(t):
                t = t.iloc[0]
                row[f"q{q} max|peak|, n<=0.05, n eta>0.004"] = (
                    f"{R._g(t['max_abs_peak_error'])}, {t['n_abs_peak_le_0.05']}, {t['n_eta_over_0.004']}")
        inp = next(x for x in res.inputs if x["arm"] == arm)
        row["n<=0.05 total"] = inp["n_abs_peak_le_0.05_total"]
        row["eligible"] = inp["eligible"]
        rows.append(row)
    return pd.DataFrame(rows)


def response_table(res: Result) -> pd.DataFrame:
    """Share / local_first response: baseline, this round's arms and the R2b reference arms ordered by (local_first, share)."""
    paired, tail, df = res.tabs["paired"], res.tabs["tail"], res.df
    items = [(BASE, "baseline (R1 parents_A)", 0, 0.0)]
    members = [(a, "this round", res.meta[a]["local_first"], res.meta[a]["share"]) for a in res.arms]
    members += [(r, "R2b reference", res.meta[r]["local_first"], res.meta[r]["share"]) for r in res.refs]
    items += sorted(members, key=lambda x: (x[2], x[3], x[0]))
    rows = []
    for arm, src, lf, share in items:
        row: Dict[str, Any] = {"arm": arm, "source": src, "local_first": "-" if arm == BASE else lf,
                               "share": "bin-balanced" if arm == BASE else share}
        for q in QS:
            r = paired[(paired.arm == arm) & (paired.baseline == BASE) & (paired.q == q) & (paired.metric == PRIMARY)]
            if len(r):
                r = r.iloc[0]
                row[f"q{q} mean diff [95% CI]"] = f"{R._g(r['mean'])} {R.ci_str(r['ci_mean_lo'], r['ci_mean_hi'])}"
        for q in QS:
            t = tail[(tail.arm == arm) & (tail.q == q)]
            if len(t):
                row[f"q{q} n<=0.05"] = int(t["n_abs_peak_le_0.05"].iloc[0])
        g = df[(df.arm == arm) & df.complete]
        if len(g):
            v = R._num(g[PRIMARY])
            row["n<=0.05 total"] = count_le(v)
            row["mean |peak|, both q"] = float(v.mean())
            row["max |peak|"] = float(v.max())
        rows.append(row)
    return pd.DataFrame(rows)


def cross_arm_tables(doc: R.Doc, res: Result, args: argparse.Namespace) -> None:
    """The cross-arm tables, the response table, the other metrics and the figure."""
    paired, crit, df = res.tabs["paired"], res.tabs["criterion"], res.df
    arms_tail = [BASE] + res.arms + res.refs + [PARENT]
    doc.h(3, "Arms at a glance")
    doc.table(arm_index(res), "waveS_arm_index")
    doc.h(3, "Share and local_first response (including the R2b reference rows)")
    doc.p("Baseline, the four arms of this round and, as descriptive reference rows, R2b's `A_peak25` (share 0.25) and "
          "`A_peak50` (share 0.50, local_first 1: the same sampler as a local_first = 1 arm of this round), ordered by "
          "(local_first, share). The reference rows are never inputs of the selection."
          + ("" if res.refs else " They were skipped: `--r2b-root` was not given or does not exist."))
    doc.table(response_table(res), "waveS_response")
    doc.h(3, "Primary metric, every comparison")
    doc.table(_seeded(B.comp_rows(paired, res.comps, PRIMARY), args), "waveS_primary", note="`paired.csv`")
    doc.h(3, "Criterion, every arm")
    doc.table(_seeded(B.crit_display(crit), args), "waveS_criterion", note="`criterion.csv`")
    doc.h(3, "Tail statistics (next to the criterion)")
    t = res.tabs["tail"]
    doc.p(f"Per arm and q over the complete runs: the maximum |peak error|, the number of runs with |peak error| <= "
          f"{TAIL_PEAK} (inclusive), the number with eta_2/DW > {B.TAIL_ETA} (the G-A limit is 0.005), the number "
          "that pass G-A with its G-N part, the mean |peak error|, the largest tail mean / e2*(0) and the number of "
          "runs above the G-A limit 0.02 of that tail mean. `parent_u1600` is the rehearsal end-of-A parent candidate.")
    doc.table(t[t.arm.isin(arms_tail)], "tail")
    doc.h(3, "Other metrics (paired, arm - A_base)")
    for m, lab in B.SECONDARY:
        sub = B.comp_rows(paired, res.comps, m)
        if len(sub):
            doc.p(f"**{lab}** (`{m}`)")
            doc.table(sub, f"waveS_metric_{m}")
    doc.h(3, "Location-free peak and its argmax d")
    doc.p("Per arm and q: the median |argmax d| of the stage-2 mean over the recovery grid, its range and the number of runs "
          "whose maximum is exactly at d = 0:")
    doc.table(res.tabs["locfree_argmax_summary"], "waveS_locfree_argmax")
    doc.h(3, "Peak-set visitation share and clamp counts (mean over the seeds)")
    cols = ["arm", "q", "peak_visit_share", "peak_visit_design_share", "d1_flagged_L_s2", "d1_flagged_L_s2_in",
            "tail2q_mean_e2hat", "tail2q_mean_abs_err_over_g2_0", "offpath_delta2_max_over_dw", "onpath_delta2_max_over_dw"]
    sp = df[df.arm.isin([BASE] + res.arms + res.refs) & df.complete][[c for c in cols + ["seed"] if c in df.columns]]
    gm = (sp.groupby(["arm", "q"]).mean(numeric_only=True).reset_index().drop(columns=["seed"], errors="ignore")
          .dropna(axis=1, how="all"))
    doc.table(gm, "waveS_specifics_mean")
    doc.h(3, "Optimisation diagnostics and cost per run")
    o = res.tabs["optimisation"]
    doc.table(o.drop(columns=["n_actor_steps_total", "median_n_actor_steps_total"], errors="ignore"), "waveS_optimisation")
    doc.table(res.tabs["cost"].drop(columns=["mean_n_actor_steps_total", "wall_ratio_vs_matched_control"],
                                    errors="ignore").dropna(axis=1, how="all"), "waveS_cost",
              note="mean over the complete runs of both q; wall time is machine-load dependent")
    doc.h(3, "Seed-level values")
    st = res.tabs["seed_table"]
    doc.table(st.dropna(axis=1, how="all"), "waveS_seed_level")


def anomalies_section(doc: R.Doc, res: Result, args: argparse.Namespace) -> None:
    """Anomalies of the wave: extraction anomalies, incomplete runs, missing roots, computed notable events."""
    df = res.df
    sub = df[df.arm.isin([BASE] + res.arms + res.refs)]
    n_an = int(sub["anomalies"].fillna("").astype(str).str.len().gt(0).sum()) if "anomalies" in sub else 0
    n_an2 = int(sub["anomalies_r2b"].fillna("").astype(str).str.len().gt(0).sum()) if "anomalies_r2b" in sub else 0
    n_done = int((sub["status"] == "done").sum())
    doc.p(f"The extraction recorded anomalies in {n_an} (`anomalies` column of `per_run.csv`) and {n_an2} (`anomalies_r2b`, a "
          f"column that exists only when it is non-empty) of the {len(sub)} rows of the arms, the baseline and the reference "
          f"arms ({n_done} with status done).")
    miss = []
    for lab, p in (("wave dir", args.wave_dir), ("R1 parents_A", str(Path(args.r1_root) / "parents_A")),
                   ("ref root", args.ref_root)):
        if not Path(p).is_dir():
            miss.append(f"{lab} `{p}`")
    if args.r2b_root and not Path(args.r2b_root).is_dir():
        miss.append(f"R2b reference root `{args.r2b_root}` (reference rows skipped)")
    doc.p("Roots that do not exist: " + ("; ".join(miss) if miss else "none."))
    bad = [t for t in res.inputs if not t["complete"]]
    if bad:
        doc.p("**INCOMPLETE ARMS (cannot be selected):** " + "; ".join(
            f"`{t['arm']}`: " + (", ".join(t["not_done_runs"]) or "paired values missing") for t in bad) + ".")
    else:
        doc.p("Every arm is complete (all planned runs done, all paired differences finite).")
    doc.p("Notable events computed from the tables, per arm:")
    for arm in res.arms + res.refs:
        ev = B.arm_notable(df, arm, BASE)
        doc.lines.append(f"- `{arm}`: " + ("; ".join(ev) if ev else "none."))
    doc.lines.append("")


def reproduce_command(args: argparse.Namespace) -> str:
    """The exact command that regenerates the CSVs, figures and reports."""
    parts = ["python tools/v2/refine_r2c_analysis.py", f"--root {R._rel(args.root)}"]
    if Path(args.wave_dir) != Path(args.root) / "waveS":
        parts.append(f"--wave-dir {args.wave_dir}")
    parts += ["--arms " + " ".join(args.arms), f"--r1-root {args.r1_root}", f"--ref-root {args.ref_root}"]
    if args.r2b_root:
        parts.append(f"--r2b-root {args.r2b_root}")
    parts += [f"--boot-seed {args.boot_seed}", f"--out {R._rel(args.out)}", f"--reports {R._rel(args.reports)}",
              f"--workers {args.workers}"]
    return " ".join(parts)


def commands_section(doc: R.Doc, args: argparse.Namespace) -> None:
    """Launch commands (from the records), the analysis command and the tests."""
    doc.h(2, "7. Commands")
    lr = launch_rows(Path(args.wave_dir))
    if len(lr):
        doc.p("Launch commands (`argv` of the launch records): " + "; ".join(f"`{a}`" for a in lr["argv"]) + ".")
    else:
        doc.p("No launch record found, so no launch command is shown.")
    doc.p("Analysis (single-threaded workers, `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, run from the "
          "repository root):")
    doc.lines += ["```", "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1", reproduce_command(args), "```", ""]
    doc.p("Tests of the selection rule (no result files needed): "
          "`python -m pytest tests/test_v2_r2c_analysis.py -p no:cacheprovider -q`.")


# --------------------------------------------------------------------------- reports
def write_pilot_report(res: Result, args: argparse.Namespace) -> Path:
    """``02_pilot_waveS.md``."""
    doc = R.Doc(Path(args.out))
    fig_dir, rep_dir = Path(args.figures), Path(args.reports)
    doc.h(1, "R2c pilot wave S: start-distribution arms (full Phase A from scratch)")
    doc.p("Candidate: the stage-2 last iterate at global update 1600 (final tier). Baseline: the R1 `parents_A` runs "
          f"(`{BASE}`). Arms: " + ", ".join(f"`{a}`" for a in res.arms) + "; each changes `start_weights` only, paired by "
          "(q, seed). Generated by `tools/v2/refine_r2c_analysis.py`; every number is read from a CSV that is cited "
          "under its table.")
    if any(not t["complete"] for t in res.inputs):
        doc.p("**INCOMPLETE: some arms have missing, failed or incomplete runs (section 6).**")
    checks_section(doc, args)
    provenance_head(doc, res, args)
    completeness_method(doc, res, args)
    doc.h(2, "4. One section per arm")
    figs = B.arm_figures(res, res.comps, fig_dir, "waveS", FIGURE_TITLE)   # type: ignore[arg-type]
    for arm in res.arms:
        arm_section(doc, res, arm, args, figs.get(arm, []), fig_dir, rep_dir)
    doc.h(2, "5. Cross-arm tables")
    cross_arm_tables(doc, res, args)
    prim = res.tabs["paired"]
    prim = prim[(prim.metric == PRIMARY)]
    files = R.fig_overview(fig_dir / "waveS_overview", prim, res.arms + res.refs, QS,
                           "Wave S and R2b reference: paired difference of |peak error| vs A_base",
                           "arm - A_base, |peak error| (negative = better)")
    doc.figures([("Wave S: paired difference of |peak error| per arm (R2b reference arms included)", files)],
                fig_dir, rep_dir)
    doc.h(2, "6. Anomalies")
    anomalies_section(doc, res, args)
    commands_section(doc, args)
    path = rep_dir / "02_pilot_waveS.md"
    path.write_text(doc.text())
    return path


def outcome_sentence(rec: Mapping[str, Any]) -> str:
    """The outcome sentence, generated from ``selection.json``."""
    if rec["selected"]:
        return f"Selected arm {rec['selected']}."
    if rec["outcome"] == "unresolved_tie":
        return ("The rule gives no answer: the first two eligible arms agree in all four components of its key; "
                "nothing is selected.")
    if rec["incomplete_arms"]:
        return ("No arm is eligible: arms with missing or incomplete runs cannot be selected; this outcome is not final.")
    return "No arm meets both parts of the criterion; the round stops after the pilot report."


def write_selection_report(res: Result, args: argparse.Namespace) -> Path:
    """``03_selection.md``: the rule verbatim, the inputs, the ranking and the outcome."""
    rec = _read_json(Path(args.out) / "selection.json")
    doc = R.Doc(Path(args.out))
    doc.h(1, "R2c selection (mechanical)")
    doc.p("Generated by `tools/v2/refine_r2c_analysis.py` from `" + R._rel(Path(args.out) / "selection.json") + "`.")
    doc.h(2, "Rule (the owner's text; emphasis and section pointers omitted)")
    doc.lines += ["> " + ln if ln else ">" for ln in rec["rule"].split("\n")] + [""]
    doc.h(2, "Interpretation implemented in `select_arm`")
    doc.lines += [f"- {s}" for s in rec["interpretation"]] + [f"- {rec['sort_key_description']}", ""]
    doc.p(f"Bootstrap seed used for every interval: {rec['boot_seed']} ({rec['n_boot']} resamples).")
    doc.h(2, "Inputs")
    inp = res.tabs["selection_inputs"]
    disp = pd.DataFrame([{
        "arm": r["arm"], "local_first": r["local_first"], "share": r["share"], "complete": r["complete"],
        **{f"(a) q{q}: mean [95% CI]": f"{R._g(r[f'a_mean_q{q}'])} {R.ci_str(r[f'a_ci_lo_q{q}'], r[f'a_ci_hi_q{q}'])}"
           f" ({'met' if r[f'a_q{q}'] else 'not met'})" for q in QS},
        "(a) both q": r["a_met"], "(b)": r["b_status"] + (f" ({r['b_violations']})" if r["b_violations"] else ""),
        **{f"n<=0.05 q{q}": r[f"n_abs_peak_le_0.05_q{q}"] for q in QS},
        "n<=0.05 total": r["n_abs_peak_le_0.05_total"], "mean |peak| both q": r["mean_abs_peak_error"],
        "eligible": r["eligible"]} for _, r in inp.iterrows()])
    doc.table(inp, "selection_inputs", disp=disp)
    bt = res.tabs["tail"]
    b_n = {q: int(bt[(bt.arm == BASE) & (bt.q == q)]["n_abs_peak_le_0.05"].iloc[0]) for q in QS if len(bt[(bt.arm == BASE) & (bt.q == q)])}
    if len(b_n) == len(QS):
        doc.p("For reference (not an arm of the selection): the baseline `A_base` has " + " + ".join(
            f"{b_n[q]} (q{q})" for q in QS) + f" = {sum(b_n.values())} runs with |peak error| <= {TAIL_PEAK}.")
    doc.p("Reasons an arm is not eligible: " + (" ".join(f"`{a}`: {'; '.join(rs)}." for a, rs in rec["ineligible"].items())
                                                 or "none, every arm is eligible."))
    doc.h(2, "Ranking of the eligible arms")
    rk = res.tabs["ranking"]
    if len(rk):
        doc.table(rk, "ranking", note="sort key = (-n runs <= 0.05, mean |peak|, -local_first, share)")
    else:
        doc.p("No eligible arm: the ranking is empty.")
    doc.h(2, "Outcome")
    doc.p(f"**{outcome_sentence(rec)}**")
    doc.p(f"Reason: {rec['reason']}")
    if rec["selected"]:
        doc.p(f"The selected arm's `start_weights` is the one pipeline change of v2.1: "
              f"`{json.dumps(rec['selected_start_weights'])}`.")
    if rec["incomplete_arms"]:
        doc.p("**INCOMPLETE ANALYSIS:** the arms " + ", ".join(f"`{a}`" for a in rec["incomplete_arms"]) +
              " have missing, failed or incomplete runs and cannot be selected; this outcome is not final.")
    doc.p("The rule was applied mechanically by `select_arm` in `tools/v2/refine_r2c_analysis.py` to the table above; "
          "no judgement entered between the table and the outcome.")
    path = Path(args.reports) / "03_selection.md"
    path.write_text(doc.text())
    return path


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    p = argparse.ArgumentParser(description="R2c pre-registered analysis of wave S and the selection rule")
    p.add_argument("--root", default=str(L.R2C_ROOT), help="result root of the round (default results/v2_refine_r2c)")
    p.add_argument("--wave-dir", default=None, help="directory that directly contains q*/seed*/<arm> (default <root>/waveS)")
    p.add_argument("--arms", nargs="+", default=None, help="default: the keys of launch_refine.R2C_WAVE_S_ARMS")
    p.add_argument("--r1-root", default=R1_ROOT, help="R1 root; the baseline is <r1-root>/parents_A")
    p.add_argument("--ref-root", default=str(L.REHEARSAL), help="rehearsal_v1_1 root (parent candidates)")
    p.add_argument("--r2b-root", default=R2B_REF_ROOT, help="R2b root for the reference arms (skipped if missing)")
    p.add_argument("--boot-seed", type=int, default=DEFAULT_BOOT_SEED)
    p.add_argument("--out", default=None, help="analysis tables dir (default <root>/analysis)")
    p.add_argument("--reports", default=str(HERE.parents[2] / "reports" / "v2" / "refine_r2c"))
    p.add_argument("--figures", default=None, help="default <reports>/figures")
    p.add_argument("--workers", type=int, default=8)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the analysis; write the CSVs, ``selection.json`` and the reports 02 and 03; exit 3 if an arm is incomplete."""
    a = build_parser().parse_args(argv)
    a.arms = list(a.arms) if a.arms else list(L.R2C_WAVE_S_ARMS)
    if len(set(a.arms)) != len(a.arms):
        raise SystemExit(f"duplicate arms: {a.arms}")
    for arm in a.arms:
        arm_meta(arm)
    a.root = str(Path(a.root).resolve())
    a.wave_dir = str(Path(a.wave_dir or Path(a.root) / "waveS").resolve())
    a.out = str(Path(a.out or Path(a.root) / "analysis").resolve())
    a.reports = str(Path(a.reports).resolve())
    a.figures = str(Path(a.figures or Path(a.reports) / "figures").resolve())
    a.r1_root, a.ref_root = str(Path(a.r1_root).resolve()), str(Path(a.ref_root).resolve())
    a.r2b_root = str(Path(a.r2b_root).resolve()) if a.r2b_root and a.r2b_root.lower() != "none" else ""
    Path(a.reports).mkdir(parents=True, exist_ok=True)
    res = compute(a)
    print(f"wrote {write_pilot_report(res, a)}", flush=True)
    print(f"wrote {write_selection_report(res, a)}", flush=True)
    print(f"selection: {res.selection['selected']!r} ({res.selection['outcome']}); {res.selection['reason']}", flush=True)
    bad = [t["arm"] for t in res.inputs if not t["complete"]]
    if bad:
        print("INCOMPLETE: arms " + ", ".join(bad) + " have missing, failed or incomplete runs and cannot be selected "
              "(see completeness.csv); exit code 3", file=sys.stderr, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
