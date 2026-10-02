"""The item list of the T=2 v2 report pack (§3 and §4 of the request): ids, titles, priority, order.

Single source for ``Pack`` (default titles / priority, outline order in the README) so that every
builder module registers items under the same definitions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# (id, type, priority, title)
_ITEMS: List[Tuple[str, str, str, str]] = [
    ("T01", "table", "core", "Study inventory"),
    ("T02", "table", "core", "Decision log"),
    ("T03", "table", "core", "Plan vs execution"),
    ("T04", "table", "core", "Parameters and derived constants"),
    ("F01", "figure", "core", "Closed-form benchmark"),
    ("T05", "table", "core", "Baseline audit findings"),
    ("T06", "table", "supp", "Resolved configuration of the locked v1.1 protocol"),
    ("T07", "table", "core", "Metric definitions"),
    ("T08", "table", "supp", "Verifier tiers"),
    ("T09", "table", "supp", "Invariant checks"),
    ("T10", "table", "supp", "dReach reach-mask check"),
    ("T11", "table", "supp", "Benchmark consistency"),
    ("T12", "table", "core", "Calibration"),
    ("F02", "figure", "supp", "Zero-effort deviation gains"),
    ("T13", "table", "core", "Flags"),
    ("T14", "table", "supp", "Snapshot drift"),
    ("T15", "table", "supp", "C7 regression"),
    ("T16", "table", "supp", "Test suite"),
    ("T17", "table", "supp", "RNG streams"),
    ("T18", "table", "core", "Reproducibility ledger"),
    ("T19", "table", "core", "Pilot design summary (Pilots 1-3)"),
    ("T20", "table", "core", "Pilot 1 final medians and IQR"),
    ("T21", "table", "core", "Pilot 1 paired differences (expected - sampled)"),
    ("F03", "figure", "core", "Pilot 1 learning curves"),
    ("F04", "figure", "core", "Pilot 1 stage-2 mapping against the closed form at u400"),
    ("F05", "figure", "supp", "Pilot 1 peak error against sigma_2(0)/q"),
    ("T22", "table", "core", "Pilot 2 final medians"),
    ("T23", "table", "core", "Pilot 2 paired comparisons"),
    ("T24", "table", "supp", "On-path definition and drift decomposition"),
    ("T25", "table", "supp", "Location of Gmax_full"),
    ("F06", "figure", "core", "Pilot 2 learning curves"),
    ("F07", "figure", "core", "Pilot 2 stage-2 mapping at the end of Phase B against the parent"),
    ("F08", "figure", "supp", "Advantage SD ratio"),
    ("T26", "table", "core", "Pilot 3 reproducibility (stochastic arm vs Pilot 2 B2)"),
    ("T27", "table", "core", "Pilot 3 final medians"),
    ("T28", "table", "core", "Pilot 3 paired differences (mean - stochastic)"),
    ("T29", "table", "core", "Pilot 3 stability"),
    ("F09", "figure", "core", "Pilot 3 learning curves"),
    ("F10", "figure", "supp", "Stage-1 effort trajectories"),
    ("T30", "table", "core", "Locked pipeline (v1.1)"),
    ("T31", "table", "core", "Criteria"),
    ("F11", "figure", "supp", "Learning-rate schedule"),
    ("T32", "table", "core", "Protocol history"),
    ("T33", "table", "supp", "Timeline (UTC)"),
    ("T34", "table", "supp", "Re-rehearsal checks R1-R6"),
    ("T35", "table", "core", "Confirmation verdict"),
    ("T36", "table", "supp", "Per-run confirmation table"),
    ("T37", "table", "core", "Distributions of the gate metrics"),
    ("T38", "table", "core", "S1 summary"),
    ("T39", "table", "core", "Reported metrics"),
    ("F12", "figure", "core", "Gate metrics against thresholds"),
    ("F13", "figure", "core", "Stage-2 mapping of all 20 runs per q"),
    ("F14", "figure", "core", "Stage-1 effort and its decomposition"),
    ("F15", "figure", "core", "EXP_root against the squared stage-1 error"),
    ("F16", "figure", "core", "Learning curves under the locked pipeline"),
    ("F17", "figure", "supp", "G_t(d) on D_t"),
    ("T40", "table", "supp", "Rehearsal vs confirmation"),
    ("T41", "table", "core", "Targets vs results"),
    ("T42", "table", "core", "Phase A extension"),
    ("F18", "figure", "core", "Extension learning curves"),
    ("T43", "table", "core", "Pilot 4 section 2a"),
    ("T44", "table", "supp", "Stage-2 tail averaging"),
    ("F19", "figure", "core", "Pilot 4 section 2a learning curves"),
    ("T45", "table", "core", "Smoothed-game share"),
    ("T46", "table", "core", "Representation floor"),
    ("T47", "table", "core", "Cusp diagnostic"),
    ("F20", "figure", "core", "Three-way peak gap"),
    ("F21", "figure", "core", "Supervised-fit trajectories"),
    ("T48", "table", "core", "Induced-target method"),
    ("F22", "figure", "core", "Stage-1 residual against effort"),
    ("T49", "table", "core", "Root stage game"),
    ("T50", "table", "core", "Anatomy of the stage-1 fluctuation"),
    ("F23", "figure", "supp", "ACF plot"),
    ("T51", "table", "core", "Pilot 4 section 2b"),
    ("F24", "figure", "core", "Pilot 4 section 2b learning curves"),
    ("T52", "table", "supp", "Pilot 4 section 6 distribution tables"),
    ("T53", "table", "core", "v1.0 rehearsal"),
    ("T54", "table", "supp", "v1.0 Check 1 and Check 2"),
    ("T55", "table", "supp", "Consolidation"),
    ("T56", "table", "core", "v1.1 global-RNG hardening"),
    ("T57", "table", "core", "Issue list"),
    ("D01", "data", "supp", "Per-run table: Pilot 1"),
    ("D02", "data", "supp", "Per-run table: Pilot 2"),
    ("D03", "data", "supp", "Per-run table: Pilot 3"),
    ("D04", "data", "supp", "Per-run table: Phase A extension"),
    ("D05", "data", "supp", "Per-run table: Pilot 4 section 2a"),
    ("D06", "data", "supp", "Per-run table: Pilot 4 section 2b"),
    ("D07", "data", "supp", "Per-run table: v1.0 rehearsal"),
    ("D08", "data", "supp", "Per-run table: v1.1 re-rehearsal"),
    ("D09", "data", "supp", "Per-run table: confirmation"),
    ("T58", "table", "supp", "Compute"),
    ("T59", "table", "supp", "Commands to reproduce"),
    ("K01", "number", "core", "Confirmation: primary passes per q, exact 95% CI and verdict"),
    ("K02", "number", "core", "Confirmation: Gmax_full/dW, median and max per q"),
    ("K03", "number", "core", "Confirmation: eta_2, median and max per q"),
    ("K04", "number", "core", "Confirmation: RMSE/e2*(0), median and max per q"),
    ("K05", "number", "core", "Confirmation: stage-2 tail mean, median and max per q"),
    ("K06", "number", "core", "Confirmation: peak error at d=0, median and range per q"),
    ("K07", "number", "core", "Confirmation: stage-1 error summary per q"),
    ("K08", "number", "core", "Confirmation: max |dev - final| for eta_2 and Gmax_full"),
    ("K09", "number", "core", "Confirmation: the v1.0 outcome"),
    ("K10", "number", "core", "Pilot 1: medians of eta_2 and peak error per arm and q, sign counts"),
    ("K11", "number", "core", "Pilot 2: drift on and off path; peak error and tail mean per arm"),
    ("K12", "number", "core", "Pilot 3: paired differences of the stage-1 metrics, within-run SD"),
    ("K13", "number", "core", "Phase A extension: peak error, tail mean and sigma_2(0) trends"),
    ("K14", "number", "core", "Pilot 4: decay - constant effects"),
    ("K15", "number", "core", "Smoothed-game share: medians across studies"),
    ("K16", "number", "core", "Supervised-fit floor of the peak error"),
    ("K17", "number", "core", "Calibration floors, zero-effort Gmax and agreement with the PI references"),
    ("K18", "number", "core", "Bit-identity checks passed out of the total"),
    ("K19", "number", "core", "Total runs and total CPU-hours over P0-P6"),
]

ITEMS: Dict[str, Dict[str, str]] = {i: {"type": t, "priority": p, "title": ti} for i, t, p, ti in _ITEMS}

OUTLINE: List[Tuple[str, List[str]]] = [
    ("Front matter", ["T01", "T02", "T03"]),
    ("Part I, section 2: Setting and baseline audit (P0)", ["T04", "F01", "T05", "T06"]),
    ("Part I, section 3: Method change (a), the full-domain MPE metric",
     ["T07", "T08", "T09", "T10", "T11", "T12", "F02"]),
    ("Part I, section 4: Method change (b), stagewise learning with frozen continuation",
     ["T13", "T14", "T15", "T16", "T17", "T18"]),
    ("Part I, section 5: Pilots",
     ["T19", "T20", "T21", "F03", "F04", "F05", "T22", "T23", "T24", "T25", "F06", "F07", "F08",
      "T26", "T27", "T28", "T29", "F09", "F10"]),
    ("Part I, section 6: Formal T=2 experiment, locked protocol and fresh-seed confirmation",
     ["T30", "T31", "F11", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "F12", "F13",
      "F14", "F15", "F16", "F17", "T40", "T41"]),
    ("Part II, section 7: Stage-2 accuracy beyond 400 updates", ["T42", "F18", "T43", "T44", "F19"]),
    ("Part II, section 8: Anatomy of the stage-2 peak gap", ["T45", "T46", "T47", "F20", "F21"]),
    ("Part II, section 9: Stage-1 accuracy", ["T48", "F22", "T49", "T50", "F23", "T51", "F24"]),
    ("Part II, section 10: From development distributions to pre-registered gates", ["T52", "T53"]),
    ("Part II, section 11: Protocol engineering and reproducibility", ["T54", "T55", "T56"]),
    ("Part II, section 12: Known issues and open questions (T=2 only)", ["T57"]),
    ("Appendices", ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "T58", "T59"]),
    ("Key numbers (key_numbers.csv)", [f"K{i:02d}" for i in range(1, 20)]),
]

SECTION_OF: Dict[str, str] = {i: sec for sec, ids in OUTLINE for i in ids}
ORDER: List[str] = [i for _, ids in OUTLINE for i in ids]
ORDER_INDEX: Dict[str, int] = {i: n for n, i in enumerate(ORDER)}


def slug(title: str, maxlen: int = 44) -> str:
    """File-name slug of a title: lowercase alphanumerics and underscores."""
    out, prev = [], True
    for ch in title.lower():
        if ch.isalnum():
            out.append(ch)
            prev = False
        elif not prev:
            out.append("_")
            prev = True
    s = "".join(out).strip("_")
    return s[:maxlen].rstrip("_")
