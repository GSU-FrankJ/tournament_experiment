"""Section 6 of the pack, part b: reported metrics, figures and targets of the confirmation.

Items: T39 (reported metrics), F12 (gate metrics against thresholds), F13 (stage-2 mapping of all
runs), F14 (stage-1 effort and its decomposition), F15 (EXP_root against the squared stage-1
error), F16 (learning curves), F17 (G_t(d) of two runs per q), T41 (0929 targets against results).

Inputs (read only): the 40 runs of the v1.1 confirmation
``results/v2_T2_locked/confirmation/q{50,60}/seed{20501..20520}/`` (gates.json, drift_test.json,
gateA_*.npz, final_*.npz, v2_checkpoints_{A,B}.csv), the pre-registered analysis output
``results/v2_T2_locked/confirmation_analysis/``, the locked protocol JSON (thresholds, constants)
and the lock-time calibration CSV. Every value is read from saved files; no forward pass or
verifier evaluation is performed (F16 uses the training-time dev-tier checkpoint CSVs, whose
update grid is common to all 40 runs). CSVs are parsed with round-trip float parsing so that
equality checks against JSON / NPZ values are exact.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import common as C
import style
import studies as S

MOD = "sec_locked_b.py"
ROOT = "results/v2_T2_locked/confirmation"
CA = "results/v2_T2_locked/confirmation_analysis"
PROTO = "protocols/v2_T2_locked_v1_1.json"
REPORT = "reports/v2/protocol_v1_1_confirmation.md"
CAL = "results/v2_T2_locked/calibration/calibration_locked.csv"
THEORY = "utils/theory_multistage.py"
QS: Tuple[int, ...] = (50, 60)
SEEDS: List[int] = list(S.SEEDS_CONF)
NOTE_CSV = "CSVs parsed with pandas float_precision='round_trip' (exact float64 values)."


# ----------------------------------------------------------------------------------------------
# shared inputs
# ----------------------------------------------------------------------------------------------

def _csv(rel: str) -> pd.DataFrame:
    """CSV with exact (round-trip) float parsing."""
    return pd.read_csv(C.abspath(rel), float_precision="round_trip")


def _rd(q: int, s: int) -> str:
    """Run directory of a confirmation run."""
    return S.run_dir("confirmation", q, s)


def _set(fname: str) -> C.SourceSet:
    """Provenance set of one per-run file over the 40 confirmation runs."""
    return C.srcs(f"{ROOT}/q*/seed*/{fname}", label=f"{ROOT}/q*/seed*/{fname}", expect=40)


def _fmt(x: float, digits: int = 4) -> str:
    """Short text of a number for notes and captions."""
    return C.fmt_cell(float(x), digits)


def _int_or_none(s: pd.Series) -> pd.Series:
    """Integer column with empty cells: object dtype with ints and None (empty in CSV and .md)."""
    return pd.Series([int(v) if pd.notna(v) else None for v in s], index=s.index, dtype=object)


class Ctx:
    """Inputs shared by the items, read once."""

    def __init__(self) -> None:
        self.proto: Dict[str, Any] = S.read_json(PROTO)
        self.per_run = (_csv(f"{CA}/per_run.csv")
                        .sort_values(["q", "seed"]).reset_index(drop=True))
        self.rep = (_csv(f"{CA}/reported_metrics.csv")
                    .sort_values(["q", "seed"]).reset_index(drop=True))
        self.gates = {(q, s): S.read_json(f"{_rd(q, s)}/gates.json")
                      for q in QS for s in SEEDS}
        self.drift = {(q, s): S.read_json(f"{_rd(q, s)}/drift_test.json")
                      for q in QS for s in SEEDS}
        self._npz: Dict[Tuple[int, int, str], Dict[str, np.ndarray]] = {}
        for df in (self.per_run, self.rep):
            keys = list(zip(df["q"].astype(int), df["seed"].astype(int)))
            if keys != [(q, s) for q in QS for s in SEEDS]:
                raise ValueError("per-run tables do not list the 40 confirmation runs in order")
        self.dw = {q: float(self.proto["records"][str(q)]["dw"]) for q in QS}
        self.e1s = {q: self._const(q, "end_of_B", "g1") for q in QS}
        self.e2s0 = {q: self._const(q, "end_of_A", "g2_at_0") for q in QS}
        self.th = self._thresholds()

    def _const(self, q: int, phase: str, key: str) -> float:
        vals = {float(self.gates[(q, s)]["reported"][phase]["final"][key]) for s in SEEDS}
        if len(vals) != 1:
            raise ValueError(f"{key} differs across runs at q={q}")
        return vals.pop()

    def _thresholds(self) -> Dict[str, float]:
        """Thresholds of G-A, G-F, G-N and S1 from the protocol JSON (all inclusive '<=')."""
        g = self.proto["gates"]
        crit = [c for blk in ("G-A", "G-F", "G-N") for c in g[blk]["all_must_hold"]]
        crit.append(self.proto["secondary"]["S1"]["criterion"])
        if any(c["op"] != "<=" for c in crit):
            raise ValueError("unexpected comparison operator in the protocol")
        return {c["metric"]: float(c["threshold"]) for c in crit}

    def npz(self, q: int, s: int, name: str) -> Dict[str, np.ndarray]:
        """Arrays of ``<run>/<name>.npz`` (cached)."""
        key = (q, s, name)
        if key not in self._npz:
            with np.load(C.abspath(f"{_rd(q, s)}/{name}.npz")) as z:
                self._npz[key] = {k: np.asarray(z[k]) for k in z.files}
        return self._npz[key]

    def rows(self, q: int) -> pd.DataFrame:
        """reported_metrics.csv rows of one q (seed order)."""
        return self.rep[self.rep["q"] == q].reset_index(drop=True)

    def pr(self, q: int) -> pd.DataFrame:
        """per_run.csv rows of one q (seed order)."""
        return self.per_run[self.per_run["q"] == q].reset_index(drop=True)


def _prose(pack: C.Pack, item: str, what: str, value: float, cell: str, where: str,
           passed: List[str]) -> None:
    """Compare a number in the report prose with the pack value at the report's precision."""
    if C.consistent(float(value), cell):
        passed.append(f"{what}: {_fmt(value)} vs report '{cell}'")
    else:
        pack.mismatch(item, what, value, f"{REPORT} ({where})", cell, "report prose")


def _seed_axis(ax, n: int = 20) -> None:
    """x axis of the per-run panels: seed - 20500."""
    ax.set_xlim(0.3, n + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("seed - 20500")


class _PlainLog(mticker.LogFormatter):
    """Log-axis labels as plain decimals (0.02, 0.001) instead of superscript powers of ten."""

    def _num_to_string(self, x: float, vmin: float, vmax: float) -> str:
        return f"{x:g}"


# ----------------------------------------------------------------------------------------------
# T39 reported metrics
# ----------------------------------------------------------------------------------------------

_G_A = "stage 2, end of Phase A (u1600)"
_G_B = "stage 1 and full policy, end of Phase B (u2200)"
_G_D = "stage-1 decomposition (residual band, final tier)"
_G_N = "dev - final difference of the tier-dependent reported metrics"
_G_T = "drift test of the frozen stage-2 snapshot"
_G_L = "location (t*, d*) of Gmax_full (final tier)"
_FR2, _FR1, _DW = "fraction of e2*(0)", "fraction of e1*(0)", "Delta W (dimensionless)"
_TI, _DMF, _EU = "tier-independent", "dev - final", "effort units [0, 100]"

# (group, quantity, per-run column, tier, units)
T39_STATS: List[Tuple[str, str, str, str, str]] = [
    (_G_A, "peak error at d = 0, signed", "A_stage2_peak_rel_err_signed", _TI, _FR2),
    (_G_A, "location-free peak error, (max_d e_hat_2(d) - e2*(0))/e2*(0)",
     "A_stage2_peak_locfree_rel_err", _TI, _FR2),
    (_G_A, "argmax d of e_hat_2 on the recovery grid (location-free peak)",
     "A_stage2_peak_locfree_argmax_d", _TI, "effort units (gap d)"),
    (_G_A, "symmetry error max_d |e_hat_2(d) - e_hat_2(-d)| / e2*(0)",
     "A_stage2_sym_err_max_over_g2_0", _TI, _FR2),
    (_G_A, "symmetry error, raw", "A_stage2_sym_err_max", _TI, "effort units, raw"),
    (_G_A, "tail max of e_hat_2 on |d| >= 2q / e2*(0)", "A_stage2_tail_max_over_g2_0", _TI, _FR2),
    (_G_A, "tail max, raw", "A_stage2_tail_max", _TI, _EU + ", raw"),
    (_G_A, "Delta_2 on path, max / dW", "A_DeltaT_over_dw_on_max", "final", _DW),
    (_G_A, "Delta_2 off path, max / dW", "A_DeltaT_over_dw_off_max", "final", _DW),
    (_G_A, "sigma_2(0), SD of the stage-2 Beta action at d = 0", "A_sigma_effort_at_0_t2", _TI,
     _EU),
    (_G_A, "smoothed-game share of the d = 0 peak gap", "A_smoothed_share_peak_gap_d0", _TI,
     "fraction of the peak gap e2*(0) - e_hat_2(0)"),
    (_G_B, "e_hat_1(0), raw", "B_e1_at_0", _TI, _EU + ", raw"),
    (_G_B, "stage-1 error (e_hat_1(0) - e1*(0))/e1*(0), signed", "B_stage1_rel_err_signed",
     _TI, _FR1),
    (_G_B, "EXP_root / dW", "B_EXP_root_over_dw", "final", _DW),
    (_G_B, "dReach / dW", "B_dReach_over_dw", "final", _DW),
    (_G_B, "Delta_max_all / dW", "B_Deltamax_all_over_dw", "final", _DW),
    (_G_B, "dFull / dW", "B_dFull_over_dw", "final", _DW),
    (_G_B, "sigma_1(0), SD of the stage-1 Beta action at d = 0", "B_sigma_effort_at_0_t1", _TI,
     _EU),
    (_G_D, "learning term (e_hat_1(0) - e~1)/e1*(0), signed", "dec_learning_rel", "final", _FR1),
    (_G_D, "inherited term (e~1 - e1*(0))/e1*(0), signed", "dec_inherited_rel", "final", _FR1),
    (_G_N, "Delta_2 on path, max / dW: dev - final", "A_dmf_DeltaT_over_dw_on_max", _DMF, _DW),
    (_G_N, "Delta_2 off path, max / dW: dev - final", "A_dmf_DeltaT_over_dw_off_max", _DMF, _DW),
    (_G_N, "EXP_root / dW: dev - final", "B_dmf_EXP_root_over_dw", _DMF, _DW),
    (_G_N, "dReach / dW: dev - final", "B_dmf_dReach_over_dw", _DMF, _DW),
    (_G_N, "Delta_max_all / dW: dev - final", "B_dmf_Deltamax_all_over_dw", _DMF, _DW),
    (_G_N, "dFull / dW: dev - final", "B_dmf_dFull_over_dw", _DMF, _DW),
    (_G_T, "max |snapshot - end-of-A mapping|, Beta mean", "drift_maxabs_mean", "n/a", _EU),
    (_G_T, "max |snapshot - end-of-A mapping|, alpha", "drift_maxabs_alpha", "n/a",
     "Beta parameter units"),
    (_G_T, "max |snapshot - end-of-A mapping|, beta", "drift_maxabs_beta", "n/a",
     "Beta parameter units"),
]

# (group, quantity, per-run boolean column, tier)
T39_COUNTS: List[Tuple[str, str, str, str]] = [
    (_G_D, "runs whose learning-term band interval contains 0", "dec_learning_contains_0",
     "final"),
    (_G_D, "runs whose inherited-term band interval contains 0", "dec_inherited_contains_0",
     "final"),
    (_G_D, "runs whose induced band is contiguous", "dec_band_contiguous", "final"),
    (_G_D, "runs with e_hat_1(0) inside the sweep range", "dec_e1_inside_sweep", "final"),
    (_G_D, "runs whose Delta_1 argmin lies at the sweep edge", "dec_argmin_at_sweep_edge",
     "final"),
    (_G_T, "runs passing the drift test", "drift_test_pass", "n/a"),
    (_G_T, "runs whose snapshot parameters are bit-identical to the end-of-A actor",
     "drift_snapshot_bit_identical", "n/a"),
]

_T39_EXTRA = ("A_stage2_sym_err_max_over_g2_0", "A_stage2_tail_max_over_g2_0",
              "A_dmf_DeltaT_over_dw_on_max", "A_dmf_DeltaT_over_dw_off_max",
              "B_dmf_EXP_root_over_dw", "B_dmf_dReach_over_dw", "B_dmf_Deltamax_all_over_dw",
              "B_dmf_dFull_over_dw", "dec_argmin_at_sweep_edge", "drift_maxabs_mean",
              "drift_maxabs_alpha", "drift_maxabs_beta", "drift_snapshot_bit_identical")
_A_COPIED = ("stage2_peak_rel_err_signed", "stage2_peak_locfree_rel_err",
             "stage2_peak_locfree_argmax_d", "stage2_sym_err_max", "stage2_tail_max",
             "DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max", "sigma_effort_at_0_t2")
_B_COPIED = ("e1_at_0", "stage1_rel_err_signed", "EXP_root_over_dw", "dReach_over_dw",
             "Deltamax_all_over_dw", "dFull_over_dw", "Gmax_full_t", "Gmax_full_d",
             "sigma_effort_at_0_t1")


def _t39_per_run(ctx: Ctx) -> pd.DataFrame:
    """reported_metrics.csv plus the T39 quantities found only in gates.json / drift_test.json."""
    df = ctx.rep.copy()
    add: Dict[str, List[Any]] = {k: [] for k in _T39_EXTRA}
    for q, s, sym in zip(df["q"], df["seed"], df["A_stage2_sym_err_max"]):
        g = ctx.gates[(int(q), int(s))]["reported"]
        a, b = g["end_of_A"], g["end_of_B"]
        add["A_stage2_sym_err_max_over_g2_0"].append(float(sym) / float(a["final"]["g2_at_0"]))
        add["A_stage2_tail_max_over_g2_0"].append(float(a["final"]["stage2_tail_max_over_g2_0"]))
        for k in ("DeltaT_over_dw_on_max", "DeltaT_over_dw_off_max"):
            add[f"A_dmf_{k}"].append(float(a["dev_minus_final"][k]))
        for k in ("EXP_root_over_dw", "dReach_over_dw", "Deltamax_all_over_dw", "dFull_over_dw"):
            add[f"B_dmf_{k}"].append(float(b["dev_minus_final"][k]))
        add["dec_argmin_at_sweep_edge"].append(bool(b["decomposition"]["argmin_at_sweep_edge"]))
        dt = ctx.drift[(int(q), int(s))]
        for k in ("mean", "alpha", "beta"):
            add[f"drift_maxabs_{k}"].append(float(dt["max_abs_diff_vs_freeze_time"][k]))
        add["drift_snapshot_bit_identical"].append(
            bool(dt["snapshot_params_bit_identical_to_end_of_A_actor"]))
    for k, v in add.items():
        df[k] = v
    return df


def _t39_checks(ctx: Ctx, df: pd.DataFrame) -> List[str]:
    """Agreement of reported_metrics.csv with gates.json and with the saved end-of-A arrays."""
    d_g, d_a = 0.0, 0.0
    for _, r in df.iterrows():
        q, s = int(r["q"]), int(r["seed"])
        g = ctx.gates[(q, s)]["reported"]
        fa, fb = g["end_of_A"]["final"], g["end_of_B"]["final"]
        for k in _A_COPIED:
            d_g = max(d_g, abs(float(r["A_" + k]) - float(fa[k])))
        for k in _B_COPIED:
            d_g = max(d_g, abs(float(r["B_" + k]) - float(fb[k])))
        d_g = max(d_g, abs(float(r["A_smoothed_share_peak_gap_d0"])
                           - float(g["end_of_A"]["smoothed_game"]["smoothed_share_peak_gap_d0"])))
        z = ctx.npz(q, s, "gateA_final")
        D, e2 = z["recovery_d_grid"], z["recovery_e2"]
        g20 = float(z["recovery_g2"][int(np.nonzero(D == 0.0)[0][0])])
        j = int(np.argmax(e2))
        d_a = max(d_a,
                  abs((float(e2[j]) - g20) / g20 - float(r["A_stage2_peak_locfree_rel_err"])),
                  abs(float(D[j]) - float(r["A_stage2_peak_locfree_argmax_d"])),
                  abs((float(e2[D == 0.0][0]) - g20) / g20
                      - float(r["A_stage2_peak_rel_err_signed"])))
    return [f"reported_metrics.csv equals gates.json 'reported' in every copied field (max abs "
            f"diff {d_g!r}, 40 runs)",
            f"location-free peak error, its argmax d and the peak error at d=0 recomputed from "
            f"gateA_final.npz (recovery_e2, recovery_g2; tools/v2/pilot4_common.py:location_free) "
            f"equal reported_metrics.csv (max abs diff {d_a!r})"]


def build_t39(pack: C.Pack, ctx: Ctx) -> None:
    """T39: reported metrics per q (median, IQR, min, max) and the counts the item asks for."""
    df = _t39_per_run(ctx)
    rows: List[Dict[str, Any]] = []
    for q in QS:
        g = df[df["q"] == q]
        for grp, qty, col, tier, unit in T39_STATS:
            st = C.median_iqr(g[col].to_numpy(dtype=float))
            rows.append({"q": q, "group": grp, "quantity": qty, "column": col, "tier": tier,
                         "units": unit, **st, "count": np.nan})
        t = g["B_Gmax_full_t"].astype(int)
        for tv, lab in ((1, "runs with t* = 1 (root, d = 0)"), (2, "runs with t* = 2")):
            rows.append({"q": q, "group": _G_L, "quantity": lab, "column": "B_Gmax_full_t",
                         "tier": "final", "units": "count of runs", "n": len(g),
                         "count": int((t == tv).sum())})
        pairs = g.groupby(["B_Gmax_full_t", "B_Gmax_full_d"]).size().reset_index(name="k")
        for _, p in pairs.sort_values(["B_Gmax_full_t", "B_Gmax_full_d"]).iterrows():
            lab = f"runs with (t*, d*) = ({int(p['B_Gmax_full_t'])}, {p['B_Gmax_full_d']:g})"
            rows.append({"q": q, "group": _G_L, "quantity": lab,
                         "column": "B_Gmax_full_t, B_Gmax_full_d", "tier": "final",
                         "units": "count of runs", "n": len(g), "count": int(p["k"])})
        for grp, qty, col, tier in T39_COUNTS:
            rows.append({"q": q, "group": grp, "quantity": qty, "column": col, "tier": tier,
                         "units": "count of runs", "n": len(g),
                         "count": int(g[col].astype(bool).sum())})
    out = pd.DataFrame(rows)[["q", "group", "quantity", "column", "tier", "units", "n", "median",
                              "q25", "q75", "min", "max", "count"]]
    out["count"] = _int_or_none(out["count"])
    checks = _t39_checks(ctx, df)
    cc = pack.crosscheck("T39", ctx.rep, REPORT,
                         header_has=["q", "seed", "A_stage2_peak_rel_err_signed",
                                     "dec_inherited_rel_hi"],
                         key_map={"q": "q", "seed": "seed"},
                         value_map={c: c for c in ctx.rep.columns if c not in ("q", "seed")
                                    and ctx.rep[c].dtype != bool},
                         heading_has="Reported metrics",
                         label="per-run reported metrics (report 4.5)")
    notes = (f"Per-run values: {CA}/reported_metrics.csv (the pre-registered analysis script's "
             "copy of each run's gates.json 'reported' block); tail max/e2*(0), the dev - final "
             "differences (gates.json reported.*.dev_minus_final, named A_dmf_*/B_dmf_*) and the "
             "argmin-at-sweep-edge flag from each run's gates.json; symmetry/e2*(0) = "
             "A_stage2_sym_err_max / g2_at_0 (gates.json); drift magnitudes and the bit-identity "
             "flag from drift_test.json. Statistics over the 20 seeds per q: median, q25, q75 "
             "(numpy linear interpolation), min, max; count rows give the number of runs (out of "
             "n) for which the stated condition holds. End of A = stage-2 last iterate at u1600; "
             "end of B = live stage-1 actor at u2200 with the frozen end-of-A stage-2 snapshot. "
             + NOTE_CSV + " Checks: " + "; ".join(checks)
             + f". Report cross-check (4.5): {cc['n_compared']} cells, "
             f"{cc['n_mismatch']} mismatches.")
    same_tier = "as in the tier column"
    docs = {
        "group": "Block of the table: which candidate / analysis the quantity belongs to (end of "
                 "Phase A = stage-2 last iterate at u1600; end of Phase B = full last iterate at "
                 "u2200)",
        "quantity": "What the row reports (statistic rows: the per-run quantity summarised; count "
                    "rows: the condition counted)",
        "column": "Per-run column the row summarises (reported_metrics.csv names; A_dmf_/B_dmf_ = "
                  "dev - final from gates.json; drift_* from drift_test.json; "
                  "dec_argmin_at_sweep_edge from gates.json)",
        "tier": {"definition": "Tier of the quantity: final = final verifier tier (state step 2, "
                               "effort step 0.5, GL 32); tier-independent = recovery grid or "
                               "direct policy query; dev - final = development minus final "
                               "tier; n/a = not a verifier quantity",
                 "units": "label"},
        "units": "Units of the median/q25/q75/min/max columns (count rows: count of runs)",
        "count": {"definition": "Count rows: number of runs (out of n) for which the condition "
                                "in 'quantity' holds; empty for statistic rows",
                  "units": "count of runs"},
        "n": {"definition": "Number of runs (seeds) behind the row", "units": "count"},
        "median": {"definition": "Median over the n seeds (statistic rows)", "tier": same_tier},
        "q25": {"definition": "25th percentile over the seeds (numpy linear interpolation)",
                "tier": same_tier},
        "q75": {"definition": "75th percentile over the seeds (numpy linear interpolation)",
                "tier": same_tier},
        "min": {"definition": "Minimum over the seeds", "tier": same_tier},
        "max": {"definition": "Maximum over the seeds", "tier": same_tier},
    }
    pack.table("T39", out, status="generated",
               sources=[C.src(f"{CA}/reported_metrics.csv"), _set("gates.json"),
                        _set("drift_test.json"), _set("gateA_final.npz")],
               script=f"{MOD}:build_t39", notes=notes, docs=docs, tier="final",
               caption="Confirmation (protocol v1.1, fresh seeds 20501-20520), per q: the reported "
                       "(not gated) metrics of the locked protocol, median, IQR, min and max over "
                       "the 20 runs, and counts of (t*, d*), of decomposition bands containing 0, "
                       "of band diagnostics and of the drift test.")


# ----------------------------------------------------------------------------------------------
# F12 gate metrics against thresholds
# ----------------------------------------------------------------------------------------------

# (per_run column, take abs, threshold key, criterion, panel title, y label)
F12_PANELS: List[Tuple[str, bool, str, str, str, str]] = [
    ("eta_final", False, "eta_T_over_dw", "G-A", r"$\eta_2$, final tier (G-A)",
     r"$\eta_2/\Delta W$"),
    ("rmse", False, "stage2_rmse_pos_over_g2_0", "G-A", r"RMSE on $|d|<2q$ (G-A)",
     r"RMSE$/e_2^*(0)$"),
    ("tail", False, "stage2_tail_mean_over_g2_0", "G-A", r"tail mean on $|d|\geq 2q$ (G-A)",
     r"tail mean$/e_2^*(0)$"),
    ("gmax_final", False, "Gmax_full_over_dw", "G-F", r"$\hat G_{max,full}$, final tier (G-F)",
     r"$\hat G_{max,full}/\Delta W$"),
    ("eta_dev_minus_final", True, "eta_T_over_dw_dev_minus_final_abs", "G-N",
     r"$|\eta_2$ dev $-$ final$|$ (G-N)", r"$|\Delta\eta_2|/\Delta W$"),
    ("gmax_dev_minus_final", True, "Gmax_full_over_dw_dev_minus_final_abs", "G-N",
     r"$|\hat G_{max}$ dev $-$ final$|$ (G-N)", r"$|\Delta\hat G_{max}|/\Delta W$"),
    ("s1", False, "stage1_rel_err_abs", "S1", "|stage-1 error| (S1, secondary)",
     r"$|\hat e_1(0)-e_1^*|/e_1^*$"),
]


def _f12_plot(ctx: Ctx):
    """F12 figure and its long-format data (one row per plotted run value)."""
    fig, axes = style.new_figure(4, 2, height=8.6)
    rows: List[Dict[str, Any]] = []
    for k, (col, take_abs, tkey, crit, title, ylab) in enumerate(F12_PANELS):
        ax = axes[k // 2, k % 2]
        thr = ctx.th[tkey]
        vmax = thr
        for q in QS:
            g = ctx.pr(q)
            v = g[col].to_numpy(dtype=float)
            v = np.abs(v) if take_abs else v
            x = (g["seed"].to_numpy() - 20500) + (-0.15 if q == 50 else 0.15)
            ax.plot(x, v, ls="none", marker=style.Q_MARKER[q], color=style.Q_COLORS[q], ms=4.0,
                    label=style.label_n(f"q={q}", len(v)))
            vmax = max(vmax, float(v.max()))
            for s, val in zip(g["seed"], v):
                rows.append({"q": q, "seed": int(s), "metric": ("abs_" if take_abs else "") + col,
                             "value": float(val), "threshold": thr, "criterion": crit,
                             "within_threshold": bool(val <= thr)})
        s1 = crit == "S1"
        ax.axhline(thr, color=style.THRESH, ls=("-." if s1 else "--"), lw=1.0,
                   label=("S1 threshold (secondary, not gated)" if s1 else "gate threshold"))
        ax.set_ylim(0.0, 1.12 * vmax)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        _seed_axis(ax)
    lax = axes[3, 1]
    lax.axis("off")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[3, 0].get_legend_handles_labels()
    lax.legend(h1 + h2[-1:], l1 + l2[-1:], loc="center",
               title="Confirmation runs (protocol v1.1)")
    return fig, pd.DataFrame(rows)


def _f12_checks(ctx: Ctx, data: pd.DataFrame) -> List[str]:
    """Plotted values against per_run.csv, the recorded verdicts and agreement.csv."""
    pr = ctx.per_run
    chk: List[str] = []
    mad = 0.0
    for col, take_abs, *_ in F12_PANELS:
        sub = data[data["metric"] == ("abs_" if take_abs else "") + col]
        ref = pr[col].to_numpy(dtype=float)
        mad = max(mad, C.max_abs_diff(sub.sort_values(["q", "seed"])["value"].to_numpy(),
                                      np.abs(ref) if take_abs else ref))
    chk.append(f"plotted values equal per_run.csv (max abs diff {mad!r})")
    for q in QS:
        dq, g = data[data["q"] == q], ctx.pr(q)
        within = dq.pivot(index="seed", columns="metric", values="within_threshold")
        ga = within[["eta_final", "rmse", "tail"]].all(axis=1).to_numpy()
        gn = within[["abs_eta_dev_minus_final", "abs_gmax_dev_minus_final"]].all(axis=1)
        gn = gn.to_numpy()
        gf, s1 = within["gmax_final"].to_numpy(dtype=bool), within["s1"].to_numpy(dtype=bool)
        same = (np.array_equal(ga, g["G-A_pass"].to_numpy(dtype=bool))
                and np.array_equal(gf, g["G-F_pass"].to_numpy(dtype=bool))
                and np.array_equal(gn, g["G-N_pass"].to_numpy(dtype=bool))
                and np.array_equal(s1, g["S1_pass"].to_numpy(dtype=bool)))
        chk.append(f"q={q}: runs within threshold G-A {int(ga.sum())}/20, G-F "
                   f"{int(within['gmax_final'].sum())}/20, G-N {int(gn.sum())}/20, S1 "
                   f"{int(within['s1'].sum())}/20; per-run within/above agrees with the "
                   f"per_run.csv pass flags: {same}")
    ag = _csv(f"{CA}/agreement.csv")
    absd = [c for c in ag.columns if c.startswith("absdiff_")]
    amax = float(ag[absd].to_numpy(dtype=float).max())
    chk.append(f"per_run.csv (recomputed from the NPZs by tools/v2/confirmation_analysis.py) "
               f"agrees with every gates.json: all_agree {int(ag['all_agree'].sum())}/40, max "
               f"absdiff {amax!r} (agreement.csv)")
    chk.append("thresholds read from protocols/v2_T2_locked_v1_1.json (gates G-A, G-F, G-N; "
               "secondary S1)")
    return chk


def _f12_prose(pack: C.Pack, ctx: Ctx) -> List[str]:
    """Numbers of the report prose (sections 4.3 and 4.4) that F12 shows."""
    pr = ctx.per_run
    passed: List[str] = []
    q60 = ctx.pr(60).set_index("seed")
    for s, cell in ((20510, "+0.153"), (20515, "-0.113")):
        _prose(pack, "F12", f"signed stage-1 error q=60 seed {s}",
               q60.loc[s, "stage1_rel_err_signed"], cell, "4.3", passed)
    for s, cell in ((20510, "0.0036"), (20515, "0.0018")):
        _prose(pack, "F12", f"Gmax_full/dW q=60 seed {s}", q60.loc[s, "gmax_final"], cell, "4.3",
               passed)
    dmf = pr[["eta_dev_minus_final", "gmax_dev_minus_final"]].to_numpy(dtype=float)
    _prose(pack, "F12", "largest |dev - final| (eta_2 and Gmax_full, both q)",
           float(np.abs(dmf).max()), "2.6e-4", "4.4", passed)
    n_eta_pos = int((pr["eta_dev_minus_final"] > 0).sum())
    n_gmax_pos = int((pr["gmax_dev_minus_final"] > 0).sum())
    gpos_max = float(pr["gmax_dev_minus_final"].max())
    if n_gmax_pos > 4:
        pack.mismatch("F12", "runs with Gmax_full dev tier > final tier (of 40)", n_gmax_pos,
                      f"{REPORT} (4.4)",
                      "'eta_2 and Gmax_full on the dev tier are lower than or equal to final in "
                      "nearly all runs'",
                      f"eta_2: {40 - n_eta_pos}/40 runs have dev <= final (the other {n_eta_pos} "
                      f"differ by {_fmt(pr['eta_dev_minus_final'].max())}); Gmax_full: "
                      f"{40 - n_gmax_pos}/40 runs have dev <= final, {n_gmax_pos} have dev > "
                      f"final by up to {_fmt(gpos_max)} dW (per_run.csv)")
    return passed


def build_f12(pack: C.Pack, ctx: Ctx) -> None:
    """F12: every run's gate-metric value against its threshold, per q."""
    fig, data = _f12_plot(ctx)
    chk = _f12_checks(ctx, data)
    value_cols = ("eta_final", "eta_dev", "eta_dev_minus_final", "rmse", "tail", "gmax_final",
                  "gmax_dev", "gmax_dev_minus_final", "s1", "stage1_rel_err_signed", "wall_sec")
    cc = pack.crosscheck("F12", ctx.per_run, REPORT,
                         header_has=["q", "seed", "eta_final", "gmax_dev_minus_final", "s1"],
                         key_map={"q": "q", "seed": "seed"},
                         value_map={c: c for c in value_cols},
                         heading_has="Per-run table, all 40",
                         label="per-run gate metrics (report 4.2)")
    passed = _f12_prose(pack, ctx)
    chk.append("report prose consistent: " + "; ".join(passed))
    chk.append(f"report cross-check (4.2): {cc['n_compared']} cells, {cc['n_mismatch']} "
               f"mismatches")
    docs = {
        "metric": "per_run.csv column plotted (abs_ = absolute value of the column): eta_final = "
                  "eta_2/dW final tier; rmse = RMSE/e2*(0); tail = tail mean/e2*(0); gmax_final = "
                  "Gmax_full/dW final tier; abs_eta_dev_minus_final, abs_gmax_dev_minus_final = "
                  "|dev - final|/dW; s1 = |stage-1 error|",
        "value": {"definition": "Plotted per-run value of the metric",
                  "units": "as the metric (dW-normalised, fraction of e2*(0) or of e1*(0))",
                  "tier": "final (eta_final, gmax_final); tier-independent (rmse, tail, s1); "
                          "dev - final (abs_* rows)"},
        "threshold": {"definition": "Threshold of the criterion (inclusive <=), read from "
                                    "protocols/v2_T2_locked_v1_1.json", "units": "as the metric"},
        "criterion": "Criterion the threshold belongs to: G-A, G-F, G-N (gates of the run pass) "
                     "or S1 (secondary, not part of the run pass)",
        "within_threshold": "value <= threshold (full float64 precision)",
    }
    pack.figure("F12", fig, data, status="generated",
                sources=[C.src(f"{CA}/per_run.csv"), C.src(f"{CA}/agreement.csv"), C.src(PROTO)],
                script=f"{MOD}:build_f12", tier="final and development",
                caption="Gate metrics of every confirmation run (protocol v1.1; seeds 20501-20520 "
                        "on the x axis; q=50 blue circles, q=60 orange squares, n=20 per q) "
                        "against the thresholds of the locked protocol: gates (dashed) G-A "
                        "(eta_2/dW <= 0.005, final tier; RMSE/e2*(0) <= 0.05; tail mean/e2*(0) "
                        "<= 0.02; RMSE and tail are tier-independent), G-F (Gmax_full/dW <= "
                        "0.01, final tier), G-N (|dev - final| of eta_2 and of Gmax_full <= "
                        "0.001 dW), and the secondary criterion S1 (dash-dot; |stage-1 error| "
                        "<= 0.10; not part of the run pass). Source: "
                        f"{CA}/per_run.csv; thresholds from {PROTO}.",
                notes=NOTE_CSV, checks=chk, docs=docs)


# ----------------------------------------------------------------------------------------------
# F13 stage-2 mapping of all runs
# ----------------------------------------------------------------------------------------------

def _f13_end_b_check(ctx: Ctx) -> Tuple[float, float]:
    """Max abs difference end of B vs end of A of the stage-2 arrays (final and dev NPZs)."""
    keys = ("recovery_d_grid", "recovery_e2", "v_t2_d_grid", "v_t2_e_hat", "v_t2_sigma_effort",
            "v_t2_alpha", "v_t2_beta")
    m_fin = m_dev = 0.0
    for q in QS:
        for s in SEEDS:
            a, b = ctx.npz(q, s, "gateA_final"), ctx.npz(q, s, "final_final")
            ad, bd = ctx.npz(q, s, "gateA_development"), ctx.npz(q, s, "final_development")
            for k in keys:
                m_fin = max(m_fin, C.max_abs_diff(a[k], b[k]))
                m_dev = max(m_dev, C.max_abs_diff(ad[k], bd[k]))
    return m_fin, m_dev


def build_f13(pack: C.Pack, ctx: Ctx) -> None:
    """F13: e_hat_2(d) of all 20 runs per q with the median and e2*(d); lower panel sigma_2(d)."""
    import sys
    sys.path.insert(0, str(C.REPO))
    from utils.theory_multistage import g2_two_stage

    fig, axes = style.new_figure(2, 2, height=5.8, sharex="col")
    parts: List[pd.DataFrame] = []
    chk: List[str] = []
    g2_dev = sig_dev = med0_dev = 0.0

    def add(q: int, panel: str, series: str, seed: Optional[int], d: np.ndarray,
            v: np.ndarray) -> None:
        parts.append(pd.DataFrame({"q": q, "panel": panel, "series": series,
                                   "seed": pd.array([seed] * len(d), dtype="Int64"), "d": d,
                                   "value": v}))

    for j, q in enumerate(QS):
        col, ls = style.Q_COLORS[q], style.Q_LINESTYLE[q]
        e2 = np.vstack([ctx.npz(q, s, "gateA_final")["recovery_e2"] for s in SEEDS])
        sg = np.vstack([ctx.npz(q, s, "gateA_final")["v_t2_sigma_effort"] for s in SEEDS])
        z0 = ctx.npz(q, SEEDS[0], "gateA_final")
        D, Dv, g2 = z0["recovery_d_grid"], z0["v_t2_d_grid"], z0["recovery_g2"]
        for s in SEEDS:
            z = ctx.npz(q, s, "gateA_final")
            if not (np.array_equal(z["recovery_d_grid"], D) and np.array_equal(z["v_t2_d_grid"], Dv)
                    and np.array_equal(z["recovery_g2"], g2)):
                raise ValueError(f"grids differ across runs at q={q}")
        spec = ctx.proto["records"][str(q)]["game"]
        g2_cf = g2_two_stage(D, float(q), float(spec["w_h"]), float(spec["w_l"]),
                             float(spec["k"]), float(spec["e_max"]))
        g2_dev = max(g2_dev, C.max_abs_diff(g2, g2_cf))
        med, med_s = np.median(e2, axis=0), np.median(sg, axis=0)
        z_r, z_v = int(np.nonzero(D == 0.0)[0][0]), int(np.nonzero(Dv == 0.0)[0][0])
        rep = ctx.rows(q)
        med0_dev = max(med0_dev, abs((med[z_r] - g2[z_r]) / g2[z_r]
                                     - float(np.median(rep["A_stage2_peak_rel_err_signed"]))))
        sig_dev = max(sig_dev, C.max_abs_diff(sg[:, z_v], rep["A_sigma_effort_at_0_t2"].to_numpy()))
        ax, bx = axes[0, j], axes[1, j]
        for i, s in enumerate(SEEDS):
            lab = style.label_n("runs", len(SEEDS)) if i == 0 else None
            ax.plot(D, e2[i], color=col, lw=style.THIN_W, alpha=0.35, label=lab)
            bx.plot(Dv, sg[i], color=col, lw=style.THIN_W, alpha=0.35, label=lab)
            add(q, "e2_hat", "run", s, D, e2[i])
            add(q, "sigma2", "run", s, Dv, sg[i])
        lab_med = style.label_n("median", len(SEEDS))
        ax.plot(D, med, color=col, ls=ls, lw=style.LINE_W, label=lab_med)
        bx.plot(Dv, med_s, color=col, ls=ls, lw=style.LINE_W, label=lab_med)
        ax.plot(D, g2, color=style.REF, lw=1.0, label=r"$e_2^*(d)$, closed form")
        add(q, "e2_hat", "median", None, D, med)
        add(q, "sigma2", "median", None, Dv, med_s)
        add(q, "e2_hat", "closed_form", None, D, g2)
        ax.set_title(f"q={q}: stage-2 mapping, end of Phase A")
        ax.set_ylabel(r"$\hat e_2(d)$ (effort units, raw)")
        bx.set_title(f"q={q}: stage-2 action SD")
        bx.set_ylabel(r"$\sigma_2(d)$ (effort units)")
        bx.set_xlabel("gap d (effort units)")
        ax.set_xlim(float(D[0]), float(D[-1]))
        ax.set_ylim(top=1.36 * max(float(e2.max()), float(g2.max())))
        ax.legend(loc="upper right")
    m_fin, m_dev = _f13_end_b_check(ctx)
    chk.append(f"end-of-B arrays (final_final.npz: recovery_e2, v_t2_e_hat, v_t2_sigma_effort, "
               f"v_t2_alpha, v_t2_beta and grids) equal the end-of-A arrays (gateA_final.npz) in "
               f"40/40 runs: max abs diff {m_fin!r}; development tier (final_development vs "
               f"gateA_development): {m_dev!r}")
    chk.append(f"recovery_g2 equals utils.theory_multistage.g2_two_stage on the recovery grid "
               f"(max abs diff {g2_dev!r})")
    chk.append(f"median e_hat_2(0)/e2*(0) - 1 equals the median of A_stage2_peak_rel_err_signed "
               f"in reported_metrics.csv (max abs diff {med0_dev!r})")
    chk.append(f"sigma_2(0) of every run equals A_sigma_effort_at_0_t2 (max abs diff {sig_dev!r})")
    if m_fin != 0.0 or m_dev != 0.0:
        pack.mismatch("F13", "end-of-B vs end-of-A stage-2 arrays, max abs diff",
                      max(m_fin, m_dev), REPORT, "frozen stage 2 (identical mapping)",
                      "stage-2 arrays changed in Phase B")
    data = pd.concat(parts, ignore_index=True)
    docs = {
        "panel": "Panel: e2_hat = stage-2 Beta-mean effort e_hat_2(d) on the recovery grid (step "
                 "0.5); sigma2 = SD sigma_2(d) of the stage-2 Beta action on the final-tier state "
                 "grid (step 2)",
        "series": "run = one confirmation run (seed given); median = pointwise median over the 20 "
                  "runs; closed_form = e2*(d) (recovery_g2 of the NPZ, = "
                  "utils.theory_multistage.g2_two_stage)",
        "seed": "Run seed (empty for the median and the closed form)",
        "d": {"definition": "Gap d at which the value is taken", "units": "effort units (gap d)"},
        "value": {"definition": "Plotted value: e_hat_2(d) or e2*(d) (raw effort) or sigma_2(d)",
                  "units": "effort units [0, 100]",
                  "tier": "tier-independent (policy query; sigma2 on the final-tier state grid)"},
    }
    srcs = [_set("gateA_final.npz"), _set("final_final.npz"), _set("gateA_development.npz"),
            _set("final_development.npz"), C.src(f"{CA}/reported_metrics.csv"), C.src(THEORY),
            C.src(PROTO)]
    pack.figure("F13", fig, data, status="generated", sources=srcs, script=f"{MOD}:build_f13",
                tier="final",
                caption="Stage-2 mapping of the 20 confirmation runs per q (protocol v1.1, seeds "
                        "20501-20520) at the end of Phase A (u1600), from each run's "
                        "gateA_final.npz. Top: e_hat_2(d) (Beta mean, raw effort units) on the "
                        "recovery grid (step 0.5) for every run (thin lines), the pointwise median "
                        "over the runs (thick) and the closed form e2*(d) (black). Bottom: "
                        "sigma_2(d), the SD of the stage-2 Beta action, on the final-tier state "
                        "grid (step 2). Stage 2 is frozen in Phase B; the end-of-B arrays "
                        "(final_final.npz) are identical. e_hat_2 and sigma_2 are policy queries "
                        "(tier-independent); n=20 per q.",
                checks=chk, docs=docs)


# ----------------------------------------------------------------------------------------------
# F14 stage-1 effort and its decomposition
# ----------------------------------------------------------------------------------------------

def build_f14(pack: C.Pack, ctx: Ctx) -> None:
    """F14: e_hat_1(0) per run against e1*(0) with +-5% / +-10% bands; learning, inherited terms."""
    import sys
    sys.path.insert(0, str(C.REPO))
    from utils.theory_multistage import g1_two_stage

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = style.new_figure(2, 2, height=6.4)
    rows: List[Dict[str, Any]] = []
    chk: List[str] = []
    e1_dev = ident = cf_dev = 0.0
    flag_bad = 0
    top_h: List[Any] = []
    top_l: List[str] = []
    for j, q in enumerate(QS):
        col, mk = style.Q_COLORS[q], style.Q_MARKER[q]
        g = ctx.rows(q)
        x = g["seed"].to_numpy() - 20500
        e1s = ctx.e1s[q]
        spec = ctx.proto["records"][str(q)]["game"]
        e1_cf = g1_two_stage(float(q), float(spec["w_h"]), float(spec["w_l"]), float(spec["k"]))
        cf_dev = max(cf_dev, abs(e1s - float(e1_cf)))
        ax = axes[0, j]
        ax.axhspan(e1s * 0.90, e1s * 1.10, color=style.INK2, alpha=0.08, lw=0)
        ax.axhspan(e1s * 0.95, e1s * 1.05, color=style.INK2, alpha=0.16, lw=0)
        ax.axhline(e1s, color=style.REF, lw=1.0)
        e1 = g["B_e1_at_0"].to_numpy(dtype=float)
        n5 = int((np.abs(e1 - e1s) <= 0.05 * e1s).sum())
        n10 = int((np.abs(e1 - e1s) <= 0.10 * e1s).sum())
        ax.plot(x, e1, ls="none", marker=mk, color=col, ms=4.5)
        top_h.append(Line2D([], [], ls="none", marker=mk, color=col, ms=4.5))
        top_l.append(style.label_n(rf"$\hat e_1(0)$, q={q}", len(e1))
                     + f": {n5} within 5%, {n10} within 10%")
        ax.set_title(f"q={q}: stage-1 effort at u2200")
        ax.set_ylabel(r"$\hat e_1(0)$ (effort units, raw)")
        lo_y, hi_y = min(e1.min(), e1s * 0.90), max(e1.max(), e1s * 1.10)
        pad = 0.06 * (hi_y - lo_y)
        ax.set_ylim(lo_y - pad, hi_y + pad)
        _seed_axis(ax)
        for s, v in zip(g["seed"], e1):
            z = ctx.npz(q, int(s), "final_final")
            e1_dev = max(e1_dev, abs(float(z["v_t1_e_hat"][0]) - v))
            rows.append({"q": q, "seed": int(s), "panel": "e1", "quantity": "e1_at_0", "value": v})
        for name, v in (("e1_star", e1s), ("band5_lo", 0.95 * e1s), ("band5_hi", 1.05 * e1s),
                        ("band10_lo", 0.90 * e1s), ("band10_hi", 1.10 * e1s)):
            rows.append({"q": q, "seed": None, "panel": "e1", "quantity": name, "value": v})
        bx = axes[1, j]
        bx.axhline(0.0, color=style.REF, lw=0.8)
        for term, off, m_term in (("learning", -0.17, mk), ("inherited", 0.17, "^")):
            v = g[f"dec_{term}_rel"].to_numpy(dtype=float)
            lo = g[f"dec_{term}_rel_lo"].to_numpy(dtype=float)
            hi = g[f"dec_{term}_rel_hi"].to_numpy(dtype=float)
            c0 = g[f"dec_{term}_contains_0"].to_numpy(dtype=bool)
            flag_bad += int(((lo <= 0) & (hi >= 0) != c0).sum())
            bx.errorbar(x + off, v, yerr=np.vstack([v - lo, hi - v]), fmt="none", ecolor=col,
                        elinewidth=1.0, capsize=1.8)
            bx.plot(x[~c0] + off, v[~c0], ls="none", marker=m_term, color=col, ms=4.0)
            bx.plot(x[c0] + off, v[c0], ls="none", marker=m_term, mfc="white", mec=col, ms=4.0)
            for s, a, b_, c_, d_ in zip(g["seed"], v, lo, hi, c0):
                rows.append({"q": q, "seed": int(s), "panel": "decomposition",
                             "quantity": f"{term}_rel", "value": a, "interval_lo": b_,
                             "interval_hi": c_, "interval_contains_0": bool(d_)})
        tot = g["dec_learning_rel"] + g["dec_inherited_rel"] - g["B_stage1_rel_err_signed"]
        ident = max(ident, float(tot.abs().max()))
        bx.set_title(f"q={q}: decomposition of the stage-1 error")
        bx.set_ylabel(r"term / $e_1^*(0)$")
        yl = bx.get_ylim()
        bx.set_ylim(yl[0], yl[1] + 0.42 * (yl[1] - yl[0]))
        nl = int(g["dec_learning_contains_0"].astype(bool).sum())
        ni = int(g["dec_inherited_contains_0"].astype(bool).sum())
        bx.legend([Line2D([], [], ls="none", marker=mk, color=col, ms=4.0),
                   Line2D([], [], ls="none", marker="^", color=col, ms=4.0),
                   Line2D([], [], ls="none", marker="o", mfc="white", mec=style.INK2, ms=4.0)],
                  [f"learning term ({nl}/20 bands contain 0)",
                   f"inherited term ({ni}/20 bands contain 0)", "open marker: band contains 0"],
                  loc="upper left")
        _seed_axis(bx)
    top_h += [Line2D([], [], color=style.REF, lw=1.0), Patch(color=style.INK2, alpha=0.16, lw=0),
              Patch(color=style.INK2, alpha=0.08, lw=0)]
    top_l += [rf"$e_1^*(0)$, closed form ({ctx.e1s[50]:.3f} at q=50, {ctx.e1s[60]:.3f} at q=60)",
              r"$e_1^*(0)\pm5\%$", r"$e_1^*(0)\pm10\%$"]
    fig.legend(top_h, top_l, loc="outside upper center", ncol=2)
    chk.append(f"plotted e_hat_1(0) (reported_metrics.csv B_e1_at_0) equals final_final.npz "
               f"v_t1_e_hat[0] in 40 runs (max abs diff {e1_dev!r})")
    chk.append(f"learning + inherited term = stage-1 error B_stage1_rel_err_signed (max abs diff "
               f"{ident!r})")
    chk.append(f"band-contains-0 flags equal lo <= 0 <= hi in 80/80 intervals: {flag_bad == 0}")
    chk.append(f"e1*(0) from gates.json (g1) equals utils.theory_multistage.g1_two_stage (max abs "
               f"diff {cf_dev!r})")
    data = pd.DataFrame(rows)
    data["seed"] = data["seed"].astype("Int64")
    docs = {
        "panel": "Panel: e1 = stage-1 effort; decomposition = learning and inherited terms",
        "quantity": "e1_at_0 = e_hat_1(0) per run (raw effort); e1_star = e1*(0); band5_lo/hi and "
                    "band10_lo/hi = e1*(0)(1 -+ 0.05) and e1*(0)(1 -+ 0.10); learning_rel = "
                    "(e_hat_1(0) - e~1)/e1*(0); inherited_rel = (e~1 - e1*(0))/e1*(0)",
        "seed": "Run seed (empty for the reference values)",
        "value": {"definition": "Plotted value of the quantity",
                  "units": "effort units [0, 100] (panel e1) or fraction of e1*(0) (panel "
                           "decomposition)",
                  "tier": "tier-independent (panel e1); final (panel decomposition, residual "
                          "band)"},
        "interval_lo": {"definition": "Lower end of the band interval of the term "
                                      "(dec_*_rel_lo)", "units": "fraction of e1*(0)",
                        "tier": "final"},
        "interval_hi": {"definition": "Upper end of the band interval of the term "
                                      "(dec_*_rel_hi)", "units": "fraction of e1*(0)",
                        "tier": "final"},
        "interval_contains_0": "Whether the band interval of the term contains 0 "
                               "(dec_*_contains_0; open marker)",
    }
    pack.figure("F14", fig, data, status="generated",
                sources=[C.src(f"{CA}/reported_metrics.csv"), _set("gates.json"),
                         _set("final_final.npz"), C.src(THEORY), C.src(PROTO)],
                script=f"{MOD}:build_f14", tier="final",
                caption="Stage-1 effort of the 20 confirmation runs per q (protocol v1.1, seeds "
                        "20501-20520 on the x axis) at the end of Phase B (u2200). Top: e_hat_1(0) "
                        "(Beta mean, raw effort units) against e1*(0) (black line) with the e1*(0) "
                        "+-5% (darker) and +-10% (lighter) bands. Bottom: the learning term "
                        "(e_hat_1(0) - e~1)/e1*(0) (q marker, left of the seed) and the inherited "
                        "term (e~1 - e1*(0))/e1*(0) (triangle, right of the seed), each with its "
                        "band interval (residual band on the final tier against the frozen stage "
                        "2); open markers: the interval contains 0. Source: "
                        f"{CA}/reported_metrics.csv (copied from gates.json). n=20 per q.",
                notes=NOTE_CSV, checks=chk, docs=docs)


# ----------------------------------------------------------------------------------------------
# F15 EXP_root against the squared stage-1 error
# ----------------------------------------------------------------------------------------------

def _ols(x2: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """OLS of y = a + b x2: (a, b, R^2) (numpy lstsq, as tools/v2/confirmation_analysis.py)."""
    A = np.column_stack([np.ones_like(x2), x2])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    r2 = 1.0 - float(res @ res) / float(((y - y.mean()) ** 2).sum())
    return float(coef[0]), float(coef[1]), r2


def build_f15(pack: C.Pack, ctx: Ctx) -> None:
    """F15: EXP_root/dW against the squared stage-1 error, with the OLS fit per q."""
    ev = _csv(f"{CA}/exp_vs_s1.csv").sort_values(["q", "seed"]).reset_index(drop=True)
    fit = _csv(f"{CA}/exp_vs_s1_fit.csv")
    fig, axes = style.new_figure(1, 2, height=3.4)
    rows: List[Dict[str, Any]] = []
    fits: List[Dict[str, Any]] = []
    d_fit = d_src = 0.0
    for j, q in enumerate(QS):
        ax = axes[0, j]
        g = ev[ev["q"] == q]
        x2 = g["stage1_rel_err_sq"].to_numpy(dtype=float)
        y = g["EXP_root_over_dw"].to_numpy(dtype=float)
        a, b, r2 = _ols(x2, y)
        f = fit[fit["q"] == q].iloc[0]
        d_fit = max(d_fit, abs(a - f["intercept"]), abs(b - f["slope_on_err_sq"]),
                    abs(r2 - f["r2"]))
        rep = ctx.rows(q)
        d_src = max(d_src,
                    C.max_abs_diff(g["stage1_rel_err_signed"], rep["B_stage1_rel_err_signed"]),
                    C.max_abs_diff(y, rep["B_EXP_root_over_dw"]),
                    C.max_abs_diff(x2, g["stage1_rel_err_signed"].to_numpy(dtype=float) ** 2))
        fits.append({"q": q, "n": len(g), "intercept": a, "slope_on_err_sq": b, "r2": r2})
        col = style.Q_COLORS[q]
        ax.plot(x2, y, ls="none", marker=style.Q_MARKER[q], color=col, ms=4.5,
                label=style.label_n("runs", len(g)))
        xl = np.array([0.0, float(x2.max())])
        ax.plot(xl, a + b * xl, color=col, ls=style.Q_LINESTYLE[q], lw=1.3,
                label=f"OLS: a = {a:.2e}, b = {b:.3f}, $R^2$ = {r2:.3f}")
        for s, xx, xs, yy in zip(g["seed"], g["stage1_rel_err_signed"], x2, y):
            rows.append({"q": q, "series": "run", "seed": int(s), "stage1_rel_err_signed": xx,
                         "x_err_sq": xs, "y_exp_root_over_dw": yy})
        for xx in xl:
            rows.append({"q": q, "series": "ols_line", "seed": None, "x_err_sq": float(xx),
                         "y_exp_root_over_dw": a + b * float(xx), "intercept": a,
                         "slope_on_err_sq": b, "r2": r2})
        ax.set_xlim(0.0, 1.05 * float(x2.max()))
        ax.set_ylim(0.0, 1.30 * max(float(y.max()), a + b * float(x2.max())))
        ax.set_title(f"q={q}")
        ax.set_xlabel(r"squared stage-1 error $((\hat e_1(0)-e_1^*)/e_1^*)^2$")
        ax.set_ylabel(r"EXP$_{root}/\Delta W$ (final tier)")
        ax.legend(loc="upper left")
    fdf = pd.DataFrame(fits)
    cc1 = pack.crosscheck("F15", fdf, REPORT,
                          header_has=["q", "n", "intercept", "slope_on_err_sq", "r2"],
                          key_map={"q": "q"},
                          value_map={"n": "n", "intercept": "intercept",
                                     "slope_on_err_sq": "slope_on_err_sq", "r2": "r2"},
                          label="OLS fit EXP_root/dW = a + b err^2 (report 4.6)")
    cc2 = pack.crosscheck("F15", ev, REPORT,
                          header_has=["q", "seed", "stage1_rel_err_sq", "EXP_root_over_dw"],
                          key_map={"q": "q", "seed": "seed"},
                          value_map={"stage1_rel_err_signed": "stage1_rel_err_signed",
                                     "stage1_rel_err_sq": "stage1_rel_err_sq",
                                     "EXP_root_over_dw": "EXP_root_over_dw"},
                          label="per-run stage-1 error and EXP_root (report 4.6)")
    passed: List[str] = []
    fq = fdf.set_index("q")
    for q, (ca, cb, cr) in ((50, ("2.5e-4", "0.270", "0.83")), (60, ("1.4e-4", "0.144", "0.98"))):
        _prose(pack, "F15", f"OLS intercept q={q}", fq.loc[q, "intercept"], ca, "4.6 text", passed)
        _prose(pack, "F15", f"OLS slope q={q}", fq.loc[q, "slope_on_err_sq"], cb, "4.6 text",
               passed)
        _prose(pack, "F15", f"OLS R^2 q={q}", fq.loc[q, "r2"], cr, "4.6 text", passed)
    chk = [f"OLS recomputed with numpy (lstsq on [1, err^2]) equals exp_vs_s1_fit.csv (max abs "
           f"diff of intercept, slope, R^2: {d_fit!r})",
           f"exp_vs_s1.csv equals reported_metrics.csv (B_stage1_rel_err_signed, "
           f"B_EXP_root_over_dw) and stage1_rel_err_sq = stage1_rel_err_signed^2 (max abs diff "
           f"{d_src!r})",
           "report prose consistent: " + "; ".join(passed),
           f"report cross-checks (4.6): fit {cc1['n_compared']} cells, per-run "
           f"{cc2['n_compared']} cells, {cc1['n_mismatch'] + cc2['n_mismatch']} mismatches"]
    data = pd.DataFrame(rows)
    data["seed"] = data["seed"].astype("Int64")
    docs = {
        "series": "run = one confirmation run; ols_line = end points of the plotted OLS line",
        "seed": "Run seed (empty for the OLS line)",
        "x_err_sq": {"definition": "Squared signed stage-1 error ((e_hat_1(0) - "
                                   "e1*(0))/e1*(0))^2 (x axis)",
                     "units": "fraction of e1*(0), squared", "tier": "tier-independent"},
        "y_exp_root_over_dw": {"definition": "EXP_root/dW, final tier (y axis; on ols_line rows "
                                             "the fitted value a + b x)",
                               "units": "Delta W (dimensionless)", "tier": "final"},
        "intercept": {"definition": "OLS intercept a of EXP_root/dW = a + b err^2 (recomputed)",
                      "units": "Delta W (dimensionless)", "tier": "final"},
        "slope_on_err_sq": {"definition": "OLS slope b on err^2 (recomputed)",
                            "units": "Delta W per unit err^2", "tier": "final"},
        "r2": {"definition": "Coefficient of determination R^2 of the OLS fit",
               "units": "dimensionless", "tier": "final"},
    }
    pack.figure("F15", fig, data, status="generated",
                sources=[C.src(f"{CA}/exp_vs_s1.csv"), C.src(f"{CA}/exp_vs_s1_fit.csv"),
                         C.src(f"{CA}/reported_metrics.csv")],
                script=f"{MOD}:build_f15", tier="final",
                caption="EXP_root/dW (final tier, end of Phase B) against the squared stage-1 "
                        "relative error err^2 = ((e_hat_1(0) - e1*(0))/e1*(0))^2 of the 20 "
                        "confirmation runs per q (protocol v1.1, seeds 20501-20520; one panel per "
                        "q), with the ordinary least-squares fit EXP_root/dW = a + b err^2 (line, "
                        "drawn over the range of the runs), recomputed with numpy. Source: "
                        f"{CA}/exp_vs_s1.csv; the fit equals {CA}/exp_vs_s1_fit.csv. n=20 per q.",
                notes=NOTE_CSV, checks=chk, docs=docs)


# ----------------------------------------------------------------------------------------------
# F16 learning curves
# ----------------------------------------------------------------------------------------------

# (phase, checkpoint column, panel title, y label, log y, threshold key or None)
F16_PANELS: List[Tuple[str, str, str, str, bool, Optional[str]]] = [
    ("A", "eta_T_over_dw", r"Phase A: $\eta_2$ (dev tier)", r"$\eta_2/\Delta W$", True,
     "eta_T_over_dw"),
    ("A", "stage2_rmse_pos_over_g2_0", "Phase A: RMSE on |d| < 2q", r"RMSE$/e_2^*(0)$", True,
     "stage2_rmse_pos_over_g2_0"),
    ("A", "stage2_peak_rel_err_signed", "Phase A: peak error at d = 0",
     r"$(\hat e_2(0)-e_2^*(0))/e_2^*(0)$", False, None),
    ("A", "stage2_tail_mean_over_g2_0", r"Phase A: tail mean on $|d|\geq 2q$",
     r"tail mean$/e_2^*(0)$", True, "stage2_tail_mean_over_g2_0"),
    ("A", "sigma_effort_at_0_t2", r"Phase A: $\sigma_2(0)$", r"$\sigma_2(0)$ (effort units)",
     False, None),
    ("B", "stage1_rel_err_signed", "Phase B: stage-1 error", r"$(\hat e_1(0)-e_1^*)/e_1^*$",
     False, "stage1_rel_err_abs"),
    ("B", "Gmax_full_over_dw", r"Phase B: $\hat G_{max,full}$ (dev tier)",
     r"$\hat G_{max,full}/\Delta W$", True, "Gmax_full_over_dw"),
    ("B", "EXP_root_over_dw", r"Phase B: EXP$_{root}$ (dev tier)", r"EXP$_{root}/\Delta W$", True,
     None),
]

_F16_TIER = ("development (eta_T_over_dw, Gmax_full_over_dw, EXP_root_over_dw); tier-independent "
             "(the recovery metrics, sigma_effort_at_0_t2, stage1_rel_err_signed)")
_KEYS_A = ("eta_T_over_dw", "stage2_rmse_pos_over_g2_0", "stage2_peak_rel_err_signed",
           "stage2_tail_mean_over_g2_0", "sigma_effort_at_0_t2")
_KEYS_B = ("stage1_rel_err_signed", "Gmax_full_over_dw", "EXP_root_over_dw")


def _f16_load(ctx: Ctx) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Checkpoint rows on the common update grid, the alignment record and the u1600/u2200 check."""
    A, B = [], []
    sets: Dict[str, Dict[Tuple[int, ...], List[str]]] = {"A": {}, "B": {}}
    extra: List[str] = []
    dmax = {"A": 0.0, "B": 0.0}
    for q in QS:
        for s in SEEDS:
            for ph, store in (("A", A), ("B", B)):
                c = _csv(f"{_rd(q, s)}/v2_checkpoints_{ph}.csv")
                if not ((c["phase"] == ph).all() and (c["verifier_tier"] == "development").all()):
                    raise ValueError(f"unexpected checkpoint rows in q={q} seed={s} phase {ph}")
                sets[ph].setdefault(tuple(int(u) for u in c["update"]), []).append(f"q{q}/s{s}")
                store.append(c.assign(q=q, seed=s))
            gr = ctx.gates[(q, s)]["reported"]
            ra = A[-1][A[-1]["update"] == 1600].iloc[0]
            rb = B[-1][B[-1]["update"] == 2200].iloc[0]
            for k in _KEYS_A:
                dmax["A"] = max(dmax["A"],
                                abs(float(ra[k]) - float(gr["end_of_A"]["development"][k])))
            for k in _KEYS_B:
                dmax["B"] = max(dmax["B"],
                                abs(float(rb[k]) - float(gr["end_of_B"]["development"][k])))
    A_df, B_df = pd.concat(A, ignore_index=True), pd.concat(B, ignore_index=True)
    rec: List[str] = []
    for ph, df in (("A", A_df), ("B", B_df)):
        common = sorted(set.intersection(*[set(t) for t in sets[ph]]))
        for t, runs in sorted(sets[ph].items(), key=lambda kv: -len(kv[1])):
            ex = sorted(set(t) - set(common))
            ex_txt = f", extra rows at {', '.join(f'u{u}' for u in ex)} in {', '.join(runs)}"
            rec.append(f"Phase {ph}: {len(runs)} run(s) with {len(t)} checkpoint rows "
                       f"(u{t[0]}-u{t[-1]})" + (ex_txt if ex else ""))
            if ex:
                for _, r in df[df["update"].isin(ex)].iterrows():
                    extra.append(f"q{int(r['q'])}/seed{int(r['seed'])} u{int(r['update'])} "
                                 f"(reason '{r['reason']}')")
        step = f" (step {common[1] - common[0]})" if len(common) > 1 else ""
        rec.append(f"Phase {ph}: common update grid of all 40 runs: {len(common)} updates, "
                   f"u{common[0]}-u{common[-1]}" + step)
        keep = df["update"].isin(common)
        if ph == "A":
            A_df = df[keep].copy()
        else:
            B_df = df[keep].copy()
    checks = [f"u1600 rows equal gates.json reported.end_of_A.development ({', '.join(_KEYS_A)}; "
              f"max abs diff {dmax['A']!r}, 40 runs)",
              f"u2200 rows equal gates.json reported.end_of_B.development ({', '.join(_KEYS_B)}; "
              f"max abs diff {dmax['B']!r}, 40 runs)"]
    if extra:
        rec.append(f"excluded off-grid rows: {'; '.join(extra)}")
    return A_df, B_df, rec, checks


def build_f16(pack: C.Pack, ctx: Ctx) -> None:
    """F16: learning curves of the confirmation runs (median and IQR per q), checkpoint CSVs."""
    A_df, B_df, align, chk0 = _f16_load(ctx)
    fig, axes = style.new_figure(4, 2, height=9.0)
    parts: List[pd.DataFrame] = []
    for k, (ph, col, title, ylab, logy, tkey) in enumerate(F16_PANELS):
        ax = axes[k // 2, k % 2]
        df = A_df if ph == "A" else B_df
        for q in QS:
            g = df[df["q"] == q]
            n_runs = int(g["seed"].nunique())
            out = style.median_iqr_curves(ax, g, "update", col, style.Q_COLORS[q],
                                          style.label_n(f"q={q}", n_runs),
                                          ls=style.Q_LINESTYLE[q], marker=style.Q_MARKER[q],
                                          extra={"q": q, "phase": ph, "series": "median_iqr"})
            if (out["n"] != 20).any():
                raise ValueError(f"F16: n != 20 at some update ({col}, q={q})")
            parts.append(out)
        if tkey is not None:
            thr = ctx.th[tkey]
            s1 = tkey == "stage1_rel_err_abs"
            lab = "S1 threshold (secondary)" if s1 else "gate threshold"
            for i, yv in enumerate([thr, -thr] if s1 else [thr]):
                ax.axhline(yv, color=style.THRESH, ls=("-." if s1 else "--"), lw=1.0,
                           label=lab if i == 0 else None)
                parts.append(pd.DataFrame([{"metric": col, "phase": ph, "series": "threshold",
                                            "value": yv, "criterion": tkey}]))
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(_PlainLog(labelOnlyBase=False))
            ax.yaxis.set_minor_formatter(_PlainLog(labelOnlyBase=False, minor_thresholds=(1, 0.4)))
        elif col != "sigma_effort_at_0_t2":
            ax.axhline(0.0, color=style.REF, lw=0.6)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xlabel("global update")
        ax.legend(loc="lower right" if col == "stage2_peak_rel_err_signed" else "upper right")
    data = pd.concat(parts, ignore_index=True)
    data = data[["series", "phase", "q", "metric", "update", "median", "q25", "q75", "min", "max",
                 "n", "value", "criterion"]]
    for c in ("q", "update", "n"):
        data[c] = data[c].astype("Int64")
    chk = chk0 + [f"alignment: {a}" for a in align] + [
        "decision: the checkpoint updates align across runs on the common grid, so the curves use "
        "the checkpoint CSVs (training-time verifier, development tier); no weight export was "
        "re-evaluated",
        "median and IQR computed per (q, update) over the 20 runs with numpy linear percentiles"]
    docs = {
        "series": "median_iqr = across-seed summary at one update; threshold = plotted threshold "
                  "line",
        "phase": "Training phase of the panel: A (stage 2 only, u1-u1600) or B (stage 1 with "
                 "frozen stage 2, u1601-u2200)",
        "metric": "Checkpoint column (v2_checkpoints_{A,B}.csv): eta_T_over_dw (dev tier), "
                  "stage2_rmse_pos_over_g2_0, stage2_peak_rel_err_signed, "
                  "stage2_tail_mean_over_g2_0 and sigma_effort_at_0_t2 (tier-independent); "
                  "stage1_rel_err_signed (tier-independent); Gmax_full_over_dw and "
                  "EXP_root_over_dw (dev tier)",
        "update": "Global update of the checkpoint",
        "median": {"definition": "Median over the 20 runs at the update", "tier": _F16_TIER},
        "q25": {"definition": "25th percentile over the runs (numpy linear interpolation)",
                "tier": _F16_TIER},
        "q75": {"definition": "75th percentile over the runs (numpy linear interpolation)",
                "tier": _F16_TIER},
        "min": {"definition": "Minimum over the runs", "tier": _F16_TIER},
        "max": {"definition": "Maximum over the runs", "tier": _F16_TIER},
        "n": {"definition": "Number of runs at the update", "units": "count"},
        "value": {"definition": "threshold rows: y value of the dashed line",
                  "units": "as the metric"},
        "criterion": "threshold rows: protocol criterion metric the threshold belongs to (G-A: "
                     "eta_T_over_dw, stage2_rmse_pos_over_g2_0, stage2_tail_mean_over_g2_0; G-F: "
                     "Gmax_full_over_dw; S1: stage1_rel_err_abs, drawn at +-0.10)",
    }
    pack.figure("F16", fig, data, status="generated",
                sources=[_set("v2_checkpoints_A.csv"), _set("v2_checkpoints_B.csv"),
                         _set("gates.json"), C.src(PROTO)],
                script=f"{MOD}:build_f16", tier="development",
                caption="Learning curves of the 40 confirmation runs under the locked v1.1 "
                        "pipeline (seeds 20501-20520): median (line) and IQR (band) over the 20 "
                        "runs per q "
                        "at each training-time verifier checkpoint (v2_checkpoints_A.csv: "
                        "u100-u1600 every 100; v2_checkpoints_B.csv: u1700-u2200 every 25). Phase "
                        "A panels: eta_2/dW, RMSE/e2*(0), signed peak error at d=0, tail "
                        "mean/e2*(0) (normalised by e2*(0), not raw) and sigma_2(0) of the stage-2 "
                        "last iterate; Phase B panels: signed stage-1 error, Gmax_full/dW and "
                        "EXP_root/dW of the live stage-1 actor with the frozen stage 2. eta_2, "
                        "Gmax_full and EXP_root are development tier (state step 4, effort step 1, "
                        "GL 16); the recovery metrics, sigma_2(0) and the stage-1 error are "
                        "tier-independent. Dashed: thresholds of G-A / G-F; dash-dot: the S1 band "
                        "(+-0.10, secondary); the gates are applied on the final tier at u1600 / "
                        "u2200. Log y axis for the deviation metrics, RMSE and tail.",
                notes=NOTE_CSV, checks=chk, docs=docs)


# ----------------------------------------------------------------------------------------------
# F17 G_t(d) for two runs per q
# ----------------------------------------------------------------------------------------------

def _pick_runs(g: pd.DataFrame) -> List[Tuple[str, int, float]]:
    """(role, seed, gmax) of the median-Gmax run and the max-Gmax run of one q.

    With n = 20 the median is the mean of the values at ranks 10 and 11, which are equidistant
    from it; the median run is the one of these two with the lower seed. The max run has the
    largest value (ties -> lower seed).
    """
    srt = g.sort_values(["gmax_final", "seed"]).reset_index(drop=True)
    n = len(srt)
    if n % 2:
        med_row = srt.iloc[n // 2]
    else:
        med_row = srt.iloc[[n // 2 - 1, n // 2]].sort_values("seed").iloc[0]
    mx = g[g["gmax_final"] == g["gmax_final"].max()].sort_values("seed").iloc[0]
    return [("median Gmax run", int(med_row["seed"]), float(med_row["gmax_final"])),
            ("max Gmax run", int(mx["seed"]), float(mx["gmax_final"]))]


def build_f17(pack: C.Pack, ctx: Ctx) -> None:
    """F17: G_2(d)/dW on D_2 with G_1(0)/dW and (t*, d*) for the median and max Gmax run per q."""
    from matplotlib.lines import Line2D

    fig, axes = style.new_figure(2, 2, height=6.0)
    rows: List[Dict[str, Any]] = []
    chk: List[str] = []
    picks_txt: List[str] = []
    d_g = d_loc = d_exp = 0.0
    for i, q in enumerate(QS):
        g = ctx.pr(q)
        med_all = float(np.median(g["gmax_final"]))
        rep = ctx.rows(q).set_index("seed")
        dw = ctx.dw[q]
        for j, (role, s, gm) in enumerate(_pick_runs(g)):
            z = ctx.npz(q, s, "final_final")
            G1, G2 = z["v_t1_G"] / dw, z["v_t2_G"] / dw
            D1, D2 = z["v_t1_d_grid"], z["v_t2_d_grid"]
            t_star = 1 if G1.max() >= G2.max() else 2
            if t_star == 1:
                d_star = float(D1[int(np.argmax(G1))])
            else:
                d_star = float(D2[int(np.argmax(G2))])
            g_star = float(max(G1.max(), G2.max()))
            d_g = max(d_g, abs(g_star - gm))
            d_loc = max(d_loc, abs(t_star - float(rep.loc[s, "B_Gmax_full_t"])),
                        abs(d_star - float(rep.loc[s, "B_Gmax_full_d"])))
            d_exp = max(d_exp, abs(float(G1[0]) - float(rep.loc[s, "B_EXP_root_over_dw"])))
            ax = axes[i, j]
            ax.plot(D2, G2, color=style.Q_COLORS[q], ls=style.Q_LINESTYLE[q], lw=1.2)
            ax.plot([float(D1[0])], [float(G1[0])], ls="none", marker="D", color=style.REF, ms=5)
            ax.plot([d_star], [g_star], ls="none", marker="o", mfc="none", mec=style.THRESH,
                    mew=1.3, ms=10)
            ax.set_title(f"q={q}, seed {s} ({role.replace(' Gmax', '')})")
            ax.text(0.02, 0.97, rf"$\hat G_{{max,full}}/\Delta W$ = {g_star:.3g}"
                    + f"\n(t*, d*) = ({t_star}, {d_star:g})", transform=ax.transAxes, va="top",
                    ha="left", fontsize=8)
            ax.set_xlabel("gap d (effort units)")
            ax.set_ylabel(r"$G_t(d)/\Delta W$ (final tier)")
            ax.set_xlim(float(D2[0]), float(D2[-1]))
            yl = ax.get_ylim()
            ax.set_ylim(min(0.0, yl[0]), yl[1] + 0.30 * (yl[1] - min(0.0, yl[0])))
            picks_txt.append(f"q={q} {role}: seed {s} (gmax_final {_fmt(gm)}; median of q "
                             f"{_fmt(med_all)})")
            for t, Dg, Gg in ((1, D1, G1), (2, D2, G2)):
                for dd, gg in zip(Dg, Gg):
                    rows.append({"q": q, "role": role, "seed": s, "t": t, "d": float(dd),
                                 "G_over_dw": float(gg),
                                 "is_argmax": bool(t == t_star and float(dd) == d_star)})
    handles = [Line2D([], [], color=style.Q_COLORS[q], ls=style.Q_LINESTYLE[q], lw=1.2)
               for q in QS]
    handles += [Line2D([], [], ls="none", marker="D", color=style.REF, ms=5),
                Line2D([], [], ls="none", marker="o", mfc="none", mec=style.THRESH, mew=1.3,
                       ms=10)]
    labels = [rf"$G_2(d)/\Delta W$ on $D_2$, q={q}" for q in QS]
    labels += [r"$G_1(0)/\Delta W$ at the root (= EXP$_{root}/\Delta W$)",
               r"argmax $(t^*, d^*)$ of $\hat G_{max,full}$"]
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    chk.append(f"max over t and d of G_t(d)/dW equals gmax_final in per_run.csv for the 4 runs "
               f"(max abs diff {d_g!r})")
    chk.append(f"(t*, d*) equals B_Gmax_full_t / B_Gmax_full_d in reported_metrics.csv (max abs "
               f"diff {d_loc!r}; ties go to t = 1 as in utils.v2_metrics.evaluate)")
    chk.append(f"G_1(0)/dW equals B_EXP_root_over_dw (max abs diff {d_exp!r})")
    chk.append("selected runs: " + "; ".join(picks_txt))
    data = pd.DataFrame(rows)
    docs = {
        "role": "Which run of the q: median Gmax run (of the two runs at ranks 10 and 11 of "
                "gmax_final, whose mean is the median, the lower seed) or max Gmax run (largest "
                "gmax_final)",
        "t": {"definition": "Stage t of the value (1: root node d = 0; 2: stage-2 grid D_2)",
              "units": "stage index"},
        "d": {"definition": "Gap d of the grid node", "units": "effort units (gap d)"},
        "G_over_dw": {"definition": "G_t(d)/dW = (V_t^BR(d) - V_t^mean(d))/dW from "
                                    "final_final.npz (v_t1_G, v_t2_G divided by dW = 4)",
                      "units": "Delta W (dimensionless)", "tier": "final"},
        "is_argmax": "Whether the node is the argmax (t*, d*) of Gmax_full (ties to t = 1)",
    }
    pack.figure("F17", fig, data, status="generated",
                sources=[C.src(f"{CA}/per_run.csv"), C.src(f"{CA}/reported_metrics.csv"),
                         _set("final_final.npz"), C.src(PROTO)],
                script=f"{MOD}:build_f17", tier="final",
                caption="Deviation gains G_t(d)/dW on the final-tier grids of the full last "
                        "iterate (end of Phase B, u2200; live stage-1 actor, frozen stage 2) for "
                        "two confirmation runs per q (protocol v1.1): the run with the median "
                        "Gmax_full/dW (of the two runs at ranks 10 and 11 out of 20, whose mean "
                        "is the median, the lower seed) and the run with the max. Line: G_2(d)/dW "
                        "on D_2 (step 2); black diamond: G_1(0)/dW at the root (= EXP_root/dW); "
                        "red circle: the argmax (t*, d*). Source: final_final.npz of each run "
                        "(v_t1_G, v_t2_G).",
                checks=chk, docs=docs)


# ----------------------------------------------------------------------------------------------
# T41 targets vs results
# ----------------------------------------------------------------------------------------------

_T41_TIER = ("final (Gmax_full, calibration values); tier-independent (stage-1 error, peak "
             "errors, RMSE, tail mean); dev - final (targets 6a/6b)")


class _T41Rows:
    """Row builder of T41 (one row per target and q)."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def stat(self, tid: str, target: str, orig: str, orig_thr: Optional[float], locked: str,
             role: str, q: int, metric: str, units: str, v: np.ndarray,
             locked_thr: Optional[float], source: str, note: str = "",
             v_locked: Optional[np.ndarray] = None) -> None:
        """Row with a per-run distribution; ``v_locked``: the locked criterion's own metric."""
        st = C.median_iqr(v)
        vl = v if v_locked is None else v_locked
        n_orig = int((v <= orig_thr).sum()) if orig_thr is not None else None
        n_lock = int((vl <= locked_thr).sum()) if locked_thr is not None else None
        self.rows.append({"target_id": tid, "target": target, "original_threshold": orig,
                          "locked_criterion": locked, "role": role, "q": q, "metric": metric,
                          "units": units, "n": st["n"], "median": st["median"], "min": st["min"],
                          "max": st["max"], "value": np.nan, "n_meet_original_target": n_orig,
                          "n_meet_locked_criterion": n_lock, "source": source, "note": note})

    def single(self, **kw: Any) -> None:
        """Row with a single value (calibration value or pass rate)."""
        self.rows.append({"median": np.nan, "min": np.nan, "max": np.nan,
                          "n_meet_original_target": None, **kw})


def _t41_targets_1_6(T: _T41Rows, ctx: Ctx, q: int) -> float:
    """Rows of the targets with a per-run distribution; returns the raw-tail check deviation."""
    th = ctx.th
    pr, rp = ctx.pr(q), ctx.rows(q)
    e2s0 = ctx.e2s0[q]
    tail_raw = np.array([float(ctx.gates[(q, s)]["reported"]["end_of_A"]["final"]
                               ["stage2_tail_mean"]) for s in SEEDS])
    tail_dev = C.max_abs_diff(tail_raw, pr["tail"].to_numpy(dtype=float) * e2s0)
    s1s = pr["stage1_rel_err_signed"].to_numpy(dtype=float)
    T.stat("1", "stage-1 relative error <= 5%", "<= 0.05", 0.05,
           f"S1: |stage-1 error| <= {th['stage1_rel_err_abs']:g} (secondary criterion, no pass "
           "rule; moved from G-F in v1.1)", "secondary (S1)", q,
           "|stage-1 error| = |e_hat_1(0) - e1*(0)|/e1*(0) (s1)", _FR1,
           pr["s1"].to_numpy(dtype=float), th["stage1_rel_err_abs"], f"{CA}/per_run.csv: s1",
           f"signed error: median {_fmt(np.median(s1s))}, range [{_fmt(s1s.min())}, "
           f"{_fmt(s1s.max())}]")
    pk = rp["A_stage2_peak_rel_err_signed"].to_numpy(dtype=float)
    T.stat("2", "stage-2 peak relative error <= 5%", "<= 0.05", 0.05,
           "reported only (not gated)", "reported only", q,
           "|peak error at d = 0| = |e_hat_2(0) - e2*(0)|/e2*(0)", _FR2, np.abs(pk), None,
           f"{CA}/reported_metrics.csv: |A_stage2_peak_rel_err_signed|",
           f"signed: median {_fmt(np.median(pk))}, range [{_fmt(pk.min())}, {_fmt(pk.max())}]")
    lf = rp["A_stage2_peak_locfree_rel_err"].to_numpy(dtype=float)
    T.stat("2b", "stage-2 peak relative error <= 5% (location-free version)", "<= 0.05", 0.05,
           "reported only (not gated)", "reported only", q,
           "|location-free peak error| = |max_d e_hat_2(d) - e2*(0)|/e2*(0)", _FR2, np.abs(lf),
           None, f"{CA}/reported_metrics.csv: |A_stage2_peak_locfree_rel_err|",
           f"signed: median {_fmt(np.median(lf))}, range [{_fmt(lf.min())}, {_fmt(lf.max())}]")
    T.stat("3", "positive-region normalised RMSE <= 5%", "<= 0.05", 0.05,
           f"G-A: RMSE/e2*(0) <= {th['stage2_rmse_pos_over_g2_0']:g}", "gate (G-A)", q,
           "RMSE over |d| < 2q / e2*(0) (rmse)", _FR2, pr["rmse"].to_numpy(dtype=float),
           th["stage2_rmse_pos_over_g2_0"], f"{CA}/per_run.csv: rmse")
    thr_t = th["stage2_tail_mean_over_g2_0"]
    T.stat("4", "tail mean effort <= 2 (effort units)", "<= 2 effort units", 2.0,
           f"G-A: tail mean/e2*(0) <= {thr_t:g} (= {thr_t * e2s0:.4g} effort units at q={q})",
           "gate (G-A)", q, "tail mean of e_hat_2 over |d| >= 2q, raw",
           "effort units [0, 100], raw", tail_raw, thr_t,
           f"{ROOT}/q{q}/seed*/gates.json: reported.end_of_A.final.stage2_tail_mean",
           f"raw tail mean = tail x e2*(0) with e2*(0) = {e2s0:.6g}; normalised tail (G-A "
           f"metric, per_run.csv tail): median {_fmt(np.median(pr['tail']))}, max "
           f"{_fmt(pr['tail'].max())}; n_meet_locked_criterion counts tail <= 0.02",
           v_locked=pr["tail"].to_numpy(dtype=float))
    T.stat("5", "Gmax_full/dW <= 0.01", "<= 0.01", 0.01,
           f"G-F: Gmax_full/dW <= {th['Gmax_full_over_dw']:g} (final tier)", "gate (G-F)", q,
           "Gmax_full/dW, final tier (gmax_final)", _DW, pr["gmax_final"].to_numpy(dtype=float),
           th["Gmax_full_over_dw"], f"{CA}/per_run.csv: gmax_final")
    for tid, col, lab in (("6a", "eta_dev_minus_final", "eta_2"),
                          ("6b", "gmax_dev_minus_final", "Gmax_full")):
        key = ("eta_T_over_dw" if col.startswith("eta") else "Gmax_full_over_dw")
        key += "_dev_minus_final_abs"
        T.stat(tid, "a formal dev-final refinement threshold",
               "none specified (a threshold to be defined)", None,
               f"G-N: |dev - final| of eta_2 and of Gmax_full each <= {th[key]:g} dW",
               "gate (G-N)", q, f"|{lab} dev - final|/dW", _DW,
               np.abs(pr[col].to_numpy(dtype=float)), th[key], f"{CA}/per_run.csv: |{col}|",
               "the 0929 target set no number; n_meet_locked_criterion counts runs within the "
               "G-N threshold")
    return tail_dev


def _t41_targets_7_9(T: _T41Rows, ctx: Ctx, q: int, cal: pd.DataFrame, pc: pd.DataFrame,
                     verdict: Dict[str, Any]) -> None:
    """Rows of the targets without a per-run distribution (calibration, discrimination, pass)."""
    pr = ctx.pr(q)
    gm = pr["gmax_final"]
    for tid, pol, target, locked, extra in (
            ("7", "analytic_eq",
             "verifier calibration: the exact equilibrium sits at the numerical floor",
             "not part of the locked protocol; lock-time calibration of the analytic equilibrium "
             "(final and development tier)", ""),
            ("8", "zero", "discrimination: misspecified policies show clearly larger deviation",
             "not part of the locked protocol; tested only on the zero-effort policy (lock-time "
             "calibration)",
             f"; confirmation runs, Gmax_full/dW final tier: median {_fmt(np.median(gm))}, max "
             f"{_fmt(gm.max())}")):
        cf = cal[(cal["q"] == q) & (cal["policy"] == pol)].set_index("tier")
        fin, dev = cf.loc["final"], cf.loc["development"]
        pi = ""
        if pol == "zero":
            pi = (f"; PI-side reference {fin['pi_ref_Gmax']:g} at (t, d) = "
                  f"({int(fin['pi_ref_t'])}, {fin['pi_ref_d']:g})")
        who = "analytic equilibrium" if pol == "analytic_eq" else "zero-effort"
        T.single(target_id=tid, target=target, original_threshold="none (qualitative)",
                 locked_criterion=locked, role="not in the locked protocol", q=q,
                 metric=f"Gmax_full/dW of the {who} policy, final tier (lock-time calibration)",
                 units=_DW, n=1, value=float(fin["Gmax_full_over_dw"]),
                 n_meet_locked_criterion=None,
                 source=f"{CAL}: q={q}, policy={pol}, tier=final / Gmax_full_over_dw",
                 note=(f"final tier {_fmt(fin['Gmax_full_over_dw'], 6)} at (t*, d*) = "
                       f"({int(fin['Gmax_full_t'])}, {fin['Gmax_full_d']:g}); development tier "
                       f"{_fmt(dev['Gmax_full_over_dw'], 6)}; not measured on the confirmation "
                       "runs" + pi + extra))
    p = pc.loc[q]
    T.single(target_id="9", target="reliability on fresh held-out seeds",
             original_threshold="none (qualitative)",
             locked_criterion=f"confirmation pass rule: {p['rule']} runs pass G-A and G-F and "
                              "G-N, for each q; the confirmation passes iff both q pass",
             role="pass rule", q=q,
             metric="run pass (G-A and G-F and G-N, exit code 0, no global-RNG violation) on "
                    "seeds 20501-20520",
             units="fraction of runs", n=int(p["n_expected"]), value=float(p["pass_rate"]),
             n_meet_locked_criterion=int(p["n_pass"]),
             source=f"{CA}/pass_counts.csv; {CA}/verdict.json",
             note=(f"exact Clopper-Pearson 95% CI [{_fmt(p['cp95_lo'])}, {_fmt(p['cp95_hi'])}]; "
                   f"q passes the rule: {bool(p['q_passes_rule'])}; overall verdict "
                   f"{verdict['overall']}; value = pass rate"))


_T41_COLS = ["target_id", "target", "original_threshold", "locked_criterion", "role", "q",
             "metric", "units", "n", "median", "min", "max", "value", "n_meet_original_target",
             "n_meet_locked_criterion", "source", "note"]


def build_t41(pack: C.Pack, ctx: Ctx) -> None:
    """T41: each 0929 target, the locked criterion it became, and the confirmation result per q."""
    cal = _csv(CAL)
    pc = _csv(f"{CA}/pass_counts.csv").set_index("q")
    verdict = S.read_json(f"{CA}/verdict.json")
    T = _T41Rows()
    tail_dev = 0.0
    for q in QS:
        tail_dev = max(tail_dev, _t41_targets_1_6(T, ctx, q))
        _t41_targets_7_9(T, ctx, q, cal, pc, verdict)
    out = pd.DataFrame(T.rows)[_T41_COLS]
    out["n"] = out["n"].astype(int)
    for c in ("n_meet_original_target", "n_meet_locked_criterion"):
        out[c] = _int_or_none(out[c])
    out["_o"] = out["target_id"].map(lambda t: (int(t[0]), t[1:]))
    out = out.sort_values(["_o", "q"], kind="stable").drop(columns="_o").reset_index(drop=True)
    # cross-checks with the report: 4.4 distributions, 4.3 S1 passes, 4.1 pass counts
    mapm = {"3": "rmse", "5": "gmax_final", "1": "s1"}
    dist = out[out["target_id"].isin(mapm)].assign(metric=lambda d: d["target_id"].map(mapm))
    cc1 = pack.crosscheck("T41", dist[["q", "metric", "median", "min", "max"]], REPORT,
                          header_has=["root", "q", "metric", "median", "p90"],
                          key_map={"q": "q", "metric": "metric"},
                          value_map={"median": "median", "min": "min", "max": "max"},
                          label="median/min/max of rmse, gmax_final, s1 (report 4.4)")
    s1c = out[out["target_id"] == "1"][["q", "n_meet_locked_criterion"]].rename(
        columns={"n_meet_locked_criterion": "S1_pass"})
    cc2 = pack.crosscheck("T41", s1c, REPORT, header_has=["q", "n", "S1_pass", "bootstrap_seed"],
                          key_map={"q": "q"}, value_map={"S1_pass": "S1_pass"},
                          label="S1 passes (report 4.3)")
    pcc = out[out["target_id"] == "9"][["q", "n_meet_locked_criterion"]].rename(
        columns={"n_meet_locked_criterion": "n_pass"})
    cc3 = pack.crosscheck("T41", pcc, REPORT, header_has=["q", "n_expected", "n_pass"],
                          key_map={"q": "q"}, value_map={"n_pass": "n_pass"}, heading_has="4.1",
                          label="primary passes (report 4.1)")
    n_mis = cc1["n_mismatch"] + cc2["n_mismatch"] + cc3["n_mismatch"]
    notes = ("Targets: the 0929 validation targets (SPEC 5.4, PI-supplied; source: PI "
             "instruction, not recorded in the repository). Locked criteria and thresholds read "
             "from protocols/v2_T2_locked_v1_1.json (gates G-A, G-F, G-N; secondary S1; "
             "confirmation pass rule). Per-run values: per_run.csv (s1, rmse, gmax_final, |dev - "
             "final|), reported_metrics.csv (peak errors), gates.json (raw tail mean); targets 7 "
             f"and 8 from the lock-time calibration ({CAL}); target 9 from pass_counts.csv and "
             "verdict.json. median/min/max over the 20 seeds per q (numpy); "
             "n_meet_original_target = runs with metric <= the 0929 threshold (empty where the "
             "target has no number); n_meet_locked_criterion = runs within the locked threshold "
             f"(target 9: runs that pass). {NOTE_CSV} Checks: raw tail mean (gates.json) equals "
             f"tail x e2*(0) from per_run.csv (max abs diff {tail_dev!r}); report cross-checks: "
             f"4.4 {cc1['n_compared']} cells, 4.3 {cc2['n_compared']} cells, 4.1 "
             f"{cc3['n_compared']} cells, {n_mis} mismatches.")
    docs = {
        "target_id": "Number of the 0929 target in the order of SPEC 5.4 (2b: location-free "
                     "variant of target 2; 6a/6b: eta_2 and Gmax_full parts of G-N)",
        "target": "0929 validation target (PI-supplied, SPEC 5.4)",
        "original_threshold": "Threshold of the 0929 target as stated by the PI",
        "locked_criterion": "Locked v1.1 criterion the target became "
                            "(protocols/v2_T2_locked_v1_1.json), or 'reported only' / 'not part "
                            "of the locked protocol'",
        "role": "Role in the locked protocol: gate (part of the run pass), secondary (S1), "
                "reported only, pass rule, or not in the locked protocol",
        "metric": "Per-run quantity summarised (column of the source file), or the single "
                  "calibration quantity",
        "units": "Units of median/min/max/value",
        "n": {"definition": "Number of runs (or calibration evaluations) behind the row",
              "units": "count"},
        "median": {"definition": "Median over the confirmation runs of the q", "tier": _T41_TIER},
        "min": {"definition": "Minimum over the confirmation runs", "tier": _T41_TIER},
        "max": {"definition": "Maximum over the confirmation runs", "tier": _T41_TIER},
        "value": {"definition": "Single value for rows without a per-run distribution: targets "
                                "7 and 8 = lock-time calibration Gmax_full/dW (final tier); "
                                "target 9 = pass rate",
                  "units": "as stated in 'units'"},
        "n_meet_original_target": {"definition": "Runs (of n) whose metric meets the 0929 "
                                                 "threshold (<=); empty when the target has no "
                                                 "numeric threshold", "units": "count"},
        "n_meet_locked_criterion": {"definition": "Runs (of n) within the threshold of the "
                                                  "locked criterion (target 9: runs that pass); "
                                                  "empty when there is none", "units": "count"},
        "source": "File and column the values were read from",
        "note": "Additional values (signed ranges, tier, references); factual only",
    }
    pack.table("T41", out, status="generated",
               sources=[C.src(f"{CA}/per_run.csv"), C.src(f"{CA}/reported_metrics.csv"),
                        C.src(f"{CA}/pass_counts.csv"), C.src(f"{CA}/verdict.json"), C.src(CAL),
                        C.src(PROTO), _set("gates.json")],
               script=f"{MOD}:build_t41", notes=notes, docs=docs, tier="final",
               caption="Each 0929 validation target (SPEC 5.4), the locked v1.1 criterion it "
                       "became, and the confirmation result per q (fresh seeds 20501-20520, n=20 "
                       "per q): median, min and max of the per-run metric and the number of runs "
                       "that meet the original target and the locked criterion. Targets 7 and 8 "
                       "are not measured on the confirmation runs; their rows give the lock-time "
                       "calibration value.")


# ----------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------

def build() -> None:
    """Build T39, F12-F17 and T41 from the confirmation records (no re-evaluation)."""
    pack = C.Pack("sec_locked_b")
    ctx = Ctx()
    build_t39(pack, ctx)
    build_f12(pack, ctx)
    build_f13(pack, ctx)
    build_f14(pack, ctx)
    build_f15(pack, ctx)
    build_f16(pack, ctx)
    build_f17(pack, ctx)
    build_t41(pack, ctx)
    pack.save_fragment()
