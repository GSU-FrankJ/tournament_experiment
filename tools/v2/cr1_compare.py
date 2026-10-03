#!/usr/bin/env python3
"""C-R1 and stage-wave reproducibility comparison against the v1.1 rehearsal.

Compares a new run root with the reference root (default: the canonical ``rehearsal_v1_1``) on
the training-relevant state as defined in ``tools/v2/v1_1_rehearsal_checks.py`` (``r1_one``) and
reports, per (q, seed), a dict of booleans and, for every difference, the FIRST differing field
with both values ("stop and report the first differing field"). ``ALL`` = every compared field
identical. The three process-global RNG states are excluded, as in v1.1. Wall-clock fields are
dropped.

Modes (``--mode``):
  cr1          the unchanged locked entry point reproduces the rehearsal: state_end_A/B.pt (actor,
               critic, opponent, frozen, opt_actor, opt_critic, rng_minibatch, rng streams, torch
               generator), all weight exports and checkpoint_weights.npz, train_history.json
               (history, stability, verifier_calls, curriculum, snapshots, weight_checkpoints),
               v2_updates.csv and v2_checkpoints_{A,B}.csv on the COMMON columns, gates.json
               (metric_values, dev_tier_values, reported, G-A, G-F, G-N, outcome), gateA_*/final_*/
               band_sweep NPZ, induced_band.json, drift_test.json.
  parentsA     a ``phase_A`` run (the u1200 parents) against the rehearsal: end-of-A state and the
               Phase-A part (updates 1..1600) of histories, logs, weight exports and CSVs; the
               end-of-A evaluation (final_*.npz against gateA_*.npz, final_v2.json scalars against
               gates.json reported.end_of_A).
  stage1_base  ``B_base`` (mode phase_B from the rehearsal state_end_A.pt) against the rehearsal:
               state_end_B.pt, the Phase-B part (updates 1601..2200) of histories, logs, exports,
               checkpoint_weights.npz, final_*.npz, and the end-of-B values (final_v2.json against
               gates.json reported.end_of_B, metric_values, dev_tier_values).
  stage2_base  ``A_base`` (mode phase_A_continue from u1200) against the rehearsal state_end_A.pt:
               like parentsA over the overlapping global updates 1201..1600. Cadence-dependent
               fields (``local``, stability drift/stable/consecutive, verifier-call reasons and
               counters, curriculum) restart with the phase and are not comparable; verifier-based
               rows are compared at the global updates present in both runs.

Columns present in only one CSV are reported (``info.csv_columns_only_in_one``), never compared;
``unexpected_new`` lists those that are not the known new columns (``d1_*``, ``n_epochs_run``,
``conc_scale``). The ``snapshot_refreshes`` counter and the other counters are reported
separately (``info``), they are not training state.

Usage:
    python tools/v2/cr1_compare.py --mode cr1 --new results/v2_refine/v11_reproduction \
        --out results/v2_refine/v11_reproduction_checks.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

import v1_1_rehearsal_checks as V1  # noqa: E402  (importable: has a main guard)

CANONICAL_ROOT = Path(
    "/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2")
REF_DEFAULT = CANONICAL_ROOT / "results" / "v2_T2_locked" / "rehearsal_v1_1"
WALL_KEYS = set(V1.WALL_KEYS)     # time_sec, update_wall_sec, elapsed_phase_wall_sec, verifier_sec
AGENT_KEYS = ("actor", "critic", "opponent", "frozen", "opt_actor", "opt_critic", "rng_minibatch")
NEW_COLUMN_PREFIXES = ("d1_",)
NEW_COLUMNS = ("n_epochs_run", "conc_scale")
IDENTITY_COLUMNS = {"run", "pilot", "arm", "mode", "reward_mode", "stage2_update_mode",
                    "adv_norm_scope", "continuation_action_mode"}
CADENCE_CSV = {"local", "reason", "consecutive_eligible"}
CADENCE_VERIFIER = {"local", "reason", "consecutive_eligible", "visitation_cumulative_phase"}
STABILITY_STATE_KEYS = ("update", "phase", "kl", "max_std_norm", "e_hat")
BIG = 10 ** 9
# metric_values key -> (tier, scalar key of final_v2.json); dev_tier_values blocks are development
METRIC_MAP = {"eta_final": ("final", "eta_T_over_dw"), "eta_dev": ("development", "eta_T_over_dw"),
              "rmse": ("final", "stage2_rmse_pos_over_g2_0"),
              "tail": ("final", "stage2_tail_mean_over_g2_0"),
              "gmax_final": ("final", "Gmax_full_over_dw"),
              "gmax_dev": ("development", "Gmax_full_over_dw"),
              "s1": ("final", "stage1_rel_err_abs")}
DEV_MAP = {("G-A", "eta_T_over_dw"): "eta_T_over_dw",
           ("G-A", "stage2_rmse_pos_over_g2_0"): "stage2_rmse_pos_over_g2_0",
           ("G-A", "stage2_tail_mean_over_g2_0"): "stage2_tail_mean_over_g2_0",
           ("G-F", "Gmax_full_over_dw"): "Gmax_full_over_dw",
           ("S1", "stage1_rel_err_abs"): "stage1_rel_err_abs"}


@dataclass(frozen=True)
class ModeSpec:
    """What one comparison mode compares (see the module docstring)."""

    name: str
    lo: int                              # first global update compared (logs, exports, CSV rows)
    hi: int                              # last global update compared
    states: Tuple[str, ...]              # full-state files (same name on both sides)
    phase: Optional[str]                 # curriculum phase compared (None = all)
    ckpt_csvs: Tuple[Tuple[str, Tuple[str, ...]], ...]   # (ref file, candidate new files)
    continuation: bool = False           # phase-local cadence restarts (stage2_base)
    cross_config: bool = False           # run/flag identity columns of the checkpoint CSV differ
    snap: str = "all"                    # snapshot-log selector
    full_gate_files: bool = False        # cr1: gates.json blocks + band/drift files
    npz_pairs: Tuple[Tuple[str, str], ...] = ()          # (ref npz, new npz)
    end_eval: Optional[str] = None       # None | "A" | "B": final_v2.json vs gates.json reported


_GATE_A_PAIRS = (("gateA_final.npz", "final_final.npz"),
                 ("gateA_development.npz", "final_development.npz"))
MODES: Dict[str, ModeSpec] = {
    "cr1": ModeSpec(
        "cr1", 1, BIG, ("state_end_A.pt", "state_end_B.pt"), None,
        (("v2_checkpoints_A.csv", ("v2_checkpoints_A.csv",)),
         ("v2_checkpoints_B.csv", ("v2_checkpoints_B.csv",))),
        full_gate_files=True, npz_pairs=tuple((n, n) for n in V1.NPZ)),
    "parentsA": ModeSpec(
        "parentsA", 1, 1600, ("state_end_A.pt",), "A",
        (("v2_checkpoints_A.csv", ("v2_checkpoints.csv", "v2_checkpoints_A.csv")),),
        cross_config=True, snap="before_B", npz_pairs=_GATE_A_PAIRS, end_eval="A"),
    "stage1_base": ModeSpec(
        "stage1_base", 1601, 2200, ("state_end_B.pt",), "B",
        (("v2_checkpoints_B.csv", ("v2_checkpoints.csv", "v2_checkpoints_B.csv")),),
        cross_config=True, snap="from_B",
        npz_pairs=(("final_final.npz", "final_final.npz"),
                   ("final_development.npz", "final_development.npz"),
                   ("checkpoint_weights.npz", "checkpoint_weights.npz")),
        end_eval="B"),
    "stage2_base": ModeSpec(
        "stage2_base", 1201, 1600, ("state_end_A.pt",), None,
        (("v2_checkpoints_A.csv", ("v2_checkpoints.csv", "v2_checkpoints_A.csv")),),
        continuation=True, cross_config=True, snap="every_in_window", npz_pairs=_GATE_A_PAIRS,
        end_eval="A"),
}


# ----------------------------------------------------------------------- first-difference finder
@dataclass
class Diff:
    """One difference: where, both values, and a note (count / max abs difference for arrays)."""

    path: str
    ref: str
    new: str
    note: str = ""

    def as_dict(self) -> Dict[str, str]:
        """JSON form."""
        d = {"path": self.path, "ref": self.ref, "new": self.new}
        if self.note:
            d["note"] = self.note
        return d


def short(v: Any, n: int = 160) -> str:
    """Compact text of a value (full float precision)."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, (np.ndarray, torch.Tensor)):
        s = f"{type(v).__name__}{tuple(v.shape)} {v.dtype}"
    else:
        s = repr(v)
    return s if len(s) <= n else s[:n - 3] + "..."


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _array_diff(a: np.ndarray, b: np.ndarray, path: str) -> Optional[Diff]:
    if a.shape != b.shape:
        return Diff(path + ".shape", str(a.shape), str(b.shape))
    if a.dtype != b.dtype:
        return Diff(path + ".dtype", str(a.dtype), str(b.dtype))
    if a.dtype.kind in "fc":
        neq = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    else:
        neq = a != b
    if not np.any(neq):
        return None
    idx = tuple(int(i) for i in np.argwhere(neq)[0])
    note = f"{int(np.sum(neq))} of {a.size} elements differ"
    if a.dtype.kind in "fiu":
        gap = np.abs(a[neq].astype(np.float64) - b[neq].astype(np.float64))
        note += f"; max|diff|={float(np.max(gap)):.6g}"
    va, vb = (a[idx], b[idx]) if idx else (a[()], b[()])
    return Diff(f"{path}{list(idx) if idx else ''}", short(va), short(vb), note)


def first_diff(a: Any, b: Any, path: str = "") -> Optional[Diff]:
    """First difference between two nested objects (tensors, arrays, dicts, lists, scalars).

    Strict: dtype, shape and every element must be equal (NaN equals NaN); numbers compare by
    value. Returns None when identical.
    """
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return Diff(path, short(a), short(b), "type")
        if a.dtype != b.dtype:
            return Diff(path + ".dtype", str(a.dtype), str(b.dtype))
        return _array_diff(a.detach().cpu().numpy(), b.detach().cpu().numpy(), path)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return Diff(path, short(a), short(b), "type")
        return _array_diff(a, b, path)
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return Diff(path, short(a), short(b), "type")
        if set(a) != set(b):
            only_a = sorted(set(a) - set(b), key=str)
            only_b = sorted(set(b) - set(a), key=str)
            return Diff(path + "{keys}", f"only in ref ({len(only_a)}): {short(only_a, 100)}",
                        f"only in new ({len(only_b)}): {short(only_b, 100)}")
        for k in sorted(a, key=str):
            d = first_diff(a[k], b[k], f"{path}.{k}" if path else str(k))
            if d is not None:
                return d
        return None
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return Diff(path, short(a), short(b), "type")
        if len(a) != len(b):
            return Diff(path + ".len", str(len(a)), str(len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d is not None:
                return d
        return None
    if isinstance(a, np.generic):
        a = a.item()
    if isinstance(b, np.generic):
        b = b.item()
    if _is_num(a) and _is_num(b):
        if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
            return None
        return None if a == b else Diff(path, short(a), short(b))
    return None if (type(a) is type(b) and a == b) else Diff(path, short(a), short(b))


# ----------------------------------------------------------------------------------- loaders
def load_state(path: Path) -> Dict[str, Any]:
    """Full-state checkpoint (``torch.load`` with ``weights_only=False``, as in v1.1)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """All arrays of an NPZ file."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_json(path: Path) -> Any:
    """JSON file."""
    with open(path) as f:
        return json.load(f)


def read_csv_text(path: Path) -> pd.DataFrame:
    """CSV as text (exact comparison, no float parsing)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def norm_conc(scales: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """None when every network is unscaled (the locked behaviour), else the scale dict."""
    if not scales or all(v is None or float(v) == 1.0 for v in scales.values()):
        return None
    return scales


def _first_existing(d: Path, names: Sequence[str]) -> Path:
    for n in names:
        if (d / n).exists():
            return d / n
    return d / names[0]


def _export_updates(d: Path, lo: int, hi: int) -> Dict[int, Path]:
    out = {}
    for p in sorted((d / "weights").glob("u*.npz")):
        m = re.fullmatch(r"u(\d+)\.npz", p.name)
        if m and lo <= int(m.group(1)) <= hi:
            out[int(m.group(1))] = p
    return out


# ------------------------------------------------------------------------ result accumulator
class Result:
    """Booleans per field plus the differences and informational items of one run."""

    def __init__(self) -> None:
        self.fields: Dict[str, bool] = {}
        self.diffs: Dict[str, Diff] = {}
        self.info: Dict[str, Any] = {}
        self._in_all: Dict[str, bool] = {}

    def check(self, name: str, fn: Callable[[], Optional[Diff]], in_all: bool = True) -> None:
        """Run one comparison; an exception (missing file, ...) counts as a difference."""
        try:
            d = fn()
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            d = Diff("<error>", f"{type(exc).__name__}: {exc}", "")
        self.fields[name] = d is None
        self._in_all[name] = in_all
        if d is not None:
            self.diffs[name] = d

    @property
    def all_identical(self) -> bool:
        """True iff every in-ALL field is identical (and at least one was compared)."""
        vals = [v for k, v in self.fields.items() if self._in_all[k]]
        return bool(vals) and all(vals)

    def first(self) -> Optional[Dict[str, Any]]:
        """First differing in-ALL field in comparison order."""
        for k, v in self.fields.items():
            if self._in_all[k] and not v:
                return {"field": k, **self.diffs[k].as_dict()}
        return None

    def as_dict(self) -> Dict[str, Any]:
        """JSON form of one run."""
        return {"fields": dict(self.fields), "ALL": self.all_identical,
                "first_difference": self.first(),
                "differences": {k: d.as_dict() for k, d in self.diffs.items()}, "info": self.info}


# -------------------------------------------------------------------- windowed log helpers
def _window(entries: List[Dict], lo: int, hi: int, phase: Optional[str] = None) -> Dict[int, Dict]:
    """Entries with ``lo <= update <= hi`` (and the phase), keyed by their global update."""
    return {int(e["update"]): e for e in entries
            if lo <= int(e["update"]) <= hi and (phase is None or e.get("phase") == phase)}


def _project(e: Dict, drop: Set[str] = frozenset(), keep: Optional[Sequence[str]] = None) -> Dict:
    return {k: v for k, v in e.items() if k not in drop and (keep is None or k in keep)}


def _keyed_diff(ref: List[Dict], new: List[Dict], spec: ModeSpec, drop: Set[str],
                keep: Optional[Sequence[str]] = None, common_only: bool = False,
                info: Optional[Dict[str, Any]] = None, tag: str = "") -> Optional[Diff]:
    """Compare two update-keyed logs over the mode window (wall-clock keys dropped)."""
    ra = _window(V1.strip(ref), spec.lo, spec.hi, spec.phase)
    rb = _window(V1.strip(new), spec.lo, spec.hi, spec.phase)
    if common_only:
        keys = sorted(set(ra) & set(rb))
        ra, rb = {k: ra[k] for k in keys}, {k: rb[k] for k in keys}
    if info is not None:
        info[f"n_compared_{tag}"] = len(ra)
    return first_diff({k: _project(v, drop, keep) for k, v in ra.items()},
                      {k: _project(v, drop, keep) for k, v in rb.items()}, tag)


def _select_snapshots(entries: List[Dict], spec: ModeSpec) -> List[Dict]:
    if spec.snap == "all":
        return entries
    reasons = [e.get("reason") for e in entries]
    if spec.snap == "before_B":
        return entries[:reasons.index("phase_B_entry")] if "phase_B_entry" in reasons else entries
    if spec.snap == "from_B":
        return entries[reasons.index("phase_B_entry"):] if "phase_B_entry" in reasons else []
    return [e for e in entries
            if spec.lo <= int(e["update"]) <= spec.hi and str(e["reason"]).startswith("every_")]


def _csv_diff(ref_path: Path, new_path: Path, spec: ModeSpec, drop: Set[str], common_only: bool,
              info: Dict[str, Any], tag: str) -> Optional[Diff]:
    """Compare two CSV files as text on the common columns over the mode window, by update."""
    ra, rb = read_csv_text(ref_path), read_csv_text(new_path)
    only_ref = [c for c in ra.columns if c not in rb.columns]
    only_new = [c for c in rb.columns if c not in ra.columns]
    known = [c for c in only_new if c.startswith(NEW_COLUMN_PREFIXES) or c in NEW_COLUMNS]
    info.setdefault("csv_columns_only_in_one", {})[tag] = {
        "only_in_ref": only_ref, "only_in_new": only_new,
        "unexpected_new": [c for c in only_new if c not in known] + only_ref}
    missing = [c for c in only_ref if c not in WALL_KEYS and c not in drop]
    if missing:   # a reference column absent from the new run is a difference (new-only columns are not)
        return Diff(f"{tag}.columns_missing_in_new", short(missing, 100), "(absent)",
                    f"{len(missing)} reference columns missing")
    cols = [c for c in ra.columns if c in rb.columns and c not in WALL_KEYS and c not in drop]
    out = []
    for df in (ra, rb):
        u = df["update"].astype(int)
        out.append(df[(u >= spec.lo) & (u <= spec.hi)].reset_index(drop=True))
    ra, rb = out
    ua, ub = ra["update"].astype(int).tolist(), rb["update"].astype(int).tolist()
    if common_only:
        keys = sorted(set(ua) & set(ub))
        ra = ra[ra["update"].astype(int).isin(keys)].reset_index(drop=True)
        rb = rb[rb["update"].astype(int).isin(keys)].reset_index(drop=True)
        ua, ub = keys, keys
    info[f"n_rows_compared_{tag}"] = len(ua)
    if ua != ub:
        return Diff(f"{tag}.updates", short(ua, 100), short(ub, 100),
                    f"{len(ua)} vs {len(ub)} rows")
    A, B = ra[cols].to_numpy(), rb[cols].to_numpy()
    neq = A != B
    if not neq.any():
        return None
    i, j = (int(x) for x in np.argwhere(neq)[0])
    return Diff(f"{tag}[update={ua[i]}].{cols[j]}", repr(A[i, j]), repr(B[i, j]),
                f"{int(neq.sum())} cells differ")


# --------------------------------------------------------------------------- the comparison
def _state_checks(res: Result, st: str, ref_dir: Path, new_dir: Path) -> None:
    cache: Dict[str, Any] = {}

    def both() -> Tuple[Dict, Dict]:
        if not cache:
            cache["r"], cache["n"] = load_state(ref_dir / st), load_state(new_dir / st)
        return cache["r"], cache["n"]

    for k in AGENT_KEYS:
        res.check(f"{st}:{k}", lambda k=k: first_diff(both()[0]["agent"].get(k),
                                                      both()[1]["agent"].get(k), k))
    res.check(f"{st}:conc_scale", lambda: first_diff(
        norm_conc(both()[0]["agent"].get("conc_scale")),
        norm_conc(both()[1]["agent"].get("conc_scale")), "conc_scale"))
    res.check(f"{st}:rng_streams", lambda: first_diff(both()[0]["rng"], both()[1]["rng"], "rng"))
    res.check(f"{st}:torch_generator", lambda: first_diff(
        both()[0]["torch_generator_state"], both()[1]["torch_generator_state"],
        "torch_generator_state"))
    try:   # informational: a continued phase has one extra phase-entry refresh
        r, n = both()
        ra, na = int(r["agent"]["snapshot_refreshes"]), int(n["agent"]["snapshot_refreshes"])
        res.info[f"{st}:snapshot_refreshes"] = {"ref": ra, "new": na, "equal": ra == na}
        keys = ("global_u", "total_episodes", "total_transitions", "next_snapshot_refresh_update")
        res.info[f"{st}:counters_equal"] = first_diff({k: r["counters"][k] for k in keys},
                                                      {k: n["counters"][k] for k in keys}) is None
    except Exception as exc:  # noqa: BLE001
        res.info[f"{st}:counters_error"] = f"{type(exc).__name__}: {exc}"


def _weights_diff(ref_dir: Path, new_dir: Path, spec: ModeSpec,
                  info: Dict[str, Any]) -> Optional[Diff]:
    ea = _export_updates(ref_dir, spec.lo, spec.hi)
    eb = _export_updates(new_dir, spec.lo, spec.hi)
    info["n_weight_exports_compared"] = len(ea)
    if set(ea) != set(eb):
        only_a, only_b = sorted(set(ea) - set(eb)), sorted(set(eb) - set(ea))
        return Diff("weights/{names}", f"only in ref ({len(only_a)}): {short(only_a, 100)}",
                    f"only in new ({len(only_b)}): {short(only_b, 100)}")
    for u in sorted(ea):
        d = first_diff(load_npz(ea[u]), load_npz(eb[u]), ea[u].name)
        if d is not None:
            return d
    return None


def _scalars_diff(ref: Dict[str, Any], new: Dict[str, Any], tag: str,
                  info: Dict[str, Any]) -> Optional[Diff]:
    """Reference's reported scalars against the new run's (keys the new run lacks are listed)."""
    common = [k for k in ref if k in new]
    info[f"{tag}_keys_not_in_new"] = [k for k in ref if k not in new]
    info[f"{tag}_n_compared"] = len(common)
    if not common:
        return Diff(tag, "no common keys", "no common keys")
    return first_diff({k: ref[k] for k in common}, {k: new[k] for k in common}, tag)


def _end_eval_checks(res: Result, spec: ModeSpec, ref_dir: Path, new_dir: Path) -> None:
    blk = f"end_of_{spec.end_eval}"
    cache: Dict[str, Any] = {}

    def fv2() -> Dict[str, Any]:
        if "fv2" not in cache:
            cache["fv2"] = load_json(new_dir / "final_v2.json")
        return cache["fv2"]

    def gates() -> Dict[str, Any]:
        if "gates" not in cache:
            cache["gates"] = load_json(ref_dir / "gates.json")
        return cache["gates"]

    for tier in ("final", "development"):
        res.check(f"{blk}.{tier}_scalars", lambda tier=tier: _scalars_diff(
            gates()["reported"][blk][tier], fv2()[tier], f"{blk}.{tier}", res.info))
    if spec.end_eval != "B":
        return

    def metric_values() -> Optional[Diff]:
        ref = gates()["metric_values"]
        new = {k: fv2()[t][s] for k, (t, s) in METRIC_MAP.items()}
        return first_diff({k: ref[k] for k in METRIC_MAP}, new, "metric_values")

    def dev_values() -> Optional[Diff]:
        ref = gates()["dev_tier_values"]
        return first_diff({f"{b}.{k}": ref[b][k] for (b, k) in DEV_MAP},
                          {f"{b}.{k}": fv2()["development"][s] for (b, k), s in DEV_MAP.items()},
                          "dev_tier_values")

    res.check("metric_values", metric_values)
    res.check("dev_tier_values", dev_values)


def compare_run(mode: str, ref_dir: Path, new_dir: Path) -> Dict[str, Any]:
    """Compare one run directory with its reference; returns the per-run dict."""
    spec = MODES[mode]
    res = Result()
    ref_dir, new_dir = Path(ref_dir), Path(new_dir)
    if not new_dir.is_dir():
        res.check("run_dir_present", lambda: Diff(str(new_dir), "present", "missing"))
        return res.as_dict()
    for st in spec.states:
        _state_checks(res, st, ref_dir, new_dir)
    res.check("weight_exports", lambda: _weights_diff(ref_dir, new_dir, spec, res.info))
    for ref_name, new_name in spec.npz_pairs:
        name = ref_name if ref_name == new_name else f"{ref_name}=={new_name}"
        res.check(name, lambda r=ref_name, n=new_name: first_diff(
            load_npz(ref_dir / r), load_npz(new_dir / n), n))
    # ---- train_history.json
    th: Dict[str, Any] = {}

    def hist(side: str) -> Dict[str, Any]:
        if side not in th:
            th[side] = load_json((ref_dir if side == "ref" else new_dir) / "train_history.json")
        return th[side]

    cont = spec.continuation
    local = {"local"} if cont else set()
    res.check("train_history:history", lambda: _keyed_diff(
        hist("ref")["history"], hist("new")["history"], spec, local, info=res.info, tag="history"))
    res.check("train_history:stability", lambda: _keyed_diff(
        hist("ref")["stability"], hist("new")["stability"], spec, set(),
        keep=STABILITY_STATE_KEYS if cont else None, info=res.info, tag="stability"))
    res.check("train_history:verifier_calls", lambda: _keyed_diff(
        hist("ref")["verifier_calls"], hist("new")["verifier_calls"], spec,
        CADENCE_VERIFIER if cont else set(), common_only=cont, info=res.info,
        tag="verifier_calls"))
    res.check("train_history:weight_checkpoints", lambda: _keyed_diff(
        hist("ref")["weight_checkpoints"], hist("new")["weight_checkpoints"], spec, local,
        info=res.info, tag="weight_checkpoints"))
    res.check("train_history:snapshots", lambda: first_diff(
        _select_snapshots(V1.strip(hist("ref")["snapshots"]), spec),
        _select_snapshots(V1.strip(hist("new")["snapshots"]), spec), "snapshots"))
    if cont:
        res.info["train_history:curriculum"] = (
            "not comparable: phase entry/exit counters restart in a continued phase")
    else:
        def sel(c: List[Dict]) -> List[Dict]:
            return [e for e in V1.strip(c) if spec.phase is None or e["phase"] == spec.phase]
        res.check("train_history:curriculum", lambda: first_diff(
            sel(hist("ref")["curriculum"]), sel(hist("new")["curriculum"]), "curriculum"))
    # ---- CSVs
    csv_drop = (IDENTITY_COLUMNS if spec.cross_config else set()) | (CADENCE_CSV if cont else set())
    res.check("v2_updates.csv", lambda: _csv_diff(
        ref_dir / "v2_updates.csv", new_dir / "v2_updates.csv", spec, local, False, res.info,
        "v2_updates.csv"))
    for ref_name, cands in spec.ckpt_csvs:
        res.check(ref_name, lambda r=ref_name, c=cands: _csv_diff(
            ref_dir / r, _first_existing(new_dir, c), spec, csv_drop, cont, res.info, r))
    # ---- evaluation / gate values
    if spec.full_gate_files:
        for f in ("induced_band.json", "drift_test.json"):
            res.check(f, lambda f=f: first_diff(load_json(ref_dir / f), load_json(new_dir / f), f))
        gr: Dict[str, Any] = {}

        def gboth() -> Tuple[Dict, Dict]:
            if not gr:
                gr["r"] = load_json(ref_dir / "gates.json")
                gr["n"] = load_json(new_dir / "gates.json")
            return gr["r"], gr["n"]

        for k in ("metric_values", "dev_tier_values", "reported", "G-A", "G-F", "G-N", "outcome"):
            res.check(f"gates:{k}", lambda k=k: first_diff(gboth()[0].get(k), gboth()[1].get(k),
                                                           f"gates.{k}"))
    if spec.end_eval:
        _end_eval_checks(res, spec, ref_dir, new_dir)
    return res.as_dict()


def run_dir(root: Path, q: int, seed: int, arm: Optional[str]) -> Path:
    """``<root>/q<q>/seed<seed>[/<arm>]``."""
    d = Path(root) / f"q{q}" / f"seed{seed}"
    return d / arm if arm else d


def compare_roots(mode: str, ref: Path, new: Path, qs: Sequence[int], seeds: Sequence[int],
                  arm: Optional[str] = None) -> Dict[str, Any]:
    """Compare every (q, seed) of ``new`` with ``ref``; the summary dict written by the CLI."""
    runs = []
    for q in qs:
        for s in seeds:
            r = compare_run(mode, run_dir(ref, q, s, None), run_dir(new, q, s, arm))
            runs.append({"q": int(q), "seed": int(s), **r})
    n_ok = sum(1 for r in runs if r["ALL"])
    failing = sorted({k for r in runs for k, v in r["fields"].items() if not v})
    firsts = [{"q": r["q"], "seed": r["seed"], **r["first_difference"]}
              for r in runs if r["first_difference"]]
    return {"tool": "tools/v2/cr1_compare.py", "mode": mode, "ref": str(ref), "new": str(new),
            "arm": arm, "window": [MODES[mode].lo, MODES[mode].hi], "n": len(runs),
            "n_identical": n_ok, "ALL": bool(runs) and n_ok == len(runs),
            "failing_fields": failing, "first_differences": firsts, "runs": runs}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI; exit code 0 iff every run is identical."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=sorted(MODES), default="cr1")
    p.add_argument("--ref", default=str(REF_DEFAULT),
                   help="dir containing q*/seed* (default: canonical rehearsal_v1_1)")
    p.add_argument("--new", required=True, help="dir containing q*/seed*[/<arm>]")
    p.add_argument("--arm", default=None,
                   help="arm subdirectory of --new (stage waves default: B_base / A_base)")
    p.add_argument("--out", default=None, help="write the JSON summary here")
    p.add_argument("--qs", type=int, nargs="+", default=list(V1.QS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(V1.SEEDS))
    a = p.parse_args(argv)
    arm = a.arm or {"stage1_base": "B_base", "stage2_base": "A_base"}.get(a.mode)
    summary = compare_roots(a.mode, Path(a.ref), Path(a.new), a.qs, a.seeds, arm)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)
    print(f"mode={a.mode} identical {summary['n_identical']}/{summary['n']} ALL={summary['ALL']}")
    for fd in summary["first_differences"]:
        print(f"  q={fd['q']} seed={fd['seed']} first differing field {fd['field']}: "
              f"path={fd['path']} ref={fd['ref']} new={fd['new']} {fd.get('note', '')}")
    return 0 if summary["ALL"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
