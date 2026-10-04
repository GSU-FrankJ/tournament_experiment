#!/usr/bin/env python3
"""Wave launcher of the T=2 accuracy-refinement round R1 (PROMPT.md sections 2.4 and 4.1-4.4).

Waves (``--wave``):
  v11_repro  C-R1: the UNCHANGED ``run/run_v2_T2_locked.py --q Q --seed S`` into
             ``<root>/v11_reproduction/q{Q}/seed{S}`` (no config is written by the launcher).
  parents_A  one ``phase_A`` run per (q, seed) with the locked-equivalent settings and
             ``full_state_at [1200]`` into ``<root>/parents_A/q{q}/seed{s}``.
  stage1     Phase-B arms (``phase_B``, parent = rehearsal_v1_1 ``state_end_A.pt``) into
             ``<root>/stage1/q{q}/seed{s}/<arm>``.
  stage2     Phase-A continuation arms (``phase_A_continue`` from
             ``parents_A/.../state_u01200.pt``) plus the method-5 pair (parent = rehearsal_v1_1
             ``state_end_A.pt``) into ``<root>/stage2/q{q}/seed{s}/<arm>``.
  r2b_v20_repro  C-R2 (round R2b): the UNCHANGED v2.0 ``run/run_v2_T2_locked.py --q Q --seed S`` into
             ``<root>/v20_reproduction/q{Q}/seed{S}`` (root defaults to ``results/v2_refine_r2b``);
             ``r2b_waveA`` / ``r2b_waveP`` are the R2b pilot waves (see ``R2B_*`` below).

Every config carries the record of the locked protocol (``protocols/v2_T2_locked_v1_1.json``)
and every new key explicitly (defaults written out). The arm tables below are module constants
so that the analysis can import them (``STAGE1_ARMS``, ``STAGE2_ARMS``, ``PARENTS_A_ARMS``,
``METHOD_ARMS``, ``EXPECTED_DIFFS``). ``--dry-run`` writes the configs and
``dryrun_<stamp>.json`` and validates every config; a real launch writes
``launch_<stamp>.json`` (planned and finished runs with return codes and wall time, nproc, load
average, free disk, the HEAD hash, ``git diff --stat <code-commit> HEAD`` and
``git status --porcelain``) and runs the jobs single-threaded through a bounded pool (at most
40 workers).

Examples (run inside tmux):
  python tools/v2/launch_refine.py --wave v11_repro --workers 20 --code-commit <sha>
  python tools/v2/launch_refine.py --wave stage1 --workers 40 --code-commit <sha> --dry-run
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, NamedTuple, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
CAN_ROOT = Path(
    "/home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2")
REHEARSAL = CAN_ROOT / "results" / "v2_T2_locked" / "rehearsal_v1_1"
PROTOCOL_PATH = ROOT / "protocols" / "v2_T2_locked_v1_1.json"
DEFAULT_ROOT = ROOT / "results" / "v2_refine"
DEFAULT_QS = (50, 60)
DEFAULT_SEEDS = tuple(range(10501, 10511))
MAX_WORKERS = 40
WAVES = ("v11_repro", "parents_A", "stage1", "stage2")
WAVE_DIR = {"v11_repro": "v11_reproduction", "parents_A": "parents_A", "stage1": "stage1",
            "stage2": "stage2", "r2b_waveA": "waveA", "r2b_waveP": "waveP",
            "r2b_v20_repro": "v20_reproduction"}
PLACEHOLDER_SHA = "<sha256 computed at launch>"

LOCKED_FLAGS = {"reward_mode": "expected", "stage2_update_mode": "frozen",
                "adv_norm_scope": "stage1_rows", "continuation_action_mode": "mean"}
PHASE_A_FLAGS = {"reward_mode": "expected", "stage2_update_mode": "joint",
                 "adv_norm_scope": "all_rows", "continuation_action_mode": "stochastic"}
LR0, LR1, LR2 = 3e-4, 3e-5, 3e-6      # locked start, locked end, polishing end
EPISODES_DEFAULT, MINIBATCH_DEFAULT = 512, 256
STAGE2_CAPS = {"A": 400, "B": 600, "C": 1000}
DETMEAN_CAPS = {"A": 1600, "B": 600, "C": 1000, "P": 200}
DETMEAN_VERIFIER_TIMEOUT = {"A": 100, "B": 25, "C": 25, "P": 50}
CTRL_CAPS = {"A": 200, "B": 600, "C": 1000}
PARENT_U1200 = "state_u01200.pt"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "kind": "continue",                 # stage 2 only: continue | ctrl200 | detmean
    "lr": None,                         # None = base window; else [(start, end, first, last)]
    "episodes_per_update": EPISODES_DEFAULT, "minibatch": MINIBATCH_DEFAULT, "target_kl": None,
    "continuation_value_mode": "sampled", "scale_last": None, "definition": "",
}

# ---- arm tables: arm -> the settings that differ from DEFAULT_SETTINGS (PROMPT.md 4.1-4.3, D5)
PARENTS_A_ARMS: Dict[str, Dict[str, Any]] = {
    "A_parent": {"definition": "Phase A of the locked pipeline (phase_A, cap 1600, locked window "
                               "1201-1600), full state saved at update 1200"},
}
STAGE1_ARMS: Dict[str, Dict[str, Any]] = {
    "B_base": {"definition": "locked Phase B (3e-4 -> 3e-5 over local 1-600), no change"},
    "B_polish1": {"lr": [(LR0, LR1, 1, 400), (LR1, LR2, 401, 600)],
                  "definition": "LR windows P1: local 1-400 3e-4 -> 3e-5, 401-600 3e-5 -> 3e-6"},
    "B_polish2": {"lr": [(LR0, LR2, 1, 600)],
                  "definition": "LR window P2: local 1-600 3e-4 -> 3e-6"},
    "B_batch": {"episodes_per_update": 2048, "minibatch": 1024,
                "definition": "episodes_per_update 2048, minibatch 1024 (40 optimizer steps "
                              "per update)"},
    "B_batch_mb256": {"episodes_per_update": 2048, "minibatch": 256,
                      "definition": "episodes_per_update 2048, minibatch 256 (secondary)"},
    "B_kl005": {"target_kl": 0.005, "definition": "target_kl 0.005 (stop the epochs of an update)"},
    "B_kl010": {"target_kl": 0.01, "definition": "target_kl 0.01 (stop the epochs of an update)"},
    "B_expcont": {"continuation_value_mode": "expected",
                  "definition": "stage-1 continuation replaced by the shock-integrated table (D4)"},
}
STAGE2_ARMS: Dict[str, Dict[str, Any]] = {
    "A_base": {"definition": "locked Phase A global 1201-1600 (local 1-400, 3e-4 -> 3e-5), "
                             "no change"},
    "A_polish1": {"lr": [(LR0, LR1, 1, 250), (LR1, LR2, 251, 400)],
                  "definition": "LR windows P1: local 1-250 3e-4 -> 3e-5, 251-400 3e-5 -> 3e-6"},
    "A_polish2": {"lr": [(LR0, LR2, 1, 400)],
                  "definition": "LR window P2: local 1-400 3e-4 -> 3e-6"},
    "A_batch": {"episodes_per_update": 2048, "minibatch": 1024,
                "definition": "episodes_per_update 2048, minibatch 1024 (20 optimizer steps "
                              "per update)"},
    "A_batch_mb256": {"episodes_per_update": 2048, "minibatch": 256,
                      "definition": "episodes_per_update 2048, minibatch 256 (secondary)"},
    "A_kl005": {"target_kl": 0.005, "definition": "target_kl 0.005 (stop the epochs of an update)"},
    "A_kl010": {"target_kl": 0.01, "definition": "target_kl 0.01 (stop the epochs of an update)"},
    "A_anneal2": {"scale_last": 2.0,
                  "definition": "concentration scale 1.0 -> 2.0 linear over local 1-400"},
    "A_anneal4": {"scale_last": 4.0,
                  "definition": "concentration scale 1.0 -> 4.0 linear over local 1-400"},
    "A_ctrl200": {"kind": "ctrl200", "lr": [(LR1, LR1, 1, 200)],
                  "definition": "method-5 control: 200 PPO updates at constant LR 3e-5 from the "
                                "baseline u1600 state (rehearsal_v1_1 state_end_A.pt)"},
    "A_detmean": {"kind": "detmean", "lr": [(LR1, LR1, 1, 200)],
                  "definition": "method 5: 200 pathwise deterministic-mean updates (phase_P), "
                                "LR constant 3e-5, from the baseline u1600 state"},
}
ARMS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "parents_A": PARENTS_A_ARMS, "stage1": STAGE1_ARMS, "stage2": STAGE2_ARMS}

# ---- round R2b (PROMPT.md D2, D3 and section 4): three one-change mechanisms on the development seeds.
# Wave A = full Phase A from scratch (the baseline is R1's parents_A); wave P = 200 updates from the
# rehearsal_v1_1 end-of-A state (comparators: R1's A_ctrl200 and A_detmean, plus the new control
# A_ctrl200_lr3e-4). Every R2b config writes the four new keys explicitly (defaults written out).
R2B_WAVES = ("r2b_waveA", "r2b_waveP")
R2B_REPRO = "r2b_v20_repro"      # C-R2: the UNCHANGED run/run_v2_T2_locked.py (v2.0) into <root>/v20_reproduction
R2B_ROOT = ROOT / "results" / "v2_refine_r2b"
R2B_DEFAULTS: Dict[str, Any] = {"start_weights": None, "clamp_likelihood": "density",
                                "pathwise_epochs": 1, "pathwise_minibatch": None}
PEAK_HALF_WIDTH = 20
PATHWISE_EPOCHS, PATHWISE_MINIBATCH = 10, 256      # 20 steps per update at 512 rows (= PPO's 20 actor steps)
R2B_WAVEA_ARMS: Dict[str, Dict[str, Any]] = {
    "A_peak25": {"r2b": {"start_weights": {"scheme": "peak_focused", "peak_half_width": PEAK_HALF_WIDTH,
                                           "peak_share": 0.25}},
                 "definition": "mechanism 1: peak-focused exploring starts, peak set = bins intersecting "
                               "(-20, 20), share 0.25 of the episodes (bin-balanced: 0.10 / 0.091)"},
    "A_peak50": {"r2b": {"start_weights": {"scheme": "peak_focused", "peak_half_width": PEAK_HALF_WIDTH,
                                           "peak_share": 0.50}},
                 "definition": "mechanism 1: peak-focused exploring starts, share 0.50"},
    "A_censored": {"r2b": {"clamp_likelihood": "censored"},
                   "definition": "mechanism 2: censored log-mass for clamped Beta draws (rollout and update)"},
}
R2B_WAVEP_ARMS: Dict[str, Dict[str, Any]] = {
    "P20_lr3e-5": {"kind": "pathwise", "lr": [(LR1, LR1, 1, 200)],
                   "r2b": {"pathwise_epochs": PATHWISE_EPOCHS, "pathwise_minibatch": PATHWISE_MINIBATCH},
                   "definition": "mechanism 3: phase P, 200 updates, E=10 x minibatch 256 (20 exact-gradient "
                                 "steps per update), LR constant 3e-5"},
    "P20_lr3e-4": {"kind": "pathwise", "lr": [(LR0, LR0, 1, 200)],
                   "r2b": {"pathwise_epochs": PATHWISE_EPOCHS, "pathwise_minibatch": PATHWISE_MINIBATCH},
                   "definition": "mechanism 3: as P20_lr3e-5 with LR constant 3e-4"},
    "A_ctrl200_lr3e-4": {"kind": "ctrl200", "lr": [(LR0, LR0, 1, 200)], "r2b": {},
                         "definition": "new control: 200 PPO updates at constant LR 3e-4 from the baseline "
                                       "u1600 state (locked Phase-A flags)"},
}
R2B_ARMS: Dict[str, Dict[str, Dict[str, Any]]] = {"r2b_waveA": R2B_WAVEA_ARMS, "r2b_waveP": R2B_WAVEP_ARMS}
V11_ARMS: Tuple[str, ...] = ("locked",)
METHOD_ARMS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "1 polish": {"stage1": ("B_polish1", "B_polish2"), "stage2": ("A_polish1", "A_polish2")},
    "2 batch": {"stage1": ("B_batch", "B_batch_mb256"), "stage2": ("A_batch", "A_batch_mb256")},
    "3 target_kl": {"stage1": ("B_kl005", "B_kl010"), "stage2": ("A_kl005", "A_kl010")},
    "4 annealing": {"stage1": (), "stage2": ("A_anneal2", "A_anneal4")},
    "5 deterministic mean": {"stage1": (), "stage2": ("A_ctrl200", "A_detmean")},
    "6 expected continuation": {"stage1": ("B_expcont",), "stage2": ()},
}
BASE_ARM = {"stage1": "B_base", "stage2": "A_base"}

# ---- pre-registered differences between an arm and its base arm (flattened config key paths)
IDENTITY_KEYS = frozenset({"arm", "arm_definition", "run", "record.run", "record.output_dir"})
_BATCH = frozenset({"budget_overrides.episodes_per_update", "ppo_overrides.minibatch"})
_BATCH256 = frozenset({"budget_overrides.episodes_per_update"})
EXPECTED_DIFFS: Dict[str, Dict[str, Tuple[str, FrozenSet[str]]]] = {
    "stage1": {
        "B_polish1": ("B_base", frozenset({"lr_decay"})),
        "B_polish2": ("B_base", frozenset({"lr_decay"})),
        "B_batch": ("B_base", _BATCH),
        "B_batch_mb256": ("B_base", _BATCH256),
        "B_kl005": ("B_base", frozenset({"ppo_overrides.target_kl"})),
        "B_kl010": ("B_base", frozenset({"ppo_overrides.target_kl"})),
        "B_expcont": ("B_base", frozenset({"continuation_value_mode"})),
    },
    "stage2": {
        "A_polish1": ("A_base", frozenset({"lr_decay"})),
        "A_polish2": ("A_base", frozenset({"lr_decay"})),
        "A_batch": ("A_base", _BATCH),
        "A_batch_mb256": ("A_base", _BATCH256),
        "A_kl005": ("A_base", frozenset({"ppo_overrides.target_kl"})),
        "A_kl010": ("A_base", frozenset({"ppo_overrides.target_kl"})),
        "A_anneal2": ("A_base", frozenset({"conc_anneal"})),
        "A_anneal4": ("A_base", frozenset({"conc_anneal"})),
        # method 5: a different parent and phase length; the matched control is A_ctrl200
        "A_ctrl200": ("A_base", frozenset({"parent_checkpoint", "parent_sha256", "lr_decay",
                                           "budget_overrides.phase_caps.A"})),
        "A_detmean": ("A_ctrl200", frozenset({"mode", "lr_decay", "budget_overrides.phase_caps.A",
                                              "budget_overrides.phase_caps.P",
                                              "budget_overrides.verifier_timeout"})),
    },
}


# ---- R2b: pre-registered differences between an arm and its comparator config. The comparator is the
# R1 builder's own config for the same (q, seed) (parents_A's A_parent, stage2's A_detmean / A_ctrl200)
# with the four R2b keys written at their defaults; ``pilot`` differs by wave and is an identity key here.
IDENTITY_KEYS_R2B = IDENTITY_KEYS | {"pilot"}
R2B_EXPECTED_DIFFS: Dict[str, Dict[str, Tuple[str, FrozenSet[str]]]] = {
    "r2b_waveA": {
        "A_peak25": ("A_parent", frozenset({"start_weights"})),
        "A_peak50": ("A_parent", frozenset({"start_weights"})),
        "A_censored": ("A_parent", frozenset({"clamp_likelihood"})),
    },
    "r2b_waveP": {
        "P20_lr3e-5": ("A_detmean", frozenset({"pathwise_epochs", "pathwise_minibatch"})),
        "P20_lr3e-4": ("A_detmean", frozenset({"pathwise_epochs", "pathwise_minibatch", "lr_decay"})),
        "A_ctrl200_lr3e-4": ("A_ctrl200", frozenset({"lr_decay"})),
    },
}


class Job(NamedTuple):
    """One planned run: its run config (``locked_entry_point`` stub for v11_repro) and out dir."""

    cfg: Dict[str, Any]
    out_dir: str


# --------------------------------------------------------------------------- helpers
def sha256_file(path: str) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=None)
def _protocol() -> Dict[str, Any]:
    with open(PROTOCOL_PATH) as f:
        return json.load(f)


def _window(w: Tuple[float, float, int, int], phase: str) -> Dict[str, Any]:
    start, end, first, last = w
    return {"phase": phase, "start_lr": start, "end_lr": end, "local_first": first,
            "local_last": last}


def _locked_window(phase: str) -> Dict[str, Any]:
    """The locked protocol's lr_decay window of a phase (deep copy)."""
    wins = [w for w in _protocol()["pipeline"]["lr_decay"] if w["phase"] == phase]
    assert len(wins) == 1, f"protocol has {len(wins)} lr_decay windows for phase {phase}"
    return copy.deepcopy(wins[0])


@functools.lru_cache(maxsize=None)
def _sha_existing(path: str) -> str:
    return sha256_file(path)


def _parent_sha(path: str, require: bool) -> str:
    if os.path.exists(path):
        return _sha_existing(path)
    if require:
        raise FileNotFoundError(f"parent checkpoint {path} does not exist")
    return PLACEHOLDER_SHA


def flatten_diff(a: Any, b: Any, prefix: str = "") -> List[str]:
    """Dotted key paths at which two configs differ (dicts recurse; lists, scalars are leaves)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for k in sorted(set(a) | set(b)):
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in a or k not in b:
                out.append(p)
            else:
                out.extend(flatten_diff(a[k], b[k], p))
        return out
    return [] if a == b else [prefix]


def arm_diff(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any]) -> FrozenSet[str]:
    """Differing config keys of two jobs, identity keys (arm, run, record.run, ...) removed."""
    return frozenset(flatten_diff(cfg_a, cfg_b)) - IDENTITY_KEYS


def r2b_comparator_config(wave: str, q: int, seed: int, root: Path, require_parents: bool = False) -> Dict[str, Any]:
    """Comparator config of an R2b wave (see ``R2B_EXPECTED_DIFFS``): the R1 builder's own output with
    the R2b defaults written out. All arms of a wave share one comparator arm per table row."""
    root = Path(root).resolve()
    out: Dict[str, Dict[str, Any]] = {}
    for arm, (ref, _) in R2B_EXPECTED_DIFFS[wave].items():
        if ref not in out:
            job = (build_parents_a(ref, q, seed, root) if wave == "r2b_waveA"
                   else build_stage2(ref, q, seed, root, require_parents))
            out[ref] = _with_r2b_keys(job.cfg, {})
    return out


def r2b_arm_diff(cfg: Dict[str, Any], ref_cfg: Dict[str, Any]) -> FrozenSet[str]:
    """Differing config keys of an R2b arm and its comparator, identity keys (incl. ``pilot``) removed."""
    return frozenset(flatten_diff(cfg, ref_cfg)) - IDENTITY_KEYS_R2B


def lr_window_problems(cfg: Dict[str, Any]) -> List[str]:
    """Consistency of the lr_decay windows with the phase caps (the DESIGN.md rules)."""
    probs: List[str] = []
    wins = cfg.get("lr_decay") or []
    caps = dict(cfg["record"]["protocol"]["phase_caps"])
    caps.update(cfg["budget_overrides"].get("phase_caps", {}))
    ab_lr = float(cfg["record"]["lr_schedule"]["ab_lr"])
    for ph in sorted({w["phase"] for w in wins}):
        ws = sorted((w for w in wins if w["phase"] == ph), key=lambda w: w["local_first"])
        for w0, w1 in zip(ws, ws[1:]):
            if w1["local_first"] != w0["local_last"] + 1:
                probs.append(f"phase {ph}: windows not contiguous at {w0['local_last']}")
        if ws[-1]["local_last"] != caps[ph]:
            probs.append(f"phase {ph}: last window ends at {ws[-1]['local_last']} "
                         f"!= cap {caps[ph]}")
        continued = cfg["mode"] in ("phase_A_continue", "phase_P") and ws[0]["local_first"] == 1
        if ws[0]["start_lr"] != ab_lr and not continued:
            probs.append(f"phase {ph}: first window starts at {ws[0]['start_lr']} != ab_lr")
    return probs


# --------------------------------------------------------------------------- config builders
def _base_config(wave: str, arm: str, q: int, seed: int, out: str, mode: str,
                 flags: Dict[str, str], settings: Dict[str, Any]) -> Dict[str, Any]:
    proto = _protocol()
    if q not in proto["q_values"]:
        raise ValueError(f"q={q} not in the protocol's q_values {proto['q_values']}")
    run = f"v2refine_{wave}_q{q}_s{seed}_{arm}"
    rec = copy.deepcopy(proto["records"][str(q)])
    rec.update(seed=int(seed), run=run, output_dir=out)
    return {
        "schema": "v2_run_config/1", "base_commit": proto["base_commit"],
        "pilot": f"v2_refine_{wave}", "arm": arm, "arm_definition": settings["definition"],
        "run": run, "q": q, "seed": int(seed), "mode": mode, "fixed_budget": True,
        "flags": dict(flags), "parent_checkpoint": None, "parent_sha256": None, "record": rec,
        "threads_per_process": copy.deepcopy(proto["threads_per_process"]),
        "budget_overrides": {},
        "ppo_overrides": {"minibatch": int(settings["minibatch"]),
                          "target_kl": settings["target_kl"]},
        "continuation_value_mode": settings["continuation_value_mode"],
        "conc_anneal": None,
        "lr_decay": [], "full_state_at": [],
    }


def _lr_windows(settings: Dict[str, Any], phase: str,
                default: Dict[str, Any]) -> List[Dict[str, Any]]:
    if settings["lr"] is None:
        return [default]
    return [_window(w, phase) for w in settings["lr"]]


def _with_r2b_keys(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Write the four R2b keys explicitly (defaults, then the arm's own values)."""
    cfg.update(copy.deepcopy(R2B_DEFAULTS))
    cfg.update(copy.deepcopy(overrides))
    return cfg


def build_parents_a(arm: str, q: int, seed: int, root: Path,
                    table: Optional[Dict[str, Dict[str, Any]]] = None, wave: str = "parents_A") -> Job:
    """Phase-A parent run (mode phase_A, locked window 1201-1600, full state at update 1200).

    The R2b wave-A arms are built by this function too (``table=R2B_WAVEA_ARMS``,
    ``wave="r2b_waveA"``): the same config plus the four R2b keys, the arm's own value set.
    """
    arms = PARENTS_A_ARMS if table is None else table
    st = {**DEFAULT_SETTINGS, **arms[arm]}
    out = str(root / WAVE_DIR[wave] / f"q{q}" / f"seed{seed}" / ("" if wave == "parents_A" else arm))
    out = out.rstrip("/")
    cfg = _base_config(wave, arm, q, seed, out, "phase_A", PHASE_A_FLAGS, st)
    cfg["budget_overrides"] = {"episodes_per_update": int(st["episodes_per_update"])}
    cfg["lr_decay"] = _lr_windows(st, "A", _locked_window("A"))
    cfg["full_state_at"] = [1200]
    if wave in R2B_WAVES:
        _with_r2b_keys(cfg, st.get("r2b", {}))
    return Job(cfg, out)


def build_stage1(arm: str, q: int, seed: int, root: Path, require_parents: bool) -> Job:
    """Phase-B arm from the rehearsal_v1_1 end-of-A state with the locked flags."""
    st = {**DEFAULT_SETTINGS, **STAGE1_ARMS[arm]}
    out = str(root / "stage1" / f"q{q}" / f"seed{seed}" / arm)
    cfg = _base_config("stage1", arm, q, seed, out, "phase_B", LOCKED_FLAGS, st)
    parent = str(REHEARSAL / f"q{q}" / f"seed{seed}" / "state_end_A.pt")
    cfg["parent_checkpoint"] = parent
    cfg["parent_sha256"] = _parent_sha(parent, require_parents)
    cfg["budget_overrides"] = {"episodes_per_update": int(st["episodes_per_update"])}
    cfg["lr_decay"] = _lr_windows(st, "B", _locked_window("B"))
    return Job(cfg, out)


def build_stage2(arm: str, q: int, seed: int, root: Path, require_parents: bool,
                 table: Optional[Dict[str, Dict[str, Any]]] = None, wave: str = "stage2") -> Job:
    """Phase-A continuation arm (or a method-5 arm) with the Phase-A flags.

    The ``continue`` arms start from ``parents_A/.../state_u01200.pt``, a state taken inside a
    phase_A run: ``phase_done == 'A'`` but ``phases_done == []`` (it is appended when the phase
    ends), on a snapshot refresh (1200 % 20 == 0). The runner has to accept such a parent in mode
    phase_A_continue; the end-to-end section of ``tests/test_v2_refine_tools.py`` runs this path.

    The R2b wave-P arms are built by this function too (``table=R2B_WAVEP_ARMS``,
    ``wave="r2b_waveP"``; kinds ``pathwise`` = ``detmean`` with the R2b optimiser-budget keys, and
    ``ctrl200``): the same config plus the four R2b keys.
    """
    arms = STAGE2_ARMS if table is None else table
    st = {**DEFAULT_SETTINGS, **arms[arm]}
    kind = st["kind"]
    out = str(root / WAVE_DIR[wave] / f"q{q}" / f"seed{seed}" / arm)
    pathwise = kind in ("detmean", "pathwise")
    mode = "phase_P" if pathwise else "phase_A_continue"
    cfg = _base_config(wave, arm, q, seed, out, mode, PHASE_A_FLAGS, st)
    if kind == "continue":
        parent = str(root / "parents_A" / f"q{q}" / f"seed{seed}" / PARENT_U1200)
    else:
        parent = str(REHEARSAL / f"q{q}" / f"seed{seed}" / "state_end_A.pt")
    cfg["parent_checkpoint"] = parent
    cfg["parent_sha256"] = _parent_sha(parent, require_parents)
    bo: Dict[str, Any] = {"phase_caps": dict({"continue": STAGE2_CAPS, "ctrl200": CTRL_CAPS,
                                              "detmean": DETMEAN_CAPS, "pathwise": DETMEAN_CAPS}[kind])}
    if pathwise:
        bo["verifier_timeout"] = dict(DETMEAN_VERIFIER_TIMEOUT)
    bo["episodes_per_update"] = int(st["episodes_per_update"])
    cfg["budget_overrides"] = bo
    phase = "P" if pathwise else "A"
    cfg["lr_decay"] = _lr_windows(st, phase, _window((LR0, LR1, 1, 400), "A"))
    cfg["conc_anneal"] = (None if st["scale_last"] is None else
                          {"phase": "A", "local_first": 1, "local_last": 400,
                           "scale_first": 1.0, "scale_last": float(st["scale_last"])})
    if wave in R2B_WAVES:
        _with_r2b_keys(cfg, st.get("r2b", {}))
    return Job(cfg, out)


def build_v11_repro(q: int, seed: int, root: Path) -> Job:
    """C-R1 stub: the locked entry point writes its own run_config.json."""
    out = str(root / "v11_reproduction" / f"q{q}" / f"seed{seed}")
    return Job({"locked_entry_point": True, "wave": "v11_repro", "q": q, "seed": int(seed),
                "run": f"v2T2locked_q{q}_s{seed}", "arm": "locked"}, out)


def build_v20_repro(q: int, seed: int, root: Path) -> Job:
    """C-R2 stub: the unchanged v2.0 locked entry point writes its own run_config.json."""
    out = str(root / WAVE_DIR[R2B_REPRO] / f"q{q}" / f"seed{seed}")
    return Job({"locked_entry_point": True, "wave": R2B_REPRO, "q": q, "seed": int(seed),
                "run": f"v2T2locked_q{q}_s{seed}", "arm": "locked"}, out)


def build_configs(wave: str, qs: Sequence[int], seeds: Sequence[int],
                  arms: Optional[Sequence[str]], root: Any,
                  require_parents: bool = True) -> List[Job]:
    """All (cfg, out_dir) of a wave, ordered q, seed, arm.

    Args:
        wave: one of ``WAVES``.
        qs: q values.
        seeds: seeds.
        arms: arm names of the wave (None = all arms of the wave in table order).
        root: results root (``<repo>/results/v2_refine`` by default in the CLI).
        require_parents: raise when a parent checkpoint is missing; when False a missing parent
            gets a placeholder SHA-256 (dry runs before the parents exist).
    """
    if wave not in WAVES + R2B_WAVES + (R2B_REPRO,):
        raise ValueError(f"wave {wave!r} not in {WAVES + R2B_WAVES + (R2B_REPRO,)}")
    root = Path(root).resolve()
    if wave == R2B_REPRO:
        if arms:
            raise ValueError(f"wave {wave} has no arms")
        return [build_v20_repro(q, s, root) for q in qs for s in seeds]
    if wave in R2B_WAVES:
        arm_table = R2B_ARMS[wave]
        arms = tuple(arm_table if arms is None else arms)
        bad = [a for a in arms if a not in arm_table]
        if bad:
            raise ValueError(f"arms {bad} not valid for wave {wave}: {list(arm_table)}")
        return [build_parents_a(a, q, s, root, arm_table, wave) if wave == "r2b_waveA"
                else build_stage2(a, q, s, root, require_parents, arm_table, wave)
                for q in qs for s in seeds for a in arms]
    table = V11_ARMS if wave == "v11_repro" else tuple(ARMS[wave])
    arms = tuple(table if arms is None else arms)
    bad = [a for a in arms if a not in table]
    if bad:
        raise ValueError(f"arms {bad} not valid for wave {wave}: {list(table)}")
    jobs: List[Job] = []
    for q in qs:
        for seed in seeds:
            if wave == "v11_repro":
                jobs.append(build_v11_repro(q, seed, root))
                continue
            for arm in arms:
                if wave == "parents_A":
                    jobs.append(build_parents_a(arm, q, seed, root))
                elif wave == "stage1":
                    jobs.append(build_stage1(arm, q, seed, root, require_parents))
                else:
                    jobs.append(build_stage2(arm, q, seed, root, require_parents))
    return jobs


def is_locked_entry(cfg: Dict[str, Any]) -> bool:
    """True for the v11_repro stub (run through run/run_v2_T2_locked.py)."""
    return bool(cfg.get("locked_entry_point"))


def job_command(cfg: Dict[str, Any], out_dir: str) -> List[str]:
    """Subprocess command of a job (never ``git``; the interpreter is the launcher's own)."""
    if is_locked_entry(cfg):
        return [PY, "-B", "-u", str(ROOT / "run" / "run_v2_T2_locked.py"), "--q", str(cfg["q"]),
                "--seed", str(cfg["seed"]), "--out-dir", out_dir]
    return [PY, "-B", "-u", str(ROOT / "run" / "run_v2_stagewise.py"), "--config",
            os.path.join(out_dir, "run_config.json"), "--out-dir", out_dir]


# --------------------------------------------------------------------------- validation (dry run)
NEW_TOP_KEYS = ("ppo_overrides", "continuation_value_mode", "conc_anneal",
                "start_weights", "clamp_likelihood", "pathwise_epochs", "pathwise_minibatch")


def legacy_projection(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The config with every key unknown to the pre-R1 ``validate_config`` removed.

    New top-level keys and ``budget_overrides.episodes_per_update`` are dropped, multi-window
    ``lr_decay`` is reduced to the first window of each phase and mode ``phase_P`` is mapped to
    ``phase_A_continue``. If this projection validates, a failure of the full config comes from
    the new keys only.
    """
    c = copy.deepcopy(cfg)
    for k in NEW_TOP_KEYS:
        c.pop(k, None)
    c["budget_overrides"].pop("episodes_per_update", None)
    if c["mode"] == "phase_P":
        c["mode"] = "phase_A_continue"
        c["budget_overrides"]["phase_caps"].pop("P", None)
        c["budget_overrides"].pop("verifier_timeout", None)
        for w in c["lr_decay"]:
            w["phase"] = "A"
    seen, keep = set(), []
    for w in c["lr_decay"]:
        if w["phase"] not in seen:
            seen.add(w["phase"])
            keep.append(w)
    c["lr_decay"] = keep
    return c


def validate_jobs(jobs: Sequence[Job]) -> List[Dict[str, Any]]:
    """``run.run_v2_stagewise.validate_config`` on every config (and on its legacy projection).

    Returns one row per job with ``ok``, ``error``, ``legacy_ok`` (only evaluated when the
    full config fails) and ``window_problems``.
    """
    sys.path.insert(0, str(ROOT))
    from run import run_v2_stagewise as rs

    rows = []
    for cfg, out in jobs:
        if is_locked_entry(cfg):
            continue
        row: Dict[str, Any] = {"out": out, "arm": cfg["arm"], "q": cfg["q"], "seed": cfg["seed"],
                               "ok": True, "error": None, "legacy_ok": None, "legacy_error": None,
                               "window_problems": lr_window_problems(cfg)}
        try:
            rs.validate_config(cfg)
        except Exception as exc:  # noqa: BLE001 - reported
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            try:
                rs.validate_config(legacy_projection(cfg))
                row["legacy_ok"] = True
            except Exception as exc2:  # noqa: BLE001
                row.update(legacy_ok=False, legacy_error=f"{type(exc2).__name__}: {exc2}")
        rows.append(row)
    return rows


def summarize_validation(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per arm: valid as written / fails only because of new keys / fails even without them."""
    arms: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        a = arms.setdefault(r["arm"], {"n": 0, "valid": 0, "fail_new_keys_only": 0,
                                       "fail_other": 0, "errors": set(), "window_problems": 0})
        a["n"] += 1
        a["valid"] += int(r["ok"])
        a["fail_new_keys_only"] += int((not r["ok"]) and bool(r["legacy_ok"]))
        a["fail_other"] += int((not r["ok"]) and r["legacy_ok"] is False)
        a["window_problems"] += int(bool(r["window_problems"]))
        if r["error"]:
            a["errors"].add(r["error"])
    return {k: {**v, "errors": sorted(v["errors"])} for k, v in arms.items()}


# --------------------------------------------------------------------------- launch machinery
def git_record(code_commit: Optional[str]) -> Dict[str, Any]:
    """HEAD hash, ``git diff --stat <code_commit> HEAD`` and ``git status --porcelain``."""
    def git(*args: str) -> str:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        if r.returncode != 0:
            return f"[git {' '.join(args)} failed] {r.stderr.strip()}"
        return r.stdout.strip()
    rec: Dict[str, Any] = {"head": git("rev-parse", "HEAD"), "code_commit": code_commit,
                           "status_porcelain": git("status", "--porcelain").splitlines()}
    if code_commit:
        rec["code_commit_resolved"] = git("rev-parse", "--verify", code_commit + "^{commit}")
        rec["diff_stat_code_commit_to_head"] = git("diff", "--stat", code_commit, "HEAD")
    return rec


def host_record(path: Path) -> Dict[str, Any]:
    """nproc, load average and free disk, recorded before a launch."""
    du = shutil.disk_usage(path)
    return {"nproc": os.cpu_count(), "loadavg_at_start": list(os.getloadavg()),
            "disk_free_bytes": du.free, "disk_total_bytes": du.total}


def child_env() -> Dict[str, str]:
    """Single-threaded environment of every run."""
    return dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                PYTHONDONTWRITEBYTECODE="1")


def run_job(cmd: Sequence[str], out_dir: str, env: Dict[str, str]) -> Dict[str, Any]:
    """Run one command, stdout/stderr to ``<out_dir>/run.log`` (opened ``x``: no overwrite)."""
    t0 = time.monotonic()
    with open(os.path.join(out_dir, "run.log"), "x") as log:
        r = subprocess.run(list(cmd), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"out": out_dir, "returncode": r.returncode, "wall_sec": time.monotonic() - t0}


def run_pool(jobs: Sequence[Job], workers: int, record: Dict[str, Any], record_path: Path) -> None:
    """Run all jobs through a bounded pool of subprocesses; the record is updated per finish."""
    env = child_env()

    def one(job: Job) -> Dict[str, Any]:
        try:
            return run_job(job_command(*job), job.out_dir, env)
        except Exception as exc:  # noqa: BLE001 - recorded as a failed run
            return {"out": job.out_dir, "returncode": None, "error": f"{type(exc).__name__}: {exc}"}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, j) for j in jobs]):
            res = fut.result()
            record["runs"].append(res)
            write_record(record_path, record)
            print(json.dumps(res), flush=True)
    record["state"] = "done" if all(r.get("returncode") == 0 for r in record["runs"]) else "failed"
    record["loadavg_at_end"] = list(os.getloadavg())
    write_record(record_path, record)


def write_record(path: Path, record: Dict[str, Any], create: bool = False) -> None:
    """Write the launch record; ``create=True`` refuses to replace an existing file."""
    with open(path, "x" if create else "w") as f:
        json.dump(record, f, indent=1)


def check_no_previous_run(jobs: Sequence[Job]) -> List[str]:
    """Out dirs that already hold a run (status.json) or a log (run.log)."""
    return [j.out_dir for j in jobs if os.path.exists(os.path.join(j.out_dir, "status.json"))
            or os.path.exists(os.path.join(j.out_dir, "run.log"))]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry (see the module docstring)."""
    p = argparse.ArgumentParser(description="R1 wave launcher", allow_abbrev=False)
    p.add_argument("--wave", choices=WAVES + R2B_WAVES + (R2B_REPRO,), required=True)
    p.add_argument("--qs", type=int, nargs="+", default=list(DEFAULT_QS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--arms", nargs="+", default=None, help="default: all arms of the wave")
    p.add_argument("--workers", type=int, default=20, help=f"at most {MAX_WORKERS}")
    p.add_argument("--root", default=None,
                   help="results root (default: results/v2_refine for the R1 waves, "
                        "results/v2_refine_r2b for the R2b waves)")
    p.add_argument("--code-commit", default=None, help="SHA of the code commit (recorded)")
    p.add_argument("--dry-run", action="store_true", help="write configs + dryrun record, validate")
    p.add_argument("--no-validate", action="store_true", help="skip validate_config (tests)")
    a = p.parse_args(argv)
    if not 1 <= a.workers <= MAX_WORKERS:
        p.error(f"--workers must be in 1..{MAX_WORKERS}; got {a.workers}")
    if a.wave in ("v11_repro", R2B_REPRO) and a.arms:
        p.error(f"wave {a.wave} has no arms")
    if any(sd not in DEFAULT_SEEDS for sd in a.seeds):   # D1/section 7: nothing outside 10501-10510
        p.error(f"seeds must be development seeds {DEFAULT_SEEDS[0]}-{DEFAULT_SEEDS[-1]}; got {a.seeds}")
    root = Path(a.root or (R2B_ROOT if a.wave in R2B_WAVES + (R2B_REPRO,) else DEFAULT_ROOT)).resolve()
    try:
        jobs = build_configs(a.wave, a.qs, a.seeds, a.arms, root, require_parents=not a.dry_run)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    busy = check_no_previous_run(jobs)
    if busy:
        p.error(f"{len(busy)} out dir(s) already hold a run (status.json or run.log), "
                f"e.g. {busy[0]}")
    validation: Dict[str, Any] = {}
    if not a.no_validate and a.wave not in ("v11_repro", R2B_REPRO):
        rows = validate_jobs(jobs)
        validation = {"per_arm": summarize_validation(rows),
                      "n_invalid": sum(1 for r in rows if not r["ok"])}
        if not a.dry_run and validation["n_invalid"]:
            print(json.dumps(validation, indent=1))
            p.error("validate_config fails for the configs above; nothing was launched")
    wave_dir = root / WAVE_DIR[a.wave]
    wave_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    record_path = wave_dir / (f"dryrun_{stamp}.json" if a.dry_run else f"launch_{stamp}.json")
    record: Dict[str, Any] = {
        "wave": a.wave, "dry_run": bool(a.dry_run),
        "argv": list(sys.argv if argv is None else argv), "started": stamp,
        "workers": a.workers, **host_record(wave_dir), **git_record(a.code_commit),
        "n_planned": len(jobs), "planned": [j.out_dir for j in jobs], "validation": validation,
        "runs": [], "state": "dry_run" if a.dry_run else "running"}
    for cfg, out in jobs:
        os.makedirs(out, exist_ok=True)
        if not is_locked_entry(cfg):
            with open(os.path.join(out, "run_config.json"), "w") as f:
                json.dump(cfg, f, indent=1)
    write_record(record_path, record, create=True)
    print(f"{len(jobs)} run(s) of wave {a.wave}; record {record_path}", flush=True)
    if validation:
        print(json.dumps(validation["per_arm"], indent=1), flush=True)
    if a.dry_run:
        return 0
    run_pool(jobs, a.workers, record, record_path)
    return 0 if record["state"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
