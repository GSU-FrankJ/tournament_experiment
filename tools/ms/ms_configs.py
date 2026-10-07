"""Run configs (``ms_run_config/1``) of the MS-R1 arms: the arm table (D6), the rule parameters and the
builder shared by the launcher, the tests and the analysis.

The v2.0 record is embedded from ``protocols/v2_T2_locked_v2_0.json`` (``records[q]``). Every config
carries every key explicitly (defaults written out). The rule / sampler parameters come from a
parameters dict (``DEFAULT_PARAMS`` = the defaults of D4/D5; the pre-registered set is passed in with
``--params`` by the launcher), so that the code does not change when a calibrated value does.

MS-R2 adds the six ``NL_*`` arms (``R2_ARM_TABLE``); MS-R3 adds the twelve ``{t1,relu,t10}_{bb,st}_s{1,16}`` arms
(``R3_ARM_TABLE``): the settings of an ``NL_*`` arm plus ``init_digest`` and, for the ``relu`` / ``t10`` actors,
``actor_variant``. The three tables are kept apart.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from run.run_ms_stagewise import REPORT_NEAR_TIE_HALF_WIDTH, SCHEMA, derive_start_info  # noqa: E402

PROTOCOL_REL = "protocols/v2_T2_locked_v2_0.json"
PROTOCOL_PATH = ROOT / PROTOCOL_REL
DEFAULT_QS = (50, 60)
DEFAULT_SEEDS = tuple(range(10501, 10511))
PILOT = "ms_r1"
PILOT_R2 = "ms_r2"
PILOT_R3 = "ms_r3"
#: prefix of the run name (``<prefix>_q<q>_s<seed>_<arm>``) by round tag
RUN_PREFIX: Dict[str, str] = {PILOT: "msr1", PILOT_R2: "msr2", PILOT_R3: "msr3"}

#: arm -> start-weights settings (D6). lambda_P None = the bin-balanced near-tie share n_near / n_bins.
ARMS: Dict[str, Dict[str, Any]] = {
    "MS_base": {"scheme": "bin_balanced", "rule_enabled": False,
                "definition": "regression arm: bin_balanced, rule disabled, fixed 1600 / 600 with v2.0's LR "
                              "windows; checks every 25 reported, would-fire recorded (C-MS1)"},
    "MS_base2400": {"scheme": "bin_balanced", "rule_enabled": False, "legacy_pipeline": "LEGACY_PIPELINE_2400",
                    "definition": "budget-matched control (addendum A1): bin_balanced, rule disabled, terminal "
                                  "stage fixed 2400 (LR 3e-4 to local 2000, linear 3e-4 -> 3e-5 over 2001-2400), "
                                  "stage 1 as MS_base; checks every 25 reported, would-fire recorded"},
    "MS_rule": {"scheme": "stratified_priority", "lambda_P": None, "alpha_global": 0.0, "rule_enabled": True,
                "definition": "stop rule and polishing alone: lambda_P = bin-balanced near-tie share, alpha_global 0"},
    "MS_s25a0": {"scheme": "stratified_priority", "lambda_P": 0.25, "alpha_global": 0.0, "rule_enabled": True,
                 "definition": "lambda_P 0.25, alpha_global 0"},
    "MS_s25a5": {"scheme": "stratified_priority", "lambda_P": 0.25, "alpha_global": 0.5, "rule_enabled": True,
                 "definition": "lambda_P 0.25, alpha_global 0.5"},
    "MS_s35a0": {"scheme": "stratified_priority", "lambda_P": 0.35, "alpha_global": 0.0, "rule_enabled": True,
                 "definition": "lambda_P 0.35, alpha_global 0"},
    "MS_s35a5": {"scheme": "stratified_priority", "lambda_P": 0.35, "alpha_global": 0.5, "rule_enabled": True,
                 "definition": "lambda_P 0.35, alpha_global 0.5"},
}
#: MS-R2 (prompt D3): terminal stage fixed 2800 with a noise landing, crossed with the start sampler. ``r2.scale`` is the
#: end value s of the concentration scale (1 = no scale is ever set, the key is absent); ``r2.stratified`` selects the
#: sampler of an MS-R1 global block (lambda_P 0.35, alpha_global 0.5) under the fixed budget.
R2_SCALES = (1, 4, 16)
R2_ARMS = tuple(f"NL_{k}_s{s}" for k in ("bb", "st") for s in R2_SCALES)
R2_ARM_TABLE: Dict[str, Dict[str, Any]] = {}     # kept apart from ARMS: the MS-R1 arm table is unchanged
for _kind in ("bb", "st"):
    for _s in R2_SCALES:
        R2_ARM_TABLE[f"NL_{_kind}_s{_s}"] = {
            "scheme": "bin_balanced" if _kind == "bb" else "stratified_priority",
            **({} if _kind == "bb" else {"lambda_P": 0.35, "alpha_global": 0.5}),
            "rule_enabled": False, "legacy_pipeline": "LEGACY_PIPELINE_NL", "round": PILOT_R2,
            "r2": {"scale": float(_s), "stratified": _kind == "st"},
            "definition": f"MS-R2 {'bin-balanced' if _kind == 'bb' else 'MS_s35a5-style stratified'} starts, fixed 2800 "
                          f"(LR 3e-4 to 2400, decay 2401-2800), concentration scale 1 -> {_s} over 2001-2200, held to 2800"}
#: MS-R3 (prompt D2-D4): actor variant x starts x noise landing, 12 arms ``{actor}_{bb|st}_s{1|16}``. Each entry is a copy
#: of the MS-R2 ``NL_{bb|st}_s{s}`` entry (same sampler settings, same fixed-2800 schedule and ramp, so the shared builder
#: block below writes the same optional keys) with the MS-R3 round tag and an ``r3`` block: ``init_digest`` is written for
#: all twelve arms and ``actor_variant`` for the ``relu`` / ``t10`` actors only (the ``t1`` arms are the MS-R2 arms re-run
#: on the new code and carry no variant key).
R3_ACTORS = ("t1", "relu", "t10")
R3_SCALES = (1, 16)
R3_ARMS = tuple(f"{a}_{k}_s{s}" for a in R3_ACTORS for k in ("bb", "st") for s in R3_SCALES)
R3_ARM_TABLE: Dict[str, Dict[str, Any]] = {}     # kept apart from ARMS and R2_ARM_TABLE: both are unchanged
R3_ACTOR_TEXT = {"t1": "the locked tanh actor on d / B (control)", "relu": "ReLU hidden units on d / B",
                 "t10": "tanh hidden units on 10 d / B (the stage feature is not scaled)"}
for _actor in R3_ACTORS:
    for _kind in ("bb", "st"):
        for _s in R3_SCALES:
            R3_ARM_TABLE[f"{_actor}_{_kind}_s{_s}"] = {
                **copy.deepcopy(R2_ARM_TABLE[f"NL_{_kind}_s{_s}"]), "round": PILOT_R3,
                "r3": {"actor": _actor, "kind": _kind},
                "definition": f"MS-R3 {_actor} actor ({R3_ACTOR_TEXT[_actor]}), "
                              f"{'bin-balanced' if _kind == 'bb' else 'MS_s35a5-style stratified'} starts, terminal stage as "
                              f"NL_{_kind}_s{_s} of MS-R2 (fixed 2800, concentration scale 1 -> {_s} over 2001-2200)"}
#: each ``t1`` arm is the MS-R2 arm of the same starts and s re-run on the new code (check C-MS5)
R3_MS_R2_REFERENCE: Dict[str, str] = {f"t1_{_k}_s{_s}": f"NL_{_k}_s{_s}" for _k in ("bb", "st") for _s in R3_SCALES}
PILOT_ARMS = ("MS_rule", "MS_s25a0", "MS_s25a5", "MS_s35a0", "MS_s35a5")
#: the budget-matched control of addendum A1: launched in the pilot wave, in the same root as the rule arms
CONTROL_2400_ARM = "MS_base2400"

#: the defaults of D4 / D5 (the calibration may move the keys D8 lists; epsilon and the coverage may not move)
DEFAULT_PARAMS: Dict[str, Any] = {
    "K": 25, "M": 3, "conc_limit": 0.04, "localized_fraction": 0.25,
    "alpha_polish": 0.5, "ema_beta": 0.5, "near_tie_half_width": 20.0,
    "stages": {
        "2": {"eps": 0.005, "rho": 0.03, "tau": 0.02, "n_block": 400, "u_cap": 2000, "n_land": 400},
        "1": {"eps": 0.005, "rho": 0.03, "tau": 0.02, "n_block": 200, "u_cap": 600, "n_land": 400},
    },
}
#: MS_base: fixed budgets and v2.0's LR windows (protocols/v2_T2_locked_v2_0.json pipeline.lr_decay)
LEGACY_PIPELINE: Dict[str, Dict[str, Any]] = {
    "budgets": {"2": 1600, "1": 600},
    "lr_windows": {"2": [{"first": 1201, "last": 1600, "start": 3e-4, "end": 3e-5}],
                   "1": [{"first": 1, "last": 600, "start": 3e-4, "end": 3e-5}]},
}
#: MS_base2400 (addendum A1): terminal stage fixed 2400, LR constant 3e-4 to local 2000 and linear 3e-4 -> 3e-5
#: over local 2001-2400 (the same ``lr_linear`` form and end value as the rule arms' landing); stage 1 as MS_base
LEGACY_PIPELINE_2400: Dict[str, Dict[str, Any]] = {
    "budgets": {"2": 2400, "1": 600},
    "lr_windows": {"2": [{"first": 2001, "last": 2400, "start": 3e-4, "end": 3e-5}],
                   "1": [{"first": 1, "last": 600, "start": 3e-4, "end": 3e-5}]},
}
#: MS-R2 (D2): terminal stage fixed 2800, LR constant 3e-4 to local 2400, linear 3e-4 -> 3e-5 over 2401-2800; stage 1 as MS_base
LEGACY_PIPELINE_NL: Dict[str, Dict[str, Any]] = {
    "budgets": {"2": 2800, "1": 600},
    "lr_windows": {"2": [{"first": 2401, "last": 2800, "start": 3e-4, "end": 3e-5}],
                   "1": [{"first": 1, "last": 600, "start": 3e-4, "end": 3e-5}]},
}
#: MS-R2 (D2): concentration scale 1 at update 2001, s at 2200 (linear), s afterwards
CONC_RAMP_FIRST, CONC_RAMP_LAST = 2001, 2200
LEGACY_PIPELINES: Dict[str, Dict[str, Dict[str, Any]]] = {"LEGACY_PIPELINE": LEGACY_PIPELINE,
                                                          "LEGACY_PIPELINE_2400": LEGACY_PIPELINE_2400,
                                                          "LEGACY_PIPELINE_NL": LEGACY_PIPELINE_NL}


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """SHA-256 of the canonical (sorted-key) JSON text of an object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> Dict[str, Any]:
    """The locked v2.0 protocol JSON."""
    with open(path) as f:
        return json.load(f)


def bin_balanced_near_tie_share(spec: GameSpec, bin_width: float, half_width: float) -> float:
    """n_near / n_bins of the terminal stage: the bin-balanced near-tie share (4/40 at q=50, 4/44 at q=60)."""
    sampler = StartSampler(spec, bin_width)
    st = sampler.strata(spec.T, half_width)
    return float(st["near"].sum()) / float(sampler.n_bins(spec.T))


def build_config(proto: Dict[str, Any], q: int, seed: int, arm: str, out_dir: str,
                 params: Optional[Dict[str, Any]] = None, protocol_path: Path = PROTOCOL_PATH,
                 T: int = 2) -> Dict[str, Any]:
    """The ``ms_run_config/1`` of one (q, seed, arm) (the v2.0 record of ``q`` embedded).

    ``T = 2`` is the experiment; ``T = 3`` (tests only) takes the same record with ``game.T = 3`` (the T = 2
    fields of the record removed) and needs ``params['stages']`` with the stages 1..3 (and, for the legacy arm,
    ``params['legacy_pipeline']``).

    Args:
        proto: The locked v2.0 protocol (dict).
        q: Noise half-width, in ``proto['q_values']``.
        seed: Run seed.
        arm: One of :data:`ARMS`, :data:`R2_ARM_TABLE` or :data:`R3_ARM_TABLE`.
        out_dir: Output directory (recorded in the embedded record, as the locked entry point does).
        params: Rule / sampler parameters (default :data:`DEFAULT_PARAMS`).
        protocol_path: The protocol file (its SHA-256 is recorded).
        T: Horizon of the game (2 or 3).

    Returns:
        The config dict (not validated; ``run.run_ms_stagewise.validate_config`` does that).
    """
    if arm not in ARMS and arm not in R2_ARM_TABLE and arm not in R3_ARM_TABLE:
        raise KeyError(f"unknown arm {arm!r}; arms are {sorted(ARMS) + sorted(R2_ARM_TABLE) + sorted(R3_ARM_TABLE)}")
    if q not in proto["q_values"]:
        raise KeyError(f"q={q} not in the protocol's q_values {proto['q_values']}")
    prm = copy.deepcopy(DEFAULT_PARAMS if params is None else params)
    a = ARMS[arm] if arm in ARMS else (R2_ARM_TABLE[arm] if arm in R2_ARM_TABLE else R3_ARM_TABLE[arm])
    rec0 = proto["records"][str(q)]
    rec = copy.deepcopy(rec0)
    pilot_tag = a.get("round", PILOT)
    run_name = f"{RUN_PREFIX[pilot_tag]}_q{q}_s{seed}_{arm}"
    rec.update(seed=int(seed), run=run_name, output_dir=out_dir)
    if T != 2:
        rec["game"]["T"] = int(T)
        for k in ("es_bins_stage2", "domain_half_stage2"):
            rec.pop(k, None)
    spec = GameSpec(**{k: rec["game"][k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")})
    bin_w = float(rec["protocol"]["es_bin_width"])
    if a["scheme"] == "bin_balanced":
        sw: Dict[str, Any] = {"scheme": "bin_balanced"}
    else:
        lam = a["lambda_P"]
        if lam is None:
            lam = bin_balanced_near_tie_share(spec, bin_w, float(prm["near_tie_half_width"]))
        sw = {"scheme": "stratified_priority", "lambda_P": float(lam), "alpha_global": float(a["alpha_global"]),
              "alpha_polish": float(prm["alpha_polish"]), "ema_beta": float(prm["ema_beta"]),
              "near_tie_half_width": float(prm["near_tie_half_width"])}
    pipeline: Dict[str, Any] = {"T": int(spec.T), "phases": list(range(int(spec.T), 0, -1)),
                                "budgets": None, "lr_windows": None}
    if a["rule_enabled"]:
        stages = {t: dict(v) for t, v in prm["stages"].items()}
    else:
        stages = {t: {k: e[k] for k in ("eps", "rho", "tau")} for t, e in prm["stages"].items()}
        if "legacy_pipeline" in a:      # an arm with its own fixed budgets (MS_base2400); T = 2 only
            if T != 2:
                raise ValueError(f"arm {arm} has its own legacy pipeline, defined for T = 2 only")
            legacy = LEGACY_PIPELINES[a["legacy_pipeline"]]
        else:
            legacy = prm.get("legacy_pipeline", LEGACY_PIPELINE if T == 2 else None)
        if legacy is None:
            raise ValueError("the legacy arm at T != 2 needs params['legacy_pipeline']")
        pipeline["budgets"] = copy.deepcopy(legacy["budgets"])
        pipeline["lr_windows"] = copy.deepcopy(legacy["lr_windows"])
    cfg: Dict[str, Any] = {
        "schema": SCHEMA, "base_commit": proto["base_commit"], "pilot": pilot_tag, "arm": arm, "run": run_name,
        "q": int(q), "seed": int(seed), "record": rec,
        "protocol_ref": {"path": PROTOCOL_REL, "sha256": sha256_file(protocol_path)},
        "record_sha256": sha256_json(rec0),
        "protocol_gates": {k: proto[k] for k in ("gates", "secondary", "stage1_decomposition")},
        "threads_per_process": proto["threads_per_process"], "pipeline": pipeline,
        "start_weights": sw,
        "rule": {"enabled": bool(a["rule_enabled"]), "K": int(prm["K"]), "M": int(prm["M"]),
                 "conc_limit": float(prm["conc_limit"]), "localized_fraction": float(prm["localized_fraction"]),
                 "stages": stages},
        "clamp_likelihood": "density", "full_state_at": [],
    }
    if sw["scheme"] == "stratified_priority":
        cfg["derived"] = derive_start_info(spec, sw, bin_w)
    if "r2" in a:                               # MS-R2: the optional runner keys (absent for every MS-R1 arm)
        cfg["noise_report"] = True
        if a["r2"]["stratified"]:
            cfg["fixed_global_sampler"] = True
        if a["r2"]["scale"] != 1.0:
            cfg["conc_scale_schedule"] = {"stage": int(spec.T), "local_first": CONC_RAMP_FIRST,
                                          "local_last": CONC_RAMP_LAST, "scale_first": 1.0,
                                          "scale_last": float(a["r2"]["scale"])}
    if "r3" in a:                               # MS-R3: the optional actor keys (absent for every earlier arm)
        cfg["init_digest"] = True
        if a["r3"]["actor"] != "t1":            # a t1 arm writes no variant key: absence means the locked actor
            cfg["actor_variant"] = a["r3"]["actor"]
    return cfg


def r3_arm_parts(arm: str) -> Tuple[str, str, int]:
    """``(actor, kind, s)`` of an MS-R3 arm: ``kind`` is ``"bb"`` (bin-balanced) or ``"st"`` (stratified), ``s`` the end
    value of the concentration scale (1 or 16).

    Raises:
        KeyError: If ``arm`` is not an MS-R3 arm.
    """
    e = R3_ARM_TABLE[arm]
    return e["r3"]["actor"], e["r3"]["kind"], int(e["r2"]["scale"])


def r2_equivalent(arm: str) -> str:
    """The MS-R2 arm ``NL_{kind}_s{s}`` with the same starts and s as an MS-R3 arm (for a ``t1`` arm: its C-MS5 reference)."""
    _, kind, s = r3_arm_parts(arm)
    return f"NL_{kind}_s{s}"


def r3_arms_of(actors: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    """The MS-R3 arms of the given actors (default: all three), in table order.

    Raises:
        ValueError: If a name is not one of :data:`R3_ACTORS`.
    """
    keep = R3_ACTORS if actors is None else tuple(actors)
    bad = [x for x in keep if x not in R3_ACTORS]
    if bad:
        raise ValueError(f"unknown actor(s) {bad}; the actors are {list(R3_ACTORS)}")
    return tuple(a for a in R3_ARMS if R3_ARM_TABLE[a]["r3"]["actor"] in keep)


def parse_actors(text: str) -> Tuple[str, ...]:
    """The ``--actors`` comma list (``"t1,relu,t10"`` or a subset) as a tuple in the canonical order of :data:`R3_ACTORS`.

    Raises:
        ValueError: If a name (also an empty one) is not one of :data:`R3_ACTORS`.
    """
    names = [x.strip() for x in text.split(",")]
    bad = [x for x in names if x not in R3_ACTORS]
    if bad:
        raise ValueError(f"unknown actor(s) {bad} in --actors {text!r}; the actors are {list(R3_ACTORS)}")
    return tuple(a for a in R3_ACTORS if a in names)


__all__ = ["ARMS", "PILOT_ARMS", "R2_ARMS", "R2_ARM_TABLE", "R2_SCALES", "CONTROL_2400_ARM", "DEFAULT_PARAMS", "LEGACY_PIPELINE",
           "LEGACY_PIPELINE_2400", "LEGACY_PIPELINE_NL", "CONC_RAMP_FIRST", "CONC_RAMP_LAST", "PILOT_R2", "PILOT_R3",
           "RUN_PREFIX", "R3_ACTORS", "R3_ARMS", "R3_ARM_TABLE", "R3_SCALES", "R3_MS_R2_REFERENCE", "r3_arm_parts",
           "r2_equivalent", "r3_arms_of", "parse_actors", "build_config", "load_protocol",
           "REPORT_NEAR_TIE_HALF_WIDTH", "DEFAULT_QS", "DEFAULT_SEEDS", "sha256_file", "sha256_json"]
