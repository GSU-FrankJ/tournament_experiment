"""Run configs of round T3-R1: the arms ``bw``, ``jt``, ``jt_es`` (T = 3) and ``jt2`` (T = 2).

  * ``bw``: backward-induction training at T = 3 through ``run/run_ms_stagewise.py`` (config
    ``ms_run_config/1``), built with ``tools/ms/ms_configs.build_config(..., T=3, arm_table=BW_ARM_TABLE)`` and
    ``reports/t3/r1/params_bw_t3.json``, then the field settings of D3 (``pilot``, ``run``, ``record.run``,
    ``init_digest``).
  * ``jt``, ``jt_es``, ``jt2``: joint whole-game training through ``run/run_joint.py`` (config
    ``joint_run_config/1``); the v2.0 record of q is embedded with ``game.T`` set to the horizon (the
    T = 2-only fields removed at T = 3, exactly the transformation of ``build_config``).

The production values of D5 (512 episodes per update, the updates, the start mixes, the decay windows,
``check_every`` 25, the bw budgets and windows) are pinned here (:data:`JOINT_BUDGETS`, :data:`BW_BUDGETS`,
:func:`production_problems`) and verified by the launcher and by the launch checks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import ms_configs as mc  # noqa: E402
from run.run_joint import ROUND, SCHEMA as JOINT_SCHEMA  # noqa: E402

PROTOCOL_REL = mc.PROTOCOL_REL
PROTOCOL_PATH = mc.PROTOCOL_PATH
PARAMS_REL = "reports/t3/r1/params_bw_t3.json"
PARAMS_PATH = ROOT / PARAMS_REL
PARAMS_SHA256 = "db5156962a2d7a3eca0b7d0440d995d84dc57bb88c7066e38d7d61e40148e976"
MS_PARAMS_REL = "reports/ms/r1/prereg_parameters.json"
MS_PARAMS_SHA256 = "0fbfc01857c5abe339ab1926361cd57d6e988da20e754fd6dea3ffa4e076b406"
QS = (50, 60)
ARMS = ("bw", "jt", "jt_es", "jt2")
JOINT_ARMS = ("jt", "jt_es", "jt2")
T_OF_ARM = {"bw": 3, "jt": 3, "jt_es": 3, "jt2": 2}
WAVES = ("smoke", "t3", "t2j")
WAVE_ARMS = {"smoke": ARMS, "t3": ("bw", "jt", "jt_es"), "t2j": ("jt2",)}
WAVE_SEEDS = {"smoke": (50500,), "t3": tuple(range(50501, 50521)), "t2j": tuple(range(30501, 30521))}
WAVE_DIR = {"smoke": "smoke", "t3": "runs", "t2j": "runs_t2"}
#: seeds that appear in no config of this round: the reserved block and the MS development seeds
FORBIDDEN_SEEDS = tuple(range(40501, 40521)) + tuple(range(10501, 10531))
DEFAULT_ROOT = ROOT / "results" / "t3_r1"

EPISODES = 512
CHECK_EVERY = 25
LR_START, LR_END = 3e-4, 3e-5
#: D5: updates, start mix and decay window (first, last) of the joint arms
JOINT_BUDGETS: Dict[str, Dict[str, Any]] = {
    "jt": {"T": 3, "updates": 1267, "starts": {"root": 512, "stage2": 0, "stage3": 0}, "window": (951, 1267)},
    "jt_es": {"T": 3, "updates": 1754, "starts": {"root": 256, "stage2": 85, "stage3": 171},
              "window": (1316, 1754)},
    "jt2": {"T": 2, "updates": 1100, "starts": {"root": 512, "stage2": 0}, "window": (826, 1100)},
}
#: D3: the legacy budgets and LR windows of the bw arm (per stage)
BW_BUDGETS: Dict[str, int] = {"3": 1600, "2": 1600, "1": 600}
BW_WINDOWS: Dict[str, List[Dict[str, Any]]] = {
    "3": [{"first": 1201, "last": 1600, "start": LR_START, "end": LR_END}],
    "2": [{"first": 1201, "last": 1600, "start": LR_START, "end": LR_END}],
    "1": [{"first": 1, "last": 600, "start": LR_START, "end": LR_END}],
}
BW_STAGE_PARAMS: Dict[str, Dict[str, float]] = {
    "3": {"eps": 0.005, "rho": 0.05, "tau": 0.02},
    "2": {"eps": 0.005, "rho": 0.03, "tau": 0.02},
    "1": {"eps": 0.005, "rho": 0.03, "tau": 0.02},
}
BW_PARAM_KEYS = ("K", "M", "conc_limit", "localized_fraction", "alpha_polish", "ema_beta", "near_tie_half_width",
                 "stages", "legacy_pipeline")

BW_ARM_TABLE: Dict[str, Dict[str, Any]] = {
    "bw": {"scheme": "bin_balanced", "rule_enabled": False,
           "definition": "T3-R1 backward-induction training (BW) at T = 3, the legacy settings of "
                         "run/run_ms_stagewise.py: bin-balanced starts, rule disabled (checks every 25 local updates "
                         "recorded), fixed budgets {3: 1600, 2: 1600, 1: 600}, LR 3e-4 with a linear decay to 3e-5 "
                         "over local updates 1201-1600 (stages 3 and 2) and 1-600 (stage 1)"}}


# ---------------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    return mc.sha256_file(Path(path))


def decay_window(updates: int) -> Tuple[int, int]:
    """The last 25 % of the updates, rounded up: ``(first, last)``."""
    return updates - math.ceil(0.25 * updates) + 1, updates


def config_text(cfg: Dict[str, Any]) -> str:
    """The text the launcher writes to ``run_config.json`` (``json.dump(cfg, f, indent=1)``)."""
    return json.dumps(cfg, indent=1)


def config_sha256(cfg: Dict[str, Any]) -> str:
    """SHA-256 of the ``run_config.json`` text of a config."""
    return hashlib.sha256(config_text(cfg).encode()).hexdigest()


def _flatten(o: Any, path: str = "") -> Dict[str, Any]:
    if isinstance(o, dict) and o:
        out: Dict[str, Any] = {}
        for k, v in o.items():
            out.update(_flatten(v, f"{path}.{k}" if path else str(k)))
        return out
    if isinstance(o, list) and o:
        out = {}
        for i, v in enumerate(o):
            out.update(_flatten(v, f"{path}[{i}]"))
        return out
    return {path: o}


def config_diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Sorted dotted paths in which two configs differ (a path present on one side only counts)."""
    fa, fb = _flatten(a), _flatten(b)
    return sorted(k for k in set(fa) | set(fb) if k not in fa or k not in fb or fa[k] != fb[k])


def run_name(q: int, seed: int, arm: str) -> str:
    """``t3r1_q{q}_s{seed}_{arm}``."""
    return f"t3r1_q{q}_s{seed}_{arm}"


def run_dir(root: Path, wave: str, q: int, seed: int, arm: str) -> Path:
    """Run directory: ``runs/q{q}/seed{s}/{arm}`` (t3), ``runs_t2/q{q}/seed{s}/jt2`` (t2j),
    ``smoke/q{q}/{arm}`` (smoke)."""
    base = Path(root) / WAVE_DIR[wave] / f"q{q}"
    return base / arm if wave == "smoke" else base / f"seed{seed}" / arm


def check_seeds(wave: str, seeds: Sequence[int]) -> None:
    """Refuse a seed outside the block of the wave and every excluded seed.

    Raises:
        ValueError: A seed of 40501-40520 or 10501-10530, or a seed outside the wave's block.
    """
    bad_forbidden = [s for s in seeds if s in FORBIDDEN_SEEDS]
    if bad_forbidden:
        raise ValueError(f"seeds {bad_forbidden} are excluded from this round (40501-40520, 10501-10530)")
    allowed = WAVE_SEEDS[wave]
    bad = [s for s in seeds if s not in allowed]
    if bad:
        raise ValueError(f"wave {wave} accepts the seeds {allowed[0]}-{allowed[-1]} only; got {bad}")


def load_protocol() -> Dict[str, Any]:
    """The locked v2.0 protocol (dict)."""
    return mc.load_protocol()


def load_bw_params(path: Optional[Path] = None) -> Dict[str, Any]:
    """``reports/t3/r1/params_bw_t3.json``.

    Raises:
        ValueError: If the key set differs from :data:`BW_PARAM_KEYS`.
    """
    with open(path or PARAMS_PATH) as f:
        prm = json.load(f)
    if set(prm) != set(BW_PARAM_KEYS):
        raise ValueError(f"parameter file keys {sorted(prm)} != {sorted(BW_PARAM_KEYS)}")
    return prm


# ---------------------------------------------------------------------------------- builders
def build_bw_config(proto: Dict[str, Any], q: int, seed: int, out_dir: str, params: Dict[str, Any],
                    protocol_path: Path = PROTOCOL_PATH) -> Dict[str, Any]:
    """The ``ms_run_config/1`` of one ``bw`` run (T = 3, legacy settings; D3).

    ``build_config(..., T=3, arm_table=BW_ARM_TABLE)``, then ``pilot`` = "t3_r1", ``run`` and ``record.run`` =
    ``t3r1_q{q}_s{seed}_bw``, ``init_digest`` = true.
    """
    cfg = mc.build_config(proto, int(q), int(seed), "bw", out_dir, params, protocol_path, T=3,
                          arm_table=BW_ARM_TABLE)
    name = run_name(q, seed, "bw")
    cfg["pilot"] = ROUND
    cfg["run"] = name
    cfg["record"]["run"] = name
    cfg["init_digest"] = True
    return cfg


def build_joint_config(proto: Dict[str, Any], q: int, seed: int, arm: str, out_dir: str,
                       protocol_path: Path = PROTOCOL_PATH) -> Dict[str, Any]:
    """The ``joint_run_config/1`` of one ``jt`` / ``jt_es`` / ``jt2`` run with the production values of D5."""
    if arm not in JOINT_ARMS:
        raise KeyError(f"unknown joint arm {arm!r}; arms are {list(JOINT_ARMS)}")
    if q not in proto["q_values"]:
        raise KeyError(f"q={q} not in the protocol's q_values {proto['q_values']}")
    b = JOINT_BUDGETS[arm]
    T = int(b["T"])
    rec0 = proto["records"][str(q)]
    rec = copy.deepcopy(rec0)
    name = run_name(q, seed, arm)
    rec.update(seed=int(seed), run=name, output_dir=out_dir)
    if T != 2:
        rec["game"]["T"] = T
        for k in ("es_bins_stage2", "domain_half_stage2"):
            rec.pop(k, None)
    first, last = b["window"]
    return {
        "schema": JOINT_SCHEMA, "base_commit": proto["base_commit"], "round": ROUND, "arm": arm, "run": name,
        "q": int(q), "seed": int(seed), "record": rec,
        "protocol_ref": {"path": PROTOCOL_REL, "sha256": sha256_file(protocol_path)},
        "record_sha256": mc.sha256_json(rec0), "threads_per_process": proto["threads_per_process"],
        "joint": {"T": T, "updates": int(b["updates"]), "starts": dict(b["starts"]),
                  "lr_windows": [{"first": first, "last": last, "start": LR_START, "end": LR_END}],
                  "check_every": CHECK_EVERY},
        "reward_mode": "expected", "clamp_likelihood": "density", "init_digest": True}


def build_config(proto: Dict[str, Any], q: int, seed: int, arm: str, out_dir: str,
                 params: Optional[Dict[str, Any]] = None,
                 protocol_path: Path = PROTOCOL_PATH) -> Dict[str, Any]:
    """The config of one run of any arm (``bw``: ``params`` is required)."""
    if arm == "bw":
        if params is None:
            raise ValueError("the bw arm needs the parameters of reports/t3/r1/params_bw_t3.json")
        return build_bw_config(proto, q, seed, out_dir, params, protocol_path)
    return build_joint_config(proto, q, seed, arm, out_dir, protocol_path)


# ---------------------------------------------------------------------------------- D5 budgets
def expected_budget(arm: str) -> Dict[str, Any]:
    """The budget of one run by D5, from the pinned values: updates, policy rows per update / in total by
    stage, minibatch steps (10 epochs of ``ceil(rows / 256)`` minibatches per update).

    Policy rows per update of ``bw`` are 512 in the stage of the phase. ``v20`` is the T = 2 reference
    (Phase A 1600 updates of 512 rows; Phase B 600 updates whose minibatches run over 1024 rows).
    """
    epochs, mb = 10, 256
    if arm == "bw":
        rows = {int(t): int(n) * EPISODES for t, n in BW_BUDGETS.items()}
        steps = sum(n * epochs * math.ceil(EPISODES / mb) for n in BW_BUDGETS.values())
        return {"updates": sum(BW_BUDGETS.values()), "rows_by_stage": dict(sorted(rows.items())),
                "rows_total": sum(rows.values()), "minibatch_steps": int(steps),
                "rows_per_update": {t: EPISODES for t in (1, 2, 3)}}
    if arm == "v20":
        rows = {2: 1600 * EPISODES, 1: 600 * EPISODES}
        steps = 1600 * epochs * math.ceil(EPISODES / mb) + 600 * epochs * math.ceil(2 * EPISODES / mb)
        return {"updates": 2200, "rows_by_stage": dict(sorted(rows.items())), "rows_total": sum(rows.values()),
                "minibatch_steps": int(steps), "rows_per_update": {1: EPISODES, 2: EPISODES}}
    b = JOINT_BUDGETS[arm]
    T, U, st = int(b["T"]), int(b["updates"]), b["starts"]
    n = [st.get("root", 0), st.get("stage2", 0), st.get("stage3", 0)]
    per_update = {t: int(sum(n[:t])) for t in range(1, T + 1)}
    rows_update = int(sum(per_update.values()))          # one minibatch partition over all learner rows
    return {"updates": U, "rows_per_update": per_update,
            "rows_by_stage": {t: per_update[t] * U for t in per_update},
            "rows_total": rows_update * U,
            "minibatch_steps": int(U * epochs * math.ceil(rows_update / mb))}


def production_problems(cfg: Dict[str, Any]) -> List[str]:
    """Departures of a planned config from the production values of D1-D5 (empty list: none)."""
    out: List[str] = []
    arm = cfg.get("arm")
    q = cfg.get("q")
    rec = cfg["record"]
    if rec["protocol"]["episodes_per_update"] != EPISODES:
        out.append(f"episodes_per_update {rec['protocol']['episodes_per_update']} != {EPISODES}")
    if cfg.get("run") != run_name(q, cfg["seed"], arm) or rec.get("run") != cfg.get("run"):
        out.append(f"run name {cfg.get('run')!r} / record.run {rec.get('run')!r} != {run_name(q, cfg['seed'], arm)!r}")
    if arm == "bw":
        pl = cfg["pipeline"]
        if pl["T"] != 3 or pl["budgets"] != BW_BUDGETS or pl["lr_windows"] != BW_WINDOWS:
            out.append("bw pipeline budgets / lr_windows differ from D3")
        if cfg["rule"]["enabled"] is not False or cfg["start_weights"] != {"scheme": "bin_balanced"}:
            out.append("bw rule / start_weights differ from D3 (rule disabled, bin_balanced)")
        stg = {t: {k: s[k] for k in ("eps", "rho", "tau")} for t, s in cfg["rule"]["stages"].items()}
        if stg != BW_STAGE_PARAMS or cfg["rule"]["K"] != 25:
            out.append("bw rule stage parameters / K differ from D3")
        if cfg.get("init_digest") is not True or cfg.get("pilot") != ROUND:
            out.append("bw init_digest / pilot differ from D3")
    elif arm in JOINT_ARMS:
        b = JOINT_BUDGETS[arm]
        j = cfg["joint"]
        first, last = b["window"]
        want = {"T": b["T"], "updates": b["updates"], "starts": b["starts"],
                "lr_windows": [{"first": first, "last": last, "start": LR_START, "end": LR_END}],
                "check_every": CHECK_EVERY}
        if j != want:
            out.append(f"joint block {j} != {want}")
        if (first, last) != decay_window(b["updates"]):
            out.append(f"decay window {(first, last)} != the last 25 % of {b['updates']} rounded up")
    else:
        out.append(f"unknown arm {arm!r}")
    return out
