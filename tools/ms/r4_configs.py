"""Run configs (``ms_run_config/1``) of the MS-R4 arms (prompt D2-D4): seven arms, two experiments.

E1 (the stage-2 tie bias): ``t1_st_s16`` and ``relu_st_s16`` (controls, MS-R3's entries of the same names copied with the
round tag ``ms_r4``) and ``lrsm_st_s16`` (the candidate actor ``lrsm``: LeakyReLU hidden units, smooth unclamped mean).
Terminal stage fixed 2800 with the s = 16 noise landing, stratified starts (cross-stratum focus, alpha 0.5).

E2 (does the development DP-BR help PPO training): the t1 actor on v2.0's budgets (terminal stage 1600, stage 1 600), no
noise landing, fixed global sampler in every arm, and only the sampling feedback differs:

    fb_a0  alpha 0 (fixed stratified sampling)               fb_r   alpha 0.5, within-stratum focus = residual map
    fb_rg  alpha 0.5, within-stratum focus = residual + gain  fb_rx  alpha 0.5, the existing cross-stratum focus (descriptive)

``tools/ms/ms_configs.build_config`` builds the shared part through its additive ``arm_table`` argument (never through the
MS-R3 lookup by name); the optional runner keys ``focus_scope`` / ``focus_map`` are added here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ms"))

import ms_configs as mc  # noqa: E402

ROUND = mc.PILOT_R4                              # "ms_r4"; run prefix "msr4"
SEEDS = tuple(range(10511, 10531))               # the new development seeds of the pilot wave (wave r4)
R4_SEEDS = SEEDS                                 # the name the prompt uses (section 2.1)
SEEDS_CMS6 = tuple(range(10501, 10511))          # the seeds of the reproduction wave (r4_cms6)
FORBIDDEN_SEEDS = tuple(range(30501, 30521)) + tuple(range(40501, 40521))   # v2.0 confirmation block; reserved block
QS = (50, 60)
E1_ARMS = ("t1_st_s16", "relu_st_s16", "lrsm_st_s16")
E2_ARMS = ("fb_a0", "fb_r", "fb_rg", "fb_rx")
ARMS = E1_ARMS + E2_ARMS
CMS6_ARMS = ("t1_st_s16", "relu_st_s16")         # the two controls, re-run on the MS-R3 seeds (check C-MS6)
#: arm -> (comparator arm, label) for the keys-that-differ listing of the pre-registration
COMPARATOR: Dict[str, str] = {"relu_st_s16": "t1_st_s16", "lrsm_st_s16": "relu_st_s16", "fb_r": "fb_a0", "fb_rg": "fb_a0",
                              "fb_rx": "fb_a0"}
ACTOR_TEXT = {"t1": "the locked tanh actor on d / B (control)", "relu": "ReLU hidden units on d / B (control)",
              "lrsm": "LeakyReLU (slope 0.01) hidden units on d / B and the smooth unclamped mean m + (1 - 2 m) sigmoid(z0)"}

R4_E1_ARM_TABLE: Dict[str, Dict[str, Any]] = {}
for _actor in ("t1", "relu", "lrsm"):
    _arm = f"{_actor}_st_s16"
    _e = copy.deepcopy(mc.R3_ARM_TABLE["t1_st_s16"])         # MS-R3's t1_st_s16 entry as the template (stratified, s = 16)
    _e.update(round=ROUND, r3={"actor": _actor, "kind": "st"}, r4={"experiment": "E1"},
              definition=f"MS-R4 E1 {_actor} actor ({ACTOR_TEXT[_actor]}), stratified starts (cross-stratum focus, alpha 0.5), "
                         f"terminal stage fixed 2800 (LR 3e-4 to 2400, decay 2401-2800), concentration scale 1 -> 16 over "
                         f"2001-2200 held to 2800; stage 1 fixed 600")
    R4_E1_ARM_TABLE[_arm] = _e
# the two controls are MS-R3's entries of the same names apart from the round tag, the definition text and the r4 block
for _arm in CMS6_ARMS:
    _a, _b = copy.deepcopy(R4_E1_ARM_TABLE[_arm]), copy.deepcopy(mc.R3_ARM_TABLE[_arm])
    for _d in (_a, _b):
        for _k in ("round", "definition", "r4"):
            _d.pop(_k, None)
    assert _a == _b, _arm

#: E2: the same sampler settings in every arm (lambda_P 0.35, EMA 0.5, near-tie half-width 20), the budgets of MS_base
#: (``LEGACY_PIPELINE``: terminal stage 1600, stage 1 600), no concentration schedule (scale 1), the fixed global sampler
_E2_DEF = {
    "fb_a0": ("alpha_global 0.0", 0.0, None, None,
              "(1) fixed stratified sampling: alpha 0 (no focus acts; the checks and the rho_bar EMA run all the same)"),
    "fb_r": ("alpha_global 0.5", 0.5, "within_strata", None,
             "(2) residual-guided sampling: alpha 0.5, the focus (the residual map rho_bar) acts inside the strata only"),
    "fb_rg": ("alpha_global 0.5", 0.5, "within_strata", "residual_gain",
              "(3) residual + deviation-gain sampling: alpha 0.5, the focus (1/2 the residual map + 1/2 the gain map) acts "
              "inside the strata only"),
    "fb_rx": ("alpha_global 0.5", 0.5, None, None,
              "descriptive: the existing cross-stratum focus (the sampler of the MS-R3 st arms), alpha 0.5"),
}
R4_E2_ARM_TABLE: Dict[str, Dict[str, Any]] = {}
for _arm, (_, _alpha, _scope, _map, _text) in _E2_DEF.items():
    R4_E2_ARM_TABLE[_arm] = {
        "scheme": "stratified_priority", "lambda_P": 0.35, "alpha_global": _alpha, "rule_enabled": False,
        "legacy_pipeline": "LEGACY_PIPELINE", "round": ROUND,
        "r2": {"scale": 1.0, "stratified": True},            # noise_report + fixed_global_sampler, no concentration schedule
        "r3": {"actor": "t1", "kind": "st"},                 # init_digest, the locked actor (no actor_variant key)
        "r4": {"experiment": "E2", "focus_scope": _scope, "focus_map": _map},
        "definition": f"MS-R4 E2 {_text}; t1 actor, terminal stage fixed 1600 (LR 3e-4 to 1200, decay 1201-1600), "
                      f"stage 1 fixed 600, no noise landing"}
R4_ARM_TABLE: Dict[str, Dict[str, Any]] = {**R4_E1_ARM_TABLE, **R4_E2_ARM_TABLE}


def experiment_of(arm: str) -> str:
    """``"E1"`` or ``"E2"``.

    Raises:
        KeyError: If ``arm`` is not an MS-R4 arm.
    """
    return str(R4_ARM_TABLE[arm]["r4"]["experiment"])


def build_config_r4(proto: Dict[str, Any], q: int, seed: int, arm: str, out_dir: str,
                    params: Optional[Dict[str, Any]] = None, protocol_path: Path = mc.PROTOCOL_PATH,
                    T: int = 2) -> Dict[str, Any]:
    """The ``ms_run_config/1`` of one (q, seed, arm) of MS-R4, resolved from :data:`R4_ARM_TABLE` (never from the MS-R3 table).

    Args:
        proto: The locked v2.0 protocol (dict).
        q: Noise half-width, 50 or 60.
        seed: Run seed.
        arm: One of :data:`ARMS`.
        out_dir: Output directory (recorded in the embedded record).
        params: Rule / sampler parameters (the pre-registered ``reports/ms/r1/prereg_parameters.json``).
        protocol_path: The protocol file.
        T: Horizon (2; 3 for tests only).

    Returns:
        The config dict (not validated).

    Raises:
        KeyError: Unknown arm or q.
    """
    cfg = mc.build_config(proto, q, seed, arm, out_dir, params, protocol_path, T=T, arm_table=R4_ARM_TABLE)
    r4 = R4_ARM_TABLE[arm]["r4"]
    if r4.get("focus_scope"):
        cfg["focus_scope"] = r4["focus_scope"]
    if r4.get("focus_map"):
        cfg["focus_map"] = r4["focus_map"]
    return cfg


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


def top_level_diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Sorted top-level keys in which two configs differ."""
    return sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))


def arm_dir(root: Path, q: int, seed: int, arm: str) -> Path:
    """``<root>/q<q>/seed<seed>/<arm>``."""
    return Path(root) / f"q{q}" / f"seed{seed}" / arm


__all__ = ["ROUND", "SEEDS", "R4_SEEDS", "SEEDS_CMS6", "FORBIDDEN_SEEDS", "QS", "E1_ARMS", "E2_ARMS", "ARMS", "CMS6_ARMS", "COMPARATOR",
           "R4_E1_ARM_TABLE", "R4_E2_ARM_TABLE", "R4_ARM_TABLE", "experiment_of", "build_config_r4", "config_text",
           "config_sha256", "config_diff", "top_level_diff", "arm_dir"]
