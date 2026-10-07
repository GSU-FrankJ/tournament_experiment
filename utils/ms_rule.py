"""Development stop rule, blocks, targeted polishing, landing and freeze of one stage (MS-R1, D4).

A pure state machine: the training loop calls :meth:`StageController.lr` and
:meth:`StageController.sampler_setting` before local update ``j`` and
:meth:`StageController.after_update` after it; the controller asks for a development check (a callable
returning a :class:`utils.ms_residual.StageDiag`) only when one is due. Nothing here reads the closed
form: the only inputs are the ``StageDiag`` values.

Rule (per stage; the defaults of D4 are in :class:`StageRule`):

  * eligible check = valid and Delta <= eps and R <= rho and (tail term void or R_tail <= tau) and
    C <= conc_limit (all inclusive);
  * training runs in blocks of ``n_block`` updates at constant LR; a check is made every ``K`` local
    updates and at every block end; the streak of consecutive eligible checks is not reset by a block
    boundary;
  * ``M`` consecutive eligible checks = development stop: the block ends at that check and the landing
    window starts at the next update;
  * a block that ends without a stop is classified from the last check: ``S`` = non-tail bins with
    ``rho_bar > rho``; *localized* iff ``1 <= |S| <= ceil(loc_frac * n_nontail)`` and ``R > rho``,
    else *broad*; localized -> the next block is a polishing block (focus ``rho_bar`` restricted to
    ``S``, alpha = alpha_polish), broad -> a global block (focus ``rho_bar``, alpha = alpha_global);
  * blocks continue until ``u_cap`` training updates; reaching it without a stop starts the landing
    window with ``budget_forced = True``;
  * landing = ``n_land`` updates with the LR linear from ``lr_base`` to ``lr_end``, the sampler
    setting (block type, alpha, focus snapshot) of the block it follows; checks continue at cadence
    ``K`` and are reported but decide nothing; the stage is frozen after the last landing update.

With ``enabled = False`` (the legacy arm) the controller runs ``fixed_budget`` updates with the LR of
``lr_legacy`` and records only the update at which the rule would have fired.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from utils.ms_residual import StageDiag, ema_update

LrLinear = Callable[[float, float, int, int, int], float]   # (start, end, first, last, local) -> lr


@dataclass(frozen=True)
class StageRule:
    """Parameters of the rule of one stage."""

    stage: int
    enabled: bool = True
    K: int = 25
    M: int = 3
    eps: float = 0.005
    rho: float = 0.03
    tau: float = 0.02
    conc_limit: float = 0.04
    n_block: int = 400
    u_cap: int = 2000
    n_land: int = 400
    loc_frac: float = 0.25
    alpha_polish: float = 0.5
    alpha_global: float = 0.0
    ema_beta: float = 0.5
    lr_base: float = 3e-4
    lr_end: float = 3e-5
    n_nontail_bins: int = 0          # non-tail ES bins of D_t (0 at t = 1)
    fixed_budget: Optional[int] = None   # legacy arm only


@dataclass
class SamplerSetting:
    """What the sampler needs for the next update."""

    block_type: str                  # "global" | "polish" | "landing" | "legacy"
    alpha: float
    focus: Optional[np.ndarray]      # non-negative per-bin weights (NaN-free) or None -> p_PM
    followed_type: Optional[str] = None   # landing: the block type it follows


def classify_block_end(R: float, rho_bar: Optional[np.ndarray], rho: float, n_nontail: int,
                       loc_frac: float) -> Tuple[str, np.ndarray]:
    """Localized / broad classification of a block that ended without a development stop.

    Returns:
        ``("localized" | "broad", S)`` with ``S`` the indices of the non-tail bins with
        ``rho_bar > rho``.
    """
    if rho_bar is None:
        return "broad", np.zeros(0, dtype=int)
    s = np.flatnonzero(np.where(np.isnan(rho_bar), -np.inf, rho_bar) > rho)
    cap = math.ceil(loc_frac * n_nontail - 1e-12)
    localized = bool(1 <= s.size <= cap and R > rho)
    return ("localized" if localized else "broad"), s


def is_eligible(rule: StageRule, diag: StageDiag) -> bool:
    """The eligibility of one check (D4): NaN fails every comparison."""
    if not diag.valid:
        return False
    tail_ok = (not diag.tail_term) or bool(diag.R_tail <= rule.tau)
    return bool(diag.delta_over_dw <= rule.eps and diag.R <= rule.rho and tail_ok
                and diag.C <= rule.conc_limit)


@dataclass
class Step:
    """What :meth:`StageController.after_update` did."""

    check: Optional[Dict[str, Any]] = None      # the check row (None if no check was due)
    events: List[str] = field(default_factory=list)
    finished: bool = False


class StageController:
    """Block / landing / freeze state machine of one stage (see the module docstring)."""

    def __init__(self, rule: StageRule, lr_linear: LrLinear,
                 lr_legacy: Optional[Callable[[int], float]] = None):
        self.rule = rule
        self.lr_linear = lr_linear
        self.lr_legacy = lr_legacy
        if rule.enabled:
            if rule.fixed_budget is not None:
                raise ValueError("an enabled rule takes no fixed budget")
            if rule.n_land < 2 or rule.n_block < 1 or rule.u_cap < 1 or rule.K < 1 or rule.M < 1:
                raise ValueError("n_land >= 2, n_block >= 1, u_cap >= 1, K >= 1, M >= 1 required")
        else:
            if rule.fixed_budget is None or rule.fixed_budget < 1 or lr_legacy is None:
                raise ValueError("the legacy rule needs a positive fixed budget and an LR function")
        self.mode = "train"
        self.rho_bar: Optional[np.ndarray] = None
        self.consec = 0
        self.n_checks = 0
        self.would_fire: Optional[int] = None          # first local update with M consecutive eligible
        self.blocks: List[Dict[str, Any]] = []
        self.landing: Optional[Dict[str, Any]] = None
        self.budget_forced = False
        self.fire_local: Optional[int] = None
        self.local_done = 0
        self._land_setting: Optional[SamplerSetting] = None
        self._S = np.zeros(0, dtype=int)
        if rule.enabled:
            self._open_block(1, "global")
        else:
            self.blocks.append({"block_id": 1, "type": "legacy", "first_local": 1,
                                "last_local": None, "exit_reason": None})
            self.block_end = int(rule.fixed_budget)
            self.block_type = "legacy"
            self.block_id = 1

    # ------------------------------------------------------------------ block bookkeeping
    def _open_block(self, first: int, btype: str, S: Optional[np.ndarray] = None) -> None:
        r = self.rule
        self.block_id = len(self.blocks) + 1
        self.block_type = btype
        self.block_end = min(first + r.n_block - 1, r.u_cap)
        self._S = np.zeros(0, dtype=int) if S is None else np.asarray(S, dtype=int)
        self.blocks.append({"block_id": self.block_id, "type": btype, "first_local": first,
                            "last_local": None, "exit_reason": None, "fire_local": None,
                            "S": [int(x) for x in self._S], "n_S": int(self._S.size),
                            "classification": None})

    def _close_block(self, j: int, reason: str) -> Dict[str, Any]:
        b = self.blocks[-1]
        b["last_local"] = int(j)
        b["exit_reason"] = reason
        return b

    # ------------------------------------------------------------------ queried by the loop
    @property
    def finished(self) -> bool:
        """True after the last landing update (or the last fixed-budget update): the stage is to be frozen."""
        return self.mode == "done"

    def lr(self, j: int) -> float:
        """LR before local update ``j``."""
        r = self.rule
        if not r.enabled:
            return float(self.lr_legacy(j))      # type: ignore[misc]
        if self.mode == "land":
            assert self.landing is not None
            return float(self.lr_linear(r.lr_base, r.lr_end, self.landing["first_local"],
                                        self.landing["last_local"], j))
        return float(r.lr_base)

    def sampler_setting(self) -> SamplerSetting:
        """Block type, alpha and focus for the next update."""
        r = self.rule
        if not r.enabled:
            return SamplerSetting("legacy", 0.0, None)
        if self.mode == "land":
            assert self._land_setting is not None
            return self._land_setting
        return self._current_setting()

    def _current_setting(self) -> SamplerSetting:
        r = self.rule
        rb = None if self.rho_bar is None else np.nan_to_num(self.rho_bar, nan=0.0)
        if self.block_type == "polish":
            if rb is None:
                focus = None
            else:
                mask = np.zeros(rb.size, dtype=bool)
                mask[self._S] = True
                focus = np.where(mask, rb, 0.0)
            return SamplerSetting("polish", r.alpha_polish, focus)
        return SamplerSetting("global", r.alpha_global, rb)

    def block_label(self) -> Tuple[int, str]:
        """(block id, type) the next update belongs to ("landing" inside the landing window)."""
        if self.mode == "land":
            return self.block_id + 1 if self.rule.enabled else self.block_id, "landing"
        return self.block_id, self.block_type

    # ------------------------------------------------------------------ after an update
    def is_check_due(self, j: int) -> bool:
        """True if a development check follows local update ``j`` (every K updates; training block ends)."""
        r = self.rule
        if j % r.K == 0:
            return True
        if self.mode == "train" and j == self.block_end:
            return True
        return False

    def after_update(self, j: int, diag_fn: Callable[[], StageDiag]) -> Step:
        """Process local update ``j`` (check if due, block / landing transitions).

        Args:
            j: The local update just done (1-based, consecutive).
            diag_fn: Returns the :class:`StageDiag` of the current candidate; called only when a check
                is due.
        """
        r = self.rule
        step = Step()
        self.local_done = j
        if self.mode == "done":
            raise RuntimeError("update after the freeze")
        diag: Optional[StageDiag] = None
        row: Optional[Dict[str, Any]] = None
        if self.is_check_due(j):
            diag = diag_fn()
            self.n_checks += 1
            if diag.valid and diag.rho_bins.size and np.isfinite(diag.rho_bins).any():
                self.rho_bar = ema_update(self.rho_bar, diag.rho_bins, r.ema_beta)
            eligible = is_eligible(r, diag)
            self.consec = self.consec + 1 if eligible else 0
            bid, btype = self.block_label()
            row = {"local": j, "block_id": bid, "block_type": btype, "mode": self.mode,
                   **diag.scalars(), "eligible": bool(eligible), "consecutive": int(self.consec)}
            step.check = row
            if self.mode == "train" and self.consec >= r.M and self.would_fire is None:
                self.would_fire = j
        if not r.enabled:
            if j >= int(r.fixed_budget):    # type: ignore[arg-type]
                self._close_block(j, "fixed_budget")
                self.mode = "done"
                step.finished = True
                step.events.append("fixed_budget_end")
            return step
        if self.mode == "land":
            assert self.landing is not None
            if j >= self.landing["last_local"]:
                self.mode = "done"
                self.landing["done"] = True
                step.finished = True
                step.events.append("freeze")
            return step
        # ---- training mode
        stop = bool(row is not None and self.consec >= r.M)
        if stop:
            b = self._close_block(j, "development_stop")
            b["fire_local"] = int(j)
            self.fire_local = int(j)
            step.events.append("development_stop")
            self._start_landing(j, forced=False)
        elif j == self.block_end:
            assert diag is not None
            if self.rule.n_nontail_bins > 0:
                cls, S = classify_block_end(diag.R, self.rho_bar, r.rho, r.n_nontail_bins, r.loc_frac)
            else:
                cls, S = "degenerate", np.zeros(0, dtype=int)
            b = self._close_block(j, "block_end")
            b["classification"] = cls
            b["S"] = [int(x) for x in S]
            b["n_S"] = int(S.size)
            b["last_check"] = diag.scalars()
            step.events.append(f"block_end:{cls}")
            if j >= r.u_cap:
                b["exit_reason"] = "cap"
                step.events.append("cap")
                self._start_landing(j, forced=True)
            else:
                nxt = "polish" if cls == "localized" else "global"
                self._open_block(j + 1, nxt, S if cls == "localized" else None)
        return step

    def _start_landing(self, j: int, forced: bool) -> None:
        r = self.rule
        last_block = self.blocks[-1]
        st = self._current_setting()
        self._land_setting = SamplerSetting("landing", st.alpha,
                                            None if st.focus is None else st.focus.copy(),
                                            followed_type=self.block_type)
        self.budget_forced = bool(forced)
        self.landing = {"first_local": j + 1, "last_local": j + r.n_land, "n_land": r.n_land,
                        "followed_block_id": last_block["block_id"],
                        "followed_block_type": self.block_type, "alpha": float(st.alpha),
                        "budget_forced": bool(forced), "done": False}
        self.mode = "land"

    # ------------------------------------------------------------------ record
    def record(self) -> Dict[str, Any]:
        """The rule-log entry of the stage (blocks, landing, fire, budget_forced)."""
        r = self.rule
        return {"stage": r.stage, "enabled": bool(r.enabled), "n_checks": int(self.n_checks),
                "blocks": [dict(b) for b in self.blocks], "landing": self.landing,
                "fire_local": self.fire_local, "would_fire_local": self.would_fire,
                "budget_forced": bool(self.budget_forced),
                "training_updates": (None if self.landing is None else self.landing["first_local"] - 1),
                "total_updates": int(self.local_done),
                "params": {k: getattr(r, k) for k in r.__dataclass_fields__}}
