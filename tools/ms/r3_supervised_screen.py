#!/usr/bin/env python
"""MS-R3 section 2.3 (a): the offline supervised screen of the actor variants at the terminal-stage tie.

No RL, no rollout, no reward. For each actor variant (``agents.ppo_curriculum.ACTOR_VARIANTS`` = t1, relu, t10) the real
``BetaActor`` is built the way the runner builds it (``run/run_ms_stagewise.py: MSRun.__init__``: the torch generator is
seeded ``int(SeedSequence([seed, q, rng_namespaces["init"]]).generate_state(1)[0])`` and the actor is the first draw from
it, as ``CurriculumPPO.__init__`` makes it; the variant is set afterwards, so the three variants of a (q, seed) start from
bit-identical weights). Everything the protocol record supplies (game, PPO hyper-parameters, namespaces, bin width,
recovery step) is read from ``protocols/v2_T2_locked_v2_0.json`` ``records[str(q)]``.

Fit. The Beta-mean effort ``e_hat(d) = e_min + e_range * mu(d)``, ``mu = clamp(sigmoid(z0), mu_clamp, 1 - mu_clamp)``
(the Beta mean), is fitted to the closed-form T = 2 stage-2 equilibrium effort ``e2*(d) = utils.theory_multistage.
g2_two_stage`` (the tent ``e0 max(0, 1 - |d| / (2q))``, ``e0 = DW / (4 k q)``; evaluation-side, it never enters RL
training) by minibatch MSE in effort units. ``z0`` is taken from the output of the actor's ``out`` layer through a forward
hook, so the real forward of every variant is used and the concentration output z1 does not enter the loss: its row of
``out.weight`` and ``out.bias[1]`` get exactly zero gradient and stay at their initial zero. The optimiser is the one of
``CurriculumPPO`` (``torch.optim.Adam(actor.parameters(), lr, betas, eps, weight_decay)`` from the record) over all actor
parameters, with ``torch.nn.utils.clip_grad_norm_`` at the record's ``max_grad_norm``; minibatch = the record's 256; the
network input is ``spec.encode_obs(T, d)`` = (1, d / B) in float32, B = 100 + 2q.

Learning rate (one value per optimiser step ``s = 1 ... steps``, the convention of the PI sandbox)::

    lr(s) = lr_start                                                      for s <= const_steps
    lr(s) = lr_start + (lr_end - lr_start) * (s - const_steps) / decay_steps   for s > const_steps

with ``lr_start`` = the record's ``ppo.lr`` (3e-4), ``lr_end`` = ``LR_END`` (3e-5), ``const_steps = steps - decay_steps``
(56,000 steps: 48,000 + 8,000; the last step has exactly ``lr_end``).

Inputs d. Each step draws a fresh minibatch of 256 gaps on D_2 = [-B, B] from a numpy ``default_rng(SeedSequence([seed, q,
SCREEN_STREAM_ID]))`` (the same stream for the three variants of a (q, seed, starts)) through the repository's own
``StartSampler``: ``bb`` (bin-balanced) = ``StartSampler.balanced`` (a bin uniformly, then a position uniformly in it);
``st`` (stratified) = ``StartSampler.stratified_priority`` with ``stratified_bin_probs(T, lambda_p=0.35,
near_tie_half_width=20, alpha=0)``: the near-tie / middle / tail stratum shares 0.35 / 1 - 0.35 - lambda_T / lambda_T (the
bin-balanced tail share), uniform bins inside a stratum, uniform position inside a bin.

Metrics at each checkpoint (``utils.v2_metrics.recovery_metrics`` on the Beta-mean policy of ``run.run_final_dp_br.
make_policy_fns``, i.e. the verifier's own definitions and recovery grid): ``tip_deficit = e2*(0) - e_hat(0)`` (effort
units; ``e_hat(0)`` = the actor at the input (1, 0)); ``rmse_pos`` (``stage2_rmse_pos``, RMSE of e_hat - e2* over the
grid nodes with |d| < 2q) and ``rmse_pos_over_g2_0``; ``tail_mean`` (``stage2_tail_mean``, mean of e_hat over |d| >= 2q)
and ``tail_mean_over_g2_0``; ``w_eff = tip_deficit / (e2*(0) / (2q))`` (units of d); ``max_abs_w_d`` = max |l1.weight[:, 1]|
in units of d / B (for ``t10`` the stored weight times ``D_FEATURE_SCALE_T10``, the effective weight on d / B);
``train_mse`` = mean minibatch MSE (effort units squared) over the steps since the previous checkpoint.

Commands::

    python tools/ms/r3_supervised_screen.py run --out results/ms_r3/supervised_screen --workers 30
        [--steps 56000] [--extended-steps 224000] [--decay-steps 8000] [--seeds 10501-10510] [--qs 50 60]
        [--actors t1 relu t10] [--starts bb st] [--no-extended] [--resume]
        [--checkpoints 16000 32000 48000] [--extended-checkpoints 16000 32000 48000 56000 112000 168000]
    python tools/ms/r3_supervised_screen.py summarise --out results/ms_r3/supervised_screen
        [--premise-step 56000] [--expected-seeds 10] [--tag TAG]

``run`` writes ``<out>/cells/<actor>_<starts>_q<q>_seed<seed>[_ext].json`` (one per cell: all checkpoint metrics, the
d-stream seed material, the initial-weight SHA-256, wall time, versions) and ``<out>/run_manifest.json`` (``_2``, ``_3``
... on a resume); with ``--workers N > 1`` each cell runs in its own spawned process (``ProcessPoolExecutor``,
``max_tasks_per_child=1``, one torch thread; ``--workers 1`` runs the cells one after another in the calling process).
It never overwrites a file: when a cell file exists it stops (exit 2, nothing written) unless ``--resume`` is given, which
skips the cells whose file exists. The EXTENDED cells (t1, bin-balanced, ``--extended-steps`` steps with checkpoints every
56,000, reported not gated) are part of the plan unless ``--no-extended`` is given or ``t1`` / ``bb`` is not among
``--actors`` / ``--starts``.
``summarise`` writes ``summary_by_cell.csv`` (long form), ``summary_median.csv`` (median, min, max over the seeds per
actor x starts x q x checkpoint), ``summary_extended.csv`` and ``premise_check.json`` (prompt section 2.5 on the main
bin-balanced cells at ``--premise-step``; it carries the per-seed values, the medians, the per-variant verdicts, the
outcome ``PASS`` / ``DROP <variant>`` / ``STOP``, and the flags ``complete`` (every actor x q group has ``--expected-seeds``
seeds) and ``init_identical_across_actors``); it refuses to overwrite (``--tag`` appends a suffix to the output names).
Exit codes: 0 success, 2 stop-and-report (bad input, existing output), 3 a cell failed / the premise check could not be
computed (a required actor x q group has no bin-balanced cell at the premise step).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.ppo_curriculum import ACTOR_VARIANTS, D_FEATURE_SCALE_T10, BetaActor, PPOConfig  # noqa: E402
from envs.curriculum_env import GameSpec, StartSampler  # noqa: E402
from run.run_final_dp_br import make_policy_fns  # noqa: E402
from run.run_final_dp_br_round3_dense import strict_dataclass  # noqa: E402
from utils.theory_multistage import g2_two_stage  # noqa: E402
from utils.v2_metrics import recovery_metrics  # noqa: E402

torch.set_num_threads(1)

# --------------------------------------------------------------------------------------- constants
TOOL = "tools/ms/r3_supervised_screen.py"
CELL_SCHEMA = "r3_supervised_screen_cell/1"
PROTOCOL_PATH = REPO / "protocols" / "v2_T2_locked_v2_0.json"
#: Namespace of the d stream, ``SeedSequence([seed, q, SCREEN_STREAM_ID])``. Outside the runner's rng namespaces
#: (0-5, ``records[q].protocol.rng_namespaces``), so the stream is never one of a run's streams.
SCREEN_STREAM_ID = 30001
#: Stratified starts = the MS-R2 ``NL_st`` shares (``tools/ms/ms_configs.py``: ``R2_ARM_TABLE["NL_st_*"]["lambda_P"]``,
#: ``DEFAULT_PARAMS["near_tie_half_width"]``); alpha 0 because a supervised fit has no verifier to focus on.
STRAT_LAMBDA_P = 0.35
STRAT_NEAR_HALF_WIDTH = 20.0
STRAT_ALPHA = 0.0
#: End of the terminal-stage LR decay (``tools/ms/ms_configs.py: LEGACY_PIPELINE_NL`` lr window end; MS-R3 D3).
LR_END = 3e-5
STEPS_MAIN = 56000
STEPS_EXTENDED = 224000
DECAY_STEPS = 8000
CHECKPOINTS_MAIN = (16000, 32000, 48000, 56000)
CHECKPOINTS_EXTENDED = (16000, 32000, 48000, 56000, 112000, 168000, 224000)
SEEDS = tuple(range(10501, 10511))
QS = (50, 60)
STARTS = ("bb", "st")
STARTS_NAME = {"bb": "bin_balanced", "st": "stratified"}
#: Per-checkpoint metrics (the order of the CSV columns).
METRICS = ("tip_deficit", "rmse_pos", "tail_mean", "w_eff", "max_abs_w_d", "e_hat_0", "tip_deficit_over_g2_0",
           "rmse_pos_over_g2_0", "tail_mean_over_g2_0", "train_mse")
#: Premise check (prompt section 2.5): thresholds.
PREMISE_STEP = 56000
PREMISE_MIN_T1_DEFICIT = 1.0      # (i)  median tip deficit of t1 >= this at both q
PREMISE_MAX_RATIO = 0.5           # (ii) median deficit of v <= this x the median deficit of t1 at both q
PREMISE_VARIANTS = ("relu", "t10")


# --------------------------------------------------------------------------------------- small helpers
def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    """HEAD commit of the repository (``unknown`` if git fails)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def write_new_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, which must not exist: never overwrites (``FileExistsError``).

    The text goes to a temporary file in the same directory first and is then hard-linked to ``path`` (``os.link`` fails
    if ``path`` exists), so a file that exists is a complete file.

    Args:
        path: Target file.
        text: Content.

    Raises:
        FileExistsError: If ``path`` exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "x") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def write_new_json(path: Path, obj: Any) -> None:
    """:func:`write_new_text` of the JSON text of ``obj`` (indent 1, trailing newline)."""
    write_new_text(path, json.dumps(obj, indent=1) + "\n")


def csv_text(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> str:
    """CSV text of ``rows`` with the columns ``fields`` (floats in ``repr`` precision)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(fields), lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in fields})
    return buf.getvalue()


def load_protocol(path: Path = PROTOCOL_PATH) -> Dict[str, Any]:
    """The locked protocol JSON (``records[str(q)]`` carries the game / PPO / protocol settings)."""
    with open(path) as f:
        return json.load(f)


def parse_seeds(tokens: Sequence[str]) -> List[int]:
    """Seeds from tokens like ``10501-10510`` (inclusive range) or ``10501``."""
    out: List[int] = []
    for tok in tokens:
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    return out


# --------------------------------------------------------------------------------------- schedule and cells
def lr_at_step(step: int, lr_start: float, lr_end: float, const_steps: int, decay_steps: int) -> float:
    """Learning rate of optimiser step ``step`` (1-indexed): constant, then linear to ``lr_end`` at the last step.

    Args:
        step: Step number, 1 ... const_steps + decay_steps.
        lr_start: Constant LR (3e-4).
        lr_end: LR of the last step (3e-5).
        const_steps: Number of constant-LR steps.
        decay_steps: Length of the linear decay.

    Returns:
        ``lr_start`` for ``step <= const_steps``, else ``lr_start + (lr_end - lr_start) (step - const_steps) /
        decay_steps``.
    """
    if step <= const_steps:
        return lr_start
    return lr_start + (lr_end - lr_start) * (step - const_steps) / decay_steps


@dataclass(frozen=True)
class CellSpec:
    """One cell of the screen: (actor, starts, q, seed) with its step budget and checkpoints."""

    actor: str
    starts: str
    q: int
    seed: int
    steps: int
    decay_steps: int
    checkpoints: Tuple[int, ...]
    extended: bool = False

    def __post_init__(self) -> None:
        if self.actor not in ACTOR_VARIANTS:
            raise ValueError(f"unknown actor {self.actor!r}; known: {ACTOR_VARIANTS}")
        if self.starts not in STARTS:
            raise ValueError(f"unknown starts {self.starts!r}; known: {STARTS}")
        if not 0 < self.decay_steps < self.steps:
            raise ValueError(f"need 0 < decay_steps < steps; got {self.decay_steps}, {self.steps}")
        if self.steps not in self.checkpoints:
            raise ValueError("the last step must be a checkpoint")

    @property
    def name(self) -> str:
        """``<actor>_<starts>_q<q>_seed<seed>`` plus ``_ext`` for an extended cell."""
        return f"{self.actor}_{self.starts}_q{self.q}_seed{self.seed}" + ("_ext" if self.extended else "")

    @property
    def filename(self) -> str:
        """The cell's JSON file name."""
        return self.name + ".json"

    @property
    def const_steps(self) -> int:
        """Number of constant-LR steps."""
        return self.steps - self.decay_steps


def norm_checkpoints(cps: Sequence[int], steps: int) -> Tuple[int, ...]:
    """Sorted unique checkpoints within ``(0, steps]``, always including ``steps``."""
    return tuple(sorted({int(c) for c in cps if 0 < int(c) <= steps} | {int(steps)}))


def plan_cells(actors: Sequence[str], starts: Sequence[str], qs: Sequence[int], seeds: Sequence[int],
               steps: int = STEPS_MAIN, decay_steps: int = DECAY_STEPS,
               checkpoints: Sequence[int] = CHECKPOINTS_MAIN, extended_steps: int = STEPS_EXTENDED,
               extended_checkpoints: Sequence[int] = CHECKPOINTS_EXTENDED, with_extended: bool = True
               ) -> List[CellSpec]:
    """The cells of a screen, extended cells first (they are four times longer), then the main grid.

    Args:
        actors: Actor variants.
        starts: Start distributions (``bb``, ``st``).
        qs: Noise half-widths.
        seeds: Seeds.
        steps: Optimiser steps of a main cell.
        decay_steps: Length of the final linear LR decay (main and extended cells).
        checkpoints: Reported steps of a main cell (``steps`` is always added).
        extended_steps: Optimiser steps of an extended cell.
        extended_checkpoints: Reported steps of an extended cell (``extended_steps`` is always added).
        with_extended: Whether to plan the extended cells (only when ``t1`` and ``bb`` are requested).

    Returns:
        The planned cells.
    """
    cells: List[CellSpec] = []
    if with_extended and "t1" in actors and "bb" in starts:
        for q in qs:
            for seed in seeds:
                cells.append(CellSpec("t1", "bb", int(q), int(seed), int(extended_steps), int(decay_steps),
                                      norm_checkpoints(extended_checkpoints, extended_steps), True))
    for q in qs:
        for seed in seeds:
            for actor in actors:
                for st in starts:
                    cells.append(CellSpec(actor, st, int(q), int(seed), int(steps), int(decay_steps),
                                          norm_checkpoints(checkpoints, steps), False))
    return cells


# --------------------------------------------------------------------------------------- init, d stream, evaluation
def actor_digest(actor: BetaActor) -> str:
    """SHA-256 of the actor's weights (sorted names ``actor.<key>``, float32 bytes): the recipe of
    ``MSRun._init_digest`` restricted to the actor (the critic is not built here)."""
    h = hashlib.sha256()
    sd = actor.state_dict()
    for k in sorted(sd):
        h.update(f"actor.{k}".encode())
        h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def build_initial_actor(cfg: PPOConfig, seed: int, q: int, namespaces: Dict[str, int], variant: str
                        ) -> Tuple[BetaActor, Dict[str, Any]]:
    """The runner's initial actor for (q, seed), with the variant set afterwards.

    Args:
        cfg: ``PPOConfig`` of the record.
        seed: Run seed.
        q: Noise half-width.
        namespaces: ``records[q].protocol.rng_namespaces``.
        variant: One of ``ACTOR_VARIANTS``.

    Returns:
        ``(actor, info)``; ``info`` carries the seed material, the torch seed and the initial-weight SHA-256.
    """
    material = [int(seed), int(q), int(namespaces["init"])]
    torch_seed = int(np.random.SeedSequence(material).generate_state(1)[0])
    gen = torch.Generator().manual_seed(torch_seed)
    actor = BetaActor(cfg.hidden, cfg.c_min, cfg.mu_clamp, gen)       # the first draw, as CurriculumPPO.__init__
    actor.variant = variant
    info = {"namespace": int(namespaces["init"]), "seed_material": material, "torch_seed": torch_seed,
            "actor_sha256": actor_digest(actor)}
    return actor, info


def make_start_draw(starts: str, spec: GameSpec, bin_width: float, seed: int, q: int
                    ) -> Tuple[Callable[[int], np.ndarray], Dict[str, Any]]:
    """The d stream of a cell: ``draw(n)`` returns n gaps on D_T from the repository's ``StartSampler``.

    Args:
        starts: ``bb`` (bin-balanced) or ``st`` (stratified, alpha 0).
        spec: Game parameters.
        bin_width: ES bin width (record ``protocol.es_bin_width``).
        seed: Run seed.
        q: Noise half-width.

    Returns:
        ``(draw, info)``; ``info`` records the seed material, the sampler, the bin probabilities and stratum shares.
    """
    sampler = StartSampler(spec, bin_width)
    t = int(spec.T)
    material = [int(seed), int(q), SCREEN_STREAM_ID]
    rng = np.random.default_rng(np.random.SeedSequence(material))
    n_bins = sampler.n_bins(t)
    if starts == "bb":
        probs = np.full(n_bins, 1.0 / n_bins)

        def draw(n: int) -> np.ndarray:
            return sampler.balanced(t, n, rng)
        how = "StartSampler.balanced"
    elif starts == "st":
        probs = sampler.stratified_bin_probs(t, STRAT_LAMBDA_P, STRAT_NEAR_HALF_WIDTH, STRAT_ALPHA)

        def draw(n: int) -> np.ndarray:
            return sampler.stratified_priority(t, n, rng, probs)
        how = (f"StartSampler.stratified_priority(stratified_bin_probs(T, lambda_p={STRAT_LAMBDA_P}, "
               f"near_tie_half_width={STRAT_NEAR_HALF_WIDTH}, alpha={STRAT_ALPHA}))")
    else:
        raise ValueError(f"unknown starts {starts!r}")
    strata = sampler.strata(t, STRAT_NEAR_HALF_WIDTH)
    info = {"stream_id": SCREEN_STREAM_ID, "seed_material": material, "starts": starts,
            "starts_name": STARTS_NAME[starts], "sampler": how, "bin_width": float(bin_width), "n_bins": int(n_bins),
            "domain_half": float(spec.domain_half(t)),
            "stratum_shares": {k: float(probs[m].sum()) for k, m in strata.items()},
            "bin_probs": [float(p) for p in probs]}
    return draw, info


class _ActorShim:
    """Exposes ``beta_params(obs, net)`` of a bare ``BetaActor`` as ``run.run_final_dp_br.make_policy_fns`` needs it
    (the float32 torch forward of training; the body of ``CurriculumPPO.beta_params`` on CPU)."""

    def __init__(self, actor: BetaActor):
        self.actor = actor

    @torch.no_grad()
    def beta_params(self, obs: np.ndarray, net: Optional[BetaActor] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(alpha, beta) float32 arrays of the actor on a float32 observation batch."""
        net = self.actor if net is None else net
        a, b = net(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
        return a.cpu().numpy(), b.cpu().numpy()


def max_abs_d_weight(actor: BetaActor) -> float:
    """max |first-layer weight on the d input| in units of d / B (``t10``: stored weight x ``D_FEATURE_SCALE_T10``)."""
    w = actor.l1.weight.detach().cpu().numpy()[:, 1].astype(np.float64)
    scale = D_FEATURE_SCALE_T10 if actor.variant == "t10" else 1.0
    return float(np.max(np.abs(w)) * scale)


def evaluate_fit(actor: BetaActor, spec: GameSpec, recovery_step: float) -> Dict[str, float]:
    """The verifier-side metrics of the actor's Beta-mean effort against the closed-form tent.

    Args:
        actor: The (fitted) actor.
        spec: Game parameters (T = 2).
        recovery_step: Recovery grid spacing (record ``protocol.recovery_step``).

    Returns:
        ``e_hat_0``, ``g2_at_0``, ``tip_deficit``, ``tip_deficit_over_g2_0``, ``rmse_pos``, ``rmse_pos_over_g2_0``,
        ``tail_mean``, ``tail_mean_over_g2_0``, ``w_eff``, ``max_abs_w_d``.
    """
    mean_fn, _ = make_policy_fns(_ActorShim(actor), spec)
    sc, _ = recovery_metrics(mean_fn, spec, recovery_step)
    g20, e0 = float(sc["g2_at_0"]), float(sc["e2_at_0"])
    deficit = g20 - e0
    return {"e_hat_0": e0, "g2_at_0": g20, "tip_deficit": deficit, "tip_deficit_over_g2_0": deficit / g20,
            "rmse_pos": float(sc["stage2_rmse_pos"]), "rmse_pos_over_g2_0": float(sc["stage2_rmse_pos_over_g2_0"]),
            "tail_mean": float(sc["stage2_tail_mean"]), "tail_mean_over_g2_0": float(sc["stage2_tail_mean_over_g2_0"]),
            "w_eff": deficit / (g20 / (2.0 * float(spec.q))), "max_abs_w_d": max_abs_d_weight(actor)}


# --------------------------------------------------------------------------------------- the fit
def fit_actor(actor: BetaActor, opt: torch.optim.Optimizer, spec: GameSpec, cfg: PPOConfig,
              draw: Callable[[int], np.ndarray], cell: CellSpec, lr_start: float, lr_end: float,
              recovery_step: float) -> List[Dict[str, Any]]:
    """Fit the Beta-mean effort to the closed-form tent; evaluate at the cell's checkpoints.

    Args:
        actor: The initial actor (modified in place).
        opt: Its Adam optimiser.
        spec: Game parameters (T = 2).
        cfg: ``PPOConfig`` of the record (minibatch, gradient-norm clip).
        draw: ``draw(n)`` -> n gaps (the d stream).
        cell: The cell (budget, checkpoints).
        lr_start: Constant LR.
        lr_end: Final LR.
        recovery_step: Recovery grid spacing for the metrics.

    Returns:
        One dict per checkpoint: ``step``, ``lr`` and the metrics of :func:`evaluate_fit` plus ``train_mse``.
    """
    wanted = set(cell.checkpoints)
    out: List[Dict[str, Any]] = []
    captured: List[torch.Tensor] = []
    handle = actor.out.register_forward_hook(lambda _m, _i, o: captured.append(o))
    mse_sum, mse_n = 0.0, 0
    try:
        for step in range(1, cell.steps + 1):
            lr = lr_at_step(step, lr_start, lr_end, cell.const_steps, cell.decay_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            d = draw(cfg.minibatch)
            x = torch.from_numpy(spec.encode_obs(spec.T, d))
            target = torch.from_numpy(
                g2_two_stage(d, spec.q, spec.w_h, spec.w_l, spec.k, spec.e_max).astype(np.float32))
            captured.clear()
            actor(x)                                                  # the real forward of the variant
            mu = torch.clamp(torch.sigmoid(captured[0][:, 0]), actor.mu_clamp, 1.0 - actor.mu_clamp)
            loss = ((spec.e_min + spec.e_range * mu - target) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            opt.step()
            mse_sum += float(loss.item())
            mse_n += 1
            if step in wanted:
                out.append({"step": step, "lr": lr, **evaluate_fit(actor, spec, recovery_step),
                            "train_mse": mse_sum / mse_n})
                mse_sum, mse_n = 0.0, 0
    finally:
        handle.remove()
    return out


def fit_cell(cell: CellSpec, protocol_path: Path = PROTOCOL_PATH) -> Dict[str, Any]:
    """Run one cell and return its JSON-able record (:func:`fit_cell_actor` without the actor)."""
    return fit_cell_actor(cell, protocol_path)[1]


def fit_cell_actor(cell: CellSpec, protocol_path: Path = PROTOCOL_PATH) -> Tuple[BetaActor, Dict[str, Any]]:
    """Run one cell.

    Args:
        cell: The cell.
        protocol_path: The locked protocol file.

    Returns:
        ``(fitted_actor, record)``; the record has the schema ``r3_supervised_screen_cell/1``.
    """
    t_wall, t_cpu = time.perf_counter(), time.process_time()
    proto = load_protocol(protocol_path)
    rec = proto["records"][str(cell.q)]
    spec = strict_dataclass(GameSpec, rec["game"], "game")
    cfg = strict_dataclass(PPOConfig, rec["ppo"], "ppo")
    P = rec["protocol"]
    actor, init_info = build_initial_actor(cfg, cell.seed, cell.q, P["rng_namespaces"], cell.actor)
    draw, stream_info = make_start_draw(cell.starts, spec, float(P["es_bin_width"]), cell.seed, cell.q)
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr, betas=cfg.adam_betas, eps=cfg.adam_eps,
                           weight_decay=cfg.weight_decay)
    cps = fit_actor(actor, opt, spec, cfg, draw, cell, float(cfg.lr), LR_END, float(P["recovery_step"]))
    versions = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}
    return actor, {
        "schema": CELL_SCHEMA, "tool": TOOL,
        "cell": {"name": cell.name, "actor": cell.actor, "starts": cell.starts, "q": cell.q, "seed": cell.seed,
                 "extended": cell.extended},
        "budget": {"steps": cell.steps, "const_steps": cell.const_steps, "decay_steps": cell.decay_steps,
                   "checkpoints": list(cell.checkpoints), "minibatch": int(cfg.minibatch),
                   "max_grad_norm": float(cfg.max_grad_norm), "lr_start": float(cfg.lr), "lr_end": LR_END,
                   "lr_formula": "lr(s) = lr_start for s <= const_steps else lr_start + (lr_end - lr_start) * "
                                 "(s - const_steps) / decay_steps, s = 1 ... steps",
                   "optimizer": {"name": "torch.optim.Adam", "betas": list(cfg.adam_betas), "eps": cfg.adam_eps,
                                 "weight_decay": cfg.weight_decay}},
        "game": rec["game"], "recovery_step": float(P["recovery_step"]),
        "target": "utils.theory_multistage.g2_two_stage(d, q, w_h, w_l, k, e_max) (evaluation-side tent)",
        "g2_at_0": cps[0]["g2_at_0"],
        "d_stream": stream_info, "init": init_info,
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "checkpoints": cps,
        "wall_sec": time.perf_counter() - t_wall, "process_cpu_sec": time.process_time() - t_cpu,
        "pid": os.getpid(), "versions": versions, "versions_match_record": versions == rec["versions"],
    }


def run_cell_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Process entry point: run the cell of ``job`` and write its JSON (never overwriting).

    Args:
        job: ``{"cell": asdict(CellSpec), "protocol_path": str, "out_dir": str}``.

    Returns:
        ``{"name", "path", "wall_sec", "final"}`` (the last checkpoint).
    """
    d = dict(job["cell"])
    d["checkpoints"] = tuple(d["checkpoints"])
    cell = CellSpec(**d)
    record = fit_cell(cell, Path(job["protocol_path"]))
    path = Path(job["out_dir"]) / "cells" / cell.filename
    write_new_json(path, record)
    return {"name": cell.name, "path": str(path), "wall_sec": record["wall_sec"], "final": record["checkpoints"][-1]}


# --------------------------------------------------------------------------------------- run command
def cmd_run(args: argparse.Namespace) -> int:
    """``run``: plan the cells, refuse to overwrite, run them in spawned single-thread processes."""
    out = Path(args.out)
    protocol = Path(args.protocol)
    if not protocol.is_file():
        print(f"STOP: protocol file {protocol} not found")
        return 2
    proto = load_protocol(protocol)
    qs = [int(q) for q in args.qs]
    bad = [q for q in qs if q not in proto["q_values"]]
    if bad:
        print(f"STOP: q {bad} not in the protocol's q_values {proto['q_values']}")
        return 2
    cps = CHECKPOINTS_MAIN if args.checkpoints is None else args.checkpoints
    cps_ext = CHECKPOINTS_EXTENDED if args.extended_checkpoints is None else args.extended_checkpoints
    try:
        cells = plan_cells(args.actors, args.starts, qs, parse_seeds(args.seeds), args.steps, args.decay_steps, cps,
                           args.extended_steps, cps_ext, not args.no_extended)
    except ValueError as exc:
        print(f"STOP: {exc}")
        return 2
    if not cells:
        print("STOP: no cells planned")
        return 2
    existing = [c for c in cells if (out / "cells" / c.filename).exists()]
    if existing and not args.resume:
        print(f"STOP: {len(existing)} of {len(cells)} cell file(s) already exist under {out / 'cells'} (first: "
              f"{existing[0].filename}); nothing was written. Pass --resume to skip the existing cells.")
        return 2
    todo = [c for c in cells if c not in existing]
    if existing:
        print(f"resume: skipping {len(existing)} existing cell file(s), {len(todo)} to run")
    if not todo:
        print("nothing to run: every planned cell file exists")
        return 0
    workers = max(1, min(int(args.workers), len(todo)))
    manifest = {
        "tool": TOOL, "tool_sha256": sha256_file(Path(__file__)), "command": args.command_line,
        "git_head": git_head(), "protocol": {"path": str(protocol), "sha256": sha256_file(protocol)},
        "constants": {"SCREEN_STREAM_ID": SCREEN_STREAM_ID, "STRAT_LAMBDA_P": STRAT_LAMBDA_P,
                      "STRAT_NEAR_HALF_WIDTH": STRAT_NEAR_HALF_WIDTH, "STRAT_ALPHA": STRAT_ALPHA, "LR_END": LR_END},
        "plan": {"n_cells": len(cells), "n_to_run": len(todo), "n_skipped_existing": len(existing),
                 "cells": [c.name for c in todo]},
        "workers": workers, "nproc": os.cpu_count(), "loadavg_at_start": list(os.getloadavg()),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    mpath = out / "run_manifest.json"
    n = 1
    while mpath.exists():
        n += 1
        mpath = out / f"run_manifest_{n}.json"
    write_new_json(mpath, manifest)
    jobs = [{"cell": asdict(c), "protocol_path": str(protocol), "out_dir": str(out)} for c in todo]
    print(f"{len(todo)} cell(s), {workers} worker(s); manifest {mpath}", flush=True)
    t0 = time.perf_counter()
    failed: List[str] = []
    n_done = 0

    def report(name: str, res: Optional[Dict[str, Any]], err: Optional[str]) -> None:
        nonlocal n_done
        n_done += 1
        if err is not None:
            failed.append(name)
            print(f"[{n_done}/{len(todo)}] FAILED {name}: {err}", flush=True)
        else:
            f = res["final"]  # type: ignore[index]
            print(f"[{n_done}/{len(todo)}] {name}  step {f['step']}  deficit {f['tip_deficit']:.4f}  "
                  f"rmse_pos {f['rmse_pos']:.4f}  wall {res['wall_sec']:.1f}s", flush=True)  # type: ignore[index]

    if workers == 1:
        for c, job in zip(todo, jobs):
            try:
                report(c.name, run_cell_job(job), None)
            except Exception as exc:  # noqa: BLE001 - report and go on with the other cells
                report(c.name, None, f"{type(exc).__name__}: {exc}")
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 max_tasks_per_child=1) as ex:
            futs = {ex.submit(run_cell_job, job): c.name for c, job in zip(todo, jobs)}
            for fut in as_completed(futs):
                try:
                    report(futs[fut], fut.result(), None)
                except Exception as exc:  # noqa: BLE001
                    report(futs[fut], None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
    print(f"done: {len(todo) - len(failed)} ok, {len(failed)} failed, wall {time.perf_counter() - t0:.1f}s")
    if failed:
        print("FAILED cells:", ", ".join(failed))
        return 3
    return 0


# --------------------------------------------------------------------------------------- summaries
def load_cells(out: Path) -> List[Dict[str, Any]]:
    """All cell records under ``<out>/cells`` (sorted by file name)."""
    cells = []
    for f in sorted((out / "cells").glob("*.json")):
        with open(f) as fh:
            c = json.load(fh)
        if c.get("schema") != CELL_SCHEMA:
            raise ValueError(f"{f}: schema {c.get('schema')!r} is not {CELL_SCHEMA!r}")
        cells.append(c)
    return cells


ROW_COLS = ["actor", "starts", "q", "seed", "extended", "cell_steps", "steps", "lr", *METRICS, "init_actor_sha256"]


def flatten(cells: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Long-form rows (one per cell x checkpoint)."""
    rows: List[Dict[str, Any]] = []
    for c in cells:
        for cp in c["checkpoints"]:
            rows.append({"actor": c["cell"]["actor"], "starts": c["cell"]["starts"], "q": int(c["cell"]["q"]),
                         "seed": int(c["cell"]["seed"]), "extended": int(bool(c["cell"]["extended"])),
                         "cell_steps": int(c["budget"]["steps"]), "steps": int(cp["step"]), "lr": cp["lr"],
                         **{m: float(cp[m]) for m in METRICS}, "init_actor_sha256": c["init"]["actor_sha256"]})
    return rows


def _sort_key(key: Tuple[str, str, int, int]) -> Tuple[int, int, int, int]:
    actor, starts, q, steps = key
    return (ACTOR_VARIANTS.index(actor), STARTS.index(starts), q, steps)


def median_table(rows: Sequence[Dict[str, Any]], extended: bool) -> List[Dict[str, Any]]:
    """Median, min and max over the seeds per actor x starts x q x checkpoint (``numpy.median``: the mean of the two
    middle values for an even number of seeds)."""
    groups: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = {}
    for r in rows:
        if bool(r["extended"]) == extended:
            groups.setdefault((r["actor"], r["starts"], r["q"], r["steps"]), []).append(r)
    table: List[Dict[str, Any]] = []
    for key in sorted(groups, key=_sort_key):
        g = groups[key]
        row: Dict[str, Any] = {"actor": key[0], "starts": key[1], "q": key[2], "steps": key[3], "n": len(g)}
        for m in METRICS:
            v = np.array([x[m] for x in g], dtype=float)
            row[f"{m}_median"], row[f"{m}_min"], row[f"{m}_max"] = float(np.median(v)), float(v.min()), float(v.max())
        table.append(row)
    return table


MEDIAN_COLS = ["actor", "starts", "q", "steps", "n"] + [f"{m}_{s}" for m in METRICS for s in ("median", "min", "max")]


class PremiseError(Exception):
    """The premise check cannot be computed (a required group has no cells)."""


def premise_check(rows: Sequence[Dict[str, Any]], step: int = PREMISE_STEP, expected_seeds: int = len(SEEDS),
                  qs: Sequence[int] = QS) -> Dict[str, Any]:
    """Prompt section 2.5 on the bin-balanced main cells at ``step``.

    (i) the median tip deficit of ``t1`` is >= 1.0 effort unit at both q; (ii) for each variant v in {relu, t10} the
    median tip deficit of v is <= 0.5 x that of ``t1`` at both q. Outcome: ``STOP`` if (i) fails or both variants fail
    (ii); ``DROP <variant>`` if exactly one fails (ii); ``PASS`` if both pass.

    Args:
        rows: Long-form rows (:func:`flatten`).
        step: The checkpoint step (56,000).
        expected_seeds: Seeds per group, for the ``complete`` flag only.
        qs: The q values (both must have cells for every actor).

    Returns:
        The ``premise_check.json`` document.

    Raises:
        PremiseError: If an actor x q group has no bin-balanced cell at ``step``.
    """
    groups: Dict[Tuple[str, int], Dict[int, Tuple[float, str]]] = {}
    for r in rows:
        if not r["extended"] and r["starts"] == "bb" and r["steps"] == step:
            groups.setdefault((r["actor"], r["q"]), {})[r["seed"]] = (float(r["tip_deficit"]),
                                                                       r["init_actor_sha256"])
    missing = [f"{a} q{q}" for a in ACTOR_VARIANTS for q in qs if (a, q) not in groups]
    if missing:
        raise PremiseError(f"no bin-balanced cell at step {step} for: {', '.join(missing)}")
    per_seed = {a: {str(q): {str(s): groups[(a, q)][s][0] for s in sorted(groups[(a, q)])} for q in qs}
                for a in ACTOR_VARIANTS}
    n_seeds = {a: {str(q): len(groups[(a, q)]) for q in qs} for a in ACTOR_VARIANTS}
    med = {a: {str(q): float(np.median(list(per_seed[a][str(q)].values()))) for q in qs} for a in ACTOR_VARIANTS}
    cond_i = {str(q): {"median_t1": med["t1"][str(q)], "threshold": PREMISE_MIN_T1_DEFICIT,
                       "pass": bool(med["t1"][str(q)] >= PREMISE_MIN_T1_DEFICIT)} for q in qs}
    pass_i = all(v["pass"] for v in cond_i.values())
    cond_ii: Dict[str, Any] = {}
    for v in PREMISE_VARIANTS:
        by_q = {}
        for q in qs:
            m_v, m_t1 = med[v][str(q)], med["t1"][str(q)]
            by_q[str(q)] = {"median_variant": m_v, "median_t1": m_t1, "limit": PREMISE_MAX_RATIO * m_t1,
                            "ratio": (m_v / m_t1) if m_t1 != 0.0 else float("nan"),
                            "pass": bool(m_v <= PREMISE_MAX_RATIO * m_t1)}
        cond_ii[v] = {"by_q": by_q, "pass": all(x["pass"] for x in by_q.values())}
    failing = [v for v in PREMISE_VARIANTS if not cond_ii[v]["pass"]]
    if not pass_i:
        outcome, why = "STOP", "(i) fails: the median tip deficit of t1 is below the threshold at some q"
    elif len(failing) == len(PREMISE_VARIANTS):
        outcome, why = "STOP", "both variants fail (ii)"
    elif len(failing) == 1:
        outcome, why = f"DROP {failing[0]}", f"{failing[0]} fails (ii), the other variant passes"
    else:
        outcome, why = "PASS", "(i) holds and both variants pass (ii)"
    shas: Dict[Tuple[int, int], set] = {}
    for (a, q), g in groups.items():
        for s, (_, sha) in g.items():
            shas.setdefault((q, s), set()).add(sha)
    return {
        "section": "prompt MS-R3 2.5 (premise check)", "step": step, "starts": "bb", "extended_cells_used": False,
        "rule": {"i": f"median tip deficit of t1 >= {PREMISE_MIN_T1_DEFICIT} effort unit at both q",
                 "ii": f"for v in {list(PREMISE_VARIANTS)}: median tip deficit of v <= {PREMISE_MAX_RATIO} x the "
                       "median of t1 at both q",
                 "median": "numpy.median over the seeds (the mean of the two middle values for an even number)",
                 "tip_deficit": "e2*(0) - e_hat(0), effort units, signed"},
        "n_seeds": n_seeds, "expected_seeds": expected_seeds,
        "complete": all(n == expected_seeds for a in n_seeds.values() for n in a.values()),
        "median_tip_deficit": med, "per_seed_tip_deficit": per_seed,
        "init_identical_across_actors": all(len(s) == 1 for s in shas.values()),
        "condition_i": {"by_q": cond_i, "pass": pass_i}, "condition_ii": cond_ii,
        "variants_failing_ii": failing, "outcome": outcome, "outcome_reason": why}


def cmd_summarise(args: argparse.Namespace) -> int:
    """``summarise``: the long-form CSV, the median tables and the premise check (never overwriting)."""
    out = Path(args.out)
    if not (out / "cells").is_dir():
        print(f"STOP: no cells directory under {out}")
        return 2
    cells = load_cells(out)
    if not cells:
        print(f"STOP: no cell files under {out / 'cells'}")
        return 2
    rows = flatten(cells)
    tag = f"_{args.tag}" if args.tag else ""
    texts: Dict[str, str] = {f"summary_by_cell{tag}.csv": csv_text(rows, ROW_COLS),
                             f"summary_median{tag}.csv": csv_text(median_table(rows, False), MEDIAN_COLS)}
    ext_table = median_table(rows, True)
    if ext_table:
        texts[f"summary_extended{tag}.csv"] = csv_text(ext_table, MEDIAN_COLS)
    premise: Optional[Dict[str, Any]] = None
    premise_err = ""
    try:
        premise = premise_check(rows, args.premise_step, args.expected_seeds)
        premise["source"] = {"out": str(out), "n_cells": len(cells), "tool": TOOL}
        texts[f"premise_check{tag}.json"] = json.dumps(premise, indent=1) + "\n"
    except PremiseError as exc:
        premise_err = str(exc)
    clash = [n for n in texts if (out / n).exists()]
    if clash:
        print(f"STOP: output file(s) already exist under {out}: {', '.join(clash)}; nothing was written "
              "(use --tag to write next to them)")
        return 2
    for n, t in texts.items():
        write_new_text(out / n, t)
    print(f"{len(cells)} cell(s), {len(rows)} row(s); wrote {', '.join(texts)}")
    for r in ext_table:                       # the extended cell at a glance (reported, not gated)
        if r["actor"] == "t1" and r["starts"] == "bb":
            print(f"  extended t1 bb q{r['q']} step {r['steps']:>6d}: median tip deficit "
                  f"{r['tip_deficit_median']:.4f} [{r['tip_deficit_min']:.4f}, {r['tip_deficit_max']:.4f}] "
                  f"rmse_pos {r['rmse_pos_median']:.4f}  (n {r['n']})")
    if premise is None:
        print(f"premise check NOT computed: {premise_err}")
        return 3
    if not premise["complete"]:
        print(f"WARNING: not every actor x q group has {args.expected_seeds} seeds: {premise['n_seeds']}")
    print(f"premise check at step {args.premise_step} (bin-balanced): outcome {premise['outcome']} "
          f"({premise['outcome_reason']})")
    for a in ACTOR_VARIANTS:
        print(f"  {a:5s} median tip deficit  " + "  ".join(f"q{q}: {premise['median_tip_deficit'][a][q]:.4f}"
                                                           for q in premise["median_tip_deficit"][a]))
    return 0


# --------------------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the tool."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the cells of the screen (one process per cell)")
    r.add_argument("--out", required=True, help="output root (cells/ and run_manifest.json are written below it)")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--steps", type=int, default=STEPS_MAIN)
    r.add_argument("--extended-steps", type=int, default=STEPS_EXTENDED)
    r.add_argument("--decay-steps", type=int, default=DECAY_STEPS)
    r.add_argument("--seeds", nargs="+", default=[f"{SEEDS[0]}-{SEEDS[-1]}"])
    r.add_argument("--qs", nargs="+", type=int, default=list(QS))
    r.add_argument("--actors", nargs="+", default=list(ACTOR_VARIANTS), choices=list(ACTOR_VARIANTS))
    r.add_argument("--starts", nargs="+", default=list(STARTS), choices=list(STARTS))
    r.add_argument("--no-extended", action="store_true")
    r.add_argument("--resume", action="store_true", help="skip the cells whose file exists")
    r.add_argument("--checkpoints", nargs="+", type=int, default=None,
                   help="reported steps of a main cell (default 16000 32000 48000 56000; --steps is always added)")
    r.add_argument("--extended-checkpoints", nargs="+", type=int, default=None,
                   help="reported steps of an extended cell (default 16000 32000 48000 56000 112000 168000 224000)")
    r.add_argument("--protocol", default=str(PROTOCOL_PATH))
    s = sub.add_parser("summarise", help="long-form CSV, median tables and the premise check")
    s.add_argument("--out", required=True)
    s.add_argument("--premise-step", type=int, default=PREMISE_STEP)
    s.add_argument("--expected-seeds", type=int, default=len(SEEDS))
    s.add_argument("--tag", default="", help="suffix of the output names (to write next to existing outputs)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the exit code (0 ok, 2 stop-and-report, 3 failed cell / premise check not computed)."""
    args = build_parser().parse_args(argv)
    args.command_line = " ".join(["python", TOOL, *(sys.argv[1:] if argv is None else [str(a) for a in argv])])
    return cmd_run(args) if args.cmd == "run" else cmd_summarise(args)


if __name__ == "__main__":
    sys.exit(main())
