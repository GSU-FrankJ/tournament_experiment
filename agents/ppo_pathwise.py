"""Pathwise (exact-gradient) terminal fine-tuning step: method 5 of the R1 refinement round.

Both players execute the Beta mean, so the executed action carries no sampling noise and there is
no likelihood-ratio gradient. The refinement phase (runner mode ``phase_P``) is therefore
exact-gradient ascent on the conditional expected terminal payoff of the final stage ``stage``
(PROMPT.md D3):

    e(d)      = e_min + e_range * alpha / (alpha + beta)      learner, float64 from the float32
                                                              (alpha, beta) of ``agent.actor``
    e_opp(d)  = the same float64 mean of ``agent.opponent`` at the observation of -d (no grad)
    R(d,e,eo) = w_l + DW * F_xi(d + e - eo) - k * e^2
    loss      = -mean_rows R(d, e(d), e_opp(d))

F_xi is the CDF of eps_own - eps_opp (triangular on [-2q, 2q]); ``torch_F_xi`` equals
``utils.theory_multistage.F_xi`` to 1e-12. The payoff function is the known game, as in the
locked ``reward_mode=expected``; the closed-form equilibrium effort never appears here.

Call pattern (the runner owns everything not listed here)::

    for g in agent.opt_actor.param_groups:       # runner sets the LR (constant 3e-5 in D3)
        g["lr"] = lr
    # runner: exploring starts / roles from the ``start`` stream, opponent refresh every 20
    # global updates, d_learner = where(roles == 0, d0, -d0)
    out = pathwise_step(agent, spec, d_learner, stage=T)
    # out: loss, grad_norm_pre_clip, foc_abs_mean, foc_abs_max, e0   (all python floats)

One call is exactly one optimizer step of the EXISTING ``agent.opt_actor`` (no new Adam, state
preserved, LR untouched) with global-norm clipping at ``max_grad_norm`` (the PRE-clip norm is
returned). No critic, no PPO ratio or clipping, no entropy term. The step draws no random
number: neither a numpy generator nor the global torch generator is touched.

Concentration head. Output row 1 of ``actor.out`` (``weight[1, :]`` and ``bias[1]``) must receive
no update. Zeroing its gradient (done explicitly after ``backward`` and before clipping, so the
returned norm excludes it) is necessary but NOT sufficient: the preserved Adam state of those
parameters (``exp_avg`` left by Phase A) still produces a nonzero step even for a zero gradient.
The row-1 weight and bias are therefore saved before the step and restored bit-exactly right
after ``opt_actor.step()``. This is how the PI's "no update" and "bit-identical head at the end of
the phase" are both delivered. Side effect, harmless in a terminal phase: the Adam moments of
row 1 keep decaying under the zero gradient (the step counter is shared by the tensor); the
parameters themselves never move. Every other parameter, the critic and its Adam state are not
touched. After the call ``p.grad`` of the actor holds the clipped gradient (row 1 zero).

Evaluation of the residual. ``foc_abs_mean`` / ``foc_abs_max`` are the mean / max over rows of
``|dR/de| = |DW * f_xi(d + e - e_opp) - 2 k e|`` at the PRE-step parameters on the same rows as
the loss (``e`` and ``e_opp`` held at their executed values; this equals autograd of R wrt e).
``loss`` and ``e0`` (learner mean effort at d = 0 of ``stage``, float64) are also pre-step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:  # typing only; keeps this module importable while the agents are edited
    from agents.ppo_curriculum import BetaActor, CurriculumPPO
    from envs.curriculum_env import GameSpec

__all__ = ["torch_F_xi", "torch_f_xi", "effort_mean", "expected_payoff", "foc_residual",
           "pathwise_loss", "pathwise_step"]


def torch_F_xi(x: torch.Tensor, q: float) -> torch.Tensor:
    """CDF of xi ~ Triangular(-2q, 2q) in torch float64 (differentiable, density as gradient).

    Same piecewise formulas and branch convention (x < 0 / x >= 0) as
    ``utils.theory_multistage.F_xi``; the tails are exactly 0 and 1 and carry zero gradient.

    Args:
        x: Evaluation points (cast to float64; a float64 input is used as is).
        q: Half-width of the per-player noise U(-q, q).

    Returns:
        CDF values in [0, 1], float64, same shape as ``x``.
    """
    x = x.to(torch.float64)
    two_q = 2.0 * q
    den = 8.0 * q * q
    u = torch.clamp(x, -two_q, two_q)
    neg = (u + two_q) ** 2 / den
    pos = 1.0 - (two_q - u) ** 2 / den
    return torch.where(x < 0.0, neg, pos)


def torch_f_xi(x: torch.Tensor, q: float) -> torch.Tensor:
    """Density of xi ~ Triangular(-2q, 2q) in torch float64 (zero outside [-2q, 2q]).

    Args:
        x: Evaluation points (cast to float64).
        q: Half-width of the per-player noise U(-q, q).

    Returns:
        Density values, float64, same shape as ``x``.
    """
    ax = torch.abs(x.to(torch.float64))
    return torch.where(ax <= 2.0 * q, (2.0 * q - ax) / (4.0 * q * q), torch.zeros_like(ax))


def effort_mean(net: "BetaActor", obs: torch.Tensor, spec: "GameSpec") -> torch.Tensor:
    """Beta-mean effort in float64 from the float32 (alpha, beta), differentiable.

    Same float path as evaluation and as ``continuation_action_mode=mean``:
    ``e_min + e_range * a64 / (a64 + b64)`` with ``a64 = alpha.double()``.

    Args:
        net: A ``BetaActor`` (live actor or lagged opponent).
        obs: (N, 2) float32 observations.
        spec: Game parameters (``e_min``, ``e_range``).

    Returns:
        (N,) float64 efforts.
    """
    alpha, beta = net(obs)
    a64 = alpha.double()
    b64 = beta.double()
    return spec.e_min + spec.e_range * (a64 / (a64 + b64))


def expected_payoff(spec: "GameSpec", d: torch.Tensor, e: torch.Tensor,
                    e_opp: torch.Tensor) -> torch.Tensor:
    """Per-row conditional expected terminal payoff w_l + DW F_xi(d + e - e_opp) - k e^2.

    Args:
        spec: Game parameters (``w_l``, ``dw``, ``k``, ``q``).
        d: Learner signed gap per row (float64).
        e: Learner effort per row (float64, may carry grad).
        e_opp: Opponent effort per row (float64).

    Returns:
        (N,) float64 payoffs.
    """
    return spec.w_l + spec.dw * torch_F_xi(d + e - e_opp, spec.q) - spec.k * e * e


def foc_residual(spec: "GameSpec", d: torch.Tensor, e: torch.Tensor,
                 e_opp: torch.Tensor) -> torch.Tensor:
    """Analytic dR/de = DW f_xi(d + e - e_opp) - 2 k e per row (float64).

    Args:
        spec: Game parameters.
        d: Learner signed gap per row (float64).
        e: Learner effort per row (float64).
        e_opp: Opponent effort per row (float64).

    Returns:
        (N,) float64 first-order-condition residuals.
    """
    return spec.dw * torch_f_xi(d + e - e_opp, spec.q) - 2.0 * spec.k * e


def pathwise_loss(agent: "CurriculumPPO", spec: "GameSpec", d_learner: np.ndarray, stage: int
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the D3 loss on the live actor graph (no optimizer interaction).

    Args:
        agent: Agent with ``actor``, ``opponent`` (lagged copy) and ``device``.
        spec: Game parameters.
        d_learner: (N,) learner signed gap per row (float64).
        stage: Final-stage index used by ``spec.encode_obs(stage, .)``.

    Returns:
        ``(loss, d, e, e_opp)``: scalar float64 loss = -mean R (carries the actor graph), the
        float64 gap tensor, the learner efforts (graph) and the opponent efforts (no grad).
    """
    d = np.asarray(d_learner, dtype=float).reshape(-1)
    dev = agent.device
    obs_l = torch.as_tensor(spec.encode_obs(stage, d), device=dev)
    obs_o = torch.as_tensor(spec.encode_obs(stage, -d), device=dev)
    d_t = torch.as_tensor(d, dtype=torch.float64, device=dev)
    with torch.no_grad():
        e_opp = effort_mean(agent.opponent, obs_o, spec)
    e = effort_mean(agent.actor, obs_l, spec)
    loss = -expected_payoff(spec, d_t, e, e_opp).mean()
    return loss, d_t, e, e_opp


def pathwise_step(agent: "CurriculumPPO", spec: "GameSpec", d_learner: np.ndarray, stage: int,
                  max_grad_norm: float = 0.5) -> Dict[str, float]:
    """One exact-gradient optimizer step of the D3 objective (see the module docstring).

    Args:
        agent: ``CurriculumPPOv2`` (``actor``, ``opponent``, ``opt_actor``, ``device``).
        spec: Game parameters.
        d_learner: (N,) learner signed gap per row (float64); the opponent sees ``-d_learner``.
        stage: Final-stage index (no hard-coded stage).
        max_grad_norm: Global-norm clip of the actor gradient.

    Returns:
        Dict with python floats ``loss``, ``grad_norm_pre_clip``, ``foc_abs_mean``,
        ``foc_abs_max`` and ``e0`` (all at the pre-step parameters; the norm excludes row 1).
    """
    actor = agent.actor
    head_w = actor.out.weight[1].detach().clone()
    head_b = actor.out.bias[1].detach().clone()
    agent.opt_actor.zero_grad(set_to_none=True)

    loss, d_t, e, e_opp = pathwise_loss(agent, spec, d_learner, stage)
    with torch.no_grad():
        foc = foc_residual(spec, d_t, e.detach(), e_opp).abs()
        obs0 = torch.as_tensor(spec.encode_obs(stage, np.zeros(1)), device=agent.device)
        e0 = effort_mean(actor, obs0, spec)[0]
    loss.backward()
    actor.out.weight.grad[1].zero_()
    actor.out.bias.grad[1].zero_()
    gn = nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    agent.opt_actor.step()
    with torch.no_grad():
        actor.out.weight[1].copy_(head_w)
        actor.out.bias[1].copy_(head_b)
    return {
        "loss": float(loss.item()),
        "grad_norm_pre_clip": float(gn),
        "foc_abs_mean": float(foc.mean().item()),
        "foc_abs_max": float(foc.max().item()),
        "e0": float(e0.item()),
    }
