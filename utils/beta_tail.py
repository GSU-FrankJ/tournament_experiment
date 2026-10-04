"""Censored log-likelihood of clamped Beta draws (R2b mechanism 2).

The actor samples a ~ Beta(alpha, beta) and stores ``clip(a, c, 1 - c)`` (``c`` = ``action_clamp``).
A row whose raw draw fell below ``c`` (above ``1 - c``) is *censored*: all that is known is the event
``A <= c`` (``A >= 1 - c``), so its likelihood is the mass of that event

    P(A <= c)     = I_c(alpha, beta)          (lower side, ``side == -1``)
    P(A >= 1 - c) = I_c(beta, alpha)          (upper side, ``side == +1``)

instead of the density at the clipped value (which is not the probability of anything).

``log_betainc_small_x`` evaluates ``log I_x(a, b)`` for small ``x`` from the series

    I_x(a, b) = x^a / (a B(a, b)) * 2F1(a, 1 - b; a + 1; x)
              = x^a / (a B(a, b)) * sum_n  a / (a + n) * (1 - b)_n / n! * x^n,

which converges fast when ``b x`` is small (the ratio of consecutive terms is ``(n - b) x / n``).
It is written in torch (``lgamma``-based, differentiable in ``a`` and ``b``) and computed in float64
inside: for ``b ~ 1e3`` the ``lgamma`` terms are ~6e3 and float32 would lose about 4e-4 of absolute
accuracy in the log. Domain: ``x b <= 0.05`` (at the default ``c = 1e-6`` that is ``b <= 5e4``; the
concentration of the actor is ``100 + softplus(.)``); outside it ``ValueError`` is raised, because
the truncated series would silently be wrong. Tested against ``scipy.special.betainc`` to a relative
error <= 1e-6 over a in [1e-4, 10], b in [1, 1000], x = 1e-6 (``tests/test_v2_r2b.py``).

``row_log_prob`` is the log-probability of a buffer of stored actions: the Beta density for every
row, the censored log-mass for the flagged rows. With no flagged row it is exactly
``Beta(alpha, beta).log_prob(actions)`` (the same torch call, nothing else is computed).
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["log_betainc_small_x", "row_log_prob", "N_TERMS", "DOMAIN_X_B"]

N_TERMS = 12          # series terms; with x b <= 0.05 the first neglected term is <= 0.05^12 / 12! ~ 5e-25
DOMAIN_X_B = 0.05     # largest allowed x * b


def log_betainc_small_x(x: float, a: torch.Tensor, b: torch.Tensor, n_terms: int = N_TERMS) -> torch.Tensor:
    """``log I_x(a, b)`` for a small scalar ``x`` (see the module docstring).

    Args:
        x: Evaluation point in (0, 1) (a Python float; ``action_clamp``).
        a: Shape ``a`` of the incomplete beta function (any float dtype; the small-shape side).
        b: Shape ``b`` (same shape as ``a``); must satisfy ``x * max(b) <= DOMAIN_X_B``.
        n_terms: Number of series terms.

    Returns:
        float64 tensor of ``log I_x(a, b)``, same shape as ``a``, carrying the autograd graph of ``a``
        and ``b``.
    """
    x = float(x)
    if not 0.0 < x < 1.0:
        raise ValueError(f"x must lie in (0, 1); got {x}")
    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)
    if b64.numel() and float(b64.detach().max()) * x > DOMAIN_X_B:
        raise ValueError(f"series domain exceeded: x * max(b) = {x * float(b64.detach().max()):.3g} > {DOMAIN_X_B}")
    log_beta = torch.lgamma(a64) + torch.lgamma(b64) - torch.lgamma(a64 + b64)
    term = torch.ones_like(a64)            # t_n = (1 - b)_n x^n / n!, t_0 = 1
    total = torch.ones_like(a64)           # n = 0: a / (a + 0) * t_0
    for n in range(1, n_terms):
        term = term * ((n - b64) * (x / n))
        total = total + term * (a64 / (a64 + n))
    return a64 * torch.log(torch.tensor(x, dtype=torch.float64)) - torch.log(a64) - log_beta + torch.log(total)


def row_log_prob(alpha: torch.Tensor, beta: torch.Tensor, actions: torch.Tensor,
                 side: Optional[torch.Tensor], clamp_c: float) -> torch.Tensor:
    """Log-probability of stored actions, censored for clamped rows.

    Args:
        alpha, beta: (N,) Beta shapes (float32, may carry grad).
        actions: (N,) stored actions (float32).
        side: (N,) int tensor in {-1, 0, +1}: -1 = the raw draw fell below ``clamp_c``, +1 = above
            ``1 - clamp_c``, 0 = unclamped; ``None`` = no row is censored (the plain density).
        clamp_c: ``action_clamp``.

    Returns:
        (N,) float32 log-probabilities: the density at the stored action for unflagged rows, the
        censored log-mass for flagged rows.
    """
    lp = torch.distributions.Beta(alpha, beta).log_prob(actions)
    if side is None:
        return lp
    lo = torch.nonzero(side < 0).squeeze(-1)
    hi = torch.nonzero(side > 0).squeeze(-1)
    if lo.numel():
        lp = lp.index_put((lo,), log_betainc_small_x(clamp_c, alpha[lo], beta[lo]).to(lp.dtype))
    if hi.numel():
        lp = lp.index_put((hi,), log_betainc_small_x(clamp_c, beta[hi], alpha[hi]).to(lp.dtype))
    return lp
