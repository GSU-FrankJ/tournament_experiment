"""v2 PPO agent: the unchanged CurriculumPPO plus a masked policy update and full-state I/O.

``CurriculumPPOv2.update`` with ``policy_mask=None`` and ``norm_mask=None`` calls the original
``CurriculumPPO.update`` untouched (joint mode, bit-identical by construction). Otherwise:

  - advantage normalization statistics (mean, population SD) are taken over ``norm_mask`` rows
    and applied to every row;
  - the minibatch partition is the same permutation of ALL rows as the original (one
    ``rng_mb.permutation(n)`` per epoch), so RNG consumption is unchanged;
  - the actor loss of a minibatch is the clipped surrogate averaged over the minibatch rows that
    are in ``policy_mask``; the other rows never enter the actor graph (exactly zero gradient);
    a minibatch without policy rows takes no actor step (counted in the diagnostics);
  - the critic step is unchanged (all rows of the minibatch).

The frozen stage-2 snapshot is a separate deep copy of the actor (eval mode, requires_grad False,
referenced by no optimizer). It is distinct from ``self.opponent`` (the lagged self-play copy).
The observation encoding is a fixed function (no normalizer), so the actor parameters are the
complete stage-2 mapping.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.ppo_curriculum import BetaActor, CurriculumPPO


def masked_actor_loss(actor: BetaActor, st: torch.Tensor, ac: torch.Tensor, olp: torch.Tensor,
                      adv: torch.Tensor, rows: torch.Tensor, clip_eps: float):
    """Clipped-surrogate actor loss over ``rows`` only.

    Args:
        actor: Live actor.
        st, ac, olp, adv: Full-buffer tensors (states, actions, old log-probs, normalized adv).
        rows: Long tensor of row indices that enter the loss.
        clip_eps: PPO clip.

    Returns:
        ``(loss, ratio, entropy_mean)``.
    """
    alpha, beta = actor(st[rows])
    dist = torch.distributions.Beta(alpha, beta)
    logp = dist.log_prob(ac[rows])
    ratio = torch.exp(logp - olp[rows])
    a_mb = adv[rows]
    surr1 = ratio * a_mb
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * a_mb
    return -torch.min(surr1, surr2).mean(), ratio, dist.entropy().mean()


class CurriculumPPOv2(CurriculumPPO):
    """CurriculumPPO + masked update + frozen stage-2 snapshot + full-state save/load."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen: Optional[BetaActor] = None

    # ------------------------------------------------------------------ frozen snapshot
    def freeze_stage2_snapshot(self) -> None:
        """Deep-copy the current actor as the frozen stage-2 mapping (no grad, eval, no optimizer)."""
        snap = copy.deepcopy(self.actor)
        for p in snap.parameters():
            p.requires_grad_(False)
        snap.eval()
        self.frozen = snap

    # ------------------------------------------------------------------ update
    def update(self, states: np.ndarray, actions: np.ndarray, old_logp: np.ndarray,
               returns: np.ndarray, advantages_raw: np.ndarray,
               policy_mask: Optional[np.ndarray] = None,
               norm_mask: Optional[np.ndarray] = None) -> Dict[str, object]:
        """PPO update; the original update when both masks are None (see module docstring)."""
        if policy_mask is None and norm_mask is None:
            return super().update(states, actions, old_logp, returns, advantages_raw)
        cfg = self.cfg
        dev = self.device
        st = torch.as_tensor(np.asarray(states, dtype=np.float32), device=dev)
        ac = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=dev)
        olp = torch.as_tensor(np.asarray(old_logp, dtype=np.float32), device=dev)
        ret = torch.as_tensor(np.asarray(returns, dtype=np.float32), device=dev)
        adv_raw = torch.as_tensor(np.asarray(advantages_raw, dtype=np.float32), device=dev)
        n = st.shape[0]
        pm = torch.ones(n, dtype=torch.bool) if policy_mask is None else torch.as_tensor(np.asarray(policy_mask, bool))
        nm = torch.ones(n, dtype=torch.bool) if norm_mask is None else torch.as_tensor(np.asarray(norm_mask, bool))
        if bool(nm.all()):
            adv_mean = adv_raw.mean()
            adv_std = adv_raw.std(unbiased=False)
        else:
            adv_mean = adv_raw[nm].mean()
            adv_std = adv_raw[nm].std(unbiased=False)
        adv = (adv_raw - adv_mean) / (adv_std + cfg.adv_norm_eps)
        prow = torch.nonzero(pm).squeeze(-1)

        pl: List[float] = []
        vl: List[float] = []
        ent: List[float] = []
        cf: List[float] = []
        gna: List[float] = []
        gnc: List[float] = []
        kl_epochs: List[float] = []
        steps = 0
        actor_steps = 0
        skipped = 0
        for _ in range(cfg.epochs):
            perm = self.rng_mb.permutation(n)
            for start in range(0, n, cfg.minibatch):
                idx = torch.as_tensor(perm[start:start + cfg.minibatch], device=dev)
                rows = idx[pm[idx]]
                if rows.numel() > 0:
                    actor_loss, ratio, entropy = masked_actor_loss(self.actor, st, ac, olp, adv, rows,
                                                                   cfg.clip_eps)
                    loss_a = actor_loss - cfg.entropy_coef * entropy
                    self.opt_actor.zero_grad(set_to_none=True)
                    loss_a.backward()
                    gn_a = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                    self.opt_actor.step()
                    pl.append(float(actor_loss.item()))
                    ent.append(float(entropy.item()))
                    cf.append(float(((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()))
                    gna.append(float(gn_a))
                    actor_steps += 1
                else:
                    skipped += 1

                v = self.critic(st[idx])
                value_loss = cfg.value_coef * F.mse_loss(v, ret[idx])
                self.opt_critic.zero_grad(set_to_none=True)
                value_loss.backward()
                gn_c = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.opt_critic.step()
                vl.append(float(value_loss.item()))
                gnc.append(float(gn_c))
                steps += 1
            with torch.no_grad():
                alpha, beta = self.actor(st[prow])
                logp = torch.distributions.Beta(alpha, beta).log_prob(ac[prow])
                log_r = logp - olp[prow]
                r = torch.exp(log_r)
                kl_epochs.append(float(((r - 1.0) - log_r).mean().item()))

        with torch.no_grad():
            alpha, beta = self.actor(st[prow])
            dist = torch.distributions.Beta(alpha, beta)
            ent_post = float(dist.entropy().mean().item())
            conc = alpha + beta
            conc_stats = (float(conc.min().item()), float(conc.mean().item()), float(conc.max().item()))

        return {
            "policy_loss": float(np.mean(pl)) if pl else float("nan"),
            "value_loss": float(np.mean(vl)),
            "entropy_minibatch_mean": float(np.mean(ent)) if ent else float("nan"),
            "entropy_post_update": ent_post,
            "entropy_post_update_effort_scale": ent_post + math.log(100.0),
            "clip_frac": float(np.mean(cf)) if cf else float("nan"),
            "grad_norm_actor_mean": float(np.mean(gna)) if gna else float("nan"),
            "grad_norm_actor_max": float(np.max(gna)) if gna else float("nan"),
            "grad_norm_critic_mean": float(np.mean(gnc)),
            "grad_norm_critic_max": float(np.max(gnc)),
            "kl_final_epoch": kl_epochs[-1],
            "kl_epochs": kl_epochs,
            "adv_raw_mean": float(adv_mean.item()),
            "adv_raw_std": float(adv_std.item()),
            "conc_buffer_min_mean_max": conc_stats,
            "n_transitions": int(n),
            "n_minibatch_steps": int(steps),
            "lr": float(self.opt_actor.param_groups[0]["lr"]),
            "n_policy_rows": int(prow.numel()),
            "n_norm_rows": int(nm.sum().item()),
            "n_actor_steps": int(actor_steps),
            "n_actor_steps_skipped_no_policy_rows": int(skipped),
        }

    # ------------------------------------------------------------------ full state
    def full_state(self) -> Dict[str, object]:
        """Everything the agent owns: weights, optimizers, refresh count, minibatch RNG, snapshot."""
        s = self.state()
        s["snapshot_refreshes"] = int(self.snapshot_refreshes)
        s["rng_minibatch"] = copy.deepcopy(self.rng_mb.bit_generator.state)
        s["frozen"] = (None if self.frozen is None else
                       {k: v.detach().cpu().clone() for k, v in self.frozen.state_dict().items()})
        return s

    def load_full_state(self, s: Dict[str, object]) -> None:
        """Inverse of :meth:`full_state` (optimizer objects are kept; their state is replaced)."""
        self.actor.load_state_dict(s["actor"])
        self.critic.load_state_dict(s["critic"])
        self.opponent.load_state_dict(s["opponent"])
        for p in self.opponent.parameters():
            p.requires_grad_(False)
        self.opponent.eval()
        self.opt_actor.load_state_dict(s["opt_actor"])
        self.opt_critic.load_state_dict(s["opt_critic"])
        self.snapshot_refreshes = int(s["snapshot_refreshes"])
        self.rng_mb.bit_generator.state = copy.deepcopy(s["rng_minibatch"])
        if s.get("frozen") is not None:
            self.freeze_stage2_snapshot()
            self.frozen.load_state_dict(s["frozen"])
        else:
            self.frozen = None
