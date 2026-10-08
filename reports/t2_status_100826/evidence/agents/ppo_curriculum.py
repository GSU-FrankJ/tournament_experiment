"""PPO agent for the final DP-BR protocol (2026-09-07): separate actor/critic, frozen snapshot.

Architecture (EXPERIMENT_SETTINGS_REVIEW section 3):
    actor : 2 -> 64 Tanh -> 64 Tanh -> 2 linear (z_mu, z_c)
    critic: 2 -> 64 Tanh -> 64 Tanh -> 1 linear, no shared trunk
    init  : hidden layers orthogonal gain sqrt(2), bias 0; actor output head
            weight 0 / bias 0; critic output orthogonal gain 1 / bias 0
    Beta  : mu = clamp(sigmoid(z_mu), 1e-6, 1-1e-6); c = 100 + softplus(z_c);
            alpha = mu c, beta = (1 - mu) c   (initial c = 100 + ln 2, mean 0.5)

Optimization: separate Adam optimizers (lr 3e-4 fixed, betas (.9,.999), eps 1e-8,
weight_decay 0), actor loss = -clipped surrogate (entropy coefficient 0), critic
loss = 0.5 MSE(V, return), no value clipping, separate global-norm clipping at 0.5
with pre-clip norms recorded, 10 epochs over reshuffled minibatches of 256 learner
transitions, per-update advantage normalization by mean / population SD + 1e-8,
empirical KL mean((r-1) - log r) on the whole buffer after every epoch.

Actions are sampled OUTSIDE torch with numpy Generators (separate learner and
opponent streams) and stored as float32 clipped to [1e-6, 1-1e-6]; log-probs are
evaluated by torch on that stored value, so rollout and update use the same action.
The network is float32 throughout.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    """Hyperparameters (defaults = the 2026-09-07 protocol)."""

    hidden: int = 64
    lr: float = 3e-4
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    epochs: int = 10
    minibatch: int = 256
    gamma: float = 1.0
    gae_lambda: float = 1.0
    c_min: float = 100.0
    mu_clamp: float = 1e-6
    action_clamp: float = 1e-6
    adv_norm_eps: float = 1e-8


def _orthogonal(layer: nn.Linear, gain: float, gen: torch.Generator) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain, generator=gen)
    nn.init.zeros_(layer.bias)


ACTOR_VARIANTS = ("t1", "relu", "t10")
D_FEATURE_SCALE_T10 = 10.0     # the d component of the actor input is multiplied by this in the "t10" variant


class BetaActor(nn.Module):
    """Mean/concentration Beta actor."""

    def __init__(self, hidden: int, c_min: float, mu_clamp: float, gen: torch.Generator):
        super().__init__()
        self.l1 = nn.Linear(2, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 2)
        self.c_min = float(c_min)
        self.mu_clamp = float(mu_clamp)
        # R1 refinement: multiplicative concentration factor (1.0 = the locked behaviour; the
        # multiplication is skipped at 1.0, so the default graph is operation-for-operation unchanged)
        self.conc_scale: float = 1.0
        # MS-R3: actor variant. "t1" = the locked tanh actor on d / B (the default path below is untouched);
        # "relu" / "t10" are set by CurriculumPPO.set_actor_variant after construction, so the initial
        # weights are drawn by exactly the same generator calls whatever the variant.
        self.variant: str = "t1"
        _orthogonal(self.l1, math.sqrt(2.0), gen)
        _orthogonal(self.l2, math.sqrt(2.0), gen)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha, beta), each of shape (N,)."""
        if self.variant != "t1":
            h = self._hidden_variant(x)
        else:
            h = torch.tanh(self.l1(x))
            h = torch.tanh(self.l2(h))
        z = self.out(h)
        mu = torch.clamp(torch.sigmoid(z[:, 0]), self.mu_clamp, 1.0 - self.mu_clamp)
        c = self.c_min + F.softplus(z[:, 1])
        if self.conc_scale != 1.0:
            c = c * self.conc_scale
        return mu * c, (1.0 - mu) * c

    def _hidden_variant(self, x: torch.Tensor) -> torch.Tensor:
        """Hidden layers of the MS-R3 variants (``relu``: ReLU units; ``t10``: tanh on ``10 d / B``)."""
        if self.variant == "t10":
            x = x * x.new_tensor([1.0, D_FEATURE_SCALE_T10])      # the stage feature (column 0) is not scaled
            act = torch.tanh
        elif self.variant == "relu":
            act = torch.relu
        else:
            raise ValueError(f"unknown actor variant {self.variant!r}")
        return act(self.l2(act(self.l1(x))))


class Critic(nn.Module):
    """State-value network."""

    def __init__(self, hidden: int, gen: torch.Generator):
        super().__init__()
        self.l1 = nn.Linear(2, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        _orthogonal(self.l1, math.sqrt(2.0), gen)
        _orthogonal(self.l2, math.sqrt(2.0), gen)
        _orthogonal(self.out, 1.0, gen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return V(x) of shape (N,)."""
        h = torch.tanh(self.l1(x))
        h = torch.tanh(self.l2(h))
        return self.out(h).squeeze(-1)


class CurriculumPPO:
    """Learner actor/critic plus a frozen opponent snapshot."""

    def __init__(self, cfg: PPOConfig, torch_gen: torch.Generator,
                 rng_minibatch: np.random.Generator, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = BetaActor(cfg.hidden, cfg.c_min, cfg.mu_clamp, torch_gen).to(self.device)
        self.critic = Critic(cfg.hidden, torch_gen).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, betas=cfg.adam_betas,
                                          eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr,
                                           betas=cfg.adam_betas, eps=cfg.adam_eps,
                                           weight_decay=cfg.weight_decay)
        self.rng_mb = rng_minibatch
        self.target_kl: Optional[float] = None   # R1: stop the epoch loop after an epoch with KL > target
        self.opponent: BetaActor = copy.deepcopy(self.actor)
        self.snapshot_refreshes: int = 0
        self.refresh_snapshot()

    # ------------------------------------------------------------------ MS-R3 actor variant
    def set_actor_variant(self, variant: str) -> None:
        """Set the actor variant of the live actor and the lagged opponent (call before any update).

        Args:
            variant: ``"t1"`` (the locked actor), ``"relu"`` or ``"t10"``. The initial weights are not touched
                (they were drawn at construction with the same generator calls for every variant).
        """
        if variant not in ACTOR_VARIANTS:
            raise ValueError(f"unknown actor variant {variant!r}; known: {ACTOR_VARIANTS}")
        self.actor.variant = variant
        self.opponent.variant = variant

    # ------------------------------------------------------------------ snapshot
    def refresh_snapshot(self) -> None:
        """Copy the current actor into the frozen opponent."""
        self.opponent.load_state_dict(self.actor.state_dict())
        self.opponent.conc_scale = self.actor.conc_scale
        self.opponent.variant = self.actor.variant
        for p in self.opponent.parameters():
            p.requires_grad_(False)
        self.opponent.eval()
        self.snapshot_refreshes += 1

    # ------------------------------------------------------------------ numpy API
    @torch.no_grad()
    def beta_params(self, obs: np.ndarray, net: Optional[BetaActor] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """(alpha, beta) as float32 numpy arrays for a float32 observation batch."""
        net = self.actor if net is None else net
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        a, b = net(x)
        return a.cpu().numpy(), b.cpu().numpy()

    def sample_actions(self, alpha: np.ndarray, beta: np.ndarray, rng: np.random.Generator
                       ) -> np.ndarray:
        """Sample normalized actions from Beta(alpha, beta) with a numpy stream (float32)."""
        a = rng.beta(alpha.astype(float), beta.astype(float))
        c = self.cfg.action_clamp
        return np.clip(a, c, 1.0 - c).astype(np.float32)

    @torch.no_grad()
    def log_prob(self, alpha: np.ndarray, beta: np.ndarray, actions: np.ndarray,
                 clamp_side: Optional[np.ndarray] = None) -> np.ndarray:
        """Beta log-density of stored actions (float32 torch).

        With ``clamp_side`` (R2b, ``clamp_likelihood="censored"``: int array in {-1, 0, +1}) the rows
        flagged -1 / +1 get the censored log-mass of ``A <= c`` / ``A >= 1 - c`` (``utils.beta_tail``)
        instead of the density at the clipped value. ``None`` (default) is the original call.
        """
        if clamp_side is None:
            dist = torch.distributions.Beta(torch.as_tensor(alpha), torch.as_tensor(beta))
            return dist.log_prob(torch.as_tensor(actions)).cpu().numpy()
        from utils.beta_tail import row_log_prob   # lazy: the default path never imports it
        return row_log_prob(torch.as_tensor(alpha), torch.as_tensor(beta), torch.as_tensor(actions),
                            torch.as_tensor(np.asarray(clamp_side, dtype=np.int8)),
                            self.cfg.action_clamp).cpu().numpy()

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> np.ndarray:
        """Critic values (float32 numpy)."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return self.critic(x).cpu().numpy()

    @torch.no_grad()
    def entropy(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Differential entropy of Beta(alpha, beta) on [0, 1]."""
        dist = torch.distributions.Beta(torch.as_tensor(alpha), torch.as_tensor(beta))
        return dist.entropy().cpu().numpy()

    # ------------------------------------------------------------------ update
    def update(self, states: np.ndarray, actions: np.ndarray, old_logp: np.ndarray,
               returns: np.ndarray, advantages_raw: np.ndarray) -> Dict[str, object]:
        """One PPO update over the learner buffer.

        Args:
            states: (N, 2) float32 observations.
            actions: (N,) float32 normalized actions.
            old_logp: (N,) float32 rollout log-probs.
            returns: (N,) float32 realized returns.
            advantages_raw: (N,) float32 raw advantages.

        Returns:
            Diagnostics dict.
        """
        cfg = self.cfg
        dev = self.device
        st = torch.as_tensor(np.asarray(states, dtype=np.float32), device=dev)
        ac = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=dev)
        olp = torch.as_tensor(np.asarray(old_logp, dtype=np.float32), device=dev)
        ret = torch.as_tensor(np.asarray(returns, dtype=np.float32), device=dev)
        adv_raw = torch.as_tensor(np.asarray(advantages_raw, dtype=np.float32), device=dev)
        adv_mean = adv_raw.mean()
        adv_std = adv_raw.std(unbiased=False)
        adv = (adv_raw - adv_mean) / (adv_std + cfg.adv_norm_eps)
        n = st.shape[0]

        pl: List[float] = []
        vl: List[float] = []
        ent: List[float] = []
        cf: List[float] = []
        gna: List[float] = []
        gnc: List[float] = []
        kl_epochs: List[float] = []
        steps = 0
        stopped = False
        for _ in range(cfg.epochs):
            perm = self.rng_mb.permutation(n)   # always drawn: stream position independent of target_kl
            if stopped:
                continue
            for start in range(0, n, cfg.minibatch):
                idx = torch.as_tensor(perm[start:start + cfg.minibatch], device=dev)
                alpha, beta = self.actor(st[idx])
                dist = torch.distributions.Beta(alpha, beta)
                logp = dist.log_prob(ac[idx])
                ratio = torch.exp(logp - olp[idx])
                a_mb = adv[idx]
                surr1 = ratio * a_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * a_mb
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().mean()
                loss_a = actor_loss - cfg.entropy_coef * entropy
                self.opt_actor.zero_grad(set_to_none=True)
                loss_a.backward()
                gn_a = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.opt_actor.step()

                v = self.critic(st[idx])
                value_loss = cfg.value_coef * F.mse_loss(v, ret[idx])
                self.opt_critic.zero_grad(set_to_none=True)
                value_loss.backward()
                gn_c = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.opt_critic.step()

                pl.append(float(actor_loss.item()))
                vl.append(float(value_loss.item()))
                ent.append(float(entropy.item()))
                cf.append(float(((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()))
                gna.append(float(gn_a))
                gnc.append(float(gn_c))
                steps += 1
            with torch.no_grad():
                alpha, beta = self.actor(st)
                logp = torch.distributions.Beta(alpha, beta).log_prob(ac)
                log_r = logp - olp
                r = torch.exp(log_r)
                kl_epochs.append(float(((r - 1.0) - log_r).mean().item()))
            if self.target_kl is not None and kl_epochs[-1] > self.target_kl:
                stopped = True

        with torch.no_grad():
            alpha, beta = self.actor(st)
            dist = torch.distributions.Beta(alpha, beta)
            ent_post = float(dist.entropy().mean().item())
            conc = (alpha + beta)
            conc_stats = (float(conc.min().item()), float(conc.mean().item()),
                          float(conc.max().item()))

        out = {
            "policy_loss": float(np.mean(pl)),
            "value_loss": float(np.mean(vl)),
            "entropy_minibatch_mean": float(np.mean(ent)),
            "entropy_post_update": ent_post,
            "entropy_post_update_effort_scale": ent_post + math.log(100.0),
            "clip_frac": float(np.mean(cf)),
            "grad_norm_actor_mean": float(np.mean(gna)),
            "grad_norm_actor_max": float(np.max(gna)),
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
        }
        if self.target_kl is not None:
            out["n_epochs_run"] = len(kl_epochs)
        return out

    # ------------------------------------------------------------------ persistence
    def state(self) -> Dict[str, object]:
        """Serializable state (weights, optimizer states, config)."""
        return {
            "actor": {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()},
            "critic": {k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()},
            "opponent": {k: v.detach().cpu().clone() for k, v in self.opponent.state_dict().items()},
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "cfg": asdict(self.cfg),
        }

    def save(self, path: str) -> None:
        """Save actor/critic/opponent weights + optimizer states."""
        torch.save(self.state(), path)

    def load_weights(self, path: str) -> Dict[str, object]:
        """Load actor/critic (and opponent) weights from :meth:`save` output."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        if "opponent" in ck:
            self.opponent.load_state_dict(ck["opponent"])
        return ck

    def export_weights_npz(self, path: str) -> None:
        """Write float32 actor/critic weights as plain numpy arrays."""
        arrs = {f"actor.{k}": v.detach().cpu().numpy() for k, v in self.actor.state_dict().items()}
        arrs.update({f"critic.{k}": v.detach().cpu().numpy()
                     for k, v in self.critic.state_dict().items()})
        if self.actor.conc_scale != 1.0:
            # exact Python float (float64); mean_effort_numpy casts it to float32 as torch does
            arrs["conc_scale"] = np.asarray(self.actor.conc_scale, dtype=np.float64)
        if self.actor.variant != "t1":
            arrs["actor_variant"] = np.asarray(self.actor.variant)    # absent = "t1", as in every earlier export
        np.savez(path, **arrs)


def mean_effort_numpy(weights: Dict[str, np.ndarray], obs: np.ndarray, c_min: float = 100.0,
                      mu_clamp: float = 1e-6, e_min: float = 0.0, e_max: float = 100.0,
                      conc_scale: Optional[float] = None, variant: Optional[str] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Framework-free float32 re-implementation of the actor for reload checks.

    Args:
        weights: Arrays from :meth:`CurriculumPPO.export_weights_npz` (actor.* keys).
        obs: (N, 2) float32 observations.
        c_min: Concentration floor.
        mu_clamp: Mean clamp.
        e_min: Lower effort bound.
        e_max: Upper effort bound.
        conc_scale: Concentration factor of an annealed actor. ``None`` (default) reads the
            ``conc_scale`` array of an exported ``.npz`` if ``weights`` carries one, else 1.0
            (unscaled). Applied in float32 as torch does. The reload reproduces (alpha, beta) to
            float32 rounding (rtol ~1e-6), as it always did; it is not a bitwise check.
        variant: MS-R3 actor variant (``"t1"``, ``"relu"``, ``"t10"``). ``None`` (default) reads the
            ``actor_variant`` entry of an export if present, else ``"t1"`` (the tanh actor on d / B,
            as every earlier export). An unknown variant raises ``ValueError``; the reload never
            falls back to the tanh forward for an export that names another variant.

    Returns:
        ``(effort_mean_float64, alpha_f32, beta_f32)``.
    """
    if conc_scale is None:
        conc_scale = float(weights["conc_scale"]) if "conc_scale" in weights else 1.0
    if variant is None:
        variant = str(np.asarray(weights["actor_variant"])) if "actor_variant" in weights else "t1"
    if variant not in ACTOR_VARIANTS:
        raise ValueError(f"unknown actor variant {variant!r}; known: {ACTOR_VARIANTS}")
    x = np.asarray(obs, dtype=np.float32)
    if variant == "t10":
        x = (x * np.array([1.0, D_FEATURE_SCALE_T10], dtype=np.float32)).astype(np.float32)
    act = (lambda a: np.maximum(a, np.float32(0.0))) if variant == "relu" else np.tanh
    h = act(x @ weights["actor.l1.weight"].T + weights["actor.l1.bias"])
    h = act(h @ weights["actor.l2.weight"].T + weights["actor.l2.bias"])
    z = h @ weights["actor.out.weight"].T + weights["actor.out.bias"]
    mu = np.clip(1.0 / (1.0 + np.exp(-z[:, 0])), mu_clamp, 1.0 - mu_clamp).astype(np.float32)
    zc = z[:, 1].astype(np.float32)
    c = (np.float32(c_min) + np.logaddexp(np.float32(0.0), zc)).astype(np.float32)
    if conc_scale != 1.0:
        c = (c * np.float32(conc_scale)).astype(np.float32)
    alpha = (mu * c).astype(np.float32)
    beta = ((np.float32(1.0) - mu) * c).astype(np.float32)
    mean = alpha.astype(float) / (alpha.astype(float) + beta.astype(float))
    return e_min + (e_max - e_min) * mean, alpha, beta
