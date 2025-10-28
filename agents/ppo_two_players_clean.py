#!/usr/bin/env python3
"""
Clean PPO agent for symmetric two-player one-stage tournament (bandit setting).

Key points:
- Uses Beta policy on normalized effort a in (0,1); maps to effort via
  e = low + a * (high - low).
- Computes log_prob on the Beta distribution for PPO ratio.
- Single-step episodes (bandit), but code supports generic GAE with done flags.
- Designed for self-play: at each env step we sample two independent actions
  from the same policy and treat the two players' experiences as separate
  transitions with shared parameters.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 1, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.alpha_head = nn.Linear(hidden, 1)
        self.beta_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        # Softplus -> (0, +inf); add 1 for numerical stability
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta = F.softplus(self.beta_head(h)) + 1.0
        value = self.value_head(h).squeeze(-1)
        return alpha, beta, value

    def dist(self, x: torch.Tensor):
        alpha, beta, value = self.forward(x)
        dist = torch.distributions.Beta(alpha, beta)
        return dist, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    steps_per_update: int = 2048
    epochs: int = 15
    minibatch_size: int = 256
    state_dim: int = 3  # [q_norm, k_norm, wgap_norm]
    hidden: int = 64
    # Opponent policy lag (self-play stabilization)
    opponent_mode: str = "ema"  # ["ema", "periodic", "snapshot"]
    opponent_sync_interval: int = 10  # sync every N PPO updates
    opponent_ema_tau: float = 0.05    # EMA coefficient
    opponent_snapshot_keep: int = 10
    opponent_history_sample_p: float = 0.5


class PPOTwoPlayersBandit:
    def __init__(self, effort_bounds: Tuple[float, float], cfg: PPOConfig = PPOConfig(), device: str = None):
        self.low, self.high = float(effort_bounds[0]), float(effort_bounds[1])
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net = ActorCritic(state_dim=cfg.state_dim, hidden=cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        # Lag opponent / history configuration
        self.opponent_mode = getattr(cfg, "opponent_mode", "ema")
        self.opponent_sync_interval = max(0, int(getattr(cfg, "opponent_sync_interval", 0) or 0))
        self.opponent_ema_tau = float(getattr(cfg, "opponent_ema_tau", 0.0))
        self.opponent_history_sample_p = max(0.0, min(1.0, float(getattr(cfg, "opponent_history_sample_p", 0.0))))
        self.opponent_snapshot_keep = max(0, int(getattr(cfg, "opponent_snapshot_keep", 0) or 0))
        if self.opponent_mode not in ("ema", "periodic", "snapshot"):
            self.opponent_mode = "periodic"

        # Lagged opponent network (frozen copy)
        self.opponent_policy = deepcopy(self.net).to(self.device)
        for p in self.opponent_policy.parameters():
            p.requires_grad_(False)
        self.opponent_policy.eval()

        # Historical snapshot pool (only populated for snapshot mode or explicit requests)
        history_maxlen = self.opponent_snapshot_keep if self.opponent_snapshot_keep > 0 else None
        self._opponent_history: deque[ActorCritic] = deque(maxlen=history_maxlen)

        # Tracking counters
        self._updates = 0
        self._last_sync_step = -1

        if self.opponent_mode == "snapshot" and self.opponent_snapshot_keep != 0:
            self._snapshot_current()

        # rollout storage
        self.reset_storage()

    # ---- action/forward helpers ----
    def _state_tensor(self, state: np.ndarray | List[float] | torch.Tensor) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            t = state
        else:
            t = torch.tensor(state, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.to(self.device)

    def _act_with_net(self, net: ActorCritic, state: torch.Tensor):
        dist, value = net.dist(state)
        a_norm = dist.sample()
        eps = 1e-6
        a_safe = a_norm.clamp(eps, 1.0 - eps)
        logp = dist.log_prob(a_safe).squeeze(-1)
        # map to effort
        effort = self.low + a_safe.squeeze(-1) * (self.high - self.low)
        return a_safe.detach(), effort.detach(), logp.detach(), value.detach()

    def act(self, state: torch.Tensor):
        return self._act_with_net(self.net, state)

    def act_opponent(self, state: torch.Tensor):
        opp_net = self._sample_opponent_policy_for_play()
        with torch.no_grad():
            dist, _ = opp_net.dist(state.to(self.device))
            eps = 1e-6
            a_norm = dist.sample()
            a_safe = a_norm.clamp(eps, 1.0 - eps)
            logp = dist.log_prob(a_safe).squeeze(-1)
            effort = self.low + a_safe.squeeze(-1) * (self.high - self.low)
        return a_safe.detach(), effort.detach(), logp.detach(), None

    def _sample_opponent_policy_for_play(self) -> ActorCritic:
        history_available = len(self._opponent_history) > 0
        use_history = (
            history_available
            and self.opponent_history_sample_p > 0.0
            and float(np.random.rand()) < float(self.opponent_history_sample_p)
        )
        if use_history:
            idx = int(np.random.randint(len(self._opponent_history)))
            return self._opponent_history[idx]
        return self.opponent_policy

    @torch.no_grad()
    def _ema_update(self, tau: float):
        tau = float(max(0.0, min(1.0, tau)))
        if tau <= 0.0:
            self._hard_copy_to_opponent()
            return
        for p_opp, p_cur in zip(self.opponent_policy.parameters(), self.net.parameters()):
            p_opp.data.lerp_(p_cur.data, tau)

    @torch.no_grad()
    def _hard_copy_to_opponent(self):
        for p_opp, p_cur in zip(self.opponent_policy.parameters(), self.net.parameters()):
            p_opp.data.copy_(p_cur.data)

    @torch.no_grad()
    def _snapshot_current(self):
        # Sync the managed opponent copy before saving and append frozen snapshot
        self._hard_copy_to_opponent()
        snap = deepcopy(self.opponent_policy)
        for p in snap.parameters():
            p.requires_grad_(False)
        snap.eval()
        self._opponent_history.append(snap)

    def evaluate_actions(self, states: torch.Tensor, actions_norm: torch.Tensor):
        dist, values = self.net.dist(states)
        eps = 1e-6
        a_safe = actions_norm.clamp(eps, 1.0 - eps)
        logp = dist.log_prob(a_safe).squeeze(-1)
        entropy = dist.entropy().mean()
        return logp, entropy, values.squeeze(-1)

    @torch.no_grad()
    def mean_action_norm(self, state: torch.Tensor) -> torch.Tensor:
        dist, _ = self.net.dist(state.to(self.device))
        eps = 1e-6
        return dist.mean.clamp(eps, 1.0 - eps)

    @torch.no_grad()
    def mean_effort(self, state: torch.Tensor) -> float:
        a_mean = self.mean_action_norm(state).squeeze().item()
        return float(self.low + a_mean * (self.high - self.low))

    @torch.no_grad()
    def value_only(self, state: torch.Tensor):
        _, value = self.net.dist(state.to(self.device))
        return value.detach()

    # ---- storage and advantage ----
    def reset_storage(self):
        self.storage: Dict[str, List[torch.Tensor]] = {
            "states": [],
            "actions_norm": [],
            "logp": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

    def store(self, state, action_norm, logp, reward, value, done):
        self.storage["states"].append(state.squeeze(0).detach().cpu())
        self.storage["actions_norm"].append(action_norm.squeeze().detach().cpu())
        self.storage["logp"].append(logp.squeeze().detach().cpu())
        self.storage["rewards"].append(torch.as_tensor(reward, dtype=torch.float32))
        self.storage["values"].append(value.detach().cpu())
        self.storage["dones"].append(torch.as_tensor(done, dtype=torch.bool))

    def _compute_gae(self):
        rewards = torch.stack(self.storage["rewards"]).to(self.device).view(-1)
        values = torch.stack(self.storage["values"]).to(self.device).view(-1)
        dones = torch.stack(self.storage["dones"]).to(self.device).view(-1)

        T = rewards.size(0)
        advantages = torch.zeros(T, device=self.device)
        lastgaelam = torch.zeros((), device=self.device)
        next_value = torch.zeros((), device=self.device)
        for t in reversed(range(T)):
            mask = (~dones[t]).float()
            delta = rewards[t] + self.cfg.gamma * next_value * mask - values[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * lastgaelam
            advantages[t] = lastgaelam
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    # ---- update ----
    def update(self):
        states = torch.stack(self.storage["states"]).to(self.device)
        actions_norm = torch.stack(self.storage["actions_norm"]).unsqueeze(-1).to(self.device)
        old_logp = torch.stack(self.storage["logp"]).to(self.device)
        advantages, returns = self._compute_gae()
        adv_mean = float(advantages.mean().item())
        adv_std = float(advantages.std(unbiased=False).item())
        # Sanity shape checks
        # All should be [T] except states [T, state_dim] and actions_norm [T,1]
        if states.dim() != 2:
            raise RuntimeError(f"states shape unexpected: {states.shape}")
        if actions_norm.dim() != 2 or actions_norm.size(1) != 1:
            raise RuntimeError(f"actions_norm shape unexpected: {actions_norm.shape}")
        if old_logp.dim() != 1:
            raise RuntimeError(f"old_logp shape unexpected: {old_logp.shape}")
        if advantages.dim() != 1 or returns.dim() != 1:
            raise RuntimeError(f"returns/advantages shapes unexpected: {advantages.shape}, {returns.shape}")
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        dataset_size = states.size(0)
        idx = np.arange(dataset_size)
        kl_values: List[float] = []
        entropy_values: List[float] = []
        for _ in range(self.cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, dataset_size, self.cfg.minibatch_size):
                mb_idx = idx[start:start + self.cfg.minibatch_size]
                mb_states = states[mb_idx]
                mb_actions = actions_norm[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_logp = old_logp[mb_idx]

                logp, entropy, values = self.evaluate_actions(mb_states, mb_actions)
                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                approx_kl = (mb_old_logp - logp).mean()
                kl_values.append(float(approx_kl.detach().cpu().item()))
                entropy_values.append(float(entropy.detach().cpu().item()))

                # ensure 1D shapes
                if values.dim() != 1:
                    values = values.view(-1)
                if mb_returns.dim() != 1:
                    mb_returns = mb_returns.view(-1)
                value_loss = F.mse_loss(values, mb_returns)
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

        self.reset_storage()
        # Update opponent with lag according to configured mode
        self._updates += 1
        if self.opponent_sync_interval > 0 and (self._updates % self.opponent_sync_interval == 0):
            if self.opponent_mode == "ema":
                self._ema_update(self.opponent_ema_tau)
            elif self.opponent_mode == "periodic":
                self._hard_copy_to_opponent()
            elif self.opponent_mode == "snapshot":
                self._snapshot_current()
            else:
                # Fallback to hard copy for unknown modes
                self._hard_copy_to_opponent()
            self._last_sync_step = self._updates

        metrics = {
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "approx_kl": float(np.mean(kl_values)) if kl_values else 0.0,
            "batch_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "opponent_history_size": float(len(self._opponent_history)),
            "opponent_last_sync": float(self._last_sync_step),
        }
        return metrics

    # ---- utility ----
    def state_from_params(self, *, q: float, k: float, w_h: float, w_l: float) -> torch.Tensor:
        # Normalize features to roughly [0,1]
        q_norm = float(q) / 60.0  # assumes q up to ~60
        k_norm = float(k) / 1e-3  # k around 4e-4 => ~0.4
        wgap_norm = float(w_h - w_l) / 10.0  # prize gap scaled by 10
        s = torch.tensor([q_norm, k_norm, wgap_norm], dtype=torch.float32, device=self.device)
        return s.unsqueeze(0)
