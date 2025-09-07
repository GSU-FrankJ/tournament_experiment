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


class PPOTwoPlayersBandit:
    def __init__(self, effort_bounds: Tuple[float, float], cfg: PPOConfig = PPOConfig(), device: str = None):
        self.low, self.high = float(effort_bounds[0]), float(effort_bounds[1])
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net = ActorCritic(state_dim=cfg.state_dim, hidden=cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

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

    def act(self, state: torch.Tensor):
        dist, value = self.net.dist(state)
        a_norm = dist.sample()
        logp = dist.log_prob(a_norm).squeeze(-1)
        # map to effort
        effort = self.low + a_norm.squeeze(-1) * (self.high - self.low)
        return a_norm.detach(), effort.detach(), logp.detach(), value.detach()

    def evaluate_actions(self, states: torch.Tensor, actions_norm: torch.Tensor):
        dist, values = self.net.dist(states)
        logp = dist.log_prob(actions_norm).squeeze(-1)
        entropy = dist.entropy().mean()
        return logp, entropy, values.squeeze(-1)

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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)
        idx = np.arange(dataset_size)
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

    # ---- utility ----
    def state_from_params(self, *, q: float, k: float, w_h: float, w_l: float) -> torch.Tensor:
        # Normalize features to roughly [0,1]
        q_norm = float(q) / 60.0  # assumes q up to ~60
        k_norm = float(k) / 1e-3  # k around 4e-4 => ~0.4
        wgap_norm = float(w_h - w_l) / 10.0  # prize gap scaled by 10
        s = torch.tensor([q_norm, k_norm, wgap_norm], dtype=torch.float32, device=self.device)
        return s.unsqueeze(0)
