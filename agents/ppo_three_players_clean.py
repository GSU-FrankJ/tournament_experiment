#!/usr/bin/env python3
"""
Clean PPO agent for symmetric three-player one-stage tournament (bandit).

True PPO features:
- Clipped objective with advantage normalization
- Value loss and entropy bonus
- Minibatch SGD over collected rollouts
- Beta policy over normalized action a in (0,1), mapped to effort bounds

Self-play setup:
- One shared policy controls three identical players
- Each episode samples three independent actions; all three transitions are
  stored as separate experiences for PPO
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 3, hidden: int = 64):
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
    epochs: int = 10
    minibatch_size: int = 256
    state_dim: int = 3
    hidden: int = 64
    opponent_sync_interval: int = 1
    opponent_ema_tau: float = 0.0


class PPOThreePlayersBandit:
    def __init__(self, effort_bounds: Tuple[float, float], cfg: PPOConfig = PPOConfig(), device: str | None = None):
        self.low, self.high = float(effort_bounds[0]), float(effort_bounds[1])
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net = ActorCritic(state_dim=cfg.state_dim, hidden=cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.opp_net = ActorCritic(state_dim=cfg.state_dim, hidden=cfg.hidden).to(self.device)
        self.opp_net.load_state_dict(self.net.state_dict())
        for p in self.opp_net.parameters():
            p.requires_grad_(False)
        self.update_counter = 0
        self.reset_storage()

    # ---- helpers ----
    def _state_tensor(self, state) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            t = state
        else:
            t = torch.tensor(state, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.to(self.device)

    def act(self, state: torch.Tensor):
        """Sample normalized action a in (0,1); map to effort within bounds."""
        dist, value = self.net.dist(state)
        a_norm = dist.sample()
        logp = dist.log_prob(a_norm).squeeze(-1)
        effort = self.low + a_norm.squeeze(-1) * (self.high - self.low)
        return a_norm.detach(), effort.detach(), logp.detach(), value.detach()

    def evaluate_actions(self, states: torch.Tensor, actions_norm: torch.Tensor):
        dist, values = self.net.dist(states)
        logp = dist.log_prob(actions_norm).squeeze(-1)
        entropy = dist.entropy().mean()
        return logp, entropy, values.squeeze(-1)

    def act_opponent(self, state: torch.Tensor):
        with torch.no_grad():
            dist, value = self.opp_net.dist(state)
            a_norm = dist.sample()
            logp = dist.log_prob(a_norm).squeeze(-1)
            effort = self.low + a_norm.squeeze(-1) * (self.high - self.low)
        return a_norm.detach(), effort.detach(), logp.detach(), value.detach()

    # ---- storage / GAE ----
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = states.size(0)
        idx = np.arange(N)
        for _ in range(self.cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.cfg.minibatch_size):
                mb = idx[start:start + self.cfg.minibatch_size]
                mb_states = states[mb]
                mb_actions = actions_norm[mb]
                mb_adv = advantages[mb]
                mb_returns = returns[mb]
                mb_old_logp = old_logp[mb]

                logp, entropy, values = self.evaluate_actions(mb_states, mb_actions)
                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.view(-1), mb_returns.view(-1))
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

        self.reset_storage()
        self.update_counter += 1
        if (
            self.cfg.opponent_sync_interval > 0
            and (self.update_counter % self.cfg.opponent_sync_interval == 0)
        ):
            tau = float(self.cfg.opponent_ema_tau)
            if tau <= 0.0:
                self.opp_net.load_state_dict(self.net.state_dict())
            else:
                with torch.no_grad():
                    for p_opp, p_net in zip(self.opp_net.parameters(), self.net.parameters()):
                        p_opp.mul_(1.0 - tau).add_(p_net, alpha=tau)

    # ---- convenience ----
    def state_from_params(self, *, q: float, k: float, w_h: float, w_l: float) -> torch.Tensor:
        q_norm = float(q) / 60.0
        k_norm = float(k) / 1e-3
        wgap_norm = float(w_h - w_l) / 10.0
        t = torch.tensor([q_norm, k_norm, wgap_norm], dtype=torch.float32, device=self.device)
        return t.unsqueeze(0)
