#!/usr/bin/env python3
"""
True PPO agent for the two-stage two-player tournament.

Design goals:
- Beta policy on normalized effort a in (0,1), mapped to effort bounds per stage.
- Standard clipped PPO objective with GAE(λ), entropy bonus, and value loss.
- Single shared network for both stages; state encodes stage + context.
- Lightweight opponent-effort running average for Stage 1 state construction.

Interface expected by run/run_two_stage.py:
- act(stage=1, opp_signal=float, bounds=(lo1, hi1)) -> (effort, logp, value, a_norm, state)
- act_with_env_obs(obs_tensor, bounds1, bounds2, deterministic=False)
- store(state, action_norm, effort, logp, value, reward, done)
- update()
- opp_avg(stage) and update_opponent_avg(stage, opponent_effort)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 8, hidden: int = 64):
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
        return torch.distributions.Beta(alpha, beta), value


@dataclass
class PPOCfg:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 10
    minibatch_size: int = 256
    hidden: int = 64
    state_dim: int = 8
    # Opponent lag settings to avoid self-play gradient cancellation
    opponent_sync_interval: int = 1  # sync every N updates
    opponent_ema_tau: float = 0.0     # 0 -> hard copy; (0,1] -> EMA


class TwoStagePPOAgent:
    def __init__(
        self,
        *,
        effort_bounds_stage1: Tuple[float, float],
        effort_bounds_stage2: Tuple[float, float],
        q_value: float,
        w_h: float,
        w_l: float,
        k1: float,
        k2: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        lr: float = 3e-4,
        hidden: int = 64,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.low1, self.high1 = float(effort_bounds_stage1[0]), float(effort_bounds_stage1[1])
        self.low2, self.high2 = float(effort_bounds_stage2[0]), float(effort_bounds_stage2[1])
        self.q = float(q_value)
        self.w_gap = float(w_h) - float(w_l)
        self.k1 = float(k1)
        self.k2 = float(k2)

        self.cfg = PPOCfg(
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            lr=lr,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            hidden=hidden,
            state_dim=8,
        )

        self.net = ActorCritic(state_dim=self.cfg.state_dim, hidden=self.cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        # Lagged opponent net (frozen)
        self.opp_net = ActorCritic(state_dim=self.cfg.state_dim, hidden=self.cfg.hidden).to(self.device)
        self.opp_net.load_state_dict(self.net.state_dict())
        for p in self.opp_net.parameters():
            p.requires_grad_(False)

        # Opponent effort running averages per stage (only stage 1 used)
        self._opp_sum = {1: 0.0, 2: 0.0}
        self._opp_cnt = {1: 0, 2: 0}

        self.reset_storage()
        self.update_counter = 0

    # ----- feature helpers -----
    def _norm01(self, x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    def _stage1_state(self, opp_signal: float) -> torch.Tensor:
        # Construct an 8-D feature vector for Stage 1
        opp_norm = self._norm01(float(opp_signal), self.low1, self.high1)
        q_norm = self.q / 60.0
        wgap_norm = self.w_gap / 10.0
        k_norm = self.k1 / 1e-3
        # [stage_id, won_s1, my_e1, opp_e1, p_win_est, q_norm, wgap_norm, k_norm]
        s = torch.tensor([1.0, 0.0, 0.0, opp_norm, 0.0, q_norm, wgap_norm, k_norm], dtype=torch.float32, device=self.device)
        return s.unsqueeze(0)

    def _stage2_state_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape expected [5]: [stage(=2), won_s1, my_e1, opp_e1, p_win_est]
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        won = obs[:, 1]
        my_e1 = obs[:, 2]
        opp_e1 = obs[:, 3]
        p_est = obs[:, 4]
        # normalize efforts by Stage-1 bounds
        my_e1_n = torch.clamp((my_e1 - self.low1) / max(1e-6, (self.high1 - self.low1)), 0.0, 1.0)
        opp_e1_n = torch.clamp((opp_e1 - self.low1) / max(1e-6, (self.high1 - self.low1)), 0.0, 1.0)
        q_norm = torch.full_like(my_e1_n, self.q / 60.0)
        wgap_norm = torch.full_like(my_e1_n, self.w_gap / 10.0)
        k_norm = torch.full_like(my_e1_n, self.k2 / 1e-3)
        stage = torch.full_like(my_e1_n, 2.0)
        s = torch.stack([stage, won, my_e1_n, opp_e1_n, p_est, q_norm, wgap_norm, k_norm], dim=1)
        return s

    # ----- acting -----
    def _act_with_state(self, state: torch.Tensor, bounds: Tuple[float, float], deterministic: bool = False):
        dist, value = self.net.dist(state)
        if deterministic:
            # use Beta mean when alpha,beta > 1 else sample and average
            alpha = dist.concentration1
            beta = dist.concentration0
            use_mean = (alpha > 1.0) & (beta > 1.0)
            a = torch.where(use_mean, (alpha - 1.0) / (alpha + beta - 2.0), dist.sample())
        else:
            a = dist.sample()
        logp = dist.log_prob(a).squeeze(-1)
        lo, hi = float(bounds[0]), float(bounds[1])
        effort = lo + a.squeeze(-1) * (hi - lo)
        return effort.detach().cpu().item(), logp.detach(), value.detach(), a.detach(), state.detach()

    def act(self, *, stage: int, opp_signal: float, bounds: Tuple[float, float], deterministic: bool = False):
        state = self._stage1_state(opp_signal)
        return self._act_with_state(state, bounds, deterministic)

    def act_opponent(self, *, stage: int, opp_signal: float, bounds: Tuple[float, float], deterministic: bool = False):
        state = self._stage1_state(opp_signal)
        dist, value = self.opp_net.dist(state)
        if deterministic:
            alpha = dist.concentration1
            beta = dist.concentration0
            use_mean = (alpha > 1.0) & (beta > 1.0)
            a = torch.where(use_mean, (alpha - 1.0) / (alpha + beta - 2.0), dist.sample())
        else:
            a = dist.sample()
        logp = dist.log_prob(a).squeeze(-1)
        lo, hi = float(bounds[0]), float(bounds[1])
        effort = lo + a.squeeze(-1) * (hi - lo)
        return effort.detach().cpu().item(), logp.detach(), value.detach(), a.detach(), state.detach()

    def act_with_env_obs(self, obs: torch.Tensor, bounds_stage1: Tuple[float, float], bounds_stage2: Tuple[float, float], deterministic: bool = False):
        state = self._stage2_state_from_obs(obs)
        return self._act_with_state(state, bounds_stage2, deterministic)

    def act_with_env_obs_opponent(self, obs: torch.Tensor, bounds_stage1: Tuple[float, float], bounds_stage2: Tuple[float, float], deterministic: bool = False):
        state = self._stage2_state_from_obs(obs)
        dist, value = self.opp_net.dist(state)
        if deterministic:
            alpha = dist.concentration1
            beta = dist.concentration0
            use_mean = (alpha > 1.0) & (beta > 1.0)
            a = torch.where(use_mean, (alpha - 1.0) / (alpha + beta - 2.0), dist.sample())
        else:
            a = dist.sample()
        logp = dist.log_prob(a).squeeze(-1)
        lo, hi = float(bounds_stage2[0]), float(bounds_stage2[1])
        effort = lo + a.squeeze(-1) * (hi - lo)
        return effort.detach().cpu().item(), logp.detach(), value.detach(), a.detach(), state.detach()

    # ----- storage -----
    def reset_storage(self):
        self.storage = {k: [] for k in ("states", "actions_norm", "logp", "rewards", "values", "dones")}

    def store(self, state: torch.Tensor, action_norm: torch.Tensor, effort: float, logp: torch.Tensor, value: torch.Tensor, reward: float, done: bool):
        self.storage["states"].append(state.squeeze(0).cpu())
        self.storage["actions_norm"].append(action_norm.squeeze().cpu())
        self.storage["logp"].append(logp.squeeze().cpu())
        self.storage["rewards"].append(torch.as_tensor(reward, dtype=torch.float32))
        self.storage["values"].append(value.squeeze().cpu())
        self.storage["dones"].append(torch.as_tensor(done, dtype=torch.bool))

    def _compute_gae(self):
        rewards = torch.stack(self.storage["rewards"]).to(self.device).view(-1)
        values = torch.stack(self.storage["values"]).to(self.device).view(-1)
        dones = torch.stack(self.storage["dones"]).to(self.device).view(-1)
        T = rewards.size(0)
        adv = torch.zeros(T, device=self.device)
        lastgaelam = torch.zeros((), device=self.device)
        next_value = torch.zeros((), device=self.device)
        for t in reversed(range(T)):
            mask = (~dones[t]).float()
            delta = rewards[t] + self.cfg.gamma * next_value * mask - values[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * lastgaelam
            adv[t] = lastgaelam
            next_value = values[t]
        returns = adv + values
        # advantage normalization
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    # ----- update -----
    def update(self):
        if len(self.storage["states"]) == 0:
            return
        states = torch.stack(self.storage["states"]).to(self.device)
        actions = torch.stack(self.storage["actions_norm"]).unsqueeze(-1).to(self.device)
        old_logp = torch.stack(self.storage["logp"]).to(self.device)
        adv, ret = self._compute_gae()

        n = states.size(0)
        idx = np.arange(n)
        for _ in range(self.cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = idx[start:start + self.cfg.minibatch_size]
                s = states[mb]
                a = actions[mb]
                oldlp = old_logp[mb]
                adv_mb = adv[mb]
                ret_mb = ret[mb]

                dist, values = self.net.dist(s)
                logp = dist.log_prob(a).squeeze(-1)
                ratio = torch.exp(logp - oldlp)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().mean()
                if values.dim() != 1:
                    values = values.view(-1)
                value_loss = F.mse_loss(values, ret_mb)
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

        self.reset_storage()
        # Sync lagged opponent
        self.update_counter += 1
        if self.cfg.opponent_sync_interval > 0 and (self.update_counter % self.cfg.opponent_sync_interval == 0):
            tau = float(self.cfg.opponent_ema_tau)
            if tau <= 0.0:
                self.opp_net.load_state_dict(self.net.state_dict())
            else:
                with torch.no_grad():
                    for p_opp, p_net in zip(self.opp_net.parameters(), self.net.parameters()):
                        p_opp.mul_(1.0 - tau).add_(p_net, alpha=tau)

    # ----- opponent averages -----
    def opp_avg(self, stage: int) -> float:
        cnt = self._opp_cnt.get(stage, 0)
        if cnt <= 0:
            # neutral default at mid of bounds
            if stage == 1:
                return 0.5 * (self.low1 + self.high1)
            return 0.5 * (self.low2 + self.high2)
        return self._opp_sum[stage] / max(1, cnt)

    def update_opponent_avg(self, *, stage: int, opponent_effort: float):
        self._opp_sum[stage] = self._opp_sum.get(stage, 0.0) + float(opponent_effort)
        self._opp_cnt[stage] = self._opp_cnt.get(stage, 0) + 1
