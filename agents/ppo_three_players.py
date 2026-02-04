#!/usr/bin/env python3
"""
Clean PPO agent for symmetric three-player one-stage tournament (bandit setting).

Key points:
- Uses Beta policy on normalized effort a in (0,1); maps to effort via
  e = low + a * (high - low).
- Computes log_prob on the Beta distribution for PPO ratio.
- Single-step episodes (bandit), but code supports generic GAE with done flags.
- Designed for pure self-play: at each env step we sample three independent actions
  from the same policy and treat the three players' experiences as separate
  transitions with shared parameters.

Key differences from two-player version:
- No opponent policy tracking (no opponent_net, sync_opponent, snapshots)
- Stores 3 transitions per environment step (vs 2 for two-player)
- All three players share the same policy (symmetric equilibrium)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """Standard actor-critic with separate alpha/beta heads for Beta distribution."""
    
    def __init__(self, state_dim: int = 1, hidden: int = 64):       
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Policy heads for Beta distribution parameters
        self.alpha_head = nn.Linear(hidden, 1)
        self.beta_head = nn.Linear(hidden, 1)
        # Value function head
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        """Forward pass returning alpha, beta, and value."""
        h = self.shared(x)
        # Softplus ensures positive params; +1 for numerical stability
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta = F.softplus(self.beta_head(h)) + 1.0
        value = self.value_head(h).squeeze(-1)
        return alpha, beta, value

    def dist(
        self,
        x: torch.Tensor,
        *,
        theory_align: bool = False,
        theory_align_conc_min: float = 0.0,
    ):
        """Get Beta distribution and value for given state."""
        alpha, beta, value = self.forward(x)
        if theory_align and theory_align_conc_min and theory_align_conc_min > 0.0:
            conc = alpha + beta
            scale = torch.clamp(theory_align_conc_min / (conc + 1e-8), min=1.0)
            alpha = alpha * scale
            beta = beta * scale
        dist = torch.distributions.Beta(alpha, beta)
        return dist, value


class ActorCriticMeanConc(nn.Module):
    """Actor-critic with mean+concentration parameterization for Beta distribution.
    
    This parameterization is more interpretable:
    - mean: E[X] = alpha / (alpha + beta)
    - concentration: alpha + beta (controls variance)
    """
    
    def __init__(
        self,
        state_dim: int = 1,
        hidden: int = 64,
        conc_min: float = 1.0,
        conc_scale: float = 1.0,
        conc_max: Optional[float] = None,
        init_bias_mean: Optional[float] = None,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.conc_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)
        self.conc_min = float(conc_min)
        self.conc_scale = float(conc_scale)
        self.conc_max = None if conc_max is None else float(conc_max)
        
        # If specified, initialize mean_head bias to target an initial mean
        if init_bias_mean is not None:
            # mean = sigmoid(mean_head(h))
            # To get initial mean ≈ init_bias_mean, set bias = logit(init_bias_mean)
            init_bias_mean = max(0.01, min(0.99, float(init_bias_mean)))
            target_logit = np.log(init_bias_mean / (1.0 - init_bias_mean))
            nn.init.constant_(self.mean_head.bias, target_logit)
            nn.init.normal_(self.mean_head.weight, mean=0.0, std=0.01)
            print(f"[ActorCriticMeanConc] Initialized mean_head bias for target mean={init_bias_mean:.3f} (logit={target_logit:.3f})")

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        mean = torch.sigmoid(self.mean_head(h))
        scale = max(self.conc_scale, 1e-8)
        conc = F.softplus(self.conc_head(h)) * scale + self.conc_min
        if self.conc_max is not None:
            conc = torch.clamp(conc, max=self.conc_max)
        # Convert mean+concentration to alpha+beta
        alpha = mean * conc
        beta = (1.0 - mean) * conc
        value = self.value_head(h).squeeze(-1)
        return alpha, beta, value

    def dist(self, x: torch.Tensor):
        alpha, beta, value = self.forward(x)
        dist = torch.distributions.Beta(alpha, beta)
        return dist, value


@dataclass
class PPOConfig:
    """Configuration for PPO agent (simplified for three-player self-play)."""
    
    # Core PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Rollout and update settings
    steps_per_update: int = 2048
    epochs: int = 15
    minibatch_size: int = 256
    
    # Network architecture
    state_dim: int = 3  # [q_norm, k_norm, wgap_norm]
    hidden: int = 64
    
    # Early-stop settings (default OFF)
    kl_early_stop: bool = False
    kl_stop_patience: int = 1
    kl_stop_threshold: Optional[float] = None
    ratio_stop_threshold: Optional[float] = None
    target_kl: float = 0.08
    
    # Theory-align experiment settings (default OFF)
    theory_align: bool = False
    theory_align_conc_min: float = 0.0
    theory_align_conc_weight: float = 0.0
    theory_align_v2: bool = False
    theory_align_v2_conc_min: float = 1.0
    theory_align_v2_conc_scale: float = 1.0
    theory_align_v2_conc_max: Optional[float] = None
    theory_align_v2_var_coef: float = 0.0
    theory_align_v2_br_coef: float = 0.0


class PPOThreePlayersBandit:
    """PPO agent for three-player symmetric self-play tournament.
    
    All three players share the same policy network. At each environment step,
    we sample three independent actions and treat each player's experience as
    a separate transition.
    """
    
    def __init__(
        self,
        effort_bounds: Tuple[float, float],
        cfg: PPOConfig = PPOConfig(),
        device: str = None,
        init_bias_mean: Optional[float] = None,
    ):
        self.low, self.high = float(effort_bounds[0]), float(effort_bounds[1])
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network based on theory_align_v2 setting
        self.use_theory_align_v2 = bool(getattr(cfg, "theory_align_v2", False))
        if self.use_theory_align_v2:
            conc_min = float(getattr(cfg, "theory_align_v2_conc_min", 1.0))
            conc_scale = float(getattr(cfg, "theory_align_v2_conc_scale", 1.0))
            conc_max = getattr(cfg, "theory_align_v2_conc_max", None)
            self.net = ActorCriticMeanConc(
                state_dim=cfg.state_dim,
                hidden=cfg.hidden,
                conc_min=conc_min,
                conc_scale=conc_scale,
                conc_max=conc_max,
                init_bias_mean=init_bias_mean,
            ).to(self.device)
        else:
            self.net = ActorCritic(state_dim=cfg.state_dim, hidden=cfg.hidden).to(self.device)
        
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        
        # Update counter for scheduling
        self._updates = 0
        
        # Rollout storage
        self.reset_storage()

    # ---- Action/forward helpers ----
    def _state_tensor(self, state: np.ndarray | List[float] | torch.Tensor) -> torch.Tensor:
        """Convert state to tensor on device."""
        if isinstance(state, torch.Tensor):
            t = state
        else:
            t = torch.tensor(state, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.to(self.device)

    def dist(self, state: torch.Tensor, *, net: Optional[nn.Module] = None):
        """Get policy distribution and value for given state."""
        target_net = net if net is not None else self.net
        if self.use_theory_align_v2:
            return target_net.dist(state)
        if self.cfg.theory_align and self.cfg.theory_align_conc_min > 0.0:
            return target_net.dist(
                state,
                theory_align=True,
                theory_align_conc_min=float(self.cfg.theory_align_conc_min),
            )
        return target_net.dist(state)

    def act(self, state: torch.Tensor):
        """Sample action from policy.
        
        Returns:
            a_safe: Normalized action in (0, 1)
            effort: Mapped effort in [low, high]
            logp: Log probability of action
            value: Value estimate
        """
        dist, value = self.dist(state)
        a_norm = dist.sample()
        eps = 1e-6
        a_safe = a_norm.clamp(eps, 1.0 - eps)
        logp = dist.log_prob(a_safe).squeeze(-1)
        # Map normalized action to effort space
        effort = self.low + a_safe.squeeze(-1) * (self.high - self.low)
        return a_safe.detach(), effort.detach(), logp.detach(), value.detach()

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions_norm: torch.Tensor,
        *,
        return_conc: bool = False,
        return_dist: bool = False,
    ):
        """Evaluate log prob and entropy for given state-action pairs."""
        dist, values = self.dist(states)
        eps = 1e-6
        a_safe = actions_norm.clamp(eps, 1.0 - eps)
        logp = dist.log_prob(a_safe).squeeze(-1)
        entropy = dist.entropy().mean()
        
        if return_conc and return_dist:
            conc = dist.concentration1 + dist.concentration0
            return logp, entropy, values.squeeze(-1), conc, dist
        if return_conc:
            conc = dist.concentration1 + dist.concentration0
            return logp, entropy, values.squeeze(-1), conc
        if return_dist:
            return logp, entropy, values.squeeze(-1), dist
        return logp, entropy, values.squeeze(-1)

    @torch.no_grad()
    def mean_action_norm(self, state: torch.Tensor) -> torch.Tensor:
        """Get mean of policy distribution (normalized action)."""
        dist, _ = self.dist(state.to(self.device))
        eps = 1e-6
        return dist.mean.clamp(eps, 1.0 - eps)

    @torch.no_grad()
    def mean_effort(self, state: torch.Tensor) -> float:
        """Get mean effort from policy."""
        a_mean = self.mean_action_norm(state).squeeze().item()
        return float(self.low + a_mean * (self.high - self.low))

    @torch.no_grad()
    def value_only(self, state: torch.Tensor):
        """Get value estimate only."""
        _, value = self.dist(state.to(self.device))
        return value.detach()

    # ---- Storage and advantage computation ----
    def reset_storage(self):
        """Reset rollout storage for new batch collection."""
        self.storage: Dict[str, List[torch.Tensor]] = {
            "states": [],
            "actions_norm": [],
            "logp": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

    def store(self, state, action_norm, logp, reward, value, done):
        """Store a single transition."""
        self.storage["states"].append(state.squeeze(0).detach().cpu())
        self.storage["actions_norm"].append(action_norm.squeeze().detach().cpu())
        self.storage["logp"].append(logp.squeeze().detach().cpu())
        self.storage["rewards"].append(torch.as_tensor(reward, dtype=torch.float32))
        self.storage["values"].append(value.detach().cpu())
        self.storage["dones"].append(torch.as_tensor(done, dtype=torch.bool))

    def _compute_gae(self):
        """Compute Generalized Advantage Estimation."""
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

    # ---- PPO update ----
    def update(self):
        """Perform PPO update on collected rollouts.
        
        Returns:
            metrics: Dictionary of training diagnostics
        """
        states = torch.stack(self.storage["states"]).to(self.device)
        actions_norm = torch.stack(self.storage["actions_norm"]).unsqueeze(-1).to(self.device)
        old_logp = torch.stack(self.storage["logp"]).to(self.device)
        rewards_tensor = torch.stack(self.storage["rewards"]).to(self.device).view(-1)
        advantages, returns = self._compute_gae()
        
        # Compute raw advantage stats BEFORE normalization (for diagnostics)
        adv_mean = float(advantages.mean().item())
        adv_std = float(advantages.std(unbiased=False).item())
        
        # Compute state/reward/value stats
        state_mean = float(states.mean().item())
        state_std = float(states.std(unbiased=False).item())
        reward_mean = float(rewards_tensor.mean().item())
        reward_std = float(rewards_tensor.std(unbiased=False).item())
        values_tensor = torch.stack(self.storage["values"]).to(self.device).view(-1)
        value_mean = float(values_tensor.mean().item())
        value_std = float(values_tensor.std(unbiased=False).item())
        
        # Sanity shape checks
        if states.dim() != 2:
            raise RuntimeError(f"states shape unexpected: {states.shape}")
        if actions_norm.dim() != 2 or actions_norm.size(1) != 1:
            raise RuntimeError(f"actions_norm shape unexpected: {actions_norm.shape}")
        if old_logp.dim() != 1:
            raise RuntimeError(f"old_logp shape unexpected: {old_logp.shape}")
        if advantages.dim() != 1 or returns.dim() != 1:
            raise RuntimeError(f"returns/advantages shapes unexpected: {advantages.shape}, {returns.shape}")
        
        # Normalize advantages (standard PPO practice)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        adv_norm_std = float(advantages.std(unbiased=False).item())

        dataset_size = states.size(0)
        clip_eps = float(self.cfg.clip_eps)
        idx = np.arange(dataset_size)
        kl_values: List[float] = []
        entropy_values: List[float] = []

        # --- Early-stop bookkeeping ---
        early_stop_triggered = False
        early_stop_reason = ""
        epoch_idx_triggered = -1
        mb_idx_triggered = -1
        breach_count = 0
        minibatches_completed = 0
        minibatches_total_planned = int(np.ceil(dataset_size / float(self.cfg.minibatch_size))) * self.cfg.epochs

        # Per-update thresholds (auto-derived if None)
        kl_stop_threshold = (
            self.cfg.kl_stop_threshold
            if self.cfg.kl_stop_threshold is not None
            else 2.0 * float(self.cfg.target_kl)
        )
        ratio_stop_threshold = (
            self.cfg.ratio_stop_threshold
            if self.cfg.ratio_stop_threshold is not None
            else 1.0 + 2.0 * clip_eps
        )

        # Per-minibatch accumulators for diagnostics
        policy_loss_list: List[float] = []
        value_loss_list: List[float] = []
        entropy_list: List[float] = []
        approx_kl_list: List[float] = []
        kl_proxy_list: List[float] = []
        kl_ms_list: List[float] = []
        clip_frac_list: List[float] = []
        ratio_mean_list: List[float] = []
        ratio_std_list: List[float] = []
        ratio_max_list: List[float] = []
        log_ratio_abs_mean_list: List[float] = []
        log_ratio_mean_list: List[float] = []
        log_ratio_std_list: List[float] = []
        grad_norm_list: List[float] = []

        for epoch_idx in range(self.cfg.epochs):
            np.random.shuffle(idx)
            for mb_local_idx, start in enumerate(range(0, dataset_size, self.cfg.minibatch_size)):
                mb_idx = idx[start:start + self.cfg.minibatch_size]
                mb_states = states[mb_idx]
                mb_actions = actions_norm[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_logp = old_logp[mb_idx]

                # Evaluate current policy on minibatch
                conc = None
                dist = None
                want_conc = self.cfg.theory_align and self.cfg.theory_align_conc_weight > 0.0
                want_dist = self.use_theory_align_v2 and (
                    self.cfg.theory_align_v2_var_coef > 0.0 or self.cfg.theory_align_v2_br_coef > 0.0
                )
                if want_conc and want_dist:
                    logp, entropy, values, conc, dist = self.evaluate_actions(
                        mb_states, mb_actions, return_conc=True, return_dist=True,
                    )
                elif want_conc:
                    logp, entropy, values, conc = self.evaluate_actions(
                        mb_states, mb_actions, return_conc=True,
                    )
                elif want_dist:
                    logp, entropy, values, dist = self.evaluate_actions(
                        mb_states, mb_actions, return_dist=True,
                    )
                else:
                    logp, entropy, values = self.evaluate_actions(mb_states, mb_actions)

                # Numerical safety guard
                if (
                    not torch.isfinite(logp).all()
                    or not torch.isfinite(entropy).all()
                    or not torch.isfinite(values).all()
                ):
                    breach_count += 1
                    if breach_count >= max(1, int(self.cfg.kl_stop_patience)):
                        early_stop_triggered = True
                        early_stop_reason = "nan_guard"
                        epoch_idx_triggered = epoch_idx
                        mb_idx_triggered = mb_local_idx
                        break
                    else:
                        continue

                # Compute PPO loss
                log_ratio = logp - mb_old_logp
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                approx_kl = (mb_old_logp - logp).mean()

                # PPO clipping diagnostics
                clip_frac = torch.mean((ratio.gt(1 + clip_eps) | ratio.lt(1 - clip_eps)).float())

                # KL proxy (non-negative) for early stop
                kl_proxy = torch.mean(ratio - 1.0 - log_ratio).item()
                ratio_max = ratio.max().item()

                # Secondary drift / stability signals
                log_ratio_abs_mean = log_ratio.abs().mean().item()
                log_ratio_mean = log_ratio.mean().item()
                log_ratio_std = log_ratio.std(unbiased=False).item()
                kl_ms = torch.mean(log_ratio * log_ratio).item()

                # Value loss
                if values.dim() != 1:
                    values = values.view(-1)
                if mb_returns.dim() != 1:
                    mb_returns = mb_returns.view(-1)
                value_loss = F.mse_loss(values, mb_returns)
                
                # Combined loss
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                
                # Optional regularization terms
                if conc is not None:
                    conc_reg = -float(self.cfg.theory_align_conc_weight) * torch.log(conc + 1e-8).mean()
                    loss = loss + conc_reg
                if self.use_theory_align_v2 and self.cfg.theory_align_v2_var_coef > 0.0:
                    if dist is None:
                        dist = self.dist(mb_states)[0]
                    alpha = dist.concentration1
                    beta = dist.concentration0
                    conc_val = alpha + beta
                    denom = (conc_val * conc_val) * (conc_val + 1.0) + 1e-8
                    var_action = (alpha * beta) / denom
                    var_effort = var_action * ((self.high - self.low) ** 2)
                    var_loss = float(self.cfg.theory_align_v2_var_coef) * var_effort.mean()
                    loss = loss + var_loss
                if self.use_theory_align_v2 and self.cfg.theory_align_v2_br_coef > 0.0:
                    if dist is None:
                        dist = self.dist(mb_states)[0]
                    a_mean = dist.mean.clamp(1e-6, 1.0 - 1e-6)
                    mean_effort = self.low + a_mean.squeeze(-1) * (self.high - self.low)
                    q = mb_states[:, 0] * 60.0
                    k = mb_states[:, 1] * 1e-3
                    w_gap = mb_states[:, 2] * 10.0
                    denom = 4.0 * q * k + 1e-8
                    e_star = (w_gap / denom).clamp(self.low, self.high)
                    br_loss = float(self.cfg.theory_align_v2_br_coef) * (mean_effort - e_star).pow(2).mean()
                    loss = loss + br_loss

                # Gradient step
                self.opt.zero_grad()
                loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm).item())
                self.opt.step()

                # Record per-minibatch stats
                policy_loss_list.append(float(policy_loss.detach().cpu().item()))
                value_loss_list.append(float(value_loss.detach().cpu().item()))
                entropy_list.append(float(entropy.detach().cpu().item()))
                approx_kl_list.append(float(approx_kl.detach().cpu().item()))
                kl_proxy_list.append(kl_proxy)
                kl_ms_list.append(kl_ms)
                clip_frac_list.append(float(clip_frac.detach().cpu().item()))
                ratio_mean_list.append(float(ratio.mean().detach().cpu().item()))
                ratio_std_list.append(float(ratio.std(unbiased=False).detach().cpu().item()))
                ratio_max_list.append(ratio_max)
                log_ratio_abs_mean_list.append(log_ratio_abs_mean)
                log_ratio_mean_list.append(log_ratio_mean)
                log_ratio_std_list.append(log_ratio_std)
                grad_norm_list.append(grad_norm)
                minibatches_completed += 1

                kl_values.append(float(approx_kl.detach().cpu().item()))
                entropy_values.append(float(entropy.detach().cpu().item()))

                # Early-stop breach logic
                breached = False
                if kl_proxy > kl_stop_threshold:
                    breached = True
                    early_stop_reason = "kl_proxy_exceeded"
                elif ratio_max > ratio_stop_threshold:
                    breached = True
                    early_stop_reason = "ratio_exceeded"

                if breached:
                    breach_count += 1
                else:
                    breach_count = 0

                if self.cfg.kl_early_stop and breach_count >= max(1, int(self.cfg.kl_stop_patience)):
                    early_stop_triggered = True
                    epoch_idx_triggered = epoch_idx
                    mb_idx_triggered = mb_local_idx
                    break

            if early_stop_triggered:
                break

        # Update counter
        self._updates += 1

        # Compile metrics
        metrics = {
            # Advantage stats
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "adv_norm_std": adv_norm_std,
            # State/reward/value stats
            "state_mean": state_mean,
            "state_std": state_std,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "value_mean": value_mean,
            "value_std": value_std,
            # Training diagnostics
            "approx_kl": float(np.mean(kl_values)) if kl_values else 0.0,
            "batch_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            # Per-minibatch aggregates
            "policy_loss_mean": float(np.mean(policy_loss_list)) if policy_loss_list else 0.0,
            "policy_loss_max": float(np.max(policy_loss_list)) if policy_loss_list else 0.0,
            "value_loss_mean": float(np.mean(value_loss_list)) if value_loss_list else 0.0,
            "value_loss_max": float(np.max(value_loss_list)) if value_loss_list else 0.0,
            "entropy_mean": float(np.mean(entropy_list)) if entropy_list else 0.0,
            "entropy_max": float(np.max(entropy_list)) if entropy_list else 0.0,
            "approx_kl_max_abs": float(np.max(np.abs(approx_kl_list))) if approx_kl_list else 0.0,
            "kl_proxy_mean": float(np.mean(kl_proxy_list)) if kl_proxy_list else 0.0,
            "kl_proxy_max": float(np.max(kl_proxy_list)) if kl_proxy_list else 0.0,
            "kl_ms_mean": float(np.mean(kl_ms_list)) if kl_ms_list else 0.0,
            "kl_ms_max": float(np.max(kl_ms_list)) if kl_ms_list else 0.0,
            "clip_frac_mean": float(np.mean(clip_frac_list)) if clip_frac_list else 0.0,
            "clip_frac_max": float(np.max(clip_frac_list)) if clip_frac_list else 0.0,
            "ratio_mean": float(np.mean(ratio_mean_list)) if ratio_mean_list else 0.0,
            "ratio_std_mean": float(np.mean(ratio_std_list)) if ratio_std_list else 0.0,
            "ratio_max": float(np.max(ratio_max_list)) if ratio_max_list else 0.0,
            "log_ratio_abs_mean": float(np.mean(log_ratio_abs_mean_list)) if log_ratio_abs_mean_list else 0.0,
            "log_ratio_mean": float(np.mean(log_ratio_mean_list)) if log_ratio_mean_list else 0.0,
            "log_ratio_std_mean": float(np.mean(log_ratio_std_list)) if log_ratio_std_list else 0.0,
            "grad_norm_mean": float(np.mean(grad_norm_list)) if grad_norm_list else 0.0,
            "grad_norm_max": float(np.max(grad_norm_list)) if grad_norm_list else 0.0,
            # Early-stop summary
            "early_stop_triggered": early_stop_triggered,
            "early_stop_reason": early_stop_reason,
            "epoch_idx_triggered": int(epoch_idx_triggered),
            "mb_idx_triggered": int(mb_idx_triggered),
            "epochs_completed": int(epoch_idx + 1 if not early_stop_triggered else epoch_idx_triggered + 1),
            "minibatches_completed": int(minibatches_completed),
            "minibatches_total_planned": int(minibatches_total_planned),
            "clip_eps_used": clip_eps,
            "ratio_stop_threshold_used": ratio_stop_threshold,
            "kl_stop_threshold_used": kl_stop_threshold,
        }
        
        # Explained variance
        returns_cpu = returns.detach()
        old_values_cpu = torch.stack(self.storage["values"]).to(self.device).view(-1)
        var_returns = returns_cpu.var(unbiased=False)
        if torch.isfinite(var_returns) and var_returns.item() > 1e-8:
            ev_old = 1.0 - torch.var(returns_cpu - old_values_cpu, unbiased=False) / (var_returns + 1e-8)
            metrics["explained_variance_oldV"] = float(ev_old.item())
        else:
            metrics["explained_variance_oldV"] = float("nan")

        # Clear storage for next rollout
        self.reset_storage()

        return metrics

    # ---- Utility methods ----
    def state_from_params(self, *, q: float, k: float, w_h: float, w_l: float) -> torch.Tensor:
        """Construct normalized state tensor from game parameters."""
        # Normalize features to roughly [0,1]
        q_norm = float(q) / 60.0      # assumes q up to ~60
        k_norm = float(k) / 1e-3      # k around 4e-4 => ~0.4
        wgap_norm = float(w_h - w_l) / 10.0  # prize gap scaled by 10
        s = torch.tensor([q_norm, k_norm, wgap_norm], dtype=torch.float32, device=self.device)
        return s.unsqueeze(0)
