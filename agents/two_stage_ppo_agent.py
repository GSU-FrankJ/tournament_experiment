"""
Two-stage PPO agent (true PPO) with Beta policy and GAE(λ).

Key features implemented as requested:
- Policy: Beta(α, β) over action a ∈ (0,1), transformed to effort e ∈ [e_min, e_max]
  via e = e_min + a * (e_max - e_min). The log_prob(e) = log_prob_Beta(a) - log(e_max - e_min)
  adds the Jacobian term for the change of variables.
- Win probability model in env is logistic (softmax/logit), configured separately.
- Advantage: GAE(λ). δ_t = r_t + γ V(s_{t+1}) - V(s_t), A_t = Σ (γλ)^l δ_{t+l}.
  We standardize advantages before updates.
- Value target: R_t = A_t + V(s_t).
- Self-play: Both players share the same policy by default. Optionally support
  fixed-opponent sampling through injected statistics.

Notes:
- The environment provides very compact states (stage indicator). We augment
  with simple features such as normalized q and a running estimate of opponent
  effort statistics to provide the network with additional signal.
"""

from __future__ import annotations

from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta


class BetaPolicyNet(nn.Module):
    """Policy-Value network producing Beta(α, β) parameters and a state-value.

    Input features (5-dim) to leverage informative env states:
    [stage_indicator_norm, q_norm, opp_signal_norm, won_stage1_norm, opp_e1_norm]
    - stage_indicator_norm: 0 for stage 1, 1 for stage 2
    - q_norm = q / 100
    - opp_signal_norm: generic opponent signal in [0,1] (EMA or revealed effort)
    - won_stage1_norm: 0/1 flag (0 in stage 1)
    - opp_e1_norm: opponent's stage1 effort normalized to [0,1] (0 if hidden/unknown)
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # Output positive concentration parameters via softplus + 1.0 for stability
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (alpha, beta, value). Alpha/Beta are positive scalars per sample."""
        h = self.feature(x)
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta = F.softplus(self.beta_head(h)) + 1.0
        value = self.value_head(h)
        return alpha.squeeze(-1), beta.squeeze(-1), value.squeeze(-1)


class TwoStagePPOAgent:
    """True PPO agent with Beta policy and GAE(λ) for two-stage tournament.

    This agent is self-play capable: by default both players share the same
    network/parameters. The caller should gather transitions for both players
    and feed them to this agent for policy updates.
    """

    def __init__(
        self,
        effort_bounds_stage1: Tuple[float, float] = (0.0, 100.0),
        effort_bounds_stage2: Tuple[float, float] = (0.0, 200.0),
        q_value: float = 25.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        lr: float = 3e-4,
        max_grad_norm: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        # Config
        self.b1 = effort_bounds_stage1
        self.b2 = effort_bounds_stage2
        self.q_value = q_value
        self.gamma = gamma
        self.lam = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks and optimizer
        self.net = BetaPolicyNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # Simple opponent statistics for state augmentation (EMA)
        self.opp_avg_effort_stage1: float = 0.0
        self.opp_avg_effort_stage2: float = 0.0
        self._opp_alpha = 0.95  # EMA factor

        # Rollout buffers
        self.buffer: Dict[str, List[torch.Tensor]] = {
            "states": [],
            "actions_a": [],  # action in (0,1)
            "efforts_e": [],  # transformed effort
            "log_probs": [],  # log prob in e-space (with Jacobian)
            "values": [],
            "rewards": [],
            "dones": [],
        }

        # Track recent efforts for reporting
        self.stage1_history: List[float] = []
        self.stage2_history: List[float] = []

    # --------- Utilities ---------
    @staticmethod
    def _scale_to_effort(a: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
        """Map a ∈ (0,1) to effort e ∈ [lo, hi]."""
        lo, hi = bounds
        return lo + a * (hi - lo)

    @staticmethod
    def _jacobian_log_term(bounds: Tuple[float, float]) -> float:
        """Return constant log|de/da| = log(e_max - e_min)."""
        lo, hi = bounds
        return float(np.log(max(hi - lo, 1e-12)))

    def _build_state(self, stage_indicator: float, opp_signal: float, bounds: Tuple[float, float]) -> torch.Tensor:
        """Construct 5-d input features for the network for generic usage.

        In Stage 1, we provide won=0 and opp_e1=0 (unknown yet). In Stage 2, caller can
        pass opponent signal (e.g., EMA or revealed effort) via opp_signal if desired.
        """
        stage_norm = 0.0 if stage_indicator <= 1.5 else 1.0
        q_norm = float(self.q_value) / 100.0
        lo, hi = bounds
        opp_signal_norm = 0.0 if hi <= lo else float(np.clip((opp_signal - lo) / (hi - lo), 0.0, 1.0))
        won_stage1_norm = 0.0
        opp_e1_norm = 0.0
        x = torch.tensor([stage_norm, q_norm, opp_signal_norm, won_stage1_norm, opp_e1_norm], dtype=torch.float32, device=self.device)
        return x

    def _build_state_from_env_obs(self, obs: torch.Tensor, bounds_stage1: Tuple[float, float], bounds_stage2: Tuple[float, float]) -> torch.Tensor:
        """Map env's 5-d observation to the 5-d features expected by the network.

        obs = [stage_indicator(1/2/0), won_stage1(0/1), my_e1, opp_e1, p_win_estimate]
        We reuse:
        - stage -> stage_norm
        - q_norm from agent config
        - opp_signal_norm from opp_e1 (if visible) or 0
        - won_stage1_norm from obs
        - opp_e1_norm from obs normalized by Stage-1 bounds
        """
        stage_indicator = float(obs[0].item())
        won_stage1 = float(obs[1].item())
        opp_e1 = float(obs[3].item())
        stage_norm = 0.0 if stage_indicator <= 1.5 else 1.0
        q_norm = float(self.q_value) / 100.0
        # Opponent signal: use revealed opponent e1 if available; else 0
        lo1, hi1 = bounds_stage1
        opp_signal_norm = 0.0 if hi1 <= lo1 else float(np.clip((opp_e1 - lo1) / (hi1 - lo1), 0.0, 1.0))
        won_stage1_norm = 1.0 if won_stage1 >= 0.5 else 0.0
        opp_e1_norm = opp_signal_norm
        x = torch.tensor([stage_norm, q_norm, opp_signal_norm, won_stage1_norm, opp_e1_norm], dtype=torch.float32, device=self.device)
        return x

    # --------- Acting ---------
    def act(self, stage: int, opp_signal: float, bounds: Tuple[float, float], deterministic: bool = False):
        """Sample an action (effort) and return useful terms for PPO update.

        Returns:
            effort_e (float), log_prob_e (float), value (float), action_a (float in (0,1)), state_tensor (torch.Tensor)
        """
        self.net.eval()
        x = self._build_state(float(stage), opp_signal, bounds)
        alpha, beta, value = self.net(x.unsqueeze(0))  # batch 1
        dist = Beta(alpha, beta)
        if deterministic:
            # Use mean of Beta as deterministic action
            a = (alpha / (alpha + beta)).clamp(1e-6, 1 - 1e-6)
        else:
            a = dist.rsample().clamp(1e-6, 1 - 1e-6)
        e = self._scale_to_effort(a, bounds)
        # log_prob in e-space = log_prob_Beta(a) - log(e_max - e_min)
        log_prob_e = dist.log_prob(a).squeeze(-1) - self._jacobian_log_term(bounds)
        return (
            float(e.item()),
            float(log_prob_e.item()),
            float(value.item()),
            float(a.item()),
            x.detach(),
        )

    def act_with_env_obs(self, obs: torch.Tensor, bounds_stage1: Tuple[float, float], bounds_stage2: Tuple[float, float], deterministic: bool = False):
        """Act given the environment-provided observation vector.

        This leverages revealed Stage-1 information for Stage-2 decisions.
        """
        self.net.eval()
        x = self._build_state_from_env_obs(obs, bounds_stage1, bounds_stage2)
        alpha, beta, value = self.net(x.unsqueeze(0))
        dist = Beta(alpha, beta)
        if deterministic:
            a = (alpha / (alpha + beta)).clamp(1e-6, 1 - 1e-6)
        else:
            a = dist.rsample().clamp(1e-6, 1 - 1e-6)
        # Choose bounds by stage from obs[0]
        stage_indicator = float(obs[0].item())
        bounds = bounds_stage1 if stage_indicator <= 1.5 else bounds_stage2
        e = self._scale_to_effort(a, bounds)
        log_prob_e = dist.log_prob(a).squeeze(-1) - self._jacobian_log_term(bounds)
        return (
            float(e.item()),
            float(log_prob_e.item()),
            float(value.item()),
            float(a.item()),
            x.detach(),
        )

    # --------- Buffer management ---------
    def store(self, state: torch.Tensor, action_a: float, effort_e: float, log_prob_e: float, value: float, reward: float, done: bool):
        """Store a single transition in the rollout buffer."""
        self.buffer["states"].append(state.detach())
        self.buffer["actions_a"].append(torch.tensor(action_a, dtype=torch.float32, device=self.device))
        self.buffer["efforts_e"].append(torch.tensor(effort_e, dtype=torch.float32, device=self.device))
        self.buffer["log_probs"].append(torch.tensor(log_prob_e, dtype=torch.float32, device=self.device))
        self.buffer["values"].append(torch.tensor(value, dtype=torch.float32, device=self.device))
        self.buffer["rewards"].append(torch.tensor(reward, dtype=torch.float32, device=self.device))
        self.buffer["dones"].append(torch.tensor(float(done), dtype=torch.float32, device=self.device))

    def _clear_buffer(self):
        for k in self.buffer:
            self.buffer[k] = []

    # --------- Learning ---------
    def _compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages using GAE(λ)."""
        rewards = torch.stack(self.buffer["rewards"])  # [T]
        values = torch.stack(self.buffer["values"])    # [T]
        dones = torch.stack(self.buffer["dones"])      # [T]

        T = rewards.shape[0]
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
                next_nonterminal = 0.0
            else:
                next_value = float(values[t + 1].item())
                next_nonterminal = 1.0 - float(dones[t].item())
            delta = float(rewards[t].item()) + self.gamma * next_value * next_nonterminal - float(values[t].item())
            gae = delta + self.gamma * self.lam * next_nonterminal * gae
            advantages[t] = gae
        returns = advantages + values
        # Normalize advantages (zero mean, unit variance)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns.detach(), advantages.detach()

    def update(self, epochs: int = 4, minibatch_size: int = 64) -> Dict[str, float]:
        """PPO clipped surrogate update with entropy and value loss."""
        if len(self.buffer["states"]) == 0:
            return {"loss": 0.0}

        states = torch.stack(self.buffer["states"])  # [T, feat]
        actions_a = torch.stack(self.buffer["actions_a"]).clamp(1e-6, 1 - 1e-6)  # [T]
        old_log_probs_e = torch.stack(self.buffer["log_probs"])  # [T]
        returns, advantages = self._compute_gae()

        T = states.shape[0]
        indices = np.arange(T)
        total_loss = 0.0

        # Precompute log-range constants for Jacobian per stage using state[0]
        stage_norm = states[:, 0]  # 0 for stage1, 1 for stage2
        log_range_s1 = torch.log(torch.tensor(self.b1[1] - self.b1[0], dtype=torch.float32, device=self.device))
        log_range_s2 = torch.log(torch.tensor(self.b2[1] - self.b2[0], dtype=torch.float32, device=self.device))
        jacobian_log = torch.where(stage_norm < 0.5, log_range_s1, log_range_s2)

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, T, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                mb_states = states[mb_idx]
                mb_actions = actions_a[mb_idx]
                mb_old_logp_e = old_log_probs_e[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advs = advantages[mb_idx]
                mb_jac_log = jacobian_log[mb_idx]

                alpha, beta, values = self.net(mb_states)
                dist = Beta(alpha, beta)
                new_logp_a = dist.log_prob(mb_actions.clamp(1e-6, 1 - 1e-6))
                new_logp_e = new_logp_a - mb_jac_log  # match stored measure

                ratio = torch.exp(new_logp_e - mb_old_logp_e)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)

                entropy = dist.entropy().mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())

        self._clear_buffer()
        # Normalize by (approximate) number of minibatches
        num_minibatches = max(1, int(np.ceil(T / max(1, minibatch_size))) * epochs)
        return {"loss": total_loss / num_minibatches}

    # --------- EMA opponent stats ---------
    def update_opponent_avg(self, stage: int, opponent_effort: float):
        """Update running average of opponent efforts for state augmentation."""
        if stage == 1:
            self.opp_avg_effort_stage1 = self._opp_alpha * self.opp_avg_effort_stage1 + (1 - self._opp_alpha) * float(opponent_effort)
        else:
            self.opp_avg_effort_stage2 = self._opp_alpha * self.opp_avg_effort_stage2 + (1 - self._opp_alpha) * float(opponent_effort)

    def opp_avg(self, stage: int) -> float:
        return self.opp_avg_effort_stage1 if stage == 1 else self.opp_avg_effort_stage2




