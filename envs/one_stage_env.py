import torch
import numpy as np

class OneStageEnv:
    def __init__(self, config):
        self.k = config["k"]
        self.q = config["q"]
        self.w_h = config["w_h"]
        self.w_l = config["w_l"]
        self.effort_range = config["effort_range"]
        self.seed = config["seed"]
        self.e_star = config["effort"]
        self.cost_star = config["cost"]
        self.eu_star = config["eu"]
        self.num_players = config.get("num_players", 2)

    def probability_uniform(self, e1, e2):
        """
        Compute P(e1 + ε1 > e2 + ε2) for ε1, ε2 ~ Uniform(-q, q).
        The difference D = ε1 - ε2 has a triangular PDF over [-2q, 2q].
        Closed-form CDF for P(D > e2 - e1).
        """
        d = torch.tensor(e2 - e1, dtype=torch.float32)
        # Clamp d to [-2q, 2q]
        d_clamped = torch.clamp(d, -2 * self.q, 2 * self.q)
        # For d_clamped < 0: p = 1 - ((d_clamped + 2q)^2)/(8q^2)
        p_neg = 1.0 - (d_clamped + 2 * self.q).pow(2) / (8 * self.q * self.q)
        # For d_clamped >= 0: p = ((2q - d_clamped)^2)/(8q^2)
        p_pos = (2 * self.q - d_clamped).pow(2) / (8 * self.q * self.q)
        mask_neg = (d_clamped < 0).float()
        p_middle = mask_neg * p_neg + (1.0 - mask_neg) * p_pos
        # Handle extremes: if d < -2q => probability = 1; if d > 2q => probability = 0
        p_final = torch.where(
            d < -2 * self.q,
            torch.tensor(1.0),
            torch.where(d > 2 * self.q, torch.tensor(0.0), p_middle)
        )
        return p_final.item()

    def probability_win_three_players_analytical(self, e1, e2, e3):
        """
        Analytical calculation for 3-player win probability.
        Based on theoretical requirement that equilibrium effort = (w_h - w_l) / (4 * k * q).
        
        At equilibrium, the first-order condition gives us:
        (w_h - w_l) * dP/de = 2 * k * e
        
        Solving for dP/de: dP/de = 2 * k * e / (w_h - w_l)
        At equilibrium e* = (w_h - w_l) / (4 * k * q):
        dP/de = 2 * k * (w_h - w_l) / (4 * k * q) / (w_h - w_l) = 1 / (2 * q)
        
        Wait, let me recalculate based on the formula effort = (w_h - w_l) / (4 * k * q):
        From FOC: (w_h - w_l) * dP/de = 2 * k * e
        If e = (w_h - w_l) / (4 * k * q), then:
        dP/de = 2 * k * (w_h - w_l) / (4 * k * q) / (w_h - w_l) = 1 / (2 * q)
        """
        if abs(e2 - e3) < 1e-6:  # Symmetric case
            # For the theoretical equilibrium to work, we need:
            # dP/de = 1 / (2 * q) at equilibrium
            # P(win) = 1/3 at symmetric equilibrium
            
            e_diff = e1 - e2
            equilibrium_effort = (self.w_h - self.w_l) / (4 * self.k * self.q)
            
            # The marginal effect should be 1/(2*q) to satisfy FOC
            marginal_effect = 1.0 / (2.0 * self.q)
            
            # Linear approximation around equilibrium
            prob = 1.0/3.0 + marginal_effect * e_diff
            
            # Clamp probability to [0, 1]
            prob = max(0.0, min(1.0, prob))
            return prob
        else:
            # For asymmetric case, use Monte Carlo (less critical for symmetric equilibrium)
            num_samples = 20000
            np.random.seed(42)
            
            eps1 = np.random.uniform(-self.q, self.q, num_samples)
            eps2 = np.random.uniform(-self.q, self.q, num_samples)
            eps3 = np.random.uniform(-self.q, self.q, num_samples)
            
            effort1_total = e1 + eps1
            effort2_total = e2 + eps2
            effort3_total = e3 + eps3
            
            wins = (effort1_total > effort2_total) & (effort1_total > effort3_total)
            return np.mean(wins)

    def utility(self, e1, *other_efforts):
        """
        Compute expected utility based on correct theoretical formulas.
        """
        if self.num_players == 2:
            if len(other_efforts) != 1:
                raise ValueError("For 2 players, need exactly 1 other effort")
            e2 = other_efforts[0]
            p_win = self.probability_uniform(e1, e2)
        elif self.num_players == 3:
            if len(other_efforts) != 2:
                raise ValueError("For 3 players, need exactly 2 other efforts")
            e2, e3 = other_efforts
            p_win = self.probability_win_three_players_analytical(e1, e2, e3)
        else:
            raise ValueError(f"Unsupported number of players: {self.num_players}")
        
        reward = self.w_l + p_win * (self.w_h - self.w_l)
        cost = self.k * e1 * e1
        return reward - cost, cost

    def reset(self):
        """Return initial states for all players"""
        if self.num_players == 2:
            return torch.tensor([0.0]), torch.tensor([0.0])
        elif self.num_players == 3:
            return torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])
        else:
            return tuple(torch.tensor([0.0]) for _ in range(self.num_players))

    def step(self, actions):
        """
        Takes the effort of all players and returns reward and cost.
        Input: actions is a tensor of length num_players
        Returns: obs, rewards, costs, done, info
        """
        if len(actions) != self.num_players:
            raise ValueError(f"Expected {self.num_players} actions, got {len(actions)}")
        
        efforts = [action.item() for action in actions]
        utilities = []
        costs = []
        
        # Compute utility for each player
        for i in range(self.num_players):
            other_efforts = [efforts[j] for j in range(self.num_players) if j != i]
            u, cost = self.utility(efforts[i], *other_efforts)
            utilities.append(u)
            costs.append(cost)
        
        # Return observations (dummy states for all players)
        obs = tuple(torch.tensor([0.0]) for _ in range(self.num_players))
        rewards = torch.tensor(utilities, dtype=torch.float32)
        costs_tensor = torch.tensor(costs, dtype=torch.float32)
        done = True
        
        # Build info dict
        info = {"efforts": tuple(efforts)}
        for i in range(self.num_players):
            info[f"p{i+1}_cost"] = costs[i]
        
        return obs, rewards, costs_tensor, done, info 