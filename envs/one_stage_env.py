import torch

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

    def utility(self, e1, e2):
        """
        Compute expected utility using uniform noise:
          P(win) = probability_uniform(e1, e2)
          reward = w_l + p_win * (w_h - w_l)
          cost = k * e1^2
        """
        p_win = self.probability_uniform(e1, e2)
        reward = self.w_l + p_win * (self.w_h - self.w_l)
        cost = self.k * e1 * e1
        return reward - cost, cost

    def reset(self):
        return torch.tensor([0.0]), torch.tensor([0.0])

    def step(self, actions):
        """
        Takes the effort of two agents and returns reward and cost. 
        Input: actions is a tensor of length 2 [e1, e2] 
        Returns: obs, rewards, costs, done, info
        """
        e1, e2 = actions[0].item(), actions[1].item()
        u1, cost1 = self.utility(e1, e2)
        u2, cost2 = self.utility(e2, e1)
        obs = torch.tensor([0.0]), torch.tensor([0.0])  # no state
        rewards = torch.tensor([u1, u2], dtype=torch.float32)
        costs = torch.tensor([cost1, cost2], dtype=torch.float32)
        done = True
        info = {"efforts": (e1, e2), "p1_cost": cost1, "p2_cost": cost2}
        return obs, rewards, costs, done, info