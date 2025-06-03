import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =============================
# 1. 环境定义: OneStageTournamentEnv
# =============================
class OneStageTournamentEnv(gym.Env):
    def __init__(self, w_H=6.0, w_L=2.0, q=25.0, k=1/3500, opponent_policy_fn=None):
        super(OneStageTournamentEnv, self).__init__()
        self.w_H = w_H              # 赢得奖励
        self.w_L = w_L              # 底奖励
        self.q = q                  # 噪声范围参数，噪声服从 U(-q, q)
        self.k = k                  # 努力成本系数
        self.opponent_policy_fn = opponent_policy_fn  # 对手策略函数
        # 状态为dummy（单一常数状态），动作为努力值（连续，取值范围[0,100]）
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
    
    def reset(self):
        return np.array([0.0], dtype=np.float32)
    
    def step(self, action):
        # clip代理动作到合法范围
        agent_effort = np.clip(action, self.action_space.low, self.action_space.high)[0]
        # 为代理和对手生成噪声
        noise_agent = np.random.uniform(-self.q, self.q)
        noise_opponent = np.random.uniform(-self.q, self.q)
        # 获取对手的努力值
        if self.opponent_policy_fn is None:
            opp_effort = np.random.uniform(self.action_space.low, self.action_space.high)[0]
        else:
            opp_effort = np.clip(self.opponent_policy_fn(np.array([0.0], dtype=np.float32)),
                                  self.action_space.low, self.action_space.high)[0]
        # 计算输出
        y_agent = agent_effort + noise_agent
        y_opp = opp_effort + noise_opponent
        win = 1.0 if y_agent > y_opp else 0.0
        # 奖励 = 底奖励 + 赢得奖金（若win） - 努力成本
        reward = self.w_L + win*(self.w_H - self.w_L) - self.k*(agent_effort**2)
        done = True  # one-shot episode
        info = {"agent_effort": agent_effort, "opp_effort": opp_effort, "win": win}
        return np.array([0.0], dtype=np.float32), reward, done, info

# =============================
# 2. 策略-价值网络：ActorCritic
# =============================
class ActorCritic(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, action_dim=1):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        # 单独学习的 log_std 参数
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.action_mean(x)
        std = torch.exp(self.log_std)
        value = self.value(x)
        return mean, std, value

    def get_action(self, x):
        mean, std, value = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, x, actions):
        mean, std, value = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value

# =============================
# 3. 超参数设置与设备配置
# =============================
horizon = 1024            # 每次更新收集的样本数（每个episode1步）
num_epochs = 10           # PPO 每次更新 epoch 数
minibatch_size = 64
ppo_clip = 0.2
value_loss_coef = 0.5
entropy_coef = 0.01
learning_rate = 3e-4
gamma = 0.99              # 折扣因子（1步问题中影响不大）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 4. 初始化策略网络及优化器
# =============================
ac = ActorCritic().to(device)
optimizer = optim.Adam(ac.parameters(), lr=learning_rate)

# 为自我博弈，创建一个对手网络，每次更新后拷贝当前策略参数
opp_ac = ActorCritic().to(device)
opp_ac.load_state_dict(ac.state_dict())
opp_ac.eval()

def opponent_policy_fn(obs):
    # 输入 obs 为 numpy 数组，返回确定性动作（取均值）
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    with torch.no_grad():
        mean, _, _ = opp_ac.forward(obs_tensor)
    return mean.cpu().numpy()

# 创建环境，传入对手策略函数实现自我博弈
env = OneStageTournamentEnv(w_H=6.0, w_L=2.0, q=25.0, k=1/3500, opponent_policy_fn=opponent_policy_fn)

# =============================
# 5. PPO 训练循环及平均奖励记录
# =============================
num_updates = 1000   # 总更新次数
reward_history = []  # 保存每次更新的平均奖励

for update in range(num_updates):
    obs_list = []
    actions_list = []
    log_probs_list = []
    rewards_list = []
    values_list = []
    ep_rewards = []

    # 收集 horizon 个 episode（每个 episode 只有1步）
    for step in range(horizon):
        obs = env.reset()  # dummy 状态
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 1]
        action, log_prob, value = ac.get_action(obs_tensor)
        action_np = action.detach().cpu().numpy()[0]
        next_obs, reward, done, info = env.step(action_np)
        
        obs_list.append(obs)
        actions_list.append(action_np)
        log_probs_list.append(log_prob.detach().cpu().numpy())
        values_list.append(value.item())
        rewards_list.append(reward)
        ep_rewards.append(reward)
    
    # 转换为张量
    obs_batch = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)  # shape [horizon, 1]
    actions_batch = torch.tensor(np.array(actions_list), dtype=torch.float32).to(device)  # [horizon, 1]
    old_log_probs_batch = torch.tensor(np.array(log_probs_list), dtype=torch.float32).to(device)  # [horizon, 1]
    returns_batch = torch.tensor(np.array(rewards_list), dtype=torch.float32).to(device)  # [horizon]
    values_batch = torch.tensor(np.array(values_list), dtype=torch.float32).to(device)  # [horizon]

    # 对于单步问题，return = reward，优势 = reward - value
    advantages = returns_batch - values_batch
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO 多次 epoch 更新：对收集的数据训练 num_epochs 次
    dataset_size = obs_batch.size(0)
    for epoch in range(num_epochs):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, minibatch_size):
            indices = permutation[i:i+minibatch_size]
            obs_mb = obs_batch[indices]
            actions_mb = actions_batch[indices]
            old_log_probs_mb = old_log_probs_batch[indices]
            returns_mb = returns_batch[indices]
            advantages_mb = advantages[indices]

            # 评估当前策略下的 log_prob、entropy 和 value
            log_probs, entropy, values = ac.evaluate_actions(obs_mb, actions_mb)
            log_probs = log_probs.sum(dim=-1)  # 对于1维动作，这里求和结果即原值

            # 计算概率比率
            ratios = torch.exp(log_probs - old_log_probs_mb.sum(dim=-1))
            surr1 = ratios * advantages_mb
            surr2 = torch.clamp(ratios, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(-1), returns_mb)
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # 更新对手策略，实现自我博弈（每次更新后拷贝当前策略）
    opp_ac.load_state_dict(ac.state_dict())

    avg_reward = np.mean(ep_rewards)
    reward_history.append(avg_reward)
    if update % 10 == 0:
        print(f"Update {update}: Average Reward: {avg_reward:.2f}")

# =============================
# 6. 绘制平均奖励曲线
# =============================
updates = np.arange(0, num_updates, 1)
plt.figure(figsize=(10, 5))
plt.plot(updates, reward_history, label='Average Reward')
plt.xlabel('Update')
plt.ylabel('Average Reward')
plt.title('Average Reward Curve')
plt.legend()
plt.grid(True)
plt.show()

print("Training completed.")