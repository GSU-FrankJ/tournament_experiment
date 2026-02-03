# 之前的 “ppo” 分支是监督回归到理论目标值，并非基于环境交互的策略梯度或 PPO。

无环境交互: 未引入任何环境，也没有从环境获得奖励、状态或回合数据；未使用已存在的一阶段/两人环境。

无策略采样与 log_prob: 网络输出经 Sigmoid 后直接当作努力值（线性缩放），没有定义动作分布、没有对动作取样、没有计算 log_prob。

无回报/优势估计: 没有折扣回报、优势（如 GAE）、baseline（价值网络）等。

无 PPO 剪切目标: 没有 old policy 固定、ratio = exp(logπθ(a|s) - logπθ_old(a|s))、clip、多 epoch/minibatch 优化等关键步骤。

# one stage two players

**PPO 训练超参数**

- gamma: 0.99 — 折扣因子. agents/ppo_two_players_clean.py:53
- gae_lambda: 0.95 — GAE λ. agents/ppo_two_players_clean.py:54
- clip_eps: 0.2 — PPO 裁剪阈值. agents/ppo_two_players_clean.py:55
- lr: 3e-4 — Adam 学习率. agents/ppo_two_players_clean.py:56
- value_coef: 0.5 — 价值损失权重. agents/ppo_two_players_clean.py:57
- entropy_coef: 0.01 — 熵正则权重. agents/ppo_two_players_clean.py:58
- max_grad_norm: 0.5 — 梯度裁剪. agents/ppo_two_players_clean.py:59
- steps_per_update: 2048 — 每次更新的采样步数. agents/ppo_two_players_clean.py:60
- epochs: 15 — 每次更新的训练轮数. agents/ppo_two_players_clean.py:61
- minibatch_size: 256 — 小批量大小. agents/ppo_two_players_clean.py:62
- state_dim: 3 — 状态为[q_norm,k_norm,w_gap_norm]. agents/ppo_two_players_clean.py:63
- hidden: 64 — MLP 隐层宽度. agents/ppo_two_players_clean.py:64

## First run:PPO 的结果偏差较大，尤其 q=25 下明显“低配”用力。

1.把不同 q 的样本混在同一批里做优势标准化，尺度不同（e* 和收益尺度随 q 变）会让梯度被“量纲大/方差小”的 q 主导，常见表现是对大 e* 的 q 学得慢或系统性偏低。

2.自博弈耦合：两个玩家用同一策略同时更新，目标是“同时最优”，但对称博弈里这会让对手也随你一起动，导致学习目标不断移动，容易偏保守（欠努力

3.训练预算虽到 100k，但每次更新 steps_per_update=2048、minibatch=128，实际“对单个 q 的有效样本”更少

## resolution:

1.单 q 训练,训练数据只来自该 q，不再在不同 q 之间混合。

2.滞后对手（policy lag）

- 新增冻结对手网络，按间隔或 EMA 同步，采样时：
  - 玩家1使用当前策略（on-policy，参与 PPO 更新）
  - 玩家2使用冻结对手策略（仅用于对局，不参与更新）
- opponent_sync_interval：每多少次 PPO 更新同步一次（默认 5）
- opponent_ema_tau：>0 时用 EMA 同步（0 为硬拷贝）

new problem:

饱和区梯度变平：当训练中频繁落入 |e_i-e_j| > 2q 的饱和区，p 的梯度为 0，学习会变困难

1.先只训练 q=55
2.entropy_coef 从 0.02 线性衰减到 0.005，前 50 次更新完成退火。
3.在 run/run_two_players.py:58-68 把 opponent_sync_interval=5，opponent_ema_tau=0.1。
在采样时用旧网给对手动作：把 agent.act(s2) 换成 agent.act_opponent(s2) 于 run/run_two_players.py:85-86。
4.学习率: 将 lr 降到 3e-4 或 2e-4（根据实际结果决定）
5.批量与轮次: 提高 steps_per_update 到 8192，减小 epochs 到 10，minibatch_size 到 512，提升稳定性、降低过拟合到极端动作的风险。

### **Run Command: ** 

python3 run/run_two_players.py --method ppo --q 55 --episodes 409600

### result: 

6.5,0.0004,0.0004,none,0.0,39.772727272727266,ppo,0.0,33.09648931026459,215.12718051671982,Poor,409600,43.395546756007434

Likely causes

1. we switched the opponent to the lagged network but still stored its transition for PPO with old_logp from the opponent. That breaks on-policy PPO’s ratio (logπθ(a)−logπθold(a)) assumption and can slow or misdirect learning.

2. 50 updates often isn’t enough for self-play stability at new hyperparameters

3. even with entropy decay, initial Beta mean ≈ 0.5 causes early dynamics; 

### solution

Removed opponent transition from PPO storage to restore on-policy updates.

对手更新：设 opponent_ema_tau=0（硬拷贝），opponent_sync_interval=10–20，减少非平稳性。

起始 0.02 左右，20–50 次更新退到 0
start_entropy, end_entropy, decay_updates = 0.02, 0.005, 50
线性退火到 0：
改成 start_entropy, end_entropy, decay_updates = 0.02, 0.0, 50

超参稳健化：lr=1e-4，epochs=15–20，minibatch_size=1024，保持 steps_per_update=8192。

评估方式：用 Beta 众数 (α−1)/(α+β−2)（α、β>1 时），再写入 CSV。
训练集与日志：让 run_ppo 尊重 CLI 的 train_qs 参数，增加每次更新后的评估与 gap 日志，方便早停和回看。

### command:

python3 run/run_two_players.py --method ppo --q 55 --episodes 819200

### result: 

3.0,6.5,0.0004,0.0004,none,0.0,39.772727272727266,ppo,0.0,33.67767930030823,218.90491545200348,Poor,819200,39.61781182072377

likely causes

熵值降至0后探索效率低下；难以进一步提升。
保守步骤：lr=1e-4，clip_eps=0.2，每15次更新进行一次对手硬拷贝。
采用半批量更新（仅存储学习者状态转换）。

solution

Entropy floor 0.002
steps_per_update 16384, epochs 15
clip_eps 0.25
opponent_sync_interval 10

### result:

3.0,6.5,0.0004,0.0004,none,0.0,39.772727272727266,ppo,0.0,34.97694730758667,227.35015749931335,Poor,1638400,31.172569773413898

lr: 2e-4 (gives larger steps near plateau).
opponent_sync_interval: 5 (closer tracking → gradients match current game).
keep steps_per_update: 16384; consider +50–100 more updates (episodes 3,276,800).

### result:

3.0,6.5,0.0004,0.0004,none,0.0,39.772727272727266,ppo,0.0,37.08954155445099,241.08202010393143,Poor,3276800,17.440707168795825

Set lr=3e-4, opponent_sync_interval=2, epochs=20 

### result:

3.0,6.5,0.0004,0.0004,none,0.0,39.772727272727266,ppo,0.0,37.91002333164215,246.41515165567398,Poor,4915200,12.107575617053271

Set opponent_sync_interval=1, add a simple late‑phase schedule for clip_eps (0.30 → 0.25), and switch to fully on‑policy self‑play for the final 50–100 updates.

### result:

3.0,6.5,0.0004,0.0004,none,0.0,39.772727272727266,ppo,0.0,38.105183839797974,247.68369495868683,Poor,4915200,10.839032314040423

clip 收尾保持稍大：把晚期 clip 区间调大一点（0.35→0.25），已接近，可再多 0.05 的上限能帮助爬升。
熵更低：最后 20–30 次更新把熵地板从 0.002 降到 0.0，聚集更紧。
可选小步长提升（仅晚期 50 次更新）：学习率从 3e-4 提到 4e-4（在训练循环里设 for g in agent.opt.param_groups: g['lr']=4e-4），结束前恢复 3e-4。

# 用当前 PPO 想“精确等于 0”在数值上基本不现实。

为什么很难等于 0

- 随机性与函数逼近误差: Beta 策略+采样/GAE 噪声+浮点精度，几乎不会让最终努力恰好等于 e*。
- 训练约束: PPO 裁剪、对手联动、学习率/熵退火都会让收敛停在 e* 附近而非精确点。
- 评估口径: CSV 的 gap 是加权差距（6.5×|e−e*|）。哪怕努力只差 0.01，也会显示 0.065，不会是 0。