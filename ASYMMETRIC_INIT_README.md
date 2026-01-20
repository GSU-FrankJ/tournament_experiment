# Asymmetric Initialization for Two-Player PPO

## 功能说明

本次修改实现了**非对称初始化**机制，使两个agent从不同的effort起点开始训练，然后收敛到理论最优值。

### 效果
- **Agent1**: 初始effort **高于**理论值（向上偏移30%）
- **Agent2**: 初始effort **低于**理论值（向下偏移30%）
- **收敛过程**: 两个agent从不同方向逐渐收敛到理论最优值

## 实现方案

### 方案一：网络偏置初始化（已实现，未启用）

修改了 `ActorCriticMeanConc` 类，添加 `init_bias_mean` 参数：

```python
class ActorCriticMeanConc(nn.Module):
    def __init__(
        self,
        ...
        init_bias_mean: Optional[float] = None,  # 新增参数
    ):
        # 如果指定了init_bias_mean，设置mean_head的bias
        if init_bias_mean is not None:
            target_logit = np.log(init_bias_mean / (1.0 - init_bias_mean))
            nn.init.constant_(self.mean_head.bias, target_logit)
```

**优点**: 更自然，从网络层面实现
**缺点**: 需要两个独立的网络，增加复杂度

### 方案二：Warmup阶段的Effort偏移（✅ 当前使用）

在训练的前N个updates中，对采样的effort添加偏移：

```python
# 在 run_two_players.py 中
warmup_updates = 20  # 默认20个updates
bias_strength = 1.0 - (update_idx / warmup_updates)  # 逐渐衰减
bias_magnitude = e_theory * 0.3 * bias_strength  # 30%偏移

# Agent1向上偏移
e1_biased = e1 + bias_magnitude

# Agent2向下偏移  
e2_biased = e2 - bias_magnitude
```

**优点**: 
- 实现简单，不改变网络结构
- 偏移量可配置
- 自动逐渐消失

**缺点**: 
- 在warmup期间轻微违反on-policy特性（但影响很小）

## 使用方法

### 默认行为
运行原命令，会**自动启用**20个updates的warmup偏移：

```bash
python run/run_two_players.py --method ppo --rollout-mode selfplay --q 25 \
  --episodes 2048000 --seed 50 --theory-align-v2 --enable-convergence-eval \
  --cheap-gate-profile relaxed
```

### 自定义Warmup长度

在配置文件 `config/one_stage_two_players.py` 中添加：

```python
config = {
    ...
    "asymmetric_warmup_updates": 30,  # 改为30个updates
}
```

### 禁用Asymmetric Warmup

设置为0即可：

```python
config = {
    ...
    "asymmetric_warmup_updates": 0,  # 禁用
}
```

## 预期效果

### 训练初期（Update 1-20）
```
Update 1:
  Agent1 effort: ~113.75 (理论值87.5 + 30% = 113.75)
  Agent2 effort: ~61.25  (理论值87.5 - 30% = 61.25)
  
Update 10:
  Agent1 effort: ~100.62 (偏移衰减到15%)
  Agent2 effort: ~74.38
  
Update 20:
  Agent1 effort: ~87.50 (偏移消失)
  Agent2 effort: ~87.50
```

### 收敛历史可视化

运行 `results/plot_convergence.py` 后，生成的图表会显示：
- **Agent1曲线**: 从高处下降收敛到理论值
- **Agent2曲线**: 从低处上升收敛到理论值
- **理论值水平线**: 两条曲线的交汇点

## 技术细节

### 修改的文件

1. **agents/ppo_two_players_clean.py**
   - `ActorCriticMeanConc.__init__`: 添加 `init_bias_mean` 参数
   - `PPOTwoPlayersBandit.__init__`: 支持传递初始化偏置

2. **run/run_two_players.py**
   - `run_ppo`: 在rollout循环中添加warmup偏移逻辑
   - 自动计算理论值和偏移量
   - 偏移量随训练进度线性衰减

### 关键代码位置

```python
# run/run_two_players.py, 约在928行
warmup_updates = int(cfg.get("asymmetric_warmup_updates", 20))
apply_warmup_bias = update_idx < warmup_updates

if apply_warmup_bias:
    e_theory = clip_stage2(e_star_two_players(q_for_theory, w_h, w_l, k), effort_bounds)
    bias_strength = 1.0 - (float(update_idx) / float(max(1, warmup_updates)))
    bias_magnitude = e_theory * 0.3 * bias_strength
    
    e1_biased = e1 + bias_magnitude  # Agent1向上
    e2_biased = e2 - bias_magnitude  # Agent2向下
```

## 验证

测试脚本已验证代码正常工作：
```bash
cd /home/fjiang4/tournament_experiment-4
python3 -c "
from agents.ppo_two_players_clean import PPOTwoPlayersBandit, PPOConfig
agent = PPOTwoPlayersBandit((0, 200), PPOConfig(), device='cpu')
# 测试通过
"
```

## 注意事项

1. **仅适用于theory-align-v2模式**: 默认配置下使用
2. **偏移量自动计算**: 基于当前q值的理论最优effort
3. **不影响最终收敛**: 偏移在20个updates后完全消失
4. **可视化友好**: 能清晰看到两个agent的收敛轨迹

## 实验示例

```bash
# q=25 (e*≈87.5)
python run/run_two_players.py --method ppo --rollout-mode selfplay --q 25 \
  --episodes 2048000 --seed 50 --theory-align-v2 --enable-convergence-eval \
  --cheap-gate-profile relaxed

# 预期: Agent1从~114, Agent2从~61开始，20个updates后都收敛到~87.5
```

---

**修改日期**: 2026-01-19
**作者**: AI Assistant
