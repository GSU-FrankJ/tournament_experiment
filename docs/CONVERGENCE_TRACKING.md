# 📊 Convergence Tracking Guide

本文档说明如何记录和可视化PPO和Gradient算法的收敛过程。

## ✨ 新增功能

### 1. **Individual Agent Effort Tracking (PPO)**

现在PPO训练会分别记录Player 1和Player 2的effort值，并在日志中显示：

```
[Update 10] q=25.0: e*=87.50, policy=86.38, gap=1.12, ...
  [Rollout] sample_avg_effort=86.32, mean_vs_sample_gap=0.06, effort_samples=8192, p1_effort=86.25, p2_effort=86.39
```

**新增字段**：
- `p1_effort`: Player 1的平均sampled effort
- `p2_effort`: Player 2的平均sampled effort

### 2. **Convergence History JSON Files**

训练完成后，会自动保存convergence history到JSON文件：

**PPO**: `results/convergence_history/ppo_q{q}_convergence.json`
**Gradient**: `results/convergence_history/gradient_q{q}_convergence.json`

#### JSON文件格式

```json
{
  "algorithm": "PPO",
  "q": 25.0,
  "theoretical_effort": 87.5,
  "steps": [0, 4096, 8192, ..., 2048000],
  "agent1_effort": [91.2, 89.4, 88.1, ..., 87.6],
  "agent2_effort": [92.1, 89.8, 88.3, ..., 87.5],
  "policy_mean_effort": [91.65, 89.6, 88.2, ..., 87.55],
  "rollout_mode": "selfplay",
  "total_episodes": 2048000
}
```

**Gradient算法**还包含额外的参数信息：

```json
{
  "algorithm": "gradient",
  "q": 25.0,
  "theoretical_effort": 87.5,
  "steps": [0, 1, 2, ..., 2000],
  "agent1_effort": [50.0, 65.2, ..., 87.45],
  "agent2_effort": [55.0, 67.1, ..., 87.55],
  "parameters": {
    "lr": 0.1,
    "grad_eps": 0.1,
    "tol": 0.0001,
    "num_samples": 64,
    "init_perturb": 1.0
  }
}
```

## 🚀 使用方法

### Step 1: 运行实验收集数据

#### PPO实验

```bash
# q=25
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 42

# q=40
python run/run_two_players.py --method ppo --q 40 --episodes 2048000 --seed 42

# q=55
python run/run_two_players.py --method ppo --q 55 --episodes 2048000 --seed 42
```

#### Gradient实验

```bash
# q=25
python run/run_two_players.py --method gradient --q 25

# q=40
python run/run_two_players.py --method gradient --q 40

# q=55
python run/run_two_players.py --method gradient --q 55
```

### Step 2: 生成收敛图

运行绘图脚本：

```bash
python plot_convergence.py
```

这将生成两个图：

1. **`results/convergence_comparison.png`**
   - 横轴：Training steps
   - 纵轴：Effort (两个agents)
   - 线条：理论值 / Gradient Agent1+2 / PPO Agent1+2
   - 面板：不同的噪声水平 (q值)

2. **`results/convergence_separate_agents.png`**
   - 上排：Agent 1在不同q值下的收敛
   - 下排：Agent 2在不同q值下的收敛
   - 每个面板显示Gradient vs PPO的对比

## 📈 示例输出

运行实验后，你会看到类似这样的输出：

```
[gradient-2p] Saved convergence history to results/convergence_history/gradient_q25.0_convergence.json
[PPO] Saved convergence history to results/convergence_history/ppo_q25.0_convergence.json
```

运行绘图脚本后：

```
============================================================
📊 Convergence Plotting Tool
============================================================
✅ Loaded PPO q=25.0 from ppo_q25.0_convergence.json
✅ Loaded PPO q=40.0 from ppo_q40.0_convergence.json
✅ Loaded PPO q=55.0 from ppo_q55.0_convergence.json
✅ Loaded gradient q=25.0 from gradient_q25.0_convergence.json
✅ Loaded gradient q=40.0 from gradient_q40.0_convergence.json
✅ Loaded gradient q=55.0 from gradient_q55.0_convergence.json

📋 Data Summary:
  PPO: 3 q values - [25.0, 40.0, 55.0]
  gradient: 3 q values - [25.0, 40.0, 55.0]

🎨 Generating plots...
✅ Convergence figure saved to results/convergence_comparison.png
✅ Separate agents figure saved to results/convergence_separate_agents.png

✅ All plots generated successfully!
```

## 🔍 技术细节

### PPO Convergence Tracking

**实现位置**: `run/run_two_players.py`

1. **数据收集** (第1052-1059行):
   ```python
   # Per-player sampled efforts (for convergence tracking)
   sample_avg_effort_p1 = last_rollout_stats.get("sample_avg_effort_p1")
   sample_avg_effort_p2 = last_rollout_stats.get("sample_avg_effort_p2")
   
   # Record convergence history
   if sample_avg_effort_p1 is not None and sample_avg_effort_p2 is not None:
       convergence_history["steps"].append(steps_done)
       convergence_history["agent1_effort"].append(float(sample_avg_effort_p1))
       convergence_history["agent2_effort"].append(float(sample_avg_effort_p2))
   ```

2. **数据来源**: `utils/rollout_stats.py`
   - `RolloutStatsAccumulator`类已经在追踪P1和P2的efforts
   - 使用Welford算法进行数值稳定的在线统计

3. **保存时机**: 训练循环结束后，在`return rows`之前

### Gradient Convergence Tracking

**实现位置**: `run/run_two_players.py`

1. **历史记录** (第378-384行):
   ```python
   history = {
       "e1_history": [float(e1)],  # Start with initial value
       "e2_history": [float(e2)],  # Start with initial value
       "step_history": [0],        # Step 0 is initial state
   }
   ```

2. **每步更新** (第401-404行):
   ```python
   # Record convergence history at every step
   history["e1_history"].append(float(e1))
   history["e2_history"].append(float(e2))
   history["step_history"].append(step)
   ```

3. **保存时机**: `run_gradient`函数完成后立即保存

## 📁 文件结构

```
results/
├── convergence_history/          # 新增：Convergence data
│   ├── ppo_q25.0_convergence.json
│   ├── ppo_q40.0_convergence.json
│   ├── ppo_q55.0_convergence.json
│   ├── gradient_q25.0_convergence.json
│   ├── gradient_q40.0_convergence.json
│   └── gradient_q55.0_convergence.json
├── convergence_comparison.png     # 新增：综合对比图
├── convergence_separate_agents.png # 新增：分agent对比图
├── one_stage_two_players.png
├── one_stage_two_players_v2.csv
└── logs/
    └── ...
```

## 🎨 自定义绘图

你可以使用保存的JSON数据创建自定义图表：

```python
import json
import matplotlib.pyplot as plt

# 加载数据
with open('results/convergence_history/ppo_q25.0_convergence.json', 'r') as f:
    ppo_data = json.load(f)

with open('results/convergence_history/gradient_q25.0_convergence.json', 'r') as f:
    gradient_data = json.load(f)

# 创建自定义图
plt.figure(figsize=(10, 6))
plt.plot(gradient_data['steps'], gradient_data['agent1_effort'], label='Gradient Agent1')
plt.plot(ppo_data['steps'], ppo_data['agent1_effort'], label='PPO Agent1')
plt.axhline(y=ppo_data['theoretical_effort'], linestyle=':', label='Theory')
plt.xlabel('Training Steps')
plt.ylabel('Effort')
plt.title('Agent 1 Convergence (q=25)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('custom_convergence.png', dpi=300)
```

## 💡 提示

1. **数据子采样**: PPO训练可能产生大量数据点。绘图脚本会自动子采样（每1000个点取1个）以提高可视化质量

2. **内存管理**: 对于超长训练run，convergence history会占用额外内存。如果内存有限，可以考虑：
   - 只在特定update间隔记录数据
   - 训练后立即保存并清空历史

3. **多次运行**: 如果运行多次实验，JSON文件会被覆盖。考虑在文件名中添加timestamp或seed

## 🐛 故障排查

### 问题：没有生成JSON文件

**检查**:
1. 确认训练成功完成（没有中途崩溃）
2. 检查`results/convergence_history/`目录是否存在
3. 查看训练日志，确认看到 "Saved convergence history" 消息

### 问题：绘图脚本报错找不到数据

**解决**:
```bash
# 检查JSON文件是否存在
ls -lh results/convergence_history/

# 验证JSON格式
python -m json.tool results/convergence_history/ppo_q25.0_convergence.json
```

### 问题：PPO日志中没有p1_effort和p2_effort

**原因**: 只有在selfplay模式下才会分别追踪P1和P2。在vs_opponent模式下，可能只有部分数据。

**解决**: 使用`--rollout-mode selfplay`运行实验

## 📝 更新日志

### 2026-01-15
- ✅ 添加PPO individual agent effort tracking
- ✅ 添加Gradient完整历史记录
- ✅ 实现JSON convergence data保存
- ✅ 创建convergence绘图脚本
- ✅ 添加本文档

---

**相关文件**:
- `run/run_two_players.py`: 主要训练脚本（包含convergence tracking）
- `utils/rollout_stats.py`: Rollout统计追踪器
- `plot_convergence.py`: 绘图脚本
- `CONVERGENCE_TRACKING.md`: 本文档
