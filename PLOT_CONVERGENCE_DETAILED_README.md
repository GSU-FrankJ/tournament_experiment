# Convergence Plot Generator - Usage Guide

## 概述

`plot_convergence_detailed.py` 是一个脚本，用于从convergence history JSON文件生成详细的收敛图表，展示两个agent的学习过程。

## 功能特性

### 生成两种类型的图表：

1. **Combined Plot** (combined.png)
   - 在同一张图上显示两个agent的effort曲线
   - 包含理论最优值参考线
   - 显示平均effort（绿色虚线）
   - 包含最终状态信息框

2. **Separated Plot** (separated.png)
   - 两个独立的子图，分别显示每个agent
   - 更清晰地观察单个agent的收敛过程
   - 每个子图都包含理论值参考和最终状态注释

## 使用方法

### 基本用法 - 生成所有图表

```bash
cd /home/fjiang4/tournament_experiment-4
python3 results/plot_convergence_detailed.py
```

这将处理 `results/convergence_history/` 中的所有JSON文件，并在 `results/convergence_plots/` 下按算法分类保存图表。

### 高级用法

#### 1. 仅处理特定算法

```bash
# 仅处理PPO算法
python3 results/plot_convergence_detailed.py --algorithm PPO

# 仅处理gradient算法
python3 results/plot_convergence_detailed.py --algorithm gradient
```

#### 2. 仅处理特定q值

```bash
# 仅处理q=55.0的数据
python3 results/plot_convergence_detailed.py --q 55.0

# 仅处理q=25.0的数据
python3 results/plot_convergence_detailed.py --q 25.0
```

#### 3. 组合过滤

```bash
# 仅处理PPO算法的q=40.0数据
python3 results/plot_convergence_detailed.py --algorithm PPO --q 40.0
```

#### 4. 处理单个文件

```bash
python3 results/plot_convergence_detailed.py \
  --file results/convergence_history/ppo_q55.0_convergence.json
```

#### 5. 自定义输出目录

```bash
python3 results/plot_convergence_detailed.py \
  --output-dir my_custom_plots
```

## 输出结构

生成的图表按以下结构组织：

```
results/convergence_plots/
├── gradient/
│   ├── q25.0_combined.png
│   ├── q25.0_separated.png
│   ├── q40.0_combined.png
│   ├── q40.0_separated.png
│   ├── q55.0_combined.png
│   └── q55.0_separated.png
└── ppo/
    ├── q25.0_combined.png
    ├── q25.0_separated.png
    ├── q40.0_combined.png
    ├── q40.0_separated.png
    ├── q55.0_combined.png
    └── q55.0_separated.png
```

## 图表内容说明

### Combined Plot 包含：
- 🔵 **Agent 1** (蓝色实线): 第一个agent的effort轨迹
- 🟠 **Agent 2** (橙色实线): 第二个agent的effort轨迹
- ⚫ **Theoretical** (黑色虚线): 理论最优effort值
- 🟢 **Average** (绿色点线): 两个agent的平均effort
- 📊 **Info Box**: 显示最终状态和gap

### Separated Plot 包含：
- **上图**: Agent 1的收敛过程
- **下图**: Agent 2的收敛过程
- 每个子图都有：
  - Agent的effort曲线
  - 理论值参考线
  - 最终状态标注框（Final值和Gap）

## 示例场景

### 场景1: 快速查看所有结果

```bash
# 生成所有图表
python3 results/plot_convergence_detailed.py

# 查看生成的文件
ls -lh results/convergence_plots/*/*.png
```

### 场景2: 比较PPO在不同q值下的表现

```bash
# 仅生成PPO的图表
python3 results/plot_convergence_detailed.py --algorithm PPO

# 打开图表进行比较
# results/convergence_plots/ppo/q25.0_combined.png
# results/convergence_plots/ppo/q40.0_combined.png
# results/convergence_plots/ppo/q55.0_combined.png
```

### 场景3: 调试特定实验

```bash
# 运行新实验
python run/run_two_players.py --method ppo --rollout-mode selfplay --q 55 \
  --episodes 2048000 --seed 50 --theory-align-v2 --enable-convergence-eval \
  --cheap-gate-profile relaxed

# 为新生成的数据创建图表
python3 results/plot_convergence_detailed.py --algorithm PPO --q 55.0
```

### 场景4: 论文插图准备

```bash
# 生成所有高分辨率图表（默认300 DPI）
python3 results/plot_convergence_detailed.py

# 图表会自动保存为高质量PNG，适合论文使用
```

## 命令行参数完整列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--convergence-dir` | str | `results/convergence_history` | convergence JSON文件所在目录 |
| `--output-dir` | str | `results/convergence_plots` | 输出图表保存目录 |
| `--algorithm` | str | None | 过滤算法 (PPO/gradient) |
| `--q` | float | None | 过滤q值 (如 25.0, 40.0, 55.0) |
| `--file` | str | None | 处理单个指定文件 |

## 输出示例

运行脚本后，您会看到：

```
============================================================
📊 Detailed Convergence Plot Generator
============================================================
Found 6 convergence files
============================================================
✅ Loaded gradient_q25.0_convergence.json

📊 Generating plots for gradient q=25.0
✅ Saved combined plot: results/convergence_plots/gradient/q25.0_combined.png
✅ Saved separated plot: results/convergence_plots/gradient/q25.0_separated.png
...
============================================================
✅ Processed 6 files
📁 Plots saved to: results/convergence_plots

✅ Done!
```

## 与其他绘图脚本的关系

- **`plot_convergence.py`**: 生成多算法对比图（所有算法和q值在一张图上）
- **`plot_convergence_detailed.py`**: 生成单个算法的详细图表（本脚本，更适合深入分析）

推荐工作流程：
1. 先运行 `plot_convergence.py` 获得总体概览
2. 再运行 `plot_convergence_detailed.py` 获得详细分析图表

## 故障排除

### 问题1: "No convergence files found"
**解决**: 确保先运行了实验，生成了convergence history文件

```bash
# 运行实验生成数据
python run/run_two_players.py --method gradient --q 25
python run/run_two_players.py --method ppo --rollout-mode selfplay --q 25 --episodes 2048000
```

### 问题2: "ModuleNotFoundError: No module named 'matplotlib'"
**解决**: 安装matplotlib

```bash
python3 -m pip install matplotlib numpy
```

### 问题3: 图表显示设备错误
**解决**: 脚本使用非交互式后端，不需要显示设备

图表会自动保存为文件，无需GUI环境。

## 自定义图表样式

如需修改图表样式（颜色、线宽、字体等），编辑脚本中的相应部分：

```python
# 在 plot_combined() 或 plot_separated() 函数中修改

# 修改颜色
ax.plot(steps, agent1_effort, color='#YOUR_COLOR', ...)

# 修改线宽
linewidth=3.0  # 更粗的线

# 修改字体大小
fontsize=16  # 更大的字体
```

---

**创建日期**: 2026-01-20
**作者**: AI Assistant
**相关文件**: `results/plot_convergence_detailed.py`
