# Gradient Parameter Sweep Guide

本指南介绍如何使用参数搜索工具来找到最优的梯度下降参数组合。

## 📋 目录

1. [快速开始](#快速开始)
2. [搜索策略](#搜索策略)
3. [参数空间](#参数空间)
4. [使用示例](#使用示例)
5. [结果分析](#结果分析)
6. [最佳实践](#最佳实践)

## 🚀 快速开始

### 1. 单Q值搜索（推荐开始）

```bash
# 随机搜索（快速，推荐）
python run/run_gradient_sweep.py --strategy random --n-trials 100 --q 40.0

# 网格搜索（全面但耗时）
python run/run_gradient_sweep.py --strategy grid --q 40.0

# 贝叶斯优化（需要安装optuna，最智能）
pip install optuna
python run/run_gradient_sweep.py --strategy bayesian --n-trials 50 --q 40.0
```

### 2. 多Q值鲁棒性搜索（推荐用于最终优化）

```bash
# 随机搜索（快速）
python run/run_gradient_sweep_multi_q.py --strategy random --n-trials 200

# 贝叶斯优化（最智能，需要安装optuna）
pip install optuna
python run/run_gradient_sweep_multi_q.py --strategy bayesian --n-trials 1000
```

### 3. 分析结果

```bash
# 分析单Q值搜索结果
python run/analyze_sweep_results.py results/sweeps/sweep_random_q40.0_*.csv

# 查看多Q值搜索结果
python run/analyze_sweep_results.py results/sweeps/sweep_multiq_random_*.csv
```

## 🔍 搜索策略

### 1. Grid Search（网格搜索）

**优点：**
- 全面覆盖参数空间
- 结果可重现
- 适合小参数空间

**缺点：**
- 组合数爆炸（6个参数 × 多个值 = 数千组合）
- 计算成本高

**适用场景：**
- 参数空间小（< 1000组合）
- 需要全面探索
- 有充足计算资源

```bash
# 限制组合数
python run/run_gradient_sweep.py --strategy grid --max-combinations 500 --q 40.0
```

### 2. Random Search（随机搜索）⭐ 推荐

**优点：**
- 计算效率高
- 适合高维参数空间
- 实现简单

**缺点：**
- 可能错过最优解
- 需要足够多的试验

**适用场景：**
- 参数空间大
- 计算资源有限
- 快速探索

```bash
# 标准随机搜索
python run/run_gradient_sweep.py --strategy random --n-trials 100 --q 40.0

# 大规模搜索
python run/run_gradient_sweep.py --strategy random --n-trials 500 --q 40.0
```

### 3. Bayesian Optimization（贝叶斯优化）

**优点：**
- 智能采样，效率最高
- 自适应探索
- 通常能找到更好的解
- 支持单Q值和多Q值搜索

**缺点：**
- 需要安装optuna
- 实现复杂
- 可能陷入局部最优

**适用场景：**
- 计算成本高（每次评估耗时）
- 需要最优解
- 有充足时间
- 多Q值鲁棒性优化（推荐）

```bash
# 安装依赖
pip install optuna

# 单Q值贝叶斯优化
python run/run_gradient_sweep.py --strategy bayesian --n-trials 50 --q 40.0

# 多Q值贝叶斯优化（推荐用于最终优化）
python run/run_gradient_sweep_multi_q.py --strategy bayesian --n-trials 1000
```

**多Q值贝叶斯优化说明：**
- 目标函数：最小化 `robustness_score`（在所有q值下的鲁棒性）
- 每次试验评估参数在多个q值（默认：25.0, 40.0, 55.0）下的表现
- 优化完成后，自动评估前20个最佳参数组合
- 适合寻找在所有条件下都表现稳定的参数

## 📊 参数空间

### 默认参数范围

| 参数 | 范围 | 说明 |
|------|------|------|
| `lr` | 0.01 - 0.20 | 学习率，控制更新步长 |
| `steps` | 500 - 3000 | 最大迭代次数 |
| `grad_eps` | 0.1 - 1.0 | 有限差分扰动大小 |
| `tol` | 1e-5 - 1e-3 | 收敛容差 |
| `num_samples` | 16 - 256 | Monte Carlo采样数 |
| `init_perturb` | 0.5 - 3.0 | 初始扰动 |

### 参数重要性排序

根据经验，参数重要性（影响结果的程度）：

1. **lr** (学习率) - 最关键，直接影响收敛
2. **grad_eps** (有限差分epsilon) - 影响梯度估计精度
3. **num_samples** (采样数) - 影响梯度估计稳定性
4. **tol** (容差) - 影响收敛判断
5. **steps** (迭代数) - 通常足够大即可
6. **init_perturb** (初始扰动) - 影响较小

## 💡 使用示例

### 示例1: 快速探索（5分钟）

```bash
# 小规模随机搜索，快速了解参数空间
python run/run_gradient_sweep.py \
    --strategy random \
    --n-trials 50 \
    --q 40.0 \
    --verbose
```

### 示例2: 中等规模搜索（30分钟）

```bash
# 中等规模搜索，平衡时间和质量
python run/run_gradient_sweep.py \
    --strategy random \
    --n-trials 200 \
    --q 40.0 \
    --output-dir results/sweeps/medium
```

### 示例3: 大规模鲁棒性搜索（2小时）

```bash
# 多Q值随机搜索
python run/run_gradient_sweep_multi_q.py \
    --strategy random \
    --n-trials 500 \
    --q-values 25.0 40.0 55.0 \
    --output-dir results/sweeps/robust

# 多Q值贝叶斯优化（推荐，更智能）
python run/run_gradient_sweep_multi_q.py \
    --strategy bayesian \
    --n-trials 1000 \
    --q-values 25.0 40.0 55.0 \
    --output-dir results/sweeps/robust
```

### 示例4: 并行加速（需要joblib）

```bash
# 安装并行支持
pip install joblib

# 使用4个并行worker
python run/run_gradient_sweep.py \
    --strategy random \
    --n-trials 200 \
    --parallel 4 \
    --q 40.0
```

### 示例5: 特定参数范围搜索

修改 `run/run_gradient_sweep.py` 中的 `get_parameter_space()` 函数：

```python
def get_parameter_space() -> Dict[str, List]:
    return {
        "lr": [0.05, 0.08, 0.10, 0.12],  # 聚焦在0.05-0.12
        "steps": [1500, 2000],  # 只测试这两个值
        "grad_eps": [0.3, 0.5, 0.8],  # 聚焦在0.3-0.8
        "tol": [1e-4],  # 固定容差
        "num_samples": [32, 64, 128],  # 测试这三个值
        "init_perturb": [1.0, 2.0],  # 只测试两个值
    }
```

## 📈 结果分析

### 1. 查看最佳参数

搜索结果会保存在 `results/sweeps/` 目录下：

- `sweep_*_summary.json` - 最佳参数摘要
- `sweep_*.csv` - 所有试验的详细结果

### 2. 可视化分析

```bash
# 生成参数敏感性图、质量分布图等
python run/analyze_sweep_results.py results/sweeps/sweep_random_q40.0_*.csv
```

生成的图表：
- `parameter_sensitivity.png` - 每个参数对gap的影响
- `quality_distribution.png` - 质量分布
- `top_parameters.png` - 前N个最佳参数组合

### 3. 解读结果

**关键指标：**

- **gap**: 与理论值的差距（越小越好）
- **quality**: Excellent/Good/Fair/Poor
- **efficiency_score**: 综合评分（gap + 收敛效率）
- **robustness_score**: 多Q值搜索的鲁棒性评分

**最佳参数特征：**

- gap < 0.5 (Excellent)
- 收敛稳定（iterations < steps）
- 对称性好（symmetry_gap < 0.1）

## 🎯 最佳实践

### 1. 搜索策略选择

```
小规模探索 (< 100 trials)     → Random Search
中等规模 (100-500 trials)     → Random Search
大规模 (500+ trials)          → Bayesian Optimization
多Q值鲁棒性搜索               → Bayesian Optimization (multi-q) ⭐ 推荐
多Q值鲁棒性搜索（快速）       → Random Search (multi-q)
```

### 2. 参数空间设计

**第一阶段：宽范围探索**
- 使用默认的宽参数范围
- 随机搜索 100-200 trials
- 识别有希望的区域

**第二阶段：精细搜索**
- 缩小到有希望的区域
- 增加采样密度
- 网格搜索或贝叶斯优化

**第三阶段：鲁棒性验证**
- 多Q值测试
- 验证参数在不同条件下的表现
- **推荐使用贝叶斯优化**：`--strategy bayesian --n-trials 1000`
- **推荐使用贝叶斯优化**：`--strategy bayesian --n-trials 1000`

### 3. 评估标准

**单Q值搜索：**
- 主要指标：`gap`（越小越好）
- 次要指标：`efficiency_score`

**多Q值搜索：**
- 主要指标：`robustness_score`（越小越好）
- 关注：`worst_gap`（最坏情况下的表现）

### 4. 计算资源管理

```bash
# 小规模测试（验证脚本工作）
--n-trials 10

# 中等规模（日常使用）
--n-trials 100-200

# 大规模（最终优化）
--n-trials 500-1000

# 并行加速（如果有多个CPU）
--parallel 4
```

### 5. 结果验证

找到最佳参数后，手动验证：

```bash
# 使用找到的最佳参数运行
python run/run_two_players.py \
    --method gradient \
    --q 40.0 \
    --grad-lr 0.08 \
    --grad-steps 1500 \
    --grad-epsilon 0.5 \
    --grad-tol 1e-4 \
    --grad-samples 64 \
    --grad-init-perturb 1.0
```

## 🔧 故障排除

### 问题1: 所有结果都是 "Poor" quality

**可能原因：**
- 参数范围不合适
- q值设置有问题
- 理论值计算错误

**解决方案：**
- 检查理论值是否正确
- 扩大参数搜索范围
- 降低tol值

### 问题2: 结果不稳定

**可能原因：**
- num_samples太小
- grad_eps不合适

**解决方案：**
- 增加num_samples到128或256
- 调整grad_eps到0.3-0.5

### 问题3: 收敛太慢

**可能原因：**
- lr太小
- steps不够

**解决方案：**
- 增加lr到0.1-0.15
- 增加steps到2000-3000

## 📚 相关文件

- `run/run_gradient_sweep.py` - 单Q值参数搜索
- `run/run_gradient_sweep_multi_q.py` - 多Q值鲁棒性搜索
- `run/analyze_sweep_results.py` - 结果分析工具
- `config/one_stage_two_players.py` - 默认配置
- `run/run_two_players.py` - 主实验脚本

## 🎓 进阶技巧

### 1. 自定义评估指标

修改 `evaluate_parameters()` 函数中的评估逻辑：

```python
# 自定义效率评分
efficiency_score = gap + 0.01 * (iterations / params["steps"]) + 0.1 * symmetry_gap
```

### 2. 条件参数空间

根据其他参数动态调整参数范围：

```python
if params["lr"] > 0.1:
    # 高学习率时，需要更多采样
    params["num_samples"] = max(params["num_samples"], 128)
```

### 3. 早停策略

在搜索过程中，如果发现某些参数组合明显不好，可以提前跳过：

```python
# 在evaluate_parameters中添加
if gap > 10.0:  # 明显不好，提前返回
    return {"status": "poor", "gap": gap, ...}
```

---

**提示**: 开始使用前，建议先用小规模试验（`--n-trials 10`）验证脚本正常工作。

