# Gradient Parameter Sweep - Quick Reference

## 🚀 快速命令

### 基础搜索

```bash
# 随机搜索（推荐开始）
python run/run_gradient_sweep.py --strategy random --n-trials 100 --q 40.0

# 网格搜索（全面但慢）
python run/run_gradient_sweep.py --strategy grid --q 40.0

# 贝叶斯优化（最智能，需要optuna）
pip install optuna
python run/run_gradient_sweep.py --strategy bayesian --n-trials 50 --q 40.0
```

### 多Q值鲁棒性搜索

```bash
# 随机搜索（快速）
python run/run_gradient_sweep_multi_q.py --strategy random --n-trials 200

# 贝叶斯优化（推荐，最智能）
pip install optuna
python run/run_gradient_sweep_multi_q.py --strategy bayesian --n-trials 1000
```

### 结果分析

```bash
# 分析搜索结果
python run/analyze_sweep_results.py results/sweeps/sweep_*.csv
```

## 📊 参数说明

| 参数 | 默认范围 | 影响 |
|------|---------|------|
| `lr` | 0.01-0.20 | 学习率，最关键 |
| `grad_eps` | 0.1-1.0 | 有限差分扰动 |
| `num_samples` | 16-256 | Monte Carlo采样数 |
| `tol` | 1e-5-1e-3 | 收敛容差 |
| `steps` | 500-3000 | 最大迭代数 |
| `init_perturb` | 0.5-3.0 | 初始扰动 |

## 🎯 推荐工作流

### 阶段1: 快速探索（5-10分钟）
```bash
python run/run_gradient_sweep.py --strategy random --n-trials 50 --q 40.0 --verbose
```

### 阶段2: 中等规模搜索（30分钟）
```bash
python run/run_gradient_sweep.py --strategy random --n-trials 200 --q 40.0
```

### 阶段3: 鲁棒性验证（1-2小时）
```bash
# 随机搜索（快速）
python run/run_gradient_sweep_multi_q.py --strategy random --n-trials 500

# 贝叶斯优化（推荐，更智能）
python run/run_gradient_sweep_multi_q.py --strategy bayesian --n-trials 1000
```

### 阶段4: 结果分析
```bash
python run/analyze_sweep_results.py results/sweeps/sweep_*.csv
```

## 📁 输出文件

- `results/sweeps/sweep_*_summary.json` - 最佳参数摘要
- `results/sweeps/sweep_*.csv` - 详细结果
- `results/sweeps/analysis/*.png` - 可视化图表

## 🔧 常用选项

```bash
# 并行加速（需要joblib）
--parallel 4

# 自定义输出目录
--output-dir results/my_sweep

# 详细输出
--verbose

# 自定义随机种子
--seed 42
```

## 💡 最佳实践

1. **从小规模开始**: 先用 `--n-trials 10` 验证
2. **逐步扩大**: 50 → 200 → 500
3. **多Q值验证**: 找到参数后，用多Q值搜索验证鲁棒性
4. **分析结果**: 使用分析工具理解参数影响

## 🎓 评估指标

- **gap**: 与理论值差距（越小越好，< 0.5为Excellent）
- **efficiency_score**: 综合评分（gap + 收敛效率）
- **robustness_score**: 多Q值鲁棒性（越小越好）

---

详细文档: `docs/gradient_sweep_guide.md`

