# 梯度参数搜索评估标准更新

## 📋 更新内容

评估标准已从**平均effort接近理论值**改为**e1和e2都接近理论值，且保持对称**。

## 🔄 主要变化

### 旧标准（已废弃）
- **主要指标**: `gap = |(e1+e2)/2 - theoretical_effort|`
- **问题**: 即使e1和e2偏离理论值，只要平均值接近，gap也会很小
- **示例**: e1=80, e2=95, theory=87.5 → gap=0（但e1和e2都偏离很大）

### 新标准（当前）
- **主要指标**: `max_gap = max(|e1 - theoretical_effort|, |e2 - theoretical_effort|)`
- **次要指标**: `symmetry_gap = |e1 - e2|`
- **综合评分**: `efficiency_score = max_gap + 0.1 * symmetry_gap + convergence_penalty`
- **优势**: 确保e1和e2都接近理论值，且保持对称

## 📊 新的评估指标

### 1. 主要指标

```python
gap_e1 = abs(e1 - theoretical_effort)
gap_e2 = abs(e2 - theoretical_effort)
max_gap = max(gap_e1, gap_e2)  # 主要评估指标
```

### 2. 辅助指标

```python
avg_gap = 0.5 * (gap_e1 + gap_e2)  # 平均gap（参考）
symmetry_gap = abs(e1 - e2)  # 对称性
```

### 3. 综合评分

```python
efficiency_score = max_gap + 0.1 * symmetry_gap + 0.01 * (iterations / steps)
```

**评分说明**:
- `max_gap`: 主要项，确保e1和e2都接近理论值
- `0.1 * symmetry_gap`: 对称性惩罚，鼓励e1 ≈ e2
- `0.01 * (iterations / steps)`: 收敛效率惩罚

## 🎯 质量评级

基于 `max_gap` 进行评级：

- **Excellent**: `max_gap < 0.5`
- **Good**: `max_gap < 1.0`
- **Fair**: `max_gap < 5.0`
- **Poor**: `max_gap ≥ 5.0`

## 📈 输出格式变化

### 单Q值搜索输出

```json
{
  "best_metrics": {
    "max_gap": 0.775427,        // 最大偏差（主要指标）
    "gap_e1": 0.775427,         // e1的偏差
    "gap_e2": 0.037093,         // e2的偏差
    "avg_gap": 0.406260,        // 平均偏差（参考）
    "e1": 86.724573,            // e1的值
    "e2": 87.462907,            // e2的值
    "theoretical_effort": 87.5, // 理论值
    "symmetry_gap": 0.738334,   // 对称性gap
    "quality": "Good",
    "efficiency_score": 0.859260
  }
}
```

### 多Q值搜索输出

```json
{
  "best_metrics": {
    "mean_gap": 0.000117,       // 平均max_gap
    "worst_gap": 0.000120,      // 最差max_gap
    "worst_q": 55.0,            // 最差情况对应的q值
    "mean_symmetry_gap": 0.0001, // 平均对称性gap
    "robustness_score": 0.000183
  },
  "per_q_details": {
    "q_25.0": {
      "e1": 87.500113,
      "e2": 87.500118,
      "theoretical": 87.5,
      "max_gap": 0.000118,
      "gap_e1": 0.000113,
      "gap_e2": 0.000118,
      "symmetry_gap": 0.000005
    }
  }
}
```

## 🔍 使用示例

### 运行搜索（评估标准已自动更新）

```bash
# 单Q值搜索
python run/run_gradient_sweep.py --strategy random --n-trials 100 --q 25.0

# 多Q值搜索
python run/run_gradient_sweep_multi_q.py --strategy random --n-trials 200
```

### 查看结果

搜索完成后，结果会显示：
- e1和e2的具体值
- 每个effort与理论值的差距
- 最大gap（主要评估指标）
- 对称性gap

### 分析结果

```bash
python run/analyze_sweep_results.py results/sweeps/sweep_*.csv
```

分析输出会显示：
- Max Gap统计（而不是平均gap）
- e1和e2的详细信息
- 对称性分析

## ✅ 验证新标准

运行测试验证新标准：

```bash
python -c "
from run.run_gradient_sweep import evaluate_parameters, get_parameter_bounds, random_search
from config.one_stage_two_players import config as base_config

cfg = dict(base_config)
bounds = get_parameter_bounds()
test_params = random_search(bounds, 1, seed=42)[0]
result = evaluate_parameters(cfg, test_params, 25.0, verbose=False)

print(f'理论值: {result[\"theoretical_effort\"]:.6f}')
print(f'e1: {result.get(\"e1\", 0):.6f} (gap: {result.get(\"gap_e1\", 0):.6f})')
print(f'e2: {result.get(\"e2\", 0):.6f} (gap: {result.get(\"gap_e2\", 0):.6f})')
print(f'最大gap: {result[\"gap\"]:.6f}')
"
```

## 🎓 设计理念

### 为什么使用max_gap而不是avg_gap？

**场景示例**:
- 理论值: 87.5
- 情况A: e1=87.0, e2=88.0 → avg_gap=0.25, max_gap=0.5 ✅ 好
- 情况B: e1=80.0, e2=95.0 → avg_gap=0.0, max_gap=7.5 ❌ 差

使用max_gap能确保**两个effort都接近理论值**，而不是只有平均值接近。

### 为什么加入对称性惩罚？

理想情况下，e1和e2应该相等（对称均衡）。对称性惩罚鼓励算法找到对称解。

## 📝 注意事项

1. **向后兼容**: 结果中仍保留 `final_effort` 字段（平均值），但主要评估使用 `max_gap`
2. **CSV文件**: 新增了 `e1`, `e2`, `gap_e1`, `gap_e2`, `avg_gap` 等字段
3. **排序**: 所有排序和最佳参数选择都基于 `efficiency_score`（包含max_gap）

## 🔗 相关文件

- `run/run_gradient_sweep.py` - 单Q值搜索（已更新）
- `run/run_gradient_sweep_multi_q.py` - 多Q值搜索（已更新，支持贝叶斯优化）
- `run/analyze_sweep_results.py` - 结果分析（已更新）

## 🆕 最新更新（2025-12-02）

### 多Q值贝叶斯优化

多Q值搜索现在支持贝叶斯优化策略：

```bash
# 安装依赖
pip install optuna

# 运行多Q值贝叶斯优化
python run/run_gradient_sweep_multi_q.py --strategy bayesian --n-trials 1000
```

**特点：**
- 使用 Optuna 的 TPE 采样器进行智能参数搜索
- 目标函数：最小化 `robustness_score`（在所有q值下的鲁棒性）
- 每次试验评估参数在多个q值下的表现
- 优化完成后，自动评估前20个最佳参数组合
- 适合寻找在所有条件下都表现稳定的参数

**推荐使用场景：**
- 最终参数优化（替代随机搜索）
- 计算资源充足时
- 需要找到最优鲁棒参数时

---

**更新日期**: 2025-12-02
**版本**: 2.1

