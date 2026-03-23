# Phase 02: Three-Players 全身检查

## 根因分析

### 已排除的假设

| 假设 | 排除证据 |
|------|---------|
| **H1: 初始 entropy 塌缩** | 3p q=25 seeds 123/456/789/1024 起始 ent=-3.7（高度集中），无崩溃，但 gap 仍 6.0-6.2 |
| **H4: 纯自对弈 vs 滞后对手** | 代码审计证实 2p **也是**纯自对弈——`act_opponent()` 从未被调用，lag_prob 被计算但未使用 |

### 已确认的结构性差异（2p vs 3p）

| 方面 | two_players | three_players |
|------|------------|---------------|
| 每步 transition 数 | 2 → 8192/update | 3 → 12288/update |
| 每 update gradient steps | 48 (8×6 epochs) | 72 (12×6 epochs) |
| 胜率 | pairwise | rank-order (3 人取最高) |
| 对手机制 | **纯自对弈** (lag 代码存在但未使用) | **纯自对弈** |
| 超参 | 全部相同 | 全部相同 |

### 确认的根因链

**1. 3p 的 per-transition 梯度信号比 2p 弱 ~27%**

模拟验证（N=500K）：advantage × effort_deviation proxy 在 effort=57 处：
- 2p: +0.0111
- 3p: +0.0081（方向正确，但弱 27%）

原因：3-player rank-order 中 30.7% 的 loser 转化 effort 在 mean 之上（push=-5.48），部分抵消 winner 的正向信号。2p 不存在此问题。

**2. 弱信号无法对抗 entropy bonus → 策略去集中化**

entropy_coef schedule（start=0.03, end=0.005）为 2p 调优。关键对比：
- 2p q=35: entropy 从 -1.86 **下降到** -4.68（集中化，Δ=-2.82）
- 3p q=25 seed=123（高集中度起步）: entropy 从 -3.69 **上升到** -2.75（去集中化，Δ=+0.94）
- 3p q=35 所有 seed: entropy 收敛到 -2.2 区间

3p 的策略**反向发散**——entropy bonus 力量超过梯度集中力。

**3. 弥散策略 → 弱 ∇log π → KL → 0 → 策略冻住**

Beta 分布的 ∇log π 正比于 concentration (α+β)。entropy -2.2 vs -4.7 意味着 3p 策略的有效梯度可能是 2p 的 1/5~1/10。

实测：update 1000 时 KL = 0.00005 (3p) vs 0.007 (2p)，差 140 倍。3p 策略实质冻结。

**4. 最终假平衡**: 在 entropy ≈ -2.2, effort ≈ NE-5.5 处，梯度力 = entropy bonus 力，但未到 NE。

### 系统性证据

所有 3p 结果都**低于 NE**，gap 5-9，跨 q 值一致：
- q=25: 81.3 vs NE=87.5（gap 6.2，包括高集中度起步的 seed）
- q=35: 57.0 vs NE=62.5（gap 5.5）
- q=40: 45.6-47.0 vs NE=54.7（gap 7.7-9.1）

## 实验计划

### 第一轮：关键验证（单 seed=42，快速）

**实验 A — Entropy coefficient 扫描**（直接测试根因）

降低 entropy_coef 让策略能集中。如果根因正确，更低的 entropy → 更集中 → 更强 ∇log π → 收敛。

```bash
# baseline (当前): entropy_start=0.03, entropy_end=0.005
# 测试更低的 entropy_end
for ent_end in 0.003 0.001 0.0005; do
  python run/run_three_players.py --method ppo --q 35 --seed 42 \
    --episodes 6144000 --exploit-eps 0.02 --entropy-end $ent_end
done
```

需先确认 `--entropy-end` CLI flag 存在。

**实验 B — 减少 gradient steps per update**（测试 72 vs 48 的影响）

将 steps_per_update 从 4096 降到 2731（使 transition 数从 12288 降到 ~8193，匹配 2p）。

```bash
python run/run_three_players.py --method ppo --q 35 --seed 42 \
  --episodes 6144000 --exploit-eps 0.02 --steps-per-update 2731
```

或者增大 minibatch_size 到 1536（12288/1536=8 minibatches/epoch，匹配 2p 的 8）。

**实验 C — 梯度审计**（确认 PPO 梯度方向准确率）

在每个 update 记录 PPO 实际更新方向 vs 解析梯度方向的一致率。如果准确率 ~50%（随机），证实信号淹没在噪声中。

### 第二轮：基于第一轮结果

如果 A 成功（entropy_end 降低后收敛）：
- 找到最小有效 entropy_end，用 5 seeds 验证
- 同时检查 q=40、q=25 是否也改善

如果 B 成功（减少 gradient steps 后收敛）：
- 说明 over-optimization 是主因，调整 steps_per_update 或 epochs

如果 A+B 都不够：
- 组合 A+B
- 考虑更激进的改动：只存 1 个 transition per step、用不同 advantage normalization

## 验证标准

- 3p q=35 mean gap ≤ 3（与 2p 的 ~1 可比的数量级）
- entropy 在训练过程中下降（而非上升）
- KL 维持在 0.001+ 水平（策略持续更新）

## 附录：解析梯度验证

在所有 effort 水平上，3p 和 2p 的解析梯度**完全相同**（ratio=1.000）。NE 都是 62.5。问题 100% 在 PPO 估计层面，不在理论层面。

```
e=50: 3p=+0.010000  2p=+0.010000
e=57: 3p=+0.004400  2p=+0.004400
e=62: 3p=+0.000400  2p=+0.000400
e=62.5: both=0 (NE)
```
