# PPO 默认配置说明

## 更新内容

从此版本开始，**PPO 方法的默认配置已更新**，使其更适合现代实验设置。

## 新的 PPO 默认值

当使用 `--method ppo` 时，以下选项将**自动启用**（无需手动指定）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rollout-mode` | `selfplay` | 两个agent都使用learner policy，存储两者的transitions |
| `--theory-align-v2` | `True` | 启用mean+concentration policy head，带variance penalty |
| `--enable-convergence-eval` | `True` | 启用收敛评估和早停机制 |
| `--cheap-gate-profile` | `relaxed` | 使用宽松的KL阈值（适合theory-align-v2） |

## 使用方法

### 简化命令（推荐）

**之前需要：**
```bash
python run/run_two_players.py --method ppo --rollout-mode selfplay --q 25 \
  --episodes 2048000 --seed 50 --theory-align-v2 --enable-convergence-eval \
  --cheap-gate-profile relaxed
```

**现在只需：**
```bash
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 50
```

所有现代配置都会自动应用！✨

### 覆盖默认值

如果需要禁用某个默认选项，可以使用对应的禁用标志：

```bash
# 禁用 theory-align-v2
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 50 \
  --no-theory-align-v2

# 禁用 convergence-eval
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 50 \
  --no-convergence-eval

# 使用不同的 rollout-mode
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 50 \
  --rollout-mode vs_opponent

# 使用不同的 cheap-gate-profile
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 50 \
  --cheap-gate-profile aggressive
```

## Gradient 方法的默认值

Gradient 方法保持传统默认值（**未改变**）：

| 参数 | 默认值 |
|------|--------|
| `--rollout-mode` | `vs_opponent` |
| `--theory-align-v2` | `False` |
| `--enable-convergence-eval` | `False` |

## 运行时确认

运行 PPO 实验时，您会在日志开头看到应用的默认值：

```
[config] PPO default: rollout_mode='selfplay'
[config] PPO default: theory_align_v2=True
[config] PPO default: enable_convergence_eval=True
[config] PPO default: cheap_gate_profile='relaxed'
[TheoryAlignV2] enabled: entropy=0, mean+conc head, var_coef=5e-2, ...
[PPO] Rollout mode: SELFPLAY
[Convergence] cheap_gate_profile=relaxed
```

## 迁移指南

### 旧脚本迁移

如果您有使用旧命令格式的脚本，它们仍然可以正常工作：

```bash
# 旧格式（仍然有效）
python run/run_two_players.py --method ppo --rollout-mode selfplay \
  --theory-align-v2 --enable-convergence-eval --cheap-gate-profile relaxed \
  --q 25 --episodes 2048000 --seed 50

# 新格式（简化，推荐）
python run/run_two_players.py --method ppo \
  --q 25 --episodes 2048000 --seed 50
```

两者效果完全相同。

### 何时需要明确指定参数

只有当您想要**非默认行为**时才需要明确指定：

```bash
# 示例1: PPO但不使用theory-align-v2
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 \
  --no-theory-align-v2

# 示例2: PPO但使用vs_opponent模式
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 \
  --rollout-mode vs_opponent

# 示例3: PPO但使用更严格的convergence profile
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 \
  --cheap-gate-profile aggressive
```

## 常见实验场景

### 场景1: 标准PPO实验（最常用）
```bash
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 50
python run/run_two_players.py --method ppo --q 40 --episodes 2048000 --seed 50
python run/run_two_players.py --method ppo --q 55 --episodes 2048000 --seed 50
```

### 场景2: 对比实验（PPO vs Gradient）
```bash
# Gradient (传统方法，用于baseline)
python run/run_two_players.py --method gradient --q 25

# PPO (现代方法，自动使用最佳配置)
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 50
```

### 场景3: 消融实验（关闭某些特性）
```bash
# 测试没有theory-align-v2的PPO
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 50 \
  --no-theory-align-v2

# 测试没有early stopping的PPO
python run/run_two_players.py --method ppo --q 25 --episodes 2048000 --seed 50 \
  --no-convergence-eval
```

## 技术细节

### 默认值应用逻辑

默认值在 `_run_cli()` 函数开始时根据 `--method` 参数应用：

```python
if args.method == "ppo":
    if args.rollout_mode is None:
        args.rollout_mode = "selfplay"
    if args.theory_align_v2 is None:
        args.theory_align_v2 = True
    if args.enable_convergence_eval is None:
        args.enable_convergence_eval = True
    if args.cheap_gate_profile is None:
        args.cheap_gate_profile = "relaxed"
```

### 参数优先级

1. **明确指定的参数** - 最高优先级
2. **Method-specific 默认值** - 次优先级（本次更新）
3. **argparse 默认值** - 最低优先级

### 相关配置文件

这些默认值与 `config/one_stage_two_players.py` 中的配置协同工作：
- `theory_align_v2` 相关参数从配置文件读取
- `convergence` 配置（包括 `relaxed` profile）在配置文件中定义
- `asymmetric_warmup_updates` 等其他配置也在配置文件中

## 帮助信息

查看完整的参数说明：

```bash
python run/run_two_players.py --help
```

重点关注：
- `--rollout-mode`: 说明了默认行为
- `--theory-align-v2` / `--no-theory-align-v2`: 启用/禁用选项
- `--enable-convergence-eval` / `--no-convergence-eval`: 启用/禁用选项
- `--cheap-gate-profile`: 说明了根据method选择的默认profile

## 向后兼容性

✅ **完全兼容** - 所有旧命令仍然有效
- 明确指定的参数会覆盖默认值
- 没有破坏性变更
- 只是简化了常用命令的书写

## 总结

**核心改进**: 现在运行PPO实验只需要指定核心参数（q, episodes, seed），所有现代化配置自动启用。

**受益**: 
- ✅ 命令更简洁
- ✅ 减少人为错误
- ✅ 自动使用最佳实践配置
- ✅ 保持向后兼容

---

**更新日期**: 2026-03-17
**版本**: v2.0
