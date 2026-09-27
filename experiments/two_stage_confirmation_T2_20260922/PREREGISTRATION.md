# T=2 held-out confirmation 预登记

**本文件在任何 confirmation run 启动之前冻结。** 协议为
`MultiStage/two_stage/protocol/FINAL_T2_PROTOCOL_20260922.json`（A400 fixed-budget /
B600 cadence 25 / C1000 cadence 25），不因本轮中途结果改动。

## 一、设计

| 项 | 值 |
| --- | --- |
| 协议 | `FINAL_T2_PROTOCOL_20260922.json`，逐字段派生自 0915 正式协议，唯一改动为 `verifier_timeout: 100 → {A:100, B:25, C:25}` |
| Seeds | **q=50 用 10101–10110，q=60 用 10111–10120**，两个列表不相交（**不配对**，与 0915 的配对设计不同） |
| 运行数 | q=50 十条 + q=60 十条 = **20 条**，20 个互不相同的 seed |
| Seed 未使用核查 | 2026-09-22 只读检查 `experiments/`、`MultiStage/`、`results/` 的全部 JSON 配置与 run 目录名：10101–10120 均未作为 `"seed"` 字段出现（在用 seed 共 39 个），也不存在对应 run 目录。结论限于已检查记录 |
| 运行器 | `run/run_final_dp_br_round3_dense.py`（worktree `candidate-search-recovery-data-92d49a`），6 个协议模块的 sha256 记入 manifest |
| 并发 | 每进程单线程，最多 10 并发 |

`weights_every = 25` 写入 manifest：每 25 个 global update 导出 actor+critic 权重。
这是**非行为字段**——权重导出是只读的、不消耗 RNG，两批 dense 已经实证轨迹逐位不变
（共享 call 上最大绝对差 0.000e+00）。它只用于事后取证，不参与任何判定。

## 二、主要报告指标

按 q 分别报告，分母固定为 N=10；比例附 95% Wilson 区间。两个 q 是**不同的博弈**，
跨 q 的合计只作描述，不作为一个 N=20 的推断。条件率分母为 0 时记 NA。

| 指标 | 定义 |
| --- | --- |
| **Candidate discovery rate** | C 预算内出现首次「同点数值有效 + dReach/ΔW ≤ 0.01 + C_dev ≤ 0.04」的 development 检查的 run 数 / 10。含 C1000 处的末次检查 |
| **Final certification rate**（条件） | `final_overall_pass` 数 / 找到候选的 run 数；同时单列 `final_joint_pass` 数 / 找到候选的 run 数 |
| **End-to-end success rate** | `final_joint_pass` 数 / 10 |
| Budget-exhaustion rate | C 预算耗尽且无候选的 run 数 / 10 |
| Operational status | 完成 / 中断 / 技术失败分别计数，不从分母中静默删除 |

`final_overall_pass` = development 与 final 均数值有效、`dReach_final/ΔW ≤ 0.01`、
且两项数值细化差 ≤ 0.002。`final_joint_pass` 再要求同一 θ 在 0.05 步长完整网格上
`C_all ≤ 0.04`。两者不得混称。

## 三、Analytical recovery + numerical refinement

对**全部 first-eligible 候选**计算（包括 final 未通过者），并另列 `final_joint_pass`
子集及其样本数。无候选 run 的候选恢复指标记 NA，末点诊断与候选统计分开。

恢复指标（沿用 0915 口径）：第一期努力的真值/估计/有符号误差/绝对误差/相对误差；
第二期完整 D₂ 与正努力域（严格 |d| < 2q，边界归 tail）的 MAE / RMSE / 最大绝对误差；
d=0 的绝对与相对误差，另报实际网格最大值与位置；tail（|d| ≥ 2q）的平均与最大努力；
evenness（完整 D₂ 上 |ê₂(d) − ê₂(−d)| 的最大与平均）；on-path 三角分布加权的
MAE / RMSE / E[ê₂] 及 |E[ê₂] − g₁|；跨期关系 ê₁ − E[ê₂] 的符号、绝对值及相对 g₁ 的比例。
on-path 主表用分半 64 点 Gauss–Legendre，另按 0.5/0.25/0.125/0.0625 四档步长细化
作求积敏感性，**不据此更换候选**。

Numerical refinement 单列：同一冻结权重在 development（state 4 / effort 1 / GL 16）
与 final（state 2 / effort 0.5 / GL 32）两档下的 `|dReach_final − dReach_dev|/ΔW` 与
`|EXP_root_final − EXP_root_dev|/ΔW`，阈值 0.002。它与「继续训练的 verifier-triggered
refinement」是不同操作，本轮不做后者。

**本轮没有独立的 analytical-recovery 通过门槛**，因此不定义 recovery 成功率，
也不把非零误差记为失败。可以报告「通过偏离认证的候选具有怎样的解析恢复精度」，
不能据认证通过宣称策略已逐点恢复解析均衡。

连续指标一律报告均值、样本 SD（ddof=1）、中位数、IQR（NumPy linear 规则）与范围。
成本报告实际 A/B/C updates、episodes、transitions、development calls、训练耗时与离线验证耗时；
首次候选的取得时间与预算耗尽分别展示。

## 四、执行规则

不依据中途结果更换配置、增删 seed 或提前结束整批。当前 runner 不支持保存随机数状态的
中途续跑；仅对**已确认的基础设施故障**且未完成的 run，允许同一 seed 从起点完整重跑，
另存 attempt 并保留异常记录，按预定 attempt 顺序采用首次完整完成者，不看结果选优。
**已完整完成但未通过搜索或认证的 run 不因失败重跑，也不补抽 seed 替换。**
最终验证未通过时保留失败候选与原结果，不恢复 PPO、不另挑 checkpoint、不放宽阈值。

## 五、统计分辨率（事前声明）

N=10 的 95% Wilson 区间：10/10 → [72.2, 100.0]；9/10 → [59.6, 98.2]；
8/10 → [49.0, 94.3]；7/10 → [39.7, 89.2]；6/10 → [31.3, 83.2]；5/10 → [23.7, 76.3]。

即 **N=10 无法把 80% 与 60% 区分开**。若用于判断 T=3 的 practical gate
（q50 ≥ 80%、q60 ≥ 90% 发现率），本轮只能给出点估计与一个覆盖面很宽的区间，
不能做「是否达标」的显著性判定。这一点在结果报告中必须复述。

## 六、与既有 40 条的关系

seeds 10001–10020 的 40 条运行是在 cadence 100 下完成的，并已被逐条诊断
（分层、漏采归因、A/B 段重放）。它们是**开发与方案选择**数据，不进入本轮分母，
本轮也不合并报告两批的率。cadence-25 在那批子集上的转化计数（4/11）只作为
协议改动的动机陈述，不作为协议的通过率。
