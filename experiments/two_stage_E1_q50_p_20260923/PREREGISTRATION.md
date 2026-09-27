# E.1 预登记：把 q=50 的单次成功概率 p 测准

**本文件在任何 E.1 run 启动之前冻结。**

## 一、目的

冻结协议下 q=50 的单次 candidate-discovery rate 目前只有一个测量：7/10 = 70.0%，
95% Wilson 区间 [39.7, 89.2]，宽 49.5 个百分点。该区间宽到无法支持两件事：

1. 判断 q=50 是否明显低于 T=3 practical gate 的 80%；
2. 确定 multiple-restart 方案中每格需要的 restart 次数 k —— k 的选择完全是 p 的函数。

E.1 追加 20 条同协议、全新 seed 的运行，把 q=50 的样本量提到 n=30。

## 二、设计

| 项 | 值 |
| --- | --- |
| 协议 | `MultiStage/two_stage/protocol/FINAL_T2_PROTOCOL_20260922.json`，逐字段不变 |
| q | 仅 50 |
| Seeds | **10121–10140**，与 10101–10120（confirmation）及 10001–10020（0915）均不相交 |
| 运行数 | **20**，固定；不因中途结果增减、不提前停止 |
| 运行器 | `run/run_final_dp_br_round3_dense.py`，协议模块 sha256 与冻结时逐一核对 |
| 并发 | 10，每进程单线程 |
| 预计成本 | q=50 单条均 247 s，约 9 分钟 |

## 三、与 confirmation 的关系（合并口径）

**confirmation 的已报告结果不因 E.1 修改。** q=50 的 7/10 是在 N=10 预登记下完成的，
它作为 confirmation 的结果原样保留。

E.1 是一次**新的、独立预登记的测量**，其 20 条运行与 confirmation 的 10 条使用同一冻结协议、
同一运行器、互不相交的 seed，因此可以合并为 **n=30** 报告一个 p 的估计。合并结果以
「q=50 单次 discovery rate，n=30」的名义单独报告，不冒充 confirmation 的修订值，
也不把两者的区间混用。

**本轮不允许再次追加。** 若 n=30 仍不足以支持某个判断，那是下一次预登记的事，
不得在看到 E.1 结果后继续加 seed —— 那会把固定样本量的测量变成序贯停止。

## 四、报告内容

1. E.1 单独的 discovery / conditional certification / end-to-end 三个率及 Wilson 区间（n=20）；
2. 与 confirmation 合并后的 q=50 discovery rate 及 Wilson 区间（n=30）；
3. 基于合并后验的 restart 预测分布 $E_p[1-(1-p)^k]$，k=1…6，**以及按表规模反推的每格 k**；
4. 无候选 run 的直接判据（最小 dev dReach/ΔW、其位置、浓度与数值有效性计数），
   与 0915、confirmation 的失败形态对照；
5. **B 段 gate 与 candidate discovery 的 2×2 表**（在 n=30 上重算）。confirmation 上
   Fisher 双侧 p = 0.016，但一格只有 2 个样本；E.1 用于检验该关联是否在更大样本上成立。
   只有成立才考虑把「B 未过门即重启」写进 restart 触发规则。

## 五、执行规则

沿用 `CONFIRMATION_T2_20260922.md` 第四节：不依据中途结果更换配置、增删 seed 或提前结束；
失败的 run 不重跑、不补抽 seed 替换；认证未通过不恢复 PPO、不另挑 checkpoint、不放宽阈值；
仅对已确认的基础设施故障允许同 seed 从起点完整重跑并保留异常记录。

## 六、事前声明的分辨率

n=30 的 95% Wilson 区间（以 70% 点估计为例）为 [52.1, 83.3]，宽 31.2 个百分点
（n=10 时为 49.5）。**n=30 仍然分不开 70% 与 80%**；E.1 的作用是把 k 的选择从猜测变成计算，
并给出 q=50 是否明显低于 gate 的方向性证据，不是对 gate 做显著性判定。
