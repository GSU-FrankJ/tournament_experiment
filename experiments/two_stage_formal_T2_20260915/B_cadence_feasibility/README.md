# Phase-B cadence 25：能不能塞进三次连续 eligible

**能，38/40 条。当前 cadence 100 下 B 只有 15/40 条靠 verifier 推进，而且 15 条全是 q=60；
q=50 是 0/20。改成 cadence 25 后，q50 变成 18/20、q60 变成 20/20，
当前 25 条 budget-exhausted 里有 23 条会翻成 verifier_passed。**

**但这只预测 B 的退出点。** B 提前退出会改变 C 的起点以及之后的一切，
候选发现率不能从这里读出来——那需要重跑。

## 方法：B 段可以精确离线重放

B 的 stability log 每 20 个 update 存了 verifier 会查询的 stage-1 与 stage-2 `e_hat` 网格
以及 `max_std_norm`，而 `verify` / `concentration_stats` 是策略网络的纯函数、不消耗 RNG，
所以 B 和 C 一样可以离线重放。**在 40 条 run 的 239 个共享 update（B-local 100…600）上，
重放的 `exp_root/ΔW` 与浓度对在线记录的最大绝对差 = 0.000e+00。**

判据口径取自各 run 冻结的 `config.json`：B 的判据是 `exp_root/ΔW ≤ 0.02`，
浓度 `≤ 0.04`，外加数值有效；`k_phase = 3` 连续；B cap 600、warmup 100。

cadence-25 的调用日程是 B-local `100 + 25j`。判定规则：存在 L 使得 `[L, L+50]`
区间内**全部** 20-update 网格点都 eligible，且 `L + 50 ≤ cap`。
（三次调用本身有两个不落在 20-update 网格上，格点之间的行为没有观测，见限制。）

## 结果

| | 实际 cadence 100 | 预测 cadence 25 |
| --- | ---: | ---: |
| B 靠 verifier 推进 | **15 / 40**（全部 q60） | **38 / 40** |
| q=50 | 0 / 20 | 18 / 20 |
| q=60 | 15 / 20 | 20 / 20 |
| B 退出的 local update | 600（25 条打满） | 中位 475，范围 [350, 600] |

瓶颈是浓度穿过 0.04 的时点，而 q 之间差别很大：**首次 eligible 的 B-local 中位数
q50 是 500、q60 是 380**（全体中位 440，范围 240–600）。cadence 100 下三次调用只能落在
B-local 400/500/600，也就是要求浓度在 **B400 之前**穿过——q50 的中位数是 B500，
所以 q50 在当前日程下**结构上不可能**靠 verifier 推进，0/20 不是偶然。
cadence 25 只要求窗口宽 50 个 update，40 条里有 38 条满足。

翻转的 23 条里，预测 B 退出中位在 B-local 525，相对现在打满 600 平均省 75 个 update。

两条例外，都在 q50：

| run | 首次 eligible | 情况 |
| --- | ---: | --- |
| tel_q50_s10002 | B540 | 之后又掉出 eligible（26 个点里只有 3 个 eligible，非单调） |
| tel_q50_s10007 | B600 | 只在 cap 那一个点 eligible |

40 条里 0 条在 B 段完全没有 eligible 点；2 条（上表两条）出现 eligible 之后又掉出去。

## 这个改动的方向性后果是两面的，没有验证

B 提前 75–225 个 update 退出，意味着 C 从一个**训练更少**的策略开始，C 预算仍是 1000。
两个方向的机制都存在：

* B 的额外 update 会继续压低 `exp_root` 与浓度 → 早退可能让 C 的起点更差；
* 但 A-stage audit 显示 B 段把 stage-2 的 tail 练坏了（|d| ≥ 2q 的平均努力
  7.8 → 13.4，**40/40 条全部上升**），而 C 段 dReach 的 90% 来自 stage-2、argmax 中位在
  d = −56 —— 早退可能反而少破坏一点。

哪一个占优不能从这份重放读出来。要知道答案只能重跑，而且因为 B 的退出点变了、
轨迹从那一点起就发散，这**不是逐位重放**，必须 40 条全跑。

## 文件

| 文件 | 行数 | 内容 |
| --- | ---: | --- |
| `B_replay_20u.csv` | 1195 | 40 条 run 的 B 段 20-update 重放：每点的 `exp_root/ΔW`、浓度、valid、判据/浓度/eligible 三个 flag、是否与在线 call 重合 |
| `B_cadence_summary.csv` | 40 | 每条 run 一行：实际退出方式与 update、在线 call 计数、首次 eligible 的 B-local、是否非单调、cadence-25 的首个可行窗口与预测退出点、相对实际退出省下的 update |
| `b_cadence.py` | — | 重放与判定，含 239 个共享点的恒等性断言 |

重跑：

```bash
cd . && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/two_stage_formal_T2_20260915/B_cadence_feasibility/b_cadence.py
```

## 限制

1. **只预测 B 的退出点。** 下游（C 的起点、候选发现、最终认证）全部不能从这里推断。
2. **25-cadence 的三次调用有两次不在 20-update 网格上。** 判定用的是「`[L, L+50]` 内所有
   20-update 网格点都 eligible」，格点之间没有观测；真实日程可能在某个未观测点上不达标。
3. **已经 verifier_passed 的 15 条轨迹是截断的**（B 提前退出，之后没有 stability 记录），
   它们的「更早可行窗口」只在已观测到的点上评估。
4. 浓度穿越时点按 20-update 分辨率定位，因此首次可行调用位置有至多一个 25-update 槽的不确定性。
