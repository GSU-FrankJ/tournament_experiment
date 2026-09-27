# A-stage audit：全部 40 条 formal T=2 run

**A 的 verifier gate 一次也没有触发——161 次 call，eligible = 0，40/40 条 budget exhausted。
而且这不是「预算差一点」：A 的两个条件在 400 个 update 内朝相反方向走，联合通过区间是空的。
A-end 质量在「后来找到候选」和「从未找到候选」两组之间测不出差异。**

据此：A 应当写成 fixed-budget terminal-stage pretraining（计划里的分支①）；
A400 vs A600/A800 的 pilot 不被这份证据支持（分支②）。

---

## 零、对象与方法

40 条正式 held-out run（`F1_A400_const_first_eligible`，manifest
`experiments/two_stage_formal_T2_20260915/manifest.json`）：q=50、q=60 各 seeds 10001–10020。
29 条找到 first-eligible 候选，11 条预算耗尽无候选（q50 十条、q60 一条）。

全部数字来自已经落盘的文件，**没有训练、没有调用 verifier、不消耗 RNG**：每条 run 的
`train_history.json` 里每次 verifier call 都存了 `e_hat_dev`，也就是 verifier 当时自己查询的
stage-2 均值努力网格（development tier，state_step=4，q50 为 101 点、q60 为 111 点）。
stage-2 形状误差就在这条网格上对闭式 g₂(d) 直接算，网格由 `dp_br_verifier.stage_grid`
重建并 assert 长度一致。40 条 run 的 protocol 常数逐条比对完全一致（脚本内 assert）。

口径（与 runner 一致，取自各 run 冻结的 `config.json`）：

| | A | B | C |
| --- | --- | --- | --- |
| 判据 | full-grid `max Δ₂/ΔW` | `exp_root/ΔW` | `dReach/ΔW` |
| 判据阈值 | 0.02 | 0.02 | 0.01 |
| 浓度阈值 | 0.04 | 0.04 | 0.04 |
| 连续要求 | k_phase=3 | k_phase=3 | k_stop=1 |
| cap / warmup / timeout | 400 / 100 / 100 | 600 / 100 / 100 | 1000 / 100 / 100 |

`inside support` 一律指 **|d| < 2q 严格**，|d| ≥ 2q 记为 tail（闭式 g₂ 在 tail 为 0），
与 recovery 表和 H1 对照表同一口径。

---

## 一、A 的 gate 从未触发，而且是结构性的

161 次 A call **全部 numerical valid**。按 A-local update 分组：

| A local | n | 判据过 | 浓度过 | eligible | 中位 max Δ₂/ΔW | 中位浓度 | 中位 stage-2 max\|err\| |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 40 | 0 | 2 | **0** | 0.08153 | 0.04226 | 38.08 |
| 200 | 40 | 0 | 16 | **0** | 0.08349 | 0.04146 | 38.66 |
| 240 | 1 | 0 | 1 | **0** | 0.14355 | 0.03585 | 52.06 |
| 300 | 39 | 3 | 10 | **0** | 0.05538 | 0.04292 | 31.96 |
| 340 | 1 | 0 | 1 | **0** | 0.14934 | 0.03378 | 53.17 |
| 400 | 40 | 9 | 4 | **0** | 0.03442 | 0.04374 | 25.04 |
| 合计 | 161 | 12 | 34 | **0** | | | |

判据只在 local 300/400 过（12 次），浓度主要在 local 100/200 过（34 次里 18 次），
**两者在同一次 call 上同时成立的次数 = 0**。图 (b) 是这 161 个点在
(判据, 浓度) 平面上的位置：它们落在一条向下倾斜的权衡带上，左下角那个 eligible 方框是空的。

结构上也不可能触发：A 只有 4 次 call（local 100/200/300/400），而 k_phase=3 要求连续三次，
所以策略最晚必须在 **local 200** 就 eligible；而 local 200 的判据中位数是 0.0835，
是阈值的 4 倍。40 条 run 的 `exit_reason` 全部是 `budget_exhausted`，
`consecutive_eligible_at_exit` 全部是 0。

**「再多给点预算」解决不了**：判据确实还在快速下降——最后 100 个 update 的中位变化是
q50 候选组 −37.2%、q50 无候选组 −26.5%、q60 候选组 −31.8%——但浓度是平的甚至上升
（同口径 +1.1% / −0.1% / +4.2%），A 末的浓度中位数 0.0437 仍高于 0.04，40 条里只有 4 条
在 A 末浓度达标、9 条在 A 末判据达标，交集仍是 0。要让 A 的门能响，得让浓度在 A 内穿过
0.04，而它并不朝那个方向走。

唯一一条浓度在 A 段就掉到 0.04 以下的 run 是 `tel_q50_s10012`（0.0361/0.0359/0.0338/0.0341），
代价是它的判据整段卡在 0.106→0.143 从未下降——这正是权衡带的极端点。
它也是 A 段唯一被 stability 触发额外 call 的 run（5 次 call，local 100/200/240/340/400，
没有 local 300；末次 call 的 trigger 是 `phase_end`）。它在 C 段是分层②「擦边」那一条。

---

## 二、A-end 质量不区分成功与失败

只有 **q=50 是 10 vs 10 的可比较组**（q60 是 19 vs 1，只能描述）。
Mann–Whitney U 双侧，Holm 在每组 6 个指标内校正，rank-biserial 为效应量
（正号 = 候选组的值更小）：

| A-end 指标 | 候选 中位 (n=10) | 无候选 中位 (n=10) | p | Holm | rank-biserial |
| --- | ---: | ---: | ---: | ---: | ---: |
| max Δ₂/ΔW | 0.03370 | 0.04391 | 0.850 | 1.000 | +0.06 |
| 浓度 max Std[e]/100 | 0.04471 | 0.04412 | 0.308 | 1.000 | −0.28 |
| stage-2 max\|ê₂−g₂\|，\|d\|<2q | 24.83 | 28.18 | 0.623 | 1.000 | +0.14 |
| stage-2 RMSE，\|d\|<2q | 10.69 | 11.74 | 0.308 | 1.000 | +0.28 |
| d=0 有符号误差 | −24.83 | −28.18 | 0.791 | 1.000 | −0.08 |
| tail 平均努力，\|d\|≥2q | 7.84 | 7.46 | 0.623 | 1.000 | −0.14 |

六项全部 p ≥ 0.31、|rank-biserial| ≤ 0.28，中位数区间大幅重叠。**A-end 质量不预测最终成败。**

**功效限制必须一起读**：n=10/10 的 MWU 对 1.2 SD 的位移只有约 67% 功效，0.8 SD 只有 35%
（正态模拟，4000 次重复）。所以这排除的是「大差异」，不是「中等差异」。

唯一方向一致的信号在**候选组内部**：A-end 越差，首个候选来得越晚。
q60（n=19）六个指标的 Spearman ρ 为 +0.48…+0.57（`err_at_d0` 与浓度取负号同向），
raw p 0.011–0.039，Holm 校正后 0.068–0.157，**无一显著**；q50（n=10）同号但更弱
（ρ +0.31…+0.61，raw p 0.064–0.39）。这六个指标彼此高度重复（A 末 `err_at_d0` 数值上
就等于 `max|err|`，因为最大误差正好在 d=0），应当当成**一个**信号看。
即便它是真的，它关系到**速度**，不关系到**成败**。

---

## 三、A→B：A 末的差异基本被洗掉

配对看同一条 run（图 c）：stage-2 inside-support 的 max|err| 从 A 末中位 25.8 降到 B 末 15.1，
40 条里 37 条下降，两组的线完全交织。A-end 与 B-end 同名指标的 Spearman
ρ 只有 0.18–0.54，Holm 校正后无一显著（最小 Holm p = 0.082）。

B 末两组仍然没有显著差异（同样 q50 10 vs 10，Holm 后全部 ≥ 0.32）。最接近的两项是：
判据 `exp_root/ΔW` 0.00364 vs 0.00551（raw p 0.141，方向符合预期）和
tail 平均努力 13.67 vs 11.66（raw p 0.054，**方向相反**——候选组的 tail 反而更差）。
两项都不足以据此下结论。

配套的 level 参考（中位数）：

| | A 末 | B 末 | 闭式 |
| --- | ---: | ---: | ---: |
| q50 ê₂(0) | 44.2 | 61.3 | 70.0 |
| q60 ê₂(0) | 34.4 | 52.6 | 58.33 |
| q50 ê₁（A 段不训练 stage 1） | — | 46.69 | g₁ = 46.67 |
| q60 ê₁ | — | 39.50 | g₁ = 38.89 |
| max\|err\| 的位置 d | 0（即峰值欠投） | ±40~48 | — |

A 末的最大误差就在 d=0，也就是峰还没建起来（q50 只到闭式峰值的 63%）；
到 B 末峰基本建好（欠投 8.5），最大误差搬到了 |d|≈40–48。

---

## 四、顺带查到的两件事

这两件不属于 A，但出自同一次提取，且都直接关系到计划的第 2 项，所以一并记下。

### 1. B 的门是纯浓度门，全样本 239 次 call

| | 次数 | 占比 |
| --- | ---: | ---: |
| B call 总数 | 239 | |
| 判据 `exp_root/ΔW ≤ 0.02` 过 | **236** | 98.7% |
| 浓度 ≤ 0.04 过 | 86 | 36.0% |
| eligible | 86 | 36.0% |

eligible 次数 = 浓度通过次数，说明**浓度是唯一的约束**。每条 run 的 eligible 次数分布是
1 次（9 条）、2 次（16 条）、3 次（15 条），而且 **`max_consecutive_eligible` 与
`n_eligible` 对 40 条全部相等**——eligible 永远是 B 的最后 k 次 call，
证实了原来在 11 条子集上看到的单调 `c…cE(E)` 模式在全样本成立。

所以 B 的 gate 不是失灵，是**浓度穿过 0.04 的时点太晚**：cadence 100 下要凑满 3 连续，
浓度必须在 B-local 400 之前穿过，25 条没做到（差 1 次的 16 条、差 2 次的 9 条）。
这比原先 11 条子集的 63/66 强得多，也把「B 改 cadence 25」的受益面量化了：
25 条 run 差的是 1–2 次连续，cadence 25 下三连续只需跨 50 个 update。

**但要注意**：改 B 的 cadence 会改变 B 的退出点（runner 在 `eligible >= k_phase` 处
直接跳出该 phase），从而改变下游全部轨迹。它和 C-only 的加密不同，**不是逐位重放**，
必须重跑才能报率。

### 2. B 阶段把 tail 练坏了，40/40 条

tail 平均努力（|d| ≥ 2q，闭式为 0）：A 末 7.79（q50）/ 8.04（q60）→ B 末 13.37 / 13.57，
**40 条 run 全部上升**，逐条差值中位 +5.5，范围 [+0.94, +13.73]。
同期 inside-support 的 max|err| 37/40 条在下降。图 (d) 是 q50 的中位曲线：
B 把峰建起来的同时，把两侧尾巴抬了起来。

机制假说（**未验证**）：B 只用 root starts，深 off-path 的 stage-2 状态在 B 段不被访问，
于是漂移；而 C 段 110 次 call 上 stage-2 贡献了 dReach 的 90%、argmax 的中位在 d = −56，
正是这一段。能区分该假说与替代解释的实验：把 B 的采样改成 root + ES 混合（像 C 一样），
看 tail 抬升与 C 段深 off-path 的 Δ₂ 是否同时下降。**没有做，不能当成已确认的原因。**

---

## 五、对计划第 3 项的直接回答

1. **分支① 成立。** A 的 verifier gate 在 161 次 call 上从未触发，40/40 budget-forced，
   成功与失败 run 一视同仁。把 A 写成 fixed-budget terminal-stage pretraining 是对的，
   同时应当把 A 的 gate 明确降级为诊断输出，而不是留一个永远不响的门在协议里。
2. **分支② 不成立。** 无候选 run 的 A-end policy 并不明显比成功 run 差（六项全部不显著，
   效应量 ≤0.28）。A400 vs A600/A800 的 pilot 不被这份证据支持。
3. **但不能反过来说「A400 已经够了」。** A 在 400 处仍在以每 100 个 update −30% 左右的
   速度改善，A400 是一次截断而非收敛点；这份 audit 说的是「A 末的差异与最终成败无关
   （在 n=10/10 的功效范围内）」，不是「A 的长度无关紧要」。
4. 如果之后要动，**更对症的杠杆是 B 的采样设计（§4.2）而不是 A 的预算**——但那是一个
   需要重跑 40 条才能验证的协议改动，且与 B/C cadence 的改动必须合并成一次 wave。

---

## 六、文件

| 文件 | 行数 | 内容 |
| --- | ---: | --- |
| `A_stage_calls.csv` | 400 | **主表**。40 run × (A 161 + B 239) 次 call，逐次给出 update/local/trigger、判据与阈值、br_pass、浓度全套、eligible 与连续计数、exp_root/ΔW 与 dReach/ΔW，以及该次 call 的 stage-2 形状误差（MAE/RMSE/max\|err\| 及其位置、d=0 误差、tail 均值与最大值、full-D₂ RMSE） |
| `A_end_quality.csv` | 40 | 每条 run 一行：身份与 outcome（search success、final pass、首个候选的 C update、无候选 run 的 dense 最小 dReach）、A/B 两段的退出方式与 call 计数、A 末与 B 末的全部指标、local 100–600 的判据/浓度/误差轨迹、最后两次 call 的相对变化、B 末 stage-1 努力与 g₁ 对照 |
| `group_comparison.csv` | 24 | q × phase × 指标的两组统计（n、均值、样本 SD、中位、range）与 MWU U/p/Holm/rank-biserial |
| `rank_correlations.csv` | 24 | Spearman：A-end 指标 vs 首个候选 C update（候选组内）、A-end vs B-end 同名指标；含 Holm |
| `phase_end_curves.npz` | — | 40 条 run 的 A 末与 B 末 stage-2 均值努力曲线，外加两个 q 的 d 网格与闭式 g₂ |
| `a_stage_audit.json` | — | 上述内容的嵌套版本，`_meta.a_gate` 记录 gate 共现结构 |
| `a_stage_audit.png` / `.svg` | — | 四面板：(a) A 判据轨迹 (b) 161 次 call 在判据–浓度平面上的权衡带与空的 eligible 区 (c) A 末→B 末配对 (d) q50 的 A 末/B 末曲线对闭式 g₂ |

生成脚本：`extract_a_stage_audit.py`（表格与 JSON）、`plot_a_stage.py`（图）。重跑：

```bash
cd . && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/two_stage_formal_T2_20260915/A_stage_audit/extract_a_stage_audit.py && OMP_NUM_THREADS=1 .venv/bin/python -B experiments/two_stage_formal_T2_20260915/A_stage_audit/plot_a_stage.py
```

---

## 七、限制（请一并转达）

1. **A 段无法做 dense 重建。** stability log 不含 stage-1 的 `e_hat`，而两阶段 verifier 需要它
   （`no_candidate_C_history/README.md` §四.3 已记）。所以 A 的分辨率就是每条 4 次 call，
   本报告关于「A 内部轨迹」的陈述只在这 4 个点上成立。
2. **失败组 n=11，其中 10 条是 q50。** 只有 q50 构成 10 vs 10 的可比较组；q60 是 19 vs 1，
   全部按描述统计处理，没有做检验。跨 q 合并会被 q 本身的水平差异混淆，因此没有合并。
3. **功效。** 见 §二。n=10/10 排除的是大差异。
4. **六个质量指标高度相关**，不是六个独立检验；Holm 在此偏保守，但方向一致性也因此不能
   当成六份独立证据。
5. **全部跨 run 比较是观察性的。** seed 不是随机分配到 outcome，A-end 质量与最终成败之间
   即使有关联也不能读成因果。
6. **这是对已完成的 cadence-100 正式批次的事后分析。** 协议一旦改动（C cadence 25、
   B cadence 25、B 采样），A 以外的下游都会变，本报告的 B/C 部分需要在新批次上重算。
