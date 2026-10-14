# MS-R4 handoff: message to the coworker

Nothing here has been sent. The PI decides whether, when and how the message goes out; no recipient and no contact detail is written in this file. The links point to branch `ms-r4` of the repository `https://github.com/GSU-FrankJ/tournament_experiment` and follow the branch head.

---

## 中文

您好，

按您的方案，我们在新的开发种子 10511–10530 上（q = 50、60，每个 q 20 个，组间配对，共 280 次运行，全部正常结束）完成了两项尝试。锁定的 v2.0 作为基准没有改动；30501–30520 和 40501–40520 没有使用。结果如下。

**尝试一：降低 stage-2 平局点偏差（`lrsm_st_s16`：LeakyReLU 隐藏层 + 不依赖硬 clamp 的平滑有界均值映射；对照 `t1_st_s16`、`relu_st_s16`）——未达目标，按约定停止这条线。**

- q = 60：三项全部满足。平均 |peak 误差| 0.0259（≤ 0.04）；相对 `t1_st_s16` 的配对差 −0.0122，95% 区间 [−0.0219, −0.0016]；20 次运行无一门槛失败。
- q = 50：平均 |peak 误差| 0.0329（≤ 0.04），但配对差 −0.0025，95% 区间 [−0.0175, +0.0196] 含 0；20 次中有 4 次未通过门槛（G-A 和 G-F 各 4 次，G-N 和 G-S 各 1 次）。对照 `t1_st_s16` 为 0/20，`relu_st_s16` 为 3/20。4 次失败中有 3 次的 peak 误差很小（0.011–0.025），是 η₂/DW 和 RMSE 超过门槛。
- 只报告组合改动的效果，不归因于 LeakyReLU 或均值映射中的某一项。`lrsm` 相对 `relu_st_s16` 没有可检测的优势（|peak 误差| 配对差 +0.0008 [−0.0212, +0.0263]、+0.0086 [−0.0000, +0.0174]）。

**尝试二：development DP-BR 的采样反馈是否帮助 PPO 训练——两个反馈组都未达目标；DP-BR 只保留为验证器，不声称它改善了训练。**

- 固定 t1 actor、预算（stage-2 1600 次、stage-1 600 次），关闭动态停止和 polishing，三组：①固定分层采样（α=0）；②残差引导（α=0.5）；③残差 + DP-BR 偏离收益（α=0.5）。②③在各层内部调整采样，层间比例与①完全相同。
- 相对①，fixed-checkpoint 的 stage-2 η₂/DW 学习曲线面积减少量（目标 ≥ 20%）：②为 +2.0% [−1.4%, +5.1%]（q=50）和 −2.0% [−4.8%, +0.8%]（q=60）；③为 +1.3% [−2.4%, +4.9%] 和 −0.9% [−3.8%, +2.3%]。所有 95% 区间上限都低于 6%。
- ②在 q=50 有一个运行（种子 10517）从通过变为 G-A 未通过（η₂/DW = 0.00516，门槛 0.005）；其余门槛不变差。
- 现有的跨层采样（描述性，第四组）在 q=50 面积小 5.4%，q=60 面积反而大 12.6%；在 200–1600 次更新窗口内两个 q 都更小（31%、15%）。它同时把约 0.12 的采样质量从近平局层移到中间层，本轮无法区分是反馈还是层比例变化的作用。

**因此**：两项各自都没有达到目标，没有可锁定的完整方案；您的方案只在两项各自有效时谈到合用需先在开发种子上验证，所以这一结果下方案没有安排 fresh-seed 确认或合用实验。v2.0 保留为 T=2 求解器，两项结果可作为消融和局限报告。是否这样处理，请您和 PI 决定。

**需要说明的两点：**

1. E1 的“平均 |peak 误差| ≤ 0.04”这一项对照 `t1_st_s16` 本身也满足（0.0354、0.0381），真正起作用的是配对改善和门槛失败。
2. E2 中“stage-1 偏离收益不变差”的预注册操作定义用的是 Gmax_full/DW，而在全部 280 次运行中它与 stage-2 的 η₂/DW 完全相等，所以实际测的是 stage-2 偏离收益；stage-1 的偏离收益本身（已除以 DW）在每次运行中都不超过 5.4×10⁻⁴，约为 0.001 容限的一半。这不改变任何结论。

每组的配置和代码版本、逐种子结果、配对区间、peak、学习曲线、最终各项门槛及运行成本都在下面的报告里：

- 总览：https://github.com/GSU-FrankJ/tournament_experiment/blob/ms-r4/reports/ms/r4/summary.md
- 决策输入（目标对照、方案的后续含义、确认规则）：https://github.com/GSU-FrankJ/tournament_experiment/blob/ms-r4/reports/ms/r4/05_decision_inputs.md
- 尝试一：https://github.com/GSU-FrankJ/tournament_experiment/blob/ms-r4/reports/ms/r4/04_e1_actor.md
- 尝试二：https://github.com/GSU-FrankJ/tournament_experiment/blob/ms-r4/reports/ms/r4/04_e2_feedback.md

---

## English

Hello,

Following your plan, we ran the two tries on the new development seeds 10511–10530 (q = 50 and 60, 20 seeds per q, paired across arms, 280 runs, all finished normally). The locked v2.0 stayed as the baseline unchanged; seeds 30501–30520 and 40501–40520 were not used. The results:

**Try 1: lower the stage-2 tie-point bias (`lrsm_st_s16`: LeakyReLU hidden units and a smooth, bounded mean map without the hard clamp; controls `t1_st_s16` and `relu_st_s16`) — target not met, the line stops as agreed.**

- q = 60: all three parts hold. Mean |peak error| 0.0259 (<= 0.04); paired difference to `t1_st_s16` -0.0122, 95 % interval [-0.0219, -0.0016]; none of the 20 runs fails a gate.
- q = 50: mean |peak error| 0.0329 (<= 0.04), but the paired difference is -0.0025 with the 95 % interval [-0.0175, +0.0196] containing 0, and 4 of 20 runs fail a gate (G-A and G-F four times each, G-N and G-S once each). The controls: `t1_st_s16` 0 of 20, `relu_st_s16` 3 of 20. Three of the four failures have a small peak error (0.011–0.025); they fail on eta_2/DW and RMSE.
- Only the effect of the combined change is reported; it is not attributed to the LeakyReLU or to the mean map. `lrsm` has no detectable advantage over `relu_st_s16` (paired |peak error| difference +0.0008 [-0.0212, +0.0263] and +0.0086 [-0.0000, +0.0174]).

**Try 2: does sampling feedback from the development DP-BR help PPO training — neither feedback arm meets the target; the DP-BR stays a verifier and no training improvement is claimed.**

- The t1 actor, budgets (stage 2: 1600 updates, stage 1: 600), no dynamic stop and no polishing are fixed. Three arms: (1) fixed stratified sampling (alpha = 0); (2) residual-guided (alpha = 0.5); (3) residual plus DP-BR deviation gain (alpha = 0.5). Arms (2) and (3) adjust the sampling inside each stratum; the strata shares are exactly those of (1).
- Reduction of the area under the stage-2 eta_2/DW curve at the fixed checkpoints against (1) (target >= 20 %): arm (2) +2.0 % [-1.4 %, +5.1 %] at q = 50 and -2.0 % [-4.8 %, +0.8 %] at q = 60; arm (3) +1.3 % [-2.4 %, +4.9 %] and -0.9 % [-3.8 %, +2.3 %]. Every 95 % interval's upper end is below 6 %.
- Under arm (2) one run at q = 50 (seed 10517) went from passing to failing G-A (eta_2/DW = 0.00516 against the limit 0.005); the other gates do not get worse.
- The existing cross-stratum sampler (descriptive, a fourth arm) has a 5.4 % smaller area at q = 50 and a 12.6 % larger one at q = 60; over updates 200–1600 its area is smaller at both q (31 %, 15 %). It also moves about 0.12 of the sampling mass from the near-tie to the middle stratum, and this round cannot separate the effect of the feedback from that of the change of shares.

**So:** neither try met its own target and there is no complete scheme to lock; your plan speaks of validating a combination on development seeds only for the case that each try is effective, so it schedules no fresh-seed confirmation or combined run for this outcome. v2.0 stays the T = 2 solver, and the two results can be reported as ablations and limitations. Whether to proceed that way is for you and the PI to decide.

**Two things to know:**

1. Part (a) of the first try (mean |peak error| <= 0.04) is met by the control `t1_st_s16` alone (0.0354, 0.0381); the parts that discriminate are the paired improvement and the gate failures.
2. The pre-registered definition of "stage-1 deviation gains do not become worse" used Gmax_full/DW, which equals the stage-2 eta_2/DW in all 280 runs, so it measured the stage-2 deviation gain; the stage-1 deviation gain itself (already divided by DW) is at most 5.4e-4 in every run, about half of the 0.001 margin. No conclusion changes.

The configuration and code version of every arm, the per-seed results, the paired intervals, the peak, the learning curves, the final gates and the run cost are in the reports:

- Overview: https://github.com/GSU-FrankJ/tournament_experiment/blob/ms-r4/reports/ms/r4/summary.md
- Decision inputs (targets, what the plan implies, the confirmation rule): https://github.com/GSU-FrankJ/tournament_experiment/blob/ms-r4/reports/ms/r4/05_decision_inputs.md
- Try 1: https://github.com/GSU-FrankJ/tournament_experiment/blob/ms-r4/reports/ms/r4/04_e1_actor.md
- Try 2: https://github.com/GSU-FrankJ/tournament_experiment/blob/ms-r4/reports/ms/r4/04_e2_feedback.md
