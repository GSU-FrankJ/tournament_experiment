# T=3 implementation pilot 报告（q60 → q50，6 个 debug seeds）

日期 2026-09-24，主机 vector2。计划：`MultiStage/three_stage/T3_IMPLEMENTATION_PLAN_20260924.md` Task 1–9。
本文件只报告 implementation/debug 结果。6 个 debug seeds 不进入任何 formal 成功率；formal seeds 未启动。

## 0. 结论摘要

- 实现与数值链条全部实际跑通，6/6 runs `done`，launcher exit code 全为 0，`make_report --check-completeness` 对 smoke 与 pilot 全部 `ok`（`reports/pilot/completeness.json`）。
  实际跑通的环节：3-stage collect、PPO update、全 T=3 DP-BR dev/final verifier、checkpoint 重读、mean/Beta 两种 self-play economics、全部表格与图。
- 在预先规定预算 A400/B600/C1800 内，**0/6 runs 找到 candidate**（q60 0/3、q50 0/3；每个 q 的 95% Wilson 上界 0.5615）。
  所有 run 的 B 都是 `budget_forced`，C 都是 `no_candidate_budget_exhausted`。conditional certification 为 N/A（0 个 candidate）。
- 没有发现实现 bug，没有 invalid verifier 调用。各项交叉检查都成立：
  - 重读 endpoint 与内存 actor 逐位一致；
  - endpoint 上重算的 dev 与原调用差 0；
  - min_dev replay 差 0；
  - mean self-play 的 MC 均值与 final DP V1_mean(0) 的差为 −1.26…+0.55 个 MCSE。
- 失败形态（相关性描述，非因果结论）：阶段 A 结束时 stage-3 最大 continuation gain 为 0.106–0.210·DW，位于内部状态 d≈−16…−24，不在 tails。
  q60 与 q50 s10411 的 B strategic 值逐步降到 0.019–0.032，但同时 stage-2/3 concentration 升到 0.04 附近并反复越过，没有出现 3 连 eligible。
  q50 s10412/s10413 的 B strategic 值在整个 B 中停在 0.10–0.19，由 stage-3 内部偏离主导。
  C 中最小 dev dReach/DW：q60 为 0.0224/0.0248/0.0243，q50 为 0.0261/0.0614/0.0584，从未 ≤0.01。
- 覆盖：直接起点的每 bin 实测计数与期望一致（无零 bin，CV≈0.01–0.02，tails 最少 2189/bin）。最大偏离所在 bin 的训练暴露量是数千（direct）到数万（continuation）次。
  现有数据不支持“覆盖不足”这一解释，因此 FORMAL_SETTINGS 保留已测试默认值（见 `FORMAL_SETTINGS.md`）。

## 1. 执行记录与版本

| 项目 | 值 |
|---|---|
| 实验目录 E | `experiments/three_stage_implementation_pilot_20260924` |
| 与计划的位置差异 | 计划写 `ROOT/experiments/...`；本会话运行在 git worktree 内，工具禁止写入主 checkout，故放在 worktree 内同一相对路径。代码全部按自身位置解析路径，可整体复制到计划位置 |
| 代码版本 | worktree HEAD d1b8443（分支 `claude/tournament-tasks-1-9-87db41`），E 为未跟踪新目录；W HEAD d1b8443，所用 W 源文件在 W 中本就未跟踪，每个 run 的 `config.json.provenance` 记录其 size/mtime 与 git status |
| W / 旧 T3 / T2 | 未修改（`git -C W status` 与开始时相同；全程 `python -B`，W 中无新文件） |
| Python | `python` 3.12.3，numpy 2.5.0，torch 2.5.1+cu121，CPU，torch_threads=1，OMP/MKL/OPENBLAS=1 |
| T2 precision job | 启动前检查：无运行中（`two_stage_q50_precision_20260924/launch_status.json` state=done） |

测试与 smoke 命令、exit code、耗时见 `logs/task8_smoke_commands.log`（每行标 `host=vector2`）：

| 步骤 | 结果 |
|---|---|
| `unittest discover`（50 cases） | OK，exit 0，≈6 s |
| smoke q60 s10400 | done，exit 0，6.9 s；6 updates / 192 episodes / 330 joint steps / 660 actions（C 每 update 69） |
| smoke q50 s10410 | done，exit 0，7.1 s；同上计数 |
| `make_report --cohort smoke --check-completeness` | exit 0，两 run complete |
| pilot q60（launch 16:56:50–17:14:03） | 3/3 exit 0，每 run wall 1027–1033 s |
| `make_report --cohort pilot --q 60 --check-completeness` | exit 0，3/3 complete；检查后再启动 q50 |
| pilot q50（launch 17:16:07–17:33:44） | 3/3 exit 0，每 run wall 1051–1055 s |
| `make_report --cohort pilot --check-completeness` | exit 0，6/6 complete |

运行期间的两处代码变更，均未影响 pilot 数据：

- q60 运行中修改了 `build_manifests.py` 的 formal 分支，使 formal settings 的所有非 game 键都会生效。重新生成后 pilot/smoke manifests 判定为 unchanged，runner 不读取该文件。
- 两组 pilot 完成后，把 `metrics.wilson` 在 k=0 时的下界从浮点残差 5.6e−17 改为精确 0，只影响报表。

单元测试覆盖范围（`tests/test_t3.py`）：

- 数值表（两 q）、512/32、1109/69、seed 排除、q 唯一来源、输出路径唯一、formal 分支；
- 局部 collector 与 W collector 的张量和三条 RNG 状态逐位相同，start→stage 全计数；
- stub agent 下的对手观测 −d、角色噪声、terminal bootstrap=0、ES2 延续到 t3 且含两个 rewards；
- 边界与越界 binning；
- 状态机脚本序列：A 不早停；B T,T,F,T,T,T 在 150 退出且 consecutive 为 1,2,0,1,2,3；invalid 归零；C 在 50 停；无 pass 到 1800；全 invalid 无 minimum；最早 B75/C25；
- verifier 阶段顺序、continuation≠δ2、常数策略 root payoff、独立标量 oracle（atol 1e−8）、terminal 解析 CDF、BR reach 前向区间并集、pmf 质量、DomainError 与容差内 roundoff、invalid 原因入日志；
- 两 q 实际 dev/final 网格点数，dense 26403/24003；
- 六项认证分项；从文件重读 checkpoint；economics 的 RNG 隔离与可重复性、常数/非对称/交换角色测试、Beta 矩；
- Wilson、聚合分母、截断 JSONL、minimum 选择；
- 真实 tiny run 端到端，以及注入 update 失败时不计数。

## 2. 逐项回答（PILOT_REPORT 10 问）

数据来源：`reports/pilot/*.csv`（per-run 副本在 `runs/<run_id>/tables/`），图在 `reports/pilot/figures/`。

### Q1 D2/D3、bins、端点、root 与 ES composition 是否实际正确

是。

- 每个 run 的 `config.json.resolved.derived` 与计划表一致：
  - q60：B=220，D2=[−220,220]，D3=[−440,440]，bins 44/88，dev 点 1/111/221，final 点 1/221/441，dense 26403；
  - q50：B=200，D2=[−200,200]，D3=[−400,400]，bins 40/80，dev 点 1/101/201，final 点 1/201/401，dense 24003。
  `completeness` 逐 run 核对了 final_eval 的 grid_points 与 dense 点数。
- 起点组成来自 `history.jsonl`，并由 completeness 逐 update 核对：A 每 update 512×ES3，B 512×ES2，C 256 root + 85 ES2 + 171 ES3。
  - 每 update joint steps 为 512/1024/1109，C 分阶段 256/341/512；
  - 全程 1,433,600 episodes、2,815,400 joint environment steps、5,630,800 physical actions，6 个 run 均无早停；
  - root 起点 460,800 = 256×1800 个，全部在 d=0（stage-1 严格 binning 容差 1e−9）；
  - ES 原始起点（角色翻转前）在所有 bin 都非零，其中 q60 A 每 bin 2200–2440，q50 A 2419–2704；
  - 端点 bin（第 0 与最后一个）也有计数；tail 最少每 bin 2219（q60 A）/ 2404（q50 A）。
- 没有越界 gap：严格 binning 在越界时会报错，而 6 个 run 都没有报错。

### Q2 A 每 bin 真实 exposure、tail 覆盖；问题出在 sampler、continuation 还是预算

- A 直接 ES3（learner-signed）实测每 bin 计数见 `coverage_summary.csv`（`phase=A, kind=visits, s3→t3`），期望值 q60 2327.27、q50 2560：

  | run | min | p05 | median | max | CV | zero bins | tail+ / tail− 总数 | tail bin 最小 |
  |---|---:|---:|---:|---:|---:|---:|---|---:|
  | q60 s10401 | 2247 | 2268 | 2326 | 2433 | 0.017 | 0 | 21031 / 20877 | 2247 |
  | q60 s10402 | 2189 | 2247 | 2328 | 2439 | 0.019 | 0 | 20822 / 21127 | 2219 |
  | q60 s10403 | 2207 | 2248 | 2327 | 2451 | 0.021 | 0 | 21151 / 20897 | 2271 |
  | q50 s10411 | 2433 | 2493 | 2558 | 2691 | 0.020 | 0 | 20556 / 20399 | 2494 |
  | q50 s10412 | 2436 | 2485 | 2563 | 2683 | 0.020 | 0 | 20557 / 20617 | 2485 |
  | q50 s10413 | 2404 | 2486 | 2554 | 2711 | 0.021 | 0 | 20195 / 20426 | 2404 |

- continuation 覆盖（新增的 start2→t3 口径）：
  - B 的 ES2→t3 有 19（q60）/18（q50）个零 bin，D3 tails（|mid| ≥ 0.8×half）计数为 0，即 ES2 延续从不到达 D3 tails；
  - C 的 root→t3 有 38–40 个零 bin，中位数每 bin 12–55，root 轨迹集中在内部；
  - D3 tails 只由 A 与 C 的直接 ES3 覆盖，C 中每个 tail bin 最少 3386（q60）/ 3639（q50）。
- 误差位置：A 结束时 stage-3 最大 continuation gain 为 q60 0.106/0.158/0.141、q50 0.162/0.210/0.180 ·DW，argmax 在 d=−16…−24。
  这是内部 bin，A 中每 bin 暴露约 2300–2600 次，与 tails 相同量级。
- 判断：直接起点 sampler 实际给出了期望的均匀计数，没有缺失或稀疏 bin，最大偏离也不在稀疏区域。所以数据不指向 sampler 问题。
  continuation 分布只决定 tails 是否被 root/ES2 轨迹访问，而最大偏离不在 tails。
  现有数据与“固定预算内内部高梯度区域的优化质量不足”相容，但 6 个 debug run 不能区分“预算”与“其他优化因素”（例如 snapshot 滞后、采样噪声下的梯度信噪比），未做区分实验。
  与 T2 相比每 bin 期望暴露减半，这本身不构成不足的证据；本 pilot 也没有观察到可归因于它的缺口。

### Q3 B 首次 eligible、最长连续 pass、第三次 pass、600 内是否退出、失败类型计数

来源 `phases.csv`（phase=B，24 次检查/run）：

| run | 首次 eligible | 最长连续 | 第三次连 pass | 600 内退出 | strategic-only | conc-only | both | invalid |
|---|---|---:|---|---|---:|---:|---:|---:|
| q60 s10401 | 无 | 0 | 无 | 否（budget_forced） | 19 | 0 | 5 | 0 |
| q60 s10402 | 无 | 0 | 无 | 否 | 10 | 0 | 14 | 0 |
| q60 s10403 | 无 | 0 | 无 | 否 | 10 | 1 | 13 | 0 |
| q50 s10411 | 无 | 0 | 无 | 否 | 4 | 0 | 20 | 0 |
| q50 s10412 | 无 | 0 | 无 | 否 | 24 | 0 | 0 | 0 |
| q50 s10413 | 无 | 0 | 无 | 否 | 24 | 0 | 0 | 0 |

144 次 B 检查中 strategic 只通过 1 次（q60 s10403 local 500，值 0.01916），但该次 concentration=0.0408 失败。

### Q4 B 动态 gain 与 concentration 曲线；C dReach 与 stage 贡献曲线

图：`figures/verifier_curves_q60.png`、`figures/verifier_curves_q50.png`；逐次数值见 `verifier_calls.csv`。

B 的 max_D2 (V2_BR−V2_mean)/DW（阈值 0.02）与 concentration（阈值 0.04）：

| run | 首值 | 最小 | 最后 | ≤0.02 次数 | 阈值穿越 | conc 首/最后 | conc>0.04 次数 |
|---|---:|---:|---:|---:|---:|---|---:|
| q60 s10401 | 0.127 | 0.0242 | 0.0242 | 0 | 0 | 0.029 / 0.0407 | 5 |
| q60 s10402 | 0.069 | 0.0213 | 0.0269 | 0 | 0 | 0.039 / 0.0397 | 14 |
| q60 s10403 | 0.109 | 0.0192 | 0.0318 | 1 | 2 | 0.034 / 0.0406 | 14 |
| q50 s10411 | 0.137 | 0.0282 | 0.0300 | 0 | 0 | 0.037 / 0.0414 | 20 |
| q50 s10412 | 0.189 | 0.1312 | 0.1430 | 0 | 0 | 0.029 / 0.0269 | 0 |
| q50 s10413 | 0.156 | 0.1016 | 0.1016 | 0 | 0 | 0.035 / 0.0308 | 0 |

B 值的构成，取自同一调用的 per-stage 值：q60 s10401 在 u425 时 V2 continuation 0.127，其中一步 δ2 最大 0.053、stage-3 gain 0.142；到 u1000 分别为 0.024 / 0.017 / 0.026。
q50 s10412 在 u1000 时分别为 0.143 / 0.066 / 0.155（stage-3 argmax d=−20）。
可见 B 的 V2 动态 gain 主要由 stage-3 内部偏离贡献，不能用 δ2 代替。

C 的 dReach/DW（72 次检查，阈值 0.01）与 stage 贡献，贡献取最后 12 次的均值：

| run | 首值 | 最小（update） | 最后 | ≤0.01 | ≤0.02 | ≤0.03 | t1 | t2 | t3 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| q60 s10401 | 0.108 | 0.0224 (2800) | 0.0224 | 0 | 0 | 9 | 0.0017 | 0.0114 | 0.0151 |
| q60 s10402 | 0.154 | 0.0248 (2475) | 0.0308 | 0 | 0 | 17 | 0.0066 | 0.0166 | 0.0073 |
| q60 s10403 | 0.102 | 0.0243 (2450) | 0.0303 | 0 | 0 | 13 | 0.0007 | 0.0160 | 0.0132 |
| q50 s10411 | 0.164 | 0.0261 (2625) | 0.0459 | 0 | 0 | 6 | — | — | — |
| q50 s10412 | 0.221 | 0.0614 (2125) | 0.0688 | 0 | 0 | 0 | — | — | — |
| q50 s10413 | 0.151 | 0.0584 (2050) | 0.0757 | 0 | 0 | 0 | — | — | — |

q50 的最后一次 stage 贡献为 t2/t3：s10411 0.0296/0.0162，s10412 0.0452/0.0233，s10413 0.0568/0.0164。stage-1 贡献在所有 run 最后都 ≤0.0075。
C 失败类型：q60 为 strategic-only 63/72/65，both 9/0/7；q50 为 strategic-only 62/72/72，both 10/0/0。conc-only 为 0，invalid 为 0。

### Q5 600 结束仍未通过的分类（不自动归为预算不足）

- q60 s10401：strategic 值单调下降，最后一次 0.0242 仍 >0.02；最后 5 次 concentration >0.04。属于“strategic 仍在阈值上方 + concentration 在末段成为瓶颈”，观察窗口内没有 3 连过。
- q60 s10402：从 local 300 起 strategic 值在 0.021–0.031 平台，concentration 在 0.040 上下反复（14/24 失败）。两个约束同时卡住。
- q60 s10403：唯一一次 strategic 过线时 concentration 恰好失败；之后 strategic 回升到 0.032。两者同时卡住。
- q50 s10411：strategic 值从 0.137 降到 0.028，local 125 起 concentration 持续 >0.04（0.0406–0.0430），concentration 是持续瓶颈。
- q50 s10412：strategic 值全程 0.131–0.189，没有下降趋势；concentration 正常。persistent strategic gap，来源是 stage-3 内部偏离。
- q50 s10413：strategic 值 0.156→0.102，缓慢下降；concentration 正常。persistent strategic gap。
- 6 个 run 都没有数值 invalid。上述观察不支持外推“更长 B 一定会成功”：q60 与 q50 s10411 的趋势在下降，但 concentration 同时上升；q50 s10412 没有下降趋势。

### Q6 C 结局、最小 dReach 及同 checkpoint 状态、B 退出类型与 C 结局

- 6/6 为 B `budget_forced` 且 C `no_candidate_budget_exhausted`，因此无法比较不同 B 退出类型下的 C 结局。
- 最小 valid C 调用来自 `failure_diagnostics.csv` 与 `checkpoints/min_dev_record.json`，同一调用的权重、arrays、coverage snapshot 一致，replay 差 0：

| run | min dReach/DW | update（C local） | conc | 贡献 t1/t2/t3 | argmax d（t2,t3） | 最大 reach 状态 | 该 bin 训练暴露（C direct / C cont / A+B direct / A+B cont） |
|---|---:|---|---:|---|---|---|---|
| q60 s10401 | 0.02239 | 2800 (1800) | 0.0284 | 0.0008/0.0088/0.0129 | −56, −8 | t3 d=−8, 0.0129 | 3455 / 25207 / 2348 / 6623 |
| q60 s10402 | 0.02476 | 2475 (1475) | 0.0276 | 0.0007/0.0179/0.0061 | 76, 124 | t2 d=76, 0.0179 | 2850 / 11632 / 6986 / 0 |
| q60 s10403 | 0.02426 | 2450 (1450) | 0.0321 | 0.0006/0.0112/0.0124 | 52, 128 | t3 d=128, 0.0124 | 2715 / 7956 / 2340 / 7102 |
| q50 s10411 | 0.02615 | 2625 (1625) | 0.0315 | 0.0005/0.0148/0.0108 | −36, 104 | t2 d=−36, 0.0148 | 3496 / 27293 / 7556 / 0 |
| q50 s10412 | 0.06141 | 2125 (1125) | 0.0291 | 0.0001/0.0257/0.0356 | 28, 100 | t3 d=100, 0.0356 | 2375 / 6293 / 2586 / 7616 |
| q50 s10413 | 0.05842 | 2050 (1050) | 0.0279 | 0.0034/0.0293/0.0258 | 36, 100 | t2 d=36, 0.0293 | 2328 / 17336 / 7707 / 0 |

- 最大 Delta 状态（max_all）与最大 reach 状态只在 q60 s10403 不同：max_all 在 t2 d=220（D2 边界），0.0140，不在 R_2 内。
- 最大偏离所在 bin 的暴露量都在数千到数万；这里只报告二者并列，不据此下因果结论。

### Q7 dev-final 数值精度、Delta_max_all 与 Reach、DP mean vs MC mean

Diagnostic terminal（无 candidate，certification=`not_applicable_no_candidate`，`final_joint_pass=false`）的 final 数值检查：

| run | dReach dev/final | Δ（/DW） | EXP dev/final | Δ | Delta_max_all final（Δ） | dfull final | dense conc | main | numeric |
|---|---|---:|---|---:|---|---:|---:|---|---|
| q60 s10401 | 0.02239/0.02236 | 3.1e−5 | 0.00820/0.00818 | 2.1e−5 | 0.0129 (0) | 0.0247 | 0.0284 | 否 | 否 |
| q60 s10402 | 0.03080/0.03091 | 1.1e−4 | 0.01568/0.01572 | 4.8e−5 | 0.0172 (5.2e−5) | 0.0309 | 0.0262 | 否 | 否 |
| q60 s10403 | 0.03028/0.03032 | 4.3e−5 | 0.01072/0.01066 | 6.2e−5 | 0.0184 (5.4e−5) | 0.0303 | 0.0299 | 否 | 否 |
| q50 s10411 | 0.04594/0.04592 | 2.2e−5 | 0.01475/0.01476 | 9.3e−6 | 0.0296 (2.7e−6) | 0.0459 | 0.0300 | 否 | 否 |
| q50 s10412 | 0.06877/0.06877 | 4.9e−6 | 0.02375/0.02372 | 3.1e−5 | 0.0452 (4.3e−5) | 0.0688 | 0.0265 | 否 | 否 |
| q50 s10413 | 0.07566/0.07644 | 7.7e−4 | 0.02502/0.02499 | 3.1e−5 | 0.0576 (7.8e−4) | 0.0764 | 0.0254 | 否 | 否 |

- 两层 verifier 均 valid。refine 检查（≤0.002）全部通过，dense concentration（≤0.04，全 D1–D3、step 0.05）全部通过。
  numeric_thresholds_pass 全部为否，唯一原因是 main（dReach_final/DW ≤0.01）不满足。
- Delta_max_all 与 Reach：final 层 q60 s10401 的 stage-2 full 最大 0.0111 在 d=220，reach 最大 0.0088 在 d=−56，是 R_t 外偏离大于 R_t 内的例子；其余 run 的 final 层二者位置相同。
  这不是全域 MPE 认证。
- DP vs MC：mean self-play U 均值与 final V1_mean(0) 的差是 MC 噪声加离散化，只作描述：

  | run | MC U（MCSE） | DP V1_mean(0) | 差/MCSE |
  |---|---|---|---:|
  | q60 s10401 | 3.07451 (0.00037) | 3.07452 | −0.02 |
  | q60 s10402 | 3.23174 (0.00048) | 3.23180 | −0.12 |
  | q60 s10403 | 3.22817 (0.00042) | 3.22794 | +0.55 |
  | q50 s10411 | 2.83997 (0.00057) | 2.83988 | +0.16 |
  | q50 s10412 | 2.89058 (0.00052) | 2.89046 | +0.22 |
  | q50 s10413 | 2.75593 (0.00062) | 2.75671 | −1.26 |

  stochastic−mean U 为 −0.0042 至 −0.0012（描述值，不是 exploitability）。

### Q8 资源、日志完整性与 formal 估算

来源 `resources.csv`、`runs.csv`，3 个 worker 并发：

| run | 总 wall | 训练 wall / CPU | update 内计算合计 | dev verifier（100 次） | final 两层 | economics（2×3×200000） | min-dev profiles | 峰值 RSS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| q60 ×3 | 1027–1033 s | 1018–1025 / 630–639 s | 619–627 s | 5.7–5.9 s | 0.33–0.41 s | 3.4–4.0 s | 0.1 s | 513–516 MiB |
| q50 ×3 | 1051–1055 s | 1043–1046 / 645–649 s | 634–638 s | 5.4–5.5 s | 0.30–0.34 s | 3.3–3.9 s | 0.1 s | 513–518 MiB |

- 训练 wall 比 CPU 多约 390 s/run。原因是每个 update 两次 fsync（history 追加 ≈50 ms，status 原子写 ≈34 ms），/home 是 md0 ext4。
  这是可测到的纯 I/O 开销，不影响数值。本 pilot 两组使用同一实现，未改动。
- 磁盘：每 run 32–35 MB（其中 tables 17 MB）；cohort 报表 98 MB。
- 日志完整性：6 个 run 的 history 行数均为 2800 = 最后完成 update；每 run 100 次计划调用全部存在，无计划外调用；无截断行；JSON 严格合法；coverage 与 self-play 直方图质量守恒；identity 与 min 同调用检查全部通过。
- formal 估算见 `FORMAL_SETTINGS.md`：N=10/q 为 20 runs，约 5.8 run·h（wall）/ 3.6 CPU·h，10 workers 约 35–60 min；N=20/q 为 40 runs，约 11.6 run·h / 7.2 CPU·h，10 workers 约 70–110 min（10 个并发写入时的 fsync 争用未测，取区间）。
  内存峰值约 10×0.52 GB。

### Q9 policies、stage effort、visitation、leader/follower（均为 diagnostic 政策）

6 个 endpoint 都是 `diagnostic_terminal`（C 1800 末权重），不是认证候选；以下描述均为 diagnostic 政策。

- e1(0)：q60 37.81/27.13/33.15，q50 42.30/38.80/43.38。
- e2(0)、e3(0)：`runs.csv` 列 `e2_at_0`、`e3_at_0`；完整曲线在 `policy_profiles.csv` 与 `figures/policy_curves_q*.png`。
- stage-3 策略是以 d≈0 为峰的近对称曲线，d=0 处 43.6–54.4。BR 在略落后状态显著更高，例如 q50 s10412 在 d=−20 时 e_hat=43.0、a_BR=69.0；q60 s10401 在 d=−20 时 41.1 对 56.1。
  静态 stage-3 基准 2ke=DW·f(0) 给出 q50 70、q60 58.3（仅作量级参照，不作正确性断言）。见 `deviations.csv`。
- 阶段期望努力按 root mean self-play occupancy 计算，pooled 600000 episodes，括号内为 MCSE。stochastic 模式值相近，见 `stage_metrics.csv`：

  | run | E[e1] | E[e2] | E[e3] |
  |---|---:|---:|---:|
  | q60 s10401 | 37.81 | 27.18 (0.015) | 27.57 (0.015) |
  | q60 s10402 | 27.13 | 23.62 (0.017) | 29.24 (0.021) |
  | q60 s10403 | 33.15 | 23.64 (0.014) | 25.57 (0.019) |
  | q50 s10411 | 42.30 | 27.58 (0.019) | 31.28 (0.021) |
  | q50 s10412 | 38.80 | 30.00 (0.018) | 32.18 (0.017) |
  | q50 s10413 | 43.38 | 27.75 (0.021) | 33.59 (0.021) |

- leader−follower 配对差（mean 模式，stage 2 / stage 3）：
  - q60：+13.03/−0.33、+16.73/+4.17、+12.10/+2.66；
  - q50：+11.00/+3.50、−6.56/+9.10、−2.96/+6.96。
  root 为 tie（n=0，均值 null）。不同 seed 在 stage 2 上符号不同，说明这些 diagnostic 政策在结构上不一致。
  曲线差见 `policy_asymmetry.csv` 与 `figures/asymmetry_q*.png`。
- visitation：training（按 phase/start/stage）、mean/stochastic self-play（player0/player1/representative）、BR chain（verifier 节点）分口径存放于 `state_visitation.csv`，图为 `figures/state_histogram_q*.png`。

### Q10 seeds 排除关系

- 本次 debug seeds 为 smoke 10400（q60）、10410（q50），pilot 10401–10403（q60）、10411–10413（q50）。全部排除在 formal 之外，也不并入任何 formal 成功率。
- 已暴露的历史 T3 seeds：旧 T3 debug 的 smoke 10300 与 runs 10301–10303。
- 启动前重新扫描了 experiments/、MultiStage/ 以及各 worktree 的 results/experiments 中的 seed 字段与 run 目录名，未发现 10400–10413 或 formal 预留 11001–11020、11101–11120 被使用过。
- formal 预留 seeds 仍全部未使用；formal manifests 尚未生成。

## 3. 未证实与限制

- 6 个 debug run 只说明在该预算与定义下的实现行为，不给出任何可靠性估计；pilot 比例只是 debug outcome。
- 未做区分“预算不足”与“其他优化因素”的实验。数据只显示：最大偏离位于高暴露的内部状态；q60 的 C dReach 在 1800 内缓慢下降但未达 0.01；q50 两个 seed 在最后约 1000 updates 内没有下降趋势。
- concentration 阈值 0.04 与 Beta c_min=100 的交互：mean≈0.5 时 std_norm≤0.04 需要 c≳150。B 中峰值努力上升时 concentration 同步上升，这是观察到的相关。
- 所有 endpoint 都不是 candidate，经济表格没有 certified/uncertified candidate 组，只有 diagnostic_terminal 组（n=6，每 q 3 个）。
