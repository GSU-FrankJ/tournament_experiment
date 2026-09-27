# T=2 held-out confirmation 结果

**20 条全部完成（returncode 0，整批 wall 472 s）。q=50 端到端 7/10 = 70%，q=60 端到端 10/10 = 100%。
17 个候选全部通过最终联合认证，条件通过率 17/17 = 100%。**

协议为冻结的 `MultiStage/two_stage/protocol/FINAL_T2_PROTOCOL_20260922.json`（A400 fixed-budget /
B600 cadence 25 / C1000 cadence 25），预登记为 `experiments/two_stage_confirmation_T2_20260922/PREREGISTRATION.md`，
两份文件都在任何 run 启动前冻结。seeds 为全新的 q50 10101–10110、q60 10111–10120，
两个列表不相交。没有重跑、替换 seed、更换候选或改动阈值；20 条 run 的 `config.record`
与 manifest 逐条精确一致，`issues` 为空。

## 一、主要结果

分母固定 N=10，方括号为 95% Wilson 区间。两个 q 分别报告；两个 q 是不同的博弈，
不合成 N=20 做推断。

| 指标 | q=50 | q=60 |
| --- | --- | --- |
| **Candidate discovery rate** | **7/10 = 70.0% [39.7, 89.2]** | **10/10 = 100.0% [72.2, 100.0]** |
| **Final certification rate**（条件，`final_overall_pass`） | **7/7 = 100.0% [64.6, 100.0]** | **10/10 = 100.0% [72.2, 100.0]** |
| Final joint rate（条件，再加 dense C_all ≤ 0.04） | 7/7 = 100.0% [64.6, 100.0] | 10/10 = 100.0% [72.2, 100.0] |
| **End-to-end success rate**（joint / 10） | **7/10 = 70.0% [39.7, 89.2]** | **10/10 = 100.0% [72.2, 100.0]** |
| Budget exhaustion，无候选 | 3/10 = 30.0% [10.8, 60.3] | 0/10 = 0.0% [0.0, 27.8] |
| Operational status | 完成 10，中断 0，技术失败 0 | 完成 10，中断 0，技术失败 0 |

两个 q 合计的条件通过率为 **17/17 = 100.0% [81.6, 100.0]**（描述性合计，非独立推断）。
没有候选在最终认证阶段失败，因此 `final_overall_pass` 与 `final_joint_pass` 计数相同。
没有候选恰好出现在 C1000 上限处（`cap_candidates` 为空）。

## 二、候选的认证数值

| | q=50 (n=7) | q=60 (n=10) |
| --- | ---: | ---: |
| final dReach/ΔW | 0.008580 ± 0.000636，范围 [0.007736, 0.009425] | 0.008741 ± 0.001069，范围 [0.006724, **0.009923**] |
| final EXP_root/ΔW | 0.002385 ± 0.000497 | 0.002191 ± 0.000632 |
| dense C_all | 0.034883 ± 0.002523，最大 0.037949 | 0.037024 ± 0.000784，最大 0.038412 |
| 首个候选的 C-local update | 中位 250，范围 [100, 925] | 中位 137.5，范围 [100, 375] |

**Numerical refinement**（同一冻结 θ 的 development 与 final 两档之差，阈值 0.002）：

| | q=50 (n=7) | q=60 (n=10) |
| --- | ---: | ---: |
| \|dReach_final − dReach_dev\|/ΔW | 4.48e-05 ± 3.70e-05，最大 1.10e-04 | 2.68e-04 ± 4.11e-04，最大 **1.15e-03** |
| \|EXP_final − EXP_dev\|/ΔW | 2.78e-05 ± 2.40e-05，最大 6.96e-05 | 9.41e-06 ± 7.53e-06，最大 2.43e-05 |

17 个候选全部远低于 0.002；最接近的是 q60 s10114 的 1.15e-03（阈值的 57%）。
另外 q60 s10111 的 final dReach/ΔW = 0.009923，是阈值 0.01 的 99.2%——通过，但余量很小。

## 三、Analytical recovery

17 个候选全部进入统计，且**恰好等于 `final_joint_pass` 子集**（没有认证失败的候选），
因此两个子集的统计完全相同，不重复列出。均值 ± 样本 SD（ddof=1）。

| 指标 | q=50 (n=7) | q=60 (n=10) |
| --- | ---: | ---: |
| 第一期相对误差 % | 2.961 ± 2.190 | 4.751 ± 3.130 |
| 正努力域 RMSE | 4.473 ± 0.543 | 5.063 ± 0.472 |
| 完整 D₂ RMSE | 6.386 ± 1.038 | 7.432 ± 0.689 |
| On-path RMSE（GL64） | 4.395 ± 0.695 | 4.796 ± 0.537 |
| d=0 相对误差 % | 12.697 ± 3.787 | 14.309 ± 5.613 |
| tail 平均努力（\|d\| ≥ 2q，闭式为 0） | 7.714 ± 1.589 | 9.395 ± 1.017 |
| evenness 最大差 | 5.190 ± 2.948 | 7.660 ± 3.023 |
| 跨期绝对差 \|ê₁ − E[ê₂]\| | 2.415 ± 1.553 | 2.709 ± 2.406 |
| On-path E 的绝对误差 | 1.811 ± 1.259 | 2.579 ± 1.241 |

与偏离认证一致的老结论仍然成立：**17 个候选在 d=0 处全部低于解析峰值**（欠投 12.7% / 14.3%），
而 tail 处闭式为 0、学到的策略摊了 7.7 / 9.4 的平均努力。**偏离认证通过不等于逐点恢复解析均衡**；
本轮没有 analytical-recovery 的通过门槛，非零误差不记为失败。

on-path 主表用分半 GL64，另按 0.5/0.25/0.125/0.0625 四档步长做求积敏感性
（`recovery_metrics.json` 的 `onpath_sensitivity`），未据此更换候选。

## 四、协议在新 seed 上的行为

| | 本轮（cadence 25，20 条） | 0915（cadence 100，40 条） |
| --- | --- | --- |
| A 段退出 | **20/20 budget_exhausted** | 40/40 budget_exhausted |
| B 段退出 | **18/20 verifier_passed**（q50 8/10、q60 10/10） | 15/40 verifier_passed（q50 **0/20**、q60 15/20） |
| B 段 local updates | 中位 512.5，范围 [325, 600] | 25 条打满 600 |
| 每 run development calls | q50 均 43.9、q60 均 20.9 | q50/q60 均约 15 |

A 的 gate 按 audit 的预期继续一次都没触发，A 就是固定预算的预训练。
**B 的 gate 恢复了作用**：离线重放预测 38/40 能凑满三次连续，实测 18/20，方向与量级都对上。

### 一个观察：B 的门与候选发现同时失败

两条 B 段仍然 budget_exhausted 的 run（q50 s10105、s10107）**同时也是**没有找到候选的 run；
18 条 B 过门的 run 里 17 条找到候选。Fisher 精确检验双侧 p = 0.016。

这是**观察性**的，而且一格只有 2 个样本：B 的门没过与 C 找不到候选很可能是同一个底层原因
（策略收敛不够）的两个表现，不能读成「B 早退导致 C 成功」。真要分辨需要干预实验。

### 三条 q=50 失败 run 的直接判据

| run | C 检查数 | valid | 浓度过 | 判据过 | 最小 dev dReach/ΔW | 高出阈值 | 位置 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q50 s10104 | 37 | 37 | 37 | 0 | 0.015958 | +59.6% | C750 |
| q50 s10105 | 37 | 37 | 37 | 0 | 0.010290 | +2.9% | C100 |
| q50 s10107 | 37 | 37 | 37 | 0 | 0.010760 | +7.6% | C100 |

三条全部是 **BR 判据从未达标**，浓度和数值有效性 37/37 全过——与 0915 那批 11 条失败的形态一致。
s10105 与 s10107 的最小值出现在 C 的第一次调用（warmup 的 C100）且之后变差，
s10104 则是全程高出阈值 60%。

## 五、成本

| | q=50 | q=60 |
| --- | ---: | ---: |
| 总 updates | 均 1547.5（中位 1637.5，范围 1050–2000） | 均 972.5（中位 950，范围 825–1150） |
| A / B / C updates | 400 / 545 / 602.5 | 400 / 415 / 157.5 |
| total episodes | 均 792,320 | 均 497,920 |
| 训练耗时 | 均 244.8 s | 均 147.2 s |
| 离线最终验证 | 均 2.03 s | 均 1.98 s |

整批 20 条在 10 并发下 472 s。成功 run 与耗尽 run 的耗时差别很大
（q50 的 C 段从 100 到 1000 个 update 都有），不能只对成功 run 取平均代表整个方案。

## 六、对 T=3 practical gate 的读法

内部 gate 是 q50 ≥ 80%、q60 ≥ 90% 的发现率，以及 P(final pass | eligible candidate) ≳ 90%。

* **q=60：10/10 = 100% [72.2, 100]**，点估计过线。
* **q=50：7/10 = 70% [39.7, 89.2]**，点估计**低于** 80%，但区间覆盖 80%。
* **P(final pass | candidate) = 17/17 = 100% [81.6, 100]**，点估计与区间下界都在 90% 附近或以上。

**预登记里已经事先声明过：N=10 分不开 80% 与 60%**，所以本轮不能对 q50 做「是否达标」的
显著性判定。它给出的是点估计 70%，以及三条失败 run 的形态——其中一条（s10104）高出阈值 60%，
属于训练侧的真失败，不是 verifier 采样问题。若要把 q50 推过 80%，需要的是针对这类失败的
训练侧改动，不是更密的验证。

## 七、文件

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 20 条 run 的完整输入清单，含 `changes_vs_20260915`、协议模块 sha256、seed 列表与不配对声明 |
| `build_manifest.py` | 从 0915 记录派生本轮清单；含「只改身份字段与协议块」「协议相对 0915 只差 verifier_timeout」「seed 全仓库未使用」三项 assert |
| `runs/FINAL_A400_B25_C25/<run>/` | 每条 run 的 `config.json`、`train_history.json`、`final_eval.json`、`arrays.npz`、`status.json`、`checkpoint.pt`、`phase_{A,B}_exit_arrays.npz`、`weights/u#####.npz` |
| `formal_results.csv` / `.json` | 20 条分类、数值指标与成本；`summary_by_q` 含全部率与 Wilson 区间 |
| `C_checks.csv` | 全部 244 次 C 检查 |
| `never_eligible_runs.csv` | 3 条无候选诊断 |
| `recovery_metrics.csv` / `.json`、`recovery_summary.md` | 17 个候选的解析恢复，含四档求积敏感性与逐 checkpoint 重算校验 |
| `figures/recovery_curves.*`、`figures/recovery_errors.*` | 恢复曲线与误差图 |
| `summarize_confirmation.py`、`recovery_analysis.py` | 从 0915 脚本逐字复制，只改路径常数与硬编码 run 数（各 9 / 10 行）；指标定义与独立重算 assert 未动 |
| `launch.py`、`launch_status.json`、`launch.log`、`logs/` | 启动记录（tmux session `conf_t2`） |

重跑：

```bash
cd . && .venv/bin/python -B experiments/two_stage_confirmation_T2_20260922/launch.py && OMP_NUM_THREADS=1 .venv/bin/python -B experiments/two_stage_confirmation_T2_20260922/summarize_confirmation.py && OMP_NUM_THREADS=1 .venv/bin/python -B experiments/two_stage_confirmation_T2_20260922/recovery_analysis.py
```
