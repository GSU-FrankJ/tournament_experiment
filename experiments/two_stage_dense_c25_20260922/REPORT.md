# Dense-C 重跑：把 phase-C verifier 频率从每 100 update 改成每 25 update

**6 条 run 全部完成（returncode 0，总 wall 319 s）。4 条找到候选并通过完整最终认证，2 条仍是 budget exhaustion。**

对象是 no-candidate 诊断里分层①的 5 条（q50 s10008/s10011/s10015/s10020、q60 s10017）
加上分层②的 q50 s10012。协议与正式 T=2 的 `F1_A400_const_first_eligible` 完全一致，只改两处：

* `protocol.verifier_timeout`：`100` → `{"A": 100, "B": 100, "C": 25}`（只加密 C）；
* `protocol.weights_every`：新增 `25`，每 25 个 global update 导出 actor+critic 权重。

phase-C warmup 仍是 100，k_stop 仍是 1，A/B 的 cadence 没动。

## 轨迹复现是逐位精确的

`verify` 与 `concentration_stats` 都是策略网络的纯函数，不消耗 RNG；权重导出是只读的。
所以加密 cadence 不改变训练轨迹，只改变 verifier 看的位置。实测：

* 全部 18 组（6 run × A/B/C）共享 verifier call 上，`exp_root`/`dreach`/`dfull`/`v_br_root`/
  `v_mean_root`/`delta_max_all`/`pdl_sum` 的最大绝对差 = **0.000e+00**；
* 每一次共享 call 的 `e_hat` 网格与 `max_std_norm` 都**完全相同**（e.g. 4/4、6/6、10/10）。

种子只由 `(seed, q, rng_namespaces)` 决定，group 名与实验名不进入，所以复现是结构性的，不是巧合。

## 结果

| run | 停在 | dev dReach/ΔW | conc | final dReach/ΔW | refine diff | C_all | overall | joint | 原 C-call 最小 | 原 final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :--: | :--: | ---: | ---: |
| q50 s10008 | **C450** | 0.009056 | 0.03314 | 0.009031 | 2.5e-05 | 0.03314 | ✅ | ✅ | 0.010030 | 0.011316 |
| q50 s10011 | **C150** | 0.008462 | 0.03696 | 0.008558 | 9.5e-05 | 0.03696 | ✅ | ✅ | 0.010471 | 0.045181 |
| q50 s10015 | **C725** | 0.009382 | 0.03260 | 0.009408 | 2.6e-05 | 0.03260 | ✅ | ✅ | 0.010906 | 0.021985 |
| q60 s10017 | **C375** | 0.008886 | 0.03338 | 0.008827 | 5.9e-05 | 0.03338 | ✅ | ✅ | 0.010589 | 0.011784 |
| q50 s10012 | 无（C1000 耗尽） | — | — | 0.028860 | 3.2e-05 | 0.03025 | ❌ | ❌ | 0.011518 | 0.028860 |
| q50 s10020 | 无（C1000 耗尽） | — | — | 0.018475 | 1.0e-05 | 0.03163 | ❌ | ❌ | 0.015235 | 0.018475 |

4 条转化的 run 都是 `k_stop_passes` 且 `development_stop_strictly_before_cap=True`，
最终认证的 main / refine / sensitivity / dense concentration 四项全过，
即按正式协议的 `final_joint_pass` 标准成立——不是只过了 development 那一关。

按正式报告的口径重算：q=50 的 search success 从 10/20 升到 **13/20**，q=60 从 19/20 升到 **20/20**。

## 为什么还有 2 条没转化：eligible window 比 25 个 update 还窄

25-update 的 call 网格与 20-update 的 stability 网格错位，合并后 C 段每 100 个 update 有
{20, 25, 40, 50, 60, 75, 80, 100} 共 8 个观测点。在这个合并网格上，
每个 eligible 点两侧最近的 ineligible 邻居：

| run | 前一个 ineligible | eligible 点 | 后一个 ineligible | 窗宽上界 |
| --- | --- | --- | --- | ---: |
| q50 s10015 | C125 = 0.01384 | **C140 = 0.00965** | C150 = 0.01301 | ≤ 25 |
| q50 s10020 | C950 = 0.01571 | **C960 = 0.00975** | C975 = 0.01246 | ≤ 25 |
| q60 s10017 | C225 = 0.01093 | **C240 = 0.00987** | C250 = 0.01036 | ≤ 25 |

s10015 在 C140 的窗口被 25-grid 整个跨过去了（C125 和 C150 都高出阈值 30–40%），
但它后来在 C725 又开出一个窗口并停住；s10020 的 C960 窗口同样被 C950/C975 跨过，
之后再没出现，所以耗尽预算。s10012 在 37 次 call 上的最小值 0.010757（高出 7.6%），
合并网格上一个 eligible 点都没有——它不是采样问题。

这条是对「是否存在 short-lived eligible policy」的直接回答：**存在，而且至少有三个窗口窄于 25 个 update。**
25 的 cadence 把命中率从 0/5 提到 4/5，但没有消除漏采。

## 文件

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 6 条 run 的完整记录，含 `changes_vs_source`、`unchanged` 与 6 个协议模块的 sha256 |
| `runs/F1_A400_const_first_eligible/<run>/` | 每条 run 的 `config.json`、`train_history.json`、`final_eval.json`、`arrays.npz`、`status.json`、`checkpoint.pt`、`phase_{A,B}_exit_arrays.npz` |
| `runs/<run>/weights/u#####.npz` | **每 25 个 global update 的 actor+critic 权重**（46–80 个/run，共 388 个） |
| `runs/<run>/train_history.json` → `weight_checkpoints` | 权重索引：`{update, phase, local, file}` |
| `C_merged_grid.csv`（278 行） | 合并的 20+25 网格：每点的 dReach/ΔW、concentration、valid/br_pass/conc_pass/eligible、来源 |
| `dense_run_summary.csv`（6 行） | 每条 run 的停止点、最终认证、窗宽上界、与原 run 的对照 |
| `dense_c25_traces.png` / `.svg` | 6 面板图：灰线为合并网格，空心圆为 25-update call，灰方块为原 100-update call，红星为停止点 |
| `logs/`、`launch_status.json`、`launch.log` | 启动记录 |
| `launch.py`、`analyze_dense.py`、`plot_dense.py` | 生成脚本 |

运行器是 `run/run_final_dp_br_round3_dense.py`（worktree `candidate-search-recovery-data-92d49a`），
是 round-2 运行器的逐字拷贝加上面两处改动；协议模块（`agents/ppo_curriculum.py`、
`envs/curriculum_env.py`、`utils/dp_br_verifier.py`、`utils/theory_multistage.py`、
`run/run_final_dp_br.py`）从 worktree `tournament-dp-br-verify-4df573` 原样复制，sha256 记在 manifest 里。

重跑：

```bash
cd . && .venv/bin/python -B experiments/two_stage_dense_c25_20260922/launch.py
```

## 需要决定的事

这 6 条是**按诊断结果挑出来的子集**，不是随机样本。要把「cadence 25」写成协议、并用它的通过率替换
正式表里的 10/20 与 19/20，必须对同一批 20 个 held-out seed 的全部 40 条 run 重跑一遍，
否则就是在已知会转化的子集上报告通过率。按 319 s / 6 条估算，40 条约 35–40 分钟（10 并发）。
