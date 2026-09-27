# Dense-C 补完：剩余 5 条 no-candidate run，以及 11 条的完整归因

**5 条全部完成（returncode 0，总 wall 315 s），全部预算耗尽、0 个候选。
至此正式 T=2 批次的 11 条 budget-exhaustion failure 全部在 cadence 25 下观测过：
其中 4 条转化成通过完整最终认证的候选，7 条没有。**

对象是第一批 dense 没有覆盖的 5 条，全部来自分层③（20-update 重放里从未跌破阈值）：
q50 的 seed 10005 / 10006 / 10007 / 10010 / 10016。

## 协议与第一批完全一致

协议常数不是重新写的：每条记录从 `experiments/two_stage_formal_T2_20260915/manifest.json` 的原始
记录出发，只施加第一批 dense 用过的三处改动，并 assert 其余字段逐个相同：

* `protocol.verifier_timeout`：`100` → `{"A": 100, "B": 100, "C": 25}`（只加密 C）；
* `protocol.weights_every`：新增 `25`；
* `output_dir` 重定向到本实验的 `runs/`。

6 个协议模块（`agents/ppo_curriculum.py`、`envs/curriculum_env.py`、`utils/dp_br_verifier.py`、
`utils/theory_multistage.py`、`run/run_final_dp_br.py`、`run/run_final_dp_br_round3_dense.py`）
的 sha256 与第一批 manifest 记录的值**逐一核对一致**，所以两批是同一份代码。
`build_manifest.py` 在生成 manifest 时执行这两项检查，不通过就不写文件。

轨迹仍然是逐位精确的：5 条 run 的 50 个共享 update（C100…C1000）上，
`dreach` 与 `max_std_norm` 与 20-update stability 重放的最大绝对差 = **0.000e+00**。

## 结果：5 条全部维持 budget exhaustion

每条 37 次 C verifier call（C100 warmup + 之后每 25），合并 20-update 重放后 77 个观测点，
**eligible 点 0 个**：

| run | 原 C-call 最小 | 20u 重放最小 | 20u+25u 合并最小 | 高出阈值 | 合并网格 eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| q50 s10005 | 0.011858 | 0.011858 | 0.011858 | +18.6% | 0 / 77 |
| q50 s10006 | 0.016077 | 0.013793 | 0.013793 | +37.9% | 0 / 77 |
| q50 s10007 | 0.013612 | 0.013219 | 0.012669 | +26.7% | 0 / 77 |
| q50 s10010 | 0.014620 | 0.011712 | 0.011677 | +16.8% | 0 / 77 |
| q50 s10016 | 0.012105 | 0.012105 | 0.011413 | +14.1% | 0 / 77 |

更密的采样确实把观测到的最小值又压低了一点（s10007 −4.2%、s10016 −5.7%、s10010 −0.3%），
但全部仍高出阈值 14% 以上。**这 5 条不是采样问题，离线重放的预测被逐条证实。**

## 11 条的完整归因

| 分层（按 20-update 重放判定） | n | cadence 25 下找到候选 | 通过完整最终认证 |
| --- | ---: | ---: | ---: |
| ① 漏采：C 段确实经过 eligible 策略 | 5 | 4 | **4** |
| ② 擦边：重放最小值高出阈值 0.5% | 1 | 0 | 0 |
| ③ 真失败：重放最小值高出 14–38% | 5 | 0 | 0 |
| 合计 | 11 | 4 | **4** |

回答「原来有多少 failure 实际是 verifier 没有及时看到 candidate」：

* **原则上 5 条**（分层①）。这 5 条的 C 段真的经过了同时满足数值有效、BR、浓度三项的策略，
  只是 cadence-100 的 verifier 没有在那些 update 上调用过。
* **实际按 cadence-25 的日程能救回 4 条**，且 4 条都通过 `final_joint_pass`
  （main / refine / sensitivity / dense concentration 四项全过），不是只过了 development。
  第 5 条 q50 s10020 的 eligible 窗口比 25 个 update 还窄，25-网格同样跨了过去
  （C950 = 0.01571，**C960 = 0.00975**，C975 = 0.01246）。
* **其余 6 条不是采样问题**：1 条擦边（s10012，37 次 call + 合并网格上一个 eligible 点都没有），
  5 条真失败。要救这 6 条只能改训练侧，不能靠加密 verification。
  （H1 已否定网络容量假说，见 `experiments/two_stage_h1_capacity_20260922/`。）

## 这些数字**不能**直接当成 cadence-25 协议的通过率

把 4 条转化并到原表会得到 q50 search success 10/20 → 13/20、q60 19/20 → 20/20，
但这是**在已知会转化的子集上报告通过率**，不是协议的率。原因是：29 条已成功的 run 对
cadence 变化**不是不变量**——25-网格包含全部 100 的倍数，所以搜索成功是单调的，
但 first-eligible 会提前到不同的 θ，而那个 θ 要重新做 final certification
（上一轮唯一的认证失败 q60 s10020 只差 5.5e-5）。要报率必须 40 条全部重跑。

## 文件

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 5 条 run 的完整记录，含 `changes_vs_source`、`companion_batch` 与 6 个协议模块的 sha256 |
| `build_manifest.py` | 从正式 manifest 派生本批 manifest；内含「只有三处改动」与「协议 sha256 未变」两项 assert |
| `runs/F1_A400_const_first_eligible/<run>/` | 每条 run 的 `config.json`、`train_history.json`、`final_eval.json`、`arrays.npz`、`status.json`、`checkpoint.pt`、`phase_{A,B}_exit_arrays.npz`、`weights/u#####.npz`（每 25 个 update） |
| `C_merged_grid.csv`（385 行） | 5 条 run 的 20+25 合并网格：每点 dReach/ΔW、浓度、valid/br_pass/conc_pass/eligible、来源 |
| `dense_run_summary.csv`（5 行） | 每条 run 的停止点、最终认证、合并网格统计、与原 run 的对照 |
| `all11_summary.csv`（11 行） | **完整归因表**：11 条 no-candidate run 的分层、原 cadence-100 结果、cadence-25 结果、是否转化，两批 dense 合并 |
| `analyze_rest.py` | 本批分析 + 11 条合并表 |
| `logs/`、`launch_status.json`、`launch.log` | 启动记录（tmux session `dense_c25_rest`） |

重跑：

```bash
cd . && .venv/bin/python -B experiments/two_stage_dense_c25_rest_20260922/launch.py && OMP_NUM_THREADS=1 .venv/bin/python -B experiments/two_stage_dense_c25_rest_20260922/analyze_rest.py
```

（`launch.py` 在 `launch_status.json` 已存在时会拒绝启动，重跑前需先移开该文件与 `logs/`、`runs/`。）
